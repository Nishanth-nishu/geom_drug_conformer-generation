"""
energy_surrogate.py — GFN2-xTB Energy Surrogate Network (v2)

An EGNN-based scalar energy predictor trained to emulate GFN2-xTB energies.

Architecture:
  - Shared EquivariantLayer backbone (same as ConformerDenoiser)
  - No time conditioning (energy is time-independent)
  - Global mean-pool over atoms → per-molecule scalar energy
  - Trained with MSE loss on normalized GFN2-xTB energies

v2 Additions:
  - get_loss_with_forces(): joint energy + chord-approx force MSE (W4 fix)
  - augment_ood(): Gaussian noise + atom-pair repulsion augmentation (W3 fix)
  - get_loss_with_ood(): Boltzmann MSE + OOD hinge contrastive loss (W3 fix)

Purpose:
  At inference time during energy-guided DDIM sampling, we need ∇_x E(x) —
  the gradient of energy with respect to 3D coordinates. Running the actual
  GFN2-xTB calculation inside the DDIM loop is too slow (50 steps × N mols).
  The surrogate gives fast, differentiable gradients.

Training (v2 two-phase):
  Phase 1: Pre-train surrogate on GEOM-Drugs energies + OOD augmentation.
           Optionally add chord-approximated force supervision.
  Phase 2: Surrogate frozen; diffusion trained with Boltzmann-weighted MSE.

Reference:
  DimeNet++ (Gasteiger et al., NeurIPS 2020): energy prediction architecture.
  SchNet (Schütt et al., NeurIPS 2017): invariant energy prediction from distances.
  Force-matching: Chmiela et al. (Sci. Adv. 2017) — gradient of energy = negative force.
  OOD robustness: concept from contrastive energy training in ML-FF literature.

Usage in energy-guided DDIM:
  surrogate = EnergySurrogate.load('checkpoints/energy_surrogate.pt')
  # Inside sampling loop:
  with torch.enable_grad():
      x_0.requires_grad_(True)
      E_pred = surrogate(x_0, atom_types, edge_index, bond_types, batch_idx)
      grad_E = torch.autograd.grad(E_pred.sum(), x_0)[0]   # ∇_x E
  x_0_guided = x_0 - gamma * grad_E
"""

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers (duplicated from conformer_diffusion to keep module standalone)
# ──────────────────────────────────────────────────────────────────────────────

def _rbf_features(dist: torch.Tensor,
                  num_rbf: int = 20,
                  d_min: float = 0.5,
                  d_max: float = 6.0) -> torch.Tensor:
    centers = torch.linspace(d_min, d_max, num_rbf, device=dist.device)
    gamma   = 2.0 / (d_max - d_min) * (num_rbf - 1)
    return torch.exp(-gamma * (dist - centers.unsqueeze(0)) ** 2)


def _remove_com(x: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
    B = int(batch_idx.max().item()) + 1
    mol_sums   = torch.zeros(B, 3, device=x.device, dtype=x.dtype)
    mol_counts = torch.zeros(B,    device=x.device, dtype=x.dtype)
    mol_counts.scatter_add_(0, batch_idx, torch.ones(x.size(0), device=x.device))
    mol_sums.scatter_add_(0, batch_idx.unsqueeze(-1).expand(-1, 3), x)
    mol_means = mol_sums / mol_counts.unsqueeze(1).clamp(min=1)
    return x - mol_means[batch_idx]


# ──────────────────────────────────────────────────────────────────────────────
# Equivariant layer for the surrogate (invariant output — no coord update needed)
# ──────────────────────────────────────────────────────────────────────────────

class EnergyEquivariantLayer(nn.Module):
    """
    EGNN-style message-passing layer for energy prediction.
    Unlike EquivariantLayer in conformer_diffusion.py, this does NOT update
    coordinates — it only updates node features h using geometry information.
    Output is E(3)-invariant (depends only on distances, not directions).
    """

    def __init__(self, hidden_dim: int, num_rbf: int = 20):
        super().__init__()
        self.num_rbf = num_rbf

        # Edge MLP: node features + RBF distance
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                h: torch.Tensor,           # (N, hidden_dim)
                x: torch.Tensor,           # (N, 3) — only used for distances
                edge_index: torch.Tensor,  # (2, E)
                bond_embed: torch.Tensor,  # (E, hidden_dim)
                ) -> torch.Tensor:         # (N, hidden_dim)
        row, col = edge_index

        diff = x[row] - x[col]
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)
        rbf  = _rbf_features(dist, num_rbf=self.num_rbf)

        edge_input = torch.cat([h[row], h[col], rbf], dim=-1)
        m_ij = self.edge_mlp(edge_input) + bond_embed

        m_agg = torch.zeros_like(h)
        m_agg.scatter_add_(0, col.unsqueeze(-1).expand(-1, h.size(-1)), m_ij)

        h_new = self.node_mlp(torch.cat([h, m_agg], dim=-1))
        return self.layer_norm(h + h_new)


# ──────────────────────────────────────────────────────────────────────────────
# Energy Surrogate Network
# ──────────────────────────────────────────────────────────────────────────────

class EnergySurrogate(nn.Module):
    """
    EGNN-based GFN2-xTB energy surrogate (v2).

    Predicts normalized per-atom GFN2-xTB energy from 3D coordinates.
    The output is a per-molecule scalar (sum over per-atom contributions).

    Architecture (SchNet-inspired atomistic decomposition):
      E_total = Σ_i e_i    where e_i = MLP(h_i)
    This decomposition ensures size-extensivity (energy scales linearly with
    molecule size) — a key property of quantum chemical energies.

    v2 enhancements:
      - get_loss_with_forces(): chord-approx force supervision for better gradients
      - augment_ood():          Gaussian noise + clash augmentation for robustness
      - get_loss_with_ood():    OOD hinge contrastive loss
    """

    def __init__(
        self,
        hidden_dim: int     = 256,
        num_layers: int     = 4,
        num_atom_types: int = 119,   # full periodic table for drug atoms
        num_bond_types: int = 5,
        num_rbf: int        = 20,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Atom type embedding (extended beyond QM9: includes S, P, Cl, Br, etc.)
        self.atom_embed = nn.Embedding(num_atom_types, hidden_dim)

        # Bond type embedding
        self.bond_embed = nn.Embedding(num_bond_types + 1, hidden_dim)

        # Message passing layers (invariant — no coord update)
        self.layers = nn.ModuleList([
            EnergyEquivariantLayer(hidden_dim, num_rbf=num_rbf)
            for _ in range(num_layers)
        ])

        # Per-atom energy contribution head (atomistic decomposition)
        self.atom_energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),   # scalar per atom
        )

    def forward(
        self,
        x: torch.Tensor,           # (N, 3) 3D coordinates (Å)
        atom_types: torch.Tensor,  # (N,) atomic numbers
        edge_index: torch.Tensor,  # (2, E) bond graph
        bond_types: torch.Tensor,  # (E,) bond orders
        batch_idx: torch.Tensor,   # (N,) molecule assignment
    ) -> torch.Tensor:             # (B,) per-molecule energy (normalized)
        # Atom features
        h = self.atom_embed(atom_types.clamp(0, self.atom_embed.num_embeddings - 1))

        # Bond features
        bond_feat = self.bond_embed(bond_types.clamp(0, self.bond_embed.num_embeddings - 1))

        # Message passing (invariant)
        for layer in self.layers:
            h = layer(h, x, edge_index, bond_feat)

        # Per-atom energy contributions
        e_atom = self.atom_energy_head(h).squeeze(-1)   # (N,)

        # Sum to per-molecule energy (size-extensive)
        B = int(batch_idx.max().item()) + 1
        e_mol = torch.zeros(B, device=x.device, dtype=x.dtype)
        e_mol.scatter_add_(0, batch_idx, e_atom)

        return e_mol   # (B,) normalized energy

    def get_loss(
        self,
        x: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor,
        bond_types: torch.Tensor,
        batch_idx: torch.Tensor,
        energy_targets: torch.Tensor,   # (B,) normalized GFN2-xTB energies
        boltzmann_weights: Optional[torch.Tensor] = None,  # (B,) sample weights
    ) -> Dict[str, torch.Tensor]:
        """
        Boltzmann-weighted MSE loss for energy prediction.

        High Boltzmann-weight conformers (low energy, thermodynamically dominant)
        contribute proportionally more to the gradient, so the surrogate is most
        accurate where it matters most for physical guidance.
        """
        e_pred = self(x, atom_types, edge_index, bond_types, batch_idx)   # (B,)
        sq_err = (e_pred - energy_targets) ** 2   # (B,)

        if boltzmann_weights is not None:
            # Weight loss by Boltzmann probability of this conformer
            # High-weight conformers (low energy) should be predicted more accurately
            w = boltzmann_weights / boltzmann_weights.sum().clamp(min=1e-8)
            loss = (w * sq_err).sum()
        else:
            loss = sq_err.mean()

        mae = (e_pred - energy_targets).abs().mean()

        return {
            'total': loss,
            'mse':   loss.detach(),
            'mae':   mae.detach(),
        }

    def get_loss_with_forces(
        self,
        x: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor,
        bond_types: torch.Tensor,
        batch_idx: torch.Tensor,
        energy_targets: torch.Tensor,
        force_targets: Optional[torch.Tensor],  # (N, 3) chord-approx forces, or None
        boltzmann_weights: Optional[torch.Tensor] = None,
        force_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Joint energy + force MSE (v2 — W4: gradient direction supervision).

        F_pred = -grad_x E_surrogate(x)  via autograd (create_graph=True).
        F_target = chord-approximated between conformer pairs (free, no xTB calls):
            F_approx ~= -dE_ij / |dx_ij|^2 * dx_ij

        Force supervision ensures grad_x E_surrogate points in the physically
        correct direction — which is what we actually use during DDIM guidance.
        Even noisy chord-approximated force targets improve gradient direction.

        Reference: Chmiela et al., Sci. Adv. 2017 — force-field training.
        """
        x_grad = x.detach().requires_grad_(True)
        e_pred = self(x_grad, atom_types, edge_index, bond_types, batch_idx)  # (B,)

        # Boltzmann-weighted energy loss
        sq_err_e = (e_pred - energy_targets) ** 2
        if boltzmann_weights is not None:
            w = boltzmann_weights / boltzmann_weights.sum().clamp(min=1e-8)
            loss_e = (w * sq_err_e).sum()
        else:
            loss_e = sq_err_e.mean()

        # Force loss via autograd: F_pred = -grad_x E
        loss_f = torch.tensor(0.0, device=x.device)
        if force_targets is not None and force_weight > 0:
            f_pred = -torch.autograd.grad(
                e_pred.sum(), x_grad,
                create_graph=True, retain_graph=True
            )[0]   # (N, 3)
            B = int(batch_idx.max().item()) + 1
            f_sq = ((f_pred - force_targets) ** 2).sum(-1)   # (N,)
            f_mol = torch.zeros(B, device=x.device)
            n_mol = torch.zeros(B, device=x.device)
            f_mol.scatter_add_(0, batch_idx, f_sq)
            n_mol.scatter_add_(0, batch_idx, torch.ones(x.size(0), device=x.device))
            loss_f = (f_mol / n_mol.clamp(min=1)).mean()

        total = loss_e + force_weight * loss_f
        return {
            'total':  total,
            'energy': loss_e.detach(),
            'force':  loss_f.detach(),
            'mae':    (e_pred - energy_targets).abs().mean().detach(),
        }

    @staticmethod
    def augment_ood(
        x: torch.Tensor,
        batch_idx: torch.Tensor,
        e_max: float,
        noise_sigma: float = 0.5,
        ood_energy_offset: float = 3.0,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        OOD augmentation for surrogate robustness (v2 — W3 fix).

        During DDIM sampling the surrogate sees x_t = sqrt(a_t)*x_0 + sqrt(1-a_t)*eps
        that are far outside the training distribution of real GEOM conformers.
        Without OOD training, the surrogate assigns arbitrary energies/gradients.

        Two augmentation strategies:
        1. Gaussian noise (sigma=0.5 A): teaches distorted coords -> high energy.
        2. Atom-pair push (steric clash): teaches overlapping atoms -> high energy.

        Returns ((x_noise, x_repulse), (e_label_noise, e_label_repulse)).
        """
        device = x.device
        B = int(batch_idx.max().item()) + 1

        # Strategy 1: Gaussian noise
        x_noise = x.detach() + torch.randn_like(x) * noise_sigma
        e_noise  = torch.full((B,), e_max + ood_energy_offset, device=device)

        # Strategy 2: Random atom-pair push (steric clash)
        x_repulse = x.detach().clone()
        n_total   = x.size(0)
        n_pairs   = max(1, n_total // 4)
        rand_i = torch.randint(0, n_total, (n_pairs,), device=device)
        rand_j = torch.randint(0, n_total, (n_pairs,), device=device)
        valid  = (batch_idx[rand_i] == batch_idx[rand_j]) & (rand_i != rand_j)
        if valid.any():
            vi, vj = rand_i[valid], rand_j[valid]
            direction = x_repulse[vj] - x_repulse[vi]
            dist = direction.norm(dim=-1, keepdim=True).clamp(min=0.3)
            # Push atom vi toward vj by 0.7 A (creates steric clash)
            x_repulse = x_repulse.clone()
            x_repulse[vi] = x_repulse[vi] + 0.7 * direction / dist
        e_repulse = torch.full((B,), e_max + 2 * ood_energy_offset, device=device)

        return (x_noise, x_repulse), (e_noise, e_repulse)

    def get_loss_with_ood(
        self,
        x: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor,
        bond_types: torch.Tensor,
        batch_idx: torch.Tensor,
        energy_targets: torch.Tensor,
        boltzmann_weights: Optional[torch.Tensor] = None,
        ood_weight: float = 0.2,
        noise_sigma: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Boltzmann energy MSE + OOD hinge contrastive loss (v2 — W3 fix).

        L = L_energy (real conformers, Boltzmann-weighted)
          + lambda_ood * L_ood (noisy/clashing coords must have high energy)

        L_ood uses hinge: max(0, E_target_ood - E_pred_ood)^2
        Teaches: surrogate must score distorted coords >= E_max + offset.
        This makes surrogate gradients meaningful even at high-noise DDIM steps.
        """
        # Real conformer energy loss
        e_real = self(x, atom_types, edge_index, bond_types, batch_idx)
        sq_err = (e_real - energy_targets) ** 2
        if boltzmann_weights is not None:
            w = boltzmann_weights / boltzmann_weights.sum().clamp(min=1e-8)
            loss_real = (w * sq_err).sum()
        else:
            loss_real = sq_err.mean()

        # OOD contrastive loss
        loss_ood = torch.tensor(0.0, device=x.device)
        if ood_weight > 0:
            e_max = energy_targets.max().item()
            (x_n, x_r), (e_ln, e_lr) = EnergySurrogate.augment_ood(
                x.detach(), batch_idx, e_max, noise_sigma=noise_sigma
            )
            e_ood_n = self(x_n, atom_types, edge_index, bond_types, batch_idx)
            e_ood_r = self(x_r, atom_types, edge_index, bond_types, batch_idx)
            # Hinge: OOD energy must be >= label (bad structures -> high score)
            loss_ood = (
                F.relu(e_ln - e_ood_n) ** 2 +
                F.relu(e_lr - e_ood_r) ** 2
            ).mean()

        total = loss_real + ood_weight * loss_ood
        return {
            'total':  total,
            'energy': loss_real.detach(),
            'ood':    loss_ood.detach(),
            'mae':    (e_real - energy_targets).abs().mean().detach(),
        }

    def predict_energy_kcal(
        self,
        x: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor,
        bond_types: torch.Tensor,
        batch_idx: torch.Tensor,
        energy_mean: float,
        energy_std: float,
    ) -> torch.Tensor:
        """
        Predict GFN2-xTB energy in kcal/mol (denormalized).
        Used for energy ranking at inference time.
        """
        e_norm = self(x, atom_types, edge_index, bond_types, batch_idx)
        # Denormalize: e_kcal_per_atom = e_norm * std + mean
        # Then sum over atoms per molecule (count atoms per molecule)
        B = int(batch_idx.max().item()) + 1
        n_atoms = torch.zeros(B, device=x.device, dtype=x.dtype)
        n_atoms.scatter_add_(0, batch_idx, torch.ones(x.size(0), device=x.device))
        e_kcal = (e_norm * energy_std + energy_mean) * n_atoms
        return e_kcal

    def gradient(
        self,
        x: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor,
        bond_types: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute grad_x E_surrogate — the energy gradient w.r.t. atomic coordinates.
        Used for energy-guided DDIM sampling (smooth power-law schedule in v2).

        Returns: (N, 3) gradient tensor (dE/dx_i for each atom i).
        """
        x_with_grad = x.detach().requires_grad_(True)
        e_mol = self(x_with_grad, atom_types, edge_index, bond_types, batch_idx)
        e_mol.sum().backward()
        grad = x_with_grad.grad.detach()
        return grad   # (N, 3)

    @classmethod
    def load(cls, checkpoint_path: str, device=None) -> 'EnergySurrogate':
        """
        Load surrogate from checkpoint safely using weights_only=True.

        The checkpoint is a state_dict only (no arbitrary Python objects).
        Config is read from JSON sidecar file (_config.json).
        """
        import json, os
        # Load only safe tensor weights
        state = torch.load(checkpoint_path, map_location=device or 'cpu',
                           weights_only=True)

        # Read config from JSON sidecar
        cfg = {'hidden_dim': 256, 'num_layers': 4, 'num_rbf': 20}
        cfg_path = checkpoint_path.replace('.pt', '_config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg_json = json.load(f)
            for k in ('hidden_dim', 'num_layers', 'num_rbf'):
                if k in cfg_json:
                    cfg[k] = int(cfg_json[k])

        model = cls(
            hidden_dim=cfg['hidden_dim'],
            num_layers=cfg['num_layers'],
            num_rbf=cfg['num_rbf'],
        )
        model.load_state_dict(state)
        if device:
            model = model.to(device)
        model.eval()
        return model


# ──────────────────────────────────────────────────────────────────────────────
# Utilities: chord-approximated force computation from conformer pairs
# ──────────────────────────────────────────────────────────────────────────────

def chord_approx_forces(
    x_i: torch.Tensor,    # (N, 3) conformer i coordinates
    x_j: torch.Tensor,    # (N, 3) conformer j coordinates
    e_i: float,           # energy of conformer i (normalized)
    e_j: float,           # energy of conformer j (normalized)
) -> torch.Tensor:
    """
    Chord approximation to the energy gradient from two conformers.

    For two conformers (x_i, E_i) and (x_j, E_j) of the same molecule,
    the force on each atom is approximated as:

        F_approx ~= -(E_j - E_i) / |dx_ij|^2 * dx_ij

    where dx_ij = x_j - x_i is the displacement between the two conformers.
    This is the finite-difference gradient along the chord direction.

    Properties:
    - Free: no additional GFN2-xTB calculations required
    - Noisy: depends on conformer sampling, not actual force field
    - Directionally useful: still improves gradient direction of surrogate

    Returns: (N, 3) force vectors (kcal/mol/A per atom, normalized units)
    """
    dx = x_j - x_i                                # (N, 3) displacement
    dx_sq = (dx ** 2).sum(-1, keepdim=True)       # (N, 1) squared norm
    dx_sq = dx_sq.clamp(min=1e-6)
    dE = e_j - e_i                                 # scalar energy difference

    # F ~= -grad E ~= -dE/|dx|^2 * dx
    forces = -(dE / dx_sq) * dx                   # (N, 3)
    return forces


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Testing EnergySurrogate v2...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EnergySurrogate(hidden_dim=64, num_layers=3, num_rbf=10).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Mini batch: 2 molecules (benzene + ethanol)
    atom_types = torch.tensor([6,6,6,6,6,6,  6,6,8], device=device)
    edge_index  = torch.tensor([
        [0,1,1,2,2,3,3,4,4,5,5,0,  6,7,7,8],
        [1,0,2,1,3,2,4,3,5,4,0,5,  7,6,8,7]
    ], device=device)
    bond_types  = torch.tensor([4,4,4,4,4,4,4,4,4,4,4,4,  1,1,1,1], dtype=torch.long, device=device)
    batch_idx   = torch.tensor([0,0,0,0,0,0,  1,1,1], device=device)
    x           = torch.randn(9, 3, device=device)

    # Forward pass
    e = model(x, atom_types, edge_index, bond_types, batch_idx)
    assert e.shape == (2,), f"Expected (2,), got {e.shape}"
    print(f"  Energy output (normalized): {e.tolist()}")

    # Gradient
    grad = model.gradient(x, atom_types, edge_index, bond_types, batch_idx)
    assert grad.shape == (9, 3), f"Expected (9,3), got {grad.shape}"
    assert not torch.isnan(grad).any(), "NaN in gradient!"
    print(f"  Gradient shape: {grad.shape}, max={grad.abs().max():.4f}")

    # Basic loss
    targets = torch.tensor([-0.5, 0.3], device=device)
    bw = torch.tensor([0.8, 0.2], device=device)
    loss_dict = model.get_loss(x, atom_types, edge_index, bond_types,
                               batch_idx, targets, bw)
    assert not torch.isnan(loss_dict['total']), "NaN loss!"
    print(f"  get_loss: {loss_dict['total'].item():.4f}")

    # Force loss (v2)
    fake_forces = torch.randn(9, 3, device=device) * 0.1
    loss_f = model.get_loss_with_forces(
        x, atom_types, edge_index, bond_types, batch_idx,
        targets, fake_forces, bw, force_weight=0.1
    )
    assert not torch.isnan(loss_f['total']), "NaN force loss!"
    print(f"  get_loss_with_forces: energy={loss_f['energy']:.4f} force={loss_f['force']:.4f}")

    # OOD augmentation (v2)
    (x_n, x_r), (e_n, e_r) = EnergySurrogate.augment_ood(x, batch_idx, e_max=1.0)
    assert x_n.shape == x.shape, f"OOD shape mismatch: {x_n.shape}"
    assert e_n.shape == (2,), f"OOD label shape: {e_n.shape}"
    print(f"  augment_ood: noise={x_n.shape}, repulse={x_r.shape}, labels_n={e_n.tolist()}")

    # OOD loss (v2)
    loss_ood = model.get_loss_with_ood(
        x, atom_types, edge_index, bond_types, batch_idx, targets, bw,
        ood_weight=0.2, noise_sigma=0.5
    )
    assert not torch.isnan(loss_ood['total']), "NaN OOD loss!"
    print(f"  get_loss_with_ood: energy={loss_ood['energy']:.4f} ood={loss_ood['ood']:.4f}")

    # Chord-approx forces
    x2 = x + torch.randn_like(x) * 0.1
    cf = chord_approx_forces(x, x2, e_i=-0.5, e_j=-0.3)
    assert cf.shape == (9, 3), f"Chord force shape: {cf.shape}"
    print(f"  chord_approx_forces: {cf.shape}, max={cf.abs().max():.4f}")

    print("\nAll EnergySurrogate v2 tests PASSED!")
