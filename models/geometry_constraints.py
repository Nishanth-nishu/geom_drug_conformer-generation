"""
geometry_constraints.py — Molecular Geometry Constraint Losses

Provides GeometryConstraints, a differentiable module that computes
physics-based penalties on predicted 3D coordinates:

  1. Bond length loss   — predicted bonds vs. ideal RDKit/UFF bond lengths
  2. Bond angle loss    — predicted angles vs. ideal valence angles
  3. Torsion loss       — soft periodic penalty encouraging gauche/anti preferences
  4. Clash (repulsion)  — soft Lennard-Jones repulsion for non-bonded atom pairs

These losses are added to the diffusion MSE during training to encourage
the model to generate chemically valid geometries. They are weighted by
geometry_weight (which ramps up over training in mol_train_expH.py).

Usage:
    gc = GeometryConstraints(bond_weight=20.0, angle_weight=3.0,
                              torsion_weight=1.0, repulsion_weight=5.0)
    total, breakdown = gc.compute_total_loss(
        pos, atom_types, edge_index, bond_types, batch_idx,
        include_angles=True, include_torsions=False
    )
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# =============================================================================
# Reference bond lengths (Angstrom) — keyed by (atomic_num_i, atomic_num_j, bond_order)
# Values from CRC Handbook / average UFF parameters
# =============================================================================

# atomic number → covalent radius (Å)
_COVALENT_RADII = {
    1:  0.31,   # H
    5:  0.84,   # B
    6:  0.76,   # C  (sp3)
    7:  0.71,   # N
    8:  0.66,   # O
    9:  0.57,   # F
    14: 1.11,   # Si
    15: 1.07,   # P
    16: 1.05,   # S
    17: 1.02,   # Cl
    35: 1.20,   # Br
    53: 1.39,   # I
}
_DEFAULT_RADIUS = 0.77  # fallback (C-like)

# Bond order corrections (multiply single-bond sum of radii by this factor)
_BOND_ORDER_FACTOR = {
    0: 1.0,    # unknown / no bond → treat like single
    1: 1.0,    # single
    2: 0.87,   # double
    3: 0.78,   # triple
    4: 0.95,   # aromatic
    5: 1.0,    # other
}

# Bond order → ideal equilibrium angle for X-A-X (degrees)
# These are rough targets; actual geometry depends on hybridisation
_IDEAL_ANGLE_BY_HYBRIDIZATION = {
    6: {1: 109.5, 2: 120.0, 3: 180.0, 4: 120.0},  # C
    7: {1: 107.0, 2: 120.0, 3: 180.0, 4: 120.0},  # N
    8: {1: 104.5, 2: 120.0, 4: 120.0},              # O
    15:{1: 107.0, 2: 120.0},                         # P
    16:{1: 103.0, 2: 119.0},                         # S
}
_DEFAULT_IDEAL_ANGLE = 109.5


def _ideal_bond_length(zi: int, zj: int, bond_order: int) -> float:
    """Estimate ideal bond length from covalent radii + bond order factor."""
    ri = _COVALENT_RADII.get(int(zi), _DEFAULT_RADIUS)
    rj = _COVALENT_RADII.get(int(zj), _DEFAULT_RADIUS)
    fac = _BOND_ORDER_FACTOR.get(int(bond_order), 1.0)
    return (ri + rj) * fac


def _ideal_angle(atom_z: int, neighbor_bond_order: int) -> float:
    """Estimate ideal X-A-X angle (degrees) for central atom with atomic number atom_z."""
    angles = _IDEAL_ANGLE_BY_HYBRIDIZATION.get(int(atom_z), {})
    return angles.get(int(neighbor_bond_order), _DEFAULT_IDEAL_ANGLE)


# =============================================================================
# Non-bonded VDW radii for clash detection
# =============================================================================

_VDW_RADII = {
    1:  1.20,   # H
    6:  1.70,   # C
    7:  1.55,   # N
    8:  1.52,   # O
    9:  1.47,   # F
    15: 1.80,   # P
    16: 1.80,   # S
    17: 1.75,   # Cl
    35: 1.85,   # Br
    53: 1.98,   # I
}
_DEFAULT_VDW = 1.70


# =============================================================================
# GEOMETRY CONSTRAINTS MODULE
# =============================================================================

class GeometryConstraints(nn.Module):
    """
    Differentiable geometry constraint losses for molecular conformers.

    All losses are soft (L1 or L2 on deviations from ideals), so gradients
    flow back through predicted coordinates during training.

    Methods
    -------
    compute_total_loss(pos, atom_types, edge_index, bond_types, batch_idx,
                       include_angles, include_torsions)
        Main entry point. Returns (total_loss: Tensor, breakdown: dict).
    """

    def __init__(self,
                 bond_weight: float = 20.0,
                 angle_weight: float = 3.0,
                 torsion_weight: float = 1.0,
                 repulsion_weight: float = 5.0,
                 bond_threshold: float = 0.20,    # Å deviation before loss kicks in
                 angle_threshold: float = 5.0,    # degrees
                 clash_scale: float = 0.5,         # fraction of VDW sum = clash distance
                 ):
        super().__init__()
        self.bond_weight      = bond_weight
        self.angle_weight     = angle_weight
        self.torsion_weight   = torsion_weight
        self.repulsion_weight = repulsion_weight
        self.bond_threshold   = bond_threshold
        self.angle_threshold  = math.radians(angle_threshold)
        self.clash_scale      = clash_scale

    # ── Bond length loss ───────────────────────────────────────────────────────

    def bond_length_loss(self,
                          pos: torch.Tensor,         # (N, 3)
                          atom_types: torch.Tensor,  # (N,)
                          edge_index: torch.Tensor,  # (2, E)
                          bond_types: torch.Tensor,  # (E,)
                          ) -> torch.Tensor:
        """
        Smooth-L1 loss on bond lengths vs. ideal values.

        For each directed edge (i→j): ideal_len = f(Z_i, Z_j, bond_order)
        Loss = SmoothL1(||pos_i - pos_j|| - ideal)
        """
        if edge_index.size(1) == 0:
            return pos.new_zeros(())

        row, col = edge_index          # row=source, col=destination
        diff = pos[row] - pos[col]     # (E, 3)
        dist = diff.norm(dim=-1)       # (E,)

        # Build ideal distances per edge (CPU lookup, vectorised over batch)
        zi = atom_types[row].cpu().tolist()
        zj = atom_types[col].cpu().tolist()
        bo = bond_types.cpu().tolist()

        ideal = torch.tensor(
            [_ideal_bond_length(z_i, z_j, b) for z_i, z_j, b in zip(zi, zj, bo)],
            device=pos.device, dtype=pos.dtype,
        )

        # Only penalise bonds that deviate > threshold (deadband)
        err = (dist - ideal).abs() - self.bond_threshold
        err = err.clamp(min=0.0)
        return F.smooth_l1_loss(err, torch.zeros_like(err), beta=0.1)

    # ── Bond angle loss ────────────────────────────────────────────────────────

    def bond_angle_loss(self,
                         pos: torch.Tensor,         # (N, 3)
                         atom_types: torch.Tensor,  # (N,)
                         edge_index: torch.Tensor,  # (2, E)
                         bond_types: torch.Tensor,  # (E,)
                         ) -> torch.Tensor:
        """
        L2 loss on X-A-Y angles vs. ideal valence angles.

        For each central atom A with ≥2 bonds, enumerate all pairs of
        neighbours (X, Y) and penalise |angle(XAY) - ideal|.

        Angle computed from:
          cos θ = (u·v) / (|u||v|),  u = pos_X - pos_A,  v = pos_Y - pos_A
        """
        N = pos.size(0)
        if edge_index.size(1) == 0:
            return pos.new_zeros(())

        row, col = edge_index   # edge i→j: row=i (source), col=j (dest)
        # Build per-atom neighbour list
        # neighbours of atom a = all col where row==a
        # For angles, we iterate over atom a as central, pairs of its neighbours

        angle_losses = []

        # Compute adjacency dict (sparse, CPU)
        adj: Dict[int, list] = {}
        for e_idx in range(row.size(0)):
            a = int(row[e_idx].item())
            b = int(col[e_idx].item())
            bo = int(bond_types[e_idx].item())
            if a not in adj:
                adj[a] = []
            adj[a].append((b, bo))

        for central_atom, neighbours in adj.items():
            if len(neighbours) < 2:
                continue

            z_a = int(atom_types[central_atom].item())
            pos_a = pos[central_atom]  # (3,)

            # Enumerate all pairs of neighbours
            for i_n in range(len(neighbours)):
                for j_n in range(i_n + 1, len(neighbours)):
                    nb_i, bo_i = neighbours[i_n]
                    nb_j, bo_j = neighbours[j_n]

                    u = pos[nb_i] - pos_a
                    v = pos[nb_j] - pos_a

                    u_norm = u.norm().clamp(min=1e-6)
                    v_norm = v.norm().clamp(min=1e-6)

                    cos_theta = (u * v).sum() / (u_norm * v_norm)
                    cos_theta = cos_theta.clamp(-1 + 1e-6, 1 - 1e-6)
                    theta = torch.acos(cos_theta)  # radians

                    # Use the bond order of the higher-priority bond for ideal angle
                    ideal_deg = _ideal_angle(z_a, max(bo_i, bo_j))
                    ideal_rad = math.radians(ideal_deg)

                    err = (theta - ideal_rad).abs() - self.angle_threshold
                    err = err.clamp(min=0.0)
                    angle_losses.append(err)

        if not angle_losses:
            return pos.new_zeros(())

        return torch.stack(angle_losses).mean()

    # ── Torsion (dihedral) loss ────────────────────────────────────────────────

    def torsion_loss(self,
                      pos: torch.Tensor,         # (N, 3)
                      edge_index: torch.Tensor,  # (2, E)
                      ) -> torch.Tensor:
        """
        Soft periodic torsion potential: V = k(1 + cos(3φ)), minimised at φ=60°,180°,300°.
        This encourages gauche/anti dihedral preferences (staggered conformations).

        Only applied to single-bond dihedrals (inferred from topology by finding
        two atoms that share a common bond edge).
        """
        if edge_index.size(1) < 2:
            return pos.new_zeros(())

        row, col = edge_index

        # Find all A-B-C-D quartets: A-B bond + B-C bond + C-D bond (3 consecutive bonds)
        # Use adjacency to find paths of length 3
        adj: Dict[int, list] = {}
        for e in range(row.size(0)):
            a, b = int(row[e].item()), int(col[e].item())
            adj.setdefault(a, []).append(b)

        torsion_vals = []
        visited_dihedrals = set()

        for b, b_neighbours in adj.items():
            for a in b_neighbours:
                for c in b_neighbours:
                    if c == a:
                        continue
                    if c not in adj:
                        continue
                    for d in adj[c]:
                        if d == b or d == a:
                            continue
                        # dihedral A-B-C-D
                        key = tuple(sorted([a, d]) + [b, c])
                        if key in visited_dihedrals:
                            continue
                        visited_dihedrals.add(key)

                        pa, pb, pc, pd = pos[a], pos[b], pos[c], pos[d]
                        b1 = pb - pa
                        b2 = pc - pb
                        b3 = pd - pc

                        # Praxitelou dihedral formula
                        n1 = torch.cross(b1, b2, dim=0)
                        n2 = torch.cross(b2, b3, dim=0)
                        b2_norm = b2 / b2.norm().clamp(min=1e-6)
                        cos_phi = (n1 * n2).sum() / (n1.norm().clamp(min=1e-6) * n2.norm().clamp(min=1e-6))
                        cos_phi = cos_phi.clamp(-1 + 1e-6, 1 - 1e-6)
                        phi = torch.atan2(
                            (torch.cross(n1, n2, dim=0) * b2_norm).sum(),
                            (n1 * n2).sum()
                        )
                        # Soft 3-fold periodic potential (staggered preference)
                        v = (1.0 + torch.cos(3.0 * phi))
                        torsion_vals.append(v)

        if not torsion_vals:
            return pos.new_zeros(())

        return torch.stack(torsion_vals).mean()

    # ── Steric repulsion (clash) loss ──────────────────────────────────────────

    def repulsion_loss(self,
                        pos: torch.Tensor,         # (N, 3)
                        atom_types: torch.Tensor,  # (N,)
                        batch_idx: torch.Tensor,   # (N,)
                        edge_index: torch.Tensor,  # (2, E) — bonded pairs (excluded from repulsion)
                        ) -> torch.Tensor:
        """
        Soft repulsion loss for non-bonded atoms that are too close.

        For each pair (i, j) not in the bond graph and within the same molecule:
          If dist < clash_scale * (VDW_i + VDW_j):
            loss += (clash_dist - dist)^2

        Efficiently computed per-molecule using cdist.
        Excluded pairs: bonded pairs from edge_index.
        """
        B = int(batch_idx.max().item()) + 1
        total_loss = pos.new_zeros(())

        # Bonded pair set for exclusion
        bonded = set()
        for e in range(edge_index.size(1)):
            i, j = int(edge_index[0, e].item()), int(edge_index[1, e].item())
            bonded.add((min(i, j), max(i, j)))

        for b in range(B):
            mask = (batch_idx == b)
            local_idx = mask.nonzero(as_tuple=True)[0]
            n = local_idx.size(0)
            if n < 2:
                continue

            p = pos[local_idx]                              # (n, 3)
            z = atom_types[local_idx].cpu().tolist()
            dist_mat = torch.cdist(p, p)                    # (n, n)

            for i in range(n):
                for j in range(i + 1, n):
                    gi = int(local_idx[i].item())
                    gj = int(local_idx[j].item())
                    if (min(gi, gj), max(gi, gj)) in bonded:
                        continue

                    vdw_sum = (_VDW_RADII.get(int(z[i]), _DEFAULT_VDW) +
                               _VDW_RADII.get(int(z[j]), _DEFAULT_VDW))
                    clash_dist = self.clash_scale * vdw_sum
                    d = dist_mat[i, j]
                    if d < clash_dist:
                        total_loss = total_loss + (clash_dist - d) ** 2

        # Normalise by number of atoms to keep scale consistent
        n_atoms = pos.size(0)
        return total_loss / max(n_atoms, 1)

    # ── Combined loss ──────────────────────────────────────────────────────────

    def compute_total_loss(self,
                            pos: torch.Tensor,
                            atom_types: torch.Tensor,
                            edge_index: torch.Tensor,
                            bond_types: torch.Tensor,
                            batch_idx: torch.Tensor,
                            include_angles: bool = True,
                            include_torsions: bool = False,
                            ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted sum of all geometry constraint losses.

        Returns
        -------
        total : Tensor — scalar total loss (weighted sum)
        breakdown : dict with keys 'bond', 'angle', 'torsion', 'repulsion'
        """
        # Bond length loss (always included)
        # Use only bond edges (exclude self-loops)
        valid_mask = (edge_index[0] != edge_index[1])
        ei_valid = edge_index[:, valid_mask]
        bt_valid = bond_types[valid_mask]

        l_bond = self.bond_length_loss(pos, atom_types, ei_valid, bt_valid)

        # Angle loss
        if include_angles:
            l_angle = self.bond_angle_loss(pos, atom_types, ei_valid, bt_valid)
        else:
            l_angle = pos.new_zeros(())

        # Torsion loss
        if include_torsions:
            l_torsion = self.torsion_loss(pos, ei_valid)
        else:
            l_torsion = pos.new_zeros(())

        # Repulsion loss (non-bonded clashes)
        if self.repulsion_weight > 0:
            l_repulsion = self.repulsion_loss(pos, atom_types, batch_idx, ei_valid)
        else:
            l_repulsion = pos.new_zeros(())

        total = (self.bond_weight      * l_bond +
                 self.angle_weight     * l_angle +
                 self.torsion_weight   * l_torsion +
                 self.repulsion_weight * l_repulsion)

        breakdown = {
            'bond':      l_bond.detach(),
            'angle':     l_angle.detach(),
            'torsion':   l_torsion.detach(),
            'repulsion': l_repulsion.detach(),
        }

        return total, breakdown


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    import torch

    print("Testing GeometryConstraints ...")
    gc = GeometryConstraints(bond_weight=20.0, angle_weight=3.0,
                              torsion_weight=1.0, repulsion_weight=5.0)

    # Ethanol-like: C-C-O, 3 atoms
    atom_types = torch.tensor([6, 6, 8])         # C, C, O
    pos = torch.tensor([[0.0, 0.0, 0.0],
                        [1.5, 0.0, 0.0],
                        [2.8, 0.9, 0.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    bond_types = torch.tensor([1, 1, 1, 1], dtype=torch.long)  # single bonds
    batch_idx = torch.zeros(3, dtype=torch.long)

    total, breakdown = gc.compute_total_loss(
        pos, atom_types, edge_index, bond_types, batch_idx,
        include_angles=True, include_torsions=False,
    )
    print(f"Total loss: {total.item():.4f}")
    for k, v in breakdown.items():
        print(f"  {k}: {v.item():.4f}")

    # Test with obviously wrong geometry (very short bonds = clashes)
    pos_bad = torch.tensor([[0.0, 0.0, 0.0],
                             [0.3, 0.0, 0.0],   # too short (0.3 vs 1.54 ideal)
                             [0.6, 0.0, 0.0]], dtype=torch.float32)
    total_bad, _ = gc.compute_total_loss(
        pos_bad, atom_types, edge_index, bond_types, batch_idx,
        include_angles=True, include_torsions=False,
    )
    print(f"Bad geometry total loss: {total_bad.item():.4f}  (should be >> good geometry)")
    assert total_bad.item() > total.item(), "Bad geometry should have higher loss"
    print("ALL TESTS PASSED")
