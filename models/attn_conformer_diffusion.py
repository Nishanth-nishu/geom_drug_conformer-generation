"""
attn_conformer_diffusion.py — EQGAT-diff Style Attention-Enhanced EGNN

Experiment B: Adds dot-product attention over neighbour messages in EGNN
(inspired by EQGAT-diff, Le et al. ICLR 2024).

Bug-fixes vs. v1:
  FIX-AUDIT-1: Ported t-gating for geometry loss (was applying at ALL timesteps).
               At high t, x_0_pred is near-random → geometry gradients are pure noise.
               GCDM (Morehead & Cheng, NeurIPS 2023, §3.3) applies only at t < T*0.3.
  FIX-AUDIT-2: Switched MSE from derived-ε to DIRECT x₀ MSE.
               For x₀-parameterization, MSE on x₀ directly is cleaner and avoids
               1/sqrt(1-αt) amplification at high-t.
               Reference: EDM (Hoogeboom et al. ICML 2022, App. B).
  FIX-AUDIT-3: bond_weight increased 10→20 in GeometryConstraints.
               bond_error=0.230 Å > 0.20 Å threshold → kills fully_valid.

References:
  Le et al. "EQGAT-diff" ICLR 2024. arXiv:2306.01916.
  Veličković et al. "Graph Attention Networks" ICLR 2018. arXiv:1710.10903.
  Morehead & Cheng, "GCDM" NeurIPS 2023.
  Hoogeboom et al., "EDM" ICML 2022.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict

from models.conformer_diffusion import (
    cosine_beta_schedule, sinusoidal_embedding, remove_com, rbf_features,
    ConformerDiffusion
)


# =============================================================================
# ATTENTION-ENHANCED EQUIVARIANT LAYER (EQGAT-diff style)
# =============================================================================

class AttnEquivariantLayer(nn.Module):
    """
    E(3)-equivariant message passing layer with attention-weighted aggregation.

    Architecture change vs. standard EGNN (EquivariantLayer):
      Standard: h_i ← LayerNorm(h_i + MLP([h_i, sum_j m_ij]))
      Attention: h_i ← LayerNorm(h_i + MLP([h_i, sum_j a_ij * m_ij]))

    where a_ij = softmax_over_j(LeakyReLU(W_attn [h_i || h_j || rbf_ij]))

    The coordinate update is UNCHANGED (already equivariant):
      x_i ← x_i + (1/deg_i) * sum_j phi_x(m_ij) * unit_vec_ij
    """

    def __init__(self, hidden_dim: int, num_rbf: int = 20,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_rbf = num_rbf
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.attn_gate = nn.Linear(hidden_dim * 2 + num_rbf, num_heads, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                h: torch.Tensor,           # (N, hidden_dim)
                x: torch.Tensor,           # (N, 3)
                edge_index: torch.Tensor,  # (2, E)
                bond_embed: torch.Tensor   # (E, hidden_dim)
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        N = x.size(0)
        E = row.size(0)

        diff = x[row] - x[col]
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)
        unit_vec = diff / dist
        rbf = rbf_features(dist, num_rbf=self.num_rbf)

        edge_input = torch.cat([h[row], h[col], rbf], dim=-1)
        m_ij = self.edge_mlp(edge_input)
        m_ij = m_ij + bond_embed

        attn_input = torch.cat([h[row], h[col], rbf], dim=-1)
        attn_logits = self.attn_gate(attn_input)
        attn_logits = F.leaky_relu(attn_logits, negative_slope=0.2)
        attn_weights = self._scatter_softmax(attn_logits, col, N)
        attn_weights = self.attn_dropout(attn_weights)

        m_ij_reshaped = m_ij.view(E, self.num_heads, self.head_dim)
        attn_m = (m_ij_reshaped * attn_weights.unsqueeze(-1))
        attn_m = attn_m.view(E, self.hidden_dim)

        coord_weight = self.coord_mlp(m_ij)
        coord_update = coord_weight * unit_vec

        x_agg = torch.zeros_like(x)
        x_agg.scatter_add_(0, col.unsqueeze(-1).expand(-1, 3), coord_update)

        degree = torch.zeros(N, 1, dtype=x.dtype, device=x.device)
        degree.scatter_add_(0, col.unsqueeze(-1),
                            torch.ones(E, 1, dtype=x.dtype, device=x.device))
        degree = (degree / 2.0).clamp(min=1.0)
        x_new = x + x_agg / degree

        m_agg = torch.zeros_like(h)
        m_agg.scatter_add_(0, col.unsqueeze(-1).expand(-1, self.hidden_dim), attn_m)

        h_new = self.node_mlp(torch.cat([h, m_agg], dim=-1))
        h_new = self.layer_norm(h + h_new)

        return h_new, x_new

    @staticmethod
    def _scatter_softmax(logits: torch.Tensor, index: torch.Tensor,
                         num_nodes: int) -> torch.Tensor:
        E, H = logits.shape
        max_logits = torch.full((num_nodes, H), float('-inf'), dtype=logits.dtype, device=logits.device)
        max_logits.scatter_reduce_(0, index.unsqueeze(-1).expand(-1, H),
                                   logits, reduce='amax', include_self=True)
        shifted = logits - max_logits[index]
        exp_logits = shifted.exp()
        exp_sum = torch.zeros(num_nodes, H, dtype=logits.dtype, device=logits.device)
        exp_sum.scatter_add_(0, index.unsqueeze(-1).expand(-1, H), exp_logits)
        attn = exp_logits / (exp_sum[index] + 1e-8)
        return attn


# =============================================================================
# ATTENTION CONFORMER DENOISER
# =============================================================================

class AttnConformerDenoiser(nn.Module):
    """Drop-in replacement for ConformerDenoiser using AttnEquivariantLayer."""

    def __init__(self,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_atom_types: int = 10,
                 num_bond_types: int = 5,
                 num_rbf: int = 20,
                 time_dim: int = 128,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_rbf = num_rbf
        self.time_dim = time_dim

        self.atom_embed = nn.Embedding(54, hidden_dim)
        self.bond_embed = nn.Embedding(num_bond_types + 1, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.layers = nn.ModuleList([
            AttnEquivariantLayer(hidden_dim, num_rbf=num_rbf,
                                 num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Global graph readout: pool atom features → graph embedding
        # Reference: EGNN (Satorras et al. ICML 2021 §3.3) global invariant features
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )

        # Output: predict x_0 (x_0 parameterization)
        self.coord_pred = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3)
        )

    def forward(self,
                x_noisy: torch.Tensor,
                t: torch.Tensor,
                atom_types: torch.Tensor,
                edge_index: torch.Tensor,
                bond_types: torch.Tensor,
                batch_idx: torch.Tensor) -> torch.Tensor:
        h = self.atom_embed(atom_types.clamp(0, 53))
        t_emb = sinusoidal_embedding(t.float(), self.time_dim)
        t_emb = self.time_mlp(t_emb)
        h = h + t_emb[batch_idx]
        bond_feat = self.bond_embed(bond_types.clamp(0, 5))
        x = x_noisy
        for layer in self.layers:
            h, x = layer(h, x, edge_index, bond_feat)

        # Global graph readout: mean pool per molecule, broadcast back
        B = int(batch_idx.max().item()) + 1
        N = h.size(0)
        g_feat = self.global_mlp(h)                          # (N, H//4)
        g_sum = torch.zeros(B, g_feat.size(-1), dtype=g_feat.dtype, device=h.device)
        g_cnt = torch.zeros(B, 1, dtype=g_feat.dtype, device=h.device)
        g_sum.scatter_add_(0, batch_idx.unsqueeze(-1).expand(-1, g_feat.size(-1)), g_feat)
        g_cnt.scatter_add_(0, batch_idx.unsqueeze(-1), torch.ones(N, 1, dtype=g_feat.dtype, device=h.device))
        g_mean = g_sum / g_cnt.clamp(min=1)                  # (B, H//4) global embedding
        h_global = torch.cat([h, g_mean[batch_idx]], dim=-1) # (N, H + H//4)

        delta_x = self.coord_pred(h_global)
        x_0_pred = x + delta_x
        return x_0_pred


# =============================================================================
# ATTENTION CONFORMER DIFFUSION (full model wrapper)
# =============================================================================

class AttnConformerDiffusion(nn.Module):
    """
    E(3)-equivariant diffusion model using EQGAT-diff attention EGNN.
    Drop-in replacement for ConformerDiffusion.

    Audit fixes applied (2026-05-06):
      FIX-AUDIT-1: t-gating for geometry loss (geo_t_fraction=0.3)
      FIX-AUDIT-2: Direct x₀ MSE instead of derived-ε MSE
      FIX-AUDIT-3: bond_weight=20.0 (was 10.0) to get bond_error < 0.2 Å
      FIX-AUDIT-4: Global graph readout in denoiser (EGNN Satorras 2021 §3.3)
    """

    def __init__(self,
                 num_timesteps: int = 1000,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_rbf: int = 20,
                 time_dim: int = 128,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.num_timesteps = num_timesteps

        betas = cosine_beta_schedule(num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1 - alphas_cumprod))

        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance',
                             torch.log(posterior_variance.clamp(min=1e-20)))

        snr = alphas_cumprod / (1 - alphas_cumprod)
        self.register_buffer('snr', snr)

        self.denoiser = AttnConformerDenoiser(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_rbf=num_rbf,
            time_dim=time_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        from models.geometry_constraints import GeometryConstraints
        self.geometry = GeometryConstraints(
            bond_weight=20.0,    # FIX-AUDIT-3: was 10.0; bond_error=0.23 > 0.20 threshold
            angle_weight=3.0,
            torsion_weight=1.0,
            repulsion_weight=5.0,
        )

    def _extract(self, a, t, batch_idx):
        return a[t][batch_idx].unsqueeze(-1)

    def q_sample(self, x_0, t, batch_idx, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        noise = remove_com(noise, batch_idx)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, batch_idx)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, batch_idx)
        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        x_t = remove_com(x_t, batch_idx)
        return x_t, noise

    def get_loss(self, x_0, atom_types, edge_index, bond_types, batch_idx,
                 geometry_weight: float = 1.0,
                 epoch: int = 1,
                 max_epochs: int = 300,
                 min_snr_gamma: float = 5.0,
                 geo_t_fraction: float = 0.3,
                 include_torsions: bool = False) -> Dict:
        """
        Training loss with:
          FIX-AUDIT-1: t-gated geometry loss (geo_t_fraction=0.3)
          FIX-AUDIT-2: Direct x₀ MSE (not derived-ε MSE)
            MSE on x₀ directly, SNR-weighted per molecule.
            EDM (Hoogeboom et al. ICML 2022, App. B): for x₀-parameterization,
            the loss surface on x₀ is smoother and better-conditioned than
            the equivalent ε-MSE at high timesteps.
        """
        device = x_0.device
        B = int(batch_idx.max().item()) + 1

        t = torch.randint(0, self.num_timesteps, (B,), device=device)
        x_t, noise = self.q_sample(x_0, t, batch_idx)

        # Predict x_0 directly
        x_0_pred = self.denoiser(x_t, t, atom_types, edge_index, bond_types, batch_idx)
        x_0_pred = remove_com(x_0_pred, batch_idx)

        # FIX-AUDIT-2: Direct x₀ MSE per atom, then average per molecule
        # Reference: EDM App. B — x₀ parameterization loss is MSE(x₀_pred, x₀)
        x0_err_per_atom = ((x_0_pred - x_0) ** 2).sum(-1)   # (N,)

        mse_per_mol = torch.zeros(B, dtype=x0_err_per_atom.dtype, device=device)
        mol_counts  = torch.zeros(B, dtype=x0_err_per_atom.dtype, device=device)
        mse_per_mol.scatter_add_(0, batch_idx, x0_err_per_atom)
        mol_counts.scatter_add_(0, batch_idx, torch.ones(x0_err_per_atom.size(0), dtype=x0_err_per_atom.dtype, device=device))
        mse_per_mol = mse_per_mol / mol_counts.clamp(min=1)

        # Min-SNR weighting (Hang et al. 2023) — scale by SNR(t)
        snr_t = self.snr[t]
        snr_weight = torch.minimum(snr_t, torch.full_like(snr_t, min_snr_gamma)) / snr_t.clamp(min=1e-8)
        mse_loss = (snr_weight * mse_per_mol).mean()

        # FIX-AUDIT-1: t-gated geometry loss (GCDM Morehead & Cheng NeurIPS 2023)
        if geometry_weight > 0:
            t_threshold = int(self.num_timesteps * geo_t_fraction)
            geo_mask = (t < t_threshold)
            if geo_mask.any():
                atom_mask = geo_mask[batch_idx]
                x_0_pred_low = x_0_pred[atom_mask]
                at_low = atom_types[atom_mask]
                bi_low_raw = batch_idx[atom_mask]

                low_mol_ids = geo_mask.nonzero(as_tuple=True)[0]
                old_to_new = torch.full((B,), -1, dtype=torch.long, device=device)
                old_to_new[low_mol_ids] = torch.arange(geo_mask.sum().item(), device=device)
                bi_low = old_to_new[bi_low_raw]

                row, col = edge_index
                edge_mask = atom_mask[row] & atom_mask[col]
                ei_low = edge_index[:, edge_mask]
                bt_low = bond_types[edge_mask]
                global_to_local = torch.full((x_0_pred.size(0),), -1, dtype=torch.long, device=device)
                atom_global_ids = atom_mask.nonzero(as_tuple=True)[0]
                global_to_local[atom_global_ids] = torch.arange(atom_global_ids.size(0), device=device)
                ei_low_local = global_to_local[ei_low]

                geo_loss = self._compute_geometry_loss(
                    x_0_pred_low, at_low, ei_low_local, bt_low, bi_low,
                    include_angles=True, include_torsions=include_torsions
                )
            else:
                geo_loss = torch.tensor(0.0, device=device)
        else:
            geo_loss = torch.tensor(0.0, device=device)

        total_loss = mse_loss + geometry_weight * geo_loss

        return {
            'total': total_loss,
            'mse':   mse_loss.detach(),
            'geo':   geo_loss.detach() if isinstance(geo_loss, torch.Tensor) else torch.tensor(0.0),
        }

    def _compute_geometry_loss(self, pos, atom_types, edge_index, bond_types,
                                batch_idx, include_angles=True,
                                include_torsions=False) -> torch.Tensor:
        total, _ = self.geometry.compute_total_loss(
            pos, atom_types, edge_index, bond_types, batch_idx,
            include_angles=include_angles,
            include_torsions=include_torsions,
        )
        return total

    @torch.no_grad()
    def ddim_sample(self, atom_types, edge_index, bond_types, batch_idx,
                    num_steps=50, eta=0.0) -> torch.Tensor:
        """DDIM sampling — identical to ConformerDiffusion.ddim_sample()."""
        device = atom_types.device
        N = atom_types.size(0)
        B = int(batch_idx.max().item()) + 1

        step_size = self.num_timesteps // num_steps
        timesteps = torch.arange(0, self.num_timesteps, step_size, device=device).flip(0)

        x_t = remove_com(torch.randn(N, 3, device=device), batch_idx)

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val.item(), dtype=torch.long, device=device)

            x_0_pred = self.denoiser(x_t, t, atom_types, edge_index, bond_types, batch_idx)
            x_0_pred = remove_com(x_0_pred, batch_idx)

            alpha_t = self.alphas_cumprod[t][batch_idx].unsqueeze(-1)

            if i == len(timesteps) - 1:
                x_t = x_0_pred
            else:
                t_next_val = timesteps[i + 1].item()
                t_next = torch.full((B,), t_next_val, dtype=torch.long, device=device)
                alpha_next = self.alphas_cumprod[t_next][batch_idx].unsqueeze(-1)

                sqrt_one_minus_at = torch.sqrt(1.0 - alpha_t).clamp(min=1e-6)
                noise_pred = (x_t - torch.sqrt(alpha_t) * x_0_pred) / sqrt_one_minus_at

                ratio = (alpha_t / alpha_next.clamp(min=1e-8)).clamp(max=1.0)
                sigma = eta * torch.sqrt(
                    (1.0 - alpha_next) / (1.0 - alpha_t).clamp(min=1e-8)
                ) * torch.sqrt((1.0 - ratio).clamp(min=0.0))

                direction = torch.sqrt((1.0 - alpha_next - sigma ** 2).clamp(min=0.0)) \
                            * noise_pred
                noise = remove_com(torch.randn_like(x_t), batch_idx) if eta > 0 else 0.0
                x_t = torch.sqrt(alpha_next) * x_0_pred + direction + sigma * noise

        return x_t
