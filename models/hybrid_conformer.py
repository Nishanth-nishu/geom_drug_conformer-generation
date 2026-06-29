"""
hybrid_conformer.py — Exp E SOTA Hybrid Model

Combines:
  - Flow Matching (from Exp C)
  - Attention-Enhanced EGNN backbone (from Exp B)
  - Torsion-Angle Auxiliary Loss (from Exp D)

Usage:
  from models.hybrid_conformer import HybridFlowMatchingConformer
  model = HybridFlowMatchingConformer(hidden_dim=256, num_layers=6, num_heads=4)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conformer_diffusion import remove_com
from models.attn_conformer_diffusion import AttnConformerDenoiser

class HybridFlowMatchingConformer(nn.Module):
    def __init__(self,
                 hidden_dim: int = 256,
                 num_layers:  int = 6,
                 num_heads:   int = 4,
                 time_dim:    int = 128):
        super().__init__()
        
        # We use the Attention-enhanced denoiser to predict v_theta
        self.velocity_net = AttnConformerDenoiser(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            time_dim=time_dim,
            num_heads=num_heads,
        )

    def _interpolate(self, x0: torch.Tensor, t: torch.Tensor,
                     batch_idx: torch.Tensor):
        eps = torch.randn_like(x0)
        eps = remove_com(eps, batch_idx)

        t_atom = t[batch_idx].unsqueeze(-1)
        x_t = (1 - t_atom) * x0 + t_atom * eps
        x_t = remove_com(x_t, batch_idx)

        v_target = eps - x0
        return x_t, v_target, eps

    def get_loss(self, coords: torch.Tensor, atom_types: torch.Tensor,
                 edge_index: torch.Tensor, bond_types: torch.Tensor,
                 batch_idx: torch.Tensor,
                 geometry_weight: float = 0.5,
                 include_torsions: bool = True,
                 **kwargs) -> dict:
        device = coords.device
        B = int(batch_idx.max().item()) + 1

        # Ensure input is in zero-CoM subspace (EDM Hoogeboom 2022, Sec 3.1)
        coords = remove_com(coords, batch_idx)

        t = torch.rand(B, device=device)
        x_t, v_target, eps = self._interpolate(coords, t, batch_idx)

        t_int = (t * 999).long().clamp(0, 999)
        t_int_per_atom = t_int[batch_idx]

        v_pred = self.velocity_net(
            x_t,
            t_int_per_atom,
            atom_types,
            edge_index,
            bond_types,
            batch_idx,
        )

        mse = F.mse_loss(v_pred, v_target)

        geo_loss = torch.tensor(0.0, device=device)
        if geometry_weight > 0:
            try:
                from models.geometry_constraints import GeometryConstraints
                gc = GeometryConstraints()
                t_atom = t[batch_idx].unsqueeze(-1)
                x0_hat = x_t - t_atom * v_pred
                x0_hat = remove_com(x0_hat, batch_idx)
                
                geo_total, _ = gc.compute_total_loss(
                    x0_hat, atom_types, edge_index, bond_types, batch_idx,
                    include_angles=True, include_torsions=include_torsions,
                )
                geo_loss = geo_total
            except Exception as e:
                pass

        total = mse + geometry_weight * geo_loss
        return {'total': total, 'mse': mse, 'geo': geo_loss}

    @torch.no_grad()
    def ode_sample(self, atom_types: torch.Tensor, edge_index: torch.Tensor,
                   bond_types: torch.Tensor, batch_idx: torch.Tensor,
                   num_steps: int = 100) -> torch.Tensor:
        N = atom_types.size(0)
        device = atom_types.device
        B = int(batch_idx.max().item()) + 1

        x = torch.randn(N, 3, device=device)
        x = remove_com(x, batch_idx)

        dt = -1.0 / num_steps
        t_vals = torch.linspace(1.0, 0.0 + (1.0 / num_steps), num_steps, device=device)

        for t_val in t_vals:
            t_mol = t_val.expand(B)
            t_int_mol = (t_mol * 999).long().clamp(0, 999)
            t_int_atom = t_int_mol[batch_idx]

            v = self.velocity_net(
                x,
                t_int_atom,
                atom_types,
                edge_index,
                bond_types,
                batch_idx,
            )
            x = x + v * dt
            x = remove_com(x, batch_idx)

        return x

    @torch.no_grad()
    def ddim_sample(self, atom_types: torch.Tensor, edge_index: torch.Tensor,
                    bond_types: torch.Tensor, batch_idx: torch.Tensor,
                    num_steps: int = 100) -> torch.Tensor:
        return self.ode_sample(atom_types, edge_index, bond_types, batch_idx, num_steps)
