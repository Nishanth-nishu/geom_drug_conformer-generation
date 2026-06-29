"""
dual_encoder_diffusion.py — Novel DualEncoderDiffusion Architecture

Novel contributions vs. existing literature:
  1. Distance-space score matching (GeoDiff) + Equivariant Attention (TorchMD-ET)
     → GeoDiff uses only SchNet/GIN, no attention.
     → TorchMD-ET has attention but is NOT a diffusion model.
  2. GFN2-xTB Boltzmann-weighted training loss (our contribution, not in GeoDiff/TorDiff)
  3. Energy-guided DDIM inference via EnergySurrogate gradient (our contribution)

Architecture overview:
  - Dual graph: local bond graph (GIN, captures topology) + global radius graph (SchNet+ET, captures 3D contacts)
  - Score prediction in DISTANCE space, not Cartesian space
    * Distance-space is SO(3)-invariant → model converges where Cartesian-space MSE cannot
    * Loss: MSE of predicted force contribution vs target denoising direction per edge
  - Equivariant reconstruction: scalar edge scores → 3D force vectors via unit-vector multiplication

Key differences from GeoDiff (Xu et al. ICML 2022):
  - We add Equivariant Multi-Head Attention (TorchMD-ET style) on top of SchNet
  - We add Boltzmann conformer weighting during training
  - We add GFN2-xTB energy guidance at inference

References:
  GeoDiff: Xu et al., ICML 2022. Distance-space diffusion, dual encoder.
  TorchMD-ET: Tholke & de Fabritiis, ICLR 2022. Equivariant attention + vector features.
  SchNet: Schütt et al., NeurIPS 2017. Distance-conditioned CFConv.
  GIN: Xu et al., ICLR 2019. Graph isomorphism network (local topology).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

import numpy as np


# =============================================================================
# NOISE SCHEDULE  (sigmoid, as per GeoDiff GEOM-Drugs config)
# =============================================================================

def sigmoid_beta_schedule(num_timesteps: int,
                           beta_start: float = 1e-7,
                           beta_end: float = 2e-3) -> torch.Tensor:
    """
    Sigmoid noise schedule as used in GeoDiff for GEOM-Drugs.
    Reference: GeoDiff configs/drugs_default.yml
    """
    betas = np.linspace(-6, 6, num_timesteps)
    betas = 1 / (np.exp(-betas) + 1)  # sigmoid
    betas = betas * (beta_end - beta_start) + beta_start
    return torch.from_numpy(betas).float()


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding for timestep conditioning. Handles odd dim."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
    )
    emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, 2*half)
    if dim % 2 == 1:  # pad if odd
        emb = F.pad(emb, (0, 1))
    return emb


# =============================================================================
# GRAPH UTILITIES  (pure PyTorch — no torch_scatter, no torch_sparse required)
# PyG 2.7 is available; we use pyg scatter where helpful but avoid hard deps.
# =============================================================================

def remove_com(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Remove center-of-mass per molecule (EDM Eq.4)."""
    B = int(batch.max().item()) + 1
    sums = torch.zeros(B, 3, device=x.device, dtype=x.dtype)
    cnts = torch.zeros(B, device=x.device, dtype=x.dtype)
    cnts.scatter_add_(0, batch, torch.ones(x.size(0), device=x.device))
    sums.scatter_add_(0, batch.unsqueeze(1).expand(-1, 3), x)
    mean = sums / cnts.unsqueeze(1).clamp(min=1)
    return x - mean[batch]


def scatter_add_2d(src: torch.Tensor, idx: torch.Tensor, dim_size: int) -> torch.Tensor:
    """scatter_add over dim=0 for 2D tensors — PyTorch native, no torch_scatter."""
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    out.scatter_add_(0, idx.unsqueeze(1).expand_as(src), src)
    return out


def scatter_add_1d(src: torch.Tensor, idx: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    out.scatter_add_(0, idx, src)
    return out


def extend_graph_order(edge_index: torch.Tensor,
                        edge_type: torch.Tensor,
                        num_nodes: int,
                        order: int = 3,
                        num_bond_types: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extend bond graph to k-th order neighbours via sparse BFS (NOT dense matmul).
    Order-2 edges correspond to angle neighbours, order-3 to dihedral neighbours.
    This is the key trick from GeoDiff common.py — gives the network implicit angle/dihedral info.

    FIX: Original used O(N²) dense adjacency matrix which OOMed on large molecules
    during DDP training. Now uses sparse BFS — O(E * order) complexity.

    Edge types:
      1..num_bond_types : actual bond types
      num_bond_types+1  : 2nd-order (angle) edges
      num_bond_types+2  : 3rd-order (dihedral) edges
    """
    from collections import defaultdict
    device = edge_index.device
    N = num_nodes

    # Build sparse adjacency list on CPU for BFS
    adj = defaultdict(set)
    for e in range(edge_index.size(1)):
        i = int(edge_index[0, e].item())
        j = int(edge_index[1, e].item())
        adj[i].add(j)

    # Track existing edges to avoid duplicates
    existing = set()
    for e in range(edge_index.size(1)):
        existing.add((int(edge_index[0, e].item()), int(edge_index[1, e].item())))

    new_edges = [(edge_index, edge_type)]

    for k in range(2, order + 1):
        found_pairs = []
        for start in range(N):
            # BFS up to k hops
            visited = {start}
            frontier = {start}
            for _hop in range(k):
                next_frontier = set()
                for n in frontier:
                    for nb in adj.get(n, set()):
                        if nb not in visited:
                            next_frontier.add(nb)
                            visited.add(nb)
                frontier = next_frontier
            # frontier now contains exactly the k-hop reachable nodes
            for end in frontier:
                if start != end and (start, end) not in existing:
                    found_pairs.append((start, end))
                    existing.add((start, end))

        if found_pairs:
            new_ei = torch.tensor(found_pairs, dtype=torch.long, device=device).t()
            new_et = torch.full((len(found_pairs),), num_bond_types + k - 1,
                                dtype=torch.long, device=device)
            new_edges.append((new_ei, new_et))
            # Also add these to the adjacency for next-order BFS
            # (NOT needed — we only want shortest-path order edges)

    all_ei = torch.cat([e[0] for e in new_edges], dim=1)
    all_et = torch.cat([e[1] for e in new_edges], dim=0)
    return all_ei, all_et


def build_radius_graph(pos: torch.Tensor,
                        batch: torch.Tensor,
                        cutoff: float = 10.0,
                        max_neighbors: int = 32) -> torch.Tensor:
    """
    Build radius graph: all pairs (i, j) within `cutoff` Å in the same molecule.

    FIX: Uses torch_cluster.radius_graph if available (much faster, CUDA-optimized),
    otherwise falls back to a vectorized cdist implementation.

    Returns edge_index (2, E_radius).
    """
    try:
        from torch_cluster import radius_graph
        return radius_graph(pos, r=cutoff, batch=batch,
                            max_num_neighbors=max_neighbors, loop=False)
    except ImportError:
        pass

    # Vectorized fallback — still faster than per-molecule Python loops
    B = int(batch.max().item()) + 1
    edges = []
    for b in range(B):
        mask = (batch == b)
        local_idx = mask.nonzero(as_tuple=True)[0]
        n = local_idx.size(0)
        if n < 2:
            continue
        p = pos[local_idx]                              # (n, 3)
        dists = torch.cdist(p, p)                       # (n, n)
        # Mask self-loops and edges beyond cutoff
        within = (dists < cutoff) & (dists > 0)
        # Cap to max_neighbors per atom (keep closest)
        if max_neighbors < n - 1:
            # Use kthvalue for efficient threshold
            k = min(max_neighbors, n - 1)
            # Set diagonal to inf so self-loops aren't in top-k
            dists_for_topk = dists.clone()
            dists_for_topk.fill_diagonal_(float('inf'))
            vals, _ = dists_for_topk.topk(k, dim=1, largest=False)
            thresh = vals[:, -1:]
            within = within & (dists <= thresh)
        local_pairs = within.nonzero(as_tuple=False)
        if local_pairs.size(0) == 0:
            continue
        global_pairs = local_idx[local_pairs]
        edges.append(global_pairs.t())
    if not edges:
        return torch.zeros(2, 0, dtype=torch.long, device=pos.device)
    return torch.cat(edges, dim=1)


def get_edge_distances(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distances for edges. Returns (E, 1)."""
    row, col = edge_index
    diff = pos[row] - pos[col]
    dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)
    return dist


def eq_transform(edge_scores: torch.Tensor,
                  pos: torch.Tensor,
                  edge_index: torch.Tensor,
                  edge_length: torch.Tensor,
                  clip_limit: Optional[float] = None) -> torch.Tensor:
    """
    Equivariant transform: per-edge scalars → per-node 3D force vectors.

    CORRECT bidirectional formulation (GeoDiff geometry.py):
      score_pos_i += score_d * (x_i - x_j) / d_ij    (force on atom i from edge i→j)
      score_pos_j -= score_d * (x_i - x_j) / d_ij    (reaction force on atom j)

    FIX: Added optional clip_limit parameter (GeoDiff's clip_norm) to prevent
    any single atom's force from dominating during sampling. Without this,
    sampling can explode due to large edge scores at high noise levels.

    Reference: GeoDiff geometry.py, Xu et al. ICML 2022, eq_transform()

    Args:
        edge_scores: (E, 1) scalar score per edge
        pos:         (N, 3) coordinates
        edge_index:  (2, E)
        edge_length: (E, 1) distances
        clip_limit:  Optional max L2 norm per atom (GeoDiff uses 1000 for global, None for local)

    Returns:
        (N, 3) per-atom 3D force vectors
    """
    row, col = edge_index
    N = pos.size(0)

    # Unit direction vector (x_i - x_j) / d_ij
    dd_dr = (pos[row] - pos[col]) / edge_length.clamp(min=1e-6)  # (E, 3)
    weighted = dd_dr * edge_scores                                  # (E, 3)

    # Bidirectional scatter: row gets +weighted, col gets -weighted
    score_pos = torch.zeros(N, 3, device=pos.device, dtype=pos.dtype)
    score_pos.scatter_add_(0, row.unsqueeze(1).expand_as(weighted), weighted)
    score_pos.scatter_add_(0, col.unsqueeze(1).expand_as(weighted), -weighted)

    # Clip per-atom force norm (GeoDiff clip_norm)
    if clip_limit is not None:
        norm = torch.norm(score_pos, dim=-1, keepdim=True).clamp(min=1e-8)
        denom = torch.where(norm > clip_limit, clip_limit / norm, torch.ones_like(norm))
        score_pos = score_pos * denom

    return score_pos


def gaussian_smearing(dist: torch.Tensor,
                       start: float = 0.0,
                       stop: float = 10.0,
                       num_gaussians: int = 50) -> torch.Tensor:
    """
    Gaussian basis function expansion for distances. Returns (E, num_gaussians).

    FIX: Extends stop by 1.0 Å beyond cutoff so edge-of-cutoff atoms
    still get meaningful RBF features (last Gaussian is not wasted at exact cutoff).
    """
    stop_adjusted = stop + 1.0  # margin beyond cutoff for smooth decay
    offset = torch.linspace(start, stop_adjusted, num_gaussians, device=dist.device)
    coeff = -0.5 / ((offset[1] - offset[0]).item() ** 2)
    d = dist.view(-1, 1) - offset.view(1, -1)
    return torch.exp(coeff * d ** 2)


# =============================================================================
# LOCAL ENCODER  (GIN-style — topology only, no coordinates)
# Reference: GeoDiff models/encoder/gin.py
# =============================================================================

class GINLocalEncoder(nn.Module):
    """
    Graph Isomorphism Network encoder for local topology.
    Operates ONLY on bond edges + higher-order topological edges.
    Does NOT use 3D coordinates — captures chemical topology.

    This encoder is crucial for ring detection, aromaticity, and
    hybridization — all of which determine the local conformational space.

    Reference: Xu et al. 'How Powerful are Graph Neural Networks?' ICLR 2019.
    """

    def __init__(self, hidden_dim: int = 256, num_convs: int = 4,
                 num_edge_types: int = 10, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_convs = num_convs

        # Atom type embedding (full periodic table coverage)
        self.atom_embed = nn.Embedding(100, hidden_dim)

        # Edge type embedding (bonds + higher-order edges)
        self.edge_embed = nn.Embedding(num_edge_types + 1, hidden_dim)

        # GIN layers: h_i = MLP(h_i + Σ_j (h_j + edge_ij))
        self.convs = nn.ModuleList()
        for _ in range(num_convs):
            self.convs.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            ))

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                atom_types: torch.Tensor,   # (N,) Long
                edge_index: torch.Tensor,   # (2, E_local) Long
                edge_type: torch.Tensor,    # (E_local,) Long
                ) -> torch.Tensor:          # (N, hidden_dim) Float

        h = self.atom_embed(atom_types.long().clamp(0, 99))  # Long→Float via Embedding
        max_et = self.edge_embed.num_embeddings - 1
        edge_feat = self.edge_embed(edge_type.long().clamp(0, max_et))  # Long→Float

        # Sanity-guard: embeddings always return float, but be explicit
        h = h.float()
        edge_feat = edge_feat.float()

        row, col = edge_index
        N = h.size(0)

        for conv in self.convs:
            # GIN aggregation: sum of (neighbor feat + edge feat)
            msg = h[row] + edge_feat                         # (E, H) Float
            agg = scatter_add_2d(msg, col, N)                # (N, H) Float
            h_new = conv(h + agg)                            # (N, H) Float
            h = h + self.dropout(h_new)                      # residual

        return h  # Float


# =============================================================================
# GLOBAL ENCODER  (SchNet + Equivariant Attention — novel combination)
# Reference: SchNet (Schütt 2017) + TorchMD-ET (Tholke & de Fabritiis 2022)
# =============================================================================

class CFConvBlock(nn.Module):
    """
    SchNet Continuous-Filter Convolution block.
    Edge features (from Gaussian smearing of distances) filter the messages.
    """

    def __init__(self, hidden_dim: int, num_gaussians: int = 50,
                 cutoff: float = 10.0, smooth: bool = True):
        super().__init__()
        self.cutoff = cutoff
        self.smooth = smooth

        # Filter network: distances → filter weights
        self.filter_net = nn.Sequential(
            nn.Linear(num_gaussians, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lin1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)

    def forward(self,
                h: torch.Tensor,           # (N, H) Float
                edge_index: torch.Tensor,  # (2, E) Long
                edge_length: torch.Tensor, # (E, 1) Float
                edge_rbf: torch.Tensor,    # (E, num_gaussians) Float
                ) -> torch.Tensor:         # (N, H) Float
        row, col = edge_index
        N = h.size(0)

        # Guard: all inputs must be float before hitting Linear
        h = h.float()
        edge_rbf = edge_rbf.float()
        edge_length = edge_length.float()

        # Cosine cutoff for smooth envelope
        if self.smooth:
            C = 0.5 * (torch.cos(edge_length * math.pi / self.cutoff) + 1.0)
            C = C * (edge_length <= self.cutoff).float() * (edge_length >= 0.0).float()
        else:
            C = (edge_length <= self.cutoff).float()  # (E, 1)

        W = self.filter_net(edge_rbf) * C              # (E, H)
        x = self.lin1(h)                               # (N, H)
        msg = x[row] * W                               # (E, H)
        agg = scatter_add_2d(msg, col, N)              # (N, H)
        return self.act(self.lin2(agg))


class EquivariantAttentionBlock(nn.Module):
    """
    Equivariant Multi-Head Attention over the radius graph.
    Maintains BOTH scalar features (h) AND vector features (vec ∈ R^{H×3}).

    This is the core novelty over GeoDiff — GeoDiff has no vector features.
    Vector features allow the model to maintain directional information
    across layers, enabling better modeling of ring systems and chirality.

    Reference: TorchMD-ET, Tholke & de Fabritiis, ICLR 2022. Sec. 2.2.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8,
                 num_rbf: int = 50, cutoff: float = 10.0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.cutoff = cutoff

        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim * 3)  # scalar + 2 vector channels

        # Distance-conditioned key/value projections (TorchMD-ET distance_influence='both')
        self.dk_proj = nn.Linear(num_rbf, hidden_dim)
        self.dv_proj = nn.Linear(num_rbf, hidden_dim * 3)

        # Vector projection
        self.vec_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)

        # Output projection
        self.o_proj = nn.Linear(hidden_dim, hidden_dim * 3)

        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.dk_proj,
                  self.dv_proj, self.o_proj]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight)

    def forward(self,
                h: torch.Tensor,           # (N, H)
                vec: torch.Tensor,         # (N, 3, H)
                edge_index: torch.Tensor,  # (2, E)
                edge_length: torch.Tensor, # (E,)
                edge_rbf: torch.Tensor,    # (E, num_rbf)
                edge_vec: torch.Tensor,    # (E, 3) unit vectors (x_i - x_j) / d_ij
                ) -> Tuple[torch.Tensor, torch.Tensor]:  # (N, H), (N, 3, H)

        row, col = edge_index
        N, H = h.size()
        nh, hd = self.num_heads, self.head_dim

        # Guard: all computation inputs must be float32
        h          = h.float()
        vec        = vec.float()
        edge_rbf   = edge_rbf.float()
        edge_length = edge_length.float()
        edge_vec   = edge_vec.float()

        h_norm = self.layer_norm(h)
        q = self.q_proj(h_norm).reshape(N, nh, hd)    # (N, nh, hd)
        k = self.k_proj(h_norm).reshape(N, nh, hd)    # (N, nh, hd)
        v = self.v_proj(h_norm).reshape(N, nh, hd * 3)  # (N, nh, 3*hd)

        # vec_proj splits into 3 × H projections for mixing vector and scalar
        vec_proj_out = self.vec_proj(h_norm)           # (N, 3H)
        vec1 = vec_proj_out[:, :H]                     # (N, H)
        vec2 = vec_proj_out[:, H:2*H]                  # (N, H)
        vec3 = vec_proj_out[:, 2*H:]                   # (N, H)

        # Distance-conditioned keys and values
        dk = F.silu(self.dk_proj(edge_rbf)).reshape(-1, nh, hd)      # (E, nh, hd)
        dv = F.silu(self.dv_proj(edge_rbf)).reshape(-1, nh, hd * 3)  # (E, nh, 3*hd)

        q_i = q[col]   # (E, nh, hd) — destination queries
        k_j = k[row]   # (E, nh, hd) — source keys
        v_j = v[row]   # (E, nh, 3*hd) — source values

        # Attention weights with distance conditioning
        attn = (q_i * k_j * dk).sum(dim=-1)  # (E, nh)
        attn = F.silu(attn)

        # Cosine cutoff — applied per edge
        cutoff_val = 0.5 * (torch.cos(edge_length * math.pi / self.cutoff) + 1.0)
        cutoff_val = cutoff_val * (edge_length <= self.cutoff).float()  # (E,)
        attn = attn * cutoff_val.unsqueeze(-1)  # (E, nh)

        # Value modulation by distance
        v_j = v_j * dv                                # (E, nh, 3*hd)
        x_j, vec1_j, vec2_j = v_j.split(hd, dim=2)  # each (E, nh, hd)

        # Scalar update
        x_j = x_j * attn.unsqueeze(2)                # (E, nh, hd)

        # Vector update: direction * scalar weight
        # edge_vec: (E, 3), vec2_j: (E, nh, hd) — contract over heads
        vec2_j_mean = vec2_j.mean(dim=1)              # (E, hd) — avg over heads for simplicity
        vec2_j_full = vec2_j_mean.unsqueeze(0).expand(3, -1, -1).permute(1, 0, 2)  # (E, 3, hd)  
        ev = edge_vec.unsqueeze(-1)                   # (E, 3, 1)
        vec_update = ev * vec2_j_full                 # (E, 3, hd) — direction-weighted value
        # Expand to full H
        pad_size = H - hd
        if pad_size > 0:
            vec_update = F.pad(vec_update, (0, pad_size))  # (E, 3, H)

        # Aggregate scalars
        x_j_flat = x_j.reshape(-1, H)                # (E, H)
        x_agg = scatter_add_2d(x_j_flat, col, N)     # (N, H)

        # Aggregate vectors
        vec_agg = torch.zeros(N, 3, H, device=h.device, dtype=h.dtype)
        vec_agg.scatter_add_(0, col.view(-1, 1, 1).expand_as(vec_update), vec_update)

        # Output projections
        o1, o2, o3 = self.o_proj(x_agg).split(H, dim=1)  # each (N, H)

        # Mix vector and scalar: vec→scalar via dot product
        # vec: (N, 3, H), vec2: (N, H) — contract: Σ_d vec[:,:,d] * vec2[:,d]
        vec_dot = (vec * vec2.unsqueeze(1)).sum(dim=1)  # (N, H) — dot along 3D axis → scalar

        # Scalar feature update: mix scalar from vector dot product
        dx = vec_dot * o2 + o3                        # (N, H)

        # Vector feature update
        dvec = vec * vec3.unsqueeze(1) + vec_agg      # (N, 3, H)

        return dx, dvec


class SchNetGlobalEncoder(nn.Module):
    """
    SchNet + Equivariant Attention encoder for the global radius graph.

    Interleaves SchNet CFConv blocks (distance-conditioned) with
    EquivariantAttentionBlocks (directional, vector-valued).

    The combination gives the model both:
    - Distance-conditioned filters (SchNet, good for energy landscapes)
    - Directional vector features (ET, good for geometry)

    Reference: SchNet (Schütt et al. NeurIPS 2017) + TorchMD-ET (Tholke 2022).
    """

    def __init__(self, hidden_dim: int = 256, num_convs: int = 6,
                 num_gaussians: int = 50, cutoff: float = 10.0,
                 num_attn_heads: int = 8, smooth: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        self.num_gaussians = num_gaussians

        self.embedding = nn.Embedding(100, hidden_dim)

        # Alternating SchNet + Attention blocks
        self.cfconv_blocks = nn.ModuleList([
            CFConvBlock(hidden_dim, num_gaussians, cutoff, smooth)
            for _ in range(num_convs)
        ])
        self.attn_blocks = nn.ModuleList([
            EquivariantAttentionBlock(hidden_dim, num_attn_heads, num_gaussians, cutoff)
            for _ in range(num_convs // 2)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_convs)
        ])

    def forward(self,
                atom_types: torch.Tensor,   # (N,)
                edge_index: torch.Tensor,   # (2, E_global)
                edge_length: torch.Tensor,  # (E, 1)
                edge_rbf: torch.Tensor,     # (E, num_gaussians)
                edge_vec: torch.Tensor,     # (E, 3) unit direction vectors
                ) -> Tuple[torch.Tensor, torch.Tensor]:  # (N, H), (N, 3, H)

        h = self.embedding(atom_types.long().clamp(0, 99)).float()  # Long→Float
        N, H = h.size()
        vec = torch.zeros(N, 3, H, device=h.device, dtype=torch.float32)

        # Guard: ensure rbf and edge_length are float
        edge_rbf    = edge_rbf.float()
        edge_length = edge_length.float()
        edge_vec    = edge_vec.float()

        attn_iter = iter(self.attn_blocks)
        for i, (cfconv, norm) in enumerate(zip(self.cfconv_blocks, self.norms)):
            # SchNet CFConv
            h_new = cfconv(h, edge_index, edge_length, edge_rbf)
            h = norm(h + h_new)

            # Every 2nd layer: add equivariant attention
            if i % 2 == 1:
                try:
                    attn = next(attn_iter)
                    dx, dvec = attn(h, vec, edge_index,
                                    edge_length.squeeze(-1), edge_rbf, edge_vec)
                    h = h + dx
                    vec = vec + dvec
                except StopIteration:
                    pass

        return h, vec


# =============================================================================
# DUAL ENCODER DENOISER
# =============================================================================

class DualEncoderDenoiser(nn.Module):
    """
    Dual encoder ε-network in distance space.

    Produces:
      edge_inv_local:  (E_local, 1) per-edge score for local graph
      edge_inv_global: (E_global, 1) per-edge score for global graph

    The scores are then converted to 3D forces via eq_transform().
    The loss is MSE between these forces and the target denoising direction.

    Architecture:
      Local:  GINLocalEncoder → h_local (N, H)
              → MLP([h_i * h_j, edge_feat]) → score_local (E_local, 1)
      Global: SchNetGlobalEncoder → h_global (N, H), vec (N, 3, H)
              → MLP([h_i * h_j, edge_feat]) → score_global (E_global, 1)
    """

    def __init__(self,
                 hidden_dim: int = 256,
                 num_convs_global: int = 6,
                 num_convs_local: int = 4,
                 num_gaussians: int = 50,
                 num_attn_heads: int = 8,
                 num_bond_types: int = 5,
                 edge_order: int = 3,
                 cutoff: float = 10.0,
                 time_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_order = edge_order
        self.num_bond_types = num_bond_types
        self.cutoff = cutoff

        # ── Local encoder (GIN) ──────────────────────────────────────────────
        num_edge_types = num_bond_types + edge_order  # bonds + higher-order
        self.local_encoder = GINLocalEncoder(
            hidden_dim=hidden_dim,
            num_convs=num_convs_local,
            num_edge_types=num_edge_types,
        )

        # ── Global encoder (SchNet + ET attention) ───────────────────────────
        self.global_encoder = SchNetGlobalEncoder(
            hidden_dim=hidden_dim,
            num_convs=num_convs_global,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            num_attn_heads=num_attn_heads,
        )

        # ── Edge feature encoder (distance + bond type → RBF + type embed) ──
        # Edge type embedding: Long index → H-dim float
        # Output H so that cat([rbf_G, embed_H]) = G+H = num_gaussians+hidden_dim
        # which is exactly what edge_encoder_local expects as input
        self.edge_type_embed_local = nn.Embedding(num_edge_types + 1, hidden_dim)
        self.edge_type_embed_global = nn.Embedding(num_edge_types + 1, hidden_dim)

        # ── Time conditioning ────────────────────────────────────────────────
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── Score prediction heads (pairwise features → scalar score per edge) ──
        # Input: [h_i * h_j (H) + edge_attr (H)] = 2H — matches GeoDiff exactly
        # Edge encoder: (RBF G + edge_type H) → H
        self.edge_encoder_local = nn.Sequential(
            nn.Linear(num_gaussians + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder_global = nn.Sequential(
            nn.Linear(num_gaussians, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.grad_local_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),  # 2H: h_i*h_j (H) + edge_attr (H)
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.grad_global_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self,
                atom_types: torch.Tensor,        # (N,)
                pos: torch.Tensor,               # (N, 3) noisy coordinates
                bond_index: torch.Tensor,        # (2, E_bond) original bonds
                bond_type: torch.Tensor,         # (E_bond,) bond orders
                batch: torch.Tensor,             # (N,)
                t: torch.Tensor,                 # (B,) timesteps
                edge_index_global: Optional[torch.Tensor] = None,  # precomputed radius graph
                ) -> Tuple[torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor]:
        """
        Returns:
            edge_inv_global: (E_global, 1) global edge scores
            edge_inv_local:  (E_local, 1) local edge scores
            edge_index_full: (2, E_total) — combined edge index
            edge_type_full:  (E_total,) — combined edge types
            edge_length_full: (E_total, 1) — combined edge lengths
            local_mask:      (E_total,) bool — which edges are local
        """
        N = atom_types.size(0)
        B = int(batch.max().item()) + 1

        # ── Step 1: Build dual graph ─────────────────────────────────────────
        # 1a. Extend bond graph to higher-order (angles/dihedrals)
        edge_index_local, edge_type_local = extend_graph_order(
            bond_index, bond_type, N, order=self.edge_order,
            num_bond_types=self.num_bond_types
        )

        # 1b. Build radius graph for global interactions
        if edge_index_global is None:
            edge_index_global = build_radius_graph(pos, batch, cutoff=self.cutoff)

        # 1c. Compute edge lengths
        edge_len_local = get_edge_distances(pos, edge_index_local)    # (E_local, 1)
        edge_len_global = get_edge_distances(pos, edge_index_global)  # (E_global, 1)

        # 1d. Combined edge index for global encoder
        # Global edges include both radius + local (bond) edges
        num_local = edge_index_local.size(1)
        num_global = edge_index_global.size(1)

        edge_index_full = torch.cat([edge_index_local, edge_index_global], dim=1)
        edge_type_global = torch.zeros(num_global, dtype=torch.long, device=pos.device)  # 0=radius
        edge_type_full = torch.cat([edge_type_local, edge_type_global], dim=0)
        edge_len_full = torch.cat([edge_len_local, edge_len_global], dim=0)

        local_mask = torch.cat([
            torch.ones(num_local, dtype=torch.bool, device=pos.device),
            torch.zeros(num_global, dtype=torch.bool, device=pos.device),
        ])

        # ── Step 2: Edge features ─────────────────────────────────────────────
        # Gaussian smearing for distance encoding
        rbf_local = gaussian_smearing(edge_len_local, 0.0, self.cutoff,
                                       self.global_encoder.num_gaussians)  # (E_local, G)
        rbf_global_all = gaussian_smearing(edge_len_full, 0.0, self.cutoff,
                                            self.global_encoder.num_gaussians)  # (E_total, G)

        # (edge_feat_local is built on-the-fly in Step 6 via edge_encoder_local)
        # No pre-computation needed here — avoids accidentally mixing Long into float ops.

        # ── Step 3: Time embedding (add to atom features via broadcast) ───────
        t_emb = sinusoidal_timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)  # (B, H)

        # ── Step 4: Local encoder ─────────────────────────────────────────────
        h_local = self.local_encoder(atom_types, edge_index_local, edge_type_local)
        h_local = h_local + t_emb[batch]  # add timestep conditioning

        # ── Step 5: Global encoder ────────────────────────────────────────────
        # Unit direction vectors for attention
        row_full, col_full = edge_index_full
        diff_full = pos[row_full] - pos[col_full]
        edge_vec_full = diff_full / edge_len_full.clamp(min=1e-6)  # (E_total, 3)

        h_global, vec_global = self.global_encoder(
            atom_types, edge_index_full, edge_len_full, rbf_global_all, edge_vec_full)
        h_global = h_global + t_emb[batch]  # add timestep conditioning

        # ── Step 6: Pairwise features → edge scores (GeoDiff assemble_atom_pair_feature) ────
        # Local: node product + encoded edge attr → 2H input
        row_l, col_l = edge_index_local
        rbf_local_enc = self.edge_encoder_local(
            torch.cat([rbf_local, self.edge_type_embed_local(
                edge_type_local.clamp(0, self.edge_type_embed_local.num_embeddings - 1)
            )], dim=-1)
        )  # (E_local, H)
        h_pair_local = h_local[row_l] * h_local[col_l]        # (E_local, H)
        score_input_local = torch.cat([h_pair_local, rbf_local_enc], dim=-1)  # (E_local, 2H)
        edge_inv_local = self.grad_local_mlp(score_input_local)   # (E_local, 1)

        # Global: node product + RBF encoded → 2H input
        row_g, col_g = edge_index_global
        rbf_g_only = gaussian_smearing(edge_len_global, 0.0, self.cutoff,
                                        self.global_encoder.num_gaussians)  # (E_global, G)
        rbf_global_enc = self.edge_encoder_global(rbf_g_only)     # (E_global, H)
        h_pair_global = h_global[row_g] * h_global[col_g]         # (E_global, H)
        score_input_global = torch.cat([h_pair_global, rbf_global_enc], dim=-1)  # (E_global, 2H)
        edge_inv_global = self.grad_global_mlp(score_input_global) # (E_global, 1)

        return (edge_inv_global, edge_inv_local,
                edge_index_full, edge_type_full, edge_len_full, local_mask,
                edge_index_local, edge_len_local, edge_index_global, edge_len_global)


# =============================================================================
# MAIN MODEL: DualEncoderDiffusion
# =============================================================================

class DualEncoderDiffusion(nn.Module):
    """
    Novel DualEncoderDiffusion for 3D molecular conformer generation.

    Key design choices:
    1. DISTANCE-SPACE SCORE MATCHING (not Cartesian x₀ MSE)
       - Invariant to rotation/translation by construction
       - Loss: MSE of per-atom force vectors (reconstructed from edge scores)
       - This is why GeoDiff converges where Cartesian-space models fail

    2. DUAL GRAPH ENCODER
       - Local: GIN on bond-extended graph (topology, angles, dihedrals)
       - Global: SchNet+ET Attention on radius graph (3D contacts, steric effects)

    3. SIGMOID NOISE SCHEDULE (5000 steps, GeoDiff GEOM-Drugs config)
       - Gradual noise injection preserves geometric structure longer

    4. GFN2-xTB ENERGY GUIDANCE at inference (our novel contribution)
       - Boltzmann conformer weighting during training
       - Energy-gradient guidance during DDIM sampling
    """

    def __init__(self,
                 hidden_dim: int = 256,
                 num_convs_global: int = 6,
                 num_convs_local: int = 4,
                 num_gaussians: int = 50,
                 num_attn_heads: int = 8,
                 num_bond_types: int = 5,
                 edge_order: int = 3,
                 cutoff: float = 10.0,
                 num_timesteps: int = 2000,
                 beta_start: float = 1e-7,
                 beta_end: float = 2e-3,
                 time_dim: int = 128,
                 w_global: float = 0.5,
                 ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.cutoff = cutoff
        self.w_global = w_global

        # Noise schedule (sigmoid, GeoDiff GEOM-Drugs)
        betas = sigmoid_beta_schedule(num_timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

        # SNR for loss weighting
        snr = alphas_cumprod / (1.0 - alphas_cumprod).clamp(min=1e-8)
        self.register_buffer('snr', snr)

        # Denoiser
        self.denoiser = DualEncoderDenoiser(
            hidden_dim=hidden_dim,
            num_convs_global=num_convs_global,
            num_convs_local=num_convs_local,
            num_gaussians=num_gaussians,
            num_attn_heads=num_attn_heads,
            num_bond_types=num_bond_types,
            edge_order=edge_order,
            cutoff=cutoff,
            time_dim=time_dim,
        )

    # ── Forward diffusion (noise injection) ────────────────────────────────────

    def q_sample(self, pos: torch.Tensor, t: torch.Tensor,
                  batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perturb coordinates at timestep t.
        Following GeoDiff: x_t = x_0 + ε * √(1-α_t) / √α_t

        (GeoDiff parameterization differs from EDM — it adds noise scaled
         by the signal-to-noise ratio's inverse square root)
        """
        a = self.alphas_cumprod[t]             # (B,)
        a_pos = a[batch].unsqueeze(-1)         # (N, 1)
        noise = torch.zeros_like(pos).normal_()
        pos_noisy = pos + noise * (1.0 - a_pos).sqrt() / a_pos.sqrt().clamp(min=1e-6)
        return pos_noisy, noise

    # ── Loss in distance space ─────────────────────────────────────────────────

    def get_loss(self,
                 pos: torch.Tensor,            # (N, 3) ground-truth coordinates
                 atom_types: torch.Tensor,     # (N,)
                 bond_index: torch.Tensor,     # (2, E_bond)
                 bond_type: torch.Tensor,      # (E_bond,)
                 batch: torch.Tensor,          # (N,)
                 min_snr_gamma: float = 5.0,
                 boltzmann_weights: Optional[torch.Tensor] = None,  # (B,)
                 anneal_power: float = 2.0,
                 ) -> Dict[str, torch.Tensor]:
        """
        Distance-space score matching loss (GeoDiff-style).

        For each edge (i, j):
          d_gt = ||x0_i - x0_j||
          d_t  = ||xt_i - xt_j||
          target = (d_gt - d_t) / √(1-α_t) * √α_t   (denoising direction in distance)
          prediction = F_ij (from denoiser)

        Loss = MSE(F_local → target_local_force) + MSE(F_global → target_global_force)

        The force targets are converted to 3D via eq_transform():
          target_force_i = Σ_j target_ij * (xt_i - xt_j) / ||xt_i - xt_j||
        """
        B = int(batch.max().item()) + 1
        N = pos.size(0)

        # Sample timesteps (antithetic — cover low and high noise)
        t = torch.randint(0, self.num_timesteps, size=(B // 2 + 1,), device=pos.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:B]

        # Perturb positions
        pos = remove_com(pos, batch)
        pos_t, _ = self.q_sample(pos, t, batch)

        a = self.alphas_cumprod[t]
        a_pos = a[batch].unsqueeze(-1)

        # Forward pass
        (edge_inv_global, edge_inv_local,
         edge_index_full, edge_type_full, edge_len_full, local_mask,
         edge_index_local, edge_len_local,
         edge_index_global, edge_len_global) = self.denoiser(
            atom_types, pos_t, bond_index, bond_type, batch, t
        )

        # ── Distance-space targets ────────────────────────────────────────────
        # Local target
        a_edge_local = a[batch[edge_index_local[0]]].unsqueeze(-1)  # (E_local, 1)
        d_gt_local = get_edge_distances(pos, edge_index_local)
        d_t_local = edge_len_local
        d_target_local = ((d_gt_local - d_t_local) /
                          (1.0 - a_edge_local).sqrt().clamp(min=1e-8) *
                          a_edge_local.sqrt())   # (E_local, 1)

        # Global target
        a_edge_global = a[batch[edge_index_global[0]]].unsqueeze(-1)
        d_gt_global = get_edge_distances(pos, edge_index_global)
        d_t_global = edge_len_global
        d_target_global = ((d_gt_global - d_t_global) /
                           (1.0 - a_edge_global).sqrt().clamp(min=1e-8) *
                           a_edge_global.sqrt())   # (E_global, 1)

        # Cutoff mask for global (only edges within cutoff contribute)
        global_cutoff_mask = (d_t_global <= self.cutoff)   # (E_global, 1)
        d_target_global = d_target_global * global_cutoff_mask
        edge_inv_global = edge_inv_global * global_cutoff_mask

        # ── Convert to 3D forces via eq_transform ────────────────────────────
        target_force_local = eq_transform(d_target_local, pos_t,
                                           edge_index_local, d_t_local)   # (N, 3)
        pred_force_local = eq_transform(edge_inv_local, pos_t,
                                         edge_index_local, d_t_local)     # (N, 3)

        target_force_global = eq_transform(d_target_global, pos_t,
                                            edge_index_global, d_t_global)  # (N, 3)
        pred_force_global = eq_transform(edge_inv_global, pos_t,
                                          edge_index_global, d_t_global)    # (N, 3)

        # ── Per-molecule losses ───────────────────────────────────────────────
        loss_local_per_atom = ((pred_force_local - target_force_local) ** 2).sum(-1)  # (N,)
        loss_global_per_atom = ((pred_force_global - target_force_global) ** 2).sum(-1)

        # Scale by global weight (GeoDiff: local weight=5, global weight=2)
        loss_per_atom = 5.0 * loss_local_per_atom + 2.0 * self.w_global * loss_global_per_atom

        # Reduce per molecule
        loss_per_mol = scatter_add_1d(loss_per_atom, batch, B)      # (B,)
        mol_counts = scatter_add_1d(torch.ones(N, device=pos.device), batch, B)
        loss_per_mol = loss_per_mol / mol_counts.clamp(min=1)        # (B,)

        # Min-SNR weighting per molecule
        snr_t = self.snr[t]   # (B,)
        snr_weight = torch.minimum(snr_t, snr_t.new_full((), min_snr_gamma)) / snr_t.clamp(min=1e-8)

        # Boltzmann weighting (novel contribution)
        if boltzmann_weights is not None:
            bw_norm = boltzmann_weights * B / boltzmann_weights.sum().clamp(min=1e-8)
            combined_weight = snr_weight * bw_norm
        else:
            combined_weight = snr_weight

        total_loss = (combined_weight * loss_per_mol).mean()

        return {
            'total': total_loss,
            'local': (5.0 * loss_per_atom.mean()).detach(),
            'global': (2.0 * self.w_global * loss_per_atom.mean()).detach(),
        }

    # ── Sampling (DDIM) ────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(self,
                    atom_types: torch.Tensor,
                    bond_index: torch.Tensor,
                    bond_type: torch.Tensor,
                    batch: torch.Tensor,
                    num_steps: int = 100,
                    w_global: float = 0.5,
                    eta: float = 0.0,
                    clip_pos: Optional[float] = None,
                    ) -> torch.Tensor:
        """
        DDIM-style sampling in distance space.
        Reconstructs coordinates via the GeoDiff DDPM-noisy update rule.
        """
        device = atom_types.device
        N = atom_types.size(0)
        B = int(batch.max().item()) + 1

        betas = self.betas
        a_cumprod = self.alphas_cumprod

        # Timestep sequence (last `num_steps` timesteps reversed)
        seq = list(range(self.num_timesteps - num_steps, self.num_timesteps))
        seq_next = [-1] + seq[:-1]

        sigmas = (1.0 - a_cumprod).sqrt() / a_cumprod.sqrt().clamp(min=1e-8)
        pos = remove_com(torch.randn(N, 3, device=device), batch) * sigmas[-1]

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((B,), i, dtype=torch.long, device=device)

            (edge_inv_global, edge_inv_local,
             _, _, _,  _,
             edge_index_local, edge_len_local,
             edge_index_global, edge_len_global) = self.denoiser(
                atom_types, pos, bond_index, bond_type, batch, t
            )

            # Local force
            f_local = eq_transform(edge_inv_local, pos, edge_index_local, edge_len_local)

            # Global force (only within cutoff)
            cutoff_mask = (edge_len_global <= self.cutoff)
            edge_inv_global = edge_inv_global * (1.0 - cutoff_mask.float()) * 0  \
                              + edge_inv_global * cutoff_mask.float()
            f_global = eq_transform(edge_inv_global, pos, edge_index_global, edge_len_global)

            eps_pos = f_local + f_global * w_global

            # DDPM-noisy update (GeoDiff Eq. 10)
            at = a_cumprod[i]
            at_next = a_cumprod[j] if j >= 0 else torch.ones(1, device=device)

            beta_t = 1.0 - at / at_next
            e = -eps_pos
            pos0_from_e = (1.0 / at).sqrt() * pos - (1.0 / at - 1).sqrt() * e

            mean_eps = (
                (at_next.sqrt() * beta_t) * pos0_from_e +
                ((1 - beta_t).sqrt() * (1 - at_next)) * pos
            ) / (1.0 - at).clamp(min=1e-8)

            noise = torch.randn_like(pos)
            mask = 1.0 - (torch.tensor(i, device=device) == 0).float()
            logvar = beta_t.log().clamp(min=-20)
            pos = mean_eps + mask * torch.exp(0.5 * logvar) * noise

            if torch.isnan(pos).any():
                print(f'  [Warning] NaN at step {i}, resetting to mean')
                pos = mean_eps

            pos = remove_com(pos, batch)
            if clip_pos is not None:
                pos = pos.clamp(-clip_pos, clip_pos)

        return pos

    @torch.no_grad()
    def energy_guided_ddim_sample(self,
                                   atom_types: torch.Tensor,
                                   bond_index: torch.Tensor,
                                   bond_type: torch.Tensor,
                                   batch: torch.Tensor,
                                   energy_surrogate,
                                   guidance_scale: float = 1.0,
                                   num_steps: int = 100,
                                   ) -> torch.Tensor:
        """
        Energy-guided DDIM: at each step, nudge predicted x₀ toward low energy.
        This is our novel contribution — not in GeoDiff or TorDiff.
        """
        device = atom_types.device
        N = atom_types.size(0)
        B = int(batch.max().item()) + 1

        a_cumprod = self.alphas_cumprod
        betas = self.betas

        seq = list(range(self.num_timesteps - num_steps, self.num_timesteps))
        seq_next = [-1] + seq[:-1]
        sigmas = (1.0 - a_cumprod).sqrt() / a_cumprod.sqrt().clamp(min=1e-8)

        pos = remove_com(torch.randn(N, 3, device=device), batch) * sigmas[-1]

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((B,), i, dtype=torch.long, device=device)

            (edge_inv_global, edge_inv_local,
             _, _, _, _,
             edge_index_local, edge_len_local,
             edge_index_global, edge_len_global) = self.denoiser(
                atom_types, pos, bond_index, bond_type, batch, t
            )

            f_local = eq_transform(edge_inv_local, pos, edge_index_local, edge_len_local)
            f_global = eq_transform(edge_inv_global, pos, edge_index_global, edge_len_global)
            eps_pos = f_local + f_global * self.w_global

            at = a_cumprod[i]
            at_next = a_cumprod[j] if j >= 0 else torch.ones(1, device=device)
            e = -eps_pos
            pos0_pred = (1.0 / at).sqrt() * pos - (1.0 / at - 1).sqrt() * e

            # Energy guidance — stronger when denoising is near final (low t → small σ)
            # Power-law schedule: γ(t) = γ_max * (1 - α_t)^0.5
            gamma_t = guidance_scale * (1.0 - at).sqrt()
            if gamma_t > 1e-6 and energy_surrogate is not None:
                pos0_g = pos0_pred.detach().requires_grad_(True)
                with torch.enable_grad():
                    E = energy_surrogate(pos0_g, atom_types, edge_index_local,
                                         bond_type, batch)
                    grad_E = torch.autograd.grad(E.sum(), pos0_g)[0]
                pos0_pred = pos0_pred - gamma_t * grad_E.detach()
                pos0_pred = remove_com(pos0_pred, batch)

            # DDPM-noisy update
            beta_t = 1.0 - at / at_next
            mean_eps = (
                (at_next.sqrt() * beta_t) * pos0_pred +
                ((1 - beta_t).sqrt() * (1 - at_next)) * pos
            ) / (1.0 - at).clamp(min=1e-8)

            noise = torch.randn_like(pos)
            mask = 1.0 - (torch.tensor(i, device=device) == 0).float()
            logvar = beta_t.log().clamp(min=-20)
            pos = mean_eps + mask * torch.exp(0.5 * logvar) * noise
            pos = remove_com(pos, batch)

        return pos
