"""
debug_dtype.py — runs one batch and prints dtype of every tensor at each model step.
Run: PYTHONPATH=. python3 debug_dtype.py
"""
import torch
import traceback

# ── 1. Import the model ───────────────────────────────────────────────────────
from models.dual_encoder_diffusion import (
    DualEncoderDiffusion, DualEncoderDenoiser,
    extend_graph_order, build_radius_graph,
    get_edge_distances, gaussian_smearing,
    sinusoidal_timestep_embedding, remove_com,
    scatter_add_2d,
)
from data.geom_drugs_dataset import make_geom_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── 2. Load one small batch ────────────────────────────────────────────────────
train_loader, val_loader = make_geom_dataloaders(
    data_path='data/geom_drugs.jsonl',
    batch_size=4,
    num_workers=0,
    max_atoms=25,
    max_mols=100,
    max_conformers=2,
)

batch = next(iter(train_loader))
pos        = batch['coordinates'].to(device)
atom_types = batch['atom_types'].to(device)
edge_index = batch['edge_index'].to(device)
bond_types = batch['bond_types'].to(device)
batch_idx  = batch['batch_idx'].to(device)

print("=== Input dtypes ===")
print(f"  pos:        {pos.dtype} {pos.shape}")
print(f"  atom_types: {atom_types.dtype} {atom_types.shape}")
print(f"  edge_index: {edge_index.dtype} {edge_index.shape}")
print(f"  bond_types: {bond_types.dtype} {bond_types.shape} vals={bond_types.unique()}")
print(f"  batch_idx:  {batch_idx.dtype} {batch_idx.shape}")

pos = remove_com(pos, batch_idx)

# ── 3. Build the model ─────────────────────────────────────────────────────────
model = DualEncoderDiffusion(
    hidden_dim=64, num_convs_global=2, num_convs_local=2,
    num_gaussians=16, num_attn_heads=4,
    num_bond_types=5, edge_order=3, cutoff=10.0,
    num_timesteps=100,
).to(device)

denoiser = model.denoiser
N = atom_types.size(0)
B = int(batch_idx.max().item()) + 1

# ── 4. Step-by-step forward with dtype printing ────────────────────────────────
print("\n=== Step 1: extend_graph_order ===")
edge_index_local, edge_type_local = extend_graph_order(
    edge_index, bond_types, N, order=3, num_bond_types=5
)
print(f"  edge_type_local: {edge_type_local.dtype} vals={edge_type_local.unique()}")
print(f"  edge_index_local: {edge_index_local.dtype} shape={edge_index_local.shape}")

print("\n=== Step 2: build_radius_graph ===")
edge_index_global = build_radius_graph(pos, batch_idx, cutoff=10.0)
print(f"  edge_index_global: {edge_index_global.dtype} shape={edge_index_global.shape}")

print("\n=== Step 3: edge distances ===")
edge_len_local  = get_edge_distances(pos, edge_index_local)
edge_len_global = get_edge_distances(pos, edge_index_global)
print(f"  edge_len_local:  {edge_len_local.dtype} shape={edge_len_local.shape}")
print(f"  edge_len_global: {edge_len_global.dtype} shape={edge_len_global.shape}")

print("\n=== Step 4: Combined edges ===")
num_local  = edge_index_local.size(1)
num_global = edge_index_global.size(1)
edge_index_full = torch.cat([edge_index_local, edge_index_global], dim=1)
edge_type_global = torch.zeros(num_global, dtype=torch.long, device=device)
edge_type_full = torch.cat([edge_type_local, edge_type_global], dim=0)
edge_len_full = torch.cat([edge_len_local, edge_len_global], dim=0)
print(f"  edge_len_full: {edge_len_full.dtype} shape={edge_len_full.shape}")

print("\n=== Step 5: gaussian_smearing ===")
rbf_local = gaussian_smearing(edge_len_local, 0.0, 10.0, 16)
rbf_global_all = gaussian_smearing(edge_len_full, 0.0, 10.0, 16)
print(f"  rbf_local: {rbf_local.dtype} shape={rbf_local.shape}")
print(f"  rbf_global_all: {rbf_global_all.dtype} shape={rbf_global_all.shape}")

print("\n=== Step 6: timestep embedding ===")
t = torch.randint(0, 100, (B,), device=device)
print(f"  t: {t.dtype}")
t_emb_raw = sinusoidal_timestep_embedding(t, denoiser.time_mlp[0].in_features)
print(f"  t_emb_raw: {t_emb_raw.dtype} shape={t_emb_raw.shape}")
t_emb = denoiser.time_mlp(t_emb_raw)
print(f"  t_emb:     {t_emb.dtype} shape={t_emb.shape}")

print("\n=== Step 7: GINLocalEncoder ===")
try:
    h_local = denoiser.local_encoder(atom_types, edge_index_local, edge_type_local)
    print(f"  h_local: {h_local.dtype} shape={h_local.shape}")
except Exception as exc:
    print(f"  ERROR in GINLocalEncoder:")
    traceback.print_exc()

print("\n=== Step 8: SchNetGlobalEncoder ===")
try:
    row_full, col_full = edge_index_full
    diff_full = pos[row_full] - pos[col_full]
    edge_vec_full = diff_full / edge_len_full.clamp(min=1e-6)
    print(f"  edge_vec_full: {edge_vec_full.dtype}")
    h_global, vec_global = denoiser.global_encoder(
        atom_types, edge_index_full, edge_len_full, rbf_global_all, edge_vec_full
    )
    print(f"  h_global: {h_global.dtype} shape={h_global.shape}")
except Exception as exc:
    print(f"  ERROR in SchNetGlobalEncoder:")
    traceback.print_exc()

print("\n=== Step 9: edge_encoder_local ===")
try:
    et_embed = denoiser.edge_type_embed_local(
        edge_type_local.clamp(0, denoiser.edge_type_embed_local.num_embeddings - 1)
    )
    print(f"  et_embed: {et_embed.dtype} shape={et_embed.shape}")
    cat_in = torch.cat([rbf_local, et_embed], dim=-1)
    print(f"  cat_in: {cat_in.dtype} shape={cat_in.shape}")
    rbf_local_enc = denoiser.edge_encoder_local(cat_in)
    print(f"  rbf_local_enc: {rbf_local_enc.dtype}")
except Exception as exc:
    print(f"  ERROR in edge_encoder_local:")
    traceback.print_exc()

print("\n=== Step 10: grad_local_mlp ===")
try:
    h_local = h_local + t_emb[batch_idx]
    h_global = h_global + t_emb[batch_idx]
    row_l, col_l = edge_index_local
    h_pair_local = h_local[row_l] * h_local[col_l]
    score_in = torch.cat([h_pair_local, rbf_local_enc], dim=-1)
    print(f"  score_in: {score_in.dtype} shape={score_in.shape}")
    edge_inv_local = denoiser.grad_local_mlp(score_in)
    print(f"  edge_inv_local: {edge_inv_local.dtype}")
except Exception as exc:
    print(f"  ERROR in grad_local_mlp:")
    traceback.print_exc()

print("\nDone — no crash! The error must come from get_loss. Running full get_loss...")
try:
    loss_dict = model.get_loss(
        pos=pos, atom_types=atom_types,
        bond_index=edge_index, bond_type=bond_types,
        batch=batch_idx, min_snr_gamma=5.0,
    )
    print(f"Loss: {loss_dict['total'].item():.6f}")
except Exception as exc:
    print(f"\nERROR in get_loss:")
    traceback.print_exc()
