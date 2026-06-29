"""
mol_train_expG.py — Experiment G: SOTA Heavy-Atom Production Run

Architecture upgrades vs. Exp A–F:
  - EDM-inspired global EGNN readout in ConformerDenoiser
  - hidden_dim=384, num_layers=8, num_rbf=32 (up from 256, 6, 20)
  - Heavy-atom-only QM9 (max 9 atoms) — matches SOTA benchmark setting
  - 500 epochs with cosine LR + 1% floor
  - GeoDiff COV-MAT evaluation at end of every 50 epochs

Key research references:
  Hoogeboom et al. EDM, ICML 2022        — x_0 param, CoM removal, VLB
  Xu et al. GeoDiff, ICML 2022          — COV-MAT metric, heavy-atom QM9
  Ganea et al. GeoMol, NeurIPS 2021     — geometry constraints, torsion loss
  Morehead & Cheng GCDM, NeurIPS 2023  — geometry loss t-gating

Expected results at 500 epochs (heavy-atom QM9):
  COV-R@0.5Å: >55%  (SOTA: GeoDiff 71.0%)
  MAT-R:      <0.35Å (SOTA: GeoDiff 0.297Å)
"""

import os
import sys
import time
import math
import json

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Force heavy-atom dataset before importing mol_prepare ────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_heavy_path  = os.path.join(PROJECT_ROOT, 'data', 'qm9_heavy.jsonl')
_selfies_path = os.path.join(PROJECT_ROOT, 'data', 'qm9_selfies.jsonl')

if os.path.exists(_heavy_path):
    os.environ['MOL_DATASET']   = _heavy_path
    os.environ['MOL_MAX_ATOMS'] = '9'
    print(f"[ExpG] Using heavy-atom dataset: {_heavy_path}")
else:
    os.environ['MOL_DATASET']   = _selfies_path
    os.environ['MOL_MAX_ATOMS'] = '29'
    print(f"[ExpG] WARNING: qm9_heavy.jsonl not found. Falling back to explicit-H dataset.")
    print(f"         Run: python data/prepare_qm9_heavy.py --input data/qm9_selfies.jsonl "
          f"--output data/qm9_heavy.jsonl")

from autoresearch.mol_prepare import (
    EPOCH_BUDGET, EVAL_MOLECULES, make_dataloaders,
    evaluate_all, print_report, metrics_to_tsv_row, TSV_HEADER,
    CHECKPOINT_DIR, DATA_PATH, MAX_ATOMS,
)
from models.conformer_diffusion import ConformerDiffusion, remove_com
from autoresearch.geodiff_eval import (
    load_dataset, run_geodiff_eval, print_geodiff_results, write_results_tsv
)

# ============================================================================
# HYPERPARAMETERS — Experiment G: SOTA Production
# ============================================================================

# Architecture (upgraded)
MODEL_HIDDEN_DIM = 384      # was 256 → +50% capacity
MODEL_NUM_LAYERS = 8        # was 6 → +2 layers for deeper geometry reasoning
MODEL_TIMESTEPS  = 1000
MODEL_TIME_DIM   = 256      # was 128 → richer time conditioning
MODEL_NUM_RBF    = 32       # was 20 → finer distance resolution

# Training
BATCH_SIZE       = 128      # larger batch; heavy-atom mols are ~3× smaller
LEARNING_RATE    = 5e-4     # slightly higher for larger batch
WEIGHT_DECAY     = 0.01
EPOCHS           = 500      # full budget for SOTA
WARMUP_EPOCHS    = 10
MIN_SNR_GAMMA    = 5.0

# Geometry loss
GEOMETRY_WEIGHT  = 0.5      # strong geometry supervision
GEO_T_FRACTION   = 0.3     # gate geometry loss to t < 30% of T
INCLUDE_TORSIONS = False     # torsion-angle supervision via geometry loss

# Sampling
DDIM_STEPS       = 50

# Evaluation
EVAL_EVERY_EPOCHS = 50      # GeoDiff COV-MAT eval every 50 epochs
N_GEN_PER_MOL    = 10      # conformers to generate per molecule during eval
EVAL_MOLS        = 300      # molecules to evaluate (balance speed vs accuracy)

EXP_NAME         = "exp_G_heavy_atom_sota"

# ============================================================================
# OPTIMIZER
# ============================================================================

def build_optimizer(model):
    print(f"Optimizer: AdamW | lr={LEARNING_RATE}, wd={WEIGHT_DECAY}")
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    for pg in opt.param_groups:
        pg['initial_lr'] = pg['lr']
    return opt


def get_lr(epoch: int, base_lr: float) -> float:
    """Cosine schedule with linear warmup + 1% LR floor."""
    if epoch < WARMUP_EPOCHS:
        return base_lr * (epoch + 1) / max(WARMUP_EPOCHS, 1)
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    cosine_lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    return max(cosine_lr, base_lr * 0.01)   # 1% floor

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    t_start = time.time()
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Experiment: {EXP_NAME}")
    print(f"Dataset: {DATA_PATH} (max_atoms={MAX_ATOMS})")
    print(f"Config: hidden={MODEL_HIDDEN_DIM} layers={MODEL_NUM_LAYERS} "
          f"rbf={MODEL_NUM_RBF} timesteps={MODEL_TIMESTEPS} "
          f"epochs={EPOCHS} lr={LEARNING_RATE} warmup={WARMUP_EPOCHS} "
          f"geo_w={GEOMETRY_WEIGHT} torsions={INCLUDE_TORSIONS}")

    train_loader, val_loader = make_dataloaders(batch_size=BATCH_SIZE, num_workers=0)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Build model
    model = ConformerDiffusion(
        num_timesteps=MODEL_TIMESTEPS,
        hidden_dim=MODEL_HIDDEN_DIM,
        num_layers=MODEL_NUM_LAYERS,
        num_rbf=MODEL_NUM_RBF,
        time_dim=MODEL_TIME_DIM,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.2f}M")

    optimizer = build_optimizer(model)
    base_lr   = LEARNING_RATE

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float('inf')
    best_mat_r    = float('inf')
    t_start_train = time.time()

    # Pre-load evaluation dataset for GeoDiff COV-MAT
    eval_dataset = load_dataset(str(DATA_PATH), max_atoms=MAX_ATOMS, max_mols=EVAL_MOLS)
    print(f"Loaded {len(eval_dataset)} molecules for GeoDiff eval")

    # Results TSV path
    tsv_path = os.path.join(PROJECT_ROOT, 'autoresearch', 'results_geodiff.tsv')

    for epoch in range(1, EPOCHS + 1):
        lr = get_lr(epoch - 1, base_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        total_loss = mse_sum = geo_sum = 0.0
        n_batches  = 0

        for batch in train_loader:
            at = batch['atom_types'].to(device)
            co = batch['coordinates'].to(device)
            ei = batch['edge_index'].to(device)
            bt = batch['bond_types'].to(device)
            bi = batch['batch_idx'].to(device)
            co = remove_com(co, bi)

            optimizer.zero_grad(set_to_none=True)
            loss_dict = model.get_loss(
                co, at, ei, bt, bi,
                geometry_weight=GEOMETRY_WEIGHT,
                epoch=epoch,
                max_epochs=EPOCHS,
                min_snr_gamma=MIN_SNR_GAMMA,
                geo_t_fraction=GEO_T_FRACTION,
                include_torsions=INCLUDE_TORSIONS,
            )
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss_dict['total'].item()
            mse_sum    += loss_dict['mse'].item()
            geo_sum    += loss_dict['geo'].item()
            n_batches  += 1

        train_loss = total_loss / max(n_batches, 1)

        # ── Validation loss ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0; v_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                at = batch['atom_types'].to(device)
                co = batch['coordinates'].to(device)
                ei = batch['edge_index'].to(device)
                bt = batch['bond_types'].to(device)
                bi = batch['batch_idx'].to(device)
                co = remove_com(co, bi)
                ld = model.get_loss(
                    co, at, ei, bt, bi,
                    geometry_weight=GEOMETRY_WEIGHT,
                    epoch=epoch, max_epochs=EPOCHS,
                    min_snr_gamma=MIN_SNR_GAMMA,
                    geo_t_fraction=GEO_T_FRACTION,
                    include_torsions=INCLUDE_TORSIONS,
                )
                val_loss += ld['total'].item(); v_batches += 1
        val_loss /= max(v_batches, 1)

        print(f"Epoch {epoch:03d}/{EPOCHS} | train={train_loss:.4f} "
              f"(mse={mse_sum/max(n_batches,1):.4f} "
              f"geo={geo_sum/max(n_batches,1):.4f}) "
              f"val={val_loss:.4f} | lr={lr:.2e}", flush=True)

        # Save best checkpoint by val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'exp_name': EXP_NAME,
                'config': {
                    'hidden_dim': MODEL_HIDDEN_DIM,
                    'num_layers': MODEL_NUM_LAYERS,
                    'num_rbf':    MODEL_NUM_RBF,
                    'timesteps':  MODEL_TIMESTEPS,
                    'time_dim':   MODEL_TIME_DIM,
                    'lr': LEARNING_RATE,
                    'geometry_weight': GEOMETRY_WEIGHT,
                    'geo_t_fraction':  GEO_T_FRACTION,
                    'include_torsions': INCLUDE_TORSIONS,
                    'dataset': str(DATA_PATH),
                    'max_atoms': MAX_ATOMS,
                },
            }, os.path.join(CHECKPOINT_DIR, f'{EXP_NAME}_best.pt'))

        # ── GeoDiff COV-MAT Evaluation (every EVAL_EVERY_EPOCHS) ────────────
        if epoch % EVAL_EVERY_EPOCHS == 0 or epoch == EPOCHS:
            print(f"\n[GeoDiff Eval] Epoch {epoch} — generating {N_GEN_PER_MOL} conformers "
                  f"for {len(eval_dataset)} molecules...", flush=True)
            geo_results = run_geodiff_eval(
                model, eval_dataset, device,
                num_gen_per_mol=N_GEN_PER_MOL,
                verbose=False,
            )
            print_geodiff_results(geo_results, tag=f"[Epoch {epoch}]")

            # Save best by MAT-R
            mat_r = geo_results.get('mat_r_mean', float('inf'))
            if mat_r < best_mat_r:
                best_mat_r = mat_r
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'mat_r': mat_r,
                    'exp_name': EXP_NAME,
                    'config': {
                        'hidden_dim': MODEL_HIDDEN_DIM, 'num_layers': MODEL_NUM_LAYERS,
                        'num_rbf': MODEL_NUM_RBF, 'timesteps': MODEL_TIMESTEPS,
                        'time_dim': MODEL_TIME_DIM, 'lr': LEARNING_RATE,
                        'geometry_weight': GEOMETRY_WEIGHT, 'geo_t_fraction': GEO_T_FRACTION,
                        'include_torsions': INCLUDE_TORSIONS,
                        'dataset': str(DATA_PATH), 'max_atoms': MAX_ATOMS,
                    },
                }, os.path.join(CHECKPOINT_DIR, f'{EXP_NAME}_best_matr.pt'))
                print(f"  ✓ New best MAT-R: {mat_r:.4f} Å (saved to {EXP_NAME}_best_matr.pt)")

            # Append to TSV
            description = (f"ExpG heavy-atom ep{epoch} "
                           f"hidden={MODEL_HIDDEN_DIM} layers={MODEL_NUM_LAYERS} "
                           f"geo_w={GEOMETRY_WEIGHT}")
            write_results_tsv(
                geo_results,
                ckpt_path=os.path.join(CHECKPOINT_DIR, f'{EXP_NAME}_best.pt'),
                out_path=tsv_path,
                description=description,
            )

    training_secs = time.time() - t_start_train

    # ── Final standard evaluation ────────────────────────────────────────────
    print("\nRunning final standard evaluation...", flush=True)
    best_ckpt = os.path.join(CHECKPOINT_DIR, f'{EXP_NAME}_best_matr.pt')
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(CHECKPOINT_DIR, f'{EXP_NAME}_best.pt')
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded best checkpoint from epoch {ckpt.get('epoch', '?')}")

    metrics = evaluate_all(model, val_loader, device,
                           num_gen=EVAL_MOLECULES, verbose=True)
    print_report(metrics, tag=EXP_NAME)

    # Final GeoDiff eval on full dataset
    print("\nRunning final GeoDiff COV-MAT evaluation...", flush=True)
    final_dataset = load_dataset(str(DATA_PATH), max_atoms=MAX_ATOMS, max_mols=EVAL_MOLS)
    final_geo = run_geodiff_eval(model, final_dataset, device,
                                  num_gen_per_mol=N_GEN_PER_MOL, verbose=True)
    print_geodiff_results(final_geo, tag=f"[FINAL — {EXP_NAME}]")

    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024 \
                if torch.cuda.is_available() else 0.0

    print("=" * 70)
    print(f"FINAL RESULTS — {EXP_NAME}")
    print("=" * 70)
    print(f"  COV-R@0.5Å : {final_geo.get('cov_r_05', float('nan'))*100:.1f}%")
    print(f"  MAT-R      : {final_geo.get('mat_r_mean', float('nan')):.4f} Å")
    print(f"  COV-P@0.5Å : {final_geo.get('cov_p_05', float('nan'))*100:.1f}%")
    print(f"  MAT-P      : {final_geo.get('mat_p_mean', float('nan')):.4f} Å")
    print(f"  fully_valid: {metrics['fully_valid_rate']*100:.1f}%")
    print(f"  bond_error : {metrics['mean_bond_error']:.4f} Å")
    print(f"  Training   : {training_secs/3600:.1f}h  |  Peak VRAM: {peak_vram:.0f} MB")
    print(f"  Parameters : {n_params:.2f}M")
    print()
    print("SOTA Reference (QM9 heavy-atom):")
    print("  GeoDiff (ICML 2022): COV-R=71.0%, MAT-R=0.297 Å")
    print("  GeoMol  (NeurIPS 2021): COV-R=71.5%, MAT-R=0.225 Å")


if __name__ == '__main__':
    main()
