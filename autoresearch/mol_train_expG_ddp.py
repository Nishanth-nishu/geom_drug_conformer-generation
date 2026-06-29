"""
mol_train_expG_ddp.py — Experiment G: SOTA Heavy-Atom, 2-GPU DDP + W&B Monitoring

Architecture:
  ConformerDiffusion  hidden=384, layers=8, rbf=32  (8.28M params)
  Heavy-atom-only QM9 (max 9 atoms)  — matches SOTA benchmark (GeoDiff/GeoMol)
  500 epochs, cosine LR + warmup, geometry loss (bond + repulsion + angle)
  GeoDiff COV-MAT evaluation every 50 epochs → COV-R & MAT-R metrics

DDP setup:
  Launched via: torchrun --nproc_per_node=2 autoresearch/mol_train_expG_ddp.py
  Per-GPU batch = 128, effective total batch = 256 (2 GPUs × 128)
  AMP (fp16) for max GPU throughput

W&B logging (main rank only):
  Project: mol_gen | Run: exp_G_heavy_atom_sota_ddp
  Logs every epoch: train_loss, val_loss, train_mse, train_geo, lr
  Logs every 50 epochs: COV-R, MAT-R, COV-P, MAT-P (GeoDiff metrics)

Key research references:
  Hoogeboom et al. EDM, ICML 2022      — x_0 parameterisation, CoM removal
  Xu et al. GeoDiff, ICML 2022        — COV-MAT metric, heavy-atom QM9
  Ganea et al. GeoMol, NeurIPS 2021   — geometry constraints
  Morehead & Cheng GCDM, NeurIPS 2023 — t-gated geometry loss
"""

import os
import sys
import time
import math
import json
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler

# W&B — imported early so rank-0 can init before training
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# AMP context — works with both PyTorch >= 2.0 and < 2.0
try:
    from torch.amp import autocast as torch_autocast
    def _amp_ctx(device_type='cuda'):
        return torch_autocast(device_type=device_type, dtype=torch.float16)
except ImportError:
    from torch.cuda.amp import autocast as _cuda_ac
    def _amp_ctx(device_type='cuda'):
        return _cuda_ac(dtype=torch.float16) if device_type == 'cuda' else nullcontext()

# ─── project root on sys.path ────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ─── force heavy-atom dataset BEFORE importing mol_prepare ───────────────────
_heavy_path = os.path.join(PROJECT_ROOT, 'data', 'qm9_heavy.jsonl')
if os.path.exists(_heavy_path):
    os.environ['MOL_DATASET']   = _heavy_path
    os.environ['MOL_MAX_ATOMS'] = '9'
else:
    raise FileNotFoundError(
        f"Heavy-atom dataset not found: {_heavy_path}\n"
        "Run: python data/prepare_qm9_heavy.py"
    )

from autoresearch.mol_prepare import (
    make_dataloaders, evaluate_all, print_report,
    CHECKPOINT_DIR, DATA_PATH, MAX_ATOMS,
)
from models.conformer_diffusion import ConformerDiffusion, remove_com
from autoresearch.geodiff_eval import (
    load_dataset, run_geodiff_eval, print_geodiff_results, write_results_tsv,
)

# =============================================================================
# HYPERPARAMETERS — Experiment G SOTA
# =============================================================================

# Architecture
MODEL_HIDDEN_DIM  = 384
MODEL_NUM_LAYERS  = 8
MODEL_TIMESTEPS   = 1000
MODEL_TIME_DIM    = 256
MODEL_NUM_RBF     = 32

# Training (per-GPU; effective total = PER_GPU_BATCH * world_size)
PER_GPU_BATCH     = 128          # → 256 total effective batch with 2 GPUs
LEARNING_RATE     = 5e-4
WEIGHT_DECAY      = 0.01
EPOCHS            = 500
WARMUP_EPOCHS     = 10
MIN_SNR_GAMMA     = 5.0

# Geometry loss
GEOMETRY_WEIGHT   = 0.5
GEO_T_FRACTION    = 0.3
INCLUDE_TORSIONS  = False        # torsion loop disabled — too slow for batch training

# Evaluation
EVAL_EVERY        = 50           # GeoDiff eval every 50 epochs
N_GEN_PER_MOL     = 10
EVAL_MOLS         = 300

EXP_NAME          = "exp_G_heavy_atom_sota_ddp"

# W&B
WANDB_PROJECT     = "mol_gen"
WANDB_API_KEY     = "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"


# =============================================================================
# HELPERS
# =============================================================================

def get_lr(epoch: int, base_lr: float) -> float:
    """Cosine LR with linear warmup + 1% floor."""
    if epoch < WARMUP_EPOCHS:
        return base_lr * (epoch + 1) / max(WARMUP_EPOCHS, 1)
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return max(base_lr * 0.5 * (1.0 + math.cos(math.pi * progress)),
               base_lr * 0.01)


def setup_ddp():
    """Initialise NCCL process group from SLURM/torchrun env vars."""
    dist.init_process_group(backend='nccl')
    local_rank  = int(os.environ.get('LOCAL_RANK', 0))
    world_size  = dist.get_world_size()
    device      = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    return local_rank, world_size, device


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ── DDP init ─────────────────────────────────────────────────────────────
    local_rank, world_size, device = setup_ddp()
    is_main = (local_rank == 0)

    # ── W&B init (rank-0 only) ────────────────────────────────────────────────
    run = None
    if is_main and HAS_WANDB:
        os.environ['WANDB_API_KEY'] = WANDB_API_KEY
        run = wandb.init(
            project=WANDB_PROJECT,
            name=EXP_NAME,
            config={
                'exp_name':        EXP_NAME,
                'hidden_dim':      MODEL_HIDDEN_DIM,
                'num_layers':      MODEL_NUM_LAYERS,
                'timesteps':       MODEL_TIMESTEPS,
                'time_dim':        MODEL_TIME_DIM,
                'num_rbf':         MODEL_NUM_RBF,
                'per_gpu_batch':   PER_GPU_BATCH,
                'effective_batch': PER_GPU_BATCH * world_size,
                'lr':              LEARNING_RATE,
                'weight_decay':    WEIGHT_DECAY,
                'epochs':          EPOCHS,
                'warmup_epochs':   WARMUP_EPOCHS,
                'geometry_weight': GEOMETRY_WEIGHT,
                'geo_t_fraction':  GEO_T_FRACTION,
                'include_torsions': INCLUDE_TORSIONS,
                'world_size':      world_size,
                'dataset':         str(DATA_PATH),
                'max_atoms':       MAX_ATOMS,
            },
            resume='allow',
        )
        print(f"[W&B] Run: {run.url}", flush=True)

    if is_main:
        print(f"\n{'='*65}")
        print(f"  Experiment G — SOTA DDP Training")
        print(f"  GPUs: {world_size}  |  Per-GPU batch: {PER_GPU_BATCH}  |  Effective batch: {PER_GPU_BATCH*world_size}")
        print(f"  Model: ConformerDiffusion hidden={MODEL_HIDDEN_DIM} layers={MODEL_NUM_LAYERS}")
        print(f"  Dataset: {DATA_PATH} (max_atoms={MAX_ATOMS})")
        print(f"  Epochs: {EPOCHS} | LR: {LEARNING_RATE} | geo_w: {GEOMETRY_WEIGHT}")
        print(f"{'='*65}", flush=True)

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_loader_base, val_loader = make_dataloaders(
        batch_size=PER_GPU_BATCH, num_workers=0
    )
    train_sampler = DistributedSampler(
        train_loader_base.dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        drop_last=True,
    )
    train_loader = TorchDataLoader(
        train_loader_base.dataset,
        batch_size=PER_GPU_BATCH,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=train_loader_base.collate_fn,
        pin_memory=True,
    )
    if is_main:
        print(f"  Train batches/GPU: {len(train_loader)} | Val batches: {len(val_loader)}", flush=True)

    # ── Model ────────────────────────────────────────────────────────────────
    raw_model = ConformerDiffusion(
        num_timesteps=MODEL_TIMESTEPS,
        hidden_dim=MODEL_HIDDEN_DIM,
        num_layers=MODEL_NUM_LAYERS,
        num_rbf=MODEL_NUM_RBF,
        time_dim=MODEL_TIME_DIM,
    ).to(device)

    n_params = sum(p.numel() for p in raw_model.parameters()) / 1e6
    if is_main:
        print(f"  Parameters: {n_params:.2f}M", flush=True)

    model = DDP(raw_model, device_ids=[local_rank], find_unused_parameters=False)

    # ── Optimizer + AMP ──────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        raw_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )
    scaler  = GradScaler()
    amp_ctx = _amp_ctx('cuda')

    # ── Pre-load GeoDiff eval set (rank-0 only) ───────────────────────────────
    eval_dataset = None
    if is_main:
        eval_dataset = load_dataset(str(DATA_PATH), max_atoms=MAX_ATOMS, max_mols=EVAL_MOLS)
        print(f"  GeoDiff eval set: {len(eval_dataset)} molecules", flush=True)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    tsv_path      = os.path.join(PROJECT_ROOT, 'autoresearch', 'results_geodiff.tsv')
    best_val_loss = float('inf')
    best_mat_r    = float('inf')
    t_train_start = time.time()

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        train_sampler.set_epoch(epoch)   # unique shuffle per epoch + rank

        lr = get_lr(epoch - 1, LEARNING_RATE)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = mse_sum = geo_sum = 0.0
        n_batches  = 0

        for batch in train_loader:
            at = batch['atom_types'].to(device, non_blocking=True)
            co = remove_com(batch['coordinates'].to(device, non_blocking=True),
                            batch['batch_idx'].to(device, non_blocking=True))
            ei = batch['edge_index'].to(device, non_blocking=True)
            bt = batch['bond_types'].to(device, non_blocking=True)
            bi = batch['batch_idx'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                loss_dict = raw_model.get_loss(
                    co, at, ei, bt, bi,
                    geometry_weight=GEOMETRY_WEIGHT,
                    epoch=epoch, max_epochs=EPOCHS,
                    min_snr_gamma=MIN_SNR_GAMMA,
                    geo_t_fraction=GEO_T_FRACTION,
                    include_torsions=INCLUDE_TORSIONS,
                )

            scaler.scale(loss_dict['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss_dict['total'].item()
            mse_sum    += loss_dict['mse'].item()
            geo_sum    += loss_dict['geo'].item()
            n_batches  += 1

        train_loss = total_loss / max(n_batches, 1)
        train_mse  = mse_sum   / max(n_batches, 1)
        train_geo  = geo_sum   / max(n_batches, 1)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        v_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                at = batch['atom_types'].to(device, non_blocking=True)
                co = remove_com(batch['coordinates'].to(device, non_blocking=True),
                                batch['batch_idx'].to(device, non_blocking=True))
                ei = batch['edge_index'].to(device, non_blocking=True)
                bt = batch['bond_types'].to(device, non_blocking=True)
                bi = batch['batch_idx'].to(device, non_blocking=True)
                with amp_ctx:
                    ld = raw_model.get_loss(
                        co, at, ei, bt, bi,
                        geometry_weight=GEOMETRY_WEIGHT,
                        epoch=epoch, max_epochs=EPOCHS,
                        min_snr_gamma=MIN_SNR_GAMMA,
                        geo_t_fraction=GEO_T_FRACTION,
                        include_torsions=INCLUDE_TORSIONS,
                    )
                val_loss += ld['total'].item()
                v_batches += 1
        val_loss /= max(v_batches, 1)

        # ── Rank-0: log + checkpoint ──────────────────────────────────────────
        if is_main:
            print(f"Epoch {epoch:03d}/{EPOCHS} | "
                  f"train={train_loss:.4f} (mse={train_mse:.4f} geo={train_geo:.4f}) "
                  f"val={val_loss:.4f} | lr={lr:.2e}", flush=True)

            # W&B epoch log
            log_dict = {
                'epoch':      epoch,
                'train/loss': train_loss,
                'train/mse':  train_mse,
                'train/geo':  train_geo,
                'val/loss':   val_loss,
                'lr':         lr,
            }

            # Save best by val loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'exp_name': EXP_NAME,
                    'config': {
                        'hidden_dim': MODEL_HIDDEN_DIM, 'num_layers': MODEL_NUM_LAYERS,
                        'num_rbf': MODEL_NUM_RBF, 'timesteps': MODEL_TIMESTEPS,
                        'time_dim': MODEL_TIME_DIM, 'lr': LEARNING_RATE,
                        'geometry_weight': GEOMETRY_WEIGHT,
                        'geo_t_fraction': GEO_T_FRACTION,
                        'include_torsions': INCLUDE_TORSIONS,
                        'dataset': str(DATA_PATH), 'max_atoms': MAX_ATOMS,
                        'world_size': world_size,
                    },
                }
                torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f'{EXP_NAME}_best.pt'))

            # ── GeoDiff COV-MAT Evaluation ────────────────────────────────────
            if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
                print(f"\n[GeoDiff Eval @ Epoch {epoch}] "
                      f"Generating {N_GEN_PER_MOL} conformers × {len(eval_dataset)} mols...",
                      flush=True)
                geo_results = run_geodiff_eval(
                    raw_model, eval_dataset, device,
                    num_gen_per_mol=N_GEN_PER_MOL, verbose=False,
                )
                print_geodiff_results(geo_results, tag=f"[Epoch {epoch}]")

                cov_r  = geo_results.get('cov_r_05', float('nan'))
                mat_r  = geo_results.get('mat_r_mean', float('inf'))
                cov_p  = geo_results.get('cov_p_05', float('nan'))
                mat_p  = geo_results.get('mat_p_mean', float('nan'))

                log_dict.update({
                    'geodiff/cov_r@0.5A': cov_r,
                    'geodiff/mat_r':      mat_r,
                    'geodiff/cov_p@0.5A': cov_p,
                    'geodiff/mat_p':      mat_p,
                })

                # Save best by MAT-R
                if mat_r < best_mat_r:
                    best_mat_r = mat_r
                    torch.save({
                        'epoch': epoch, 'model_state_dict': raw_model.state_dict(),
                        'mat_r': mat_r, 'exp_name': EXP_NAME,
                        'config': ckpt['config'],
                    }, os.path.join(CHECKPOINT_DIR, f'{EXP_NAME}_best_matr.pt'))
                    print(f"  ✓ New best MAT-R: {mat_r:.4f} Å", flush=True)

                # Append to TSV
                write_results_tsv(
                    geo_results,
                    ckpt_path=os.path.join(CHECKPOINT_DIR, f'{EXP_NAME}_best.pt'),
                    out_path=tsv_path,
                    description=(f"ExpG-DDP ep{epoch} hidden={MODEL_HIDDEN_DIM} "
                                 f"layers={MODEL_NUM_LAYERS} geo_w={GEOMETRY_WEIGHT} "
                                 f"gpus={world_size}"),
                )

            # W&B step log
            if run is not None:
                wandb.log(log_dict, step=epoch)

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL REPORT (rank-0)
    # ─────────────────────────────────────────────────────────────────────────
    training_secs = time.time() - t_train_start

    if is_main:
        # Load best checkpoint
        best_ckpt = os.path.join(CHECKPOINT_DIR, f'{EXP_NAME}_best_matr.pt')
        if not os.path.exists(best_ckpt):
            best_ckpt = os.path.join(CHECKPOINT_DIR, f'{EXP_NAME}_best.pt')
        if os.path.exists(best_ckpt):
            ck = torch.load(best_ckpt, map_location=device, weights_only=False)
            raw_model.load_state_dict(ck['model_state_dict'])
            print(f"\nLoaded best checkpoint (epoch {ck.get('epoch','?')})")

        print("\nRunning final GeoDiff COV-MAT evaluation...", flush=True)
        final_geo = run_geodiff_eval(raw_model, eval_dataset, device,
                                     num_gen_per_mol=N_GEN_PER_MOL, verbose=True)
        print_geodiff_results(final_geo, tag=f"[FINAL — {EXP_NAME}]")

        peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024

        print("\n" + "="*65)
        print(f"FINAL RESULTS — {EXP_NAME}")
        print("="*65)
        print(f"  COV-R@0.5Å : {final_geo.get('cov_r_05', float('nan'))*100:.1f}%  "
              f"(SOTA GeoDiff: 71.0%)")
        print(f"  MAT-R      : {final_geo.get('mat_r_mean', float('nan')):.4f} Å  "
              f"(SOTA GeoDiff: 0.297 Å)")
        print(f"  COV-P@0.5Å : {final_geo.get('cov_p_05', float('nan'))*100:.1f}%")
        print(f"  MAT-P      : {final_geo.get('mat_p_mean', float('nan')):.4f} Å")
        print(f"  Training   : {training_secs/3600:.1f}h | Peak VRAM/GPU: {peak_vram:.0f} MB")
        print(f"  Parameters : {n_params:.2f}M | World size: {world_size}")

        # Final W&B summary
        if run is not None:
            wandb.summary.update({
                'final/cov_r@0.5A':  final_geo.get('cov_r_05', float('nan')),
                'final/mat_r':       final_geo.get('mat_r_mean', float('nan')),
                'final/cov_p@0.5A':  final_geo.get('cov_p_05', float('nan')),
                'final/mat_p':       final_geo.get('mat_p_mean', float('nan')),
                'final/best_mat_r':  best_mat_r,
                'training_hours':    training_secs / 3600,
                'peak_vram_mb':      peak_vram,
                'n_params_M':        n_params,
            })
            wandb.finish()

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
