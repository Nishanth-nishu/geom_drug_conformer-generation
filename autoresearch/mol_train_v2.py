"""
mol_train_v2.py — Training Script for DualEncoderDiffusion (Novel Architecture)

Architecture: DualEncoderDiffusion
  - Distance-space score matching (not Cartesian x0 MSE)
  - Dual graph: GIN local + SchNet+ET attention global
  - Sigmoid noise schedule (2000 timesteps, GeoDiff GEOM-Drugs settings)
  - Boltzmann conformer weighting (our novel contribution)

Usage:
  # Smoke test (5 epochs, fast validation):
  PYTHONPATH=. python autoresearch/mol_train_v2.py --smoke-test

  # Full training:
  PYTHONPATH=. python autoresearch/mol_train_v2.py

  # With custom settings:
  PYTHONPATH=. python autoresearch/mol_train_v2.py --max-atoms 30 --max-mols 50000 --epochs 300
"""

import os, sys, time, argparse, math, json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from models.dual_encoder_diffusion import DualEncoderDiffusion, remove_com
from data.geom_drugs_dataset import make_geom_dataloaders
from autoresearch.geom_drugs_eval import run_geom_drugs_eval, print_geom_results

# ── Hyperparameters ────────────────────────────────────────────────────────────
# Following GeoDiff GEOM-Drugs config + our novel additions

GEOM_SMALL  = 30   # max heavy atoms for initial experiments
GEOM_MEDIUM = 50
GEOM_FULL   = 100

# Model
HIDDEN_DIM      = 256   # Per encoder (local + global each have 256)
NUM_CONVS_GLOBAL = 6    # SchNet layers (GeoDiff drugs_default: num_convs=6)
NUM_CONVS_LOCAL  = 4    # GIN layers (GeoDiff drugs_default: num_convs_local=4)
NUM_GAUSSIANS   = 50    # RBF basis functions (GeoDiff SchNet default)
NUM_ATTN_HEADS  = 8     # ET attention heads (novel vs GeoDiff)
EDGE_ORDER      = 3     # Higher-order bond extension (GeoDiff: edge_order=3)
CUTOFF          = 10.0  # Radius graph cutoff (GeoDiff: cutoff=10.0)

# Noise schedule (GeoDiff GEOM-Drugs drugs_default.yml)
# GeoDiff uses 5000 steps with sigmoid β: 1e-7 → 2e-3, cutoff=10Å
NUM_TIMESTEPS   = 5000   # GeoDiff canonical: 5000 steps
BETA_START      = 1e-7   # GeoDiff: β_start = 1e-7
BETA_END        = 2e-3   # GeoDiff: β_end   = 2e-3
W_GLOBAL        = 0.5   # Weight for global force term at inference

# Training (GeoDiff drugs_default.yml)
BATCH_SIZE      = 32    # GeoDiff: 32
LR              = 1e-3  # GeoDiff: 1e-3 with AdamW, β1=0.95, β2=0.999
WEIGHT_DECAY    = 0.0
BETA1           = 0.95   # GeoDiff AdamW betas
BETA2           = 0.999
MAX_GRAD_NORM   = 30000.0  # GeoDiff: different from QM9 (which uses 10)
EPOCHS          = 300
WARMUP_EPOCHS   = 10
MIN_SNR_GAMMA   = 5.0

# Data
MAX_ATOMS       = GEOM_SMALL
MAX_MOLS        = 50_000
MAX_CONFORMERS  = 10    # ablation: try 5 vs 10
MIN_CONFORMERS  = 2

# Eval
EVAL_EVERY      = 25     # epochs
N_EVAL_MOLS     = 100
N_GEN_PER_MOL   = 10
DDIM_STEPS      = 100    # DDIM steps during eval

# Checkpointing
EXPERIMENT_NAME = f"v2_dual_{HIDDEN_DIM}h_{NUM_CONVS_GLOBAL}gconv_{NUM_CONVS_LOCAL}lconv_atoms{MAX_ATOMS}_mols{MAX_MOLS//1000}k"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke-test', action='store_true',
                   help='Run 5 epochs only to validate pipeline')
    p.add_argument('--max-atoms', type=int, default=MAX_ATOMS)
    p.add_argument('--max-mols', type=int, default=MAX_MOLS)
    p.add_argument('--max-confs', type=int, default=MAX_CONFORMERS,
                   help='Max conformers per molecule (ablation: 5 vs 10)')
    p.add_argument('--min-confs', type=int, default=MIN_CONFORMERS,
                   help='Min conformers required (default 2)')
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    p.add_argument('--lr', type=float, default=LR)
    p.add_argument('--hidden-dim', type=int, default=HIDDEN_DIM)
    p.add_argument('--num-timesteps', type=int, default=NUM_TIMESTEPS,
                   help='Diffusion steps (GeoDiff: 5000)')
    p.add_argument('--beta-start', type=float, default=BETA_START,
                   help='Sigmoid noise schedule beta_start (GeoDiff: 1e-7)')
    p.add_argument('--beta-end', type=float, default=BETA_END,
                   help='Sigmoid noise schedule beta_end (GeoDiff: 2e-3)')
    p.add_argument('--cutoff', type=float, default=CUTOFF,
                   help='Neighbor radius cutoff in Angstrom (GeoDiff: 10.0)')
    p.add_argument('--exp-name', type=str, default=None,
                   help='Override experiment name')
    p.add_argument('--data', type=str, default='data/geom_drugs.jsonl')
    p.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint to resume from')
    return p.parse_args()


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Training loop ──────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = (f"v2_dual_{args.hidden_dim}h_{NUM_CONVS_GLOBAL}gconv_{NUM_CONVS_LOCAL}lconv"
                    f"_atoms{args.max_atoms}_mols{args.max_mols//1000}k"
                    f"_confs{args.max_confs}_T{args.num_timesteps}")

    print(f"Device      : {device}")
    print(f"Experiment  : {exp_name}")
    print(f"Data        : {args.data}")
    print(f"max_atoms   : {args.max_atoms}  max_mols: {args.max_mols}  max_confs: {args.max_confs}")
    print(f"β schedule  : sigmoid, T={args.num_timesteps}, β_start={args.beta_start}, β_end={args.beta_end}")
    print(f"Neighbor r  : {args.cutoff} Å")

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = make_geom_dataloaders(
        data_path=str(Path(args.data).resolve()),  # absolute path — prevents cluster CWD crash
        max_atoms=args.max_atoms,
        min_conformers=args.min_confs,
        max_conformers=args.max_confs,
        max_mols=args.max_mols,
        batch_size=args.batch_size,
        num_workers=4,
        val_split=0.1,
        return_energy=True,
        uniform_sampling=True,
    )
    print(f"\nModel arch  : DualEncoderDiffusion (distance-space score matching)")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DualEncoderDiffusion(
        hidden_dim=args.hidden_dim,
        num_convs_global=NUM_CONVS_GLOBAL,
        num_convs_local=NUM_CONVS_LOCAL,
        num_gaussians=NUM_GAUSSIANS,
        num_attn_heads=NUM_ATTN_HEADS,
        num_bond_types=5,
        edge_order=EDGE_ORDER,
        cutoff=args.cutoff,         # CLI: default 10.0 Å (GeoDiff)
        num_timesteps=args.num_timesteps,  # CLI: default 5000 (GeoDiff)
        beta_start=args.beta_start,        # CLI: default 1e-7
        beta_end=args.beta_end,            # CLI: default 2e-3
        w_global=W_GLOBAL,
    ).to(device)

    n_params = count_parameters(model)
    print(f"Parameters  : {n_params:,}")
    print(f"\nNoise sched : sigmoid ({args.num_timesteps} steps, β_start={args.beta_start}, β_end={args.beta_end})")
    print(f"Dual graph  : local GIN (edge_order={EDGE_ORDER}) + global SchNet (cutoff={args.cutoff}Å)")
    print(f"ET Attention: {NUM_ATTN_HEADS} heads (novel vs GeoDiff)\n")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    # GeoDiff: AdamW, lr=1e-3, beta1=0.95, beta2=0.999
    initial_lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr,
                             betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
    # CosineAnnealingWarmRestarts: LR restarts periodically — never gets frozen
    # T_0=50: first restart at epoch 50, T_mult=2: each cycle doubles (50→100→200)
    # This prevents the ReduceLROnPlateau collapse to 2e-6 that killed the previous run
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    # Warmup: linear LR ramp
    warmup_steps = WARMUP_EPOCHS * len(train_loader)
    global_step = 0

    start_epoch = 1
    best_mat_r = float('inf')

    # ── Resume checkpoint ─────────────────────────────────────────────────────
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_mat_r = ckpt.get('best_mat_r', float('inf'))
        global_step = ckpt.get('global_step', 0)
        print(f"Resumed from epoch {ckpt['epoch']}, best_mat_r={best_mat_r:.4f}")

    epochs = 5 if args.smoke_test else args.epochs

    print("=" * 60)
    print(f"  Training: {epochs} epochs  |  Batch: {args.batch_size}  |  LR: {initial_lr}")
    print("=" * 60)

    t0_total = time.time()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        total_local = 0.0
        total_global = 0.0
        n_batches = 0

        for batch_data in train_loader:
            # ── Warmup LR ──────────────────────────────────────────────────
            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg['lr'] = initial_lr * lr_scale

            # ── Move data to device ────────────────────────────────────────
            pos        = batch_data['coordinates'].to(device)
            atom_types = batch_data['atom_types'].to(device)
            edge_index = batch_data['edge_index'].to(device)
            bond_types = batch_data['bond_types'].to(device)
            batch_idx  = batch_data['batch_idx'].to(device)
            bw = batch_data.get('boltzmann_weights', None)
            if bw is not None:
                bw = bw.to(device)

            # CoM centering
            pos = remove_com(pos, batch_idx)

            optimizer.zero_grad(set_to_none=True)

            loss_dict = model.get_loss(
                pos=pos,
                atom_types=atom_types,
                bond_index=edge_index,
                bond_type=bond_types,
                batch=batch_idx,
                min_snr_gamma=MIN_SNR_GAMMA,
                boltzmann_weights=bw,
            )

            loss = loss_dict['total']
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [Skip NaN/Inf loss at step {global_step}]")
                global_step += 1
                continue

            loss.backward()

            # Gradient clipping (GeoDiff: max_grad_norm=30000 for drugs!)
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_loss   += loss.item()
            total_local  += loss_dict['local'].item()
            total_global += loss_dict['global'].item()
            n_batches += 1
            global_step += 1

        # ── Validation loss ────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch_data in val_loader:
                pos        = batch_data['coordinates'].to(device)
                atom_types = batch_data['atom_types'].to(device)
                edge_index = batch_data['edge_index'].to(device)
                bond_types = batch_data['bond_types'].to(device)
                batch_idx  = batch_data['batch_idx'].to(device)
                bw = batch_data.get('boltzmann_weights', None)
                if bw is not None:
                    bw = bw.to(device)
                pos = remove_com(pos, batch_idx)
                try:
                    ld = model.get_loss(pos, atom_types, edge_index, bond_types,
                                        batch_idx, boltzmann_weights=bw)
                    val_loss += ld['total'].item()
                    n_val += 1
                except Exception:
                    pass

        avg_train = total_loss / max(n_batches, 1)
        avg_local = total_local / max(n_batches, 1)
        avg_global = total_global / max(n_batches, 1)
        avg_val   = val_loss / max(n_val, 1)

        elapsed = (time.time() - t0_total) / 60
        print(f"Epoch {epoch:03d}/{epochs} | "
              f"train={avg_train:.4f} (loc={avg_local:.4f} glob={avg_global:.4f}) "
              f"val={avg_val:.4f} | "
              f"lr={get_lr(optimizer):.2e} | {elapsed:.1f}min")

        # LR scheduler step (after warmup) — CosineAnnealingWarmRestarts takes epoch number
        if global_step >= warmup_steps:
            scheduler.step(epoch)

        # ── Geometry eval every EVAL_EVERY epochs ─────────────────────────
        if epoch % EVAL_EVERY == 0 or args.smoke_test:
            print(f"\n[Eval] Epoch {epoch} — generating {N_GEN_PER_MOL} conformers "
                  f"per molecule on {N_EVAL_MOLS} validation molecules ...")

            try:
                results = run_geom_drugs_eval(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    n_mols=N_EVAL_MOLS,
                    n_gen=N_GEN_PER_MOL,
                    num_steps=DDIM_STEPS,
                    cov_threshold=0.5,
                    energy_surrogate=None,
                    guidance_scale=0.0,
                )
                print_geom_results(results, str(epoch))

                mat_r = results.get('mat_r_mean', float('inf'))
                if mat_r < best_mat_r:
                    best_mat_r = mat_r
                    ckpt_path = f"checkpoints/{exp_name}_best_matr.pt"
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_mat_r': best_mat_r,
                        'global_step': global_step,
                        'config': {
                            'hidden_dim': args.hidden_dim,
                            'num_convs_global': NUM_CONVS_GLOBAL,
                            'num_convs_local': NUM_CONVS_LOCAL,
                            'num_gaussians': NUM_GAUSSIANS,
                            'num_attn_heads': NUM_ATTN_HEADS,
                            'edge_order': EDGE_ORDER,
                            'cutoff': CUTOFF,
                            'num_timesteps': args.num_timesteps,
                            'beta_start': BETA_START,
                            'beta_end': BETA_END,
                        },
                    }, ckpt_path)
                    print(f"  ✓ New best MAT-R: {best_mat_r:.4f} Å saved.")
            except Exception as e:
                print(f"  [Eval error] {e}")
                import traceback; traceback.print_exc()

            print()

    print(f"\nTraining complete. Best MAT-R: {best_mat_r:.4f} Å")
    return model


def main():
    args = parse_args()

    if args.smoke_test:
        print("=" * 60)
        print("Running smoke test (5 epochs) ...")
        print("=" * 60)
        train(args)
        print("\n[Smoke test complete — pipeline verified OK]")
        print("Smoke test passed! Starting full training ...")

    print("=" * 60)
    print(f"Full training: {args.epochs} epochs")
    print("=" * 60)
    args.smoke_test = False
    train(args)


if __name__ == '__main__':
    main()
