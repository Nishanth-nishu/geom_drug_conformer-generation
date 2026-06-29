"""
mol_train_expB.py — Experiment B: EQGAT-diff Attention-Enhanced EGNN

Research hypothesis:
  Standard EGNN uses unweighted sum-pooling over all neighbour messages.
  EQGAT-diff (Le et al. ICLR 2024) adds multi-head dot-product attention
  over the incoming messages, letting each atom selectively attend to its
  most informative neighbours.

  Expected benefit:
  - Chiral centres can distinguish all 4 neighbours (critical for R/S)
  - Aromatic systems couple across the pi-system (planarity)
  - Torsion angles around sp2-sp3 bonds are better resolved
  - Result: ~10% lower MAT-R vs. baseline

Key differences from Experiment A (baseline):
  - Model class: AttnConformerDiffusion (attention EGNN)
  - num_heads=4 (4-head attention per layer)
  - dropout=0.1 (applied to attention weights)
  - Slightly larger: ~3.1M params vs ~2.8M (attention projections added)

All other hyperparameters, optimizer, and harness identical to Exp A.

Reference:
  Le et al. "EQGAT-diff: a novel equivariant graph attention model for
  molecular 3D generation" ICLR 2024. arXiv:2306.01916.
"""

import os, sys, time, math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoresearch.mol_prepare import (
    EPOCH_BUDGET, EVAL_MOLECULES, make_dataloaders,
    evaluate_all, print_report, CHECKPOINT_DIR,
)
from models.conformer_diffusion import remove_com
from models.attn_conformer_diffusion import AttnConformerDiffusion

# ============================================================================
# HYPERPARAMETERS — Experiment B
# ============================================================================

MODEL_HIDDEN_DIM = 256
MODEL_NUM_LAYERS = 6
MODEL_TIMESTEPS  = 1000
MODEL_TIME_DIM   = 128
MODEL_NUM_HEADS  = 4          # NEW: attention heads per layer
MODEL_DROPOUT    = 0.1        # NEW: applied to attention weights

BATCH_SIZE       = 64
LEARNING_RATE    = 1e-4
WEIGHT_DECAY     = 0.01
OPTIMIZER        = "adamw"

EPOCHS           = EPOCH_BUDGET
GEOMETRY_WEIGHT  = 0.1
WARMUP_EPOCHS    = 5
MIN_SNR_GAMMA    = 5.0
DDIM_STEPS       = 50

SAVE_BEST        = True
EXP_NAME         = "exp_B_attention_egnn"

# ============================================================================
# OPTIMIZER (same LR-fix as Exp A)
# ============================================================================

def build_optimizer(model):
    print(f"Optimizer: AdamW | lr={LEARNING_RATE}, wd={WEIGHT_DECAY}")
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    for pg in opt.param_groups:
        pg['initial_lr'] = pg['lr']
    return opt


def get_lr(epoch, base_lr):
    """Cosine LR schedule with warmup and floor (FIX-AUDIT-5)."""
    if epoch < WARMUP_EPOCHS:
        return base_lr * (epoch + 1) / max(WARMUP_EPOCHS, 1)
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    cosine_lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    return max(cosine_lr, base_lr * 0.01)  # floor at 1% of peak

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    t_start = time.time()
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Experiment: {EXP_NAME}")
    print(f"Config: hidden={MODEL_HIDDEN_DIM} layers={MODEL_NUM_LAYERS} "
          f"heads={MODEL_NUM_HEADS} dropout={MODEL_DROPOUT} "
          f"epochs={EPOCHS} opt={OPTIMIZER} geo_w={GEOMETRY_WEIGHT}")

    train_loader, val_loader = make_dataloaders(batch_size=BATCH_SIZE)

    # Attention-enhanced model (drop-in for ConformerDiffusion)
    model = AttnConformerDiffusion(
        num_timesteps=MODEL_TIMESTEPS,
        hidden_dim=MODEL_HIDDEN_DIM,
        num_layers=MODEL_NUM_LAYERS,
        time_dim=MODEL_TIME_DIM,
        num_heads=MODEL_NUM_HEADS,
        dropout=MODEL_DROPOUT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.2f}M")

    optimizer = build_optimizer(model)
    base_lr   = LEARNING_RATE
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float('inf')
    t_start_train = time.time()

    for epoch in range(1, EPOCHS + 1):
        lr = get_lr(epoch - 1, base_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.train()
        total_loss = mse_sum = geo_sum = 0.0
        n_batches = 0
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
                epoch=epoch, max_epochs=EPOCHS,
                min_snr_gamma=MIN_SNR_GAMMA,
            )
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss_dict['total'].item()
            mse_sum    += loss_dict['mse'].item()
            geo_sum    += loss_dict['geo'].item()
            n_batches  += 1

        train_loss = total_loss / max(n_batches, 1)

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
                ld = model.get_loss(co, at, ei, bt, bi,
                                    geometry_weight=GEOMETRY_WEIGHT,
                                    epoch=epoch, max_epochs=EPOCHS,
                                    min_snr_gamma=MIN_SNR_GAMMA)
                val_loss += ld['total'].item(); v_batches += 1
        val_loss /= max(v_batches, 1)

        print(f"Epoch {epoch:03d}/{EPOCHS} | train={train_loss:.4f} "
              f"(mse={mse_sum/max(n_batches,1):.4f} geo={geo_sum/max(n_batches,1):.4f}) "
              f"val={val_loss:.4f} | lr={lr:.2e}", flush=True)

        if SAVE_BEST and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_loss': val_loss, 'exp_name': EXP_NAME,
                'config': {
                    'hidden_dim': MODEL_HIDDEN_DIM, 'num_layers': MODEL_NUM_LAYERS,
                    'num_heads': MODEL_NUM_HEADS, 'dropout': MODEL_DROPOUT,
                    'timesteps': MODEL_TIMESTEPS, 'geometry_weight': GEOMETRY_WEIGHT,
                },
            }, os.path.join(CHECKPOINT_DIR, f'{EXP_NAME}_best.pt'))

    training_secs = time.time() - t_start_train

    print("\nRunning final evaluation...", flush=True)
    best_ckpt = os.path.join(CHECKPOINT_DIR, f'{EXP_NAME}_best.pt')
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    metrics = evaluate_all(model, val_loader, device,
                           num_gen=EVAL_MOLECULES, verbose=True)
    print_report(metrics, tag=EXP_NAME)
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024 \
                if torch.cuda.is_available() else 0.0

    print("---")
    print(f"fully_valid:    {metrics['fully_valid_rate']:.6f}")
    print(f"mat_r:          {metrics['mat_r']:.6f}")
    print(f"rmsd_mean:      {metrics['rmsd_mean']:.6f}")
    print(f"strain_kcal:    {metrics['mean_strain_kcal']:.6f}")
    print(f"cov_r:          {metrics['cov_r']:.6f}")
    print(f"validity:       {metrics['validity']:.6f}")
    print(f"bond_error:     {metrics['mean_bond_error']:.6f}")
    print(f"training_secs:  {training_secs:.1f}")
    print(f"total_secs:     {time.time()-t_start:.1f}")
    print(f"peak_vram_mb:   {peak_vram:.1f}")
    print(f"num_epochs:     {EPOCHS}")
    print(f"num_params_M:   {n_params:.2f}")
    print(f"optimizer:      {OPTIMIZER}")
    print(f"geometry_weight:{GEOMETRY_WEIGHT}")
    print(f"exp_name:       {EXP_NAME}")


if __name__ == '__main__':
    main()
