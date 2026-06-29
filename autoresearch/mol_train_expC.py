"""
mol_train_expC.py — Experiment C: Conditional Flow Matching (CFM)

Research hypothesis: CFM (Lipman et al. ICLR 2023) learns straight-line ODE paths,
requiring only 20 NFE vs DDPM's 1000 (or 50 DDIM). For molecular conformers:
  - 4-5x faster inference at equal quality
  - No noise schedule to tune
  - Better x0_hat estimates from linear trajectory

Reference: Lipman et al. "Flow Matching for Generative Modeling" ICLR 2023.
"""

import os, sys, time, math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoresearch.mol_prepare import (
    EPOCH_BUDGET, EVAL_MOLECULES, make_dataloaders,
    evaluate_all, print_report, CHECKPOINT_DIR,
)
from models.conformer_diffusion import remove_com
from models.flow_matching import FlowMatchingConformer

# ============================================================================
# HYPERPARAMETERS — Experiment C
# ============================================================================

MODEL_HIDDEN_DIM = 256
MODEL_NUM_LAYERS = 6
MODEL_TIME_DIM   = 128

BATCH_SIZE       = 64
LEARNING_RATE    = 1e-4
WEIGHT_DECAY     = 0.01
OPTIMIZER        = "adamw"

EPOCHS           = EPOCH_BUDGET
# CFM Fix: Remove geometry loss from training.
# At t≈1 (high noise), x0_hat = x_t - t*v_pred is pure random noise, giving
# chaotic geometry gradients that destabilize the velocity network.
# This explains the 28k-67k kcal/mol strain energy in Exp C results.
# Reference: Yim et al. (FrameDiff, ICML 2023) applies geo constraints only
# as post-sampling refinement, NOT during CFM training.
GEOMETRY_WEIGHT  = 0.0       # FIXED: was 0.05, causing training instability
WARMUP_EPOCHS    = 5
ODE_STEPS        = 20         # CFM only needs 20 steps (Lipman 2023) — was 100

SAVE_BEST        = True
EXP_NAME         = "exp_C_flow_matching"


def build_optimizer(model):
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


def main():
    t_start = time.time()
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Experiment: {EXP_NAME}")
    print(f"Config: hidden={MODEL_HIDDEN_DIM} layers={MODEL_NUM_LAYERS} "
          f"epochs={EPOCHS} geo_w={GEOMETRY_WEIGHT} ode_steps={ODE_STEPS}")

    train_loader, val_loader = make_dataloaders(batch_size=BATCH_SIZE)

    model = FlowMatchingConformer(
        hidden_dim=MODEL_HIDDEN_DIM,
        num_layers=MODEL_NUM_LAYERS,
        time_dim=MODEL_TIME_DIM,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.2f}M")

    optimizer = build_optimizer(model)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float('inf')
    t_start_train = time.time()

    for epoch in range(1, EPOCHS + 1):
        lr = get_lr(epoch - 1, LEARNING_RATE)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.train()
        total_loss = mse_sum = geo_sum = 0.0
        n_batches = 0
        for batch in train_loader:
            at = batch['atom_types'].to(device)
            co = remove_com(batch['coordinates'].to(device), batch['batch_idx'].to(device))
            ei = batch['edge_index'].to(device)
            bt = batch['bond_types'].to(device)
            bi = batch['batch_idx'].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss_dict = model.get_loss(co, at, ei, bt, bi,
                                       geometry_weight=GEOMETRY_WEIGHT)
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss_dict['total'].item()
            mse_sum    += loss_dict['mse'].item()
            geo_v = loss_dict['geo']
            geo_sum += geo_v.item() if isinstance(geo_v, torch.Tensor) else float(geo_v)
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_loss = 0.0; v_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                at = batch['atom_types'].to(device)
                co = remove_com(batch['coordinates'].to(device), batch['batch_idx'].to(device))
                ei = batch['edge_index'].to(device)
                bt = batch['bond_types'].to(device)
                bi = batch['batch_idx'].to(device)
                ld = model.get_loss(co, at, ei, bt, bi, geometry_weight=GEOMETRY_WEIGHT)
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
            }, os.path.join(CHECKPOINT_DIR, f'{EXP_NAME}_best.pt'))

    training_secs = time.time() - t_start_train

    print("\nRunning final evaluation (ODE sampler, 20 steps)...", flush=True)
    best_ckpt = os.path.join(CHECKPOINT_DIR, f'{EXP_NAME}_best.pt')
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    # FlowMatchingConformer.ddim_sample() → ode_sample(num_steps=20)
    import functools
    model.ddim_sample = functools.partial(model.ode_sample, num_steps=ODE_STEPS)

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
    print(f"ode_steps:      {ODE_STEPS}")
    print(f"exp_name:       {EXP_NAME}")


if __name__ == '__main__':
    main()
