"""
mol_train_expE.py — Experiment E: SOTA Hybrid (Flow Matching + Attention-EGNN)

Combines:
  - Conditional Flow Matching (Exp C) — straight ODE trajectories, 20 NFE
  - Attention-Enhanced EGNN backbone (Exp B) — EQGAT-diff style
  - Geometry-aware training at low timesteps only (inspired by GCDM)

Critical bug fixes vs. original ExpE:
  FIX-5: add remove_com(ct, bi) before get_loss in BOTH train AND validation loops.
          Missing this breaks E(3) equivariance: x0 is not CoM-free but eps is,
          so the interpolation x_t = (1-t)*x0 + t*eps has a non-zero CoM drift.
  FIX-6: weight_decay changed from 1e-12 to 0.01 (matching EDM/Exp A/B/D).
          With no regularization the model overfits training noise.
"""

import os, sys, time, math
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hybrid_conformer import HybridFlowMatchingConformer
from models.conformer_diffusion import remove_com
from autoresearch.mol_prepare import (
    EPOCH_BUDGET, EVAL_MOLECULES, make_dataloaders,
    evaluate_all, print_report, CHECKPOINT_DIR,
)

# Configuration for SOTA Hybrid
EXP_NAME         = "exp_E_sota_hybrid"
HIDDEN_DIM       = 256
NUM_LAYERS       = 6
NUM_HEADS        = 4

BATCH_SIZE       = 64
LEARNING_RATE    = 1e-4
WEIGHT_DECAY     = 0.01        # FIX-6: was 1e-12 (no regularization → overfitting)
OPTIMIZER        = "adamw"

EPOCHS           = EPOCH_BUDGET
GEOMETRY_WEIGHT  = 0.0         # CFM fix: geo loss at high t sends chaotic gradients
                               # (see ExpC analysis: explains 28k-67k strain energy)
INCLUDE_TORSIONS = False       # consistent with GEOMETRY_WEIGHT=0
WARMUP_EPOCHS    = 5
ODE_STEPS        = 20          # CFM needs only 20 steps (Lipman 2023)
SAVE_BEST        = True


def get_lr(epoch, base_lr):
    """Cosine LR schedule with warmup and floor (FIX-AUDIT-5)."""
    if epoch < WARMUP_EPOCHS:
        return base_lr * (epoch + 1) / max(WARMUP_EPOCHS, 1)
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    cosine_lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    return max(cosine_lr, base_lr * 0.01)  # floor at 1% of peak LR (FIX-AUDIT-5)


def main():
    t_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Experiment: {EXP_NAME}")
    print(f"Config: hidden={HIDDEN_DIM} layers={NUM_LAYERS} heads={NUM_HEADS} "
          f"epochs={EPOCHS} geo_w={GEOMETRY_WEIGHT} ode={ODE_STEPS} wd={WEIGHT_DECAY}")

    model = HybridFlowMatchingConformer(
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params/1e6:.2f}M")

    train_loader, val_loader = make_dataloaders(batch_size=BATCH_SIZE)

    # FIX-6: weight_decay=0.01 (was 1e-12 → essentially no regularization)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    for pg in optimizer.param_groups:
        pg['initial_lr'] = pg['lr']

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val = float('inf')
    t_start_train = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Cosine schedule with warmup (same pattern as Exp A/B/C/D)
        lr = get_lr(epoch - 1, LEARNING_RATE)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.train()
        total_loss = 0.0
        mse_loss_total = 0.0
        geo_loss_total = 0.0

        for batch in train_loader:
            at = batch['atom_types'].to(device)
            ct = batch['coordinates'].to(device)
            ei = batch['edge_index'].to(device)
            bt = batch['bond_types'].to(device)
            bi = batch['batch_idx'].to(device)

            # FIX-5: Remove CoM BEFORE passing to model.
            # HybridFlowMatchingConformer interpolates: x_t = (1-t)*x0 + t*eps
            # where eps = remove_com(randn). If x0 has non-zero CoM, the
            # interpolated x_t has CoM drift → model learns to generate
            # off-center molecules. EDM (Hoogeboom 2022, Sec 3.1) requires
            # zero-CoM inputs throughout training.
            ct = remove_com(ct, bi)

            optimizer.zero_grad()
            ld = model.get_loss(
                ct, at, ei, bt, bi,
                geometry_weight=GEOMETRY_WEIGHT,
                include_torsions=INCLUDE_TORSIONS,
            )
            loss = ld['total']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            mse_loss_total += ld['mse'].item()
            if isinstance(ld['geo'], torch.Tensor):
                geo_loss_total += ld['geo'].item()

        train_loss = total_loss / len(train_loader)
        train_mse = mse_loss_total / len(train_loader)
        train_geo = geo_loss_total / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                at = batch['atom_types'].to(device)
                ct = batch['coordinates'].to(device)
                ei = batch['edge_index'].to(device)
                bt = batch['bond_types'].to(device)
                bi = batch['batch_idx'].to(device)

                # FIX-5: Same CoM removal required in validation loop!
                ct = remove_com(ct, bi)

                ld = model.get_loss(
                    ct, at, ei, bt, bi,
                    geometry_weight=GEOMETRY_WEIGHT,
                    include_torsions=INCLUDE_TORSIONS,
                )
                val_loss += ld['total'].item()

        val_loss /= len(val_loader)
        lr_curr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:03d}/{EPOCHS} | train={train_loss:.4f} "
              f"(mse={train_mse:.4f} geo={train_geo:.4f}) val={val_loss:.4f} "
              f"| lr={lr_curr:.2e}", flush=True)

        if SAVE_BEST and val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"{EXP_NAME}_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'exp_name': EXP_NAME,
                'config': {
                    'hidden_dim': HIDDEN_DIM, 'num_layers': NUM_LAYERS,
                    'num_heads': NUM_HEADS, 'ode_steps': ODE_STEPS,
                    'geometry_weight': GEOMETRY_WEIGHT,
                },
            }, ckpt_path)

    training_secs = time.time() - t_start_train

    print("\nRunning final evaluation...", flush=True)
    best_ckpt = os.path.join(CHECKPOINT_DIR, f"{EXP_NAME}_best.pt")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    # Evaluate with ODE sampler (20 steps) — ddim_sample is aliased to ode_sample
    metrics = evaluate_all(model, val_loader, device, num_gen=EVAL_MOLECULES, verbose=True)
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
    print(f"num_params_M:   {n_params/1e6:.2f}")
    print(f"optimizer:      {OPTIMIZER}")
    print(f"geometry_weight:{GEOMETRY_WEIGHT}")
    print(f"ode_steps:      {ODE_STEPS}")
    print(f"exp_name:       {EXP_NAME}")


if __name__ == "__main__":
    main()
