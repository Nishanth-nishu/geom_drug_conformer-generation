"""
mol_train_expD.py — Experiment D: Torsion-Angle Auxiliary Loss (TorDiff-inspired)

Research hypothesis:
  The geometry loss in Exp A only applies bond/angle constraints.
  TorDiff (Jing et al. NeurIPS 2022) shows that explicit dihedral angle
  supervision is THE key differentiator for low MAT-R on QM9:
  conformer accuracy is dominated by torsion angle correctness, not
  just bond lengths/angles.

  Strategy:
  - Enable include_torsions=True in geometry loss
  - Increase geometry_weight to 0.5 (stronger geometry supervision)
  - Keep everything else identical to baseline (Exp A)

  Expected: 15-25% reduction in MAT-R (matching published TorDiff numbers)

Reference:
  Jing et al. "Torsional Diffusion for Molecular Conformer Generation"
  NeurIPS 2022. arXiv:2206.01729

  Ganea et al. "GeoMol: Torsional Graph Neural Network for Molecular
  Conformer Generation and Property Prediction" NeurIPS 2021.
"""

import os, sys, time, math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoresearch.mol_prepare import (
    EPOCH_BUDGET, EVAL_MOLECULES, make_dataloaders,
    evaluate_all, print_report, CHECKPOINT_DIR,
)
from models.conformer_diffusion import ConformerDiffusion, remove_com

# ============================================================================
# HYPERPARAMETERS — Experiment D
# ============================================================================

MODEL_HIDDEN_DIM = 256
MODEL_NUM_LAYERS = 6
MODEL_TIMESTEPS  = 1000
MODEL_TIME_DIM   = 128

BATCH_SIZE       = 64
LEARNING_RATE    = 1e-4
WEIGHT_DECAY     = 0.01
OPTIMIZER        = "adamw"

EPOCHS           = EPOCH_BUDGET
GEOMETRY_WEIGHT  = 0.5       # INCREASED: stronger geometry supervision
INCLUDE_TORSIONS = True      # KEY CHANGE: enable torsion angle loss on x_0_pred
                             # (NOTE: must pass this into get_loss, NOT on GT coords!)
WARMUP_EPOCHS    = 5
MIN_SNR_GAMMA    = 5.0
DDIM_STEPS       = 50

SAVE_BEST        = True
EXP_NAME         = "exp_D_torsion_aux"


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
          f"epochs={EPOCHS} geo_w={GEOMETRY_WEIGHT} torsions={INCLUDE_TORSIONS}")

    train_loader, val_loader = make_dataloaders(batch_size=BATCH_SIZE)

    model = ConformerDiffusion(
        num_timesteps=MODEL_TIMESTEPS,
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
            co = batch['coordinates'].to(device)
            ei = batch['edge_index'].to(device)
            bt = batch['bond_types'].to(device)
            bi = batch['batch_idx'].to(device)
            co = remove_com(co, bi)

            optimizer.zero_grad(set_to_none=True)
            # KEY CHANGE: pass include_torsions=True so torsion loss is computed
            # on x_0_pred (the predicted clean coords) — NOT on ground-truth 'co'.
            # The old approach of computing torsion_loss(co, ...) was wrong because
            # GT QM9 DFT coords already minimize torsion energy → loss ≈ 0.0.
            loss_dict = model.get_loss(
                co, at, ei, bt, bi,
                geometry_weight=GEOMETRY_WEIGHT,
                epoch=epoch, max_epochs=EPOCHS,
                min_snr_gamma=MIN_SNR_GAMMA,
                include_torsions=INCLUDE_TORSIONS,  # ← torsion on x_0_pred
            )
            loss = loss_dict['total']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
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
                    'geometry_weight': GEOMETRY_WEIGHT,
                    'include_torsions': INCLUDE_TORSIONS,
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
    print(f"include_torsions:{INCLUDE_TORSIONS}")
    print(f"exp_name:       {EXP_NAME}")


if __name__ == '__main__':
    main()
