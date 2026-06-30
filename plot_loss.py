#!/usr/bin/env python3
"""
plot_loss_curves.py — Professional loss curve visualizations for v2_confs5 and v2_confs10.
Outputs two separate PNG files, one per experiment.
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.45,
    'grid.color': '#cccccc',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Colors ─────────────────────────────────────────────────────────────────────
TRAIN_COLOR = '#2563EB'   # deep blue
VAL_COLOR   = '#DC2626'   # vivid red
LOCAL_COLOR = '#7C3AED'   # purple
GLOB_COLOR  = '#059669'   # emerald
LR_COLOR    = '#D97706'   # amber

def parse_log(filepath):
    """Parse a training log and return structured data."""
    pattern = re.compile(
        r"Epoch\s+(\d+)/\d+\s+\|\s+"
        r"train=([\d.]+)\s+\(loc=([\d.]+)\s+glob=([\d.]+)\)\s+"
        r"val=([\d.]+)\s+\|\s+lr=([\deE.+-]+)"
    )

    epochs, train_losses, val_losses = [], [], []
    local_losses, global_losses, lrs = [], [], []

    # Skip the smoke-test section (001/5) — we want only the full run (001/300+)
    full_run_started = False

    with open(filepath, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            ep   = int(m.group(1))
            t    = float(m.group(2))
            loc  = float(m.group(3))
            glob = float(m.group(4))
            v    = float(m.group(5))
            lr   = float(m.group(6))

            # Detect transition from smoke-test to full run by epoch reset
            if ep == 1 and epochs and epochs[-1] > 1:
                full_run_started = True
                epochs.clear(); train_losses.clear(); val_losses.clear()
                local_losses.clear(); global_losses.clear(); lrs.clear()

            epochs.append(ep)
            train_losses.append(t)
            val_losses.append(v)
            local_losses.append(loc)
            global_losses.append(glob)
            lrs.append(lr)

    return {
        'epochs':        np.array(epochs),
        'train':         np.array(train_losses),
        'val':           np.array(val_losses),
        'local':         np.array(local_losses),
        'global_':       np.array(global_losses),
        'lr':            np.array(lrs),
    }


def smooth(arr, window=5):
    """Simple moving average for smoother curves."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window//2, window//2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


def plot_experiment(data, exp_name, confs, out_path):
    fig = plt.figure(figsize=(14, 10), facecolor='#FAFAFA')
    fig.suptitle(
        f'DualEncoderDiffusion — Conformers={confs}\n'
        f'{exp_name}',
        fontsize=14, fontweight='bold', color='#1a1a2e', y=0.98
    )

    # ── Subplot layout: 3 rows ─────────────────────────────────────────────────
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35,
                          left=0.08, right=0.97, top=0.90, bottom=0.07)

    epochs = data['epochs']
    train  = data['train']
    val    = data['val']
    local  = data['local']
    glob   = data['global_']
    lr     = data['lr']

    # ─── 1. Main loss (top, wide) ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(epochs, train, color=TRAIN_COLOR, alpha=0.25, linewidth=0.8)
    ax1.plot(epochs, smooth(train, 7), color=TRAIN_COLOR, linewidth=2.2,
             label='Train Loss (smoothed)')
    ax1.plot(epochs, val, color=VAL_COLOR, alpha=0.25, linewidth=0.8)
    ax1.plot(epochs, smooth(val, 7), color=VAL_COLOR, linewidth=2.2,
             label='Val Loss (smoothed)')

    best_val_ep = epochs[np.argmin(val)]
    best_val    = val.min()
    ax1.axvline(best_val_ep, color=VAL_COLOR, linestyle=':', alpha=0.7, linewidth=1.4)
    ax1.annotate(f'Best Val={best_val:.1f}\n@ Ep{best_val_ep}',
                 xy=(best_val_ep, best_val),
                 xytext=(best_val_ep + max(len(epochs)//20, 3), best_val + (val.max()-val.min())*0.1),
                 fontsize=9.5, color=VAL_COLOR,
                 arrowprops=dict(arrowstyle='->', color=VAL_COLOR, lw=1.2))

    ax1.set_title('Total Loss (Train vs. Validation)', fontweight='semibold', pad=6)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score Matching Loss')
    ax1.legend(loc='upper right')
    ax1.set_xlim(epochs[0], epochs[-1])

    # ─── 2. Local loss ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(epochs, local, color=LOCAL_COLOR, alpha=0.2, linewidth=0.8)
    ax2.plot(epochs, smooth(local, 7), color=LOCAL_COLOR, linewidth=2.2,
             label='Local (GIN bond graph)')
    ax2.set_title('Local Loss (Bond Topology GIN)', fontweight='semibold', pad=5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.set_xlim(epochs[0], epochs[-1])

    # ─── 3. Global loss ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(epochs, glob, color=GLOB_COLOR, alpha=0.2, linewidth=0.8)
    ax3.plot(epochs, smooth(glob, 7), color=GLOB_COLOR, linewidth=2.2,
             label='Global (SchNet+ET radius)')
    ax3.set_title('Global Loss (Radius Graph SchNet+ET)', fontweight='semibold', pad=5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.set_xlim(epochs[0], epochs[-1])

    # ─── 4. Train/Val Gap (generalization) ────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    gap = train - val
    ax4.fill_between(epochs, gap, 0,
                     where=(gap >= 0), color=TRAIN_COLOR, alpha=0.25, label='Overfitting zone')
    ax4.fill_between(epochs, gap, 0,
                     where=(gap < 0), color=VAL_COLOR, alpha=0.25, label='Underfitting zone')
    ax4.plot(epochs, smooth(gap, 7), color='#374151', linewidth=2, label='Train−Val gap')
    ax4.axhline(0, color='gray', linewidth=1, linestyle='-')
    ax4.set_title('Generalization Gap (Train − Val)', fontweight='semibold', pad=5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss Difference')
    ax4.legend(fontsize=9)
    ax4.set_xlim(epochs[0], epochs[-1])

    # ─── 5. Learning Rate Schedule ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.semilogy(epochs, lr, color=LR_COLOR, linewidth=2.2, label='Learning Rate')
    ax5.set_title('Learning Rate Schedule', fontweight='semibold', pad=5)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('LR (log scale)')
    ax5.legend()
    ax5.set_xlim(epochs[0], epochs[-1])

    # ─── Watermark ────────────────────────────────────────────────────────────
    fig.text(0.99, 0.005, f'Total epochs logged: {len(epochs)}  |  '
             f'Best Train: {train.min():.2f}  |  Best Val: {best_val:.2f}',
             ha='right', va='bottom', fontsize=8.5, color='#6b7280')

    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print(f"✓ Saved: {out_path}")


def main():
    base = Path('logs')
    out  = Path('.')

    configs = [
        {
            'log': base / 'v2_confs5_2637673.log',
            'confs': 5,
            'name': 'v2_dual_confs5_T5000_atoms30_mols50000',
            'out': out / 'loss_curves_confs5.png',
        },
        {
            'log': base / 'v2_confs10_2637674.log',
            'confs': 10,
            'name': 'v2_dual_confs10_T5000_atoms30_mols50000',
            'out': out / 'loss_curves_confs10.png',
        },
    ]

    for cfg in configs:
        print(f"\nParsing: {cfg['log']}")
        data = parse_log(cfg['log'])
        print(f"  Epochs found: {len(data['epochs'])} | "
              f"Train range: {data['train'].min():.1f}–{data['train'].max():.1f} | "
              f"Val range: {data['val'].min():.1f}–{data['val'].max():.1f}")
        plot_experiment(data, cfg['name'], cfg['confs'], cfg['out'])

    print("\nAll done! Files: loss_curves_confs5.png, loss_curves_confs10.png")


if __name__ == '__main__':
    main()
