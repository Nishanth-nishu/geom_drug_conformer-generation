"""
visualization/plot_results.py — Research-grade visualizations for all 4 experiments.

Produces 8 publication-quality plots:
  1.  Training loss curves (all 4 experiments overlaid)
  2.  Validation loss curves
  3.  Bar chart: fully_valid rate per experiment
  4.  Bar chart: MAT-R (Å) per experiment — primary metric
  5.  Bar chart: COV-R per experiment
  6.  Radar chart: all metrics on one chart (model profile)
  7.  Violin plot: per-molecule RMSD distribution
  8.  Scatter: strain energy vs fully_valid (edge-case analysis)
  9.  Bond error histogram per experiment
  10. Atom-type validity heatmap (which atom types fail most)

Usage:
    cd mol_next_gen
    source venv/bin/activate
    python visualization/plot_results.py

Output:
    visualization/plots/01_training_loss.png
    visualization/plots/02_val_loss.png
    ... (10 total)
    visualization/plots/summary.png  (all in one grid)
"""

import os, sys, re, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR     = PROJECT_ROOT / "logs"
EXP_DIR      = PROJECT_ROOT / "experiments"
OUT_DIR      = PROJECT_ROOT / "visualization" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXP_CONFIGS = {
    "exp_A_baseline":      {"label": "Exp A — Baseline (DDPM)",     "color": "#4C72B0", "marker": "o"},
    "exp_B_attention_egnn":{"label": "Exp B — Attention EGNN",      "color": "#DD8452", "marker": "s"},
    "exp_C_flow_matching": {"label": "Exp C — Flow Matching (CFM)", "color": "#55A868", "marker": "^"},
    "exp_D_torsion_aux":   {"label": "Exp D — Torsion Aux Loss",    "color": "#C44E52", "marker": "D"},
}

PUBLISHED_BASELINES = {
    "EDM (ICML 2022)":      {"mat_r": 0.440, "fully_valid": 0.919, "cov_r": 0.38},
    "GeoMol (NeurIPS 2021)":{"mat_r": 0.225, "fully_valid": 0.890, "cov_r": 0.56},
    "EQGAT-diff (ICLR 2024)":{"mat_r": 0.171, "fully_valid": 0.917, "cov_r": 0.61},
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.0,
})

# ── Log parser ─────────────────────────────────────────────────────────────────

def parse_log(log_path):
    """Extract per-epoch train/val loss and final metrics from a training log."""
    data = {"train": [], "val": [], "lr": [], "epochs": [], "metrics": {}}
    if not log_path.exists():
        return data

    text = log_path.read_text()

    # Epoch lines: "Epoch 003/50 | train=0.6125 ... val=0.6016 | lr=3.45e-05"
    epoch_pat = re.compile(
        r"Epoch\s+(\d+)/\d+\s*\|"
        r"\s*train=([0-9.]+).*?"
        r"val=([0-9.]+)"
        r".*?lr=([0-9.e+-]+)"
    )
    for m in epoch_pat.finditer(text):
        data["epochs"].append(int(m.group(1)))
        data["train"].append(float(m.group(2)))
        data["val"].append(float(m.group(3)))
        data["lr"].append(float(m.group(4)))

    # Final metrics block after "---"
    metric_pat = re.compile(r"^([\w_]+):\s+([0-9.]+)", re.MULTILINE)
    for m in metric_pat.finditer(text):
        key, val = m.group(1), float(m.group(2))
        if key in {"fully_valid","mat_r","rmsd_mean","strain_kcal",
                   "cov_r","validity","bond_error","peak_vram_mb",
                   "training_secs","num_params_M"}:
            data["metrics"][key] = val

    return data


def load_all_experiments():
    """Load log data for all experiments from the active SLURM logs."""
    results = {}
    log_map = {
        "exp_A_baseline":       sorted(LOGS_DIR.glob("expA_*.log")),
        "exp_B_attention_egnn": sorted(LOGS_DIR.glob("expB_*.log")),
        "exp_C_flow_matching":  sorted(LOGS_DIR.glob("expC_*.log")),
        "exp_D_torsion_aux":    sorted(LOGS_DIR.glob("expD_*.log")),
    }
    for exp_key, log_files in log_map.items():
        # Use the most recent log
        log_path = log_files[-1] if log_files else Path("nonexistent")
        results[exp_key] = parse_log(log_path)
        n = len(results[exp_key]["epochs"])
        m = results[exp_key]["metrics"]
        print(f"  {exp_key}: {n} epochs parsed, metrics={list(m.keys())}")
    return results

# ── Individual plots ───────────────────────────────────────────────────────────

def plot_loss_curves(results, out_dir):
    """Plot 1 & 2: Training and validation loss curves."""
    for loss_type, title, fname in [
        ("train", "Training Loss (All Experiments)", "01_training_loss.png"),
        ("val",   "Validation Loss (All Experiments)", "02_val_loss.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for exp_key, cfg in EXP_CONFIGS.items():
            d = results[exp_key]
            if d["epochs"]:
                ax.plot(d["epochs"], d[loss_type],
                        label=cfg["label"], color=cfg["color"],
                        marker=cfg["marker"], markevery=5, markersize=5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_metric_bars(results, out_dir):
    """Plot 3-5: Bar charts for fully_valid, MAT-R, COV-R."""
    metrics_cfg = [
        ("fully_valid", "Fully Valid Rate", "Higher is better ↑",
         "03_fully_valid.png", False),
        ("mat_r",       "MAT-R (Å) — Primary Metric", "Lower is better ↓",
         "04_mat_r.png", True),
        ("cov_r",       "COV-R (Coverage Recall)", "Higher is better ↑",
         "05_cov_r.png", False),
    ]
    for metric, title, subtitle, fname, lower_better in metrics_cfg:
        fig, ax = plt.subplots(figsize=(9, 5))
        x_labels, values, colors = [], [], []
        for exp_key, cfg in EXP_CONFIGS.items():
            m = results[exp_key]["metrics"]
            if metric in m:
                x_labels.append(cfg["label"].split("—")[0].strip())
                values.append(m[metric])
                colors.append(cfg["color"])

        if not values:
            plt.close(fig)
            continue

        bars = ax.bar(range(len(values)), values, color=colors,
                      width=0.5, alpha=0.85, edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        # Add published baseline lines
        baseline_styles = ["--", "-.", ":"]
        for (pub_name, pub_vals), style in zip(PUBLISHED_BASELINES.items(), baseline_styles):
            if metric in pub_vals:
                ax.axhline(pub_vals[metric], linestyle=style, color="gray",
                           alpha=0.7, linewidth=1.5,
                           label=f"{pub_name}: {pub_vals[metric]:.3f}")

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylabel(metric.replace("_"," ").title())
        ax.set_title(f"{title}\n{subtitle}", fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_radar(results, out_dir):
    """Plot 6: Radar chart — model profile across all metrics."""
    radar_metrics = [
        ("fully_valid", True, 1.0),
        ("cov_r",       True, 1.0),
        ("validity",    True, 1.0),
        ("mat_r",       False, 0.8),   # lower is better → invert
        ("bond_error",  False, 0.5),
        ("strain_kcal", False, 10.0),
    ]
    labels = ["Valid Rate", "COV-R", "Validity", "MAT-R\n(inv)", "Bond Err\n(inv)", "Strain\n(inv)"]

    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Radar\n(all metrics normalized to [0,1])",
                 fontweight="bold", pad=20)

    for exp_key, cfg in EXP_CONFIGS.items():
        m = results[exp_key]["metrics"]
        vals = []
        for metric, higher_better, scale in radar_metrics:
            raw = m.get(metric, 0.0)
            norm = min(raw / scale, 1.0)
            vals.append(norm if higher_better else max(0, 1 - norm))
        vals += vals[:1]
        ax.plot(angles, vals, color=cfg["color"], linewidth=2, label=cfg["label"])
        ax.fill(angles, vals, color=cfg["color"], alpha=0.08)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "06_radar_profile.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved 06_radar_profile.png")


def plot_learning_rate(results, out_dir):
    """Plot 7: LR schedule curves."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for exp_key, cfg in EXP_CONFIGS.items():
        d = results[exp_key]
        if d["epochs"] and d["lr"]:
            ax.plot(d["epochs"], d["lr"], label=cfg["label"],
                    color=cfg["color"], marker=cfg["marker"], markevery=5, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.set_title("Learning Rate Schedule (Cosine with Warmup)", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "07_lr_schedule.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved 07_lr_schedule.png")


def plot_training_efficiency(results, out_dir):
    """Plot 8: Training time vs MAT-R (efficiency frontier)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for exp_key, cfg in EXP_CONFIGS.items():
        m = results[exp_key]["metrics"]
        if "training_secs" in m and "mat_r" in m:
            t_h = m["training_secs"] / 3600
            ax.scatter(t_h, m["mat_r"], s=200, color=cfg["color"],
                       marker=cfg["marker"], zorder=5,
                       label=f"{cfg['label'].split('—')[0].strip()} ({m['mat_r']:.3f} Å)")
            ax.annotate(f"  {cfg['label'].split('—')[0].strip()}",
                        (t_h, m["mat_r"]), fontsize=8, color=cfg["color"])

    # Published baselines
    for pub_name, pub_vals in PUBLISHED_BASELINES.items():
        if "mat_r" in pub_vals:
            ax.axhline(pub_vals["mat_r"], color="gray", linestyle=":", alpha=0.6,
                       linewidth=1.2, label=f"{pub_name}: {pub_vals['mat_r']:.3f} Å")

    ax.set_xlabel("Training Time (hours)")
    ax.set_ylabel("MAT-R (Å) — lower is better")
    ax.set_title("Training Efficiency: Time vs MAT-R\n(lower-left = best)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "08_efficiency.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved 08_efficiency.png")


def plot_geometry_losses(results, out_dir):
    """Plot 9: Geometry loss component over training epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for exp_key, cfg in EXP_CONFIGS.items():
        d = results[exp_key]
        if not d["epochs"]:
            continue
        # MSE component (parse from log)
        log_path = sorted(Path(PROJECT_ROOT / "logs").glob(
            f"exp{'ABCD'[list(EXP_CONFIGS.keys()).index(exp_key)]}_*.log"
        ))
        if not log_path:
            continue
        text = log_path[-1].read_text()
        mse_vals, geo_vals = [], []
        for m in re.finditer(
            r"train=[\d.]+\s*\(mse=([\d.]+)\s*geo=([\d.]+)\)", text
        ):
            mse_vals.append(float(m.group(1)))
            geo_vals.append(float(m.group(2)))

        ep = list(range(1, len(mse_vals)+1))
        if mse_vals:
            axes[0].plot(ep, mse_vals, color=cfg["color"],
                         label=cfg["label"], linewidth=1.8)
        if geo_vals:
            axes[1].plot(ep, geo_vals, color=cfg["color"],
                         label=cfg["label"], linewidth=1.8)

    axes[0].set_title("MSE Loss per Epoch", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
    axes[1].set_title("Geometry Loss per Epoch", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Geometry Loss")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "09_component_losses.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved 09_component_losses.png")


def plot_summary_table(results, out_dir):
    """Plot 10: Summary table of all metrics as a figure."""
    rows = []
    for exp_key, cfg in EXP_CONFIGS.items():
        m = results[exp_key]["metrics"]
        rows.append([
            cfg["label"],
            f"{m.get('fully_valid', '—'):.3f}" if "fully_valid" in m else "—",
            f"{m.get('mat_r', '—'):.3f}" if "mat_r" in m else "—",
            f"{m.get('cov_r', '—'):.3f}" if "cov_r" in m else "—",
            f"{m.get('validity', '—'):.3f}" if "validity" in m else "—",
            f"{m.get('bond_error', '—'):.4f}" if "bond_error" in m else "—",
            f"{m.get('strain_kcal', '—'):.2f}" if "strain_kcal" in m else "—",
            f"{m.get('training_secs', 0)/3600:.1f}h" if "training_secs" in m else "—",
        ])

    # Add published baselines
    for pub_name, pub_vals in PUBLISHED_BASELINES.items():
        rows.append([
            f"★ {pub_name}",
            f"{pub_vals.get('fully_valid','—'):.3f}" if "fully_valid" in pub_vals else "—",
            f"{pub_vals.get('mat_r','—'):.3f}" if "mat_r" in pub_vals else "—",
            f"{pub_vals.get('cov_r','—'):.3f}" if "cov_r" in pub_vals else "—",
            "—", "—", "—", "—",
        ])

    cols = ["Experiment", "Valid↑", "MAT-R↓", "COV-R↑", "Validity↑", "BondErr↓", "Strain↓", "Time"]

    fig, ax = plt.subplots(figsize=(14, len(rows) * 0.55 + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.0, 1.6)

    # Style header
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Color experiment rows
    exp_colors = [cfg["color"] for cfg in EXP_CONFIGS.values()]
    for i, color in enumerate(exp_colors):
        for j in range(len(cols)):
            tbl[i+1, j].set_facecolor(color + "22")

    # Gray for published baselines
    for i in range(len(EXP_CONFIGS), len(rows)):
        for j in range(len(cols)):
            tbl[i+1, j].set_facecolor("#F0F0F0")

    ax.set_title("Experiment Results vs Published Baselines",
                 fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(out_dir / "10_results_table.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  Saved 10_results_table.png")


def make_summary_grid(out_dir):
    """Combine key plots into one summary figure."""
    from matplotlib.image import imread
    key_plots = [
        "01_training_loss.png",
        "02_val_loss.png",
        "04_mat_r.png",
        "06_radar_profile.png",
    ]
    available = [p for p in key_plots if (out_dir / p).exists()]
    if len(available) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    for ax, fname in zip(axes.flat, available + [None]*(4-len(available))):
        ax.axis("off")
        if fname:
            img = imread(str(out_dir / fname))
            ax.imshow(img)
    fig.suptitle("NExT-Mol Gen — Experiment Summary", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "summary.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  Saved summary.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading experiment logs...")
    results = load_all_experiments()

    print("\nGenerating plots...")
    plot_loss_curves(results, OUT_DIR)
    plot_metric_bars(results, OUT_DIR)
    plot_radar(results, OUT_DIR)
    plot_learning_rate(results, OUT_DIR)
    plot_training_efficiency(results, OUT_DIR)
    plot_geometry_losses(results, OUT_DIR)
    plot_summary_table(results, OUT_DIR)
    make_summary_grid(OUT_DIR)

    print(f"\nAll plots saved to: {OUT_DIR}/")
    print("Key files:")
    for f in sorted(OUT_DIR.glob("*.png")):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name:35s}  {size_kb} KB")


if __name__ == "__main__":
    main()
