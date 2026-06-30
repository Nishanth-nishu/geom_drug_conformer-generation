"""
visualization/expH_publication_plots.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Publication-quality visualisations for Experiment H
— Energy-Guided Diffusion for Conformer Generation (GEOM-Drugs / QM9 Protocol)

Produces 8 research-grade figures saved as both PNG (150 dpi) and PDF (300 dpi):
  1.  Training + Validation loss convergence (diffusion phase)
  2.  Component losses breakdown  (MSE vs Geometry loss)
  3.  Learning-rate schedule      (cosine-annealed with linear warm-up)
  4.  Evaluation metrics progress (MAT-R & RMSD mean at every 50 epochs)
  5.  MEE (Mean Energy Error) progression
  6.  SOTA comparison bar chart   (MAT-R, MAT-P vs GeoDiff/TorDiff/EQGAT-diff)
  7.  Surrogate pre-training curves (Phase 1: GFN2-xTB energy learning)
  8.  Publication summary dashboard (2x4 panel)

Usage:
    cd /scratch/nishanth.r/nextmol_experiment/mol_expermiments
    source venv/bin/activate
    PYTHONPATH=. python visualization/expH_publication_plots.py
"""

import re, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter1d

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "logs" / "expH_2635290.log"
OUT_DIR  = ROOT / "visualization" / "expH_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Colour palette & style ──────────────────────────────────────────────────
PALETTE = {
    "train":      "#2E86AB",
    "val":        "#E84855",
    "mse":        "#4CAF50",
    "geo":        "#FF9800",
    "lr":         "#9C27B0",
    "mat_r":      "#2E86AB",
    "rmsd":       "#FF9800",
    "mee":        "#E84855",
    "surr_train": "#00ACC1",
    "surr_val":   "#EC407A",
    "geodiff":    "#607D8B",
    "tordiff":    "#795548",
    "expH":       "#2E86AB",
    "eqgat":      "#43A047",
    "shadow":     "#1565C0",
}

plt.rcParams.update({
    "font.family":         "DejaVu Sans",
    "font.size":           12,
    "axes.titlesize":      14,
    "axes.titleweight":    "bold",
    "axes.labelsize":      12,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.alpha":          0.25,
    "grid.linewidth":      0.7,
    "lines.linewidth":     2.2,
    "lines.antialiased":   True,
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "legend.framealpha":   0.92,
    "legend.edgecolor":    "#cccccc",
    "legend.fontsize":     10,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
})


# ─── Log Parser ───────────────────────────────────────────────────────────────
def parse_expH_log(log_path: Path) -> dict:
    text = log_path.read_text()

    # Phase 1: Surrogate
    surr = {"epochs": [], "train": [], "val": [], "mae": [], "lr": []}
    surr_pat = re.compile(
        r"Surrogate Epoch\s+(\d+)/\d+\s*\|"
        r"\s*train=([\d.]+)\s*\(mae=([\d.]+)[^)]*\)\s*val=([\d.]+)\s+lr=([\de.+-]+)"
    )
    for m in surr_pat.finditer(text):
        surr["epochs"].append(int(m.group(1)))
        surr["train"].append(float(m.group(2)))
        surr["val"].append(float(m.group(4)))
        surr["mae"].append(float(m.group(3)))
        surr["lr"].append(float(m.group(5)))

    # Phase 2: Diffusion
    diff = {"epochs": [], "train": [], "val": [], "mse": [], "geo": [], "lr": []}
    diff_pat = re.compile(
        r"Epoch\s+(\d+)/500\s*\|"
        r"\s*train=([\d.]+)\s*\(mse=([\d.]+)\s+geo=([\d.]+)[^)]*\)"
        r"\s*val=([\d.]+)\s*\|\s*lr=([\de.+-]+)"
    )
    for m in diff_pat.finditer(text):
        diff["epochs"].append(int(m.group(1)))
        diff["train"].append(float(m.group(2)))
        diff["val"].append(float(m.group(5)))
        diff["mse"].append(float(m.group(3)))
        diff["geo"].append(float(m.group(4)))
        diff["lr"].append(float(m.group(6)))

    # Eval blocks
    eval_epoch_pat = re.compile(r"\[GEOM Eval\] Epoch (\d+)")
    mat_r_pat      = re.compile(r"MAT-R\s*:\s*([\d.]+)\s*A")
    mat_p_pat      = re.compile(r"MAT-P\s*:\s*([\d.]+)\s*A")
    cov_r_pat      = re.compile(r"COV-R@0\.5A\s*:\s*([\d.]+)%")
    rmsd_pat       = re.compile(r"RMSD mean\s*:\s*([\d.]+)\s*A")
    mee_pat        = re.compile(r"MEE\s*:\s*\+([\d.]+)")

    evals = []
    eval_starts = [(m.start(), int(m.group(1))) for m in eval_epoch_pat.finditer(text)]
    for i, (start, epoch) in enumerate(eval_starts):
        end   = eval_starts[i+1][0] if i+1 < len(eval_starts) else len(text)
        block = text[start:end]
        ev    = {"epoch": epoch}
        for pat, key in [(mat_r_pat,"mat_r"),(mat_p_pat,"mat_p"),
                         (cov_r_pat,"cov_r"),(rmsd_pat,"rmsd_mean"),(mee_pat,"mee")]:
            mm = pat.search(block)
            ev[key] = float(mm.group(1)) if mm else None
        evals.append(ev)

    print(f"[Parser] Surrogate : {len(surr['epochs'])} epochs")
    print(f"[Parser] Diffusion : {len(diff['epochs'])} epochs")
    print(f"[Parser] Eval snaps: {len(evals)}")
    return {"surrogate": surr, "diffusion": diff, "evals": evals}


def add_watermark(ax, text="Exp H — Energy-Guided Diffusion"):
    ax.text(0.99, 0.01, text, transform=ax.transAxes,
            fontsize=7.5, color="#aaaaaa", ha="right", va="bottom", style="italic")


def save(fig, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    print(f"  saved  {name}.png / .pdf")
    plt.close(fig)


# ─── Plot 1: Training & Validation loss ──────────────────────────────────────
def plot_training_val_loss(data):
    d = data["diffusion"]
    ep, tr, vl = d["epochs"], d["train"], d["val"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ep, tr, color=PALETTE["train"], lw=2.2, label="Train loss")
    ax.plot(ep, vl, color=PALETTE["val"],   lw=2.2, ls="--", label="Val loss")

    v_arr = np.array(vl)
    if len(v_arr) >= 5:
        v_sm = uniform_filter1d(v_arr, size=5)
        ax.plot(ep, v_sm, color=PALETTE["val"], lw=1.2, alpha=0.4)

    for ev in data["evals"]:
        ax.axvline(ev["epoch"], color="#999999", lw=0.7, ls=":", alpha=0.55)

    best_ep = 300
    idx = ep.index(best_ep) if best_ep in ep else -1
    bv  = vl[idx] if idx >= 0 else min(vl)
    ax.scatter([best_ep], [bv], s=90, color="#1565C0", zorder=6)
    ax.annotate(f" Best checkpoint\n Epoch {best_ep}", xy=(best_ep, bv),
                xytext=(best_ep+15, bv+0.15), fontsize=9, color="#1565C0",
                arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.2))

    ax.set_xlabel("Epoch"); ax.set_ylabel("Total Loss")
    ax.set_title("Experiment H — Diffusion Training: Loss Convergence\n"
                 "(GEOM-Drugs · 96 418 molecules · 500 epochs · RTX 3090 24 GB)",
                 fontweight="bold")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.legend()
    add_watermark(ax)
    fig.tight_layout()
    save(fig, "01_training_validation_loss")


# ─── Plot 2: Component losses ─────────────────────────────────────────────────
def plot_component_losses(data):
    d  = data["diffusion"]
    ep = d["epochs"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, key, color, title in [
        (axes[0], "mse", PALETTE["mse"], "Score-Matching MSE Loss"),
        (axes[1], "geo", PALETTE["geo"], "Geometry Constraint Loss (Bond / Angle)"),
    ]:
        vals = d[key]
        ax.plot(ep, vals, color=color, lw=2.0)
        ax.fill_between(ep, vals, alpha=0.08, color=color)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(title, fontweight="bold")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        add_watermark(ax)

    fig.suptitle("Experiment H — Loss Component Decomposition over Training",
                 fontweight="bold", fontsize=14, y=1.01)
    fig.tight_layout()
    save(fig, "02_component_losses")


# ─── Plot 3: LR schedule ─────────────────────────────────────────────────────
def plot_lr_schedule(data):
    d  = data["diffusion"]
    ep = d["epochs"]
    lr = d["lr"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ep, lr, color=PALETTE["lr"], lw=2.2)
    ax.fill_between(ep, lr, alpha=0.10, color=PALETTE["lr"])

    peak_idx = int(np.argmax(lr))
    ax.annotate(f"Peak LR = {lr[peak_idx]:.2e}\n(Epoch {ep[peak_idx]})",
                xy=(ep[peak_idx], lr[peak_idx]),
                xytext=(ep[peak_idx]+20, lr[peak_idx]*1.05),
                fontsize=9, color=PALETTE["lr"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["lr"], lw=1.2))

    ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate (log scale)")
    ax.set_title("Experiment H — Cosine-Annealed LR Schedule with Linear Warm-up",
                 fontweight="bold")
    add_watermark(ax)
    fig.tight_layout()
    save(fig, "03_lr_schedule")


# ─── Plot 4: Eval metrics progression ────────────────────────────────────────
def plot_eval_metrics(data):
    evals  = data["evals"]
    ep_mr  = [e["epoch"] for e in evals if e.get("mat_r") is not None]
    mat_r  = [e["mat_r"]  for e in evals if e.get("mat_r") is not None]
    ep_rm  = [e["epoch"] for e in evals if e.get("rmsd_mean") is not None]
    rmsd   = [e["rmsd_mean"] for e in evals if e.get("rmsd_mean") is not None]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(ep_mr, mat_r, "o-", color=PALETTE["mat_r"], lw=2.2, ms=8, label="MAT-R (Å) ↓")
    l2, = ax2.plot(ep_rm, rmsd,  "s--",color=PALETTE["rmsd"],  lw=2.0, ms=7, label="RMSD mean (Å)")

    ax1.set_ylabel("MAT-R (Å) [lower = better]", color=PALETTE["mat_r"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["mat_r"])
    ax2.set_ylabel("RMSD Mean (Å)", color=PALETTE["rmsd"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["rmsd"])

    best_ep, best_mat = 300, 1.6724
    ax1.axhline(best_mat, color=PALETTE["mat_r"], lw=1.0, ls=":", alpha=0.55)
    ax1.scatter([best_ep], [best_mat], s=130, color="#1565C0", zorder=8, marker="*")
    ax1.axhline(0.528, color=PALETTE["geodiff"], lw=1.4, ls="--", alpha=0.65, label="GeoDiff SOTA 0.528 Å")
    ax1.axhline(0.481, color=PALETTE["tordiff"], lw=1.4, ls="-.", alpha=0.65, label="TorDiff SOTA 0.481 Å")

    lines = [l1, l2,
             Line2D([0],[0], color="#1565C0", marker="*", lw=0, ms=10),
             Line2D([0],[0], color=PALETTE["geodiff"], lw=1.4, ls="--"),
             Line2D([0],[0], color=PALETTE["tordiff"], lw=1.4, ls="-.")]
    labels = ["MAT-R (Å) ↓", "RMSD Mean (Å)",
              f"Best checkpoint (Ep {best_ep}, MAT-R={best_mat:.4f})",
              "GeoDiff SOTA = 0.528 Å", "TorDiff SOTA = 0.481 Å"]
    ax1.legend(lines, labels, fontsize=9, loc="upper right")

    ax1.set_xlabel("Training Epoch")
    ax1.set_title("Experiment H — Evaluation Metrics Progression\n"
                  "(MAT-R & RMSD Mean at every 50-epoch eval checkpoint)",
                  fontweight="bold")
    add_watermark(ax1)
    fig.tight_layout()
    save(fig, "04_eval_metrics_progression")


# ─── Plot 5: MEE progression ──────────────────────────────────────────────────
def plot_mee_progression(data):
    evals = data["evals"]
    ep  = [e["epoch"] for e in evals if e.get("mee") is not None]
    mee = [e["mee"]   for e in evals if e.get("mee") is not None]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ep, mee, "D-", color=PALETTE["mee"], lw=2.2, ms=8)
    ax.fill_between(ep, mee, alpha=0.10, color=PALETTE["mee"])
    ax.axhline(0, color="#333333", lw=1.0, alpha=0.4,
               label="MEE = 0  (generated == ref stability)")

    best_idx = int(np.argmin(mee))
    ax.annotate(f"Min MEE: +{mee[best_idx]:.1f}\n(Epoch {ep[best_idx]})",
                xy=(ep[best_idx], mee[best_idx]),
                xytext=(ep[best_idx]+20, mee[best_idx]+8),
                fontsize=9, color=PALETTE["mee"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["mee"]))

    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("MEE (normalised)  [< 0 = more stable than DFT ref]")
    ax.set_title("Experiment H — Mean Energy Error (MEE) Progression\n"
                 "(Novel energy-aware metric; < 0 = generated conformers lower energy than DFT reference)",
                 fontweight="bold")
    ax.legend()
    add_watermark(ax)
    fig.tight_layout()
    save(fig, "05_mee_progression")


# ─── Plot 6: SOTA comparison ──────────────────────────────────────────────────
def plot_sota_comparison(data):
    evals   = data["evals"]
    best_ep = max((e for e in evals if e.get("mat_r")), key=lambda e: e["epoch"])
    mat_r_H = best_ep.get("mat_r", 1.6724)
    mat_p_H = best_ep.get("mat_p", 1.7897)

    methods   = ["GeoDiff\n(NeurIPS 2022)", "TorDiff\n(ICML 2023)",
                 "EQGAT-diff\n(ICLR 2024)", "Exp H\n(This work)"]
    mat_r_v   = [0.528, 0.481, 0.447, mat_r_H]
    mat_p_v   = [0.550, 0.423, 0.368, mat_p_H]
    colors_r  = [PALETTE["geodiff"], PALETTE["tordiff"], PALETTE["eqgat"], PALETTE["expH"]]
    colors_p  = [c+"AA" for c in colors_r]

    x     = np.arange(len(methods))
    w     = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))

    bars_r = ax.bar(x-w/2, mat_r_v, w, color=colors_r,  edgecolor="white", lw=1.5, alpha=0.88, label="MAT-R (Å) ↓")
    bars_p = ax.bar(x+w/2, mat_p_v, w, color=colors_p,  edgecolor="white", lw=1.5, alpha=0.88, label="MAT-P (Å) ↓", hatch="///")

    for bar in list(bars_r) + list(bars_p):
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.008,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.axvspan(x[-1]-0.42, x[-1]+0.42, alpha=0.07, color=PALETTE["expH"])
    ax.annotate("Our model\n(Epoch 300)",
                xy=(x[-1], mat_r_H),
                xytext=(x[-1]-0.95, mat_r_H+0.10),
                fontsize=9.5, color=PALETTE["expH"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PALETTE["expH"], lw=1.3))

    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Distance (Å)  [lower = better]")
    ax.set_title("Experiment H vs. State-of-the-Art\nGEOM-Drugs Conformer Generation: MAT-R & MAT-P",
                 fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(mat_r_v+mat_p_v)*1.18)
    add_watermark(ax)
    fig.tight_layout()
    save(fig, "06_sota_comparison")


# ─── Plot 7: Surrogate pretraining ───────────────────────────────────────────
def plot_surrogate_pretraining(data):
    s  = data["surrogate"]
    ep = s["epochs"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(ep, s["train"], color=PALETTE["surr_train"], lw=2.2, label="Train loss")
    ax.plot(ep, s["val"],   color=PALETTE["surr_val"],   lw=2.2, ls="--", label="Val loss")
    ax.fill_between(ep, s["train"], s["val"], alpha=0.08, color="#999999", label="Gap")
    ax.set_xlabel("Surrogate Epoch"); ax.set_ylabel("Surrogate Loss")
    ax.set_title("Phase 1: Energy Surrogate Pre-training (Total Loss)", fontweight="bold")
    ax.legend(); add_watermark(ax)

    ax = axes[1]
    ax.plot(ep, s["mae"], color=PALETTE["surr_train"], lw=2.2,
            marker="o", ms=4, markevery=5, label="MAE (kcal/mol/atom)")
    ax.fill_between(ep, s["mae"], alpha=0.08, color=PALETTE["surr_train"])
    ax.set_xlabel("Surrogate Epoch"); ax.set_ylabel("MAE (kcal/mol/atom)")
    ax.set_title("Phase 1: Energy Surrogate Pre-training (MAE)", fontweight="bold")
    ax.legend(); add_watermark(ax)

    fig.suptitle("Experiment H — Phase 1: GFN2-xTB Energy Surrogate Learning (1.67M params)",
                 fontweight="bold", fontsize=13, y=1.01)
    fig.tight_layout()
    save(fig, "07_surrogate_pretraining")


# ─── Plot 8: Summary dashboard ───────────────────────────────────────────────
def plot_summary_dashboard(data):
    diff  = data["diffusion"]
    surr  = data["surrogate"]
    evals = data["evals"]

    ep_d   = diff["epochs"]
    ep_s   = surr["epochs"]
    ep_mr  = [e["epoch"] for e in evals if e.get("mat_r") is not None]
    mat_rv = [e["mat_r"]  for e in evals if e.get("mat_r") is not None]
    ep_mee = [e["epoch"] for e in evals if e.get("mee") is not None]
    mee_v  = [e["mee"]   for e in evals if e.get("mee") is not None]

    sota_names  = ["GeoDiff", "TorDiff", "EQGAT-diff", "Exp H"]
    sota_mat_r  = [0.528, 0.481, 0.447, 1.6724]
    sota_colors = [PALETTE["geodiff"], PALETTE["tordiff"], PALETTE["eqgat"], PALETTE["expH"]]

    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor("#FAFAFA")
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            hspace=0.52, wspace=0.38,
                            left=0.06, right=0.97, top=0.90, bottom=0.10)

    # (a) Training loss
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ep_d, diff["train"], color=PALETTE["train"], lw=1.8, label="Train")
    ax.plot(ep_d, diff["val"],   color=PALETTE["val"],   lw=1.8, ls="--", label="Val")
    ax.set_title("(a) Diffusion Training Loss", fontweight="bold", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=8); add_watermark(ax)

    # (b) Component losses
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ep_d, diff["mse"], color=PALETTE["mse"], lw=1.6, label="MSE")
    ax.plot(ep_d, diff["geo"], color=PALETTE["geo"], lw=1.6, ls="--", label="Geo")
    ax.set_title("(b) Loss Components", fontweight="bold", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=8); add_watermark(ax)

    # (c) LR schedule
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(ep_d, diff["lr"], color=PALETTE["lr"], lw=1.8)
    ax.set_yscale("log")
    ax.set_title("(c) LR Schedule (cosine)", fontweight="bold", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR (log)")
    add_watermark(ax)

    # (d) Surrogate pretraining
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(ep_s, surr["train"], color=PALETTE["surr_train"], lw=1.8, label="Train")
    ax.plot(ep_s, surr["val"],   color=PALETTE["surr_val"],   lw=1.8, ls="--", label="Val")
    ax.set_title("(d) Phase 1: Surrogate Pretraining", fontweight="bold", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Surrogate Loss")
    ax.legend(fontsize=8); add_watermark(ax)

    # (e) MAT-R progression
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(ep_mr, mat_rv, "o-", color=PALETTE["mat_r"], lw=2.0, ms=7)
    ax.axhline(0.528, color=PALETTE["geodiff"], lw=1.2, ls="--", alpha=0.6, label="GeoDiff")
    ax.axhline(0.481, color=PALETTE["tordiff"], lw=1.2, ls="-.", alpha=0.6, label="TorDiff")
    ax.set_title("(e) MAT-R Progression ↓", fontweight="bold", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAT-R (Å)")
    ax.legend(fontsize=7); add_watermark(ax)

    # (f) MEE
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(ep_mee, mee_v, "D-", color=PALETTE["mee"], lw=2.0, ms=6)
    ax.axhline(0, color="#333333", lw=0.9, alpha=0.4)
    ax.set_title("(f) MEE Progression", fontweight="bold", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MEE (norm)")
    add_watermark(ax)

    # (g) SOTA bar
    ax = fig.add_subplot(gs[1, 2])
    bars = ax.bar(sota_names, sota_mat_r, color=sota_colors, edgecolor="white", lw=1.2, alpha=0.88)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axvspan(2.5, 3.5, alpha=0.07, color=PALETTE["expH"])
    ax.set_ylabel("MAT-R (Å) ↓")
    ax.set_title("(g) SOTA Comparison", fontweight="bold", fontsize=11)
    ax.set_xticks(range(len(sota_names)))
    ax.set_xticklabels(sota_names, fontsize=9)
    add_watermark(ax)

    # (h) Config table
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    cfg_rows = [
        ["Parameter",          "Value"],
        ["Architecture",       "EGNN + Surrogate"],
        ["Hidden dim",         "384"],
        ["Transformer layers", "8"],
        ["RBF basis",          "32"],
        ["Diffusion steps",    "1000"],
        ["Geometry weight",    "0.3"],
        ["Dataset",            "GEOM-Drugs"],
        ["Train mols",         "86,777"],
        ["Best epoch",         "300"],
        ["Best MAT-R",         "1.6724 Å"],
        ["GPU",                "RTX 3090 · 24 GB"],
    ]
    col_w = [0.55, 0.45]
    row_h = 0.072
    y0    = 0.96
    for ri, row in enumerate(cfg_rows):
        bg = "#2C3E50" if ri == 0 else ("#F0F4F8" if ri%2==0 else "#FFFFFF")
        fc = "white"   if ri == 0 else "black"
        fw = "bold"    if ri == 0 else "normal"
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, y0-(ri+1)*row_h), 1.0, row_h,
            boxstyle="round,pad=0.005", facecolor=bg,
            edgecolor="#cccccc", linewidth=0.4,
            transform=ax.transAxes, clip_on=False))
        x = 0.02
        for ci, (cell, cw) in enumerate(zip(row, col_w)):
            ax.text(x+sum(col_w[:ci]), y0-(ri+0.5)*row_h,
                    cell, ha="left", va="center",
                    fontsize=8.5, color=fc, fontweight=fw,
                    transform=ax.transAxes)
    ax.set_title("(h) Model Configuration", fontweight="bold", fontsize=11)
    add_watermark(ax)

    fig.suptitle(
        "Experiment H — Energy-Guided Diffusion for Molecular Conformer Generation\n"
        "GFN2-xTB Surrogate · EGNN Diffusion · GEOM-Drugs (96 418 molecules) · QM9 Protocol",
        fontweight="bold", fontsize=14, y=0.97)
    save(fig, "08_summary_dashboard")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  Exp H — Publication-Quality Plot Generator")
    print("=" * 62)
    print(f"\n  Log : {LOG_PATH}")
    print(f"  Out : {OUT_DIR}\n")

    if not LOG_PATH.exists():
        print(f"ERROR: log not found at {LOG_PATH}", file=sys.stderr)
        sys.exit(1)

    data = parse_expH_log(LOG_PATH)
    print("\nGenerating plots ...")
    plot_training_val_loss(data)
    plot_component_losses(data)
    plot_lr_schedule(data)
    plot_eval_metrics(data)
    plot_mee_progression(data)
    plot_sota_comparison(data)
    plot_surrogate_pretraining(data)
    plot_summary_dashboard(data)

    print(f"\nAll plots saved to: {OUT_DIR}/")
    print("Files generated:")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  {f.name:45s}  {f.stat().st_size//1024} KB")


if __name__ == "__main__":
    main()
