#!/bin/bash
#SBATCH --job-name=mol_expC_cfm
#SBATCH --output=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/expC_%j.log
#SBATCH --error=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/expC_%j.log
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=96:00:00
#SBATCH --nodelist=gnode118

# =============================================================================
# Experiment C — Conditional Flow Matching (CFM)
#
# Research goal: Replace DDPM with CFM for faster, more accurate generation.
# Model:         FlowMatchingConformer (EGNN backbone, Euler ODE sampler)
# Key changes:
#   - Training loss: CFM MSE || v_pred - (eps - x0) ||^2
#   - Sampling: 20-step Euler ODE (vs DDIM 50 steps)
#   - No noise schedule tuning (simpler, fewer hyperparameters)
#   - Bug fixes applied: correct argument order + x0_hat broadcast
#
# Hypothesis: Straight ODE trajectories (Lipman 2023) → faster convergence
# and better geometry accuracy vs curved stochastic DDPM paths.
# At 50 epochs we expect equal or better fully_valid vs Exp A at 2x inference speed.
#
# Expected: fully_valid >= Exp A, 4x faster inference
# Runtime:  ~2-3h (same as Exp A, inference is faster)
#
# Reference: Lipman et al. "Flow Matching for Generative Modeling" ICLR 2023.
#            arXiv:2210.02747
# =============================================================================

set -euo pipefail

PROJ=/scratch/nishanth.r/nextmol_experiment/mol_next_gen
cd "$PROJ"

export PIP_CACHE_DIR=/scratch/nishanth.r/pip_cache
export HF_HOME=/scratch/nishanth.r/hf_cache
export TORCH_HOME=/scratch/nishanth.r/torch_cache
export TMPDIR=/scratch/nishanth.r/tmp
export PYTHONPATH="$PROJ"

source venv/bin/activate

EXP_NAME="exp_C_flow_matching"
EXP_DIR="$PROJ/experiments/$EXP_NAME"
mkdir -p "$EXP_DIR" logs

echo "============================================================"
echo "  Experiment C: Conditional Flow Matching"
echo "  Job ID    : ${SLURM_JOB_ID:-interactive}"
echo "  Node      : $(hostname)"
echo "  GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Start     : $(date)"
echo "============================================================"
python -c "import torch; print(f'  PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}')"
echo ""

if [ ! -f "$PROJ/data/qm9_selfies.jsonl" ]; then
    echo "ERROR: data/qm9_selfies.jsonl not found. Run prepare_qm9.py first."
    exit 1
fi

python -c "
from models.flow_matching import FlowMatchingConformer
from autoresearch.mol_prepare import make_dataloaders
print('  All imports OK')
m = FlowMatchingConformer(hidden_dim=256, num_layers=6)
n = sum(p.numel() for p in m.parameters())/1e6
print(f'  Model params: {n:.2f}M')
# Quick forward test
import torch
at = torch.tensor([6,6,8], dtype=torch.long)
ei = torch.tensor([[0,1,1,2],[1,0,2,1]], dtype=torch.long)
bt = torch.tensor([1,1,1,1], dtype=torch.long)
bi = torch.tensor([0,0,0], dtype=torch.long)
co = torch.randn(3, 3)
ld = m.get_loss(co, at, ei, bt, bi)
print(f'  Forward pass OK, loss={ld[\"total\"].item():.4f}')
"

LOG="$EXP_DIR/train.log"
echo "Running: python autoresearch/mol_train_expC.py → $LOG"
echo ""
python autoresearch/mol_train_expC.py 2>&1 | tee "$LOG"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "Training exit code: $EXIT_CODE"

python3 - <<PYEOF
import json, datetime, re

log_text = open("$LOG").read()

def extract(pattern, default="0"):
    m = re.search(rf'^{pattern}:\s*(\S+)', log_text, re.MULTILINE)
    return m.group(1) if m else default

metrics = {
    "exp_name":        "$EXP_NAME",
    "exit_code":       $EXIT_CODE,
    "fully_valid":     float(extract("fully_valid", "0")),
    "mat_r":           float(extract("mat_r", "999")),
    "rmsd_mean":       float(extract("rmsd_mean", "999")),
    "strain_kcal":     float(extract("strain_kcal", "0")),
    "cov_r":           float(extract("cov_r", "0")),
    "validity":        float(extract("validity", "0")),
    "bond_error":      float(extract("bond_error", "0")),
    "training_secs":   float(extract("training_secs", "0")),
    "peak_vram_mb":    float(extract("peak_vram_mb", "0")),
    "num_params_M":    float(extract("num_params_M", "0")),
    "ode_steps":       int(extract("ode_steps", "20")),
    "timestamp":       datetime.datetime.now().isoformat(),
    "model":           "FlowMatchingConformer (CFM, Euler ODE 20 steps)",
    "key_changes":     "Replaced DDPM with Conditional Flow Matching (Lipman ICLR 2023)",
}
with open("$EXP_DIR/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))

status = "keep" if metrics["fully_valid"] > 0.01 else ("crash" if $EXIT_CODE != 0 else "discard")
row = (f"nocommit\t{metrics['fully_valid']:.6f}\t{metrics['mat_r']:.6f}\t"
       f"{metrics['rmsd_mean']:.6f}\t{metrics['strain_kcal']:.2f}\t"
       f"{metrics['cov_r']:.6f}\t{metrics['validity']:.6f}\t"
       f"{metrics['bond_error']:.6f}\t{status}\t"
       f"Exp C: flow matching CFM Euler ODE 20steps 50ep")
with open("$PROJ/autoresearch/results.tsv", "a") as f:
    f.write(row + "\n")
print(f"Appended to results.tsv: {status}")
PYEOF

CKPT_SRC="$PROJ/checkpoints/${EXP_NAME}_best.pt"
if [ -f "$CKPT_SRC" ]; then
    cp "$CKPT_SRC" "$EXP_DIR/checkpoint_best.pt"
fi

echo ""
echo "============================================================"
echo "  Experiment C complete!"
FVALID=$(grep "^fully_valid:" "$LOG" | tail -1 | awk '{print $2}' 2>/dev/null || echo "N/A")
MATR=$(grep "^mat_r:" "$LOG" | tail -1 | awk '{print $2}' 2>/dev/null || echo "N/A")
echo "  fully_valid = ${FVALID}"
echo "  mat_r       = ${MATR}"
echo "  End: $(date)"
echo "============================================================"
