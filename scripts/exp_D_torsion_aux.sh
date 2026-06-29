#!/bin/bash
#SBATCH --job-name=mol_expD_tors
#SBATCH --output=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/expD_%j.log
#SBATCH --error=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/expD_%j.log
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=96:00:00
#SBATCH --nodelist=gnode118

# =============================================================================
# Experiment D — Torsion-Angle Auxiliary Loss (TorDiff-inspired)
#
# Research goal: Does direct dihedral supervision reduce MAT-R?
# Model:         ConformerDiffusion (same as Exp A baseline)
# Key changes:
#   - geometry_weight = 0.5 (increased from 0.1)
#   - INCLUDE_TORSIONS = True: adds OPLS-AA torsion energy loss to training
#   - Torsion loss computed on ground-truth coordinates each batch
#
# Hypothesis: Bond/angle supervision alone is insufficient for conformer
# accuracy. The dominant source of MAT-R error is torsion angle deviation.
# TorDiff showed 30-40% lower MAT-R by explicitly supervising dihedral angles.
# We apply a similar torsion auxiliary loss within the existing DDPM framework.
#
# Expected: 15-25% lower MAT-R vs Exp A (same fully_valid or slightly lower)
# Runtime:  ~3-4h (torsion computation adds O(bonds) overhead per batch)
#
# Reference: Jing et al. "Torsional Diffusion for Molecular Conformer
# Generation" NeurIPS 2022. arXiv:2206.01729
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

EXP_NAME="exp_D_torsion_aux"
EXP_DIR="$PROJ/experiments/$EXP_NAME"
mkdir -p "$EXP_DIR" logs

echo "============================================================"
echo "  Experiment D: Torsion-Angle Auxiliary Loss"
echo "  Job ID    : ${SLURM_JOB_ID:-interactive}"
echo "  Node      : $(hostname)"
echo "  GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Start     : $(date)"
echo "  Key change: geometry_weight=0.5, torsion_loss=ENABLED (OPLS-AA)"
echo "============================================================"
python -c "import torch; print(f'  PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}')"
echo ""

if [ ! -f "$PROJ/data/qm9_selfies.jsonl" ]; then
    echo "ERROR: data/qm9_selfies.jsonl not found. Run prepare_qm9.py first."
    exit 1
fi

python -c "
from models.conformer_diffusion import ConformerDiffusion
from models.geometry_constraints import GeometryConstraints
from autoresearch.mol_prepare import make_dataloaders
print('  All imports OK')
gc = GeometryConstraints()
import torch
pos = torch.randn(6, 3)
at = torch.tensor([6,6,6,8,7,6])
ei = torch.tensor([[0,1,1,2,2,3,3,4,4,5],[1,0,2,1,3,2,4,3,5,4]])
bt = torch.tensor([1,1,1,1,2,2,1,1,1,1])
bi = torch.zeros(6, dtype=torch.long)
tl = gc.compute_torsion_loss(pos, at, ei, bt, bi)
print(f'  Torsion loss smoke test: {tl.item():.4f} OK')
"

LOG="$EXP_DIR/train.log"
echo "Running: python autoresearch/mol_train_expD.py → $LOG"
echo ""
python autoresearch/mol_train_expD.py 2>&1 | tee "$LOG"
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
    "exp_name":           "$EXP_NAME",
    "exit_code":          $EXIT_CODE,
    "fully_valid":        float(extract("fully_valid", "0")),
    "mat_r":              float(extract("mat_r", "999")),
    "rmsd_mean":          float(extract("rmsd_mean", "999")),
    "strain_kcal":        float(extract("strain_kcal", "0")),
    "cov_r":              float(extract("cov_r", "0")),
    "validity":           float(extract("validity", "0")),
    "bond_error":         float(extract("bond_error", "0")),
    "training_secs":      float(extract("training_secs", "0")),
    "peak_vram_mb":       float(extract("peak_vram_mb", "0")),
    "num_params_M":       float(extract("num_params_M", "0")),
    "include_torsions":   str(extract("include_torsions", "True")),
    "timestamp":          datetime.datetime.now().isoformat(),
    "model":              "ConformerDiffusion + torsion aux loss (geo_w=0.5)",
    "key_changes":        "OPLS-AA torsion auxiliary loss, geometry_weight=0.5 (Jing NeurIPS 2022)",
}
with open("$EXP_DIR/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))

status = "keep" if metrics["fully_valid"] > 0.01 else ("crash" if $EXIT_CODE != 0 else "discard")
row = (f"nocommit\t{metrics['fully_valid']:.6f}\t{metrics['mat_r']:.6f}\t"
       f"{metrics['rmsd_mean']:.6f}\t{metrics['strain_kcal']:.2f}\t"
       f"{metrics['cov_r']:.6f}\t{metrics['validity']:.6f}\t"
       f"{metrics['bond_error']:.6f}\t{status}\t"
       f"Exp D: torsion aux loss geo_w=0.5 OPLS-AA 50ep")
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
echo "  Experiment D complete!"
FVALID=$(grep "^fully_valid:" "$LOG" | tail -1 | awk '{print $2}' 2>/dev/null || echo "N/A")
MATR=$(grep "^mat_r:" "$LOG" | tail -1 | awk '{print $2}' 2>/dev/null || echo "N/A")
echo "  fully_valid = ${FVALID}"
echo "  mat_r       = ${MATR}"
echo "  End: $(date)"
echo "============================================================"
