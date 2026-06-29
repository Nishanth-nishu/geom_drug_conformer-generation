#!/bin/bash
#SBATCH --job-name=mol_expE_sota
#SBATCH --output=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/expE_%j.log
#SBATCH --error=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/expE_%j.log
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=96:00:00
#SBATCH --nodelist=gnode118

# =============================================================================
# Experiment E — SOTA Hybrid Model
#
# Combines the best components from our ablation study:
#   - Flow Matching (faster, more stable integration over many steps)
#   - Attention EGNN (better modeling of complex topologies like rings)
#   - Torsion Auxiliary Loss (OPLS-AA supervision for dihedral stability)
#
# Training is extended to 200 epochs to allow convergence, targeting >85% validity.
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

EXP_NAME="exp_E_sota_hybrid"
EXP_DIR="$PROJ/experiments/$EXP_NAME"
mkdir -p "$EXP_DIR" logs

echo "============================================================"
echo "  Experiment E: SOTA Hybrid Model"
echo "  Job ID    : ${SLURM_JOB_ID:-interactive}"
echo "  Node      : $(hostname)"
echo "  GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Start     : $(date)"
echo "  Dataset   : ${MOL_DATASET:-data/qm9_selfies.jsonl}"
echo "============================================================"
python -c "import torch; print(f'  PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}')"
echo ""

DATASET=${MOL_DATASET:-data/qm9_selfies.jsonl}
if [ ! -f "$PROJ/$DATASET" ]; then
    echo "ERROR: $DATASET not found. Please run the appropriate prepare script first."
    exit 1
fi

LOG="$EXP_DIR/train.log"
echo "Running: python autoresearch/mol_train_expE.py → $LOG"
echo ""
python autoresearch/mol_train_expE.py 2>&1 | tee "$LOG"
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
    "num_params_M":    float(extract("num_params_M", "0")),
    "timestamp":       datetime.datetime.now().isoformat(),
    "model":           "HybridFlowMatchingConformer (CFM+Attn+Torsion)",
    "key_changes":     "Combined SOTA techniques trained for 200 epochs",
}
with open("$EXP_DIR/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))

status = "keep" if metrics["fully_valid"] > 0.01 else ("crash" if $EXIT_CODE != 0 else "discard")
row = (f"nocommit\t{metrics['fully_valid']:.6f}\t{metrics['mat_r']:.6f}\t"
       f"{metrics['rmsd_mean']:.6f}\t{metrics['strain_kcal']:.2f}\t"
       f"{metrics['cov_r']:.6f}\t{metrics['validity']:.6f}\t"
       f"{metrics['bond_error']:.6f}\t{status}\t"
       f"Exp E: SOTA Hybrid CFM+Attn+Torsion 200ep")
with open("$PROJ/autoresearch/results.tsv", "a") as f:
    f.write(row + "\n")
print(f"Appended to results.tsv: {status}")
PYEOF

CKPT_SRC="$PROJ/checkpoints/${EXP_NAME}_best.pt"
if [ -f "$CKPT_SRC" ]; then
    cp "$CKPT_SRC" "$EXP_DIR/checkpoint_best.pt"
    echo "Checkpoint saved: $EXP_DIR/checkpoint_best.pt"
fi

echo ""
echo "============================================================"
echo "  Experiment E complete!"
FVALID=$(grep "^fully_valid:" "$LOG" | tail -1 | awk '{print $2}' 2>/dev/null || echo "N/A")
MATR=$(grep "^mat_r:" "$LOG" | tail -1 | awk '{print $2}' 2>/dev/null || echo "N/A")
echo "  fully_valid = ${FVALID}"
echo "  mat_r       = ${MATR}"
echo "  End: $(date)"
echo "============================================================"
