#!/bin/bash
#SBATCH --job-name=geom_v2_dual
#SBATCH --partition=plafnet2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/geom_v2_%j.log
#SBATCH --error=logs/geom_v2_%j.log

# ── STOP OLD JOBS FIRST ──────────────────────────────────────────────────────
# Before submitting this script, cancel old failing jobs:
#   scancel <jobid_geom_baseline>    (currently running, COV-R=0% after 134 epochs)
#   scancel <jobid_expH>             (if running)
# Then submit this:
#   sbatch submit_v2.sh
# ─────────────────────────────────────────────────────────────────────────────

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "SLURM_JOB_GPUS = $SLURM_JOB_GPUS"
echo "=========================================="
echo "============================================================"
echo "Job started : $(hostname) at $(date)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Architecture: DualEncoderDiffusion (NOVEL — distance-space score matching)"
echo "Key changes vs old model:"
echo "  ✓ Distance-space loss (not Cartesian x0 MSE) — THIS IS WHY OLD MODEL GAVE COV-R=0%"
echo "  ✓ Dual graph: GIN local + SchNet+ET global with radius graph"
echo "  ✓ Sigmoid noise schedule (2000 steps, GeoDiff GEOM-Drugs config)"
echo "  ✓ Boltzmann conformer weighting (our novel contribution)"
echo "  ✓ AdamW optimizer (β1=0.95, β2=0.999, grad_norm=30000)"
echo "============================================================"

cd /scratch/nishanth.r/nextmol_experiment/mol_expermiments
source venv/bin/activate

# ── Configuration ─────────────────────────────────────────────────────────────
MAX_ATOMS=30
MAX_MOLS=50000
MAX_CONFS=10
HIDDEN_DIM=256
NUM_TIMESTEPS=2000
BATCH_SIZE=32
EPOCHS=300

echo "Dataset     : data/geom_drugs.jsonl (96418 molecules)"
echo "Max atoms   : $MAX_ATOMS"
echo "Max mols    : $MAX_MOLS"
echo "Max confs   : $MAX_CONFS"
echo "Hidden dim  : $HIDDEN_DIM"
echo "Timesteps   : $NUM_TIMESTEPS (sigmoid schedule)"
echo "Batch size  : $BATCH_SIZE"
echo "Epochs      : $EPOCHS"
echo "============================================================"

# ── Smoke test first (validates pipeline, ~20 min) ────────────────────────────
echo "Running smoke test (5 epochs) ..."
echo "============================================================"
PYTHONPATH=. python autoresearch/mol_train_v2.py \
    --smoke-test \
    --max-atoms $MAX_ATOMS \
    --max-mols $MAX_MOLS \
    --max-confs $MAX_CONFS \
    --hidden-dim $HIDDEN_DIM \
    --num-timesteps $NUM_TIMESTEPS \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS

SMOKE_EXIT=$?
if [ $SMOKE_EXIT -ne 0 ]; then
    echo "Smoke test FAILED (exit $SMOKE_EXIT). Aborting."
    exit 1
fi

echo "Smoke test passed! Starting full training ..."
echo "============================================================"
echo "Full training: $EPOCHS epochs"
echo "============================================================"

PYTHONPATH=. python autoresearch/mol_train_v2.py \
    --max-atoms $MAX_ATOMS \
    --max-mols $MAX_MOLS \
    --max-confs $MAX_CONFS \
    --hidden-dim $HIDDEN_DIM \
    --num-timesteps $NUM_TIMESTEPS \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS

echo "============================================================"
echo "Job finished : $(date)"
echo "============================================================"
