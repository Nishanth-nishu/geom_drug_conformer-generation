#!/bin/bash
# =============================================================================
# submit_v2_fixed.sh — DualEncoderDiffusion v2 | Fixed Architecture
# =============================================================================
# Changes from previous failed run (v2_confs5_2637673):
#   1. CosineAnnealingWarmRestarts (T_0=50) instead of ReduceLROnPlateau
#      (LR no longer collapses to 2e-6 by epoch 200)
#   2. MLPEdgeEncoder (multiplicative gating = GeoDiff design)
#   3. Absolute data path (no more FileNotFoundError mid-training)
#   4. 5000 timesteps (GeoDiff canonical for drugs, was 2000)
# =============================================================================
#SBATCH --job-name=geom_v2_fixed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratch/nishanth.r/nextmol_experiment/mol_expermiments/logs/geom_v2_fixed_%j.log
#SBATCH --error=/scratch/nishanth.r/nextmol_experiment/mol_expermiments/logs/geom_v2_fixed_%j.log

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "SLURM_JOB_GPUS = $SLURM_JOB_GPUS"
echo "=========================================="
echo "Job    : DualEncoderDiffusion v2 FIXED"
echo "Node   : $(hostname)"
echo "Started: $(date)"
echo "GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Fixes  : MLPEdgeEncoder + CosineAnnealing + AbsolutePath + T=5000"
echo "=========================================="

cd /scratch/nishanth.r/nextmol_experiment/mol_expermiments
source venv/bin/activate

# Use absolute path — critical fix to prevent mid-training FileNotFoundError
DATA=/scratch/nishanth.r/nextmol_experiment/mol_expermiments/data/geom_drugs.jsonl

if [ ! -f "$DATA" ]; then
    echo "[ERROR] Dataset not found: $DATA"
    exit 1
fi
echo "Dataset: $DATA ($(wc -l < $DATA) molecules)"

# ── Smoke test (5 epochs, validate pipeline) ──────────────────────────────────
echo ""
echo "Step 1: Smoke test (5 epochs) ..."
PYTHONPATH=. python3 autoresearch/mol_train_v2.py \
    --smoke-test \
    --max-atoms  25 \
    --max-mols   30000 \
    --max-confs  10 \
    --batch-size 64 \
    --lr         1e-3 \
    --hidden-dim 256 \
    --num-timesteps 5000 \
    --data       $DATA

if [ $? -ne 0 ]; then
    echo "[ERROR] Smoke test failed — aborting."
    exit 1
fi

echo "Smoke test passed."

# ── Full training ──────────────────────────────────────────────────────────────
echo ""
echo "Step 2: Full training (500 epochs) ..."
PYTHONPATH=. python3 autoresearch/mol_train_v2.py \
    --max-atoms  25 \
    --max-mols   30000 \
    --max-confs  10 \
    --batch-size 64 \
    --lr         1e-3 \
    --hidden-dim 256 \
    --num-timesteps 5000 \
    --epochs     500 \
    --data       $DATA

echo ""
echo "Job completed: $(date)"
