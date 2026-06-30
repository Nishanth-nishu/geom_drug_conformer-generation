#!/bin/bash
# =============================================================================
# submit_v2_full.sh — GeoDiff-Novel  |  max_heavy=100  |  Full GEOM-Drugs
# FIXED: absolute data path + 5000 timesteps (GeoDiff canonical for Drugs)
# =============================================================================
#SBATCH --job-name=geom_v2_full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/nishanth.r/nextmol_experiment/mol_expermiments/logs/geom_v2_full_%j.log
#SBATCH --error=/scratch/nishanth.r/nextmol_experiment/mol_expermiments/logs/geom_v2_full_%j.log

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "SLURM_JOB_GPUS = $SLURM_JOB_GPUS"
echo "=========================================="
echo "Job    : DualEncoderDiffusion v2 — GEOM-Full (N<=100, all 96K mols)"
echo "Node   : $(hostname)"
echo "Started: $(date)"
echo "GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=========================================="

cd /scratch/nishanth.r/nextmol_experiment/mol_expermiments
source venv/bin/activate

# Absolute path — prevents mid-training FileNotFoundError on cluster
DATA=/scratch/nishanth.r/nextmol_experiment/mol_expermiments/data/geom_drugs.jsonl

if [ ! -f "$DATA" ]; then
    echo "[ERROR] Dataset not found: $DATA"
    exit 1
fi
echo "Dataset: $DATA ($(wc -l < $DATA) molecules)"

# ── Smoke test first ────────────────────────────────────────────────────────
echo ""
echo "Step 1: Smoke test ..."
PYTHONPATH=. python3 autoresearch/mol_train_v2.py \
    --smoke-test \
    --max-atoms  100 \
    --max-mols   -1 \
    --max-confs  10 \
    --batch-size 16 \
    --lr         1e-3 \
    --hidden-dim 256 \
    --num-timesteps 5000 \
    --data       $DATA

if [ $? -ne 0 ]; then
    echo "[ERROR] Smoke test failed — aborting."
    exit 1
fi
echo "Smoke test passed."

# ── Full training ───────────────────────────────────────────────────────────
echo ""
echo "Step 2: Full training (300 epochs) ..."
PYTHONPATH=. python3 autoresearch/mol_train_v2.py \
    --max-atoms  100 \
    --max-mols   -1 \
    --max-confs  10 \
    --batch-size 16 \
    --lr         1e-3 \
    --hidden-dim 256 \
    --num-timesteps 5000 \
    --epochs     300 \
    --data       $DATA

echo ""
echo "Job completed: $(date)"
