#!/bin/bash
# =============================================================================
# submit_v2_small.sh — GeoDiff-Novel  |  max_heavy=25  |  GEOM-Small subset
# =============================================================================
# Goal: Fastest convergence proof-of-concept.
# Molecules with <= 25 heavy atoms: ~30,000 mols in GEOM-Drugs.
# Expected: COV-R > 20% by epoch 100, approaching GeoDiff by epoch 300.
#
# Why 25 atoms?
#   - Distribution closest to QM9 (N<=9) — smallest generalization gap
#   - Radius graph has ~50-150 edges vs 300-1000 for 100-atom mols
#   - Fits in 8G GPU RAM with batch=64
#   - Fastest iteration: ~3 min/epoch → can validate in 2-3 days
# =============================================================================
#SBATCH --job-name=geom_v2_small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/geom_v2_small_%j.log
#SBATCH --error=logs/geom_v2_small_%j.log

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "SLURM_JOB_GPUS = $SLURM_JOB_GPUS"
echo "=========================================="
echo "Job    : DualEncoderDiffusion v2 — GEOM-Small (N<=25)"
echo "Node   : $(hostname)"
echo "Started: $(date)"
echo "GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=========================================="

cd /scratch/nishanth.r/nextmol_experiment/mol_expermiments
source venv/bin/activate

# ── Smoke test first (5 epochs, <10 min) ──────────────────────────────────────
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
    --num-timesteps 2000 \
    --data       data/geom_drugs.jsonl

echo "Smoke test done."

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
    --num-timesteps 2000 \
    --epochs     500 \
    --data       data/geom_drugs.jsonl

echo ""
echo "Job completed: $(date)"
