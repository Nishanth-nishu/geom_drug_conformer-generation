#!/bin/bash
# =============================================================================
# submit_v2_full.sh — GeoDiff-Novel  |  max_heavy=100  |  Full GEOM-Drugs
# =============================================================================
# Goal: Full GEOM-Drugs benchmark matching GeoDiff paper exactly.
# All 96,418 molecules (up to 100 heavy atoms).
#
# Why 100 atoms (full)?
#   - This is the EXACT benchmark GeoDiff/TorDiff/ConfGF papers use
#   - Direct apples-to-apples comparison for paper submission
#   - Harder: longer training (~12-15 min/epoch)
#   - Submit this AFTER small/medium confirm COV-R > 0%
#
# IMPORTANT: Run small and medium first. Only submit this if:
#   - small experiment gives COV-R > 10% by epoch 50
#   - medium experiment gives COV-R > 5% by epoch 50
# =============================================================================
#SBATCH --job-name=geom_v2_full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/geom_v2_full_%j.log
#SBATCH --error=logs/geom_v2_full_%j.log

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
    --num-timesteps 2000 \
    --data       data/geom_drugs.jsonl

echo "Smoke test done."

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
    --num-timesteps 2000 \
    --epochs     300 \
    --data       data/geom_drugs.jsonl

echo ""
echo "Job completed: $(date)"
