#!/bin/bash
# =============================================================================
# submit_v2_medium.sh — GeoDiff-Novel  |  max_heavy=50  |  GEOM-Medium subset
# =============================================================================
# Goal: Main experiment matching GeoDiff paper's test distribution.
# Molecules with <= 50 heavy atoms: ~75,000 mols in GEOM-Drugs.
# This is the PUBLICATION-QUALITY experiment.
#
# Why 50 atoms?
#   - GeoDiff paper evaluates on this range (most drug-like molecules)
#   - Comparable to TorDiff and ConfGF benchmark splits
#   - Realistic drug-like complexity: rings, stereocenters, macrocycles
#   - ~6-8 min/epoch on RTX 3090 → completes 300 epochs in ~36-48h
#
# Novel contributions vs GeoDiff:
#   1. Equivariant Attention (TorchMD-ET style) — no attention in GeoDiff
#   2. Boltzmann-weighted training loss — our contribution
#   3. Energy-guided DDIM inference (GFN2-xTB surrogate)
#   4. Dual graph with learnable edge type embeddings across bond orders
# =============================================================================
#SBATCH --job-name=geom_v2_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/geom_v2_medium_%j.log
#SBATCH --error=logs/geom_v2_medium_%j.log

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "SLURM_JOB_GPUS = $SLURM_JOB_GPUS"
echo "=========================================="
echo "Job    : DualEncoderDiffusion v2 — GEOM-Medium (N<=50)"
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
    --max-atoms  50 \
    --max-mols   50000 \
    --max-confs  10 \
    --batch-size 32 \
    --lr         1e-3 \
    --hidden-dim 256 \
    --num-timesteps 2000 \
    --data       data/geom_drugs.jsonl

echo "Smoke test done."

# ── Full training ───────────────────────────────────────────────────────────
echo ""
echo "Step 2: Full training (300 epochs) ..."
PYTHONPATH=. python3 autoresearch/mol_train_v2.py \
    --max-atoms  50 \
    --max-mols   50000 \
    --max-confs  10 \
    --batch-size 32 \
    --lr         1e-3 \
    --hidden-dim 256 \
    --num-timesteps 2000 \
    --epochs     300 \
    --data       data/geom_drugs.jsonl

echo ""
echo "Job completed: $(date)"
