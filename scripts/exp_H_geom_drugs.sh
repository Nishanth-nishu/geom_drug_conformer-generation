#!/usr/bin/env bash
#
# exp_H_geom_drugs.sh — SLURM script for Experiment H:
#   Energy-Guided Diffusion on GEOM-Drugs with GFN2-xTB Boltzmann Weighting
#
# Architecture: ConformerDiffusion (EGNN, hidden=384, L=8) + EnergySurrogate
# Training:     Two-phase (surrogate pre-training → Boltzmann-weighted diffusion)
# Dataset:      GEOM-Drugs (multi-conformer, heavy atoms, ~100k molecules)
# Evaluation:   COV-R/MAT-R/COV-P/MAT-P (GEOM benchmark protocol)
#
# SOTA Targets (GEOM-Drugs, Table 2 from GeoDiff/TorDiff):
#   GeoDiff (ICML 2022):     COV-R=56.4%, MAT-R=0.528 Å
#   TorDiff (NeurIPS 2022):  COV-R=72.7%, MAT-R=0.481 Å
#   OUR TARGET:              COV-R≥50%,   MAT-R≤0.60 Å (competitive first run)
#
# Usage:
#   sbatch scripts/exp_H_geom_drugs.sh
#   # Or to run interactively with direct path override:
#   bash scripts/exp_H_geom_drugs.sh --data /path/to/geom_drugs.jsonl
#
# Requirements:
#   1. GEOM-Drugs data must be prepared first:
#      python data/prepare_geom_drugs.py \
#          --input /path/to/drugs_crude.msgpack \
#          --output data/geom_drugs.jsonl \
#          --heavy-only --stats
#   2. Virtual environment must be activated or conda env set

#SBATCH --job-name=expH_geom_drugs
#SBATCH --output=logs/expH_%j.log
#SBATCH --error=logs/expH_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Environment setup
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================================"
echo "  Experiment H: Energy-Guided GEOM-Drugs Diffusion"
echo "  Project root: $PROJECT_DIR"
echo "  Node: $(hostname)"
echo "  Date: $(date)"
echo "========================================================"

# Activate virtual environment
if [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
    echo "Activated venv: $PROJECT_DIR/venv"
elif [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "Using conda env: $CONDA_DEFAULT_ENV"
else
    echo "WARNING: No virtual environment found. Using system Python."
fi

# GPU info
if command -v nvidia-smi &>/dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# Data paths
# ─────────────────────────────────────────────────────────────────────────────

GEOM_DATA="${PROJECT_DIR}/data/geom_drugs.jsonl"
MSGPACK_SEARCH=(
    "${PROJECT_DIR}/data/drugs_crude.msgpack"
    "${PROJECT_DIR}/data/geom/drugs_crude.msgpack"
    "/data/geom/drugs_crude.msgpack"
    "/scratch/geom/drugs_crude.msgpack"
)

# Check if data file exists; if not, try to find msgpack and prepare
if [ ! -f "$GEOM_DATA" ]; then
    echo "GEOM-Drugs JSONL not found at $GEOM_DATA"
    echo "Searching for msgpack file..."

    MSGPACK_PATH=""
    for p in "${MSGPACK_SEARCH[@]}"; do
        if [ -f "$p" ]; then
            MSGPACK_PATH="$p"
            echo "  Found: $p"
            break
        fi
    done

    # Also check for drugs_rdkit/ directory
    RDKIT_DIR="${PROJECT_DIR}/data/drugs_rdkit"
    if [ -z "$MSGPACK_PATH" ] && [ -d "$RDKIT_DIR" ]; then
        echo "  Found drugs_rdkit directory: $RDKIT_DIR"
        MSGPACK_PATH="$RDKIT_DIR"
    fi

    if [ -n "$MSGPACK_PATH" ]; then
        echo "Preparing GEOM-Drugs dataset..."
        cd "$PROJECT_DIR"
        python data/prepare_geom_drugs.py \
            --input "$MSGPACK_PATH" \
            --output "$GEOM_DATA" \
            --heavy-only \
            --stats
        echo "Dataset prepared: $GEOM_DATA"
    else
        echo ""
        echo "ERROR: GEOM-Drugs msgpack/directory not found."
        echo ""
        echo "Please download from Harvard Dataverse:"
        echo "  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF"
        echo ""
        echo "Option A (msgpack):"
        echo "  Place drugs_crude.msgpack in $PROJECT_DIR/data/"
        echo "  Then run: python data/prepare_geom_drugs.py --input data/drugs_crude.msgpack"
        echo ""
        echo "Option B (rdkit pickles):"
        echo "  Extract drugs_rdkit.tar.gz to $PROJECT_DIR/data/drugs_rdkit/"
        echo "  Then run: python data/prepare_geom_drugs.py --input data/drugs_rdkit/"
        echo ""
        exit 1
    fi
fi

echo "Dataset: $GEOM_DATA"
wc -l "$GEOM_DATA" | awk '{print "  Lines (molecules):", $1}'

# ─────────────────────────────────────────────────────────────────────────────
# Create output directories
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/checkpoints"
mkdir -p "$PROJECT_DIR/experiments/exp_H_geom_drugs_gfn2"

# ─────────────────────────────────────────────────────────────────────────────
# Run training
# ─────────────────────────────────────────────────────────────────────────────

cd "$PROJECT_DIR"

echo ""
echo "Starting Experiment H training..."
echo "  Phase 1: Energy Surrogate Pre-training (GFN2-xTB energies)"
echo "  Phase 2: Boltzmann-Weighted Diffusion Training"
echo "  Phase 3: Energy-Guided DDIM Evaluation"
echo ""

PYTHONPATH="$PROJECT_DIR" python autoresearch/mol_train_expH.py \
    --data "$GEOM_DATA"

EXIT_CODE=$?

echo ""
echo "========================================================"
echo "  Experiment H completed (exit code: $EXIT_CODE)"
echo "  Results: experiments/exp_H_geom_drugs_gfn2/final_metrics.json"
echo "  Best checkpoints: checkpoints/exp_H_geom_drugs_gfn2_best*.pt"
echo "========================================================"

exit $EXIT_CODE
