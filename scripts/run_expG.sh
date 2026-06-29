#!/bin/bash
# run_expG.sh — Launch Experiment G training robustly
# Usage: bash scripts/run_expG.sh [&]

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV=/scratch/nishanth.r/nextmol_experiment/GeoDiff/venv

echo "[run_expG] Project: $PROJECT_DIR"
echo "[run_expG] Venv: $VENV"
echo "[run_expG] Started: $(date)"

source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT_DIR"
export MOL_DATASET="$PROJECT_DIR/data/qm9_heavy.jsonl"
export MOL_MAX_ATOMS=9

mkdir -p "$PROJECT_DIR/logs"
LOG="$PROJECT_DIR/logs/expG_$(date +%Y%m%d_%H%M%S).log"
echo "[run_expG] Log: $LOG"

python -u "$PROJECT_DIR/autoresearch/mol_train_expG.py" 2>&1 | tee "$LOG"

echo "[run_expG] Done: $(date)"
