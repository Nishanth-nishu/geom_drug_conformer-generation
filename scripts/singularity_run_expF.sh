#!/bin/bash
# ============================================================
# singularity_run_expF.sh — Pull from Docker Hub via Singularity
#                            and run Exp F on any SLURM node
#
# HOW IT WORKS:
#   1. Pulls nishanthr23/nextmol from Docker Hub → converts to .sif
#   2. Rsyncs data from gnode118 if not present locally
#   3. Runs Exp F inside the container with --nv GPU passthrough
#
# SINGULARITY is available on all plafnet2 nodes:
#   /usr/local/apps/singularity-ce-4.2.2/bin/singularity
#
# Usage:
#   # Submit from login node to ANY plafnet2 node (no gnode118 restriction):
#   sbatch --partition=plafnet2 --account=plafnet2 \
#          --gres=gpu:2 --cpus-per-task=8 --mem=32G \
#          --time=4-00:00:00 \
#          scripts/singularity_run_expF.sh
#
#   # Or run directly on a node you're already on:
#   bash scripts/singularity_run_expF.sh
# ============================================================

#SBATCH --job-name=expF_singularity
#SBATCH --output=/tmp/expF_sing_%j.log
#SBATCH --error=/tmp/expF_sing_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
# NOTE: No --nodelist — runs on any available plafnet2 node with 2 GPUs

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
DOCKER_IMAGE="docker://nishanthr23/nextmol:latest"
SOURCE_NODE="gnode118"
SOURCE_DIR="/scratch/nishanth.r/nextmol_experiment/mol_next_gen"

# Per-node local working directory
NODE_WORKDIR="/scratch/nishanth.r/nextmol_experiment/mol_next_gen"
SIF_PATH="/scratch/nishanth.r/nextmol_latest.sif"  # shared-ish location

SINGULARITY=/usr/local/apps/singularity-ce-4.2.2/bin/singularity

echo "============================================================"
echo "  mol_next_gen — Singularity Runner"
echo "  Node     : $(hostname)"
echo "  Job ID   : ${SLURM_JOB_ID:-local}"
echo "  GPUs     : $(nvidia-smi --list-gpus | wc -l)"
echo "  Date     : $(date)"
echo "============================================================"

# ── Step 1: Data sync (if not gnode118) ──────────────────────────────────────
if [ "$(hostname)" != "gnode118" ] && [ "$(hostname)" != "gnode118.local" ]; then
    echo ""
    echo "--- Not gnode118 — syncing code + data ---"
    mkdir -p "$NODE_WORKDIR/data" "$NODE_WORKDIR/experiments" \
             "$NODE_WORKDIR/logs" "$NODE_WORKDIR/checkpoints"

    # Sync code (fast, small)
    rsync -az --progress \
        -e "ssh -o StrictHostKeyChecking=no" \
        --exclude="venv/" --exclude="data/*.jsonl" \
        --exclude="experiments/" --exclude="logs/" \
        --exclude="checkpoints/" --exclude=".git/" \
        "${SOURCE_NODE}:${SOURCE_DIR}/" "$NODE_WORKDIR/" 2>/dev/null \
        && echo "  Code synced from $SOURCE_NODE" \
        || { echo "  WARNING: Code rsync failed — using git clone"; \
             git clone https://github.com/Nishanth-nishu/mol_expermiments.git "$NODE_WORKDIR" 2>/dev/null || true; }

    # Sync data (large, ~70MB each)
    for dataset in qm9_heavy.jsonl qm9_selfies.jsonl; do
        dst="$NODE_WORKDIR/data/$dataset"
        if [ ! -f "$dst" ]; then
            echo "  Syncing $dataset..."
            rsync -az \
                -e "ssh -o StrictHostKeyChecking=no" \
                "${SOURCE_NODE}:${SOURCE_DIR}/data/${dataset}" "$dst" 2>/dev/null \
                && echo "  OK: $dataset ($(wc -l < "$dst") molecules)" \
                || echo "  WARNING: Could not sync $dataset"
        else
            echo "  Already present: $dataset"
        fi
    done
else
    echo "  On gnode118 — using local /scratch directly"
fi

# ── Step 2: Pull Docker image → Singularity SIF ───────────────────────────────
echo ""
echo "--- Singularity: Pull $DOCKER_IMAGE ---"
if [ ! -f "$SIF_PATH" ]; then
    echo "  First-time pull (~5-10 min, converts Docker layers to SIF)..."
    $SINGULARITY pull --force "$SIF_PATH" "$DOCKER_IMAGE"
    echo "  SIF ready: $SIF_PATH ($(du -sh "$SIF_PATH" | cut -f1))"
else
    # Check if image is stale (older than 7 days)
    if find "$SIF_PATH" -mtime +7 -print | grep -q .; then
        echo "  SIF is >7 days old — refreshing..."
        $SINGULARITY pull --force "$SIF_PATH" "$DOCKER_IMAGE"
    else
        echo "  Using cached SIF: $SIF_PATH"
    fi
fi

# ── Step 3: Run Exp F inside Singularity ─────────────────────────────────────
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
EXP_DIR="$NODE_WORKDIR/experiments/exp_F_heavy_atom"
mkdir -p "$EXP_DIR" "$NODE_WORKDIR/logs" "$NODE_WORKDIR/checkpoints"

LOG_FILE="$NODE_WORKDIR/logs/expF_sing_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}.log"

echo ""
echo "--- Running Exp F: $NUM_GPUS GPU(s) ---"
echo "--- Log: $LOG_FILE ---"

$SINGULARITY exec \
    --nv \
    --bind "$NODE_WORKDIR/data":/workspace/data \
    --bind "$NODE_WORKDIR/experiments":/workspace/experiments \
    --bind "$NODE_WORKDIR/logs":/workspace/logs \
    --bind "$NODE_WORKDIR/checkpoints":/workspace/checkpoints \
    --env PYTHONPATH=/workspace \
    --env NCCL_DEBUG=WARN \
    --env OMP_NUM_THREADS=4 \
    "$SIF_PATH" \
    bash -c "
        torchrun \
            --nproc_per_node=$NUM_GPUS \
            --master_addr=localhost \
            --master_port=29504 \
            /workspace/autoresearch/mol_train_ddp.py \
            --exp F \
            --data /workspace/data/qm9_heavy.jsonl
    " 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "  DONE | $(date)"
echo "  Log saved: $LOG_FILE"
echo "============================================================"

# Copy results back to gnode118 if we're on a different node
if [ "$(hostname)" != "gnode118" ] && [ "$(hostname)" != "gnode118.local" ]; then
    echo "--- Copying results back to gnode118 ---"
    rsync -az \
        -e "ssh -o StrictHostKeyChecking=no" \
        "$NODE_WORKDIR/experiments/" \
        "${SOURCE_NODE}:${SOURCE_DIR}/experiments/" 2>/dev/null \
        && echo "  Results synced back to gnode118" \
        || echo "  WARNING: Results sync failed — check $LOG_FILE on $(hostname)"
fi
