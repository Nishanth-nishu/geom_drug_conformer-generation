#!/bin/bash
# ============================================================
# docker_run_expF.sh — Run Exp F via Docker on any node
#
# This script pulls nishanthr23/nextmol and runs Exp F.
# No local Python installation needed — everything is in the image.
#
# Requirements:
#   - docker with nvidia-container-toolkit (nvidia-docker2)
#   - OR singularity/apptainer (auto-detected)
#
# Usage (interactive / no SLURM):
#   bash scripts/docker_run_expF.sh
#
# Usage (SLURM, submitting this as a job):
#   sbatch --partition=plafnet2 --account=plafnet2 \
#          --gres=gpu:2 --cpus-per-task=8 --mem=32G \
#          --job-name=expF_docker \
#          --output=logs/expF_docker_%j.log \
#          scripts/docker_run_expF.sh
#
# Data mount strategy:
#   /workspace/data       ← your data directory (read)
#   /workspace/experiments← experiment outputs (read-write)
#   /workspace/logs       ← training logs (read-write)
# ============================================================

set -euo pipefail

IMAGE="nishanthr23/nextmol:latest"
PROJECT_ROOT=/scratch/nishanth.r/nextmol_experiment/mol_next_gen

DATA_DIR="$PROJECT_ROOT/data"
EXP_DIR="$PROJECT_ROOT/experiments"
LOG_DIR="$PROJECT_ROOT/logs"
CKPT_DIR="$PROJECT_ROOT/checkpoints"

mkdir -p "$EXP_DIR" "$LOG_DIR" "$CKPT_DIR"

echo "============================================================"
echo "  mol_next_gen — Docker Runner"
echo "  Image:  $IMAGE"
echo "  Node:   $(hostname)"
echo "  GPUs:   $(nvidia-smi --list-gpus 2>/dev/null | wc -l) available"
echo "  Date:   $(date)"
echo "============================================================"

NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)

# ── Detect container runtime ──────────────────────────────────────────────────
if command -v docker &>/dev/null && docker info &>/dev/null; then
    RUNTIME="docker"
elif command -v singularity &>/dev/null; then
    RUNTIME="singularity"
elif command -v apptainer &>/dev/null; then
    RUNTIME="apptainer"
else
    echo "ERROR: No container runtime found (docker/singularity/apptainer)."
    echo "Install nvidia-docker2 or singularity on this node."
    exit 1
fi

echo "Using runtime: $RUNTIME"

# ── Run ───────────────────────────────────────────────────────────────────────
if [ "$RUNTIME" = "docker" ]; then
    # Pull latest image
    docker pull "$IMAGE"

    docker run \
        --gpus all \
        --rm \
        --shm-size=8g \
        --network=host \
        -e NCCL_DEBUG=WARN \
        -e OMP_NUM_THREADS=4 \
        -v "$DATA_DIR":/workspace/data:ro \
        -v "$EXP_DIR":/workspace/experiments \
        -v "$LOG_DIR":/workspace/logs \
        -v "$CKPT_DIR":/workspace/checkpoints \
        "$IMAGE" \
        bash -c "
            torchrun \
                --nproc_per_node=$NUM_GPUS \
                --master_addr=localhost \
                --master_port=29502 \
                autoresearch/mol_train_ddp.py \
                --exp F \
                --data /workspace/data/qm9_heavy.jsonl \
                2>&1 | tee /workspace/logs/expF_docker_$(date +%Y%m%d_%H%M%S).log
        "

elif [ "$RUNTIME" = "singularity" ] || [ "$RUNTIME" = "apptainer" ]; then
    # Convert Docker image to Singularity SIF (cached)
    SIF_PATH="$PROJECT_ROOT/nextmol_latest.sif"

    if [ ! -f "$SIF_PATH" ]; then
        echo "Building Singularity SIF from Docker Hub (first time, ~10 min)..."
        $RUNTIME pull "$SIF_PATH" docker://"$IMAGE"
    else
        echo "Using cached SIF: $SIF_PATH"
    fi

    $RUNTIME exec \
        --nv \
        --bind "$DATA_DIR":/workspace/data \
        --bind "$EXP_DIR":/workspace/experiments \
        --bind "$LOG_DIR":/workspace/logs \
        --bind "$CKPT_DIR":/workspace/checkpoints \
        "$SIF_PATH" \
        bash -c "
            cd /workspace && \
            torchrun \
                --nproc_per_node=$NUM_GPUS \
                --master_addr=localhost \
                --master_port=29502 \
                autoresearch/mol_train_ddp.py \
                --exp F \
                --data /workspace/data/qm9_heavy.jsonl \
                2>&1 | tee /workspace/logs/expF_docker_$(date +%Y%m%d_%H%M%S).log
        "
fi

echo "============================================================"
echo "  Exp F complete! End: $(date)"
echo "============================================================"
