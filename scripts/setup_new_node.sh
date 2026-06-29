#!/bin/bash
# ============================================================
# setup_new_node.sh — Bootstrap mol_next_gen on ANY new node
#
# CONTEXT: /scratch is LOCAL to each node (not NFS-shared).
# This script installs everything from scratch on a new node.
#
# TWO MODES:
#   Mode A (Docker, recommended if available):
#     - Pulls nishanthr23/nextmol from Docker Hub
#     - Rsyncs data from gnode118
#     - Runs container directly
#
#   Mode B (Git + venv, if no Docker):
#     - git clone from GitHub
#     - Creates venv + installs all deps
#     - Rsyncs data from gnode118
#     - Ready for SLURM or direct torchrun
#
# Usage (run on the NEW node, or submit as a SLURM prolog job):
#   bash setup_new_node.sh [TARGET_NODE] [MODE]
#
# Examples:
#   bash setup_new_node.sh                    # auto-detect mode on current node
#   bash setup_new_node.sh gnode115 docker    # setup gnode115 via Docker
#   bash setup_new_node.sh gnode115 venv      # setup gnode115 via git+venv
#   ssh gnode115 "bash /path/to/setup_new_node.sh"  # run remotely
# ============================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_REPO="https://github.com/Nishanth-nishu/mol_expermiments.git"
DOCKER_IMAGE="nishanthr23/nextmol:latest"
SOURCE_NODE="gnode118"
SOURCE_PROJECT="/scratch/nishanth.r/nextmol_experiment/mol_next_gen"
INSTALL_DIR="/scratch/nishanth.r/nextmol_experiment/mol_next_gen"
DATA_FILES=("data/qm9_selfies.jsonl" "data/qm9_heavy.jsonl")

# Parse args
TARGET_NODE="${1:-$(hostname)}"
MODE="${2:-auto}"   # auto | docker | venv

echo "============================================================"
echo "  mol_next_gen Node Bootstrap"
echo "  Target   : $TARGET_NODE"
echo "  Mode     : $MODE"
echo "  Date     : $(date)"
echo "============================================================"

# ── Detect mode if auto ───────────────────────────────────────────────────────
if [ "$MODE" = "auto" ]; then
    if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
        MODE="docker"
        echo "Auto-detected: Docker available → using Docker mode"
    else
        MODE="venv"
        echo "Auto-detected: No Docker → using git+venv mode"
    fi
fi

# ── Helper: rsync data from source node ──────────────────────────────────────
sync_data() {
    local dest_dir="$1"
    echo ""
    echo "--- Syncing data from $SOURCE_NODE ---"
    mkdir -p "$dest_dir/data"

    for f in "${DATA_FILES[@]}"; do
        local src="$SOURCE_NODE:$SOURCE_PROJECT/$f"
        local dst="$dest_dir/$f"
        if [ ! -f "$dst" ]; then
            echo "  Copying $f (~70MB, please wait)..."
            rsync -avz --progress \
                -e "ssh -o StrictHostKeyChecking=no" \
                "$src" "$dst" 2>/dev/null \
                && echo "  OK: $f ($(wc -l < "$dst") lines)" \
                || echo "  WARNING: Could not rsync $f — copy manually"
        else
            echo "  Already exists: $f ($(wc -l < "$dst") lines)"
        fi
    done
}

# ============================================================
# MODE A: DOCKER
# ============================================================
if [ "$MODE" = "docker" ]; then
    echo ""
    echo "=== MODE A: Docker ==="

    # Pull image
    echo "--- Pulling nishanthr23/nextmol ---"
    docker pull "$DOCKER_IMAGE"

    # Rsync data
    sync_data "$INSTALL_DIR"

    # Create experiment/log dirs
    mkdir -p "$INSTALL_DIR/experiments" "$INSTALL_DIR/logs" "$INSTALL_DIR/checkpoints"

    echo ""
    echo "============================================================"
    echo "  Docker setup COMPLETE on $(hostname)"
    echo ""
    echo "  Run Exp F:"
    echo "    bash $INSTALL_DIR/scripts/docker_run_expF.sh"
    echo ""
    echo "  Or SLURM (update --nodelist to $(hostname)):"
    echo "    sbatch --nodelist=$(hostname) --partition=plafnet2 \\"
    echo "           --account=plafnet2 --gres=gpu:2 \\"
    echo "           $INSTALL_DIR/scripts/docker_run_expF.sh"
    echo "============================================================"
    exit 0
fi

# ============================================================
# MODE B: GIT + VENV
# ============================================================
echo ""
echo "=== MODE B: Git + Venv ==="

# Step 1: Clone or update repo
echo ""
echo "--- Setting up project from GitHub ---"
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "Repo exists — pulling latest..."
    git -C "$INSTALL_DIR" pull origin main 2>/dev/null || git -C "$INSTALL_DIR" pull origin master 2>/dev/null || true
elif [ -d "$INSTALL_DIR" ] && [ "$(ls -A $INSTALL_DIR)" ]; then
    echo "Directory exists but no git — initializing..."
    # Just pull latest scripts from source if on same cluster
    rsync -avz --progress \
        -e "ssh -o StrictHostKeyChecking=no" \
        "$SOURCE_NODE:$SOURCE_PROJECT/" "$INSTALL_DIR/" \
        --exclude="venv/" --exclude="data/*.jsonl" \
        --exclude="experiments/" --exclude="logs/" \
        --exclude="checkpoints/" 2>/dev/null || {
        echo "rsync failed — cloning from GitHub..."
        mkdir -p "$(dirname $INSTALL_DIR)"
        git clone "$GITHUB_REPO" "$INSTALL_DIR"
    }
else
    echo "Cloning from GitHub: $GITHUB_REPO"
    mkdir -p "$(dirname $INSTALL_DIR)"
    git clone "$GITHUB_REPO" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# Step 2: Create venv
echo ""
echo "--- Creating Python virtual environment ---"
VENV_DIR="$INSTALL_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "venv created: $VENV_DIR"
else
    echo "venv already exists: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q

# Step 3: Install PyTorch (CUDA 12.1)
echo ""
echo "--- Installing PyTorch 2.5.1+cu121 ---"
pip install --no-cache-dir -q \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

echo "--- Installing torch-geometric + RDKit + all deps ---"
pip install --no-cache-dir -q \
    torch-geometric==2.7.0 \
    rdkit==2026.3.1 \
    numpy==2.2.6 scipy==1.15.3 matplotlib==3.10.9 \
    pandas==2.3.3 networkx==3.4.2 selfies==2.2.0 \
    tqdm==4.67.3 seaborn==0.13.2 psutil==7.2.2 requests==2.33.1

# Step 4: Sync data
sync_data "$INSTALL_DIR"

# Step 5: Build heavy-atom dataset if missing
if [ ! -f "$INSTALL_DIR/data/qm9_heavy.jsonl" ] && [ -f "$INSTALL_DIR/data/qm9_selfies.jsonl" ]; then
    echo ""
    echo "--- Building heavy-atom dataset ---"
    python "$INSTALL_DIR/data/prepare_qm9_heavy.py" \
        --input  "$INSTALL_DIR/data/qm9_selfies.jsonl" \
        --output "$INSTALL_DIR/data/qm9_heavy.jsonl" \
        --use-rdkit --max-atoms 9
fi

mkdir -p "$INSTALL_DIR/experiments" "$INSTALL_DIR/logs" "$INSTALL_DIR/checkpoints"

# Step 6: Verify
echo ""
echo "--- Verification ---"
PYTHONPATH="$INSTALL_DIR" python3 -c "
import torch
print(f'  PyTorch {torch.__version__} | CUDA={torch.cuda.is_available()} | GPUs={torch.cuda.device_count()}')
import rdkit; print(f'  RDKit {rdkit.__version__}')
from models.conformer_diffusion import ConformerDiffusion; print('  ConformerDiffusion: OK')
from models.attn_conformer_diffusion import AttnConformerDiffusion; print('  AttnConformerDiffusion: OK')
print('  ALL IMPORTS OK')
"

echo ""
echo "============================================================"
echo "  SETUP COMPLETE on $(hostname)"
echo ""
echo "  To run Exp F directly:"
echo "    source $VENV_DIR/bin/activate"
echo "    NUM_GPUS=\$(nvidia-smi --list-gpus | wc -l)"
echo "    PYTHONPATH=$INSTALL_DIR torchrun --nproc_per_node=\$NUM_GPUS \\"
echo "        $INSTALL_DIR/autoresearch/mol_train_ddp.py --exp F"
echo ""
echo "  To submit via SLURM from gnode118:"
echo "    sbatch --nodelist=\$(hostname) --partition=plafnet2 \\"
echo "           --account=plafnet2 --gres=gpu:2 \\"
echo "           $INSTALL_DIR/scripts/exp_F_heavy_atom_ddp.sh"
echo "============================================================"
