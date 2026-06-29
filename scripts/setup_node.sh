#!/bin/bash
# ============================================================
# setup_node.sh — One-command setup on any new cluster node
#
# This script sets up the complete mol_next_gen environment
# on a fresh node WITHOUT needing Docker. It creates a Python
# venv and installs all dependencies from scratch.
#
# Usage:
#   cd /scratch/nishanth.r/nextmol_experiment/mol_next_gen
#   bash scripts/setup_node.sh
#
# After setup, run experiments with:
#   source venv/bin/activate
#   torchrun --nproc_per_node=2 autoresearch/mol_train_ddp.py --exp F
#
# If Docker IS available, use docker_run_expF.sh instead —
#   it's faster (pre-built image, no compilation).
#
# Time estimate: ~10-15 min first time (downloading PyTorch + RDKit)
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
VENV_DIR="$PROJECT_ROOT/venv"

echo "============================================================"
echo "  mol_next_gen Node Setup"
echo "  Node:    $(hostname)"
echo "  Python:  $(python3 --version 2>/dev/null || echo 'not found')"
echo "  Project: $PROJECT_ROOT"
echo "  Date:    $(date)"
echo "============================================================"

# ── 1. Create virtual environment ────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "--- Creating Python virtual environment ---"
    python3 -m venv "$VENV_DIR"
    echo "venv created at $VENV_DIR"
else
    echo "venv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python3 -m pip install --upgrade pip setuptools wheel -q

# ── 2. Install PyTorch (CUDA 12.1) ───────────────────────────────────────────
echo ""
echo "--- Installing PyTorch 2.5.1+cu121 ---"
pip install --no-cache-dir -q \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

echo "--- Installing torch-geometric ---"
pip install --no-cache-dir -q torch-geometric==2.7.0

# ── 3. Install all other dependencies ─────────────────────────────────────────
echo ""
echo "--- Installing remaining dependencies ---"
pip install --no-cache-dir -q \
    rdkit==2026.3.1 \
    numpy==2.2.6 \
    scipy==1.15.3 \
    matplotlib==3.10.9 \
    pandas==2.3.3 \
    networkx==3.4.2 \
    selfies==2.2.0 \
    tqdm==4.67.3 \
    seaborn==0.13.2 \
    psutil==7.2.2 \
    requests==2.33.1

# ── 4. Verify installation ────────────────────────────────────────────────────
echo ""
echo "--- Verification ---"
python3 -c "
import sys; sys.path.insert(0,'$PROJECT_ROOT')
import torch
print(f'  PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}')
import rdkit; print(f'  RDKit {rdkit.__version__}')
import numpy; print(f'  NumPy {numpy.__version__}')
from models.conformer_diffusion import ConformerDiffusion; print('  ConformerDiffusion: OK')
from models.attn_conformer_diffusion import AttnConformerDiffusion; print('  AttnConformerDiffusion: OK')
"

# ── 5. Create heavy-atom dataset if not present ───────────────────────────────
if [ -f "data/qm9_selfies.jsonl" ] && [ ! -f "data/qm9_heavy.jsonl" ]; then
    echo ""
    echo "--- Preparing heavy-atom dataset ---"
    python3 data/prepare_qm9_heavy.py \
        --input  data/qm9_selfies.jsonl \
        --output data/qm9_heavy.jsonl \
        --use-rdkit --max-atoms 9
    echo "  Heavy-atom dataset ready: $(wc -l < data/qm9_heavy.jsonl) molecules"
elif [ -f "data/qm9_heavy.jsonl" ]; then
    echo ""
    echo "  Heavy-atom dataset exists: $(wc -l < data/qm9_heavy.jsonl) molecules"
fi

# ── 6. Done ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  SETUP COMPLETE on $(hostname)"
echo ""
echo "  To run Experiment F (auto-detect GPUs):"
echo "    source venv/bin/activate"
echo "    NUM_GPUS=\$(nvidia-smi --list-gpus | wc -l)"
echo "    torchrun --nproc_per_node=\$NUM_GPUS \\"
echo "        autoresearch/mol_train_ddp.py --exp F"
echo ""
echo "  Or submit to SLURM:"
echo "    sbatch --partition=plafnet2 --account=plafnet2 \\"
echo "           --gres=gpu:2 scripts/exp_F_heavy_atom_ddp.sh"
echo "============================================================"
