#!/bin/bash
# setup_venv.sh — Recreate the Python venv after SLURM scratch purge
# Run this from: /scratch/nishanth.r/nextmol_experiment/mol_expermiments

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

echo "============================================================"
echo "  Recreating venv at: $VENV_DIR"
echo "============================================================"

# Remove stale venv directory if it exists but is broken
if [ -d "$VENV_DIR" ]; then
    echo "Removing broken venv..."
    rm -rf "$VENV_DIR"
fi

# Create fresh venv using system python3
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo ""
echo "Python: $(python --version)"
echo "pip:    $(pip --version)"

# Upgrade pip first
pip install --upgrade pip

echo ""
echo "============================================================"
echo "  Installing PyTorch (CUDA 11.8 — matches gnode118 driver)"
echo "============================================================"
# Check CUDA version and install appropriate torch
# gnode118 has RTX 3090 — CUDA 11.x or 12.x depending on driver
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP "release \K[0-9]+\.[0-9]+" || echo "unknown")
echo "Detected CUDA: $CUDA_VER"

if python -c "import torch; print(torch.__version__)" 2>/dev/null; then
    echo "PyTorch already installed, skipping."
else
    # Install PyTorch — try cu118 first (common on older HPC nodes)
    # Adjust the --index-url if your cluster uses a different CUDA version
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
fi

echo ""
echo "============================================================"
echo "  Installing other requirements"
echo "============================================================"
pip install numpy

# rdkit — needed only for data prep (prepare_geom_drugs.py), but install anyway
pip install rdkit

# msgpack — needed only for parsing original GEOM msgpack files
pip install msgpack

echo ""
echo "============================================================"
echo "  Verifying installation"
echo "============================================================"
python -c "
import torch, numpy, rdkit, msgpack
print(f'torch   : {torch.__version__}  (CUDA available: {torch.cuda.is_available()})')
print(f'numpy   : {numpy.__version__}')
print(f'rdkit   : {rdkit.__version__}')
print(f'msgpack : {msgpack.version}')
if torch.cuda.is_available():
    print(f'GPU     : {torch.cuda.get_device_name(0)}')
"

echo ""
echo "============================================================"
echo "  Done! Activate with:"
echo "    source venv/bin/activate"
echo "============================================================"
