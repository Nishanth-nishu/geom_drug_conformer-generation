#!/bin/bash
# ============================================================
# exp_F_heavy_atom_ddp.sh — Exp F: Heavy-Atom SOTA with DDP
#
# Architecture: AttnConformerDiffusion (hidden=384, 8 layers, 6 heads)
# Dataset:      heavy-atom-only QM9 (max 9 atoms, no explicit H)
# Training:     500 epochs, AMP fp16, DDP, batch=256/GPU
# Expected:     fully_valid ~50-65%, MAT-R ~0.28-0.35 Å
# ============================================================

#SBATCH --job-name=expF_heavy_ddp
#SBATCH --output=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/expF_%j.log
#SBATCH --error=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/expF_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118  # /scratch is LOCAL to gnode118 — data not on other nodes
                             # To expand: run scripts/setup_new_node.sh on target node first

PROJECT=/scratch/nishanth.r/nextmol_experiment/mol_next_gen

echo "============================================================"
echo "  Experiment F: SOTA Heavy-Atom + DDP"
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Node      : $(hostname)"
echo "  Start     : $(date)"
echo "============================================================"

# Strict mode AFTER initial diagnostics
set -euo pipefail

cd "$PROJECT"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.free --format=csv,noheader

# Activate venv (absolute path — robust across all nodes)
source "$PROJECT/venv/bin/activate"
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA={torch.cuda.is_available()} | GPUs={torch.cuda.device_count()}')"

# Heavy-atom dataset (should already exist at /scratch which is NFS-shared)
if [ ! -f "$PROJECT/data/qm9_heavy.jsonl" ]; then
    echo "Preparing heavy-atom dataset..."
    python "$PROJECT/data/prepare_qm9_heavy.py" \
        --input  "$PROJECT/data/qm9_selfies.jsonl" \
        --output "$PROJECT/data/qm9_heavy.jsonl" \
        --use-rdkit --max-atoms 9
fi
echo "Dataset: $(wc -l < "$PROJECT/data/qm9_heavy.jsonl") molecules"

# DDP training
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
mkdir -p "$PROJECT/experiments/exp_F_heavy_atom"

export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4
export PYTHONPATH="$PROJECT"

echo "Starting torchrun with $NUM_GPUS GPUs..."
torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_addr=localhost \
    --master_port=29503 \
    "$PROJECT/autoresearch/mol_train_ddp.py" \
    --exp F \
    --data "$PROJECT/data/qm9_heavy.jsonl" \
    2>&1 | tee "$PROJECT/experiments/exp_F_heavy_atom/train.log"

echo "============================================================"
echo "  Exp F DONE | End: $(date)"
echo "============================================================"

python -c "
import json, os
f = '$PROJECT/experiments/exp_F_heavy_atom/metrics.json'
if os.path.exists(f):
    m = json.load(open(f))
    print(f'  fully_valid = {m.get(\"fully_valid\", \"N/A\")}')
    print(f'  mat_r       = {m.get(\"mat_r\", \"N/A\")}')
    print(f'  bond_error  = {m.get(\"bond_error\", \"N/A\")}')
" 2>/dev/null || true
