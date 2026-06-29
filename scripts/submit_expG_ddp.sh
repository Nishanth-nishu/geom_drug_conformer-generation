#!/bin/bash
# ============================================================
# submit_expG_ddp.sh — Experiment G: SOTA Heavy-Atom with DDP + W&B
#
# Architecture: ConformerDiffusion (hidden=384, 8 layers, rbf=32)
# Dataset:      heavy-atom-only QM9 (max 9 atoms, no explicit H)
# Training:     500 epochs, AMP fp16, DDP, batch=128/GPU (256 effective)
# Monitoring:   Weights & Biases (W&B) + GeoDiff COV-MAT eval
# Expected:     fully_valid ~60-70%, MAT-R < 0.30 Å
# ============================================================

#SBATCH --job-name=expG_ddp
#SBATCH --output=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/expG_ddp_%j.log
#SBATCH --error=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/expG_ddp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2                  # Request 2 GPUs
#SBATCH --cpus-per-task=8             # 8 CPUs
#SBATCH --mem=32G                     # 32 GB RAM
#SBATCH --time=4-00:00:00             # 4 days
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2            # Required for plafnet2 partition

PROJECT=/scratch/nishanth.r/nextmol_experiment/mol_next_gen
VENV=/scratch/nishanth.r/nextmol_experiment/GeoDiff/venv

echo "============================================================"
echo "  Experiment G: SOTA Heavy-Atom + DDP + W&B"
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Node      : $(hostname)"
echo "  GPUs      : ${SLURM_GPUS_ON_NODE:-N/A}"
echo "  Start     : $(date)"
echo "============================================================"

# Strict mode AFTER initial diagnostics
set -euo pipefail

cd "$PROJECT"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.free --format=csv,noheader

# Activate venv
source "$VENV/bin/activate"
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA={torch.cuda.is_available()} | GPUs={torch.cuda.device_count()}')"

# Environment variables for Python script
export PYTHONPATH="$PROJECT"
export MOL_DATASET="$PROJECT/data/qm9_heavy.jsonl"
export MOL_MAX_ATOMS=9
export WANDB_API_KEY="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"
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Starting torchrun with $NUM_GPUS GPUs..."

torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    --nnodes=1 \
    "$PROJECT/autoresearch/mol_train_expG_ddp.py"

EXIT_CODE=$?

echo "============================================================"
echo "  Exp G DONE | End: $(date) | Exit: $EXIT_CODE"
echo "============================================================"
