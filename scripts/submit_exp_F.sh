#!/bin/bash
# ============================================================
# submit_exp_F.sh — Optimal Exp F submission
#
# Submits Exp F to best available node:
#   - 4 GPUs on gnode118 (2 free now, but requesting all 4 blocks exp D/E)
#   - 2 GPUs on gnode115 or gnode116 (idle nodes, safe choice)
#   - Uses whichever is available first
# ============================================================

set -euo pipefail

PROJECT=/scratch/nishanth.r/nextmol_experiment/mol_next_gen
cd "$PROJECT"

# Check which nodes are free
FREE_NODES=$(sinfo -p plafnet2 -N --states=idle --format="%N" --noheader 2>/dev/null | head -1)
MIXED_NODE="gnode118"

echo "Free nodes: ${FREE_NODES:-none}"
echo "Mixed node (gnode118) has GPUs 0,1 free"

# Prefer 2-GPU idle node, fallback to mixed with specific GPUs
if [ -n "$FREE_NODES" ]; then
    echo "Submitting Exp F to idle 2-GPU node..."
    sbatch --job-name=expF_heavy_ddp \
           --output=logs/expF_%j.log \
           --error=logs/expF_%j.log \
           --nodes=1 \
           --ntasks-per-node=1 \
           --gres=gpu:2 \
           --cpus-per-task=8 \
           --mem=32G \
           --time=4-00:00:00 \
           --partition=plafnet2 \
           --nodelist="$FREE_NODES" \
           scripts/exp_F_heavy_atom_ddp.sh
else
    echo "No idle nodes. Submitting to mixed queue (gnode118, 2 GPUs)..."
    sbatch --job-name=expF_heavy_ddp \
           --output=logs/expF_%j.log \
           --error=logs/expF_%j.log \
           --nodes=1 \
           --ntasks-per-node=1 \
           --gres=gpu:2 \
           --cpus-per-task=8 \
           --mem=32G \
           --time=4-00:00:00 \
           --partition=plafnet2 \
           scripts/exp_F_heavy_atom_ddp.sh
fi

echo ""
echo "Current queue:"
squeue -u $USER --format="%.10i %.16j %.8T %.10M %R"
