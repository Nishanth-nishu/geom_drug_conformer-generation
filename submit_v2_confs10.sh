#!/bin/bash
#SBATCH --job-name=v2_confs10
#SBATCH --output=/scratch/nishanth.r/nextmol_experiment/mol_expermiments/logs/v2_confs10_%j.log
#SBATCH --error=/scratch/nishanth.r/nextmol_experiment/mol_expermiments/logs/v2_confs10_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118

# ─────────────────────────────────────────────────────────────────────────────
# Conformer Ablation: 10 conformers/molecule
#
# Architecture : DualEncoderDiffusion (distance-space score matching)
# Noise schedule: sigmoid β, 1e-7 → 2e-3  (GeoDiff drugs_default.yml)
# Diffusion steps: 5000                    (GeoDiff canonical)
# Neighbor radius: 10 Å                    (GeoDiff: cutoff=10.0)
# Conformers    : TOP-10 by Boltzmann weight
#
# Compare against submit_v2_confs5.sh (5 conformers) to measure
# how conformer count affects COV-R/MAT-R.
# ─────────────────────────────────────────────────────────────────────────────

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "SLURM_JOB_GPUS = $SLURM_JOB_GPUS"
echo "=========================================="

# ── Configuration ─────────────────────────────────────────────────────────────
MAX_ATOMS=30
MAX_MOLS=50000
MAX_CONFS=10          # ← KEY: 10 conformers per molecule
MIN_CONFS=2
HIDDEN_DIM=256
NUM_TIMESTEPS=5000    # GeoDiff canonical: 5000 steps
BETA_START=1e-7       # GeoDiff: β_start
BETA_END=2e-3         # GeoDiff: β_end
CUTOFF=10.0           # GeoDiff: neighbor radius (Å)
BATCH_SIZE=32
EPOCHS=300

EXP_NAME="v2_dual_confs10_T5000_atoms${MAX_ATOMS}_mols${MAX_MOLS}"

echo "============================================================"
echo "Job started : $(hostname) at $(date)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Experiment  : $EXP_NAME"
echo "Architecture: DualEncoderDiffusion"
echo "Conformers  : $MAX_CONFS (TOP-10 by Boltzmann weight)"
echo "Timesteps   : $NUM_TIMESTEPS (sigmoid schedule)"
echo "β range     : $BETA_START → $BETA_END"
echo "Radius      : $CUTOFF Å"
echo "Max atoms   : $MAX_ATOMS"
echo "Max mols    : $MAX_MOLS"
echo "============================================================"

cd /scratch/nishanth.r/nextmol_experiment/mol_expermiments
source venv/bin/activate

DATA_PATH="data/geom_drugs.jsonl"
if [ ! -f "$DATA_PATH" ]; then
    echo "[ERROR] Dataset not found: $DATA_PATH"
    exit 1
fi

LINES=$(wc -l < "$DATA_PATH")
echo "Dataset: $DATA_PATH ($LINES molecules)"

mkdir -p logs

# ── Smoke test ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "Running smoke test (5 epochs) ..."
echo "============================================================"

PYTHONPATH=. python autoresearch/mol_train_v2.py \
    --smoke-test \
    --max-atoms $MAX_ATOMS \
    --max-mols $MAX_MOLS \
    --max-confs $MAX_CONFS \
    --min-confs $MIN_CONFS \
    --hidden-dim $HIDDEN_DIM \
    --num-timesteps $NUM_TIMESTEPS \
    --beta-start $BETA_START \
    --beta-end $BETA_END \
    --cutoff $CUTOFF \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --exp-name $EXP_NAME \
    --data $DATA_PATH

if [ $? -ne 0 ]; then
    echo "[ERROR] Smoke test failed. Aborting."
    exit 1
fi

echo "Smoke test passed! Starting full training ..."
echo "============================================================"
echo "Full training: $EPOCHS epochs | 10 conformers/mol | T=$NUM_TIMESTEPS"
echo "============================================================"

# ── Full training ─────────────────────────────────────────────────────────────
PYTHONPATH=. python autoresearch/mol_train_v2.py \
    --max-atoms $MAX_ATOMS \
    --max-mols $MAX_MOLS \
    --max-confs $MAX_CONFS \
    --min-confs $MIN_CONFS \
    --hidden-dim $HIDDEN_DIM \
    --num-timesteps $NUM_TIMESTEPS \
    --beta-start $BETA_START \
    --beta-end $BETA_END \
    --cutoff $CUTOFF \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --exp-name $EXP_NAME \
    --data $DATA_PATH

echo "============================================================"
echo "Job completed at $(date)"
echo "============================================================"
