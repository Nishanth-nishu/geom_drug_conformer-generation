#!/bin/bash
#SBATCH --job-name=download_qm9
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/nishanth.r/nextmol_experiment/mol_expermiments/logs/download_qm9_%j.log
#SBATCH --error=/scratch/nishanth.r/nextmol_experiment/mol_expermiments/logs/download_qm9_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118
echo "=========================================="
echo "Job    : Download and prep GEOM QM9"
echo "Node   : $(hostname)"
echo "Started: $(date)"
echo "=========================================="

cd /scratch/nishanth.r/nextmol_experiment/mol_expermiments
source venv/bin/activate

mkdir -p data/geom_raw

echo "Starting download and conversion of qm9_crude.msgpack.tar.gz (~1 GB)..."
PYTHONPATH=. python3 data/prepare_geom_qm9.py --out data/geom_qm9.jsonl --min-confs 2

echo "Job completed: $(date)"
