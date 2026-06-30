#!/bin/bash
#SBATCH --job-name=download_geom
#SBATCH --output=/scratch/nishanth.r/nextmol_experiment/mol_expermiments/logs/download_geom_%j.log
#SBATCH --error=/scratch/nishanth.r/nextmol_experiment/mol_expermiments/logs/download_geom_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118

echo "=========================================="
echo "Job    : Download and prep GEOM Drugs"
echo "Node   : $(hostname)"
echo "Started: $(date)"
echo "=========================================="

cd /scratch/nishanth.r/nextmol_experiment/mol_expermiments
source venv/bin/activate

mkdir -p data/geom_raw

echo "Starting download of rdkit_folder.tar.gz (50 GB)..."
# Resume download if partially downloaded (-c flag)
wget -c "https://dataverse.harvard.edu/api/access/datafile/4327252" -O data/geom_raw/rdkit_folder.tar.gz

echo "Download complete, starting conversion..."
PYTHONPATH=. python3 data/prepare_geom_drugs.py \
    --tar data/geom_raw/rdkit_folder.tar.gz \
    --out data/geom_drugs.jsonl \
    --max-atoms 100 \
    --min-confs 2

echo "Job completed: $(date)"
