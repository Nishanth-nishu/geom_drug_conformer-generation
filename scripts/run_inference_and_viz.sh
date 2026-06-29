#!/bin/bash
#SBATCH --job-name=mol_infer_viz
#SBATCH --output=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/infer_viz_%j.log
#SBATCH --error=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/infer_viz_%j.log
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --nodelist=gnode118

# =============================================================================
# Post-training: Run inference for all 4 experiments + generate all visualizations
#
# This script:
#   1. Generates 200 molecules per experiment from the best checkpoint
#   2. Saves each molecule as .pdb, .mol2, .sdf in generated/exp_*/
#   3. Saves edge cases (invalid molecules) in generated/exp_*/edge_cases/
#   4. Runs the full 10-plot visualization suite
#   5. Commits all results to the GitHub repo
#
# Run AFTER all 4 training jobs complete:
#   sbatch scripts/run_inference_and_viz.sh
# =============================================================================

set -euo pipefail

PROJ=/scratch/nishanth.r/nextmol_experiment/mol_next_gen
cd "$PROJ"

export PYTHONPATH="$PROJ"
source venv/bin/activate

echo "============================================================"
echo "  NExT-Mol Gen: Inference + Visualization"
echo "  Job ID : ${SLURM_JOB_ID:-interactive}"
echo "  Node   : $(hostname)"
echo "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Start  : $(date)"
echo "============================================================"

# ── Verify checkpoints exist ──────────────────────────────────────────────────
echo ""
echo "=== Checking checkpoints ==="
for exp in exp_A_baseline exp_B_attention_egnn exp_C_flow_matching exp_D_torsion_aux; do
    ckpt="checkpoints/${exp}_best.pt"
    if [ -f "$ckpt" ]; then
        SIZE=$(du -sh "$ckpt" | cut -f1)
        echo "  [OK] $ckpt ($SIZE)"
    else
        echo "  [MISSING] $ckpt — inference will skip this experiment"
    fi
done

# ── Step 1: Generate molecules → PDB, MOL2, SDF ──────────────────────────────
echo ""
echo "=== Step 1: Inference — generating 200 molecules per experiment ==="
python autoresearch/mol_infer.py --all --num-molecules 200 2>&1
echo ""
echo "Generated files:"
find generated/ -name "*.pdb" | wc -l | xargs -I{} echo "  {} PDB files"
find generated/ -name "*.mol2" | wc -l | xargs -I{} echo "  {} MOL2 files"
find generated/ -name "*.sdf" | wc -l | xargs -I{} echo "  {} SDF files"

# ── Step 2: Visualizations ────────────────────────────────────────────────────
echo ""
echo "=== Step 2: Generating 10 research-grade plots ==="
python visualization/plot_results.py 2>&1
echo ""
echo "Plots generated:"
ls -lh visualization/plots/*.png 2>/dev/null || echo "  No plots found"

# ── Step 3: Print comparison table ───────────────────────────────────────────
echo ""
echo "=== Step 3: Inference Results Comparison ==="
python3 - <<'PYEOF'
import json
from pathlib import Path

proj = Path("/scratch/nishanth.r/nextmol_experiment/mol_next_gen")
exps = ["exp_A_baseline","exp_B_attention_egnn","exp_C_flow_matching","exp_D_torsion_aux"]
labels = {"exp_A_baseline":"A: Baseline","exp_B_attention_egnn":"B: Attention",
          "exp_C_flow_matching":"C: FlowMatch","exp_D_torsion_aux":"D: Torsion"}

print(f"\n{'Experiment':<22} {'Valid%':>8} {'BondErr':>10} {'#Valid':>7} {'Sampler':>10} {'Steps':>6}")
print("-" * 68)
for exp in exps:
    f = proj / "generated" / exp / "summary.json"
    if f.exists():
        s = json.loads(f.read_text())
        print(f"{labels[exp]:<22} "
              f"{s['fully_valid_rate']*100:>7.1f}% "
              f"{s['mean_bond_error']:>10.4f} "
              f"{s['valid']:>7d} "
              f"{s['sampler']:>10} "
              f"{s['num_steps']:>6d}")
    else:
        print(f"{labels[exp]:<22}  (no results yet)")
PYEOF

# ── Step 4: Commit results to GitHub ─────────────────────────────────────────
echo ""
echo "=== Step 4: Committing results to GitHub ==="
git config user.email "nishanth.r@research.iiit.ac.in" 2>/dev/null || true
git config user.name "Nishanth R" 2>/dev/null || true

# Stage results (but not binary checkpoints or large data)
git add autoresearch/results.tsv 2>/dev/null || true
git add generated/*/summary.json 2>/dev/null || true
git add visualization/plots/*.png 2>/dev/null || true
git add visualization/plot_results.py 2>/dev/null || true
git add autoresearch/mol_infer.py 2>/dev/null || true
git add scripts/run_inference_and_viz.sh 2>/dev/null || true

# Add experiment train logs
for exp in exp_A_baseline exp_B_attention_egnn exp_C_flow_matching exp_D_torsion_aux; do
    git add "experiments/${exp}/" 2>/dev/null || true
done

# Get final metrics for commit message
FVALID=$(grep "^fully_valid:" experiments/exp_A_baseline/train.log 2>/dev/null | tail -1 | awk '{print $2}' || echo "?")
MATR=$(grep "^mat_r:" experiments/exp_A_baseline/train.log 2>/dev/null | tail -1 | awk '{print $2}' || echo "?")

git commit -m "results: inference + visualization complete

Exp A baseline: fully_valid=${FVALID}, mat_r=${MATR}
Generated 200 molecules per experiment
- PDB/MOL2/SDF files in generated/exp_*/
- 10 research plots in visualization/plots/
- Edge cases in generated/exp_*/edge_cases/" 2>/dev/null || echo "Nothing to commit"

git push origin main 2>&1 && echo "Pushed to GitHub" || echo "Push failed — check credentials"

echo ""
echo "============================================================"
echo "  COMPLETE — $(date)"
echo ""
echo "  To view generated molecules:"
echo "    pymol generated/exp_A_baseline/mol_0000.pdb"
echo "    chimera generated/exp_B_attention_egnn/mol_0000.mol2"
echo ""
echo "  To view all plots:"
echo "    ls visualization/plots/"
echo "    # or on your laptop: scp gnode118:$PROJ/visualization/plots/ ."
echo "============================================================"
