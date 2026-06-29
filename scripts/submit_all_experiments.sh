#!/bin/bash
# submit_all_experiments.sh — Wait for QM9 data then submit all 4 experiments
# Run as: bash scripts/submit_all_experiments.sh &

set -euo pipefail
PROJ=/scratch/nishanth.r/nextmol_experiment/mol_next_gen
DATA="$PROJ/data/qm9_selfies.jsonl"

echo "============================================================"
echo "  NExT-Mol Gen: Experiment Submission Script"
echo "  Waiting for QM9 data preparation to complete..."
echo "  Data path: $DATA"
echo "  Start: $(date)"
echo "============================================================"

# Wait for data file to exist and have meaningful size (>1MB)
while true; do
    if [ -f "$DATA" ]; then
        SIZE=$(stat -c%s "$DATA" 2>/dev/null || echo 0)
        if [ "$SIZE" -gt 1000000 ]; then
            echo ""
            echo "  Data ready: $(wc -l < "$DATA") molecules (${SIZE} bytes)"
            break
        fi
    fi
    PREP_LOG="$PROJ/logs/prepare_qm9.log"
    if [ -f "$PREP_LOG" ]; then
        LAST=$(tail -1 "$PREP_LOG" 2>/dev/null || echo "...")
        echo -ne "\r  Still preparing... $LAST"
    fi
    sleep 10
done

echo ""
echo "============================================================"
echo "  Submitting 4 experiments to SLURM"
echo "============================================================"

cd "$PROJ"

JOB_A=$(sbatch --parsable scripts/exp_A_baseline.sh)
echo "  Submitted Exp A (baseline):         Job $JOB_A"

JOB_B=$(sbatch --parsable scripts/exp_B_attention_egnn.sh)
echo "  Submitted Exp B (attention EGNN):   Job $JOB_B"

JOB_C=$(sbatch --parsable scripts/exp_C_flow_matching.sh)
echo "  Submitted Exp C (flow matching):    Job $JOB_C"

JOB_D=$(sbatch --parsable scripts/exp_D_torsion_aux.sh)
echo "  Submitted Exp D (torsion aux):      Job $JOB_D"

echo ""
echo "============================================================"
echo "  All 4 experiments submitted!"
echo "  Monitor with:  squeue -u nishanth.r"
echo "  Follow logs:   tail -f logs/expA_${JOB_A}.log"
echo "  Results TSV:   cat autoresearch/results.tsv"
echo "  Submitted: $(date)"
echo "============================================================"

JOB_E=$(sbatch --parsable scripts/exp_E_sota_hybrid.sh)
echo "  Submitted Exp E (sota hybrid):      Job $JOB_E"
