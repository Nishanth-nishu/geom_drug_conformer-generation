#!/bin/bash
# monitor_expG.sh — Quick status check for Experiment G training
# Usage: bash scripts/monitor_expG.sh

LOGF=$(ls -t /scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/expG_2026*.log 2>/dev/null | head -1)
PID=$(cat /scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/expG_pid.txt 2>/dev/null)

echo "============================================================"
echo "  Experiment G — Training Monitor"
echo "  Log: $LOGF"
echo "  PID: $PID"
echo "============================================================"

# Check if process is still running
if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
    echo "  Status: RUNNING ✓"
else
    echo "  Status: NOT RUNNING (may have finished or crashed)"
fi

echo ""
echo "--- GPU ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null

echo ""
echo "--- Last 20 log lines ---"
tail -20 "$LOGF"

echo ""
echo "--- GeoDiff results so far ---"
cat /scratch/nishanth.r/nextmol_experiment/mol_next_gen/autoresearch/results_geodiff.tsv 2>/dev/null || echo "(none yet)"

echo ""
echo "--- Checkpoints ---"
ls -lh /scratch/nishanth.r/nextmol_experiment/mol_next_gen/checkpoints/exp_G* 2>/dev/null || echo "(none yet)"
