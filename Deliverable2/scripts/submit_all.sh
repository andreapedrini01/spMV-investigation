#!/bin/bash
# Run this on the login node (baldo). It submits the build job, then the run
# jobs with a dependency so they start only after the build succeeds. Each job
# stays within the 5-minute edu-short limit (build once, one matrix per strong
# job, one weak job).
set -e

mkdir -p outputs

echo "Submitting build job..."
BUILD_JID=$(sbatch --parsable scripts/sbatch_build.sh)
echo "  build job: $BUILD_JID"

echo "Submitting strong-scaling jobs (one per matrix)..."
shopt -s nullglob
for MTX in matrices/*/*.mtx; do
    JID=$(sbatch --parsable --dependency=afterok:"$BUILD_JID" scripts/sbatch_strong.sh "$MTX")
    echo "  strong $(basename "$MTX"): $JID"
done

echo "Submitting weak-scaling job..."
WJID=$(sbatch --parsable --dependency=afterok:"$BUILD_JID" scripts/sbatch_weak.sh)
echo "  weak: $WJID"

echo
echo "All jobs submitted. Watch them with:  squeue -u \$USER"
echo "When finished, merge the CSVs with:   bash scripts/collect_csv.sh"
