#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --job-name=dspmv_strong
#SBATCH --output=outputs/strong_%j.out
#SBATCH --error=outputs/strong_%j.err

# Strong scaling for ONE matrix, P in {1,2,4}. One matrix per job keeps us
# under the 5-minute edu-short limit. The binary must already be built
# (run scripts/sbatch_build.sh first). Pass the matrix as the first argument:
#   sbatch scripts/sbatch_strong.sh matrices/cant/cant.mtx
module load OpenMPI
module load CUDA/12.5.0

mkdir -p outputs

MTX="${1:-${MTX:-}}"
if [ -z "$MTX" ] || [ ! -f "$MTX" ]; then
    echo "ERROR: matrix file not found. Usage: sbatch scripts/sbatch_strong.sh <matrix.mtx>"
    exit 1
fi

if [ ! -x ./bin/dspmv ]; then
    echo "ERROR: ./bin/dspmv not found. Build first with scripts/sbatch_build.sh"
    exit 1
fi

name=$(basename "$(dirname "$MTX")")
LOG="outputs/strong_${name}_${SLURM_JOB_ID}.log"
CSV="outputs/strong_${name}_${SLURM_JOB_ID}.csv"
: > "$LOG"

for P in 1 2 4; do
    echo ">>> matrix=$MTX P=$P" | tee -a "$LOG"
    mpirun -np "$P" ./bin/dspmv "$MTX" >> "$LOG" 2>&1
done

# CSV: header once, then data rows (both emitted on stderr, prefixed CSVROW,)
grep -m1 '^CSVROW,matrix,ranks,' "$LOG" | sed 's/^CSVROW,//' > "$CSV"
grep '^CSVROW,' "$LOG" | grep -v ',matrix,ranks,' | sed 's/^CSVROW,//' >> "$CSV"
echo "Wrote $CSV"
