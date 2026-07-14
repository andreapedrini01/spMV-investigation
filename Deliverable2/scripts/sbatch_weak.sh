#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --job-name=dspmv_weak
#SBATCH --output=outputs/weak_%j.out
#SBATCH --error=outputs/weak_%j.err

# Weak scaling: N = ROWS_PER_RANK * P, so per-rank work is constant while P
# grows. A flat time curve means the algorithm is weakly scalable; growth is
# the communication overhead added by P. Build first (scripts/sbatch_build.sh).
module purge
module load OpenMpi/4.1.5-CUDA-12.3.2   # CUDA-aware MPI (smcuda BTL, GPUDirect P2P)
module load CUDA/12.3.2

mkdir -p outputs

if [ ! -x ./bin/dspmv ]; then
    echo "ERROR: ./bin/dspmv not found. Build first with scripts/sbatch_build.sh"
    exit 1
fi

ROWS_PER_RANK="${ROWS_PER_RANK:-200000}"  # owned rows per GPU
NNZ_PER_ROW="${NNZ_PER_ROW:-32}"          # nonzeros per row (constant work/rank)

LOG="outputs/weak_${SLURM_JOB_ID}.log"
CSV="outputs/weak_${SLURM_JOB_ID}.csv"
: > "$LOG"

for PAT in random banded; do
    for P in 1 2 4; do
        echo ">>> gen pattern=$PAT rows/rank=$ROWS_PER_RANK nnz/row=$NNZ_PER_ROW P=$P" | tee -a "$LOG"
        mpirun -np "$P" ./bin/dspmv --gen "$ROWS_PER_RANK" "$NNZ_PER_ROW" \
               --pattern "$PAT" >> "$LOG" 2>&1
    done
done

grep -m1 '^CSVROW,matrix,ranks,' "$LOG" | sed 's/^CSVROW,//' > "$CSV"
grep '^CSVROW,' "$LOG" | grep -v ',matrix,ranks,' | sed 's/^CSVROW,//' >> "$CSV"
echo "Wrote $CSV"
