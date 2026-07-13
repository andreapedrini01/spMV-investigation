#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=dspmv_weak
#SBATCH --output=outputs/weak_%j.out
#SBATCH --error=outputs/weak_%j.err

# Weak scaling: synthetic matrices with N = ROWS_PER_RANK * P, so the per-rank
# work is constant while P grows. A flat time curve means the algorithm is
# weakly scalable; any growth is the communication overhead added by P.
module load OpenMPI
module load CUDA/12.5.0

mkdir -p outputs
make clean
make NCCL=1 NVML=1

ROWS_PER_RANK=200000   # owned rows per GPU
NNZ_PER_ROW=32         # nonzeros per row (constant work per rank)

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
