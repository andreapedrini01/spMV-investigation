#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=dspmv_strong
#SBATCH --output=outputs/strong_%j.out
#SBATCH --error=outputs/strong_%j.err

# Strong scaling: fixed real matrices, increasing GPU count P in {1,2,4}.
module load OpenMPI
module load CUDA/12.5.0

mkdir -p outputs
make clean
# Build with NCCL and NVML enabled (bonus); drop the flags if the libraries
# are unavailable: `make` for the plain build.
make NCCL=1 NVML=1

LOG="outputs/strong_${SLURM_JOB_ID}.log"
CSV="outputs/strong_${SLURM_JOB_ID}.csv"
: > "$LOG"

for MTX in matrices/*/*.mtx; do
    [ -f "$MTX" ] || continue
    for P in 1 2 4; do
        echo ">>> matrix=$MTX P=$P" | tee -a "$LOG"
        mpirun -np "$P" ./bin/dspmv "$MTX" >> "$LOG" 2>&1
    done
done

# Extract the CSV rows (header once, then data) emitted on stderr.
grep -m1 '^CSVROW,matrix,ranks,' "$LOG" | sed 's/^CSVROW,//' > "$CSV"
grep '^CSVROW,' "$LOG" | grep -v ',matrix,ranks,' | sed 's/^CSVROW,//' >> "$CSV"
echo "Wrote $CSV"
