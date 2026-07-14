#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --job-name=dspmv_build
#SBATCH --output=outputs/build_%j.out
#SBATCH --error=outputs/build_%j.err

module purge
module load OpenMpi/4.1.5-CUDA-12.3.2
module load CUDA/12.3.2                 # auto-loaded by the OpenMPI module; explicit for clarity
# NCCL is NOT available as a standalone module on the edu partition (only inside
# NVHPC, which conflicts with this toolchain), so we do not build it here. The
# NCCL path stays behind the USE_NCCL compile flag for portability.

mkdir -p outputs
make clean
# NVML bonus (per-rank GPU UUID print); fall back to a plain build if NVML is
# unavailable. Staging + CUDA-aware are always built.
make -j"${SLURM_CPUS_PER_TASK:-4}" NVML=1 \
  || { echo "=== NVML build failed, falling back to plain build ==="; make clean; make -j"${SLURM_CPUS_PER_TASK:-4}"; }

echo "Build finished:"
ls -l bin/dspmv
