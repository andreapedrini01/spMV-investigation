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

# Compile on a compute node (the login node's module assembler segfaults).
# No GPU is needed to cross-compile for sm_80, so we request gpu:0 and use
# several CPUs to build the translation units in parallel.
# CUDA-aware OpenMPI (smcuda BTL => direct GPU-to-GPU): this is the build we
# want so device pointers passed to MPI use GPUDirect P2P, not host staging.
module purge
module load OpenMpi/4.1.5-CUDA-12.3.2
module load CUDA/12.3.2                 # auto-loaded by the OpenMPI module; explicit for clarity
module load NCCL 2>/dev/null || true    # not available as a module on edu; build falls back to plain

mkdir -p outputs
make clean
# Bonus build (NCCL + NVML); fall back to the plain build if those libs are
# missing, so we still get a working binary.
make -j"${SLURM_CPUS_PER_TASK:-4}" NCCL=1 NVML=1 \
  || { echo "=== bonus build failed, falling back to plain build ==="; make clean; make -j"${SLURM_CPUS_PER_TASK:-4}"; }

echo "Build finished:"
ls -l bin/dspmv
