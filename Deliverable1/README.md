# Deliverable 1 — SpMV Investigation

Sparse Matrix-Vector Multiplication (SpMV) on GPU using CUDA. Compares CSR-Scalar, CSR-Vector (warp shuffle and shared memory variants), and cuSPARSE across 10 SuiteSparse matrices with different sparsity patterns.

## Project Structure

```
Deliverable1/
├── src/
│   ├── main.cu          # Entry point: reads .mtx, runs all kernels, prints results
│   ├── mmio.cu / mmio.h # Matrix Market I/O (NIST reference library)
│   ├── spmv_cpu.cu / .h # CPU baseline (COO + CSR) for validation
│   ├── spmv_gpu.cu / .h # GPU kernels: CSR-Scalar, CSR-Vector, cuSPARSE
│   └── utils.cu / .h    # Utilities: MTX reader, COO->CSR, stats, device info
├── scripts/
│   ├── download_matrices.sh  # Downloads 10 matrices from SuiteSparse
│   └── sbatch_run.sh         # SLURM batch script for cluster execution
├── matrices/                  # Downloaded .mtx files (not tracked in git)
├── outputs/                   # SLURM output logs
├── Makefile
└── README.md
```

## Environment

This project was developed and tested on the DISI cluster at the University of Trento. The compute nodes have NVIDIA A30 GPUs (Ampere, sm_80). Jobs are submitted through SLURM on the `edu-short` partition with the `gpu.computing26` account.

Software used:
- CUDA 11.8 (`module load CUDA/11.8.0`)
- cuSPARSE (ships with the CUDA Toolkit)
- `wget` for downloading matrices from SuiteSparse

## Build

```bash
make        # builds bin/spmv
make clean  # removes bin/
```

The Makefile targets `sm_80` by default. If you need a different architecture, edit `NVCC_FLAGS`.

## Download Matrices

Run this from the login node:

```bash
chmod +x scripts/download_matrices.sh
./scripts/download_matrices.sh
```

This downloads 10 matrices into `matrices/`. The selection covers small regular matrices (cage4, olm1000), medium structured ones (west2021, mac_econ_fwd500), FEM matrices with regular nnz/row (cant, consph, cop20k_A), a protein structure (pdb1HYS), a power-law web graph (webbase-1M), and a circuit simulation with irregular structure (scircuit).

## Run

Single matrix:
```bash
./bin/spmv matrices/cant/cant.mtx
```

All matrices on the cluster:
```bash
sbatch scripts/sbatch_run.sh
```

## Output

For each matrix the program prints:
- Matrix dimensions and NNZ count
- GPU device properties
- Per-kernel timing (mean over 10 iterations, after 4 warmup runs)
- GFLOP/s (using 2·nnz flops per SpMV)
- Effective memory bandwidth (GB/s)
- Total absolute error between CPU and GPU results
- A CSV summary line for easy data collection

## Kernels

| Kernel | Strategy | Key Feature |
|--------|----------|-------------|
| CSR-Scalar | 1 thread per row | Simple, poor load balance on irregular matrices |
| CSR-Vector (shuffle) | 1 warp per row | Warp-level reduction via `__shfl_down_sync` |
| CSR-Vector (shmem) | 1 warp per row | Reduction through shared memory |
| cuSPARSE | Library | NVIDIA's optimized SpMV implementation |

## Validation

GPU results are validated against the CPU CSR implementation by computing the sum of absolute differences across all output elements, following the same approach used in the course labs.
