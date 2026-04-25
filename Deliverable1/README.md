# Deliverable 1 — SpMV Investigation

Sparse Matrix-Vector Multiplication (SpMV) on GPU using CUDA. Compares CSR-Scalar, CSR-Vector (warp shuffle and shared memory variants), and cuSPARSE across 10 SuiteSparse matrices with different sparsity patterns.

## Project Structure

```
Deliverable1/
├── src/
│   ├── main.cu          # Entry point: reads .mtx, runs all kernels, prints results
│   ├── mmio.c / mmio.h  # Matrix Market I/O (based on NIST reference)
│   ├── spmv_cpu.c / .h  # CPU baseline (COO + CSR) for validation
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

## Requirements

- CUDA Toolkit (tested with 11.8+)
- GPU with compute capability sm_80 (Ampere A100) or sm_86 (A30/A40)
- cuSPARSE (included with CUDA Toolkit)
- `wget` for downloading matrices

## Build

```bash
make        # builds bin/spmv
make clean  # removes bin/
```

To target a different GPU architecture, edit `NVCC_FLAGS` in the Makefile (e.g., `-arch=sm_86` for A30/A40).

## Download Matrices

```bash
chmod +x scripts/download_matrices.sh
./scripts/download_matrices.sh
```

This downloads 10 matrices into `matrices/`. The selection covers:
- Small regular matrices (cage4, olm1000)
- Medium structured matrices (west2021, mac_econ_fwd500)
- FEM matrices with regular nnz/row (cant, consph, cop20k_A)
- Protein structure (pdb1HYS)
- Power-law web graph (webbase-1M)
- Circuit simulation with irregular structure (scircuit)

## Run

Single matrix:
```bash
./bin/spmv matrices/cant/cant.mtx
```

Optional tolerance parameter:
```bash
./bin/spmv matrices/cant/cant.mtx 1e-3
```

All matrices on the cluster:
```bash
sbatch scripts/sbatch_run.sh
```

## Output

For each matrix, the program prints:
- Matrix dimensions and NNZ count
- GPU device properties
- Per-kernel timing (mean ± std over 10 iterations, after 4 warmup runs)
- GFLOP/s (using 2·nnz flops per SpMV)
- Effective memory bandwidth (GB/s)
- Validation result against CPU reference (with configurable tolerance)
- CSV summary line for easy data collection

## Kernels

| Kernel | Strategy | Key Feature |
|--------|----------|-------------|
| CSR-Scalar | 1 thread per row | Simple, poor load balance on irregular matrices |
| CSR-Vector (shuffle) | 1 warp per row | Warp-level reduction via `__shfl_down_sync` |
| CSR-Vector (shmem) | 1 warp per row | Reduction through shared memory |
| cuSPARSE | Library | NVIDIA's optimized SpMV implementation |

## Validation

GPU results are compared element-wise against the CPU CSR implementation. Default tolerance is 1e-4 (configurable via command line). Mismatches are reported with index and values for debugging.
