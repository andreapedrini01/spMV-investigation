# Distributed SpMV with MPI CUDA-aware (Deliverable 2)

Distributed sparse matrix-vector multiplication `y = A*x` across multiple GPUs.
One MPI rank drives one GPU. Rows are distributed with a 1D cyclic rule
`owner(i) = i % P`, so the k-th local row of rank r has global index `r + k*P`.
The local kernel is the CSR-Vector kernel reused from Deliverable 1.

Two communication modes exchange the input vector:

- `allgather`: every rank gathers the whole `x` (`MPI_Allgatherv`). Simple,
  memory `O(N)` per rank.
- `ghost`: each rank exchanges only the remote entries it references
  (`MPI_Alltoallv`), with a local/global index remap into a compact buffer.
  Memory `O(n_local + n_ghost)` and lower communication volume.

Each mode runs with host staging, CUDA-aware MPI (device pointers passed to
MPI), and optionally NCCL (`allgather` only).

## Build

On the cluster:

    module load OpenMPI
    module load CUDA/12.5.0
    make                 # plain build (MPI staging + CUDA-aware)
    make NCCL=1 NVML=1   # add the NCCL transport and the GPU-UUID check

The build targets `sm_80` (A30). Change with `make ARCH=sm_XX`.

## Run

Real matrix (strong scaling):

    mpirun -np 4 ./bin/dspmv matrices/cant/cant.mtx

Generated matrix (weak scaling), with `N = rows_per_rank * P`:

    mpirun -np 4 ./bin/dspmv --gen 200000 32 --pattern random

Options:

    --mode allgather|ghost        run only one mode (default: both)
    --transport staging|aware|nccl|all   (default: staging + aware [+ nccl])
    --iters N                     timed iterations (default 10)
    --warmup N                    warmup iterations (default 4)
    --pattern random|banded       generator pattern (default random)
    --seed S                      generator seed (default 42)
    --no-check                    skip validation

## Scripts

    bash scripts/download_matrices.sh   # fetch the SuiteSparse matrices
    sbatch scripts/sbatch_strong.sh     # strong scaling, P in {1,2,4}
    sbatch scripts/sbatch_weak.sh       # weak scaling, P in {1,2,4}

Both jobs write a full log and a `.csv` under `outputs/`.

## Output

Timing uses `MPI_Wtime` with a barrier before each iteration; the per-iteration
SpMV time is the maximum over ranks. The kernel time (`t_comp`) is measured with
CUDA events and the communication time (`t_comm`) around the exchange, so the
two add up to the SpMV time. A CSV line per (matrix, P, mode, transport) is
emitted on stderr (prefixed `CSVROW,`) with these columns:

    matrix, ranks, mode, transport, nnz_global, t_spmv_ms, std_ms,
    t_comm_ms, t_comp_ms, gflops, nnz_min, nnz_max,
    commvol_send_max, commvol_recv_max, mem_bytes_max, abs_error

`report/generate_plots.py` turns a CSV into the scaling figures.

## Validation

Because `x` is derived deterministically from the column index, every rank
recomputes a CPU reference for the rows it owns and compares it against the
distributed result (`abs_error`). For file matrices the distributed `y` is also
gathered on rank 0 and checked against a full serial SpMV, which validates the
distribution end to end.
