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

## Cluster notes (DISI edu-short)

- Build on a **compute node**, not the login node `baldo`: loading the OpenMPI
  module on the login CPU pulls a `binutils` whose assembler crashes there.
  The compute nodes (edu01/edu02) build fine.
- The `edu-short` partition has a **5-minute wall-time limit**, so we build once
  and keep each run job short (one matrix per strong job).
- The CUDA-aware transport needs an **OpenMPI built with CUDA support**. Check
  after loading with `ompi_info | grep -i cuda`, and confirm at runtime: the
  program prints `MPI CUDA-aware support (runtime): 1`. If it reports 0, load the
  CUDA-aware OpenMPI module instead (e.g. `OpenMpi/4.1.5-CUDA-...`). Staging and
  NCCL do not need it.
- edu01 has hyper-threading on and is shared, so treat absolute times as
  indicative; the warmup + repeated iterations and reported std cover the noise.

## Build

Grab a short interactive shell on a compute node (≤ 5 min), or use the build
job below:

    srun --partition=edu-short --account=gpu.computing26 --nodes=1 \
         --gres=gpu:0 --ntasks=1 --cpus-per-task=8 --time=00:05:00 --pty bash

    module purge
    module load OpenMpi/4.1.5-CUDA-12.3.2   # CUDA-aware MPI (smcuda BTL, GPUDirect P2P)
    module load CUDA/12.3.2
    make -j                    # MPI staging + CUDA-aware
    make -j NCCL=1 NVML=1      # add NCCL (if available) and the GPU-UUID check

Use the CUDA-aware OpenMPI (`OpenMpi/4.1.5-CUDA-12.3.2`, `ompi_info` shows the
`smcuda` BTL): the generic `OpenMPI` module uses plain UCX, so device pointers
passed to MPI are staged through the host and the "cuda-aware" transport is much
slower. The Makefile auto-detects the MPI paths from `mpicc`, so it works with
either module. The build targets `sm_80` (A30); change with `make ARCH=sm_XX`.

## Run

Interactive smoke test (add `--oversubscribe` if the shell has few slots):

    mpirun --oversubscribe -np 4 ./bin/dspmv --gen 20000 32     # generated
    mpirun -np 4 ./bin/dspmv matrices/cant/cant.mtx             # real matrix

Options:

    --mode allgather|ghost        run only one mode (default: both)
    --transport staging|aware|nccl|all   (default: staging + aware [+ nccl])
    --iters N                     timed iterations (default 10)
    --warmup N                    warmup iterations (default 4)
    --pattern random|banded       generator pattern (default random)
    --seed S                      generator seed (default 42)
    --no-check                    skip validation

## Scripts (batch)

From the login node, submit everything at once (build, then runs after it):

    bash scripts/download_matrices.sh   # fetch the SuiteSparse matrices
    bash scripts/submit_all.sh          # build job + one strong job per matrix + weak job
    # when the jobs finish:
    bash scripts/collect_csv.sh         # merge into outputs/strong_all.csv, weak_all.csv

Or run the pieces manually (build must finish first):

    sbatch scripts/sbatch_build.sh
    sbatch scripts/sbatch_strong.sh matrices/cant/cant.mtx   # one matrix per job
    sbatch scripts/sbatch_weak.sh

Each job writes a full log and a `.csv` under `outputs/`.

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
