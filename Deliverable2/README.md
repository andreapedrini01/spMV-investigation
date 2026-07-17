# Distributed SpMV with CUDA-aware MPI — Deliverable 2

Distributed sparse matrix-vector multiplication `y = A*x` over multiple GPUs,
one MPI rank per GPU. Rows are distributed with a 1D cyclic rule
`owner(i) = i % P` (the k-th local row of rank `r` has global index `r + k*P`).
The node-local product uses the CSR-Vector kernel from Deliverable 1.

The input vector `x` is exchanged with two communication modes:

- **allgather** — every rank reconstructs the whole `x` with `MPI_Allgatherv`
  (memory `O(N)` per rank);
- **ghost** — each rank exchanges only the remote entries it references with
  `MPI_Alltoallv`, using a global→compact index remap (memory
  `O(n_local + n_ghost)`, lower communication volume).

Each mode runs over two transports: host **staging** and **CUDA-aware** MPI
(device pointers passed to MPI and moved device-to-device by a GPUDirect-capable
build). Both produce identical results.

The results reported in the paper were produced with the setup and scripts
below, on the `edu01` node (4× NVIDIA A30) of the DISI cluster.

## Environment (DISI cluster, edu-short)

- Partition `edu-short`, account `gpu.computing26`, up to 4× NVIDIA A30 (`sm_80`).
- Modules: `OpenMpi/4.1.5-CUDA-12.3.2` (CUDA-aware, `smcuda` BTL) and `CUDA/12.3.2`.
- The build must run on a **compute node** (e.g. `edu01`): the login node cannot
  assemble the objects with the OpenMPI module loaded.
- `edu-short` has a 5-minute wall-time limit per job; the workflow builds once
  and runs one matrix per strong-scaling job.

To confirm the CUDA-aware transport is active: `ompi_info | grep -i cuda` shows
the `smcuda` BTL, and the program prints `MPI CUDA-aware support (runtime): 1`.
The generic `OpenMPI` module (plain UCX) stages device buffers through the host
and yields much slower "cuda-aware" times.

## Reproduce the results

    # 1. Fetch the SuiteSparse matrices used for strong scaling
    bash scripts/download_matrices.sh

    # 2. Build, then run everything (runs start after the build succeeds)
    bash scripts/submit_all.sh

    # 3. When the jobs finish, merge the per-job CSVs
    bash scripts/collect_csv.sh
    #    -> outputs/strong_all.csv, outputs/weak_all.csv

The steps can also be submitted individually (build must finish first):

    sbatch scripts/sbatch_build.sh
    sbatch scripts/sbatch_strong.sh matrices/cant/cant.mtx   # one matrix per job
    sbatch scripts/sbatch_weak.sh                            # weak scaling

Or build interactively on a compute node:

    srun --partition=edu-short --account=gpu.computing26 --nodes=1 \
         --gres=gpu:0 --ntasks=1 --cpus-per-task=8 --time=00:05:00 --pty bash
    module purge
    module load OpenMpi/4.1.5-CUDA-12.3.2
    module load CUDA/12.3.2
    make -j

`make ARCH=sm_XX` targets a different GPU; `make NCCL=1` and `make NVML=1`
enable the optional NCCL AllGather transport and the per-rank GPU-UUID print.

## Run a single case

    mpirun -np 4 ./bin/dspmv matrices/cant/cant.mtx      # real matrix
    mpirun -np 4 ./bin/dspmv --gen 200000 32             # generated (weak-scaling size)

Options:

    --mode allgather|ghost         run one mode (default: both)
    --transport staging|aware|all  run one transport (default: staging + aware)
    --pattern random|banded        generator pattern (default: random)
    --iters N / --warmup N         timed / warmup iterations (default 10 / 4)
    --seed S                       generator seed (default 42)
    --no-check                     skip validation

## Output

Each `(matrix, P, mode, transport)` case prints one CSV line on stderr, prefixed
`CSVROW,`; the scripts strip the prefix into `outputs/*.csv`. Timing uses
`MPI_Wtime` around a barrier before each iteration and takes the maximum over
ranks; `t_comp` is measured with CUDA events and `t_comm` around the exchange,
so `t_comp + t_comm` add up to `t_spmv`. Columns:

    matrix, ranks, mode, transport, nnz_global, t_spmv_ms, std_ms,
    t_comm_ms, t_comp_ms, gflops, nnz_min, nnz_max,
    commvol_send_max, commvol_recv_max, mem_bytes_max, abs_error

## Validation

`x` is derived deterministically from the column index, so every rank recomputes
a CPU reference for the rows it owns and compares it against the distributed
result (`abs_error`). For file matrices the distributed `y` is additionally
gathered on rank 0 and checked against a full serial SpMV, validating the
distribution end to end.
