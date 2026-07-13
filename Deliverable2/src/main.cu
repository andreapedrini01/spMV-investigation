#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>

#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h> /* MPIX_Query_cuda_support (OpenMPI) */
#endif

#ifdef USE_NVML
#include <nvml.h>
#endif

#include "utils.h"
#include "distributed.h"
#include "gen_matrix.h"
#include "spmv_cpu_ref.h"

/* one rank per GPU: split the world by node, then map the node-local rank to a
 * device (works on a single node and across nodes) */
static int map_rank_to_gpu(int rank)
{
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    int dev_count = 0;
    cudaError_t e = cudaGetDeviceCount(&dev_count);
    if (e != cudaSuccess || dev_count == 0) {
        fprintf(stderr, "Rank %d: no CUDA device visible (%s)\n", rank, cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    int dev = local_rank % dev_count;
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

#ifdef USE_NVML
    /* confirm distinct physical GPUs by printing each device's UUID (once) */
    char uuid[NVML_DEVICE_UUID_BUFFER_SIZE] = "n/a";
    if (nvmlInit() == NVML_SUCCESS) {
        nvmlDevice_t h;
        if (nvmlDeviceGetHandleByIndex(dev, &h) == NVML_SUCCESS)
            nvmlDeviceGetUUID(h, uuid, NVML_DEVICE_UUID_BUFFER_SIZE);
        nvmlShutdown();
    }
    printf("Rank %d -> local rank %d -> GPU %d (%s, %s)\n", rank, local_rank, dev, prop.name, uuid);
#else
    printf("Rank %d -> local rank %d -> GPU %d (%s)\n", rank, local_rank, dev, prop.name);
#endif
    return local_rank;
}

/* sum of absolute differences, reduced across ranks */
static double reduce_error(float local_err, MPI_Comm comm)
{
    double le = local_err, ge = 0.0;
    MPI_Allreduce(&le, &ge, 1, MPI_DOUBLE, MPI_SUM, comm);
    return ge;
}

static long long allreduce_ll(long long v, MPI_Op op, MPI_Comm comm)
{
    long long r = 0;
    MPI_Allreduce(&v, &r, 1, MPI_LONG_LONG, op, comm);
    return r;
}

static void mean_std(const double *v, int n, double *mean, double *std)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += v[i];
    *mean = s / n;
    double var = 0.0;
    for (int i = 0; i < n; i++) { double d = v[i] - *mean; var += d * d; }
    *std = (n > 1) ? sqrt(var / (n - 1)) : 0.0;
}

static const char *transport_name(Transport t)
{
    switch (t) {
        case XPORT_STAGING: return "staging";
        case XPORT_CUDA_AWARE: return "cuda-aware";
        case XPORT_NCCL: return "nccl";
    }
    return "?";
}

/* run warmup + timed iterations for one (mode, transport); every rank ends up
 * with the same reduced statistics (Allreduce max over ranks per iteration) */
static void measure_variant(CommMode mode, Transport xport, SpmvCtxA *A, SpmvCtxB *B,
                            int warmup, int iters, MPI_Comm comm,
                            double *tspmv_ms, double *std_ms, double *tcomm_ms, double *tcomp_ms)
{
    double tc = 0.0, tp = 0.0;
    for (int w = 0; w < warmup; w++) {
        if (mode == MODE_ALLGATHER) spmvA_run(A, comm, xport, &tc, &tp);
        else spmvB_run(B, comm, xport, &tc, &tp);
    }

    double *tot = (double *)malloc((size_t)iters * sizeof(double));
    double *cm = (double *)malloc((size_t)iters * sizeof(double));
    double *cp = (double *)malloc((size_t)iters * sizeof(double));

    for (int it = 0; it < iters; it++) {
        MPI_Barrier(comm);
        double t0 = MPI_Wtime();
        if (mode == MODE_ALLGATHER) spmvA_run(A, comm, xport, &tc, &tp);
        else spmvB_run(B, comm, xport, &tc, &tp);
        double t1 = MPI_Wtime();

        double my = t1 - t0, gt = 0.0, gc = 0.0, gp = 0.0;
        MPI_Allreduce(&my, &gt, 1, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(&tc, &gc, 1, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(&tp, &gp, 1, MPI_DOUBLE, MPI_MAX, comm);
        tot[it] = gt * 1000.0;
        cm[it] = gc * 1000.0;
        cp[it] = gp * 1000.0;
    }

    double m, s;
    mean_std(tot, iters, &m, &s);
    *tspmv_ms = m; *std_ms = s;
    double dummy;
    mean_std(cm, iters, tcomm_ms, &dummy);
    mean_std(cp, iters, tcomp_ms, &dummy);

    free(tot); free(cm); free(cp);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank = 0, P = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm comm = MPI_COMM_WORLD;

    /* ---- CLI ---- */
    const char *matrix_path = NULL;
    int gen_mode = 0, gen_rows = 0, gen_nnz = 0;
    GenPattern gen_pat = GEN_RANDOM;
    unsigned int seed = 42;
    int warmup = 4, iters = 10;
    int do_A = 1, do_B = 1;
    int want_staging = 1, want_aware = 1, want_nccl = -1; /* -1 = auto */
    int do_check = 1;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--gen") && i + 2 < argc) {
            gen_mode = 1; gen_rows = atoi(argv[++i]); gen_nnz = atoi(argv[++i]);
        } else if (!strcmp(a, "--pattern") && i + 1 < argc) {
            gen_pat = !strcmp(argv[++i], "banded") ? GEN_BANDED : GEN_RANDOM;
        } else if (!strcmp(a, "--seed") && i + 1 < argc) {
            seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(a, "--iters") && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (!strcmp(a, "--warmup") && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (!strcmp(a, "--mode") && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "allgather")) do_B = 0;
            else if (!strcmp(m, "ghost")) do_A = 0;
        } else if (!strcmp(a, "--transport") && i + 1 < argc) {
            const char *t = argv[++i];
            if (!strcmp(t, "all")) { want_staging = want_aware = 1; want_nccl = -1; }
            else { want_staging = want_aware = 0; want_nccl = 0;
                if (!strcmp(t, "staging")) want_staging = 1;
                else if (!strcmp(t, "aware")) want_aware = 1;
                else if (!strcmp(t, "nccl")) want_nccl = 1; }
        } else if (!strcmp(a, "--no-check")) {
            do_check = 0;
        } else if (a[0] != '-' && !gen_mode) {
            matrix_path = a;
        }
    }

    if (!gen_mode && !matrix_path) {
        if (rank == 0)
            fprintf(stderr, "Usage: mpirun -np P %s <matrix.mtx> [opts]\n"
                            "       mpirun -np P %s --gen <rows_per_rank> <nnz_per_row> "
                            "[--pattern random|banded] [--seed S] [opts]\n"
                            "opts: --mode allgather|ghost  --transport staging|aware|nccl|all  "
                            "--iters N --warmup N --no-check\n", argv[0], argv[0]);
        MPI_Finalize();
        return 1;
    }

    int local_rank = map_rank_to_gpu(rank);
    (void)local_rank;

    if (spmv_nccl_available()) spmv_nccl_init(rank, P, comm);

    /* transports to run (NCCL only for Mode A) */
    Transport xlist[3];
    int nx = 0;
    if (want_staging) xlist[nx++] = XPORT_STAGING;
    if (want_aware) xlist[nx++] = XPORT_CUDA_AWARE;
    int use_nccl = (want_nccl == 1) || (want_nccl == -1 && spmv_nccl_available());
    if (want_nccl == 1 && !spmv_nccl_available()) {
        if (rank == 0) fprintf(stderr, "NCCL requested but binary built without USE_NCCL\n");
        use_nccl = 0;
    }
    if (use_nccl) xlist[nx++] = XPORT_NCCL;

    if (rank == 0) {
        printf("========================================================\n");
        printf("Distributed SpMV (1D cyclic) | ranks=%d\n", P);
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        printf("MPI CUDA-aware support (runtime): %d\n", MPIX_Query_cuda_support());
#else
        printf("MPI CUDA-aware support: not reported by MPIX; assuming enabled\n");
#endif
        printf("NCCL: %s\n", spmv_nccl_available() ? "available" : "not built");
        print_device_properties();
        printf("========================================================\n");
    }

    /* ---- build the local matrix ---- */
    int M = 0, N = 0;
    int *coo_row = NULL, *coo_col = NULL;
    float *coo_val = NULL;
    int file_nnz = 0;
    LocalCSR mat;

    if (gen_mode) {
        gen_local_csr(rank, P, gen_rows, gen_nnz, gen_pat, seed, &mat);
        M = mat.nrows_global; N = mat.ncols;
        if (rank == 0)
            printf("Generated matrix: N=%d, rows/rank=%d, nnz/row=%d, pattern=%s, seed=%u\n",
                   N, gen_rows, gen_nnz, gen_pat == GEN_BANDED ? "banded" : "random", seed);
    } else {
        if (rank == 0) {
            if (read_mtx_file(matrix_path, &M, &N, &file_nnz, &coo_row, &coo_col, &coo_val) != 0) {
                fprintf(stderr, "Failed to read %s\n", matrix_path);
                MPI_Abort(comm, EXIT_FAILURE);
            }
            printf("Matrix %s: M=%d N=%d nnz=%d (%.1f nnz/row)\n",
                   matrix_path, M, N, file_nnz, (double)file_nnz / M);
        }
        distribute_matrix_cyclic(rank, P, comm, M, N, file_nnz,
                                 coo_row, coo_col, coo_val, &mat);
        MPI_Bcast(&M, 1, MPI_INT, 0, comm);
        MPI_Bcast(&N, 1, MPI_INT, 0, comm);
    }

    long long nnz_global = allreduce_ll(mat.nnz_local, MPI_SUM, comm);

    /* ---- per-rank CPU reference (x is deterministic, so no communication) ---- */
    float *y_ref_local = (float *)malloc((size_t)(mat.n_local > 0 ? mat.n_local : 1) * sizeof(float));
    spmv_local_reference(&mat, y_ref_local);
    float *y_test = (float *)malloc((size_t)(mat.n_local > 0 ? mat.n_local : 1) * sizeof(float));

    /* ---- distribution/NNZ + memory metrics (per mode, transport-independent) ---- */
    long long nnz_min = allreduce_ll(mat.nnz_local, MPI_MIN, comm);
    long long nnz_max = allreduce_ll(mat.nnz_local, MPI_MAX, comm);
    if (rank == 0) {
        printf("NNZ per rank: min=%lld avg=%.1f max=%lld total=%lld  (imbalance max/avg=%.2f)\n",
               nnz_min, (double)nnz_global / P, nnz_max,
               nnz_global, (double)nnz_max / ((double)nnz_global / P));
        printf("--------------------------------------------------------\n");
        printf("%-10s %-11s %10s %10s %10s %9s %9s\n",
               "mode", "transport", "t_spmv[ms]", "t_comm[ms]", "t_comp[ms]", "GFLOP/s", "err");
        fprintf(stderr, "CSVROW,matrix,ranks,mode,transport,nnz_global,t_spmv_ms,std_ms,"
                        "t_comm_ms,t_comp_ms,gflops,nnz_min,nnz_max,commvol_send_max,"
                        "commvol_recv_max,mem_bytes_max,abs_error\n");
    }

    const char *mtx_label = gen_mode ? (gen_pat == GEN_BANDED ? "gen-banded" : "gen-random")
                                     : (strrchr(matrix_path, '/') ? strrchr(matrix_path, '/') + 1
                                                                   : matrix_path);

    /* ---- Mode A: Allgather ---- */
    if (do_A) {
        SpmvCtxA ctx;
        spmvA_init(&ctx, &mat, rank, P, comm);
        long long send_max = allreduce_ll((long long)(P - 1) * ctx.nx_local, MPI_MAX, comm);
        long long recv_max = allreduce_ll((long long)(N - ctx.nx_local), MPI_MAX, comm);
        long long mem_max = allreduce_ll((long long)spmvA_bytes(&ctx), MPI_MAX, comm);

        for (int t = 0; t < nx; t++) {
            double ts, sd, tcm, tcp;
            measure_variant(MODE_ALLGATHER, xlist[t], &ctx, NULL, warmup, iters, comm, &ts, &sd, &tcm, &tcp);
            double err = 0.0;
            if (do_check) {
                spmvA_copy_y(&ctx, y_test);
                err = reduce_error(validate_results(y_ref_local, y_test, mat.n_local), comm);
            }
            double gflops = (2.0 * (double)nnz_global) / ((ts / 1000.0) * 1e9);
            if (rank == 0) {
                printf("%-10s %-11s %10.4f %10.4f %10.4f %9.2f %9.2e\n",
                       "allgather", transport_name(xlist[t]), ts, tcm, tcp, gflops, err);
                fprintf(stderr, "CSVROW,%s,%d,allgather,%s,%lld,%.4f,%.4f,%.4f,%.4f,%.2f,"
                                "%lld,%lld,%lld,%lld,%lld,%.3e\n",
                        mtx_label, P, transport_name(xlist[t]), nnz_global, ts, sd, tcm, tcp,
                        gflops, nnz_min, nnz_max, send_max, recv_max, mem_max, err);
            }
        }
        spmvA_free(&ctx);
    }

    /* ---- Mode B: ghost exchange (staging + aware only) ---- */
    if (do_B) {
        SpmvCtxB ctx;
        spmvB_init(&ctx, &mat, rank, P, comm);
        long long send_max = allreduce_ll((long long)ctx.total_send, MPI_MAX, comm);
        long long recv_max = allreduce_ll((long long)ctx.n_ghost, MPI_MAX, comm);
        long long send_avg_sum = allreduce_ll((long long)ctx.total_send, MPI_SUM, comm);
        long long mem_max = allreduce_ll((long long)spmvB_bytes(&ctx), MPI_MAX, comm);
        if (rank == 0)
            printf("Ghost comm volume/rank (send): avg=%.0f max=%lld values/SpMV\n",
                   (double)send_avg_sum / P, send_max);

        for (int t = 0; t < nx; t++) {
            if (xlist[t] == XPORT_NCCL) continue; /* ghost mode: MPI transports only */
            double ts, sd, tcm, tcp;
            measure_variant(MODE_GHOST, xlist[t], NULL, &ctx, warmup, iters, comm, &ts, &sd, &tcm, &tcp);
            double err = 0.0;
            if (do_check) {
                spmvB_copy_y(&ctx, y_test);
                err = reduce_error(validate_results(y_ref_local, y_test, mat.n_local), comm);
            }
            double gflops = (2.0 * (double)nnz_global) / ((ts / 1000.0) * 1e9);
            if (rank == 0) {
                printf("%-10s %-11s %10.4f %10.4f %10.4f %9.2f %9.2e\n",
                       "ghost", transport_name(xlist[t]), ts, tcm, tcp, gflops, err);
                fprintf(stderr, "CSVROW,%s,%d,ghost,%s,%lld,%.4f,%.4f,%.4f,%.4f,%.2f,"
                                "%lld,%lld,%lld,%lld,%lld,%.3e\n",
                        mtx_label, P, transport_name(xlist[t]), nnz_global, ts, sd, tcm, tcp,
                        gflops, nnz_min, nnz_max, send_max, recv_max, mem_max, err);
            }
        }
        spmvB_free(&ctx);
    }

    /* ---- file mode: validate the distribution end-to-end against a full
     * serial SpMV on rank 0, then reassemble the distributed y ---- */
    if (do_check && !gen_mode) {
        SpmvCtxA ctx;
        spmvA_init(&ctx, &mat, rank, P, comm);
        double tc, tp;
        spmvA_run(&ctx, comm, want_aware ? XPORT_CUDA_AWARE : XPORT_STAGING, &tc, &tp);
        spmvA_copy_y(&ctx, y_test);

        float *y_dist_full = NULL, *y_ref_full = NULL;
        if (rank == 0) {
            y_dist_full = (float *)malloc((size_t)M * sizeof(float));
            y_ref_full = (float *)malloc((size_t)M * sizeof(float));
        }
        gather_y_to_root(y_test, mat.n_local, rank, P, M, comm, y_dist_full);

        if (rank == 0) {
            int *rp, *ci; float *cv;
            coo_to_csr(coo_row, coo_col, coo_val, M, file_nnz, &rp, &ci, &cv);
            float *x_full = (float *)malloc((size_t)N * sizeof(float));
            for (int j = 0; j < N; j++) x_full[j] = xvec_value(j);
            spmv_csr_full_reference(rp, ci, cv, M, x_full, y_ref_full);
            float e = validate_results(y_ref_full, y_dist_full, M);
            printf("--------------------------------------------------------\n");
            printf("End-to-end check vs serial full SpMV on rank 0: abs error = %.3e\n", e);
            free(rp); free(ci); free(cv); free(x_full);
            free(y_dist_full); free(y_ref_full);
        }
        spmvA_free(&ctx);
    }

    free(y_ref_local); free(y_test);
    free_local_csr(&mat);
    if (rank == 0 && !gen_mode) { free(coo_row); free(coo_col); free(coo_val); }

    if (spmv_nccl_available()) spmv_nccl_finalize();
    MPI_Finalize();
    return 0;
}
