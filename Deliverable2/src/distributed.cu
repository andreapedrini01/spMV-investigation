#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "distributed.h"
#include "utils.h"
#include "spmv_local.h"

#ifdef USE_NCCL
#include <nccl.h>
#endif

#define TPB 256
static inline int nblocks(int n) { return (n + TPB - 1) / TPB; }

/* ---- small allocation helpers (avoid zero-size allocations) ---- */
static void *dmalloc(size_t bytes)
{
    void *p = NULL;
    if (bytes == 0) bytes = 1;
    CUDA_CHECK(cudaMalloc(&p, bytes));
    return p;
}
static void *hmalloc(size_t bytes)
{
    if (bytes == 0) bytes = 1;
    void *p = malloc(bytes);
    if (!p) { fprintf(stderr, "host malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); }
    return p;
}

/* ---- reorder / pack / scatter kernels ---- */
static __global__ void k_scatter_by_perm(int n, const int *perm, const float *in, float *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[perm[i]] = in[i];
}
static __global__ void k_gather_by_index(int n, const int *idx, const float *in, float *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[idx[i]];
}
static __global__ void k_permute(int n, const int *dst, const int *src, const float *in, float *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[dst[i]] = in[src[i]];
}

/* ---- comparators / search ---- */
static int cmp_int(const void *a, const void *b)
{
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}
static int bsearch_int(const int *arr, int n, int key)
{
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (arr[mid] == key) return mid;
        if (arr[mid] < key) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

/* ======================================================================== */
static void build_local_csr(int rank, int P, int M, int N, int nnz_local,
                            const int *lr, const int *lc, const float *lv,
                            LocalCSR *out)
{
    int n_local = count_cyclic(M, P, rank);
    int *row_ptr = (int *)calloc(n_local + 1, sizeof(int));

    /* count per local row; local row k has global index rank + k*P, so k = i/P */
    for (int e = 0; e < nnz_local; e++) {
        int k = lr[e] / P;
        row_ptr[k + 1]++;
    }
    for (int k = 0; k < n_local; k++)
        row_ptr[k + 1] += row_ptr[k];

    int *col = (int *)hmalloc((size_t)nnz_local * sizeof(int));
    float *val = (float *)hmalloc((size_t)nnz_local * sizeof(float));
    int *cursor = (int *)hmalloc((size_t)(n_local + 1) * sizeof(int));
    memcpy(cursor, row_ptr, (size_t)(n_local + 1) * sizeof(int));

    for (int e = 0; e < nnz_local; e++) {
        int k = lr[e] / P;
        int d = cursor[k]++;
        col[d] = lc[e];   /* keep GLOBAL column index */
        val[d] = lv[e];
    }
    free(cursor);

    out->n_local = n_local;
    out->ncols = N;
    out->nnz_local = nnz_local;
    out->nrows_global = M;
    out->row_ptr = row_ptr;
    out->col_idx = col;
    out->val = val;
}

void distribute_matrix_cyclic(int rank, int P, MPI_Comm comm,
                              int M, int N, int nnz_global,
                              const int *coo_row, const int *coo_col, const float *coo_val,
                              LocalCSR *out)
{
    int dims[3] = { M, N, nnz_global };
    MPI_Bcast(dims, 3, MPI_INT, 0, comm);
    M = dims[0]; N = dims[1]; nnz_global = dims[2];

    int *sendcounts = NULL, *sdispls = NULL;
    int *sr = NULL, *sc = NULL;
    float *sv = NULL;

    if (rank == 0) {
        sendcounts = (int *)calloc(P, sizeof(int));
        sdispls = (int *)malloc((size_t)P * sizeof(int));
        for (int e = 0; e < nnz_global; e++)
            sendcounts[coo_row[e] % P]++;
        sdispls[0] = 0;
        for (int p = 1; p < P; p++)
            sdispls[p] = sdispls[p - 1] + sendcounts[p - 1];

        /* bucket the nonzeros by destination rank (counting sort) */
        sr = (int *)hmalloc((size_t)nnz_global * sizeof(int));
        sc = (int *)hmalloc((size_t)nnz_global * sizeof(int));
        sv = (float *)hmalloc((size_t)nnz_global * sizeof(float));
        int *cursor = (int *)malloc((size_t)P * sizeof(int));
        memcpy(cursor, sdispls, (size_t)P * sizeof(int));
        for (int e = 0; e < nnz_global; e++) {
            int dst = coo_row[e] % P;
            int pos = cursor[dst]++;
            sr[pos] = coo_row[e];
            sc[pos] = coo_col[e];
            sv[pos] = coo_val[e];
        }
        free(cursor);
    }

    int nnz_local = 0;
    MPI_Scatter(sendcounts, 1, MPI_INT, &nnz_local, 1, MPI_INT, 0, comm);

    int *lr = (int *)hmalloc((size_t)nnz_local * sizeof(int));
    int *lc = (int *)hmalloc((size_t)nnz_local * sizeof(int));
    float *lv = (float *)hmalloc((size_t)nnz_local * sizeof(float));

    MPI_Scatterv(sr, sendcounts, sdispls, MPI_INT, lr, nnz_local, MPI_INT, 0, comm);
    MPI_Scatterv(sc, sendcounts, sdispls, MPI_INT, lc, nnz_local, MPI_INT, 0, comm);
    MPI_Scatterv(sv, sendcounts, sdispls, MPI_FLOAT, lv, nnz_local, MPI_FLOAT, 0, comm);

    build_local_csr(rank, P, M, N, nnz_local, lr, lc, lv, out);

    free(lr); free(lc); free(lv);
    if (rank == 0) { free(sr); free(sc); free(sv); free(sendcounts); free(sdispls); }

    /* sanity: the pieces must sum back to the global nnz */
    long long loc = nnz_local, tot = 0;
    MPI_Reduce(&loc, &tot, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
    if (rank == 0 && tot != (long long)nnz_global)
        fprintf(stderr, "WARNING: distributed nnz %lld != global nnz %d\n", tot, nnz_global);
}

void free_local_csr(LocalCSR *m)
{
    if (!m) return;
    free(m->row_ptr); free(m->col_idx); free(m->val);
    m->row_ptr = NULL; m->col_idx = NULL; m->val = NULL;
}

/* ======================================================================== */
/* Mode A: Allgather the full x on every rank.                              */
/* ======================================================================== */
void spmvA_init(SpmvCtxA *ctx, const LocalCSR *m, int rank, int P, MPI_Comm comm)
{
    (void)comm;
    memset(ctx, 0, sizeof(*ctx));
    int N = m->ncols;
    ctx->N = N; ctx->P = P; ctx->rank = rank;
    ctx->n_local = m->n_local; ctx->nnz_local = m->nnz_local;
    ctx->nx_local = count_cyclic(N, P, rank);
    ctx->maxnx = (N + P - 1) / P;

    ctx->recvcounts = (int *)hmalloc((size_t)P * sizeof(int));
    ctx->recvdispls = (int *)hmalloc((size_t)P * sizeof(int));
    for (int p = 0; p < P; p++) ctx->recvcounts[p] = count_cyclic(N, P, p);
    ctx->recvdispls[0] = 0;
    for (int p = 1; p < P; p++) ctx->recvdispls[p] = ctx->recvdispls[p - 1] + ctx->recvcounts[p - 1];

    /* gathered position (rank-major, contiguous) -> global index */
    ctx->gather_perm = (int *)hmalloc((size_t)N * sizeof(int));
    for (int p = 0; p < P; p++)
        for (int mm = 0; mm < ctx->recvcounts[p]; mm++)
            ctx->gather_perm[ctx->recvdispls[p] + mm] = p + mm * P;
    ctx->d_gather_perm = (int *)dmalloc((size_t)N * sizeof(int));
    CUDA_CHECK(cudaMemcpy(ctx->d_gather_perm, ctx->gather_perm, (size_t)N * sizeof(int),
                          cudaMemcpyHostToDevice));

    /* device CSR (global column indices) */
    ctx->d_row_ptr = (int *)dmalloc((size_t)(m->n_local + 1) * sizeof(int));
    ctx->d_col_idx = (int *)dmalloc((size_t)m->nnz_local * sizeof(int));
    ctx->d_val = (float *)dmalloc((size_t)m->nnz_local * sizeof(float));
    ctx->d_y = (float *)dmalloc((size_t)m->n_local * sizeof(float));
    CUDA_CHECK(cudaMemcpy(ctx->d_row_ptr, m->row_ptr, (size_t)(m->n_local + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_col_idx, m->col_idx, (size_t)m->nnz_local * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_val, m->val, (size_t)m->nnz_local * sizeof(float),
                          cudaMemcpyHostToDevice));

    /* owned x (padded to maxnx so the same buffer feeds ncclAllGather) */
    ctx->h_x_owned = (float *)hmalloc((size_t)ctx->maxnx * sizeof(float));
    for (int mm = 0; mm < ctx->maxnx; mm++)
        ctx->h_x_owned[mm] = (mm < ctx->nx_local) ? xvec_value(rank + mm * P) : 0.0f;
    ctx->h_x_gathered = (float *)hmalloc((size_t)N * sizeof(float));
    ctx->h_x_full = (float *)hmalloc((size_t)N * sizeof(float));

    ctx->d_x_owned = (float *)dmalloc((size_t)ctx->maxnx * sizeof(float));
    ctx->d_x_gathered = (float *)dmalloc((size_t)N * sizeof(float));
    ctx->d_x_full = (float *)dmalloc((size_t)N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(ctx->d_x_owned, ctx->h_x_owned, (size_t)ctx->maxnx * sizeof(float),
                          cudaMemcpyHostToDevice));

#ifdef USE_NCCL
    /* padded (strided) gathered layout for ncclAllGather: block p spans
     * [p*maxnx, p*maxnx + nx_local(p)); the tail is padding to ignore. */
    ctx->d_x_gathered_nccl = (float *)dmalloc((size_t)P * ctx->maxnx * sizeof(float));
    int *srcpos = (int *)hmalloc((size_t)N * sizeof(int));
    for (int p = 0; p < P; p++)
        for (int mm = 0; mm < ctx->recvcounts[p]; mm++)
            srcpos[ctx->recvdispls[p] + mm] = p * ctx->maxnx + mm;
    ctx->d_nccl_srcpos = (int *)dmalloc((size_t)N * sizeof(int));
    CUDA_CHECK(cudaMemcpy(ctx->d_nccl_srcpos, srcpos, (size_t)N * sizeof(int),
                          cudaMemcpyHostToDevice));
    free(srcpos);
    ctx->d_nccl_dst = ctx->d_gather_perm; /* same destination mapping */
#endif

    CUDA_CHECK(cudaEventCreate(&ctx->ev_start));
    CUDA_CHECK(cudaEventCreate(&ctx->ev_stop));
}

void spmvA_run(SpmvCtxA *ctx, MPI_Comm comm, Transport xport,
               double *t_comm, double *t_comp)
{
    double c0, c1;
    int N = ctx->N;

    if (xport == XPORT_STAGING) {
        c0 = MPI_Wtime();
        MPI_Allgatherv(ctx->h_x_owned, ctx->nx_local, MPI_FLOAT,
                       ctx->h_x_gathered, ctx->recvcounts, ctx->recvdispls, MPI_FLOAT, comm);
        for (int q = 0; q < N; q++)
            ctx->h_x_full[ctx->gather_perm[q]] = ctx->h_x_gathered[q];
        CUDA_CHECK(cudaMemcpy(ctx->d_x_full, ctx->h_x_full, (size_t)N * sizeof(float),
                              cudaMemcpyHostToDevice));
        c1 = MPI_Wtime();
        *t_comm = c1 - c0;
    } else if (xport == XPORT_CUDA_AWARE) {
        c0 = MPI_Wtime();
        MPI_Allgatherv(ctx->d_x_owned, ctx->nx_local, MPI_FLOAT,
                       ctx->d_x_gathered, ctx->recvcounts, ctx->recvdispls, MPI_FLOAT, comm);
        k_scatter_by_perm<<<nblocks(N), TPB>>>(N, ctx->d_gather_perm, ctx->d_x_gathered, ctx->d_x_full);
        CUDA_CHECK(cudaDeviceSynchronize());
        c1 = MPI_Wtime();
        *t_comm = c1 - c0;
    } else { /* XPORT_NCCL */
#ifdef USE_NCCL
        extern ncclComm_t spmv_nccl_comm(void);
        c0 = MPI_Wtime();
        ncclAllGather(ctx->d_x_owned, ctx->d_x_gathered_nccl, ctx->maxnx,
                      ncclFloat, spmv_nccl_comm(), 0);
        CUDA_CHECK(cudaStreamSynchronize(0));
        k_permute<<<nblocks(N), TPB>>>(N, ctx->d_nccl_dst, ctx->d_nccl_srcpos,
                                       ctx->d_x_gathered_nccl, ctx->d_x_full);
        CUDA_CHECK(cudaDeviceSynchronize());
        c1 = MPI_Wtime();
        *t_comm = c1 - c0;
#else
        fprintf(stderr, "NCCL transport requested but built without USE_NCCL\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
    }

    /* local SpMV: col_idx are global indices into the full x of length N */
    CUDA_CHECK(cudaEventRecord(ctx->ev_start));
    spmv_csr_vector_launch(ctx->n_local, ctx->d_row_ptr, ctx->d_col_idx, ctx->d_val,
                           ctx->d_x_full, ctx->d_y, 0);
    CUDA_CHECK(cudaEventRecord(ctx->ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ctx->ev_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ctx->ev_start, ctx->ev_stop));
    *t_comp = ms / 1000.0;
}

void spmvA_copy_y(SpmvCtxA *ctx, float *h_y_local)
{
    CUDA_CHECK(cudaMemcpy(h_y_local, ctx->d_y, (size_t)ctx->n_local * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

size_t spmvA_bytes(const SpmvCtxA *ctx)
{
    size_t b = 0;
    b += (size_t)(ctx->n_local + 1) * sizeof(int);      /* row_ptr */
    b += (size_t)ctx->nnz_local * sizeof(int);          /* col_idx */
    b += (size_t)ctx->nnz_local * sizeof(float);        /* val */
    b += (size_t)ctx->n_local * sizeof(float);          /* y */
    b += (size_t)ctx->N * sizeof(float);                /* x_full */
    b += (size_t)ctx->N * sizeof(float);                /* x_gathered */
    b += (size_t)ctx->maxnx * sizeof(float);            /* x_owned */
    b += (size_t)ctx->N * sizeof(int);                  /* gather_perm */
    return b;
}

void spmvA_free(SpmvCtxA *ctx)
{
    free(ctx->recvcounts); free(ctx->recvdispls); free(ctx->gather_perm);
    free(ctx->h_x_owned); free(ctx->h_x_gathered); free(ctx->h_x_full);
    cudaFree(ctx->d_gather_perm);
    cudaFree(ctx->d_row_ptr); cudaFree(ctx->d_col_idx); cudaFree(ctx->d_val); cudaFree(ctx->d_y);
    cudaFree(ctx->d_x_owned); cudaFree(ctx->d_x_gathered); cudaFree(ctx->d_x_full);
#ifdef USE_NCCL
    cudaFree(ctx->d_x_gathered_nccl); cudaFree(ctx->d_nccl_srcpos);
#endif
    cudaEventDestroy(ctx->ev_start); cudaEventDestroy(ctx->ev_stop);
}

/* ======================================================================== */
/* Mode B: exchange only the referenced ghost entries of x.                 */
/* ======================================================================== */
void spmvB_init(SpmvCtxB *ctx, const LocalCSR *m, int rank, int P, MPI_Comm comm)
{
    memset(ctx, 0, sizeof(*ctx));
    int N = m->ncols;
    ctx->N = N; ctx->P = P; ctx->rank = rank;
    ctx->n_local = m->n_local; ctx->nnz_local = m->nnz_local;
    ctx->n_owned = count_cyclic(N, P, rank);

    /* collect distinct remote (ghost) columns referenced by the local matrix */
    int *tmp = (int *)hmalloc((size_t)m->nnz_local * sizeof(int));
    int cnt = 0;
    for (int e = 0; e < m->nnz_local; e++) {
        int j = m->col_idx[e];
        if (j % P != rank) tmp[cnt++] = j;
    }
    qsort(tmp, cnt, sizeof(int), cmp_int);
    int n_ghost = 0;
    for (int i = 0; i < cnt; i++)
        if (i == 0 || tmp[i] != tmp[i - 1]) tmp[n_ghost++] = tmp[i];
    ctx->n_ghost = n_ghost;
    ctx->x_ext_len = ctx->n_owned + n_ghost;
    int *ghost_global = tmp; /* first n_ghost entries are the sorted unique ghosts */

    /* remap column indices into the compact x_ext buffer:
     *   owned column j  -> j / P               (position within owned block)
     *   ghost column j  -> n_owned + ghost_pos (position within ghost block) */
    int *remap_col = (int *)hmalloc((size_t)m->nnz_local * sizeof(int));
    for (int e = 0; e < m->nnz_local; e++) {
        int j = m->col_idx[e];
        if (j % P == rank) {
            remap_col[e] = j / P;
        } else {
            int g = bsearch_int(ghost_global, n_ghost, j);
            remap_col[e] = ctx->n_owned + g;
        }
    }

    /* how many ghosts we pull from each rank, grouped by owner */
    ctx->recv_counts = (int *)calloc(P, sizeof(int));
    ctx->recv_displs = (int *)hmalloc((size_t)P * sizeof(int));
    for (int g = 0; g < n_ghost; g++)
        ctx->recv_counts[ghost_global[g] % P]++;
    ctx->recv_displs[0] = 0;
    for (int p = 1; p < P; p++)
        ctx->recv_displs[p] = ctx->recv_displs[p - 1] + ctx->recv_counts[p - 1];

    /* requested global indices ordered by partner + landing slot in x_ext */
    int *req_global = (int *)hmalloc((size_t)n_ghost * sizeof(int));
    ctx->ghost_ext_pos = (int *)hmalloc((size_t)n_ghost * sizeof(int));
    int *cursor = (int *)calloc(P, sizeof(int));
    for (int g = 0; g < n_ghost; g++) {
        int gj = ghost_global[g];
        int p = gj % P;
        int slot = ctx->recv_displs[p] + cursor[p]++;
        req_global[slot] = gj;
        ctx->ghost_ext_pos[slot] = ctx->n_owned + g;
    }
    free(cursor);

    /* exchange counts, then the requested indices, so each rank learns which
     * of its owned x components the others need */
    ctx->send_counts = (int *)hmalloc((size_t)P * sizeof(int));
    ctx->send_displs = (int *)hmalloc((size_t)P * sizeof(int));
    MPI_Alltoall(ctx->recv_counts, 1, MPI_INT, ctx->send_counts, 1, MPI_INT, comm);
    ctx->send_displs[0] = 0;
    for (int p = 1; p < P; p++)
        ctx->send_displs[p] = ctx->send_displs[p - 1] + ctx->send_counts[p - 1];
    ctx->total_send = ctx->send_displs[P - 1] + ctx->send_counts[P - 1];

    int *recv_req_global = (int *)hmalloc((size_t)ctx->total_send * sizeof(int));
    MPI_Alltoallv(req_global, ctx->recv_counts, ctx->recv_displs, MPI_INT,
                  recv_req_global, ctx->send_counts, ctx->send_displs, MPI_INT, comm);

    /* map the requested global indices to our owned x positions */
    ctx->send_local_pos = (int *)hmalloc((size_t)ctx->total_send * sizeof(int));
    for (int t = 0; t < ctx->total_send; t++)
        ctx->send_local_pos[t] = recv_req_global[t] / P; /* owner is this rank -> position */
    free(recv_req_global);
    free(req_global);

    /* device index arrays for the aware pack/scatter */
    ctx->d_send_local_pos = (int *)dmalloc((size_t)ctx->total_send * sizeof(int));
    ctx->d_ghost_ext_pos = (int *)dmalloc((size_t)n_ghost * sizeof(int));
    CUDA_CHECK(cudaMemcpy(ctx->d_send_local_pos, ctx->send_local_pos,
                          (size_t)ctx->total_send * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_ghost_ext_pos, ctx->ghost_ext_pos,
                          (size_t)n_ghost * sizeof(int), cudaMemcpyHostToDevice));

    /* device CSR with remapped column indices */
    ctx->d_row_ptr = (int *)dmalloc((size_t)(m->n_local + 1) * sizeof(int));
    ctx->d_col_idx = (int *)dmalloc((size_t)m->nnz_local * sizeof(int));
    ctx->d_val = (float *)dmalloc((size_t)m->nnz_local * sizeof(float));
    ctx->d_y = (float *)dmalloc((size_t)m->n_local * sizeof(float));
    CUDA_CHECK(cudaMemcpy(ctx->d_row_ptr, m->row_ptr, (size_t)(m->n_local + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_col_idx, remap_col, (size_t)m->nnz_local * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_val, m->val, (size_t)m->nnz_local * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(remap_col);

    /* x_ext buffers; the owned block is constant across SpMV iterations */
    ctx->h_x_ext = (float *)hmalloc((size_t)ctx->x_ext_len * sizeof(float));
    for (int mm = 0; mm < ctx->n_owned; mm++)
        ctx->h_x_ext[mm] = xvec_value(rank + mm * P);
    ctx->h_send_buf = (float *)hmalloc((size_t)ctx->total_send * sizeof(float));
    ctx->h_recv_buf = (float *)hmalloc((size_t)n_ghost * sizeof(float));

    ctx->d_x_ext = (float *)dmalloc((size_t)ctx->x_ext_len * sizeof(float));
    ctx->d_send_buf = (float *)dmalloc((size_t)ctx->total_send * sizeof(float));
    ctx->d_recv_buf = (float *)dmalloc((size_t)n_ghost * sizeof(float));
    CUDA_CHECK(cudaMemcpy(ctx->d_x_ext, ctx->h_x_ext, (size_t)ctx->n_owned * sizeof(float),
                          cudaMemcpyHostToDevice));

    free(tmp);

    CUDA_CHECK(cudaEventCreate(&ctx->ev_start));
    CUDA_CHECK(cudaEventCreate(&ctx->ev_stop));
}

void spmvB_run(SpmvCtxB *ctx, MPI_Comm comm, Transport xport,
               double *t_comm, double *t_comp)
{
    double c0, c1;

    if (xport == XPORT_STAGING) {
        c0 = MPI_Wtime();
        for (int t = 0; t < ctx->total_send; t++)
            ctx->h_send_buf[t] = ctx->h_x_ext[ctx->send_local_pos[t]];
        MPI_Alltoallv(ctx->h_send_buf, ctx->send_counts, ctx->send_displs, MPI_FLOAT,
                      ctx->h_recv_buf, ctx->recv_counts, ctx->recv_displs, MPI_FLOAT, comm);
        for (int k = 0; k < ctx->n_ghost; k++)
            ctx->h_x_ext[ctx->ghost_ext_pos[k]] = ctx->h_recv_buf[k];
        CUDA_CHECK(cudaMemcpy(ctx->d_x_ext + ctx->n_owned, ctx->h_x_ext + ctx->n_owned,
                              (size_t)ctx->n_ghost * sizeof(float), cudaMemcpyHostToDevice));
        c1 = MPI_Wtime();
        *t_comm = c1 - c0;
    } else if (xport == XPORT_CUDA_AWARE) {
        c0 = MPI_Wtime();
        k_gather_by_index<<<nblocks(ctx->total_send), TPB>>>(
            ctx->total_send, ctx->d_send_local_pos, ctx->d_x_ext, ctx->d_send_buf);
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Alltoallv(ctx->d_send_buf, ctx->send_counts, ctx->send_displs, MPI_FLOAT,
                      ctx->d_recv_buf, ctx->recv_counts, ctx->recv_displs, MPI_FLOAT, comm);
        k_scatter_by_perm<<<nblocks(ctx->n_ghost), TPB>>>(
            ctx->n_ghost, ctx->d_ghost_ext_pos, ctx->d_recv_buf, ctx->d_x_ext);
        CUDA_CHECK(cudaDeviceSynchronize());
        c1 = MPI_Wtime();
        *t_comm = c1 - c0;
    } else {
        fprintf(stderr, "Ghost mode supports staging and CUDA-aware transports only\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    CUDA_CHECK(cudaEventRecord(ctx->ev_start));
    spmv_csr_vector_launch(ctx->n_local, ctx->d_row_ptr, ctx->d_col_idx, ctx->d_val,
                           ctx->d_x_ext, ctx->d_y, 0);
    CUDA_CHECK(cudaEventRecord(ctx->ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ctx->ev_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ctx->ev_start, ctx->ev_stop));
    *t_comp = ms / 1000.0;
}

void spmvB_copy_y(SpmvCtxB *ctx, float *h_y_local)
{
    CUDA_CHECK(cudaMemcpy(h_y_local, ctx->d_y, (size_t)ctx->n_local * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

size_t spmvB_bytes(const SpmvCtxB *ctx)
{
    size_t b = 0;
    b += (size_t)(ctx->n_local + 1) * sizeof(int);
    b += (size_t)ctx->nnz_local * sizeof(int);
    b += (size_t)ctx->nnz_local * sizeof(float);
    b += (size_t)ctx->n_local * sizeof(float);
    b += (size_t)ctx->x_ext_len * sizeof(float);        /* x_ext */
    b += (size_t)ctx->total_send * sizeof(float);       /* send_buf */
    b += (size_t)ctx->n_ghost * sizeof(float);          /* recv_buf */
    b += (size_t)ctx->total_send * sizeof(int);         /* send positions */
    b += (size_t)ctx->n_ghost * sizeof(int);            /* ghost positions */
    return b;
}

void spmvB_free(SpmvCtxB *ctx)
{
    free(ctx->recv_counts); free(ctx->recv_displs);
    free(ctx->send_counts); free(ctx->send_displs);
    free(ctx->send_local_pos); free(ctx->ghost_ext_pos);
    free(ctx->h_x_ext); free(ctx->h_send_buf); free(ctx->h_recv_buf);
    cudaFree(ctx->d_send_local_pos); cudaFree(ctx->d_ghost_ext_pos);
    cudaFree(ctx->d_row_ptr); cudaFree(ctx->d_col_idx); cudaFree(ctx->d_val); cudaFree(ctx->d_y);
    cudaFree(ctx->d_x_ext); cudaFree(ctx->d_send_buf); cudaFree(ctx->d_recv_buf);
    cudaEventDestroy(ctx->ev_start); cudaEventDestroy(ctx->ev_stop);
}

/* ======================================================================== */
void gather_y_to_root(const float *h_y_local, int n_local, int rank, int P, int M,
                      MPI_Comm comm, float *h_y_full)
{
    int *recvcounts = NULL, *displs = NULL;
    float *gathered = NULL;
    if (rank == 0) {
        recvcounts = (int *)malloc((size_t)P * sizeof(int));
        displs = (int *)malloc((size_t)P * sizeof(int));
        for (int p = 0; p < P; p++) recvcounts[p] = count_cyclic(M, P, p);
        displs[0] = 0;
        for (int p = 1; p < P; p++) displs[p] = displs[p - 1] + recvcounts[p - 1];
        gathered = (float *)malloc((size_t)M * sizeof(float));
    }

    MPI_Gatherv(h_y_local, n_local, MPI_FLOAT, gathered, recvcounts, displs, MPI_FLOAT, 0, comm);

    if (rank == 0) {
        for (int p = 0; p < P; p++)
            for (int mm = 0; mm < recvcounts[p]; mm++)
                h_y_full[p + mm * P] = gathered[displs[p] + mm];
        free(recvcounts); free(displs); free(gathered);
    }
}

/* ======================================================================== */
/* NCCL bonus lifecycle                                                     */
/* ======================================================================== */
#ifdef USE_NCCL
static ncclComm_t g_nccl_comm;
static int g_nccl_ready = 0;
ncclComm_t spmv_nccl_comm(void) { return g_nccl_comm; }

int spmv_nccl_available(void) { return 1; }

void spmv_nccl_init(int rank, int P, MPI_Comm comm)
{
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm);
    ncclCommInitRank(&g_nccl_comm, P, id, rank);
    g_nccl_ready = 1;
}

void spmv_nccl_finalize(void)
{
    if (g_nccl_ready) { ncclCommDestroy(g_nccl_comm); g_nccl_ready = 0; }
}
#else
int spmv_nccl_available(void) { return 0; }
void spmv_nccl_init(int rank, int P, MPI_Comm comm) { (void)rank; (void)P; (void)comm; }
void spmv_nccl_finalize(void) {}
#endif
