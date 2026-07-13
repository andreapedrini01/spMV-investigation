#ifndef DISTRIBUTED_H
#define DISTRIBUTED_H

#include <mpi.h>
#include <cuda_runtime.h>

/*
 * 1D cyclic (modulo) distribution and distributed SpMV.
 *
 * Row i is owned by rank owner(i) = i % P. The k-th local row of rank r has
 * global index i = r + k*P, so the inverse map is k = i / P (valid because
 * i % P == r and r < P). Column indices of x follow the same cyclic rule.
 *
 * Two communication modes are provided:
 *   Mode A  - Allgather the whole x on every rank (simple, memory O(N)).
 *   Mode B  - exchange only the ghost entries actually referenced (memory
 *             O(n_local + n_ghost), lower communication volume).
 * Each mode runs with host staging or with CUDA-aware MPI (device pointers).
 */

/* Number of items rank `rank` owns out of `total` under the cyclic rule. */
static inline int count_cyclic(int total, int P, int rank)
{
    return total / P + (rank < (total % P) ? 1 : 0);
}

/* Host-side CSR of the rows a rank owns. Column indices are GLOBAL. */
typedef struct {
    int n_local;       /* rows owned by this rank */
    int ncols;         /* = N (global columns) */
    int nnz_local;     /* nonzeros owned */
    int nrows_global;  /* = M */
    int *row_ptr;      /* [n_local + 1] */
    int *col_idx;      /* [nnz_local], global column indices */
    float *val;        /* [nnz_local] */
} LocalCSR;

/* Communication mode selector. */
typedef enum { MODE_ALLGATHER = 0, MODE_GHOST = 1 } CommMode;

/* Transport selector for a single SpMV. */
typedef enum { XPORT_STAGING = 0, XPORT_CUDA_AWARE = 1, XPORT_NCCL = 2 } Transport;

/* Mode A context: full x gathered on every rank. */
typedef struct {
    int N, P, rank, nx_local;
    int *recvcounts, *recvdispls;   /* [P] Allgatherv layout (rank-major) */
    int *gather_perm;               /* [N] host: gathered position -> global index */
    int *d_gather_perm;             /* [N] device copy for the aware reorder kernel */
    /* device CSR (global column indices) */
    int *d_row_ptr, *d_col_idx;
    float *d_val, *d_y;
    int n_local, nnz_local;
    /* x buffers */
    float *h_x_owned, *h_x_gathered, *h_x_full;
    float *d_x_owned, *d_x_gathered, *d_x_full;
    /* NCCL bonus: ncclAllGather needs equal per-rank counts, so we pad to
     * maxnx = ceil(N/P) and reorder from the strided padded layout. */
    int maxnx;
    float *d_x_gathered_nccl;
    int *d_nccl_srcpos, *d_nccl_dst; /* [N] strided gathered pos -> global index */
    cudaEvent_t ev_start, ev_stop;
} SpmvCtxA;

/* Mode B context: ghost-only exchange with local/global index remap. */
typedef struct {
    int N, P, rank;
    int n_owned;        /* x components owned = nx_local */
    int n_ghost;        /* distinct remote x components referenced */
    int x_ext_len;      /* n_owned + n_ghost */
    int total_send;     /* owned values other ranks request from us */
    int *recv_counts, *recv_displs; /* [P] ghosts we pull from each rank */
    int *send_counts, *send_displs; /* [P] owned values each rank pulls from us */
    int *send_local_pos;            /* [total_send] owned positions to pack */
    int *ghost_ext_pos;             /* [n_ghost] x_ext slot per received value */
    int *d_send_local_pos;          /* device copy for the aware pack kernel */
    int *d_ghost_ext_pos;           /* device copy for the aware scatter kernel */
    /* device CSR (column indices remapped into x_ext) */
    int *d_row_ptr, *d_col_idx;
    float *d_val, *d_y;
    int n_local, nnz_local;
    /* x buffers */
    float *h_x_ext, *h_send_buf, *h_recv_buf;
    float *d_x_ext, *d_send_buf, *d_recv_buf;
    cudaEvent_t ev_start, ev_stop;
} SpmvCtxB;

#ifdef __cplusplus
extern "C" {
#endif

/* Read (on rank 0) and cyclically distribute the matrix. Broadcasts M and N.
 * On entry rank 0 holds the full COO (0-indexed); other ranks pass NULL/0.
 * On exit every rank holds its LocalCSR (global column indices). */
void distribute_matrix_cyclic(int rank, int P, MPI_Comm comm,
                              int M, int N, int nnz_global,
                              const int *coo_row, const int *coo_col, const float *coo_val,
                              LocalCSR *out);

void free_local_csr(LocalCSR *m);

/* NCCL bonus lifecycle. These are always declared; when the code is built
 * without USE_NCCL, spmv_nccl_available() returns 0 and init/finalize are
 * no-ops. This keeps NCCL types out of the header. */
int  spmv_nccl_available(void);
void spmv_nccl_init(int rank, int P, MPI_Comm comm);
void spmv_nccl_finalize(void);

/* Mode A lifecycle. spmvA_run performs one distributed SpMV and returns this
 * rank's communication and compute time (seconds) through the out params. */
void spmvA_init(SpmvCtxA *ctx, const LocalCSR *m, int rank, int P, MPI_Comm comm);
void spmvA_run(SpmvCtxA *ctx, MPI_Comm comm, Transport xport,
               double *t_comm, double *t_comp);
void spmvA_copy_y(SpmvCtxA *ctx, float *h_y_local);
size_t spmvA_bytes(const SpmvCtxA *ctx); /* per-rank device footprint */
void spmvA_free(SpmvCtxA *ctx);

/* Mode B lifecycle. */
void spmvB_init(SpmvCtxB *ctx, const LocalCSR *m, int rank, int P, MPI_Comm comm);
void spmvB_run(SpmvCtxB *ctx, MPI_Comm comm, Transport xport,
               double *t_comm, double *t_comp);
void spmvB_copy_y(SpmvCtxB *ctx, float *h_y_local);
size_t spmvB_bytes(const SpmvCtxB *ctx);
void spmvB_free(SpmvCtxB *ctx);

/* Reassemble the distributed y on rank 0 in natural global order (validation
 * and optional output only; not on the timed path). */
void gather_y_to_root(const float *h_y_local, int n_local, int rank, int P, int M,
                      MPI_Comm comm, float *h_y_full);

#ifdef __cplusplus
}
#endif

#endif /* DISTRIBUTED_H */
