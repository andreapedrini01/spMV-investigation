#include <cuda_runtime.h>
#include "spmv_local.h"

#define WARP_SIZE 32
#define BLOCK_SIZE 256

/* CSR-Scalar: one thread per row. Simple, but neighbouring threads read
 * non-contiguous val[] and load imbalance grows with row-length variance. */
__global__
void kernel_spmv_csr_scalar(int nrows, const int *row_ptr, const int *col_idx,
                            const float *val, const float *x, float *y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        float sum = 0.0f;
        int rs = row_ptr[row];
        int re = row_ptr[row + 1];
        for (int j = rs; j < re; j++)
            sum += val[j] * x[col_idx[j]];
        y[row] = sum;
    }
}

/* CSR-Vector: one warp per row. Lanes stride over the row so their val[] and
 * col_idx[] reads coalesce, then a shuffle reduction combines the partials. */
__global__
void kernel_spmv_csr_vector(int nrows, const int *row_ptr, const int *col_idx,
                            const float *val, const float *x, float *y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int row = warp_id;

    if (row < nrows) {
        float sum = 0.0f;
        int rs = row_ptr[row];
        int re = row_ptr[row + 1];

        for (int j = rs + lane; j < re; j += WARP_SIZE)
            sum += val[j] * x[col_idx[j]];

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

        if (lane == 0)
            y[row] = sum;
    }
}

void spmv_csr_scalar_launch(int nrows, const int *d_row_ptr, const int *d_col_idx,
                            const float *d_val, const float *d_x, float *d_y,
                            cudaStream_t stream)
{
    if (nrows <= 0) return;
    int threads = BLOCK_SIZE;
    int blocks = (nrows + threads - 1) / threads;
    kernel_spmv_csr_scalar<<<blocks, threads, 0, stream>>>(
        nrows, d_row_ptr, d_col_idx, d_val, d_x, d_y);
}

void spmv_csr_vector_launch(int nrows, const int *d_row_ptr, const int *d_col_idx,
                            const float *d_val, const float *d_x, float *d_y,
                            cudaStream_t stream)
{
    if (nrows <= 0) return;
    int total_threads = nrows * WARP_SIZE; /* one warp per row */
    int threads = BLOCK_SIZE;
    int blocks = (total_threads + threads - 1) / threads;
    kernel_spmv_csr_vector<<<blocks, threads, 0, stream>>>(
        nrows, d_row_ptr, d_col_idx, d_val, d_x, d_y);
}
