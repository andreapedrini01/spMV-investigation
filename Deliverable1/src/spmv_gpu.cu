#include <stdio.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include "spmv_gpu.h"

#define WARP_SIZE 32
#define BLOCK_SIZE 256

/* ========================================================================
 * Kernel 1: CSR-Scalar — one thread per row
 * Straightforward parallelization. Each thread computes the dot product
 * for a single row. Simple but suffers from load imbalance when row
 * lengths vary, and adjacent threads access non-contiguous memory.
 * ======================================================================== */
__global__
void kernel_spmv_csr_scalar(int nrows, const int *row_ptr, const int *col_idx,
                            const float *val, const float *x, float *y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        float sum = 0.0f;
        int rs = row_ptr[row];
        int re = row_ptr[row + 1];
        for (int j = rs; j < re; j++) {
            sum += val[j] * x[col_idx[j]];
        }
        y[row] = sum;
    }
}

/* ========================================================================
 * Kernel 2: CSR-Vector — one warp per row, warp shuffle reduction
 * 32 threads collaborate on each row. Adjacent threads in a warp access
 * contiguous elements in val[] and col_idx[], giving better coalescing.
 * The partial sums are reduced with __shfl_down_sync, avoiding shared
 * memory entirely for the reduction step.
 * ======================================================================== */
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

        for (int j = rs + lane; j < re; j += WARP_SIZE) {
            sum += val[j] * x[col_idx[j]];
        }

        /* warp-level reduction */
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        if (lane == 0) {
            y[row] = sum;
        }
    }
}

/* ========================================================================
 * Kernel 3: CSR-Vector with shared memory reduction
 * Same warp-per-row strategy, but the reduction is done through shared
 * memory instead of warp shuffles. This variant is included to explicitly
 * demonstrate shared memory usage as required by the deliverable.
 * ======================================================================== */
__global__
void kernel_spmv_csr_vector_shmem(int nrows, const int *row_ptr, const int *col_idx,
                                  const float *val, const float *x, float *y)
{
    extern __shared__ float sdata[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int row = warp_id;

    if (row < nrows) {
        float sum = 0.0f;
        int rs = row_ptr[row];
        int re = row_ptr[row + 1];

        for (int j = rs + lane; j < re; j += WARP_SIZE) {
            sum += val[j] * x[col_idx[j]];
        }

        /* store partial sum in shared memory */
        sdata[threadIdx.x] = sum;
        __syncwarp();

        /* tree reduction within the warp, all in shared memory */
        if (lane < 16) sdata[threadIdx.x] += sdata[threadIdx.x + 16];
        __syncwarp();
        if (lane < 8)  sdata[threadIdx.x] += sdata[threadIdx.x + 8];
        __syncwarp();
        if (lane < 4)  sdata[threadIdx.x] += sdata[threadIdx.x + 4];
        __syncwarp();
        if (lane < 2)  sdata[threadIdx.x] += sdata[threadIdx.x + 2];
        __syncwarp();
        if (lane < 1)  sdata[threadIdx.x] += sdata[threadIdx.x + 1];
        __syncwarp();

        if (lane == 0) {
            y[row] = sdata[threadIdx.x];
        }
    }
}

/* ---- CSR-Scalar launcher ---- */
float run_spmv_csr_scalar(int nrows, int *d_row_ptr, int *d_col_idx, float *d_val,
                          float *d_x, float *d_y, float *timings)
{
    int threads = BLOCK_SIZE;
    int blocks = (nrows + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* warmup */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        kernel_spmv_csr_scalar<<<blocks, threads>>>(nrows, d_row_ptr, d_col_idx, d_val, d_x, d_y);
        cudaDeviceSynchronize();
    }

    /* timed runs */
    float total = 0.0f;
    for (int i = 0; i < MEASURE_ITERS; i++) {
        cudaMemset(d_y, 0, nrows * sizeof(float));
        cudaEventRecord(start);
        kernel_spmv_csr_scalar<<<blocks, threads>>>(nrows, d_row_ptr, d_col_idx, d_val, d_x, d_y);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        timings[i] = ms;
        total += ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total / MEASURE_ITERS;
}

/* ---- CSR-Vector (warp shuffle) launcher ---- */
float run_spmv_csr_vector(int nrows, int *d_row_ptr, int *d_col_idx, float *d_val,
                          float *d_x, float *d_y, float *timings)
{
    /* each warp handles one row, so we need nrows warps total */
    int total_threads = nrows * WARP_SIZE;
    int threads = BLOCK_SIZE;
    int blocks = (total_threads + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP_ITERS; i++) {
        kernel_spmv_csr_vector<<<blocks, threads>>>(nrows, d_row_ptr, d_col_idx, d_val, d_x, d_y);
        cudaDeviceSynchronize();
    }

    float total = 0.0f;
    for (int i = 0; i < MEASURE_ITERS; i++) {
        cudaMemset(d_y, 0, nrows * sizeof(float));
        cudaEventRecord(start);
        kernel_spmv_csr_vector<<<blocks, threads>>>(nrows, d_row_ptr, d_col_idx, d_val, d_x, d_y);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        timings[i] = ms;
        total += ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total / MEASURE_ITERS;
}

/* ---- CSR-Vector shared memory launcher ---- */
float run_spmv_csr_vector_shmem(int nrows, int *d_row_ptr, int *d_col_idx, float *d_val,
                                float *d_x, float *d_y, float *timings)
{
    int total_threads = nrows * WARP_SIZE;
    int threads = BLOCK_SIZE;
    int blocks = (total_threads + threads - 1) / threads;
    size_t shmem_bytes = threads * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP_ITERS; i++) {
        kernel_spmv_csr_vector_shmem<<<blocks, threads, shmem_bytes>>>(
            nrows, d_row_ptr, d_col_idx, d_val, d_x, d_y);
        cudaDeviceSynchronize();
    }

    float total = 0.0f;
    for (int i = 0; i < MEASURE_ITERS; i++) {
        cudaMemset(d_y, 0, nrows * sizeof(float));
        cudaEventRecord(start);
        kernel_spmv_csr_vector_shmem<<<blocks, threads, shmem_bytes>>>(
            nrows, d_row_ptr, d_col_idx, d_val, d_x, d_y);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        timings[i] = ms;
        total += ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total / MEASURE_ITERS;
}

/* ---- cuSPARSE launcher ---- */
float run_spmv_cusparse(int nrows, int ncols, int nnz,
                        int *d_row_ptr, int *d_col_idx, float *d_val,
                        float *d_x, float *d_y, float *timings)
{
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    cusparseCreateCsr(&matA, nrows, ncols, nnz,
                      d_row_ptr, d_col_idx, d_val,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseCreateDnVec(&vecX, ncols, d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, nrows, d_y, CUDA_R_32F);

    float alpha = 1.0f, beta = 0.0f;
    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY,
                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

    void *buffer = NULL;
    cudaMalloc(&buffer, bufferSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* warmup */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, vecX, &beta, vecY,
                     CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);
        cudaDeviceSynchronize();
    }

    /* timed runs */
    float total = 0.0f;
    for (int i = 0; i < MEASURE_ITERS; i++) {
        cudaMemset(d_y, 0, nrows * sizeof(float));

        /* need to re-create vecY descriptor since we zeroed the buffer */
        cusparseDestroyDnVec(vecY);
        cusparseCreateDnVec(&vecY, nrows, d_y, CUDA_R_32F);

        cudaEventRecord(start);
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, vecX, &beta, vecY,
                     CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        timings[i] = ms;
        total += ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(buffer);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);

    return total / MEASURE_ITERS;
}
