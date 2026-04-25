#ifndef SPMV_GPU_H
#define SPMV_GPU_H

/*
 * GPU kernel wrappers for SpMV.
 * Each function handles launch configuration, timing, and returns the average kernel time in ms.
 */

#define WARMUP_ITERS 4
#define MEASURE_ITERS 10

/* CSR-Scalar: one thread per row (straightforward parallelization) */
float run_spmv_csr_scalar(int nrows, int *d_row_ptr, int *d_col_idx, float *d_val,
                          float *d_x, float *d_y, float *timings);

/* CSR-Vector: one warp per row with warp shuffle reduction */
float run_spmv_csr_vector(int nrows, int *d_row_ptr, int *d_col_idx, float *d_val,
                          float *d_x, float *d_y, float *timings);

/* CSR-Vector with explicit shared memory reduction */
float run_spmv_csr_vector_shmem(int nrows, int *d_row_ptr, int *d_col_idx, float *d_val,
                                float *d_x, float *d_y, float *timings);

/* cuSPARSE wrapper */
float run_spmv_cusparse(int nrows, int ncols, int nnz,
                        int *d_row_ptr, int *d_col_idx, float *d_val,
                        float *d_x, float *d_y, float *timings);

#endif /* SPMV_GPU_H */
