#ifndef SPMV_LOCAL_H
#define SPMV_LOCAL_H

#include <cuda_runtime.h>

/*
 * Node-local SpMV kernels, reused from Deliverable 1. Each rank runs one of
 * these on the CSR slice of the rows it owns. The kernels are exposed as
 * single launches (no internal timing) so the caller can wrap the launch with
 * cudaEvents and separate compute time from communication time.
 *
 * The x buffer and col_idx must be consistent: in the Allgather mode col_idx
 * holds global indices into a full x of length N; in the ghost mode col_idx is
 * remapped into the compact x_ext buffer.
 */

/* CSR-Scalar: one thread per row. */
void spmv_csr_scalar_launch(int nrows, const int *d_row_ptr, const int *d_col_idx,
                            const float *d_val, const float *d_x, float *d_y,
                            cudaStream_t stream);

/* CSR-Vector: one warp per row with a warp-shuffle reduction. This is the
 * default local kernel; it coalesces the val/col_idx reads within a warp. */
void spmv_csr_vector_launch(int nrows, const int *d_row_ptr, const int *d_col_idx,
                            const float *d_val, const float *d_x, float *d_y,
                            cudaStream_t stream);

#endif /* SPMV_LOCAL_H */
