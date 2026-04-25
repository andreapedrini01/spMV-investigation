#ifndef SPMV_CPU_H
#define SPMV_CPU_H

/*
 * CPU reference implementations for SpMV.
 * Used for validation of GPU results.
 */

/* COO SpMV: y[row[i]] += val[i] * x[col[i]] for each nonzero */
void spmv_coo_cpu(const int *row_indices, const int *col_indices, const float *values,
                  int nnz, const float *x, float *y);

/* CSR SpMV: standard row-by-row dot product */
void spmv_csr_cpu(const int *row_ptr, const int *col_idx, const float *values,
                  int nrows, const float *x, float *y);

#endif /* SPMV_CPU_H */
