#include "spmv_cpu.h"

void spmv_coo_cpu(const int *row_indices, const int *col_indices, const float *values,
                  int nnz, const float *x, float *y)
{
    for (int i = 0; i < nnz; i++) {
        y[row_indices[i]] += values[i] * x[col_indices[i]];
    }
}

void spmv_csr_cpu(const int *row_ptr, const int *col_idx, const float *values,
                  int nrows, const float *x, float *y)
{
    for (int i = 0; i < nrows; i++) {
        float sum = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            sum += values[j] * x[col_idx[j]];
        }
        y[i] = sum;
    }
}
