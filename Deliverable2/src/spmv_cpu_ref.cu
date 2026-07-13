#include "spmv_cpu_ref.h"
#include "utils.h"

void spmv_local_reference(const LocalCSR *m, float *y_local)
{
    for (int k = 0; k < m->n_local; k++) {
        float sum = 0.0f;
        for (int e = m->row_ptr[k]; e < m->row_ptr[k + 1]; e++)
            sum += m->val[e] * xvec_value(m->col_idx[e]); /* global column */
        y_local[k] = sum;
    }
}

void spmv_csr_full_reference(const int *row_ptr, const int *col_idx, const float *val,
                             int nrows, const float *x, float *y)
{
    for (int i = 0; i < nrows; i++) {
        float sum = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
            sum += val[j] * x[col_idx[j]];
        y[i] = sum;
    }
}
