#ifndef SPMV_CPU_REF_H
#define SPMV_CPU_REF_H

#include "distributed.h"

/*
 * CPU references for validation.
 *
 * spmv_local_reference computes y for the rows a rank owns using the local CSR
 * (global column indices) and the deterministic x, so every rank checks its own
 * slice without communicating. This exercises the communication, the local
 * kernel and the ghost remap.
 *
 * spmv_csr_full_reference is the plain serial SpMV over a full CSR; rank 0 uses
 * it to check the distribution end-to-end.
 */
void spmv_local_reference(const LocalCSR *m, float *y_local);

void spmv_csr_full_reference(const int *row_ptr, const int *col_idx, const float *val,
                             int nrows, const float *x, float *y);

#endif /* SPMV_CPU_REF_H */
