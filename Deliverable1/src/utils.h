#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>

/*
 * Utility functions: device info, statistics, validation.
 */

/* Print GPU device properties */
void print_device_properties(void);

/* Arithmetic mean of an array */
float arithmetic_mean(const float *v, int len);

/* Sample standard deviation */
float compute_std(const float *v, int len, float mean);

/* Validate GPU result against CPU reference.
 * Returns the total absolute error (sum of |ref[i] - test[i]|). */
float validate_results(const float *ref, const float *test, int len);

/* Read a Matrix Market file and return COO arrays (0-indexed).
 * Handles symmetric matrices by expanding (i,j) -> (j,i).
 * Handles pattern matrices by assigning value 1.0.
 * Returns 0 on success, nonzero on error.
 * *nnz_out may be larger than the file's nnz for symmetric matrices. */
int read_mtx_file(const char *filename,
                  int *nrows_out, int *ncols_out, int *nnz_out,
                  int **row_indices, int **col_indices, float **values);

/* Sort COO arrays by (row, col) and convert to CSR in-place.
 * Allocates row_ptr internally. */
void coo_to_csr(const int *row_indices, const int *col_indices, const float *coo_values,
                int nrows, int nnz,
                int **row_ptr_out, int **col_idx_out, float **csr_values_out);

#endif /* UTILS_H */
