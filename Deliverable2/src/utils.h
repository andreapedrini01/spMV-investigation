#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>

/*
 * Shared helpers for the distributed SpMV: error checking, the Matrix Market
 * reader (reused from Deliverable 1), COO->CSR conversion, statistics, and a
 * deterministic vector generator so every rank can build its slice of x
 * without any communication.
 */

/* Abort the whole job on a CUDA error: a single failing rank must not leave
 * the others waiting in a collective. */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                 \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                           \
        }                                                                      \
    } while (0)

/* __host__ __device__ qualifier that also compiles under a plain host compiler
 * (used by the standalone logic tests). */
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

/* Deterministic entries of the global input vector x. Both the owning rank and
 * the serial reference derive x[j] from j alone (Knuth multiplicative hash),
 * so the distributed x needs no broadcast and validation stays exact. */
HD static inline float xvec_value(int j)
{
    unsigned int h = (unsigned int)j * 2654435761u;
    return (float)(h & 0x00FFFFFFu) / (float)0x01000000u; /* in [0, 1) */
}

/* Print properties of every visible CUDA device (called once, on rank 0). */
void print_device_properties(void);

float arithmetic_mean(const float *v, int len);
float compute_std(const float *v, int len, float mean);

/* Sum of absolute differences between a reference and a test vector. */
float validate_results(const float *ref, const float *test, int len);

/* Read a Matrix Market file into COO arrays (0-indexed). Symmetric matrices are
 * expanded (i,j)->(j,i); pattern matrices get value 1.0. Returns 0 on success.
 * *nnz_out may exceed the file's nnz because of symmetric expansion. */
int read_mtx_file(const char *filename,
                  int *nrows_out, int *ncols_out, int *nnz_out,
                  int **row_indices, int **col_indices, float **values);

/* Sort COO by (row, col) and build CSR. Allocates all three output arrays. */
void coo_to_csr(const int *row_indices, const int *col_indices, const float *coo_values,
                int nrows, int nnz,
                int **row_ptr_out, int **col_idx_out, float **csr_values_out);

#endif /* UTILS_H */
