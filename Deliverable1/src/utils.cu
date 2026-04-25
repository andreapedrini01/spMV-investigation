#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "utils.h"
#include "mmio.h"

/* ======================================================================== */
void print_device_properties(void)
{
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    printf("CUDA devices found: %d\n", dev_count);

    for (int i = 0; i < dev_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("\n--- Device %d: %s ---\n", i, prop.name);
        printf("  Compute capability:    %d.%d\n", prop.major, prop.minor);
        printf("  SMs:                   %d\n", prop.multiProcessorCount);
        printf("  Max threads/block:     %d\n", prop.maxThreadsPerBlock);
        printf("  Max threads/SM:        %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Shared mem/block:      %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Memory clock:          %.0f MHz\n", prop.memoryClockRate * 1e-3f);
        printf("  Memory bus width:      %d bit\n", prop.memoryBusWidth);
        printf("  Peak memory bandwidth: %.1f GB/s\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  Global memory:         %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
    printf("\n");
}

/* ======================================================================== */
float arithmetic_mean(const float *v, int len)
{
    float sum = 0.0f;
    for (int i = 0; i < len; i++)
        sum += v[i];
    return sum / (float)len;
}

float compute_std(const float *v, int len, float mean)
{
    float var = 0.0f;
    for (int i = 0; i < len; i++) {
        float d = v[i] - mean;
        var += d * d;
    }
    var /= (float)(len - 1);
    return sqrtf(var);
}

/* ======================================================================== */
int validate_results(const float *ref, const float *test, int len, float tol)
{
    int mismatches = 0;
    for (int i = 0; i < len; i++) {
        if (fabsf(ref[i] - test[i]) > tol) {
            if (mismatches < 5) {
                printf("  MISMATCH at index %d: ref=%.6f  test=%.6f  diff=%.6e\n",
                       i, ref[i], test[i], fabsf(ref[i] - test[i]));
            }
            mismatches++;
        }
    }
    return mismatches;
}

/* ======================================================================== */
int read_mtx_file(const char *filename,
                  int *nrows_out, int *ncols_out, int *nnz_out,
                  int **row_indices, int **col_indices, float **values)
{
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: cannot open file %s\n", filename);
        return -1;
    }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) {
        fprintf(stderr, "Error: could not read Matrix Market banner in %s\n", filename);
        fclose(f);
        return -1;
    }

    if (!mm_is_matrix(matcode) || !mm_is_coordinate(matcode)) {
        fprintf(stderr, "Error: only sparse (coordinate) matrices are supported\n");
        fclose(f);
        return -1;
    }

    int M, N, nz;
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
        fprintf(stderr, "Error: could not read matrix size\n");
        fclose(f);
        return -1;
    }

    int is_symmetric = mm_is_symmetric(matcode);
    int is_pattern = mm_is_pattern(matcode);

    /* allocate enough space: symmetric matrices can have up to 2*nz entries */
    int max_nnz = is_symmetric ? 2 * nz : nz;
    int *rows = (int *)malloc(max_nnz * sizeof(int));
    int *cols = (int *)malloc(max_nnz * sizeof(int));
    float *vals = (float *)malloc(max_nnz * sizeof(float));

    if (!rows || !cols || !vals) {
        fprintf(stderr, "Error: memory allocation failed for %d entries\n", max_nnz);
        fclose(f);
        return -1;
    }

    int count = 0;
    for (int i = 0; i < nz; i++) {
        int r, c;
        double v = 1.0;

        if (is_pattern) {
            if (fscanf(f, "%d %d", &r, &c) != 2) {
                fprintf(stderr, "Error: failed to read entry %d\n", i);
                fclose(f); free(rows); free(cols); free(vals);
                return -1;
            }
            v = 1.0;
        } else {
            if (fscanf(f, "%d %d %lg", &r, &c, &v) != 3) {
                fprintf(stderr, "Error: failed to read entry %d\n", i);
                fclose(f); free(rows); free(cols); free(vals);
                return -1;
            }
        }

        /* convert from 1-based to 0-based indexing */
        r--; c--;

        rows[count] = r;
        cols[count] = c;
        vals[count] = (float)v;
        count++;

        /* for symmetric matrices, add the transposed entry (skip diagonal) */
        if (is_symmetric && r != c) {
            rows[count] = c;
            cols[count] = r;
            vals[count] = (float)v;
            count++;
        }
    }

    fclose(f);

    *nrows_out = M;
    *ncols_out = N;
    *nnz_out = count;
    *row_indices = rows;
    *col_indices = cols;
    *values = vals;

    return 0;
}

/* ======================================================================== */

/* comparison function for sorting COO entries by (row, col) */
typedef struct {
    int row;
    int col;
    float val;
} COOEntry;

static int coo_compare(const void *a, const void *b)
{
    const COOEntry *ea = (const COOEntry *)a;
    const COOEntry *eb = (const COOEntry *)b;
    if (ea->row != eb->row) return ea->row - eb->row;
    return ea->col - eb->col;
}

void coo_to_csr(const int *row_indices, const int *col_indices, const float *coo_values,
                int nrows, int nnz,
                int **row_ptr_out, int **col_idx_out, float **csr_values_out)
{
    /* pack into sortable struct */
    COOEntry *entries = (COOEntry *)malloc(nnz * sizeof(COOEntry));
    for (int i = 0; i < nnz; i++) {
        entries[i].row = row_indices[i];
        entries[i].col = col_indices[i];
        entries[i].val = coo_values[i];
    }

    qsort(entries, nnz, sizeof(COOEntry), coo_compare);

    /* build CSR arrays */
    int *rp = (int *)calloc(nrows + 1, sizeof(int));
    int *ci = (int *)malloc(nnz * sizeof(int));
    float *cv = (float *)malloc(nnz * sizeof(float));

    for (int i = 0; i < nnz; i++) {
        rp[entries[i].row + 1]++;
    }
    for (int i = 0; i < nrows; i++) {
        rp[i + 1] += rp[i];
    }

    /* entries are already sorted, so we can fill directly */
    for (int i = 0; i < nnz; i++) {
        ci[i] = entries[i].col;
        cv[i] = entries[i].val;
    }

    free(entries);

    *row_ptr_out = rp;
    *col_idx_out = ci;
    *csr_values_out = cv;
}
