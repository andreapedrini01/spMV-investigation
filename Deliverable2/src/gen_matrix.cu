#include <stdio.h>
#include <stdlib.h>

#include "gen_matrix.h"
#include "utils.h"

/* Small reproducible PRNG so a generated matrix is identical for a given
 * (seed, rank, P, rows_per_rank, pattern) regardless of process count. */
static inline unsigned int xorshift32(unsigned int *s)
{
    unsigned int x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}
static inline float rng_float(unsigned int *s)
{
    return (float)(xorshift32(s) & 0x00FFFFFFu) / (float)0x01000000u; /* [0, 1) */
}
static int cmp_int_gen(const void *a, const void *b)
{
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

void gen_local_csr(int rank, int P, int rows_per_rank, int nnz_per_row,
                   GenPattern pattern, unsigned int seed, LocalCSR *out)
{
    int N = rows_per_rank * P;               /* square global matrix */
    int n_local = count_cyclic(N, P, rank);  /* == rows_per_rank */

    int rownnz = nnz_per_row;
    if (rownnz > N) rownnz = N;
    if (rownnz < 1) rownnz = 1;
    int cap_per_row = rownnz + 1;            /* banded may yield one extra */

    int *row_ptr = (int *)calloc(n_local + 1, sizeof(int));
    int *col = (int *)malloc((size_t)n_local * cap_per_row * sizeof(int));
    float *val = (float *)malloc((size_t)n_local * cap_per_row * sizeof(float));
    int *rowcols = (int *)malloc((size_t)cap_per_row * sizeof(int));
    if (!row_ptr || !col || !val || !rowcols) {
        fprintf(stderr, "gen_local_csr: allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int nnz = 0;
    for (int k = 0; k < n_local; k++) {
        int i = rank + k * P;                /* global row index */
        unsigned int st = seed ^ (unsigned int)((long long)i * 2654435761u);
        if (st == 0) st = 0x9E3779B9u;

        int rc = 0;
        if (pattern == GEN_BANDED) {
            /* symmetric band centred on the diagonal */
            int half = rownnz / 2;
            for (int d = -half; d <= half; d++) {
                int c = i + d;
                if (c >= 0 && c < N) rowcols[rc++] = c;
            }
        } else {
            /* diagonal plus distinct uniform-random columns */
            rowcols[rc++] = i;
            int attempts = 0, attempt_cap = 20 * rownnz + 100;
            while (rc < rownnz && attempts++ < attempt_cap) {
                int c = (int)(xorshift32(&st) % (unsigned int)N);
                int dup = 0;
                for (int t = 0; t < rc; t++)
                    if (rowcols[t] == c) { dup = 1; break; }
                if (!dup) rowcols[rc++] = c;
            }
            qsort(rowcols, rc, sizeof(int), cmp_int_gen);
        }

        for (int t = 0; t < rc; t++) {
            col[nnz] = rowcols[t];
            val[nnz] = rng_float(&st) + 0.5f; /* keep values away from zero */
            nnz++;
        }
        row_ptr[k + 1] = nnz;
    }

    free(rowcols);

    out->n_local = n_local;
    out->ncols = N;
    out->nnz_local = nnz;
    out->nrows_global = N;
    out->row_ptr = row_ptr;
    out->col_idx = col;
    out->val = val;
}
