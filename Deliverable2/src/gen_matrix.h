#ifndef GEN_MATRIX_H
#define GEN_MATRIX_H

#include "distributed.h"

/*
 * Synthetic sparse matrices for weak scaling. The global matrix is square with
 * N = rows_per_rank * P, so each rank owns exactly rows_per_rank rows and the
 * total work grows linearly with P. Every rank fills only the rows it owns
 * under the cyclic rule, so there is no rank-0 bottleneck and no communication
 * during generation. Column indices are global (consumed by the same
 * distributed SpMV path as file matrices).
 */

typedef enum {
    GEN_RANDOM = 0, /* uniform random columns: many ghosts, communication-heavy */
    GEN_BANDED = 1  /* columns near the diagonal: fewer ghosts, structured */
} GenPattern;

/* Generate this rank's local CSR slice directly. nnz_per_row includes the
 * diagonal; it is clamped to N if it exceeds the matrix width. */
void gen_local_csr(int rank, int P, int rows_per_rank, int nnz_per_row,
                   GenPattern pattern, unsigned int seed, LocalCSR *out);

#endif /* GEN_MATRIX_H */
