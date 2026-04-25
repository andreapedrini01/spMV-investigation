/*
 *  Matrix Market I/O library for ANSI C
 *
 *  Reused from the NIST reference implementation:
 *  https://math.nist.gov/MatrixMarket/mmio-c.html
 *
 */

#ifndef MM_IO_H
#define MM_IO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MM_MAX_LINE_LENGTH 1025
#define MM_MAX_TOKEN_LENGTH 64
#define MatrixMarketBanner "%%MatrixMarket"
#define MM_MTX_STR         "matrix"
#define MM_SPARSE_STR      "coordinate"
#define MM_DENSE_STR       "array"
#define MM_REAL_STR        "real"
#define MM_COMPLEX_STR     "complex"
#define MM_PATTERN_STR     "pattern"
#define MM_INT_STR         "integer"
#define MM_GENERAL_STR     "general"
#define MM_SYMM_STR        "symmetric"
#define MM_HERM_STR        "hermitian"
#define MM_SKEW_STR        "skew-symmetric"

typedef char MM_typecode[4];

/* typecode positions */
#define MM_OBJ_IDX   0
#define MM_FMT_IDX   1
#define MM_FIELD_IDX 2
#define MM_SYMM_IDX  3

/* typecode queries */
#define mm_is_matrix(typecode)     ((typecode)[MM_OBJ_IDX] == 'M')
#define mm_is_sparse(typecode)     ((typecode)[MM_FMT_IDX] == 'C')
#define mm_is_coordinate(typecode) ((typecode)[MM_FMT_IDX] == 'C')
#define mm_is_dense(typecode)      ((typecode)[MM_FMT_IDX] == 'A')
#define mm_is_array(typecode)      ((typecode)[MM_FMT_IDX] == 'A')
#define mm_is_complex(typecode)    ((typecode)[MM_FIELD_IDX] == 'C')
#define mm_is_real(typecode)       ((typecode)[MM_FIELD_IDX] == 'R')
#define mm_is_pattern(typecode)    ((typecode)[MM_FIELD_IDX] == 'P')
#define mm_is_integer(typecode)    ((typecode)[MM_FIELD_IDX] == 'I')
#define mm_is_symmetric(typecode)  ((typecode)[MM_SYMM_IDX] == 'S')
#define mm_is_general(typecode)    ((typecode)[MM_SYMM_IDX] == 'G')
#define mm_is_skew(typecode)       ((typecode)[MM_SYMM_IDX] == 'K')
#define mm_is_hermitian(typecode)  ((typecode)[MM_SYMM_IDX] == 'H')

/* typecode setters */
#define mm_set_matrix(typecode)     ((*typecode)[MM_OBJ_IDX] = 'M')
#define mm_set_coordinate(typecode) ((*typecode)[MM_FMT_IDX] = 'C')
#define mm_set_array(typecode)      ((*typecode)[MM_FMT_IDX] = 'A')
#define mm_set_dense(typecode)      mm_set_array(typecode)
#define mm_set_sparse(typecode)     mm_set_coordinate(typecode)
#define mm_set_complex(typecode)    ((*typecode)[MM_FIELD_IDX] = 'C')
#define mm_set_real(typecode)       ((*typecode)[MM_FIELD_IDX] = 'R')
#define mm_set_pattern(typecode)    ((*typecode)[MM_FIELD_IDX] = 'P')
#define mm_set_integer(typecode)    ((*typecode)[MM_FIELD_IDX] = 'I')
#define mm_set_symmetric(typecode)  ((*typecode)[MM_SYMM_IDX] = 'S')
#define mm_set_general(typecode)    ((*typecode)[MM_SYMM_IDX] = 'G')
#define mm_set_skew(typecode)       ((*typecode)[MM_SYMM_IDX] = 'K')
#define mm_set_hermitian(typecode)  ((*typecode)[MM_SYMM_IDX] = 'H')

#define mm_initialize_typecode(typecode) \
    ((*typecode)[0] = (*typecode)[1] = (*typecode)[2] = ' ', (*typecode)[3] = 'G')

/* error codes */
#define MM_COULD_NOT_READ_FILE   11
#define MM_PREMATURE_EOF         12
#define MM_NOT_MTX               13
#define MM_NO_HEADER             14
#define MM_UNSUPPORTED_TYPE      15
#define MM_LINE_TOO_LONG         16
#define MM_COULD_NOT_WRITE_FILE  17

/* function prototypes */
int mm_read_banner(FILE *f, MM_typecode *matcode);
int mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz);
int mm_read_mtx_array_size(FILE *f, int *M, int *N);
char *mm_typecode_to_str(MM_typecode matcode);

#endif /* MM_IO_H */
