/*
 *  Matrix Market I/O library for ANSI C
 *
 *  Reused from the NIST reference implementation:
 *  https://math.nist.gov/MatrixMarket/mmio-c.html
 */

#include "mmio.h"

int mm_read_banner(FILE *f, MM_typecode *matcode)
{
    char line[MM_MAX_LINE_LENGTH];
    char banner[MM_MAX_TOKEN_LENGTH];
    char mtx[MM_MAX_TOKEN_LENGTH];
    char crd[MM_MAX_TOKEN_LENGTH];
    char data_type[MM_MAX_TOKEN_LENGTH];
    char storage_scheme[MM_MAX_TOKEN_LENGTH];
    char *p;

    mm_initialize_typecode(matcode);

    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
        return MM_PREMATURE_EOF;

    if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type, storage_scheme) != 5)
        return MM_PREMATURE_EOF;

    /* convert to lower case */
    for (p = mtx; *p != '\0'; *p = tolower(*p), p++);
    for (p = crd; *p != '\0'; *p = tolower(*p), p++);
    for (p = data_type; *p != '\0'; *p = tolower(*p), p++);
    for (p = storage_scheme; *p != '\0'; *p = tolower(*p), p++);

    if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
        return MM_NO_HEADER;

    /* object */
    if (strcmp(mtx, MM_MTX_STR) != 0)
        return MM_UNSUPPORTED_TYPE;
    mm_set_matrix(matcode);

    /* format */
    if (strcmp(crd, MM_SPARSE_STR) == 0)
        mm_set_sparse(matcode);
    else if (strcmp(crd, MM_DENSE_STR) == 0)
        mm_set_dense(matcode);
    else
        return MM_UNSUPPORTED_TYPE;

    /* field */
    if (strcmp(data_type, MM_REAL_STR) == 0)
        mm_set_real(matcode);
    else if (strcmp(data_type, MM_COMPLEX_STR) == 0)
        mm_set_complex(matcode);
    else if (strcmp(data_type, MM_PATTERN_STR) == 0)
        mm_set_pattern(matcode);
    else if (strcmp(data_type, MM_INT_STR) == 0)
        mm_set_integer(matcode);
    else
        return MM_UNSUPPORTED_TYPE;

    /* symmetry */
    if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
        mm_set_general(matcode);
    else if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
        mm_set_symmetric(matcode);
    else if (strcmp(storage_scheme, MM_HERM_STR) == 0)
        mm_set_hermitian(matcode);
    else if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
        mm_set_skew(matcode);
    else
        return MM_UNSUPPORTED_TYPE;

    return 0;
}

int mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz)
{
    char line[MM_MAX_LINE_LENGTH];

    /* skip comment lines */
    do {
        if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
            return MM_PREMATURE_EOF;
    } while (line[0] == '%');

    if (sscanf(line, "%d %d %d", M, N, nz) == 3)
        return 0;
    else {
        /* try reading the rest */
        int num_items_read;
        do {
            num_items_read = fscanf(f, "%d %d %d", M, N, nz);
            if (num_items_read == EOF)
                return MM_PREMATURE_EOF;
        } while (num_items_read != 3);
    }

    return 0;
}

int mm_read_mtx_array_size(FILE *f, int *M, int *N)
{
    char line[MM_MAX_LINE_LENGTH];

    /* skip comment lines */
    do {
        if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
            return MM_PREMATURE_EOF;
    } while (line[0] == '%');

    if (sscanf(line, "%d %d", M, N) == 2)
        return 0;
    else {
        int num_items_read;
        do {
            num_items_read = fscanf(f, "%d %d", M, N);
            if (num_items_read == EOF)
                return MM_PREMATURE_EOF;
        } while (num_items_read != 2);
    }

    return 0;
}

char *mm_typecode_to_str(MM_typecode matcode)
{
    static char buffer[MM_MAX_TOKEN_LENGTH + 4];
    char type[MM_MAX_TOKEN_LENGTH];
    char storage[MM_MAX_TOKEN_LENGTH];
    char field[MM_MAX_TOKEN_LENGTH];
    char symm[MM_MAX_TOKEN_LENGTH];

    strcpy(type, "matrix");

    if (mm_is_sparse(matcode))
        strcpy(storage, "coordinate");
    else
        strcpy(storage, "array");

    if (mm_is_real(matcode))
        strcpy(field, "real");
    else if (mm_is_complex(matcode))
        strcpy(field, "complex");
    else if (mm_is_pattern(matcode))
        strcpy(field, "pattern");
    else if (mm_is_integer(matcode))
        strcpy(field, "integer");
    else
        strcpy(field, "???");

    if (mm_is_general(matcode))
        strcpy(symm, "general");
    else if (mm_is_symmetric(matcode))
        strcpy(symm, "symmetric");
    else if (mm_is_hermitian(matcode))
        strcpy(symm, "hermitian");
    else if (mm_is_skew(matcode))
        strcpy(symm, "skew-symmetric");
    else
        strcpy(symm, "???");

    sprintf(buffer, "%s %s %s %s", type, storage, field, symm);
    return buffer;
}
