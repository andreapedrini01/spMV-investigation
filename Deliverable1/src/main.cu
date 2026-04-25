#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#include "utils.h"
#include "spmv_cpu.h"
#include "spmv_gpu.h"

#define TOLERANCE 1e-4f
#define NUM_KERNELS 4

/* ======================================================================== */
static void print_separator(void)
{
    printf("==================================================================\n");
}

/* ======================================================================== */
int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path/to/matrix.mtx>\n", argv[0]);
        return 1;
    }

    const char *mtx_path = argv[1];

    /* optional tolerance override */
    float tol = TOLERANCE;
    if (argc >= 3) {
        tol = atof(argv[2]);
    }

    print_separator();
    print_device_properties();
    print_separator();

    /* ---- Read matrix ---- */
    printf("Reading matrix: %s\n", mtx_path);

    int nrows, ncols, nnz;
    int *coo_row, *coo_col;
    float *coo_val;

    if (read_mtx_file(mtx_path, &nrows, &ncols, &nnz, &coo_row, &coo_col, &coo_val) != 0) {
        fprintf(stderr, "Failed to read matrix file.\n");
        return 1;
    }

    printf("  Rows: %d  Cols: %d  NNZ: %d\n", nrows, ncols, nnz);
    printf("  Avg NNZ/row: %.1f\n", (double)nnz / nrows);
    print_separator();

    /* ---- Convert COO -> CSR ---- */
    int *csr_row_ptr, *csr_col_idx;
    float *csr_val;
    coo_to_csr(coo_row, coo_col, coo_val, nrows, nnz,
               &csr_row_ptr, &csr_col_idx, &csr_val);

    /* ---- Generate random dense vector x ---- */
    srand(42);
    float *x = (float *)malloc(ncols * sizeof(float));
    for (int i = 0; i < ncols; i++) {
        x[i] = (float)rand() / (float)RAND_MAX;
    }

    /* ---- CPU reference (CSR) ---- */
    float *y_cpu = (float *)calloc(nrows, sizeof(float));
    spmv_csr_cpu(csr_row_ptr, csr_col_idx, csr_val, nrows, x, y_cpu);
    printf("CPU reference SpMV completed.\n");

    /* ---- Allocate device memory ---- */
    int *d_row_ptr, *d_col_idx;
    float *d_val, *d_x, *d_y;

    cudaMalloc(&d_row_ptr, (nrows + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_val, nnz * sizeof(float));
    cudaMalloc(&d_x, ncols * sizeof(float));
    cudaMalloc(&d_y, nrows * sizeof(float));

    cudaMemcpy(d_row_ptr, csr_row_ptr, (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, csr_col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, csr_val, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, ncols * sizeof(float), cudaMemcpyHostToDevice);

    float *y_gpu = (float *)malloc(nrows * sizeof(float));
    float timings[MEASURE_ITERS];

    /* store per-kernel results to avoid running twice */
    const char *kernel_names[NUM_KERNELS] = {
        "CSR-Scalar", "CSR-Vector-Shuffle", "CSR-Vector-Shmem", "cuSPARSE"
    };
    float mean_ms_all[NUM_KERNELS];
    float std_ms_all[NUM_KERNELS];
    int   pass_all[NUM_KERNELS];

    /* precompute bandwidth constants */
    double bytes_moved = (double)nnz * (sizeof(float) + sizeof(int))
                       + (double)(nrows + 1) * sizeof(int)
                       + (double)ncols * sizeof(float)
                       + (double)nrows * sizeof(float);

    print_separator();
    printf("Running GPU kernels (warmup=%d, iters=%d, tol=%.1e)\n",
           WARMUP_ITERS, MEASURE_ITERS, tol);
    print_separator();

    /* ---- Kernel 0: CSR-Scalar ---- */
    {
        cudaMemset(d_y, 0, nrows * sizeof(float));
        float mean_ms = run_spmv_csr_scalar(nrows, d_row_ptr, d_col_idx, d_val, d_x, d_y, timings);
        float std_ms = compute_std(timings, MEASURE_ITERS, mean_ms);

        cudaMemcpy(y_gpu, d_y, nrows * sizeof(float), cudaMemcpyDeviceToHost);
        int mismatches = validate_results(y_cpu, y_gpu, nrows, tol);

        double gflops = (2.0 * nnz) / ((mean_ms / 1000.0) * 1e9);
        double bw = bytes_moved / ((mean_ms / 1000.0) * 1e9);

        printf("  %-22s  time = %8.4f ms (std = %.4f)  GFLOP/s = %7.3f  BW = %7.2f GB/s\n",
               kernel_names[0], mean_ms, std_ms, gflops, bw);
        printf("  Validation: %s (%d mismatches)\n",
               mismatches == 0 ? "PASS" : "FAIL", mismatches);

        mean_ms_all[0] = mean_ms;
        std_ms_all[0] = std_ms;
        pass_all[0] = (mismatches == 0);
    }

    /* ---- Kernel 1: CSR-Vector (warp shuffle) ---- */
    {
        cudaMemset(d_y, 0, nrows * sizeof(float));
        float mean_ms = run_spmv_csr_vector(nrows, d_row_ptr, d_col_idx, d_val, d_x, d_y, timings);
        float std_ms = compute_std(timings, MEASURE_ITERS, mean_ms);

        cudaMemcpy(y_gpu, d_y, nrows * sizeof(float), cudaMemcpyDeviceToHost);
        int mismatches = validate_results(y_cpu, y_gpu, nrows, tol);

        double gflops = (2.0 * nnz) / ((mean_ms / 1000.0) * 1e9);
        double bw = bytes_moved / ((mean_ms / 1000.0) * 1e9);

        printf("  %-22s  time = %8.4f ms (std = %.4f)  GFLOP/s = %7.3f  BW = %7.2f GB/s\n",
               kernel_names[1], mean_ms, std_ms, gflops, bw);
        printf("  Validation: %s (%d mismatches)\n",
               mismatches == 0 ? "PASS" : "FAIL", mismatches);

        mean_ms_all[1] = mean_ms;
        std_ms_all[1] = std_ms;
        pass_all[1] = (mismatches == 0);
    }

    /* ---- Kernel 2: CSR-Vector (shared memory) ---- */
    {
        cudaMemset(d_y, 0, nrows * sizeof(float));
        float mean_ms = run_spmv_csr_vector_shmem(nrows, d_row_ptr, d_col_idx, d_val, d_x, d_y, timings);
        float std_ms = compute_std(timings, MEASURE_ITERS, mean_ms);

        cudaMemcpy(y_gpu, d_y, nrows * sizeof(float), cudaMemcpyDeviceToHost);
        int mismatches = validate_results(y_cpu, y_gpu, nrows, tol);

        double gflops = (2.0 * nnz) / ((mean_ms / 1000.0) * 1e9);
        double bw = bytes_moved / ((mean_ms / 1000.0) * 1e9);

        printf("  %-22s  time = %8.4f ms (std = %.4f)  GFLOP/s = %7.3f  BW = %7.2f GB/s\n",
               kernel_names[2], mean_ms, std_ms, gflops, bw);
        printf("  Validation: %s (%d mismatches)\n",
               mismatches == 0 ? "PASS" : "FAIL", mismatches);

        mean_ms_all[2] = mean_ms;
        std_ms_all[2] = std_ms;
        pass_all[2] = (mismatches == 0);
    }

    /* ---- Kernel 3: cuSPARSE ---- */
    {
        cudaMemset(d_y, 0, nrows * sizeof(float));
        float mean_ms = run_spmv_cusparse(nrows, ncols, nnz,
                                          d_row_ptr, d_col_idx, d_val, d_x, d_y, timings);
        float std_ms = compute_std(timings, MEASURE_ITERS, mean_ms);

        cudaMemcpy(y_gpu, d_y, nrows * sizeof(float), cudaMemcpyDeviceToHost);
        int mismatches = validate_results(y_cpu, y_gpu, nrows, tol);

        double gflops = (2.0 * nnz) / ((mean_ms / 1000.0) * 1e9);
        double bw = bytes_moved / ((mean_ms / 1000.0) * 1e9);

        printf("  %-22s  time = %8.4f ms (std = %.4f)  GFLOP/s = %7.3f  BW = %7.2f GB/s\n",
               kernel_names[3], mean_ms, std_ms, gflops, bw);
        printf("  Validation: %s (%d mismatches)\n",
               mismatches == 0 ? "PASS" : "FAIL", mismatches);

        mean_ms_all[3] = mean_ms;
        std_ms_all[3] = std_ms;
        pass_all[3] = (mismatches == 0);
    }

    print_separator();

    /* ---- CSV summary (from stored results, no re-run) ---- */
    const char *matrix_name = strrchr(mtx_path, '/');
    matrix_name = matrix_name ? matrix_name + 1 : mtx_path;

    printf("\nSUMMARY (CSV):\n");
    printf("matrix,nrows,ncols,nnz,kernel,mean_ms,std_ms,gflops,bw_gbs,valid\n");

    for (int k = 0; k < NUM_KERNELS; k++) {
        double mean_s = mean_ms_all[k] / 1000.0;
        double gflops = (2.0 * nnz) / (mean_s * 1e9);
        double bw_gbs = bytes_moved / (mean_s * 1e9);

        printf("%s,%d,%d,%d,%s,%.4f,%.4f,%.3f,%.2f,%s\n",
               matrix_name, nrows, ncols, nnz, kernel_names[k],
               mean_ms_all[k], std_ms_all[k], gflops, bw_gbs,
               pass_all[k] ? "PASS" : "FAIL");
    }

    /* ---- Cleanup ---- */
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_y);

    free(coo_row);
    free(coo_col);
    free(coo_val);
    free(csr_row_ptr);
    free(csr_col_idx);
    free(csr_val);
    free(x);
    free(y_cpu);
    free(y_gpu);

    return 0;
}
