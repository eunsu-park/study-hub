/*
 * spmv_demo.cu — Lesson 19: Sparse Matrix Operations
 *
 * Demonstrates:
 *   - CSR (Compressed Sparse Row) format construction
 *   - Custom SpMV (y = A*x) kernel using CSR
 *   - cuSPARSE SpMV via cusparseSpMV (generic API, CUDA 11+)
 *   - Performance comparison: custom vs cuSPARSE
 *
 * Build:  nvcc -O2 -arch=sm_80 spmv_demo.cu -o spmv_demo -lcusparse
 * Run:    ./spmv_demo
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cusparse.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)
#define CUSPARSE_CHECK(x) do { cusparseStatus_t s=(x); if(s!=CUSPARSE_STATUS_SUCCESS){ \
    fprintf(stderr,"cuSPARSE error %d\n",(int)s); exit(1); } } while(0)

static const int M   = 4096;     // rows
static const int N   = 4096;     // cols
static const int NNZ_PER_ROW = 8; // sparsity

// ── Custom CSR SpMV ──────────────────────────────────────────────────────────
// One thread per row — suitable for uniform sparsity
__global__ void spmv_csr(int m, const int *row_ptr, const int *col_idx,
                          const float *vals, const float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    float sum = 0.f;
    for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++)
        sum += vals[j] * x[col_idx[j]];
    y[row] = sum;
}

int main(void) {
    int nnz = M * NNZ_PER_ROW;

    // Build a synthetic banded CSR matrix on host
    int   *h_row_ptr = (int   *)malloc((M + 1) * sizeof(int));
    int   *h_col_idx = (int   *)malloc(nnz     * sizeof(int));
    float *h_vals    = (float *)malloc(nnz     * sizeof(float));
    float *h_x       = (float *)malloc(N       * sizeof(float));
    float *h_y       = (float *)malloc(M       * sizeof(float));
    float *h_ref     = (float *)malloc(M       * sizeof(float));

    srand(42);
    for (int i = 0; i < N; i++) h_x[i] = (float)rand() / RAND_MAX;

    int ptr = 0;
    h_row_ptr[0] = 0;
    for (int r = 0; r < M; r++) {
        for (int k = 0; k < NNZ_PER_ROW; k++) {
            h_col_idx[ptr] = (r * NNZ_PER_ROW + k) % N;
            h_vals[ptr]    = (float)rand() / RAND_MAX;
            ptr++;
        }
        h_row_ptr[r + 1] = ptr;
    }

    // CPU reference
    for (int r = 0; r < M; r++) {
        float s = 0.f;
        for (int j = h_row_ptr[r]; j < h_row_ptr[r + 1]; j++)
            s += h_vals[j] * h_x[h_col_idx[j]];
        h_ref[r] = s;
    }

    // Allocate device
    int   *d_row_ptr, *d_col_idx;
    float *d_vals, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz     * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals,    nnz     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x,       N       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,       M       * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr, (M+1)*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, h_col_idx, nnz  *sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals,    h_vals,    nnz  *sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x,       h_x,       N    *sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    // ── Custom kernel ──
    int threads = 256, blocks = (M + threads - 1) / threads;
    cudaEventRecord(t0);
    spmv_csr<<<blocks, threads>>>(M, d_row_ptr, d_col_idx, d_vals, d_x, d_y);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_custom; cudaEventElapsedTime(&ms_custom, t0, t1);
    CUDA_CHECK(cudaMemcpy(h_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.f;
    for (int i = 0; i < M; i++) max_err = fmaxf(max_err, fabsf(h_y[i] - h_ref[i]));

    printf("SpMV (%dx%d, %d nnz, nnz/row=%d)\n", M, N, nnz, NNZ_PER_ROW);
    printf("  Custom CSR   : %.3f ms  max_err=%e  %s\n",
           ms_custom, max_err, max_err < 1e-3f ? "PASS" : "FAIL");

    // ── cuSPARSE ──
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, M, N, nnz,
        d_row_ptr, d_col_idx, d_vals,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, N, d_x, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, M, d_y, CUDA_R_32F));

    float alpha = 1.f, beta = 0.f;
    size_t buf_sz;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_sz));
    void *d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, buf_sz));

    cudaEventRecord(t0);
    CUSPARSE_CHECK(cusparseSpMV(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf));
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_cusparse; cudaEventElapsedTime(&ms_cusparse, t0, t1);

    CUDA_CHECK(cudaMemcpy(h_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost));
    max_err = 0.f;
    for (int i = 0; i < M; i++) max_err = fmaxf(max_err, fabsf(h_y[i] - h_ref[i]));
    printf("  cuSPARSE     : %.3f ms  max_err=%e  %s\n",
           ms_cusparse, max_err, max_err < 1e-3f ? "PASS" : "FAIL");

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
    cudaFree(d_buf);
    cudaFree(d_row_ptr); cudaFree(d_col_idx);
    cudaFree(d_vals); cudaFree(d_x); cudaFree(d_y);
    free(h_row_ptr); free(h_col_idx); free(h_vals);
    free(h_x); free(h_y); free(h_ref);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
