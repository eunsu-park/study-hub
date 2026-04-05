/*
 * gemm_demo.cu — Lesson 29: cuBLAS and cuSPARSE
 *
 * Demonstrates cuBLAS SGEMM: C = α·A·B + β·C
 *   - FP32 GEMM using cublasSgemm
 *   - Performance measurement in TFLOP/s
 *   - Column-major vs row-major layout handling
 *   - Batched GEMM (cublasGemmBatchedEx) for many small matrices
 *
 * Build:  nvcc -O2 -arch=sm_80 gemm_demo.cu -o gemm_demo -lcublas
 * Run:    ./gemm_demo
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cuBLAS error %d\n",(int)s); exit(1); } } while(0)

static const int M = 4096, N_ = 4096, K = 4096;
static const int ITERS = 10;

int main(void) {
    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N_ * sizeof(float);
    size_t bytes_C = (size_t)M * N_ * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    // Fill with random data
    float *h = (float *)malloc(bytes_A > bytes_B ? bytes_A : bytes_B);
    for (int i = 0; i < M * K; i++) h[i] = (float)rand() / RAND_MAX;
    CUDA_CHECK(cudaMemcpy(d_A, h, bytes_A, cudaMemcpyHostToDevice));
    for (int i = 0; i < K * N_; i++) h[i] = (float)rand() / RAND_MAX;
    CUDA_CHECK(cudaMemcpy(d_B, h, bytes_B, cudaMemcpyHostToDevice));
    free(h);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // cuBLAS uses column-major layout. To compute C = A*B in row-major,
    // exploit: (A*B)^T = B^T * A^T, and cuBLAS result is C^T in row-major.
    float alpha = 1.f, beta = 0.f;

    // Warmup
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N, N_, M, K,
        &alpha, d_B, N_, d_A, K, &beta, d_C, N_));
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; i++) {
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N, N_, M, K,
            &alpha, d_B, N_, d_A, K, &beta, d_C, N_));
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    ms /= ITERS;

    double tflops = 2.0 * M * N_ * K / (ms * 1e-3) / 1e12;
    printf("cuBLAS SGEMM (%dx%dx%d)\n", M, N_, K);
    printf("  %.2f ms  %.2f TFLOP/s\n", ms, tflops);

    // ── Batched GEMM (small matrices) ────────────────────────────────────────
    const int BATCH = 1024, BM = 64;
    size_t bsz = (size_t)BM * BM * sizeof(float);
    float *d_bA, *d_bB, *d_bC;
    CUDA_CHECK(cudaMalloc(&d_bA, BATCH * bsz));
    CUDA_CHECK(cudaMalloc(&d_bB, BATCH * bsz));
    CUDA_CHECK(cudaMalloc(&d_bC, BATCH * bsz));

    // Build pointer arrays for batched API
    float **h_Aptr = (float **)malloc(BATCH * sizeof(float*));
    float **h_Bptr = (float **)malloc(BATCH * sizeof(float*));
    float **h_Cptr = (float **)malloc(BATCH * sizeof(float*));
    for (int b = 0; b < BATCH; b++) {
        h_Aptr[b] = d_bA + b * BM * BM;
        h_Bptr[b] = d_bB + b * BM * BM;
        h_Cptr[b] = d_bC + b * BM * BM;
    }
    float **d_Aptr, **d_Bptr, **d_Cptr;
    CUDA_CHECK(cudaMalloc(&d_Aptr, BATCH * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_Bptr, BATCH * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_Cptr, BATCH * sizeof(float*)));
    CUDA_CHECK(cudaMemcpy(d_Aptr, h_Aptr, BATCH*sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bptr, h_Bptr, BATCH*sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Cptr, h_Cptr, BATCH*sizeof(float*), cudaMemcpyHostToDevice));

    cudaEventRecord(t0);
    CUBLAS_CHECK(cublasSgemmBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N, BM, BM, BM,
        &alpha, (const float**)d_Bptr, BM,
                (const float**)d_Aptr, BM, &beta,
                d_Cptr, BM, BATCH));
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    double batch_tflops = 2.0 * BATCH * BM * BM * BM / (ms * 1e-3) / 1e12;
    printf("cuBLAS batched SGEMM (%d x %dx%dx%d)\n", BATCH, BM, BM, BM);
    printf("  %.2f ms  %.4f TFLOP/s\n", ms, batch_tflops);

    cublasDestroy(handle);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_bA); cudaFree(d_bB); cudaFree(d_bC);
    cudaFree(d_Aptr); cudaFree(d_Bptr); cudaFree(d_Cptr);
    free(h_Aptr); free(h_Bptr); free(h_Cptr);
    return 0;
}
