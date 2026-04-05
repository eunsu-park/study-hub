/*
 * fp16_gemm.cu — Lesson 30: Mixed Precision and Tensor Cores
 *
 * Demonstrates FP16 (half-precision) GEMM using cuBLAS Tensor Cores:
 *   - cuBLAS cublasGemmEx with CUDA_R_16F inputs and FP32 accumulator
 *   - Manual FP32→FP16 conversion using __float2half
 *   - Performance comparison: FP32 vs FP16 GEMM
 *   - Notes on Tensor Core activation requirements (M,N,K multiples of 8/16)
 *
 * Build:  nvcc -O2 -arch=sm_80 fp16_gemm.cu -o fp16_gemm -lcublas
 * Run:    ./fp16_gemm
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cuBLAS error %d\n",(int)s); exit(1); } } while(0)

static const int M = 4096, N_ = 4096, K = 4096;
static const int ITERS = 10;

// ── Convert FP32 array to FP16 on device ──────────────────────────────────────
__global__ void cvt_fp32_to_fp16(const float *in, __half *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

int main(void) {
    // FP32 buffers
    float *d_Af32, *d_Bf32, *d_Cf32;
    CUDA_CHECK(cudaMalloc(&d_Af32, (size_t)M * K  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Bf32, (size_t)K  * N_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Cf32, (size_t)M * N_ * sizeof(float)));

    // Fill with random FP32 data
    float *h = (float *)malloc((size_t)M * K * sizeof(float));
    for (int i = 0; i < M * K; i++) h[i] = (float)rand() / RAND_MAX - 0.5f;
    CUDA_CHECK(cudaMemcpy(d_Af32, h, M * K * sizeof(float), cudaMemcpyHostToDevice));
    for (int i = 0; i < K * N_; i++) h[i] = (float)rand() / RAND_MAX - 0.5f;
    CUDA_CHECK(cudaMemcpy(d_Bf32, h, K * N_ * sizeof(float), cudaMemcpyHostToDevice));
    free(h);

    // FP16 buffers
    __half *d_Af16, *d_Bf16;
    CUDA_CHECK(cudaMalloc(&d_Af16, (size_t)M * K  * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_Bf16, (size_t)K  * N_ * sizeof(__half)));

    int threads = 256;
    cvt_fp32_to_fp16<<<(M*K  + threads-1)/threads, threads>>>(d_Af32, d_Af16, M*K );
    cvt_fp32_to_fp16<<<(K*N_ + threads-1)/threads, threads>>>(d_Bf32, d_Bf16, K*N_);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    float alpha = 1.f, beta = 0.f;

    // ── FP32 baseline ─────────────────────────────────────────────────────────
    for (int i = 0; i < 3; i++)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N_, M, K, &alpha, d_Bf32, N_, d_Af32, K, &beta, d_Cf32, N_);
    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; i++)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N_, M, K, &alpha, d_Bf32, N_, d_Af32, K, &beta, d_Cf32, N_);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_fp32; cudaEventElapsedTime(&ms_fp32, t0, t1);
    ms_fp32 /= ITERS;

    // ── FP16 + Tensor Cores ───────────────────────────────────────────────────
    // Use CUBLAS_GEMM_DEFAULT_TENSOR_OP to enable Tensor Cores when possible.
    // Input: FP16, accumulator: FP32 (best precision/performance trade-off).
    for (int i = 0; i < 3; i++)
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N_, M, K, &alpha,
                     d_Bf16, CUDA_R_16F, N_,
                     d_Af16, CUDA_R_16F, K,
                     &beta,  d_Cf32, CUDA_R_32F, N_,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; i++)
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N_, M, K, &alpha,
                     d_Bf16, CUDA_R_16F, N_,
                     d_Af16, CUDA_R_16F, K,
                     &beta,  d_Cf32, CUDA_R_32F, N_,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_fp16; cudaEventElapsedTime(&ms_fp16, t0, t1);
    ms_fp16 /= ITERS;

    double flops = 2.0 * M * N_ * K;
    printf("Mixed Precision GEMM (%dx%dx%d)\n", M, N_, K);
    printf("  FP32 (cuBLAS)     : %6.2f ms  %5.2f TFLOP/s\n",
           ms_fp32, flops / (ms_fp32 * 1e-3) / 1e12);
    printf("  FP16 TensorCores  : %6.2f ms  %5.2f TFLOP/s  (%.1fx)\n",
           ms_fp16, flops / (ms_fp16 * 1e-3) / 1e12, ms_fp32 / ms_fp16);

    cublasDestroy(handle);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_Af32); cudaFree(d_Bf32); cudaFree(d_Cf32);
    cudaFree(d_Af16); cudaFree(d_Bf16);
    return 0;
}
