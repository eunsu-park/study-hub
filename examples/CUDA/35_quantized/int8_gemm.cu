/*
 * int8_gemm.cu — Lesson 35: Quantized Kernels (INT8)
 *
 * Demonstrates INT8 quantized matrix multiplication:
 *   1. Symmetric per-tensor quantization (float → int8)
 *   2. Custom INT8 GEMM using __dp4a (DP4A) hardware dot-product
 *   3. Dequantization of INT32 accumulator back to FP32
 *   4. cuBLAS INT8 GEMM via cublasGemmEx (for reference)
 *
 * DP4A: dot product of 4 signed int8 values → int32
 *   __dp4a(a, b, c): c += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
 *
 * Build:  nvcc -O2 -arch=sm_80 int8_gemm.cu -o int8_gemm -lcublas
 * Run:    ./int8_gemm
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
// __dp4a is a built-in PTX instruction available on sm_61+; no extra header needed.

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cuBLAS error %d line %d\n",(int)s,__LINE__); exit(1); } } while(0)

static const int M = 1024, N_ = 1024, K = 1024;

// ── Quantization helpers ──────────────────────────────────────────────────────
__global__ void quantize_fp32_to_int8(const float *src, int8_t *dst,
                                       float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = roundf(src[i] / scale);
    v = fmaxf(-127.f, fminf(127.f, v));
    dst[i] = (int8_t)v;
}

__global__ void dequantize_int32_to_fp32(const int *src, float *dst,
                                          float scale_a, float scale_b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) dst[i] = (float)src[i] * scale_a * scale_b;
}

// ── Custom INT8 GEMM using DP4A ────────────────────────────────────────────────
// Layout: A is M×K, B is K×N (row-major). Tile = 16×16.
// K must be divisible by 4 (DP4A processes 4 elements at once).
static const int TILE_I8 = 16;

__global__ void gemm_int8(const int8_t *A, const int8_t *B, int *C,
                           int m, int n, int k) {
    __shared__ int8_t sA[TILE_I8][TILE_I8 * 4];   // 16 rows × 64 elements
    __shared__ int8_t sB[TILE_I8][TILE_I8 * 4];

    int row = blockIdx.y * TILE_I8 + threadIdx.y;
    int col = blockIdx.x * TILE_I8 + threadIdx.x;
    int acc = 0;

    for (int kt = 0; kt < k; kt += TILE_I8 * 4) {
        // Load tiles (each thread loads 4 consecutive int8 values)
        for (int j = 0; j < 4; j++) {
            int a_col = kt + threadIdx.x * 4 / TILE_I8 * TILE_I8 + j;  // simplified
            sA[threadIdx.y][threadIdx.x * 4 + j] =
                (row < m && kt + threadIdx.x * 4 + j < k) ?
                A[row * k + kt + threadIdx.x * 4 + j] : 0;
            sB[threadIdx.y][threadIdx.x * 4 + j] =
                (kt + threadIdx.y * 4 + j < k && col < n) ?
                B[(kt + threadIdx.y * 4 + j) * n + col] : 0;
        }
        __syncthreads();

        // DP4A: process 4 int8 values per cycle
        for (int ki = 0; ki < TILE_I8; ki++) {
            // Pack 4 consecutive int8 into int32 for __dp4a
            int a_pack = *reinterpret_cast<const int*>(&sA[threadIdx.y][ki * 4]);
            int b_pack = *reinterpret_cast<const int*>(&sB[ki][threadIdx.x * 4]);
            acc = __dp4a(a_pack, b_pack, acc);
        }
        __syncthreads();
    }

    if (row < m && col < n)
        C[row * n + col] = acc;
}

int main(void) {
    // FP32 reference data
    float *h_Af = (float *)malloc(M * K * sizeof(float));
    float *h_Bf = (float *)malloc(K * N_ * sizeof(float));
    for (int i = 0; i < M * K; i++) h_Af[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.2f;
    for (int i = 0; i < K * N_; i++) h_Bf[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.2f;

    // Find scales
    float scale_A = 0.f, scale_B = 0.f;
    for (int i = 0; i < M*K;  i++) scale_A = fmaxf(scale_A, fabsf(h_Af[i]));
    for (int i = 0; i < K*N_; i++) scale_B = fmaxf(scale_B, fabsf(h_Bf[i]));
    scale_A /= 127.f; scale_B /= 127.f;

    float *d_Af, *d_Bf, *d_Cf;
    int8_t *d_Ai, *d_Bi;
    int    *d_Ci;
    CUDA_CHECK(cudaMalloc(&d_Af, M * K  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Bf, K  * N_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Cf, M * N_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ai, M * K  * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_Bi, K  * N_ * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_Ci, M * N_ * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_Af, h_Af, M*K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bf, h_Bf, K*N_* sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    quantize_fp32_to_int8<<<(M*K +threads-1)/threads, threads>>>(d_Af, d_Ai, scale_A, M*K);
    quantize_fp32_to_int8<<<(K*N_+threads-1)/threads, threads>>>(d_Bf, d_Bi, scale_B, K*N_);

    // FP32 reference via cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = 1.f, beta = 0.f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N_, M, K,
                &alpha, d_Bf, N_, d_Af, K, &beta, d_Cf, N_);
    float *h_ref = (float *)malloc(M * N_ * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_ref, d_Cf, M*N_*sizeof(float), cudaMemcpyDeviceToHost));

    // Custom INT8 GEMM
    dim3 block(TILE_I8, TILE_I8);
    dim3 grid((N_+TILE_I8-1)/TILE_I8, (M+TILE_I8-1)/TILE_I8);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    gemm_int8<<<grid, block>>>(d_Ai, d_Bi, d_Ci, M, N_, K);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    // Dequantize
    dequantize_int32_to_fp32<<<(M*N_+threads-1)/threads, threads>>>(
        d_Ci, d_Cf, scale_A, scale_B, M*N_);
    float *h_out = (float *)malloc(M * N_ * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_out, d_Cf, M*N_*sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.f;
    for (int i = 0; i < M * N_; i++)
        max_err = fmaxf(max_err, fabsf(h_out[i] - h_ref[i]));

    double tflops = 2.0 * M * N_ * K / (ms * 1e-3) / 1e12;
    printf("INT8 GEMM (%dx%dx%d)\n", M, N_, K);
    printf("  Custom DP4A: %.2f ms  %.3f TFLOP/s  max_err=%.3e\n",
           ms, tflops, max_err);

    cublasDestroy(handle);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_Af); cudaFree(d_Bf); cudaFree(d_Cf);
    cudaFree(d_Ai); cudaFree(d_Bi); cudaFree(d_Ci);
    free(h_Af); free(h_Bf); free(h_ref); free(h_out);
    return 0;
}
