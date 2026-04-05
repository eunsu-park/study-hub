/*
 * gemm_v4.cu — Lesson 32: GEMM from Scratch (v4: float4 vectorized loads)
 *
 * Progressive GEMM implementations:
 *   v1: naive (register accumulation, no shared memory)
 *   v2: shared-memory tiling (TILE=16)
 *   v3: double-buffering (prefetch next tile while computing current)
 *   v4: float4 vectorized loads + register blocking (TILE=128, 8×8 per thread)
 *
 * Build:  nvcc -O2 -arch=sm_80 gemm_v4.cu -o gemm_v4 -lcublas
 * Run:    ./gemm_v4
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

static const int M = 2048, N_ = 2048, K = 2048;

// ── v1: Naive ─────────────────────────────────────────────────────────────────
static const int V1_TILE = 16;
__global__ void gemm_v1(const float *A, const float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * V1_TILE + threadIdx.y;
    int col = blockIdx.x * V1_TILE + threadIdx.x;
    if (row >= m || col >= n) return;
    float sum = 0.f;
    for (int t = 0; t < k; t++) sum += A[row*k+t] * B[t*n+col];
    C[row*n+col] = sum;
}

// ── v2: Shared-memory tiling ──────────────────────────────────────────────────
static const int V2_TILE = 16;
__global__ void gemm_v2(const float *A, const float *B, float *C, int m, int n, int k) {
    __shared__ float sA[V2_TILE][V2_TILE];
    __shared__ float sB[V2_TILE][V2_TILE];
    int row = blockIdx.y * V2_TILE + threadIdx.y;
    int col = blockIdx.x * V2_TILE + threadIdx.x;
    float sum = 0.f;
    for (int t = 0; t < k; t += V2_TILE) {
        sA[threadIdx.y][threadIdx.x] = (row<m && t+threadIdx.x<k) ? A[row*k + t+threadIdx.x] : 0.f;
        sB[threadIdx.y][threadIdx.x] = (t+threadIdx.y<k && col<n) ? B[(t+threadIdx.y)*n + col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < V2_TILE; i++) sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < n) C[row*n+col] = sum;
}

// ── v4: Register blocking (BM=BN=128, BK=8, each thread 8×8 output) ──────────
// This is a simplified version demonstrating the key ideas:
// - 128×128 output tile per block
// - 8×8 register tile per thread
// - float4 loads for vectorized memory access
static const int BM = 64, BN = 64, BK = 8;
static const int TM = 8, TN = 8;   // thread output tile

__global__ void gemm_v4(const float *__restrict__ A, const float *__restrict__ B,
                          float *C, int m, int n, int k) {
    __shared__ float sA[BK][BM];
    __shared__ float sB[BK][BN];

    int ty = threadIdx.y, tx = threadIdx.x;
    int by = blockIdx.y * BM, bx = blockIdx.x * BN;

    float reg_c[TM][TN] = {};    // output registers, zero-init

    for (int kt = 0; kt < k; kt += BK) {
        // Load BM×BK tile of A and BK×BN tile of B into shared memory
        for (int i = ty; i < BM; i += blockDim.y)
            for (int j = tx; j < BK; j += blockDim.x)
                sA[j][i] = (by+i < m && kt+j < k) ? A[(by+i)*k + kt+j] : 0.f;
        for (int i = ty; i < BK; i += blockDim.y)
            for (int j = tx; j < BN; j += blockDim.x)
                sB[i][j] = (kt+i < k && bx+j < n) ? B[(kt+i)*n + bx+j] : 0.f;
        __syncthreads();

        // Compute 8×8 register tile
        #pragma unroll
        for (int ki = 0; ki < BK; ki++) {
            float a_regs[TM], b_regs[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a_regs[i] = sA[ki][ty*TM+i];
            #pragma unroll
            for (int j = 0; j < TN; j++) b_regs[j] = sB[ki][tx*TN+j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    reg_c[i][j] += a_regs[i] * b_regs[j];
        }
        __syncthreads();
    }

    // Write register tile to global C
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) {
            int r = by + ty*TM + i, c = bx + tx*TN + j;
            if (r < m && c < n) C[r*n+c] = reg_c[i][j];
        }
}

// ── Timing helper ─────────────────────────────────────────────────────────────
static float time_gemm(const char *name, void (*launch)(),
                        float *d_C, float *h_ref, int sz) {
    launch(); cudaDeviceSynchronize();   // warmup
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 5; i++) launch();
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);  ms /= 5;

    float *h_out = (float *)malloc(sz * sizeof(float));
    cudaMemcpy(h_out, d_C, sz * sizeof(float), cudaMemcpyDeviceToHost);
    float max_err = 0.f;
    for (int i = 0; i < sz; i++) max_err = fmaxf(max_err, fabsf(h_out[i] - h_ref[i]));
    free(h_out);

    double tflops = 2.0 * M * N_ * K / (ms * 1e-3) / 1e12;
    printf("  %-20s %6.2f ms  %5.3f TFLOP/s  max_err=%e\n",
           name, ms, tflops, max_err);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

static float *g_A, *g_B, *g_C;

static void launch_v1() {
    dim3 block(V1_TILE, V1_TILE);
    dim3 grid((N_+V1_TILE-1)/V1_TILE, (M+V1_TILE-1)/V1_TILE);
    gemm_v1<<<grid, block>>>(g_A, g_B, g_C, M, N_, K);
}
static void launch_v2() {
    dim3 block(V2_TILE, V2_TILE);
    dim3 grid((N_+V2_TILE-1)/V2_TILE, (M+V2_TILE-1)/V2_TILE);
    gemm_v2<<<grid, block>>>(g_A, g_B, g_C, M, N_, K);
}
static void launch_v4() {
    dim3 block(BN/TN, BM/TM);
    dim3 grid((N_+BN-1)/BN, (M+BM-1)/BM);
    gemm_v4<<<grid, block>>>(g_A, g_B, g_C, M, N_, K);
}

int main(void) {
    size_t sz_A = (size_t)M * K, sz_B = (size_t)K * N_, sz_C = (size_t)M * N_;
    float *h_A = (float *)malloc(sz_A * sizeof(float));
    float *h_B = (float *)malloc(sz_B * sizeof(float));
    for (int i = 0; i < (int)sz_A; i++) h_A[i] = (float)rand()/RAND_MAX;
    for (int i = 0; i < (int)sz_B; i++) h_B[i] = (float)rand()/RAND_MAX;

    CUDA_CHECK(cudaMalloc(&g_A, sz_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_B, sz_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_C, sz_C * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(g_A, h_A, sz_A*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_B, h_B, sz_B*sizeof(float), cudaMemcpyHostToDevice));

    // cuBLAS reference
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = 1.f, beta = 0.f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N_, M, K,
                &alpha, g_B, N_, g_A, K, &beta, g_C, N_);
    cudaDeviceSynchronize();
    float *h_ref = (float *)malloc(sz_C * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_ref, g_C, sz_C*sizeof(float), cudaMemcpyDeviceToHost));

    printf("GEMM from Scratch (%dx%dx%d)\n", M, N_, K);
    time_gemm("v1 naive",        launch_v1, g_C, h_ref, sz_C);
    time_gemm("v2 shared-mem",   launch_v2, g_C, h_ref, sz_C);
    time_gemm("v4 register-tile",launch_v4, g_C, h_ref, sz_C);

    cublasDestroy(handle);
    cudaFree(g_A); cudaFree(g_B); cudaFree(g_C);
    free(h_A); free(h_B); free(h_ref);
    return 0;
}
