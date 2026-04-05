/*
 * tiled_matmul.cu — Lesson 05: Shared Memory and Tiling
 *
 * Demonstrates:
 *   - Tiled matrix multiplication C = A × B
 *   - __shared__ memory allocation and __syncthreads() usage
 *   - Bank-conflict reduction via +1 column padding
 *   - Performance comparison: naive vs tiled
 *
 * Build:  nvcc -O2 -arch=sm_80 tiled_matmul.cu -o tiled_matmul -lcublas
 * Run:    ./tiled_matmul
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int N    = 512;   // matrix dimension
static const int TILE = 16;    // shared-memory tile size

// ── Naive matmul ──────────────────────────────────────────────────────────────
// Each thread reads entire rows/columns from global memory (no reuse).
__global__ void matmul_naive(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;

    float sum = 0.f;
    for (int k = 0; k < n; k++)
        sum += A[row * n + k] * B[k * n + col];
    C[row * n + col] = sum;
}

// ── Tiled matmul ──────────────────────────────────────────────────────────────
// Loads TILE×TILE sub-tiles of A and B into shared memory; each element
// of global memory is loaded only once per tile → TILE× fewer global reads.
__global__ void matmul_tiled(const float *A, const float *B, float *C, int n) {
    // +1 padding on A tile to avoid bank conflicts when accessing columns
    __shared__ float sA[TILE][TILE + 1];
    __shared__ float sB[TILE][TILE + 1];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.f;
    int tiles = (n + TILE - 1) / TILE;

    for (int t = 0; t < tiles; t++) {
        // Collaboratively load one tile from A and B
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < n && a_col < n) ? A[row * n + a_col] : 0.f;
        sB[threadIdx.y][threadIdx.x] = (b_row < n && col < n) ? B[b_row * n + col] : 0.f;
        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
static float elapsed_ms(cudaEvent_t t0, cudaEvent_t t1) {
    float ms; cudaEventElapsedTime(&ms, t0, t1); return ms;
}

int main(void) {
    const size_t bytes = (size_t)N * N * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    // Naive
    cudaEventRecord(t0);
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_naive = elapsed_ms(t0, t1);
    CUDA_CHECK(cudaMemcpy(h_ref, d_C, bytes, cudaMemcpyDeviceToHost));

    // Tiled
    cudaEventRecord(t0);
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_tiled = elapsed_ms(t0, t1);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify
    float max_err = 0.f;
    for (int i = 0; i < N * N; i++)
        max_err = fmaxf(max_err, fabsf(h_C[i] - h_ref[i]));

    double flops = 2.0 * N * N * N;
    printf("Matrix multiply (%dx%d), TILE=%d\n", N, N, TILE);
    printf("  Naive  : %7.2f ms  %5.1f GFLOP/s\n",
           ms_naive, flops / (ms_naive * 1e-3) / 1e9);
    printf("  Tiled  : %7.2f ms  %5.1f GFLOP/s\n",
           ms_tiled, flops / (ms_tiled * 1e-3) / 1e9);
    printf("  Max err: %e\n", max_err);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return (max_err < 1e-2f) ? 0 : 1;
}
