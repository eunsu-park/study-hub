/*
 * vector_add.cu — Lesson 02: CUDA Programming Model
 *
 * Demonstrates:
 *   - Kernel definition with __global__
 *   - Grid/block launch syntax <<<grid, block>>>
 *   - cudaMalloc / cudaMemcpy / cudaFree
 *   - Error checking with cudaGetLastError
 *
 * Build:  nvcc -O2 -arch=sm_80 vector_add.cu -o vector_add
 * Run:    ./vector_add
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ── Kernel ────────────────────────────────────────────────────────────────────
// Each thread computes one output element: C[i] = A[i] + B[i]
__global__ void vector_add(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
    if (i < n)                                        // bounds guard
        C[i] = A[i] + B[i];
}

// ── Host code ─────────────────────────────────────────────────────────────────
int main(void) {
    const int N     = 1 << 20;          // 1 M elements
    const size_t SZ = N * sizeof(float);

    // Allocate and initialise host arrays
    float *h_A = (float *)malloc(SZ);
    float *h_B = (float *)malloc(SZ);
    float *h_C = (float *)malloc(SZ);
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, SZ));
    CUDA_CHECK(cudaMalloc(&d_B, SZ));
    CUDA_CHECK(cudaMalloc(&d_C, SZ));

    // Copy H→D
    CUDA_CHECK(cudaMemcpy(d_A, h_A, SZ, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, SZ, cudaMemcpyHostToDevice));

    // Launch: 256 threads/block, ceil(N/256) blocks
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy D→H and verify
    CUDA_CHECK(cudaMemcpy(h_C, d_C, SZ, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_C[i] - (float)N) > 1e-3f) { ok = false; break; }
    }
    printf("vector_add (%d elements): %s\n", N, ok ? "PASS" : "FAIL");

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return ok ? 0 : 1;
}
