/*
 * histogram_atomic.cu — Lesson 07: Atomic Operations
 *
 * Demonstrates:
 *   - Global atomicAdd (baseline)
 *   - Shared-memory atomic privatization (per-block histogram)
 *   - atomicAdd for 64-bit (unsigned long long)
 *   - Performance comparison showing why privatization matters
 *
 * Build:  nvcc -O2 -arch=sm_80 histogram_atomic.cu -o histogram_atomic
 * Run:    ./histogram_atomic
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int N_BINS  = 256;
static const int N       = 1 << 24;   // 16 M elements
static const int THREADS = 256;

// ── Naive: one atomic per element, direct to global histogram ─────────────────
__global__ void hist_global(const unsigned char *data, unsigned int *hist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(&hist[data[i]], 1u);
}

// ── Privatized: accumulate in shared memory, merge once to global ─────────────
__global__ void hist_privatized(const unsigned char *data,
                                unsigned int *hist, int n) {
    __shared__ unsigned int s_hist[N_BINS];

    // Initialise shared histogram
    for (int b = threadIdx.x; b < N_BINS; b += blockDim.x)
        s_hist[b] = 0;
    __syncthreads();

    // Accumulate in shared memory
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(&s_hist[data[i]], 1u);
    __syncthreads();

    // Merge shared histogram into global
    for (int b = threadIdx.x; b < N_BINS; b += blockDim.x)
        atomicAdd(&hist[b], s_hist[b]);
}

int main(void) {
    // Generate random byte data on host
    unsigned char *h_data = (unsigned char *)malloc(N);
    for (int i = 0; i < N; i++) h_data[i] = (unsigned char)(rand() & 0xFF);

    // Build reference histogram on CPU
    unsigned int h_ref[N_BINS] = {0};
    for (int i = 0; i < N; i++) h_ref[h_data[i]]++;

    unsigned char *d_data;
    unsigned int  *d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, N));
    CUDA_CHECK(cudaMalloc(&d_hist, N_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice));

    int blocks = (N + THREADS - 1) / THREADS;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    unsigned int h_hist[N_BINS];
    float ms;

    auto verify = [&](const char *name) {
        CUDA_CHECK(cudaMemcpy(h_hist, d_hist,
                              N_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        bool ok = (memcmp(h_hist, h_ref, N_BINS * sizeof(unsigned int)) == 0);
        printf("  %-15s %.3f ms  %s\n", name, ms, ok ? "PASS" : "FAIL");
    };

    // Global atomic
    CUDA_CHECK(cudaMemset(d_hist, 0, N_BINS * sizeof(unsigned int)));
    cudaEventRecord(t0);
    hist_global<<<blocks, THREADS>>>(d_data, d_hist, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    verify("global atomic");

    // Privatized
    CUDA_CHECK(cudaMemset(d_hist, 0, N_BINS * sizeof(unsigned int)));
    // Use fewer blocks (each block loops over input) for privatized variant
    int priv_blocks = 256;
    cudaEventRecord(t0);
    hist_privatized<<<priv_blocks, THREADS>>>(d_data, d_hist, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    verify("privatized");

    printf("\nHistogram benchmark: %d elements, %d bins\n", N, N_BINS);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_data); cudaFree(d_hist);
    free(h_data);
    return 0;
}
