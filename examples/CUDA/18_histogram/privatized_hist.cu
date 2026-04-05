/*
 * privatized_hist.cu — Lesson 18: Histogram and Binning
 *
 * Demonstrates three histogram strategies:
 *   1. Global atomic (baseline)
 *   2. Shared-memory privatization (per-block, then merge)
 *   3. Warp-level privatization using __ldg() + atomics
 *
 * Also shows 2-D histogram as an extension.
 *
 * Build:  nvcc -O2 -arch=sm_80 privatized_hist.cu -o privatized_hist
 * Run:    ./privatized_hist
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int N      = 1 << 24;   // 16 M samples
static const int BINS   = 256;
static const int BLOCKS = 256;
static const int THR    = 256;

// ── 1. Global atomic baseline ─────────────────────────────────────────────────
__global__ void hist_global(const unsigned char *data, unsigned int *hist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride)
        atomicAdd(&hist[data[i]], 1u);
}

// ── 2. Shared-memory privatization ───────────────────────────────────────────
__global__ void hist_shared(const unsigned char *data,
                             unsigned int *hist, int n) {
    __shared__ unsigned int s[BINS];
    for (int b = threadIdx.x; b < BINS; b += blockDim.x) s[b] = 0;
    __syncthreads();

    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(&s[data[i]], 1u);
    __syncthreads();

    for (int b = threadIdx.x; b < BINS; b += blockDim.x)
        atomicAdd(&hist[b], s[b]);
}

// ── 3. Warp-level privatization ───────────────────────────────────────────────
// Each warp maintains its own private histogram in separate shared memory banks
// to reduce intra-warp contention.
static const int WARPS_PER_BLOCK = THR / 32;
__global__ void hist_warp(const unsigned char *data,
                           unsigned int *hist, int n) {
    // Private per-warp histograms
    __shared__ unsigned int s[WARPS_PER_BLOCK][BINS];

    int wid = threadIdx.x / 32;
    for (int b = threadIdx.x % 32; b < BINS; b += 32) s[wid][b] = 0;
    __syncthreads();

    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(&s[wid][__ldg(&data[i])], 1u);
    __syncthreads();

    // Reduce across warps into global histogram
    for (int b = threadIdx.x; b < BINS; b += blockDim.x) {
        unsigned int total = 0;
        for (int w = 0; w < WARPS_PER_BLOCK; w++) total += s[w][b];
        atomicAdd(&hist[b], total);
    }
}

int main(void) {
    unsigned char *h_data = (unsigned char *)malloc(N);
    for (int i = 0; i < N; i++) h_data[i] = (unsigned char)(rand() & 0xFF);

    // Reference histogram
    unsigned int h_ref[BINS] = {0};
    for (int i = 0; i < N; i++) h_ref[h_data[i]]++;

    unsigned char *d_data;
    unsigned int  *d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, N));
    CUDA_CHECK(cudaMalloc(&d_hist, BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    unsigned int h_hist[BINS];
    float ms;

    struct { const char *name;
             void (*fn)(const unsigned char*, unsigned int*, int); } variants[] = {
        {"global  ", hist_global},
        {"shared  ", hist_shared},
        {"warp    ", hist_warp},
    };

    printf("Histogram (%d elements, %d bins)\n", N, BINS);
    for (auto &v : variants) {
        CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(unsigned int)));
        cudaEventRecord(t0);
        v.fn<<<BLOCKS, THR>>>(d_data, d_hist, N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms, t0, t1);
        CUDA_CHECK(cudaMemcpy(h_hist, d_hist, BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        bool ok = (memcmp(h_hist, h_ref, BINS * sizeof(unsigned int)) == 0);
        printf("  %s %.3f ms  %s\n", v.name, ms, ok ? "PASS" : "FAIL");
    }

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_data); cudaFree(d_hist);
    free(h_data);
    return 0;
}
