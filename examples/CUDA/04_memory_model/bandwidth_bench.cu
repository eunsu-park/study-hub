/*
 * bandwidth_bench.cu — Lesson 04: CUDA Memory Model
 *
 * Demonstrates:
 *   - Global, constant, and shared memory spaces
 *   - Device bandwidth measurement (GB/s) for each memory tier
 *   - cudaMemGetInfo for device memory stats
 *
 * Build:  nvcc -O2 -arch=sm_80 bandwidth_bench.cu -o bandwidth_bench
 * Run:    ./bandwidth_bench
 */

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int WARMUP  = 3;
static const int ITERS   = 20;
static const int N       = 1 << 24;   // 16 M floats = 64 MB
static const size_t BYTES = (size_t)N * sizeof(float);

// ── Global memory copy ───────────────────────────────────────────────────────
__global__ void global_copy(const float * __restrict__ src,
                             float       * __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// ── Constant memory read ─────────────────────────────────────────────────────
static const int CONST_N = 4096;
__constant__ float c_data[CONST_N];

__global__ void const_read(float *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = c_data[i % CONST_N];
}

// ── Shared memory round-trip ─────────────────────────────────────────────────
// Each block loads a tile into shared memory, then writes it back.
static const int SMEM_TILE = 256;
__global__ void smem_roundtrip(const float *src, float *dst, int n) {
    __shared__ float smem[SMEM_TILE];
    int base = blockIdx.x * blockDim.x;
    int i    = base + threadIdx.x;
    if (i < n) {
        smem[threadIdx.x] = src[i];
        __syncthreads();
        dst[i] = smem[threadIdx.x] * 1.0f;   // prevent dead-code elim
    }
}

// ── Timing helper ─────────────────────────────────────────────────────────────
static double bench(void (*fn)(void), int iters) {
    // warmup
    for (int i = 0; i < WARMUP; i++) fn();
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++) fn();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (double)ms / iters;
}

// Global pointers for lambda-less callbacks
static float *g_src, *g_dst;
static int    g_n;

static void run_global() {
    int threads = 256, blocks = (g_n + threads - 1) / threads;
    global_copy<<<blocks, threads>>>(g_src, g_dst, g_n);
}
static void run_const() {
    int threads = 256, blocks = (g_n + threads - 1) / threads;
    const_read<<<blocks, threads>>>(g_dst, g_n);
}
static void run_smem() {
    int threads = SMEM_TILE, blocks = (g_n + threads - 1) / threads;
    smem_roundtrip<<<blocks, threads>>>(g_src, g_dst, g_n);
}

int main(void) {
    // Report device memory
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    printf("Device memory: %.1f MB free / %.1f MB total\n",
           free_mem / 1e6, total_mem / 1e6);

    CUDA_CHECK(cudaMalloc(&g_src, BYTES));
    CUDA_CHECK(cudaMalloc(&g_dst, BYTES));
    CUDA_CHECK(cudaMemset(g_src, 0, BYTES));
    g_n = N;

    // Upload constant data
    float h_const[CONST_N];
    for (int i = 0; i < CONST_N; i++) h_const[i] = (float)i;
    CUDA_CHECK(cudaMemcpyToSymbol(c_data, h_const, CONST_N * sizeof(float)));

    printf("\nBandwidth benchmark (%d elements = %.0f MB):\n", N, BYTES / 1e6);

    double ms_g = bench(run_global, ITERS);
    double bw_g = 2.0 * BYTES / (ms_g * 1e-3) / 1e9;   // read + write
    printf("  Global memory copy  : %6.2f ms → %6.1f GB/s\n", ms_g, bw_g);

    double ms_c = bench(run_const, ITERS);
    double bw_c = BYTES / (ms_c * 1e-3) / 1e9;          // read only
    printf("  Constant memory read: %6.2f ms → %6.1f GB/s\n", ms_c, bw_c);

    double ms_s = bench(run_smem, ITERS);
    double bw_s = 2.0 * BYTES / (ms_s * 1e-3) / 1e9;
    printf("  Shared memory r/t   : %6.2f ms → %6.1f GB/s\n", ms_s, bw_s);

    cudaFree(g_src); cudaFree(g_dst);
    return 0;
}
