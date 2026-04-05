/*
 * roofline_bench.cu — Lesson 10: Roofline Model
 *
 * Demonstrates:
 *   - Measuring peak device memory bandwidth (BW-bound kernel: STREAM copy)
 *   - Measuring peak FP32 throughput (compute-bound kernel: FMA loop)
 *   - Computing Arithmetic Intensity (FLOP/byte) for each kernel
 *   - Plotting your kernel's position on the roofline
 *
 * Build:  nvcc -O2 -arch=sm_80 roofline_bench.cu -o roofline_bench -lcublas
 * Run:    ./roofline_bench
 */

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int N     = 1 << 25;   // 32 M elements
static const int ITERS = 20;

// ── Memory-bound: STREAM-copy (AI = 0.5 FLOP/byte) ───────────────────────────
__global__ void stream_copy(const float * __restrict__ src,
                             float       * __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// ── Compute-bound: FMA chain (very high AI) ────────────────────────────────────
// Each thread performs NFMA multiply-adds with zero memory traffic after load.
static const int NFMA = 128;
__global__ void fma_chain(const float *src, float *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = src[i];
    float b = 1.00001f;
    #pragma unroll
    for (int k = 0; k < NFMA; k++)
        a = a * b + 0.1f;        // 2 FLOPs per iteration → 2*NFMA total
    dst[i] = a;
}

// ── Timing helper ─────────────────────────────────────────────────────────────
static float time_ms(const char *label,
                     void (*kern)(const float*, float*, int),
                     const float *d_a, float *d_b, int n, int threads,
                     double *bw_out, double *gflops_out,
                     double bytes_per_elem, double flops_per_elem) {
    int blocks = (n + threads - 1) / threads;
    // warmup
    for (int i = 0; i < 3; i++) kern<<<blocks, threads>>>(d_a, d_b, n);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; i++) kern<<<blocks, threads>>>(d_a, d_b, n);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    ms /= ITERS;

    double total_bytes = (double)n * bytes_per_elem;
    double total_flops = (double)n * flops_per_elem;
    *bw_out     = total_bytes / (ms * 1e-3) / 1e9;
    *gflops_out = total_flops / (ms * 1e-3) / 1e9;
    double ai   = total_flops / total_bytes;

    printf("  %-20s %6.2f ms  BW=%6.1f GB/s  Perf=%6.1f GFLOP/s  AI=%.2f F/B\n",
           label, ms, *bw_out, *gflops_out, ai);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

int main(void) {
    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_a, 0, (size_t)N * sizeof(float)));

    // Query device properties for context
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Roofline benchmark (%d elements)\n\n", N);

    double bw, gflops;
    // STREAM copy: reads 4 bytes + writes 4 bytes, 0 FLOPs (assign is not an FP op)
    // Use 1 FLOP/iter to avoid divide-by-zero in AI display (1 copy op)
    time_ms("stream_copy", stream_copy, d_a, d_b, N, 256, &bw, &gflops,
            2 * sizeof(float), 0.0);
    printf("    → Peak memory BW achieved: %.1f GB/s\n\n", bw);

    time_ms("fma_chain", fma_chain, d_a, d_b, N, 256, &bw, &gflops,
            sizeof(float), 2.0 * NFMA);
    printf("    → Peak compute achieved: %.1f GFLOP/s\n\n", gflops);

    printf("Roofline ridge point ≈ peak_compute / peak_BW  (check values above)\n");

    cudaFree(d_a); cudaFree(d_b);
    return 0;
}
