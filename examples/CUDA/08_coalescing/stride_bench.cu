/*
 * stride_bench.cu — Lesson 08: Memory Coalescing
 *
 * Demonstrates the impact of memory access stride on effective bandwidth.
 * A stride-1 access pattern (coalesced) achieves peak bandwidth; larger
 * strides create cache-line waste and reduce throughput dramatically.
 *
 * Build:  nvcc -O2 -arch=sm_80 stride_bench.cu -o stride_bench
 * Run:    ./stride_bench
 */

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int N       = 1 << 25;   // 32 M floats = 128 MB
static const int THREADS = 256;
static const int ITERS   = 10;

// ── Strided read kernel ───────────────────────────────────────────────────────
// Thread i reads from index i*stride.
// stride=1  → coalesced (32 consecutive threads read 32 consecutive floats)
// stride=32 → every thread reads a different cache line → no coalescing
__global__ void strided_read(const float * __restrict__ in,
                              float       * __restrict__ out,
                              int n, int stride) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int index  = tid * stride;
    if (index < n)
        out[tid] = in[index];
}

int main(void) {
    const size_t bytes = (size_t)N * sizeof(float);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_in, 0, bytes));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    int strides[] = {1, 2, 4, 8, 16, 32, 64, 128};
    printf("Stride benchmark (%d M floats = %d MB)\n",
           N >> 20, (int)(bytes >> 20));
    printf("%-8s %8s %12s\n", "Stride", "Time(ms)", "BW(GB/s)");
    printf("%-8s %8s %12s\n", "------", "--------", "--------");

    for (int stride : strides) {
        // Elements we can read without overflow
        int safe_n   = N / stride;
        int blocks   = (safe_n + THREADS - 1) / THREADS;

        // Warmup
        strided_read<<<blocks, THREADS>>>(d_in, d_out, N, stride);
        cudaDeviceSynchronize();

        cudaEventRecord(t0);
        for (int it = 0; it < ITERS; it++)
            strided_read<<<blocks, THREADS>>>(d_in, d_out, N, stride);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float ms; cudaEventElapsedTime(&ms, t0, t1);
        ms /= ITERS;

        // Actual bytes touched (each thread reads 4 B but loads a full 128 B cache line)
        // Report useful BW (data moved)
        double bw = (double)safe_n * sizeof(float) / (ms * 1e-3) / 1e9;
        printf("%-8d %8.3f %12.1f\n", stride, ms, bw);
    }

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
