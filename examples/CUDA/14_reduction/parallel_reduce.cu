/*
 * parallel_reduce.cu — Lesson 14: Parallel Reduction
 *
 * Demonstrates a production-quality two-pass reduction:
 *   Pass 1 — each block reduces its chunk → per-block partial sums
 *   Pass 2 — a single block reduces all partial sums
 *
 * Four kernel variants (V0→V3) show progressive optimizations:
 *   V0: divergent interleaved addressing
 *   V1: sequential (divergence-free)
 *   V2: + unrolled last warp (no __syncthreads for last 6 steps)
 *   V3: + warp shuffle (no shared memory in last warp)
 *
 * Build:  nvcc -O2 -arch=sm_80 parallel_reduce.cu -o parallel_reduce
 * Run:    ./parallel_reduce
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)
#define FULL_MASK 0xffffffff

static const int N       = 1 << 24;   // 16 M
static const int THREADS = 256;
static const int ITERS   = 20;

// ── V0: interleaved (divergent) ───────────────────────────────────────────────
__global__ void reduce_v0(const float *g_in, float *g_out, int n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? g_in[i] : 0.f;
    __syncthreads();
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = s[0];
}

// ── V1: sequential addressing (divergence-free) ───────────────────────────────
__global__ void reduce_v1(const float *g_in, float *g_out, int n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? g_in[i] : 0.f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = s[0];
}

// ── V2: last-warp unrolled ────────────────────────────────────────────────────
__device__ void warp_reduce(volatile float *s, int tid) {
    s[tid] += s[tid + 32];
    s[tid] += s[tid + 16];
    s[tid] += s[tid +  8];
    s[tid] += s[tid +  4];
    s[tid] += s[tid +  2];
    s[tid] += s[tid +  1];
}

__global__ void reduce_v2(const float *g_in, float *g_out, int n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? g_in[i] : 0.f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid < 32) warp_reduce(s, tid);
    if (tid == 0) g_out[blockIdx.x] = s[0];
}

// ── V3: warp shuffle (no shared memory for the last warp) ────────────────────
__global__ void reduce_v3(const float *g_in, float *g_out, int n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? g_in[i] : 0.f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    // Final 32 threads: use warp shuffle instead of shared memory
    if (tid < 32) {
        float v = s[tid];
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(FULL_MASK, v, off);
        if (tid == 0) g_out[blockIdx.x] = v;
    }
}

// ── Host-side two-pass driver ─────────────────────────────────────────────────
static double run_two_pass(void (*kern)(const float*, float*, int),
                            const float *d_in, float *d_tmp, float *h_tmp,
                            int n, int blocks) {
    size_t smem = THREADS * sizeof(float);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int it = 0; it < ITERS; it++) {
        kern<<<blocks, THREADS, smem>>>(d_in,  d_tmp, n);
        kern<<<1,      THREADS, smem>>>(d_tmp, d_tmp, blocks);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaMemcpy(h_tmp, d_tmp, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (double)ms / ITERS;
}

int main(void) {
    const size_t bytes = (size_t)N * sizeof(float);
    int blocks = (N + THREADS - 1) / THREADS;

    float *h_in  = (float *)malloc(bytes);
    float *h_tmp = (float *)malloc(blocks * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = 1.f;
    float ref = (float)N;

    float *d_in, *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp, blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    struct { const char *name; void (*fn)(const float*, float*, int); } variants[] = {
        {"V0 (divergent)",  reduce_v0},
        {"V1 (sequential)", reduce_v1},
        {"V2 (unroll warp)",reduce_v2},
        {"V3 (shuffle)",    reduce_v3},
    };

    printf("Parallel reduce (%d elements, %d iters)\n", N, ITERS);
    printf("%-20s %8s %10s\n", "Variant", "ms", "Result");
    for (auto &v : variants) {
        double ms = run_two_pass(v.fn, d_in, d_tmp, h_tmp, N, blocks);
        printf("  %-18s %8.3f %10.0f %s\n",
               v.name, ms, h_tmp[0], fabsf(h_tmp[0] - ref) < 1.f ? "OK" : "FAIL");
    }

    cudaFree(d_in); cudaFree(d_tmp);
    free(h_in); free(h_tmp);
    return 0;
}
