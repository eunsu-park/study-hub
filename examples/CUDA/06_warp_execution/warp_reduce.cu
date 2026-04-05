/*
 * warp_reduce.cu — Lesson 06: Warp Execution and Divergence
 *
 * Demonstrates:
 *   - Warp divergence in naive reduce (interleaved addressing)
 *   - Divergence-free reduce (sequential addressing)
 *   - Warp shuffle reduce with __shfl_down_sync (no shared memory needed)
 *   - FULL_MASK and lane/warp ID utilities
 *
 * Build:  nvcc -O2 -arch=sm_80 warp_reduce.cu -o warp_reduce
 * Run:    ./warp_reduce
 */

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

#define FULL_MASK 0xffffffff
static const int N       = 1 << 22;   // 4 M elements
static const int THREADS = 256;

// ── Naive reduction (warp divergent) ─────────────────────────────────────────
// stride starts at 1 → threads 0,2,4,… are active → immediate divergence
__global__ void reduce_divergent(const float *g_in, float *g_out, int n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? g_in[i] : 0.f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0)          // diverges within a warp
            s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = s[0];
}

// ── Sequential addressing (divergence-free) ───────────────────────────────────
// Active threads are contiguous → no divergence within a warp
__global__ void reduce_nodiv(const float *g_in, float *g_out, int n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? g_in[i] : 0.f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = s[0];
}

// ── Warp-shuffle reduce ───────────────────────────────────────────────────────
// Uses register-to-register shuffle — no shared memory, no __syncthreads.
// After the warp reduction, lane 0 holds the warp sum.
__device__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(FULL_MASK, v, offset);
    return v;   // valid only in lane 0
}

__global__ void reduce_shuffle(const float *g_in, float *g_out, int n) {
    extern __shared__ float warp_partials[];   // one slot per warp
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (i < n) ? g_in[i] : 0.f;
    val = warp_sum(val);

    int lane   = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) warp_partials[warp_id] = val;
    __syncthreads();

    // Final reduction across warps (single warp)
    int n_warps = blockDim.x / 32;
    if (threadIdx.x < n_warps) {
        val = warp_partials[threadIdx.x];
        val = warp_sum(val);
        if (threadIdx.x == 0) g_out[blockIdx.x] = val;
    }
}

// ── Host helper: sum block results on CPU ─────────────────────────────────────
static float host_sum(const float *arr, int n) {
    float s = 0.f;
    for (int i = 0; i < n; i++) s += arr[i];
    return s;
}

// ── Run one variant and return elapsed ms ────────────────────────────────────
typedef void (*ReduceKernel)(const float*, float*, int);

static double run(ReduceKernel kern, const float *d_in, float *d_tmp,
                  float *h_tmp, int n, int blocks) {
    size_t smem = THREADS * sizeof(float);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    kern<<<blocks, THREADS, smem>>>(d_in, d_tmp, n);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaMemcpy(h_tmp, d_tmp, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

int main(void) {
    const size_t bytes = (size_t)N * sizeof(float);
    int blocks = (N + THREADS - 1) / THREADS;

    float *h_in  = (float *)malloc(bytes);
    float *h_tmp = (float *)malloc(blocks * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = 1.f;   // expected sum = N

    float *d_in, *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp, blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    printf("Warp reduce benchmark (%d elements)\n", N);

    double ms; float sum;

    ms  = run(reduce_divergent, d_in, d_tmp, h_tmp, N, blocks);
    sum = host_sum(h_tmp, blocks);
    printf("  Divergent   : %.3f ms  sum=%.0f %s\n", ms, sum, sum==(float)N?"OK":"FAIL");

    ms  = run(reduce_nodiv, d_in, d_tmp, h_tmp, N, blocks);
    sum = host_sum(h_tmp, blocks);
    printf("  No-diverge  : %.3f ms  sum=%.0f %s\n", ms, sum, sum==(float)N?"OK":"FAIL");

    ms  = run(reduce_shuffle, d_in, d_tmp, h_tmp, N, blocks);
    sum = host_sum(h_tmp, blocks);
    printf("  Shuffle     : %.3f ms  sum=%.0f %s\n", ms, sum, sum==(float)N?"OK":"FAIL");

    cudaFree(d_in); cudaFree(d_tmp);
    free(h_in); free(h_tmp);
    return 0;
}
