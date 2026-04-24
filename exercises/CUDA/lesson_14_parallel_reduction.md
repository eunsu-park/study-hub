# Lesson 14 — Parallel Reduction (per-lesson exercise)

Prerequisites: L03 (thread indexing), L05 (shared memory), L06 (warp execution).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex` (adjust `sm_??` for your GPU).

Parallel reduction sums an array using a tree. The classic pattern:

1. Each block loads a chunk into shared memory.
2. Threads do a tree reduction within the block (stride halves each step).
3. The block writes its partial sum; a final pass aggregates the per-block results.

Correctness is the easy part. Performance lessons you should derive from this exercise:

- Warp divergence during the tree (stride-based indexing vs. sequential addressing)
- Bank conflicts in shared memory
- Why `volatile` is needed for the final warp (or why we now use `__syncwarp`)

---

## Exercise 14.1 — Naive Block-Level Reduction

**Difficulty**: ★★

### Problem

Implement `__global__ void reduce_naive(const float *input, float *block_sums, int n)` where each block reduces `blockDim.x` elements from `input` and writes its partial sum to `block_sums[blockIdx.x]`. Assume `n` is a multiple of `blockDim.x`.

### Starter

```cuda
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                               \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        std::exit(1);                                                       \
    }                                                                       \
} while (0)

__global__ void reduce_naive(const float *input, float *block_sums, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // TODO: tree reduction
    //   for stride = blockDim.x/2 down to 1, stride /= 2:
    //     if tid < stride: sdata[tid] += sdata[tid + stride]
    //     __syncthreads()

    if (tid == 0) block_sums[blockIdx.x] = sdata[0];
}

int main(void) {
    int N = 1 << 20;    // 1M elements
    float *h_in  = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;   // sum should be exactly N

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (N / 256) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256;
    int grid  = (N + block - 1) / block;
    reduce_naive<<<grid, block, block * sizeof(float)>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());

    // Host-side final sum
    float *h_out = new float[grid];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, grid * sizeof(float), cudaMemcpyDeviceToHost));
    double total = 0.0;
    for (int i = 0; i < grid; i++) total += h_out[i];
    printf("sum = %.1f (expected %d)\n", total, N);

    delete[] h_in; delete[] h_out;
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    return 0;
}
```

---

## Exercise 14.2 — Eliminate Warp Divergence

**Difficulty**: ★★★

The naive kernel uses `if (tid % (2*s) == 0)` style indexing, which causes half the warp to stall on every step. Rewrite the reduction loop using **sequential addressing** so that the active threads are always the first `stride` of the warp:

```cuda
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

Measure both kernels with `cudaEventElapsedTime`. The sequential-addressing version should be roughly 2× faster on pre-Volta GPUs; on modern GPUs with Independent Thread Scheduling the gap narrows but is still measurable.

---

## Exercise 14.3 — Warp-Level Primitives — Bonus

**Difficulty**: ★★★★

Replace the last six iterations of the tree (where `stride ≤ 32`) with `__shfl_down_sync`. A warp reduction needs no shared memory and no `__syncthreads` — just the shuffle intrinsic. Profile and report the speedup vs. 14.2.
