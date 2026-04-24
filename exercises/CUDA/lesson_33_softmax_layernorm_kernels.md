# Lesson 33 — Softmax and LayerNorm Kernels (per-lesson exercise)

Prerequisites: L14 (reduction), L31 (cooperative groups), basic familiarity with transformer math.

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

Both softmax and layernorm reduce along the last dimension of a `[B, D]` matrix. They share a two-pass structure:

- **Softmax**: pass 1 finds the max; pass 2 computes `exp(x - max)` and its sum; pass 3 divides.
- **LayerNorm**: pass 1 finds the mean; pass 2 finds the variance; pass 3 normalizes.

Modern implementations fuse the passes using warp reductions and online algorithms.

---

## Exercise 33.1 — Block-Per-Row Softmax

**Difficulty**: ★★★

### Problem

Implement `__global__ void softmax_row(const float *x, float *y, int D)` where each block processes one row, `blockDim.x` threads cooperate on one row of length `D`. Assume `D` is a power of two and fits in shared memory.

Steps inside the block:
1. Load row into shared memory.
2. Block-wide reduction to find `max_x`.
3. Write `y[i] = expf(x[i] - max_x)` into shared memory.
4. Block-wide reduction for the normalizer `Z`.
5. Divide: `y[i] /= Z` and write back to global.

### Starter

```cuda
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__global__ void softmax_row(const float *x, float *y, int D) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    const float *row_in  = x + row * D;
    float       *row_out = y + row * D;

    // 1. Load
    sdata[tid] = (tid < D) ? row_in[tid] : -INFINITY;
    __syncthreads();

    // 2. Reduce max — TODO: tree reduction with max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_x = sdata[0];
    __syncthreads();

    // 3. exp(x - max)
    float val = (tid < D) ? expf(row_in[tid] - max_x) : 0.0f;
    sdata[tid] = val;
    __syncthreads();

    // 4. Reduce sum — TODO: tree reduction with +
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float Z = sdata[0];
    __syncthreads();

    // 5. Divide and write
    if (tid < D) row_out[tid] = val / Z;
}

int main(void) {
    const int B = 4, D = 8;
    float h_x[B * D] = {
        1, 2, 3, 4, 5, 6, 7, 8,
        0, 0, 0, 0, 0, 0, 0, 0,
        10, -10, 0, 0, 0, 0, 0, 0,
        1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007,    // numerical-stability check
    };
    float h_y[B * D];
    float *d_x, *d_y;
    cudaMalloc(&d_x, sizeof(h_x));
    cudaMalloc(&d_y, sizeof(h_y));
    cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice);

    softmax_row<<<B, D, D * sizeof(float)>>>(d_x, d_y, D);
    cudaMemcpy(h_y, d_y, sizeof(h_y), cudaMemcpyDeviceToHost);

    for (int r = 0; r < B; r++) {
        float sum = 0;
        for (int i = 0; i < D; i++) { printf("%.4f ", h_y[r * D + i]); sum += h_y[r * D + i]; }
        printf("  (sum=%.4f)\n", sum);
    }

    cudaFree(d_x); cudaFree(d_y);
    return 0;
}
```

Every row should sum to ≈1.0. Row 4 is the numerical-stability test: without the `-max_x` subtraction, `expf(1007)` overflows to infinity and every row entry becomes NaN.

---

## Exercise 33.2 — LayerNorm with Welford's Online Algorithm

**Difficulty**: ★★★★

### Problem

The two-pass mean/variance computation is wasteful. Welford's algorithm computes both in one pass with stable running statistics:

```
for each x:
    count += 1
    delta = x - mean
    mean += delta / count
    M2 += delta * (x - mean)     // note: the SECOND (x - mean) uses updated mean
variance = M2 / count
```

Implement a block-per-row LayerNorm using Welford. Combine per-thread statistics with the Chan et al. parallel combination formula when reducing across threads within a block.

This exercise is a rite of passage for CUDA kernel writers — get the combination math right once, and everyone reads the same paper from then on.
