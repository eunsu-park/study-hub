# Lesson 15 — Parallel Scan (Prefix Sum) (per-lesson exercise)

Prerequisites: L14 (reduction), L05 (shared memory), L06 (warp execution).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

Scan (prefix sum) is a foundational parallel primitive. Many algorithms — stream compaction, radix sort, sparse matrix operations — reduce to "run a scan, then a gather". Two variants:

- **Inclusive**: `out[i] = a[0] + a[1] + ... + a[i]`
- **Exclusive**: `out[i] = a[0] + a[1] + ... + a[i-1]`, with `out[0] = 0`

---

## Exercise 15.1 — Block-Level Hillis-Steele Scan

**Difficulty**: ★★

### Problem

Implement an inclusive scan on a single block (up to 1024 elements) using the Hillis-Steele step algorithm. It runs in `log2(n)` passes; each thread reads from `tid - d` and writes to `tid`:

```
for d = 1, 2, 4, ..., n/2:
    if tid >= d: sdata[tid] += sdata[tid - d]
    __syncthreads()
```

Writing in place is a race. You MUST either double-buffer or split into read-then-write phases.

### Starter

```cuda
#include <cstdio>
#include <cuda_runtime.h>

__global__ void hillis_steele_inclusive(const float *in, float *out, int n) {
    extern __shared__ float sdata[];  // size = 2 * n floats for ping-pong
    int tid = threadIdx.x;

    float *buf_a = sdata;
    float *buf_b = sdata + blockDim.x;
    buf_a[tid] = (tid < n) ? in[tid] : 0.0f;
    __syncthreads();

    for (int d = 1; d < n; d *= 2) {
        if (tid >= d) buf_b[tid] = buf_a[tid - d] + buf_a[tid];
        else          buf_b[tid] = buf_a[tid];
        __syncthreads();
        // swap read/write buffers
        float *tmp = buf_a; buf_a = buf_b; buf_b = tmp;
    }

    if (tid < n) out[tid] = buf_a[tid];
}

int main(void) {
    const int N = 16;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1);   // 1..16

    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    hillis_steele_inclusive<<<1, N, 2 * N * sizeof(float)>>>(d_in, d_out, N);
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("inclusive scan of 1..16:\n  ");
    for (int i = 0; i < N; i++) printf("%.0f ", h_out[i]);
    printf("\nexpected: 1 3 6 10 15 21 28 36 45 55 66 78 91 105 120 136\n");

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
```

---

## Exercise 15.2 — Multi-Block Scan

**Difficulty**: ★★★

A single block handles up to 1024 elements. For larger arrays, use the **two-pass** scheme:

1. Each block computes a local scan and writes its total to `block_sums[blockIdx.x]`.
2. A single-block scan of `block_sums` produces per-block offsets.
3. Each block adds its offset to its local scan result.

Implement this for `N = 4096` and verify against a host-side reference scan. The key correctness test: `out[N-1]` must equal the sum of the input.

---

## Exercise 15.3 — Exclusive Scan via Shift — Bonus

**Difficulty**: ★★

Given an inclusive-scan kernel, produce the exclusive-scan result by shifting the output right by one and filling `out[0] = 0`. Write this as a post-processing kernel that reads the inclusive result and writes the exclusive. Discuss why a direct exclusive-scan implementation can be one step faster (the first Hillis-Steele iteration can be fused with the shift).
