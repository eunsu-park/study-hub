# Lesson 32 — GEMM From Scratch (per-lesson exercise)

Prerequisites: L05 (tiling), L08 (coalescing), L09 (occupancy).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

GEMM (GEneral Matrix Multiply) $C = A B + \beta C$ is the single most-optimized kernel in computing. Writing a reasonable version by hand teaches most of what GPU performance engineering covers. Writing a cuBLAS-competitive version is a career.

This exercise walks through three progressively-better implementations.

---

## Exercise 32.1 — Naive GEMM (Baseline)

**Difficulty**: ★

### Problem

Each thread computes one output element $C[i, j]$ by looping over $K$:

```cuda
__global__ void gemm_naive(const float *A, const float *B, float *C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0;
        for (int k = 0; k < K; k++) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}
```

Compile and time on `M = N = K = 1024`. Record GFLOPS (each GEMM does $2 M N K$ flops).

---

## Exercise 32.2 — Shared-Memory Tiled GEMM

**Difficulty**: ★★★

### Problem

The naive version reads each element of $A$ and $B$ $N$ and $M$ times respectively from global memory. Tile both into shared memory so reads are amortized:

```
for each output tile (blockIdx.y, blockIdx.x) of size TILE_M x TILE_N:
    accumulator = zeros(TILE_M, TILE_N)
    for each inner tile along K of size TILE_K:
        load A_tile (TILE_M x TILE_K) into shared memory
        load B_tile (TILE_K x TILE_N) into shared memory
        __syncthreads()
        for kk in 0..TILE_K:
            accumulator += A_tile[:, kk] * B_tile[kk, :]
        __syncthreads()
    write accumulator to C
```

Use `TILE_M = TILE_N = 32, TILE_K = 32`. On `M = N = K = 1024`, expect a 3–5× speedup over 32.1.

---

## Exercise 32.3 — Register Tiling — Bonus

**Difficulty**: ★★★★

Each thread computes a small output tile (e.g. 4×4) held in registers instead of a single element. This increases arithmetic intensity — every shared-memory load now feeds 4×4 = 16 multiply-adds instead of 1. Combine with double-buffering (async copy while computing the previous tile) to hit within 80% of cuBLAS.

At this point you have written what CUTLASS calls a "Block-level GEMM" and you understand why CUTLASS's 30,000 lines of C++ exist.

---

## Performance Reporting Template

For each of 32.1, 32.2, 32.3 record:

| Kernel | Time (ms) | GFLOPS | % of peak | % of cuBLAS |
|--------|-----------|--------|-----------|-------------|
| Naive | | | | |
| Tiled | | | | |
| Register-tiled | | | | |

A100 peak FP32 is ~19.5 TFLOPS; a hand-tuned register-tiled kernel should reach 7–10 TFLOPS on 1024-cubed, cuBLAS reaches 17 TFLOPS. The gap from register-tiled to cuBLAS is async copies, split-K, and tensor cores — pick one to add next.
