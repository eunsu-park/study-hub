# Lesson 4 — Optimized Matrix Multiply (per-lesson exercise)

Prerequisites: L02 (memory layout), L03 (BLAS basics).

Compile: `gcc -std=c11 -Wall -Wextra -O3 -march=native -o ex ex.c -lm`

GEMM is the canonical "naive vs. tuned" benchmark in C. The naive triple-loop is one-line trivial; making it fast is a graduate course. This exercise walks through the three foundational optimizations that account for most of the speedup before SIMD/CUDA enter the picture.

---

## Exercise 4.1 — Naive Triple Loop

**Difficulty**: ★

```c
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

void gemm_naive(const float *A, const float *B, float *C,
                int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0;
            for (int k = 0; k < K; k++)
                acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

int main(void) {
    int N = 512;
    float *A = malloc(N * N * sizeof(float));
    float *B = malloc(N * N * sizeof(float));
    float *C = malloc(N * N * sizeof(float));
    for (int i = 0; i < N * N; i++) { A[i] = rand() / (float)RAND_MAX; B[i] = rand() / (float)RAND_MAX; }

    clock_t t0 = clock();
    gemm_naive(A, B, C, N, N, N);
    double sec = (double)(clock() - t0) / CLOCKS_PER_SEC;

    double gflops = (2.0 * N * N * N) / sec / 1e9;
    printf("naive: %.2fs  %.2f GFLOPS\n", sec, gflops);

    free(A); free(B); free(C);
    return 0;
}
```

Record the GFLOPS. On a modern laptop this is typically 1–4 GFLOPS — well below peak.

---

## Exercise 4.2 — Loop Reorder for Cache Friendliness

**Difficulty**: ★★

The innermost `B[k * N + j]` access strides `N` between iterations — a cache disaster for moderate $N$. Reordering to `i, k, j` makes the inner loop access `B[k * N + j]` for varying $j$ (sequential, cache-friendly) and `A[i * K + k]` is scalar (loaded once):

```c
void gemm_ikj(const float *A, const float *B, float *C, int M, int N, int K) {
    /* Zero C first since the inner accumulation is now per-element */
    for (int i = 0; i < M * N; i++) C[i] = 0;

    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            float a = A[i * K + k];
            for (int j = 0; j < N; j++)
                C[i * N + j] += a * B[k * N + j];
        }
}
```

This should be 2–4× faster than the naive version on the same hardware. The reason is purely cache locality — same FLOPs, same algorithm, different access pattern.

---

## Exercise 4.3 — Block (Tiled) GEMM

**Difficulty**: ★★★

Even after loop reorder, large matrices outgrow L1 cache. Tiling iterates over $T \times T$ blocks of $C$, computing each block fully before moving on:

```c
#define BLOCK 64

void gemm_tiled(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N * N; i++) C[i] = 0;

    for (int ii = 0; ii < N; ii += BLOCK)
        for (int kk = 0; kk < N; kk += BLOCK)
            for (int jj = 0; jj < N; jj += BLOCK)
                for (int i = ii; i < ii + BLOCK; i++)
                    for (int k = kk; k < kk + BLOCK; k++) {
                        float a = A[i * N + k];
                        for (int j = jj; j < jj + BLOCK; j++)
                            C[i * N + j] += a * B[k * N + j];
                    }
}
```

For `BLOCK = 64`, each working set ($64 \times 64$ float = 16 KiB per block) fits in L1. Expect another 1.5–3× speedup over 4.2.

---

## Exercise 4.4 — OpenBLAS Comparison — Bonus

**Difficulty**: ★

Install OpenBLAS (`apt install libopenblas-dev`) and call `cblas_sgemm`:

```c
#include <cblas.h>

cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
```

OpenBLAS will be 5–20× faster than your tiled version. The remaining gap is SIMD intrinsics, register tiling, and packing — concepts covered by CUTLASS-style libraries. The lesson: hand-rolling GEMM gets you within 2–3× of vendor libraries, but never beating them is the right outcome.

---

## Reporting Template

| Implementation | Time (s) | GFLOPS | Speedup over naive |
|----------------|----------|--------|--------------------|
| Naive (4.1) | | | 1.0× |
| Loop-reorder (4.2) | | | |
| Tiled (4.3) | | | |
| OpenBLAS (4.4) | | | |
