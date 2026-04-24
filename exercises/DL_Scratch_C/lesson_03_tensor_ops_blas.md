# Lesson 3 — Tensor Ops and BLAS Basics (per-lesson exercise)

Prerequisites: L02 (memory layout).

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

BLAS (Basic Linear Algebra Subroutines) defines three levels of operations:

- **Level 1**: vector–vector (e.g., `axpy`, dot product) — O(n) work, O(n) data
- **Level 2**: matrix–vector (e.g., `gemv`) — O(n²) work, O(n²) data
- **Level 3**: matrix–matrix (e.g., `gemm`) — O(n³) work, O(n²) data

The arithmetic intensity (FLOPs per byte) rises from 1 to ~n. Level 3 ops are the only ones that hide memory latency well — that is why GEMM is the king kernel.

---

## Exercise 3.1 — Level 1: AXPY and Dot Product

**Difficulty**: ★

### Problem

Implement two kernels:

```c
/* y = a*x + y */
void axpy(int n, float a, const float *x, float *y);

/* return sum(x[i] * y[i]) */
float dot(int n, const float *x, const float *y);
```

Time both at $n = 10^6$ in a tight loop (10000 iterations). Compute GFLOPS = (loop count × $2n$) / time. You should see 1–4 GFLOPS — bandwidth-bound, well below CPU peak.

---

## Exercise 3.2 — Level 2: GEMV

**Difficulty**: ★★

### Problem

Implement matrix-vector multiply:

```c
/* y = alpha * A * x + beta * y, A is M x N row-major */
void gemv(int M, int N, float alpha, const float *A,
          const float *x, float beta, float *y) {
    for (int i = 0; i < M; i++) {
        float acc = 0;
        for (int j = 0; j < N; j++) acc += A[i * N + j] * x[j];
        y[i] = alpha * acc + beta * y[i];
    }
}
```

Time at $M = N = 4096$. Compute GFLOPS — should be 2–6, slightly higher than Level 1 because each load of $x[j]$ is reused $M$ times.

---

## Exercise 3.3 — Vectorized AXPY

**Difficulty**: ★★

Use SSE/AVX intrinsics to process 8 floats per iteration:

```c
#include <immintrin.h>

void axpy_avx(int n, float a, const float *x, float *y) {
    __m256 va = _mm256_set1_ps(a);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, vx, vy));
    }
    for (; i < n; i++) y[i] = a * x[i] + y[i];   // tail
}
```

Compare against your scalar AXPY. Speedup should be 4–8× depending on memory bandwidth. For pure compute kernels (with arrays in cache) AVX gets close to 8×; for big arrays it is bandwidth-limited and the gap shrinks.

---

## Exercise 3.4 — Activation Vector Op — Bonus

**Difficulty**: ★★

Implement `void relu_inplace(float *x, int n)` and `void sigmoid_inplace(float *x, int n)`. Time both. The ReLU is bandwidth-bound (one compare + one write per element). The sigmoid involves `expf`, which is compute-bound — ~50 ns per element on most CPUs without vectorization.

The take-home: cheap activations are the same speed as a memory copy; expensive ones add real cost. This is why "fused" kernels (Lesson 36 in CUDA) wrap activation around the preceding matmul instead of running it as a separate pass.
