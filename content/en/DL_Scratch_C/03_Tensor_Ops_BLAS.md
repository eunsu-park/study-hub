# 03. Tensor Ops and BLAS

**Previous**: [Memory Layout and Strides](./02_Memory_Layout_and_Strides.md) | **Next**: [Optimized Matmul](./04_Optimized_Matmul.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement element-wise operations (add, mul, ReLU, GELU, SiLU) over flat float arrays
2. Implement reduction operations (sum, max, mean) along arbitrary axes
3. Write a correct naive matrix multiplication and analyze its FLOPs
4. Call OpenBLAS `cblas_sgemm` to perform high-performance GEMM
5. Benchmark naive vs. OpenBLAS matmul and explain the performance gap

---

## 1. Element-Wise Operations

All element-wise ops iterate over `numel` elements and apply a scalar function:

```c
// ops.h
#pragma once
#include "tensor.h"

// Element-wise binary ops (in-place: out = a OP b, broadcast not handled yet)
void tensor_add(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_mul(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_sub(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_div(Tensor *out, const Tensor *a, const Tensor *b);

// Scalar ops
void tensor_add_scalar(Tensor *out, const Tensor *a, float scalar);
void tensor_mul_scalar(Tensor *out, const Tensor *a, float scalar);

// Activation functions
void tensor_relu   (Tensor *out, const Tensor *x);
void tensor_gelu   (Tensor *out, const Tensor *x);  // GPT-2 activation
void tensor_silu   (Tensor *out, const Tensor *x);  // Llama activation (sigmoid * x)
void tensor_sigmoid(Tensor *out, const Tensor *x);

// Reductions
float  tensor_sum  (const Tensor *x);
float  tensor_max  (const Tensor *x);
float  tensor_mean (const Tensor *x);
Tensor *tensor_sum_axis (const Tensor *x, int axis, bool keepdim);

// Matrix multiplication
void tensor_matmul (Tensor *out, const Tensor *a, const Tensor *b);       // naive
void tensor_matmul_blas(Tensor *out, const Tensor *a, const Tensor *b);   // OpenBLAS
```

### Implementation: Element-Wise

```c
// ops.c
#include "ops.h"
#include <math.h>

void tensor_add(Tensor *out, const Tensor *a, const Tensor *b) {
    assert(a->numel == b->numel && a->numel == out->numel);
    for (size_t i = 0; i < a->numel; i++)
        out->data[i] = a->data[i] + b->data[i];
}

void tensor_mul(Tensor *out, const Tensor *a, const Tensor *b) {
    assert(a->numel == b->numel && a->numel == out->numel);
    for (size_t i = 0; i < a->numel; i++)
        out->data[i] = a->data[i] * b->data[i];
}

void tensor_relu(Tensor *out, const Tensor *x) {
    for (size_t i = 0; i < x->numel; i++)
        out->data[i] = x->data[i] > 0.0f ? x->data[i] : 0.0f;
}
```

### GELU and SiLU

```c
#include <math.h>

// GELU: used in GPT-2 FFN
// Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
void tensor_gelu(Tensor *out, const Tensor *x) {
    const float sqrt2_over_pi = 0.7978845608f;  // sqrt(2/π)
    const float coef = 0.044715f;
    for (size_t i = 0; i < x->numel; i++) {
        float v = x->data[i];
        float inner = sqrt2_over_pi * (v + coef * v * v * v);
        out->data[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

// SiLU (Swish): used in Llama FFN
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
void tensor_silu(Tensor *out, const Tensor *x) {
    for (size_t i = 0; i < x->numel; i++) {
        float v = x->data[i];
        out->data[i] = v / (1.0f + expf(-v));
    }
}
```

---

## 2. Reductions

Reductions collapse elements along all or one dimension.

```c
float tensor_sum(const Tensor *x) {
    float acc = 0.0f;
    for (size_t i = 0; i < x->numel; i++) acc += x->data[i];
    return acc;
}

float tensor_max(const Tensor *x) {
    assert(x->numel > 0);
    float m = x->data[0];
    for (size_t i = 1; i < x->numel; i++)
        if (x->data[i] > m) m = x->data[i];
    return m;
}

float tensor_mean(const Tensor *x) {
    return tensor_sum(x) / (float)x->numel;
}
```

### Axis Reduction

Reducing along axis `ax` for a 2D tensor of shape `[M, N]`:

| axis | operation | output shape |
|------|-----------|-------------|
| 0 | sum over rows | `[N]` |
| 1 | sum over columns | `[M]` |

```c
// 2D specialization for axis reduction (generalize in later lessons)
Tensor *tensor_sum_axis2d(const Tensor *x, int axis) {
    assert(x->ndim == 2);
    size_t M = x->shape[0], N = x->shape[1];

    if (axis == 0) {
        size_t out_shape[] = {N};
        Tensor *out = tensor_zeros(1, out_shape);
        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < N; j++)
                out->data[j] += x->data[i * N + j];
        return out;
    } else {  // axis == 1
        size_t out_shape[] = {M};
        Tensor *out = tensor_zeros(1, out_shape);
        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < N; j++)
                out->data[i] += x->data[i * N + j];
        return out;
    }
}
```

---

## 3. Naive Matrix Multiplication

Matrix multiplication `C = A * B` for shapes `A[M, K]`, `B[K, N]`, `C[M, N]`:

```
C[i][j] = sum_{k=0}^{K-1} A[i][k] * B[k][j]
```

```c
// Naive 3-loop matmul — correct but slow
void tensor_matmul_naive(Tensor *C, const Tensor *A, const Tensor *B) {
    assert(A->ndim == 2 && B->ndim == 2 && C->ndim == 2);
    size_t M = A->shape[0], K = A->shape[1], N = B->shape[1];
    assert(B->shape[0] == K && C->shape[0] == M && C->shape[1] == N);

    // Zero out C
    memset(C->data, 0, M * N * sizeof(float));

    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            for (size_t k = 0; k < K; k++)
                C->data[i * N + j] += A->data[i * K + k] * B->data[k * N + j];
}
```

### FLOPs Analysis

For `C = A[M,K] * B[K,N]`:
- Each `C[i,j]` requires `K` multiply-adds
- Total FLOPs = `2 * M * K * N` (one multiply + one add per inner product step)

For GPT-2's attention Q projection `[batch * seq, d_model] * [d_model, d_head]`:
- `M = 512`, `K = 768`, `N = 64`
- FLOPs = `2 * 512 * 768 * 64 ≈ 50M` per batch item per head

---

## 4. BLAS: Basic Linear Algebra Subprograms

OpenBLAS provides highly optimized GEMM (General Matrix-Matrix Multiply) using SIMD and multi-threading. The standard interface is `cblas_sgemm`.

### CBLAS SGEMM Signature

```c
void cblas_sgemm(
    CBLAS_LAYOUT    layout,    // CblasRowMajor or CblasColMajor
    CBLAS_TRANSPOSE TransA,    // CblasNoTrans or CblasTrans
    CBLAS_TRANSPOSE TransB,
    int             M,         // rows of A and C
    int             N,         // cols of B and C
    int             K,         // cols of A, rows of B
    float           alpha,     // scalar multiplier: C = alpha*A*B + beta*C
    const float    *A,
    int             lda,       // leading dimension of A (stride between rows)
    const float    *B,
    int             ldb,
    float           beta,      // scalar for C accumulation
    float          *C,
    int             ldc
);
```

### Wrapper

```c
#include <cblas.h>

void tensor_matmul_blas(Tensor *C, const Tensor *A, const Tensor *B) {
    assert(A->ndim == 2 && B->ndim == 2 && C->ndim == 2);
    int M = (int)A->shape[0];
    int K = (int)A->shape[1];
    int N = (int)B->shape[1];
    assert((int)B->shape[0] == K);
    assert((int)C->shape[0] == M && (int)C->shape[1] == N);

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        1.0f,           // alpha
        A->data, K,     // A, lda = K (row stride for row-major)
        B->data, N,     // B, ldb = N
        0.0f,           // beta (overwrite C)
        C->data, N      // C, ldc = N
    );
}
```

> **Leading Dimension**: In row-major, `lda` is the number of columns in the physical storage (which may differ from the logical `K` if the tensor is a view with non-trivial strides). For contiguous tensors, `lda = K`.

---

## 5. Benchmark: Naive vs. OpenBLAS

```c
// benchmark_matmul.c
#include <time.h>
#include <stdio.h>
#include "tensor.h"
#include "ops.h"

double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    int sizes[] = {128, 256, 512, 1024, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        size_t N = sizes[s];
        size_t shape[] = {N, N};
        Tensor *A = tensor_zeros(2, shape);
        Tensor *B = tensor_zeros(2, shape);
        Tensor *C = tensor_zeros(2, shape);

        // Fill with random values
        for (size_t i = 0; i < N * N; i++) {
            A->data[i] = (float)rand() / RAND_MAX;
            B->data[i] = (float)rand() / RAND_MAX;
        }

        double flops = 2.0 * N * N * N;

        // Naive matmul
        double t0 = get_time_ms();
        if (N <= 512) {  // Only for small N (naive is too slow for large)
            tensor_matmul_naive(C, A, B);
        }
        double t1 = get_time_ms();

        // BLAS matmul
        double t2 = get_time_ms();
        tensor_matmul_blas(C, A, B);
        double t3 = get_time_ms();

        double blas_gflops = flops / (t3 - t2) / 1e6;  // GFLOPs/s

        printf("N=%4zu  BLAS: %6.1f GFLOP/s  (%5.1f ms)", N, blas_gflops, t3 - t2);
        if (N <= 512)
            printf("  Naive: %5.1f ms  Speedup: %.0fx", t1 - t0, (t1 - t0) / (t3 - t2));
        printf("\n");

        tensor_free(A); tensor_free(B); tensor_free(C);
    }
    return 0;
}
```

**Typical results on a modern CPU (single-threaded OpenBLAS, AVX2)**:

```
N= 128  BLAS:  120 GFLOP/s   (0.0 ms)  Naive:   0.5 ms  Speedup:  25x
N= 256  BLAS:  180 GFLOP/s   (0.2 ms)  Naive:   3.2 ms  Speedup:  16x
N= 512  BLAS:  210 GFLOP/s   (1.3 ms)  Naive:  55.0 ms  Speedup:  42x
N=1024  BLAS:  230 GFLOP/s  (11.0 ms)
N=2048  BLAS:  240 GFLOP/s  (85.0 ms)
```

The gap is enormous because OpenBLAS uses:
1. **AVX2/AVX-512** — 8 or 16 floats per instruction
2. **Loop tiling** — keeps data in L1/L2 cache
3. **Multi-threading** — uses all cores

In L04, we implement these optimizations ourselves and close the gap.

---

## 6. FLOP/Byte Ratio (Arithmetic Intensity)

A key concept for understanding performance bottlenecks:

```
Arithmetic Intensity = FLOPs / Bytes read from memory

Naive matmul N×N:
  FLOPs  = 2 * N^3
  Bytes  = 3 * N^2 * 4  (read A, B; write C)
  AI     = 2N^3 / (12N^2) = N/6

For N=1024: AI ≈ 170 FLOPs/byte
Modern CPU: ~24 GFLOP/s (AVX2 single-thread), ~50 GB/s memory bandwidth
  → Compute-bound threshold: 24e9 / 50e9 = 0.48 FLOPs/byte
  → AI=170 >> 0.48: matmul is STRONGLY compute-bound
  → We have room to improve throughput, not just memory bandwidth
```

This is the **roofline model** — a useful framework for analyzing compute-bound vs. memory-bound operations.

---

## Key Takeaways

- Element-wise ops are trivially parallelizable; the loop body is just a scalar function
- GELU uses `tanh` (expensive); SiLU uses `sigmoid` (cheaper) — Llama's SwiGLU uses two of these
- Naive matmul is `O(N^3)` with terrible cache behavior for large N
- OpenBLAS achieves near-peak FLOP/s through SIMD + tiling + threading
- Arithmetic intensity determines whether a kernel is memory-bound or compute-bound

---

**Next**: [04. Optimized Matmul](./04_Optimized_Matmul.md) — Implement loop tiling and AVX2 intrinsics to build a SGEMM that approaches OpenBLAS performance.
