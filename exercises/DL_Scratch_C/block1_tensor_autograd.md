# Block 1 — Tensors & Autograd (L01–L07)

Prerequisites: L01 (tensor layout), L02 (broadcasting), L03 (softmax), L04 (autograd basics), L05 (arena allocator), L06 (matmul), L07 (finite differences).

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

---

## Exercise 1.1 — `matrix_add_inplace` with Broadcasting

**Difficulty**: ★

### Problem

Implement `matrix_add_inplace(float *A, const float *b, int N, int M)` that adds a column vector `b` of shape `[N, 1]` to each column of matrix `A` of shape `[N, M]`, in-place.

Rules:
- `A` is stored row-major: element `(i, j)` is at `A[i*M + j]`.
- `b` has `N` elements; `b[i]` is added to the entire `i`-th row of `A`.
- Do not allocate any extra memory.

### Starter Code

```c
#include <stdio.h>
#include <stdlib.h>

/* Add column vector b[N] to every column of A[N][M] in-place. */
void matrix_add_inplace(float *A, const float *b, int N, int M) {
    /* TODO: iterate over rows and columns; add b[i] to A[i*M + j] */
}

/* ---- test harness ---- */
int main(void) {
    /* A = [[1,2,3],[4,5,6]], b = [[10],[20]] */
    float A[2][3] = {{1,2,3},{4,5,6}};
    float b[2]    = {10, 20};

    matrix_add_inplace((float *)A, b, 2, 3);

    /* Expected: [[11,12,13],[24,25,26]] */
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++)
            printf("%.1f ", A[i][j]);
        printf("\n");
    }
    return 0;
}
```

### Test Cases

| Input A | Input b | Expected A after call |
|---------|---------|----------------------|
| `[[1,2,3],[4,5,6]]` | `[10,20]` | `[[11,12,13],[24,25,26]]` |
| `[[0]]` | `[5]` | `[[5]]` |
| `[[1,1],[1,1],[1,1]]` | `[1,2,3]` | `[[2,2],[3,3],[4,4]]` |

### Hints

1. Two nested loops suffice: outer over `i` (rows), inner over `j` (cols).
2. The broadcast value for row `i` is simply `b[i]`.
3. Index arithmetic: element `(i,j)` of row-major A is `A[i*M + j]`.

### Solution Approach

The function is a simple double loop. The key insight is that broadcasting `[N,1]` onto `[N,M]` means every element in row `i` receives the same addend `b[i]`. No pointer tricks required — just write the index formula carefully.

---

## Exercise 1.2 — `softmax_2d` Row-Wise with Numerical Stability

**Difficulty**: ★★

### Problem

Implement `softmax_2d(float *X, int N, int M)` that applies softmax to each row of an `[N, M]` matrix in-place.

Requirements:
- Use the **max-subtraction** trick to avoid overflow: subtract `max(row)` before computing `exp`.
- Normalize each row so it sums to 1.
- Operate in-place (overwrite `X`).

### Starter Code

```c
#include <stdio.h>
#include <math.h>
#include <float.h>

void softmax_2d(float *X, int N, int M) {
    for (int i = 0; i < N; i++) {
        float *row = X + i * M;

        /* Step 1: find row maximum */
        float max_val = -FLT_MAX;
        /* TODO */

        /* Step 2: subtract max, exponentiate */
        /* TODO */

        /* Step 3: sum of exps */
        float sum = 0.0f;
        /* TODO */

        /* Step 4: divide by sum */
        /* TODO */
    }
}

/* ---- test harness ---- */
int main(void) {
    float X[2][3] = {{1.0f, 2.0f, 3.0f}, {1.0f, 1.0f, 1.0f}};
    softmax_2d((float *)X, 2, 3);

    /* Row 0 expected ≈ [0.0900, 0.2447, 0.6652] */
    /* Row 1 expected = [0.3333, 0.3333, 0.3333] */
    for (int i = 0; i < 2; i++) {
        float s = 0.0f;
        for (int j = 0; j < 3; j++) {
            printf("%.4f ", X[i][j]);
            s += X[i][j];
        }
        printf("  (sum=%.4f)\n", s);
    }
    return 0;
}
```

### Test Cases

| Input row | Expected output |
|-----------|----------------|
| `[1, 2, 3]` | `[0.0900, 0.2447, 0.6652]` |
| `[1, 1, 1]` | `[0.3333, 0.3333, 0.3333]` |
| `[0, 0, 1000]` | `[~0, ~0, ~1]` (must not produce NaN/Inf) |
| `[-1000, -1000, -1000]` | `[0.3333, 0.3333, 0.3333]` (must not underflow to 0/0) |

### Hints

1. If you skip the max subtraction, `exp(1000)` will overflow to `Inf`.
2. After subtracting the max, the largest value in the row becomes `exp(0) = 1`.
3. The sum can never be 0 because at least one element is `exp(0) = 1`.

### Solution Approach

Three passes over each row: (1) find max, (2) exponentiate in-place after subtracting max, (3) accumulate sum and divide. This is the standard numerically stable softmax. Total work is O(N*M) time, O(1) extra space.

---

## Exercise 1.3 — Extend Autograd: `relu` and `sigmoid`

**Difficulty**: ★★

### Problem

You have a minimal autograd engine with `Tensor` structs, `backward_fn` pointers, and a topological sort. Extend it to support two new ops:

1. `tensor_relu(Tensor *a)` — element-wise max(0, x), with correct backward.
2. `tensor_sigmoid(Tensor *a)` — element-wise 1/(1+exp(-x)), with correct backward.

For each op you must:
- Compute the forward value.
- Register a `backward_fn` that propagates `grad` back to the input.

### Starter Code

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_NODES 256

typedef struct Tensor {
    float  *data;
    float  *grad;
    int     numel;
    void  (*backward_fn)(struct Tensor *self);
    struct Tensor *inputs[2];
} Tensor;

Tensor *tensor_new(int numel) {
    Tensor *t  = calloc(1, sizeof(Tensor));
    t->data    = calloc(numel, sizeof(float));
    t->grad    = calloc(numel, sizeof(float));
    t->numel   = numel;
    return t;
}

/* --- existing ops (provided) --- */
static void add_backward(Tensor *self) {
    for (int i = 0; i < self->numel; i++) {
        self->inputs[0]->grad[i] += self->grad[i];
        self->inputs[1]->grad[i] += self->grad[i];
    }
}
Tensor *tensor_add(Tensor *a, Tensor *b) {
    Tensor *out = tensor_new(a->numel);
    for (int i = 0; i < a->numel; i++) out->data[i] = a->data[i] + b->data[i];
    out->backward_fn = add_backward;
    out->inputs[0] = a; out->inputs[1] = b;
    return out;
}

/* --- YOUR WORK --- */

static void relu_backward(Tensor *self) {
    /* TODO: grad flows through only where forward output > 0 */
}

Tensor *tensor_relu(Tensor *a) {
    Tensor *out = tensor_new(a->numel);
    /* TODO: forward pass */
    out->backward_fn = relu_backward;
    out->inputs[0] = a;
    return out;
}

static void sigmoid_backward(Tensor *self) {
    /* TODO: d_sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
             but you already have sigmoid(x) stored in self->data */
}

Tensor *tensor_sigmoid(Tensor *a) {
    Tensor *out = tensor_new(a->numel);
    /* TODO: forward pass */
    out->backward_fn = sigmoid_backward;
    out->inputs[0] = a;
    return out;
}

/* minimal backward (assumes linear chain, no branching) */
void backward(Tensor *root) {
    root->grad[0] = 1.0f;
    /* walk backwards through inputs chain */
    Tensor *cur = root;
    while (cur) {
        if (cur->backward_fn) cur->backward_fn(cur);
        cur = cur->inputs[0];
    }
}

int main(void) {
    /* Test relu */
    Tensor *x = tensor_new(4);
    x->data[0] = -2; x->data[1] = -0.5f; x->data[2] = 0; x->data[3] = 3;
    Tensor *r = tensor_relu(x);
    backward(r);
    /* Expected r->data: [0, 0, 0, 3] */
    /* Expected x->grad: [0, 0, 0, 1] */
    printf("relu forward:  %.1f %.1f %.1f %.1f\n",
           r->data[0], r->data[1], r->data[2], r->data[3]);
    printf("relu grad:     %.1f %.1f %.1f %.1f\n",
           x->grad[0], x->grad[1], x->grad[2], x->grad[3]);

    /* Test sigmoid */
    Tensor *y  = tensor_new(1); y->data[0] = 0.0f;
    Tensor *s  = tensor_sigmoid(y);
    backward(s);
    /* Expected s->data[0] = 0.5, y->grad[0] = 0.25 */
    printf("sigmoid(0) = %.4f  grad = %.4f\n", s->data[0], y->grad[0]);
    return 0;
}
```

### Test Cases

| Op | Input | Expected forward | Expected grad (dL/dx, L=output[0]) |
|----|-------|------------------|------------------------------------|
| relu | [-2, -0.5, 0, 3] | [0, 0, 0, 3] | [0, 0, 0, 1] |
| sigmoid | [0] | [0.5] | [0.25] |
| sigmoid | [100] | [~1.0] | [~0] |

### Hints

1. ReLU backward: the gate is `self->data[i] > 0`. If the output was positive, gradient flows; otherwise it is zeroed.
2. Sigmoid backward: the derivative is `s*(1-s)` where `s = self->data[i]` (already computed during forward).
3. Remember to **accumulate** (`+=`) gradients, not overwrite them — a tensor may receive gradient from multiple consumers.

### Solution Approach

Both backward functions are one-liners per element. For ReLU: `input->grad[i] += (self->data[i] > 0) ? self->grad[i] : 0`. For sigmoid: `input->grad[i] += self->data[i] * (1 - self->data[i]) * self->grad[i]`. The trick is that sigmoid's derivative reuses the already-computed output value.

---

## Exercise 1.4 — Arena Allocator with `arena_reset()`

**Difficulty**: ★★

### Problem

Implement a bump-pointer arena allocator:

```c
typedef struct { char *buf; size_t cap; size_t used; } Arena;

Arena  arena_create(size_t capacity);
void  *arena_alloc (Arena *a, size_t bytes);
void   arena_reset (Arena *a);   /* resets used=0, does NOT free buf */
void   arena_destroy(Arena *a);
```

Rules:
- `arena_alloc` must align to 8 bytes (round up `bytes` to next multiple of 8).
- `arena_alloc` returns `NULL` if there is not enough space (no partial allocation).
- `arena_reset` sets `used = 0` so the memory can be reused from the start. It does **not** call `free`.
- `arena_destroy` calls `free` exactly once on the backing buffer.

### Starter Code

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

typedef struct {
    char   *buf;
    size_t  cap;
    size_t  used;
} Arena;

Arena arena_create(size_t capacity) {
    Arena a;
    /* TODO */
    return a;
}

void *arena_alloc(Arena *a, size_t bytes) {
    /* TODO: align bytes up to 8, check space, bump pointer */
    return NULL;
}

void arena_reset(Arena *a) {
    /* TODO */
}

void arena_destroy(Arena *a) {
    /* TODO */
}

int main(void) {
    Arena a = arena_create(1024);

    int   *p1 = arena_alloc(&a, sizeof(int));
    float *p2 = arena_alloc(&a, sizeof(float) * 3);
    assert(p1 != NULL && p2 != NULL);

    *p1 = 42;
    p2[0] = 1.0f; p2[1] = 2.0f; p2[2] = 3.0f;
    printf("p1=%d  p2=[%.1f %.1f %.1f]\n", *p1, p2[0], p2[1], p2[2]);

    /* After reset, pointers obtained before reset are stale — don't use them */
    arena_reset(&a);
    printf("used after reset: %zu (expected 0)\n", a.used);

    int *p3 = arena_alloc(&a, sizeof(int));
    assert(p3 != NULL);
    *p3 = 99;
    printf("p3=%d\n", *p3);

    /* Overflow test */
    void *big = arena_alloc(&a, 2048);
    assert(big == NULL);
    printf("overflow returns NULL: OK\n");

    arena_destroy(&a);
    return 0;
}
```

### Test Cases

- After `arena_create(1024)`, `a.used == 0` and `a.cap == 1024`.
- Allocating `sizeof(int)=4` bytes bumps `used` to 8 (aligned).
- After `arena_reset`, `used == 0`.
- Allocating more than `cap` returns `NULL`.
- `arena_destroy` must not crash (use valgrind to verify no leak).

### Hints

1. Alignment: `aligned = (bytes + 7) & ~(size_t)7` rounds up to the next multiple of 8.
2. After alignment, check `a->used + aligned <= a->cap` before bumping.
3. Save the pointer `a->buf + a->used` before bumping `used`.
4. `arena_reset` is literally one line.

### Solution Approach

Bump-pointer allocation is the simplest possible allocator: keep a "high-water mark" (`used`) and advance it on each allocation. Alignment ensures all returned pointers are 8-byte aligned, which satisfies most platform requirements. Reset just zeroes the watermark — the backing buffer is untouched, making reset O(1) regardless of how many allocations were made.

---

## Exercise 1.5 — Verify `matmul_backward` via Finite Differences

**Difficulty**: ★★★

### Problem

You have a forward function `matmul(A, B, C, M, K, N)` that computes `C = A @ B` (A is M×K, B is K×N, C is M×N).

Implement `matmul_backward` and verify it with the **finite-difference gradient check**:

For each scalar input element `A[i][j]`, numerically approximate `∂L/∂A[i][j]` as:

```
(L(A + ε*e_ij) - L(A - ε*e_ij)) / (2ε)
```

where `L = sum(C)` (simple scalar loss), `ε = 1e-4`.

Your analytical gradient must match the numerical gradient within tolerance `1e-3`.

### Starter Code

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* C[M][N] = A[M][K] @ B[K][N] */
void matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    memset(C, 0, M * N * sizeof(float));
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++)
                C[i*N + j] += A[i*K + k] * B[k*N + j];
}

/*
 * dA[M][K] += dC[M][N] @ B^T[N][K]
 * dB[K][N] += A^T[K][M] @ dC[M][N]
 *
 * For loss L = sum(C), dC is an all-ones matrix.
 */
void matmul_backward(
    const float *A,  const float *B,
    float *dA, float *dB,          /* accumulate here */
    const float *dC,               /* upstream gradient */
    int M, int K, int N)
{
    /* TODO: compute dA and dB */
}

float sum_all(const float *x, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i];
    return s;
}

int main(void) {
    int M=3, K=4, N=2;
    float A[3*4], B[4*2], C[3*2];
    float dA_anal[3*4], dB_anal[4*2];
    float dA_num [3*4], dB_num [4*2];
    float dC[3*2];

    /* Initialize with small values */
    for (int i = 0; i < M*K; i++) A[i] = (float)(i+1)*0.1f;
    for (int i = 0; i < K*N; i++) B[i] = (float)(i+1)*0.05f;
    for (int i = 0; i < M*N; i++) dC[i] = 1.0f; /* dL/dC = 1 for L=sum(C) */

    /* Analytical gradient */
    memset(dA_anal, 0, sizeof(dA_anal));
    memset(dB_anal, 0, sizeof(dB_anal));
    matmul_backward(A, B, dA_anal, dB_anal, dC, M, K, N);

    /* Numerical gradient for A */
    float eps = 1e-4f;
    for (int i = 0; i < M*K; i++) {
        float orig = A[i];
        A[i] = orig + eps; matmul(A, B, C, M, K, N); float Lp = sum_all(C, M*N);
        A[i] = orig - eps; matmul(A, B, C, M, K, N); float Lm = sum_all(C, M*N);
        A[i] = orig;
        dA_num[i] = (Lp - Lm) / (2.0f * eps);
    }

    /* TODO: similarly compute dB_num for B */

    /* Check */
    int ok = 1;
    for (int i = 0; i < M*K; i++) {
        float diff = fabsf(dA_anal[i] - dA_num[i]);
        if (diff > 1e-3f) { printf("FAIL dA[%d]: anal=%.6f num=%.6f\n", i, dA_anal[i], dA_num[i]); ok=0; }
    }
    /* TODO: check dB similarly */
    if (ok) printf("PASS: all gradients match within tolerance\n");
    return 0;
}
```

### Test Cases

- For `L = sum(A @ B)` with shapes (3,4)×(4,2):
  - `dA[i][k] = sum_j(dC[i][j] * B[k][j])` — this is `dC @ B^T`.
  - `dB[k][j] = sum_i(A[i][k] * dC[i][j])` — this is `A^T @ dC`.
- All analytical values must satisfy `|anal - num| < 1e-3`.

### Hints

1. For `L = sum(C)`, `dC` is an all-ones matrix of shape M×N.
2. The analytical formula `dA = dC @ B^T` means: `dA[i][k] += dC[i][j] * B[k][j]` for all j.
3. Similarly `dB = A^T @ dC` means: `dB[k][j] += A[i][k] * dC[i][j]` for all i.
4. Finite differences are slow but reliable — use them as a correctness oracle, not for production.

### Solution Approach

Write two triple loops, mirroring the forward loop. For dA the contraction is over `j` (the output column dimension); for dB the contraction is over `i` (the output row dimension). After you have working code, compare against the numerical approximation element-by-element and ensure all errors are below 1e-3. If they are not, check whether you are accumulating (`+=`) correctly and that index strides match.
