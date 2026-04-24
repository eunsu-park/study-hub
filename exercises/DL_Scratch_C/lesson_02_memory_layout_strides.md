# Lesson 2 — Memory Layout and Strides (per-lesson exercise)

Prerequisites: L01 (why C for DL), basic C pointer arithmetic.

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

A tensor in memory is just a contiguous block of floats plus metadata describing how indices map to offsets. The metadata — shape and strides — is what lets one underlying buffer represent transposes, slices, and broadcasts without copying data.

This is the unsung infrastructure under PyTorch's `view`, `permute`, and `broadcast_to`. Implement a tiny tensor struct and the pattern is yours forever.

---

## Exercise 2.1 — Tensor Struct and Indexing

**Difficulty**: ★★

### Problem

Define:

```c
typedef struct {
    float *data;
    int shape[4];
    int strides[4];   /* strides[i] = elements (not bytes) to advance index i by 1 */
    int ndim;
} Tensor;
```

Implement `float* tensor_at(Tensor *t, int i, int j, int k, int l)` that returns a pointer to element $(i, j, k, l)$ using:

$$\text{offset} = i \cdot s_0 + j \cdot s_1 + k \cdot s_2 + l \cdot s_3$$

For lower-rank tensors, ignore the trailing indices.

### Starter

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float *data;
    int shape[4];
    int strides[4];
    int ndim;
} Tensor;

float *tensor_at(Tensor *t, int i, int j, int k, int l) {
    int offset = 0;
    int idx[4] = {i, j, k, l};
    for (int d = 0; d < t->ndim; d++) offset += idx[d] * t->strides[d];
    return &t->data[offset];
}

int main(void) {
    /* 2x3 row-major tensor */
    float buf[6] = {1, 2, 3, 4, 5, 6};
    Tensor t = {.data = buf, .shape = {2, 3, 0, 0},
                .strides = {3, 1, 0, 0}, .ndim = 2};

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) printf("%.1f ", *tensor_at(&t, i, j, 0, 0));
        printf("\n");
    }
    /* Expected: 1 2 3 / 4 5 6 */
    return 0;
}
```

---

## Exercise 2.2 — Zero-Copy Transpose

**Difficulty**: ★★

### Problem

Transposing a 2D tensor is trivially zero-copy: swap shape and strides:

```c
Tensor tensor_transpose_2d(Tensor t) {
    Tensor out = t;
    out.shape[0] = t.shape[1];
    out.shape[1] = t.shape[0];
    out.strides[0] = t.strides[1];
    out.strides[1] = t.strides[0];
    return out;
}
```

After this call, `tensor_at(&out, i, j)` returns `&buf[i * old_s1 + j * old_s0]` — which is exactly the transposed-element location.

Demonstrate by transposing a 2x3 tensor and reading values; the underlying buffer has not been touched. Discuss when this is faster than copying (almost always for one-shot reads, sometimes slower for follow-up matrix multiplies because the transposed view is non-contiguous).

---

## Exercise 2.3 — Broadcasting via Stride 0

**Difficulty**: ★★★

A broadcast tensor uses `stride = 0` along the broadcast dimension, so incrementing the index does not change the offset. Implement:

```c
Tensor tensor_broadcast_to(Tensor t, int new_shape[4], int new_ndim);
```

Rules: a broadcastable dimension is one of size 1 (or absent on the right of the input). Fill in `strides = 0` on broadcast dimensions; copy original strides on aligned dimensions.

Verify: `tensor_at(&broadcast, i, j)` should return the same value for all $j$ when broadcasting along axis 1.

---

## Exercise 2.4 — Contiguous-Check and Copy — Bonus

**Difficulty**: ★★

A tensor is "C-contiguous" iff `strides[d] = product(shape[d+1:])`. Implement `int is_c_contiguous(const Tensor *t)`.

Then implement `Tensor tensor_make_contiguous(const Tensor *t)` that allocates a fresh buffer and copies elements in C-order. This is the pattern that PyTorch's `.contiguous()` implements; transposes need it before a kernel that assumes contiguous layout.
