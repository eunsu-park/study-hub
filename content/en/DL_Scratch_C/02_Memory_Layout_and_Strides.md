# 02. Memory Layout and Strides

**Previous**: [Why C/C++ for Deep Learning?](./01_Why_C_for_DL.md) | **Next**: [Tensor Ops and BLAS](./03_Tensor_Ops_BLAS.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain row-major (C-order) vs. column-major (Fortran-order) memory layout
2. Calculate element addresses using stride arithmetic for N-dimensional tensors
3. Implement zero-copy views: reshape, transpose, and slice without copying data
4. Detect and handle non-contiguous tensors
5. Explain cache-line alignment and its impact on matmul performance

---

## 1. Memory Layout Fundamentals

A tensor is conceptually multi-dimensional, but physically stored as a **flat, contiguous block of floats** in memory.

### Row-Major (C-Order)

In row-major layout, the **last index varies fastest**. This is the default in C, NumPy (default), and PyTorch.

```
Matrix A (2×3):
  A[0][0] A[0][1] A[0][2]
  A[1][0] A[1][1] A[1][2]

Memory (contiguous):
  index: 0       1       2       3       4       5
  value: A[0,0]  A[0,1]  A[0,2]  A[1,0]  A[1,1]  A[1,2]
```

To access `A[i][j]`:
```
offset = i * ncols + j
       = i * stride[0] + j * stride[1]
```

For shape `[2, 3]`, strides are `[3, 1]`.

### Column-Major (Fortran-Order)

The **first index varies fastest**. Used in Fortran, MATLAB, and the underlying storage of cuBLAS.

```
Matrix A (2×3) in column-major:
  offset:  0       1       2       3       4       5
  value:   A[0,0]  A[1,0]  A[0,1]  A[1,1]  A[0,2]  A[1,2]
```

Strides for shape `[2, 3]` in column-major: `[1, 2]`.

> **Why it matters**: cuBLAS expects column-major matrices. When calling `cublasSgemm`, you often need to transpose the operation to work with row-major PyTorch/C tensors. Understanding strides lets you call cuBLAS without copying data.

### The General Rule: Strides

For a tensor of shape `[d0, d1, ..., d_{n-1}]` in **row-major**:
```
stride[n-1] = 1
stride[k]   = stride[k+1] * shape[k+1]   for k = n-2 down to 0
```

The address of element `t[i0, i1, ..., i_{n-1}]` is:
```
offset = sum(i_k * stride[k])   for k = 0..n-1
address = data + offset
```

---

## 2. Extending the Tensor Struct

```c
// tensor.h
#pragma once
#include <stddef.h>
#include <stdbool.h>

#define TENSOR_MAX_DIMS 8

typedef struct Tensor {
    float  *data;                      // Pointer to raw data (may be shared)
    size_t  shape[TENSOR_MAX_DIMS];
    size_t  strides[TENSOR_MAX_DIMS];  // In elements, not bytes
    int     ndim;
    size_t  numel;
    bool    owns_data;                 // false → view (do not free data)

    // Autograd fields (added in L05)
    struct Tensor *grad;
    void (*backward_fn)(struct Tensor *self);
    void *backward_ctx;
    bool  requires_grad;
} Tensor;

// Allocation
Tensor *tensor_zeros(int ndim, const size_t *shape);
Tensor *tensor_ones(int ndim, const size_t *shape);
Tensor *tensor_from_data(float *data, int ndim, const size_t *shape, bool owns);

// Views (zero-copy)
Tensor *tensor_view(Tensor *src, int ndim, const size_t *new_shape);
Tensor *tensor_transpose(Tensor *src, int dim0, int dim1);
Tensor *tensor_slice(Tensor *src, int dim, size_t start, size_t end);

// Properties
bool   tensor_is_contiguous(const Tensor *t);
Tensor *tensor_contiguous(Tensor *t);   // returns contiguous copy if needed

void   tensor_free(Tensor *t);
void   tensor_print(const Tensor *t, const char *name);
```

---

## 3. Implementing Views

### Reshape (View)

A reshape is valid only when the tensor is contiguous (strides are standard row-major). It creates a new `Tensor` header pointing to the *same* data.

```c
Tensor *tensor_view(Tensor *src, int new_ndim, const size_t *new_shape) {
    // Verify total element count matches
    size_t numel = 1;
    for (int i = 0; i < new_ndim; i++) numel *= new_shape[i];
    assert(numel == src->numel && "view: element count mismatch");
    assert(tensor_is_contiguous(src) && "view: source must be contiguous");

    Tensor *t     = (Tensor *)calloc(1, sizeof(Tensor));
    t->data       = src->data;    // Shared pointer — no copy!
    t->ndim       = new_ndim;
    t->numel      = numel;
    t->owns_data  = false;        // Do not free on tensor_free(t)

    for (int i = 0; i < new_ndim; i++) t->shape[i] = new_shape[i];

    // Compute row-major strides for new shape
    t->strides[new_ndim - 1] = 1;
    for (int i = new_ndim - 2; i >= 0; i--)
        t->strides[i] = t->strides[i + 1] * new_shape[i + 1];

    return t;
}
```

### Transpose

A transpose swaps two dimensions' strides — no data movement.

```c
Tensor *tensor_transpose(Tensor *src, int dim0, int dim1) {
    assert(dim0 < src->ndim && dim1 < src->ndim);

    Tensor *t = (Tensor *)calloc(1, sizeof(Tensor));
    t->data      = src->data;
    t->ndim      = src->ndim;
    t->numel     = src->numel;
    t->owns_data = false;

    memcpy(t->shape,   src->shape,   src->ndim * sizeof(size_t));
    memcpy(t->strides, src->strides, src->ndim * sizeof(size_t));

    // Swap shape and stride for the two dimensions
    size_t tmp_shape  = t->shape[dim0];
    t->shape[dim0]    = t->shape[dim1];
    t->shape[dim1]    = tmp_shape;

    size_t tmp_stride  = t->strides[dim0];
    t->strides[dim0]   = t->strides[dim1];
    t->strides[dim1]   = tmp_stride;

    return t;
}
```

**Example**: Transposing a `[4, 3]` matrix
```
Original:  shape=[4,3], strides=[3,1]
Transposed: shape=[3,4], strides=[1,3]

Access T[i][j] = data[ i*1 + j*3 ] = data[ j*3 + i ]
               = A[j][i]  ✓
```

---

## 4. Contiguity Check

A tensor is **contiguous** when its strides match the standard row-major layout for its shape.

```c
bool tensor_is_contiguous(const Tensor *t) {
    size_t expected = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        if (t->strides[i] != expected) return false;
        expected *= t->shape[i];
    }
    return true;
}
```

Non-contiguous tensors (e.g., transposed) cannot be reshaped via `view`. They must be made contiguous first by copying data into a new buffer with standard strides.

```c
Tensor *tensor_contiguous(Tensor *t) {
    if (tensor_is_contiguous(t)) return t;

    Tensor *out = tensor_zeros(t->ndim, t->shape);
    // Iterate all elements using stride-based indexing
    size_t coords[TENSOR_MAX_DIMS] = {0};
    for (size_t flat = 0; flat < t->numel; flat++) {
        // Compute source offset from coords + strides
        size_t src_offset = 0;
        for (int d = 0; d < t->ndim; d++)
            src_offset += coords[d] * t->strides[d];

        out->data[flat] = t->data[src_offset];

        // Increment coords (right-to-left)
        for (int d = t->ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < t->shape[d]) break;
            coords[d] = 0;
        }
    }
    return out;
}
```

---

## 5. Cache-Line Alignment

Modern CPUs load data in **64-byte cache lines**. A `float` is 4 bytes, so one cache line holds 16 floats.

```
Matrix A [1024 x 1024]:
  Row access (A[i][j], A[i][j+1], ...): contiguous → 1 cache miss per 16 elements
  Col access (A[0][j], A[1][j], ...):   stride=1024 → 1 cache miss per element
```

For matmul `C = A * B`, accessing `B` column-by-column causes **cache thrashing**. The fix is:
1. Transpose `B` before multiplication (B^T is accessed row-by-row)
2. Or use **tiling** (access a block of B that fits in L1 cache — covered in L04)

**Memory alignment**: Allocate data on a 64-byte boundary for SIMD efficiency:

```c
#include <stdlib.h>

float *alloc_aligned(size_t numel) {
    void *ptr = NULL;
    // posix_memalign guarantees alignment to given boundary
    if (posix_memalign(&ptr, 64, numel * sizeof(float)) != 0) {
        fprintf(stderr, "alloc_aligned: allocation failed\n");
        exit(1);
    }
    return (float *)ptr;
}
```

---

## 6. Practical: NCHW vs NHWC

For convolution, two common 4D layouts exist:

| Layout | Shape | Usage |
|--------|-------|-------|
| **NCHW** | [batch, channels, height, width] | PyTorch default, CUDA-preferred |
| **NHWC** | [batch, height, width, channels] | TensorFlow default, ARM-preferred |

```
NCHW strides for [N, C, H, W]:
  stride[0] = C * H * W   (step over one image)
  stride[1] = H * W       (step over one channel)
  stride[2] = W           (step over one row)
  stride[3] = 1           (step over one pixel)

NHWC strides for [N, H, W, C]:
  stride[0] = H * W * C
  stride[1] = W * C
  stride[2] = C
  stride[3] = 1
```

In C, you can switch between layouts by **changing strides** — no data copy needed — but your conv kernel must use the right stride formula to access neighbors correctly.

---

## 7. Hands-On Exercises

### Exercise 1: Stride Calculation

Given tensor `t` of shape `[3, 4, 5]` (row-major), compute:
- `t.strides` (answer: `[20, 5, 1]`)
- Flat index of `t[2][1][3]` (answer: `2*20 + 1*5 + 3*1 = 48`)

### Exercise 2: Transpose and Verify

```c
// Create 3×4 matrix, fill with sequential values
// Transpose to get 4×3 matrix
// Verify that transposed[i][j] == original[j][i]
```

### Exercise 3: Non-contiguous View

```c
// Take a [6, 6] matrix
// Slice rows 1..4 (a view, not contiguous in full sense)
// Make it contiguous and verify the data is correct
```

---

## Key Takeaways

- Strides are the key abstraction enabling zero-copy reshape, transpose, and slice
- Row-major (C-order): strides `[d1*d2*...*dn, d2*...*dn, ..., 1]`
- A view shares the underlying data pointer — only the header (shape, strides) changes
- Non-contiguous tensors (from transpose or slice) require a copy before reshape
- Cache-line alignment (64 bytes) and contiguous access patterns are critical for SIMD performance

---

**Next**: [03. Tensor Ops and BLAS](./03_Tensor_Ops_BLAS.md) — Implement element-wise operations, reductions, and a naive matmul, then benchmark against OpenBLAS.
