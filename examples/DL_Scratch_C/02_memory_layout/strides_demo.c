/*
 * strides_demo.c - Tensor memory layout, strides, and zero-copy views
 *
 * Demonstrates:
 *   - Row-major vs column-major indexing
 *   - Strides as the universal indexing language
 *   - Zero-copy 2D transpose via stride swap
 *   - Broadcasting via stride 0
 *   - Contiguity check
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o strides_demo strides_demo.c
 * Run:    ./strides_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DIMS 4

typedef struct {
    float *data;        /* shared underlying buffer */
    int    shape[MAX_DIMS];
    int    strides[MAX_DIMS];   /* in elements, not bytes */
    int    ndim;
} Tensor;

/* Locate element (i, j, k, l); use 0 for unused indices. */
static float *tensor_at(const Tensor *t, int i, int j, int k, int l) {
    int idx[MAX_DIMS] = {i, j, k, l};
    int off = 0;
    for (int d = 0; d < t->ndim; d++) off += idx[d] * t->strides[d];
    return &t->data[off];
}

/* Construct a 2D row-major view over an existing buffer. */
static Tensor view_2d_row_major(float *buf, int rows, int cols) {
    Tensor t = {.data = buf, .ndim = 2};
    t.shape[0]   = rows;     t.shape[1]   = cols;
    t.strides[0] = cols;     t.strides[1] = 1;
    return t;
}

/* Zero-copy 2D transpose: swap shape and stride of dims 0, 1. */
static Tensor transpose_2d(Tensor t) {
    Tensor out = t;
    int s0 = t.shape[0],   s1 = t.shape[1];
    int t0 = t.strides[0], t1 = t.strides[1];
    out.shape[0]   = s1;     out.shape[1]   = s0;
    out.strides[0] = t1;     out.strides[1] = t0;
    return out;
}

/* Broadcast a length-N vector to shape (M, N) by setting stride[0] = 0.
   Reading any (m, j) returns the same buf[j] regardless of m. */
static Tensor broadcast_row(float *buf, int N, int M) {
    Tensor t = {.data = buf, .ndim = 2};
    t.shape[0]   = M;        t.shape[1]   = N;
    t.strides[0] = 0;        t.strides[1] = 1;
    return t;
}

/* A tensor is C-contiguous when stride[i] = product(shape[i+1:]). */
static int is_c_contiguous(const Tensor *t) {
    int expected = 1;
    for (int d = t->ndim - 1; d >= 0; d--) {
        if (t->strides[d] != expected) return 0;
        expected *= t->shape[d];
    }
    return 1;
}

static void print_2d(const char *label, const Tensor *t) {
    printf("%s (shape=[%d,%d] strides=[%d,%d] contig=%s)\n",
           label, t->shape[0], t->shape[1], t->strides[0], t->strides[1],
           is_c_contiguous(t) ? "yes" : "no");
    for (int i = 0; i < t->shape[0]; i++) {
        printf("  ");
        for (int j = 0; j < t->shape[1]; j++) printf("%6.1f ", *tensor_at(t, i, j, 0, 0));
        printf("\n");
    }
}

int main(void) {
    /* 1. A real 2x3 buffer in row-major order. */
    float buf[6] = {1, 2, 3, 4, 5, 6};
    Tensor M = view_2d_row_major(buf, 2, 3);
    print_2d("M (row-major view)", &M);

    /* 2. Transpose by swapping strides — no allocation, no copy. */
    Tensor MT = transpose_2d(M);
    print_2d("M^T (zero-copy transpose)", &MT);
    /* Both views share the same buffer; mutating one is visible through the other. */
    *tensor_at(&MT, 0, 1, 0, 0) = 99;   /* writes buf[3] = 99 */
    print_2d("after MT(0,1)=99 — M shows it too", &M);

    /* Reset for the broadcast demo */
    for (int i = 0; i < 6; i++) buf[i] = (float)(i + 1);

    /* 3. Broadcast a 3-vector to a 4x3 view via stride 0 in axis 0. */
    float row[3] = {10.0f, 20.0f, 30.0f};
    Tensor B = broadcast_row(row, /*N=*/3, /*M=*/4);
    print_2d("broadcast of [10,20,30] to (4,3)", &B);

    /* 4. Contiguity matters for kernels that assume packed layout. */
    printf("\nContiguity test:\n");
    printf("  M:  %s\n", is_c_contiguous(&M)  ? "C-contiguous" : "non-contiguous");
    printf("  MT: %s\n", is_c_contiguous(&MT) ? "C-contiguous" : "non-contiguous");
    printf("  B:  %s\n", is_c_contiguous(&B)  ? "C-contiguous" : "non-contiguous");
    printf("\n(MT is the textbook 'non-contiguous' case — many kernels would need\n"
           " a `.contiguous()` copy before consuming it.)\n");

    return 0;
}
