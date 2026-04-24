# Lesson 8 — Convolution from Scratch (per-lesson exercise)

Prerequisites: L02 (memory layout), L04 (matmul).

Compile: `gcc -std=c11 -Wall -Wextra -O3 -o ex ex.c -lm`

A conv layer with `[N, C_in, H, W]` input, `[C_out, C_in, kH, kW]` kernel, stride 1, no padding, produces output of shape `[N, C_out, H - kH + 1, W - kW + 1]`. Three implementations of increasing sophistication:

1. Direct 7-loop version — the textbook formula, slow but right.
2. im2col + GEMM — turn convolution into matrix multiply; reuse your fast GEMM.
3. Implicit GEMM (skip materializing im2col) — bonus.

---

## Exercise 8.1 — Direct Convolution

**Difficulty**: ★★

### Problem

Implement `conv2d_direct(...)` with the canonical 7-loop nesting:

```
for n in 0..N
  for cout in 0..C_out
    for h in 0..H_out
      for w in 0..W_out
        acc = bias[cout]
        for cin in 0..C_in
          for kh in 0..kH
            for kw in 0..kW
              acc += in[n, cin, h+kh, w+kw] * weight[cout, cin, kh, kw]
        out[n, cout, h, w] = acc
```

### Starter

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void conv2d_direct(const float *in,
                   const float *weight,
                   const float *bias,
                   float *out,
                   int N, int C_in, int H, int W,
                   int C_out, int kH, int kW) {
    int H_out = H - kH + 1;
    int W_out = W - kW + 1;

    /* helper indices into the 4D layouts */
    #define IN(n,c,y,x)  in[((n) * C_in + (c)) * H * W + (y) * W + (x)]
    #define W4(co,ci,y,x) weight[(((co) * C_in + (ci)) * kH + (y)) * kW + (x)]
    #define OUT(n,c,y,x) out[((n) * C_out + (c)) * H_out * W_out + (y) * W_out + (x)]

    for (int n = 0; n < N; n++)
        for (int co = 0; co < C_out; co++)
            for (int h = 0; h < H_out; h++)
                for (int w = 0; w < W_out; w++) {
                    float acc = bias[co];
                    for (int ci = 0; ci < C_in; ci++)
                        for (int kh = 0; kh < kH; kh++)
                            for (int kw = 0; kw < kW; kw++)
                                acc += IN(n, ci, h + kh, w + kw) * W4(co, ci, kh, kw);
                    OUT(n, co, h, w) = acc;
                }

    #undef IN
    #undef W4
    #undef OUT
}
```

Verify on a tiny case ($N=1, C_{in}=1, H=W=4, C_{out}=1, kH=kW=3$) by hand-computing one output and comparing.

---

## Exercise 8.2 — im2col + GEMM

**Difficulty**: ★★★

### Problem

Reshape the input patches into a `[C_in*kH*kW, H_out*W_out]` matrix and the weights into a `[C_out, C_in*kH*kW]` matrix. Then convolution becomes one matrix multiply:

```
output[C_out, H_out*W_out] = weight_mat @ im2col_mat
```

Implement `im2col(in, ..., col)` and reuse your `gemm_naive` from L04. Time the result against direct convolution at `N=1, C_in=64, C_out=128, H=W=64, kH=kW=3` — im2col + GEMM is typically 5–10× faster because GEMM hits cache and SIMD harder than seven nested loops.

The trade-off: im2col blows up memory by `kH*kW`× (each input pixel appears in up to `kH*kW` columns). For very large inputs and small batch sizes, this transient memory cost is noticeable.

---

## Exercise 8.3 — Padded and Strided Variants

**Difficulty**: ★★

Generalize 8.1 to support padding and stride:

```c
void conv2d_direct_strided(const float *in, ..., float *out,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int padH, int padW,
                           int strideH, int strideW);
```

Output dimensions: `H_out = (H + 2*padH - kH) / strideH + 1`. Pad lookups: a read at `IN(n, ci, h - padH, w - padW)` returns 0 when out of bounds. This is `same` padding when `padH = (kH - 1) / 2`.

---

## Exercise 8.4 — Implicit GEMM — Bonus

**Difficulty**: ★★★★

Modern frameworks fuse im2col into the GEMM kernel itself — the GEMM iterates over output elements but reads input pixels through index math instead of a materialized im2col matrix. This is what cuDNN's "implicit GEMM" does.

Modify your `gemm_ikj` from L04 so that, instead of indexing `B[k * N + j]`, it computes the corresponding `(ci, kh, kw)` and reads `in[...]` directly. The result has the speed of GEMM with the memory footprint of direct convolution.
