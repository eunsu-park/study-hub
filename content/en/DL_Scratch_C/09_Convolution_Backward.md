# 09. Convolution Backward

**Previous**: [Convolution from Scratch](./08_Convolution_from_Scratch.md) | **Next**: [Pooling Layers](./10_Pooling_Layers.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Derive the gradient of the loss with respect to the convolution input (∂L/∂X)
2. Derive the gradient of the loss with respect to the filter weights (∂L/∂W)
3. Derive the gradient with respect to the bias (∂L/∂b)
4. Implement all three backward passes in C using im2col
5. Verify backward correctness numerically using finite differences

---

## 1. Forward Pass Recap

The forward convolution computes:

```
Y[n][oc][oh][ow] = Σ_{ic,kh,kw} X[n][ic][oh*s+kh*d][ow*s+kw*d] * W[oc][ic][kh][kw]
                  + b[oc]

where s = stride, d = dilation
```

During backprop we receive `∂L/∂Y` (same shape as `Y`) and must compute:
- `∂L/∂X` — to pass gradient to the previous layer
- `∂L/∂W` — to update filter weights
- `∂L/∂b` — to update bias

---

## 2. Bias Gradient

The bias `b[oc]` is added to every spatial position of output channel `oc`:

```
∂L/∂b[oc] = Σ_{n,oh,ow} ∂L/∂Y[n][oc][oh][ow]
```

Implementation:

```c
// bias_backward: dL/db[oc] = sum over N,OH,OW of dL/dY
void bias_backward(
    const float *dY,  // [N, C_out, OH, OW]
    float       *db,  // [C_out]
    int N, int C_out, int OH, int OW) {

    memset(db, 0, C_out * sizeof(float));
    for (int n  = 0; n  < N;    n++)
    for (int oc = 0; oc < C_out; oc++)
    for (int oh = 0; oh < OH;   oh++)
    for (int ow = 0; ow < OW;   ow++)
        db[oc] += NCHW(dY, N, C_out, OH, OW, n, oc, oh, ow);
}
```

---

## 3. Filter Gradient

By the chain rule:

```
∂L/∂W[oc][ic][kh][kw] = Σ_{n,oh,ow} ∂L/∂Y[n][oc][oh][ow]
                          × X[n][ic][oh*s+kh*d][ow*s+kw*d]
```

Each output position `(oh, ow)` contributes to every filter element it touched during forward.

### via im2col

Using the im2col matrix from the forward pass:

```
col:    [N*OH*OW, C_in*KH*KW]
dY_mat: [N*OH*OW, C_out]   (reshape dY as [M, C_out])

dW = dY_mat^T × col        → [C_out, C_in*KH*KW] = W shape
```

```c
// weight_backward: dL/dW using im2col
void weight_backward(
    const float *col,   // [M, K]  im2col output (already computed)
    const float *dY,    // [N, C_out, OH, OW]
    float       *dW,    // [C_out, C_in, KH, KW]
    int M, int K, int C_out) {

    // dW[C_out, K] = dY[M, C_out]^T  ×  col[M, K]
    // = cblas_sgemm: C = A^T * B
    cblas_sgemm(CblasRowMajor,
                CblasTrans, CblasNoTrans,
                C_out, K, M,
                1.0f, dY, C_out,
                col,  K,
                1.0f, dW, K);  // accumulate (+=)
}
```

---

## 4. Input Gradient (Full Convolution)

The hardest gradient. By chain rule:

```
∂L/∂X[n][ic][ih][iw] = Σ_{oc,kh,kw} ∂L/∂Y[n][oc][oh][ow]
                         × W[oc][ic][kh][kw]

where oh = (ih - kh*d + pad) / s  (only integer positions)
      ow = (iw - kw*d + pad) / s
```

This is equivalent to **transposed convolution** (or "full convolution") — the gradient flows backward through each kernel position.

### Via col2im

The im2col backward is `col2im`: scatter `dL/d(col)` back to input gradients.

```
dcol = dY_mat × W          → [M, K]  (M = N*OH*OW, K = C_in*KH*KW)
dX   = col2im(dcol, ...)   → [N, C_in, H, W]
```

```c
// col2im: scatter dcol back to dX (reverse of im2col)
void col2im(
    const float *col,  // [N*OH*OW, C_in*KH*KW]
    int N, int C_in, int H, int W,
    int KH, int KW, int OH, int OW,
    int stride, int pad, int dilation,
    float *dX) {  // [N, C_in, H, W] — accumulated into

    int col_w = C_in * KH * KW;

    for (int n  = 0; n  < N;   n++)
    for (int oh = 0; oh < OH;  oh++)
    for (int ow = 0; ow < OW;  ow++) {
        int row = n * OH * OW + oh * OW + ow;

        for (int ic = 0; ic < C_in; ic++)
        for (int kh = 0; kh < KH;   kh++)
        for (int kw = 0; kw < KW;   kw++) {
            int col_idx = ic * KH * KW + kh * KW + kw;
            int ih = oh * stride + kh * dilation - pad;
            int iw = ow * stride + kw * dilation - pad;

            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                NCHW(dX, N, C_in, H, W, n, ic, ih, iw)
                    += col[row * col_w + col_idx];
        }
    }
}

// input_backward: compute dX from dY and W
void input_backward(
    const float *dY,    // [N, C_out, OH, OW]
    const float *W,     // [C_out, C_in, KH, KW]
    float       *dcol,  // [M, K] — temporary buffer
    float       *dX,    // [N, C_in, H, W] — output
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW, int OH, int OW,
    int stride, int pad, int dilation) {

    int M = N * OH * OW;
    int K = C_in * KH * KW;

    // dcol[M, K] = dY[M, C_out] × W[C_out, K]
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                M, K, C_out,
                1.0f, dY, C_out,
                W,    K,
                0.0f, dcol, K);

    // scatter dcol → dX
    memset(dX, 0, N * C_in * H * W * sizeof(float));
    col2im(dcol, N, C_in, H, W, KH, KW, OH, OW, stride, pad, dilation, dX);
}
```

---

## 5. Complete Backward Function

Combining all three:

```c
// conv2d_backward: computes dX, dW, db given dY
// Caller must allocate:
//   dX:  [N, C_in, H, W]
//   dW:  [C_out, C_in, KH, KW]  (zero-initialized before call)
//   db:  [C_out]                 (zero-initialized before call)
void conv2d_backward(
    const float *X,     // input  [N, C_in, H, W]
    const float *W,     // weight [C_out, C_in, KH, KW]
    const float *dY,    // output gradient [N, C_out, OH, OW]
    float       *dX,    // input gradient  [N, C_in, H, W]
    float       *dW,    // weight gradient [C_out, C_in, KH, KW]
    float       *db,    // bias gradient   [C_out]
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW, int OH, int OW,
    int stride, int pad, int dilation) {

    int M = N * OH * OW;
    int K = C_in * KH * KW;

    // 1. im2col of input (needed for dW)
    float *col = (float *)malloc(M * K * sizeof(float));
    im2col(X, N, C_in, H, W, KH, KW, OH, OW, stride, pad, dilation, col);

    // 2. Bias gradient
    bias_backward(dY, db, N, C_out, OH, OW);

    // 3. Weight gradient: dW += dY^T × col
    weight_backward(col, dY, dW, M, K, C_out);

    // 4. Input gradient: dcol = dY × W, then col2im
    float *dcol = (float *)malloc(M * K * sizeof(float));
    input_backward(dY, W, dcol, dX, N, C_in, H, W,
                   C_out, KH, KW, OH, OW, stride, pad, dilation);

    free(col);
    free(dcol);
}
```

---

## 6. Numerical Gradient Verification

Finite differences verify analytical gradients:

```
∂f/∂x_i ≈ (f(x + ε*e_i) - f(x - ε*e_i)) / (2ε)
```

```c
#define EPS 1e-4f

// Verify dL/dX by finite differences
static void verify_input_grad(
    const float *X, const float *W, const float *dY,
    int N, int C_in, int H, int W_,
    int C_out, int KH, int KW, int OH, int OW,
    int stride, int pad, int dilation) {

    int input_size  = N * C_in * H * W_;
    int output_size = N * C_out * OH * OW;

    float *X_plus  = (float *)malloc(input_size  * sizeof(float));
    float *X_minus = (float *)malloc(input_size  * sizeof(float));
    float *Y_plus  = (float *)malloc(output_size * sizeof(float));
    float *Y_minus = (float *)malloc(output_size * sizeof(float));
    float *dX_num  = (float *)malloc(input_size  * sizeof(float));
    float *dX_ana  = (float *)malloc(input_size  * sizeof(float));
    float *dW      = (float *)calloc(C_out * C_in * KH * KW, sizeof(float));
    float *db      = (float *)calloc(C_out, sizeof(float));

    memset(dX_ana, 0, input_size * sizeof(float));
    conv2d_backward(X, W, dY, dX_ana, dW, db,
                    N, C_in, H, W_, C_out, KH, KW, OH, OW,
                    stride, pad, dilation);

    // Finite difference for each input element
    int max_errors = 0;
    for (int i = 0; i < input_size; i++) {
        memcpy(X_plus,  X, input_size * sizeof(float));
        memcpy(X_minus, X, input_size * sizeof(float));
        X_plus[i]  += EPS;
        X_minus[i] -= EPS;

        conv2d_naive(X_plus,  N, C_in, H, W_, W, C_out, KH, KW,
                     Y_plus,  OH, OW, stride, pad, dilation);
        conv2d_naive(X_minus, N, C_in, H, W_, W, C_out, KH, KW,
                     Y_minus, OH, OW, stride, pad, dilation);

        // Numerical gradient = dY · (Y+ - Y-) / (2ε)
        float num_grad = 0.0f;
        for (int j = 0; j < output_size; j++)
            num_grad += dY[j] * (Y_plus[j] - Y_minus[j]) / (2.0f * EPS);

        dX_num[i] = num_grad;

        float rel_err = fabsf(dX_ana[i] - dX_num[i]) /
                        (fabsf(dX_num[i]) + 1e-8f);
        if (rel_err > 1e-3f) {
            printf("dX mismatch at i=%d: ana=%.6f  num=%.6f  rel=%.4f\n",
                   i, dX_ana[i], dX_num[i], rel_err);
            max_errors++;
        }
    }

    if (max_errors == 0)
        printf("dX gradient check PASSED (%d elements)\n", input_size);

    free(X_plus); free(X_minus); free(Y_plus); free(Y_minus);
    free(dX_num); free(dX_ana); free(dW); free(db);
}

// Main test
static void test_conv2d_backward(void) {
    int N=1, C_in=1, H=4, W=4, C_out=1, KH=3, KW=3;
    int stride=1, pad=0, dilation=1;
    int OH = conv_output_size(H, KH, stride, pad, dilation);  // = 2
    int OW = conv_output_size(W, KW, stride, pad, dilation);  // = 2

    float X[]  = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    float Wt[] = {1,0,-1, 1,0,-1, 1,0,-1};
    float dY[] = {1,1, 1,1};  // uniform gradient

    verify_input_grad(X, Wt, dY, N, C_in, H, W, C_out, KH, KW, OH, OW,
                      stride, pad, dilation);
}
```

---

## 7. Gradient Shapes Summary

```
Forward:   X [N,Cin,H,W]  ×  W [Cout,Cin,KH,KW]  →  Y [N,Cout,OH,OW]

Backward (given dY [N,Cout,OH,OW]):
  dX [N,Cin,H,W]        = col2im( dY_mat × W )
                          dY_mat: [M, Cout], W: [Cout, K]  → dcol: [M, K]
  dW [Cout,Cin,KH,KW]   = dY_mat^T × col
                          dY_mat: [M, Cout]^T, col: [M, K] → [Cout, K]
  db [Cout]              = sum(dY, axis=(N,OH,OW))

where M = N*OH*OW,  K = Cin*KH*KW
```

---

## Key Takeaways

- **dX (input gradient)** = transpose convolution: scatter dY through the same filter positions that created Y
- **dW (filter gradient)** = correlation of X with dY: same as forward but with input and output gradient swapped
- **db (bias gradient)** = sum of dY over spatial and batch dimensions
- All three backward passes reuse im2col/col2im — no new indexing logic beyond the forward pass
- **Finite difference verification** is mandatory before trusting analytical gradients: use ε=1e-4, check relative error < 1e-3

---

**Next**: [10. Pooling Layers](./10_Pooling_Layers.md) — Max pooling, average pooling, global average pooling, and their backward passes (argmax masking for max pool).
