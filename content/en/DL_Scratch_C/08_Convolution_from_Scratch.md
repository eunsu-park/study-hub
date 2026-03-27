# 08. Convolution from Scratch

**Previous**: [Memory Manager](./07_Memory_Manager.md) | **Next**: [Convolution Backward](./09_Convolution_Backward.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement 2D convolution with stride, padding, and dilation from scratch in C
2. Explain the im2col transformation that converts convolution to matrix multiplication
3. Measure the FLOP/byte ratio and identify why convolution is compute-bound at large sizes
4. Apply padding modes: valid, same, and full
5. Verify your implementation numerically against a reference

---

## 1. The Convolution Operation

Convolution slides a small filter (kernel) over a 2D input, computing element-wise products summed into an output:

```
Input:  H × W × C_in  (height, width, input channels)
Filter: K × K × C_in × C_out  (kernel height, width, input channels, output channels)
Output: H_out × W_out × C_out

H_out = (H + 2*pad - (K-1)*dilation - 1) / stride + 1
W_out = (W + 2*pad - (K-1)*dilation - 1) / stride + 1

Output[n][oc][oh][ow] =
    sum_{ic,kh,kw} Input[n][ic][oh*stride+kh*dil][ow*stride+kw*dil] * Filter[oc][ic][kh][kw]
```

---

## 2. Data Layout: NCHW

We use NCHW (batch, channel, height, width) — standard for CPU/CUDA implementations:

```c
// Access element at [n][c][h][w] in NCHW tensor
#define NCHW(ptr, N,C,H,W, n,c,h,w) \
    ((ptr)[(n)*(C)*(H)*(W) + (c)*(H)*(W) + (h)*(W) + (w)])
```

---

## 3. Naive 2D Convolution

Six nested loops — directly implementing the formula:

```c
// conv2d_naive.c
// Input:  [N, C_in, H, W]
// Weight: [C_out, C_in, KH, KW]
// Output: [N, C_out, OH, OW]
void conv2d_naive(
    const float *input,  int N, int C_in,  int H,  int W,
    const float *weight, int C_out, int KH, int KW,
    float       *output, int OH, int OW,
    int stride, int pad, int dilation) {

    for (int n  = 0; n  < N;    n++)
    for (int oc = 0; oc < C_out; oc++)
    for (int oh = 0; oh < OH;   oh++)
    for (int ow = 0; ow < OW;   ow++) {
        float sum = 0.0f;
        for (int ic = 0; ic < C_in;  ic++)
        for (int kh = 0; kh < KH;    kh++)
        for (int kw = 0; kw < KW;    kw++) {
            int ih = oh * stride + kh * dilation - pad;
            int iw = ow * stride + kw * dilation - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                float x = NCHW(input,  N, C_in, H,  W,  n,  ic, ih, iw);
                float w = NCHW(weight, C_out, C_in, KH, KW, oc, ic, kh, kw);
                sum += x * w;
            }
        }
        NCHW(output, N, C_out, OH, OW, n, oc, oh, ow) = sum;
    }
}
```

**FLOPs**: `N × C_out × OH × OW × C_in × KH × KW × 2`

For ResNet-50 first layer (N=1, C_in=3, C_out=64, H=224, K=7, stride=2):
- `1 × 64 × 112 × 112 × 3 × 7 × 7 × 2 ≈ 236 million FLOPs`

---

## 4. im2col: Turning Convolution into GEMM

The **im2col** transformation rearranges the input tensor so that convolution becomes a single matrix multiplication:

```
im2col output:  [N * OH * OW, C_in * KH * KW]  — each row = one receptive field
Weight matrix:  [C_out, C_in * KH * KW]
Output:         [N * OH * OW, C_out]  → reshape to [N, C_out, OH, OW]

Convolution = im2col(input) × weight^T
```

### im2col Implementation

```c
// im2col: extract receptive fields into columns
// out: [N * OH * OW, C_in * KH * KW]
void im2col(
    const float *input, int N, int C_in, int H, int W,
    int KH, int KW, int OH, int OW,
    int stride, int pad, int dilation,
    float *col) {

    int col_w = C_in * KH * KW;  // columns per patch

    for (int n  = 0; n  < N;    n++)
    for (int oh = 0; oh < OH;   oh++)
    for (int ow = 0; ow < OW;   ow++) {
        int row = n * OH * OW + oh * OW + ow;

        for (int ic = 0; ic < C_in; ic++)
        for (int kh = 0; kh < KH;   kh++)
        for (int kw = 0; kw < KW;   kw++) {
            int col_idx = ic * KH * KW + kh * KW + kw;
            int ih = oh * stride + kh * dilation - pad;
            int iw = ow * stride + kw * dilation - pad;

            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                col[row * col_w + col_idx] = NCHW(input, N, C_in, H, W, n, ic, ih, iw);
            else
                col[row * col_w + col_idx] = 0.0f;  // padding
        }
    }
}

// Convolution via im2col + GEMM
void conv2d_im2col(
    const float *input,  int N, int C_in, int H, int W,
    const float *weight, int C_out, int KH, int KW,
    float       *output, int OH, int OW,
    int stride, int pad, int dilation) {

    int M = N * OH * OW;        // rows of col matrix
    int K = C_in * KH * KW;    // inner dimension
    int n_out = C_out;          // rows of weight

    float *col = (float *)malloc(M * K * sizeof(float));
    im2col(input, N, C_in, H, W, KH, KW, OH, OW, stride, pad, dilation, col);

    // output[M, C_out] = col[M, K] @ weight^T[K, C_out]
    // Using CBLAS: C = alpha*A*B + beta*C
    // A = col [M×K], B = weight [C_out×K] (transposed → [K×C_out])
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasTrans,
                M, n_out, K,
                1.0f, col, K,
                weight, K,
                0.0f, output, n_out);

    free(col);
}
```

---

## 5. Performance Analysis

```
3×3 conv, C_in=C_out=64, OH=OW=56, N=1:
  FLOP = 1 × 64 × 56 × 56 × 64 × 3 × 3 × 2 = 229 million
  im2col memory: 56 × 56 × 64 × 3 × 3 = 18 MB (temporary buffer)
  Weight memory: 64 × 64 × 3 × 3 = 0.15 MB
  Arithmetic intensity = 229e6 / (18e6 + 0.15e6) / 4 ≈ 3.2 FLOP/byte

  → Memory-bound for small inputs, compute-bound for large batch sizes
  → im2col increases memory footprint by factor K²: mitigated by caching
```

The im2col buffer size is a problem for large kernels. Alternative: **Winograd convolution** reduces FLOP count for 3×3 filters by ~2.25×.

---

## 6. Depthwise Convolution

A variant where each input channel is filtered independently (no cross-channel mixing):

```c
// Depthwise conv: [N, C, H, W] → [N, C, OH, OW] (same C)
// Each channel has its own [KH, KW] filter
void depthwise_conv2d(
    const float *input,  int N, int C, int H, int W,
    const float *weight,                  int KH, int KW,
    float       *output,                  int OH, int OW,
    int stride, int pad) {

    for (int n  = 0; n  < N;  n++)
    for (int c  = 0; c  < C;  c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float sum = 0.0f;
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                sum += NCHW(input, N, C, H, W, n, c, ih, iw)
                     * weight[c * KH * KW + kh * KW + kw];
            }
        }
        NCHW(output, N, C, OH, OW, n, c, oh, ow) = sum;
    }
}
```

**FLOP reduction**: `C_out × C_in × K² → C × K²` (reduces by `C_in / 1` factor)

Used in MobileNet to reduce computation by ~8–9× compared to standard convolution.

---

## 7. Output Size Calculator

```c
int conv_output_size(int in_size, int kernel, int stride, int pad, int dilation) {
    return (in_size + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
}

// Validate output size matches allocation
void conv2d_validate(int H, int W, int KH, int KW,
                     int stride, int pad, int dilation,
                     int *OH, int *OW) {
    *OH = conv_output_size(H, KH, stride, pad, dilation);
    *OW = conv_output_size(W, KW, stride, pad, dilation);
    assert(*OH > 0 && *OW > 0);
}
```

Common padding modes:

```c
// Valid (no padding): output shrinks by (K-1)
int pad_valid = 0;

// Same (output same size as input for stride=1):
int pad_same = (K - 1) / 2;  // for odd K

// Full (output grows by K-1):
int pad_full = K - 1;
```

---

## 8. Numerical Verification

```c
static void test_conv2d(void) {
    // Tiny example: 1×1×4×4 input, 1×1×3×3 filter
    int N=1, C_in=1, H=4, W=4, C_out=1, KH=3, KW=3;
    int stride=1, pad=0, dilation=1;
    int OH = conv_output_size(H, KH, stride, pad, dilation);  // = 2
    int OW = conv_output_size(W, KW, stride, pad, dilation);  // = 2

    float input[]  = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    float filter[] = {1,0,-1, 1,0,-1, 1,0,-1};  // horizontal edge detector
    float output_naive[4], output_im2col[4];

    conv2d_naive(input, N,C_in,H,W, filter,C_out,KH,KW,
                 output_naive, OH,OW, stride,pad,dilation);
    conv2d_im2col(input, N,C_in,H,W, filter,C_out,KH,KW,
                  output_im2col, OH,OW, stride,pad,dilation);

    for (int i = 0; i < OH * OW; i++) {
        float diff = fabsf(output_naive[i] - output_im2col[i]);
        assert(diff < 1e-4f);
    }
    printf("conv2d test PASSED\n");
    // Expected output: [-3,-3, -3,-3] (each receptive field has zero column sum)
}
```

---

## Key Takeaways

- Convolution = six nested loops over (N, C_out, OH, OW, C_in, KH, KW) — directly maps to the formula
- **im2col** converts convolution to a single `sgemm` call — leverages heavily optimized BLAS; standard in frameworks (cuDNN, PyTorch, TensorFlow)
- im2col trades compute efficiency for memory overhead: the col buffer is `K²` times larger than the input
- **Depthwise convolution** applies K filters per channel instead of C_out per input — used in MobileNet to reduce FLOPs by ~8×
- Always validate with a small case: compare naive vs im2col output numerically before scaling up

---

**Next**: [09. Convolution Backward](./09_Convolution_Backward.md) — Derive the backward pass for convolution: input gradient (full convolution), filter gradient, and bias gradient — then verify with finite differences.
