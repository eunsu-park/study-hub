# 17. Depthwise Separable Convolution

**Previous**: [ResNet and Skip Connections](./16_ResNet_and_Skip_Connections.md) | **Next**: [Squeeze-Excitation and Attention](./18_Squeeze_Excitation_and_Attention.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how depthwise separable convolution factorizes standard convolution
2. Calculate the exact FLOP reduction compared to standard convolution
3. Implement depthwise convolution forward and backward in C
4. Implement pointwise (1×1) convolution as a matrix multiply
5. Build a MobileNet-style inverted residual block

---

## 1. Standard vs Depthwise Separable Convolution

### Standard Convolution

```
Input:  [N, C_in, H, W]
Filter: [C_out, C_in, K, K]
Output: [N, C_out, OH, OW]

FLOPs = N × C_out × OH × OW × C_in × K × K × 2
      = 2 × N × OH × OW × C_out × C_in × K²
```

### Depthwise Separable Convolution

Split into two steps:

```
Step 1 — Depthwise conv: each channel filtered independently
  Input:  [N, C_in, H, W]
  Filter: [C_in, 1, K, K]  (one K×K filter per input channel, NO cross-channel)
  Output: [N, C_in, OH, OW]
  FLOPs = 2 × N × C_in × OH × OW × K²

Step 2 — Pointwise conv (1×1 conv): mix channels
  Input:  [N, C_in, OH, OW]
  Filter: [C_out, C_in, 1, 1]
  Output: [N, C_out, OH, OW]
  FLOPs = 2 × N × C_out × OH × OW × C_in
```

### FLOP Ratio

```
Total DWS FLOPs = 2 × N × OH × OW × C_in × (K² + C_out)

Ratio vs standard:
  DWS / standard = (K² + C_out) / (C_in × K² / C_in × C_out... wait, let me redo)

Standard: 2 × N × OH × OW × C_out × C_in × K²
DWS:      2 × N × OH × OW × C_in × K² + 2 × N × OH × OW × C_out × C_in
        = 2 × N × OH × OW × C_in × (K² + C_out)

Ratio = (K² + C_out) / (C_out × K²)
      = 1/C_out + 1/K²

For K=3, C_out=256:  1/256 + 1/9 ≈ 0.115  →  8.7× fewer FLOPs
For K=3, C_out=128:  1/128 + 1/9 ≈ 0.119  →  8.4× fewer FLOPs
For K=5, C_out=256:  1/256 + 1/25 ≈ 0.044 → 22.8× fewer FLOPs
```

---

## 2. Depthwise Convolution

### Forward Pass

```c
// depthwise_conv2d_forward: each channel has its own [KH, KW] filter
// Input:   [N, C, H, W]
// Weight:  [C, 1, KH, KW]  stored as [C, KH*KW]
// Output:  [N, C, OH, OW]
void depthwise_conv2d_forward(
    const float *input,   // [N, C, H, W]
    const float *weight,  // [C, KH, KW]  (one filter per channel)
    const float *bias,    // [C]  (may be NULL)
    float       *output,  // [N, C, OH, OW]
    int N, int C, int H, int W,
    int KH, int KW, int OH, int OW,
    int stride, int pad) {

    for (int n  = 0; n  < N;  n++)
    for (int c  = 0; c  < C;  c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float sum = bias ? bias[c] : 0.0f;
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                sum += NCHW(input, N, C, H, W, n, c, ih, iw)
                     * weight[c * KH * KW + kh * KW + kw];
        }
        NCHW(output, N, C, OH, OW, n, c, oh, ow) = sum;
    }
}
```

### Backward Pass

```c
// depthwise_conv2d_backward: compute dX, dW, db
void depthwise_conv2d_backward(
    const float *input,   // [N, C, H, W]
    const float *weight,  // [C, KH, KW]
    const float *dY,      // [N, C, OH, OW]
    float       *dX,      // [N, C, H, W]   — zero-initialized
    float       *dW,      // [C, KH, KW]    — zero-initialized
    float       *db,      // [C]             — zero-initialized (may be NULL)
    int N, int C, int H, int W,
    int KH, int KW, int OH, int OW,
    int stride, int pad) {

    for (int n  = 0; n  < N;  n++)
    for (int c  = 0; c  < C;  c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float grad = NCHW(dY, N, C, OH, OW, n, c, oh, ow);
        if (db) db[c] += grad;

        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                float x_val = NCHW(input, N, C, H, W, n, c, ih, iw);
                // dW
                dW[c * KH * KW + kh * KW + kw] += grad * x_val;
                // dX
                NCHW(dX, N, C, H, W, n, c, ih, iw)
                    += grad * weight[c * KH * KW + kh * KW + kw];
            }
        }
    }
}
```

---

## 3. Pointwise Convolution (1×1 conv)

A 1×1 convolution is equivalent to a matrix multiply along the channel dimension:

```c
// pointwise_conv2d: 1×1 conv = matmul across channels at each spatial position
// Internally: reshape [N, C_in, H, W] → [N*H*W, C_in]
//             then matmul with weight [C_out, C_in]^T
void pointwise_conv2d(
    const float *input,   // [N, C_in, H, W]
    const float *weight,  // [C_out, C_in]
    const float *bias,    // [C_out]
    float       *output,  // [N, C_out, H, W]
    int N, int C_in, int C_out, int H, int W) {

    int M = N * H * W;  // number of spatial positions

    // Reinterpret input as [N*H*W, C_in] — requires NHWC-like view for matmul
    // Strategy: transpose to NHWC, matmul, transpose back to NCHW
    float *X_nhwc = malloc(M * C_in * sizeof(float));
    float *Y_nhwc = malloc(M * C_out * sizeof(float));

    // NCHW → NHWC
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C_in; c++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++)
        X_nhwc[n * H * W * C_in + h * W * C_in + w * C_in + c]
            = NCHW(input, N, C_in, H, W, n, c, h, w);

    // Matmul: [M, C_in] × [C_in, C_out] → [M, C_out]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, C_out, C_in,
                1.0f, X_nhwc, C_in,
                weight, C_in,
                0.0f, Y_nhwc, C_out);

    // Add bias
    if (bias) {
        for (int i = 0; i < M; i++)
        for (int c = 0; c < C_out; c++)
            Y_nhwc[i * C_out + c] += bias[c];
    }

    // NHWC → NCHW
    for (int n = 0; n < N; n++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++)
    for (int c = 0; c < C_out; c++)
        NCHW(output, N, C_out, H, W, n, c, h, w)
            = Y_nhwc[n * H * W * C_out + h * W * C_out + w * C_out + c];

    free(X_nhwc); free(Y_nhwc);
}
```

---

## 4. MobileNet-Style Inverted Residual Block

MobileNetV2 uses "inverted residuals": expand channels → depthwise → compress:

```
Standard residual (ResNet):
  wide → narrow → wide  (bottleneck compresses)

Inverted residual (MobileNetV2):
  narrow → wide → narrow  (expansion then compression)

Structure:
  x → PW(t×C_in) → BN → ReLU6 → DW(stride) → BN → ReLU6 → PW(C_out) → BN
  ↓_____________________________________________________↑ (if stride=1 and C_in==C_out)

Where t = expansion factor (typically 6)
```

```c
// ReLU6: clamps activations to [0, 6] — avoids large activations in low-precision
void relu6_forward(float *X, int size) {
    for (int i = 0; i < size; i++)
        X[i] = fmaxf(0.0f, fminf(6.0f, X[i]));
}

typedef struct {
    int C_in, C_mid, C_out, stride;
    float *pw1_w;   // [C_mid, C_in]       expand
    float *dw_w;    // [C_mid, KH, KW]     depthwise
    float *pw2_w;   // [C_out, C_mid]      project
    // BN for each sub-layer...
    int use_residual;  // 1 if stride=1 and C_in==C_out
} InvertedResidual;

// FLOP count for an inverted residual block
long inverted_residual_flops(int C_in, int C_out, int H, int W, int t, int K, int stride) {
    int C_mid = C_in * t;
    int OH = (H + stride - 1) / stride;
    int OW = (W + stride - 1) / stride;
    long pw1_flops = 2L * H * W * C_mid * C_in;
    long dw_flops  = 2L * OH * OW * C_mid * K * K;
    long pw2_flops = 2L * OH * OW * C_out * C_mid;
    return pw1_flops + dw_flops + pw2_flops;
}
```

---

## 5. MobileNet vs ResNet: FLOP Comparison

```
Task: 3×3 conv on [1, 128, 56, 56] → [1, 256, 56, 56]

Standard conv:  2 × 256 × 56 × 56 × 128 × 9 = 924M FLOPs
DWS conv:       2 × 128 × 56 × 56 × 9  (DW) = 71.7M
              + 2 × 256 × 56 × 56 × 128 (PW) = 102.8M
              = 174.5M FLOPs  →  5.3× fewer

MobileNetV1 vs ResNet-50 (ImageNet):
  MobileNetV1: 569M FLOPs, 72.0% top-1
  ResNet-50:  4100M FLOPs, 76.1% top-1
  → MobileNet is 7.2× cheaper in FLOPs for only 4.1% accuracy drop

MobileNetV2 (inverted residuals):
  300M FLOPs, 72.0% top-1 (vs MobileNetV1 at 569M)
  → inverted residuals + linear bottleneck further reduces computation
```

---

## 6. When to Use Depthwise Separable Conv

```
Hardware considerations:
  - DWS: good on ARM CPUs (NEON SIMD), mobile NPUs
  - Standard: good on GPU (large matmul is more GPU-efficient)
  - Depthwise alone has low arithmetic intensity → memory-bound on GPU

Use DWS when:
  - Deploying on mobile/edge hardware
  - FLOPs budget is the primary constraint
  - Batch size is small (inference)

Prefer standard conv when:
  - GPU training where cuDNN/cuBLAS GEMM is highly optimized
  - Accuracy is the primary goal (DWS loses some representational capacity)
```

---

## Key Takeaways

- **Depthwise separable conv** = depthwise (spatial filtering per channel) + pointwise (1×1 channel mixing)
- FLOP reduction: `1/C_out + 1/K²` — approximately 8-9× for K=3, C_out≥64
- Depthwise backward: gradients flow channel-independently — no cross-channel mixing in dX, dW
- Pointwise conv = matrix multiply — implement via `cblas_sgemm` after NCHW→NHWC reshape
- **MobileNetV2** uses inverted residuals (expand → DW → project) for state-of-the-art efficiency

---

**Next**: [18. Squeeze-Excitation and Attention](./18_Squeeze_Excitation_and_Attention.md) — Channel attention (SE blocks), spatial attention (CBAM), and how these prepare for Vision Transformers.
