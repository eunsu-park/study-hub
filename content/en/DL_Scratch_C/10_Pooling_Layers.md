# 10. Pooling Layers

**Previous**: [Convolution Backward](./09_Convolution_Backward.md) | **Next**: [Batch Normalization](./11_Batch_Normalization.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement max pooling forward and its argmax-based backward pass
2. Implement average pooling forward and backward
3. Implement global average pooling (GAP) for classification heads
4. Explain why pooling has no learnable parameters yet still has non-trivial backward passes
5. Verify pooling backward with finite differences

---

## 1. Why Pooling?

Pooling reduces spatial dimensions while retaining dominant features:

```
Input:  [N, C, H, W]
Output: [N, C, OH, OW]    where OH = (H - K) / stride + 1

Benefits:
  - Downsamples feature maps (reduces memory and compute)
  - Introduces local translation invariance (max pool)
  - Reduces spatial dimensions before fully-connected layers
  - Global average pooling eliminates FC layers entirely (ResNet, EfficientNet)
```

---

## 2. Max Pooling

### Forward Pass

Each output is the maximum value within the pooling window:

```c
// max_pool2d_forward: [N, C, H, W] → [N, C, OH, OW]
// Also stores argmax indices for backward pass
void max_pool2d_forward(
    const float *input,   // [N, C, H, W]
    float       *output,  // [N, C, OH, OW]
    int         *argmax,  // [N, C, OH, OW] — index into flattened [H*W] per channel
    int N, int C, int H, int W,
    int KH, int KW, int OH, int OW,
    int stride, int pad) {

    for (int n  = 0; n  < N;  n++)
    for (int c  = 0; c  < C;  c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float max_val = -FLT_MAX;
        int   max_idx = -1;

        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                float val = NCHW(input, N, C, H, W, n, c, ih, iw);
                if (val > max_val) {
                    max_val = val;
                    max_idx = ih * W + iw;  // flattened index within [H×W]
                }
            }
        }

        NCHW(output, N, C, OH, OW, n, c, oh, ow) = max_val;
        NCHW(argmax, N, C, OH, OW, n, c, oh, ow) = max_idx;
    }
}
```

### Backward Pass

Gradient flows only to the position that held the maximum value (argmax masking):

```c
// max_pool2d_backward: route dY gradient to argmax positions
void max_pool2d_backward(
    const float *dY,      // [N, C, OH, OW]
    const int   *argmax,  // [N, C, OH, OW]
    float       *dX,      // [N, C, H, W]  — must be zero-initialized
    int N, int C, int H, int W,
    int OH, int OW) {

    for (int n  = 0; n  < N;  n++)
    for (int c  = 0; c  < C;  c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float grad = NCHW(dY,     N, C, OH, OW, n, c, oh, ow);
        int   idx  = NCHW(argmax, N, C, OH, OW, n, c, oh, ow);

        if (idx >= 0) {
            int ih = idx / W;
            int iw = idx % W;
            NCHW(dX, N, C, H, W, n, c, ih, iw) += grad;
        }
    }
}
```

**Key insight**: When multiple output positions overlap (stride < K), a single input element can receive gradients from several outputs — hence `+=`.

---

## 3. Average Pooling

### Forward Pass

Each output is the mean of the pooling window:

```c
// avg_pool2d_forward: [N, C, H, W] → [N, C, OH, OW]
void avg_pool2d_forward(
    const float *input,
    float       *output,
    int N, int C, int H, int W,
    int KH, int KW, int OH, int OW,
    int stride, int pad) {

    for (int n  = 0; n  < N;  n++)
    for (int c  = 0; c  < C;  c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float sum = 0.0f;
        int   cnt = 0;

        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                sum += NCHW(input, N, C, H, W, n, c, ih, iw);
                cnt++;
            }
        }

        NCHW(output, N, C, OH, OW, n, c, oh, ow) = (cnt > 0) ? sum / cnt : 0.0f;
    }
}
```

### Backward Pass

Gradient distributes uniformly across the pooling window:

```c
// avg_pool2d_backward: distribute dY/count back to each window element
void avg_pool2d_backward(
    const float *dY,
    float       *dX,   // zero-initialized
    int N, int C, int H, int W,
    int KH, int KW, int OH, int OW,
    int stride, int pad) {

    for (int n  = 0; n  < N;  n++)
    for (int c  = 0; c  < C;  c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        // Count valid positions in this window
        int cnt = 0;
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) cnt++;
        }

        float grad_per_elem = NCHW(dY, N, C, OH, OW, n, c, oh, ow) / cnt;

        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                NCHW(dX, N, C, H, W, n, c, ih, iw) += grad_per_elem;
        }
    }
}
```

---

## 4. Global Average Pooling (GAP)

GAP collapses each feature map to a single scalar — replaces the large FC layer in ResNet and EfficientNet:

```c
// gap_forward: [N, C, H, W] → [N, C]
void gap_forward(
    const float *input,
    float       *output,  // [N, C]
    int N, int C, int H, int W) {

    int spatial = H * W;
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            sum += NCHW(input, N, C, H, W, n, c, h, w);
        output[n * C + c] = sum / spatial;
    }
}

// gap_backward: [N, C] → [N, C, H, W]
void gap_backward(
    const float *dOut,  // [N, C]
    float       *dX,    // [N, C, H, W] — zero-initialized
    int N, int C, int H, int W) {

    float spatial = (float)(H * W);
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++) {
        float grad = dOut[n * C + c] / spatial;
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            NCHW(dX, N, C, H, W, n, c, h, w) += grad;
    }
}
```

**Comparison — FC vs GAP**:

```
ResNet-50 without GAP:
  conv(2048, 7×7) → flatten(2048×7×7 = 100352) → FC(100352, 1000)
  FC parameters: 100352 × 1000 = 100M params

ResNet-50 with GAP:
  conv(2048, 7×7) → GAP → [N, 2048] → FC(2048, 1000)
  FC parameters: 2048 × 1000 = 2M params  (50× reduction)
```

---

## 5. Numerical Verification

```c
static void test_max_pool_backward(void) {
    int N=1, C=1, H=4, W=4, KH=2, KW=2, stride=2, pad=0;
    int OH = (H - KH) / stride + 1;  // = 2
    int OW = (W - KW) / stride + 1;  // = 2

    float X[]  = {3,1, 4,2,  1,5, 9,6,  2,7, 8,3,  0,4, 6,1};
    float dY[] = {1.0f, 1.0f, 1.0f, 1.0f};  // uniform upstream gradient

    int   argmax[4];
    float Y[4], dX_ana[16], dX_num[16];
    memset(dX_ana, 0, sizeof(dX_ana));

    max_pool2d_forward(X, Y, argmax, N, C, H, W, KH, KW, OH, OW, stride, pad);
    max_pool2d_backward(dY, argmax, dX_ana, N, C, H, W, OH, OW);

    // Finite differences
    for (int i = 0; i < 16; i++) {
        float X2[16];
        float Y_plus[4], Y_minus[4];
        memcpy(X2, X, sizeof(X));

        X2[i] += 1e-4f;
        int dummy_argmax[4];
        max_pool2d_forward(X2, Y_plus, dummy_argmax, N,C,H,W,KH,KW,OH,OW,stride,pad);

        memcpy(X2, X, sizeof(X));
        X2[i] -= 1e-4f;
        max_pool2d_forward(X2, Y_minus, dummy_argmax, N,C,H,W,KH,KW,OH,OW,stride,pad);

        float num_grad = 0.0f;
        for (int j = 0; j < 4; j++)
            num_grad += dY[j] * (Y_plus[j] - Y_minus[j]) / (2e-4f);
        dX_num[i] = num_grad;

        float err = fabsf(dX_ana[i] - dX_num[i]);
        if (err > 1e-3f)
            printf("FAIL i=%d ana=%.4f num=%.4f\n", i, dX_ana[i], dX_num[i]);
    }
    printf("max_pool backward PASSED\n");
}
```

---

## 6. Pooling in a Network

Typical placement in a CNN:

```
Conv → ReLU → MaxPool   (early layers: aggressive downsampling)
Conv → BN → ReLU        (middle layers: no pooling, preserve resolution)
Conv → GAP → FC         (final layer: spatial collapse, then classification)

Stride-2 conv vs MaxPool:
  MaxPool:    no parameters, translation invariant, keeps max feature
  Stride-2:   learned downsampling (ResNet post-v1 prefers this)
  Both:       2× spatial reduction per application
```

---

## Key Takeaways

- **Max pool forward** saves argmax indices — needed during backward to route gradients correctly
- **Max pool backward** is argmax masking: gradient flows only to the winning position, zero to all others
- **Average pool backward** distributes gradient uniformly: `dX += dY / count` for each window element
- **Global average pooling** reduces each [H×W] feature map to a scalar — eliminates large FC layers (50× param reduction in ResNet)
- Both max and avg pooling have no learnable parameters but still have non-trivial backward passes

---

**Next**: [11. Batch Normalization](./11_Batch_Normalization.md) — BN forward (train/eval modes), backward pass through mean and variance, running statistics, and gamma/beta parameters.
