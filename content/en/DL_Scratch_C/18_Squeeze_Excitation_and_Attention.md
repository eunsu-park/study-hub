# 18. Squeeze-Excitation and Attention

**Previous**: [Depthwise Separable Convolution](./17_Depthwise_Separable_Conv.md) | **Next**: [EfficientNet Scaling](./19_EfficientNet_Scaling.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement the Squeeze-Excitation (SE) block for channel attention
2. Implement CBAM's spatial attention module
3. Explain how SE blocks improve accuracy with minimal parameter cost
4. Connect channel attention to the self-attention mechanism in Transformers
5. Apply SE blocks inside a ResNet bottleneck (SE-ResNet)

---

## 1. The Motivation for Attention in CNNs

Standard convolutions treat all channels equally — each filter mixes all input channels uniformly. In practice, some channels are more informative for a given input than others.

**Channel attention** answers: *given this input, which channels should be amplified?*

```
Standard conv: Y = W × X  (fixed weights, channel importances fixed)
SE block:      Y = σ(FC(GAP(X))) ⊙ X  (dynamic per-channel scaling)

⊙ = element-wise multiplication (broadcast over spatial dims)
σ = sigmoid (output in [0,1] = attention weights)
```

---

## 2. Squeeze-Excitation Block

Hu et al. (2018) — won ILSVRC 2017.

```
Structure:
  X [N, C, H, W]
    ↓ Squeeze: GlobalAvgPool → [N, C]
    ↓ Excitation: FC(C → C/r) → ReLU → FC(C/r → C) → Sigmoid → [N, C]
    ↓ Scale: broadcast multiply → [N, C, H, W]
    = Y

  r = reduction ratio (default 16): controls parameter cost
  FC layers are small: C×C/r + C/r×C = 2C²/r params
  For C=256, r=16: 2×256²/16 = 8192 params  (vs conv: 256×256×9 = 589824)
```

### Implementation

```c
typedef struct {
    int C, C_reduced;     // C_reduced = C / reduction
    float *fc1_w;         // [C_reduced, C]
    float *fc1_b;         // [C_reduced]
    float *fc2_w;         // [C, C_reduced]
    float *fc2_b;         // [C]
} SEBlock;

SEBlock *se_create(int C, int reduction) {
    SEBlock *se = calloc(1, sizeof(SEBlock));
    se->C = C;
    se->C_reduced = C / reduction;
    se->fc1_w = malloc(se->C_reduced * C * sizeof(float));
    se->fc1_b = calloc(se->C_reduced, sizeof(float));
    se->fc2_w = malloc(C * se->C_reduced * sizeof(float));
    se->fc2_b = calloc(C, sizeof(float));
    // He init for fc1, fc2
    he_init(se->fc1_w, C, se->C_reduced);
    he_init(se->fc2_w, se->C_reduced, C);
    return se;
}

// Sigmoid activation
void sigmoid_forward(float *X, int size) {
    for (int i = 0; i < size; i++)
        X[i] = 1.0f / (1.0f + expf(-X[i]));
}

// se_forward: apply SE attention to X, result in Y (or in-place if Y == X)
void se_forward(
    SEBlock     *se,
    const float *X,     // [N, C, H, W]
    float       *Y,     // [N, C, H, W]
    float       *gap,   // [N, C] — GAP output
    float       *fc1,   // [N, C_reduced]
    float       *fc2,   // [N, C] — attention weights
    int N, int C, int H, int W) {

    // 1. Squeeze: Global Average Pooling [N,C,H,W] → [N,C]
    gap_forward(X, gap, N, C, H, W);

    // 2. Excitation FC1: [N,C] × [C,C_r]^T → [N,C_r] + bias → ReLU
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, se->C_reduced, C,
                1.0f, gap, C, se->fc1_w, C,
                0.0f, fc1, se->C_reduced);
    for (int n = 0; n < N; n++)
    for (int j = 0; j < se->C_reduced; j++)
        fc1[n * se->C_reduced + j] += se->fc1_b[j];
    relu_forward(fc1, N * se->C_reduced);

    // 3. Excitation FC2: [N,C_r] × [C_r,C]^T → [N,C] + bias → Sigmoid
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, C, se->C_reduced,
                1.0f, fc1, se->C_reduced, se->fc2_w, se->C_reduced,
                0.0f, fc2, C);
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
        fc2[n * C + c] += se->fc2_b[c];
    sigmoid_forward(fc2, N * C);

    // 4. Scale: broadcast attention weights over spatial dims
    if (Y != X) memcpy(Y, X, (size_t)N * C * H * W * sizeof(float));
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++) {
        float scale = fc2[n * C + c];
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            NCHW(Y, N, C, H, W, n, c, h, w) *= scale;
    }
}
```

---

## 3. SE-ResNet Block

Insert SE after the last BN but before the shortcut add:

```c
// SE-ResNet basic block forward
void se_resblock_forward(
    ResidualBlock *blk,
    SEBlock       *se,
    const float   *X,
    float         *Y,
    // ... buffers ...
    int N, int H, int W, int training) {

    // Standard ResBlock forward (up to BN2, no ReLU yet)
    resblock_pre_relu_forward(blk, X, Y, ..., N, H, W, training);

    int OH = conv_output_size(H, 3, blk->stride, 1, 1);
    int OW = conv_output_size(W, 3, blk->stride, 1, 1);

    // SE attention on main path (before adding shortcut)
    float *gap = malloc(N * blk->C_out * sizeof(float));
    float *fc1 = malloc(N * (blk->C_out / 16) * sizeof(float));
    float *fc2 = malloc(N * blk->C_out * sizeof(float));
    se_forward(se, Y, Y, gap, fc1, fc2, N, blk->C_out, OH, OW);
    free(gap); free(fc1); free(fc2);

    // Add shortcut, then ReLU
    add_shortcut_and_relu(blk, X, Y, N, H, W, OH, OW);
}
```

**Parameter overhead**:

```
SE block with r=16, C=256:
  FC1: 256 × 16 = 4,096
  FC2: 16 × 256 = 4,096
  Total: 8,192 params

ResNet-50 has ~25M params total
SE-ResNet-50 adds ~2.5M params (10% overhead)
ImageNet accuracy: ResNet-50 76.1% → SE-ResNet-50 77.6% (+1.5%)
```

---

## 4. Spatial Attention (CBAM)

CBAM (Woo et al., 2018) combines channel + spatial attention:

```
Channel Attention: "Which channels are important?"
  → same as SE block

Spatial Attention: "Which locations in each channel are important?"
  Input:  [N, C, H, W]
  Pool:   channel max-pool + channel avg-pool → [N, 2, H, W]
  Conv:   7×7 conv → [N, 1, H, W] → Sigmoid
  Output: spatial attention map [N, 1, H, W] × X
```

```c
// cbam_spatial_forward: compute spatial attention map
void cbam_spatial_forward(
    const float *X,         // [N, C, H, W]
    const float *sp_conv_w, // [1, 2, 7, 7] spatial conv filter
    float       *Y,         // [N, C, H, W] — output
    float       *pool_feats, // [N, 2, H, W] — channel-pooled
    float       *sp_attn,    // [N, 1, H, W] — spatial attention map
    int N, int C, int H, int W) {

    // Channel max-pool and avg-pool
    for (int n = 0; n < N; n++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++) {
        float max_v = -FLT_MAX, avg_v = 0.0f;
        for (int c = 0; c < C; c++) {
            float v = NCHW(X, N, C, H, W, n, c, h, w);
            if (v > max_v) max_v = v;
            avg_v += v;
        }
        avg_v /= C;
        pool_feats[n * 2 * H * W + 0 * H * W + h * W + w] = avg_v;
        pool_feats[n * 2 * H * W + 1 * H * W + h * W + w] = max_v;
    }

    // 7×7 conv on [N, 2, H, W] → [N, 1, H, W]
    conv2d_naive(pool_feats, N, 2, H, W,
                 sp_conv_w, 1, 7, 7,
                 sp_attn, H, W, 1, 3, 1);  // pad=3 for same size

    // Sigmoid
    sigmoid_forward(sp_attn, N * H * W);

    // Multiply attention map into X
    if (Y != X) memcpy(Y, X, (size_t)N * C * H * W * sizeof(float));
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++)
        NCHW(Y, N, C, H, W, n, c, h, w)
            *= sp_attn[n * H * W + h * W + w];
}
```

---

## 5. Connection to Transformer Self-Attention

SE block is a precursor to Transformer self-attention:

```
SE block (channel attention):
  q, k, v all come from GAP(X)       → global context
  attention = σ(FC(FC(GAP(X))))      → per-channel gate
  output = attention ⊙ X             → recalibrate channels

Transformer self-attention:
  Q = X × W_Q, K = X × W_K, V = X × W_V
  attention = softmax(Q×K^T / √d)
  output = attention × V             → recalibrate positions

Key difference:
  SE: attention over channels (C-dimensional gate)
  Self-attention: attention over sequence positions (N-dimensional weight matrix)
  ViT: applies self-attention to patches of an image → full spatial recalibration
```

---

## 6. Accuracy Impact

```
ImageNet top-1 accuracy:

ResNet-50:         76.1%
SE-ResNet-50:      77.6%  (+1.5%)  → 2.5M extra params
CBAM-ResNet-50:    77.3%  (+1.2%)  → 0.1M extra params (spatial conv tiny)

MobileNetV1:       70.9%
SE-MobileNetV1:    73.0%  (+2.1%)  ← larger gain on lightweight networks

Conclusion: SE blocks have the best accuracy/parameter tradeoff
            especially beneficial for small/mobile architectures
```

---

## Key Takeaways

- **SE block**: GlobalAvgPool → FC → ReLU → FC → Sigmoid → channel-wise scale; 10% param overhead for ~1.5% accuracy gain
- **Channel attention** answers "which channels matter for this input" dynamically
- **CBAM** adds spatial attention on top of channel attention — learns "where to look"
- SE block is a direct predecessor to Transformer self-attention; both compute attention weights and apply them multiplicatively
- Insert SE block after the last conv/BN in a residual block, before the shortcut addition

---

**Next**: [19. EfficientNet Scaling](./19_EfficientNet_Scaling.md) — Compound scaling of depth, width, and resolution; the EfficientNet-B0 architecture; NAS search space concepts.
