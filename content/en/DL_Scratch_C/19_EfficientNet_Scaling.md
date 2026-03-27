# 19. EfficientNet Scaling

**Previous**: [Squeeze-Excitation and Attention](./18_Squeeze_Excitation_and_Attention.md) | **Next**: [Modern CNN Benchmark](./20_Modern_CNN_Benchmark.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the compound scaling rule and why it outperforms single-dimension scaling
2. Calculate EfficientNet-Bx dimensions from the base B0 architecture
3. Implement the MBConv (mobile inverted bottleneck conv) block with SE
4. Build EfficientNet-B0 by stacking MBConv blocks
5. Profile FLOPs and parameter counts for EfficientNet variants

---

## 1. The Scaling Problem

Prior networks scale along a single dimension:

```
Depth scaling (ResNet):    add more layers
  ResNet-18 → 34 → 50 → 101 → 152
  Accuracy plateaus due to vanishing gradients

Width scaling (WideResNet): increase channel count
  WideResNet-28×10 (10× wider per layer)
  Accuracy saturates without depth

Resolution scaling:        use larger input images
  32×32 → 224×224 → 480×480
  Requires more memory and compute

Problem: scaling one dimension hits diminishing returns
         while another dimension becomes the bottleneck
```

### Compound Scaling (Tan & Le, 2019)

Scale all three dimensions together with a fixed ratio:

```
Given a compound coefficient φ:
  depth:      d = α^φ
  width:      w = β^φ
  resolution: r = γ^φ

Subject to: α × β² × γ² ≈ 2  (total FLOP scaling ≈ 2^φ)

For EfficientNet: α=1.2, β=1.1, γ=1.15 (found by NAS grid search on φ=1)

EfficientNet-Bx scaling table:
  B0: d=1.0, w=1.0, r=224  (baseline)
  B1: d=1.1, w=1.0, r=240
  B2: d=1.2, w=1.1, r=260
  B3: d=1.4, w=1.2, r=300
  B4: d=1.8, w=1.4, r=380
  B5: d=2.2, w=1.6, r=456
  B6: d=2.6, w=1.8, r=528
  B7: d=3.1, w=2.0, r=600
```

---

## 2. EfficientNet-B0 Architecture

B0 is the NAS-found baseline — 7 stages of MBConv blocks:

```
Stage  Block    #Layers  C_in→C_out  k  Stride  Expand
  0    MBConv1     1      32→16      3    1       1    (no expansion)
  1    MBConv6     2      16→24      3    2       6
  2    MBConv6     2      24→40      5    2       6
  3    MBConv6     3      40→80      3    2       6
  4    MBConv6     3      80→112     5    1       6
  5    MBConv6     4      112→192    5    2       6
  6    MBConv6     1      192→320    3    1       6

Stem:   Conv(3→32, 3×3, s=2)        [N,32,112,112]
Stages 0-6:                          [N,320,7,7]
Head:   Conv(320→1280, 1×1)          [N,1280,7,7]
GAP:                                 [N,1280]
Dropout(0.2)
FC:     1280→1000

Total params: ~5.3M
ImageNet top-1: 77.1%  (vs ResNet-50: 76.1% at 25.6M params)
```

---

## 3. MBConv Block with SE

```c
typedef struct {
    int C_in, C_mid, C_out, K, stride;
    // Expansion PW (omitted if expand=1)
    float *pw1_w;    // [C_mid, C_in, 1, 1]
    BatchNorm *bn1;
    // Depthwise
    float *dw_w;     // [C_mid, K, K]
    float *dw_b;     // [C_mid]
    BatchNorm *bn2;
    // SE (reduction=4 in EfficientNet)
    SEBlock *se;
    // Projection PW
    float *pw2_w;    // [C_out, C_mid, 1, 1]
    BatchNorm *bn3;
    int skip;        // 1 if identity residual applies
} MBConvBlock;

void mbconv_forward(
    MBConvBlock *blk,
    const float *X,    // [N, C_in, H, W]
    float       *Y,    // [N, C_out, OH, OW]
    int N, int H, int W, int training) {

    int OH = conv_output_size(H, blk->K, blk->stride, blk->K/2, 1);
    int OW = conv_output_size(W, blk->K, blk->stride, blk->K/2, 1);

    float *buf = malloc((size_t)N * blk->C_mid * H * W * sizeof(float));

    // 1. Expansion PW (skip if expand=1, i.e., C_mid == C_in)
    if (blk->C_mid != blk->C_in) {
        pointwise_conv2d(X, blk->pw1_w, NULL, buf, N, blk->C_in, blk->C_mid, H, W);
        // BN + SiLU (Swish): x * sigmoid(x)
        float *xhat = malloc((size_t)N * blk->C_mid * H * W * sizeof(float));
        if (training)
            bn_forward_train(buf, blk->bn1->gamma, blk->bn1->beta, buf,
                             blk->bn1->mean, blk->bn1->var, xhat,
                             blk->bn1->run_mean, blk->bn1->run_var, 0.1f,
                             N, blk->C_mid, H, W);
        swish_forward(buf, N * blk->C_mid * H * W);
        free(xhat);
    } else {
        memcpy(buf, X, (size_t)N * blk->C_in * H * W * sizeof(float));
    }

    // 2. Depthwise conv
    float *dw_out = malloc((size_t)N * blk->C_mid * OH * OW * sizeof(float));
    depthwise_conv2d_forward(buf, blk->dw_w, blk->dw_b, dw_out,
                             N, blk->C_mid, H, W,
                             blk->K, blk->K, OH, OW,
                             blk->stride, blk->K / 2);
    // BN + SiLU
    float *xhat2 = malloc((size_t)N * blk->C_mid * OH * OW * sizeof(float));
    if (training)
        bn_forward_train(dw_out, blk->bn2->gamma, blk->bn2->beta, dw_out,
                         blk->bn2->mean, blk->bn2->var, xhat2,
                         blk->bn2->run_mean, blk->bn2->run_var, 0.1f,
                         N, blk->C_mid, OH, OW);
    swish_forward(dw_out, N * blk->C_mid * OH * OW);
    free(xhat2); free(buf);

    // 3. SE block
    float *gap_buf = malloc(N * blk->C_mid * sizeof(float));
    float *fc1_buf = malloc(N * (blk->C_mid / 4) * sizeof(float));
    float *fc2_buf = malloc(N * blk->C_mid * sizeof(float));
    se_forward(blk->se, dw_out, dw_out, gap_buf, fc1_buf, fc2_buf,
               N, blk->C_mid, OH, OW);
    free(gap_buf); free(fc1_buf); free(fc2_buf);

    // 4. Projection PW → BN (no activation after last PW in MBConv)
    pointwise_conv2d(dw_out, blk->pw2_w, NULL, Y, N, blk->C_mid, blk->C_out, OH, OW);
    float *xhat3 = malloc((size_t)N * blk->C_out * OH * OW * sizeof(float));
    if (training)
        bn_forward_train(Y, blk->bn3->gamma, blk->bn3->beta, Y,
                         blk->bn3->mean, blk->bn3->var, xhat3,
                         blk->bn3->run_mean, blk->bn3->run_var, 0.1f,
                         N, blk->C_out, OH, OW);
    free(xhat3); free(dw_out);

    // 5. Identity residual (skip connection) — only when dimensions match
    if (blk->skip) {
        int sz = N * blk->C_out * OH * OW;
        for (int i = 0; i < sz; i++) Y[i] += X[i];
    }
}

// SiLU (Swish): x * sigmoid(x) — activation used in EfficientNet
void swish_forward(float *X, int size) {
    for (int i = 0; i < size; i++)
        X[i] = X[i] * (1.0f / (1.0f + expf(-X[i])));
}
```

---

## 4. Compound Scaling in Code

```c
typedef struct {
    float depth_mult;   // multiply #layers per stage
    float width_mult;   // multiply channel counts
    int   resolution;   // input image size
} EfficientNetConfig;

// Round channels to nearest multiple of 8 (hardware alignment)
int round_channels(float c, float width_mult) {
    int ch = (int)(c * width_mult);
    int divisor = 8;
    int new_ch = ((ch + divisor/2) / divisor) * divisor;
    if (new_ch < (int)(0.9f * ch)) new_ch += divisor;
    return new_ch;
}

// Round layer count (depth) to at least 1
int round_depth(float d, float depth_mult) {
    return (int)ceilf(d * depth_mult);
}

static const EfficientNetConfig EFFICIENTNET_CONFIGS[] = {
    {1.0f, 1.0f, 224},  // B0
    {1.1f, 1.0f, 240},  // B1
    {1.2f, 1.1f, 260},  // B2
    {1.4f, 1.2f, 300},  // B3
    {1.8f, 1.4f, 380},  // B4
    {2.2f, 1.6f, 456},  // B5
    {2.6f, 1.8f, 528},  // B6
    {3.1f, 2.0f, 600},  // B7
};

// Build EfficientNet-Bx model
void efficientnet_build(int b, int *total_params) {
    EfficientNetConfig cfg = EFFICIENTNET_CONFIGS[b];

    // Stage channel widths (B0 baseline)
    int ch[] = {32, 16, 24, 40, 80, 112, 192, 320, 1280};
    int depths[] = {1, 2, 2, 3, 3, 4, 1};  // per-stage repeats

    printf("EfficientNet-B%d config (resolution=%d):\n", b, cfg.resolution);
    *total_params = 0;
    for (int s = 0; s < 7; s++) {
        int Ci = round_channels(ch[s],   cfg.width_mult);
        int Co = round_channels(ch[s+1], cfg.width_mult);
        int d  = round_depth(depths[s],  cfg.depth_mult);
        printf("  Stage %d: C_in=%3d → C_out=%3d, %d blocks\n", s, Ci, Co, d);
    }
}
```

---

## 5. FLOP and Parameter Comparison

```
Network         Params  FLOPs   ImageNet top-1
ResNet-50       25.6M   4.1B    76.1%
MobileNetV2     3.4M    300M    72.0%
EfficientNet-B0 5.3M    390M    77.1%  ← better than ResNet-50 with 5× fewer params
EfficientNet-B1 7.8M    700M    79.1%
EfficientNet-B4 19.3M   4.2B    82.9%  ← same FLOPs as ResNet-50, +6.8% accuracy
EfficientNet-B7 66M     37B     84.3%

EfficientNet-B0 vs MobileNetV2 (similar FLOPs):
  MobileNetV2:  72.0%  (no SE, no Swish)
  B0:           77.1%  (+5.1% from SE + Swish + NAS architecture)
```

---

## 6. Stochastic Depth (Drop Path)

EfficientNet uses stochastic depth during training — randomly skip entire MBConv blocks:

```c
// drop_path: randomly zero out a sample in the batch (per-sample, per-block)
void drop_path_forward(float *X, int N, int C, int H, int W,
                       float drop_rate, int training) {
    if (!training || drop_rate == 0.0f) return;

    // Each sample in the batch is kept or dropped independently
    for (int n = 0; n < N; n++) {
        if ((float)rand() / RAND_MAX < drop_rate) {
            // Zero out this sample's contribution (skip the block)
            float *sample = X + (long)n * C * H * W;
            memset(sample, 0, (long)C * H * W * sizeof(float));
        }
    }
    // Scale surviving samples to maintain expected value
    float scale = 1.0f / (1.0f - drop_rate);
    for (int i = 0; i < N * C * H * W; i++) X[i] *= scale;
}
```

---

## Key Takeaways

- **Compound scaling**: scale depth × width × resolution jointly with a fixed ratio — outperforms any single-dimension scaling
- EfficientNet-B0 found by NAS: 7 stages of MBConv blocks with SE attention and SiLU activation
- **MBConv**: expand → DW(K×K) → SE → project — combines depthwise separable conv with channel attention
- EfficientNet-B0 achieves ResNet-50 accuracy with 5× fewer parameters by combining all previous innovations
- SiLU/Swish activation (`x × σ(x)`) outperforms ReLU by ~0.5-1% accuracy on deep networks

---

**Next**: [20. Modern CNN Benchmark](./20_Modern_CNN_Benchmark.md) — Compare LeNet, VGG, ResNet-20, and EfficientNet-B0 on CIFAR-10/100: accuracy, speed, and memory tradeoffs.
