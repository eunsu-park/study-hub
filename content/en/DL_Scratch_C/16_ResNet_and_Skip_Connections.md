# 16. ResNet and Skip Connections

**Previous**: [VGG and Deep Networks](./15_VGG_and_Deep_Networks.md) | **Next**: [Depthwise Separable Convolution](./17_Depthwise_Separable_Conv.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why skip connections solve the vanishing gradient problem
2. Implement the residual block with identity and projection shortcuts in C
3. Build ResNet-20 for CIFAR-10 using the building blocks from previous lessons
4. Derive the backward pass through a residual connection (gradient splitting)
5. Compare ResNet-20 vs VGG-style networks on depth vs accuracy

---

## 1. The Degradation Problem

He et al. (2015) observed a counter-intuitive phenomenon:

```
Adding more layers to a plain network makes training accuracy WORSE:
  Plain-20:  8.82% error on CIFAR-10
  Plain-56: 13.47% error on CIFAR-10   ← deeper = worse!

This is not overfitting (training error also gets worse).
The network simply cannot learn to propagate identity through many layers.
```

The fix: teach the network to learn **residuals** (corrections) rather than complete transformations.

---

## 2. Residual Block

Instead of learning `H(x)`, learn `F(x) = H(x) - x` (the residual):

```
Forward:
  y = F(x, {W}) + x
  where F is 2 or 3 stacked conv+BN+ReLU layers

For identity:    learn F(x) = 0  (easy — just zero weights)
For correction:  learn small F(x) = Δ

Backward:
  ∂L/∂x = ∂L/∂y × (∂F/∂x + I)
                       ↑ identity matrix
         = ∂L/∂y × ∂F/∂x  +  ∂L/∂y

The gradient always receives at least ∂L/∂y via the shortcut path.
No vanishing gradient even across 100+ layers.
```

---

## 3. ResNet Building Blocks

### Basic Block (ResNet-18/34/20)

Two 3×3 convolutions:

```
x → Conv(3×3) → BN → ReLU → Conv(3×3) → BN → (+x) → ReLU → output
↓___________________________________________↑  (skip)
```

```c
typedef struct {
    // Main path: conv1 → bn1 → relu → conv2 → bn2
    float *conv1_w, *conv1_b;   // [C_out, C_in, 3, 3]
    BatchNorm *bn1;
    float *conv2_w, *conv2_b;   // [C_out, C_out, 3, 3]
    BatchNorm *bn2;
    // Projection shortcut (when stride>1 or C_in != C_out)
    float *proj_w, *proj_b;     // [C_out, C_in, 1, 1]  (NULL if identity)
    BatchNorm *proj_bn;         // BN on projection
    int C_in, C_out, stride;
} ResidualBlock;

// Forward pass
void resblock_forward(
    ResidualBlock *blk,
    const float   *X,      // [N, C_in, H, W]
    float         *Y,      // [N, C_out, OH, OW]
    float         *buf1,   // [N, C_out, OH, OW] — intermediate
    float         *buf_sc, // [N, C_out, OH, OW] — shortcut branch
    float         *xhat1, *xhat2, *xhat_sc,  // BN saved states
    int N, int H, int W,
    int training) {

    int OH = conv_output_size(H, 3, blk->stride, 1, 1);
    int OW = conv_output_size(W, 3, blk->stride, 1, 1);

    // Main path: conv1 → BN → ReLU
    conv2d_im2col(X, N, blk->C_in, H, W,
                  blk->conv1_w, blk->C_out, 3, 3,
                  buf1, OH, OW, blk->stride, 1, 1);
    add_bias_chw(buf1, blk->conv1_b, N, blk->C_out, OH, OW);
    // BN
    bn_forward_train(buf1, blk->bn1->gamma, blk->bn1->beta, buf1,
                     blk->bn1->mean, blk->bn1->var, xhat1,
                     blk->bn1->run_mean, blk->bn1->run_var,
                     0.1f, N, blk->C_out, OH, OW);
    relu_forward(buf1, N * blk->C_out * OH * OW);

    // Main path: conv2 → BN (no ReLU yet — add shortcut first)
    conv2d_im2col(buf1, N, blk->C_out, OH, OW,
                  blk->conv2_w, blk->C_out, 3, 3,
                  Y, OH, OW, 1, 1, 1);
    add_bias_chw(Y, blk->conv2_b, N, blk->C_out, OH, OW);
    bn_forward_train(Y, blk->bn2->gamma, blk->bn2->beta, Y,
                     blk->bn2->mean, blk->bn2->var, xhat2,
                     blk->bn2->run_mean, blk->bn2->run_var,
                     0.1f, N, blk->C_out, OH, OW);

    // Shortcut branch
    if (blk->proj_w) {
        // Projection: 1×1 conv + BN (when stride > 1 or channel mismatch)
        conv2d_im2col(X, N, blk->C_in, H, W,
                      blk->proj_w, blk->C_out, 1, 1,
                      buf_sc, OH, OW, blk->stride, 0, 1);
        add_bias_chw(buf_sc, blk->proj_b, N, blk->C_out, OH, OW);
        bn_forward_train(buf_sc, blk->proj_bn->gamma, blk->proj_bn->beta, buf_sc,
                         blk->proj_bn->mean, blk->proj_bn->var, xhat_sc,
                         blk->proj_bn->run_mean, blk->proj_bn->run_var,
                         0.1f, N, blk->C_out, OH, OW);
    } else {
        // Identity shortcut: copy X directly
        memcpy(buf_sc, X, N * blk->C_out * OH * OW * sizeof(float));
    }

    // Add shortcut to main path, then ReLU
    int sz = N * blk->C_out * OH * OW;
    for (int i = 0; i < sz; i++) Y[i] += buf_sc[i];
    relu_forward(Y, sz);
}
```

---

## 4. Backward Through a Skip Connection

The gradient splits at the addition:

```
∂L/∂x = ∂L/∂y × ∂y/∂x
       = ∂L/∂y × (∂F/∂x + ∂shortcut/∂x)

For identity shortcut (∂shortcut/∂x = I):
  dX_total = dX_from_main_path + dY   (gradient from skip path passes through unchanged)

For projection shortcut:
  dX_total = dX_from_main_path + dX_from_projection_conv
```

```c
// resblock_backward: compute dX from dY
void resblock_backward(
    ResidualBlock *blk,
    const float   *X, const float *buf1,  // saved from forward
    const float   *dY,
    float         *dX,    // [N, C_in, H, W]
    float         *dW1, float *db1,
    float         *dW2, float *db2,
    float         *dWp, float *dbp,   // projection (if any)
    float         *dgamma1, *dbeta1,
    float         *dgamma2, *dbeta2,
    float         *dgammap, *dbetap,
    const float   *xhat1, *xhat2, *xhat_sc,
    int N, int H, int W) {

    int OH = conv_output_size(H, 3, blk->stride, 1, 1);
    int OW = conv_output_size(W, 3, blk->stride, 1, 1);

    // ---- Shortcut gradient ----
    float *dX_skip = calloc(N * blk->C_in * H * W, sizeof(float));
    if (blk->proj_w) {
        // dY → BN backward → 1×1 conv backward
        float *dY_proj = malloc(N * blk->C_out * OH * OW * sizeof(float));
        memcpy(dY_proj, dY, N * blk->C_out * OH * OW * sizeof(float));
        float *d_proj_in = malloc(N * blk->C_out * OH * OW * sizeof(float));
        bn_backward(dY_proj, xhat_sc, blk->proj_bn->gamma, blk->proj_bn->var,
                    d_proj_in, dgammap, dbetap, N, blk->C_out, OH, OW);
        float *dcol = malloc(N * OH * OW * blk->C_in * 1 * 1 * sizeof(float));
        input_backward(d_proj_in, blk->proj_w, dcol, dX_skip,
                       N, blk->C_in, H, W, blk->C_out, 1, 1, OH, OW,
                       blk->stride, 0, 1);
        // weight gradient
        float *col = malloc(N * OH * OW * blk->C_in * sizeof(float));
        im2col(X, N, blk->C_in, H, W, 1, 1, OH, OW, blk->stride, 0, 1, col);
        weight_backward(col, d_proj_in, dWp, N * OH * OW, blk->C_in, blk->C_out);
        free(dY_proj); free(d_proj_in); free(dcol); free(col);
    } else {
        // Identity: skip gradient is just dY
        memcpy(dX_skip, dY, N * blk->C_in * H * W * sizeof(float));
    }

    // ---- Main path backward (BN2 → Conv2 → ReLU → BN1 → Conv1) ----
    float *dY_main = malloc(N * blk->C_out * OH * OW * sizeof(float));
    memcpy(dY_main, dY, N * blk->C_out * OH * OW * sizeof(float));

    // BN2 backward
    float *d_conv2_out = malloc(N * blk->C_out * OH * OW * sizeof(float));
    bn_backward(dY_main, xhat2, blk->bn2->gamma, blk->bn2->var,
                d_conv2_out, dgamma2, dbeta2, N, blk->C_out, OH, OW);

    // Conv2 backward
    float *d_relu1 = calloc(N * blk->C_out * OH * OW, sizeof(float));
    float *dcol2 = malloc(N * OH * OW * blk->C_out * 9 * sizeof(float));
    input_backward(d_conv2_out, blk->conv2_w, dcol2, d_relu1,
                   N, blk->C_out, OH, OW, blk->C_out, 3, 3, OH, OW, 1, 1, 1);
    float *col2 = malloc(N * OH * OW * blk->C_out * 9 * sizeof(float));
    im2col(buf1, N, blk->C_out, OH, OW, 3, 3, OH, OW, 1, 1, 1, col2);
    weight_backward(col2, d_conv2_out, dW2, N * OH * OW, blk->C_out * 9, blk->C_out);
    free(dcol2); free(col2); free(dY_main); free(d_conv2_out);

    // ReLU backward (buf1 = pre-ReLU output after relu_forward, stored as Y>0 mask)
    relu_backward(d_relu1, buf1, N * blk->C_out * OH * OW);

    // BN1 backward
    float *d_conv1_out = malloc(N * blk->C_out * OH * OW * sizeof(float));
    bn_backward(d_relu1, xhat1, blk->bn1->gamma, blk->bn1->var,
                d_conv1_out, dgamma1, dbeta1, N, blk->C_out, OH, OW);
    free(d_relu1);

    // Conv1 backward → produces dX_main
    float *dX_main = calloc(N * blk->C_in * H * W, sizeof(float));
    float *dcol1 = malloc(N * OH * OW * blk->C_in * 9 * sizeof(float));
    input_backward(d_conv1_out, blk->conv1_w, dcol1, dX_main,
                   N, blk->C_in, H, W, blk->C_out, 3, 3, OH, OW,
                   blk->stride, 1, 1);
    float *col1 = malloc(N * OH * OW * blk->C_in * 9 * sizeof(float));
    im2col(X, N, blk->C_in, H, W, 3, 3, OH, OW, blk->stride, 1, 1, col1);
    weight_backward(col1, d_conv1_out, dW1, N * OH * OW, blk->C_in * 9, blk->C_out);
    free(d_conv1_out); free(dcol1); free(col1);

    // Total dX = main + skip
    int in_sz = N * blk->C_in * H * W;
    for (int i = 0; i < in_sz; i++) dX[i] = dX_main[i] + dX_skip[i];
    free(dX_main); free(dX_skip);
}
```

---

## 5. ResNet-20 for CIFAR-10

He et al. (2016) proposed a CIFAR-specific ResNet:

```
Input: [N, 3, 32, 32]

Stem: Conv(3→16, 3×3, p=1) → BN → ReLU    [N, 16, 32, 32]

Stage 1: 3 × ResBlock(16→16,  s=1)         [N, 16, 32, 32]
Stage 2: 3 × ResBlock(16→32,  s=2 first)   [N, 32, 16, 16]
Stage 3: 3 × ResBlock(32→64,  s=2 first)   [N, 64, 8, 8]

GAP:  [N, 64]
FC:   64 → 10

Parameters:
  Stem:           (3×3×3+1)×16    = 448
  Stage 1: 3×[ (3×3×16+1)×16×2 ] = 3×4,640 = 13,920
  Stage 2: 3×[ (3×3×32+1)×32×2 ] = 3×18,496 = 55,488
    + projection: (1×1×16+1)×32  = 544
  Stage 3: 3×[ (3×3×64+1)×64×2 ] = 3×73,856 = 221,568
    + projection: (1×1×32+1)×64  = 2,112
  FC:  (64+1)×10                 = 650
  Total:                         ≈ 270K parameters

Performance:
  ResNet-20:  91.25% on CIFAR-10
  ResNet-56:  93.03% on CIFAR-10
  VGG-16:     ~93%   (but 522× more parameters!)
```

---

## 6. Identity vs Projection Shortcut

When to use each:

```
Identity shortcut (X → output directly):
  Condition: stride=1 AND C_in == C_out
  Cost: zero parameters

Projection shortcut (1×1 conv + BN):
  Condition: stride > 1 OR C_in != C_out
  Cost: C_in × C_out × 1×1 parameters
  Purpose: match spatial size (stride) and channel count

ResNet option A: pad identity with zeros (no extra params)
ResNet option B: projection on downsampling only  ← original paper (best practice)
ResNet option C: all shortcuts are projections
```

---

## 7. Gradient Flow: Plain vs ResNet

```
50-layer plain network:
  Gradient at layer 1 ≈ product of 50 Jacobians
  Magnitude: (0.9)^50 ≈ 0.005 — factor 200× smaller than output

50-layer ResNet:
  ∂L/∂x_0 = Σ_k ∂L/∂x_k   (sum over all skip paths to the loss)
  At least one direct path always carries full gradient magnitude
  → No exponential decay
```

---

## Key Takeaways

- **Residual block**: `y = F(x) + x` — network learns the correction F, not the full mapping H
- **Skip connection backward**: gradient splits at the `+` node; the skip path carries `∂L/∂y` unchanged through all layers → eliminates vanishing gradient
- **Projection shortcut**: 1×1 conv when dimensions change (stride or channels); identity otherwise
- **ResNet-20** achieves 91.25% on CIFAR-10 with only 270K parameters vs VGG's 138M — 512× more efficient
- The key insight: making identity the default (zero residual = identity) is easier to optimize than learning identity through a stack of convolutions

---

**Next**: [17. Depthwise Separable Convolution](./17_Depthwise_Separable_Conv.md) — Factorize standard convolution into depthwise + pointwise steps, achieving ~8× FLOP reduction — the foundation of MobileNet.
