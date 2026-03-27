# 11. Batch Normalization

**Previous**: [Pooling Layers](./10_Pooling_Layers.md) | **Next**: [Data Pipeline for Images](./12_Data_Pipeline_Images.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement the BN forward pass in both training and inference (eval) modes
2. Maintain running mean and variance for inference
3. Derive and implement the BN backward pass through the mean and variance computation
4. Apply the gamma (scale) and beta (shift) learnable parameters
5. Explain why BN accelerates training and where it is placed relative to ReLU

---

## 1. The Batch Normalization Formula

Given a mini-batch of activations over the batch and spatial dimensions:

```
Input:  X  [N, C, H, W]

For each channel c:
  mean[c]    = (1/M) Σ X[n,c,h,w]       where M = N*H*W
  var[c]     = (1/M) Σ (X[n,c,h,w] - mean[c])²
  X_hat[c]   = (X - mean[c]) / sqrt(var[c] + ε)
  Y          = gamma[c] * X_hat + beta[c]
```

`gamma` and `beta` are learnable parameters (initialized to 1 and 0).

---

## 2. Training Mode Forward Pass

```c
#define BN_EPS 1e-5f

// bn_forward_train: normalize over (N, H, W) per channel
// Stores mean, var, X_hat for use in backward pass
void bn_forward_train(
    const float *X,       // [N, C, H, W]
    const float *gamma,   // [C]
    const float *beta,    // [C]
    float       *Y,       // [N, C, H, W]
    float       *mean,    // [C] — saved for backward
    float       *var,     // [C] — saved for backward
    float       *X_hat,   // [N, C, H, W] — saved for backward
    float       *run_mean, // [C] — updated in-place (EMA)
    float       *run_var,  // [C] — updated in-place (EMA)
    float       momentum,  // typically 0.1
    int N, int C, int H, int W) {

    int M = N * H * W;  // number of elements per channel

    for (int c = 0; c < C; c++) {
        // Compute batch mean
        float m = 0.0f;
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            m += NCHW(X, N, C, H, W, n, c, h, w);
        m /= M;
        mean[c] = m;

        // Compute batch variance
        float v = 0.0f;
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float diff = NCHW(X, N, C, H, W, n, c, h, w) - m;
            v += diff * diff;
        }
        v /= M;
        var[c] = v;

        float inv_std = 1.0f / sqrtf(v + BN_EPS);

        // Normalize and scale
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float x_norm = (NCHW(X, N, C, H, W, n, c, h, w) - m) * inv_std;
            NCHW(X_hat, N, C, H, W, n, c, h, w) = x_norm;
            NCHW(Y,     N, C, H, W, n, c, h, w) = gamma[c] * x_norm + beta[c];
        }

        // Update running statistics (exponential moving average)
        run_mean[c] = (1.0f - momentum) * run_mean[c] + momentum * m;
        run_var[c]  = (1.0f - momentum) * run_var[c]  + momentum * v;
    }
}
```

---

## 3. Inference (Eval) Mode Forward Pass

During inference, use the fixed running statistics — no batch dependency:

```c
// bn_forward_eval: use stored running mean/var (no random batch dependency)
void bn_forward_eval(
    const float *X,
    const float *gamma,
    const float *beta,
    float       *Y,
    const float *run_mean,  // [C] — frozen
    const float *run_var,   // [C] — frozen
    int N, int C, int H, int W) {

    for (int c = 0; c < C; c++) {
        float inv_std = 1.0f / sqrtf(run_var[c] + BN_EPS);
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float x_norm = (NCHW(X, N, C, H, W, n, c, h, w) - run_mean[c]) * inv_std;
            NCHW(Y, N, C, H, W, n, c, h, w) = gamma[c] * x_norm + beta[c];
        }
    }
}
```

**Training vs Eval modes**:

```
Training:  mean/var computed per batch → noisy regularization (helps generalization)
Eval:      mean/var from EMA → deterministic output (required for deployment)

Bug: forgetting to switch to eval mode inflates test variance (variable BN stats per forward call)
```

---

## 4. Backward Pass Derivation

The BN backward is the trickiest part because the mean and variance are functions of X.

Define:
```
M      = N*H*W           (batch spatial size)
σ      = sqrt(var + ε)   (standard deviation)
x_hat  = (x - μ) / σ    (normalized input, saved in forward)
Y      = γ * x_hat + β
```

The gradients are (standard derivation via chain rule through μ and σ²):

```
dγ  = Σ (dY * x_hat)                            [C]
dβ  = Σ dY                                       [C]
dx_hat = dY * γ                                  [N,C,H,W]

dX = (1/M*σ) * [ M*dx_hat
                 - Σ(dx_hat)
                 - x_hat * Σ(dx_hat * x_hat) ]
```

Implementation:

```c
// bn_backward: compute dX, dgamma, dbeta from dY
void bn_backward(
    const float *dY,     // [N, C, H, W]
    const float *X_hat,  // [N, C, H, W] — saved from forward
    const float *gamma,  // [C]
    const float *var,    // [C] — batch variance, saved from forward
    float       *dX,     // [N, C, H, W]
    float       *dgamma, // [C]
    float       *dbeta,  // [C]
    int N, int C, int H, int W) {

    int M = N * H * W;

    for (int c = 0; c < C; c++) {
        float inv_std = 1.0f / sqrtf(var[c] + BN_EPS);

        // dγ and dβ
        float dg = 0.0f, db_val = 0.0f;
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float dy   = NCHW(dY,    N, C, H, W, n, c, h, w);
            float xhat = NCHW(X_hat, N, C, H, W, n, c, h, w);
            dg     += dy * xhat;
            db_val += dy;
        }
        dgamma[c] += dg;
        dbeta[c]  += db_val;

        // dx_hat = dY * gamma[c]
        // Sum1 = Σ dx_hat,  Sum2 = Σ (dx_hat * x_hat)
        float sum1 = 0.0f, sum2 = 0.0f;
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float dxhat = NCHW(dY, N, C, H, W, n, c, h, w) * gamma[c];
            float xhat  = NCHW(X_hat, N, C, H, W, n, c, h, w);
            sum1 += dxhat;
            sum2 += dxhat * xhat;
        }

        // dX = (inv_std / M) * [M*dx_hat - sum1 - x_hat*sum2]
        float scale = inv_std / M;
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float dxhat = NCHW(dY,    N, C, H, W, n, c, h, w) * gamma[c];
            float xhat  = NCHW(X_hat, N, C, H, W, n, c, h, w);
            NCHW(dX, N, C, H, W, n, c, h, w) =
                scale * (M * dxhat - sum1 - xhat * sum2);
        }
    }
}
```

---

## 5. BN Layer Struct

Encapsulating all BN state:

```c
typedef struct {
    int C;
    float *gamma, *beta;       // learnable params [C]
    float *dgamma, *dbeta;     // gradients [C]
    float *run_mean, *run_var; // running statistics [C]
    float *mean, *var;         // batch statistics [C] — saved for backward
    float *X_hat;              // normalized input [N,C,H,W] — saved for backward
    float  momentum;           // EMA decay (default 0.1)
    int    N, H, W;            // saved from last forward call
} BatchNorm;

BatchNorm *bn_create(int C, int N_max, int H_max, int W_max) {
    BatchNorm *bn = calloc(1, sizeof(BatchNorm));
    bn->C = C;
    bn->gamma    = malloc(C * sizeof(float));
    bn->beta     = calloc(C, sizeof(float));  // zero init
    bn->dgamma   = calloc(C, sizeof(float));
    bn->dbeta    = calloc(C, sizeof(float));
    bn->run_mean = calloc(C, sizeof(float));
    bn->run_var  = malloc(C * sizeof(float));
    bn->mean     = malloc(C * sizeof(float));
    bn->var      = malloc(C * sizeof(float));
    bn->X_hat    = malloc(N_max * C * H_max * W_max * sizeof(float));
    bn->momentum = 0.1f;

    // Initialize gamma to 1.0
    for (int c = 0; c < C; c++) {
        bn->gamma[c]   = 1.0f;
        bn->run_var[c] = 1.0f;  // avoid div-by-zero on first eval call
    }
    return bn;
}

void bn_free(BatchNorm *bn) {
    free(bn->gamma); free(bn->beta);
    free(bn->dgamma); free(bn->dbeta);
    free(bn->run_mean); free(bn->run_var);
    free(bn->mean); free(bn->var); free(bn->X_hat);
    free(bn);
}
```

---

## 6. BN Placement

The canonical placement in modern CNNs:

```
Original (2015 paper): Conv → BN → ReLU
Modern practice:       Conv → BN → ReLU  (most common, used in ResNet)
Pre-activation:        BN → ReLU → Conv  (used in ResNet-v2)

Note: Transformer uses LayerNorm (per-sample, per-channel),
      not BN (per-channel across batch).
      BN requires batch size > 1; LN works for batch size = 1.
```

---

## 7. Why BN Works

```
Without BN:
  - Layer outputs drift toward large values as parameters update
  - Later layers see shifting distributions (internal covariate shift)
  - Need careful initialization and low learning rates

With BN:
  - Each layer's output is renormalized to ~N(0,1) per channel
  - Gradients are well-conditioned regardless of parameter scale
  - Acts as regularizer (mini-batch statistics add noise)
  - Allows 10× larger learning rates → faster convergence
```

---

## Key Takeaways

- **Training mode**: compute mean/var per batch; update running EMA; save x_hat for backward
- **Eval mode**: use frozen running mean/var — never compute statistics from test data
- **Backward** is complex: gradient flows through the normalization itself (via μ and σ), not just through the gamma/beta scaling
- `dgamma = Σ(dY * x_hat)` and `dbeta = Σ(dY)` are straightforward; `dX` involves two correction terms that account for the batch-level constraints
- BN makes training dramatically more stable — virtually every modern CNN uses it

---

**Next**: [12. Data Pipeline for Images](./12_Data_Pipeline_Images.md) — Loading images with STB, converting between NHWC/NCHW, and data augmentation (flip, crop, normalize).
