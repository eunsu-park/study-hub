# Block 3 — Modern CNN Architectures (L15–L20)

Prerequisites: L15 (VGG), L16 (ResNet), L17 (MobileNet/depthwise), L18 (SE blocks), L19 (EfficientNet compound scaling), L20 (architecture search).

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

---

## Exercise 3.1 — Count VGG-16 Parameters

**Difficulty**: ★

### Problem

Write a C program that computes the **total parameter count** of VGG-16 (as published, with 1000-class ImageNet head) using the formula:

```
Conv layer params = (K * K * C_in + 1) * C_out       (with bias)
FC layer params   = (C_in + 1) * C_out               (with bias)
```

VGG-16 layer spec:

| Group | Layers | C_in | C_out | K |
|-------|--------|------|-------|---|
| 1 | Conv, Conv | 3, 64 | 64, 64 | 3 |
| 2 | Conv, Conv | 64, 128 | 128, 128 | 3 |
| 3 | Conv, Conv, Conv | 128, 256, 256 | 256, 256, 256 | 3 |
| 4 | Conv, Conv, Conv | 256, 512, 512 | 512, 512, 512 | 3 |
| 5 | Conv, Conv, Conv | 512, 512, 512 | 512, 512, 512 | 3 |
| FC1 | FC | 512*7*7 | 4096 | — |
| FC2 | FC | 4096 | 4096 | — |
| FC3 | FC | 4096 | 1000 | — |

### Starter Code

```c
#include <stdio.h>
#include <stdint.h>

typedef struct { int K; int C_in; int C_out; } ConvSpec;
typedef struct { int C_in; int C_out; }         FCSpec;

int64_t conv_params(int K, int C_in, int C_out) {
    /* TODO: return (K*K*C_in + 1) * C_out */
    return 0;
}

int64_t fc_params(int C_in, int C_out) {
    /* TODO: return (C_in + 1) * C_out */
    return 0;
}

int main(void) {
    /* VGG-16 architecture */
    ConvSpec convs[] = {
        {3,   3,  64}, {3,  64,  64},
        {3,  64, 128}, {3, 128, 128},
        {3, 128, 256}, {3, 256, 256}, {3, 256, 256},
        {3, 256, 512}, {3, 512, 512}, {3, 512, 512},
        {3, 512, 512}, {3, 512, 512}, {3, 512, 512},
    };
    FCSpec fcs[] = {
        {512*7*7, 4096}, {4096, 4096}, {4096, 1000}
    };

    int64_t total = 0;
    int n_conv = sizeof(convs)/sizeof(convs[0]);
    int n_fc   = sizeof(fcs)/sizeof(fcs[0]);

    printf("Convolutional layers:\n");
    for (int i = 0; i < n_conv; i++) {
        int64_t p = conv_params(convs[i].K, convs[i].C_in, convs[i].C_out);
        printf("  Conv %2d: %10lld params\n", i+1, (long long)p);
        total += p;
    }
    printf("Fully-connected layers:\n");
    for (int i = 0; i < n_fc; i++) {
        int64_t p = fc_params(fcs[i].C_in, fcs[i].C_out);
        printf("  FC   %2d: %10lld params\n", i+1, (long long)p);
        total += p;
    }
    printf("\nTotal: %lld params (expected ~138,357,544)\n", (long long)total);
    return 0;
}
```

### Test Cases

| Component | Expected params |
|-----------|----------------|
| First conv (3→64, K=3) | 1,792 |
| FC1 (25088→4096) | 102,764,544 |
| **Total** | **138,357,544** |

### Hints

1. Be careful with integer overflow — use `int64_t` for the running total.
2. FC1 dominates: 512×7×7 = 25,088 inputs × 4,096 outputs ≈ 103M params.
3. Don't forget the `+1` for biases in both formulas.

### Solution Approach

Plug the layer dimensions into the two formulas and sum. The point is to understand that most VGG-16 parameters live in the fully-connected layers, not the convolutions — a key motivation for global average pooling in later architectures.

---

## Exercise 3.2 — ResNet Projection Shortcut

**Difficulty**: ★★

### Problem

Implement the **projection shortcut** used in ResNet when the number of channels changes between residual blocks (and/or the spatial resolution is halved by stride=2).

The projection is: `shortcut = conv1x1(x, stride=s) -> BN -> (no activation)`

Then the output is: `out = relu(F(x) + shortcut(x))`

Implement `projection_shortcut(const float *x, int C_in, int H, int W, int C_out, int stride, const float *W_proj, float *out)` where:
- `W_proj` is `[C_out, C_in, 1, 1]` (1×1 conv weights, no bias for simplicity)
- `stride` is 1 or 2

Output shape: `[C_out, H/stride, W/stride]`.

### Starter Code

```c
#include <stdio.h>
#include <string.h>
#include <math.h>

/*
 * 1x1 conv with stride, no padding, no bias.
 * Input:  [C_in][H][W]
 * Weight: [C_out][C_in]   (1x1, so spatial dims collapsed)
 * Output: [C_out][H_out][W_out]  where H_out = H/stride
 */
void conv1x1_stride(const float *x, int C_in, int H, int W,
                    const float *W_proj, int C_out, int stride,
                    float *out) {
    int H_out = H / stride;
    int W_out = W / stride;
    /* TODO: for each (c_out, h, w) compute dot product over C_in channels
             sampling input at (h*stride, w*stride) */
}

/*
 * Minimal BN (no learnable params, uses batch statistics for simplicity)
 * Normalizes a CHW tensor across spatial dims per channel.
 */
void bn_spatial(float *x, int C, int H, int W) {
    /* TODO: for each channel c, compute mean and var over H*W,
             then normalize in-place */
}

int main(void) {
    /* Input: C_in=4, H=4, W=4 */
    int C_in=4, H=4, W=4, C_out=8, stride=2;
    int H_out = H/stride, W_out = W/stride;

    float x[4*4*4];
    for (int i = 0; i < C_in*H*W; i++) x[i] = (float)i * 0.01f;

    /* Identity projection weights (for testing) */
    float W_proj[8*4]; /* [C_out][C_in] */
    memset(W_proj, 0, sizeof(W_proj));
    /* First 4 output channels copy input channels 1:1 */
    for (int i = 0; i < 4; i++) W_proj[i*C_in + i] = 1.0f;

    float out[8*2*2];
    conv1x1_stride(x, C_in, H, W, W_proj, C_out, stride, out);

    printf("Output shape: [%d][%d][%d]\n", C_out, H_out, W_out);
    printf("out[0][0][0] = %.4f (expected = x[0][0][0] = 0.0000)\n", out[0]);
    printf("out[0][0][1] = %.4f (expected = x[0][0][2] = 0.0200)\n", out[1]);
    /* Channel 4 (zero weights) should be 0 */
    printf("out[4][0][0] = %.4f (expected 0.0000)\n", out[4*H_out*W_out]);
    return 0;
}
```

### Test Cases

- With identity `W_proj` (first C_in output channels copy input) and stride=2, output at position `(c, h, w)` equals input at `(c, h*2, w*2)`.
- With stride=1, output shape is `[C_out, H, W]`.
- With stride=2, output shape is `[C_out, H/2, W/2]`.

### Hints

1. The 1×1 conv with stride just samples the input at every `stride`-th position.
2. After the conv, apply BN before adding to the main path.
3. The addition `F(x) + shortcut(x)` requires that both tensors have the same shape — that is the whole point of the projection.

### Solution Approach

The 1×1 conv loops over output positions `(c_out, h, w)` and accumulates `sum_{c_in} W[c_out][c_in] * x[c_in][h*stride][w*stride]`. No spatial kernel loop is needed because K=1. This is a key building block for ResNet-50/101/152 where channel dimensions triple at each stage boundary.

---

## Exercise 3.3 — FLOP Ratio: Depthwise+Pointwise vs Standard Conv

**Difficulty**: ★

### Problem

Compute the FLOP ratio of a depthwise separable convolution (DW + PW) versus a standard convolution for:
- Kernel size K = 3
- Input channels C_in = 64
- Output channels C_out = 128
- Spatial size H × W = 1 × 1 (FLOPs per output spatial position)

Standard conv FLOPs per output position: `2 * K^2 * C_in * C_out`
DW conv FLOPs per output position: `2 * K^2 * C_in`
PW conv FLOPs per output position: `2 * C_in * C_out`

Print the ratio `(DW+PW) / standard` and confirm it matches the theoretical formula `1/C_out + 1/K^2`.

### Starter Code

```c
#include <stdio.h>

int main(void) {
    int K = 3, C_in = 64, C_out = 128;

    /* FLOPs per output position (factor of 2 for multiply-add) */
    long long std_flops = 0; /* TODO */
    long long dw_flops  = 0; /* TODO */
    long long pw_flops  = 0; /* TODO */
    long long sep_flops = dw_flops + pw_flops;

    printf("Standard conv:       %lld FLOPs\n", std_flops);
    printf("Depthwise:           %lld FLOPs\n", dw_flops);
    printf("Pointwise:           %lld FLOPs\n", pw_flops);
    printf("Depthwise+Pointwise: %lld FLOPs\n", sep_flops);

    double ratio    = (double)sep_flops / std_flops;
    double theory   = 1.0 / C_out + 1.0 / (K * K);
    printf("\nMeasured ratio:  %.6f\n", ratio);
    printf("Theoretical 1/C_out + 1/K^2 = %.6f\n", theory);
    /* They should match */
    return 0;
}
```

### Test Cases

For K=3, C_in=64, C_out=128:
- Standard FLOPs: 2 × 9 × 64 × 128 = 147,456
- DW FLOPs: 2 × 9 × 64 = 1,152
- PW FLOPs: 2 × 64 × 128 = 16,384
- Ratio ≈ 0.1181 ≈ 1/128 + 1/9

### Hints

1. The "2" factor accounts for multiply and add as separate operations.
2. The formula `1/C_out + 1/K^2` holds exactly because it comes from dividing by `2*K^2*C_in*C_out`.
3. For typical values (C_out=256, K=3), the ratio is ~0.116 — about 8-9× fewer FLOPs.

### Solution Approach

Substitute numbers into the three formulas, compute the ratio, and verify algebraically. This exercise builds intuition for why MobileNet is so much cheaper than VGG despite similar accuracy.

---

## Exercise 3.4 — SE Block (Squeeze-and-Excitation)

**Difficulty**: ★★

### Problem

Implement the Squeeze-and-Excitation (SE) block:

1. **Squeeze**: Global average pooling over spatial dims → `[C]` vector.
2. **Excitation**: Two FC layers with ReLU and Sigmoid: `s = sigmoid(W2 * relu(W1 * z))` where `W1: [C/r, C]`, `W2: [C, C/r]`, `r=16`.
3. **Scale**: Multiply each channel of the input by the corresponding scalar in `s`.

### Starter Code

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define R 16  /* reduction ratio */

void relu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) if (x[i] < 0) x[i] = 0;
}

void sigmoid_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = 1.0f / (1.0f + expf(-x[i]));
}

void fc(const float *in, int in_dim, const float *W, int out_dim, float *out) {
    for (int o = 0; o < out_dim; o++) {
        out[o] = 0;
        for (int i = 0; i < in_dim; i++) out[o] += W[o*in_dim + i] * in[i];
    }
}

/*
 * SE block in-place.
 * x layout: [C][H][W]
 * W1: [C/R][C], W2: [C][C/R]
 */
void se_block(float *x, int C, int H, int W,
              const float *W1, const float *W2) {
    int Cr = C / R;

    /* Step 1: Global Average Pooling -> z[C] */
    float z[C];
    /* TODO */

    /* Step 2: W1 -> relu */
    float h[Cr];
    /* TODO */

    /* Step 3: W2 -> sigmoid -> s[C] */
    float s[C];
    /* TODO */

    /* Step 4: Scale each channel */
    /* TODO */
}

int main(void) {
    int C=32, H=8, W=8;
    int Cr = C/R;

    float *x  = calloc(C*H*W, sizeof(float));
    float *W1 = calloc(Cr*C,  sizeof(float));
    float *W2 = calloc(C*Cr,  sizeof(float));

    /* Initialize x: channel c gets constant value c+1 */
    for (int c = 0; c < C; c++)
        for (int hw = 0; hw < H*W; hw++)
            x[c*H*W + hw] = (float)(c + 1);

    /* Identity-ish W1, W2 for testing: W1 and W2 are zeros */
    /* With zero weights, excitation = sigmoid(0) = 0.5 for all channels */
    /* => all channels scaled by 0.5 */

    se_block(x, C, H, W, W1, W2);

    /* All values should be halved */
    printf("x[0][0] = %.4f (expected 0.5 = 1 * 0.5)\n", x[0]);
    printf("x[1*H*W] = %.4f (expected 1.0 = 2 * 0.5)\n", x[H*W]);

    free(x); free(W1); free(W2);
    return 0;
}
```

### Test Cases

- With zero weights `W1` and `W2`: excitation values are all `sigmoid(0) = 0.5`, so every element of `x` is halved.
- After SE with non-trivial weights, channels with higher squeeze values are amplified more.
- The output shape must equal the input shape `[C][H][W]`.

### Hints

1. GAP: for channel `c`, average all `H*W` values of `x[c*H*W ... c*H*W+H*W-1]`.
2. The excitation network is just two FC layers: reduce from C to C/R, then expand back.
3. Scaling: `x[c*H*W + hw] *= s[c]` for all spatial positions.

### Solution Approach

Four sequential steps: GAP, FC+ReLU, FC+Sigmoid, channel-wise multiply. The power of SE blocks is that they learn which channels are most informative for a given input — a form of learned attention over channels. The reduction ratio r=16 keeps the parameter overhead small (2*C*C/16 per block).

---

## Exercise 3.5 — EfficientNet Compound Scaling

**Difficulty**: ★★

### Problem

EfficientNet scales a baseline network by:
- Depth multiplier:     `d = α^φ`
- Width multiplier:     `w = β^φ`
- Resolution multiplier: `r = γ^φ`

Subject to the constraint: `α * β^2 * γ^2 ≈ 2` (so that total FLOPs ≈ 2^φ).

For EfficientNet-B0→B7, the baseline coefficients found by NAS are:
- `α = 1.2`, `β = 1.1`, `γ = 1.15`
- B0: φ=0, B1: φ=0.5, B2: φ=1, B3: φ=2, B4: φ=3.5, B5: φ=5, B6: φ=6, B7: φ=7

Implement `compute_scaling(double alpha, double beta, double gamma, double phi)` and print depth, width, and resolution multipliers for B0 through B7. Also verify the constraint `α * β^2 * γ^2`.

### Starter Code

```c
#include <stdio.h>
#include <math.h>

typedef struct {
    double depth;
    double width;
    double resolution;
} Scaling;

Scaling compute_scaling(double alpha, double beta, double gamma, double phi) {
    Scaling s;
    /* TODO: s.depth = pow(alpha, phi), etc. */
    return s;
}

int main(void) {
    double alpha = 1.2, beta = 1.1, gamma_r = 1.15;

    /* Verify constraint */
    double constraint = alpha * beta * beta * gamma_r * gamma_r;
    printf("Constraint alpha*beta^2*gamma^2 = %.4f (should be ~2.0)\n", constraint);

    double phis[] = {0, 0.5, 1, 2, 3.5, 5, 6, 7};
    int n = sizeof(phis)/sizeof(phis[0]);

    printf("\n%-8s %-10s %-10s %-12s\n", "Model", "Depth", "Width", "Resolution");
    printf("%-8s %-10s %-10s %-12s\n", "-----", "-----", "-----", "----------");
    for (int i = 0; i < n; i++) {
        Scaling s = compute_scaling(alpha, beta, gamma_r, phis[i]);
        printf("B%-7d %-10.3f %-10.3f %-12.3f\n",
               i, s.depth, s.width, s.resolution);
    }

    /* Base resolution for EfficientNet-B0 is 224. Print actual resolutions. */
    printf("\nActual input resolutions (base=224):\n");
    int base_res = 224;
    for (int i = 0; i < n; i++) {
        Scaling s = compute_scaling(alpha, beta, gamma_r, phis[i]);
        int res = (int)(base_res * s.resolution);
        printf("B%d: %d x %d\n", i, res, res);
    }
    return 0;
}
```

### Test Cases

| Model | φ | Expected depth | Expected width | Expected resolution |
|-------|---|---------------|---------------|---------------------|
| B0 | 0 | 1.000 | 1.000 | 1.000 |
| B1 | 0.5 | 1.095 | 1.049 | 1.072 |
| B7 | 7 | 6.183 | 2.100 | 3.631 |

EfficientNet-B7 input resolution: `224 * 3.631 ≈ 600×600` (actual: 600×600).

### Hints

1. `pow(x, phi)` from `<math.h>` computes x^φ.
2. When φ=0, all multipliers are 1 by definition (x^0 = 1).
3. The constraint `α * β^2 * γ^2 ≈ 2` ensures FLOP complexity scales as 2^φ, not faster.

### Solution Approach

Each multiplier is an independent power-of-base computation. The interesting insight is that resolution is squared in the FLOP constraint (FLOPs scale as r^2 for spatial dimensions), which is why γ < α — you get diminishing returns by increasing resolution alone.
