# Lesson 16 — ResNet and Skip Connections (per-lesson exercise)

Prerequisites: L08 (convolution), L11 (batch norm), L13 (LeNet/AlexNet).

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

ResNet (2015) made very deep networks trainable by introducing **skip connections** — a "highway" path that bypasses each residual block. The block computes a residual function $F(x)$ and the layer output is $x + F(x)$, so the gradient can flow through the identity path even if $F$'s gradient is small.

This single idea is why every modern transformer also uses skip connections (after every attention and FFN sub-layer).

---

## Exercise 16.1 — Basic Residual Block (forward)

**Difficulty**: ★★★

### Problem

A basic residual block (ResNet-18/34 style) is:

```
input x
  ├─→ conv 3×3 → BN → ReLU → conv 3×3 → BN ─┐
  └────────────────────────────────────────→ + → ReLU → output
```

Implement `residual_block_forward(input, weights, output, ...)` that:

1. Runs the two conv+BN+ReLU sub-layers on the input.
2. Adds the original input element-wise to the second sub-layer's output.
3. Applies the final ReLU.

```c
void residual_block_forward(const float *x,
                            const float *W1, const float *W2,
                            const float *bn1_g, const float *bn1_b,
                            const float *bn2_g, const float *bn2_b,
                            float *out,
                            int N, int C, int H, int W) {
    /* Allocate intermediate buffers */
    /* TODO:
         h1 = conv(x, W1)
         h1 = batch_norm(h1, bn1_g, bn1_b)
         h1 = relu(h1)
         h2 = conv(h1, W2)
         h2 = batch_norm(h2, bn2_g, bn2_b)
         out = relu(h2 + x)
    */
    (void)x; (void)W1; (void)W2;
    (void)bn1_g; (void)bn1_b; (void)bn2_g; (void)bn2_b;
    (void)out; (void)N; (void)C; (void)H; (void)W;
}
```

Key constraint: the input shape must match the second sub-layer's output shape so they can be added. Convolutions in the block use `padding = 1` to preserve spatial size.

---

## Exercise 16.2 — Downsampling Block

**Difficulty**: ★★

When a residual block changes channel count or spatial resolution, the skip connection cannot be a literal identity — it needs a **projection** (typically a 1×1 convolution) to match shapes:

```
input x (C_in × H × W)
  ├─→ conv 3×3 stride=2 → BN → ReLU → conv 3×3 → BN ─┐
  └─→ conv 1×1 stride=2 → BN ────────────────────────→ + → ReLU → output (C_out × H/2 × W/2)
```

Implement this variant. The 1×1 conv has `kH = kW = 1` and (typically) no bias; it serves only to align channel counts and spatial resolution.

This is the second-most-common pattern in ResNet — the "feature size doubling" stage between successive resolution levels.

---

## Exercise 16.3 — Build ResNet-18 — Bonus

**Difficulty**: ★★★

Stack the basic and downsampling blocks per the ResNet-18 recipe:

| Stage | Output size | # blocks | Block type | Output channels |
|-------|-------------|----------|------------|-----------------|
| stem | 56 × 56 | — | conv 7×7 stride=2 → maxpool 3×3 stride=2 | 64 |
| 1 | 56 × 56 | 2 | basic | 64 |
| 2 | 28 × 28 | 2 | downsample, basic | 128 |
| 3 | 14 × 14 | 2 | downsample, basic | 256 |
| 4 | 7 × 7 | 2 | downsample, basic | 512 |
| head | 1 × 1 | 1 | global avg pool → fc 1000 | 1000 |

Total: 18 weight layers (2 in each of 8 blocks + stem + final FC) — the source of the "-18" in the name.

Verify the output is shape `[N, 1000]` for an `[N, 3, 224, 224]` input. Param count should be ~11M.

---

## Exercise 16.4 — Why Skip Connections Help — Conceptual

**Difficulty**: ★

Build TWO mini networks: a 30-layer "plain" CNN (each layer is conv + BN + ReLU, no skip) and a 30-layer ResNet (15 residual blocks). Train both on a synthetic regression problem (input → desired output is a fixed nonlinear map).

You should see:

- The plain network's training loss plateaus or even diverges around layer depth 20+ — gradients vanish/explode.
- The ResNet trains stably to convergence.

This is the experimental result the original ResNet paper reproduced on ImageNet and that made deep networks (50+, 100+, 200+ layers) practical.
