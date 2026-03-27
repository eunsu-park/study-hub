# 13. LeNet and AlexNet

**Previous**: [Data Pipeline for Images](./12_Data_Pipeline_Images.md) | **Next**: [Training CNN on CIFAR-10](./14_Training_CNN_CIFAR10.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement LeNet-5 from scratch using the conv, pool, and activation primitives built in previous lessons
2. Describe the architectural innovations that AlexNet introduced in 2012
3. Build AlexNet-style layers (LRN is optional; focus on conv + ReLU + pool stacking)
4. Count parameters in each layer and understand memory requirements
5. Trace a forward pass through both networks with concrete shapes

---

## 1. LeNet-5 (LeCun, 1998)

### Architecture

```
Input: [N, 1, 32, 32]   (grayscale, padded from 28×28 MNIST)

C1:  Conv(1→6,  5×5, s=1, p=0)  → [N, 6, 28, 28]   + Tanh
S2:  AvgPool(2×2, s=2)           → [N, 6, 14, 14]
C3:  Conv(6→16, 5×5, s=1, p=0)  → [N, 16, 10, 10]  + Tanh
S4:  AvgPool(2×2, s=2)           → [N, 16, 5, 5]
C5:  Conv(16→120, 5×5, s=1, p=0)→ [N, 120, 1, 1]   + Tanh
F6:  FC(120 → 84)                + Tanh
OUT: FC(84 → 10)                 + Softmax
```

Parameter count:

```
C1:  (5×5×1  + 1) × 6    =   156
C3:  (5×5×6  + 1) × 16   = 2,416
C5:  (5×5×16 + 1) × 120  = 48,120
F6:  (120 + 1) × 84       = 10,164
OUT: (84 + 1) × 10         =    850
Total:                      ~61,706 parameters
```

### Implementation

```c
// LeNet-5 forward pass
// Input: [N, 1, 32, 32]   Output: logits [N, 10]
void lenet5_forward(
    const float *X,          // [N, 1, 32, 32]
    float       *logits,     // [N, 10]
    LeNet5Weights *wt,       // weights struct
    LeNet5Buffers *buf,      // intermediate buffers
    int N) {

    // C1: Conv(1→6, 5×5) + Tanh
    int OH1 = 28, OW1 = 28;
    conv2d_naive(X, N, 1, 32, 32,
                 wt->c1_w, 6, 5, 5,
                 buf->c1_out, OH1, OW1, 1, 0, 1);
    add_bias_chw(buf->c1_out, wt->c1_b, N, 6, OH1, OW1);
    apply_tanh(buf->c1_out, N * 6 * OH1 * OW1);

    // S2: AvgPool(2×2, s=2)
    avg_pool2d_forward(buf->c1_out, buf->s2_out,
                       N, 6, OH1, OW1, 2, 2, 14, 14, 2, 0);

    // C3: Conv(6→16, 5×5) + Tanh
    int OH3 = 10, OW3 = 10;
    conv2d_naive(buf->s2_out, N, 6, 14, 14,
                 wt->c3_w, 16, 5, 5,
                 buf->c3_out, OH3, OW3, 1, 0, 1);
    add_bias_chw(buf->c3_out, wt->c3_b, N, 16, OH3, OW3);
    apply_tanh(buf->c3_out, N * 16 * OH3 * OW3);

    // S4: AvgPool(2×2, s=2)
    avg_pool2d_forward(buf->c3_out, buf->s4_out,
                       N, 16, OH3, OW3, 2, 2, 5, 5, 2, 0);

    // C5: Conv(16→120, 5×5) → [N, 120, 1, 1]
    conv2d_naive(buf->s4_out, N, 16, 5, 5,
                 wt->c5_w, 120, 5, 5,
                 buf->c5_out, 1, 1, 1, 0, 1);
    add_bias_chw(buf->c5_out, wt->c5_b, N, 120, 1, 1);
    apply_tanh(buf->c5_out, N * 120);

    // F6: FC(120 → 84) + Tanh
    // buf->c5_out shape after GAP: [N, 120]
    matmul(buf->c5_out, wt->f6_w, buf->f6_out, N, 120, 84);
    add_bias_vec(buf->f6_out, wt->f6_b, N, 84);
    apply_tanh(buf->f6_out, N * 84);

    // OUT: FC(84 → 10)
    matmul(buf->f6_out, wt->out_w, logits, N, 84, 10);
    add_bias_vec(logits, wt->out_b, N, 10);
    // Softmax applied during loss computation
}

// Helper: add bias to CHW output (broadcast across H,W)
void add_bias_chw(float *X, const float *b, int N, int C, int H, int W) {
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++) {
        float bv = b[c];
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            NCHW(X, N, C, H, W, n, c, h, w) += bv;
    }
}
```

---

## 2. AlexNet (Krizhevsky et al., 2012)

AlexNet's key innovations over LeNet:

```
1. ReLU activation (instead of tanh) — avoids vanishing gradient
2. MaxPool (instead of average pool) — sharper feature selection
3. Dropout (p=0.5) in FC layers — reduces overfitting
4. Data augmentation (crops, flips, color jitter)
5. GPU training on two GTX 580s — demonstrated GPU DL scalability
```

### Architecture (adapted for CIFAR-10, 32×32 input)

Original AlexNet was designed for ImageNet 224×224. For CIFAR-10, we adapt the first layers:

```
Input: [N, 3, 32, 32]

L1: Conv(3→64,   3×3, s=1, p=1)  → [N, 64, 32, 32]  + ReLU
L2: Conv(64→192, 3×3, s=1, p=1)  → [N, 192, 32, 32] + ReLU + MaxPool(2×2,s=2)
    → [N, 192, 16, 16]
L3: Conv(192→384,3×3, s=1, p=1)  → [N, 384, 16, 16] + ReLU
L4: Conv(384→256,3×3, s=1, p=1)  → [N, 256, 16, 16] + ReLU
L5: Conv(256→256,3×3, s=1, p=1)  → [N, 256, 16, 16] + ReLU + MaxPool(2×2,s=2)
    → [N, 256, 8, 8]
GAP: GlobalAvgPool                 → [N, 256]
FC1: FC(256 → 256) + ReLU + Dropout(0.5)
FC2: FC(256 → 10)
```

Parameter count:

```
L1: (3×3×3   + 1) × 64  =   1,792
L2: (3×3×64  + 1) × 192 = 110,784
L3: (3×3×192 + 1) × 384 = 663,936
L4: (3×3×384 + 1) × 256 = 884,992
L5: (3×3×256 + 1) × 256 = 590,080
FC1: (256 + 1) × 256    =  65,792
FC2: (256 + 1) × 10     =   2,570
Total:                    ~2.3M parameters
```

### Dropout

```c
// dropout_forward: apply inverted dropout (scale during training)
// Stores mask for backward pass
void dropout_forward(
    float   *X,    // in-place
    uint8_t *mask, // [size] — 1 = kept, 0 = dropped
    int size, float p,   // drop probability
    int training) {

    if (!training) return;  // no dropout during eval

    for (int i = 0; i < size; i++) {
        mask[i] = ((float)rand() / RAND_MAX) > p ? 1 : 0;
        X[i] *= mask[i] / (1.0f - p);  // inverted dropout scaling
    }
}

// dropout_backward: zero out gradients for dropped units
void dropout_backward(
    float         *dX,
    const uint8_t *mask,
    int size, float p) {

    for (int i = 0; i < size; i++)
        dX[i] *= mask[i] / (1.0f - p);
}
```

### ReLU

```c
// relu_forward: in-place
void relu_forward(float *X, int size) {
    for (int i = 0; i < size; i++)
        X[i] = X[i] > 0.0f ? X[i] : 0.0f;
}

// relu_backward: pass gradient only where X > 0
// X here = pre-ReLU activations or equivalently output > 0
void relu_backward(float *dX, const float *Y, int size) {
    for (int i = 0; i < size; i++)
        dX[i] *= (Y[i] > 0.0f) ? 1.0f : 0.0f;
}
```

---

## 3. Weight Initialization

Proper initialization prevents gradient vanishing/explosion:

```c
#include <math.h>

// He initialization: suitable for ReLU networks
// std = sqrt(2 / fan_in)
void he_init(float *W, int fan_in, int fan_out) {
    float std = sqrtf(2.0f / fan_in);
    for (int i = 0; i < fan_in * fan_out; i++)
        W[i] = randn() * std;
}

// Xavier/Glorot initialization: suitable for tanh/sigmoid networks
// std = sqrt(2 / (fan_in + fan_out))
void xavier_init(float *W, int fan_in, int fan_out) {
    float std = sqrtf(2.0f / (fan_in + fan_out));
    for (int i = 0; i < fan_in * fan_out; i++)
        W[i] = randn() * std;
}

// Box-Muller transform for standard normal samples
float randn(void) {
    float u1 = (float)(rand() + 1) / ((float)RAND_MAX + 1);
    float u2 = (float)rand()       / ((float)RAND_MAX + 1);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

// Initialize LeNet-5 (Xavier, tanh networks)
void lenet5_init_weights(LeNet5Weights *wt) {
    xavier_init(wt->c1_w, 1 * 5 * 5, 6);
    xavier_init(wt->c3_w, 6 * 5 * 5, 16);
    xavier_init(wt->c5_w, 16 * 5 * 5, 120);
    xavier_init(wt->f6_w, 120, 84);
    xavier_init(wt->out_w, 84, 10);
    // biases to zero
}
```

---

## 4. Architecture Comparison

```
          LeNet-5          AlexNet (adapted)
Input     1×32×32          3×32×32
Params    ~62K             ~2.3M
Activation Tanh            ReLU
Pooling   AvgPool          MaxPool
Dropout   No               Yes (FC layers)
BN        No               No (predates BN)
Augment   None             Flip + Crop + Jitter

CIFAR-10 accuracy (approx):
  LeNet-5:         ~68%
  AlexNet (small): ~85%
  ResNet-20:       ~92%
```

---

## 5. Sanity Check: Shape Trace

```c
static void shape_trace_lenet5(void) {
    int N = 2;
    // [2, 1, 32, 32] → C1 → [2, 6, 28, 28]
    printf("After C1:  [%d, 6, 28, 28]\n", N);
    // → S2 → [2, 6, 14, 14]
    printf("After S2:  [%d, 6, 14, 14]\n", N);
    // → C3 → [2, 16, 10, 10]
    printf("After C3:  [%d, 16, 10, 10]\n", N);
    // → S4 → [2, 16, 5, 5]
    printf("After S4:  [%d, 16, 5, 5]\n", N);
    // → C5 → [2, 120, 1, 1]
    printf("After C5:  [%d, 120, 1, 1]\n", N);
    // → F6 → [2, 84]
    printf("After F6:  [%d, 84]\n", N);
    // → OUT → [2, 10]
    printf("After OUT: [%d, 10]\n", N);
}
```

---

## Key Takeaways

- **LeNet-5**: 62K params, tanh + avg-pool, suited for MNIST/small grayscale images
- **AlexNet**: 2.3M params, ReLU + max-pool + dropout — the 2012 ImageNet breakthrough
- **He initialization** (`std = sqrt(2/fan_in)`) is the right choice for ReLU networks; **Xavier** for tanh
- Dropout during training, bypassed during eval — remember to save and use the mask in backward
- Both networks reuse the same building blocks: conv, pool, activation, matmul — from the previous lessons

---

**Next**: [14. Training CNN on CIFAR-10](./14_Training_CNN_CIFAR10.md) — End-to-end training: data loader + forward pass + cross-entropy loss + backward + SGD optimizer + accuracy measurement.
