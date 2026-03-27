# 15. VGG and Deep Networks

**Previous**: [Training CNN on CIFAR-10](./14_Training_CNN_CIFAR10.md) | **Next**: [ResNet and Skip Connections](./16_ResNet_and_Skip_Connections.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the VGG design philosophy: small filters stacked deep
2. Count parameters in VGG-16 and explain where memory is spent
3. Demonstrate why very deep networks suffer from vanishing gradients
4. Implement VGG-style convolutional blocks in C
5. Measure the receptive field growth with depth

---

## 1. VGG Design Philosophy

VGG (Simonyan & Zisserman, 2014) showed that depth — using a stack of 3×3 convolutions — is the critical factor for performance. The key insight:

```
Two 3×3 convolutions stacked have the same receptive field as one 5×5:
  Receptive field:  3×3 → 5×5 (after 2 layers) → 7×7 (after 3 layers)

But 2 × (3×3×C²) params = 18C² vs one 5×5×C² = 25C²  (28% fewer params)
And 2 layers of 3×3 apply ReLU twice → more nonlinearity
```

### VGG-16 Architecture

```
Input: [N, 3, 224, 224]

Block 1: Conv(3→64,3×3,p=1)×2  → MaxPool(2×2)  → [N, 64, 112, 112]
Block 2: Conv(64→128,3×3,p=1)×2 → MaxPool(2×2)  → [N, 128, 56, 56]
Block 3: Conv(128→256,3×3,p=1)×3 → MaxPool(2×2) → [N, 256, 28, 28]
Block 4: Conv(256→512,3×3,p=1)×3 → MaxPool(2×2) → [N, 512, 14, 14]
Block 5: Conv(512→512,3×3,p=1)×3 → MaxPool(2×2) → [N, 512, 7, 7]
Flatten: [N, 25088]
FC1:  25088 → 4096 + ReLU + Dropout(0.5)
FC2:  4096  → 4096 + ReLU + Dropout(0.5)
FC3:  4096  → 1000
```

Parameter count:

```
Block 1: (3×3×3+1)×64 + (3×3×64+1)×64         =    1,792 +    36,928 =    38,720
Block 2: (3×3×64+1)×128 + (3×3×128+1)×128      =   73,856 +   147,584 =   221,440
Block 3: 3×(3×3×256+1)×256 + …                 =  590,080 + 1,180,160 + 1,180,160 = 2,950,400 (approx)
Block 4: 3×(3×3×512+1)×512                     =  2,359,808 × 3 = 7,079,424 (approx)
Block 5: same                                   =  7,079,424
FC1:   25088×4096 + 4096                        = 102,764,544
FC2:   4096×4096 + 4096                         =  16,781,312
FC3:   4096×1000 + 1000                         =   4,097,000
Total:                                           ≈ 138M parameters

Breakdown:
  Conv layers:  ~14.7M   (11% of total)
  FC layers:   ~123.6M   (89% of total) ← most params are in FC!
```

**Key insight**: VGG spends 89% of its parameters in three FC layers. ResNet eliminates these with GAP.

---

## 2. VGG Block Implementation

```c
// VGG convolutional block: Conv → BN → ReLU (repeated `n_convs` times)
typedef struct {
    int n_convs;
    // Conv weights for each sub-layer
    float **conv_w;   // [n_convs] each [C_out, C_in, 3, 3]
    float **conv_b;   // [n_convs] each [C_out]
    // BN params (modern VGG adds BN)
    BatchNorm **bn;   // [n_convs]
    int C_in, C_out;
} VGGBlock;

// VGGBlock forward pass
void vgg_block_forward(
    VGGBlock    *blk,
    const float *X,     // [N, C_in, H, W]
    float       *Y,     // [N, C_out, H, W]
    float       **bufs, // intermediate buffers [n_convs]
    int N, int H, int W,
    int training) {

    const float *cur_in = X;
    int C_cur = blk->C_in;

    for (int i = 0; i < blk->n_convs; i++) {
        float *cur_out = (i < blk->n_convs - 1) ? bufs[i] : Y;
        int C_out = blk->C_out;

        // Conv(3×3, pad=1) — keeps H×W constant
        int OH = conv_output_size(H, 3, 1, 1, 1);  // same size
        int OW = conv_output_size(W, 3, 1, 1, 1);
        conv2d_im2col(cur_in, N, C_cur, H, W,
                      blk->conv_w[i], C_out, 3, 3,
                      cur_out, OH, OW, 1, 1, 1);
        add_bias_chw(cur_out, blk->conv_b[i], N, C_out, H, W);

        // BN
        float *xhat = malloc(N * C_out * H * W * sizeof(float));
        bn_forward_train(cur_out, blk->bn[i]->gamma, blk->bn[i]->beta,
                         cur_out, blk->bn[i]->mean, blk->bn[i]->var, xhat,
                         blk->bn[i]->run_mean, blk->bn[i]->run_var,
                         0.1f, N, C_out, H, W);
        free(xhat);

        // ReLU in-place
        relu_forward(cur_out, N * C_out * H * W);

        cur_in = cur_out;
        C_cur  = C_out;
    }
}
```

---

## 3. Receptive Field Growth

With same-padding 3×3 convolutions and stride=1, the receptive field (RF) grows linearly:

```
Layer depth:     1    2    3    4    5    6    7    8    9   10   13
RF (stride=1):   3    5    7    9   11   13   15   17   19   21   27

After each MaxPool(stride=2), effective RF doubles:
  Block 1 (2 convs): RF = 5×5  → after MaxPool: covers 10×10 of input
  Block 2 (2 convs): RF = 14×14 effective in input space
  Block 3 (3 convs): RF covers ~62×62 of the original 224×224 input
  Block 5: RF covers ~212×212 ≈ nearly the full image
```

Computing RF:

```c
int receptive_field(int n_layers, int kernel, int stride_per_layer) {
    int rf = 1;
    for (int i = 0; i < n_layers; i++)
        rf = rf + (kernel - 1) * stride_per_layer;
    return rf;
}
// n_convs=13 (VGG-16 conv layers), kernel=3, effective_stride=1 each
// RF = 1 + 12*2 = 25 (ignoring pooling strides — add them separately)
```

---

## 4. Vanishing Gradients in Deep Networks

As depth increases without skip connections, backpropagated gradients shrink:

```
Gradient through L layers of tanh:
  ∂L/∂x_0 = ∏_{i=1}^{L} (∂x_i/∂x_{i-1}) = ∏ (W_i × tanh'(x_i))

tanh'(x) ≤ 1.0, typical weights ≈ 0.5 → product shrinks as 0.5^L

L=10:  0.5^10 ≈ 0.001   (gradient 1000× smaller)
L=20:  0.5^20 ≈ 1e-6    (gradient 1 million× smaller)
```

**ReLU** partially solves this — gradient is either 0 or 1 (no shrinkage for active units):

```
∂ReLU/∂x = 1 if x > 0
            0 if x ≤ 0  ← dead neuron problem
```

**But ReLU doesn't fully solve the problem** — very deep networks (>20 layers) still degrade.

```
VGG-16 test accuracy on ImageNet:
  VGG-11 (8 conv): 70.4% top-1
  VGG-13 (10 conv): 71.3% top-1
  VGG-16 (13 conv): 74.4% top-1  ← sweet spot
  VGG-19 (16 conv): 74.5% top-1  ← barely improves! depth limit reached
  ResNet-50:        76.1% top-1   ← skip connections break through
```

---

## 5. Gradient Flow Monitoring

```c
// Compute L2 norm of a gradient tensor (diagnostic)
float grad_norm(const float *grad, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) sum += grad[i] * grad[i];
    return sqrtf(sum);
}

// Monitor gradient norms during training (per layer)
void print_gradient_norms(VGGNet *model) {
    printf("Gradient norms per block:\n");
    for (int blk = 0; blk < 5; blk++) {
        float norm = grad_norm(model->blocks[blk].conv_w[0],
                               model->blocks[blk].C_out * model->blocks[blk].C_in * 9);
        printf("  Block %d: ||dW|| = %.6f\n", blk + 1, norm);
    }
}
// Expected output (healthy training):
// Block 5: ||dW|| = 0.002341
// Block 4: ||dW|| = 0.001987
// Block 3: ||dW|| = 0.000812   ← smaller — some vanishing
// Block 2: ||dW|| = 0.000213   ← further shrinkage
// Block 1: ||dW|| = 0.000047   ← severe vanishing without skip connections
```

---

## 6. VGG vs AlexNet vs ResNet Summary

```
                AlexNet     VGG-16      ResNet-50
Year            2012        2014        2015
Depth           8 layers    16 layers   50 layers
Params          60M         138M        25M
ImageNet top-1  57.1%       74.4%       76.1%
Skip conn       No          No          Yes
FC layers       3 large     3 large     None (GAP)

Memory (forward, batch=1, FP32):
  AlexNet:   ~4 MB activations
  VGG-16:  ~500 MB activations (dominates GPU memory!)
  ResNet-50: ~100 MB activations
```

VGG's activation memory (500MB) is why it was replaced by ResNet in production — not accuracy.

---

## Key Takeaways

- **VGG design rule**: use only 3×3 convolutions; double channels at each MaxPool
- Two 3×3 convs = one 5×5 receptive field with 28% fewer parameters and an extra nonlinearity
- **89% of VGG's 138M parameters are in the FC layers** — GAP eliminates these entirely
- Deep networks (>20 layers) hit a gradient degradation wall without skip connections — VGG-19 is barely better than VGG-16
- ReLU helps gradient flow vs tanh but doesn't solve deep network degradation — that required residual connections

---

**Next**: [16. ResNet and Skip Connections](./16_ResNet_and_Skip_Connections.md) — Residual blocks, identity and projection shortcuts, why skip connections solve the vanishing gradient problem, and implementing ResNet-20 for CIFAR-10.
