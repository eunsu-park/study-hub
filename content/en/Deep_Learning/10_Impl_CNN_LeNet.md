# 10. CNN (LeNet)

[Previous: Transfer Learning](./09_Transfer_Learning.md) | [Next: VGG](./11_Impl_VGG.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the historical significance of LeNet-5 and describe the architectural innovations it introduced to CNN design.
2. Calculate the output spatial dimensions of a convolutional layer given input size, kernel size, padding, and stride.
3. Describe the role of average pooling (subsampling) in LeNet-5 and explain how it reduces spatial resolution.
4. Implement LeNet-5 from scratch in PyTorch, correctly specifying each layer's parameters and activation functions.
5. Train LeNet-5 on the MNIST dataset and evaluate classification performance using accuracy and loss metrics.
6. Analyze the learned convolutional filters and feature maps to develop intuition about what each layer represents.

---

## Theory & Principles

LeNet-5 (LeCun et al. 1998) is the original convolutional network. Re-implementing it today is partly a history lesson and partly a useful baseline — every concept that later networks add (ReLU, BN, residuals, attention) can be measured against this starting point. The architecture is also small enough that you can derive its parameter count and per-layer activation shapes by hand, which is the right exercise for understanding any CNN.

This section covers:

- **A.** LeNet-5's design choices and what 1998 looked like
- **B.** Why subsampling (pooling) was non-trivial then and now
- **C.** The math of parameter counting in convolutional networks
- **D.** What replaced LeNet's choices in modern CNNs (and why)

### A. LeNet-5 Design

LeNet-5 was designed for `32 x 32` grayscale digits (MNIST). Its layer plan:

```
Input  32 x 32 x 1
C1     6 maps,  5x5 conv, no padding   ->  28 x 28 x 6
S2     2x2 average pool, stride 2      ->  14 x 14 x 6
C3     16 maps, 5x5 conv               ->  10 x 10 x 16
S4     2x2 average pool, stride 2      ->   5 x  5 x 16
C5     120 maps, 5x5 conv (=fully connected on 5x5 input)
F6     84 fully-connected units
Out    10 RBF units (softmax-equivalent)
```

Activations were **tanh** (not ReLU; ReLU would not become standard until 2010). The optimizer was a careful hand-tuned SGD with momentum. The total parameter count is around 60,000 — three to four orders of magnitude less than even the smallest modern CNN.

### B. Subsampling (Pooling) Then and Now

LeNet's S2 and S4 layers did *trainable* subsampling: each `2x2` patch was averaged, then multiplied by a learnable scalar and added to a learnable bias before tanh. Modern architectures simplified this to plain max- or average-pooling with no parameters, because experiments showed the trainable subsampling provided little benefit and cost extra parameters. This decision saved nothing in 1998 (the network was tiny anyway) but mattered a lot later, when deep CNNs needed every parameter to do real work.

The deeper question — what should subsampling do — has multiple modern answers:

- **Max pooling**: keep the strongest activation in each patch (robust to noise).
- **Average pooling**: keep the mean (smoother, less spiky).
- **Strided convolution**: learn the downsampling filter (most expressive, used in ResNet).
- **Attention pooling / global average pooling**: aggregate spatially in a learned weighted way (used in modern classifiers and ViT's CLS token).

### C. Parameter Counting

For a convolutional layer with `C_in` input channels, `C_out` output channels, kernel `K x K`, the parameter count is:

```
weights = K * K * C_in * C_out
biases  = C_out
total   = K * K * C_in * C_out + C_out
```

For LeNet's C1: `5 * 5 * 1 * 6 + 6 = 156`. C3: `5 * 5 * 6 * 16 + 16 = 2,416`. C5: `5 * 5 * 16 * 120 + 120 = 48,120` — the C5 layer alone holds ~80% of the network's weights, because at that point the tensor is small spatially but has many channels and connects to many output channels.

This is a recurring pattern: the parameter count is dominated by the layers that mix many channels, not by the early layers with few channels.

### D. What Replaced LeNet's Choices

| LeNet (1998) | Modern Replacement | Why |
|--------------|--------------------|-----|
| tanh activation | ReLU / GELU | Non-saturating, avoids vanishing gradients |
| Trainable pooling | Max / strided conv | Simpler, no benefit from extra parameters |
| Hand-tuned SGD | Adam / AdamW | Adaptive per-parameter learning rates |
| No regularization | Dropout, BN, weight decay | Needed once networks grew larger |
| 5x5 kernels | Stacks of 3x3 | More expressive per parameter |
| RBF output | Softmax + cross-entropy | Better-behaved gradients |

Each of these is a topic in later lessons. LeNet is the baseline against which all of them are measured.

### From Theory to the Code Below

| Theory concept | Code construct in this lesson |
|----------------|-------------------------------|
| LeNet layer plan | The `nn.Sequential` of Conv2d / AvgPool / Linear |
| Modernized activations | Substituting `nn.ReLU()` for tanh |
| Parameter counting | `sum(p.numel() for p in model.parameters())` |
| Training recipe | `Adam` + cross-entropy + (no need for) careful init |

---


## Overview

LeNet-5 is the first successful Convolutional Neural Network proposed by Yann LeCun in 1998. It showed excellent performance on handwritten digit recognition (MNIST) and laid the foundation for modern CNNs.

---

## Mathematical Background

### 1. Convolution Operation

```
2D Convolution:
(I * K)[i,j] = Σ_m Σ_n I[i+m, j+n] · K[m, n]

Where:
- I: input image (H × W)
- K: kernel/filter (k_h × k_w)
- *: convolution operation

Output size:
H_out = (H_in + 2P - K) / S + 1
W_out = (W_in + 2P - K) / S + 1

- P: padding
- S: stride
- K: kernel size
```

### 2. Pooling Operation

```
Max Pooling:
y[i,j] = max(x[i*s:i*s+k, j*s:j*s+k])

Average Pooling:
y[i,j] = mean(x[i*s:i*s+k, j*s:j*s+k])

Purpose:
1. Reduce spatial resolution (down-sampling)
2. Increase translation invariance
3. Reduce parameters/computation
```

### 3. Backpropagation through Convolution

```
Forward:
Y = X * W + b

Backward:

∂L/∂W = X * ∂L/∂Y  (cross-correlation)

∂L/∂X = ∂L/∂Y * rot180(W)  (full convolution)

∂L/∂b = Σ ∂L/∂Y
```

---

## LeNet-5 Architecture

```
Input: 32×32 grayscale image

Layer 1: Conv (5×5, 6 filters) → 28×28×6
         + Tanh + AvgPool (2×2) → 14×14×6

Layer 2: Conv (5×5, 16 filters) → 10×10×16
         + Tanh + AvgPool (2×2) → 5×5×16

Layer 3: Conv (5×5, 120 filters) → 1×1×120
         + Tanh

Layer 4: FC (120 → 84) + Tanh

Layer 5: FC (84 → 10) (output)

Parameters:
- Conv1: 5×5×1×6 + 6 = 156
- Conv2: 5×5×6×16 + 16 = 2,416
- Conv3: 5×5×16×120 + 120 = 48,120
- FC1: 120×84 + 84 = 10,164
- FC2: 84×10 + 10 = 850
- Total: ~61,706 parameters
```

---

## File Structure

```
03_CNN_LeNet/
├── README.md                      # This file
├── numpy/
│   ├── conv_numpy.py             # NumPy Convolution implementation
│   ├── pooling_numpy.py          # NumPy Pooling implementation
│   └── lenet_numpy.py            # Complete LeNet NumPy implementation
├── pytorch_lowlevel/
│   └── lenet_lowlevel.py         # Using F.conv2d, not nn.Conv2d
├── paper/
│   └── lenet_paper.py            # Exact paper architecture reproduction
└── exercises/
    ├── 01_visualize_filters.md   # Filter visualization
    └── 02_receptive_field.md     # Receptive field calculation
```

---

## Core Concepts

### 1. Local Connectivity

```
Fully Connected:
- Every input connects to every output
- Parameters: H_in × W_in × H_out × W_out

Convolution:
- Only local region connections (kernel size)
- Parameters: K × K × C_in × C_out
- Efficient through parameter sharing
```

### 2. Parameter Sharing

```
Same filter applied across entire image
→ Translation equivariance
→ Detects same features at any location
```

### 3. Hierarchical Features

```
Layer 1: Edges, corners (low-level)
Layer 2: Textures, patterns (mid-level)
Layer 3: Object parts (high-level)
Layer 4+: Complete objects (semantic)
```

---

## Implementation Levels

### Level 1: NumPy From-Scratch (numpy/)
- Direct implementation of convolution with loops
- im2col optimization
- Manual backpropagation implementation

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)
- Use F.conv2d, F.max_pool2d
- Don't use nn.Conv2d
- Manual parameter management

### Level 3: Paper Implementation (paper/)
- Reproduce original paper architecture
- Tanh activation (instead of ReLU)
- Average Pooling (instead of Max)

---

## Learning Checklist

- [ ] Understand convolution formula
- [ ] Memorize output size calculation formula
- [ ] Understand im2col technique
- [ ] Derive conv backward
- [ ] Understand max pooling backward
- [ ] Memorize LeNet architecture

---

## References

- LeCun et al. (1998). "Gradient-Based Learning Applied to Document Recognition"
- [CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)
- [../Deep_Learning/08_CNN_Basics.md](../Deep_Learning/08_CNN_Basics.md)
