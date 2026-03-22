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
