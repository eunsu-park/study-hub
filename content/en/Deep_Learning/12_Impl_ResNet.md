# 12. ResNet

[Previous: VGG](./11_Impl_VGG.md) | [Next: RNN Basics](./13_RNN_Basics.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the degradation problem in deep networks and describe why it cannot be attributed solely to overfitting or vanishing gradients.
2. Formulate the residual learning framework — F(x) + x — and explain why learning residual mappings is easier than learning unreferenced mappings.
3. Distinguish between the basic residual block and the bottleneck residual block, and explain when each is used.
4. Implement a ResNet variant (e.g., ResNet-50) from scratch in PyTorch, correctly constructing residual blocks with shortcut connections.
5. Handle dimension-matching shortcuts (projection shortcuts) when the input and output dimensions differ across blocks.
6. Evaluate the trade-offs between ResNet depth variants (ResNet-18, 34, 50, 101, 152) in terms of accuracy, parameter count, and computational cost.

---

## Overview

ResNet (Residual Network) won 1st place in ILSVRC 2015 and is a revolutionary model. Kaiming He et al. proposed **Skip Connections (Residual Connections)** that enable training networks with hundreds of layers.

> "Solving the degradation problem where performance decreases as depth increases"

---

## Mathematical Background

### 1. Degradation Problem

```
Problem: Performance degrades as network gets deeper

Observation:
- 56-layer network < 20-layer network (CIFAR-10)
- This is not overfitting (even training error is high)
- Optimization difficulty (vanishing/exploding gradient)

Ideal situation:
- Deeper network ≥ Shallow network
- Should at least be able to learn identity mapping
```

### 2. Residual Learning

```
Traditional approach:
  H(x) = desired output
  Network directly learns H(x)

Residual approach:
  F(x) = H(x) - x  (residual)
  H(x) = F(x) + x  (original goal)

Why is this easier?
- Learning identity mapping: Just need F(x) = 0
- Learning small changes: Easier than large changes
- Gradient flow: Direct propagation through addition operation
```

### 3. Skip Connection Gradient

```
Forward:
  y = F(x) + x

Backward:
  ∂L/∂x = ∂L/∂y × (∂F/∂x + 1)
              ↑
          Always ≥ 1!

Result:
- Gradient propagates directly through at least path of 1
- Gradient maintained even in hundreds of layers
- Solves vanishing gradient
```

### 4. Dimension Matching (Projection Shortcut)

```
When dimensions differ (stride=2 or channel change):

Option A: Zero Padding
  x_padded = pad(x, extra_channels)

Option B: 1×1 Convolution (paper choice)
  shortcut = Conv1×1(x)

  x: (N, 64, 56, 56)
  ↓ stride=2, channels 64→128
  y: (N, 128, 28, 28)

  shortcut = Conv1×1(64→128, stride=2)
```

---

## ResNet Architecture

### BasicBlock vs Bottleneck

```
BasicBlock (ResNet-18, 34):
┌─────────────────────────┐
│  Conv 3×3, BN, ReLU     │
│  Conv 3×3, BN           │
│         ↓               │
│    + ← shortcut         │
│       ReLU              │
└─────────────────────────┘

Bottleneck (ResNet-50, 101, 152):
┌─────────────────────────┐
│  Conv 1×1, BN, ReLU     │  ← Channel reduction
│  Conv 3×3, BN, ReLU     │  ← Main operation
│  Conv 1×1, BN           │  ← Channel expansion
│         ↓               │
│    + ← shortcut         │
│       ReLU              │
└─────────────────────────┘

Bottleneck advantages:
- Reduce channels before 3×3 operation → less computation
- More layers with same computation
```

### ResNet Variant Comparison

| Model | Layers | Block | Block Count | Params |
|-------|--------|-------|-------------|--------|
| ResNet-18 | 18 | Basic | [2,2,2,2] | 11.7M |
| ResNet-34 | 34 | Basic | [3,4,6,3] | 21.8M |
| ResNet-50 | 50 | Bottleneck | [3,4,6,3] | 25.6M |
| ResNet-101 | 101 | Bottleneck | [3,4,23,3] | 44.5M |
| ResNet-152 | 152 | Bottleneck | [3,8,36,3] | 60.2M |

### ResNet-50 Detailed Structure

```
Input: 224×224×3

Conv1: 7×7, 64, stride=2, padding=3
  → (112×112×64)
MaxPool: 3×3, stride=2, padding=1
  → (56×56×64)

Layer1: Bottleneck × 3 (64→256)
  → (56×56×256)

Layer2: Bottleneck × 4 (128→512, stride=2)
  → (28×28×512)

Layer3: Bottleneck × 6 (256→1024, stride=2)
  → (14×14×1024)

Layer4: Bottleneck × 3 (512→2048, stride=2)
  → (7×7×2048)

AdaptiveAvgPool: → (1×1×2048)
FC: 2048 → 1000
```

---

## File Structure

```
05_ResNet/
├── README.md                      # This file
├── pytorch_lowlevel/
│   └── resnet_lowlevel.py        # F.conv2d, manual BN
├── paper/
│   └── resnet_paper.py           # Exact paper reproduction
├── analysis/
│   └── gradient_flow.py          # Skip connection effect analysis
└── exercises/
    ├── 01_gradient_analysis.md   # Compare gradient flow
    └── 02_ablation_study.md      # Compare shortcut types
```

---

## Core Concepts

### 1. Why Identity Mapping Matters

```python
# Pre-activation ResNet (v2)
def forward(self, x):
    identity = x

    out = self.bn1(x)
    out = F.relu(out)
    out = self.conv1(out)

    out = self.bn2(out)
    out = F.relu(out)
    out = self.conv2(out)

    return out + identity  # Clean identity path

# Post-activation (original)
def forward(self, x):
    identity = self.shortcut(x)

    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    out = F.relu(out + identity)  # ReLU transforms identity
    return out
```

### 2. ResNet as Ensemble Perspective

```
ResNet can be viewed as an ensemble of various depth paths

n blocks → 2^n possible paths
- Paths that "skip" some blocks
- Paths that go through all blocks

Experiment: Performance maintained even after removing some blocks during training
→ Various depth paths learned together
```

### 3. Role of Batch Normalization

```
Why BN is important in ResNet:

1. Reduces internal covariate shift
   - Stabilizes layer input distribution

2. Enables higher learning rate
   - Faster convergence

3. Regularization effect
   - Noise from mini-batch statistics

4. Improves gradient flow
   - Stabilizes gradient through normalization
```

### 4. Developments After ResNet

```
ResNeXt (2017):
- Introduces cardinality with grouped convolution
- ResNeXt-50: ResNet-101 performance, fewer parameters

DenseNet (2017):
- Connects all layers to all subsequent layers
- Maximizes feature reuse

EfficientNet (2019):
- Simultaneous scaling of width, depth, resolution
- Compound scaling

RegNet (2020):
- Optimal network structure search
- Simple and regular design
```

---

## Implementation Levels

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)

- F.conv2d, manual BatchNorm
- Manual implementation of BasicBlock, Bottleneck
- Implement shortcut projection
- Manual parameter management

### Level 3: Paper Implementation (paper/)

- Complete ResNet-18/34/50/101/152
- Pre-activation ResNet (v2)
- Compare Zero-padding vs Projection shortcut

### Level 4: Code Analysis (analysis/)

- Analyze torchvision ResNet code
- Visualize gradient flow
- Experiment with removing intermediate blocks

---

## Learning Checklist

- [ ] Understand degradation problem
- [ ] Derive residual learning formula
- [ ] Gradient benefits of skip connections
- [ ] Difference between BasicBlock vs Bottleneck
- [ ] Memorize ResNet-50 architecture
- [ ] Implementation method of projection shortcut
- [ ] Difference between Pre/Post-activation
- [ ] Understand ResNet ensemble perspective

---

## References

- He et al. (2015). "Deep Residual Learning for Image Recognition"
- He et al. (2016). "Identity Mappings in Deep Residual Networks" (v2)
- [torchvision ResNet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
- [d2l.ai: ResNet](https://d2l.ai/chapter_convolutional-modern/resnet.html)
- [../04_VGG/README.md](../04_VGG/README.md)
