# 11. VGG

[Previous: CNN (LeNet)](./10_Impl_CNN_LeNet.md) | [Next: ResNet](./12_Impl_ResNet.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the VGGNet architectural philosophy of using very deep networks with uniformly small 3×3 convolutional filters.
2. Analyze the receptive field equivalence between stacked 3×3 convolutions and larger single convolutions, and compare their parameter counts.
3. Describe the VGG-16 and VGG-19 configurations, including block structure, channel progression, and classifier head design.
4. Implement a VGGNet family member (e.g., VGG-16) from scratch in PyTorch using modular block construction.
5. Apply transfer learning with a pretrained VGGNet to a custom classification task using fine-tuning techniques.
6. Identify the limitations of VGGNet (parameter count, memory footprint) and explain how later architectures addressed them.

---

## Overview

VGGNet finished 2nd in ILSVRC 2014, proposed by Karen Simonyan and Andrew Zisserman. The paper "Very Deep Convolutional Networks for Large-Scale Image Recognition" demonstrated that **stacking small 3x3 filters deeply** is effective.

---

## Mathematical Background

### 1. Effect of 3x3 Filter Stacking

```
Why stack multiple 3x3 filters?

Two 3x3 convs ≈ One 5x5 conv (same receptive field)
Three 3x3 convs ≈ One 7x7 conv

Advantages:
1. Reduced parameters:
   - 7x7: 49C² parameters
   - 3x3 × 3: 27C² parameters (45% reduction)

2. Increased non-linearity:
   - 7x7: 1 ReLU
   - 3x3 × 3: 3 ReLUs → can learn more complex functions
```

### 2. Receptive Field Calculation

```
Receptive field increases as layers stack:

RF = (RF_prev - 1) × stride + kernel_size

Example (stride=1, kernel=3):
- Layer 1: RF = 3
- Layer 2: RF = 5
- Layer 3: RF = 7
- Layer 4: RF = 9
...

After MaxPool (kernel=2, stride=2):
- RF doubles
```

### 3. Feature Map Size Changes

```
Conv (stride=1, padding=1, kernel=3):
  H_out = H_in  (maintains size)

MaxPool (kernel=2, stride=2):
  H_out = H_in / 2  (halves size)

224 → [Conv×2] → 224 → Pool → 112 → [Conv×2] → 112 → Pool → 56 → ...
```

---

## VGG Architecture

### VGG Variant Comparison

| Configuration | VGG11 | VGG13 | VGG16 | VGG19 |
|---------------|-------|-------|-------|-------|
| Conv Layers | 8 | 10 | 13 | 16 |
| FC Layers | 3 | 3 | 3 | 3 |
| Total Layers | 11 | 13 | 16 | 19 |
| Parameters | 133M | 133M | 138M | 144M |

### VGG16 Detailed Structure

```
Input: 224×224×3 RGB image

Block 1: [Conv3-64] × 2 + MaxPool
  (224×224×3) → (224×224×64) → (112×112×64)

Block 2: [Conv3-128] × 2 + MaxPool
  (112×112×64) → (112×112×128) → (56×56×128)

Block 3: [Conv3-256] × 3 + MaxPool
  (56×56×128) → (56×56×256) → (28×28×256)

Block 4: [Conv3-512] × 3 + MaxPool
  (28×28×256) → (28×28×512) → (14×14×512)

Block 5: [Conv3-512] × 3 + MaxPool
  (14×14×512) → (14×14×512) → (7×7×512)

Classifier:
  Flatten: 7×7×512 = 25,088
  FC1: 25088 → 4096 + ReLU + Dropout
  FC2: 4096 → 4096 + ReLU + Dropout
  FC3: 4096 → 1000 (classes)

Parameter distribution:
- Conv layers: ~15M (11%)
- FC layers: ~124M (89%)  ← Most!
```

### VGG Configuration

```python
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
# 'M' = MaxPool
```

---

## File Structure

```
04_VGG/
├── README.md                      # This file
├── pytorch_lowlevel/
│   └── vgg_lowlevel.py           # Using F.conv2d, F.linear
├── paper/
│   └── vgg_paper.py              # Exact paper architecture reproduction
└── exercises/
    ├── 01_feature_visualization.md   # Visualize feature maps per block
    └── 02_transfer_learning.md       # Use pretrained weights
```

---

## Core Concepts

### 1. Deep & Narrow vs Shallow & Wide

```
Before VGG: Large filters + shallow networks
  - AlexNet: 11×11, 5×5 filters
  - Few layers

VGG: Small filters + deep networks
  - Only 3×3 filters (+ some 1×1)
  - 16~19 layers

Conclusion: Depth is crucial for performance
```

### 2. Uniform Structure

```
VGG design principles:

1. All Conv are 3×3, stride=1, padding=1
2. All MaxPool are 2×2, stride=2
3. Double channels per block (64→128→256→512)
4. Simple and regular → easy to understand/implement
```

### 3. VGG Limitations

```
Disadvantages:
1. Too many parameters (138M, ResNet-50: 25M)
2. High memory consumption (FC layers)
3. Slow training
4. Gradient vanishing (as it gets deeper)

Follow-up research:
- GoogLeNet: Efficiency with Inception modules
- ResNet: Deeper with skip connections
- MobileNet: Depthwise separable conv
```

### 4. VGG as Feature Extractor

```
VGG widely used as feature extractor:

1. Style Transfer
   - Content: block4_conv2
   - Style: block1~5_conv1

2. Perceptual Loss
   - Compare VGG features instead of pixel loss

3. Object Detection
   - VGG backbone + detection head
```

---

## Implementation Levels

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)

- Use F.conv2d, F.max_pool2d, F.linear
- Don't use nn.Conv2d, nn.Linear
- Manual parameter initialization and management
- Block-wise modularization

### Level 3: Paper Implementation (paper/)

- Reproduce all paper settings
- Add Batch Normalization (VGG-BN)
- Support various VGG variants

---

## Learning Checklist

- [ ] Understand advantages of 3×3 filter stacking
- [ ] Master receptive field calculation method
- [ ] Memorize VGG16 architecture
- [ ] Understand parameter distribution (Conv vs FC)
- [ ] How to use VGG as feature extractor
- [ ] Compare VGG limitations with follow-up models

---

## References

- Simonyan & Zisserman (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition"
- [torchvision VGG](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)
- [CS231n: ConvNets](https://cs231n.github.io/convolutional-networks/)
- [../03_CNN_LeNet/README.md](../03_CNN_LeNet/README.md)
