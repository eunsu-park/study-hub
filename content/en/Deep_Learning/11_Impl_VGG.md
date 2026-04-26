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

## Theory & Principles

VGG (Simonyan & Zisserman 2014) is the architectural commitment to one principle: depth via stacked `3x3` convolutions, with everything else (kernel sizes, pooling factors, channel doubling) standardized. Implementing VGG from scratch is the cleanest way to internalize how a deep CNN's parameter budget actually distributes across layers, and to feel the pain (memory, compute) that motivated every later improvement (BN, residuals, depthwise convolutions).

This section covers:

- **A.** The VGG design principle: small kernels, deep stacks, regular doubling
- **B.** Why VGG works without batch normalization (and what cost)
- **C.** Parameter and FLOP analysis: where the cost lives
- **D.** The VGG family (VGG-11/13/16/19) as a depth ablation

### A. The VGG Principle

VGG's architecture follows three rules:

1. All convolutions are `3x3`, stride 1, padding 1 (output spatial size unchanged).
2. All pooling is `2x2`, stride 2 (output spatial size halved).
3. Channel count doubles each time spatial size halves: `64 -> 128 -> 256 -> 512 -> 512`.

The conv blocks come in groups (the "block" structure): repeat `Conv -> ReLU` two or three times, then `MaxPool`. VGG-16 has the pattern `[2, 2, 3, 3, 3]` convs per block, total 13 conv layers + 3 fully-connected layers = 16 weight layers.

Two stacked `3x3` convs produce a `5x5` receptive field with `2 * 9 = 18` weights per channel; one `5x5` conv produces the same RF with `25` weights. Three stacked `3x3` produce `7x7` RF with `27` weights vs `49`. VGG buys *receptive field per parameter* by going deeper rather than wider in the kernel.

### B. VGG Without BatchNorm

VGG predates BatchNorm by a year. It trained successfully (16-19 layers were considered very deep at the time) using:

- **Careful initialization**: pretrain a shallower network, then add layers using its weights as initialization.
- **Small learning rate** with manual schedule (drop by 10x when validation loss plateaus).
- **Long training**: 74 epochs on ImageNet, weeks of compute.

When BN was introduced, VGG-BN versions trained much more easily — fewer warm-start tricks, larger learning rates, faster convergence. Modern reimplementations almost always use BN even though the original VGG paper does not. This is a reminder that "the network architecture trained" depends on the contemporaneous toolbox; many old papers' architectures are easier to reproduce today thanks to better training infrastructure.

### C. Parameter and FLOP Analysis

VGG-16 has ~138 million parameters. Where do they live?

- Conv layers (13 total): ~14.7 million params (~10% of total)
- FC layers (3 total): ~123.6 million params (~90% of total)

The FC layers dominate because the first one maps `7 * 7 * 512 = 25,088` activations to 4,096 units: `25,088 * 4,096 = ~103M` parameters in one matrix. Modern architectures (ResNet, DenseNet, ViT) replaced this with **global average pooling** followed by one small FC layer, dropping the FC parameter cost by 10-100x. VGG's FC dominance is the single biggest reason it has fallen out of favor for deployment.

For FLOPs, the picture inverts: convolutions dominate because they apply at every spatial location. VGG-16 inference is ~15.5 GFLOPs, mostly in the conv layers, with a small contribution from the (huge) FC layers because they only run once.

### D. VGG-11/13/16/19 as Depth Ablation

The original VGG paper compared four depths with the same overall pattern:

| Variant | Conv layers | Params | ImageNet top-5 error |
|---------|-------------|--------|----------------------|
| VGG-11  | 8           | 133M   | 10.4% |
| VGG-13  | 10          | 133M   |  9.9% |
| VGG-16  | 13          | 138M   |  8.8% |
| VGG-19  | 16          | 144M   |  9.0% |

Two takeaways:

1. **Depth helps, until it doesn't.** VGG-19 was actually slightly worse than VGG-16, which the authors attributed to optimization difficulty at that depth. ResNet (the next year) addressed exactly this: residual connections let networks go to 152+ layers without optimization breakdown.
2. **Most parameters are in the FC head**, so deeper conv stacks barely change the total parameter count. This made VGG a clean experimental design but a poor template for parameter-efficient networks.

### From Theory to the Code Below

| Theory concept | Code construct in this lesson |
|----------------|-------------------------------|
| 3x3 conv + ReLU block | `nn.Conv2d(c, c, 3, padding=1)` followed by `nn.ReLU(inplace=True)` |
| MaxPool downsampling | `nn.MaxPool2d(2, 2)` after each block |
| Channel doubling | `64 -> 128 -> 256 -> 512` channel sequence |
| FC parameter dominance | The 4096-unit FC layers with ~100M params |

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
