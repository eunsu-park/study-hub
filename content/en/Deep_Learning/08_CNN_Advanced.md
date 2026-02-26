# 08. Advanced CNN - Famous Architectures

[Previous: CNN Basics](./07_CNN_Basics.md) | [Next: Transfer Learning](./09_Transfer_Learning.md)

---

## Learning Objectives

- Understand VGG, ResNet, and EfficientNet architectures
- Learn Skip Connection and Residual Learning
- Understand training problems of deep networks and solutions
- Implement with PyTorch

---

## 1. VGG (2014)

### Core Ideas

- Use only small filters (3×3)
- Improve performance by increasing depth
- Simple and consistent structure

### Architecture (VGG16)

```
Input 224×224×3
  ↓
Conv 3×3, 64 ×2 → MaxPool → 112×112×64
  ↓
Conv 3×3, 128 ×2 → MaxPool → 56×56×128
  ↓
Conv 3×3, 256 ×3 → MaxPool → 28×28×256
  ↓
Conv 3×3, 512 ×3 → MaxPool → 14×14×512
  ↓
Conv 3×3, 512 ×3 → MaxPool → 7×7×512
  ↓
FC 4096 → FC 4096 → FC 1000
```

### PyTorch Implementation

```python
def make_vgg_block(in_ch, out_ch, num_convs):
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv2d(
            in_ch if i == 0 else out_ch,
            out_ch, 3, padding=1
        ))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            make_vgg_block(3, 64, 2),
            make_vgg_block(64, 128, 2),
            make_vgg_block(128, 256, 3),
            make_vgg_block(256, 512, 3),
            make_vgg_block(512, 512, 3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

---

## 2. ResNet (2015)

Simply stacking more layers should help, but experiments showed accuracy *decreasing* beyond ~20 layers (the degradation problem). This was surprising — a deeper network should be at least as good as a shallow one, because the extra layers could just learn identity mappings. The key insight: it is easier to learn a residual F(x) than the full mapping H(x). If the optimal mapping is close to identity, learning F(x) = 0 is trivial, whereas learning H(x) = x from scratch is not.

### Problem: Vanishing Gradients

- Gradients vanish as network gets deeper
- Simply stacking layers degrades performance

### Solution: Residual Connection

```
        ┌─────────────────┐
        │                 │
x ──────┼───► Conv ──► Conv ──►(+)──► ReLU ──► Output
        │                 ↑
        └────────(identity)┘

Output = F(x) + x   (Residual Learning)
```

### Key Insight

- Learning identity function becomes easier
- Gradients flow directly through skip connections
- Can train networks with 1000+ layers

**Gradient flow through residual connections**: The output is H(x) = F(x) + x. During backpropagation: dL/dx = dL/dH * (dF/dx + 1). The crucial "+1" term means the gradient always has a direct path through the skip connection, bypassing the conv layers entirely. Even if dF/dx vanishes (as it does in very deep networks), the gradient is at least dL/dH * 1 — no vanishing gradient, even in 1000-layer networks.

### PyTorch Implementation

```python
class BasicBlock(nn.Module):
    """ResNet basic block"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        # Normalize activations per channel — stabilizes training by reducing
        # internal covariate shift, allowing higher learning rates
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # downsample: 1×1 conv that matches dimensions when residual and
        # main path differ in channels or spatial size (e.g., stride=2)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            # Project identity to match out's shape so addition is valid
            identity = self.downsample(x)

        out += identity  # Skip connection! Gradient flows through this '+' unchanged
        out = F.relu(out)
        return out
```

### Bottleneck Block (ResNet-50+)

A 1x1 convolution with C_out < C_in reduces the channel dimension — think of it as a learned linear combination across channels, projecting high-dimensional features into a compact subspace. This reduces computation for the subsequent expensive 3x3 convolution. For example, reducing 256 channels to 64 before a 3x3 conv cuts the FLOPs by 16x for that layer.

```python
class Bottleneck(nn.Module):
    """1×1 → 3×3 → 1×1 structure"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # 1×1 conv: reduce channels (e.g., 256→64) to cut 3×3 conv cost
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 3×3 conv: the only spatially-aware layer in the bottleneck —
        # operates on the reduced channel dimension for efficiency
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1×1 conv: expand back to 4× channels for the residual addition
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out
```

---

## 3. ResNet Variants

### Pre-activation ResNet

```
Original: x → Conv → BN → ReLU → Conv → BN → (+) → ReLU
Pre-act: x → BN → ReLU → Conv → BN → ReLU → Conv → (+)
```

### ResNeXt

```python
# Using grouped convolution
self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                       groups=32, padding=1)
```

### SE-ResNet (Squeeze-and-Excitation)

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y  # Channel recalibration
```

---

## 4. EfficientNet (2019)

### Core Ideas

- Balanced scaling of depth, width, and resolution
- Compound Scaling

```
depth: α^φ
width: β^φ
resolution: γ^φ

α × β² × γ² ≈ 2 (computation constraint)
```

### MBConv Block

```python
class MBConv(nn.Module):
    """Mobile Inverted Bottleneck"""
    def __init__(self, in_ch, out_ch, expand_ratio, stride, se_ratio=0.25):
        super().__init__()
        hidden = in_ch * expand_ratio

        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        ) if expand_ratio != 1 else nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )

        self.se = SEBlock(hidden, int(in_ch * se_ratio))

        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        self.use_skip = stride == 1 and in_ch == out_ch

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_skip:
            out = out + x
        return out
```

---

## 5. Architecture Comparison

| Model | Parameters | Top-1 Acc | Features |
|-------|-----------|-----------|----------|
| VGG16 | 138M | 71.5% | Simple, memory-intensive |
| ResNet-50 | 26M | 76.0% | Skip Connection |
| ResNet-152 | 60M | 78.3% | Deeper version |
| EfficientNet-B0 | 5.3M | 77.1% | Efficient |
| EfficientNet-B7 | 66M | 84.3% | Best performance |

---

## 6. torchvision Pretrained Models

```python
import torchvision.models as models

# Load pretrained models
resnet50 = models.resnet50(weights='IMAGENET1K_V2')
efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')
vgg16 = models.vgg16(weights='IMAGENET1K_V1')

# Feature extraction
resnet50.eval()
for param in resnet50.parameters():
    param.requires_grad = False

# Replace last layer (transfer learning)
resnet50.fc = nn.Linear(2048, 10)  # 10 classes
```

---

## 7. Model Selection Guide

### Recommendations by Use Case

| Situation | Recommended Model |
|-----------|------------------|
| Fast inference needed | MobileNet, EfficientNet-B0 |
| High accuracy needed | EfficientNet-B4~B7 |
| Educational/understanding | VGG, ResNet-18 |
| Memory constraints | MobileNet, ShuffleNet |

### Practical Tips

```python
# Check model size
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Calculate FLOPs (thop package)
from thop import profile
flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224),))
```

---

## Summary

### Core Concepts

1. **VGG**: Repeating small filters, deep networks
2. **ResNet**: Solve vanishing gradients with Skip Connections
3. **EfficientNet**: Efficient scaling

### Evolution

```
LeNet (1998)
  ↓
AlexNet (2012) - GPU usage
  ↓
VGG (2014) - Deeper
  ↓
GoogLeNet (2014) - Inception module
  ↓
ResNet (2015) - Skip Connection
  ↓
EfficientNet (2019) - Compound Scaling
  ↓
Vision Transformer (2020) - Attention
```

---

## Exercises

### Exercise 1: ResNet Skip Connection — Why It Works

Conceptually and empirically verify the benefit of skip connections.

1. Build a plain 6-layer CNN (no skip connections) and a 6-block ResNet for CIFAR-10.
2. Train both for 20 epochs with identical hyperparameters.
3. Compare final test accuracy and examine how training loss evolves.
4. Inspect the gradient norm at the first layer during training for both models.
5. Explain in your own words why adding `out += identity` allows gradients to flow more freely.

### Exercise 2: Implement a Custom BasicBlock

Implement `BasicBlock` from scratch and verify it against PyTorch's built-in ResNet.

1. Implement `BasicBlock(in_channels, out_channels, stride)` as shown in the lesson, including the downsample shortcut when channels or stride change.
2. Stack 4 such blocks to create a small ResNet-like model for MNIST.
3. Confirm that `output.shape == (batch, num_classes)` for a forward pass.
4. Count total parameters and compare against a plain CNN with similar depth.

### Exercise 3: Squeeze-and-Excitation Channel Attention

Add channel attention to an existing model and measure the accuracy improvement.

1. Implement `SEBlock(channels, reduction=16)` as shown in the lesson.
2. Wrap the second conv layer of `CIFAR10Net` from the previous lesson with an SE block.
3. Train both the baseline and SE-augmented model on CIFAR-10 for 20 epochs.
4. Report the test accuracy difference. Explain the mechanism: how does the SE block decide which channels to amplify?

### Exercise 4: Model Efficiency Trade-offs

Use `count_parameters` and `thop.profile` to compare architectures side by side.

1. Load `vgg16`, `resnet50`, and `efficientnet_b0` from torchvision (no pretrained weights needed).
2. Count trainable parameters for each using `count_parameters`.
3. Compute FLOPs for a 224×224 input using `thop.profile`.
4. Create a table showing parameters, FLOPs, and ImageNet Top-1 accuracy (from the lesson's table).
5. Which model gives the best accuracy-per-parameter ratio? Explain your reasoning.

### Exercise 5: Architecture Evolution Experiment

Trace the historical improvements by training LeNet, a VGG-style net, and a ResNet-style net on CIFAR-10.

1. Implement a 2-conv LeNet-style net, a 4-conv VGG-style block, and a 4-block ResNet.
2. Train all three for 25 epochs with the same optimizer (Adam, lr=0.001).
3. Plot learning curves (training loss and test accuracy) for all three on a single chart.
4. Describe the qualitative differences in convergence speed and final performance.

---

## Next Steps

In [09_Transfer_Learning.md](./09_Transfer_Learning.md), we'll learn transfer learning using pretrained models.
