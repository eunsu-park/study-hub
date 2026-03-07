# Lesson 6: Efficient Architectures

[Previous: Knowledge Distillation](./05_Knowledge_Distillation.md) | [Next: Neural Architecture Search](./07_Neural_Architecture_Search.md)

---

## Learning Objectives

- Understand depthwise separable convolutions and why they reduce computation drastically
- Implement MobileNet V1/V2 building blocks (depthwise separable, inverted residuals)
- Apply compound scaling (EfficientNet) to balance depth, width, and resolution
- Analyze ShuffleNet's channel shuffle operation for cross-group information flow
- Compare SqueezeNet, GhostNet, and FBNet design strategies
- Apply design principles for building custom efficient networks

---

## 1. The Efficiency Problem

Standard convolutions are computationally expensive. A single Conv2d layer with `C_in` input channels, `C_out` output channels, kernel size `K`, and spatial dimensions `H x W` requires:

```
FLOPs = C_in × C_out × K × K × H × W

Example: Conv2d(256, 512, 3, padding=1) on 14×14 feature map
FLOPs = 256 × 512 × 3 × 3 × 14 × 14 = 231M FLOPs (for a single layer!)
```

Efficient architectures reduce this cost through clever factorizations and design patterns.

```python
import torch
import torch.nn as nn


def count_conv_flops(in_channels, out_channels, kernel_size, spatial_size):
    """Count FLOPs for a standard Conv2d (no bias)."""
    k = kernel_size
    h, w = spatial_size, spatial_size
    flops = in_channels * out_channels * k * k * h * w
    params = in_channels * out_channels * k * k
    return flops, params


def count_dw_pw_flops(in_channels, out_channels, kernel_size, spatial_size):
    """Count FLOPs for depthwise + pointwise (depthwise separable) convolution."""
    k = kernel_size
    h, w = spatial_size, spatial_size

    # Depthwise: each input channel is convolved independently
    dw_flops = in_channels * k * k * h * w
    dw_params = in_channels * k * k

    # Pointwise: 1x1 convolution to mix channels
    pw_flops = in_channels * out_channels * h * w
    pw_params = in_channels * out_channels

    return dw_flops + pw_flops, dw_params + pw_params


# Compare standard vs depthwise separable
in_c, out_c, k, s = 256, 512, 3, 14

std_flops, std_params = count_conv_flops(in_c, out_c, k, s)
dw_flops, dw_params = count_dw_pw_flops(in_c, out_c, k, s)

print(f"Standard Conv2d:      {std_flops/1e6:>8.1f}M FLOPs, {std_params/1e3:>6.1f}K params")
print(f"Depthwise Separable:  {dw_flops/1e6:>8.1f}M FLOPs, {dw_params/1e3:>6.1f}K params")
print(f"FLOPs reduction:      {std_flops/dw_flops:.1f}x")
print(f"Params reduction:     {std_params/dw_params:.1f}x")
```

---

## 2. MobileNet V1: Depthwise Separable Convolutions

MobileNet V1 (Howard et al., 2017) replaces standard convolutions with **depthwise separable convolutions**, reducing computation by ~8-9x.

### 2.1 Depthwise Separable Convolution

A standard convolution is factored into two steps:

```
Standard Convolution:
  Input (C_in, H, W) → Conv(C_in, C_out, K×K) → Output (C_out, H, W)
  FLOPs: C_in × C_out × K² × H × W

Depthwise Separable Convolution:
  Step 1: Depthwise — each channel independently
    Input (C_in, H, W) → Conv(C_in, C_in, K×K, groups=C_in) → (C_in, H, W)
    FLOPs: C_in × K² × H × W

  Step 2: Pointwise — mix channels with 1×1 conv
    (C_in, H, W) → Conv(C_in, C_out, 1×1) → Output (C_out, H, W)
    FLOPs: C_in × C_out × H × W

  Total FLOPs: C_in × (K² + C_out) × H × W
  Reduction:   1/C_out + 1/K²  ≈  8-9x for K=3
```

```python
import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution = Depthwise Conv + Pointwise Conv.
    Used in MobileNetV1 as the core building block.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1):
        super().__init__()
        # Depthwise: groups=in_channels means each channel has its own filter
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # Key: each channel independently
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU6(inplace=True)

        # Pointwise: 1x1 convolution to mix channels
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.depthwise(x)))
        x = self.relu2(self.bn2(self.pointwise(x)))
        return x


# Compare parameter counts
std_conv = nn.Conv2d(64, 128, 3, padding=1, bias=False)
dw_sep_conv = DepthwiseSeparableConv(64, 128)

std_params = sum(p.numel() for p in std_conv.parameters())
dw_params = sum(p.numel() for p in dw_sep_conv.parameters())

print(f"Standard Conv params:  {std_params:,}")
print(f"DW Separable params:   {dw_params:,}")
print(f"Reduction: {std_params / dw_params:.1f}x")
```

### 2.2 MobileNet V1 Architecture

```python
import torch
import torch.nn as nn


class MobileNetV1(nn.Module):
    """
    Simplified MobileNet V1 architecture.

    Architecture: 1 standard conv, then 13 depthwise separable convs.
    Width multiplier (alpha) scales channel counts.
    Resolution multiplier (rho) scales input resolution.
    """

    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()
        def _c(channels):
            """Apply width multiplier."""
            return max(8, int(channels * width_mult))

        # First standard convolution
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, _c(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(_c(32)),
            nn.ReLU6(inplace=True),
        )

        # Depthwise separable convolution blocks
        # (in_channels, out_channels, stride)
        cfg = [
            (32, 64, 1),
            (64, 128, 2), (128, 128, 1),
            (128, 256, 2), (256, 256, 1),
            (256, 512, 2),
            (512, 512, 1), (512, 512, 1), (512, 512, 1),
            (512, 512, 1), (512, 512, 1),
            (512, 1024, 2), (1024, 1024, 1),
        ]

        layers = []
        for in_c, out_c, stride in cfg:
            layers.append(DepthwiseSeparableConv(
                _c(in_c), _c(out_c),
                kernel_size=3, stride=stride, padding=1,
            ))
        self.features = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(_c(1024), num_classes)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# Width multiplier comparison
for alpha in [1.0, 0.75, 0.5, 0.25]:
    model = MobileNetV1(num_classes=10, width_mult=alpha)
    params = sum(p.numel() for p in model.parameters())
    print(f"MobileNetV1 (alpha={alpha}): {params/1e6:.2f}M params")
```

---

## 3. MobileNet V2: Inverted Residuals

MobileNet V2 (Sandler et al., 2018) introduces **inverted residual blocks** with linear bottlenecks, improving accuracy while maintaining efficiency.

### 3.1 Inverted Residual Block

Standard residual: wide → narrow → wide (bottleneck)
Inverted residual: narrow → wide → narrow (expansion)

```
Standard Residual Block          Inverted Residual Block
(ResNet bottleneck):             (MobileNetV2):
┌─────┐                         ┌─────┐
│ 256 │ (wide)                   │  64 │ (narrow) ←── input
├─────┤                         ├─────┤
│  64 │ (narrow bottleneck)      │ 384 │ (wide expansion) ←── 1×1 conv, expand 6x
├─────┤                         ├─────┤
│  64 │ (3×3 conv)               │ 384 │ (depthwise 3×3 conv)
├─────┤                         ├─────┤
│ 256 │ (wide)                   │  64 │ (narrow) ←── 1×1 conv, project
└──┬──┘                         └──┬──┘
   │ + skip connection              │ + skip connection (when stride=1, in=out)
```

```python
import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    """
    MobileNetV2 Inverted Residual Block.

    1. Expand: 1×1 conv to expand channels by factor t
    2. Depthwise: 3×3 depthwise conv (spatial processing)
    3. Project: 1×1 conv to reduce channels back (LINEAR, no ReLU!)
    4. Skip connection when stride=1 and in_channels == out_channels
    """

    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        layers = []
        # Expand (skip if expand_ratio == 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])

        # Project (LINEAR — no activation!)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # No ReLU here! Linear bottleneck preserves information.
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


# Demonstrate the inverted residual block
block = InvertedResidual(in_channels=64, out_channels=64, stride=1, expand_ratio=6)
x = torch.randn(1, 64, 28, 28)
out = block(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print(f"Params: {sum(p.numel() for p in block.parameters()):,}")

# The linear bottleneck is crucial — adding ReLU to the projection
# destroys information in the narrow bottleneck and hurts accuracy
```

### 3.2 Why Linear Bottlenecks?

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_bottleneck_experiment():
    """
    Demonstrate why ReLU in narrow bottlenecks destroys information.

    When the bottleneck dimension is small, ReLU zeros out a large
    fraction of values, causing irreversible information loss.
    """
    # Simulate a narrow bottleneck
    for dim in [2, 4, 8, 16, 64, 256]:
        x = torch.randn(1000, dim)

        # After ReLU, what fraction of information survives?
        x_relu = F.relu(x)
        surviving = (x_relu > 0).float().mean().item()

        # Information preservation (cosine similarity of reconstructed data)
        # In a well-trained network, the projection layer tries to reconstruct
        print(f"  Dim {dim:>3}: {surviving:.1%} values survive ReLU "
              f"(~{(1-surviving):.0%} information lost)")

    print("\nConclusion: In narrow bottlenecks (dim ≤ 16),")
    print("ReLU destroys too much information → use linear activation instead.")


linear_bottleneck_experiment()
```

---

## 4. EfficientNet: Compound Scaling

EfficientNet (Tan & Le, 2019) introduces **compound scaling** — scaling depth, width, and resolution together using a fixed ratio, rather than scaling one dimension at a time.

### 4.1 The Scaling Problem

```
Single-dimension scaling (suboptimal):
  Wider:   ResNet-18 → ResNet-18-Wide (more channels)
  Deeper:  ResNet-18 → ResNet-50 → ResNet-152 (more layers)
  Higher:  224×224 → 299×299 → 480×480 (larger input)

Each dimension has diminishing returns when scaled alone.

Compound scaling (EfficientNet):
  Scale all three dimensions together with a fixed ratio:
    depth:      d = alpha^phi
    width:      w = beta^phi
    resolution: r = gamma^phi

  where:
    alpha × beta^2 × gamma^2 ≈ 2  (to roughly double FLOPs)
    phi controls the total compute budget
```

```python
import torch
import torch.nn as nn
import math


class CompoundScaling:
    """
    EfficientNet compound scaling calculator.

    Base model (B0): depth=1.0, width=1.0, resolution=224
    Scale factors found by grid search:
      alpha = 1.2 (depth)
      beta  = 1.1 (width)
      gamma = 1.15 (resolution)
    """

    def __init__(self, alpha=1.2, beta=1.1, gamma=1.15):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def scale(self, phi):
        """
        Compute scaled dimensions for a given compound coefficient phi.
        """
        depth = self.alpha ** phi
        width = self.beta ** phi
        resolution = self.gamma ** phi

        # Approximate FLOP increase (depth * width^2 * resolution^2)
        flop_ratio = depth * (width ** 2) * (resolution ** 2)

        return {
            "phi": phi,
            "depth_mult": round(depth, 2),
            "width_mult": round(width, 2),
            "resolution": int(224 * resolution),
            "approx_flop_ratio": round(flop_ratio, 1),
        }


scaler = CompoundScaling()
print(f"{'Model':<15} {'phi':>4} {'Depth':>6} {'Width':>6} {'Res':>5} {'FLOPs':>6}")
print("-" * 45)

# EfficientNet-B0 through B7
for model_name, phi in [
    ("B0", 0), ("B1", 0.5), ("B2", 1), ("B3", 2),
    ("B4", 3), ("B5", 4), ("B6", 5), ("B7", 6),
]:
    s = scaler.scale(phi)
    print(f"EfficientNet-{model_name:<5} {s['phi']:>4} "
          f"{s['depth_mult']:>6} {s['width_mult']:>6} "
          f"{s['resolution']:>5} {s['approx_flop_ratio']:>5}x")
```

### 4.2 MBConv Block (EfficientNet Building Block)

EfficientNet uses the **MBConv** block (Mobile Inverted Bottleneck with Squeeze-and-Excitation):

```python
import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block.
    Learns per-channel attention weights.

    1. Squeeze: Global average pool → (B, C, 1, 1)
    2. Excite: FC → ReLU → FC → Sigmoid → channel weights
    3. Scale: Multiply input by channel weights
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=True),
            nn.SiLU(inplace=True),  # EfficientNet uses SiLU (Swish)
            nn.Conv2d(reduced, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.squeeze(x)
        scale = self.excite(scale)
        return x * scale


class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck Conv (MBConv) with Squeeze-and-Excitation.
    This is the core building block of EfficientNet.

    Structure:
    1. Expand: 1×1 conv (expand channels by factor t)
    2. Depthwise: K×K depthwise conv
    3. SE: Squeeze-and-Excitation attention
    4. Project: 1×1 conv (reduce channels)
    5. Skip connection + stochastic depth (during training)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, expand_ratio=6, se_ratio=0.25,
                 drop_path_rate=0.0):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = int(in_channels * expand_ratio)
        padding = (kernel_size - 1) // 2

        layers = []

        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=stride,
                      padding=padding, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        ])

        # Squeeze-and-Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            layers.append(SqueezeExcitation(hidden_dim, hidden_dim // se_channels))

        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)
        self.drop_path_rate = drop_path_rate

    def _drop_path(self, x):
        """Stochastic depth: randomly drop the entire block during training."""
        if not self.training or self.drop_path_rate == 0:
            return x
        keep_prob = 1 - self.drop_path_rate
        mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep_prob
        return x * mask / keep_prob

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = self._drop_path(out)
            return x + out
        return out


# Test MBConv block
block = MBConv(32, 32, kernel_size=3, stride=1, expand_ratio=6, se_ratio=0.25)
x = torch.randn(1, 32, 56, 56)
out = block(x)
print(f"MBConv: {x.shape} → {out.shape}")
print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")
```

---

## 5. ShuffleNet: Channel Shuffle

ShuffleNet (Zhang et al., 2018) uses **group convolutions** (more efficient than standard convs) with a **channel shuffle** operation to enable cross-group information flow.

### 5.1 The Group Convolution Problem

```
Standard 1×1 Conv:               Group 1×1 Conv (groups=3):
┌─────────────┐                  ┌────┬────┬────┐
│ All C_in     │                  │ G1 │ G2 │ G3 │  Each group operates
│ channels     │                  │    │    │    │  independently — no
│ interact     │                  │    │    │    │  cross-group information!
└─────────────┘                  └────┴────┴────┘
FLOPs: C_in × C_out              FLOPs: C_in × C_out / g
```

**Channel shuffle** fixes this by rearranging channels after group convolution:

```python
import torch


def channel_shuffle(x, groups):
    """
    Channel shuffle operation for ShuffleNet.

    Rearranges channels so that information from different groups is mixed.

    Input channels:  [G1_c0, G1_c1, G2_c0, G2_c1, G3_c0, G3_c1]
    After shuffle:   [G1_c0, G2_c0, G3_c0, G1_c1, G2_c1, G3_c1]

    Implementation: reshape → transpose → flatten
    (B, C, H, W) → (B, g, C//g, H, W) → (B, C//g, g, H, W) → (B, C, H, W)
    """
    B, C, H, W = x.shape
    assert C % groups == 0, f"Channels {C} not divisible by groups {groups}"

    x = x.view(B, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    x = x.view(B, C, H, W)
    return x


# Demonstrate channel shuffle
x = torch.arange(12).float().view(1, 12, 1, 1)
print(f"Before shuffle: {x.view(-1).tolist()}")

shuffled = channel_shuffle(x, groups=3)
print(f"After shuffle:  {shuffled.view(-1).tolist()}")
# Group 1 (0,1,2,3), Group 2 (4,5,6,7), Group 3 (8,9,10,11)
# → Interleaved: 0,4,8,1,5,9,2,6,10,3,7,11
```

### 5.2 ShuffleNet V2 Unit

```python
import torch
import torch.nn as nn


class ShuffleNetV2Block(nn.Module):
    """
    ShuffleNet V2 building block.

    Key design principles from ShuffleNet V2 paper:
    1. Equal channel width minimizes memory access cost
    2. Excessive group convolution increases memory access cost
    3. Network fragmentation reduces parallelism
    4. Element-wise operations (ReLU, Add) are non-negligible

    ShuffleNet V2 uses channel split + concatenation instead of
    group conv + add.
    """

    def __init__(self, channels, stride=1):
        super().__init__()
        self.stride = stride
        branch_channels = channels // 2

        if stride == 1:
            # Channel split: half stays, half goes through conv
            self.branch = nn.Sequential(
                # 1×1 conv
                nn.Conv2d(branch_channels, branch_channels, 1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                # 3×3 depthwise conv
                nn.Conv2d(branch_channels, branch_channels, 3,
                          padding=1, groups=branch_channels, bias=False),
                nn.BatchNorm2d(branch_channels),
                # 1×1 conv
                nn.Conv2d(branch_channels, branch_channels, 1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
        else:
            # stride=2: both branches process all channels
            self.branch1 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride=2,
                          padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.Conv2d(channels, branch_channels, 1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(channels, branch_channels, 1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_channels, branch_channels, 3, stride=2,
                          padding=1, groups=branch_channels, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.Conv2d(branch_channels, branch_channels, 1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if self.stride == 1:
            # Channel split
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.branch(x2)], dim=1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)

        # Channel shuffle
        return channel_shuffle(out, groups=2)


block = ShuffleNetV2Block(channels=64, stride=1)
x = torch.randn(1, 64, 28, 28)
out = block(x)
print(f"ShuffleNetV2 block: {x.shape} → {out.shape}")
```

---

## 6. SqueezeNet and GhostNet

### 6.1 SqueezeNet: Fire Module

SqueezeNet (Iandola et al., 2016) achieves AlexNet-level accuracy with 50x fewer parameters using the **Fire module**: squeeze (reduce channels with 1x1) then expand (parallel 1x1 and 3x3).

```python
import torch
import torch.nn as nn


class FireModule(nn.Module):
    """
    SqueezeNet Fire Module.

    1. Squeeze: 1×1 conv reduces channels (e.g., 128 → 16)
    2. Expand:  Parallel 1×1 and 3×3 convs, then concatenate
       - 1×1 branch: captures point-wise features
       - 3×3 branch: captures spatial features
    """

    def __init__(self, in_channels, squeeze_channels,
                 expand1x1_channels, expand3x3_channels):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, 1, bias=False),
            nn.BatchNorm2d(squeeze_channels),
            nn.ReLU(inplace=True),
        )
        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand1x1_channels, 1, bias=False),
            nn.BatchNorm2d(expand1x1_channels),
            nn.ReLU(inplace=True),
        )
        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand3x3_channels, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(expand3x3_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        squeezed = self.squeeze(x)
        return torch.cat([
            self.expand1x1(squeezed),
            self.expand3x3(squeezed),
        ], dim=1)


# Fire module: 128 → squeeze to 16 → expand to 64+64 = 128
fire = FireModule(128, squeeze_channels=16, expand1x1_channels=64, expand3x3_channels=64)
x = torch.randn(1, 128, 14, 14)
out = fire(x)
print(f"Fire module: {x.shape} → {out.shape}")
print(f"Parameters: {sum(p.numel() for p in fire.parameters()):,}")

# Compare with standard conv
std = nn.Conv2d(128, 128, 3, padding=1, bias=False)
print(f"Standard Conv: {sum(p.numel() for p in std.parameters()):,} params")
```

### 6.2 GhostNet: Ghost Module

GhostNet (Han et al., 2020) generates feature maps cheaply by producing a few "intrinsic" features with standard convolution and then generating "ghost" features via cheap linear operations.

```python
import torch
import torch.nn as nn
import math


class GhostModule(nn.Module):
    """
    GhostNet Ghost Module.

    Instead of generating all output channels with a standard convolution:
    1. Generate C/s "intrinsic" features with a standard 1×1 conv
    2. Generate (s-1) "ghost" features per intrinsic feature using
       cheap depthwise 3×3 convolutions
    3. Concatenate intrinsic + ghost features

    Reduction factor: roughly s× (default s=2 → 2× reduction)
    """

    def __init__(self, in_channels, out_channels, kernel_size=1,
                 ratio=2, dw_kernel_size=3):
        super().__init__()
        init_channels = math.ceil(out_channels / ratio)
        ghost_channels = init_channels * (ratio - 1)

        # Primary convolution (intrinsic features)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
        )

        # Cheap operation (ghost features via depthwise conv)
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, ghost_channels, dw_kernel_size,
                      padding=dw_kernel_size // 2, groups=init_channels,
                      bias=False),
            nn.BatchNorm2d(ghost_channels),
            nn.ReLU(inplace=True),
        )

        self.out_channels = out_channels

    def forward(self, x):
        intrinsic = self.primary_conv(x)
        ghost = self.cheap_operation(intrinsic)
        out = torch.cat([intrinsic, ghost], dim=1)
        return out[:, :self.out_channels, :, :]  # Trim to exact channel count


# Compare Ghost module vs standard conv
ghost = GhostModule(64, 128, ratio=2)
std = nn.Conv2d(64, 128, 1, bias=False)

ghost_params = sum(p.numel() for p in ghost.parameters())
std_params = sum(p.numel() for p in std.parameters())

x = torch.randn(1, 64, 28, 28)
print(f"Ghost module: {ghost_params:,} params → output {ghost(x).shape}")
print(f"Standard 1×1: {std_params:,} params → output {std(x).shape}")
print(f"Reduction: {std_params / ghost_params:.1f}x")
```

---

## 7. Design Principles for Efficient Networks

### 7.1 Summary of Key Principles

```python
def efficient_network_design_principles():
    """
    Key design principles compiled from MobileNet, ShuffleNet,
    EfficientNet, and other efficient architecture papers.
    """
    principles = {
        "1. Depthwise Separable Conv": (
            "Factor standard conv into depthwise + pointwise. "
            "Reduces FLOPs by K^2 (≈9x for 3×3)."
        ),
        "2. Inverted Residuals": (
            "Expand → depthwise → project. Linear bottleneck "
            "preserves information in narrow layers."
        ),
        "3. Compound Scaling": (
            "Scale depth, width, and resolution together, not independently. "
            "EfficientNet's alpha^phi, beta^phi, gamma^phi."
        ),
        "4. SE Attention": (
            "Squeeze-and-Excitation adds channel attention with minimal cost "
            "(~2-10% more params, +0.5-1% accuracy)."
        ),
        "5. Activation Functions": (
            "ReLU6 (MobileNet), SiLU/Swish (EfficientNet), H-Swish (MobileNetV3). "
            "SiLU provides small but consistent gains."
        ),
        "6. Stochastic Depth": (
            "Randomly drop entire blocks during training for regularization. "
            "Drop rate 0.2-0.3 for deep networks."
        ),
        "7. Balanced Width/Depth": (
            "Very deep but narrow networks are slow due to memory access overhead. "
            "Very wide but shallow networks underfit."
        ),
        "8. Minimize Element-wise Ops": (
            "ShuffleNet V2 finding: element-wise ops (ReLU, Sigmoid, Add) "
            "can dominate runtime on mobile hardware."
        ),
    }

    for name, desc in principles.items():
        print(f"\n{name}")
        print(f"  {desc}")


efficient_network_design_principles()
```

### 7.2 Architecture Comparison Table

| Model | Params | FLOPs | Top-1 (ImageNet) | Key Innovation |
|-------|--------|-------|-------------------|----------------|
| **MobileNetV1** (1.0) | 4.2M | 569M | 70.6% | Depthwise separable conv |
| **MobileNetV2** (1.0) | 3.4M | 300M | 72.0% | Inverted residual + linear bottleneck |
| **MobileNetV3-L** | 5.4M | 219M | 75.2% | H-Swish + NAS + SE |
| **ShuffleNetV2** (1.0) | 2.3M | 146M | 69.4% | Channel split + shuffle |
| **SqueezeNet** | 1.2M | 833M | 57.5% | Fire module (squeeze-expand) |
| **EfficientNet-B0** | 5.3M | 390M | 77.1% | Compound scaling + MBConv + SE |
| **EfficientNet-B3** | 12M | 1.8G | 81.6% | Scaled B0 |
| **GhostNet** (1.0) | 5.2M | 141M | 73.9% | Ghost module (cheap linear ops) |
| **FBNet-C** | 5.5M | 375M | 74.9% | Differentiable NAS |

```python
import torch
import torchvision.models as models
from thop import profile


def benchmark_efficient_models():
    """Compare popular efficient architectures on standard metrics."""
    model_configs = {
        "MobileNetV2": models.mobilenet_v2,
        "MobileNetV3-Small": models.mobilenet_v3_small,
        "MobileNetV3-Large": models.mobilenet_v3_large,
        "EfficientNet-B0": models.efficientnet_b0,
        "ShuffleNetV2-1.0": models.shufflenet_v2_x1_0,
        "SqueezeNet-1.1": models.squeezenet1_1,
    }

    dummy = torch.randn(1, 3, 224, 224)
    print(f"{'Model':<22} {'Params':>8} {'FLOPs':>10} {'Size(MB)':>9}")
    print("-" * 52)

    for name, model_fn in model_configs.items():
        model = model_fn(weights=None)
        model.eval()

        params = sum(p.numel() for p in model.parameters())
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        size_mb = sum(p.numel() * p.element_size()
                      for p in model.parameters()) / 1e6

        print(f"{name:<22} {params/1e6:>7.1f}M {flops/1e9:>9.2f}G {size_mb:>8.1f}")


# benchmark_efficient_models()  # Requires thop: pip install thop
```

---

## 8. Building Custom Efficient Networks

### 8.1 Choosing the Right Building Block

```python
def select_building_block(target_flops_budget, accuracy_priority, hardware):
    """
    Guide for selecting efficient architecture building blocks.
    """
    if target_flops_budget < 100:  # MFLOPs
        print("Ultra-low budget: Use depthwise separable convs")
        print("  → MobileNetV1-style or ShuffleNet V2 blocks")
        print("  → Width multiplier 0.25-0.5")
        return "depthwise_separable"

    elif target_flops_budget < 500:
        if accuracy_priority:
            print("Low budget, accuracy focus: MBConv with SE")
            print("  → EfficientNet-B0 style")
            return "mbconv_se"
        else:
            print("Low budget, speed focus: Inverted residual")
            print("  → MobileNetV2 style, no SE")
            return "inverted_residual"

    elif target_flops_budget < 2000:
        print("Medium budget: MBConv + compound scaling")
        print("  → EfficientNet-B1 to B3 range")
        return "mbconv_scaled"

    else:
        print("High budget: Standard architectures are fine")
        print("  → ResNet, ConvNeXt, ViT")
        return "standard"


# Examples
select_building_block(50, accuracy_priority=False, hardware="mcu")
print()
select_building_block(300, accuracy_priority=True, hardware="mobile")
```

---

## Summary

| Architecture | Key Technique | FLOPs Reduction | Best For |
|-------------|---------------|----------------|----------|
| **MobileNetV1** | Depthwise separable conv | ~9x per layer | Mobile, straightforward |
| **MobileNetV2** | Inverted residual + linear bottleneck | ~9x + better accuracy | Mobile, general purpose |
| **EfficientNet** | Compound scaling + MBConv + SE | Optimal scaling | Best accuracy/FLOPs tradeoff |
| **ShuffleNet** | Channel shuffle + split | ~5-9x | Ultra-low FLOPs |
| **SqueezeNet** | Fire module (squeeze-expand) | ~50x fewer params | Minimal model size |
| **GhostNet** | Ghost features via cheap ops | ~2x per module | Drop-in replacement |

---

## Exercises

### Exercise 1: Depthwise Separable Convolution

1. Implement a standard 3x3 Conv2d and a DepthwiseSeparableConv with the same in/out channels
2. Count parameters and FLOPs for both
3. Verify the theoretical reduction ratio (1/C_out + 1/K^2)
4. Benchmark inference time on CPU for input size (1, 64, 224, 224) with output channels 128

### Exercise 2: MobileNetV2 on CIFAR-10

1. Build a MobileNetV2-style model (scaled down for CIFAR-10's 32x32 images)
2. Train on CIFAR-10 and record accuracy
3. Compare with a standard CNN of similar parameter count
4. Apply width multipliers (0.5, 0.75, 1.0) and plot accuracy vs FLOPs

### Exercise 3: Compound Scaling Experiment

1. Start with a tiny base model (e.g., 3 MBConv blocks, width=16, resolution=32)
2. Scale using three strategies: (a) only depth, (b) only width, (c) compound scaling
3. For each strategy, scale to approximately the same total FLOPs
4. Train each variant on CIFAR-10 and compare accuracy
5. Does compound scaling consistently outperform single-dimension scaling?

### Exercise 4: Architecture Comparison

1. Train MobileNetV2, ShuffleNetV2, and EfficientNet-B0 (from torchvision) on CIFAR-10
2. Measure: accuracy, parameter count, FLOPs, and inference latency on CPU
3. Plot accuracy vs latency and accuracy vs model size
4. Which model offers the best tradeoff for each metric?

### Exercise 5: Custom Efficient Block

1. Design your own efficient convolution block combining ideas from this lesson
2. Requirements: depthwise separable conv + some form of attention (SE or similar) + skip connection
3. Build a small network using your block and train on CIFAR-10
4. Compare with standard MBConv of similar parameter count

---

[Previous: Knowledge Distillation](./05_Knowledge_Distillation.md) | [Overview](./00_Overview.md) | [Next: Neural Architecture Search](./07_Neural_Architecture_Search.md)

**License**: CC BY-NC 4.0
