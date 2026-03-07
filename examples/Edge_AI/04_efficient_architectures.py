"""
04. Efficient Architectures

Demonstrates MobileNet-style depthwise separable convolutions
and compares FLOPs/parameters with standard convolutions.

Covers:
- Standard convolution vs depthwise separable convolution
- MobileNet V1 building block implementation
- MobileNet V2 inverted residual block
- FLOPs and parameter count comparison
- Width multiplier and resolution multiplier

Requirements:
    pip install torch torchvision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("Edge AI — Efficient Architectures")
print("=" * 60)


# ============================================
# 1. Standard Convolution
# ============================================
print("\n[1] Standard Convolution")
print("-" * 40)


class StandardConv(nn.Module):
    """Standard convolution block: Conv2d + BN + ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size,
            stride=stride, padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# Standard conv: 32 -> 64 channels, 3x3 kernel
std_conv = StandardConv(32, 64, kernel_size=3)
std_params = sum(p.numel() for p in std_conv.parameters())
print(f"Standard Conv (32->64, 3x3):")
print(f"  Parameters: {std_params:,}")
print(f"  Computation: in_ch * out_ch * k * k * H * W")
print(f"  = 32 * 64 * 3 * 3 * H * W = {32 * 64 * 9:,} * H * W multiply-adds")


# ============================================
# 2. Depthwise Separable Convolution
# ============================================
print("\n[2] Depthwise Separable Convolution")
print("-" * 40)
print("Splits standard conv into two steps:")
print("  1. Depthwise: one filter per input channel (spatial filtering)")
print("  2. Pointwise: 1x1 conv to mix channels (channel mixing)\n")


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: Depthwise + Pointwise."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        # Depthwise: groups=in_ch means one filter per channel
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size,
            stride=stride, padding=kernel_size // 2,
            groups=in_ch, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)

        # Pointwise: 1x1 convolution to mix channels
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.depthwise(x)))
        x = self.relu2(self.bn2(self.pointwise(x)))
        return x


# Depthwise separable conv: 32 -> 64 channels, 3x3 kernel
dws_conv = DepthwiseSeparableConv(32, 64, kernel_size=3)
dws_params = sum(p.numel() for p in dws_conv.parameters())
print(f"Depthwise Separable Conv (32->64, 3x3):")
print(f"  Parameters: {dws_params:,}")
print(f"  Depthwise:  in_ch * k * k = {32 * 9} (spatial)")
print(f"  Pointwise:  in_ch * out_ch * 1 * 1 = {32 * 64:,} (channel)")
print(f"  Total compute: {32 * 9 + 32 * 64:,} * H * W multiply-adds")
print(f"\n  Reduction ratio: {std_params / dws_params:.1f}x fewer parameters")
print(f"  FLOPs ratio:     {(32 * 64 * 9) / (32 * 9 + 32 * 64):.1f}x fewer FLOPs")


# ============================================
# 3. MobileNet V1 Block
# ============================================
print("\n[3] MobileNet V1 Building Block")
print("-" * 40)


class MobileNetV1Block(nn.Module):
    """MobileNet V1: stack of depthwise separable convolutions."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, 3, stride=stride, padding=1,
            groups=in_ch, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.dw(x)))
        x = self.relu(self.bn2(self.pw(x)))
        return x


v1_block = MobileNetV1Block(64, 128, stride=2)
x = torch.randn(1, 64, 56, 56)
out = v1_block(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print(f"Params: {sum(p.numel() for p in v1_block.parameters()):,}")


# ============================================
# 4. MobileNet V2 Inverted Residual Block
# ============================================
print("\n[4] MobileNet V2 Inverted Residual Block")
print("-" * 40)
print("Key idea: expand to higher dimension, apply depthwise conv,")
print("then project back to lower dimension. Uses linear bottleneck.\n")


class InvertedResidual(nn.Module):
    """MobileNet V2 inverted residual block with linear bottleneck."""

    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        mid_ch = in_ch * expand_ratio

        layers = []
        # Expansion (pointwise) — skip if expand_ratio == 1
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU6(inplace=True),
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(
                mid_ch, mid_ch, 3, stride=stride, padding=1,
                groups=mid_ch, bias=False
            ),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
        ])

        # Projection (pointwise, NO activation — linear bottleneck)
        layers.extend([
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


v2_block = InvertedResidual(24, 24, stride=1, expand_ratio=6)
x = torch.randn(1, 24, 56, 56)
out = v2_block(x)
print(f"Inverted Residual (24->24, expand=6, stride=1):")
print(f"  Input:    {x.shape}")
print(f"  Expanded: [1, {24 * 6}, 56, 56]")
print(f"  Output:   {out.shape}")
print(f"  Residual: {v2_block.use_residual}")
print(f"  Params:   {sum(p.numel() for p in v2_block.parameters()):,}")


# ============================================
# 5. FLOPs Comparison
# ============================================
print("\n[5] FLOPs Comparison: Standard vs Depthwise Separable")
print("-" * 40)


def count_conv_flops(in_ch, out_ch, kernel_size, h, w, groups=1):
    """Count multiply-add operations for a convolution."""
    k = kernel_size
    flops_per_output = (in_ch // groups) * k * k
    output_elements = out_ch * h * w
    return flops_per_output * output_elements


h, w = 224, 224

configs = [
    ("3->64, k=3", 3, 64, 3),
    ("64->128, k=3", 64, 128, 3),
    ("128->256, k=3", 128, 256, 3),
    ("256->512, k=3", 256, 512, 3),
]

print(f"{'Config':<18} {'Standard FLOPs':>18} {'DW Sep FLOPs':>18} {'Ratio':>8}")
print("-" * 66)

for name, in_ch, out_ch, k in configs:
    # Standard conv FLOPs
    std_flops = count_conv_flops(in_ch, out_ch, k, h, w)

    # Depthwise separable FLOPs
    dw_flops = count_conv_flops(in_ch, in_ch, k, h, w, groups=in_ch)  # depthwise
    pw_flops = count_conv_flops(in_ch, out_ch, 1, h, w)  # pointwise
    dws_flops = dw_flops + pw_flops

    ratio = std_flops / dws_flops

    print(f"{name:<18} {std_flops:>18,} {dws_flops:>18,} {ratio:>7.1f}x")

    # Spatial dimensions halve after each block
    h, w = h // 2, w // 2


# ============================================
# 6. Width Multiplier
# ============================================
print("\n[6] Width Multiplier (alpha)")
print("-" * 40)
print("Scales the number of channels by alpha to trade accuracy for speed.\n")


def build_mobilenet_v1(alpha=1.0, num_classes=1000):
    """Build a simplified MobileNet V1 with width multiplier."""
    def ch(c):
        return max(1, int(c * alpha))

    layers = [
        StandardConv(3, ch(32), kernel_size=3, stride=2),
        MobileNetV1Block(ch(32), ch(64)),
        MobileNetV1Block(ch(64), ch(128), stride=2),
        MobileNetV1Block(ch(128), ch(128)),
        MobileNetV1Block(ch(128), ch(256), stride=2),
        MobileNetV1Block(ch(256), ch(256)),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(ch(256), num_classes),
    ]
    return nn.Sequential(*layers)


print(f"{'Alpha':<8} {'Total Params':>15}")
print("-" * 25)
for alpha in [1.0, 0.75, 0.5, 0.25]:
    model = build_mobilenet_v1(alpha=alpha, num_classes=10)
    params = sum(p.numel() for p in model.parameters())
    print(f"{alpha:<8} {params:>15,}")

print()
print("Key takeaways:")
print("- Depthwise separable convs reduce FLOPs by ~8-9x vs standard")
print("- MobileNet V2 adds inverted residuals and linear bottlenecks")
print("- Width multiplier (alpha) trades accuracy for efficiency")
print("- Resolution multiplier (rho) trades input size for speed")
