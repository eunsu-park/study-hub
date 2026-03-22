"""
Exercises for Lesson 37: Modern Deep Learning Architectures
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# === Exercise 1: ConvNeXt Block ===
# Problem: Implement a ConvNeXt block from scratch.
# Key design choices vs ResNet: depthwise conv 7x7, inverted bottleneck,
# LayerNorm instead of BatchNorm, GELU instead of ReLU.

def exercise_1():
    """Implement and test a ConvNeXt block."""

    class ConvNeXtBlock(nn.Module):
        """ConvNeXt block: depthwise conv + inverted bottleneck + LayerNorm + GELU."""

        def __init__(self, dim, expansion_ratio=4, kernel_size=7, layer_scale_init=1e-6):
            super().__init__()
            # Depthwise separable convolution (groups=dim means per-channel)
            self.dwconv = nn.Conv2d(
                dim, dim, kernel_size=kernel_size,
                padding=kernel_size // 2, groups=dim
            )
            self.norm = nn.LayerNorm(dim, eps=1e-6)
            self.pwconv1 = nn.Linear(dim, expansion_ratio * dim)  # Expand
            self.act = nn.GELU()
            self.pwconv2 = nn.Linear(expansion_ratio * dim, dim)  # Project
            # Learnable per-channel scale (stabilizes training)
            self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))

        def forward(self, x):
            shortcut = x
            x = self.dwconv(x)
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C) for LayerNorm
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            x = self.gamma * x
            x = x.permute(0, 3, 1, 2)  # Back to (N, C, H, W)
            return shortcut + x

    torch.manual_seed(42)
    block = ConvNeXtBlock(dim=64)
    x = torch.randn(2, 64, 56, 56)
    y = block(x)

    print("  Input shape: {}".format(x.shape))
    print("  Output shape: {}".format(y.shape))
    print("  Shape preserved (residual): {}".format(x.shape == y.shape))

    # Count parameters
    n_params = sum(p.numel() for p in block.parameters())
    print("  Block parameters: {:,}".format(n_params))


# === Exercise 2: Depthwise vs Standard Convolution — Parameter Count ===
# Problem: Compare the number of parameters in a standard 3x3 conv vs a
# depthwise separable conv for C_in=64, C_out=128, kernel=3.

def exercise_2():
    """Compare parameter counts: standard vs depthwise separable convolution."""
    C_in, C_out, K = 64, 128, 3

    # Standard conv: C_out * C_in * K * K + C_out (bias)
    std_params = C_out * C_in * K * K + C_out
    print("  Standard conv ({}->{}, {}x{}): {:,} params".format(C_in, C_out, K, K, std_params))

    # Depthwise separable: depthwise + pointwise
    # Depthwise: C_in * 1 * K * K + C_in (one filter per input channel)
    # Pointwise: C_out * C_in * 1 * 1 + C_out
    dw_params = C_in * K * K + C_in
    pw_params = C_out * C_in * 1 * 1 + C_out
    dws_params = dw_params + pw_params
    print("  Depthwise separable ({}->{}, {}x{}): {:,} params".format(
        C_in, C_out, K, K, dws_params))
    print("  Reduction ratio: {:.2f}x".format(std_params / dws_params))

    # Verify with PyTorch
    std_conv = nn.Conv2d(C_in, C_out, K, padding=1)
    dw_conv = nn.Conv2d(C_in, C_in, K, padding=1, groups=C_in)
    pw_conv = nn.Conv2d(C_in, C_out, 1)

    pt_std = sum(p.numel() for p in std_conv.parameters())
    pt_dws = sum(p.numel() for p in dw_conv.parameters()) + \
             sum(p.numel() for p in pw_conv.parameters())
    print("  PyTorch verification - standard: {:,}, DWS: {:,}".format(pt_std, pt_dws))


# === Exercise 3: EfficientNet Fused-MBConv vs Standard MBConv ===
# Problem: Implement a minimal MBConv block (MobileNetV2 style) and a
# Fused-MBConv block (EfficientNetV2 style), and compare their FLOPs/params.

def exercise_3():
    """Implement MBConv vs Fused-MBConv blocks."""

    class MBConv(nn.Module):
        """Mobile Inverted Bottleneck Conv: expand -> depthwise -> project."""

        def __init__(self, in_channels, out_channels, expand_ratio=4, stride=1):
            super().__init__()
            mid_channels = in_channels * expand_ratio
            self.use_residual = (stride == 1 and in_channels == out_channels)
            self.block = nn.Sequential(
                # Expand with pointwise
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(),
                # Depthwise conv
                nn.Conv2d(mid_channels, mid_channels, 3, stride=stride,
                          padding=1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(),
                # Project
                nn.Conv2d(mid_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        def forward(self, x):
            out = self.block(x)
            if self.use_residual:
                return x + out
            return out

    class FusedMBConv(nn.Module):
        """Fused-MBConv: fused expand+depthwise -> project (fewer ops on accelerators)."""

        def __init__(self, in_channels, out_channels, expand_ratio=4, stride=1):
            super().__init__()
            mid_channels = in_channels * expand_ratio
            self.use_residual = (stride == 1 and in_channels == out_channels)
            self.block = nn.Sequential(
                # Fused: standard 3x3 conv replaces expand+depthwise
                nn.Conv2d(in_channels, mid_channels, 3, stride=stride,
                          padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(),
                # Project
                nn.Conv2d(mid_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        def forward(self, x):
            out = self.block(x)
            if self.use_residual:
                return x + out
            return out

    torch.manual_seed(0)
    x = torch.randn(2, 32, 56, 56)

    mb = MBConv(32, 32, expand_ratio=4)
    fused = FusedMBConv(32, 32, expand_ratio=4)

    mb_params = sum(p.numel() for p in mb.parameters())
    fused_params = sum(p.numel() for p in fused.parameters())

    print("  MBConv output shape: {}".format(mb(x).shape))
    print("  FusedMBConv output shape: {}".format(fused(x).shape))
    print("  MBConv params: {:,}".format(mb_params))
    print("  FusedMBConv params: {:,}".format(fused_params))
    print("  Note: FusedMBConv is more hardware-friendly on accelerators "
          "(fewer kernel launches)")


# === Exercise 4: Architecture Selection with timm (conceptual) ===
# Problem: Without loading actual weights, demonstrate how to inspect
# modern architecture families using pure PyTorch equivalents and compare
# model complexity (parameters, output shapes).

def exercise_4():
    """Compare modern architecture building blocks with parameter counts."""

    # Simplified ResNet-style block vs ConvNeXt-style block for same channels
    class ResNetBlock(nn.Module):
        """Standard residual block (BatchNorm + ReLU)."""

        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(x + out)

    class ConvNeXtBlock(nn.Module):
        """ConvNeXt block (LayerNorm + GELU + inverted bottleneck)."""

        def __init__(self, channels):
            super().__init__()
            self.dwconv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
            self.norm = nn.LayerNorm(channels)
            self.pwconv1 = nn.Linear(channels, 4 * channels)
            self.pwconv2 = nn.Linear(4 * channels, channels)
            self.gamma = nn.Parameter(1e-6 * torch.ones(channels))

        def forward(self, x):
            res = x
            x = self.dwconv(x)
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = F.gelu(self.pwconv1(x))
            x = self.pwconv2(x)
            x = self.gamma * x
            return res + x.permute(0, 3, 1, 2)

    channels = 96
    resnet_block = ResNetBlock(channels)
    convnext_block = ConvNeXtBlock(channels)

    x = torch.randn(1, channels, 56, 56)

    resnet_params = sum(p.numel() for p in resnet_block.parameters())
    convnext_params = sum(p.numel() for p in convnext_block.parameters())

    print("  ResNet block params: {:,}".format(resnet_params))
    print("  ConvNeXt block params: {:,}".format(convnext_params))
    print("  ResNet output shape: {}".format(resnet_block(x).shape))
    print("  ConvNeXt output shape: {}".format(convnext_block(x).shape))
    print("  ConvNeXt key improvements: 7x7 DW-conv, LayerNorm, GELU, "
          "inverted bottleneck, layer scale")


if __name__ == "__main__":
    print("=== Exercise 1: ConvNeXt Block Implementation ===")
    exercise_1()
    print("\n=== Exercise 2: Depthwise vs Standard Convolution Params ===")
    exercise_2()
    print("\n=== Exercise 3: MBConv vs Fused-MBConv ===")
    exercise_3()
    print("\n=== Exercise 4: ResNet vs ConvNeXt Block Comparison ===")
    exercise_4()
    print("\nAll exercises completed!")
