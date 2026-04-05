"""
Exercises for Lesson 06: Efficient Architectures
Topic: Edge_AI

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# === Exercise 1: Depthwise Separable Conv FLOPs Derivation ===
# Problem: Derive and verify the FLOPs reduction factor for depthwise
# separable convolutions compared to standard convolutions.

def exercise_1():
    """Derive FLOPs reduction factor for depthwise separable convolution."""
    print("  Standard Convolution FLOPs:")
    print("    F_std = D_k * D_k * C_in * C_out * H * W")
    print()
    print("  Depthwise Separable Convolution FLOPs:")
    print("    F_dw  = D_k * D_k * C_in * H * W        (depthwise)")
    print("    F_pw  = C_in * C_out * H * W              (pointwise)")
    print("    F_dws = F_dw + F_pw")
    print()
    print("  Reduction ratio:")
    print("    F_std / F_dws = (D_k^2 * C_in * C_out) / (D_k^2 * C_in + C_in * C_out)")
    print("                  = 1 / (1/C_out + 1/D_k^2)")
    print()

    # Verify with concrete numbers
    configs = [
        (3, 32, 64, 56, 56),    # Typical early layer
        (3, 64, 128, 28, 28),   # Mid layer
        (3, 128, 256, 14, 14),  # Late layer
        (3, 256, 512, 7, 7),    # Very late layer
        (5, 64, 128, 28, 28),   # 5x5 kernel
    ]

    print(f"  {'Config':<25} {'F_std':>12} {'F_dws':>12} {'Ratio':>8} {'Theoretical':>12}")
    print("  " + "-" * 72)

    for dk, cin, cout, h, w in configs:
        f_std = dk * dk * cin * cout * h * w
        f_dw = dk * dk * cin * h * w
        f_pw = cin * cout * h * w
        f_dws = f_dw + f_pw
        ratio = f_std / f_dws
        theoretical = 1 / (1 / cout + 1 / (dk * dk))

        config_str = f"k={dk},in={cin},out={cout}"
        print(f"  {config_str:<25} {f_std:>12,} {f_dws:>12,} {ratio:>7.1f}x {theoretical:>11.1f}x")

    print("\n  For 3x3 convs with large C_out: reduction is roughly 8-9x")
    print("  For 5x5 convs: reduction increases (1/25 vs 1/9 factor)")


# === Exercise 2: MobileNet V2 Inverted Residual Analysis ===
# Problem: Implement an inverted residual block and analyze the
# expansion-projection pattern's impact on FLOPs and params.

def exercise_2():
    """Analyze MobileNet V2 inverted residual block."""

    class InvertedResidual(nn.Module):
        def __init__(self, cin, cout, expand_ratio, stride=1):
            super().__init__()
            mid = cin * expand_ratio
            self.use_residual = stride == 1 and cin == cout
            layers = []
            if expand_ratio != 1:
                layers += [nn.Conv2d(cin, mid, 1, bias=False),
                           nn.BatchNorm2d(mid), nn.ReLU6(inplace=True)]
            layers += [nn.Conv2d(mid, mid, 3, stride=stride, padding=1,
                                 groups=mid, bias=False),
                       nn.BatchNorm2d(mid), nn.ReLU6(inplace=True)]
            layers += [nn.Conv2d(mid, cout, 1, bias=False),
                       nn.BatchNorm2d(cout)]
            self.block = nn.Sequential(*layers)

        def forward(self, x):
            out = self.block(x)
            return x + out if self.use_residual else out

    print("  Inverted Residual: thin -> EXPAND -> depthwise -> PROJECT -> thin\n")

    # Analyze different expansion ratios
    cin, cout, h, w = 24, 24, 56, 56
    print(f"  Input: {cin} channels, {h}x{w}")
    print(f"  {'Expand':>8} {'Mid Ch':>8} {'Params':>10} {'Residual':>10}")
    print("  " + "-" * 40)

    for t in [1, 2, 3, 6, 8]:
        block = InvertedResidual(cin, cout, expand_ratio=t)
        params = sum(p.numel() for p in block.parameters())
        print(f"  {t:>8} {cin * t:>8} {params:>10,} {'Yes':>10}")

    print("\n  Why 'inverted'?")
    print("    Classical residual: wide -> narrow -> wide (bottleneck)")
    print("    Inverted residual: narrow -> wide -> narrow")
    print("    Key insight: the high-dimensional expanded space is")
    print("    internal to the block; the residual connection operates")
    print("    on the low-dimensional bottleneck representation.")
    print("    This saves both memory (small skip tensors) and compute.")


# === Exercise 3: Width Multiplier and Resolution Multiplier ===
# Problem: Calculate how width and resolution multipliers affect
# FLOPs and parameters independently.

def exercise_3():
    """Analyze width and resolution multiplier effects."""

    def mobilenet_flops(alpha=1.0, rho=1.0, base_channels=None):
        """Estimate MobileNet FLOPs with multipliers."""
        if base_channels is None:
            base_channels = [32, 64, 128, 128, 256, 256, 512, 512, 512,
                             512, 512, 512, 1024, 1024]

        h = int(224 * rho)
        w = h
        total_flops = 0
        total_params = 0

        for i, cout in enumerate(base_channels):
            cin = int(base_channels[i - 1] * alpha) if i > 0 else 3
            cout = int(cout * alpha)
            stride = 2 if i in [0, 2, 4, 6] else 1

            # Depthwise conv
            dw_flops = 3 * 3 * cin * h * w
            dw_params = 3 * 3 * cin
            # Pointwise conv
            pw_flops = cin * cout * h * w
            pw_params = cin * cout

            total_flops += dw_flops + pw_flops
            total_params += dw_params + pw_params

            if stride == 2:
                h = h // 2
                w = w // 2

        return total_flops, total_params

    print("  Width Multiplier (alpha) — scales channel count:\n")
    print(f"  {'alpha':>6} {'FLOPs':>15} {'Params':>12} {'FLOPs ratio':>12}")
    print("  " + "-" * 48)
    base_flops, _ = mobilenet_flops(alpha=1.0)
    for alpha in [1.0, 0.75, 0.5, 0.25]:
        flops, params = mobilenet_flops(alpha=alpha)
        print(f"  {alpha:>6.2f} {flops:>15,} {params:>12,} "
              f"{flops/base_flops:>11.2f}x")

    print(f"\n  FLOPs scale as O(alpha^2) because both C_in and C_out are scaled")

    print("\n  Resolution Multiplier (rho) — scales input resolution:\n")
    print(f"  {'rho':>6} {'Resolution':>12} {'FLOPs':>15} {'FLOPs ratio':>12}")
    print("  " + "-" * 48)
    for rho in [1.0, 0.857, 0.714, 0.571]:
        flops, params = mobilenet_flops(rho=rho)
        res = int(224 * rho)
        print(f"  {rho:>6.3f} {res:>7}x{res:<4} {flops:>15,} "
              f"{flops/base_flops:>11.2f}x")

    print(f"\n  FLOPs scale as O(rho^2) because H and W are both scaled")

    # Combined effect
    print("\n  Combined alpha x rho:")
    for alpha, rho in [(0.75, 0.857), (0.5, 0.714), (0.25, 0.571)]:
        flops, params = mobilenet_flops(alpha=alpha, rho=rho)
        res = int(224 * rho)
        print(f"    alpha={alpha}, rho={rho}: {flops:,} FLOPs "
              f"({flops/base_flops:.2f}x), {res}x{res}")


# === Exercise 4: Efficient Architecture Comparison ===
# Problem: Compare MobileNet, ShuffleNet, and SqueezeNet design
# principles and their efficiency tradeoffs.

def exercise_4():
    """Compare efficient architecture design principles."""
    architectures = [
        {
            "name": "MobileNet V1",
            "year": 2017,
            "key_op": "Depthwise Separable Conv",
            "params": "3.4M",
            "flops": "569M",
            "top1_imagenet": "70.6%",
            "principle": (
                "Replace standard convolutions with depthwise separable "
                "convolutions everywhere. Width multiplier for scaling."
            ),
        },
        {
            "name": "MobileNet V2",
            "year": 2018,
            "key_op": "Inverted Residual + Linear Bottleneck",
            "params": "3.4M",
            "flops": "300M",
            "top1_imagenet": "72.0%",
            "principle": (
                "Inverted residual: expand-depthwise-project. Linear "
                "bottleneck (no ReLU in projection) preserves information."
            ),
        },
        {
            "name": "ShuffleNet V2",
            "year": 2018,
            "key_op": "Channel Split + Channel Shuffle",
            "params": "2.3M",
            "flops": "146M",
            "top1_imagenet": "69.4%",
            "principle": (
                "Channel split: half channels skip, half processed. "
                "Channel shuffle: redistribute channels across groups. "
                "Practical guidelines: equal channel widths, reduce group conv."
            ),
        },
        {
            "name": "EfficientNet-B0",
            "year": 2019,
            "key_op": "MBConv + Compound Scaling",
            "params": "5.3M",
            "flops": "390M",
            "top1_imagenet": "77.1%",
            "principle": (
                "Compound scaling: scale depth, width, and resolution "
                "together with fixed ratios (d=1.2, w=1.1, r=1.15). "
                "NAS-discovered base architecture."
            ),
        },
        {
            "name": "SqueezeNet",
            "year": 2016,
            "key_op": "Fire Module (Squeeze + Expand)",
            "params": "1.2M",
            "flops": "833M",
            "top1_imagenet": "57.5%",
            "principle": (
                "Fire module: squeeze layer (1x1 conv, few channels) "
                "followed by expand layer (mix of 1x1 and 3x3). "
                "Reduces parameters but FLOPs can be high."
            ),
        },
    ]

    print("  Efficient Architecture Comparison:\n")
    print(f"  {'Architecture':<18} {'Year':>5} {'Params':>8} {'FLOPs':>8} {'Top-1':>7}")
    print("  " + "-" * 50)
    for arch in architectures:
        print(f"  {arch['name']:<18} {arch['year']:>5} {arch['params']:>8} "
              f"{arch['flops']:>8} {arch['top1_imagenet']:>7}")

    print("\n  Design principles:")
    for arch in architectures:
        print(f"\n  {arch['name']}:")
        print(f"    Key operation: {arch['key_op']}")
        print(f"    Principle: {arch['principle']}")

    print("\n  Key insight: FLOPs alone don't determine actual speed.")
    print("  Memory access cost (MAC), parallelism, and hardware support")
    print("  all affect real-world inference time. ShuffleNet V2 showed")
    print("  that practical speed guidelines matter more than FLOPs.")


if __name__ == "__main__":
    print("=== Exercise 1: Depthwise Separable Conv FLOPs ===")
    exercise_1()
    print("\n=== Exercise 2: MobileNet V2 Inverted Residual ===")
    exercise_2()
    print("\n=== Exercise 3: Width and Resolution Multipliers ===")
    exercise_3()
    print("\n=== Exercise 4: Efficient Architecture Comparison ===")
    exercise_4()
    print("\nAll exercises completed!")
