"""
Exercises for Lesson 03: Quantization
Topic: Edge_AI

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn as nn
import torch.quantization as quant
import numpy as np


# === Exercise 1: Scale and Zero-Point Computation ===
# Problem: Manually compute quantization parameters (scale, zero_point)
# for affine (asymmetric) and symmetric quantization.

def exercise_1():
    """Compute quantization scale and zero-point manually."""
    # Float range to quantize
    float_min, float_max = -2.5, 7.3
    # INT8 range
    qmin, qmax = -128, 127

    print("  Asymmetric (Affine) Quantization:")
    print(f"    Float range: [{float_min}, {float_max}]")
    print(f"    Int8 range:  [{qmin}, {qmax}]")

    # Scale: maps float range to int range
    scale = (float_max - float_min) / (qmax - qmin)
    # Zero point: the int value that maps to float 0
    zero_point = round(qmin - float_min / scale)
    zero_point = max(qmin, min(qmax, zero_point))  # Clamp to int range

    print(f"    scale = ({float_max} - {float_min}) / ({qmax} - {qmin}) = {scale:.6f}")
    print(f"    zero_point = round({qmin} - {float_min}/{scale:.6f}) = {zero_point}")

    # Quantize and dequantize example values
    test_values = [0.0, 1.0, -2.5, 7.3, 3.14]
    print(f"\n    {'Float':>8} -> {'Quantized':>10} -> {'Dequantized':>12} {'Error':>10}")
    for v in test_values:
        q = round(v / scale + zero_point)
        q = max(qmin, min(qmax, q))  # Clamp
        deq = (q - zero_point) * scale
        err = abs(v - deq)
        print(f"    {v:>8.3f} -> {q:>10} -> {deq:>12.6f} {err:>10.6f}")

    print(f"\n    Max quantization error: {scale / 2:.6f} (= scale/2)")

    # Symmetric quantization
    print("\n  Symmetric Quantization:")
    abs_max = max(abs(float_min), abs(float_max))
    sym_scale = abs_max / qmax
    sym_zp = 0  # Always zero for symmetric

    print(f"    abs_max = {abs_max}")
    print(f"    scale = {abs_max} / {qmax} = {sym_scale:.6f}")
    print(f"    zero_point = {sym_zp} (always zero)")
    print(f"    Effective range: [{-abs_max}, {abs_max}]")
    print(f"    Wasted range: {abs_max - abs(float_min):.1f} on negative side"
          if abs_max == float_max else
          f"    Wasted range: {abs_max - float_max:.1f} on positive side")


# === Exercise 2: PTQ vs QAT Comparison ===
# Problem: Implement and compare Post-Training Quantization and
# Quantization-Aware Training on a simple model.

def exercise_2():
    """Compare PTQ and QAT approaches."""
    torch.manual_seed(42)

    # Simple model
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant = quant.QuantStub()
            self.fc1 = nn.Linear(784, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 10)
            self.dequant = quant.DeQuantStub()

        def forward(self, x):
            x = self.quant(x)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            x = self.dequant(x)
            return x

    # Generate synthetic data
    X = torch.randn(500, 784)
    y = torch.randint(0, 10, (500,))

    # Train FP32 model
    model_fp32 = SimpleNet()
    optimizer = torch.optim.Adam(model_fp32.parameters(), lr=1e-3)
    model_fp32.train()
    for epoch in range(20):
        out = model_fp32(X)
        loss = nn.CrossEntropyLoss()(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model_fp32.eval()

    # FP32 baseline accuracy
    with torch.no_grad():
        fp32_preds = model_fp32(X).argmax(dim=1)
        fp32_acc = (fp32_preds == y).float().mean().item()

    # PTQ: quantize after training
    model_ptq = SimpleNet()
    model_ptq.load_state_dict(model_fp32.state_dict())
    model_ptq.eval()
    model_ptq.qconfig = quant.get_default_qconfig("fbgemm")
    model_ptq_prepared = quant.prepare(model_ptq)

    # Calibration
    with torch.no_grad():
        model_ptq_prepared(X[:100])
    model_ptq_quantized = quant.convert(model_ptq_prepared)

    with torch.no_grad():
        ptq_preds = model_ptq_quantized(X).argmax(dim=1)
        ptq_acc = (ptq_preds == y).float().mean().item()

    # QAT: simulate quantization during training
    model_qat = SimpleNet()
    model_qat.train()
    model_qat.qconfig = quant.get_default_qat_qconfig("fbgemm")
    model_qat_prepared = quant.prepare_qat(model_qat)

    optimizer_qat = torch.optim.Adam(model_qat_prepared.parameters(), lr=1e-3)
    for epoch in range(20):
        out = model_qat_prepared(X)
        loss = nn.CrossEntropyLoss()(out, y)
        optimizer_qat.zero_grad()
        loss.backward()
        optimizer_qat.step()

    model_qat_prepared.eval()
    model_qat_quantized = quant.convert(model_qat_prepared)

    with torch.no_grad():
        qat_preds = model_qat_quantized(X).argmax(dim=1)
        qat_acc = (qat_preds == y).float().mean().item()

    print(f"  {'Method':<20} {'Accuracy':>10}")
    print("  " + "-" * 32)
    print(f"  {'FP32 (baseline)':<20} {fp32_acc:>10.1%}")
    print(f"  {'PTQ (INT8)':<20} {ptq_acc:>10.1%}")
    print(f"  {'QAT (INT8)':<20} {qat_acc:>10.1%}")

    print(f"\n  PTQ accuracy drop: {(fp32_acc - ptq_acc)*100:.1f}%")
    print(f"  QAT accuracy drop: {(fp32_acc - qat_acc)*100:.1f}%")
    print("  QAT typically recovers more accuracy by learning to handle")
    print("  quantization noise during training.")


# === Exercise 3: Mixed-Precision Quantization ===
# Problem: Design a mixed-precision strategy where sensitive layers
# stay at higher precision while others use INT8.

def exercise_3():
    """Design a mixed-precision quantization strategy."""

    class MixedPrecisionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)   # First layer: sensitive
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)   # Middle: can quantize
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # Middle: can quantize
            self.bn3 = nn.BatchNorm2d(128)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(128, 10)                    # Last layer: sensitive

        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)))
            x = torch.relu(self.bn3(self.conv3(x)))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    model = MixedPrecisionNet()

    # Analyze parameter distribution per layer
    print("  Layer sensitivity analysis:\n")
    print(f"  {'Layer':<12} {'Params':>10} {'Weight Range':>20} {'Sensitivity':<15}")
    print("  " + "-" * 60)

    sensitive_layers = {"conv1", "fc"}  # First and last typically sensitive

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight
            w_range = f"[{w.min().item():.3f}, {w.max().item():.3f}]"
            n_params = w.numel()
            sensitivity = "HIGH (keep FP32)" if name in sensitive_layers else "LOW (use INT8)"
            print(f"  {name:<12} {n_params:>10,} {w_range:>20} {sensitivity}")

    # Calculate mixed-precision savings
    total_params = sum(p.numel() for p in model.parameters())
    int8_params = 0
    fp32_params = 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            n = module.weight.numel()
            if name in sensitive_layers:
                fp32_params += n
            else:
                int8_params += n

    mixed_size = fp32_params * 4 + int8_params * 1
    fp32_size = total_params * 4
    int8_size = total_params * 1

    print(f"\n  Size comparison:")
    print(f"    Full FP32:       {fp32_size / 1024:.1f} KB (1.0x)")
    print(f"    Full INT8:       {int8_size / 1024:.1f} KB ({fp32_size/int8_size:.1f}x)")
    print(f"    Mixed precision: {mixed_size / 1024:.1f} KB ({fp32_size/mixed_size:.1f}x)")
    print(f"    FP32 layers: {fp32_params:,} params ({fp32_params/total_params:.0%})")
    print(f"    INT8 layers: {int8_params:,} params ({int8_params/total_params:.0%})")


# === Exercise 4: Quantization Error Analysis ===
# Problem: Measure and visualize quantization error at different bit widths.

def exercise_4():
    """Analyze quantization error at different bit widths."""
    torch.manual_seed(42)

    # Generate weights from a trained-like distribution
    weights = torch.randn(1000) * 0.5  # Typical weight distribution

    print("  Quantization error analysis:\n")
    print(f"  {'Bits':>6} {'Levels':>8} {'Scale':>12} {'MAE':>12} {'MSE':>12} "
          f"{'SQNR (dB)':>12}")
    print("  " + "-" * 66)

    for bits in [2, 4, 8, 16, 32]:
        if bits == 32:
            # FP32 baseline (no quantization error)
            print(f"  {bits:>6} {'—':>8} {'—':>12} {'0':>12} {'0':>12} {'inf':>12}")
            continue

        n_levels = 2 ** bits
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1

        # Symmetric quantization
        abs_max = weights.abs().max().item()
        scale = abs_max / qmax

        # Quantize
        quantized = torch.clamp(torch.round(weights / scale), qmin, qmax)
        dequantized = quantized * scale

        # Error metrics
        error = weights - dequantized
        mae = error.abs().mean().item()
        mse = (error ** 2).mean().item()
        signal_power = (weights ** 2).mean().item()
        sqnr = 10 * np.log10(signal_power / mse) if mse > 0 else float('inf')

        print(f"  {bits:>6} {n_levels:>8} {scale:>12.8f} {mae:>12.8f} "
              f"{mse:>12.8f} {sqnr:>12.1f}")

    print("\n  INT8 (8-bit) is the sweet spot:")
    print("  - 256 quantization levels")
    print("  - ~40 dB SQNR (negligible error for most models)")
    print("  - 4x size reduction vs FP32")
    print("  - Widely supported by hardware accelerators")


if __name__ == "__main__":
    print("=== Exercise 1: Scale and Zero-Point Computation ===")
    exercise_1()
    print("\n=== Exercise 2: PTQ vs QAT Comparison ===")
    exercise_2()
    print("\n=== Exercise 3: Mixed-Precision Quantization ===")
    exercise_3()
    print("\n=== Exercise 4: Quantization Error Analysis ===")
    exercise_4()
    print("\nAll exercises completed!")
