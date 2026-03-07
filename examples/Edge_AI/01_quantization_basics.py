"""
01. Quantization Basics

Demonstrates PyTorch dynamic and static quantization techniques for
reducing model size and inference latency on edge devices.

Covers:
- Dynamic quantization (weights quantized, activations quantized at runtime)
- Static quantization (weights + activations quantized using calibration data)
- Per-tensor vs per-channel quantization
- Model size and speed comparison

Requirements:
    pip install torch torchvision
"""

import torch
import torch.nn as nn
import torch.quantization as quant
import time
import os
import tempfile

print("=" * 60)
print("Edge AI — Quantization Basics")
print("=" * 60)


# ============================================
# 1. Simple Model for Quantization
# ============================================
print("\n[1] Define a Simple CNN Model")
print("-" * 40)


class SimpleCNN(nn.Module):
    """A small CNN suitable for quantization demonstrations."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


model = SimpleCNN()
model.eval()
print(f"Model: SimpleCNN")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================
# 2. Model Size Helper
# ============================================

def get_model_size_mb(model, path=None):
    """Save model to disk and return file size in MB."""
    if path is None:
        path = os.path.join(tempfile.gettempdir(), "temp_model.pt")
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    os.remove(path)
    return size_mb


def measure_inference_time(model, input_tensor, n_runs=100):
    """Measure average inference time over n_runs."""
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(input_tensor)

    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model(input_tensor)
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000  # ms per inference


fp32_size = get_model_size_mb(model)
print(f"\nFP32 model size: {fp32_size:.2f} MB")


# ============================================
# 3. Dynamic Quantization
# ============================================
print("\n[2] Dynamic Quantization")
print("-" * 40)
print("Weights are quantized ahead of time; activations are quantized")
print("dynamically at runtime. Best for RNNs and linear-heavy models.")

# Dynamic quantization: only affects nn.Linear layers by default
model_dynamic = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},       # Layers to quantize
    dtype=torch.qint8  # Target dtype
)

dynamic_size = get_model_size_mb(model_dynamic)
print(f"\nDynamic quantized model size: {dynamic_size:.2f} MB")
print(f"Size reduction: {fp32_size / dynamic_size:.1f}x")

# Verify outputs are close
dummy_input = torch.randn(1, 1, 28, 28)
with torch.no_grad():
    fp32_out = model(dummy_input)
    dynamic_out = model_dynamic(dummy_input)

max_diff = (fp32_out - dynamic_out).abs().max().item()
print(f"Max output difference (FP32 vs dynamic INT8): {max_diff:.6f}")


# ============================================
# 4. Static Quantization
# ============================================
print("\n[3] Static Quantization")
print("-" * 40)
print("Both weights AND activations are quantized using calibration data.")
print("Requires inserting QuantStub/DeQuantStub and running calibration.")


class QuantizableCNN(nn.Module):
    """CNN with QuantStub/DeQuantStub for static quantization."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.quant = quant.QuantStub()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.dequant = quant.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


# Step 1: Create model and set quantization config
model_static = QuantizableCNN()
model_static.eval()

# Use fbgemm backend (x86) or qnnpack (ARM)
backend = "fbgemm" if not torch.backends.quantized.engine == "qnnpack" else "qnnpack"
model_static.qconfig = quant.get_default_qconfig(backend)
print(f"Quantization backend: {backend}")

# Step 2: Fuse modules (Conv+ReLU, Linear+ReLU) for better performance
model_static_fused = quant.fuse_modules(
    model_static,
    [
        ["features.0", "features.1"],  # Conv2d + ReLU
        ["features.3", "features.4"],  # Conv2d + ReLU
        ["classifier.0", "classifier.1"],  # Linear + ReLU
    ]
)

# Step 3: Prepare for calibration (inserts observers)
model_prepared = quant.prepare(model_static_fused)

# Step 4: Calibration with representative data
print("Running calibration with 100 random samples...")
calibration_data = torch.randn(100, 1, 28, 28)
with torch.no_grad():
    for i in range(0, 100, 10):
        batch = calibration_data[i:i + 10]
        model_prepared(batch)

# Step 5: Convert to quantized model
model_quantized = quant.convert(model_prepared)

static_size = get_model_size_mb(model_quantized)
print(f"\nStatic quantized model size: {static_size:.2f} MB")
print(f"Size reduction vs FP32: {fp32_size / static_size:.1f}x")


# ============================================
# 5. Per-Tensor vs Per-Channel Quantization
# ============================================
print("\n[4] Per-Tensor vs Per-Channel Quantization")
print("-" * 40)

# Per-tensor: one scale/zero_point for the entire tensor
# Per-channel: one scale/zero_point per output channel (better accuracy)

print("Per-tensor:  single (scale, zero_point) for all weights in a layer")
print("Per-channel: separate (scale, zero_point) per output channel")
print()

# Demonstrate the difference
weight = torch.randn(64, 32, 3, 3)  # Conv2d weight: [out_ch, in_ch, kH, kW]

# Per-tensor quantization
scale_pt, zp_pt = torch.quantization.observer.MinMaxObserver()(weight)
print(f"Per-tensor: 1 scale ({scale_pt:.6f}), 1 zero_point ({zp_pt})")

# Per-channel quantization (channel dim = 0 for conv weights)
observer_pc = torch.quantization.observer.PerChannelMinMaxObserver(
    ch_axis=0, dtype=torch.qint8
)
observer_pc(weight)
scale_pc, zp_pc = observer_pc.calculate_qparams()
print(f"Per-channel: {len(scale_pc)} scales, {len(zp_pc)} zero_points")
print(f"  Scale range: [{scale_pc.min():.6f}, {scale_pc.max():.6f}]")
print()
print("Per-channel is generally more accurate because each channel")
print("can have a different dynamic range.")


# ============================================
# 6. Comparison Summary
# ============================================
print("\n[5] Quantization Comparison Summary")
print("-" * 40)

print(f"{'Method':<25} {'Size (MB)':<12} {'Reduction':<12}")
print("-" * 49)
print(f"{'FP32 (baseline)':<25} {fp32_size:<12.2f} {'1.0x':<12}")
print(f"{'Dynamic (Linear only)':<25} {dynamic_size:<12.2f} {fp32_size/dynamic_size:<12.1f}x")
print(f"{'Static (full INT8)':<25} {static_size:<12.2f} {fp32_size/static_size:<12.1f}x")

print()
print("Key takeaways:")
print("- Dynamic quantization: easy to apply, good for Linear/LSTM layers")
print("- Static quantization: better compression, requires calibration data")
print("- Per-channel quantization: better accuracy than per-tensor")
print("- INT8 typically provides 2-4x size reduction with <1% accuracy loss")
