"""
07. Model Profiling

Demonstrates profiling techniques for evaluating model efficiency
on edge devices: parameter count, FLOPs, memory, and inference time.

Covers:
- Parameter counting by layer
- FLOPs estimation (manual and with thop)
- Memory footprint analysis
- Inference time benchmarking
- Model comparison table

Requirements:
    pip install torch torchvision
    pip install thop  # optional, for automated FLOPs counting
"""

import torch
import torch.nn as nn
import time
import sys

print("=" * 60)
print("Edge AI — Model Profiling")
print("=" * 60)


# ============================================
# 1. Models to Profile
# ============================================
print("\n[1] Define Models of Different Sizes")
print("-" * 40)


class TinyModel(nn.Module):
    """Tiny model for MCU deployment."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(16, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class SmallModel(nn.Module):
    """Small model for mobile deployment."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class MediumModel(nn.Module):
    """Medium model for GPU edge (e.g., Jetson)."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


models = {
    "TinyModel": TinyModel(),
    "SmallModel": SmallModel(),
    "MediumModel": MediumModel(),
}

for name, m in models.items():
    m.eval()


# ============================================
# 2. Parameter Counting
# ============================================
print("\n[2] Parameter Count by Layer")
print("-" * 40)


def count_parameters(model, verbose=False):
    """Count parameters, optionally showing per-layer breakdown."""
    total = 0
    trainable = 0
    if verbose:
        print(f"  {'Layer':<40} {'Shape':<20} {'Params':>10}")
        print("  " + "-" * 72)

    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
        if verbose:
            print(f"  {name:<40} {str(list(param.shape)):<20} {n:>10,}")

    if verbose:
        print("  " + "-" * 72)
        print(f"  {'Total':<40} {'':<20} {total:>10,}")
        print(f"  {'Trainable':<40} {'':<20} {trainable:>10,}")
    return total, trainable


# Show detailed breakdown for SmallModel
print("SmallModel parameter breakdown:")
count_parameters(models["SmallModel"], verbose=True)


# ============================================
# 3. FLOPs Estimation (Manual)
# ============================================
print("\n[3] FLOPs Estimation")
print("-" * 40)


def estimate_conv_flops(module, input_shape):
    """Estimate FLOPs for a Conv2d layer (multiply-add operations)."""
    _, c_in, h_in, w_in = input_shape
    c_out = module.out_channels
    k = module.kernel_size[0]
    s = module.stride[0]
    h_out = (h_in + 2 * module.padding[0] - k) // s + 1
    w_out = (w_in + 2 * module.padding[1] - k) // s + 1
    groups = module.groups

    # Each output element: (c_in/groups) * k * k multiply-adds
    flops = c_out * h_out * w_out * (c_in // groups) * k * k
    return flops, (1, c_out, h_out, w_out)


def estimate_linear_flops(module, input_features):
    """Estimate FLOPs for a Linear layer."""
    return module.in_features * module.out_features


def estimate_model_flops(model, input_shape=(1, 3, 32, 32)):
    """Estimate total FLOPs by walking through model layers."""
    total_flops = 0
    current_shape = input_shape

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            flops, current_shape = estimate_conv_flops(module, current_shape)
            total_flops += flops
        elif isinstance(module, nn.Linear):
            flops = estimate_linear_flops(module, None)
            total_flops += flops
        elif isinstance(module, nn.MaxPool2d):
            k = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
            s = module.stride if isinstance(module.stride, int) else module.stride[0]
            _, c, h, w = current_shape
            h_out = h // s
            w_out = w // s
            current_shape = (1, c, h_out, w_out)
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            _, c, _, _ = current_shape
            out_size = module.output_size
            if isinstance(out_size, int):
                out_size = (out_size, out_size)
            current_shape = (1, c, out_size[0], out_size[1])

    return total_flops


print(f"{'Model':<15} {'FLOPs (Manual)':>18}")
print("-" * 35)
for name, m in models.items():
    flops = estimate_model_flops(m, input_shape=(1, 3, 32, 32))
    if flops > 1e6:
        print(f"{name:<15} {flops/1e6:>15.2f} M")
    else:
        print(f"{name:<15} {flops:>15,}")


# Automated FLOPs with thop (if available)
print()
try:
    from thop import profile

    print("Automated FLOPs counting with thop:")
    dummy = torch.randn(1, 3, 32, 32)
    for name, m in models.items():
        flops, params = profile(m, inputs=(dummy,), verbose=False)
        print(f"  {name:<15} FLOPs={flops/1e6:.2f}M, Params={params/1e3:.1f}K")
except ImportError:
    print("thop not installed (pip install thop) — using manual estimation only")


# ============================================
# 4. Memory Footprint
# ============================================
print("\n[4] Memory Footprint Analysis")
print("-" * 40)


def model_memory_mb(model, input_shape=(1, 3, 32, 32), dtype_bytes=4):
    """Estimate model memory: parameters + buffers + activation maps."""
    # Parameter memory
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_mem = sum(b.numel() * b.element_size() for b in model.buffers())

    # Activation memory (estimate by running forward pass)
    activation_sizes = []
    hooks = []

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            activation_sizes.append(output.numel() * output.element_size())

    for module in model.modules():
        if not isinstance(module, nn.Sequential) and module is not model:
            hooks.append(module.register_forward_hook(hook_fn))

    dummy = torch.randn(*input_shape)
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    activation_mem = sum(activation_sizes)

    return {
        "params_mb": param_mem / (1024 ** 2),
        "buffers_mb": buffer_mem / (1024 ** 2),
        "activations_mb": activation_mem / (1024 ** 2),
        "total_mb": (param_mem + buffer_mem + activation_mem) / (1024 ** 2),
    }


print(f"{'Model':<15} {'Params':>10} {'Buffers':>10} {'Activations':>12} {'Total':>10}")
print("-" * 60)
for name, m in models.items():
    mem = model_memory_mb(m)
    print(f"{name:<15} {mem['params_mb']:>9.3f}M {mem['buffers_mb']:>9.3f}M "
          f"{mem['activations_mb']:>11.3f}M {mem['total_mb']:>9.3f}M")


# ============================================
# 5. Inference Time Benchmarking
# ============================================
print("\n[5] Inference Time Benchmarking")
print("-" * 40)


def benchmark_inference(model, input_shape=(1, 3, 32, 32),
                        n_warmup=50, n_runs=200):
    """Benchmark inference time with proper warmup."""
    dummy = torch.randn(*input_shape)
    model.eval()

    # Warmup (important for CPU cache and JIT)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model(dummy)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

    times = sorted(times)
    return {
        "mean_ms": sum(times) / len(times),
        "median_ms": times[len(times) // 2],
        "p95_ms": times[int(0.95 * len(times))],
        "p99_ms": times[int(0.99 * len(times))],
        "min_ms": times[0],
        "max_ms": times[-1],
    }


print(f"{'Model':<15} {'Mean':>8} {'Median':>8} {'P95':>8} {'P99':>8} {'Min':>8} {'Max':>8}")
print("-" * 65)
for name, m in models.items():
    stats = benchmark_inference(m, n_warmup=30, n_runs=100)
    print(f"{name:<15} {stats['mean_ms']:>7.2f}ms {stats['median_ms']:>7.2f}ms "
          f"{stats['p95_ms']:>7.2f}ms {stats['p99_ms']:>7.2f}ms "
          f"{stats['min_ms']:>7.2f}ms {stats['max_ms']:>7.2f}ms")


# ============================================
# 6. Throughput Measurement
# ============================================
print("\n[6] Throughput (samples/second)")
print("-" * 40)

for batch_size in [1, 8, 32]:
    print(f"\nBatch size = {batch_size}:")
    for name, m in models.items():
        dummy = torch.randn(batch_size, 3, 32, 32)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                m(dummy)

        # Measure
        n_runs = 50
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                m(dummy)
        elapsed = time.perf_counter() - start

        throughput = (n_runs * batch_size) / elapsed
        print(f"  {name:<15} {throughput:>10.1f} samples/sec")


# ============================================
# 7. Summary Comparison
# ============================================
print("\n[7] Model Profiling Summary")
print("-" * 40)

print(f"\n{'Model':<15} {'Params':>10} {'FLOPs':>10} {'Memory':>10} {'Latency':>10}")
print("-" * 58)
for name, m in models.items():
    params = sum(p.numel() for p in m.parameters())
    flops = estimate_model_flops(m)
    mem = model_memory_mb(m)
    stats = benchmark_inference(m, n_warmup=20, n_runs=50)

    params_str = f"{params/1e3:.1f}K" if params < 1e6 else f"{params/1e6:.1f}M"
    flops_str = f"{flops/1e6:.1f}M" if flops < 1e9 else f"{flops/1e9:.1f}G"

    print(f"{name:<15} {params_str:>10} {flops_str:>10} "
          f"{mem['total_mb']:>9.2f}M {stats['median_ms']:>9.2f}ms")

print()
print("Key takeaways:")
print("- Always profile BEFORE optimizing (don't guess the bottleneck)")
print("- Latency percentiles (P95, P99) matter more than mean for real-time")
print("- Memory includes params + buffers + activations (all add up)")
print("- Batch size affects throughput but not single-sample latency")
print("- Profile on the TARGET device, not your development machine")
