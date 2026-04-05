"""
11. Edge Hardware Simulation

Simulates hardware-aware model evaluation across different edge
devices (MCU, mobile CPU, mobile GPU, NPU) with realistic constraints.

Covers:
- Hardware specification modeling (memory, compute, power)
- FLOPs and memory footprint estimation
- Hardware-aware feasibility checks
- Multi-device deployment planning
- Power and thermal budget analysis

Requirements:
    pip install torch numpy
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

print("=" * 60)
print("Edge AI — Edge Hardware Simulation")
print("=" * 60)


# ============================================
# 1. Hardware Specification Models
# ============================================
print("\n[1] Edge Hardware Specifications")
print("-" * 40)


@dataclass
class HardwareSpec:
    """Specification of an edge hardware device."""
    name: str
    compute_tops: float       # Tera operations per second
    memory_mb: float          # Available RAM in MB
    storage_mb: float         # Available flash/storage in MB
    power_budget_w: float     # Power budget in watts
    supported_dtypes: list    # Supported data types
    has_npu: bool             # Neural processing unit available
    has_gpu: bool             # GPU available
    clock_mhz: float          # CPU clock speed

    def __repr__(self):
        return (f"{self.name} ({self.compute_tops:.2f} TOPS, "
                f"{self.memory_mb:.0f} MB RAM, {self.power_budget_w:.1f} W)")


DEVICES = {
    "cortex_m7": HardwareSpec(
        name="ARM Cortex-M7 (MCU)",
        compute_tops=0.001,
        memory_mb=1.0,
        storage_mb=2.0,
        power_budget_w=0.5,
        supported_dtypes=["int8", "int16"],
        has_npu=False,
        has_gpu=False,
        clock_mhz=480,
    ),
    "rpi4": HardwareSpec(
        name="Raspberry Pi 4",
        compute_tops=0.013,
        memory_mb=2048,
        storage_mb=8192,
        power_budget_w=6.0,
        supported_dtypes=["float32", "float16", "int8"],
        has_npu=False,
        has_gpu=True,
        clock_mhz=1500,
    ),
    "jetson_nano": HardwareSpec(
        name="NVIDIA Jetson Nano",
        compute_tops=0.472,
        memory_mb=4096,
        storage_mb=16384,
        power_budget_w=10.0,
        supported_dtypes=["float32", "float16", "int8"],
        has_npu=False,
        has_gpu=True,
        clock_mhz=1430,
    ),
    "pixel_7": HardwareSpec(
        name="Google Pixel 7 (Tensor G2)",
        compute_tops=4.0,
        memory_mb=3072,
        storage_mb=32768,
        power_budget_w=5.0,
        supported_dtypes=["float32", "float16", "int8", "int4"],
        has_npu=True,
        has_gpu=True,
        clock_mhz=2850,
    ),
    "coral_tpu": HardwareSpec(
        name="Google Coral Edge TPU",
        compute_tops=4.0,
        memory_mb=64,
        storage_mb=256,
        power_budget_w=2.0,
        supported_dtypes=["int8"],
        has_npu=True,
        has_gpu=False,
        clock_mhz=500,
    ),
}

for key, dev in DEVICES.items():
    print(f"  {dev}")


# ============================================
# 2. Model Profiling Utilities
# ============================================
print("\n[2] Model Profiling (FLOPs and Memory)")
print("-" * 40)


def count_flops_conv2d(module, input_shape):
    """Estimate FLOPs for a Conv2d layer."""
    _, c_in, h_in, w_in = input_shape
    c_out = module.out_channels
    k = module.kernel_size[0]
    stride = module.stride[0]
    h_out = (h_in + 2 * module.padding[0] - k) // stride + 1
    w_out = (w_in + 2 * module.padding[1] - k) // stride + 1
    groups = module.groups
    flops = 2 * (c_in // groups) * k * k * c_out * h_out * w_out
    return flops, (1, c_out, h_out, w_out)


def count_flops_linear(module, input_shape):
    """Estimate FLOPs for a Linear layer."""
    flops = 2 * module.in_features * module.out_features
    return flops, (input_shape[0], module.out_features)


def estimate_model_profile(model, input_shape=(1, 3, 224, 224)):
    """Estimate total FLOPs and parameter memory for a model."""
    total_flops = 0
    total_params = 0
    current_shape = input_shape

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            flops, current_shape = count_flops_conv2d(module, current_shape)
            total_flops += flops
            total_params += sum(p.numel() for p in module.parameters())
        elif isinstance(module, nn.Linear):
            flat_shape = (current_shape[0], np.prod(current_shape[1:]))
            flops, current_shape = count_flops_linear(module, flat_shape)
            total_flops += flops
            total_params += sum(p.numel() for p in module.parameters())

    return {
        "flops": total_flops,
        "mflops": total_flops / 1e6,
        "params": total_params,
        "size_fp32_mb": total_params * 4 / (1024 * 1024),
        "size_int8_mb": total_params * 1 / (1024 * 1024),
        "size_fp16_mb": total_params * 2 / (1024 * 1024),
    }


# Define test models of varying complexity
class TinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.flatten(1))


class MediumConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.flatten(1))


class LargeConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(512, 100)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.flatten(1))


models = {
    "TinyConvNet": TinyConvNet(),
    "MediumConvNet": MediumConvNet(),
    "LargeConvNet": LargeConvNet(),
}

profiles = {}
for name, m in models.items():
    m.eval()
    profiles[name] = estimate_model_profile(m, (1, 3, 64, 64))

print(f"{'Model':<18} {'MFLOPs':<10} {'Params':<12} {'FP32 (MB)':<12} {'INT8 (MB)'}")
print("-" * 62)
for name, p in profiles.items():
    print(f"{name:<18} {p['mflops']:<10.1f} {p['params']:<12,} "
          f"{p['size_fp32_mb']:<12.3f} {p['size_int8_mb']:.3f}")


# ============================================
# 3. Hardware Feasibility Check
# ============================================
print("\n[3] Hardware Feasibility Matrix")
print("-" * 40)


def check_feasibility(profile: dict, hw: HardwareSpec, dtype="int8"):
    """Check if a model can run on given hardware."""
    size_key = f"size_{dtype}_mb"
    model_size = profile.get(size_key, profile["size_fp32_mb"])

    checks = {
        "dtype_supported": dtype in hw.supported_dtypes,
        "fits_memory": model_size * 2 < hw.memory_mb,  # 2x for activations
        "fits_storage": model_size < hw.storage_mb,
    }

    # Estimate inference time (very rough)
    ops_per_second = hw.compute_tops * 1e12
    if dtype == "int8":
        ops_per_second *= 2  # INT8 typically 2x throughput
    est_latency_ms = (profile["flops"] / ops_per_second) * 1000
    checks["latency_ms"] = est_latency_ms
    checks["feasible"] = all([
        checks["dtype_supported"],
        checks["fits_memory"],
        checks["fits_storage"],
    ])
    return checks


# Print feasibility matrix
print(f"\n{'Model':<18}", end="")
for dev_key in DEVICES:
    print(f" {dev_key:<14}", end="")
print()
print("-" * (18 + 15 * len(DEVICES)))

for model_name, profile in profiles.items():
    print(f"{model_name:<18}", end="")
    for dev_key, dev in DEVICES.items():
        result = check_feasibility(profile, dev)
        symbol = "OK" if result["feasible"] else "NO"
        latency = result["latency_ms"]
        print(f" {symbol} ({latency:>5.1f}ms) ", end="")
    print()


# ============================================
# 4. Power and Thermal Analysis
# ============================================
print("\n[4] Power and Thermal Budget Analysis")
print("-" * 40)


def estimate_power_consumption(profile: dict, hw: HardwareSpec,
                               fps_target: float = 30.0):
    """Estimate power draw for running inference at target FPS."""
    ops_per_second = hw.compute_tops * 1e12
    time_per_frame_s = profile["flops"] / ops_per_second
    utilization = time_per_frame_s * fps_target

    # Power scales roughly with utilization (simplified model)
    idle_power_w = hw.power_budget_w * 0.3
    active_power_w = hw.power_budget_w * min(utilization, 1.0)
    total_power_w = idle_power_w + active_power_w * 0.7

    return {
        "utilization_pct": min(utilization * 100, 100),
        "power_w": total_power_w,
        "within_budget": total_power_w <= hw.power_budget_w,
        "battery_hours_3000mah": (3.0 * 3.7) / total_power_w if total_power_w > 0 else float("inf"),
    }


print(f"\nPower analysis at 30 FPS target (MediumConvNet, INT8):")
print(f"{'Device':<28} {'Util %':<10} {'Power (W)':<12} {'Battery (h)':<12} {'OK?'}")
print("-" * 72)

for dev_key, dev in DEVICES.items():
    power = estimate_power_consumption(profiles["MediumConvNet"], dev, fps_target=30)
    ok = "YES" if power["within_budget"] else "NO"
    batt = f"{power['battery_hours_3000mah']:.1f}" if power["battery_hours_3000mah"] < 100 else ">100"
    print(f"{dev.name:<28} {power['utilization_pct']:<10.1f} "
          f"{power['power_w']:<12.2f} {batt:<12} {ok}")


# ============================================
# 5. Multi-Device Deployment Plan
# ============================================
print("\n[5] Deployment Recommendation")
print("-" * 40)


def recommend_deployment(profile: dict, devices: Dict[str, HardwareSpec]):
    """Recommend the best device for a given model profile."""
    candidates = []
    for key, dev in devices.items():
        result = check_feasibility(profile, dev)
        if result["feasible"]:
            power = estimate_power_consumption(profile, dev)
            candidates.append({
                "device": key,
                "name": dev.name,
                "latency_ms": result["latency_ms"],
                "power_w": power["power_w"],
                "score": 1.0 / (result["latency_ms"] + 1) * (1.0 / (power["power_w"] + 0.1)),
            })
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


for model_name, profile in profiles.items():
    recs = recommend_deployment(profile, DEVICES)
    if recs:
        best = recs[0]
        print(f"{model_name:<18} -> {best['name']}")
        print(f"  Latency: {best['latency_ms']:.2f} ms, "
              f"Power: {best['power_w']:.2f} W")
    else:
        print(f"{model_name:<18} -> No suitable device found")


# ============================================
# 6. Memory Layout Simulation
# ============================================
print("\n[6] Memory Layout Simulation")
print("-" * 40)
print("Simulate how model weights and activations fit in device memory.\n")


def simulate_memory_layout(profile: dict, hw: HardwareSpec, dtype="int8"):
    """Simulate memory allocation on device."""
    size_key = f"size_{dtype}_mb"
    weight_mb = profile.get(size_key, profile["size_fp32_mb"])

    # Estimate peak activation memory (rough: proportional to FLOPs)
    activation_mb = weight_mb * 0.5  # Simplified estimate

    # Runtime overhead (interpreter, stack, buffers)
    overhead_mb = 0.1 if hw.memory_mb < 10 else 1.0

    total_mb = weight_mb + activation_mb + overhead_mb
    free_mb = hw.memory_mb - total_mb

    return {
        "weights_mb": weight_mb,
        "activations_mb": activation_mb,
        "overhead_mb": overhead_mb,
        "total_mb": total_mb,
        "free_mb": free_mb,
        "fits": free_mb > 0,
    }


# Show memory layout for MediumConvNet on each device
print(f"MediumConvNet memory layout (INT8):")
print(f"{'Device':<28} {'Weights':<10} {'Activations':<13} {'Total':<10} {'Free':<10}")
print("-" * 71)
for dev_key, dev in DEVICES.items():
    mem = simulate_memory_layout(profiles["MediumConvNet"], dev)
    free_str = f"{mem['free_mb']:.2f}" if mem["fits"] else "OOM"
    print(f"{dev.name:<28} {mem['weights_mb']:<10.3f} "
          f"{mem['activations_mb']:<13.3f} {mem['total_mb']:<10.3f} {free_str}")


# ============================================
# 7. Summary
# ============================================
print("\n[7] Summary")
print("-" * 40)
print("Key takeaways:")
print("- Edge devices vary by 1000x in compute (MCU vs mobile NPU)")
print("- INT8 quantization is essential for MCU and Coral TPU deployment")
print("- Memory is often the binding constraint on microcontrollers")
print("- Power budgets determine feasibility for battery-powered devices")
print("- Always profile FLOPs, memory, and power before selecting hardware")
