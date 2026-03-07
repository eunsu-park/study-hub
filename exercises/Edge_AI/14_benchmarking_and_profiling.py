"""
Exercises for Lesson 14: Benchmarking and Profiling
Topic: Edge_AI

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn as nn
import time
import numpy as np


# === Exercise 1: Proper Latency Measurement ===
# Problem: Implement correct latency measurement with warmup,
# statistical analysis, and proper timing methodology.

def exercise_1():
    """Implement proper latency measurement methodology."""
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10),
    )
    model.eval()

    input_tensor = torch.randn(1, 3, 224, 224)

    # Wrong way: no warmup, single measurement
    with torch.no_grad():
        start = time.perf_counter()
        model(input_tensor)
        wrong_time = (time.perf_counter() - start) * 1000
    print(f"  WRONG: Single measurement (no warmup): {wrong_time:.2f} ms")
    print("  (Includes JIT compilation, cache misses, etc.)\n")

    # Right way: warmup + multiple runs + statistics
    n_warmup = 50
    n_runs = 200

    # Warmup (populates caches, triggers JIT)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(input_tensor)

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model(input_tensor)
            latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(sorted(latencies))

    print(f"  RIGHT: {n_runs} measurements after {n_warmup} warmup runs:")
    print(f"    Mean:   {latencies.mean():.3f} ms")
    print(f"    Median: {np.median(latencies):.3f} ms")
    print(f"    Std:    {latencies.std():.3f} ms")
    print(f"    Min:    {latencies.min():.3f} ms")
    print(f"    Max:    {latencies.max():.3f} ms")
    print(f"    P90:    {np.percentile(latencies, 90):.3f} ms")
    print(f"    P95:    {np.percentile(latencies, 95):.3f} ms")
    print(f"    P99:    {np.percentile(latencies, 99):.3f} ms")

    # Coefficient of variation
    cv = latencies.std() / latencies.mean() * 100
    print(f"    CV:     {cv:.1f}% ({'stable' if cv < 10 else 'high variance'})")

    print("\n  Best practices:")
    print("  - Always warmup (50+ iterations)")
    print("  - Measure 200+ runs for statistical significance")
    print("  - Report median (robust to outliers) and P95/P99 (tail latency)")
    print("  - Use time.perf_counter() (not time.time()) for precision")
    print("  - Disable power saving / turbo boost for consistent results")


# === Exercise 2: Roofline Model Analysis ===
# Problem: Apply the roofline model to determine whether a model
# is compute-bound or memory-bound on a given device.

def exercise_2():
    """Apply roofline model to analyze model bottlenecks."""
    # Device specs
    devices = [
        {
            "name": "NVIDIA Jetson Orin Nano",
            "peak_gflops": 40 * 2,  # 40 TOPS INT8, ~80 GFLOPS FP32 equiv
            "memory_bw_gbps": 68,    # LPDDR5
        },
        {
            "name": "Raspberry Pi 5 (Cortex-A76)",
            "peak_gflops": 15,       # FP32 theoretical
            "memory_bw_gbps": 17,    # LPDDR4X
        },
        {
            "name": "Apple A17 Pro (Neural Engine)",
            "peak_gflops": 2000,     # INT8 TOPS -> GOPS equiv
            "memory_bw_gbps": 17,    # LPDDR5
        },
    ]

    # Model workloads
    models = [
        {
            "name": "MobileNet-V2 (FP32)",
            "gflops": 0.3,
            "model_bytes_mb": 14,
            "activation_bytes_mb": 10,
        },
        {
            "name": "ResNet-50 (FP32)",
            "gflops": 4.1,
            "model_bytes_mb": 98,
            "activation_bytes_mb": 100,
        },
        {
            "name": "YOLOv5s (FP32)",
            "gflops": 7.2,
            "model_bytes_mb": 28,
            "activation_bytes_mb": 50,
        },
    ]

    print("  Roofline Model Analysis:\n")
    print("  The roofline model identifies whether a workload is:")
    print("  - Compute-bound: limited by peak FLOPS (high arithmetic intensity)")
    print("  - Memory-bound:  limited by memory bandwidth (low arithmetic intensity)")
    print(f"  Crossover point = Peak GFLOPS / Memory BW (FLOPS/byte)\n")

    for device in devices:
        ridge_point = device['peak_gflops'] / device['memory_bw_gbps']
        print(f"  [{device['name']}]")
        print(f"    Peak: {device['peak_gflops']} GFLOPS, "
              f"BW: {device['memory_bw_gbps']} GB/s, "
              f"Ridge: {ridge_point:.1f} FLOPS/byte")
        print()

        print(f"    {'Model':<25} {'GFLOPs':>8} {'Data(MB)':>10} {'AI':>8} "
              f"{'Bound':>12} {'Time(ms)':>10}")
        print("    " + "-" * 76)

        for model in models:
            total_data_mb = model['model_bytes_mb'] + model['activation_bytes_mb']
            total_data_gb = total_data_mb / 1024

            # Arithmetic intensity = FLOPs / bytes accessed
            ai = model['gflops'] / total_data_gb  # FLOPS / byte

            if ai >= ridge_point:
                bound = "Compute"
                time_ms = (model['gflops'] / device['peak_gflops']) * 1000
            else:
                bound = "Memory"
                time_ms = (total_data_gb / device['memory_bw_gbps']) * 1000

            print(f"    {model['name']:<25} {model['gflops']:>8.1f} "
                  f"{total_data_mb:>10} {ai:>7.1f} {bound:>12} {time_ms:>9.1f}")

        print()

    print("  Implications:")
    print("  - Memory-bound: quantization helps (smaller data movement)")
    print("  - Compute-bound: pruning helps (fewer operations)")
    print("  - Many edge models are memory-bound due to limited bandwidth")


# === Exercise 3: Power Consumption Estimation ===
# Problem: Estimate power consumption and battery life for continuous
# inference on different edge platforms.

def exercise_3():
    """Estimate power consumption and battery life."""
    scenarios = [
        {
            "device": "Jetson Orin Nano (15W mode)",
            "idle_power_w": 3.0,
            "inference_power_w": 12.0,
            "inference_time_ms": 10,
            "duty_cycle": 1.0,  # Continuous inference
            "battery_wh": None,  # Plugged in
        },
        {
            "device": "Raspberry Pi 5 + Coral USB",
            "idle_power_w": 3.5,
            "inference_power_w": 6.0,
            "inference_time_ms": 25,
            "duty_cycle": 1.0,
            "battery_wh": None,
        },
        {
            "device": "Smartphone (Snapdragon 8 Gen 2)",
            "idle_power_w": 0.5,
            "inference_power_w": 3.0,
            "inference_time_ms": 15,
            "duty_cycle": 0.1,  # 10% duty cycle (on demand)
            "battery_wh": 18.0,  # ~5000 mAh @ 3.6V
        },
        {
            "device": "STM32H7 (MCU)",
            "idle_power_w": 0.01,
            "inference_power_w": 0.3,
            "inference_time_ms": 50,
            "duty_cycle": 0.01,  # Infer once per second
            "battery_wh": 1.0,  # CR2032 coin cell
        },
    ]

    print("  Power Consumption and Battery Life Analysis:\n")

    for s in scenarios:
        # Average power = idle * (1-duty) + inference * duty
        avg_power = (s['idle_power_w'] * (1 - s['duty_cycle']) +
                     s['inference_power_w'] * s['duty_cycle'])

        # Energy per inference (Joules)
        energy_per_inference = s['inference_power_w'] * s['inference_time_ms'] / 1000

        # Inferences per second at duty cycle
        if s['duty_cycle'] < 1.0:
            # Duty cycle limited
            fps = s['duty_cycle'] * (1000 / s['inference_time_ms'])
        else:
            fps = 1000 / s['inference_time_ms']

        print(f"  [{s['device']}]")
        print(f"    Idle power:     {s['idle_power_w']:.2f} W")
        print(f"    Inference power: {s['inference_power_w']:.1f} W")
        print(f"    Inference time:  {s['inference_time_ms']} ms")
        print(f"    Duty cycle:      {s['duty_cycle']:.0%}")
        print(f"    Average power:   {avg_power:.2f} W")
        print(f"    Energy/inference: {energy_per_inference * 1000:.1f} mJ")
        print(f"    Throughput:      {fps:.1f} inferences/sec")

        if s['battery_wh']:
            battery_life_hours = s['battery_wh'] / avg_power
            if battery_life_hours > 24:
                print(f"    Battery life:    {battery_life_hours / 24:.1f} days "
                      f"({s['battery_wh']} Wh)")
            else:
                print(f"    Battery life:    {battery_life_hours:.1f} hours "
                      f"({s['battery_wh']} Wh)")
        else:
            print(f"    Battery life:    N/A (plugged in)")

        # Daily energy cost (electricity)
        daily_kwh = avg_power * 24 / 1000
        daily_cost = daily_kwh * 0.12  # $0.12/kWh
        print(f"    Daily energy:    {daily_kwh:.3f} kWh (${daily_cost:.3f}/day)")
        print()


# === Exercise 4: Layer-Level Profiling ===
# Problem: Profile individual layers of a model to identify
# bottleneck layers for optimization.

def exercise_4():
    """Profile model at layer level to find bottlenecks."""
    torch.manual_seed(42)

    class ProfilableModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(256)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(256, 1000)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            x = x.flatten(1)
            x = self.fc(x)
            return x

    model = ProfilableModel()
    model.eval()

    # Profile each layer
    layer_times = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            # Note: this measures wall time including scheduling
            pass
        return hook_fn

    # Manual layer-by-layer profiling
    input_tensor = torch.randn(1, 3, 224, 224)
    n_runs = 100

    # Warmup
    with torch.no_grad():
        for _ in range(50):
            model(input_tensor)

    # Profile layer by layer
    layers = [
        ("conv1+bn1+relu", lambda x: model.relu(model.bn1(model.conv1(x)))),
        ("maxpool", lambda x: model.pool(x)),
        ("conv2+bn2+relu", lambda x: model.relu(model.bn2(model.conv2(x)))),
        ("conv3+bn3+relu", lambda x: model.relu(model.bn3(model.conv3(x)))),
        ("avgpool", lambda x: model.avgpool(x)),
        ("flatten+fc", lambda x: model.fc(x.flatten(1))),
    ]

    # Simulate layer profiling by measuring sequential stages
    x = input_tensor
    shapes = [("input", list(x.shape))]

    print("  Layer-Level Profiling:\n")
    print(f"  {'Layer':<20} {'Input Shape':<22} {'Output Shape':<22} "
          f"{'Time (ms)':>10} {'%':>6}")
    print("  " + "-" * 84)

    total_time = 0
    results = []

    with torch.no_grad():
        for name, layer_fn in layers:
            input_shape = list(x.shape)

            # Warmup this layer
            for _ in range(20):
                layer_fn(x)

            # Time it
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                out = layer_fn(x)
                times.append((time.perf_counter() - start) * 1000)

            avg_time = np.median(times)
            total_time += avg_time
            output_shape = list(out.shape)
            results.append((name, input_shape, output_shape, avg_time))
            x = out  # Feed to next layer

    for name, in_shape, out_shape, t in results:
        pct = (t / total_time) * 100
        print(f"  {name:<20} {str(in_shape):<22} {str(out_shape):<22} "
              f"{t:>10.3f} {pct:>5.1f}%")

    print(f"  {'TOTAL':<20} {'':<22} {'':<22} {total_time:>10.3f}")

    # Identify bottleneck
    bottleneck = max(results, key=lambda r: r[3])
    print(f"\n  Bottleneck layer: {bottleneck[0]} "
          f"({bottleneck[3] / total_time * 100:.0f}% of total time)")
    print(f"  Optimization target: focus compression on this layer first")

    print("\n  Profiling tips:")
    print("  - Profile on the TARGET device (not dev machine)")
    print("  - Conv layers typically dominate (60-80% of time)")
    print("  - Fuse BN into Conv for inference (eliminates BN time)")
    print("  - Large feature maps (early layers) are often memory-bound")


if __name__ == "__main__":
    print("=== Exercise 1: Proper Latency Measurement ===")
    exercise_1()
    print("\n=== Exercise 2: Roofline Model Analysis ===")
    exercise_2()
    print("\n=== Exercise 3: Power Consumption Estimation ===")
    exercise_3()
    print("\n=== Exercise 4: Layer-Level Profiling ===")
    exercise_4()
    print("\nAll exercises completed!")
