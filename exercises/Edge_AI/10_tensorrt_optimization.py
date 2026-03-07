"""
Exercises for Lesson 10: TensorRT Optimization
Topic: Edge_AI

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn as nn
import numpy as np


# === Exercise 1: TensorRT Optimization Concepts ===
# Problem: Explain the key TensorRT optimizations and how they
# reduce inference latency on NVIDIA GPUs.

def exercise_1():
    """Explain TensorRT optimization techniques."""
    optimizations = [
        {
            "name": "Layer Fusion",
            "description": (
                "Combines multiple sequential operations into a single GPU kernel. "
                "Reduces kernel launch overhead and memory bandwidth usage."
            ),
            "examples": [
                "Conv + BatchNorm + ReLU -> single fused kernel",
                "Conv + Add (residual) + ReLU -> fused kernel",
                "FC + Bias + Activation -> single kernel",
            ],
            "benefit": "Fewer kernel launches, less memory traffic",
        },
        {
            "name": "Precision Calibration (INT8/FP16)",
            "description": (
                "Reduces numerical precision from FP32 to FP16 or INT8. "
                "Tensor Cores accelerate lower-precision math."
            ),
            "examples": [
                "FP32 -> FP16: 2x throughput on Tensor Cores",
                "FP32 -> INT8:  4x throughput, needs calibration data",
                "Mixed precision: sensitive layers in FP16, rest in INT8",
            ],
            "benefit": "2-4x throughput, 2-4x less memory",
        },
        {
            "name": "Kernel Auto-Tuning",
            "description": (
                "Profiles multiple kernel implementations for each layer "
                "and selects the fastest one for the specific GPU and input size."
            ),
            "examples": [
                "Choose between Winograd, FFT, or direct convolution",
                "Select optimal tile sizes for matrix multiplication",
                "Pick best memory layout (NCHW vs NHWC)",
            ],
            "benefit": "Optimal kernel selection per hardware",
        },
        {
            "name": "Dynamic Tensor Memory",
            "description": (
                "Reuses GPU memory across layers that don't execute simultaneously. "
                "Reduces peak memory allocation."
            ),
            "examples": [
                "Layer A output buffer reused for Layer C input",
                "Temporary buffers freed immediately after use",
                "Memory pool prevents allocation/deallocation overhead",
            ],
            "benefit": "Reduced GPU memory footprint",
        },
        {
            "name": "Multi-Stream Execution",
            "description": (
                "Overlaps independent operations using CUDA streams. "
                "Enables concurrent execution of non-dependent layers."
            ),
            "examples": [
                "Parallel branches in Inception/ResNet processed concurrently",
                "Compute and data transfer overlap",
            ],
            "benefit": "Better GPU utilization",
        },
    ]

    for opt in optimizations:
        print(f"  [{opt['name']}]")
        print(f"    {opt['description']}")
        print(f"    Examples:")
        for ex in opt['examples']:
            print(f"      - {ex}")
        print(f"    Benefit: {opt['benefit']}")
        print()


# === Exercise 2: Layer Fusion Simulation ===
# Problem: Simulate the effect of layer fusion by comparing
# separate vs fused operations in PyTorch.

def exercise_2():
    """Simulate layer fusion effects."""
    import time

    torch.manual_seed(42)
    device = "cpu"

    # Separate operations (no fusion)
    class SeparateOps(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(64, 128, 3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(128)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv(x)      # Intermediate tensor saved
            x = self.bn(x)        # Another intermediate
            x = self.relu(x)      # Another intermediate
            return x

    # Fused operations (simulated — PyTorch can fuse Conv+BN in eval mode)
    class FusedOps(nn.Module):
        def __init__(self, separate_model):
            super().__init__()
            # Fuse Conv + BN manually
            conv = separate_model.conv
            bn = separate_model.bn

            # Fused weight = BN_weight / sqrt(BN_var + eps) * Conv_weight
            bn_scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            fused_weight = conv.weight * bn_scale.view(-1, 1, 1, 1)
            fused_bias = bn.bias - bn.running_mean * bn_scale

            self.conv = nn.Conv2d(64, 128, 3, padding=1)
            self.conv.weight = nn.Parameter(fused_weight)
            self.conv.bias = nn.Parameter(fused_bias)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))  # Only 2 ops instead of 3

    # Compare
    separate = SeparateOps()
    separate.eval()

    # Run forward once to initialize BN running stats
    dummy = torch.randn(4, 64, 32, 32)
    separate.train()
    for _ in range(10):
        separate(dummy)
    separate.eval()

    fused = FusedOps(separate)
    fused.eval()

    # Verify outputs match
    x = torch.randn(1, 64, 32, 32)
    with torch.no_grad():
        out_sep = separate(x)
        out_fused = fused(x)

    max_diff = (out_sep - out_fused).abs().max().item()
    print(f"  Output difference (separate vs fused): {max_diff:.8f}")
    print(f"  Outputs match: {max_diff < 1e-5}")

    # Memory analysis
    sep_params = sum(p.numel() for p in separate.parameters())
    fused_params = sum(p.numel() for p in fused.parameters())
    print(f"\n  Separate: {sep_params:,} params (Conv + BN weight/bias/running stats)")
    print(f"  Fused:    {fused_params:,} params (Conv with bias only)")

    print(f"\n  Fusion benefits:")
    print(f"    - Removes BN entirely (folded into Conv weights)")
    print(f"    - 1 kernel launch instead of 3")
    print(f"    - No intermediate tensor storage (Conv->BN->ReLU -> Conv+ReLU)")
    print(f"    - In TensorRT, this happens automatically")


# === Exercise 3: INT8 Calibration Strategy ===
# Problem: Design an INT8 calibration dataset and analyze the impact
# of different calibration strategies on accuracy.

def exercise_3():
    """Design INT8 calibration strategy for TensorRT."""
    calibration_strategies = [
        {
            "name": "Min-Max Calibration",
            "method": "Track min/max values of activations",
            "formula": "scale = max(|min|, |max|) / 127",
            "pros": "Simple, fast",
            "cons": "Sensitive to outliers — a single extreme value can waste range",
            "accuracy": "Lowest (outlier-sensitive)",
        },
        {
            "name": "Entropy Calibration (TensorRT default)",
            "method": "Minimize KL divergence between FP32 and INT8 distributions",
            "formula": "Find threshold T that minimizes KL(FP32_hist || INT8_hist)",
            "pros": "Robust to outliers, clips extreme values",
            "cons": "Slower than min-max, needs histogram computation",
            "accuracy": "Best for most models",
        },
        {
            "name": "Percentile Calibration",
            "method": "Use 99.99th percentile instead of absolute max",
            "formula": "scale = percentile(|activations|, 99.99) / 127",
            "pros": "Simple outlier handling",
            "cons": "Percentile choice is a hyperparameter",
            "accuracy": "Good (middle ground)",
        },
    ]

    print("  INT8 Calibration Strategies:\n")
    for strat in calibration_strategies:
        print(f"  [{strat['name']}]")
        print(f"    Method: {strat['method']}")
        print(f"    Formula: {strat['formula']}")
        print(f"    Pros: {strat['pros']}")
        print(f"    Cons: {strat['cons']}")
        print(f"    Accuracy: {strat['accuracy']}")
        print()

    # Demonstrate the effect of outliers
    print("  Outlier sensitivity demonstration:\n")
    np.random.seed(42)
    activations = np.random.randn(10000).astype(np.float32)
    # Add a few outliers
    activations_with_outlier = activations.copy()
    activations_with_outlier[0] = 50.0  # Extreme outlier

    for name, data in [("Normal", activations),
                       ("With outlier", activations_with_outlier)]:
        abs_max = np.abs(data).max()
        p9999 = np.percentile(np.abs(data), 99.99)

        scale_minmax = abs_max / 127
        scale_percentile = p9999 / 127

        # Quantization error with each scale
        def quant_error(data, scale):
            q = np.clip(np.round(data / scale), -128, 127)
            deq = q * scale
            return np.mean((data - deq) ** 2)

        err_minmax = quant_error(data, scale_minmax)
        err_percentile = quant_error(data, scale_percentile)

        print(f"  {name}:")
        print(f"    abs_max={abs_max:.2f}, p99.99={p9999:.4f}")
        print(f"    MinMax MSE: {err_minmax:.6f}, Percentile MSE: {err_percentile:.6f}")

    print(f"\n  Calibration dataset guidelines:")
    print(f"    - Use 100-1000 representative samples")
    print(f"    - Samples should cover the input distribution")
    print(f"    - More diverse > more quantity")
    print(f"    - Avoid all-zeros or adversarial inputs")
    print(f"    - Entropy calibration is the recommended default")


# === Exercise 4: TensorRT Engine Building Workflow ===
# Problem: Outline the complete TensorRT engine building workflow
# from ONNX model to optimized inference.

def exercise_4():
    """TensorRT engine building workflow."""
    workflow = [
        {
            "step": "1. Export Model to ONNX",
            "command": "torch.onnx.export(model, dummy, 'model.onnx', opset_version=13)",
            "notes": [
                "Set model.eval() before export",
                "Use opset >= 11 for TensorRT compatibility",
                "Enable do_constant_folding=True",
            ],
        },
        {
            "step": "2. Create TensorRT Builder",
            "command": (
                "logger = trt.Logger(trt.Logger.WARNING)\n"
                "    builder = trt.Builder(logger)\n"
                "    network = builder.create_network(\n"
                "        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))"
            ),
            "notes": [
                "EXPLICIT_BATCH is required for ONNX models",
                "Logger level controls build verbosity",
            ],
        },
        {
            "step": "3. Parse ONNX Model",
            "command": (
                "parser = trt.OnnxParser(network, logger)\n"
                "    parser.parse_from_file('model.onnx')"
            ),
            "notes": [
                "Check parser errors if parsing fails",
                "Unsupported ops need custom plugins",
            ],
        },
        {
            "step": "4. Configure Builder",
            "command": (
                "config = builder.create_builder_config()\n"
                "    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)\n"
                "    config.set_flag(trt.BuilderFlag.FP16)  # or INT8"
            ),
            "notes": [
                "Workspace size limits temporary GPU memory",
                "FP16: set_flag(BuilderFlag.FP16)",
                "INT8: set_flag(BuilderFlag.INT8) + calibrator",
            ],
        },
        {
            "step": "5. Build Engine",
            "command": "engine = builder.build_serialized_network(network, config)",
            "notes": [
                "This step takes minutes (kernel auto-tuning)",
                "Engine is GPU-specific (not portable)",
                "Serialize to disk for reuse: save engine bytes to file",
            ],
        },
        {
            "step": "6. Run Inference",
            "command": (
                "runtime = trt.Runtime(logger)\n"
                "    engine = runtime.deserialize_cuda_engine(engine_bytes)\n"
                "    context = engine.create_execution_context()\n"
                "    context.execute_v2(bindings=[input_ptr, output_ptr])"
            ),
            "notes": [
                "Allocate GPU memory for inputs/outputs",
                "Use CUDA streams for async execution",
                "Engine deserialization is fast (< 1 second)",
            ],
        },
    ]

    print("  TensorRT Engine Building Workflow:\n")
    for w in workflow:
        print(f"  {w['step']}")
        print(f"    Code: {w['command']}")
        print(f"    Notes:")
        for note in w['notes']:
            print(f"      - {note}")
        print()

    # Performance expectations
    print("  Typical TensorRT speedups (vs PyTorch FP32):\n")
    benchmarks = [
        ("ResNet-50", "1.8ms", "0.7ms", "0.4ms"),
        ("MobileNet-V2", "0.9ms", "0.5ms", "0.3ms"),
        ("BERT-base", "5.2ms", "2.1ms", "1.4ms"),
        ("YOLOv5s", "3.1ms", "1.2ms", "0.8ms"),
    ]
    print(f"  {'Model':<16} {'PyTorch FP32':>14} {'TRT FP16':>10} {'TRT INT8':>10}")
    print("  " + "-" * 52)
    for model, pt, fp16, int8 in benchmarks:
        print(f"  {model:<16} {pt:>14} {fp16:>10} {int8:>10}")
    print("\n  (Benchmarks on NVIDIA Jetson AGX Orin, batch=1)")


if __name__ == "__main__":
    print("=== Exercise 1: TensorRT Optimization Concepts ===")
    exercise_1()
    print("\n=== Exercise 2: Layer Fusion Simulation ===")
    exercise_2()
    print("\n=== Exercise 3: INT8 Calibration Strategy ===")
    exercise_3()
    print("\n=== Exercise 4: TensorRT Engine Building Workflow ===")
    exercise_4()
    print("\nAll exercises completed!")
