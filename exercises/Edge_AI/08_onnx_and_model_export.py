"""
Exercises for Lesson 08: ONNX and Model Export
Topic: Edge_AI

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import tempfile


# === Exercise 1: ONNX Export with Custom Operations ===
# Problem: Export a model with custom operations to ONNX and handle
# unsupported operations by rewriting them with supported ones.

def exercise_1():
    """Handle custom operations during ONNX export."""

    class ModelWithCustomOps(nn.Module):
        """Model with operations that may not directly export to ONNX."""

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.bn = nn.BatchNorm2d(16)
            self.fc = nn.Linear(16, 10)

        def forward(self, x):
            x = torch.relu(self.bn(self.conv(x)))
            # Global average pooling (ONNX-friendly version)
            x = x.mean(dim=[2, 3])  # Instead of custom pooling
            x = self.fc(x)
            return x

    model = ModelWithCustomOps()
    model.eval()

    dummy = torch.randn(1, 3, 32, 32)
    onnx_path = os.path.join(tempfile.gettempdir(), "custom_ops.onnx")

    # Export
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["image"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )

    print(f"  Model exported to ONNX: {onnx_path}")
    print(f"  File size: {os.path.getsize(onnx_path) / 1024:.1f} KB")

    # Verify with ONNX
    try:
        import onnx
        model_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(model_onnx)
        print(f"  Validation: PASSED")
        print(f"  Nodes: {len(model_onnx.graph.node)}")

        # List operations used
        op_types = set(n.op_type for n in model_onnx.graph.node)
        print(f"  Operations used: {sorted(op_types)}")
    except ImportError:
        print("  (onnx package not installed, skipping validation)")

    print("\n  Tips for ONNX-compatible code:")
    print("  - Use x.mean(dim=[2,3]) instead of custom pool functions")
    print("  - Avoid Python control flow dependent on tensor values")
    print("  - Use opset_version >= 11 for most modern operators")
    print("  - Set model.eval() to fold BatchNorm into Conv")

    os.remove(onnx_path)


# === Exercise 2: ONNX Graph Inspection ===
# Problem: Export a model and inspect the ONNX graph structure,
# including input/output shapes, node count, and operator types.

def exercise_2():
    """Inspect ONNX graph structure and metadata."""

    class InspectionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 10),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    model = InspectionModel()
    model.eval()

    onnx_path = os.path.join(tempfile.gettempdir(), "inspect.onnx")
    torch.onnx.export(
        model, torch.randn(1, 1, 28, 28), onnx_path,
        opset_version=13,
        input_names=["input"], output_names=["output"],
    )

    try:
        import onnx
        from collections import Counter

        m = onnx.load(onnx_path)
        graph = m.graph

        print("  ONNX Graph Inspection:\n")

        # Inputs
        print("  Inputs:")
        for inp in graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            dtype = inp.type.tensor_type.elem_type
            print(f"    {inp.name}: shape={shape}, dtype={dtype}")

        # Outputs
        print("\n  Outputs:")
        for out in graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"    {out.name}: shape={shape}")

        # Operation counts
        op_counts = Counter(n.op_type for n in graph.node)
        print(f"\n  Total nodes: {len(graph.node)}")
        print("  Operation counts:")
        for op, count in op_counts.most_common():
            print(f"    {op:<20} {count}")

        # Initializers (weights)
        print(f"\n  Initializers (weights/biases): {len(graph.initializer)}")
        total_params = 0
        for init in graph.initializer:
            size = np.prod(init.dims) if init.dims else 0
            total_params += size
        print(f"  Total parameters: {total_params:,}")

    except ImportError:
        print("  (onnx package not installed)")

        # Alternative: manual parameter analysis
        total = sum(p.numel() for p in model.parameters())
        print(f"\n  Model parameter analysis (PyTorch):")
        print(f"    Total parameters: {total:,}")
        for name, param in model.named_parameters():
            print(f"    {name:<35} {list(param.shape)}")

    if os.path.exists(onnx_path):
        os.remove(onnx_path)


# === Exercise 3: ONNX Operator Compatibility Check ===
# Problem: Check if a model's operations are supported by different
# ONNX runtime backends (CPU, GPU, TensorRT, NNAPI).

def exercise_3():
    """Check operator compatibility across ONNX backends."""
    # Common ONNX operators and their backend support
    operator_support = {
        "Conv": {"CPU": True, "CUDA": True, "TensorRT": True, "NNAPI": True,
                 "CoreML": True, "EdgeTPU": True},
        "Relu": {"CPU": True, "CUDA": True, "TensorRT": True, "NNAPI": True,
                 "CoreML": True, "EdgeTPU": True},
        "MatMul": {"CPU": True, "CUDA": True, "TensorRT": True, "NNAPI": True,
                   "CoreML": True, "EdgeTPU": True},
        "BatchNormalization": {"CPU": True, "CUDA": True, "TensorRT": True,
                                "NNAPI": True, "CoreML": True, "EdgeTPU": False},
        "Softmax": {"CPU": True, "CUDA": True, "TensorRT": True, "NNAPI": True,
                    "CoreML": True, "EdgeTPU": True},
        "Resize": {"CPU": True, "CUDA": True, "TensorRT": True, "NNAPI": False,
                   "CoreML": True, "EdgeTPU": False},
        "NonMaxSuppression": {"CPU": True, "CUDA": True, "TensorRT": False,
                               "NNAPI": False, "CoreML": False, "EdgeTPU": False},
        "LayerNormalization": {"CPU": True, "CUDA": True, "TensorRT": True,
                                "NNAPI": False, "CoreML": True, "EdgeTPU": False},
        "GatherElements": {"CPU": True, "CUDA": True, "TensorRT": True,
                            "NNAPI": False, "CoreML": False, "EdgeTPU": False},
        "ScatterND": {"CPU": True, "CUDA": True, "TensorRT": False,
                       "NNAPI": False, "CoreML": False, "EdgeTPU": False},
    }

    backends = ["CPU", "CUDA", "TensorRT", "NNAPI", "CoreML", "EdgeTPU"]

    print("  ONNX Operator Compatibility Matrix:\n")
    header = f"  {'Operator':<22}"
    for b in backends:
        header += f" {b:>8}"
    print(header)
    print("  " + "-" * (22 + 9 * len(backends)))

    for op, support in operator_support.items():
        row = f"  {op:<22}"
        for b in backends:
            status = "Y" if support.get(b, False) else "N"
            row += f" {status:>8}"
        print(row)

    # Example: check a model's ops against a target backend
    model_ops = ["Conv", "Relu", "BatchNormalization", "MatMul", "Softmax"]
    target = "NNAPI"

    print(f"\n  Compatibility check for target backend: {target}")
    all_supported = True
    for op in model_ops:
        supported = operator_support.get(op, {}).get(target, False)
        status = "supported" if supported else "NOT SUPPORTED"
        if not supported:
            all_supported = False
        print(f"    {op:<22} -> {status}")

    print(f"\n  Verdict: {'All ops supported' if all_supported else 'Some ops need fallback to CPU'}")
    print("  Unsupported ops will fall back to CPU, causing performance penalties")
    print("  due to data transfer between accelerator and CPU.")


# === Exercise 4: Model Export Best Practices ===
# Problem: Demonstrate common export pitfalls and their solutions.

def exercise_4():
    """Common ONNX export pitfalls and solutions."""

    print("  Common ONNX Export Pitfalls:\n")

    # Pitfall 1: Dynamic shapes not declared
    print("  1. Dynamic shapes not declared:")
    print("     Problem: Model exported with fixed batch size")
    print("     Solution: Use dynamic_axes parameter")
    print("     ```python")
    print("     torch.onnx.export(model, dummy, path,")
    print("         dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})")
    print("     ```\n")

    # Pitfall 2: Python control flow
    print("  2. Data-dependent control flow:")
    print("     Problem: if/else based on tensor values")
    print("     Solution: Use torch.where() or rewrite as arithmetic")

    class BadModel(nn.Module):
        def forward(self, x):
            # BAD: Python if based on tensor value
            # if x.sum() > 0: return x * 2
            # GOOD: torch.where
            return torch.where(x > 0, x * 2, x * 0.5)

    model_good = BadModel()
    model_good.eval()
    dummy = torch.randn(1, 10)
    path = os.path.join(tempfile.gettempdir(), "good_model.onnx")
    torch.onnx.export(model_good, dummy, path, opset_version=13)
    print(f"     torch.where model exported: {os.path.getsize(path)} bytes")
    os.remove(path)

    # Pitfall 3: Not setting eval mode
    print("\n  3. Not setting eval() mode:")
    print("     Problem: BatchNorm uses running stats in eval, batch stats in train")
    print("     Solution: Always call model.eval() before export")
    print("     Effect: BN is folded into Conv weights (faster inference)")

    # Pitfall 4: Opset version too low
    print("\n  4. Opset version too low:")
    print("     Problem: Newer PyTorch ops need higher opset versions")
    opset_requirements = {
        "Basic CNN (Conv, ReLU, Pool)": 7,
        "Resize / Upsample": 11,
        "Dynamic shapes": 11,
        "ScatterND": 11,
        "Transformer (MultiHeadAttn)": 13,
        "Grid Sampler": 16,
    }
    for op, ver in opset_requirements.items():
        print(f"     {op:<35} >= opset {ver}")

    print("\n     Recommendation: Use opset 13+ for most modern models")

    # Pitfall 5: In-place operations
    print("\n  5. In-place operations:")
    print("     Problem: x.relu_() may cause tracing issues")
    print("     Solution: Use x = torch.relu(x) instead")
    print("     In-place ops modify the tensor directly and can")
    print("     confuse the ONNX tracer's computation graph.")


if __name__ == "__main__":
    print("=== Exercise 1: ONNX Export with Custom Operations ===")
    exercise_1()
    print("\n=== Exercise 2: ONNX Graph Inspection ===")
    exercise_2()
    print("\n=== Exercise 3: Operator Compatibility Check ===")
    exercise_3()
    print("\n=== Exercise 4: Model Export Best Practices ===")
    exercise_4()
    print("\nAll exercises completed!")
