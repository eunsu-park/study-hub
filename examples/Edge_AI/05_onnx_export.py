"""
05. ONNX Export and Inference

Demonstrates exporting a PyTorch model to ONNX format,
optimizing the graph, and running inference with ONNX Runtime.

Covers:
- PyTorch model export to ONNX
- ONNX model inspection and validation
- ONNX graph optimization
- ONNX Runtime inference
- Output verification (PyTorch vs ONNX Runtime)
- Dynamic axes for variable batch sizes

Requirements:
    pip install torch torchvision onnx onnxruntime
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import tempfile

print("=" * 60)
print("Edge AI — ONNX Export and Inference")
print("=" * 60)


# ============================================
# 1. Define a Model
# ============================================
print("\n[1] Define Model for Export")
print("-" * 40)


class ClassifierCNN(nn.Module):
    """Simple CNN classifier for demonstration."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = ClassifierCNN()
model.eval()  # Important: set to eval mode before export
print(f"Model: ClassifierCNN")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================
# 2. Export to ONNX
# ============================================
print("\n[2] Export PyTorch Model to ONNX")
print("-" * 40)

# Create dummy input matching expected input shape
dummy_input = torch.randn(1, 1, 28, 28)

# Export path
onnx_path = os.path.join(tempfile.gettempdir(), "classifier.onnx")

# Export with dynamic batch size
torch.onnx.export(
    model,                      # Model
    dummy_input,                # Example input
    onnx_path,                  # Output file
    export_params=True,         # Include trained weights
    opset_version=13,           # ONNX opset version
    do_constant_folding=True,   # Optimize constant folding
    input_names=["input"],      # Input tensor names
    output_names=["output"],    # Output tensor names
    dynamic_axes={              # Variable-length axes
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)

file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
print(f"ONNX model saved to: {onnx_path}")
print(f"ONNX file size: {file_size_mb:.2f} MB")


# ============================================
# 3. Validate ONNX Model
# ============================================
print("\n[3] Validate ONNX Model")
print("-" * 40)

try:
    import onnx

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation: PASSED")

    # Inspect graph
    print(f"\nModel graph info:")
    print(f"  IR version: {onnx_model.ir_version}")
    print(f"  Opset version: {onnx_model.opset_import[0].version}")
    print(f"  Graph inputs: {[i.name for i in onnx_model.graph.input]}")
    print(f"  Graph outputs: {[o.name for o in onnx_model.graph.output]}")
    print(f"  Number of nodes: {len(onnx_model.graph.node)}")

    # Show first few operations
    print(f"\n  First 5 operations:")
    for i, node in enumerate(onnx_model.graph.node[:5]):
        print(f"    {i}: {node.op_type} ({node.name})")
except ImportError:
    print("onnx package not installed — skipping validation")


# ============================================
# 4. ONNX Runtime Inference
# ============================================
print("\n[4] ONNX Runtime Inference")
print("-" * 40)

try:
    import onnxruntime as ort

    # Create inference session
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )

    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"Using providers: {session.get_providers()}")

    # Get input/output details
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f"\nInput:  name={input_info.name}, shape={input_info.shape}, "
          f"type={input_info.type}")
    print(f"Output: name={output_info.name}, shape={output_info.shape}, "
          f"type={output_info.type}")

    # Run inference
    test_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
    ort_outputs = session.run(
        None,  # Run all outputs
        {"input": test_input}
    )

    print(f"\nInference output shape: {ort_outputs[0].shape}")
    print(f"Output (logits): {ort_outputs[0][0][:5]}...")  # First 5 logits

except ImportError:
    print("onnxruntime not installed — skipping inference")
    ort = None


# ============================================
# 5. Verify Outputs Match
# ============================================
print("\n[5] Verify PyTorch vs ONNX Runtime Outputs")
print("-" * 40)

if ort is not None:
    # Use the same input for both
    test_tensor = torch.from_numpy(test_input)

    with torch.no_grad():
        pytorch_output = model(test_tensor).numpy()

    ort_output = session.run(None, {"input": test_input})[0]

    # Compare
    max_diff = np.abs(pytorch_output - ort_output).max()
    mean_diff = np.abs(pytorch_output - ort_output).mean()
    all_close = np.allclose(pytorch_output, ort_output, atol=1e-5)

    print(f"Max absolute difference:  {max_diff:.8f}")
    print(f"Mean absolute difference: {mean_diff:.8f}")
    print(f"All close (atol=1e-5):    {all_close}")

    # Compare predictions
    pt_pred = pytorch_output.argmax(axis=1)
    ort_pred = ort_output.argmax(axis=1)
    print(f"\nPyTorch prediction:      {pt_pred}")
    print(f"ONNX Runtime prediction: {ort_pred}")
    print(f"Predictions match:       {np.array_equal(pt_pred, ort_pred)}")


# ============================================
# 6. Dynamic Batch Size
# ============================================
print("\n[6] Dynamic Batch Size Inference")
print("-" * 40)

if ort is not None:
    for batch_size in [1, 4, 16, 64]:
        batch_input = np.random.randn(batch_size, 1, 28, 28).astype(np.float32)
        outputs = session.run(None, {"input": batch_input})
        print(f"Batch size {batch_size:>2}: input={batch_input.shape} -> "
              f"output={outputs[0].shape}")


# ============================================
# 7. Performance Comparison
# ============================================
print("\n[7] Performance: PyTorch vs ONNX Runtime")
print("-" * 40)

if ort is not None:
    n_runs = 200
    test_np = np.random.randn(1, 1, 28, 28).astype(np.float32)
    test_pt = torch.from_numpy(test_np)

    # Warmup
    for _ in range(20):
        with torch.no_grad():
            model(test_pt)
        session.run(None, {"input": test_np})

    # PyTorch timing
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model(test_pt)
    pt_time = (time.perf_counter() - start) / n_runs * 1000

    # ONNX Runtime timing
    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {"input": test_np})
    ort_time = (time.perf_counter() - start) / n_runs * 1000

    print(f"PyTorch inference:      {pt_time:.3f} ms/sample")
    print(f"ONNX Runtime inference: {ort_time:.3f} ms/sample")
    print(f"Speedup: {pt_time / ort_time:.2f}x")

# Cleanup
if os.path.exists(onnx_path):
    os.remove(onnx_path)

print()
print("Key takeaways:")
print("- ONNX provides cross-framework model portability")
print("- dynamic_axes enables variable batch sizes at inference time")
print("- ONNX Runtime applies graph optimizations (constant folding, fusion)")
print("- Output differences should be < 1e-5 (floating point precision)")
print("- ONNX Runtime is often faster than vanilla PyTorch on CPU")
