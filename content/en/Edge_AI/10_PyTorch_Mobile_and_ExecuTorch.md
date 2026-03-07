# 10. PyTorch Mobile and ExecuTorch

**Previous**: [TensorFlow Lite](./09_TensorFlow_Lite.md) | **Next**: [Edge Hardware](./11_Edge_Hardware.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Export PyTorch models using TorchScript tracing and scripting
2. Distinguish when to use `torch.jit.trace` versus `torch.jit.script`
3. Deploy models on mobile devices using PyTorch Mobile (legacy) workflow
4. Build the ExecuTorch pipeline from export through quantization to deployment
5. Configure the XNNPACK delegate for CPU-optimized inference
6. Implement and register custom operators for edge-specific needs

---

While TensorFlow Lite dominates the mobile/embedded space historically, PyTorch's ecosystem has rapidly matured for edge deployment. ExecuTorch, the successor to PyTorch Mobile, provides a clean export-to-deploy pipeline built on `torch.export` and the ATen operator set. If your training workflow lives in PyTorch, staying within the PyTorch ecosystem for deployment avoids the error-prone model translation step entirely. This lesson covers both the legacy TorchScript path and the modern ExecuTorch approach so you can work with existing codebases and greenfield projects alike.

---

## 1. TorchScript

### 1.1 Why TorchScript?

```
+-----------------------------------------------------------------+
|              Python PyTorch  vs  TorchScript                     |
+-----------------------------------------------------------------+
|                                                                   |
|   Python (Eager Mode)             TorchScript (Graph Mode)       |
|   +------------------+           +------------------+            |
|   | Python runtime   |           | Self-contained   |            |
|   | required         |           | C++ runtime      |            |
|   +------------------+           +------------------+            |
|   | Dynamic shapes,  |           | Static graph,    |            |
|   | control flow     |           | optimizable      |            |
|   +------------------+           +------------------+            |
|   | Easy debugging   |           | Cross-language   |            |
|   |                  |           | deployment       |            |
|   +------------------+           +------------------+            |
|                                                                   |
|   TorchScript bridges training (Python) and deployment (C++/    |
|   mobile) by capturing model logic in an intermediate repr.      |
|                                                                   |
+-----------------------------------------------------------------+
```

### 1.2 torch.jit.trace

```python
#!/usr/bin/env python3
"""Model export using torch.jit.trace."""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple CNN for demonstration."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def trace_export(model: nn.Module, input_shape: tuple, output_path: str):
    """Export model using tracing.

    Tracing records operations by running the model with example input.
    It captures the exact sequence of operations that were executed,
    producing a static graph.

    Limitation: tracing does NOT capture data-dependent control flow.
    If your forward() has if/else branches that depend on input values,
    only the branch taken during tracing is recorded.
    """
    model.eval()
    example_input = torch.randn(*input_shape)

    # Trace the model
    traced_model = torch.jit.trace(model, example_input)

    # Verify: compare traced output with original
    with torch.no_grad():
        original_out = model(example_input)
        traced_out = traced_model(example_input)

    max_diff = (original_out - traced_out).abs().max().item()
    print(f"Max output difference: {max_diff:.2e}")

    # Save
    traced_model.save(output_path)
    print(f"Saved traced model to {output_path}")

    return traced_model


if __name__ == "__main__":
    model = SimpleCNN(num_classes=10)
    traced = trace_export(model, (1, 3, 224, 224), "model_traced.pt")

    # Verify the saved model loads correctly
    loaded = torch.jit.load("model_traced.pt")
    test_input = torch.randn(1, 3, 224, 224)
    output = loaded(test_input)
    print(f"Output shape: {output.shape}")
```

### 1.3 torch.jit.script

```python
#!/usr/bin/env python3
"""Model export using torch.jit.script for control-flow models."""

import torch
import torch.nn as nn


class DynamicModel(nn.Module):
    """Model with data-dependent control flow."""

    def __init__(self):
        super().__init__()
        self.conv_small = nn.Conv2d(3, 16, 3, padding=1)
        self.conv_large = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_small = nn.Linear(16, 10)
        self.fc_large = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Data-dependent branch: torch.jit.trace would miss one path
        if x.shape[2] > 128:
            x = self.conv_large(x)
            x = self.pool(x).flatten(1)
            x = self.fc_large(x)
        else:
            x = self.conv_small(x)
            x = self.pool(x).flatten(1)
            x = self.fc_small(x)
        return x


def script_export(model: nn.Module, output_path: str):
    """Export using scripting.

    Scripting analyzes the Python source code and compiles it into
    TorchScript IR. It preserves control flow (if/else, for loops,
    while loops) but requires that the code is TorchScript-compatible
    (no arbitrary Python objects, limited use of Python builtins).
    """
    model.eval()

    scripted_model = torch.jit.script(model)

    # The scripted model preserves both branches
    print("TorchScript graph:")
    print(scripted_model.graph)

    scripted_model.save(output_path)
    print(f"\nSaved scripted model to {output_path}")

    return scripted_model


if __name__ == "__main__":
    model = DynamicModel()
    scripted = script_export(model, "model_scripted.pt")

    # Test both code paths
    small_input = torch.randn(1, 3, 64, 64)
    large_input = torch.randn(1, 3, 256, 256)

    print(f"Small input output: {scripted(small_input).shape}")
    print(f"Large input output: {scripted(large_input).shape}")
```

### 1.4 Trace vs Script Decision Guide

| Criterion | `torch.jit.trace` | `torch.jit.script` |
|-----------|-------------------|---------------------|
| **Control flow** | Ignores branches not taken | Preserves all branches |
| **Dynamic shapes** | Fixed to example input shape | Supports dynamic shapes |
| **Ease of use** | Very easy (just provide input) | May require code changes |
| **Python features** | Any Python code runs during trace | Limited to TorchScript subset |
| **Best for** | Simple feedforward models | Models with if/else, loops |
| **Debugging** | Compare outputs with original | Read TorchScript errors |

```python
# Hybrid approach: script the control-flow parts, trace the rest
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SimpleCNN()  # Traceable
        self.threshold = 0.5

    @torch.jit.export
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        probs = torch.softmax(logits, dim=1)
        # Control flow preserved by scripting
        if probs.max() > self.threshold:
            return probs.argmax(dim=1)
        else:
            return torch.tensor([-1])  # Uncertain

# Script the outer model (it will trace-compatible inner modules automatically)
# scripted = torch.jit.script(HybridModel())
```

---

## 2. PyTorch Mobile (Legacy)

### 2.1 Mobile Optimization

```python
#!/usr/bin/env python3
"""PyTorch Mobile optimization pipeline."""

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile


def export_for_mobile(model: torch.nn.Module,
                      input_shape: tuple,
                      output_path: str):
    """Export and optimize a model for mobile deployment.

    The mobile optimizer applies graph transformations:
    - Conv + BatchNorm fusion (eliminates BN at inference time)
    - Dropout removal (no-op at eval mode, but the op node remains)
    - Insert packed linear/conv representations
    These reduce latency and model size on mobile CPUs.
    """
    model.eval()
    example_input = torch.randn(*input_shape)

    # Trace the model
    traced = torch.jit.trace(model, example_input)

    # Apply mobile-specific optimizations
    optimized = optimize_for_mobile(traced)

    # Save with Lite interpreter format
    optimized._save_for_lite_interpreter(output_path)
    print(f"Saved mobile-optimized model: {output_path}")

    return optimized


def compare_model_sizes(original_path: str, optimized_path: str):
    """Compare file sizes of original vs mobile-optimized models."""
    import os
    orig_size = os.path.getsize(original_path) / 1024
    opt_size = os.path.getsize(optimized_path) / 1024
    reduction = (1 - opt_size / orig_size) * 100

    print(f"Original:  {orig_size:.1f} KB")
    print(f"Optimized: {opt_size:.1f} KB")
    print(f"Reduction: {reduction:.1f}%")
```

### 2.2 Android Integration (Java/Kotlin)

```java
// Android: Load and run PyTorch Mobile model
// build.gradle: implementation 'org.pytorch:pytorch_android_lite:2.1.0'

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

public class ModelInference {
    private Module model;

    public void loadModel(String modelPath) {
        model = LiteModuleLoader.load(modelPath);
    }

    public float[] predict(float[] inputData, long[] shape) {
        // Create input tensor
        Tensor inputTensor = Tensor.fromBlob(inputData, shape);

        // Run inference
        Tensor outputTensor = model.forward(IValue.from(inputTensor))
                                   .toTensor();

        // Get output
        return outputTensor.getDataAsFloatArray();
    }
}
```

---

## 3. ExecuTorch Pipeline

### 3.1 ExecuTorch Architecture

```
+-----------------------------------------------------------------+
|                    ExecuTorch Pipeline                            |
+-----------------------------------------------------------------+
|                                                                   |
|   1. Export            2. Quantize         3. Delegate            |
|   +-----------+       +-----------+       +-----------+          |
|   | torch     |       | PT2E      |       | Backend   |          |
|   | .export() |------>| Quantizer |------>| Delegate  |          |
|   +-----------+       +-----------+       +-----------+          |
|        |                   |                   |                  |
|   ATen dialect        Quantized graph    Delegate-specific       |
|   (clean IR)          (int8 ops)         partitioning            |
|        |                   |                   |                  |
|        v                   v                   v                  |
|   +-----------+       +-----------+       +-----------+          |
|   | Edge      |       | Edge      |       | .pte      |          |
|   | Dialect   |       | Program   |       | artifact  |          |
|   +-----------+       +-----------+       +-----------+          |
|                                                |                  |
|   4. Deploy                                    |                  |
|   +--------------------------------------------v---------+       |
|   |  ExecuTorch Runtime (C++)                             |       |
|   |  - Minimal footprint (~100 KB base)                   |       |
|   |  - No Python dependency                               |       |
|   |  - Platform: Android, iOS, MCU, Linux                 |       |
|   +-------------------------------------------------------+       |
|                                                                   |
+-----------------------------------------------------------------+
```

### 3.2 Export with torch.export

```python
#!/usr/bin/env python3
"""ExecuTorch: Step 1 -- Export using torch.export."""

import torch
import torch.nn as nn
from torch.export import export, ExportedProgram


class EfficientNet_Lite(nn.Module):
    """Simplified efficient model for edge deployment."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            # Depthwise separable conv
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def export_model(model: nn.Module, input_shape: tuple) -> ExportedProgram:
    """Export model using torch.export.

    torch.export produces a clean ATen-level IR (intermediate representation)
    that is fully functional (no Python dependency). Unlike TorchScript,
    it captures the entire computation graph soundly, including dynamic
    control flow via torch.cond and torch.map.
    """
    model.eval()
    example_input = (torch.randn(*input_shape),)

    exported = export(model, example_input)

    # Inspect the exported graph
    print("Exported graph:")
    print(exported.graph_module.graph)

    return exported


if __name__ == "__main__":
    model = EfficientNet_Lite(num_classes=10)
    exported = export_model(model, (1, 3, 224, 224))
    print(f"\nExport successful: {type(exported)}")
```

### 3.3 Quantize with PT2E

```python
#!/usr/bin/env python3
"""ExecuTorch: Step 2 -- Quantize using PT2E quantization."""

import torch
from torch.export import export
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)


def quantize_for_executorch(model: torch.nn.Module,
                            example_input: tuple,
                            calibration_data: list = None):
    """Quantize a model using PT2E quantization.

    PT2E quantization works on the exported graph rather than the
    eager model. It inserts observer nodes, calibrates with real data,
    then converts observers to quantize/dequantize operations.
    """
    model.eval()

    # Step 1: Export to get ATen graph
    exported = export(model, example_input)

    # Step 2: Configure quantizer
    # XNNPACKQuantizer produces models optimized for XNNPACK delegate
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config())

    # Step 3: Prepare -- inserts observers into the graph
    prepared = prepare_pt2e(exported.graph_module, quantizer)

    # Step 4: Calibrate with representative data
    if calibration_data is None:
        # Use dummy data if no calibration set provided
        calibration_data = [torch.randn_like(example_input[0]) for _ in range(100)]

    with torch.no_grad():
        for data in calibration_data:
            prepared(data)

    # Step 5: Convert observers to quantize/dequantize ops
    quantized = convert_pt2e(prepared)

    print("Quantization complete")
    return quantized


if __name__ == "__main__":
    from torch.nn import Module

    # Simple model for demonstration
    class SmallModel(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.relu = torch.nn.ReLU()
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(16, 10)

        def forward(self, x):
            x = self.relu(self.conv(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    model = SmallModel()
    example = (torch.randn(1, 3, 64, 64),)
    quantized = quantize_for_executorch(model, example)
```

### 3.4 Lower to ExecuTorch and Save

```python
#!/usr/bin/env python3
"""ExecuTorch: Step 3 -- Lower to edge dialect and save .pte."""

import torch
from torch.export import export
from executorch.exir import to_edge, EdgeProgramManager
from executorch.exir import ExecutorchBackendConfig


def lower_to_executorch(model: torch.nn.Module,
                        example_input: tuple,
                        output_path: str = "model.pte"):
    """Full pipeline: export -> edge dialect -> ExecuTorch program.

    The .pte (Portable ExecuTorch) file contains:
    - The computation graph in Edge dialect
    - Constant data (weights, biases)
    - Delegate-specific blobs (if backend delegation was applied)
    - Metadata for the ExecuTorch runtime
    """
    model.eval()

    # Export
    exported = export(model, example_input)

    # Lower to Edge dialect
    edge_program: EdgeProgramManager = to_edge(exported)

    # Generate ExecuTorch program
    et_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(
            extract_constant_segment=True,  # Separate weights for mmap
        )
    )

    # Save
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    print(f"Saved ExecuTorch program: {output_path}")
    print(f"Size: {len(et_program.buffer) / 1024:.1f} KB")

    return et_program
```

---

## 4. XNNPACK Delegate

### 4.1 What is XNNPACK?

```
+-----------------------------------------------------------------+
|                     XNNPACK Delegate                             |
+-----------------------------------------------------------------+
|                                                                   |
|   XNNPACK is a highly optimized library for floating-point       |
|   and quantized neural network inference on ARM and x86 CPUs.    |
|                                                                   |
|   Key optimizations:                                             |
|   - NEON (ARM) / SSE/AVX (x86) SIMD vectorization               |
|   - Winograd convolution for 3x3 kernels                        |
|   - Indirect convolution algorithm                               |
|   - Multi-threaded execution with pthreadpool                    |
|   - Operator fusion (Conv+ReLU, Conv+Add+ReLU)                  |
|                                                                   |
|   Supported operations (partial list):                           |
|   - Convolution (regular, depthwise, transposed)                 |
|   - Fully connected / Linear                                     |
|   - Average/Max pooling                                          |
|   - Element-wise (Add, Multiply, etc.)                           |
|   - Softmax, Sigmoid, ReLU, Clamp                               |
|                                                                   |
+-----------------------------------------------------------------+
```

### 4.2 Using XNNPACK with ExecuTorch

```python
#!/usr/bin/env python3
"""Delegate model execution to XNNPACK backend in ExecuTorch."""

import torch
from torch.export import export
from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)


def delegate_to_xnnpack(model: torch.nn.Module,
                        example_input: tuple,
                        output_path: str = "model_xnnpack.pte"):
    """Partition and delegate supported ops to XNNPACK.

    The partitioner analyzes the graph and groups contiguous
    XNNPACK-compatible operations into subgraphs. Each subgraph
    is lowered to an XNNPACK-optimized blob. Unsupported ops
    remain in the portable (CPU) backend.
    """
    model.eval()

    # Export
    exported = export(model, example_input)

    # Lower to edge
    edge = to_edge(exported)

    # Delegate to XNNPACK
    edge = edge.to_backend(XnnpackPartitioner())

    # Generate program
    et_program = edge.to_executorch()

    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    print(f"XNNPACK-delegated model saved: {output_path}")
    return et_program


if __name__ == "__main__":
    from torchvision.models import mobilenet_v2

    model = mobilenet_v2(weights=None, num_classes=10)
    example = (torch.randn(1, 3, 224, 224),)

    delegate_to_xnnpack(model, example, "mobilenet_xnnpack.pte")
```

### 4.3 XNNPACK + Quantization Combined

```python
#!/usr/bin/env python3
"""Combine PT2E quantization with XNNPACK delegation."""

import torch
from torch.export import export
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)


def quantize_and_delegate(model: torch.nn.Module,
                          example_input: tuple,
                          calibration_data: list,
                          output_path: str = "model_q8_xnnpack.pte"):
    """Full pipeline: export -> quantize -> delegate -> save.

    This is the recommended production pipeline for ExecuTorch
    on CPU-based edge devices. INT8 quantization reduces model size
    by ~4x, and XNNPACK accelerates quantized inference using
    optimized int8 kernels with NEON/SSE dot-product instructions.
    """
    model.eval()

    # Export
    exported = export(model, example_input)

    # Quantize
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config()
    )
    prepared = prepare_pt2e(exported.graph_module, quantizer)

    with torch.no_grad():
        for data in calibration_data:
            prepared(data)

    quantized = convert_pt2e(prepared)

    # Re-export the quantized module
    quantized_exported = export(quantized, example_input)

    # Lower to edge and delegate
    edge = to_edge(quantized_exported)
    edge = edge.to_backend(XnnpackPartitioner())
    et_program = edge.to_executorch()

    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    print(f"Quantized + XNNPACK model: {output_path}")
    print(f"Size: {len(et_program.buffer) / 1024:.1f} KB")
```

---

## 5. Custom Operators

### 5.1 Why Custom Ops?

```
+-----------------------------------------------------------------+
|                  Custom Operator Use Cases                       |
+-----------------------------------------------------------------+
|                                                                   |
|   When standard operators are not enough:                        |
|                                                                   |
|   1. Domain-specific preprocessing                               |
|      - Sensor-specific normalization                             |
|      - Custom image transforms                                   |
|                                                                   |
|   2. Hardware-specific acceleration                              |
|      - Vendor DSP/NPU kernels                                   |
|      - FPGA-accelerated operations                               |
|                                                                   |
|   3. Fused operations for performance                            |
|      - Multi-head attention fused kernel                         |
|      - Custom activation functions                               |
|                                                                   |
|   4. Post-processing                                             |
|      - Non-max suppression (NMS)                                 |
|      - Custom decoding logic                                     |
|                                                                   |
+-----------------------------------------------------------------+
```

### 5.2 Defining Custom Ops in PyTorch

```python
#!/usr/bin/env python3
"""Define and register custom operators for edge deployment."""

import torch
from torch import Tensor


# --- Method 1: torch.library (recommended for ExecuTorch) ---

# Define a custom op namespace
torch.library.define(
    "edge_ops::fused_relu_threshold",
    "(Tensor x, float threshold) -> Tensor"
)

@torch.library.impl("edge_ops::fused_relu_threshold", "cpu")
def fused_relu_threshold_cpu(x: Tensor, threshold: float) -> Tensor:
    """Fused ReLU + threshold: max(0, x) where x > threshold, else 0.

    Fusing two operations into one kernel avoids an intermediate
    memory allocation and an extra pass over the data.
    """
    return torch.where(x > threshold, x, torch.zeros_like(x))

# Register the abstract (meta) implementation for shape inference
@torch.library.impl_abstract("edge_ops::fused_relu_threshold")
def fused_relu_threshold_meta(x: Tensor, threshold: float) -> Tensor:
    return torch.empty_like(x)


# --- Method 2: Using the op in a model ---

class ModelWithCustomOp(torch.nn.Module):
    def __init__(self, threshold: float = 0.1):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.fc = torch.nn.Linear(16, 10)
        self.threshold = threshold

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = torch.ops.edge_ops.fused_relu_threshold(x, self.threshold)
        x = x.mean(dim=[2, 3])  # Global average pool
        return self.fc(x)


if __name__ == "__main__":
    model = ModelWithCustomOp(threshold=0.1)
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    print(f"Output shape: {out.shape}")
```

### 5.3 Custom Op C++ Kernel for ExecuTorch

```cpp
// custom_ops.cpp -- Register a custom op kernel for ExecuTorch runtime

#include <executorch/runtime/kernel/kernel_includes.h>

namespace edge_ops {

// The kernel implementation for the ExecuTorch runtime
Tensor& fused_relu_threshold_out(
    RuntimeContext& ctx,
    const Tensor& input,
    double threshold,
    Tensor& out) {

    // Validate shapes
    ET_CHECK_MSG(
        input.sizes() == out.sizes(),
        "Input and output shapes must match"
    );

    const float* in_data = input.const_data_ptr<float>();
    float* out_data = out.mutable_data_ptr<float>();
    const int64_t numel = input.numel();
    const float thresh = static_cast<float>(threshold);

    for (int64_t i = 0; i < numel; ++i) {
        out_data[i] = (in_data[i] > thresh) ? in_data[i] : 0.0f;
    }

    return out;
}

// Register with ExecuTorch
EXECUTORCH_LIBRARY(edge_ops, "fused_relu_threshold.out", fused_relu_threshold_out);

}  // namespace edge_ops
```

---

## 6. Deployment Patterns

### 6.1 ExecuTorch C++ Runtime

```cpp
// inference_app.cpp -- Run ExecuTorch model in C++

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

using namespace torch::executor;

int main(int argc, char* argv[]) {
    // Initialize runtime
    runtime_init();

    // Load model
    auto loader = util::FileDataLoader::from("model.pte");
    auto program = Program::load(&loader.get());

    // Create method (default: "forward")
    auto method = program->load_method("forward");

    // Prepare input
    float input_data[1 * 3 * 224 * 224];
    // ... fill with preprocessed data ...

    auto input_tensor = from_blob(
        input_data,
        {1, 3, 224, 224},
        ScalarType::Float
    );

    // Execute
    auto outputs = method->execute({input_tensor});

    // Read output
    auto output_tensor = outputs[0].toTensor();
    float* result = output_tensor.data_ptr<float>();

    // Process result
    int predicted_class = 0;
    float max_score = result[0];
    for (int i = 1; i < 10; i++) {
        if (result[i] > max_score) {
            max_score = result[i];
            predicted_class = i;
        }
    }

    printf("Predicted class: %d (score: %.4f)\n",
           predicted_class, max_score);

    return 0;
}
```

### 6.2 Python Runtime for Prototyping

```python
#!/usr/bin/env python3
"""Run ExecuTorch .pte model from Python (for testing)."""

import torch
import numpy as np
from executorch.runtime import Runtime, Program, Method


def run_pte_model(model_path: str, input_data: np.ndarray):
    """Load and run a .pte model using the Python ExecuTorch runtime."""
    runtime = Runtime.get()
    program = runtime.load_program(model_path)
    method = program.load_method("forward")

    input_tensor = torch.from_numpy(input_data).float()
    outputs = method.execute([input_tensor])

    return outputs[0].numpy()


if __name__ == "__main__":
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    result = run_pte_model("model.pte", dummy_input)
    print(f"Output shape: {result.shape}")
    print(f"Top class: {result.argmax()}")
```

---

## Practice Exercises

### Exercise 1: TorchScript Export
1. Create a model with an if/else branch in `forward()` based on input tensor values
2. Export it with both `torch.jit.trace` and `torch.jit.script`
3. Feed inputs that trigger both branches and compare outputs -- observe how tracing misses one path

### Exercise 2: ExecuTorch Pipeline
1. Train a small CNN on MNIST
2. Walk through the full ExecuTorch pipeline: export -> quantize (PT2E with XNNPACK quantizer) -> delegate (XNNPACK) -> save .pte
3. Measure model size at each stage

### Exercise 3: Custom Operator
1. Define a custom `hard_swish` activation: `x * relu6(x + 3) / 6`
2. Register it with `torch.library`
3. Use it in a model, export with `torch.export`, and verify it appears in the exported graph

---

**Previous**: [TensorFlow Lite](./09_TensorFlow_Lite.md) | **Next**: [Edge Hardware](./11_Edge_Hardware.md)
