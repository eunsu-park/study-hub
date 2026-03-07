# 레슨 10: PyTorch Mobile과 ExecuTorch

[이전: TensorFlow Lite](./09_TensorFlow_Lite.md) | [다음: 엣지 하드웨어](./11_Edge_Hardware.md)

## 학습 목표

이 레슨을 완료하면 다음을 수행할 수 있습니다:

1. TorchScript 트레이싱과 스크립팅을 사용하여 PyTorch 모델을 내보낸다
2. `torch.jit.trace`와 `torch.jit.script`를 언제 사용해야 하는지 구별한다
3. PyTorch Mobile (레거시) 워크플로우를 사용하여 모바일 디바이스에 모델을 배포한다
4. 내보내기에서 양자화, 배포까지 ExecuTorch 파이프라인을 구축한다
5. CPU 최적화 추론을 위해 XNNPACK delegate를 구성한다
6. 엣지 특화 요구사항에 맞는 커스텀 연산자를 구현하고 등록한다

---

TensorFlow Lite가 역사적으로 모바일/임베디드 분야를 지배해 왔지만, PyTorch 생태계도 엣지 배포 분야에서 빠르게 성숙해졌습니다. PyTorch Mobile의 후속인 ExecuTorch는 `torch.export`와 ATen 연산자 세트를 기반으로 구축된 깔끔한 내보내기-배포 파이프라인을 제공합니다. 학습 워크플로우가 PyTorch에 있다면, 배포에서도 PyTorch 생태계 내에 머무르는 것이 오류가 발생하기 쉬운 모델 변환 단계를 완전히 피할 수 있습니다. 이 레슨에서는 레거시 TorchScript 경로와 최신 ExecuTorch 접근법을 모두 다루어, 기존 코드베이스와 신규 프로젝트 모두에서 작업할 수 있도록 합니다.

---

## 1. TorchScript

### 1.1 TorchScript가 필요한 이유

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

### 1.4 Trace vs Script 결정 가이드

| 기준 | `torch.jit.trace` | `torch.jit.script` |
|------|-------------------|---------------------|
| **제어 흐름** | 실행되지 않은 분기 무시 | 모든 분기 보존 |
| **동적 형상** | 예제 입력 형상에 고정 | 동적 형상 지원 |
| **사용 편의성** | 매우 쉬움 (입력만 제공) | 코드 변경이 필요할 수 있음 |
| **Python 기능** | 트레이싱 중 모든 Python 코드 실행 | TorchScript 부분 집합으로 제한 |
| **최적 사용 대상** | 단순 순방향 모델 | if/else, 루프가 있는 모델 |
| **디버깅** | 원본과 출력 비교 | TorchScript 오류 메시지 확인 |

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

## 2. PyTorch Mobile (레거시)

### 2.1 모바일 최적화

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

### 2.2 Android 통합 (Java/Kotlin)

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

## 3. ExecuTorch 파이프라인

### 3.1 ExecuTorch 아키텍처

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

### 3.2 torch.export를 사용한 내보내기

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

### 3.3 PT2E를 사용한 양자화

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

### 3.4 ExecuTorch로 변환 및 저장

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

### 4.1 XNNPACK란?

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

### 4.2 ExecuTorch에서 XNNPACK 사용하기

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

### 4.3 XNNPACK + 양자화 결합

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

## 5. 커스텀 연산자

### 5.1 커스텀 연산자가 필요한 이유

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

### 5.2 PyTorch에서 커스텀 연산자 정의하기

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

### 5.3 ExecuTorch용 커스텀 연산자 C++ 커널

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

## 6. 배포 패턴

### 6.1 ExecuTorch C++ 런타임

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

### 6.2 프로토타이핑을 위한 Python 런타임

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

## 연습 문제

### 연습 1: TorchScript 내보내기
1. `forward()`에서 입력 텐서 값에 따른 if/else 분기가 있는 모델을 생성하십시오
2. `torch.jit.trace`와 `torch.jit.script` 두 가지 방법으로 내보내십시오
3. 두 분기를 모두 트리거하는 입력을 넣어 출력을 비교하십시오 -- 트레이싱이 하나의 경로를 놓치는 것을 관찰하십시오

### 연습 2: ExecuTorch 파이프라인
1. MNIST에서 작은 CNN을 훈련시키십시오
2. 전체 ExecuTorch 파이프라인을 수행하십시오: 내보내기 -> 양자화 (XNNPACK 양자화기를 사용한 PT2E) -> 위임 (XNNPACK) -> .pte 저장
3. 각 단계에서 모델 크기를 측정하십시오

### 연습 3: 커스텀 연산자
1. 커스텀 `hard_swish` 활성화 함수를 정의하십시오: `x * relu6(x + 3) / 6`
2. `torch.library`를 사용하여 등록하십시오
3. 모델에서 사용하고, `torch.export`로 내보낸 후 내보낸 그래프에 나타나는지 확인하십시오

---

[이전: TensorFlow Lite](./09_TensorFlow_Lite.md) | [다음: 엣지 하드웨어](./11_Edge_Hardware.md)
