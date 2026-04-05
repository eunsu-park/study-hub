# 레슨 8: ONNX와 모델 내보내기

[이전: 신경 아키텍처 탐색](./07_Neural_Architecture_Search.md) | [다음: TFLite와 모바일 런타임](./09_TensorFlow_Lite.md)

---

## 학습 목표

- ONNX 포맷을 이해한다: 계산 그래프, 연산자, opset 버전
- PyTorch와 TensorFlow 모델을 올바른 동적 축과 메타데이터로 ONNX로 내보낸다
- ONNX Runtime 최적화를 적용하여 추론 속도를 향상시킨다
- 그래프 최적화를 수행한다: 상수 폴딩, 연산자 융합, 중복 제거
- ONNX를 중간 표현으로 사용하여 프레임워크 간 모델을 변환한다
- 상호운용성 테스트를 통해 내보낸 모델의 정확성을 검증한다

---

## 1. ONNX란?

**ONNX**(Open Neural Network Exchange)는 머신러닝 모델을 표현하기 위한 개방형 포맷입니다. 공통 연산자 집합과 표준 파일 포맷을 정의하여, 서로 다른 프레임워크(PyTorch, TensorFlow, JAX) 간에 모델을 전달하고 다양한 런타임(ONNX Runtime, TensorRT, OpenVINO, CoreML)에 배포할 수 있게 합니다.

```
Training Frameworks              ONNX                 Inference Runtimes
┌─────────────┐                                      ┌─────────────────┐
│  PyTorch    │──export──┐                    ┌─────▶│ ONNX Runtime    │
└─────────────┘          │    ┌──────────┐    │      └─────────────────┘
┌─────────────┐          ├───▶│  .onnx   │────┤      ┌─────────────────┐
│ TensorFlow  │──export──┤    │  file    │    ├─────▶│ TensorRT        │
└─────────────┘          │    └──────────┘    │      └─────────────────┘
┌─────────────┐          │                    │      ┌─────────────────┐
│    JAX      │──export──┘                    ├─────▶│ OpenVINO        │
└─────────────┘                               │      └─────────────────┘
                                              │      ┌─────────────────┐
                                              └─────▶│ CoreML / NNAPI  │
                                                     └─────────────────┘
```

### 1.1 ONNX 모델 구조

ONNX 모델은 방향 비순환 그래프(DAG)입니다:

```
ONNX Model
├── ModelProto (최상위 컨테이너)
│   ├── opset_import: [{"": 17}]     ← 연산자 세트 버전
│   ├── graph: GraphProto
│   │   ├── input: [ValueInfoProto]   ← 입력 텐서 사양 (이름, 타입, 형상)
│   │   ├── output: [ValueInfoProto]  ← 출력 텐서 사양
│   │   ├── node: [NodeProto]         ← 계산 노드 (연산자)
│   │   ├── initializer: [TensorProto]← 사전 훈련된 가중치
│   │   └── value_info: [...]         ← 중간 텐서 메타데이터
│   └── metadata_props: [...]          ← 선택적 키-값 메타데이터
```

```python
import onnx


def inspect_onnx_model(model_path):
    """
    Load and inspect an ONNX model's structure.
    """
    model = onnx.load(model_path)

    # Basic info
    print(f"=== ONNX Model: {model_path} ===")
    print(f"IR version: {model.ir_version}")
    print(f"Opset version: {model.opset_import[0].version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")

    graph = model.graph

    # Inputs
    print(f"\nInputs ({len(graph.input)}):")
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param
                 for d in inp.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
        print(f"  {inp.name}: {dtype} {shape}")

    # Outputs
    print(f"\nOutputs ({len(graph.output)}):")
    for out in graph.output:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param
                 for d in out.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
        print(f"  {out.name}: {dtype} {shape}")

    # Nodes (operators)
    print(f"\nNodes ({len(graph.node)}):")
    op_counts = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op}: {count}")

    # Weights
    print(f"\nInitializers (weights): {len(graph.initializer)}")
    total_bytes = sum(
        len(init.raw_data) if init.raw_data else 0
        for init in graph.initializer
    )
    print(f"Total weight size: {total_bytes / 1e6:.1f} MB")

    # Validate
    try:
        onnx.checker.check_model(model)
        print(f"\nValidation: PASSED")
    except onnx.checker.ValidationError as e:
        print(f"\nValidation: FAILED - {e}")

    return model
```

### 1.2 ONNX 연산자와 Opset

ONNX는 표준 연산자 집합(Conv, MatMul, Relu 등)을 **opset**으로 버전 관리합니다. 각 opset 버전은 새로운 연산자를 추가하거나 기존 연산자를 수정합니다.

| Opset | 주요 추가 사항 |
|-------|------------------|
| 9 | BatchNormalization, Where |
| 11 | Resize, ScatterElements, DynamicQuantizeLinear |
| 13 | Squeeze/Unsqueeze가 축을 속성이 아닌 입력으로 받음 |
| 14 | Reshape에서 allowzero 지원 |
| 15 | Shape에 start/end 속성 |
| 17 | LayerNormalization, GroupNormalization |
| 18 | BitwiseAnd/Or/Xor, CenterCropPad |
| 20 | GridSample, AffineGrid |

```python
import onnx

# List available operators for a given opset
def list_onnx_operators(opset_version=17):
    """List operators available in a given ONNX opset version."""
    schema = onnx.defs.get_all_schemas_with_history()
    ops_in_version = set()
    for s in schema:
        if s.domain == "" and s.since_version <= opset_version:
            ops_in_version.add(s.name)
    print(f"ONNX Opset {opset_version}: {len(ops_in_version)} operators")
    # Print first 20
    for op in sorted(ops_in_version)[:20]:
        print(f"  {op}")
    print(f"  ... and {len(ops_in_version) - 20} more")


# list_onnx_operators(17)
```

---

## 2. PyTorch 모델을 ONNX로 내보내기

### 2.1 기본 내보내기

```python
import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.model(x)


# Create and export
model = SimpleClassifier()
model.eval()

# Dummy input — defines the input shape and dtype
dummy_input = torch.randn(1, 784)

# Export to ONNX
torch.onnx.export(
    model,                          # PyTorch model
    dummy_input,                    # Example input
    "classifier.onnx",             # Output file path
    export_params=True,             # Include trained weights
    opset_version=17,               # ONNX opset version
    do_constant_folding=True,       # Optimize: fold constants
    input_names=["input"],          # Name the input tensor
    output_names=["output"],        # Name the output tensor
    dynamic_axes={                  # Allow variable batch size
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)
print("Model exported to classifier.onnx")
```

### 2.2 CNN 모델 내보내기

```python
import torch
import torch.nn as nn
import torchvision.models as models


def export_vision_model(model_name="mobilenet_v2", output_path="model.onnx"):
    """
    Export a torchvision model to ONNX.
    Handles common pitfalls: eval mode, dynamic axes, opset version.
    """
    # Load pretrained model
    model = getattr(models, model_name)(weights="IMAGENET1K_V1")
    model.eval()  # Critical: must be in eval mode (affects BN, Dropout)

    # Create dummy input matching expected dimensions
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        # Metadata
        verbose=False,
    )

    # Verify the exported model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Check file size
    import os
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Exported {model_name} to {output_path} ({size_mb:.1f} MB)")

    return output_path


# Export MobileNetV2
export_vision_model("mobilenet_v2", "mobilenet_v2.onnx")
```

### 2.3 동적 형상 처리

많은 엣지 AI 애플리케이션에서는 가변 길이 입력(다양한 이미지 크기, 가변 시퀀스 길이)을 처리해야 합니다. ONNX는 이를 위해 **동적 축(dynamic axes)**을 지원합니다.

```python
import torch
import torch.nn as nn


class VariableLengthModel(nn.Module):
    """Model that handles variable-length sequences."""

    def __init__(self, vocab_size=10000, embed_dim=128, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, 64, batch_first=True)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_length)
        embedded = self.embedding(input_ids)
        output, (hidden, _) = self.lstm(embedded)
        # Use last hidden state
        logits = self.classifier(hidden.squeeze(0))
        return logits


model = VariableLengthModel()
model.eval()

# Export with dynamic batch AND sequence length
dummy = torch.randint(0, 10000, (1, 32))  # batch=1, seq_len=32

torch.onnx.export(
    model,
    dummy,
    "text_classifier.onnx",
    opset_version=17,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {
            0: "batch_size",
            1: "sequence_length",  # Variable sequence length
        },
        "logits": {0: "batch_size"},
    },
)
print("Exported model with dynamic batch and sequence length")
```

### 2.4 내보내기 시 흔한 실수

```python
# Pitfall 1: 모델이 eval 모드가 아닌 경우
# BAD: BatchNorm uses running stats, Dropout is active during training
model.train()
torch.onnx.export(model, dummy, "bad_model.onnx")  # Wrong behavior!

# GOOD: Always set eval mode before export
model.eval()
torch.onnx.export(model, dummy, "good_model.onnx")


# Pitfall 2: 지원되지 않는 연산
# Some Python control flow doesn't export well
class BadModel(nn.Module):
    def forward(self, x):
        if x.shape[0] > 1:  # Data-dependent control flow!
            return x * 2
        return x

# Workaround: use torch.where or refactor to avoid data-dependent branches
class GoodModel(nn.Module):
    def forward(self, x):
        return x * 2  # Always same computation graph


# Pitfall 3: dynamic_axes 누락
# Without dynamic_axes, the batch dimension is hardcoded
# This means you can only run inference with the exact batch size used during export


# Pitfall 4: 잘못된 opset 버전
# Too low: missing operators (e.g., opset 9 lacks many modern ops)
# Too high: runtime may not support latest opset
# Recommendation: opset 17 is widely supported as of 2024
```

---

## 3. TensorFlow 모델을 ONNX로 내보내기

```python
# Using tf2onnx to convert TensorFlow/Keras models to ONNX

# Installation: pip install tf2onnx

# Method 1: Command-line conversion
# python -m tf2onnx.convert --saved-model ./saved_model_dir --output model.onnx --opset 17

# Method 2: Python API
def export_keras_to_onnx(keras_model, output_path, opset=17):
    """
    Convert a Keras/TensorFlow model to ONNX.
    """
    import tf2onnx
    import tensorflow as tf

    # Get input signature
    input_signature = [
        tf.TensorSpec(shape=inp.shape, dtype=inp.dtype, name=inp.name)
        for inp in keras_model.inputs
    ]

    # Convert
    onnx_model, _ = tf2onnx.convert.from_keras(
        keras_model,
        input_signature=input_signature,
        opset=opset,
        output_path=output_path,
    )

    print(f"Converted Keras model to {output_path}")
    return onnx_model


# Example usage:
# import tensorflow as tf
# keras_model = tf.keras.applications.MobileNetV2(weights="imagenet")
# export_keras_to_onnx(keras_model, "mobilenet_v2_tf.onnx")
```

---

## 4. ONNX Runtime

ONNX Runtime(ORT)은 Microsoft의 고성능 ONNX 모델 추론 엔진입니다. **Execution Provider**를 통해 CPU, GPU, 전용 하드웨어를 지원합니다.

### 4.1 ONNX Runtime 기본 추론

```python
import numpy as np
import onnxruntime as ort


def run_onnx_inference(model_path, input_data):
    """
    Run inference using ONNX Runtime.
    """
    # Create inference session
    session = ort.InferenceSession(model_path)

    # Inspect inputs and outputs
    print("Model inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.type} {inp.shape}")

    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.type} {out.shape}")

    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    result = session.run(
        [output_name],
        {input_name: input_data},
    )

    return result[0]


# Example
# Assuming classifier.onnx was exported earlier
input_data = np.random.randn(1, 784).astype(np.float32)
# output = run_onnx_inference("classifier.onnx", input_data)
# print(f"Output shape: {output.shape}")
```

### 4.2 Execution Provider

ONNX Runtime은 execution provider를 통해 가장 적합한 하드웨어를 선택합니다:

```python
import onnxruntime as ort


def create_optimized_session(model_path, device="cpu"):
    """
    Create an ONNX Runtime session with optimal execution provider.
    """
    # Available providers
    available = ort.get_available_providers()
    print(f"Available providers: {available}")

    # Select provider based on target device
    if device == "cuda" and "CUDAExecutionProvider" in available:
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2 GB limit
                "cudnn_conv_algo_search": "EXHAUSTIVE",
            }),
            "CPUExecutionProvider",  # Fallback
        ]
    elif device == "tensorrt" and "TensorrtExecutionProvider" in available:
        providers = [
            ("TensorrtExecutionProvider", {
                "device_id": 0,
                "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,
                "trt_fp16_enable": True,
            }),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    elif device == "openvino" and "OpenVINOExecutionProvider" in available:
        providers = [
            ("OpenVINOExecutionProvider", {
                "device_type": "CPU",
            }),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    # Session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=providers,
    )

    print(f"Using providers: {session.get_providers()}")
    return session
```

### 4.3 ONNX Runtime 벤치마킹

```python
import numpy as np
import onnxruntime as ort
import time


def benchmark_onnx_runtime(model_path, input_shape, num_warmup=20, num_runs=100):
    """
    Benchmark ONNX Runtime inference latency.
    """
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # Generate random input
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(num_warmup):
        session.run(None, {input_name: input_data})

    # Measure
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: input_data})
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    print(f"ONNX Runtime Benchmark ({model_path}):")
    print(f"  Mean:   {latencies.mean():.2f} ms")
    print(f"  Std:    {latencies.std():.2f} ms")
    print(f"  Min:    {latencies.min():.2f} ms")
    print(f"  Max:    {latencies.max():.2f} ms")
    print(f"  P50:    {np.percentile(latencies, 50):.2f} ms")
    print(f"  P99:    {np.percentile(latencies, 99):.2f} ms")
    print(f"  FPS:    {1000 / latencies.mean():.1f}")

    return latencies


# benchmark_onnx_runtime("mobilenet_v2.onnx", (1, 3, 224, 224))
```

---

## 5. 그래프 최적화

ONNX 그래프는 중복 제거, 연산 융합, 성능 개선을 위해 최적화할 수 있습니다.

### 5.1 ONNX 그래프 최적화

```python
import onnx
from onnxruntime.transformers import optimizer


def optimize_onnx_model(input_path, output_path):
    """
    Apply ONNX graph optimizations.
    """
    import onnxoptimizer

    model = onnx.load(input_path)

    # Available optimization passes
    available_passes = onnxoptimizer.get_available_passes()
    print(f"Available optimization passes ({len(available_passes)}):")
    for p in available_passes[:10]:
        print(f"  {p}")
    print(f"  ... and {len(available_passes) - 10} more")

    # Apply all safe optimizations
    optimized = onnxoptimizer.optimize(model, [
        "eliminate_deadend",          # Remove unused nodes
        "eliminate_identity",          # Remove identity ops
        "eliminate_nop_dropout",       # Remove dropout (eval mode)
        "eliminate_nop_pad",           # Remove zero-padding
        "eliminate_unused_initializer",# Remove unused weights
        "extract_constant_to_initializer",
        "fuse_add_bias_into_conv",     # Fold bias into Conv
        "fuse_bn_into_conv",           # Fuse BatchNorm into Conv
        "fuse_consecutive_transposes", # Merge adjacent transposes
        "fuse_matmul_add_bias_into_gemm",  # MatMul+Add → Gemm
    ])

    onnx.save(optimized, output_path)

    # Compare sizes
    import os
    orig_size = os.path.getsize(input_path)
    opt_size = os.path.getsize(output_path)
    print(f"\nOriginal: {orig_size / 1e6:.1f} MB ({len(model.graph.node)} nodes)")
    print(f"Optimized: {opt_size / 1e6:.1f} MB ({len(optimized.graph.node)} nodes)")
    print(f"Node reduction: {len(model.graph.node) - len(optimized.graph.node)}")

    return optimized


# optimize_onnx_model("mobilenet_v2.onnx", "mobilenet_v2_optimized.onnx")
```

### 5.2 수동 그래프 조작

```python
import onnx
import numpy as np
from onnx import helper, TensorProto


def create_onnx_model_manually():
    """
    Build an ONNX model from scratch to understand the graph structure.

    Creates a simple model: output = ReLU(MatMul(input, W) + bias)
    """
    # Define weight and bias as initializers
    W = np.random.randn(784, 128).astype(np.float32)
    b = np.zeros(128).astype(np.float32)

    W_init = helper.make_tensor("W", TensorProto.FLOAT, [784, 128], W.flatten())
    b_init = helper.make_tensor("bias", TensorProto.FLOAT, [128], b.flatten())

    # Define computation nodes
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["input", "W"],
        outputs=["matmul_out"],
        name="matmul",
    )

    add_node = helper.make_node(
        "Add",
        inputs=["matmul_out", "bias"],
        outputs=["add_out"],
        name="add_bias",
    )

    relu_node = helper.make_node(
        "Relu",
        inputs=["add_out"],
        outputs=["output"],
        name="relu",
    )

    # Define input/output specs
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                                ["batch_size", 784])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                                 ["batch_size", 128])

    # Build graph
    graph = helper.make_graph(
        nodes=[matmul_node, add_node, relu_node],
        name="simple_model",
        inputs=[input_info],
        outputs=[output_info],
        initializer=[W_init, b_init],
    )

    # Build model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.producer_name = "manual_builder"

    # Validate
    onnx.checker.check_model(model)

    # Save
    onnx.save(model, "manual_model.onnx")
    print("Manually created ONNX model saved")
    print(f"Nodes: {[n.name for n in graph.node]}")

    return model


# create_onnx_model_manually()
```

### 5.3 상수 폴딩과 연산자 융합

```
최적화 전:                         최적화 후:
┌─────────┐                        ┌─────────────────┐
│  Conv2d  │                        │  FusedConvBnReLU│
└────┬─────┘                        │  (단일 커널)     │
     │                              └────────┬────────┘
┌────▼─────┐                                 │
│ BatchNorm│  ← 상수 가중치가                  │
└────┬─────┘    Conv에 폴딩됨                  │
     │                                        │
┌────▼─────┐                                 │
│   ReLU   │                                 │
└────┬─────┘                                 ▼

3개 노드 → 1개 융합 노드 → 커널 호출 횟수 감소, 메모리 트래픽 감소
```

```python
import onnxruntime as ort


def demonstrate_ort_optimizations(model_path):
    """
    Show the effect of ONNX Runtime's built-in optimizations.
    """
    # Level 0: No optimization
    opts_0 = ort.SessionOptions()
    opts_0.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    # Level 1: Basic (constant folding, redundancy elimination)
    opts_1 = ort.SessionOptions()
    opts_1.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

    # Level 2: Extended (+ operator fusion like Conv+BN+ReLU)
    opts_2 = ort.SessionOptions()
    opts_2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    # Level 3: All (+ layout optimizations, cross-node fusion)
    opts_3 = ort.SessionOptions()
    opts_3.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Save optimized models for inspection
    for level, opts, name in [
        (0, opts_0, "none"),
        (1, opts_1, "basic"),
        (2, opts_2, "extended"),
        (3, opts_3, "all"),
    ]:
        opts.optimized_model_filepath = f"optimized_level_{name}.onnx"
        session = ort.InferenceSession(model_path, opts)

    print("Optimized models saved. Compare node counts with inspect_onnx_model().")


# demonstrate_ort_optimizations("mobilenet_v2.onnx")
```

---

## 6. 프레임워크 간 모델 변환

ONNX는 프레임워크 간 다리 역할을 합니다. 다음은 일반적인 변환 경로입니다.

### 6.1 PyTorch에서 TensorFlow로 (ONNX 경유)

```python
def pytorch_to_tensorflow_via_onnx(pytorch_model, input_shape, output_dir):
    """
    Convert PyTorch model → ONNX → TensorFlow SavedModel.

    Pipeline: PyTorch → torch.onnx.export → .onnx → onnx-tf → SavedModel
    """
    import torch
    import onnx
    from onnx_tf.backend import prepare

    # Step 1: Export PyTorch to ONNX
    pytorch_model.eval()
    dummy = torch.randn(*input_shape)
    onnx_path = "temp_model.onnx"

    torch.onnx.export(
        pytorch_model, dummy, onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"Step 1: Exported to ONNX ({onnx_path})")

    # Step 2: Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"Step 2: ONNX model validated")

    # Step 3: Convert ONNX to TensorFlow
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(output_dir)
    print(f"Step 3: Exported TensorFlow SavedModel to {output_dir}")

    return output_dir


# Usage:
# pytorch_to_tensorflow_via_onnx(model, (1, 3, 224, 224), "./tf_model")
```

### 6.2 변환 호환성 매트릭스

| From → To | ONNX 경유 | 직접 변환 | 참고 |
|-----------|----------|--------|-------|
| PyTorch → ONNX | `torch.onnx.export` | N/A | 가장 잘 지원되는 경로 |
| TF/Keras → ONNX | `tf2onnx` | N/A | 잘 지원됨 |
| ONNX → TensorFlow | `onnx-tf` | N/A | 일부 연산 불일치 |
| ONNX → TFLite | ONNX → TF → TFLite | `onnx2tflite` | 2단계 필요 |
| ONNX → CoreML | `onnx-coreml` | `coremltools` | coremltools 직접 사용이 더 좋음 |
| ONNX → TensorRT | ORT TensorRT EP | `trtexec` | 네이티브 TensorRT 지원 |

---

## 7. 상호운용성 테스트

변환 후, 내보낸 모델이 원본과 동일한 출력을 생성하는지 반드시 검증해야 합니다.

### 7.1 수치 검증

```python
import numpy as np
import torch
import onnxruntime as ort


def validate_onnx_export(pytorch_model, onnx_path, input_shape,
                          num_samples=10, rtol=1e-3, atol=1e-5):
    """
    Validate that an ONNX model produces the same outputs as the
    original PyTorch model.

    Args:
        pytorch_model: Original PyTorch model (eval mode)
        onnx_path: Path to exported ONNX model
        input_shape: Shape of input tensor (with batch dim)
        num_samples: Number of random inputs to test
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        True if all outputs match within tolerance
    """
    pytorch_model.eval()
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    all_passed = True
    max_abs_diff = 0
    max_rel_diff = 0

    for i in range(num_samples):
        # Generate random input
        np_input = np.random.randn(*input_shape).astype(np.float32)
        torch_input = torch.from_numpy(np_input)

        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(torch_input).numpy()

        # ONNX Runtime inference
        onnx_output = session.run(None, {input_name: np_input})[0]

        # Compare
        abs_diff = np.abs(pytorch_output - onnx_output).max()
        rel_diff = np.abs(
            (pytorch_output - onnx_output)
            / np.maximum(np.abs(pytorch_output), 1e-8)
        ).max()

        max_abs_diff = max(max_abs_diff, abs_diff)
        max_rel_diff = max(max_rel_diff, rel_diff)

        passed = np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)
        if not passed:
            all_passed = False
            print(f"  Sample {i}: FAILED (abs_diff={abs_diff:.6e}, "
                  f"rel_diff={rel_diff:.6e})")

    print(f"\n=== Validation Results ===")
    print(f"Samples tested: {num_samples}")
    print(f"Max absolute difference: {max_abs_diff:.6e}")
    print(f"Max relative difference: {max_rel_diff:.6e}")
    print(f"Tolerance: rtol={rtol}, atol={atol}")
    print(f"Result: {'PASSED' if all_passed else 'FAILED'}")

    return all_passed


# Usage:
# model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1")
# model.eval()
# torch.onnx.export(model, torch.randn(1, 3, 224, 224), "mobilenet.onnx", opset_version=17)
# validate_onnx_export(model, "mobilenet.onnx", (1, 3, 224, 224))
```

### 7.2 형상 및 타입 검증

```python
import onnx
import numpy as np
import onnxruntime as ort


def validate_dynamic_shapes(onnx_path, input_name="input",
                             test_shapes=None):
    """
    Verify that a model with dynamic axes works correctly
    with different input shapes.
    """
    session = ort.InferenceSession(onnx_path)

    if test_shapes is None:
        # Default: test various batch sizes and (if applicable) sequence lengths
        test_shapes = [
            (1, 3, 224, 224),
            (4, 3, 224, 224),
            (16, 3, 224, 224),
            (32, 3, 224, 224),
        ]

    print(f"Testing dynamic shapes on {onnx_path}:")
    for shape in test_shapes:
        try:
            input_data = np.random.randn(*shape).astype(np.float32)
            output = session.run(None, {input_name: input_data})
            print(f"  Input {shape} → Output {output[0].shape}: OK")
        except Exception as e:
            print(f"  Input {shape} → FAILED: {e}")


# validate_dynamic_shapes("mobilenet_v2.onnx", "image")
```

### 7.3 엔드-투-엔드 비교 파이프라인

```python
import torch
import numpy as np
import onnxruntime as ort
import time


class ModelExportValidator:
    """
    Comprehensive validation of model export quality.

    Checks:
    1. Numerical accuracy (output matches original)
    2. Dynamic shape support
    3. Performance (latency comparison)
    4. Model metadata correctness
    """

    def __init__(self, pytorch_model, onnx_path, input_shape):
        self.pytorch_model = pytorch_model.eval()
        self.onnx_path = onnx_path
        self.input_shape = input_shape
        self.session = ort.InferenceSession(onnx_path)

    def run_all_checks(self):
        """Run all validation checks."""
        results = {}

        print("=" * 60)
        print("ONNX Export Validation Report")
        print("=" * 60)

        # Check 1: Numerical accuracy
        print("\n1. Numerical Accuracy")
        results["numerical"] = self._check_numerical()

        # Check 2: Argmax agreement (classification consistency)
        print("\n2. Classification Agreement")
        results["agreement"] = self._check_argmax_agreement()

        # Check 3: Performance
        print("\n3. Performance Comparison")
        results["performance"] = self._check_performance()

        # Check 4: Model info
        print("\n4. Model Information")
        results["info"] = self._check_model_info()

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        all_ok = all(
            v if isinstance(v, bool) else v.get("passed", True)
            for v in results.values()
        )
        print(f"Overall: {'PASSED' if all_ok else 'FAILED'}")

        return results

    def _check_numerical(self, num_samples=20, atol=1e-4):
        diffs = []
        for _ in range(num_samples):
            np_input = np.random.randn(*self.input_shape).astype(np.float32)
            with torch.no_grad():
                pt_out = self.pytorch_model(torch.from_numpy(np_input)).numpy()
            ort_out = self.session.run(None, {
                self.session.get_inputs()[0].name: np_input
            })[0]
            diffs.append(np.abs(pt_out - ort_out).max())

        max_diff = max(diffs)
        mean_diff = np.mean(diffs)
        passed = max_diff < atol
        print(f"  Max diff: {max_diff:.2e} (threshold: {atol})")
        print(f"  Mean diff: {mean_diff:.2e}")
        print(f"  Result: {'PASSED' if passed else 'FAILED'}")
        return {"passed": passed, "max_diff": max_diff}

    def _check_argmax_agreement(self, num_samples=100):
        agreements = 0
        for _ in range(num_samples):
            np_input = np.random.randn(*self.input_shape).astype(np.float32)
            with torch.no_grad():
                pt_class = self.pytorch_model(torch.from_numpy(np_input)).argmax(1).item()
            ort_class = self.session.run(None, {
                self.session.get_inputs()[0].name: np_input
            })[0].argmax(1).item()
            if pt_class == ort_class:
                agreements += 1

        pct = 100 * agreements / num_samples
        print(f"  Agreement: {agreements}/{num_samples} ({pct:.1f}%)")
        return pct == 100

    def _check_performance(self, num_runs=50):
        np_input = np.random.randn(*self.input_shape).astype(np.float32)
        torch_input = torch.from_numpy(np_input)

        # PyTorch
        with torch.no_grad():
            for _ in range(10):  # warmup
                self.pytorch_model(torch_input)
            start = time.perf_counter()
            for _ in range(num_runs):
                self.pytorch_model(torch_input)
            pt_time = (time.perf_counter() - start) / num_runs * 1000

        # ONNX Runtime
        input_name = self.session.get_inputs()[0].name
        for _ in range(10):  # warmup
            self.session.run(None, {input_name: np_input})
        start = time.perf_counter()
        for _ in range(num_runs):
            self.session.run(None, {input_name: np_input})
        ort_time = (time.perf_counter() - start) / num_runs * 1000

        speedup = pt_time / ort_time
        print(f"  PyTorch:     {pt_time:.2f} ms")
        print(f"  ONNX Runtime: {ort_time:.2f} ms")
        print(f"  Speedup:     {speedup:.2f}x")
        return {"pytorch_ms": pt_time, "ort_ms": ort_time, "speedup": speedup}

    def _check_model_info(self):
        import onnx
        import os
        model = onnx.load(self.onnx_path)
        file_size = os.path.getsize(self.onnx_path)
        print(f"  File size:    {file_size / 1e6:.1f} MB")
        print(f"  Opset:        {model.opset_import[0].version}")
        print(f"  Nodes:        {len(model.graph.node)}")
        print(f"  Initializers: {len(model.graph.initializer)}")
        return {"file_size_mb": file_size / 1e6, "num_nodes": len(model.graph.node)}


# Usage:
# model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1").eval()
# torch.onnx.export(model, torch.randn(1,3,224,224), "mv2.onnx", opset_version=17,
#                    input_names=["input"], output_names=["output"],
#                    dynamic_axes={"input":{0:"batch"}, "output":{0:"batch"}})
# validator = ModelExportValidator(model, "mv2.onnx", (1, 3, 224, 224))
# validator.run_all_checks()
```

---

## 8. ONNX 내보내기 모범 사례

### 8.1 내보내기 체크리스트

```python
def onnx_export_checklist():
    """
    Checklist for reliable ONNX model export.
    """
    checklist = [
        ("Set model to eval mode", "model.eval() before export"),
        ("Use appropriate opset", "opset_version=17 (widely supported)"),
        ("Enable constant folding", "do_constant_folding=True"),
        ("Name inputs/outputs", "input_names=['input'], output_names=['output']"),
        ("Set dynamic axes", "For variable batch sizes and sequence lengths"),
        ("Validate after export", "onnx.checker.check_model(model)"),
        ("Numerical comparison", "Compare PyTorch vs ONNX Runtime outputs"),
        ("Test dynamic shapes", "Verify with different batch sizes"),
        ("Benchmark performance", "Compare latency: PyTorch vs ONNX Runtime"),
        ("Check operator support", "Verify all ops are supported in target runtime"),
    ]

    print("=== ONNX Export Checklist ===")
    for i, (item, detail) in enumerate(checklist, 1):
        print(f"  [{' '}] {i:>2}. {item}")
        print(f"       {detail}")


onnx_export_checklist()
```

### 8.2 지원되지 않는 연산 처리

```python
import torch
import torch.nn as nn


class CustomOpWorkaround(nn.Module):
    """
    Example: replacing operations that don't export well to ONNX.
    """

    def forward(self, x):
        # BAD: torch.unique is not supported in many ONNX opsets
        # unique_values = torch.unique(x)

        # GOOD: Decompose into supported operations
        # Or register a custom ONNX operator

        # BAD: Python-level loops
        # for i in range(x.shape[0]):
        #     x[i] = x[i] * 2

        # GOOD: Vectorized operations
        x = x * 2

        return x


# For truly custom operators, register an ONNX symbolic function:
def register_custom_onnx_op():
    """
    Register a custom PyTorch operation for ONNX export.
    """
    from torch.onnx import register_custom_op_symbolic

    # Define how to represent a custom op in the ONNX graph
    def custom_relu_symbolic(g, input):
        return g.op("Relu", input)  # Map to standard ONNX Relu

    # Register
    register_custom_op_symbolic("custom::my_relu", custom_relu_symbolic, opset_version=17)
    print("Custom op registered for ONNX export")
```

---

## 요약

| 개념 | 핵심 요점 |
|---------|-------------|
| **ONNX 포맷** | ML 모델 교환을 위한 개방형 표준 (그래프 + 연산자 + 가중치) |
| **Opset 버전** | 연산자 호환성 -- 광범위한 지원을 위해 opset 17 사용 |
| **PyTorch 내보내기** | eval 모드, 동적 축, 상수 폴딩과 함께 `torch.onnx.export()` |
| **TF/Keras 내보내기** | 변환에 `tf2onnx` 사용 |
| **ONNX Runtime** | CPU/GPU/TensorRT provider를 갖춘 고성능 추론 엔진 |
| **그래프 최적화** | 상수 폴딩, 연산자 융합, 데드 코드 제거 |
| **상호운용성 테스트** | 항상 수치적으로 검증 -- 원본과 내보낸 출력 비교 |
| **동적 형상** | 가변 배치 크기와 시퀀스 길이를 위해 `dynamic_axes` 사용 |
| **변환 경로** | ONNX는 PyTorch, TensorFlow, CoreML, TensorRT 등을 연결 |

---

## 연습 문제

### 연습 1: 기본 내보내기 및 검증

1. MNIST에서 작은 CNN을 훈련하십시오(>98% 정확도)
2. opset 17과 동적 배치 크기로 ONNX로 내보내십시오
3. `onnx.checker.check_model`로 검증하십시오
4. ONNX Runtime으로 추론을 실행하고 PyTorch와 출력을 비교하십시오(1e-5 이내 일치해야 함)
5. 지연 시간을 벤치마크하십시오: PyTorch vs ONNX Runtime (CPU)

### 연습 2: 모델 검사

1. torchvision의 MobileNetV2를 ONNX로 내보내십시오
2. `inspect_onnx_model` 함수로 분석하십시오: 노드 유형, 입출력 형상, 모델 크기
3. ONNX 옵티마이저를 적용하고 최적화 전후의 노드 수를 비교하십시오
4. 어떤 최적화 패스가 가장 큰 영향을 미쳤습니까?

### 연습 3: 동적 형상 테스트

1. 동적 배치와 시퀀스 길이를 가진 텍스트 분류 모델(예: 작은 LSTM)을 내보내십시오
2. 입력 형상 (1, 10), (1, 50), (1, 200), (8, 32), (32, 128)으로 테스트하십시오
3. 모든 형상이 올바른 출력을 생성하는지 검증하십시오(PyTorch와 비교)
4. 추론 지연 시간이 시퀀스 길이에 비례하여 증가하는지 확인하십시오

### 연습 4: ONNX Runtime 최적화 레벨

1. ResNet-18을 ONNX로 내보내십시오
2. 4가지 최적화 레벨(DISABLE_ALL, BASIC, EXTENDED, ALL)의 ONNX Runtime 세션을 만드십시오
3. 각 레벨에서 지연 시간을 벤치마크하십시오
4. 최적화된 모델을 저장하고 노드 수를 비교하십시오
5. 어떤 최적화 레벨이 최고의 속도 향상을 제공합니까?

### 연습 5: 크로스 프레임워크 변환

1. PyTorch CNN을 ONNX로 내보내십시오
2. `onnx-tf`를 사용하여 ONNX 모델을 TensorFlow로 변환하십시오
3. 동일한 입력에 대해 PyTorch, ONNX Runtime, TensorFlow 세 가지로 추론을 실행하십시오
4. 세 프레임워크 간 출력을 비교하십시오. 일치합니까?
5. 각 런타임의 지연 시간을 측정하십시오

---

[이전: 신경 아키텍처 탐색](./07_Neural_Architecture_Search.md) | [개요](./00_Overview.md) | [다음: TFLite와 모바일 런타임](./09_TensorFlow_Lite.md)

**License**: CC BY-NC 4.0
