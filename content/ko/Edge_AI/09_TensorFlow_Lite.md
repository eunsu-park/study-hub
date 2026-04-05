# 레슨 9: TensorFlow Lite

[이전: ONNX와 모델 내보내기](./08_ONNX_and_Model_Export.md) | [다음: PyTorch Mobile과 ExecuTorch](./10_PyTorch_Mobile_and_ExecuTorch.md)

---

## 학습 목표

- TFLite 변환기를 사용하여 훈련된 모델을 엣지 배포 가능한 포맷으로 변환한다
- 훈련 후 양자화 및 프루닝을 포함한 최적화 옵션을 적용한다
- 적절한 입출력 처리를 통해 TFLite 인터프리터로 추론을 실행한다
- 하드웨어 delegate(GPU, NNAPI, Coral Edge TPU)를 설정하여 가속한다
- TFLite for Microcontrollers를 사용하여 마이크로컨트롤러에 모델을 배포한다
- 다양한 최적화 구성에서 모델 성능을 벤치마크한다

---

TensorFlow Lite는 워크스테이션에서 강력한 모델을 훈련하는 것과 자원이 제한된 엣지 디바이스에서 실행하는 것 사이의 격차를 해소하는 Google의 배포 프레임워크입니다. 이전 레슨에서 압축 이론을 다루었다면, 이번 레슨은 실제 TFLite 도구 체인에 초점을 맞춥니다 -- 훈련된 모델을 변환하는 것부터 최적화를 거쳐 스마트폰에서 마이크로컨트롤러까지 모든 것에 배포하는 과정을 다룹니다. TFLite를 마스터하는 것이 중요한 이유는, Android, iOS, Linux 임베디드 시스템, 베어메탈 MCU를 아우르는 가장 널리 채택된 엣지 추론 프레임워크이기 때문입니다.

---

## 1. TFLite 변환기

### 1.1 변환 파이프라인

```
+-----------------------------------------------------------------+
|                   TFLite Conversion Pipeline                     |
+-----------------------------------------------------------------+
|                                                                   |
|   Source Formats              Converter           Output          |
|   +----------------+    +------------------+   +-------------+   |
|   | SavedModel     |--->|                  |-->| .tflite     |   |
|   +----------------+    |   TFLiteConverter |   | FlatBuffer  |   |
|   | Keras .h5      |--->|                  |-->| format      |   |
|   +----------------+    |  - Optimization  |   +-------------+   |
|   | Concrete       |--->|  - Quantization  |                     |
|   | Functions      |    |  - Op selection  |                     |
|   +----------------+    +------------------+                     |
|                                                                   |
+-----------------------------------------------------------------+
```

### 1.2 다양한 소스로부터의 변환

```python
#!/usr/bin/env python3
"""TFLite conversion from various source formats."""

import tensorflow as tf
import numpy as np

# --- Method 1: From SavedModel (recommended for production) ---
def convert_from_saved_model(saved_model_dir: str) -> bytes:
    """Convert a SavedModel directory to TFLite."""
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    return tflite_model

# --- Method 2: From Keras model ---
def convert_from_keras(model_path: str) -> bytes:
    """Convert a Keras .h5 or SavedModel-format model to TFLite."""
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    return tflite_model

# --- Method 3: From concrete functions (advanced) ---
def convert_from_concrete_function(model: tf.keras.Model) -> bytes:
    """Convert using a concrete function for fine-grained control."""
    # Trace the model with a specific input signature
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 224, 224, 3], dtype=tf.float32)
    ])
    def serve(x):
        return model(x, training=False)

    concrete_func = serve.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()
    return tflite_model


# Usage
if __name__ == "__main__":
    tflite_bytes = convert_from_saved_model("./my_saved_model")
    with open("model.tflite", "wb") as f:
        f.write(tflite_bytes)
    print(f"Converted model size: {len(tflite_bytes) / 1024:.1f} KB")
```

### 1.3 미지원 연산 처리

```python
#!/usr/bin/env python3
"""Handle models with ops not natively supported in TFLite."""

import tensorflow as tf

def convert_with_select_ops(saved_model_dir: str) -> bytes:
    """Use TF Select ops for unsupported operations.

    TFLite supports a subset of TF operations natively. When your model
    uses ops outside this subset (e.g., tf.py_function, complex string ops),
    enabling Select TF ops pulls in the full TF runtime for those specific
    operations. This increases binary size but preserves model functionality.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # Allow TF ops that are not natively in TFLite
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,        # Default TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS            # Fallback to TF ops
    ]

    # Allow custom ops if needed (e.g., from tf-addons)
    converter.allow_custom_ops = False  # Set True only if you have custom op kernels

    tflite_model = converter.convert()
    return tflite_model


def inspect_model_ops(tflite_path: str):
    """List all operators used in a TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get operator details
    ops = set()
    for i in range(interpreter._interpreter.NumNodes()):
        node = interpreter._interpreter.NodeName(i)
        ops.add(node)

    print("Operators in model:")
    for op in sorted(ops):
        print(f"  - {op}")

    return ops
```

---

## 2. 최적화 옵션

### 2.1 양자화 전략 비교

| 전략 | 가중치 | 활성화 | 크기 감소 | 속도 향상 | 정확도 손실 |
|------|--------|--------|----------|----------|------------|
| **없음 (FP32)** | float32 | float32 | 1x | 1x | 없음 |
| **동적 범위** | int8 | float32 | ~4x | ~2-3x | 최소 |
| **Float16** | float16 | float16 | ~2x | ~1.5x (GPU) | 무시할 수준 |
| **완전 정수 (INT8)** | int8 | int8 | ~4x | ~3-4x | 적음 |
| **INT8 + Float 폴백** | int8 | int8/float | ~4x | ~2-3x | 매우 적음 |

### 2.2 훈련 후 양자화

```python
#!/usr/bin/env python3
"""Post-training quantization with all strategies."""

import tensorflow as tf
import numpy as np
from pathlib import Path


def representative_dataset_gen(dataset_path: str = None, num_samples: int = 200):
    """Generate representative data for calibration.

    The representative dataset is critical for full integer quantization.
    The converter uses these samples to measure the dynamic range of each
    activation tensor, then chooses quantization parameters (scale and
    zero-point) that minimize clipping. Use real data, not random noise.
    """
    # In production, load actual calibration images
    for _ in range(num_samples):
        sample = np.random.randn(1, 224, 224, 3).astype(np.float32)
        yield [sample]


def quantize_model(model_path: str, strategy: str, output_dir: str = "."):
    """Apply post-training quantization.

    Args:
        model_path: Path to SavedModel or .h5 file
        strategy: One of 'dynamic', 'float16', 'int8', 'int8_float_fallback'
        output_dir: Directory for the output .tflite file
    """
    if model_path.endswith(".h5"):
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

    if strategy == "dynamic":
        # Quantize weights to int8 at rest; activations remain float32 at runtime
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    elif strategy == "float16":
        # Reduce precision to float16 -- best when running on GPU delegate
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    elif strategy == "int8":
        # Full integer: both weights and activations are int8
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    elif strategy == "int8_float_fallback":
        # INT8 where possible, float32 fallback for unsupported ops
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen

    tflite_model = converter.convert()

    output_path = Path(output_dir) / f"model_{strategy}.tflite"
    output_path.write_bytes(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"[{strategy}] Size: {size_kb:.1f} KB -> {output_path}")
    return output_path


# Compare all strategies
if __name__ == "__main__":
    model_path = "my_saved_model"

    for strategy in ["dynamic", "float16", "int8_float_fallback"]:
        quantize_model(model_path, strategy, output_dir="./quantized")
```

### 2.3 양자화 인식 훈련

```python
#!/usr/bin/env python3
"""Quantization-Aware Training (QAT) for minimal accuracy loss."""

import tensorflow as tf
import tensorflow_model_optimization as tfmot

def apply_qat(base_model: tf.keras.Model,
              train_dataset,
              epochs: int = 5) -> tf.keras.Model:
    """Apply quantization-aware training.

    QAT inserts fake-quantization nodes during training so the model
    learns to compensate for quantization error. This typically recovers
    most of the accuracy lost by post-training quantization.
    """
    # Annotate the model for quantization-aware training
    qat_model = tfmot.quantization.keras.quantize_model(base_model)

    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Training with quantization awareness...")
    qat_model.fit(train_dataset, epochs=epochs)

    return qat_model


def qat_to_tflite(qat_model: tf.keras.Model) -> bytes:
    """Convert a QAT model to a fully quantized TFLite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    print(f"QAT TFLite model size: {len(tflite_model) / 1024:.1f} KB")
    return tflite_model
```

### 2.4 변환 전 프루닝

```python
#!/usr/bin/env python3
"""Structured and unstructured pruning with TF Model Optimization."""

import tensorflow as tf
import tensorflow_model_optimization as tfmot

def prune_model(base_model: tf.keras.Model,
                train_dataset,
                target_sparsity: float = 0.5,
                epochs: int = 10) -> tf.keras.Model:
    """Apply magnitude-based weight pruning.

    Pruning sets small-magnitude weights to zero. Combined with
    post-training quantization, pruned models compress significantly
    better because runs of zeros compress well in the FlatBuffer format.
    """
    # Define pruning schedule: ramp from 0% to target_sparsity over training
    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=target_sparsity,
            begin_step=0,
            end_step=1000
        )
    }

    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        base_model, **pruning_params
    )

    pruned_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Pruning callback updates the mask each step
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

    pruned_model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=callbacks
    )

    # Strip pruning wrappers for export
    stripped_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    return stripped_model


def pruned_to_tflite(stripped_model: tf.keras.Model) -> bytes:
    """Convert pruned model to TFLite with compression."""
    converter = tf.lite.TFLiteConverter.from_keras_model(stripped_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    print(f"Pruned + quantized size: {len(tflite_model) / 1024:.1f} KB")
    return tflite_model
```

---

## 3. TFLite 인터프리터

### 3.1 인터프리터 아키텍처

```
+-----------------------------------------------------------------+
|                   TFLite Interpreter                             |
+-----------------------------------------------------------------+
|                                                                   |
|   +------------------+                                           |
|   |  .tflite model   |  FlatBuffer (no parsing overhead)        |
|   +--------+---------+                                           |
|            |                                                     |
|            v                                                     |
|   +------------------+    +--------------+                       |
|   |   Interpreter    |--->|  Op Resolver |                       |
|   +--------+---------+    +--------------+                       |
|            |                     |                                |
|            v                     v                                |
|   +------------------+    +--------------+                       |
|   |  Tensor Arena    |    | Built-in Ops |                       |
|   | (pre-allocated)  |    | Custom Ops   |                       |
|   +------------------+    +--------------+                       |
|            |                                                     |
|            v                                                     |
|   +------------------+                                           |
|   |    Delegate      |  Optional: GPU, NNAPI, EdgeTPU           |
|   |   (hardware      |                                           |
|   |    acceleration)  |                                           |
|   +------------------+                                           |
|                                                                   |
+-----------------------------------------------------------------+
```

### 3.2 기본 추론 루프

```python
#!/usr/bin/env python3
"""TFLite interpreter with detailed input/output handling."""

import numpy as np
import time

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


class EdgeInferenceEngine:
    """Wrapper for TFLite inference with multi-output support."""

    def __init__(self, model_path: str, num_threads: int = 4):
        self.interpreter = Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self._log_model_info()

    def _log_model_info(self):
        """Print model input/output specifications."""
        print("=== Model Info ===")
        for i, inp in enumerate(self.input_details):
            print(f"  Input[{i}]: name={inp['name']}, "
                  f"shape={inp['shape']}, dtype={inp['dtype']}")
        for i, out in enumerate(self.output_details):
            print(f"  Output[{i}]: name={out['name']}, "
                  f"shape={out['shape']}, dtype={out['dtype']}")

    def _quantize_input(self, data: np.ndarray, detail: dict) -> np.ndarray:
        """Apply quantization parameters to input if the model is quantized."""
        if detail["dtype"] == np.float32:
            return data.astype(np.float32)

        # For quantized models, apply scale and zero-point
        quant_params = detail.get("quantization_parameters", {})
        scale = quant_params.get("scales", [1.0])[0]
        zero_point = quant_params.get("zero_points", [0])[0]

        quantized = (data / scale + zero_point).astype(detail["dtype"])
        return quantized

    def _dequantize_output(self, data: np.ndarray, detail: dict) -> np.ndarray:
        """Dequantize output back to float if needed."""
        if detail["dtype"] == np.float32:
            return data

        quant_params = detail.get("quantization_parameters", {})
        scale = quant_params.get("scales", [1.0])[0]
        zero_point = quant_params.get("zero_points", [0])[0]

        return (data.astype(np.float32) - zero_point) * scale

    def predict(self, input_data: np.ndarray) -> list:
        """Run inference and return dequantized outputs."""
        # Set input tensor
        processed = self._quantize_input(input_data, self.input_details[0])
        self.interpreter.set_tensor(self.input_details[0]["index"], processed)

        # Invoke
        self.interpreter.invoke()

        # Gather all outputs
        outputs = []
        for detail in self.output_details:
            raw = self.interpreter.get_tensor(detail["index"])
            outputs.append(self._dequantize_output(raw, detail))

        return outputs

    def benchmark(self, num_runs: int = 100) -> dict:
        """Measure inference latency."""
        input_shape = self.input_details[0]["shape"]
        dummy = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(5):
            self.predict(dummy)

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.predict(dummy)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "p50_ms": np.percentile(times, 50),
            "p95_ms": np.percentile(times, 95),
            "p99_ms": np.percentile(times, 99),
        }


if __name__ == "__main__":
    engine = EdgeInferenceEngine("model.tflite", num_threads=4)
    stats = engine.benchmark(num_runs=200)
    print(f"\nLatency: {stats['mean_ms']:.2f} ms "
          f"(p95={stats['p95_ms']:.2f}, p99={stats['p99_ms']:.2f})")
```

### 3.3 동적 입력 형상

```python
#!/usr/bin/env python3
"""Handle models with dynamic input dimensions."""

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


def run_with_dynamic_shape(model_path: str, input_data: np.ndarray):
    """Resize interpreter tensors at runtime for variable-size inputs.

    Some models are exported with dynamic batch or spatial dimensions
    (shape contains -1). You must call resize_tensor_input before
    allocate_tensors to set the actual dimensions.
    """
    interpreter = Interpreter(model_path=model_path)

    # Resize to match actual input
    input_details = interpreter.get_input_details()
    interpreter.resize_tensor_input(
        input_details[0]["index"],
        input_data.shape
    )
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_details[0]["index"],
                           input_data.astype(np.float32))
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    return interpreter.get_tensor(output_details[0]["index"])


# Example: variable batch size
for batch_size in [1, 4, 8]:
    dummy = np.random.randn(batch_size, 224, 224, 3).astype(np.float32)
    result = run_with_dynamic_shape("model.tflite", dummy)
    print(f"Batch {batch_size}: output shape = {result.shape}")
```

---

## 4. 하드웨어 Delegate

### 4.1 Delegate 개요

```
+-----------------------------------------------------------------+
|                  TFLite Delegate System                          |
+-----------------------------------------------------------------+
|                                                                   |
|   Interpreter                                                    |
|   +-----------------------------------------------------------+ |
|   |  Op 1 (Conv2D)  -> [GPU Delegate]   -> GPU execution      | |
|   |  Op 2 (Add)     -> [CPU]            -> CPU fallback       | |
|   |  Op 3 (DepthConv)-> [GPU Delegate]  -> GPU execution      | |
|   |  Op 4 (Softmax) -> [CPU]            -> CPU fallback       | |
|   +-----------------------------------------------------------+ |
|                                                                   |
|   Delegate partitions the graph:                                 |
|   - Supported ops run on accelerator hardware                    |
|   - Unsupported ops fall back to CPU automatically               |
|                                                                   |
+-----------------------------------------------------------------+
```

Delegate는 TFLite 그래프의 특정 연산을 가속 하드웨어로 오프로드하는 메커니즘입니다. 인터프리터는 그래프를 분석하여 지원되는 연산을 delegate에 위임하고, 지원되지 않는 연산은 자동으로 CPU에서 실행합니다. 이를 통해 GPU, DSP, TPU와 같은 하드웨어의 성능을 활용할 수 있습니다.

### 4.2 GPU Delegate

```python
#!/usr/bin/env python3
"""Configure GPU delegate for TFLite inference."""

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter, load_delegate


def create_gpu_interpreter(model_path: str) -> Interpreter:
    """Create interpreter with GPU delegate.

    The GPU delegate offloads supported operations to the device GPU
    via OpenGL ES (Android), Metal (iOS), or OpenCL (Linux).
    Float16 models get the best GPU acceleration because mobile GPUs
    have native FP16 compute units.
    """
    # GPU delegate options
    gpu_delegate = load_delegate("libtensorflowlite_gpu_delegate.so", options={
        "precision_loss_allowed": "1",     # Allow FP16 precision
        "inference_preference": "0",        # 0=fast, 1=balanced, 2=low_power
        "cache_directory": "/tmp/gpu_cache",
        "model_token": "my_model"
    })

    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[gpu_delegate]
    )
    interpreter.allocate_tensors()
    return interpreter


# Android-specific GPU delegate (for reference)
gpu_android_options = {
    "precision_loss_allowed": True,
    "inference_preference": "SUSTAINED_SPEED",
    "serialization_dir": "/data/local/tmp",
}
```

### 4.3 NNAPI Delegate (Android)

```python
#!/usr/bin/env python3
"""NNAPI delegate for Android Neural Networks API acceleration."""

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter, load_delegate


def create_nnapi_interpreter(model_path: str) -> Interpreter:
    """Create interpreter with NNAPI delegate.

    NNAPI is Android's hardware abstraction layer for neural network
    acceleration. It automatically routes operations to the best
    available hardware (DSP, NPU, GPU) on the device. Available
    on Android 8.1+ with accelerator-specific drivers.
    """
    nnapi_delegate = load_delegate("libnnapi_delegate.so", options={
        "execution_preference": "sustained_speed",  # or 'fast_single_answer', 'low_power'
        "allow_fp16": "true",
        "disallow_nnapi_cpu": "true",  # Prevent NNAPI CPU fallback (use TFLite CPU instead)
    })

    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[nnapi_delegate]
    )
    interpreter.allocate_tensors()
    return interpreter
```

NNAPI는 Android의 신경망 가속을 위한 하드웨어 추상화 계층입니다. 디바이스에서 사용 가능한 최적의 하드웨어(DSP, NPU, GPU)로 연산을 자동 라우팅합니다. Android 8.1 이상에서 가속기별 드라이버와 함께 사용할 수 있습니다.

### 4.4 Coral Edge TPU Delegate

```python
#!/usr/bin/env python3
"""Google Coral Edge TPU delegate for INT8 acceleration."""

import numpy as np
import time

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter, load_delegate


def create_coral_interpreter(model_path: str) -> Interpreter:
    """Create interpreter with Coral Edge TPU delegate.

    Requirements:
    - Model must be fully INT8 quantized (no float ops)
    - Compile with edgetpu_compiler before deployment
    - Install libedgetpu runtime on the host

    The Edge TPU runs at 4 TOPS (int8) and draws only 2W,
    making it ideal for always-on vision applications.
    """
    edgetpu_delegate = load_delegate("libedgetpu.so.1")

    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[edgetpu_delegate]
    )
    interpreter.allocate_tensors()
    return interpreter


def coral_benchmark(model_path: str, num_runs: int = 100):
    """Compare CPU vs Edge TPU inference speed."""
    input_shape = None

    results = {}

    for name, use_tpu in [("CPU", False), ("Edge TPU", True)]:
        if use_tpu:
            delegate = load_delegate("libedgetpu.so.1")
            interp = Interpreter(
                model_path=model_path,
                experimental_delegates=[delegate]
            )
        else:
            interp = Interpreter(model_path=model_path)

        interp.allocate_tensors()
        inp = interp.get_input_details()
        out = interp.get_output_details()
        input_shape = inp[0]["shape"]

        dummy = np.random.randint(
            -128, 127, size=input_shape
        ).astype(inp[0]["dtype"])

        # Warmup
        for _ in range(5):
            interp.set_tensor(inp[0]["index"], dummy)
            interp.invoke()

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            interp.set_tensor(inp[0]["index"], dummy)
            interp.invoke()
            times.append((time.perf_counter() - start) * 1000)

        results[name] = np.mean(times)
        print(f"{name}: {np.mean(times):.2f} ms (std={np.std(times):.2f})")

    speedup = results["CPU"] / results["Edge TPU"]
    print(f"\nEdge TPU speedup: {speedup:.1f}x")
```

Coral Edge TPU를 사용하려면 모델이 완전히 INT8로 양자화되어야 하며(float 연산 없음), 배포 전에 `edgetpu_compiler`로 컴파일해야 합니다. Edge TPU는 4 TOPS(int8)의 성능을 2W만으로 제공하므로, 상시 동작하는 비전 애플리케이션에 이상적입니다.

### 4.5 Edge TPU 모델 컴파일

```bash
# Install Edge TPU compiler
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
    sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update && sudo apt-get install edgetpu-compiler

# Compile an INT8 TFLite model for Edge TPU
edgetpu_compiler model_int8.tflite

# Output: model_int8_edgetpu.tflite
# The compiler maps supported ops to the Edge TPU and reports
# which ops fall back to CPU (ideally zero fallback ops).
```

---

## 5. 마이크로컨트롤러를 위한 TFLite

### 5.1 TFLM 아키텍처

```
+-----------------------------------------------------------------+
|            TFLite for Microcontrollers (TFLM)                    |
+-----------------------------------------------------------------+
|                                                                   |
|   Constraints:                                                   |
|   - No OS required (bare-metal)                                  |
|   - No dynamic memory allocation (arena-based)                   |
|   - No filesystem (model embedded in binary)                     |
|   - Typical RAM: 16 KB - 512 KB                                 |
|   - Typical Flash: 64 KB - 2 MB                                 |
|                                                                   |
|   Supported Hardware:                                            |
|   - ARM Cortex-M (M0, M3, M4, M7, M33, M55)                    |
|   - ESP32 (Xtensa, RISC-V)                                      |
|   - Arduino Nano 33 BLE Sense                                   |
|   - Raspberry Pi Pico (RP2040)                                   |
|                                                                   |
+-----------------------------------------------------------------+
```

TFLM(TFLite for Microcontrollers)은 OS 없이 베어메탈 환경에서 동작하며, 동적 메모리 할당이 없고(arena 기반), 파일 시스템이 필요 없습니다(모델이 바이너리에 임베딩됨). 일반적으로 16 KB ~ 512 KB RAM과 64 KB ~ 2 MB Flash를 가진 마이크로컨트롤러에서 실행됩니다.

### 5.2 TFLM C++ 추론

```cpp
// tflm_inference.cc -- TFLite Micro inference on an MCU
// This example shows the "hello world" pattern for TFLM.

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model data (compiled into Flash as a C array)
#include "model_data.h"

namespace {
    // Tensor arena: pre-allocated memory for all intermediate tensors.
    // Size depends on your model. Start large (20 KB) and reduce until
    // allocation fails, then add 10-20% headroom.
    constexpr int kTensorArenaSize = 16 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];
}

void setup() {
    // 1. Load the model from Flash
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        // Handle version mismatch
        return;
    }

    // 2. Register only the ops your model needs (saves code size)
    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddFullyConnected();

    // 3. Create interpreter with the pre-allocated arena
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize
    );
    interpreter.AllocateTensors();

    // 4. Prepare input
    TfLiteTensor* input = interpreter.input(0);
    // Fill input->data.f or input->data.int8 with sensor data

    // 5. Run inference
    TfLiteStatus status = interpreter.Invoke();
    if (status != kTfLiteOk) {
        // Handle error
        return;
    }

    // 6. Read output
    TfLiteTensor* output = interpreter.output(0);
    float prediction = output->data.f[0];
}

void loop() {
    // Repeated inference in the main loop
    // Read sensor -> fill input -> invoke -> read output -> act
}
```

### 5.3 모델을 C 배열로 변환

```bash
# Convert .tflite to C array for embedding in firmware
xxd -i model.tflite > model_data.cc

# Or use the dedicated tool
python3 -c "
import sys
data = open('model.tflite', 'rb').read()
print('const unsigned char model_data[] = {')
for i in range(0, len(data), 12):
    chunk = data[i:i+12]
    print('  ' + ', '.join(f'0x{b:02x}' for b in chunk) + ',')
print('};')
print(f'const unsigned int model_data_len = {len(data)};')
" > model_data.h
```

---

## 6. 모델 벤치마킹

### 6.1 종합 벤치마킹 도구

```python
#!/usr/bin/env python3
"""Comprehensive TFLite model benchmarking."""

import numpy as np
import time
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


@dataclass
class BenchmarkResult:
    model_name: str
    model_size_kb: float
    input_shape: list
    mean_latency_ms: float
    std_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_fps: float
    num_threads: int


def benchmark_tflite(model_path: str,
                     num_threads: int = 4,
                     num_warmup: int = 10,
                     num_runs: int = 200) -> BenchmarkResult:
    """Run a full benchmark suite on a TFLite model."""
    model_size_kb = os.path.getsize(model_path) / 1024

    interpreter = Interpreter(
        model_path=model_path,
        num_threads=num_threads
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = list(input_details[0]["shape"])
    input_dtype = input_details[0]["dtype"]

    # Generate dummy input matching expected dtype
    if input_dtype == np.float32:
        dummy = np.random.randn(*input_shape).astype(np.float32)
    elif input_dtype == np.int8:
        dummy = np.random.randint(-128, 127, size=input_shape).astype(np.int8)
    elif input_dtype == np.uint8:
        dummy = np.random.randint(0, 255, size=input_shape).astype(np.uint8)
    else:
        dummy = np.random.randn(*input_shape).astype(input_dtype)

    # Warmup
    for _ in range(num_warmup):
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()

    # Timed runs
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()
        interpreter.get_tensor(output_details[0]["index"])
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)

    return BenchmarkResult(
        model_name=Path(model_path).stem,
        model_size_kb=round(model_size_kb, 1),
        input_shape=input_shape,
        mean_latency_ms=round(float(np.mean(latencies)), 2),
        std_latency_ms=round(float(np.std(latencies)), 2),
        p50_latency_ms=round(float(np.percentile(latencies, 50)), 2),
        p95_latency_ms=round(float(np.percentile(latencies, 95)), 2),
        p99_latency_ms=round(float(np.percentile(latencies, 99)), 2),
        throughput_fps=round(1000.0 / float(np.mean(latencies)), 1),
        num_threads=num_threads,
    )


def compare_models(model_paths: list, num_threads: int = 4):
    """Benchmark and compare multiple TFLite models side by side."""
    results = []
    for path in model_paths:
        result = benchmark_tflite(path, num_threads=num_threads)
        results.append(result)

    # Print comparison table
    print(f"\n{'Model':<30} {'Size(KB)':>10} {'Mean(ms)':>10} "
          f"{'P95(ms)':>10} {'FPS':>8}")
    print("-" * 75)
    for r in results:
        print(f"{r.model_name:<30} {r.model_size_kb:>10.1f} "
              f"{r.mean_latency_ms:>10.2f} {r.p95_latency_ms:>10.2f} "
              f"{r.throughput_fps:>8.1f}")

    return results


def thread_scaling_test(model_path: str, max_threads: int = 8):
    """Test how inference scales with thread count."""
    print(f"\nThread scaling: {Path(model_path).stem}")
    print(f"{'Threads':>8} {'Mean(ms)':>10} {'FPS':>8} {'Speedup':>8}")
    print("-" * 40)

    baseline = None
    for t in range(1, max_threads + 1):
        result = benchmark_tflite(model_path, num_threads=t, num_runs=100)
        if baseline is None:
            baseline = result.mean_latency_ms
        speedup = baseline / result.mean_latency_ms
        print(f"{t:>8} {result.mean_latency_ms:>10.2f} "
              f"{result.throughput_fps:>8.1f} {speedup:>8.2f}x")


if __name__ == "__main__":
    models = [
        "model_fp32.tflite",
        "model_dynamic.tflite",
        "model_float16.tflite",
        "model_int8.tflite",
    ]

    existing = [m for m in models if os.path.exists(m)]
    if existing:
        compare_models(existing)
        thread_scaling_test(existing[0])
```

### 6.2 TFLite 벤치마크 도구 (커맨드 라인)

```bash
# Google provides a dedicated benchmark binary
# Download for your platform from tensorflow.org

# Basic benchmark
./benchmark_model --graph=model.tflite --num_threads=4 --num_runs=50

# With GPU delegate
./benchmark_model --graph=model.tflite \
    --use_gpu=true \
    --gpu_precision_loss_allowed=true

# With NNAPI delegate
./benchmark_model --graph=model.tflite \
    --use_nnapi=true

# Detailed profiling (per-op timing)
./benchmark_model --graph=model.tflite \
    --num_threads=4 \
    --enable_op_profiling=true
```

---

## 연습 문제

### 연습 1: 변환 파이프라인
간단한 CNN을 CIFAR-10으로 훈련한 후:
1. 최적화 없음, 동적 양자화, 완전 INT8 양자화로 TFLite로 변환합니다
2. 각 변형의 파일 크기를 비교하고 테스트 셋에서 정확도를 측정합니다

### 연습 2: Delegate 탐색
섹션 3.2의 `EdgeInferenceEngine` 클래스를 사용하여:
1. 동일한 모델을 1, 2, 4개 스레드로 실행하고 지연 시간 결과를 그래프로 그립니다
2. Coral USB Accelerator가 있다면 CPU 대 Edge TPU 지연 시간을 비교합니다

### 연습 3: 마이크로컨트롤러 배포
1. 키워드 인식 모델(예: 오디오 특징에 대한 "yes"/"no" 분류)을 훈련합니다
2. INT8로 양자화하고 C 헤더 배열로 변환합니다
3. Arduino Nano 33 BLE Sense를 대상으로 TFLM 추론 코드 스켈레톤을 작성합니다

---

[이전: ONNX와 모델 내보내기](./08_ONNX_and_Model_Export.md) | [다음: PyTorch Mobile과 ExecuTorch](./10_PyTorch_Mobile_and_ExecuTorch.md)
