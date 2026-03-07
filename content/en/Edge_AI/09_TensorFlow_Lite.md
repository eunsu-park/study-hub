# 09. TensorFlow Lite

**Previous**: [Model Compression Techniques](./08_Model_Compression_Techniques.md) | **Next**: [PyTorch Mobile and ExecuTorch](./10_PyTorch_Mobile_and_ExecuTorch.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Use the TFLite converter to transform trained models into edge-deployable format
2. Apply optimization options including post-training quantization and pruning
3. Run inference using the TFLite interpreter with proper input/output handling
4. Configure hardware delegates (GPU, NNAPI, Coral Edge TPU) for acceleration
5. Deploy models to microcontrollers using TFLite for Microcontrollers
6. Benchmark model performance across different optimization configurations

---

TensorFlow Lite is Google's deployment framework that bridges the gap between training powerful models on workstations and running them on resource-constrained edge devices. While earlier lessons covered compression theory, this lesson focuses on the practical TFLite toolchain -- from converting a trained model through optimization to deployment on everything from smartphones to microcontrollers. Mastering TFLite is essential because it remains the most widely adopted edge inference framework, with support spanning Android, iOS, Linux embedded systems, and bare-metal MCUs.

---

## 1. TFLite Converter

### 1.1 Conversion Pipeline

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

### 1.2 Conversion from Different Sources

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

### 1.3 Handling Unsupported Ops

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

## 2. Optimization Options

### 2.1 Quantization Strategies Comparison

| Strategy | Weights | Activations | Size Reduction | Speed Gain | Accuracy Loss |
|----------|---------|-------------|----------------|------------|---------------|
| **None (FP32)** | float32 | float32 | 1x | 1x | None |
| **Dynamic Range** | int8 | float32 | ~4x | ~2-3x | Minimal |
| **Float16** | float16 | float16 | ~2x | ~1.5x (GPU) | Negligible |
| **Full Integer (INT8)** | int8 | int8 | ~4x | ~3-4x | Small |
| **INT8 + Float fallback** | int8 | int8/float | ~4x | ~2-3x | Very small |

### 2.2 Post-Training Quantization

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

### 2.3 Quantization-Aware Training

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

### 2.4 Pruning Before Conversion

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

## 3. TFLite Interpreter

### 3.1 Interpreter Architecture

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

### 3.2 Basic Inference Loop

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

### 3.3 Dynamic Input Shapes

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

## 4. Hardware Delegates

### 4.1 Delegate Overview

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

### 4.5 Edge TPU Model Compilation

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

## 5. TFLite for Microcontrollers

### 5.1 TFLM Architecture

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

### 5.2 TFLM C++ Inference

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

### 5.3 Converting Model to C Array

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

## 6. Model Benchmarking

### 6.1 Comprehensive Benchmarking Tool

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

### 6.2 TFLite Benchmark Tool (Command Line)

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

## Practice Exercises

### Exercise 1: Conversion Pipeline
Train a simple CNN on CIFAR-10, then:
1. Convert to TFLite with no optimization, dynamic quantization, and full INT8 quantization
2. Compare file sizes and measure accuracy on the test set for each variant

### Exercise 2: Delegate Exploration
Using the `EdgeInferenceEngine` class from Section 3.2:
1. Run the same model with 1, 2, and 4 threads and plot the latency results
2. If you have a Coral USB Accelerator, compare CPU vs Edge TPU latency

### Exercise 3: Microcontroller Deployment
1. Train a keyword spotting model (e.g., "yes"/"no" classification on audio features)
2. Quantize to INT8 and convert to a C header array
3. Write the TFLM inference code skeleton targeting Arduino Nano 33 BLE Sense

---

**Previous**: [Model Compression Techniques](./08_Model_Compression_Techniques.md) | **Next**: [PyTorch Mobile and ExecuTorch](./10_PyTorch_Mobile_and_ExecuTorch.md)
