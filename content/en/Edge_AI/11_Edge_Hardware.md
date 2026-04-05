# 11. Edge Hardware

**Previous**: [PyTorch Mobile and ExecuTorch](./10_PyTorch_Mobile_and_ExecuTorch.md) | **Next**: [On-Device Training](./12_On_Device_Training.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare NPU, TPU, and GPU architectures and their suitability for edge AI workloads
2. Select the appropriate NVIDIA Jetson platform (Nano, Xavier, Orin) for a given use case
3. Deploy INT8 models on Google Coral Edge TPU with performance expectations
4. Identify the capabilities and constraints of Apple Neural Engine, Qualcomm Hexagon, and Intel Movidius
5. Evaluate FPGA-based AI acceleration trade-offs versus fixed-function accelerators
6. Apply a systematic hardware selection framework for edge AI projects

---

Choosing the right hardware is one of the most consequential decisions in an edge AI project. A model that runs at 30 FPS on an NVIDIA Jetson Orin may struggle at 2 FPS on a Raspberry Pi, and a model compiled for Coral Edge TPU will not run at all if it contains float operations. This lesson surveys the major edge AI hardware platforms, explains their architectural differences, and provides a practical decision framework so you can match hardware to application requirements rather than learning about mismatches in production.

---

## 1. NPU / TPU / GPU Comparison

### 1.1 Architecture Overview

```
+-----------------------------------------------------------------+
|            Edge AI Accelerator Architectures                     |
+-----------------------------------------------------------------+
|                                                                   |
|   GPU (Graphics Processing Unit)                                 |
|   +----------------------------------------------------------+  |
|   |  Thousands of small cores, SIMT architecture              |  |
|   |  General-purpose parallel compute                         |  |
|   |  Good for: training + inference, flexible workloads       |  |
|   |  Example: NVIDIA Jetson (Maxwell/Volta/Ampere GPU)        |  |
|   +----------------------------------------------------------+  |
|                                                                   |
|   TPU (Tensor Processing Unit)                                   |
|   +----------------------------------------------------------+  |
|   |  Systolic array for matrix multiply                       |  |
|   |  Fixed-function, high throughput per watt                 |  |
|   |  Good for: INT8 inference at low power                    |  |
|   |  Example: Google Coral Edge TPU (4 TOPS @ 2W)             |  |
|   +----------------------------------------------------------+  |
|                                                                   |
|   NPU (Neural Processing Unit)                                   |
|   +----------------------------------------------------------+  |
|   |  Application-specific neural network accelerator          |  |
|   |  Integrated into SoC (phone, laptop, IoT chip)            |  |
|   |  Good for: always-on low-power inference                  |  |
|   |  Example: Apple ANE, Qualcomm Hexagon, Samsung NPU        |  |
|   +----------------------------------------------------------+  |
|                                                                   |
+-----------------------------------------------------------------+
```

### 1.2 Comparison Table

| Feature | GPU | TPU | NPU |
|---------|-----|-----|-----|
| **Precision** | FP32/FP16/INT8 | INT8 (primary) | INT8/INT16 (varies) |
| **Power** | 5-40W (edge) | 2-5W | 0.5-5W |
| **Flexibility** | High (any workload) | Low (specific ops) | Medium (NN-focused) |
| **Programmability** | CUDA, OpenCL | Compiler-driven | Vendor SDK only |
| **Training** | Yes | No (edge) | No |
| **Latency** | Low-medium | Very low | Very low |
| **Ecosystem** | Mature (CUDA) | Growing | Vendor-locked |
| **Best for** | Complex models, multi-task | Single-model always-on | Mobile/SoC integration |

### 1.3 Performance Density Comparison

```python
#!/usr/bin/env python3
"""Compare edge hardware performance density (TOPS/Watt)."""

edge_hardware = [
    {"name": "Raspberry Pi 4 (CPU)",   "tops": 0.01,  "watts": 5.0},
    {"name": "Coral Edge TPU (USB)",    "tops": 4.0,   "watts": 2.5},
    {"name": "Jetson Nano",            "tops": 0.47,  "watts": 10.0},
    {"name": "Jetson Xavier NX",       "tops": 21.0,  "watts": 15.0},
    {"name": "Jetson Orin Nano",       "tops": 40.0,  "watts": 15.0},
    {"name": "Jetson AGX Orin",        "tops": 275.0, "watts": 60.0},
    {"name": "Apple ANE (M1)",         "tops": 11.0,  "watts": 8.0},
    {"name": "Qualcomm Hexagon 698",   "tops": 15.0,  "watts": 5.0},
    {"name": "Intel Movidius Myriad X", "tops": 4.0,   "watts": 1.5},
    {"name": "Hailo-8",               "tops": 26.0,  "watts": 2.5},
]

# Sort by TOPS/Watt (efficiency)
ranked = sorted(
    edge_hardware,
    key=lambda h: h["tops"] / h["watts"],
    reverse=True
)

print(f"{'Hardware':<30} {'TOPS':>8} {'Watts':>8} {'TOPS/W':>10}")
print("-" * 60)
for hw in ranked:
    efficiency = hw["tops"] / hw["watts"]
    print(f"{hw['name']:<30} {hw['tops']:>8.1f} {hw['watts']:>8.1f} "
          f"{efficiency:>10.2f}")
```

---

## 2. NVIDIA Jetson Family

### 2.1 Jetson Product Line

```
+-----------------------------------------------------------------+
|                NVIDIA Jetson Family (2024)                        |
+-----------------------------------------------------------------+
|                                                                   |
|   Entry                 Mid-Range              High-End           |
|   +-----------+        +-----------+        +-----------+        |
|   | Jetson    |        | Xavier NX |        | AGX Orin  |        |
|   | Orin Nano |        | / Orin NX |        | / IGX     |        |
|   +-----------+        +-----------+        +-----------+        |
|   40 TOPS              100 TOPS             275 TOPS             |
|   7-15W                15-25W               15-60W               |
|   $199                 $399-599             $999-1999            |
|                                                                   |
|   All run:                                                       |
|   - JetPack (Ubuntu + CUDA + cuDNN + TensorRT)                  |
|   - Full Linux with GPU acceleration                             |
|   - Docker containers for isolation                              |
|                                                                   |
+-----------------------------------------------------------------+
```

### 2.2 Jetson Specifications

| Spec | Orin Nano | Orin NX | AGX Orin |
|------|-----------|---------|----------|
| **GPU** | 1024 CUDA cores (Ampere) | 2048 CUDA cores | 2048 CUDA + 64 Tensor cores |
| **CPU** | 6-core Cortex-A78AE | 8-core Cortex-A78AE | 12-core Cortex-A78AE |
| **AI Perf** | 40 TOPS (INT8) | 100 TOPS (INT8) | 275 TOPS (INT8) |
| **RAM** | 4-8 GB LPDDR5 | 8-16 GB LPDDR5 | 32-64 GB LPDDR5 |
| **Storage** | NVMe | NVMe | 64 GB eMMC + NVMe |
| **Power** | 7-15W | 10-25W | 15-60W |
| **Video** | 2x 4K decode | 4x 4K decode | 8x 4K decode |
| **Use case** | Smart cameras, robots | Multi-model pipelines | Autonomous vehicles |

### 2.3 Jetson Setup and TensorRT Inference

```python
#!/usr/bin/env python3
"""NVIDIA Jetson: TensorRT inference pipeline."""

import numpy as np
import time

# TensorRT is pre-installed in JetPack
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TensorRTInference:
    """TensorRT engine wrapper for Jetson deployment."""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        # Load serialized engine
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate device memory
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = np.prod(shape) * np.dtype(dtype).itemsize

            device_mem = cuda.mem_alloc(size)
            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    "name": name, "shape": shape, "dtype": dtype,
                    "device": device_mem, "host": np.empty(shape, dtype=dtype)
                })
            else:
                self.outputs.append({
                    "name": name, "shape": shape, "dtype": dtype,
                    "device": device_mem, "host": np.empty(shape, dtype=dtype)
                })

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on GPU."""
        # Copy input to device
        np.copyto(self.inputs[0]["host"], input_data.ravel()
                  .reshape(self.inputs[0]["shape"]))
        cuda.memcpy_htod_async(
            self.inputs[0]["device"],
            self.inputs[0]["host"],
            self.stream
        )

        # Execute
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # Copy output to host
        cuda.memcpy_dtoh_async(
            self.outputs[0]["host"],
            self.outputs[0]["device"],
            self.stream
        )
        self.stream.synchronize()

        return self.outputs[0]["host"].copy()

    def benchmark(self, num_runs: int = 100) -> dict:
        """Measure throughput and latency."""
        dummy = np.random.randn(
            *self.inputs[0]["shape"]
        ).astype(self.inputs[0]["dtype"])

        # Warmup
        for _ in range(10):
            self.infer(dummy)

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.infer(dummy)
            times.append((time.perf_counter() - start) * 1000)

        return {
            "mean_ms": np.mean(times),
            "p95_ms": np.percentile(times, 95),
            "fps": 1000.0 / np.mean(times),
        }


if __name__ == "__main__":
    engine = TensorRTInference("model.engine")
    stats = engine.benchmark(200)
    print(f"Latency: {stats['mean_ms']:.2f} ms | FPS: {stats['fps']:.1f}")
```

### 2.4 ONNX to TensorRT Conversion

```bash
# Convert ONNX model to TensorRT engine on Jetson
# trtexec is included in JetPack

# FP32 engine
trtexec --onnx=model.onnx --saveEngine=model_fp32.engine

# FP16 engine (2x faster on Jetson GPUs with Tensor Cores)
trtexec --onnx=model.onnx --saveEngine=model_fp16.engine --fp16

# INT8 engine (requires calibration data)
trtexec --onnx=model.onnx --saveEngine=model_int8.engine \
    --int8 --calib=calibration_cache.bin

# Benchmark the engine
trtexec --loadEngine=model_fp16.engine --batch=1 --iterations=200
```

---

## 3. Google Coral (Edge TPU)

### 3.1 Coral Product Line

| Product | Interface | Power | Use Case |
|---------|-----------|-------|----------|
| **USB Accelerator** | USB 3.0 | ~2.5W | Add to any Linux host |
| **Dev Board** | Standalone | ~5W | Prototyping (SoM + carrier) |
| **Dev Board Micro** | Standalone | ~1W | MCU with Edge TPU |
| **M.2 Accelerator** | M.2 E-key | ~2W | Embedded integration |
| **Mini PCIe** | PCIe | ~2W | Industrial/gateway |

### 3.2 Edge TPU Constraints

```
+-----------------------------------------------------------------+
|               Edge TPU Constraints                               |
+-----------------------------------------------------------------+
|                                                                   |
|   The Edge TPU achieves 4 TOPS at 2W by trading flexibility     |
|   for efficiency. Key constraints:                               |
|                                                                   |
|   1. INT8 only -- model must be fully quantized to int8          |
|      No float operations allowed on the TPU                      |
|                                                                   |
|   2. Supported ops -- only a subset of TFLite ops:               |
|      Conv2D, DepthwiseConv2D, FullyConnected, AveragePool,      |
|      MaxPool, Concatenation, Reshape, Softmax, L2Norm,          |
|      Add, Mul, ResizeBilinear, Logistic, Mean                   |
|                                                                   |
|   3. No op after first unsupported op runs on TPU                |
|      The compiler maps a contiguous prefix of the graph          |
|      to the TPU; everything after the first fallback runs        |
|      on CPU. Design your model to keep supported ops together.   |
|                                                                   |
|   4. On-chip SRAM is 8 MB                                       |
|      Models with parameter tensors exceeding 8 MB spill to      |
|      host memory, dramatically reducing throughput.              |
|                                                                   |
+-----------------------------------------------------------------+
```

### 3.3 Coral Pipeline

```python
#!/usr/bin/env python3
"""Full Coral Edge TPU deployment pipeline."""

import numpy as np
import time
import subprocess
from pathlib import Path

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter, load_delegate


def compile_for_edgetpu(tflite_path: str) -> str:
    """Compile an INT8 TFLite model for Edge TPU.

    The edgetpu_compiler analyzes each operation and maps it to
    the Edge TPU if supported. It reports which operations were
    mapped and which fall back to CPU.
    """
    result = subprocess.run(
        ["edgetpu_compiler", tflite_path, "-o", "."],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Compilation failed: {result.stderr}")
        return None

    # Output file has _edgetpu suffix
    stem = Path(tflite_path).stem
    edgetpu_path = f"{stem}_edgetpu.tflite"
    return edgetpu_path


class CoralInference:
    """Inference engine for Coral Edge TPU."""

    def __init__(self, model_path: str, use_tpu: bool = True):
        delegates = []
        if use_tpu:
            delegates.append(load_delegate("libedgetpu.so.1"))

        self.interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=delegates
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference."""
        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            input_data.astype(self.input_details[0]["dtype"])
        )
        self.interpreter.invoke()
        return self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )

    def benchmark(self, num_runs: int = 200) -> dict:
        """Compare CPU vs Edge TPU latency."""
        shape = self.input_details[0]["shape"]
        dtype = self.input_details[0]["dtype"]

        if dtype == np.uint8:
            dummy = np.random.randint(0, 255, size=shape).astype(np.uint8)
        else:
            dummy = np.random.randint(-128, 127, size=shape).astype(np.int8)

        # Warmup
        for _ in range(10):
            self.predict(dummy)

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.predict(dummy)
            times.append((time.perf_counter() - start) * 1000)

        return {
            "mean_ms": np.mean(times),
            "p95_ms": np.percentile(times, 95),
            "fps": 1000.0 / np.mean(times),
        }


if __name__ == "__main__":
    # Compare CPU vs Edge TPU
    model_path = "mobilenet_v2_int8_edgetpu.tflite"

    print("=== Edge TPU ===")
    tpu_engine = CoralInference(model_path, use_tpu=True)
    tpu_stats = tpu_engine.benchmark()
    print(f"  Latency: {tpu_stats['mean_ms']:.2f} ms | FPS: {tpu_stats['fps']:.1f}")

    print("\n=== CPU ===")
    cpu_engine = CoralInference(model_path, use_tpu=False)
    cpu_stats = cpu_engine.benchmark()
    print(f"  Latency: {cpu_stats['mean_ms']:.2f} ms | FPS: {cpu_stats['fps']:.1f}")

    speedup = cpu_stats["mean_ms"] / tpu_stats["mean_ms"]
    print(f"\nEdge TPU speedup: {speedup:.1f}x")
```

---

## 4. Apple Neural Engine

### 4.1 ANE Architecture

```
+-----------------------------------------------------------------+
|                Apple Neural Engine (ANE)                          |
+-----------------------------------------------------------------+
|                                                                   |
|   Integrated into Apple Silicon (A-series, M-series):            |
|                                                                   |
|   A14 Bionic:  11 TOPS (16-core ANE)                            |
|   A15 Bionic:  15.8 TOPS (16-core ANE)                          |
|   M1:          11 TOPS (16-core ANE)                             |
|   M2:          15.8 TOPS (16-core ANE)                           |
|   M3:          18 TOPS (16-core ANE)                             |
|   M4:          38 TOPS (16-core ANE)                             |
|                                                                   |
|   Access via Core ML framework:                                  |
|   - Automatic dispatch to ANE / GPU / CPU                        |
|   - coremltools for model conversion                             |
|   - MLComputeUnits: .all, .cpuAndNeuralEngine, .cpuOnly          |
|                                                                   |
+-----------------------------------------------------------------+
```

### 4.2 Core ML Conversion

```python
#!/usr/bin/env python3
"""Convert PyTorch model to Core ML for Apple Neural Engine."""

import torch
import coremltools as ct


def convert_to_coreml(model: torch.nn.Module,
                      input_shape: tuple,
                      output_path: str = "model.mlpackage"):
    """Convert a PyTorch model to Core ML format.

    Core ML automatically decides whether to run on ANE, GPU, or CPU
    based on the operations in the model. To maximize ANE utilization:
    - Use standard convolution, pooling, and linear ops
    - Avoid custom ops or unsupported activations
    - Prefer float16 precision (ANE is optimized for FP16)
    """
    model.eval()
    example_input = torch.randn(*input_shape)
    traced = torch.jit.trace(model, example_input)

    # Convert to Core ML
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(shape=input_shape)],
        convert_to="mlprogram",           # Modern format (vs "neuralnetwork")
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,  # FP16 for ANE
    )

    mlmodel.save(output_path)
    print(f"Core ML model saved: {output_path}")

    # Print model metadata
    spec = mlmodel.get_spec()
    print(f"Compute units: {spec.description}")

    return mlmodel


def set_compute_units(model_path: str,
                      units: str = "ALL") -> None:
    """Configure which compute units the model can use.

    Options:
    - ALL: ANE + GPU + CPU (default, lets Core ML choose)
    - CPU_AND_NE: ANE + CPU only (avoid GPU for battery)
    - CPU_ONLY: CPU only (debugging)
    - CPU_AND_GPU: GPU + CPU (skip ANE)
    """
    import coremltools as ct

    model = ct.models.MLModel(model_path)

    # This is set at prediction time, not in the model file
    # Shown here for reference
    compute_options = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    }
    print(f"Available compute units: {list(compute_options.keys())}")
```

---

## 5. Qualcomm Hexagon and Intel Movidius

### 5.1 Qualcomm Hexagon DSP/NPU

```
+-----------------------------------------------------------------+
|               Qualcomm AI Stack                                  |
+-----------------------------------------------------------------+
|                                                                   |
|   Application                                                    |
|   +----------------------------------------------------------+  |
|   |  Qualcomm AI Engine Direct (QNN)                          |  |
|   |  Unified API for all Qualcomm accelerators                |  |
|   +----------------------------------------------------------+  |
|        |              |              |              |             |
|   +--------+    +--------+    +--------+    +--------+          |
|   | Hexagon|    | Adreno |    | Kryo   |    | Sensing|          |
|   | NPU    |    | GPU    |    | CPU    |    | Hub    |          |
|   +--------+    +--------+    +--------+    +--------+          |
|   INT8/INT16    FP16/FP32    FP32          Always-on            |
|   15+ TOPS      ~2 TFLOPS    General      low-power             |
|                                                                   |
|   Snapdragon 8 Gen 3: 45+ TOPS (NPU)                           |
|   Snapdragon X Elite: 45 TOPS (NPU) for laptops                |
|                                                                   |
+-----------------------------------------------------------------+
```

### 5.2 Intel Movidius (OpenVINO)

```python
#!/usr/bin/env python3
"""Intel Movidius / OpenVINO inference for edge deployment."""

from openvino.runtime import Core
import numpy as np
import time


class OpenVINOInference:
    """OpenVINO inference engine for Intel hardware.

    OpenVINO supports:
    - Intel CPUs (optimized with AVX-512, VNNI)
    - Intel integrated GPUs
    - Intel Movidius Myriad X VPU (4 TOPS, 1.5W)
    - Intel NPU (Meteor Lake and newer)
    """

    def __init__(self, model_path: str, device: str = "CPU"):
        """
        Args:
            model_path: Path to ONNX or OpenVINO IR (.xml/.bin)
            device: "CPU", "GPU", "MYRIAD" (VPU), "NPU"
        """
        self.core = Core()
        self.model = self.core.read_model(model_path)

        # Compile for target device
        self.compiled = self.core.compile_model(self.model, device)
        self.infer_request = self.compiled.create_infer_request()

        # Get input/output info
        self.input_name = self.model.input(0).any_name
        self.input_shape = list(self.model.input(0).shape)
        self.output_name = self.model.output(0).any_name

        print(f"Loaded model on {device}: input={self.input_shape}")

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference."""
        self.infer_request.infer({self.input_name: input_data})
        return self.infer_request.get_output_tensor(0).data.copy()

    def benchmark(self, num_runs: int = 100) -> dict:
        """Measure inference performance."""
        dummy = np.random.randn(*self.input_shape).astype(np.float32)

        for _ in range(10):
            self.predict(dummy)

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.predict(dummy)
            times.append((time.perf_counter() - start) * 1000)

        return {
            "mean_ms": np.mean(times),
            "p95_ms": np.percentile(times, 95),
            "fps": 1000.0 / np.mean(times),
        }


if __name__ == "__main__":
    # Compare devices
    for device in ["CPU", "GPU"]:
        try:
            engine = OpenVINOInference("model.onnx", device=device)
            stats = engine.benchmark()
            print(f"{device}: {stats['mean_ms']:.2f} ms | "
                  f"{stats['fps']:.1f} FPS")
        except Exception as e:
            print(f"{device}: Not available ({e})")
```

---

## 6. FPGA for AI

### 6.1 FPGA vs Fixed-Function Accelerators

```
+-----------------------------------------------------------------+
|          FPGA vs Fixed-Function (ASIC/TPU/NPU)                   |
+-----------------------------------------------------------------+
|                                                                   |
|   FPGA                             Fixed-Function                |
|   +--------------------------+    +--------------------------+   |
|   | Reconfigurable fabric   |    | Hardwired logic          |   |
|   | Custom data paths       |    | Optimized for specific   |   |
|   | Any bit-width (1-64)    |    | operations (INT8/FP16)   |   |
|   | Field-updatable         |    | Fixed after fabrication  |   |
|   +--------------------------+    +--------------------------+   |
|                                                                   |
|   Advantages:                     Advantages:                    |
|   + Custom precision (INT4, etc)  + Higher peak TOPS/W           |
|   + Custom pipeline stages        + Lower unit cost at scale     |
|   + Low, deterministic latency    + Simpler toolchain            |
|   + Update hardware after deploy  + No FPGA expertise needed     |
|                                                                   |
|   Disadvantages:                  Disadvantages:                 |
|   - Higher unit cost              - Fixed to supported ops       |
|   - Complex development (HDL)     - New model = may need new HW  |
|   - Lower peak throughput         - Less flexibility             |
|   - Long development cycle                                       |
|                                                                   |
|   Best for: low-latency, custom precision, small batch,          |
|   regulated industries, research                                 |
|                                                                   |
+-----------------------------------------------------------------+
```

### 6.2 FPGA AI Frameworks

| Framework | Vendor | Description |
|-----------|--------|-------------|
| **Vitis AI** | AMD/Xilinx | DPU overlay for CNN inference |
| **OpenVINO** | Intel | Supports Intel FPGAs (Arria, Stratix) |
| **hls4ml** | Open source | HLS synthesis from Keras/PyTorch models |
| **FINN** | AMD/Xilinx | Quantized NN on FPGA (binary/ternary) |

```python
#!/usr/bin/env python3
"""hls4ml: Convert a Keras model to FPGA firmware (HLS C++)."""

# hls4ml generates synthesizable HLS C++ from a trained model.
# The output can be compiled with Vivado HLS for AMD/Xilinx FPGAs.

import tensorflow as tf
import hls4ml
import numpy as np


def convert_to_hls(model: tf.keras.Model,
                   output_dir: str = "hls_project",
                   fpga_part: str = "xcu250-figd2104-2L-e"):
    """Convert Keras model to HLS project for FPGA synthesis."""

    # Configure HLS conversion
    hls_config = hls4ml.utils.config_from_keras_model(model, granularity="name")

    # Set precision per layer (reduce for smaller FPGA footprint)
    for layer_name in hls_config["LayerName"]:
        hls_config["LayerName"][layer_name]["Precision"] = "ap_fixed<16,6>"
        hls_config["LayerName"][layer_name]["ReuseFactor"] = 4

    # Convert
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=hls_config,
        output_dir=output_dir,
        fpga_part=fpga_part,
        clock_period=5,  # 200 MHz target
    )

    # Compile (generates HLS C++ and runs C-simulation)
    hls_model.compile()

    # Predict (C-simulation)
    test_input = np.random.randn(1, 16).astype(np.float32)
    hls_output = hls_model.predict(test_input)
    keras_output = model.predict(test_input)

    print(f"Keras output:  {keras_output[0][:5]}")
    print(f"HLS output:    {hls_output[0][:5]}")
    print(f"Max diff:      {np.abs(keras_output - hls_output).max():.6f}")

    # Generate synthesis report
    hls_model.build(synth=True)

    return hls_model
```

---

## 7. Hardware Selection Criteria

### 7.1 Decision Framework

```python
#!/usr/bin/env python3
"""Edge hardware selection decision framework."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ProjectRequirements:
    """Define your edge AI project requirements."""
    model_size_mb: float          # Trained model size
    target_latency_ms: float      # Maximum acceptable inference latency
    target_fps: float             # Frames/inferences per second
    power_budget_watts: float     # Maximum power consumption
    unit_cost_usd: float          # Per-unit hardware budget
    needs_training: bool          # On-device training required?
    environment: str              # "indoor", "outdoor", "industrial"
    connectivity: str             # "always_on", "intermittent", "offline"
    volume: str                   # "prototype", "hundreds", "thousands", "millions"
    precision: str                # "fp32", "fp16", "int8"


HARDWARE_DB = {
    "Raspberry Pi 5": {
        "tops": 0.02, "watts": 8, "cost": 80,
        "training": False, "precision": ["fp32", "fp16", "int8"],
        "pros": "Cheap, huge community, full Linux",
        "cons": "No accelerator, slow inference",
    },
    "Coral USB Accelerator": {
        "tops": 4.0, "watts": 2.5, "cost": 60,
        "training": False, "precision": ["int8"],
        "pros": "Very fast INT8, low power, plug-and-play",
        "cons": "INT8 only, limited model size (8 MB on-chip)",
    },
    "Jetson Orin Nano": {
        "tops": 40, "watts": 15, "cost": 249,
        "training": True, "precision": ["fp32", "fp16", "int8"],
        "pros": "CUDA + TensorRT, good ecosystem, powerful",
        "cons": "Higher power, higher cost",
    },
    "Jetson AGX Orin": {
        "tops": 275, "watts": 60, "cost": 1999,
        "training": True, "precision": ["fp32", "fp16", "int8"],
        "pros": "Massive performance, multi-model, training capable",
        "cons": "Expensive, high power",
    },
    "Hailo-8": {
        "tops": 26, "watts": 2.5, "cost": 100,
        "training": False, "precision": ["int8", "int16"],
        "pros": "Excellent TOPS/W, M.2 form factor",
        "cons": "Smaller ecosystem, compiler constraints",
    },
}


def recommend_hardware(req: ProjectRequirements) -> list:
    """Recommend hardware based on project requirements."""
    candidates = []

    for name, hw in HARDWARE_DB.items():
        # Filter by hard constraints
        if hw["watts"] > req.power_budget_watts * 1.2:
            continue
        if hw["cost"] > req.unit_cost_usd:
            continue
        if req.needs_training and not hw["training"]:
            continue
        if req.precision not in hw["precision"]:
            continue

        # Score remaining candidates
        score = 0
        score += min(hw["tops"] / 10, 10)   # Performance (cap at 10)
        score += (1 - hw["watts"] / req.power_budget_watts) * 5  # Power efficiency
        score += (1 - hw["cost"] / req.unit_cost_usd) * 3  # Cost efficiency

        candidates.append((name, score, hw))

    # Sort by score
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


if __name__ == "__main__":
    req = ProjectRequirements(
        model_size_mb=5.0,
        target_latency_ms=50,
        target_fps=20,
        power_budget_watts=15,
        unit_cost_usd=300,
        needs_training=False,
        environment="indoor",
        connectivity="always_on",
        volume="hundreds",
        precision="int8",
    )

    results = recommend_hardware(req)
    print("Hardware Recommendations:")
    print("-" * 60)
    for name, score, hw in results:
        print(f"  {name:<25} Score: {score:.1f}")
        print(f"    {hw['tops']} TOPS | {hw['watts']}W | ${hw['cost']}")
        print(f"    Pros: {hw['pros']}")
        print(f"    Cons: {hw['cons']}")
        print()
```

---

## Practice Exercises

### Exercise 1: Hardware Comparison
1. Pick two hardware platforms from this lesson (e.g., Coral USB and Jetson Orin Nano)
2. Deploy the same MobileNetV2 model on both (INT8 for Coral, FP16 for Jetson)
3. Compare latency, throughput, power consumption, and accuracy

### Exercise 2: Selection Framework
1. Define requirements for a warehouse security camera system (10 cameras, person/vehicle detection, outdoor, 24/7)
2. Use the `recommend_hardware` function to rank candidates
3. Justify your top pick with a cost-performance analysis

### Exercise 3: TensorRT Optimization
1. Convert an ONNX object detection model (e.g., SSD-MobileNet) to TensorRT FP16
2. Benchmark FP32 vs FP16 vs INT8 engines using `trtexec`
3. Measure accuracy on a validation set for each precision level

---

**Previous**: [PyTorch Mobile and ExecuTorch](./10_PyTorch_Mobile_and_ExecuTorch.md) | **Next**: [On-Device Training](./12_On_Device_Training.md)
