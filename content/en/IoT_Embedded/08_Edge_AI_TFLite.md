# 08. Edge AI - TensorFlow Lite

**Previous**: [HTTP/REST for IoT](./07_HTTP_REST_for_IoT.md) | **Next**: [Edge AI - ONNX Runtime](./09_Edge_AI_ONNX.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain Edge AI concepts and compare them with cloud-based AI inference
2. Describe the TensorFlow Lite workflow from model training to edge deployment
3. Convert Keras models to TFLite format with various quantization options
4. Run inference on Raspberry Pi using the TFLite interpreter
5. Build an image classifier that runs locally on edge hardware
6. Integrate Edge AI results with MQTT for IoT data pipelines

---

Processing AI models on edge devices rather than in the cloud eliminates network latency, preserves privacy, and keeps your system working even when the internet goes down. For IoT applications like real-time defect detection on a factory floor or person detection on a security camera, waiting hundreds of milliseconds for a cloud round-trip is simply not an option. TensorFlow Lite makes it possible to run neural networks on devices as small as a Raspberry Pi.

---

> **Analogy: Branch Office** -- Cloud AI is like sending every decision to corporate headquarters -- accurate but slow and expensive. Edge AI is the local branch office that handles routine decisions (is this a person? is this temperature abnormal?) on-site in milliseconds, only escalating unusual cases to headquarters. TensorFlow Lite is the compact toolkit that fits in the branch office.

## 1. Edge AI Concepts

### 1.1 What is Edge AI?

**Edge AI** performs AI inference directly on edge devices (Raspberry Pi, smartphones, etc.) rather than in the cloud.

```
┌─────────────────────────────────────────────────────────────┐
│              Cloud AI vs Edge AI                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Cloud AI                        Edge AI                    │
│   ┌─────────┐                       ┌─────────┐             │
│   │  Sensor │                       │  Sensor │             │
│   └────┬────┘                       └────┬────┘             │
│        │ Data                            │ Data              │
│        ▼                                 ▼                  │
│   ┌─────────┐                       ┌─────────┐             │
│   │ Network │                       │   Edge  │             │
│   └────┬────┘                       │   AI    │             │
│        │                            └────┬────┘             │
│        ▼                                 │ Result            │
│   ┌─────────┐                            ▼                  │
│   │  Cloud  │                       ┌─────────┐             │
│   │   AI    │                       │  Action │             │
│   └────┬────┘                       └─────────┘             │
│        │                                                    │
│        ▼                            Advantages:              │
│   ┌─────────┐                       • Low latency (< 50ms)  │
│   │  Action │                       • Offline operation     │
│   └─────────┘                       • Privacy               │
│                                     • Cost reduction         │
│   Disadvantages:                                             │
│   • High latency (100ms+)                                    │
│   • Network dependency                                       │
│   • Data transfer costs                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Edge AI Use Cases

| Domain | Applications |
|--------|--------------|
| **Smart Home** | Face recognition door locks, voice recognition |
| **Industry** | Defect detection, predictive maintenance |
| **Healthcare** | Wearable health monitoring |
| **Agriculture** | Crop disease detection, pest identification |
| **Automotive** | ADAS, pedestrian detection |

### 1.3 Edge AI Framework Comparison

| Framework | Developer | Features | Hardware Support |
|-----------|-----------|----------|------------------|
| **TensorFlow Lite** | Google | General-purpose, rich ecosystem | CPU, GPU, Edge TPU |
| **ONNX Runtime** | Microsoft | Multi-framework compatible | CPU, GPU |
| **OpenVINO** | Intel | Intel optimized | Intel CPU/GPU |
| **TensorRT** | NVIDIA | NVIDIA GPU optimized | NVIDIA GPU |

---

## 2. TensorFlow Lite Overview

### 2.1 TFLite Features

```python
# TensorFlow Lite features
tflite_features = {
    "Lightweight": "Model size reduction (1/4 with quantization)",
    "Optimized": "Mobile/embedded inference optimization",
    "Hardware Acceleration": "GPU, Edge TPU, DSP support",
    "Cross-platform": "Android, iOS, Linux, MCU",
    "Operators": "TF operator subset support"
}
```

### 2.2 Installation on Raspberry Pi

```bash
# Method 1: tflite-runtime (recommended, lightweight)
pip install tflite-runtime

# Method 2: Full TensorFlow (heavy)
# pip install tensorflow

# Additional packages
pip install numpy pillow
```

### 2.3 TFLite Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                TensorFlow Lite Workflow                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   1. Model Training (PC/Cloud)                               │
│      ┌────────────────────┐                                 │
│      │  TensorFlow/Keras  │                                 │
│      │     Model (.h5)    │                                 │
│      └─────────┬──────────┘                                 │
│                │                                             │
│   2. Model Conversion                                        │
│      ┌─────────▼──────────┐                                 │
│      │   TFLite Converter │                                 │
│      │ (quantization opt) │                                 │
│      └─────────┬──────────┘                                 │
│                │                                             │
│   3. Optimized Model                                         │
│      ┌─────────▼──────────┐                                 │
│      │   model.tflite     │                                 │
│      │ (lightweight model)│                                 │
│      └─────────┬──────────┘                                 │
│                │                                             │
│   4. Edge Deployment                                         │
│      ┌─────────▼──────────┐                                 │
│      │  TFLite Runtime    │                                 │
│      │  (Raspberry Pi)    │                                 │
│      └────────────────────┘                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Model Conversion (.tflite)

### 3.1 Basic Conversion

```python
#!/usr/bin/env python3
"""Convert TensorFlow model to TFLite"""

import tensorflow as tf

# Load existing Keras model
model = tf.keras.models.load_model('my_model.h5')

# Create converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Convert
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
```

### 3.2 Quantization

```python
#!/usr/bin/env python3
"""Model optimization with quantization"""

import tensorflow as tf
import numpy as np

def load_model():
    return tf.keras.models.load_model('my_model.h5')

def convert_to_tflite(model, quantization='none'):
    """
    Quantization options:
    - 'none': Default (float32)
    - 'dynamic': Dynamic range quantization (weights only)
    - 'float16': Float16 quantization
    - 'int8': Full integer quantization (requires representative dataset)
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization == 'dynamic':
        # Dynamic range quantization (easiest)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    elif quantization == 'float16':
        # Float16 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    elif quantization == 'int8':
        # Full integer quantization (maximum optimization)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # Representative dataset required
        def representative_dataset():
            for _ in range(100):
                yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]

        converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()
    return tflite_model

# Convert and compare sizes
model = load_model()

for quant in ['none', 'dynamic', 'float16']:
    tflite_model = convert_to_tflite(model, quant)
    size_kb = len(tflite_model) / 1024

    with open(f'model_{quant}.tflite', 'wb') as f:
        f.write(tflite_model)

    print(f"{quant}: {size_kb:.2f} KB")
```

### 3.3 Pre-trained Model Conversion

```python
#!/usr/bin/env python3
"""Convert MobileNet to TFLite"""

import tensorflow as tf

# Load MobileNetV2
model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=True
)

# Convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open('mobilenet_v2.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Conversion complete: {len(tflite_model) / (1024*1024):.2f} MB")
```

---

## 4. Inference on Raspberry Pi

### 4.1 TFLite Interpreter Basics

```python
#!/usr/bin/env python3
"""TFLite inference basics"""

import numpy as np

# Use tflite-runtime (Raspberry Pi)
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

class TFLiteModel:
    """TFLite model wrapper"""

    def __init__(self, model_path: str):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Input/output information
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Input shape
        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']

    def get_input_shape(self):
        """Return input shape"""
        return self.input_shape

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Perform inference"""
        # Set input
        # Why astype(self.input_dtype) matters: INT8-quantized models expect int8
        # input. If you pass float32 data to an INT8 model, TFLite does NOT raise
        # an error -- it silently reinterprets the bytes, producing garbage output
        # that looks like valid (but wrong) predictions. Always cast to the model's
        # expected dtype. Check input_details[0]['dtype'] when debugging unexpected
        # classification results.
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            input_data.astype(self.input_dtype)
        )

        # Invoke inference
        self.interpreter.invoke()

        # Get output
        output = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )

        return output

# Usage example
if __name__ == "__main__":
    model = TFLiteModel("model.tflite")
    print(f"Input shape: {model.get_input_shape()}")

    # Dummy input
    input_data = np.random.rand(*model.get_input_shape()).astype(np.float32)

    output = model.predict(input_data)
    print(f"Output shape: {output.shape}")
```

### 4.2 Performance Benchmarking

```python
#!/usr/bin/env python3
"""TFLite inference performance measurement"""

import numpy as np
import time

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

def benchmark_model(model_path: str, num_runs: int = 100):
    """Model performance benchmark"""
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # Warmup
    dummy_input = np.random.rand(*input_shape).astype(input_dtype)
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()

        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])

        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1000 / avg_time

    print(f"=== {model_path} ===")
    print(f"Average inference time: {avg_time:.2f} ms (+/- {std_time:.2f})")
    print(f"FPS: {fps:.1f}")
    print(f"Input shape: {input_shape}")

    return avg_time

# Compare multiple models
if __name__ == "__main__":
    models = [
        "model_none.tflite",
        "model_dynamic.tflite",
        "model_float16.tflite"
    ]

    for model in models:
        try:
            benchmark_model(model)
            print()
        except Exception as e:
            print(f"{model}: Error - {e}")
```

---

## 5. Image Classification Example

### 5.1 ImageNet Classification

```python
#!/usr/bin/env python3
"""TFLite image classification (MobileNet)"""

import numpy as np
from PIL import Image
import time

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

class ImageClassifier:
    """TFLite image classifier"""

    def __init__(self, model_path: str, labels_path: str = None):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Check input size
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]

        # Load labels
        self.labels = []
        if labels_path:
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Image preprocessing"""
        # Resize
        image = image.resize((self.input_width, self.input_height))

        # Convert to NumPy array
        input_data = np.array(image, dtype=np.float32)

        # Why [-1, 1] normalization: MobileNet was trained with pixel values scaled
        # to [-1, 1] using exactly this formula. Using a different normalization
        # (e.g., /255.0 which gives [0, 1]) produces WRONG results silently --
        # the model still outputs class probabilities, but they are meaningless
        # because the input distribution doesn't match what the model learned.
        # Always match the preprocessing to the model's training pipeline.
        input_data = (input_data - 127.5) / 127.5

        # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)

        return input_data

    def classify(self, image_path: str, top_k: int = 5) -> list:
        """Image classification"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_data = self.preprocess(image)

        # Inference
        start = time.perf_counter()

        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            input_data
        )
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )[0]

        inference_time = (time.perf_counter() - start) * 1000

        # Top-K results
        top_indices = output.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
            score = float(output[idx])
            results.append({
                "class_id": int(idx),
                "label": label,
                "score": score
            })

        return {
            "results": results,
            "inference_time_ms": inference_time
        }

# Usage example
if __name__ == "__main__":
    classifier = ImageClassifier(
        model_path="mobilenet_v2.tflite",
        labels_path="imagenet_labels.txt"
    )

    result = classifier.classify("test_image.jpg")

    print(f"Inference time: {result['inference_time_ms']:.2f} ms")
    print("\nClassification results:")
    for r in result['results']:
        print(f"  {r['label']}: {r['score']:.4f}")
```

### 5.2 Real-time Camera Classification

```python
#!/usr/bin/env python3
"""Real-time image classification with Pi Camera"""

import numpy as np
from PIL import Image
import time
import io

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

try:
    from picamera2 import Picamera2
    HAS_CAMERA = True
except ImportError:
    HAS_CAMERA = False
    print("picamera2 not found: simulation mode")

class RealtimeClassifier:
    """Real-time image classifier"""

    def __init__(self, model_path: str, labels_path: str):
        # Load model
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]

        # Labels
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Initialize camera
        if HAS_CAMERA:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self.camera.configure(config)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Frame preprocessing"""
        image = Image.fromarray(frame)
        image = image.resize((self.input_width, self.input_height))

        input_data = np.array(image, dtype=np.float32)
        # Why [-1, 1]: must match MobileNet's training normalization (see 5.1)
        input_data = (input_data - 127.5) / 127.5
        input_data = np.expand_dims(input_data, axis=0)

        return input_data

    def classify_frame(self, frame: np.ndarray) -> dict:
        """Classify single frame"""
        input_data = self.preprocess(frame)

        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            input_data
        )
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )[0]

        top_idx = output.argmax()

        return {
            "label": self.labels[top_idx] if top_idx < len(self.labels) else "unknown",
            "score": float(output[top_idx])
        }

    def run(self, duration: float = 30):
        """Run real-time classification"""
        if not HAS_CAMERA:
            print("Camera not found")
            return

        self.camera.start()
        print(f"Real-time classification started ({duration}s)")

        start_time = time.time()
        frame_count = 0

        try:
            while time.time() - start_time < duration:
                frame = self.camera.capture_array()
                result = self.classify_frame(frame)

                frame_count += 1
                print(f"\r[{frame_count}] {result['label']}: {result['score']:.2f}", end="")

        except KeyboardInterrupt:
            pass
        finally:
            self.camera.stop()
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"\n\nFPS: {fps:.1f}")

if __name__ == "__main__":
    classifier = RealtimeClassifier(
        model_path="mobilenet_v2.tflite",
        labels_path="imagenet_labels.txt"
    )
    classifier.run(duration=60)
```

### 5.3 IoT Integration Example

```python
#!/usr/bin/env python3
"""TFLite + MQTT: Publish image classification results"""

import numpy as np
from PIL import Image
import json
import time
import paho.mqtt.client as mqtt

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

class AIEdgeNode:
    """AI edge node"""

    def __init__(self, model_path: str, labels_path: str,
                 mqtt_broker: str = "localhost"):
        # TFLite model
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        # MQTT client
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect(mqtt_broker, 1883)
        self.mqtt_client.loop_start()

        self.node_id = "edge_ai_01"

    def classify_and_publish(self, image_path: str):
        """Classify and publish results to MQTT"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.input_shape[2], self.input_shape[1]))

        input_data = np.array(image, dtype=np.float32)
        # Why [-1, 1]: must match MobileNet's training normalization (see 5.1)
        input_data = (input_data - 127.5) / 127.5
        input_data = np.expand_dims(input_data, axis=0)

        # Inference
        start = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        inference_time = (time.perf_counter() - start) * 1000

        # Generate results
        top_idx = output.argsort()[-3:][::-1]
        predictions = [
            {
                "label": self.labels[idx] if idx < len(self.labels) else "unknown",
                "score": float(output[idx])
            }
            for idx in top_idx
        ]

        result = {
            "node_id": self.node_id,
            "image": image_path,
            "predictions": predictions,
            "inference_time_ms": round(inference_time, 2),
            "timestamp": time.time()
        }

        # Publish to MQTT
        topic = f"edge/{self.node_id}/classification"
        self.mqtt_client.publish(topic, json.dumps(result))

        print(f"Published: {topic}")
        print(f"  Top-1: {predictions[0]['label']} ({predictions[0]['score']:.2f})")

        return result

    def shutdown(self):
        """Cleanup"""
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()

# Usage example
if __name__ == "__main__":
    node = AIEdgeNode(
        model_path="mobilenet_v2.tflite",
        labels_path="imagenet_labels.txt",
        mqtt_broker="localhost"
    )

    try:
        # Classify test image
        node.classify_and_publish("test_image.jpg")
    finally:
        node.shutdown()
```

---

## Practice Exercises

### Exercise 1: Model Conversion
1. Convert a Keras model to TFLite
2. Apply dynamic quantization and compare sizes

### Exercise 2: Performance Optimization
1. Compare performance of FP32, FP16, INT8 versions of the same model
2. Measure FPS on Raspberry Pi

### Exercise 3: Real-time Classification
1. Implement real-time image classification with Pi Camera
2. Publish classification results to MQTT

---

**Previous**: [HTTP/REST for IoT](./07_HTTP_REST_for_IoT.md) | **Next**: [Edge AI - ONNX Runtime](./09_Edge_AI_ONNX.md)
