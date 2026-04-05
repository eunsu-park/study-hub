# 09. Edge AI - ONNX Runtime

**Previous**: [Edge AI - TensorFlow Lite](./08_Edge_AI_TFLite.md) | **Next**: [Home Automation Project](./10_Home_Automation_Project.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the ONNX open format and its role in cross-framework model portability
2. Compare ONNX Runtime with TensorFlow Lite for edge deployment
3. Convert PyTorch and TensorFlow models to the ONNX format
4. Validate, simplify, and quantize ONNX models for edge performance
5. Run inference using ONNX Runtime on Raspberry Pi
6. Implement object detection with a YOLO ONNX model

---

Not every machine learning model is trained in TensorFlow. If your team uses PyTorch, scikit-learn, or another framework, ONNX provides a universal interchange format so you can still deploy models to edge devices. ONNX Runtime gives you a second high-performance inference engine alongside TFLite, ensuring you are never locked into a single framework for your IoT AI workloads.

---

## 1. ONNX Overview

### 1.1 What is ONNX?

**ONNX (Open Neural Network Exchange)** is an open format that provides model compatibility across various ML frameworks.

```
┌─────────────────────────────────────────────────────────────┐
│                    ONNX Ecosystem                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Training Frameworks                  Inference Engines    │
│   ┌──────────┐                                              │
│   │ PyTorch  │────┐                                         │
│   └──────────┘    │                                         │
│   ┌──────────┐    │        ┌──────────┐    ┌──────────────┐ │
│   │TensorFlow│────┼──────▶│  ONNX    │───▶│ONNX Runtime  │ │
│   └──────────┘    │        │ (.onnx)  │    │(cross-platform)│ │
│   ┌──────────┐    │        └──────────┘    └──────────────┘ │
│   │  Keras   │────┤                               │         │
│   └──────────┘    │                               ▼         │
│   ┌──────────┐    │                        ┌──────────────┐ │
│   │ Sklearn  │────┘                        │ Deployment   │ │
│   └──────────┘                             │ • Raspberry Pi│ │
│                                            │ • Windows    │ │
│                                            │ • Android    │ │
│                                            │ • iOS        │ │
│                                            └──────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 ONNX vs TFLite

| Feature | ONNX | TFLite |
|---------|------|--------|
| **Developer** | Microsoft + Partners | Google |
| **Framework Support** | PyTorch, TF, Sklearn, etc. | TensorFlow/Keras |
| **Format** | .onnx (Protobuf) | .tflite (FlatBuffer) |
| **Optimization** | ONNX Runtime | TF Lite Interpreter |
| **Quantization** | Supported | Supported |
| **Hardware** | CPU, GPU, NPU | CPU, GPU, Edge TPU |

### 1.3 ONNX Runtime Features

```python
# ONNX Runtime features
onnx_runtime_features = {
    "cross_platform": "Windows, Linux, macOS, Android, iOS",
    "hardware_acceleration": "CPU, CUDA, TensorRT, DirectML, OpenVINO",
    "multi_language": "Python, C++, C#, Java, JavaScript",
    "optimization": "Graph optimization, quantization, operator fusion",
    "flexibility": "Run models converted from various frameworks"
}
```

---

## 2. ONNX Runtime Installation

### 2.1 Raspberry Pi Installation

```bash
# Basic ONNX Runtime (CPU)
pip install onnxruntime

# ARM64 optimized version (Raspberry Pi OS 64bit)
# pip install onnxruntime --extra-index-url https://aiinfra.pkgs.visualstudio.com/...

# Additional packages
pip install numpy pillow onnx

# For model conversion (on PC)
pip install tf2onnx torch onnx-simplifier
```

### 2.2 Installation Verification

```python
#!/usr/bin/env python3
"""ONNX Runtime installation verification"""

import onnxruntime as ort
import numpy as np

# Version check
print(f"ONNX Runtime version: {ort.__version__}")

# Available providers (execution backends)
providers = ort.get_available_providers()
print(f"Available providers: {providers}")

# Simple test
# Run dummy model
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

print("\nONNX Runtime working properly!")
```

---

## 3. Model Conversion

### 3.1 PyTorch to ONNX

```python
#!/usr/bin/env python3
"""Convert PyTorch model to ONNX"""

import torch
import torch.nn as nn

# Example model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def export_to_onnx(model, output_path: str, input_shape: tuple):
    """Export PyTorch model to ONNX"""
    model.eval()

    # Dummy input
    dummy_input = torch.randn(*input_shape)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=13
    )

    print(f"ONNX model saved: {output_path}")

# Usage example
if __name__ == "__main__":
    model = SimpleNet()
    export_to_onnx(model, "simple_net.onnx", (1, 10))
```

### 3.2 TensorFlow/Keras to ONNX

```bash
# Using tf2onnx (command line)
python -m tf2onnx.convert \
    --saved-model tensorflow_model/ \
    --output model.onnx \
    --opset 13
```

```python
#!/usr/bin/env python3
"""Convert TensorFlow/Keras model to ONNX"""

import tensorflow as tf
import tf2onnx
import onnx

def keras_to_onnx(model_path: str, output_path: str):
    """Convert Keras model to ONNX"""
    # Load Keras model
    model = tf.keras.models.load_model(model_path)

    # Convert to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        opset=13,
        output_path=output_path
    )

    print(f"Conversion complete: {output_path}")

# Usage example
keras_to_onnx("my_model.h5", "my_model.onnx")
```

### 3.3 Model Validation and Simplification

```python
#!/usr/bin/env python3
"""ONNX model validation and simplification"""

import onnx
from onnxsim import simplify

def validate_and_simplify(model_path: str, output_path: str = None):
    """Validate and optimize ONNX model"""
    # Load model
    model = onnx.load(model_path)

    # Validate
    try:
        onnx.checker.check_model(model)
        print("Model validation passed")
    except Exception as e:
        print(f"Validation failed: {e}")
        return

    # Model information
    print(f"\nModel information:")
    print(f"  IR version: {model.ir_version}")
    print(f"  Opset: {model.opset_import[0].version}")
    print(f"  Graph name: {model.graph.name}")

    # Input/output information
    print(f"\nInputs:")
    for input in model.graph.input:
        print(f"  {input.name}: {input.type}")

    print(f"\nOutputs:")
    for output in model.graph.output:
        print(f"  {output.name}: {output.type}")

    # Simplification (remove redundant operations, optimize graph)
    simplified_model, check = simplify(model)

    if check:
        print("\nSimplification successful")

        if output_path:
            onnx.save(simplified_model, output_path)
            print(f"Saved: {output_path}")

            # Size comparison
            import os
            orig_size = os.path.getsize(model_path) / 1024
            new_size = os.path.getsize(output_path) / 1024
            print(f"\nSize: {orig_size:.1f}KB -> {new_size:.1f}KB")

        return simplified_model
    else:
        print("Simplification failed")
        return model

# Usage example
if __name__ == "__main__":
    validate_and_simplify("model.onnx", "model_simplified.onnx")
```

---

## 4. Inference Execution

### 4.1 Basic Inference

```python
#!/usr/bin/env python3
"""ONNX Runtime basic inference"""

import onnxruntime as ort
import numpy as np

class ONNXModel:
    """ONNX model wrapper"""

    def __init__(self, model_path: str, providers: list = None):
        if providers is None:
            providers = ['CPUExecutionProvider']

        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        # Create session
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        # Input/output information
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

    def get_input_shape(self):
        return self.input_shape

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference"""
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        return outputs[0]

# Usage example
if __name__ == "__main__":
    model = ONNXModel("model.onnx")

    print(f"Input shape: {model.get_input_shape()}")

    # Dummy input
    input_data = np.random.randn(1, 10).astype(np.float32)
    output = model.predict(input_data)

    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
```

### 4.2 Batch Inference

```python
#!/usr/bin/env python3
"""ONNX Runtime batch inference"""

import onnxruntime as ort
import numpy as np
import time

def batch_inference(model_path: str, data: np.ndarray,
                    batch_size: int = 32) -> np.ndarray:
    """Perform batch inference"""
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    results = []
    num_samples = len(data)

    for i in range(0, num_samples, batch_size):
        batch = data[i:i + batch_size]
        output = session.run([output_name], {input_name: batch})[0]
        results.append(output)

    return np.concatenate(results, axis=0)

# Performance measurement
def benchmark_batch_sizes(model_path: str, input_shape: tuple):
    """Compare performance by batch size"""
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name

    total_samples = 1000

    for batch_size in [1, 8, 16, 32, 64]:
        data = np.random.randn(total_samples, *input_shape[1:]).astype(np.float32)

        start = time.perf_counter()

        for i in range(0, total_samples, batch_size):
            batch = data[i:i + batch_size]
            _ = session.run(None, {input_name: batch})

        elapsed = time.perf_counter() - start
        throughput = total_samples / elapsed

        print(f"Batch size {batch_size:2d}: {throughput:.1f} samples/sec")
```

### 4.3 Quantized Inference

```python
#!/usr/bin/env python3
"""ONNX Runtime quantization"""

import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(model_path: str, output_path: str):
    """Apply dynamic quantization"""
    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QInt8
    )

    import os
    orig_size = os.path.getsize(model_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)

    print(f"Original: {orig_size:.2f} MB")
    print(f"Quantized: {new_size:.2f} MB")
    print(f"Compression ratio: {orig_size / new_size:.1f}x")

# Usage example
quantize_model("model.onnx", "model_quantized.onnx")
```

---

## 5. Object Detection Example

### 5.1 Using YOLO ONNX Model

```python
#!/usr/bin/env python3
"""YOLOv5 ONNX object detection"""

import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2

class YOLODetector:
    """YOLOv5 ONNX object detector"""

    # COCO classes
    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(self, model_path: str, conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Input information
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_shape = input_info.shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def preprocess(self, image: np.ndarray) -> tuple:
        """Image preprocessing"""
        orig_height, orig_width = image.shape[:2]

        # Resize
        resized = cv2.resize(image, (self.input_width, self.input_height))

        # BGR to RGB, HWC to CHW
        input_data = resized[:, :, ::-1].transpose(2, 0, 1)

        # Normalize (0-1)
        input_data = input_data.astype(np.float32) / 255.0

        # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)

        # Save scale ratio
        scale = (orig_width / self.input_width, orig_height / self.input_height)

        return input_data, scale

    def postprocess(self, output: np.ndarray, scale: tuple) -> list:
        """Output post-processing"""
        predictions = output[0]

        boxes = []
        scores = []
        class_ids = []

        for pred in predictions:
            confidence = pred[4]

            if confidence > self.conf_threshold:
                class_probs = pred[5:]
                class_id = np.argmax(class_probs)
                class_score = class_probs[class_id]

                if class_score > self.conf_threshold:
                    # Box coordinates (center_x, center_y, width, height)
                    cx, cy, w, h = pred[:4]

                    # Convert to original scale
                    x1 = int((cx - w / 2) * scale[0])
                    y1 = int((cy - h / 2) * scale[1])
                    x2 = int((cx + w / 2) * scale[0])
                    y2 = int((cy + h / 2) * scale[1])

                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(confidence * class_score))
                    class_ids.append(int(class_id))

        # NMS (Non-Maximum Suppression)
        if boxes:
            indices = cv2.dnn.NMSBoxes(
                boxes, scores, self.conf_threshold, self.iou_threshold
            )

            results = []
            for i in indices:
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                results.append({
                    'box': boxes[idx],
                    'score': scores[idx],
                    'class_id': class_ids[idx],
                    'class_name': self.CLASSES[class_ids[idx]]
                })

            return results

        return []

    def detect(self, image: np.ndarray) -> list:
        """Object detection"""
        input_data, scale = self.preprocess(image)

        outputs = self.session.run(None, {self.input_name: input_data})

        detections = self.postprocess(outputs[0], scale)

        return detections

    def draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """Visualize detection results"""
        result = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = f"{det['class_name']}: {det['score']:.2f}"

            # Box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return result

# Usage example
if __name__ == "__main__":
    detector = YOLODetector("yolov5s.onnx")

    # Load image
    image = cv2.imread("test_image.jpg")

    # Detect
    detections = detector.detect(image)

    print(f"Detected objects: {len(detections)}")
    for det in detections:
        print(f"  {det['class_name']}: {det['score']:.2f}")

    # Save result
    result_image = detector.draw_detections(image, detections)
    cv2.imwrite("result.jpg", result_image)
```

### 5.2 Real-time Object Detection

```python
#!/usr/bin/env python3
"""Real-time object detection (Pi Camera + ONNX)"""

import numpy as np
import cv2
import time

try:
    from picamera2 import Picamera2
    HAS_CAMERA = True
except ImportError:
    HAS_CAMERA = False

# YOLODetector class is the same as above

class RealtimeDetector:
    """Real-time object detector"""

    def __init__(self, model_path: str):
        self.detector = YOLODetector(model_path)

        if HAS_CAMERA:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self.camera.configure(config)

    def run(self, duration: float = 60, display: bool = False):
        """Run real-time detection"""
        if not HAS_CAMERA:
            print("No camera available")
            return

        self.camera.start()
        print(f"Real-time detection started ({duration} seconds)")

        start_time = time.time()
        frame_count = 0
        fps_time = time.time()

        try:
            while time.time() - start_time < duration:
                # Capture frame
                frame = self.camera.capture_array()

                # Convert to BGR (OpenCV format)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Detect
                detections = self.detector.detect(frame_bgr)

                frame_count += 1

                # Calculate FPS
                if frame_count % 10 == 0:
                    elapsed = time.time() - fps_time
                    fps = 10 / elapsed
                    fps_time = time.time()

                    print(f"\rFPS: {fps:.1f}, Detections: {len(detections)}", end="")

                    for det in detections:
                        print(f" | {det['class_name']}", end="")

                # Display (optional)
                if display:
                    result = self.detector.draw_detections(frame_bgr, detections)
                    cv2.imshow("Detection", result)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            pass
        finally:
            self.camera.stop()
            if display:
                cv2.destroyAllWindows()

            total_time = time.time() - start_time
            avg_fps = frame_count / total_time
            print(f"\n\nAverage FPS: {avg_fps:.1f}")

if __name__ == "__main__":
    detector = RealtimeDetector("yolov5s.onnx")
    detector.run(duration=30, display=False)
```

### 5.3 Publishing Detection Results via MQTT

```python
#!/usr/bin/env python3
"""Publish object detection results via MQTT"""

import paho.mqtt.client as mqtt
import json
import time

class DetectionPublisher:
    """Detection result MQTT publisher"""

    def __init__(self, model_path: str, mqtt_broker: str = "localhost"):
        self.detector = YOLODetector(model_path)

        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect(mqtt_broker, 1883)
        self.mqtt_client.loop_start()

        self.node_id = "detector_01"

    def process_and_publish(self, image_path: str):
        """Process image and publish results"""
        import cv2

        image = cv2.imread(image_path)
        if image is None:
            print(f"Image load failed: {image_path}")
            return

        # Detect
        start = time.perf_counter()
        detections = self.detector.detect(image)
        inference_time = (time.perf_counter() - start) * 1000

        # Create result
        result = {
            "node_id": self.node_id,
            "image": image_path,
            "detections": [
                {
                    "class": det['class_name'],
                    "score": round(det['score'], 3),
                    "box": det['box']
                }
                for det in detections
            ],
            "count": len(detections),
            "inference_time_ms": round(inference_time, 2),
            "timestamp": time.time()
        }

        # MQTT publish
        topic = f"edge/{self.node_id}/detection"
        self.mqtt_client.publish(topic, json.dumps(result))

        print(f"Published: {topic}")
        print(f"  Detections: {len(detections)}, Time: {inference_time:.1f}ms")

        return result

    def shutdown(self):
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()

if __name__ == "__main__":
    publisher = DetectionPublisher("yolov5s.onnx")

    try:
        publisher.process_and_publish("test_image.jpg")
    finally:
        publisher.shutdown()
```

---

## Practice Problems

### Problem 1: Model Conversion
1. Convert a PyTorch image classification model to ONNX.
2. Validate and simplify the converted model.

### Problem 2: Performance Comparison
1. Compare inference speed between TFLite and ONNX Runtime.
2. Measure throughput by batch size.

### Problem 3: Real-time Detection
1. Implement real-time object detection using a YOLO model.
2. Publish detection results via MQTT.

---

**Previous**: [Edge AI - TensorFlow Lite](./08_Edge_AI_TFLite.md) | **Next**: [Home Automation Project](./10_Home_Automation_Project.md)
