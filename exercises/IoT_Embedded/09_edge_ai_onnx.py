"""
Exercises for Lesson 09: Edge AI - ONNX Runtime
Topic: IoT_Embedded

Solutions to practice problems from the lesson.
Simulates ONNX model conversion, runtime comparison with TFLite,
and real-time object detection pipeline.

On a real system:
    pip install onnxruntime
    pip install onnx onnxsim
    pip install torch torchvision  # for PyTorch model export
"""

import time
import random
import json
from datetime import datetime


# ---------------------------------------------------------------------------
# Simulated ONNX operations
# ---------------------------------------------------------------------------

class SimulatedPyTorchModel:
    """Simulate a PyTorch image classification model.

    In real PyTorch:
        import torch
        import torchvision.models as models
        model = models.resnet18(pretrained=True)
        model.eval()
    """

    def __init__(self, name, input_shape, num_classes, num_params):
        self.name = name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_params = num_params
        self.size_bytes = num_params * 4  # FP32: 4 bytes per param


class SimulatedONNXConverter:
    """Simulate PyTorch -> ONNX conversion and model simplification.

    The ONNX (Open Neural Network Exchange) format is an open standard
    for representing ML models. It decouples training framework (PyTorch,
    TensorFlow) from inference runtime (ONNX Runtime, TensorRT).

    Conversion steps:
    1. torch.onnx.export() -- traces the model and saves computation graph
    2. onnx.checker.check_model() -- validates the exported model
    3. onnxsim.simplify() -- fuses ops, removes redundancies (optional)
    """

    @staticmethod
    def export(model, opset_version=13):
        """Export PyTorch model to ONNX format.

        Real code:
            dummy_input = torch.randn(*model.input_shape)
            torch.onnx.export(
                model, dummy_input, 'model.onnx',
                opset_version=opset_version,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}}
            )
        """
        # ONNX is slightly larger than raw PyTorch due to graph metadata
        onnx_size = int(model.size_bytes * 1.05)
        return {
            "name": f"{model.name}.onnx",
            "size_bytes": onnx_size,
            "size_mb": onnx_size / 1024 / 1024,
            "opset_version": opset_version,
            "num_nodes": model.num_params // 1000,  # Approximate
            "valid": True,
        }

    @staticmethod
    def simplify(onnx_model):
        """Simplify ONNX model by fusing operators and removing redundancy.

        onnxsim typically reduces model size by 5-15% and can improve
        inference speed by eliminating unnecessary intermediate operations.

        Real code:
            import onnxsim
            model_simplified, check = onnxsim.simplify(onnx.load('model.onnx'))
            onnx.save(model_simplified, 'model_simplified.onnx')
        """
        simplified_size = int(onnx_model["size_bytes"] * 0.90)
        simplified_nodes = int(onnx_model["num_nodes"] * 0.85)
        return {
            "name": onnx_model["name"].replace(".onnx", "_simplified.onnx"),
            "size_bytes": simplified_size,
            "size_mb": simplified_size / 1024 / 1024,
            "opset_version": onnx_model["opset_version"],
            "num_nodes": simplified_nodes,
            "valid": True,
            "nodes_removed": onnx_model["num_nodes"] - simplified_nodes,
            "size_reduction_pct": (1 - simplified_size / onnx_model["size_bytes"]) * 100,
        }

    @staticmethod
    def validate(onnx_model):
        """Validate ONNX model structure.

        Real code:
            import onnx
            model = onnx.load('model.onnx')
            onnx.checker.check_model(model)
        """
        return onnx_model["valid"]


class SimulatedONNXRuntime:
    """Simulate ONNX Runtime inference session.

    ONNX Runtime optimizes inference using:
    - Graph optimization (constant folding, operator fusion)
    - Hardware-specific execution providers (CPU, CUDA, TensorRT, OpenVINO)
    - Quantization-aware execution

    On Raspberry Pi, ONNX Runtime uses the CPU execution provider.
    For NVIDIA Jetson, the CUDA or TensorRT providers offer GPU acceleration.
    """

    def __init__(self, model_info, batch_size=1):
        self.model_info = model_info
        self.batch_size = batch_size
        # Base latency per image (ms) on Raspberry Pi 4
        self._base_latency_ms = 85.0

    def run(self, input_data=None):
        """Run inference.

        Real code:
            import onnxruntime as ort
            session = ort.InferenceSession('model.onnx')
            input_name = session.get_inputs()[0].name
            result = session.run(None, {input_name: input_data})
        """
        # Batch processing: per-image latency decreases with larger batches
        # due to better memory locality and vectorization
        batch_factor = 1.0 - 0.1 * min(self.batch_size - 1, 5)  # Up to 50% savings
        latency = self._base_latency_ms * self.batch_size * batch_factor
        latency *= random.uniform(0.85, 1.15)  # Jitter

        # Simulate detection results
        detections = []
        for _ in range(self.batch_size):
            num_objects = random.randint(1, 5)
            frame_detections = []
            for _ in range(num_objects):
                frame_detections.append({
                    "class": random.choice([
                        "person", "car", "bicycle", "dog", "cat",
                        "truck", "bus", "motorcycle", "traffic light",
                    ]),
                    "confidence": round(random.uniform(0.5, 0.98), 3),
                    "bbox": [
                        random.randint(0, 300),   # x1
                        random.randint(0, 300),   # y1
                        random.randint(300, 640),  # x2
                        random.randint(300, 480),  # y2
                    ],
                })
            detections.append(frame_detections)

        time.sleep(latency * 0.001)  # Shortened for demo

        return detections, latency


class SimulatedTFLiteRuntime:
    """Simulate TFLite runtime for comparison benchmarks."""

    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self._base_latency_ms = 95.0  # Slightly slower than ONNX on average

    def run(self, input_data=None):
        latency = self._base_latency_ms * self.batch_size
        latency *= random.uniform(0.85, 1.15)
        time.sleep(latency * 0.001)
        return latency


# ---------------------------------------------------------------------------
# Exercise Solutions
# ---------------------------------------------------------------------------

# === Problem 1: Model Conversion ===
# Problem: Convert a PyTorch image classification model to ONNX.
# Validate and simplify the converted model.

def problem_1():
    """Solution: PyTorch to ONNX conversion with validation and simplification."""

    print("  PyTorch to ONNX Model Conversion\n")

    # Create a simulated ResNet-18 model
    model = SimulatedPyTorchModel(
        name="ResNet18",
        input_shape=(1, 3, 224, 224),  # NCHW format
        num_classes=1000,
        num_params=11_700_000,  # ~11.7M parameters
    )

    print(f"    --- Original PyTorch Model ---")
    print(f"    Name: {model.name}")
    print(f"    Input: {model.input_shape}")
    print(f"    Parameters: {model.num_params:,}")
    print(f"    Size: {model.size_bytes / 1024 / 1024:.1f} MB\n")

    # Step 1: Export to ONNX
    print(f"    --- Step 1: Export to ONNX ---")
    onnx_model = SimulatedONNXConverter.export(model, opset_version=13)
    print(f"    Output: {onnx_model['name']}")
    print(f"    Size: {onnx_model['size_mb']:.1f} MB")
    print(f"    Opset version: {onnx_model['opset_version']}")
    print(f"    Graph nodes: {onnx_model['num_nodes']}")

    # Step 2: Validate
    print(f"\n    --- Step 2: Validate ---")
    is_valid = SimulatedONNXConverter.validate(onnx_model)
    print(f"    Model valid: {is_valid}")

    # Step 3: Simplify
    print(f"\n    --- Step 3: Simplify ---")
    simplified = SimulatedONNXConverter.simplify(onnx_model)
    print(f"    Output: {simplified['name']}")
    print(f"    Size: {simplified['size_mb']:.1f} MB "
          f"({simplified['size_reduction_pct']:.1f}% reduction)")
    print(f"    Nodes: {simplified['num_nodes']} "
          f"({simplified['nodes_removed']} removed)")

    # Comparison summary
    print(f"\n    --- Conversion Summary ---")
    print(f"    {'Stage':<25} {'Size (MB)':>10} {'Nodes':>8}")
    print(f"    {'-'*25} {'-'*10} {'-'*8}")
    print(f"    {'PyTorch (FP32)':<25} {model.size_bytes/1024/1024:>8.1f} {'N/A':>8}")
    print(f"    {'ONNX (exported)':<25} {onnx_model['size_mb']:>8.1f} {onnx_model['num_nodes']:>8}")
    print(f"    {'ONNX (simplified)':<25} {simplified['size_mb']:>8.1f} {simplified['num_nodes']:>8}")

    # Reference code
    print("""
    --- Reference Code (real PyTorch + ONNX) ---

    import torch
    import torchvision.models as models
    import onnx
    import onnxsim

    # Step 1: Export
    model = models.resnet18(pretrained=True)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, 'resnet18.onnx',
                      opset_version=13,
                      input_names=['input'],
                      output_names=['output'])

    # Step 2: Validate
    onnx_model = onnx.load('resnet18.onnx')
    onnx.checker.check_model(onnx_model)

    # Step 3: Simplify
    simplified, ok = onnxsim.simplify(onnx_model)
    onnx.save(simplified, 'resnet18_simplified.onnx')
    """)


# === Problem 2: Performance Comparison ===
# Problem: Compare TFLite vs ONNX Runtime inference speed.
# Measure throughput by batch size.

def problem_2():
    """Solution: TFLite vs ONNX Runtime performance benchmark."""

    import statistics

    print("  TFLite vs ONNX Runtime Performance Comparison\n")

    batch_sizes = [1, 2, 4, 8]
    num_runs = 20  # Per batch size

    results = {"onnx": {}, "tflite": {}}

    for batch_size in batch_sizes:
        onnx_rt = SimulatedONNXRuntime(
            model_info={"name": "yolov5s.onnx"}, batch_size=batch_size
        )
        tflite_rt = SimulatedTFLiteRuntime(batch_size=batch_size)

        onnx_latencies = []
        tflite_latencies = []

        for _ in range(num_runs):
            _, onnx_lat = onnx_rt.run()
            onnx_latencies.append(onnx_lat)

            tflite_lat = tflite_rt.run()
            tflite_latencies.append(tflite_lat)

        results["onnx"][batch_size] = {
            "mean": statistics.mean(onnx_latencies),
            "std": statistics.stdev(onnx_latencies) if len(onnx_latencies) > 1 else 0,
            "throughput": batch_size * 1000 / statistics.mean(onnx_latencies),
        }
        results["tflite"][batch_size] = {
            "mean": statistics.mean(tflite_latencies),
            "std": statistics.stdev(tflite_latencies) if len(tflite_latencies) > 1 else 0,
            "throughput": batch_size * 1000 / statistics.mean(tflite_latencies),
        }

    # Print comparison table
    print(f"    {'Batch':<7} {'ONNX RT (ms)':>14} {'TFLite (ms)':>14} "
          f"{'ONNX Tput':>11} {'TFLite Tput':>13}")
    print(f"    {'-'*7} {'-'*14} {'-'*14} {'-'*11} {'-'*13}")

    for bs in batch_sizes:
        o = results["onnx"][bs]
        t = results["tflite"][bs]
        print(f"    {bs:<7} {o['mean']:>10.1f} +/- {o['std']:>3.0f} "
              f"{t['mean']:>10.1f} +/- {t['std']:>3.0f} "
              f"{o['throughput']:>9.1f}/s {t['throughput']:>11.1f}/s")

    # Throughput graph (text-based)
    print(f"\n    Throughput comparison (images/second):")
    for bs in batch_sizes:
        o_tput = results["onnx"][bs]["throughput"]
        t_tput = results["tflite"][bs]["throughput"]
        o_bar = "#" * int(o_tput / 2)
        t_bar = "#" * int(t_tput / 2)
        print(f"    Batch {bs}:")
        print(f"      ONNX:   [{o_bar:<30}] {o_tput:.1f}/s")
        print(f"      TFLite: [{t_bar:<30}] {t_tput:.1f}/s")

    print("""
    Key observations:
    - ONNX Runtime typically shows slightly lower latency than TFLite
      on ARM CPUs due to its aggressive graph optimization passes
    - Throughput improves with batch size for both runtimes because
      larger batches amortize memory access overhead
    - For real-time single-frame inference (batch=1), the difference
      is often <10% -- choose based on your model's source framework
    - TFLite is lighter-weight (smaller binary, less memory); ONNX
      Runtime offers more operator coverage and better PyTorch compatibility
    """)


# === Problem 3: Real-time Detection ===
# Problem: Real-time object detection using YOLO model with MQTT output.

def problem_3():
    """Solution: Real-time YOLO object detection with MQTT publishing."""

    print("  Real-time Object Detection (YOLO + ONNX Runtime)\n")

    onnx_runtime = SimulatedONNXRuntime(
        model_info={"name": "yolov5s.onnx"},
        batch_size=1,
    )

    device_id = "cam_entrance"
    mqtt_log = []

    def mqtt_publish(topic, payload):
        """Simulate MQTT publish for detection results.

        Real code:
            import paho.mqtt.client as mqtt
            client = mqtt.Client()
            client.connect("localhost", 1883)
            client.publish(topic, json.dumps(payload), qos=1)
        """
        mqtt_log.append({"topic": topic, "payload": payload})

    print(f"    Device: {device_id}")
    print(f"    Model: YOLOv5s (ONNX)")
    print(f"    MQTT topic: detection/{device_id}/objects\n")

    # Process 8 frames
    print(f"    {'Frame':>6} {'Objects':>8} {'Latency':>10} {'FPS':>6}  Top Detections")
    print(f"    {'-'*6} {'-'*8} {'-'*10} {'-'*6}  {'-'*30}")

    for frame_num in range(1, 9):
        detections, latency_ms = onnx_runtime.run()
        frame_dets = detections[0]  # Batch size 1
        fps = 1000.0 / latency_ms

        # Format top detections
        top_dets = sorted(frame_dets, key=lambda d: d["confidence"], reverse=True)[:3]
        det_str = ", ".join(f"{d['class']}({d['confidence']:.0%})" for d in top_dets)

        print(f"    {frame_num:>6} {len(frame_dets):>8} "
              f"{latency_ms:>8.1f}ms {fps:>5.1f}  {det_str}")

        # Publish to MQTT
        result = {
            "device_id": device_id,
            "frame": frame_num,
            "timestamp": datetime.now().isoformat(),
            "latency_ms": round(latency_ms, 1),
            "num_objects": len(frame_dets),
            "detections": [
                {
                    "class": d["class"],
                    "confidence": d["confidence"],
                    "bbox": d["bbox"],
                }
                for d in frame_dets
            ],
        }
        mqtt_publish(f"detection/{device_id}/objects", result)

        time.sleep(0.1)

    # Show sample MQTT message
    print(f"\n    MQTT messages published: {len(mqtt_log)}")
    print(f"\n    Sample MQTT payload:")
    sample = mqtt_log[-1]["payload"]
    print(f"    Topic: detection/{device_id}/objects")
    print(f"    {json.dumps(sample, indent=6)}")

    print("""
    Production pipeline reference:

        import onnxruntime as ort
        import cv2
        import paho.mqtt.client as mqtt
        import numpy as np

        session = ort.InferenceSession('yolov5s.onnx')
        cap = cv2.VideoCapture(0)  # Pi Camera

        mqtt_client = mqtt.Client()
        mqtt_client.connect("localhost", 1883)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess: resize, normalize, NCHW format
            input_img = cv2.resize(frame, (640, 640))
            input_img = input_img.astype(np.float32) / 255.0
            input_img = np.transpose(input_img, (2, 0, 1))
            input_img = np.expand_dims(input_img, 0)

            # Inference
            outputs = session.run(None, {'images': input_img})

            # Post-process (NMS, confidence thresholding)
            detections = postprocess(outputs, conf_threshold=0.5, iou_threshold=0.45)

            # Publish
            mqtt_client.publish(
                f'detection/{device_id}/objects',
                json.dumps(detections),
                qos=1
            )
    """)


# === Run All Exercises ===
if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 09: Edge AI - ONNX Runtime - Exercise Solutions")
    print("=" * 70)

    print("\n\n>>> Problem 1: Model Conversion")
    print("-" * 50)
    problem_1()

    print("\n\n>>> Problem 2: Performance Comparison")
    print("-" * 50)
    problem_2()

    print("\n\n>>> Problem 3: Real-time Detection")
    print("-" * 50)
    problem_3()
