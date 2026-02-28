"""
Exercises for Lesson 08: Edge AI - TensorFlow Lite
Topic: IoT_Embedded

Solutions to practice problems from the lesson.
Simulates TFLite model conversion, quantization comparison, and
real-time classification pipeline without requiring actual TensorFlow
or Raspberry Pi hardware.

On a real Raspberry Pi:
    pip install tflite-runtime
    pip install tensorflow  # for model conversion
    pip install opencv-python  # for camera capture
"""

import time
import random
import json
import os
from datetime import datetime


# ---------------------------------------------------------------------------
# Simulated TFLite operations
# ---------------------------------------------------------------------------

class SimulatedKerasModel:
    """Simulate a Keras model for conversion exercises.

    In real TensorFlow:
        model = tf.keras.models.load_model('my_model.h5')
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    """

    def __init__(self, name, input_shape, num_classes, num_params):
        self.name = name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_params = num_params
        # Estimate FP32 model size: 4 bytes per parameter + overhead
        self.size_bytes = num_params * 4 + 1024

    def summary(self):
        print(f"    Model: {self.name}")
        print(f"    Input shape: {self.input_shape}")
        print(f"    Output classes: {self.num_classes}")
        print(f"    Parameters: {self.num_params:,}")
        print(f"    Size (FP32): {self.size_bytes / 1024 / 1024:.1f} MB")


class SimulatedTFLiteConverter:
    """Simulate TFLite model conversion and quantization.

    Quantization reduces model size and increases inference speed by
    converting 32-bit floats to lower precision formats:
    - Dynamic range: weights to INT8, activations stay FP32 at runtime
    - FP16: all values to 16-bit float (2x compression, minimal accuracy loss)
    - Full INT8: everything to 8-bit integer (4x compression, needs calibration data)
    """

    @staticmethod
    def convert(model, quantization=None):
        """Convert model to TFLite format with optional quantization.

        Returns (simulated_model_bytes, size_bytes, estimated_accuracy_delta).
        """
        base_size = model.size_bytes

        if quantization is None:
            # No quantization: FP32
            size = base_size
            accuracy_delta = 0.0
            label = "FP32 (no quantization)"
        elif quantization == "dynamic":
            # Dynamic range: ~3-4x smaller
            size = int(base_size * 0.28)
            accuracy_delta = -0.5  # Typical: < 1% accuracy loss
            label = "Dynamic range quantization"
        elif quantization == "fp16":
            # FP16: ~2x smaller
            size = int(base_size * 0.50)
            accuracy_delta = -0.2
            label = "FP16 quantization"
        elif quantization == "int8":
            # Full INT8: ~4x smaller
            size = int(base_size * 0.25)
            accuracy_delta = -1.0  # Can lose more accuracy without good calibration
            label = "Full INT8 quantization"
        else:
            raise ValueError(f"Unknown quantization: {quantization}")

        return {
            "label": label,
            "quantization": quantization,
            "size_bytes": size,
            "size_mb": size / 1024 / 1024,
            "compression_ratio": base_size / size,
            "accuracy_delta_pct": accuracy_delta,
        }


class SimulatedTFLiteInterpreter:
    """Simulate TFLite runtime interpreter for inference benchmarking.

    On a real Raspberry Pi:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path='model.tflite')
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
    """

    IMAGENET_CLASSES = [
        "cat", "dog", "car", "bicycle", "airplane", "boat", "bus", "train",
        "person", "bird", "horse", "sheep", "cow", "bottle", "chair",
        "dining table", "potted plant", "sofa", "tv monitor", "laptop",
    ]

    def __init__(self, model_info, model_name="MobileNetV2"):
        self.model_info = model_info
        self.model_name = model_name
        # Simulate inference time based on quantization type
        # On a Pi 4: FP32 ~100ms, FP16 ~70ms, INT8 ~30ms per frame
        self._base_latency = {
            None: 0.100,
            "dynamic": 0.045,
            "fp16": 0.070,
            "int8": 0.030,
        }

    def invoke(self, input_data=None):
        """Run inference on input data (simulated).

        Returns (class_label, confidence, latency_ms).
        """
        quant = self.model_info.get("quantization")
        base = self._base_latency.get(quant, 0.100)
        # Add realistic jitter (+/- 20%)
        latency = base * random.uniform(0.8, 1.2)

        # Simulate classification result
        class_label = random.choice(self.IMAGENET_CLASSES)
        confidence = random.uniform(0.6, 0.99)

        time.sleep(latency * 0.1)  # Shortened for demo

        return class_label, confidence, latency * 1000  # ms


# ---------------------------------------------------------------------------
# Exercise Solutions
# ---------------------------------------------------------------------------

# === Exercise 1: Model Conversion ===
# Problem: Convert a Keras model to TFLite, apply dynamic quantization,
# compare sizes.

def exercise_1():
    """Solution: TFLite model conversion with quantization comparison."""

    print("  TFLite Model Conversion\n")

    # Simulate a MobileNetV2 model (commonly used for edge AI)
    model = SimulatedKerasModel(
        name="MobileNetV2",
        input_shape=(1, 224, 224, 3),
        num_classes=1000,
        num_params=3_400_000,  # ~3.4M parameters
    )

    print("    --- Original Keras Model ---")
    model.summary()

    # Convert with different quantization options
    conversions = [
        None,       # FP32 (no quantization)
        "dynamic",  # Dynamic range
        "fp16",     # FP16
        "int8",     # Full INT8
    ]

    print("\n    --- Conversion Results ---\n")
    print(f"    {'Method':<30} {'Size (MB)':>10} {'Compression':>12} {'Accuracy Delta':>15}")
    print(f"    {'-'*30} {'-'*10} {'-'*12} {'-'*15}")

    results = []
    for quant in conversions:
        result = SimulatedTFLiteConverter.convert(model, quantization=quant)
        results.append(result)
        print(f"    {result['label']:<30} {result['size_mb']:>8.1f} MB "
              f"{result['compression_ratio']:>10.1f}x "
              f"{result['accuracy_delta_pct']:>+13.1f}%")

    # Conversion code reference (for real TensorFlow)
    print("""
    --- Reference Code (real TensorFlow) ---

    import tensorflow as tf

    # Load Keras model
    model = tf.keras.models.load_model('mobilenetv2.h5')

    # FP32 conversion (no quantization)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = converter.convert()

    # Dynamic range quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dynamic = converter.convert()

    # FP16 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter.convert()

    # Full INT8 quantization (requires representative dataset)
    def representative_dataset():
        for _ in range(100):
            yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_int8 = converter.convert()
    """)


# === Exercise 2: Performance Optimization ===
# Problem: Compare FP32, FP16, INT8 performance. Measure FPS on Raspberry Pi.

def exercise_2():
    """Solution: Performance comparison of quantization levels on Pi."""

    print("  Performance Optimization: Quantization Benchmark\n")

    model = SimulatedKerasModel(
        name="MobileNetV2",
        input_shape=(1, 224, 224, 3),
        num_classes=1000,
        num_params=3_400_000,
    )

    quantization_levels = [
        (None, "FP32"),
        ("fp16", "FP16"),
        ("int8", "INT8"),
    ]

    num_inferences = 50  # Reduced for demo; production uses 100+
    results = []

    for quant, label in quantization_levels:
        model_info = SimulatedTFLiteConverter.convert(model, quantization=quant)
        interpreter = SimulatedTFLiteInterpreter(model_info)

        latencies = []
        for _ in range(num_inferences):
            _, _, latency_ms = interpreter.invoke()
            latencies.append(latency_ms)

        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        fps = 1000.0 / avg_latency

        results.append({
            "label": label,
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "fps": fps,
            "size_mb": model_info["size_mb"],
        })

    # Print comparison table
    print(f"    {'Format':<8} {'Avg Latency':>12} {'Min':>10} {'Max':>10} "
          f"{'FPS':>8} {'Size':>10}")
    print(f"    {'-'*8} {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

    for r in results:
        print(f"    {r['label']:<8} {r['avg_latency_ms']:>10.1f}ms "
              f"{r['min_latency_ms']:>8.1f}ms {r['max_latency_ms']:>8.1f}ms "
              f"{r['fps']:>6.1f} {r['size_mb']:>8.1f} MB")

    # Speedup analysis
    fp32_latency = results[0]["avg_latency_ms"]
    print(f"\n    Speedup vs FP32:")
    for r in results[1:]:
        speedup = fp32_latency / r["avg_latency_ms"]
        print(f"      {r['label']}: {speedup:.1f}x faster")

    print("""
    Key takeaways for Raspberry Pi Edge AI:
    - INT8 gives the best speed (~3x over FP32) with 4x smaller model
    - FP16 is a good middle ground: ~1.4x speedup with minimal accuracy loss
    - For real-time applications (>10 FPS), INT8 is typically required on Pi 4
    - Always validate accuracy on your specific dataset after quantization
    """)


# === Exercise 3: Real-time Classification ===
# Problem: Real-time image classification with Pi Camera, publish to MQTT.

def exercise_3():
    """Solution: Real-time classification pipeline with MQTT output.

    Architecture:
    1. Pi Camera captures frame (simulated)
    2. Preprocess: resize to 224x224, normalize to [0,1]
    3. TFLite interpreter runs inference
    4. Post-process: get top-k classes and confidences
    5. Publish result to MQTT topic: camera/<device_id>/classification
    """

    print("  Real-time Image Classification Pipeline\n")

    model = SimulatedKerasModel(
        name="MobileNetV2",
        input_shape=(1, 224, 224, 3),
        num_classes=1000,
        num_params=3_400_000,
    )
    model_info = SimulatedTFLiteConverter.convert(model, quantization="int8")
    interpreter = SimulatedTFLiteInterpreter(model_info)

    # Simulated MQTT publish log
    mqtt_messages = []

    def mqtt_publish(topic, payload, qos=1):
        """Simulate MQTT publish.

        Real implementation:
            import paho.mqtt.client as mqtt
            client = mqtt.Client()
            client.connect("localhost", 1883)
            client.publish(topic, json.dumps(payload), qos=qos)
        """
        mqtt_messages.append({"topic": topic, "payload": payload})

    device_id = "pi_cam_001"

    print(f"    Device: {device_id}")
    print(f"    Model: {model.name} ({model_info['label']})")
    print(f"    MQTT topic: camera/{device_id}/classification\n")

    # Simulate 10 frames of real-time classification
    print(f"    {'Frame':>6} {'Class':<15} {'Confidence':>11} {'Latency':>10} {'FPS':>6}")
    print(f"    {'-'*6} {'-'*15} {'-'*11} {'-'*10} {'-'*6}")

    total_latency = 0
    for frame_num in range(1, 11):
        # Step 1: Capture frame (simulated)
        # Real: frame = picamera2.capture_array()

        # Step 2: Preprocess
        # Real: input_data = cv2.resize(frame, (224, 224)) / 255.0

        # Step 3: Inference
        class_label, confidence, latency_ms = interpreter.invoke()
        total_latency += latency_ms

        fps = 1000.0 / latency_ms

        print(f"    {frame_num:>6} {class_label:<15} {confidence:>9.1%} "
              f"{latency_ms:>8.1f}ms {fps:>5.1f}")

        # Step 4: Publish to MQTT
        result = {
            "device_id": device_id,
            "frame": frame_num,
            "class": class_label,
            "confidence": round(confidence, 3),
            "latency_ms": round(latency_ms, 1),
            "timestamp": datetime.now().isoformat(),
        }
        mqtt_publish(f"camera/{device_id}/classification", result)

    avg_fps = 1000.0 / (total_latency / 10)
    print(f"\n    Average FPS: {avg_fps:.1f}")
    print(f"    MQTT messages published: {len(mqtt_messages)}")

    # Show sample MQTT message
    print(f"\n    Sample MQTT message:")
    print(f"    Topic: camera/{device_id}/classification")
    print(f"    Payload: {json.dumps(mqtt_messages[-1]['payload'], indent=6)}")

    print("""
    Production pipeline reference (real hardware):

        import cv2
        from picamera2 import Picamera2
        import tflite_runtime.interpreter as tflite
        import paho.mqtt.client as mqtt

        # Initialize
        camera = Picamera2()
        camera.configure(camera.create_preview_configuration(
            main={"size": (640, 480)}))
        camera.start()

        interpreter = tflite.Interpreter(model_path='model.tflite')
        interpreter.allocate_tensors()

        mqtt_client = mqtt.Client()
        mqtt_client.connect("localhost", 1883)

        while True:
            frame = camera.capture_array()
            input_data = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_index)

            class_id = np.argmax(output[0])
            confidence = output[0][class_id]

            mqtt_client.publish(
                f"camera/{device_id}/classification",
                json.dumps({"class": labels[class_id],
                           "confidence": float(confidence)})
            )
    """)


# === Run All Exercises ===
if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 08: Edge AI - TensorFlow Lite - Exercise Solutions")
    print("=" * 70)

    print("\n\n>>> Exercise 1: Model Conversion")
    print("-" * 50)
    exercise_1()

    print("\n\n>>> Exercise 2: Performance Optimization")
    print("-" * 50)
    exercise_2()

    print("\n\n>>> Exercise 3: Real-time Classification")
    print("-" * 50)
    exercise_3()
