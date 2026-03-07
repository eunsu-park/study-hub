"""
06. TFLite Conversion

Demonstrates converting a TensorFlow/Keras model to TensorFlow Lite
format with various quantization options for mobile/embedded deployment.

Covers:
- Basic TFLite conversion (FP32)
- Post-training dynamic range quantization
- Post-training full integer quantization (INT8)
- Float16 quantization
- Model size comparison
- TFLite interpreter inference

Requirements:
    pip install tensorflow numpy
"""

import numpy as np
import os
import tempfile
import time

print("=" * 60)
print("Edge AI — TFLite Conversion")
print("=" * 60)

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not installed. Install with: pip install tensorflow")
    print("This example requires TensorFlow for TFLite conversion.")
    exit(0)


# ============================================
# 1. Build a Keras Model
# ============================================
print("\n[1] Build Keras Model")
print("-" * 40)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Save the Keras model
keras_path = os.path.join(tempfile.gettempdir(), "model.keras")
model.save(keras_path)
keras_size = os.path.getsize(keras_path) / 1024
print(f"\nKeras model size: {keras_size:.1f} KB")


# ============================================
# 2. Basic TFLite Conversion (FP32)
# ============================================
print("\n[2] Basic TFLite Conversion (FP32)")
print("-" * 40)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_fp32 = converter.convert()

fp32_path = os.path.join(tempfile.gettempdir(), "model_fp32.tflite")
with open(fp32_path, "wb") as f:
    f.write(tflite_fp32)

fp32_size = len(tflite_fp32) / 1024
print(f"FP32 TFLite model size: {fp32_size:.1f} KB")


# ============================================
# 3. Dynamic Range Quantization
# ============================================
print("\n[3] Dynamic Range Quantization")
print("-" * 40)
print("Quantizes weights to INT8 at conversion time.")
print("Activations remain in FP32 (quantized dynamically at runtime).\n")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_dynamic = converter.convert()

dynamic_path = os.path.join(tempfile.gettempdir(), "model_dynamic.tflite")
with open(dynamic_path, "wb") as f:
    f.write(tflite_dynamic)

dynamic_size = len(tflite_dynamic) / 1024
print(f"Dynamic quantized size: {dynamic_size:.1f} KB")
print(f"Reduction vs FP32: {fp32_size / dynamic_size:.1f}x")


# ============================================
# 4. Full Integer Quantization (INT8)
# ============================================
print("\n[4] Full Integer Quantization (INT8)")
print("-" * 40)
print("Both weights AND activations are quantized to INT8.")
print("Requires a representative dataset for calibration.\n")


def representative_data_gen():
    """Generate representative data for INT8 calibration."""
    for _ in range(100):
        # Random data simulating normalized images
        data = np.random.randn(1, 28, 28, 1).astype(np.float32)
        yield [data]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Force full integer quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_int8 = converter.convert()

int8_path = os.path.join(tempfile.gettempdir(), "model_int8.tflite")
with open(int8_path, "wb") as f:
    f.write(tflite_int8)

int8_size = len(tflite_int8) / 1024
print(f"INT8 quantized size: {int8_size:.1f} KB")
print(f"Reduction vs FP32: {fp32_size / int8_size:.1f}x")


# ============================================
# 5. Float16 Quantization
# ============================================
print("\n[5] Float16 Quantization")
print("-" * 40)
print("Weights quantized to FP16. Good balance of size and accuracy.")
print("Activations computed in FP16 on GPU, FP32 on CPU.\n")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter.convert()

fp16_path = os.path.join(tempfile.gettempdir(), "model_fp16.tflite")
with open(fp16_path, "wb") as f:
    f.write(tflite_fp16)

fp16_size = len(tflite_fp16) / 1024
print(f"FP16 quantized size: {fp16_size:.1f} KB")
print(f"Reduction vs FP32: {fp32_size / fp16_size:.1f}x")


# ============================================
# 6. TFLite Interpreter Inference
# ============================================
print("\n[6] TFLite Interpreter Inference (FP32 model)")
print("-" * 40)

# Load the FP32 TFLite model
interpreter = tf.lite.Interpreter(model_path=fp32_path)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input:  shape={input_details[0]['shape']}, "
      f"dtype={input_details[0]['dtype']}")
print(f"Output: shape={output_details[0]['shape']}, "
      f"dtype={output_details[0]['dtype']}")

# Run inference
test_input = np.random.randn(1, 28, 28, 1).astype(np.float32)
interpreter.set_tensor(input_details[0]["index"], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]["index"])

print(f"\nInference output shape: {output.shape}")
print(f"Predicted class: {output.argmax()}")
print(f"Confidence: {output.max():.4f}")


# ============================================
# 7. Comparison Summary
# ============================================
print("\n[7] TFLite Quantization Comparison")
print("-" * 40)

print(f"\n{'Method':<28} {'Size (KB)':<12} {'Reduction':<12} {'Notes'}")
print("-" * 80)
print(f"{'Keras (original)':<28} {keras_size:<12.1f} {'—':<12} {'Full model + optimizer state'}")
print(f"{'TFLite FP32':<28} {fp32_size:<12.1f} {'1.0x':<12} {'Baseline TFLite'}")
print(f"{'TFLite Dynamic Range':<28} {dynamic_size:<12.1f} "
      f"{fp32_size/dynamic_size:<12.1f}x {'INT8 weights, FP32 activations'}")
print(f"{'TFLite FP16':<28} {fp16_size:<12.1f} "
      f"{fp32_size/fp16_size:<12.1f}x {'FP16 weights, GPU-accelerated'}")
print(f"{'TFLite INT8 (full)':<28} {int8_size:<12.1f} "
      f"{fp32_size/int8_size:<12.1f}x {'INT8 weights + activations'}")

# Cleanup
for path in [keras_path, fp32_path, dynamic_path, int8_path, fp16_path]:
    if os.path.exists(path):
        os.remove(path)

print()
print("Key takeaways:")
print("- Dynamic range quantization: easiest, ~2-4x smaller, minimal accuracy loss")
print("- Full INT8 quantization: best for MCUs/Edge TPU, needs calibration data")
print("- FP16 quantization: good for GPU inference, ~2x smaller")
print("- Always measure accuracy on your dataset after quantization")
