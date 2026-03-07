"""
Exercises for Lesson 09: TFLite and Mobile Runtimes
Topic: Edge_AI

Solutions to practice problems from the lesson.
"""


# === Exercise 1: TFLite Conversion Options ===
# Problem: Compare the different TFLite quantization options and
# their tradeoffs for a classification model.

def exercise_1():
    """Compare TFLite quantization conversion options."""
    options = [
        {
            "name": "No quantization (FP32)",
            "code": "converter = tf.lite.TFLiteConverter.from_keras_model(model)",
            "size_ratio": 1.0,
            "accuracy_loss": "0%",
            "hw_accel": "CPU only (no delegate benefit)",
            "use_case": "Development/debugging baseline",
        },
        {
            "name": "Dynamic Range Quantization",
            "code": "converter.optimizations = [tf.lite.Optimize.DEFAULT]",
            "size_ratio": 0.25,
            "accuracy_loss": "< 1%",
            "hw_accel": "CPU (some delegate support)",
            "use_case": "Easiest optimization, good default choice",
        },
        {
            "name": "Full Integer (INT8) Quantization",
            "code": (
                "converter.optimizations = [tf.lite.Optimize.DEFAULT]\n"
                "    converter.representative_dataset = calibration_gen\n"
                "    converter.target_spec.supported_ops = "
                "[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]"
            ),
            "size_ratio": 0.25,
            "accuracy_loss": "1-2%",
            "hw_accel": "Edge TPU, NNAPI, Hexagon DSP",
            "use_case": "Maximum edge acceleration, MCU deployment",
        },
        {
            "name": "Float16 Quantization",
            "code": (
                "converter.optimizations = [tf.lite.Optimize.DEFAULT]\n"
                "    converter.target_spec.supported_types = [tf.float16]"
            ),
            "size_ratio": 0.5,
            "accuracy_loss": "< 0.5%",
            "hw_accel": "GPU delegate (FP16 compute)",
            "use_case": "Mobile GPU inference, minimal accuracy impact",
        },
    ]

    print("  TFLite Conversion Options Comparison:\n")
    for opt in options:
        print(f"  [{opt['name']}]")
        print(f"    Size:     {opt['size_ratio']:.0%} of original")
        print(f"    Accuracy: {opt['accuracy_loss']} loss")
        print(f"    HW Accel: {opt['hw_accel']}")
        print(f"    Use case: {opt['use_case']}")
        print()

    print("  Decision tree:")
    print("    - Need Edge TPU / NNAPI?  -> Full INT8")
    print("    - Need GPU delegate?      -> FP16")
    print("    - Quick optimization?      -> Dynamic range")
    print("    - Accuracy critical?       -> FP16 or dynamic range")


# === Exercise 2: TFLite Delegate Selection ===
# Problem: Given a target device, select the optimal TFLite delegate
# and explain the execution flow.

def exercise_2():
    """Select optimal TFLite delegate for target hardware."""
    delegates = [
        {
            "name": "GPU Delegate",
            "hardware": "Mobile GPU (Adreno, Mali, PowerVR)",
            "platforms": "Android, iOS",
            "quantization": "FP16, FP32",
            "latency_factor": "2-7x speedup vs CPU",
            "setup": (
                "interpreter = tf.lite.Interpreter(\n"
                "    model_path='model.tflite',\n"
                "    experimental_delegates=[\n"
                "        tf.lite.experimental.load_delegate('libdelegate.so')\n"
                "    ])"
            ),
            "limitations": [
                "Not all ops supported (falls back to CPU)",
                "Initial delegate creation adds startup latency",
                "Memory copies between CPU and GPU add overhead for small models",
            ],
        },
        {
            "name": "NNAPI Delegate",
            "hardware": "Android HW accelerators (NPU, DSP, GPU)",
            "platforms": "Android 8.1+",
            "quantization": "INT8 (best), FP16, FP32",
            "latency_factor": "3-10x speedup (vendor-dependent)",
            "setup": (
                "interpreter = tf.lite.Interpreter(\n"
                "    model_path='model.tflite',\n"
                "    experimental_delegates=[\n"
                "        tf.lite.experimental.load_delegate(\n"
                "            'libneuralnetworks.so')\n"
                "    ])"
            ),
            "limitations": [
                "Behavior varies by Android version and vendor",
                "Op support depends on specific SoC",
                "Need to test on actual target device",
            ],
        },
        {
            "name": "Core ML Delegate",
            "hardware": "Apple Neural Engine, GPU, CPU",
            "platforms": "iOS 12+",
            "quantization": "FP16, FP32 (INT8 limited)",
            "latency_factor": "5-15x on Neural Engine",
            "setup": "Use Core ML tools to convert, not TFLite delegate",
            "limitations": [
                "Apple ecosystem only",
                "Best with Core ML native models, not TFLite",
                "Neural Engine requires compatible model structure",
            ],
        },
        {
            "name": "Edge TPU Delegate",
            "hardware": "Google Coral Edge TPU",
            "platforms": "Linux (Coral Dev Board, USB Accelerator)",
            "quantization": "INT8 only (fully quantized required)",
            "latency_factor": "10-50x vs CPU for supported ops",
            "setup": (
                "interpreter = tf.lite.Interpreter(\n"
                "    model_path='model_edgetpu.tflite',\n"
                "    experimental_delegates=[\n"
                "        tf.lite.experimental.load_delegate(\n"
                "            'libedgetpu.so.1')\n"
                "    ])"
            ),
            "limitations": [
                "MUST be fully INT8 quantized",
                "Limited op set (common CNN ops only)",
                "Non-supported ops run on CPU (very slow fallback)",
            ],
        },
    ]

    for d in delegates:
        print(f"  [{d['name']}]")
        print(f"    Hardware:  {d['hardware']}")
        print(f"    Platforms: {d['platforms']}")
        print(f"    Best with: {d['quantization']}")
        print(f"    Speedup:   {d['latency_factor']}")
        print(f"    Limitations:")
        for lim in d['limitations']:
            print(f"      - {lim}")
        print()

    # Decision matrix
    print("  Delegate Selection Matrix:")
    print(f"  {'Device':<25} {'Delegate':<18} {'Quantization':<15}")
    print("  " + "-" * 60)
    selections = [
        ("Android phone (Snapdragon)", "NNAPI", "INT8"),
        ("Android phone (GPU focus)", "GPU Delegate", "FP16"),
        ("iPhone / iPad", "Core ML", "FP16"),
        ("Google Coral", "Edge TPU", "INT8 (full)"),
        ("Raspberry Pi", "XNNPACK (CPU)", "FP32 / dynamic"),
        ("Linux server (no GPU)", "XNNPACK (CPU)", "Dynamic range"),
    ]
    for device, delegate, quant in selections:
        print(f"  {device:<25} {delegate:<18} {quant}")


# === Exercise 3: Runtime Comparison ===
# Problem: Compare ONNX Runtime, TFLite, and PyTorch Mobile for
# different deployment scenarios.

def exercise_3():
    """Compare mobile/edge inference runtimes."""
    runtimes = [
        {
            "name": "TensorFlow Lite",
            "framework": "TensorFlow",
            "platforms": "Android, iOS, Linux, MCU",
            "model_format": ".tflite (FlatBuffer)",
            "quantization": "INT8, FP16, dynamic range",
            "hw_accel": "GPU, NNAPI, Edge TPU, Hexagon DSP, CoreML",
            "binary_size": "~1 MB (select ops)",
            "strengths": [
                "Widest hardware delegate support",
                "Smallest binary for MCU (TFLite Micro)",
                "Strong INT8 quantization pipeline",
            ],
            "weaknesses": [
                "Model conversion can fail on complex ops",
                "Less flexible than ONNX for PyTorch models",
            ],
        },
        {
            "name": "ONNX Runtime",
            "framework": "Framework-agnostic (PyTorch, TF, etc.)",
            "platforms": "Windows, Linux, macOS, Android, iOS",
            "model_format": ".onnx (Protobuf)",
            "quantization": "INT8, FP16",
            "hw_accel": "CUDA, TensorRT, DirectML, NNAPI, CoreML",
            "binary_size": "~5-20 MB",
            "strengths": [
                "Framework-agnostic (works with any model)",
                "Graph optimizations (fusion, constant folding)",
                "TensorRT integration for NVIDIA GPUs",
            ],
            "weaknesses": [
                "Larger binary than TFLite",
                "Less MCU support (no microcontroller target)",
            ],
        },
        {
            "name": "PyTorch Mobile (ExecuTorch)",
            "framework": "PyTorch",
            "platforms": "Android, iOS, Linux",
            "model_format": ".pte (ExecuTorch) / .ptl (lite)",
            "quantization": "INT8, FP16 (via quantization API)",
            "hw_accel": "XNNPACK, CoreML, Qualcomm QNN, Vulkan",
            "binary_size": "~5-10 MB",
            "strengths": [
                "Seamless PyTorch workflow (no conversion step)",
                "Dynamic shapes natively supported",
                "Active development (ExecuTorch)",
            ],
            "weaknesses": [
                "Newer ecosystem, less mature",
                "Fewer hardware delegates than TFLite",
            ],
        },
    ]

    for rt in runtimes:
        print(f"  [{rt['name']}]")
        print(f"    Framework:  {rt['framework']}")
        print(f"    Platforms:  {rt['platforms']}")
        print(f"    Format:     {rt['model_format']}")
        print(f"    Binary:     {rt['binary_size']}")
        print(f"    HW Accel:   {rt['hw_accel']}")
        print(f"    Strengths:  {', '.join(rt['strengths'])}")
        print(f"    Weaknesses: {', '.join(rt['weaknesses'])}")
        print()

    print("  Recommendation by scenario:")
    print("    - PyTorch model on Android/iOS:   ONNX Runtime or ExecuTorch")
    print("    - TensorFlow model on mobile:      TFLite")
    print("    - MCU/microcontroller deployment:   TFLite Micro")
    print("    - NVIDIA GPU edge device:           ONNX Runtime + TensorRT")
    print("    - Google Coral:                     TFLite + Edge TPU delegate")


# === Exercise 4: Mobile Inference Pipeline Design ===
# Problem: Design a complete mobile inference pipeline for real-time
# object detection on Android.

def exercise_4():
    """Design a mobile inference pipeline for Android."""
    pipeline = {
        "app": "Real-time Object Detection on Android",
        "target": "Snapdragon 8 Gen 2 (Qualcomm)",
        "model": "SSD-MobileNet-V2 (INT8 quantized)",
        "runtime": "TFLite with NNAPI delegate",
        "stages": [
            {
                "stage": "1. Camera Capture",
                "implementation": "CameraX API (Android Jetpack)",
                "output": "YUV_420_888 frame (1280x720)",
                "latency": "~0 ms (async callback)",
            },
            {
                "stage": "2. Frame Preprocessing",
                "implementation": "GPU compute shader or RenderScript",
                "steps": [
                    "YUV to RGB conversion",
                    "Resize to 300x300 (model input)",
                    "Normalize to [-1, 1] or [0, 255] (INT8)",
                    "Convert to ByteBuffer (NHWC format)",
                ],
                "latency": "~3 ms",
            },
            {
                "stage": "3. Model Inference",
                "implementation": "TFLite Interpreter with NNAPI",
                "steps": [
                    "Allocate tensors (first call only)",
                    "Set input tensor",
                    "Run inference",
                    "Get output tensors (boxes, scores, classes)",
                ],
                "latency": "~8 ms (NNAPI), ~25 ms (CPU)",
            },
            {
                "stage": "4. Postprocessing",
                "implementation": "Java/Kotlin on CPU",
                "steps": [
                    "Filter detections by confidence threshold (> 0.5)",
                    "Non-Maximum Suppression (IoU > 0.5)",
                    "Map class indices to labels",
                    "Scale bounding boxes to display resolution",
                ],
                "latency": "~1 ms",
            },
            {
                "stage": "5. UI Rendering",
                "implementation": "Canvas overlay on camera preview",
                "steps": [
                    "Draw bounding boxes",
                    "Render class labels and confidence scores",
                    "Update at camera frame rate",
                ],
                "latency": "~2 ms",
            },
        ],
    }

    print(f"  {pipeline['app']}")
    print(f"  Target: {pipeline['target']}")
    print(f"  Model:  {pipeline['model']}")
    print(f"  Runtime: {pipeline['runtime']}\n")

    total_latency = 0
    for s in pipeline['stages']:
        lat = float(s['latency'].split('~')[1].split(' ')[0])
        total_latency += lat
        print(f"  {s['stage']}")
        print(f"    Implementation: {s['implementation']}")
        if 'steps' in s:
            for step in s['steps']:
                print(f"      - {step}")
        print(f"    Latency: {s['latency']}")
        print()

    fps = 1000 / total_latency
    print(f"  Total pipeline latency: ~{total_latency:.0f} ms")
    print(f"  Achievable frame rate: ~{fps:.0f} FPS")
    print(f"  Meets real-time requirement: {'Yes' if fps >= 30 else 'No'} (30 FPS target)")


if __name__ == "__main__":
    print("=== Exercise 1: TFLite Conversion Options ===")
    exercise_1()
    print("\n=== Exercise 2: TFLite Delegate Selection ===")
    exercise_2()
    print("\n=== Exercise 3: Runtime Comparison ===")
    exercise_3()
    print("\n=== Exercise 4: Mobile Inference Pipeline Design ===")
    exercise_4()
    print("\nAll exercises completed!")
