"""
Exercises for Lesson 11: Mobile Deployment
Topic: Edge_AI

Solutions to practice problems from the lesson.
"""

import math


# === Exercise 1: Mobile Inference Pipeline Design ===
# Problem: Design an inference pipeline for real-time image classification
# on Android, choosing appropriate APIs and data formats.

def exercise_1():
    """Design Android inference pipeline for image classification."""
    pipeline = {
        "Android": {
            "camera": "CameraX (Jetpack) — async frame callback",
            "preprocess": (
                "ImageAnalysis.Analyzer:\n"
                "      1. YUV_420_888 -> RGB bitmap (RenderScript or libyuv)\n"
                "      2. Center crop to 1:1 aspect ratio\n"
                "      3. Resize to 224x224 (Bitmap.createScaledBitmap)\n"
                "      4. Normalize: (pixel / 255.0 - mean) / std\n"
                "      5. Pack into ByteBuffer (NHWC, float32 or uint8)"
            ),
            "inference": (
                "TFLite Interpreter:\n"
                "      interpreter.run(inputBuffer, outputBuffer)\n"
                "      NNAPI delegate for hardware acceleration"
            ),
            "postprocess": (
                "1. Softmax on output logits\n"
                "      2. argmax for predicted class\n"
                "      3. Map index to class label\n"
                "      4. Display with confidence score"
            ),
        },
        "iOS": {
            "camera": "AVFoundation — CMSampleBuffer callback",
            "preprocess": (
                "VNImageRequestHandler:\n"
                "      1. CVPixelBuffer -> CGImage\n"
                "      2. VNCoreMLRequest handles resize/normalize\n"
                "      3. Or manual: vImage for fast resize + normalize"
            ),
            "inference": (
                "Core ML:\n"
                "      let prediction = try model.prediction(input: mlInput)\n"
                "      Neural Engine automatically selected if available"
            ),
            "postprocess": (
                "1. VNClassificationObservation contains label + confidence\n"
                "      2. Sort by confidence\n"
                "      3. Display top-K results"
            ),
        },
    }

    for platform, stages in pipeline.items():
        print(f"  [{platform} Pipeline]")
        for stage, impl in stages.items():
            print(f"    {stage}: {impl}")
        print()

    # Performance comparison
    print("  Expected performance (MobileNet-V2, phone-class device):\n")
    print(f"  {'Stage':<18} {'Android (NNAPI)':>16} {'iOS (CoreML)':>16}")
    print("  " + "-" * 52)
    stages = [
        ("Camera capture", "0 ms (async)", "0 ms (async)"),
        ("Preprocess", "3-5 ms", "2-3 ms"),
        ("Inference", "8-15 ms", "5-10 ms"),
        ("Postprocess", "< 1 ms", "< 1 ms"),
        ("UI update", "1-2 ms", "1-2 ms"),
        ("Total", "13-23 ms", "9-16 ms"),
    ]
    for stage, android, ios in stages:
        print(f"  {stage:<18} {android:>16} {ios:>16}")

    print("\n  Both platforms achieve 30+ FPS for real-time classification.")


# === Exercise 2: NNAPI vs CoreML Feature Comparison ===
# Problem: Compare Android NNAPI and iOS CoreML capabilities
# for deploying neural networks on mobile devices.

def exercise_2():
    """Compare NNAPI and CoreML for mobile ML deployment."""
    comparison = {
        "API Level": {
            "NNAPI": "Android 8.1+ (API 27), improved each version",
            "CoreML": "iOS 11+, major updates yearly (Neural Engine iOS 12+)",
        },
        "Hardware Access": {
            "NNAPI": "Vendor-specific: Qualcomm HTA/DSP, Samsung NPU, MediaTek APU",
            "CoreML": "Apple Neural Engine, GPU, CPU (automatic selection)",
        },
        "Model Format": {
            "NNAPI": "TFLite (.tflite) with NNAPI delegate",
            "CoreML": "Core ML Model (.mlmodel / .mlpackage)",
        },
        "Quantization": {
            "NNAPI": "INT8 (best acceleration), FP16, FP32",
            "CoreML": "FP16 (default on Neural Engine), INT8 (iOS 17+)",
        },
        "Supported Ops": {
            "NNAPI": "~100 ops, varies by vendor and Android version",
            "CoreML": "~200+ ops, consistent across Apple devices",
        },
        "Fragmentation": {
            "NNAPI": "HIGH — different behavior per vendor/SoC/Android version",
            "CoreML": "LOW — Apple controls full hardware+software stack",
        },
        "Fallback": {
            "NNAPI": "Unsupported ops fall back to CPU (performance cliff)",
            "CoreML": "Automatic: Neural Engine -> GPU -> CPU (graceful)",
        },
        "Debugging": {
            "NNAPI": "Limited: benchmark tool, systrace, vendor tools",
            "CoreML": "Xcode ML Performance tab, Instruments, Core ML Profiler",
        },
    }

    print("  NNAPI vs CoreML Comparison:\n")
    for feature, values in comparison.items():
        print(f"  [{feature}]")
        print(f"    NNAPI:  {values['NNAPI']}")
        print(f"    CoreML: {values['CoreML']}")
        print()

    print("  Key takeaway:")
    print("  CoreML is more consistent (Apple controls everything)")
    print("  NNAPI offers wider hardware diversity but more fragmentation")
    print("  Always test on actual target devices for NNAPI deployments")


# === Exercise 3: Mobile Memory Budget ===
# Problem: Calculate memory budget for an ML model running alongside
# a typical mobile app, considering OS and app memory constraints.

def exercise_3():
    """Calculate mobile ML memory budget."""
    # Typical mobile device memory layout
    device_configs = [
        {
            "name": "Budget phone (4GB RAM)",
            "total_ram_gb": 4,
            "os_reserved_gb": 1.5,
            "background_apps_gb": 1.0,
            "app_budget_gb": 0.5,
        },
        {
            "name": "Mid-range phone (6GB RAM)",
            "total_ram_gb": 6,
            "os_reserved_gb": 1.5,
            "background_apps_gb": 1.5,
            "app_budget_gb": 0.8,
        },
        {
            "name": "Flagship phone (12GB RAM)",
            "total_ram_gb": 12,
            "os_reserved_gb": 2.0,
            "background_apps_gb": 3.0,
            "app_budget_gb": 1.5,
        },
    ]

    print("  Mobile Memory Budget Analysis:\n")

    for config in device_configs:
        print(f"  [{config['name']}]")
        app_budget_mb = config['app_budget_gb'] * 1024

        # App memory breakdown
        app_ui_mb = 50          # UI rendering, views
        app_image_cache_mb = 30  # Image caching
        app_other_mb = 20        # Network, database, etc.
        app_overhead_mb = app_ui_mb + app_image_cache_mb + app_other_mb

        ml_budget_mb = app_budget_mb - app_overhead_mb

        print(f"    Total RAM:         {config['total_ram_gb']:.0f} GB")
        print(f"    OS reserved:       {config['os_reserved_gb']:.1f} GB")
        print(f"    Background apps:   {config['background_apps_gb']:.1f} GB")
        print(f"    App budget:        {app_budget_mb:.0f} MB")
        print(f"    App overhead:      {app_overhead_mb:.0f} MB")
        print(f"    ML budget:         {ml_budget_mb:.0f} MB")

        # What fits in this budget?
        models = [
            ("MobileNet-V2 FP32", 14),
            ("MobileNet-V2 INT8", 3.5),
            ("EfficientNet-B0 FP32", 20),
            ("EfficientNet-B0 INT8", 5),
            ("YOLOv5s FP32", 28),
            ("YOLOv5s INT8", 7),
            ("ResNet-50 FP32", 98),
            ("BERT-tiny INT8", 17),
        ]

        print(f"\n    Models that fit ({ml_budget_mb:.0f} MB budget):")
        for model_name, size_mb in models:
            # Model size + activation memory (roughly 2x model for inference)
            total_mb = size_mb * 2  # Model + activation estimate
            fits = "YES" if total_mb <= ml_budget_mb else "NO"
            print(f"      {model_name:<25} {size_mb:>5.1f}MB model, "
                  f"~{total_mb:.0f}MB total -> {fits}")
        print()

    print("  Guidelines:")
    print("  - Model size < 50% of ML budget (leave room for activations)")
    print("  - Use INT8 quantization to fit larger models")
    print("  - Memory-map the model file (don't load entirely into RAM)")
    print("  - Release ML resources when app goes to background")


# === Exercise 4: Cross-Platform Deployment Strategy ===
# Problem: Design a deployment strategy for an app that needs to
# run on both Android and iOS with optimal performance.

def exercise_4():
    """Design cross-platform mobile ML deployment strategy."""
    strategies = [
        {
            "name": "Strategy A: Single Model (TFLite everywhere)",
            "approach": "One TFLite model for both platforms",
            "android": "TFLite + NNAPI delegate",
            "ios": "TFLite + CoreML delegate (or GPU delegate)",
            "pros": [
                "Single model to maintain",
                "Consistent behavior across platforms",
                "Simpler CI/CD pipeline",
            ],
            "cons": [
                "Not optimal for iOS (CoreML native is faster)",
                "TFLite iOS binary adds ~2MB to app",
                "Missing some CoreML-specific optimizations",
            ],
            "best_for": "Small teams, MVP, accuracy-critical apps",
        },
        {
            "name": "Strategy B: Platform-Native (TFLite + CoreML)",
            "approach": "Convert model to each platform's native format",
            "android": "TFLite (.tflite) + NNAPI delegate",
            "ios": "Core ML (.mlmodel) via coremltools",
            "pros": [
                "Best performance on each platform",
                "Full access to platform-specific features",
                "Neural Engine fully utilized on iOS",
            ],
            "cons": [
                "Two conversion pipelines to maintain",
                "Potential output differences between platforms",
                "Need testing on both platforms independently",
            ],
            "best_for": "Production apps, performance-critical",
        },
        {
            "name": "Strategy C: ONNX Runtime Mobile",
            "approach": "ONNX model + ONNX Runtime on both platforms",
            "android": "ONNX Runtime + NNAPI EP",
            "ios": "ONNX Runtime + CoreML EP",
            "pros": [
                "Framework-agnostic (PyTorch, TF, etc.)",
                "Single ONNX model for both platforms",
                "Good optimization passes",
            ],
            "cons": [
                "Larger binary size (~10-20MB)",
                "Less mature mobile support than TFLite",
                "Delegate support still evolving",
            ],
            "best_for": "PyTorch-first teams, complex models",
        },
    ]

    for strat in strategies:
        print(f"  [{strat['name']}]")
        print(f"    Android: {strat['android']}")
        print(f"    iOS:     {strat['ios']}")
        print(f"    Pros:")
        for p in strat['pros']:
            print(f"      + {p}")
        print(f"    Cons:")
        for c in strat['cons']:
            print(f"      - {c}")
        print(f"    Best for: {strat['best_for']}")
        print()

    print("  Recommendation:")
    print("  Start with Strategy A (single TFLite model) for simplicity.")
    print("  Move to Strategy B (platform-native) when performance matters.")
    print("  Use Strategy C (ONNX Runtime) if your pipeline is PyTorch-based")
    print("  and you want to avoid TensorFlow dependency.")


if __name__ == "__main__":
    print("=== Exercise 1: Mobile Inference Pipeline Design ===")
    exercise_1()
    print("\n=== Exercise 2: NNAPI vs CoreML Feature Comparison ===")
    exercise_2()
    print("\n=== Exercise 3: Mobile Memory Budget ===")
    exercise_3()
    print("\n=== Exercise 4: Cross-Platform Deployment Strategy ===")
    exercise_4()
    print("\nAll exercises completed!")
