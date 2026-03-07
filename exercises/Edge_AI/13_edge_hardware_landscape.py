"""
Exercises for Lesson 13: Edge Hardware Landscape
Topic: Edge_AI

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Hardware Comparison Matrix ===
# Problem: Create a detailed comparison of edge AI hardware platforms
# across key metrics relevant to deployment decisions.

def exercise_1():
    """Compare edge AI hardware platforms."""
    platforms = [
        {
            "name": "NVIDIA Jetson Orin Nano",
            "type": "GPU SoM",
            "tops": 40,
            "power_w": "7-15",
            "memory": "8 GB LPDDR5",
            "price_usd": "$199",
            "sw_stack": "TensorRT, CUDA, DeepStream",
            "best_for": "Video analytics, robotics",
        },
        {
            "name": "NVIDIA Jetson AGX Orin",
            "type": "GPU SoM",
            "tops": 275,
            "power_w": "15-60",
            "memory": "32/64 GB LPDDR5",
            "price_usd": "$999-1599",
            "sw_stack": "TensorRT, CUDA, Isaac",
            "best_for": "Autonomous vehicles, high-end robotics",
        },
        {
            "name": "Google Coral Dev Board",
            "type": "Edge TPU",
            "tops": 4,
            "power_w": "2",
            "memory": "1 GB LPDDR4",
            "price_usd": "$129",
            "sw_stack": "TFLite, PyCoral",
            "best_for": "INT8 classification, detection",
        },
        {
            "name": "Coral USB Accelerator",
            "type": "Edge TPU (USB)",
            "tops": 4,
            "power_w": "2.5",
            "memory": "Uses host RAM",
            "price_usd": "$60",
            "sw_stack": "TFLite + Edge TPU delegate",
            "best_for": "Add ML to Raspberry Pi",
        },
        {
            "name": "Hailo-8",
            "type": "NPU (M.2)",
            "tops": 26,
            "power_w": "2.5",
            "memory": "Uses host RAM",
            "price_usd": "$89-149",
            "sw_stack": "Hailo Model Zoo, TAPPAS",
            "best_for": "Multi-stream video, smart cameras",
        },
        {
            "name": "Intel Movidius Myriad X",
            "type": "VPU",
            "tops": 4,
            "power_w": "1.5",
            "memory": "512 MB",
            "price_usd": "$79 (NCS2)",
            "sw_stack": "OpenVINO",
            "best_for": "Computer vision at ultra-low power",
        },
        {
            "name": "Raspberry Pi 5 (CPU only)",
            "type": "CPU (ARM)",
            "tops": 0.1,
            "power_w": "5-12",
            "memory": "4/8 GB LPDDR4X",
            "price_usd": "$60-80",
            "sw_stack": "TFLite, ONNX Runtime, PyTorch",
            "best_for": "Prototyping, simple models",
        },
        {
            "name": "STM32H7 (Cortex-M7)",
            "type": "MCU",
            "tops": 0.001,
            "power_w": "0.3",
            "memory": "512 KB SRAM",
            "price_usd": "$15",
            "sw_stack": "TFLite Micro, CMSIS-NN, STM32Cube.AI",
            "best_for": "Keyword spotting, anomaly detection",
        },
    ]

    print("  Edge AI Hardware Comparison:\n")
    print(f"  {'Platform':<26} {'Type':<12} {'TOPS':>6} {'Power':>8} "
          f"{'Memory':>16} {'Price':>8}")
    print("  " + "-" * 80)

    for p in platforms:
        print(f"  {p['name']:<26} {p['type']:<12} {p['tops']:>6} "
              f"{p['power_w']:>6}W {p['memory']:>16} {p['price_usd']:>8}")

    # TOPS per Watt efficiency
    print(f"\n  Energy Efficiency (TOPS/Watt):")
    print(f"  {'Platform':<26} {'TOPS/W':>8}")
    print("  " + "-" * 36)
    for p in platforms:
        max_power = float(p['power_w'].split('-')[-1])
        tops_per_watt = p['tops'] / max_power if max_power > 0 else 0
        print(f"  {p['name']:<26} {tops_per_watt:>8.1f}")


# === Exercise 2: Hardware Selection for Use Cases ===
# Problem: Given specific deployment requirements, select the
# optimal hardware platform and justify the choice.

def exercise_2():
    """Select hardware for specific deployment scenarios."""
    scenarios = [
        {
            "use_case": "Smart doorbell (person detection)",
            "requirements": {
                "latency": "< 100ms per frame",
                "power": "< 5W (battery + solar)",
                "model": "SSD-MobileNet-V2 INT8 (~3MB)",
                "cost": "< $50 per unit (volume)",
                "environment": "Outdoor, -20 to 60C",
            },
            "recommendation": "Google Coral USB Accelerator + Raspberry Pi Zero 2",
            "reasoning": (
                "Coral provides 4 TOPS at 2W for INT8 models. "
                "MobileNet runs in ~10ms. Low cost. "
                "USB form factor allows compact design."
            ),
            "alternative": "Hailo-8L (lower power, higher cost)",
        },
        {
            "use_case": "Factory quality inspection (8 cameras)",
            "requirements": {
                "latency": "< 30ms per frame",
                "power": "< 100W total",
                "model": "YOLOv5m with custom defect classes (~25MB)",
                "cost": "< $2000 for compute unit",
                "environment": "Indoor, controlled",
            },
            "recommendation": "NVIDIA Jetson Orin Nano",
            "reasoning": (
                "40 TOPS handles multiple camera streams. "
                "TensorRT optimization for YOLOv5. "
                "DeepStream SDK for multi-stream pipeline. "
                "CUDA ecosystem for custom preprocessing."
            ),
            "alternative": "Jetson AGX Orin if more cameras needed",
        },
        {
            "use_case": "Wearable health monitor (ECG anomaly)",
            "requirements": {
                "latency": "< 500ms (batch of 10 heartbeats)",
                "power": "< 50mW (coin cell battery, months of life)",
                "model": "Tiny 1D-CNN anomaly detector (~10KB)",
                "cost": "< $10 per unit",
                "environment": "Body-worn, sweat-resistant",
            },
            "recommendation": "STM32L4 (Cortex-M4, ultra-low power)",
            "reasoning": (
                "< 1mW active power, months on coin cell. "
                "10KB model fits easily in 256KB flash. "
                "CMSIS-NN for optimized INT8 inference. "
                "$5 in volume."
            ),
            "alternative": "nRF5340 (if BLE connectivity needed)",
        },
        {
            "use_case": "Autonomous delivery robot (navigation)",
            "requirements": {
                "latency": "< 20ms (real-time path planning)",
                "power": "< 50W (robot battery budget)",
                "model": "Multiple: LiDAR segmentation + camera detection + planner",
                "cost": "< $1500 for compute stack",
                "environment": "Outdoor, urban, all weather",
            },
            "recommendation": "NVIDIA Jetson AGX Orin",
            "reasoning": (
                "275 TOPS handles multiple concurrent models. "
                "CUDA for LiDAR point cloud processing. "
                "Isaac ROS for robotics integration. "
                "GPU needed for real-time sensor fusion."
            ),
            "alternative": "Dual Jetson Orin Nano for redundancy",
        },
    ]

    for s in scenarios:
        print(f"  [{s['use_case']}]")
        for req_name, req_val in s['requirements'].items():
            print(f"    {req_name}: {req_val}")
        print(f"    -> Recommendation: {s['recommendation']}")
        print(f"    Reasoning: {s['reasoning']}")
        print(f"    Alternative: {s['alternative']}")
        print()


# === Exercise 3: NPU vs GPU vs CPU Analysis ===
# Problem: Compare inference on NPU, GPU, and CPU for different
# model types and explain when each excels.

def exercise_3():
    """Compare NPU, GPU, and CPU for different workloads."""
    workloads = [
        {
            "model": "MobileNet-V2 INT8 (classification)",
            "cpu_ms": 45,
            "gpu_ms": 8,
            "npu_ms": 2,
            "winner": "NPU",
            "reason": "Standard CNN — NPU is purpose-built for this",
        },
        {
            "model": "ResNet-50 FP32 (classification)",
            "cpu_ms": 200,
            "gpu_ms": 5,
            "npu_ms": 15,
            "winner": "GPU",
            "reason": "FP32 favors GPU (NPU optimized for INT8)",
        },
        {
            "model": "BERT-base INT8 (NLP)",
            "cpu_ms": 500,
            "gpu_ms": 12,
            "npu_ms": 25,
            "winner": "GPU",
            "reason": "Attention layers have irregular memory access patterns",
        },
        {
            "model": "Tiny keyword spotter (10KB)",
            "cpu_ms": 2,
            "gpu_ms": 5,
            "npu_ms": 3,
            "winner": "CPU",
            "reason": "Model too small — GPU/NPU startup overhead dominates",
        },
        {
            "model": "YOLOv5s INT8 (detection)",
            "cpu_ms": 150,
            "gpu_ms": 10,
            "npu_ms": 5,
            "winner": "NPU",
            "reason": "Standard convolution-heavy INT8 model",
        },
        {
            "model": "Custom GAN (FP32, irregular ops)",
            "cpu_ms": 800,
            "gpu_ms": 15,
            "npu_ms": "N/A",
            "winner": "GPU",
            "reason": "NPU lacks support for many GAN operations",
        },
    ]

    print("  NPU vs GPU vs CPU Performance Comparison:\n")
    print(f"  {'Model':<40} {'CPU':>7} {'GPU':>7} {'NPU':>7} {'Winner':>8}")
    print("  " + "-" * 72)

    for w in workloads:
        npu_str = f"{w['npu_ms']}ms" if isinstance(w['npu_ms'], (int, float)) else w['npu_ms']
        print(f"  {w['model']:<40} {w['cpu_ms']:>5}ms {w['gpu_ms']:>5}ms "
              f"{npu_str:>7} {w['winner']:>8}")

    print("\n  When each processor type wins:")
    print("    CPU: Very small models, non-standard ops, no batch processing")
    print("    GPU: FP32/FP16, large models, irregular memory access, custom ops")
    print("    NPU: INT8 CNNs, standard architectures, power-constrained")
    print("\n  Rule of thumb:")
    print("    - NPU for production inference (INT8 optimized)")
    print("    - GPU for flexibility and large models")
    print("    - CPU as fallback for unsupported operations")


# === Exercise 4: Edge Hardware Trends ===
# Problem: Analyze trends in edge AI hardware and predict
# implications for model deployment in the next 2-3 years.

def exercise_4():
    """Analyze edge AI hardware trends."""
    trends = [
        {
            "trend": "TOPS/Watt Improvement",
            "2022": "2-4 TOPS/W (typical edge NPU)",
            "2025": "10-20 TOPS/W (Hailo-10H, Qualcomm Hexagon)",
            "implication": (
                "Models that needed a Jetson GPU in 2022 can now run "
                "on a $50 NPU module. Larger models move to the edge."
            ),
        },
        {
            "trend": "On-Device Training",
            "2022": "Inference only on most edge devices",
            "2025": "Fine-tuning and federated learning on edge GPUs",
            "implication": (
                "Models can adapt to local data without cloud connectivity. "
                "Privacy-preserving personalization becomes practical."
            ),
        },
        {
            "trend": "Transformer Acceleration",
            "2022": "NPUs optimized for CNNs, poor transformer support",
            "2025": "Dedicated attention/transformer units in new NPUs",
            "implication": (
                "Vision Transformers (ViT) and small language models "
                "become viable on edge NPUs. Model architecture choices expand."
            ),
        },
        {
            "trend": "Heterogeneous Computing",
            "2022": "Single accelerator (GPU or NPU)",
            "2025": "SoCs with CPU + GPU + NPU + DSP on single chip",
            "implication": (
                "Runtime can split a model across processors: "
                "attention on GPU, convolutions on NPU, postprocess on CPU."
            ),
        },
        {
            "trend": "Chiplet Architecture",
            "2022": "Monolithic chips",
            "2025": "Modular chiplets allow custom compute configurations",
            "implication": (
                "Edge devices can be customized per workload: "
                "more NPU tiles for vision, more CPU for control."
            ),
        },
    ]

    print("  Edge AI Hardware Trends (2022 -> 2025+):\n")
    for t in trends:
        print(f"  [{t['trend']}]")
        print(f"    2022: {t['2022']}")
        print(f"    2025: {t['2025']}")
        print(f"    Impact: {t['implication']}")
        print()

    print("  What this means for developers:")
    print("  1. Optimize for NPU first (becoming the default accelerator)")
    print("  2. INT8 quantization is table stakes (all NPUs support it)")
    print("  3. Transformer models are coming to edge — prepare now")
    print("  4. On-device training enables personalization use cases")
    print("  5. Hardware diversity means test on actual target devices")


if __name__ == "__main__":
    print("=== Exercise 1: Hardware Comparison Matrix ===")
    exercise_1()
    print("\n=== Exercise 2: Hardware Selection for Use Cases ===")
    exercise_2()
    print("\n=== Exercise 3: NPU vs GPU vs CPU Analysis ===")
    exercise_3()
    print("\n=== Exercise 4: Edge Hardware Trends ===")
    exercise_4()
    print("\nAll exercises completed!")
