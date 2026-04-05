"""
Exercises for Lesson 02: Model Compression Overview
Topic: Edge_AI

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn as nn
import numpy as np


# === Exercise 1: Compression Taxonomy ===
# Problem: For each compression technique, describe (a) what it reduces,
# (b) typical compression ratio, and (c) when to use it.

def exercise_1():
    """Compression technique taxonomy and comparison."""
    techniques = {
        "Quantization": {
            "reduces": "Numerical precision (FP32 -> INT8/INT4)",
            "ratio": "2-4x size reduction, 2-4x speedup on supported HW",
            "when": (
                "First technique to try. Works on any model. "
                "PTQ needs no retraining; QAT if PTQ accuracy drops."
            ),
            "accuracy_loss": "< 1% (PTQ), < 0.5% (QAT)",
        },
        "Pruning": {
            "reduces": "Number of weights (sets to zero or removes channels)",
            "ratio": "2-10x for unstructured, 1.5-3x for structured",
            "when": (
                "Over-parameterized models. Structured pruning for real "
                "speedup without sparse hardware."
            ),
            "accuracy_loss": "< 1% at 50-80% sparsity with fine-tuning",
        },
        "Knowledge Distillation": {
            "reduces": "Model architecture size (train smaller student)",
            "ratio": "3-10x parameter reduction",
            "when": (
                "When you can design a smaller architecture. "
                "Requires teacher model and training data."
            ),
            "accuracy_loss": "1-3% vs teacher, often better than training from scratch",
        },
        "Neural Architecture Search": {
            "reduces": "Architecture complexity (finds efficient designs)",
            "ratio": "Variable (designs optimal architecture for constraints)",
            "when": (
                "Large-scale deployment where NAS search cost is amortized. "
                "Hardware-aware NAS for specific target devices."
            ),
            "accuracy_loss": "Often matches or exceeds hand-designed models",
        },
        "Low-Rank Factorization": {
            "reduces": "Weight matrix dimensions via SVD/Tucker decomposition",
            "ratio": "2-5x for FC layers, 1.5-3x for conv layers",
            "when": (
                "Large fully-connected layers. "
                "Can be combined with other techniques."
            ),
            "accuracy_loss": "1-2% with proper rank selection",
        },
    }

    print("  Model Compression Taxonomy:\n")
    for name, info in techniques.items():
        print(f"  {name}:")
        print(f"    Reduces:      {info['reduces']}")
        print(f"    Ratio:        {info['ratio']}")
        print(f"    Accuracy:     {info['accuracy_loss']}")
        print(f"    When to use:  {info['when']}")
        print()


# === Exercise 2: Compression Ratio Calculation ===
# Problem: Calculate theoretical and actual compression ratios
# for different techniques applied to a ResNet-18 style model.

def exercise_2():
    """Calculate compression ratios for a model."""

    # Simplified ResNet-18 stats
    total_params = 11_689_512
    param_bytes_fp32 = total_params * 4  # 4 bytes per FP32

    print(f"  Baseline model: {total_params:,} parameters")
    print(f"  FP32 size: {param_bytes_fp32 / 1e6:.1f} MB\n")

    compressions = [
        {
            "name": "INT8 Quantization",
            "bytes_per_param": 1,
            "params_remaining": total_params,
        },
        {
            "name": "INT4 Quantization",
            "bytes_per_param": 0.5,
            "params_remaining": total_params,
        },
        {
            "name": "50% Unstructured Pruning + FP32",
            "bytes_per_param": 4,  # Still FP32 (sparse format has overhead)
            "params_remaining": total_params // 2,
        },
        {
            "name": "50% Pruning + INT8",
            "bytes_per_param": 1,
            "params_remaining": total_params // 2,
        },
        {
            "name": "Knowledge Distillation (5x smaller student)",
            "bytes_per_param": 4,
            "params_remaining": total_params // 5,
        },
        {
            "name": "KD (5x smaller) + INT8",
            "bytes_per_param": 1,
            "params_remaining": total_params // 5,
        },
    ]

    print(f"  {'Technique':<40} {'Size (MB)':>10} {'Ratio':>8}")
    print("  " + "-" * 60)

    for c in compressions:
        size = c["params_remaining"] * c["bytes_per_param"]
        ratio = param_bytes_fp32 / size
        print(f"  {c['name']:<40} {size / 1e6:>10.1f} {ratio:>7.1f}x")

    print("\n  Key insight: Combining techniques (pruning + quantization +")
    print("  distillation) achieves multiplicative compression ratios.")


# === Exercise 3: Compression Strategy Selection ===
# Problem: Given a deployment target, recommend a compression strategy.

def exercise_3():
    """Select compression strategy based on deployment constraints."""
    scenarios = [
        {
            "target": "NVIDIA Jetson Nano (4GB RAM, 128-core GPU)",
            "model": "YOLOv5s (7.2M params, 16.5 GFLOPs)",
            "constraint": "< 30ms inference, < 500MB RAM",
            "strategy": [
                "1. Start with FP16 inference (TensorRT) — 2x speedup, free",
                "2. Try INT8 with TensorRT calibration — 4x speedup",
                "3. If still too slow: prune + retrain, then re-quantize",
            ],
            "reasoning": (
                "Jetson has GPU with TensorRT support. FP16/INT8 are "
                "the lowest-effort, highest-impact optimizations."
            ),
        },
        {
            "target": "ARM Cortex-M7 MCU (512KB RAM, 216MHz)",
            "model": "Person detector (current: 2M params)",
            "constraint": "< 256KB model, < 100ms inference",
            "strategy": [
                "1. Design tiny architecture (MobileNet-v2 0.25x or MCUNet)",
                "2. Train with knowledge distillation from large detector",
                "3. INT8 quantization (TFLite Micro)",
                "4. Structured pruning to fit RAM budget",
            ],
            "reasoning": (
                "256KB is extremely tight. Must start with architecture "
                "that fits, then compress further. KD helps recover accuracy."
            ),
        },
        {
            "target": "Mobile phone (Snapdragon 888, 8GB RAM)",
            "model": "Image segmentation (DeepLab, 40M params)",
            "constraint": "< 50ms, < 100MB app size",
            "strategy": [
                "1. Replace backbone with MobileNet-v3 (architecture change)",
                "2. Convert to TFLite with dynamic range quantization",
                "3. Use NNAPI delegate for hardware acceleration",
                "4. If accuracy drops: QAT instead of PTQ",
            ],
            "reasoning": (
                "Mobile NPUs are optimized for specific architectures. "
                "MobileNet + NNAPI is the standard mobile deployment path."
            ),
        },
    ]

    for s in scenarios:
        print(f"  Target: {s['target']}")
        print(f"  Model:  {s['model']}")
        print(f"  Constraint: {s['constraint']}")
        print(f"  Strategy:")
        for step in s['strategy']:
            print(f"    {step}")
        print(f"  Reasoning: {s['reasoning']}")
        print()


# === Exercise 4: Accuracy-Efficiency Pareto Analysis ===
# Problem: Given multiple compressed model variants, identify the
# Pareto-optimal set (no model is both faster AND more accurate).

def exercise_4():
    """Identify Pareto-optimal models from compression variants."""
    # (name, accuracy%, latency_ms)
    models = [
        ("Baseline FP32", 95.2, 120),
        ("INT8 PTQ", 94.8, 35),
        ("INT8 QAT", 95.0, 35),
        ("Pruned 50% FP32", 94.5, 90),
        ("Pruned 50% INT8", 94.0, 28),
        ("Pruned 80% INT8", 91.5, 18),
        ("Student (KD)", 93.5, 15),
        ("Student (KD) INT8", 93.0, 8),
        ("MobileNet baseline", 92.0, 20),
        ("MobileNet INT8", 91.8, 10),
    ]

    print("  All model variants:")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Latency':>10}")
    print("  " + "-" * 47)
    for name, acc, lat in models:
        print(f"  {name:<25} {acc:>9.1f}% {lat:>9}ms")

    # Find Pareto front: a model is Pareto-optimal if no other model
    # has BOTH higher accuracy AND lower latency
    pareto = []
    for i, (name_i, acc_i, lat_i) in enumerate(models):
        dominated = False
        for j, (name_j, acc_j, lat_j) in enumerate(models):
            if i != j and acc_j >= acc_i and lat_j <= lat_i:
                if acc_j > acc_i or lat_j < lat_i:  # Strictly better in at least one
                    dominated = True
                    break
        if not dominated:
            pareto.append((name_i, acc_i, lat_i))

    print("\n  Pareto-optimal models (no model is both faster AND more accurate):")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Latency':>10}")
    print("  " + "-" * 47)
    for name, acc, lat in sorted(pareto, key=lambda x: -x[1]):
        print(f"  {name:<25} {acc:>9.1f}% {lat:>9}ms")

    print(f"\n  {len(pareto)} out of {len(models)} models are Pareto-optimal")
    print("  Use case determines which Pareto point to choose:")
    print("  - Safety-critical: highest accuracy on the front")
    print("  - Real-time: lowest latency on the front")


if __name__ == "__main__":
    print("=== Exercise 1: Compression Taxonomy ===")
    exercise_1()
    print("\n=== Exercise 2: Compression Ratio Calculation ===")
    exercise_2()
    print("\n=== Exercise 3: Compression Strategy Selection ===")
    exercise_3()
    print("\n=== Exercise 4: Accuracy-Efficiency Pareto Analysis ===")
    exercise_4()
    print("\nAll exercises completed!")
