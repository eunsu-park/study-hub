# Lesson 2: Model Compression Overview

[Previous: Edge AI Fundamentals](./01_Edge_AI_Fundamentals.md) | [Next: Quantization](./03_Quantization.md)

---

## Learning Objectives

- Understand the taxonomy of model compression techniques
- Compare pruning, quantization, distillation, and architecture search at a high level
- Analyze accuracy-size tradeoffs using compression ratios and Pareto curves
- Design a multi-stage compression pipeline for edge deployment
- Evaluate which compression technique(s) to apply for a given scenario

---

## 1. Why Compress Models?

Modern deep learning models are large. A ResNet-50 has 25.6 million parameters (97 MB in FP32). A BERT-Base has 110 million parameters (440 MB). GPT-3 has 175 billion parameters (700 GB). None of these fit comfortably on edge devices in their original form.

**Model compression** reduces the size, latency, and compute requirements of a trained model while preserving as much accuracy as possible.

```
Original Model                     Compressed Model
┌─────────────────┐               ┌─────────────────┐
│ ResNet-50       │    Compress   │ ResNet-50       │
│ 97 MB (FP32)   │──────────────▶│ 24 MB (INT8)    │
│ 4.1 GFLOPS     │               │ 1.0 GFLOPS      │
│ 76.1% top-1    │               │ 75.5% top-1     │
│ 45ms (CPU)     │               │ 15ms (CPU)      │
└─────────────────┘               └─────────────────┘
     4x smaller, 3x faster, 0.6% accuracy loss
```

### 1.1 The Overparameterization Hypothesis

Why can we compress models at all? Because neural networks are **overparameterized**: they have far more parameters than necessary to represent the learned function. This redundancy is useful during training (it makes optimization easier) but wasteful during inference.

```python
import torch
import torch.nn as nn
import numpy as np


def analyze_weight_distribution(model):
    """
    Analyze weight distribution to understand compression potential.
    Many weights cluster near zero — these can be pruned or quantized.
    """
    all_weights = []
    for name, param in model.named_parameters():
        if "weight" in name:
            all_weights.append(param.data.cpu().flatten())

    weights = torch.cat(all_weights)
    total = weights.numel()

    # What fraction of weights are near zero?
    thresholds = [0.01, 0.05, 0.1, 0.2]
    print(f"Total weights: {total:,}")
    print(f"Weight statistics:")
    print(f"  Mean:   {weights.mean():.6f}")
    print(f"  Std:    {weights.std():.6f}")
    print(f"  Min:    {weights.min():.6f}")
    print(f"  Max:    {weights.max():.6f}")
    print(f"\nWeights near zero:")
    for t in thresholds:
        near_zero = (weights.abs() < t).sum().item()
        print(f"  |w| < {t}: {near_zero:,} ({100 * near_zero / total:.1f}%)")


# Example: analyze a pretrained ResNet-18
model = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V1")
analyze_weight_distribution(model)
# Typical output: ~30-50% of weights have magnitude < 0.05
```

---

## 2. Compression Taxonomy

There are four main families of model compression techniques. They are **complementary** — you can (and often should) combine them.

```
                    Model Compression Techniques
                              │
          ┌───────────┬───────┴───────┬──────────────┐
          │           │               │              │
    ┌─────▼─────┐ ┌──▼────────┐ ┌────▼─────┐ ┌─────▼──────┐
    │  Pruning  │ │Quantization│ │Knowledge │ │Architecture│
    │           │ │            │ │Distillat.│ │  Search    │
    └─────┬─────┘ └──┬────────┘ └────┬─────┘ └─────┬──────┘
          │          │               │              │
    Remove        Reduce          Train a        Design
    redundant     numerical       smaller         efficient
    weights       precision       student         architectures
    or neurons    (FP32→INT8)     model           automatically
```

### 2.1 Quantization

**Idea**: Reduce the numerical precision of weights and activations from 32-bit floating-point to lower bit-widths (INT8, INT4, or even binary).

| Precision | Bits | Size Reduction | Typical Accuracy Drop |
|-----------|------|---------------|----------------------|
| FP32 | 32 | 1x (baseline) | 0% |
| FP16 | 16 | 2x | ~0% |
| INT8 | 8 | 4x | 0.1-1% |
| INT4 | 4 | 8x | 1-3% |
| Binary | 1 | 32x | 5-15% |

```python
import torch

# Quick demonstration: FP32 vs INT8 memory
fp32_tensor = torch.randn(1000, 1000)  # 4 bytes per element
int8_tensor = fp32_tensor.to(torch.int8)  # 1 byte per element

fp32_bytes = fp32_tensor.element_size() * fp32_tensor.numel()
int8_bytes = int8_tensor.element_size() * int8_tensor.numel()

print(f"FP32: {fp32_bytes / 1e6:.1f} MB")  # 4.0 MB
print(f"INT8: {int8_bytes / 1e6:.1f} MB")  # 1.0 MB
print(f"Reduction: {fp32_bytes / int8_bytes:.0f}x")  # 4x
```

**When to use**: Almost always. Quantization is the most universally applicable compression technique with the best effort-to-benefit ratio.

**Covered in detail**: [Lesson 3 — Quantization](./03_Quantization.md)

### 2.2 Pruning

**Idea**: Remove weights, neurons, or entire layers that contribute least to model accuracy. The resulting sparse model requires fewer computations.

```
Before Pruning (Dense):          After Pruning (Sparse):
┌─────────────────────┐         ┌─────────────────────┐
│ ● ● ● ● ● ● ● ● ● │         │ ● ○ ● ○ ○ ● ○ ● ○ │
│ ● ● ● ● ● ● ● ● ● │         │ ○ ● ○ ○ ● ○ ● ○ ○ │
│ ● ● ● ● ● ● ● ● ● │         │ ● ○ ○ ● ○ ○ ○ ● ● │
│ ● ● ● ● ● ● ● ● ● │         │ ○ ○ ● ○ ● ● ○ ○ ○ │
└─────────────────────┘         └─────────────────────┘
  81 parameters (100%)            30 parameters (37%)
                                  → 63% sparsity
```

| Type | Granularity | Speedup | Hardware Support |
|------|------------|---------|-----------------|
| Unstructured | Individual weights | Requires sparse hardware | Limited (sparse tensor cores) |
| Structured (channel) | Entire filters/channels | Direct speedup on all hardware | Excellent |
| Structured (block) | N x M blocks | Moderate | NVIDIA Ampere+ |

**When to use**: When you need to reduce FLOPs and can fine-tune after pruning. Structured pruning is preferred for hardware compatibility.

**Covered in detail**: [Lesson 4 — Pruning](./04_Pruning.md)

### 2.3 Knowledge Distillation

**Idea**: Train a small **student** model to mimic the behavior of a large **teacher** model. The student learns from the teacher's soft probability outputs, which contain richer information than hard labels alone.

```
Teacher (Large)                    Student (Small)
┌───────────────┐                 ┌───────────────┐
│ ResNet-152    │   Soft Labels   │ MobileNetV2   │
│ 60M params   │────────────────▶│ 3.4M params   │
│ 78.3% acc    │  (Knowledge     │ 73.5% acc     │
│ 230 MB       │   Transfer)     │ 14 MB         │
└───────────────┘                 └───────────────┘

Without distillation: Student alone gets 71.9% accuracy
With distillation:    Student gets 73.5% accuracy (+1.6%)
```

```python
import torch
import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, true_labels,
                      temperature=4.0, alpha=0.7):
    """
    Combined distillation loss.

    Args:
        student_logits: Raw outputs from student model
        teacher_logits: Raw outputs from teacher model
        true_labels: Ground-truth class labels
        temperature: Softens probability distributions (higher = softer)
        alpha: Balance between soft and hard loss (0 = hard only, 1 = soft only)

    Returns:
        Combined loss scalar
    """
    # Soft loss: KL divergence between softened distributions
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
    soft_loss *= temperature ** 2  # Scale by T^2 (Hinton et al., 2015)

    # Hard loss: standard cross-entropy with true labels
    hard_loss = F.cross_entropy(student_logits, true_labels)

    # Combined loss
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

**When to use**: When you can afford to train a new (smaller) architecture. Especially effective when combined with quantization.

**Covered in detail**: [Lesson 5 — Knowledge Distillation](./05_Knowledge_Distillation.md)

### 2.4 Neural Architecture Search (NAS)

**Idea**: Automatically search for efficient model architectures, optionally with hardware-aware constraints (latency, memory, energy).

```
Search Space                    Search Strategy              Result
┌─────────────────┐            ┌──────────────┐            ┌──────────────┐
│ • Conv types    │            │ • RL         │            │ Discovered   │
│ • Kernel sizes  │──search──▶│ • Evolutionary│──train──▶│ Architecture │
│ • Skip connects │            │ • One-shot   │            │ (Optimized)  │
│ • Channel widths│            │ • Gradient   │            │              │
└─────────────────┘            └──────────────┘            └──────────────┘

Hardware Constraint: Latency < 25ms on Pixel phone
```

| Method | Search Cost | Key Innovation |
|--------|------------|----------------|
| NASNet (2017) | 48,000 GPU-hours | RL-based controller |
| DARTS (2019) | 4 GPU-days | Differentiable search |
| EfficientNet (2019) | ~5,000 GPU-hours | Compound scaling |
| MnasNet (2019) | ~40,000 GPU-hours | Hardware-aware (latency) |
| FBNet (2019) | 9 GPU-hours | Differentiable + HW-aware |
| OFA (2020) | Train once | Once-for-all elastic networks |

**When to use**: When designing a new architecture family for a specific hardware target. High upfront cost but optimal results.

**Covered in detail**: [Lesson 7 — Neural Architecture Search](./07_Neural_Architecture_Search.md)

---

## 3. Compression Ratios and Accuracy Tradeoffs

### 3.1 Understanding Compression Ratio

```python
def compression_metrics(original_size_mb, compressed_size_mb,
                        original_acc, compressed_acc,
                        original_latency_ms, compressed_latency_ms):
    """Calculate standard compression metrics."""
    return {
        "size_reduction": f"{original_size_mb / compressed_size_mb:.1f}x",
        "accuracy_drop": f"{original_acc - compressed_acc:.2f}%",
        "speedup": f"{original_latency_ms / compressed_latency_ms:.1f}x",
        "efficiency_ratio": (compressed_acc / compressed_latency_ms)
                            / (original_acc / original_latency_ms),
    }


# Example: ResNet-50 with INT8 quantization
metrics = compression_metrics(
    original_size_mb=97,
    compressed_size_mb=24,
    original_acc=76.1,
    compressed_acc=75.5,
    original_latency_ms=45,
    compressed_latency_ms=15,
)
for k, v in metrics.items():
    print(f"  {k}: {v}")
```

### 3.2 Accuracy-Size Pareto Curve

The goal is to find models on the **Pareto frontier**: the set of models where no other model is both smaller and more accurate.

```
Accuracy (%)
    │
 78 │                              ● ResNet-50 (FP32)
    │                         ● ResNet-50 (INT8)
 76 │                    ● ResNet-50 (pruned+INT8)
    │              ● EfficientNet-B0 (FP32)
 74 │         ● EfficientNet-B0 (INT8)
    │    ● MobileNetV2 (FP32)
 72 │  ● MobileNetV2 (INT8)
    │● MobileNetV2 (pruned+INT8)
 70 │
    │
 68 │
    └───────────────────────────────────────── Model Size (MB)
     5   10   20    30    40    50   60   80   100

    ─── Pareto Frontier (desirable: up and to the left)
```

```python
import torch


def build_pareto_table():
    """
    Reference data: popular model compression results on ImageNet.
    Sources: torchvision model zoo, papers.
    """
    models = [
        # (Name, Size MB, Top-1 Acc, Latency ms CPU)
        ("ResNet-50 FP32",             97,   76.1, 45),
        ("ResNet-50 INT8",             24,   75.5, 15),
        ("ResNet-50 Pruned+INT8",      15,   74.8, 11),
        ("EfficientNet-B0 FP32",       20,   77.1, 35),
        ("EfficientNet-B0 INT8",        5,   76.5, 12),
        ("MobileNetV2 FP32",           14,   71.9, 22),
        ("MobileNetV2 INT8",            4,   71.2,  8),
        ("MobileNetV2 Pruned+INT8",     2,   69.5,  5),
        ("SqueezeNet FP32",             5,   58.2, 15),
        ("MobileNetV3-Small INT8",      2,   67.5,  4),
    ]

    print(f"{'Model':<30} {'Size(MB)':>8} {'Acc(%)':>7} {'Lat(ms)':>8} {'Acc/Lat':>8}")
    print("-" * 65)
    for name, size, acc, lat in models:
        eff = acc / lat
        print(f"{name:<30} {size:>8} {acc:>7.1f} {lat:>8} {eff:>8.2f}")


build_pareto_table()
```

### 3.3 Combining Techniques

Compression techniques are **multiplicative** in their benefits:

| Technique | Size Reduction | Speedup | Accuracy Drop |
|-----------|---------------|---------|---------------|
| Quantization (INT8) alone | 4x | 2-3x | 0.1-1% |
| Pruning (50%) alone | 2x | 1.5-2x | 0.5-2% |
| Distillation alone | 3-10x | 2-5x | 1-3% |
| Quant + Pruning | 8x | 3-5x | 1-3% |
| Quant + Pruning + Distillation | 10-20x | 5-10x | 2-5% |

---

## 4. Compression Pipeline Workflow

### 4.1 The Standard Pipeline

```
┌──────────────┐
│ Trained      │
│ FP32 Model   │
└──────┬───────┘
       │
       ▼
┌──────────────┐    Optional: train a smaller student model
│ Knowledge    │    using the original as teacher
│ Distillation │
└──────┬───────┘
       │
       ▼
┌──────────────┐    Remove redundant weights/channels
│ Pruning      │    then fine-tune to recover accuracy
│ + Fine-tune  │
└──────┬───────┘
       │
       ▼
┌──────────────┐    Reduce precision: FP32 → INT8 (or INT4)
│ Quantization │    PTQ (fast) or QAT (more accurate)
│              │
└──────┬───────┘
       │
       ▼
┌──────────────┐    Convert to runtime format:
│ Export &     │    ONNX, TFLite, TensorRT, CoreML
│ Optimize     │
└──────┬───────┘
       │
       ▼
┌──────────────┐    Measure latency, accuracy, memory
│ Benchmark    │    on target device
│ on Device    │
└──────────────┘
```

### 4.2 Pipeline Implementation

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class CompressionPipeline:
    """
    A step-by-step compression pipeline.
    Each step measures the model state before and after.
    """

    def __init__(self, model, calibration_loader, test_loader):
        self.model = model
        self.calibration_loader = calibration_loader
        self.test_loader = test_loader
        self.history = []

    def _snapshot(self, label):
        """Record model metrics at this stage."""
        size_mb = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / 1e6
        nonzero = sum(
            (p != 0).sum().item() for p in self.model.parameters()
        )
        total = sum(p.numel() for p in self.model.parameters())
        sparsity = 1.0 - (nonzero / total)
        self.history.append({
            "stage": label,
            "size_mb": size_mb,
            "total_params": total,
            "nonzero_params": nonzero,
            "sparsity": sparsity,
        })
        print(f"[{label}] Size: {size_mb:.1f} MB, "
              f"Params: {total:,}, Sparsity: {sparsity:.1%}")

    def step1_baseline(self):
        """Record baseline metrics."""
        self._snapshot("Baseline (FP32)")

    def step2_prune(self, amount=0.3):
        """Apply global unstructured pruning."""
        parameters_to_prune = []
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, "weight"))

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )

        # Make pruning permanent
        for module, name in parameters_to_prune:
            prune.remove(module, name)

        self._snapshot(f"Pruned ({amount:.0%})")

    def step3_quantize(self):
        """Apply dynamic quantization (INT8)."""
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8,
        )
        self._snapshot("Quantized (INT8)")

    def step4_export_onnx(self, output_path, input_shape=(1, 3, 224, 224)):
        """Export to ONNX format."""
        dummy_input = torch.randn(*input_shape)
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        import os
        file_size_mb = os.path.getsize(output_path) / 1e6
        print(f"[ONNX Export] File size: {file_size_mb:.1f} MB → {output_path}")

    def summary(self):
        """Print compression summary."""
        print("\n=== Compression Pipeline Summary ===")
        print(f"{'Stage':<25} {'Size(MB)':>10} {'Sparsity':>10}")
        print("-" * 47)
        for entry in self.history:
            print(f"{entry['stage']:<25} {entry['size_mb']:>10.1f} "
                  f"{entry['sparsity']:>10.1%}")
        if len(self.history) >= 2:
            ratio = self.history[0]["size_mb"] / max(self.history[-1]["size_mb"], 0.1)
            print(f"\nOverall compression ratio: {ratio:.1f}x")


# Usage example (pseudocode — requires actual model and data)
# model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
# pipeline = CompressionPipeline(model, calib_loader, test_loader)
# pipeline.step1_baseline()
# pipeline.step2_prune(amount=0.4)
# pipeline.step3_quantize()
# pipeline.step4_export_onnx("resnet18_compressed.onnx")
# pipeline.summary()
```

---

## 5. Choosing the Right Compression Strategy

### 5.1 Decision Framework

```
                    Start
                      │
          ┌───────────┴───────────┐
          │ Can you train/fine-   │
          │ tune the model?       │
          └───────────┬───────────┘
                 yes/  \no
                /       \
         ┌─────▼──┐  ┌──▼────────┐
         │ Budget  │  │ PTQ only  │
         │ for NAS?│  │ (Lesson 3)│
         └────┬────┘  └───────────┘
          yes/ \no
           /    \
    ┌─────▼──┐  ┌──▼──────────────┐
    │  NAS   │  │ Fine-tune based │
    │(L07)   │  │  approach:      │
    └────────┘  │ QAT + Pruning   │
                │ + Distillation  │
                └─────────────────┘
```

### 5.2 Strategy by Scenario

| Scenario | Recommended Strategy | Rationale |
|----------|---------------------|-----------|
| Quick deployment, no retraining | PTQ (INT8) | Zero-cost, 4x smaller |
| Need best accuracy, can retrain | QAT + structured pruning | Recovers accuracy via fine-tuning |
| Deploying to MCU (<1MB RAM) | Distillation → tiny model + INT8 | Need architecture change |
| New product, flexible timeline | Hardware-aware NAS + QAT | Optimal for specific hardware |
| LLM compression | GPTQ / AWQ + INT4 | Specialized for Transformer weights |
| Real-time video (>30 FPS) | Efficient architecture + TensorRT | Latency-focused optimization |

### 5.3 Practical Decision Code

```python
def recommend_compression(
    model_size_mb: float,
    target_size_mb: float,
    can_retrain: bool,
    target_hardware: str,
    accuracy_tolerance: float,  # maximum acceptable accuracy drop (%)
) -> list:
    """
    Recommend compression techniques based on constraints.

    Returns a list of recommended steps in order.
    """
    compression_needed = model_size_mb / target_size_mb
    steps = []

    # Step 1: Quantization is almost always beneficial
    if compression_needed >= 2:
        if can_retrain and accuracy_tolerance < 1.0:
            steps.append("Quantization-Aware Training (QAT) → INT8")
        else:
            steps.append("Post-Training Quantization (PTQ) → INT8")
        compression_needed /= 4  # INT8 gives ~4x reduction

    # Step 2: Pruning if more compression needed
    if compression_needed >= 1.5:
        if target_hardware in ("gpu", "npu"):
            steps.append("Structured pruning (channel) → 50% sparsity")
        else:
            steps.append("Unstructured pruning → 70% sparsity")
        compression_needed /= 2

    # Step 3: Distillation if still too large or architecture change needed
    if compression_needed >= 2 or target_size_mb < 5:
        steps.append("Knowledge distillation → smaller architecture")

    # Step 4: NAS if starting fresh
    if not steps or (can_retrain and target_size_mb < 10):
        steps.append("Consider hardware-aware NAS for optimal architecture")

    return steps


# Example: compress ResNet-50 for Raspberry Pi
steps = recommend_compression(
    model_size_mb=97,
    target_size_mb=10,
    can_retrain=True,
    target_hardware="cpu",
    accuracy_tolerance=2.0,
)
print("Recommended compression steps:")
for i, step in enumerate(steps, 1):
    print(f"  {i}. {step}")
```

---

## 6. Measuring Compression Quality

### 6.1 Key Metrics

Beyond accuracy and size, you should track:

```python
import time
import torch


class CompressionEvaluator:
    """Evaluate compressed model quality across multiple dimensions."""

    def __init__(self, original_model, compressed_model, test_loader, device="cpu"):
        self.original = original_model.to(device).eval()
        self.compressed = compressed_model.to(device).eval()
        self.test_loader = test_loader
        self.device = device

    def compare_accuracy(self):
        """Compare top-1 accuracy of original vs compressed model."""
        results = {}
        for name, model in [("original", self.original),
                            ("compressed", self.compressed)]:
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            results[name] = 100.0 * correct / total
        results["accuracy_drop"] = results["original"] - results["compressed"]
        return results

    def compare_latency(self, input_shape=(1, 3, 224, 224), num_runs=100):
        """Compare inference latency."""
        results = {}
        dummy = torch.randn(*input_shape, device=self.device)
        for name, model in [("original", self.original),
                            ("compressed", self.compressed)]:
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    model(dummy)

            # Measure
            times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start = time.perf_counter()
                    model(dummy)
                    times.append((time.perf_counter() - start) * 1000)

            results[name] = {
                "mean_ms": sum(times) / len(times),
                "min_ms": min(times),
                "max_ms": max(times),
            }
        results["speedup"] = (results["original"]["mean_ms"]
                              / results["compressed"]["mean_ms"])
        return results

    def compare_size(self):
        """Compare model parameter count and memory footprint."""
        results = {}
        for name, model in [("original", self.original),
                            ("compressed", self.compressed)]:
            params = sum(p.numel() for p in model.parameters())
            size_bytes = sum(
                p.numel() * p.element_size() for p in model.parameters()
            )
            nonzero = sum(
                (p != 0).sum().item() for p in model.parameters()
            )
            results[name] = {
                "params": params,
                "size_mb": size_bytes / 1e6,
                "nonzero_params": nonzero,
                "sparsity": 1.0 - nonzero / params if params > 0 else 0,
            }
        results["compression_ratio"] = (
            results["original"]["size_mb"]
            / max(results["compressed"]["size_mb"], 1e-6)
        )
        return results

    def output_fidelity(self, input_shape=(1, 3, 224, 224), num_samples=100):
        """
        Measure how closely the compressed model matches the original's outputs.
        High fidelity = compressed model preserves prediction ranking.
        """
        agreements = 0
        kl_divs = []
        with torch.no_grad():
            for i, (inputs, _) in enumerate(self.test_loader):
                if i >= num_samples:
                    break
                inputs = inputs.to(self.device)
                orig_out = torch.softmax(self.original(inputs), dim=1)
                comp_out = torch.softmax(self.compressed(inputs), dim=1)

                # Top-1 agreement
                agreements += (orig_out.argmax(1) == comp_out.argmax(1)).sum().item()

                # KL divergence
                kl = torch.nn.functional.kl_div(
                    comp_out.log(), orig_out, reduction="batchmean"
                )
                kl_divs.append(kl.item())

        total = min(num_samples * self.test_loader.batch_size,
                    len(self.test_loader.dataset))
        return {
            "top1_agreement": 100.0 * agreements / total,
            "mean_kl_divergence": sum(kl_divs) / len(kl_divs),
        }
```

### 6.2 Compression Quality Score

```python
def compression_quality_score(accuracy_drop, speedup, compression_ratio):
    """
    A composite score balancing accuracy retention, speed gain, and size reduction.
    Higher is better. Penalizes large accuracy drops exponentially.

    Score = (speedup * compression_ratio) / (1 + accuracy_drop)^2
    """
    penalty = (1 + accuracy_drop / 100) ** 2
    return (speedup * compression_ratio) / penalty


# Compare strategies
strategies = [
    ("PTQ INT8 only",           0.6,  3.0, 4.0),
    ("Pruning 50% + INT8",      1.5,  4.5, 6.0),
    ("Distillation + INT8",     2.0,  5.0, 8.0),
    ("NAS + QAT + pruning",     0.8, 6.0, 10.0),
]

print(f"{'Strategy':<30} {'Acc Drop':>9} {'Speedup':>8} {'Ratio':>6} {'Score':>7}")
print("-" * 62)
for name, drop, speed, ratio in strategies:
    score = compression_quality_score(drop, speed, ratio)
    print(f"{name:<30} {drop:>8.1f}% {speed:>7.1f}x {ratio:>5.0f}x {score:>7.1f}")
```

---

## 7. Common Pitfalls

### 7.1 Accuracy Collapse After Aggressive Compression

```python
# BAD: Applying extreme compression in one step
# This often causes accuracy to collapse
model_quantized = quantize(model, bits=4)        # 8x smaller but 5% accuracy drop
model_pruned = prune(model_quantized, amount=0.8) # Further 5x but accuracy crashes

# GOOD: Gradual compression with recovery fine-tuning between steps
model_pruned = prune(model, amount=0.5)
model_pruned = fine_tune(model_pruned, epochs=10)  # Recover accuracy
model_quantized = quantize(model_pruned, bits=8)   # Then quantize
# Result: similar compression with much less accuracy loss
```

### 7.2 Ignoring Hardware Compatibility

```python
# BAD: Unstructured pruning on mobile CPU
# Mobile CPUs cannot accelerate sparse matrix operations efficiently
# The model is 70% sparse but runs at the SAME speed!

# GOOD: Use structured (channel) pruning for mobile
# Removing entire channels directly reduces the Conv2d dimensions
# → Guaranteed speedup on any hardware
```

### 7.3 Not Measuring on Target Device

```python
# BAD: "My model is 10x smaller, so it must be 10x faster"
# Reality: memory bandwidth, cache effects, and operator support
# mean that size reduction != latency reduction

# GOOD: Always benchmark on actual target hardware
# results = benchmark_on_device(model, device="raspberry_pi")
```

---

## Summary

| Technique | Size Reduction | Accuracy Impact | Training Required | Best For |
|-----------|---------------|-----------------|-------------------|----------|
| **Quantization** | 2-8x | Minimal (0.1-1%) | Optional (PTQ vs QAT) | Universal first step |
| **Pruning** | 2-5x | Low-moderate | Yes (fine-tuning) | Reducing FLOPs |
| **Distillation** | 3-10x | Moderate (1-3%) | Yes (full training) | Architecture change |
| **NAS** | Optimal | Minimal | Yes (search + train) | New architecture design |

The **recommended order** for compression: Distillation (if architecture change needed) -> Pruning -> Quantization -> Export & Optimize.

---

## Exercises

### Exercise 1: Compression Analysis

Take a pretrained ResNet-18 from torchvision:
1. Measure its baseline size (MB), parameter count, and FLOPs
2. Apply `torch.quantization.quantize_dynamic` (INT8) and measure the same metrics
3. Apply 50% global L1 unstructured pruning (before quantization) and measure
4. Apply both pruning + quantization and measure
5. Create a table comparing all four configurations

### Exercise 2: Pareto Frontier

Using torchvision pretrained models (ResNet-18/34/50, MobileNetV2/V3, EfficientNet-B0/B1):
1. Record each model's parameter count, FP32 size, and reported ImageNet accuracy
2. Apply INT8 dynamic quantization to each and record the new size
3. Plot accuracy vs size for both FP32 and INT8 variants
4. Identify which models lie on the Pareto frontier

### Exercise 3: Strategy Recommendation

For each scenario below, recommend a compression pipeline and justify your choices:
1. Deploy a text classifier (BERT-Base, 440 MB) on an Android phone with 4 GB RAM
2. Deploy an object detector (YOLOv8-M, 50 MB) on a Jetson Nano at 30 FPS
3. Deploy a keyword spotter on an STM32 MCU with 512 KB SRAM

### Exercise 4: Pipeline Implementation

Using the `CompressionPipeline` class from Section 4.2:
1. Extend it with an `evaluate_accuracy` method that measures top-1 accuracy at each stage
2. Run the full pipeline on a ResNet-18 pretrained on CIFAR-10
3. Create a summary table showing size, sparsity, and accuracy at each stage
4. Determine the maximum pruning ratio before accuracy drops more than 2%

---

[Previous: Edge AI Fundamentals](./01_Edge_AI_Fundamentals.md) | [Overview](./00_Overview.md) | [Next: Quantization](./03_Quantization.md)

**License**: CC BY-NC 4.0
