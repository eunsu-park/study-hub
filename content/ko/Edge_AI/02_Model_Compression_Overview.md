# 레슨 2: 모델 압축 개요

[이전: Edge AI 기초](./01_Edge_AI_Fundamentals.md) | [다음: Quantization](./03_Quantization.md)

---

## 학습 목표

- 모델 압축 기법의 분류 체계를 이해합니다
- Pruning, quantization, distillation, 아키텍처 탐색을 상위 수준에서 비교합니다
- 압축 비율과 Pareto 곡선을 사용하여 정확도-크기 트레이드오프를 분석합니다
- 엣지 배포를 위한 다단계 압축 파이프라인을 설계합니다
- 주어진 시나리오에 어떤 압축 기법을 적용할지 평가합니다

---

## 1. 왜 모델을 압축하는가?

현대 딥러닝 모델은 크기가 큽니다. ResNet-50은 2,560만 개의 파라미터(FP32로 97 MB)를 가지고 있습니다. BERT-Base는 1억 1,000만 개의 파라미터(440 MB)를 가지고 있습니다. GPT-3는 1,750억 개의 파라미터(700 GB)를 가지고 있습니다. 이 중 어떤 것도 원래 형태로는 엣지 디바이스에 편안하게 맞지 않습니다.

**모델 압축**은 가능한 한 많은 정확도를 유지하면서 학습된 모델의 크기, 지연 시간, 연산 요구 사항을 줄입니다.

```
원본 모델                          압축된 모델
┌─────────────────┐               ┌─────────────────┐
│ ResNet-50       │    압축       │ ResNet-50       │
│ 97 MB (FP32)   │──────────────▶│ 24 MB (INT8)    │
│ 4.1 GFLOPS     │               │ 1.0 GFLOPS      │
│ 76.1% top-1    │               │ 75.5% top-1     │
│ 45ms (CPU)     │               │ 15ms (CPU)      │
└─────────────────┘               └─────────────────┘
     4배 작아지고, 3배 빨라지며, 0.6% 정확도 손실
```

### 1.1 과잉 파라미터화 가설

왜 모델을 압축할 수 있을까요? 신경망은 **과잉 파라미터화**되어 있기 때문입니다: 학습된 함수를 표현하는 데 필요한 것보다 훨씬 많은 파라미터를 가지고 있습니다. 이 중복성은 학습 중에는 유용하지만(최적화를 더 쉽게 만들어줌) 추론 중에는 낭비입니다.

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

## 2. 압축 분류 체계

모델 압축 기법에는 네 가지 주요 계열이 있습니다. 이들은 **상호 보완적**이며 — 결합하여 사용할 수 있고 종종 그래야 합니다.

```
                    모델 압축 기법
                              │
          ┌───────────┬───────┴───────┬──────────────┐
          │           │               │              │
    ┌─────▼─────┐ ┌──▼────────┐ ┌────▼─────┐ ┌─────▼──────┐
    │  Pruning  │ │Quantization│ │Knowledge │ │Architecture│
    │           │ │            │ │Distillat.│ │  Search    │
    └─────┬─────┘ └──┬────────┘ └────┬─────┘ └─────┬──────┘
          │          │               │              │
    중복된        수치            더 작은        효율적인
    가중치        정밀도를         student        아키텍처를
    또는 뉴런     줄임            모델을         자동으로
    을 제거      (FP32→INT8)     학습           설계
```

### 2.1 Quantization

**아이디어**: 가중치와 활성값의 수치 정밀도를 32비트 부동 소수점에서 더 낮은 비트폭(INT8, INT4, 또는 바이너리)으로 줄입니다.

| 정밀도 | 비트 | 크기 감소 | 일반적 정확도 하락 |
|-----------|------|---------------|----------------------|
| FP32 | 32 | 1x (기준) | 0% |
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

**사용 시기**: 거의 항상. Quantization은 노력 대비 효과 비율이 가장 좋은 가장 보편적으로 적용 가능한 압축 기법입니다.

**상세 내용**: [레슨 3 — Quantization](./03_Quantization.md)

### 2.2 Pruning

**아이디어**: 모델 정확도에 가장 적게 기여하는 가중치, 뉴런, 또는 전체 레이어를 제거합니다. 결과적으로 더 적은 연산을 필요로 하는 희소 모델이 됩니다.

```
Pruning 전 (Dense):              Pruning 후 (Sparse):
┌─────────────────────┐         ┌─────────────────────┐
│ ● ● ● ● ● ● ● ● ● │         │ ● ○ ● ○ ○ ● ○ ● ○ │
│ ● ● ● ● ● ● ● ● ● │         │ ○ ● ○ ○ ● ○ ● ○ ○ │
│ ● ● ● ● ● ● ● ● ● │         │ ● ○ ○ ● ○ ○ ○ ● ● │
│ ● ● ● ● ● ● ● ● ● │         │ ○ ○ ● ○ ● ● ○ ○ ○ │
└─────────────────────┘         └─────────────────────┘
  81개 파라미터 (100%)            30개 파라미터 (37%)
                                  → 63% 희소성
```

| 유형 | 세분성 | 속도 향상 | 하드웨어 지원 |
|------|------------|---------|-----------------|
| 비구조적 | 개별 가중치 | 희소 하드웨어 필요 | 제한적 (희소 텐서 코어) |
| 구조적 (채널) | 전체 필터/채널 | 모든 하드웨어에서 직접 속도 향상 | 우수 |
| 구조적 (블록) | N x M 블록 | 보통 | NVIDIA Ampere+ |

**사용 시기**: FLOPs를 줄여야 하고 pruning 후 미세 조정이 가능한 경우. 하드웨어 호환성을 위해 구조적 pruning이 선호됩니다.

**상세 내용**: [레슨 4 — Pruning](./04_Pruning.md)

### 2.3 Knowledge Distillation

**아이디어**: 작은 **student** 모델이 큰 **teacher** 모델의 동작을 모방하도록 학습시킵니다. Student는 하드 레이블만으로는 얻을 수 없는 더 풍부한 정보를 포함하는 teacher의 soft 확률 출력에서 학습합니다.

```
Teacher (대형)                     Student (소형)
┌───────────────┐                 ┌───────────────┐
│ ResNet-152    │   Soft Labels   │ MobileNetV2   │
│ 60M params   │────────────────▶│ 3.4M params   │
│ 78.3% acc    │  (Knowledge     │ 73.5% acc     │
│ 230 MB       │   Transfer)     │ 14 MB         │
└───────────────┘                 └───────────────┘

Distillation 없이: Student 단독으로 71.9% 정확도
Distillation 적용: Student가 73.5% 정확도 (+1.6%)
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

**사용 시기**: 새로운 (더 작은) 아키텍처를 학습시킬 여유가 있을 때. Quantization과 결합할 때 특히 효과적입니다.

**상세 내용**: [레슨 5 — Knowledge Distillation](./05_Knowledge_Distillation.md)

### 2.4 Neural Architecture Search (NAS)

**아이디어**: 효율적인 모델 아키텍처를 자동으로 탐색하며, 선택적으로 하드웨어 인식 제약 조건(지연 시간, 메모리, 에너지)을 포함합니다.

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

| 방법 | 탐색 비용 | 핵심 혁신 |
|--------|------------|----------------|
| NASNet (2017) | 48,000 GPU 시간 | RL 기반 컨트롤러 |
| DARTS (2019) | 4 GPU 일 | 미분 가능한 탐색 |
| EfficientNet (2019) | ~5,000 GPU 시간 | 복합 스케일링 |
| MnasNet (2019) | ~40,000 GPU 시간 | 하드웨어 인식 (지연 시간) |
| FBNet (2019) | 9 GPU 시간 | 미분 가능 + 하드웨어 인식 |
| OFA (2020) | 한 번만 학습 | Once-for-all 탄성 네트워크 |

**사용 시기**: 특정 하드웨어 타겟을 위한 새로운 아키텍처 계열을 설계할 때. 초기 비용이 높지만 최적의 결과를 제공합니다.

**상세 내용**: [레슨 7 — Neural Architecture Search](./07_Neural_Architecture_Search.md)

---

## 3. 압축 비율과 정확도 트레이드오프

### 3.1 압축 비율 이해

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

### 3.2 정확도-크기 Pareto 곡선

목표는 **Pareto 프론티어** 위의 모델을 찾는 것입니다: 다른 어떤 모델도 더 작으면서 동시에 더 정확하지 않은 모델의 집합입니다.

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

    ─── Pareto 프론티어 (바람직한 방향: 위쪽과 왼쪽)
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

### 3.3 기법 결합

압축 기법은 이점이 **곱셈적**입니다:

| 기법 | 크기 감소 | 속도 향상 | 정확도 하락 |
|-----------|---------------|---------|---------------|
| Quantization (INT8) 단독 | 4x | 2-3x | 0.1-1% |
| Pruning (50%) 단독 | 2x | 1.5-2x | 0.5-2% |
| Distillation 단독 | 3-10x | 2-5x | 1-3% |
| Quant + Pruning | 8x | 3-5x | 1-3% |
| Quant + Pruning + Distillation | 10-20x | 5-10x | 2-5% |

---

## 4. 압축 파이프라인 워크플로우

### 4.1 표준 파이프라인

```
┌──────────────┐
│ 학습된       │
│ FP32 모델    │
└──────┬───────┘
       │
       ▼
┌──────────────┐    선택: 원본을 teacher로 사용하여
│ Knowledge    │    더 작은 student 모델을 학습
│ Distillation │
└──────┬───────┘
       │
       ▼
┌──────────────┐    중복 가중치/채널을 제거한 후
│ Pruning      │    정확도 회복을 위해 미세 조정
│ + 미세 조정  │
└──────┬───────┘
       │
       ▼
┌──────────────┐    정밀도 감소: FP32 → INT8 (또는 INT4)
│ Quantization │    PTQ (빠름) 또는 QAT (더 정확)
│              │
└──────┬───────┘
       │
       ▼
┌──────────────┐    런타임 형식으로 변환:
│ 내보내기 &   │    ONNX, TFLite, TensorRT, CoreML
│ 최적화       │
└──────┬───────┘
       │
       ▼
┌──────────────┐    지연 시간, 정확도, 메모리
│ 타겟 디바이스│    측정
│ 에서 벤치마크│
└──────────────┘
```

### 4.2 파이프라인 구현

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

## 5. 적합한 압축 전략 선택

### 5.1 의사결정 프레임워크

```
                    시작
                      │
          ┌───────────┴───────────┐
          │ 모델을 학습/미세      │
          │ 조정할 수 있는가?     │
          └───────────┬───────────┘
                 예/    \아니오
                /       \
         ┌─────▼──┐  ┌──▼────────┐
         │ NAS    │  │ PTQ만     │
         │ 예산?  │  │ (레슨 3)  │
         └────┬────┘  └───────────┘
          예/  \아니오
           /    \
    ┌─────▼──┐  ┌──▼──────────────┐
    │  NAS   │  │ 미세 조정 기반  │
    │(L07)   │  │  접근법:        │
    └────────┘  │ QAT + Pruning   │
                │ + Distillation  │
                └─────────────────┘
```

### 5.2 시나리오별 전략

| 시나리오 | 권장 전략 | 근거 |
|----------|---------------------|-----------|
| 빠른 배포, 재학습 없음 | PTQ (INT8) | 비용 없음, 4배 축소 |
| 최고 정확도 필요, 재학습 가능 | QAT + 구조적 pruning | 미세 조정으로 정확도 회복 |
| MCU에 배포 (<1MB RAM) | Distillation → 소형 모델 + INT8 | 아키텍처 변경 필요 |
| 새 제품, 유연한 일정 | 하드웨어 인식 NAS + QAT | 특정 하드웨어에 최적 |
| LLM 압축 | GPTQ / AWQ + INT4 | Transformer 가중치에 특화 |
| 실시간 비디오 (>30 FPS) | 효율적 아키텍처 + TensorRT | 지연 시간 중심 최적화 |

### 5.3 실용적 의사결정 코드

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

## 6. 압축 품질 측정

### 6.1 핵심 메트릭

정확도와 크기 외에도 다음을 추적해야 합니다:

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

### 6.2 압축 품질 점수

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

## 7. 일반적인 실수

### 7.1 공격적인 압축 후 정확도 붕괴

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

### 7.2 하드웨어 호환성 무시

```python
# BAD: Unstructured pruning on mobile CPU
# Mobile CPUs cannot accelerate sparse matrix operations efficiently
# The model is 70% sparse but runs at the SAME speed!

# GOOD: Use structured (channel) pruning for mobile
# Removing entire channels directly reduces the Conv2d dimensions
# → Guaranteed speedup on any hardware
```

### 7.3 타겟 디바이스에서 측정하지 않음

```python
# BAD: "My model is 10x smaller, so it must be 10x faster"
# Reality: memory bandwidth, cache effects, and operator support
# mean that size reduction != latency reduction

# GOOD: Always benchmark on actual target hardware
# results = benchmark_on_device(model, device="raspberry_pi")
```

---

## 정리

| 기법 | 크기 감소 | 정확도 영향 | 학습 필요 여부 | 최적 용도 |
|-----------|---------------|-----------------|-------------------|----------|
| **Quantization** | 2-8x | 최소 (0.1-1%) | 선택 (PTQ vs QAT) | 보편적 첫 단계 |
| **Pruning** | 2-5x | 낮음-보통 | 예 (미세 조정) | FLOPs 감소 |
| **Distillation** | 3-10x | 보통 (1-3%) | 예 (전체 학습) | 아키텍처 변경 |
| **NAS** | 최적 | 최소 | 예 (탐색 + 학습) | 새 아키텍처 설계 |

**권장 압축 순서**: Distillation (아키텍처 변경이 필요한 경우) -> Pruning -> Quantization -> 내보내기 및 최적화.

---

## 연습 문제

### 연습 문제 1: 압축 분석

torchvision의 사전 학습된 ResNet-18을 사용하여:
1. 기준 크기(MB), 파라미터 수, FLOPs를 측정하십시오
2. `torch.quantization.quantize_dynamic` (INT8)을 적용하고 동일한 메트릭을 측정하십시오
3. 50% 전역 L1 비구조적 pruning을 (quantization 전에) 적용하고 측정하십시오
4. Pruning + quantization 모두 적용하고 측정하십시오
5. 네 가지 구성을 비교하는 표를 만드십시오

### 연습 문제 2: Pareto 프론티어

torchvision 사전 학습 모델(ResNet-18/34/50, MobileNetV2/V3, EfficientNet-B0/B1)을 사용하여:
1. 각 모델의 파라미터 수, FP32 크기, 보고된 ImageNet 정확도를 기록하십시오
2. 각각에 INT8 동적 quantization을 적용하고 새 크기를 기록하십시오
3. FP32와 INT8 변형 모두에 대해 정확도 vs 크기를 플롯하십시오
4. Pareto 프론티어 위에 있는 모델을 식별하십시오

### 연습 문제 3: 전략 추천

아래 각 시나리오에 대해 압축 파이프라인을 추천하고 선택을 정당화하십시오:
1. 4 GB RAM의 Android 폰에 텍스트 분류기(BERT-Base, 440 MB)를 배포
2. Jetson Nano에서 30 FPS로 객체 검출기(YOLOv8-M, 50 MB)를 배포
3. 512 KB SRAM의 STM32 MCU에 키워드 검출기를 배포

### 연습 문제 4: 파이프라인 구현

섹션 4.2의 `CompressionPipeline` 클래스를 사용하여:
1. 각 단계에서 top-1 정확도를 측정하는 `evaluate_accuracy` 메서드를 확장하십시오
2. CIFAR-10에서 사전 학습된 ResNet-18에 전체 파이프라인을 실행하십시오
3. 각 단계에서의 크기, 희소성, 정확도를 보여주는 요약 표를 만드십시오
4. 정확도가 2% 이상 떨어지기 전의 최대 pruning 비율을 결정하십시오

---

[이전: Edge AI 기초](./01_Edge_AI_Fundamentals.md) | [개요](./00_Overview.md) | [다음: Quantization](./03_Quantization.md)

**License**: CC BY-NC 4.0
