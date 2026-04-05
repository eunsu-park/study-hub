# 레슨 1: Edge AI 기초

[다음: 모델 압축 개요](./02_Model_Compression_Overview.md)

---

## 학습 목표

- 클라우드 추론과 엣지 추론의 차이를 이해합니다
- 엣지 배포의 지연 시간, 프라이버시, 비용, 신뢰성 트레이드오프를 분석합니다
- 클라우드에서 온디바이스까지의 엣지 컴퓨팅 스펙트럼을 파악합니다
- 산업 전반에 걸친 일반적인 Edge AI 사용 사례를 식별합니다
- Edge AI 설계를 형성하는 제약 조건(메모리, 전력, 연산)을 인식합니다

---

## 1. Edge AI란?

**Edge AI**는 데이터를 중앙 클라우드 서버로 보내는 대신, 데이터 소스에 위치하거나 근접한 디바이스에서 직접 머신러닝 추론(그리고 때로는 학습)을 수행하는 것을 말합니다. "엣지"는 공장의 강력한 GPU 워크스테이션부터 웨어러블 센서의 작은 마이크로컨트롤러까지 다양할 수 있습니다.

```
전통적인 클라우드 AI:
┌──────────┐    Network    ┌───────────┐    Network    ┌──────────┐
│  Sensor  │──────────────▶│   Cloud   │──────────────▶│  Action  │
│  Device  │   (50-200ms)  │  Server   │   (50-200ms)  │  Device  │
└──────────┘               └───────────┘               └──────────┘
                        총 지연 시간: 100-400ms+

Edge AI:
┌──────────────────────┐
│   Edge Device        │
│  ┌────────┐          │
│  │ Sensor │──▶ Model │──▶ Action     총 지연 시간: 1-50ms
│  └────────┘          │
└──────────────────────┘
```

핵심 통찰은 많은 AI 애플리케이션이 **실시간 응답**을 필요로 한다는 것입니다 — 자율주행 차량은 보행자를 식별하기 위해 클라우드 서버의 200ms 응답을 기다릴 수 없습니다. Edge AI는 데이터가 있는 곳으로 연산을 가져옵니다.

### 1.1 왜 지금 Edge AI가 중요한가

여러 수렴하는 트렌드가 Edge AI를 실용적으로 만들었습니다:

1. **모델 압축**: Quantization과 pruning 같은 기법으로 최소한의 정확도 손실로 모델을 4~10배 축소할 수 있습니다
2. **전용 하드웨어**: NPU, TPU, 신경망 가속기가 이제 스마트폰과 IoT 칩에 내장되어 있습니다
3. **프레임워크 지원**: ONNX Runtime, TensorFlow Lite, TensorRT가 배포를 접근 가능하게 합니다
4. **프라이버시 규제**: GDPR, HIPAA 및 기타 법률이 데이터를 디바이스에 보관하도록 장려합니다

```python
# A simple demonstration: comparing model sizes for edge deployment
import torch
import torch.nn as nn

class CloudModel(nn.Module):
    """A typical cloud-scale model: large and accurate."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class EdgeModel(nn.Module):
    """An edge-optimized model: smaller, faster, still useful."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Depthwise separable convolution (MobileNet-style)
            nn.Conv2d(3, 3, 3, padding=1, groups=3),  # Depthwise
            nn.Conv2d(3, 16, 1),                        # Pointwise
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16),
            nn.Conv2d(16, 32, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


cloud = CloudModel()
edge = EdgeModel()

cloud_params, _ = count_parameters(cloud)
edge_params, _ = count_parameters(edge)

print(f"Cloud model parameters: {cloud_params:,}")    # ~355K
print(f"Edge model  parameters: {edge_params:,}")      # ~1.3K
print(f"Compression ratio:      {cloud_params / edge_params:.1f}x")
```

---

## 2. 엣지 vs 클라우드 추론

### 2.1 클라우드 추론

클라우드 추론에서는 데이터가 디바이스에서 캡처되어 네트워크를 통해 강력한 서버로 전송되고, 처리된 후 결과가 다시 전송됩니다.

**장점**:
- 사실상 무제한의 연산 및 메모리
- 모델 업데이트가 쉬움 (서버에 재배포만 하면 됨)
- 가장 크고 정확한 모델을 실행할 수 있음
- 중앙 집중식 로깅 및 모니터링

**단점**:
- 네트워크 지연 시간 (왕복 50-200ms, 종종 그 이상)
- 안정적인 연결이 필요
- 지속적인 클라우드 연산 비용 (GPU 인스턴스는 비쌈)
- 프라이버시 위험: 원시 데이터가 디바이스를 떠남
- 대용량 데이터(비디오, 오디오)의 대역폭 비용

### 2.2 엣지 추론

엣지 추론에서는 모델이 로컬 디바이스에서 전적으로 실행됩니다.

**장점**:
- 초저지연 (1-50ms)
- 오프라인 작동 (네트워크 의존성 없음)
- 데이터가 디바이스에 남아 있음 (설계에 의한 프라이버시)
- 추론당 클라우드 비용 없음
- 감소된 대역폭 소비

**단점**:
- 제한된 연산, 메모리, 전력
- 모델 업데이트에 OTA(Over-the-Air) 배포 필요
- 프로덕션에서 디버깅 및 모니터링이 더 어려움
- 효율성을 위해 정확도를 희생할 수 있음

### 2.3 비교 표

| 항목 | 클라우드 추론 | 엣지 추론 |
|--------|----------------|----------------|
| **지연 시간** | 100-400ms | 1-50ms |
| **프라이버시** | 데이터가 디바이스를 떠남 | 데이터가 로컬에 유지 |
| **연결성** | 필수 | 불필요 |
| **모델 크기** | 무제한 | 제한적 (1MB-500MB) |
| **연산** | GPU/TPU 클러스터 | CPU/NPU/MCU |
| **전력** | 데이터 센터 전력 | 배터리/태양광 |
| **비용 모델** | 추론당 (API) | 디바이스당 (하드웨어) |
| **업데이트 주기** | 즉시 (서버 측) | OTA 배포 |
| **정확도** | 최신 기술 수준 | 양호 (SOTA의 1-5% 이내) |

### 2.4 하이브리드 아키텍처

실제로 많은 시스템은 **하이브리드 접근 방식**을 사용합니다: 실시간 결정을 위해 디바이스에서 경량 모델을 실행하고 복잡한 쿼리는 클라우드로 오프로드합니다.

```python
# Pseudocode: hybrid edge-cloud inference pipeline
class HybridInferencePipeline:
    """
    Run a fast edge model locally. If confidence is below threshold,
    fall back to a more accurate cloud model.
    """
    def __init__(self, edge_model, cloud_endpoint, confidence_threshold=0.85):
        self.edge_model = edge_model
        self.cloud_endpoint = cloud_endpoint
        self.threshold = confidence_threshold

    def predict(self, input_tensor):
        # Step 1: Run edge model (fast, local)
        with torch.no_grad():
            edge_output = self.edge_model(input_tensor)
            probabilities = torch.softmax(edge_output, dim=1)
            confidence, prediction = probabilities.max(dim=1)

        # Step 2: If confident, return edge result
        if confidence.item() >= self.threshold:
            return {
                "prediction": prediction.item(),
                "confidence": confidence.item(),
                "source": "edge",
                "latency_ms": 5,  # Typical edge latency
            }

        # Step 3: Otherwise, query cloud for higher accuracy
        cloud_result = self._query_cloud(input_tensor)
        return {
            "prediction": cloud_result["class"],
            "confidence": cloud_result["confidence"],
            "source": "cloud",
            "latency_ms": 150,  # Typical cloud latency
        }

    def _query_cloud(self, input_tensor):
        # In production, this would be an HTTP request to a cloud endpoint
        # requests.post(self.cloud_endpoint, data=serialize(input_tensor))
        return {"class": 0, "confidence": 0.99}  # Placeholder
```

---

## 3. 엣지 컴퓨팅 스펙트럼

Edge AI는 이진적(클라우드 또는 디바이스)이 아닙니다. 각각 다른 연산 예산을 가진 배포 대상의 **스펙트럼**이 존재합니다:

```
 Cloud              Fog/Edge Server       Gateway           On-Device
┌──────────┐       ┌──────────┐       ┌──────────┐       ┌──────────┐
│ GPU/TPU  │       │ Edge GPU │       │ SBC/SoC  │       │ MCU/NPU  │
│ Cluster  │       │ (Jetson) │       │ (RPi)    │       │ (STM32)  │
├──────────┤       ├──────────┤       ├──────────┤       ├──────────┤
│ >100 TOPS│       │ 5-30 TOPS│       │ 0.5-5TOPS│       │ <1 TOPS  │
│ >32GB RAM│       │ 4-16GB   │       │ 1-8GB    │       │ 256KB-2MB│
│ ~300W    │       │ 10-30W   │       │ 5-15W    │       │ 1-500mW  │
│ Unlimited│       │ ~$500    │       │ ~$50     │       │ ~$5      │
└──────────┘       └──────────┘       └──────────┘       └──────────┘
     │                   │                  │                  │
     ▼                   ▼                  ▼                  ▼
 ResNet-152          YOLOv8-L          MobileNetV3        TinyML
 GPT-4               ViT-B            EfficientNet-B0    (8-bit CNNs)
 Stable Diffusion   Whisper-medium    DistilBERT         Keyword spotting
```

### 3.1 배포 티어

| 티어 | 하드웨어 예시 | 연산 | 메모리 | 전력 | 대표 모델 |
|------|-----------------|---------|--------|-------|----------------|
| **클라우드** | A100 GPU, TPU v4 | >100 TOPS | >32 GB | ~300W | 풀사이즈 Transformer, 디퓨전 모델 |
| **엣지 서버** | Jetson AGX Orin, Intel NUC | 5-30 TOPS | 4-16 GB | 10-30W | YOLOv8, ViT-Base, Whisper |
| **게이트웨이** | Raspberry Pi 5, Google Coral | 0.5-5 TOPS | 1-8 GB | 5-15W | MobileNet, EfficientNet, DistilBERT |
| **온디바이스** | STM32, ESP32, Arduino | <1 TOPS | 256KB-2MB | 1-500mW | 8비트 CNN, 키워드 검출 |

### 3.2 적합한 티어 선택

```python
def recommend_deployment_tier(
    latency_requirement_ms: float,
    model_size_mb: float,
    power_budget_watts: float,
    connectivity: str,  # "always", "intermittent", "none"
    privacy_required: bool,
) -> str:
    """Simple heuristic for deployment tier selection."""

    if connectivity == "none" or privacy_required:
        # Must run on-device
        if model_size_mb < 1 and power_budget_watts < 0.5:
            return "on-device (MCU)"
        elif model_size_mb < 50 and power_budget_watts < 15:
            return "gateway (SBC)"
        else:
            return "edge server (GPU)"

    if latency_requirement_ms < 10:
        return "on-device or gateway"
    elif latency_requirement_ms < 50:
        return "gateway or edge server"
    elif latency_requirement_ms < 200:
        return "edge server or cloud"
    else:
        return "cloud"


# Example usage
tier = recommend_deployment_tier(
    latency_requirement_ms=20,
    model_size_mb=5,
    power_budget_watts=3,
    connectivity="intermittent",
    privacy_required=True,
)
print(f"Recommended tier: {tier}")
# Output: Recommended tier: gateway (SBC)
```

---

## 4. Edge AI 사용 사례

### 4.1 컴퓨터 비전

| 응용 분야 | 디바이스 | 모델 | 지연 시간 | 비고 |
|------------|--------|-------|---------|-------|
| 자율주행 | Jetson AGX Orin | YOLOv8 + 차선 검출 | <30ms | 안전 필수 |
| 스마트 카메라 (사람 감지) | Google Coral | MobileNet SSD | <10ms | 상시 가동, 저전력 |
| 산업 결함 검사 | Jetson Nano | EfficientNet-B0 | <50ms | 공장 현장, 클라우드 없음 |
| 드론 장애물 회피 | Raspberry Pi + Coral | 깊이 추정 CNN | <20ms | 무게 및 전력 제한 |

### 4.2 자연어 처리

| 응용 분야 | 디바이스 | 모델 | 지연 시간 | 비고 |
|------------|--------|-------|---------|-------|
| 키워드 검출 ("Hey Siri") | Apple Neural Engine | Tiny RNN/CNN | <5ms | 상시 가동, <1mW |
| 온디바이스 번역 | 스마트폰 NPU | DistilBERT / Marian | <100ms | 오프라인 가능 |
| 음성 비서 | 스마트 스피커 | Streaming ASR | <200ms | 웨이크워드 온디바이스, 나머지 하이브리드 |

### 4.3 센서 및 시계열

| 응용 분야 | 디바이스 | 모델 | 지연 시간 | 비고 |
|------------|--------|-------|---------|-------|
| 예측 유지보수 | MCU (STM32) | 소형 이상 탐지기 | <1ms | 진동/온도 센서 |
| 제스처 인식 | 웨어러블 (nRF52) | 1D CNN | <10ms | 가속도계 + 자이로스코프 |
| ECG 이상 감지 | 의료용 웨어러블 | Tiny LSTM | <5ms | FDA 규제 제약 |

### 4.4 코드 예제: 추론 지연 시간 측정

```python
import torch
import torch.nn as nn
import time


def benchmark_inference(model, input_shape, device="cpu", num_warmup=10, num_runs=100):
    """
    Measure inference latency on a given device.

    Args:
        model: PyTorch model
        input_shape: tuple, e.g., (1, 3, 224, 224)
        device: "cpu" or "cuda"
        num_warmup: warmup iterations (not measured)
        num_runs: measured iterations

    Returns:
        dict with mean, std, min, max latency in milliseconds
    """
    model = model.to(device).eval()
    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup — ensures CUDA kernels are compiled, caches are warm
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Synchronize before timing (important for CUDA)
    if device == "cuda":
        torch.cuda.synchronize()

    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

    latencies = torch.tensor(latencies)
    return {
        "mean_ms": latencies.mean().item(),
        "std_ms": latencies.std().item(),
        "min_ms": latencies.min().item(),
        "max_ms": latencies.max().item(),
        "p50_ms": latencies.median().item(),
        "p99_ms": latencies.quantile(0.99).item(),
    }


# Example: benchmark a simple model
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(16, 10),
)

results = benchmark_inference(model, (1, 3, 224, 224), device="cpu")
print(f"CPU Inference: {results['mean_ms']:.2f} +/- {results['std_ms']:.2f} ms")
print(f"  P50: {results['p50_ms']:.2f} ms, P99: {results['p99_ms']:.2f} ms")
```

---

## 5. Edge AI 제약 조건

제약 조건을 이해하는 것은 엣지에 적합한 모델을 설계하는 데 필수적입니다.

### 5.1 메모리 제약

엣지 디바이스는 클라우드 GPU에 비해 심각하게 제한된 RAM을 가지고 있습니다:

| 디바이스 | RAM | 모델 크기 제한 |
|--------|-----|-----------------|
| NVIDIA A100 (클라우드) | 80 GB HBM | 실질적 제한 없음 |
| Jetson AGX Orin | 32 GB 공유 | 모델에 ~8 GB |
| Raspberry Pi 5 | 8 GB | 모델에 ~2 GB |
| Google Coral | 1 GB | 모델에 ~100 MB |
| STM32H7 (MCU) | 1 MB SRAM | 모델에 ~500 KB |
| ESP32-S3 | 512 KB SRAM | 모델에 ~200 KB |

배포된 모델의 **메모리 분석**:

```python
def estimate_memory_usage(model, input_shape, dtype_bytes=4):
    """
    Estimate peak memory during inference.

    Components:
    1. Model weights (parameters)
    2. Activations (intermediate tensors during forward pass)
    3. Input/output buffers
    """
    # 1. Model weights
    param_memory = sum(p.numel() * dtype_bytes for p in model.parameters())

    # 2. Estimate activation memory (rough heuristic)
    # In practice, use torch.cuda.max_memory_allocated() for precise measurement
    dummy = torch.randn(*input_shape)
    activation_sizes = []

    hooks = []
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            activation_sizes.append(output.numel() * dtype_bytes)

    for module in model.modules():
        hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    activation_memory = sum(activation_sizes)

    # 3. Input/output
    io_memory = dummy.numel() * dtype_bytes * 2  # input + output

    total = param_memory + activation_memory + io_memory
    return {
        "parameters_mb": param_memory / 1e6,
        "activations_mb": activation_memory / 1e6,
        "io_mb": io_memory / 1e6,
        "total_mb": total / 1e6,
    }


# Example
model = torch.hub.load("pytorch/vision", "mobilenet_v2", weights=None)
mem = estimate_memory_usage(model, (1, 3, 224, 224))
print(f"Parameters:  {mem['parameters_mb']:.1f} MB")
print(f"Activations: {mem['activations_mb']:.1f} MB")
print(f"Total:       {mem['total_mb']:.1f} MB")
```

### 5.2 연산 제약

연산 능력은 **TOPS**(초당 테라 연산) 또는 **GFLOPS**로 측정됩니다:

```
연산 예산 vs 모델 요구 사항:

Device Compute:    MCU       RPi       Coral     Jetson    A100
                   0.01      0.5       4         275       312
                   TOPS      TOPS      TOPS      TOPS      TOPS
                    │         │         │          │         │
Model Requirement:  │         │         │          │         │
  Keyword CNN       ▼         │         │          │         │
  (0.005 TOPS)     OK        │         │          │         │
                              │         │          │         │
  MobileNetV2                 ▼         │          │         │
  (0.3 TOPS)                 OK        │          │         │
                                        │          │         │
  YOLOv8-S                              ▼          │         │
  (3 TOPS)                             OK         │         │
                                                    │         │
  ViT-Large                                         ▼         │
  (60 TOPS)                                        OK        │
                                                              │
  GPT-3 (175B)                                                ▼
  (300+ TOPS)                                                OK
```

### 5.3 전력 제약

전력은 배터리로 작동하는 엣지 디바이스에서 가장 제약이 큰 요소인 경우가 많습니다:

| 디바이스 유형 | 전력 예산 | 배터리 수명 목표 | 시사점 |
|----------------|-------------|-------------------|-------------|
| 상시 가동 센서 | <1 mW | 수년 (코인 셀) | 소형 모델만, 드문 추론 |
| 웨어러블 | 10-100 mW | 1-7일 | 소형 CNN, 듀티 사이클링 |
| 스마트폰 | 1-5 W | 1일 | MobileNet 수준 모델 |
| 드론 / 로봇 | 5-30 W | 30-60분 | 더 큰 모델, 그러나 여전히 제한 |
| 엣지 서버 | 30-300 W | AC 전원 | 풀사이즈 모델 |

### 5.4 대역폭 제약

원시 데이터를 클라우드로 전송하면 대역폭이 소모되고 비용이 발생합니다:

```python
def calculate_bandwidth_cost(
    data_rate_mbps: float,
    hours_per_day: float,
    cost_per_gb: float = 0.09,  # AWS egress pricing
) -> dict:
    """Calculate monthly bandwidth cost for cloud inference."""
    daily_gb = (data_rate_mbps * hours_per_day * 3600) / (8 * 1024)
    monthly_gb = daily_gb * 30
    monthly_cost = monthly_gb * cost_per_gb
    return {
        "daily_gb": daily_gb,
        "monthly_gb": monthly_gb,
        "monthly_cost_usd": monthly_cost,
    }

# Example: 1080p camera streaming at 5 Mbps, 24/7
cost = calculate_bandwidth_cost(data_rate_mbps=5, hours_per_day=24)
print(f"Monthly data: {cost['monthly_gb']:.0f} GB")
print(f"Monthly cost: ${cost['monthly_cost_usd']:.2f}")
# Monthly data: 1,582 GB -> Monthly cost: $142.38

# With edge AI: only send alerts (maybe 1 MB/day)
# Monthly cost: ~$0.003
```

---

## 6. Edge AI 개발 워크플로우

일반적인 Edge AI 프로젝트는 다음 파이프라인을 따릅니다:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   1. 학습    │───▶│ 2. 압축     │───▶│  3. 내보내기 │───▶│  4. 배포    │
│  (Cloud GPU)│    │  (Quantize, │    │ (ONNX, TF   │    │  (Device)   │
│             │    │   Prune,    │    │  Lite, etc.) │    │             │
│             │    │   Distill)  │    │              │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
  풀 정밀도           압축된              최적화된            엣지 디바이스에서
  FP32 모델         INT8/INT4           런타임              실행
  (100 MB)          (25 MB)             (20 MB)            (실시간)
```

```python
# High-level overview of the edge AI pipeline in PyTorch
import torch
import torch.quantization as quant

# Step 1: Train a model (covered in Deep_Learning)
model = train_model(train_loader, epochs=100)

# Step 2: Compress — e.g., post-training quantization
model.eval()
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Step 3: Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    quantized_model,
    dummy_input,
    "model_edge.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
)

# Step 4: Deploy with ONNX Runtime
import onnxruntime as ort

session = ort.InferenceSession("model_edge.onnx")
result = session.run(None, {"input": dummy_input.numpy()})
```

---

## 7. Edge AI 핵심 메트릭

Edge AI 솔루션을 평가할 때 다음 메트릭을 추적합니다:

| 메트릭 | 설명 | 측정 방법 |
|--------|-------------|---------------|
| **정확도** | 태스크별 (top-1, mAP, F1) | 테스트 세트에서 표준 평가 |
| **지연 시간** | 엔드투엔드 추론 시간 (ms) | 벽시계 시간, p50/p99 |
| **처리량** | 초당 추론 수 (FPS) | 지속적 부하 테스트 |
| **모델 크기** | 디스크/플래시 사용량 (MB) | 내보낸 모델의 파일 크기 |
| **피크 RAM** | 추론 중 최대 메모리 | 프로파일러 또는 메모리 훅 |
| **전력** | 추론당 에너지 (mJ) | 하드웨어 전력 모니터 |
| **MAC/FLOPs** | 연산 비용 | `thop` 또는 `fvcore` 라이브러리 |

```python
# Measuring FLOPs with thop
from thop import profile

model = torch.hub.load("pytorch/vision", "mobilenet_v2", weights=None)
dummy = torch.randn(1, 3, 224, 224)

flops, params = profile(model, inputs=(dummy,), verbose=False)
print(f"FLOPs:  {flops / 1e9:.2f} GFLOPs")
print(f"Params: {params / 1e6:.2f} M")
# MobileNetV2: ~0.3 GFLOPs, 3.4M params
```

---

## 8. Edge AI 에코시스템 개요

Edge AI 에코시스템은 프레임워크, 런타임, 하드웨어에 걸쳐 있습니다:

```
Training Frameworks          Export Formats          Inference Runtimes
┌───────────────┐           ┌──────────────┐       ┌──────────────────┐
│ PyTorch       │──export──▶│ ONNX         │──────▶│ ONNX Runtime     │
│ TensorFlow    │──export──▶│ TFLite       │──────▶│ TFLite Runtime   │
│ JAX           │           │ TorchScript  │──────▶│ LibTorch         │
└───────────────┘           │ SavedModel   │       │ TensorRT         │
                            └──────────────┘       │ OpenVINO         │
                                                   │ CoreML           │
    Hardware Targets                               │ NNAPI            │
    ┌───────────────────────────────┐              │ TFLite Micro     │
    │ NVIDIA Jetson (TensorRT)      │              └──────────────────┘
    │ Google Coral (Edge TPU)       │
    │ Intel (OpenVINO + Movidius)   │
    │ Qualcomm (SNPE / QNN)        │
    │ Apple (CoreML + ANE)         │
    │ ARM (CMSIS-NN, ARM NN)       │
    │ Microcontrollers (TFLite µ)  │
    └───────────────────────────────┘
```

---

## 정리

- **Edge AI**는 디바이스에서 ML 추론을 실행하며, 모델 크기를 줄이는 대신 지연 시간, 프라이버시, 오프라인 기능을 확보합니다
- **엣지 컴퓨팅 스펙트럼**은 강력한 엣지 서버(Jetson)부터 소형 마이크로컨트롤러(STM32)까지 범위가 있습니다
- 핵심 트레이드오프: **지연 시간 vs 정확도**, **프라이버시 vs 편의성**, **비용 vs 기능**
- 엣지 디바이스는 **메모리**(KB~GB), **연산**(GFLOPS~TOPS), **전력**(mW~W)에 의해 제약됩니다
- **Edge AI 파이프라인**은 학습, 압축, 내보내기, 디바이스별 배포로 구성됩니다
- **하이브리드 아키텍처**는 속도를 위한 온디바이스 추론과 정확도를 위한 클라우드 폴백을 결합합니다

---

## 연습 문제

### 연습 문제 1: 제약 조건 분석

실제 Edge AI 응용 분야(예: 스마트 도어벨, 산업용 로봇, 의료용 웨어러블)를 선택하십시오. 선택한 응용 분야에 대해:
1. 배포 티어(클라우드, 엣지 서버, 게이트웨이, 온디바이스)를 식별하십시오
2. 제약 조건을 나열하십시오: 최대 지연 시간, 가용 메모리, 전력 예산, 연결성
3. 적합한 모델 아키텍처와 압축 전략을 제안하십시오

### 연습 문제 2: 지연 시간 벤치마킹

섹션 4의 `benchmark_inference` 함수를 사용하여:
1. CPU에서 `torchvision.models.resnet18`을 벤치마킹하십시오
2. CPU에서 `torchvision.models.mobilenet_v2`를 벤치마킹하십시오
3. 지연 시간, 파라미터 수, FLOPs를 비교하십시오
4. ImageNet top-1 정확도(ResNet-18: 69.8%, MobileNetV2: 71.9%)를 사용하여 "밀리초당 정확도"를 계산하십시오

### 연습 문제 3: 메모리 추정

1. `estimate_memory_usage` 함수를 사용하여 ResNet-50, MobileNetV2, EfficientNet-B0의 메모리 사용량을 비교하십시오
2. 각 모델에 대해 어떤 엣지 디바이스 티어가 호스팅할 수 있는지 결정하십시오 (섹션 5.1의 메모리 표 참조)
3. 배치 크기가 피크 메모리에 어떤 영향을 미치는지 확인하십시오. 배치 크기 1, 4, 16, 32로 측정하십시오

### 연습 문제 4: 대역폭 비용 분석

100대의 보안 카메라 각각이 1080p 비디오를 4 Mbps로 24시간 연중무휴 스트리밍합니다.
1. 모든 추론이 클라우드에서 수행될 경우 월간 대역폭 비용을 계산하십시오
2. Edge AI가 카메라당 일일 업로드를 10 MB(알림 클립만)로 줄인 경우 비용을 계산하십시오
3. 각 엣지 디바이스 비용이 $150일 때 손익분기점(월 단위)은 무엇입니까?

---

[개요](./00_Overview.md) | [다음: 모델 압축 개요](./02_Model_Compression_Overview.md)

**License**: CC BY-NC 4.0
