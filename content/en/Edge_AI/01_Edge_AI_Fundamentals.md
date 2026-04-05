# Lesson 1: Edge AI Fundamentals

[Next: Model Compression Overview](./02_Model_Compression_Overview.md)

---

## Learning Objectives

- Understand the distinction between cloud inference and edge inference
- Analyze the latency, privacy, cost, and reliability tradeoffs of edge deployment
- Map the edge computing spectrum from cloud to on-device
- Identify common edge AI use cases across industries
- Recognize the constraints (memory, power, compute) that shape edge AI design

---

## 1. What is Edge AI?

**Edge AI** refers to running machine learning inference (and sometimes training) directly on devices located at or near the data source, rather than sending data to a centralized cloud server. The "edge" can be anything from a powerful GPU workstation in a factory to a tiny microcontroller in a wearable sensor.

```
Traditional Cloud AI:
┌──────────┐    Network    ┌───────────┐    Network    ┌──────────┐
│  Sensor  │──────────────▶│   Cloud   │──────────────▶│  Action  │
│  Device  │   (50-200ms)  │  Server   │   (50-200ms)  │  Device  │
└──────────┘               └───────────┘               └──────────┘
                        Total latency: 100-400ms+

Edge AI:
┌──────────────────────┐
│   Edge Device        │
│  ┌────────┐          │
│  │ Sensor │──▶ Model │──▶ Action     Total latency: 1-50ms
│  └────────┘          │
└──────────────────────┘
```

The fundamental insight is that many AI applications require **real-time responses** — an autonomous vehicle cannot wait 200ms for a cloud server to identify a pedestrian. Edge AI brings the computation to where the data lives.

### 1.1 Why Edge AI Matters Now

Several converging trends have made edge AI practical:

1. **Model Compression**: Techniques like quantization and pruning can shrink models by 4-10x with minimal accuracy loss
2. **Specialized Hardware**: NPUs, TPUs, and neural accelerators are now embedded in smartphones and IoT chips
3. **Framework Support**: ONNX Runtime, TensorFlow Lite, and TensorRT make deployment accessible
4. **Privacy Regulations**: GDPR, HIPAA, and other laws incentivize keeping data on-device

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

## 2. Edge vs Cloud Inference

### 2.1 Cloud Inference

In cloud inference, data is captured on-device, transmitted over the network to a powerful server, processed, and the result is sent back.

**Advantages**:
- Virtually unlimited compute and memory
- Easy to update models (just redeploy on the server)
- Can run the largest, most accurate models
- Centralized logging and monitoring

**Disadvantages**:
- Network latency (50-200ms round trip, often more)
- Requires reliable connectivity
- Ongoing cloud compute costs (GPU instances are expensive)
- Privacy risk: raw data leaves the device
- Bandwidth costs for high-volume data (video, audio)

### 2.2 Edge Inference

In edge inference, the model runs entirely on the local device.

**Advantages**:
- Ultra-low latency (1-50ms)
- Works offline (no network dependency)
- Data stays on-device (privacy by design)
- No per-inference cloud cost
- Reduced bandwidth consumption

**Disadvantages**:
- Limited compute, memory, and power
- Model updates require OTA (over-the-air) deployment
- Harder to debug and monitor in production
- May sacrifice accuracy for efficiency

### 2.3 Comparison Table

| Factor | Cloud Inference | Edge Inference |
|--------|----------------|----------------|
| **Latency** | 100-400ms | 1-50ms |
| **Privacy** | Data leaves device | Data stays local |
| **Connectivity** | Required | Not required |
| **Model Size** | Unlimited | Constrained (1MB-500MB) |
| **Compute** | GPU/TPU clusters | CPU/NPU/MCU |
| **Power** | Data center power | Battery/solar |
| **Cost Model** | Per-inference (API) | Per-device (hardware) |
| **Update Cycle** | Instant (server-side) | OTA deployment |
| **Accuracy** | State-of-the-art | Good (within 1-5% of SOTA) |

### 2.4 Hybrid Architecture

In practice, many systems use a **hybrid approach**: run a lightweight model on-device for real-time decisions and offload complex queries to the cloud.

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

## 3. The Edge Computing Spectrum

Edge AI is not binary (cloud or device). There is a **spectrum** of deployment targets, each with different compute budgets:

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

### 3.1 Deployment Tiers

| Tier | Example Hardware | Compute | Memory | Power | Typical Models |
|------|-----------------|---------|--------|-------|----------------|
| **Cloud** | A100 GPU, TPU v4 | >100 TOPS | >32 GB | ~300W | Full-size Transformers, diffusion models |
| **Edge Server** | Jetson AGX Orin, Intel NUC | 5-30 TOPS | 4-16 GB | 10-30W | YOLOv8, ViT-Base, Whisper |
| **Gateway** | Raspberry Pi 5, Google Coral | 0.5-5 TOPS | 1-8 GB | 5-15W | MobileNet, EfficientNet, DistilBERT |
| **On-Device** | STM32, ESP32, Arduino | <1 TOPS | 256KB-2MB | 1-500mW | 8-bit CNNs, keyword spotting |

### 3.2 Choosing the Right Tier

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

## 4. Edge AI Use Cases

### 4.1 Computer Vision

| Application | Device | Model | Latency | Notes |
|------------|--------|-------|---------|-------|
| Autonomous driving | Jetson AGX Orin | YOLOv8 + lane detection | <30ms | Safety-critical |
| Smart camera (person detection) | Google Coral | MobileNet SSD | <10ms | Always-on, low power |
| Industrial defect inspection | Jetson Nano | EfficientNet-B0 | <50ms | Factory floor, no cloud |
| Drone obstacle avoidance | Raspberry Pi + Coral | Depth estimation CNN | <20ms | Weight and power constrained |

### 4.2 Natural Language Processing

| Application | Device | Model | Latency | Notes |
|------------|--------|-------|---------|-------|
| Keyword spotting ("Hey Siri") | Apple Neural Engine | Tiny RNN/CNN | <5ms | Always-on, <1mW |
| On-device translation | Smartphone NPU | DistilBERT / Marian | <100ms | Offline capable |
| Voice assistant | Smart speaker | Streaming ASR | <200ms | Wake-word on-device, rest hybrid |

### 4.3 Sensor and Time-Series

| Application | Device | Model | Latency | Notes |
|------------|--------|-------|---------|-------|
| Predictive maintenance | MCU (STM32) | Tiny anomaly detector | <1ms | Vibration/temperature sensors |
| Gesture recognition | Wearable (nRF52) | 1D CNN | <10ms | Accelerometer + gyroscope |
| ECG anomaly detection | Medical wearable | Tiny LSTM | <5ms | FDA regulatory constraints |

### 4.4 Code Example: Measuring Inference Latency

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

## 5. Edge AI Constraints

Understanding constraints is essential for designing edge-appropriate models.

### 5.1 Memory Constraints

Edge devices have severely limited RAM compared to cloud GPUs:

| Device | RAM | Model Size Limit |
|--------|-----|-----------------|
| NVIDIA A100 (cloud) | 80 GB HBM | No practical limit |
| Jetson AGX Orin | 32 GB shared | ~8 GB for model |
| Raspberry Pi 5 | 8 GB | ~2 GB for model |
| Google Coral | 1 GB | ~100 MB for model |
| STM32H7 (MCU) | 1 MB SRAM | ~500 KB for model |
| ESP32-S3 | 512 KB SRAM | ~200 KB for model |

**Memory breakdown** for a deployed model:

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

### 5.2 Compute Constraints

Compute capability is measured in **TOPS** (Tera Operations Per Second) or **GFLOPS**:

```
Compute Budget vs Model Requirements:

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

### 5.3 Power Constraints

Power is often the most binding constraint for battery-operated edge devices:

| Device Category | Power Budget | Battery Life Target | Implication |
|----------------|-------------|-------------------|-------------|
| Always-on sensor | <1 mW | Years (coin cell) | Only tiny models, infrequent inference |
| Wearable | 10-100 mW | 1-7 days | Small CNNs, duty cycling |
| Smartphone | 1-5 W | 1 day | MobileNet-class models |
| Drone / robot | 5-30 W | 30-60 min | Larger models, but still constrained |
| Edge server | 30-300 W | AC powered | Full-size models |

### 5.4 Bandwidth Constraints

Sending raw data to the cloud consumes bandwidth and incurs costs:

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

## 6. The Edge AI Development Workflow

A typical edge AI project follows this pipeline:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   1. Train  │───▶│ 2. Compress │───▶│  3. Export   │───▶│  4. Deploy  │
│  (Cloud GPU)│    │  (Quantize, │    │ (ONNX, TF   │    │  (Device)   │
│             │    │   Prune,    │    │  Lite, etc.) │    │             │
│             │    │   Distill)  │    │              │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
  Full-precision     Compressed         Optimized          Running on
  FP32 model        INT8/INT4           runtime            edge device
  (100 MB)          (25 MB)             (20 MB)            (real-time)
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

## 7. Key Metrics for Edge AI

When evaluating an edge AI solution, track these metrics:

| Metric | Description | How to Measure |
|--------|-------------|---------------|
| **Accuracy** | Task-specific (top-1, mAP, F1) | Standard evaluation on test set |
| **Latency** | End-to-end inference time (ms) | Wall-clock time, p50/p99 |
| **Throughput** | Inferences per second (FPS) | Sustained load test |
| **Model Size** | Disk/flash footprint (MB) | File size of exported model |
| **Peak RAM** | Maximum memory during inference | Profiler or memory hooks |
| **Power** | Energy per inference (mJ) | Hardware power monitor |
| **MACs/FLOPs** | Computational cost | `thop` or `fvcore` library |

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

## 8. Edge AI Ecosystem Overview

The edge AI ecosystem spans frameworks, runtimes, and hardware:

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

## Summary

- **Edge AI** runs ML inference on-device, trading model size for latency, privacy, and offline capability
- The **edge computing spectrum** ranges from powerful edge servers (Jetson) to tiny microcontrollers (STM32)
- Key tradeoffs: **latency vs accuracy**, **privacy vs convenience**, **cost vs capability**
- Edge devices are constrained by **memory** (KB to GB), **compute** (GFLOPS to TOPS), and **power** (mW to W)
- The **edge AI pipeline** involves training, compression, export, and device-specific deployment
- **Hybrid architectures** combine on-device inference for speed with cloud fallback for accuracy

---

## Exercises

### Exercise 1: Constraint Analysis

Pick a real-world edge AI application (e.g., smart doorbell, industrial robot, medical wearable). For your chosen application:
1. Identify the deployment tier (cloud, edge server, gateway, on-device)
2. List the constraints: maximum latency, available memory, power budget, connectivity
3. Suggest an appropriate model architecture and compression strategy

### Exercise 2: Latency Benchmarking

Using the `benchmark_inference` function from Section 4:
1. Benchmark `torchvision.models.resnet18` on CPU
2. Benchmark `torchvision.models.mobilenet_v2` on CPU
3. Compare their latency, parameter count, and FLOPs
4. Calculate the "accuracy per millisecond" using ImageNet top-1 accuracy (ResNet-18: 69.8%, MobileNetV2: 71.9%)

### Exercise 3: Memory Estimation

1. Using the `estimate_memory_usage` function, compare the memory footprint of ResNet-50, MobileNetV2, and EfficientNet-B0
2. For each model, determine which edge device tier(s) could host it (refer to the memory table in Section 5.1)
3. How does batch size affect peak memory? Measure with batch sizes 1, 4, 16, and 32

### Exercise 4: Bandwidth Cost Analysis

A fleet of 100 security cameras each streams 1080p video at 4 Mbps, 24/7.
1. Calculate the monthly bandwidth cost if all inference is done in the cloud
2. Calculate the cost if edge AI reduces uploads to 10 MB/day per camera (alert clips only)
3. What is the break-even point (in months) if each edge device costs $150?

---

[Overview](./00_Overview.md) | [Next: Model Compression Overview](./02_Model_Compression_Overview.md)

**License**: CC BY-NC 4.0
