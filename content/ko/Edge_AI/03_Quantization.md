# 레슨 3: Quantization

[이전: 모델 압축 개요](./02_Model_Compression_Overview.md) | [다음: Pruning](./04_Pruning.md)

---

## 학습 목표

- Quantization의 수학적 기초(아핀 매핑, scale, zero-point)를 이해합니다
- Post-Training Quantization(PTQ)과 Quantization-Aware Training(QAT)을 비교합니다
- 합성곱 및 선형 레이어에 대한 INT8 및 INT4 quantization을 구현합니다
- 대칭 vs 비대칭, per-tensor vs per-channel quantization을 구분합니다
- Quantization 오류를 최소화하기 위한 캘리브레이션 방법을 적용합니다
- 정확도와 효율성의 균형을 맞추는 혼합 정밀도 전략을 설계합니다

---

## 1. Quantization이란?

Quantization은 연속적인 부동 소수점 값을 이산적인 정수 집합으로 매핑합니다. 신경망의 맥락에서, 가중치와 활성값의 정밀도를 32비트 부동 소수점(FP32)에서 더 낮은 비트폭, 일반적으로 8비트 정수(INT8) 또는 4비트 정수(INT4)로 줄입니다.

```
FP32 Weight:  0.0372  →  Quantize  →  INT8: 19   (scale=0.002, zero_point=0)
FP32 Weight: -0.1543  →  Quantize  →  INT8: -77  (scale=0.002, zero_point=0)

FP32 range: [-3.4e38, 3.4e38]  →  INT8 range: [-128, 127]
Storage: 4 bytes per value      →  Storage: 1 byte per value (4x smaller)
```

### 1.1 왜 Quantization이 작동하는가

신경망은 가중치의 작은 변동에 강건합니다. FP32 값을 INT8로 반올림할 때 발생하는 quantization 오류는 일반적으로 잘 학습된 모델의 노이즈 허용 범위 내에 있습니다.

```python
import torch
import torch.nn as nn


def demonstrate_quantization_robustness():
    """
    Show that adding small noise (simulating quantization error)
    has minimal effect on model predictions.
    """
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
    )
    model.eval()

    # Random input
    x = torch.randn(1, 100)

    # Original prediction
    with torch.no_grad():
        original_output = model(x)

    # Add quantization-like noise to weights
    with torch.no_grad():
        for param in model.parameters():
            noise_scale = param.abs().max() / 128  # INT8-level noise
            noise = torch.randn_like(param) * noise_scale
            param.add_(noise)

    # Prediction after noise
    with torch.no_grad():
        noisy_output = model(x)

    # Compare
    diff = (original_output - noisy_output).abs()
    print(f"Max absolute difference:  {diff.max().item():.6f}")
    print(f"Mean absolute difference: {diff.mean().item():.6f}")
    print(f"Original argmax:  {original_output.argmax().item()}")
    print(f"Noisy argmax:     {noisy_output.argmax().item()}")
    print(f"Prediction match: {original_output.argmax() == noisy_output.argmax()}")


demonstrate_quantization_robustness()
```

---

## 2. Quantization 수학

### 2.1 아핀 Quantization 매핑

핵심 수식은 부동 소수점 값 `r`을 정수 `q`로 매핑합니다:

```
Quantize:    q = round(r / S) + Z
Dequantize:  r_approx = (q - Z) * S

where:
  r = 실수 부동 소수점 수
  q = 양자화된 정수
  S = 스케일 팩터 (float)
  Z = 영점 (integer)
```

```python
import torch


def quantize_tensor(tensor, num_bits=8, symmetric=True):
    """
    Quantize a floating-point tensor to fixed-point integers.

    Args:
        tensor: FP32 tensor to quantize
        num_bits: Target bit-width (8, 4, or 2)
        symmetric: If True, zero_point = 0 and range is symmetric

    Returns:
        quantized tensor (int), scale, zero_point
    """
    if symmetric:
        # Symmetric: range is [-2^(n-1)+1, 2^(n-1)-1], zero_point = 0
        qmin = -(2 ** (num_bits - 1)) + 1  # -127 for INT8
        qmax = 2 ** (num_bits - 1) - 1      #  127 for INT8

        abs_max = tensor.abs().max()
        scale = abs_max / qmax
        zero_point = 0
    else:
        # Asymmetric: range is [0, 2^n - 1], zero_point can be non-zero
        qmin = 0
        qmax = 2 ** num_bits - 1  # 255 for INT8

        rmin = tensor.min()
        rmax = tensor.max()
        scale = (rmax - rmin) / (qmax - qmin)
        zero_point = round((-rmin / scale).item())
        zero_point = max(qmin, min(qmax, zero_point))

    # Quantize
    q = torch.clamp(torch.round(tensor / scale) + zero_point, qmin, qmax).to(torch.int8)

    return q, scale, zero_point


def dequantize_tensor(q_tensor, scale, zero_point):
    """Convert quantized integers back to floating-point approximation."""
    return (q_tensor.float() - zero_point) * scale


# Example
original = torch.tensor([0.5, -0.3, 1.2, -0.8, 0.0, 0.15])
print(f"Original FP32:   {original}")

q, scale, zp = quantize_tensor(original, num_bits=8, symmetric=True)
print(f"Quantized INT8:  {q}")
print(f"Scale: {scale:.6f}, Zero-point: {zp}")

reconstructed = dequantize_tensor(q, scale, zp)
print(f"Dequantized:     {reconstructed}")
print(f"Error:           {(original - reconstructed).abs()}")
print(f"Max error:       {(original - reconstructed).abs().max():.6f}")
```

### 2.2 대칭 vs 비대칭 Quantization

```
대칭 Quantization:
  FP32 범위: [-α, +α]  →  INT8 범위: [-127, +127]
  Zero-point = 0
  더 간단한 연산 (zero-point 오프셋 없음)
  적합: 가중치 (일반적으로 0 중심)

  ─────────────┼─────────────
  -α           0            +α
  -127         0           +127

비대칭 Quantization:
  FP32 범위: [rmin, rmax]  →  INT8 범위: [0, 255]
  Zero-point ≠ 0
  전체 INT8 범위를 더 효율적으로 사용
  적합: 활성값 (ReLU 이후 종종 음이 아닌 값)

  ──────────────────────────
  rmin                  rmax
  0                      255
  └─── zero_point가 0.0에 매핑
```

```python
import torch


def compare_symmetric_asymmetric(tensor):
    """Compare symmetric and asymmetric quantization on the same tensor."""
    print(f"Input: {tensor}")
    print(f"Range: [{tensor.min():.3f}, {tensor.max():.3f}]")

    # Symmetric
    q_sym, s_sym, zp_sym = quantize_tensor(tensor, num_bits=8, symmetric=True)
    r_sym = dequantize_tensor(q_sym, s_sym, zp_sym)

    # Asymmetric
    q_asym, s_asym, zp_asym = quantize_tensor(tensor, num_bits=8, symmetric=False)
    r_asym = dequantize_tensor(q_asym, s_asym, zp_asym)

    print(f"\nSymmetric  - Scale: {s_sym:.6f}, ZP: {zp_sym}")
    print(f"  Error (MSE): {((tensor - r_sym) ** 2).mean():.8f}")

    print(f"\nAsymmetric - Scale: {s_asym:.6f}, ZP: {zp_asym}")
    print(f"  Error (MSE): {((tensor - r_asym) ** 2).mean():.8f}")


# Case 1: Symmetric data (weights) — symmetric wins or ties
weights = torch.randn(8) * 0.1
print("=== 가중치 유사 데이터 (0 중심) ===")
compare_symmetric_asymmetric(weights)

# Case 2: Asymmetric data (ReLU activations) — asymmetric wins
activations = torch.relu(torch.randn(8))
print("\n=== 활성값 유사 데이터 (음이 아닌 값) ===")
compare_symmetric_asymmetric(activations)
```

### 2.3 Per-Tensor vs Per-Channel Quantization

```
Per-Tensor: 전체 텐서에 대해 하나의 scale과 zero-point
  장점: 간단하고 빠름
  단점: 한 채널의 범위가 매우 다르면 정밀도가 낭비됨

Per-Channel: 각 출력 채널에 대해 별도의 scale과 zero-point
  장점: 훨씬 나은 정확도 (각 채널이 전체 범위를 사용)
  단점: 연산이 약간 더 복잡

Per-Channel은 가중치에 강력히 권장됩니다.
Per-Tensor는 일반적으로 활성값에 충분합니다.
```

```python
import torch
import torch.nn as nn


def per_channel_quantize(weight_tensor, num_bits=8, axis=0):
    """
    Quantize a weight tensor with per-channel scales.

    For Conv2d weight shape (C_out, C_in, H, W), axis=0 means
    one scale per output channel.
    """
    num_channels = weight_tensor.shape[axis]
    qmin = -(2 ** (num_bits - 1)) + 1
    qmax = 2 ** (num_bits - 1) - 1

    scales = torch.zeros(num_channels)
    quantized = torch.zeros_like(weight_tensor, dtype=torch.int8)

    for c in range(num_channels):
        # Select the slice for this channel
        if axis == 0:
            channel_data = weight_tensor[c]
        else:
            raise ValueError("Only axis=0 supported in this example")

        abs_max = channel_data.abs().max()
        scale = abs_max / qmax if abs_max > 0 else 1.0
        scales[c] = scale
        quantized[c] = torch.clamp(
            torch.round(channel_data / scale), qmin, qmax
        ).to(torch.int8)

    return quantized, scales


# Example: Conv2d weight with different channel ranges
conv = nn.Conv2d(3, 16, 3, bias=False)
with torch.no_grad():
    # Artificially make channel 0 have much larger weights
    conv.weight[0] *= 10.0

# Per-tensor quantization
q_tensor, s_tensor, _ = quantize_tensor(conv.weight.data, num_bits=8, symmetric=True)
r_tensor = dequantize_tensor(q_tensor, s_tensor, 0)
per_tensor_error = ((conv.weight.data - r_tensor) ** 2).mean()

# Per-channel quantization
q_channel, s_channel = per_channel_quantize(conv.weight.data, num_bits=8)
r_channel = torch.zeros_like(conv.weight.data)
for c in range(16):
    r_channel[c] = q_channel[c].float() * s_channel[c]
per_channel_error = ((conv.weight.data - r_channel) ** 2).mean()

print(f"Per-tensor  MSE: {per_tensor_error:.8f}")
print(f"Per-channel MSE: {per_channel_error:.8f}")
print(f"Per-channel is {per_tensor_error / per_channel_error:.1f}x more precise")
```

---

## 3. Post-Training Quantization (PTQ)

PTQ는 사전 학습된 모델을 **재학습 없이** 양자화합니다. 가장 빠르고 간단한 접근 방식입니다.

### 3.1 동적 Quantization

가중치는 미리 양자화되고, 활성값은 런타임에 동적으로 양자화됩니다.

```python
import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


# Create and "train" the model (using random weights for demonstration)
model = SimpleClassifier()
model.eval()

# Dynamic quantization: quantize Linear layers to INT8
quantized_model = torch.quantization.quantize_dynamic(
    model,
    qconfig_spec={nn.Linear},  # Which layer types to quantize
    dtype=torch.qint8,          # Target dtype
)

# Compare sizes
def model_size_mb(model):
    """Calculate model size in MB."""
    torch.save(model.state_dict(), "/tmp/temp_model.pt")
    import os
    size = os.path.getsize("/tmp/temp_model.pt") / 1e6
    os.remove("/tmp/temp_model.pt")
    return size


print(f"Original model size:  {model_size_mb(model):.2f} MB")
print(f"Quantized model size: {model_size_mb(quantized_model):.2f} MB")

# Verify output similarity
x = torch.randn(1, 784)
with torch.no_grad():
    orig_out = model(x)
    quant_out = quantized_model(x)
print(f"Output difference: {(orig_out - quant_out).abs().max():.6f}")
```

### 3.2 정적 Quantization

가중치와 활성값 모두 양자화됩니다. 활성값 범위를 결정하기 위한 **캘리브레이션 단계**가 필요합니다.

```python
import torch
import torch.nn as nn
from torch.quantization import (
    get_default_qconfig,
    prepare,
    convert,
    QuantStub,
    DeQuantStub,
)


class QuantizableClassifier(nn.Module):
    """Model with explicit quant/dequant stubs for static quantization."""

    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.quant = QuantStub()     # Quantize input
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dequant = DeQuantStub()  # Dequantize output

    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)
        return x


# Step 1: Create model and set evaluation mode
model = QuantizableClassifier()
model.eval()

# Step 2: Specify quantization configuration
# 'x86' backend is optimized for Intel CPUs
# 'qnnpack' backend is optimized for ARM/mobile
model.qconfig = get_default_qconfig("x86")

# Step 3: Prepare — inserts observer modules that record tensor statistics
model_prepared = prepare(model)

# Step 4: Calibrate — run representative data through the model
# The observers collect min/max statistics for each tensor
def calibrate(model, calibration_loader, num_batches=100):
    """Run calibration data through the model to collect activation ranges."""
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(calibration_loader):
            if i >= num_batches:
                break
            model(inputs)
    print(f"Calibration complete: {min(i+1, num_batches)} batches processed")


# Simulate calibration with random data
# In practice, use a representative subset of your training data
fake_loader = [(torch.randn(32, 784), torch.zeros(32)) for _ in range(100)]
calibrate(model_prepared, fake_loader)

# Step 5: Convert — replace float modules with quantized equivalents
model_quantized = convert(model_prepared)

print("\nQuantized model structure:")
print(model_quantized)

# The model now uses quantized operations internally
x = torch.randn(1, 784)
with torch.no_grad():
    output = model_quantized(x)
print(f"\nOutput shape: {output.shape}")
```

### 3.3 캘리브레이션 방법

캘리브레이션 단계는 활성값 quantization을 위한 **클리핑 범위**를 결정합니다. 각 방법은 이상값 처리와 정밀도 간에 트레이드오프가 있습니다:

| 방법 | 설명 | 적합한 경우 |
|--------|------------|---------|
| **MinMax** | 관측된 최소/최대 값을 사용 | 단순한 경우, 이상값 없음 |
| **Percentile** | 99.99번째 백분위수에서 클리핑 | 이상값이 있는 데이터 |
| **MSE (L2)** | 평균 제곱 오차를 최소화 | 범용 |
| **Entropy (KL)** | KL 발산을 최소화 | 긴 꼬리를 가진 활성값 분포 |

```python
import torch
import numpy as np


def calibrate_minmax(tensor):
    """MinMax calibration: use the full observed range."""
    return tensor.min().item(), tensor.max().item()


def calibrate_percentile(tensor, percentile=99.99):
    """Percentile calibration: clip outliers beyond the percentile."""
    lower = np.percentile(tensor.numpy(), 100 - percentile)
    upper = np.percentile(tensor.numpy(), percentile)
    return lower, upper


def calibrate_mse(tensor, num_bits=8):
    """
    MSE calibration: find the clipping range that minimizes
    quantization error (mean squared error).
    """
    best_mse = float("inf")
    best_range = (tensor.min().item(), tensor.max().item())

    abs_max = tensor.abs().max().item()
    # Search over candidate clipping thresholds
    for alpha in np.linspace(0.5, 1.0, 100):
        clip_val = alpha * abs_max
        clipped = tensor.clamp(-clip_val, clip_val)

        # Simulate quantization
        scale = clip_val / (2 ** (num_bits - 1) - 1)
        q = torch.round(clipped / scale)
        r = q * scale

        mse = ((tensor - r) ** 2).mean().item()
        if mse < best_mse:
            best_mse = mse
            best_range = (-clip_val, clip_val)

    return best_range


# Compare calibration methods on data with outliers
torch.manual_seed(42)
activations = torch.randn(10000)
# Add some outliers
activations[0] = 10.0
activations[1] = -8.0

print("Calibration ranges for data with outliers:")
print(f"  MinMax:     {calibrate_minmax(activations)}")
print(f"  Percentile: {calibrate_percentile(activations)}")
print(f"  MSE:        {calibrate_mse(activations)}")
```

---

## 4. Quantization-Aware Training (QAT)

QAT는 학습 중에 quantization을 시뮬레이션하여 모델이 quantization 오류를 최소화하도록 가중치를 적응시킵니다. 이는 PTQ보다 더 나은 정확도를 제공하며, 특히 낮은 비트폭(INT4)에서 효과적입니다.

### 4.1 Fake Quantization

QAT 동안 **fake quantization** 연산자가 순전파에 삽입됩니다. 이들은 양자화하고 즉시 역양자화하여, 역전파를 위한 FP32 연산을 유지하면서 quantization 노이즈를 도입합니다.

```
Fake quantization이 포함된 순전파:
  FP32 weight → Quantize → Dequantize → FP32 (quantization 노이즈 포함)
                  │              │
                  └──── INT8 반올림 오류를 시뮬레이션

역전파:
  Straight-Through Estimator (STE) 사용:
  기울기가 반올림 연산을 변경 없이 통과
```

```python
import torch
import torch.nn as nn


class FakeQuantize(torch.autograd.Function):
    """
    Simulate quantization in the forward pass.
    Use Straight-Through Estimator (STE) in the backward pass.
    """
    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        # Quantize then dequantize
        x_q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
        x_dq = (x_q - zero_point) * scale
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: pass gradient through unchanged
        return grad_output, None, None, None, None


def fake_quantize(x, num_bits=8):
    """Apply fake quantization to a tensor."""
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    scale = x.abs().max() / qmax
    zero_point = 0
    return FakeQuantize.apply(x, scale, zero_point, qmin, qmax)


# Demonstrate: gradient flows through fake quantization
x = torch.randn(4, requires_grad=True)
y = fake_quantize(x, num_bits=8)
loss = y.sum()
loss.backward()
print(f"Input:    {x.data}")
print(f"Fake Q:   {y.data}")
print(f"Gradient: {x.grad}")  # Should be all 1s (STE)
```

### 4.2 QAT 학습 루프

```python
import torch
import torch.nn as nn
from torch.quantization import (
    get_default_qat_qconfig,
    prepare_qat,
    convert,
    QuantStub,
    DeQuantStub,
)


class QATClassifier(nn.Module):
    """Model ready for Quantization-Aware Training."""

    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.quant = QuantStub()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = self.dequant(x)
        return x


def train_qat(model, train_loader, val_loader, epochs=10, lr=1e-3):
    """
    Quantization-Aware Training loop.

    Steps:
    1. Start from a pretrained FP32 model
    2. Insert fake quantization modules
    3. Fine-tune for a few epochs (the model learns to tolerate quantization noise)
    4. Convert to actual quantized model
    """
    # Step 1: Set QAT configuration
    model.train()
    model.qconfig = get_default_qat_qconfig("x86")

    # Step 2: Prepare for QAT — inserts fake quantize modules
    model_qat = prepare_qat(model)
    print("Model prepared for QAT (fake quantizers inserted)")

    # Step 3: Fine-tune
    optimizer = torch.optim.Adam(model_qat.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model_qat.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model_qat(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {total_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.1f}%")

    # Step 4: Convert to quantized model
    model_qat.eval()
    quantized_model = convert(model_qat)
    print("\nQAT complete — model converted to quantized format")

    return quantized_model


# Usage (with real data loaders):
# model = QATClassifier()
# model.load_state_dict(pretrained_weights)  # Start from pretrained
# q_model = train_qat(model, train_loader, val_loader, epochs=5)
```

### 4.3 PTQ vs QAT 비교

| 항목 | PTQ | QAT |
|--------|-----|-----|
| **학습 필요** | 아니오 | 예 (미세 조정) |
| **캘리브레이션 데이터** | 소량 부분 집합 (100-1000 샘플) | 전체 학습 세트 |
| **시간** | 분 | 시간 (몇 에포크) |
| **정확도 (INT8)** | 0.5-2% 하락 | 0.1-0.5% 하락 |
| **정확도 (INT4)** | 3-10% 하락 | 1-3% 하락 |
| **사용 시기** | 빠른 배포, INT8 | 최고 정확도, INT4 |
| **복잡도** | 낮음 | 보통 |

---

## 5. 혼합 정밀도 Quantization

모든 레이어가 quantization에 동일하게 민감한 것은 아닙니다. **혼합 정밀도**는 정확도-효율성 트레이드오프를 극대화하기 위해 레이어마다 다른 비트폭을 할당합니다.

### 5.1 레이어 민감도 분석

```python
import torch
import torch.nn as nn
import copy


def layer_sensitivity_analysis(model, test_loader, device="cpu"):
    """
    Measure how sensitive each layer is to quantization.

    For each layer:
    1. Quantize ONLY that layer to INT8
    2. Measure accuracy drop
    3. Layers with larger drops should keep higher precision
    """
    model.eval()

    # Baseline accuracy (all FP32)
    baseline_acc = evaluate_accuracy(model, test_loader, device)
    print(f"Baseline accuracy (FP32): {baseline_acc:.2f}%\n")

    sensitivities = {}

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue

        # Create a copy with only this layer quantized
        model_copy = copy.deepcopy(model)
        for n, m in model_copy.named_modules():
            if n == name:
                # Simulate quantization by adding noise proportional to INT8 error
                with torch.no_grad():
                    for param in m.parameters():
                        scale = param.abs().max() / 127
                        quantized = torch.round(param / scale) * scale
                        param.copy_(quantized)
                break

        # Measure accuracy with this layer quantized
        layer_acc = evaluate_accuracy(model_copy, test_loader, device)
        drop = baseline_acc - layer_acc
        sensitivities[name] = {
            "accuracy": layer_acc,
            "drop": drop,
            "num_params": sum(p.numel() for p in module.parameters()),
        }
        print(f"  {name:<30} Acc: {layer_acc:.2f}% (drop: {drop:+.2f}%)")

    # Sort by sensitivity (most sensitive first)
    sorted_layers = sorted(sensitivities.items(), key=lambda x: -x[1]["drop"])
    print("\n=== Layer Sensitivity Ranking (most sensitive first) ===")
    for name, info in sorted_layers:
        print(f"  {name:<30} Drop: {info['drop']:+.2f}%, Params: {info['num_params']:,}")

    return sensitivities


def evaluate_accuracy(model, test_loader, device="cpu"):
    """Evaluate model accuracy on test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total
```

### 5.2 혼합 정밀도 할당

```python
def assign_mixed_precision(sensitivities, accuracy_budget=1.0):
    """
    Assign bit-widths to layers based on sensitivity analysis.

    Strategy:
    - Highly sensitive layers (drop > 0.5%): Keep FP16
    - Moderately sensitive (0.1-0.5%): Use INT8
    - Low sensitivity (< 0.1%): Use INT4
    """
    assignments = {}
    for name, info in sensitivities.items():
        drop = info["drop"]
        if drop > accuracy_budget * 0.5:
            assignments[name] = {"bits": 16, "reason": "high sensitivity"}
        elif drop > accuracy_budget * 0.1:
            assignments[name] = {"bits": 8, "reason": "moderate sensitivity"}
        else:
            assignments[name] = {"bits": 4, "reason": "low sensitivity"}

    # Summary
    total_params = sum(info["num_params"] for info in sensitivities.values())
    weighted_bits = sum(
        sensitivities[name]["num_params"] * assignments[name]["bits"]
        for name in assignments
    )
    avg_bits = weighted_bits / total_params if total_params > 0 else 0

    print(f"\n=== Mixed-Precision Assignment ===")
    print(f"Average bit-width: {avg_bits:.1f} bits")
    for name, assignment in assignments.items():
        params = sensitivities[name]["num_params"]
        print(f"  {name:<30} → INT{assignment['bits']} ({assignment['reason']})")

    return assignments
```

### 5.3 FP16 vs INT8 vs INT4 비교

| 정밀도 | 비트 | 메모리 | 연산 속도 | 정확도 영향 | 하드웨어 지원 |
|-----------|------|--------|--------------|----------------|-----------------|
| FP32 | 32 | 1x | 1x | 기준 | 범용 |
| FP16 | 16 | 2x 감소 | 2x 빠름 (GPU) | ~0% | GPU, NPU |
| BF16 | 16 | 2x 감소 | 2x 빠름 (GPU) | ~0% | A100+, TPU |
| INT8 | 8 | 4x 감소 | 2-4x 빠름 | 0.1-1% | GPU, CPU, NPU |
| INT4 | 4 | 8x 감소 | 2-4x 빠름* | 1-5% | 제한적 (LLM 중심) |
| Binary | 1 | 32x 감소 | 32x 빠름* | 5-20% | 커스텀 하드웨어 |

*속도 향상은 패킹된 정수 연산에 대한 하드웨어 지원에 따라 다릅니다.

---

## 6. PyTorch를 사용한 실전 Quantization

### 6.1 Conv 기반 모델 양자화

```python
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub


class QuantizableCNN(nn.Module):
    """A simple CNN ready for quantization."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.quant = QuantStub()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """
        Fuse Conv-BN-ReLU into a single operation for better quantized performance.
        Must be called before quantization.
        """
        torch.quantization.fuse_modules(
            self.features,
            [["0", "1", "2"], ["4", "5", "6"]],  # Conv-BN-ReLU groups
            inplace=True,
        )


# Full quantization workflow
model = QuantizableCNN()
model.eval()

# Step 1: Fuse operations (important for accuracy and speed)
model.fuse_model()

# Step 2: Set quantization config
model.qconfig = torch.quantization.get_default_qconfig("x86")

# Step 3: Prepare (insert observers)
model_prepared = torch.quantization.prepare(model)

# Step 4: Calibrate with representative data
with torch.no_grad():
    for _ in range(100):
        dummy = torch.randn(16, 1, 28, 28)
        model_prepared(dummy)

# Step 5: Convert to quantized model
model_quantized = torch.quantization.convert(model_prepared)

# Verify
x = torch.randn(1, 1, 28, 28)
with torch.no_grad():
    output = model_quantized(x)
print(f"Quantized output shape: {output.shape}")
print(f"Quantized model:\n{model_quantized}")
```

### 6.2 연산자 퓨전

연산 퓨전은 양자화된 성능에 매우 중요합니다. 퓨전된 모듈은 단일 커널로 실행되어 레이어 간의 반복적인 quantize/dequantize 단계를 방지합니다.

```
퓨전 전:                           퓨전 후:
Conv → Quant → BN → Quant → ReLU   ConvBnReLU (단일 퓨전 연산)
  3회 quantize/dequantize 쌍         중간 quantization 0회
  3회 커널 실행                      1회 커널 실행
```

```python
# Common fusion patterns in PyTorch
# Must be done BEFORE quantization preparation

# Conv + BN + ReLU
torch.quantization.fuse_modules(model, [["conv", "bn", "relu"]], inplace=True)

# Conv + BN
torch.quantization.fuse_modules(model, [["conv", "bn"]], inplace=True)

# Linear + ReLU
torch.quantization.fuse_modules(model, [["linear", "relu"]], inplace=True)

# Conv + ReLU (no BN)
torch.quantization.fuse_modules(model, [["conv", "relu"]], inplace=True)
```

---

## 7. LLM을 위한 INT4 Quantization

INT4 quantization은 대규모 언어 모델을 엣지 디바이스에 배포하는 데 특히 중요합니다.

### 7.1 GPTQ 스타일 가중치 Quantization

```python
import torch
import torch.nn as nn


def gptq_quantize_layer(weight, num_bits=4, group_size=128):
    """
    Simplified GPTQ-style quantization for a linear layer.

    GPTQ quantizes weights column-by-column, using the Hessian
    (second-order information) to compensate for quantization error.

    This simplified version uses group-wise quantization:
    weights are divided into groups of `group_size`, each with
    its own scale and zero-point.
    """
    rows, cols = weight.shape
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1

    # Group-wise quantization
    num_groups = (cols + group_size - 1) // group_size
    scales = torch.zeros(rows, num_groups)
    zeros = torch.zeros(rows, num_groups, dtype=torch.int32)
    quantized = torch.zeros_like(weight, dtype=torch.int8)

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, cols)
        group = weight[:, start:end]

        # Per-group scale and zero-point
        group_min = group.min(dim=1, keepdim=True)[0]
        group_max = group.max(dim=1, keepdim=True)[0]
        scale = (group_max - group_min) / (qmax - qmin)
        scale = scale.clamp(min=1e-8)
        zero_point = torch.round(-group_min / scale).to(torch.int32)

        # Quantize
        q = torch.clamp(
            torch.round(group / scale) + zero_point,
            qmin, qmax
        ).to(torch.int8)

        quantized[:, start:end] = q
        scales[:, g] = scale.squeeze()
        zeros[:, g] = zero_point.squeeze()

    return quantized, scales, zeros


# Example: quantize a linear layer to INT4
linear = nn.Linear(4096, 4096, bias=False)
q_weight, scales, zeros = gptq_quantize_layer(linear.weight.data, num_bits=4)

# Memory savings
fp32_size = linear.weight.numel() * 4  # 4 bytes per float32
int4_size = linear.weight.numel() * 0.5 + scales.numel() * 4 + zeros.numel() * 4  # 0.5 bytes per int4 + metadata

print(f"FP32 size: {fp32_size / 1e6:.1f} MB")
print(f"INT4 size: {int4_size / 1e6:.1f} MB (including scales)")
print(f"Compression: {fp32_size / int4_size:.1f}x")
```

---

## 8. Quantization 디버깅 및 검증

### 8.1 Quantization에 민감한 레이어 감지

```python
import torch
import torch.nn as nn


def compare_layer_outputs(fp32_model, quantized_model, sample_input):
    """
    Compare intermediate outputs between FP32 and quantized model
    to identify layers with high quantization error.
    """
    fp32_activations = {}
    quant_activations = {}

    # Hook to capture intermediate outputs
    def make_hook(storage, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                storage[name] = output.detach()
        return hook

    # Register hooks on both models
    for name, module in fp32_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
            module.register_forward_hook(make_hook(fp32_activations, name))

    for name, module in quantized_model.named_modules():
        # Quantized modules have different names; match by position
        if hasattr(module, "weight") or isinstance(module, nn.ReLU):
            module.register_forward_hook(make_hook(quant_activations, name))

    # Run forward pass
    with torch.no_grad():
        fp32_model(sample_input)
        quantized_model(sample_input)

    # Compare
    print(f"{'Layer':<30} {'FP32 Mean':>10} {'Quant Mean':>10} {'MSE':>12} {'SQNR(dB)':>10}")
    print("-" * 75)

    for name in fp32_activations:
        if name in quant_activations:
            fp32_act = fp32_activations[name].float()
            quant_act = quant_activations[name].float()

            if fp32_act.shape == quant_act.shape:
                mse = ((fp32_act - quant_act) ** 2).mean().item()
                signal_power = (fp32_act ** 2).mean().item()
                sqnr = 10 * torch.log10(
                    torch.tensor(signal_power / max(mse, 1e-10))
                ).item()

                print(f"{name:<30} {fp32_act.mean():>10.4f} {quant_act.mean():>10.4f} "
                      f"{mse:>12.6f} {sqnr:>10.1f}")
```

### 8.2 Signal-to-Quantization-Noise Ratio (SQNR)

```python
def compute_sqnr(original, quantized):
    """
    Compute SQNR in dB. Higher SQNR = less quantization noise.

    Rule of thumb: each additional bit adds ~6 dB of SQNR.
    - INT8:  ~48 dB (8 * 6)
    - INT4:  ~24 dB (4 * 6)
    - INT2:  ~12 dB (2 * 6)
    """
    noise = original - quantized
    signal_power = (original ** 2).mean()
    noise_power = (noise ** 2).mean()

    if noise_power == 0:
        return float("inf")

    sqnr_db = 10 * torch.log10(signal_power / noise_power)
    return sqnr_db.item()


# Example
tensor = torch.randn(1000)
for bits in [8, 4, 2]:
    q, scale, zp = quantize_tensor(tensor, num_bits=bits, symmetric=True)
    r = dequantize_tensor(q, scale, zp)
    sqnr = compute_sqnr(tensor, r)
    print(f"INT{bits}: SQNR = {sqnr:.1f} dB, "
          f"Theoretical max ≈ {bits * 6.02:.1f} dB")
```

---

## 정리

| 개념 | 핵심 요점 |
|---------|-------------|
| **Quantization** | FP32를 INT8/INT4로 매핑하여 크기를 4-8배 줄임 |
| **대칭** | Zero-point = 0, 가중치에 최적 |
| **비대칭** | 0이 아닌 zero-point, 활성값(ReLU 이후)에 최적 |
| **Per-channel** | 출력 채널당 별도 scale, 훨씬 나은 정확도 |
| **PTQ** | 재학습 불필요, 빠름, 0.5-2% 정확도 하락 |
| **QAT** | Fake quantization으로 미세 조정, 최고 정확도 |
| **캘리브레이션** | MinMax, Percentile, MSE, Entropy — 활성값 범위를 결정 |
| **혼합 정밀도** | 민감도에 따라 레이어별 다른 비트폭 할당 |
| **연산자 퓨전** | Conv-BN-ReLU를 퓨전하여 quantize/dequantize 단계 감소 |

---

## 연습 문제

### 연습 문제 1: Quantization 처음부터 구현

대칭 및 비대칭 모드 모두에 대해 `quantize_tensor`와 `dequantize_tensor`를 구현하십시오:
1. (100,) 형태의 랜덤 텐서를 두 모드를 사용하여 INT8로 양자화하십시오
2. 원본과 복원 텐서 간의 MSE를 측정하십시오
3. INT4와 INT2에 대해 반복하십시오
4. MSE vs 비트폭을 플롯하십시오

### 연습 문제 2: MNIST에서 PTQ

1. MNIST에서 소형 CNN을 >98% 정확도로 학습하십시오
2. 동적 quantization (`quantize_dynamic`)을 적용하고 정확도를 측정하십시오
3. 캘리브레이션(100 배치)으로 정적 quantization을 적용하고 정확도를 측정하십시오
4. FP32, 동적 INT8, 정적 INT8에 대해 모델 크기와 지연 시간을 비교하십시오

### 연습 문제 3: QAT 실험

1. 연습 문제 2의 PTQ 모델부터 시작하십시오
2. 5 에포크 동안 QAT를 적용하고 정확도를 측정하십시오
3. QAT 정확도 vs PTQ 정확도를 비교하십시오
4. 실험: INT4에서 QAT는 어떻게 되는가? 몇 에포크가 필요한가?

### 연습 문제 4: 캘리브레이션 방법 비교

1. CIFAR-10에서 ResNet-18을 학습하십시오
2. 네 가지 캘리브레이션 방법으로 정적 PTQ를 적용하십시오: MinMax, Percentile (99.9%), MSE, Entropy
3. 각 캘리브레이션 방법의 정확도를 비교하십시오
4. 어떤 방법이 가장 잘 작동하는가? INT4에서 답이 달라지는가?

### 연습 문제 5: 혼합 정밀도 전략

1. ResNet-18에 대해 레이어 민감도 분석을 실행하십시오
2. 민감도에 따라 비민감 레이어에 INT4, 민감 레이어에 INT8을 할당하십시오
3. 이 혼합 정밀도 모델을 균일 INT8과 비교하십시오
4. 전체 평균 비트폭은 얼마이며 정확도는 어떻게 비교되는가?

---

[이전: 모델 압축 개요](./02_Model_Compression_Overview.md) | [개요](./00_Overview.md) | [다음: Pruning](./04_Pruning.md)

**License**: CC BY-NC 4.0
