# 02. 신경망 기초

[이전: 텐서와 오토그래드](./01_Tensors_and_Autograd.md) | [다음: 역전파](./03_Backpropagation.md)

---

## 학습 목표

- 퍼셉트론과 다층 퍼셉트론(MLP) 이해
- 활성화 함수의 역할과 종류
- PyTorch의 `nn.Module`로 신경망 구축

---

## 1. 퍼셉트론 (Perceptron)

가장 기본적인 신경망 단위입니다. 퍼셉트론은 생물학적 뉴런에서 느슨하게 영감을 받았습니다: 수상돌기(dendrite)가 신호(입력)를 받고, 세포체(soma)가 가중합을 계산하고, 축삭 소구(axon hillock)가 임계값(활성화 함수)을 적용하고, 축삭(axon)이 출력을 전달합니다. 실제 뉴런은 훨씬 복잡하지만, 이 비유는 핵심 아이디어를 잘 포착합니다 — 정보를 모으고, 집계하고, 결과에 따라 발화(fire)하거나 하지 않는 것입니다.

```
Input(x₁) ──w₁──┐
                │
Input(x₂) ──w₂──┼──▶ Σ(wᵢxᵢ + b) ──▶ Activation ──▶ Output(y)
                │
Input(x₃) ──w₃──┘
```

### 수식

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = Σwᵢxᵢ + b
y = activation(z)
```

### NumPy 구현

```python
import numpy as np

def perceptron(x, w, b, activation):
    z = np.dot(x, w) + b
    return activation(z)

# Example: Simple linear output
x = np.array([1.0, 2.0, 3.0])
w = np.array([0.5, -0.3, 0.8])
b = 0.1

z = np.dot(x, w) + b  # 1*0.5 + 2*(-0.3) + 3*0.8 + 0.1 = 2.4
```

---

## 2. 활성화 함수 (Activation Functions)

비선형 활성화 함수가 없다면, N개의 선형 층을 쌓아도 결국 단일 행렬 곱셈 `W_N ... W_2 W_1 x = W x`로 축약됩니다. 네트워크가 아무리 깊어도 선형 매핑만 학습할 수 있습니다. 비선형 활성화 함수가 이 축약을 막아, 네트워크가 *임의의* 연속 함수를 근사할 수 있게 합니다 — 이를 **보편 근사 정리(Universal Approximation Theorem)**라고 합니다.

비선형성을 추가하여 복잡한 패턴을 학습합니다.

### 주요 활성화 함수

| 함수 | 수식 | 특징 |
|------|------|------|
| Sigmoid | σ(x) = 1/(1+e⁻ˣ) | 출력 0~1, 기울기 소실 문제 |
| Tanh | tanh(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | 출력 -1~1 |
| ReLU | max(0, x) | 가장 많이 사용, 간단하고 효과적 |
| Leaky ReLU | max(αx, x) | 음수 영역에서 작은 기울기 |
| GELU | x·Φ(x) | Transformer에서 사용 |

**왜 시그모이드는 기울기 소실을 유발할까요?** 시그모이드의 미분은 `σ'(x) = σ(x)(1 - σ(x))`이며, `x = 0`에서 최댓값 `σ'(0) = 0.25`에 도달합니다. 깊은 네트워크에서는 기울기가 층마다 곱해집니다. 10개 층만 지나도 기울기는 최대 `0.25^10 ≈ 0.0000009`으로 줄어들어 사실상 0이 됩니다. 초기 층은 거의 학습 신호를 받지 못해 학습이 정체됩니다.

**ReLU가 이를 해결하는 이유:** 양수 입력에 대해 ReLU의 미분은 정확히 1이므로, 깊이에 관계없이 기울기가 그대로 통과합니다. 이것이 ReLU(와 그 변형들)가 훨씬 더 깊은 네트워크의 학습을 가능하게 한 이유입니다.

### NumPy 구현

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)
```

### PyTorch

```python
import torch.nn.functional as F

y = F.sigmoid(x)
y = F.relu(x)
y = F.tanh(x)
```

---

## 3. 다층 퍼셉트론 (MLP)

여러 층을 쌓아 복잡한 함수를 근사합니다.

```
Input Layer ──▶ Hidden Layer 1 ──▶ Hidden Layer 2 ──▶ Output Layer
(n units)        (h1 units)          (h2 units)          (m units)
```

### 순전파 (Forward Pass)

```python
# 2-layer MLP forward pass
z1 = x @ W1 + b1       # First linear transformation
a1 = relu(z1)          # Activation
z2 = a1 @ W2 + b2      # Second linear transformation
y = softmax(z2)        # Output (for classification)
```

---

## 4. PyTorch nn.Module

PyTorch에서 신경망을 정의하는 표준 방법입니다.

### 기본 구조

```python
import torch
import torch.nn as nn

# Why inherit nn.Module?  It automatically tracks all parameters (for optimizer),
# handles device transfer (model.to('cuda')), enables save/load (state_dict),
# and provides training/eval mode switching.
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    # Why define forward()?  This method is called automatically when you do
    # model(x).  It defines the computation graph — the path data takes through
    # layers.  PyTorch records each operation here for autograd.
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

### nn.Sequential 사용

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

---

## 5. 가중치 초기화

적절한 초기화는 학습 성능에 큰 영향을 미칩니다.

| 방법 | 특징 | 사용 |
|------|------|------|
| Xavier/Glorot | Sigmoid, Tanh에 적합 | `nn.init.xavier_uniform_` |
| He/Kaiming | ReLU에 적합 | `nn.init.kaiming_uniform_` |
| 영 초기화 | 사용 금지 (대칭성 문제) | - |

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

model.apply(init_weights)
```

---

## 6. 실습: XOR 문제 해결

단층 퍼셉트론으로 해결 불가능한 XOR 문제를 MLP로 해결합니다.

### 데이터

```
Input      Output
(0, 0) → 0
(0, 1) → 1
(1, 0) → 1
(1, 1) → 0
```

### MLP 구조

```
Input(2) ──▶ Hidden(4) ──▶ Output(1)
```

### PyTorch 구현

```python
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(1000):
    pred = model(X)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 7. NumPy vs PyTorch 비교

### MLP 순전파

```python
# NumPy (manual)
def forward_numpy(x, W1, b1, W2, b2):
    z1 = x @ W1 + b1
    a1 = np.maximum(0, z1)  # ReLU
    z2 = a1 @ W2 + b2
    return z2

# PyTorch (automatic)
class MLP(nn.Module):
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### 핵심 차이

| 항목 | NumPy | PyTorch |
|------|-------|---------|
| 순전파 | 직접 구현 | `forward()` 메서드 |
| 역전파 | 직접 미분 계산 | `loss.backward()` 자동 |
| 파라미터 관리 | 배열로 직접 관리 | `model.parameters()` |

---

## 정리

### 핵심 개념

1. **퍼셉트론**: 선형 변환 + 활성화 함수
2. **활성화 함수**: 비선형성 추가 (ReLU 권장)
3. **MLP**: 여러 층을 쌓아 복잡한 함수 학습
4. **nn.Module**: PyTorch의 신경망 기본 클래스

### NumPy로 구현하면서 배우는 것

- 행렬 연산의 의미
- 활성화 함수의 수학적 정의
- 순전파의 데이터 흐름

---

## 다음 단계

[역전파 이해](./03_Backpropagation.md)에서 역전파 알고리즘을 NumPy로 직접 구현합니다.
