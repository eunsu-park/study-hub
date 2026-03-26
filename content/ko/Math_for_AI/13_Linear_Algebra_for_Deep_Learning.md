# 13. 딥러닝을 위한 선형대수

## 학습 목표

- 텐서의 개념과 차원, 축(axis) 연산을 이해하고 활용한다
- 아인슈타인 표기법과 einsum을 사용한 효율적인 텐서 연산을 습득한다
- 자동 미분의 원리(포워드/리버스 모드)와 계산 그래프를 이해한다
- 딥러닝에서의 수치 안정성 문제와 해결책을 파악한다
- 가중치 초기화의 수학적 이론과 실제 적용을 학습한다
- 배치/레이어 정규화, 잔차 연결 등의 수학적 배경을 이해한다

---

## 1. 텐서 연산 (Tensor Operations)

### 1.1 텐서의 계층 구조

**스칼라 (0-텐서)**: 단일 숫자
```python
import numpy as np
import torch

s = 3.14
```

**벡터 (1-텐서)**: 1차원 배열
```python
v = np.array([1, 2, 3])  # shape: (3,)
```

**행렬 (2-텐서)**: 2차원 배열
```python
M = np.array([[1, 2], [3, 4]])  # shape: (2, 2)
```

**텐서 (n-텐서)**: n차원 배열
```python
T = np.random.randn(3, 4, 5)  # shape: (3, 4, 5)
```

### 1.2 텐서의 축(Axis)과 차원

딥러닝에서 전형적인 텐서 형태:
- **이미지**: (batch, channels, height, width) 또는 (batch, height, width, channels)
- **시퀀스**: (batch, sequence_length, features)
- **가중치**: (output_features, input_features)

```python
# 축 이해하기
batch_images = np.random.randn(32, 3, 224, 224)  # NCHW 형식
print(f"Shape: {batch_images.shape}")
print(f"배치 크기: {batch_images.shape[0]}")
print(f"채널 수: {batch_images.shape[1]}")
print(f"높이: {batch_images.shape[2]}")
print(f"너비: {batch_images.shape[3]}")

# 축에 따른 연산
batch_mean = batch_images.mean(axis=0)  # 배치 평균: (3, 224, 224)
spatial_mean = batch_images.mean(axis=(2, 3))  # 공간 평균: (32, 3)
global_mean = batch_images.mean()  # 전체 평균: 스칼라

print(f"배치 평균 shape: {batch_mean.shape}")
print(f"공간 평균 shape: {spatial_mean.shape}")
print(f"전체 평균: {global_mean}")
```

### 1.3 브로드캐스팅 (Broadcasting)

NumPy와 PyTorch는 크기가 다른 배열을 자동으로 확장하여 연산합니다.

**브로드캐스팅 규칙**:
1. 차원 수가 다르면, 작은 쪽의 앞에 차원 1을 추가
2. 각 차원에서 크기가 1이거나 같으면 호환
3. 크기 1인 차원은 다른 크기로 확장

```python
# 브로드캐스팅 예제
A = np.random.randn(32, 3, 224, 224)  # 배치 이미지
mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)  # 채널별 평균
std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)   # 채널별 표준편차

# 정규화: 브로드캐스팅 적용
A_normalized = (A - mean) / std
print(f"정규화된 shape: {A_normalized.shape}")  # 여전히 (32, 3, 224, 224)

# 브로드캐스팅 시각화
print("\n브로드캐스팅 과정:")
print(f"A: {A.shape}")
print(f"mean: {mean.shape} -> 브로드캐스트 -> {A.shape}")
print(f"결과: {A_normalized.shape}")
```

### 1.4 텐서 형태 변환

```python
import torch

# 다양한 형태 변환
x = torch.randn(32, 3, 224, 224)

# 1. reshape: 연속 메모리에서만 가능
x_flat = x.reshape(32, -1)  # (32, 3*224*224)
print(f"Flatten: {x_flat.shape}")

# 2. view vs reshape
x_view = x.view(32, 3, -1)  # (32, 3, 224*224)
x_reshaped = x.reshape(32, 3, -1)  # 동일

# 3. transpose: 축 교환
x_transposed = x.transpose(1, 2)  # (32, 224, 3, 224)
print(f"Transpose: {x_transposed.shape}")

# 4. permute: 임의 순서로 축 재배치
x_permuted = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
print(f"Permute: {x_permuted.shape}")

# 5. squeeze/unsqueeze: 차원 추가/제거
x_squeezed = torch.randn(32, 1, 224, 224).squeeze(1)  # (32, 224, 224)
x_unsqueezed = x_squeezed.unsqueeze(1)  # (32, 1, 224, 224)
print(f"Squeeze: {x_squeezed.shape}, Unsqueeze: {x_unsqueezed.shape}")
```

---

## 2. 아인슈타인 표기법 (Einstein Notation / einsum)

### 2.1 아인슈타인 합산 규칙

**핵심 아이디어**: 반복되는 인덱스는 자동으로 합산됩니다.

**전통적 표기**:
$$
C_{ik} = \sum_{j} A_{ij} B_{jk}
$$

**아인슈타인 표기**:
$$
C_{ik} = A_{ij} B_{jk}
$$

$j$가 양쪽에 나타나므로 암묵적으로 합산됩니다.

### 2.2 einsum 기본 사용법

```python
# 1. 행렬-벡터 곱: y = Ax
A = np.random.randn(3, 4)
x = np.random.randn(4)
y1 = A @ x                      # 전통적 방법
y2 = np.einsum('ij,j->i', A, x)  # einsum
print(f"행렬-벡터 곱 일치: {np.allclose(y1, y2)}")

# 2. 행렬 곱: C = AB
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C1 = A @ B
C2 = np.einsum('ij,jk->ik', A, B)
print(f"행렬 곱 일치: {np.allclose(C1, C2)}")

# 3. 대각합(trace): tr(A) = Σ A_ii
A = np.random.randn(4, 4)
trace1 = np.trace(A)
trace2 = np.einsum('ii->', A)
print(f"대각합 일치: {np.allclose(trace1, trace2)}")

# 4. 외적: C_ij = a_i b_j
a = np.array([1, 2, 3])
b = np.array([4, 5])
C1 = np.outer(a, b)
C2 = np.einsum('i,j->ij', a, b)
print(f"외적 일치: {np.allclose(C1, C2)}")
```

### 2.3 배치 연산

딥러닝에서 가장 유용한 경우입니다.

```python
# 배치 행렬 곱 (bmm)
# A: (batch, n, m), B: (batch, m, p) -> C: (batch, n, p)
batch_size, n, m, p = 32, 10, 20, 15
A = np.random.randn(batch_size, n, m)
B = np.random.randn(batch_size, m, p)

C1 = np.einsum('bij,bjk->bik', A, B)
# torch 비교
A_torch = torch.from_numpy(A)
B_torch = torch.from_numpy(B)
C2 = torch.bmm(A_torch, B_torch).numpy()
print(f"배치 행렬 곱 일치: {np.allclose(C1, C2)}")

print(f"입력 shapes: A={A.shape}, B={B.shape}")
print(f"출력 shape: C={C1.shape}")
```

### 2.4 어텐션 메커니즘

트랜스포머의 핵심 연산을 einsum으로 표현하면 매우 간결합니다.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

```python
def attention_einsum(Q, K, V):
    """
    Q: (batch, query_len, d_k)
    K: (batch, key_len, d_k)
    V: (batch, key_len, d_v)
    """
    d_k = Q.shape[-1]

    # QK^T: (batch, query_len, key_len)
    scores = np.einsum('bqd,bkd->bqk', Q, K) / np.sqrt(d_k)

    # Softmax
    attention_weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attention_weights /= attention_weights.sum(axis=-1, keepdims=True)

    # Attention weights × V: (batch, query_len, d_v)
    output = np.einsum('bqk,bkv->bqv', attention_weights, V)

    return output, attention_weights

# 테스트
batch, seq_len, d_model = 32, 10, 64
Q = np.random.randn(batch, seq_len, d_model)
K = np.random.randn(batch, seq_len, d_model)
V = np.random.randn(batch, seq_len, d_model)

output, weights = attention_einsum(Q, K, V)
print(f"Attention output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

### 2.5 복잡한 텐서 연산

```python
# 1. 이변량 곱(Bilinear form): x^T A y
A = np.random.randn(5, 5)
x = np.random.randn(5)
y = np.random.randn(5)

result1 = x @ A @ y
result2 = np.einsum('i,ij,j->', x, A, y)
print(f"이변량 곱 일치: {np.allclose(result1, result2)}")

# 2. 배치 이변량 곱
batch_size = 32
x_batch = np.random.randn(batch_size, 5)
y_batch = np.random.randn(batch_size, 5)
result = np.einsum('bi,ij,bj->b', x_batch, A, y_batch)
print(f"배치 이변량 곱 shape: {result.shape}")

# 3. 텐서 축약(contraction)
# A: (i,j,k), B: (k,l,m) -> C: (i,j,l,m)
A = np.random.randn(3, 4, 5)
B = np.random.randn(5, 6, 7)
C = np.einsum('ijk,klm->ijlm', A, B)
print(f"텐서 축약 결과 shape: {C.shape}")
```

---

## 3. 자동 미분 (Automatic Differentiation)

### 3.1 계산 그래프 (Computational Graph)

모든 계산을 방향성 비순환 그래프(DAG)로 표현합니다.

```python
import torch

# 계산 그래프 예제
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# 순방향 계산
z = x**2 + x*y + y**2
z.backward()  # 역전파

print(f"z = {z.item()}")
print(f"∂z/∂x = {x.grad.item()}")  # 2x + y = 2*2 + 3 = 7
print(f"∂z/∂y = {y.grad.item()}")  # x + 2y = 2 + 2*3 = 8

# 계산 그래프 시각화 (개념적)
"""
      x=2         y=3
       ↓           ↓
    x²=4    →  x*y=6  ←  y²=9
       ↓           ↓        ↓
       └─────→  z = 4+6+9 = 19
"""
```

### 3.2 포워드 모드 (Forward Mode)

**방향**: 입력 → 출력 (야코비안-벡터 곱, JVP)

**계산**: $\frac{\partial y}{\partial x} \cdot v$ (방향 $v$에 대한 방향 도함수)

**장점**: 입력이 적을 때 효율적 ($n_{\text{in}} \ll n_{\text{out}}$)

**예**: $f: \mathbb{R} \to \mathbb{R}^{1000}$

```python
# 포워드 모드 개념 (PyTorch는 기본적으로 리버스 모드 사용)
def forward_mode_example():
    """포워드 모드의 개념적 구현"""
    # f(x) = x^2 + 2x + 1
    # df/dx = 2x + 2

    x = 3.0
    dx = 1.0  # 방향 벡터 (보통 1)

    # 동시에 값과 도함수 계산
    y = x**2 + 2*x + 1      # y = 16
    dy = 2*x*dx + 2*dx      # dy/dx = 8

    return y, dy

y, dy = forward_mode_example()
print(f"f(3) = {y}, f'(3) = {dy}")
```

### 3.3 리버스 모드 (Reverse Mode / Backpropagation)

**방향**: 출력 → 입력 (벡터-야코비안 곱, VJP)

**계산**: $v^T \cdot \frac{\partial y}{\partial x}$ (출력에 대한 그래디언트)

**장점**: 출력이 적을 때 효율적 ($n_{\text{out}} \ll n_{\text{in}}$)

**딥러닝**: 손실 함수는 스칼라 ($n_{\text{out}} = 1$) → 리버스 모드가 최적

```python
# 리버스 모드 (역전파)
def reverse_mode_demo():
    """리버스 모드 자동 미분 시연"""
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    w = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    b = torch.tensor([1.0, 1.0], requires_grad=True)

    # 순방향: y = w^T x + b
    y = x @ w + b  # (2,)

    # 손실: L = ||y||^2
    loss = (y ** 2).sum()

    # 역전파
    loss.backward()

    print(f"x.grad shape: {x.grad.shape}")  # (3,)
    print(f"w.grad shape: {w.grad.shape}")  # (3, 2)
    print(f"b.grad shape: {b.grad.shape}")  # (2,)

    return loss.item()

loss = reverse_mode_demo()
print(f"Loss: {loss}")
```

### 3.4 왜 리버스 모드가 딥러닝에 적합한가?

**신경망**: $f: \mathbb{R}^n \to \mathbb{R}$ (파라미터 공간 → 손실)

- **입력 차원**: $n \sim 10^6$ ~ $10^9$ (파라미터 수)
- **출력 차원**: 1 (손실 함수)

**비용 비교**:
- 포워드 모드: $O(n)$ 패스 필요 (각 입력에 대해)
- 리버스 모드: $O(1)$ 패스 (한 번의 역전파)

**결론**: 리버스 모드는 $O(n)$배 효율적입니다!

```python
# 효율성 비교 시뮬레이션
import time

def forward_pass(params):
    """간단한 신경망 순방향 패스"""
    x = torch.randn(1000, 1000)
    for p in params:
        x = torch.relu(x @ p)
    return x.sum()

# 큰 신경망 시뮬레이션
n_params = 10
param_list = [torch.randn(1000, 1000, requires_grad=True) for _ in range(n_params)]

# 리버스 모드 (표준 역전파)
start = time.time()
loss = forward_pass(param_list)
loss.backward()
reverse_time = time.time() - start

print(f"리버스 모드 시간: {reverse_time:.4f}초")
print(f"총 파라미터 수: {sum(p.numel() for p in param_list):,}")
print(f"한 번의 역전파로 모든 그래디언트 계산 완료")
```

---

## 4. 수치 안정성 (Numerical Stability)

### 4.1 부동소수점 산술의 한계

IEEE 754 표준:
- **Float32**: 약 $10^{-38}$ ~ $10^{38}$, 정밀도 $\sim 10^{-7}$
- **언더플로우**: 너무 작은 수 → 0
- **오버플로우**: 너무 큰 수 → inf

```python
# 부동소수점 한계 시연
print(f"Float32 최소값: {np.finfo(np.float32).min}")
print(f"Float32 최대값: {np.finfo(np.float32).max}")
print(f"Float32 엡실론: {np.finfo(np.float32).eps}")

# 언더플로우
x = np.float32(1e-40)
print(f"\n1e-40 (float32): {x}")  # 0.0

# 오버플로우
x = np.float32(1e40)
print(f"1e40 (float32): {x}")  # inf

# 정밀도 손실
x = np.float32(1.0 + 1e-8)
print(f"1.0 + 1e-8 (float32): {x}")  # 1.0 (손실됨)
```

### 4.2 Log-Sum-Exp 트릭

문제: $\log \sum_{i=1}^{n} e^{x_i}$ 계산 시 오버플로우

**직접 계산**:
```python
x = np.array([1000, 1001, 1002], dtype=np.float32)
result_naive = np.log(np.sum(np.exp(x)))
print(f"직접 계산: {result_naive}")  # inf (오버플로우!)
```

**Log-Sum-Exp 트릭**:
$$
\log \sum_{i} e^{x_i} = a + \log \sum_{i} e^{x_i - a}
$$
여기서 $a = \max_i x_i$

```python
def log_sum_exp(x):
    """수치적으로 안정한 log-sum-exp"""
    a = np.max(x)
    return a + np.log(np.sum(np.exp(x - a)))

result_stable = log_sum_exp(x)
print(f"LSE 트릭: {result_stable}")  # 1002.407 (정확함)

# scipy 비교
from scipy.special import logsumexp
result_scipy = logsumexp(x)
print(f"Scipy: {result_scipy}")
print(f"일치: {np.isclose(result_stable, result_scipy)}")
```

### 4.3 소프트맥스의 수치 안정 구현

**표준 소프트맥스**:
$$
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

**문제**: 큰 $x_i$에서 오버플로우

**안정 버전**:
$$
\text{softmax}(x)_i = \frac{e^{x_i - \max_j x_j}}{\sum_j e^{x_j - \max_j x_j}}
$$

```python
def softmax_naive(x):
    """불안정한 소프트맥스"""
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def softmax_stable(x):
    """수치적으로 안정한 소프트맥스"""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# 테스트
x = np.array([1000, 1001, 1002], dtype=np.float32)

print("불안정 버전:")
result_naive = softmax_naive(x)
print(f"  결과: {result_naive}")
print(f"  합: {result_naive.sum()}")

print("\n안정 버전:")
result_stable = softmax_stable(x)
print(f"  결과: {result_stable}")
print(f"  합: {result_stable.sum()}")

# 배치 처리
batch_logits = np.random.randn(32, 1000).astype(np.float32)
probs = softmax_stable(batch_logits)
print(f"\n배치 소프트맥스: {probs.shape}, 합: {probs.sum(axis=1)[:5]}")
```

### 4.4 그래디언트 클리핑

폭발하는 그래디언트를 방지합니다.

```python
def clip_gradients(gradients, max_norm=1.0):
    """그래디언트 노름 클리핑"""
    total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in gradients:
            g *= clip_coef
    return gradients, total_norm

# PyTorch 예제
model = torch.nn.Linear(100, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 학습 루프 내에서
for batch in range(10):
    loss = (model(torch.randn(32, 100)) ** 2).sum()
    loss.backward()

    # 그래디언트 클리핑
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    optimizer.zero_grad()
```

---

## 5. 가중치 초기화 이론

### 5.1 문제: 신호 소실/폭발

적절한 초기화가 없으면:
- **신호 소실**: 활성화가 0으로 수렴 → 그래디언트 소멸
- **신호 폭발**: 활성화가 무한대로 발산 → 그래디언트 폭발

**목표**: 각 레이어에서 활성화와 그래디언트의 분산을 유지

### 5.2 Xavier/Glorot 초기화

**설정**: 선형 레이어 $y = Wx + b$, 활성화 함수 없음

**가정**:
- $x_i$는 평균 0, 분산 $\text{Var}(x)$
- $W_{ij}$는 독립, 평균 0

**분산 전파**:
$$
\text{Var}(y_i) = n_{\text{in}} \cdot \text{Var}(W) \cdot \text{Var}(x)
$$

**순방향**: $\text{Var}(y) = \text{Var}(x)$ 원할 때
$$
\text{Var}(W) = \frac{1}{n_{\text{in}}}
$$

**역방향**: 그래디언트 분산 유지 원할 때
$$
\text{Var}(W) = \frac{1}{n_{\text{out}}}
$$

**Xavier 초기화**: 절충안
$$
\text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}
$$

```python
def xavier_uniform(n_in, n_out):
    """Xavier 균등 초기화"""
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))

def xavier_normal(n_in, n_out):
    """Xavier 정규 초기화"""
    std = np.sqrt(2 / (n_in + n_out))
    return np.random.normal(0, std, (n_in, n_out))

# 분산 전파 검증
n_layers = 10
layer_sizes = [100] * (n_layers + 1)
x = np.random.randn(1000, layer_sizes[0])

activations = [x]
for i in range(n_layers):
    W = xavier_uniform(layer_sizes[i], layer_sizes[i+1])
    x = x @ W  # 선형 변환 (활성화 없음)
    activations.append(x)

# 각 레이어 분산 확인
variances = [np.var(a) for a in activations]
print("Xavier 초기화 - 레이어별 분산:")
for i, var in enumerate(variances):
    print(f"  레이어 {i}: {var:.4f}")
```

### 5.3 He 초기화 (ReLU용)

ReLU는 음수를 0으로 만들어 분산을 절반으로 줄입니다.

**He 초기화**:
$$
\text{Var}(W) = \frac{2}{n_{\text{in}}}
$$

```python
def he_normal(n_in, n_out):
    """He 정규 초기화 (ReLU용)"""
    std = np.sqrt(2 / n_in)
    return np.random.normal(0, std, (n_in, n_out))

# ReLU 네트워크 시뮬레이션
x = np.random.randn(1000, 100)
activations_he = [x]

for i in range(10):
    W = he_normal(100, 100)
    x = x @ W
    x = np.maximum(0, x)  # ReLU
    activations_he.append(x)

variances_he = [np.var(a) for a in activations_he]
print("\nHe 초기화 + ReLU - 레이어별 분산:")
for i, var in enumerate(variances_he):
    print(f"  레이어 {i}: {var:.4f}")

# Xavier vs He 비교
x_xavier = np.random.randn(1000, 100)
activations_xavier = [x_xavier]
for i in range(10):
    W = xavier_uniform(100, 100)
    x_xavier = x_xavier @ W
    x_xavier = np.maximum(0, x_xavier)
    activations_xavier.append(x_xavier)

print("\nXavier 초기화 + ReLU (부적절):")
for i, a in enumerate(activations_xavier):
    print(f"  레이어 {i}: 분산 {np.var(a):.4f}, 0 비율 {(a==0).mean():.2%}")
```

### 5.4 PyTorch 초기화

```python
import torch.nn as nn

# 기본 초기화
layer = nn.Linear(100, 50)
print(f"기본 초기화: 평균 {layer.weight.mean():.4f}, 표준편차 {layer.weight.std():.4f}")

# Xavier 초기화
nn.init.xavier_uniform_(layer.weight)
print(f"Xavier: 평균 {layer.weight.mean():.4f}, 표준편차 {layer.weight.std():.4f}")

# He 초기화
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
print(f"He: 평균 {layer.weight.mean():.4f}, 표준편차 {layer.weight.std():.4f}")

# 전체 모델 초기화
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)
model.apply(init_weights)
```

---

## 6. 정규화와 잔차 연결

### 6.1 배치 정규화 (Batch Normalization)

**수식**:
$$
\hat{x} = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}}
$$

$$
y = \gamma \hat{x} + \beta
$$

여기서 $\gamma, \beta$는 학습 가능한 파라미터입니다.

```python
class BatchNorm1d:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x, training=True):
        if training:
            # 배치 통계 계산
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)

            # 러닝 통계 업데이트 (지수 이동 평균)
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                                self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                               self.momentum * batch_var

            # 정규화
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            # 추론 시: 러닝 통계 사용
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        # 스케일/시프트
        return self.gamma * x_norm + self.beta

# 테스트
bn = BatchNorm1d(10)
x_train = np.random.randn(32, 10) * 10 + 5  # 평균 5, 표준편차 10
x_normed = bn.forward(x_train, training=True)

print(f"입력: 평균 {x_train.mean(axis=0).mean():.2f}, 표준편차 {x_train.std(axis=0).mean():.2f}")
print(f"정규화 후: 평균 {x_normed.mean(axis=0).mean():.4f}, 표준편차 {x_normed.std(axis=0).mean():.4f}")
```

### 6.2 레이어 정규화 (Layer Normalization)

배치 차원 대신 특성 차원을 따라 정규화합니다 (트랜스포머에서 사용).

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    """레이어 정규화"""
    # x: (batch, features)
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

x = np.random.randn(32, 128) * 5 + 10
gamma = np.ones(128)
beta = np.zeros(128)
x_ln = layer_norm(x, gamma, beta)

print(f"레이어 정규화: 샘플별 평균 {x_ln.mean(axis=1)[:5]}")
```

### 6.3 잔차 연결의 그래디언트 흐름

ResNet의 핵심: $y = F(x) + x$

**그래디언트**:
$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \left(1 + \frac{\partial F}{\partial x}\right)
$$

항등 경로 ($+1$)가 그래디언트를 직접 전파합니다.

```python
# 잔차 연결 효과 시뮬레이션
def plain_network(x, depth=50):
    """일반 네트워크"""
    for _ in range(depth):
        W = np.random.randn(100, 100) * 0.01
        x = np.tanh(x @ W)
    return x

def residual_network(x, depth=50):
    """잔차 네트워크"""
    for _ in range(depth):
        W = np.random.randn(100, 100) * 0.01
        F_x = np.tanh(x @ W)
        x = F_x + x  # 잔차 연결
    return x

x = np.random.randn(10, 100)

y_plain = plain_network(x, depth=50)
y_residual = residual_network(x, depth=50)

print(f"일반 네트워크: 평균 {y_plain.mean():.6f}, 표준편차 {y_plain.std():.6f}")
print(f"잔차 네트워크: 평균 {y_residual.mean():.6f}, 표준편차 {y_residual.std():.6f}")
print(f"\n일반 네트워크는 신호가 소실되었습니다!")
```

### 6.4 그래디언트 소실/폭발 분석

**체인 룰**:
$$
\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial y_L} \prod_{i=2}^{L} \frac{\partial y_i}{\partial y_{i-1}}
$$

**문제**: 야코비안의 곱
- $\|\frac{\partial y_i}{\partial y_{i-1}}\| < 1$ → 그래디언트 소실
- $\|\frac{\partial y_i}{\partial y_{i-1}}\| > 1$ → 그래디언트 폭발

**해결책**:
1. 적절한 초기화 (Xavier, He)
2. 배치 정규화
3. 잔차 연결
4. 그래디언트 클리핑

---

## 연습 문제

### 문제 1: 텐서 조작
다음을 PyTorch로 구현하세요:
(a) (32, 3, 64, 64) 이미지 배치를 (32, 64, 64, 3)으로 변환
(b) 각 이미지의 공간 평균을 계산하여 (32, 3) 텐서 생성
(c) 채널별 표준편차를 계산하여 정규화

### 문제 2: einsum 마스터
einsum을 사용하여 다음을 구현하세요:
(a) 3D 배치 행렬의 대각합: (batch, n, n) → (batch,)
(b) 다중 헤드 어텐션의 핵심 연산 (Q, K 곱)
(c) 4차원 텐서 축약: (a,b,c,d) × (c,d,e,f) → (a,b,e,f)

### 문제 3: 수치 안정성
(a) 안정한 log-softmax 함수를 구현하세요 (log-sum-exp 사용)
(b) 매우 큰 로짓 값 [1000, 2000, 3000]에서 테스트
(c) PyTorch의 F.log_softmax와 결과 비교

### 문제 4: 초기화 실험
(a) 10층 신경망을 Xavier, He, 랜덤(0.01) 초기화로 각각 생성
(b) ReLU 활성화와 함께 순방향 패스 실행
(c) 각 레이어의 활성화 분산을 플롯하고 비교

### 문제 5: 배치 정규화 구현
(a) 2D 배치 정규화를 처음부터 구현 (forward + backward)
(b) PyTorch의 nn.BatchNorm2d와 결과 비교
(c) 훈련/추론 모드의 차이를 검증

---

## 참고 자료

### 서적
- **Deep Learning** (Goodfellow et al., 2016) - Chapter 6 (Deep Networks), Chapter 8 (Optimization)
- **Dive into Deep Learning** (Zhang et al.) - Interactive book with code

### 논문
- Glorot & Bengio (2010), "Understanding the difficulty of training deep feedforward neural networks" - Xavier 초기화
- He et al. (2015), "Delving Deep into Rectifiers" - He 초기화
- Ioffe & Szegedy (2015), "Batch Normalization"
- He et al. (2016), "Deep Residual Learning for Image Recognition" - ResNet

### 온라인 자료
- [PyTorch einsum tutorial](https://pytorch.org/docs/stable/generated/torch.einsum.html)
- [Efficient Attention with einsum](https://rockt.github.io/2018/04/30/einsum)
- [Numerical Stability in Deep Learning](https://towardsdatascience.com)
- [Weight Initialization Guide](https://pytorch.org/docs/stable/nn.init.html)

### 도구
- **PyTorch**: 자동 미분 및 텐서 연산
- **NumPy**: 수치 계산
- **TensorBoard**: 활성화/그래디언트 시각화
