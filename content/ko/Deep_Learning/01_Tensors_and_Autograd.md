# 01. 텐서와 오토그래드

[다음: 신경망 기초](./02_Neural_Network_Basics.md)

---

> **PyTorch 2.x 안내**: 이 레슨은 PyTorch 2.0+ (2023년~)를 기준으로 합니다.
>
> PyTorch 2.0의 주요 기능:
> - `torch.compile()`: 그래프 컴파일로 학습/추론 속도 대폭 향상
> - `torch.func`: 함수 변환 (vmap, grad, jacrev 등)
> - 향상된 CUDA 그래프 지원
>
> 설치: `pip install torch>=2.0`

## 학습 목표

- 텐서(Tensor)의 개념과 NumPy 배열과의 차이점 이해
- PyTorch의 자동 미분(Autograd) 시스템 이해
- GPU 연산의 기초
- (PyTorch 2.x) torch.compile 기초

---

## 1. 텐서란?

텐서는 다차원 배열을 일반화한 개념입니다.

| 차원 | 이름 | 예시 |
|------|------|------|
| 0D | 스칼라 | 단일 숫자 (5) |
| 1D | 벡터 | [1, 2, 3] |
| 2D | 행렬 | [[1,2], [3,4]] |
| 3D | 3D 텐서 | 이미지 (H, W, C) |
| 4D | 4D 텐서 | 배치 이미지 (N, C, H, W) |

---

## 2. NumPy vs PyTorch 텐서 비교

NumPy가 이미 n차원 배열을 제공하는데 왜 새로운 자료구조가 필요할까요? NumPy 배열은 CPU에서만 동작하며 기울기 추적(gradient tracking) 개념이 없습니다. PyTorch 텐서는 추가적인 메타데이터를 가지고 있습니다 — `device`(CPU 또는 GPU), `requires_grad`(연산 기록 여부), 그리고 계산 그래프(computational graph)에 대한 참조 — 이것들이 함께 자동 미분(automatic differentiation)을 가능하게 하며, 이는 모든 신경망 학습의 근간입니다. 한마디로, PyTorch 텐서는 NumPy 배열에 *모델 학습에 필요한 기록 장치*를 더한 것입니다.

### 생성

```python
import numpy as np
import torch

# NumPy
np_arr = np.array([1, 2, 3])
np_zeros = np.zeros((3, 4))
np_rand = np.random.randn(3, 4)

# PyTorch
pt_tensor = torch.tensor([1, 2, 3])
pt_zeros = torch.zeros(3, 4)
pt_rand = torch.randn(3, 4)
```

### 변환

```python
# NumPy → PyTorch
tensor = torch.from_numpy(np_arr)

# PyTorch → NumPy
array = tensor.numpy()  # Only works for CPU tensors
```

### 주요 차이점

| 기능 | NumPy | PyTorch |
|------|-------|---------|
| GPU 지원 | ❌ | ✅ (`tensor.to('cuda')`) |
| 자동 미분 | ❌ | ✅ (`requires_grad=True`) |
| 기본 타입 | float64 | float32 |
| 메모리 공유 | - | `from_numpy`는 공유 |

---

## 3. 자동 미분 (Autograd)

PyTorch의 핵심 기능으로, 역전파를 자동으로 계산합니다.

신경망을 학습하려면 손실(loss)을 모든 매개변수에 대해 미분해야 합니다 — 수백만 개의 편미분이 필요할 수 있습니다. 이를 손으로 계산하는 것은 비현실적입니다. Autograd는 순전파(forward pass) 중 모든 연산을 계산 그래프에 기록하고, 그래프를 역순으로 탐색하여 연쇄법칙(chain rule)을 통해 모든 기울기를 자동으로 계산함으로써 이 문제를 해결합니다. 이것이 "모델 정의"에서 "모델 학습"으로의 도약을 거의 힘들이지 않고 가능하게 하는 핵심입니다.

### 기본 사용법

```python
# Why: requires_grad=True tells PyTorch to record every operation on this tensor
# into the computational graph, so that gradients can be computed later via .backward().
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1  # y = x² + 3x + 1

# Why: .backward() traverses the computational graph in reverse (topological order)
# to compute all partial derivatives via the chain rule.
y.backward()

# Check gradient
print(x.grad)  # tensor([7.])  # dy/dx = 2x + 3 = 2*2 + 3 = 7
```

### 계산 그래프

```
    x ─────┐
           │
    x² ────┼──▶ + ──▶ y
           │
    3x ────┘
```

- **순전파(Forward pass)**: 입력에서 출력으로 연산합니다. 각 연산(`**`, `*`, `+`)은 방향성 비순환 그래프(DAG)의 노드로 기록됩니다. PyTorch는 이 그래프를 동적으로 구축합니다 — 연산을 실행할 때마다 새로운 그래프가 생성됩니다.
- **역전파(Backward pass)**: 출력에서 시작하여 PyTorch가 그래프를 역순(위상 정렬 역순)으로 순회하며 각 노드에서 연쇄법칙(Chain Rule)을 적용하여 ∂y/∂x를 계산합니다. `.backward()` 완료 후 그래프는 기본적으로 **소멸**됩니다(`retain_graph=False`), 메모리를 해제합니다.

**연쇄법칙(Chain Rule) 실습 — 구체적 예시.** 합성 함수 `y = f(g(x))`에서 `g(x) = x²`, `f(u) = 3u + 1`인 경우, `x = 2`일 때:

```
Forward:  g = x² = 4,   y = 3g + 1 = 13
Backward: dy/dg = 3,    dg/dx = 2x = 4
          dy/dx = (dy/dg) × (dg/dx) = 3 × 4 = 12
```

각 노드는 자신의 *국소 미분*(local derivative, 입력 대비 출력의 변화율)만 알면 되고, 연쇄법칙이 이를 곱해줍니다. 이것이 바로 autograd가 계산 그래프의 모든 노드에서 수행하는 작업입니다 — 네트워크가 아무리 깊어도 동일한 원리입니다.

### 기울기 누적과 초기화

```python
# PyTorch accumulates gradients by default — calling backward() adds to
# existing .grad values rather than replacing them.  This is intentional:
# it allows gradient accumulation across multiple mini-batches (useful when
# the desired batch size exceeds GPU memory).  However, in a standard
# training loop you must zero gradients before each step, otherwise the
# optimizer uses the *sum* of all past gradients.
x.grad.zero_()  # Reset to 0; without this, gradients from previous steps pile up
```

---

## 4. 연산과 브로드캐스팅

```python
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# Basic operations
c = a + b           # Element-wise addition
c = a * b           # Element-wise multiplication (Hadamard product)
c = a @ b           # Matrix multiplication
c = torch.matmul(a, b)  # Matrix multiplication

# Broadcasting
a = torch.tensor([[1], [2], [3]])  # (3, 1)
b = torch.tensor([10, 20, 30])     # (3,)
c = a + b  # (3, 3) automatic expansion
```

---

## 5. GPU 연산

```python
# Check GPU availability
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Move tensor to GPU
x = torch.randn(1000, 1000)
x_gpu = x.to(device)
# Or
x_gpu = x.cuda()

# Operations (performed on the same device)
y_gpu = x_gpu @ x_gpu

# Bring result back to CPU
y_cpu = y_gpu.cpu()
```

---

## 6. 실습: NumPy vs PyTorch 자동 미분 비교

### 문제: f(x) = x³ + 2x² - 5x + 3의 x=2에서 미분값 구하기

수학적 해:
- f'(x) = 3x² + 4x - 5
- f'(2) = 3(4) + 4(2) - 5 = 12 + 8 - 5 = 15

### NumPy (수동 미분)

```python
import numpy as np

def f(x):
    return x**3 + 2*x**2 - 5*x + 3

def df(x):
    """Manually compute derivative"""
    return 3*x**2 + 4*x - 5

x = 2.0
print(f"f({x}) = {f(x)}")
print(f"f'({x}) = {df(x)}")  # 15.0
```

### PyTorch (자동 미분)

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x**3 + 2*x**2 - 5*x + 3

y.backward()
print(f"f({x.item()}) = {y.item()}")
print(f"f'({x.item()}) = {x.grad.item()}")  # 15.0
```

---

## 7. 주의사항

### in-place 연산

```python
# In-place operations can conflict with autograd
x = torch.tensor([1.0], requires_grad=True)
# x += 1  # May cause error
x = x + 1  # Create new tensor (safe)
```

### 기울기 추적 비활성화

```python
# Why: During inference we don't need gradients, so wrapping in torch.no_grad()
# skips building the computational graph — saving memory and improving speed
# (typically 20-30% faster for forward-only passes).
with torch.no_grad():
    y = model(x)  # No gradient computation

# Or
x.requires_grad = False
```

### detach()

```python
# Detach from computational graph — creates a new tensor that shares the
# same data but is not part of the autograd graph.  Common uses:
#   1. Prevent gradients flowing into a frozen sub-network (e.g., target
#      network in DQN, discriminator update in GANs)
#   2. Convert a tracked tensor to a plain value for logging/plotting
y = x.detach()  # y has the same values as x but no gradient history
```

---

## 8. PyTorch 2.x 새 기능

### torch.compile()

PyTorch 2.0의 핵심 기능으로, 모델을 컴파일하여 성능을 향상시킵니다.

```python
import torch

# Define model
model = MyModel()

# Compile the model (PyTorch 2.0+)
compiled_model = torch.compile(model)

# Usage is the same
output = compiled_model(input_data)
```

### 컴파일 모드

```python
# Default mode (balanced)
model = torch.compile(model)

# Maximum performance mode
model = torch.compile(model, mode="max-autotune")

# Memory-saving mode
model = torch.compile(model, mode="reduce-overhead")
```

### torch.func (함수 변환)

```python
from torch.func import vmap, grad, jacrev

# vmap: Automatic batch operations
def single_fn(x):
    return x ** 2

batched_fn = vmap(single_fn)
result = batched_fn(torch.randn(10, 3))  # Batch processing

# grad: Functional gradients
def f(x):
    return (x ** 2).sum()

grad_f = grad(f)
x = torch.randn(3)
print(grad_f(x))  # 2 * x
```

### 주의사항

```python
# torch.compile has compilation overhead on first run
# Warm-up recommended for production

# Dynamic shapes may cause recompilation
# Mitigate with dynamic=True option
model = torch.compile(model, dynamic=True)
```

---

## 정리

### NumPy에서 이해해야 할 것
- 텐서는 다차원 배열
- 행렬 연산 (곱셈, 전치, 브로드캐스팅)

### PyTorch에서 추가되는 것
- `requires_grad`: 자동 미분 활성화
- `backward()`: 역전파 수행
- `grad`: 계산된 기울기
- GPU 가속

### PyTorch 2.x 추가 기능
- `torch.compile()`: 성능 최적화
- `torch.func`: 함수형 변환 (vmap, grad)

---

## 다음 단계

[신경망 기초](./02_Neural_Network_Basics.md)에서 이 텐서와 자동 미분을 사용해 신경망을 구축합니다.
