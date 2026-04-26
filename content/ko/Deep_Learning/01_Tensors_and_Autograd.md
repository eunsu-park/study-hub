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

## 이론과 원리

PyTorch API를 다루기 전에, 텐서가 무엇*인지*와 오토그래드가 무엇을 *하는지*를 분리해서 이해하는 것이 도움이 됩니다. 텐서는 정확한 메모리 레이아웃을 가진 타입이 지정된 다차원 버퍼이고, 오토그래드는 그 버퍼에 대한 연산을 기록하여 기록을 거꾸로 재생함으로써 미분을 계산하는 런타임입니다. 두 아이디어 모두 — 레이아웃과 기록 — 어떤 딥러닝 프레임워크와도 독립적으로 존재하며, 이를 이해하면 본 레슨의 나머지가 왜 이런 모습인지 분명해집니다.

이 섹션에서 다루는 내용:

- **A.** 텐서 메모리 레이아웃: 스토리지(storage), 스트라이드(stride), 연속성(contiguity) 불변식
- **B.** 역방향 자동 미분(reverse-mode AD)과 그것이 O(N)인 이유
- **C.** 유향 비순환 그래프(DAG)로서의 계산 그래프
- **D.** Forward vs Reverse mode, 각각이 유리한 경우

### A. 메모리 레이아웃: 스토리지, 스트라이드, 연속성

PyTorch 텐서는 **스토리지(storage)**라 불리는 1차원 연속 메모리 블록 위의 *뷰(view)*입니다. 같은 스토리지를 2D 행렬, 3D 이미지, 또는 전치된 행렬로 바이트 복사 없이 재해석할 수 있습니다. 세 가지 메타데이터가 n차원 인덱스 `(i_0, ..., i_{n-1})`을 스토리지의 오프셋으로 변환합니다:

```
offset(i_0, ..., i_{n-1}) = base_offset + sum_k stride[k] * i_k
```

행 우선(row-major, C-contiguous) 텐서의 형상이 `(d_0, d_1, ..., d_{n-1})`이라면, 스트라이드는 후행 차원의 누적 곱입니다:

```
stride[k] = prod_{j > k} d_j
```

스트라이드가 이 공식과 일치하고 `base_offset = 0`일 때 텐서는 **연속(contiguous)**입니다. `transpose`, `permute`, `narrow` 같은 연산은 비연속 *뷰*를 만들어 — 스트라이드는 불규칙해지지만 스토리지는 그대로입니다. 그래서 `view()`는 연속성을 요구(행 우선 공식을 가정)하고, `reshape()`는 필요시 복사로 폴백합니다. 뷰와 복사의 차이를 아는 것은 O(1) 연산과 O(N) 연산의 차이입니다.

### B. 역방향 자동 미분(Reverse-Mode AD)

미분 가능한 원시(primitive) 연산의 합성으로 구성된 스칼라 함수 `L = f(x_1, ..., x_n)`이 주어지면, 자동 미분은 `dL/dx_i`를 정확히(부동소수점 정밀도까지) 계산하며, 수치적 근사가 아닙니다. 역방향 AD는 다음 두 단계로 이를 수행합니다:

1. **순전파(forward pass)**: 각 원시 연산 `y = g(u, v, ...)`를 평가하고, 나중에 지역(local) 야코비안을 계산할 수 있을 만큼 `(u, v, ...)`의 정보를 기억합니다.
2. **역전파(backward pass)**: `dL/dL = 1`에서 시작해 그래프를 역순으로 순회하며 각 지역 야코비안을 곱해 `dL/du`, `dL/dv`, ...를 누적합니다.

핵심은 비용입니다. 순전파가 `N`개의 원시 연산을 실행해 하나의 스칼라 `L`을 만든다면, 역전파도 `O(N)` 연산을 실행합니다 — 입력 개수와 무관합니다. 따라서 10억 파라미터 손실의 그래디언트를 계산하는 비용은 순전파 한 번과 거의 같으며, 이것이 딥러닝을 실용적으로 만드는 이유입니다. (Forward-mode AD는 정반대 트레이드오프를 가집니다. D 참조.)

### C. DAG로서의 계산 그래프

`requires_grad=True`인 각 텐서가 하나의 노드입니다. 각 연산은 새 노드를 만들고 입력들에 대한 엣지를 기록하며, 지역 야코비안 곱셈 방법을 아는 `grad_fn`을 함께 저장합니다. 그 결과가 순전파 연산이 실행되며 동적으로 만들어지는 **유향 비순환 그래프(DAG)**입니다 (PyTorch의 "define-by-run" 모델). 스칼라 잎(leaf)에서 `.backward()`가 호출되면 그래프는 역위상 순서로 순회되며, 체인 룰이 엣지마다 적용됩니다:

```
dL/du = sum over outgoing edges  dL/dy * dy/du
```

DAG 구조(순환 없음)가 단일 선형 시간 역전파를 가능하게 합니다. `.backward()` 후에는 `retain_graph=True`가 전달되지 않는 한 그래프가 해제됩니다 — 같은 그래프에 두 번째 `.backward()`가 기본적으로 에러를 내는 이유입니다.

### D. Forward vs Reverse: 각각이 유리한 경우

함수 `f: R^n -> R^m`의 전체 야코비안은 `m * n`개 항목을 가집니다. 두 모드는 이를 다르게 누적합니다:

- **Forward mode**는 접선 벡터 `dx`를 그래프 따라 전파하며, 입력 차원당 `O(N)` 비용입니다. `n << m`일 때(입력 적고 출력 많음) 유리합니다.
- **Reverse mode**는 코탄젠트 벡터 `dL`를 그래프 거꾸로 전파하며, 출력 차원당 `O(N)` 비용입니다. `n >> m`일 때(입력 많고 출력 하나) 유리합니다.

딥러닝은 `n >> m`의 극단(수십억 파라미터, 스칼라 손실 하나)에 위치하므로 reverse mode가 자연스러운 선택이며, `loss.backward()`는 정확히 시드 `1`을 가진 한 번의 reverse-mode 스윕입니다. PyTorch 2.x의 `torch.func.jacrev`와 `torch.func.jacfwd`는 전체 야코비안이나 고차 미분이 필요할 때 두 모드를 명시적으로 노출합니다.

### 이론에서 아래 코드로

| 이론 개념 | 본 레슨의 코드 구성 |
|-----------|---------------------|
| 스토리지 + 스트라이드 | `tensor.stride()`, `.view()` vs `.reshape()`, `.contiguous()` |
| Reverse-mode AD | `requires_grad=True`, `.backward()`, `.grad` |
| 계산 그래프(DAG) | `grad_fn`, `retain_graph`, `torch.no_grad()` |
| 모드 선택 | `torch.func.grad`, `jacrev`, `jacfwd` |

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
