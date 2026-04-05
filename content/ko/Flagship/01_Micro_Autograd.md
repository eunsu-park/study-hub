# 01. 마이크로 오토그래드 (Micro Autograd)

**난이도: ⭐⭐⭐ (고급)**

## 학습 목표

- 연산 그래프(Computation Graph)의 구조와 역할 이해
- 역방향 자동 미분(Reverse-mode Automatic Differentiation)의 원리 구현
- 연쇄 법칙(Chain Rule)이 역전파(Backpropagation)에서 어떻게 적용되는지 파악
- `Value` 클래스를 활용한 스칼라 오토그래드 엔진 구축
- 간단한 신경망을 처음부터 학습시키기

**관련 토픽**: Deep_Learning, Math_for_AI, Calculus_and_Differential_Equations

---

## 1. 이론적 배경

### 1.1 연산 그래프 (Computation Graph)

**연산 그래프**는 수학 표현식을 방향성 비순환 그래프(DAG)로 표현한 것입니다. 각 노드는 연산(operation) 또는 값(value)을 나타내며, 간선은 데이터의 흐름을 표시합니다.

```
예시: L = (a * b + c)^2

        L
        │
       sq          ← 제곱 연산
        │
        +
       / \
      *   c        ← 덧셈 연산
     / \
    a   b          ← 곱셈 연산
```

연산 그래프의 핵심 가치:
1. **순전파(Forward Pass)**: 입력에서 출력 방향으로 값 계산
2. **역전파(Backward Pass)**: 출력에서 입력 방향으로 그래디언트 계산
3. **자동 미분**: 수동으로 미분 공식을 유도할 필요 없이 그래디언트를 자동 계산

### 1.2 역방향 자동 미분 (Reverse-mode Autodiff)

역방향 자동 미분은 출력에서 시작하여 입력 방향으로 그래디언트를 전파합니다. 이 방식은 출력이 하나이고 입력이 여러 개인 경우 (딥러닝의 손실 함수가 대표적) 매우 효율적입니다.

**수치 미분 vs 자동 미분**:

```python
# 수치 미분 (느리고 근사적)
def numerical_grad(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps)) / (2 * eps)

# 자동 미분 (정확하고 효율적)
# → 연산 그래프를 역추적하며 연쇄 법칙 적용
```

### 1.3 연쇄 법칙 (Chain Rule)

연쇄 법칙은 합성 함수의 미분을 구성 함수들의 미분의 곱으로 분해합니다:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

이것이 역전파의 핵심입니다. 각 노드는 **로컬 그래디언트**(local gradient)만 계산하면 되고, 이를 상위 노드에서 내려온 그래디언트와 곱하면 전체 그래디언트를 얻습니다.

```
     ∂L/∂L = 1.0   (시작점)
         │
         ▼
     ∂L/∂y = ∂L/∂L · ∂L/∂y    (연쇄 법칙 적용)
         │
         ▼
     ∂L/∂x = ∂L/∂y · ∂y/∂x    (연쇄 법칙 적용)
```

---

## 2. 구현 워크스루

### 2.1 Value 클래스 — 핵심 자료 구조

`Value` 클래스는 스칼라 값을 감싸고, 연산 그래프와 그래디언트 정보를 추적합니다.

```python
class Value:
    """Stores a scalar value and its gradient."""

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
```

핵심 설계 포인트:
- `data`: 실제 스칼라 값
- `grad`: 이 노드에 대한 손실의 그래디언트 (∂L/∂self)
- `_backward`: 역전파 시 호출되는 클로저(closure)
- `_prev`: 이 노드를 생성한 자식 노드들의 집합
- `_op`: 디버깅용 연산 이름

### 2.2 산술 연산과 로컬 그래디언트

각 연산은 순전파 결과를 계산하고, 역전파를 위한 `_backward` 클로저를 정의합니다.

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += out.grad    # ∂(a+b)/∂a = 1
        other.grad += out.grad   # ∂(a+b)/∂b = 1
    out._backward = _backward
    return out

def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += other.data * out.grad  # ∂(a*b)/∂a = b
        other.grad += self.data * out.grad  # ∂(a*b)/∂b = a
    out._backward = _backward
    return out
```

**중요**: `+=`를 사용하는 이유 — 하나의 값이 여러 연산에 사용될 경우, 그래디언트가 **누적**되어야 합니다 (다변수 연쇄 법칙).

### 2.3 활성화 함수

```python
def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
    out = Value(t, (self,), 'tanh')

    def _backward():
        self.grad += (1 - t**2) * out.grad  # dtanh/dx = 1 - tanh²(x)
    out._backward = _backward
    return out

def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

    def _backward():
        self.grad += (out.data > 0) * out.grad  # dReLU/dx = 1 if x > 0 else 0
    out._backward = _backward
    return out
```

### 2.4 역전파 실행

위상 정렬(Topological Sort)을 사용하여 올바른 순서로 역전파를 실행합니다.

```python
def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(self)
    self.grad = 1.0
    for v in reversed(topo):
        v._backward()
```

### 2.5 신경망 프리미티브

`Value` 클래스 위에 뉴런(Neuron), 레이어(Layer), MLP를 구축합니다.

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

---

## 3. 핵심 설계 결정

### 3.1 왜 스칼라인가?

실제 딥러닝 프레임워크는 텐서 연산을 사용하지만, 스칼라 구현은:
- 자동 미분의 **본질적 메커니즘**을 명확히 드러냄
- 각 값의 그래디언트 흐름을 개별적으로 추적 가능
- 코드가 수학 공식과 일대일로 대응

### 3.2 왜 클로저(Closure)인가?

`_backward` 함수를 클로저로 정의함으로써:
- 순전파 시점의 로컬 변수를 캡처
- 별도의 역전파 그래프 자료구조 불필요
- 간결하고 직관적인 구현

### 3.3 그래디언트 누적

```python
# 잘못된 구현: 그래디언트 덮어쓰기
self.grad = out.grad  # ❌

# 올바른 구현: 그래디언트 누적
self.grad += out.grad  # ✅
```

변수가 여러 연산에 사용되면 (예: `y = x + x`), 각 경로의 그래디언트가 모두 합산되어야 합니다.

---

## 4. 연습문제

### 연습문제 1: 거듭제곱 연산 추가

`Value` 클래스에 `__pow__` 메서드를 구현하세요. 그래디언트는 $\partial(x^n)/\partial x = n \cdot x^{n-1}$입니다.

### 연습문제 2: 시그모이드 활성화

`sigmoid` 메서드를 구현하세요: $\sigma(x) = \frac{1}{1 + e^{-x}}$, 그래디언트는 $\sigma(x)(1 - \sigma(x))$입니다.

### 연습문제 3: 수치 미분 검증

자동 미분으로 계산한 그래디언트와 수치 미분 결과를 비교하는 `grad_check` 함수를 작성하세요.

### 연습문제 4: 간단한 학습 루프

XOR 데이터셋에 대해 `MLP([2, 4, 1])`을 학습시키고, 손실이 감소하는 과정을 관찰하세요.

### 연습문제 5: 시각화

Graphviz를 사용하여 연산 그래프를 시각화하는 `draw_dot(root)` 함수를 구현하세요.

---

## 5. 참고 자료

- Karpathy, A. (2022). *micrograd* — 교육용 오토그래드 엔진. https://github.com/karpathy/micrograd
- Baydin, A. G., et al. (2018). "Automatic Differentiation in Machine Learning: a Survey." *JMLR* 18(153):1-43.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 6.5: Back-Propagation. MIT Press.
- Paszke, A., et al. (2017). "Automatic differentiation in PyTorch." *NeurIPS Workshop*.

---

**이전 레슨**: [00_Overview.md](00_Overview.md) — Flagship 개요
**다음 레슨**: [02_Tiny_GAN.md](02_Tiny_GAN.md) — 적대적 생성 네트워크 구현
