# 레슨 3: 연쇄 법칙과 계산 그래프

## 학습 목표

- 합성 함수에 대한 다변수 연쇄 법칙을 서술하고 적용한다
- 계산을 노드와 에지를 가진 유향 비순환 그래프(DAG)로 표현한다
- 순방향 모드와 역방향 모드 자동 미분을 구별한다
- 역방향 연쇄 법칙의 적용으로서 역전파를 유도한다
- NumPy로 간단한 계산 그래프와 역전파를 처음부터 구현한다
- 일반적인 DL 연산(선형, ReLU, 시그모이드, 손실)을 통한 그래디언트를 계산한다
- 역방향 모드가 출력 수에 대해 $O(1)$인 이유를 이해한다

---

## 1. 다변수 연쇄 법칙

### 1.1 단일 변수 복습

$y = f(g(x))$에 대해:

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx} = f'(g(x)) \cdot g'(x)$$

### 1.2 다변수 연쇄 법칙

$L = f(\mathbf{g}(\mathbf{x}))$일 때:

$$\frac{\partial L}{\partial x_j} = \sum_{i=1}^{n} \frac{\partial L}{\partial g_i} \frac{\partial g_i}{\partial x_j}$$

행렬 형태로, 야코비안 $\mathbf{J}_\mathbf{g} \in \mathbb{R}^{n \times m}$을 사용하면:

$$\nabla_\mathbf{x} L = \mathbf{J}_\mathbf{g}^\top \nabla_\mathbf{g} L$$

**이것이 역전파의 기본 방정식입니다**: 상류 그래디언트에 국소 야코비안의 전치를 곱합니다.

---

## 2. 계산 그래프

### 2.1 계산 그래프란?

**계산 그래프**는 유향 비순환 그래프(DAG)입니다:
- **노드**는 값(텐서)을 나타냄
- **에지**는 연산(함수)을 나타냄
- **리프 노드**는 입력과 매개변수
- **루트 노드**는 출력(보통 손실 스칼라)

각 노드는 다음을 저장합니다:
1. 순전파에서의 **값**
2. 역전파에서의 **그래디언트** $\frac{\partial L}{\partial \text{node}}$

### 2.2 구현: 최소 계산 그래프

```python
import numpy as np

class Value:
    """자동 미분을 지원하는 계산 그래프의 노드."""

    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._children = set(children)
        self._op = op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """위상 정렬 후 역순으로 역전파."""
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
```

---

## 3. 순방향 모드 vs 역방향 모드

### 3.1 핵심 비교

| 속성 | 순방향 모드 | 역방향 모드 |
|------|-----------|-----------|
| 필요한 패스 | 입력당 하나 | 출력당 하나 |
| 적합한 경우 | 적은 입력, 많은 출력 | 많은 입력, 적은 출력 |
| DL 시나리오 | $n \sim 10^9$ 매개변수, 1 손실 | **역방향 모드 승** |
| 메모리 | 낮음 (스트림) | 높음 (순전파 값 저장) |

딥러닝은 하나의 스칼라 출력(손실)과 수백만 개의 입력(매개변수)을 가지므로, 역방향 모드가 확실한 승자입니다.

---

## 4. 일반적인 층의 역전파

### 4.1 선형 층

**순방향**: $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$

**역방향**: $\frac{\partial L}{\partial \mathbf{z}}$ (상류 그래디언트)가 주어지면:
- $\frac{\partial L}{\partial \mathbf{W}} = \frac{\partial L}{\partial \mathbf{z}} \mathbf{x}^\top$
- $\frac{\partial L}{\partial \mathbf{x}} = \mathbf{W}^\top \frac{\partial L}{\partial \mathbf{z}}$
- $\frac{\partial L}{\partial \mathbf{b}} = \frac{\partial L}{\partial \mathbf{z}}$

### 4.2 원소별 활성화

원소별 함수 $\mathbf{a} = \phi(\mathbf{z})$에 대해:

$$\frac{\partial L}{\partial \mathbf{z}} = \frac{\partial L}{\partial \mathbf{a}} \odot \phi'(\mathbf{z})$$

| 활성화 | $\phi(z)$ | $\phi'(z)$ |
|--------|----------|-----------|
| ReLU | $\max(0, z)$ | $\mathbf{1}[z > 0]$ |
| 시그모이드 | $\sigma(z)$ | $\sigma(z)(1 - \sigma(z))$ |
| Tanh | $\tanh(z)$ | $1 - \tanh^2(z)$ |

---

## 5. 벡터-야코비안 곱 (VJP)

역전파는 전체 야코비안 행렬을 명시적으로 형성하지 않습니다. 대신 **벡터-야코비안 곱** (VJP)을 계산합니다:

$$\bar{\mathbf{x}} = \bar{\mathbf{y}}^\top \mathbf{J}$$

여기서 $\bar{\mathbf{y}} = \frac{\partial L}{\partial \mathbf{y}}$는 상류 그래디언트이고 $\mathbf{J} = \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$는 국소 야코비안입니다.

---

## 6. 그래디언트 흐름 병리

### 6.1 그래디언트 소실

활성화 도함수가 $< 1$일 때 (시그모이드 출력은 $(0, 0.25)$), 여러 이런 인수를 곱하면 그래디언트가 지수적으로 줄어듭니다.

### 6.2 완화 방법

| 문제 | 해결책 | 작동 이유 |
|------|--------|----------|
| 소실 | ReLU 활성화 | $z > 0$에서 $\text{ReLU}'(z) = 1$ (축소 없음) |
| 소실 | 잔차 연결 | 직접 경로: $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x} + f(\mathbf{x})) = \mathbf{I} + \mathbf{J}_f$ |
| 폭발 | 그래디언트 클리핑 | $\|\nabla L\|$을 임계값에서 제한 |
| 둘 다 | 배치 정규화 | 사전 활성화를 정규화, 야코비안 노름 안정화 |
| 둘 다 | 적절한 초기화 | Xavier/He 초기화로 $\text{Var}(\text{output}) \approx \text{Var}(\text{input})$ 설정 |

---

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| 다변수 연쇄 법칙 | $\nabla_\mathbf{x} L = \mathbf{J}^\top \nabla_\mathbf{y} L$ |
| 계산 그래프 | 연산의 DAG; 순전파가 값을 저장, 역전파가 그래디언트를 전파 |
| 순방향 모드 | 입력당 한 패스; $n$ 매개변수에 $O(n)$ |
| 역방향 모드 | 출력당 한 패스; 스칼라 손실에 $O(1)$ -- 이것이 역전파 |
| VJP | $\bar{\mathbf{x}} = \bar{\mathbf{y}}^\top \mathbf{J}$: 역전파가 전체 야코비안을 형성하지 않는 방법 |
| 그래디언트 병리 | 반복된 야코비안 곱셈으로 인한 소실/폭발 그래디언트 |

---

## 연습문제

1. `Value` 클래스를 `__sub__`, `__pow__`, `tanh`를 올바른 backward 메서드와 함께 지원하도록 확장하세요.
2. 3층 네트워크(은닉층 2개)의 역전파를 구현하고 유한 차분으로 검증하세요.
3. 교차 엔트로피 손실 $L = -\sum y_i \log \hat{y}_i$의 로짓(소프트맥스 이전)에 대한 그래디언트를 계산하세요.
4. 50층 네트워크에서 시그모이드 vs ReLU vs tanh 활성화의 그래디언트 노름을 비교하세요.
5. 잔차 블록 $\mathbf{y} = \mathbf{x} + f(\mathbf{x})$를 구현하고 그래디언트 소실을 완화함을 보이세요.

---

**다음**: [04. 야코비안과 헤시안](04_Jacobian_and_Hessian.md)
