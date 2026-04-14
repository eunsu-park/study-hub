# 레슨 2: 편미분과 그래디언트

## 학습 목표

- 다변수 함수의 편미분을 해석적, 수치적으로 계산한다
- 그래디언트 벡터를 구성하고 최대 상승 방향으로 기하학적으로 해석한다
- 방향 도함수를 계산하고 그래디언트와의 관계를 이해한다
- 그래디언트 장과 등고선을 시각화한다
- 간단한 최적화 문제에 그래디언트를 적용한다 (2차 함수에서의 경사 하강법)
- 유한 차분법을 사용하여 해석적 그래디언트를 검증한다
- 그래디언트와 1차 테일러 근사의 연결을 통해 경사 하강법이 작동하는 이유를 이해한다

---

## 1. 단일 변수에서 다변수로

단일 변수 미적분에서 도함수 $f'(x)$는 수직선을 따라 $f$의 변화율을 알려줍니다. 딥러닝에서 손실 함수는 수백만 개의 매개변수에 동시에 의존합니다. 미분을 여러 변수의 함수로 일반화해야 합니다.

### 1.1 다변수 함수

함수 $f: \mathbb{R}^n \to \mathbb{R}$은 벡터 $\mathbf{x} = (x_1, x_2, \ldots, x_n)$을 받아 스칼라를 반환합니다. 딥러닝에서:

- $\mathbf{x}$는 모든 모델 매개변수 (가중치와 편향을 하나의 벡터로 평탄화)
- $f(\mathbf{x}) = L(\mathbf{x})$는 손실 함수

### 1.2 편미분 정의

$x_i$에 대한 $f$의 **편미분**은 다른 모든 변수를 고정한 채 $x_i$만 변화시킬 때 $f$가 어떻게 변하는지 측정합니다:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_n)}{h}$$

**직관**: 산 위에 서 있다고 상상해보세요 (손실 표면). 편미분 $\frac{\partial f}{\partial x_1}$은 $x_1$ 방향으로만 걸을 때 경사가 얼마나 가파른지 알려줍니다.

### 1.3 수치적 편미분

실제로 해석적 그래디언트를 **유한 차분**으로 검증합니다:

**중심 차분**: $\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + h\mathbf{e}_i) - f(\mathbf{x} - h\mathbf{e}_i)}{2h}$ -- 정확도 $O(h^2)$

```python
import numpy as np
import matplotlib.pyplot as plt

def f_vec(x):
    """벡터 함수로서의 f."""
    return x[0]**2 + 3*x[1]**2 - 2*x[0]*x[1] + x[0] - 4*x[1] + 5

def grad_f_analytical(x):
    """해석적 그래디언트."""
    return np.array([
        2*x[0] - 2*x[1] + 1,
        6*x[1] - 2*x[0] - 4
    ])

def grad_f_numerical(x, h=1e-5):
    """중심 차분 그래디언트."""
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        grad[i] = (f_vec(x + h * e_i) - f_vec(x - h * e_i)) / (2 * h)
    return grad

# 테스트 점에서 비교
x_test = np.array([1.0, 2.0])
g_ana = grad_f_analytical(x_test)
g_num = grad_f_numerical(x_test)

print(f"해석적 그래디언트: {g_ana}")
print(f"수치적 그래디언트:  {g_num}")
print(f"최대 차이: {np.max(np.abs(g_ana - g_num)):.2e}")
```

---

## 2. 그래디언트 벡터

### 2.1 정의

$f: \mathbb{R}^n \to \mathbb{R}$의 점 $\mathbf{x}$에서의 **그래디언트**는 모든 편미분의 벡터입니다:

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

### 2.2 기하학적 해석

그래디언트는 두 가지 핵심적인 기하학적 성질을 가집니다:

1. **방향**: $\nabla f(\mathbf{x})$는 $\mathbf{x}$에서 $f$의 **최대 상승** 방향을 가리킴
2. **크기**: $\|\nabla f(\mathbf{x})\|$는 그 최대 방향에서의 변화율
3. **수직성**: $\nabla f(\mathbf{x})$는 점 $\mathbf{x}$에서 등고선 $f(\mathbf{x}) = c$에 수직

**결론**: $f$를 최소화하려면 $-\nabla f(\mathbf{x})$ 방향으로 이동해야 합니다.

### 2.3 왜 그래디언트가 오르막을 가리키는가: 1차 테일러 근사

그래디언트와 최적화의 연결은 테일러 전개에서 옵니다:

$$f(\mathbf{x} + \boldsymbol{\delta}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \boldsymbol{\delta}$$

$f$의 변화량은 대략 $\nabla f(\mathbf{x})^\top \boldsymbol{\delta}$입니다. 단위 스텝 $\|\boldsymbol{\delta}\| = 1$에 대해, 이는 $\boldsymbol{\delta}$가 $\nabla f(\mathbf{x})$에 평행할 때 최대화됩니다 (코시-슈바르츠 부등식).

$f$를 **감소**시키려면 $\boldsymbol{\delta} = -\eta \nabla f(\mathbf{x})$를 선택합니다 (작은 $\eta > 0$).

---

## 3. 방향 도함수

### 3.1 정의

단위 벡터 $\mathbf{u}$ 방향으로의 $f$의 **방향 도함수**:

$$D_\mathbf{u} f(\mathbf{x}) = \nabla f(\mathbf{x})^\top \mathbf{u}$$

이는 $\mathbf{u}$ 방향으로 걸을 때 $f$의 변화율을 측정합니다.

### 3.2 핵심 성질

- 최대 방향 도함수: $\mathbf{u} = \frac{\nabla f}{\|\nabla f\|}$ 방향, 값은 $\|\nabla f\|$
- 최소 방향 도함수: $\mathbf{u} = -\frac{\nabla f}{\|\nabla f\|}$ 방향, 값은 $-\|\nabla f\|$
- 영 방향 도함수: $\nabla f$에 수직인 방향

---

## 4. 경사 하강법

### 4.1 알고리즘

경사 하강법은 함수를 최소화하기 위해 매개변수를 반복적으로 업데이트합니다:

$$\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \eta \nabla f(\mathbf{x}^{(t)})$$

여기서 $\eta > 0$는 **학습률**입니다.

### 4.2 이차 함수에서의 경사 하강법

이차 함수 $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top \mathbf{A} \mathbf{x} - \mathbf{b}^\top \mathbf{x} + c$에서 (대칭 양정부호 $\mathbf{A}$):

$$\nabla f(\mathbf{x}) = \mathbf{A}\mathbf{x} - \mathbf{b}$$

최솟값은 $\mathbf{x}^* = \mathbf{A}^{-1}\mathbf{b}$에 있습니다.

**수렴 속도**는 **조건수** $\kappa = \lambda_{\max}(\mathbf{A}) / \lambda_{\min}(\mathbf{A})$에 의존합니다:
- $\kappa \approx 1$: 빠른 수렴 (거의 원형 등고선)
- $\kappa \gg 1$: 느린 수렴 (길쭉한 타원형 등고선)

### 4.3 학습률 선택

이차 함수에서 수렴을 위해 학습률은 $\eta < \frac{2}{\lambda_{\max}(\mathbf{A})}$를 만족해야 합니다. 최적 고정 학습률:

$$\eta^* = \frac{2}{\lambda_{\max} + \lambda_{\min}}$$

---

## 5. 일반적인 DL 함수의 그래디언트

### 5.1 시그모이드

$$\sigma(x) = \frac{1}{1 + e^{-x}}, \quad \sigma'(x) = \sigma(x)(1 - \sigma(x))$$

### 5.2 ReLU

$$\text{ReLU}(x) = \max(0, x), \quad \text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}$$

### 5.3 MSE 손실

$$L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2, \quad \frac{\partial L}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)$$

---

## 6. 고차원에서의 그래디언트

### 6.1 그래디언트 계산의 차원의 저주

$n$개 매개변수를 가진 함수에서 전체 그래디언트를 계산하려면 $n$개의 편미분이 필요합니다. 유한 차분 그래디언트는 $2n$번의 함수 평가가 필요합니다. $n \sim 10^9$인 현대 신경망에서는 이것이 비현실적입니다.

**이것이 역전파 (역방향 자동 미분)가 중요한 이유입니다** -- $n$에 관계없이 단일 순전파에 비례하는 시간으로 전체 그래디언트를 계산합니다. 이것은 레슨 03에서 유도합니다.

### 6.2 학습 중 그래디언트 노름

학습 중 $\|\nabla L\|$을 모니터링하면 중요한 동역학을 드러냅니다:

- **그래디언트 소실**: $\|\nabla L\| \to 0$ 조기에, 학습이 정체
- **그래디언트 폭발**: $\|\nabla L\| \to \infty$, 학습이 발산
- **건강한 학습**: $\|\nabla L\|$이 최솟값에 접근하면서 점진적으로 감소

---

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| 편미분 | 다른 변수를 고정한 채 하나의 변수를 변화시킬 때 $f$의 변화율 |
| 그래디언트 | 모든 편미분의 벡터; 최대 상승 방향을 가리킴 |
| 방향 도함수 | $D_\mathbf{u} f = \nabla f \cdot \mathbf{u}$; 그래디언트를 방향 $\mathbf{u}$에 투영 |
| 경사 하강법 | $\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f$; 수렴은 조건수에 의존 |
| 유한 차분 | 중심 차분은 $O(h^2)$ 정확도; 그래디언트 검사에 사용 |
| 그래디언트 모니터링 | $\|\nabla L\|$을 추적하여 소실/폭발 그래디언트 감지 |

---

## 연습문제

1. $f(x_1, x_2, x_3) = x_1 x_2 + x_2 x_3^2 - \ln(x_1)$의 그래디언트를 해석적으로 계산하고 수치적으로 검증하세요.
2. 로젠브록 함수 $f(x, y) = (1 - x)^2 + 100(y - x^2)^2$를 최소화하는 경사 하강법을 구현하세요.
3. 로젠브록 함수에서 다양한 학습률을 실험하고 수렴 곡선을 그리세요.
4. $f(x, y) = e^{xy} + \sin(x + y)$의 $(0, \pi)$에서 $(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}})$ 방향의 방향 도함수를 계산하세요.
5. 해석적 그래디언트와 수치적 그래디언트를 상대 오차 메트릭으로 비교하는 그래디언트 검사기를 구현하세요.

---

**다음**: [03. 연쇄 법칙과 계산 그래프](03_Chain_Rule_and_Computation_Graphs.md)
