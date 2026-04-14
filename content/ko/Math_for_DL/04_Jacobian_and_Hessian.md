# 레슨 4: 야코비안과 헤시안

## 학습 목표

- 벡터 함수의 야코비안 행렬을 정의하고 계산한다
- 야코비안을 비선형 사상의 선형 근사로 이해한다
- 헤시안 행렬을 정의하고 스칼라 함수의 곡률로 해석한다
- 헤시안을 사용한 2차 테일러 전개를 계산한다
- 헤시안의 고유값과 손실 표면의 국소 기하학을 연결한다
- 뉴턴 방법과 헤시안의 관계를 이해한다
- 대규모 DL에서 2차 방법이 비실용적인 이유와 존재하는 근사를 인식한다
- 헤시안과 피셔 정보 행렬의 관계를 이해한다

---

## 1. 야코비안 행렬

### 1.1 정의

함수 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$의 **야코비안**은 모든 1차 편미분의 $m \times n$ 행렬입니다:

$$\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} \in \mathbb{R}^{m \times n}, \quad J_{ij} = \frac{\partial f_i}{\partial x_j}$$

행 $i$는 $f_i$의 그래디언트; 열 $j$는 $x_j$가 변할 때 모든 출력이 어떻게 변하는지 알려줍니다.

### 1.2 선형 근사로서의 야코비안

$$\mathbf{f}(\mathbf{x}_0 + \boldsymbol{\delta}) \approx \mathbf{f}(\mathbf{x}_0) + \mathbf{J}(\mathbf{x}_0) \boldsymbol{\delta}$$

### 1.3 DL 함수의 야코비안

**소프트맥스 야코비안**: $\mathbf{s} = \text{softmax}(\mathbf{z})$에 대해

$$\frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)$$

$$\mathbf{J}_{\text{softmax}} = \text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top$$

```python
import numpy as np

def compute_jacobian_numerical(f, x, eps=1e-5):
    """중심 차분으로 f: R^n -> R^m의 야코비안을 계산."""
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(f(x))
    n = len(x)
    m = len(f0)
    J = np.zeros((m, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = eps
        J[:, j] = (f(x + e_j) - f(x - e_j)) / (2 * eps)
    return J
```

---

## 2. 헤시안 행렬

### 2.1 정의

스칼라 함수 $f: \mathbb{R}^n \to \mathbb{R}$의 **헤시안**은 2차 편미분의 $n \times n$ 행렬입니다:

$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

슈바르츠 정리에 의해 $\mathbf{H}$는 **대칭**입니다.

### 2.2 곡률로서의 헤시안

2차 테일러 전개:

$$f(\mathbf{x}_0 + \boldsymbol{\delta}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top \boldsymbol{\delta} + \frac{1}{2} \boldsymbol{\delta}^\top \mathbf{H}(\mathbf{x}_0) \boldsymbol{\delta}$$

### 2.3 고유값 해석

| 헤시안 고유값 | 임계점 유형 |
|-------------|-----------|
| 모두 $\lambda_i > 0$ | 극소점 |
| 모두 $\lambda_i < 0$ | 극대점 |
| 부호 혼합 | **안장점** |
| 일부 $\lambda_i = 0$ | 퇴화 (결론 불가) |

---

## 3. 딥러닝에서의 안장점

고차원 최적화에서 안장점은 극소점보다 훨씬 흔합니다. 임계점이 극소점이 되려면 **모든** $n$개 고유값이 양이어야 합니다. 고차원에서 이것이 우연히 일어날 확률은 지수적으로 작습니다.

---

## 4. 뉴턴 방법

### 4.1 아이디어

$$\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \mathbf{H}^{-1} \nabla f(\mathbf{x}^{(t)})$$

**직관**: 최대 하강 방향을 따르는 대신, 뉴턴 방법은 국소 이차 근사의 최솟값으로 직접 점프합니다.

### 4.2 DL에서 비실용적인 이유

| 문제 | 상세 |
|------|------|
| 메모리 | $\mathbf{H} \in \mathbb{R}^{n \times n}$; $n = 10^6$이면 $\sim 4$ TB 필요 |
| 계산 | $\mathbf{H}^{-1}\mathbf{g}$ 계산에 $O(n^3)$ |
| 비볼록성 | $\mathbf{H}$가 비정부호; 뉴턴 스텝이 상승할 수 있음 |
| 확률성 | 미니배치 그래디언트가 잡음; 헤시안 추정은 더 잡음 |

---

## 5. 실용적 헤시안 근사

### 5.1 대각 헤시안 근사

$$\mathbf{H} \approx \text{diag}(H_{11}, H_{22}, \ldots, H_{nn})$$

**AdaGrad/RMSProp/Adam**에서 사용: 누적된 제곱 그래디언트가 대각 헤시안 항을 근사합니다.

### 5.2 가우스-뉴턴 근사

손실 $L = \frac{1}{2}\|\mathbf{r}(\boldsymbol{\theta})\|^2$에서 2차 항을 생략:

$$\mathbf{H} \approx \mathbf{J}_\mathbf{r}^\top \mathbf{J}_\mathbf{r}$$

항상 양 반정부호이므로 안장점 문제를 피합니다.

### 5.3 피셔 정보 행렬

$$\mathbf{F} = \mathbb{E}\left[\nabla \ell \, \nabla \ell^\top\right]$$

**자연 경사 하강법**에서 업데이트: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \mathbf{F}^{-1} \nabla L$

### 5.4 헤시안-벡터 곱

$\mathbf{H}$를 명시적으로 형성하지 않고도 임의의 벡터 $\mathbf{v}$에 대해 $\mathbf{H}\mathbf{v}$를 계산할 수 있습니다:

$$\mathbf{H}\mathbf{v} = \lim_{\epsilon \to 0} \frac{\nabla f(\mathbf{x} + \epsilon \mathbf{v}) - \nabla f(\mathbf{x})}{\epsilon}$$

두 번의 그래디언트 평가만 필요하며 $O(n)$의 비용입니다.

---

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| 야코비안 | 1차 도함수의 $m \times n$ 행렬; 비선형 사상의 최적 선형 근사 |
| 소프트맥스 야코비안 | $\text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top$; 랭크 $n-1$, 대칭 |
| 헤시안 | 2차 도함수의 $n \times n$ 행렬; 곡률 부호화 |
| 고유값 해석 | 양 = 그릇, 음 = 능선, 혼합 = 안장점 |
| 뉴턴 방법 | $\mathbf{H}^{-1} \nabla f$ 사용; 2차 수렴이지만 $O(n^2)$ 메모리 |
| HVP | 유한 차분으로 $\mathbf{H}\mathbf{v}$: $O(n)$ 비용, $\mathbf{H}$ 저장 불필요 |
| 실용적 근사 | 대각 (Adam), 가우스-뉴턴, 피셔 정보 |

---

## 연습문제

1. $\mathbf{f}(x, y) = (x^2 y, \sin(xy), e^x + y)$의 야코비안을 해석적으로 계산하고 수치적으로 검증하세요.
2. $f(x, y) = x^4 + y^4 - 2x^2 y^2$의 헤시안을 계산하고 모든 임계점을 분류하세요.
3. 로젠브록 함수에서 뉴턴 방법을 구현하고 경사 하강법과 수렴을 비교하세요.
4. $\mathbf{z} = (1, 2, 3)$에서 소프트맥스 야코비안을 계산하고 각 행의 합이 0임을 검증하세요.
5. 헤시안-벡터 곱 함수를 구현하고 거듭제곱법으로 최대 고유값을 찾으세요.

---

**다음**: [05. 최적화 이론](05_Optimization_Theory.md)
