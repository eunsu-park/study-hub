# 레슨 12: 가제어성과 가관측성(Controllability and Observability)

## 학습 목표

- LTI 시스템의 가제어성(controllability)과 가관측성(observability)을 정의하고 판별한다
- 가제어성·가관측성 행렬을 구성하고 rank를 계산한다
- PBH (Popov-Belevitch-Hautus) 판별법을 이해한다
- 가제어성/가관측성을 전달함수 극점-영점 소거와 연관 짓는다
- 시스템을 가제어/비가제어 및 가관측/비가관측 부분으로 분해한다
- 수치적으로 rank 검사를 수행하고 불안정성을 숨기는 rank-deficient 경우를 인식한다

## 0. 이 두 성질이 현대 제어를 떠받치는 이유

레슨 11이 상태 공간을 소개했고, 이번 레슨은 그것으로 유용한 일을 할 수 있는지 결정하는 두 질문을 소개한다. 이 둘은 이후 모든 제어기와 관측기 설계의 전제조건이다.

- **가제어성은 "원하는 곳으로 상태를 옮길 수 있는가?"**다. 가제어성이 없으면 어떤 $u(t)$ 선택 — 이득, 스케줄, 최적 — 도 막힌 모드를 움직이지 못한다. 제어 엔지니어의 일은 시작하기 전에 끝난다.
- **가관측성은 "보이는 것으로부터 상태를 알아낼 수 있는가?"**다. 가관측성이 없으면 어떤 관측기나 필터도 측정값에서 숨은 상태를 복원할 수 없다. 추정기의 일은 시작하기 전에 끝난다.
- **둘이 함께 최소성(minimality)을 정의한다.** 가제어이고 가관측인 시스템은 입출력 거동에 대한 가장 작은 가능한 상태를 가진다. 그보다 큰 것은 낭비된 상태 — 수학적으로 존재하지만 출력에 영향을 주거나 입력에 의해 영향받을 수 없는 모드 — 이다.

이 성질들은 또한 이미 본 역설을 설명한다: 전달함수와 상태 공간 모델이 안정성에 대해 다른 답을 줄 수 있다. 상태 공간 관점이 정직하다; 전달함수는 극점-영점 소거를 통해 비가제어 또는 비가관측 모드를 조용히 떨어뜨린다. 숨은 모드가 불안정하면 BIBO 안정성은 거짓말이다.

## 1. 동기

전달함수 분석은 **외부에 보이는** 거동만 포착한다. 상태 공간 분석은 다음을 드러낸다:

- **가제어성**: 입력 $u(t)$가 상태 $x(t)$를 임의의 원하는 값으로 끌고 갈 수 있는가?
- **가관측성**: $y(t)$와 $u(t)$ 측정으로부터 내부 상태 $x(t)$를 결정할 수 있는가?

이 성질들은 제어기 및 관측기 설계의 기본이다. 가제어가 아닌 상태는 영향받을 수 없고; 가관측이 아닌 상태는 추정될 수 없다.

## 2. 가제어성

### 2.1 정의

시스템 $(A, B)$가 **가제어**라 함은, 임의의 초기 상태 $x(0) = x_0$와 임의의 원하는 최종 상태 $x_f$에 대해, 상태를 $x_0$에서 $x_f$로 끌고 가는 유한 시간 $t_f > 0$과 입력 $u(t)$가 존재하는 것이다.

### 2.2 가제어성 행렬

**정리:** 시스템 $(A, B)$가 가제어일 필요충분조건은 **가제어성 행렬(controllability matrix)**이 full rank인 것이다:

$$\mathcal{C} = \begin{bmatrix} B & AB & A^2B & \cdots & A^{n-1}B \end{bmatrix}$$

$$\text{rank}(\mathcal{C}) = n$$

단일 입력 시스템의 경우, $\mathcal{C}$는 $n \times n$이고 가제어성은 $\det(\mathcal{C}) \neq 0$을 요구한다.

### 2.3 예제

$A = \begin{bmatrix} 0 & 1 \\ -2 & -3 \end{bmatrix}$, $B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$:

$$\mathcal{C} = \begin{bmatrix} B & AB \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ 1 & -3 \end{bmatrix}$$

$\det(\mathcal{C}) = 0 \cdot (-3) - 1 \cdot 1 = -1 \neq 0$ → **가제어**.

### 2.4 비가제어 예제

$A = \begin{bmatrix} -1 & 0 \\ 0 & -2 \end{bmatrix}$, $B = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$:

$$\mathcal{C} = \begin{bmatrix} 1 & -1 \\ 0 & 0 \end{bmatrix}$$

$\text{rank}(\mathcal{C}) = 1 < 2$ → **비가제어**. 두 번째 상태 $x_2$는 입력과 무관하게 $\dot{x}_2 = -2x_2$로 진화한다 — 영향받을 수 없다.

## 3. 가관측성

### 3.1 정의

시스템 $(A, C)$가 **가관측**이라 함은, 유한 시간 구간 $[0, t_f]$에서 출력 $y(t)$와 입력 $u(t)$로부터 초기 상태 $x(0)$가 유일하게 결정될 수 있는 것이다.

### 3.2 가관측성 행렬

**정리:** 시스템 $(A, C)$가 가관측일 필요충분조건은 **가관측성 행렬(observability matrix)**이 full rank인 것이다:

$$\mathcal{O} = \begin{bmatrix} C \\ CA \\ CA^2 \\ \vdots \\ CA^{n-1} \end{bmatrix}$$

$$\text{rank}(\mathcal{O}) = n$$

### 3.3 쌍대성(Duality)

가제어성과 가관측성 사이에는 근본적 **쌍대성**이 있다:

$$(A, B) \text{는 가제어} \iff (A^T, B^T) \text{는 가관측}$$

$$(A, C) \text{는 가관측} \iff (A^T, C^T) \text{는 가제어}$$

이는 가제어성에 대한 모든 정리가 가관측성에 대한 쌍대 정리를 가짐을 의미.

## 4. PBH 판별법

### 4.1 PBH 가제어성 판별

$(A, B)$가 가제어일 필요충분조건:

$$\text{rank}\begin{bmatrix} sI - A & B \end{bmatrix} = n \quad \forall s \in \mathbb{C}$$

동등하게, $q^T B = 0$인 $A$의 좌고유벡터(left eigenvector) $q^T$가 없는 것:

$$q^T A = \lambda q^T \text{ 이고 } q^T B = 0 \implies \text{비가제어}$$

**해석:** 모드의 고유벡터가 $B$에 직교하면 그 모드는 비가제어.

### 4.2 PBH 가관측성 판별

$(A, C)$가 가관측일 필요충분조건:

$$\text{rank}\begin{bmatrix} sI - A \\ C \end{bmatrix} = n \quad \forall s \in \mathbb{C}$$

동등하게, $Cv = 0$인 $A$의 고유벡터 $v$가 없는 것:

$$Av = \lambda v \text{ 이고 } Cv = 0 \implies \text{비가관측}$$

**해석:** 모드의 고유벡터가 $C$의 영공간에 있으면 그 모드는 비가관측.

### 4.3 두 판별법이 존재하는 이유

Kalman rank 판별은 계산이 더 빠르다(하나의 rank, $n \times nm$ 행렬). PBH 판별은 해석이 더 빠르다(어느 모드가 문제인지 고유값으로 식별). 실무에서는 Kalman으로 문제를 감지하고 PBH로 어떤 모드가 원인인지 진단한다.

## 5. 전달함수와의 관계

### 5.1 극점-영점 소거

전달함수 $G(s) = C(sI-A)^{-1}B + D$는 극점-영점 소거가 있으면 상태 공간 모델보다 낮은 차수일 수 있다.

**핵심 정리:** $G(s)$의 극점-영점 소거는 **비가제어** 또는 **비가관측** (또는 둘 다)인 모드에 대응한다.

### 5.2 예제

다음을 고려:

$$A = \begin{bmatrix} -1 & 0 \\ 0 & -3 \end{bmatrix}, \quad B = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \quad C = \begin{bmatrix} 1 & 0 \end{bmatrix}$$

전달함수:

$$G(s) = C(sI-A)^{-1}B = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} \frac{1}{s+1} & 0 \\ 0 & \frac{1}{s+3} \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \frac{1}{s+1}$$

$s = -3$의 극점이 전달함수에 나타나지 않는다. 확인: 시스템은 가제어($B$가 두 상태를 모두 자극)이지만 $C = [1 \; 0]$은 $x_2$를 관측하지 않는다 → $s = -3$ 모드는 **비가관측**.

### 5.3 내부 안정성 vs. BIBO 안정성

- **BIBO 안정성**은 전달함수 극점에 의존(외부에 보임)
- **내부 안정성(internal stability)**은 $A$의 고유값에 의존(모든 모드)

극점-영점 소거에 의해 불안정 모드가 숨겨지면 시스템은 BIBO 안정이지만 내부적으로 불안정할 수 있다. 위험하다 — 숨은 불안정 모드가 내부적으로 무한히 증가한다.

### 5.4 파이썬으로 수치 검사

```python
import numpy as np
from scipy.linalg import matrix_rank

A = np.array([[-1, 0], [0, -3]], dtype=float)
B = np.array([[1], [1]], dtype=float)
C = np.array([[1, 0]], dtype=float)
n = A.shape[0]

# Kalman 가제어성 및 가관측성 행렬
Ctrb = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])
Obsv = np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)])

print(f"rank(Ctrb) = {matrix_rank(Ctrb)} / {n}  → {'controllable' if matrix_rank(Ctrb) == n else 'NOT controllable'}")
print(f"rank(Obsv) = {matrix_rank(Obsv)} / {n}  → {'observable' if matrix_rank(Obsv) == n else 'NOT observable'}")
```

위 예제에 대해 `rank(Ctrb) = 2` (가제어) 및 `rank(Obsv) = 1` (비가관측)을 봐야 한다. `python-control`은 `ctrb`, `obsv`, `pole_zero_cancellation` 도우미를 제공하여 같은 로직을 더 친화적인 API로 감싼다.

> **수치적 주의**: 부동소수점 행렬에 대한 rank 테스트는 임계값에 민감하다. 행렬이 잘 스케일되지 않을 때 `numpy.linalg.matrix_rank(M, tol=...)`에 $\sigma_1 \cdot \epsilon \cdot \max(m,n)$에 기반한 명시적 허용오차를 사용하라. 더 나은 방법: SVD를 직접 사용하고 특이값 격차를 보라 — $\sigma_n / \sigma_1 = 0.001$인 "rank-deficient" 행렬은 "비가제어"라기보다는 "겨우 가제어"이며 약한 모드를 끌고 가는 데 더 많은 제어 노력이 필요할 것이다.

## 6. Kalman 분해(Kalman Decomposition)

임의의 LTI 시스템은 네 부분으로 분해될 수 있다:

```
┌──────────────────────────────────────────┐
│    ┌────────────┐    ┌────────────┐      │
│    │ Controllable│    │ Controllable│      │
│    │ Observable  │ →  │ Unobservable│      │
│    └────────────┘    └────────────┘      │
│         ↓                  ↓             │
│    ┌────────────┐    ┌────────────┐      │
│    │Uncontrollable│  │Uncontrollable│    │
│    │ Observable  │    │ Unobservable│     │
│    └────────────┘    └────────────┘      │
└──────────────────────────────────────────┘
```

전달함수에 나타나는 것은 **가제어 및 가관측** 부분 시스템뿐이다. 다른 세 부분은 입출력 관점에서 숨겨져 있다.

실현 $(A, B, C, D)$가 가제어이고 가관측이면 **최소(minimal)**라 부른다 — 주어진 전달함수에 대한 가능한 가장 작은 상태 차원을 가진다.

## 7. 가제어성 및 가관측성 Gramian

### 7.1 가제어성 Gramian

$$W_c(t) = \int_0^t e^{A\tau}BB^T e^{A^T\tau} \, d\tau$$

$(A, B)$가 가제어일 필요충분조건은 어떤 $t > 0$에 대해 $W_c(t) > 0$ (양정치)인 것.

안정 시스템에서, 무한 구간 가제어성 Gramian $W_c = \int_0^\infty e^{A\tau}BB^T e^{A^T\tau} d\tau$는 **Lyapunov 방정식**을 만족:

$$AW_c + W_c A^T + BB^T = 0$$

### 7.2 가관측성 Gramian

$$W_o(t) = \int_0^t e^{A^T\tau}C^T C e^{A\tau} \, d\tau$$

$(A, C)$가 가관측일 필요충분조건은 어떤 $t > 0$에 대해 $W_o(t) > 0$인 것.

Gramian은 각 상태가 **얼마나 쉽게** 제어되거나 관측될 수 있는지를 정량화한다 — 모델 축소(balanced truncation)에 사용된다.

## 8. 흔한 함정

1. **부동소수점에서 "rank-deficient"를 이진 판정으로 취급.** 실제 시스템은 "매우 가제어"(잘 조건화된 $\mathcal{C}$)에서 "겨우 가제어"(거의 rank-deficient)까지 스펙트럼에 있다. 항상 $\mathcal{C}$의 가장 작은 특이값을 검사하라 — 작은 값은 약한 모드를 움직이는 데 막대한 제어 노력이 필요함을 의미한다.
2. **최소성이 전달함수가 아니라 실현 단위라는 것을 잊음.** 같은 전달함수의 두 실현은 하나가 숨은 모드를 가지면 다른 상태 차원을 가질 수 있다. 최소 실현은 가장 작은 것이다.
3. **"비가제어이지만 안정"인 경우 무시.** 비가제어 모드가 LHP에 있으면 영향을 줄 수 없지만 무해하게 감쇠한다. 많은 물리 시스템이 그런 모드를 가진다(예: 스프링-질량-댐퍼를 노드에서 측정 — 자연 모드가 거기서 보이지 않음). 감지하되 항상 결정적인 문제는 아니다.
4. **안전-중요 분석에 BIBO 안정성 신뢰.** 항공 인증이 상태 공간 분석을 요구하는 이유가 정확히 이 레슨이다: 입출력 거동이 멀쩡해 보이는 동안 숨은 RHP 모드가 비행기를 추락시킬 수 있다. 안전 논증에는 항상 $A$의 고유값을 사용하라.
5. **$n > 30$에 대해 $\mathcal{C}$를 단순하게 구축.** $A$가 유의한 크기의 고유값을 가질 때 $A^k B$는 수치적으로 폭발한다. 고차 시스템에 대해 $A^{n-1}$을 계산하는 대신 staircase 형태(`scipy.signal.ss2zpk` 더하기 controllability staircase)를 사용하라.
6. **Gramian의 양정치성을 full rank와 혼동.** 안정 시스템의 경우 $W_c > 0$ (양정치)는 가제어성과 동등하다. 불안정 시스템에는 Kalman/PBH 판별을 직접 사용 — $W_c$를 정의하는 적분이 수렴하지 않을 수 있다.

## 연습 문제

### 연습 문제 1: 가제어성·가관측성 검사

다음 시스템:

$$A = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ -6 & -11 & -6 \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}, \quad C = \begin{bmatrix} 1 & 0 & 0 \end{bmatrix}$$

1. 가제어성 행렬을 계산하고 시스템이 가제어인지 판정하라
2. 가관측성 행렬을 계산하고 시스템이 가관측인지 판정하라
3. 전달함수를 구하고 극점-영점 소거가 없는지 검증하라

### 연습 문제 2: PBH 판별

$A = \begin{bmatrix} -2 & 1 \\ 0 & -2 \end{bmatrix}$, $B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$, $C = \begin{bmatrix} 1 & 0 \end{bmatrix}$:

1. $s = -2$에서 가제어성에 대한 PBH 판별을 적용하라
2. $s = -2$에서 가관측성에 대한 PBH 판별을 적용하라
3. 전달함수를 구하라 — 이것은 최소 실현인가?

### 연습 문제 3: 숨은 모드

$A = \begin{bmatrix} -1 & 0 \\ 0 & 2 \end{bmatrix}$, $B = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$, $C = \begin{bmatrix} 1 & 0 \end{bmatrix}$:

1. 시스템은 BIBO 안정인가(전달함수에서)?
2. 시스템은 내부적으로 안정인가?
3. 이 시스템의 위험은 무엇인가?

### 연습 문제 4: 수치 점검

5.4절의 파이썬 스니펫으로 위 세 연습 문제 모두에 대해 가제어성·가관측성을 확인하라. rank 기반 답이 손으로 유도한 것과 일치하는지 확인하라. 연습 문제 3에 대해서는 추가로 $A$의 고유값을 출력하여 전달함수가 보여 주지 않는 숨은 불안정 모드를 보라.

### 연습 문제 5: 거의 비가제어 드릴

$A = \begin{bmatrix} -1 & 0 \\ 0 & -2 \end{bmatrix}$, $B = \begin{bmatrix} 1 \\ \epsilon \end{bmatrix}$, $\epsilon \in \{1, 0.1, 0.01, 0.001\}$. 시스템은 임의의 $\epsilon > 0$에 대해 완전 가제어이지만 가제어성이 점점 약해진다. 각 $\epsilon$에 대해 $\mathcal{C}$의 가장 작은 특이값을 계산하고, 그 값이 두 번째 모드를 움직이는 데 필요한 제어 노력과 어떻게 관련되는지 설명하라.

---

*이전: [레슨 11 — 상태 공간 표현](11_State_Space_Representation.md) | 다음: [레슨 13 — 상태 귀환과 관측기 설계](13_State_Feedback_and_Observers.md)*
