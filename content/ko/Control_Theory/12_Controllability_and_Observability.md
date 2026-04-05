# 레슨 12: 가제어성과 가관측성(Controllability and Observability)

## 학습 목표

- LTI 시스템의 가제어성(controllability)과 가관측성(observability)을 정의하고 판별한다
- 가제어 행렬(controllability matrix)과 가관측 행렬(observability matrix)을 구성하고 계수(rank)를 계산한다
- PBH(Popov-Belevitch-Hautus) 판별법을 이해한다
- 가제어성/가관측성과 전달 함수(transfer function)의 극-영점 소거(pole-zero cancellation) 간의 관계를 파악한다
- 시스템을 가제어/불가제어(uncontrollable) 및 가관측/불가관측(unobservable) 부분으로 분해한다

## 1. 동기(Motivation)

전달 함수 해석은 **외부에서 관측 가능한** 거동만을 포착한다. 상태 공간(state-space) 해석은 다음을 밝혀 준다:

- **가제어성(Controllability)**: 입력 $u(t)$가 상태 $x(t)$를 원하는 임의의 값으로 구동할 수 있는가?
- **가관측성(Observability)**: $y(t)$와 $u(t)$ 측정값으로부터 내부 상태 $x(t)$를 결정할 수 있는가?

이 두 성질은 제어기(controller)와 관측기(observer) 설계의 근간이다. 가제어하지 않은 상태는 영향을 줄 수 없고, 가관측하지 않은 상태는 추정할 수 없다.

## 2. 가제어성(Controllability)

### 2.1 정의

시스템 $(A, B)$가 **가제어(controllable)**하다는 것은, 임의의 초기 상태 $x(0) = x_0$와 임의의 목표 상태 $x_f$에 대해, 유한 시간 $t_f > 0$와 상태를 $x_0$에서 $x_f$로 구동하는 입력 $u(t)$가 존재함을 의미한다.

### 2.2 가제어 행렬(Controllability Matrix)

**정리:** 시스템 $(A, B)$가 가제어인 것과 **가제어 행렬(controllability matrix)**이 완전 계수(full rank)를 갖는 것은 동치이다:

$$\mathcal{C} = \begin{bmatrix} B & AB & A^2B & \cdots & A^{n-1}B \end{bmatrix}$$

$$\text{rank}(\mathcal{C}) = n$$

단일 입력 시스템의 경우 $\mathcal{C}$는 $n \times n$ 행렬이며, 가제어성은 $\det(\mathcal{C}) \neq 0$을 요구한다.

### 2.3 예시

시스템 $A = \begin{bmatrix} 0 & 1 \\ -2 & -3 \end{bmatrix}$, $B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$에 대해:

$$\mathcal{C} = \begin{bmatrix} B & AB \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ 1 & -3 \end{bmatrix}$$

$\det(\mathcal{C}) = 0 \cdot (-3) - 1 \cdot 1 = -1 \neq 0$ → **가제어(Controllable)**.

### 2.4 불가제어 예시

$A = \begin{bmatrix} -1 & 0 \\ 0 & -2 \end{bmatrix}$, $B = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$:

$$\mathcal{C} = \begin{bmatrix} 1 & -1 \\ 0 & 0 \end{bmatrix}$$

$\text{rank}(\mathcal{C}) = 1 < 2$ → **불가제어(Not controllable)**. 두 번째 상태 $x_2$는 입력에 관계없이 $\dot{x}_2 = -2x_2$로 진화한다 — 영향을 줄 수 없다.

## 3. 가관측성(Observability)

### 3.1 정의

시스템 $(A, C)$가 **가관측(observable)**하다는 것은, 유한 시간 구간 $[0, t_f]$ 동안의 출력 $y(t)$와 입력 $u(t)$로부터 초기 상태 $x(0)$를 유일하게 결정할 수 있음을 의미한다.

### 3.2 가관측 행렬(Observability Matrix)

**정리:** 시스템 $(A, C)$가 가관측인 것과 **가관측 행렬(observability matrix)**이 완전 계수(full rank)를 갖는 것은 동치이다:

$$\mathcal{O} = \begin{bmatrix} C \\ CA \\ CA^2 \\ \vdots \\ CA^{n-1} \end{bmatrix}$$

$$\text{rank}(\mathcal{O}) = n$$

### 3.3 쌍대성(Duality)

가제어성과 가관측성 사이에는 근본적인 **쌍대성(duality)**이 존재한다:

$$(A, B) \text{ is controllable} \iff (A^T, B^T) \text{ is observable}$$

$$(A, C) \text{ is observable} \iff (A^T, C^T) \text{ is controllable}$$

이는 가제어성에 관한 모든 정리가 가관측성에 대한 쌍대 정리를 가짐을 의미한다.

## 4. PBH 판별법(PBH Test)

### 4.1 PBH 가제어성 판별법

$(A, B)$가 가제어인 것과 다음 조건은 동치이다:

$$\text{rank}\begin{bmatrix} sI - A & B \end{bmatrix} = n \quad \forall s \in \mathbb{C}$$

동치 조건으로, $A$의 좌 고유벡터(left eigenvector) $q^T$가 존재하여 $q^T B = 0$이면 불가제어이다:

$$q^T A = \lambda q^T \text{ and } q^T B = 0 \implies \text{not controllable}$$

**해석:** 어떤 모드의 고유벡터가 $B$에 직교하면 그 모드는 불가제어이다.

### 4.2 PBH 가관측성 판별법

$(A, C)$가 가관측인 것과 다음 조건은 동치이다:

$$\text{rank}\begin{bmatrix} sI - A \\ C \end{bmatrix} = n \quad \forall s \in \mathbb{C}$$

동치 조건으로, $A$의 고유벡터(eigenvector) $v$가 존재하여 $Cv = 0$이면 불가관측이다:

$$Av = \lambda v \text{ and } Cv = 0 \implies \text{not observable}$$

**해석:** 어떤 모드의 고유벡터가 $C$의 영공간(null space)에 속하면 그 모드는 불가관측이다.

## 5. 전달 함수와의 연관성

### 5.1 극-영점 소거(Pole-Zero Cancellations)

전달 함수 $G(s) = C(sI-A)^{-1}B + D$는 극-영점 소거가 발생하면 상태 공간 모델보다 낮은 차수를 가질 수 있다.

**핵심 정리:** $G(s)$에서 극-영점 소거는 **불가제어** 또는 **불가관측**(혹은 둘 다)인 모드에 대응한다.

### 5.2 예시

다음 시스템을 고려하자:

$$A = \begin{bmatrix} -1 & 0 \\ 0 & -3 \end{bmatrix}, \quad B = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \quad C = \begin{bmatrix} 1 & 0 \end{bmatrix}$$

전달 함수:

$$G(s) = C(sI-A)^{-1}B = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} \frac{1}{s+1} & 0 \\ 0 & \frac{1}{s+3} \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \frac{1}{s+1}$$

$s = -3$의 극이 전달 함수에 나타나지 않는다. 확인해 보면: 시스템은 가제어하지만(두 상태 모두 $B$에 의해 여기됨), $C = [1 \; 0]$은 $x_2$를 관측하지 않으므로 → $s = -3$에서의 모드는 **불가관측(unobservable)**이다.

### 5.3 내부 안정성(Internal Stability) 대 BIBO 안정성

- **BIBO 안정성(BIBO stability)**은 전달 함수의 극(외부에서 가시적)에 의존한다
- **내부 안정성(Internal stability)**은 $A$의 고유값(모든 모드)에 의존한다

불안정한 모드가 극-영점 소거로 숨겨지면, 시스템은 BIBO 안정적이면서 내부적으로 불안정할 수 있다. 이는 위험한 상황이다 — 숨겨진 불안정 모드는 내부적으로 무한히 성장하게 된다.

## 6. 칼만 분해(Kalman Decomposition)

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

**가제어이고 가관측인** 부분 시스템만이 전달 함수에 나타난다. 나머지 세 부분은 입력-출력 관점에서 숨겨진다.

실현(realization) $(A, B, C, D)$이 가제어이면서 동시에 가관측이면 **최소 실현(minimal realization)**이라 부른다 — 주어진 전달 함수에 대해 가능한 가장 작은 상태 차원을 갖는다.

## 7. 가제어성 및 가관측성 그라미안(Gramians)

### 7.1 가제어성 그라미안(Controllability Gramian)

$$W_c(t) = \int_0^t e^{A\tau}BB^T e^{A^T\tau} \, d\tau$$

$(A, B)$가 가제어인 것과 어떤 $t > 0$에 대해 $W_c(t) > 0$(양정치, positive definite)인 것은 동치이다.

안정한 시스템에 대해, 무한 지평선(infinite-horizon) 가제어성 그라미안 $W_c = \int_0^\infty e^{A\tau}BB^T e^{A^T\tau} d\tau$는 다음 **Lyapunov 방정식**을 만족한다:

$$AW_c + W_c A^T + BB^T = 0$$

### 7.2 가관측성 그라미안(Observability Gramian)

$$W_o(t) = \int_0^t e^{A^T\tau}C^T C e^{A\tau} \, d\tau$$

$(A, C)$가 가관측인 것과 어떤 $t > 0$에 대해 $W_o(t) > 0$인 것은 동치이다.

그라미안은 각 상태가 얼마나 **쉽게** 제어되거나 관측될 수 있는지를 정량화한다 — 모델 축소(model reduction)의 균형 절단법(balanced truncation)에 활용된다.

## 연습 문제

### 연습 1: 가제어성 및 가관측성 확인

다음 시스템에 대해:

$$A = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ -6 & -11 & -6 \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}, \quad C = \begin{bmatrix} 1 & 0 & 0 \end{bmatrix}$$

1. 가제어 행렬을 계산하고 시스템의 가제어 여부를 판별하라
2. 가관측 행렬을 계산하고 시스템의 가관측 여부를 판별하라
3. 전달 함수를 구하고 극-영점 소거가 없음을 확인하라

### 연습 2: PBH 판별법

$A = \begin{bmatrix} -2 & 1 \\ 0 & -2 \end{bmatrix}$, $B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$, $C = \begin{bmatrix} 1 & 0 \end{bmatrix}$인 시스템에 대해:

1. $s = -2$에서 가제어성에 대한 PBH 판별법을 적용하라
2. $s = -2$에서 가관측성에 대한 PBH 판별법을 적용하라
3. 전달 함수를 구하라 — 이것이 최소 실현인가?

### 연습 3: 숨겨진 모드(Hidden Modes)

$A = \begin{bmatrix} -1 & 0 \\ 0 & 2 \end{bmatrix}$, $B = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$, $C = \begin{bmatrix} 1 & 0 \end{bmatrix}$를 고려하라:

1. 전달 함수 관점에서 시스템이 BIBO 안정적인가?
2. 시스템이 내부적으로 안정한가?
3. 이 시스템의 위험성은 무엇인가?

---

*이전: [레슨 11 — 상태 공간 표현](11_State_Space_Representation.md) | 다음: [레슨 13 — 상태 피드백 및 관측기 설계](13_State_Feedback_and_Observers.md)*
