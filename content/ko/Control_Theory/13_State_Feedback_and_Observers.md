# 레슨 13: 상태 피드백 및 관측기 설계(State Feedback and Observer Design)

## 학습 목표

- 극배치(pole placement)를 이용한 상태 피드백 제어기(state feedback controller)를 설계한다
- 제어기-관측기 결합 설계를 위한 분리 원리(separation principle)를 이해한다
- 상태 추정을 위한 완전 차수 루엔버거 관측기(full-order Luenberger observer)를 설계한다
- 관측기 기반 상태 피드백 제어기를 구현한다
- SISO 극배치를 위한 아커만 공식(Ackermann's formula)을 적용한다

## 1. 상태 피드백 제어(State Feedback Control)

### 1.1 완전 상태 피드백(Full State Feedback)

모든 상태가 측정 가능하다면 **상태 피드백(state feedback)**을 사용할 수 있다:

$$u(t) = -Kx(t) + r(t)$$

여기서 $K \in \mathbb{R}^{m \times n}$은 **피드백 이득 행렬(feedback gain matrix)**이고, $r(t)$는 기준 명령(reference command)이다.

폐루프(closed-loop) 시스템은 다음과 같이 된다:

$$\dot{x} = (A - BK)x + Br$$
$$y = Cx$$

폐루프 극(closed-loop poles)은 $A - BK$의 고유값이다.

### 1.2 극배치 정리(Pole Placement Theorem)

**정리:** 시스템 $(A, B)$가 가제어(controllable)하면, $K$의 적절한 선택으로 $A - BK$의 고유값을 **임의의** 원하는 위치에 배치할 수 있다.

이것이 핵심 결과이다: 가제어성은 임의의 극배치를 보장한다.

### 1.3 설계 절차(SISO)

**주어진 것:** 원하는 폐루프 극 $s_1, s_2, \ldots, s_n$.

**원하는 특성 다항식(desired characteristic polynomial):**

$$\Delta_d(s) = (s - s_1)(s - s_2)\cdots(s - s_n) = s^n + \alpha_{n-1}s^{n-1} + \cdots + \alpha_0$$

**방법 1: 직접 비교(Direct comparison)**

1. $\det(sI - A + BK) = s^n + f_{n-1}(K)s^{n-1} + \cdots + f_0(K)$를 계산한다
2. $i = 0, \ldots, n-1$에 대해 $f_i(K) = \alpha_i$로 놓는다
3. $K = [k_1 \; k_2 \; \cdots \; k_n]$에 대한 $n$개의 방정식을 풀어 구한다

**방법 2: 가제어 정규형으로 변환(Transform to CCF)**

시스템이 가제어 정규형(controllable canonical form, CCF)으로 표현되면 이득은 단순히:

$$K = [\alpha_0 - a_0 \;\; \alpha_1 - a_1 \;\; \cdots \;\; \alpha_{n-1} - a_{n-1}]$$

여기서 $a_i$는 원래 특성 다항식의 계수이다.

### 1.4 아커만 공식(Ackermann's Formula, SISO)

$$K = \begin{bmatrix} 0 & 0 & \cdots & 0 & 1 \end{bmatrix} \mathcal{C}^{-1} \Delta_d(A)$$

여기서 $\mathcal{C}$는 가제어 행렬(controllability matrix)이고, $\Delta_d(A)$는 원하는 특성 다항식을 행렬 $A$에서 평가한 것이다:

$$\Delta_d(A) = A^n + \alpha_{n-1}A^{n-1} + \cdots + \alpha_1 A + \alpha_0 I$$

### 1.5 예시

$A = \begin{bmatrix} 0 & 1 \\ -2 & -3 \end{bmatrix}$, $B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$

원하는 극: $s = -5, -5$ → $\Delta_d(s) = s^2 + 10s + 25$

현재: $\Delta(s) = s^2 + 3s + 2$

CCF에서 (이 시스템은 이미 $a_0 = 2, a_1 = 3$인 CCF 형태):

$$K = [25 - 2 \;\; 10 - 3] = [23 \;\; 7]$$

검증: $A - BK = \begin{bmatrix} 0 & 1 \\ -2-23 & -3-7 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ -25 & -10 \end{bmatrix}$ 이며 고유값은 $-5, -5$이다. ✓

## 2. 상태 피드백을 이용한 기준 추적(Reference Tracking with State Feedback)

상태 피드백 $u = -Kx$는 출력을 영으로 구동할 뿐, 영이 아닌 기준값으로는 추적하지 않는다. 계단 기준(step reference) $r$을 추적하려면:

### 2.1 전향 이득(Feedforward Gain)

$$u = -Kx + N_r r$$

여기서 $N_r$은 정상 상태에서 $y_{ss} = r$이 되도록 선택한다:

$$N_r = \frac{1}{C(-A + BK)^{-1}B}$$

동치로, 폐루프의 DC 이득을 이용하면: $N_r = 1/G_{cl}(0)$.

### 2.2 적분 동작(Integral Action)

적분기 상태 $x_I = \int (r - y) dt$를 추가한다:

$$\dot{x}_I = r - Cx$$

증가 시스템(augmented system):

$$\begin{bmatrix} \dot{x} \\ \dot{x}_I \end{bmatrix} = \begin{bmatrix} A & 0 \\ -C & 0 \end{bmatrix} \begin{bmatrix} x \\ x_I \end{bmatrix} + \begin{bmatrix} B \\ 0 \end{bmatrix} u + \begin{bmatrix} 0 \\ 1 \end{bmatrix} r$$

피드백 $u = -[K \; K_I] [x^T \; x_I]^T$를 사용하면 적분기가 계단 기준 및 계단 외란(step disturbance)에 대한 **정상 상태 오차 영(zero steady-state error)**을 보장한다.

## 3. 관측기 설계(Observer Design)

### 3.1 동기

실제 상황에서는 모든 상태가 측정 가능하지 않다. **관측기(observer)** 또는 **추정기(estimator)**는 측정된 출력과 알려진 입력으로부터 상태를 재구성한다.

### 3.2 완전 차수 루엔버거 관측기(Full-Order Luenberger Observer)

$$\dot{\hat{x}} = A\hat{x} + Bu + L(y - C\hat{x})$$

여기서 $\hat{x}$는 추정 상태이고, $L \in \mathbb{R}^{n \times p}$는 **관측기 이득 행렬(observer gain matrix)**이다.

추정 오차(estimation error) $e = x - \hat{x}$는 다음을 만족한다:

$$\dot{e} = (A - LC)e$$

$A - LC$의 모든 고유값이 음의 실수부를 가지면 $e(t) \to 0$이 지수적으로 수렴한다.

### 3.3 관측기 극배치(Observer Pole Placement)

**정리:** 시스템 $(A, C)$가 가관측(observable)하면, $L$의 적절한 선택으로 $A - LC$의 고유값을 **임의의** 원하는 위치에 배치할 수 있다.

이것은 상태 피드백 극배치 정리의 **쌍대(dual)**이다.

**설계 경험칙:** 관측기 극을 제어기 극보다 3-5배 빠르게 배치한다(추정값이 제어기가 정확한 상태 정보를 필요로 하기 전에 수렴하도록).

### 3.4 관측기 이득에 대한 아커만 공식(Ackermann's Formula for Observer Gain)

$$L = \Delta_o(A) \mathcal{O}^{-1} \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix}$$

여기서 $\mathcal{O}$는 가관측 행렬(observability matrix)이고, $\Delta_o(A)$는 원하는 관측기 특성 다항식을 $A$에서 평가한 것이다.

### 3.5 예시

같은 시스템 사용: $A = \begin{bmatrix} 0 & 1 \\ -2 & -3 \end{bmatrix}$, $C = \begin{bmatrix} 1 & 0 \end{bmatrix}$

원하는 관측기 극: $s = -15, -15$ → $\Delta_o(s) = s^2 + 30s + 225$

$$\mathcal{O} = \begin{bmatrix} C \\ CA \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$

$$L = \Delta_o(A) \mathcal{O}^{-1} \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

$$\Delta_o(A) = A^2 + 30A + 225I = \begin{bmatrix} -2 & -3 \\ 6 & 7 \end{bmatrix} + \begin{bmatrix} 0 & 30 \\ -60 & -90 \end{bmatrix} + \begin{bmatrix} 225 & 0 \\ 0 & 225 \end{bmatrix} = \begin{bmatrix} 223 & 27 \\ -54 & 142 \end{bmatrix}$$

$$L = \begin{bmatrix} 223 & 27 \\ -54 & 142 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 27 \\ 142 \end{bmatrix}$$

## 4. 분리 원리(The Separation Principle)

### 4.1 결합 제어기-관측기(Combined Controller-Observer)

관측기 기반 상태 피드백 제어기는 추정 상태를 사용한다:

$$u = -K\hat{x} + N_r r$$

```
     r  →(+)→ [N_r] →(+)→ [Plant] → y
                       ↑              |
                  [-K] ← x̂ ← [Observer] ← y, u
```

### 4.2 분리 원리(Separation Principle)

**정리(분리 원리):** 결합 제어기-관측기 시스템의 폐루프 고유값은 제어기 고유값과 관측기 고유값의 **합집합(union)**이다:

$$\sigma(A_{cl}) = \sigma(A - BK) \cup \sigma(A - LC)$$

이것이 의미하는 바:
- 제어기는 관측기와 **독립적으로** 설계할 수 있다
- 관측기는 제어기와 **독립적으로** 설계할 수 있다
- 두 부분 시스템이 모두 안정하면 결합 시스템도 안정하다

### 4.3 증명 개요(Proof Sketch)

결합 상태는 $e = x - \hat{x}$일 때 $[x^T \; e^T]^T$이다:

$$\begin{bmatrix} \dot{x} \\ \dot{e} \end{bmatrix} = \begin{bmatrix} A-BK & BK \\ 0 & A-LC \end{bmatrix} \begin{bmatrix} x \\ e \end{bmatrix} + \begin{bmatrix} B N_r \\ 0 \end{bmatrix} r$$

블록 삼각 구조(block-triangular structure)에 의해 $2n \times 2n$ 행렬의 고유값은 정확히 $A-BK$와 $A-LC$의 고유값이 된다.

## 5. 축소 차수 관측기(Reduced-Order Observers)

일부 상태가 직접 측정된다면, 나머지 상태만 추정하면 된다.

$y = Cx$에서 $C$가 계수(rank) $p$를 가질 때, $n - p$개의 상태만 추정이 필요하다. **축소 차수 관측기(reduced-order observer)**는 $n$ 대신 $n - p$ 차원을 가져 계산 비용을 줄인다.

설계는 측정된 상태와 비측정 상태를 분리하는 변환된 좌표계를 사용한다는 점에서 유사한 원리를 따른다.

## 6. 설계 고려 사항(Design Considerations)

### 6.1 극 선택 지침(Pole Selection Guidelines)

**제어기 극(Controller poles):**
- 시간 영역 사양(정착 시간, 오버슈트)을 만족해야 한다
- 과도한 제어 노력을 요구하지 않아야 한다(매우 빠른 극 지양)
- 복소수 극은 켤레쌍으로 존재한다

**관측기 극(Observer poles):**
- 제어기 극보다 3-5배 빠른 것이 일반적인 규칙
- 너무 빠르면 → 잡음과 모델링 오차에 민감해진다
- 너무 느리면 → 과도 상태에서 추정이 부정확해진다

### 6.2 강건성(Robustness)

완전 상태 정보를 이용한 상태 피드백은 우수한 강건성(infinite gain margin for SISO)을 가진다. 그러나 관측기 기반 피드백은 **강건성이 낮을 수 있다** — 관측기를 사용하면 LQR의 보장된 여유(margins)가 사라진다.

이로 인해 **LQG/LTR(Loop Transfer Recovery)** 절차의 동기가 생긴다. 여기서 관측기 설계를 수정하여 상태 피드백 설계의 강건성을 회복한다.

## 연습 문제

### 연습 1: 극배치(Pole Placement)

$A = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}$, $B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$ (이중 적분기, double integrator):

1. 폐루프 극을 $s = -3 \pm j4$에 배치하도록 $K$를 설계하라
2. 폐루프 자연 주파수(natural frequency)와 감쇠비(damping ratio)를 계산하라
3. $C = [1 \; 0]$이면, 정상 상태 추적 오차가 영이 되도록 $N_r$을 계산하라

### 연습 2: 관측기 설계(Observer Design)

연습 1의 시스템에서 $C = \begin{bmatrix} 1 & 0 \end{bmatrix}$일 때:

1. 가관측성을 검증하라
2. 극을 $s = -15 \pm j20$에 배치하는 관측기를 설계하라
3. 완전한 관측기 상태 방정식을 작성하라

### 연습 3: 분리 원리 검증(Separation Principle Verification)

연습 1의 제어기와 연습 2의 관측기를 사용하여:

1. $4 \times 4$ 폐루프 시스템 행렬을 작성하라
2. 고유값이 제어기 극과 관측기 극의 합집합임을 검증하라
3. 계단 응답을 시뮬레이션하고 완전 상태 피드백(관측기 없음)과 비교하라

---

*이전: [레슨 12 — 가제어성과 가관측성](12_Controllability_and_Observability.md) | 다음: [레슨 14 — 최적 제어: LQR 및 칼만 필터](14_Optimal_Control.md)*
