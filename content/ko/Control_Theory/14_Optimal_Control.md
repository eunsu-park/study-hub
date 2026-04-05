# 레슨 14: 최적 제어 — LQR과 칼만 필터(Kalman Filter)

## 학습 목표

- 선형 이차 조절기(Linear-Quadratic Regulator, LQR) 문제를 수식으로 정형화한다
- 대수 리카티 방정식(Algebraic Riccati Equation)을 풀어 최적 이득을 구한다
- 잡음 환경에서 상태 추정을 위한 칼만 필터(Kalman Filter)를 설계한다
- LQR과 칼만 필터를 결합하여 LQG 제어기를 구성한다
- LQR의 강인성(robustness) 특성과 LQG의 한계를 이해한다

## 1. 최적 제어의 필요성

극점 배치(pole placement)는 폐루프 극점 위치를 자유롭게 선택할 수 있지만, **어떤** 극점 위치가 최선인지는 알려주지 않는다. 최적 제어(optimal control)는 체계적인 틀을 제공한다:

- 추적 오차와 제어 노력을 벌하는 **비용 함수(cost function)**를 정의한다
- 비용을 **최소화**하는 제어 법칙을 찾는다
- 결과는 성능과 노력 사이의 균형을 원칙적으로 결정한다

## 2. 선형 이차 조절기(Linear-Quadratic Regulator, LQR)

### 2.1 문제 정형화

**플랜트:** $\dot{x} = Ax + Bu$

**비용 함수:**

$$J = \int_0^\infty \left[ x^T(t) Q x(t) + u^T(t) R u(t) \right] dt$$

여기서:
- $Q \geq 0$ (양반정치): **상태 가중 행렬(state weighting matrix)** — 상태 편차를 벌한다
- $R > 0$ (양정치): **제어 가중 행렬(control weighting matrix)** — 제어 노력을 벌한다

**목표:** $J$를 최소화하는 $u(t)$를 구한다.

### 2.2 최적해

**정리:** $(A, B)$가 가제어(controllable, 또는 안정화 가능)하고 $(A, Q^{1/2})$가 가관측(observable, 또는 검출 가능)하면, 최적 제어 법칙은:

$$u^*(t) = -Kx(t), \quad K = R^{-1}B^T P$$

여기서 $P$는 **대수 리카티 방정식(algebraic Riccati equation, ARE)**의 유일한 양정치 해:

$$A^T P + PA - PBR^{-1}B^T P + Q = 0$$

### 2.3 LQR의 특성

**보장된 안정성:** 폐루프 시스템 $\dot{x} = (A - BK)x$는 항상 점근적으로 안정하다.

**보장된 강인성(SISO):**
- **이득 여유(Gain margin):** $[1/2, \infty)$ — 이득을 절반으로 줄이거나 무한대로 늘려도 시스템이 안정을 유지한다
- **위상 여유(Phase margin):** $\geq 60°$

이는 주목할 만한 보장이다 — 다른 어떤 선형 설계 방법도 이러한 강한 여유를 자동으로 제공하지 않는다.

### 2.4 $Q$와 $R$ 조정

**물리적 해석:**
- 큰 $Q_{ii}$: $x_i$를 빠르게 영으로 수렴시킴 (빠른 응답)
- 큰 $R_{jj}$: 채널 $j$의 큰 제어 입력을 벌함 (작은 제어 노력)
- $Q/R$ 비율: 성능과 제어 노력 사이의 트레이드오프

**일반적인 선택:**
- $Q = \text{diag}(q_1, \ldots, q_n)$, $R = \rho I$ — 단일 매개변수 $\rho$로 트레이드오프 조정
- **브라이슨 법칙(Bryson's rule):** $Q_{ii} = 1/x_{i,\max}^2$, $R_{jj} = 1/u_{j,\max}^2$ — 최대 허용값으로 정규화

### 2.5 예제

이중 적분기: $A = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}$, $B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$

$Q = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$, $R = [1]$로 설정 시:

ARE를 풀면 $P = \begin{bmatrix} \sqrt{3} & 1 \\ 1 & \sqrt{3} \end{bmatrix}$

$$K = R^{-1}B^T P = \begin{bmatrix} 1 & \sqrt{3} \end{bmatrix}$$

폐루프 극점: $s = -\frac{\sqrt{3}}{2} \pm j\frac{1}{2}$ → $\omega_n = 1$, $\zeta = \frac{\sqrt{3}}{2} \approx 0.87$.

$Q_{11}$ 증가 → 빠른 응답, 더 많은 제어 노력. $R$ 증가 → 느리고 부드러운 응답.

## 3. 칼만 필터(Kalman Filter)

### 3.1 문제 정형화

**잡음이 있는 플랜트:**

$$\dot{x} = Ax + Bu + Gw$$
$$y = Cx + v$$

여기서:
- $w(t)$: **프로세스 잡음(process noise)** (외란, 모델 불확실성) — $E[ww^T] = W$
- $v(t)$: **측정 잡음(measurement noise)** (센서 잡음) — $E[vv^T] = V$
- 둘 다 백색(white), 영평균(zero-mean), 가우시안(Gaussian)

**목표:** 잡음이 섞인 측정값 $y(t)$로부터 $x(t)$의 최적 추정값 $\hat{x}(t)$를 구한다.

### 3.2 칼만 필터(연속 시간)

$$\dot{\hat{x}} = A\hat{x} + Bu + L(y - C\hat{x})$$

최적 이득은:

$$L = P_f C^T V^{-1}$$

여기서 $P_f$는 **필터 대수 리카티 방정식(filter algebraic Riccati equation)**의 해:

$$AP_f + P_f A^T - P_f C^T V^{-1} C P_f + GWG^T = 0$$

### 3.3 특성

- 칼만 필터는 **최적** 선형 추정기이다 ($E[\|x - \hat{x}\|^2]$ 최소화)
- 레슨 13의 루엔베르거 관측기(Luenberger observer)와 같은 구조이지만 이득이 최적으로 선택된다
- 필터는 모델에 대한 신뢰와 측정값에 대한 신뢰를 **균형** 있게 조정한다:
  - 큰 $W$ (잡음이 많은 모델) → 큰 $L$ (측정값을 더 신뢰)
  - 큰 $V$ (잡음이 많은 센서) → 작은 $L$ (모델을 더 신뢰)

### 3.4 LQR과의 쌍대성(Duality)

칼만 필터와 LQR은 **쌍대(dual)** 문제이다:

| LQR | 칼만 필터(Kalman Filter) |
|-----|---------------|
| $A^TP + PA - PBR^{-1}B^TP + Q = 0$ | $AP_f + P_fA^T - P_fC^TV^{-1}CP_f + GWG^T = 0$ |
| $K = R^{-1}B^TP$ | $L = P_fC^TV^{-1}$ |
| 상태 가중 $Q$ | 프로세스 잡음 $GWG^T$ |
| 제어 가중 $R$ | 측정 잡음 $V$ |
| 피드백 이득(Feedback gain) $K$ | 관측기 이득(Observer gain) $L$ |

## 4. LQG 제어

### 4.1 LQR + 칼만 필터 결합

**선형 이차 가우시안(Linear-Quadratic-Gaussian, LQG)** 제어기는 다음을 결합한다:
- 최적 상태 피드백을 위한 LQR
- 최적 상태 추정을 위한 칼만 필터

$$u = -K\hat{x}, \quad K = R^{-1}B^T P \quad \text{(LQR)}$$
$$\dot{\hat{x}} = A\hat{x} + Bu + L(y - C\hat{x}), \quad L = P_f C^T V^{-1} \quad \text{(Kalman filter)}$$

### 4.2 분리 원리(Separation Principle) (확률적)

**확실성 등가 원리(certainty equivalence principle)**는 LQR과 칼만 필터를 독립적으로 설계할 수 있음을 보장한다 — 레슨 13의 분리 원리가 확률적 환경에서도 동일하게 적용된다.

### 4.3 LQG 전달 함수

LQG 제어기는 동적 보상기(dynamic compensator)이다:

$$G_{LQG}(s) = K(sI - A + BK + LC)^{-1}L$$

이는 차수 $n$의 진유리(proper) 전달 함수이다 (플랜트와 동일한 차수).

## 5. LQG의 강인성 한계

### 5.1 문제점

LQR은 보장된 여유값($GM = [1/2, \infty)$, $PM \geq 60°$)을 가지지만, **LQG는 보장된 여유값이 없다**. 칼만 필터는 LQR의 강인성을 임의로 저하시킬 수 있다.

이는 1970년대의 중요한 발견이었다 (Doyle, 1978) — 최적 제어가 자동으로 강인한 제어를 보장하지 않음을 보여주었다.

### 5.2 루프 전달 복원(Loop Transfer Recovery, LTR)

**LQG/LTR**은 루프 전달 함수가 LQR 루프 전달 함수에 근사하도록 칼만 필터를 설계하여 LQR 강인성을 복원하려는 방법이다:

$$L(j\omega) \approx K(j\omega I - A)^{-1}B \quad \text{(플랜트 입력에서)}$$

이는 프로세스 잡음 공분산을 증가시킴으로써 달성된다: $W \to qBB^T$ ($q \to \infty$).

## 6. 유한 구간 LQR(Finite-Horizon LQR)

### 6.1 시변 리카티 방정식

유한 시간 구간 $[0, t_f]$에서:

$$J = x^T(t_f) S_f x(t_f) + \int_0^{t_f} \left[ x^T Q x + u^T R u \right] dt$$

최적 이득은 $K(t) = R^{-1}B^T P(t)$이며, $P(t)$는 **미분 리카티 방정식(differential Riccati equation)**을 만족한다:

$$-\dot{P} = A^T P + PA - PBR^{-1}B^T P + Q, \quad P(t_f) = S_f$$

이는 $t = t_f$에서 $t = 0$으로 **역방향**으로 적분된다. $t_f \to \infty$이면 $P(t)$는 정상 상태 ARE 해로 수렴한다.

## 연습 문제

### 연습 1: LQR 설계

시스템 $A = \begin{bmatrix} 0 & 1 \\ -1 & -1 \end{bmatrix}$, $B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$에 대해:

1. $Q = I$, $R = 1$로 ARE를 풀어라 ($P$의 3개 유일 원소에 대한 3개의 비선형 방정식 계를 세우고 풀어라)
2. 최적 이득 $K$를 계산하라
3. 폐루프 극점을 구하고 안정성을 검증하라
4. $R$을 $10$으로 늘리면 극점은 어떻게 변하는가? $0.1$로 줄이면?

### 연습 2: 칼만 필터

같은 시스템에서 $C = \begin{bmatrix} 1 & 0 \end{bmatrix}$, 프로세스 잡음 강도 $W = 0.1$, 측정 잡음 $V = 1$로 설정할 때:

1. 필터 ARE를 세워라
2. 칼만 이득(Kalman gain) $L$을 계산하라
3. 관측기 극점은 어디에 위치하는가?

### 연습 3: LQR 특성

SISO 경우에 LQR 복귀차(return difference)가 다음을 만족함을 보여라:

$$|1 + K(j\omega I - A)^{-1}B| \geq 1 \quad \forall \omega$$

힌트: $1 + K(j\omega I - A)^{-1}B = 1 + R^{-1}B^T P(j\omega I - A)^{-1}B$에서 출발하여 ARE를 활용하라.

이것이 이득 여유와 위상 여유에 대해 무엇을 의미하는가?

---

*이전: [레슨 13 — 상태 피드백과 관측기 설계](13_State_Feedback_and_Observers.md) | 다음: [레슨 15 — 디지털 제어 시스템](15_Digital_Control.md)*
