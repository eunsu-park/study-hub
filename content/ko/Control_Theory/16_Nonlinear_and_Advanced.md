# 레슨 16: 비선형 제어와 고급 주제

## 학습 목표

- 비선형 제어가 필요한 이유와 선형 방법이 실패하는 경우를 이해한다
- 리아프노프 안정성 이론(Lyapunov Stability Theory)을 적용하여 비선형 시스템을 분석한다
- 기술 함수(Describing Function) 방법을 사용하여 극한 사이클(Limit Cycle)을 예측한다
- 슬라이딩 모드 제어(Sliding Mode Control) 원리를 이해한다
- 모델 예측 제어(Model Predictive Control, MPC)와 적응 제어(Adaptive Control) 개념을 개관한다

## 1. 왜 비선형 제어인가?

모든 실제 시스템은 비선형이다. 선형화(레슨 2)는 동작점 근방에서 유효하지만 다음 경우에는 실패한다:

- 시스템이 **넓은 범위**에서 동작할 때 (작은 섭동이 아닌 큰 각도에서의 진자)
- 비선형성이 **본질적**일 때 (포화(Saturation), 불감대(Dead Zone), 릴레이(Relay), 백래시(Backlash), 히스테리시스(Hysteresis))
- **전역적** 안정성 보장이 필요할 때 (평형점 근처의 국소적 보장이 아닌)
- 시스템에 **다중 평형점**이 있을 때 (상태가 어느 평형점으로 수렴하는가?)

### 1.1 일반적인 비선형성

| 비선형성 | 설명 | 영향 |
|-------------|-------------|--------|
| 포화(Saturation) | 출력이 $\pm u_{\max}$로 제한됨 | 제어 권한 제한, 와인드업(Windup) |
| 불감대(Dead Zone) | $|u| < \delta$이면 출력 없음 | 정상 상태 오차, 극한 사이클 |
| 릴레이(Relay, On-Off) | 이진 출력 ($\pm M$) | 채터링(Chattering), 극한 사이클 |
| 백래시(Backlash) | 기계적 결합의 히스테리시스 | 위상 지연, 진동 |
| 쿨롱 마찰(Coulomb Friction) | 운동에 반하는 일정 마찰 | 스틱-슬립(Stick-Slip), 극한 사이클 |
| 양자화(Quantization) | 이산 출력 레벨 | 설정점 주위의 극한 사이클 |

### 1.2 비선형 현상

선형 시스템에서는 **발생할 수 없지만** 비선형 시스템에서는 흔히 나타나는 거동:
- **극한 사이클(Limit Cycles)**: 초기 조건에 무관한 지속적인 진동
- **분기(Bifurcations)**: 매개변수 변화에 따른 거동의 질적 변화
- **카오스(Chaos)**: 초기 조건에 대한 민감한 의존성
- **다중 평형점(Multiple Equilibria)**: 초기 조건에 따라 달라지는 상이한 정상 상태
- **유한 이탈 시간(Finite Escape Time)**: 유한한 시간 내에 상태가 무한대로 발산

## 2. 리아프노프 안정성 이론(Lyapunov Stability Theory)

### 2.1 개념

알렉산드르 리아프노프(Aleksandr Lyapunov, 1892)는 미분 방정식을 **풀지 않고** 안정성을 증명하는 방법을 개발하였다. 핵심 아이디어: 항상 감소하는 "에너지 유사" 함수를 찾을 수 있다면, 시스템은 반드시 평형점에 수렴하고 있는 것이다.

### 2.2 리아프노프 직접법(Lyapunov's Direct Method)

평형점이 $x = 0$인 자율 시스템 $\dot{x} = f(x)$를 고려한다.

**정의:** 함수 $V(x): \mathbb{R}^n \to \mathbb{R}$이 **리아프노프 함수(Lyapunov Function)**가 되려면:
1. $V(0) = 0$
2. $V(x) > 0$ (모든 $x \neq 0$에 대해, 양정치(Positive Definite))
3. $\dot{V}(x) = \nabla V \cdot f(x) \leq 0$ (궤적을 따라)

**정리 (리아프노프):**

| $\dot{V}$의 조건 | 결론 |
|------------------------|------------|
| $\dot{V}(x) \leq 0$ (음반정치(Negative Semidefinite)) | 안정 (리아프노프 의미에서) |
| $\dot{V}(x) < 0$ ($x \neq 0$에 대해, 음정치(Negative Definite)) | 점근적 안정(Asymptotically Stable) |
| $\dot{V}(x) \leq -\alpha V(x)$ (어떤 $\alpha > 0$에 대해) | 지수 안정(Exponentially Stable) |

이 조건들이 원점 근방뿐만 아니라 모든 $x$에 대해 성립하면, 안정성은 **전역적(Global)**이다.

### 2.3 예시: 감쇠 진자

$$\ddot{\theta} + \frac{b}{ml^2}\dot{\theta} + \frac{g}{l}\sin\theta = 0$$

상태: $x_1 = \theta$, $x_2 = \dot{\theta}$. 평형점: $x = 0$.

**리아프노프 후보 함수** (전체 에너지):

$$V(x) = \frac{1}{2}ml^2 x_2^2 + mgl(1 - \cos x_1) > 0 \quad (x \neq 0)$$

$$\dot{V} = ml^2 x_2 \dot{x}_2 + mgl(\sin x_1)\dot{x}_1 = ml^2 x_2\left(-\frac{b}{ml^2}x_2 - \frac{g}{l}\sin x_1\right) + mgl x_2 \sin x_1$$

$$= -bx_2^2 \leq 0$$

$\dot{V} = 0$은 $x_2 = 0$일 때만 성립하며, 라살 불변 원리(LaSalle's Invariance Principle)에 의해 $x_2 = 0$인 유일한 불변 집합은 $x_1 = 0$이므로, 원점은 **전역적으로 점근 안정**하다.

### 2.4 리아프노프 함수 찾기

리아프노프 함수를 찾는 일반적인 알고리즘은 없지만, 일반적인 접근 방법은 다음과 같다:
- **에너지 방법(Energy Method)**: 물리적 에너지(운동 + 퍼텐셜) 사용
- **이차 형식(Quadratic Form)**: $V(x) = x^T P x$ (여기서 $P > 0$) — 선형 시스템에 유효 ($\dot{V} < 0$은 $A^T P + PA < 0$, 즉 리아프노프 방정식(Lyapunov Equation)과 동치)
- **가변 구배법(Variable Gradient Method)**: $\nabla V = M(x)x$로 가정하고 $M$을 풀기
- **SOS (제곱합, Sum of Squares)**: 다항식 시스템의 계산적 방법 — 반정치 프로그래밍(Semidefinite Programming)을 사용하여 $V$ 탐색

### 2.5 선형 시스템에 대한 리아프노프

$V = x^T P x$로 $\dot{x} = Ax$에 대해:

$$\dot{V} = x^T(A^T P + PA)x = -x^T Q x$$

리아프노프 방정식 $A^T P + PA = -Q$는 임의의 $Q > 0$에 대해 $A$가 안정적(모든 고유값이 좌반평면에 있을 때)인 경우에만 유일한 양정치 해 $P$를 가진다. 이는 대안적인 안정성 검증 방법이다.

## 3. 기술 함수 해석(Describing Function Analysis)

### 3.1 목적

**기술 함수(Describing Function)** 방법은 비선형 피드백 시스템에서 **극한 사이클(Limit Cycles)**(지속적인 진동)을 예측한다.

### 3.2 구성

```
     ┌────────┐      ┌────────┐
r=0 →(+)→│ N(a,ω) │→───→│ G(jω)  │→── y
     ↑   └────────┘      └────────┘  |
     └────────────────────────────────┘
```

여기서 $N(a, \omega)$는 비선형성의 **기술 함수** — 진폭 $a$의 정현파 입력에 대한 출력의 기본 고조파 비율이다.

### 3.3 기술 함수

무기억 비선형성 $y = n(x)$에 대한 정현파 입력 $x(t) = a\sin(\omega t)$에서, 기술 함수는:

$$N(a) = \frac{1}{\pi a}\int_0^{2\pi} n(a\sin\theta) e^{j\theta} d\theta$$

**대칭** 비선형성(홀함수)의 경우, $N(a)$는 **실수**이다.

### 3.4 일반적인 기술 함수

| 비선형성 | $N(a)$ |
|-------------|--------|
| 이상적 릴레이(Ideal Relay) ($\pm M$) | $\frac{4M}{\pi a}$ |
| 포화(Saturation) (기울기 $k$, 한계 $M$) | $\frac{2k}{\pi}\left[\sin^{-1}\frac{M}{ka} + \frac{M}{ka}\sqrt{1-\left(\frac{M}{ka}\right)^2}\right]$ ($a > M/k$인 경우) |
| 불감대(Dead Zone) (폭 $2\delta$) | $\frac{k}{\pi}\left[\pi - 2\sin^{-1}\frac{\delta}{a} - \frac{2\delta}{a}\sqrt{1-\left(\frac{\delta}{a}\right)^2}\right]$ ($a > \delta$인 경우) |

### 3.5 극한 사이클 예측

진폭 $a$, 주파수 $\omega$에서 극한 사이클이 존재하는 조건:

$$1 + N(a)G(j\omega) = 0 \quad \Rightarrow \quad G(j\omega) = -\frac{1}{N(a)}$$

**그래픽적 방법:** 같은 나이퀴스트 평면에 $G(j\omega)$와 $-1/N(a)$를 그린다. 교점이 극한 사이클을 예측한다.

**극한 사이클의 안정성:** $a$가 증가할 때 $-1/N(a)$가 $G(j\omega)$로 둘러싸인 영역 **안으로** 이동하면, 극한 사이클은 **안정**(자기 유지)하다.

## 4. 슬라이딩 모드 제어(Sliding Mode Control)

### 4.1 개념

**슬라이딩 모드 제어(SMC)**는 시스템 상태를 **슬라이딩 표면(Sliding Surface)** $\sigma(x) = 0$으로 강제하고, 고주파 스위칭을 사용하여 그 상태를 유지한다. 슬라이딩 표면 위에서는 시스템 동역학의 차수가 낮아지며, 원하는 거동을 위해 설계할 수 있다.

### 4.2 2차 시스템 설계

$\ddot{y} = f(x) + bu$에 대해 슬라이딩 표면을 정의한다:

$$\sigma = \dot{e} + \lambda e$$

여기서 $e = y - y_d$는 추종 오차이고, $\lambda > 0$은 표면 위의 수렴 속도를 결정한다.

**제어 법칙:**

$$u = \frac{1}{b}\left[-f(x) + \ddot{y}_d - \lambda\dot{e} - k\text{sign}(\sigma)\right]$$

여기서 $k > 0$은 스위칭 이득이다. $\text{sign}(\sigma)$ 항이 상태를 $\sigma = 0$으로 구동한다.

### 4.3 도달 조건(Reaching Condition)

상태가 슬라이딩 표면에 도달하도록 보장하려면:

$$\sigma\dot{\sigma} < 0$$

외란과 불확실성을 극복하기에 $k$가 충분히 크면 이 조건이 보장된다.

### 4.4 채터링(Chattering)

불연속적인 $\text{sign}(\sigma)$는 **채터링** — $\sigma = 0$ 주위의 고주파 진동 — 을 유발한다. 완화 방법:
- **경계층(Boundary Layer)**: $\text{sign}(\sigma)$를 $\text{sat}(\sigma/\phi)$로 대체 (여기서 $\phi$는 층 두께)
- **초고주파 비틀기 알고리즘(Super-Twisting Algorithm)**: 채터링을 줄이는 고차 슬라이딩 모드
- **도달 법칙 접근(Reaching Law Approach)**: $\dot{\sigma} = -k|\sigma|^\alpha \text{sign}(\sigma)$ 사용 ($0 < \alpha < 1$)

## 5. 모델 예측 제어(Model Predictive Control, MPC)

### 5.1 개념

**MPC**는 각 시간 단계에서 최적화 문제를 풀어 제어를 수행한다:
1. 모델을 사용하여 수평선(Horizon) $N$ 동안 미래 시스템 거동을 **예측**
2. 예측 궤적에 대한 비용 함수를 **최적화**
3. 첫 번째 제어 입력만 **적용**
4. 다음 시간 단계에서 **반복** (이동 수평선(Receding Horizon))

### 5.2 공식화

각 시간 단계 $k$에서 다음을 푼다:

$$\min_{u[k], \ldots, u[k+N-1]} \sum_{i=0}^{N-1} \left[ x[k+i]^T Q x[k+i] + u[k+i]^T R u[k+i] \right] + x[k+N]^T P_f x[k+N]$$

제약 조건:
- $x[k+i+1] = Ax[k+i] + Bu[k+i]$ (모델)
- $u_{\min} \leq u[k+i] \leq u_{\max}$ (입력 제약)
- $x_{\min} \leq x[k+i] \leq x_{\max}$ (상태 제약)

### 5.3 장점

- **제약을 명시적으로 처리** (구동기 한계, 안전 범위)
- **체계적인 다입출력(MIMO) 설계**
- **예측 기능** (알려진 미래 기준값 반영 가능)
- **최적성** (예측 수평선 내에서)

### 5.4 과제

- 계산 비용 (실시간으로 최적화 문제를 풀어야 함)
- 정확한 모델 필요
- 안정성 보장을 위한 신중한 설계 필요 (종단 비용 $P_f$, 종단 제약)
- 수평선 $N$, 가중치 $Q$, $R$ 및 제약 조건 튜닝

### 5.5 MPC 대 LQR

| 특성 | LQR | MPC |
|---------|-----|-----|
| 수평선 | 무한 | 유한 (이동) |
| 제약 | 처리 불가 | 명시적으로 처리 |
| 계산 | 오프라인 (ARE 한 번 풀기) | 온라인 (매 단계 QP 풀기) |
| 최적성 | 전역 최적 (비제약) | 국소 최적 (제약) |
| 다입출력(MIMO) | 가능 | 가능 |

제약이 없을 때 $N \to \infty$이면 MPC는 LQR로 수렴한다.

## 6. 적응 제어(Adaptive Control)

### 6.1 동기

플랜트 매개변수가 **미지** 또는 **시변**인 경우, 고정 제어기는 성능이 저하될 수 있다. 적응 제어는 실시간으로 제어기 매개변수를 조정한다.

### 6.2 기준 모델 적응 제어(Model Reference Adaptive Control, MRAC)

```
Reference ──→ [Reference Model] ──→ y_m (desired output)
    |                                      |
    └──→ [Adaptive Controller] ──→ [Plant] ──→ y (actual output)
              ↑                              |
              └── [Adaptation Law] ←── e = y - y_m
```

적응 법칙은 리아프노프 기반 설계 또는 MIT 규칙을 사용하여 $y(t) \to y_m(t)$가 되도록 제어기 매개변수를 조정한다.

### 6.3 자기 동조 조절기(Self-Tuning Regulators)

1. 온라인으로 플랜트 매개변수를 **추정** (재귀 최소 제곱법(Recursive Least Squares) 등 사용)
2. 추정된 플랜트에 대한 제어기를 **설계** (극 배치(Pole Placement), LQR 등 사용)
3. **적용**하고 반복

이 방법은 시스템 식별과 제어기 설계를 폐루프 형태로 결합한다.

## 7. 요약: 제어 이론 전경

```
                    Control Theory
                         │
         ┌───────────────┼───────────────┐
    Classical           Modern          Advanced
         │                │                │
    ├── Transfer fn   ├── State-space  ├── Nonlinear
    ├── Root locus    ├── Pole place.  ├── Lyapunov
    ├── Bode/Nyquist  ├── LQR/LQG     ├── Sliding mode
    ├── PID           ├── Kalman       ├── MPC
    └── Lead/lag      └── H∞ robust    ├── Adaptive
                                       └── Learning-based
```

**핵심 교훈:** 작동하는 가장 단순한 방법부터 시작한다. PID는 대부분의 산업 제어 문제를 처리한다. 제약 조건, 다입출력 결합, 비선형성, 또는 엄격한 성능 요구 사항이 요구될 때 고급 방법을 사용한다.

## 연습 문제

### 연습 1: 리아프노프 해석

반 데르 폴 발진기(Van der Pol Oscillator)에 대해:

$$\ddot{x} - \mu(1 - x^2)\dot{x} + x = 0$$

1. 평형점을 구하라
2. 평형점 주위에서 선형화하고, $\mu > 0$일 때의 국소 안정성을 분석하라
3. 리아프노프 함수 $V = \frac{1}{2}(x^2 + \dot{x}^2)$를 시도한다. $\dot{V}$를 계산하라. 안정성을 결론지을 수 있는가?
4. $\mu > 0$일 때 반 데르 폴 발진기는 어떤 현상을 나타내는가? (힌트: 극한 사이클)

### 연습 2: 기술 함수

피드백 시스템에 릴레이 비선형성 ($\pm 1$)이 $G(s) = \frac{10}{s(s+1)(s+2)}$와 직렬로 연결되어 있다.

1. 릴레이에 대한 기술 함수를 작성하라
2. $-1/N(a)$를 구하고 복소 평면에 스케치하라
3. $G(j\omega)$와의 교점을 찾아 극한 사이클 진폭과 주파수를 예측하라

### 연습 3: MPC 개념

이산 시간 이중 적분기(Discrete-Time Double Integrator) $x[k+1] = \begin{bmatrix} 1 & T \\ 0 & 1 \end{bmatrix} x[k] + \begin{bmatrix} T^2/2 \\ T \end{bmatrix} u[k]$ ($T = 0.1$ s, 제약 $|u| \leq 1$):

1. $Q = I$, $R = 0.1$, $N = 10$으로 MPC 최적화 문제를 공식화하라
2. 제약이 없으면 이는 유한 수평선 LQR이다 — $|u| \leq 1$이면 무엇이 바뀌는가?
3. 이동 수평선(첫 $u[k]$만 적용하고 재풀기)이 외란 처리에 도움이 되는 이유를 설명하라

---

*이전: [레슨 15 — 디지털 제어 시스템](15_Digital_Control.md)*

*돌아가기: [개요](00_Overview.md)*
