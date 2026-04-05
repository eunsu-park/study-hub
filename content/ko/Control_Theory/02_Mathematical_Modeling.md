# 레슨 2: 물리 시스템의 수학적 모델링

## 학습 목표

- 기계, 전기, 전기기계 시스템에 대한 미분방정식 모델 유도
- 물리 도메인 간의 유추(analogy) 파악
- 평형점(equilibrium point) 주변에서 비선형 모델 선형화
- 물리 모델을 제어 해석에 적합한 표준 형식으로 변환

## 1. 수학적 모델링이 필요한 이유

제어 설계에는 **수학적 모델(mathematical model)** — 플랜트(plant)의 동적 거동을 기술하는 방정식 집합 — 이 필요하다. 모델은 핵심 물리를 포착하면서도 해석과 제어기 합성이 가능할 만큼 단순해야 한다.

**모델링 접근법:**
- **제1원리(First principles)**: 물리 법칙(뉴턴 법칙, 키르히호프 법칙, 보존 법칙)에서 유도
- **시스템 식별(System identification)**: 측정된 입출력 데이터에 모델을 맞춤 (심화 과정에서 다룸)
- **혼합(Hybrid)**: 물리 기반 구조와 데이터 적합 파라미터를 결합

이 레슨에서는 제1원리 모델링에 집중한다.

## 2. 기계 시스템

### 2.1 병진 시스템(Translational Systems)

세 가지 기본 요소:

| 요소 | 법칙 | 방정식 |
|------|------|--------|
| 질량(Mass) $m$ | 뉴턴 제2법칙 | $F = m\ddot{x}$ |
| 댐퍼(Damper) $b$ | 점성 마찰(Viscous friction) | $F = b\dot{x}$ |
| 스프링(Spring) $k$ | 훅의 법칙(Hooke's law) | $F = kx$ |

**예제: 질량-스프링-댐퍼 시스템**

```
    F(t)
    →  ┌───┐
   ────┤ m ├────┬──── wall
       └───┘    │
          ├──┤b├──┤  (damper)
          ├──/\/\──┤  (spring, k)
```

뉴턴 제2법칙을 적용하면:

$$m\ddot{x}(t) + b\dot{x}(t) + kx(t) = F(t)$$

이것은 상수 계수를 가진 **2차 선형 상미분방정식(second-order linear ODE)**이다.

**표준 형식** ($m$으로 나눔):

$$\ddot{x} + 2\zeta\omega_n \dot{x} + \omega_n^2 x = \frac{F(t)}{m}$$

여기서 $\omega_n = \sqrt{k/m}$은 **고유 진동수(natural frequency)**, $\zeta = b/(2\sqrt{mk})$는 **감쇠비(damping ratio)**이다.

### 2.2 회전 시스템(Rotational Systems)

| 요소 | 법칙 | 방정식 |
|------|------|--------|
| 관성 모멘트(Moment of inertia) $J$ | 뉴턴 회전 법칙 | $\tau = J\ddot{\theta}$ |
| 회전 댐퍼(Rotational damper) $B$ | 점성 마찰 | $\tau = B\dot{\theta}$ |
| 비틀림 스프링(Torsional spring) $K$ | 훅의 법칙(회전) | $\tau = K\theta$ |

**예제: 단순 진자(소각도 근사)**

길이 $l$, 질량 $m$인 진자:

$$ml^2 \ddot{\theta} + mgl\sin\theta = \tau(t)$$

소각도 선형화($\sin\theta \approx \theta$):

$$ml^2 \ddot{\theta} + mgl\theta = \tau(t)$$

### 2.3 기어 트레인(Gear Trains)

기어비 $N = N_2/N_1$ (이 수 비율)을 가진 기어 트레인은 토크와 속도를 변환한다:

$$\theta_2 = \frac{N_1}{N_2}\theta_1, \quad \tau_2 = \frac{N_2}{N_1}\tau_1$$

입력축에서 본 등가 관성(reflected inertia):

$$J_{\text{eff}} = J_1 + \left(\frac{N_1}{N_2}\right)^2 J_2$$

## 3. 전기 시스템

### 3.1 수동 소자(Passive Elements)

| 소자 | 전압-전류 관계 | 임피던스(Impedance) $Z(s)$ |
|------|--------------|--------------------------|
| 저항(Resistor) $R$ | $v = Ri$ | $R$ |
| 인덕터(Inductor) $L$ | $v = L\frac{di}{dt}$ | $Ls$ |
| 커패시터(Capacitor) $C$ | $v = \frac{1}{C}\int i \, dt$ | $\frac{1}{Cs}$ |

### 3.2 키르히호프 법칙(Kirchhoff's Laws)

- **KVL** (키르히호프 전압 법칙): 루프 내 전압의 합 = 0
- **KCL** (키르히호프 전류 법칙): 노드에서 전류의 합 = 0

**예제: 직렬 RLC 회로**

KVL을 적용하면:

$$L\frac{di}{dt} + Ri + \frac{1}{C}\int i \, dt = v_{\text{in}}(t)$$

미분하고 $v_C = \frac{1}{C}\int i \, dt$를 대입하면:

$$LC\ddot{v}_C + RC\dot{v}_C + v_C = v_{\text{in}}(t)$$

이 식은 질량-스프링-댐퍼 시스템과 **동일한 형태**를 갖는다!

### 3.3 연산 증폭기(Op-Amp) 회로

이상적인 연산 증폭기 가정: 무한 입력 임피던스, 0 출력 임피던스, 무한 이득.

**반전 증폭기(Inverting amplifier):**

$$v_{\text{out}} = -\frac{R_f}{R_{\text{in}}} v_{\text{in}}$$

**적분기(Integrator):**

$$v_{\text{out}} = -\frac{1}{R_{\text{in}}C_f} \int v_{\text{in}} \, dt$$

**미분기(Differentiator):**

$$v_{\text{out}} = -R_f C_{\text{in}} \frac{dv_{\text{in}}}{dt}$$

연산 증폭기 회로는 아날로그 제어기(PID)를 구현하는 데 흔히 사용된다.

## 4. 전기기계 시스템(Electromechanical Systems)

### 4.1 직류 전동기(DC Motor)

직류 전동기는 전기 에너지를 기계 에너지로 변환한다. 핵심 방정식은 다음과 같다:

**전기부(전기자 회로(armature circuit)):**

$$L_a \frac{di_a}{dt} + R_a i_a + K_b \dot{\theta} = v_a(t)$$

**기계부(회전자(rotor)):**

$$J\ddot{\theta} + B\dot{\theta} = K_t i_a$$

여기서:
- $v_a$: 인가 전압
- $i_a$: 전기자 전류
- $K_b$: 역기전력(back-EMF) 상수
- $K_t$: 토크 상수 (SI 단위에서 $K_t = K_b$)
- $R_a, L_a$: 전기자 저항 및 인덕턴스

**전달 함수(Transfer function)** ($V_a(s)$에서 $\Theta(s)$로, $L_a \approx 0$ 가정):

$$\frac{\Theta(s)}{V_a(s)} = \frac{K_t}{s(JR_a s + BR_a + K_t K_b)}$$

직류 전동기는 제어 공학에서 가장 중요한 플랜트 중 하나로, 로봇공학, 디스크 드라이브, 프린터 등 수많은 응용에 사용된다.

## 5. 물리 도메인 간의 유추

물리 시스템의 수학적 구조는 도메인을 가로질러 종종 동일하다:

| 기계(병진) | 기계(회전) | 전기 | 유체 | 열 |
|-----------|-----------|------|------|-----|
| 힘(Force) $F$ | 토크(Torque) $\tau$ | 전압(Voltage) $v$ | 압력(Pressure) $P$ | 온도(Temp.) $T$ |
| 속도(Velocity) $\dot{x}$ | 각속도(Angular vel.) $\dot{\theta}$ | 전류(Current) $i$ | 유량(Flow rate) $Q$ | 열 흐름(Heat flow) $q$ |
| 질량(Mass) $m$ | 관성(Inertia) $J$ | 인덕턴스(Inductance) $L$ | 관성(Inertance) | — |
| 댐퍼(Damper) $b$ | 회전 댐퍼(Rot. damper) $B$ | 저항(Resistance) $R$ | 유체 저항 | 열저항(Thermal resist.) |
| 스프링(Spring) $k$ | 비틀림 스프링(Torsion spring) $K$ | 일래스턴스(Elastance) $1/C$ | 유체 커패시턴스 | 열용량(Thermal cap.) |
| 변위(Displacement) $x$ | 각도(Angle) $\theta$ | 전하(Charge) $q$ | 체적(Volume) | 열(Heat) $Q$ |

이러한 유추를 통해 한 도메인에서 개발된 기법을 다른 도메인에 직접 적용할 수 있다.

## 6. 선형화(Linearization)

대부분의 실제 시스템은 **비선형(nonlinear)**이다. 선형 제어 이론은 **동작점(operating point)** (평형점)에서의 **선형화(linearization)** 이후에만 적용 가능하다.

### 6.1 평형점(Equilibrium Point)

평형점 $(\bar{x}, \bar{u})$은 $\dot{x} = f(\bar{x}, \bar{u}) = 0$을 만족한다.

### 6.2 테일러 급수 선형화(Taylor Series Linearization)

비선형 시스템 $\dot{x} = f(x, u)$가 주어졌을 때, 섭동 변수(perturbation variable)를 정의한다:

$$\delta x = x - \bar{x}, \quad \delta u = u - \bar{u}$$

$f$를 테일러 급수로 전개하고 1차 항만 유지하면:

$$\delta\dot{x} \approx \frac{\partial f}{\partial x}\bigg|_{(\bar{x},\bar{u})} \delta x + \frac{\partial f}{\partial u}\bigg|_{(\bar{x},\bar{u})} \delta u$$

이를 통해 평형점 근방에서 유효한 **선형 근사(linear approximation)**를 얻는다.

### 6.3 예제: 비선형 진자

완전한 비선형 방정식:

$$\ddot{\theta} + \frac{g}{l}\sin\theta = \frac{\tau}{ml^2}$$

**평형점:** $\bar{\theta} = 0$, $\bar{\tau} = 0$

**선형화:** 소각도에서 $\sin\theta \approx \theta$:

$$\ddot{\theta} + \frac{g}{l}\theta = \frac{\tau}{ml^2}$$

**평형점:** $\bar{\theta} = \pi$ (역진자(inverted pendulum)), $\bar{\tau} = 0$

$\delta\theta = \theta - \pi$로 놓으면: $\sin(\pi + \delta\theta) = -\sin(\delta\theta) \approx -\delta\theta$이므로:

$$\delta\ddot{\theta} - \frac{g}{l}\delta\theta = \frac{\tau}{ml^2}$$

부호 변화에 주목하라 — 역진자 평형점은 $\delta\theta$ 계수가 양수이므로 **불안정(unstable)**하고, 매달린 평형점은 계수가 음수이므로 **안정(stable)**하다.

## 7. 물리 모델에서 전달 함수로

전달 함수를 구하는 표준 절차:

1. **식별**: 물리 요소와 상호 연결 파악
2. **적용**: 물리 법칙 적용 (뉴턴, 키르히호프 등)
3. **유도**: 미분방정식 도출
4. **선형화**: 필요한 경우 선형화
5. **라플라스 변환 적용**: (초기 조건은 0으로 가정)
6. **풀기**: $G(s) = Y(s)/U(s)$ 계산

**예제:** 질량-스프링-댐퍼 $m\ddot{x} + b\dot{x} + kx = F$에서:

$$ms^2 X(s) + bsX(s) + kX(s) = F(s)$$

$$G(s) = \frac{X(s)}{F(s)} = \frac{1}{ms^2 + bs + k}$$

## 연습 문제

### 연습 1: 기계 시스템 모델링

두 질량 시스템에서 $m_1$과 $m_2$가 스프링 $k_{12}$와 댐퍼 $b_{12}$로 연결되어 있다. 질량 $m_1$은 스프링 $k_1$과 댐퍼 $b_1$으로 벽에 연결되어 있다. 힘 $F(t)$는 $m_2$에 가해진다.

1. 각 질량에 대한 자유물체도(free-body diagram)를 그려라
2. 연립 미분방정식을 작성하라
3. 전달 함수 $X_2(s)/F(s)$를 구하라

### 연습 2: 전기-기계 유추

입력 전압 $v_{\text{in}}$과 출력 $v_C$를 가진 직렬 RLC 회로에서:

1. 미분방정식을 작성하라
2. 기계적 유추를 찾아라 (각 전기 소자에 대응하는 기계 소자는?)
3. 전달 함수 $V_C(s)/V_{\text{in}}(s)$를 구하고 $\omega_n$과 $\zeta$를 식별하라

### 연습 3: 선형화

탱크 시스템의 비선형 동역학:

$$A\dot{h} = q_{\text{in}} - c\sqrt{h}$$

여기서 $h$는 수위, $A$는 단면적, $q_{\text{in}}$은 입력 유량, $c$는 밸브 계수이다.

1. 일정한 입력 $\bar{q}_{\text{in}}$의 함수로 평형 수위 $\bar{h}$를 구하라
2. 이 평형점 주변에서 선형화하라
3. $\delta q_{\text{in}}$에서 $\delta h$로의 전달 함수를 구하라

---

*이전: [레슨 1 — 제어 시스템 개요](01_Introduction_to_Control_Systems.md) | 다음: [레슨 3 — 전달 함수와 블록 선도](03_Transfer_Functions_and_Block_Diagrams.md)*
