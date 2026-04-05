# 레슨 11: 상태 공간 표현 (State-Space Representation)

## 학습 목표

- 동적 시스템을 상태 공간(State Space) 형식으로 표현한다
- 전달 함수와 상태 공간 모델 간의 변환을 수행한다
- 가제어 표준형(Controllable Canonical Form), 가관측 표준형(Observable Canonical Form), 대각형(Modal Form)을 구별한다
- 상태 공간 표현과 전달 함수 표현의 관계를 이해한다
- 상태 전이 행렬(State Transition Matrix)을 계산하고 상태 방정식을 푼다

## 1. 전달 함수에서 상태 공간으로 (From Transfer Functions to State Space)

전달 함수는 입출력 거동만을 포착한다. **상태 공간 표현(State-Space Representation)**은 내부 동역학 전체를 포착하여 다음을 가능하게 한다:
- MIMO(다입력 다출력, Multi-Input Multi-Output) 시스템 해석
- 내부 안정성 해석 (BIBO뿐만 아니라)
- 체계적인 제어기 및 관측기 설계
- 비선형 시스템 처리 (선형화된 상태 모델을 통해)

## 2. 상태 공간 방정식 (State-Space Equations)

연속 시간 LTI 시스템의 상태 공간 형식:

$$\dot{x}(t) = Ax(t) + Bu(t) \quad \text{(상태 방정식, state equation)}$$
$$y(t) = Cx(t) + Du(t) \quad \text{(출력 방정식, output equation)}$$

여기서:
- $x(t) \in \mathbb{R}^n$: **상태 벡터(state vector)** ($n$ = 시스템 차수)
- $u(t) \in \mathbb{R}^m$: **입력 벡터(input vector)**
- $y(t) \in \mathbb{R}^p$: **출력 벡터(output vector)**
- $A \in \mathbb{R}^{n \times n}$: **시스템 행렬(system matrix)** (또는 상태 행렬)
- $B \in \mathbb{R}^{n \times m}$: **입력 행렬(input matrix)**
- $C \in \mathbb{R}^{p \times n}$: **출력 행렬(output matrix)**
- $D \in \mathbb{R}^{p \times m}$: **순방향 행렬(feedforward matrix)** (흔히 영행렬)

### 2.1 블록 다이어그램 (Block Diagram)

```
u(t) → [B] →(+)→ [∫] → x(t) → [C] →(+)→ y(t)
              ↑                        ↑
              └── [A] ←───────────┘    [D] ← u(t)
```

## 3. 상태 공간 모델 유도 (Deriving State-Space Models)

### 3.1 미분 방정식으로부터 (From Differential Equations)

**예제:** 질량-스프링-댐퍼(Mass-Spring-Damper): $m\ddot{y} + b\dot{y} + ky = F$

상태 변수 선택: $x_1 = y$, $x_2 = \dot{y}$

$$\dot{x}_1 = x_2$$
$$\dot{x}_2 = -\frac{k}{m}x_1 - \frac{b}{m}x_2 + \frac{1}{m}F$$

행렬 형식으로:

$$\begin{bmatrix} \dot{x}_1 \\ \dot{x}_2 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ -k/m & -b/m \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} 0 \\ 1/m \end{bmatrix} F$$

$$y = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

### 3.2 전달 함수로부터 (From Transfer Functions)

$G(s) = \frac{b_1 s + b_0}{s^2 + a_1 s + a_0}$가 주어지면, **가제어 표준형(Controllable Canonical Form)**은:

$$A = \begin{bmatrix} 0 & 1 \\ -a_0 & -a_1 \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad C = \begin{bmatrix} b_0 & b_1 \end{bmatrix}, \quad D = 0$$

$n$차 시스템 $G(s) = \frac{b_{n-1}s^{n-1} + \cdots + b_0}{s^n + a_{n-1}s^{n-1} + \cdots + a_0}$에 대해:

$$A = \begin{bmatrix} 0 & 1 & 0 & \cdots & 0 \\ 0 & 0 & 1 & \cdots & 0 \\ \vdots & & & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & 1 \\ -a_0 & -a_1 & -a_2 & \cdots & -a_{n-1} \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 0 \\ 1 \end{bmatrix}$$

### 3.3 상태 공간에서 전달 함수로 (From State Space to Transfer Function)

라플라스 변환(초기 조건 영)을 적용하면:

$$sX(s) = AX(s) + BU(s) \Rightarrow X(s) = (sI - A)^{-1}BU(s)$$

$$Y(s) = [C(sI - A)^{-1}B + D]U(s)$$

따라서:

$$G(s) = C(sI - A)^{-1}B + D$$

## 4. 표준형 (Canonical Forms)

### 4.1 가제어 표준형 (Controllable Canonical Form, CCF)

위에서 설명한 형식이다. $A$의 마지막 행에 특성 다항식(Characteristic Polynomial) 계수의 부호를 반전한 값이 놓인다.

**특성:** 정의에 의해 항상 가제어(Controllable).

### 4.2 가관측 표준형 (Observable Canonical Form, OCF)

$$A = \begin{bmatrix} 0 & 0 & \cdots & 0 & -a_0 \\ 1 & 0 & \cdots & 0 & -a_1 \\ 0 & 1 & \cdots & 0 & -a_2 \\ \vdots & & \ddots & & \vdots \\ 0 & 0 & \cdots & 1 & -a_{n-1} \end{bmatrix}, \quad C = \begin{bmatrix} 0 & 0 & \cdots & 0 & 1 \end{bmatrix}$$

**특성:** 항상 가관측(Observable). 참고: OCF는 CCF의 **전치(Transpose)** 형태이다 ($B$와 $C$도 전치).

### 4.3 대각형 (모달형, Diagonal/Modal Form)

$A$가 서로 다른 고유값 $\lambda_1, \ldots, \lambda_n$을 가지면 대각화할 수 있다:

$$\bar{A} = T^{-1}AT = \text{diag}(\lambda_1, \ldots, \lambda_n)$$

여기서 $T = [v_1 \; v_2 \; \cdots \; v_n]$은 고유 벡터(Eigenvector)로 이루어진 행렬이다.

대각형에서 각 상태는 독립적으로 진화한다 — 시스템이 $n$개의 1차 모드(Mode)로 분리된다.

### 4.4 조르당 형 (Jordan Form)

$A$가 중복 고유값을 가지면 대각 형식이 존재하지 않을 수 있다. **조르당 표준형(Jordan Normal Form)**이 이를 처리한다:

$$J = \begin{bmatrix} J_1 & & \\ & J_2 & \\ & & \ddots \end{bmatrix}, \quad J_i = \begin{bmatrix} \lambda_i & 1 & \\ & \lambda_i & 1 \\ & & \ddots & 1 \\ & & & \lambda_i \end{bmatrix}$$

## 5. 상태 전이 행렬 (State Transition Matrix)

### 5.1 동차 해 (Homogeneous Solution)

초기 조건 $x(0) = x_0$에서의 $\dot{x} = Ax$ 해:

$$x(t) = e^{At} x_0$$

여기서 **행렬 지수(Matrix Exponential)**는:

$$e^{At} = \Phi(t) = I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + \cdots$$

### 5.2 상태 전이 행렬의 성질 (Properties of the State Transition Matrix)

- $\Phi(0) = I$
- $\Phi(t_1 + t_2) = \Phi(t_1)\Phi(t_2)$
- $\Phi^{-1}(t) = \Phi(-t)$
- $\dot{\Phi}(t) = A\Phi(t)$
- $\Phi(t) = \mathcal{L}^{-1}\{(sI - A)^{-1}\}$

### 5.3 완전 해 (Complete Solution)

초기 조건 $x(0) = x_0$에서의 $\dot{x} = Ax + Bu$ 해:

$$x(t) = e^{At}x_0 + \int_0^t e^{A(t-\tau)}Bu(\tau) \, d\tau$$

첫 번째 항은 **자연 응답(Natural Response)** (초기 조건에 의한 것), 두 번째 항은 **강제 응답(Forced Response)** (컨볼루션 적분)이다.

### 5.4 $e^{At}$ 계산 방법 (Computing $e^{At}$)

**방법 1: 라플라스 변환(Laplace Transform)**

$$e^{At} = \mathcal{L}^{-1}\{(sI - A)^{-1}\}$$

**방법 2: 대각화(Diagonalization)** ($A$가 대각화 가능한 경우)

$$e^{At} = Te^{\Lambda t}T^{-1} = T \text{diag}(e^{\lambda_1 t}, \ldots, e^{\lambda_n t}) T^{-1}$$

**방법 3: 케일리-해밀턴 정리(Cayley-Hamilton Theorem)**

$n \times n$ 행렬에 대해 $e^{At} = \alpha_0(t)I + \alpha_1(t)A + \cdots + \alpha_{n-1}(t)A^{n-1}$이며, 계수는 각 고유값에 대해 $e^{\lambda_i t} = \alpha_0 + \alpha_1\lambda_i + \cdots + \alpha_{n-1}\lambda_i^{n-1}$을 만족한다.

## 6. 고유값과 안정성 (Eigenvalues and Stability)

$A$의 고유값은 전달 함수의 극점(Pole)과 같다. 시스템은:

- **점근 안정(Asymptotically Stable):** 모든 고유값에 대해 $\text{Re}(\lambda_i) < 0$
- **한계 안정(Marginally Stable):** 모든 고유값에 대해 $\text{Re}(\lambda_i) \leq 0$이고, 허수축 위에 중복 고유값 없음
- **불안정(Unstable):** 적어도 하나의 고유값에 대해 $\text{Re}(\lambda_i) > 0$

**특성 다항식(Characteristic Polynomial):**

$$\det(sI - A) = s^n + a_{n-1}s^{n-1} + \cdots + a_0$$

이는 전달 함수 접근 방식의 특성 다항식과 동일하다.

## 7. 유사 변환 (Similarity Transformations)

두 상태 공간 실현 $(A, B, C, D)$와 $(\bar{A}, \bar{B}, \bar{C}, \bar{D})$가 동일한 전달 함수를 나타낼 필요충분조건은 **유사 변환(Similarity Transformation)** $T$에 의해 연결되는 것이다:

$$\bar{A} = T^{-1}AT, \quad \bar{B} = T^{-1}B, \quad \bar{C} = CT, \quad \bar{D} = D$$

유사 변환 하에서 보존되는 주요 특성:
- 고유값 (극점)
- 전달 함수
- 가제어성과 가관측성 (랭크 조건)
- 시스템 차수

## 연습 문제

### 연습 1: 상태 공간 모델링

DC 모터의 방정식:
- $L_a \frac{di_a}{dt} + R_a i_a + K_b \dot{\theta} = v_a$
- $J\ddot{\theta} + B\dot{\theta} = K_t i_a$

$x_1 = \theta$, $x_2 = \dot{\theta}$, $x_3 = i_a$로 설정할 때:

1. 입력 $u = v_a$, 출력 $y = \theta$로 하여 시스템을 상태 공간 형식 $(A, B, C, D)$로 나타내라
2. $G(s) = C(sI - A)^{-1}B$를 이용하여 전달 함수 $\Theta(s)/V_a(s)$를 구하라

### 연습 2: 표준형 변환

$G(s) = \frac{2s + 3}{s^3 + 4s^2 + 5s + 6}$이 주어졌을 때:

1. 가제어 표준형을 작성하라
2. 가관측 표준형을 작성하라
3. $A$의 고유값을 구하고, $G(s)$의 극점과 일치하는지 확인하라

### 연습 3: 상태 전이 행렬

시스템 $A = \begin{bmatrix} 0 & 1 \\ -2 & -3 \end{bmatrix}$에 대해:

1. 고유값을 구하라
2. 라플라스 변환 방법을 사용하여 $e^{At}$를 계산하라
3. 입력이 없고 $x(0) = [1 \; 0]^T$일 때 $x(t)$를 구하라

---

*이전: [레슨 10 — 앞섬-뒤짐 보상](10_Lead_Lag_Compensation.md) | 다음: [레슨 12 — 가제어성과 가관측성](12_Controllability_and_Observability.md)*
