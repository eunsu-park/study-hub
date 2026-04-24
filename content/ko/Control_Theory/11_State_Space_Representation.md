# 레슨 11: 상태 공간 표현 (State-Space Representation)

## 학습 목표

- 동적 시스템을 상태 공간(State Space) 형식으로 표현한다
- 전달함수와 상태 공간 모델 간 변환을 수행한다
- 가제어 표준형(controllable canonical form), 가관측 표준형(observable canonical form), 대각(모드) 형태(diagonal/modal form)를 식별한다
- 상태 공간 표현과 전달함수 표현의 관계를 이해한다
- 상태 천이 행렬(state transition matrix)을 계산하고 상태 방정식을 푼다
- 파이썬으로 상태 공간 객체를 만들고 전달함수와의 변환을 검증한다

## 0. 왜 전달함수에서 상태 공간으로 옮기는가?

전달함수 기계(machinery) 8개 레슨 후, 새로운 표현? 자체 도구 모음을 가질 만한 세 가지 이유:

- **MIMO가 자연스럽다.** 전달함수는 다항식의 단일 비율이다. 상태 공간 모델은 행렬 — 4 입력, 6 출력, 12 상태 항공기 모델은 `(A: 12×12, B: 12×4, C: 6×12, D: 6×4)`이고 수학은 SISO 경우와 다르지 않게 읽힌다.
- **내부 안정성이 보인다.** 두 전달함수가 동일하면서 하나는 내부적으로 불안정한 시스템(영점에 의해 소거된 숨은 RHP 극점)을 나타낼 수 있다. 상태 공간은 그것을 숨길 수 없다 — $A$의 고유값은 소거 여부와 관계없이 실제 폐루프 극점이다.
- **현대 제어기는 상태 공간이다.** LQR, 칼만 필터, 모델 예측 제어, $H_\infty$ — 1960년대 이후 설계된 모든 제어기는 상태 공간으로 정식화된다. 전달함수는 교육적으로 친숙하지만 차수 10 정도에서 확장이 멈춘다.

머릿속 모델: 전달함수는 "주파수 응답을 가진 블랙 박스"; 상태 공간 모델은 "상태를 측정·제어·추정할 수 있는 내부 메커니즘". 사고의 전환은 입출력에서 내부로 이루어진다.

## 1. 전달함수에서 상태 공간으로

전달함수는 입출력 거동만 포착한다. **상태 공간 표현(state-space representation)**은 전체 내부 동역학을 포착하여 다음을 가능케 한다:
- MIMO(다입력, 다출력) 시스템 분석
- 내부 안정성 분석(BIBO만이 아님)
- 체계적인 제어기 및 관측기 설계
- 비선형 시스템 처리(선형화된 상태 모델을 통해)

## 2. 상태 공간 방정식

연속 시간 LTI 시스템의 상태 공간 형식:

$$\dot{x}(t) = Ax(t) + Bu(t) \quad \text{(상태 방정식)}$$
$$y(t) = Cx(t) + Du(t) \quad \text{(출력 방정식)}$$

여기서:
- $x(t) \in \mathbb{R}^n$: **상태 벡터(state vector)** ($n$ = 시스템 차수)
- $u(t) \in \mathbb{R}^m$: **입력 벡터**
- $y(t) \in \mathbb{R}^p$: **출력 벡터**
- $A \in \mathbb{R}^{n \times n}$: **시스템 행렬(system matrix)** (또는 상태 행렬)
- $B \in \mathbb{R}^{n \times m}$: **입력 행렬**
- $C \in \mathbb{R}^{p \times n}$: **출력 행렬**
- $D \in \mathbb{R}^{p \times m}$: **전향(feedforward) 행렬** (보통 0)

### 2.1 블록 선도

```
u(t) → [B] →(+)→ [∫] → x(t) → [C] →(+)→ y(t)
              ↑                        ↑
              └── [A] ←───────────┘    [D] ← u(t)
```

## 3. 상태 공간 모델 유도

### 3.1 미분방정식으로부터

**예제:** 질량-스프링-댐퍼: $m\ddot{y} + b\dot{y} + ky = F$

상태 변수 선택: $x_1 = y$, $x_2 = \dot{y}$

$$\dot{x}_1 = x_2$$
$$\dot{x}_2 = -\frac{k}{m}x_1 - \frac{b}{m}x_2 + \frac{1}{m}F$$

행렬 형태:

$$\begin{bmatrix} \dot{x}_1 \\ \dot{x}_2 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ -k/m & -b/m \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} 0 \\ 1/m \end{bmatrix} F$$

$$y = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

### 3.2 전달함수로부터

$G(s) = \frac{b_1 s + b_0}{s^2 + a_1 s + a_0}$가 주어지면, **가제어 표준형(controllable canonical form, CCF)**:

$$A = \begin{bmatrix} 0 & 1 \\ -a_0 & -a_1 \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad C = \begin{bmatrix} b_0 & b_1 \end{bmatrix}, \quad D = 0$$

$n$차 시스템 $G(s) = \frac{b_{n-1}s^{n-1} + \cdots + b_0}{s^n + a_{n-1}s^{n-1} + \cdots + a_0}$:

$$A = \begin{bmatrix} 0 & 1 & 0 & \cdots & 0 \\ 0 & 0 & 1 & \cdots & 0 \\ \vdots & & & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & 1 \\ -a_0 & -a_1 & -a_2 & \cdots & -a_{n-1} \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 0 \\ 1 \end{bmatrix}$$

### 3.3 상태 공간에서 전달함수로

라플라스 변환(0 초기 조건):

$$sX(s) = AX(s) + BU(s) \Rightarrow X(s) = (sI - A)^{-1}BU(s)$$

$$Y(s) = [C(sI - A)^{-1}B + D]U(s)$$

따라서:

$$G(s) = C(sI - A)^{-1}B + D$$

### 3.4 파이썬으로 변환

몇 줄로 양방향 매핑이 확인된다. 1:1 대응은 $G$에 극점-영점 소거가 없을 때만 한 방향 — 모든 상태 공간 실현은 유효한 전달함수이지만, 주어진 전달함수에 대해 동등하지 않은 여러 상태 공간 실현이 존재한다(표준형들 포함).

```python
import numpy as np
from control import tf, ss, ss2tf, tf2ss

# 전달함수에서 시작
G = tf([2, 3], [1, 4, 5, 6])
print("Transfer function:", G)

# 상태 공간으로 변환 (python-control 기본은 가제어 표준형)
sys_ss = tf2ss(G)
print("A =\n", sys_ss.A)
print("B =\n", sys_ss.B)
print("C =", sys_ss.C)
print("D =", sys_ss.D)

# 전달함수로 왕복
G_back = ss2tf(sys_ss)
print("Round-trip TF:", G_back)
```

왕복 $G \to (A, B, C, D) \to G$는 정확히 일치해야 한다. 그렇지 않다면 가장 흔한 원인은 숨은 모드 — 변환이 조용히 떨어뜨린 비가제어 또는 비가관측 상태 — 이다.

## 4. 표준형(Canonical Forms)

### 4.1 가제어 표준형(Controllable Canonical Form, CCF)

위에서 보인 대로. $A$의 마지막 행은 특성 다항식의 음수 계수를 담는다.

**성질:** 항상 가제어(by construction).

### 4.2 가관측 표준형(Observable Canonical Form, OCF)

$$A = \begin{bmatrix} 0 & 0 & \cdots & 0 & -a_0 \\ 1 & 0 & \cdots & 0 & -a_1 \\ 0 & 1 & \cdots & 0 & -a_2 \\ \vdots & & \ddots & & \vdots \\ 0 & 0 & \cdots & 1 & -a_{n-1} \end{bmatrix}, \quad C = \begin{bmatrix} 0 & 0 & \cdots & 0 & 1 \end{bmatrix}$$

**성질:** 항상 가관측. 참고: OCF는 CCF의 **전치(transpose)**이다($B$와 $C$도 전치).

### 4.3 대각(모드) 형태(Diagonal / Modal Form)

$A$가 서로 다른 고유값 $\lambda_1, \ldots, \lambda_n$를 가지면 대각화 가능:

$$\bar{A} = T^{-1}AT = \text{diag}(\lambda_1, \ldots, \lambda_n)$$

여기서 $T = [v_1 \; v_2 \; \cdots \; v_n]$은 고유벡터 행렬.

대각 형태의 각 상태는 독립적으로 진화한다 — 시스템이 $n$개의 1차 모드로 분리된다.

### 4.4 Jordan 형태(Jordan Form)

$A$가 중복 고유값을 가지면 대각 형태가 존재하지 않을 수 있다. **Jordan 정규형(Jordan normal form)**이 이를 처리한다:

$$J = \begin{bmatrix} J_1 & & \\ & J_2 & \\ & & \ddots \end{bmatrix}, \quad J_i = \begin{bmatrix} \lambda_i & 1 & \\ & \lambda_i & 1 \\ & & \ddots & 1 \\ & & & \lambda_i \end{bmatrix}$$

### 4.5 형태 선택

| 형태 | 가장 적합한 용도 |
|------|----------|
| 가제어 표준형 | 상태 귀환 설계 ($u = -Kx$) — 이득 배치가 직접적 |
| 가관측 표준형 | 관측기 설계 — 출력이 단일 상태에 결합 |
| 대각 / 모드 | 우세 모드 분석; 분리된 시뮬레이션; LQR 가중치 |
| Jordan | 중복 고유값을 가진 이론적 분석 |
| 물리적 (예: 위 SMD) | 모델을 실제 시스템과 매칭하는 데 가장 적합; 직관 보존 |

실무에서는 보통 모델링을 위해 시스템을 물리 형태로 유지한 다음, 설계를 위해 표준형이나 모드 형태로 변환한다.

## 5. 상태 천이 행렬(State Transition Matrix)

### 5.1 동차 해(Homogeneous Solution)

$\dot{x} = Ax$, 초기 조건 $x(0) = x_0$:

$$x(t) = e^{At} x_0$$

여기서 **행렬 지수(matrix exponential)**:

$$e^{At} = \Phi(t) = I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + \cdots$$

### 5.2 상태 천이 행렬의 성질

- $\Phi(0) = I$
- $\Phi(t_1 + t_2) = \Phi(t_1)\Phi(t_2)$
- $\Phi^{-1}(t) = \Phi(-t)$
- $\dot{\Phi}(t) = A\Phi(t)$
- $\Phi(t) = \mathcal{L}^{-1}\{(sI - A)^{-1}\}$

### 5.3 완전 해(Complete Solution)

$\dot{x} = Ax + Bu$, 초기 조건 $x(0) = x_0$:

$$x(t) = e^{At}x_0 + \int_0^t e^{A(t-\tau)}Bu(\tau) \, d\tau$$

첫 항은 **자연 응답(natural response)**(초기 조건에 의한 것), 둘째 항은 **강제 응답(forced response)**(합성곱 적분).

### 5.4 $e^{At}$ 계산

**방법 1: 라플라스 변환**

$$e^{At} = \mathcal{L}^{-1}\{(sI - A)^{-1}\}$$

**방법 2: 대각화** ($A$가 대각화 가능한 경우)

$$e^{At} = Te^{\Lambda t}T^{-1} = T \text{diag}(e^{\lambda_1 t}, \ldots, e^{\lambda_n t}) T^{-1}$$

**방법 3: Cayley-Hamilton 정리**

$n \times n$ 행렬에 대해, $e^{At} = \alpha_0(t)I + \alpha_1(t)A + \cdots + \alpha_{n-1}(t)A^{n-1}$, 여기서 계수는 각 고유값에 대해 $e^{\lambda_i t} = \alpha_0 + \alpha_1\lambda_i + \cdots + \alpha_{n-1}\lambda_i^{n-1}$를 만족.

**방법 4 (수치, 소프트웨어 기본): 스케일링과 제곱 + Padé 근사**, `scipy.linalg.expm`에 구현됨. 차수 ~50 이하의 행렬에는 사실상 완벽; 그 이상은 Krylov 방법 사용. 프로그래밍 시 사용할 방법이다.

```python
from scipy.linalg import expm
import numpy as np

A = np.array([[0, 1], [-2, -3]], dtype=float)
print("e^(A * 0.5) =\n", expm(A * 0.5))
```

## 6. 고유값과 안정성

$A$의 고유값은 전달함수의 극점이다. 시스템은:

- **점근적 안정:** 모든 고유값이 $\text{Re}(\lambda_i) < 0$
- **경계 안정:** 모든 고유값이 $\text{Re}(\lambda_i) \leq 0$이고 허수축 위 중복 고유값이 없음
- **불안정:** 적어도 하나의 고유값이 $\text{Re}(\lambda_i) > 0$

**특성 다항식:**

$$\det(sI - A) = s^n + a_{n-1}s^{n-1} + \cdots + a_0$$

이는 전달함수 접근에서의 특성 다항식과 같다.

> **상태 공간 테스트가 전달함수 테스트보다 엄격한 이유:** 전달함수가 $s = +1$에서 극점-영점 소거를 가지면, 다항식 $\det(sI - A)$은 여전히 고유값 $\lambda = 1$을 가진다 — 행렬 관점은 모드를 잃지 않는다. 이것이 "내부 안정성"이 $G(s)$의 안정성만이 아니라 $A$의 고유값 확인을 요구하는 형식적 이유이다.

## 7. 유사 변환(Similarity Transformations)

두 상태 공간 실현 $(A, B, C, D)$와 $(\bar{A}, \bar{B}, \bar{C}, \bar{D})$가 같은 전달함수를 나타낼 필요충분조건은 **유사 변환** $T$에 의해 관련되는 것이다:

$$\bar{A} = T^{-1}AT, \quad \bar{B} = T^{-1}B, \quad \bar{C} = CT, \quad \bar{D} = D$$

유사 변환에서 보존되는 주요 성질:
- 고유값(극점)
- 전달함수
- 가제어성과 가관측성(rank 조건)
- 시스템 차수

## 8. 흔한 함정

1. **"상태"와 "출력"의 혼동.** 상태는 당신이 선택하는 내부 변수; 출력은 측정하는 것이다. 3차 시스템은 3개의 상태를 가지지만 그중 1개만 출력할 수도 있다. 초보자들은 종종 "출력은 상태이다"라고 쓰는데 — $C = I$인 경우에만 참.
2. **CCF를 유일한 상태 공간 모델로 취급.** 주어진 전달함수는 무한히 많은 상태 공간 실현을 허용한다. CCF는 설계용 유용한 기본값이지만, 모델링에는 보통 물리 형태가 더 해석 가능하다.
3. **수치로 충분한데 $sI - A$를 기호적으로 역산.** 수치 작업에서는 `scipy.signal.ss2tf`로 $C(sI-A)^{-1}B$를 계산하는 것이 최선 — 손 역산이 망가뜨리는 ill-conditioned $A$ 행렬도 처리한다.
4. **Jordan 블록 잘못 처리.** $A$가 고유벡터의 완전한 집합이 부족한 중복 고유값을 가지면, 단순 대각화는 조용히 실패한다(잘못된 결과). `numpy.linalg.eig`를 사용하고 고유벡터 행렬의 rank를 확인하라; $n$보다 작으면 Jordan 구조를 자동 처리하는 `scipy.linalg.expm`로 전환.
5. **유사 변환 후 숨은 모드.** $T$가 잘못 조건화되면, 변환된 시스템이 원본은 그렇지 않은데도 수치적으로 비가제어 또는 비가관측이 될 수 있다. 수치 변환 후 항상 가제어성/가관측성 확인(레슨 12).
6. **진성이지만 엄밀하게 진성이 아닌 전달함수에 대해 $D \neq 0$ 잊음.** $G(s) = (s+1)/(s+2)$는 DC 이득 $1/2$, 고주파 이득 $1$ — 차이는 $D = 1$에 인코딩된다. $D = 0$으로 두면 고주파 내용이 조용히 잘려 나간다.

## 연습 문제

### 연습 문제 1: 상태 공간 모델링

DC 모터 방정식:
- $L_a \frac{di_a}{dt} + R_a i_a + K_b \dot{\theta} = v_a$
- $J\ddot{\theta} + B\dot{\theta} = K_t i_a$

$x_1 = \theta$, $x_2 = \dot{\theta}$, $x_3 = i_a$로:

1. 입력 $u = v_a$, 출력 $y = \theta$인 상태 공간 형태 $(A, B, C, D)$로 작성
2. $G(s) = C(sI - A)^{-1}B$를 사용하여 전달함수 $\Theta(s)/V_a(s)$를 구하라

### 연습 문제 2: 표준형

$G(s) = \frac{2s + 3}{s^3 + 4s^2 + 5s + 6}$:

1. 가제어 표준형을 작성하라
2. 가관측 표준형을 작성하라
3. $A$의 고유값을 구하고 $G(s)$의 극점과 일치함을 검증하라

### 연습 문제 3: 상태 천이 행렬

시스템 $A = \begin{bmatrix} 0 & 1 \\ -2 & -3 \end{bmatrix}$:

1. 고유값을 구하라
2. 라플라스 변환 방법으로 $e^{At}$를 계산하라
3. 입력 없이 $x(0) = [1 \; 0]^T$에 대한 $x(t)$를 구하라

### 연습 문제 4: 수치 왕복

`python-control`의 `tf2ss`와 `ss2tf` (또는 MATLAB 등가물)를 사용해 $G(s) = \frac{s+1}{(s+2)(s+3)}$을 상태 공간으로 변환하고 다시 돌려라. 복원된 전달함수가 원본과 수치 정밀도 내로 일치하는지 검증하라. $G(s) = \frac{s+1}{(s+2)^2}$ (중복 극점)에 대해 반복하고 무엇이 달라지는지 논하라.

### 연습 문제 5: 유사 변환 드릴

3.1절 SMD 시스템에서 $m = 1, b = 2, k = 5$를 가정하고, 고유벡터 행렬 $T$를 사용해 대각 형태로 변환하라. 변환된 $\bar{A}$가 원본 $A$의 고유값으로 대각이고, 입출력 거동(전달함수)이 변하지 않음을 검증하라.

---

*이전: [레슨 10 — 앞섬-뒤짐 보상](10_Lead_Lag_Compensation.md) | 다음: [레슨 12 — 가제어성과 가관측성](12_Controllability_and_Observability.md)*
