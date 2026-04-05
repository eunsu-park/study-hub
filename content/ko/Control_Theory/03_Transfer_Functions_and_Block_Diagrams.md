# 레슨 3: 전달 함수와 블록 선도

## 학습 목표

- LTI 시스템의 전달 함수(transfer function) 정의 및 계산
- 극점(poles), 영점(zeros)과 시스템 거동에 미치는 영향 파악
- 블록 선도 대수(block diagram algebra) 규칙을 이용한 블록 선도 조작
- 복잡한 블록 선도를 단일 전달 함수로 간소화
- 신호 흐름 선도(signal flow graph)에 메이슨 이득 공식(Mason's gain formula) 적용

## 1. 전달 함수 정의

LTI 시스템의 **전달 함수(transfer function)**는 초기 조건이 0일 때 출력의 라플라스 변환과 입력의 라플라스 변환의 비이다:

$$G(s) = \frac{Y(s)}{U(s)} = \frac{b_m s^m + b_{m-1}s^{m-1} + \cdots + b_0}{a_n s^n + a_{n-1}s^{n-1} + \cdots + a_0}$$

**주요 특성:**
- LTI 시스템에 대해서만 정의됨
- 입력 신호에 독립적
- 입출력 거동에 관한 모든 정보를 포함 (단, 내부 동역학은 제외)
- $m \leq n$ (분자 차수 $\leq$ 분모 차수)이면 시스템은 **진분수형(proper)**
- $m < n$이면 시스템은 **순진분수형(strictly proper)**

## 2. 극점과 영점(Poles and Zeros)

### 2.1 정의

전달 함수를 인수분해하면:

$$G(s) = K \frac{(s - z_1)(s - z_2)\cdots(s - z_m)}{(s - p_1)(s - p_2)\cdots(s - p_n)}$$

- **영점(Zeros)** $z_1, \ldots, z_m$: 분자의 근 — $G(s) = 0$이 되는 $s$ 값
- **극점(Poles)** $p_1, \ldots, p_n$: 분모의 근 — $G(s) \to \infty$가 되는 $s$ 값
- **이득(Gain)** $K$: 최고차 계수의 비

### 2.2 극점 위치와 시간 영역 거동

극점은 시스템의 자연 응답(natural response)을 결정한다:

| 극점 위치 | 시간 응답 | 안정성 |
|----------|----------|--------|
| 실수, 음수 ($s = -a$) | $e^{-at}$ (감쇠 지수) | 안정 |
| 실수, 양수 ($s = +a$) | $e^{at}$ (발산 지수) | 불안정 |
| 복소수 쌍 $s = -\sigma \pm j\omega$ | $e^{-\sigma t}\sin(\omega t + \phi)$ (감쇠 진동) | $\sigma > 0$이면 안정 |
| 순허수 쌍 $s = \pm j\omega$ | $\sin(\omega t + \phi)$ (지속 진동) | 임계 안정 |
| $s = 0$ | 상수 (적분기) | 임계 안정 |

**우세 극점(dominant poles)** (허수축에 가장 가까운 극점)이 주로 과도 응답(transient response)을 결정한다.

### 2.3 영점의 영향

영점은 안정성에 영향을 주지 않지만 과도 응답에는 영향을 미친다:
- **좌반면(Left half-plane) 영점**: 응답을 빠르게 함
- **우반면(Right half-plane, RHP) 영점**: 초기 역응답(역방향 undershoot 후 추종) 유발, 달성 가능한 대역폭 제한
- **극점 근방의 영점**: 해당 극점의 기여를 근사적으로 상쇄

## 3. 표준 전달 함수 형식

### 3.1 1차 시스템(First-Order System)

$$G(s) = \frac{K}{\tau s + 1}$$

- **DC 이득(DC gain)**: $K = G(0)$
- **시정수(Time constant)**: $\tau$ (최종값의 63.2%에 도달하는 시간)
- $s = -1/\tau$에 단일 실수 극점

### 3.2 2차 시스템(Second-Order System)

$$G(s) = \frac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}$$

- **고유 진동수(Natural frequency)**: $\omega_n$
- **감쇠비(Damping ratio)**: $\zeta$
- 극점: $s = -\zeta\omega_n \pm \omega_n\sqrt{\zeta^2 - 1}$

| 감쇠 유형 | $\zeta$ 범위 | 극점 유형 | 응답 |
|----------|------------|---------|------|
| 과감쇠(Overdamped) | $\zeta > 1$ | 두 개의 실수 음수 | 느림, 진동 없음 |
| 임계 감쇠(Critically damped) | $\zeta = 1$ | 중복 실수 | 진동 없이 가장 빠름 |
| 부족 감쇠(Underdamped) | $0 < \zeta < 1$ | 복소 켤레 쌍 | 진동, 감쇠 |
| 비감쇠(Undamped) | $\zeta = 0$ | 순허수 | 지속 진동 |

### 3.3 고차 시스템(Higher-Order Systems)

고차 시스템은 종종 **우세 극점(dominant poles)**으로 근사할 수 있다 — 다른 극점이 허수축으로부터 최소 5배 이상 멀리 있는 경우, 허수축에 가장 가까운 극점이 과도 응답을 지배한다.

## 4. 블록 선도 대수(Block Diagram Algebra)

### 4.1 기본 연결 구조

**직렬(Series/Cascade) 연결:**

```
U → [G₁] → [G₂] → Y
```

$$G(s) = G_1(s) G_2(s)$$

**병렬(Parallel) 연결:**

```
    ┌→ [G₁] →┐
U → ┤         ├→(+)→ Y
    └→ [G₂] →┘
```

$$G(s) = G_1(s) + G_2(s)$$

**음의 피드백(Negative feedback):**

```
R →(+)→ [G] → Y
    ↑          |
    └── [H] ←──┘
```

$$\frac{Y(s)}{R(s)} = \frac{G(s)}{1 + G(s)H(s)}$$

**양의 피드백(Positive feedback):**

$$\frac{Y(s)}{R(s)} = \frac{G(s)}{1 - G(s)H(s)}$$

### 4.2 블록 선도 간소화 규칙

| 연산 | 변환 전 | 변환 후 |
|------|--------|--------|
| 가합점을 블록 앞으로 이동 | $G(E_1 + E_2)$ | $GE_1 + GE_2$ |
| 가합점을 블록 뒤로 이동 | $GE_1 + E_2$ | $G(E_1 + E_2/G)$ |
| 분기점을 블록 앞으로 이동 | $G$ 후에 분기 | $G$ 전에 분기, 브랜치에 $G$ 삽입 |
| 분기점을 블록 뒤로 이동 | $G$ 전에 분기 | $G$ 후에 분기, 브랜치에 $1/G$ 삽입 |
| 가합점 순서 교환 | $(\pm A \pm B) \pm C$ | $(\pm A \pm C) \pm B$ |

### 4.3 간소화 예제

다음 시스템을 고려하자:

```
R →(+)→ [G₁] →(+)→ [G₂] → Y
    ↑              ↑        |
    |    D(s) ─────┘        |
    └────── [H] ←───────────┘
```

**단계 1:** 외란 경로를 합산:
$$Y(s) = G_2(s)[G_1(s)E(s) + D(s)]$$

**단계 2:** $E(s) = R(s) - H(s)Y(s)$로 외부 루프를 닫으면:

$$Y(s) = \frac{G_1(s)G_2(s)}{1 + G_1(s)G_2(s)H(s)} R(s) + \frac{G_2(s)}{1 + G_1(s)G_2(s)H(s)} D(s)$$

이를 통해 피드백이 $\frac{1}{1 + G_1 G_2 H}$ 인자만큼 외란을 감쇠시킴을 알 수 있다.

## 5. 폐루프 전달 함수(Closed-Loop Transfer Functions)

전진 경로(forward path) $G(s)$와 피드백 경로(feedback path) $H(s)$를 가진 표준 피드백 시스템에서:

### 5.1 주요 전달 함수

| 전달 함수 | 명칭 | 공식 |
|---------|------|------|
| $T(s) = \frac{Y}{R}$ | 폐루프 (상보 감도(complementary sensitivity)) | 단위 피드백 시 $\frac{GH_{\text{ff}}}{1+GH}$ 또는 $\frac{G}{1+GH}$ |
| $S(s) = \frac{E}{R}$ | 감도(Sensitivity) | $\frac{1}{1+GH}$ |
| $\frac{Y}{D}$ | 외란-출력(Disturbance-to-output) | $\frac{G_2}{1+G_1 G_2 H}$ |

### 5.2 기본 제약 관계

$$S(s) + T(s) = 1$$

이는 감도와 상보 감도가 **항상 트레이드오프 관계**임을 의미한다: 두 값을 동시에 작게 만들 수 없다. 이는 피드백 제어에서 가장 근본적인 한계 중 하나이다.

## 6. 신호 흐름 선도와 메이슨 공식(Signal Flow Graphs and Mason's Formula)

### 6.1 신호 흐름 선도(Signal Flow Graphs, SFGs)

블록 선도의 대안 — 노드(node)는 신호를 나타내고, 유향 에지(branch)는 이득을 나타낸다.

**블록 선도에서 변환:**
- 각 신호는 **노드**가 됨
- 각 블록은 전달 함수를 이득으로 갖는 **브랜치**가 됨

### 6.2 메이슨 이득 공식(Mason's Gain Formula)

신호 흐름 선도에서 입력-출력 전달 함수는:

$$T = \frac{\sum_k P_k \Delta_k}{\Delta}$$

여기서:
- $P_k$: $k$번째 전진 경로(forward path)의 이득
- $\Delta = 1 - \sum L_i + \sum L_iL_j - \sum L_iL_jL_k + \cdots$
  - $L_i$: $i$번째 개별 루프의 이득
  - $L_iL_j$: 비접촉 두 루프 이득의 곱
  - $L_iL_jL_k$: 비접촉 세 루프 이득의 곱
- $\Delta_k$: $k$번째 전진 경로에 대한 $\Delta$의 여인수(cofactor) (경로 $k$와 접촉하는 모든 루프를 제거)

**두 루프가 비접촉(non-touching)**이란 공통 노드를 공유하지 않음을 의미한다.

### 6.3 메이슨 공식 예제

전진 경로와 루프를 갖는 시스템을 고려하자:
- 전진 경로 1: $P_1 = G_1 G_2 G_3$ (루프 $L_1$, $L_2$와 접촉)
- 전진 경로 2: $P_2 = G_4$ (루프 $L_1$과 접촉)
- 루프 1: $L_1 = -G_1 G_2 H_1$
- 루프 2: $L_2 = -G_2 G_3 H_2$
- 루프 $L_1$과 $L_2$는 접촉 ($G_2$ 이후 노드를 공유)

$$\Delta = 1 - (L_1 + L_2) = 1 + G_1 G_2 H_1 + G_2 G_3 H_2$$

$$\Delta_1 = 1 \quad \text{(모든 루프가 경로 1과 접촉)}$$
$$\Delta_2 = 1 - L_2 = 1 + G_2 G_3 H_2 \quad \text{(루프 2가 경로 2와 비접촉)}$$

$$T = \frac{G_1 G_2 G_3 + G_4(1 + G_2 G_3 H_2)}{1 + G_1 G_2 H_1 + G_2 G_3 H_2}$$

## 7. 특성 방정식(Characteristic Equation)

**특성 방정식(characteristic equation)**은 폐루프 전달 함수의 분모를 0으로 놓음으로써 얻는다:

$$1 + G(s)H(s) = 0$$

또는 $T(s)$의 분모 다항식:

$$a_n s^n + a_{n-1}s^{n-1} + \cdots + a_0 = 0$$

특성 방정식의 근이 **폐루프 극점(closed-loop poles)**이다. 모든 폐루프 안정성 해석 기법(라우스-허비츠(Routh-Hurwitz), 근궤적(root locus), 나이퀴스트(Nyquist))은 근본적으로 이 방정식을 분석한다.

## 연습 문제

### 연습 1: 극점과 영점

다음 전달 함수에 대해:

$$G(s) = \frac{2(s + 3)}{(s + 1)(s^2 + 4s + 8)}$$

1. 모든 극점과 영점을 구하라
2. 극점이 실수인지 복소수인지 확인하라
3. 시스템의 안정성을 판별하라
4. $s$-평면에서 극점-영점 선도(pole-zero plot)를 그려라

### 연습 2: 블록 선도 간소화

다음 블록 선도를 간소화하여 $Y(s)/R(s)$를 구하라:

```
R →(+)→ [G₁] →(+)→ [G₂] →(+)→ [G₃] → Y
    ↑              ↑              ↑     |
    |              └── [H₂] ←────┘     |
    └───────────── [H₁] ←──────────────┘
```

### 연습 3: 메이슨 이득 공식

어떤 시스템의 신호 흐름 선도가 다음 특성을 가진다:
- 전진 경로: $P_1 = ABCD$, $P_2 = AEFD$
- 루프: $L_1 = -BG$, $L_2 = -CH$, $L_3 = -EFHG$
- $L_1$과 $L_2$는 비접촉; 다른 모든 쌍은 접촉

메이슨 공식을 사용하여 $T = Y/R$을 구하라.

---

*이전: [레슨 2 — 물리 시스템의 수학적 모델링](02_Mathematical_Modeling.md) | 다음: [레슨 4 — 시간 영역 해석](04_Time_Domain_Analysis.md)*
