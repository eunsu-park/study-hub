# 레슨 7: 주파수 응답 — 보드 선도(Bode Plots)

## 학습 목표

- LTI 시스템의 주파수 응답(Frequency Response) 개념 이해
- 전달함수로부터 보드 크기 선도와 위상 선도 작성
- 빠른 스케치를 위한 점근 근사(Asymptotic Approximation) 활용
- 보드 선도에서 이득 여유(Gain Margin)와 위상 여유(Phase Margin) 읽기
- 주파수 영역 사양을 적용하여 시스템 성능 평가

## 1. 주파수 응답 개념

**주파수 응답(Frequency Response)**은 시스템이 서로 다른 주파수의 정현파 입력에 어떻게 반응하는지를 나타낸다.

전달함수 $G(s)$를 갖는 안정한 LTI 시스템에서 입력이 다음과 같을 때:

$$u(t) = A\sin(\omega t)$$

정상상태 출력은 다음과 같다:

$$y_{ss}(t) = A|G(j\omega)|\sin(\omega t + \angle G(j\omega))$$

시스템은 진폭을 $|G(j\omega)|$배 **스케일**하고, 위상을 $\angle G(j\omega)$만큼 **이동**시킨다.

### 1.1 주파수 응답 함수(Frequency Response Function)

$$G(j\omega) = |G(j\omega)| e^{j\angle G(j\omega)}$$

- **크기(Magnitude):** $|G(j\omega)| = \sqrt{[\text{Re}(G(j\omega))]^2 + [\text{Im}(G(j\omega))]^2}$
- **위상(Phase):** $\angle G(j\omega) = \tan^{-1}\frac{\text{Im}(G(j\omega))}{\text{Re}(G(j\omega))}$

## 2. 보드 선도 기초

**보드 선도(Bode plot)**는 두 개의 그래프로 구성된다:
1. **크기 선도(Magnitude plot)**: $20\log_{10}|G(j\omega)|$ (데시벨, dB) vs. $\log_{10}\omega$
2. **위상 선도(Phase plot)**: $\angle G(j\omega)$ (도, degree) vs. $\log_{10}\omega$

### 2.1 로그 스케일 선도의 장점

- 전달함수의 곱이 개별 보드 선도의 **합**으로 표현됨
- 거듭제곱은 **곱셈**이 됨: $|G^n| \text{ dB} = n \times |G| \text{ dB}$
- 직선 점근 근사를 쉽게 그릴 수 있음
- 넓은 주파수 범위(디케이드 단위)를 다룰 수 있음

### 2.2 데시벨 스케일(Decibel Scale)

$$|G|_{\text{dB}} = 20\log_{10}|G(j\omega)|$$

| 선형값 | dB |
|--------|-----|
| $1$ | $0$ dB |
| $2$ | $6$ dB |
| $10$ | $20$ dB |
| $100$ | $40$ dB |
| $0.1$ | $-20$ dB |
| $0.01$ | $-40$ dB |

## 3. 기본 요소의 보드 선도

임의의 전달함수는 다음과 같은 기본 요소로 분해할 수 있다:

$$G(s) = \frac{K \prod(1 + s/z_i) \prod(1 + 2\zeta_k s/\omega_{n_k} + s^2/\omega_{n_k}^2)}{s^N \prod(1 + s/p_j) \prod(1 + 2\zeta_l s/\omega_{n_l} + s^2/\omega_{n_l}^2)}$$

### 3.1 상수 이득 $K$

- 크기: $20\log_{10}|K|$ dB (수평선)
- 위상: $K > 0$이면 $0°$, $K < 0$이면 $-180°$

### 3.2 적분기/미분기(Integrator/Differentiator) $s^{\pm N}$

**적분기(Integrator)** $1/s$:
- 크기: $-20$ dB/디케이드 ($\omega = 1$에서 0 dB를 통과하는 직선)
- 위상: $-90°$ (상수)

**이중 적분기(Double integrator)** $1/s^2$:
- 크기: $-40$ dB/디케이드
- 위상: $-180°$

### 3.3 1차 인수(First-Order Factor) $(1 + s/\omega_0)^{\pm 1}$

**1차 영점(First-order zero)** $(1 + s/\omega_0)$:

| 주파수 | 크기 점근선 | 위상 |
|--------|-----------|------|
| $\omega \ll \omega_0$ | $0$ dB | $0°$ |
| $\omega = \omega_0$ | $+3$ dB | $+45°$ |
| $\omega \gg \omega_0$ | $+20$ dB/디케이드 | $+90°$ |

- **코너(절점) 주파수(Corner/break frequency):** $\omega_0$
- 위상 전이: $0.1\omega_0$에서 $10\omega_0$까지

**1차 극점(First-order pole)** $1/(1 + s/\omega_0)$: 크기와 위상은 영점과 부호가 반대이다.

### 3.4 2차 인수(Second-Order Factor)

**미감쇠 2차 인수(Underdamped quadratic)** $\frac{1}{1 + 2\zeta s/\omega_n + s^2/\omega_n^2}$:

| 주파수 | 크기 점근선 | 위상 |
|--------|-----------|------|
| $\omega \ll \omega_n$ | $0$ dB | $0°$ |
| $\omega = \omega_n$ | $-20\log_{10}(2\zeta)$ dB (공진 피크) | $-90°$ |
| $\omega \gg \omega_n$ | $-40$ dB/디케이드 | $-180°$ |

공진 피크 높이는 $\zeta$에 따라 달라진다:
- $\zeta = 0.1$: $+14$ dB 피크
- $\zeta = 0.3$: $+5.7$ dB 피크
- $\zeta = 0.707$: $0$ dB (피크 없음, $\omega_n$에서 $-3$ dB)
- $\zeta \geq 0.707$: 공진 피크 없음

**공진 주파수(Resonant frequency):** $\omega_r = \omega_n\sqrt{1 - 2\zeta^2}$ ($\zeta < 1/\sqrt{2}$인 경우)

**공진 피크 크기(Resonant peak magnitude):** $M_r = \frac{1}{2\zeta\sqrt{1-\zeta^2}}$

## 4. 복합 보드 선도 작성

### 4.1 절차

1. $G(s)$를 **시상수 형태(time-constant form)**로 재작성: 상수를 묶어 각 항이 $(1 + s/\omega_i)$ 또는 $(1 + 2\zeta s/\omega_n + s^2/\omega_n^2)$ 꼴이 되도록 인수분해
2. 모든 코너 주파수를 파악하고 정렬
3. 각 요소의 기여분을 개별적으로 선도
4. 모든 크기 기여분을 **합산** (dB는 더함)
5. 모든 위상 기여분을 **합산**

### 4.2 예제

$$G(s) = \frac{100(s+1)}{s(s+10)} = \frac{10(1+s)}{s(1+s/10)}$$

DC 이득(적분기 제거 후 $\omega \to 0$에서 정규화): $\omega = 1$에서 크기는 $10/\omega = 10$ → $20$ dB.

**구성 요소:**
- 이득 $10$: $+20$ dB
- 적분기 $1/s$: $-20$ dB/디케이드, $\omega = 1$에서 $0$ dB 통과
- $\omega = 1$에서 영점: 상향 절점
- $\omega = 10$에서 극점: 하향 절점

**점근 크기:**
- $\omega < 1$: 기울기 $-20$ dB/디케이드 (적분기만)
- $1 < \omega < 10$: 기울기 $0$ dB/디케이드 (적분기 + 영점 상쇄)
- $\omega > 10$: 기울기 $-20$ dB/디케이드 (적분기 + 영점 + 극점)

**위상:**
- $\omega \to 0$: $-90°$ (적분기) + $0°$ + $0°$ = $-90°$
- $\omega = 1$: $-90°$ + $45°$ + $(-5.7°)$ ≈ $-51°$
- $\omega = 10$: $-90°$ + $84.3°$ + $(-45°)$ ≈ $-51°$
- $\omega \to \infty$: $-90°$ + $90°$ + $(-90°)$ = $-90°$

## 5. 보드 선도에서 안정 여유 읽기

### 5.1 이득 여유(Gain Margin, GM)

**이득 여유(GM)**는 시스템이 불안정해지기 전까지 이득을 얼마나 더 증가시킬 수 있는지를 나타낸다(dB 단위):

$$GM = -20\log_{10}|G(j\omega_{pc})| \quad \text{dB}$$

여기서 $\omega_{pc}$는 **위상 교차 주파수(Phase crossover frequency)**이다($\angle G(j\omega) = -180°$가 되는 주파수).

**해석:** 이득 여유는 루프 이득이 $-180°$ 위상 교차점에서 0 dB에 도달하기까지 $K$를 몇 dB 더 증가시킬 수 있는지를 나타낸다.

### 5.2 위상 여유(Phase Margin, PM)

**위상 여유(PM)**는 이득 교차점에서 $-180°$에 도달하기까지 추가로 필요한 위상 지연이다:

$$PM = 180° + \angle G(j\omega_{gc})$$

여기서 $\omega_{gc}$는 **이득 교차 주파수(Gain crossover frequency)**이다($|G(j\omega)| = 0$ dB, 즉 단위 이득이 되는 주파수).

### 5.3 여유로 본 안정성

| 조건 | 안정? |
|------|-------|
| $GM > 0$ dB 이고 $PM > 0°$ | 안정 |
| $GM < 0$ dB 이거나 $PM < 0°$ | 불안정 |
| $GM = 0$ dB 이거나 $PM = 0°$ | 임계 안정(Marginally stable) |

**일반적인 설계 목표:**
- $GM \geq 6$ dB (이득 허용치 2배)
- $PM \geq 30°$에서 $60°$
- $PM \approx 60°$이면 $\zeta \approx 0.6$ (속도와 감쇠의 균형)

### 5.4 위상 여유와 감쇠비의 관계

2차 시스템에서 위상 여유와 감쇠비(Damping ratio)는 근사적으로 다음 관계를 갖는다:

$$PM \approx 100\zeta \quad \text{도(degrees), } \zeta < 0.7$$

더 정확하게는: $\zeta \approx PM/100$ ($PM < 70°$인 경우).

## 6. 주파수 영역 사양(Frequency-Domain Specifications)

| 사양 | 기호 | 의미 |
|------|------|------|
| **대역폭(Bandwidth)** | $\omega_{BW}$ | $|T(j\omega)|$가 $-3$ dB까지 감소하는 주파수 |
| **공진 피크(Resonant peak)** | $M_r$ | $|T(j\omega)|$의 최댓값 |
| **이득 교차 주파수(Gain crossover frequency)** | $\omega_{gc}$ | 폐루프 대역폭의 근사값 |

**시간 영역과의 연관:**
- 큰 $\omega_{BW}$ → 빠른 응답(작은 $t_r$)
- 큰 $M_r$ → 더 많은 오버슈트(Overshoot)
- $\omega_{BW} \approx \omega_n\sqrt{(1-2\zeta^2) + \sqrt{4\zeta^4 - 4\zeta^2 + 2}}$ (2차 시스템)

## 연습 문제

### 연습 1: 보드 선도 작성

다음에 대해 점근 보드 선도(크기 및 위상)를 스케치하라:

$$G(s) = \frac{50(s+5)}{s(s+2)(s+50)}$$

1. 시상수 형태로 재작성
2. 모든 코너 주파수 파악
3. 점근 크기 선도와 위상 선도 작성
4. 이득 여유와 위상 여유 결정

### 연습 2: 보드 선도에서 시스템 식별

어떤 시스템의 보드 크기 선도가 다음과 같이 나타난다:
- $\omega < 2$에서 기울기 $-20$ dB/디케이드
- $2 < \omega < 10$에서 기울기 $-40$ dB/디케이드
- $\omega > 10$에서 기울기 $-60$ dB/디케이드
- $\omega = 1$에서 크기 $20$ dB

전달함수를 구하라.

### 연습 3: 안정 여유 분석

$G(s) = \frac{K}{s(0.1s+1)(0.01s+1)}$에 대해:

1. $K = 10$일 때 이득 여유와 위상 여유를 구하라
2. 시스템이 안정한 $K$의 최댓값을 구하라
3. 위상 여유가 $45°$가 되는 $K$ 값은?

---

*이전: [레슨 6 — 근궤적법](06_Root_Locus.md) | 다음: [레슨 8 — 나이퀴스트 안정도 기준](08_Nyquist_Stability.md)*
