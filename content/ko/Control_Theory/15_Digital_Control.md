# 레슨 15: 디지털 제어 시스템(Digital Control Systems)

## 학습 목표

- 샘플링(sampling) 과정과 제어 시스템에 미치는 영향을 이해한다
- Z-변환(Z-transform)을 적용하여 이산 시간 시스템을 분석한다
- 연속 시간 제어기를 이산 시간 구현으로 변환한다
- Z-영역에서 직접 디지털 제어기를 설계한다
- 이산 시간 시스템의 안정성을 분석한다

## 1. 디지털 제어가 필요한 이유

현대의 제어기는 **디지털 컴퓨터**(마이크로컨트롤러, DSP, FPGA)로 구현된다. 연속 시간 신호는 다음 과정을 거친다:

1. 이산 시각 $t = kT$에서 **샘플링(sampled)** (여기서 $T$는 샘플링 주기)
2. 유한 정밀도로 **양자화(quantized)** (A/D 변환)
3. 디지털 제어기 알고리즘으로 **처리(processed)**
4. 연속 시간 신호로 **재구성(reconstructed)** (영차 유지(Zero-Order Hold, ZOH)를 사용한 D/A 변환)

```
r(t)→[Sampler]→r[k]→[Digital Controller]→u[k]→[ZOH]→u(t)→[Plant]→y(t)
                                                                   ↓
                                                            [Sampler]←─┘
                                                              y[k]
```

### 1.1 디지털 제어의 장점

- 유연성 (하드웨어가 아닌 소프트웨어로 알고리즘 변경 가능)
- 재현성 (부품 드리프트 없음)
- 복잡한 알고리즘 구현 가능 (적응형, 최적, 비선형)
- 디지털 신호의 잡음 내성
- 로깅 및 모니터링 용이

### 1.2 샘플링 정리(Sampling Theorem)

**섀넌/나이퀴스트 샘플링 정리(Shannon/Nyquist sampling theorem):** 대역폭 $\omega_B$의 신호를 재구성하려면 샘플링 주파수가 다음을 만족해야 한다:

$$\omega_s = \frac{2\pi}{T} > 2\omega_B$$

제어 실무에서 샘플링 속도는 보통 다음과 같이 선택한다:

$$\omega_s \geq (10\text{–}30) \times \omega_{BW}$$

여기서 $\omega_{BW}$는 폐루프 대역폭이다. 이를 통해 디지털 제어기가 연속 시간 대응물과 유사하게 동작하도록 보장한다.

## 2. 영차 유지(Zero-Order Hold, ZOH)

**영차 유지(Zero-Order Hold, ZOH)**는 이산 제어 신호를 구간별 상수(piecewise-constant) 연속 신호로 변환한다:

$$u(t) = u[k] \quad \text{for } kT \leq t < (k+1)T$$

ZOH의 전달 함수:

$$G_{\text{ZOH}}(s) = \frac{1 - e^{-sT}}{s}$$

ZOH는 약 $T/2$의 시간 지연을 유발하며, $T$가 너무 크면 위상 지연이 증가하여 시스템이 불안정해질 수 있다.

## 3. Z-변환(Z-Transform)

### 3.1 정의

이산 시간 수열 $f[k]$의 Z-변환은:

$$F(z) = \mathcal{Z}\{f[k]\} = \sum_{k=0}^{\infty} f[k] z^{-k}$$

Z-변환은 라플라스 변환(Laplace transform)의 이산 시간 대응물이다.

### 3.2 주요 Z-변환 쌍

| $f[k]$ | $F(z)$ |
|---------|--------|
| $\delta[k]$ (단위 충격) | $1$ |
| $u[k]$ (단위 계단) | $\frac{z}{z-1}$ |
| $k u[k]$ (램프) | $\frac{Tz}{(z-1)^2}$ |
| $a^k u[k]$ | $\frac{z}{z-a}$ |
| $e^{-akT} u[k]$ | $\frac{z}{z-e^{-aT}}$ |

### 3.3 주요 성질

| 성질 | 시간 영역 | Z-영역 |
|----------|------------|----------|
| 선형성(Linearity) | $af_1[k] + bf_2[k]$ | $aF_1(z) + bF_2(z)$ |
| 시간 지연(Time shift) | $f[k-1]$ | $z^{-1}F(z)$ |
| 시간 전진(Time advance) | $f[k+1]$ | $zF(z) - zf[0]$ |
| 최종값(Final value) | $\lim_{k\to\infty} f[k]$ | $\lim_{z\to 1}(z-1)F(z)$ |

### 3.4 $s$에서 $z$로의 사상(s-to-z Mapping)

$s$-영역과 $z$-영역 사이의 기본 관계:

$$z = e^{sT}$$

이 사상의 대응 관계:
- $s$-평면의 좌반 평면 ($\text{Re}(s) < 0$) → 단위 원 내부 ($|z| < 1$)
- 허수 축 ($\text{Re}(s) = 0$) → 단위 원 ($|z| = 1$)
- $s$-평면의 우반 평면 ($\text{Re}(s) > 0$) → 단위 원 외부 ($|z| > 1$)

**Z-영역에서의 안정성:** 이산 시간 시스템은 모든 극점이 **단위 원 내부** $|z| < 1$에 있을 때, 그리고 그때만 안정하다.

## 4. 이산 시간 전달 함수

### 4.1 펄스 전달 함수(Pulse Transfer Function)

이산 시간 시스템 $y[k] = G(z) U(z)$에 대해:

$$G(z) = \frac{Y(z)}{U(z)} = \frac{b_m z^m + b_{m-1}z^{m-1} + \cdots + b_0}{z^n + a_{n-1}z^{n-1} + \cdots + a_0}$$

### 4.2 ZOH 등가 이산화(ZOH-Equivalent Discretization)

연속 플랜트 $G_p(s)$가 주어질 때, **ZOH 등가** 이산 전달 함수는:

$$G_d(z) = (1 - z^{-1})\mathcal{Z}\left\{\frac{G_p(s)}{s}\right\}$$

이는 ZOH가 앞에 붙은 연속 플랜트의 동작을 샘플링 시각에서 정확히 재현한다.

### 4.3 예제

$G_p(s) = \frac{1}{s+a}$에 대해:

$$\frac{G_p(s)}{s} = \frac{1}{s(s+a)} = \frac{1}{a}\left(\frac{1}{s} - \frac{1}{s+a}\right)$$

$$\mathcal{Z}\left\{\frac{G_p(s)}{s}\right\} = \frac{1}{a}\left(\frac{z}{z-1} - \frac{z}{z-e^{-aT}}\right)$$

$$G_d(z) = \frac{1-e^{-aT}}{z - e^{-aT}}$$

## 5. 연속 제어기의 이산화

### 5.1 일반적인 방법들

연속 제어기 $G_c(s)$가 주어질 때, Z-영역에서 근사화한다:

**전진 오일러(Forward Euler)** (전진 차분):

$$s \approx \frac{z-1}{T} \quad \Rightarrow \quad G_d(z) = G_c(s)\big|_{s=(z-1)/T}$$

**후진 오일러(Backward Euler)** (후진 차분):

$$s \approx \frac{z-1}{Tz} \quad \Rightarrow \quad G_d(z) = G_c(s)\big|_{s=(z-1)/(Tz)}$$

**터스틴 방법(Tustin's method)** (쌍선형 변환(bilinear transform), 사다리꼴 규칙):

$$s \approx \frac{2}{T}\frac{z-1}{z+1} \quad \Rightarrow \quad G_d(z) = G_c(s)\big|_{s=\frac{2}{T}\frac{z-1}{z+1}}$$

### 5.2 비교

| 방법 | 안정성 보존 | 주파수 뒤틀림(Frequency Warping) | 정확도 |
|--------|----------------------|-------------------|----------|
| 전진 오일러(Forward Euler) | 아니오 (안정한 시스템을 불안정하게 사상 가능) | 아니오 | 낮음 |
| 후진 오일러(Backward Euler) | 예 (좌반 평면을 단위 원 내부로 사상) | 아니오 | 낮음 |
| 터스틴(Tustin) | 예 | 예 (주파수 뒤틀림 발생) | 좋음 |
| ZOH 등가(ZOH-equivalent) | 예 | 해당 없음 (샘플링 시각에서 정확) | $t = kT$에서 정확 |

**터스틴 주파수 뒤틀림:** 쌍선형 변환은 $\omega_s$ (연속)를 $\omega_d$ (이산)로 비선형적으로 사상한다:

$$\omega_d = \frac{2}{T}\tan\frac{\omega_s T}{2}$$

중요한 주파수(교차 주파수, 노치 주파수)에 대해서는 쌍선형 변환 적용 전에 연속 주파수를 조정하는 **사전 뒤틀림(pre-warp)**을 수행한다.

### 5.3 디지털 PID 구현

연속 PID $u(t) = K_p e + K_i \int e \, dt + K_d \dot{e}$는 다음과 같이 된다:

**위치형(Positional form)** (적분에 터스틴, 미분에 후진 오일러 사용):

$$u[k] = K_p e[k] + K_i T \sum_{j=0}^{k} e[j] + K_d \frac{e[k] - e[k-1]}{T}$$

**속도형(Velocity, incremental form)** (선호됨 — 자연스럽게 적분 와인드업(integral windup) 방지):

$$\Delta u[k] = K_p(e[k] - e[k-1]) + K_i T e[k] + K_d \frac{e[k] - 2e[k-1] + e[k-2]}{T}$$

$$u[k] = u[k-1] + \Delta u[k]$$

## 6. 이산 시간 안정성 분석

### 6.1 쥬리 안정성 검사(Jury Stability Test)

루스-헐비츠 판별법(Routh-Hurwitz criterion)의 이산 시간 대응물이다. 특성 다항식:

$$P(z) = a_n z^n + a_{n-1}z^{n-1} + \cdots + a_0$$

**필요 조건** (모두 성립해야 함):
1. $P(1) > 0$
2. $(-1)^n P(-1) > 0$
3. $|a_0| < a_n$

완전한 쥬리 배열(Jury array)이 충분 조건을 제공한다 (루스 배열과 유사).

### 6.2 루스 판별법 적용을 위한 쌍선형 변환

대안적인 방법: $z = \frac{w+1}{w-1}$ (쌍선형 변환)을 $P(z)$에 대입하여 $w$에 대한 다항식을 구한다. $z$-영역의 단위 원이 $w$-영역의 허수 축으로 사상되므로, 표준 루스-헐비츠 판별법을 $w$-다항식에 적용할 수 있다.

## 7. 직접 디지털 설계(Direct Digital Design)

연속 시간에서 설계 후 이산화하는 대신, Z-영역에서 **직접** 설계한다:

### 7.1 데드비트 제어(Dead-Beat Control)

**유한 시간 내에 영 오차**를 달성하는 폐루프 전달 함수를 선택한다 ($n$차 시스템의 경우 보통 $n$번의 샘플링 주기 내):

$$T(z) = z^{-d}$$

적절한 지연 $d$에 대해. 제어기는:

$$G_c(z) = \frac{T(z)}{(1 - T(z))G_d(z)}$$

데드비트 제어기는 빠르지만 큰 제어 신호가 필요하고 모델 오차에 민감하다.

### 7.2 이산 근궤적(Discrete Root Locus)

근궤적(root locus) 기법은 Z-영역에서도 동일하게 적용되며, 안정성 경계는 허수 축 대신 **단위 원**이다.

## 연습 문제

### 연습 1: ZOH 이산화

플랜트 $G_p(s) = \frac{5}{s(s+5)}$, 샘플링 주기 $T = 0.1$ s에 대해:

1. ZOH 등가 이산 전달 함수 $G_d(z)$를 계산하라
2. 이산 극점을 구하고 연속 극점에 $z = e^{sT}$를 적용한 결과와 일치하는지 확인하라
3. DC 이득 검증: 계단 응답에 대해 $G_d(1) = G_p(0) \cdot T$

### 연습 2: 디지털 PID

연속 PI 제어기 $G_c(s) = 2(1 + \frac{1}{0.5s})$를 $T = 0.05$ s로 디지털 구현한다.

1. 터스틴 방법으로 이산화하라
2. 차분 방정식 $u[k] = f(u[k-1], e[k], e[k-1])$을 작성하라
3. 속도형 구현과 비교하라

### 연습 3: 안정성 분석

이산 시간 특성 다항식 $P(z) = z^3 - 1.2z^2 + 0.5z - 0.1$에 대해:

1. 쥬리 검사의 필요 조건을 적용하라
2. 시스템의 안정성 여부를 판단하라
3. 실제 근을 구하고 검증하라

---

*이전: [레슨 14 — 최적 제어: LQR과 칼만 필터](14_Optimal_Control.md) | 다음: [레슨 16 — 비선형 제어와 고급 주제](16_Nonlinear_and_Advanced.md)*
