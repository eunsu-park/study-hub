# 신호와 시스템

**다음**: [02. LTI 시스템과 합성곱](./02_LTI_Systems_and_Convolution.md)

---

신호는 우리 주변 어디에나 존재합니다. 사람의 목소리, 회로의 전압, 이미지의 픽셀, 시간에 따른 주가 등이 그 예입니다. 신호 처리(Signal Processing)는 이러한 양들을 설명하고, 분석하고, 조작하기 위한 엄밀한 수학적 체계를 제공합니다. 신호를 처리하기 전에, 신호를 설명하고 신호를 처리하는 시스템을 기술하기 위한 정밀한 언어가 필요합니다.

이 레슨에서는 기초 어휘를 확립합니다. 신호가 무엇인지, 신호를 어떻게 분류하는지, 기본 구성 요소 신호들이 어떻게 생겼는지, 기본 연산을 통해 신호를 어떻게 변환하는지, 그리고 신호를 처리하는 시스템이 어떤 특성을 가지는지를 배웁니다.

**난이도**: ⭐⭐

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 연속시간(continuous-time) 및 이산시간(discrete-time) 신호를 수학적으로 정의하기
2. 신호를 특성에 따라 분류하기 (결정론적/확률적, 주기적/비주기적, 에너지/전력)
3. 기본 신호 유형 인식 및 생성하기 (임펄스, 계단, 지수, 사인파)
4. 기본 신호 연산 수행하기 (이동, 스케일링, 반전)
5. 시스템의 입출력 모델 정의하기
6. 핵심 시스템 특성 식별 및 검증하기 (선형성, 시불변성, 인과성, 안정성)
7. BIBO 안정성 기준 적용하기

---

## 목차

1. [신호란 무엇인가?](#1-신호란-무엇인가)
2. [연속시간 및 이산시간 신호](#2-연속시간-및-이산시간-신호)
3. [신호 분류](#3-신호-분류)
4. [기본 신호](#4-기본-신호)
5. [에너지 신호와 전력 신호](#5-에너지-신호와-전력-신호)
6. [신호 연산](#6-신호-연산)
7. [시스템이란 무엇인가?](#7-시스템이란-무엇인가)
8. [시스템 특성](#8-시스템-특성)
9. [BIBO 안정성](#9-bibo-안정성)
10. [Python 예제](#10-python-예제)
11. [요약](#11-요약)
12. [연습 문제](#12-연습-문제)
13. [참고 문헌](#13-참고-문헌)

---

## 1. 신호란 무엇인가?

**신호(signal)**란 물리적 현상에 관한 정보를 전달하는 함수입니다. 수학적으로, 신호는 독립 변수(보통 시간 또는 공간)에서 종속 변수(진폭, 전압, 강도 등)로의 사상(mapping)입니다.

### 형식적 정의

**연속시간(CT, Continuous-Time) 신호**는 다음과 같은 함수입니다:

$$x(t) : \mathbb{R} \to \mathbb{R} \quad (\text{or } \mathbb{C})$$

여기서 $t$는 연속 실수 변수로, 일반적으로 시간을 나타냅니다.

**이산시간(DT, Discrete-Time) 신호**는 다음과 같은 수열입니다:

$$x[n] : \mathbb{Z} \to \mathbb{R} \quad (\text{or } \mathbb{C})$$

여기서 $n$은 정수 인덱스입니다.

> **표기 규약**: 이 강좌 전반에서 연속시간 신호에는 괄호 $x(t)$를, 이산시간 신호에는 대괄호 $x[n]$을 사용합니다. 이는 신호 처리 문헌에서 보편적으로 사용되는 규약입니다.

### 신호의 예

| 영역 | 신호 | 독립 변수 | 종속 변수 |
|--------|--------|---------------------|-------------------|
| 오디오 | 음성 파형 | 시간 | 기압 / 전압 |
| 영상 | 사진 | 공간 좌표 $(x, y)$ | 밝기 / 색상 |
| 비디오 | 영화 | 공간 + 시간 $(x, y, t)$ | 밝기 |
| 금융 | 주가 | 시간 (이산: 일) | 가격 |
| 지진학 | 지진계 기록 | 시간 | 지면 변위 |
| 생체의학 | 심전도(ECG) | 시간 | 전압 |
| 통신 | 변조된 반송파 | 시간 | 전자기장 |

---

## 2. 연속시간 및 이산시간 신호

### 2.1 연속시간 신호

연속시간 신호 $x(t)$는 어떤 구간(또는 전체 $\mathbb{R}$)의 모든 $t$ 값에 대해 정의됩니다. 신호의 진폭은 연속 범위 내의 임의의 값을 취할 수 있습니다.

**예시**:
- 아날로그 전압: $x(t) = 3\sin(2\pi \cdot 440 \cdot t)$ (440 Hz 음)
- 지수 감쇠: $x(t) = e^{-t}u(t)$ (여기서 $u(t)$는 단위 계단 함수)
- 가우시안 펄스: $x(t) = e^{-t^2/2\sigma^2}$

### 2.2 이산시간 신호

이산시간 신호 $x[n]$은 정수 값의 $n$에서만 정의됩니다. 연속시간 신호를 표본화(sampling)하면 이산시간 신호가 됩니다:

$$x[n] = x_c(nT_s)$$

여기서 $T_s$는 표본화 주기(sampling period)이고 $f_s = 1/T_s$는 표본화율(sampling rate)입니다.

**예시**:
- 일일 기온 측정값
- 디지털화된 오디오 (CD 음질의 경우 초당 44,100 샘플)
- 영상의 픽셀 행

### 2.3 아날로그 vs. 디지털

시간 축과 진폭 축을 구분하는 것이 중요합니다:

| | 연속 진폭 | 이산 진폭 |
|---|---|---|
| **연속 시간** | 아날로그 신호 | — |
| **이산 시간** | 표본화된 신호 | 디지털 신호 |

**디지털 신호(digital signal)**는 시간에서도 이산적이고 진폭도 양자화된 신호로, 컴퓨터가 실제로 처리하는 신호입니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Continuous-time vs discrete-time signal visualization ---
t = np.linspace(0, 0.01, 1000)      # "continuous" (densely sampled)
f0 = 440                              # 440 Hz (A4 note)
x_ct = np.sin(2 * np.pi * f0 * t)    # continuous-time sinusoid

# Discrete-time version: sampled at 8000 Hz
fs = 8000
n = np.arange(0, int(0.01 * fs))     # sample indices
Ts = 1 / fs
x_dt = np.sin(2 * np.pi * f0 * n * Ts)  # discrete-time sinusoid

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Continuous-time
axes[0].plot(t * 1000, x_ct, 'b-', linewidth=1.5)
axes[0].set_title('Continuous-Time Signal: $x(t) = \\sin(2\\pi \\cdot 440 \\cdot t)$')
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 10])

# Discrete-time
axes[1].stem(n * Ts * 1000, x_dt, linefmt='r-', markerfmt='ro', basefmt='k-')
axes[1].set_title(f'Discrete-Time Signal: $x[n] = \\sin(2\\pi \\cdot 440 \\cdot n / {fs})$')
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Amplitude')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 10])

plt.tight_layout()
plt.savefig('ct_vs_dt_signal.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 3. 신호 분류

신호는 여러 독립적인 축에 따라 분류할 수 있습니다. 이러한 분류를 이해하면 올바른 분석 도구를 선택하는 데 도움이 됩니다.

### 3.1 결정론적 vs. 확률적

| 특성 | 결정론적(Deterministic) | 확률적(Random/Stochastic) |
|----------|--------------|-------------------|
| 정의 | 수학적 표현으로 완전히 기술됨 | 통계적 특성으로 기술됨 |
| 예측 | 미래 값을 정확히 계산 가능 | 미래 값은 확률적으로만 예측 가능 |
| 예시 | $x(t) = A\cos(\omega_0 t + \phi)$ | 열 잡음, 음성 |
| 분석 도구 | 변환 방법 (푸리에, 라플라스) | 상관 함수, 전력 스펙트럼 밀도 |

### 3.2 주기적 vs. 비주기적

연속시간 신호 $x(t)$가 주기 $T > 0$를 가진 **주기적(periodic)** 신호이려면:

$$x(t + T) = x(t) \quad \text{모든 } t \text{에 대해}$$

이를 만족하는 가장 작은 $T$가 **기본 주기(fundamental period)** $T_0$이며, $f_0 = 1/T_0$는 **기본 주파수(fundamental frequency)**입니다.

이산시간 신호 $x[n]$이 주기 $N$ (양의 정수)을 가진 주기적 신호이려면:

$$x[n + N] = x[n] \quad \text{모든 } n \text{에 대해}$$

> **중요**: 연속시간 사인파 $\cos(\omega_0 t)$는 항상 주기적입니다. 이산시간 사인파 $\cos(\omega_0 n)$은 $\omega_0 / (2\pi)$가 유리수인 경우**에만** 주기적입니다.

### 3.3 우함수와 기함수 신호

모든 신호는 우(even) 성분과 기(odd) 성분으로 분해할 수 있습니다:

**우함수 신호(Even signal)**: $x_e(t) = x_e(-t)$ ($t = 0$에 대해 대칭)

**기함수 신호(Odd signal)**: $x_o(t) = -x_o(-t)$ ($t = 0$에 대해 반대칭)

**분해**:

$$x_e(t) = \frac{x(t) + x(-t)}{2}, \qquad x_o(t) = \frac{x(t) - x(-t)}{2}$$

따라서 $x(t) = x_e(t) + x_o(t)$가 성립합니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Even/Odd decomposition ---
t = np.linspace(-3, 3, 1000)

# Original signal: asymmetric exponential
x = np.exp(-t) * (t >= 0)  # right-sided exponential

# Even and odd parts
x_flipped = np.exp(t) * (t <= 0)  # x(-t)
x_even = 0.5 * (x + x_flipped)
x_odd = 0.5 * (x - x_flipped)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(t, x, 'b-', linewidth=2)
axes[0].set_title('Original: $x(t) = e^{-t}u(t)$')
axes[0].set_xlabel('t')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, x_even, 'r-', linewidth=2)
axes[1].set_title('Even part: $x_e(t)$')
axes[1].set_xlabel('t')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, x_odd, 'g-', linewidth=2)
axes[2].set_title('Odd part: $x_o(t)$')
axes[2].set_xlabel('t')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('even_odd_decomposition.png', dpi=150, bbox_inches='tight')
plt.show()

# Verify reconstruction
print("Reconstruction error:", np.max(np.abs(x - (x_even + x_odd))))
```

### 3.4 실수 vs. 복소수 신호

대부분의 물리적 신호는 실수값을 가지지만, 복소수 신호(complex signal)는 신호 처리에서 필수적입니다:

$$z(t) = x(t) + jy(t) = A(t)e^{j\phi(t)}$$

여기서 $A(t) = |z(t)|$는 **포락선(envelope)** (순간 진폭)이고, $\phi(t)$는 **순간 위상(instantaneous phase)**입니다. 복소수 신호는 통신의 기저대역(baseband) 표현, 레이더, 푸리에 분석에서 자연스럽게 등장합니다.

### 3.5 인과적, 역인과적, 비인과적 신호

- **인과적(Causal)**: $t < 0$에서 $x(t) = 0$
- **역인과적(Anticausal)**: $t > 0$에서 $x(t) = 0$
- **비인과적(Noncausal)**: $t < 0$과 $t > 0$ 모두에서 값이 존재함

---

## 4. 기본 신호

여러 기본 신호들이 더 복잡한 신호를 구성하고 분석하기 위한 빌딩 블록 역할을 합니다.

### 4.1 단위 임펄스 (디랙 델타 / 크로네커 델타)

**연속시간** — **디랙 델타 함수(Dirac delta function)** $\delta(t)$:

$$\delta(t) = 0 \text{ for } t \neq 0, \qquad \int_{-\infty}^{\infty} \delta(t) \, dt = 1$$

정의적 특성은 **체질 특성(sifting property)**입니다:

$$\int_{-\infty}^{\infty} x(t)\delta(t - t_0) \, dt = x(t_0)$$

이는 $\delta(t)$가 특정 지점에서 함수의 값을 "골라낸다"는 것을 의미합니다.

**이산시간** — **크로네커 델타(Kronecker delta)** (또는 단위 샘플) $\delta[n]$:

$$\delta[n] = \begin{cases} 1 & n = 0 \\ 0 & n \neq 0 \end{cases}$$

체질 특성:

$$\sum_{k=-\infty}^{\infty} x[k]\delta[n - k] = x[n]$$

### 4.2 단위 계단 함수

**연속시간**:

$$u(t) = \begin{cases} 1 & t > 0 \\ 0 & t < 0 \end{cases}$$

($t = 0$에서의 값은 일반적으로 $1/2$로 정의되지만 실제로는 거의 중요하지 않습니다.)

**이산시간**:

$$u[n] = \begin{cases} 1 & n \geq 0 \\ 0 & n < 0 \end{cases}$$

**관계식**: $u(t) = \int_{-\infty}^{t} \delta(\tau) \, d\tau$이고 $\delta(t) = \frac{du(t)}{dt}$ (분포 의미에서).

이산시간의 경우: $u[n] = \sum_{k=-\infty}^{n} \delta[k]$이고 $\delta[n] = u[n] - u[n-1]$.

### 4.3 직사각형 펄스

$$\text{rect}\left(\frac{t}{\tau}\right) = \begin{cases} 1 & |t| < \tau/2 \\ 1/2 & |t| = \tau/2 \\ 0 & |t| > \tau/2 \end{cases} = u\left(t + \frac{\tau}{2}\right) - u\left(t - \frac{\tau}{2}\right)$$

직사각형 펄스는 표본화, 윈도우잉(windowing), 통신에서 기본적인 역할을 합니다.

### 4.4 실수 지수 함수

**연속시간**: $x(t) = Ce^{at}$ (여기서 $C, a \in \mathbb{R}$)

- $a > 0$: 증가하는 지수 함수
- $a < 0$: 감쇠하는 지수 함수 (예: RC 회로 방전)
- $a = 0$: 상수 $C$

**이산시간**: $x[n] = Ca^n$

- $|a| > 1$: 증가
- $|a| < 1$: 감쇠
- $|a| = 1$: 일정한 크기

### 4.5 사인파 신호

**연속시간**:

$$x(t) = A\cos(\omega_0 t + \phi)$$

여기서:
- $A$ = 진폭(amplitude)
- $\omega_0 = 2\pi f_0$ = 각주파수(angular frequency, rad/s)
- $f_0$ = 주파수(frequency, Hz)
- $T_0 = 1/f_0$ = 주기(period, s)
- $\phi$ = 위상(phase, rad)

**이산시간**:

$$x[n] = A\cos(\omega_0 n + \phi)$$

여기서 $\omega_0$는 디지털 주파수(digital frequency, rad/sample)입니다. 이산시간 사인파가 주기적이 되려면 $\omega_0/(2\pi)$가 유리수여야 합니다.

### 4.6 복소 지수 함수

**복소 지수 함수(complex exponential)**는 신호 처리 전체에서 가장 중요한 신호라고 할 수 있습니다:

**연속시간**:

$$x(t) = Ce^{st}, \quad s = \sigma + j\omega$$

$\sigma = 0$일 때: $x(t) = Ce^{j\omega_0 t} = C[\cos(\omega_0 t) + j\sin(\omega_0 t)]$ (오일러 공식)

**이산시간**:

$$x[n] = Ce^{j\omega_0 n}$$

복소 지수 함수는 LTI 시스템의 **고유 함수(eigenfunctions)**입니다. LTI 시스템에 복소 지수 함수를 입력하면, 출력은 동일한 복소 지수 함수에 복소 상수(시스템의 주파수 응답)가 곱해진 형태입니다. 이 특성이 주파수 영역 분석의 기반입니다.

### 4.7 싱크 함수

$$\text{sinc}(t) = \frac{\sin(\pi t)}{\pi t}$$

로피탈 정리에 의해 $\text{sinc}(0) = 1$입니다. 싱크 함수(sinc function)는 표본화 이론에서 이상적인 보간 커널이며, 직사각형 펄스의 푸리에 변환입니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Fundamental signals gallery ---
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

t = np.linspace(-3, 5, 1000)
n = np.arange(-5, 20)

# 1. Unit step (continuous-time)
axes[0, 0].plot(t, np.where(t >= 0, 1.0, 0.0), 'b-', linewidth=2)
axes[0, 0].set_title('Unit Step $u(t)$')
axes[0, 0].set_xlabel('t')
axes[0, 0].set_ylim([-0.2, 1.4])
axes[0, 0].grid(True, alpha=0.3)

# 2. Unit impulse (discrete-time, approximation)
axes[0, 1].stem(np.arange(-5, 6), np.where(np.arange(-5, 6) == 0, 1.0, 0.0),
                linefmt='r-', markerfmt='ro', basefmt='k-')
axes[0, 1].set_title('Unit Impulse $\\delta[n]$')
axes[0, 1].set_xlabel('n')
axes[0, 1].set_ylim([-0.2, 1.4])
axes[0, 1].grid(True, alpha=0.3)

# 3. Rectangular pulse
tau = 2.0
rect = np.where(np.abs(t) <= tau / 2, 1.0, 0.0)
axes[0, 2].plot(t, rect, 'g-', linewidth=2)
axes[0, 2].set_title(f'Rectangular Pulse rect$(t/{tau:.0f})$')
axes[0, 2].set_xlabel('t')
axes[0, 2].set_ylim([-0.2, 1.4])
axes[0, 2].grid(True, alpha=0.3)

# 4. Decaying exponential
axes[1, 0].plot(t, np.where(t >= 0, np.exp(-t), 0.0), 'b-', linewidth=2)
axes[1, 0].set_title('Decaying Exponential $e^{-t}u(t)$')
axes[1, 0].set_xlabel('t')
axes[1, 0].grid(True, alpha=0.3)

# 5. Growing exponential
axes[1, 1].plot(t[(t >= 0) & (t <= 3)],
                np.exp(0.5 * t[(t >= 0) & (t <= 3)]), 'r-', linewidth=2)
axes[1, 1].set_title('Growing Exponential $e^{0.5t}u(t)$')
axes[1, 1].set_xlabel('t')
axes[1, 1].grid(True, alpha=0.3)

# 6. Sinusoid
axes[1, 2].plot(t, np.cos(2 * np.pi * t), 'g-', linewidth=2)
axes[1, 2].set_title('Sinusoid $\\cos(2\\pi t)$')
axes[1, 2].set_xlabel('t')
axes[1, 2].grid(True, alpha=0.3)

# 7. Complex exponential (real and imaginary parts)
omega0 = 2 * np.pi * 0.5
axes[2, 0].plot(t, np.cos(omega0 * t), 'b-', linewidth=1.5, label='Real')
axes[2, 0].plot(t, np.sin(omega0 * t), 'r--', linewidth=1.5, label='Imag')
axes[2, 0].set_title('Complex Exponential $e^{j\\omega_0 t}$')
axes[2, 0].set_xlabel('t')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# 8. Damped sinusoid
sigma = -0.5
axes[2, 1].plot(t[t >= 0], np.exp(sigma * t[t >= 0]) * np.cos(omega0 * t[t >= 0]),
                'purple', linewidth=2)
axes[2, 1].plot(t[t >= 0], np.exp(sigma * t[t >= 0]), 'k--', linewidth=1, alpha=0.5)
axes[2, 1].plot(t[t >= 0], -np.exp(sigma * t[t >= 0]), 'k--', linewidth=1, alpha=0.5)
axes[2, 1].set_title('Damped Sinusoid $e^{\\sigma t}\\cos(\\omega_0 t)$')
axes[2, 1].set_xlabel('t')
axes[2, 1].grid(True, alpha=0.3)

# 9. Sinc function
t_sinc = np.linspace(-5, 5, 1000)
axes[2, 2].plot(t_sinc, np.sinc(t_sinc), 'm-', linewidth=2)
axes[2, 2].set_title('Sinc Function sinc$(t)$')
axes[2, 2].set_xlabel('t')
axes[2, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fundamental_signals.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 5. 에너지 신호와 전력 신호

에너지와 전력의 개념은 신호의 "크기"에 따라 신호를 분류하는 방법을 제공합니다.

### 5.1 신호 에너지

**연속시간**:

$$E_x = \int_{-\infty}^{\infty} |x(t)|^2 \, dt$$

**이산시간**:

$$E_x = \sum_{n=-\infty}^{\infty} |x[n]|^2$$

### 5.2 신호 전력

**연속시간** (시간 평균 전력):

$$P_x = \lim_{T \to \infty} \frac{1}{2T} \int_{-T}^{T} |x(t)|^2 \, dt$$

**이산시간**:

$$P_x = \lim_{N \to \infty} \frac{1}{2N+1} \sum_{n=-N}^{N} |x[n]|^2$$

### 5.3 분류

| 분류 | 에너지 | 전력 | 예시 |
|-------|--------|-------|----------|
| **에너지 신호** | $0 < E_x < \infty$ | $P_x = 0$ | 펄스, 감쇠 지수 함수 |
| **전력 신호** | $E_x = \infty$ | $0 < P_x < \infty$ | 사인파, 주기 신호, 확률 과정 |
| **해당 없음** | $E_x = \infty$ | $P_x = \infty$ | 증가하는 지수 함수 $e^t$ |

> 어떤 신호도 에너지 신호이면서 동시에 전력 신호가 될 수 없습니다.

### 5.4 예제 계산

**감쇠 지수 함수** $x(t) = e^{-at}u(t)$, $a > 0$:

$$E_x = \int_0^{\infty} e^{-2at} \, dt = \frac{1}{2a}$$

이 값이 유한하므로 **에너지 신호**이며 $P_x = 0$입니다.

**사인파** $x(t) = A\cos(\omega_0 t)$:

$$P_x = \lim_{T \to \infty} \frac{1}{2T} \int_{-T}^{T} A^2 \cos^2(\omega_0 t) \, dt = \frac{A^2}{2}$$

이 신호는 유한한 전력을 가지지만 에너지는 무한하므로 **전력 신호**입니다.

```python
import numpy as np

# --- Energy and power computation ---

# Energy signal: exponential pulse
a = 1.0
t = np.linspace(0, 20, 100000)  # approximate [0, infinity)
dt = t[1] - t[0]
x_energy = np.exp(-a * t)
E_numerical = np.trapz(x_energy**2, t)
E_analytical = 1 / (2 * a)
print(f"Decaying exponential (a={a}):")
print(f"  Energy (numerical):  {E_numerical:.6f}")
print(f"  Energy (analytical): {E_analytical:.6f}")
print(f"  This is an energy signal (finite energy, zero power)")

print()

# Power signal: sinusoid
A = 3.0
f0 = 5.0
T_eval = 100  # average over many periods
t_sin = np.linspace(-T_eval, T_eval, 1000000)
dt_sin = t_sin[1] - t_sin[0]
x_power = A * np.cos(2 * np.pi * f0 * t_sin)
P_numerical = np.trapz(x_power**2, t_sin) / (2 * T_eval)
P_analytical = A**2 / 2
print(f"Sinusoid (A={A}, f0={f0} Hz):")
print(f"  Power (numerical):  {P_numerical:.6f}")
print(f"  Power (analytical): {P_analytical:.6f}")
print(f"  This is a power signal (infinite energy, finite power)")
```

---

## 6. 신호 연산

신호 연산은 신호의 독립 변수 또는 진폭에 적용되는 변환입니다.

### 6.1 시간 이동(Time Shifting)

$$y(t) = x(t - t_0)$$

- $t_0 > 0$: **지연(delay)** (오른쪽 이동)
- $t_0 < 0$: **전진(advance)** (왼쪽 이동)

이산시간: $y[n] = x[n - n_0]$

### 6.2 시간 스케일링(Time Scaling)

$$y(t) = x(at)$$

- $|a| > 1$: **압축(compression)** (신호가 "빨라짐")
- $|a| < 1$: **팽창(expansion)** (신호가 "느려짐")
- $a < 0$: 시간 반전 포함

### 6.3 시간 반전 (반사, Time Reversal)

$$y(t) = x(-t)$$

신호가 $t = 0$을 기준으로 "뒤집힙니다". $a = -1$인 시간 스케일링의 특수한 경우입니다.

### 6.4 진폭 스케일링(Amplitude Scaling)

$$y(t) = Ax(t)$$

진폭을 계수 $A$만큼 스케일링합니다. $A < 0$이면 신호가 반전됩니다.

### 6.5 신호 덧셈과 곱셈

**덧셈**: $z(t) = x(t) + y(t)$ (샘플별 합)

**곱셈**: $z(t) = x(t) \cdot y(t)$ (샘플별 곱; 변조(modulation)/믹싱에 사용됨)

### 6.6 연산 결합

여러 연산이 결합될 때는 순서가 중요합니다. $y(t) = x(at - b)$의 경우:

**방법 1** (먼저 이동, 그 다음 스케일링):
1. $t$를 $t - b/a$로 바꾸어 $x(t - b/a)$ 구하기 ($b/a$만큼 이동)
2. $t$를 $at$로 바꾸어 $x(at - b)$ 구하기 ($a$만큼 스케일링)

**방법 2** (먼저 스케일링, 그 다음 이동):
1. $t$를 $at$로 바꾸어 $x(at)$ 구하기 ($a$만큼 스케일링)
2. $t$를 $t - b/a$로 바꾸되 주의: $at$의 $t$를 바꾸면 $a(t - b/a) = at - b$가 됩니다.

> 가장 안전한 접근 방법은 항상 안에서 밖으로 작업하며, 원래 신호의 각 특징점이 새로운 신호에서 어디로 이동하는지 파악하는 것입니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Signal operations demonstration ---
t = np.linspace(-4, 8, 1000)

# Original: triangular pulse
def tri_pulse(t, width=2.0):
    """Triangular pulse centered at t=0 with given width."""
    return np.maximum(0, 1 - np.abs(t) / width)

x = tri_pulse(t)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Original
axes[0, 0].plot(t, x, 'b-', linewidth=2)
axes[0, 0].set_title('Original: $x(t)$')
axes[0, 0].set_xlabel('t')
axes[0, 0].grid(True, alpha=0.3)

# Time shift (delay by 3)
axes[0, 1].plot(t, tri_pulse(t - 3), 'r-', linewidth=2)
axes[0, 1].plot(t, x, 'b--', linewidth=1, alpha=0.3, label='original')
axes[0, 1].set_title('Time Shift: $x(t - 3)$ (delay by 3)')
axes[0, 1].set_xlabel('t')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Time scaling (compression by 2)
axes[1, 0].plot(t, tri_pulse(2 * t), 'g-', linewidth=2)
axes[1, 0].plot(t, x, 'b--', linewidth=1, alpha=0.3, label='original')
axes[1, 0].set_title('Time Compression: $x(2t)$')
axes[1, 0].set_xlabel('t')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Time scaling (expansion by 0.5)
axes[1, 1].plot(t, tri_pulse(0.5 * t), 'm-', linewidth=2)
axes[1, 1].plot(t, x, 'b--', linewidth=1, alpha=0.3, label='original')
axes[1, 1].set_title('Time Expansion: $x(0.5t)$')
axes[1, 1].set_xlabel('t')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Time reversal
axes[2, 0].plot(t, tri_pulse(-t), 'orange', linewidth=2)
axes[2, 0].plot(t, x, 'b--', linewidth=1, alpha=0.3, label='original')
axes[2, 0].set_title('Time Reversal: $x(-t)$')
axes[2, 0].set_xlabel('t')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# Combined: x(2t - 3)
axes[2, 1].plot(t, tri_pulse(2 * t - 3), 'cyan', linewidth=2)
axes[2, 1].plot(t, x, 'b--', linewidth=1, alpha=0.3, label='original')
axes[2, 1].set_title('Combined: $x(2t - 3)$ (compress then shift)')
axes[2, 1].set_xlabel('t')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('signal_operations.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 7. 시스템이란 무엇인가?

**시스템(system)**은 입력 신호를 출력 신호로 변환하는 과정의 수학적 모델입니다.

$$x(t) \xrightarrow{\quad \mathcal{T}\{\cdot\} \quad} y(t) = \mathcal{T}\{x(t)\}$$

또는 이산시간에서:

$$x[n] \xrightarrow{\quad \mathcal{T}\{\cdot\} \quad} y[n] = \mathcal{T}\{x[n]\}$$

### 시스템의 예

| 시스템 | 입력 | 출력 | 영역 |
|--------|-------|--------|--------|
| 증폭기 | 전압 $x(t)$ | 증폭된 전압 $Ax(t)$ | 전자공학 |
| 이동 평균 | $x[n]$ | $\frac{1}{M}\sum_{k=0}^{M-1}x[n-k]$ | 신호 평활화 |
| 미분기 | $x(t)$ | $dx(t)/dt$ | 연속시간 |
| 누산기 | $x[n]$ | $\sum_{k=-\infty}^{n}x[k]$ | 이산시간 |
| 에코 시스템 | $x[n]$ | $x[n] + \alpha x[n-D]$ | 오디오 |
| 변조기 | $x(t)$ | $x(t)\cos(\omega_c t)$ | 통신 |

### 시스템 상호연결

시스템은 세 가지 기본 구성으로 결합할 수 있습니다:

**직렬(Cascade)**: $y = \mathcal{T}_2\{\mathcal{T}_1\{x\}\}$

```
x → [T₁] → [T₂] → y
```

**병렬(Parallel)**: $y = \mathcal{T}_1\{x\} + \mathcal{T}_2\{x\}$

```
     ┌─[T₁]─┐
x ──►│       ├──► (+) → y
     └─[T₂]─┘
```

**피드백(Feedback)**: 출력이 입력으로 되먹임

```
x ──►(+)──► [T₁] ──► y
      ▲               │
      └───── [T₂] ◄───┘
```

---

## 8. 시스템 특성

시스템 특성은 어떤 분석 도구를 사용할 수 있는지를 결정합니다. 신호 처리에서 가장 중요한 특성들을 아래에 설명합니다.

### 8.1 메모리 / 무기억성

시스템이 **무기억(memoryless)**이라면, 임의의 시간에서 출력이 그 순간의 입력에만 의존합니다:

$$y(t) = f(x(t)) \quad \text{(과거나 미래에 대한 의존 없음)}$$

**무기억 예시**: $y(t) = 3x(t)$, $y[n] = x^2[n]$

**기억 있음**: $y[n] = x[n] + x[n-1]$ (과거에 의존), $y(t) = \int_{-\infty}^{t} x(\tau) d\tau$

### 8.2 선형성(Linearity)

시스템이 **선형(linear)**이라면 **중첩의 원리(superposition)**를 만족합니다. 임의의 신호 $x_1, x_2$와 스칼라 $a, b$에 대해:

$$\mathcal{T}\{ax_1 + bx_2\} = a\mathcal{T}\{x_1\} + b\mathcal{T}\{x_2\}$$

이는 두 가지 부분 특성을 결합합니다:
- **가산성(Additivity)**: $\mathcal{T}\{x_1 + x_2\} = \mathcal{T}\{x_1\} + \mathcal{T}\{x_2\}$
- **균질성(Homogeneity, 스케일링)**: $\mathcal{T}\{ax\} = a\mathcal{T}\{x\}$

> 선형성의 결과: $\mathcal{T}\{0\} = 0$. 시스템이 영(zero) 입력에 대해 비영(nonzero) 출력을 생성하면 비선형입니다.

**선형**: $y(t) = 3x(t) + 2\frac{dx}{dt}$, 이동 평균

**비선형**: $y(t) = x^2(t)$, $y[n] = \log(x[n])$, $y(t) = x(t) + 1$ ($\mathcal{T}\{0\} = 0$ 위반)

### 8.3 시불변성(Time Invariance)

시스템이 **시불변(time-invariant)** (또는 이동 불변)이라면, 입력에 시간 이동을 적용하면 출력에도 동일한 시간 이동이 발생합니다:

$$\text{만약 } \mathcal{T}\{x(t)\} = y(t) \text{이면, } \mathcal{T}\{x(t - t_0)\} = y(t - t_0)$$

즉, 시스템은 입력이 언제 인가되는지에 관계없이 동일하게 작동합니다.

**시불변**: $y(t) = x(t-1)$, $y[n] = x[n] - x[n-1]$

**시변**: $y(t) = x(2t)$ (시간 스케일링), $y(t) = (\cos t) \cdot x(t)$ (시변 계수)

### 8.4 인과성(Causality)

시스템이 **인과적(causal)**이라면, 임의의 시간에서 출력이 현재 및 과거 입력에만 의존합니다:

$$y(t_0) \text{는 오직 } \{x(t) : t \leq t_0\} \text{에만 의존}$$

**인과적**: $y[n] = x[n] + x[n-1]$, $y(t) = \int_{-\infty}^{t} x(\tau)d\tau$

**비인과적**: $y[n] = x[n+1]$ (미래에 의존), 이상적인 저역통과 필터

> 물리적으로 실현 가능한 모든 실시간 시스템은 반드시 인과적이어야 합니다. 그러나 비인과적 시스템은 전체 신호가 사용 가능한 오프라인(배치) 처리에서 유용합니다.

### 8.5 안정성(Stability, 비형식적)

시스템이 **안정(stable)**이라면 유계(bounded) 입력은 유계 출력을 생성합니다. 이를 다음 절에서 **BIBO 안정성**으로 형식화합니다.

**안정**: $y(t) = e^{-t}x(t)$ ($t \geq 0$에 대해)

**불안정**: 상수 입력에 대한 $y(t) = \int_{0}^{t} x(\tau)d\tau$ (출력이 무한히 증가)

### 8.6 가역성(Invertibility)

시스템이 **가역적(invertible)**이라면 서로 다른 입력이 서로 다른 출력을 생성하며, 역 시스템 $\mathcal{T}^{-1}$이 존재합니다:

$$\mathcal{T}^{-1}\{\mathcal{T}\{x\}\} = x$$

**가역적**: $y(t) = 2x(t)$ (역 시스템: $x(t) = y(t)/2$)

**비가역적**: $y(t) = x^2(t)$ ($x$의 부호를 복원할 수 없음)

```python
import numpy as np

# --- Testing system properties ---

# Test linearity of a system
def system_linear(x):
    """Linear system: y[n] = 2*x[n] + x[n-1]"""
    y = np.zeros_like(x)
    y[0] = 2 * x[0]
    for i in range(1, len(x)):
        y[i] = 2 * x[i] + x[i - 1]
    return y

def system_nonlinear(x):
    """Nonlinear system: y[n] = x[n]^2"""
    return x ** 2

def test_linearity(system, name):
    """Test linearity using superposition."""
    np.random.seed(42)
    x1 = np.random.randn(20)
    x2 = np.random.randn(20)
    a, b = 3.0, -2.0

    lhs = system(a * x1 + b * x2)           # T{a*x1 + b*x2}
    rhs = a * system(x1) + b * system(x2)   # a*T{x1} + b*T{x2}

    error = np.max(np.abs(lhs - rhs))
    is_linear = error < 1e-10
    print(f"{name}: max superposition error = {error:.2e} -> {'LINEAR' if is_linear else 'NONLINEAR'}")

test_linearity(system_linear, "y[n] = 2x[n] + x[n-1]")
test_linearity(system_nonlinear, "y[n] = x[n]^2")

print()

# Test time invariance
def test_time_invariance(system, name, delay=3):
    """Test time invariance by comparing shifted input vs shifted output."""
    np.random.seed(42)
    x = np.random.randn(30)

    # Method 1: shift input, then apply system
    x_delayed = np.zeros_like(x)
    x_delayed[delay:] = x[:-delay]
    y1 = system(x_delayed)

    # Method 2: apply system, then shift output
    y = system(x)
    y2 = np.zeros_like(y)
    y2[delay:] = y[:-delay]

    error = np.max(np.abs(y1 - y2))
    is_ti = error < 1e-10
    print(f"{name}: max TI error = {error:.2e} -> {'TIME-INVARIANT' if is_ti else 'TIME-VARYING'}")

def system_tv(x):
    """Time-varying system: y[n] = n * x[n]"""
    n = np.arange(len(x), dtype=float)
    return n * x

test_time_invariance(system_linear, "y[n] = 2x[n] + x[n-1]")
test_time_invariance(system_tv, "y[n] = n * x[n]")
```

---

## 9. BIBO 안정성

### 9.1 정의

시스템이 **유계 입력 유계 출력(BIBO, Bounded-Input Bounded-Output) 안정**이라면, 모든 유계 입력이 유계 출력을 생성합니다:

$$|x(t)| \leq M_x < \infty \quad \Rightarrow \quad |y(t)| \leq M_y < \infty$$

어떤 유한한 $M_y$에 대해 위 조건이 성립합니다.

### 9.2 LTI 시스템의 BIBO 안정성

임펄스 응답 $h(t)$를 가진 LTI 시스템의 BIBO 안정성은 임펄스 응답의 **절대 적분 가능성(absolute integrability)**과 동치입니다:

**연속시간**:

$$\int_{-\infty}^{\infty} |h(t)| \, dt < \infty$$

**이산시간**:

$$\sum_{n=-\infty}^{\infty} |h[n]| < \infty$$

**증명 개요** (이산시간): $|x[n]| \leq M_x$이면,

$$|y[n]| = \left|\sum_k h[k] x[n-k]\right| \leq \sum_k |h[k]| \cdot |x[n-k]| \leq M_x \sum_k |h[k]|$$

따라서 $|y[n]| \leq M_x \cdot \sum_k |h[k]|$. 합이 수렴하면 출력은 유계입니다.

### 9.3 예시

**안정**: $h[n] = (0.5)^n u[n]$

$$\sum_{n=0}^{\infty} |0.5^n| = \sum_{n=0}^{\infty} 0.5^n = \frac{1}{1 - 0.5} = 2 < \infty \quad \checkmark$$

**불안정**: $h[n] = u[n]$ (단위 계단, 즉 누산기/적분기)

$$\sum_{n=0}^{\infty} |1| = \infty \quad \times$$

**경계적 안정(Marginally stable)**: $h[n] = \cos(\omega_0 n) u[n]$ (감쇠 없이 진동)

$$\sum_{n=0}^{\infty} |\cos(\omega_0 n)| = \infty \quad \times \text{ (BIBO 불안정)}$$

```python
import numpy as np

# --- BIBO stability check ---

def check_bibo_stability(h, name, max_terms=10000):
    """Check BIBO stability by computing sum of |h[n]|."""
    abs_sum = np.sum(np.abs(h[:max_terms]))
    # Check if the partial sum is still growing significantly
    abs_sum_half = np.sum(np.abs(h[:max_terms // 2]))
    converging = abs(abs_sum - abs_sum_half) / max(abs_sum, 1e-15) < 0.01
    print(f"{name}:")
    print(f"  Sum |h[n]| (N={max_terms}): {abs_sum:.4f}")
    print(f"  Convergent: {converging}")
    print(f"  BIBO Stable: {'Yes' if converging and abs_sum < 1e6 else 'No'}")
    print()

n = np.arange(10000)

# Stable: decaying exponential
h_stable = 0.5**n
check_bibo_stability(h_stable, "h[n] = 0.5^n * u[n]")

# Unstable: unit step (accumulator)
h_unstable = np.ones(10000)
check_bibo_stability(h_unstable, "h[n] = u[n] (accumulator)")

# Stable: finite impulse response
h_fir = np.array([0.25, 0.5, 0.25])
h_fir_padded = np.zeros(10000)
h_fir_padded[:3] = h_fir
check_bibo_stability(h_fir_padded, "h[n] = [0.25, 0.5, 0.25] (FIR)")
```

---

## 10. Python 예제

### 10.1 일반 신호 생성 및 분석

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

# --- Comprehensive signal generation toolkit ---

class SignalGenerator:
    """Generate common signals for analysis."""

    def __init__(self, duration=1.0, fs=1000):
        self.fs = fs
        self.duration = duration
        self.t = np.arange(0, duration, 1 / fs)
        self.N = len(self.t)

    def sinusoid(self, freq, amp=1.0, phase=0.0):
        return amp * np.cos(2 * np.pi * freq * self.t + phase)

    def square_wave(self, freq, amp=1.0, duty=0.5):
        return amp * sig.square(2 * np.pi * freq * self.t, duty=duty)

    def sawtooth(self, freq, amp=1.0):
        return amp * sig.sawtooth(2 * np.pi * freq * self.t)

    def gaussian_pulse(self, center, width):
        return np.exp(-((self.t - center) ** 2) / (2 * width ** 2))

    def chirp(self, f0, f1, method='linear'):
        return sig.chirp(self.t, f0, self.duration, f1, method=method)

    def white_noise(self, power=1.0):
        return np.sqrt(power) * np.random.randn(self.N)

    def energy(self, x):
        """Compute signal energy."""
        return np.trapz(np.abs(x) ** 2, self.t)

    def power(self, x):
        """Compute signal average power."""
        return np.trapz(np.abs(x) ** 2, self.t) / self.duration


# Demonstrate
gen = SignalGenerator(duration=0.1, fs=8000)

signals = {
    '5 Hz Sinusoid': gen.sinusoid(5),
    '5 Hz Square': gen.square_wave(5),
    '5 Hz Sawtooth': gen.sawtooth(5),
    'Gaussian Pulse': gen.gaussian_pulse(0.05, 0.01),
    'Chirp (10-100 Hz)': gen.chirp(10, 100),
}

fig, axes = plt.subplots(len(signals), 1, figsize=(12, 2.5 * len(signals)))
for ax, (name, x) in zip(axes, signals.items()):
    ax.plot(gen.t * 1000, x, linewidth=1.5)
    E = gen.energy(x)
    P = gen.power(x)
    ax.set_title(f'{name}  |  Energy={E:.4f}, Power={P:.4f}')
    ax.set_xlabel('Time (ms)')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('signal_gallery.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 10.2 이산시간 신호 주기성 검사

```python
import numpy as np
from fractions import Fraction

def check_dt_periodicity(omega0):
    """
    Check if discrete-time sinusoid cos(omega0 * n) is periodic.
    Periodic if and only if omega0 / (2*pi) is rational.
    """
    ratio = omega0 / (2 * np.pi)

    # Use Fraction for exact rational approximation
    frac = Fraction(ratio).limit_denominator(1000)

    # Check if the approximation is close enough to be exact
    if abs(float(frac) - ratio) < 1e-10:
        N = frac.denominator  # fundamental period
        print(f"omega0 = {omega0:.6f}")
        print(f"  omega0/(2pi) = {ratio:.6f} = {frac}")
        print(f"  PERIODIC with fundamental period N = {N}")
        return N
    else:
        print(f"omega0 = {omega0:.6f}")
        print(f"  omega0/(2pi) = {ratio:.6f} (irrational)")
        print(f"  NOT PERIODIC")
        return None

# Test cases
print("=== Discrete-Time Periodicity Test ===\n")
check_dt_periodicity(np.pi / 4)       # pi/4 -> 1/8 -> periodic, N=8
print()
check_dt_periodicity(2 * np.pi / 3)   # 2pi/3 -> 1/3 -> periodic, N=3
print()
check_dt_periodicity(1.0)             # 1/(2pi) is irrational -> not periodic
print()
check_dt_periodicity(np.pi)           # pi/(2pi) = 1/2 -> periodic, N=2
```

### 10.3 시스템 특성 검증 모음

```python
import numpy as np

def verify_system_properties(system, name, N=50, delay=5):
    """Comprehensive system property verification."""
    print(f"=== System: {name} ===")
    np.random.seed(42)

    x1 = np.random.randn(N)
    x2 = np.random.randn(N)
    a, b = 2.5, -1.3

    # 1. Linearity
    lhs = system(a * x1 + b * x2)
    rhs = a * system(x1) + b * system(x2)
    lin_err = np.max(np.abs(lhs - rhs))
    print(f"  Linearity error:       {lin_err:.2e} -> {'LINEAR' if lin_err < 1e-10 else 'NONLINEAR'}")

    # 2. Time Invariance
    x_del = np.zeros(N)
    x_del[delay:] = x1[:N - delay]
    y_shifted_input = system(x_del)
    y_then_shift = np.zeros(N)
    y = system(x1)
    y_then_shift[delay:] = y[:N - delay]
    ti_err = np.max(np.abs(y_shifted_input - y_then_shift))
    print(f"  Time-invariance error: {ti_err:.2e} -> {'TIME-INVARIANT' if ti_err < 1e-10 else 'TIME-VARYING'}")

    # 3. Causality (check if output at n depends on future inputs)
    # Modify future of x1 and check if current output changes
    x_mod = x1.copy()
    x_mod[N // 2:] = np.random.randn(N - N // 2)  # change future
    y_orig = system(x1)
    y_mod = system(x_mod)
    causal_err = np.max(np.abs(y_orig[:N // 2] - y_mod[:N // 2]))
    print(f"  Causality error:       {causal_err:.2e} -> {'CAUSAL' if causal_err < 1e-10 else 'NONCAUSAL'}")

    # 4. Memory
    # Check if output depends on anything other than current input
    memory = False
    x_test = np.zeros(N)
    x_test[N // 2] = 1.0
    y_test = system(x_test)
    if np.sum(np.abs(y_test) > 1e-10) > 1:
        memory = True
    print(f"  Memory:                {'WITH MEMORY' if memory else 'MEMORYLESS'}")
    print()

# Test various systems
def sys_gain(x):
    return 3 * x

def sys_delay(x):
    y = np.zeros_like(x)
    y[1:] = x[:-1]
    return y

def sys_ma3(x):
    """3-point moving average"""
    y = np.zeros_like(x)
    for i in range(len(x)):
        total = x[i]
        count = 1
        if i >= 1:
            total += x[i - 1]
            count += 1
        if i >= 2:
            total += x[i - 2]
            count += 1
        y[i] = total / 3
    return y

def sys_square(x):
    return x ** 2

def sys_tv_gain(x):
    n = np.arange(len(x), dtype=float)
    return n * x

verify_system_properties(sys_gain, "y[n] = 3x[n] (gain)")
verify_system_properties(sys_delay, "y[n] = x[n-1] (unit delay)")
verify_system_properties(sys_ma3, "y[n] = (x[n]+x[n-1]+x[n-2])/3 (MA)")
verify_system_properties(sys_square, "y[n] = x[n]^2 (squarer)")
verify_system_properties(sys_tv_gain, "y[n] = n*x[n] (time-varying)")
```

---

## 11. 요약

### 핵심 개념

| 개념 | 정의 | 중요한 이유 |
|---------|-----------|---------------|
| 신호 | 정보를 전달하는 함수 | 학습의 근본 대상 |
| CT vs DT | $x(t)$ vs $x[n]$ | 적용할 수학적 도구 결정 |
| 에너지 신호 | $0 < E < \infty$, $P = 0$ | 과도 신호 (펄스) |
| 전력 신호 | $E = \infty$, $0 < P < \infty$ | 지속 신호 (사인파) |
| 시스템 | 입출력 사상 $\mathcal{T}$ | 신호를 처리함 |
| 선형성 | 중첩의 원리 성립 | 분해 방법 가능 |
| 시불변성 | 시간 이동에 의해 거동이 불변 | 시스템 파라미터가 일정 |
| 인과성 | 출력이 과거/현재에만 의존 | 물리적 실현 가능성 |
| BIBO 안정성 | 유계 입력 $\Rightarrow$ 유계 출력 | 시스템이 "폭발"하지 않음 |
| LTI 시스템 | 선형 + 시불변 | 신호 처리의 기반 |

### 신호 처리 계층

```
                    신호와 시스템 (이 레슨)
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
         LTI 시스템    푸리에     Z-변환
        과 합성곱      분석
                │           │           │
                └───────────┼───────────┘
                            ▼
                    디지털 필터
                    & 응용
```

LTI 시스템의 개념은 절대적으로 핵심적입니다. 선형성 덕분에 입력을 기본 요소(복소 지수 함수 등)로 분해하여 각 요소를 별도로 분석한 후 결과를 합산할 수 있습니다. 시불변성은 시스템이 입력이 언제 도착하든 동일하게 응답하도록 보장합니다. 이 두 특성이 결합되어 신호 처리의 근간을 이루는 강력한 변환 영역 방법을 가능하게 합니다.

---

## 12. 연습 문제

### 연습 1: 신호 분류

각 신호를 (a) 연속시간 또는 이산시간, (b) 주기적 또는 비주기적, (c) 에너지 또는 전력 신호, (d) 인과적 또는 비인과적으로 분류하세요.

1. $x(t) = 3\cos(100\pi t + \pi/3)$
2. $x[n] = (-0.5)^n u[n]$
3. $x(t) = e^{-2|t|}$
4. $x[n] = \cos(0.3\pi n)$
5. $x(t) = u(t) - u(t-5)$
6. $x[n] = 2^n u[-n]$

### 연습 2: 우함수-기함수 분해

다음 신호를 우함수 성분과 기함수 성분으로 분해하세요. 세 가지 모두 그래프로 나타내세요.

$$x(t) = \begin{cases} t + 1 & 0 \leq t \leq 1 \\ 2 & 1 < t \leq 2 \\ 0 & \text{그 외} \end{cases}$$

### 연습 3: 에너지 계산

다음 신호들의 에너지를 해석적으로 계산하고, Python으로 수치적으로 검증하세요:

1. $x(t) = 2e^{-3t}u(t)$
2. $x[n] = (0.8)^n u[n]$
3. $x(t) = \text{rect}(t/4)$ (폭이 4인 직사각형 펄스)

### 연습 4: 시스템 특성

각 시스템에 대해 (i) 선형, (ii) 시불변, (iii) 인과적, (iv) 안정, (v) 무기억 여부를 판단하세요. 근거를 제시하세요.

1. $y(t) = x(t-2)$
2. $y[n] = nx[n]$
3. $y(t) = \cos(x(t))$
4. $y[n] = x[-n]$
5. $y(t) = x(t)\cos(2\pi f_0 t)$
6. $y[n] = \sum_{k=n-2}^{n+2} x[k]$ (5점 중심 평균)
7. $y(t) = x(t) + 3$

### 연습 5: BIBO 안정성

다음 임펄스 응답을 가진 LTI 시스템이 BIBO 안정한지 판단하세요:

1. $h(t) = e^{-2t}u(t)$
2. $h[n] = u[n]$
3. $h[n] = (0.9)^{|n|}$
4. $h(t) = \frac{\sin(t)}{t}$
5. $h[n] = \delta[n] - 0.5\delta[n-1]$

### 연습 6: Python 구현

이산시간 신호 `x`와 표본화율 `fs`를 받아 다음을 포함하는 딕셔너리를 반환하는 Python 함수 `signal_analyzer(x, fs)`를 작성하세요:
- 신호 지속 시간
- 최대 절댓값
- 에너지
- 평균 전력
- 신호가 근사적으로 주기적인지 여부 (자기상관 함수를 이용하여 검출)
- 추정된 기본 주파수 (주기적인 경우)

사인파, 처프(chirp) 신호, 백색 잡음으로 테스트하세요.

### 연습 7: 신호 연산 심화

$x(t)$를 $t = 0$에서 피크 진폭이 1이고 지지(support)가 $[-1, 1]$인 삼각형 펄스라 할 때:

1. $y_1(t) = x(2t - 3)$의 스케치를 그리고 Python 함수를 작성하세요.
2. $y_2(t) = x(-t + 1) + x(t - 1)$의 스케치를 그리고 Python 함수를 작성하세요.
3. $x(t)$, $y_1(t)$, $y_2(t)$의 에너지를 계산하세요. 에너지들 사이의 관계를 설명하세요.

### 연습 8: 복소 지수 함수와 페이저

1. $x(t) = 3\cos(10t + \pi/4) + 4\sin(10t - \pi/6)$을 단일 사인파 $A\cos(10t + \phi)$로 표현하세요. $A$와 $\phi$를 구하세요.
2. 페이저 다이어그램을 그리세요.
3. 복소 지수 함수를 이용하여 $x(t) = \text{Re}\{Ce^{j10t}\}$로 표현하세요. $C$를 구하세요.

---

## 13. 참고 문헌

1. Oppenheim, A. V. & Willsky, A. S. *Signals and Systems* (2nd ed.), Ch. 1. Prentice Hall, 1997.
2. Haykin, S. & Van Veen, B. *Signals and Systems* (2nd ed.), Ch. 1-2. Wiley, 2003.
3. Lathi, B. P. & Green, R. A. *Linear Systems and Signals* (3rd ed.), Ch. 1. Oxford University Press, 2018.

---

[다음: 02. LTI 시스템과 합성곱](./02_LTI_Systems_and_Convolution.md) | [개요](./00_Overview.md)
