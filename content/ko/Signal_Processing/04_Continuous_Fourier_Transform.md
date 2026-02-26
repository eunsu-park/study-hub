# 연속 푸리에 변환(Continuous Fourier Transform)

**이전**: [03. 푸리에 급수와 응용](./03_Fourier_Series_and_Applications.md) | **다음**: [05. 표본화와 복원](./05_Sampling_and_Reconstruction.md)

---

푸리에 급수는 주기 신호를 조화 관련 사인파들로 분해합니다. 그러나 대부분의 실제 신호들 — 음성, 레이더 펄스, 과도 진동 — 은 **비주기적**입니다. 이러한 신호를 주파수 영역에서 분석하려면 푸리에 급수를 비주기 신호로 확장해야 합니다. 그 결과가 바로 **연속시간 푸리에 변환(Continuous-Time Fourier Transform, CTFT)**이며, 이는 신호를 시간 영역에서 연속 주파수 영역 표현으로 변환합니다.

푸리에 변환은 과학과 공학 전반에 걸쳐 가장 강력하고 보편적인 도구 중 하나입니다. 이 레슨에서는 변환을 기본 원리로부터 유도하고, 핵심 성질들을 목록화하며, 일반적인 변환 쌍(transform pair) 표를 작성하고, 이를 통해 주파수 영역 분석과 필터링이 어떻게 가능한지 보여줍니다.

**난이도**: ⭐⭐⭐

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 비주기 신호에 대한 푸리에 급수의 극한으로서 푸리에 변환 유도
2. 주요 CTFT 성질 (선형성, 이동, 스케일링, 쌍대성, 합성곱) 증명
3. 표준 신호의 순방향 및 역 푸리에 변환 계산
4. 합성곱 정리를 적용하여 LTI 시스템 분석 단순화
5. 주파수 영역 표현을 물리적으로 해석
6. 이상적인 주파수 선택 필터 (저역통과, 고역통과, 대역통과) 설명
7. 대역폭과 시간-대역폭 곱 (불확정성 원리) 이해
8. Python (FFT)을 이용한 연속 신호 푸리에 분석

---

## 목차

1. [푸리에 급수에서 푸리에 변환으로](#1-푸리에-급수에서-푸리에-변환으로)
2. [푸리에 변환의 정의](#2-푸리에-변환의-정의)
3. [존재 조건](#3-존재-조건)
4. [일반적인 변환 쌍](#4-일반적인-변환-쌍)
5. [푸리에 변환의 성질](#5-푸리에-변환의-성질)
6. [합성곱 정리](#6-합성곱-정리)
7. [곱셈 정리(윈도잉)](#7-곱셈-정리윈도잉)
8. [파르스발 정리](#8-파르스발-정리)
9. [주파수 영역 분석](#9-주파수-영역-분석)
10. [이상적인 필터](#10-이상적인-필터)
11. [대역폭과 시간-대역폭 곱](#11-대역폭과-시간-대역폭-곱)
12. [Python 예제](#12-python-예제)
13. [요약](#13-요약)
14. [연습 문제](#14-연습-문제)
15. [참고 문헌](#15-참고-문헌)

---

## 1. 푸리에 급수에서 푸리에 변환으로

### 1.1 극한 과정

주기 $T_0$을 가지며 한 주기 구간에서 비주기 신호 $x(t)$와 같은 주기 신호 $\tilde{x}(t)$를 고려합니다:

$$\tilde{x}(t) = x(t) \text{ for } -T_0/2 < t < T_0/2$$

복소 푸리에 급수는 다음과 같습니다:

$$\tilde{x}(t) = \sum_{n=-\infty}^{\infty} c_n e^{jn\omega_0 t}, \quad c_n = \frac{1}{T_0} \int_{-T_0/2}^{T_0/2} x(t) e^{-jn\omega_0 t} \, dt$$

**포락선 함수(envelope function)**를 정의합니다:

$$X(\omega) \equiv T_0 \cdot c_n \bigg|_{\omega = n\omega_0} = \int_{-T_0/2}^{T_0/2} x(t) e^{-j\omega t} \, dt$$

이제 $T_0 \to \infty$로 취하면, 조화파 간격 $\omega_0 = 2\pi/T_0 \to 0$이 되고, 이산 주파수 $n\omega_0$는 연속 변수 $\omega$가 되며, 합은 적분이 됩니다:

$$\sum_n c_n e^{jn\omega_0 t} = \sum_n \frac{X(n\omega_0)}{T_0} e^{jn\omega_0 t} = \frac{1}{2\pi} \sum_n X(n\omega_0) e^{jn\omega_0 t} \omega_0$$

$\omega_0 \to d\omega$로 가면:

$$x(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} X(\omega) e^{j\omega t} \, d\omega$$

그리고 $X(\omega)$의 적분 한계는 무한대로 확장됩니다:

$$X(\omega) = \int_{-\infty}^{\infty} x(t) e^{-j\omega t} \, dt$$

이 두 방정식의 쌍이 푸리에 변환을 정의합니다.

### 1.2 물리적 해석

| 푸리에 급수 | 푸리에 변환 |
|---------------|-------------------|
| 주기 신호 | 비주기 신호 |
| 이산 스펙트럼 (선 스펙트럼) | 연속 스펙트럼 (스펙트럼 밀도) |
| 계수 $c_n$ (무차원) | $X(\omega)$는 [신호 $\times$ 시간]의 단위 |
| $n\omega_0$에서만 주파수 존재 | 모든 주파수 $\omega \in \mathbb{R}$ |

---

## 2. 푸리에 변환의 정의

### 2.1 순방향 변환 (분석)

$$X(\omega) = \mathcal{F}\{x(t)\} = \int_{-\infty}^{\infty} x(t) \, e^{-j\omega t} \, dt$$

$X(\omega)$를 $x(t)$의 **푸리에 변환**, **스펙트럼**, 또는 **주파수 영역 표현**이라고 합니다.

### 2.2 역변환 (합성)

$$x(t) = \mathcal{F}^{-1}\{X(\omega)\} = \frac{1}{2\pi} \int_{-\infty}^{\infty} X(\omega) \, e^{j\omega t} \, d\omega$$

### 2.3 대안적 표기 규약

문헌에서 세 가지 규약이 일반적으로 사용됩니다:

| 규약 | 순방향 | 역방향 | 사용처 |
|-----------|---------|---------|---------|
| $\omega$ (각주파수) | $\int x \, e^{-j\omega t} dt$ | $\frac{1}{2\pi}\int X \, e^{j\omega t} d\omega$ | Oppenheim, Haykin (이 강좌) |
| $f$ (일반 주파수) | $\int x \, e^{-j2\pi ft} dt$ | $\int X \, e^{j2\pi ft} df$ | Bracewell, 공학 실무 |
| 대칭형 | $\frac{1}{\sqrt{2\pi}}\int x \, e^{-j\omega t} dt$ | $\frac{1}{\sqrt{2\pi}}\int X \, e^{j\omega t} d\omega$ | 물리학, 양자역학 |

`scipy.fft`가 사용하는 $f$-규약은 다음과 같습니다:

$$X(f) = \int_{-\infty}^{\infty} x(t) \, e^{-j2\pi ft} \, dt, \qquad x(t) = \int_{-\infty}^{\infty} X(f) \, e^{j2\pi ft} \, df$$

변환: $X(\omega) = X(f)|_{f=\omega/(2\pi)}$ ($\omega = 2\pi f$ 관계 사용).

### 2.4 크기와 위상

$X(\omega)$는 일반적으로 복소수이므로:

$$X(\omega) = |X(\omega)| e^{j\angle X(\omega)}$$

- $|X(\omega)|$ = **크기 스펙트럼** (진폭 스펙트럼 밀도)
- $\angle X(\omega)$ = **위상 스펙트럼**
- $|X(\omega)|^2$ = **에너지 스펙트럼 밀도(ESD)**

실수 신호에 대해: $X(-\omega) = X^*(\omega)$ (**켤레 대칭**)이므로:
- $|X(-\omega)| = |X(\omega)|$ (크기는 우함수)
- $\angle X(-\omega) = -\angle X(\omega)$ (위상은 기함수)

---

## 3. 존재 조건

### 3.1 충분 조건

$x(t)$가 **절대 적분 가능(absolutely integrable)**하면 푸리에 변환이 존재합니다:

$$\int_{-\infty}^{\infty} |x(t)| \, dt < \infty$$

이는 $|X(\omega)| \leq \int |x(t)| \, dt < \infty$ (변환이 유계)를 보장합니다.

### 3.2 더 넓은 존재 조건

많은 중요한 신호들 ($\sin(\omega_0 t)$, $u(t)$, $\delta(t)$ 등)은 절대 적분 가능하지 않지만, **분포(distributions)** (일반화 함수)를 사용하여 일반화된 의미에서 여전히 푸리에 변환을 가집니다. 이들의 변환은 디랙 델타 함수를 포함합니다:

$$\mathcal{F}\{\delta(t)\} = 1, \qquad \mathcal{F}\{1\} = 2\pi\delta(\omega)$$

$$\mathcal{F}\{e^{j\omega_0 t}\} = 2\pi\delta(\omega - \omega_0)$$

이러한 일반화 변환은 주기 신호와 상수를 푸리에 변환 체계 내에서 다루는 데 필수적입니다.

---

## 4. 일반적인 변환 쌍

### 4.1 푸리에 변환 쌍 표

| 신호 $x(t)$ | 변환 $X(\omega)$ | 비고 |
|:---:|:---:|:---|
| $\delta(t)$ | $1$ | 임펄스는 모든 주파수를 동등하게 포함 |
| $1$ | $2\pi\delta(\omega)$ | 상수(DC)는 $\omega = 0$에서 단일 주파수 |
| $e^{j\omega_0 t}$ | $2\pi\delta(\omega - \omega_0)$ | 복소 지수는 단일 스펙트럼 선 |
| $\cos(\omega_0 t)$ | $\pi[\delta(\omega - \omega_0) + \delta(\omega + \omega_0)]$ | 두 개의 스펙트럼 선 |
| $\sin(\omega_0 t)$ | $\frac{\pi}{j}[\delta(\omega - \omega_0) - \delta(\omega + \omega_0)]$ | 두 개의 스펙트럼 선 |
| $u(t)$ | $\pi\delta(\omega) + \frac{1}{j\omega}$ | 단위 계단 |
| $e^{-at}u(t)$, $a > 0$ | $\frac{1}{a + j\omega}$ | 인과 지수 |
| $e^{-a\|t\|}$, $a > 0$ | $\frac{2a}{a^2 + \omega^2}$ | 양측 지수 |
| $\text{rect}(t/\tau)$ | $\tau \, \text{sinc}(\omega\tau / 2\pi)$ | 직사각형 $\leftrightarrow$ sinc |
| $\text{sinc}(Wt)$ | $\frac{1}{W}\text{rect}(\omega / (2\pi W))$ | sinc $\leftrightarrow$ 직사각형 (쌍대성) |
| $e^{-t^2/(2\sigma^2)}$ | $\sigma\sqrt{2\pi} \, e^{-\sigma^2\omega^2/2}$ | 가우시안 $\leftrightarrow$ 가우시안 |
| $\text{tri}(t/\tau)$ | $\tau \, \text{sinc}^2(\omega\tau / 2\pi)$ | 삼각형 $\leftrightarrow$ sinc$^2$ |
| $\text{sgn}(t)$ | $\frac{2}{j\omega}$ | 부호 함수 |
| $\delta(t - t_0)$ | $e^{-j\omega t_0}$ | 이동된 임펄스 |

### 4.2 유도 예시

**인과 지수** $x(t) = e^{-at}u(t)$, $a > 0$:

$$X(\omega) = \int_0^{\infty} e^{-at} e^{-j\omega t} \, dt = \int_0^{\infty} e^{-(a+j\omega)t} \, dt = \frac{1}{a + j\omega}$$

크기: $|X(\omega)| = \frac{1}{\sqrt{a^2 + \omega^2}}$

위상: $\angle X(\omega) = -\arctan(\omega/a)$

**직사각 펄스** $x(t) = \text{rect}(t/\tau)$:

$$X(\omega) = \int_{-\tau/2}^{\tau/2} e^{-j\omega t} \, dt = \frac{e^{j\omega\tau/2} - e^{-j\omega\tau/2}}{j\omega} = \tau \cdot \frac{\sin(\omega\tau/2)}{\omega\tau/2} = \tau \, \text{sinc}\left(\frac{\omega\tau}{2\pi}\right)$$

**가우시안 펄스** $x(t) = e^{-t^2/(2\sigma^2)}$:

$$X(\omega) = \int_{-\infty}^{\infty} e^{-t^2/(2\sigma^2)} e^{-j\omega t} \, dt$$

지수의 제곱 완성:

$$-\frac{t^2}{2\sigma^2} - j\omega t = -\frac{1}{2\sigma^2}(t + j\sigma^2\omega)^2 - \frac{\sigma^2\omega^2}{2}$$

$$X(\omega) = e^{-\sigma^2\omega^2/2} \int_{-\infty}^{\infty} e^{-(t+j\sigma^2\omega)^2/(2\sigma^2)} \, dt = \sigma\sqrt{2\pi} \, e^{-\sigma^2\omega^2/2}$$

가우시안은 (스케일링까지) 자기 자신의 푸리에 변환입니다. 시간 영역에서 좁은 가우시안은 주파수 영역에서 넓은 가우시안을 만들고, 그 반대도 마찬가지입니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Fourier transform pairs visualization ---

fig, axes = plt.subplots(4, 2, figsize=(14, 16))

# 1. Causal exponential
a = 2.0
t = np.linspace(-1, 5, 1000)
x1 = np.where(t >= 0, np.exp(-a * t), 0.0)
omega = np.linspace(-30, 30, 1000)
X1 = 1 / (a + 1j * omega)

axes[0, 0].plot(t, x1, 'b-', linewidth=2)
axes[0, 0].set_title(f'$x(t) = e^{{-{a}t}}u(t)$')
axes[0, 0].set_xlabel('t')
axes[0, 0].fill_between(t, x1, alpha=0.2)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(omega, np.abs(X1), 'r-', linewidth=2, label='$|X(\\omega)|$')
axes[0, 1].plot(omega, np.angle(X1), 'g--', linewidth=1.5, label='$\\angle X(\\omega)$')
axes[0, 1].set_title(f'$X(\\omega) = 1/({a} + j\\omega)$')
axes[0, 1].set_xlabel('$\\omega$ (rad/s)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 2. Rectangular pulse
tau = 2.0
t2 = np.linspace(-4, 4, 1000)
x2 = np.where(np.abs(t2) <= tau / 2, 1.0, 0.0)
X2 = tau * np.sinc(omega * tau / (2 * np.pi))

axes[1, 0].plot(t2, x2, 'b-', linewidth=2)
axes[1, 0].set_title(f'$x(t) = \\mathrm{{rect}}(t/{tau})$')
axes[1, 0].set_xlabel('t')
axes[1, 0].fill_between(t2, x2, alpha=0.2)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(omega, X2, 'r-', linewidth=2)
axes[1, 1].set_title(f'$X(\\omega) = {tau} \\cdot \\mathrm{{sinc}}(\\omega \\cdot {tau}/(2\\pi))$')
axes[1, 1].set_xlabel('$\\omega$ (rad/s)')
axes[1, 1].grid(True, alpha=0.3)

# 3. Gaussian pulse
for sigma in [0.5, 1.0, 2.0]:
    x3 = np.exp(-t2**2 / (2 * sigma**2))
    X3 = sigma * np.sqrt(2 * np.pi) * np.exp(-sigma**2 * omega**2 / 2)
    axes[2, 0].plot(t2, x3, linewidth=2, label=f'$\\sigma={sigma}$')
    axes[2, 1].plot(omega, X3, linewidth=2, label=f'$\\sigma={sigma}$')

axes[2, 0].set_title('Gaussian: $x(t) = e^{-t^2/(2\\sigma^2)}$')
axes[2, 0].set_xlabel('t')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].set_title('$X(\\omega) = \\sigma\\sqrt{2\\pi} \\cdot e^{-\\sigma^2\\omega^2/2}$')
axes[2, 1].set_xlabel('$\\omega$ (rad/s)')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

# 4. Two-sided exponential
x4 = np.exp(-a * np.abs(t2))
X4 = 2 * a / (a**2 + omega**2)

axes[3, 0].plot(t2, x4, 'b-', linewidth=2)
axes[3, 0].set_title(f'$x(t) = e^{{-{a}|t|}}$')
axes[3, 0].set_xlabel('t')
axes[3, 0].fill_between(t2, x4, alpha=0.2)
axes[3, 0].grid(True, alpha=0.3)

axes[3, 1].plot(omega, X4, 'r-', linewidth=2)
axes[3, 1].set_title(f'$X(\\omega) = {2*a}/({a**2} + \\omega^2)$')
axes[3, 1].set_xlabel('$\\omega$ (rad/s)')
axes[3, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transform_pairs.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 5. 푸리에 변환의 성질

푸리에 변환의 성질들은 매우 강력합니다 — 이를 통해 적분을 다시 계산하지 않고도 간단한 변환으로부터 복잡한 신호의 변환을 결정할 수 있습니다.

### 5.1 선형성(Linearity)

$$\mathcal{F}\{ax_1(t) + bx_2(t)\} = aX_1(\omega) + bX_2(\omega)$$

주파수 영역에서 중첩(superposition)이 성립합니다.

### 5.2 시간 이동(Time Shifting)

$$\mathcal{F}\{x(t - t_0)\} = e^{-j\omega t_0} X(\omega)$$

시간 지연은 스펙트럼에 **선형 위상** $-\omega t_0$를 추가합니다. **크기 스펙트럼은 변하지 않습니다** — 신호를 시간 축에서 이동해도 주파수 내용은 바뀌지 않고 위상 관계만 바뀝니다.

**증명**:

$$\int x(t - t_0) e^{-j\omega t} dt \overset{\tau = t - t_0}{=} \int x(\tau) e^{-j\omega(\tau + t_0)} d\tau = e^{-j\omega t_0} X(\omega)$$

### 5.3 주파수 이동(Frequency Shifting) — 변조

$$\mathcal{F}\{x(t) e^{j\omega_0 t}\} = X(\omega - \omega_0)$$

시간 영역에서 복소 지수를 곱하면 스펙트럼이 $\omega_0$만큼 **이동**합니다. 이것이 통신에서 **변조(modulation)**의 수학적 기초입니다.

**따름정리** (코사인 변조):

$$\mathcal{F}\{x(t)\cos(\omega_0 t)\} = \frac{1}{2}[X(\omega - \omega_0) + X(\omega + \omega_0)]$$

스펙트럼이 $\pm\omega_0$로 이동하고 진폭이 절반이 됩니다.

### 5.4 시간 스케일링(Time Scaling)

$$\mathcal{F}\{x(at)\} = \frac{1}{|a|} X\left(\frac{\omega}{a}\right)$$

- 시간 압축 ($|a| > 1$)은 주파수 **팽창**을 유발
- 시간 팽창 ($|a| < 1$)은 주파수 **압축**을 유발
- $|X(\omega)|$ 아래 면적은 에너지 보존을 위해 $1/|a|$로 스케일됨

이것이 **불확정성 원리**의 수학적 표현입니다: 시간과 주파수에서 동시에 임의로 국소화(localized)될 수 없습니다.

### 5.5 시간 반전(Time Reversal)

$$\mathcal{F}\{x(-t)\} = X(-\omega)$$

$a = -1$인 스케일링의 특수 경우입니다. 실수 신호에 대해 $X(-\omega) = X^*(\omega)$이므로:

$$\mathcal{F}\{x(-t)\} = X^*(\omega) \quad \text{(실수 신호)}$$

### 5.6 쌍대성(Duality)

$x(t) \leftrightarrow X(\omega)$이면:

$$X(t) \leftrightarrow 2\pi \, x(-\omega)$$

**예시**: $\text{rect}(t/\tau) \leftrightarrow \tau\,\text{sinc}(\omega\tau/(2\pi))$이므로, 쌍대성에 의해:

$$\tau\,\text{sinc}(\tau t/(2\pi)) \leftrightarrow 2\pi \, \text{rect}(-\omega/\tau) = 2\pi\,\text{rect}(\omega/\tau)$$

이것이 sinc 함수의 변환을 얻는 방법입니다.

### 5.7 시간 미분(Differentiation in Time)

$$\mathcal{F}\left\{\frac{d^n x}{dt^n}\right\} = (j\omega)^n X(\omega)$$

시간 미분은 주파수에서 $j\omega$를 곱하는 것에 해당합니다. 고주파수가 증폭됩니다 — 미분은 **고역통과 연산**입니다.

### 5.8 적분(Integration)

$$\mathcal{F}\left\{\int_{-\infty}^{t} x(\tau) \, d\tau\right\} = \frac{X(\omega)}{j\omega} + \pi X(0)\delta(\omega)$$

적분은 $j\omega$로 나누는 것에 해당합니다 (DC 항 포함). 저주파수가 증폭됩니다 — 적분은 **저역통과 연산**입니다.

### 5.9 주파수 미분(Differentiation in Frequency)

$$\mathcal{F}\{(-jt)^n x(t)\} = \frac{d^n X(\omega)}{d\omega^n}$$

또는 동등하게:

$$\mathcal{F}\{t^n x(t)\} = j^n \frac{d^n X(\omega)}{d\omega^n}$$

### 5.10 성질 요약 표

| 성질 | 시간 영역 | 주파수 영역 |
|----------|------------|-----------------|
| 선형성 | $ax_1 + bx_2$ | $aX_1 + bX_2$ |
| 시간 이동 | $x(t - t_0)$ | $e^{-j\omega t_0}X(\omega)$ |
| 주파수 이동 | $x(t)e^{j\omega_0 t}$ | $X(\omega - \omega_0)$ |
| 스케일링 | $x(at)$ | $\frac{1}{|a|}X(\omega/a)$ |
| 반전 | $x(-t)$ | $X(-\omega)$ |
| 쌍대성 | $X(t)$ | $2\pi x(-\omega)$ |
| 시간 미분 | $\frac{dx}{dt}$ | $j\omega X(\omega)$ |
| 주파수 미분 | $(-jt)x(t)$ | $\frac{dX}{d\omega}$ |
| 합성곱 | $x * h$ | $X \cdot H$ |
| 곱셈 | $x \cdot w$ | $\frac{1}{2\pi}X * W$ |
| 켤레 | $x^*(t)$ | $X^*(-\omega)$ |
| 파르스발 | $\int|x|^2 dt$ | $\frac{1}{2\pi}\int|X|^2 d\omega$ |

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Fourier transform properties demonstration ---

# Use FFT to approximate the CTFT
def approx_ctft(x, t):
    """Approximate CTFT using FFT."""
    dt = t[1] - t[0]
    N = len(t)
    X = np.fft.fftshift(np.fft.fft(x)) * dt
    omega = np.fft.fftshift(np.fft.fftfreq(N, dt)) * 2 * np.pi
    return omega, X

# Setup
fs = 1000
t = np.arange(-5, 5, 1 / fs)

# Original signal: Gaussian pulse
sigma = 0.5
x = np.exp(-t**2 / (2 * sigma**2))
omega, X = approx_ctft(x, t)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# 1. Time shifting
t0 = 1.5
x_shifted = np.exp(-(t - t0)**2 / (2 * sigma**2))
_, X_shifted = approx_ctft(x_shifted, t)

axes[0, 0].plot(t, x, 'b-', linewidth=2, label='$x(t)$')
axes[0, 0].plot(t, x_shifted, 'r--', linewidth=2, label=f'$x(t - {t0})$')
axes[0, 0].set_title('Time Shifting')
axes[0, 0].legend()
axes[0, 0].set_xlabel('t')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(omega, np.abs(X), 'b-', linewidth=2, label='$|X(\\omega)|$')
axes[0, 1].plot(omega, np.abs(X_shifted), 'r--', linewidth=2,
                label='$|X_{shifted}(\\omega)|$')
axes[0, 1].set_title('Magnitude unchanged by time shift')
axes[0, 1].legend()
axes[0, 1].set_xlabel('$\\omega$')
axes[0, 1].set_xlim([-20, 20])
axes[0, 1].grid(True, alpha=0.3)

# 2. Time scaling
x_compressed = np.exp(-(2 * t)**2 / (2 * sigma**2))
x_expanded = np.exp(-(0.5 * t)**2 / (2 * sigma**2))
_, X_comp = approx_ctft(x_compressed, t)
_, X_exp = approx_ctft(x_expanded, t)

axes[1, 0].plot(t, x, 'b-', linewidth=2, label='$x(t)$')
axes[1, 0].plot(t, x_compressed, 'r--', linewidth=2, label='$x(2t)$ compressed')
axes[1, 0].plot(t, x_expanded, 'g--', linewidth=2, label='$x(0.5t)$ expanded')
axes[1, 0].set_title('Time Scaling')
axes[1, 0].legend()
axes[1, 0].set_xlabel('t')
axes[1, 0].set_xlim([-4, 4])
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(omega, np.abs(X), 'b-', linewidth=2, label='$|X(\\omega)|$')
axes[1, 1].plot(omega, np.abs(X_comp), 'r--', linewidth=2,
                label='$|X_{comp}(\\omega)|$ expanded')
axes[1, 1].plot(omega, np.abs(X_exp), 'g--', linewidth=2,
                label='$|X_{exp}(\\omega)|$ compressed')
axes[1, 1].set_title('Compression in time = expansion in frequency')
axes[1, 1].legend(fontsize=8)
axes[1, 1].set_xlabel('$\\omega$')
axes[1, 1].set_xlim([-20, 20])
axes[1, 1].grid(True, alpha=0.3)

# 3. Frequency shifting (modulation)
omega_c = 20.0  # carrier frequency
x_mod = x * np.cos(omega_c * t)
_, X_mod = approx_ctft(x_mod, t)

axes[2, 0].plot(t, x, 'b-', linewidth=1, alpha=0.5, label='$x(t)$ envelope')
axes[2, 0].plot(t, x_mod, 'r-', linewidth=1, label='$x(t)\\cos(\\omega_c t)$')
axes[2, 0].set_title(f'Frequency Shifting (modulation, $\\omega_c={omega_c}$)')
axes[2, 0].legend()
axes[2, 0].set_xlabel('t')
axes[2, 0].set_xlim([-3, 3])
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].plot(omega, np.abs(X), 'b-', linewidth=2, label='Original $|X(\\omega)|$')
axes[2, 1].plot(omega, np.abs(X_mod), 'r-', linewidth=2,
                label='$|X_{mod}(\\omega)|$')
axes[2, 1].set_title('Spectrum shifted to $\\pm\\omega_c$')
axes[2, 1].legend()
axes[2, 1].set_xlabel('$\\omega$')
axes[2, 1].set_xlim([-40, 40])
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ft_properties.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6. 합성곱 정리

### 6.1 공식

$$\mathcal{F}\{x(t) * h(t)\} = X(\omega) \cdot H(\omega)$$

**시간 영역에서의 합성곱은 주파수 영역에서의 곱셈입니다.**

이것은 아마도 푸리에 변환의 가장 중요한 성질입니다. 이를 통해 LTI 시스템의 출력을 다음과 같이 계산할 수 있습니다:

1. 입력 변환: $x(t) \to X(\omega)$
2. 주파수 응답 곱셈: $Y(\omega) = X(\omega) \cdot H(\omega)$
3. 역변환: $Y(\omega) \to y(t)$

### 6.2 증명

$$\mathcal{F}\{x * h\} = \int \left[\int x(\tau) h(t - \tau) d\tau\right] e^{-j\omega t} dt$$

적분 순서 교환:

$$= \int x(\tau) \left[\int h(t - \tau) e^{-j\omega t} dt\right] d\tau$$

내부 적분에서 $u = t - \tau$로 치환:

$$= \int x(\tau) \left[\int h(u) e^{-j\omega(u + \tau)} du\right] d\tau = \int x(\tau) e^{-j\omega\tau} d\tau \cdot \int h(u) e^{-j\omega u} du$$

$$= X(\omega) \cdot H(\omega)$$

### 6.3 주파수 영역에서의 LTI 시스템 분석

임펄스 응답 $h(t)$와 입력 $x(t)$를 가진 LTI 시스템에 대해:

$$Y(\omega) = H(\omega) \cdot X(\omega)$$

여기서 $H(\omega) = \mathcal{F}\{h(t)\}$는 **주파수 응답** (또는 **전달 함수**)입니다.

출력 스펙트럼은 주파수 응답에 의해 형성됩니다:
- $|H(\omega)| > 1$인 주파수에서: 증폭
- $|H(\omega)| < 1$인 주파수에서: 감쇠
- $|H(\omega)| = 0$인 주파수에서: 완전 제거 (영점)

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Convolution theorem demonstration ---

fs = 1000
t = np.arange(-5, 5, 1 / fs)
N = len(t)
dt = 1 / fs

# Input: sum of two frequencies
f1, f2 = 3, 15  # Hz
x = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

# LTI system: lowpass (exponential decay)
a = 10 * 2 * np.pi  # time constant
h = np.where(t >= 0, a * np.exp(-a * t), 0.0)
h /= np.sum(h) * dt  # normalize for unity DC gain

# Time-domain convolution
y_time = np.convolve(x, h, mode='full')[:N] * dt

# Frequency-domain multiplication
omega = np.fft.fftshift(np.fft.fftfreq(N, dt)) * 2 * np.pi
X = np.fft.fftshift(np.fft.fft(x)) * dt
H = np.fft.fftshift(np.fft.fft(h)) * dt
Y_freq = X * H
y_freq = np.real(np.fft.ifft(np.fft.ifftshift(Y_freq))) / dt

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Time domain signals
axes[0, 0].plot(t, x, 'b-', linewidth=1)
axes[0, 0].set_title('Input $x(t)$: 3 Hz + 15 Hz')
axes[0, 0].set_xlabel('t (s)')
axes[0, 0].set_xlim([-1, 2])
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t, h, 'r-', linewidth=2)
axes[0, 1].set_title('Impulse Response $h(t)$: Lowpass')
axes[0, 1].set_xlabel('t (s)')
axes[0, 1].set_xlim([-0.1, 0.5])
axes[0, 1].grid(True, alpha=0.3)

# Frequency domain
f_hz = omega / (2 * np.pi)
axes[1, 0].plot(f_hz, np.abs(X), 'b-', linewidth=1.5)
axes[1, 0].set_title('Input Spectrum $|X(\\omega)|$')
axes[1, 0].set_xlabel('Frequency (Hz)')
axes[1, 0].set_xlim([-30, 30])
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(f_hz, np.abs(H), 'r-', linewidth=2)
axes[1, 1].set_title('Frequency Response $|H(\\omega)|$')
axes[1, 1].set_xlabel('Frequency (Hz)')
axes[1, 1].set_xlim([-30, 30])
axes[1, 1].grid(True, alpha=0.3)

# Output comparison
axes[2, 0].plot(t, y_time, 'g-', linewidth=1.5, label='Time-domain conv')
axes[2, 0].plot(t, y_freq, 'k--', linewidth=1, label='Freq-domain mult')
axes[2, 0].set_title('Output $y(t) = x * h$ (both methods agree)')
axes[2, 0].set_xlabel('t (s)')
axes[2, 0].set_xlim([-1, 2])
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].plot(f_hz, np.abs(Y_freq), 'g-', linewidth=2)
axes[2, 1].set_title('Output Spectrum $|Y(\\omega)| = |X \\cdot H|$')
axes[2, 1].set_xlabel('Frequency (Hz)')
axes[2, 1].set_xlim([-30, 30])
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('convolution_theorem.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 7. 곱셈 정리(윈도잉)

### 7.1 공식

$$\mathcal{F}\{x(t) \cdot w(t)\} = \frac{1}{2\pi} X(\omega) * W(\omega)$$

**시간 영역에서의 곱셈은 주파수 영역에서의 합성곱입니다** ($1/(2\pi)$ 스케일링).

### 7.2 윈도잉(Windowing)

유한 구간 $[-T/2, T/2]$에서 신호를 관측하면, 암묵적으로 직사각 창 $w(t) = \text{rect}(t/T)$를 곱하는 것과 같습니다.

관측된 스펙트럼은 실제 $X(\omega)$가 아니라 **합성곱된** (번져진) 스펙트럼입니다:

$$X_{\text{obs}}(\omega) = \frac{1}{2\pi} X(\omega) * W(\omega)$$

여기서 $W(\omega) = T\,\text{sinc}(\omega T/(2\pi))$는 직사각 창의 변환입니다.

이 합성곱은:
- 스펙트럼 피크를 **넓힘** (주파수 분해능 감소)
- **사이드로브(sidelobe)** 생성 (sinc 함수의 스펙트럼 누설)

### 7.3 창 함수(Window Functions)

서로 다른 창 함수들은 주엽 너비(분해능)와 부엽 레벨(누설) 사이의 트레이드오프를 제공합니다:

| 창 함수 | 주엽 너비 | 첫 번째 부엽 (dB) | 사용 사례 |
|--------|----------------|---------------------|----------|
| 직사각형(Rectangular) | 가장 좁음 | -13 | 최대 분해능 |
| 해밍(Hamming) | 1.8x 직사각형 | -42 | 범용 |
| 해닝(Hanning) | 2.0x 직사각형 | -31 | 범용 |
| 블랙만(Blackman) | 2.9x 직사각형 | -58 | 낮은 누설 |
| 카이저(Kaiser) ($\beta$) | 가변 | 가변 | 조정 가능한 트레이드오프 |

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- Windowing effect on spectrum ---

fs = 1000
N = 1024
t = np.arange(N) / fs

# Signal: two close sinusoids
f1, f2 = 50, 55  # Hz (5 Hz apart)
x = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

# Window functions
windows = {
    'Rectangular': np.ones(N),
    'Hamming': np.hamming(N),
    'Hanning': np.hanning(N),
    'Blackman': np.blackman(N),
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

freq = np.fft.rfftfreq(N, 1 / fs)

for ax, (name, w) in zip(axes.flat, windows.items()):
    x_windowed = x * w
    X = np.abs(np.fft.rfft(x_windowed)) / np.sum(w) * 2

    ax.plot(freq, 20 * np.log10(X + 1e-12), linewidth=1.5)
    ax.set_title(f'{name} Window')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_xlim([20, 80])
    ax.set_ylim([-80, 5])
    ax.axvline(x=f1, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=f2, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3)

plt.suptitle('Effect of Windowing on Spectral Analysis', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('windowing_effect.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 8. 파르스발 정리

### 8.1 공식 (에너지 버전)

$$E_x = \int_{-\infty}^{\infty} |x(t)|^2 \, dt = \frac{1}{2\pi} \int_{-\infty}^{\infty} |X(\omega)|^2 \, d\omega$$

또는 일반 주파수 $f$를 사용하면:

$$\int_{-\infty}^{\infty} |x(t)|^2 \, dt = \int_{-\infty}^{\infty} |X(f)|^2 \, df$$

### 8.2 해석

- **좌변**: 전체 신호 에너지 (시간 영역에서 계산)
- **우변**: 스펙트럼 밀도로부터 계산된 전체 에너지

$|X(\omega)|^2$는 **에너지 스펙트럼 밀도(ESD)** — 각 무한소 주파수 대역에 얼마나 많은 에너지가 포함되어 있는지 알려줍니다.

### 8.3 일반화 파르스발 (레일리 정리)

두 신호에 대해:

$$\int_{-\infty}^{\infty} x(t) y^*(t) \, dt = \frac{1}{2\pi} \int_{-\infty}^{\infty} X(\omega) Y^*(\omega) \, d\omega$$

### 8.4 주파수 대역의 에너지

주파수 대역 $[\omega_1, \omega_2]$에서의 에너지는:

$$E_{[\omega_1, \omega_2]} = \frac{1}{2\pi} \int_{\omega_1}^{\omega_2} |X(\omega)|^2 \, d\omega + \frac{1}{2\pi} \int_{-\omega_2}^{-\omega_1} |X(\omega)|^2 \, d\omega$$

켤레 대칭을 가진 실수 신호에 대해:

$$E_{[\omega_1, \omega_2]} = \frac{1}{\pi} \int_{\omega_1}^{\omega_2} |X(\omega)|^2 \, d\omega$$

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Parseval's theorem verification ---

fs = 1000
t = np.arange(-5, 5, 1 / fs)
dt = 1 / fs

# Gaussian pulse
sigma = 0.3
x = np.exp(-t**2 / (2 * sigma**2))

# Time-domain energy
E_time = np.trapz(np.abs(x)**2, t)

# Frequency-domain energy
N = len(t)
omega = np.fft.fftshift(np.fft.fftfreq(N, dt)) * 2 * np.pi
X = np.fft.fftshift(np.fft.fft(x)) * dt
E_freq = np.trapz(np.abs(X)**2, omega) / (2 * np.pi)

# Analytical energy
E_analytical = sigma * np.sqrt(np.pi)

print(f"=== Parseval's Theorem Verification ===")
print(f"Gaussian pulse with sigma = {sigma}")
print(f"Time-domain energy:      {E_time:.8f}")
print(f"Frequency-domain energy: {E_freq:.8f}")
print(f"Analytical energy:       {E_analytical:.8f}")
print(f"Error (time vs freq):    {abs(E_time - E_freq):.2e}")

# Energy distribution across frequency bands
ESD = np.abs(X)**2 / (2 * np.pi)  # energy spectral density

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(t, x, 'b-', linewidth=2)
axes[0].fill_between(t, x, alpha=0.2)
axes[0].set_title(f'$x(t)$: Gaussian ($\\sigma = {sigma}$)')
axes[0].set_xlabel('t')
axes[0].grid(True, alpha=0.3)

axes[1].plot(omega, np.abs(X)**2, 'r-', linewidth=2)
axes[1].fill_between(omega, np.abs(X)**2, alpha=0.2, color='red')
axes[1].set_title('Energy Spectral Density $|X(\\omega)|^2$')
axes[1].set_xlabel('$\\omega$ (rad/s)')
axes[1].set_xlim([-30, 30])
axes[1].grid(True, alpha=0.3)

# Cumulative energy as function of bandwidth
omega_pos = omega[omega >= 0]
ESD_pos = np.abs(X[omega >= 0])**2
cum_energy = np.cumsum(ESD_pos) * (omega_pos[1] - omega_pos[0]) / np.pi
axes[2].plot(omega_pos, cum_energy / E_time * 100, 'g-', linewidth=2)
axes[2].axhline(y=90, color='gray', linestyle='--', label='90%')
axes[2].axhline(y=99, color='lightgray', linestyle='--', label='99%')
axes[2].set_title("Cumulative Energy vs Bandwidth (Parseval's)")
axes[2].set_xlabel('$\\omega$ (rad/s)')
axes[2].set_ylabel('% of total energy')
axes[2].set_xlim([0, 30])
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('parseval_ctft.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 9. 주파수 영역 분석

### 9.1 스펙트럼이 알려주는 것

푸리에 변환은 신호를 구성 주파수들로 분해합니다:

| 스펙트럼 특징 | 시간 영역 해석 |
|-----------------|---------------------------|
| $\omega_0$에서의 피크 | $\omega_0$ 주파수에서의 지배적 진동 |
| 넓은 주엽 | 짧은 지속 시간 펄스 (시간-대역폭 트레이드오프) |
| 좁은 주엽 | 긴 지속 시간 또는 주기 신호 |
| 평탄한 스펙트럼 | 임펄스 신호 (모든 주파수 존재) |
| 급격한 위상 변화 | 신호의 갑작스러운 전환 |
| 대칭 크기 | 실수값 신호 |

### 9.2 복합 신호의 스펙트럼 분석

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Spectral analysis of various signals ---

fs = 4000
duration = 2.0
t = np.arange(0, duration, 1 / fs)
N = len(t)

signals = {
    'Pure tone (440 Hz)': np.sin(2 * np.pi * 440 * t),
    'Two tones (440 + 880 Hz)': np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t),
    'AM signal (carrier 500 Hz)': (1 + 0.5 * np.cos(2 * np.pi * 50 * t)) * np.cos(2 * np.pi * 500 * t),
    'Chirp (100 to 1000 Hz)': np.sin(2 * np.pi * (100 * t + 450 * t**2 / (2 * duration))),
    'Gaussian pulse (t=1s)': np.exp(-(t - 1.0)**2 / (2 * 0.01**2)),
    'White noise': np.random.randn(N),
}

fig, axes = plt.subplots(len(signals), 2, figsize=(16, 3.5 * len(signals)))

freq = np.fft.rfftfreq(N, 1 / fs)

for row, (name, x) in enumerate(signals.items()):
    # Time domain
    axes[row, 0].plot(t[:2000], x[:2000], 'b-', linewidth=0.8)
    axes[row, 0].set_title(f'{name}')
    axes[row, 0].set_xlabel('Time (s)')
    axes[row, 0].set_ylabel('Amplitude')
    axes[row, 0].grid(True, alpha=0.3)

    # Frequency domain
    X = np.abs(np.fft.rfft(x * np.hanning(N))) * 2 / N
    axes[row, 1].plot(freq, 20 * np.log10(X + 1e-12), 'r-', linewidth=0.8)
    axes[row, 1].set_title(f'Spectrum of {name}')
    axes[row, 1].set_xlabel('Frequency (Hz)')
    axes[row, 1].set_ylabel('Magnitude (dB)')
    axes[row, 1].set_xlim([0, 1500])
    axes[row, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spectral_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 9.3 LTI 시스템 분석

주파수 응답 $H(\omega)$는 LTI 시스템이 각 주파수를 어떻게 수정하는지 완전히 설명합니다:

$$Y(\omega) = H(\omega) X(\omega)$$

인과 1차 시스템 $y'(t) + ay(t) = x(t)$에 대해:

푸리에 변환 적용: $j\omega Y(\omega) + aY(\omega) = X(\omega)$

$$H(\omega) = \frac{Y(\omega)}{X(\omega)} = \frac{1}{a + j\omega}$$

- **크기**: $|H(\omega)| = 1/\sqrt{a^2 + \omega^2}$ (저역통과, 단조 감소)
- **3 dB 차단 주파수**: $\omega_c = a$ ($|H| = 1/\sqrt{2}$)
- **기울기**: $\omega \gg a$에서 $-20$ dB/decade (1차 시스템)
- **위상**: $\angle H(\omega) = -\arctan(\omega/a)$ (위상 지연이 주파수에 따라 증가)

---

## 10. 이상적인 필터

### 10.1 이상적인 저역통과 필터

$$H_{LP}(\omega) = \begin{cases} 1 & |\omega| \leq \omega_c \\ 0 & |\omega| > \omega_c \end{cases} = \text{rect}\left(\frac{\omega}{2\omega_c}\right)$$

- $\omega_c$ 이하의 모든 주파수를 왜곡 없이 통과
- $\omega_c$ 이상의 모든 주파수를 완전히 제거
- 임펄스 응답: $h(t) = \frac{\omega_c}{\pi}\text{sinc}\left(\frac{\omega_c t}{\pi}\right)$

> 이상적인 저역통과 필터는 **비인과적** ($t < 0$에서 $h(t) \neq 0$)이므로 **물리적으로 실현 불가능**합니다. 실제 필터는 이 이상적 형태를 근사합니다.

### 10.2 이상적인 고역통과 필터

$$H_{HP}(\omega) = 1 - H_{LP}(\omega) = \begin{cases} 0 & |\omega| \leq \omega_c \\ 1 & |\omega| > \omega_c \end{cases}$$

임펄스 응답: $h(t) = \delta(t) - \frac{\omega_c}{\pi}\text{sinc}\left(\frac{\omega_c t}{\pi}\right)$

### 10.3 이상적인 대역통과 필터

$$H_{BP}(\omega) = \begin{cases} 1 & \omega_1 \leq |\omega| \leq \omega_2 \\ 0 & \text{otherwise} \end{cases}$$

### 10.4 이상적인 대역저지(노치) 필터

$$H_{BS}(\omega) = 1 - H_{BP}(\omega) = \begin{cases} 0 & \omega_1 \leq |\omega| \leq \omega_2 \\ 1 & \text{otherwise} \end{cases}$$

### 10.5 이상적인 전역통과 필터

$$|H_{AP}(\omega)| = 1, \quad \angle H_{AP}(\omega) = \phi(\omega)$$

크기는 변경하지 않고 위상만 변경합니다. 위상 등화(phase equalization)에 사용됩니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Ideal filters and their impulse responses ---

omega = np.linspace(-60, 60, 2000)
omega_c = 20  # cutoff frequency
omega_1, omega_2 = 15, 25  # bandpass edges
t = np.linspace(-2, 2, 2000)

fig, axes = plt.subplots(4, 2, figsize=(14, 14))

# 1. Lowpass
H_lp = np.where(np.abs(omega) <= omega_c, 1.0, 0.0)
h_lp = omega_c / np.pi * np.sinc(omega_c * t / np.pi)

axes[0, 0].plot(omega, H_lp, 'b-', linewidth=2)
axes[0, 0].set_title(f'Ideal Lowpass ($\\omega_c = {omega_c}$)')
axes[0, 0].set_xlabel('$\\omega$ (rad/s)')
axes[0, 0].set_ylabel('$|H(\\omega)|$')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t, h_lp, 'b-', linewidth=2)
axes[0, 1].set_title('Impulse Response: $h(t) = \\frac{\\omega_c}{\\pi}\\mathrm{sinc}(\\omega_c t / \\pi)$')
axes[0, 1].set_xlabel('t')
axes[0, 1].axvline(x=0, color='gray', linestyle=':', alpha=0.5)
axes[0, 1].grid(True, alpha=0.3)

# 2. Highpass
H_hp = 1 - H_lp
h_hp_approx = np.zeros_like(t)
# delta(t) approximation + sinc
dt = t[1] - t[0]
delta_idx = np.argmin(np.abs(t))
h_hp_approx[delta_idx] = 1 / dt
h_hp_approx -= h_lp

axes[1, 0].plot(omega, H_hp, 'r-', linewidth=2)
axes[1, 0].set_title(f'Ideal Highpass ($\\omega_c = {omega_c}$)')
axes[1, 0].set_xlabel('$\\omega$ (rad/s)')
axes[1, 0].set_ylabel('$|H(\\omega)|$')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(t, -h_lp, 'r-', linewidth=2)
axes[1, 1].set_title('$h(t) = \\delta(t) - h_{LP}(t)$ (sinc part shown)')
axes[1, 1].set_xlabel('t')
axes[1, 1].grid(True, alpha=0.3)

# 3. Bandpass
H_bp = np.where((np.abs(omega) >= omega_1) & (np.abs(omega) <= omega_2), 1.0, 0.0)
omega_m = (omega_1 + omega_2) / 2
B = omega_2 - omega_1
h_bp = B / np.pi * np.sinc(B * t / (2 * np.pi)) * np.cos(omega_m * t)

axes[2, 0].plot(omega, H_bp, 'g-', linewidth=2)
axes[2, 0].set_title(f'Ideal Bandpass ($\\omega_1={omega_1}$, $\\omega_2={omega_2}$)')
axes[2, 0].set_xlabel('$\\omega$ (rad/s)')
axes[2, 0].set_ylabel('$|H(\\omega)|$')
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].plot(t, h_bp, 'g-', linewidth=1.5)
axes[2, 1].set_title('Bandpass Impulse Response')
axes[2, 1].set_xlabel('t')
axes[2, 1].grid(True, alpha=0.3)

# 4. Bandstop
H_bs = 1 - H_bp

axes[3, 0].plot(omega, H_bs, 'm-', linewidth=2)
axes[3, 0].set_title(f'Ideal Bandstop (Notch)')
axes[3, 0].set_xlabel('$\\omega$ (rad/s)')
axes[3, 0].set_ylabel('$|H(\\omega)|$')
axes[3, 0].grid(True, alpha=0.3)

axes[3, 1].text(0.5, 0.5, '$h(t) = \\delta(t) - h_{BP}(t)$',
               transform=axes[3, 1].transAxes, fontsize=14, ha='center', va='center')
axes[3, 1].set_title('Bandstop = All - Bandpass')
axes[3, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ideal_filters.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 11. 대역폭과 시간-대역폭 곱

### 11.1 대역폭 정의

신호의 대역폭을 정의하는 여러 방법이 있습니다:

| 정의 | 수식 | 설명 |
|-----------|---------|-------------|
| **3 dB 대역폭** | $B_{3dB}$: $\|X(\omega_c)\|^2 = \frac{1}{2}\|X(0)\|^2$ | 반전력(half-power) 대역폭 |
| **영점 간 대역폭** | 첫 번째 영점 사이 거리 | sinc 형 스펙트럼에 주로 사용 |
| **등가 잡음 대역폭** | $B_{eq} = \frac{\int|X(\omega)|^2 d\omega}{|X(\omega_0)|^2}$ | 같은 피크와 에너지를 가진 직사각형 |
| **RMS 대역폭** | $B_{rms} = \sqrt{\frac{\int\omega^2|X(\omega)|^2 d\omega}{\int|X(\omega)|^2 d\omega}}$ | $|X|^2$의 표준 편차 |
| **본질 대역폭** | $B_{ess}$: $p$%의 에너지 포함 | 주로 $p = 99$ |

### 11.2 시간-대역폭 곱 (불확정성 원리)

신호 지속 시간 $\Delta t$와 대역폭 $\Delta\omega$ 사이에는 근본적인 제약이 있습니다:

$$\Delta t \cdot \Delta\omega \geq \frac{1}{2}$$

여기서 $\Delta t$와 $\Delta\omega$는 시간과 주파수에서의 RMS 지속 시간으로 정의됩니다:

$$\Delta t = \sqrt{\frac{\int t^2 |x(t)|^2 dt}{\int |x(t)|^2 dt}}, \quad \Delta\omega = \sqrt{\frac{\int \omega^2 |X(\omega)|^2 d\omega}{\int |X(\omega)|^2 d\omega}}$$

**등호** $\Delta t \cdot \Delta\omega = 1/2$는 **가우시안 펄스에서만** 달성됩니다 — 이것이 시간-주파수 결합 의미에서 가장 컴팩트한 신호입니다.

### 11.3 실제적 의미

- 짧은 펄스 (작은 $\Delta t$)는 넓은 대역폭이 필요 (큰 $\Delta\omega$)
- 협대역 신호 (작은 $\Delta\omega$)는 긴 지속 시간이 필요 (큰 $\Delta t$)
- 시간과 주파수에서 동시에 매우 좁게 설계된 신호는 만들 수 없음
- 이것은 양자역학의 하이젠베르크 불확정성 원리와 유사한 신호 처리 개념

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Time-bandwidth product demonstration ---

fs = 10000
t = np.arange(-5, 5, 1 / fs)
dt = 1 / fs

def compute_time_bandwidth_product(x, t):
    """Compute RMS duration, RMS bandwidth, and their product."""
    # Normalize energy
    E = np.trapz(np.abs(x)**2, t)
    x_norm = np.abs(x)**2 / E

    # RMS duration
    t_mean = np.trapz(t * x_norm, t)
    t2_mean = np.trapz(t**2 * x_norm, t)
    delta_t = np.sqrt(t2_mean - t_mean**2)

    # Spectrum
    N = len(t)
    omega = np.fft.fftshift(np.fft.fftfreq(N, dt)) * 2 * np.pi
    X = np.fft.fftshift(np.fft.fft(x)) * dt
    X_norm = np.abs(X)**2 / np.trapz(np.abs(X)**2, omega) * (2 * np.pi)

    # RMS bandwidth
    omega_mean = np.trapz(omega * X_norm, omega)
    omega2_mean = np.trapz(omega**2 * X_norm, omega)
    delta_omega = np.sqrt(omega2_mean - omega_mean**2)

    return delta_t, delta_omega, delta_t * delta_omega

# Test with different Gaussian widths
sigmas = [0.1, 0.2, 0.5, 1.0, 2.0]

print("=== Time-Bandwidth Product for Gaussian Pulses ===")
print(f"{'sigma':>8} | {'Delta_t':>10} | {'Delta_omega':>12} | {'TBP':>10} | {'Limit (0.5)':>12}")
print("-" * 60)

delta_ts = []
delta_omegas = []

for sigma in sigmas:
    x = np.exp(-t**2 / (2 * sigma**2))
    dt_val, dw_val, tbp = compute_time_bandwidth_product(x, t)
    delta_ts.append(dt_val)
    delta_omegas.append(dw_val)
    print(f"{sigma:>8.1f} | {dt_val:>10.4f} | {dw_val:>12.4f} | {tbp:>10.4f} | {'0.5000':>12}")

print()

# Compare with rectangular pulse
print("=== TBP for Other Pulse Shapes ===")
shapes = {
    'Gaussian (sigma=0.5)': np.exp(-t**2 / (2 * 0.5**2)),
    'Rectangular (width=1)': np.where(np.abs(t) <= 0.5, 1.0, 0.0),
    'Triangular (width=2)': np.maximum(0, 1 - np.abs(t)),
    'Exponential (a=2)': np.where(t >= 0, np.exp(-2 * t), 0.0),
}

for name, x in shapes.items():
    dt_val, dw_val, tbp = compute_time_bandwidth_product(x, t)
    print(f"{name:>30}: TBP = {tbp:.4f} (limit = 0.5)")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for sigma in [0.2, 0.5, 1.0]:
    x = np.exp(-t**2 / (2 * sigma**2))
    axes[0].plot(t, x, linewidth=2, label=f'$\\sigma={sigma}$')

    N = len(t)
    omega = np.fft.fftshift(np.fft.fftfreq(N, dt)) * 2 * np.pi
    X = np.fft.fftshift(np.fft.fft(x)) * dt
    axes[1].plot(omega, np.abs(X), linewidth=2, label=f'$\\sigma={sigma}$')

axes[0].set_title('Gaussian Pulses (time domain)')
axes[0].set_xlabel('t')
axes[0].set_xlim([-3, 3])
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Spectra (frequency domain)')
axes[1].set_xlabel('$\\omega$ (rad/s)')
axes[1].set_xlim([-30, 30])
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Time-Bandwidth Tradeoff: Narrow in time = Wide in frequency', fontsize=12)
plt.tight_layout()
plt.savefig('time_bandwidth.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 12. Python 예제

### 12.1 종합 CTFT 분석 툴킷

```python
import numpy as np
import matplotlib.pyplot as plt

class CTFTAnalyzer:
    """Continuous-Time Fourier Transform analysis toolkit using FFT."""

    def __init__(self, fs=10000, duration=10.0):
        self.fs = fs
        self.dt = 1 / fs
        self.duration = duration
        self.t = np.arange(-duration / 2, duration / 2, self.dt)
        self.N = len(self.t)

    def ctft(self, x):
        """Approximate CTFT using FFT."""
        X = np.fft.fftshift(np.fft.fft(x)) * self.dt
        omega = np.fft.fftshift(np.fft.fftfreq(self.N, self.dt)) * 2 * np.pi
        return omega, X

    def ictft(self, X, omega=None):
        """Approximate inverse CTFT using IFFT."""
        X_unshifted = np.fft.ifftshift(X)
        x = np.fft.ifft(X_unshifted) * self.N * self.dt / (2 * np.pi)
        # Correct scaling
        x = np.real(x) / self.dt
        return x

    def energy(self, x):
        """Signal energy in time domain."""
        return np.trapz(np.abs(x)**2, self.t)

    def energy_spectral_density(self, X, omega):
        """Energy from frequency domain (Parseval's)."""
        return np.trapz(np.abs(X)**2, omega) / (2 * np.pi)

    def bandwidth_3db(self, X, omega):
        """Compute 3-dB bandwidth."""
        mag = np.abs(X)
        peak = np.max(mag)
        threshold = peak / np.sqrt(2)
        above = omega[mag >= threshold]
        if len(above) > 0:
            return above[-1] - above[0]
        return 0

    def analyze(self, x, title="Signal"):
        """Complete time-frequency analysis."""
        omega, X = self.ctft(x)

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))

        # Time domain
        axes[0, 0].plot(self.t, x, 'b-', linewidth=1)
        axes[0, 0].set_title(f'{title} — Time Domain')
        axes[0, 0].set_xlabel('t (s)')
        axes[0, 0].set_ylabel('x(t)')
        axes[0, 0].grid(True, alpha=0.3)

        # Magnitude spectrum
        axes[0, 1].plot(omega, np.abs(X), 'r-', linewidth=1)
        axes[0, 1].set_title('Magnitude Spectrum $|X(\\omega)|$')
        axes[0, 1].set_xlabel('$\\omega$ (rad/s)')
        axes[0, 1].grid(True, alpha=0.3)

        # Phase spectrum
        phase = np.angle(X)
        # Mask small values
        phase[np.abs(X) < np.max(np.abs(X)) * 0.01] = 0
        axes[0, 2].plot(omega, phase, 'g-', linewidth=0.5)
        axes[0, 2].set_title('Phase Spectrum $\\angle X(\\omega)$')
        axes[0, 2].set_xlabel('$\\omega$ (rad/s)')
        axes[0, 2].set_ylabel('Phase (rad)')
        axes[0, 2].grid(True, alpha=0.3)

        # Log magnitude (dB)
        mag_db = 20 * np.log10(np.abs(X) + 1e-12)
        axes[1, 0].plot(omega, mag_db, 'r-', linewidth=1)
        axes[1, 0].set_title('Log Magnitude (dB)')
        axes[1, 0].set_xlabel('$\\omega$ (rad/s)')
        axes[1, 0].set_ylabel('dB')
        axes[1, 0].grid(True, alpha=0.3)

        # Energy spectral density
        ESD = np.abs(X)**2
        axes[1, 1].plot(omega, ESD, 'm-', linewidth=1)
        axes[1, 1].set_title('Energy Spectral Density $|X(\\omega)|^2$')
        axes[1, 1].set_xlabel('$\\omega$ (rad/s)')
        axes[1, 1].grid(True, alpha=0.3)

        # Energy distribution
        E_total = self.energy(x)
        omega_pos = omega[omega >= 0]
        ESD_pos = ESD[omega >= 0]
        cum_energy = np.cumsum(ESD_pos) * (omega_pos[1] - omega_pos[0]) / np.pi
        axes[1, 2].plot(omega_pos, cum_energy / E_total * 100, 'k-', linewidth=2)
        axes[1, 2].axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90%')
        axes[1, 2].axhline(y=99, color='lightgray', linestyle='--', alpha=0.5, label='99%')
        axes[1, 2].set_title('Cumulative Energy (%)')
        axes[1, 2].set_xlabel('$\\omega$ (rad/s)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        # Print statistics
        E_freq = self.energy_spectral_density(X, omega)
        bw = self.bandwidth_3db(X, omega)
        print(f"=== {title} Analysis ===")
        print(f"  Energy (time):  {E_total:.6f}")
        print(f"  Energy (freq):  {E_freq:.6f}")
        print(f"  3-dB bandwidth: {bw:.2f} rad/s = {bw/(2*np.pi):.2f} Hz")

        plt.tight_layout()
        plt.savefig(f'ctft_analysis_{title.replace(" ", "_")}.png',
                    dpi=150, bbox_inches='tight')
        plt.show()


# Demonstrate
analyzer = CTFTAnalyzer(fs=10000, duration=10.0)

# Gaussian pulse
sigma = 0.1
x_gauss = np.exp(-analyzer.t**2 / (2 * sigma**2))
analyzer.analyze(x_gauss, "Gaussian Pulse")
```

### 12.2 주파수 영역에서의 필터링

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Frequency-domain filtering ---

fs = 8000
duration = 1.0
t = np.arange(0, duration, 1 / fs)
N = len(t)

# Create a signal with multiple frequency components
f_components = [200, 500, 1200, 2500]
amplitudes = [1.0, 0.8, 0.5, 0.3]
x = sum(a * np.sin(2 * np.pi * f * t) for a, f in zip(amplitudes, f_components))
x += 0.2 * np.random.randn(N)  # add noise

# Design filters in frequency domain
freq = np.fft.rfftfreq(N, 1 / fs)
X = np.fft.rfft(x)

# Lowpass: keep below 800 Hz
fc_lp = 800
H_lp = np.where(freq <= fc_lp, 1.0, 0.0)
# Smooth transition (Gaussian rolloff instead of brick wall)
H_lp_smooth = np.exp(-(np.maximum(0, freq - fc_lp))**2 / (2 * 50**2))

# Bandpass: 400-600 Hz
H_bp = np.exp(-((freq - 500)**2) / (2 * 60**2))

# Apply filters
y_lp = np.fft.irfft(X * H_lp_smooth, N)
y_bp = np.fft.irfft(X * H_bp, N)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Original
axes[0, 0].plot(t[:400], x[:400], 'b-', linewidth=0.8)
axes[0, 0].set_title('Original Signal')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(freq, 20 * np.log10(np.abs(X) / N + 1e-12), 'b-', linewidth=0.8)
for f in f_components:
    axes[0, 1].axvline(x=f, color='gray', linestyle=':', alpha=0.5)
axes[0, 1].set_title('Original Spectrum')
axes[0, 1].set_xlabel('Frequency (Hz)')
axes[0, 1].set_ylabel('dB')
axes[0, 1].grid(True, alpha=0.3)

# Lowpass filtered
axes[1, 0].plot(t[:400], y_lp[:400], 'r-', linewidth=0.8)
axes[1, 0].set_title(f'Lowpass Filtered ($f_c = {fc_lp}$ Hz)')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].grid(True, alpha=0.3)

Y_lp = np.fft.rfft(y_lp)
axes[1, 1].plot(freq, 20 * np.log10(np.abs(Y_lp) / N + 1e-12), 'r-', linewidth=0.8)
axes[1, 1].plot(freq, 20 * np.log10(H_lp_smooth + 1e-12), 'k--', linewidth=1.5,
                label='Filter')
axes[1, 1].set_title('Lowpass Output Spectrum')
axes[1, 1].set_xlabel('Frequency (Hz)')
axes[1, 1].set_ylabel('dB')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Bandpass filtered
axes[2, 0].plot(t[:400], y_bp[:400], 'g-', linewidth=0.8)
axes[2, 0].set_title('Bandpass Filtered (center = 500 Hz)')
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].grid(True, alpha=0.3)

Y_bp = np.fft.rfft(y_bp)
axes[2, 1].plot(freq, 20 * np.log10(np.abs(Y_bp) / N + 1e-12), 'g-', linewidth=0.8)
axes[2, 1].plot(freq, 20 * np.log10(H_bp + 1e-12), 'k--', linewidth=1.5,
                label='Filter')
axes[2, 1].set_title('Bandpass Output Spectrum')
axes[2, 1].set_xlabel('Frequency (Hz)')
axes[2, 1].set_ylabel('dB')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('freq_domain_filtering.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 12.3 성질 검증 스위트

```python
import numpy as np

# --- Systematic verification of Fourier transform properties ---

fs = 10000
duration = 10.0
dt = 1 / fs
t = np.arange(-duration / 2, duration / 2, dt)
N = len(t)

def fft_ctft(x):
    """Approximate CTFT using FFT."""
    return np.fft.fftshift(np.fft.fft(x)) * dt

omega = np.fft.fftshift(np.fft.fftfreq(N, dt)) * 2 * np.pi

# Test signal: Gaussian
sigma = 0.3
x = np.exp(-t**2 / (2 * sigma**2))
X = fft_ctft(x)

print("=== Fourier Transform Property Verification ===\n")

# 1. Linearity
a, b = 2.5, -1.3
x1 = np.exp(-t**2 / (2 * 0.3**2))
x2 = np.exp(-t**2 / (2 * 0.5**2))
X_linear_lhs = fft_ctft(a * x1 + b * x2)
X_linear_rhs = a * fft_ctft(x1) + b * fft_ctft(x2)
print(f"1. Linearity:       error = {np.max(np.abs(X_linear_lhs - X_linear_rhs)):.2e}")

# 2. Time shifting
t0 = 1.0
x_shifted = np.exp(-(t - t0)**2 / (2 * sigma**2))
X_shift_lhs = fft_ctft(x_shifted)
X_shift_rhs = X * np.exp(-1j * omega * t0)
print(f"2. Time shifting:   error = {np.max(np.abs(X_shift_lhs - X_shift_rhs)):.2e}")

# 3. Frequency shifting
omega0 = 10.0
x_modulated = x * np.exp(1j * omega0 * t)
X_mod_lhs = fft_ctft(x_modulated)
# Shift X by omega0
X_mod_rhs = np.interp(omega - omega0, omega, np.real(X)) + \
            1j * np.interp(omega - omega0, omega, np.imag(X))
# This is approximate due to interpolation; use a simpler check
# Check that the peak moved
peak_original = omega[np.argmax(np.abs(X))]
peak_modulated = omega[np.argmax(np.abs(X_mod_lhs))]
print(f"3. Freq shifting:   peak moved from {peak_original:.1f} to {peak_modulated:.1f} "
      f"(expected {peak_original + omega0:.1f})")

# 4. Time scaling
a_scale = 2.0
x_scaled = np.exp(-(a_scale * t)**2 / (2 * sigma**2))
X_scale_lhs = fft_ctft(x_scaled)
# Compare with analytical: sigma/a * sqrt(2pi) * exp(-sigma^2 * omega^2 / (2*a^2))
X_scale_analytical = (sigma / a_scale) * np.sqrt(2 * np.pi) * \
                     np.exp(-sigma**2 * omega**2 / (2 * a_scale**2))
# Normalize for comparison
ratio = np.abs(X_scale_lhs[N//2]) / X_scale_analytical[N//2] if X_scale_analytical[N//2] != 0 else 1
print(f"4. Time scaling:    peak ratio = {ratio:.4f} (expected ~1.0)")

# 5. Parseval's theorem
E_time = np.trapz(np.abs(x)**2, t)
E_freq = np.trapz(np.abs(X)**2, omega) / (2 * np.pi)
print(f"5. Parseval's:      E_time={E_time:.6f}, E_freq={E_freq:.6f}, "
      f"error={abs(E_time-E_freq):.2e}")

# 6. Convolution theorem
h = np.where(t >= 0, np.exp(-2 * t), 0.0)
H = fft_ctft(h)
y_conv = np.convolve(x, h, mode='full')[:N] * dt
Y_conv = fft_ctft(y_conv)
Y_mult = X * H
conv_error = np.max(np.abs(Y_conv - Y_mult)) / np.max(np.abs(Y_conv))
print(f"6. Convolution thm: relative error = {conv_error:.2e}")

# 7. Differentiation
# Numerical derivative
dx_dt = np.gradient(x, dt)
X_diff_lhs = fft_ctft(dx_dt)
X_diff_rhs = (1j * omega) * X
# Compare (ignoring edges where gradient is inaccurate)
mask = np.abs(omega) < 50
diff_error = np.max(np.abs(X_diff_lhs[mask] - X_diff_rhs[mask])) / np.max(np.abs(X_diff_rhs[mask]))
print(f"7. Differentiation: relative error = {diff_error:.2e}")
```

### 12.4 변환 쌍 갤러리

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Gallery of Fourier transform pairs ---

fs = 10000
t = np.arange(-5, 5, 1 / fs)
N = len(t)
dt = 1 / fs
omega = np.fft.fftshift(np.fft.fftfreq(N, dt)) * 2 * np.pi

def ctft(x):
    return np.fft.fftshift(np.fft.fft(x)) * dt

pairs = [
    ('$e^{-2t}u(t)$',
     np.where(t >= 0, np.exp(-2 * t), 0.0),
     '$\\frac{1}{2+j\\omega}$',
     1 / (2 + 1j * omega)),

    ('$e^{-2|t|}$',
     np.exp(-2 * np.abs(t)),
     '$\\frac{4}{4+\\omega^2}$',
     4 / (4 + omega**2)),

    ('$\\mathrm{rect}(t)$',
     np.where(np.abs(t) <= 0.5, 1.0, 0.0),
     '$\\mathrm{sinc}(\\omega/(2\\pi))$',
     np.sinc(omega / (2 * np.pi))),

    ('$\\mathrm{tri}(t)$',
     np.maximum(0, 1 - np.abs(t)),
     '$\\mathrm{sinc}^2(\\omega/(2\\pi))$',
     np.sinc(omega / (2 * np.pi))**2),

    ('$e^{-t^2/2}$',
     np.exp(-t**2 / 2),
     '$\\sqrt{2\\pi}e^{-\\omega^2/2}$',
     np.sqrt(2 * np.pi) * np.exp(-omega**2 / 2)),

    ('$te^{-t}u(t)$',
     np.where(t >= 0, t * np.exp(-t), 0.0),
     '$\\frac{1}{(1+j\\omega)^2}$',
     1 / (1 + 1j * omega)**2),
]

fig, axes = plt.subplots(len(pairs), 2, figsize=(14, 3 * len(pairs)))

for row, (t_label, x, f_label, X_analytical) in enumerate(pairs):
    # Numerical transform
    X_numerical = ctft(x)

    # Time domain
    axes[row, 0].plot(t, x, 'b-', linewidth=2)
    axes[row, 0].set_title(f'$x(t) = $ {t_label}')
    axes[row, 0].set_xlabel('t')
    axes[row, 0].set_xlim([-3, 5])
    axes[row, 0].grid(True, alpha=0.3)

    # Frequency domain (compare numerical and analytical)
    axes[row, 1].plot(omega, np.abs(X_numerical), 'r-', linewidth=2,
                     label='FFT (numerical)')
    axes[row, 1].plot(omega, np.abs(X_analytical), 'k--', linewidth=1.5,
                     label='Analytical', alpha=0.7)
    axes[row, 1].set_title(f'$X(\\omega) = $ {f_label}')
    axes[row, 1].set_xlabel('$\\omega$ (rad/s)')
    axes[row, 1].set_xlim([-30, 30])
    axes[row, 1].legend(fontsize=8)
    axes[row, 1].grid(True, alpha=0.3)

plt.suptitle('Fourier Transform Pair Gallery: Numerical vs Analytical', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('transform_pair_gallery.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 13. 요약

### 핵심 개념

| 개념 | 핵심 내용 |
|---------|----------|
| 푸리에 변환 | 비주기 신호를 연속 주파수 성분으로 분해 |
| 역변환 | 스펙트럼으로부터 신호 재구성 |
| 합성곱 정리 | 시간의 합성곱 = 주파수의 곱셈 |
| 곱셈 정리 | 시간의 곱셈 = 주파수의 합성곱 |
| 파르스발 정리 | 에너지는 시간 영역과 주파수 영역 사이에서 보존 |
| 시간-대역폭 곱 | $\Delta t \cdot \Delta\omega \geq 1/2$ (불확정성 원리) |

### 변환 성질 빠른 참조

```
시간 영역  ←→  주파수 영역
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x(t-t₀)     ←→  e^{-jωt₀} X(ω)      시간 이동 = 선형 위상
x(t)e^{jω₀t} ←→  X(ω-ω₀)            변조 = 스펙트럼 이동
x(at)        ←→  (1/|a|)X(ω/a)       압축 ↔ 팽창
dx/dt        ←→  jω X(ω)             미분 = × jω
x*h          ←→  X·H                  합성곱 = 곱셈
x·w          ←→  (1/2π) X*W           곱셈 = 합성곱
```

### 급수에서 변환으로: 큰 그림

```
    주기 신호                    비주기 신호
    ──────────────              ─────────────────
    푸리에 급수                  푸리에 변환
    cn (이산)                   X(ω) (연속)
    선 스펙트럼                  연속 스펙트럼
    Σ cn e^{jnω₀t}             ∫ X(ω) e^{jωt} dω/(2π)
         │                              │
         └──── T₀ → ∞로 취하면 ─────────┘
               cn → X(ω)dω/(2π)
               nω₀ → ω
               Σ → ∫
```

---

## 14. 연습 문제

### 연습 문제 1: 변환 계산

다음 신호들의 푸리에 변환을 해석적으로 계산하세요. Python으로 검증하세요.

1. $x(t) = e^{-3t}u(t) - e^{-5t}u(t)$
2. $x(t) = te^{-2t}u(t)$ (주파수 미분 성질 사용)
3. $x(t) = e^{-|t|}\cos(10t)$ (변조 + $e^{-|t|}$의 알려진 변환 사용)
4. $x(t) = \text{rect}(t) \cdot \cos(20\pi t)$ (변조된 직사각 펄스)

### 연습 문제 2: 성질 응용

$e^{-at}u(t) \leftrightarrow 1/(a + j\omega)$의 푸리에 변환과 변환 성질만을 사용하여 (적분으로 다시 유도하지 말고) 다음을 구하세요:

1. $\mathcal{F}\{e^{-a(t-3)}u(t-3)\}$ (시간 이동)
2. $\mathcal{F}\{e^{-at}u(t) \cdot e^{j5t}\}$ (주파수 이동)
3. $\mathcal{F}\{e^{-2at}u(2t)\}$ (스케일링; 주의 필요!)
4. $\mathcal{F}\{te^{-at}u(t)\}$ (주파수 미분)
5. $\mathcal{F}\{\frac{d}{dt}[e^{-at}u(t)]\}$ (시간 미분; $t=0$에서의 델타 주의)

### 연습 문제 3: 합성곱 정리 응용

1. 주파수 영역을 사용하여 $e^{-t}u(t) * e^{-2t}u(t)$를 계산하세요 (변환, 곱셈, 역변환). 직접 합성곱으로 검증하세요.
2. 입력이 $\omega_c = 5$ rad/s인 $x(t) = e^{-t}u(t)$일 때 1차 저역통과 시스템 $H(\omega) = 1/(1 + j\omega/\omega_c)$의 출력을 구하세요.
3. $x(t) = \text{sinc}(Bt)$ 신호가 차단 주파수 $\omega_c$의 이상적 저역통과 필터를 통과할 때, (a) $\omega_c > \pi B$? (b) $\omega_c < \pi B$? 일 때 출력은?

### 연습 문제 4: 파르스발 정리

1. 파르스발 정리를 사용하여 $x(t) = \frac{1}{1+t^2}$의 에너지를 계산하세요 (힌트: 먼저 $X(\omega)$를 구하세요).
2. 가우시안 펄스 $e^{-t^2/(2\sigma^2)}$의 에너지 중 주파수 대역 $|\omega| \leq 1/\sigma$ 내에 있는 비율은?
3. 너비 $T$의 직사각 펄스는 에너지가 $T$입니다. 99% 에너지 기준으로 얼마나 많은 대역폭이 필요한가요?

### 연습 문제 5: 필터 설계

1. 60 Hz 험(hum)과 광대역 잡음으로 오염된 신호에서 1 kHz 톤을 추출하는 주파수 영역 필터를 설계하세요. FFT를 사용하여 구현하고, 테스트 신호에 적용하고, 필터링 전후의 스펙트럼을 그리세요.
2. FFT를 사용하여 이상적 미분기 $H(\omega) = j\omega$를 구현하세요. 가우시안 펄스에 적용하고 해석적 도함수와 비교하세요.
3. 이상적 저역통과 필터를 계단 입력에 적용하면 왜 링잉이 발생하나요? 깁스 현상(Gibbs phenomenon) 측면에서 링잉을 정량화하세요.

### 연습 문제 6: 시간-대역폭 곱

1. 다음에 대한 시간-대역폭 곱을 계산하세요:
   - 직사각 펄스 $\text{rect}(t/T)$
   - 가우시안 펄스 $e^{-\pi t^2}$ (정확히 0.5가 나와야 함)
   - 1차 지수 $e^{-t}u(t)$
   - 상승 코사인 펄스 $\frac{1}{2}(1 + \cos(\pi t/T))$ ($|t| \leq T$에서)
2. 시간-대역폭 곱 효율로 이 펄스들을 순위 매기세요.
3. 레이더 시스템에는 어떤 펄스 형태를 선택하겠습니까? 통신 시스템에는요? 설명하세요.

### 연습 문제 7: 쌍대성

1. 알려진 쌍 $e^{-a|t|} \leftrightarrow 2a/(a^2 + \omega^2)$로부터 쌍대성을 사용하여 $\mathcal{F}\{1/(a^2 + t^2)\}$를 구하세요.
2. 알려진 쌍 $\text{rect}(t/\tau) \leftrightarrow \tau\,\text{sinc}(\omega\tau/(2\pi))$로부터 쌍대성을 사용하여 $\mathcal{F}\{\text{sinc}(Wt)\}$를 구하세요.
3. 두 결과 모두 수치적으로 검증하세요.

### 연습 문제 8: 종합 분석

여러 주파수 성분을 포함하는 음성, 음악 또는 합성 신호를 2초 녹음하거나 생성하세요.

1. 크기 스펙트럼과 위상 스펙트럼을 계산하고 그리세요
2. 지배적인 주파수 성분을 식별하세요
3. 하나의 성분을 추출하는 대역통과 필터를 설계하고 적용하세요
4. 필터링 전후의 에너지를 계산하세요; 통과 대역에 있는 비율은?
5. 스펙트럼 계산 전에 시간 영역 윈도잉(해밍)을 적용하세요. 윈도잉하지 않은 결과와 비교하세요.
6. 분석 창 길이를 변화시키고 시간-주파수 분해능 트레이드오프를 관찰하세요

---

## 15. 참고 문헌

1. Oppenheim, A. V. & Willsky, A. S. *Signals and Systems* (2nd ed.), Ch. 4-5. Prentice Hall, 1997.
2. Haykin, S. & Van Veen, B. *Signals and Systems* (2nd ed.), Ch. 4-5. Wiley, 2003.
3. Bracewell, R. N. *The Fourier Transform and Its Applications* (3rd ed.). McGraw-Hill, 2000.
4. Lathi, B. P. & Green, R. A. *Linear Systems and Signals* (3rd ed.), Ch. 7. Oxford University Press, 2018.
5. Mallat, S. *A Wavelet Tour of Signal Processing* (3rd ed.), Ch. 2. Academic Press, 2009.

---

[이전: 03. 푸리에 급수와 응용](./03_Fourier_Series_and_Applications.md) | [다음: 05. 표본화와 복원](./05_Sampling_and_Reconstruction.md) | [개요](./00_Overview.md)
