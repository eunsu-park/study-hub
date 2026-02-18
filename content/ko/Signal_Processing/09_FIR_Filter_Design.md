# FIR 필터 설계(FIR Filter Design)

## 학습 목표

- 디지털 필터 설계의 사양과 용어를 이해한다
- FIR 필터 설계를 위한 창함수법(Window Method)을 마스터하고 창함수들을 비교한다
- 주파수 표본화(Frequency Sampling) 및 최적 등리플(Parks-McClellan) 설계 방법을 학습한다
- 선형 위상(Linear Phase) FIR 필터 유형과 그 제약 조건을 이해한다
- Python의 `scipy.signal` 모듈을 사용하여 FIR 필터를 설계한다
- 직접 합성곱(Direct Convolution), 중첩 가산법(Overlap-Add), 중첩 저장법(Overlap-Save)을 이용한 FIR 필터링을 구현한다
- 전이 대역폭(Transition Width), 저지대역 감쇠(Stopband Attenuation), 연산 비용 측면에서 설계 방법들을 비교한다

---

## 목차

1. [FIR 필터 설계 사양](#1-fir-필터-설계-사양)
2. [이상적인 필터와 임펄스 응답](#2-이상적인-필터와-임펄스-응답)
3. [창함수법](#3-창함수법)
4. [창함수](#4-창함수)
5. [카이저 창함수 설계](#5-카이저-창함수-설계)
6. [주파수 표본화법](#6-주파수-표본화법)
7. [최적 등리플 설계: Parks-McClellan 알고리즘](#7-최적-등리플-설계-parks-mcclellan-알고리즘)
8. [선형 위상 FIR 필터](#8-선형-위상-fir-필터)
9. [설계 방법 비교](#9-설계-방법-비교)
10. [FIR 필터 구현](#10-fir-필터-구현)
11. [Python 구현](#11-python-구현)
12. [연습 문제](#12-연습-문제)

---

## 1. FIR 필터 설계 사양

### 1.1 필터 용어

디지털 필터는 **주파수 응답(Frequency Response)** $H(e^{j\omega})$로 특성화되며, 이는 신호의 각 주파수 성분이 어떻게 변형되는지를 나타낸다. 주요 사양은 다음과 같다:

**주파수 대역:**
- **통과대역(Passband)**: 최소한의 왜곡으로 통과시켜야 할 주파수, $0 \leq \omega \leq \omega_p$
- **저지대역(Stopband)**: 감쇠시켜야 할 주파수, $\omega_s \leq \omega \leq \pi$
- **전이 대역(Transition Band)**: 통과대역과 저지대역 사이의 영역, $\omega_p < \omega < \omega_s$

**허용 오차(Tolerances):**
- **통과대역 리플(Passband Ripple)** $\delta_1$: 통과대역에서 단위로부터의 최대 편차
- **저지대역 감쇠(Stopband Attenuation)** $\delta_2$: 저지대역에서의 최대 크기

```
진폭 응답 사양(Magnitude Response Specification)
|H(e^jω)|
    ↑
1+δ₁ ─ ─ ─ ─┐
    │        │
  1 ├────────┤
    │        │
1-δ₁ ─ ─ ─ ─┘        ╲
    │                    ╲
  δ₂ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┬─────────
    │                      │
  0 ├──────┬──────┬────────┬─────→ ω
    0     ωp    ωs        π

         통과대역│전이 │ 저지대역
                │대역 │
```

### 1.2 데시벨 사양

실제로 필터 사양은 종종 데시벨(dB)로 표현된다:

$$A_p = -20\log_{10}(1 - \delta_1) \quad \text{(통과대역 리플, dB)}$$

$$A_s = -20\log_{10}(\delta_2) \quad \text{(저지대역 감쇠, dB)}$$

일반적인 사양 예시:
- 통과대역 리플: $A_p \leq 0.1$ dB ($\delta_1 \approx 0.0115$ 의미)
- 저지대역 감쇠: $A_s \geq 60$ dB ($\delta_2 = 0.001$ 의미)

### 1.3 FIR vs IIR 트레이드오프

```
┌────────────────────────────────────────────────────────┐
│              FIR vs IIR 비교                            │
├──────────────────┬─────────────────┬───────────────────┤
│ 특성             │ FIR             │ IIR               │
├──────────────────┼─────────────────┼───────────────────┤
│ 안정성           │ 항상 안정       │ 불안정 가능       │
│ 선형 위상        │ 쉽게 달성       │ 불가능            │
│ 필터 차수        │ 높음            │ 낮음              │
│ 연산량           │ 곱셈 많음       │ 곱셈 적음         │
│ 설계 방법        │ Window, PM, FS  │ 아날로그 원형     │
│ 군지연           │ 일정            │ 비일정            │
│ 반올림 오차      │ 덜 민감         │ 더 민감           │
│ 구현             │ 피드백 없음     │ 피드백 필요       │
└──────────────────┴─────────────────┴───────────────────┘
```

차수 $M$의 FIR 필터는 다음의 전달 함수(Transfer Function)를 갖는다:

$$H(z) = \sum_{n=0}^{M} h[n] z^{-n}$$

출력은 유한 합성곱(Finite Convolution)으로 표현된다:

$$y[n] = \sum_{k=0}^{M} h[k] x[n-k]$$

---

## 2. 이상적인 필터와 임펄스 응답

### 2.1 이상적인 저역통과 필터

이상적인 저역통과 필터(Ideal Lowpass Filter)의 주파수 응답은:

$$H_d(e^{j\omega}) = \begin{cases} 1, & |\omega| \leq \omega_c \\ 0, & \omega_c < |\omega| \leq \pi \end{cases}$$

임펄스 응답은 역 DTFT를 통해 구한다:

$$h_d[n] = \frac{1}{2\pi} \int_{-\pi}^{\pi} H_d(e^{j\omega}) e^{j\omega n} d\omega = \frac{\sin(\omega_c n)}{\pi n}$$

이는 다음과 같은 특성을 가진 **sinc 함수**이다:
- 지속 시간이 무한하다 (비인과적, 실현 불가)
- $n = 0$에 대해 대칭이다

### 2.2 다른 이상적인 필터 유형

**이상적인 고역통과 필터(Ideal Highpass Filter):**

$$h_d[n] = \delta[n] - \frac{\sin(\omega_c n)}{\pi n}$$

**이상적인 대역통과 필터(Ideal Bandpass Filter)** ($\omega_{c1} < |\omega| < \omega_{c2}$):

$$h_d[n] = \frac{\sin(\omega_{c2} n)}{\pi n} - \frac{\sin(\omega_{c1} n)}{\pi n}$$

**이상적인 대역저지 필터(Ideal Bandstop Filter):**

$$h_d[n] = \delta[n] - \frac{\sin(\omega_{c2} n)}{\pi n} + \frac{\sin(\omega_{c1} n)}{\pi n}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def ideal_lowpass(omega_c, M):
    """Compute ideal lowpass filter impulse response (truncated)."""
    n = np.arange(-M, M + 1)
    h = np.zeros_like(n, dtype=float)

    # Handle n = 0 case (sinc(0) = omega_c / pi)
    center = M
    h[center] = omega_c / np.pi

    # Non-zero indices
    nonzero = np.concatenate([np.arange(0, center), np.arange(center + 1, 2 * M + 1)])
    h[nonzero] = np.sin(omega_c * n[nonzero]) / (np.pi * n[nonzero])

    return n, h

# Plot ideal lowpass impulse responses for different cutoff frequencies
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, wc, title in zip(axes, [np.pi/4, np.pi/2, 3*np.pi/4],
                           ['ωc = π/4', 'ωc = π/2', 'ωc = 3π/4']):
    n, h = ideal_lowpass(wc, 30)
    ax.stem(n, h, linefmt='b-', markerfmt='bo', basefmt='k-')
    ax.set_xlabel('n')
    ax.set_ylabel('h[n]')
    ax.set_title(f'Ideal LP Impulse Response ({title})')
    ax.set_xlim(-32, 32)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ideal_lowpass_responses.png', dpi=150)
plt.close()
```

---

## 3. 창함수법

### 3.1 기본 원리

창함수법(Window Method)은 FIR 설계에서 가장 단순한 방법이다. 기본 아이디어는:

1. 이상적인 (무한) 임펄스 응답 $h_d[n]$에서 시작한다
2. 길이 $N = M + 1$인 유한 지속 시간 창함수 $w[n]$을 곱한다
3. 필터를 인과적(Causal)으로 만들기 위해 결과를 이동시킨다

$$h[n] = h_d[n - M/2] \cdot w[n], \quad n = 0, 1, \ldots, M$$

주파수 영역에서 이 곱셈은 합성곱(Convolution)에 해당한다:

$$H(e^{j\omega}) = \frac{1}{2\pi} H_d(e^{j\omega}) * W(e^{j\omega})$$

### 3.2 창함수 적용의 효과

창함수의 주파수 응답 $W(e^{j\omega})$은 다음을 결정한다:

1. **주엽 폭(Mainlobe Width)**: 전이 대역폭을 제어한다 -- 주엽이 넓을수록 전이 대역이 넓어진다
2. **부엽 레벨(Sidelobe Level)**: 저지대역 감쇠를 결정한다 -- 부엽이 높을수록 저지대역 감쇠가 작아진다
3. **부엽 감소율(Sidelobe Decay Rate)**: 주파수에 따라 부엽이 얼마나 빨리 감소하는지

```
창함수 스펙트럼 특성(Window Spectrum Properties)
|W(e^jω)|
    ↑
    │   ╱╲        주엽(Mainlobe)
    │  ╱  ╲       폭 = Δω_ML
    │ ╱    ╲
    │╱      ╲  ╱╲
    │        ╲╱  ╲  ╱╲       최대 부엽
    │         │   ╲╱  ╲      레벨 (dB)
    │         │    │   ╲╱╲
    ├─────────┼────┼─────┼──→ ω
    0      Δω_ML/2          π
```

### 3.3 깁스 현상(Gibbs Phenomenon)

직사각형 창(Rectangular Window)으로 이상적인 임펄스 응답을 절단하면 깁스 현상(Gibbs Phenomenon)이 발생하여 필터 차수 $M$에 관계없이 대역 경계에서 약 9%의 오버슈트(약 $-21$ dB 저지대역 감쇠)가 발생한다. 필터 차수 $M$을 늘리면 전이 폭은 좁아지지만 리플 진폭은 줄어들지 않는다.

```python
def windowed_lowpass(omega_c, M, window_type='rectangular'):
    """Design lowpass FIR filter using window method."""
    n = np.arange(0, M + 1)
    alpha = M / 2  # Delay for linear phase

    # Ideal lowpass impulse response (shifted for causality)
    h_d = np.zeros(M + 1)
    for i in range(M + 1):
        if n[i] == alpha:
            h_d[i] = omega_c / np.pi
        else:
            h_d[i] = np.sin(omega_c * (n[i] - alpha)) / (np.pi * (n[i] - alpha))

    # Apply window
    if window_type == 'rectangular':
        w = np.ones(M + 1)
    elif window_type == 'hamming':
        w = signal.windows.hamming(M + 1)
    elif window_type == 'hanning':
        w = signal.windows.hann(M + 1)
    elif window_type == 'blackman':
        w = signal.windows.blackman(M + 1)
    else:
        w = np.ones(M + 1)

    h = h_d * w
    return h

# Demonstrate Gibbs phenomenon with rectangular window
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
omega_c = np.pi / 2

for idx, M in enumerate([10, 20, 50, 100]):
    ax = axes[idx // 2, idx % 2]
    h = windowed_lowpass(omega_c, M, 'rectangular')

    # Frequency response
    w_freq, H = signal.freqz(h, worN=2048)
    H_dB = 20 * np.log10(np.abs(H) + 1e-15)

    ax.plot(w_freq / np.pi, H_dB, 'b-', linewidth=1.5)
    ax.axhline(-21, color='r', linestyle='--', alpha=0.5, label='-21 dB (Gibbs)')
    ax.axvline(0.5, color='g', linestyle='--', alpha=0.5, label='ωc = π/2')
    ax.set_xlabel('Normalized Frequency (×π rad/sample)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f'Rectangular Window, M = {M}')
    ax.set_ylim(-80, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Gibbs Phenomenon: Stopband ripple remains ~-21 dB', fontsize=12)
plt.tight_layout()
plt.savefig('gibbs_phenomenon.png', dpi=150)
plt.close()
```

---

## 4. 창함수

### 4.1 일반적인 창함수

각 창함수는 주엽 폭과 부엽 레벨 사이의 트레이드오프를 조절한다. 더 매끄러운 창함수일수록 주엽이 넓지만 부엽은 낮다.

#### 직사각형 창(Rectangular Window)

$$w[n] = 1, \quad 0 \leq n \leq M$$

- **주엽 폭**: $\Delta\omega_\text{ML} = 4\pi / (M+1)$
- **최대 부엽**: $-13$ dB
- **최소 저지대역 감쇠**: $-21$ dB
- **전이 폭**: $\Delta\omega \approx 0.92 \cdot 2\pi / (M+1)$

#### 한(Hann) 창 (해닝(Hanning) 창)

$$w[n] = 0.5 - 0.5\cos\left(\frac{2\pi n}{M}\right), \quad 0 \leq n \leq M$$

- **주엽 폭**: $\Delta\omega_\text{ML} = 8\pi / (M+1)$
- **최대 부엽**: $-31$ dB
- **최소 저지대역 감쇠**: $-44$ dB
- **전이 폭**: $\Delta\omega \approx 3.11 \cdot 2\pi / (M+1)$

#### 해밍 창(Hamming Window)

$$w[n] = 0.54 - 0.46\cos\left(\frac{2\pi n}{M}\right), \quad 0 \leq n \leq M$$

- **주엽 폭**: $\Delta\omega_\text{ML} = 8\pi / (M+1)$
- **최대 부엽**: $-41$ dB
- **최소 저지대역 감쇠**: $-53$ dB
- **전이 폭**: $\Delta\omega \approx 3.32 \cdot 2\pi / (M+1)$

#### 블랙만 창(Blackman Window)

$$w[n] = 0.42 - 0.5\cos\left(\frac{2\pi n}{M}\right) + 0.08\cos\left(\frac{4\pi n}{M}\right), \quad 0 \leq n \leq M$$

- **주엽 폭**: $\Delta\omega_\text{ML} = 12\pi / (M+1)$
- **최대 부엽**: $-57$ dB
- **최소 저지대역 감쇠**: $-74$ dB
- **전이 폭**: $\Delta\omega \approx 5.56 \cdot 2\pi / (M+1)$

### 4.2 창함수 비교표

```
┌─────────────┬──────────────┬───────────────┬───────────────────┐
│ 창함수      │ 최대 부엽    │ 최소 저지대역 │ 근사 전이폭       │
│             │ 레벨 (dB)   │ 감쇠 (dB)     │ (rad)             │
├─────────────┼──────────────┼───────────────┼───────────────────┤
│ Rectangular │ -13          │ -21           │ 0.92·(2π/M)       │
│ Hann        │ -31          │ -44           │ 3.11·(2π/M)       │
│ Hamming     │ -41          │ -53           │ 3.32·(2π/M)       │
│ Blackman    │ -57          │ -74           │ 5.56·(2π/M)       │
│ Kaiser(β=6) │ -44          │ -60           │ adjustable        │
│ Kaiser(β=9) │ -69          │ -90           │ adjustable        │
└─────────────┴──────────────┴───────────────┴───────────────────┘
```

### 4.3 창함수와 스펙트럼 시각화

```python
def compare_windows(M=50):
    """Compare common window functions in time and frequency domains."""
    windows = {
        'Rectangular': np.ones(M + 1),
        'Hann': signal.windows.hann(M + 1),
        'Hamming': signal.windows.hamming(M + 1),
        'Blackman': signal.windows.blackman(M + 1),
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    colors = ['blue', 'orange', 'green', 'red']

    # Time domain
    ax = axes[0]
    for (name, w), color in zip(windows.items(), colors):
        ax.plot(np.arange(M + 1), w, color=color, linewidth=2, label=name)
    ax.set_xlabel('Sample n')
    ax.set_ylabel('w[n]')
    ax.set_title('Window Functions (Time Domain)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Frequency domain (log magnitude)
    ax = axes[1]
    nfft = 4096
    for (name, w), color in zip(windows.items(), colors):
        W = np.fft.fft(w, nfft)
        W_shift = np.fft.fftshift(W)
        freq = np.linspace(-np.pi, np.pi, nfft)
        W_dB = 20 * np.log10(np.abs(W_shift) / np.max(np.abs(W_shift)) + 1e-15)
        ax.plot(freq / np.pi, W_dB, color=color, linewidth=1.5, label=name)

    ax.set_xlabel('Normalized Frequency (×π rad/sample)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Window Spectra (Frequency Domain)')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-100, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('window_comparison.png', dpi=150)
    plt.close()

compare_windows(M=50)
```

### 4.4 창함수법 설계 절차

1. **사양 결정**: $\omega_p$, $\omega_s$, $\delta_1$, $\delta_2$
2. **창함수 유형 선택**: 요구되는 저지대역 감쇠를 기반으로 선택
3. **전이 폭 계산**: $\Delta\omega = \omega_s - \omega_p$
4. **필터 차수 결정**: 창함수의 전이 폭 공식으로부터 $M$ 결정
5. **차단 주파수 설정**: $\omega_c = (\omega_p + \omega_s) / 2$
6. **이상적인 임펄스 응답 계산**: $M/2$만큼 이동된 $h_d[n]$ 계산
7. **창함수 적용**: $h[n] = h_d[n] \cdot w[n]$ 계산

```python
def design_fir_window(wp, ws, delta_s, window_type='hamming'):
    """
    Design FIR lowpass filter using window method.

    Parameters:
        wp: passband edge (normalized, 0 to pi)
        ws: stopband edge (normalized, 0 to pi)
        delta_s: stopband attenuation (linear)
        window_type: 'rectangular', 'hann', 'hamming', or 'blackman'

    Returns:
        h: filter coefficients
        M: filter order
    """
    # Transition width
    delta_w = ws - wp

    # Cutoff frequency (midpoint)
    wc = (wp + ws) / 2

    # Determine filter order M from window type
    transition_coefficients = {
        'rectangular': 0.92,
        'hann':        3.11,
        'hamming':     3.32,
        'blackman':    5.56,
    }

    coeff = transition_coefficients[window_type]
    M = int(np.ceil(coeff * 2 * np.pi / delta_w))
    if M % 2 == 1:
        M += 1  # Ensure even order for Type I filter

    # Compute windowed ideal lowpass
    n = np.arange(0, M + 1)
    alpha = M / 2

    h_d = np.where(
        n == alpha,
        wc / np.pi,
        np.sin(wc * (n - alpha)) / (np.pi * (n - alpha))
    )

    # Apply window
    windows = {
        'rectangular': np.ones(M + 1),
        'hann': signal.windows.hann(M + 1),
        'hamming': signal.windows.hamming(M + 1),
        'blackman': signal.windows.blackman(M + 1),
    }
    w = windows[window_type]
    h = h_d * w

    return h, M

# Example: design lowpass filter
wp = 0.3 * np.pi  # Passband edge
ws = 0.5 * np.pi  # Stopband edge

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
window_types = ['rectangular', 'hann', 'hamming', 'blackman']

for ax, wtype in zip(axes.flat, window_types):
    h, M = design_fir_window(wp, ws, 0.001, wtype)
    w_freq, H = signal.freqz(h, worN=4096)
    H_dB = 20 * np.log10(np.abs(H) + 1e-15)

    ax.plot(w_freq / np.pi, H_dB, 'b-', linewidth=1.5)
    ax.axvline(0.3, color='g', linestyle='--', alpha=0.7, label=f'ωp = 0.3π')
    ax.axvline(0.5, color='r', linestyle='--', alpha=0.7, label=f'ωs = 0.5π')
    ax.set_xlabel('Normalized Frequency (×π rad/sample)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f'{wtype.capitalize()} Window, M = {M}')
    ax.set_ylim(-100, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('FIR Lowpass Filter Design: Window Method Comparison', fontsize=13)
plt.tight_layout()
plt.savefig('fir_window_method.png', dpi=150)
plt.close()
```

---

## 5. 카이저 창함수 설계

### 5.1 카이저 창함수 정의

카이저 창함수(Kaiser Window)는 주엽 폭과 부엽 레벨 사이의 트레이드오프를 제어하는 **연속 조절 가능한** 파라미터 $\beta$를 제공한다:

$$w[n] = \frac{I_0\left(\beta\sqrt{1 - \left(\frac{2n}{M} - 1\right)^2}\right)}{I_0(\beta)}, \quad 0 \leq n \leq M$$

여기서 $I_0(\cdot)$는 제1종 0차 변형 베셀 함수(Zeroth-Order Modified Bessel Function of the First Kind)이다.

### 5.2 카이저의 경험적 공식

원하는 저지대역 감쇠 $A_s$ (dB)가 주어졌을 때:

**파라미터 $\beta$:**

$$\beta = \begin{cases} 0.1102(A_s - 8.7), & A_s > 50 \\ 0.5842(A_s - 21)^{0.4} + 0.07886(A_s - 21), & 21 \leq A_s \leq 50 \\ 0, & A_s < 21 \end{cases}$$

**필터 차수 $M$:**

$$M = \left\lceil \frac{A_s - 7.95}{2.285 \cdot \Delta\omega} \right\rceil$$

여기서 $\Delta\omega = \omega_s - \omega_p$는 전이 폭이다.

### 5.3 카이저 창함수 설계 절차

```python
def kaiser_fir_design(wp, ws, As_dB):
    """
    Design FIR lowpass filter using Kaiser window.

    Parameters:
        wp: passband edge (normalized, 0 to pi)
        ws: stopband edge (normalized, 0 to pi)
        As_dB: stopband attenuation in dB

    Returns:
        h: filter coefficients
        M: filter order
        beta: Kaiser window parameter
    """
    # Transition width
    delta_w = ws - wp

    # Cutoff frequency
    wc = (wp + ws) / 2

    # Kaiser's empirical formula for beta
    if As_dB > 50:
        beta = 0.1102 * (As_dB - 8.7)
    elif As_dB >= 21:
        beta = 0.5842 * (As_dB - 21)**0.4 + 0.07886 * (As_dB - 21)
    else:
        beta = 0.0

    # Filter order
    M = int(np.ceil((As_dB - 7.95) / (2.285 * delta_w)))
    if M % 2 == 1:
        M += 1  # Ensure even order for Type I

    # Ideal lowpass impulse response
    n = np.arange(0, M + 1)
    alpha = M / 2
    h_d = np.where(
        n == alpha,
        wc / np.pi,
        np.sin(wc * (n - alpha)) / (np.pi * (n - alpha))
    )

    # Kaiser window
    w = signal.windows.kaiser(M + 1, beta)

    h = h_d * w
    return h, M, beta

# Design filters with different stopband attenuations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
specs = [
    (0.3 * np.pi, 0.5 * np.pi, 30),
    (0.3 * np.pi, 0.5 * np.pi, 50),
    (0.3 * np.pi, 0.5 * np.pi, 70),
    (0.3 * np.pi, 0.5 * np.pi, 90),
]

for ax, (wp, ws, As) in zip(axes.flat, specs):
    h, M, beta = kaiser_fir_design(wp, ws, As)
    w_freq, H = signal.freqz(h, worN=4096)
    H_dB = 20 * np.log10(np.abs(H) + 1e-15)

    ax.plot(w_freq / np.pi, H_dB, 'b-', linewidth=1.5)
    ax.axhline(-As, color='r', linestyle='--', alpha=0.5, label=f'-{As} dB')
    ax.set_xlabel('Normalized Frequency (×π rad/sample)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f'Kaiser: As={As} dB, β={beta:.2f}, M={M}')
    ax.set_ylim(-As - 20, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Kaiser Window FIR Design: Varying Stopband Attenuation', fontsize=13)
plt.tight_layout()
plt.savefig('kaiser_design.png', dpi=150)
plt.close()
```

### 5.4 scipy.signal.kaiserord 사용

```python
# scipy provides convenient functions for Kaiser window design
from scipy.signal import kaiserord, firwin

# Specification
fs = 8000  # Sample rate (Hz)
f_pass = 1000  # Passband edge (Hz)
f_stop = 1500  # Stopband edge (Hz)
A_stop = 60  # Stopband attenuation (dB)

# Compute transition width in normalized frequency
nyquist = fs / 2
width = (f_stop - f_pass) / nyquist  # Normalized transition width

# Kaiser window parameters
numtaps, beta = kaiserord(A_stop, width)
if numtaps % 2 == 0:
    numtaps += 1  # firwin needs odd number of taps for Type I

# Design the filter
cutoff = (f_pass + f_stop) / 2 / nyquist  # Normalized cutoff
h = firwin(numtaps, cutoff, window=('kaiser', beta))

print(f"Filter order: {numtaps - 1}")
print(f"Number of taps: {numtaps}")
print(f"Kaiser beta: {beta:.4f}")

# Frequency response
w, H = signal.freqz(h, worN=4096, fs=fs)
H_dB = 20 * np.log10(np.abs(H) + 1e-15)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Magnitude response
axes[0].plot(w, H_dB, 'b-', linewidth=1.5)
axes[0].axhline(-A_stop, color='r', linestyle='--', label=f'-{A_stop} dB')
axes[0].axvline(f_pass, color='g', linestyle='--', alpha=0.5, label=f'fp={f_pass} Hz')
axes[0].axvline(f_stop, color='orange', linestyle='--', alpha=0.5, label=f'fs={f_stop} Hz')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Magnitude (dB)')
axes[0].set_title('Magnitude Response')
axes[0].set_ylim(-100, 5)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Impulse response
axes[1].stem(h, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[1].set_xlabel('Sample n')
axes[1].set_ylabel('h[n]')
axes[1].set_title(f'Impulse Response (M={numtaps - 1})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kaiser_scipy.png', dpi=150)
plt.close()
```

---

## 6. 주파수 표본화법

### 6.1 원리

주파수 표본화법(Frequency Sampling Method)은 $N$개의 등간격 주파수 점에서 원하는 주파수 응답을 지정하고 역 DFT를 통해 필터 계수를 계산하여 FIR 필터를 설계하는 방법이다:

$$H[k] = H(e^{j 2\pi k / N}), \quad k = 0, 1, \ldots, N-1$$

$$h[n] = \frac{1}{N} \sum_{k=0}^{N-1} H[k] e^{j 2\pi k n / N}, \quad n = 0, 1, \ldots, N-1$$

### 6.2 전이 표본(Transition Samples)

핵심 아이디어는 **전이 대역 표본을 최적화**하여 저지대역 리플을 줄일 수 있다는 것이다. 1에서 0으로 급격히 전환하는 대신, 중간 값 $T_1, T_2, \ldots$를 전이 대역에 배치한다.

```
주파수 표본 H[k]:
H[k]
  ↑
  1 ─ ● ─ ● ─ ● ─ ●
  │                   ╲
T₁│                    ●   ← 전이 표본 (최적화됨)
  │                      ╲
  0 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─● ─ ● ─ ● ─ ● ─ ●
  └────────────────────────────────────────→ k
```

### 6.3 구현

```python
def freq_sampling_lowpass(N, cutoff_bin, transition_values=None):
    """
    FIR filter design via frequency sampling.

    Parameters:
        N: number of samples (filter length)
        cutoff_bin: passband ends at bin index cutoff_bin
        transition_values: list of transition band sample values

    Returns:
        h: filter coefficients (real)
    """
    H = np.zeros(N, dtype=complex)

    # Passband (bins 0 to cutoff_bin)
    H[:cutoff_bin + 1] = 1.0

    # Mirror for negative frequencies (ensure real h[n])
    H[N - cutoff_bin:] = 1.0

    # Transition band samples
    if transition_values is not None:
        for i, T in enumerate(transition_values):
            H[cutoff_bin + 1 + i] = T
            H[N - cutoff_bin - 1 - i] = T

    # Inverse DFT to get filter coefficients
    h = np.real(np.fft.ifft(H))

    # Circular shift for causal filter
    h = np.roll(h, N // 2)

    return h

# Compare with and without optimized transition samples
N = 33
cutoff_bin = 5

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Without transition samples
h1 = freq_sampling_lowpass(N, cutoff_bin)
w1, H1 = signal.freqz(h1, worN=4096)

# With optimized transition sample
h2 = freq_sampling_lowpass(N, cutoff_bin, transition_values=[0.5])
w2, H2 = signal.freqz(h2, worN=4096)

# With two optimized transition samples
h3 = freq_sampling_lowpass(N, cutoff_bin, transition_values=[0.59, 0.11])
w3, H3 = signal.freqz(h3, worN=4096)

axes[0].plot(w1 / np.pi, 20 * np.log10(np.abs(H1) + 1e-15), label='No transition')
axes[0].plot(w2 / np.pi, 20 * np.log10(np.abs(H2) + 1e-15), label='T=[0.5]')
axes[0].plot(w3 / np.pi, 20 * np.log10(np.abs(H3) + 1e-15), label='T=[0.59, 0.11]')
axes[0].set_xlabel('Normalized Frequency (×π rad/sample)')
axes[0].set_ylabel('Magnitude (dB)')
axes[0].set_title('Frequency Sampling: Effect of Transition Values')
axes[0].set_ylim(-100, 5)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].stem(np.arange(N), h3, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[1].set_xlabel('Sample n')
axes[1].set_ylabel('h[n]')
axes[1].set_title('Impulse Response (Optimized Transition)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('freq_sampling.png', dpi=150)
plt.close()
```

---

## 7. 최적 등리플 설계: Parks-McClellan 알고리즘

### 7.1 미니맥스 최적화(Minimax Optimization)

Parks-McClellan (PM) 알고리즘은 Remez 교환 알고리즘(Remez Exchange Algorithm)으로도 알려져 있으며, 주파수 영역에서 최대 가중 오차를 최소화하는 FIR 필터를 설계한다:

$$\min_{h} \max_{\omega \in \text{bands}} \left| W(\omega) \left[ H_d(\omega) - H(\omega) \right] \right|$$

여기서:
- $H_d(\omega)$는 원하는 주파수 응답
- $H(\omega) = \sum_{n=0}^{M} h[n] e^{-j\omega n}$는 실제 응답
- $W(\omega)$는 가중 함수

### 7.2 등리플 특성(Equiripple Property)

체비쇼프 정리(Chebyshev Theorem)는 최적 해가 **등리플(Equiripple)** 오차를 나타냄을 보장한다 -- 오차는 지정된 모든 대역에서 동일한 진폭으로 $\pm\delta$ 사이를 진동한다:

```
등리플 오차 거동:
          통과대역                      저지대역
    ↑    +δ₁ ─ ─ ─ ╱╲    ╱╲
H(ω)│          ╱  ╲ ╱  ╲
    │    ─────╱────╲╱────╲──── 1.0
    │
    │                                     ╱╲    ╱╲    ╱╲
    │     -δ₁ ─ ─ ─               ──╲──╱──╲──╱──╲──╱──
    │                          +δ₂ ─ ─╲╱─ ─ ╲╱─ ─ ╲╱─
    │                          -δ₂ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
    └──────────────────────────────────────────→ ω
```

### 7.3 Remez 교환 알고리즘

알고리즘은 반복적으로:

1. $(M/2 + 2)$개의 극점 주파수(Extremal Frequency) 점으로 **초기화**한다
2. 등리플 거동을 만들어내는 다항식을 구하는 보간 문제를 **풀이**한다
3. 오차가 최대인 새로운 극점을 **탐색**한다
4. 현재 극점 집합을 새 집합으로 **교환**한다
5. 수렴할 때까지 (극점이 안정될 때까지) **반복**한다

### 7.4 scipy.signal.remez를 이용한 구현

```python
from scipy.signal import remez, freqz

def parks_mcclellan_lowpass(numtaps, fp, fs, fs_hz=1.0, weight=None):
    """
    Design lowpass FIR filter using Parks-McClellan algorithm.

    Parameters:
        numtaps: number of filter taps (M + 1)
        fp: passband edge frequency (Hz)
        fs_freq: stopband edge frequency (Hz)
        fs_hz: sampling frequency (Hz)
        weight: weighting [W_pass, W_stop]

    Returns:
        h: filter coefficients
    """
    if weight is None:
        weight = [1, 1]

    # Bands: [0, fp, fs, fs_hz/2]
    bands = [0, fp, fs, fs_hz / 2]
    desired = [1, 0]  # Passband = 1, Stopband = 0

    h = remez(numtaps, bands, desired, weight=weight, fs=fs_hz)
    return h

# Example: lowpass filter design
fs_hz = 8000
numtaps = 51
fp = 1000  # Passband edge (Hz)
f_stop = 1500  # Stopband edge (Hz)

# Different weighting schemes
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

weights = [
    ([1, 1], 'Equal weight'),
    ([1, 10], 'Stopband 10× weight'),
    ([10, 1], 'Passband 10× weight'),
    ([1, 100], 'Stopband 100× weight'),
]

for ax, (w, title) in zip(axes.flat, weights):
    h = parks_mcclellan_lowpass(numtaps, fp, f_stop, fs_hz, weight=w)
    freq, H = signal.freqz(h, worN=4096, fs=fs_hz)
    H_dB = 20 * np.log10(np.abs(H) + 1e-15)

    ax.plot(freq, H_dB, 'b-', linewidth=1.5)
    ax.axvline(fp, color='g', linestyle='--', alpha=0.5, label=f'fp={fp} Hz')
    ax.axvline(f_stop, color='r', linestyle='--', alpha=0.5, label=f'fs={f_stop} Hz')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f'PM Design: {title}')
    ax.set_ylim(-80, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Parks-McClellan Filter Design with Different Weights', fontsize=13)
plt.tight_layout()
plt.savefig('parks_mcclellan.png', dpi=150)
plt.close()
```

### 7.5 대역통과 필터 예시

```python
# Bandpass filter using Parks-McClellan
fs_hz = 8000
numtaps = 101

# Bandpass: pass 1000-2000 Hz, stop below 500 and above 2500
bands = [0, 500, 1000, 2000, 2500, fs_hz / 2]
desired = [0, 1, 0]
weight = [1, 1, 1]

h_bp = remez(numtaps, bands, desired, weight=weight, fs=fs_hz)

# Frequency response
freq, H_bp = signal.freqz(h_bp, worN=4096, fs=fs_hz)
H_bp_dB = 20 * np.log10(np.abs(H_bp) + 1e-15)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(freq, H_bp_dB, 'b-', linewidth=1.5)
axes[0].axvspan(1000, 2000, alpha=0.1, color='green', label='Passband')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Magnitude (dB)')
axes[0].set_title('Bandpass FIR (Parks-McClellan)')
axes[0].set_ylim(-80, 5)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(freq, np.abs(H_bp), 'b-', linewidth=1.5)
axes[1].axvspan(1000, 2000, alpha=0.1, color='green', label='Passband')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('|H(f)|')
axes[1].set_title('Magnitude Response (Linear)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bandpass_pm.png', dpi=150)
plt.close()
```

---

## 8. 선형 위상 FIR 필터

### 8.1 선형 위상의 중요성

필터가 다음과 같이 표현될 수 있을 때 **선형 위상(Linear Phase)**을 갖는다고 한다:

$$H(e^{j\omega}) = |H(e^{j\omega})| \cdot e^{-j\omega \tau}$$

여기서 $\tau$는 일정한 군지연(Group Delay)이다. 선형 위상은 모든 주파수 성분이 동일한 시간만큼 지연됨을 의미하며, 파형 모양을 보존한다.

**군지연:**

$$\tau(\omega) = -\frac{d\angle H(e^{j\omega})}{d\omega} = \text{상수} = \frac{M}{2}$$

### 8.2 대칭 조건

선형 위상 FIR 필터는 대칭 또는 반대칭 임펄스 응답을 필요로 한다:

**대칭(Symmetric)** (타입 I 및 II): $h[n] = h[M - n]$

**반대칭(Antisymmetric)** (타입 III 및 IV): $h[n] = -h[M - n]$

### 8.3 선형 위상 FIR 필터의 네 가지 유형

```
┌────────┬──────────────┬──────────────────────┬───────────────────┐
│ 유형   │ 대칭성       │ 차수 M               │ 적합한 필터 유형  │
├────────┼──────────────┼──────────────────────┼───────────────────┤
│ I      │ 대칭         │ 짝수                 │ 모든 필터 유형    │
│ II     │ 대칭         │ 홀수                 │ 저역통과, 대역통과│
│ III    │ 반대칭       │ 짝수                 │ 대역통과, 힐버트, │
│        │              │                      │ 미분기            │
│ IV     │ 반대칭       │ 홀수                 │ 고역통과, 대역통과│
│        │              │                      │ 힐버트            │
└────────┴──────────────┴──────────────────────┴───────────────────┘
```

**타입 I** (짝수 차수, 대칭):
- $H(e^{j0}) = \sum h[n]$ (제약 없음)
- $H(e^{j\pi}) = \sum (-1)^n h[n]$ (제약 없음)
- 모든 필터 유형에 적합

**타입 II** (홀수 차수, 대칭):
- $H(e^{j\pi}) = 0$ 항상 성립
- 고역통과(Highpass) 또는 대역저지(Bandstop) 필터에 사용 불가

**타입 III** (짝수 차수, 반대칭):
- $H(e^{j0}) = 0$ 및 $H(e^{j\pi}) = 0$ 항상 성립
- 대역통과, 힐버트 변환(Hilbert Transform), 미분기(Differentiator)에 적합

**타입 IV** (홀수 차수, 반대칭):
- $H(e^{j0}) = 0$ 항상 성립
- 저역통과 필터에 사용 불가

### 8.4 진폭 응답 공식

타입 I 필터 (짝수 $M$, 대칭)의 경우, 주파수 응답은:

$$H(e^{j\omega}) = e^{-j\omega M/2} A(\omega)$$

여기서 **진폭 응답(Amplitude Response)**은:

$$A(\omega) = h[M/2] + 2 \sum_{k=1}^{M/2} h[M/2 - k] \cos(k\omega)$$

이는 실수 값 함수로, 통과대역/저지대역 거동 분석이 용이하다.

```python
def analyze_linear_phase(h, filter_type=None):
    """Analyze linear phase properties of FIR filter."""
    M = len(h) - 1

    # Check symmetry
    is_symmetric = np.allclose(h, h[::-1])
    is_antisymmetric = np.allclose(h, -h[::-1])

    if is_symmetric and M % 2 == 0:
        ftype = "Type I"
    elif is_symmetric and M % 2 == 1:
        ftype = "Type II"
    elif is_antisymmetric and M % 2 == 0:
        ftype = "Type III"
    elif is_antisymmetric and M % 2 == 1:
        ftype = "Type IV"
    else:
        ftype = "Not linear phase"

    # Frequency response
    w, H = signal.freqz(h, worN=4096)

    # Group delay
    w_gd, gd = signal.group_delay((h, [1]), w=4096)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Magnitude
    axes[0, 0].plot(w / np.pi, 20 * np.log10(np.abs(H) + 1e-15), 'b-')
    axes[0, 0].set_ylabel('Magnitude (dB)')
    axes[0, 0].set_title(f'{ftype}: Magnitude Response')
    axes[0, 0].grid(True, alpha=0.3)

    # Phase
    axes[0, 1].plot(w / np.pi, np.unwrap(np.angle(H)), 'r-')
    axes[0, 1].set_ylabel('Phase (radians)')
    axes[0, 1].set_title(f'{ftype}: Phase Response')
    axes[0, 1].grid(True, alpha=0.3)

    # Group delay
    axes[1, 0].plot(w_gd / np.pi, gd, 'g-')
    axes[1, 0].axhline(M / 2, color='r', linestyle='--', alpha=0.5,
                         label=f'M/2 = {M/2}')
    axes[1, 0].set_ylabel('Group Delay (samples)')
    axes[1, 0].set_title('Group Delay')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Impulse response (with symmetry visualization)
    axes[1, 1].stem(h, linefmt='b-', markerfmt='bo', basefmt='k-')
    axes[1, 1].axvline(M / 2, color='r', linestyle='--', alpha=0.5,
                        label='Center of symmetry')
    axes[1, 1].set_ylabel('h[n]')
    axes[1, 1].set_title(f'Impulse Response (M={M})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    for ax in [axes[1, 0], axes[1, 1]]:
        ax.set_xlabel('Normalized Frequency (×π)' if ax == axes[1, 0] else 'Sample n')
    for ax in [axes[0, 0], axes[0, 1]]:
        ax.set_xlabel('Normalized Frequency (×π)')

    plt.suptitle(f'Linear Phase Analysis: {ftype}', fontsize=13)
    plt.tight_layout()
    return ftype

# Example: Type I lowpass
h_type1 = firwin(51, 0.4)  # 50th order, symmetric
ftype = analyze_linear_phase(h_type1)
plt.savefig('linear_phase_analysis.png', dpi=150)
plt.close()
print(f"Detected filter type: {ftype}")
```

---

## 9. 설계 방법 비교

### 9.1 직접 비교

```python
def compare_design_methods(numtaps=51, wp=0.3, ws=0.5):
    """
    Compare window, Kaiser, and Parks-McClellan methods for the same specs.
    """
    wc_norm = (wp + ws) / 2  # Cutoff for window methods

    # Method 1: Hamming window
    h_hamming = firwin(numtaps, wc_norm, window='hamming')

    # Method 2: Kaiser window (60 dB attenuation)
    h_kaiser = firwin(numtaps, wc_norm, window=('kaiser', 5.65))

    # Method 3: Parks-McClellan
    h_pm = remez(numtaps, [0, wp / 2, ws / 2, 0.5], [1, 0])

    # Frequency responses
    methods = [
        ('Hamming Window', h_hamming),
        ('Kaiser Window (β=5.65)', h_kaiser),
        ('Parks-McClellan', h_pm),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['blue', 'green', 'red']

    # Overlay magnitude responses
    ax = axes[0, 0]
    for (name, h), color in zip(methods, colors):
        w, H = signal.freqz(h, worN=4096)
        ax.plot(w / np.pi, 20 * np.log10(np.abs(H) + 1e-15),
                color=color, linewidth=1.5, label=name)
    ax.set_xlabel('Normalized Frequency (×π)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Magnitude Response Comparison')
    ax.set_ylim(-80, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Passband detail
    ax = axes[0, 1]
    for (name, h), color in zip(methods, colors):
        w, H = signal.freqz(h, worN=4096)
        ax.plot(w / np.pi, 20 * np.log10(np.abs(H) + 1e-15),
                color=color, linewidth=1.5, label=name)
    ax.set_xlabel('Normalized Frequency (×π)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Passband Detail')
    ax.set_xlim(0, wp + 0.05)
    ax.set_ylim(-1, 0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Transition band detail
    ax = axes[1, 0]
    for (name, h), color in zip(methods, colors):
        w, H = signal.freqz(h, worN=4096)
        ax.plot(w / np.pi, np.abs(H), color=color, linewidth=1.5, label=name)
    ax.set_xlabel('Normalized Frequency (×π)')
    ax.set_ylabel('|H(f)|')
    ax.set_title('Transition Band Detail')
    ax.set_xlim(wp - 0.05, ws + 0.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Impulse responses
    ax = axes[1, 1]
    n = np.arange(numtaps)
    for (name, h), color in zip(methods, colors):
        ax.plot(n, h, color=color, linewidth=1.5, marker='o',
                markersize=3, label=name)
    ax.set_xlabel('Sample n')
    ax.set_ylabel('h[n]')
    ax.set_title('Impulse Responses')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'FIR Design Comparison (N={numtaps})', fontsize=13)
    plt.tight_layout()
    plt.savefig('design_comparison.png', dpi=150)
    plt.close()

compare_design_methods(numtaps=51, wp=0.3, ws=0.5)
```

### 9.2 설계 방법 요약

```
┌──────────────────┬──────────────┬────────────────┬──────────────────┐
│ 방법             │ 최적성       │ 제어           │ 최적 용도        │
├──────────────────┼──────────────┼────────────────┼──────────────────┤
│ 창함수법         │ 준최적       │ 제한적         │ 빠른 설계,       │
│                  │              │ (이산 창함수   │ 교육용           │
│                  │              │  선택)         │                  │
├──────────────────┼──────────────┼────────────────┼──────────────────┤
│ 카이저 창함수    │ 준최적에 근접│ 연속 β         │ 제어와 단순성의  │
│                  │              │ 파라미터       │ 좋은 균형        │
├──────────────────┼──────────────┼────────────────┼──────────────────┤
│ 주파수 표본화    │ 준최적       │ 전이 표본      │ 하드웨어 구현,   │
│                  │              │                │ DFT 기반 필터    │
├──────────────────┼──────────────┼────────────────┼──────────────────┤
│ Parks-McClellan  │ 최적         │ 가중 함수,     │ 실제 제품,       │
│                  │ (미니맥스)   │ 다중 대역      │ 엄격한 사양      │
└──────────────────┴──────────────┴────────────────┴──────────────────┘
```

**핵심 인사이트**: 동일한 필터 차수에서 Parks-McClellan은 가장 작은 최대 오차(등리플)를 달성한다. 창함수법은 더 높은 주파수에서 단조 증가하는 저지대역 감쇠를 갖는 필터를 생성하는데, 이는 높은 주파수에서의 감쇠를 "낭비"하는 셈이다.

---

## 10. FIR 필터 구현

### 10.1 직접 합성곱(Direct Convolution)

가장 간단한 구현 방법은 출력을 직접 합성곱으로 계산한다:

$$y[n] = \sum_{k=0}^{M} h[k] x[n-k]$$

**복잡도**: $M$차 필터로 $N$개의 표본을 필터링하는 데 $O(MN)$.

```python
def fir_direct(x, h):
    """Direct-form FIR filter implementation."""
    M = len(h)
    N = len(x)
    y = np.zeros(N + M - 1)

    for n in range(N + M - 1):
        for k in range(M):
            if 0 <= n - k < N:
                y[n] += h[k] * x[n - k]

    return y

# Using numpy's convolve (optimized)
x = np.random.randn(1000)
h = firwin(51, 0.3)

y_direct = np.convolve(x, h, mode='full')
print(f"Input length: {len(x)}, Filter length: {len(h)}, Output length: {len(y_direct)}")
```

### 10.2 FFT 기반 필터링

긴 신호의 경우 FFT 기반 합성곱이 훨씬 빠르다:

$$y = \text{IFFT}[\text{FFT}(x) \cdot \text{FFT}(h)]$$

**복잡도**: $O(MN)$ 대신 $O(N \log N)$.

FFT가 더 빨라지는 교차점은 일반적으로 $M \approx 50-100$이다.

### 10.3 중첩 가산법(Overlap-Add Method)

입력이 세그먼트로 도착하는 실시간 또는 블록 기반 처리를 위한 방법:

1. 입력 $x[n]$을 길이 $L$의 비중첩 블록으로 **분할**한다
2. 각 블록을 길이 $L + M$으로 **영 패딩**한다
3. 각 블록을 필터 $h[n]$과 **FFT 합성곱**한다
4. 인접 출력 블록들을 **중첩 가산**한다 (마지막 $M$개의 표본이 중첩됨)

```python
def overlap_add(x, h, L=256):
    """
    Overlap-add FIR filtering.

    Parameters:
        x: input signal
        h: FIR filter coefficients
        L: block size

    Returns:
        y: filtered output
    """
    M = len(h)
    N = len(x)
    N_fft = L + M - 1

    # Zero-pad filter to FFT length
    H = np.fft.fft(h, N_fft)

    # Output array
    y = np.zeros(N + M - 1)

    # Process blocks
    num_blocks = int(np.ceil(N / L))

    for i in range(num_blocks):
        # Extract block
        start = i * L
        end = min(start + L, N)
        x_block = np.zeros(L)
        x_block[:end - start] = x[start:end]

        # FFT convolution
        X_block = np.fft.fft(x_block, N_fft)
        y_block = np.real(np.fft.ifft(X_block * H))

        # Overlap-add
        y[start:start + N_fft] += y_block

    return y[:N + M - 1]

# Verify overlap-add produces same result as direct convolution
x = np.random.randn(5000)
h = firwin(101, 0.3)

y_direct = np.convolve(x, h)
y_ola = overlap_add(x, h, L=256)

error = np.max(np.abs(y_direct - y_ola))
print(f"Maximum error between direct and overlap-add: {error:.2e}")
```

### 10.4 중첩 저장법(Overlap-Save Method)

중첩 가산법의 대안:

1. 입력 블록을 $M-1$개의 표본만큼 **중첩**시킨다 (각 블록에 이전 블록의 $M-1$개 표본 포함)
2. 원형 합성곱(Circular Convolution)을 이용하여 **FFT 합성곱**한다 (길이 $L+M-1$)
3. 각 출력 블록의 첫 $M-1$개 표본을 **버린다** (원형 감김에 의해 오염됨)
4. 나머지 표본들을 **연결**한다

```python
def overlap_save(x, h, L=256):
    """
    Overlap-save FIR filtering.

    Parameters:
        x: input signal
        h: FIR filter coefficients
        L: block size (output samples per block)

    Returns:
        y: filtered output
    """
    M = len(h)
    N = len(x)
    N_fft = L + M - 1

    # Zero-pad filter to FFT length
    H = np.fft.fft(h, N_fft)

    # Prepend M-1 zeros to input
    x_padded = np.concatenate([np.zeros(M - 1), x])
    N_padded = len(x_padded)

    # Output
    y = np.zeros(N + M - 1)

    output_idx = 0
    block_start = 0

    while block_start < N_padded:
        # Extract overlapping block of length N_fft
        block_end = min(block_start + N_fft, N_padded)
        x_block = np.zeros(N_fft)
        x_block[:block_end - block_start] = x_padded[block_start:block_end]

        # Circular convolution via FFT
        X_block = np.fft.fft(x_block, N_fft)
        y_block = np.real(np.fft.ifft(X_block * H))

        # Keep only valid samples (discard first M-1)
        valid = y_block[M - 1:]
        valid_len = min(len(valid), N + M - 1 - output_idx)
        y[output_idx:output_idx + valid_len] = valid[:valid_len]

        output_idx += L
        block_start += L  # Advance by L (not N_fft)

    return y[:N + M - 1]

# Verify overlap-save
y_ols = overlap_save(x, h, L=256)
error_ols = np.max(np.abs(y_direct[:len(y_ols)] - y_ols[:len(y_direct)]))
print(f"Maximum error between direct and overlap-save: {error_ols:.2e}")
```

### 10.5 성능 비교

```python
import time

def benchmark_methods(signal_lengths, filter_order=100):
    """Benchmark different FIR filtering methods."""
    h = firwin(filter_order + 1, 0.3)
    results = {name: [] for name in ['Direct (np.convolve)', 'Overlap-Add', 'scipy.fftconvolve']}

    for N in signal_lengths:
        x = np.random.randn(N)

        # Direct convolution
        t0 = time.time()
        _ = np.convolve(x, h)
        results['Direct (np.convolve)'].append(time.time() - t0)

        # Overlap-add
        t0 = time.time()
        _ = overlap_add(x, h, L=1024)
        results['Overlap-Add'].append(time.time() - t0)

        # scipy FFT convolve
        from scipy.signal import fftconvolve
        t0 = time.time()
        _ = fftconvolve(x, h)
        results['scipy.fftconvolve'].append(time.time() - t0)

    return results

signal_lengths = [1000, 5000, 10000, 50000, 100000, 500000]
results = benchmark_methods(signal_lengths)

plt.figure(figsize=(10, 6))
for name, times in results.items():
    plt.loglog(signal_lengths, times, 'o-', linewidth=2, label=name)
plt.xlabel('Signal Length N')
plt.ylabel('Time (seconds)')
plt.title('FIR Filtering: Performance Comparison')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('fir_performance.png', dpi=150)
plt.close()
```

---

## 11. Python 구현

### 11.1 완전한 설계 예시

```python
import numpy as np
from scipy import signal
from scipy.signal import firwin, remez, freqz, kaiserord
import matplotlib.pyplot as plt

def complete_fir_design(fs, f_pass, f_stop, A_stop_dB, method='kaiser'):
    """
    Complete FIR lowpass filter design workflow.

    Parameters:
        fs: sampling frequency (Hz)
        f_pass: passband edge (Hz)
        f_stop: stopband edge (Hz)
        A_stop_dB: stopband attenuation (dB)
        method: 'kaiser', 'hamming', or 'remez'

    Returns:
        h: filter coefficients
        info: design information dictionary
    """
    nyq = fs / 2

    if method == 'kaiser':
        width = (f_stop - f_pass) / nyq
        numtaps, beta = kaiserord(A_stop_dB, width)
        if numtaps % 2 == 0:
            numtaps += 1
        cutoff = (f_pass + f_stop) / 2 / nyq
        h = firwin(numtaps, cutoff, window=('kaiser', beta))
        info = {'method': 'Kaiser', 'numtaps': numtaps, 'beta': beta}

    elif method == 'hamming':
        # Estimate order from transition width
        delta_f = (f_stop - f_pass) / fs
        numtaps = int(np.ceil(3.32 / delta_f)) + 1
        if numtaps % 2 == 0:
            numtaps += 1
        cutoff = (f_pass + f_stop) / 2 / nyq
        h = firwin(numtaps, cutoff, window='hamming')
        info = {'method': 'Hamming', 'numtaps': numtaps}

    elif method == 'remez':
        # Start with estimated order, iterate if needed
        delta_f = (f_stop - f_pass) / fs
        numtaps = int(np.ceil((-20 * np.log10(np.sqrt(0.001 * 0.001)) - 13) /
                              (14.6 * delta_f))) + 1
        if numtaps % 2 == 0:
            numtaps += 1

        delta_s = 10**(-A_stop_dB / 20)
        weight = [1, 1 / delta_s]  # Weight stopband more
        h = remez(numtaps, [0, f_pass, f_stop, nyq], [1, 0],
                  weight=[1, 1], fs=fs)
        info = {'method': 'Parks-McClellan', 'numtaps': numtaps}

    return h, info

# Design and analyze
fs = 16000
f_pass = 2000
f_stop = 3000
A_stop = 60

fig, axes = plt.subplots(3, 2, figsize=(14, 14))
methods = ['kaiser', 'hamming', 'remez']

for i, method in enumerate(methods):
    h, info = complete_fir_design(fs, f_pass, f_stop, A_stop, method=method)

    # Frequency response
    w, H = freqz(h, worN=4096, fs=fs)
    H_dB = 20 * np.log10(np.abs(H) + 1e-15)

    # Magnitude
    ax = axes[i, 0]
    ax.plot(w, H_dB, 'b-', linewidth=1.5)
    ax.axhline(-A_stop, color='r', linestyle='--', alpha=0.5, label=f'-{A_stop} dB')
    ax.axvline(f_pass, color='g', linestyle='--', alpha=0.5)
    ax.axvline(f_stop, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f"{info['method']} (N={info['numtaps']})")
    ax.set_ylim(-100, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Group delay
    w_gd, gd = signal.group_delay((h, [1]), w=4096, fs=fs)
    ax = axes[i, 1]
    ax.plot(w_gd, gd, 'g-', linewidth=1.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Group Delay (samples)')
    ax.set_title(f"{info['method']}: Group Delay")
    ax.grid(True, alpha=0.3)

plt.suptitle('Complete FIR Filter Design Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('complete_fir_design.png', dpi=150)
plt.close()
```

### 11.2 실제 신호 필터링

```python
def demonstrate_filtering():
    """Demonstrate FIR filtering on a multi-tone signal."""
    fs = 8000
    t = np.arange(0, 1, 1/fs)

    # Create signal: 200 Hz + 800 Hz + 2000 Hz + noise
    x = (np.sin(2 * np.pi * 200 * t) +
         0.5 * np.sin(2 * np.pi * 800 * t) +
         0.3 * np.sin(2 * np.pi * 2000 * t) +
         0.1 * np.random.randn(len(t)))

    # Design lowpass filter (cutoff ~1000 Hz)
    h = firwin(101, 1000, fs=fs, window='hamming')

    # Apply filter
    y = signal.lfilter(h, 1, x)

    # Compensate for group delay
    delay = (len(h) - 1) // 2

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Time domain
    t_ms = t * 1000
    axes[0].plot(t_ms[:500], x[:500], 'b-', alpha=0.7, label='Original')
    axes[0].plot(t_ms[:500], y[delay:delay + 500], 'r-', linewidth=2,
                  label='Filtered (delay compensated)')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Time Domain')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Spectrum comparison
    from scipy.fft import fft, fftfreq

    N = len(x)
    freq = fftfreq(N, 1/fs)[:N//2]
    X = 2 / N * np.abs(fft(x))[:N//2]
    Y = 2 / N * np.abs(fft(y))[:N//2]

    axes[1].plot(freq, X, 'b-', alpha=0.7, label='Original')
    axes[1].plot(freq, Y, 'r-', linewidth=2, label='Filtered')
    axes[1].axvline(1000, color='g', linestyle='--', alpha=0.5, label='Cutoff')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('Frequency Domain')
    axes[1].set_xlim(0, fs / 2)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Spectrogram
    f_spec, t_spec, Sxx = signal.spectrogram(x, fs, nperseg=256)
    f_spec_f, t_spec_f, Syy = signal.spectrogram(y, fs, nperseg=256)

    axes[2].pcolormesh(t_spec * 1000, f_spec, 10 * np.log10(Sxx + 1e-15),
                        shading='gouraud', cmap='viridis')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_title('Spectrogram (Original Signal)')

    plt.tight_layout()
    plt.savefig('fir_filtering_demo.png', dpi=150)
    plt.close()

demonstrate_filtering()
```

### 11.3 다중 대역 필터 설계

```python
# Design a multi-band filter using firwin2
from scipy.signal import firwin2

# Arbitrary magnitude response
numtaps = 201
freq_points = [0, 0.1, 0.15, 0.3, 0.35, 0.5, 0.55, 0.7, 0.75, 1.0]
gain_points = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]

h_multiband = firwin2(numtaps, freq_points, gain_points)

w, H = freqz(h_multiband, worN=4096)
H_dB = 20 * np.log10(np.abs(H) + 1e-15)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Desired vs actual
axes[0].plot(np.array(freq_points) / 2, gain_points, 'ro-',
             linewidth=2, label='Desired', markersize=8)
axes[0].plot(w / np.pi / 2, np.abs(H), 'b-', linewidth=1.5, label='Actual')
axes[0].set_xlabel('Normalized Frequency')
axes[0].set_ylabel('Magnitude')
axes[0].set_title('Multi-band Filter: Magnitude')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(w / np.pi, H_dB, 'b-', linewidth=1.5)
axes[1].set_xlabel('Normalized Frequency (×π)')
axes[1].set_ylabel('Magnitude (dB)')
axes[1].set_title('Multi-band Filter: dB Scale')
axes[1].set_ylim(-80, 5)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multiband_fir.png', dpi=150)
plt.close()
```

---

## 12. 연습 문제

### 연습 문제 1: 창함수 선택

다음 사양으로 저역통과 FIR 필터를 설계하라:
- 표본화 주파수: $f_s = 10000$ Hz
- 통과대역 경계: $f_p = 1500$ Hz
- 저지대역 경계: $f_s = 2000$ Hz
- 최소 저지대역 감쇠: $60$ dB

(a) 어떤 창함수 유형이 저지대역 사양을 충족할 수 있는가?

(b) 각 가능한 창함수에 대해 필요한 필터 차수 $M$을 계산하라.

(c) `scipy.signal.firwin`을 사용하여 필터를 설계하고 사양이 충족되는지 검증하라.

(d) 통과대역 리플과 전이 폭 측면에서 결과 필터들을 비교하라.

### 연습 문제 2: 카이저 창함수 설계

디지털 필터가 다음을 만족해야 한다:
- 통과대역: $0 \leq f \leq 3$ kHz, 리플 $\leq 0.1$ dB
- 저지대역: $f \geq 4$ kHz, 감쇠 $\geq 50$ dB
- 표본화율: $f_s = 20$ kHz

(a) 경험적 공식을 이용하여 카이저 창함수 파라미터 $\beta$를 계산하라.

(b) 최소 필터 차수 $M$을 계산하라.

(c) 필터를 설계하고 진폭 응답, 위상 응답, 군지연을 그래프로 나타내라.

(d) 실제 통과대역 리플과 저지대역 감쇠를 측정하여 설계가 사양을 충족하는지 검증하라.

### 연습 문제 3: Parks-McClellan 대역통과 필터

Parks-McClellan 알고리즘을 사용하여 대역통과 FIR 필터를 설계하라:
- $f_s = 44100$ Hz (CD 오디오 레이트)
- 저지대역 1: $0$에서 $800$ Hz (감쇠 $\geq 40$ dB)
- 통과대역: $1000$에서 $3000$ Hz (리플 $\leq 0.5$ dB)
- 저지대역 2: $3500$에서 $22050$ Hz (감쇠 $\geq 40$ dB)

(a) 양쪽 저지대역에서 등리플을 위한 적절한 가중치를 선택하라.

(b) 최소 필터 차수를 결정하라.

(c) 진폭 응답을 그래프로 나타내고 사양을 검증하라.

(d) 처프 신호(100 Hz에서 10 kHz로 스윕)에 필터를 적용하고 필터링 전후의 스펙트로그램을 표시하라.

### 연습 문제 4: 중첩 가산법 구현

중첩 가산법을 사용하는 실시간 FIR 필터 프로세서를 구현하라:

(a) 블록 단위로 오디오를 처리하는 `StreamingFIRFilter` 클래스를 작성하라.

(b) 64, 256, 1024, 4096 표본의 블록 크기로 테스트하라.

(c) 출력이 직접 합성곱 (`np.convolve`)과 일치하는지 검증하라.

(d) 블록 크기의 함수로 블록당 처리 시간을 측정하고 그래프로 나타내라.

(e) 44100 Hz에서 10초 오디오 신호에 200탭 필터를 적용할 때 최적 블록 크기를 찾아라.

### 연습 문제 5: 선형 위상 제약

(a) 타입 II FIR 필터 (홀수 차수, 대칭 계수)가 항상 $H(e^{j\pi}) = 0$임을 증명하라. 왜 이것이 고역통과 필터 설계에 부적합한가?

(b) 차단 주파수 $0.6\pi$에서 고역통과 FIR 필터를 다음으로 설계하라:
   - 타입 I (짝수 차수)
   - 타입 IV (홀수 차수, 반대칭)

   결과를 비교하라. 어느 것이 $\omega = \pi$ 근처에서 더 좋은 성능을 보이는가?

(c) 차수 30의 힐버트 변환 필터(타입 III)를 설계하라. 진폭 및 위상 응답을 그래프로 나타내라. $\omega = 0$과 $\omega = \pi$에서 어떤 일이 발생하는가?

### 연습 문제 6: 비교 연구

사양: $f_s = 16000$ Hz, $f_p = 2000$ Hz, $f_\text{stop} = 2500$ Hz, $A_s = 50$ dB:

(a) (i) 해밍 창함수, (ii) 카이저 창함수, (iii) Parks-McClellan을 사용하여 필터를 설계하라.

(b) 각 방법에 대해 다음을 보고하라: 필터 차수, 통과대역 리플 (dB), 실제 저지대역 감쇠 (dB), 전이 폭 (Hz).

(c) 4개의 서브플롯이 포함된 단일 그림을 생성하라: 진폭 응답 오버레이, 통과대역 상세, 임펄스 응답, 영점-극점 선도(Pole-Zero Plot).

(d) 어떤 방법이 가장 낮은 필터 차수에서 최선의 결과를 제공하는가? 답을 정당화하라.

### 연습 문제 7: 최소 위상 FIR 필터

(a) 해밍 창함수법을 사용하여 차수 40의 선형 위상 저역통과 FIR 필터를 설계하라.

(b) 켑스트럼 방법(Cepstral Method)을 사용하여 최소 위상(Minimum-Phase) FIR 필터로 변환하라:
   - 켑스트럼 $\hat{h}[n] = \text{IFFT}(\log|\text{FFT}(h)|)$ 계산
   - 최소 위상 버전 구성

(c) 두 필터의 진폭 응답, 위상 응답, 군지연을 비교하라.

(d) 언제 최소 위상을 선형 위상보다 선호하는가? 예시를 들어 논의하라.

---

## 참고 문헌

1. **Oppenheim, A. V., & Schafer, R. W. (2010).** *Discrete-Time Signal Processing* (3rd ed.). Pearson. Chapters 7-8.
2. **Proakis, J. G., & Manolakis, D. G. (2007).** *Digital Signal Processing* (4th ed.). Pearson. Chapter 10.
3. **Parks, T. W., & Burrus, C. S. (1987).** *Digital Filter Design*. Wiley.
4. **Kaiser, J. F. (1974).** "Nonrecursive Digital Filter Design Using the I0-sinh Window Function." *Proceedings of the 1974 IEEE International Symposium on Circuits and Systems*.
5. **McClellan, J. H., Parks, T. W., & Rabiner, L. R. (1973).** "A Computer Program for Designing Optimum FIR Linear Phase Digital Filters." *IEEE Trans. Audio Electroacoustics*, 21(6), 506-526.
6. **SciPy Documentation** -- `scipy.signal` module: https://docs.scipy.org/doc/scipy/reference/signal.html

---

## 탐색

- 이전: [08. Z 변환과 전달 함수](08_Z_Transform_and_Transfer_Functions.md)
- 다음: [10. IIR 필터 설계](10_IIR_Filter_Design.md)
- [개요로 돌아가기](00_Overview.md)
