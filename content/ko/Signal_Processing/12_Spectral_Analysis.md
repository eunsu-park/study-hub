# 스펙트럼 분석(Spectral Analysis)

## 학습 목표

- 전력 스펙트럼 밀도(Power Spectral Density, PSD)의 정의와 성질 이해
- 자기상관과 PSD를 연결하는 위너-킨친 정리(Wiener-Khinchin theorem) 마스터
- 비모수적 스펙트럼 추정 방법(주기도, Bartlett, Welch, Blackman-Tukey) 학습
- 스펙트럼 추정에서의 해상도-분산 트레이드오프(resolution-variance tradeoff) 이해
- 모수적 스펙트럼 추정 방법(AR, MA, ARMA 모델) 학습
- 모델 차수 선택 기준(AIC, BIC) 적용
- 교차 스펙트럼 분석(Cross-spectral analysis) 및 코히런스(coherence) 계산
- 시변 스펙트럼 분석을 위한 스펙트로그램(spectrogram) 생성 및 해석
- Python의 `scipy.signal`과 `matplotlib`을 사용한 스펙트럼 분석 구현

---

## 목차

1. [스펙트럼 분석 소개](#1-스펙트럼-분석-소개)
2. [전력 스펙트럼 밀도](#2-전력-스펙트럼-밀도)
3. [위너-킨친 정리](#3-위너-킨친-정리)
4. [주기도](#4-주기도)
5. [Bartlett 방법](#5-bartlett-방법)
6. [Welch 방법](#6-welch-방법)
7. [Blackman-Tukey 방법](#7-blackman-tukey-방법)
8. [해상도-분산 트레이드오프](#8-해상도-분산-트레이드오프)
9. [모수적 방법: AR 모델](#9-모수적-방법-ar-모델)
10. [모수적 방법: MA 및 ARMA](#10-모수적-방법-ma-및-arma)
11. [모델 차수 선택](#11-모델-차수-선택)
12. [교차 스펙트럼 분석과 코히런스](#12-교차-스펙트럼-분석과-코히런스)
13. [스펙트로그램](#13-스펙트로그램)
14. [Python 구현](#14-python-구현)
15. [연습 문제](#15-연습-문제)

---

## 1. 스펙트럼 분석 소개

### 1.1 문제 정의

유한한 관측 구간의 랜덤 또는 결정론적 신호가 주어졌을 때, 신호의 전력(또는 에너지)이 주파수에 따라 어떻게 분포하는지 추정하고자 한다. 이것이 **스펙트럼 추정(spectral estimation)** 문제이다.

유한한 결정론적 신호의 정확한 스펙트럼 표현을 계산하는 DFT와 달리, 스펙트럼 분석은 다음을 다룬다:
- **랜덤 과정(random processes)**: 정확한 값이 아닌 통계적 성질을 갖는 신호
- **유한 데이터**: 무한 지속 시간 과정의 $N$개 샘플만 관측
- **추정 불확실성**: 모든 추정치는 편향(bias)과 분산(variance)을 가짐

### 1.2 응용 분야

```
┌──────────────────────────────────────────────────────────────────┐
│                Spectral Analysis Applications                    │
├────────────────────┬─────────────────────────────────────────────┤
│ Audio/Speech       │ Pitch detection, formant analysis,          │
│                    │ noise characterization                      │
├────────────────────┼─────────────────────────────────────────────┤
│ Communications     │ Channel characterization, interference      │
│                    │ detection, modulation classification        │
├────────────────────┼─────────────────────────────────────────────┤
│ Biomedical         │ EEG/ECG analysis, sleep staging,            │
│                    │ heart rate variability                      │
├────────────────────┼─────────────────────────────────────────────┤
│ Radar/Sonar        │ Target detection, Doppler estimation        │
├────────────────────┼─────────────────────────────────────────────┤
│ Vibration Analysis │ Machine health monitoring, modal analysis   │
├────────────────────┼─────────────────────────────────────────────┤
│ Geophysics         │ Seismic analysis, ocean wave spectra        │
├────────────────────┼─────────────────────────────────────────────┤
│ Astrophysics       │ Pulsar timing, gravitational wave detection │
└────────────────────┴─────────────────────────────────────────────┘
```

### 1.3 두 가지 접근법

**비모수적 방법(Non-parametric methods)**: 신호에 대한 가정을 최소화한다. 데이터로부터 PSD를 직접 추정한다(예: 주기도, Welch 방법).

**모수적 방법(Parametric methods)**: 신호가 특정 모델(예: AR, MA, ARMA)에 의해 생성된다고 가정한다. 모델 파라미터를 추정한 후, 모델로부터 PSD를 계산한다.

---

## 2. 전력 스펙트럼 밀도

### 2.1 랜덤 과정에 대한 정의

광의 정상(Wide-Sense Stationary, WSS) 랜덤 과정 $x[n]$에 대해, **전력 스펙트럼 밀도(Power Spectral Density, PSD)**는 다음과 같이 정의된다:

$$S_x(e^{j\omega}) = \sum_{k=-\infty}^{\infty} r_{xx}[k] e^{-j\omega k}$$

여기서 $r_{xx}[k] = E\{x[n] x^*[n-k]\}$는 **자기상관 수열(autocorrelation sequence)**이다.

### 2.2 PSD의 성질

1. **실수값**: $S_x(e^{j\omega}) \in \mathbb{R}$, 모든 $\omega$에서
2. **비음수**: $S_x(e^{j\omega}) \geq 0$, 모든 $\omega$에서
3. **주기성**: $S_x(e^{j\omega}) = S_x(e^{j(\omega + 2\pi)})$
4. **우함수 대칭** (실수 $x[n]$인 경우): $S_x(e^{j\omega}) = S_x(e^{-j\omega})$
5. **총 전력**: $r_{xx}[0] = E\{|x[n]|^2\} = \frac{1}{2\pi} \int_{-\pi}^{\pi} S_x(e^{j\omega}) d\omega$

### 2.3 필터링된 과정의 PSD

$y[n] = h[n] * x[n]$ (LTI 필터링)이면:

$$S_y(e^{j\omega}) = |H(e^{j\omega})|^2 S_x(e^{j\omega})$$

이는 입력과 출력 전력 스펙트럼 간의 근본적인 관계이다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

def psd_of_filtered_noise():
    """Demonstrate PSD of filtered white noise."""
    np.random.seed(42)
    N = 10000
    fs = 1000

    # White noise: flat PSD
    x = np.random.randn(N)

    # Design bandpass filter
    sos = signal.butter(4, [100, 200], btype='bandpass', fs=fs, output='sos')
    y = signal.sosfilt(sos, x)

    # Estimate PSDs using Welch's method
    f_x, Pxx = signal.welch(x, fs=fs, nperseg=512)
    f_y, Pyy = signal.welch(y, fs=fs, nperseg=512)

    # Theoretical: |H(f)|^2 * Pxx
    w, H = signal.sosfreqz(sos, worN=len(f_x), fs=fs)
    Pyy_theory = np.abs(H)**2 * np.mean(Pxx)  # Pxx ≈ constant for white noise

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Input PSD
    axes[0, 0].semilogy(f_x, Pxx, 'b-', linewidth=1.5)
    axes[0, 0].set_title('Input PSD (White Noise)')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('PSD (V²/Hz)')
    axes[0, 0].grid(True, alpha=0.3)

    # Filter response
    axes[0, 1].plot(w, 20*np.log10(np.abs(H) + 1e-15), 'g-', linewidth=1.5)
    axes[0, 1].set_title('|H(f)|² (Bandpass Filter)')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude (dB)')
    axes[0, 1].grid(True, alpha=0.3)

    # Output PSD: measured vs theoretical
    axes[1, 0].semilogy(f_y, Pyy, 'r-', linewidth=1.5, label='Estimated')
    axes[1, 0].semilogy(w, Pyy_theory, 'k--', linewidth=1.5, label='Theoretical')
    axes[1, 0].set_title('Output PSD: Sy = |H|² · Sx')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('PSD (V²/Hz)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Time domain
    t = np.arange(N) / fs
    axes[1, 1].plot(t[:500], x[:500], 'b-', alpha=0.5, label='Input')
    axes[1, 1].plot(t[:500], y[:500], 'r-', linewidth=1.5, label='Output')
    axes[1, 1].set_title('Time Domain')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('PSD of Filtered Process: Sy(f) = |H(f)|² Sx(f)', fontsize=14)
    plt.tight_layout()
    plt.savefig('psd_filtered_noise.png', dpi=150)
    plt.close()

psd_of_filtered_noise()
```

---

## 3. 위너-킨친 정리

### 3.1 정리 내용

**위너-킨친 정리(Wiener-Khinchin theorem)**는 광의 정상 과정에 대해, 전력 스펙트럼 밀도와 자기상관 함수가 푸리에 변환 쌍을 이룬다고 말한다:

$$S_x(e^{j\omega}) = \sum_{k=-\infty}^{\infty} r_{xx}[k] e^{-j\omega k} \quad \text{(DTFT)}$$

$$r_{xx}[k] = \frac{1}{2\pi} \int_{-\pi}^{\pi} S_x(e^{j\omega}) e^{j\omega k} d\omega \quad \text{(역 DTFT)}$$

### 3.2 중요성

이 정리는 스펙트럼 추정에 두 가지 근본적인 접근법을 제공한다:

1. **직접 접근법**: 데이터로부터 PSD를 직접 추정 (주기도 방법)
2. **간접 접근법**: 먼저 자기상관 $\hat{r}_{xx}[k]$를 추정한 후 푸리에 변환 (Blackman-Tukey 방법)

### 3.3 수치 검증

```python
def verify_wiener_khinchin():
    """Verify the Wiener-Khinchin theorem numerically."""
    np.random.seed(42)
    N = 50000
    fs = 1000

    # Generate colored noise (AR process)
    # x[n] = 0.9*x[n-1] + w[n]
    a = 0.9
    w = np.random.randn(N)
    x = np.zeros(N)
    x[0] = w[0]
    for n in range(1, N):
        x[n] = a * x[n-1] + w[n]

    # Method 1: Direct PSD estimation (periodogram)
    f_direct, Pxx_direct = signal.periodogram(x, fs=fs)

    # Method 2: Via autocorrelation (Wiener-Khinchin)
    max_lag = 500
    r_xx = np.correlate(x, x, mode='full')[N-1-max_lag:N+max_lag] / N
    lags = np.arange(-max_lag, max_lag + 1)

    # DTFT of autocorrelation
    omega = np.linspace(0, np.pi, len(f_direct))
    Pxx_wk = np.zeros(len(omega))
    for i, w_val in enumerate(omega):
        Pxx_wk[i] = np.real(np.sum(r_xx * np.exp(-1j * w_val * lags))) / fs

    # Theoretical PSD for AR(1): sigma_w^2 / |1 - a*e^(-jw)|^2
    sigma_w = 1.0
    Pxx_theory = sigma_w**2 / (np.abs(1 - a * np.exp(-1j * omega))**2) / fs

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Autocorrelation
    axes[0, 0].plot(lags, r_xx, 'b-', linewidth=1.5)
    axes[0, 0].set_title('Estimated Autocorrelation r_xx[k]')
    axes[0, 0].set_xlabel('Lag k')
    axes[0, 0].set_ylabel('r_xx[k]')
    axes[0, 0].grid(True, alpha=0.3)

    # Theoretical autocorrelation: r_xx[k] = sigma_w^2 * a^|k| / (1-a^2)
    r_theory = sigma_w**2 * a**np.abs(lags) / (1 - a**2)
    axes[0, 1].plot(lags, r_xx, 'b-', alpha=0.5, label='Estimated')
    axes[0, 1].plot(lags, r_theory, 'r--', linewidth=1.5, label='Theoretical')
    axes[0, 1].set_title('Autocorrelation: Estimated vs Theoretical')
    axes[0, 1].set_xlabel('Lag k')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # PSD comparison
    f_hz = omega / (2 * np.pi) * fs
    axes[1, 0].semilogy(f_direct, Pxx_direct, 'b-', alpha=0.3, label='Periodogram')
    axes[1, 0].semilogy(f_hz, Pxx_wk, 'g-', linewidth=2, label='Wiener-Khinchin')
    axes[1, 0].semilogy(f_hz, Pxx_theory, 'r--', linewidth=2, label='Theoretical')
    axes[1, 0].set_title('PSD Comparison')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('PSD (V²/Hz)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, which='both', alpha=0.3)

    # PSD in dB
    axes[1, 1].plot(f_hz, 10*np.log10(Pxx_wk + 1e-15), 'g-', linewidth=2,
                     label='Wiener-Khinchin')
    axes[1, 1].plot(f_hz, 10*np.log10(Pxx_theory + 1e-15), 'r--', linewidth=2,
                     label='Theoretical')
    axes[1, 1].set_title('PSD (dB Scale)')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('PSD (dB/Hz)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Wiener-Khinchin Theorem Verification', fontsize=14)
    plt.tight_layout()
    plt.savefig('wiener_khinchin.png', dpi=150)
    plt.close()

verify_wiener_khinchin()
```

---

## 4. 주기도

### 4.1 정의

**주기도(periodogram)**는 DFT 크기의 제곱을 기반으로 하는 가장 단순한 PSD 추정량이다:

$$\hat{S}_x^{(\text{per})}(f_k) = \frac{1}{N f_s} |X[k]|^2 = \frac{1}{N f_s} \left| \sum_{n=0}^{N-1} x[n] e^{-j 2\pi k n / N} \right|^2$$

여기서 $f_k = k f_s / N$이고 $k = 0, 1, \ldots, N-1$이다.

### 4.2 통계적 성질

**편향(Bias)**: 주기도의 기댓값은 Fejer 커널과 합성곱된 참 PSD이다:

$$E\left\{\hat{S}_x^{(\text{per})}(e^{j\omega})\right\} = \frac{1}{2\pi} S_x(e^{j\omega}) * W_N(e^{j\omega})$$

여기서 $W_N(e^{j\omega}) = \frac{1}{N}\left|\frac{\sin(N\omega/2)}{\sin(\omega/2)}\right|^2$는 Fejer 커널(Dirichlet 커널의 제곱)이다.

- $N \to \infty$일수록 편향이 감소한다 (점근적으로 비편향)
- 유한 $N$에서는 유한 관측 윈도우로 인한 스펙트럼 누설(spectral leakage)이 존재한다

**분산(Variance)**: 주기도는 **일치 추정량이 아니다** -- 분산은 $N$이 증가해도 감소하지 않는다:

$$\text{Var}\left\{\hat{S}_x^{(\text{per})}(e^{j\omega})\right\} \approx S_x^2(e^{j\omega})$$

이는 데이터 길이에 상관없이 주기도가 크게 변동함을 의미하며, 매끄러운 PSD 추정에는 신뢰할 수 없다.

### 4.3 구현 및 시연

```python
def periodogram_analysis():
    """Analyze the properties of the periodogram estimator."""
    np.random.seed(42)
    fs = 1000

    # True signal: two sinusoids in noise
    f1, f2 = 100, 120  # Hz (close frequencies)
    A1, A2 = 1.0, 0.8

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Different data lengths
    for idx, N in enumerate([64, 256, 1024, 4096]):
        ax = axes[idx // 2, idx % 2]
        t = np.arange(N) / fs
        x = (A1 * np.sin(2*np.pi*f1*t) +
             A2 * np.sin(2*np.pi*f2*t) +
             np.random.randn(N) * 0.5)

        # Periodogram
        f, Pxx = signal.periodogram(x, fs=fs)
        ax.semilogy(f, Pxx, 'b-', linewidth=0.8, alpha=0.8)

        # Mark true frequencies
        ax.axvline(f1, color='r', linestyle='--', alpha=0.5, label=f'{f1} Hz')
        ax.axvline(f2, color='g', linestyle='--', alpha=0.5, label=f'{f2} Hz')

        # Frequency resolution
        delta_f = fs / N
        ax.set_title(f'N={N}, Δf = {delta_f:.1f} Hz '
                     f'({"resolved" if delta_f < abs(f2-f1) else "NOT resolved"})')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (V²/Hz)')
        ax.set_xlim(0, 300)
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.3)

    plt.suptitle('Periodogram: Effect of Data Length on Resolution', fontsize=14)
    plt.tight_layout()
    plt.savefig('periodogram_analysis.png', dpi=150)
    plt.close()

periodogram_analysis()
```

### 4.4 주기도 분산 시연

```python
def periodogram_variance():
    """Show that periodogram variance doesn't decrease with N."""
    np.random.seed(42)
    fs = 1000

    # AR(1) process (known PSD)
    a = 0.9
    sigma_w = 1.0
    num_realizations = 50

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, N in zip(axes, [256, 1024, 4096]):
        for trial in range(num_realizations):
            w = np.random.randn(N) * sigma_w
            x = np.zeros(N)
            x[0] = w[0]
            for n in range(1, N):
                x[n] = a * x[n-1] + w[n]

            f, Pxx = signal.periodogram(x, fs=fs)
            ax.semilogy(f, Pxx, 'b-', alpha=0.1, linewidth=0.5)

        # Theoretical PSD
        f_theory = np.linspace(0, fs/2, 500)
        omega = 2 * np.pi * f_theory / fs
        Pxx_theory = sigma_w**2 / (np.abs(1 - a*np.exp(-1j*omega))**2) / fs
        ax.semilogy(f_theory, Pxx_theory, 'r-', linewidth=2, label='True PSD')

        ax.set_title(f'N = {N} ({num_realizations} realizations)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (V²/Hz)')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)

    plt.suptitle('Periodogram Variance Does NOT Decrease with N', fontsize=14)
    plt.tight_layout()
    plt.savefig('periodogram_variance.png', dpi=150)
    plt.close()

periodogram_variance()
```

---

## 5. Bartlett 방법

### 5.1 원리

**Bartlett 방법(Bartlett's method)** (1948)은 여러 주기도를 평균하여 분산을 줄인다:

1. 데이터를 길이 $L$의 $K$개 비중첩 구간으로 분할한다 ($N = KL$)
2. 각 구간의 주기도를 계산한다
3. $K$개 주기도의 평균을 구한다

$$\hat{S}_x^{(\text{Bartlett})}(f) = \frac{1}{K} \sum_{i=0}^{K-1} \hat{S}_{x_i}^{(\text{per})}(f)$$

### 5.2 성질

**분산 감소**: $K$개의 독립적인 주기도를 평균하면:

$$\text{Var}\left\{\hat{S}_x^{(\text{Bartlett})}\right\} \approx \frac{1}{K} S_x^2(e^{j\omega})$$

**해상도 손실**: 각 구간이 더 짧으므로 ($L = N/K$), 주파수 해상도는:

$$\Delta f = \frac{f_s}{L} = \frac{K f_s}{N}$$

이것이 **해상도-분산 트레이드오프**이다: 분산을 $K$배 줄이면 해상도가 $K$배 나빠진다.

### 5.3 구현

```python
def bartlett_method(x, fs, K):
    """
    Bartlett's method for PSD estimation.

    Parameters:
        x: input signal
        fs: sampling frequency
        K: number of non-overlapping segments

    Returns:
        f: frequency vector
        Pxx: PSD estimate
    """
    N = len(x)
    L = N // K  # Segment length

    # Compute periodogram of each segment
    f = np.fft.rfftfreq(L, 1/fs)
    Pxx_sum = np.zeros(len(f))

    for i in range(K):
        segment = x[i*L : (i+1)*L]
        X_seg = np.fft.rfft(segment)
        Pxx_seg = np.abs(X_seg)**2 / (L * fs)
        Pxx_sum += Pxx_seg

    Pxx = Pxx_sum / K
    return f, Pxx

# Compare periodogram vs Bartlett with different K
np.random.seed(42)
N = 4096
fs = 1000

# AR(1) process
a = 0.9
w = np.random.randn(N)
x = np.zeros(N)
x[0] = w[0]
for n in range(1, N):
    x[n] = a * x[n-1] + w[n]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

configs = [
    (1, 'Periodogram (K=1)'),
    (4, 'Bartlett (K=4)'),
    (16, 'Bartlett (K=16)'),
    (64, 'Bartlett (K=64)'),
]

# Theoretical PSD
f_theory = np.linspace(0, fs/2, 500)
omega = 2 * np.pi * f_theory / fs
Pxx_theory = 1.0 / (np.abs(1 - a*np.exp(-1j*omega))**2) / fs

for ax, (K, title) in zip(axes.flat, configs):
    f, Pxx = bartlett_method(x, fs, K)
    L = N // K

    ax.semilogy(f, Pxx, 'b-', linewidth=1, alpha=0.8, label='Estimate')
    ax.semilogy(f_theory, Pxx_theory, 'r-', linewidth=2, label='True PSD')
    ax.set_title(f'{title}, L={L}, Δf={fs/L:.1f} Hz')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (V²/Hz)')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

plt.suptitle("Bartlett's Method: Resolution-Variance Tradeoff", fontsize=14)
plt.tight_layout()
plt.savefig('bartlett_method.png', dpi=150)
plt.close()
```

---

## 6. Welch 방법

### 6.1 Bartlett 방법과의 차이점

**Welch 방법(Welch's method)** (1967)은 두 가지 핵심 개선을 통해 Bartlett 방법을 확장한다:

1. **중첩 구간(Overlapping segments)**: 구간이 일정 비율(보통 50%)만큼 중첩되어, 동일한 데이터에서 더 많은 구간을 얻는다
2. **윈도잉(Windowing)**: 각 구간에 윈도우 함수(예: Hanning)를 곱하여 스펙트럼 누설을 줄인다

$$\hat{S}_x^{(\text{Welch})}(f) = \frac{1}{K} \sum_{i=0}^{K-1} \frac{1}{L U} \left| \sum_{n=0}^{L-1} w[n] x_i[n] e^{-j2\pi f n / f_s} \right|^2$$

여기서 $U = \frac{1}{L}\sum_{n=0}^{L-1} w^2[n]$은 윈도우 전력 정규화이다.

### 6.2 성질

- **더 많은 구간**: 50% 중첩과 $N$ 전체 샘플로 $K \approx 2N/L - 1$개의 구간 획득 (Bartlett의 약 두 배)
- **낮은 분산**: 더 많은 평균화와 윈도잉된 누설 감소
- **동일한 해상도**: 구간 길이 $L$에 의해 결정

### 6.3 구현 및 분석

```python
def welch_analysis():
    """Comprehensive Welch's method analysis."""
    np.random.seed(42)
    N = 8192
    fs = 1000

    # Signal: two closely spaced tones + broadband noise
    t = np.arange(N) / fs
    f1, f2 = 100, 108  # 8 Hz apart
    x = (np.sin(2*np.pi*f1*t) + 0.7*np.sin(2*np.pi*f2*t) +
         0.5*np.random.randn(N))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Effect of segment length (nperseg)
    for i, nperseg in enumerate([128, 512, 2048]):
        f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        axes[0, i].semilogy(f, Pxx, 'b-', linewidth=1.5)
        axes[0, i].axvline(f1, color='r', linestyle='--', alpha=0.5)
        axes[0, i].axvline(f2, color='g', linestyle='--', alpha=0.5)
        axes[0, i].set_title(f'nperseg = {nperseg}, Δf = {fs/nperseg:.1f} Hz')
        axes[0, i].set_xlabel('Frequency (Hz)')
        axes[0, i].set_ylabel('PSD')
        axes[0, i].set_xlim(50, 200)
        axes[0, i].grid(True, which='both', alpha=0.3)

    # 2. Effect of overlap
    nperseg = 512
    for i, overlap_frac in enumerate([0, 0.5, 0.75]):
        noverlap = int(nperseg * overlap_frac)
        f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
        n_segs = (N - noverlap) // (nperseg - noverlap) if noverlap < nperseg else 1
        axes[1, i].semilogy(f, Pxx, 'b-', linewidth=1.5)
        axes[1, i].axvline(f1, color='r', linestyle='--', alpha=0.5)
        axes[1, i].axvline(f2, color='g', linestyle='--', alpha=0.5)
        axes[1, i].set_title(f'Overlap = {overlap_frac*100:.0f}%, '
                              f'K ≈ {n_segs} segments')
        axes[1, i].set_xlabel('Frequency (Hz)')
        axes[1, i].set_ylabel('PSD')
        axes[1, i].set_xlim(50, 200)
        axes[1, i].grid(True, which='both', alpha=0.3)

    plt.suptitle("Welch's Method: Segment Length and Overlap Effects", fontsize=14)
    plt.tight_layout()
    plt.savefig('welch_analysis.png', dpi=150)
    plt.close()

welch_analysis()
```

### 6.4 윈도우 효과

```python
def welch_window_comparison():
    """Compare different windows in Welch's method."""
    np.random.seed(42)
    N = 4096
    fs = 1000

    # Signal with a weak tone near a strong one
    t = np.arange(N) / fs
    x = (np.sin(2*np.pi*100*t) +
         0.01*np.sin(2*np.pi*130*t) +  # 40 dB weaker
         0.1*np.random.randn(N))

    windows = ['boxcar', 'hann', 'hamming', 'blackman', 'blackmanharris']
    nperseg = 512

    fig, ax = plt.subplots(figsize=(14, 6))

    for window in windows:
        f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg,
                               noverlap=nperseg//2, window=window)
        ax.plot(f, 10*np.log10(Pxx + 1e-15), linewidth=1.5, label=window)

    ax.axvline(100, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(130, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (dB/Hz)')
    ax.set_title('Window Function Effect: Detecting a Weak Tone (-40 dB)')
    ax.set_xlim(50, 200)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('welch_windows.png', dpi=150)
    plt.close()

welch_window_comparison()
```

---

## 7. Blackman-Tukey 방법

### 7.1 원리

**Blackman-Tukey 방법(Blackman-Tukey method)** (1958)은 위너-킨친 정리에 기반한 간접적 접근법이다:

1. 자기상관 수열 $\hat{r}_{xx}[k]$를 추정한다
2. 자기상관에 윈도우 $w[k]$를 적용한다 (지연 범위를 제한하기 위해)
3. 윈도잉된 자기상관의 DTFT를 계산한다

$$\hat{S}_x^{(\text{BT})}(e^{j\omega}) = \sum_{k=-(M-1)}^{M-1} w[k] \hat{r}_{xx}[k] e^{-j\omega k}$$

여기서 $M$은 최대 지연(maximum lag)이다.

### 7.2 자기상관 추정

**편향(biased)** 자기상관 추정치:

$$\hat{r}_{xx}[k] = \frac{1}{N} \sum_{n=0}^{N-1-|k|} x[n+|k|] x^*[n], \quad |k| \leq N-1$$

**비편향(unbiased)** 추정치:

$$\hat{r}_{xx}^{(\text{unbiased})}[k] = \frac{1}{N - |k|} \sum_{n=0}^{N-1-|k|} x[n+|k|] x^*[n]$$

편향 추정치가 선호되는데, 이는 음이 아닌 PSD를 보장하는 반면 비편향 추정치는 음수 PSD 값을 생성할 수 있기 때문이다.

### 7.3 지연 윈도우

지연 윈도우(lag window) $w[k]$는 PSD 추정에 적용되는 평활화를 제어한다:
- **좁은 윈도우** ($M$ 작음): 매끄러운 PSD, 낮은 분산, 낮은 해상도
- **넓은 윈도우** ($M$ 큼): 거친 PSD, 높은 분산, 높은 해상도

일반적인 지연 윈도우: Bartlett (삼각형), Parzen, Tukey (코사인 테이퍼).

### 7.4 구현

```python
def blackman_tukey_method(x, fs, max_lag, window='bartlett'):
    """
    Blackman-Tukey PSD estimation.

    Parameters:
        x: input signal
        fs: sampling frequency
        max_lag: maximum lag for autocorrelation
        window: lag window type

    Returns:
        f: frequency vector
        Pxx: PSD estimate
    """
    N = len(x)

    # Biased autocorrelation estimate
    r_xx = np.correlate(x, x, mode='full') / N
    center = N - 1
    r_xx_lags = r_xx[center - max_lag:center + max_lag + 1]
    lags = np.arange(-max_lag, max_lag + 1)

    # Apply lag window
    if window == 'bartlett':
        w = 1 - np.abs(lags) / max_lag
    elif window == 'parzen':
        u = np.abs(lags) / max_lag
        w = np.where(u <= 0.5,
                     1 - 6*u**2 + 6*u**3,
                     2*(1 - u)**3)
    elif window == 'tukey':
        w = 0.5 * (1 + np.cos(np.pi * lags / max_lag))
    else:
        w = np.ones(len(lags))

    r_windowed = r_xx_lags * w

    # Compute PSD via DTFT
    nfft = 2048
    f = np.linspace(0, fs/2, nfft//2 + 1)
    omega = 2 * np.pi * f / fs

    Pxx = np.zeros(len(f))
    for i, w_val in enumerate(omega):
        Pxx[i] = np.real(np.sum(r_windowed * np.exp(-1j * w_val * lags))) / fs

    # Ensure non-negative
    Pxx = np.maximum(Pxx, 0)

    return f, Pxx

# Compare with different max_lag values
np.random.seed(42)
N = 4096
fs = 1000

a = 0.9
w = np.random.randn(N)
x = np.zeros(N)
x[0] = w[0]
for n in range(1, N):
    x[n] = a * x[n-1] + w[n]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Theoretical
f_theory = np.linspace(0, fs/2, 500)
omega_t = 2 * np.pi * f_theory / fs
Pxx_theory = 1.0 / (np.abs(1 - a*np.exp(-1j*omega_t))**2) / fs

for ax, max_lag in zip(axes.flat, [16, 64, 256, 1024]):
    f_bt, Pxx_bt = blackman_tukey_method(x, fs, max_lag, window='bartlett')

    ax.semilogy(f_bt, Pxx_bt, 'b-', linewidth=1.5, label='BT estimate')
    ax.semilogy(f_theory, Pxx_theory, 'r--', linewidth=2, label='True PSD')
    ax.set_title(f'Blackman-Tukey (max_lag = {max_lag})')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (V²/Hz)')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

plt.suptitle('Blackman-Tukey Method: Effect of Maximum Lag', fontsize=14)
plt.tight_layout()
plt.savefig('blackman_tukey.png', dpi=150)
plt.close()
```

---

## 8. 해상도-분산 트레이드오프

### 8.1 근본적 한계

모든 비모수적 스펙트럼 추정량은 근본적인 트레이드오프에 직면한다:

$$\text{해상도} \times \text{분산} \approx \text{상수}$$

- 더 나은 주파수 해상도를 위해서는 더 긴 관측 윈도우가 필요하고, 이는 평균 수가 줄어들어 분산이 높아짐을 의미한다
- 낮은 분산을 위해서는 더 많은 평균화가 필요하고, 이는 더 짧은 구간과 낮은 해상도를 의미한다

### 8.2 정량적 분석

길이 $L$의 $K$개 구간을 사용하는 Welch 방법에서:

- **해상도**: $\Delta f = f_s / L$
- **정규화 분산**: $\text{Var}\{\hat{S}\} / S^2 \approx 1/K$
- **곱**: $\Delta f \times \text{Var}\{\hat{S}\} / S^2 \approx f_s / (KL) = f_s / N$

**해상도 대역폭(Resolution Bandwidth, RBW)**과 **등가 자유도(Equivalent Degrees of Freedom, EDOF)**가 이 트레이드오프를 특성화한다:

$$\text{EDOF} \approx 2K \quad \text{(비중첩 구간의 경우)}$$

### 8.3 시각화

```python
def resolution_variance_tradeoff():
    """Visualize the resolution-variance tradeoff."""
    np.random.seed(42)
    N = 8192
    fs = 1000

    # Two tones: 100 Hz and 110 Hz (10 Hz apart)
    t = np.arange(N) / fs
    x = np.sin(2*np.pi*100*t) + np.sin(2*np.pi*110*t) + 0.5*np.random.randn(N)

    # Sweep segment length
    nperseg_values = [64, 128, 256, 512, 1024, 2048, 4096]
    resolutions = []
    variances = []

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for i, nperseg in enumerate(nperseg_values):
        # Multiple realizations to estimate variance
        psds = []
        for trial in range(50):
            x_trial = (np.sin(2*np.pi*100*t) + np.sin(2*np.pi*110*t) +
                       0.5*np.random.randn(N))
            f, Pxx = signal.welch(x_trial, fs=fs, nperseg=nperseg,
                                   noverlap=nperseg//2)
            psds.append(Pxx)

        psds = np.array(psds)
        mean_psd = np.mean(psds, axis=0)
        std_psd = np.std(psds, axis=0)

        resolutions.append(fs / nperseg)
        variances.append(np.mean(std_psd**2 / (mean_psd**2 + 1e-15)))

        if i < 7:
            ax = axes[i // 4, i % 4]
            ax.semilogy(f, mean_psd, 'b-', linewidth=1.5)
            ax.fill_between(f, mean_psd - std_psd, mean_psd + std_psd,
                           alpha=0.2, color='blue')
            ax.axvline(100, color='r', linestyle='--', alpha=0.5)
            ax.axvline(110, color='g', linestyle='--', alpha=0.5)
            ax.set_title(f'L={nperseg}, Δf={fs/nperseg:.1f} Hz')
            ax.set_xlim(50, 200)
            ax.set_xlabel('Freq (Hz)')
            ax.grid(True, which='both', alpha=0.3)

    # Summary plot
    ax_summary = axes[1, 3]
    ax_summary.loglog(resolutions, variances, 'ro-', markersize=8, linewidth=2)
    ax_summary.set_xlabel('Resolution Δf (Hz)')
    ax_summary.set_ylabel('Normalized Variance')
    ax_summary.set_title('Resolution-Variance Tradeoff')
    ax_summary.grid(True, which='both', alpha=0.3)

    plt.suptitle('Resolution vs Variance Tradeoff in Welch\'s Method', fontsize=14)
    plt.tight_layout()
    plt.savefig('resolution_variance.png', dpi=150)
    plt.close()

resolution_variance_tradeoff()
```

---

## 9. 모수적 방법: AR 모델

### 9.1 자기회귀(AR) 모델

차수 $p$의 AR 과정은 다음을 만족한다:

$$x[n] = -\sum_{k=1}^{p} a_k x[n-k] + w[n]$$

여기서 $w[n]$은 분산 $\sigma_w^2$인 백색 잡음이다.

AR($p$) 과정의 PSD:

$$S_x(e^{j\omega}) = \frac{\sigma_w^2}{|A(e^{j\omega})|^2} = \frac{\sigma_w^2}{\left|1 + \sum_{k=1}^{p} a_k e^{-j\omega k}\right|^2}$$

### 9.2 AR 모델을 사용하는 이유

AR 모델이 인기 있는 이유:
- PSD가 스펙트럼 피크 모델링에 적합한 매끄럽고 뾰족한 형태를 가짐
- 파라미터 추정이 선형 방정식 풀기(Yule-Walker)로 귀결됨
- 짧은 데이터에서도 탁월한 주파수 해상도를 달성할 수 있음
- 많은 자연 신호(음성, 진동)가 AR 과정으로 잘 모델링됨

### 9.3 Yule-Walker 방정식

AR 파라미터는 **Yule-Walker 방정식**으로 결정된다:

$$\begin{bmatrix} r_{xx}[0] & r_{xx}[1] & \cdots & r_{xx}[p-1] \\ r_{xx}[1] & r_{xx}[0] & \cdots & r_{xx}[p-2] \\ \vdots & \vdots & \ddots & \vdots \\ r_{xx}[p-1] & r_{xx}[p-2] & \cdots & r_{xx}[0] \end{bmatrix} \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_p \end{bmatrix} = -\begin{bmatrix} r_{xx}[1] \\ r_{xx}[2] \\ \vdots \\ r_{xx}[p] \end{bmatrix}$$

행렬 형태로: $\mathbf{R} \mathbf{a} = -\mathbf{r}$

### 9.4 Levinson-Durbin 알고리즘

Levinson-Durbin 알고리즘은 Yule-Walker 방정식을 효율적으로 $O(p^2)$ 연산으로 풀 수 있다 (직접 행렬 역산의 $O(p^3)$ 대비):

```python
def levinson_durbin(r, order):
    """
    Levinson-Durbin recursion for AR parameter estimation.

    Parameters:
        r: autocorrelation sequence r[0], r[1], ..., r[order]
        order: AR model order p

    Returns:
        a: AR coefficients [a1, a2, ..., ap]
        sigma2: driving noise variance
        reflection_coeffs: reflection coefficients (PARCOR)
    """
    # Initialize
    a = np.zeros(order)
    reflection_coeffs = np.zeros(order)

    # Order 1
    a[0] = -r[1] / r[0]
    reflection_coeffs[0] = a[0]
    sigma2 = r[0] * (1 - a[0]**2)

    # Recursion
    for m in range(1, order):
        # Compute reflection coefficient
        numerator = r[m + 1] + np.sum(a[:m] * r[m:0:-1])
        k_m = -numerator / sigma2
        reflection_coeffs[m] = k_m

        # Update AR coefficients
        a_new = np.zeros(order)
        a_new[m] = k_m
        for j in range(m):
            a_new[j] = a[j] + k_m * a[m - 1 - j]
        a = a_new

        # Update variance
        sigma2 = sigma2 * (1 - k_m**2)

    return a, sigma2, reflection_coeffs

def ar_psd(a, sigma2, fs, nfft=1024):
    """Compute PSD from AR parameters."""
    f = np.linspace(0, fs/2, nfft//2 + 1)
    omega = 2 * np.pi * f / fs

    # A(e^jw) = 1 + a1*e^(-jw) + a2*e^(-j2w) + ...
    A_freq = np.ones(len(f), dtype=complex)
    for k in range(len(a)):
        A_freq += a[k] * np.exp(-1j * (k+1) * omega)

    Pxx = sigma2 / (np.abs(A_freq)**2 * fs)
    return f, Pxx
```

### 9.5 AR 스펙트럼 추정

```python
def ar_spectral_estimation():
    """Demonstrate AR spectral estimation."""
    np.random.seed(42)
    N = 256  # Short data record
    fs = 1000

    # True signal: sum of two narrow peaks
    t = np.arange(N) / fs
    x = (np.sin(2*np.pi*100*t) + 0.7*np.sin(2*np.pi*120*t) +
         0.3*np.random.randn(N))

    # Autocorrelation estimate (biased)
    r_full = np.correlate(x, x, mode='full') / N
    center = N - 1
    max_order = 50

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Welch's method (for reference)
    f_welch, Pxx_welch = signal.welch(x, fs=fs, nperseg=min(N, 128))

    # Different AR orders
    orders = [2, 4, 10, 20, 30, 50]

    for ax, p in zip(axes.flat, orders):
        r = r_full[center:center + p + 1]

        # Levinson-Durbin
        a, sigma2, refl = levinson_durbin(r, p)

        # AR PSD
        f_ar, Pxx_ar = ar_psd(a, sigma2, fs)

        ax.semilogy(f_welch, Pxx_welch, 'b-', alpha=0.5, label='Welch')
        ax.semilogy(f_ar, Pxx_ar, 'r-', linewidth=2, label=f'AR({p})')
        ax.axvline(100, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(120, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'AR Order p = {p}')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD')
        ax.set_xlim(0, 300)
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.3)

    plt.suptitle(f'AR Spectral Estimation (N={N} samples)', fontsize=14)
    plt.tight_layout()
    plt.savefig('ar_spectral.png', dpi=150)
    plt.close()

ar_spectral_estimation()
```

### 9.6 Burg 방법

Burg 방법은 먼저 자기상관을 계산하지 않고 데이터에서 직접 AR 파라미터를 추정하여 더 나은 해상도를 얻는다:

```python
def burg_method(x, order):
    """
    Burg's method for AR parameter estimation.

    Parameters:
        x: input signal
        order: AR model order

    Returns:
        a: AR coefficients
        sigma2: prediction error power
    """
    N = len(x)

    # Initialize forward and backward prediction errors
    ef = x.copy()
    eb = x.copy()

    a = np.zeros(order)
    sigma2 = np.mean(np.abs(x)**2)

    for m in range(order):
        # Compute reflection coefficient
        ef_m = ef[m+1:]
        eb_m = eb[m:-1]

        num = -2 * np.sum(ef_m * eb_m)
        den = np.sum(ef_m**2) + np.sum(eb_m**2)
        k = num / den if den > 0 else 0

        # Update AR coefficients
        a_new = np.zeros(order)
        a_new[m] = k
        for j in range(m):
            a_new[j] = a[j] + k * a[m - 1 - j]
        a = a_new

        # Update prediction errors
        ef_new = ef[m+1:] + k * eb[m:-1]
        eb_new = eb[m:-1] + k * ef[m+1:]
        ef[m+1:len(ef_new)+m+1] = ef_new
        eb[m:len(eb_new)+m] = eb_new

        # Update variance
        sigma2 = sigma2 * (1 - k**2)

    return a, sigma2

# Compare Burg vs Yule-Walker for a short data record
np.random.seed(42)
N = 64  # Very short!
fs = 1000
t = np.arange(N) / fs

# Two close sinusoids
x = np.sin(2*np.pi*100*t) + 0.8*np.sin(2*np.pi*115*t) + 0.3*np.random.randn(N)

p = 20  # AR order

# Yule-Walker
r_full = np.correlate(x, x, mode='full') / N
center = N - 1
r = r_full[center:center + p + 1]
a_yw, sigma2_yw, _ = levinson_durbin(r, p)
f_yw, Pxx_yw = ar_psd(a_yw, sigma2_yw, fs)

# Burg
a_burg, sigma2_burg = burg_method(x, p)
f_burg, Pxx_burg = ar_psd(a_burg, sigma2_burg, fs)

# Periodogram
f_per, Pxx_per = signal.periodogram(x, fs=fs)

fig, ax = plt.subplots(figsize=(12, 6))
ax.semilogy(f_per, Pxx_per, 'b-', alpha=0.5, label='Periodogram')
ax.semilogy(f_yw, Pxx_yw, 'g-', linewidth=2, label=f'Yule-Walker AR({p})')
ax.semilogy(f_burg, Pxx_burg, 'r-', linewidth=2, label=f'Burg AR({p})')
ax.axvline(100, color='gray', linestyle='--', alpha=0.5)
ax.axvline(115, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD')
ax.set_title(f'Burg vs Yule-Walker (N={N}, p={p})')
ax.set_xlim(0, 250)
ax.legend()
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('burg_method.png', dpi=150)
plt.close()
```

---

## 10. 모수적 방법: MA 및 ARMA

### 10.1 이동 평균(MA) 모델

차수 $q$의 MA 과정:

$$x[n] = w[n] + \sum_{k=1}^{q} b_k w[n-k]$$

PSD:

$$S_x(e^{j\omega}) = \sigma_w^2 \left|1 + \sum_{k=1}^{q} b_k e^{-j\omega k}\right|^2$$

MA 모델은 스펙트럼 **영점(nulls)** (스펙트럼의 영)이 있는 과정에 유용하다.

### 10.2 ARMA 모델

ARMA($p, q$) 과정은 AR과 MA를 결합한다:

$$x[n] = -\sum_{k=1}^{p} a_k x[n-k] + w[n] + \sum_{k=1}^{q} b_k w[n-k]$$

PSD:

$$S_x(e^{j\omega}) = \sigma_w^2 \frac{|B(e^{j\omega})|^2}{|A(e^{j\omega})|^2}$$

여기서 $B(z) = 1 + b_1 z^{-1} + \cdots + b_q z^{-q}$이고 $A(z) = 1 + a_1 z^{-1} + \cdots + a_p z^{-p}$이다.

ARMA 모델은 순수 AR 또는 MA 모델보다 적은 파라미터로 스펙트럼 피크(AR 부분)와 스펙트럼 영점(MA 부분)을 모두 표현할 수 있다.

### 10.3 ARMA 파라미터 추정

```python
from scipy.optimize import minimize

def arma_psd_estimate(x, p, q, fs, nfft=1024):
    """
    Estimate ARMA(p,q) PSD using a two-stage method.

    Stage 1: Estimate AR model of high order to approximate the signal
    Stage 2: Fit ARMA(p,q) model to the AR-estimated autocorrelation

    Parameters:
        x: input signal
        p: AR order
        q: MA order
        fs: sampling frequency
        nfft: FFT size for PSD computation

    Returns:
        f: frequency vector
        Pxx: PSD estimate
    """
    N = len(x)

    # Stage 1: High-order AR estimate
    high_order = max(p + q, 30)
    r_full = np.correlate(x, x, mode='full') / N
    center = N - 1
    r = r_full[center:center + high_order + 1]
    a_high, sigma2_high, _ = levinson_durbin(r, high_order)

    # Stage 2: Fit ARMA(p,q) - simplified approach using scipy
    # We minimize the prediction error
    def arma_cost(params):
        a_params = params[:p]
        b_params = params[p:p+q]
        sigma = params[-1]

        # Compute theoretical autocorrelation from ARMA params
        # and compare with estimated autocorrelation
        try:
            # Simple cost: prediction error on the data
            # Forward filter: A(z) * x[n] should be close to MA filtered noise
            y = x.copy()
            for k in range(p):
                y[k+1:] += a_params[k] * x[:N-k-1]
            return np.sum(y**2) / N
        except Exception:
            return 1e10

    # Initial guess from AR model
    x0 = np.zeros(p + q + 1)
    if p > 0:
        r_init = r_full[center:center + p + 1]
        a_init, sigma_init, _ = levinson_durbin(r_init, p)
        x0[:p] = a_init
    x0[-1] = 1.0

    # For simplicity, fall back to AR PSD as approximation
    r_ar = r_full[center:center + p + 1]
    a_ar, sigma2_ar, _ = levinson_durbin(r_ar, max(p, 1))

    f = np.linspace(0, fs/2, nfft//2 + 1)
    omega = 2 * np.pi * f / fs

    A_freq = np.ones(len(f), dtype=complex)
    for k in range(len(a_ar)):
        A_freq += a_ar[k] * np.exp(-1j * (k+1) * omega)

    Pxx = sigma2_ar / (np.abs(A_freq)**2 * fs)
    return f, Pxx

# Demonstrate: signal that needs ARMA model
np.random.seed(42)
N = 1024
fs = 1000

# True ARMA(2,2) process
b_true = np.array([1.0, 0.5, 0.3])  # MA coefficients
a_true = np.array([1.0, -1.5, 0.85])  # AR coefficients

w = np.random.randn(N)
x_arma = signal.lfilter(b_true, a_true, w)

# Compare AR and ARMA estimates
fig, ax = plt.subplots(figsize=(12, 6))

# Welch (reference)
f_welch, Pxx_welch = signal.welch(x_arma, fs=fs, nperseg=256)
ax.semilogy(f_welch, Pxx_welch, 'b-', alpha=0.5, linewidth=1, label='Welch')

# AR estimates of different orders
for p in [2, 5, 10, 20]:
    r = np.correlate(x_arma, x_arma, mode='full')[N-1:N-1+p+1] / N
    a_est, sig2, _ = levinson_durbin(r, p)
    f_ar, Pxx_ar = ar_psd(a_est, sig2, fs)
    ax.semilogy(f_ar, Pxx_ar, linewidth=1.5, label=f'AR({p})')

# True ARMA PSD
f_true = np.linspace(0, fs/2, 512)
omega_t = 2 * np.pi * f_true / fs
B_freq = np.polyval(b_true[::-1], np.exp(-1j * omega_t))
A_freq = np.polyval(a_true[::-1], np.exp(-1j * omega_t))
Pxx_true = np.abs(B_freq)**2 / (np.abs(A_freq)**2 * fs)
ax.semilogy(f_true, Pxx_true, 'k--', linewidth=2, label='True ARMA(2,2)')

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD')
ax.set_title('AR Approximation of ARMA Process')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('arma_comparison.png', dpi=150)
plt.close()
```

---

## 11. 모델 차수 선택

### 11.1 문제

올바른 모델 차수 $p$를 선택하는 것이 매우 중요하다:
- **너무 낮음**: 과소적합(underfitting), 스펙트럼 피크 누락, 편향된 추정
- **너무 높음**: 과대적합(overfitting), 가짜 피크, 높은 분산

### 11.2 정보 기준

**아카이케 정보 기준(Akaike Information Criterion, AIC)**:

$$\text{AIC}(p) = N \ln(\hat{\sigma}_p^2) + 2p$$

**베이지안 정보 기준(Bayesian Information Criterion, BIC)** (Schwarz 기준이라고도 함):

$$\text{BIC}(p) = N \ln(\hat{\sigma}_p^2) + p \ln(N)$$

여기서 $\hat{\sigma}_p^2$는 차수 $p$에서의 추정 예측 오차 분산이다.

- AIC는 차수를 과대추정하는 경향이 있다
- BIC는 올바른 차수를 선택하는 경향이 있다 (일치 추정량)
- 둘 다 복잡도에 페널티를 주지만 BIC는 큰 $N$에서 더 무겁게 페널티를 준다

### 11.3 최종 예측 오차(FPE)

$$\text{FPE}(p) = \hat{\sigma}_p^2 \cdot \frac{N + p + 1}{N - p - 1}$$

### 11.4 구현

```python
def model_order_selection(x, max_order=50):
    """
    Select AR model order using AIC, BIC, and FPE.

    Parameters:
        x: input signal
        max_order: maximum order to test

    Returns:
        best_orders: dictionary of best orders by each criterion
        criteria: dictionary of criterion values
    """
    N = len(x)

    # Autocorrelation
    r_full = np.correlate(x, x, mode='full') / N
    center = N - 1

    orders = np.arange(1, max_order + 1)
    aic_values = np.zeros(max_order)
    bic_values = np.zeros(max_order)
    fpe_values = np.zeros(max_order)

    for i, p in enumerate(orders):
        r = r_full[center:center + p + 1]
        try:
            a, sigma2, _ = levinson_durbin(r, p)
            sigma2 = max(sigma2, 1e-15)  # Prevent log of zero
        except Exception:
            sigma2 = 1e-15

        aic_values[i] = N * np.log(sigma2) + 2 * p
        bic_values[i] = N * np.log(sigma2) + p * np.log(N)
        fpe_values[i] = sigma2 * (N + p + 1) / max(N - p - 1, 1)

    best_aic = orders[np.argmin(aic_values)]
    best_bic = orders[np.argmin(bic_values)]
    best_fpe = orders[np.argmin(fpe_values)]

    return {
        'AIC': best_aic,
        'BIC': best_bic,
        'FPE': best_fpe,
    }, {
        'AIC': aic_values,
        'BIC': bic_values,
        'FPE': fpe_values,
        'orders': orders,
    }

# Example: known AR(4) process
np.random.seed(42)
N = 1024
fs = 1000

# True AR(4)
a_true = [1.0, -2.7607, 3.8106, -2.6535, 0.9238]  # 4 poles
w = np.random.randn(N)
x_ar4 = signal.lfilter([1], a_true, w)

best_orders, criteria = model_order_selection(x_ar4, max_order=30)

print(f"True order: 4")
print(f"AIC selects: p = {best_orders['AIC']}")
print(f"BIC selects: p = {best_orders['BIC']}")
print(f"FPE selects: p = {best_orders['FPE']}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, values) in zip(axes, [('AIC', criteria['AIC']),
                                       ('BIC', criteria['BIC']),
                                       ('FPE', criteria['FPE'])]):
    ax.plot(criteria['orders'], values, 'b-o', markersize=3, linewidth=1.5)
    best_p = best_orders[name]
    ax.axvline(best_p, color='r', linestyle='--', label=f'Best: p={best_p}')
    ax.axvline(4, color='g', linestyle=':', label='True: p=4')
    ax.set_xlabel('Model Order p')
    ax.set_ylabel(name)
    ax.set_title(f'{name} (Best: p={best_p})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Model Order Selection for AR(4) Process', fontsize=14)
plt.tight_layout()
plt.savefig('model_order_selection.png', dpi=150)
plt.close()
```

---

## 12. 교차 스펙트럼 분석과 코히런스

### 12.1 교차 전력 스펙트럼 밀도

두 신호 $x[n]$과 $y[n]$ 사이의 **교차 PSD(cross-PSD)**:

$$S_{xy}(e^{j\omega}) = \sum_{k=-\infty}^{\infty} r_{xy}[k] e^{-j\omega k}$$

여기서 $r_{xy}[k] = E\{x[n] y^*[n-k]\}$는 교차상관(cross-correlation)이다.

교차 PSD는 일반적으로 **복소수값**이다:

$$S_{xy}(e^{j\omega}) = |S_{xy}(e^{j\omega})| e^{j\phi_{xy}(\omega)}$$

### 12.2 코히런스

**크기 제곱 코히런스(Magnitude-Squared Coherence, MSC)**는 각 주파수에서 두 신호 간의 선형 관계를 측정한다:

$$C_{xy}(f) = \frac{|S_{xy}(f)|^2}{S_{xx}(f) \cdot S_{yy}(f)}$$

성질:
- $0 \leq C_{xy}(f) \leq 1$
- $C_{xy}(f) = 1$: 주파수 $f$에서 완전한 선형 관계
- $C_{xy}(f) = 0$: 주파수 $f$에서 선형 관계 없음
- 각 주파수에서의 $R^2$ (결정 계수)에 유사

### 12.3 위상 스펙트럼

**교차 스펙트럼 위상(cross-spectral phase)**은 대응하는 주파수 성분 간의 위상 차이를 보여준다:

$$\phi_{xy}(f) = \angle S_{xy}(f) = \arctan\frac{\text{Im}\{S_{xy}(f)\}}{\text{Re}\{S_{xy}(f)\}}$$

### 12.4 구현

```python
def cross_spectral_analysis():
    """Demonstrate cross-spectral analysis and coherence."""
    np.random.seed(42)
    N = 8192
    fs = 1000

    # System: y is a filtered and delayed version of x, plus independent noise
    t = np.arange(N) / fs

    # Input: broadband noise + tone
    x = np.random.randn(N) + 2*np.sin(2*np.pi*100*t)

    # System: bandpass filter + delay
    sos = signal.butter(4, [80, 200], btype='bandpass', fs=fs, output='sos')
    delay_samples = 20

    y_filtered = signal.sosfilt(sos, x)
    y = np.zeros(N)
    y[delay_samples:] = y_filtered[:N - delay_samples]
    y += 0.5 * np.random.randn(N)  # Add independent noise

    # Compute cross-spectral quantities
    f, Pxx = signal.welch(x, fs=fs, nperseg=512)
    f, Pyy = signal.welch(y, fs=fs, nperseg=512)
    f, Pxy = signal.csd(x, y, fs=fs, nperseg=512)

    # Coherence
    f_coh, Cxy = signal.coherence(x, y, fs=fs, nperseg=512)

    # Phase
    phase_xy = np.angle(Pxy)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Auto-PSDs
    axes[0, 0].semilogy(f, Pxx, 'b-', label='Sxx')
    axes[0, 0].semilogy(f, Pyy, 'r-', label='Syy')
    axes[0, 0].set_title('Auto Power Spectral Densities')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('PSD')
    axes[0, 0].legend()
    axes[0, 0].grid(True, which='both', alpha=0.3)

    # Cross-PSD magnitude
    axes[0, 1].semilogy(f, np.abs(Pxy), 'g-', linewidth=1.5)
    axes[0, 1].set_title('Cross-PSD |Sxy|')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('|Sxy|')
    axes[0, 1].grid(True, which='both', alpha=0.3)

    # Coherence
    axes[0, 2].plot(f_coh, Cxy, 'm-', linewidth=1.5)
    axes[0, 2].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 2].axhspan(80/500, 200/500, alpha=0.1, color='green',
                         transform=axes[0, 2].get_xaxis_transform())
    axes[0, 2].set_title('Coherence Cxy(f)')
    axes[0, 2].set_xlabel('Frequency (Hz)')
    axes[0, 2].set_ylabel('Coherence')
    axes[0, 2].set_ylim(0, 1.1)
    axes[0, 2].grid(True, alpha=0.3)

    # Phase spectrum (only where coherence is significant)
    significant = Cxy > 0.3
    f_sig = f_coh[significant]
    phase_sig = phase_xy[significant]

    axes[1, 0].plot(f_sig, phase_sig * 180 / np.pi, 'k.', markersize=3)
    axes[1, 0].set_title('Cross-Spectral Phase (where C > 0.3)')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Phase (degrees)')
    axes[1, 0].grid(True, alpha=0.3)

    # Transfer function estimate
    H_est = Pxy / (Pxx + 1e-15)
    w_sys, H_sys = signal.sosfreqz(sos, worN=len(f), fs=fs)

    axes[1, 1].plot(f, 20*np.log10(np.abs(H_est) + 1e-15), 'b-', alpha=0.7,
                     label='Estimated |H(f)|')
    axes[1, 1].plot(w_sys, 20*np.log10(np.abs(H_sys) + 1e-15), 'r--',
                     linewidth=2, label='True |H(f)|')
    axes[1, 1].set_title('Transfer Function Estimate')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude (dB)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Time-domain
    axes[1, 2].plot(t[:500]*1000, x[:500], 'b-', alpha=0.5, label='x')
    axes[1, 2].plot(t[:500]*1000, y[:500], 'r-', alpha=0.5, label='y')
    axes[1, 2].set_title('Time Domain')
    axes[1, 2].set_xlabel('Time (ms)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle('Cross-Spectral Analysis and Coherence', fontsize=14)
    plt.tight_layout()
    plt.savefig('cross_spectral.png', dpi=150)
    plt.close()

cross_spectral_analysis()
```

---

## 13. 스펙트로그램

### 13.1 시간-주파수 분석

**스펙트로그램(spectrogram)**은 신호의 주파수 내용이 시간에 따라 어떻게 변하는지를 보여준다. 단시간 푸리에 변환(Short-Time Fourier Transform, STFT)의 시퀀스로 계산된다:

$$S(t, f) = \left| \text{STFT}\{x[n]\}(t, f) \right|^2 = \left| \sum_{m=-\infty}^{\infty} x[m] w[m - t] e^{-j2\pi f m} \right|^2$$

스펙트로그램은 본질적으로 신호를 따라 윈도우를 슬라이딩하여 얻은 시변 PSD이다.

### 13.2 파라미터

- **윈도우 길이** (nperseg): 주파수 해상도 결정 ($\Delta f = f_s / \text{nperseg}$)
- **중첩(Overlap)**: 시간 해상도 결정 (더 많은 중첩 = 더 세밀한 시간 스텝)
- **윈도우 타입**: 스펙트럼 누설 제어 (Welch 방법과 동일)

### 13.3 시간-주파수 해상도

**불확정성 원리(uncertainty principle)**는 동시 시간 및 주파수 해상도를 제한한다:

$$\Delta t \cdot \Delta f \geq \frac{1}{4\pi}$$

- **긴 윈도우**: 높은 주파수 해상도, 낮은 시간 해상도
- **짧은 윈도우**: 높은 시간 해상도, 낮은 주파수 해상도

### 13.4 구현

```python
def spectrogram_analysis():
    """Demonstrate spectrogram with different parameters."""
    fs = 8000
    duration = 2.0
    t = np.arange(0, duration, 1/fs)
    N = len(t)

    # Create a time-varying signal
    # Chirp (frequency sweep from 100 to 3000 Hz)
    x_chirp = signal.chirp(t, f0=100, t1=duration, f1=3000, method='linear')

    # Add some tone bursts
    x_bursts = np.zeros(N)
    burst_start = int(0.5 * fs)
    burst_end = int(1.0 * fs)
    x_bursts[burst_start:burst_end] = np.sin(2*np.pi*1500*t[burst_start:burst_end])

    burst_start2 = int(1.2 * fs)
    burst_end2 = int(1.5 * fs)
    x_bursts[burst_start2:burst_end2] = 0.5*np.sin(2*np.pi*2500*t[burst_start2:burst_end2])

    x = x_chirp + x_bursts + 0.1*np.random.randn(N)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # Time domain
    axes[0, 0].plot(t, x, 'b-', linewidth=0.5)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Signal (Chirp + Tone Bursts + Noise)')
    axes[0, 0].grid(True, alpha=0.3)

    # Spectrum (overall)
    f_welch, Pxx = signal.welch(x, fs=fs, nperseg=1024)
    axes[0, 1].semilogy(f_welch, Pxx, 'b-', linewidth=1.5)
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('PSD')
    axes[0, 1].set_title('Overall PSD (Welch)')
    axes[0, 1].grid(True, which='both', alpha=0.3)

    # Spectrograms with different window sizes
    for idx, (nperseg, title_prefix) in enumerate([(64, 'Short'), (256, 'Medium'),
                                                     (1024, 'Long')]):
        row = 1 + idx // 2
        col = idx % 2
        if idx == 2:
            row, col = 2, 0

        f_spec, t_spec, Sxx = signal.spectrogram(x, fs=fs, nperseg=nperseg,
                                                    noverlap=nperseg*3//4,
                                                    window='hann')
        im = axes[row, col].pcolormesh(t_spec, f_spec,
                                        10*np.log10(Sxx + 1e-15),
                                        shading='gouraud', cmap='viridis',
                                        vmin=-60, vmax=0)
        axes[row, col].set_xlabel('Time (s)')
        axes[row, col].set_ylabel('Frequency (Hz)')
        axes[row, col].set_title(f'{title_prefix} Window (nperseg={nperseg}, '
                                  f'Δf={fs/nperseg:.0f} Hz, '
                                  f'Δt={nperseg/fs*1000:.1f} ms)')
        plt.colorbar(im, ax=axes[row, col], label='PSD (dB)')

    # matplotlib specgram for comparison
    axes[2, 1].specgram(x, NFFT=256, Fs=fs, noverlap=192,
                         cmap='viridis', vmin=-60, vmax=0)
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Frequency (Hz)')
    axes[2, 1].set_title('matplotlib specgram (NFFT=256)')

    plt.suptitle('Spectrogram: Time-Frequency Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig('spectrogram_analysis.png', dpi=150)
    plt.close()

spectrogram_analysis()
```

### 13.5 대화형 스펙트로그램 예제

```python
def music_like_spectrogram():
    """Create a spectrogram of a music-like signal."""
    fs = 44100
    duration = 3.0
    t = np.arange(0, duration, 1/fs)

    # Simulate a simple melody
    notes_hz = {
        'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
        'G4': 392.00, 'A4': 440.00, 'B4': 493.88, 'C5': 523.25,
    }

    melody = ['C4', 'E4', 'G4', 'C5', 'G4', 'E4', 'C4', 'D4',
              'F4', 'A4', 'C5', 'A4']
    note_duration = duration / len(melody)

    x = np.zeros(len(t))

    for i, note in enumerate(melody):
        start_idx = int(i * note_duration * fs)
        end_idx = int((i + 1) * note_duration * fs)
        if end_idx > len(t):
            end_idx = len(t)

        t_note = t[start_idx:end_idx] - t[start_idx]
        f0 = notes_hz[note]

        # Envelope (ADSR-like)
        env_len = end_idx - start_idx
        attack = int(0.05 * fs)
        decay = int(0.1 * fs)
        release = int(0.05 * fs)

        envelope = np.ones(env_len)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[attack:attack+decay] = np.linspace(1, 0.7, decay)
        envelope[-release:] = np.linspace(0.7, 0, release)

        # Add harmonics
        for harmonic in [1, 2, 3, 4]:
            x[start_idx:end_idx] += (envelope *
                np.sin(2*np.pi*f0*harmonic*t_note) / harmonic)

    x += 0.02 * np.random.randn(len(x))  # Add slight noise

    # Spectrogram
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(t, x, 'b-', linewidth=0.3)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Music-like Signal (Time Domain)')
    axes[0].grid(True, alpha=0.3)

    f_spec, t_spec, Sxx = signal.spectrogram(x, fs=fs, nperseg=4096,
                                               noverlap=3072, window='hann')
    im = axes[1].pcolormesh(t_spec, f_spec, 10*np.log10(Sxx + 1e-15),
                             shading='gouraud', cmap='magma',
                             vmin=-80, vmax=-10)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_ylim(0, 3000)
    axes[1].set_title('Spectrogram')
    plt.colorbar(im, ax=axes[1], label='PSD (dB)')

    plt.tight_layout()
    plt.savefig('music_spectrogram.png', dpi=150)
    plt.close()

music_like_spectrogram()
```

---

## 14. Python 구현

### 14.1 종합 스펙트럼 분석 툴킷

```python
class SpectralAnalyzer:
    """Complete spectral analysis toolkit."""

    def __init__(self, x, fs):
        self.x = np.array(x, dtype=float)
        self.fs = fs
        self.N = len(x)

    def periodogram(self, nfft=None, window='boxcar'):
        """Compute periodogram."""
        return signal.periodogram(self.x, fs=self.fs, nfft=nfft, window=window)

    def welch(self, nperseg=256, noverlap=None, window='hann'):
        """Compute Welch's PSD estimate."""
        if noverlap is None:
            noverlap = nperseg // 2
        return signal.welch(self.x, fs=self.fs, nperseg=nperseg,
                           noverlap=noverlap, window=window)

    def ar_psd(self, order=None, method='burg', nfft=1024):
        """Compute AR model PSD."""
        if order is None:
            # Auto-select using BIC
            best_orders, _ = model_order_selection(self.x, max_order=50)
            order = best_orders['BIC']
            print(f"Auto-selected AR order: p = {order} (BIC)")

        if method == 'burg':
            a, sigma2 = burg_method(self.x, order)
        else:
            r_full = np.correlate(self.x, self.x, mode='full') / self.N
            center = self.N - 1
            r = r_full[center:center + order + 1]
            a, sigma2, _ = levinson_durbin(r, order)

        f = np.linspace(0, self.fs/2, nfft//2 + 1)
        omega = 2 * np.pi * f / self.fs

        A_freq = np.ones(len(f), dtype=complex)
        for k in range(len(a)):
            A_freq += a[k] * np.exp(-1j * (k+1) * omega)

        Pxx = sigma2 / (np.abs(A_freq)**2 * self.fs)
        return f, Pxx

    def spectrogram(self, nperseg=256, noverlap=None, window='hann'):
        """Compute spectrogram."""
        if noverlap is None:
            noverlap = nperseg * 3 // 4
        return signal.spectrogram(self.x, fs=self.fs, nperseg=nperseg,
                                   noverlap=noverlap, window=window)

    def comprehensive_analysis(self, save_path='comprehensive_spectral.png'):
        """Perform and plot comprehensive spectral analysis."""
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))

        # Time domain
        t = np.arange(self.N) / self.fs
        axes[0, 0].plot(t, self.x, 'b-', linewidth=0.5)
        axes[0, 0].set_title('Time Domain')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)

        # Periodogram
        f_per, Pxx_per = self.periodogram()
        axes[0, 1].semilogy(f_per, Pxx_per, 'b-', alpha=0.5, linewidth=0.5)
        axes[0, 1].set_title('Periodogram')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('PSD')
        axes[0, 1].grid(True, which='both', alpha=0.3)

        # Welch
        f_w, Pxx_w = self.welch(nperseg=min(self.N // 4, 512))
        axes[1, 0].semilogy(f_w, Pxx_w, 'g-', linewidth=1.5)
        axes[1, 0].set_title("Welch's Method")
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('PSD')
        axes[1, 0].grid(True, which='both', alpha=0.3)

        # AR PSD
        f_ar, Pxx_ar = self.ar_psd()
        axes[1, 1].semilogy(f_ar, Pxx_ar, 'r-', linewidth=1.5,
                             label='AR (Burg)')
        axes[1, 1].semilogy(f_w, Pxx_w, 'g--', alpha=0.5, label='Welch')
        axes[1, 1].set_title('AR Spectral Estimate')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('PSD')
        axes[1, 1].legend()
        axes[1, 1].grid(True, which='both', alpha=0.3)

        # Spectrogram
        f_spec, t_spec, Sxx = self.spectrogram(nperseg=min(self.N // 8, 256))
        im = axes[2, 0].pcolormesh(t_spec, f_spec,
                                     10*np.log10(Sxx + 1e-15),
                                     shading='gouraud', cmap='viridis')
        axes[2, 0].set_title('Spectrogram')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Frequency (Hz)')
        plt.colorbar(im, ax=axes[2, 0], label='PSD (dB)')

        # Autocorrelation
        max_lag = min(500, self.N // 4)
        r = np.correlate(self.x, self.x, mode='full') / self.N
        center = self.N - 1
        lags = np.arange(-max_lag, max_lag + 1)
        axes[2, 1].plot(lags / self.fs * 1000,
                         r[center-max_lag:center+max_lag+1], 'b-', linewidth=1)
        axes[2, 1].set_title('Autocorrelation')
        axes[2, 1].set_xlabel('Lag (ms)')
        axes[2, 1].set_ylabel('r_xx[k]')
        axes[2, 1].grid(True, alpha=0.3)

        plt.suptitle('Comprehensive Spectral Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        return fig

# Example usage
np.random.seed(42)
fs = 1000
N = 4096
t = np.arange(N) / fs

# Complex signal: tones + AR noise + chirp
x = (np.sin(2*np.pi*50*t) +
     0.5*np.sin(2*np.pi*120*t) +
     signal.chirp(t, 200, t[-1], 300) * 0.3 +
     0.3*np.random.randn(N))

analyzer = SpectralAnalyzer(x, fs)
analyzer.comprehensive_analysis()
```

### 14.2 실세계 예제: 진동 분석

```python
def vibration_analysis_demo():
    """Simulate vibration analysis of a rotating machine."""
    fs = 10000  # 10 kHz sampling
    duration = 5.0
    t = np.arange(0, duration, 1/fs)
    N = len(t)

    # Fundamental rotation frequency
    f_rot = 30  # Hz (1800 RPM)

    # Normal vibration: fundamental + harmonics
    x_normal = (0.5 * np.sin(2*np.pi*f_rot*t) +
                0.2 * np.sin(2*np.pi*2*f_rot*t) +
                0.1 * np.sin(2*np.pi*3*f_rot*t) +
                0.05 * np.random.randn(N))

    # Faulty vibration: bearing defect frequency
    f_bearing = 4.7 * f_rot  # Ball pass frequency (outer race)
    x_faulty = (0.5 * np.sin(2*np.pi*f_rot*t) +
                0.2 * np.sin(2*np.pi*2*f_rot*t) +
                0.1 * np.sin(2*np.pi*3*f_rot*t) +
                0.3 * np.sin(2*np.pi*f_bearing*t) +  # Bearing defect
                0.15 * np.sin(2*np.pi*2*f_bearing*t) +
                0.05 * np.random.randn(N))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PSDs
    f_n, Pxx_n = signal.welch(x_normal, fs=fs, nperseg=2048)
    f_f, Pxx_f = signal.welch(x_faulty, fs=fs, nperseg=2048)

    axes[0, 0].semilogy(f_n, Pxx_n, 'b-', linewidth=1.5)
    axes[0, 0].set_title('Normal Machine PSD')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('PSD')
    axes[0, 0].set_xlim(0, 500)
    for h in range(1, 4):
        axes[0, 0].axvline(h*f_rot, color='g', linestyle='--', alpha=0.5)
    axes[0, 0].grid(True, which='both', alpha=0.3)

    axes[0, 1].semilogy(f_f, Pxx_f, 'r-', linewidth=1.5)
    axes[0, 1].set_title('Faulty Machine PSD (Bearing Defect)')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('PSD')
    axes[0, 1].set_xlim(0, 500)
    for h in range(1, 4):
        axes[0, 1].axvline(h*f_rot, color='g', linestyle='--', alpha=0.5)
    axes[0, 1].axvline(f_bearing, color='red', linestyle=':', alpha=0.7,
                         label=f'BPFO = {f_bearing:.0f} Hz')
    axes[0, 1].axvline(2*f_bearing, color='red', linestyle=':', alpha=0.5)
    axes[0, 1].legend()
    axes[0, 1].grid(True, which='both', alpha=0.3)

    # Overlay comparison
    axes[1, 0].semilogy(f_n, Pxx_n, 'b-', linewidth=1.5, label='Normal')
    axes[1, 0].semilogy(f_f, Pxx_f, 'r-', linewidth=1.5, label='Faulty')
    axes[1, 0].set_title('Comparison')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('PSD')
    axes[1, 0].set_xlim(0, 500)
    axes[1, 0].legend()
    axes[1, 0].grid(True, which='both', alpha=0.3)

    # Spectrogram (faulty - developing fault)
    # Simulate fault developing over time
    x_developing = np.zeros(N)
    fault_amplitude = np.linspace(0, 0.4, N)  # Gradually increasing
    x_developing = (0.5 * np.sin(2*np.pi*f_rot*t) +
                    0.2 * np.sin(2*np.pi*2*f_rot*t) +
                    fault_amplitude * np.sin(2*np.pi*f_bearing*t) +
                    0.05 * np.random.randn(N))

    f_spec, t_spec, Sxx = signal.spectrogram(x_developing, fs=fs,
                                               nperseg=2048, noverlap=1536)
    im = axes[1, 1].pcolormesh(t_spec, f_spec, 10*np.log10(Sxx + 1e-15),
                                shading='gouraud', cmap='inferno',
                                vmin=-60, vmax=-10)
    axes[1, 1].set_title('Developing Fault (Spectrogram)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Frequency (Hz)')
    axes[1, 1].set_ylim(0, 500)
    plt.colorbar(im, ax=axes[1, 1], label='PSD (dB)')

    plt.suptitle('Vibration Spectral Analysis: Machine Condition Monitoring', fontsize=14)
    plt.tight_layout()
    plt.savefig('vibration_analysis.png', dpi=150)
    plt.close()

vibration_analysis_demo()
```

---

## 15. 연습 문제

### 연습 문제 1: 주기도 vs Welch

100, 150, 200 Hz에서 진폭 1.0, 0.5, 0.1인 세 개의 정현파와 가법 백색 가우시안 잡음(SNR = 10 dB)으로 구성된 신호를 생성하라. $f_s = 1000$ Hz, $N = 4096$ 샘플을 사용한다.

(a) 주기도를 계산하고 그려라. 세 개의 음조를 모두 식별할 수 있는가?

(b) `nperseg = 256`과 `512`를 사용하여 Welch 방법을 적용하라. 어느 설정이 음조를 더 잘 분해하는가?

(c) 윈도우 함수 선택(`boxcar`, `hann`, `blackmanharris`)이 가장 약한 음조(200 Hz)를 감지하는 능력에 어떤 영향을 미치는가?

(d) 100번의 몬테 카를로 시행을 실행하여 각 PSD 추정치의 분산을 계산하라. 각 방법에 대해 주파수의 함수로 분산을 그려라.

### 연습 문제 2: AR 모델 식별

알려진 계수를 갖는 AR(4) 과정을 생성하라:
$x[n] + 1.5x[n-1] - 0.75x[n-2] + 0.2x[n-3] - 0.05x[n-4] = w[n]$

(a) $N = 512$ 샘플을 생성하고 차수 $p = 2, 4, 6, 8, 12$에 대해 Yule-Walker 방법을 사용하여 AR 파라미터를 추정하라.

(b) 각 차수에 대해 추정된 PSD와 참 PSD를 함께 계산하고 그려라.

(c) AIC와 BIC를 사용하여 최적 차수를 선택하라. 둘 다 동의하는가?

(d) $N = 64$ 샘플로 반복하라. 데이터 길이 감소가 결과에 어떤 영향을 미치는가?

### 연습 문제 3: 교차 스펙트럼 분석

두 센서가 구조물의 서로 다른 위치에서 진동을 측정한다. 이들 사이의 시스템 전달 함수는 150 Hz를 중심으로 하는 4차 대역통과 필터이다.

(a) 입력(광대역 랜덤 가진)과 출력(필터링 + 독립 잡음)을 시뮬레이션하라.

(b) 코히런스 함수를 추정하라. 어느 주파수에서 코히런스가 가장 높은가?

(c) 교차 스펙트럼 밀도로부터 전달 함수 크기와 위상을 추정하라.

(d) 위상 스펙트럼으로부터 군지연(group delay)을 계산하라. 알려진 필터 군지연과 일치하는가?

### 연습 문제 4: 박쥐 반향 위치 측정 스펙트로그램

다음으로 구성된 합성 박쥐 반향 위치 측정 신호를 만들어라:
- 각각 100 kHz에서 50 kHz로 5 ms 동안 스윕하는 5개의 처프(chirp) 시퀀스
- 50 ms의 펄스 간격
- -20 dB의 가법 잡음

(a) Hanning 윈도우를 사용하여 스펙트로그램을 그려라. `nperseg = 64, 256, 1024`로 실험하여 최적의 절충점을 찾아라.

(b) 스펙트로그램 피크에서 순간 주파수를 측정하라. 알려진 처프 속도와 비교하라.

(c) 에코(감쇠되고 지연된 복사본)를 추가하라. 스펙트로그램에서 에코를 감지할 수 있는가? 최소 감지 가능 지연은 얼마인가?

### 연습 문제 5: 해상도 한계

주파수 $f_1$과 $f_2 = f_1 + \Delta f$의 등폭 두 정현파가 잡음 속에 있다.

(a) $f_1 = 100$ Hz, $f_s = 1000$ Hz, $N = 256$일 때: 주기도가 두 음조를 분해할 수 있는 최소 $\Delta f$는 얼마인가? 실험적으로 검증하라.

(b) $\Delta f = 5$ Hz에 대해 주기도, Welch 방법 ($\text{nperseg} = 128$), AR Burg 방법 ($p = 20$)의 해상도를 비교하라.

(c) SNR이 해상도에 어떤 영향을 미치는가? SNR = 0, 10, 20, 40 dB로 테스트하고 각 방법에 대해 최소 분해 가능한 $\Delta f$를 그려라.

### 연습 문제 6: 실세계 응용: EEG 분석

다음을 포함하는 단순화된 EEG 신호를 시뮬레이션하라:
- "눈 감음" 상태 동안의 알파 대역 활동(8-13 Hz)
- "눈 뜸" 상태 동안의 베타 대역 활동(13-30 Hz)
- $t = 5$ s에서 눈 감음에서 눈 뜸으로의 전환

(a) $f_s = 256$ Hz에서 두 상태 간에 전환하는 10초 신호를 만들어라.

(b) 스펙트로그램을 계산하고 전환 지점을 식별하라.

(c) 시간에 따른 알파와 베타 대역의 대역 전력(주파수 대역에서 적분된 총 PSD)을 계산하라. 이를 시계열로 그려라.

(d) 알파/베타 전력 비율을 계산하라. 상태 전환을 얼마나 명확하게 감지할 수 있는가?

### 연습 문제 7: 짧은 데이터에서 Burg vs Welch

점점 더 짧은 데이터 기록에 대해 Burg의 AR 방법과 Welch 방법을 비교하라.

(a) 100 Hz와 108 Hz에서 두 음조를 갖는 신호를 생성하라 ($f_s = 1000$ Hz). $N$을 32에서 4096까지 (2의 거듭제곱) 변화시켜라.

(b) 각 $N$에 대해 Welch (nperseg = N/2)와 Burg (order = 20) PSD 추정치를 모두 계산하라.

(c) "해상도 성공" 기준을 정의하라: 두 피크 사이에 최소 3 dB의 노치가 있으면 두 음조가 분해된 것으로 본다. 각 방법에 대해 최소 $N$을 결정하라.

(d) 두 방법에 대해 $N$의 함수로 해상도 성공률(100번의 몬테 카를로 시행에 걸쳐)을 그려라.

---

## 참고문헌

1. **Oppenheim, A. V., & Schafer, R. W. (2010).** *Discrete-Time Signal Processing* (3rd ed.). Pearson. Chapter 10.
2. **Kay, S. M. (1988).** *Modern Spectral Estimation: Theory and Application*. Prentice Hall. [모수적 방법의 고전 참고서]
3. **Stoica, P., & Moses, R. L. (2005).** *Spectral Analysis of Signals*. Prentice Hall. [우수한 현대적 처리]
4. **Welch, P. D. (1967).** "The Use of Fast Fourier Transform for the Estimation of Power Spectra." *IEEE Trans. Audio Electroacoustics*, 15(2), 70-73.
5. **Burg, J. P. (1975).** "Maximum Entropy Spectral Analysis." PhD Dissertation, Stanford University.
6. **Marple, S. L. (1987).** *Digital Spectral Analysis with Applications*. Prentice Hall.
7. **SciPy Documentation** -- 스펙트럼 분석 함수: https://docs.scipy.org/doc/scipy/reference/signal.html#spectral-analysis

---

## 내비게이션

- 이전: [11. 다중 레이트 신호 처리](11_Multirate_Processing.md)
- 다음: [13. 적응 필터](13_Adaptive_Filters.md)
- [개요로 돌아가기](00_Overview.md)
