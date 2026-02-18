# 표본화와 신호 복원

## 개요

표본화(Sampling)는 연속 시간(아날로그) 신호와 이산 시간(디지털) 신호를 연결하는 다리입니다. 표본화 과정, 그 한계, 그리고 원래 신호를 충실히 복원하는 방법을 이해하는 것은 모든 디지털 신호 처리(DSP)의 근본입니다. 이 레슨에서는 나이퀴스트-섀넌(Nyquist-Shannon) 표본화 정리, 에일리어싱(aliasing), 에일리어싱 방지 전략, 복원 기술, 그리고 실용적인 ADC/DAC 고려 사항을 다룹니다.

**학습 목표:**
- 이상적인 표본화의 수학적 체계를 이해한다
- 나이퀴스트-섀넌 표본화 정리를 진술하고 증명한다
- 에일리어싱을 식별하고 방지한다
- sinc 보간법(sinc interpolation)과 홀드 회로를 사용한 복원을 구현한다
- 실용적인 ADC/DAC 시스템과 오버샘플링(oversampling)의 이점을 분석한다

**선수 과목:** [04. 푸리에 변환과 주파수 영역](04_Fourier_Transform.md)

---

## 1. 연속 시간 신호의 표본화

### 1.1 왜 표본화를 하는가?

디지털 시스템(컴퓨터, DSP 칩, 마이크로컨트롤러)은 연속 시간 신호를 직접 처리할 수 없습니다. 아날로그 신호를 숫자의 나열로 변환해야 합니다. 핵심 질문: **어떤 조건에서 원래 신호를 샘플로부터 완벽하게 복원할 수 있는가?**

### 1.2 표본화 과정

연속 시간 신호 $x(t)$가 주어지면, 표본화 주기(sampling period) $T_s$로 표본화하면 이산 시간 신호를 얻습니다:

$$x[n] = x(nT_s), \quad n \in \mathbb{Z}$$

표본화 주파수(sample rate)는:

$$f_s = \frac{1}{T_s} \quad \text{(Hz)}$$

각 표본화 주파수는:

$$\Omega_s = 2\pi f_s = \frac{2\pi}{T_s} \quad \text{(rad/s)}$$

```python
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_sampling():
    """Demonstrate the basic sampling process."""
    # Continuous-time signal (simulated with dense grid)
    t_cont = np.linspace(0, 1, 10000)
    f_signal = 5  # 5 Hz sine wave
    x_cont = np.sin(2 * np.pi * f_signal * t_cont)

    # Sample at different rates
    sample_rates = [10, 20, 50]  # Hz

    fig, axes = plt.subplots(len(sample_rates), 1, figsize=(12, 10))

    for ax, fs in zip(axes, sample_rates):
        Ts = 1.0 / fs
        n_samples = np.arange(0, 1, Ts)
        x_samples = np.sin(2 * np.pi * f_signal * n_samples)

        ax.plot(t_cont, x_cont, 'b-', alpha=0.5, label='Continuous signal')
        ax.stem(n_samples, x_samples, linefmt='r-', markerfmt='ro',
                basefmt='k-', label=f'Samples (fs={fs} Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Sampling at fs = {fs} Hz (Ts = {Ts:.3f} s)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sampling_basic.png', dpi=150)
    plt.show()

demonstrate_sampling()
```

---

## 2. 임펄스 열을 이용한 이상적 표본화

### 2.1 수학적 모델

이상적 표본화는 $x(t)$에 임펄스 열(Dirac 빗, Dirac comb)을 곱합니다:

$$s(t) = \sum_{n=-\infty}^{\infty} \delta(t - nT_s)$$

표본화된 신호는:

$$x_s(t) = x(t) \cdot s(t) = \sum_{n=-\infty}^{\infty} x(nT_s) \, \delta(t - nT_s)$$

### 2.2 주파수 영역 분석

임펄스 열의 푸리에 변환은 또 다른 임펄스 열입니다:

$$S(j\Omega) = \frac{2\pi}{T_s} \sum_{k=-\infty}^{\infty} \delta\!\left(\Omega - k\Omega_s\right)$$

시간 영역에서의 곱셈은 주파수 영역에서의 합성곱에 해당하므로:

$$X_s(j\Omega) = \frac{1}{2\pi} X(j\Omega) * S(j\Omega)$$

대입하고 정리하면:

$$\boxed{X_s(j\Omega) = \frac{1}{T_s} \sum_{k=-\infty}^{\infty} X\!\left(j(\Omega - k\Omega_s)\right)}$$

이것이 **표본화의 근본적 결과**입니다: 표본화된 신호의 스펙트럼은 원래 스펙트럼 $X(j\Omega)$의 주기적 반복이며, $\Omega_s$의 정수 배만큼 이동하고 $1/T_s$로 스케일됩니다.

### 2.3 스펙트럼 복제 시각화

```python
def visualize_spectral_replication():
    """Show how sampling replicates the spectrum periodically."""
    # Original signal: bandlimited with max frequency B
    B = 5  # Hz (bandwidth)
    f = np.linspace(-20, 20, 2000)

    # Triangular spectrum (simplified representation)
    X_orig = np.maximum(0, 1 - np.abs(f) / B)

    # Sampling frequency
    fs_values = [15, 10, 7]  # Hz
    titles = [
        f'fs = 15 Hz > 2B = 10 Hz (No aliasing)',
        f'fs = 10 Hz = 2B (Critical sampling)',
        f'fs = 7 Hz < 2B (Aliasing!)'
    ]

    fig, axes = plt.subplots(len(fs_values) + 1, 1, figsize=(14, 12))

    # Original spectrum
    axes[0].plot(f, X_orig, 'b-', linewidth=2)
    axes[0].fill_between(f, X_orig, alpha=0.3)
    axes[0].set_title('Original Spectrum X(f)')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('|X(f)|')
    axes[0].set_xlim(-20, 20)
    axes[0].grid(True, alpha=0.3)

    for ax, fs, title in zip(axes[1:], fs_values, titles):
        # Sum of shifted replicas
        X_sampled = np.zeros_like(f)
        for k in range(-5, 6):
            X_sampled += np.maximum(0, 1 - np.abs(f - k * fs) / B)

        ax.plot(f, X_sampled, 'r-', linewidth=2)
        ax.fill_between(f, X_sampled, alpha=0.3, color='red')

        # Show individual replicas as dashed
        for k in range(-3, 4):
            replica = np.maximum(0, 1 - np.abs(f - k * fs) / B)
            ax.plot(f, replica, 'b--', alpha=0.4, linewidth=1)

        # Mark Nyquist boundaries
        ax.axvline(fs / 2, color='green', linestyle=':', linewidth=1.5,
                    label=f'fs/2 = {fs/2} Hz')
        ax.axvline(-fs / 2, color='green', linestyle=':', linewidth=1.5)

        ax.set_title(title)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('|Xs(f)|')
        ax.set_xlim(-20, 20)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('spectral_replication.png', dpi=150)
    plt.show()

visualize_spectral_replication()
```

---

## 3. 나이퀴스트-섀넌 표본화 정리

### 3.1 정리의 진술

**정리 (나이퀴스트-섀넌):** $B$ Hz 이상의 주파수 성분이 없는 대역 제한 신호(bandlimited signal) $x(t)$ (즉, $|\Omega| > 2\pi B$에서 $X(j\Omega) = 0$)는, 표본화 속도가 다음 조건을 만족하면 샘플 $x[n] = x(nT_s)$로부터 완전히 복원될 수 있습니다:

$$\boxed{f_s > 2B}$$

또는 동치 표현:

$$T_s < \frac{1}{2B}$$

### 3.2 핵심 용어 정의

| 용어 | 정의 | 표현 |
|------|-----------|------------|
| **나이퀴스트 레이트(Nyquist rate)** | 에일리어싱을 방지하는 최소 표본화 속도 | $f_{\text{Nyquist}} = 2B$ |
| **나이퀴스트 주파수(Nyquist frequency)** | 표본화 속도 $f_s$에서 표현 가능한 최고 주파수 | $f_{\text{max}} = f_s / 2$ |
| **나이퀴스트 간격(Nyquist interval)** | 샘플 사이의 최대 간격 | $T_{\text{Nyquist}} = 1/(2B)$ |

> **중요한 구별:** 나이퀴스트 *레이트*는 신호에 의존하고($2B$), 나이퀴스트 *주파수*는 표본화 속도에 의존합니다($f_s/2$). 이 둘은 임계 표본화 조건에서만 같습니다.

### 3.3 증명 개요

1. 표본화된 스펙트럼은 $X_s(j\Omega) = \frac{1}{T_s} \sum_k X(j(\Omega - k\Omega_s))$입니다.
2. $\Omega_s > 2(2\pi B)$이면, 이동된 복제본들이 겹치지 않습니다.
3. 차단 주파수 $\Omega_s / 2$의 이상적 저역통과 필터를 적용하여 $X(j\Omega)$를 복원할 수 있습니다.
4. 따라서 $x(t)$는 샘플로부터 완전히 복원 가능합니다.

### 3.4 복원 공식

표본화 정리가 만족되면, 원래 신호를 정확히 복원할 수 있습니다:

$$\boxed{x(t) = \sum_{n=-\infty}^{\infty} x[n] \, \operatorname{sinc}\!\left(\frac{t - nT_s}{T_s}\right)}$$

여기서 $\operatorname{sinc}(u) = \frac{\sin(\pi u)}{\pi u}$입니다.

이것이 **휘태커-섀넌 보간 공식(Whittaker-Shannon interpolation formula)**(이상적 sinc 보간법)입니다.

```python
def demonstrate_nyquist_theorem():
    """Demonstrate the sampling theorem with different rates."""
    # Signal: sum of sinusoids
    f1, f2 = 3, 7  # Hz
    B = 7  # Bandwidth
    nyquist_rate = 2 * B  # 14 Hz

    t_cont = np.linspace(0, 1, 10000)
    x_cont = np.sin(2 * np.pi * f1 * t_cont) + 0.5 * np.sin(2 * np.pi * f2 * t_cont)

    sample_rates = [30, 14, 10]  # Above, at, below Nyquist rate
    labels = [
        f'fs = 30 Hz > 2B = {nyquist_rate} Hz (Well sampled)',
        f'fs = 14 Hz = 2B = {nyquist_rate} Hz (Critical)',
        f'fs = 10 Hz < 2B = {nyquist_rate} Hz (Undersampled!)'
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    for ax, fs, label in zip(axes, sample_rates, labels):
        Ts = 1.0 / fs
        n_samples = np.arange(0, 1, Ts)
        x_samples = np.sin(2 * np.pi * f1 * n_samples) + \
                    0.5 * np.sin(2 * np.pi * f2 * n_samples)

        # Sinc interpolation to reconstruct
        t_recon = np.linspace(0, 1, 10000)
        x_recon = np.zeros_like(t_recon)
        for i, ns in enumerate(n_samples):
            x_recon += x_samples[i] * np.sinc((t_recon - ns) / Ts)

        ax.plot(t_cont, x_cont, 'b-', alpha=0.4, linewidth=1, label='Original')
        ax.plot(t_recon, x_recon, 'r-', linewidth=1.5, label='Reconstructed')
        ax.stem(n_samples, x_samples, linefmt='g-', markerfmt='go',
                basefmt='k-', label='Samples')
        ax.set_title(label)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('nyquist_theorem.png', dpi=150)
    plt.show()

demonstrate_nyquist_theorem()
```

---

## 4. 에일리어싱

### 4.1 에일리어싱이란 무엇인가?

에일리어싱(aliasing)은 표본화 속도가 충분하지 않을 때($f_s < 2B$) 발생하며, 스펙트럼 복제본들이 겹칩니다. 고주파 성분이 "접혀" 더 낮은 주파수로 나타나며, 실제 저주파 신호 내용과 **구별할 수 없게** 됩니다.

표본화 속도 $f_s$에서 주파수 $f_0$의 성분이 에일리어싱되면, 에일리어스 주파수는:

$$f_{\text{alias}} = |f_0 - k \cdot f_s|, \quad k = \operatorname{round}(f_0 / f_s)$$

더 정확하게, 표본화 후 겉보기 주파수는:

$$f_{\text{apparent}} = \left| \left( (f_0 + f_s/2) \bmod f_s \right) - f_s/2 \right|$$

### 4.2 에일리어싱 예제

```python
def demonstrate_aliasing():
    """Show aliasing with concrete sinusoidal examples."""
    fs = 20  # Sampling frequency: 20 Hz
    nyquist = fs / 2  # 10 Hz

    # Three signals: below, at, and above Nyquist
    freqs = [3, 10, 17]  # Hz
    # f=3 Hz: well below Nyquist, sampled correctly
    # f=10 Hz: at Nyquist, ambiguous
    # f=17 Hz: above Nyquist, aliases to |17 - 20| = 3 Hz

    t_cont = np.linspace(0, 1, 10000)
    Ts = 1.0 / fs
    t_samples = np.arange(0, 1, Ts)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    for ax, f in zip(axes, freqs):
        x_cont = np.cos(2 * np.pi * f * t_cont)
        x_samples = np.cos(2 * np.pi * f * t_samples)

        ax.plot(t_cont, x_cont, 'b-', alpha=0.5, linewidth=1,
                label=f'Original: {f} Hz')

        # Show the aliased frequency
        if f > nyquist:
            f_alias = fs - f  # For f < fs
            x_alias = np.cos(2 * np.pi * f_alias * t_cont)
            ax.plot(t_cont, x_alias, 'r--', linewidth=1.5,
                    label=f'Alias: {f_alias} Hz')

        ax.stem(t_samples, x_samples, linefmt='g-', markerfmt='go',
                basefmt='k-', label='Samples')
        ax.set_title(f'f = {f} Hz, fs = {fs} Hz, Nyquist = {nyquist} Hz')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('aliasing_demo.png', dpi=150)
    plt.show()

demonstrate_aliasing()
```

### 4.3 실제 에일리어싱 시나리오

| 시나리오 | 원인 | 효과 |
|----------|-------|--------|
| **오디오** | 나이퀴스트/2 초과 주파수 녹음 | 거칠고 금속성 아티팩트 |
| **비디오** | 영화 회전 바퀴 (24 fps) | 바퀴가 거꾸로 돌아가는 것처럼 보임 |
| **사진** | 미세한 패턴 (무아레 효과) | 가짜 색상/패턴 아티팩트 |
| **의료 영상** | 불충분한 공간 표본화 | MRI/CT에서 공간 에일리어싱 |
| **레이더** | PRF가 목표물 속도에 비해 낮음 | 속도 모호성 |

### 4.4 마차 바퀴 효과

시간적 에일리어싱의 고전적인 예는 영화에서의 "마차 바퀴 효과"입니다:

- 바퀴가 $f_{\text{wheel}}$ Hz(초당 회전수)로 회전합니다
- 카메라가 $f_s = 24$ fps로 표본화합니다
- $f_{\text{wheel}} > 12$ Hz이면, 바퀴는 거꾸로 돌거나 잘못된 속도로 회전하는 것처럼 보입니다

```python
def wagon_wheel_effect():
    """Simulate the wagon wheel aliasing effect."""
    fps = 24  # Camera frame rate

    # Wheel rotation frequencies
    rotation_freqs = [5, 12, 23, 25, 47, 48]

    print("Wagon Wheel Effect (Camera: 24 fps)")
    print("=" * 50)

    for f_rot in rotation_freqs:
        # Apparent frequency after aliasing
        f_apparent = f_rot % fps
        if f_apparent > fps / 2:
            f_apparent = f_apparent - fps
        print(f"  Actual: {f_rot:3d} Hz -> Apparent: {f_apparent:+6.1f} Hz "
              f"({'backward' if f_apparent < 0 else 'forward'})")

wagon_wheel_effect()
```

출력:
```
Wagon Wheel Effect (Camera: 24 fps)
==================================================
  Actual:   5 Hz -> Apparent:   +5.0 Hz (forward)
  Actual:  12 Hz -> Apparent:  +12.0 Hz (forward)
  Actual:  23 Hz -> Apparent:   -1.0 Hz (backward)
  Actual:  25 Hz -> Apparent:   +1.0 Hz (forward)
  Actual:  47 Hz -> Apparent:   -1.0 Hz (backward)
  Actual:  48 Hz -> Apparent:   +0.0 Hz (forward)
```

---

## 5. 에일리어싱 방지 필터

### 5.1 목적

**에일리어싱 방지 필터(anti-aliasing filter)**는 표본화 *이전*에 연속 시간 신호에 적용되는 저역통과 필터입니다. $f_s/2$ 이상의 주파수 성분을 제거(감쇠)하여 에일리어싱을 방지합니다.

### 5.2 이상적 필터 vs. 실용적 에일리어싱 방지

| 특성 | 이상적 필터 | 실용적 필터 |
|----------|-------------|-----------------|
| 차단 주파수 | $f_s/2$에서 벽돌 벽(brick-wall) | 점진적 감쇠 |
| 전이 대역 | 폭 없음 | 유한한 폭 |
| 저지 대역 감쇠 | 무한 | 유한 (예: -80 dB) |
| 인과성 | 비인과적 | 인과적 |
| 구현 | 불가능 | 아날로그 회로 |

### 5.3 실용적 고려 사항

이상적인 벽돌 벽 필터는 실현 불가능하므로, 실용적 에일리어싱 방지는 다음을 포함합니다:

1. **가드 밴드**: 필터 감쇠 롤오프를 허용하기 위해 $2B$보다 다소 높게 표본화
2. **필터 차수**: 고차 필터는 더 급격한 롤오프를 갖지만 위상 왜곡이 더 큼
3. **일반적 선택**: 버터워스(Butterworth, 최대 평탄도), 체비쇼프(Chebyshev, 더 급격한 롤오프), 타원(Elliptic, 가장 급격한 롤오프)

```python
from scipy import signal

def design_anti_aliasing_filter():
    """Design and compare anti-aliasing filters."""
    fs = 1000  # Sampling frequency
    f_nyquist = fs / 2
    f_cutoff = 400  # Desired signal bandwidth

    # Normalized cutoff for analog filter design
    Wn = 2 * np.pi * f_cutoff  # Angular frequency

    orders = [2, 4, 8]
    f_plot = np.logspace(1, 3.5, 1000)  # 10 Hz to ~3 kHz
    w_plot = 2 * np.pi * f_plot

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Butterworth filters of different orders
    for order in orders:
        b, a = signal.butter(order, Wn, btype='low', analog=True)
        w, H = signal.freqs(b, a, worN=w_plot)

        mag_db = 20 * np.log10(np.abs(H))
        phase_deg = np.angle(H, deg=True)

        axes[0].semilogx(w / (2 * np.pi), mag_db,
                         label=f'Butterworth order {order}')
        axes[1].semilogx(w / (2 * np.pi), phase_deg,
                         label=f'Butterworth order {order}')

    # Mark important frequencies
    for ax in axes:
        ax.axvline(f_cutoff, color='green', linestyle='--', alpha=0.7,
                   label=f'Cutoff = {f_cutoff} Hz')
        ax.axvline(f_nyquist, color='red', linestyle='--', alpha=0.7,
                   label=f'Nyquist = {f_nyquist} Hz')
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title('Anti-Aliasing Filter: Magnitude Response')
    axes[0].set_ylim(-80, 5)

    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Phase (degrees)')
    axes[1].set_title('Anti-Aliasing Filter: Phase Response')

    plt.tight_layout()
    plt.savefig('anti_aliasing_filter.png', dpi=150)
    plt.show()

design_anti_aliasing_filter()
```

---

## 6. 신호 복원

### 6.1 이상적 복원 (sinc 보간법)

이상적 복원은 휘태커-섀넌 보간 공식을 사용합니다:

$$x_r(t) = \sum_{n=-\infty}^{\infty} x[n] \, \operatorname{sinc}\!\left(\frac{t - nT_s}{T_s}\right)$$

주파수 영역에서, 이것은 다음과 같은 이상적 저역통과 필터를 적용하는 것과 동등합니다:
- 통과 대역 이득: $T_s$
- 차단 주파수: $\Omega_s / 2$

**이상적 sinc 보간법의 특성:**
- 모든 샘플 지점을 정확히 통과: $x_r(nT_s) = x[n]$
- 표본화 정리가 만족되면 완벽한 복원
- 비인과적 (과거와 미래의 모든 샘플 필요)
- sinc 함수의 무한한 지지로 인해 비실용적

```python
def sinc_interpolation_demo():
    """Demonstrate ideal sinc interpolation for reconstruction."""
    # Original signal
    f1, f2 = 3, 7
    t_true = np.linspace(0, 1, 10000)
    x_true = np.sin(2 * np.pi * f1 * t_true) + 0.5 * np.cos(2 * np.pi * f2 * t_true)

    # Sample at fs = 20 Hz (Nyquist rate = 14 Hz, so well sampled)
    fs = 20
    Ts = 1.0 / fs
    n_samples = np.arange(0, 1, Ts)
    x_samples = np.sin(2 * np.pi * f1 * n_samples) + \
                0.5 * np.cos(2 * np.pi * f2 * n_samples)

    # Sinc interpolation
    t_recon = np.linspace(0, 1, 10000)
    x_recon = np.zeros_like(t_recon)

    for n, (tn, xn) in enumerate(zip(n_samples, x_samples)):
        x_recon += xn * np.sinc((t_recon - tn) / Ts)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Reconstruction
    axes[0].plot(t_true, x_true, 'b-', linewidth=1, alpha=0.5, label='Original')
    axes[0].plot(t_recon, x_recon, 'r-', linewidth=1.5, label='Sinc reconstructed')
    axes[0].stem(n_samples, x_samples, linefmt='g-', markerfmt='go',
                 basefmt='k-', label='Samples')
    axes[0].set_title(f'Sinc Interpolation (fs = {fs} Hz)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reconstruction error
    # Interpolate true signal at same points for comparison
    x_true_at_recon = np.sin(2 * np.pi * f1 * t_recon) + \
                      0.5 * np.cos(2 * np.pi * f2 * t_recon)
    error = x_true_at_recon - x_recon

    axes[1].plot(t_recon, error, 'k-', linewidth=0.5)
    axes[1].set_title(f'Reconstruction Error (max = {np.max(np.abs(error)):.6f})')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Error')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sinc_interpolation.png', dpi=150)
    plt.show()

sinc_interpolation_demo()
```

### 6.2 개별 sinc 기여

```python
def visualize_sinc_contributions():
    """Show how individual sinc functions combine for reconstruction."""
    fs = 5  # Low rate for visualization
    Ts = 1.0 / fs
    n_samples = np.arange(0, 1, Ts)
    x_samples = np.array([0.0, 0.8, 0.3, -0.5, -0.9])

    t = np.linspace(-0.2, 1.2, 1000)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Individual sinc contributions
    total = np.zeros_like(t)
    colors = plt.cm.tab10(np.linspace(0, 1, len(n_samples)))

    for i, (tn, xn) in enumerate(zip(n_samples, x_samples)):
        sinc_contrib = xn * np.sinc((t - tn) / Ts)
        total += sinc_contrib
        axes[0].plot(t, sinc_contrib, '--', color=colors[i], alpha=0.6,
                     label=f'x[{i}]={xn:.1f} * sinc')
        axes[0].plot(tn, xn, 'o', color=colors[i], markersize=8)

    axes[0].set_title('Individual Sinc Contributions')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Total reconstruction
    axes[1].plot(t, total, 'r-', linewidth=2, label='Reconstructed signal')
    axes[1].stem(n_samples, x_samples, linefmt='g-', markerfmt='go',
                 basefmt='k-', label='Samples')
    axes[1].set_title('Sum of All Sinc Functions')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sinc_contributions.png', dpi=150)
    plt.show()

visualize_sinc_contributions()
```

---

## 7. 영차 홀드와 일차 홀드

### 7.1 영차 홀드(ZOH, Zero-Order Hold)

가장 단순한 실용적 복원 방법: 다음 샘플까지 각 샘플 값을 일정하게 유지합니다.

$$x_{\text{ZOH}}(t) = x[n], \quad nT_s \leq t < (n+1)T_s$$

동치 표현으로, 직사각형 펄스와의 합성곱:

$$x_{\text{ZOH}}(t) = \sum_{n} x[n] \, p(t - nT_s)$$

여기서 $p(t) = \begin{cases} 1 & 0 \leq t < T_s \\ 0 & \text{기타} \end{cases}$

ZOH의 주파수 응답은:

$$H_{\text{ZOH}}(j\Omega) = T_s \, e^{-j\Omega T_s/2} \, \operatorname{sinc}\!\left(\frac{\Omega T_s}{2\pi}\right)$$

이로 인해 발생하는 것:
- **진폭 왜곡**: sinc 엔벨로프가 고주파를 감쇠
- **위상 왜곡**: 반 샘플 지연 ($T_s/2$)

### 7.2 일차 홀드(FOH, First-Order Hold)

연속적인 샘플 사이의 선형 보간:

$$x_{\text{FOH}}(t) = x[n] + \frac{x[n+1] - x[n]}{T_s}(t - nT_s), \quad nT_s \leq t < (n+1)T_s$$

동치 표현으로, 삼각형 펄스와의 합성곱:

$$\Lambda(t) = \begin{cases} 1 - |t|/T_s & |t| \leq T_s \\ 0 & \text{기타} \end{cases}$$

FOH는 ZOH보다 더 나은 고주파 근사를 제공하지만, 전체 샘플 하나의 지연을 도입합니다.

### 7.3 복원 방법 비교

```python
def compare_reconstruction_methods():
    """Compare ZOH, FOH, and sinc reconstruction."""
    # Original signal
    f_sig = 3  # Hz
    t_true = np.linspace(0, 1, 10000)
    x_true = np.sin(2 * np.pi * f_sig * t_true) + \
             0.3 * np.sin(2 * np.pi * 7 * t_true)

    # Sample at fs = 20 Hz
    fs = 20
    Ts = 1.0 / fs
    t_samples = np.arange(0, 1, Ts)
    x_samples = np.sin(2 * np.pi * f_sig * t_samples) + \
                0.3 * np.sin(2 * np.pi * 7 * t_samples)

    t_recon = np.linspace(0, 1, 10000)

    # 1. Zero-Order Hold
    x_zoh = np.zeros_like(t_recon)
    for i in range(len(t_samples)):
        if i < len(t_samples) - 1:
            mask = (t_recon >= t_samples[i]) & (t_recon < t_samples[i + 1])
        else:
            mask = t_recon >= t_samples[i]
        x_zoh[mask] = x_samples[i]

    # 2. First-Order Hold (linear interpolation)
    x_foh = np.interp(t_recon, t_samples, x_samples)

    # 3. Sinc interpolation
    x_sinc = np.zeros_like(t_recon)
    for tn, xn in zip(t_samples, x_samples):
        x_sinc += xn * np.sinc((t_recon - tn) / Ts)

    # Plot comparison
    methods = [
        ('Zero-Order Hold (ZOH)', x_zoh, 'orange'),
        ('First-Order Hold (FOH)', x_foh, 'purple'),
        ('Sinc Interpolation (Ideal)', x_sinc, 'red'),
    ]

    fig, axes = plt.subplots(len(methods), 1, figsize=(14, 12))

    for ax, (name, x_rec, color) in zip(axes, methods):
        ax.plot(t_true, x_true, 'b-', alpha=0.4, linewidth=1, label='Original')
        ax.plot(t_recon, x_rec, '-', color=color, linewidth=1.5, label=name)
        ax.stem(t_samples, x_samples, linefmt='g-', markerfmt='go',
                basefmt='k-', label='Samples')

        # Compute error
        x_true_interp = np.sin(2 * np.pi * f_sig * t_recon) + \
                        0.3 * np.sin(2 * np.pi * 7 * t_recon)
        error = np.sqrt(np.mean((x_true_interp - x_rec) ** 2))

        ax.set_title(f'{name}  |  RMSE = {error:.4f}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reconstruction_comparison.png', dpi=150)
    plt.show()

compare_reconstruction_methods()
```

---

## 8. 실용적인 ADC/DAC 고려 사항

### 8.1 아날로그-디지털 변환(ADC)

실용적인 ADC는 세 단계를 포함합니다:

1. **에일리어싱 방지 필터**: 표본화 이전의 아날로그 저역통과 필터
2. **샘플-앤-홀드**: 순간적인 아날로그 값을 포착
3. **양자화(Quantization)**: 연속 진폭을 이산 레벨로 매핑

주요 ADC 사양:

| 매개변수 | 설명 | 일반적인 값 |
|-----------|-------------|----------------|
| 분해능(Resolution) | 비트 수 | 8, 12, 16, 24 비트 |
| 표본화 속도 | 초당 샘플 수 | 44.1 kHz (오디오), 1 MSPS (일반) |
| SNR | 신호 대 잡음비 | $N$ 비트에서 $\approx 6.02N + 1.76$ dB |
| ENOB | 유효 비트 수 | 모든 잡음 원인 고려 |
| INL/DNL | 적분/차분 비선형성 | 이상적으로 < 1 LSB |

### 8.2 양자화

양자화는 $N$비트 ADC에 대해 연속 진폭을 $2^N$개의 이산 레벨 중 하나로 매핑합니다.

**균일 양자화기:**

$$x_q = \Delta \cdot \left\lfloor \frac{x}{\Delta} + 0.5 \right\rfloor$$

여기서 $\Delta = \frac{x_{\max} - x_{\min}}{2^N}$은 양자화 스텝 크기(LSB)입니다.

**양자화 오차(잡음):**

$$e_q = x_q - x, \quad -\frac{\Delta}{2} \leq e_q < \frac{\Delta}{2}$$

균일하게 분포된 양자화 잡음에 대한 **신호 대 양자화 잡음비(SQNR)**:

$$\text{SQNR} \approx 6.02 N + 1.76 \text{ dB}$$

```python
def demonstrate_quantization():
    """Demonstrate ADC quantization effects."""
    # Original signal
    t = np.linspace(0, 0.01, 10000)
    f_sig = 440  # Hz (A4 note)
    x = np.sin(2 * np.pi * f_sig * t)

    bit_depths = [2, 4, 8, 16]

    fig, axes = plt.subplots(len(bit_depths), 1, figsize=(14, 12))

    for ax, bits in zip(axes, bit_depths):
        n_levels = 2 ** bits
        delta = 2.0 / n_levels  # Range [-1, 1]

        # Quantize
        x_q = np.round(x / delta) * delta
        x_q = np.clip(x_q, -1.0, 1.0)

        # Quantization error
        error = x_q - x

        ax.plot(t * 1000, x, 'b-', alpha=0.4, linewidth=1, label='Original')
        ax.plot(t * 1000, x_q, 'r-', linewidth=1, label=f'{bits}-bit quantized')

        sqnr = 6.02 * bits + 1.76
        actual_sqnr = 10 * np.log10(np.mean(x**2) / np.mean(error**2)) \
            if np.mean(error**2) > 0 else float('inf')

        ax.set_title(f'{bits}-bit Quantization ({n_levels} levels) | '
                     f'SQNR: theoretical={sqnr:.1f} dB, actual={actual_sqnr:.1f} dB')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quantization.png', dpi=150)
    plt.show()

demonstrate_quantization()
```

### 8.3 디지털-아날로그 변환(DAC)

DAC는 디지털 샘플을 아날로그 신호로 변환합니다:

1. **D/A 변환**: 디지털 코드를 아날로그 전압/전류로 변환
2. **복원 필터**: 계단형 출력을 매끄럽게 하는 저역통과 필터

대부분의 DAC는 ZOH를 내재적으로 사용하여 계단형 출력을 생성하며, 스펙트럼 이미지(spectral image)를 제거하기 위해 DAC 후 복원 필터가 필요합니다.

### 8.4 ZOH 왜곡 보상

ZOH가 sinc 롤오프를 도입하므로, DAC 이전에 디지털적으로 **sinc 보상 필터**(역 sinc)를 적용할 수 있습니다:

$$H_{\text{comp}}(e^{j\omega}) = \frac{\omega T_s / 2}{\sin(\omega T_s / 2)}$$

이 전처리 등화는 ZOH 이후의 출력이 통과 대역에서 평탄하도록 디지털 신호를 미리 보정합니다.

---

## 9. 오버샘플링과 그 이점

### 9.1 오버샘플링이란 무엇인가?

오버샘플링(oversampling)은 나이퀴스트 레이트보다 훨씬 높은 속도로 표본화하는 것을 의미합니다:

$$f_s = M \cdot 2B$$

여기서 $M > 1$은 **오버샘플링 비율(oversampling ratio)**입니다.

### 9.2 오버샘플링의 이점

1. **에일리어싱 방지 필터 요건 완화**: 더 넓은 전이 대역으로 더 단순한 아날로그 필터 허용
2. **잡음 성형을 통한 SNR 향상**: 양자화 잡음이 더 넓은 대역폭으로 분산되고, 필터링으로 대역 외 잡음 제거 가능
3. **SNR 향상**: $M$배 오버샘플링은 $10 \log_{10}(M)$ dB 향상 제공 (잡음 성형 전)
4. **더 쉬운 복원**: 더 넓은 스펙트럼 간격으로 복원 필터 단순화

### 9.3 시그마-델타($\Sigma\Delta$) 오버샘플링 ADC

현대 고해상도 ADC(오디오, 계측기기)는 $\Sigma\Delta$ 변조를 사용합니다:

1. $M \times f_s$에서 오버샘플링 (예: $M = 64$ 또는 $128$)
2. 1비트 양자화기(비교기) 사용
3. 양자화 잡음을 고주파로 성형(잡음 성형)
4. 원하는 속도로 디지털 필터링 및 데시메이션(decimation)

유효 분해능: $N_{\text{eff}} \approx N + \frac{(2L+1)}{2} \log_2(M) - \frac{1}{2} \log_2\!\left(\frac{\pi^{2L}}{2L+1}\right)$

여기서 $L$은 잡음 성형 루프의 차수이고, $N$은 양자화기 분해능입니다.

```python
def demonstrate_oversampling():
    """Show the benefits of oversampling."""
    # Original signal: 100 Hz sinusoid
    f_sig = 100  # Hz
    B = 200      # Signal bandwidth

    # Nyquist rate
    f_nyquist = 2 * B  # 400 Hz

    oversampling_ratios = [1, 2, 4, 8, 16]

    print("Oversampling Benefits")
    print("=" * 60)
    print(f"Signal bandwidth B = {B} Hz, Nyquist rate = {f_nyquist} Hz")
    print(f"{'M':>4s} | {'fs (Hz)':>10s} | {'SNR gain (dB)':>15s} | "
          f"{'Transition band':>18s}")
    print("-" * 60)

    for M in oversampling_ratios:
        fs = M * f_nyquist
        snr_gain = 10 * np.log10(M) if M > 1 else 0
        transition = fs / 2 - B
        print(f"{M:4d} | {fs:10d} | {snr_gain:15.2f} | "
              f"{transition:14.0f} Hz")

    # Visualize filter requirements
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # M = 1: tight filter needed
    f = np.linspace(0, 800, 1000)
    fs1 = 400
    spec = np.where(f <= B, 1, 0)
    axes[0].fill_between(f, spec, alpha=0.3, color='blue', label='Signal')
    axes[0].axvline(fs1 / 2, color='red', linestyle='--',
                    label=f'Nyquist freq = {fs1/2} Hz')
    axes[0].axvline(fs1 - B, color='orange', linestyle='--',
                    label=f'First alias starts = {fs1-B} Hz')

    # Anti-aliasing region (narrow!)
    axes[0].axvspan(B, fs1 / 2, alpha=0.2, color='green',
                    label=f'Transition band = {fs1/2 - B} Hz')
    axes[0].set_title(f'M = 1 (fs = {fs1} Hz): Narrow transition band')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # M = 4: relaxed filter
    fs4 = 4 * 400
    axes[1].fill_between(f, spec, alpha=0.3, color='blue', label='Signal')
    axes[1].axvline(fs4 / 2, color='red', linestyle='--',
                    label=f'Nyquist freq = {fs4/2} Hz')

    f_ext = np.linspace(0, 2000, 2000)
    spec_ext = np.where(f_ext <= B, 1, 0)
    axes[1].fill_between(f_ext[:800], spec_ext[:800], alpha=0.3, color='blue')
    axes[1].axvspan(B, fs4 / 2, alpha=0.2, color='green',
                    label=f'Transition band = {fs4/2 - B} Hz')
    axes[1].set_title(f'M = 4 (fs = {fs4} Hz): Wide transition band')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_xlim(0, 800)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('oversampling.png', dpi=150)
    plt.show()

demonstrate_oversampling()
```

### 9.4 오버샘플링 후 데시메이션

오버샘플링과 디지털 필터링 후, 표본화 속도를 원하는 출력 속도로 줄일 수 있습니다(데시메이션):

1. 날카로운 디지털 저역통과 필터 적용 (정밀하게 구현 용이)
2. $M$배로 다운샘플링 (매 $M$번째 샘플만 보존)

이것이 현대 $\Sigma\Delta$ ADC 아키텍처의 기반입니다.

---

## 10. 다중 속도 신호 처리 미리 보기

### 10.1 데시메이션(Decimation, 다운샘플링)

표본화 속도를 $M$배 줄이기:

$$y[n] = x[nM]$$

에일리어싱을 방지하기 위해 먼저 에일리어싱 방지 필터를 적용해야 합니다.

### 10.2 보간(Interpolation, 업샘플링)

표본화 속도를 $L$배 늘리기:

$$w[n] = \begin{cases} x[n/L] & n = 0, \pm L, \pm 2L, \ldots \\ 0 & \text{기타} \end{cases}$$

이어서 저역통과(이미지 방지, anti-imaging) 필터를 적용합니다.

### 10.3 표본화 속도 변환

$L/M$의 유리수 속도 변환:

$$\text{} L\text{배 업샘플링} \;\rightarrow\; \text{필터} \;\rightarrow\; M\text{배 다운샘플링}$$

```python
def demonstrate_decimation_interpolation():
    """Show basic multirate operations: decimation and interpolation."""
    # Original signal
    fs = 1000
    t = np.arange(0, 0.1, 1 / fs)
    x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

    # Decimation by M=4
    M = 4
    # Anti-aliasing filter before decimation
    b_dec, a_dec = signal.butter(8, 1.0 / M, btype='low')
    x_filtered = signal.filtfilt(b_dec, a_dec, x)
    x_dec = x_filtered[::M]
    t_dec = t[::M]

    # Interpolation by L=4
    L = 4
    x_up = np.zeros(len(x_dec) * L)
    x_up[::L] = x_dec
    # Anti-imaging filter after upsampling
    b_int, a_int = signal.butter(8, 1.0 / L, btype='low')
    x_interp = L * signal.filtfilt(b_int, a_int, x_up)
    t_interp = np.arange(len(x_interp)) / fs

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    axes[0].plot(t * 1000, x, 'b-', linewidth=1)
    axes[0].set_title(f'Original Signal (fs = {fs} Hz, {len(x)} samples)')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    axes[1].stem(t_dec * 1000, x_dec, linefmt='r-', markerfmt='ro',
                 basefmt='k-')
    axes[1].set_title(f'Decimated by M={M} (fs = {fs//M} Hz, '
                      f'{len(x_dec)} samples)')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_interp * 1000, x_interp, 'g-', linewidth=1,
                 label='Interpolated')
    axes[2].plot(t[:len(x_interp)] * 1000, x[:len(x_interp)], 'b--',
                 alpha=0.5, linewidth=1, label='Original')
    axes[2].set_title(f'Interpolated back to fs = {fs} Hz')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('multirate.png', dpi=150)
    plt.show()

demonstrate_decimation_interpolation()
```

---

## 11. 요약

```
┌─────────────────────────────────────────────────────────────────┐
│                  Sampling & Reconstruction                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Sampling:                                                       │
│    x[n] = x(nTs)                                                │
│    Spectrum: Xs(jΩ) = (1/Ts) Σ X(j(Ω - kΩs))                  │
│                                                                  │
│  Nyquist-Shannon Theorem:                                        │
│    fs > 2B  →  perfect reconstruction possible                  │
│    fs ≤ 2B  →  aliasing (irreversible distortion)               │
│                                                                  │
│  Anti-Aliasing:                                                  │
│    Analog LPF before sampling to remove f > fs/2                │
│    Practical: guard band + filter roll-off                      │
│                                                                  │
│  Reconstruction Methods:                                         │
│    Ideal: sinc interpolation (impractical)                      │
│    ZOH:   staircase, sinc distortion (simple)                   │
│    FOH:   linear interpolation (better)                         │
│    Post-filter: smooth staircase output                         │
│                                                                  │
│  Oversampling (fs >> 2B):                                        │
│    - Relaxes anti-aliasing filter requirements                  │
│    - SNR gain: 10 log10(M) dB                                   │
│    - Enables Sigma-Delta ADC architecture                       │
│                                                                  │
│  Practical ADC/DAC:                                              │
│    ADC: anti-alias → sample-hold → quantize                     │
│    DAC: D/A → ZOH → reconstruction filter                      │
│    SQNR ≈ 6.02N + 1.76 dB                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. 연습 문제

### 연습 문제 1: 표본화 속도 결정

신호가 100 Hz, 250 Hz, 500 Hz의 성분을 포함하고 있습니다.

**(a)** 에일리어싱을 방지하기 위한 최소 표본화 속도는 얼마인가?

**(b)** 신호를 800 Hz로 표본화할 경우, 어떤 성분이 에일리어싱되고 어떤 주파수로 나타나는가?

**(c)** 실용적으로 권장하는 표본화 속도는 얼마이고 그 이유는?

### 연습 문제 2: 에일리어싱 분석

1 kHz 정현파 신호를 1.5 kHz로 표본화합니다.

**(a)** 에일리어스가 나타나는 주파수는?

**(b)** 원래 신호와 에일리어싱된 신호를 모두 보여주는 시각적 데모를 위한 Python 코드를 작성하세요.

**(c)** 원하는 대역폭이 600 Hz일 때, 이 문제를 방지하는 에일리어싱 방지 필터를 설계하세요 (종류, 차수, 차단 주파수 명시).

### 연습 문제 3: sinc 보간법 구현

완전한 sinc 보간 함수를 구현하세요:

```python
def sinc_reconstruct(samples, Ts, t_recon):
    """
    Reconstruct a continuous-time signal from samples using sinc interpolation.

    Parameters:
        samples: array of sample values x[n]
        Ts: sampling period
        t_recon: array of time points for reconstruction

    Returns:
        x_recon: reconstructed signal values
    """
    # Your implementation here
    pass
```

**(a)** 알려진 대역 제한 신호로 함수를 테스트하고 완벽한 복원을 검증하세요.

**(b)** 사용된 sinc 항의 수(절단)의 함수로 복원 오차를 측정하세요.

**(c)** 다양한 절단 길이에 대한 계산 시간 대 정확도를 비교하세요.

### 연습 문제 4: 양자화 잡음 분석

**(a)** $N$ 비트 균일 양자화기와 풀스케일 정현파 입력에 대한 SQNR 공식 $\text{SQNR} = 6.02N + 1.76$ dB를 유도하세요.

**(b)** $N = 4, 8, 12, 16$ 비트에 대해 이 공식을 실증적으로 검증하는 Python 코드를 작성하세요.

**(c)** 신호가 양자화기 범위의 절반만 사용할 경우 SQNR은 어떻게 변하는가?

### 연습 문제 5: ZOH 주파수 응답

**(a)** $f_s = 8000$ Hz에서 ZOH 복원의 크기 응답과 위상 응답을 그래프로 그리세요.

**(b)** sinc 롤오프로 인해 $f = 3000$ Hz에서의 감쇠를 계산하세요.

**(c)** 디지털 sinc 보상 필터를 설계하고 (31탭 FIR 계수 제공), 이것이 통과 대역 드룹(passband droop)을 교정함을 보이세요.

### 연습 문제 6: 오버샘플링 대 비트 깊이 트레이드오프

오디오 응용에서 16비트 유효 분해능(SQNR $\approx$ 98 dB)이 필요합니다.

**(a)** 잡음 성형 없이 12비트 ADC와 오버샘플링을 사용하면, 필요한 오버샘플링 비율 $M$은 얼마인가?

**(b)** 1차 잡음 성형($\Sigma\Delta$)으로 어떤 오버샘플링 비율로 충분한가?

**(c)** 1차 $\Sigma\Delta$ 변조기를 시뮬레이션하고, 출력 스펙트럼을 플롯하여 잡음 성형 동작을 검증하는 Python 코드를 작성하세요.

### 연습 문제 7: 완전한 ADC/DAC 파이프라인

Python으로 완전한 표본화-복원 파이프라인을 구축하세요:

1. 테스트 신호 생성 (정현파의 합)
2. 에일리어싱 방지 필터 적용
3. 지정된 속도로 표본화
4. $N$ 비트로 양자화
5. (a) ZOH, (b) 선형 보간, (c) sinc 보간을 사용하여 복원
6. 복원 저역통과 필터 적용
7. 세 방법 모두 비교: RMSE, SNR, 시각적 품질

---

## 13. 더 읽을 거리

- Oppenheim, Willsky, Nawab. *Signals and Systems*, 2판. 7장.
- Oppenheim, Schafer. *Discrete-Time Signal Processing*, 3판. 4-5장.
- Proakis, Manolakis. *Digital Signal Processing*, 4판. 6장.
- Lyons, R. G. *Understanding Digital Signal Processing*, 3판. 2-3장.
- Smith, S. W. *The Scientist and Engineer's Guide to Digital Signal Processing*, 3-4장. (무료 온라인: dspguide.com)

---

**이전**: [04. 푸리에 변환과 주파수 영역](04_Fourier_Transform.md) | **다음**: [06. 이산 푸리에 변환](06_Discrete_Fourier_Transform.md)
