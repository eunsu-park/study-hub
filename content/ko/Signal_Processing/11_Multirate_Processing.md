# 다중률 신호 처리(Multirate Signal Processing)

## 학습 목표

- 데시메이션(Decimation, 다운샘플링)과 보간(Interpolation, 업샘플링)의 기초 이해
- 샘플률 변경이 스펙트럼에 미치는 영향 분석
- 효율적인 다중률 구현을 위한 노블 항등식(Noble Identities) 습득
- 률 변환을 위한 앤티앨리어싱(anti-aliasing) 및 앤티이미징(anti-imaging) 필터 설계
- 계산 효율적인 필터링을 위한 폴리페이즈 분해(polyphase decomposition) 구현
- 연속된 보간과 데시메이션을 이용한 유리수 률 변환(rational rate conversion) 적용
- 직교 미러 필터(QMF, Quadrature Mirror Filters)를 포함한 필터 뱅크 구조 이해
- Python의 `scipy.signal` 모듈을 사용한 다중률 시스템 구현

---

## 목차

1. [다중률 시스템 소개](#1-다중률-시스템-소개)
2. [다운샘플링 (데시메이션)](#2-다운샘플링-데시메이션)
3. [업샘플링 (보간)](#3-업샘플링-보간)
4. [데시메이션 및 보간 필터](#4-데시메이션-및-보간-필터)
5. [노블 항등식](#5-노블-항등식)
6. [폴리페이즈 분해](#6-폴리페이즈-분해)
7. [유리수 률 변환](#7-유리수-률-변환)
8. [다단계 률 변환](#8-다단계-률-변환)
9. [필터 뱅크](#9-필터-뱅크)
10. [응용](#10-응용)
11. [Python 구현](#11-python-구현)
12. [연습 문제](#12-연습-문제)

---

## 1. 다중률 시스템 소개

### 1.1 왜 다중률 처리인가?

다중률 신호 처리(Multirate signal processing)는 둘 이상의 샘플률로 동작하는 시스템을 다룹니다. 주된 동기는 다음과 같습니다:

- **샘플률 변환**: 44.1 kHz(CD)와 48 kHz(전문 오디오) 간의 오디오 변환
- **계산 효율성**: 각 단계에 필요한 최저 률로 처리
- **대역폭 감소**: 협대역 신호를 데시메이션하여 데이터 률 감소
- **서브밴드 코딩(Subband coding)**: 효율적인 압축을 위해 신호를 주파수 대역으로 분할
- **시그마-델타(Sigma-delta) ADC/DAC**: 속도와 해상도를 교환하는 오버샘플링 변환기

```
┌──────────────────────────────────────────────────────────────────┐
│              Multirate System Building Blocks                    │
│                                                                  │
│  Downsampler (↓M):          Upsampler (↑L):                     │
│  ┌───────┐                  ┌───────┐                            │
│  │  ↓M   │  Keep every      │  ↑L   │  Insert (L-1)             │
│  │       │  M-th sample     │       │  zeros between             │
│  └───────┘                  └───────┘  samples                   │
│                                                                  │
│  Decimator:                 Interpolator:                        │
│  ┌────────┐  ┌───────┐     ┌───────┐  ┌────────┐               │
│  │Anti-   │→ │  ↓M   │     │  ↑L   │→ │Anti-   │               │
│  │alias   │  │       │     │       │  │image   │               │
│  │filter  │  └───────┘     └───────┘  │filter  │               │
│  └────────┘                           └────────┘               │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 기본 연산

다중률 처리에서 두 가지 기본 연산:

**인수 $M$만큼 다운샘플링** ($M$번째 샘플마다 유지):

$$y[n] = x[nM]$$

**인수 $L$만큼 업샘플링** (샘플 사이에 $L-1$개의 영(zero) 삽입):

$$y[n] = \begin{cases} x[n/L], & n = 0, \pm L, \pm 2L, \ldots \\ 0, & \text{otherwise} \end{cases}$$

---

## 2. 다운샘플링 (데시메이션)

### 2.1 시간 영역 연산

정수 인수 $M$만큼 다운샘플링하면 $M$번째 샘플마다 유지됩니다:

$$x_d[n] = x[nM]$$

출력 샘플률은 $f_s' = f_s / M$입니다.

### 2.2 주파수 영역 분석

다운샘플된 신호의 DTFT(이산 시간 푸리에 변환)는:

$$X_d(e^{j\omega}) = \frac{1}{M} \sum_{k=0}^{M-1} X\left(e^{j(\omega - 2\pi k)/M}\right)$$

이 공식은 두 가지 효과를 나타냅니다:
1. **주파수 스케일링**: 스펙트럼이 $M$배 늘어납니다 (주파수가 $[-\pi, \pi]$로 압축됨)
2. **앨리어싱(Aliasing)**: $M$개의 이동된 스펙트럼 복사본이 중첩됩니다

```
Downsampling by M=2: Spectral Effect

Original X(e^jω):
         ╱╲
        ╱  ╲
       ╱    ╲
──────╱──────╲──────
   -π    0    π

After ↓2, X_d(e^jω) = (1/2)[X(e^(jω/2)) + X(e^(j(ω-2π)/2))]:

              ╱╲
Stretched:   ╱  ╲
            ╱    ╲
 ╲         ╱      ╲         ╱  Aliased copies overlap!
  ╲       ╱        ╲       ╱
───╲─────╱──────────╲─────╱───
   -π    0           π

If the original signal is bandlimited to |ω| < π/M,
no aliasing occurs.
```

### 2.3 앨리어싱 시연

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def demonstrate_downsampling(M=4):
    """Demonstrate the spectral effects of downsampling."""
    fs = 8000
    t = np.arange(0, 0.1, 1/fs)
    N = len(t)

    # Create signal with multiple tones
    f1, f2, f3 = 200, 800, 1500  # Hz
    x = np.sin(2*np.pi*f1*t) + 0.5*np.sin(2*np.pi*f2*t) + 0.3*np.sin(2*np.pi*f3*t)

    # Downsample without anti-aliasing filter
    x_down_nofilter = x[::M]
    fs_down = fs / M

    # Downsample with anti-aliasing filter
    # Cutoff at fs/(2M) to prevent aliasing
    sos = signal.butter(8, fs / (2 * M), fs=fs, output='sos')
    x_filtered = signal.sosfilt(sos, x)
    x_down_filtered = x_filtered[::M]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Original signal
    freq_orig = np.fft.rfftfreq(N, 1/fs)
    X_orig = np.abs(np.fft.rfft(x)) / N

    axes[0, 0].plot(t[:200] * 1000, x[:200], 'b-')
    axes[0, 0].set_title(f'Original Signal (fs = {fs} Hz)')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(freq_orig, 20*np.log10(X_orig + 1e-15), 'b-')
    axes[0, 1].set_title('Original Spectrum')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude (dB)')
    axes[0, 1].set_xlim(0, fs/2)
    axes[0, 1].grid(True, alpha=0.3)

    # Downsampled without filter (aliased)
    N_down = len(x_down_nofilter)
    freq_down = np.fft.rfftfreq(N_down, 1/fs_down)
    X_down = np.abs(np.fft.rfft(x_down_nofilter)) / N_down

    t_down = np.arange(N_down) / fs_down
    axes[1, 0].plot(t_down[:50] * 1000, x_down_nofilter[:50], 'r-o', markersize=3)
    axes[1, 0].set_title(f'Downsampled ↓{M} (NO filter) fs = {fs_down:.0f} Hz')
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(freq_down, 20*np.log10(X_down + 1e-15), 'r-')
    axes[1, 1].set_title('Spectrum (Aliased!)')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude (dB)')
    axes[1, 1].set_xlim(0, fs_down/2)
    axes[1, 1].axvline(fs_down/2, color='gray', linestyle='--', alpha=0.5,
                         label=f'Nyquist = {fs_down/2:.0f} Hz')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Downsampled with anti-aliasing filter
    N_down_f = len(x_down_filtered)
    freq_down_f = np.fft.rfftfreq(N_down_f, 1/fs_down)
    X_down_f = np.abs(np.fft.rfft(x_down_filtered)) / N_down_f

    t_down_f = np.arange(N_down_f) / fs_down
    axes[2, 0].plot(t_down_f[:50] * 1000, x_down_filtered[:50], 'g-o', markersize=3)
    axes[2, 0].set_title(f'Decimated ↓{M} (WITH anti-alias filter)')
    axes[2, 0].set_xlabel('Time (ms)')
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(freq_down_f, 20*np.log10(X_down_f + 1e-15), 'g-')
    axes[2, 1].set_title('Spectrum (No aliasing)')
    axes[2, 1].set_xlabel('Frequency (Hz)')
    axes[2, 1].set_ylabel('Magnitude (dB)')
    axes[2, 1].set_xlim(0, fs_down/2)
    axes[2, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Downsampling by M = {M}', fontsize=14)
    plt.tight_layout()
    plt.savefig('downsampling_demo.png', dpi=150)
    plt.close()

demonstrate_downsampling(M=4)
```

---

## 3. 업샘플링 (보간)

### 3.1 시간 영역 연산

인수 $L$만큼 업샘플링하면 연속된 샘플 사이에 $L-1$개의 영을 삽입합니다:

$$x_u[n] = \begin{cases} x[n/L], & n = 0, \pm L, \pm 2L, \ldots \\ 0, & \text{otherwise} \end{cases}$$

출력 샘플률은 $f_s' = L \cdot f_s$입니다.

### 3.2 주파수 영역 분석

업샘플된 신호의 DTFT는:

$$X_u(e^{j\omega}) = X(e^{j\omega L})$$

이는 스펙트럼을 $L$배 압축하여 $[0, 2\pi]$ 범위에 $L-1$개의 스펙트럼 이미지(이미징, imaging) 복사본을 생성합니다.

```
Upsampling by L=3: Spectral Effect

Original X(e^jω):
         ╱╲
        ╱  ╲
       ╱    ╲
──────╱──────╲──────
   -π    0    π

After ↑3, X_u(e^jω) = X(e^(j3ω)):

   ╱╲    ╱╲    ╱╲       Images (unwanted copies)
  ╱  ╲  ╱  ╲  ╱  ╲
 ╱    ╲╱    ╲╱    ╲
╱──────────────────╲
-π    -π/3   0   π/3    π

Apply lowpass filter at ω = π/3 to remove images!
```

### 3.3 이미징 시연

```python
def demonstrate_upsampling(L=4):
    """Demonstrate the spectral effects of upsampling."""
    fs = 2000
    t = np.arange(0, 0.1, 1/fs)
    N = len(t)

    # Create a bandlimited signal
    f1 = 200  # Hz
    x = np.sin(2*np.pi*f1*t)

    # Upsample: insert zeros
    x_up = np.zeros(len(x) * L)
    x_up[::L] = x
    fs_up = fs * L

    # Apply anti-imaging (interpolation) filter
    # Lowpass at fs/2 with gain L
    h_interp = signal.firwin(63, fs/2, fs=fs_up) * L
    x_interp = np.convolve(x_up, h_interp, mode='same')

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Original
    freq_orig = np.fft.rfftfreq(N, 1/fs)
    X_orig = np.abs(np.fft.rfft(x)) / N

    axes[0, 0].stem(t[:30] * 1000, x[:30], linefmt='b-', markerfmt='bo',
                     basefmt='k-')
    axes[0, 0].set_title(f'Original (fs = {fs} Hz)')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(freq_orig, X_orig, 'b-', linewidth=1.5)
    axes[0, 1].set_title('Original Spectrum')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_xlim(0, fs/2)
    axes[0, 1].grid(True, alpha=0.3)

    # Upsampled (with zero-insertion, before filtering)
    N_up = len(x_up)
    freq_up = np.fft.rfftfreq(N_up, 1/fs_up)
    X_up = np.abs(np.fft.rfft(x_up)) / N_up

    t_up = np.arange(N_up) / fs_up
    axes[1, 0].stem(t_up[:120] * 1000, x_up[:120], linefmt='r-',
                     markerfmt='ro', basefmt='k-')
    axes[1, 0].set_title(f'Upsampled ↑{L} (zero-inserted)')
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(freq_up, X_up, 'r-', linewidth=1.5)
    axes[1, 1].set_title('Spectrum (Imaging artifacts visible)')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    for k in range(1, L):
        axes[1, 1].axvline(k * fs, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlim(0, fs_up/2)
    axes[1, 1].grid(True, alpha=0.3)

    # Interpolated (after anti-imaging filter)
    X_interp = np.abs(np.fft.rfft(x_interp)) / N_up

    axes[2, 0].plot(t_up[:120] * 1000, x_interp[:120], 'g-', linewidth=1.5)
    axes[2, 0].set_title(f'Interpolated (after LP filter, gain = {L})')
    axes[2, 0].set_xlabel('Time (ms)')
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(freq_up, X_interp, 'g-', linewidth=1.5)
    axes[2, 1].set_title('Spectrum (Images removed)')
    axes[2, 1].set_xlabel('Frequency (Hz)')
    axes[2, 1].set_xlim(0, fs_up/2)
    axes[2, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Upsampling by L = {L}', fontsize=14)
    plt.tight_layout()
    plt.savefig('upsampling_demo.png', dpi=150)
    plt.close()

demonstrate_upsampling(L=4)
```

---

## 4. 데시메이션 및 보간 필터

### 4.1 데시메이션 필터 요구사항

$M$만큼 다운샘플링하기 전에, **앤티앨리어싱 필터(anti-aliasing filter)**가 앨리어싱을 방지하기 위해 신호 대역폭을 제한해야 합니다:

$$H_\text{dec}(e^{j\omega}) = \begin{cases} 1, & |\omega| < \pi/M \\ 0, & \pi/M \leq |\omega| \leq \pi \end{cases}$$

완전한 데시메이터:

$$y[n] = \left[\sum_{k} h[k] \cdot x[nM - k]\right]$$

참고: 먼저 필터링한 후 다운샘플링합니다. 필터는 **높은** 샘플률에서 동작합니다.

### 4.2 보간 필터 요구사항

$L$만큼 업샘플링한 후, **앤티이미징 필터(anti-imaging filter)**가 스펙트럼 이미지를 제거합니다:

$$H_\text{interp}(e^{j\omega}) = \begin{cases} L, & |\omega| < \pi/L \\ 0, & \pi/L \leq |\omega| \leq \pi \end{cases}$$

이득(gain) $L$은 영 삽입으로 인한 진폭 감소를 보상합니다.

### 4.3 데시메이션용 필터 설계

```python
def design_decimation_filter(M, filter_order=None, transition_width=0.1):
    """
    Design anti-aliasing filter for decimation by factor M.

    Parameters:
        M: decimation factor
        filter_order: FIR filter order (auto-computed if None)
        transition_width: normalized transition width (relative to pi/M)

    Returns:
        h: FIR filter coefficients
    """
    # Cutoff frequency: pi/M (with some transition)
    cutoff = (1 - transition_width) / M

    if filter_order is None:
        # Estimate order for 60 dB stopband attenuation
        filter_order = int(np.ceil(60 / (22 * transition_width / M * np.pi))) * 2 + 1

    h = signal.firwin(filter_order, cutoff)
    return h

# Design filters for different decimation factors
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, M in zip(axes.flat, [2, 4, 8, 16]):
    h = design_decimation_filter(M, filter_order=129)
    w, H = signal.freqz(h, worN=4096)
    H_dB = 20 * np.log10(np.abs(H) + 1e-15)

    ax.plot(w / np.pi, H_dB, 'b-', linewidth=1.5)
    ax.axvline(1/M, color='r', linestyle='--', alpha=0.7,
               label=f'π/{M} (cutoff)')
    ax.set_xlabel('Normalized Frequency (×π)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f'Anti-aliasing Filter for ↓{M}')
    ax.set_ylim(-80, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Decimation Filter Design', fontsize=13)
plt.tight_layout()
plt.savefig('decimation_filters.png', dpi=150)
plt.close()
```

---

## 5. 노블 항등식

### 5.1 항등식

노블 항등식(Noble Identities)은 필터를 률 변경 연산 전후로 이동할 수 있게 하여 효율적인 구현에 핵심적입니다:

**항등식 1** (다운샘플러):

$$\boxed{H(z^M) \rightarrow \downarrow M \equiv \downarrow M \rightarrow H(z)}$$

$M$만큼 다운샘플링 뒤의 필터 $H(z^M)$은 먼저 다운샘플링한 후 $H(z)$로 필터링하는 것과 동일합니다.

**항등식 2** (업샘플러):

$$\boxed{\uparrow L \rightarrow H(z^L) \equiv H(z) \rightarrow \uparrow L}$$

$L$만큼 업샘플링 뒤의 필터 $H(z^L)$은 먼저 $H(z)$로 필터링한 후 업샘플링하는 것과 동일합니다.

```
Noble Identity 1 (Downsampler):

  x[n] ──▶ H(z^M) ──▶ ↓M ──▶ y[n]
                ≡
  x[n] ──▶ ↓M ──▶ H(z) ──▶ y[n]

Noble Identity 2 (Upsampler):

  x[n] ──▶ ↑L ──▶ H(z^L) ──▶ y[n]
                ≡
  x[n] ──▶ H(z) ──▶ ↑L ──▶ y[n]
```

### 5.2 노블 항등식이 중요한 이유

- **계산 절감**: 필터를 낮은 률 쪽으로 이동하면 곱셈-누산(multiply-accumulate) 연산 수가 $M$ (또는 $L$)배 감소
- **폴리페이즈의 기초**: 항등식이 폴리페이즈 분해를 가능하게 함
- **주의**: 항등식은 $H(z^M)$ 또는 $H(z^L)$에만 적용되며, 임의의 $H(z)$에는 적용되지 않음

```python
def verify_noble_identity(M=3):
    """Verify Noble Identity 1 by comparing both implementations."""
    # Design a filter H(z)
    h = signal.firwin(31, 0.3)  # Original filter

    # Create H(z^M) by inserting M-1 zeros between coefficients
    h_stretched = np.zeros(len(h) * M - (M - 1))
    h_stretched[::M] = h

    # Input signal
    np.random.seed(42)
    x = np.random.randn(500)

    # Implementation 1: H(z^M) then ↓M
    y1 = np.convolve(x, h_stretched, mode='full')
    y1_down = y1[::M]

    # Implementation 2: ↓M then H(z)
    x_down = x[::M]
    y2 = np.convolve(x_down, h, mode='full')

    # Compare (trim to same length)
    min_len = min(len(y1_down), len(y2))
    error = np.max(np.abs(y1_down[:min_len] - y2[:min_len]))
    print(f"Noble Identity verification (M={M}):")
    print(f"  Max error: {error:.2e}")
    print(f"  Operations saved: {M}x fewer multiplies in low-rate version")

    return y1_down[:min_len], y2[:min_len]

y1, y2 = verify_noble_identity(M=3)
```

---

## 6. 폴리페이즈 분해

### 6.1 개념

**폴리페이즈 분해(polyphase decomposition)**는 필터를 $M$ (또는 $L$)개의 서브필터로 분할하며, 각 서브필터는 더 낮은 샘플률에서 동작합니다. 이는 다중률 필터링의 가장 효율적인 구현 방법입니다.

필터 $H(z) = \sum_{n=0}^{N-1} h[n] z^{-n}$에 대해, Type I 폴리페이즈 분해는:

$$H(z) = \sum_{k=0}^{M-1} z^{-k} E_k(z^M)$$

여기서 각 폴리페이즈 성분은:

$$E_k(z) = \sum_{n=0}^{\lfloor (N-1)/M \rfloor} h[nM + k] z^{-n}, \quad k = 0, 1, \ldots, M-1$$

### 6.2 폴리페이즈 데시메이션

```
Standard Decimation:                  Polyphase Decimation:
                                      (M times more efficient!)

x[n] ─▶ H(z) ─▶ ↓M ─▶ y[n]         x[n] ─┬─▶ ↓M ─▶ E₀(z) ─┬
                                            │                    │
  Computes M·N                              ├─▶ z⁻¹ ─▶ ↓M ─▶ E₁(z) ─┤ ─▶ Σ ─▶ y[n]
  multiplies per                            │                    │
  output sample                             ├─▶ z⁻² ─▶ ↓M ─▶ E₂(z) ─┤
                                            │                    │
                                            └─▶ z⁻(M-1)─▶↓M─▶E_{M-1} ┘

                                      Computes N multiplies per
                                      output sample (M× savings!)
```

### 6.3 폴리페이즈 보간

```
Standard Interpolation:               Polyphase Interpolation:

x[n] ─▶ ↑L ─▶ H(z) ─▶ y[n]          x[n] ─┬─▶ R₀(z) ─▶ ↑L ─────┬
                                             │                       │
  Computes L·N                               ├─▶ R₁(z) ─▶ ↑L ─▶ z⁻¹┤─▶ Σ ─▶ y[n]
  multiplies per                             │                       │
  input sample                               ├─▶ R₂(z) ─▶ ↑L ─▶ z⁻²┤
                                             │                       │
                                             └─▶ R_{L-1} ─▶↑L─▶z⁻(L-1)┘
```

### 6.4 구현

```python
class PolyphaseDecimator:
    """Efficient decimation using polyphase decomposition."""

    def __init__(self, h, M):
        """
        Parameters:
            h: prototype lowpass filter coefficients
            M: decimation factor
        """
        self.M = M
        self.N = len(h)

        # Pad h to a multiple of M
        pad_len = (M - self.N % M) % M
        h_padded = np.concatenate([h, np.zeros(pad_len)])

        # Polyphase decomposition: E_k[n] = h[nM + k]
        self.polyphase_filters = []
        for k in range(M):
            E_k = h_padded[k::M]
            self.polyphase_filters.append(E_k)

        self.subfilter_len = len(self.polyphase_filters[0])

    def process(self, x):
        """Decimate input signal x."""
        N_out = len(x) // self.M

        # Create polyphase input components
        # Trim to multiple of M
        x_trimmed = x[:N_out * self.M]

        y = np.zeros(N_out + self.subfilter_len - 1)

        for k in range(self.M):
            # k-th polyphase input: x[nM + k] (already at low rate)
            x_k = x_trimmed[k::self.M]
            # Filter with k-th polyphase component
            y_k = np.convolve(x_k, self.polyphase_filters[k])
            y[:len(y_k)] += y_k

        return y[:N_out]

    def get_info(self):
        """Print polyphase decomposition info."""
        print(f"Decimation factor: M = {self.M}")
        print(f"Original filter length: {self.N}")
        print(f"Number of polyphase filters: {self.M}")
        print(f"Subfilter length: {self.subfilter_len}")
        print(f"Computational savings: {self.M}x")


class PolyphaseInterpolator:
    """Efficient interpolation using polyphase decomposition."""

    def __init__(self, h, L):
        """
        Parameters:
            h: prototype lowpass filter coefficients (with gain L)
            L: interpolation factor
        """
        self.L = L
        self.N = len(h)

        # Pad h to a multiple of L
        pad_len = (L - self.N % L) % L
        h_padded = np.concatenate([h, np.zeros(pad_len)])

        # Polyphase decomposition for interpolation
        self.polyphase_filters = []
        for k in range(L):
            R_k = h_padded[k::L]
            self.polyphase_filters.append(R_k)

        self.subfilter_len = len(self.polyphase_filters[0])

    def process(self, x):
        """Interpolate input signal x."""
        N_in = len(x)
        N_out = N_in * self.L

        y = np.zeros(N_out + (self.subfilter_len - 1) * self.L)

        for k in range(self.L):
            # Filter with k-th polyphase component
            y_k = np.convolve(x, self.polyphase_filters[k])
            # Place at correct positions in output
            y[k::self.L][:len(y_k)] = y_k

        return y[:N_out]


# Verify polyphase implementation
np.random.seed(42)
M = 4
fs = 8000

# Design anti-aliasing filter
h = signal.firwin(128, 1.0 / M)

# Input signal
x = np.random.randn(2000)

# Standard decimation
x_filtered = np.convolve(x, h, mode='same')
y_standard = x_filtered[::M]

# Polyphase decimation
decimator = PolyphaseDecimator(h, M)
decimator.get_info()
y_polyphase = decimator.process(x)

# Compare
min_len = min(len(y_standard), len(y_polyphase))
error = np.max(np.abs(y_standard[:min_len] - y_polyphase[:min_len]))
print(f"\nMax error between standard and polyphase: {error:.2e}")

# Plot comparison
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(y_standard[:200], 'b-', label='Standard', alpha=0.8)
axes[0].plot(y_polyphase[:200], 'r--', label='Polyphase', alpha=0.8)
axes[0].set_title('Decimation Output Comparison')
axes[0].set_xlabel('Sample')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].semilogy(np.abs(y_standard[:min_len] - y_polyphase[:min_len]), 'k-')
axes[1].set_title(f'Absolute Difference (max: {error:.2e})')
axes[1].set_xlabel('Sample')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('polyphase_verification.png', dpi=150)
plt.close()
```

---

## 7. 유리수 률 변환

### 7.1 L/M에 의한 률 변경

샘플률을 유리수 인수 $L/M$으로 변환하려면:

1. **업샘플링**: $L$만큼 ($L-1$개의 영 삽입)
2. **필터링**: 차단 주파수 $\omega_c = \min(\pi/L, \pi/M)$의 저역통과 필터 적용
3. **다운샘플링**: $M$만큼 ($M$번째 샘플마다 유지)

연산 순서가 중요합니다: **먼저 보간한 후 데시메이션**합니다.

```
Rational Rate Conversion (L/M):

x[n] ──▶ ↑L ──▶ H(z) ──▶ ↓M ──▶ y[n]
  fs              cutoff = min(π/L, π/M)       fs' = fs × L/M
```

### 7.2 예제: 44.1 kHz에서 48 kHz로 변환

CD 오디오(44.1 kHz)를 전문 오디오(48 kHz)로 변환:

$$\frac{48000}{44100} = \frac{480}{441} = \frac{160}{147}$$

따라서 $L = 160$, $M = 147$입니다.

### 7.3 구현

```python
def rational_rate_convert(x, L, M, filter_order=None):
    """
    Convert sampling rate by rational factor L/M.

    Parameters:
        x: input signal
        L: interpolation factor
        M: decimation factor
        filter_order: FIR filter order (auto if None)

    Returns:
        y: resampled signal
    """
    if filter_order is None:
        filter_order = 2 * max(L, M) * 10 + 1

    # Design lowpass filter at min(pi/L, pi/M)
    cutoff = min(1.0 / L, 1.0 / M)
    h = signal.firwin(filter_order, cutoff) * L  # Gain of L for interpolation

    # Step 1: Upsample by L
    x_up = np.zeros(len(x) * L)
    x_up[::L] = x

    # Step 2: Filter
    x_filtered = np.convolve(x_up, h, mode='same')

    # Step 3: Downsample by M
    y = x_filtered[::M]

    return y

# Example: 44100 to 48000 Hz (simplified to L=160, M=147)
# For demonstration, use smaller factors: 48/44.1 ≈ 320/294 = 160/147
# Simplify further for demo: approximate as 8/7 ≈ 1.143
L, M = 160, 147

# Create test signal at 44100 Hz
fs_in = 44100
fs_out = fs_in * L / M
t_in = np.arange(0, 0.01, 1/fs_in)

# Test with 1 kHz sine
f_test = 1000
x = np.sin(2 * np.pi * f_test * t_in)

# Resample
y = rational_rate_convert(x, L, M, filter_order=2001)

# Also use scipy.signal.resample for comparison
y_scipy = signal.resample(x, int(len(x) * L / M))

print(f"Input: {len(x)} samples at {fs_in} Hz")
print(f"Output: {len(y)} samples at {fs_out:.1f} Hz")
print(f"Rate conversion factor: {L}/{M} = {L/M:.6f}")
print(f"Expected: {fs_out:.1f} Hz")

# Verify frequency preservation
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Time domain
t_out = np.arange(len(y)) / fs_out
axes[0].plot(t_in[:200] * 1000, x[:200], 'b-o', markersize=3,
              label=f'Input ({fs_in} Hz)', alpha=0.7)
axes[0].plot(t_out[:int(200*L/M)] * 1000, y[:int(200*L/M)], 'r-o',
              markersize=3, label=f'Output ({fs_out:.0f} Hz)', alpha=0.7)
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Sample Rate Conversion: Time Domain')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Frequency domain
N_in = len(x)
N_out = len(y)
freq_in = np.fft.rfftfreq(N_in, 1/fs_in)
freq_out = np.fft.rfftfreq(N_out, 1/fs_out)
X = np.abs(np.fft.rfft(x)) / N_in
Y = np.abs(np.fft.rfft(y)) / N_out

axes[1].plot(freq_in, 20*np.log10(X + 1e-15), 'b-', label='Input')
axes[1].plot(freq_out, 20*np.log10(Y + 1e-15), 'r-', label='Output')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Magnitude (dB)')
axes[1].set_title('Frequency Domain')
axes[1].set_xlim(0, 5000)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rational_conversion.png', dpi=150)
plt.close()
```

---

## 8. 다단계 률 변환

### 8.1 동기

큰 률 변경 인수(예: $M = 100$)의 경우, 단일 단계 데시메이터는 매우 날카로운(고차) 앤티앨리어싱 필터를 필요로 합니다. **다단계 데시메이션(Multistage decimation)**은 여러 개의 작은 데시메이션 단계를 연속으로 연결하여 총 계산 비용을 줄입니다.

$M = M_1 \cdot M_2 \cdot \ldots \cdot M_K$이면, 각 단계에서 더 단순한 필터를 사용할 수 있습니다.

```
Single-stage: x ──▶ H(z) ──▶ ↓100 ──▶ y
              Very sharp filter, high order

Multistage:   x ──▶ H₁(z) ──▶ ↓10 ──▶ H₂(z) ──▶ ↓10 ──▶ y
              Two moderate filters, lower total order
```

### 8.2 최적 단계 할당

$M$의 최적 인수분해는 사양에 따라 달라집니다. 일반적인 휴리스틱은 인수를 가능한 한 균등하게 분배하는 것입니다.

$M = M_1 \cdot M_2$의 경우:
- 단계 1 필터: 샘플률 $f_s$에서 $\pi / M$에 차단 주파수
- 단계 2 필터: 샘플률 $f_s / M_1$에서 $\pi / M_2$에 차단 주파수

### 8.3 구현 및 비교

```python
def multistage_decimation(x, factors, fs):
    """
    Multistage decimation.

    Parameters:
        x: input signal
        factors: list of decimation factors [M1, M2, ...]
        fs: input sampling frequency

    Returns:
        y: decimated signal
        total_ops: estimated total multiply operations
    """
    y = x.copy()
    current_fs = fs
    total_filter_taps = 0

    for i, M in enumerate(factors):
        # Design filter for this stage
        overall_M_remaining = np.prod(factors[i:])
        cutoff = 1.0 / overall_M_remaining

        # Estimate filter order (60 dB attenuation)
        transition = cutoff * 0.2
        numtaps = int(np.ceil(60 / (22 * transition))) | 1  # Ensure odd

        h = signal.firwin(numtaps, cutoff)
        total_filter_taps += numtaps

        # Filter and decimate
        y_filtered = np.convolve(y, h, mode='same')
        y = y_filtered[::M]
        current_fs /= M

        print(f"  Stage {i+1}: ↓{M}, filter order={numtaps-1}, "
              f"rate={current_fs:.0f} Hz, output samples={len(y)}")

    return y, total_filter_taps

# Compare single-stage vs multistage for M=16
fs = 16000
M_total = 16
t = np.arange(0, 0.5, 1/fs)
x = np.sin(2*np.pi*100*t) + 0.5*np.sin(2*np.pi*200*t)

print("Single-stage decimation (M=16):")
h_single = signal.firwin(513, 1.0/M_total)
x_single = np.convolve(x, h_single, mode='same')[::M_total]
print(f"  Filter order: {len(h_single)-1}, output samples: {len(x_single)}")

print("\nTwo-stage decimation (4×4):")
y_2stage, taps_2 = multistage_decimation(x, [4, 4], fs)

print("\nThree-stage decimation (2×2×4):")
y_3stage, taps_3 = multistage_decimation(x, [2, 2, 4], fs)

print("\nFour-stage decimation (2×2×2×2):")
y_4stage, taps_4 = multistage_decimation(x, [2, 2, 2, 2], fs)

# Compare spectra
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fs_out = fs / M_total

configs = [
    ('Single stage (↓16)', x_single),
    ('Two stages (↓4, ↓4)', y_2stage),
    ('Three stages (↓2, ↓2, ↓4)', y_3stage),
    ('Four stages (↓2, ↓2, ↓2, ↓2)', y_4stage),
]

for ax, (title, y) in zip(axes.flat, configs):
    N_y = len(y)
    freq = np.fft.rfftfreq(N_y, 1/fs_out)
    Y = np.abs(np.fft.rfft(y)) / N_y

    ax.plot(freq, 20*np.log10(Y + 1e-15), 'b-', linewidth=1.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(title)
    ax.set_xlim(0, fs_out/2)
    ax.set_ylim(-100, 0)
    ax.grid(True, alpha=0.3)

plt.suptitle(f'Multistage Decimation Comparison (M={M_total})', fontsize=13)
plt.tight_layout()
plt.savefig('multistage_decimation.png', dpi=150)
plt.close()
```

---

## 9. 필터 뱅크

### 9.1 분석-합성 필터 뱅크

**필터 뱅크(filter bank)**는 신호를 여러 주파수 서브밴드로 분할하고 재구성할 수 있습니다. 이는 오디오 코딩(MP3, AAC), 이미지 압축(JPEG2000), 그리고 다른 많은 응용의 기반입니다.

```
Analysis (Decomposition):              Synthesis (Reconstruction):

        ┌─ H₀(z) ─▶ ↓M ─▶ v₀[n] ──▶ ↑M ─▶ G₀(z) ─┐
        │                                              │
x[n] ──┤─ H₁(z) ─▶ ↓M ─▶ v₁[n] ──▶ ↑M ─▶ G₁(z) ──┼──▶ Σ ──▶ x̂[n]
        │                                              │
        └─ H_{M-1} ─▶ ↓M ─▶ v_{M-1} ─▶ ↑M ─▶ G_{M-1}┘

Analysis filters: H₀, H₁, ..., H_{M-1}
Synthesis filters: G₀, G₁, ..., G_{M-1}
```

**완전 재구성(Perfect Reconstruction, PR)**을 위한 조건:

$$\hat{X}(z) = X(z) \quad \text{(with some delay)}$$

### 9.2 2채널 필터 뱅크

가장 단순한 경우: 저역통과($H_0$)와 고역통과($H_1$) 필터를 사용하는 $M = 2$.

**완전 재구성 조건 (2채널):**

1. **앨리어싱 없음**: $H_0(-z)G_0(z) + H_1(-z)G_1(z) = 0$
2. **왜곡 없음**: $H_0(z)G_0(z) + H_1(z)G_1(z) = 2z^{-d}$ (순수 지연)

### 9.3 직교 미러 필터 (QMF)

QMF 뱅크에서 고역통과 필터는 저역통과 필터를 변조하여 얻습니다:

$$H_1(z) = H_0(-z) \quad \Rightarrow \quad h_1[n] = (-1)^n h_0[n]$$

이는 앨리어싱 소거 조건을 자동으로 만족합니다. 합성 필터의 경우:

$$G_0(z) = H_0(z), \quad G_1(z) = -H_1(z) = -H_0(-z)$$

```python
def qmf_filter_bank(h0, x):
    """
    Two-channel QMF analysis-synthesis filter bank.

    Parameters:
        h0: lowpass analysis filter
        x: input signal

    Returns:
        x_hat: reconstructed signal
        v0, v1: subband signals
    """
    M = len(h0)

    # Analysis filters
    # h0: lowpass (given)
    # h1: highpass (modulated version)
    h1 = h0 * ((-1) ** np.arange(len(h0)))

    # Synthesis filters (for QMF)
    g0 = h0.copy()
    g1 = -h1.copy()

    # Analysis: filter then downsample
    x_low = np.convolve(x, h0, mode='full')
    x_high = np.convolve(x, h1, mode='full')

    # Downsample by 2
    v0 = x_low[::2]
    v1 = x_high[::2]

    # Synthesis: upsample then filter
    u0 = np.zeros(len(v0) * 2)
    u0[::2] = v0
    u1 = np.zeros(len(v1) * 2)
    u1[::2] = v1

    y0 = np.convolve(u0, g0, mode='full')
    y1 = np.convolve(u1, g1, mode='full')

    # Sum
    min_len = min(len(y0), len(y1))
    x_hat = y0[:min_len] + y1[:min_len]

    return x_hat, v0, v1

# Design QMF lowpass filter
h0 = signal.firwin(32, 0.5, window='hamming')

# Test with a signal
fs = 8000
t = np.arange(0, 0.1, 1/fs)
x = np.sin(2*np.pi*200*t) + 0.5*np.sin(2*np.pi*3000*t)

x_hat, v0, v1 = qmf_filter_bank(h0, x)

# Compare
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Original
axes[0, 0].plot(x[:300], 'b-')
axes[0, 0].set_title('Original Signal')
axes[0, 0].grid(True, alpha=0.3)

# Subbands
axes[1, 0].plot(v0[:150], 'g-')
axes[1, 0].set_title('Lowpass Subband (v0)')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(v1[:150], 'r-')
axes[1, 1].set_title('Highpass Subband (v1)')
axes[1, 1].grid(True, alpha=0.3)

# Reconstructed
delay = len(h0) - 1  # Account for filter delay
x_hat_aligned = x_hat[delay:delay + len(x)]
axes[2, 0].plot(x[:300], 'b-', alpha=0.5, label='Original')
axes[2, 0].plot(x_hat_aligned[:300], 'r--', alpha=0.8, label='Reconstructed')
axes[2, 0].set_title('Reconstruction Comparison')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# Reconstruction error
recon_error = x[:min(len(x), len(x_hat_aligned))] - x_hat_aligned[:min(len(x), len(x_hat_aligned))]
axes[2, 1].plot(recon_error[:300], 'k-')
axes[2, 1].set_title(f'Reconstruction Error (max: {np.max(np.abs(recon_error)):.4f})')
axes[2, 1].grid(True, alpha=0.3)

# Spectra
N = len(x)
freq = np.fft.rfftfreq(N, 1/fs)
X = np.abs(np.fft.rfft(x)) / N
axes[0, 1].plot(freq, 20*np.log10(X + 1e-15), 'b-')
axes[0, 1].set_title('Original Spectrum')
axes[0, 1].set_xlabel('Frequency (Hz)')
axes[0, 1].grid(True, alpha=0.3)

plt.suptitle('Two-Channel QMF Filter Bank', fontsize=14)
plt.tight_layout()
plt.savefig('qmf_filter_bank.png', dpi=150)
plt.close()
```

### 9.4 켤레 직교 필터 (CQF)

2채널 시스템에서 완전 재구성을 위해 **CQF(Conjugate Quadrature Filters)**(Johnston 필터라고도 함)는 다음을 만족합니다:

$$|H_0(e^{j\omega})|^2 + |H_0(e^{j(\omega - \pi)})|^2 = 1$$

이것이 **전력 상보(power complementary)** 조건으로, 정확한 완전 재구성을 이끌어냅니다.

---

## 10. 응용

### 10.1 오디오 샘플률 변환

```python
def audio_sample_rate_conversion():
    """Demonstrate audio sample rate conversion."""
    # Simulate a simple audio signal at 44100 Hz
    fs_in = 44100
    duration = 0.05
    t = np.arange(0, duration, 1/fs_in)

    # Chirp signal (frequency sweep)
    x = signal.chirp(t, f0=100, t1=duration, f1=10000, method='logarithmic')

    # Convert to 48000 Hz using scipy.signal.resample_poly
    L = 160  # Upsample factor
    M = 147  # Downsample factor (48000/44100 = 160/147)

    y = signal.resample_poly(x, L, M)
    fs_out = fs_in * L / M

    # Convert to 22050 Hz (downsample by 2)
    y_half = signal.resample_poly(x, 1, 2)
    fs_half = fs_in / 2

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Spectrograms
    for ax, sig, fs_sig, title in [
        (axes[0], x, fs_in, f'Original ({fs_in} Hz)'),
        (axes[1], y, fs_out, f'Resampled to {fs_out:.0f} Hz'),
        (axes[2], y_half, fs_half, f'Resampled to {fs_half:.0f} Hz'),
    ]:
        f_spec, t_spec, Sxx = signal.spectrogram(sig, fs_sig, nperseg=256)
        ax.pcolormesh(t_spec * 1000, f_spec, 10*np.log10(Sxx + 1e-15),
                       shading='gouraud', cmap='viridis')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (ms)')
        ax.set_title(title)

    plt.suptitle('Audio Sample Rate Conversion', fontsize=14)
    plt.tight_layout()
    plt.savefig('audio_src.png', dpi=150)
    plt.close()

audio_sample_rate_conversion()
```

### 10.2 시그마-델타 ADC 개념

시그마-델타($\Sigma\Delta$) ADC는 오버샘플링(oversampling)과 잡음 성형(noise shaping)을 사용합니다:

1. **오버샘플링**: 률 $f_s = R \cdot f_\text{Nyquist}$로 샘플링 (여기서 $R$은 오버샘플링 비율, 예: 64 또는 256)
2. **잡음 성형**: 양자화 잡음을 고주파 영역으로 밀어냄
3. **데시메이션**: 다중률 체인을 사용하여 나이퀴스트(Nyquist) 률에서 최종 출력 획득

```python
def sigma_delta_demo():
    """Simplified sigma-delta ADC demonstration."""
    # Analog-like signal (oversampled)
    OSR = 64  # Oversampling ratio
    fs_nyquist = 8000
    fs_oversampled = fs_nyquist * OSR

    duration = 0.01
    t = np.arange(0, duration, 1/fs_oversampled)

    # Clean signal
    f_sig = 1000
    x = 0.5 * np.sin(2 * np.pi * f_sig * t)

    # First-order sigma-delta modulator
    y_sd = np.zeros(len(x))
    integrator = 0.0

    for n in range(len(x)):
        integrator += x[n] - y_sd[max(n-1, 0)]
        y_sd[n] = 1.0 if integrator >= 0 else -1.0

    # Decimate the 1-bit stream
    # Use a CIC-like filter (simple averaging)
    h_dec = signal.firwin(OSR * 4 + 1, 1.0 / OSR)
    y_filtered = np.convolve(y_sd, h_dec, mode='same')
    y_decimated = y_filtered[::OSR]

    fs_out = fs_oversampled / OSR
    t_out = np.arange(len(y_decimated)) / fs_out

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Original signal
    axes[0].plot(t[:2000] * 1000, x[:2000], 'b-', linewidth=1.5)
    axes[0].set_title(f'Input Signal ({f_sig} Hz)')
    axes[0].set_xlabel('Time (ms)')
    axes[0].grid(True, alpha=0.3)

    # Sigma-delta output (1-bit)
    axes[1].plot(t[:2000] * 1000, y_sd[:2000], 'r-', linewidth=0.5)
    axes[1].set_title(f'Sigma-Delta 1-bit Output (fs = {fs_oversampled/1000:.0f} kHz)')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].grid(True, alpha=0.3)

    # Decimated output
    axes[2].plot(t_out[:int(len(t_out)*0.8)] * 1000,
                 y_decimated[:int(len(y_decimated)*0.8)], 'g-', linewidth=1.5)
    axes[2].set_title(f'Decimated Output (fs = {fs_out/1000:.0f} kHz)')
    axes[2].set_xlabel('Time (ms)')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Sigma-Delta ADC: Oversampling + Decimation', fontsize=14)
    plt.tight_layout()
    plt.savefig('sigma_delta.png', dpi=150)
    plt.close()

sigma_delta_demo()
```

---

## 11. Python 구현

### 11.1 다중률 처리를 위한 scipy.signal 사용

```python
# scipy.signal provides convenient functions for multirate processing

# 1. Simple decimation
fs = 48000
t = np.arange(0, 0.1, 1/fs)
x = np.sin(2*np.pi*1000*t) + 0.3*np.sin(2*np.pi*5000*t)

# Decimate by 4 (includes anti-aliasing filter)
y_dec = signal.decimate(x, 4, ftype='fir')
print(f"Decimated: {len(x)} -> {len(y_dec)} samples")

# 2. Resample to arbitrary rate
y_resample = signal.resample(x, int(len(x) * 44100 / 48000))
print(f"Resampled 48k->44.1k: {len(x)} -> {len(y_resample)} samples")

# 3. Polyphase resampling (most efficient)
y_poly = signal.resample_poly(x, 441, 480)  # 48000 * 441/480 = 44100
print(f"Polyphase resample: {len(x)} -> {len(y_poly)} samples")

# 4. Upfirdn: combined upsample-filter-downsample
h = signal.firwin(128, 0.25)
y_upfirdn = signal.upfirdn(h, x, up=3, down=4)
print(f"upfirdn (up=3, down=4): {len(x)} -> {len(y_upfirdn)} samples")
```

### 11.2 완전한 다중률 처리 파이프라인

```python
def multirate_pipeline(x, fs_in, fs_out):
    """
    Complete sample rate conversion pipeline.

    Parameters:
        x: input signal
        fs_in: input sampling rate
        fs_out: output sampling rate

    Returns:
        y: resampled signal
        info: processing information
    """
    from math import gcd

    # Find rational approximation
    # Use exact ratio if both are integers
    g = gcd(int(fs_out), int(fs_in))
    L = int(fs_out) // g
    M = int(fs_in) // g

    # Simplify if factors are too large
    max_factor = 1000
    while L > max_factor or M > max_factor:
        g2 = gcd(L, M)
        if g2 > 1:
            L //= g2
            M //= g2
        else:
            # Approximate
            ratio = fs_out / fs_in
            best_L, best_M, best_error = 1, 1, abs(ratio - 1)
            for test_M in range(1, max_factor + 1):
                test_L = round(ratio * test_M)
                if 1 <= test_L <= max_factor:
                    error = abs(test_L / test_M - ratio)
                    if error < best_error:
                        best_L, best_M, best_error = test_L, test_M, error
            L, M = best_L, best_M
            break

    info = {
        'L': L,
        'M': M,
        'actual_ratio': L / M,
        'desired_ratio': fs_out / fs_in,
        'ratio_error': abs(L/M - fs_out/fs_in),
    }

    print(f"Rate conversion: {fs_in} -> {fs_out} Hz")
    print(f"  L/M = {L}/{M} = {L/M:.6f}")
    print(f"  Desired ratio: {fs_out/fs_in:.6f}")
    print(f"  Error: {info['ratio_error']:.2e}")

    # Use scipy's efficient polyphase resampler
    y = signal.resample_poly(x, L, M)

    info['input_samples'] = len(x)
    info['output_samples'] = len(y)

    return y, info

# Test various conversions
test_rates = [
    (44100, 48000, 'CD to Pro Audio'),
    (48000, 44100, 'Pro Audio to CD'),
    (44100, 22050, 'CD to Half Rate'),
    (16000, 8000, 'Wideband to Narrowband'),
    (96000, 44100, 'Hi-Res to CD'),
]

fs_source = 44100
t = np.arange(0, 0.01, 1/fs_source)
x = signal.chirp(t, 100, 0.01, 15000)

fig, axes = plt.subplots(len(test_rates), 1, figsize=(14, 3*len(test_rates)))

for ax, (fs_in, fs_out, desc) in zip(axes, test_rates):
    # Generate at source rate
    t_in = np.arange(0, 0.01, 1/fs_in)
    x_in = signal.chirp(t_in, 100, 0.01, min(15000, fs_in/2 - 1000))

    y, info = multirate_pipeline(x_in, fs_in, fs_out)

    # Plot spectrum
    freq_out = np.fft.rfftfreq(len(y), 1/fs_out)
    Y = np.abs(np.fft.rfft(y)) / len(y)
    ax.plot(freq_out, 20*np.log10(Y + 1e-15), 'b-', linewidth=1)
    ax.set_title(f'{desc}: {fs_in} -> {fs_out} Hz (L={info["L"]}, M={info["M"]})')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('dB')
    ax.set_xlim(0, fs_out/2)
    ax.set_ylim(-80, 0)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multirate_pipeline.png', dpi=150)
plt.close()
```

---

## 12. 연습 문제

### 연습 1: 다운샘플링 분석

신호 $x[n] = \cos(0.2\pi n) + \cos(0.7\pi n)$이 $f_s = 10$ kHz로 샘플링되어 있습니다.

(a) $x[n]$의 DTFT를 스케치하시오.

(b) $M = 2$로 다운샘플링하시오. 어느 주파수 성분이 앨리어싱되는지, 결과 스펙트럼이 어떻게 보이는지 분석적으로 결정하시오.

(c) $M = 4$로 다운샘플링하시오. 앨리어싱이 발생합니까? 어떤 성분이 겹칩니까?

(d) Python으로 구현: 원본 신호와 두 다운샘플링 버전의 스펙트럼을 그리시오. 분석적 예측을 검증하시오.

(e) $0.2\pi$ 성분을 왜곡하지 않고 $M = 4$ 데시메이션을 허용하는 앤티앨리어싱 필터를 설계하시오.

### 연습 2: 보간 품질

8 kHz로 샘플링된 1 kHz 사인파에서 시작하여:

(a) 영 삽입만 사용하여 $L = 4$로 업샘플링하시오. 스펙트럼을 그리고 이미징 아티팩트를 식별하시오.

(b) 세 가지 다른 방법을 사용하여 보간을 적용하시오:
   - 선형 보간(Linear interpolation)
   - FIR 저역통과 필터 (32탭)
   - FIR 저역통과 필터 (128탭)

(c) 고품질 참조(예: `scipy.signal.resample`)에 대한 각 방법의 SNR을 계산하여 보간 품질을 비교하시오.

(d) 이상적인 연속 사인파에 중첩된 모든 방법의 시간 영역 파형을 그리시오.

### 연습 3: 폴리페이즈 구현

128탭 FIR 필터와 데시메이션 인수 $M = 8$이 주어졌을 때:

(a) 폴리페이즈 분해를 수동으로 구현하시오. 서브필터는 몇 개입니까? 각 길이는 얼마입니까?

(b) 블록 단위로 신호를 처리하는(스트리밍 시뮬레이션) `PolyphaseDecimator` 클래스를 구현하시오.

(c) 출력이 `scipy.signal.decimate`와 일치하는지 검증하시오.

(d) 표준 vs 폴리페이즈 구현을 벤치마크하시오. 속도 향상 인수를 측정하시오.

### 연습 4: 유리수 률 변환

11025 Hz에서 8000 Hz로 신호를 변환하시오.

(a) 가장 단순한 $L/M$ 비율을 찾으시오. 힌트: $8000/11025 = ?$

(b) 필요한 앤티앨리어싱/앤티이미징 필터를 설계하시오.

(c) `signal.resample_poly`를 사용하여 변환을 구현하시오.

(d) 500, 1000, 2000, 3500 Hz의 톤을 포함하는 테스트 신호를 생성하시오. 변환 후 어떤 톤이 살아남고 어떤 것이 감쇠되어야 합니까? 스펙트럼 분석으로 검증하시오.

### 연습 5: 다단계 데시메이션 최적화

$M = 48$로 데시메이션해야 합니다.

(a) 48을 2~4 단계로 인수분해하는 모든 가능한 방법을 나열하시오 (예: $48 = 2 \times 24$, $48 = 2 \times 4 \times 6$ 등).

(b) 각 인수분해에 대해 출력 샘플당 총 곱셈-누산 연산 수를 추정하시오. 각 단계의 필터 차수는 $10 \times M_\text{stage}$로 가정합니다.

(c) 어떤 인수분해가 최적입니까? 구현하고 출력 품질을 검증하시오.

(d) 계산 시간 및 출력 SNR 측면에서 단일 단계 데시메이션과 비교하시오.

### 연습 6: QMF 필터 뱅크

오디오 처리를 위한 2채널 QMF 필터 뱅크를 설계하시오:

(a) QMF에 적합한 32탭 저역통과 프로토타입 필터를 설계하시오.

(b) 고역통과 분석 필터와 두 합성 필터를 구성하시오.

(c) 음악과 유사한 신호(배음의 합)를 처리하고 거의 완전한 재구성을 검증하시오.

(d) 필터 차수의 함수로 재구성 오차를 계산하시오 (8, 16, 32, 64, 128탭 시도).

(e) 서브밴드 코딩 구현: 저역통과 서브밴드를 16비트로, 고역통과 서브밴드를 8비트로 양자화하시오. 재구성하고 SNR을 측정하시오.

### 연습 7: CIC 필터

**CIC(Cascaded Integrator-Comb, 연속 적분기-빗 필터)**는 승수가 없는 계산 효율적인 데시메이션 필터입니다.

(a) $M$만큼 데시메이션하기 위한 단일 단계 CIC 필터(적분기 + 빗)를 구현하시오.

(b) CIC 필터의 주파수 응답을 유도하시오: $H(z) = \frac{1}{M}\frac{1 - z^{-M}}{1 - z^{-1}}$.

(c) $M = 8, 16, 32$에 대한 크기 응답을 그리고 이상적인 저역통과와 비교하시오.

(d) 다단계 CIC 필터($K$개 연속 단계)를 구현하고 $K$가 증가함에 따라 통과대역 드룹(droop)과 저지대역 감쇠가 어떻게 개선되는지 보이시오.

(e) 시그마-델타 ADC를 위한 CIC 기반 데시메이션 체인을 설계하시오: 64배 오버샘플링 후 CIC 데시메이션으로 최종 률에 도달.

---

## 참고문헌

1. **Vaidyanathan, P. P. (1993).** *Multirate Systems and Filter Banks*. Prentice Hall. [권위 있는 참고서]
2. **Oppenheim, A. V., & Schafer, R. W. (2010).** *Discrete-Time Signal Processing* (3rd ed.). Pearson. Chapter 4.
3. **Crochiere, R. E., & Rabiner, L. R. (1983).** *Multirate Digital Signal Processing*. Prentice Hall.
4. **Lyons, R. G. (2010).** *Understanding Digital Signal Processing* (3rd ed.). Pearson. Chapter 10.
5. **Fliege, N. J. (1994).** *Multirate Digital Signal Processing*. Wiley.
6. **SciPy Documentation** -- `scipy.signal.resample_poly`, `scipy.signal.decimate`: https://docs.scipy.org/doc/scipy/reference/signal.html

---

## 탐색

- 이전: [10. IIR 필터 설계](10_IIR_Filter_Design.md)
- 다음: [12. 스펙트럼 분석](12_Spectral_Analysis.md)
- [개요로 돌아가기](00_Overview.md)
