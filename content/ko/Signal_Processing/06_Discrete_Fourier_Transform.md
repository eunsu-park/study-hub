# 이산 푸리에 변환(Discrete Fourier Transform)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. DTFT, DFT, FFT 간의 관계를 설명하고 DFT가 DTFT를 샘플링함으로써 어떻게 도출되는지 기술할 수 있습니다.
2. 선형성, 원형 시프트(Circular Shift), 원형 합성곱(Circular Convolution), 파르스발 정리(Parseval's Theorem)를 포함한 DFT 성질을 적용할 수 있습니다.
3. 스펙트럼 누설(Spectral Leakage)을 설명하고 이를 줄이기 위한 윈도잉(Windowing) 전략(Hann, Hamming, Blackman)을 구현할 수 있습니다.
4. NumPy를 사용하여 FFT 알고리즘을 구현하고 실제 신호의 스펙트럼 분석에 활용할 수 있습니다.
5. 쿨리-튜키(Cooley-Tukey) 알고리즘의 분할 정복(Divide-and-Conquer) 구조를 분석하고 O(N log N) 계산 이점을 설명할 수 있습니다.

---

## 개요

이산 푸리에 변환(Discrete Fourier Transform, DFT)은 디지털 신호 처리의 계산적 핵심 도구입니다. DFT는 유한 길이의 표본 시퀀스를 유한 길이의 주파수 성분 시퀀스로 변환합니다. 고속 푸리에 변환(Fast Fourier Transform, FFT) 알고리즘과 결합하여, DFT는 효율적인 스펙트럼 분석, 필터링, 상관관계 계산 등 수많은 응용을 가능하게 합니다. 이 레슨은 DFT 이론, 성질, FFT 알고리즘, 그리고 실용적 사용법을 다룹니다.

**선수 과목:** [05. 표본화와 복원](05_Sampling_and_Reconstruction.md)

---

## 1. DTFT에서 DFT로

### 1.1 이산시간 푸리에 변환(DTFT) 복습

이산시간 신호 $x[n]$에 대해 DTFT는:

$$X(e^{j\omega}) = \sum_{n=-\infty}^{\infty} x[n] \, e^{-j\omega n}$$

DTFT의 주요 성질:
- **입력**: 무한 길이 이산 시퀀스
- **출력**: $\omega$에 대한 연속적이고 주기적인 함수 (주기 $2\pi$)
- **존재**: 절대 합산 가능성 또는 유한 에너지 필요
- **역변환**: $x[n] = \frac{1}{2\pi} \int_{-\pi}^{\pi} X(e^{j\omega}) \, e^{j\omega n} \, d\omega$

### 1.2 문제: DTFT는 연속함수

컴퓨터는 연속 함수를 저장하거나 계산할 수 없습니다. **이산** 주파수 표현이 필요합니다. DFT는 DTFT를 $N$개의 등간격 주파수에서 표본화하여 이 문제를 해결합니다.

### 1.3 DTFT의 표본화로서의 DFT

길이 $N$의 유한 길이 신호 $x[n]$에 대해, DFT는 다음 주파수에서 DTFT를 표본화합니다:

$$\omega_k = \frac{2\pi k}{N}, \quad k = 0, 1, \ldots, N-1$$

$$X[k] = X(e^{j\omega})\bigg|_{\omega = 2\pi k/N} = \sum_{n=0}^{N-1} x[n] \, e^{-j2\pi kn/N}$$

이는 다음을 의미합니다:
- DFT는 DTFT의 $N$개 주파수 표본을 제공
- 각 DFT 빈(bin) $k$는 주파수 $f_k = k \cdot f_s / N$에 해당
- 주파수 분해능은 $\Delta f = f_s / N$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def dtft_vs_dft():
    """Compare DTFT (dense sampling) with DFT (N-point sampling)."""
    # Signal: finite-length sequence
    N = 8
    x = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)  # Rectangular pulse

    # "DTFT" (computed at many points)
    omega = np.linspace(0, 2 * np.pi, 1024)
    X_dtft = np.zeros(len(omega), dtype=complex)
    for n in range(N):
        X_dtft += x[n] * np.exp(-1j * omega * n)

    # DFT (N points)
    X_dft = np.fft.fft(x)
    omega_dft = 2 * np.pi * np.arange(N) / N

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Magnitude
    axes[0].plot(omega / np.pi, np.abs(X_dtft), 'b-', linewidth=1,
                 label='|DTFT| (continuous)')
    axes[0].stem(omega_dft / np.pi, np.abs(X_dft), linefmt='r-',
                 markerfmt='ro', basefmt='k-', label=f'|DFT| (N={N} points)')
    axes[0].set_xlabel(r'Frequency ($\omega/\pi$)')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title('DTFT vs DFT: Magnitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Phase
    axes[1].plot(omega / np.pi, np.angle(X_dtft), 'b-', linewidth=1,
                 label='Phase of DTFT')
    axes[1].stem(omega_dft / np.pi, np.angle(X_dft), linefmt='r-',
                 markerfmt='ro', basefmt='k-', label=f'Phase of DFT')
    axes[1].set_xlabel(r'Frequency ($\omega/\pi$)')
    axes[1].set_ylabel('Phase (radians)')
    axes[1].set_title('DTFT vs DFT: Phase')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dtft_vs_dft.png', dpi=150)
    plt.show()

dtft_vs_dft()
```

---

## 2. DFT 정의와 역 DFT

### 2.1 순방향 DFT

시퀀스 $x[n]$의 $N$점 DFT ($n = 0, 1, \ldots, N-1$):

$$\boxed{X[k] = \sum_{n=0}^{N-1} x[n] \, W_N^{kn}, \quad k = 0, 1, \ldots, N-1}$$

여기서 $W_N = e^{-j2\pi/N}$은 **회전 인자(twiddle factor)** ($N$번째 단위 근)입니다.

### 2.2 역 DFT (IDFT)

$$\boxed{x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \, W_N^{-kn}, \quad n = 0, 1, \ldots, N-1}$$

### 2.3 행렬 형식

DFT는 행렬-벡터 곱으로 표현할 수 있습니다:

$$\mathbf{X} = \mathbf{W}_N \mathbf{x}$$

여기서 DFT 행렬은:

$$[\mathbf{W}_N]_{k,n} = W_N^{kn} = e^{-j2\pi kn/N}$$

$N = 4$일 때:

$$\mathbf{W}_4 = \begin{bmatrix} 1 & 1 & 1 & 1 \\ 1 & -j & -1 & j \\ 1 & -1 & 1 & -1 \\ 1 & j & -1 & -j \end{bmatrix}$$

DFT 행렬은 **유니타리(unitary)** (스케일링까지): $\mathbf{W}_N^{-1} = \frac{1}{N}\mathbf{W}_N^*$.

```python
def dft_matrix_demo():
    """Demonstrate the DFT as a matrix operation."""
    N = 8

    # Build DFT matrix
    n = np.arange(N)
    k = np.arange(N)
    W = np.exp(-2j * np.pi * np.outer(k, n) / N)

    # Test signal
    x = np.random.randn(N)

    # DFT via matrix multiplication
    X_matrix = W @ x

    # DFT via numpy.fft
    X_fft = np.fft.fft(x)

    # Verify they match
    print("DFT Matrix Computation")
    print("=" * 50)
    print(f"x = {x}")
    print(f"\nX (matrix) = {X_matrix}")
    print(f"X (FFT)    = {X_fft}")
    print(f"\nMax difference: {np.max(np.abs(X_matrix - X_fft)):.2e}")

    # Verify unitarity
    W_inv = np.conj(W) / N
    x_recovered = W_inv @ X_matrix
    print(f"\nRecovered x: {x_recovered.real}")
    print(f"Recovery error: {np.max(np.abs(x - x_recovered.real)):.2e}")

    # Visualize the DFT matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(W.real, cmap='RdBu_r', aspect='equal')
    axes[0].set_title(f'Re(W_{N})')
    axes[0].set_xlabel('n (time index)')
    axes[0].set_ylabel('k (frequency index)')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(W.imag, cmap='RdBu_r', aspect='equal')
    axes[1].set_title(f'Im(W_{N})')
    axes[1].set_xlabel('n (time index)')
    axes[1].set_ylabel('k (frequency index)')
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig('dft_matrix.png', dpi=150)
    plt.show()

dft_matrix_demo()
```

### 2.4 DFT 빈의 물리적 해석

$f_s$로 표본화된 신호에 대해, $k$번째 DFT 빈은 다음에 해당합니다:

| 물리량 | 표현식 |
|----------|-----------|
| 주파수 (Hz) | $f_k = k \cdot f_s / N$ |
| 각주파수 | $\omega_k = 2\pi k / N$ |
| 주파수 분해능 | $\Delta f = f_s / N$ |
| 최고 주파수 (빈 $N/2$) | $f_s / 2$ (나이퀴스트) |

실수 신호의 경우, $X[k]$와 $X[N-k]$는 켤레 복소수이므로, $k > N/2$인 빈은 음의 주파수를 나타냅니다. 크기 스펙트럼은 $N/2$에 대해 대칭입니다.

---

## 3. DFT 성질

### 3.1 선형성(Linearity)

$$\text{DFT}\{a \cdot x_1[n] + b \cdot x_2[n]\} = a \cdot X_1[k] + b \cdot X_2[k]$$

### 3.2 원형(주기적) 이동

시간에서 $m$만큼의 원형 이동은 주파수에서 위상 회전에 해당합니다:

$$x[(n-m) \bmod N] \quad \xleftrightarrow{\text{DFT}} \quad X[k] \cdot W_N^{mk} = X[k] \cdot e^{-j2\pi mk/N}$$

> 참고: 이것은 **원형** 이동으로, 선형 이동이 아닙니다. 신호가 $N$으로 모듈러(modular) 감싸집니다.

### 3.3 원형 합성곱

선형 합성곱과 원형 합성곱은 근본적으로 다릅니다.

두 $N$점 시퀀스의 **원형 합성곱**:

$$(x_1 \circledast x_2)[n] = \sum_{m=0}^{N-1} x_1[m] \, x_2[(n-m) \bmod N]$$

DFT 곱셈 성질:

$$\boxed{x_1[n] \circledast x_2[n] \quad \xleftrightarrow{\text{DFT}} \quad X_1[k] \cdot X_2[k]}$$

원형 합성곱을 사용하여 **선형** 합성곱을 계산하려면 (overlap-add 필터링에 필요):
- 두 시퀀스를 모두 길이 $\geq N_1 + N_2 - 1$로 영 패딩
- 영 패딩된 시퀀스의 원형 합성곱 계산
- 결과가 선형 합성곱과 같음

```python
def circular_vs_linear_convolution():
    """Demonstrate circular vs. linear convolution."""
    # Two short sequences
    x1 = np.array([1, 2, 3, 4])
    x2 = np.array([1, 0, -1])

    N1, N2 = len(x1), len(x2)

    # Linear convolution (length N1+N2-1 = 6)
    y_linear = np.convolve(x1, x2)

    # Circular convolution (length = max(N1, N2) = 4)
    N_circ = max(N1, N2)
    x2_padded = np.zeros(N_circ)
    x2_padded[:N2] = x2
    X1_circ = np.fft.fft(x1, N_circ)
    X2_circ = np.fft.fft(x2_padded, N_circ)
    y_circular = np.real(np.fft.ifft(X1_circ * X2_circ))

    # Linear convolution via DFT (zero-pad to N1+N2-1)
    N_linear = N1 + N2 - 1
    X1_lin = np.fft.fft(x1, N_linear)
    X2_lin = np.fft.fft(x2, N_linear)
    y_fft_linear = np.real(np.fft.ifft(X1_lin * X2_lin))

    print("Circular vs. Linear Convolution")
    print("=" * 60)
    print(f"x1 = {x1}")
    print(f"x2 = {x2}")
    print(f"\nLinear convolution (np.convolve): {y_linear}")
    print(f"Circular convolution (N={N_circ}): {y_circular}")
    print(f"Linear conv via FFT (N={N_linear}): {np.round(y_fft_linear, 6)}")
    print(f"\nCircular != Linear (aliasing in circular)")
    print(f"FFT linear matches np.convolve: "
          f"{np.allclose(y_linear, y_fft_linear)}")

circular_vs_linear_convolution()
```

출력:
```
Circular vs. Linear Convolution
============================================================
x1 = [1 2 3 4]
x2 = [1 0 -1]

Linear convolution (np.convolve): [ 1  2  2  2 -3 -4]
Circular convolution (N=4): [-3.  2.  2.  2.]
Linear conv via FFT (N=6): [ 1.  2.  2.  2. -3. -4.]

Circular != Linear (aliasing in circular)
FFT linear matches np.convolve: True
```

### 3.4 파르스발 정리 (에너지 보존)

$$\boxed{\sum_{n=0}^{N-1} |x[n]|^2 = \frac{1}{N} \sum_{k=0}^{N-1} |X[k]|^2}$$

전체 에너지는 시간 영역과 주파수 영역 사이에서 보존됩니다 ($1/N$ 인수까지).

```python
def verify_parseval():
    """Verify Parseval's theorem numerically."""
    N = 256
    x = np.random.randn(N)
    X = np.fft.fft(x)

    energy_time = np.sum(np.abs(x) ** 2)
    energy_freq = np.sum(np.abs(X) ** 2) / N

    print("Parseval's Theorem Verification")
    print("=" * 40)
    print(f"Time domain energy:  {energy_time:.10f}")
    print(f"Freq domain energy:  {energy_freq:.10f}")
    print(f"Difference:          {abs(energy_time - energy_freq):.2e}")

verify_parseval()
```

### 3.5 DFT 성질 요약

| 성질 | 시간 영역 | 주파수 영역 |
|----------|------------|-----------------|
| 선형성 | $ax_1[n] + bx_2[n]$ | $aX_1[k] + bX_2[k]$ |
| 원형 이동 | $x[(n-m) \bmod N]$ | $W_N^{mk} X[k]$ |
| 변조 | $W_N^{-ln} x[n]$ | $X[(k-l) \bmod N]$ |
| 원형 합성곱 | $x_1 \circledast x_2$ | $X_1[k] \cdot X_2[k]$ |
| 곱셈 | $x_1[n] \cdot x_2[n]$ | $\frac{1}{N} X_1 \circledast X_2$ |
| 켤레화 | $x^*[n]$ | $X^*[(-k) \bmod N]$ |
| 시간 반전 | $x[(-n) \bmod N]$ | $X[(-k) \bmod N]$ |
| 파르스발 | $\sum |x[n]|^2$ | $\frac{1}{N}\sum |X[k]|^2$ |

---

## 4. 영 패딩과 주파수 분해능

### 4.1 주파수 분해능

DFT는 $\Delta f = f_s / N$ 간격으로 $N$개의 주파수 표본을 제공합니다. 주파수 표본의 밀도를 높이려면 (DTFT를 보간하려면), 신호에 **영 패딩(zero-padding)**을 적용할 수 있습니다.

> **중요:** 영 패딩은 DFT 점 수를 늘려 (더 세밀한 주파수 격자) 주파수 표본 밀도를 높이지만, 실제 스펙트럼 분해능은 **향상되지 않습니다**. 실제 분해능은 신호 지속 시간 $T = N \cdot T_s$에 의해 결정됩니다.

### 4.2 영 패딩 시연

```python
def zero_padding_demo():
    """Demonstrate the effect of zero-padding on frequency resolution."""
    fs = 100  # Hz
    T = 0.5   # 0.5 seconds of data
    N = int(fs * T)  # 50 samples

    # Two closely-spaced sinusoids
    f1, f2 = 20, 22  # Hz (2 Hz apart)
    n = np.arange(N)
    x = np.cos(2 * np.pi * f1 * n / fs) + np.cos(2 * np.pi * f2 * n / fs)

    # DFT with different amounts of zero-padding
    nfft_values = [N, 2 * N, 4 * N, 16 * N]

    fig, axes = plt.subplots(len(nfft_values), 1, figsize=(14, 12))

    for ax, nfft in zip(axes, nfft_values):
        X = np.fft.fft(x, n=nfft)
        freqs = np.fft.fftfreq(nfft, d=1/fs)

        # Plot only positive frequencies
        pos = freqs >= 0
        ax.plot(freqs[pos], 20 * np.log10(np.abs(X[pos]) / N + 1e-12),
                'b-', linewidth=1)
        ax.axvline(f1, color='red', linestyle='--', alpha=0.5, label=f'f1={f1} Hz')
        ax.axvline(f2, color='green', linestyle='--', alpha=0.5, label=f'f2={f2} Hz')
        delta_f = fs / nfft
        ax.set_title(f'NFFT = {nfft} (zero-padded from {N}) | '
                     f'Frequency spacing = {delta_f:.2f} Hz')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_xlim(10, 35)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('zero_padding.png', dpi=150)
    plt.show()

zero_padding_demo()
```

### 4.3 진정한 분해능 대 영 패딩

| 개념 | 결정 요인 | 영 패딩의 효과 |
|---------|--------------|----------------------|
| **주파수 분해능** ($\Delta f$) | 신호 지속 시간 $T$ | 변화 없음 |
| **DFT 빈 간격** | NFFT = $N$ + 영 패딩 | 감소 (더 세밀한 격자) |
| **두 톤 분해 능력** | $\Delta f \approx 1/T$ | 향상되지 않음 |
| **시각적 외관** | 둘 다 | 더 부드러운 스펙트럼 |

---

## 5. 스펙트럼 누설과 윈도잉

### 5.1 누설 문제

유한 길이 신호의 DFT를 계산할 때, 신호가 주기 $N$으로 주기적이라고 암묵적으로 가정합니다. 신호 주파수가 DFT 빈에 정확히 떨어지지 않으면, 에너지가 인접한 빈으로 "누설"됩니다.

**원인:** 무한 신호를 $N$개의 표본으로 절단하는 것은 직사각형 창을 곱하는 것과 같습니다. 주파수 영역에서 이는 실제 스펙트럼을 직사각형 창의 sinc 형 푸리에 변환과 합성곱하는 것입니다.

### 5.2 창 함수(Window Functions)

스펙트럼 누설을 줄이기 위해, DFT를 계산하기 전에 신호에 **창 함수** $w[n]$를 곱합니다:

$$y[n] = x[n] \cdot w[n]$$

일반적인 창 함수:

| 창 함수 | 주엽 너비 | 부엽 레벨 | 사용 사례 |
|--------|----------------|-----------------|----------|
| 직사각형(Rectangular) | 가장 좁음 ($2\pi/N$) | -13 dB | 최대 분해능 |
| 해닝(Hann) | $4\pi/N$ | -31 dB | 범용 |
| 해밍(Hamming) | $4\pi/N$ | -42 dB | 음성 처리 |
| 블랙만(Blackman) | $6\pi/N$ | -58 dB | 높은 동적 범위 |
| 카이저(Kaiser) ($\beta$) | 가변 | 가변 | 조정 가능한 트레이드오프 |
| 평탄 상단(Flat-top) | 넓음 | 매우 낮음 | 진폭 정확도 |

**트레이드오프:** 더 넓은 주엽 = 낮은 주파수 분해능, 하지만 낮은 부엽 = 적은 누설.

### 5.3 창 함수 비교

```python
def compare_windows():
    """Compare different window functions and their spectral properties."""
    N = 64
    n = np.arange(N)

    windows = {
        'Rectangular': np.ones(N),
        'Hann': np.hanning(N),
        'Hamming': np.hamming(N),
        'Blackman': np.blackman(N),
        'Kaiser (beta=8)': np.kaiser(N, 8),
        'Kaiser (beta=14)': np.kaiser(N, 14),
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    nfft = 4096  # Large FFT for smooth spectrum

    for name, w in windows.items():
        # Time domain
        axes[0].plot(n, w, linewidth=1.5, label=name)

        # Frequency domain (log magnitude)
        W = np.fft.fft(w, nfft)
        W_shifted = np.fft.fftshift(W)
        freq = np.arange(nfft) - nfft // 2
        mag_db = 20 * np.log10(np.abs(W_shifted) / np.max(np.abs(W_shifted)) + 1e-12)
        axes[1].plot(freq[:nfft // 2] * 2 * np.pi / nfft, mag_db[:nfft // 2],
                     linewidth=1.5, label=name)

    axes[0].set_title('Window Functions (Time Domain)')
    axes[0].set_xlabel('Sample index n')
    axes[0].set_ylabel('w[n]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Window Functions (Frequency Domain)')
    axes[1].set_xlabel(r'Normalized frequency ($\omega$)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_ylim(-120, 5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('windows.png', dpi=150)
    plt.show()

compare_windows()
```

### 5.4 실제 스펙트럼 누설

```python
def demonstrate_leakage():
    """Show spectral leakage and how windowing helps."""
    fs = 100  # Hz
    N = 64
    n = np.arange(N)
    t = n / fs

    # Case 1: frequency on a DFT bin (no leakage)
    f_on_bin = fs * 10 / N  # Exactly bin 10
    x_on = np.sin(2 * np.pi * f_on_bin * t)

    # Case 2: frequency between bins (leakage)
    f_off_bin = fs * 10.5 / N  # Between bins 10 and 11
    x_off = np.sin(2 * np.pi * f_off_bin * t)

    # Case 3: off-bin with Hann window
    w = np.hanning(N)
    x_off_windowed = x_off * w

    nfft = 512
    freqs = np.fft.fftfreq(nfft, d=1/fs)[:nfft // 2]

    cases = [
        (f'On-bin: f = {f_on_bin:.2f} Hz (Rectangular)', x_on),
        (f'Off-bin: f = {f_off_bin:.2f} Hz (Rectangular)', x_off),
        (f'Off-bin: f = {f_off_bin:.2f} Hz (Hann window)', x_off_windowed),
    ]

    fig, axes = plt.subplots(len(cases), 1, figsize=(14, 10))

    for ax, (title, sig) in zip(axes, cases):
        X = np.fft.fft(sig, nfft)
        mag = np.abs(X[:nfft // 2])
        mag_db = 20 * np.log10(mag / np.max(mag) + 1e-12)

        ax.plot(freqs, mag_db, 'b-', linewidth=1)
        ax.set_title(title)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_ylim(-80, 5)
        ax.set_xlim(0, fs / 2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('spectral_leakage.png', dpi=150)
    plt.show()

demonstrate_leakage()
```

---

## 6. 고속 푸리에 변환(FFT)

### 6.1 계산 문제

DFT를 직접 계산하려면:
- $N$개의 출력값 각각에 대해 $N$번의 복소 곱셈과 $N-1$번의 복소 덧셈
- 총: $O(N^2)$ 연산

$N = 10^6$일 때: $\sim 10^{12}$ 연산 (실시간 처리에는 비현실적).

### 6.2 쿨리-튜키 알고리즘

FFT는 회전 인자 $W_N^{kn}$의 대칭성과 주기성을 이용합니다.

**핵심 관찰:**
1. $W_N^{k+N} = W_N^k$ (주기성)
2. $W_N^{k+N/2} = -W_N^k$ (대칭성 / 반주기 부정)
3. $W_N^{2kn} = W_{N/2}^{kn}$ (축약)

**기수-2 시간-분할(Radix-2 Decimation-in-Time, DIT):**

$x[n]$을 짝수 인덱스와 홀수 인덱스 부분 시퀀스로 분할:

$$X[k] = \underbrace{\sum_{r=0}^{N/2-1} x[2r] \, W_{N/2}^{kr}}_{G[k] \text{ (짝수 항)}} + W_N^k \underbrace{\sum_{r=0}^{N/2-1} x[2r+1] \, W_{N/2}^{kr}}_{H[k] \text{ (홀수 항)}}$$

이를 통해 **버터플라이(butterfly)** 계산이 주어집니다:

$$X[k] = G[k] + W_N^k \cdot H[k]$$
$$X[k + N/2] = G[k] - W_N^k \cdot H[k]$$

$k = 0, 1, \ldots, N/2 - 1$에 대해.

각 단계는 문제 크기를 절반으로 줄여, $N/2$개의 버터플라이로 이루어진 $\log_2 N$개의 단계를 제공합니다.

**복잡도:** $O(N \log_2 N)$

| $N$ | DFT ($N^2$) | FFT ($N \log_2 N$) | 속도 향상 |
|-----|------------|-------------------|---------|
| 64 | 4,096 | 384 | 10.7x |
| 256 | 65,536 | 2,048 | 32x |
| 1,024 | 1,048,576 | 10,240 | 102x |
| 65,536 | $4.3 \times 10^9$ | $1.0 \times 10^6$ | 4,096x |
| 1,048,576 | $1.1 \times 10^{12}$ | $2.1 \times 10^7$ | 52,429x |

### 6.3 버터플라이 다이어그램

```
Stage 0 (N=2 DFTs):     Stage 1 (N=4 DFTs):     Stage 2 (N=8 DFT):

x[0] ─────●──────────── ●──────────────────────── ●──── X[0]
           │╲            │╲                        │╲
x[4] ─────●──────────── │  ╲                      │  ╲
                  W⁰    │   ●──────────────────── │   ●── X[1]
x[2] ─────●──────────── ●  ╱           W⁰        │  ╱
           │╲            │╱                        │╱
x[6] ─────●──────────── ●──────────────────────── ●──── X[2]
                                W²                │
x[1] ─────●──────────── ●──────────────────────── ●──── X[3]
           │╲            │╲                        │
x[5] ─────●──────────── │  ╲                      │
                  W⁰    │   ●──────────────────── ●──── X[4]
x[3] ─────●──────────── ●  ╱           W¹
           │╲            │╱
x[7] ─────●──────────── ●──────────────────────── ...
```

### 6.4 기수-2 DIT 구현

```python
def fft_dit(x):
    """Radix-2 Decimation-in-Time FFT (recursive implementation)."""
    N = len(x)

    # Base case
    if N == 1:
        return x.copy()

    # Check power of 2
    if N & (N - 1) != 0:
        raise ValueError("Length must be a power of 2")

    # Split into even and odd
    x_even = x[0::2]
    x_odd = x[1::2]

    # Recursive FFT
    G = fft_dit(x_even)
    H = fft_dit(x_odd)

    # Twiddle factors
    k = np.arange(N // 2)
    W = np.exp(-2j * np.pi * k / N)

    # Butterfly
    X = np.zeros(N, dtype=complex)
    X[:N // 2] = G + W * H
    X[N // 2:] = G - W * H

    return X


def fft_dit_iterative(x):
    """Iterative in-place radix-2 DIT FFT."""
    N = len(x)
    stages = int(np.log2(N))

    # Bit-reversal permutation
    X = np.array(x, dtype=complex)
    for i in range(N):
        j = int(bin(i)[2:].zfill(stages)[::-1], 2)
        if i < j:
            X[i], X[j] = X[j], X[i]

    # Butterfly stages
    for s in range(1, stages + 1):
        m = 2 ** s
        wm = np.exp(-2j * np.pi / m)

        for k in range(0, N, m):
            w = 1.0
            for j in range(m // 2):
                t = w * X[k + j + m // 2]
                u = X[k + j]
                X[k + j] = u + t
                X[k + j + m // 2] = u - t
                w *= wm

    return X


def verify_fft_implementations():
    """Verify custom FFT implementations against numpy."""
    N = 256
    x = np.random.randn(N) + 1j * np.random.randn(N)

    X_numpy = np.fft.fft(x)
    X_recursive = fft_dit(x)
    X_iterative = fft_dit_iterative(x)

    print("FFT Implementation Verification")
    print("=" * 50)
    print(f"N = {N}")
    print(f"Max error (recursive):  {np.max(np.abs(X_recursive - X_numpy)):.2e}")
    print(f"Max error (iterative):  {np.max(np.abs(X_iterative - X_numpy)):.2e}")

verify_fft_implementations()
```

### 6.5 기수-2 주파수-분할(Decimation-in-Frequency, DIF)

DIF 변형은 입력 대신 출력을 분할합니다:

$$X[2r] = \sum_{n=0}^{N/2-1} (x[n] + x[n + N/2]) \, W_{N/2}^{rn}$$

$$X[2r+1] = \sum_{n=0}^{N/2-1} (x[n] - x[n + N/2]) \, W_N^n \, W_{N/2}^{rn}$$

DIF 버터플라이는 먼저 뺄셈을 수행한 후 회전 인자를 곱합니다 (DIT와 반대 순서).

### 6.6 FFT 복잡도 비교

```python
import time

def benchmark_dft_vs_fft():
    """Benchmark direct DFT vs FFT for various sizes."""
    sizes = [2**k for k in range(4, 15)]  # 16 to 16384

    results = []

    for N in sizes:
        x = np.random.randn(N) + 1j * np.random.randn(N)

        # Direct DFT (only for small N)
        if N <= 4096:
            n_arr = np.arange(N)
            W_matrix = np.exp(-2j * np.pi * np.outer(n_arr, n_arr) / N)
            start = time.perf_counter()
            for _ in range(3):
                X_direct = W_matrix @ x
            t_direct = (time.perf_counter() - start) / 3
        else:
            t_direct = None

        # FFT
        start = time.perf_counter()
        for _ in range(100):
            X_fft = np.fft.fft(x)
        t_fft = (time.perf_counter() - start) / 100

        results.append((N, t_direct, t_fft))

    print("DFT vs FFT Benchmark")
    print("=" * 65)
    print(f"{'N':>8s} | {'DFT (ms)':>12s} | {'FFT (ms)':>12s} | {'Speedup':>10s}")
    print("-" * 65)
    for N, t_d, t_f in results:
        t_d_str = f"{t_d*1000:.4f}" if t_d else "N/A"
        speedup = f"{t_d/t_f:.1f}x" if t_d else "N/A"
        print(f"{N:8d} | {t_d_str:>12s} | {t_f*1000:.4f} | {speedup:>10s}")

benchmark_dft_vs_fft()
```

---

## 7. 실용적인 FFT 활용

### 7.1 numpy.fft 사용법

```python
def practical_fft_guide():
    """Comprehensive guide to using numpy.fft for spectral analysis."""
    # Generate a test signal
    fs = 1000  # Sampling rate
    T = 1.0    # Duration
    N = int(fs * T)
    t = np.arange(N) / fs

    # Signal: three sinusoids + noise
    f1, f2, f3 = 50, 120, 300  # Hz
    a1, a2, a3 = 1.0, 0.5, 0.3  # Amplitudes
    x = (a1 * np.sin(2 * np.pi * f1 * t) +
         a2 * np.sin(2 * np.pi * f2 * t) +
         a3 * np.sin(2 * np.pi * f3 * t) +
         0.2 * np.random.randn(N))  # Noise

    # Compute FFT
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, d=1/fs)

    # For real signals, use rfft (returns only positive frequencies)
    X_r = np.fft.rfft(x)
    freqs_r = np.fft.rfftfreq(N, d=1/fs)

    # Magnitude spectrum
    mag = np.abs(X_r) * 2 / N  # Scale for single-sided spectrum
    mag[0] /= 2  # DC component (no doubling)
    if N % 2 == 0:
        mag[-1] /= 2  # Nyquist component

    # Power spectral density (PSD)
    psd = np.abs(X_r) ** 2 / (N * fs)
    psd[1:-1] *= 2  # Double non-DC, non-Nyquist bins

    fig, axes = plt.subplots(4, 1, figsize=(14, 14))

    # Time domain
    axes[0].plot(t[:200] * 1000, x[:200], 'b-', linewidth=0.5)
    axes[0].set_title('Time Domain Signal')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # Magnitude spectrum (linear)
    axes[1].plot(freqs_r, mag, 'b-', linewidth=1)
    axes[1].set_title('Single-Sided Magnitude Spectrum')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    # Magnitude spectrum (dB)
    mag_db = 20 * np.log10(mag + 1e-12)
    axes[2].plot(freqs_r, mag_db, 'b-', linewidth=1)
    axes[2].set_title('Magnitude Spectrum (dB)')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Magnitude (dB)')
    axes[2].grid(True, alpha=0.3)

    # PSD
    psd_db = 10 * np.log10(psd + 1e-12)
    axes[3].plot(freqs_r, psd_db, 'b-', linewidth=1)
    axes[3].set_title('Power Spectral Density')
    axes[3].set_xlabel('Frequency (Hz)')
    axes[3].set_ylabel('PSD (dB/Hz)')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('practical_fft.png', dpi=150)
    plt.show()

    # Print detected peaks
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(mag, height=0.1)
    print("\nDetected Frequency Peaks:")
    print(f"{'Frequency (Hz)':>15s} | {'Amplitude':>10s}")
    print("-" * 30)
    for p in peaks:
        print(f"{freqs_r[p]:15.1f} | {mag[p]:10.4f}")

practical_fft_guide()
```

### 7.2 NumPy의 주요 FFT 함수들

| 함수 | 설명 |
|----------|-----------|
| `np.fft.fft(x, n)` | 복소/실수 신호의 N점 FFT |
| `np.fft.ifft(X, n)` | 역 FFT |
| `np.fft.rfft(x, n)` | 실수 입력용 FFT (양의 주파수만) |
| `np.fft.irfft(X, n)` | 실수 출력용 역 FFT |
| `np.fft.fftfreq(n, d)` | `fft` 출력의 주파수 배열 |
| `np.fft.rfftfreq(n, d)` | `rfft` 출력의 주파수 배열 |
| `np.fft.fftshift(X)` | 영 주파수를 중앙으로 이동 |
| `np.fft.ifftshift(X)` | `fftshift` 되돌리기 |

### 7.3 일반적인 실수(Pitfalls)

```python
def fft_pitfalls():
    """Demonstrate common FFT mistakes and correct usage."""
    fs = 1000
    N = 1000
    t = np.arange(N) / fs
    x = np.sin(2 * np.pi * 100 * t)  # 100 Hz sine

    print("Common FFT Pitfalls")
    print("=" * 60)

    # Pitfall 1: Wrong frequency axis
    X = np.fft.fft(x)
    freqs_wrong = np.arange(N)  # WRONG: just bin indices
    freqs_right = np.fft.fftfreq(N, d=1/fs)  # RIGHT: Hz

    print("\n1. Frequency axis:")
    peak_bin = np.argmax(np.abs(X[:N//2]))
    print(f"   Peak at bin {peak_bin}")
    print(f"   Wrong freq: {freqs_wrong[peak_bin]} (meaningless)")
    print(f"   Right freq: {freqs_right[peak_bin]} Hz")

    # Pitfall 2: Forgetting to scale amplitude
    X_r = np.fft.rfft(x)
    print(f"\n2. Amplitude scaling:")
    print(f"   Raw |X[peak]| = {np.abs(X_r[peak_bin]):.1f}")
    print(f"   Scaled 2*|X|/N = {2*np.abs(X_r[peak_bin])/N:.4f} (correct)")

    # Pitfall 3: Aliasing due to non-power-of-2 length
    # numpy.fft handles any length, but power-of-2 is fastest
    for n in [1000, 1024, 1023]:
        start = time.perf_counter()
        for _ in range(1000):
            np.fft.fft(np.random.randn(n))
        elapsed = (time.perf_counter() - start)
        print(f"\n3. FFT of length {n}: {elapsed:.4f}s for 1000 runs")

fft_pitfalls()
```

---

## 8. 스펙트럼 분석 응용

### 8.1 음악 화음 분석

```python
def analyze_chord():
    """Spectral analysis of a musical chord (C major)."""
    fs = 8000  # Hz
    T = 1.0
    N = int(fs * T)
    t = np.arange(N) / fs

    # C major chord: C4=261.63, E4=329.63, G4=392.00 Hz
    notes = {
        'C4': 261.63,
        'E4': 329.63,
        'G4': 392.00,
    }

    # Generate chord with harmonics
    x = np.zeros(N)
    for name, f0 in notes.items():
        for harmonic in range(1, 5):
            amplitude = 1.0 / harmonic  # Harmonic series decay
            x += amplitude * np.sin(2 * np.pi * f0 * harmonic * t)

    x = x / np.max(np.abs(x))  # Normalize
    x += 0.01 * np.random.randn(N)  # Add slight noise

    # Windowed FFT
    window = np.hanning(N)
    X = np.fft.rfft(x * window)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    mag_db = 20 * np.log10(np.abs(X) / np.max(np.abs(X)) + 1e-12)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Time domain (first 50 ms)
    samples_50ms = int(0.05 * fs)
    axes[0].plot(t[:samples_50ms] * 1000, x[:samples_50ms], 'b-', linewidth=0.5)
    axes[0].set_title('C Major Chord (Time Domain)')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # Frequency domain
    axes[1].plot(freqs, mag_db, 'b-', linewidth=0.5)
    for name, f0 in notes.items():
        axes[1].axvline(f0, color='red', linestyle='--', alpha=0.5)
        axes[1].annotate(name, xy=(f0, 0), xytext=(f0 + 10, 5),
                         fontsize=9, color='red')
    axes[1].set_title('Spectrum of C Major Chord')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_xlim(0, 2000)
    axes[1].set_ylim(-60, 5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('chord_analysis.png', dpi=150)
    plt.show()

analyze_chord()
```

### 8.2 단시간 푸리에 변환(STFT) / 스펙트로그램

비정상 신호의 경우, STFT는 신호를 겹치는 세그먼트로 나누어 각각에 창을 적용하고 FFT를 계산합니다:

$$\text{STFT}\{x[n]\}(m, k) = \sum_{n=0}^{L-1} x[n + mH] \, w[n] \, e^{-j2\pi kn/N_{\text{FFT}}}$$

여기서 $L$은 창 길이, $H$는 홉(hop) 크기, $N_{\text{FFT}}$는 FFT 크기입니다.

```python
def spectrogram_demo():
    """Create a spectrogram of a chirp signal."""
    fs = 8000
    T = 2.0
    N = int(fs * T)
    t = np.arange(N) / fs

    # Linear chirp from 100 Hz to 3000 Hz
    f0, f1 = 100, 3000
    x = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * T) * t ** 2))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Time domain
    axes[0].plot(t, x, 'b-', linewidth=0.3)
    axes[0].set_title('Linear Chirp Signal (100 Hz to 3000 Hz)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # Spectrogram
    nfft = 256
    noverlap = nfft * 3 // 4
    axes[1].specgram(x, NFFT=nfft, Fs=fs, noverlap=noverlap,
                     cmap='viridis')
    axes[1].set_title('Spectrogram')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.savefig('spectrogram.png', dpi=150)
    plt.show()

spectrogram_demo()
```

### 8.3 FFT 기반 필터링 (Overlap-Add 방법)

```python
def fft_filtering():
    """Demonstrate frequency-domain filtering using overlap-add."""
    fs = 1000
    T = 1.0
    N = int(fs * T)
    t = np.arange(N) / fs

    # Signal: 50 Hz + 200 Hz + noise
    x = (np.sin(2 * np.pi * 50 * t) +
         0.5 * np.sin(2 * np.pi * 200 * t) +
         0.3 * np.random.randn(N))

    # Design a lowpass FIR filter (keep < 100 Hz)
    M = 101  # Filter length
    h = signal.firwin(M, cutoff=100, fs=fs)

    # Method 1: Time-domain convolution
    y_time = np.convolve(x, h, mode='same')

    # Method 2: FFT-based (overlap-add)
    nfft = 2 ** int(np.ceil(np.log2(N + M - 1)))
    X = np.fft.fft(x, nfft)
    H = np.fft.fft(h, nfft)
    Y = X * H
    y_fft = np.real(np.fft.ifft(Y))[:N]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    axes[0].plot(t, x, 'b-', linewidth=0.5)
    axes[0].set_title('Original Signal (50 Hz + 200 Hz + Noise)')
    axes[0].set_xlabel('Time (s)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, y_time, 'r-', linewidth=0.5, label='Time-domain')
    axes[1].plot(t, y_fft[:N], 'g--', linewidth=0.5, label='FFT-based')
    axes[1].set_title('Filtered Signal (Lowpass < 100 Hz)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Compare spectra
    f_orig = np.fft.rfftfreq(N, d=1/fs)
    X_orig = np.abs(np.fft.rfft(x))
    X_filt = np.abs(np.fft.rfft(y_fft[:N]))

    axes[2].plot(f_orig, 20 * np.log10(X_orig / np.max(X_orig) + 1e-12),
                 'b-', alpha=0.5, label='Original')
    axes[2].plot(f_orig, 20 * np.log10(X_filt / np.max(X_filt) + 1e-12),
                 'r-', label='Filtered')
    axes[2].set_title('Spectrum Comparison')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Magnitude (dB)')
    axes[2].set_ylim(-80, 5)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fft_filtering.png', dpi=150)
    plt.show()

fft_filtering()
```

---

## 9. 2차원 DFT

### 9.1 정의

2D DFT는 이미지와 공간 데이터로 자연스럽게 확장됩니다:

$$X[k_1, k_2] = \sum_{n_1=0}^{N_1-1} \sum_{n_2=0}^{N_2-1} x[n_1, n_2] \, e^{-j2\pi(k_1 n_1/N_1 + k_2 n_2/N_2)}$$

2D FFT는 각 차원을 따라 1D FFT로 계산할 수 있습니다 (분리 가능성).

```python
def fft_2d_demo():
    """Demonstrate 2D FFT on a simple image pattern."""
    N = 256

    # Create a test image with known spatial frequencies
    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y)

    # Two spatial frequencies
    img = (np.sin(2 * np.pi * 10 * X / N) +
           np.sin(2 * np.pi * 20 * Y / N) +
           0.5 * np.sin(2 * np.pi * (15 * X + 25 * Y) / N))

    # 2D FFT
    F = np.fft.fft2(img)
    F_shifted = np.fft.fftshift(F)
    mag = np.log10(np.abs(F_shifted) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Spatial Domain (Image)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    axes[1].imshow(mag, cmap='hot', extent=[-N//2, N//2, -N//2, N//2])
    axes[1].set_title('2D FFT Magnitude (log scale)')
    axes[1].set_xlabel('kx')
    axes[1].set_ylabel('ky')

    plt.tight_layout()
    plt.savefig('fft_2d.png', dpi=150)
    plt.show()

fft_2d_demo()
```

---

## 10. 요약

```
┌─────────────────────────────────────────────────────────────────┐
│               이산 푸리에 변환(Discrete Fourier Transform, DFT)   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  DTFT → DFT:                                                    │
│    DFT는 DTFT를 N개의 등간격 주파수에서 표본화                   │
│    X[k] = Σ x[n] W_N^(kn),  W_N = e^(-j2π/N)                  │
│                                                                  │
│  주요 성질:                                                      │
│    - 원형 합성곱 ↔ 점별 곱셈                                    │
│    - 파르스발: Σ|x[n]|² = (1/N)Σ|X[k]|²                       │
│    - 원형 이동 ↔ 위상 회전                                      │
│                                                                  │
│  분해능:                                                         │
│    Δf = fs/N  (신호 지속 시간에 의해 결정)                       │
│    영 패딩: 더 세밀한 격자, 분해능 향상 아님                     │
│                                                                  │
│  스펙트럼 누설:                                                  │
│    원인: 유한 신호 길이 (직사각형 창)                            │
│    해결: 창 함수 적용 (Hann, Hamming, Blackman 등)              │
│    트레이드오프: 더 넓은 주엽 vs. 낮은 부엽                      │
│                                                                  │
│  FFT (쿨리-튜키):                                               │
│    O(N²) → O(N log N)으로 감소                                  │
│    DIT: 짝수/홀수 분할 → 버터플라이 계산                         │
│    실용: numpy.fft.fft / rfft 사용                              │
│                                                                  │
│  실용 팁:                                                        │
│    - 실수 신호에는 rfft 사용 (2배 효율)                          │
│    - 스케일: 단측 진폭에 2|X[k]|/N                              │
│    - 누설 감소를 위해 FFT 전 창 함수 적용                        │
│    - FFT 기반 선형 합성곱을 위해 영 패딩                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. 연습 문제

### 연습 문제 1: 직접 DFT 계산

$x[n] = \{1, 0, -1, 0\}$의 4점 DFT를 정의를 사용하여 직접 계산하세요.

**(a)** $X[0], X[1], X[2], X[3]$에 대한 네 개의 합산을 전부 적으세요.

**(b)** `np.fft.fft`를 사용하여 답을 검증하세요.

**(c)** 역 DFT를 적용하여 $x[n]$을 복원하고 검증하세요.

### 연습 문제 2: 원형 합성곱

$x_1 = \{1, 2, 3, 4\}$, $x_2 = \{1, 0, 1, 0\}$가 주어졌을 때:

**(a)** 4점 원형 합성곱을 직접 계산하세요.

**(b)** DFT 곱셈 성질을 사용하여 검증하세요: $\text{IDFT}\{X_1[k] \cdot X_2[k]\}$.

**(c)** 선형 합성곱과 비교하세요. 원형 합성곱이 선형 합성곱과 같아지려면 영 패딩된 시퀀스가 얼마나 길어야 하나요?

### 연습 문제 3: 주파수 분해능 도전

$f_s = 1000$ Hz로 표본화된 $f_1 = 100$ Hz와 $f_2 = 103$ Hz의 두 정현파가 있습니다.

**(a)** 이 두 주파수를 분해하기 위한 최소 표본 수 $N$은 얼마인가요 ($\Delta f \leq 3$ Hz)?

**(b)** 신호를 생성하고 FFT를 적용하여 두 피크를 분해할 수 있음을 보이세요. $N = 256$과 $N = 512$를 모두 시도해보세요.

**(c)** Hann 창을 적용하고 반복하세요. 윈도잉이 두 톤의 분해에 도움이 되나요, 방해가 되나요? 왜인가요?

### 연습 문제 4: 쿨리-튜키 FFT 구현

**(a)** 기수-2 DIF FFT를 구현하세요 (이 레슨에서 보여준 DIT 구현과 보완).

**(b)** 블루스타인(Bluestein) 알고리즘 또는 다음 2의 제곱수로 영 패딩하여 임의 길이 지원을 추가하세요.

**(c)** $N = 2^{10}$부터 $2^{20}$까지 `np.fft.fft`와 구현을 벤치마크하세요.

### 연습 문제 5: 실제 오디오의 스펙트럼 분석

짧은 오디오 클립 (WAV 파일)을 녹음하거나 구하세요. Python을 사용하여:

**(a)** 오디오를 불러와 파형을 그리세요.

**(b)** Hann 창을 사용하여 크기 스펙트럼을 계산하고 그리세요.

**(c)** 겹치는 창으로 스펙트로그램을 만드세요. 시간에 따른 지배적인 주파수 성분을 식별하세요.

**(d)** 특정 주파수 대역을 제거하는 주파수 영역 필터링을 구현하고 결과를 들어보세요.

### 연습 문제 6: 윈도잉 실험

신호 $x[n] = \sin(2\pi \cdot 100n / f_s) + 0.01 \sin(2\pi \cdot 150n / f_s)$를 생성하세요. $f_s = 1000$ Hz, $N = 256$.

**(a)** 창 없이 (직사각형), 약한 150 Hz 성분을 볼 수 있나요? 누설 부엽의 피크 레벨은?

**(b)** Hann, Hamming, Blackman, 카이저($\beta = 12$) 창을 적용하세요. 어떤 창이 약한 성분을 가장 잘 드러내나요?

**(c)** 창 유형의 함수로 동적 범위 (강한 신호 근처에서 약한 신호를 감지하는 능력)를 그리세요.

### 연습 문제 7: 2D FFT 응용

**(a)** 회색조 이미지를 불러와 2D FFT를 계산하세요.

**(b)** 크기 스펙트럼을 표시하세요 (로그 스케일, 중앙 정렬).

**(c)** 고주파 성분을 영으로 만들고 FFT를 역변환하여 주파수 영역 저역통과 필터링을 구현하세요. 서로 다른 차단 반경에 대한 블러 결과를 보이세요.

**(d)** 주파수 영역에서 고역통과 필터링 (에지 검출)을 구현하세요.

---

## 12. 추가 읽기

- Oppenheim, Schafer. *Discrete-Time Signal Processing*, 3rd ed. Chapters 5, 8, 9.
- Proakis, Manolakis. *Digital Signal Processing*, 4th ed. Chapters 5, 7.
- Cooley, Tukey. "An Algorithm for the Machine Calculation of Complex Fourier Series," *Math. of Computation*, 1965.
- Smith, S. W. *The Scientist and Engineer's Guide to DSP*, Chapters 8-12.
- Lyons, R. G. *Understanding Digital Signal Processing*, 3rd ed. Chapters 3-4.

---

**이전**: [05. 표본화와 복원](05_Sampling_and_Reconstruction.md) | **다음**: [07. Z-변환](07_Z_Transform.md)
