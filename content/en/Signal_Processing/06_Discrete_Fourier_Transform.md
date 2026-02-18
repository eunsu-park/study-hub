# Discrete Fourier Transform

## Overview

The Discrete Fourier Transform (DFT) is the computational workhorse of digital signal processing. It converts a finite-length sequence of samples into a finite-length sequence of frequency components. Coupled with the Fast Fourier Transform (FFT) algorithm, the DFT enables efficient spectral analysis, filtering, correlation, and countless other applications. This lesson covers DFT theory, properties, the FFT algorithm, and practical usage.

**Learning Objectives:**
- Understand the relationship between DTFT, DFT, and FFT
- Apply DFT properties including circular convolution and Parseval's theorem
- Explain spectral leakage and windowing strategies
- Implement and use FFT for spectral analysis of real-world signals
- Understand the Cooley-Tukey algorithm and its computational advantages

**Prerequisites:** [05. Sampling and Reconstruction](05_Sampling_and_Reconstruction.md)

---

## 1. From DTFT to DFT

### 1.1 Discrete-Time Fourier Transform (DTFT) Review

For a discrete-time signal $x[n]$, the DTFT is:

$$X(e^{j\omega}) = \sum_{n=-\infty}^{\infty} x[n] \, e^{-j\omega n}$$

Key properties of the DTFT:
- **Input**: infinite-length discrete sequence
- **Output**: continuous, periodic function of $\omega$ (period $2\pi$)
- **Existence**: requires absolute summability or finite energy
- **Inverse**: $x[n] = \frac{1}{2\pi} \int_{-\pi}^{\pi} X(e^{j\omega}) \, e^{j\omega n} \, d\omega$

### 1.2 The Problem: DTFT is Continuous

Computers cannot store or compute with continuous functions. We need a **discrete** frequency representation. The DFT solves this by sampling the DTFT at $N$ equally spaced frequencies.

### 1.3 DFT as Sampling of the DTFT

For a finite-length signal $x[n]$ of length $N$, the DFT samples the DTFT at:

$$\omega_k = \frac{2\pi k}{N}, \quad k = 0, 1, \ldots, N-1$$

$$X[k] = X(e^{j\omega})\bigg|_{\omega = 2\pi k/N} = \sum_{n=0}^{N-1} x[n] \, e^{-j2\pi kn/N}$$

This means:
- The DFT provides $N$ frequency samples of the DTFT
- Each DFT bin $k$ corresponds to frequency $f_k = k \cdot f_s / N$
- The frequency resolution is $\Delta f = f_s / N$

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

## 2. DFT Definition and Inverse DFT

### 2.1 Forward DFT

The $N$-point DFT of a sequence $x[n]$, $n = 0, 1, \ldots, N-1$:

$$\boxed{X[k] = \sum_{n=0}^{N-1} x[n] \, W_N^{kn}, \quad k = 0, 1, \ldots, N-1}$$

where $W_N = e^{-j2\pi/N}$ is the **twiddle factor** (the $N$-th root of unity).

### 2.2 Inverse DFT (IDFT)

$$\boxed{x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \, W_N^{-kn}, \quad n = 0, 1, \ldots, N-1}$$

### 2.3 Matrix Form

The DFT can be written as a matrix-vector product:

$$\mathbf{X} = \mathbf{W}_N \mathbf{x}$$

where the DFT matrix is:

$$[\mathbf{W}_N]_{k,n} = W_N^{kn} = e^{-j2\pi kn/N}$$

For $N = 4$:

$$\mathbf{W}_4 = \begin{bmatrix} 1 & 1 & 1 & 1 \\ 1 & -j & -1 & j \\ 1 & -1 & 1 & -1 \\ 1 & j & -1 & -j \end{bmatrix}$$

The DFT matrix is **unitary** (up to scaling): $\mathbf{W}_N^{-1} = \frac{1}{N}\mathbf{W}_N^*$.

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

### 2.4 Physical Interpretation of DFT Bins

For a signal sampled at rate $f_s$, the $k$-th DFT bin corresponds to:

| Quantity | Expression |
|----------|-----------|
| Frequency (Hz) | $f_k = k \cdot f_s / N$ |
| Angular frequency | $\omega_k = 2\pi k / N$ |
| Frequency resolution | $\Delta f = f_s / N$ |
| Highest frequency (bin $N/2$) | $f_s / 2$ (Nyquist) |

For real signals, $X[k]$ and $X[N-k]$ are complex conjugates, so bins $k > N/2$ represent negative frequencies. The magnitude spectrum is symmetric about $N/2$.

---

## 3. DFT Properties

### 3.1 Linearity

$$\text{DFT}\{a \cdot x_1[n] + b \cdot x_2[n]\} = a \cdot X_1[k] + b \cdot X_2[k]$$

### 3.2 Circular (Cyclic) Shift

A circular shift by $m$ in time corresponds to a phase rotation in frequency:

$$x[(n-m) \bmod N] \quad \xleftrightarrow{\text{DFT}} \quad X[k] \cdot W_N^{mk} = X[k] \cdot e^{-j2\pi mk/N}$$

> Note: This is a **circular** shift, not a linear shift. The signal wraps around modulo $N$.

### 3.3 Circular Convolution

Linear convolution and circular convolution are fundamentally different.

**Circular convolution** of two $N$-point sequences:

$$(x_1 \circledast x_2)[n] = \sum_{m=0}^{N-1} x_1[m] \, x_2[(n-m) \bmod N]$$

The DFT multiplication property:

$$\boxed{x_1[n] \circledast x_2[n] \quad \xleftrightarrow{\text{DFT}} \quad X_1[k] \cdot X_2[k]}$$

To compute **linear** convolution using circular convolution (needed for overlap-add, filtering):
- Zero-pad both sequences to length $\geq N_1 + N_2 - 1$
- Compute circular convolution of the zero-padded sequences
- The result equals the linear convolution

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

Output:
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

### 3.4 Parseval's Theorem (Energy Conservation)

$$\boxed{\sum_{n=0}^{N-1} |x[n]|^2 = \frac{1}{N} \sum_{k=0}^{N-1} |X[k]|^2}$$

The total energy is preserved between time and frequency domains (up to the $1/N$ factor).

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

### 3.5 Summary of DFT Properties

| Property | Time Domain | Frequency Domain |
|----------|------------|-----------------|
| Linearity | $ax_1[n] + bx_2[n]$ | $aX_1[k] + bX_2[k]$ |
| Circular shift | $x[(n-m) \bmod N]$ | $W_N^{mk} X[k]$ |
| Modulation | $W_N^{-ln} x[n]$ | $X[(k-l) \bmod N]$ |
| Circular convolution | $x_1 \circledast x_2$ | $X_1[k] \cdot X_2[k]$ |
| Multiplication | $x_1[n] \cdot x_2[n]$ | $\frac{1}{N} X_1 \circledast X_2$ |
| Conjugation | $x^*[n]$ | $X^*[(-k) \bmod N]$ |
| Time reversal | $x[(-n) \bmod N]$ | $X[(-k) \bmod N]$ |
| Parseval's | $\sum |x[n]|^2$ | $\frac{1}{N}\sum |X[k]|^2$ |

---

## 4. Zero-Padding and Frequency Resolution

### 4.1 Frequency Resolution

The DFT provides $N$ frequency samples spaced $\Delta f = f_s / N$ apart. To increase the density of frequency samples (interpolate the DTFT), we can **zero-pad** the signal.

> **Important:** Zero-padding increases the number of DFT points (finer frequency grid) but does **not** increase the true spectral resolution. True resolution is determined by the signal duration $T = N \cdot T_s$.

### 4.2 Zero-Padding Demonstration

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

### 4.3 True Resolution vs. Zero-Padding

| Concept | Determined by | Effect of zero-padding |
|---------|--------------|----------------------|
| **Frequency resolution** ($\Delta f$) | Signal duration $T$ | No change |
| **DFT bin spacing** | NFFT = $N$ + zero-padding | Decreases (finer grid) |
| **Ability to resolve two tones** | $\Delta f \approx 1/T$ | Does not improve |
| **Visual appearance** | Both | Smoother spectrum |

---

## 5. Spectral Leakage and Windowing

### 5.1 The Leakage Problem

When we compute the DFT of a finite-length signal, we implicitly assume the signal is periodic with period $N$. If the signal frequency does not fall exactly on a DFT bin, the energy "leaks" into adjacent bins.

**Cause:** Truncating an infinite signal to $N$ samples is equivalent to multiplying by a rectangular window. In the frequency domain, this convolves the true spectrum with the sinc-like Fourier transform of the rectangular window.

### 5.2 Window Functions

To reduce spectral leakage, we multiply the signal by a **window function** $w[n]$ before computing the DFT:

$$y[n] = x[n] \cdot w[n]$$

Common window functions:

| Window | Main Lobe Width | Side Lobe Level | Use Case |
|--------|----------------|-----------------|----------|
| Rectangular | Narrowest ($2\pi/N$) | -13 dB | Maximum resolution |
| Hann (Hanning) | $4\pi/N$ | -31 dB | General purpose |
| Hamming | $4\pi/N$ | -42 dB | Speech processing |
| Blackman | $6\pi/N$ | -58 dB | High dynamic range |
| Kaiser ($\beta$) | Variable | Variable | Adjustable trade-off |
| Flat-top | Wide | Very low | Amplitude accuracy |

**Trade-off:** Wider main lobe = worse frequency resolution, but lower side lobes = less leakage.

### 5.3 Window Comparison

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

### 5.4 Spectral Leakage in Action

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

## 6. Fast Fourier Transform (FFT)

### 6.1 The Computational Problem

Direct computation of the DFT requires:
- $N$ complex multiplications and $N-1$ complex additions for each of $N$ output values
- Total: $O(N^2)$ operations

For $N = 10^6$: $\sim 10^{12}$ operations (impractical for real-time processing).

### 6.2 The Cooley-Tukey Algorithm

The FFT exploits the symmetry and periodicity of the twiddle factors $W_N^{kn}$.

**Key observations:**
1. $W_N^{k+N} = W_N^k$ (periodicity)
2. $W_N^{k+N/2} = -W_N^k$ (symmetry / half-period negation)
3. $W_N^{2kn} = W_{N/2}^{kn}$ (reduction)

**Radix-2 Decimation-in-Time (DIT):**

Split $x[n]$ into even-indexed and odd-indexed subsequences:

$$X[k] = \underbrace{\sum_{r=0}^{N/2-1} x[2r] \, W_{N/2}^{kr}}_{G[k] \text{ (even terms)}} + W_N^k \underbrace{\sum_{r=0}^{N/2-1} x[2r+1] \, W_{N/2}^{kr}}_{H[k] \text{ (odd terms)}}$$

This gives the **butterfly** computation:

$$X[k] = G[k] + W_N^k \cdot H[k]$$
$$X[k + N/2] = G[k] - W_N^k \cdot H[k]$$

for $k = 0, 1, \ldots, N/2 - 1$.

Each stage halves the problem size, giving $\log_2 N$ stages of $N/2$ butterflies each.

**Complexity:** $O(N \log_2 N)$

| $N$ | DFT ($N^2$) | FFT ($N \log_2 N$) | Speedup |
|-----|------------|-------------------|---------|
| 64 | 4,096 | 384 | 10.7x |
| 256 | 65,536 | 2,048 | 32x |
| 1,024 | 1,048,576 | 10,240 | 102x |
| 65,536 | $4.3 \times 10^9$ | $1.0 \times 10^6$ | 4,096x |
| 1,048,576 | $1.1 \times 10^{12}$ | $2.1 \times 10^7$ | 52,429x |

### 6.3 Butterfly Diagram

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

### 6.4 Radix-2 DIT Implementation

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

### 6.5 Radix-2 Decimation-in-Frequency (DIF)

The DIF variant splits the output rather than the input:

$$X[2r] = \sum_{n=0}^{N/2-1} (x[n] + x[n + N/2]) \, W_{N/2}^{rn}$$

$$X[2r+1] = \sum_{n=0}^{N/2-1} (x[n] - x[n + N/2]) \, W_N^n \, W_{N/2}^{rn}$$

DIF butterflies subtract first, then multiply by twiddle factors (opposite order to DIT).

### 6.6 FFT Complexity Comparison

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

## 7. Practical FFT Usage

### 7.1 Using numpy.fft

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

### 7.2 Important FFT Functions in NumPy

| Function | Description |
|----------|-----------|
| `np.fft.fft(x, n)` | N-point FFT of complex/real signal |
| `np.fft.ifft(X, n)` | Inverse FFT |
| `np.fft.rfft(x, n)` | FFT for real input (positive freqs only) |
| `np.fft.irfft(X, n)` | Inverse FFT for real output |
| `np.fft.fftfreq(n, d)` | Frequency array for `fft` output |
| `np.fft.rfftfreq(n, d)` | Frequency array for `rfft` output |
| `np.fft.fftshift(X)` | Shift zero-frequency to center |
| `np.fft.ifftshift(X)` | Undo `fftshift` |

### 7.3 Common Pitfalls

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

## 8. Spectral Analysis Applications

### 8.1 Analyzing a Musical Chord

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

### 8.2 Short-Time Fourier Transform (STFT) / Spectrogram

For non-stationary signals, the STFT divides the signal into overlapping segments, windows each, and computes the FFT:

$$\text{STFT}\{x[n]\}(m, k) = \sum_{n=0}^{L-1} x[n + mH] \, w[n] \, e^{-j2\pi kn/N_{\text{FFT}}}$$

where $L$ is window length, $H$ is hop size, and $N_{\text{FFT}}$ is the FFT size.

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

### 8.3 FFT-Based Filtering (Overlap-Add Method)

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

## 9. Two-Dimensional DFT

### 9.1 Definition

The 2D DFT extends naturally to images and spatial data:

$$X[k_1, k_2] = \sum_{n_1=0}^{N_1-1} \sum_{n_2=0}^{N_2-1} x[n_1, n_2] \, e^{-j2\pi(k_1 n_1/N_1 + k_2 n_2/N_2)}$$

The 2D FFT can be computed as 1D FFTs along each dimension (separability).

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

## 10. Summary

```
┌─────────────────────────────────────────────────────────────────┐
│               Discrete Fourier Transform (DFT)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  DTFT → DFT:                                                    │
│    DFT samples the DTFT at N equally spaced frequencies         │
│    X[k] = Σ x[n] W_N^(kn),  W_N = e^(-j2π/N)                  │
│                                                                  │
│  Key Properties:                                                 │
│    - Circular convolution ↔ Pointwise multiplication            │
│    - Parseval: Σ|x[n]|² = (1/N)Σ|X[k]|²                       │
│    - Circular shift ↔ Phase rotation                            │
│                                                                  │
│  Resolution:                                                     │
│    Δf = fs/N  (determined by signal duration)                   │
│    Zero-padding: finer grid, NOT finer resolution               │
│                                                                  │
│  Spectral Leakage:                                               │
│    Cause: finite signal length (rectangular window)             │
│    Fix: apply window function (Hann, Hamming, Blackman, etc.)   │
│    Trade-off: wider main lobe vs. lower side lobes              │
│                                                                  │
│  FFT (Cooley-Tukey):                                             │
│    Reduces O(N²) → O(N log N)                                   │
│    DIT: split even/odd → butterfly computation                  │
│    Practical: use numpy.fft.fft / rfft                          │
│                                                                  │
│  Practical Tips:                                                 │
│    - Use rfft for real signals (2x efficiency)                  │
│    - Scale: 2|X[k]|/N for single-sided amplitude               │
│    - Window before FFT to reduce leakage                        │
│    - Zero-pad for FFT-based linear convolution                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. Exercises

### Exercise 1: DFT by Hand

Compute the 4-point DFT of $x[n] = \{1, 0, -1, 0\}$ by hand using the definition.

**(a)** Write out all four summations for $X[0], X[1], X[2], X[3]$.

**(b)** Verify your answer using `np.fft.fft`.

**(c)** Apply the inverse DFT to recover $x[n]$ and verify.

### Exercise 2: Circular Convolution

Given $x_1 = \{1, 2, 3, 4\}$ and $x_2 = \{1, 0, 1, 0\}$:

**(a)** Compute the 4-point circular convolution by hand.

**(b)** Verify using the DFT multiplication property: $\text{IDFT}\{X_1[k] \cdot X_2[k]\}$.

**(c)** Compare with linear convolution. How long must the zero-padded sequences be for the circular convolution to equal the linear convolution?

### Exercise 3: Frequency Resolution Challenge

Two sinusoids at frequencies $f_1 = 100$ Hz and $f_2 = 103$ Hz are sampled at $f_s = 1000$ Hz.

**(a)** What is the minimum number of samples $N$ needed to resolve these two frequencies (i.e., $\Delta f \leq 3$ Hz)?

**(b)** Generate the signal, apply an FFT, and show that you can resolve the two peaks. Try both $N = 256$ and $N = 512$ samples.

**(c)** Apply a Hann window and repeat. Does windowing help or hurt the ability to resolve the two tones? Why?

### Exercise 4: Implement the Cooley-Tukey FFT

**(a)** Implement a radix-2 DIF FFT (complement the DIT implementation shown in this lesson).

**(b)** Add support for arbitrary lengths using the Bluestein algorithm or zero-padding to the next power of 2.

**(c)** Benchmark your implementation against `np.fft.fft` for $N = 2^{10}$ through $2^{20}$.

### Exercise 5: Spectral Analysis of Real Audio

Record or obtain a short audio clip (WAV file). Using Python:

**(a)** Load the audio and plot its waveform.

**(b)** Compute and plot the magnitude spectrum using a Hann window.

**(c)** Create a spectrogram with overlapping windows. Identify the dominant frequency components over time.

**(d)** Implement frequency-domain filtering to remove a specific frequency band and listen to the result.

### Exercise 6: Windowing Experiment

Generate a signal with two sinusoids: $x[n] = \sin(2\pi \cdot 100n / f_s) + 0.01 \sin(2\pi \cdot 150n / f_s)$ where $f_s = 1000$ Hz, $N = 256$.

**(a)** With no window (rectangular), can you see the weak 150 Hz component? What is the peak level of the leakage side lobes?

**(b)** Apply Hann, Hamming, Blackman, and Kaiser ($\beta = 12$) windows. Which window reveals the weak component best?

**(c)** Plot the dynamic range (ability to detect weak signals near strong ones) as a function of window type.

### Exercise 7: 2D FFT Application

**(a)** Load a grayscale image and compute its 2D FFT.

**(b)** Display the magnitude spectrum (log-scaled, centered).

**(c)** Implement frequency-domain lowpass filtering by zeroing out high-frequency components and inverting the FFT. Show the blurred result for different cutoff radii.

**(d)** Implement highpass filtering (edge detection) in the frequency domain.

---

## 12. Further Reading

- Oppenheim, Schafer. *Discrete-Time Signal Processing*, 3rd ed. Chapters 5, 8, 9.
- Proakis, Manolakis. *Digital Signal Processing*, 4th ed. Chapters 5, 7.
- Cooley, Tukey. "An Algorithm for the Machine Calculation of Complex Fourier Series," *Math. of Computation*, 1965.
- Smith, S. W. *The Scientist and Engineer's Guide to DSP*, Chapters 8-12.
- Lyons, R. G. *Understanding Digital Signal Processing*, 3rd ed. Chapters 3-4.

---

**Previous**: [05. Sampling and Reconstruction](05_Sampling_and_Reconstruction.md) | **Next**: [07. Z-Transform](07_Z_Transform.md)
