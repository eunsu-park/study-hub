# Multirate Signal Processing

## Learning Objectives

- Understand the fundamentals of decimation (downsampling) and interpolation (upsampling)
- Analyze the spectral effects of sample rate changes
- Master the Noble identities for efficient multirate implementations
- Design anti-aliasing and anti-imaging filters for rate conversion
- Implement polyphase decomposition for computationally efficient filtering
- Apply rational rate conversion using cascaded interpolation and decimation
- Understand filter bank structures including Quadrature Mirror Filters (QMF)
- Implement multirate systems using Python's `scipy.signal` module

---

## Table of Contents

1. [Introduction to Multirate Systems](#1-introduction-to-multirate-systems)
2. [Downsampling (Decimation)](#2-downsampling-decimation)
3. [Upsampling (Interpolation)](#3-upsampling-interpolation)
4. [Decimation and Interpolation Filters](#4-decimation-and-interpolation-filters)
5. [Noble Identities](#5-noble-identities)
6. [Polyphase Decomposition](#6-polyphase-decomposition)
7. [Rational Rate Conversion](#7-rational-rate-conversion)
8. [Multistage Rate Conversion](#8-multistage-rate-conversion)
9. [Filter Banks](#9-filter-banks)
10. [Applications](#10-applications)
11. [Python Implementation](#11-python-implementation)
12. [Exercises](#12-exercises)

---

## 1. Introduction to Multirate Systems

### 1.1 Why Multirate Processing?

Multirate signal processing involves systems that operate at more than one sampling rate. Common motivations include:

- **Sample rate conversion**: Converting audio between 44.1 kHz (CD) and 48 kHz (professional audio)
- **Computational efficiency**: Processing at the lowest rate necessary for each stage
- **Bandwidth reduction**: Decimating narrowband signals to reduce data rates
- **Subband coding**: Splitting signals into frequency bands for efficient compression
- **Sigma-delta ADC/DAC**: Oversampled converters that trade speed for resolution

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

### 1.2 Basic Operations

The two fundamental operations in multirate processing:

**Downsampling by factor $M$** (keep every $M$-th sample):

$$y[n] = x[nM]$$

**Upsampling by factor $L$** (insert $L-1$ zeros between samples):

$$y[n] = \begin{cases} x[n/L], & n = 0, \pm L, \pm 2L, \ldots \\ 0, & \text{otherwise} \end{cases}$$

---

## 2. Downsampling (Decimation)

### 2.1 Time-Domain Operation

Downsampling by integer factor $M$ retains every $M$-th sample:

$$x_d[n] = x[nM]$$

The output sampling rate is $f_s' = f_s / M$.

### 2.2 Frequency-Domain Analysis

The DTFT of the downsampled signal is:

$$X_d(e^{j\omega}) = \frac{1}{M} \sum_{k=0}^{M-1} X\left(e^{j(\omega - 2\pi k)/M}\right)$$

This formula reveals two effects:
1. **Frequency scaling**: The spectrum is stretched by factor $M$ (frequencies compressed into $[-\pi, \pi]$)
2. **Aliasing**: $M$ shifted copies of the spectrum are superimposed

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

### 2.3 Aliasing Demonstration

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

## 3. Upsampling (Interpolation)

### 3.1 Time-Domain Operation

Upsampling by factor $L$ inserts $L-1$ zeros between consecutive samples:

$$x_u[n] = \begin{cases} x[n/L], & n = 0, \pm L, \pm 2L, \ldots \\ 0, & \text{otherwise} \end{cases}$$

The output sampling rate is $f_s' = L \cdot f_s$.

### 3.2 Frequency-Domain Analysis

The DTFT of the upsampled signal is:

$$X_u(e^{j\omega}) = X(e^{j\omega L})$$

This compresses the spectrum by factor $L$, creating $L-1$ spectral images (replicas) in the range $[0, 2\pi]$.

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

### 3.3 Imaging Demonstration

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

## 4. Decimation and Interpolation Filters

### 4.1 Decimation Filter Requirements

Before downsampling by $M$, an **anti-aliasing filter** must limit the signal bandwidth to prevent aliasing:

$$H_\text{dec}(e^{j\omega}) = \begin{cases} 1, & |\omega| < \pi/M \\ 0, & \pi/M \leq |\omega| \leq \pi \end{cases}$$

The complete decimator:

$$y[n] = \left[\sum_{k} h[k] \cdot x[nM - k]\right]$$

Note: We filter first, then downsample. The filter operates at the **high** sampling rate.

### 4.2 Interpolation Filter Requirements

After upsampling by $L$, an **anti-imaging filter** removes the spectral images:

$$H_\text{interp}(e^{j\omega}) = \begin{cases} L, & |\omega| < \pi/L \\ 0, & \pi/L \leq |\omega| \leq \pi \end{cases}$$

The gain of $L$ compensates for the amplitude reduction caused by zero insertion.

### 4.3 Filter Design for Decimation

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

## 5. Noble Identities

### 5.1 The Identities

The Noble identities allow moving filters across rate-changing operations, which is crucial for efficient implementations:

**Identity 1** (downsampler):

$$\boxed{H(z^M) \rightarrow \downarrow M \equiv \downarrow M \rightarrow H(z)}$$

A filter $H(z^M)$ followed by a downsampler by $M$ is equivalent to first downsampling, then filtering with $H(z)$.

**Identity 2** (upsampler):

$$\boxed{\uparrow L \rightarrow H(z^L) \equiv H(z) \rightarrow \uparrow L}$$

An upsampler by $L$ followed by a filter $H(z^L)$ is equivalent to first filtering with $H(z)$, then upsampling.

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

### 5.2 Why Noble Identities Matter

- **Computational savings**: Moving the filter to the lower-rate side reduces the number of multiply-accumulate operations by a factor of $M$ (or $L$)
- **Foundation for polyphase**: The identities enable polyphase decomposition
- **Caution**: The identities only work for $H(z^M)$ or $H(z^L)$, not arbitrary $H(z)$

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

## 6. Polyphase Decomposition

### 6.1 Concept

The **polyphase decomposition** splits a filter into $M$ (or $L$) subfilters, each operating at the lower sampling rate. This is the most efficient implementation of multirate filtering.

For a filter $H(z) = \sum_{n=0}^{N-1} h[n] z^{-n}$, the Type I polyphase decomposition is:

$$H(z) = \sum_{k=0}^{M-1} z^{-k} E_k(z^M)$$

where each polyphase component is:

$$E_k(z) = \sum_{n=0}^{\lfloor (N-1)/M \rfloor} h[nM + k] z^{-n}, \quad k = 0, 1, \ldots, M-1$$

### 6.2 Polyphase Decimation

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

### 6.3 Polyphase Interpolation

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

### 6.4 Implementation

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

## 7. Rational Rate Conversion

### 7.1 Rate Change by L/M

To convert the sampling rate by a rational factor $L/M$:

1. **Upsample** by $L$ (insert $L-1$ zeros)
2. **Filter** with a lowpass filter at cutoff $\omega_c = \min(\pi/L, \pi/M)$
3. **Downsample** by $M$ (keep every $M$-th sample)

The order of operations is critical: **interpolate first, then decimate**.

```
Rational Rate Conversion (L/M):

x[n] ──▶ ↑L ──▶ H(z) ──▶ ↓M ──▶ y[n]
  fs              cutoff = min(π/L, π/M)       fs' = fs × L/M
```

### 7.2 Example: 44.1 kHz to 48 kHz Conversion

Converting CD audio (44.1 kHz) to professional audio (48 kHz):

$$\frac{48000}{44100} = \frac{480}{441} = \frac{160}{147}$$

So $L = 160$ and $M = 147$.

### 7.3 Implementation

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

## 8. Multistage Rate Conversion

### 8.1 Motivation

For large rate-change factors (e.g., $M = 100$), a single-stage decimator requires a very sharp (high-order) anti-aliasing filter. **Multistage decimation** cascades multiple smaller decimation stages, reducing the total computational cost.

If $M = M_1 \cdot M_2 \cdot \ldots \cdot M_K$, each stage uses a simpler filter.

```
Single-stage: x ──▶ H(z) ──▶ ↓100 ──▶ y
              Very sharp filter, high order

Multistage:   x ──▶ H₁(z) ──▶ ↓10 ──▶ H₂(z) ──▶ ↓10 ──▶ y
              Two moderate filters, lower total order
```

### 8.2 Optimal Stage Allocation

The optimal factorization of $M$ depends on the specifications. A common heuristic is to distribute the factor as evenly as possible.

For $M = M_1 \cdot M_2$:
- Stage 1 filter: cutoff at $\pi / M$ at rate $f_s$
- Stage 2 filter: cutoff at $\pi / M_2$ at rate $f_s / M_1$

### 8.3 Implementation and Comparison

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

## 9. Filter Banks

### 9.1 Analysis-Synthesis Filter Bank

A **filter bank** splits a signal into multiple frequency subbands and can reconstruct it. This is the foundation of audio coding (MP3, AAC), image compression (JPEG2000), and many other applications.

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

For **perfect reconstruction** (PR):

$$\hat{X}(z) = X(z) \quad \text{(with some delay)}$$

### 9.2 Two-Channel Filter Bank

The simplest case: $M = 2$ with lowpass ($H_0$) and highpass ($H_1$) filters.

**Conditions for perfect reconstruction (two-channel):**

1. **No aliasing**: $H_0(-z)G_0(z) + H_1(-z)G_1(z) = 0$
2. **No distortion**: $H_0(z)G_0(z) + H_1(z)G_1(z) = 2z^{-d}$ (pure delay)

### 9.3 Quadrature Mirror Filters (QMF)

In a QMF bank, the highpass filter is obtained by modulating the lowpass filter:

$$H_1(z) = H_0(-z) \quad \Rightarrow \quad h_1[n] = (-1)^n h_0[n]$$

This automatically satisfies the alias cancellation condition. For the synthesis filters:

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

### 9.4 Conjugate Quadrature Filters (CQF)

For perfect reconstruction in two-channel systems, the **CQF** (also called Johnston filters) satisfy:

$$|H_0(e^{j\omega})|^2 + |H_0(e^{j(\omega - \pi)})|^2 = 1$$

This is the **power complementary** condition, which leads to exact perfect reconstruction.

---

## 10. Applications

### 10.1 Audio Sample Rate Conversion

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

### 10.2 Sigma-Delta ADC Concept

A sigma-delta ($\Sigma\Delta$) ADC uses oversampling and noise shaping:

1. **Oversample** at rate $f_s = R \cdot f_\text{Nyquist}$ (where $R$ is the oversampling ratio, e.g., 64 or 256)
2. **Noise shaping** pushes quantization noise to high frequencies
3. **Decimate** using a multirate chain to get the final output at the Nyquist rate

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

## 11. Python Implementation

### 11.1 Using scipy.signal for Multirate Processing

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

### 11.2 Complete Multirate Processing Pipeline

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

## 12. Exercises

### Exercise 1: Downsampling Analysis

Consider the signal $x[n] = \cos(0.2\pi n) + \cos(0.7\pi n)$ sampled at $f_s = 10$ kHz.

(a) Sketch the DTFT of $x[n]$.

(b) Downsample by $M = 2$. Determine analytically which frequency components alias and what the resulting spectrum looks like.

(c) Downsample by $M = 4$. Does aliasing occur? Which components overlap?

(d) Implement in Python: plot the spectra of the original signal and both downsampled versions. Verify your analytical predictions.

(e) Design an anti-aliasing filter that allows $M = 4$ decimation without distorting the $0.2\pi$ component.

### Exercise 2: Interpolation Quality

Starting with a 1 kHz sine wave sampled at 8 kHz:

(a) Upsample by $L = 4$ using zero-insertion only. Plot the spectrum and identify the imaging artifacts.

(b) Apply interpolation using three different methods:
   - Linear interpolation
   - FIR lowpass filter (32 taps)
   - FIR lowpass filter (128 taps)

(c) Compare the interpolation quality by computing the SNR of each method relative to a high-quality reference (e.g., `scipy.signal.resample`).

(d) Plot the time-domain waveforms of all methods overlaid on the ideal continuous sinusoid.

### Exercise 3: Polyphase Implementation

Given a 128-tap FIR filter and decimation factor $M = 8$:

(a) Implement the polyphase decomposition manually. How many subfilters are there? What is the length of each?

(b) Implement a `PolyphaseDecimator` class that processes the signal block-by-block (simulate streaming).

(c) Verify the output matches `scipy.signal.decimate`.

(d) Benchmark the standard vs polyphase implementations. Measure the speedup factor.

### Exercise 4: Rational Rate Conversion

Convert a signal from 11025 Hz to 8000 Hz.

(a) Find the simplest $L/M$ ratio. Hint: $8000/11025 = ?$

(b) Design the required anti-aliasing/anti-imaging filter.

(c) Implement the conversion using `signal.resample_poly`.

(d) Generate a test signal containing tones at 500, 1000, 2000, and 3500 Hz. After conversion, which tones should survive and which should be attenuated? Verify with spectral analysis.

### Exercise 5: Multistage Decimation Optimization

You need to decimate by $M = 48$.

(a) List all possible factorizations of 48 into 2-4 stages (e.g., $48 = 2 \times 24$, $48 = 2 \times 4 \times 6$, etc.).

(b) For each factorization, estimate the total number of multiply-accumulate operations per output sample. Assume the filter order for each stage is $10 \times M_\text{stage}$.

(c) Which factorization is optimal? Implement it and verify the output quality.

(d) Compare with single-stage decimation in terms of computation time and output SNR.

### Exercise 6: QMF Filter Bank

Design a two-channel QMF filter bank for audio processing:

(a) Design a 32-tap lowpass prototype filter suitable for QMF.

(b) Construct the highpass analysis filter and both synthesis filters.

(c) Process a music-like signal (sum of harmonics) and verify near-perfect reconstruction.

(d) Compute the reconstruction error as a function of the filter order (try 8, 16, 32, 64, 128 taps).

(e) Implement subband coding: quantize the lowpass subband to 16 bits and the highpass subband to 8 bits. Reconstruct and measure the SNR.

### Exercise 7: CIC Filter

The **Cascaded Integrator-Comb (CIC)** filter is a computationally efficient decimation filter that requires no multipliers.

(a) Implement a single-stage CIC filter (integrator + comb) for decimation by $M$.

(b) Derive the frequency response of the CIC filter: $H(z) = \frac{1}{M}\frac{1 - z^{-M}}{1 - z^{-1}}$.

(c) Plot the magnitude response for $M = 8, 16, 32$ and compare with an ideal lowpass.

(d) Implement a multi-stage CIC filter ($K$ cascaded stages) and show how passband droop and stopband attenuation improve with $K$.

(e) Design a CIC-based decimation chain for a sigma-delta ADC: oversample at 64x, then CIC decimate to the final rate.

---

## References

1. **Vaidyanathan, P. P. (1993).** *Multirate Systems and Filter Banks*. Prentice Hall. [The definitive reference]
2. **Oppenheim, A. V., & Schafer, R. W. (2010).** *Discrete-Time Signal Processing* (3rd ed.). Pearson. Chapter 4.
3. **Crochiere, R. E., & Rabiner, L. R. (1983).** *Multirate Digital Signal Processing*. Prentice Hall.
4. **Lyons, R. G. (2010).** *Understanding Digital Signal Processing* (3rd ed.). Pearson. Chapter 10.
5. **Fliege, N. J. (1994).** *Multirate Digital Signal Processing*. Wiley.
6. **SciPy Documentation** -- `scipy.signal.resample_poly`, `scipy.signal.decimate`: https://docs.scipy.org/doc/scipy/reference/signal.html

---

## Navigation

- Previous: [10. IIR Filter Design](10_IIR_Filter_Design.md)
- Next: [12. Spectral Analysis](12_Spectral_Analysis.md)
- [Back to Overview](00_Overview.md)
