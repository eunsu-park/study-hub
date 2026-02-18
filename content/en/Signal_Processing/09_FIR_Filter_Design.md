# FIR Filter Design

## Learning Objectives

- Understand the specifications and terminology of digital filter design
- Master the window method for FIR filter design and compare window functions
- Learn frequency sampling and optimal equiripple (Parks-McClellan) design methods
- Understand linear phase FIR filter types and their constraints
- Design FIR filters using Python's `scipy.signal` module
- Implement FIR filtering using direct convolution, overlap-add, and overlap-save methods
- Compare design methods in terms of transition width, stopband attenuation, and computational cost

---

## Table of Contents

1. [FIR Filter Design Specifications](#1-fir-filter-design-specifications)
2. [Ideal Filters and Impulse Responses](#2-ideal-filters-and-impulse-responses)
3. [Window Method](#3-window-method)
4. [Window Functions](#4-window-functions)
5. [Kaiser Window Design](#5-kaiser-window-design)
6. [Frequency Sampling Method](#6-frequency-sampling-method)
7. [Optimal Equiripple Design: Parks-McClellan Algorithm](#7-optimal-equiripple-design-parks-mcclellan-algorithm)
8. [Linear Phase FIR Filters](#8-linear-phase-fir-filters)
9. [Comparison of Design Methods](#9-comparison-of-design-methods)
10. [FIR Filter Implementation](#10-fir-filter-implementation)
11. [Python Implementation](#11-python-implementation)
12. [Exercises](#12-exercises)

---

## 1. FIR Filter Design Specifications

### 1.1 Filter Terminology

A digital filter is characterized by its **frequency response** $H(e^{j\omega})$, which specifies how different frequency components of a signal are modified. The key specifications include:

**Frequency bands:**
- **Passband**: frequencies that should pass through with minimal distortion, $0 \leq \omega \leq \omega_p$
- **Stopband**: frequencies that should be attenuated, $\omega_s \leq \omega \leq \pi$
- **Transition band**: the region between passband and stopband, $\omega_p < \omega < \omega_s$

**Tolerances:**
- **Passband ripple** $\delta_1$: maximum deviation from unity in the passband
- **Stopband attenuation** $\delta_2$: maximum magnitude in the stopband

```
Magnitude Response Specification
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

         Passband │Trans.│ Stopband
                  │ band │
```

### 1.2 Decibel Specifications

In practice, filter specifications are often given in decibels:

$$A_p = -20\log_{10}(1 - \delta_1) \quad \text{(passband ripple in dB)}$$

$$A_s = -20\log_{10}(\delta_2) \quad \text{(stopband attenuation in dB)}$$

Common specification examples:
- Passband ripple: $A_p \leq 0.1$ dB (implies $\delta_1 \approx 0.0115$)
- Stopband attenuation: $A_s \geq 60$ dB (implies $\delta_2 = 0.001$)

### 1.3 FIR vs IIR Trade-offs

```
┌────────────────────────────────────────────────────────┐
│              FIR vs IIR Comparison                      │
├──────────────────┬─────────────────┬───────────────────┤
│ Property         │ FIR             │ IIR               │
├──────────────────┼─────────────────┼───────────────────┤
│ Stability        │ Always stable   │ Can be unstable   │
│ Linear phase     │ Easily achieved │ Not possible      │
│ Filter order     │ Higher          │ Lower             │
│ Computation      │ More multiplies │ Fewer multiplies  │
│ Design methods   │ Window, PM, FS  │ Analog prototypes │
│ Group delay      │ Constant        │ Non-constant      │
│ Round-off errors │ Less sensitive  │ More sensitive    │
│ Implementation   │ No feedback     │ Feedback required │
└──────────────────┴─────────────────┴───────────────────┘
```

An FIR filter of order $M$ has the transfer function:

$$H(z) = \sum_{n=0}^{M} h[n] z^{-n}$$

The output is a finite convolution:

$$y[n] = \sum_{k=0}^{M} h[k] x[n-k]$$

---

## 2. Ideal Filters and Impulse Responses

### 2.1 Ideal Lowpass Filter

The ideal lowpass filter has the frequency response:

$$H_d(e^{j\omega}) = \begin{cases} 1, & |\omega| \leq \omega_c \\ 0, & \omega_c < |\omega| \leq \pi \end{cases}$$

Its impulse response is obtained via the inverse DTFT:

$$h_d[n] = \frac{1}{2\pi} \int_{-\pi}^{\pi} H_d(e^{j\omega}) e^{j\omega n} d\omega = \frac{\sin(\omega_c n)}{\pi n}$$

This is a **sinc function** that is:
- Infinite in duration (non-causal, unrealizable)
- Symmetric about $n = 0$

### 2.2 Other Ideal Filter Types

**Ideal highpass filter:**

$$h_d[n] = \delta[n] - \frac{\sin(\omega_c n)}{\pi n}$$

**Ideal bandpass filter** ($\omega_{c1} < |\omega| < \omega_{c2}$):

$$h_d[n] = \frac{\sin(\omega_{c2} n)}{\pi n} - \frac{\sin(\omega_{c1} n)}{\pi n}$$

**Ideal bandstop filter:**

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

## 3. Window Method

### 3.1 Basic Principle

The window method is the simplest FIR design approach. The idea is to:

1. Start with the ideal (infinite) impulse response $h_d[n]$
2. Multiply by a finite-duration window $w[n]$ of length $N = M + 1$
3. Shift the result to make the filter causal

$$h[n] = h_d[n - M/2] \cdot w[n], \quad n = 0, 1, \ldots, M$$

In the frequency domain, this multiplication corresponds to convolution:

$$H(e^{j\omega}) = \frac{1}{2\pi} H_d(e^{j\omega}) * W(e^{j\omega})$$

### 3.2 Effect of Windowing

The window's frequency response $W(e^{j\omega})$ determines:

1. **Mainlobe width**: Controls the transition bandwidth -- wider mainlobe means wider transition band
2. **Sidelobe level**: Determines stopband attenuation -- higher sidelobes mean less stopband attenuation
3. **Sidelobe decay rate**: How quickly sidelobes decrease with frequency

```
Window Spectrum Properties
|W(e^jω)|
    ↑
    │   ╱╲        Mainlobe
    │  ╱  ╲       width = Δω_ML
    │ ╱    ╲
    │╱      ╲  ╱╲
    │        ╲╱  ╲  ╱╲       Peak sidelobe
    │         │   ╲╱  ╲      level (dB)
    │         │    │   ╲╱╲
    ├─────────┼────┼─────┼──→ ω
    0      Δω_ML/2          π
```

### 3.3 Gibbs Phenomenon

When truncating the ideal impulse response with a rectangular window, the Gibbs phenomenon causes approximately 9% overshoot (about $-21$ dB stopband attenuation) at the band edge, regardless of the filter length. Increasing the filter order $M$ only narrows the transition width but does not reduce the ripple amplitude.

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

## 4. Window Functions

### 4.1 Common Windows

Each window function trades off mainlobe width against sidelobe level. Smoother windows have wider mainlobes but lower sidelobes.

#### Rectangular Window

$$w[n] = 1, \quad 0 \leq n \leq M$$

- **Mainlobe width**: $\Delta\omega_\text{ML} = 4\pi / (M+1)$
- **Peak sidelobe**: $-13$ dB
- **Minimum stopband attenuation**: $-21$ dB
- **Transition width**: $\Delta\omega \approx 0.92 \cdot 2\pi / (M+1)$

#### Hann (Hanning) Window

$$w[n] = 0.5 - 0.5\cos\left(\frac{2\pi n}{M}\right), \quad 0 \leq n \leq M$$

- **Mainlobe width**: $\Delta\omega_\text{ML} = 8\pi / (M+1)$
- **Peak sidelobe**: $-31$ dB
- **Minimum stopband attenuation**: $-44$ dB
- **Transition width**: $\Delta\omega \approx 3.11 \cdot 2\pi / (M+1)$

#### Hamming Window

$$w[n] = 0.54 - 0.46\cos\left(\frac{2\pi n}{M}\right), \quad 0 \leq n \leq M$$

- **Mainlobe width**: $\Delta\omega_\text{ML} = 8\pi / (M+1)$
- **Peak sidelobe**: $-41$ dB
- **Minimum stopband attenuation**: $-53$ dB
- **Transition width**: $\Delta\omega \approx 3.32 \cdot 2\pi / (M+1)$

#### Blackman Window

$$w[n] = 0.42 - 0.5\cos\left(\frac{2\pi n}{M}\right) + 0.08\cos\left(\frac{4\pi n}{M}\right), \quad 0 \leq n \leq M$$

- **Mainlobe width**: $\Delta\omega_\text{ML} = 12\pi / (M+1)$
- **Peak sidelobe**: $-57$ dB
- **Minimum stopband attenuation**: $-74$ dB
- **Transition width**: $\Delta\omega \approx 5.56 \cdot 2\pi / (M+1)$

### 4.2 Window Comparison Table

```
┌─────────────┬──────────────┬───────────────┬───────────────────┐
│ Window      │ Peak Sidelobe│ Min Stopband  │ Approx. Trans.    │
│             │ Level (dB)   │ Atten. (dB)   │ Width (rad)       │
├─────────────┼──────────────┼───────────────┼───────────────────┤
│ Rectangular │ -13          │ -21           │ 0.92·(2π/M)       │
│ Hann        │ -31          │ -44           │ 3.11·(2π/M)       │
│ Hamming     │ -41          │ -53           │ 3.32·(2π/M)       │
│ Blackman    │ -57          │ -74           │ 5.56·(2π/M)       │
│ Kaiser(β=6) │ -44          │ -60           │ adjustable        │
│ Kaiser(β=9) │ -69          │ -90           │ adjustable        │
└─────────────┴──────────────┴───────────────┴───────────────────┘
```

### 4.3 Visualizing Windows and Their Spectra

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

### 4.4 Design Steps Using Window Method

1. **Determine specifications**: $\omega_p$, $\omega_s$, $\delta_1$, $\delta_2$
2. **Select window type** based on required stopband attenuation
3. **Compute transition width**: $\Delta\omega = \omega_s - \omega_p$
4. **Determine filter order** $M$ from the window's transition width formula
5. **Set cutoff frequency**: $\omega_c = (\omega_p + \omega_s) / 2$
6. **Compute ideal impulse response** $h_d[n]$ shifted by $M/2$
7. **Apply window** to get $h[n] = h_d[n] \cdot w[n]$

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

## 5. Kaiser Window Design

### 5.1 Kaiser Window Definition

The Kaiser window provides a **continuously adjustable** parameter $\beta$ that controls the trade-off between mainlobe width and sidelobe level:

$$w[n] = \frac{I_0\left(\beta\sqrt{1 - \left(\frac{2n}{M} - 1\right)^2}\right)}{I_0(\beta)}, \quad 0 \leq n \leq M$$

where $I_0(\cdot)$ is the zeroth-order modified Bessel function of the first kind.

### 5.2 Kaiser's Empirical Formulas

Given the desired stopband attenuation $A_s$ (in dB):

**Parameter $\beta$:**

$$\beta = \begin{cases} 0.1102(A_s - 8.7), & A_s > 50 \\ 0.5842(A_s - 21)^{0.4} + 0.07886(A_s - 21), & 21 \leq A_s \leq 50 \\ 0, & A_s < 21 \end{cases}$$

**Filter order $M$:**

$$M = \left\lceil \frac{A_s - 7.95}{2.285 \cdot \Delta\omega} \right\rceil$$

where $\Delta\omega = \omega_s - \omega_p$ is the transition width.

### 5.3 Kaiser Window Design Procedure

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

### 5.4 Using scipy.signal.kaiserord

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

## 6. Frequency Sampling Method

### 6.1 Principle

The frequency sampling method designs an FIR filter by specifying the desired frequency response at $N$ equally spaced frequency points and computing the filter coefficients via the inverse DFT:

$$H[k] = H(e^{j 2\pi k / N}), \quad k = 0, 1, \ldots, N-1$$

$$h[n] = \frac{1}{N} \sum_{k=0}^{N-1} H[k] e^{j 2\pi k n / N}, \quad n = 0, 1, \ldots, N-1$$

### 6.2 Transition Samples

The key insight is that **transition band samples** can be optimized to reduce stopband ripple. Instead of a hard transition from 1 to 0, intermediate values $T_1, T_2, \ldots$ are placed in the transition band.

```
Frequency samples H[k]:
H[k]
  ↑
  1 ─ ● ─ ● ─ ● ─ ●
  │                   ╲
T₁│                    ●   ← Transition sample (optimized)
  │                      ╲
  0 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─● ─ ● ─ ● ─ ● ─ ●
  └────────────────────────────────────────→ k
```

### 6.3 Implementation

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

## 7. Optimal Equiripple Design: Parks-McClellan Algorithm

### 7.1 Minimax Optimization

The Parks-McClellan (PM) algorithm, also known as the Remez exchange algorithm, designs FIR filters that minimize the maximum weighted error in the frequency domain:

$$\min_{h} \max_{\omega \in \text{bands}} \left| W(\omega) \left[ H_d(\omega) - H(\omega) \right] \right|$$

where:
- $H_d(\omega)$ is the desired frequency response
- $H(\omega) = \sum_{n=0}^{M} h[n] e^{-j\omega n}$ is the actual response
- $W(\omega)$ is a weighting function

### 7.2 Equiripple Property

The Chebyshev theorem guarantees that the optimal solution exhibits an **equiripple** error -- the error oscillates between $\pm\delta$ with equal amplitude across all specified bands:

```
Equiripple Error Behavior:
          Passband                      Stopband
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

### 7.3 Remez Exchange Algorithm

The algorithm iteratively:

1. **Initialize** with $(M/2 + 2)$ extremal frequency points
2. **Solve** the interpolation problem to find the polynomial that produces equiripple behavior
3. **Search** for new extremal points where the error is maximum
4. **Exchange** the current extremal set with the new one
5. **Repeat** until convergence (extremal points stabilize)

### 7.4 Implementation with scipy.signal.remez

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

### 7.5 Bandpass Filter Example

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

## 8. Linear Phase FIR Filters

### 8.1 Importance of Linear Phase

A filter has **linear phase** if its frequency response can be written as:

$$H(e^{j\omega}) = |H(e^{j\omega})| \cdot e^{-j\omega \tau}$$

where $\tau$ is a constant group delay. Linear phase means all frequency components are delayed by the same amount, preserving the waveform shape.

**Group delay:**

$$\tau(\omega) = -\frac{d\angle H(e^{j\omega})}{d\omega} = \text{constant} = \frac{M}{2}$$

### 8.2 Symmetry Conditions

Linear phase FIR filters require symmetric or antisymmetric impulse responses:

**Symmetric** (Type I and II): $h[n] = h[M - n]$

**Antisymmetric** (Type III and IV): $h[n] = -h[M - n]$

### 8.3 Four Types of Linear Phase FIR Filters

```
┌────────┬──────────────┬──────────────────────┬───────────────────┐
│ Type   │ Symmetry     │ Order M              │ Suitable For      │
├────────┼──────────────┼──────────────────────┼───────────────────┤
│ I      │ Symmetric    │ Even                 │ All filter types  │
│ II     │ Symmetric    │ Odd                  │ LP, BP only       │
│ III    │ Antisymmetric│ Even                 │ BP, Hilbert, Diff │
│ IV     │ Antisymmetric│ Odd                  │ HP, BP, Hilbert   │
└────────┴──────────────┴──────────────────────┴───────────────────┘
```

**Type I** (even order, symmetric):
- $H(e^{j0}) = \sum h[n]$ (no constraint)
- $H(e^{j\pi}) = \sum (-1)^n h[n]$ (no constraint)
- Suitable for any filter type

**Type II** (odd order, symmetric):
- $H(e^{j\pi}) = 0$ always
- Cannot be used for highpass or bandstop filters

**Type III** (even order, antisymmetric):
- $H(e^{j0}) = 0$ and $H(e^{j\pi}) = 0$ always
- Suitable for bandpass, Hilbert transform, differentiator

**Type IV** (odd order, antisymmetric):
- $H(e^{j0}) = 0$ always
- Cannot be used for lowpass filters

### 8.4 Amplitude Response Formulas

For a Type I filter (even $M$, symmetric), the frequency response is:

$$H(e^{j\omega}) = e^{-j\omega M/2} A(\omega)$$

where the **amplitude response** is:

$$A(\omega) = h[M/2] + 2 \sum_{k=1}^{M/2} h[M/2 - k] \cos(k\omega)$$

This is a real-valued function, making it easy to analyze passband/stopband behavior.

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

## 9. Comparison of Design Methods

### 9.1 Head-to-Head Comparison

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

### 9.2 Summary of Design Methods

```
┌──────────────────┬──────────────┬────────────────┬──────────────────┐
│ Method           │ Optimality   │ Control        │ Best For         │
├──────────────────┼──────────────┼────────────────┼──────────────────┤
│ Window           │ Suboptimal   │ Limited        │ Quick design,    │
│                  │              │ (discrete      │ education        │
│                  │              │  window choice)│                  │
├──────────────────┼──────────────┼────────────────┼──────────────────┤
│ Kaiser Window    │ Near-optimal │ Continuous β   │ Good balance     │
│                  │              │ parameter      │ of control and   │
│                  │              │                │ simplicity       │
├──────────────────┼──────────────┼────────────────┼──────────────────┤
│ Freq. Sampling   │ Suboptimal   │ Transition     │ Hardware impl.,  │
│                  │              │ samples        │ DFT-based filters│
├──────────────────┼──────────────┼────────────────┼──────────────────┤
│ Parks-McClellan  │ Optimal      │ Weight func,   │ Production,      │
│                  │ (minimax)    │ multiple bands │ stringent specs  │
└──────────────────┴──────────────┴────────────────┴──────────────────┘
```

**Key insight**: For the same filter order, Parks-McClellan achieves the smallest maximum error (equiripple). Window methods produce filters with monotonically increasing stopband attenuation, which "wastes" attenuation at higher frequencies.

---

## 10. FIR Filter Implementation

### 10.1 Direct Convolution

The most straightforward implementation computes the output as a direct convolution:

$$y[n] = \sum_{k=0}^{M} h[k] x[n-k]$$

**Complexity**: $O(MN)$ for filtering $N$ samples with an $M$-th order filter.

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

### 10.2 FFT-Based Filtering

For long signals, FFT-based convolution is much faster:

$$y = \text{IFFT}[\text{FFT}(x) \cdot \text{FFT}(h)]$$

**Complexity**: $O(N \log N)$ instead of $O(MN)$.

The crossover point where FFT becomes faster is typically $M \approx 50-100$.

### 10.3 Overlap-Add Method

For real-time or block-based processing where the input arrives in segments:

1. **Divide** input $x[n]$ into non-overlapping blocks of length $L$
2. **Zero-pad** each block to length $L + M$
3. **FFT-convolve** each block with the filter $h[n]$
4. **Overlap-add** adjacent output blocks (last $M$ samples overlap)

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

### 10.4 Overlap-Save Method

An alternative to overlap-add:

1. **Overlap** input blocks by $M-1$ samples (each block has $M-1$ samples from previous block)
2. **FFT-convolve** using circular convolution (length $L+M-1$)
3. **Discard** first $M-1$ samples of each output block (corrupted by circular wrap-around)
4. **Concatenate** remaining samples

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

### 10.5 Performance Comparison

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

## 11. Python Implementation

### 11.1 Complete Design Example

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

### 11.2 Filtering a Real Signal

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

### 11.3 Multi-band Filter Design

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

## 12. Exercises

### Exercise 1: Window Selection

Design a lowpass FIR filter with the following specifications:
- Sampling frequency: $f_s = 10000$ Hz
- Passband edge: $f_p = 1500$ Hz
- Stopband edge: $f_s = 2000$ Hz
- Minimum stopband attenuation: $60$ dB

(a) Which window type(s) can meet the stopband specification?

(b) For each viable window, compute the required filter order $M$.

(c) Design the filter using `scipy.signal.firwin` and verify the specifications are met.

(d) Compare the resulting filters in terms of passband ripple and transition width.

### Exercise 2: Kaiser Window Design

A digital filter must satisfy:
- Passband: $0 \leq f \leq 3$ kHz with ripple $\leq 0.1$ dB
- Stopband: $f \geq 4$ kHz with attenuation $\geq 50$ dB
- Sampling rate: $f_s = 20$ kHz

(a) Compute the Kaiser window parameter $\beta$ using the empirical formula.

(b) Compute the minimum filter order $M$.

(c) Design the filter and plot the magnitude response, phase response, and group delay.

(d) Verify the design meets specifications by measuring actual passband ripple and stopband attenuation.

### Exercise 3: Parks-McClellan Bandpass Filter

Design a bandpass FIR filter using the Parks-McClellan algorithm:
- $f_s = 44100$ Hz (CD audio rate)
- Stopband 1: $0$ to $800$ Hz (attenuation $\geq 40$ dB)
- Passband: $1000$ to $3000$ Hz (ripple $\leq 0.5$ dB)
- Stopband 2: $3500$ to $22050$ Hz (attenuation $\geq 40$ dB)

(a) Choose appropriate weights for equal ripple in both stopbands.

(b) Determine the minimum filter order.

(c) Plot the magnitude response and verify the specifications.

(d) Apply the filter to a chirp signal (sweep from 100 Hz to 10 kHz) and display the spectrogram before and after filtering.

### Exercise 4: Overlap-Add Implementation

Implement a real-time FIR filter processor using the overlap-add method:

(a) Write a class `StreamingFIRFilter` that processes audio in blocks.

(b) Test with block sizes of 64, 256, 1024, and 4096 samples.

(c) Verify the output matches direct convolution (`np.convolve`).

(d) Measure and plot processing time per block as a function of block size.

(e) Find the optimal block size for a 200-tap filter applied to a 10-second audio signal at 44100 Hz.

### Exercise 5: Linear Phase Constraints

(a) Prove that a Type II FIR filter (odd order, symmetric coefficients) always has $H(e^{j\pi}) = 0$. Why does this make it unsuitable for highpass filter design?

(b) Design a highpass FIR filter with cutoff at $0.6\pi$ using:
   - Type I (even order)
   - Type IV (odd order, antisymmetric)

   Compare the results. Which has better performance near $\omega = \pi$?

(c) Design a Hilbert transform filter (Type III) of order 30. Plot its magnitude and phase response. What happens at $\omega = 0$ and $\omega = \pi$?

### Exercise 6: Comparison Study

For the specifications: $f_s = 16000$ Hz, $f_p = 2000$ Hz, $f_\text{stop} = 2500$ Hz, $A_s = 50$ dB:

(a) Design the filter using (i) Hamming window, (ii) Kaiser window, (iii) Parks-McClellan.

(b) For each method, report: filter order, passband ripple (dB), actual stopband attenuation (dB), transition width (Hz).

(c) Create a single figure with 4 subplots: overlay of magnitude responses, passband detail, impulse responses, and pole-zero plots.

(d) Which method gives the best result for the lowest filter order? Justify your answer.

### Exercise 7: Minimum-Phase FIR Filter

(a) Design a linear-phase lowpass FIR filter of order 40 using the Hamming window method.

(b) Convert it to a minimum-phase FIR filter using the cepstral method:
   - Compute the cepstrum $\hat{h}[n] = \text{IFFT}(\log|\text{FFT}(h)|)$
   - Construct the minimum-phase version

(c) Compare the magnitude responses, phase responses, and group delays of both filters.

(d) When would you prefer minimum-phase over linear-phase? Discuss with examples.

---

## References

1. **Oppenheim, A. V., & Schafer, R. W. (2010).** *Discrete-Time Signal Processing* (3rd ed.). Pearson. Chapters 7-8.
2. **Proakis, J. G., & Manolakis, D. G. (2007).** *Digital Signal Processing* (4th ed.). Pearson. Chapter 10.
3. **Parks, T. W., & Burrus, C. S. (1987).** *Digital Filter Design*. Wiley.
4. **Kaiser, J. F. (1974).** "Nonrecursive Digital Filter Design Using the I0-sinh Window Function." *Proceedings of the 1974 IEEE International Symposium on Circuits and Systems*.
5. **McClellan, J. H., Parks, T. W., & Rabiner, L. R. (1973).** "A Computer Program for Designing Optimum FIR Linear Phase Digital Filters." *IEEE Trans. Audio Electroacoustics*, 21(6), 506-526.
6. **SciPy Documentation** -- `scipy.signal` module: https://docs.scipy.org/doc/scipy/reference/signal.html

---

## Navigation

- Previous: [08. Z-Transform and Transfer Functions](08_Z_Transform_and_Transfer_Functions.md)
- Next: [10. IIR Filter Design](10_IIR_Filter_Design.md)
- [Back to Overview](00_Overview.md)
