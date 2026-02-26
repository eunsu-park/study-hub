# Digital Filter Fundamentals

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between FIR and IIR filters in terms of structure, stability, phase response, and computational cost.
2. Analyze digital filters using difference equations, transfer functions (Z-domain), and frequency response.
3. Explain linear phase conditions for FIR filters and describe their importance in phase-sensitive applications.
4. Interpret filter specifications including passband ripple, stopband attenuation, and transition width.
5. Implement FIR and IIR filters in direct form, cascade, and lattice structures.
6. Recognize and evaluate quantization effects (coefficient quantization, overflow, limit cycles) in fixed-point implementations.

---

## Overview

Digital filters are the workhorses of signal processing, shaping signal spectra by selectively passing or rejecting frequency components. This lesson covers the two fundamental filter types -- FIR and IIR -- their representations, design trade-offs, implementation structures, and practical considerations including quantization effects. Understanding these fundamentals is essential before tackling specific filter design methods.

**Prerequisites:** [07. Z-Transform](07_Z_Transform.md)

---

## 1. Digital Filter Types: FIR and IIR

### 1.1 FIR (Finite Impulse Response)

An FIR filter has an impulse response of finite duration. The output depends only on current and past inputs:

$$y[n] = \sum_{k=0}^{M} b_k \, x[n-k]$$

**Characteristics:**
- Non-recursive (no feedback)
- Always stable (no poles except at $z = 0$)
- Can achieve exact linear phase
- Requires more coefficients for sharp frequency selectivity
- Transfer function: $H(z) = \sum_{k=0}^{M} b_k z^{-k}$ (all zeros, no poles)

### 1.2 IIR (Infinite Impulse Response)

An IIR filter has an impulse response of infinite duration. The output depends on inputs and previous outputs:

$$y[n] = \sum_{k=0}^{M} b_k \, x[n-k] - \sum_{k=1}^{N} a_k \, y[n-k]$$

**Characteristics:**
- Recursive (uses feedback)
- Can be unstable if poles are outside the unit circle
- Cannot achieve exact linear phase (in general)
- More efficient: achieves sharp transitions with fewer coefficients
- Transfer function: $H(z) = \frac{\sum_{k=0}^{M} b_k z^{-k}}{1 + \sum_{k=1}^{N} a_k z^{-k}}$ (poles and zeros)

### 1.3 FIR vs. IIR Comparison

| Property | FIR | IIR |
|----------|-----|-----|
| Stability | Always stable | Can be unstable |
| Linear phase | Achievable | Generally not |
| Order for sharp cutoff | High (50-200+) | Low (4-10) |
| Computational cost | More multiply-adds | Fewer multiply-adds |
| Transient response | Finite (M samples) | Infinite (decaying) |
| Design methods | Windowing, Parks-McClellan | Butterworth, Chebyshev, Elliptic |
| Analog prototype | Not required | Often used |
| Sensitivity to quantization | Low | Higher |
| Latency | Higher (longer filter) | Lower |
| Group delay | Constant (linear phase) | Frequency-dependent |

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def compare_fir_iir():
    """Compare FIR and IIR lowpass filters with similar specifications."""
    fs = 1000  # Hz
    f_cutoff = 100  # Hz

    # FIR: 51-tap lowpass using Hamming window
    fir_order = 50
    b_fir = signal.firwin(fir_order + 1, f_cutoff, fs=fs)
    a_fir = [1.0]

    # IIR: 4th-order Butterworth lowpass
    iir_order = 4
    b_iir, a_iir = signal.butter(iir_order, f_cutoff, fs=fs)

    # Frequency responses
    w_fir, H_fir = signal.freqz(b_fir, a_fir, worN=2048, fs=fs)
    w_iir, H_iir = signal.freqz(b_iir, a_iir, worN=2048, fs=fs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Magnitude response
    axes[0, 0].plot(w_fir, 20 * np.log10(np.abs(H_fir) + 1e-12),
                    'b-', linewidth=1.5, label=f'FIR (order {fir_order})')
    axes[0, 0].plot(w_iir, 20 * np.log10(np.abs(H_iir) + 1e-12),
                    'r-', linewidth=1.5, label=f'IIR Butterworth (order {iir_order})')
    axes[0, 0].axvline(f_cutoff, color='green', linestyle='--', alpha=0.5,
                        label=f'Cutoff = {f_cutoff} Hz')
    axes[0, 0].set_title('Magnitude Response')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('Magnitude (dB)')
    axes[0, 0].set_ylim(-80, 5)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Phase response
    axes[0, 1].plot(w_fir, np.unwrap(np.angle(H_fir)) * 180 / np.pi,
                    'b-', linewidth=1.5, label='FIR')
    axes[0, 1].plot(w_iir, np.unwrap(np.angle(H_iir)) * 180 / np.pi,
                    'r-', linewidth=1.5, label='IIR')
    axes[0, 1].set_title('Phase Response')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Phase (degrees)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Group delay
    w_gd_fir, gd_fir = signal.group_delay((b_fir, a_fir), w=2048, fs=fs)
    w_gd_iir, gd_iir = signal.group_delay((b_iir, a_iir), w=2048, fs=fs)

    axes[1, 0].plot(w_gd_fir, gd_fir, 'b-', linewidth=1.5, label='FIR')
    axes[1, 0].plot(w_gd_iir, gd_iir, 'r-', linewidth=1.5, label='IIR')
    axes[1, 0].set_title('Group Delay')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Delay (samples)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, fs / 2)

    # Impulse response
    N_imp = 80
    imp = np.zeros(N_imp)
    imp[0] = 1.0

    y_fir = signal.lfilter(b_fir, a_fir, imp)
    y_iir = signal.lfilter(b_iir, a_iir, imp)

    axes[1, 1].stem(np.arange(N_imp), y_fir, linefmt='b-', markerfmt='b.',
                    basefmt='k-', label='FIR')
    axes[1, 1].stem(np.arange(N_imp), y_iir, linefmt='r-', markerfmt='r.',
                    basefmt='k-', label='IIR')
    axes[1, 1].set_title('Impulse Response')
    axes[1, 1].set_xlabel('n')
    axes[1, 1].set_ylabel('h[n]')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'FIR (order {fir_order}) vs. IIR Butterworth (order {iir_order})',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('fir_vs_iir.png', dpi=150)
    plt.show()

    # Print coefficients
    print("FIR vs IIR Comparison")
    print("=" * 50)
    print(f"FIR: {fir_order + 1} coefficients (multiply-adds per sample)")
    print(f"IIR: {len(b_iir) + len(a_iir) - 1} coefficients "
          f"({len(b_iir)} numerator + {len(a_iir) - 1} denominator)")

compare_fir_iir()
```

---

## 2. Difference Equations and Transfer Functions

### 2.1 General Difference Equation

The general linear constant-coefficient difference equation (LCCDE):

$$\sum_{k=0}^{N} a_k \, y[n-k] = \sum_{k=0}^{M} b_k \, x[n-k]$$

with $a_0 = 1$ by convention:

$$y[n] = \sum_{k=0}^{M} b_k \, x[n-k] - \sum_{k=1}^{N} a_k \, y[n-k]$$

### 2.2 Transfer Function in z-Domain

Taking the Z-transform (assuming zero initial conditions):

$$H(z) = \frac{Y(z)}{X(z)} = \frac{\sum_{k=0}^{M} b_k z^{-k}}{1 + \sum_{k=1}^{N} a_k z^{-k}} = \frac{B(z)}{A(z)}$$

In factored form:

$$H(z) = \frac{b_0}{a_0} \cdot \frac{\prod_{k=1}^{M}(1 - z_k z^{-1})}{\prod_{k=1}^{N}(1 - p_k z^{-1})}$$

where $z_k$ are the zeros and $p_k$ are the poles.

### 2.3 Frequency Response

The frequency response is obtained by evaluating $H(z)$ on the unit circle:

$$H(e^{j\omega}) = |H(e^{j\omega})| \, e^{j\phi(\omega)}$$

- $|H(e^{j\omega})|$: **Magnitude response** (amplitude gain at frequency $\omega$)
- $\phi(\omega) = \angle H(e^{j\omega})$: **Phase response** (phase shift at frequency $\omega$)

```python
def difference_equation_to_transfer_function():
    """Demonstrate the relationship between difference equation,
    transfer function, and frequency response."""
    # Example: y[n] = 0.5(x[n] + x[n-1]) - Simple moving average
    # This is a 2-tap FIR (order 1)
    b_ma = [0.5, 0.5]
    a_ma = [1.0]

    # Example: y[n] = 0.9*y[n-1] + 0.1*x[n] - First-order IIR lowpass
    b_iir = [0.1]
    a_iir = [1.0, -0.9]

    systems = [
        ('2-tap Moving Average (FIR)', b_ma, a_ma),
        ('First-order IIR Lowpass', b_iir, a_iir),
    ]

    fig, axes = plt.subplots(len(systems), 3, figsize=(16, 8))

    for i, (name, b, a) in enumerate(systems):
        # Impulse response
        N = 30
        imp = np.zeros(N)
        imp[0] = 1.0
        h = signal.lfilter(b, a, imp)

        axes[i, 0].stem(np.arange(N), h, linefmt='b-', markerfmt='bo',
                        basefmt='k-')
        axes[i, 0].set_title(f'{name}\nImpulse Response')
        axes[i, 0].set_xlabel('n')
        axes[i, 0].set_ylabel('h[n]')
        axes[i, 0].grid(True, alpha=0.3)

        # Magnitude response
        w, H = signal.freqz(b, a, worN=1024)
        axes[i, 1].plot(w / np.pi, 20 * np.log10(np.abs(H) + 1e-12),
                        'b-', linewidth=2)
        axes[i, 1].set_title('Magnitude Response')
        axes[i, 1].set_xlabel(r'$\omega / \pi$')
        axes[i, 1].set_ylabel('dB')
        axes[i, 1].set_ylim(-40, 5)
        axes[i, 1].grid(True, alpha=0.3)

        # Phase response
        axes[i, 2].plot(w / np.pi, np.unwrap(np.angle(H)) * 180 / np.pi,
                        'r-', linewidth=2)
        axes[i, 2].set_title('Phase Response')
        axes[i, 2].set_xlabel(r'$\omega / \pi$')
        axes[i, 2].set_ylabel('Degrees')
        axes[i, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('diff_eq_to_tf.png', dpi=150)
    plt.show()

difference_equation_to_transfer_function()
```

---

## 3. FIR Filter Characteristics

### 3.1 FIR Transfer Function

$$H(z) = \sum_{k=0}^{M} b_k z^{-k} = b_0 + b_1 z^{-1} + b_2 z^{-2} + \cdots + b_M z^{-M}$$

- $M$ zeros (plus $M$ poles at the origin $z = 0$)
- Order $M$, length $M + 1$
- Always stable (all poles at origin)

### 3.2 FIR as a Polynomial in $z^{-1}$

The FIR filter is simply a polynomial evaluator. Each coefficient $b_k$ is also the impulse response value $h[k]$:

$$h[n] = b_n, \quad n = 0, 1, \ldots, M$$

### 3.3 Common FIR Filter Types

```python
def common_fir_filters():
    """Demonstrate common FIR filter building blocks."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # 1. Moving average (lowpass)
    M = 10
    b_ma = np.ones(M + 1) / (M + 1)
    w, H_ma = signal.freqz(b_ma, [1.0], worN=2048)

    axes[0, 0].stem(np.arange(M + 1), b_ma, linefmt='b-', markerfmt='bo',
                    basefmt='k-')
    axes[0, 0].set_title(f'Moving Average (M={M}): Coefficients')
    axes[0, 0].set_xlabel('n')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(w / np.pi, 20 * np.log10(np.abs(H_ma) + 1e-12),
                    'b-', linewidth=2)
    axes[0, 1].set_title('Moving Average: Magnitude Response')
    axes[0, 1].set_xlabel(r'$\omega / \pi$')
    axes[0, 1].set_ylabel('dB')
    axes[0, 1].set_ylim(-60, 5)
    axes[0, 1].grid(True, alpha=0.3)

    # 2. Differentiator
    b_diff = [1, -1]
    w, H_diff = signal.freqz(b_diff, [1.0], worN=2048)

    axes[1, 0].stem(np.arange(len(b_diff)), b_diff, linefmt='r-',
                    markerfmt='ro', basefmt='k-')
    axes[1, 0].set_title('First Difference (Differentiator): Coefficients')
    axes[1, 0].set_xlabel('n')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(w / np.pi, np.abs(H_diff), 'r-', linewidth=2)
    axes[1, 1].set_title('Differentiator: Magnitude Response')
    axes[1, 1].set_xlabel(r'$\omega / \pi$')
    axes[1, 1].set_ylabel('|H|')
    axes[1, 1].grid(True, alpha=0.3)

    # 3. Comb filter
    D = 8  # Delay
    b_comb = np.zeros(D + 1)
    b_comb[0] = 1
    b_comb[D] = -1
    w, H_comb = signal.freqz(b_comb, [1.0], worN=2048)

    axes[2, 0].stem(np.arange(len(b_comb)), b_comb, linefmt='g-',
                    markerfmt='go', basefmt='k-')
    axes[2, 0].set_title(f'Comb Filter (D={D}): Coefficients')
    axes[2, 0].set_xlabel('n')
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(w / np.pi, 20 * np.log10(np.abs(H_comb) + 1e-12),
                    'g-', linewidth=2)
    axes[2, 1].set_title('Comb Filter: Magnitude Response')
    axes[2, 1].set_xlabel(r'$\omega / \pi$')
    axes[2, 1].set_ylabel('dB')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('common_fir.png', dpi=150)
    plt.show()

common_fir_filters()
```

---

## 4. IIR Filter Characteristics

### 4.1 IIR Transfer Function

$$H(z) = \frac{B(z)}{A(z)} = \frac{\sum_{k=0}^{M} b_k z^{-k}}{1 + \sum_{k=1}^{N} a_k z^{-k}}$$

- Has both poles (from $A(z)$) and zeros (from $B(z)$)
- Filter order $= \max(M, N)$, typically $N$ (the denominator order)
- Stability requires all poles inside the unit circle

### 4.2 IIR Filter Families

The classical IIR filter design starts from analog prototypes:

| Family | Passband | Stopband | Transition | Phase |
|--------|----------|----------|-----------|-------|
| **Butterworth** | Maximally flat | Monotonic | Widest | Smoothest |
| **Chebyshev Type I** | Equiripple | Monotonic | Narrower | More nonlinear |
| **Chebyshev Type II** | Monotonic | Equiripple | Narrower | More nonlinear |
| **Elliptic (Cauer)** | Equiripple | Equiripple | Narrowest | Most nonlinear |
| **Bessel** | Nearly flat | Poor | Widest | Most linear |

```python
def compare_iir_families():
    """Compare the four main IIR filter families."""
    fs = 1000
    f_cutoff = 100
    order = 5

    # Design filters
    filters = {
        'Butterworth': signal.butter(order, f_cutoff, fs=fs),
        'Chebyshev I (1dB ripple)': signal.cheby1(order, 1, f_cutoff, fs=fs),
        'Chebyshev II (40dB atten)': signal.cheby2(order, 40, f_cutoff, fs=fs),
        'Elliptic (1dB/40dB)': signal.ellip(order, 1, 40, f_cutoff, fs=fs),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['blue', 'red', 'green', 'orange']

    # Magnitude response
    ax = axes[0, 0]
    for (name, (b, a)), color in zip(filters.items(), colors):
        w, H = signal.freqz(b, a, worN=4096, fs=fs)
        ax.plot(w, 20 * np.log10(np.abs(H) + 1e-12), color=color,
                linewidth=1.5, label=name)
    ax.axvline(f_cutoff, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'Magnitude Response (Order {order})')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_ylim(-80, 5)
    ax.set_xlim(0, 250)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Passband detail
    ax = axes[0, 1]
    for (name, (b, a)), color in zip(filters.items(), colors):
        w, H = signal.freqz(b, a, worN=4096, fs=fs)
        ax.plot(w, 20 * np.log10(np.abs(H) + 1e-12), color=color,
                linewidth=1.5, label=name)
    ax.axvline(f_cutoff, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Passband Detail')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_ylim(-3, 1)
    ax.set_xlim(0, 120)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Group delay
    ax = axes[1, 0]
    for (name, (b, a)), color in zip(filters.items(), colors):
        w, gd = signal.group_delay((b, a), w=4096, fs=fs)
        ax.plot(w, gd, color=color, linewidth=1.5, label=name)
    ax.set_title('Group Delay')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Delay (samples)')
    ax.set_xlim(0, 200)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Pole-zero plot for Butterworth
    ax = axes[1, 1]
    b_bw, a_bw = filters['Butterworth']
    zeros = np.roots(b_bw)
    poles = np.roots(a_bw)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5)
    ax.plot(np.real(zeros), np.imag(zeros), 'bo', markersize=8, label='Zeros')
    ax.plot(np.real(poles), np.imag(poles), 'rx', markersize=10,
            markeredgewidth=2, label='Poles')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title(f'Butterworth Order {order}: Pole-Zero Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('iir_families.png', dpi=150)
    plt.show()

compare_iir_families()
```

### 4.3 Potential Instability of IIR Filters

```python
def demonstrate_iir_instability():
    """Show how IIR filters can become unstable."""
    # Stable filter: pole at 0.95
    b_stable = [1.0]
    a_stable = [1.0, -0.95]

    # Unstable filter: pole at 1.05
    b_unstable = [1.0]
    a_unstable = [1.0, -1.05]

    N = 50
    imp = np.zeros(N)
    imp[0] = 1.0

    h_stable = signal.lfilter(b_stable, a_stable, imp)
    h_unstable = signal.lfilter(b_unstable, a_unstable, imp)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].stem(np.arange(N), h_stable, linefmt='g-', markerfmt='go',
                 basefmt='k-')
    axes[0].set_title('Stable IIR (pole = 0.95)')
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('h[n]')
    axes[0].grid(True, alpha=0.3)

    axes[1].stem(np.arange(N), h_unstable, linefmt='r-', markerfmt='ro',
                 basefmt='k-')
    axes[1].set_title('UNSTABLE IIR (pole = 1.05)')
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('h[n]')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('iir_instability.png', dpi=150)
    plt.show()

    print("IIR Stability Demonstration")
    print("=" * 50)
    print(f"Stable: h[49] = {h_stable[49]:.6e}")
    print(f"Unstable: h[49] = {h_unstable[49]:.6e}")

demonstrate_iir_instability()
```

---

## 5. Frequency Response: Magnitude and Phase

### 5.1 Magnitude Response

$$|H(e^{j\omega})| = \sqrt{\text{Re}[H(e^{j\omega})]^2 + \text{Im}[H(e^{j\omega})]^2}$$

Often expressed in decibels: $|H|_{\text{dB}} = 20 \log_{10}|H(e^{j\omega})|$

### 5.2 Phase Response

$$\phi(\omega) = \angle H(e^{j\omega}) = \arctan\!\left(\frac{\text{Im}[H]}{\text{Re}[H]}\right)$$

### 5.3 Group Delay

The **group delay** is the negative derivative of the phase response:

$$\tau_g(\omega) = -\frac{d\phi(\omega)}{d\omega}$$

Group delay represents the delay experienced by the envelope of a narrowband signal centered at frequency $\omega$. For a linear phase filter, the group delay is constant.

### 5.4 Phase Delay

The **phase delay** is:

$$\tau_p(\omega) = -\frac{\phi(\omega)}{\omega}$$

Phase delay represents the delay experienced by a pure sinusoid at frequency $\omega$.

For linear phase: $\phi(\omega) = -\omega \tau_0$, so $\tau_g = \tau_p = \tau_0$ (constant).

```python
def phase_and_group_delay():
    """Demonstrate phase response, group delay, and phase delay."""
    fs = 1000

    # FIR with linear phase (symmetric coefficients)
    N_fir = 31
    b_fir = signal.firwin(N_fir, 200, fs=fs)
    a_fir = [1.0]

    # IIR (nonlinear phase)
    b_iir, a_iir = signal.butter(6, 200, fs=fs)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    titles = ['FIR (Linear Phase)', 'IIR Butterworth (Nonlinear Phase)']
    systems = [(b_fir, a_fir), (b_iir, a_iir)]

    for col, ((b, a), title) in enumerate(zip(systems, titles)):
        w, H = signal.freqz(b, a, worN=2048, fs=fs)

        # Magnitude
        axes[0, col].plot(w, 20 * np.log10(np.abs(H) + 1e-12),
                          'b-', linewidth=2)
        axes[0, col].set_title(f'{title}\nMagnitude Response')
        axes[0, col].set_xlabel('Frequency (Hz)')
        axes[0, col].set_ylabel('dB')
        axes[0, col].set_ylim(-80, 5)
        axes[0, col].grid(True, alpha=0.3)

        # Phase (unwrapped)
        phase = np.unwrap(np.angle(H))
        axes[1, col].plot(w, phase * 180 / np.pi, 'r-', linewidth=2)
        axes[1, col].set_title('Phase Response')
        axes[1, col].set_xlabel('Frequency (Hz)')
        axes[1, col].set_ylabel('Phase (degrees)')
        axes[1, col].grid(True, alpha=0.3)

        # Group delay
        w_gd, gd = signal.group_delay((b, a), w=2048, fs=fs)
        axes[2, col].plot(w_gd, gd, 'g-', linewidth=2)
        axes[2, col].set_title('Group Delay')
        axes[2, col].set_xlabel('Frequency (Hz)')
        axes[2, col].set_ylabel('Delay (samples)')
        axes[2, col].set_xlim(0, fs / 2)
        axes[2, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('phase_group_delay.png', dpi=150)
    plt.show()

phase_and_group_delay()
```

---

## 6. Linear Phase FIR Filters

### 6.1 Why Linear Phase Matters

A linear phase filter has the phase response:

$$\phi(\omega) = -\alpha \omega + \beta$$

where $\alpha$ is the constant group delay and $\beta$ is 0 or $\pi/2$.

**Importance:**
- All frequency components are delayed by the same amount
- No phase distortion of the signal waveform
- Critical for applications like audio, communications, and image processing
- Preserves the shape of transient signals

### 6.2 Symmetry Conditions

An FIR filter $h[n]$ of length $N = M+1$ has linear phase if and only if its coefficients satisfy one of two symmetry conditions:

**Symmetric (Type I and II):**
$$h[n] = h[M-n], \quad n = 0, 1, \ldots, M$$

**Anti-symmetric (Type III and IV):**
$$h[n] = -h[M-n], \quad n = 0, 1, \ldots, M$$

### 6.3 The Four Types of Linear Phase FIR Filters

| Type | Symmetry | Length ($M+1$) | Phase ($\beta$) | Suitable for |
|------|----------|----------------|-------|-------------|
| **I** | Symmetric | Odd | 0 | LP, HP, BP, BS |
| **II** | Symmetric | Even | 0 | LP, BP only |
| **III** | Anti-symmetric | Odd | $\pi/2$ | BP, differentiator |
| **IV** | Anti-symmetric | Even | $\pi/2$ | HP, BP, differentiator, Hilbert |

**Constraints:**
- Type II: $H(e^{j\pi}) = 0$ (cannot be highpass)
- Type III: $H(e^{j0}) = 0$ and $H(e^{j\pi}) = 0$ (cannot be lowpass or highpass)
- Type IV: $H(e^{j0}) = 0$ (cannot be lowpass)

```python
def linear_phase_types():
    """Demonstrate all four types of linear phase FIR filters."""
    fig, axes = plt.subplots(4, 3, figsize=(16, 16))

    # Type I: Symmetric, Odd length (lowpass)
    M = 20  # Order (even -> odd length M+1=21)
    h1 = signal.firwin(M + 1, 0.4)  # Symmetric by construction
    assert len(h1) % 2 == 1  # Odd length

    # Type II: Symmetric, Even length (lowpass)
    M2 = 19  # Order (odd -> even length M+1=20)
    h2 = signal.firwin(M2 + 1, 0.4)

    # Type III: Anti-symmetric, Odd length (bandpass/differentiator)
    M3 = 20
    h3 = signal.firwin(M3 + 1, [0.2, 0.8], pass_zero=False)
    # Make it anti-symmetric: h[n] = -h[M-n]
    h3_anti = (h3 - h3[::-1]) / 2
    # But more naturally, use remez for differentiator
    h3_diff = signal.remez(M3 + 1, [0.05, 0.95], [1], type='differentiator')

    # Type IV: Anti-symmetric, Even length (Hilbert transformer)
    M4 = 19
    h4 = signal.remez(M4 + 1, [0.05, 0.95], [1], type='hilbert')

    types_data = [
        ('Type I (Symmetric, Odd)', h1, 'blue'),
        ('Type II (Symmetric, Even)', h2, 'red'),
        ('Type III (Anti-sym, Odd)', h3_diff, 'green'),
        ('Type IV (Anti-sym, Even)', h4, 'orange'),
    ]

    for row, (name, h, color) in enumerate(types_data):
        N = len(h)
        n = np.arange(N)

        # Coefficients
        axes[row, 0].stem(n, h, linefmt=f'{color[0]}-', markerfmt=f'{color[0]}o',
                          basefmt='k-')
        axes[row, 0].set_title(f'{name}\nCoefficients (N={N})')
        axes[row, 0].set_xlabel('n')
        axes[row, 0].grid(True, alpha=0.3)

        # Check symmetry
        is_symmetric = np.allclose(h, h[::-1], atol=1e-10)
        is_antisymmetric = np.allclose(h, -h[::-1], atol=1e-10)
        sym_text = 'Symmetric' if is_symmetric else \
                   'Anti-symmetric' if is_antisymmetric else 'Neither'
        axes[row, 0].text(0.95, 0.95, sym_text, transform=axes[row, 0].transAxes,
                          ha='right', va='top', fontsize=9,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Magnitude response
        w, H_freq = signal.freqz(h, [1.0], worN=2048)
        axes[row, 1].plot(w / np.pi, 20 * np.log10(np.abs(H_freq) + 1e-12),
                          color=color, linewidth=2)
        axes[row, 1].set_title('Magnitude Response')
        axes[row, 1].set_xlabel(r'$\omega / \pi$')
        axes[row, 1].set_ylabel('dB')
        axes[row, 1].set_ylim(-80, 10)
        axes[row, 1].grid(True, alpha=0.3)

        # Group delay
        w_gd, gd = signal.group_delay((h, [1.0]), w=2048)
        axes[row, 2].plot(w_gd / np.pi, gd, color=color, linewidth=2)
        axes[row, 2].set_title(f'Group Delay (expected: {(N-1)/2:.1f})')
        axes[row, 2].set_xlabel(r'$\omega / \pi$')
        axes[row, 2].set_ylabel('Samples')
        axes[row, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('linear_phase_types.png', dpi=150)
    plt.show()

linear_phase_types()
```

### 6.4 Proof: Symmetric FIR Has Linear Phase

For a Type I filter ($M$ even, symmetric: $h[n] = h[M-n]$):

$$H(e^{j\omega}) = e^{-j\omega M/2} \underbrace{\left[ h[M/2] + 2\sum_{k=1}^{M/2} h[M/2-k] \cos(k\omega) \right]}_{\tilde{H}(\omega) \text{ (real-valued)}}$$

The amplitude response $\tilde{H}(\omega)$ is purely real, and the phase is exactly $-\omega M/2$ (linear), plus an additional shift of $\pi$ when $\tilde{H}(\omega) < 0$.

---

## 7. Filter Specifications

### 7.1 Standard Specifications

A typical filter is specified by:

```
    |H(e^jw)|
    ↑
1+δp ─ ─ ─ ─┐
    │        │  Passband ripple
1   │────────│
1-δp ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─
    │        │\
    │        │ \ Transition
    │        │  \  band
    │        │   \
 δs ─ ─ ─ ─ ─ ─ ─\─ ─ ─ ─ ─ ─
    │              │~~~~~~~~~~~
  0 ├──────┬──────┬──────┬────→ ω
    0      ωp    ωs             π
```

| Parameter | Symbol | Definition |
|-----------|--------|-----------|
| Passband edge | $\omega_p$ | Frequency where passband ends |
| Stopband edge | $\omega_s$ | Frequency where stopband begins |
| Passband ripple | $\delta_p$ | Maximum deviation from 1 in passband |
| Stopband attenuation | $\delta_s$ | Maximum level in stopband |
| Transition width | $\Delta\omega = \omega_s - \omega_p$ | Width of transition band |

### 7.2 Decibel Conversions

$$\text{Passband ripple (dB)} = -20 \log_{10}(1 - \delta_p)$$

$$\text{Stopband attenuation (dB)} = -20 \log_{10}(\delta_s)$$

| Specification | Linear | dB |
|--------------|--------|-----|
| 1% passband ripple | $\delta_p = 0.01$ | 0.087 dB |
| 0.1 dB ripple | $\delta_p \approx 0.0115$ | 0.1 dB |
| 1 dB ripple | $\delta_p \approx 0.109$ | 1 dB |
| 40 dB stopband | $\delta_s = 0.01$ | 40 dB |
| 60 dB stopband | $\delta_s = 0.001$ | 60 dB |
| 80 dB stopband | $\delta_s = 0.0001$ | 80 dB |

### 7.3 Filter Order Estimation

**FIR (Kaiser's formula):**

$$M \approx \frac{-20\log_{10}(\sqrt{\delta_p \delta_s}) - 13}{14.6 \cdot \Delta f / f_s}$$

**IIR (Butterworth):**

$$N \geq \frac{\log(\delta_p / \delta_s)}{2\log(\omega_p / \omega_s)}$$

```python
def filter_specification_demo():
    """Demonstrate filter specifications with a practical design."""
    fs = 8000  # Hz

    # Specifications
    f_pass = 1000   # Passband edge (Hz)
    f_stop = 1500   # Stopband edge (Hz)
    delta_p = 0.01  # Passband ripple (linear)
    delta_s = 0.001 # Stopband attenuation (linear)

    # Convert to dB
    rp_db = -20 * np.log10(1 - delta_p)
    rs_db = -20 * np.log10(delta_s)

    print("Filter Specifications")
    print("=" * 50)
    print(f"Sampling rate: {fs} Hz")
    print(f"Passband edge: {f_pass} Hz")
    print(f"Stopband edge: {f_stop} Hz")
    print(f"Transition width: {f_stop - f_pass} Hz")
    print(f"Passband ripple: {delta_p} ({rp_db:.3f} dB)")
    print(f"Stopband attenuation: {delta_s} ({rs_db:.1f} dB)")

    # Estimate FIR order (Kaiser)
    delta_f = (f_stop - f_pass) / fs
    M_kaiser = int(np.ceil(
        (-20 * np.log10(np.sqrt(delta_p * delta_s)) - 13) / (14.6 * delta_f)
    ))
    print(f"\nEstimated FIR order (Kaiser): {M_kaiser}")

    # Estimate IIR Butterworth order
    N_butter, Wn = signal.buttord(f_pass, f_stop, rp_db, rs_db, fs=fs)
    print(f"Estimated Butterworth order: {N_butter}")

    # Design both filters
    b_fir = signal.firwin(M_kaiser + 1, (f_pass + f_stop) / 2, fs=fs)

    b_iir, a_iir = signal.butter(N_butter, Wn, fs=fs)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    w_fir, H_fir = signal.freqz(b_fir, [1.0], worN=4096, fs=fs)
    w_iir, H_iir = signal.freqz(b_iir, a_iir, worN=4096, fs=fs)

    ax = axes[0]
    ax.plot(w_fir, 20 * np.log10(np.abs(H_fir) + 1e-12), 'b-',
            linewidth=1.5, label=f'FIR (order {M_kaiser})')
    ax.plot(w_iir, 20 * np.log10(np.abs(H_iir) + 1e-12), 'r-',
            linewidth=1.5, label=f'Butterworth (order {N_butter})')

    # Specification boundaries
    ax.axhline(-rp_db, color='green', linestyle=':', alpha=0.7,
               label=f'Passband: -{rp_db:.3f} dB')
    ax.axhline(-rs_db, color='orange', linestyle=':', alpha=0.7,
               label=f'Stopband: -{rs_db:.1f} dB')
    ax.axvline(f_pass, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(f_stop, color='gray', linestyle='--', alpha=0.5)
    ax.axvspan(0, f_pass, alpha=0.05, color='green')
    ax.axvspan(f_stop, fs / 2, alpha=0.05, color='red')

    ax.set_title('Filter Design Meeting Specifications')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_ylim(-80, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Passband detail
    ax = axes[1]
    ax.plot(w_fir, 20 * np.log10(np.abs(H_fir) + 1e-12), 'b-',
            linewidth=1.5, label=f'FIR (order {M_kaiser})')
    ax.plot(w_iir, 20 * np.log10(np.abs(H_iir) + 1e-12), 'r-',
            linewidth=1.5, label=f'Butterworth (order {N_butter})')
    ax.axhline(-rp_db, color='green', linestyle=':', alpha=0.7)
    ax.set_title('Passband Detail')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_xlim(0, f_stop * 1.2)
    ax.set_ylim(-3, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('filter_specs.png', dpi=150)
    plt.show()

filter_specification_demo()
```

---

## 8. Filter Structures

### 8.1 Direct Form I

The most straightforward implementation of the general difference equation:

$$y[n] = \sum_{k=0}^{M} b_k x[n-k] - \sum_{k=1}^{N} a_k y[n-k]$$

**Signal flow:**
```
x[n] ──→[b0]──→(+)──→(+)──→ y[n]
  │              ↑      ↑
  ├→[z⁻¹]→[b1]──┘      │
  │                     │
  ├→[z⁻¹]→[b2]──→(+)   │
  ⋮              ↑      │
                       [-a1]←[z⁻¹]←── y[n]
                        │
                       [-a2]←[z⁻¹]
                        ⋮
```

**Memory requirement:** $M + N$ delay elements.

### 8.2 Direct Form II (Canonical Form)

Reduces memory by sharing delay elements between numerator and denominator:

$$w[n] = x[n] - \sum_{k=1}^{N} a_k w[n-k]$$
$$y[n] = \sum_{k=0}^{M} b_k w[n-k]$$

**Memory requirement:** $\max(M, N)$ delay elements (canonical: minimum possible).

```
x[n] ──→(+)──→ w[n] ──→[b0]──→(+)──→ y[n]
          ↑      │               ↑
          │      └→[z⁻¹]→w[n-1]─┤
          │      │    │          │
          │      │   [b1]───────┘
          │      │
         [-a1]───┤
          ↑      └→[z⁻¹]→w[n-2]─┐
          │           │          │
         [-a2]────────┘    [b2]──┘
```

### 8.3 Transposed Direct Form II

Obtained by transposing the signal flow graph of Direct Form II (reversing signal flow, swapping branch points and summing junctions):

$$v_1[n] = b_M x[n] - a_N y[n]$$
$$v_k[n] = v_{k-1}[n-1] + b_{M-k} x[n] - a_{N-k} y[n], \quad k = 2, \ldots, N$$
$$y[n] = v_N[n-1] + b_0 x[n]$$

This form is numerically preferred for floating-point implementations because intermediate results have smaller dynamic range.

### 8.4 Cascade (Second-Order Sections) Form

Factor $H(z)$ into second-order sections (biquads):

$$H(z) = G \prod_{k=1}^{L} \frac{b_{0k} + b_{1k}z^{-1} + b_{2k}z^{-2}}{1 + a_{1k}z^{-1} + a_{2k}z^{-2}}$$

where $L = \lceil N/2 \rceil$.

**Advantages:**
- Each section is a simple second-order filter (biquad)
- Much less sensitive to coefficient quantization
- Easier to tune individual sections
- Standard in audio processing

```python
def demonstrate_filter_structures():
    """Implement and compare different filter structures."""
    # Design a 6th-order Butterworth lowpass
    order = 6
    fs = 8000
    fc = 1000
    b, a = signal.butter(order, fc, fs=fs)

    # Direct Form implementation
    def direct_form_1(b, a, x):
        """Direct Form I implementation."""
        M = len(b) - 1
        N_a = len(a) - 1
        y = np.zeros(len(x))
        x_buf = np.zeros(M + 1)
        y_buf = np.zeros(N_a + 1)

        for n in range(len(x)):
            # Shift buffers
            x_buf[1:] = x_buf[:-1]
            x_buf[0] = x[n]

            # FIR part
            y[n] = np.dot(b, x_buf)

            # IIR part
            if N_a > 0:
                y[n] -= np.dot(a[1:], y_buf[:N_a])

            # Update output buffer
            y_buf[1:] = y_buf[:-1]
            y_buf[0] = y[n]

        return y

    def direct_form_2(b, a, x):
        """Direct Form II (canonical) implementation."""
        M = len(b) - 1
        N_a = len(a) - 1
        K = max(M, N_a)
        y = np.zeros(len(x))
        w = np.zeros(K + 1)

        for n in range(len(x)):
            # Compute w[n]
            w[0] = x[n]
            for k in range(1, N_a + 1):
                w[0] -= a[k] * w[k]

            # Compute y[n]
            y[n] = 0
            for k in range(M + 1):
                y[n] += b[k] * w[k]

            # Shift w
            w[1:] = w[:-1]

        return y

    # Test signal
    N = 200
    t = np.arange(N) / fs
    x = np.sin(2 * np.pi * 500 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t)

    # Compare implementations
    y_scipy = signal.lfilter(b, a, x)
    y_df1 = direct_form_1(b, a, x)
    y_df2 = direct_form_2(b, a, x)

    # Second-order sections (cascade form)
    sos = signal.butter(order, fc, fs=fs, output='sos')
    y_sos = signal.sosfilt(sos, x)

    print("Filter Structure Comparison")
    print("=" * 50)
    print(f"Filter: {order}th-order Butterworth, fc={fc} Hz, fs={fs} Hz")
    print(f"Coefficients b: {len(b)}, a: {len(a)}")
    print(f"\nMax error vs. scipy.lfilter:")
    print(f"  Direct Form I:  {np.max(np.abs(y_df1 - y_scipy)):.2e}")
    print(f"  Direct Form II: {np.max(np.abs(y_df2 - y_scipy)):.2e}")
    print(f"  SOS (cascade):  {np.max(np.abs(y_sos - y_scipy)):.2e}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(t * 1000, x, 'b-', alpha=0.5, label='Input')
    axes[0].plot(t * 1000, y_scipy, 'r-', linewidth=2, label='Filtered (scipy)')
    axes[0].set_title('Input and Filtered Signal')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t * 1000, y_df1 - y_scipy, 'b-', label='DF1 error')
    axes[1].plot(t * 1000, y_df2 - y_scipy, 'r-', label='DF2 error')
    axes[1].plot(t * 1000, y_sos - y_scipy, 'g-', label='SOS error')
    axes[1].set_title('Error vs. scipy.lfilter (numerical precision)')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('filter_structures.png', dpi=150)
    plt.show()

    # Print SOS sections
    print(f"\nSecond-Order Sections (SOS):")
    print(f"Number of sections: {len(sos)}")
    for i, section in enumerate(sos):
        print(f"  Section {i+1}: b={section[:3]}, a={section[3:]}")

demonstrate_filter_structures()
```

### 8.5 Parallel Form

Factor $H(z)$ into a sum of second-order sections:

$$H(z) = c_0 + \sum_{k=1}^{L} \frac{b_{0k} + b_{1k}z^{-1}}{1 + a_{1k}z^{-1} + a_{2k}z^{-2}}$$

Obtained via partial fraction expansion of $H(z)$.

### 8.6 Lattice Structure

For FIR filters, the lattice structure uses **reflection coefficients** $k_m$:

$$f_m[n] = f_{m-1}[n] + k_m \, g_{m-1}[n-1]$$
$$g_m[n] = k_m \, f_{m-1}[n] + g_{m-1}[n-1]$$

with $f_0[n] = g_0[n] = x[n]$.

**Advantages:**
- Stability is guaranteed if $|k_m| < 1$ for all $m$
- Each stage can be tested independently
- Used in speech coding (LPC)

```python
def lattice_filter_demo():
    """Demonstrate lattice FIR filter structure."""
    def fir_to_lattice(b):
        """Convert FIR coefficients to lattice (reflection) coefficients."""
        M = len(b) - 1
        a = np.array(b, dtype=float) / b[0]
        k = np.zeros(M)

        for m in range(M, 0, -1):
            k[m - 1] = a[m]
            if abs(k[m - 1]) >= 1:
                raise ValueError(f"Unstable: |k[{m-1}]| = {abs(k[m-1]):.4f} >= 1")
            # Levinson-Durbin step-down
            a_new = np.zeros(m)
            for i in range(1, m):
                a_new[i] = (a[i] - k[m - 1] * a[m - i]) / (1 - k[m - 1] ** 2)
            a = a_new

        return k

    def lattice_filter(k, x):
        """Apply lattice FIR filter with reflection coefficients k."""
        M = len(k)
        N = len(x)
        y = np.zeros(N)

        # State: g[m] for each stage
        g = np.zeros(M)

        for n in range(N):
            f = x[n]
            g_new = np.zeros(M)

            for m in range(M):
                g_old = g[m] if m < M else 0
                f_new = f + k[m] * g[m]
                g_new[m] = k[m] * f + g[m]
                f = f_new

            # Shift g states
            g[1:] = g_new[:-1]
            g[0] = x[n]

            y[n] = f

        return y

    # Example: simple FIR filter
    b = np.array([1.0, 0.5, -0.3, 0.1])

    try:
        k = fir_to_lattice(b)
        print("Lattice Filter Coefficients")
        print("=" * 40)
        print(f"FIR coefficients: {b}")
        print(f"Lattice (reflection) coefficients: {k}")
        print(f"All |k| < 1: {all(np.abs(k) < 1)}")
    except ValueError as e:
        print(f"Error: {e}")

lattice_filter_demo()
```

### 8.7 Structure Comparison Summary

| Structure | Memory | Multiplies/sample | Sensitivity | Notes |
|-----------|--------|-------------------|-------------|-------|
| Direct Form I | $M + N$ | $M + N + 1$ | Moderate | Simple, straightforward |
| Direct Form II | $\max(M,N)$ | $M + N + 1$ | Higher for IIR | Canonical (min memory) |
| Transposed DF II | $\max(M,N)$ | $M + N + 1$ | Better for float | Preferred for floating-point |
| Cascade (SOS) | $2L$ | $5L$ | Low | Best for fixed-point IIR |
| Parallel | $2L$ | $3L + 1$ | Low | Good for parallel HW |
| Lattice | $M$ | $2M$ | Very low | Stability guaranteed by $|k|<1$ |

---

## 9. Quantization Effects

### 9.1 Sources of Quantization Error

In fixed-point implementations, three types of quantization errors arise:

1. **Coefficient quantization**: Filter coefficients are rounded to the available word length
2. **Input quantization**: ADC quantization noise
3. **Arithmetic quantization**: Rounding after multiplications (product round-off)

### 9.2 Coefficient Quantization

When filter coefficients are quantized (e.g., to 16-bit fixed-point), the poles and zeros shift from their designed positions. This can:
- Change the frequency response
- Make a stable filter unstable
- Increase passband ripple or reduce stopband attenuation

**IIR filters are much more sensitive** to coefficient quantization than FIR filters because small changes in denominator coefficients cause large pole movements, especially for high-order filters.

```python
def coefficient_quantization_effects():
    """Demonstrate the effect of coefficient quantization on filter response."""
    # Design an 8th-order bandpass Butterworth
    order = 8
    fs = 8000
    f_low, f_high = 800, 1200  # Hz
    b, a = signal.butter(order, [f_low, f_high], btype='band', fs=fs)

    # Quantize coefficients to different bit widths
    bit_widths = [32, 16, 12, 10, 8]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    w_ref, H_ref = signal.freqz(b, a, worN=4096, fs=fs)
    axes[0].plot(w_ref, 20 * np.log10(np.abs(H_ref) + 1e-12),
                 'k-', linewidth=2, label='Float64 (reference)')

    for bits in bit_widths:
        # Quantize coefficients
        scale = 2 ** (bits - 1)
        b_q = np.round(b * scale) / scale
        a_q = np.round(a * scale) / scale

        # Check stability
        poles_q = np.roots(a_q)
        stable = all(np.abs(poles_q) < 1)

        w_q, H_q = signal.freqz(b_q, a_q, worN=4096, fs=fs)

        label = f'{bits}-bit'
        if not stable:
            label += ' (UNSTABLE!)'

        axes[0].plot(w_q, 20 * np.log10(np.abs(H_q) + 1e-12),
                     linewidth=1, label=label, alpha=0.8)

    axes[0].set_title(f'Coefficient Quantization Effect ({order}th-order Bandpass)')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_ylim(-80, 10)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Compare with SOS implementation (much more robust)
    sos = signal.butter(order, [f_low, f_high], btype='band', fs=fs,
                        output='sos')

    for bits in [16, 10, 8]:
        scale = 2 ** (bits - 1)
        sos_q = np.round(sos * scale) / scale

        # Check stability of each section
        stable = True
        for section in sos_q:
            poles = np.roots(section[3:])
            if any(np.abs(poles) >= 1):
                stable = False
                break

        w_q, H_q = signal.sosfreqz(sos_q, worN=4096, fs=fs)
        label = f'SOS {bits}-bit'
        if not stable:
            label += ' (UNSTABLE!)'
        axes[1].plot(w_q, 20 * np.log10(np.abs(H_q) + 1e-12),
                     linewidth=1.5, label=label)

    axes[1].plot(w_ref, 20 * np.log10(np.abs(H_ref) + 1e-12),
                 'k-', linewidth=2, label='Reference')
    axes[1].set_title('SOS (Cascade) Form: Much More Robust to Quantization')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_ylim(-80, 10)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quantization_effects.png', dpi=150)
    plt.show()

coefficient_quantization_effects()
```

### 9.3 Overflow and Limit Cycles

**Overflow:** When intermediate results exceed the fixed-point range, causing wrap-around (two's complement) or saturation. Saturation arithmetic is preferred to prevent large errors.

**Limit cycles:** In fixed-point IIR filters, quantization of intermediate values can cause the output to oscillate persistently even after the input has become zero. There are two types:

1. **Granular limit cycles (dead-band effect):** Small-amplitude oscillations around zero due to quantization of recursive computations
2. **Overflow limit cycles:** Large-amplitude oscillations caused by arithmetic overflow in feedback paths

```python
def demonstrate_limit_cycles():
    """Demonstrate limit cycles in quantized IIR filters."""
    # Second-order IIR system
    # y[n] = 1.5*y[n-1] - 0.85*y[n-2] + x[n]
    a1, a2 = -1.5, 0.85

    N = 200
    bits = 8
    scale = 2 ** (bits - 1)

    # Input: impulse followed by zeros
    x = np.zeros(N)
    x[0] = 1.0

    # Float implementation (no quantization)
    y_float = np.zeros(N)
    for n in range(N):
        y_float[n] = x[n]
        if n >= 1:
            y_float[n] -= a1 * y_float[n - 1]
        if n >= 2:
            y_float[n] -= a2 * y_float[n - 2]

    # Fixed-point with rounding
    y_fixed = np.zeros(N)
    for n in range(N):
        y_val = x[n]
        if n >= 1:
            y_val -= a1 * y_fixed[n - 1]
        if n >= 2:
            y_val -= a2 * y_fixed[n - 2]
        # Quantize output
        y_fixed[n] = np.round(y_val * scale) / scale

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(np.arange(N), y_float, 'b-', linewidth=1,
                 label='Float64 (decays to zero)')
    axes[0].set_title('Floating-Point Implementation')
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('y[n]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.arange(N), y_fixed, 'r-', linewidth=1,
                 label=f'{bits}-bit fixed-point (may show limit cycle)')
    axes[1].set_title(f'Fixed-Point Implementation ({bits}-bit)')
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('y[n]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('limit_cycles.png', dpi=150)
    plt.show()

    # Check for limit cycle
    tail = y_fixed[-50:]
    if np.max(np.abs(tail)) > 0.5 / scale:
        print(f"Limit cycle detected! Tail amplitude: {np.max(np.abs(tail)):.6f}")
    else:
        print("No significant limit cycle in this example.")

demonstrate_limit_cycles()
```

### 9.4 Mitigation Strategies

| Problem | Solution |
|---------|----------|
| Coefficient quantization | Use cascade (SOS) form; use higher precision |
| Overflow | Saturation arithmetic; proper scaling |
| Granular limit cycles | Dithering; magnitude truncation |
| Overflow limit cycles | Saturation arithmetic; SOS with proper ordering |
| General sensitivity | Use SOS for IIR; FIR is inherently more robust |

---

## 10. Practical Filter Application Example

```python
def practical_filtering_example():
    """Complete practical example: filtering a noisy ECG-like signal."""
    fs = 500  # Hz (typical ECG sampling rate)
    T = 5.0   # 5 seconds
    N = int(fs * T)
    t = np.arange(N) / fs

    # Simulate ECG-like signal (simplified)
    ecg = np.zeros(N)
    heart_rate = 72  # BPM
    beat_period = fs * 60 / heart_rate

    for beat in range(int(T * heart_rate / 60) + 1):
        beat_start = int(beat * beat_period)
        if beat_start + 50 < N:
            # QRS complex (simplified as a narrow Gaussian)
            idx = np.arange(max(0, beat_start - 25), min(N, beat_start + 25))
            ecg[idx] += np.exp(-0.5 * ((idx - beat_start) / 3) ** 2)
            # T-wave
            if beat_start + 80 < N:
                idx_t = np.arange(beat_start + 40, min(N, beat_start + 80))
                ecg[idx_t] += 0.3 * np.exp(-0.5 * ((idx_t - beat_start - 60) / 10) ** 2)

    # Add noise
    powerline = 0.3 * np.sin(2 * np.pi * 50 * t)  # 50 Hz powerline
    high_freq = 0.1 * np.random.randn(N)  # High-frequency noise
    baseline = 0.2 * np.sin(2 * np.pi * 0.3 * t)  # Baseline wander
    noisy_ecg = ecg + powerline + high_freq + baseline

    # Design filters

    # 1. Notch filter to remove 50 Hz powerline (IIR)
    f_notch = 50
    Q = 30  # Quality factor
    b_notch, a_notch = signal.iirnotch(f_notch, Q, fs=fs)

    # 2. Bandpass filter for ECG (0.5-40 Hz) (FIR)
    b_bp = signal.firwin(101, [0.5, 40], pass_zero=False, fs=fs)

    # 3. Alternative: Butterworth bandpass (IIR)
    b_bp_iir, a_bp_iir = signal.butter(4, [0.5, 40], btype='band', fs=fs)

    # Apply filters
    y_notch = signal.filtfilt(b_notch, a_notch, noisy_ecg)
    y_fir_bp = signal.filtfilt(b_bp, [1.0], y_notch)
    y_iir_bp = signal.filtfilt(b_bp_iir, a_bp_iir, noisy_ecg)

    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))

    time_range = (1.0, 3.0)  # Show 2 seconds
    mask = (t >= time_range[0]) & (t <= time_range[1])

    axes[0].plot(t[mask], ecg[mask], 'b-', linewidth=1)
    axes[0].set_title('Clean ECG Signal')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t[mask], noisy_ecg[mask], 'r-', linewidth=0.5)
    axes[1].set_title('Noisy ECG (50 Hz + HF noise + baseline wander)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t[mask], y_fir_bp[mask], 'g-', linewidth=1, label='FIR bandpass')
    axes[2].plot(t[mask], ecg[mask], 'b--', linewidth=0.5, alpha=0.5,
                 label='Original')
    axes[2].set_title('Filtered (Notch + FIR Bandpass)')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t[mask], y_iir_bp[mask], 'm-', linewidth=1,
                 label='IIR bandpass')
    axes[3].plot(t[mask], ecg[mask], 'b--', linewidth=0.5, alpha=0.5,
                 label='Original')
    axes[3].set_title('Filtered (IIR Butterworth Bandpass)')
    axes[3].set_ylabel('Amplitude')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ecg_filtering.png', dpi=150)
    plt.show()

    # Compare filter responses
    fig, ax = plt.subplots(figsize=(12, 5))
    w_n, H_n = signal.freqz(b_notch, a_notch, worN=4096, fs=fs)
    w_f, H_f = signal.freqz(b_bp, [1.0], worN=4096, fs=fs)
    w_i, H_i = signal.freqz(b_bp_iir, a_bp_iir, worN=4096, fs=fs)

    ax.plot(w_n, 20 * np.log10(np.abs(H_n) + 1e-12), 'r-',
            label='50 Hz Notch')
    ax.plot(w_f, 20 * np.log10(np.abs(H_f) + 1e-12), 'g-',
            label='FIR Bandpass (0.5-40 Hz)')
    ax.plot(w_i, 20 * np.log10(np.abs(H_i) + 1e-12), 'm-',
            label='IIR Bandpass (0.5-40 Hz)')
    ax.set_title('Filter Frequency Responses')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_xlim(0, 100)
    ax.set_ylim(-60, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ecg_filter_responses.png', dpi=150)
    plt.show()

practical_filtering_example()
```

---

## 11. Summary

```
┌─────────────────────────────────────────────────────────────────┐
│               Digital Filter Fundamentals                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Filter Types:                                                   │
│    FIR: y[n] = Σ bk x[n-k]   (non-recursive, always stable)   │
│    IIR: y[n] = Σ bk x[n-k] - Σ ak y[n-k]   (recursive)       │
│                                                                  │
│  Transfer Function:                                              │
│    H(z) = B(z)/A(z)                                             │
│    FIR: H(z) = polynomial (all-zero)                            │
│    IIR: H(z) = rational (poles and zeros)                       │
│                                                                  │
│  Linear Phase:                                                   │
│    FIR with symmetric/anti-symmetric coefficients               │
│    4 types (I-IV) with different constraints                    │
│    Constant group delay → no waveform distortion                │
│                                                                  │
│  Specifications:                                                 │
│    Passband ripple δp, Stopband attenuation δs                  │
│    Transition width Δω = ωs - ωp                                │
│    Trade-off: sharper transition → higher order                 │
│                                                                  │
│  Filter Structures:                                              │
│    Direct Form I/II → simple but sensitive                      │
│    Transposed DF II → better for floating-point                 │
│    Cascade (SOS) → robust to quantization (preferred!)          │
│    Lattice → inherent stability check via |k| < 1              │
│                                                                  │
│  Quantization Effects:                                           │
│    Coefficient quantization → pole/zero shift                   │
│    Arithmetic rounding → noise floor                            │
│    Overflow → saturation arithmetic needed                      │
│    Limit cycles → IIR-specific problem                          │
│    Mitigation: use SOS form + adequate word length              │
│                                                                  │
│  FIR vs. IIR Decision Guide:                                    │
│    Need linear phase? → FIR                                     │
│    Need minimum order? → IIR                                    │
│    Fixed-point implementation? → FIR or SOS-IIR                 │
│    Guaranteed stability? → FIR                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. Exercises

### Exercise 1: FIR Filter Analysis

Given the FIR filter $h[n] = \{1, -2, 3, -2, 1\}$:

**(a)** Write the difference equation and transfer function $H(z)$.

**(b)** Find and plot all zeros. Verify they have a specific symmetry pattern.

**(c)** Plot the magnitude and phase response. Does this filter have linear phase?

**(d)** Classify the filter type (I, II, III, or IV) and identify what types of frequency-selective filtering it can perform (lowpass, highpass, bandpass, bandstop).

**(e)** Filter the signal $x[n] = \cos(0.2\pi n) + \cos(0.8\pi n)$ and explain the result.

### Exercise 2: IIR Filter Design and Analysis

Design a 4th-order Chebyshev Type I lowpass filter with 0.5 dB passband ripple and cutoff at 2 kHz, sampled at 8 kHz.

**(a)** Use `scipy.signal.cheby1` to obtain the filter coefficients.

**(b)** Plot the pole-zero diagram and verify all poles are inside the unit circle.

**(c)** Plot magnitude response, phase response, and group delay.

**(d)** Compare with a Butterworth filter of the same order and cutoff. Which has better stopband attenuation? Which has more constant group delay?

**(e)** Convert to SOS form and verify the frequency response matches.

### Exercise 3: Linear Phase Verification

**(a)** Design a Type I linear phase FIR bandpass filter (51 taps) with passband 1000-2000 Hz at fs = 8000 Hz.

**(b)** Design a Type III linear phase FIR differentiator (51 taps).

**(c)** For each filter, verify numerically that:
- Coefficients satisfy the expected symmetry condition
- Group delay is constant across all frequencies
- Phase response is linear (or piecewise linear with jumps of $\pi$)

### Exercise 4: Filter Structure Implementation

Implement from scratch (without using `scipy.signal.lfilter`):

**(a)** Direct Form I for the filter $H(z) = \frac{1 + 0.5z^{-1}}{1 - 0.9z^{-1} + 0.81z^{-2}}$

**(b)** Direct Form II for the same filter.

**(c)** Cascade form using two biquad sections.

**(d)** Process a 1000-sample test signal with all three implementations and verify they produce identical results (within floating-point precision).

### Exercise 5: Quantization Experiment

Take a 10th-order IIR elliptic bandpass filter (300-3400 Hz at 8000 Hz sampling rate):

**(a)** Implement in Direct Form II with 16-bit coefficient quantization. Is it stable?

**(b)** Implement the same filter in cascade (SOS) form with 16-bit quantization. Is it stable?

**(c)** Plot the frequency response of both quantized implementations vs. the reference (float64). Which structure better preserves the desired response?

**(d)** Find the minimum number of bits required for stability in each structure.

### Exercise 6: Real-Time Filter Simulation

Implement a sample-by-sample filtering function (simulating real-time processing):

```python
class RealtimeFilter:
    def __init__(self, b, a):
        # Initialize state
        pass

    def process_sample(self, x_n):
        # Process one sample, return one output sample
        pass
```

**(a)** Implement using Direct Form II Transposed.

**(b)** Test with a streaming input (process samples one at a time) and verify it matches batch processing with `scipy.signal.lfilter`.

**(c)** Measure the execution time per sample for different filter orders (4, 8, 16, 32).

### Exercise 7: Multi-Rate Filter Bank

Design a 3-band filter bank that splits a signal into low (0-1 kHz), mid (1-3 kHz), and high (3-4 kHz) bands at fs = 8 kHz:

**(a)** Design appropriate bandpass filters (choose FIR or IIR and justify).

**(b)** Apply the filter bank to a test signal containing components in all three bands.

**(c)** Reconstruct the original signal by summing the three filtered outputs.

**(d)** Measure the reconstruction error. What causes imperfect reconstruction?

---

## 13. Further Reading

- Oppenheim, Schafer. *Discrete-Time Signal Processing*, 3rd ed. Chapters 5-6.
- Proakis, Manolakis. *Digital Signal Processing*, 4th ed. Chapters 7-9.
- Mitra, S. K. *Digital Signal Processing: A Computer-Based Approach*, 4th ed. Chapters 8-9.
- Smith, S. W. *The Scientist and Engineer's Guide to DSP*, Chapters 14-20.
- Lyons, R. G. *Understanding Digital Signal Processing*, 3rd ed. Chapters 5-7.
- Jackson, L. B. *Digital Filters and Signal Processing*, Chapter 11 (Quantization effects).

---

**Previous**: [07. Z-Transform](07_Z_Transform.md) | **Next**: [09. FIR Filter Design](09_FIR_Filter_Design.md)
