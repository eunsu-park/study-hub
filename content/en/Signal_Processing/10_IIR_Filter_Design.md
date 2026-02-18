# IIR Filter Design

## Learning Objectives

- Understand the characteristics of classical analog prototype filters (Butterworth, Chebyshev, Elliptic)
- Master the bilinear transform and impulse invariance methods for analog-to-digital conversion
- Learn to determine filter order from frequency-domain specifications
- Design IIR digital filters using Python's `scipy.signal` module
- Verify filter stability using pole-zero analysis
- Compare filter types and select the appropriate design for a given application

---

## Table of Contents

1. [Introduction to IIR Filter Design](#1-introduction-to-iir-filter-design)
2. [Analog Prototype Filters](#2-analog-prototype-filters)
3. [Butterworth Filters](#3-butterworth-filters)
4. [Chebyshev Type I Filters](#4-chebyshev-type-i-filters)
5. [Chebyshev Type II Filters](#5-chebyshev-type-ii-filters)
6. [Elliptic (Cauer) Filters](#6-elliptic-cauer-filters)
7. [Analog-to-Digital Conversion](#7-analog-to-digital-conversion)
8. [Bilinear Transform](#8-bilinear-transform)
9. [Impulse Invariance Method](#9-impulse-invariance-method)
10. [Complete IIR Design Procedure](#10-complete-iir-design-procedure)
11. [Stability Analysis](#11-stability-analysis)
12. [Comparison of Filter Types](#12-comparison-of-filter-types)
13. [Python Implementation](#13-python-implementation)
14. [Exercises](#14-exercises)

---

## 1. Introduction to IIR Filter Design

### 1.1 IIR Filter Structure

An IIR (Infinite Impulse Response) filter has both feedforward and feedback paths. Its transfer function is a ratio of polynomials:

$$H(z) = \frac{B(z)}{A(z)} = \frac{\sum_{k=0}^{M} b_k z^{-k}}{1 + \sum_{k=1}^{N} a_k z^{-k}}$$

The difference equation is:

$$y[n] = \sum_{k=0}^{M} b_k x[n-k] - \sum_{k=1}^{N} a_k y[n-k]$$

### 1.2 Design Approach

Unlike FIR filters (designed directly in the discrete domain), IIR filters are typically designed by:

1. **Starting with an analog prototype** $H_a(s)$ with well-known properties
2. **Transforming** the analog filter to a digital filter $H(z)$

```
┌──────────────────────────────────────────────────────────────────┐
│                 IIR Filter Design Pipeline                       │
│                                                                  │
│  Digital Specs     Analog Specs     Analog Filter    Digital     │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐    ┌────────┐  │
│  │ ωp, ωs   │ ──▶ │ Ωp, Ωs   │ ──▶ │ Ha(s)    │ ──▶│ H(z)   │  │
│  │ δ₁, δ₂   │     │ δ₁, δ₂   │     │          │    │        │  │
│  └──────────┘     └──────────┘     └──────────┘    └────────┘  │
│                                                                  │
│  Step 1:           Step 2:          Step 3:         Step 4:     │
│  Specify digital   Pre-warp to      Design analog   Apply       │
│  requirements      analog specs     prototype       BLT/IIM    │
└──────────────────────────────────────────────────────────────────┘
```

### 1.3 Why Start with Analog Prototypes?

- **Decades of theory**: Butterworth, Chebyshev, and elliptic filter designs are well-established
- **Closed-form solutions**: Pole/zero locations can be computed analytically
- **Optimal properties**: Each type is optimal in a specific sense (flatness, equiripple, minimum order)
- **Frequency transformations**: Lowpass prototypes can be transformed to highpass, bandpass, bandstop

---

## 2. Analog Prototype Filters

### 2.1 Analog Filter Specifications

An analog lowpass filter specification consists of:

- **Passband edge frequency** $\Omega_p$
- **Stopband edge frequency** $\Omega_s$
- **Passband ripple** $R_p$ (dB) or tolerance $\epsilon$
- **Stopband attenuation** $A_s$ (dB)

The relationship between ripple parameter $\epsilon$ and passband ripple:

$$R_p = 10\log_{10}(1 + \epsilon^2) \quad \text{dB}$$

### 2.2 Normalized Lowpass Prototype

All classical designs start with a **normalized lowpass prototype** with passband edge at $\Omega_p = 1$ rad/s. The prototype is then frequency-scaled and transformed to the desired filter type.

**Selectivity factor:**

$$k = \frac{\Omega_p}{\Omega_s} < 1$$

A smaller $k$ means the transition band is wider relative to the passband, making the design easier.

---

## 3. Butterworth Filters

### 3.1 Magnitude Response

The Butterworth filter provides the **maximally flat magnitude response** in the passband. Its squared magnitude response is:

$$|H_a(j\Omega)|^2 = \frac{1}{1 + \left(\Omega / \Omega_c\right)^{2N}}$$

where $N$ is the filter order and $\Omega_c$ is the $-3$ dB cutoff frequency.

**Properties:**
- All derivatives of $|H_a(j\Omega)|^2$ up to order $2N-1$ are zero at $\Omega = 0$
- Monotonically decreasing magnitude
- $|H_a(j\Omega_c)|^2 = 1/2$ ($-3$ dB at cutoff)
- Rolloff rate: $-20N$ dB/decade in the stopband

### 3.2 Pole Locations

The poles of the Butterworth filter lie on a circle of radius $\Omega_c$ in the $s$-plane, equally spaced:

$$s_k = \Omega_c \exp\left[j\frac{\pi}{2N}(2k + N - 1)\right], \quad k = 0, 1, \ldots, 2N-1$$

For a stable (causal) filter, we select only the left-half-plane poles ($\text{Re}(s_k) < 0$).

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def butterworth_poles(N, Omega_c=1.0):
    """Compute Butterworth filter poles in the s-plane."""
    poles = []
    for k in range(2 * N):
        s_k = Omega_c * np.exp(1j * np.pi * (2 * k + N - 1) / (2 * N))
        if np.real(s_k) < 0:  # Left-half plane only
            poles.append(s_k)
    return np.array(poles)

# Visualize pole locations for different orders
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, N in zip(axes, [2, 4, 8]):
    poles = butterworth_poles(N)
    theta = np.linspace(0, 2 * np.pi, 200)

    # Unit circle
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    # Poles
    ax.plot(np.real(poles), np.imag(poles), 'rx', markersize=12, markeredgewidth=2)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title(f'Butterworth N={N} Poles')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 0.5)
    ax.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.savefig('butterworth_poles.png', dpi=150)
plt.close()
```

### 3.3 Order Determination

Given specifications $(\Omega_p, \Omega_s, R_p, A_s)$, the minimum Butterworth order is:

$$N \geq \frac{\log\left(\frac{10^{A_s/10} - 1}{10^{R_p/10} - 1}\right)}{2\log(\Omega_s / \Omega_p)}$$

```python
def butterworth_order(Omega_p, Omega_s, Rp_dB, As_dB):
    """Compute minimum Butterworth filter order."""
    numerator = np.log10((10**(As_dB / 10) - 1) / (10**(Rp_dB / 10) - 1))
    denominator = 2 * np.log10(Omega_s / Omega_p)
    N = int(np.ceil(numerator / denominator))
    return N

# Example
N = butterworth_order(1.0, 2.0, 1.0, 40)
print(f"Minimum Butterworth order: N = {N}")
# Compare with scipy
N_scipy, Wn = signal.buttord(1.0, 2.0, 1.0, 40, analog=True)
print(f"scipy.signal.buttord: N = {N_scipy}, Wn = {Wn:.4f}")
```

### 3.4 Magnitude Response Visualization

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Different orders
orders = [1, 2, 4, 8, 16]
for N in orders:
    b, a = signal.butter(N, 1.0, analog=True)
    w, H = signal.freqs(b, a, worN=np.logspace(-1, 1, 1000))
    H_dB = 20 * np.log10(np.abs(H))

    axes[0].plot(w, H_dB, linewidth=1.5, label=f'N={N}')
    axes[1].plot(w, np.abs(H), linewidth=1.5, label=f'N={N}')

axes[0].set_xlabel('Frequency (rad/s)')
axes[0].set_ylabel('Magnitude (dB)')
axes[0].set_title('Butterworth: Magnitude (dB)')
axes[0].set_xlim(0.1, 10)
axes[0].set_ylim(-80, 5)
axes[0].axhline(-3, color='gray', linestyle=':', alpha=0.5, label='-3 dB')
axes[0].axvline(1.0, color='gray', linestyle=':', alpha=0.5)
axes[0].set_xscale('log')
axes[0].legend()
axes[0].grid(True, which='both', alpha=0.3)

axes[1].set_xlabel('Frequency (rad/s)')
axes[1].set_ylabel('|H(jΩ)|')
axes[1].set_title('Butterworth: Magnitude (Linear)')
axes[1].set_xlim(0, 3)
axes[1].axhline(1/np.sqrt(2), color='gray', linestyle=':', alpha=0.5, label='-3 dB')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('butterworth_responses.png', dpi=150)
plt.close()
```

---

## 4. Chebyshev Type I Filters

### 4.1 Magnitude Response

The Chebyshev Type I filter has an **equiripple passband** and monotonic stopband:

$$|H_a(j\Omega)|^2 = \frac{1}{1 + \epsilon^2 T_N^2(\Omega / \Omega_p)}$$

where $T_N(\cdot)$ is the Chebyshev polynomial of the first kind of degree $N$:

$$T_N(x) = \begin{cases} \cos(N \cos^{-1}(x)), & |x| \leq 1 \\ \cosh(N \cosh^{-1}(x)), & |x| > 1 \end{cases}$$

**Properties:**
- Equiripple in the passband with ripple $\epsilon$
- Passband ripple: $R_p = 10\log_{10}(1 + \epsilon^2)$ dB
- Sharper transition than Butterworth for the same order
- Monotonically decreasing stopband

### 4.2 Chebyshev Polynomials

The first few Chebyshev polynomials:

$$T_0(x) = 1, \quad T_1(x) = x, \quad T_2(x) = 2x^2 - 1$$

$$T_3(x) = 4x^3 - 3x, \quad T_4(x) = 8x^4 - 8x^2 + 1$$

Recurrence relation: $T_{N+1}(x) = 2x \cdot T_N(x) - T_{N-1}(x)$

### 4.3 Order Determination

$$N \geq \frac{\cosh^{-1}\left(\sqrt{\frac{10^{A_s/10} - 1}{10^{R_p/10} - 1}}\right)}{\cosh^{-1}(\Omega_s / \Omega_p)}$$

### 4.4 Pole Locations

The poles of a Chebyshev Type I filter lie on an ellipse in the $s$-plane:

$$s_k = \sigma_k + j\omega_k$$

where:

$$\sigma_k = -\sinh\left(\frac{1}{N}\sinh^{-1}\left(\frac{1}{\epsilon}\right)\right) \sin\left(\frac{(2k-1)\pi}{2N}\right)$$

$$\omega_k = \cosh\left(\frac{1}{N}\sinh^{-1}\left(\frac{1}{\epsilon}\right)\right) \cos\left(\frac{(2k-1)\pi}{2N}\right)$$

for $k = 1, 2, \ldots, N$.

```python
def chebyshev1_analysis(N, Rp_dB=1.0):
    """Analyze Chebyshev Type I filter."""
    epsilon = np.sqrt(10**(Rp_dB / 10) - 1)

    # Pole locations
    poles = []
    for k in range(1, N + 1):
        theta_k = (2 * k - 1) * np.pi / (2 * N)
        sigma_k = -np.sinh(np.arcsinh(1 / epsilon) / N) * np.sin(theta_k)
        omega_k = np.cosh(np.arcsinh(1 / epsilon) / N) * np.cos(theta_k)
        poles.append(sigma_k + 1j * omega_k)

    return np.array(poles), epsilon

# Compare Butterworth and Chebyshev
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for N in [2, 4, 6, 8]:
    # Chebyshev Type I (1 dB ripple)
    b, a = signal.cheby1(N, 1.0, 1.0, analog=True)
    w, H = signal.freqs(b, a, worN=np.logspace(-1, 1, 1000))
    axes[0].plot(w, 20 * np.log10(np.abs(H)), linewidth=1.5, label=f'N={N}')

axes[0].set_xlabel('Frequency (rad/s)')
axes[0].set_ylabel('Magnitude (dB)')
axes[0].set_title('Chebyshev Type I (Rp = 1 dB)')
axes[0].set_xlim(0.1, 10)
axes[0].set_ylim(-80, 5)
axes[0].set_xscale('log')
axes[0].legend()
axes[0].grid(True, which='both', alpha=0.3)

# Pole-zero plot for N=4
poles_butter = butterworth_poles(4)
poles_cheby, _ = chebyshev1_analysis(4, 1.0)

theta = np.linspace(0, 2 * np.pi, 200)
axes[1].plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit circle')
axes[1].plot(np.real(poles_butter), np.imag(poles_butter), 'bx',
             markersize=12, markeredgewidth=2, label='Butterworth')
axes[1].plot(np.real(poles_cheby), np.imag(poles_cheby), 'r+',
             markersize=12, markeredgewidth=2, label='Chebyshev I')
axes[1].set_xlabel('Real')
axes[1].set_ylabel('Imaginary')
axes[1].set_title('Pole Locations: Butterworth vs Chebyshev I (N=4)')
axes[1].set_aspect('equal')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(-1.5, 0.5)
axes[1].set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.savefig('chebyshev1_analysis.png', dpi=150)
plt.close()
```

---

## 5. Chebyshev Type II Filters

### 5.1 Magnitude Response

The Chebyshev Type II (inverse Chebyshev) filter has a **flat passband** and **equiripple stopband**:

$$|H_a(j\Omega)|^2 = \frac{1}{1 + \frac{1}{\epsilon^2 T_N^2(\Omega_s / \Omega)}}$$

**Properties:**
- Monotonically decreasing (flat) passband
- Equiripple behavior in the stopband
- Has both poles and zeros (unlike Type I which has only poles)
- Zeros lie on the $j\Omega$ axis, creating nulls in the stopband

### 5.2 Comparison with Type I

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

N = 5

# Chebyshev Type I (1 dB ripple)
b1, a1 = signal.cheby1(N, 1.0, 1.0, analog=True)
w1, H1 = signal.freqs(b1, a1, worN=np.logspace(-1, 1, 2000))

# Chebyshev Type II (40 dB stopband attenuation)
b2, a2 = signal.cheby2(N, 40, 1.0, analog=True)
w2, H2 = signal.freqs(b2, a2, worN=np.logspace(-1, 1, 2000))

# Butterworth (same order)
b_bw, a_bw = signal.butter(N, 1.0, analog=True)
w_bw, H_bw = signal.freqs(b_bw, a_bw, worN=np.logspace(-1, 1, 2000))

# dB plot
for w, H, name, style in [(w_bw, H_bw, 'Butterworth', 'b-'),
                            (w1, H1, 'Chebyshev I', 'r-'),
                            (w2, H2, 'Chebyshev II', 'g-')]:
    axes[0].plot(w, 20 * np.log10(np.abs(H) + 1e-15), style,
                 linewidth=1.5, label=name)

axes[0].set_xlabel('Frequency (rad/s)')
axes[0].set_ylabel('Magnitude (dB)')
axes[0].set_title(f'Filter Comparison (N={N})')
axes[0].set_xlim(0.1, 10)
axes[0].set_ylim(-80, 5)
axes[0].set_xscale('log')
axes[0].legend()
axes[0].grid(True, which='both', alpha=0.3)

# Passband detail (linear scale)
for w, H, name, style in [(w_bw, H_bw, 'Butterworth', 'b-'),
                            (w1, H1, 'Chebyshev I', 'r-'),
                            (w2, H2, 'Chebyshev II', 'g-')]:
    axes[1].plot(w, np.abs(H), style, linewidth=1.5, label=name)

axes[1].set_xlabel('Frequency (rad/s)')
axes[1].set_ylabel('|H(jΩ)|')
axes[1].set_title('Passband Detail')
axes[1].set_xlim(0, 2)
axes[1].set_ylim(0, 1.2)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chebyshev2_comparison.png', dpi=150)
plt.close()
```

---

## 6. Elliptic (Cauer) Filters

### 6.1 Magnitude Response

The elliptic filter achieves the **minimum order** for given specifications by distributing ripple in both passband and stopband:

$$|H_a(j\Omega)|^2 = \frac{1}{1 + \epsilon^2 R_N^2(\Omega / \Omega_p)}$$

where $R_N(\cdot)$ is a rational Chebyshev (Jacobian elliptic) function.

**Properties:**
- Equiripple in both passband and stopband
- Sharpest possible transition for given order
- Most efficient in terms of order vs specifications
- Both poles and zeros

### 6.2 Order Advantage

For the same specifications, the required filter order satisfies:

$$N_\text{Elliptic} \leq N_\text{Chebyshev} \leq N_\text{Butterworth}$$

```python
def compare_filter_orders(Rp_dB, As_dB, wp, ws):
    """Compare required orders for different filter types."""
    # Butterworth
    N_butter, Wn_butter = signal.buttord(wp, ws, Rp_dB, As_dB, analog=True)

    # Chebyshev Type I
    N_cheby1, Wn_cheby1 = signal.cheb1ord(wp, ws, Rp_dB, As_dB, analog=True)

    # Chebyshev Type II
    N_cheby2, Wn_cheby2 = signal.cheb2ord(wp, ws, Rp_dB, As_dB, analog=True)

    # Elliptic
    N_ellip, Wn_ellip = signal.ellipord(wp, ws, Rp_dB, As_dB, analog=True)

    print(f"Specifications: Rp={Rp_dB} dB, As={As_dB} dB, wp={wp}, ws={ws}")
    print(f"{'Filter Type':<20} {'Order N':<10} {'Natural Freq':>12}")
    print("-" * 45)
    print(f"{'Butterworth':<20} {N_butter:<10} {Wn_butter:>12.4f}")
    print(f"{'Chebyshev I':<20} {N_cheby1:<10} {Wn_cheby1:>12.4f}")
    print(f"{'Chebyshev II':<20} {N_cheby2:<10} {Wn_cheby2:>12.4f}")
    print(f"{'Elliptic':<20} {N_ellip:<10} {Wn_ellip:>12.4f}")

    return N_butter, N_cheby1, N_cheby2, N_ellip

orders = compare_filter_orders(1.0, 60, 1.0, 1.5)
```

### 6.3 Elliptic Filter Visualization

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Same specifications, different filter types
Rp = 1.0  # dB
As = 60   # dB
wp = 2 * np.pi * 1000  # rad/s
ws = 2 * np.pi * 1300  # rad/s

filters = [
    ('Butterworth', signal.buttord, signal.butter),
    ('Chebyshev I', signal.cheb1ord, signal.cheby1),
    ('Chebyshev II', signal.cheb2ord, signal.cheby2),
    ('Elliptic', signal.ellipord, signal.ellip),
]

for ax, (name, ord_func, design_func) in zip(axes.flat, filters):
    N, Wn = ord_func(wp, ws, Rp, As, analog=True)

    if 'Chebyshev I' in name:
        b, a = design_func(N, Rp, Wn, analog=True)
    elif 'Chebyshev II' in name:
        b, a = design_func(N, As, Wn, analog=True)
    elif 'Elliptic' in name:
        b, a = design_func(N, Rp, As, Wn, analog=True)
    else:
        b, a = design_func(N, Wn, analog=True)

    w, H = signal.freqs(b, a, worN=np.linspace(0, 3 * ws, 5000))
    H_dB = 20 * np.log10(np.abs(H) + 1e-15)

    ax.plot(w / (2 * np.pi), H_dB, 'b-', linewidth=1.5)
    ax.axhline(-Rp, color='g', linestyle='--', alpha=0.5, label=f'-{Rp} dB')
    ax.axhline(-As, color='r', linestyle='--', alpha=0.5, label=f'-{As} dB')
    ax.axvline(wp / (2 * np.pi), color='g', linestyle=':', alpha=0.5)
    ax.axvline(ws / (2 * np.pi), color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f'{name} (N={N})')
    ax.set_ylim(-80, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(f'Filter Type Comparison (Rp={Rp} dB, As={As} dB)', fontsize=13)
plt.tight_layout()
plt.savefig('filter_type_comparison.png', dpi=150)
plt.close()
```

---

## 7. Analog-to-Digital Conversion

### 7.1 Overview of Methods

Three primary methods convert analog filters $H_a(s)$ to digital filters $H(z)$:

```
┌─────────────────────────────────────────────────────────────────┐
│            Analog-to-Digital Conversion Methods                  │
├──────────────────┬──────────────────────────────────────────────┤
│ Method           │ Mapping s → z                                │
├──────────────────┼──────────────────────────────────────────────┤
│ Bilinear         │ s = (2/T)(z-1)/(z+1)                        │
│ Transform        │ - No aliasing                                │
│                  │ - Frequency warping (correctable)            │
│                  │ - Most widely used                           │
├──────────────────┼──────────────────────────────────────────────┤
│ Impulse          │ h[n] = T · h_a(nT)                           │
│ Invariance       │ - Preserves impulse response shape           │
│                  │ - Aliasing in frequency domain               │
│                  │ - Only for bandlimited filters (LP, BP)      │
├──────────────────┼──────────────────────────────────────────────┤
│ Matched          │ Map poles: s_k → z_k = e^(s_k T)             │
│ Z-Transform      │ Map zeros: same mapping                      │
│                  │ - Simple but no formal optimality            │
└──────────────────┴──────────────────────────────────────────────┘
```

---

## 8. Bilinear Transform

### 8.1 The Mapping

The bilinear transform (BLT) maps the entire $s$-plane to the $z$-plane via:

$$s = \frac{2}{T} \cdot \frac{z - 1}{z + 1}$$

or equivalently:

$$z = \frac{1 + (T/2)s}{1 - (T/2)s}$$

where $T$ is the sampling period.

### 8.2 Key Properties

**Stability preservation**: The left half of the $s$-plane maps to the interior of the unit circle in the $z$-plane. This ensures that a stable analog filter always produces a stable digital filter.

**Frequency mapping**: The imaginary axis ($s = j\Omega$) maps to the unit circle ($z = e^{j\omega}$) with the nonlinear frequency warping:

$$\omega = 2\arctan\left(\frac{\Omega T}{2}\right) \quad \Leftrightarrow \quad \Omega = \frac{2}{T}\tan\left(\frac{\omega}{2}\right)$$

### 8.3 Frequency Warping

```
Bilinear Transform Frequency Warping:

Ω (analog)     ω (digital)
   ↑              ↑
   │     ╱       π│─────────────────╱
   │   ╱          │              ╱
   │  ╱           │            ╱
   │ ╱            │         ╱      Nonlinear
   │╱             │      ╱        compression
 0 ├───→ ω     0 ├───╱──────────→ Ω
   0    π         0

- Low frequencies: Ω ≈ ω (nearly linear)
- High frequencies: compressed into [0, π]
- Ω = ∞ maps to ω = π
```

```python
# Visualize frequency warping
T = 1.0  # Sampling period
omega = np.linspace(0, np.pi - 0.01, 1000)
Omega = (2 / T) * np.tan(omega / 2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(Omega, omega / np.pi, 'b-', linewidth=2, label='BLT mapping')
axes[0].plot(Omega, Omega * T / np.pi, 'r--', linewidth=1.5, label='Linear (ideal)')
axes[0].set_xlabel('Analog Frequency Ω (rad/s)')
axes[0].set_ylabel('Digital Frequency ω/π')
axes[0].set_title('Bilinear Transform: Frequency Warping')
axes[0].set_xlim(0, 20)
axes[0].set_ylim(0, 1.1)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Warping effect at different frequencies
freq_analog = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
freq_digital = 2 * np.arctan(freq_analog * T / 2)

axes[1].bar(range(len(freq_analog)),
            freq_digital / (freq_analog * T) * 100 - 100,
            tick_label=[f'{f:.1f}' for f in freq_analog])
axes[1].set_xlabel('Analog Frequency Ω (rad/s)')
axes[1].set_ylabel('Warping Error (%)')
axes[1].set_title('Frequency Warping Error')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bilinear_warping.png', dpi=150)
plt.close()
```

### 8.4 Pre-warping

To ensure critical frequencies are mapped correctly, we **pre-warp** the analog specifications:

$$\Omega_p' = \frac{2}{T}\tan\left(\frac{\omega_p}{2}\right), \quad \Omega_s' = \frac{2}{T}\tan\left(\frac{\omega_s}{2}\right)$$

Then design the analog filter with these pre-warped frequencies, and apply the bilinear transform.

### 8.5 Step-by-Step BLT Design

```python
def iir_design_blt(wp_digital, ws_digital, Rp_dB, As_dB, fs, ftype='butter'):
    """
    Complete IIR design using bilinear transform with pre-warping.

    Parameters:
        wp_digital: digital passband edge (Hz)
        ws_digital: digital stopband edge (Hz)
        Rp_dB: passband ripple (dB)
        As_dB: stopband attenuation (dB)
        fs: sampling frequency (Hz)
        ftype: 'butter', 'cheby1', 'cheby2', or 'ellip'

    Returns:
        b, a: digital filter coefficients
        N: filter order
    """
    T = 1 / fs

    # Step 1: Pre-warp digital frequencies to analog
    Omega_p = 2 * fs * np.tan(np.pi * wp_digital / fs)
    Omega_s = 2 * fs * np.tan(np.pi * ws_digital / fs)

    print(f"Pre-warped frequencies: Ωp = {Omega_p:.2f}, Ωs = {Omega_s:.2f} rad/s")

    # Step 2: Determine order
    if ftype == 'butter':
        N, Wn = signal.buttord(Omega_p, Omega_s, Rp_dB, As_dB, analog=True)
        # Step 3: Design analog prototype
        ba, aa = signal.butter(N, Wn, analog=True)
    elif ftype == 'cheby1':
        N, Wn = signal.cheb1ord(Omega_p, Omega_s, Rp_dB, As_dB, analog=True)
        ba, aa = signal.cheby1(N, Rp_dB, Wn, analog=True)
    elif ftype == 'cheby2':
        N, Wn = signal.cheb2ord(Omega_p, Omega_s, Rp_dB, As_dB, analog=True)
        ba, aa = signal.cheby2(N, As_dB, Wn, analog=True)
    elif ftype == 'ellip':
        N, Wn = signal.ellipord(Omega_p, Omega_s, Rp_dB, As_dB, analog=True)
        ba, aa = signal.ellip(N, Rp_dB, As_dB, Wn, analog=True)

    print(f"Analog filter order: N = {N}")

    # Step 4: Bilinear transform
    b, a = signal.bilinear(ba, aa, fs)

    return b, a, N

# Example
fs = 8000
wp = 1000  # Hz
ws = 1500  # Hz
Rp = 1.0   # dB
As = 60    # dB

b, a, N = iir_design_blt(wp, ws, Rp, As, fs, ftype='ellip')
print(f"\nDigital filter coefficients:")
print(f"b = {b}")
print(f"a = {a}")

# Verify
w, H = signal.freqz(b, a, worN=4096, fs=fs)
H_dB = 20 * np.log10(np.abs(H) + 1e-15)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Magnitude
axes[0, 0].plot(w, H_dB, 'b-', linewidth=1.5)
axes[0, 0].axhline(-Rp, color='g', linestyle='--', alpha=0.5, label=f'-{Rp} dB')
axes[0, 0].axhline(-As, color='r', linestyle='--', alpha=0.5, label=f'-{As} dB')
axes[0, 0].axvline(wp, color='g', linestyle=':', alpha=0.5)
axes[0, 0].axvline(ws, color='r', linestyle=':', alpha=0.5)
axes[0, 0].set_xlabel('Frequency (Hz)')
axes[0, 0].set_ylabel('Magnitude (dB)')
axes[0, 0].set_title(f'Elliptic IIR (N={N}): Magnitude')
axes[0, 0].set_ylim(-80, 5)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Phase
axes[0, 1].plot(w, np.unwrap(np.angle(H)) * 180 / np.pi, 'r-', linewidth=1.5)
axes[0, 1].set_xlabel('Frequency (Hz)')
axes[0, 1].set_ylabel('Phase (degrees)')
axes[0, 1].set_title('Phase Response')
axes[0, 1].grid(True, alpha=0.3)

# Group delay
w_gd, gd = signal.group_delay((b, a), w=4096, fs=fs)
axes[1, 0].plot(w_gd, gd, 'g-', linewidth=1.5)
axes[1, 0].set_xlabel('Frequency (Hz)')
axes[1, 0].set_ylabel('Group Delay (samples)')
axes[1, 0].set_title('Group Delay (Non-constant!)')
axes[1, 0].grid(True, alpha=0.3)

# Pole-zero plot
z_zeros, z_poles, k = signal.tf2zpk(b, a)
theta = np.linspace(0, 2 * np.pi, 200)
axes[1, 1].plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
axes[1, 1].plot(np.real(z_zeros), np.imag(z_zeros), 'bo', markersize=10, label='Zeros')
axes[1, 1].plot(np.real(z_poles), np.imag(z_poles), 'rx', markersize=10,
                markeredgewidth=2, label='Poles')
axes[1, 1].set_xlabel('Real')
axes[1, 1].set_ylabel('Imaginary')
axes[1, 1].set_title('Pole-Zero Plot')
axes[1, 1].set_aspect('equal')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('IIR Filter Design via Bilinear Transform', fontsize=13)
plt.tight_layout()
plt.savefig('iir_blt_design.png', dpi=150)
plt.close()
```

---

## 9. Impulse Invariance Method

### 9.1 Principle

The impulse invariance method sets the digital filter's impulse response equal to samples of the analog impulse response:

$$h[n] = T \cdot h_a(nT)$$

where $h_a(t)$ is the analog impulse response and $T$ is the sampling period.

### 9.2 Mapping

If the analog filter has a partial fraction expansion:

$$H_a(s) = \sum_{k=1}^{N} \frac{A_k}{s - s_k}$$

then the digital filter is:

$$H(z) = T \sum_{k=1}^{N} \frac{A_k}{1 - e^{s_k T} z^{-1}}$$

Each analog pole $s_k$ maps to a digital pole $z_k = e^{s_k T}$.

### 9.3 Aliasing Problem

The frequency response of the digital filter is a sum of shifted copies of the analog response:

$$H(e^{j\omega}) = \frac{1}{T} \sum_{k=-\infty}^{\infty} H_a\left(j\frac{\omega - 2\pi k}{T}\right)$$

This causes **aliasing** if $H_a(j\Omega)$ is not bandlimited, which makes impulse invariance unsuitable for highpass and bandstop filters.

### 9.4 Implementation Example

```python
def impulse_invariance(ba, aa, fs):
    """
    Apply impulse invariance method to convert analog filter to digital.

    Parameters:
        ba, aa: analog filter coefficients (transfer function form)
        fs: sampling frequency

    Returns:
        bd, ad: digital filter coefficients
    """
    T = 1 / fs

    # Convert to zeros, poles, gain form
    z_a, p_a, k_a = signal.tf2zpk(ba, aa)

    # Partial fraction expansion
    residues, poles, direct = signal.residue(ba, aa)

    # Map poles: z_k = exp(s_k * T)
    z_poles = np.exp(poles * T)

    # Construct digital transfer function
    # H(z) = T * sum(A_k / (1 - e^(s_k*T) * z^-1))
    bd = np.array([0.0])
    ad = np.array([1.0])

    for A_k, z_k in zip(residues, z_poles):
        # Each term: T * A_k / (1 - z_k * z^-1)
        bd_k = np.array([T * A_k])
        ad_k = np.array([1, -z_k])

        # Combine fractions
        bd_new = np.convolve(bd, ad_k) + np.convolve(bd_k, ad)
        ad_new = np.convolve(ad, ad_k)
        bd = bd_new
        ad = ad_new

    # Ensure real coefficients
    bd = np.real(bd)
    ad = np.real(ad)

    return bd, ad

# Compare BLT and impulse invariance for a Butterworth lowpass
fs = 8000
N = 4
fc = 1000  # Cutoff frequency (Hz)

# Analog prototype
Omega_c = 2 * np.pi * fc
ba, aa = signal.butter(N, Omega_c, analog=True)

# Method 1: Bilinear transform
bd_blt, ad_blt = signal.bilinear(ba, aa, fs)

# Method 2: Impulse invariance
bd_ii, ad_ii = impulse_invariance(ba, aa, fs)

# Compare frequency responses
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

w_blt, H_blt = signal.freqz(bd_blt, ad_blt, worN=4096, fs=fs)
w_ii, H_ii = signal.freqz(bd_ii, ad_ii, worN=4096, fs=fs)

# Also plot the analog response for reference
w_a, H_a = signal.freqs(ba, aa, worN=np.linspace(0, 2 * np.pi * fs / 2, 4096))

axes[0].plot(w_a / (2 * np.pi), 20 * np.log10(np.abs(H_a) + 1e-15),
             'k--', linewidth=1.5, label='Analog', alpha=0.5)
axes[0].plot(w_blt, 20 * np.log10(np.abs(H_blt) + 1e-15),
             'b-', linewidth=1.5, label='Bilinear Transform')
axes[0].plot(w_ii, 20 * np.log10(np.abs(H_ii) + 1e-15),
             'r-', linewidth=1.5, label='Impulse Invariance')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Magnitude (dB)')
axes[0].set_title('BLT vs Impulse Invariance')
axes[0].set_ylim(-80, 5)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Near Nyquist detail
axes[1].plot(w_blt, 20 * np.log10(np.abs(H_blt) + 1e-15),
             'b-', linewidth=1.5, label='BLT')
axes[1].plot(w_ii, 20 * np.log10(np.abs(H_ii) + 1e-15),
             'r-', linewidth=1.5, label='Impulse Invariance')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Magnitude (dB)')
axes[1].set_title('Detail Near Nyquist (aliasing visible)')
axes[1].set_xlim(2000, 4000)
axes[1].set_ylim(-60, -20)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('impulse_invariance.png', dpi=150)
plt.close()
```

---

## 10. Complete IIR Design Procedure

### 10.1 Step-by-Step Procedure

```
┌──────────────────────────────────────────────────────────────────┐
│                  IIR Filter Design Steps                         │
│                                                                  │
│  1. Specify digital requirements:                                │
│     - Passband/stopband edges (ωp, ωs or fp, fs)               │
│     - Passband ripple Rp (dB)                                    │
│     - Stopband attenuation As (dB)                               │
│                                                                  │
│  2. Choose filter type:                                          │
│     - Butterworth: maximally flat                                │
│     - Chebyshev I: equiripple passband                           │
│     - Chebyshev II: equiripple stopband                          │
│     - Elliptic: minimum order                                    │
│                                                                  │
│  3. Choose conversion method (usually BLT)                       │
│                                                                  │
│  4. Pre-warp critical frequencies                                │
│                                                                  │
│  5. Determine minimum analog filter order                        │
│                                                                  │
│  6. Design analog prototype                                      │
│                                                                  │
│  7. Apply analog-to-digital transformation                       │
│                                                                  │
│  8. Verify specifications                                        │
│     - Check passband ripple                                      │
│     - Check stopband attenuation                                 │
│     - Check stability (all poles inside unit circle)             │
│                                                                  │
│  9. Implement (Direct Form II, SOS cascade, etc.)                │
└──────────────────────────────────────────────────────────────────┘
```

### 10.2 Using scipy.signal Direct Design Functions

SciPy provides high-level functions that handle pre-warping internally:

```python
def design_all_types(fs, f_pass, f_stop, Rp, As):
    """Design IIR lowpass filter using all four types."""
    results = {}

    # Direct digital design (scipy handles pre-warping internally)
    # Butterworth
    N_b, Wn_b = signal.buttord(f_pass, f_stop, Rp, As, fs=fs)
    b_b, a_b = signal.butter(N_b, Wn_b, fs=fs)
    results['Butterworth'] = (b_b, a_b, N_b)

    # Chebyshev Type I
    N_c1, Wn_c1 = signal.cheb1ord(f_pass, f_stop, Rp, As, fs=fs)
    b_c1, a_c1 = signal.cheby1(N_c1, Rp, Wn_c1, fs=fs)
    results['Chebyshev I'] = (b_c1, a_c1, N_c1)

    # Chebyshev Type II
    N_c2, Wn_c2 = signal.cheb2ord(f_pass, f_stop, Rp, As, fs=fs)
    b_c2, a_c2 = signal.cheby2(N_c2, As, Wn_c2, fs=fs)
    results['Chebyshev II'] = (b_c2, a_c2, N_c2)

    # Elliptic
    N_e, Wn_e = signal.ellipord(f_pass, f_stop, Rp, As, fs=fs)
    b_e, a_e = signal.ellip(N_e, Rp, As, Wn_e, fs=fs)
    results['Elliptic'] = (b_e, a_e, N_e)

    return results

# Design and compare
fs = 16000
f_pass = 2000
f_stop = 2500
Rp = 0.5
As = 60

results = design_all_types(fs, f_pass, f_stop, Rp, As)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, (name, (b, a, N)) in zip(axes.flat, results.items()):
    w, H = signal.freqz(b, a, worN=4096, fs=fs)
    H_dB = 20 * np.log10(np.abs(H) + 1e-15)

    ax.plot(w, H_dB, 'b-', linewidth=1.5)
    ax.axhline(-Rp, color='g', linestyle='--', alpha=0.5, label=f'Rp = -{Rp} dB')
    ax.axhline(-As, color='r', linestyle='--', alpha=0.5, label=f'As = -{As} dB')
    ax.axvline(f_pass, color='g', linestyle=':', alpha=0.5)
    ax.axvline(f_stop, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f'{name} (N={N})')
    ax.set_ylim(-80, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(f'IIR Filter Comparison (fp={f_pass} Hz, fs_stop={f_stop} Hz)', fontsize=13)
plt.tight_layout()
plt.savefig('iir_all_types.png', dpi=150)
plt.close()

# Print summary
print(f"\n{'Filter Type':<15} {'Order':<8} {'Actual Rp (dB)':<16} {'Actual As (dB)':<16}")
print("-" * 55)
for name, (b, a, N) in results.items():
    w, H = signal.freqz(b, a, worN=4096, fs=fs)
    H_mag = np.abs(H)
    f = w

    # Passband ripple
    pass_idx = f <= f_pass
    Rp_actual = -20 * np.log10(np.min(H_mag[pass_idx]))

    # Stopband attenuation
    stop_idx = f >= f_stop
    As_actual = -20 * np.log10(np.max(H_mag[stop_idx]))

    print(f"{name:<15} {N:<8} {Rp_actual:<16.4f} {As_actual:<16.4f}")
```

---

## 11. Stability Analysis

### 11.1 Stability Criterion

A digital IIR filter is **stable** if and only if all poles of $H(z)$ lie strictly inside the unit circle:

$$|z_k| < 1, \quad \forall k$$

### 11.2 Second-Order Sections (SOS) for Numerical Stability

High-order IIR filters implemented in direct form can suffer from coefficient quantization and numerical precision issues. The solution is to implement as a **cascade of second-order sections (biquads)**:

$$H(z) = \prod_{i=1}^{L} \frac{b_{0i} + b_{1i}z^{-1} + b_{2i}z^{-2}}{1 + a_{1i}z^{-1} + a_{2i}z^{-2}}$$

```python
def stability_analysis(b, a, title=""):
    """Analyze IIR filter stability."""
    # Poles and zeros
    zeros, poles, gain = signal.tf2zpk(b, a)

    # Check stability
    pole_mags = np.abs(poles)
    is_stable = np.all(pole_mags < 1.0)
    max_pole_mag = np.max(pole_mags) if len(poles) > 0 else 0

    print(f"Filter: {title}")
    print(f"  Number of poles: {len(poles)}")
    print(f"  Number of zeros: {len(zeros)}")
    print(f"  Maximum pole magnitude: {max_pole_mag:.6f}")
    print(f"  Stable: {is_stable}")
    print(f"  Stability margin: {1.0 - max_pole_mag:.6f}")

    # Pole-zero plot
    fig, ax = plt.subplots(figsize=(8, 8))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3, label='Unit circle')
    ax.plot(np.real(zeros), np.imag(zeros), 'bo', markersize=10,
            label=f'Zeros ({len(zeros)})')
    ax.plot(np.real(poles), np.imag(poles), 'rx', markersize=10,
            markeredgewidth=2, label=f'Poles ({len(poles)})')

    # Color poles by stability
    for p in poles:
        color = 'green' if np.abs(p) < 1 else 'red'
        circle = plt.Circle((np.real(p), np.imag(p)), 0.03,
                            fill=True, color=color, alpha=0.3)
        ax.add_patch(circle)

    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title(f'Pole-Zero Plot: {title}\n(Stable: {is_stable}, '
                 f'Max |pole| = {max_pole_mag:.4f})')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    plt.tight_layout()
    return fig, is_stable

# Design a high-order filter and check stability
N = 12
b_high, a_high = signal.butter(N, 0.3)  # 12th-order Butterworth
fig, stable = stability_analysis(b_high, a_high, f"Butterworth N={N}")
plt.savefig('stability_analysis.png', dpi=150)
plt.close()

# Convert to SOS form for better numerical stability
sos = signal.tf2sos(b_high, a_high)
print(f"\nSOS representation: {sos.shape[0]} second-order sections")
print(f"SOS coefficients:\n{sos}")
```

### 11.3 Numerical Comparison: tf vs SOS

```python
def compare_tf_vs_sos(order=20):
    """Compare direct form vs SOS implementation for high-order filter."""
    # Design high-order filter
    b, a = signal.butter(order, 0.1)
    sos = signal.butter(order, 0.1, output='sos')

    # Test signal
    np.random.seed(42)
    x = np.random.randn(1000)

    # Filter using tf form
    y_tf = signal.lfilter(b, a, x)

    # Filter using SOS form
    y_sos = signal.sosfilt(sos, x)

    # Compare
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Frequency response comparison
    w_tf, H_tf = signal.freqz(b, a, worN=4096)
    w_sos, H_sos = signal.sosfreqz(sos, worN=4096)

    axes[0, 0].plot(w_tf / np.pi, 20 * np.log10(np.abs(H_tf) + 1e-15),
                     'b-', linewidth=1.5, label='tf form')
    axes[0, 0].plot(w_sos / np.pi, 20 * np.log10(np.abs(H_sos) + 1e-15),
                     'r--', linewidth=1.5, label='SOS form')
    axes[0, 0].set_xlabel('Normalized Frequency (×π)')
    axes[0, 0].set_ylabel('Magnitude (dB)')
    axes[0, 0].set_title('Frequency Response')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Output comparison
    axes[0, 1].plot(y_tf[:200], 'b-', alpha=0.7, label='tf form')
    axes[0, 1].plot(y_sos[:200], 'r--', alpha=0.7, label='SOS form')
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Output')
    axes[0, 1].set_title('Output Signal')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Difference
    diff = np.abs(y_tf - y_sos)
    axes[1, 0].semilogy(diff, 'k-')
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel('|y_tf - y_sos|')
    axes[1, 0].set_title(f'Numerical Difference (max: {np.max(diff):.2e})')
    axes[1, 0].grid(True, alpha=0.3)

    # Pole magnitudes
    zeros_tf, poles_tf, _ = signal.tf2zpk(b, a)
    axes[1, 1].plot(np.abs(poles_tf), 'rx', markersize=10, markeredgewidth=2)
    axes[1, 1].axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Unit circle')
    axes[1, 1].set_xlabel('Pole Index')
    axes[1, 1].set_ylabel('|pole|')
    axes[1, 1].set_title(f'Pole Magnitudes (N={order})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Direct Form vs SOS (Order={order})', fontsize=13)
    plt.tight_layout()
    plt.savefig('tf_vs_sos.png', dpi=150)
    plt.close()

compare_tf_vs_sos(order=20)
```

---

## 12. Comparison of Filter Types

### 12.1 Summary Table

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     IIR Filter Type Comparison                          │
├──────────────┬─────────────┬────────────┬────────────┬────────────────┤
│              │ Butterworth │ Chebyshev I│ Chebyshev II│ Elliptic      │
├──────────────┼─────────────┼────────────┼────────────┼────────────────┤
│ Passband     │ Maximally   │ Equiripple │ Monotonic  │ Equiripple    │
│              │ flat        │            │            │               │
├──────────────┼─────────────┼────────────┼────────────┼────────────────┤
│ Stopband     │ Monotonic   │ Monotonic  │ Equiripple │ Equiripple    │
├──────────────┼─────────────┼────────────┼────────────┼────────────────┤
│ Transition   │ Widest      │ Medium     │ Medium     │ Sharpest      │
│ band         │             │            │            │               │
├──────────────┼─────────────┼────────────┼────────────┼────────────────┤
│ Order for    │ Highest     │ Medium     │ Medium     │ Lowest        │
│ given specs  │             │            │            │               │
├──────────────┼─────────────┼────────────┼────────────┼────────────────┤
│ Group delay  │ Most        │ Less       │ Better than│ Most          │
│              │ uniform     │ uniform    │ Type I     │ non-uniform   │
├──────────────┼─────────────┼────────────┼────────────┼────────────────┤
│ Use case     │ General,    │ Sharp      │ Flat       │ Minimum order,│
│              │ smooth      │ cutoff,    │ passband,  │ tight specs   │
│              │ response    │ ok ripple  │ ok stopband│               │
└──────────────┴─────────────┴────────────┴────────────┴────────────────┘
```

### 12.2 Comprehensive Visual Comparison

```python
def comprehensive_comparison():
    """Complete comparison of all four IIR filter types."""
    fs = 16000
    fp = 2000
    fstop = 3000
    Rp = 1.0
    As = 50

    filters = {}

    # Butterworth
    N, Wn = signal.buttord(fp, fstop, Rp, As, fs=fs)
    sos = signal.butter(N, Wn, fs=fs, output='sos')
    filters['Butterworth'] = (sos, N)

    # Chebyshev I
    N, Wn = signal.cheb1ord(fp, fstop, Rp, As, fs=fs)
    sos = signal.cheby1(N, Rp, Wn, fs=fs, output='sos')
    filters['Chebyshev I'] = (sos, N)

    # Chebyshev II
    N, Wn = signal.cheb2ord(fp, fstop, Rp, As, fs=fs)
    sos = signal.cheby2(N, As, Wn, fs=fs, output='sos')
    filters['Chebyshev II'] = (sos, N)

    # Elliptic
    N, Wn = signal.ellipord(fp, fstop, Rp, As, fs=fs)
    sos = signal.ellip(N, Rp, As, Wn, fs=fs, output='sos')
    filters['Elliptic'] = (sos, N)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'Butterworth': 'blue', 'Chebyshev I': 'red',
              'Chebyshev II': 'green', 'Elliptic': 'purple'}

    # Magnitude response overlay
    ax = axes[0, 0]
    for name, (sos, N) in filters.items():
        w, H = signal.sosfreqz(sos, worN=4096, fs=fs)
        ax.plot(w, 20 * np.log10(np.abs(H) + 1e-15),
                color=colors[name], linewidth=1.5, label=f'{name} (N={N})')
    ax.axhline(-Rp, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(-As, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Magnitude Response')
    ax.set_ylim(-70, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Passband detail
    ax = axes[0, 1]
    for name, (sos, N) in filters.items():
        w, H = signal.sosfreqz(sos, worN=4096, fs=fs)
        ax.plot(w, 20 * np.log10(np.abs(H) + 1e-15),
                color=colors[name], linewidth=1.5, label=f'{name} (N={N})')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Passband Detail')
    ax.set_xlim(0, fp * 1.2)
    ax.set_ylim(-2, 0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Group delay
    ax = axes[1, 0]
    for name, (sos, N) in filters.items():
        b, a = signal.sos2tf(sos)
        w_gd, gd = signal.group_delay((b, a), w=4096, fs=fs)
        ax.plot(w_gd, gd, color=colors[name], linewidth=1.5, label=f'{name}')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Group Delay (samples)')
    ax.set_title('Group Delay')
    ax.set_xlim(0, fs / 2)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Step response
    ax = axes[1, 1]
    step = np.ones(200)
    for name, (sos, N) in filters.items():
        y = signal.sosfilt(sos, step)
        ax.plot(y, color=colors[name], linewidth=1.5, label=f'{name}')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.set_title('Step Response')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Comprehensive IIR Filter Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('comprehensive_comparison.png', dpi=150)
    plt.close()

comprehensive_comparison()
```

---

## 13. Python Implementation

### 13.1 Complete IIR Design Workflow

```python
def complete_iir_workflow(fs, filter_type, band_type, freqs, Rp, As):
    """
    Complete IIR filter design and analysis workflow.

    Parameters:
        fs: sampling frequency (Hz)
        filter_type: 'butter', 'cheby1', 'cheby2', 'ellip'
        band_type: 'lowpass', 'highpass', 'bandpass', 'bandstop'
        freqs: (f_pass, f_stop) or ((fp1, fp2), (fs1, fs2)) for bandpass/stop
        Rp: passband ripple (dB)
        As: stopband attenuation (dB)

    Returns:
        sos: second-order sections
        info: design information
    """
    f_pass, f_stop = freqs

    # Order determination
    ord_funcs = {
        'butter': signal.buttord,
        'cheby1': signal.cheb1ord,
        'cheby2': signal.cheb2ord,
        'ellip': signal.ellipord,
    }
    N, Wn = ord_funcs[filter_type](f_pass, f_stop, Rp, As, fs=fs)

    # Design
    design_funcs = {
        'butter': lambda: signal.butter(N, Wn, btype=band_type, fs=fs, output='sos'),
        'cheby1': lambda: signal.cheby1(N, Rp, Wn, btype=band_type, fs=fs, output='sos'),
        'cheby2': lambda: signal.cheby2(N, As, Wn, btype=band_type, fs=fs, output='sos'),
        'ellip': lambda: signal.ellip(N, Rp, As, Wn, btype=band_type, fs=fs, output='sos'),
    }
    sos = design_funcs[filter_type]()

    info = {
        'filter_type': filter_type,
        'band_type': band_type,
        'order': N,
        'Wn': Wn,
        'Rp': Rp,
        'As': As,
    }

    return sos, info

# Example: Bandpass filter
fs = 44100
sos_bp, info_bp = complete_iir_workflow(
    fs=fs,
    filter_type='ellip',
    band_type='bandpass',
    freqs=([800, 3000], [500, 3500]),
    Rp=0.5,
    As=50
)

print(f"Bandpass filter: {info_bp}")

# Apply to signal
t = np.arange(0, 0.5, 1/fs)
# Speech-like signal: mix of tones + noise
x = (0.5 * np.sin(2*np.pi*200*t) +    # Low freq
     np.sin(2*np.pi*1000*t) +           # Mid freq (passband)
     0.8 * np.sin(2*np.pi*2000*t) +     # Mid freq (passband)
     0.3 * np.sin(2*np.pi*5000*t) +     # High freq
     0.2 * np.random.randn(len(t)))      # Noise

y = signal.sosfilt(sos_bp, x)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Time domain
axes[0].plot(t[:2000] * 1000, x[:2000], 'b-', alpha=0.7, label='Input')
axes[0].plot(t[:2000] * 1000, y[:2000], 'r-', linewidth=1.5, label='Filtered')
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Time Domain: Bandpass Filtering')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Spectrum
N_fft = len(x)
freqs_fft = np.fft.rfftfreq(N_fft, 1/fs)
X = np.abs(np.fft.rfft(x)) / N_fft
Y = np.abs(np.fft.rfft(y)) / N_fft

axes[1].plot(freqs_fft, 20*np.log10(X + 1e-15), 'b-', alpha=0.7, label='Input')
axes[1].plot(freqs_fft, 20*np.log10(Y + 1e-15), 'r-', linewidth=1.5, label='Filtered')
axes[1].axvspan(800, 3000, alpha=0.1, color='green', label='Passband')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Magnitude (dB)')
axes[1].set_title('Frequency Domain')
axes[1].set_xlim(0, 8000)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Filter frequency response
w, H = signal.sosfreqz(sos_bp, worN=4096, fs=fs)
axes[2].plot(w, 20*np.log10(np.abs(H) + 1e-15), 'b-', linewidth=1.5)
axes[2].axvspan(800, 3000, alpha=0.1, color='green', label='Passband')
axes[2].set_xlabel('Frequency (Hz)')
axes[2].set_ylabel('Magnitude (dB)')
axes[2].set_title(f'Filter Response (Elliptic N={info_bp["order"]})')
axes[2].set_xlim(0, 8000)
axes[2].set_ylim(-70, 5)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iir_bandpass_demo.png', dpi=150)
plt.close()
```

### 13.2 Real-Time IIR Filtering with SOS

```python
class RealtimeIIRFilter:
    """Real-time IIR filter using second-order sections."""

    def __init__(self, sos):
        """
        Initialize with SOS coefficients.

        Parameters:
            sos: second-order sections array (L x 6)
        """
        self.sos = np.array(sos, dtype=np.float64)
        self.n_sections = self.sos.shape[0]
        # State variables: 2 delay elements per section
        self.state = np.zeros((self.n_sections, 2))

    def process_sample(self, x):
        """Process a single input sample."""
        for i in range(self.n_sections):
            b0, b1, b2, a0, a1, a2 = self.sos[i]

            # Direct Form II Transposed
            y = b0 * x + self.state[i, 0]
            self.state[i, 0] = b1 * x - a1 * y + self.state[i, 1]
            self.state[i, 1] = b2 * x - a2 * y

            x = y  # Output of this section is input to next

        return y

    def process_block(self, x_block):
        """Process a block of samples."""
        y = np.zeros_like(x_block)
        for n in range(len(x_block)):
            y[n] = self.process_sample(x_block[n])
        return y

    def reset(self):
        """Reset filter state."""
        self.state = np.zeros((self.n_sections, 2))

# Demonstration
sos = signal.butter(6, 1000, fs=8000, output='sos')
filt = RealtimeIIRFilter(sos)

# Process in blocks (simulating real-time)
np.random.seed(42)
x = np.random.randn(1000)
block_size = 64

y_realtime = np.zeros_like(x)
for i in range(0, len(x), block_size):
    block = x[i:i + block_size]
    y_realtime[i:i + len(block)] = filt.process_block(block)

# Compare with scipy batch processing
y_batch = signal.sosfilt(sos, x)

error = np.max(np.abs(y_realtime - y_batch))
print(f"Maximum error between real-time and batch: {error:.2e}")
```

---

## 14. Exercises

### Exercise 1: Butterworth Filter Design

Design a digital Butterworth lowpass filter with:
- Sampling frequency: $f_s = 10000$ Hz
- Passband edge: $f_p = 1500$ Hz with $\leq 0.5$ dB ripple
- Stopband edge: $f_s = 2000$ Hz with $\geq 40$ dB attenuation

(a) Compute the required filter order using the Butterworth order formula.

(b) Pre-warp the digital frequencies to analog frequencies.

(c) Design the analog Butterworth prototype and apply the bilinear transform.

(d) Verify the design meets specifications. Plot magnitude response, phase response, group delay, and pole-zero diagram.

(e) Compare your manual design with `scipy.signal.butter`. Are the results identical?

### Exercise 2: Chebyshev Filter Comparison

For the specifications: $f_s = 8000$ Hz, $f_p = 1000$ Hz, $f_\text{stop} = 1200$ Hz, $R_p = 1$ dB, $A_s = 50$ dB:

(a) Design both Chebyshev Type I and Type II filters.

(b) Compare their frequency responses on a single plot.

(c) Which has better group delay flatness in the passband? Show quantitatively.

(d) Apply both filters to a chirp signal (sweep 100 Hz to 4000 Hz) and compare the outputs.

### Exercise 3: Elliptic Filter for Audio

Design an elliptic lowpass filter for audio anti-aliasing:
- Input sampling rate: 96 kHz (to be downsampled to 48 kHz)
- Passband: 0 to 20 kHz with ripple $\leq 0.01$ dB
- Stopband: 24 kHz to 48 kHz with attenuation $\geq 96$ dB (16-bit precision)

(a) Determine the minimum filter order.

(b) Implement using SOS form.

(c) Plot the magnitude response on both linear and dB scales.

(d) Compute and plot the group delay. Is the group delay variation acceptable for audio?

### Exercise 4: Bilinear Transform by Hand

Given the first-order analog lowpass filter:

$$H_a(s) = \frac{\Omega_c}{s + \Omega_c}$$

(a) Apply the bilinear transform $s = \frac{2}{T}\frac{z-1}{z+1}$ to derive $H(z)$.

(b) For $\Omega_c = 2\pi \times 1000$ rad/s and $f_s = 8000$ Hz, compute the digital filter coefficients $b_0, b_1, a_0, a_1$.

(c) Verify by comparing the frequency response of $H(z)$ with the warped analog response.

(d) Repeat for a second-order Butterworth filter with the same cutoff.

### Exercise 5: Stability Investigation

(a) Design a 16th-order Chebyshev Type I filter ($R_p = 3$ dB, $\omega_c = 0.1\pi$) in both `tf` and `sos` forms.

(b) Compute and display the pole magnitudes for both representations.

(c) Filter a white noise signal using both representations. Is the `tf` form output valid?

(d) At what order does the `tf` form typically break down for Chebyshev Type I filters? Experiment with orders 4, 8, 12, 16, 20, 24.

### Exercise 6: IIR vs FIR Comparison

For the specification: $f_s = 16000$ Hz, $f_p = 3000$ Hz, $f_\text{stop} = 3500$ Hz, $A_s = 60$ dB:

(a) Design both an IIR (elliptic) and FIR (Parks-McClellan) filter.

(b) Compare: filter orders, computational cost per sample, group delay, magnitude response.

(c) Filter a speech-like signal (sum of harmonics at 100, 200, ..., 5000 Hz). Compare the time-domain outputs visually and by computing the SNR improvement.

(d) For which application scenarios would you choose IIR over FIR, and vice versa?

### Exercise 7: Bandstop (Notch) Filter

Design a narrow bandstop filter to remove 60 Hz powerline interference from an ECG signal:
- Sampling frequency: 500 Hz
- Remove: 59-61 Hz
- Passband: 0-55 Hz and 65-250 Hz
- Passband ripple: $\leq 0.1$ dB
- Stopband attenuation: $\geq 30$ dB

(a) Design using an elliptic prototype. What order is needed?

(b) Plot the magnitude response with a zoomed view around 60 Hz.

(c) Generate a synthetic ECG signal with 60 Hz contamination and demonstrate the filter's effectiveness.

(d) Compute and compare the group delay for Butterworth vs elliptic implementations.

---

## References

1. **Oppenheim, A. V., & Schafer, R. W. (2010).** *Discrete-Time Signal Processing* (3rd ed.). Pearson. Chapter 7.
2. **Proakis, J. G., & Manolakis, D. G. (2007).** *Digital Signal Processing* (4th ed.). Pearson. Chapter 11.
3. **Parks, T. W., & Burrus, C. S. (1987).** *Digital Filter Design*. Wiley.
4. **Antoniou, A. (2006).** *Digital Signal Processing: Signals, Systems, and Filters*. McGraw-Hill.
5. **SciPy Documentation** -- Filter design functions: https://docs.scipy.org/doc/scipy/reference/signal.html
6. **Smith, S. W. (1997).** *The Scientist and Engineer's Guide to Digital Signal Processing*. Available free at http://www.dspguide.com/

---

## Navigation

- Previous: [09. FIR Filter Design](09_FIR_Filter_Design.md)
- Next: [11. Multirate Signal Processing](11_Multirate_Processing.md)
- [Back to Overview](00_Overview.md)
