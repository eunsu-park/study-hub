# Continuous Fourier Transform

**Previous**: [03. Fourier Series and Applications](./03_Fourier_Series_and_Applications.md) | **Next**: [05. Sampling and Reconstruction](./05_Sampling_and_Reconstruction.md)

---

The Fourier series decomposes periodic signals into harmonically related sinusoids. But most real-world signals — a speech utterance, a radar pulse, a transient vibration — are **aperiodic**. To analyze these signals in the frequency domain, we need to extend the Fourier series to nonperiodic signals. The result is the **Continuous-Time Fourier Transform (CTFT)**, which maps a signal from the time domain to a continuous frequency-domain representation.

The Fourier transform is one of the most powerful and ubiquitous tools in all of science and engineering. This lesson develops the transform from first principles, catalogs its essential properties, builds up a table of common transform pairs, and shows how it enables frequency-domain analysis and filtering.

**Difficulty**: ⭐⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Derive the Fourier transform as the limit of the Fourier series for aperiodic signals
2. State and prove key CTFT properties (linearity, shifting, scaling, duality, convolution)
3. Compute forward and inverse Fourier transforms of standard signals
4. Apply the convolution theorem to simplify LTI system analysis
5. Interpret frequency-domain representations physically
6. Describe ideal frequency-selective filters (lowpass, highpass, bandpass)
7. Understand bandwidth and the time-bandwidth product (uncertainty principle)
8. Use Python (FFT) for continuous-signal Fourier analysis

---

## Table of Contents

1. [From Fourier Series to Fourier Transform](#1-from-fourier-series-to-fourier-transform)
2. [Definition of the Fourier Transform](#2-definition-of-the-fourier-transform)
3. [Existence Conditions](#3-existence-conditions)
4. [Common Transform Pairs](#4-common-transform-pairs)
5. [Properties of the Fourier Transform](#5-properties-of-the-fourier-transform)
6. [Convolution Theorem](#6-convolution-theorem)
7. [Multiplication Theorem (Windowing)](#7-multiplication-theorem-windowing)
8. [Parseval's Theorem](#8-parsevals-theorem)
9. [Frequency Domain Analysis](#9-frequency-domain-analysis)
10. [Ideal Filters](#10-ideal-filters)
11. [Bandwidth and Time-Bandwidth Product](#11-bandwidth-and-time-bandwidth-product)
12. [Python Examples](#12-python-examples)
13. [Summary](#13-summary)
14. [Exercises](#14-exercises)
15. [References](#15-references)

---

## 1. From Fourier Series to Fourier Transform

### 1.1 The Limiting Process

Consider a periodic signal $\tilde{x}(t)$ with period $T_0$ that equals an aperiodic signal $x(t)$ over one period:

$$\tilde{x}(t) = x(t) \text{ for } -T_0/2 < t < T_0/2$$

The complex Fourier series is:

$$\tilde{x}(t) = \sum_{n=-\infty}^{\infty} c_n e^{jn\omega_0 t}, \quad c_n = \frac{1}{T_0} \int_{-T_0/2}^{T_0/2} x(t) e^{-jn\omega_0 t} \, dt$$

Define the **envelope function**:

$$X(\omega) \equiv T_0 \cdot c_n \bigg|_{\omega = n\omega_0} = \int_{-T_0/2}^{T_0/2} x(t) e^{-j\omega t} \, dt$$

Now let $T_0 \to \infty$. The spacing between harmonics $\omega_0 = 2\pi/T_0 \to 0$, the discrete frequencies $n\omega_0$ become a continuous variable $\omega$, and the sum becomes an integral:

$$\sum_n c_n e^{jn\omega_0 t} = \sum_n \frac{X(n\omega_0)}{T_0} e^{jn\omega_0 t} = \frac{1}{2\pi} \sum_n X(n\omega_0) e^{jn\omega_0 t} \omega_0$$

As $\omega_0 \to d\omega$:

$$x(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} X(\omega) e^{j\omega t} \, d\omega$$

And the integration limit in $X(\omega)$ extends to infinity:

$$X(\omega) = \int_{-\infty}^{\infty} x(t) e^{-j\omega t} \, dt$$

This pair of equations defines the Fourier transform.

### 1.2 Physical Interpretation

| Fourier Series | Fourier Transform |
|---------------|-------------------|
| Periodic signals | Aperiodic signals |
| Discrete spectrum (line spectrum) | Continuous spectrum (spectral density) |
| Coefficients $c_n$ (dimensionless) | $X(\omega)$ has units of [signal $\times$ time] |
| Frequencies at $n\omega_0$ only | All frequencies $\omega \in \mathbb{R}$ |

---

## 2. Definition of the Fourier Transform

### 2.1 Forward Transform (Analysis)

$$X(\omega) = \mathcal{F}\{x(t)\} = \int_{-\infty}^{\infty} x(t) \, e^{-j\omega t} \, dt$$

$X(\omega)$ is called the **Fourier transform**, **spectrum**, or **frequency-domain representation** of $x(t)$.

### 2.2 Inverse Transform (Synthesis)

$$x(t) = \mathcal{F}^{-1}\{X(\omega)\} = \frac{1}{2\pi} \int_{-\infty}^{\infty} X(\omega) \, e^{j\omega t} \, d\omega$$

### 2.3 Alternative Notation Conventions

Three conventions are common in the literature:

| Convention | Forward | Inverse | Used By |
|-----------|---------|---------|---------|
| $\omega$ (angular frequency) | $\int x \, e^{-j\omega t} dt$ | $\frac{1}{2\pi}\int X \, e^{j\omega t} d\omega$ | Oppenheim, Haykin (this course) |
| $f$ (ordinary frequency) | $\int x \, e^{-j2\pi ft} dt$ | $\int X \, e^{j2\pi ft} df$ | Bracewell, engineering practice |
| Symmetric | $\frac{1}{\sqrt{2\pi}}\int x \, e^{-j\omega t} dt$ | $\frac{1}{\sqrt{2\pi}}\int X \, e^{j\omega t} d\omega$ | Physics, quantum mechanics |

The $f$-convention (used by `scipy.fft`) is:

$$X(f) = \int_{-\infty}^{\infty} x(t) \, e^{-j2\pi ft} \, dt, \qquad x(t) = \int_{-\infty}^{\infty} X(f) \, e^{j2\pi ft} \, df$$

Conversion: $X(\omega) = X(f)|_{f=\omega/(2\pi)}$ with $\omega = 2\pi f$.

### 2.4 Magnitude and Phase

Since $X(\omega)$ is generally complex:

$$X(\omega) = |X(\omega)| e^{j\angle X(\omega)}$$

- $|X(\omega)|$ = **magnitude spectrum** (amplitude spectral density)
- $\angle X(\omega)$ = **phase spectrum**
- $|X(\omega)|^2$ = **energy spectral density** (ESD)

For real signals: $X(-\omega) = X^*(\omega)$ (**conjugate symmetry**), so:
- $|X(-\omega)| = |X(\omega)|$ (magnitude is even)
- $\angle X(-\omega) = -\angle X(\omega)$ (phase is odd)

---

## 3. Existence Conditions

### 3.1 Sufficient Condition

The Fourier transform exists if $x(t)$ is **absolutely integrable**:

$$\int_{-\infty}^{\infty} |x(t)| \, dt < \infty$$

This guarantees $|X(\omega)| \leq \int |x(t)| \, dt < \infty$ (the transform is bounded).

### 3.2 Broader Existence

Many important signals (like $\sin(\omega_0 t)$, $u(t)$, $\delta(t)$) are not absolutely integrable but still have Fourier transforms in a generalized sense using **distributions** (generalized functions). Their transforms involve Dirac delta functions:

$$\mathcal{F}\{\delta(t)\} = 1, \qquad \mathcal{F}\{1\} = 2\pi\delta(\omega)$$

$$\mathcal{F}\{e^{j\omega_0 t}\} = 2\pi\delta(\omega - \omega_0)$$

These generalized transforms are essential for handling periodic signals and constants within the Fourier transform framework.

---

## 4. Common Transform Pairs

### 4.1 Table of Fourier Transform Pairs

| Signal $x(t)$ | Transform $X(\omega)$ | Notes |
|:---:|:---:|:---|
| $\delta(t)$ | $1$ | Impulse has all frequencies equally |
| $1$ | $2\pi\delta(\omega)$ | Constant (DC) is a single frequency at $\omega = 0$ |
| $e^{j\omega_0 t}$ | $2\pi\delta(\omega - \omega_0)$ | Complex exponential is a single spectral line |
| $\cos(\omega_0 t)$ | $\pi[\delta(\omega - \omega_0) + \delta(\omega + \omega_0)]$ | Two spectral lines |
| $\sin(\omega_0 t)$ | $\frac{\pi}{j}[\delta(\omega - \omega_0) - \delta(\omega + \omega_0)]$ | Two spectral lines |
| $u(t)$ | $\pi\delta(\omega) + \frac{1}{j\omega}$ | Unit step |
| $e^{-at}u(t)$, $a > 0$ | $\frac{1}{a + j\omega}$ | Causal exponential |
| $e^{-a\|t\|}$, $a > 0$ | $\frac{2a}{a^2 + \omega^2}$ | Two-sided exponential |
| $\text{rect}(t/\tau)$ | $\tau \, \text{sinc}(\omega\tau / 2\pi)$ | Rect $\leftrightarrow$ sinc |
| $\text{sinc}(Wt)$ | $\frac{1}{W}\text{rect}(\omega / (2\pi W))$ | Sinc $\leftrightarrow$ rect (duality) |
| $e^{-t^2/(2\sigma^2)}$ | $\sigma\sqrt{2\pi} \, e^{-\sigma^2\omega^2/2}$ | Gaussian $\leftrightarrow$ Gaussian |
| $\text{tri}(t/\tau)$ | $\tau \, \text{sinc}^2(\omega\tau / 2\pi)$ | Triangle $\leftrightarrow$ sinc$^2$ |
| $\text{sgn}(t)$ | $\frac{2}{j\omega}$ | Sign function |
| $\delta(t - t_0)$ | $e^{-j\omega t_0}$ | Shifted impulse |

### 4.2 Derivation Examples

**Causal exponential** $x(t) = e^{-at}u(t)$, $a > 0$:

$$X(\omega) = \int_0^{\infty} e^{-at} e^{-j\omega t} \, dt = \int_0^{\infty} e^{-(a+j\omega)t} \, dt = \frac{1}{a + j\omega}$$

Magnitude: $|X(\omega)| = \frac{1}{\sqrt{a^2 + \omega^2}}$

Phase: $\angle X(\omega) = -\arctan(\omega/a)$

**Rectangular pulse** $x(t) = \text{rect}(t/\tau)$:

$$X(\omega) = \int_{-\tau/2}^{\tau/2} e^{-j\omega t} \, dt = \frac{e^{j\omega\tau/2} - e^{-j\omega\tau/2}}{j\omega} = \tau \cdot \frac{\sin(\omega\tau/2)}{\omega\tau/2} = \tau \, \text{sinc}\left(\frac{\omega\tau}{2\pi}\right)$$

**Gaussian pulse** $x(t) = e^{-t^2/(2\sigma^2)}$:

$$X(\omega) = \int_{-\infty}^{\infty} e^{-t^2/(2\sigma^2)} e^{-j\omega t} \, dt$$

Complete the square in the exponent:

$$-\frac{t^2}{2\sigma^2} - j\omega t = -\frac{1}{2\sigma^2}(t + j\sigma^2\omega)^2 - \frac{\sigma^2\omega^2}{2}$$

$$X(\omega) = e^{-\sigma^2\omega^2/2} \int_{-\infty}^{\infty} e^{-(t+j\sigma^2\omega)^2/(2\sigma^2)} \, dt = \sigma\sqrt{2\pi} \, e^{-\sigma^2\omega^2/2}$$

The Gaussian is its own Fourier transform (up to scaling). A narrow Gaussian in time gives a wide Gaussian in frequency, and vice versa.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Fourier transform pairs visualization ---

fig, axes = plt.subplots(4, 2, figsize=(14, 16))

# 1. Causal exponential
a = 2.0
t = np.linspace(-1, 5, 1000)
x1 = np.where(t >= 0, np.exp(-a * t), 0.0)
omega = np.linspace(-30, 30, 1000)
X1 = 1 / (a + 1j * omega)

axes[0, 0].plot(t, x1, 'b-', linewidth=2)
axes[0, 0].set_title(f'$x(t) = e^{{-{a}t}}u(t)$')
axes[0, 0].set_xlabel('t')
axes[0, 0].fill_between(t, x1, alpha=0.2)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(omega, np.abs(X1), 'r-', linewidth=2, label='$|X(\\omega)|$')
axes[0, 1].plot(omega, np.angle(X1), 'g--', linewidth=1.5, label='$\\angle X(\\omega)$')
axes[0, 1].set_title(f'$X(\\omega) = 1/({a} + j\\omega)$')
axes[0, 1].set_xlabel('$\\omega$ (rad/s)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 2. Rectangular pulse
tau = 2.0
t2 = np.linspace(-4, 4, 1000)
x2 = np.where(np.abs(t2) <= tau / 2, 1.0, 0.0)
X2 = tau * np.sinc(omega * tau / (2 * np.pi))

axes[1, 0].plot(t2, x2, 'b-', linewidth=2)
axes[1, 0].set_title(f'$x(t) = \\mathrm{{rect}}(t/{tau})$')
axes[1, 0].set_xlabel('t')
axes[1, 0].fill_between(t2, x2, alpha=0.2)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(omega, X2, 'r-', linewidth=2)
axes[1, 1].set_title(f'$X(\\omega) = {tau} \\cdot \\mathrm{{sinc}}(\\omega \\cdot {tau}/(2\\pi))$')
axes[1, 1].set_xlabel('$\\omega$ (rad/s)')
axes[1, 1].grid(True, alpha=0.3)

# 3. Gaussian pulse
for sigma in [0.5, 1.0, 2.0]:
    x3 = np.exp(-t2**2 / (2 * sigma**2))
    X3 = sigma * np.sqrt(2 * np.pi) * np.exp(-sigma**2 * omega**2 / 2)
    axes[2, 0].plot(t2, x3, linewidth=2, label=f'$\\sigma={sigma}$')
    axes[2, 1].plot(omega, X3, linewidth=2, label=f'$\\sigma={sigma}$')

axes[2, 0].set_title('Gaussian: $x(t) = e^{-t^2/(2\\sigma^2)}$')
axes[2, 0].set_xlabel('t')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].set_title('$X(\\omega) = \\sigma\\sqrt{2\\pi} \\cdot e^{-\\sigma^2\\omega^2/2}$')
axes[2, 1].set_xlabel('$\\omega$ (rad/s)')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

# 4. Two-sided exponential
x4 = np.exp(-a * np.abs(t2))
X4 = 2 * a / (a**2 + omega**2)

axes[3, 0].plot(t2, x4, 'b-', linewidth=2)
axes[3, 0].set_title(f'$x(t) = e^{{-{a}|t|}}$')
axes[3, 0].set_xlabel('t')
axes[3, 0].fill_between(t2, x4, alpha=0.2)
axes[3, 0].grid(True, alpha=0.3)

axes[3, 1].plot(omega, X4, 'r-', linewidth=2)
axes[3, 1].set_title(f'$X(\\omega) = {2*a}/({a**2} + \\omega^2)$')
axes[3, 1].set_xlabel('$\\omega$ (rad/s)')
axes[3, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transform_pairs.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 5. Properties of the Fourier Transform

The properties of the Fourier transform are immensely powerful — they allow us to determine transforms of complex signals from simpler ones without recomputing integrals.

### 5.1 Linearity

$$\mathcal{F}\{ax_1(t) + bx_2(t)\} = aX_1(\omega) + bX_2(\omega)$$

Superposition holds in the frequency domain.

### 5.2 Time Shifting

$$\mathcal{F}\{x(t - t_0)\} = e^{-j\omega t_0} X(\omega)$$

A time delay adds a **linear phase** $-\omega t_0$ to the spectrum. The **magnitude spectrum is unchanged** — shifting a signal in time does not change its frequency content, only the phase relationships.

**Proof**:

$$\int x(t - t_0) e^{-j\omega t} dt \overset{\tau = t - t_0}{=} \int x(\tau) e^{-j\omega(\tau + t_0)} d\tau = e^{-j\omega t_0} X(\omega)$$

### 5.3 Frequency Shifting (Modulation)

$$\mathcal{F}\{x(t) e^{j\omega_0 t}\} = X(\omega - \omega_0)$$

Multiplying by a complex exponential in time **shifts the spectrum** by $\omega_0$. This is the mathematical basis of **modulation** in communications.

**Corollary** (cosine modulation):

$$\mathcal{F}\{x(t)\cos(\omega_0 t)\} = \frac{1}{2}[X(\omega - \omega_0) + X(\omega + \omega_0)]$$

The spectrum is shifted to $\pm\omega_0$ and halved in amplitude.

### 5.4 Time Scaling

$$\mathcal{F}\{x(at)\} = \frac{1}{|a|} X\left(\frac{\omega}{a}\right)$$

- Compressing in time ($|a| > 1$) **expands** in frequency
- Expanding in time ($|a| < 1$) **compresses** in frequency
- The area under $|X(\omega)|$ is scaled by $1/|a|$ to conserve energy

This is the mathematical expression of the **uncertainty principle**: you cannot be arbitrarily localized in both time and frequency simultaneously.

### 5.5 Time Reversal

$$\mathcal{F}\{x(-t)\} = X(-\omega)$$

A special case of scaling with $a = -1$. For real signals, $X(-\omega) = X^*(\omega)$, so:

$$\mathcal{F}\{x(-t)\} = X^*(\omega) \quad \text{(real signals)}$$

### 5.6 Duality

If $x(t) \leftrightarrow X(\omega)$, then:

$$X(t) \leftrightarrow 2\pi \, x(-\omega)$$

**Example**: Since $\text{rect}(t/\tau) \leftrightarrow \tau\,\text{sinc}(\omega\tau/(2\pi))$, by duality:

$$\tau\,\text{sinc}(\tau t/(2\pi)) \leftrightarrow 2\pi \, \text{rect}(-\omega/\tau) = 2\pi\,\text{rect}(\omega/\tau)$$

This is how we obtain the transform of the sinc function.

### 5.7 Differentiation in Time

$$\mathcal{F}\left\{\frac{d^n x}{dt^n}\right\} = (j\omega)^n X(\omega)$$

Differentiation in time corresponds to multiplication by $j\omega$ in frequency. High frequencies are amplified — differentiation is a **highpass operation**.

### 5.8 Integration

$$\mathcal{F}\left\{\int_{-\infty}^{t} x(\tau) \, d\tau\right\} = \frac{X(\omega)}{j\omega} + \pi X(0)\delta(\omega)$$

Integration corresponds to division by $j\omega$ (plus a DC term). Low frequencies are amplified — integration is a **lowpass operation**.

### 5.9 Differentiation in Frequency

$$\mathcal{F}\{(-jt)^n x(t)\} = \frac{d^n X(\omega)}{d\omega^n}$$

or equivalently:

$$\mathcal{F}\{t^n x(t)\} = j^n \frac{d^n X(\omega)}{d\omega^n}$$

### 5.10 Property Summary Table

| Property | Time Domain | Frequency Domain |
|----------|------------|-----------------|
| Linearity | $ax_1 + bx_2$ | $aX_1 + bX_2$ |
| Time shift | $x(t - t_0)$ | $e^{-j\omega t_0}X(\omega)$ |
| Freq shift | $x(t)e^{j\omega_0 t}$ | $X(\omega - \omega_0)$ |
| Scaling | $x(at)$ | $\frac{1}{|a|}X(\omega/a)$ |
| Reversal | $x(-t)$ | $X(-\omega)$ |
| Duality | $X(t)$ | $2\pi x(-\omega)$ |
| Time diff | $\frac{dx}{dt}$ | $j\omega X(\omega)$ |
| Freq diff | $(-jt)x(t)$ | $\frac{dX}{d\omega}$ |
| Convolution | $x * h$ | $X \cdot H$ |
| Multiplication | $x \cdot w$ | $\frac{1}{2\pi}X * W$ |
| Conjugate | $x^*(t)$ | $X^*(-\omega)$ |
| Parseval | $\int|x|^2 dt$ | $\frac{1}{2\pi}\int|X|^2 d\omega$ |

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Fourier transform properties demonstration ---

# Use FFT to approximate the CTFT
def approx_ctft(x, t):
    """Approximate CTFT using FFT."""
    dt = t[1] - t[0]
    N = len(t)
    X = np.fft.fftshift(np.fft.fft(x)) * dt
    omega = np.fft.fftshift(np.fft.fftfreq(N, dt)) * 2 * np.pi
    return omega, X

# Setup
fs = 1000
t = np.arange(-5, 5, 1 / fs)

# Original signal: Gaussian pulse
sigma = 0.5
x = np.exp(-t**2 / (2 * sigma**2))
omega, X = approx_ctft(x, t)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# 1. Time shifting
t0 = 1.5
x_shifted = np.exp(-(t - t0)**2 / (2 * sigma**2))
_, X_shifted = approx_ctft(x_shifted, t)

axes[0, 0].plot(t, x, 'b-', linewidth=2, label='$x(t)$')
axes[0, 0].plot(t, x_shifted, 'r--', linewidth=2, label=f'$x(t - {t0})$')
axes[0, 0].set_title('Time Shifting')
axes[0, 0].legend()
axes[0, 0].set_xlabel('t')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(omega, np.abs(X), 'b-', linewidth=2, label='$|X(\\omega)|$')
axes[0, 1].plot(omega, np.abs(X_shifted), 'r--', linewidth=2,
                label='$|X_{shifted}(\\omega)|$')
axes[0, 1].set_title('Magnitude unchanged by time shift')
axes[0, 1].legend()
axes[0, 1].set_xlabel('$\\omega$')
axes[0, 1].set_xlim([-20, 20])
axes[0, 1].grid(True, alpha=0.3)

# 2. Time scaling
x_compressed = np.exp(-(2 * t)**2 / (2 * sigma**2))
x_expanded = np.exp(-(0.5 * t)**2 / (2 * sigma**2))
_, X_comp = approx_ctft(x_compressed, t)
_, X_exp = approx_ctft(x_expanded, t)

axes[1, 0].plot(t, x, 'b-', linewidth=2, label='$x(t)$')
axes[1, 0].plot(t, x_compressed, 'r--', linewidth=2, label='$x(2t)$ compressed')
axes[1, 0].plot(t, x_expanded, 'g--', linewidth=2, label='$x(0.5t)$ expanded')
axes[1, 0].set_title('Time Scaling')
axes[1, 0].legend()
axes[1, 0].set_xlabel('t')
axes[1, 0].set_xlim([-4, 4])
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(omega, np.abs(X), 'b-', linewidth=2, label='$|X(\\omega)|$')
axes[1, 1].plot(omega, np.abs(X_comp), 'r--', linewidth=2,
                label='$|X_{comp}(\\omega)|$ expanded')
axes[1, 1].plot(omega, np.abs(X_exp), 'g--', linewidth=2,
                label='$|X_{exp}(\\omega)|$ compressed')
axes[1, 1].set_title('Compression in time = expansion in frequency')
axes[1, 1].legend(fontsize=8)
axes[1, 1].set_xlabel('$\\omega$')
axes[1, 1].set_xlim([-20, 20])
axes[1, 1].grid(True, alpha=0.3)

# 3. Frequency shifting (modulation)
omega_c = 20.0  # carrier frequency
x_mod = x * np.cos(omega_c * t)
_, X_mod = approx_ctft(x_mod, t)

axes[2, 0].plot(t, x, 'b-', linewidth=1, alpha=0.5, label='$x(t)$ envelope')
axes[2, 0].plot(t, x_mod, 'r-', linewidth=1, label='$x(t)\\cos(\\omega_c t)$')
axes[2, 0].set_title(f'Frequency Shifting (modulation, $\\omega_c={omega_c}$)')
axes[2, 0].legend()
axes[2, 0].set_xlabel('t')
axes[2, 0].set_xlim([-3, 3])
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].plot(omega, np.abs(X), 'b-', linewidth=2, label='Original $|X(\\omega)|$')
axes[2, 1].plot(omega, np.abs(X_mod), 'r-', linewidth=2,
                label='$|X_{mod}(\\omega)|$')
axes[2, 1].set_title('Spectrum shifted to $\\pm\\omega_c$')
axes[2, 1].legend()
axes[2, 1].set_xlabel('$\\omega$')
axes[2, 1].set_xlim([-40, 40])
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ft_properties.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6. Convolution Theorem

### 6.1 Statement

$$\mathcal{F}\{x(t) * h(t)\} = X(\omega) \cdot H(\omega)$$

**Convolution in time is multiplication in frequency.**

This is perhaps the single most important property of the Fourier transform. It means that the output of an LTI system can be computed by:

1. Transform the input: $x(t) \to X(\omega)$
2. Multiply by the frequency response: $Y(\omega) = X(\omega) \cdot H(\omega)$
3. Inverse transform: $Y(\omega) \to y(t)$

### 6.2 Proof

$$\mathcal{F}\{x * h\} = \int \left[\int x(\tau) h(t - \tau) d\tau\right] e^{-j\omega t} dt$$

Swap integration order:

$$= \int x(\tau) \left[\int h(t - \tau) e^{-j\omega t} dt\right] d\tau$$

Inner integral: let $u = t - \tau$:

$$= \int x(\tau) \left[\int h(u) e^{-j\omega(u + \tau)} du\right] d\tau = \int x(\tau) e^{-j\omega\tau} d\tau \cdot \int h(u) e^{-j\omega u} du$$

$$= X(\omega) \cdot H(\omega)$$

### 6.3 LTI System Analysis in the Frequency Domain

For an LTI system with impulse response $h(t)$ and input $x(t)$:

$$Y(\omega) = H(\omega) \cdot X(\omega)$$

where $H(\omega) = \mathcal{F}\{h(t)\}$ is the **frequency response** (also called the **transfer function**).

The output spectrum is shaped by the frequency response:
- At frequencies where $|H(\omega)| > 1$: amplification
- At frequencies where $|H(\omega)| < 1$: attenuation
- At frequencies where $|H(\omega)| = 0$: complete removal (null)

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Convolution theorem demonstration ---

fs = 1000
t = np.arange(-5, 5, 1 / fs)
N = len(t)
dt = 1 / fs

# Input: sum of two frequencies
f1, f2 = 3, 15  # Hz
x = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

# LTI system: lowpass (exponential decay)
a = 10 * 2 * np.pi  # time constant
h = np.where(t >= 0, a * np.exp(-a * t), 0.0)
h /= np.sum(h) * dt  # normalize for unity DC gain

# Time-domain convolution
y_time = np.convolve(x, h, mode='full')[:N] * dt

# Frequency-domain multiplication
omega = np.fft.fftshift(np.fft.fftfreq(N, dt)) * 2 * np.pi
X = np.fft.fftshift(np.fft.fft(x)) * dt
H = np.fft.fftshift(np.fft.fft(h)) * dt
Y_freq = X * H
y_freq = np.real(np.fft.ifft(np.fft.ifftshift(Y_freq))) / dt

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Time domain signals
axes[0, 0].plot(t, x, 'b-', linewidth=1)
axes[0, 0].set_title('Input $x(t)$: 3 Hz + 15 Hz')
axes[0, 0].set_xlabel('t (s)')
axes[0, 0].set_xlim([-1, 2])
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t, h, 'r-', linewidth=2)
axes[0, 1].set_title('Impulse Response $h(t)$: Lowpass')
axes[0, 1].set_xlabel('t (s)')
axes[0, 1].set_xlim([-0.1, 0.5])
axes[0, 1].grid(True, alpha=0.3)

# Frequency domain
f_hz = omega / (2 * np.pi)
axes[1, 0].plot(f_hz, np.abs(X), 'b-', linewidth=1.5)
axes[1, 0].set_title('Input Spectrum $|X(\\omega)|$')
axes[1, 0].set_xlabel('Frequency (Hz)')
axes[1, 0].set_xlim([-30, 30])
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(f_hz, np.abs(H), 'r-', linewidth=2)
axes[1, 1].set_title('Frequency Response $|H(\\omega)|$')
axes[1, 1].set_xlabel('Frequency (Hz)')
axes[1, 1].set_xlim([-30, 30])
axes[1, 1].grid(True, alpha=0.3)

# Output comparison
axes[2, 0].plot(t, y_time, 'g-', linewidth=1.5, label='Time-domain conv')
axes[2, 0].plot(t, y_freq, 'k--', linewidth=1, label='Freq-domain mult')
axes[2, 0].set_title('Output $y(t) = x * h$ (both methods agree)')
axes[2, 0].set_xlabel('t (s)')
axes[2, 0].set_xlim([-1, 2])
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].plot(f_hz, np.abs(Y_freq), 'g-', linewidth=2)
axes[2, 1].set_title('Output Spectrum $|Y(\\omega)| = |X \\cdot H|$')
axes[2, 1].set_xlabel('Frequency (Hz)')
axes[2, 1].set_xlim([-30, 30])
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('convolution_theorem.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 7. Multiplication Theorem (Windowing)

### 7.1 Statement

$$\mathcal{F}\{x(t) \cdot w(t)\} = \frac{1}{2\pi} X(\omega) * W(\omega)$$

**Multiplication in time is convolution in frequency** (scaled by $1/(2\pi)$).

### 7.2 Windowing

When we observe a signal for a finite duration $[-T/2, T/2]$, we are implicitly multiplying it by a rectangular window $w(t) = \text{rect}(t/T)$.

The observed spectrum is not the true $X(\omega)$ but the **convolved** (smeared) spectrum:

$$X_{\text{obs}}(\omega) = \frac{1}{2\pi} X(\omega) * W(\omega)$$

where $W(\omega) = T\,\text{sinc}(\omega T/(2\pi))$ is the transform of the rectangular window.

This convolution:
- **Broadens** spectral peaks (reduces frequency resolution)
- Creates **sidelobes** (spectral leakage from the sinc function)

### 7.3 Window Functions

Different window functions trade off between main-lobe width (resolution) and sidelobe level (leakage):

| Window | Main Lobe Width | First Sidelobe (dB) | Use Case |
|--------|----------------|---------------------|----------|
| Rectangular | Narrowest | -13 | Maximum resolution |
| Hamming | 1.8x rect | -42 | General purpose |
| Hanning | 2.0x rect | -31 | General purpose |
| Blackman | 2.9x rect | -58 | Low leakage |
| Kaiser ($\beta$) | Variable | Variable | Tunable tradeoff |

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- Windowing effect on spectrum ---

fs = 1000
N = 1024
t = np.arange(N) / fs

# Signal: two close sinusoids
f1, f2 = 50, 55  # Hz (5 Hz apart)
x = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

# Window functions
windows = {
    'Rectangular': np.ones(N),
    'Hamming': np.hamming(N),
    'Hanning': np.hanning(N),
    'Blackman': np.blackman(N),
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

freq = np.fft.rfftfreq(N, 1 / fs)

for ax, (name, w) in zip(axes.flat, windows.items()):
    x_windowed = x * w
    X = np.abs(np.fft.rfft(x_windowed)) / np.sum(w) * 2

    ax.plot(freq, 20 * np.log10(X + 1e-12), linewidth=1.5)
    ax.set_title(f'{name} Window')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_xlim([20, 80])
    ax.set_ylim([-80, 5])
    ax.axvline(x=f1, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=f2, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3)

plt.suptitle('Effect of Windowing on Spectral Analysis', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('windowing_effect.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 8. Parseval's Theorem

### 8.1 Statement (Energy Version)

$$E_x = \int_{-\infty}^{\infty} |x(t)|^2 \, dt = \frac{1}{2\pi} \int_{-\infty}^{\infty} |X(\omega)|^2 \, d\omega$$

Or using ordinary frequency $f$:

$$\int_{-\infty}^{\infty} |x(t)|^2 \, dt = \int_{-\infty}^{\infty} |X(f)|^2 \, df$$

### 8.2 Interpretation

- **Left side**: Total signal energy (computed in the time domain)
- **Right side**: Total energy computed from the spectral density

$|X(\omega)|^2$ is the **Energy Spectral Density (ESD)** — it tells us how much energy is contained in each infinitesimal frequency band.

### 8.3 Generalized Parseval (Rayleigh's Theorem)

For two signals:

$$\int_{-\infty}^{\infty} x(t) y^*(t) \, dt = \frac{1}{2\pi} \int_{-\infty}^{\infty} X(\omega) Y^*(\omega) \, d\omega$$

### 8.4 Energy in a Frequency Band

The energy in the frequency band $[\omega_1, \omega_2]$ is:

$$E_{[\omega_1, \omega_2]} = \frac{1}{2\pi} \int_{\omega_1}^{\omega_2} |X(\omega)|^2 \, d\omega + \frac{1}{2\pi} \int_{-\omega_2}^{-\omega_1} |X(\omega)|^2 \, d\omega$$

For real signals with conjugate symmetry:

$$E_{[\omega_1, \omega_2]} = \frac{1}{\pi} \int_{\omega_1}^{\omega_2} |X(\omega)|^2 \, d\omega$$

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Parseval's theorem verification ---

fs = 1000
t = np.arange(-5, 5, 1 / fs)
dt = 1 / fs

# Gaussian pulse
sigma = 0.3
x = np.exp(-t**2 / (2 * sigma**2))

# Time-domain energy
E_time = np.trapz(np.abs(x)**2, t)

# Frequency-domain energy
N = len(t)
omega = np.fft.fftshift(np.fft.fftfreq(N, dt)) * 2 * np.pi
X = np.fft.fftshift(np.fft.fft(x)) * dt
E_freq = np.trapz(np.abs(X)**2, omega) / (2 * np.pi)

# Analytical energy
E_analytical = sigma * np.sqrt(np.pi)

print(f"=== Parseval's Theorem Verification ===")
print(f"Gaussian pulse with sigma = {sigma}")
print(f"Time-domain energy:      {E_time:.8f}")
print(f"Frequency-domain energy: {E_freq:.8f}")
print(f"Analytical energy:       {E_analytical:.8f}")
print(f"Error (time vs freq):    {abs(E_time - E_freq):.2e}")

# Energy distribution across frequency bands
ESD = np.abs(X)**2 / (2 * np.pi)  # energy spectral density

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(t, x, 'b-', linewidth=2)
axes[0].fill_between(t, x, alpha=0.2)
axes[0].set_title(f'$x(t)$: Gaussian ($\\sigma = {sigma}$)')
axes[0].set_xlabel('t')
axes[0].grid(True, alpha=0.3)

axes[1].plot(omega, np.abs(X)**2, 'r-', linewidth=2)
axes[1].fill_between(omega, np.abs(X)**2, alpha=0.2, color='red')
axes[1].set_title('Energy Spectral Density $|X(\\omega)|^2$')
axes[1].set_xlabel('$\\omega$ (rad/s)')
axes[1].set_xlim([-30, 30])
axes[1].grid(True, alpha=0.3)

# Cumulative energy as function of bandwidth
omega_pos = omega[omega >= 0]
ESD_pos = np.abs(X[omega >= 0])**2
cum_energy = np.cumsum(ESD_pos) * (omega_pos[1] - omega_pos[0]) / np.pi
axes[2].plot(omega_pos, cum_energy / E_time * 100, 'g-', linewidth=2)
axes[2].axhline(y=90, color='gray', linestyle='--', label='90%')
axes[2].axhline(y=99, color='lightgray', linestyle='--', label='99%')
axes[2].set_title("Cumulative Energy vs Bandwidth (Parseval's)")
axes[2].set_xlabel('$\\omega$ (rad/s)')
axes[2].set_ylabel('% of total energy')
axes[2].set_xlim([0, 30])
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('parseval_ctft.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 9. Frequency Domain Analysis

### 9.1 What the Spectrum Tells Us

The Fourier transform decomposes a signal into its constituent frequencies:

| Spectral Feature | Time-Domain Interpretation |
|-----------------|---------------------------|
| Peak at $\omega_0$ | Dominant oscillation at frequency $\omega_0$ |
| Wide main lobe | Short-duration pulse (time-bandwidth tradeoff) |
| Narrow main lobe | Long-duration or periodic signal |
| Flat spectrum | Impulsive signal (all frequencies present) |
| Rapid phase changes | Abrupt signal transitions |
| Symmetric magnitude | Real-valued signal |

### 9.2 Spectral Analysis of Composite Signals

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Spectral analysis of various signals ---

fs = 4000
duration = 2.0
t = np.arange(0, duration, 1 / fs)
N = len(t)

signals = {
    'Pure tone (440 Hz)': np.sin(2 * np.pi * 440 * t),
    'Two tones (440 + 880 Hz)': np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t),
    'AM signal (carrier 500 Hz)': (1 + 0.5 * np.cos(2 * np.pi * 50 * t)) * np.cos(2 * np.pi * 500 * t),
    'Chirp (100 to 1000 Hz)': np.sin(2 * np.pi * (100 * t + 450 * t**2 / (2 * duration))),
    'Gaussian pulse (t=1s)': np.exp(-(t - 1.0)**2 / (2 * 0.01**2)),
    'White noise': np.random.randn(N),
}

fig, axes = plt.subplots(len(signals), 2, figsize=(16, 3.5 * len(signals)))

freq = np.fft.rfftfreq(N, 1 / fs)

for row, (name, x) in enumerate(signals.items()):
    # Time domain
    axes[row, 0].plot(t[:2000], x[:2000], 'b-', linewidth=0.8)
    axes[row, 0].set_title(f'{name}')
    axes[row, 0].set_xlabel('Time (s)')
    axes[row, 0].set_ylabel('Amplitude')
    axes[row, 0].grid(True, alpha=0.3)

    # Frequency domain
    X = np.abs(np.fft.rfft(x * np.hanning(N))) * 2 / N
    axes[row, 1].plot(freq, 20 * np.log10(X + 1e-12), 'r-', linewidth=0.8)
    axes[row, 1].set_title(f'Spectrum of {name}')
    axes[row, 1].set_xlabel('Frequency (Hz)')
    axes[row, 1].set_ylabel('Magnitude (dB)')
    axes[row, 1].set_xlim([0, 1500])
    axes[row, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spectral_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 9.3 LTI System Analysis

The frequency response $H(\omega)$ completely describes how an LTI system modifies each frequency:

$$Y(\omega) = H(\omega) X(\omega)$$

For a causal first-order system $y'(t) + ay(t) = x(t)$:

Taking the Fourier transform: $j\omega Y(\omega) + aY(\omega) = X(\omega)$

$$H(\omega) = \frac{Y(\omega)}{X(\omega)} = \frac{1}{a + j\omega}$$

- **Magnitude**: $|H(\omega)| = 1/\sqrt{a^2 + \omega^2}$ (lowpass, monotonically decreasing)
- **3 dB cutoff**: at $\omega_c = a$ where $|H| = 1/\sqrt{2}$
- **Rolloff**: $-20$ dB/decade for $\omega \gg a$ (first-order system)
- **Phase**: $\angle H(\omega) = -\arctan(\omega/a)$ (phase lag increases with frequency)

---

## 10. Ideal Filters

### 10.1 Ideal Lowpass Filter

$$H_{LP}(\omega) = \begin{cases} 1 & |\omega| \leq \omega_c \\ 0 & |\omega| > \omega_c \end{cases} = \text{rect}\left(\frac{\omega}{2\omega_c}\right)$$

- Passes all frequencies below $\omega_c$ without distortion
- Completely removes all frequencies above $\omega_c$
- Impulse response: $h(t) = \frac{\omega_c}{\pi}\text{sinc}\left(\frac{\omega_c t}{\pi}\right)$

> The ideal lowpass filter is **noncausal** ($h(t) \neq 0$ for $t < 0$) and therefore **not physically realizable**. Real filters approximate this ideal.

### 10.2 Ideal Highpass Filter

$$H_{HP}(\omega) = 1 - H_{LP}(\omega) = \begin{cases} 0 & |\omega| \leq \omega_c \\ 1 & |\omega| > \omega_c \end{cases}$$

Impulse response: $h(t) = \delta(t) - \frac{\omega_c}{\pi}\text{sinc}\left(\frac{\omega_c t}{\pi}\right)$

### 10.3 Ideal Bandpass Filter

$$H_{BP}(\omega) = \begin{cases} 1 & \omega_1 \leq |\omega| \leq \omega_2 \\ 0 & \text{otherwise} \end{cases}$$

### 10.4 Ideal Bandstop (Notch) Filter

$$H_{BS}(\omega) = 1 - H_{BP}(\omega) = \begin{cases} 0 & \omega_1 \leq |\omega| \leq \omega_2 \\ 1 & \text{otherwise} \end{cases}$$

### 10.5 Ideal Allpass Filter

$$|H_{AP}(\omega)| = 1, \quad \angle H_{AP}(\omega) = \phi(\omega)$$

Changes only the phase, not the magnitude. Used for phase equalization.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Ideal filters and their impulse responses ---

omega = np.linspace(-60, 60, 2000)
omega_c = 20  # cutoff frequency
omega_1, omega_2 = 15, 25  # bandpass edges
t = np.linspace(-2, 2, 2000)

fig, axes = plt.subplots(4, 2, figsize=(14, 14))

# 1. Lowpass
H_lp = np.where(np.abs(omega) <= omega_c, 1.0, 0.0)
h_lp = omega_c / np.pi * np.sinc(omega_c * t / np.pi)

axes[0, 0].plot(omega, H_lp, 'b-', linewidth=2)
axes[0, 0].set_title(f'Ideal Lowpass ($\\omega_c = {omega_c}$)')
axes[0, 0].set_xlabel('$\\omega$ (rad/s)')
axes[0, 0].set_ylabel('$|H(\\omega)|$')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t, h_lp, 'b-', linewidth=2)
axes[0, 1].set_title('Impulse Response: $h(t) = \\frac{\\omega_c}{\\pi}\\mathrm{sinc}(\\omega_c t / \\pi)$')
axes[0, 1].set_xlabel('t')
axes[0, 1].axvline(x=0, color='gray', linestyle=':', alpha=0.5)
axes[0, 1].grid(True, alpha=0.3)

# 2. Highpass
H_hp = 1 - H_lp
h_hp_approx = np.zeros_like(t)
# delta(t) approximation + sinc
dt = t[1] - t[0]
delta_idx = np.argmin(np.abs(t))
h_hp_approx[delta_idx] = 1 / dt
h_hp_approx -= h_lp

axes[1, 0].plot(omega, H_hp, 'r-', linewidth=2)
axes[1, 0].set_title(f'Ideal Highpass ($\\omega_c = {omega_c}$)')
axes[1, 0].set_xlabel('$\\omega$ (rad/s)')
axes[1, 0].set_ylabel('$|H(\\omega)|$')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(t, -h_lp, 'r-', linewidth=2)
axes[1, 1].set_title('$h(t) = \\delta(t) - h_{LP}(t)$ (sinc part shown)')
axes[1, 1].set_xlabel('t')
axes[1, 1].grid(True, alpha=0.3)

# 3. Bandpass
H_bp = np.where((np.abs(omega) >= omega_1) & (np.abs(omega) <= omega_2), 1.0, 0.0)
omega_m = (omega_1 + omega_2) / 2
B = omega_2 - omega_1
h_bp = B / np.pi * np.sinc(B * t / (2 * np.pi)) * np.cos(omega_m * t)

axes[2, 0].plot(omega, H_bp, 'g-', linewidth=2)
axes[2, 0].set_title(f'Ideal Bandpass ($\\omega_1={omega_1}$, $\\omega_2={omega_2}$)')
axes[2, 0].set_xlabel('$\\omega$ (rad/s)')
axes[2, 0].set_ylabel('$|H(\\omega)|$')
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].plot(t, h_bp, 'g-', linewidth=1.5)
axes[2, 1].set_title('Bandpass Impulse Response')
axes[2, 1].set_xlabel('t')
axes[2, 1].grid(True, alpha=0.3)

# 4. Bandstop
H_bs = 1 - H_bp

axes[3, 0].plot(omega, H_bs, 'm-', linewidth=2)
axes[3, 0].set_title(f'Ideal Bandstop (Notch)')
axes[3, 0].set_xlabel('$\\omega$ (rad/s)')
axes[3, 0].set_ylabel('$|H(\\omega)|$')
axes[3, 0].grid(True, alpha=0.3)

axes[3, 1].text(0.5, 0.5, '$h(t) = \\delta(t) - h_{BP}(t)$',
               transform=axes[3, 1].transAxes, fontsize=14, ha='center', va='center')
axes[3, 1].set_title('Bandstop = All - Bandpass')
axes[3, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ideal_filters.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 11. Bandwidth and Time-Bandwidth Product

### 11.1 Bandwidth Definitions

There are several ways to define the bandwidth of a signal:

| Definition | Formula | Description |
|-----------|---------|-------------|
| **3 dB bandwidth** | $B_{3dB}$: $\|X(\omega_c)\|^2 = \frac{1}{2}\|X(0)\|^2$ | Half-power bandwidth |
| **Null-to-null** | Distance between first zeros | Often used for sinc-like spectra |
| **Equivalent noise bandwidth** | $B_{eq} = \frac{\int|X(\omega)|^2 d\omega}{|X(\omega_0)|^2}$ | Rectangle with same peak and energy |
| **RMS bandwidth** | $B_{rms} = \sqrt{\frac{\int\omega^2|X(\omega)|^2 d\omega}{\int|X(\omega)|^2 d\omega}}$ | Standard deviation of $|X|^2$ |
| **Essential bandwidth** | $B_{ess}$: contains $p$% of energy | Typically $p = 99$ |

### 11.2 Time-Bandwidth Product (Uncertainty Principle)

A fundamental constraint relates signal duration $\Delta t$ and bandwidth $\Delta\omega$:

$$\Delta t \cdot \Delta\omega \geq \frac{1}{2}$$

where $\Delta t$ and $\Delta\omega$ are defined as the RMS durations in time and frequency:

$$\Delta t = \sqrt{\frac{\int t^2 |x(t)|^2 dt}{\int |x(t)|^2 dt}}, \quad \Delta\omega = \sqrt{\frac{\int \omega^2 |X(\omega)|^2 d\omega}{\int |X(\omega)|^2 d\omega}}$$

The **equality** $\Delta t \cdot \Delta\omega = 1/2$ is achieved **only by the Gaussian pulse** — it is the most compact signal in the joint time-frequency sense.

### 11.3 Practical Implications

- A short pulse (small $\Delta t$) must have wide bandwidth (large $\Delta\omega$)
- A narrowband signal (small $\Delta\omega$) must have long duration (large $\Delta t$)
- You cannot design a signal that is both very short in time and very narrow in frequency
- This is the signal processing analog of the Heisenberg uncertainty principle in quantum mechanics

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Time-bandwidth product demonstration ---

fs = 10000
t = np.arange(-5, 5, 1 / fs)
dt = 1 / fs

def compute_time_bandwidth_product(x, t):
    """Compute RMS duration, RMS bandwidth, and their product."""
    # Normalize energy
    E = np.trapz(np.abs(x)**2, t)
    x_norm = np.abs(x)**2 / E

    # RMS duration
    t_mean = np.trapz(t * x_norm, t)
    t2_mean = np.trapz(t**2 * x_norm, t)
    delta_t = np.sqrt(t2_mean - t_mean**2)

    # Spectrum
    N = len(t)
    omega = np.fft.fftshift(np.fft.fftfreq(N, dt)) * 2 * np.pi
    X = np.fft.fftshift(np.fft.fft(x)) * dt
    X_norm = np.abs(X)**2 / np.trapz(np.abs(X)**2, omega) * (2 * np.pi)

    # RMS bandwidth
    omega_mean = np.trapz(omega * X_norm, omega)
    omega2_mean = np.trapz(omega**2 * X_norm, omega)
    delta_omega = np.sqrt(omega2_mean - omega_mean**2)

    return delta_t, delta_omega, delta_t * delta_omega

# Test with different Gaussian widths
sigmas = [0.1, 0.2, 0.5, 1.0, 2.0]

print("=== Time-Bandwidth Product for Gaussian Pulses ===")
print(f"{'sigma':>8} | {'Delta_t':>10} | {'Delta_omega':>12} | {'TBP':>10} | {'Limit (0.5)':>12}")
print("-" * 60)

delta_ts = []
delta_omegas = []

for sigma in sigmas:
    x = np.exp(-t**2 / (2 * sigma**2))
    dt_val, dw_val, tbp = compute_time_bandwidth_product(x, t)
    delta_ts.append(dt_val)
    delta_omegas.append(dw_val)
    print(f"{sigma:>8.1f} | {dt_val:>10.4f} | {dw_val:>12.4f} | {tbp:>10.4f} | {'0.5000':>12}")

print()

# Compare with rectangular pulse
print("=== TBP for Other Pulse Shapes ===")
shapes = {
    'Gaussian (sigma=0.5)': np.exp(-t**2 / (2 * 0.5**2)),
    'Rectangular (width=1)': np.where(np.abs(t) <= 0.5, 1.0, 0.0),
    'Triangular (width=2)': np.maximum(0, 1 - np.abs(t)),
    'Exponential (a=2)': np.where(t >= 0, np.exp(-2 * t), 0.0),
}

for name, x in shapes.items():
    dt_val, dw_val, tbp = compute_time_bandwidth_product(x, t)
    print(f"{name:>30}: TBP = {tbp:.4f} (limit = 0.5)")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for sigma in [0.2, 0.5, 1.0]:
    x = np.exp(-t**2 / (2 * sigma**2))
    axes[0].plot(t, x, linewidth=2, label=f'$\\sigma={sigma}$')

    N = len(t)
    omega = np.fft.fftshift(np.fft.fftfreq(N, dt)) * 2 * np.pi
    X = np.fft.fftshift(np.fft.fft(x)) * dt
    axes[1].plot(omega, np.abs(X), linewidth=2, label=f'$\\sigma={sigma}$')

axes[0].set_title('Gaussian Pulses (time domain)')
axes[0].set_xlabel('t')
axes[0].set_xlim([-3, 3])
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Spectra (frequency domain)')
axes[1].set_xlabel('$\\omega$ (rad/s)')
axes[1].set_xlim([-30, 30])
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Time-Bandwidth Tradeoff: Narrow in time = Wide in frequency', fontsize=12)
plt.tight_layout()
plt.savefig('time_bandwidth.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 12. Python Examples

### 12.1 Comprehensive CTFT Analysis Toolkit

```python
import numpy as np
import matplotlib.pyplot as plt

class CTFTAnalyzer:
    """Continuous-Time Fourier Transform analysis toolkit using FFT."""

    def __init__(self, fs=10000, duration=10.0):
        self.fs = fs
        self.dt = 1 / fs
        self.duration = duration
        self.t = np.arange(-duration / 2, duration / 2, self.dt)
        self.N = len(self.t)

    def ctft(self, x):
        """Approximate CTFT using FFT."""
        X = np.fft.fftshift(np.fft.fft(x)) * self.dt
        omega = np.fft.fftshift(np.fft.fftfreq(self.N, self.dt)) * 2 * np.pi
        return omega, X

    def ictft(self, X, omega=None):
        """Approximate inverse CTFT using IFFT."""
        X_unshifted = np.fft.ifftshift(X)
        x = np.fft.ifft(X_unshifted) * self.N * self.dt / (2 * np.pi)
        # Correct scaling
        x = np.real(x) / self.dt
        return x

    def energy(self, x):
        """Signal energy in time domain."""
        return np.trapz(np.abs(x)**2, self.t)

    def energy_spectral_density(self, X, omega):
        """Energy from frequency domain (Parseval's)."""
        return np.trapz(np.abs(X)**2, omega) / (2 * np.pi)

    def bandwidth_3db(self, X, omega):
        """Compute 3-dB bandwidth."""
        mag = np.abs(X)
        peak = np.max(mag)
        threshold = peak / np.sqrt(2)
        above = omega[mag >= threshold]
        if len(above) > 0:
            return above[-1] - above[0]
        return 0

    def analyze(self, x, title="Signal"):
        """Complete time-frequency analysis."""
        omega, X = self.ctft(x)

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))

        # Time domain
        axes[0, 0].plot(self.t, x, 'b-', linewidth=1)
        axes[0, 0].set_title(f'{title} — Time Domain')
        axes[0, 0].set_xlabel('t (s)')
        axes[0, 0].set_ylabel('x(t)')
        axes[0, 0].grid(True, alpha=0.3)

        # Magnitude spectrum
        axes[0, 1].plot(omega, np.abs(X), 'r-', linewidth=1)
        axes[0, 1].set_title('Magnitude Spectrum $|X(\\omega)|$')
        axes[0, 1].set_xlabel('$\\omega$ (rad/s)')
        axes[0, 1].grid(True, alpha=0.3)

        # Phase spectrum
        phase = np.angle(X)
        # Mask small values
        phase[np.abs(X) < np.max(np.abs(X)) * 0.01] = 0
        axes[0, 2].plot(omega, phase, 'g-', linewidth=0.5)
        axes[0, 2].set_title('Phase Spectrum $\\angle X(\\omega)$')
        axes[0, 2].set_xlabel('$\\omega$ (rad/s)')
        axes[0, 2].set_ylabel('Phase (rad)')
        axes[0, 2].grid(True, alpha=0.3)

        # Log magnitude (dB)
        mag_db = 20 * np.log10(np.abs(X) + 1e-12)
        axes[1, 0].plot(omega, mag_db, 'r-', linewidth=1)
        axes[1, 0].set_title('Log Magnitude (dB)')
        axes[1, 0].set_xlabel('$\\omega$ (rad/s)')
        axes[1, 0].set_ylabel('dB')
        axes[1, 0].grid(True, alpha=0.3)

        # Energy spectral density
        ESD = np.abs(X)**2
        axes[1, 1].plot(omega, ESD, 'm-', linewidth=1)
        axes[1, 1].set_title('Energy Spectral Density $|X(\\omega)|^2$')
        axes[1, 1].set_xlabel('$\\omega$ (rad/s)')
        axes[1, 1].grid(True, alpha=0.3)

        # Energy distribution
        E_total = self.energy(x)
        omega_pos = omega[omega >= 0]
        ESD_pos = ESD[omega >= 0]
        cum_energy = np.cumsum(ESD_pos) * (omega_pos[1] - omega_pos[0]) / np.pi
        axes[1, 2].plot(omega_pos, cum_energy / E_total * 100, 'k-', linewidth=2)
        axes[1, 2].axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90%')
        axes[1, 2].axhline(y=99, color='lightgray', linestyle='--', alpha=0.5, label='99%')
        axes[1, 2].set_title('Cumulative Energy (%)')
        axes[1, 2].set_xlabel('$\\omega$ (rad/s)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        # Print statistics
        E_freq = self.energy_spectral_density(X, omega)
        bw = self.bandwidth_3db(X, omega)
        print(f"=== {title} Analysis ===")
        print(f"  Energy (time):  {E_total:.6f}")
        print(f"  Energy (freq):  {E_freq:.6f}")
        print(f"  3-dB bandwidth: {bw:.2f} rad/s = {bw/(2*np.pi):.2f} Hz")

        plt.tight_layout()
        plt.savefig(f'ctft_analysis_{title.replace(" ", "_")}.png',
                    dpi=150, bbox_inches='tight')
        plt.show()


# Demonstrate
analyzer = CTFTAnalyzer(fs=10000, duration=10.0)

# Gaussian pulse
sigma = 0.1
x_gauss = np.exp(-analyzer.t**2 / (2 * sigma**2))
analyzer.analyze(x_gauss, "Gaussian Pulse")
```

### 12.2 Filtering in the Frequency Domain

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Frequency-domain filtering ---

fs = 8000
duration = 1.0
t = np.arange(0, duration, 1 / fs)
N = len(t)

# Create a signal with multiple frequency components
f_components = [200, 500, 1200, 2500]
amplitudes = [1.0, 0.8, 0.5, 0.3]
x = sum(a * np.sin(2 * np.pi * f * t) for a, f in zip(amplitudes, f_components))
x += 0.2 * np.random.randn(N)  # add noise

# Design filters in frequency domain
freq = np.fft.rfftfreq(N, 1 / fs)
X = np.fft.rfft(x)

# Lowpass: keep below 800 Hz
fc_lp = 800
H_lp = np.where(freq <= fc_lp, 1.0, 0.0)
# Smooth transition (Gaussian rolloff instead of brick wall)
H_lp_smooth = np.exp(-(np.maximum(0, freq - fc_lp))**2 / (2 * 50**2))

# Bandpass: 400-600 Hz
H_bp = np.exp(-((freq - 500)**2) / (2 * 60**2))

# Apply filters
y_lp = np.fft.irfft(X * H_lp_smooth, N)
y_bp = np.fft.irfft(X * H_bp, N)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Original
axes[0, 0].plot(t[:400], x[:400], 'b-', linewidth=0.8)
axes[0, 0].set_title('Original Signal')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(freq, 20 * np.log10(np.abs(X) / N + 1e-12), 'b-', linewidth=0.8)
for f in f_components:
    axes[0, 1].axvline(x=f, color='gray', linestyle=':', alpha=0.5)
axes[0, 1].set_title('Original Spectrum')
axes[0, 1].set_xlabel('Frequency (Hz)')
axes[0, 1].set_ylabel('dB')
axes[0, 1].grid(True, alpha=0.3)

# Lowpass filtered
axes[1, 0].plot(t[:400], y_lp[:400], 'r-', linewidth=0.8)
axes[1, 0].set_title(f'Lowpass Filtered ($f_c = {fc_lp}$ Hz)')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].grid(True, alpha=0.3)

Y_lp = np.fft.rfft(y_lp)
axes[1, 1].plot(freq, 20 * np.log10(np.abs(Y_lp) / N + 1e-12), 'r-', linewidth=0.8)
axes[1, 1].plot(freq, 20 * np.log10(H_lp_smooth + 1e-12), 'k--', linewidth=1.5,
                label='Filter')
axes[1, 1].set_title('Lowpass Output Spectrum')
axes[1, 1].set_xlabel('Frequency (Hz)')
axes[1, 1].set_ylabel('dB')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Bandpass filtered
axes[2, 0].plot(t[:400], y_bp[:400], 'g-', linewidth=0.8)
axes[2, 0].set_title('Bandpass Filtered (center = 500 Hz)')
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].grid(True, alpha=0.3)

Y_bp = np.fft.rfft(y_bp)
axes[2, 1].plot(freq, 20 * np.log10(np.abs(Y_bp) / N + 1e-12), 'g-', linewidth=0.8)
axes[2, 1].plot(freq, 20 * np.log10(H_bp + 1e-12), 'k--', linewidth=1.5,
                label='Filter')
axes[2, 1].set_title('Bandpass Output Spectrum')
axes[2, 1].set_xlabel('Frequency (Hz)')
axes[2, 1].set_ylabel('dB')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('freq_domain_filtering.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 12.3 Property Verification Suite

```python
import numpy as np

# --- Systematic verification of Fourier transform properties ---

fs = 10000
duration = 10.0
dt = 1 / fs
t = np.arange(-duration / 2, duration / 2, dt)
N = len(t)

def fft_ctft(x):
    """Approximate CTFT using FFT."""
    return np.fft.fftshift(np.fft.fft(x)) * dt

omega = np.fft.fftshift(np.fft.fftfreq(N, dt)) * 2 * np.pi

# Test signal: Gaussian
sigma = 0.3
x = np.exp(-t**2 / (2 * sigma**2))
X = fft_ctft(x)

print("=== Fourier Transform Property Verification ===\n")

# 1. Linearity
a, b = 2.5, -1.3
x1 = np.exp(-t**2 / (2 * 0.3**2))
x2 = np.exp(-t**2 / (2 * 0.5**2))
X_linear_lhs = fft_ctft(a * x1 + b * x2)
X_linear_rhs = a * fft_ctft(x1) + b * fft_ctft(x2)
print(f"1. Linearity:       error = {np.max(np.abs(X_linear_lhs - X_linear_rhs)):.2e}")

# 2. Time shifting
t0 = 1.0
x_shifted = np.exp(-(t - t0)**2 / (2 * sigma**2))
X_shift_lhs = fft_ctft(x_shifted)
X_shift_rhs = X * np.exp(-1j * omega * t0)
print(f"2. Time shifting:   error = {np.max(np.abs(X_shift_lhs - X_shift_rhs)):.2e}")

# 3. Frequency shifting
omega0 = 10.0
x_modulated = x * np.exp(1j * omega0 * t)
X_mod_lhs = fft_ctft(x_modulated)
# Shift X by omega0
X_mod_rhs = np.interp(omega - omega0, omega, np.real(X)) + \
            1j * np.interp(omega - omega0, omega, np.imag(X))
# This is approximate due to interpolation; use a simpler check
# Check that the peak moved
peak_original = omega[np.argmax(np.abs(X))]
peak_modulated = omega[np.argmax(np.abs(X_mod_lhs))]
print(f"3. Freq shifting:   peak moved from {peak_original:.1f} to {peak_modulated:.1f} "
      f"(expected {peak_original + omega0:.1f})")

# 4. Time scaling
a_scale = 2.0
x_scaled = np.exp(-(a_scale * t)**2 / (2 * sigma**2))
X_scale_lhs = fft_ctft(x_scaled)
# Compare with analytical: sigma/a * sqrt(2pi) * exp(-sigma^2 * omega^2 / (2*a^2))
X_scale_analytical = (sigma / a_scale) * np.sqrt(2 * np.pi) * \
                     np.exp(-sigma**2 * omega**2 / (2 * a_scale**2))
# Normalize for comparison
ratio = np.abs(X_scale_lhs[N//2]) / X_scale_analytical[N//2] if X_scale_analytical[N//2] != 0 else 1
print(f"4. Time scaling:    peak ratio = {ratio:.4f} (expected ~1.0)")

# 5. Parseval's theorem
E_time = np.trapz(np.abs(x)**2, t)
E_freq = np.trapz(np.abs(X)**2, omega) / (2 * np.pi)
print(f"5. Parseval's:      E_time={E_time:.6f}, E_freq={E_freq:.6f}, "
      f"error={abs(E_time-E_freq):.2e}")

# 6. Convolution theorem
h = np.where(t >= 0, np.exp(-2 * t), 0.0)
H = fft_ctft(h)
y_conv = np.convolve(x, h, mode='full')[:N] * dt
Y_conv = fft_ctft(y_conv)
Y_mult = X * H
conv_error = np.max(np.abs(Y_conv - Y_mult)) / np.max(np.abs(Y_conv))
print(f"6. Convolution thm: relative error = {conv_error:.2e}")

# 7. Differentiation
# Numerical derivative
dx_dt = np.gradient(x, dt)
X_diff_lhs = fft_ctft(dx_dt)
X_diff_rhs = (1j * omega) * X
# Compare (ignoring edges where gradient is inaccurate)
mask = np.abs(omega) < 50
diff_error = np.max(np.abs(X_diff_lhs[mask] - X_diff_rhs[mask])) / np.max(np.abs(X_diff_rhs[mask]))
print(f"7. Differentiation: relative error = {diff_error:.2e}")
```

### 12.4 Transform Pair Gallery

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Gallery of Fourier transform pairs ---

fs = 10000
t = np.arange(-5, 5, 1 / fs)
N = len(t)
dt = 1 / fs
omega = np.fft.fftshift(np.fft.fftfreq(N, dt)) * 2 * np.pi

def ctft(x):
    return np.fft.fftshift(np.fft.fft(x)) * dt

pairs = [
    ('$e^{-2t}u(t)$',
     np.where(t >= 0, np.exp(-2 * t), 0.0),
     '$\\frac{1}{2+j\\omega}$',
     1 / (2 + 1j * omega)),

    ('$e^{-2|t|}$',
     np.exp(-2 * np.abs(t)),
     '$\\frac{4}{4+\\omega^2}$',
     4 / (4 + omega**2)),

    ('$\\mathrm{rect}(t)$',
     np.where(np.abs(t) <= 0.5, 1.0, 0.0),
     '$\\mathrm{sinc}(\\omega/(2\\pi))$',
     np.sinc(omega / (2 * np.pi))),

    ('$\\mathrm{tri}(t)$',
     np.maximum(0, 1 - np.abs(t)),
     '$\\mathrm{sinc}^2(\\omega/(2\\pi))$',
     np.sinc(omega / (2 * np.pi))**2),

    ('$e^{-t^2/2}$',
     np.exp(-t**2 / 2),
     '$\\sqrt{2\\pi}e^{-\\omega^2/2}$',
     np.sqrt(2 * np.pi) * np.exp(-omega**2 / 2)),

    ('$te^{-t}u(t)$',
     np.where(t >= 0, t * np.exp(-t), 0.0),
     '$\\frac{1}{(1+j\\omega)^2}$',
     1 / (1 + 1j * omega)**2),
]

fig, axes = plt.subplots(len(pairs), 2, figsize=(14, 3 * len(pairs)))

for row, (t_label, x, f_label, X_analytical) in enumerate(pairs):
    # Numerical transform
    X_numerical = ctft(x)

    # Time domain
    axes[row, 0].plot(t, x, 'b-', linewidth=2)
    axes[row, 0].set_title(f'$x(t) = $ {t_label}')
    axes[row, 0].set_xlabel('t')
    axes[row, 0].set_xlim([-3, 5])
    axes[row, 0].grid(True, alpha=0.3)

    # Frequency domain (compare numerical and analytical)
    axes[row, 1].plot(omega, np.abs(X_numerical), 'r-', linewidth=2,
                     label='FFT (numerical)')
    axes[row, 1].plot(omega, np.abs(X_analytical), 'k--', linewidth=1.5,
                     label='Analytical', alpha=0.7)
    axes[row, 1].set_title(f'$X(\\omega) = $ {f_label}')
    axes[row, 1].set_xlabel('$\\omega$ (rad/s)')
    axes[row, 1].set_xlim([-30, 30])
    axes[row, 1].legend(fontsize=8)
    axes[row, 1].grid(True, alpha=0.3)

plt.suptitle('Fourier Transform Pair Gallery: Numerical vs Analytical', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('transform_pair_gallery.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 13. Summary

### Core Concepts

| Concept | Key Idea |
|---------|----------|
| Fourier Transform | Decomposes aperiodic signals into continuous frequency components |
| Inverse Transform | Reconstructs the signal from its spectrum |
| Convolution Theorem | Convolution in time = multiplication in frequency |
| Multiplication Theorem | Multiplication in time = convolution in frequency |
| Parseval's Theorem | Energy is conserved between time and frequency domains |
| Time-Bandwidth Product | $\Delta t \cdot \Delta\omega \geq 1/2$ (uncertainty principle) |

### Transform Properties Quick Reference

```
Time Domain  ←→  Frequency Domain
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x(t-t₀)     ←→  e^{-jωt₀} X(ω)      Time shift = linear phase
x(t)e^{jω₀t} ←→  X(ω-ω₀)            Modulation = spectrum shift
x(at)        ←→  (1/|a|)X(ω/a)       Compression ↔ expansion
dx/dt        ←→  jω X(ω)             Differentiation = × jω
x*h          ←→  X·H                  Convolution = multiplication
x·w          ←→  (1/2π) X*W           Multiplication = convolution
```

### From Series to Transform: The Big Picture

```
    Periodic signals              Aperiodic signals
    ──────────────              ─────────────────
    Fourier Series              Fourier Transform
    cn (discrete)               X(ω) (continuous)
    Line spectrum               Continuous spectrum
    Σ cn e^{jnω₀t}             ∫ X(ω) e^{jωt} dω/(2π)
         │                              │
         └──── As T₀ → ∞ ──────────────┘
               cn → X(ω)dω/(2π)
               nω₀ → ω
               Σ → ∫
```

---

## 14. Exercises

### Exercise 1: Transform Computation

Compute the Fourier transform of the following signals analytically. Verify with Python.

1. $x(t) = e^{-3t}u(t) - e^{-5t}u(t)$
2. $x(t) = te^{-2t}u(t)$ (use the differentiation-in-frequency property)
3. $x(t) = e^{-|t|}\cos(10t)$ (use modulation + known transform of $e^{-|t|}$)
4. $x(t) = \text{rect}(t) \cdot \cos(20\pi t)$ (modulated rectangular pulse)

### Exercise 2: Property Application

Using the Fourier transform of $e^{-at}u(t) \leftrightarrow 1/(a + j\omega)$ and transform properties only (no re-derivation from the integral), find:

1. $\mathcal{F}\{e^{-a(t-3)}u(t-3)\}$ (time shift)
2. $\mathcal{F}\{e^{-at}u(t) \cdot e^{j5t}\}$ (frequency shift)
3. $\mathcal{F}\{e^{-2at}u(2t)\}$ (scaling; be careful!)
4. $\mathcal{F}\{te^{-at}u(t)\}$ (frequency differentiation)
5. $\mathcal{F}\{\frac{d}{dt}[e^{-at}u(t)]\}$ (time differentiation; note the delta at $t=0$)

### Exercise 3: Convolution Theorem Applications

1. Compute $e^{-t}u(t) * e^{-2t}u(t)$ using the frequency domain (transform, multiply, inverse transform). Verify by direct convolution.
2. Find the output of a first-order lowpass system $H(\omega) = 1/(1 + j\omega/\omega_c)$ when the input is $x(t) = e^{-t}u(t)$ with $\omega_c = 5$ rad/s.
3. A signal $x(t) = \text{sinc}(Bt)$ passes through an ideal lowpass filter with cutoff $\omega_c$. What is the output when (a) $\omega_c > \pi B$? (b) $\omega_c < \pi B$?

### Exercise 4: Parseval's Theorem

1. Compute the energy of $x(t) = \frac{1}{1+t^2}$ using Parseval's theorem (hint: find $X(\omega)$ first).
2. What fraction of the energy of a Gaussian pulse $e^{-t^2/(2\sigma^2)}$ lies within the frequency band $|\omega| \leq 1/\sigma$?
3. A rectangular pulse of width $T$ has energy $T$. How much bandwidth (in the 99%-energy sense) does it need?

### Exercise 5: Filter Design

1. Design a frequency-domain filter to extract a 1 kHz tone from a signal contaminated by 60 Hz hum and broadband noise. Implement using FFT, apply to a test signal, and plot before/after spectra.
2. Implement an ideal differentiator $H(\omega) = j\omega$ using FFT. Apply to a Gaussian pulse and compare with the analytical derivative.
3. Why does the ideal lowpass filter ring when applied to a step input? Quantify the ringing in terms of Gibbs phenomenon.

### Exercise 6: Time-Bandwidth Product

1. Compute the time-bandwidth product for:
   - Rectangular pulse $\text{rect}(t/T)$
   - Gaussian pulse $e^{-\pi t^2}$ (should get exactly 0.5)
   - First-order exponential $e^{-t}u(t)$
   - Raised cosine pulse $\frac{1}{2}(1 + \cos(\pi t/T))$ for $|t| \leq T$
2. Rank these pulses by their time-bandwidth product efficiency.
3. Which pulse shape would you choose for a radar system? For a communications system? Explain.

### Exercise 7: Duality

1. Use duality to find $\mathcal{F}\{1/(a^2 + t^2)\}$ from the known pair $e^{-a|t|} \leftrightarrow 2a/(a^2 + \omega^2)$.
2. Use duality to find $\mathcal{F}\{\text{sinc}(Wt)\}$ from the known pair $\text{rect}(t/\tau) \leftrightarrow \tau\,\text{sinc}(\omega\tau/(2\pi))$.
3. Verify both results numerically.

### Exercise 8: Comprehensive Analysis

Record or generate a 2-second audio signal containing speech, music, or a synthetic signal with multiple frequency components.

1. Compute and plot the magnitude and phase spectra
2. Identify the dominant frequency components
3. Design and apply a bandpass filter to isolate one component
4. Compute the energy before and after filtering; what fraction was in the passband?
5. Apply time-domain windowing (Hamming) before computing the spectrum. Compare with the unwindowed result.
6. Vary the analysis window length and observe the time-frequency resolution tradeoff

---

## 15. References

1. Oppenheim, A. V. & Willsky, A. S. *Signals and Systems* (2nd ed.), Ch. 4-5. Prentice Hall, 1997.
2. Haykin, S. & Van Veen, B. *Signals and Systems* (2nd ed.), Ch. 4-5. Wiley, 2003.
3. Bracewell, R. N. *The Fourier Transform and Its Applications* (3rd ed.). McGraw-Hill, 2000.
4. Lathi, B. P. & Green, R. A. *Linear Systems and Signals* (3rd ed.), Ch. 7. Oxford University Press, 2018.
5. Mallat, S. *A Wavelet Tour of Signal Processing* (3rd ed.), Ch. 2. Academic Press, 2009.

---

[Previous: 03. Fourier Series and Applications](./03_Fourier_Series_and_Applications.md) | [Next: 05. Sampling and Reconstruction](./05_Sampling_and_Reconstruction.md) | [Overview](./00_Overview.md)
