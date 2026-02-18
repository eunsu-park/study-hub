# Fourier Series and Applications

**Previous**: [02. LTI Systems and Convolution](./02_LTI_Systems_and_Convolution.md) | **Next**: [04. Continuous Fourier Transform](./04_Continuous_Fourier_Transform.md)

---

In Lesson 02 we saw that LTI systems are completely characterized by their impulse response, and that convolution computes the output for any input. But convolution in the time domain can be tedious. There is a much more elegant approach: decompose signals into complex exponentials, which are **eigenfunctions** of LTI systems. The response to each exponential is simply a scaling, and the total output is the sum of these scaled exponentials.

For **periodic signals**, this decomposition is the **Fourier series**. This lesson develops the Fourier series from its mathematical foundations, explores its convergence properties, and demonstrates its power in analyzing periodic phenomena.

**Difficulty**: ⭐⭐⭐

**Learning Objectives**:
- Derive the trigonometric and complex exponential forms of the Fourier series
- Compute Fourier coefficients for standard waveforms
- State and apply the Dirichlet conditions for convergence
- Explain the Gibbs phenomenon and its practical implications
- Apply Parseval's theorem to compute signal power from spectral coefficients
- Interpret line spectra (amplitude and phase spectra)
- Use Python to compute Fourier series approximations and visualize convergence

---

## Table of Contents

1. [Periodic Signals Review](#1-periodic-signals-review)
2. [Trigonometric Fourier Series](#2-trigonometric-fourier-series)
3. [Complex Exponential Fourier Series](#3-complex-exponential-fourier-series)
4. [Computing Fourier Coefficients](#4-computing-fourier-coefficients)
5. [Convergence of Fourier Series](#5-convergence-of-fourier-series)
6. [Gibbs Phenomenon](#6-gibbs-phenomenon)
7. [Parseval's Theorem](#7-parsevals-theorem)
8. [Line Spectra](#8-line-spectra)
9. [Applications: Standard Waveform Decompositions](#9-applications-standard-waveform-decompositions)
10. [Discrete-Time Fourier Series](#10-discrete-time-fourier-series)
11. [Python Examples](#11-python-examples)
12. [Summary](#12-summary)
13. [Exercises](#13-exercises)
14. [References](#14-references)

---

## 1. Periodic Signals Review

### 1.1 Definition

A continuous-time signal $x(t)$ is **periodic** with period $T$ if:

$$x(t + T) = x(t) \quad \text{for all } t$$

The smallest positive $T$ satisfying this is the **fundamental period** $T_0$. The **fundamental frequency** is:

$$f_0 = \frac{1}{T_0} \quad \text{(Hz)}, \qquad \omega_0 = \frac{2\pi}{T_0} \quad \text{(rad/s)}$$

### 1.2 Harmonics

The **$n$-th harmonic** of a periodic signal with fundamental frequency $\omega_0$ has frequency $n\omega_0$. The fundamental is the first harmonic ($n = 1$). The signal at $2\omega_0$ is the second harmonic, and so on.

### 1.3 Sum of Periodic Signals

If $x_1(t)$ has period $T_1$ and $x_2(t)$ has period $T_2$, their sum $x_1(t) + x_2(t)$ is periodic if and only if $T_1/T_2$ is a **rational number**. The period of the sum is the **least common multiple** of $T_1$ and $T_2$.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Periodic signals and harmonics ---
t = np.linspace(0, 4, 2000)

# Fundamental and harmonics
f0 = 1.0  # 1 Hz fundamental
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

for n, ax in zip([1, 2, 3, 5], axes):
    signal = np.cos(2 * np.pi * n * f0 * t)
    ax.plot(t, signal, linewidth=1.5)
    ax.set_title(f'Harmonic n={n}: $\\cos(2\\pi \\cdot {n} \\cdot {f0} \\cdot t)$, '
                 f'frequency = {n * f0} Hz, period = {1/(n*f0):.3f} s')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 4])

plt.tight_layout()
plt.savefig('harmonics.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 2. Trigonometric Fourier Series

### 2.1 The Representation

The **trigonometric Fourier series** expresses a periodic signal $x(t)$ with period $T_0$ as:

$$x(t) = a_0 + \sum_{n=1}^{\infty} \left[ a_n \cos(n\omega_0 t) + b_n \sin(n\omega_0 t) \right]$$

where $\omega_0 = 2\pi/T_0$ and the **Fourier coefficients** are:

$$a_0 = \frac{1}{T_0} \int_{T_0} x(t) \, dt \quad \text{(DC component / average value)}$$

$$a_n = \frac{2}{T_0} \int_{T_0} x(t) \cos(n\omega_0 t) \, dt, \quad n = 1, 2, 3, \ldots$$

$$b_n = \frac{2}{T_0} \int_{T_0} x(t) \sin(n\omega_0 t) \, dt, \quad n = 1, 2, 3, \ldots$$

The integral $\int_{T_0}$ means integration over any one complete period.

### 2.2 Derivation from Orthogonality

The key to finding the coefficients is the **orthogonality** of trigonometric functions over one period $[0, T_0]$:

$$\int_0^{T_0} \cos(m\omega_0 t) \cos(n\omega_0 t) \, dt = \begin{cases} T_0 & m = n = 0 \\ T_0/2 & m = n \neq 0 \\ 0 & m \neq n \end{cases}$$

$$\int_0^{T_0} \sin(m\omega_0 t) \sin(n\omega_0 t) \, dt = \begin{cases} T_0/2 & m = n \neq 0 \\ 0 & m \neq n \end{cases}$$

$$\int_0^{T_0} \cos(m\omega_0 t) \sin(n\omega_0 t) \, dt = 0 \quad \text{for all } m, n$$

To find $a_n$: multiply both sides of the Fourier series by $\cos(n\omega_0 t)$ and integrate over one period. Due to orthogonality, all terms vanish except the one matching $n$, yielding the coefficient formula.

### 2.3 Compact Trigonometric Form

Each harmonic can be combined into a single sinusoid:

$$x(t) = C_0 + \sum_{n=1}^{\infty} C_n \cos(n\omega_0 t + \phi_n)$$

where:

$$C_0 = a_0, \quad C_n = \sqrt{a_n^2 + b_n^2}, \quad \phi_n = -\arctan\left(\frac{b_n}{a_n}\right)$$

### 2.4 Symmetry Shortcuts

Signal symmetry can simplify coefficient computation:

| Signal Property | Consequence |
|----------------|-------------|
| **Even**: $x(t) = x(-t)$ | $b_n = 0$ for all $n$ (only cosine terms) |
| **Odd**: $x(t) = -x(-t)$ | $a_n = 0$ for all $n$ (only sine terms, $a_0 = 0$) |
| **Half-wave symmetry**: $x(t + T_0/2) = -x(t)$ | $a_n = b_n = 0$ for even $n$ (only odd harmonics) |

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Orthogonality verification ---
T0 = 2 * np.pi  # period
omega0 = 2 * np.pi / T0  # = 1
t = np.linspace(0, T0, 10000)
dt = t[1] - t[0]

print("=== Orthogonality of Trigonometric Functions ===\n")

# cos(m*t) * cos(n*t) integrals
print("cos(m*t) * cos(n*t) over [0, 2*pi]:")
for m in range(4):
    for n in range(4):
        integral = np.trapz(np.cos(m * t) * np.cos(n * t), t)
        if abs(integral) > 1e-8:
            print(f"  m={m}, n={n}: {integral:.4f} (expected {T0 if m==n==0 else T0/2:.4f})")

print("\ncos(m*t) * sin(n*t) over [0, 2*pi]:")
all_zero = True
for m in range(4):
    for n in range(1, 4):
        integral = np.trapz(np.cos(m * t) * np.sin(n * t), t)
        if abs(integral) > 1e-8:
            all_zero = False
            print(f"  m={m}, n={n}: {integral:.4f}")
if all_zero:
    print("  All integrals are zero (as expected)")
```

---

## 3. Complex Exponential Fourier Series

### 3.1 The Representation

The **complex exponential Fourier series** (often called the **exponential Fourier series**) is:

$$x(t) = \sum_{n=-\infty}^{\infty} c_n \, e^{jn\omega_0 t}$$

where the **complex Fourier coefficients** are:

$$c_n = \frac{1}{T_0} \int_{T_0} x(t) \, e^{-jn\omega_0 t} \, dt, \quad n = 0, \pm 1, \pm 2, \ldots$$

### 3.2 Relationship to Trigonometric Coefficients

$$c_0 = a_0$$

$$c_n = \frac{a_n - jb_n}{2}, \quad c_{-n} = \frac{a_n + jb_n}{2} = c_n^* \quad (n > 0)$$

Conversely:

$$a_n = 2\text{Re}(c_n) = c_n + c_{-n}$$

$$b_n = -2\text{Im}(c_n) = j(c_n - c_{-n})$$

For a **real signal**: $c_{-n} = c_n^*$ (conjugate symmetry).

### 3.3 Why Use the Complex Form?

The complex exponential form is preferred in signal processing because:

1. **Compact notation**: Single summation instead of separate $a_n$ and $b_n$
2. **Eigenfunction property**: $e^{jn\omega_0 t}$ are eigenfunctions of LTI systems
3. **Leads naturally to the Fourier transform**: The transform is the continuous-frequency extension
4. **Frequency content is directly visible**: $c_n$ at frequency $n\omega_0$ gives both magnitude and phase

### 3.4 Derivation

Start with Euler's formula:

$$\cos(n\omega_0 t) = \frac{e^{jn\omega_0 t} + e^{-jn\omega_0 t}}{2}, \quad \sin(n\omega_0 t) = \frac{e^{jn\omega_0 t} - e^{-jn\omega_0 t}}{2j}$$

Substitute into the trigonometric series and collect terms with $e^{jn\omega_0 t}$ for positive and negative $n$, yielding the complex form.

---

## 4. Computing Fourier Coefficients

### 4.1 Square Wave

A **square wave** with period $T_0$ and amplitude $A$:

$$x(t) = \begin{cases} A & 0 < t < T_0/2 \\ -A & T_0/2 < t < T_0 \end{cases}$$

**Trigonometric coefficients** (odd function shifted by $T_0/4$, but let us compute directly):

$$a_0 = 0 \quad \text{(zero average)}$$

$$a_n = 0 \quad \text{(for all } n \text{, by half-wave symmetry)}$$

$$b_n = \begin{cases} \frac{4A}{n\pi} & n \text{ odd} \\ 0 & n \text{ even} \end{cases}$$

Therefore:

$$x(t) = \frac{4A}{\pi} \left[\sin(\omega_0 t) + \frac{1}{3}\sin(3\omega_0 t) + \frac{1}{5}\sin(5\omega_0 t) + \cdots \right]$$

**Complex coefficients**:

$$c_n = \begin{cases} \frac{-2jA}{n\pi} & n \text{ odd} \\ 0 & n \text{ even (including } n = 0\text{)} \end{cases}$$

### 4.2 Sawtooth Wave

A sawtooth wave rising linearly from $-A$ to $A$ over one period $[0, T_0)$:

$$x(t) = A\left(\frac{2t}{T_0} - 1\right), \quad 0 \leq t < T_0$$

**Coefficients**:

$$a_0 = 0, \quad a_n = 0$$

$$b_n = \frac{-2A}{n\pi}(-1)^{n+1} = \frac{2A}{n\pi}(-1)^{n+1} \cdot (-1) = \frac{(-1)^{n+1} \cdot 2A}{n\pi}$$

More precisely:

$$b_n = \frac{-2A}{n\pi} \quad \Rightarrow \quad x(t) = \frac{-2A}{\pi}\sum_{n=1}^{\infty} \frac{(-1)^n}{n} \sin(n\omega_0 t)$$

### 4.3 Triangle Wave

A triangle wave with amplitude $A$ and period $T_0$:

$$x(t) = \begin{cases} \frac{4A}{T_0}t & 0 \leq t \leq T_0/4 \\ A - \frac{4A}{T_0}(t - T_0/4) & T_0/4 \leq t \leq 3T_0/4 \\ -A + \frac{4A}{T_0}(t - 3T_0/4) & 3T_0/4 \leq t \leq T_0 \end{cases}$$

Since it is an even function with half-wave symmetry:

$$a_n = \begin{cases} \frac{8A}{n^2\pi^2} & n = 1, 5, 9, \ldots \\ \frac{-8A}{n^2\pi^2} & n = 3, 7, 11, \ldots \\ 0 & n \text{ even} \end{cases}$$

$$b_n = 0 \quad \text{for all } n$$

Compact form:

$$x(t) = \frac{8A}{\pi^2} \sum_{k=0}^{\infty} \frac{(-1)^k}{(2k+1)^2} \cos((2k+1)\omega_0 t)$$

### 4.4 Rectified Sinusoid (Full-Wave Rectifier)

$$x(t) = |A\sin(\omega_0 t)|$$

This has period $T_0/2$ (fundamental frequency $2\omega_0$), so the Fourier series uses even harmonics of the original $\omega_0$:

$$x(t) = \frac{2A}{\pi} - \frac{4A}{\pi}\sum_{n=1}^{\infty} \frac{1}{4n^2 - 1} \cos(2n\omega_0 t)$$

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Fourier coefficients computation ---

def compute_fourier_coefficients(x_func, T0, N_harmonics, N_points=10000):
    """
    Compute complex Fourier coefficients numerically.

    Parameters:
        x_func: function of t returning signal values
        T0: fundamental period
        N_harmonics: number of harmonics (positive + negative)
        N_points: integration resolution

    Returns:
        n_values: harmonic indices
        cn: complex Fourier coefficients
    """
    omega0 = 2 * np.pi / T0
    t = np.linspace(0, T0, N_points, endpoint=False)
    dt = T0 / N_points
    x_vals = x_func(t)

    n_values = np.arange(-N_harmonics, N_harmonics + 1)
    cn = np.zeros(len(n_values), dtype=complex)

    for i, n in enumerate(n_values):
        cn[i] = np.mean(x_vals * np.exp(-1j * n * omega0 * t))

    return n_values, cn


# Square wave
T0 = 1.0
A = 1.0
def square_wave(t):
    return A * np.sign(np.sin(2 * np.pi * t / T0))

n_vals, cn = compute_fourier_coefficients(square_wave, T0, 20)

# Display non-negligible coefficients
print("Square Wave Fourier Coefficients (|cn| > 0.01):")
print(f"{'n':>4} | {'|cn|':>10} | {'angle(cn) (deg)':>16} | {'Expected |cn|':>14}")
for n, c in zip(n_vals, cn):
    if abs(c) > 0.01:
        expected = 2 * A / (abs(n) * np.pi) if n % 2 != 0 else 0
        print(f"{n:>4} | {abs(c):>10.6f} | {np.angle(c)*180/np.pi:>16.1f} | {expected:>14.6f}")
```

---

## 5. Convergence of Fourier Series

### 5.1 Dirichlet Conditions

The Fourier series converges to $x(t)$ at all points where $x(t)$ is continuous, provided the **Dirichlet conditions** are satisfied:

1. $x(t)$ is absolutely integrable over one period: $\int_{T_0} |x(t)| \, dt < \infty$
2. $x(t)$ has a finite number of maxima and minima in one period
3. $x(t)$ has a finite number of discontinuities in one period

At a discontinuity, the Fourier series converges to the **midpoint**:

$$\text{FS}\{x(t_0)\} = \frac{x(t_0^+) + x(t_0^-)}{2}$$

### 5.2 Types of Convergence

**Pointwise convergence**: $\lim_{N \to \infty} S_N(t) = x(t)$ at each specific $t$ (except possibly at discontinuities).

**Uniform convergence**: $\lim_{N \to \infty} \max_t |x(t) - S_N(t)| = 0$. This is stronger and requires $x(t)$ to be continuous.

**Mean-square (L2) convergence**: $\lim_{N \to \infty} \int_{T_0} |x(t) - S_N(t)|^2 \, dt = 0$. This always holds for square-integrable signals.

### 5.3 Rate of Convergence

The rate at which Fourier coefficients decay depends on signal smoothness:

| Signal Property | Coefficient Decay | Example |
|----------------|-------------------|---------|
| Discontinuous | $|c_n| \sim 1/n$ | Square wave |
| Continuous, derivative discontinuous | $|c_n| \sim 1/n^2$ | Triangle wave |
| Continuous with continuous derivative | $|c_n| \sim 1/n^3$ | Parabolic wave |
| $k$ times differentiable | $|c_n| \sim 1/n^{k+1}$ | Smoother signals |
| Infinitely differentiable | Faster than any power of $1/n$ | Gaussian |

> **Key insight**: Smoother signals have faster-decaying Fourier coefficients, meaning fewer terms are needed for a good approximation. Discontinuities cause slow ($1/n$) decay, which manifests as the Gibbs phenomenon.

---

## 6. Gibbs Phenomenon

### 6.1 Description

When a Fourier series approximates a signal with a jump discontinuity, the partial sum $S_N(t)$ exhibits **ringing** near the discontinuity. As $N$ increases:

- The overshoot/undershoot narrows in width (moves closer to the discontinuity)
- The **peak overshoot remains approximately 9%** of the jump magnitude, regardless of $N$

This is the **Gibbs phenomenon**, named after J. Willard Gibbs.

### 6.2 Mathematical Description

For a unit step discontinuity (jump from 0 to 1), the Gibbs overshoot converges to:

$$\frac{1}{\pi} \int_0^{\pi} \frac{\sin(u)}{u} \, du - \frac{1}{2} \approx 0.0895$$

This means the peak of the Fourier partial sum reaches approximately $\frac{1}{2} + 0.0895 = 0.5895$ instead of $0.5$, an overshoot of about **8.95%** of the jump.

### 6.3 Practical Implications

- Gibbs phenomenon is inherent to Fourier series — it cannot be eliminated by taking more terms
- It can be reduced by **windowing** (e.g., Fejer summation, Lanczos sigma factors)
- In filter design, it manifests as **passband ripple** (see Lesson 09)
- In image processing, it causes **ringing artifacts** at sharp edges

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Gibbs phenomenon demonstration ---
T0 = 2 * np.pi
omega0 = 1.0
t = np.linspace(-np.pi, 3 * np.pi, 5000)

# Square wave Fourier series partial sums
def square_fourier_partial(t, N):
    """Partial sum of square wave Fourier series with N harmonics."""
    result = np.zeros_like(t)
    for k in range(1, N + 1):
        if k % 2 == 1:  # odd harmonics only
            result += (4 / (k * np.pi)) * np.sin(k * t)
    return result

# True square wave
x_true = np.sign(np.sin(t))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

N_values = [3, 9, 31, 101]
for ax, N in zip(axes.flat, N_values):
    S_N = square_fourier_partial(t, N)
    ax.plot(t / np.pi, x_true, 'b--', linewidth=1, alpha=0.5, label='Square wave')
    ax.plot(t / np.pi, S_N, 'r-', linewidth=1.5, label=f'$S_{{{N}}}(t)$')

    # Mark the Gibbs overshoot
    if N > 5:
        # Find max near discontinuity at t=0
        mask = (t > 0) & (t < np.pi / 2)
        max_val = np.max(S_N[mask])
        overshoot_pct = (max_val - 1.0) * 100
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        ax.annotate(f'Overshoot: {overshoot_pct:.1f}%',
                    xy=(0.15, max_val), fontsize=10, color='darkred')

    ax.set_title(f'N = {N} harmonics')
    ax.set_xlabel('$t / \\pi$')
    ax.set_ylabel('Amplitude')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.4, 1.4])
    ax.set_xlim([-0.5, 2.5])

plt.suptitle('Gibbs Phenomenon: Fourier Series of Square Wave', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('gibbs_phenomenon.png', dpi=150, bbox_inches='tight')
plt.show()

# Measure overshoot vs N
print("\n=== Gibbs Overshoot vs Number of Harmonics ===")
print(f"{'N':>6} | {'Max Value':>10} | {'Overshoot %':>12}")
print("-" * 35)
for N in [5, 11, 21, 51, 101, 201, 501, 1001]:
    t_fine = np.linspace(0.001, 0.5, 50000)
    S = square_fourier_partial(t_fine, N)
    max_val = np.max(S)
    overshoot = (max_val - 1.0) * 100
    print(f"{N:>6} | {max_val:>10.6f} | {overshoot:>11.4f}%")
```

---

## 7. Parseval's Theorem

### 7.1 Statement

**Parseval's theorem** (for Fourier series) states that the average power of a periodic signal equals the sum of squared magnitudes of its Fourier coefficients:

$$\frac{1}{T_0} \int_{T_0} |x(t)|^2 \, dt = \sum_{n=-\infty}^{\infty} |c_n|^2$$

In terms of trigonometric coefficients:

$$\frac{1}{T_0} \int_{T_0} |x(t)|^2 \, dt = a_0^2 + \frac{1}{2}\sum_{n=1}^{\infty} (a_n^2 + b_n^2) = \sum_{n=0}^{\infty} \frac{C_n^2}{2 - \delta_{n0}}$$

### 7.2 Interpretation

Parseval's theorem is a **conservation of energy** statement:

- **Left side**: Total average power computed in the time domain
- **Right side**: Sum of power contributions from each harmonic

Each harmonic $n$ contributes power $|c_n|^2 + |c_{-n}|^2 = 2|c_n|^2$ (for $n \neq 0$) to the total.

### 7.3 Example: Square Wave Power

For a square wave with amplitude $A$:

$$P_x = \frac{1}{T_0} \int_{T_0} A^2 \, dt = A^2$$

From Fourier coefficients ($|c_n| = 2A/(n\pi)$ for odd $n$):

$$\sum_{n \text{ odd}} |c_n|^2 = 2 \sum_{k=0}^{\infty} \left(\frac{2A}{(2k+1)\pi}\right)^2 = \frac{8A^2}{\pi^2} \sum_{k=0}^{\infty} \frac{1}{(2k+1)^2} = \frac{8A^2}{\pi^2} \cdot \frac{\pi^2}{8} = A^2 \quad \checkmark$$

This also proves the beautiful identity: $\sum_{k=0}^{\infty} \frac{1}{(2k+1)^2} = \frac{\pi^2}{8}$.

### 7.4 Power Spectrum

The **power spectrum** of a periodic signal is the set of values $\{|c_n|^2\}$ plotted against frequency $n\omega_0$. It shows how power is distributed across harmonics.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Parseval's theorem verification ---

T0 = 1.0
omega0 = 2 * np.pi / T0
A = 1.0

# Square wave
def square_wave(t):
    return A * np.sign(np.sin(2 * np.pi * t / T0))

t = np.linspace(0, T0, 100000, endpoint=False)
x = square_wave(t)

# Time-domain power
P_time = np.mean(x**2)

# Frequency-domain power (Parseval's)
N_max = 200
n_vals = np.arange(-N_max, N_max + 1)
cn = np.zeros(len(n_vals), dtype=complex)
for i, n in enumerate(n_vals):
    cn[i] = np.mean(x * np.exp(-1j * n * omega0 * t))

P_freq = np.sum(np.abs(cn)**2)

# Cumulative power contribution
cn_positive = cn[n_vals >= 0]
n_positive = n_vals[n_vals >= 0]
power_each = np.abs(cn_positive)**2
power_each[1:] *= 2  # double for n > 0 (conjugate symmetry)
P_cumulative = np.cumsum(power_each)

print(f"=== Parseval's Theorem Verification (Square Wave) ===")
print(f"Time-domain power:      P = {P_time:.8f}")
print(f"Frequency-domain power: P = {P_freq:.8f}")
print(f"Error: {abs(P_time - P_freq):.2e}")
print()

# How many harmonics capture 99% of power?
threshold = 0.99 * P_time
n_99 = n_positive[np.searchsorted(P_cumulative, threshold)]
print(f"Harmonics for 99% power: n = {n_99}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Power spectrum
axes[0].stem(n_positive[:30], power_each[:30] / P_time * 100,
             linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0].set_title('Power Spectrum (% of Total Power)')
axes[0].set_xlabel('Harmonic number n')
axes[0].set_ylabel('Power contribution (%)')
axes[0].grid(True, alpha=0.3)

# Cumulative power
axes[1].plot(n_positive[:50], P_cumulative[:50] / P_time * 100, 'r-o', markersize=3)
axes[1].axhline(y=99, color='gray', linestyle='--', label='99%')
axes[1].axhline(y=95, color='lightgray', linestyle='--', label='95%')
axes[1].set_title("Cumulative Power (Parseval's)")
axes[1].set_xlabel('Number of harmonics included')
axes[1].set_ylabel('Cumulative power (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('parseval_theorem.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 8. Line Spectra

### 8.1 Definition

The spectrum of a periodic signal is **discrete** (a set of lines at multiples of $\omega_0$), in contrast to aperiodic signals which have continuous spectra (Lesson 04).

The **line spectrum** consists of two plots:

1. **Amplitude spectrum**: $|c_n|$ vs. $n\omega_0$ (or simply vs. $n$)
2. **Phase spectrum**: $\angle c_n$ vs. $n\omega_0$

For real signals, the amplitude spectrum is **even** ($|c_n| = |c_{-n}|$) and the phase spectrum is **odd** ($\angle c_n = -\angle c_{-n}$).

### 8.2 One-Sided vs. Two-Sided Spectra

| Type | Frequency Range | Coefficients Used |
|------|----------------|-------------------|
| **Two-sided** | $-\infty < n < \infty$ | Complex $c_n$ |
| **One-sided** | $n \geq 0$ | Compact $C_n = 2|c_n|$ ($C_0 = |c_0|$) |

The one-sided spectrum uses amplitudes that are twice the complex coefficient magnitude (for $n > 0$) because the energy at frequency $n\omega_0$ is split between $c_n$ and $c_{-n}$.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Line spectra for different waveforms ---

T0 = 1.0
omega0 = 2 * np.pi / T0
t = np.linspace(0, T0, 10000, endpoint=False)
N_harm = 20

waveforms = {
    'Square Wave': lambda t: np.sign(np.sin(2 * np.pi * t / T0)),
    'Sawtooth Wave': lambda t: 2 * (t / T0 - np.floor(t / T0 + 0.5)),
    'Triangle Wave': lambda t: 2 * np.abs(2 * (t / T0 - np.floor(t / T0 + 0.5))) - 1,
    'Half-Rectified Sine': lambda t: np.maximum(0, np.sin(2 * np.pi * t / T0)),
}

fig, axes = plt.subplots(len(waveforms), 3, figsize=(16, 3.5 * len(waveforms)))

for row, (name, func) in enumerate(waveforms.items()):
    x = func(t)

    # Compute complex Fourier coefficients
    n_range = np.arange(-N_harm, N_harm + 1)
    cn = np.zeros(len(n_range), dtype=complex)
    for i, n in enumerate(n_range):
        cn[i] = np.mean(x * np.exp(-1j * n * omega0 * t))

    # Time-domain signal
    axes[row, 0].plot(t, x, 'b-', linewidth=1.5)
    axes[row, 0].set_title(f'{name}')
    axes[row, 0].set_xlabel('t')
    axes[row, 0].set_ylabel('x(t)')
    axes[row, 0].grid(True, alpha=0.3)

    # Amplitude spectrum (two-sided)
    axes[row, 1].stem(n_range, np.abs(cn), linefmt='r-', markerfmt='ro', basefmt='k-')
    axes[row, 1].set_title(f'Amplitude Spectrum $|c_n|$')
    axes[row, 1].set_xlabel('Harmonic n')
    axes[row, 1].set_ylabel('$|c_n|$')
    axes[row, 1].grid(True, alpha=0.3)

    # Phase spectrum (two-sided)
    phase = np.angle(cn)
    # Zero out phase for negligible coefficients
    phase[np.abs(cn) < 1e-10] = 0
    axes[row, 2].stem(n_range, phase * 180 / np.pi, linefmt='g-', markerfmt='go',
                      basefmt='k-')
    axes[row, 2].set_title(f'Phase Spectrum $\\angle c_n$')
    axes[row, 2].set_xlabel('Harmonic n')
    axes[row, 2].set_ylabel('Phase (degrees)')
    axes[row, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('line_spectra.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 9. Applications: Standard Waveform Decompositions

### 9.1 Fourier Series Approximation Gallery

Let us visualize how Fourier series progressively reconstruct standard waveforms.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Progressive Fourier reconstruction ---

T0 = 2 * np.pi
omega0 = 1.0
t = np.linspace(-np.pi, 3 * np.pi, 2000)

def fourier_square(t, N):
    """Square wave: sum of (4/n*pi)*sin(n*t) for odd n."""
    result = np.zeros_like(t)
    for n in range(1, N + 1, 2):
        result += (4 / (n * np.pi)) * np.sin(n * t)
    return result

def fourier_sawtooth(t, N):
    """Sawtooth: sum of (-2/n*pi)*(-1)^n * sin(n*t)."""
    result = np.zeros_like(t)
    for n in range(1, N + 1):
        result += (2 / (n * np.pi)) * ((-1)**(n + 1)) * np.sin(n * t)
    return result

def fourier_triangle(t, N):
    """Triangle: sum of (8/n^2*pi^2)*(-1)^k * cos(n*t) for odd n."""
    result = np.zeros_like(t)
    k = 0
    for n in range(1, N + 1, 2):
        result += (8 / (n**2 * np.pi**2)) * ((-1)**k) * np.cos(n * t)
        k += 1
    return result

# True waveforms
square_true = np.sign(np.sin(t))
sawtooth_true = (t + np.pi) % (2 * np.pi) / np.pi - 1
triangle_true = 2 * np.abs(2 * ((t + np.pi) / (2 * np.pi) - np.floor((t + np.pi) / (2 * np.pi) + 0.5))) - 1

waveforms = [
    ('Square Wave', square_true, fourier_square),
    ('Sawtooth Wave', sawtooth_true, fourier_sawtooth),
    ('Triangle Wave', triangle_true, fourier_triangle),
]

N_terms_list = [1, 3, 7, 21]

fig, axes = plt.subplots(len(waveforms), len(N_terms_list), figsize=(18, 10))

for row, (name, x_true, fourier_func) in enumerate(waveforms):
    for col, N in enumerate(N_terms_list):
        x_approx = fourier_func(t, N)
        axes[row, col].plot(t / np.pi, x_true, 'b--', linewidth=1, alpha=0.4,
                           label='True')
        axes[row, col].plot(t / np.pi, x_approx, 'r-', linewidth=1.5,
                           label=f'N={N}')
        axes[row, col].set_ylim([-1.5, 1.5])
        axes[row, col].set_xlim([-0.5, 2.5])
        if row == 0:
            axes[row, col].set_title(f'N = {N} harmonics')
        if col == 0:
            axes[row, col].set_ylabel(name)
        axes[row, col].legend(fontsize=8)
        axes[row, col].grid(True, alpha=0.3)

plt.suptitle('Fourier Series Convergence for Standard Waveforms', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('fourier_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 9.2 Physical Application: Heat Conduction

The Fourier series was originally developed by Joseph Fourier to solve the heat equation. Consider a metal rod of length $L$ with initial temperature distribution $f(x)$ and both ends held at 0 degrees.

The temperature at position $x$ and time $t$ is:

$$u(x, t) = \sum_{n=1}^{\infty} b_n \sin\left(\frac{n\pi x}{L}\right) e^{-\alpha (n\pi/L)^2 t}$$

where $b_n$ are the Fourier sine series coefficients of $f(x)$ and $\alpha$ is the thermal diffusivity.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Heat equation solution using Fourier series ---

L = 1.0          # rod length
alpha = 0.01     # thermal diffusivity
N_terms = 50     # Fourier terms

# Initial temperature: step function (hot in the middle)
def initial_temp(x):
    return np.where((x > 0.25) & (x < 0.75), 1.0, 0.0)

# Compute Fourier sine coefficients
x_int = np.linspace(0, L, 10000)
dx = x_int[1] - x_int[0]
f_x = initial_temp(x_int)

bn = np.zeros(N_terms + 1)
for n in range(1, N_terms + 1):
    bn[n] = (2 / L) * np.trapz(f_x * np.sin(n * np.pi * x_int / L), x_int)

# Solution at various times
x = np.linspace(0, L, 500)
times = [0, 0.5, 2, 5, 10, 20]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for ax, t_val in zip(axes.flat, times):
    u = np.zeros_like(x)
    for n in range(1, N_terms + 1):
        u += bn[n] * np.sin(n * np.pi * x / L) * np.exp(-alpha * (n * np.pi / L)**2 * t_val)

    ax.plot(x, u, 'r-', linewidth=2)
    ax.fill_between(x, u, alpha=0.2, color='red')
    ax.set_title(f't = {t_val}')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Temperature u(x, t)')
    ax.set_ylim([-0.1, 1.1])
    ax.grid(True, alpha=0.3)

plt.suptitle('Heat Equation: Fourier Series Solution', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('heat_equation_fourier.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 9.3 Signal Synthesis: Additive Sound Synthesis

Fourier series is the theoretical basis for **additive synthesis** in audio engineering: building complex sounds by summing pure tones (sinusoids).

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Additive sound synthesis ---

fs = 44100  # CD quality sampling rate
duration = 0.5
t = np.arange(int(fs * duration)) / fs

# Synthesize different timbres at A4 (440 Hz)
f0 = 440

def synthesize(harmonics, amplitudes, phases=None):
    """Additive synthesis from harmonic specification."""
    if phases is None:
        phases = np.zeros(len(harmonics))
    signal = np.zeros_like(t)
    for n, amp, phi in zip(harmonics, amplitudes, phases):
        signal += amp * np.sin(2 * np.pi * n * f0 * t + phi)
    # Normalize
    signal /= np.max(np.abs(signal))
    return signal

# Different timbres
timbres = {
    'Pure Tone (1 harmonic)': synthesize([1], [1.0]),
    'Clarinet-like (odd harmonics)': synthesize(
        [1, 3, 5, 7, 9, 11],
        [1.0, 0.75, 0.5, 0.14, 0.5, 0.12]
    ),
    'Bright/Sawtooth (all harmonics)': synthesize(
        range(1, 16),
        [1.0 / n for n in range(1, 16)]
    ),
    'Organ-like (specific harmonics)': synthesize(
        [1, 2, 3, 4, 6, 8],
        [1.0, 0.5, 0.3, 0.25, 0.1, 0.05]
    ),
}

fig, axes = plt.subplots(len(timbres), 2, figsize=(16, 3 * len(timbres)))

for row, (name, signal) in enumerate(timbres.items()):
    # Time domain (show 3 cycles)
    n_show = int(3 * fs / f0)
    axes[row, 0].plot(t[:n_show] * 1000, signal[:n_show], 'b-', linewidth=1.5)
    axes[row, 0].set_title(f'{name} — Time Domain')
    axes[row, 0].set_xlabel('Time (ms)')
    axes[row, 0].set_ylabel('Amplitude')
    axes[row, 0].grid(True, alpha=0.3)

    # Frequency domain
    N = len(signal)
    freqs = np.fft.rfftfreq(N, 1 / fs)
    spectrum = np.abs(np.fft.rfft(signal)) / N * 2
    axes[row, 1].stem(freqs[:30 * len(t) // fs],
                      spectrum[:30 * len(t) // fs],
                      linefmt='r-', markerfmt='ro', basefmt='k-')
    axes[row, 1].set_title(f'{name} — Spectrum')
    axes[row, 1].set_xlabel('Frequency (Hz)')
    axes[row, 1].set_ylabel('Amplitude')
    axes[row, 1].set_xlim([0, 8000])
    axes[row, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('additive_synthesis.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 10. Discrete-Time Fourier Series

### 10.1 Definition

A discrete-time periodic signal $x[n]$ with period $N$ can be represented as:

$$x[n] = \sum_{k=0}^{N-1} c_k \, e^{j(2\pi/N)kn}$$

where the **DT Fourier series coefficients** are:

$$c_k = \frac{1}{N} \sum_{n=0}^{N-1} x[n] \, e^{-j(2\pi/N)kn}, \quad k = 0, 1, \ldots, N-1$$

### 10.2 Key Differences from CT Fourier Series

| Property | Continuous-Time | Discrete-Time |
|----------|----------------|---------------|
| Number of harmonics | Infinite | Finite ($N$) |
| Coefficient index range | $n \in \mathbb{Z}$ (infinite) | $k = 0, 1, \ldots, N-1$ (finite) |
| Series is | Approximation (truncated) | **Exact** representation |
| Convergence issues | Gibbs phenomenon, Dirichlet conditions | None (finite sum) |

The DT Fourier series is always exact with $N$ coefficients because a periodic sequence with period $N$ has only $N$ independent values.

### 10.3 Connection to DFT

The DT Fourier series coefficients are essentially the **Discrete Fourier Transform (DFT)** of one period, scaled by $1/N$. This connection is central to Lesson 06.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Discrete-time Fourier series ---

N = 16  # period
n = np.arange(N)

# Example: discrete-time square wave
x = np.ones(N)
x[N//2:] = -1

# DT Fourier coefficients
ck = np.zeros(N, dtype=complex)
for k in range(N):
    ck[k] = (1 / N) * np.sum(x * np.exp(-1j * 2 * np.pi * k * n / N))

# Reconstruction (should be exact)
x_reconstructed = np.zeros(N, dtype=complex)
for k in range(N):
    x_reconstructed += ck[k] * np.exp(1j * 2 * np.pi * k * n / N)

print("Original:      ", x)
print("Reconstructed: ", np.real(x_reconstructed).round(10))
print("Max error:     ", np.max(np.abs(x - np.real(x_reconstructed))))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].stem(n, x, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0].set_title(f'DT Signal $x[n]$ (period N={N})')
axes[0].set_xlabel('n')
axes[0].grid(True, alpha=0.3)

axes[1].stem(np.arange(N), np.abs(ck), linefmt='r-', markerfmt='ro', basefmt='k-')
axes[1].set_title('Amplitude $|c_k|$')
axes[1].set_xlabel('k')
axes[1].grid(True, alpha=0.3)

axes[2].stem(np.arange(N), np.angle(ck) * 180 / np.pi, linefmt='g-', markerfmt='go',
             basefmt='k-')
axes[2].set_title('Phase $\\angle c_k$ (degrees)')
axes[2].set_xlabel('k')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dt_fourier_series.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 11. Python Examples

### 11.1 Comprehensive Fourier Series Analyzer

```python
import numpy as np
import matplotlib.pyplot as plt

class FourierSeriesAnalyzer:
    """Complete Fourier series analysis toolkit."""

    def __init__(self, signal_func, T0, name="Signal"):
        self.signal_func = signal_func
        self.T0 = T0
        self.omega0 = 2 * np.pi / T0
        self.name = name
        self.t = np.linspace(0, T0, 10000, endpoint=False)
        self.x = signal_func(self.t)

    def compute_coefficients(self, N_max):
        """Compute complex Fourier coefficients c_n for |n| <= N_max."""
        n_range = np.arange(-N_max, N_max + 1)
        cn = np.zeros(len(n_range), dtype=complex)
        for i, n in enumerate(n_range):
            cn[i] = np.mean(self.x * np.exp(-1j * n * self.omega0 * self.t))
        return n_range, cn

    def reconstruct(self, t, n_range, cn):
        """Reconstruct signal from Fourier coefficients."""
        x_approx = np.zeros_like(t, dtype=complex)
        for n, c in zip(n_range, cn):
            x_approx += c * np.exp(1j * n * self.omega0 * t)
        return np.real(x_approx)

    def analyze(self, N_max=30, N_show=[1, 3, 7, 15, 30]):
        """Complete analysis with plots."""
        n_range, cn = self.compute_coefficients(N_max)

        fig = plt.figure(figsize=(16, 14))

        # 1. Original signal
        ax1 = fig.add_subplot(3, 2, 1)
        t_plot = np.linspace(-self.T0/2, 1.5*self.T0, 3000)
        ax1.plot(t_plot / self.T0, self.signal_func(t_plot), 'b-', linewidth=2)
        ax1.set_title(f'{self.name} — Time Domain')
        ax1.set_xlabel('t / T₀')
        ax1.set_ylabel('x(t)')
        ax1.grid(True, alpha=0.3)

        # 2. Amplitude spectrum
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.stem(n_range, np.abs(cn), linefmt='r-', markerfmt='ro', basefmt='k-')
        ax2.set_title('Amplitude Spectrum $|c_n|$')
        ax2.set_xlabel('Harmonic n')
        ax2.set_ylabel('$|c_n|$')
        ax2.grid(True, alpha=0.3)

        # 3. Progressive reconstruction
        ax3 = fig.add_subplot(3, 2, 3)
        t_recon = np.linspace(0, 2 * self.T0, 2000)
        ax3.plot(t_recon / self.T0, self.signal_func(t_recon), 'k--',
                linewidth=1, alpha=0.4, label='True')
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(N_show)))
        for N, color in zip(N_show, colors):
            n_sub = np.arange(-N, N + 1)
            cn_sub = cn[(n_range >= -N) & (n_range <= N)]
            x_approx = self.reconstruct(t_recon, n_sub, cn_sub)
            ax3.plot(t_recon / self.T0, x_approx, color=color,
                    linewidth=1, label=f'N={N}')
        ax3.set_title('Progressive Reconstruction')
        ax3.set_xlabel('t / T₀')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 4. Reconstruction error vs N
        ax4 = fig.add_subplot(3, 2, 4)
        N_test = np.arange(1, N_max + 1)
        errors = []
        for N in N_test:
            n_sub = np.arange(-N, N + 1)
            cn_sub = cn[(n_range >= -N) & (n_range <= N)]
            x_approx = self.reconstruct(self.t, n_sub, cn_sub)
            mse = np.mean((self.x - x_approx)**2)
            errors.append(mse)
        ax4.semilogy(N_test, errors, 'b-o', markersize=3)
        ax4.set_title('Mean Squared Error vs N')
        ax4.set_xlabel('Number of harmonics N')
        ax4.set_ylabel('MSE')
        ax4.grid(True, alpha=0.3)

        # 5. Power spectrum
        ax5 = fig.add_subplot(3, 2, 5)
        P_total = np.mean(self.x**2)
        power_contributions = np.abs(cn)**2
        n_pos = n_range[n_range >= 0]
        cn_pos = cn[n_range >= 0]
        P_cumulative = np.cumsum(np.abs(cn_pos)**2)
        # Add symmetric part
        for i, n in enumerate(n_pos):
            if n > 0:
                P_cumulative[i:] += np.abs(cn[n_range == -n])[0]**2
        ax5.plot(n_pos, P_cumulative / P_total * 100, 'r-o', markersize=3)
        ax5.axhline(y=99, color='gray', linestyle='--', label='99%')
        ax5.set_title("Cumulative Power (Parseval's)")
        ax5.set_xlabel('Max harmonic')
        ax5.set_ylabel('% of total power')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Phase spectrum
        ax6 = fig.add_subplot(3, 2, 6)
        phase = np.angle(cn)
        phase[np.abs(cn) < 1e-10] = 0
        ax6.stem(n_range, phase * 180 / np.pi, linefmt='g-', markerfmt='go',
                basefmt='k-')
        ax6.set_title('Phase Spectrum $\\angle c_n$ (degrees)')
        ax6.set_xlabel('Harmonic n')
        ax6.set_ylabel('Phase (°)')
        ax6.grid(True, alpha=0.3)

        plt.suptitle(f'Fourier Series Analysis: {self.name}', fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(f'fourier_analysis_{self.name.replace(" ", "_")}.png',
                    dpi=150, bbox_inches='tight')
        plt.show()


# Analyze different waveforms
T0 = 1.0

# Square wave
analyzer = FourierSeriesAnalyzer(
    lambda t: np.sign(np.sin(2 * np.pi * t / T0)),
    T0, "Square Wave"
)
analyzer.analyze()

# Triangle wave
analyzer = FourierSeriesAnalyzer(
    lambda t: 2 * np.abs(2 * (t / T0 - np.floor(t / T0 + 0.5))) - 1,
    T0, "Triangle Wave"
)
analyzer.analyze()
```

### 11.2 Gibbs Phenomenon Mitigation

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Gibbs phenomenon mitigation using sigma factors ---

T0 = 2 * np.pi
t = np.linspace(-0.5, 2.5 * np.pi, 2000)
N = 30

# Square wave Fourier coefficients
def cn_square(n):
    if n == 0:
        return 0
    if n % 2 == 0:
        return 0
    return -2j / (n * np.pi)

# Standard partial sum
def partial_sum(t, N, window_func=None):
    result = np.zeros_like(t, dtype=complex)
    for n in range(-N, N + 1):
        c = cn_square(n)
        if window_func is not None:
            c *= window_func(n, N)
        result += c * np.exp(1j * n * t)
    return np.real(result)

# Sigma factors for Gibbs mitigation
def lanczos_sigma(n, N):
    """Lanczos sigma factor: sinc(n/N)."""
    if n == 0:
        return 1.0
    return np.sinc(n / N)

def fejer_sigma(n, N):
    """Fejer (Cesaro) kernel: 1 - |n|/N."""
    return max(0, 1 - abs(n) / N)

def raised_cosine_sigma(n, N):
    """Raised cosine (Hanning-like)."""
    return 0.5 * (1 + np.cos(np.pi * n / N))

# Compare methods
x_true = np.sign(np.sin(t))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

methods = [
    ("Standard Fourier (N=30)", None),
    ("Lanczos sigma factors", lanczos_sigma),
    ("Fejer (Cesaro) summation", fejer_sigma),
    ("Raised cosine window", raised_cosine_sigma),
]

for ax, (name, sigma) in zip(axes.flat, methods):
    S = partial_sum(t, N, sigma)
    ax.plot(t / np.pi, x_true, 'b--', linewidth=1, alpha=0.4, label='True')
    ax.plot(t / np.pi, S, 'r-', linewidth=1.5, label=name)
    ax.set_ylim([-1.4, 1.4])
    ax.set_xlim([-0.15, 2.2])
    ax.set_title(name)
    ax.set_xlabel('$t / \\pi$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Gibbs Phenomenon Mitigation Methods', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('gibbs_mitigation.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 11.3 Interactive Coefficient Explorer

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Effect of modifying individual Fourier coefficients ---

T0 = 1.0
omega0 = 2 * np.pi / T0
t = np.linspace(0, 2 * T0, 1000)
N_harm = 10

# Start with square wave coefficients
cn_original = np.zeros(2 * N_harm + 1, dtype=complex)
n_range = np.arange(-N_harm, N_harm + 1)

for i, n in enumerate(n_range):
    if n != 0 and n % 2 != 0:
        cn_original[i] = -2j / (n * np.pi)

def reconstruct(cn, t, n_range, omega0):
    x = np.zeros_like(t, dtype=complex)
    for c, n in zip(cn, n_range):
        x += c * np.exp(1j * n * omega0 * t)
    return np.real(x)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Original
x_orig = reconstruct(cn_original, t, n_range, omega0)
axes[0, 0].plot(t / T0, x_orig, 'b-', linewidth=2)
axes[0, 0].set_title('Original Square Wave (N=10)')
axes[0, 0].grid(True, alpha=0.3)

# Remove 3rd harmonic
cn_no3 = cn_original.copy()
cn_no3[n_range == 3] = 0
cn_no3[n_range == -3] = 0
x_no3 = reconstruct(cn_no3, t, n_range, omega0)
axes[0, 1].plot(t / T0, x_orig, 'b--', alpha=0.3, label='Original')
axes[0, 1].plot(t / T0, x_no3, 'r-', linewidth=2, label='No 3rd harmonic')
axes[0, 1].set_title('Remove 3rd Harmonic')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Double the 3rd harmonic
cn_double3 = cn_original.copy()
cn_double3[n_range == 3] *= 2
cn_double3[n_range == -3] *= 2
x_double3 = reconstruct(cn_double3, t, n_range, omega0)
axes[1, 0].plot(t / T0, x_orig, 'b--', alpha=0.3, label='Original')
axes[1, 0].plot(t / T0, x_double3, 'g-', linewidth=2, label='3rd harmonic x2')
axes[1, 0].set_title('Double the 3rd Harmonic')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Add even harmonics
cn_even = cn_original.copy()
for n in [2, 4, 6]:
    idx_pos = np.where(n_range == n)[0][0]
    idx_neg = np.where(n_range == -n)[0][0]
    cn_even[idx_pos] = 0.3 / n
    cn_even[idx_neg] = 0.3 / n
x_even = reconstruct(cn_even, t, n_range, omega0)
axes[1, 1].plot(t / T0, x_orig, 'b--', alpha=0.3, label='Original')
axes[1, 1].plot(t / T0, x_even, 'm-', linewidth=2, label='+ even harmonics')
axes[1, 1].set_title('Add Even Harmonics (breaks half-wave symmetry)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Phase shift all harmonics by pi/4
cn_phased = cn_original * np.exp(1j * np.pi / 4 * np.abs(n_range))
x_phased = reconstruct(cn_phased, t, n_range, omega0)
axes[2, 0].plot(t / T0, x_orig, 'b--', alpha=0.3, label='Original')
axes[2, 0].plot(t / T0, x_phased, 'orange', linewidth=2, label='Phase shifted')
axes[2, 0].set_title('Phase Shift Each Harmonic by $n \\cdot \\pi/4$')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# Random phases (same magnitude)
np.random.seed(42)
cn_random_phase = np.abs(cn_original) * np.exp(1j * 2 * np.pi * np.random.rand(len(cn_original)))
# Keep conjugate symmetry for real output
for i, n in enumerate(n_range):
    if n < 0:
        cn_random_phase[i] = np.conj(cn_random_phase[n_range == -n][0])
x_random = reconstruct(cn_random_phase, t, n_range, omega0)
axes[2, 1].plot(t / T0, x_orig, 'b--', alpha=0.3, label='Original')
axes[2, 1].plot(t / T0, x_random, 'cyan', linewidth=2, label='Random phases')
axes[2, 1].set_title('Random Phases (same magnitudes)')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

for ax in axes.flat:
    ax.set_xlabel('$t / T_0$')
    ax.set_ylim([-1.8, 1.8])

plt.suptitle('Effect of Modifying Fourier Coefficients', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('coefficient_explorer.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 12. Summary

### Key Formulas

| Form | Series | Coefficients |
|------|--------|-------------|
| Trigonometric | $x(t) = a_0 + \sum_{n=1}^{\infty}[a_n\cos(n\omega_0 t) + b_n\sin(n\omega_0 t)]$ | $a_0 = \frac{1}{T_0}\int x \, dt$, $a_n = \frac{2}{T_0}\int x\cos(n\omega_0 t) \, dt$, $b_n = \frac{2}{T_0}\int x\sin(n\omega_0 t) \, dt$ |
| Complex exponential | $x(t) = \sum_{n=-\infty}^{\infty} c_n e^{jn\omega_0 t}$ | $c_n = \frac{1}{T_0}\int x(t) e^{-jn\omega_0 t} \, dt$ |
| Compact | $x(t) = C_0 + \sum_{n=1}^{\infty} C_n\cos(n\omega_0 t + \phi_n)$ | $C_n = \sqrt{a_n^2 + b_n^2}$, $\phi_n = -\arctan(b_n/a_n)$ |

### Conceptual Hierarchy

```
            Periodic Signal x(t)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  Dirichlet Conditions    Fourier Coefficients
  (convergence check)     cn or (an, bn)
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              Line Spectra  Parseval's   Reconstruction
              |cn| vs n    Power sum     Partial sums
                                        Gibbs phenomenon
```

### Key Takeaways

1. **Fourier series** decomposes periodic signals into harmonically related sinusoids
2. **Orthogonality** of trigonometric functions is the mathematical foundation
3. The **complex exponential form** is preferred in signal processing (eigenfunctions of LTI systems)
4. Coefficient **decay rate** reflects signal smoothness ($1/n$ for discontinuous, $1/n^2$ for continuous)
5. **Gibbs phenomenon**: 9% overshoot at discontinuities cannot be eliminated by adding more terms
6. **Parseval's theorem**: power is conserved between time and frequency domains
7. **Line spectra** give a complete frequency-domain picture of periodic signals

---

## 13. Exercises

### Exercise 1: Fourier Coefficient Computation

Compute the Fourier series coefficients (both trigonometric and complex exponential) for:

1. $x(t) = |\sin(\omega_0 t)|$ (full-wave rectified sine)
2. $x(t) = \cos^2(\omega_0 t)$ (hint: use trigonometric identity first)
3. A pulse train: $x(t) = 1$ for $|t| < \tau/2$, $x(t) = 0$ for $\tau/2 < |t| < T_0/2$, periodic with period $T_0$

### Exercise 2: Symmetry Exploitation

For each signal, identify the symmetry type (even, odd, half-wave) and determine which Fourier coefficients are zero without computing:

1. $x(t) = t^2$ on $[-\pi, \pi]$, periodic
2. $x(t) = t$ on $[-\pi, \pi]$, periodic
3. $x(t) = |t|$ on $[-\pi, \pi]$, periodic
4. The staircase: $x(t) = +1$ for $0 < t < \pi$, $x(t) = -1$ for $-\pi < t < 0$

### Exercise 3: Gibbs Phenomenon Investigation

1. Compute and plot the Fourier partial sum $S_N(t)$ of a square wave for $N = 5, 21, 101$
2. For each $N$, find the peak overshoot numerically
3. Show that the overshoot percentage converges to approximately 8.95%
4. Implement Lanczos sigma factors and demonstrate the reduced overshoot
5. Compare the MSE of standard Fourier vs. Lanczos-windowed partial sums

### Exercise 4: Parseval's Theorem Applications

1. Use Parseval's theorem to compute $\sum_{n=1}^{\infty} \frac{1}{n^2}$ from the sawtooth wave
2. Use Parseval's theorem to compute $\sum_{k=0}^{\infty} \frac{1}{(2k+1)^4}$ from the triangle wave
3. For a pulse train with duty cycle $d = \tau/T_0$:
   - Compute the power as a function of $d$
   - Plot the power spectrum for $d = 0.1, 0.25, 0.5$
   - Verify Parseval's theorem numerically for each case

### Exercise 5: Signal Reconstruction Challenge

Given only the amplitude spectrum $|c_n|$ and phase spectrum $\angle c_n$ (for $|n| \leq 10$) of a periodic signal with $T_0 = 1$ s:

| $n$ | $|c_n|$ | $\angle c_n$ (rad) |
|-----|---------|-------------------|
| 0 | 0.5 | 0 |
| 1 | 0.8 | $-\pi/4$ |
| 2 | 0.3 | $-\pi/2$ |
| 3 | 0.6 | $\pi/3$ |
| 5 | 0.2 | $-\pi/6$ |
| 7 | 0.1 | $\pi/4$ |

1. Reconstruct and plot $x(t)$
2. Compute the average power of $x(t)$ using Parseval's theorem
3. What happens to the signal shape if all phases are set to zero?
4. What happens if all phases are randomized?

### Exercise 6: Fourier Series of a Real-World Signal

1. Generate a sawtooth wave at 100 Hz, sampled at 44100 Hz, for 0.1 seconds
2. Compute its Fourier coefficients using both the analytical formula and numerical integration
3. Compare the two results
4. Reconstruct the signal using only the first 5, 10, 20 harmonics
5. Compute the SNR (in dB) of each reconstruction compared to the original

### Exercise 7: Fourier Series and LTI Systems

A periodic input $x(t) = \sum c_n e^{jn\omega_0 t}$ is applied to an LTI system with frequency response $H(j\omega)$.

1. Show that the output is also periodic with the same period
2. Show that the output Fourier coefficients are $d_n = c_n \cdot H(jn\omega_0)$
3. A square wave with $f_0 = 100$ Hz passes through an RC lowpass filter with cutoff frequency 500 Hz. Compute the output Fourier coefficients and reconstruct the output signal.
4. How does the filter affect the Gibbs phenomenon?

### Exercise 8: Discrete-Time Fourier Series

1. Generate a discrete-time periodic sequence $x[n]$ with period $N = 32$ representing a discrete square wave
2. Compute all $N = 32$ DT Fourier series coefficients
3. Verify that the reconstruction is exact
4. Compare with the DFT computed using `np.fft.fft`
5. Demonstrate that the DT Fourier series has no Gibbs phenomenon (explain why)

---

## 14. References

1. Oppenheim, A. V. & Willsky, A. S. *Signals and Systems* (2nd ed.), Ch. 3-4. Prentice Hall, 1997.
2. Haykin, S. & Van Veen, B. *Signals and Systems* (2nd ed.), Ch. 3, 6. Wiley, 2003.
3. Lathi, B. P. & Green, R. A. *Linear Systems and Signals* (3rd ed.), Ch. 6. Oxford University Press, 2018.
4. Boas, M. L. *Mathematical Methods in the Physical Sciences* (3rd ed.), Ch. 7. Wiley, 2006.
5. Smith, S. W. *The Scientist and Engineer's Guide to Digital Signal Processing*, Ch. 13. California Technical Publishing, 1997.

---

[Previous: 02. LTI Systems and Convolution](./02_LTI_Systems_and_Convolution.md) | [Next: 04. Continuous Fourier Transform](./04_Continuous_Fourier_Transform.md) | [Overview](./00_Overview.md)
