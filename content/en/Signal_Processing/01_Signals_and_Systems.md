# Signals and Systems

**Next**: [02. LTI Systems and Convolution](./02_LTI_Systems_and_Convolution.md)

---

Signals are all around us — the sound of a voice, the voltage across a circuit, the pixels in an image, the stock price over time. Signal processing gives us a rigorous mathematical framework for describing, analyzing, and manipulating these quantities. Before we can process signals, we need a precise language for describing them and the systems that act upon them.

This lesson establishes the foundational vocabulary: what signals are, how we classify them, what elementary building-block signals look like, how we transform signals through basic operations, and what properties characterize the systems that process them.

**Difficulty**: ⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Define continuous-time and discrete-time signals mathematically
2. Classify signals by their properties (deterministic/random, periodic/aperiodic, energy/power)
3. Recognize and generate fundamental signal types (impulse, step, exponential, sinusoid)
4. Perform basic signal operations (shifting, scaling, reflection)
5. Define the input-output model of a system
6. Identify and verify key system properties (linearity, time-invariance, causality, stability)
7. Apply BIBO stability criteria

---

## Table of Contents

1. [What Is a Signal?](#1-what-is-a-signal)
2. [Continuous-Time and Discrete-Time Signals](#2-continuous-time-and-discrete-time-signals)
3. [Signal Classifications](#3-signal-classifications)
4. [Fundamental Signals](#4-fundamental-signals)
5. [Energy and Power Signals](#5-energy-and-power-signals)
6. [Signal Operations](#6-signal-operations)
7. [What Is a System?](#7-what-is-a-system)
8. [System Properties](#8-system-properties)
9. [BIBO Stability](#9-bibo-stability)
10. [Python Examples](#10-python-examples)
11. [Summary](#11-summary)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. What Is a Signal?

A **signal** is a function that conveys information about a physical phenomenon. Mathematically, a signal is a mapping from an independent variable (usually time or space) to a dependent variable (amplitude, voltage, intensity, etc.).

### Formal Definition

A **continuous-time (CT) signal** is a function:

$$x(t) : \mathbb{R} \to \mathbb{R} \quad (\text{or } \mathbb{C})$$

where $t$ is a continuous real variable, typically representing time.

A **discrete-time (DT) signal** is a sequence:

$$x[n] : \mathbb{Z} \to \mathbb{R} \quad (\text{or } \mathbb{C})$$

where $n$ is an integer index.

> **Notation Convention**: Throughout this course, we use parentheses $x(t)$ for continuous-time signals and square brackets $x[n]$ for discrete-time signals. This is a universal convention in the signal processing literature.

### Examples of Signals

| Domain | Signal | Independent Variable | Dependent Variable |
|--------|--------|---------------------|-------------------|
| Audio | Speech waveform | Time | Air pressure / voltage |
| Image | Photograph | Spatial coordinates $(x, y)$ | Intensity / color |
| Video | Movie | Spatial + time $(x, y, t)$ | Intensity |
| Finance | Stock price | Time (discrete: days) | Price |
| Seismology | Seismogram | Time | Ground displacement |
| Biomedical | ECG | Time | Voltage |
| Communications | Modulated carrier | Time | Electromagnetic field |

---

## 2. Continuous-Time and Discrete-Time Signals

### 2.1 Continuous-Time Signals

A continuous-time signal $x(t)$ is defined for every value of $t$ in some interval (or all of $\mathbb{R}$). The signal amplitude can take any value in a continuous range.

**Examples**:
- Analog voltage: $x(t) = 3\sin(2\pi \cdot 440 \cdot t)$ (a 440 Hz tone)
- Exponential decay: $x(t) = e^{-t}u(t)$ where $u(t)$ is the unit step
- Gaussian pulse: $x(t) = e^{-t^2/2\sigma^2}$

### 2.2 Discrete-Time Signals

A discrete-time signal $x[n]$ is defined only at integer values of $n$. It often arises from sampling a continuous-time signal:

$$x[n] = x_c(nT_s)$$

where $T_s$ is the sampling period and $f_s = 1/T_s$ is the sampling rate.

**Examples**:
- Daily temperature readings
- Digitized audio (44,100 samples/second for CD quality)
- Pixel rows in an image

### 2.3 Analog vs. Digital

It is important to distinguish between the time axis and the amplitude axis:

| | Continuous amplitude | Discrete amplitude |
|---|---|---|
| **Continuous time** | Analog signal | — |
| **Discrete time** | Sampled signal | Digital signal |

A **digital signal** is both discrete in time and quantized in amplitude — this is what computers actually process.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Continuous-time vs discrete-time signal visualization ---
t = np.linspace(0, 0.01, 1000)      # "continuous" (densely sampled)
f0 = 440                              # 440 Hz (A4 note)
x_ct = np.sin(2 * np.pi * f0 * t)    # continuous-time sinusoid

# Discrete-time version: sampled at 8000 Hz
fs = 8000
n = np.arange(0, int(0.01 * fs))     # sample indices
Ts = 1 / fs
x_dt = np.sin(2 * np.pi * f0 * n * Ts)  # discrete-time sinusoid

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Continuous-time
axes[0].plot(t * 1000, x_ct, 'b-', linewidth=1.5)
axes[0].set_title('Continuous-Time Signal: $x(t) = \\sin(2\\pi \\cdot 440 \\cdot t)$')
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 10])

# Discrete-time
axes[1].stem(n * Ts * 1000, x_dt, linefmt='r-', markerfmt='ro', basefmt='k-')
axes[1].set_title(f'Discrete-Time Signal: $x[n] = \\sin(2\\pi \\cdot 440 \\cdot n / {fs})$')
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Amplitude')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 10])

plt.tight_layout()
plt.savefig('ct_vs_dt_signal.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 3. Signal Classifications

Signals can be classified along several orthogonal axes. Understanding these classifications helps choose the right analysis tools.

### 3.1 Deterministic vs. Random

| Property | Deterministic | Random (Stochastic) |
|----------|--------------|-------------------|
| Definition | Completely specified by a mathematical expression | Described by statistical properties |
| Prediction | Future values can be computed exactly | Future values can only be predicted probabilistically |
| Example | $x(t) = A\cos(\omega_0 t + \phi)$ | Thermal noise, speech |
| Analysis tool | Transform methods (Fourier, Laplace) | Correlation, power spectral density |

### 3.2 Periodic vs. Aperiodic

A continuous-time signal $x(t)$ is **periodic** with period $T > 0$ if:

$$x(t + T) = x(t) \quad \text{for all } t$$

The smallest such $T$ is the **fundamental period** $T_0$, and $f_0 = 1/T_0$ is the **fundamental frequency**.

A discrete-time signal $x[n]$ is periodic with period $N$ (a positive integer) if:

$$x[n + N] = x[n] \quad \text{for all } n$$

> **Important**: A continuous-time sinusoid $\cos(\omega_0 t)$ is always periodic. A discrete-time sinusoid $\cos(\omega_0 n)$ is periodic **only if** $\omega_0 / (2\pi)$ is a rational number.

### 3.3 Even and Odd Signals

Any signal can be decomposed into even and odd components:

**Even signal**: $x_e(t) = x_e(-t)$ (symmetric about $t = 0$)

**Odd signal**: $x_o(t) = -x_o(-t)$ (anti-symmetric about $t = 0$)

**Decomposition**:

$$x_e(t) = \frac{x(t) + x(-t)}{2}, \qquad x_o(t) = \frac{x(t) - x(-t)}{2}$$

so that $x(t) = x_e(t) + x_o(t)$.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Even/Odd decomposition ---
t = np.linspace(-3, 3, 1000)

# Original signal: asymmetric exponential
x = np.exp(-t) * (t >= 0)  # right-sided exponential

# Even and odd parts
x_flipped = np.exp(t) * (t <= 0)  # x(-t)
x_even = 0.5 * (x + x_flipped)
x_odd = 0.5 * (x - x_flipped)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(t, x, 'b-', linewidth=2)
axes[0].set_title('Original: $x(t) = e^{-t}u(t)$')
axes[0].set_xlabel('t')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, x_even, 'r-', linewidth=2)
axes[1].set_title('Even part: $x_e(t)$')
axes[1].set_xlabel('t')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, x_odd, 'g-', linewidth=2)
axes[2].set_title('Odd part: $x_o(t)$')
axes[2].set_xlabel('t')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('even_odd_decomposition.png', dpi=150, bbox_inches='tight')
plt.show()

# Verify reconstruction
print("Reconstruction error:", np.max(np.abs(x - (x_even + x_odd))))
```

### 3.4 Real vs. Complex

Most physical signals are real-valued, but complex signals are essential in signal processing:

$$z(t) = x(t) + jy(t) = A(t)e^{j\phi(t)}$$

where $A(t) = |z(t)|$ is the **envelope** (instantaneous amplitude) and $\phi(t)$ is the **instantaneous phase**. Complex signals arise naturally in communications (baseband representation), radar, and Fourier analysis.

### 3.5 Causal, Anticausal, and Noncausal

- **Causal**: $x(t) = 0$ for $t < 0$
- **Anticausal**: $x(t) = 0$ for $t > 0$
- **Noncausal**: nonzero for both $t < 0$ and $t > 0$

---

## 4. Fundamental Signals

Several elementary signals serve as building blocks for constructing and analyzing more complex signals.

### 4.1 Unit Impulse (Dirac Delta / Kronecker Delta)

**Continuous-time** — the **Dirac delta function** $\delta(t)$:

$$\delta(t) = 0 \text{ for } t \neq 0, \qquad \int_{-\infty}^{\infty} \delta(t) \, dt = 1$$

The defining property is the **sifting property**:

$$\int_{-\infty}^{\infty} x(t)\delta(t - t_0) \, dt = x(t_0)$$

This means $\delta(t)$ "picks out" the value of a function at a specific point.

**Discrete-time** — the **Kronecker delta** (or unit sample) $\delta[n]$:

$$\delta[n] = \begin{cases} 1 & n = 0 \\ 0 & n \neq 0 \end{cases}$$

with the sifting property:

$$\sum_{k=-\infty}^{\infty} x[k]\delta[n - k] = x[n]$$

### 4.2 Unit Step

**Continuous-time**:

$$u(t) = \begin{cases} 1 & t > 0 \\ 0 & t < 0 \end{cases}$$

(The value at $t = 0$ is often defined as $1/2$ but this rarely matters in practice.)

**Discrete-time**:

$$u[n] = \begin{cases} 1 & n \geq 0 \\ 0 & n < 0 \end{cases}$$

**Relationship**: $u(t) = \int_{-\infty}^{t} \delta(\tau) \, d\tau$ and $\delta(t) = \frac{du(t)}{dt}$ (in the distributional sense).

For discrete-time: $u[n] = \sum_{k=-\infty}^{n} \delta[k]$ and $\delta[n] = u[n] - u[n-1]$.

### 4.3 Rectangular Pulse

$$\text{rect}\left(\frac{t}{\tau}\right) = \begin{cases} 1 & |t| < \tau/2 \\ 1/2 & |t| = \tau/2 \\ 0 & |t| > \tau/2 \end{cases} = u\left(t + \frac{\tau}{2}\right) - u\left(t - \frac{\tau}{2}\right)$$

The rectangular pulse is fundamental in sampling, windowing, and communications.

### 4.4 Real Exponential

**Continuous-time**: $x(t) = Ce^{at}$ where $C, a \in \mathbb{R}$

- $a > 0$: growing exponential
- $a < 0$: decaying exponential (e.g., RC circuit discharge)
- $a = 0$: constant $C$

**Discrete-time**: $x[n] = Ca^n$

- $|a| > 1$: growing
- $|a| < 1$: decaying
- $|a| = 1$: constant magnitude

### 4.5 Sinusoidal Signal

**Continuous-time**:

$$x(t) = A\cos(\omega_0 t + \phi)$$

where:
- $A$ = amplitude
- $\omega_0 = 2\pi f_0$ = angular frequency (rad/s)
- $f_0$ = frequency (Hz)
- $T_0 = 1/f_0$ = period (s)
- $\phi$ = phase (rad)

**Discrete-time**:

$$x[n] = A\cos(\omega_0 n + \phi)$$

where $\omega_0$ is the digital frequency (rad/sample). For a discrete-time sinusoid to be periodic, $\omega_0/(2\pi)$ must be rational.

### 4.6 Complex Exponential

The **complex exponential** is arguably the most important signal in all of signal processing:

**Continuous-time**:

$$x(t) = Ce^{st}, \quad s = \sigma + j\omega$$

When $\sigma = 0$: $x(t) = Ce^{j\omega_0 t} = C[\cos(\omega_0 t) + j\sin(\omega_0 t)]$ (Euler's formula)

**Discrete-time**:

$$x[n] = Ce^{j\omega_0 n}$$

Complex exponentials are **eigenfunctions** of LTI systems — when you input a complex exponential into an LTI system, the output is the same complex exponential scaled by a complex constant (the system's frequency response). This property is the foundation of frequency-domain analysis.

### 4.7 Sinc Function

$$\text{sinc}(t) = \frac{\sin(\pi t)}{\pi t}$$

with $\text{sinc}(0) = 1$ by L'Hopital's rule. The sinc function is the ideal interpolation kernel in sampling theory and the Fourier transform of the rectangular pulse.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Fundamental signals gallery ---
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

t = np.linspace(-3, 5, 1000)
n = np.arange(-5, 20)

# 1. Unit step (continuous-time)
axes[0, 0].plot(t, np.where(t >= 0, 1.0, 0.0), 'b-', linewidth=2)
axes[0, 0].set_title('Unit Step $u(t)$')
axes[0, 0].set_xlabel('t')
axes[0, 0].set_ylim([-0.2, 1.4])
axes[0, 0].grid(True, alpha=0.3)

# 2. Unit impulse (discrete-time, approximation)
axes[0, 1].stem(np.arange(-5, 6), np.where(np.arange(-5, 6) == 0, 1.0, 0.0),
                linefmt='r-', markerfmt='ro', basefmt='k-')
axes[0, 1].set_title('Unit Impulse $\\delta[n]$')
axes[0, 1].set_xlabel('n')
axes[0, 1].set_ylim([-0.2, 1.4])
axes[0, 1].grid(True, alpha=0.3)

# 3. Rectangular pulse
tau = 2.0
rect = np.where(np.abs(t) <= tau / 2, 1.0, 0.0)
axes[0, 2].plot(t, rect, 'g-', linewidth=2)
axes[0, 2].set_title(f'Rectangular Pulse rect$(t/{tau:.0f})$')
axes[0, 2].set_xlabel('t')
axes[0, 2].set_ylim([-0.2, 1.4])
axes[0, 2].grid(True, alpha=0.3)

# 4. Decaying exponential
axes[1, 0].plot(t, np.where(t >= 0, np.exp(-t), 0.0), 'b-', linewidth=2)
axes[1, 0].set_title('Decaying Exponential $e^{-t}u(t)$')
axes[1, 0].set_xlabel('t')
axes[1, 0].grid(True, alpha=0.3)

# 5. Growing exponential
axes[1, 1].plot(t[(t >= 0) & (t <= 3)],
                np.exp(0.5 * t[(t >= 0) & (t <= 3)]), 'r-', linewidth=2)
axes[1, 1].set_title('Growing Exponential $e^{0.5t}u(t)$')
axes[1, 1].set_xlabel('t')
axes[1, 1].grid(True, alpha=0.3)

# 6. Sinusoid
axes[1, 2].plot(t, np.cos(2 * np.pi * t), 'g-', linewidth=2)
axes[1, 2].set_title('Sinusoid $\\cos(2\\pi t)$')
axes[1, 2].set_xlabel('t')
axes[1, 2].grid(True, alpha=0.3)

# 7. Complex exponential (real and imaginary parts)
omega0 = 2 * np.pi * 0.5
axes[2, 0].plot(t, np.cos(omega0 * t), 'b-', linewidth=1.5, label='Real')
axes[2, 0].plot(t, np.sin(omega0 * t), 'r--', linewidth=1.5, label='Imag')
axes[2, 0].set_title('Complex Exponential $e^{j\\omega_0 t}$')
axes[2, 0].set_xlabel('t')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# 8. Damped sinusoid
sigma = -0.5
axes[2, 1].plot(t[t >= 0], np.exp(sigma * t[t >= 0]) * np.cos(omega0 * t[t >= 0]),
                'purple', linewidth=2)
axes[2, 1].plot(t[t >= 0], np.exp(sigma * t[t >= 0]), 'k--', linewidth=1, alpha=0.5)
axes[2, 1].plot(t[t >= 0], -np.exp(sigma * t[t >= 0]), 'k--', linewidth=1, alpha=0.5)
axes[2, 1].set_title('Damped Sinusoid $e^{\\sigma t}\\cos(\\omega_0 t)$')
axes[2, 1].set_xlabel('t')
axes[2, 1].grid(True, alpha=0.3)

# 9. Sinc function
t_sinc = np.linspace(-5, 5, 1000)
axes[2, 2].plot(t_sinc, np.sinc(t_sinc), 'm-', linewidth=2)
axes[2, 2].set_title('Sinc Function sinc$(t)$')
axes[2, 2].set_xlabel('t')
axes[2, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fundamental_signals.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 5. Energy and Power Signals

The concepts of energy and power provide a way to classify signals based on their "size."

### 5.1 Signal Energy

**Continuous-time**:

$$E_x = \int_{-\infty}^{\infty} |x(t)|^2 \, dt$$

**Discrete-time**:

$$E_x = \sum_{n=-\infty}^{\infty} |x[n]|^2$$

### 5.2 Signal Power

**Continuous-time** (time-average power):

$$P_x = \lim_{T \to \infty} \frac{1}{2T} \int_{-T}^{T} |x(t)|^2 \, dt$$

**Discrete-time**:

$$P_x = \lim_{N \to \infty} \frac{1}{2N+1} \sum_{n=-N}^{N} |x[n]|^2$$

### 5.3 Classification

| Class | Energy | Power | Examples |
|-------|--------|-------|----------|
| **Energy signal** | $0 < E_x < \infty$ | $P_x = 0$ | Pulses, decaying exponentials |
| **Power signal** | $E_x = \infty$ | $0 < P_x < \infty$ | Sinusoids, periodic signals, random processes |
| **Neither** | $E_x = \infty$ | $P_x = \infty$ | Growing exponential $e^t$ |

> A signal cannot be both an energy signal and a power signal.

### 5.4 Example Calculations

**Decaying exponential** $x(t) = e^{-at}u(t)$, $a > 0$:

$$E_x = \int_0^{\infty} e^{-2at} \, dt = \frac{1}{2a}$$

This is finite, so it is an **energy signal** with $P_x = 0$.

**Sinusoid** $x(t) = A\cos(\omega_0 t)$:

$$P_x = \lim_{T \to \infty} \frac{1}{2T} \int_{-T}^{T} A^2 \cos^2(\omega_0 t) \, dt = \frac{A^2}{2}$$

This has finite power but infinite energy, so it is a **power signal**.

```python
import numpy as np

# --- Energy and power computation ---

# Energy signal: exponential pulse
a = 1.0
t = np.linspace(0, 20, 100000)  # approximate [0, infinity)
dt = t[1] - t[0]
x_energy = np.exp(-a * t)
E_numerical = np.trapz(x_energy**2, t)
E_analytical = 1 / (2 * a)
print(f"Decaying exponential (a={a}):")
print(f"  Energy (numerical):  {E_numerical:.6f}")
print(f"  Energy (analytical): {E_analytical:.6f}")
print(f"  This is an energy signal (finite energy, zero power)")

print()

# Power signal: sinusoid
A = 3.0
f0 = 5.0
T_eval = 100  # average over many periods
t_sin = np.linspace(-T_eval, T_eval, 1000000)
dt_sin = t_sin[1] - t_sin[0]
x_power = A * np.cos(2 * np.pi * f0 * t_sin)
P_numerical = np.trapz(x_power**2, t_sin) / (2 * T_eval)
P_analytical = A**2 / 2
print(f"Sinusoid (A={A}, f0={f0} Hz):")
print(f"  Power (numerical):  {P_numerical:.6f}")
print(f"  Power (analytical): {P_analytical:.6f}")
print(f"  This is a power signal (infinite energy, finite power)")
```

---

## 6. Signal Operations

Signal operations are transformations applied to the independent variable or the amplitude of a signal.

### 6.1 Time Shifting

$$y(t) = x(t - t_0)$$

- $t_0 > 0$: **delay** (shift right)
- $t_0 < 0$: **advance** (shift left)

Discrete-time: $y[n] = x[n - n_0]$

### 6.2 Time Scaling

$$y(t) = x(at)$$

- $|a| > 1$: **compression** (signal is "sped up")
- $|a| < 1$: **expansion** (signal is "slowed down")
- $a < 0$: includes time reversal

### 6.3 Time Reversal (Reflection)

$$y(t) = x(-t)$$

The signal is "flipped" about $t = 0$. This is a special case of time scaling with $a = -1$.

### 6.4 Amplitude Scaling

$$y(t) = Ax(t)$$

Scales the amplitude by factor $A$. If $A < 0$, the signal is also inverted.

### 6.5 Signal Addition and Multiplication

**Addition**: $z(t) = x(t) + y(t)$ (sample-by-sample sum)

**Multiplication**: $z(t) = x(t) \cdot y(t)$ (sample-by-sample product; used in modulation/mixing)

### 6.6 Combining Operations

When multiple operations are combined, the order matters. For $y(t) = x(at - b)$:

**Method 1** (shift first, then scale):
1. Replace $t$ by $t - b/a$ to get $x(t - b/a)$ (shift by $b/a$)
2. Replace $t$ by $at$ to get $x(at - b)$ (scale by $a$)

**Method 2** (scale first, then shift):
1. Replace $t$ by $at$ to get $x(at)$ (scale by $a$)
2. Replace $t$ by $t - b/a$ — but be careful: we replace $t$ in the argument $at$, giving $a(t - b/a) = at - b$

> The safest approach is to always work from the inside out, determining where each feature of the original signal maps to in the new signal.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Signal operations demonstration ---
t = np.linspace(-4, 8, 1000)

# Original: triangular pulse
def tri_pulse(t, width=2.0):
    """Triangular pulse centered at t=0 with given width."""
    return np.maximum(0, 1 - np.abs(t) / width)

x = tri_pulse(t)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Original
axes[0, 0].plot(t, x, 'b-', linewidth=2)
axes[0, 0].set_title('Original: $x(t)$')
axes[0, 0].set_xlabel('t')
axes[0, 0].grid(True, alpha=0.3)

# Time shift (delay by 3)
axes[0, 1].plot(t, tri_pulse(t - 3), 'r-', linewidth=2)
axes[0, 1].plot(t, x, 'b--', linewidth=1, alpha=0.3, label='original')
axes[0, 1].set_title('Time Shift: $x(t - 3)$ (delay by 3)')
axes[0, 1].set_xlabel('t')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Time scaling (compression by 2)
axes[1, 0].plot(t, tri_pulse(2 * t), 'g-', linewidth=2)
axes[1, 0].plot(t, x, 'b--', linewidth=1, alpha=0.3, label='original')
axes[1, 0].set_title('Time Compression: $x(2t)$')
axes[1, 0].set_xlabel('t')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Time scaling (expansion by 0.5)
axes[1, 1].plot(t, tri_pulse(0.5 * t), 'm-', linewidth=2)
axes[1, 1].plot(t, x, 'b--', linewidth=1, alpha=0.3, label='original')
axes[1, 1].set_title('Time Expansion: $x(0.5t)$')
axes[1, 1].set_xlabel('t')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Time reversal
axes[2, 0].plot(t, tri_pulse(-t), 'orange', linewidth=2)
axes[2, 0].plot(t, x, 'b--', linewidth=1, alpha=0.3, label='original')
axes[2, 0].set_title('Time Reversal: $x(-t)$')
axes[2, 0].set_xlabel('t')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# Combined: x(2t - 3)
axes[2, 1].plot(t, tri_pulse(2 * t - 3), 'cyan', linewidth=2)
axes[2, 1].plot(t, x, 'b--', linewidth=1, alpha=0.3, label='original')
axes[2, 1].set_title('Combined: $x(2t - 3)$ (compress then shift)')
axes[2, 1].set_xlabel('t')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('signal_operations.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 7. What Is a System?

A **system** is a mathematical model of a process that transforms an input signal into an output signal.

$$x(t) \xrightarrow{\quad \mathcal{T}\{\cdot\} \quad} y(t) = \mathcal{T}\{x(t)\}$$

or in discrete-time:

$$x[n] \xrightarrow{\quad \mathcal{T}\{\cdot\} \quad} y[n] = \mathcal{T}\{x[n]\}$$

### Examples of Systems

| System | Input | Output | Domain |
|--------|-------|--------|--------|
| Amplifier | Voltage $x(t)$ | Amplified voltage $Ax(t)$ | Electronics |
| Moving average | $x[n]$ | $\frac{1}{M}\sum_{k=0}^{M-1}x[n-k]$ | Signal smoothing |
| Differentiator | $x(t)$ | $dx(t)/dt$ | Continuous-time |
| Accumulator | $x[n]$ | $\sum_{k=-\infty}^{n}x[k]$ | Discrete-time |
| Echo system | $x[n]$ | $x[n] + \alpha x[n-D]$ | Audio |
| Modulator | $x(t)$ | $x(t)\cos(\omega_c t)$ | Communications |

### System Interconnections

Systems can be combined in three basic configurations:

**Cascade (series)**: $y = \mathcal{T}_2\{\mathcal{T}_1\{x\}\}$

```
x → [T₁] → [T₂] → y
```

**Parallel**: $y = \mathcal{T}_1\{x\} + \mathcal{T}_2\{x\}$

```
     ┌─[T₁]─┐
x ──►│       ├──► (+) → y
     └─[T₂]─┘
```

**Feedback**: output is fed back to the input

```
x ──►(+)──► [T₁] ──► y
      ▲               │
      └───── [T₂] ◄───┘
```

---

## 8. System Properties

System properties determine what analysis tools we can use. The most important properties for signal processing are listed below.

### 8.1 Memory / Memoryless

A system is **memoryless** if the output at any time depends only on the input at that same time:

$$y(t) = f(x(t)) \quad \text{(no dependence on past or future)}$$

**Memoryless examples**: $y(t) = 3x(t)$, $y[n] = x^2[n]$

**With memory**: $y[n] = x[n] + x[n-1]$ (depends on past), $y(t) = \int_{-\infty}^{t} x(\tau) d\tau$

### 8.2 Linearity

A system is **linear** if it satisfies **superposition**: for any signals $x_1, x_2$ and scalars $a, b$:

$$\mathcal{T}\{ax_1 + bx_2\} = a\mathcal{T}\{x_1\} + b\mathcal{T}\{x_2\}$$

This combines two sub-properties:
- **Additivity**: $\mathcal{T}\{x_1 + x_2\} = \mathcal{T}\{x_1\} + \mathcal{T}\{x_2\}$
- **Homogeneity (scaling)**: $\mathcal{T}\{ax\} = a\mathcal{T}\{x\}$

> A consequence of linearity: $\mathcal{T}\{0\} = 0$. If a system produces nonzero output for zero input, it is nonlinear.

**Linear**: $y(t) = 3x(t) + 2\frac{dx}{dt}$, moving average

**Nonlinear**: $y(t) = x^2(t)$, $y[n] = \log(x[n])$, $y(t) = x(t) + 1$ (violates $\mathcal{T}\{0\} = 0$)

### 8.3 Time Invariance

A system is **time-invariant** (or shift-invariant) if a time shift in the input produces the same time shift in the output:

$$\text{If } \mathcal{T}\{x(t)\} = y(t), \text{ then } \mathcal{T}\{x(t - t_0)\} = y(t - t_0)$$

In other words, the system behaves the same regardless of when the input is applied.

**Time-invariant**: $y(t) = x(t-1)$, $y[n] = x[n] - x[n-1]$

**Time-varying**: $y(t) = x(2t)$ (time scaling), $y(t) = (\cos t) \cdot x(t)$ (time-varying coefficient)

### 8.4 Causality

A system is **causal** if the output at any time depends only on present and past inputs:

$$y(t_0) \text{ depends only on } \{x(t) : t \leq t_0\}$$

**Causal**: $y[n] = x[n] + x[n-1]$, $y(t) = \int_{-\infty}^{t} x(\tau)d\tau$

**Noncausal**: $y[n] = x[n+1]$ (depends on future), ideal lowpass filter

> All physically realizable real-time systems must be causal. However, noncausal systems are useful in offline (batch) processing where the entire signal is available.

### 8.5 Stability (Informal)

A system is **stable** if bounded inputs produce bounded outputs. We formalize this as **BIBO stability** in the next section.

**Stable**: $y(t) = e^{-t}x(t)$ for $t \geq 0$

**Unstable**: $y(t) = \int_{0}^{t} x(\tau)d\tau$ for a constant input (output grows without bound)

### 8.6 Invertibility

A system is **invertible** if distinct inputs produce distinct outputs, meaning there exists an inverse system $\mathcal{T}^{-1}$ such that:

$$\mathcal{T}^{-1}\{\mathcal{T}\{x\}\} = x$$

**Invertible**: $y(t) = 2x(t)$ (inverse: $x(t) = y(t)/2$)

**Not invertible**: $y(t) = x^2(t)$ (cannot recover sign of $x$)

```python
import numpy as np

# --- Testing system properties ---

# Test linearity of a system
def system_linear(x):
    """Linear system: y[n] = 2*x[n] + x[n-1]"""
    y = np.zeros_like(x)
    y[0] = 2 * x[0]
    for i in range(1, len(x)):
        y[i] = 2 * x[i] + x[i - 1]
    return y

def system_nonlinear(x):
    """Nonlinear system: y[n] = x[n]^2"""
    return x ** 2

def test_linearity(system, name):
    """Test linearity using superposition."""
    np.random.seed(42)
    x1 = np.random.randn(20)
    x2 = np.random.randn(20)
    a, b = 3.0, -2.0

    lhs = system(a * x1 + b * x2)           # T{a*x1 + b*x2}
    rhs = a * system(x1) + b * system(x2)   # a*T{x1} + b*T{x2}

    error = np.max(np.abs(lhs - rhs))
    is_linear = error < 1e-10
    print(f"{name}: max superposition error = {error:.2e} -> {'LINEAR' if is_linear else 'NONLINEAR'}")

test_linearity(system_linear, "y[n] = 2x[n] + x[n-1]")
test_linearity(system_nonlinear, "y[n] = x[n]^2")

print()

# Test time invariance
def test_time_invariance(system, name, delay=3):
    """Test time invariance by comparing shifted input vs shifted output."""
    np.random.seed(42)
    x = np.random.randn(30)

    # Method 1: shift input, then apply system
    x_delayed = np.zeros_like(x)
    x_delayed[delay:] = x[:-delay]
    y1 = system(x_delayed)

    # Method 2: apply system, then shift output
    y = system(x)
    y2 = np.zeros_like(y)
    y2[delay:] = y[:-delay]

    error = np.max(np.abs(y1 - y2))
    is_ti = error < 1e-10
    print(f"{name}: max TI error = {error:.2e} -> {'TIME-INVARIANT' if is_ti else 'TIME-VARYING'}")

def system_tv(x):
    """Time-varying system: y[n] = n * x[n]"""
    n = np.arange(len(x), dtype=float)
    return n * x

test_time_invariance(system_linear, "y[n] = 2x[n] + x[n-1]")
test_time_invariance(system_tv, "y[n] = n * x[n]")
```

---

## 9. BIBO Stability

### 9.1 Definition

A system is **Bounded-Input Bounded-Output (BIBO) stable** if every bounded input produces a bounded output:

$$|x(t)| \leq M_x < \infty \quad \Rightarrow \quad |y(t)| \leq M_y < \infty$$

for some finite $M_y$.

### 9.2 BIBO Stability for LTI Systems

For an LTI system with impulse response $h(t)$, BIBO stability is equivalent to **absolute integrability** of the impulse response:

**Continuous-time**:

$$\int_{-\infty}^{\infty} |h(t)| \, dt < \infty$$

**Discrete-time**:

$$\sum_{n=-\infty}^{\infty} |h[n]| < \infty$$

**Proof sketch** (discrete-time): If $|x[n]| \leq M_x$, then

$$|y[n]| = \left|\sum_k h[k] x[n-k]\right| \leq \sum_k |h[k]| \cdot |x[n-k]| \leq M_x \sum_k |h[k]|$$

So $|y[n]| \leq M_x \cdot \sum_k |h[k]|$. If the sum converges, the output is bounded.

### 9.3 Examples

**Stable**: $h[n] = (0.5)^n u[n]$

$$\sum_{n=0}^{\infty} |0.5^n| = \sum_{n=0}^{\infty} 0.5^n = \frac{1}{1 - 0.5} = 2 < \infty \quad \checkmark$$

**Unstable**: $h[n] = u[n]$ (unit step, i.e., an accumulator/integrator)

$$\sum_{n=0}^{\infty} |1| = \infty \quad \times$$

**Marginally stable**: $h[n] = \cos(\omega_0 n) u[n]$ (oscillates but does not decay)

$$\sum_{n=0}^{\infty} |\cos(\omega_0 n)| = \infty \quad \times \text{ (BIBO unstable)}$$

```python
import numpy as np

# --- BIBO stability check ---

def check_bibo_stability(h, name, max_terms=10000):
    """Check BIBO stability by computing sum of |h[n]|."""
    abs_sum = np.sum(np.abs(h[:max_terms]))
    # Check if the partial sum is still growing significantly
    abs_sum_half = np.sum(np.abs(h[:max_terms // 2]))
    converging = abs(abs_sum - abs_sum_half) / max(abs_sum, 1e-15) < 0.01
    print(f"{name}:")
    print(f"  Sum |h[n]| (N={max_terms}): {abs_sum:.4f}")
    print(f"  Convergent: {converging}")
    print(f"  BIBO Stable: {'Yes' if converging and abs_sum < 1e6 else 'No'}")
    print()

n = np.arange(10000)

# Stable: decaying exponential
h_stable = 0.5**n
check_bibo_stability(h_stable, "h[n] = 0.5^n * u[n]")

# Unstable: unit step (accumulator)
h_unstable = np.ones(10000)
check_bibo_stability(h_unstable, "h[n] = u[n] (accumulator)")

# Stable: finite impulse response
h_fir = np.array([0.25, 0.5, 0.25])
h_fir_padded = np.zeros(10000)
h_fir_padded[:3] = h_fir
check_bibo_stability(h_fir_padded, "h[n] = [0.25, 0.5, 0.25] (FIR)")
```

---

## 10. Python Examples

### 10.1 Generating and Analyzing Common Signals

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

# --- Comprehensive signal generation toolkit ---

class SignalGenerator:
    """Generate common signals for analysis."""

    def __init__(self, duration=1.0, fs=1000):
        self.fs = fs
        self.duration = duration
        self.t = np.arange(0, duration, 1 / fs)
        self.N = len(self.t)

    def sinusoid(self, freq, amp=1.0, phase=0.0):
        return amp * np.cos(2 * np.pi * freq * self.t + phase)

    def square_wave(self, freq, amp=1.0, duty=0.5):
        return amp * sig.square(2 * np.pi * freq * self.t, duty=duty)

    def sawtooth(self, freq, amp=1.0):
        return amp * sig.sawtooth(2 * np.pi * freq * self.t)

    def gaussian_pulse(self, center, width):
        return np.exp(-((self.t - center) ** 2) / (2 * width ** 2))

    def chirp(self, f0, f1, method='linear'):
        return sig.chirp(self.t, f0, self.duration, f1, method=method)

    def white_noise(self, power=1.0):
        return np.sqrt(power) * np.random.randn(self.N)

    def energy(self, x):
        """Compute signal energy."""
        return np.trapz(np.abs(x) ** 2, self.t)

    def power(self, x):
        """Compute signal average power."""
        return np.trapz(np.abs(x) ** 2, self.t) / self.duration


# Demonstrate
gen = SignalGenerator(duration=0.1, fs=8000)

signals = {
    '5 Hz Sinusoid': gen.sinusoid(5),
    '5 Hz Square': gen.square_wave(5),
    '5 Hz Sawtooth': gen.sawtooth(5),
    'Gaussian Pulse': gen.gaussian_pulse(0.05, 0.01),
    'Chirp (10-100 Hz)': gen.chirp(10, 100),
}

fig, axes = plt.subplots(len(signals), 1, figsize=(12, 2.5 * len(signals)))
for ax, (name, x) in zip(axes, signals.items()):
    ax.plot(gen.t * 1000, x, linewidth=1.5)
    E = gen.energy(x)
    P = gen.power(x)
    ax.set_title(f'{name}  |  Energy={E:.4f}, Power={P:.4f}')
    ax.set_xlabel('Time (ms)')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('signal_gallery.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 10.2 Discrete-Time Signal Periodicity Test

```python
import numpy as np
from fractions import Fraction

def check_dt_periodicity(omega0):
    """
    Check if discrete-time sinusoid cos(omega0 * n) is periodic.
    Periodic if and only if omega0 / (2*pi) is rational.
    """
    ratio = omega0 / (2 * np.pi)

    # Use Fraction for exact rational approximation
    frac = Fraction(ratio).limit_denominator(1000)

    # Check if the approximation is close enough to be exact
    if abs(float(frac) - ratio) < 1e-10:
        N = frac.denominator  # fundamental period
        print(f"omega0 = {omega0:.6f}")
        print(f"  omega0/(2pi) = {ratio:.6f} = {frac}")
        print(f"  PERIODIC with fundamental period N = {N}")
        return N
    else:
        print(f"omega0 = {omega0:.6f}")
        print(f"  omega0/(2pi) = {ratio:.6f} (irrational)")
        print(f"  NOT PERIODIC")
        return None

# Test cases
print("=== Discrete-Time Periodicity Test ===\n")
check_dt_periodicity(np.pi / 4)       # pi/4 -> 1/8 -> periodic, N=8
print()
check_dt_periodicity(2 * np.pi / 3)   # 2pi/3 -> 1/3 -> periodic, N=3
print()
check_dt_periodicity(1.0)             # 1/(2pi) is irrational -> not periodic
print()
check_dt_periodicity(np.pi)           # pi/(2pi) = 1/2 -> periodic, N=2
```

### 10.3 System Property Verification Suite

```python
import numpy as np

def verify_system_properties(system, name, N=50, delay=5):
    """Comprehensive system property verification."""
    print(f"=== System: {name} ===")
    np.random.seed(42)

    x1 = np.random.randn(N)
    x2 = np.random.randn(N)
    a, b = 2.5, -1.3

    # 1. Linearity
    lhs = system(a * x1 + b * x2)
    rhs = a * system(x1) + b * system(x2)
    lin_err = np.max(np.abs(lhs - rhs))
    print(f"  Linearity error:       {lin_err:.2e} -> {'LINEAR' if lin_err < 1e-10 else 'NONLINEAR'}")

    # 2. Time Invariance
    x_del = np.zeros(N)
    x_del[delay:] = x1[:N - delay]
    y_shifted_input = system(x_del)
    y_then_shift = np.zeros(N)
    y = system(x1)
    y_then_shift[delay:] = y[:N - delay]
    ti_err = np.max(np.abs(y_shifted_input - y_then_shift))
    print(f"  Time-invariance error: {ti_err:.2e} -> {'TIME-INVARIANT' if ti_err < 1e-10 else 'TIME-VARYING'}")

    # 3. Causality (check if output at n depends on future inputs)
    # Modify future of x1 and check if current output changes
    x_mod = x1.copy()
    x_mod[N // 2:] = np.random.randn(N - N // 2)  # change future
    y_orig = system(x1)
    y_mod = system(x_mod)
    causal_err = np.max(np.abs(y_orig[:N // 2] - y_mod[:N // 2]))
    print(f"  Causality error:       {causal_err:.2e} -> {'CAUSAL' if causal_err < 1e-10 else 'NONCAUSAL'}")

    # 4. Memory
    # Check if output depends on anything other than current input
    memory = False
    x_test = np.zeros(N)
    x_test[N // 2] = 1.0
    y_test = system(x_test)
    if np.sum(np.abs(y_test) > 1e-10) > 1:
        memory = True
    print(f"  Memory:                {'WITH MEMORY' if memory else 'MEMORYLESS'}")
    print()

# Test various systems
def sys_gain(x):
    return 3 * x

def sys_delay(x):
    y = np.zeros_like(x)
    y[1:] = x[:-1]
    return y

def sys_ma3(x):
    """3-point moving average"""
    y = np.zeros_like(x)
    for i in range(len(x)):
        total = x[i]
        count = 1
        if i >= 1:
            total += x[i - 1]
            count += 1
        if i >= 2:
            total += x[i - 2]
            count += 1
        y[i] = total / 3
    return y

def sys_square(x):
    return x ** 2

def sys_tv_gain(x):
    n = np.arange(len(x), dtype=float)
    return n * x

verify_system_properties(sys_gain, "y[n] = 3x[n] (gain)")
verify_system_properties(sys_delay, "y[n] = x[n-1] (unit delay)")
verify_system_properties(sys_ma3, "y[n] = (x[n]+x[n-1]+x[n-2])/3 (MA)")
verify_system_properties(sys_square, "y[n] = x[n]^2 (squarer)")
verify_system_properties(sys_tv_gain, "y[n] = n*x[n] (time-varying)")
```

---

## 11. Summary

### Key Concepts

| Concept | Definition | Why It Matters |
|---------|-----------|---------------|
| Signal | Function conveying information | Fundamental object of study |
| CT vs DT | $x(t)$ vs $x[n]$ | Determines which math tools apply |
| Energy signal | $0 < E < \infty$, $P = 0$ | Transient signals (pulses) |
| Power signal | $E = \infty$, $0 < P < \infty$ | Persistent signals (sinusoids) |
| System | Input-output mapping $\mathcal{T}$ | Processes signals |
| Linearity | Superposition holds | Enables decomposition methods |
| Time invariance | Behavior unchanged by time shifts | System parameters are constant |
| Causality | Output depends only on past/present | Physical realizability |
| BIBO stability | Bounded input $\Rightarrow$ bounded output | System does not "blow up" |
| LTI system | Linear + time-invariant | Foundation of signal processing |

### Signal Processing Hierarchy

```
                    Signals & Systems (this lesson)
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
         LTI Systems    Fourier     Z-Transform
        & Convolution   Analysis
                │           │           │
                └───────────┼───────────┘
                            ▼
                    Digital Filters
                    & Applications
```

The concept of LTI systems is absolutely central: linearity allows us to decompose inputs into elementary components (like complex exponentials), analyze each component separately, and add the results. Time invariance ensures the system responds the same way regardless of when the input arrives. Together, these properties enable the powerful transform-domain methods that form the backbone of signal processing.

---

## 12. Exercises

### Exercise 1: Signal Classification

Classify each signal as (a) continuous-time or discrete-time, (b) periodic or aperiodic, (c) energy or power signal, (d) causal or noncausal.

1. $x(t) = 3\cos(100\pi t + \pi/3)$
2. $x[n] = (-0.5)^n u[n]$
3. $x(t) = e^{-2|t|}$
4. $x[n] = \cos(0.3\pi n)$
5. $x(t) = u(t) - u(t-5)$
6. $x[n] = 2^n u[-n]$

### Exercise 2: Even-Odd Decomposition

Decompose the following signal into its even and odd components. Plot all three.

$$x(t) = \begin{cases} t + 1 & 0 \leq t \leq 1 \\ 2 & 1 < t \leq 2 \\ 0 & \text{otherwise} \end{cases}$$

### Exercise 3: Energy Computation

Compute the energy of the following signals analytically, then verify numerically with Python:

1. $x(t) = 2e^{-3t}u(t)$
2. $x[n] = (0.8)^n u[n]$
3. $x(t) = \text{rect}(t/4)$ (rectangular pulse of width 4)

### Exercise 4: System Properties

For each system, determine whether it is (i) linear, (ii) time-invariant, (iii) causal, (iv) stable, (v) memoryless. Show your reasoning.

1. $y(t) = x(t-2)$
2. $y[n] = nx[n]$
3. $y(t) = \cos(x(t))$
4. $y[n] = x[-n]$
5. $y(t) = x(t)\cos(2\pi f_0 t)$
6. $y[n] = \sum_{k=n-2}^{n+2} x[k]$ (5-point centered average)
7. $y(t) = x(t) + 3$

### Exercise 5: BIBO Stability

Determine whether the LTI systems with the following impulse responses are BIBO stable:

1. $h(t) = e^{-2t}u(t)$
2. $h[n] = u[n]$
3. $h[n] = (0.9)^{|n|}$
4. $h(t) = \frac{\sin(t)}{t}$
5. $h[n] = \delta[n] - 0.5\delta[n-1]$

### Exercise 6: Python Implementation

Write a Python function `signal_analyzer(x, fs)` that takes a discrete-time signal `x` and sampling rate `fs`, and returns a dictionary containing:
- Signal duration
- Maximum absolute value
- Energy
- Average power
- Whether the signal is approximately periodic (use autocorrelation to detect)
- Estimated fundamental frequency (if periodic)

Test it with a sinusoid, a chirp signal, and white noise.

### Exercise 7: Signal Operations Challenge

Given $x(t)$ as a triangular pulse with peak amplitude 1 at $t = 0$ and support $[-1, 1]$:

1. Sketch and write a Python function for $y_1(t) = x(2t - 3)$
2. Sketch and write a Python function for $y_2(t) = x(-t + 1) + x(t - 1)$
3. Compute the energy of $x(t)$, $y_1(t)$, and $y_2(t)$. Explain the relationship between the energies.

### Exercise 8: Complex Exponentials and Phasors

1. Express $x(t) = 3\cos(10t + \pi/4) + 4\sin(10t - \pi/6)$ as a single sinusoid $A\cos(10t + \phi)$. Find $A$ and $\phi$.
2. Plot the phasor diagram.
3. Express $x(t)$ using complex exponentials: $x(t) = \text{Re}\{Ce^{j10t}\}$. Find $C$.

---

## 13. References

1. Oppenheim, A. V. & Willsky, A. S. *Signals and Systems* (2nd ed.), Ch. 1. Prentice Hall, 1997.
2. Haykin, S. & Van Veen, B. *Signals and Systems* (2nd ed.), Ch. 1-2. Wiley, 2003.
3. Lathi, B. P. & Green, R. A. *Linear Systems and Signals* (3rd ed.), Ch. 1. Oxford University Press, 2018.

---

[Next: 02. LTI Systems and Convolution](./02_LTI_Systems_and_Convolution.md) | [Overview](./00_Overview.md)
