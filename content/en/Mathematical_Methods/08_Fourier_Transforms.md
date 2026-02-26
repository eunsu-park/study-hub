# 08. Fourier Transforms

## Learning Objectives

- Understand the transition **from Fourier series to Fourier transform** and derive the spectral representation for aperiodic functions
- Prove and apply **key properties of Fourier transforms** (linearity, shift, scaling, differentiation)
- Calculate and physically interpret core transform pairs such as **Gaussian, rectangular function, Dirac delta function**
- Understand the **convolution theorem** and apply it to practical applications such as signal filtering
- Understand the principles of **Discrete Fourier Transform (DFT) and FFT** algorithms and explain the meaning of the Nyquist theorem
- Apply Fourier transforms to physics applications such as **uncertainty principle, Fraunhofer diffraction, and spectral analysis**

---

## 1. From Fourier Series to Fourier Transform

### 1.1 Generalization from Periodic to Aperiodic Functions

The Fourier series of a function $f(x)$ with period $T$ can be written as:

$$f(x) = \sum_{n=-\infty}^{\infty} c_n \cdot \exp\left(i \cdot \frac{2\pi n x}{T}\right)$$

where the complex Fourier coefficients $c_n$ are:

$$c_n = \frac{1}{T} \int_{-T/2}^{T/2} f(x) \cdot \exp\left(-i \cdot \frac{2\pi n x}{T}\right) dx$$

Defining the fundamental frequency as $\omega_0 = 2\pi/T$ and the frequency spacing as $\Delta\omega = \omega_0 = 2\pi/T$:

$$f(x) = \sum_{n=-\infty}^{\infty} c_n \cdot \exp(i n \omega_0 x)$$

$$c_n = \frac{1}{T} \int_{-T/2}^{T/2} f(x) \cdot \exp(-i n \omega_0 x) \, dx$$

**Key idea**: As the period $T$ approaches infinity ($T \to \infty$), the discrete frequencies $n\omega_0$ become a continuous variable $\omega$, and the sum becomes an integral.

$$T \to \infty \quad \Rightarrow \quad \Delta\omega \to 0$$

$$n \omega_0 \to \omega \quad \text{(continuous variable)}$$

$$\sum \to \int$$

In this limit, defining $c_n \cdot T$ as a new function $F(\omega)$:

$$F(\omega) = \lim_{T \to \infty} c_n \cdot T = \int_{-\infty}^{\infty} f(x) \cdot e^{-i\omega x} \, dx$$

This is precisely the **Fourier transform**.

### 1.2 Definition of Continuous Fourier Transform

**Fourier Transform**:

$$F(\omega) = \int_{-\infty}^{\infty} f(x) \cdot e^{-i\omega x} \, dx$$

**Inverse Fourier Transform**:

$$f(x) = \frac{1}{2\pi} \int_{-\infty}^{\infty} F(\omega) \cdot e^{i\omega x} \, d\omega$$

> **Note on conventions**: The placement of the $2\pi$ factor varies across textbooks and disciplines. The Boas textbook follows the above convention, while physics often uses a symmetric convention ($1/\sqrt{2\pi}$ on both sides). Any convention works as long as the transform-inverse transform pair is consistent.

**Existence condition**: If $f(x)$ is absolutely integrable, the Fourier transform exists:

$$\int_{-\infty}^{\infty} |f(x)| \, dx < \infty$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft as sp_fft

# Example: observe the transition from discrete spectrum to continuous spectrum as period T increases

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
periods = [2, 5, 20]

for idx, T in enumerate(periods):
    # Gaussian function on a finite interval
    x = np.linspace(-T/2, T/2, 1000)
    sigma = 0.5
    f_x = np.exp(-x**2 / (2 * sigma**2))

    # Time domain
    axes[0, idx].plot(x, f_x, 'b-', linewidth=1.5)
    axes[0, idx].set_title(f'f(x), T = {T}')
    axes[0, idx].set_xlabel('x')
    axes[0, idx].set_xlim(-5, 5)
    axes[0, idx].grid(True, alpha=0.3)

    # Fourier coefficients (discrete spectrum)
    n_max = 30
    n_vals = np.arange(-n_max, n_max + 1)
    omega_0 = 2 * np.pi / T
    c_n = []
    for n in n_vals:
        integrand = f_x * np.exp(-1j * n * omega_0 * x)
        cn = np.trapz(integrand, x) / T
        c_n.append(np.abs(cn))

    omega_vals = n_vals * omega_0
    axes[1, idx].stem(omega_vals, c_n, linefmt='b-', markerfmt='bo', basefmt='k-')
    axes[1, idx].set_title(f'|c_n|, T = {T}')
    axes[1, idx].set_xlabel('omega')
    axes[1, idx].set_xlim(-15, 15)
    axes[1, idx].grid(True, alpha=0.3)

plt.suptitle('Increasing period T: discrete spectrum approaches continuous spectrum', fontsize=14)
plt.tight_layout()
plt.savefig('fourier_discrete_to_continuous.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 2. Properties of Fourier Transform

### 2.1 Linearity and Symmetry

**Linearity**: The Fourier transform is a linear operator:

$$\mathcal{F}[a f(x) + b g(x)] = a F(\omega) + b G(\omega)$$

where $a, b$ are constants, and $F(\omega) = \mathcal{F}[f]$, $G(\omega) = \mathcal{F}[g]$.

**Symmetry (Duality)**:

If the Fourier transform of $f(x)$ is $F(\omega)$, then:

$$\mathcal{F}[F(x)] = 2\pi f(-\omega)$$

That is, swapping the roles of time and frequency domains returns the original function (reflected).

**Symmetry for real functions**: If $f(x)$ is real:

$$F(-\omega) = F^*(\omega) \quad \text{(Hermitian symmetry)}$$

$$|F(-\omega)| = |F(\omega)| \quad \text{(magnitude spectrum is even)}$$

### 2.2 Time Shift and Frequency Shift

**Time Shift**: Shifting a function by $x_0$ only changes the phase in the frequency domain:

$$\mathcal{F}[f(x - x_0)] = e^{-i\omega x_0} \cdot F(\omega)$$

The magnitude spectrum $|F(\omega)|$ remains unchanged; only the phase changes by $\omega x_0$.

**Frequency Shift (Modulation)**: Shifting by $\omega_0$ in the frequency domain:

$$\mathcal{F}[f(x) \cdot e^{i\omega_0 x}] = F(\omega - \omega_0)$$

This is the principle of **modulation** in communications. Multiplying by a carrier frequency $\omega_0$ shifts the spectrum by $\omega_0$.

### 2.3 Scaling Theorem

When the time axis of a function is compressed/expanded by a factor $a$:

$$\mathcal{F}[f(ax)] = \frac{1}{|a|} F\left(\frac{\omega}{a}\right)$$

**Physical meaning**: If a signal becomes narrower in the time domain ($a > 1$), it becomes broader in the frequency domain. This is the mathematical basis of the **uncertainty principle**.

```python
import numpy as np
import matplotlib.pyplot as plt

# Scaling theorem visualization: narrow pulse ↔ wide spectrum

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
a_values = [0.5, 1.0, 2.0]

x = np.linspace(-5, 5, 1000)
dx = x[1] - x[0]
omega = np.linspace(-10, 10, 1000)

for idx, a in enumerate(a_values):
    # Gaussian: f(ax) = exp(-(ax)^2 / 2)
    f = np.exp(-(a * x)**2 / 2)

    # Analytical Fourier transform: (1/|a|) * sqrt(2*pi) * exp(-omega^2 / (2*a^2))
    F_analytical = (1 / abs(a)) * np.sqrt(2 * np.pi) * np.exp(-omega**2 / (2 * a**2))

    axes[0, idx].plot(x, f, 'b-', linewidth=2)
    axes[0, idx].set_title(f'f({a}x) = exp(-({a}x)²/2)')
    axes[0, idx].set_xlabel('x')
    axes[0, idx].set_ylim(-0.1, 1.5)
    axes[0, idx].grid(True, alpha=0.3)

    axes[1, idx].plot(omega, F_analytical, 'r-', linewidth=2)
    axes[1, idx].set_title(f'F(omega), a = {a}')
    axes[1, idx].set_xlabel('omega')
    axes[1, idx].set_ylim(-0.1, 6)
    axes[1, idx].grid(True, alpha=0.3)

plt.suptitle('Scaling theorem: narrow pulse ↔ wide spectrum', fontsize=14)
plt.tight_layout()
plt.savefig('scaling_theorem.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 2.4 Differentiation and Integration

**Differentiation theorem**: Differentiation in the time domain corresponds to multiplication by $i\omega$ in the frequency domain:

$$\mathcal{F}[f'(x)] = i\omega \cdot F(\omega)$$

$$\mathcal{F}[f^{(n)}(x)] = (i\omega)^n \cdot F(\omega)$$

This property allows transforming differential equations into algebraic equations.

**Example**: Heat equation

$$\frac{\partial u}{\partial t} = k \frac{\partial^2 u}{\partial x^2}$$

Applying Fourier transform in $x$ to both sides:

$$\frac{dU}{dt} = k (i\omega)^2 U = -k\omega^2 U$$

This is an ODE with $\omega$ as a parameter, with solution:

$$U(\omega, t) = U(\omega, 0) \cdot e^{-k\omega^2 t}$$

**Differentiation in frequency domain**: Differentiation in frequency corresponds to multiplying by $-ix$ in time domain:

$$\mathcal{F}[-ix \cdot f(x)] = F'(\omega)$$

$$\mathcal{F}[(-ix)^n \cdot f(x)] = F^{(n)}(\omega)$$

---

## 3. Important Transform Pairs

### 3.1 Gaussian Function

The Fourier transform of a Gaussian is again a Gaussian. This property makes the Gaussian special:

$$f(x) = e^{-ax^2} \quad (a > 0)$$

$$F(\omega) = \sqrt{\frac{\pi}{a}} \cdot e^{-\omega^2/(4a)}$$

**Proof sketch**: Completing the square in the exponential:

$$F(\omega) = \int_{-\infty}^{\infty} e^{-ax^2} \cdot e^{-i\omega x} \, dx$$

$$= \int_{-\infty}^{\infty} \exp\left[-a\left(x + \frac{i\omega}{2a}\right)^2 - \frac{\omega^2}{4a}\right] dx$$

$$= e^{-\omega^2/(4a)} \sqrt{\frac{\pi}{a}}$$

The last step uses the Gaussian integral: $\int e^{-au^2} \, du = \sqrt{\pi/a}$.

> **Key point**: The Gaussian is an **eigenfunction** of the Fourier transform — its form is preserved under transformation. The product of the width in time domain $\sigma_x = 1/\sqrt{2a}$ and the width in frequency domain $\sigma_\omega = \sqrt{2a}$ is always constant: $\sigma_x \cdot \sigma_\omega = 1$.

### 3.2 Rectangle Function (rect) and sinc Function

**Rectangle function**:

$$\text{rect}(x/a) = \begin{cases} 1, & |x| < a/2 \\ 0, & |x| > a/2 \end{cases}$$

**Fourier transform**:

$$\mathcal{F}[\text{rect}(x/a)] = a \cdot \text{sinc}\left(\frac{\omega a}{2\pi}\right) = a \cdot \frac{\sin(\omega a/2)}{\omega a/2}$$

where the sinc function is $\text{sinc}(u) = \sin(\pi u) / (\pi u)$.

```python
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
widths = [1.0, 2.0, 4.0]

for idx, a in enumerate(widths):
    # Rectangle function
    x = np.linspace(-5, 5, 1000)
    rect = np.where(np.abs(x) < a / 2, 1.0, 0.0)

    # Analytical Fourier transform: a * sin(omega*a/2) / (omega*a/2)
    omega = np.linspace(-20, 20, 1000)
    # Handle singularity at omega = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        F_omega = a * np.sinc(omega * a / (2 * np.pi))

    axes[0, idx].plot(x, rect, 'b-', linewidth=2)
    axes[0, idx].set_title(f'rect(x/{a}), width = {a}')
    axes[0, idx].set_xlabel('x')
    axes[0, idx].set_ylim(-0.2, 1.5)
    axes[0, idx].grid(True, alpha=0.3)

    axes[1, idx].plot(omega, F_omega, 'r-', linewidth=1.5)
    axes[1, idx].set_title(f'F(omega), a = {a}')
    axes[1, idx].set_xlabel('omega')
    axes[1, idx].grid(True, alpha=0.3)

plt.suptitle('Rectangle function ↔ sinc function: wider width gives narrower spectrum', fontsize=14)
plt.tight_layout()
plt.savefig('rect_sinc_pair.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Physical meaning**: A narrow slit (small rect width) creates a wide diffraction pattern (broad sinc), and a wide slit creates a narrow diffraction pattern. This is the essence of Fraunhofer diffraction.

### 3.3 Dirac Delta Function

The **Dirac delta function** $\delta(x)$ is a distribution satisfying:

$$\int_{-\infty}^{\infty} \delta(x) \cdot g(x) \, dx = g(0)$$

$$\delta(x) = 0 \quad (x \neq 0)$$

$$\int_{-\infty}^{\infty} \delta(x) \, dx = 1$$

**Fourier transform of Dirac delta**:

$$\mathcal{F}[\delta(x)] = \int_{-\infty}^{\infty} \delta(x) \cdot e^{-i\omega x} \, dx = 1$$

That is, the spectrum of a Dirac delta is uniformly spread across all frequencies — a **white spectrum**.

**Conversely**: The Fourier transform of the constant function 1 is $2\pi\delta(\omega)$:

$$\mathcal{F}[1] = 2\pi \delta(\omega)$$

> **Physical meaning**: An infinitely short impulse in the time domain contains all frequency components equally. Conversely, a pure tone lasting forever (constant) has only a single frequency.

**Shifted delta function**:

$$\mathcal{F}[\delta(x - x_0)] = e^{-i\omega x_0}$$

$$\mathcal{F}[e^{i\omega_0 x}] = 2\pi \delta(\omega - \omega_0)$$

### 3.4 Table of Major Transform Pairs

| $f(x)$ | $F(\omega)$ | Notes |
|------|----------|------|
| $e^{-a|x|}$ | $\frac{2a}{a^2 + \omega^2}$ | Lorentzian |
| $e^{-ax^2}$ | $\sqrt{\pi/a} \cdot e^{-\omega^2/(4a)}$ | Gaussian |
| $\text{rect}(x/a)$ | $a \cdot \text{sinc}(\omega a/(2\pi))$ | Rectangle ↔ sinc |
| $\delta(x)$ | $1$ | Impulse ↔ white |
| $1$ | $2\pi\delta(\omega)$ | Constant ↔ DC |
| $e^{i\omega_0 x}$ | $2\pi\delta(\omega - \omega_0)$ | Complex exponential |
| $\cos(\omega_0 x)$ | $\pi[\delta(\omega-\omega_0) + \delta(\omega+\omega_0)]$ | Cosine |
| $x^n e^{-a|x|}$ | $\frac{d^n}{d\omega^n}\left[\frac{2a}{a^2+\omega^2}\right]$ modified | Damped polynomial |
| $\frac{1}{x^2 + a^2}$ | $\frac{\pi}{a} e^{-a|\omega|}$ | Lorentz inverse |
| $e^{-ax} u(x)$ | $\frac{1}{a + i\omega}$ | One-sided exponential, $u(x)$=step function |

---

## 4. Convolution Theorem

### 4.1 Definition of Convolution

The **convolution** of two functions $f(x)$ and $g(x)$ is defined as:

$$(f * g)(x) = \int_{-\infty}^{\infty} f(t) \cdot g(x - t) \, dt$$

Geometrically, this operation reflects $g$, slides it over $f$, and calculates the overlapping area.

**Commutativity of convolution**: $f * g = g * f$

### 4.2 Time Domain Convolution ↔ Frequency Domain Multiplication

**Convolution Theorem**:

$$\mathcal{F}[f * g] = F(\omega) \cdot G(\omega)$$

That is, **convolution in the time domain corresponds to multiplication in the frequency domain**.

**The converse also holds**:

$$\mathcal{F}[f \cdot g] = \frac{1}{2\pi} (F * G)(\omega) \quad \text{(transform of product)}$$

Multiplication in the time domain corresponds to convolution in the frequency domain.

**Proof**:

$$\mathcal{F}[f * g] = \int_{-\infty}^{\infty} \left[\int_{-\infty}^{\infty} f(t) g(x-t) \, dt\right] e^{-i\omega x} \, dx$$

Exchanging the order of integration and substituting $u = x - t$:

$$= \int_{-\infty}^{\infty} f(t) e^{-i\omega t} \, dt \cdot \int_{-\infty}^{\infty} g(u) e^{-i\omega u} \, du = F(\omega) \cdot G(\omega)$$

### 4.3 Application: Filtering

The most important application of the convolution theorem is **signal filtering**.

Passing an input signal $x(t)$ through a filter $h(t)$:

$$y(t) = (x * h)(t) \quad \text{(time domain: convolution)}$$

$$Y(\omega) = X(\omega) \cdot H(\omega) \quad \text{(frequency domain: multiplication)}$$

$H(\omega)$ is called the **transfer function** or **frequency response**.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# Convolution theorem visualization: signal filtering

np.random.seed(42)
N = 1024
dt = 0.01
t = np.arange(N) * dt

# Original signal: clean sine wave + noise
freq_signal = 5.0  # 5 Hz signal
signal_clean = np.sin(2 * np.pi * freq_signal * t)
noise = 0.5 * np.random.randn(N)
signal_noisy = signal_clean + noise

# Gaussian low-pass filter
sigma_filter = 0.05
t_filter = np.arange(-50, 51) * dt
h = np.exp(-t_filter**2 / (2 * sigma_filter**2))
h /= h.sum()  # normalize

# Time domain convolution
signal_filtered = fftconvolve(signal_noisy, h, mode='same')

# Frequency domain verification
freqs = np.fft.fftfreq(N, dt)
X_noisy = np.fft.fft(signal_noisy)
H_freq = np.fft.fft(h, N)  # FFT of filter padded to same length
Y_freq = X_noisy * H_freq

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Time domain
axes[0, 0].plot(t, signal_noisy, 'gray', alpha=0.7, label='noisy signal')
axes[0, 0].plot(t, signal_clean, 'b-', linewidth=2, label='original signal')
axes[0, 0].set_title('Input signal (time domain)')
axes[0, 0].set_xlabel('t (s)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[1, 0].plot(t_filter, h, 'g-', linewidth=2)
axes[1, 0].set_title('Gaussian filter h(t)')
axes[1, 0].set_xlabel('t (s)')
axes[1, 0].grid(True, alpha=0.3)

axes[2, 0].plot(t, signal_filtered, 'r-', linewidth=2, label='filtered output')
axes[2, 0].plot(t, signal_clean, 'b--', linewidth=1, label='original signal')
axes[2, 0].set_title('Output signal y(t) = (x * h)(t)')
axes[2, 0].set_xlabel('t (s)')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# Frequency domain
mask = freqs >= 0
axes[0, 1].plot(freqs[mask], np.abs(X_noisy[mask]) / N, 'gray', alpha=0.7)
axes[0, 1].set_title('|X(omega)| input spectrum')
axes[0, 1].set_xlabel('Frequency (Hz)')
axes[0, 1].set_xlim(0, 50)
axes[0, 1].grid(True, alpha=0.3)

axes[1, 1].plot(freqs[mask], np.abs(H_freq[mask]), 'g-', linewidth=2)
axes[1, 1].set_title('|H(omega)| filter frequency response')
axes[1, 1].set_xlabel('Frequency (Hz)')
axes[1, 1].set_xlim(0, 50)
axes[1, 1].grid(True, alpha=0.3)

axes[2, 1].plot(freqs[mask], np.abs(Y_freq[mask]) / N, 'r-', linewidth=2)
axes[2, 1].set_title('|Y(omega)| = |X(omega)| * |H(omega)|')
axes[2, 1].set_xlabel('Frequency (Hz)')
axes[2, 1].set_xlim(0, 50)
axes[2, 1].grid(True, alpha=0.3)

plt.suptitle('Convolution theorem: time-domain convolution = frequency-domain multiplication', fontsize=14)
plt.tight_layout()
plt.savefig('convolution_theorem.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 5. Discrete Fourier Transform (DFT) and FFT

### 5.1 From Continuous to Discrete

Real measurement data is not continuous but **sampled** at regular intervals $\Delta t$:

$$x_n = f(n \Delta t), \quad n = 0, 1, 2, \ldots, N-1$$

With $N$ samples total, the entire observation time is $T = N \Delta t$.

Approximating the integral of the continuous Fourier transform with a discrete sum yields the DFT.

### 5.2 Definition of DFT

**Discrete Fourier Transform (DFT)**:

$$X_k = \sum_{n=0}^{N-1} x_n \cdot \exp\left(-i \frac{2\pi k n}{N}\right), \quad k = 0, 1, \ldots, N-1$$

**Inverse DFT**:

$$x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \cdot \exp\left(i \frac{2\pi k n}{N}\right), \quad n = 0, 1, \ldots, N-1$$

**Frequency resolution**: The physical frequency corresponding to the $k$-th frequency component is:

$$f_k = \frac{k}{N \Delta t} = \frac{k}{T}$$

The frequency resolution $\Delta f = 1/T$ is inversely proportional to the observation time.

### 5.3 Overview of FFT Algorithm

Direct computation of the DFT requires $O(N^2)$ operations. The **Fast Fourier Transform (FFT)** reduces this to $O(N \log N)$.

**Cooley-Tukey algorithm** (1965) key idea:

1. Split the DFT of $N$ points into DFTs of $N/2$ even and $N/2$ odd indices:

$$X_k = \sum_{\text{even } n} x_n W^{kn} + \sum_{\text{odd } n} x_n W^{kn} = E_k + W^k O_k$$

where $W = \exp(-i 2\pi / N)$ (twiddle factor)

2. Recursively repeating this process completes in $\log_2(N)$ stages.

```python
import numpy as np
import time
import matplotlib.pyplot as plt

# DFT vs FFT computation time comparison

def dft_naive(x):
    """Naive DFT implementation: O(N^2)"""
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

# Measure time for various sizes
sizes = [64, 128, 256, 512, 1024]
times_dft = []
times_fft = []

for N in sizes:
    x = np.random.randn(N)

    # DFT (naive implementation)
    t_start = time.perf_counter()
    X_dft = dft_naive(x)
    t_dft = time.perf_counter() - t_start
    times_dft.append(t_dft)

    # FFT (NumPy)
    t_start = time.perf_counter()
    X_fft = np.fft.fft(x)
    t_fft = time.perf_counter() - t_start
    times_fft.append(t_fft)

    # Verify results
    error = np.max(np.abs(X_dft - X_fft))
    print(f"N={N:5d}: DFT={t_dft:.4f}s, FFT={t_fft:.6f}s, "
          f"speedup={t_dft/t_fft:.0f}x, max_error={error:.2e}")

plt.figure(figsize=(10, 6))
plt.loglog(sizes, times_dft, 'ro-', label='DFT O(N²)', markersize=8)
plt.loglog(sizes, times_fft, 'bs-', label='FFT O(N log N)', markersize=8)
plt.xlabel('N (data size)')
plt.ylabel('Computation time (s)')
plt.title('DFT vs FFT computation time comparison')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('dft_vs_fft_timing.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.4 Nyquist Theorem and Aliasing

**Nyquist-Shannon Sampling Theorem**:

> When the maximum frequency in a signal is $f_{\max}$, the sampling frequency $f_s$ must be at least $2f_{\max}$ to perfectly reconstruct the original signal.

$$f_s \geq 2 f_{\max} \quad \text{(Nyquist condition)}$$

$$f_{\text{Nyquist}} = \frac{f_s}{2} \quad \text{(Nyquist frequency: maximum representable frequency)}$$

**Aliasing**: When the Nyquist condition is not satisfied, high-frequency components incorrectly fold into low frequencies.

```python
import numpy as np
import matplotlib.pyplot as plt

# Aliasing visualization

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original continuous signal: 10 Hz sine wave
f_signal = 10.0  # Hz
t_continuous = np.linspace(0, 1, 10000)
signal_continuous = np.sin(2 * np.pi * f_signal * t_continuous)

# Case 1: adequate sampling (f_s = 50 Hz > 2 * 10 Hz)
f_s_good = 50.0
t_good = np.arange(0, 1, 1/f_s_good)
signal_good = np.sin(2 * np.pi * f_signal * t_good)

# Case 2: insufficient sampling (f_s = 12 Hz < 2 * 10 Hz)
f_s_bad = 12.0
t_bad = np.arange(0, 1, 1/f_s_bad)
signal_bad = np.sin(2 * np.pi * f_signal * t_bad)

# Aliased frequency: |f_signal - f_s_bad| = |10 - 12| = 2 Hz
f_alias = abs(f_signal - f_s_bad)
signal_alias = np.sin(2 * np.pi * f_alias * t_continuous)

# Time domain: adequate sampling
axes[0, 0].plot(t_continuous, signal_continuous, 'b-', alpha=0.3, label=f'original {f_signal} Hz')
axes[0, 0].stem(t_good, signal_good, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0, 0].set_title(f'Adequate sampling: f_s = {f_s_good} Hz')
axes[0, 0].set_xlabel('t (s)')
axes[0, 0].set_xlim(0, 0.5)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Time domain: insufficient sampling
axes[0, 1].plot(t_continuous, signal_continuous, 'b-', alpha=0.3, label=f'original {f_signal} Hz')
axes[0, 1].plot(t_continuous, signal_alias, 'r--', alpha=0.5, label=f'alias {f_alias} Hz')
axes[0, 1].stem(t_bad, signal_bad, linefmt='r-', markerfmt='ro', basefmt='k-')
axes[0, 1].set_title(f'Insufficient sampling: f_s = {f_s_bad} Hz (aliasing occurs!)')
axes[0, 1].set_xlabel('t (s)')
axes[0, 1].set_xlim(0, 0.5)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Frequency domain: adequate sampling
N_good = len(signal_good)
freqs_good = np.fft.fftfreq(N_good, 1/f_s_good)
X_good = np.fft.fft(signal_good) / N_good
mask_good = freqs_good >= 0
axes[1, 0].stem(freqs_good[mask_good], 2 * np.abs(X_good[mask_good]),
                linefmt='b-', markerfmt='bo', basefmt='k-')
axes[1, 0].axvline(x=f_s_good/2, color='g', linestyle='--', label=f'f_Nyquist = {f_s_good/2} Hz')
axes[1, 0].set_title('Spectrum (adequate sampling)')
axes[1, 0].set_xlabel('Frequency (Hz)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Frequency domain: insufficient sampling
N_bad = len(signal_bad)
freqs_bad = np.fft.fftfreq(N_bad, 1/f_s_bad)
X_bad = np.fft.fft(signal_bad) / N_bad
mask_bad = freqs_bad >= 0
axes[1, 1].stem(freqs_bad[mask_bad], 2 * np.abs(X_bad[mask_bad]),
                linefmt='r-', markerfmt='ro', basefmt='k-')
axes[1, 1].axvline(x=f_s_bad/2, color='g', linestyle='--', label=f'f_Nyquist = {f_s_bad/2} Hz')
axes[1, 1].set_title('Spectrum (insufficient sampling → aliasing)')
axes[1, 1].set_xlabel('Frequency (Hz)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Nyquist theorem and aliasing', fontsize=14)
plt.tight_layout()
plt.savefig('nyquist_aliasing.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6. Physics Applications

### 6.1 Uncertainty Principle (Time-Frequency Relation)

**Mathematical uncertainty principle**: For any function $f(x)$ and its Fourier transform $F(\omega)$:

$$\Delta x \cdot \Delta\omega \geq \frac{1}{2}$$

where $\Delta x$, $\Delta\omega$ are standard deviations in time/frequency domains.

**Connection to quantum mechanics**: By the de Broglie relation $p = \hbar k$ ($k = 2\pi/\lambda$), this becomes the position-momentum uncertainty principle:

$$\Delta x \cdot \Delta p \geq \frac{\hbar}{2}$$

This is a fundamental principle of quantum mechanics, but its mathematical essence is a **property of the Fourier transform**.

```python
import numpy as np
import matplotlib.pyplot as plt

# Uncertainty principle visualization: Gaussian wave packets

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
sigmas = [0.5, 1.0, 2.0]  # width in position space

x = np.linspace(-10, 10, 2000)
k = np.linspace(-10, 10, 2000)

for idx, sigma_x in enumerate(sigmas):
    # Position-space wavefunction (Gaussian wave packet)
    k0 = 3.0  # central wavenumber
    psi_x = (1 / (2 * np.pi * sigma_x**2)**0.25) * \
            np.exp(-x**2 / (4 * sigma_x**2)) * np.exp(1j * k0 * x)

    # Momentum-space wavefunction (Fourier transform)
    sigma_k = 1 / (2 * sigma_x)  # width in momentum space
    psi_k = (2 * sigma_x**2 / np.pi)**0.25 * \
            np.exp(-(k - k0)**2 * sigma_x**2)

    # Uncertainty product
    product = sigma_x * sigma_k

    # Position space
    axes[0, idx].plot(x, np.abs(psi_x)**2, 'b-', linewidth=2, label=f'|psi(x)|^2')
    axes[0, idx].fill_between(x, np.abs(psi_x)**2, alpha=0.2, color='blue')
    axes[0, idx].axvspan(-sigma_x, sigma_x, alpha=0.1, color='red', label=f'sigma_x = {sigma_x}')
    axes[0, idx].set_title(f'Position space: sigma_x = {sigma_x}')
    axes[0, idx].set_xlabel('x')
    axes[0, idx].legend(fontsize=9)
    axes[0, idx].grid(True, alpha=0.3)

    # Momentum space
    axes[1, idx].plot(k, np.abs(psi_k)**2, 'r-', linewidth=2, label=f'|psi(k)|^2')
    axes[1, idx].fill_between(k, np.abs(psi_k)**2, alpha=0.2, color='red')
    axes[1, idx].axvspan(k0 - sigma_k, k0 + sigma_k, alpha=0.1, color='blue',
                          label=f'sigma_k = {sigma_k:.2f}')
    axes[1, idx].set_title(f'Momentum space: sigma_k = {sigma_k:.2f}\n'
                           f'sigma_x * sigma_k = {product:.2f} >= 0.5')
    axes[1, idx].set_xlabel('k')
    axes[1, idx].legend(fontsize=9)
    axes[1, idx].grid(True, alpha=0.3)

plt.suptitle('Heisenberg uncertainty principle: Delta_x * Delta_k >= 1/2', fontsize=14)
plt.tight_layout()
plt.savefig('uncertainty_principle.png', dpi=150, bbox_inches='tight')
plt.show()
```

> **Gaussian wave packets** are the only functions that satisfy the equality in the uncertainty principle (minimum uncertainty state).

### 6.2 Fraunhofer Diffraction

The **Fraunhofer diffraction** pattern of light passing through a single slit is the **Fourier transform** of the aperture function:

$$U(\theta) \sim \mathcal{F}[A(x)]$$

where $A(x)$ is the aperture function, and the relationship between observation angle $\theta$ and spatial frequency is:

$$\omega = \frac{2\pi}{\lambda} \sin\theta$$

**Single slit**: Since $A(x) = \text{rect}(x/a)$:

$$U(\theta) \sim \text{sinc}\left(\frac{\pi a \sin\theta}{\lambda}\right)$$

$$I(\theta) \sim \text{sinc}^2\left(\frac{\pi a \sin\theta}{\lambda}\right)$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Fraunhofer diffraction: single slit and double slit

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
wavelength = 500e-9  # 500 nm (green light)
k = 2 * np.pi / wavelength

theta = np.linspace(-0.05, 0.05, 2000)  # observation angle (radians)

# --- Single slit ---
slit_widths = [10e-6, 20e-6, 50e-6]  # 10, 20, 50 micrometers

for idx, a in enumerate(slit_widths):
    beta = np.pi * a * np.sin(theta) / wavelength
    # sinc² pattern (handle singularity at beta=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        intensity = np.where(beta == 0, 1.0, (np.sin(beta) / beta)**2)

    axes[0, idx].plot(np.degrees(theta), intensity, 'b-', linewidth=1.5)
    axes[0, idx].set_title(f'Single slit: a = {a*1e6:.0f} um')
    axes[0, idx].set_xlabel('theta (deg)')
    axes[0, idx].set_ylabel('I / I_0')
    axes[0, idx].grid(True, alpha=0.3)

# --- Double slit ---
a = 20e-6       # slit width
d_values = [50e-6, 100e-6, 200e-6]  # slit separation

for idx, d in enumerate(d_values):
    beta = np.pi * a * np.sin(theta) / wavelength
    delta = np.pi * d * np.sin(theta) / wavelength

    with np.errstate(divide='ignore', invalid='ignore'):
        envelope = np.where(beta == 0, 1.0, (np.sin(beta) / beta)**2)
    interference = np.cos(delta)**2
    intensity = envelope * interference

    axes[1, idx].plot(np.degrees(theta), intensity, 'r-', linewidth=1, label='total pattern')
    axes[1, idx].plot(np.degrees(theta), envelope, 'b--', linewidth=1, alpha=0.5, label='diffraction envelope')
    axes[1, idx].set_title(f'Double slit: a = {a*1e6:.0f} um, d = {d*1e6:.0f} um')
    axes[1, idx].set_xlabel('theta (deg)')
    axes[1, idx].set_ylabel('I / I_0')
    axes[1, idx].legend(fontsize=9)
    axes[1, idx].grid(True, alpha=0.3)

plt.suptitle('Fraunhofer diffraction = Fourier transform of the aperture function', fontsize=14)
plt.tight_layout()
plt.savefig('fraunhofer_diffraction.png', dpi=150, bbox_inches='tight')
plt.show()
```

For a **double slit**, the aperture function is the sum of two rectangles:

$$A(x) = \text{rect}\left(\frac{x - d/2}{a}\right) + \text{rect}\left(\frac{x + d/2}{a}\right)$$

By linearity and shift theorem of the Fourier transform:

$$I(\theta) \sim \text{sinc}^2(\beta) \cdot \cos^2(\delta)$$

$$= \text{(diffraction envelope)} \times \text{(interference fringes)}$$

### 6.3 Signal Processing and Spectral Analysis

The most widely used FFT application in physics experiments is **spectral analysis**.

```python
import numpy as np
import matplotlib.pyplot as plt

# Practical application: spectral analysis of a composite signal

# Sampling configuration
f_s = 1000  # sampling frequency (Hz)
T = 2.0     # total observation time (s)
N = int(f_s * T)
t = np.arange(N) / f_s

# Composite signal: multiple frequency components + noise
# A typical scenario in physics experiments
f1, A1 = 50, 1.0    # 50 Hz, amplitude 1.0 (primary signal)
f2, A2 = 120, 0.5   # 120 Hz, amplitude 0.5 (near 2nd harmonic)
f3, A3 = 300, 0.2   # 300 Hz, amplitude 0.2 (weak component)

signal = (A1 * np.sin(2 * np.pi * f1 * t) +
          A2 * np.sin(2 * np.pi * f2 * t) +
          A3 * np.sin(2 * np.pi * f3 * t))

# Add white noise
np.random.seed(42)
noise_level = 0.8
signal_noisy = signal + noise_level * np.random.randn(N)

# Compute FFT
X = np.fft.fft(signal_noisy)
freqs = np.fft.fftfreq(N, 1/f_s)

# Power Spectral Density
psd = (2.0 / N) * np.abs(X[:N//2])**2
freqs_pos = freqs[:N//2]

# Spectrum after applying windowing (Hanning window)
window = np.hanning(N)
signal_windowed = signal_noisy * window
X_windowed = np.fft.fft(signal_windowed)
# Window correction: normalize for energy conservation
window_correction = np.sum(window**2) / N
psd_windowed = (2.0 / (N * window_correction)) * np.abs(X_windowed[:N//2])**2

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Time domain signal
axes[0, 0].plot(t[:200], signal_noisy[:200], 'gray', alpha=0.7, label='noisy signal')
axes[0, 0].plot(t[:200], signal[:200], 'b-', linewidth=1.5, label='original signal')
axes[0, 0].set_title('Time domain signal (first 0.2 s)')
axes[0, 0].set_xlabel('t (s)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Amplitude spectrum
amplitude_spectrum = (2.0 / N) * np.abs(X[:N//2])
axes[0, 1].plot(freqs_pos, amplitude_spectrum, 'b-', linewidth=1)
axes[0, 1].set_title('Amplitude spectrum |X(f)|')
axes[0, 1].set_xlabel('Frequency (Hz)')
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].set_xlim(0, 400)
axes[0, 1].grid(True, alpha=0.3)
# Mark peak frequencies
for f_peak, A_peak, label in [(f1, A1, '50 Hz'), (f2, A2, '120 Hz'), (f3, A3, '300 Hz')]:
    axes[0, 1].annotate(label, xy=(f_peak, A_peak), fontsize=10,
                        arrowprops=dict(arrowstyle='->', color='red'),
                        xytext=(f_peak + 20, A_peak + 0.1))

# Power spectral density (linear scale)
axes[1, 0].plot(freqs_pos, psd, 'b-', linewidth=1, alpha=0.5, label='rectangular window')
axes[1, 0].plot(freqs_pos, psd_windowed, 'r-', linewidth=1.5, label='Hanning window')
axes[1, 0].set_title('Power Spectral Density (PSD)')
axes[1, 0].set_xlabel('Frequency (Hz)')
axes[1, 0].set_ylabel('Power')
axes[1, 0].set_xlim(0, 400)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Power spectral density (log scale, dB)
psd_db = 10 * np.log10(psd_windowed + 1e-20)  # avoid log(0)
axes[1, 1].plot(freqs_pos, psd_db, 'r-', linewidth=1.5)
axes[1, 1].set_title('Power Spectral Density (dB scale)')
axes[1, 1].set_xlabel('Frequency (Hz)')
axes[1, 1].set_ylabel('Power (dB)')
axes[1, 1].set_xlim(0, 400)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Spectral analysis using FFT', fontsize=14)
plt.tight_layout()
plt.savefig('spectral_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Practical tips for spectral analysis**:

1. **Windowing**: When applying FFT to finite data, apply window functions such as Hanning, Hamming, or Blackman to reduce **spectral leakage** caused by discontinuities at the boundaries.

2. **Zero-padding**: Adding zeros after the data increases the FFT size, providing an **interpolation** effect on the frequency axis (the frequency resolution itself does not change).

3. **dB scale**: Displaying the power spectrum in dB (decibels) allows viewing components with very different magnitudes together: $P_{\text{dB}} = 10 \log_{10}(P)$.

---

## Practice Problems

### Problem 1: Basic Transform Calculation

Find the Fourier transforms of the following functions analytically.

(a) $f(x) = e^{-3|x|}$

(b) $f(x) = x e^{-x^2}$

(c) $f(x) = \frac{1}{1 + x^2}$

> **Hint**: (a) Divide into positive and negative regions for integration. (b) Use the differentiation property of Gaussian transforms. (c) Use the residue theorem or refer to transform pair tables.

### Problem 2: Using Properties

Using the fact that the Fourier transform of $f(x) = e^{-x^2}$ is $F(\omega) = \sqrt{\pi} e^{-\omega^2/4}$:

(a) Find the Fourier transform of $g(x) = e^{-(x-3)^2}$. (time shift)

(b) Find the Fourier transform of $h(x) = e^{-4x^2}$. (scaling)

(c) Find the Fourier transform of $p(x) = x^2 e^{-x^2}$. (frequency differentiation)

### Problem 3: Convolution

(a) Calculate $\text{rect}(x) * \text{rect}(x)$ (direct integration or using transforms). Show that the result is a triangle function.

(b) Show that for the heat equation, when the initial temperature distribution is $\delta(x)$, the solution can be expressed as a convolution of Gaussians.

### Problem 4: FFT Practice

```python
# Complete the following code to find the frequency components in an unknown signal.

import numpy as np
import matplotlib.pyplot as plt

# Generate unknown signal
np.random.seed(2024)
f_s = 500  # sampling frequency
t = np.arange(0, 2, 1/f_s)

# 3 hidden frequency components + noise
signal = (0.7 * np.sin(2*np.pi*___*t) +
          1.2 * np.sin(2*np.pi*___*t) +
          0.4 * np.cos(2*np.pi*___*t) +
          0.5 * np.random.randn(len(t)))

# TODO: Perform FFT and find the hidden frequencies
# 1. Compute FFT with np.fft.fft()
# 2. Compute frequency axis with np.fft.fftfreq()
# 3. Plot the amplitude spectrum and identify the peak frequencies
```

### Problem 5: Physics Applications

(a) When light with wavelength $\lambda = 632.8$ nm (He-Ne laser) passes through a single slit of width $a = 50$ μm, find the angle $\theta$ of the first minimum.

(b) For a Gaussian wave packet $\psi(x) = A e^{-x^2/(4\sigma^2)} e^{ik_0 x}$:
- Find the momentum space wavefunction $\phi(k)$
- Calculate $\Delta x \cdot \Delta k$ and verify that it satisfies the uncertainty principle

(c) Explain why the ground state of a harmonic oscillator with mass $m$ has Gaussian form, connecting it to the uncertainty principle.

### Problem 6: Parseval's Theorem

**Parseval's theorem**: Prove that energy is conserved in time and frequency domains.

$$\int_{-\infty}^{\infty} |f(x)|^2 \, dx = \frac{1}{2\pi} \int_{-\infty}^{\infty} |F(\omega)|^2 \, d\omega$$

> **Hint**: Express the product of $f(x)$ and $f^*(x)$ as an inverse transform and exchange the order of integration.

---

## References

### Textbooks

- **Mary L. Boas**, *Mathematical Methods in the Physical Sciences*, 3rd Edition
  - Chapter 7: Fourier Series and Transforms
  - Chapter 15: Integral Transforms (advanced)
- **Arfken, Weber, Harris**, *Mathematical Methods for Physicists*, 7th Edition, Chapter 20
- **Riley, Hobson, Bence**, *Mathematical Methods for Physics and Engineering*, Chapter 13

### Advanced Topics

- **Bracewell, R.N.**, *The Fourier Transform and Its Applications*, McGraw-Hill
  - Classic reference known as the bible of Fourier transforms
- **Oppenheim, A.S. & Willsky, A.S.**, *Signals and Systems*, Prentice Hall
  - Fourier transforms from a signal processing perspective
- **Goodman, J.W.**, *Introduction to Fourier Optics*, W.H. Freeman
  - Applications of Fourier transforms in optics (diffraction, imaging)

### Online Resources

- **3Blue1Brown**: "But what is the Fourier Transform?" (visual intuition)
- **MIT OCW 18.03**: Differential Equations (Fourier series/transform lectures)
- **NumPy FFT Documentation**: https://numpy.org/doc/stable/reference/routines.fft.html
- **SciPy FFT Documentation**: https://docs.scipy.org/doc/scipy/reference/fft.html

---

## Next Lesson

**[07. Partial Differential Equations and Boundary Value Problems](07_PDEs_Boundary_Value.md)** uses Fourier transforms to solve key PDEs in physics such as the heat equation, wave equation, and Laplace equation. We cover solution techniques based on boundary conditions and eigenfunction expansions.
