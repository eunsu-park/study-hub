# LTI Systems and Convolution

**Previous**: [01. Signals and Systems](./01_Signals_and_Systems.md) | **Next**: [03. Fourier Series and Applications](./03_Fourier_Series_and_Applications.md)

---

In Lesson 01 we established the language of signals and the properties that characterize systems. Among all possible systems, one class stands above the rest in importance: **Linear Time-Invariant (LTI) systems**. The behavior of any LTI system is completely determined by a single function — its **impulse response** — and the output for any input can be computed through **convolution**. This lesson develops these ideas from first principles and shows why convolution is the central operation in signal processing.

**Difficulty**: ⭐⭐⭐

**Learning Objectives**:
- Explain why LTI systems are so important in signal processing
- Derive the convolution integral and convolution sum from the superposition principle
- Compute convolutions analytically and numerically
- Apply the properties of convolution (commutativity, associativity, distributivity)
- Relate step response to impulse response
- Analyze cascade, parallel, and feedback system interconnections
- Determine BIBO stability from the impulse response
- Compute the frequency response of an LTI system

---

## Table of Contents

1. [Why LTI Systems?](#1-why-lti-systems)
2. [Impulse Response](#2-impulse-response)
3. [The Convolution Sum (Discrete-Time)](#3-the-convolution-sum-discrete-time)
4. [The Convolution Integral (Continuous-Time)](#4-the-convolution-integral-continuous-time)
5. [Computing Convolutions](#5-computing-convolutions)
6. [Properties of Convolution](#6-properties-of-convolution)
7. [Step Response](#7-step-response)
8. [System Interconnections](#8-system-interconnections)
9. [LTI System Stability](#9-lti-system-stability)
10. [Frequency Response of LTI Systems](#10-frequency-response-of-lti-systems)
11. [Python Examples](#11-python-examples)
12. [Summary](#12-summary)
13. [Exercises](#13-exercises)
14. [References](#14-references)

---

## 1. Why LTI Systems?

### 1.1 The Power of Linearity and Time Invariance

An LTI system satisfies two properties simultaneously:

**Linearity** (superposition):

$$\mathcal{T}\{a x_1(t) + b x_2(t)\} = a \mathcal{T}\{x_1(t)\} + b \mathcal{T}\{x_2(t)\}$$

**Time Invariance** (shift invariance):

$$\text{If } x(t) \to y(t), \text{ then } x(t - t_0) \to y(t - t_0)$$

Together, these properties give us an extraordinary capability: if we know how the system responds to a single, simple input (the impulse), we can determine the response to **any** input. This is because:

1. **Linearity** lets us decompose any input into a weighted sum of elementary components
2. **Time invariance** ensures the system responds to each shifted component in the same way
3. **Linearity** again lets us add up all the individual responses

### 1.2 Representation of Signals Using Impulses

Any discrete-time signal can be written as a weighted sum of shifted impulses:

$$x[n] = \sum_{k=-\infty}^{\infty} x[k] \delta[n - k]$$

This is simply the **sifting property** restated. Each $x[k]\delta[n-k]$ is a scaled, shifted impulse.

For continuous-time:

$$x(t) = \int_{-\infty}^{\infty} x(\tau) \delta(t - \tau) \, d\tau$$

This decomposition is the key that unlocks convolution.

---

## 2. Impulse Response

### 2.1 Definition

The **impulse response** $h(t)$ (or $h[n]$) is the output of the system when the input is the unit impulse:

$$h(t) = \mathcal{T}\{\delta(t)\}, \qquad h[n] = \mathcal{T}\{\delta[n]\}$$

For an LTI system, the impulse response **completely characterizes** the system. No other information is needed.

### 2.2 Why the Impulse Response Characterizes the System

**Discrete-time derivation**: If the input $\delta[n]$ produces $h[n]$, then by time invariance, $\delta[n-k]$ produces $h[n-k]$. By homogeneity, $x[k]\delta[n-k]$ produces $x[k]h[n-k]$. By additivity:

$$y[n] = \mathcal{T}\left\{\sum_k x[k]\delta[n-k]\right\} = \sum_k x[k] h[n-k]$$

This is the **convolution sum**.

**Continuous-time derivation**: Similarly:

$$y(t) = \mathcal{T}\left\{\int x(\tau)\delta(t-\tau)d\tau\right\} = \int x(\tau) h(t - \tau) \, d\tau$$

This is the **convolution integral**.

### 2.3 Examples of Impulse Responses

| System | Impulse Response | Properties |
|--------|-----------------|------------|
| Ideal delay by $D$ | $h[n] = \delta[n - D]$ | FIR, causal, stable |
| Moving average (length $M$) | $h[n] = \frac{1}{M}\sum_{k=0}^{M-1}\delta[n-k]$ | FIR, causal, stable |
| First-order recursive | $h[n] = a^n u[n]$ | IIR, causal, stable if $\|a\| < 1$ |
| Ideal lowpass filter | $h(t) = \text{sinc}(2Bt)$ | Noncausal, not realizable |
| RC lowpass circuit | $h(t) = \frac{1}{RC}e^{-t/RC}u(t)$ | Causal, stable |

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Impulse response examples ---
n = np.arange(-5, 30)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 1. Ideal delay (D=3)
D = 3
h_delay = np.where(n == D, 1.0, 0.0)
axes[0, 0].stem(n, h_delay, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0, 0].set_title(f'Ideal Delay: $h[n] = \\delta[n - {D}]$')
axes[0, 0].set_xlabel('n')
axes[0, 0].set_ylabel('h[n]')
axes[0, 0].grid(True, alpha=0.3)

# 2. Moving average (M=5)
M = 5
h_ma = np.where((n >= 0) & (n < M), 1.0 / M, 0.0)
axes[0, 1].stem(n, h_ma, linefmt='r-', markerfmt='ro', basefmt='k-')
axes[0, 1].set_title(f'Moving Average (M={M}): $h[n] = \\frac{{1}}{{{M}}}$, $0 \\leq n < {M}$')
axes[0, 1].set_xlabel('n')
axes[0, 1].set_ylabel('h[n]')
axes[0, 1].grid(True, alpha=0.3)

# 3. First-order recursive (a=0.8)
a = 0.8
h_recursive = np.where(n >= 0, a**n, 0.0)
axes[1, 0].stem(n, h_recursive, linefmt='g-', markerfmt='go', basefmt='k-')
axes[1, 0].set_title(f'First-Order Recursive: $h[n] = ({a})^n u[n]$')
axes[1, 0].set_xlabel('n')
axes[1, 0].set_ylabel('h[n]')
axes[1, 0].grid(True, alpha=0.3)

# 4. RC lowpass (continuous-time, simulated)
t = np.linspace(-1, 8, 1000)
RC = 1.0
h_rc = np.where(t >= 0, (1 / RC) * np.exp(-t / RC), 0.0)
axes[1, 1].plot(t, h_rc, 'm-', linewidth=2)
axes[1, 1].set_title(f'RC Lowpass: $h(t) = \\frac{{1}}{{RC}}e^{{-t/RC}}u(t)$, RC={RC}')
axes[1, 1].set_xlabel('t')
axes[1, 1].set_ylabel('h(t)')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].fill_between(t[t >= 0], h_rc[t >= 0], alpha=0.2, color='m')

plt.tight_layout()
plt.savefig('impulse_responses.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 3. The Convolution Sum (Discrete-Time)

### 3.1 Definition

The **convolution sum** computes the output $y[n]$ of a discrete-time LTI system with impulse response $h[n]$ and input $x[n]$:

$$y[n] = x[n] * h[n] = \sum_{k=-\infty}^{\infty} x[k] \, h[n - k]$$

The asterisk $*$ denotes convolution (not multiplication).

### 3.2 Interpretation

For each output sample $y[n]$:

1. **Flip** $h[k]$ about $k = 0$ to get $h[-k]$
2. **Shift** by $n$ to get $h[n - k]$
3. **Multiply** $x[k]$ and $h[n - k]$ element-wise
4. **Sum** over all $k$

This "flip-shift-multiply-sum" procedure is the mechanical recipe for computing convolution.

### 3.3 Graphical Convolution

For hand computation, graphical convolution is invaluable. The idea is to fix $n$ and visualize the overlap between $x[k]$ and $h[n-k]$ as $k$ varies.

**Example**: Convolve $x[n] = \{1, 2, 3\}$ (for $n = 0, 1, 2$) with $h[n] = \{1, 1, 1, 1\}$ (for $n = 0, 1, 2, 3$):

The output length is $\text{len}(x) + \text{len}(h) - 1 = 3 + 4 - 1 = 6$.

| $n$ | Overlapping products | $y[n]$ |
|-----|---------------------|--------|
| 0 | $1 \cdot 1$ | 1 |
| 1 | $1 \cdot 2 + 1 \cdot 1$ | 3 |
| 2 | $1 \cdot 3 + 1 \cdot 2 + 1 \cdot 1$ | 6 |
| 3 | $1 \cdot 3 + 1 \cdot 2 + 1 \cdot 1$ | 6 |
| 4 | $1 \cdot 3 + 1 \cdot 2$ | 5 |
| 5 | $1 \cdot 3$ | 3 |

So $y[n] = \{1, 3, 6, 6, 5, 3\}$.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Graphical convolution step-by-step ---
x = np.array([1, 2, 3])
h = np.array([1, 1, 1, 1])

# Full convolution
y = np.convolve(x, h)
print("x =", x)
print("h =", h)
print("y = x * h =", y)

# Visualize step-by-step
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

n_values = [0, 1, 2, 3, 4, 5]
k = np.arange(-2, 8)

for idx, (ax, n_val) in enumerate(zip(axes.flat, n_values)):
    # x[k]
    x_full = np.zeros_like(k, dtype=float)
    for i, ki in enumerate(k):
        if 0 <= ki < len(x):
            x_full[i] = x[ki]

    # h[n-k] (flipped and shifted)
    h_shifted = np.zeros_like(k, dtype=float)
    for i, ki in enumerate(k):
        idx_h = n_val - ki
        if 0 <= idx_h < len(h):
            h_shifted[i] = h[idx_h]

    # Product
    product = x_full * h_shifted

    ax.stem(k, x_full, linefmt='b-', markerfmt='bo', basefmt='k-',
            label='$x[k]$')
    ax.stem(k + 0.15, h_shifted, linefmt='r-', markerfmt='rs', basefmt='k-',
            label='$h[n-k]$')

    # Highlight overlap region
    overlap_mask = product != 0
    if np.any(overlap_mask):
        for ki, pi in zip(k[overlap_mask], product[overlap_mask]):
            ax.annotate(f'{pi:.0f}', (ki + 0.07, max(x_full[k == ki][0],
                        h_shifted[k == ki][0]) + 0.15),
                        ha='center', fontsize=9, color='green', fontweight='bold')

    ax.set_title(f'$n = {n_val}$: $y[{n_val}] = {y[n_val]:.0f}$')
    ax.set_xlabel('k')
    ax.legend(fontsize=8)
    ax.set_ylim([-0.5, 4])
    ax.grid(True, alpha=0.3)

plt.suptitle('Graphical Convolution: Flip-Shift-Multiply-Sum', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('graphical_convolution.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 3.4 Special Convolution Results

**Convolution with impulse** (identity):

$$x[n] * \delta[n] = x[n]$$

**Convolution with shifted impulse** (delay):

$$x[n] * \delta[n - n_0] = x[n - n_0]$$

**Convolution with unit step**:

$$x[n] * u[n] = \sum_{k=-\infty}^{n} x[k] \quad \text{(running sum / accumulator)}$$

---

## 4. The Convolution Integral (Continuous-Time)

### 4.1 Definition

The **convolution integral** is the continuous-time counterpart:

$$y(t) = x(t) * h(t) = \int_{-\infty}^{\infty} x(\tau) \, h(t - \tau) \, d\tau$$

### 4.2 Interpretation

The procedure mirrors the discrete case:

1. **Flip**: $h(\tau) \to h(-\tau)$
2. **Shift**: $h(-\tau) \to h(t - \tau)$
3. **Multiply**: $x(\tau) \cdot h(t - \tau)$
4. **Integrate**: $\int_{-\infty}^{\infty} (\cdot) \, d\tau$

### 4.3 Example: Exponential Convolved with Step

Compute $y(t) = e^{-at}u(t) * u(t)$ for $a > 0$.

$$y(t) = \int_{-\infty}^{\infty} e^{-a\tau}u(\tau) \cdot u(t - \tau) \, d\tau$$

The integrand is nonzero only when $\tau \geq 0$ (from $u(\tau)$) and $\tau \leq t$ (from $u(t-\tau)$), so for $t \geq 0$:

$$y(t) = \int_0^t e^{-a\tau} \, d\tau = \frac{1}{a}(1 - e^{-at}), \quad t \geq 0$$

For $t < 0$: $y(t) = 0$.

Therefore: $y(t) = \frac{1}{a}(1 - e^{-at})u(t)$

### 4.4 Example: Rectangular Pulse Self-Convolution

Compute $y(t) = \text{rect}(t) * \text{rect}(t)$ where $\text{rect}(t)$ is 1 for $|t| \leq 1/2$.

By the flip-shift-integrate procedure:

$$y(t) = \int_{-\infty}^{\infty} \text{rect}(\tau) \cdot \text{rect}(t - \tau) \, d\tau$$

The overlap between two rectangles of width 1 centered at 0 and $t$ gives a **triangular pulse**:

$$y(t) = \text{tri}(t) = \begin{cases} 1 - |t| & |t| \leq 1 \\ 0 & |t| > 1 \end{cases}$$

> This is a fundamental result: convolving a rectangle with itself produces a triangle. Convolving again produces a smoother shape, and in the limit, the result approaches a Gaussian (by the Central Limit Theorem).

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Continuous-time convolution examples ---

# Example 1: exp(-at)u(t) * u(t)
a = 2.0
t = np.linspace(-1, 5, 1000)
dt = t[1] - t[0]

h = np.where(t >= 0, np.exp(-a * t), 0.0)
x = np.where(t >= 0, 1.0, 0.0)

# Numerical convolution
y_numerical = np.convolve(h, x, mode='full') * dt
t_conv = np.arange(len(y_numerical)) * dt + 2 * t[0]

# Analytical result
y_analytical = np.where(t >= 0, (1/a) * (1 - np.exp(-a * t)), 0.0)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0, 0].plot(t, h, 'b-', linewidth=2)
axes[0, 0].set_title('$h(t) = e^{-2t}u(t)$')
axes[0, 0].set_xlabel('t')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t, x, 'r-', linewidth=2)
axes[0, 1].set_title('$x(t) = u(t)$')
axes[0, 1].set_xlabel('t')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(t, y_analytical, 'g-', linewidth=2, label='Analytical')
axes[1, 0].plot(t_conv[:len(t)], y_numerical[:len(t)], 'k--', linewidth=1.5,
                label='Numerical', alpha=0.7)
axes[1, 0].set_title('$y(t) = \\frac{1}{a}(1 - e^{-at})u(t)$')
axes[1, 0].set_xlabel('t')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Example 2: rect * rect = triangle
t2 = np.linspace(-2, 2, 1000)
dt2 = t2[1] - t2[0]
rect = np.where(np.abs(t2) <= 0.5, 1.0, 0.0)
tri_conv = np.convolve(rect, rect, mode='full') * dt2
t2_conv = np.arange(len(tri_conv)) * dt2 + 2 * t2[0]

tri_analytical = np.maximum(0, 1 - np.abs(t2))

axes[1, 1].plot(t2, tri_analytical, 'purple', linewidth=2, label='tri(t) analytical')
axes[1, 1].plot(t2_conv[:len(t2)] + t2[0], tri_conv[:len(t2)], 'k--',
                linewidth=1.5, label='rect*rect numerical', alpha=0.7)
axes[1, 1].set_title('rect$(t)$ * rect$(t)$ = tri$(t)$')
axes[1, 1].set_xlabel('t')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('convolution_examples.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 4.5 Convolution with Causal Signals

When both $x(t)$ and $h(t)$ are causal (zero for $t < 0$), the integral limits simplify:

$$y(t) = \int_0^t x(\tau) h(t - \tau) \, d\tau, \quad t \geq 0$$

This is because $x(\tau) = 0$ for $\tau < 0$ and $h(t - \tau) = 0$ for $\tau > t$.

---

## 5. Computing Convolutions

### 5.1 Analytical Methods

**Method 1: Direct integration** — Identify the limits of integration where the integrand is nonzero, then integrate.

**Method 2: Laplace/Z-transform** — In the transform domain, convolution becomes multiplication:

$$\mathcal{L}\{x * h\} = X(s) \cdot H(s)$$

$$\mathcal{Z}\{x * h\} = X(z) \cdot H(z)$$

This is often much simpler than direct computation. We will explore this in detail in later lessons.

### 5.2 Numerical Methods

**Direct implementation** (naive, $O(N^2)$):

```python
def convolve_direct(x, h):
    """Direct convolution: O(N*M) where N=len(x), M=len(h)."""
    N = len(x)
    M = len(h)
    y = np.zeros(N + M - 1)
    for n in range(N + M - 1):
        for k in range(M):
            if 0 <= n - k < N:
                y[n] += h[k] * x[n - k]
    return y
```

**FFT-based convolution** (fast, $O(N \log N)$):

```python
def convolve_fft(x, h):
    """FFT-based convolution: O(N log N)."""
    N = len(x) + len(h) - 1
    # Pad to next power of 2 for FFT efficiency
    N_fft = 2 ** int(np.ceil(np.log2(N)))
    X = np.fft.fft(x, N_fft)
    H = np.fft.fft(h, N_fft)
    y = np.real(np.fft.ifft(X * H))
    return y[:N]
```

The FFT approach exploits the **convolution theorem**: convolution in time equals multiplication in frequency. For large signals, this is dramatically faster.

### 5.3 NumPy and SciPy Functions

```python
import numpy as np
from scipy import signal

x = np.array([1, 2, 3, 4, 5])
h = np.array([0.2, 0.3, 0.5])

# Full convolution (output length = len(x) + len(h) - 1)
y_full = np.convolve(x, h, mode='full')
print("Full:", y_full)

# Same-size output (centered, length = max(len(x), len(h)))
y_same = np.convolve(x, h, mode='same')
print("Same:", y_same)

# Valid (only where signals fully overlap, length = max(N,M) - min(N,M) + 1)
y_valid = np.convolve(x, h, mode='valid')
print("Valid:", y_valid)

# scipy.signal.fftconvolve for large arrays (FFT-based)
y_fft = signal.fftconvolve(x, h, mode='full')
print("FFT:", y_fft)
```

### 5.4 Convolution Modes Comparison

| Mode | Output Length | Description |
|------|-------------|-------------|
| `'full'` | $N + M - 1$ | Complete convolution result |
| `'same'` | $\max(N, M)$ | Output same size as largest input |
| `'valid'` | $|N - M| + 1$ | Only where inputs fully overlap |

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Convolution modes visualization ---
x = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
h = np.array([0.2, 0.6, 0.2])

y_full = np.convolve(x, h, mode='full')
y_same = np.convolve(x, h, mode='same')
y_valid = np.convolve(x, h, mode='valid')

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

axes[0].stem(range(len(x)), x, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0].set_title(f'Input x[n] (length {len(x)})')
axes[0].grid(True, alpha=0.3)

axes[1].stem(range(len(y_full)), y_full, linefmt='r-', markerfmt='ro', basefmt='k-')
axes[1].set_title(f"mode='full' (length {len(y_full)} = {len(x)} + {len(h)} - 1)")
axes[1].grid(True, alpha=0.3)

axes[2].stem(range(len(y_same)), y_same, linefmt='g-', markerfmt='go', basefmt='k-')
axes[2].set_title(f"mode='same' (length {len(y_same)})")
axes[2].grid(True, alpha=0.3)

axes[3].stem(range(len(y_valid)), y_valid, linefmt='m-', markerfmt='mo', basefmt='k-')
axes[3].set_title(f"mode='valid' (length {len(y_valid)})")
axes[3].grid(True, alpha=0.3)

for ax in axes:
    ax.set_xlabel('n')

plt.tight_layout()
plt.savefig('convolution_modes.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6. Properties of Convolution

Convolution satisfies several algebraic properties that are both theoretically important and practically useful.

### 6.1 Commutativity

$$x[n] * h[n] = h[n] * x[n]$$

or equivalently:

$$\sum_k x[k]h[n-k] = \sum_k h[k]x[n-k]$$

**Proof**: Substitute $m = n - k$ in the first sum.

**Implication**: The roles of input and system can be interchanged. When computing convolution, we can choose to "flip" whichever signal is more convenient.

### 6.2 Associativity

$$(x * h_1) * h_2 = x * (h_1 * h_2)$$

**Implication**: A cascade of two LTI systems with impulse responses $h_1$ and $h_2$ is equivalent to a single LTI system with impulse response $h_1 * h_2$, regardless of the cascade order.

### 6.3 Distributivity over Addition

$$x * (h_1 + h_2) = x * h_1 + x * h_2$$

**Implication**: A parallel combination of two LTI systems is equivalent to a single system whose impulse response is the sum of the individual impulse responses.

### 6.4 Identity Element

$$x * \delta = x$$

The impulse $\delta$ is the identity for convolution, just as 1 is the identity for multiplication.

### 6.5 Shift Property

$$x[n] * \delta[n - n_0] = x[n - n_0]$$

Convolution with a shifted impulse delays the signal.

### 6.6 Width Property

If $x[n]$ has support $[N_1, N_2]$ and $h[n]$ has support $[M_1, M_2]$, then $y[n] = x[n] * h[n]$ has support $[N_1 + M_1, N_2 + M_2]$.

The output **duration** (width) equals the sum of input durations (minus 1 for discrete-time).

### 6.7 Convolution with Scaled Impulse Pair (Echo System)

A common application: the system $h[n] = \delta[n] + \alpha \delta[n - D]$ (echo with delay $D$ and attenuation $\alpha$) produces:

$$y[n] = x[n] + \alpha x[n - D]$$

```python
import numpy as np

# --- Verify convolution properties ---
np.random.seed(42)
x = np.random.randn(20)
h1 = np.random.randn(10)
h2 = np.random.randn(8)

# Commutativity
y1 = np.convolve(x, h1)
y2 = np.convolve(h1, x)
print(f"Commutativity error: {np.max(np.abs(y1 - y2)):.2e}")

# Associativity
y_assoc1 = np.convolve(np.convolve(x, h1), h2)
y_assoc2 = np.convolve(x, np.convolve(h1, h2))
print(f"Associativity error: {np.max(np.abs(y_assoc1 - y_assoc2)):.2e}")

# Distributivity
y_dist1 = np.convolve(x, h1 + h2[:len(h1)])  # need same length for addition
h_padded = np.zeros(max(len(h1), len(h2)))
h_padded[:len(h1)] += h1
h_padded2 = np.zeros(max(len(h1), len(h2)))
h_padded2[:len(h2)] += h2
y_dist_lhs = np.convolve(x, h_padded + h_padded2)
y_dist_rhs = np.zeros(len(y_dist_lhs))
y_r1 = np.convolve(x, h_padded)
y_r2 = np.convolve(x, h_padded2)
max_len = max(len(y_r1), len(y_r2))
y_dist_rhs_a = np.zeros(max_len)
y_dist_rhs_a[:len(y_r1)] += y_r1
y_dist_rhs_a[:len(y_r2)] += y_r2
print(f"Distributivity error: {np.max(np.abs(y_dist_lhs - y_dist_rhs_a[:len(y_dist_lhs)])):.2e}")

# Identity
delta = np.zeros(1)
delta[0] = 1.0
y_id = np.convolve(x, delta)
print(f"Identity error: {np.max(np.abs(x - y_id[:len(x)])):.2e}")
```

---

## 7. Step Response

### 7.1 Definition

The **step response** $s(t)$ (or $s[n]$) is the output when the input is the unit step:

$$s(t) = h(t) * u(t) = \int_{-\infty}^{t} h(\tau) \, d\tau$$

$$s[n] = h[n] * u[n] = \sum_{k=-\infty}^{n} h[k]$$

### 7.2 Relationship to Impulse Response

The step response is the **running integral** (continuous) or **running sum** (discrete) of the impulse response:

$$s(t) = \int_{-\infty}^{t} h(\tau) \, d\tau \quad \Leftrightarrow \quad h(t) = \frac{ds(t)}{dt}$$

$$s[n] = \sum_{k=-\infty}^{n} h[k] \quad \Leftrightarrow \quad h[n] = s[n] - s[n-1]$$

This means we can determine the impulse response by differentiating (or first-differencing) the step response, and vice versa.

### 7.3 Example

For $h[n] = (0.8)^n u[n]$:

$$s[n] = \sum_{k=0}^{n} (0.8)^k = \frac{1 - (0.8)^{n+1}}{1 - 0.8} = 5(1 - (0.8)^{n+1}), \quad n \geq 0$$

As $n \to \infty$: $s[\infty] = 5$, which is the **DC gain** of the system.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Impulse response and step response ---
n = np.arange(0, 30)
a = 0.8

h = a ** n  # impulse response
s = np.cumsum(h)  # step response = running sum of h

# Analytical step response
s_analytical = 5 * (1 - 0.8 ** (n + 1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].stem(n, h, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0].set_title(f'Impulse Response: $h[n] = ({a})^n u[n]$')
axes[0].set_xlabel('n')
axes[0].set_ylabel('h[n]')
axes[0].grid(True, alpha=0.3)

axes[1].stem(n, s, linefmt='r-', markerfmt='ro', basefmt='k-', label='Numerical')
axes[1].plot(n, s_analytical, 'k--', linewidth=1.5, label='Analytical')
axes[1].axhline(y=5, color='gray', linestyle=':', label='DC gain = 5')
axes[1].set_title(f'Step Response: $s[n] = 5(1 - {a}^{{n+1}})$')
axes[1].set_xlabel('n')
axes[1].set_ylabel('s[n]')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step_response.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 8. System Interconnections

LTI systems can be combined in several ways, and the resulting composite system is also LTI.

### 8.1 Cascade (Series) Connection

```
x → [h₁] → [h₂] → y
```

**Equivalent impulse response**: $h_{\text{eq}} = h_1 * h_2$

By associativity and commutativity of convolution:

$$y = x * h_1 * h_2 = x * (h_1 * h_2) = x * (h_2 * h_1)$$

The order of cascade stages does not matter (for LTI systems).

### 8.2 Parallel Connection

```
     ┌─[h₁]─┐
x ──►│       ├──► (+) → y
     └─[h₂]─┘
```

**Equivalent impulse response**: $h_{\text{eq}} = h_1 + h_2$

By distributivity:

$$y = x * h_1 + x * h_2 = x * (h_1 + h_2)$$

### 8.3 Feedback Connection

```
x → (+) → [h₁] → y
      ↑            │
      └── [h₂] ◄──┘
```

With negative feedback: $e[n] = x[n] - h_2[n] * y[n]$ and $y[n] = h_1[n] * e[n]$.

This gives: $y = h_1 * (x - h_2 * y)$

In the transform domain (where convolution becomes multiplication):

$$Y(z) = H_1(z)(X(z) - H_2(z)Y(z))$$

$$H_{\text{eq}}(z) = \frac{Y(z)}{X(z)} = \frac{H_1(z)}{1 + H_1(z)H_2(z)}$$

Feedback is fundamental for creating IIR (recursive) filters and control systems.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- System interconnections ---
n = np.arange(0, 40)

# Two simple systems
a1, a2 = 0.7, 0.5
h1 = np.where(n >= 0, a1**n, 0.0)
h2 = np.where(n >= 0, a2**n, 0.0)

# Input signal
x = np.zeros(40)
x[0] = 1.0  # impulse input

# Cascade
h_cascade = np.convolve(h1, h2)[:40]
y_cascade = np.convolve(x, h_cascade)[:40]

# Parallel
h_parallel = h1 + h2
y_parallel = np.convolve(x, h_parallel)[:40]

# Verify cascade commutativity
h_cascade_rev = np.convolve(h2, h1)[:40]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0, 0].stem(n, h1, linefmt='b-', markerfmt='bo', basefmt='k-', label='$h_1$')
axes[0, 0].stem(n + 0.2, h2, linefmt='r-', markerfmt='rs', basefmt='k-', label='$h_2$')
axes[0, 0].set_title('Individual Impulse Responses')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].stem(n, h_cascade, linefmt='g-', markerfmt='go', basefmt='k-',
                label='$h_1 * h_2$')
axes[0, 1].stem(n + 0.2, h_cascade_rev, linefmt='m-', markerfmt='ms', basefmt='k-',
                label='$h_2 * h_1$', alpha=0.5)
axes[0, 1].set_title('Cascade: $h_1 * h_2 = h_2 * h_1$')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].stem(n, h_parallel, linefmt='orange', markerfmt='o', basefmt='k-')
axes[1, 0].set_title('Parallel: $h_1 + h_2$')
axes[1, 0].grid(True, alpha=0.3)

# Compare cascade and parallel responses
axes[1, 1].stem(n, y_cascade, linefmt='g-', markerfmt='go', basefmt='k-',
                label='Cascade')
axes[1, 1].stem(n + 0.2, y_parallel, linefmt='orange', markerfmt='o', basefmt='k-',
                label='Parallel', alpha=0.7)
axes[1, 1].set_title('Impulse Response Comparison')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

for ax in axes.flat:
    ax.set_xlabel('n')

plt.tight_layout()
plt.savefig('system_interconnections.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 9. LTI System Stability

### 9.1 BIBO Stability Criterion

As established in Lesson 01, an LTI system is BIBO stable if and only if its impulse response is **absolutely summable** (discrete) or **absolutely integrable** (continuous):

**Discrete-time**:

$$\sum_{n=-\infty}^{\infty} |h[n]| < \infty$$

**Continuous-time**:

$$\int_{-\infty}^{\infty} |h(t)| \, dt < \infty$$

### 9.2 Stability of Common Systems

**First-order recursive**: $h[n] = a^n u[n]$

$$\sum_{n=0}^{\infty} |a|^n = \frac{1}{1 - |a|} < \infty \iff |a| < 1$$

Stable when the pole is inside the unit circle.

**FIR system**: $h[n]$ has finite duration (say $N$ samples)

$$\sum_{n} |h[n]| = \text{finite sum} < \infty$$

FIR systems are **always BIBO stable** (assuming finite coefficient values).

**Ideal integrator / accumulator**: $h[n] = u[n]$

$$\sum_{n=0}^{\infty} 1 = \infty$$

Unstable. A constant input produces linearly growing output.

### 9.3 Causality and Stability

For a causal LTI system, the impulse response satisfies $h[n] = 0$ for $n < 0$ (or $h(t) = 0$ for $t < 0$).

In the Z-transform domain, a causal system is stable when all poles of $H(z)$ lie **inside the unit circle** $|z| < 1$.

In the Laplace domain, a causal system is stable when all poles of $H(s)$ lie in the **left half-plane** $\text{Re}(s) < 0$.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Stability analysis for first-order systems ---
n = np.arange(0, 50)

poles = [0.3, 0.7, 0.95, 1.0, 1.05]
labels = ['a=0.3 (stable)', 'a=0.7 (stable)', 'a=0.95 (stable)',
          'a=1.0 (marginally unstable)', 'a=1.05 (unstable)']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for a, label in zip(poles, labels):
    h = a ** n
    axes[0].plot(n, h, linewidth=1.5, label=label)

axes[0].set_title('Impulse Response $h[n] = a^n u[n]$')
axes[0].set_xlabel('n')
axes[0].set_ylabel('h[n]')
axes[0].set_ylim([-1, 10])
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Cumulative sum of |h[n]| (should converge for stable systems)
for a, label in zip(poles, labels):
    h = a ** n
    cum_sum = np.cumsum(np.abs(h))
    axes[1].plot(n, cum_sum, linewidth=1.5, label=label)

axes[1].set_title('Running Sum $\\sum_{k=0}^{n} |h[k]|$')
axes[1].set_xlabel('n')
axes[1].set_ylabel('Partial sum')
axes[1].set_ylim([0, 50])
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stability_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 10. Frequency Response of LTI Systems

### 10.1 Complex Exponentials as Eigenfunctions

The defining property that makes frequency-domain analysis possible:

> If the input to an LTI system is a complex exponential $x[n] = e^{j\omega n}$, the output is:
>
> $$y[n] = H(e^{j\omega}) \cdot e^{j\omega n}$$
>
> where $H(e^{j\omega})$ is the **frequency response** of the system.

**Proof**:

$$y[n] = \sum_k h[k] e^{j\omega(n-k)} = e^{j\omega n} \sum_k h[k] e^{-j\omega k} = e^{j\omega n} H(e^{j\omega})$$

where:

$$H(e^{j\omega}) = \sum_{n=-\infty}^{\infty} h[n] e^{-j\omega n}$$

This is the **Discrete-Time Fourier Transform (DTFT)** of the impulse response.

### 10.2 Magnitude and Phase Response

The frequency response is generally complex:

$$H(e^{j\omega}) = |H(e^{j\omega})| e^{j\angle H(e^{j\omega})}$$

- $|H(e^{j\omega})|$ is the **magnitude response** — how much the system amplifies or attenuates each frequency
- $\angle H(e^{j\omega})$ is the **phase response** — how much the system delays each frequency component

### 10.3 Continuous-Time Frequency Response

For continuous-time LTI systems:

$$H(j\omega) = \int_{-\infty}^{\infty} h(t) e^{-j\omega t} \, dt$$

This is the **Fourier transform** of $h(t)$. If the input is $x(t) = e^{j\omega_0 t}$, the output is $y(t) = H(j\omega_0)e^{j\omega_0 t}$.

### 10.4 Example: Moving Average Filter

For a 5-point moving average: $h[n] = \frac{1}{5}$ for $n = 0, 1, 2, 3, 4$:

$$H(e^{j\omega}) = \frac{1}{5} \sum_{n=0}^{4} e^{-j\omega n} = \frac{1}{5} \cdot \frac{1 - e^{-j5\omega}}{1 - e^{-j\omega}}$$

The magnitude:

$$|H(e^{j\omega})| = \frac{1}{5} \left|\frac{\sin(5\omega/2)}{\sin(\omega/2)}\right|$$

This is a lowpass filter — it passes low frequencies and attenuates high frequencies.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Frequency response of common systems ---
omega = np.linspace(-np.pi, np.pi, 1000)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Moving average (M=5)
M = 5
H_ma = np.zeros_like(omega, dtype=complex)
for n in range(M):
    H_ma += (1 / M) * np.exp(-1j * omega * n)

axes[0, 0].plot(omega / np.pi, 20 * np.log10(np.abs(H_ma) + 1e-12), 'b-', linewidth=2)
axes[0, 0].set_title(f'Moving Average (M={M}) — Magnitude Response')
axes[0, 0].set_xlabel('Normalized Frequency ($\\omega/\\pi$)')
axes[0, 0].set_ylabel('Magnitude (dB)')
axes[0, 0].set_ylim([-30, 5])
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(omega / np.pi, np.angle(H_ma), 'r-', linewidth=2)
axes[0, 1].set_title(f'Moving Average (M={M}) — Phase Response')
axes[0, 1].set_xlabel('Normalized Frequency ($\\omega/\\pi$)')
axes[0, 1].set_ylabel('Phase (radians)')
axes[0, 1].grid(True, alpha=0.3)

# 2. First-order IIR: h[n] = a^n u[n], H(z) = 1/(1-az^{-1})
for a in [0.5, 0.8, 0.95]:
    H_iir = 1 / (1 - a * np.exp(-1j * omega))
    axes[1, 0].plot(omega / np.pi, 20 * np.log10(np.abs(H_iir)),
                    linewidth=2, label=f'a={a}')

axes[1, 0].set_title('First-Order IIR — Magnitude Response')
axes[1, 0].set_xlabel('Normalized Frequency ($\\omega/\\pi$)')
axes[1, 0].set_ylabel('Magnitude (dB)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 3. Effect of filtering on a signal
np.random.seed(42)
N = 500
n_sig = np.arange(N)
x_signal = np.sin(2 * np.pi * 0.05 * n_sig) + 0.5 * np.sin(2 * np.pi * 0.4 * n_sig)
x_noisy = x_signal + 0.3 * np.random.randn(N)

# Apply 11-point moving average
M_filt = 11
h_filt = np.ones(M_filt) / M_filt
y_filtered = np.convolve(x_noisy, h_filt, mode='same')

axes[1, 1].plot(n_sig, x_noisy, 'gray', linewidth=0.5, alpha=0.7, label='Noisy input')
axes[1, 1].plot(n_sig, y_filtered, 'b-', linewidth=1.5, label=f'MA({M_filt}) filtered')
axes[1, 1].plot(n_sig, np.sin(2 * np.pi * 0.05 * n_sig), 'r--', linewidth=1.5,
                label='Low-freq component')
axes[1, 1].set_title('Moving Average as Lowpass Filter')
axes[1, 1].set_xlabel('n')
axes[1, 1].legend(fontsize=9)
axes[1, 1].set_xlim([0, 200])
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('frequency_response.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 11. Python Examples

### 11.1 Complete Convolution Toolkit

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

class ConvolutionToolkit:
    """Comprehensive toolkit for computing and analyzing convolutions."""

    @staticmethod
    def convolve_direct(x, h):
        """Direct O(NM) convolution."""
        N, M = len(x), len(h)
        y = np.zeros(N + M - 1)
        for n in range(N + M - 1):
            for k in range(M):
                if 0 <= n - k < N:
                    y[n] += h[k] * x[n - k]
        return y

    @staticmethod
    def convolve_fft(x, h):
        """FFT-based O(N log N) convolution."""
        N = len(x) + len(h) - 1
        N_fft = 2 ** int(np.ceil(np.log2(N)))
        X = np.fft.fft(x, N_fft)
        H = np.fft.fft(h, N_fft)
        y = np.real(np.fft.ifft(X * H))
        return y[:N]

    @staticmethod
    def benchmark(N_values, M=50):
        """Compare direct vs FFT convolution speed."""
        print(f"{'N':>8} | {'Direct (ms)':>12} | {'FFT (ms)':>12} | {'NumPy (ms)':>12} | {'Speedup':>8}")
        print("-" * 60)

        for N in N_values:
            x = np.random.randn(N)
            h = np.random.randn(M)

            # Direct (only for small N)
            if N <= 5000:
                t0 = time.perf_counter()
                y_direct = ConvolutionToolkit.convolve_direct(x, h)
                t_direct = (time.perf_counter() - t0) * 1000
            else:
                t_direct = float('inf')

            # FFT
            t0 = time.perf_counter()
            y_fft = ConvolutionToolkit.convolve_fft(x, h)
            t_fft = (time.perf_counter() - t0) * 1000

            # NumPy
            t0 = time.perf_counter()
            y_np = np.convolve(x, h)
            t_numpy = (time.perf_counter() - t0) * 1000

            speedup = t_direct / t_fft if t_direct != float('inf') else float('inf')
            t_direct_str = f"{t_direct:.2f}" if t_direct != float('inf') else "skipped"
            print(f"{N:>8} | {t_direct_str:>12} | {t_fft:>12.2f} | {t_numpy:>12.2f} | {speedup:>8.1f}x")


# Run benchmark
print("=== Convolution Performance Benchmark ===\n")
ConvolutionToolkit.benchmark([100, 500, 1000, 5000, 10000, 50000])
```

### 11.2 Echo Simulation

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Echo system simulation ---
fs = 8000  # sampling rate
duration = 0.5
n = np.arange(int(fs * duration))
t = n / fs

# Original signal: sum of two sinusoids
f1, f2 = 200, 500
x = 0.7 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t)

# Fade in/out
fade_len = int(0.02 * fs)
x[:fade_len] *= np.linspace(0, 1, fade_len)
x[-fade_len:] *= np.linspace(1, 0, fade_len)

# Echo system: h[n] = delta[n] + 0.6*delta[n-D1] + 0.3*delta[n-D2]
D1 = int(0.1 * fs)   # 100ms delay
D2 = int(0.25 * fs)  # 250ms delay
h_echo = np.zeros(D2 + 1)
h_echo[0] = 1.0
h_echo[D1] = 0.6
h_echo[D2] = 0.3

# Apply echo
y = np.convolve(x, h_echo)
t_out = np.arange(len(y)) / fs

fig, axes = plt.subplots(3, 1, figsize=(14, 8))

axes[0].plot(t * 1000, x, 'b-', linewidth=0.8)
axes[0].set_title('Original Signal')
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)

axes[1].stem(np.arange(len(h_echo)) / fs * 1000, h_echo,
             linefmt='r-', markerfmt='ro', basefmt='k-')
axes[1].set_title(f'Echo Impulse Response: $\\delta[n] + 0.6\\delta[n-{D1}] + 0.3\\delta[n-{D2}]$')
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('h[n]')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_out * 1000, y, 'g-', linewidth=0.8)
axes[2].set_title('Output with Echo')
axes[2].set_xlabel('Time (ms)')
axes[2].set_ylabel('Amplitude')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('echo_simulation.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 11.3 System Identification from Step Response

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Recover impulse response from step response ---

# Unknown system: 3-tap FIR h = [0.5, 1.0, 0.5]
h_true = np.array([0.5, 1.0, 0.5])

# Apply unit step to get step response
N = 30
u = np.ones(N)  # unit step
s = np.convolve(u, h_true)[:N]

# Recover impulse response by first differencing
h_recovered = np.zeros(N)
h_recovered[0] = s[0]
h_recovered[1:] = np.diff(s)

print("True impulse response:", h_true)
print("Recovered (first 5):", h_recovered[:5])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

n = np.arange(N)

axes[0].stem(np.arange(len(h_true)), h_true, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0].set_title('True Impulse Response $h[n]$')
axes[0].set_xlabel('n')
axes[0].grid(True, alpha=0.3)

axes[1].stem(n, s, linefmt='r-', markerfmt='ro', basefmt='k-')
axes[1].set_title('Measured Step Response $s[n]$')
axes[1].set_xlabel('n')
axes[1].grid(True, alpha=0.3)

axes[2].stem(n[:10], h_recovered[:10], linefmt='g-', markerfmt='go', basefmt='k-')
axes[2].set_title('Recovered $h[n] = s[n] - s[n-1]$')
axes[2].set_xlabel('n')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('system_identification.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 11.4 Frequency Response Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Frequency response from impulse response ---

def plot_freq_response(h, fs=1.0, title="System"):
    """Plot magnitude and phase response of a discrete-time system."""
    # Compute DTFT at 1024 frequency points
    N_freq = 1024
    omega = np.linspace(0, np.pi, N_freq)
    H = np.zeros(N_freq, dtype=complex)
    for k, w in enumerate(omega):
        for n, hn in enumerate(h):
            H[k] += hn * np.exp(-1j * w * n)

    mag_db = 20 * np.log10(np.abs(H) + 1e-12)
    phase = np.unwrap(np.angle(H))
    group_delay = -np.diff(phase) / np.diff(omega)

    freq = omega * fs / (2 * np.pi) if fs != 1.0 else omega / np.pi

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    # Magnitude
    axes[0].plot(freq, mag_db, 'b-', linewidth=2)
    axes[0].set_title(f'{title} — Magnitude Response')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].grid(True, alpha=0.3)

    # Phase
    axes[1].plot(freq, phase * 180 / np.pi, 'r-', linewidth=2)
    axes[1].set_title(f'{title} — Phase Response')
    axes[1].set_ylabel('Phase (degrees)')
    axes[1].grid(True, alpha=0.3)

    # Group delay
    axes[2].plot(freq[:-1], group_delay, 'g-', linewidth=2)
    axes[2].set_title(f'{title} — Group Delay')
    axes[2].set_xlabel('Frequency (Hz)' if fs != 1.0 else 'Normalized Frequency ($\\omega/\\pi$)')
    axes[2].set_ylabel('Samples')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'freq_response_{title.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()


# Example 1: 7-point moving average
M = 7
h_ma = np.ones(M) / M
plot_freq_response(h_ma, title="7-Point Moving Average")

# Example 2: Difference filter (high-pass)
h_diff = np.array([1, -1])
plot_freq_response(h_diff, title="First Difference")

# Example 3: Bandpass-like filter
h_bp = np.array([0.1, -0.2, 0.5, 1.0, 0.5, -0.2, 0.1])
plot_freq_response(h_bp, title="Bandpass-Like FIR")
```

---

## 12. Summary

### Key Formulas

| Concept | Discrete-Time | Continuous-Time |
|---------|--------------|-----------------|
| Convolution | $y[n] = \sum_k x[k]h[n-k]$ | $y(t) = \int x(\tau)h(t-\tau)d\tau$ |
| Impulse response | $h[n] = \mathcal{T}\{\delta[n]\}$ | $h(t) = \mathcal{T}\{\delta(t)\}$ |
| Step response | $s[n] = \sum_{k \leq n} h[k]$ | $s(t) = \int_{-\infty}^{t} h(\tau)d\tau$ |
| Frequency response | $H(e^{j\omega}) = \sum_n h[n]e^{-j\omega n}$ | $H(j\omega) = \int h(t)e^{-j\omega t}dt$ |
| BIBO stability | $\sum |h[n]| < \infty$ | $\int |h(t)|dt < \infty$ |

### Convolution Properties

| Property | Expression |
|----------|-----------|
| Commutativity | $x * h = h * x$ |
| Associativity | $(x * h_1) * h_2 = x * (h_1 * h_2)$ |
| Distributivity | $x * (h_1 + h_2) = x * h_1 + x * h_2$ |
| Identity | $x * \delta = x$ |
| Shift | $x * \delta_{n_0} = x[n - n_0]$ |
| Width | Support of $y$ = sum of supports |

### Conceptual Map

```
          Impulse δ[n]
              │
              ▼
    LTI System T{·}  ──────►  Impulse Response h[n]
              │                        │
              │                ┌───────┼───────┐
              │                ▼       ▼       ▼
              │           Stability  Freq.   Step
              │           Check      Response Response
              │
    Any Input x[n]
              │
              ▼
    y[n] = x[n] * h[n]  (CONVOLUTION)
```

The key insight of this lesson: **for LTI systems, everything reduces to convolution**. Once you know the impulse response, you know everything about the system.

---

## 13. Exercises

### Exercise 1: Analytical Convolution

Compute the following convolutions by hand. Verify your results using `np.convolve()`.

1. $x[n] = \{2, 1, -1\}$ and $h[n] = \{1, 3, 2\}$ (starting at $n = 0$)
2. $x[n] = u[n] - u[n-4]$ and $h[n] = (0.5)^n u[n]$
3. $x[n] = (0.8)^n u[n]$ and $h[n] = (0.6)^n u[n]$ (use the closed-form of geometric series)

### Exercise 2: Continuous-Time Convolution

Compute analytically:

1. $y(t) = e^{-t}u(t) * e^{-2t}u(t)$
2. $y(t) = u(t) * u(t)$ (what is the result, and why does it reveal a stability issue?)
3. $y(t) = \text{rect}(t/2) * e^{-t}u(t)$

### Exercise 3: System Analysis

A causal LTI system is described by the difference equation:

$$y[n] = 0.5 y[n-1] + x[n]$$

1. Find the impulse response $h[n]$ by setting $x[n] = \delta[n]$ and iterating
2. Is the system BIBO stable? Prove your answer
3. Compute the step response
4. Find the frequency response $H(e^{j\omega})$
5. Plot the magnitude and phase response using Python

### Exercise 4: Cascade vs. Parallel

Two LTI systems have impulse responses:

$$h_1[n] = (0.7)^n u[n], \qquad h_2[n] = (0.5)^n u[n]$$

1. Find the impulse response of the cascade connection $h_1 * h_2$
2. Find the impulse response of the parallel connection $h_1 + h_2$
3. Verify numerically that the cascade order does not matter: $h_1 * h_2 = h_2 * h_1$
4. For input $x[n] = \delta[n] - 0.3\delta[n-1]$, compute and plot the output for both configurations

### Exercise 5: Moving Average Filter Analysis

For an $M$-point moving average filter $h[n] = \frac{1}{M}$ for $n = 0, 1, \ldots, M-1$:

1. Derive the frequency response $H(e^{j\omega})$ in closed form
2. At what frequencies are the nulls (zeros of $H$)?
3. What is the 3-dB bandwidth as a function of $M$?
4. Plot the magnitude response for $M = 3, 7, 15, 31$ on the same graph
5. Apply each filter to a signal containing 50 Hz and 400 Hz components at $f_s = 1000$ Hz. Which $M$ best separates them?

### Exercise 6: Convolution in Practice

1. Generate a 1-second chirp signal sweeping from 100 Hz to 2000 Hz at $f_s = 8000$ Hz
2. Create an echo system with three reflections at 50ms, 120ms, and 200ms with decreasing amplitudes
3. Convolve the chirp with the echo system
4. Compare the direct convolution time vs. FFT convolution time
5. Plot the input and output spectrograms side by side

### Exercise 7: System Identification

You are given access to an unknown LTI system (a black box). You can apply any input and observe the output.

1. Apply a unit impulse to directly measure $h[n]$
2. Apply a unit step to measure $s[n]$, then recover $h[n] = s[n] - s[n-1]$
3. Apply white noise of length 10000 and use cross-correlation to estimate $h[n]$
4. Compare the three methods. Which gives the most accurate result with the least signal energy?

Implement this using the following black-box system (do not look at the implementation when testing):

```python
def black_box_system(x):
    """Unknown LTI system — treat as a black box."""
    h = np.array([0.2, 0.5, 1.0, 0.5, 0.2, -0.1, -0.05])
    return np.convolve(x, h, mode='full')[:len(x)]
```

### Exercise 8: Deconvolution

Given $y[n] = x[n] * h[n]$ where $h[n]$ is known and $y[n]$ is measured, recovering $x[n]$ is called **deconvolution**.

1. Generate a sparse signal $x[n]$ with 5 nonzero values in 100 samples
2. Convolve with $h[n] = [1, 0.5, 0.25]$ to get $y[n]$
3. Add small noise: $y_{\text{noisy}}[n] = y[n] + 0.01 \cdot w[n]$ where $w$ is white noise
4. Attempt deconvolution using FFT: $X = Y/H$ in the frequency domain
5. Explain why direct deconvolution is problematic (hint: consider frequencies where $H \approx 0$)
6. Implement Wiener deconvolution: $\hat{X} = \frac{H^* Y}{|H|^2 + \lambda}$ and show the improvement

---

## 14. References

1. Oppenheim, A. V. & Willsky, A. S. *Signals and Systems* (2nd ed.), Ch. 2-3. Prentice Hall, 1997.
2. Oppenheim, A. V. & Schafer, R. W. *Discrete-Time Signal Processing* (3rd ed.), Ch. 2. Pearson, 2010.
3. Haykin, S. & Van Veen, B. *Signals and Systems* (2nd ed.), Ch. 2-4. Wiley, 2003.
4. Smith, S. W. *The Scientist and Engineer's Guide to Digital Signal Processing*, Ch. 6-7. California Technical Publishing, 1997. (Free online: dspguide.com)

---

[Previous: 01. Signals and Systems](./01_Signals_and_Systems.md) | [Next: 03. Fourier Series and Applications](./03_Fourier_Series_and_Applications.md) | [Overview](./00_Overview.md)
