# Lesson 15: Digital Control Systems

## Learning Objectives

- Understand the sampling process and its effects on control systems
- Apply the Z-transform to analyze discrete-time systems
- Convert continuous-time controllers to discrete-time implementations
- Design digital controllers directly in the Z-domain
- Analyze stability of discrete-time systems

## 1. Why Digital Control?

Modern controllers are implemented on **digital computers** (microcontrollers, DSPs, FPGAs). The continuous-time signals must be:

1. **Sampled** at discrete time instants $t = kT$ (where $T$ is the sampling period)
2. **Quantized** to finite precision (A/D conversion)
3. **Processed** by the digital controller algorithm
4. **Reconstructed** as a continuous-time signal (D/A conversion with zero-order hold)

```
r(t)→[Sampler]→r[k]→[Digital Controller]→u[k]→[ZOH]→u(t)→[Plant]→y(t)
                                                                   ↓
                                                            [Sampler]←─┘
                                                              y[k]
```

### 1.1 Advantages of Digital Control

- Flexibility (algorithm changes via software, not hardware)
- Repeatability (no component drift)
- Complex algorithms feasible (adaptive, optimal, nonlinear)
- Noise immunity of digital signals
- Easy logging and monitoring

### 1.2 Sampling Theorem

**Shannon/Nyquist sampling theorem:** To reconstruct a signal with bandwidth $\omega_B$, the sampling frequency must satisfy:

$$\omega_s = \frac{2\pi}{T} > 2\omega_B$$

In control practice, the sampling rate is typically chosen as:

$$\omega_s \geq (10\text{–}30) \times \omega_{BW}$$

where $\omega_{BW}$ is the closed-loop bandwidth. This ensures the digital controller behaves similarly to its continuous-time counterpart.

## 2. Zero-Order Hold (ZOH)

The **zero-order hold** converts the discrete control signal to a piecewise-constant continuous signal:

$$u(t) = u[k] \quad \text{for } kT \leq t < (k+1)T$$

Transfer function of the ZOH:

$$G_{\text{ZOH}}(s) = \frac{1 - e^{-sT}}{s}$$

The ZOH introduces a time delay of approximately $T/2$, which adds phase lag and can destabilize the system if $T$ is too large.

## 3. The Z-Transform

### 3.1 Definition

The Z-transform of a discrete-time sequence $f[k]$ is:

$$F(z) = \mathcal{Z}\{f[k]\} = \sum_{k=0}^{\infty} f[k] z^{-k}$$

The Z-transform is the discrete-time analog of the Laplace transform.

### 3.2 Key Z-Transform Pairs

| $f[k]$ | $F(z)$ |
|---------|--------|
| $\delta[k]$ (unit impulse) | $1$ |
| $u[k]$ (unit step) | $\frac{z}{z-1}$ |
| $k u[k]$ (ramp) | $\frac{Tz}{(z-1)^2}$ |
| $a^k u[k]$ | $\frac{z}{z-a}$ |
| $e^{-akT} u[k]$ | $\frac{z}{z-e^{-aT}}$ |

### 3.3 Key Properties

| Property | Time Domain | Z-Domain |
|----------|------------|----------|
| Linearity | $af_1[k] + bf_2[k]$ | $aF_1(z) + bF_2(z)$ |
| Time shift | $f[k-1]$ | $z^{-1}F(z)$ |
| Time advance | $f[k+1]$ | $zF(z) - zf[0]$ |
| Final value | $\lim_{k\to\infty} f[k]$ | $\lim_{z\to 1}(z-1)F(z)$ |

### 3.4 The s-to-z Mapping

The fundamental relationship between $s$ and $z$ domains:

$$z = e^{sT}$$

This maps:
- Left half of $s$-plane ($\text{Re}(s) < 0$) → inside the unit circle ($|z| < 1$)
- Imaginary axis ($\text{Re}(s) = 0$) → unit circle ($|z| = 1$)
- Right half of $s$-plane ($\text{Re}(s) > 0$) → outside the unit circle ($|z| > 1$)

**Stability in the Z-domain:** A discrete-time system is stable if and only if all poles are **inside the unit circle** $|z| < 1$.

## 4. Discrete-Time Transfer Functions

### 4.1 Pulse Transfer Function

For a discrete-time system $y[k] = G(z) U(z)$:

$$G(z) = \frac{Y(z)}{U(z)} = \frac{b_m z^m + b_{m-1}z^{m-1} + \cdots + b_0}{z^n + a_{n-1}z^{n-1} + \cdots + a_0}$$

### 4.2 ZOH-Equivalent Discretization

Given a continuous plant $G_p(s)$, the **ZOH-equivalent** discrete transfer function is:

$$G_d(z) = (1 - z^{-1})\mathcal{Z}\left\{\frac{G_p(s)}{s}\right\}$$

This exactly captures the behavior of the continuous plant preceded by a ZOH at the sampling instants.

### 4.3 Example

For $G_p(s) = \frac{1}{s+a}$:

$$\frac{G_p(s)}{s} = \frac{1}{s(s+a)} = \frac{1}{a}\left(\frac{1}{s} - \frac{1}{s+a}\right)$$

$$\mathcal{Z}\left\{\frac{G_p(s)}{s}\right\} = \frac{1}{a}\left(\frac{z}{z-1} - \frac{z}{z-e^{-aT}}\right)$$

$$G_d(z) = \frac{1-e^{-aT}}{z - e^{-aT}}$$

## 5. Discretization of Continuous Controllers

### 5.1 Common Methods

Given a continuous controller $G_c(s)$, approximate in the Z-domain:

**Forward Euler** (forward difference):

$$s \approx \frac{z-1}{T} \quad \Rightarrow \quad G_d(z) = G_c(s)\big|_{s=(z-1)/T}$$

**Backward Euler** (backward difference):

$$s \approx \frac{z-1}{Tz} \quad \Rightarrow \quad G_d(z) = G_c(s)\big|_{s=(z-1)/(Tz)}$$

**Tustin's method** (bilinear transform, trapezoidal rule):

$$s \approx \frac{2}{T}\frac{z-1}{z+1} \quad \Rightarrow \quad G_d(z) = G_c(s)\big|_{s=\frac{2}{T}\frac{z-1}{z+1}}$$

### 5.2 Comparison

| Method | Stability Preservation | Frequency Warping | Accuracy |
|--------|----------------------|-------------------|----------|
| Forward Euler | No (can map stable to unstable) | No | Low |
| Backward Euler | Yes (maps LHP to inside unit circle) | No | Low |
| Tustin | Yes | Yes (frequency warping) | Good |
| ZOH-equivalent | Yes | N/A (exact at sample times) | Exact at $t = kT$ |

**Tustin's frequency warping:** The bilinear transform maps $\omega_s$ (continuous) to $\omega_d$ (discrete) nonlinearly:

$$\omega_d = \frac{2}{T}\tan\frac{\omega_s T}{2}$$

For important frequencies (crossover, notch), **pre-warp** by adjusting the continuous frequency before applying the bilinear transform.

### 5.3 Digital PID Implementation

The continuous PID $u(t) = K_p e + K_i \int e \, dt + K_d \dot{e}$ becomes:

**Positional form** (using Tustin for integral, backward Euler for derivative):

$$u[k] = K_p e[k] + K_i T \sum_{j=0}^{k} e[j] + K_d \frac{e[k] - e[k-1]}{T}$$

**Velocity (incremental) form** (preferred — avoids integral windup naturally):

$$\Delta u[k] = K_p(e[k] - e[k-1]) + K_i T e[k] + K_d \frac{e[k] - 2e[k-1] + e[k-2]}{T}$$

$$u[k] = u[k-1] + \Delta u[k]$$

## 6. Discrete-Time Stability Analysis

### 6.1 Jury Stability Test

The discrete-time analog of the Routh-Hurwitz criterion. For the characteristic polynomial:

$$P(z) = a_n z^n + a_{n-1}z^{n-1} + \cdots + a_0$$

**Necessary conditions** (all must hold):
1. $P(1) > 0$
2. $(-1)^n P(-1) > 0$
3. $|a_0| < a_n$

The full Jury array provides the sufficient condition (analogous to the Routh array).

### 6.2 Bilinear Transformation to Use Routh

An alternative: substitute $z = \frac{w+1}{w-1}$ (bilinear transform) into $P(z)$ to get a polynomial in $w$. The unit circle in $z$ maps to the imaginary axis in $w$, so the standard Routh-Hurwitz criterion can be applied to the $w$-polynomial.

## 7. Direct Digital Design

Instead of designing in continuous time and discretizing, design **directly** in the Z-domain:

### 7.1 Dead-Beat Control

Choose the closed-loop transfer function to achieve **zero error in finite time** (typically $n$ sampling periods for an $n$-th order system):

$$T(z) = z^{-d}$$

for some delay $d$. The controller is:

$$G_c(z) = \frac{T(z)}{(1 - T(z))G_d(z)}$$

Dead-beat controllers are fast but require large control signals and are sensitive to model errors.

### 7.2 Discrete Root Locus

The root locus technique applies identically in the Z-domain, with the stability boundary being the **unit circle** instead of the imaginary axis.

## Practice Exercises

### Exercise 1: ZOH Discretization

For the plant $G_p(s) = \frac{5}{s(s+5)}$ with sampling period $T = 0.1$ s:

1. Compute the ZOH-equivalent discrete transfer function $G_d(z)$
2. Find the discrete poles and verify they correspond to $z = e^{sT}$ applied to the continuous poles
3. Check the DC gain: $G_d(1) = G_p(0) \cdot T$ for the step response

### Exercise 2: Digital PID

A continuous PI controller $G_c(s) = 2(1 + \frac{1}{0.5s})$ is to be implemented digitally with $T = 0.05$ s.

1. Discretize using Tustin's method
2. Write the difference equation $u[k] = f(u[k-1], e[k], e[k-1])$
3. Compare with the velocity form implementation

### Exercise 3: Stability Analysis

For the discrete-time characteristic polynomial $P(z) = z^3 - 1.2z^2 + 0.5z - 0.1$:

1. Apply the necessary conditions of the Jury test
2. Determine if the system is stable
3. Find the actual roots and verify

---

*Previous: [Lesson 14 — Optimal Control: LQR and Kalman Filter](14_Optimal_Control.md) | Next: [Lesson 16 — Nonlinear Control and Advanced Topics](16_Nonlinear_and_Advanced.md)*
