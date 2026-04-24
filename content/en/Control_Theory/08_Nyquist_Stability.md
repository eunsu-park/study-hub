# Lesson 8: Nyquist Stability Criterion

## Learning Objectives

- Understand the Nyquist contour and its role in stability analysis
- State and apply the Nyquist stability criterion
- Sketch Nyquist plots for common transfer functions
- Determine closed-loop stability from the Nyquist plot
- Read gain margin and phase margin from the Nyquist diagram
- Plot Nyquist diagrams numerically and recognize the visual cues that signal trouble

## 0. Why Yet Another Stability Test?

Routh tells you whether the closed loop is stable from the polynomial. Root locus tells you how the poles move with gain. Nyquist tells you something the other two cannot: **how stable, and how robust to delay or modeling error**. Three reasons it earns a lesson of its own:

- **It works on measured data.** Sweep a real plant with a sine generator, record magnitude and phase, plot — done. No model needed. This is how 1940s aircraft control engineers tuned autopilots before Routh-Hurwitz was practical.
- **It handles dead time naturally.** A pure delay $e^{-sT}$ is irrational; Routh and root locus cannot touch it. Nyquist eats it for breakfast — delay just adds linear phase.
- **It exposes margin.** The distance from the Nyquist curve to the critical point $-1$ tells you how much gain you can multiply by, or how much phase you can add, before the system goes unstable. Routh gives a yes/no; Nyquist gives a "how much further can we push?"

The conceptual leap: stability is no longer a polynomial root question — it is a **encirclement counting** question on a frequency-response plot.

## 1. Motivation

The Routh-Hurwitz criterion works with the characteristic polynomial, and root locus tracks poles as gain varies. The **Nyquist criterion** determines closed-loop stability entirely from the **open-loop frequency response** — which can be measured experimentally even when no analytical model is available.

Key advantages:
- Works with measured frequency response data
- Handles time delays (which are difficult for root locus and Routh)
- Provides a clear graphical interpretation of stability margins
- Reveals the mechanism by which instability occurs

## 2. Mathematical Foundation

### 2.1 The Principle of the Argument

**Cauchy's theorem (principle of the argument):** If a contour $\Gamma_s$ in the $s$-plane encircles $Z$ zeros and $P$ poles of a function $F(s)$ (counted with multiplicity) in the clockwise direction, then the image $\Gamma_F = F(\Gamma_s)$ encircles the origin $N$ times clockwise, where:

$$N = Z - P$$

### 2.2 Application to Control

For a closed-loop system, the characteristic equation is:

$$1 + G(s)H(s) = 0$$

Define $F(s) = 1 + G(s)H(s)$. We need to check whether $F(s)$ has zeros in the right half-plane (RHP).

Choose $\Gamma_s$ = the **Nyquist contour** (entire right half-plane boundary):
- The imaginary axis from $-j\infty$ to $+j\infty$
- A semicircle of infinite radius closing to the right

Then: $Z$ = number of closed-loop RHP poles (what we want to find), $P$ = number of open-loop RHP poles (known).

Since mapping through $F(s) = 1 + G(s)H(s)$ shifts the plot by 1, encirclements of the origin by $F(s)$ equal encirclements of **the point $-1 + j0$** by $G(s)H(s)$.

## 3. The Nyquist Stability Criterion

### 3.1 Statement

Let $P$ = number of open-loop poles of $G(s)H(s)$ in the open right half-plane. Let $N$ = number of clockwise encirclements of the point $(-1, 0)$ by the Nyquist plot of $G(s)H(s)$.

Then the number of closed-loop RHP poles is:

$$Z = N + P$$

**The closed-loop system is stable if and only if $Z = 0$, i.e., $N = -P$** (the Nyquist plot must encircle $(-1, 0)$ exactly $P$ times counterclockwise).

### 3.2 Special Case: Stable Open-Loop System

If the open-loop system is stable ($P = 0$), then for closed-loop stability:

$$Z = N = 0$$

The Nyquist plot must **not encircle** the point $(-1, 0)$.

### 3.3 Counting Encirclements

- Draw a line from $(-1, 0)$ to infinity in any direction
- Count crossings: clockwise crossing = $+1$, counterclockwise = $-1$
- Net count = $N$

A common pictorial mnemonic: stand at $(-1, 0)$ and watch the Nyquist plot trace itself out. Each time the curve passes you on your left going downward (or right going upward, etc.), that is one clockwise encirclement; the opposite is counterclockwise. With practice you can read $N$ in a glance.

## 4. Sketching Nyquist Plots

### 4.1 Nyquist Contour Segments

The Nyquist plot maps three segments of the contour:

1. **Positive imaginary axis** ($s = j\omega$, $\omega: 0 \to \infty$): This is just the polar plot of $G(j\omega)H(j\omega)$
2. **Infinite semicircle** ($s = Re^{j\theta}$, $R \to \infty$): Usually maps to a single point (often the origin for strictly proper systems)
3. **Negative imaginary axis** ($s = -j\omega$, $\omega: \infty \to 0$): The mirror image (conjugate) of segment 1

### 4.2 Handling Poles on the Imaginary Axis

If $G(s)H(s)$ has poles on the imaginary axis, the Nyquist contour must detour around them with small semicircular indentations to the right.

**For a pole at the origin** (integrator): The small semicircle $s = \epsilon e^{j\theta}$ ($\theta: -90° \to +90°$) maps to a large arc going to infinity and sweeping through $180°$ per integrator.

### 4.3 Example: Type 0 System

$G(s) = \frac{K}{(s+1)(s+2)}$ (no integrator, $P = 0$)

- At $\omega = 0$: $G(0) = K/2$ (real, positive)
- As $\omega \to \infty$: $|G| \to 0$, $\angle G \to -180°$
- The plot starts at $(K/2, 0)$ and spirals toward the origin

For stability: the plot must not encircle $(-1, 0)$. Since the maximum negative real-axis intercept is finite, there exists a maximum $K$ for stability.

### 4.4 Example: Type 1 System

$G(s) = \frac{K}{s(s+1)}$ ($P = 0$, pole at origin)

- Detour around origin: large arc from $+90°$ to $-90°$ at infinite radius
- At $\omega = 0^+$: $|G| \to \infty$, $\angle G = -90°$
- As $\omega \to \infty$: $|G| \to 0$, $\angle G \to -180°$
- The plot comes from $-j\infty$, crosses the negative real axis, and approaches the origin

**Real-axis crossing:** Set $\text{Im}[G(j\omega)] = 0$ and solve for $\omega$ to find where the plot crosses the real axis. At this frequency, the gain determines the distance from $(-1, 0)$.

### 4.5 Plotting in Python

A few lines of `python-control` produce a publication-quality Nyquist plot:

```python
import numpy as np
import matplotlib.pyplot as plt
from control import TransferFunction, nyquist_plot

# Type 1 example: G(s) = 1 / (s(s+1)(s+2))
G = TransferFunction([1], [1, 3, 2, 0])

omega = np.logspace(-2, 2, 500)
nyquist_plot(G, omega=omega)
plt.title("Nyquist plot of 1 / [s(s+1)(s+2)]")
plt.grid(True)
plt.show()
```

The marker on $(-1, 0)$ is what your eye should be drawn to. If the curve passes between $(-1, 0)$ and the origin (i.e., crosses the negative real axis to the left of $-1$), the closed loop with unity gain is unstable.

If `python-control` is unavailable, the manual sweep is short:

```python
omega = np.logspace(-2, 2, 1000)
s = 1j * omega
G = 1 / (s * (s + 1) * (s + 2))
plt.figure(figsize=(6, 6))
plt.plot(G.real, G.imag, 'b')
plt.plot(G.real, -G.imag, 'b--')      # mirror — negative-omega half
plt.plot(-1, 0, 'rx', markersize=10)  # critical point
plt.axhline(0, color='gray', lw=0.5); plt.axvline(0, color='gray', lw=0.5)
plt.xlabel("Re[G]"); plt.ylabel("Im[G]"); plt.axis('equal'); plt.grid(True)
plt.show()
```

## 5. Stability Margins on the Nyquist Plot

### 5.1 Gain Margin

The gain margin is the reciprocal of $|G(j\omega_{pc})|$ where $\omega_{pc}$ is the phase crossover frequency ($\angle G = -180°$):

$$GM = \frac{1}{|G(j\omega_{pc})|}$$

Graphically: the gain margin is the factor by which the Nyquist plot can be **scaled up** before it passes through $(-1, 0)$.

On the Nyquist plot, the gain margin is the distance from the origin to the negative real axis crossing, relative to the distance from the origin to $(-1, 0)$:

$$GM_{\text{dB}} = 20\log_{10}\frac{1}{|G(j\omega_{pc})|}$$

### 5.2 Phase Margin

The phase margin is measured where the Nyquist plot crosses the unit circle ($|G| = 1$):

$$PM = 180° + \angle G(j\omega_{gc})$$

Graphically: the angular distance from the $(-1, 0)$ direction to the point where the plot crosses the unit circle.

### 5.3 Geometric Interpretation

The distance from the Nyquist plot to the critical point $(-1, 0)$ represents the **robustness** of the system. This motivates the sensitivity peak:

$$M_s = \max_\omega |S(j\omega)| = \max_\omega \frac{1}{|1 + G(j\omega)H(j\omega)|}$$

$M_s$ is the inverse of the minimum distance from the Nyquist plot to $(-1, 0)$.

**Typical design:** $M_s \leq 2$ (6 dB), which guarantees $GM \geq 2$ (6 dB) and $PM \geq 29°$.

### 5.4 What Margins Are Worth in Practice

A few rules of thumb worth memorizing:

| Industry | Typical $GM$ | Typical $PM$ | Notes |
|----------|--------------|--------------|-------|
| Process control | 6–12 dB | $30°$–$60°$ | High noise, slow plants |
| Aerospace flight control | $\geq 6$ dB | $\geq 45°$ | Mil-spec values |
| Servo positioning | 3–6 dB | $40°$–$60°$ | Trade speed vs. ringing |
| Power electronics | 6 dB | $45°$ | Switching noise budget |

A loop with $GM = 30$ dB is over-designed (slow); $GM = 1$ dB is one model error away from instability. The middle band is where engineering happens.

## 6. Systems with Time Delay

A pure time delay $e^{-sT}$ has:
- $|e^{-j\omega T}| = 1$ (unity magnitude for all $\omega$)
- $\angle e^{-j\omega T} = -\omega T$ (phase decreases linearly with $\omega$)

Time delay adds unbounded phase lag without affecting magnitude. On the Nyquist plot, it causes the curve to spiral inward, making encirclements more likely.

**Example:** $G(s) = \frac{Ke^{-sT}}{s+1}$

The Nyquist plot of $K/(s+1)$ is a semicircle. With delay, the plot spirals — and for large enough $KT$, it will encircle $(-1, 0)$.

The Nyquist criterion handles this naturally, while Routh-Hurwitz and root locus cannot (they require rational transfer functions).

> **Delay-margin shortcut**: From the phase margin, the delay margin is $T_d = \text{PM (rad)} / \omega_{gc}$. A loop with $PM = 60°$ ($\approx 1.05$ rad) at $\omega_{gc} = 10$ rad/s tolerates an extra 105 ms of delay before going unstable. Network-controlled systems live or die by this number.

## 7. Nyquist Criterion for Non-Minimum Phase Systems

A **non-minimum phase** system has open-loop RHP poles or zeros. If $P > 0$ (open-loop RHP poles), stability requires $P$ counterclockwise encirclements of $(-1, 0)$.

This is important for systems like:
- Inverted pendulum
- Certain aircraft dynamics (non-minimum phase zeros)
- Systems with internal positive feedback

## 8. Common Pitfalls

1. **Forgetting the negative-$\omega$ half.** The Nyquist plot covers $\omega: -\infty \to +\infty$; many textbook plots show only $\omega \geq 0$ and you must mentally mirror. Numerical libraries usually plot both halves automatically — confirm before counting encirclements.
2. **Counting encirclements with the wrong sign convention.** Some textbooks call clockwise positive, others negative. Internal consistency matters more than which one you pick — but state your convention and stick with it.
3. **Mishandling poles on the imaginary axis.** Integrator poles produce infinite arcs on the Nyquist plot. The arc goes the OPPOSITE way you might expect — for a single integrator, $-90°$ to $+90°$ on the small detour maps to $+90°$ to $-90°$ on the infinite arc (because $1/s$ inverts the angle). Skip this and your encirclement count can be off by 2.
4. **Reading margins from the polar plot at the wrong frequency.** Gain margin is measured at $\omega_{pc}$ (phase = $-180°$), phase margin at $\omega_{gc}$ (gain = 1). Beginners sometimes read both at the same frequency, which is generally wrong.
5. **Assuming "no encirclement = stable" without checking $P$.** The criterion is $N + P = 0$. If your open-loop already has RHP poles, "no encirclement" actually means UNSTABLE. The number of CCW encirclements must equal $P$, not $0$.
6. **Trusting a small-axes plot at the critical point.** When the curve passes very close to $-1$ ($M_s$ near 2), a coarsely-sampled $\omega$ grid can fool the eye — sample more densely near $\omega_{pc}$ before declaring victory.

## Practice Exercises

### Exercise 1: Nyquist Plot and Stability

For $G(s) = \frac{K}{s(s+1)(s+2)}$:

1. Sketch the Nyquist plot (showing the key features)
2. Find the real-axis crossing point (frequency and magnitude)
3. Determine the maximum $K$ for closed-loop stability using the Nyquist criterion
4. Verify your answer using the Routh-Hurwitz criterion

### Exercise 2: Non-Minimum Phase

For $G(s) = \frac{K(s-1)}{s(s+2)}$ (has an RHP zero):

1. How many open-loop RHP poles are there? ($P = ?$)
2. Sketch the Nyquist plot for $K = 5$
3. Does the Nyquist plot encircle $(-1, 0)$? How many times?
4. Is the closed-loop system stable?

### Exercise 3: Time Delay

A system has $G(s) = \frac{2e^{-0.5s}}{s+1}$.

1. Find the phase crossover frequency (where $\angle G(j\omega) = -180°$)
2. Find the gain margin
3. What is the maximum additional time delay the system can tolerate before becoming unstable?

### Exercise 4: Margin Design

A plant has open-loop transfer function $G(s) = \frac{10}{(s+1)(s+5)}$. Design a unity-feedback gain $K$ such that the closed loop has $PM \geq 45°$. Use the Python snippet from Section 4.5 to plot the Nyquist diagram with $K$ as a multiplicative factor and iterate.

### Exercise 5: Sensitivity Peak

Continuing Exercise 4, compute $M_s$ and verify that your design satisfies $M_s \leq 2$. If not, reduce $K$ until it does. Plot the Nyquist diagram and shade the disk of radius $1/M_s$ centered at $-1$ — your curve should not enter this disk.

---

*Previous: [Lesson 7 — Frequency Response: Bode Plots](07_Bode_Plots.md) | Next: [Lesson 9 — PID Control](09_PID_Control.md)*
