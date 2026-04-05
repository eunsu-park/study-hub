# Lesson 4: Time-Domain Analysis

## Learning Objectives

- Compute and interpret the step response of first-order and second-order systems
- Define and calculate time-domain specifications (rise time, settling time, overshoot, steady-state error)
- Relate second-order system parameters ($\zeta$, $\omega_n$) to time-domain performance
- Analyze steady-state error using the final value theorem and system type
- Apply error constants ($K_p$, $K_v$, $K_a$) to determine tracking accuracy

## 1. Standard Test Signals

Control engineers analyze system performance using standard inputs:

| Signal | Time Domain | Laplace Transform |
|--------|------------|-------------------|
| Impulse $\delta(t)$ | $\delta(t)$ | $1$ |
| Step $u(t)$ | $u(t)$ | $\frac{1}{s}$ |
| Ramp $tu(t)$ | $tu(t)$ | $\frac{1}{s^2}$ |
| Parabola $\frac{1}{2}t^2 u(t)$ | $\frac{1}{2}t^2 u(t)$ | $\frac{1}{s^3}$ |

The **step response** is the most commonly used test because it reveals both transient and steady-state behavior.

## 2. First-Order System Response

$$G(s) = \frac{K}{\tau s + 1}$$

### 2.1 Step Response

For a unit step input $R(s) = 1/s$:

$$Y(s) = \frac{K}{s(\tau s + 1)}$$

$$y(t) = K(1 - e^{-t/\tau})$$

**Key characteristics:**
- Final value: $K$ (the DC gain)
- At $t = \tau$: $y(\tau) = K(1 - e^{-1}) = 0.632K$ (63.2% of final value)
- At $t = 4\tau$: $y = 0.982K$ (98.2% — essentially settled)
- No overshoot, no oscillation
- Rise time (10% to 90%): $t_r = 2.2\tau$

### 2.2 Impulse Response

$$y(t) = \frac{K}{\tau}e^{-t/\tau}$$

The impulse response decays exponentially with the same time constant.

## 3. Second-Order System Response

$$G(s) = \frac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}$$

The step response depends critically on the damping ratio $\zeta$:

### 3.1 Underdamped Case ($0 < \zeta < 1$)

Poles: $s = -\sigma \pm j\omega_d$ where $\sigma = \zeta\omega_n$ and $\omega_d = \omega_n\sqrt{1-\zeta^2}$

$$y(t) = 1 - \frac{e^{-\sigma t}}{\sqrt{1-\zeta^2}}\sin(\omega_d t + \phi)$$

where $\phi = \cos^{-1}\zeta$.

This is the most common and interesting case — the response oscillates while approaching the final value.

### 3.2 Critically Damped Case ($\zeta = 1$)

Poles: $s = -\omega_n$ (repeated)

$$y(t) = 1 - (1 + \omega_n t)e^{-\omega_n t}$$

Fastest non-oscillatory response.

### 3.3 Overdamped Case ($\zeta > 1$)

Poles: $s = -\zeta\omega_n \pm \omega_n\sqrt{\zeta^2 - 1}$ (two distinct real negative)

$$y(t) = 1 + \frac{1}{2}\left(\frac{e^{s_1 t}}{s_1/\omega_n^2} + \frac{e^{s_2 t}}{s_2/\omega_n^2}\right)$$

Slower than critically damped, no oscillation.

## 4. Time-Domain Specifications

For the step response of a system with final value $y_{\text{final}}$:

```
y(t)
 ^
 |        M_p
 |    ┌────*────┐
 |   /  ╲      / ╲
 | /     ╲────     ───── y_final ──────
 |/            (within ±2% or ±5%)
 |
 ├──┤  ├──────────────────┤
 0  t_r       t_s                    t →
```

| Specification | Symbol | Definition |
|--------------|--------|------------|
| **Rise time** | $t_r$ | Time to go from 10% to 90% of final value |
| **Peak time** | $t_p$ | Time to first peak |
| **Maximum overshoot** | $M_p$ | $\frac{y_{\max} - y_{\text{final}}}{y_{\text{final}}} \times 100\%$ |
| **Settling time** | $t_s$ | Time to stay within $\pm 2\%$ (or $\pm 5\%$) of final value |
| **Steady-state error** | $e_{ss}$ | $\lim_{t\to\infty} [r(t) - y(t)]$ |

### 4.1 Formulas for Second-Order Underdamped Systems

$$t_r \approx \frac{1.8}{\omega_n} \quad \text{(approximate, for } 0.3 < \zeta < 0.8\text{)}$$

$$t_p = \frac{\pi}{\omega_d} = \frac{\pi}{\omega_n\sqrt{1-\zeta^2}}$$

$$M_p = e^{-\pi\zeta/\sqrt{1-\zeta^2}} \times 100\%$$

$$t_s \approx \frac{4}{\zeta\omega_n} \quad \text{(2% criterion)} \qquad t_s \approx \frac{3}{\zeta\omega_n} \quad \text{(5% criterion)}$$

### 4.2 Design Implications

These formulas reveal fundamental trade-offs:
- **Faster response** (larger $\omega_n$) $\Rightarrow$ smaller $t_r$, $t_p$, $t_s$, but requires more control effort
- **Less overshoot** (larger $\zeta$) $\Rightarrow$ smaller $M_p$, but slower response (larger $t_r$)
- $t_s$ depends on $\sigma = \zeta\omega_n$ — to reduce settling time, increase both $\zeta$ and $\omega_n$

**Typical design targets:** $\zeta \approx 0.4\text{–}0.8$ balances speed and overshoot.

### 4.3 Pole Placement Perspective

The second-order specifications map directly to regions in the $s$-plane:

- $t_s \leq T_s$: poles must satisfy $\text{Re}(s) \leq -4/T_s$ (left of a vertical line)
- $M_p \leq M$: poles must satisfy $\zeta \geq \zeta_{\min}$ (inside a wedge from origin)
- $t_p \leq T_p$: poles must satisfy $\omega_d \geq \pi/T_p$ (above a horizontal line)

The feasible region is the **intersection** of these constraints.

## 5. Effects of Additional Poles and Zeros

### 5.1 Additional Poles

A third pole at $s = -p_3$ adds a slower component:
- If $|p_3| \gg \sigma$: negligible effect (pole is "fast" relative to dominant pair)
- If $|p_3| \approx \sigma$: significantly increases rise time and settling time
- Rule of thumb: if $|p_3| > 5\sigma$, the third pole can be ignored

### 5.2 Additional Zeros

A zero at $s = -z$ affects overshoot:
- **LHP zero close to dominant poles**: increases overshoot and speeds up response
- **LHP zero far from dominant poles**: negligible effect
- **RHP zero** ($s = +z$): causes initial undershoot (non-minimum phase behavior)

## 6. Steady-State Error Analysis

### 6.1 Final Value Theorem

If $Y(s)$ has all poles in the left half-plane (except possibly at $s = 0$):

$$y(\infty) = \lim_{s \to 0} sY(s)$$

For a unity-feedback system with open-loop transfer function $G(s)$:

$$e_{ss} = \lim_{s \to 0} sE(s) = \lim_{s \to 0} \frac{sR(s)}{1 + G(s)}$$

### 6.2 System Type

The **system type** is the number of free integrators (poles at $s = 0$) in the open-loop transfer function:

$$G(s) = \frac{K \prod(s - z_i)}{s^N \prod(s - p_j)} \quad \Rightarrow \quad \text{Type } N$$

### 6.3 Error Constants and Steady-State Error

| Input | Error Constant | Formula | Type 0 | Type 1 | Type 2 |
|-------|---------------|---------|--------|--------|--------|
| Step $1/s$ | $K_p = \lim_{s\to 0} G(s)$ | $\frac{1}{1+K_p}$ | $\frac{1}{1+K_p}$ | $0$ | $0$ |
| Ramp $1/s^2$ | $K_v = \lim_{s\to 0} sG(s)$ | $\frac{1}{K_v}$ | $\infty$ | $\frac{1}{K_v}$ | $0$ |
| Parabola $1/s^3$ | $K_a = \lim_{s\to 0} s^2 G(s)$ | $\frac{1}{K_a}$ | $\infty$ | $\infty$ | $\frac{1}{K_a}$ |

**Key insight:** Each integrator in the loop eliminates steady-state error for one more level of input complexity. But integrators also affect stability (add phase lag), so there is a trade-off.

### 6.4 Example

Given $G(s) = \frac{100}{s(s+5)}$ (Type 1 system):

- $K_p = \lim_{s\to 0} G(s) = \infty$ → zero step error
- $K_v = \lim_{s\to 0} sG(s) = 100/5 = 20$ → ramp error = $1/20 = 5\%$
- $K_a = \lim_{s\to 0} s^2 G(s) = 0$ → infinite parabolic error

## Practice Exercises

### Exercise 1: Second-Order Specifications

A unity-feedback system has the open-loop transfer function:

$$G(s) = \frac{50}{s(s+5)}$$

1. Find the closed-loop transfer function
2. Identify $\omega_n$ and $\zeta$
3. Compute $M_p$, $t_p$, and $t_s$ (2% criterion)
4. Compute the steady-state error for a unit ramp input

### Exercise 2: Dominant Poles

A system has closed-loop poles at $s = -2 \pm j3$ and $s = -20$.

1. Can the third pole be neglected? Justify.
2. Estimate $M_p$ and $t_s$ using the dominant second-order approximation.

### Exercise 3: System Type Design

Design a controller $G_c(s) = K(s+a)/s$ such that the system with plant $G_p(s) = 1/(s+2)$ has:
- Zero steady-state error for a step input
- Steady-state error $\leq 0.02$ for a unit ramp input

What is the minimum value of $K$ required?

---

*Previous: [Lesson 3 — Transfer Functions and Block Diagrams](03_Transfer_Functions_and_Block_Diagrams.md) | Next: [Lesson 5 — Stability Analysis](05_Stability_Analysis.md)*
