# Lesson 9: PID Control

## Learning Objectives

- Understand the structure and operation of PID controllers
- Describe the effect of each term (P, I, D) on system performance
- Tune PID controllers using classical methods (Ziegler-Nichols, Cohen-Coon)
- Recognize practical issues: integrator windup, derivative kick, noise amplification
- Apply PID control to common plant types

## 1. The PID Controller

The **PID controller** (Proportional-Integral-Derivative) is the most widely used controller in industry. It computes the control signal from the error $e(t) = r(t) - y(t)$:

$$u(t) = K_p e(t) + K_i \int_0^t e(\tau) \, d\tau + K_d \frac{de(t)}{dt}$$

**Transfer function (ideal form):**

$$G_c(s) = K_p + \frac{K_i}{s} + K_d s = K_p\left(1 + \frac{1}{T_i s} + T_d s\right)$$

where $T_i = K_p/K_i$ is the **integral time** and $T_d = K_d/K_p$ is the **derivative time**.

### 1.1 Why PID?

- Handles the three most important control tasks: tracking, disturbance rejection, and noise filtering
- Only 3 parameters to tune
- Works well for a wide variety of plants
- Over 90% of industrial controllers are PID (or PI)

## 2. Effect of Each Term

### 2.1 Proportional (P) Action

$$u_P(t) = K_p e(t)$$

- Output is proportional to the current error
- **Increases** gain → reduces steady-state error, speeds up response
- **But:** Cannot eliminate steady-state error for step disturbances (in Type 0 systems)
- Too much $K_p$ → instability (reduced phase margin)

### 2.2 Integral (I) Action

$$u_I(t) = K_i \int_0^t e(\tau) \, d\tau$$

- Accumulates past error
- **Eliminates steady-state error** for step inputs (adds a pole at $s = 0$, increasing system type)
- **But:** Adds phase lag → can destabilize the system
- Slow to respond to rapid changes

### 2.3 Derivative (D) Action

$$u_D(t) = K_d \frac{de(t)}{dt}$$

- Responds to the rate of change of error
- **Anticipates** future error → improves transient response, adds damping
- **But:** Amplifies high-frequency noise
- Has no effect on steady-state error

### 2.4 Summary Table

| Action | Effect on Rise Time | Effect on Overshoot | Effect on Settling Time | Steady-State Error |
|--------|-------------------|--------------------|------------------------|--------------------|
| Increase $K_p$ | Decrease | Increase | Small change | Decrease |
| Increase $K_i$ | Decrease | Increase | Increase | Eliminate |
| Increase $K_d$ | Small change | Decrease | Decrease | No effect |

**Caveat:** This table gives general trends. The actual effect depends on the specific plant and operating point.

## 3. Common PID Configurations

### 3.1 P Controller

$$G_c(s) = K_p$$

Simplest possible controller. Useful when the plant already has an integrator or when some steady-state error is acceptable.

### 3.2 PI Controller

$$G_c(s) = K_p\left(1 + \frac{1}{T_i s}\right) = K_p \frac{T_i s + 1}{T_i s}$$

The workhorse of industry. Zero steady-state error for step inputs with manageable complexity.

The PI controller adds a **zero** at $s = -1/T_i$ and a **pole** at $s = 0$. The zero partially compensates the phase lag introduced by the integrator.

### 3.3 PD Controller

$$G_c(s) = K_p(1 + T_d s)$$

Improves transient response without adding an integrator. Useful when steady-state error is handled by the plant itself (Type 1 or higher).

### 3.4 PID Controller

$$G_c(s) = K_p\left(1 + \frac{1}{T_i s} + T_d s\right) = K_p \frac{T_d T_i s^2 + T_i s + 1}{T_i s}$$

Combines all three actions. The numerator has two zeros that can be placed to improve both transient and steady-state performance.

## 4. PID Tuning Methods

### 4.1 Ziegler-Nichols: Open-Loop Method (Process Reaction Curve)

Apply a step input to the open-loop plant and measure the response:

```
y(t)
 ^            _______________
 |           /
 |          / ← tangent at steepest point
 |     ____/
 |    /
 |───┘
 └──────┬──┬─────────────→ t
        L   T
```

- $L$: **delay time** (from step to tangent line intersecting the time axis)
- $T$: **time constant** (tangent line to reach final value)
- $K_0$: plant DC gain

**Ziegler-Nichols open-loop tuning rules:**

| Controller | $K_p$ | $T_i$ | $T_d$ |
|-----------|-------|-------|-------|
| P | $T/(K_0 L)$ | — | — |
| PI | $0.9T/(K_0 L)$ | $L/0.3$ | — |
| PID | $1.2T/(K_0 L)$ | $2L$ | $0.5L$ |

### 4.2 Ziegler-Nichols: Closed-Loop Method (Ultimate Gain)

1. Set $K_i = 0$, $K_d = 0$ (P-only control)
2. Increase $K_p$ until the system exhibits sustained oscillation
3. Record the **ultimate gain** $K_u$ and **ultimate period** $T_u$

**Ziegler-Nichols closed-loop tuning rules:**

| Controller | $K_p$ | $T_i$ | $T_d$ |
|-----------|-------|-------|-------|
| P | $0.5K_u$ | — | — |
| PI | $0.45K_u$ | $T_u/1.2$ | — |
| PID | $0.6K_u$ | $T_u/2$ | $T_u/8$ |

**Note:** Ziegler-Nichols tuning typically gives aggressive settings with about 25% overshoot. Further refinement is usually needed.

### 4.3 Cohen-Coon Method

Uses the same open-loop measurements ($K_0$, $L$, $T$) but provides less aggressive tuning. Better for plants with large dead time.

### 4.4 Internal Model Control (IMC) Tuning

Based on the plant model $G_p(s)$, design the controller to approximately invert the plant:

For a first-order plus dead time (FOPDT) model $G_p = \frac{K_0 e^{-Ls}}{\tau s + 1}$:

$$K_p = \frac{\tau}{K_0(\lambda + L)}, \quad T_i = \tau, \quad T_d = 0$$

where $\lambda$ is the desired closed-loop time constant (the single tuning parameter).

- Larger $\lambda$ → slower but more robust
- Smaller $\lambda$ → faster but less robust

## 5. Practical PID Issues

### 5.1 Derivative Kick

When the setpoint changes abruptly, $de/dt$ produces a large spike (derivative kick). Solution: apply derivative action to the output only (**derivative on measurement**):

$$u_D = -K_d \frac{dy}{dt} \quad \text{instead of} \quad u_D = K_d \frac{de}{dt}$$

### 5.2 Integrator Windup

When the actuator saturates (reaches its physical limit), the integral term continues accumulating error, causing large overshoot when the system returns to the linear region.

**Anti-windup solutions:**
- **Clamping**: Stop integrating when the output is saturated
- **Back-calculation**: Reduce the integral term based on the difference between the desired and actual actuator output
- **Conditional integration**: Only integrate when the error is small

### 5.3 Derivative Noise Amplification

Pure differentiation amplifies high-frequency noise. Use a **filtered derivative**:

$$D(s) = \frac{K_d s}{1 + s/(N\omega_c)}$$

where $N$ is typically 5-20. This limits the derivative gain at high frequencies.

### 5.4 Practical PID Form

Combining all practical modifications:

$$G_c(s) = K_p\left(1 + \frac{1}{T_i s}\right) - K_d \frac{s}{1 + s/N_f} \cdot Y(s)/E(s)$$

with anti-windup on the integrator.

## 6. PID Design Example

**Plant:** DC motor with $G_p(s) = \frac{10}{s(s+5)}$ (Type 1, no steady-state error for step)

**Requirement:** Zero overshoot, settling time $< 2$ s, zero ramp error.

**Design:**
1. PI controller needed (Type 2 for zero ramp error): $G_c(s) = K_p(1 + 1/(T_i s))$
2. Adding derivative for damping: PID with $G_c(s) = K_p(1 + 1/(T_i s) + T_d s)$

**Ziegler-Nichols (closed-loop):** With P-only, find $K_u$ from Routh: $s^2 + 5s + 10K_p = 0$ → $K_u = \infty$ (Type 1 is always stable with P-only). This method doesn't directly apply here.

**Alternative (pole placement):** Choose desired closed-loop poles, then solve for PID parameters — a more systematic approach covered in the state-space lessons.

## Practice Exercises

### Exercise 1: PID Effect Analysis

For a unity-feedback system with plant $G_p(s) = \frac{1}{s+1}$:

1. With P control ($K_p = 10$), find the closed-loop transfer function, steady-state step error, and $M_p$
2. Add integral action ($T_i = 2$) and repeat
3. Add derivative action ($T_d = 0.1$) and analyze the effect on overshoot

### Exercise 2: Ziegler-Nichols Tuning

A plant's step response shows: delay $L = 0.5$ s, time constant $T = 3$ s, DC gain $K_0 = 2$.

1. Compute PID parameters using the Ziegler-Nichols open-loop method
2. Compute PI parameters using the same method
3. Using the IMC method with $\lambda = 1$ s, compute PI parameters and compare

### Exercise 3: Anti-Windup

Explain why a PI controller with $K_p = 5$, $K_i = 10$ controlling a plant with actuator saturation at $\pm 1$ will exhibit windup when tracking a step of magnitude 2. Describe how back-calculation anti-windup would help.

---

*Previous: [Lesson 8 — Nyquist Stability Criterion](08_Nyquist_Stability.md) | Next: [Lesson 10 — Lead-Lag Compensation](10_Lead_Lag_Compensation.md)*
