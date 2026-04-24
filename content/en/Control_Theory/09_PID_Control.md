# Lesson 9: PID Control

## Learning Objectives

- Understand the structure and operation of PID controllers
- Describe the effect of each term (P, I, D) on system performance
- Tune PID controllers using classical methods (Ziegler-Nichols, Cohen-Coon, IMC)
- Recognize practical issues: integrator windup, derivative kick, noise amplification
- Apply PID control to common plant types
- Implement a PID controller in code with sane defaults and simulate its behavior

## 0. Why PID Dominates — A Short History

If a plant designer needs a controller today, the odds they pick PID are roughly 9 in 10. That fact has held for more than seventy years, across petrochemical plants, process lines, aircraft autopilots, consumer thermostats, and quadcopter flight stacks. Three things explain this dominance:

- **It is simple enough to tune without a model.** A technician can get a usable loop with a stopwatch, a step change, and a chart recorder. Modern methods (state-space, $H_\infty$) almost always require a model of the plant; PID tolerates surprising amounts of ignorance.
- **Three knobs are enough for most first-principles plants.** Any plant with dominant first- or second-order dynamics can be stabilized and made to track a reference with three gains. More sophisticated dynamics (dead time, resonance) stretch the formula, but the three-knob intuition carries a long way.
- **Every practicing engineer already knows it.** In a field where operators must diagnose live failures at 3 a.m., a controller whose behavior is understood by everyone on-call is a safety feature.

The limitations are real — PID is mediocre on highly nonlinear plants, plants with large dead times, and MIMO systems with strong interactions — but those limitations are outliers in most industries.

Keep a concrete analogy in your head as you read:

- **Proportional** is a spring pushing the system toward the setpoint, proportional to how far off it is.
- **Integral** is a ratchet that accumulates past errors and keeps pushing even when the error is small.
- **Derivative** is a shock absorber that resists rapid changes.

A car suspension has all three; so does a PID loop. The correspondence is not coincidental — both are linear feedback systems with first-principles physics.

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

### 2.5 Worked Example: P-Only on a First-Order Plant

Consider plant $G_p(s) = 2/(s+1)$ — DC gain 2, time constant 1 s. Close the loop with a P controller:

$$\frac{Y(s)}{R(s)} = \frac{K_p \cdot 2}{(s+1) + K_p \cdot 2} = \frac{2K_p}{s + (1 + 2K_p)}$$

The closed-loop is still first-order, with time constant $1/(1+2K_p)$ and DC gain $2K_p/(1+2K_p)$.

| $K_p$ | Closed-loop DC gain | Closed-loop $\tau$ | Steady-state step error |
|-------|--------------------|--------------------|-------------------------|
| $0.5$ | $0.50$ | $0.50\,\text{s}$ | $50\%$ |
| $2$ | $0.80$ | $0.20\,\text{s}$ | $20\%$ |
| $10$ | $0.95$ | $0.048\,\text{s}$ | $5\%$ |
| $100$ | $0.995$ | $0.005\,\text{s}$ | $0.5\%$ |

Two lessons from this table:

1. Raising $K_p$ drives the steady-state error toward zero but never reaches it — this is why pure P leaves offset in Type 0 plants.
2. Raising $K_p$ also speeds the response (smaller $\tau$). For a first-order plant that is essentially free; for anything higher-order, huge $K_p$ destabilizes the loop. The trade-off only surfaces once order $\geq 2$.

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

### 4.5 A Guided Tuning Walkthrough

Consider a FOPDT plant identified from a step response: $K_0 = 1.5$, $T = 4\,\text{s}$, $L = 0.8\,\text{s}$.

**Ziegler-Nichols PID (open-loop):**

$$K_p = \frac{1.2 \cdot 4}{1.5 \cdot 0.8} = 4.0, \quad T_i = 2 \cdot 0.8 = 1.6\,\text{s}, \quad T_d = 0.5 \cdot 0.8 = 0.4\,\text{s}$$

Expected behavior: about 25% overshoot, aggressive — good starting point, not a finished design.

**IMC-PI tuning with $\lambda = 1\,\text{s}$:**

$$K_p = \frac{4}{1.5 \cdot (1 + 0.8)} = 1.48, \quad T_i = 4\,\text{s}, \quad T_d = 0$$

Much more conservative: lower gain, no derivative action. Expected to be smooth with minimal overshoot but noticeably slower than Z-N.

**IMC-PI with $\lambda = 0.4\,\text{s}$:** $K_p = \frac{4}{1.5 \cdot 1.2} = 2.22$. Roughly midway between Z-N and conservative IMC.

Which is "right"? The answer is the set that meets your specs with the robustness margin you can live with. Simulate all three; pick the one whose trade-off matches the application.

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

### 5.5 Reference Implementation

The practical form in code, compact enough to drop into a real system:

```python
class PID:
    """Discrete-time PID with filtered derivative, derivative-on-measurement,
    and back-calculation anti-windup. One call per sample period."""

    def __init__(self, kp, ki, kd, dt,
                 u_min=-float("inf"), u_max=float("inf"),
                 n_filter=10.0, kt=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.u_min, self.u_max = u_min, u_max
        self.n = n_filter                       # derivative filter strength
        self.kt = kt if kt is not None else max(ki, 1e-6)   # back-calc gain
        self.integral = 0.0
        self.prev_y = None
        self.prev_d = 0.0

    def step(self, r, y):
        e = r - y

        # Proportional
        p = self.kp * e

        # Derivative on measurement (avoids derivative kick on setpoint changes),
        # with first-order filter so high-frequency noise cannot ring up the term.
        if self.prev_y is None:
            self.prev_y = y
        dy = (y - self.prev_y) / self.dt
        d = self.prev_d + self.n * self.dt * (-self.kd * dy - self.prev_d)
        self.prev_d = d
        self.prev_y = y

        # Integral + back-calculation anti-windup: clamp the output, then use the
        # amount we clamped to pull the integral back toward the linear region.
        u_unclamped = p + self.integral + d
        u = max(self.u_min, min(self.u_max, u_unclamped))
        self.integral += self.dt * (self.ki * e + self.kt * (u - u_unclamped))

        return u
```

Every piece is motivated: derivative-on-measurement kills kick on setpoint steps; the filtered derivative ensures noise cannot amplify unboundedly; back-calculation feeds the saturation gap back to the integrator at gain $K_t$ so that when the actuator desaturates, the loop has not accumulated a large unusable $I$.

## 6. PID Design Example

**Plant:** DC motor with $G_p(s) = \frac{10}{s(s+5)}$ (Type 1, no steady-state error for step)

**Requirement:** Zero overshoot, settling time $< 2$ s, zero ramp error.

**Design:**
1. PI controller needed (Type 2 for zero ramp error): $G_c(s) = K_p(1 + 1/(T_i s))$
2. Adding derivative for damping: PID with $G_c(s) = K_p(1 + 1/(T_i s) + T_d s)$

**Ziegler-Nichols (closed-loop):** With P-only, find $K_u$ from Routh: $s^2 + 5s + 10K_p = 0$ → $K_u = \infty$ (Type 1 is always stable with P-only). This method doesn't directly apply here.

**Alternative (pole placement):** Choose desired closed-loop poles, then solve for PID parameters — a more systematic approach covered in the state-space lessons.

## 7. Simulation in Python

Paste the snippet below alongside the `PID` class from Section 5.5 and you have a complete tunable test harness:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def simulate(plant_num, plant_den, pid, t_end=10.0, dt=0.01, setpoint=1.0):
    """Closed-loop simulation with a PID controller and a scalar plant."""
    sys = signal.TransferFunction(plant_num, plant_den)
    ss = sys.to_ss()            # continuous state-space for the plant
    x = np.zeros(ss.A.shape[0])

    t_hist, y_hist, u_hist = [], [], []
    n_steps = int(t_end / dt)
    for k in range(n_steps):
        y = float((ss.C @ x)[0])
        u = pid.step(setpoint, y)

        # Simple forward-Euler step on the plant ẋ = A x + B u
        x = x + dt * (ss.A @ x + ss.B.flatten() * u)

        t_hist.append(k * dt)
        y_hist.append(y)
        u_hist.append(u)

    return np.array(t_hist), np.array(y_hist), np.array(u_hist)


# Plant: first-order with DC gain 2 and time constant 1 s
plant_num, plant_den = [2], [1, 1]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

for label, gains in [
    ("P",  (2.0, 0.0, 0.0)),
    ("PI", (2.0, 3.0, 0.0)),
    ("PID", (2.0, 3.0, 0.3)),
]:
    pid = PID(*gains, dt=0.01, u_min=-5, u_max=5)
    t, y, u = simulate(plant_num, plant_den, pid)
    ax1.plot(t, y, label=label)
    ax2.plot(t, u, label=label)

ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
ax1.set_ylabel("output y")
ax2.set_ylabel("control u")
ax2.set_xlabel("time [s]")
for ax in (ax1, ax2):
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.show()
```

Run it, then experiment: drop $K_p$ from 2 to 0.5 and watch the offset (P-only) reappear. Raise $K_i$ from 3 to 10 and watch the overshoot grow. Add $K_d = 0.3$ and the overshoot shrinks back. These three sliders are PID — the rest of the field is just knowing when each slider is lying to you.

## 8. Common Pitfalls

1. **Tuning on the wrong metric.** An engineer optimizing for 5% overshoot on a step may inadvertently degrade disturbance rejection. Always measure both the setpoint-to-output and disturbance-to-output responses.
2. **Forgetting to include measurement dynamics.** A thermocouple with a 2-second lag is part of the plant for PID design purposes, even if it is a sensor, not the process itself. Leaving it out of the model produces tuning that is too aggressive.
3. **Using PID on a plant with integrating dynamics and expecting I to "help".** A plant of the form $1/s$ already rejects step errors with P alone; adding I produces a double integrator that is marginally stable. Recognize when your plant is already Type 1 or higher.
4. **Derivative on error without filtering.** On real hardware this is almost always a bug — the ADC and sensor noise pour straight into the actuator through $K_d$. Always filter, always consider derivative-on-measurement instead.
5. **Blindly applying Ziegler-Nichols.** Z-N is a starting point, not a final answer. Its 25%-overshoot design choice was appropriate for 1940s process control; modern applications usually want tighter overshoot, so Z-N gains are typically halved or detuned after the first simulation.
6. **Sample-time mismatch.** The discrete-time PID in Section 5.5 assumes the `step()` call happens every `dt` seconds. Jitter of 20%+ in that cadence — common in non-RTOS systems — silently detunes the loop. Either run on a real-time scheduler or re-derive gains using the actual measured sample period.
7. **Anti-windup applied to the wrong signal.** Back-calculation reduces the integrator based on the difference between the saturated and unsaturated controller outputs. Clamping without any feedback path leaves the integrator frozen at its saturation value — when the setpoint finally becomes reachable, the integrator still has an enormous stored value.

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

### Exercise 4: Derivative Filtering

Given a PID with $K_p = 2$, $K_i = 1$, $K_d = 0.5$, and a sensor with white noise of standard deviation 0.01 at 100 Hz sample rate, compute the RMS control signal contribution from the derivative term for an unfiltered $D(s) = K_d s$ versus a filtered $D(s) = K_d s / (1 + s/N)$ with $N = 10$.

### Exercise 5: Simulation

Extend the Python harness in Section 7 to a second-order plant $G_p(s) = 4 / (s^2 + 2s + 4)$. Tune a PID using Z-N closed-loop (find $K_u$ by sweep, then compute $T_u$ from the oscillation period) and compare the resulting response to an IMC-PI design.

---

*Previous: [Lesson 8 — Nyquist Stability Criterion](08_Nyquist_Stability.md) | Next: [Lesson 10 — Lead-Lag Compensation](10_Lead_Lag_Compensation.md)*
