# Lesson 1: Introduction to Control Systems

## Learning Objectives

- Define what a control system is and why it is needed
- Distinguish between open-loop and closed-loop (feedback) control
- Identify the components of a feedback control system
- Understand the role of feedback in improving performance and robustness
- Survey applications of control theory across engineering and science

## 1. What Is a Control System?

A **control system** is an interconnection of components that manages, commands, or regulates the behavior of a dynamic system to achieve a desired output.

**Examples from everyday life:**
- Thermostat regulating room temperature
- Cruise control maintaining car speed
- Human hand reaching for an object (biological feedback)
- Autopilot keeping an aircraft on course

The central question of control theory: *Given a system (plant) that we want to behave in a certain way, how do we design a controller to achieve that behavior?*

## 2. Open-Loop vs. Closed-Loop Control

### 2.1 Open-Loop Control

An **open-loop** system applies a pre-computed input without measuring the actual output.

```
Reference → [Controller] → [Plant] → Output
```

**Example:** A toaster with a timer. You set the time, but the toaster doesn't check whether the bread is actually toasted.

**Characteristics:**
- Simple and inexpensive
- Cannot compensate for disturbances or model errors
- Requires accurate knowledge of the plant
- No stability issues from feedback

### 2.2 Closed-Loop (Feedback) Control

A **closed-loop** system measures the actual output and uses the **error** (difference between desired and actual) to adjust the input.

```
Reference →(+)→ [Controller] → [Plant] → Output
             ↑                           |
             └──── [Sensor] ←───────────┘
                    (feedback)
```

**Example:** A thermostat. It measures the current temperature, compares it to the setpoint, and turns the heater on or off accordingly.

**Characteristics:**
- Compensates for disturbances and model uncertainty
- Reduces sensitivity to parameter variations
- Can introduce instability if poorly designed
- More complex than open-loop

### 2.3 Comparison

| Property | Open-Loop | Closed-Loop |
|----------|-----------|-------------|
| Disturbance rejection | Poor | Good |
| Sensitivity to parameters | High | Low |
| Complexity | Low | Higher |
| Stability risk | None | Possible |
| Accuracy | Depends on model | Self-correcting |
| Cost | Lower | Higher |

## 3. Components of a Feedback Control System

### 3.1 Standard Block Diagram

```
         e(t)        u(t)         y(t)
r(t) →(+)→ [Controller] → [Plant] → Output
        ↑    (G_c)          (G_p)      |
        |                              |
        └────── [Sensor (H)] ←────────┘
                  b(t)
```

| Symbol | Name | Description |
|--------|------|-------------|
| $r(t)$ | Reference (setpoint) | Desired output |
| $e(t)$ | Error signal | $e(t) = r(t) - b(t)$ |
| $u(t)$ | Control signal (actuator input) | Output of controller |
| $y(t)$ | Plant output | Actual system output |
| $b(t)$ | Feedback signal | Measured output (possibly scaled) |
| $G_c$ | Controller | Processes error to generate control action |
| $G_p$ | Plant (process) | The system being controlled |
| $H$ | Sensor (feedback element) | Measures the output |

### 3.2 Disturbances and Noise

In practice, systems are affected by:
- **Disturbances** $d(t)$: External inputs that perturb the plant (wind, load changes)
- **Measurement noise** $n(t)$: Sensor imperfections that corrupt the feedback signal

```
                    d(t)
                     ↓
r(t) →(+)→ [G_c] →(+)→ [G_p] →(+)→ y(t)
        ↑                         ↑ n(t)
        └──── [H] ←──────────────┘
```

A well-designed controller rejects disturbances while not amplifying noise excessively.

## 4. Control System Performance Objectives

A controller is designed to satisfy multiple (often competing) objectives:

### 4.1 Stability

The system must be **stable**: bounded inputs produce bounded outputs, and the system returns to equilibrium after perturbation. This is the *most fundamental requirement* — an unstable system is useless and potentially dangerous.

### 4.2 Tracking

The output $y(t)$ should follow the reference $r(t)$ as closely as possible:
- **Zero steady-state error** for common reference signals (step, ramp)
- **Fast transient response** (short rise time, settling time)
- **Minimal overshoot**

### 4.3 Disturbance Rejection

The controller should minimize the effect of external disturbances on the output.

### 4.4 Robustness

The system should perform well despite:
- Uncertainty in plant parameters
- Unmodeled dynamics
- Variations in operating conditions

### 4.5 Sensitivity

**Sensitivity** measures how much the system behavior changes when parameters change. Feedback reduces sensitivity — this is one of its primary advantages.

The **sensitivity function**:

$$S(s) = \frac{1}{1 + G_c(s) G_p(s) H(s)}$$

When the **loop gain** $L(s) = G_c(s) G_p(s) H(s)$ is large, $S(s) \approx 0$ and the output is insensitive to plant variations.

## 5. Historical Context

| Year | Milestone |
|------|-----------|
| ~270 BC | Ctesibius: water clock with float regulator |
| 1788 | Watt: centrifugal governor for steam engines |
| 1868 | Maxwell: "On Governors" — first mathematical stability analysis |
| 1922 | Minorsky: PID control for ship steering |
| 1932 | Nyquist: frequency-domain stability criterion |
| 1938 | Bode: logarithmic frequency plots |
| 1948 | Evans: root locus method |
| 1960 | Kalman: state-space methods, LQR, Kalman filter |
| 1970s+ | Robust control ($H_\infty$), adaptive control, nonlinear control |
| 2000s+ | Model predictive control (MPC), learning-based control |

## 6. Classification of Control Systems

### 6.1 By Signal Type

- **Continuous-time**: Signals and systems operate in continuous time $t$
- **Discrete-time**: Signals sampled at intervals $T$ (digital control)

### 6.2 By Linearity

- **Linear**: Superposition holds — the overwhelming focus of classical and modern control theory
- **Nonlinear**: No superposition — requires specialized methods (Lyapunov, sliding mode, feedback linearization)

### 6.3 By Number of Inputs/Outputs

- **SISO**: Single-input, single-output (classical control)
- **MIMO**: Multiple-input, multiple-output (modern state-space methods)

### 6.4 By Time Variation

- **Time-invariant (LTI)**: Parameters do not change over time
- **Time-varying**: Parameters change — e.g., rocket losing fuel mass

## 7. Mathematical Preliminaries Review

Control theory builds on concepts from differential equations and Laplace transforms. A brief review of the most essential ones:

### 7.1 Linear ODE

A linear, time-invariant (LTI) system is described by:

$$a_n \frac{d^n y}{dt^n} + a_{n-1} \frac{d^{n-1} y}{dt^{n-1}} + \cdots + a_1 \frac{dy}{dt} + a_0 y = b_m \frac{d^m u}{dt^m} + \cdots + b_0 u$$

### 7.2 Laplace Transform

The Laplace transform converts differential equations into algebraic equations:

$$\mathcal{L}\{f(t)\} = F(s) = \int_0^\infty f(t) e^{-st} \, dt$$

Key property: $\mathcal{L}\left\{\frac{d^n f}{dt^n}\right\} = s^n F(s) - s^{n-1}f(0) - \cdots - f^{(n-1)}(0)$

With zero initial conditions, differentiation becomes multiplication by $s$, making LTI system analysis purely algebraic.

### 7.3 Transfer Function (Preview)

The **transfer function** is the ratio of the Laplace-transformed output to input (zero initial conditions):

$$G(s) = \frac{Y(s)}{U(s)}$$

This concept is central to everything in Lessons 3-10 and will be developed fully in Lesson 3.

## Practice Exercises

### Exercise 1: System Identification

For each system, identify the reference input, plant, sensor, disturbance, and controller:

1. An air conditioning system maintaining room temperature at 22°C
2. A self-driving car maintaining a lane on a highway
3. A voltage regulator keeping the output of a power supply at 5V

### Exercise 2: Open-Loop vs. Closed-Loop

1. A washing machine runs for a fixed 30-minute cycle. Is this open-loop or closed-loop? What could make it closed-loop?
2. Explain why cruise control in a car cannot be purely open-loop on hilly terrain.
3. Give an example where open-loop control is preferred over closed-loop control.

### Exercise 3: Feedback Effects

Consider a system where the plant gain $G_p$ can vary between 8 and 12 (nominal = 10).

1. In open-loop, the controller gain is set to $G_c = 0.1$ to achieve unity overall gain. What is the range of the output gain?
2. In closed-loop (unity feedback $H = 1$) with $G_c = 100$, compute the closed-loop gain $\frac{G_c G_p}{1 + G_c G_p}$ for $G_p = 8, 10, 12$. How much does the output vary?
3. What is the cost of high loop gain for reducing sensitivity?

---

*Next: [Lesson 2 — Mathematical Modeling of Physical Systems](02_Mathematical_Modeling.md)*
