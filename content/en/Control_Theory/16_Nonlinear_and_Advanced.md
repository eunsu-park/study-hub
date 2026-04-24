# Lesson 16: Nonlinear Control and Advanced Topics

## Learning Objectives

- Understand why nonlinear control is necessary and where linear methods fail
- Apply Lyapunov's stability theory to analyze nonlinear systems
- Use the describing function method to predict limit cycles
- Understand sliding mode control principles
- Survey model predictive control (MPC) and adaptive control concepts
- Place each advanced method in the larger control landscape and recognize when each is the right tool

## 0. Where Linear Tools Stop Working

Lessons 1–15 lived almost entirely in linear-time-invariant land. That is not because the world is linear — it isn't — but because near an operating point, most plants are linear ENOUGH that linear methods produce useful answers. This lesson is about the cases where "near an operating point" is not good enough.

Three concrete signs that linear methods are about to fail:

- **The operating envelope is wide.** A pendulum linearized around 0° is fine for ±5°. At ±90°, $\sin\theta \approx \theta$ underestimates the restoring force by a factor of 0.6 — the linear model lies. Aircraft, robots, and chemical reactors all routinely operate over ranges where linearization is suspect.
- **The nonlinearity is the point.** Saturation, dead zone, relay, hysteresis, friction — these are not bugs to be linearized away. They are essential features of every real actuator and many sensors. A controller that ignores them will hit them in production.
- **You need GLOBAL guarantees.** "Locally stable" is the answer linear analysis gives. "Stable from any initial condition in a half-meter ball around the equilibrium" requires Lyapunov-style arguments — Section 2 below.

This lesson is necessarily a survey, not a complete treatment. Each section below introduces enough to recognize when the technique applies; full mastery is a graduate course per topic.

## 1. Why Nonlinear Control?

All real systems are nonlinear. Linearization (Lesson 2) works near an operating point, but fails when:

- The system operates over a **wide range** (pendulum at large angles, not just small perturbations)
- The nonlinearity is **essential** (saturation, dead zone, relay, backlash, hysteresis)
- **Global** stability guarantees are needed (not just local near an equilibrium)
- The system has **multiple equilibria** (which one does the state converge to?)

### 1.1 Common Nonlinearities

| Nonlinearity | Description | Effect |
|-------------|-------------|--------|
| Saturation | Output limited to $\pm u_{\max}$ | Limits control authority, windup |
| Dead zone | No output for $|u| < \delta$ | Steady-state error, limit cycles |
| Relay (on-off) | Binary output ($\pm M$) | Chattering, limit cycles |
| Backlash | Hysteresis in mechanical coupling | Phase lag, oscillation |
| Coulomb friction | Constant friction opposing motion | Stick-slip, limit cycles |
| Quantization | Discrete output levels | Limit cycles around setpoint |

### 1.2 Nonlinear Phenomena

Behaviors that **cannot occur** in linear systems but are common in nonlinear ones:
- **Limit cycles**: Sustained oscillation independent of initial conditions
- **Bifurcations**: Qualitative change in behavior as a parameter varies
- **Chaos**: Sensitive dependence on initial conditions
- **Multiple equilibria**: Different steady states depending on initial conditions
- **Finite escape time**: State goes to infinity in finite time

## 2. Lyapunov Stability Theory

### 2.1 Concept

Aleksandr Lyapunov (1892) developed a method to prove stability **without solving** the differential equations. The idea: if we can find an "energy-like" function that always decreases, the system must be approaching equilibrium.

### 2.2 Lyapunov's Direct Method

Consider the autonomous system $\dot{x} = f(x)$ with equilibrium at $x = 0$.

**Definition:** A function $V(x): \mathbb{R}^n \to \mathbb{R}$ is a **Lyapunov function** if:
1. $V(0) = 0$
2. $V(x) > 0$ for all $x \neq 0$ (positive definite)
3. $\dot{V}(x) = \nabla V \cdot f(x) \leq 0$ along trajectories

**Theorem (Lyapunov):**

| Condition on $\dot{V}$ | Conclusion |
|------------------------|------------|
| $\dot{V}(x) \leq 0$ (negative semidefinite) | Stable (in the sense of Lyapunov) |
| $\dot{V}(x) < 0$ for $x \neq 0$ (negative definite) | Asymptotically stable |
| $\dot{V}(x) \leq -\alpha V(x)$ for some $\alpha > 0$ | Exponentially stable |

If these conditions hold for all $x$ (not just near the origin), the stability is **global**.

### 2.3 Example: Damped Pendulum

$$\ddot{\theta} + \frac{b}{ml^2}\dot{\theta} + \frac{g}{l}\sin\theta = 0$$

State: $x_1 = \theta$, $x_2 = \dot{\theta}$. Equilibrium: $x = 0$.

**Candidate Lyapunov function** (total energy):

$$V(x) = \frac{1}{2}ml^2 x_2^2 + mgl(1 - \cos x_1) > 0 \quad (x \neq 0)$$

$$\dot{V} = ml^2 x_2 \dot{x}_2 + mgl(\sin x_1)\dot{x}_1 = ml^2 x_2\left(-\frac{b}{ml^2}x_2 - \frac{g}{l}\sin x_1\right) + mgl x_2 \sin x_1$$

$$= -bx_2^2 \leq 0$$

Since $\dot{V} = 0$ only when $x_2 = 0$ (and by LaSalle's invariance principle, $x_1 = 0$ is the only invariant set where $x_2 = 0$), the origin is **globally asymptotically stable**.

### 2.4 Finding Lyapunov Functions

There is no general algorithm for finding Lyapunov functions, but common approaches include:
- **Energy method**: Use physical energy (kinetic + potential)
- **Quadratic form**: $V(x) = x^T P x$ where $P > 0$ — works for linear systems ($\dot{V} < 0$ iff $A^T P + PA < 0$, the Lyapunov equation)
- **Variable gradient method**: Assume $\nabla V = M(x)x$ and solve for $M$
- **SOS (Sum of Squares)**: Computational method for polynomial systems — search for $V$ using semidefinite programming

### 2.5 Lyapunov for Linear Systems

For $\dot{x} = Ax$ with $V = x^T P x$:

$$\dot{V} = x^T(A^T P + PA)x = -x^T Q x$$

The Lyapunov equation $A^T P + PA = -Q$ has a unique positive definite solution $P$ for any $Q > 0$ if and only if $A$ is stable (all eigenvalues in the LHP). This is an alternative stability test.

> **Why energy works**: in the pendulum example, $V$ is literally the kinetic + potential energy. Friction dissipates it. The argument generalizes — if you can identify ANY scalar quantity that decreases monotonically along trajectories, you have a Lyapunov function. The art is recognizing the right "energy" for non-mechanical systems.

## 3. Describing Function Analysis

### 3.1 Purpose

The **describing function** method predicts **limit cycles** (sustained oscillations) in nonlinear feedback systems.

### 3.2 Setup

```
     ┌────────┐      ┌────────┐
r=0 →(+)→│ N(a,ω) │→───→│ G(jω)  │→── y
     ↑   └────────┘      └────────┘  |
     └────────────────────────────────┘
```

where $N(a, \omega)$ is the **describing function** of the nonlinearity — the ratio of the fundamental harmonic of the output to a sinusoidal input of amplitude $a$.

### 3.3 Describing Function

For a sinusoidal input $x(t) = a\sin(\omega t)$ to a memoryless nonlinearity $y = n(x)$, the describing function is:

$$N(a) = \frac{1}{\pi a}\int_0^{2\pi} n(a\sin\theta) e^{j\theta} d\theta$$

For **symmetric** nonlinearities (odd function), $N(a)$ is **real**.

### 3.4 Common Describing Functions

| Nonlinearity | $N(a)$ |
|-------------|--------|
| Ideal relay ($\pm M$) | $\frac{4M}{\pi a}$ |
| Saturation (slope $k$, limit $M$) | $\frac{2k}{\pi}\left[\sin^{-1}\frac{M}{ka} + \frac{M}{ka}\sqrt{1-\left(\frac{M}{ka}\right)^2}\right]$ for $a > M/k$ |
| Dead zone (width $2\delta$) | $\frac{k}{\pi}\left[\pi - 2\sin^{-1}\frac{\delta}{a} - \frac{2\delta}{a}\sqrt{1-\left(\frac{\delta}{a}\right)^2}\right]$ for $a > \delta$ |

### 3.5 Predicting Limit Cycles

A limit cycle exists at amplitude $a$ and frequency $\omega$ if:

$$1 + N(a)G(j\omega) = 0 \quad \Rightarrow \quad G(j\omega) = -\frac{1}{N(a)}$$

**Graphically:** Plot $G(j\omega)$ and $-1/N(a)$ on the same Nyquist plane. Intersections predict limit cycles.

**Stability of the limit cycle:** If increasing $a$ moves $-1/N(a)$ **into** the region encircled by $G(j\omega)$, the limit cycle is **stable** (self-sustaining).

## 4. Sliding Mode Control

### 4.1 Concept

**Sliding mode control (SMC)** forces the system state onto a **sliding surface** $\sigma(x) = 0$ and keeps it there using high-frequency switching. On the sliding surface, the system dynamics are reduced in order and can be designed for desired behavior.

### 4.2 Design for Second-Order System

For $\ddot{y} = f(x) + bu$, define the sliding surface:

$$\sigma = \dot{e} + \lambda e$$

where $e = y - y_d$ is the tracking error and $\lambda > 0$ determines the convergence rate on the surface.

**Control law:**

$$u = \frac{1}{b}\left[-f(x) + \ddot{y}_d - \lambda\dot{e} - k\text{sign}(\sigma)\right]$$

where $k > 0$ is the switching gain. The $\text{sign}(\sigma)$ term drives the state toward $\sigma = 0$.

### 4.3 Reaching Condition

To ensure the state reaches the sliding surface:

$$\sigma\dot{\sigma} < 0$$

This is guaranteed if $k$ is large enough to overcome disturbances and uncertainties.

### 4.4 Chattering

The discontinuous $\text{sign}(\sigma)$ causes **chattering** — high-frequency oscillation around $\sigma = 0$. Mitigation approaches:
- **Boundary layer**: Replace $\text{sign}(\sigma)$ with $\text{sat}(\sigma/\phi)$ where $\phi$ is the layer width
- **Super-twisting algorithm**: Higher-order sliding mode that reduces chattering
- **Reaching law approach**: Use $\dot{\sigma} = -k|\sigma|^\alpha \text{sign}(\sigma)$ with $0 < \alpha < 1$

## 5. Model Predictive Control (MPC)

### 5.1 Concept

**MPC** solves an optimization problem at each time step:
1. **Predict** the future system behavior over a horizon $N$ using the model
2. **Optimize** a cost function over the predicted trajectory
3. **Apply** only the first control input
4. **Repeat** at the next time step (receding horizon)

### 5.2 Formulation

At each time step $k$, solve:

$$\min_{u[k], \ldots, u[k+N-1]} \sum_{i=0}^{N-1} \left[ x[k+i]^T Q x[k+i] + u[k+i]^T R u[k+i] \right] + x[k+N]^T P_f x[k+N]$$

subject to:
- $x[k+i+1] = Ax[k+i] + Bu[k+i]$ (model)
- $u_{\min} \leq u[k+i] \leq u_{\max}$ (input constraints)
- $x_{\min} \leq x[k+i] \leq x_{\max}$ (state constraints)

### 5.3 Advantages

- **Handles constraints explicitly** (actuator limits, safety bounds)
- **Systematic MIMO design**
- **Preview capability** (can incorporate known future references)
- **Optimal** (within the prediction horizon)

### 5.4 Challenges

- Computational cost (solving an optimization problem in real-time)
- Requires a good model
- Stability guarantees require careful design (terminal cost $P_f$, terminal constraints)
- Tuning the horizon $N$, weights $Q$, $R$, and constraints

### 5.5 MPC vs. LQR

| Feature | LQR | MPC |
|---------|-----|-----|
| Horizon | Infinite | Finite (receding) |
| Constraints | Cannot handle | Handles explicitly |
| Computation | Offline (solve ARE once) | Online (solve QP each step) |
| Optimality | Globally optimal (unconstrained) | Locally optimal (constrained) |
| MIMO | Yes | Yes |

As $N \to \infty$ with no constraints, MPC converges to LQR.

> **MPC's killer feature**: every other technique in this course handles constraints by ignoring them at design time and patching with anti-windup or saturation. MPC is the only one that puts constraints inside the optimization, which means the controller **knows** the actuator can saturate and chooses inputs that respect the limit. For high-performance industrial control with hard limits (chemical reactors, vehicle stability), MPC has become the default.

## 6. Adaptive Control

### 6.1 Motivation

When plant parameters are **unknown** or **time-varying**, a fixed controller may not perform well. Adaptive control adjusts the controller parameters in real-time.

### 6.2 Model Reference Adaptive Control (MRAC)

```
Reference ──→ [Reference Model] ──→ y_m (desired output)
    |                                      |
    └──→ [Adaptive Controller] ──→ [Plant] ──→ y (actual output)
              ↑                              |
              └── [Adaptation Law] ←── e = y - y_m
```

The adaptation law adjusts controller parameters to make $y(t) \to y_m(t)$, using Lyapunov-based design or MIT rule.

### 6.3 Self-Tuning Regulators

1. **Estimate** plant parameters online (using recursive least squares or similar)
2. **Design** a controller for the estimated plant (using pole placement, LQR, etc.)
3. **Apply** and repeat

This combines system identification with controller design in a closed loop.

## 7. Summary: Control Theory Landscape

```
                    Control Theory
                         │
         ┌───────────────┼───────────────┐
    Classical           Modern          Advanced
         │                │                │
    ├── Transfer fn   ├── State-space  ├── Nonlinear
    ├── Root locus    ├── Pole place.  ├── Lyapunov
    ├── Bode/Nyquist  ├── LQR/LQG     ├── Sliding mode
    ├── PID           ├── Kalman       ├── MPC
    └── Lead/lag      └── H∞ robust    ├── Adaptive
                                       └── Learning-based
```

**Key takeaway:** Start with the simplest method that works. PID handles the majority of industrial control problems. Use advanced methods when constraints, MIMO coupling, nonlinearities, or strict performance requirements demand them.

## 8. Common Pitfalls

1. **Trying Lyapunov on the wrong scalar.** Not every positive function works as $V$. The art is choosing one whose derivative along trajectories is negative — energy works for mechanical systems, but other domains need different choices (entropy for thermodynamic, Hamiltonian for conservative systems).
2. **Trusting the describing function on heavily-distorted output.** The describing function uses ONLY the fundamental harmonic. If the nonlinearity creates strong harmonics that the loop's $G(j\omega)$ does not filter, the prediction can be off — DF is reliable when $G(j\omega)$ is low-pass with a clear cutoff well below the second harmonic.
3. **Ignoring chattering in real SMC implementations.** "$\text{sign}(\sigma)$" implies infinite switching frequency, which no actuator delivers. Boundary-layer SMC trades a small steady-state error for finite switching — almost always the right trade in practice.
4. **MPC without recursive feasibility analysis.** A naive MPC formulation can hit a state from which no feasible input sequence exists in $N$ steps — the optimization just fails. Robust MPC and reachability tools (Sections 5.4) prevent this; skipping them is fine for benchtop demos and dangerous in production.
5. **Adaptive control as a magic answer to model uncertainty.** MRAC and self-tuners assume the structure of the plant is known and only the parameters are unknown. If the structure itself is wrong (e.g., you assumed second-order but the plant is third-order), adaptation can drive parameters in the wrong direction and destabilize the loop.
6. **Reaching for nonlinear methods too early.** A linear controller designed at the right operating point with sensible robustness margins handles 80%+ of "nonlinear" problems. Reach for SMC, MPC, or feedback linearization only when you have a clear reason that gain-scheduled linear control will not suffice.

## Practice Exercises

### Exercise 1: Lyapunov Analysis

For the Van der Pol oscillator:

$$\ddot{x} - \mu(1 - x^2)\dot{x} + x = 0$$

1. Find the equilibrium point
2. Linearize around the equilibrium and analyze local stability for $\mu > 0$
3. Try the Lyapunov function $V = \frac{1}{2}(x^2 + \dot{x}^2)$. Compute $\dot{V}$. Can you conclude stability?
4. What does the Van der Pol oscillator exhibit for $\mu > 0$? (Hint: limit cycle)

### Exercise 2: Describing Function

A feedback system has a relay nonlinearity ($\pm 1$) in series with $G(s) = \frac{10}{s(s+1)(s+2)}$.

1. Write the describing function for the relay
2. Find $-1/N(a)$ and sketch it on the complex plane
3. Find the intersection with $G(j\omega)$ to predict the limit cycle amplitude and frequency

### Exercise 3: MPC Concept

A discrete-time double integrator $x[k+1] = \begin{bmatrix} 1 & T \\ 0 & 1 \end{bmatrix} x[k] + \begin{bmatrix} T^2/2 \\ T \end{bmatrix} u[k]$ with $T = 0.1$ s and constraint $|u| \leq 1$:

1. With $Q = I$, $R = 0.1$, $N = 10$, formulate the MPC optimization problem
2. Without constraints, this is finite-horizon LQR — what changes when $|u| \leq 1$?
3. Explain why receding horizon (applying only $u[k]$ and re-solving) helps with disturbances

### Exercise 4: Pick the Right Tool

For each scenario, pick ONE of (linear gain-scheduled, Lyapunov-based nonlinear, SMC, MPC, adaptive) and justify in one sentence:

1. Quadcopter altitude hold under varying payload mass.
2. Chemical reactor with hard temperature limits and known dynamics.
3. Hard disk drive head positioning at nanometer accuracy with friction.
4. Aircraft autopilot crossing flight regimes from low-speed approach to high-speed cruise.
5. Inverted pendulum on a cart, swinging up from hanging-down position.

### Exercise 5: Linearization Failure Demonstration

Take the pendulum equation $\ddot{\theta} + (b/ml^2)\dot{\theta} + (g/l)\sin\theta = u$. Design a state-feedback gain that stabilizes the linearization around $\theta = 0$. Simulate the closed loop on the ORIGINAL nonlinear system from initial conditions $\theta_0 \in \{0.1, 0.5, 1.0, 2.0\}$ rad. At what initial angle does the linear controller fail (the pendulum either does not converge or oscillates wildly)? Discuss what nonlinear technique would handle the failing case.

---

*Previous: [Lesson 15 — Digital Control Systems](15_Digital_Control.md)*

*Return to: [Overview](00_Overview.md)*
