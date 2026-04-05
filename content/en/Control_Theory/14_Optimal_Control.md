# Lesson 14: Optimal Control — LQR and Kalman Filter

## Learning Objectives

- Formulate the linear-quadratic regulator (LQR) problem
- Solve the algebraic Riccati equation for the optimal gain
- Design Kalman filters for state estimation in the presence of noise
- Combine LQR and Kalman filter into the LQG controller
- Understand the robustness properties of LQR and the limitations of LQG

## 1. Motivation for Optimal Control

Pole placement gives us the freedom to choose closed-loop pole locations, but it does not tell us **which** pole locations are best. Optimal control provides a systematic framework:

- Define a **cost function** that penalizes tracking error and control effort
- Find the control law that **minimizes** the cost
- The result balances performance against effort in a principled way

## 2. Linear-Quadratic Regulator (LQR)

### 2.1 Problem Formulation

**Plant:** $\dot{x} = Ax + Bu$

**Cost function:**

$$J = \int_0^\infty \left[ x^T(t) Q x(t) + u^T(t) R u(t) \right] dt$$

where:
- $Q \geq 0$ (positive semidefinite): **state weighting matrix** — penalizes state deviation
- $R > 0$ (positive definite): **control weighting matrix** — penalizes control effort

**Goal:** Find $u(t)$ that minimizes $J$.

### 2.2 The Optimal Solution

**Theorem:** If $(A, B)$ is controllable (or stabilizable) and $(A, Q^{1/2})$ is observable (or detectable), the optimal control law is:

$$u^*(t) = -Kx(t), \quad K = R^{-1}B^T P$$

where $P$ is the unique positive definite solution of the **algebraic Riccati equation (ARE)**:

$$A^T P + PA - PBR^{-1}B^T P + Q = 0$$

### 2.3 Properties of LQR

**Guaranteed stability:** The closed-loop system $\dot{x} = (A - BK)x$ is always asymptotically stable.

**Guaranteed robustness (SISO):**
- **Gain margin:** $[1/2, \infty)$ — the gain can be halved or increased to infinity and the system remains stable
- **Phase margin:** $\geq 60°$

These are remarkable guarantees — no other linear design method provides such strong margins automatically.

### 2.4 Tuning $Q$ and $R$

**Physical interpretation:**
- Large $Q_{ii}$: aggressively drive $x_i$ to zero (fast response)
- Large $R_{jj}$: penalize large control inputs on channel $j$ (small control effort)
- $Q/R$ ratio: trade-off between performance and control effort

**Common choices:**
- $Q = \text{diag}(q_1, \ldots, q_n)$, $R = \rho I$ — single parameter $\rho$ tunes the trade-off
- **Bryson's rule:** $Q_{ii} = 1/x_{i,\max}^2$, $R_{jj} = 1/u_{j,\max}^2$ — normalize by maximum acceptable values

### 2.5 Example

Double integrator: $A = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}$, $B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$

With $Q = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$, $R = [1]$:

Solving the ARE yields $P = \begin{bmatrix} \sqrt{3} & 1 \\ 1 & \sqrt{3} \end{bmatrix}$

$$K = R^{-1}B^T P = \begin{bmatrix} 1 & \sqrt{3} \end{bmatrix}$$

Closed-loop poles: $s = -\frac{\sqrt{3}}{2} \pm j\frac{1}{2}$ → $\omega_n = 1$, $\zeta = \frac{\sqrt{3}}{2} \approx 0.87$.

Increasing $Q_{11}$ → faster response, more control effort. Increasing $R$ → slower, gentler response.

## 3. Kalman Filter

### 3.1 Problem Formulation

**Plant with noise:**

$$\dot{x} = Ax + Bu + Gw$$
$$y = Cx + v$$

where:
- $w(t)$: **process noise** (disturbances, model uncertainty) — $E[ww^T] = W$
- $v(t)$: **measurement noise** (sensor noise) — $E[vv^T] = V$
- Both are white, zero-mean, Gaussian

**Goal:** Find the best estimate $\hat{x}(t)$ of $x(t)$ given the noisy measurements $y(t)$.

### 3.2 The Kalman Filter (Continuous-Time)

$$\dot{\hat{x}} = A\hat{x} + Bu + L(y - C\hat{x})$$

The optimal gain is:

$$L = P_f C^T V^{-1}$$

where $P_f$ is the solution of the **filter algebraic Riccati equation**:

$$AP_f + P_f A^T - P_f C^T V^{-1} C P_f + GWG^T = 0$$

### 3.3 Properties

- The Kalman filter is the **optimal** linear estimator (minimizes $E[\|x - \hat{x}\|^2]$)
- It has the same structure as the Luenberger observer (Lesson 13) but with the gain chosen optimally
- The filter **balances** trust in the model vs. trust in measurements:
  - Large $W$ (noisy model) → large $L$ (trust measurements more)
  - Large $V$ (noisy sensors) → small $L$ (trust model more)

### 3.4 Duality with LQR

The Kalman filter and LQR are **dual** problems:

| LQR | Kalman Filter |
|-----|---------------|
| $A^TP + PA - PBR^{-1}B^TP + Q = 0$ | $AP_f + P_fA^T - P_fC^TV^{-1}CP_f + GWG^T = 0$ |
| $K = R^{-1}B^TP$ | $L = P_fC^TV^{-1}$ |
| State weighting $Q$ | Process noise $GWG^T$ |
| Control weighting $R$ | Measurement noise $V$ |
| Feedback gain $K$ | Observer gain $L$ |

## 4. LQG Control

### 4.1 Combining LQR + Kalman Filter

The **Linear-Quadratic-Gaussian (LQG)** controller combines:
- LQR for optimal state feedback
- Kalman filter for optimal state estimation

$$u = -K\hat{x}, \quad K = R^{-1}B^T P \quad \text{(LQR)}$$
$$\dot{\hat{x}} = A\hat{x} + Bu + L(y - C\hat{x}), \quad L = P_f C^T V^{-1} \quad \text{(Kalman filter)}$$

### 4.2 Separation Principle (Stochastic)

The **certainty equivalence principle** guarantees that the LQR and Kalman filter can be designed independently — the same separation principle from Lesson 13 applies in the stochastic setting.

### 4.3 LQG Transfer Function

The LQG controller is a dynamic compensator:

$$G_{LQG}(s) = K(sI - A + BK + LC)^{-1}L$$

This is a proper transfer function of order $n$ (same as the plant).

## 5. Robustness Limitations of LQG

### 5.1 The Problem

While LQR has guaranteed margins ($GM = [1/2, \infty)$, $PM \geq 60°$), **LQG has no guaranteed margins**. The Kalman filter can arbitrarily degrade the robustness of LQR.

This was a major discovery in the 1970s (Doyle, 1978) — it showed that optimal control does not automatically give robust control.

### 5.2 Loop Transfer Recovery (LTR)

**LQG/LTR** attempts to recover the LQR robustness by designing the Kalman filter to make the loop transfer function approximate the LQR loop transfer function:

$$L(j\omega) \approx K(j\omega I - A)^{-1}B \quad \text{at the plant input}$$

This is achieved by increasing the process noise covariance: $W \to qBB^T$ as $q \to \infty$.

## 6. Finite-Horizon LQR

### 6.1 Time-Varying Riccati Equation

For a finite time horizon $[0, t_f]$:

$$J = x^T(t_f) S_f x(t_f) + \int_0^{t_f} \left[ x^T Q x + u^T R u \right] dt$$

The optimal gain is $K(t) = R^{-1}B^T P(t)$ where $P(t)$ satisfies the **differential Riccati equation**:

$$-\dot{P} = A^T P + PA - PBR^{-1}B^T P + Q, \quad P(t_f) = S_f$$

This is integrated **backward** from $t = t_f$ to $t = 0$. As $t_f \to \infty$, $P(t)$ converges to the steady-state ARE solution.

## Practice Exercises

### Exercise 1: LQR Design

For the system $A = \begin{bmatrix} 0 & 1 \\ -1 & -1 \end{bmatrix}$, $B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$:

1. With $Q = I$ and $R = 1$, solve the ARE (set up the equation and solve the system of 3 nonlinear equations for the 3 unique elements of $P$)
2. Compute the optimal gain $K$
3. Find the closed-loop poles and verify stability
4. How do the poles change if $R$ is increased to $10$? To $0.1$?

### Exercise 2: Kalman Filter

For the same system with $C = \begin{bmatrix} 1 & 0 \end{bmatrix}$, process noise intensity $W = 0.1$, and measurement noise $V = 1$:

1. Set up the filter ARE
2. Compute the Kalman gain $L$
3. Where are the observer poles?

### Exercise 3: LQR Properties

Show that for the SISO case, the LQR return difference satisfies:

$$|1 + K(j\omega I - A)^{-1}B| \geq 1 \quad \forall \omega$$

Hint: Start from $1 + K(j\omega I - A)^{-1}B = 1 + R^{-1}B^T P(j\omega I - A)^{-1}B$ and use the ARE.

What does this imply about gain margin and phase margin?

---

*Previous: [Lesson 13 — State Feedback and Observer Design](13_State_Feedback_and_Observers.md) | Next: [Lesson 15 — Digital Control Systems](15_Digital_Control.md)*
