# Lesson 14: Optimal Control — LQR and Kalman Filter

## Learning Objectives

- Formulate the linear-quadratic regulator (LQR) problem
- Solve the algebraic Riccati equation for the optimal gain
- Design Kalman filters for state estimation in the presence of noise
- Combine LQR and Kalman filter into the LQG controller
- Understand the robustness properties of LQR and the limitations of LQG
- Solve LQR/Kalman problems numerically with `scipy.linalg.solve_continuous_are`

## 0. Why "Optimal" Earned Its Own Lesson

Pole placement (Lesson 13) lets you put closed-loop poles anywhere — but it does not say where you _should_ put them. Optimal control flips the question: instead of guessing pole locations, define a cost that captures what you care about, and let the math pick the gain that minimizes it.

Three reasons LQR/Kalman dominate modern control practice:

- **One scalar tuning knob (Q vs R).** A 6-state system has 6 closed-loop poles, but LQR boils the design choice down to "how aggressive" via the ratio of $Q$ to $R$. Pole placement requires picking 6 numbers; LQR picks them all from a handful of weights.
- **Free robustness on the LQR side.** The result is automatically stable and has gain margin in $[1/2, \infty)$ and phase margin $\geq 60°$. No pole-placement design provides this guarantee for free.
- **State estimation falls out by duality.** The Kalman filter is the LQR's mirror twin — same Riccati equation with $A \to A^T$, $B \to C^T$, etc. Learning one problem gives you the other.

Mental picture: $J = \int (x^T Q x + u^T R u)\,dt$ is "how unhappy will I be after the controller has finished?" — the controller minimizes future unhappiness, weighted by your $Q$ and $R$ choices.

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

### 2.6 Solving the ARE in Python

The ARE is a quadratic matrix equation; solving by hand for $n > 2$ is painful and the symbolic solution rarely simplifies. SciPy's `solve_continuous_are` is the right answer:

```python
import numpy as np
from scipy.linalg import solve_continuous_are

A = np.array([[0, 1], [0, 0]], dtype=float)
B = np.array([[0], [1]], dtype=float)
Q = np.diag([1.0, 0.0])
R = np.array([[1.0]])

P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
print("P =\n", P)
print("K =", K)

# Closed-loop poles
A_cl = A - B @ K
print("closed-loop eigenvalues =", np.linalg.eigvals(A_cl))
```

For the double-integrator above, you should see $K \approx [1.0, 1.732]$ and eigenvalues at $-0.866 \pm 0.5j$ — matching the analytical result.

To sweep $Q$ vs $R$, vary one weight and re-solve in a loop. The closed-loop poles trace out a curve in the $s$-plane — the same kind of insight root locus gives, but parameterized by the cost ratio instead of an open-loop gain.

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

Practical consequence: code that solves the LQR ARE solves the Kalman ARE by transposing inputs. The same `scipy.linalg.solve_continuous_are(A.T, C.T, G@W@G.T, V)` returns the filter $P_f$ — one routine, two algorithms.

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

> **Practical note**: in safety-critical applications (aerospace, automotive), LQG without LTR is rarely deployed. Either you use full LQG/LTR, or you skip Kalman entirely and use a model-based observer with explicit margin checks. The 6 dB / 60° guarantee of pure LQR is too valuable to give up unintentionally.

## 6. Finite-Horizon LQR

### 6.1 Time-Varying Riccati Equation

For a finite time horizon $[0, t_f]$:

$$J = x^T(t_f) S_f x(t_f) + \int_0^{t_f} \left[ x^T Q x + u^T R u \right] dt$$

The optimal gain is $K(t) = R^{-1}B^T P(t)$ where $P(t)$ satisfies the **differential Riccati equation**:

$$-\dot{P} = A^T P + PA - PBR^{-1}B^T P + Q, \quad P(t_f) = S_f$$

This is integrated **backward** from $t = t_f$ to $t = 0$. As $t_f \to \infty$, $P(t)$ converges to the steady-state ARE solution.

> **Why backward integration?** The terminal cost $S_f$ provides the boundary condition AT $t_f$, not at $t = 0$. Solving forward from $t = 0$ would require knowing $P(0)$ — the very thing we are trying to compute. Backward sweep is mathematically equivalent and numerically clean.

## 7. Common Pitfalls

1. **Treating LQR as "magical robustness".** The 6 dB / 60° guarantees are for the LQR loop **with full state feedback** — i.e., $u = -Kx$ with $x$ measured directly. Using LQR with an observer (LQG) breaks the guarantee. If you need robustness from LQG, you need LTR or a different design.
2. **Choosing $Q$ and $R$ with wildly different scales.** When state and control magnitudes differ by 6 orders of magnitude, the conditioning of the ARE solver suffers. Use Bryson's rule or normalize states/inputs to $[0, 1]$ ranges before designing.
3. **Forgetting the (A, Q^(1/2)) detectability condition.** The ARE has a unique stabilizing solution only when this is satisfied. If your $Q$ is rank-deficient AND the unobserved-by-Q modes are unstable in $A$, the ARE fails — sometimes silently in numerical software.
4. **Over-trusting numerical ARE solvers near critical conditioning.** `solve_continuous_are` uses the Schur method, which is robust but not bulletproof. Always verify the residual `A.T @ P + P @ A - P @ B @ inv(R) @ B.T @ P + Q` is small (e.g., Frobenius norm < $10^{-8}$).
5. **Tuning $Q$ and $R$ to match a desired closed-loop pole pattern.** The mapping $(Q, R) \to$ poles is non-linear and weight-matrix-dependent. If you have specific pole targets, use direct pole placement (Lesson 13). LQR is for "minimize cost," not "place poles at $X$."
6. **Forgetting that finite-horizon $K(t)$ is time-varying.** Implementing the time-varying $K(t)$ in software requires storing the trajectory $P(t)$ from the backward sweep. Many beginners apply the steady-state $K$ — which is suboptimal, sometimes badly so for short horizons.

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

### Exercise 4: Numerical LQR Sweep

Use the Python snippet from Section 2.6 to solve the LQR for the double integrator with $R \in \{0.01, 0.1, 1, 10, 100\}$. For each, plot the closed-loop pole locations on the $s$-plane. Verify that smaller $R$ pushes poles further left (faster response, more effort).

### Exercise 5: LQG Robustness Demo — Conceptual

Construct a small example (2-state plant) where adding a Kalman filter to a robust LQR design produces an LQG controller with phase margin below $30°$. Suggest one LTR-style modification (large $W$ on $BB^T$) and show numerically that the margin is restored.

---

*Previous: [Lesson 13 — State Feedback and Observer Design](13_State_Feedback_and_Observers.md) | Next: [Lesson 15 — Digital Control Systems](15_Digital_Control.md)*
