# Lesson 13: State Feedback and Observer Design

## Learning Objectives

- Design state feedback controllers using pole placement
- Understand the separation principle for combined controller-observer design
- Design full-order Luenberger observers for state estimation
- Implement observer-based state feedback controllers
- Apply Ackermann's formula for SISO pole placement
- Compute pole-placement gains numerically and verify the separation principle in code

## 0. The Pivot from "Read Outputs" to "Manipulate States"

Lessons 4–10 worked entirely with input-output transfer functions. Lessons 11–12 introduced the state vector. This lesson is where state space pays off: with the state in hand (or estimated), you can place every closed-loop pole exactly where you want, instead of negotiating with whatever the plant gives you.

Three things this unlocks:

- **Arbitrary pole placement.** If $(A, B)$ is controllable, you can put the closed-loop poles anywhere in the complex plane — not just nudge them toward the open-loop zeros like a root locus would. This is a complete leap in design freedom.
- **Observer-based control.** Most plants do not give you the full state vector at the output. The Luenberger observer estimates the missing states from $y$ and $u$ — and the **separation principle** says you can design the controller and observer independently.
- **A platform for LQR / Kalman / MPC.** Every advanced controller in Lesson 14 and beyond is "design feedback gain" + "design state estimator." Mastering this lesson is the prerequisite to everything that follows.

Mental picture: state feedback is "reach inside the plant and push the state toward zero with a custom-shaped force"; the observer is "watch the outputs over time and infer what the inside state must be."

## 1. State Feedback Control

### 1.1 Full State Feedback

If all states are measurable, we can use **state feedback**:

$$u(t) = -Kx(t) + r(t)$$

where $K \in \mathbb{R}^{m \times n}$ is the **feedback gain matrix** and $r(t)$ is a reference command.

The closed-loop system becomes:

$$\dot{x} = (A - BK)x + Br$$
$$y = Cx$$

The closed-loop poles are the eigenvalues of $A - BK$.

### 1.2 Pole Placement Theorem

**Theorem:** If the system $(A, B)$ is controllable, then the eigenvalues of $A - BK$ can be placed at **any** desired locations by appropriate choice of $K$.

This is the fundamental result: controllability guarantees arbitrary pole placement.

### 1.3 Design Procedure (SISO)

**Given:** Desired closed-loop poles $s_1, s_2, \ldots, s_n$.

**Desired characteristic polynomial:**

$$\Delta_d(s) = (s - s_1)(s - s_2)\cdots(s - s_n) = s^n + \alpha_{n-1}s^{n-1} + \cdots + \alpha_0$$

**Method 1: Direct comparison**

1. Compute $\det(sI - A + BK) = s^n + f_{n-1}(K)s^{n-1} + \cdots + f_0(K)$
2. Set $f_i(K) = \alpha_i$ for $i = 0, \ldots, n-1$
3. Solve the $n$ equations for $K = [k_1 \; k_2 \; \cdots \; k_n]$

**Method 2: Transform to CCF**

If the system is in controllable canonical form, the gain is simply:

$$K = [\alpha_0 - a_0 \;\; \alpha_1 - a_1 \;\; \cdots \;\; \alpha_{n-1} - a_{n-1}]$$

where $a_i$ are the original characteristic polynomial coefficients.

### 1.4 Ackermann's Formula (SISO)

$$K = \begin{bmatrix} 0 & 0 & \cdots & 0 & 1 \end{bmatrix} \mathcal{C}^{-1} \Delta_d(A)$$

where $\mathcal{C}$ is the controllability matrix and $\Delta_d(A)$ is the desired characteristic polynomial evaluated at the matrix $A$:

$$\Delta_d(A) = A^n + \alpha_{n-1}A^{n-1} + \cdots + \alpha_1 A + \alpha_0 I$$

### 1.5 Example

$A = \begin{bmatrix} 0 & 1 \\ -2 & -3 \end{bmatrix}$, $B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$

Desired poles: $s = -5, -5$ → $\Delta_d(s) = s^2 + 10s + 25$

Current: $\Delta(s) = s^2 + 3s + 2$

In CCF (this system is already in CCF with $a_0 = 2, a_1 = 3$):

$$K = [25 - 2 \;\; 10 - 3] = [23 \;\; 7]$$

Verify: $A - BK = \begin{bmatrix} 0 & 1 \\ -2-23 & -3-7 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ -25 & -10 \end{bmatrix}$ with eigenvalues $-5, -5$. ✓

### 1.6 Pole Placement in Python

`scipy.signal.place_poles` does the work — robust enough for any $n$ up to ~30:

```python
import numpy as np
from scipy.signal import place_poles

A = np.array([[0, 1], [-2, -3]], dtype=float)
B = np.array([[0], [1]], dtype=float)

desired = np.array([-5, -5.001])   # tiny separation needed for numerical robustness
K = place_poles(A, B, desired).gain_matrix
print("K =", K)
print("Closed-loop eigenvalues =", np.linalg.eigvals(A - B @ K))
```

The `1e-3` separation is a real numerical issue — `place_poles` rejects exactly-repeated targets because the algorithm internally needs the eigenvectors to span $\mathbb{R}^n$. For exactly-repeated poles, fall back to Ackermann's formula or the `acker` function in `python-control`.

## 2. Reference Tracking with State Feedback

State feedback $u = -Kx$ drives the output to zero, not to a nonzero reference. To track a step reference $r$:

### 2.1 Feedforward Gain

$$u = -Kx + N_r r$$

where $N_r$ is chosen so that $y_{ss} = r$ in steady state:

$$N_r = \frac{1}{C(-A + BK)^{-1}B}$$

Equivalently, using the DC gain of the closed-loop: $N_r = 1/G_{cl}(0)$.

### 2.2 Integral Action

Add an integrator state $x_I = \int (r - y) dt$:

$$\dot{x}_I = r - Cx$$

Augmented system:

$$\begin{bmatrix} \dot{x} \\ \dot{x}_I \end{bmatrix} = \begin{bmatrix} A & 0 \\ -C & 0 \end{bmatrix} \begin{bmatrix} x \\ x_I \end{bmatrix} + \begin{bmatrix} B \\ 0 \end{bmatrix} u + \begin{bmatrix} 0 \\ 1 \end{bmatrix} r$$

With feedback $u = -[K \; K_I] [x^T \; x_I]^T$, the integrator ensures **zero steady-state error** for step references and step disturbances.

> **Why feedforward alone is fragile**: $N_r = 1/G_{cl}(0)$ depends on the EXACT plant DC gain. Any modeling error (e.g., the actual plant has a slightly different DC gain) leaves a steady-state offset. Integral action compensates automatically — it is the robust counterpart to feedforward.

## 3. Observer Design

### 3.1 Motivation

In practice, not all states are measurable. An **observer** (or **estimator**) reconstructs the state from measured outputs and known inputs.

### 3.2 Full-Order Luenberger Observer

$$\dot{\hat{x}} = A\hat{x} + Bu + L(y - C\hat{x})$$

where $\hat{x}$ is the estimated state and $L \in \mathbb{R}^{n \times p}$ is the **observer gain matrix**.

The estimation error $e = x - \hat{x}$ satisfies:

$$\dot{e} = (A - LC)e$$

If all eigenvalues of $A - LC$ have negative real parts, $e(t) \to 0$ exponentially.

### 3.3 Observer Pole Placement

**Theorem:** If the system $(A, C)$ is observable, then the eigenvalues of $A - LC$ can be placed at **any** desired locations by appropriate choice of $L$.

This is the **dual** of the state feedback pole placement theorem.

**Design rule of thumb:** Place observer poles 3-5 times faster than the controller poles (so the estimate converges before the controller needs accurate state information).

### 3.4 Ackermann's Formula for Observer Gain

$$L = \Delta_o(A) \mathcal{O}^{-1} \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix}$$

where $\mathcal{O}$ is the observability matrix and $\Delta_o(A)$ is the desired observer characteristic polynomial evaluated at $A$.

### 3.5 Example

Using the same system: $A = \begin{bmatrix} 0 & 1 \\ -2 & -3 \end{bmatrix}$, $C = \begin{bmatrix} 1 & 0 \end{bmatrix}$

Desired observer poles: $s = -15, -15$ → $\Delta_o(s) = s^2 + 30s + 225$

$$\mathcal{O} = \begin{bmatrix} C \\ CA \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$

$$L = \Delta_o(A) \mathcal{O}^{-1} \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

$$\Delta_o(A) = A^2 + 30A + 225I = \begin{bmatrix} -2 & -3 \\ 6 & 7 \end{bmatrix} + \begin{bmatrix} 0 & 30 \\ -60 & -90 \end{bmatrix} + \begin{bmatrix} 225 & 0 \\ 0 & 225 \end{bmatrix} = \begin{bmatrix} 223 & 27 \\ -54 & 142 \end{bmatrix}$$

$$L = \begin{bmatrix} 223 & 27 \\ -54 & 142 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 27 \\ 142 \end{bmatrix}$$

### 3.6 Observer Design via Duality

Because observer design is dual to state-feedback design (Lesson 12), `place_poles` works for observers too — by transposing $A$ and using $C^T$ as the "$B$":

```python
C = np.array([[1, 0]], dtype=float)
desired_obs = np.array([-15, -15.001])
L = place_poles(A.T, C.T, desired_obs).gain_matrix.T
print("L =\n", L)
print("Observer eigenvalues =", np.linalg.eigvals(A - L @ C))
```

The same library function does both jobs, with no conceptual extra work — just the transpose trick.

## 4. The Separation Principle

### 4.1 Combined Controller-Observer

The observer-based state feedback controller uses estimated states:

$$u = -K\hat{x} + N_r r$$

```
     r  →(+)→ [N_r] →(+)→ [Plant] → y
                       ↑              |
                  [-K] ← x̂ ← [Observer] ← y, u
```

### 4.2 Separation Principle

**Theorem (Separation Principle):** The closed-loop eigenvalues of the combined controller-observer system are the **union** of the controller eigenvalues and the observer eigenvalues:

$$\sigma(A_{cl}) = \sigma(A - BK) \cup \sigma(A - LC)$$

This means:
- The controller can be designed **independently** of the observer
- The observer can be designed **independently** of the controller
- The combined system is stable if both subsystems are stable

### 4.3 Proof Sketch

The combined state is $[x^T \; e^T]^T$ where $e = x - \hat{x}$:

$$\begin{bmatrix} \dot{x} \\ \dot{e} \end{bmatrix} = \begin{bmatrix} A-BK & BK \\ 0 & A-LC \end{bmatrix} \begin{bmatrix} x \\ e \end{bmatrix} + \begin{bmatrix} B N_r \\ 0 \end{bmatrix} r$$

The block-triangular structure means the eigenvalues of the $2n \times 2n$ matrix are precisely those of $A-BK$ and $A-LC$.

## 5. Reduced-Order Observers

If some states are directly measured, we only need to estimate the remaining states.

For $y = Cx$ where $C$ has rank $p$, only $n - p$ states need estimation. The **reduced-order observer** has dimension $n - p$ instead of $n$, reducing computational cost.

The design follows a similar principle but with a transformed coordinate system that separates measured and unmeasured states.

## 6. Design Considerations

### 6.1 Pole Selection Guidelines

**Controller poles:**
- Must satisfy time-domain specifications (settling time, overshoot)
- Should not require excessive control effort (avoid very fast poles)
- Complex poles come in conjugate pairs

**Observer poles:**
- 3-5× faster than controller poles (typical rule)
- Too fast → sensitive to noise and modeling errors
- Too slow → poor estimation during transients

### 6.2 Robustness

State feedback with full state knowledge has excellent robustness properties (infinite gain margin for SISO). However, observer-based feedback can have **poor robustness** — the guaranteed margins of LQR are lost when an observer is used.

This motivates the **LQG/LTR** (Loop Transfer Recovery) procedure, where the observer design is modified to recover the robustness of the state feedback design.

## 7. Common Pitfalls

1. **Trying pole placement on an uncontrollable mode.** If $(A, B)$ is not controllable, no $K$ moves the uncontrollable mode. `place_poles` will throw an error or return a $K$ that places only the controllable poles. Always check controllability first (Lesson 12).
2. **Placing poles unrealistically fast.** Pole placement says you CAN put a pole at $s = -1000$ — it does not say you SHOULD. The required control effort scales with the pole displacement; very fast poles imply huge $|u(t)|$ that saturates the actuator, breaking the linear design.
3. **Ignoring the noise penalty of fast observer poles.** A 100× faster observer is also 100× more sensitive to measurement noise. The "3–5×" rule of thumb balances convergence speed against noise amplification.
4. **Repeated poles causing numerical issues.** `place_poles` (and most modern algorithms) wants distinct target poles. If you genuinely need repeated poles, perturb them by $10^{-3}$ relative or use Ackermann's formula directly.
5. **Using the separation principle on a nonlinear plant.** It is a strictly LTI result. For nonlinear systems with extended Kalman filters, the principle does not hold; controller and observer interact in non-obvious ways. Verify by simulation, not by separation.
6. **Forgetting the observer is itself a dynamic system that must be initialized.** Setting $\hat{x}(0) = 0$ when the true state is far from zero produces a long transient where $e(0)$ is large. Use any prior knowledge to initialize $\hat{x}$ closer to truth.

## Practice Exercises

### Exercise 1: Pole Placement

For $A = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}$, $B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$ (double integrator):

1. Design $K$ to place closed-loop poles at $s = -3 \pm j4$
2. Compute the closed-loop natural frequency and damping ratio
3. If $C = [1 \; 0]$, compute $N_r$ for zero steady-state tracking error

### Exercise 2: Observer Design

For the system in Exercise 1 with $C = \begin{bmatrix} 1 & 0 \end{bmatrix}$:

1. Verify observability
2. Design an observer with poles at $s = -15 \pm j20$
3. Write the full observer state equations

### Exercise 3: Separation Principle Verification

Using the controller from Exercise 1 and observer from Exercise 2:

1. Write the $4 \times 4$ closed-loop system matrix
2. Verify that the eigenvalues are the union of controller and observer poles
3. Simulate the step response and compare with full-state-feedback (no observer)

### Exercise 4: Numerical Pole Placement Drill

Using the snippet from Section 1.6, design a controller for the third-order system $A = \begin{bmatrix}0 & 1 & 0\\0 & 0 & 1\\-1 & -2 & -3\end{bmatrix}$, $B = \begin{bmatrix}0\\0\\1\end{bmatrix}$ with desired poles at $\{-2, -3, -4\}$. Verify the closed-loop eigenvalues match. Then re-do with desired poles $\{-2, -2, -4\}$ — `place_poles` will reject; explain why and use Ackermann's formula instead.

### Exercise 5: Pole-Speed Trade-off

For the controller in Exercise 1, sweep the pole magnitude from $|s| = 1$ to $|s| = 100$ (keep the angle constant). For each pole choice, compute the maximum $|u(t)|$ during a unit-step response. Plot the relationship and identify where actuator saturation (assume $|u| \leq 10$) becomes a hard limit.

---

*Previous: [Lesson 12 — Controllability and Observability](12_Controllability_and_Observability.md) | Next: [Lesson 14 — Optimal Control: LQR and Kalman Filter](14_Optimal_Control.md)*
