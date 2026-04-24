# Lesson 12: Controllability and Observability

## Learning Objectives

- Define and test controllability and observability for LTI systems
- Construct controllability and observability matrices and compute their rank
- Understand the PBH (Popov-Belevitch-Hautus) test
- Relate controllability/observability to transfer function pole-zero cancellations
- Decompose systems into controllable/uncontrollable and observable/unobservable parts
- Compute the rank tests numerically and recognize the rank-deficient cases that hide instability

## 0. Why These Two Properties Underwrite Modern Control

Lesson 11 introduced state space; this lesson introduces the two questions that determine whether you can do anything useful with it. They are the prerequisite for every controller and observer design that follows.

- **Controllability is "can I move the state where I want?"** Without controllability, no choice of $u(t)$ — gain, schedule, optimal — moves a stuck mode. The control engineer's job ends before it begins.
- **Observability is "can I figure out the state from what I see?"** Without observability, no observer or filter can recover hidden states from measurements. The estimator's job ends before it begins.
- **Together they define minimality.** A system that is both controllable and observable has the smallest possible state for its input-output behavior. Anything bigger is wasted state — modes that exist mathematically but cannot affect the output or be affected by the input.

These properties also explain a paradox you have already seen: a transfer function and a state-space model can disagree about stability. The state-space view is honest; the transfer function silently drops uncontrollable or unobservable modes via pole-zero cancellation. If a hidden mode is unstable, BIBO stability is a lie.

## 1. Motivation

Transfer function analysis captures only the **externally visible** behavior. State-space analysis reveals whether:

- **Controllability**: Can the input $u(t)$ drive the state $x(t)$ to any desired value?
- **Observability**: Can we determine the internal state $x(t)$ from measurements of $y(t)$ and $u(t)$?

These properties are fundamental for controller and observer design. A state that is not controllable cannot be influenced; a state that is not observable cannot be estimated.

## 2. Controllability

### 2.1 Definition

A system $(A, B)$ is **controllable** if, for any initial state $x(0) = x_0$ and any desired final state $x_f$, there exists a finite time $t_f > 0$ and an input $u(t)$ that drives the state from $x_0$ to $x_f$.

### 2.2 Controllability Matrix

**Theorem:** The system $(A, B)$ is controllable if and only if the **controllability matrix** has full rank:

$$\mathcal{C} = \begin{bmatrix} B & AB & A^2B & \cdots & A^{n-1}B \end{bmatrix}$$

$$\text{rank}(\mathcal{C}) = n$$

For a single-input system, $\mathcal{C}$ is $n \times n$ and controllability requires $\det(\mathcal{C}) \neq 0$.

### 2.3 Example

For the system $A = \begin{bmatrix} 0 & 1 \\ -2 & -3 \end{bmatrix}$, $B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$:

$$\mathcal{C} = \begin{bmatrix} B & AB \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ 1 & -3 \end{bmatrix}$$

$\det(\mathcal{C}) = 0 \cdot (-3) - 1 \cdot 1 = -1 \neq 0$ → **Controllable**.

### 2.4 Uncontrollable Example

$A = \begin{bmatrix} -1 & 0 \\ 0 & -2 \end{bmatrix}$, $B = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$:

$$\mathcal{C} = \begin{bmatrix} 1 & -1 \\ 0 & 0 \end{bmatrix}$$

$\text{rank}(\mathcal{C}) = 1 < 2$ → **Not controllable**. The second state $x_2$ evolves as $\dot{x}_2 = -2x_2$ regardless of the input — it cannot be influenced.

## 3. Observability

### 3.1 Definition

A system $(A, C)$ is **observable** if the initial state $x(0)$ can be uniquely determined from the output $y(t)$ and input $u(t)$ over a finite time interval $[0, t_f]$.

### 3.2 Observability Matrix

**Theorem:** The system $(A, C)$ is observable if and only if the **observability matrix** has full rank:

$$\mathcal{O} = \begin{bmatrix} C \\ CA \\ CA^2 \\ \vdots \\ CA^{n-1} \end{bmatrix}$$

$$\text{rank}(\mathcal{O}) = n$$

### 3.3 Duality

There is a fundamental **duality** between controllability and observability:

$$(A, B) \text{ is controllable} \iff (A^T, B^T) \text{ is observable}$$

$$(A, C) \text{ is observable} \iff (A^T, C^T) \text{ is controllable}$$

This means every theorem about controllability has a dual theorem about observability.

## 4. PBH Test

### 4.1 PBH Controllability Test

$(A, B)$ is controllable if and only if:

$$\text{rank}\begin{bmatrix} sI - A & B \end{bmatrix} = n \quad \forall s \in \mathbb{C}$$

Equivalently, there is no left eigenvector $q^T$ of $A$ such that $q^T B = 0$:

$$q^T A = \lambda q^T \text{ and } q^T B = 0 \implies \text{not controllable}$$

**Interpretation:** A mode is uncontrollable if its eigenvector is orthogonal to $B$.

### 4.2 PBH Observability Test

$(A, C)$ is observable if and only if:

$$\text{rank}\begin{bmatrix} sI - A \\ C \end{bmatrix} = n \quad \forall s \in \mathbb{C}$$

Equivalently, there is no eigenvector $v$ of $A$ such that $Cv = 0$:

$$Av = \lambda v \text{ and } Cv = 0 \implies \text{not observable}$$

**Interpretation:** A mode is unobservable if its eigenvector is in the null space of $C$.

### 4.3 Why Two Tests Exist

The Kalman rank test is faster to compute (one rank, on an $n \times nm$ matrix); the PBH test is faster to interpret (which mode is the problem, identified by its eigenvalue). In practice you use Kalman to detect the problem and PBH to diagnose which mode caused it.

## 5. Connection to Transfer Functions

### 5.1 Pole-Zero Cancellations

The transfer function $G(s) = C(sI-A)^{-1}B + D$ may have a lower order than the state-space model if there are pole-zero cancellations.

**Key theorem:** A pole-zero cancellation in $G(s)$ corresponds to a mode that is either **uncontrollable** or **unobservable** (or both).

### 5.2 Example

Consider:

$$A = \begin{bmatrix} -1 & 0 \\ 0 & -3 \end{bmatrix}, \quad B = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \quad C = \begin{bmatrix} 1 & 0 \end{bmatrix}$$

Transfer function:

$$G(s) = C(sI-A)^{-1}B = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} \frac{1}{s+1} & 0 \\ 0 & \frac{1}{s+3} \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \frac{1}{s+1}$$

The pole at $s = -3$ does not appear in the transfer function. Checking: the system is controllable (both states are excited by $B$), but $C = [1 \; 0]$ does not observe $x_2$ → the mode at $s = -3$ is **unobservable**.

### 5.3 Internal Stability vs. BIBO Stability

- **BIBO stability** depends on the transfer function poles (externally visible)
- **Internal stability** depends on the eigenvalues of $A$ (all modes)

A system can be BIBO stable but internally unstable if an unstable mode is hidden by pole-zero cancellation. This is dangerous — the hidden unstable mode will grow unbounded internally.

### 5.4 Numerical Test in Python

```python
import numpy as np
from scipy.linalg import matrix_rank

A = np.array([[-1, 0], [0, -3]], dtype=float)
B = np.array([[1], [1]], dtype=float)
C = np.array([[1, 0]], dtype=float)
n = A.shape[0]

# Kalman controllability and observability matrices
Ctrb = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])
Obsv = np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)])

print(f"rank(Ctrb) = {matrix_rank(Ctrb)} / {n}  → {'controllable' if matrix_rank(Ctrb) == n else 'NOT controllable'}")
print(f"rank(Obsv) = {matrix_rank(Obsv)} / {n}  → {'observable' if matrix_rank(Obsv) == n else 'NOT observable'}")
```

For the example above, you should see `rank(Ctrb) = 2` (controllable) and `rank(Obsv) = 1` (NOT observable). `python-control` provides `ctrb`, `obsv`, and `pole_zero_cancellation` helpers that wrap the same logic with a friendlier API.

> **Numerical caveat**: rank tests on floating-point matrices are sensitive to thresholding. Use `numpy.linalg.matrix_rank(M, tol=...)` with an explicit tolerance based on $\sigma_1 \cdot \epsilon \cdot \max(m,n)$ when the matrix is poorly scaled. Better still, use the SVD directly and look at the singular-value gap — a "rank-deficient" matrix with $\sigma_n / \sigma_1 = 0.001$ is more "barely controllable" than "uncontrollable" and will need more control effort to drive the weak mode.

## 6. Kalman Decomposition

Any LTI system can be decomposed into four parts:

```
┌──────────────────────────────────────────┐
│    ┌────────────┐    ┌────────────┐      │
│    │ Controllable│    │ Controllable│      │
│    │ Observable  │ →  │ Unobservable│      │
│    └────────────┘    └────────────┘      │
│         ↓                  ↓             │
│    ┌────────────┐    ┌────────────┐      │
│    │Uncontrollable│  │Uncontrollable│    │
│    │ Observable  │    │ Unobservable│     │
│    └────────────┘    └────────────┘      │
└──────────────────────────────────────────┘
```

Only the **controllable and observable** subsystem appears in the transfer function. The other three parts are hidden from the input-output perspective.

A realization $(A, B, C, D)$ is called **minimal** if it is both controllable and observable — it has the smallest possible state dimension for the given transfer function.

## 7. Controllability and Observability Gramians

### 7.1 Controllability Gramian

$$W_c(t) = \int_0^t e^{A\tau}BB^T e^{A^T\tau} \, d\tau$$

$(A, B)$ is controllable if and only if $W_c(t) > 0$ (positive definite) for some $t > 0$.

For stable systems, the infinite-horizon controllability Gramian $W_c = \int_0^\infty e^{A\tau}BB^T e^{A^T\tau} d\tau$ satisfies the **Lyapunov equation**:

$$AW_c + W_c A^T + BB^T = 0$$

### 7.2 Observability Gramian

$$W_o(t) = \int_0^t e^{A^T\tau}C^T C e^{A\tau} \, d\tau$$

$(A, C)$ is observable if and only if $W_o(t) > 0$ for some $t > 0$.

The Gramians quantify **how easily** each state can be controlled or observed — they are used in model reduction (balanced truncation).

## 8. Common Pitfalls

1. **Treating "rank-deficient" as a binary verdict in floating point.** Real systems live on a spectrum from "very controllable" (well-conditioned $\mathcal{C}$) to "barely controllable" (almost rank-deficient). Always inspect the smallest singular value of $\mathcal{C}$ — small values mean huge control effort to move the weak mode.
2. **Forgetting that minimality is per realization, not per transfer function.** Two realizations of the same transfer function can have different state dimensions if one has hidden modes. The minimal realization is the smallest one.
3. **Ignoring the "uncontrollable but stable" case.** If an uncontrollable mode is in the LHP, you cannot affect it but it decays harmlessly. Many physical systems have such modes (e.g., spring-mass-damper measured at a node — natural mode is invisible there). Detect them, but they are not always a deal-breaker.
4. **Trusting BIBO stability for safety-critical analysis.** The reason aviation certification requires state-space analysis is exactly this lesson: a hidden RHP mode can crash a plane while the input-output behavior looks fine. Always use the eigenvalues of $A$ for safety arguments.
5. **Building $\mathcal{C}$ for $n > 30$ naïvely.** $A^k B$ blows up numerically when $A$ has eigenvalues with significant magnitudes. Use the staircase form (`scipy.signal.ss2zpk` plus controllability staircase) for high-order systems instead of computing $A^{n-1}$.
6. **Confusing the Gramian's positivity with full rank.** $W_c > 0$ (positive definite) is equivalent to controllability for stable systems. For unstable systems use the Kalman/PBH tests directly — the integral that defines $W_c$ may not converge.

## Practice Exercises

### Exercise 1: Controllability and Observability Check

For the system:

$$A = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ -6 & -11 & -6 \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}, \quad C = \begin{bmatrix} 1 & 0 & 0 \end{bmatrix}$$

1. Compute the controllability matrix and determine if the system is controllable
2. Compute the observability matrix and determine if the system is observable
3. Find the transfer function and verify there are no pole-zero cancellations

### Exercise 2: PBH Test

For the system with $A = \begin{bmatrix} -2 & 1 \\ 0 & -2 \end{bmatrix}$, $B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$, $C = \begin{bmatrix} 1 & 0 \end{bmatrix}$:

1. Apply the PBH test for controllability at $s = -2$
2. Apply the PBH test for observability at $s = -2$
3. Find the transfer function — is this a minimal realization?

### Exercise 3: Hidden Modes

Consider $A = \begin{bmatrix} -1 & 0 \\ 0 & 2 \end{bmatrix}$, $B = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$, $C = \begin{bmatrix} 1 & 0 \end{bmatrix}$:

1. Is the system BIBO stable (from the transfer function)?
2. Is the system internally stable?
3. What is the danger of this system?

### Exercise 4: Numerical Spot-Check

Use the Python snippet from Section 5.4 to check controllability and observability of all three exercises above. Confirm that the rank-based answers match what you derived by hand. For Exercise 3, additionally print the eigenvalues of $A$ to see the hidden unstable mode that the transfer function does not show.

### Exercise 5: Almost-Uncontrollable Drill

Take $A = \begin{bmatrix} -1 & 0 \\ 0 & -2 \end{bmatrix}$, $B = \begin{bmatrix} 1 \\ \epsilon \end{bmatrix}$ for $\epsilon \in \{1, 0.1, 0.01, 0.001\}$. The system is fully controllable for any $\epsilon > 0$ but the controllability is increasingly weak. Compute the smallest singular value of $\mathcal{C}$ for each $\epsilon$ and explain how that value relates to the control effort needed to move the second mode.

---

*Previous: [Lesson 11 — State-Space Representation](11_State_Space_Representation.md) | Next: [Lesson 13 — State Feedback and Observer Design](13_State_Feedback_and_Observers.md)*
