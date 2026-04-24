# Lesson 11: State-Space Representation

## Learning Objectives

- Represent dynamic systems in state-space form
- Convert between transfer functions and state-space models
- Identify controllable canonical, observable canonical, and diagonal (modal) forms
- Understand the relationship between state-space and transfer function representations
- Compute the state transition matrix and solve state equations
- Build state-space objects in Python and verify the conversion to/from transfer functions

## 0. Why Switch From Transfer Functions to State Space?

Eight lessons of transfer-function machinery, then a new representation? Three reasons it earns its own toolkit:

- **MIMO is natural here.** A transfer function is a single ratio of polynomials. A state-space model is matrices — a 4-input, 6-output, 12-state aircraft model is `(A: 12×12, B: 12×4, C: 6×12, D: 6×4)` and the math reads no differently than the SISO case.
- **Internal stability is visible.** Two transfer functions can be identical while one represents an internally unstable system (a hidden RHP pole cancelled by a zero). State space cannot hide it — the eigenvalues of $A$ are the actual closed-loop poles, cancellations or not.
- **Modern controllers are state-space.** LQR, Kalman filter, model predictive control, $H_\infty$ — every controller designed since the 1960s is formulated in state space. Transfer functions are pedagogically friendly but stop scaling around order 10.

Mental model: a transfer function is a "black box with a frequency response"; a state-space model is "an internal mechanism whose state can be measured, controlled, or estimated." The shift in mindset is from input-output to internal.

## 1. From Transfer Functions to State Space

Transfer functions capture only the input-output behavior. **State-space representation** captures the full internal dynamics, enabling:
- MIMO (multi-input, multi-output) system analysis
- Internal stability analysis (not just BIBO)
- Systematic controller and observer design
- Handling of nonlinear systems (via linearized state models)

## 2. State-Space Equations

A continuous-time LTI system in state-space form:

$$\dot{x}(t) = Ax(t) + Bu(t) \quad \text{(state equation)}$$
$$y(t) = Cx(t) + Du(t) \quad \text{(output equation)}$$

where:
- $x(t) \in \mathbb{R}^n$: **state vector** ($n$ = system order)
- $u(t) \in \mathbb{R}^m$: **input vector**
- $y(t) \in \mathbb{R}^p$: **output vector**
- $A \in \mathbb{R}^{n \times n}$: **system matrix** (or state matrix)
- $B \in \mathbb{R}^{n \times m}$: **input matrix**
- $C \in \mathbb{R}^{p \times n}$: **output matrix**
- $D \in \mathbb{R}^{p \times m}$: **feedforward matrix** (often zero)

### 2.1 Block Diagram

```
u(t) → [B] →(+)→ [∫] → x(t) → [C] →(+)→ y(t)
              ↑                        ↑
              └── [A] ←───────────┘    [D] ← u(t)
```

## 3. Deriving State-Space Models

### 3.1 From Differential Equations

**Example:** Mass-spring-damper: $m\ddot{y} + b\dot{y} + ky = F$

Choose state variables: $x_1 = y$, $x_2 = \dot{y}$

$$\dot{x}_1 = x_2$$
$$\dot{x}_2 = -\frac{k}{m}x_1 - \frac{b}{m}x_2 + \frac{1}{m}F$$

In matrix form:

$$\begin{bmatrix} \dot{x}_1 \\ \dot{x}_2 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ -k/m & -b/m \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} 0 \\ 1/m \end{bmatrix} F$$

$$y = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

### 3.2 From Transfer Functions

Given $G(s) = \frac{b_1 s + b_0}{s^2 + a_1 s + a_0}$, the **controllable canonical form** is:

$$A = \begin{bmatrix} 0 & 1 \\ -a_0 & -a_1 \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad C = \begin{bmatrix} b_0 & b_1 \end{bmatrix}, \quad D = 0$$

For an $n$-th order system $G(s) = \frac{b_{n-1}s^{n-1} + \cdots + b_0}{s^n + a_{n-1}s^{n-1} + \cdots + a_0}$:

$$A = \begin{bmatrix} 0 & 1 & 0 & \cdots & 0 \\ 0 & 0 & 1 & \cdots & 0 \\ \vdots & & & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & 1 \\ -a_0 & -a_1 & -a_2 & \cdots & -a_{n-1} \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 0 \\ 1 \end{bmatrix}$$

### 3.3 From State Space to Transfer Function

Taking the Laplace transform (zero initial conditions):

$$sX(s) = AX(s) + BU(s) \Rightarrow X(s) = (sI - A)^{-1}BU(s)$$

$$Y(s) = [C(sI - A)^{-1}B + D]U(s)$$

Therefore:

$$G(s) = C(sI - A)^{-1}B + D$$

### 3.4 Conversion in Python

A few lines confirm the bidirectional mapping. The 1:1 correspondence is one-way only when $G$ has no pole-zero cancellations — every state-space realization is a valid transfer function, but multiple non-equivalent state-space realizations exist for any given transfer function (canonical forms among them).

```python
import numpy as np
from control import tf, ss, ss2tf, tf2ss

# Start from a transfer function
G = tf([2, 3], [1, 4, 5, 6])
print("Transfer function:", G)

# Convert to state space (controllable canonical form by default in python-control)
sys_ss = tf2ss(G)
print("A =\n", sys_ss.A)
print("B =\n", sys_ss.B)
print("C =", sys_ss.C)
print("D =", sys_ss.D)

# Round-trip back to transfer function
G_back = ss2tf(sys_ss)
print("Round-trip TF:", G_back)
```

The round-trip $G \to (A, B, C, D) \to G$ should match exactly. If it does not, the most common cause is hidden modes — uncontrollable or unobservable states that the conversion silently drops.

## 4. Canonical Forms

### 4.1 Controllable Canonical Form (CCF)

As shown above. The last row of $A$ contains the negated coefficients of the characteristic polynomial.

**Property:** Always controllable (by construction).

### 4.2 Observable Canonical Form (OCF)

$$A = \begin{bmatrix} 0 & 0 & \cdots & 0 & -a_0 \\ 1 & 0 & \cdots & 0 & -a_1 \\ 0 & 1 & \cdots & 0 & -a_2 \\ \vdots & & \ddots & & \vdots \\ 0 & 0 & \cdots & 1 & -a_{n-1} \end{bmatrix}, \quad C = \begin{bmatrix} 0 & 0 & \cdots & 0 & 1 \end{bmatrix}$$

**Property:** Always observable. Note: OCF is the **transpose** of CCF (with $B$ and $C$ transposed as well).

### 4.3 Diagonal (Modal) Form

If $A$ has distinct eigenvalues $\lambda_1, \ldots, \lambda_n$, we can diagonalize:

$$\bar{A} = T^{-1}AT = \text{diag}(\lambda_1, \ldots, \lambda_n)$$

where $T = [v_1 \; v_2 \; \cdots \; v_n]$ is the matrix of eigenvectors.

Each state in diagonal form evolves independently — the system is decoupled into $n$ first-order modes.

### 4.4 Jordan Form

If $A$ has repeated eigenvalues, the diagonal form may not exist. The **Jordan normal form** handles this:

$$J = \begin{bmatrix} J_1 & & \\ & J_2 & \\ & & \ddots \end{bmatrix}, \quad J_i = \begin{bmatrix} \lambda_i & 1 & \\ & \lambda_i & 1 \\ & & \ddots & 1 \\ & & & \lambda_i \end{bmatrix}$$

### 4.5 Choosing Between Forms

| Form | Best for |
|------|----------|
| Controllable canonical | Designing state feedback ($u = -Kx$) — gain placement is direct |
| Observable canonical | Designing observers — output couples to a single state |
| Diagonal / modal | Analyzing dominant modes; decoupled simulation; LQR weighting |
| Jordan | Theoretical analysis with repeated eigenvalues |
| Physical (e.g. SMD above) | Best for matching the model to a real system; preserves intuition |

In practice, you will usually keep the system in its physical form for modeling, then transform to canonical or modal form for design.

## 5. State Transition Matrix

### 5.1 Homogeneous Solution

For $\dot{x} = Ax$ with initial condition $x(0) = x_0$:

$$x(t) = e^{At} x_0$$

where the **matrix exponential** is:

$$e^{At} = \Phi(t) = I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + \cdots$$

### 5.2 Properties of the State Transition Matrix

- $\Phi(0) = I$
- $\Phi(t_1 + t_2) = \Phi(t_1)\Phi(t_2)$
- $\Phi^{-1}(t) = \Phi(-t)$
- $\dot{\Phi}(t) = A\Phi(t)$
- $\Phi(t) = \mathcal{L}^{-1}\{(sI - A)^{-1}\}$

### 5.3 Complete Solution

For $\dot{x} = Ax + Bu$ with initial condition $x(0) = x_0$:

$$x(t) = e^{At}x_0 + \int_0^t e^{A(t-\tau)}Bu(\tau) \, d\tau$$

The first term is the **natural response** (due to initial conditions), and the second is the **forced response** (convolution integral).

### 5.4 Computing $e^{At}$

**Method 1: Laplace transform**

$$e^{At} = \mathcal{L}^{-1}\{(sI - A)^{-1}\}$$

**Method 2: Diagonalization** (if $A$ is diagonalizable)

$$e^{At} = Te^{\Lambda t}T^{-1} = T \text{diag}(e^{\lambda_1 t}, \ldots, e^{\lambda_n t}) T^{-1}$$

**Method 3: Cayley-Hamilton theorem**

For an $n \times n$ matrix, $e^{At} = \alpha_0(t)I + \alpha_1(t)A + \cdots + \alpha_{n-1}(t)A^{n-1}$, where the coefficients satisfy $e^{\lambda_i t} = \alpha_0 + \alpha_1\lambda_i + \cdots + \alpha_{n-1}\lambda_i^{n-1}$ for each eigenvalue.

**Method 4 (numeric, default for software): scaling and squaring + Padé approximant**, as implemented in `scipy.linalg.expm`. For matrices up to order ~50 it is essentially perfect; beyond that, use Krylov methods. This is the method to use when programming.

```python
from scipy.linalg import expm
import numpy as np

A = np.array([[0, 1], [-2, -3]], dtype=float)
print("e^(A * 0.5) =\n", expm(A * 0.5))
```

## 6. Eigenvalues and Stability

The eigenvalues of $A$ are the poles of the transfer function. The system is:

- **Asymptotically stable:** all eigenvalues have $\text{Re}(\lambda_i) < 0$
- **Marginally stable:** all eigenvalues have $\text{Re}(\lambda_i) \leq 0$ with no repeated eigenvalues on the imaginary axis
- **Unstable:** at least one eigenvalue has $\text{Re}(\lambda_i) > 0$

**Characteristic polynomial:**

$$\det(sI - A) = s^n + a_{n-1}s^{n-1} + \cdots + a_0$$

This is the same characteristic polynomial as in the transfer function approach.

> **Why the state-space test is stricter than the transfer-function test:** if the transfer function has a pole-zero cancellation at $s = +1$, the polynomial $\det(sI - A)$ still has the eigenvalue $\lambda = 1$ — the matrix view does not lose modes. This is the formal reason "internal stability" requires checking eigenvalues of $A$, not just stability of $G(s)$.

## 7. Similarity Transformations

Two state-space realizations $(A, B, C, D)$ and $(\bar{A}, \bar{B}, \bar{C}, \bar{D})$ represent the same transfer function if and only if they are related by a **similarity transformation** $T$:

$$\bar{A} = T^{-1}AT, \quad \bar{B} = T^{-1}B, \quad \bar{C} = CT, \quad \bar{D} = D$$

Key properties preserved under similarity transformations:
- Eigenvalues (poles)
- Transfer function
- Controllability and observability (rank conditions)
- System order

## 8. Common Pitfalls

1. **Confusing "states" with "outputs".** States are internal variables you choose; outputs are what you measure. A 3rd-order system has 3 states but might output only 1 of them. Beginners often write "the output is the state" — true only when $C = I$.
2. **Treating CCF as the unique state-space model.** A given transfer function admits infinitely many state-space realizations. CCF is a useful default for design but the physical form is usually more interpretable for modeling.
3. **Inverting $sI - A$ symbolically when numeric is enough.** For numerical work, $C(sI-A)^{-1}B$ is best computed with `scipy.signal.ss2tf` — it handles ill-conditioned $A$ matrices that hand-inversion mangles.
4. **Mishandling Jordan blocks.** When $A$ has repeated eigenvalues that lack a full set of eigenvectors, naive diagonalization fails silently (you get wrong results). Use `numpy.linalg.eig` and check the rank of the eigenvector matrix; if it is less than $n$, switch to `scipy.linalg.expm` which handles Jordan structure automatically.
5. **Hidden modes after a similarity transform.** If $T$ is poorly conditioned, the transformed system may be numerically uncontrollable or unobservable even though the original was not. Always check controllability/observability after a numeric transform (Lesson 12).
6. **Forgetting that $D \neq 0$ for proper-but-not-strictly-proper transfer functions.** $G(s) = (s+1)/(s+2)$ has DC gain $1/2$ and high-frequency gain $1$ — the difference is encoded in $D = 1$. Setting $D = 0$ silently truncates the high-frequency content.

## Practice Exercises

### Exercise 1: State-Space Modeling

A DC motor has the equations:
- $L_a \frac{di_a}{dt} + R_a i_a + K_b \dot{\theta} = v_a$
- $J\ddot{\theta} + B\dot{\theta} = K_t i_a$

With $x_1 = \theta$, $x_2 = \dot{\theta}$, $x_3 = i_a$:

1. Write the system in state-space form $(A, B, C, D)$ with input $u = v_a$ and output $y = \theta$
2. Find the transfer function $\Theta(s)/V_a(s)$ using $G(s) = C(sI - A)^{-1}B$

### Exercise 2: Canonical Forms

Given $G(s) = \frac{2s + 3}{s^3 + 4s^2 + 5s + 6}$:

1. Write the controllable canonical form
2. Write the observable canonical form
3. Find the eigenvalues of $A$ and verify they match the poles of $G(s)$

### Exercise 3: State Transition Matrix

For the system $A = \begin{bmatrix} 0 & 1 \\ -2 & -3 \end{bmatrix}$:

1. Find the eigenvalues
2. Compute $e^{At}$ using the Laplace transform method
3. Find $x(t)$ for $x(0) = [1 \; 0]^T$ with no input

### Exercise 4: Numerical Round-Trip

Using `tf2ss` and `ss2tf` from `python-control` (or the equivalent in MATLAB), convert $G(s) = \frac{s+1}{(s+2)(s+3)}$ to state space and back. Verify that the recovered transfer function matches the original to within numerical precision. Repeat for $G(s) = \frac{s+1}{(s+2)^2}$ (repeated poles) and discuss what changes.

### Exercise 5: Similarity Drill

Given the SMD system from Section 3.1 with $m = 1, b = 2, k = 5$, transform to diagonal form using the eigenvector matrix $T$. Verify that the transformed $\bar{A}$ is diagonal with the eigenvalues of the original $A$, and that the input/output behavior (transfer function) is unchanged.

---

*Previous: [Lesson 10 — Lead-Lag Compensation](10_Lead_Lag_Compensation.md) | Next: [Lesson 12 — Controllability and Observability](12_Controllability_and_Observability.md)*
