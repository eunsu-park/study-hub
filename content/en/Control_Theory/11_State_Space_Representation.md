# Lesson 11: State-Space Representation

## Learning Objectives

- Represent dynamic systems in state-space form
- Convert between transfer functions and state-space models
- Identify controllable canonical, observable canonical, and diagonal (modal) forms
- Understand the relationship between state-space and transfer function representations
- Compute the state transition matrix and solve state equations

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

## 6. Eigenvalues and Stability

The eigenvalues of $A$ are the poles of the transfer function. The system is:

- **Asymptotically stable:** all eigenvalues have $\text{Re}(\lambda_i) < 0$
- **Marginally stable:** all eigenvalues have $\text{Re}(\lambda_i) \leq 0$ with no repeated eigenvalues on the imaginary axis
- **Unstable:** at least one eigenvalue has $\text{Re}(\lambda_i) > 0$

**Characteristic polynomial:**

$$\det(sI - A) = s^n + a_{n-1}s^{n-1} + \cdots + a_0$$

This is the same characteristic polynomial as in the transfer function approach.

## 7. Similarity Transformations

Two state-space realizations $(A, B, C, D)$ and $(\bar{A}, \bar{B}, \bar{C}, \bar{D})$ represent the same transfer function if and only if they are related by a **similarity transformation** $T$:

$$\bar{A} = T^{-1}AT, \quad \bar{B} = T^{-1}B, \quad \bar{C} = CT, \quad \bar{D} = D$$

Key properties preserved under similarity transformations:
- Eigenvalues (poles)
- Transfer function
- Controllability and observability (rank conditions)
- System order

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

---

*Previous: [Lesson 10 — Lead-Lag Compensation](10_Lead_Lag_Compensation.md) | Next: [Lesson 12 — Controllability and Observability](12_Controllability_and_Observability.md)*
