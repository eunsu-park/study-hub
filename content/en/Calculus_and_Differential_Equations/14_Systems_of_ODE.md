# 14. Systems of Ordinary Differential Equations

## Learning Objectives

- Convert higher-order ODEs into equivalent first-order systems
- Solve homogeneous linear systems $\mathbf{X}' = A\mathbf{X}$ using the eigenvalue method
- Classify equilibrium points in the phase plane (node, saddle, spiral, center)
- Analyze stability of linear and nonlinear systems near equilibrium points
- Construct phase portraits and perform eigenvalue analysis using Python

---

## 1. From Higher-Order to First-Order Systems

### 1.1 The Conversion Trick

Any single ODE of order $n$ can be rewritten as a system of $n$ first-order ODEs. This is not just a theoretical convenience -- it is how **all numerical solvers work** (including `solve_ivp`).

**Example:** Convert $y'' + 3y' + 2y = 0$ to a first-order system.

Define new variables:

$$x_1 = y, \quad x_2 = y'$$

Then:

$$x_1' = x_2$$
$$x_2' = y'' = -3y' - 2y = -2x_1 - 3x_2$$

In matrix form:

$$\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ -2 & -3 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

**General pattern:** For $y^{(n)} + a_{n-1}y^{(n-1)} + \cdots + a_1 y' + a_0 y = 0$, set $x_k = y^{(k-1)}$ for $k = 1, \ldots, n$. The last equation expresses $y^{(n)}$ in terms of the others.

### 1.2 Coupled Systems

Many real-world problems naturally involve **coupled** equations -- multiple quantities influencing each other.

**Example: Predator-Prey (Lotka-Volterra)**

$$\frac{dx}{dt} = \alpha x - \beta xy \quad \text{(prey)}$$
$$\frac{dy}{dt} = -\gamma y + \delta xy \quad \text{(predator)}$$

- $x$: prey population, $y$: predator population
- $\alpha$: prey birth rate, $\beta$: predation rate
- $\gamma$: predator death rate, $\delta$: predator reproduction from prey

The two species are coupled: prey growth depends on predator numbers, and vice versa. Neither equation can be solved independently.

---

## 2. Linear Systems: $\mathbf{X}' = A\mathbf{X}$

### 2.1 Matrix Formulation

A **linear homogeneous system** with constant coefficients:

$$\mathbf{X}'(t) = A\mathbf{X}(t)$$

where $\mathbf{X}(t) = \begin{pmatrix} x_1(t) \\ x_2(t) \end{pmatrix}$ and $A$ is a constant matrix.

For a scalar equation $x' = ax$, the solution is $x = Ce^{at}$. By analogy, we guess $\mathbf{X} = \mathbf{v} e^{\lambda t}$ where $\mathbf{v}$ is a constant vector.

### 2.2 The Eigenvalue Method

Substituting $\mathbf{X} = \mathbf{v} e^{\lambda t}$ into $\mathbf{X}' = A\mathbf{X}$:

$$\lambda \mathbf{v} e^{\lambda t} = A\mathbf{v} e^{\lambda t} \implies A\mathbf{v} = \lambda\mathbf{v}$$

This is the **eigenvalue problem**: find eigenvalues $\lambda$ and eigenvectors $\mathbf{v}$ of $A$.

**Steps:**
1. Solve $\det(A - \lambda I) = 0$ for eigenvalues $\lambda_1, \lambda_2$
2. For each $\lambda_i$, find the eigenvector $\mathbf{v}_i$ from $(A - \lambda_i I)\mathbf{v}_i = \mathbf{0}$
3. General solution: $\mathbf{X}(t) = C_1 \mathbf{v}_1 e^{\lambda_1 t} + C_2 \mathbf{v}_2 e^{\lambda_2 t}$

**Example:** $A = \begin{pmatrix} 1 & 3 \\ 1 & -1 \end{pmatrix}$

Characteristic equation: $(1 - \lambda)(-1 - \lambda) - 3 = \lambda^2 - 4 = 0$

Eigenvalues: $\lambda_1 = 2$, $\lambda_2 = -2$

For $\lambda_1 = 2$: $(A - 2I)\mathbf{v} = 0 \implies \mathbf{v}_1 = \begin{pmatrix} 3 \\ 1 \end{pmatrix}$

For $\lambda_2 = -2$: $(A + 2I)\mathbf{v} = 0 \implies \mathbf{v}_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$

General solution:

$$\mathbf{X}(t) = C_1 \begin{pmatrix} 3 \\ 1 \end{pmatrix} e^{2t} + C_2 \begin{pmatrix} 1 \\ -1 \end{pmatrix} e^{-2t}$$

### 2.3 Complex Eigenvalues

When $A$ is real and eigenvalues are $\lambda = \alpha \pm i\beta$, the solution involves oscillation:

$$\mathbf{X}(t) = e^{\alpha t}\left[C_1(\mathbf{a}\cos\beta t - \mathbf{b}\sin\beta t) + C_2(\mathbf{a}\sin\beta t + \mathbf{b}\cos\beta t)\right]$$

where $\mathbf{v} = \mathbf{a} + i\mathbf{b}$ is the (complex) eigenvector for $\lambda = \alpha + i\beta$.

### 2.4 Repeated Eigenvalues

If $\lambda$ is a repeated eigenvalue with only one independent eigenvector $\mathbf{v}$, we need a **generalized eigenvector** $\mathbf{w}$ satisfying $(A - \lambda I)\mathbf{w} = \mathbf{v}$:

$$\mathbf{X}(t) = C_1 \mathbf{v} e^{\lambda t} + C_2 (\mathbf{v}t + \mathbf{w})e^{\lambda t}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Eigenvalue method: analytical vs numerical ---
A = np.array([[1, 3], [1, -1]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Analytical solution with IC x1(0)=1, x2(0)=0
# X(0) = C1*v1 + C2*v2 = (1, 0)
# Solve: C1*(3,1) + C2*(1,-1) = (1,0)
# 3*C1 + C2 = 1, C1 - C2 = 0 => C1 = C2 = 1/4
C1, C2 = 0.25, 0.25
v1 = eigenvectors[:, 0]
v2 = eigenvectors[:, 1]
lam1, lam2 = eigenvalues

t = np.linspace(0, 2, 200)
x1_anal = C1 * v1[0] * np.exp(lam1 * t) + C2 * v2[0] * np.exp(lam2 * t)
x2_anal = C1 * v1[1] * np.exp(lam1 * t) + C2 * v2[1] * np.exp(lam2 * t)

# Numerical solution
def system(t, X):
    return A @ X

sol = solve_ivp(system, (0, 2), [1, 0], t_eval=t)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, x1_anal, 'b-', linewidth=2, label='$x_1$ (analytical)')
ax.plot(t, x2_anal, 'r-', linewidth=2, label='$x_2$ (analytical)')
ax.plot(sol.t, sol.y[0], 'b--', linewidth=2, label='$x_1$ (numerical)')
ax.plot(sol.t, sol.y[1], 'r--', linewidth=2, label='$x_2$ (numerical)')
ax.set_xlabel('Time t')
ax.set_ylabel('$x_i(t)$')
ax.set_title('Solution of $\\mathbf{X}\' = A\\mathbf{X}$ (Saddle Point)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
```

---

## 3. Phase Plane Analysis

### 3.1 The Phase Portrait

For a 2D system $\mathbf{X}' = A\mathbf{X}$, we can plot trajectories in the **phase plane** -- the $(x_1, x_2)$ plane. The behavior of all trajectories near the origin is determined entirely by the eigenvalues.

### 3.2 Classification of Equilibrium Points

| Eigenvalues | Type | Stability | Phase Portrait |
|-------------|------|-----------|----------------|
| $\lambda_1, \lambda_2 > 0$ | Unstable node | Unstable | Trajectories move away from origin |
| $\lambda_1, \lambda_2 < 0$ | Stable node | Asympt. stable | All trajectories approach origin |
| $\lambda_1 > 0 > \lambda_2$ | Saddle | Unstable | Trajectories approach along one eigenvector, diverge along the other |
| $\alpha \pm i\beta$, $\alpha < 0$ | Stable spiral | Asympt. stable | Spirals inward |
| $\alpha \pm i\beta$, $\alpha > 0$ | Unstable spiral | Unstable | Spirals outward |
| $\pm i\beta$ (pure imaginary) | Center | Stable (not asympt.) | Closed elliptical orbits |
| $\lambda_1 = \lambda_2 < 0$ (2 eigenvectors) | Stable star node | Asympt. stable | Straight-line trajectories inward |
| $\lambda_1 = \lambda_2 < 0$ (1 eigenvector) | Stable degenerate node | Asympt. stable | Trajectories tangent to eigenvector |

**Key insight:** The **real part** of eigenvalues determines stability (negative = stable), while the **imaginary part** determines oscillation (nonzero = spirals/oscillations).

### 3.3 Stability Summary

- **Asymptotically stable:** All eigenvalues have **negative real parts** -- trajectories converge to the origin
- **Stable (neutrally):** Eigenvalues are pure imaginary -- trajectories neither grow nor decay
- **Unstable:** At least one eigenvalue has a **positive real part** -- some trajectories diverge

**Connection to the trace-determinant plane:** For $2 \times 2$ matrices, let $\tau = \text{tr}(A)$ and $\Delta = \det(A)$:

- $\Delta < 0$: saddle
- $\Delta > 0$, $\tau^2 - 4\Delta > 0$, $\tau < 0$: stable node
- $\Delta > 0$, $\tau^2 - 4\Delta < 0$, $\tau < 0$: stable spiral
- $\Delta > 0$, $\tau = 0$: center

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def plot_phase_portrait(A, title, ax, xlim=(-3, 3), ylim=(-3, 3)):
    """Plot the phase portrait for X' = AX."""
    # Direction field
    x = np.linspace(xlim[0], xlim[1], 20)
    y = np.linspace(ylim[0], ylim[1], 20)
    X, Y = np.meshgrid(x, y)
    U = A[0, 0] * X + A[0, 1] * Y
    V = A[1, 0] * X + A[1, 1] * Y
    magnitude = np.sqrt(U**2 + V**2)
    magnitude[magnitude == 0] = 1
    ax.quiver(X, Y, U/magnitude, V/magnitude, magnitude,
              cmap='coolwarm', alpha=0.4, scale=25)

    # Trajectories from various initial conditions
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    for r in [0.5, 1.5, 2.5]:
        for theta in angles:
            x0 = r * np.cos(theta)
            y0 = r * np.sin(theta)
            def system(t, state): return A @ state
            sol = solve_ivp(system, (0, 5), [x0, y0],
                           t_eval=np.linspace(0, 5, 300),
                           max_step=0.05)
            ax.plot(sol.y[0], sol.y[1], 'b-', linewidth=0.5, alpha=0.7)

    # Plot eigenvectors for real eigenvalues
    evals, evecs = np.linalg.eig(A)
    if np.all(np.isreal(evals)):
        for i in range(2):
            v = np.real(evecs[:, i])
            v = v / np.linalg.norm(v) * 3
            ax.plot([-v[0], v[0]], [-v[1], v[1]], 'r-', linewidth=2)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Six representative cases
matrices = [
    (np.array([[-1, 0], [0, -2]]),   'Stable Node\n$\\lambda=-1,-2$'),
    (np.array([[1, 0], [0, 2]]),      'Unstable Node\n$\\lambda=1,2$'),
    (np.array([[2, 0], [0, -1]]),     'Saddle Point\n$\\lambda=2,-1$'),
    (np.array([[-0.5, 2], [-2, -0.5]]), 'Stable Spiral\n$\\lambda=-0.5\\pm 2i$'),
    (np.array([[0.3, 2], [-2, 0.3]]),   'Unstable Spiral\n$\\lambda=0.3\\pm 2i$'),
    (np.array([[0, 1], [-1, 0]]),       'Center\n$\\lambda=\\pm i$'),
]

for idx, (A_mat, title) in enumerate(matrices):
    row, col = divmod(idx, 3)
    evals = np.linalg.eigvals(A_mat)
    plot_phase_portrait(A_mat, title, axes[row, col])

plt.tight_layout()
plt.show()
```

---

## 4. Nonlinear Systems and Linearization

### 4.1 Nonlinear Systems

Most real systems are nonlinear:

$$\frac{dx}{dt} = f(x, y), \quad \frac{dy}{dt} = g(x, y)$$

Exact analytical solutions are usually impossible, but we can still understand the **qualitative behavior** near equilibrium points.

### 4.2 Equilibrium Points

An **equilibrium point** (or fixed point) $(x^*, y^*)$ satisfies:

$$f(x^*, y^*) = 0 \quad \text{and} \quad g(x^*, y^*) = 0$$

At an equilibrium, the system is at rest -- all derivatives are zero.

### 4.3 Linearization

Near an equilibrium $(x^*, y^*)$, expand in Taylor series and keep only linear terms:

$$\begin{pmatrix} u' \\ v' \end{pmatrix} \approx J \begin{pmatrix} u \\ v \end{pmatrix}$$

where $u = x - x^*$, $v = y - y^*$, and $J$ is the **Jacobian matrix**:

$$J = \begin{pmatrix} \frac{\partial f}{\partial x} & \frac{\partial f}{\partial y} \\ \frac{\partial g}{\partial x} & \frac{\partial g}{\partial y} \end{pmatrix}_{(x^*, y^*)}$$

**The Hartman-Grobman theorem** guarantees that if the eigenvalues of $J$ have nonzero real parts (the equilibrium is **hyperbolic**), then the behavior of the nonlinear system near the equilibrium is **qualitatively the same** as the linearized system.

**Warning:** For centers ($\lambda = \pm i\beta$), linearization is inconclusive -- the nonlinear terms can turn a center into a spiral.

### 4.4 Example: Lotka-Volterra Predator-Prey

$$\dot{x} = x(1 - y), \quad \dot{y} = y(x - 1)$$

**Equilibria:** $(0, 0)$ and $(1, 1)$

**At $(0, 0)$:** $J = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$, eigenvalues $\lambda = 1, -1$ -- **saddle point** (unstable)

**At $(1, 1)$:** $J = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$, eigenvalues $\lambda = \pm i$ -- **center** (linearization suggests closed orbits). In fact, the full nonlinear system does have closed orbits (a conserved quantity exists).

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Lotka-Volterra Predator-Prey System ---
def lotka_volterra(t, state):
    """dx/dt = x(1-y), dy/dt = y(x-1)."""
    x, y = state
    return [x * (1 - y), y * (x - 1)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Phase portrait
for x0 in np.arange(0.5, 3.5, 0.5):
    for y0 in np.arange(0.5, 3.5, 0.5):
        sol = solve_ivp(lotka_volterra, (0, 20), [x0, y0],
                       t_eval=np.linspace(0, 20, 2000),
                       max_step=0.01)
        axes[0].plot(sol.y[0], sol.y[1], 'b-', linewidth=0.5, alpha=0.5)

# Mark equilibrium point
axes[0].plot(1, 1, 'ro', markersize=10, label='Equilibrium (1,1)')
axes[0].set_xlabel('Prey x')
axes[0].set_ylabel('Predator y')
axes[0].set_title('Lotka-Volterra Phase Portrait')
axes[0].set_xlim(0, 4)
axes[0].set_ylim(0, 4)
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')

# Time series from one initial condition
sol = solve_ivp(lotka_volterra, (0, 30), [2.0, 1.0],
               t_eval=np.linspace(0, 30, 2000), max_step=0.01)

axes[1].plot(sol.t, sol.y[0], 'b-', linewidth=2, label='Prey x(t)')
axes[1].plot(sol.t, sol.y[1], 'r-', linewidth=2, label='Predator y(t)')
axes[1].set_xlabel('Time t')
axes[1].set_ylabel('Population')
axes[1].set_title('Lotka-Volterra Time Series')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# --- Linearization analysis ---
from sympy import symbols, Matrix

x, y = symbols('x y', positive=True)
f = x * (1 - y)
g = y * (x - 1)

J = Matrix([[f.diff(x), f.diff(y)],
            [g.diff(x), g.diff(y)]])

print("Jacobian:")
print(J)

# At equilibrium (1, 1)
J_eq = J.subs([(x, 1), (y, 1)])
print(f"\nJacobian at (1,1):\n{J_eq}")
print(f"Eigenvalues: {J_eq.eigenvals()}")
```

---

## 5. The Damped Pendulum: A Complete Example

The nonlinear pendulum equation $\ddot{\theta} + \beta\dot{\theta} + \omega_0^2\sin\theta = 0$ is a rich example combining everything in this lesson.

Converting to a system: $x_1 = \theta$, $x_2 = \dot{\theta}$:

$$x_1' = x_2$$
$$x_2' = -\omega_0^2\sin x_1 - \beta x_2$$

**Equilibria:** $(n\pi, 0)$ for integer $n$.
- $\theta = 0$ (hanging down): $J = \begin{pmatrix} 0 & 1 \\ -\omega_0^2 & -\beta \end{pmatrix}$ -- **stable** (spiral or node, depending on damping)
- $\theta = \pi$ (balanced upside-down): $J = \begin{pmatrix} 0 & 1 \\ \omega_0^2 & -\beta \end{pmatrix}$ -- **unstable** (saddle)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Damped pendulum phase portrait ---
omega0 = 2.0
beta = 0.3

def pendulum(t, state):
    """theta' = omega, omega' = -omega0^2 * sin(theta) - beta * omega."""
    theta, omega = state
    return [omega, -omega0**2 * np.sin(theta) - beta * omega]

fig, ax = plt.subplots(figsize=(12, 8))

# Trajectories from many initial conditions
for theta0 in np.linspace(-3*np.pi, 3*np.pi, 20):
    for omega0_ic in np.linspace(-8, 8, 8):
        sol = solve_ivp(pendulum, (0, 20), [theta0, omega0_ic],
                       t_eval=np.linspace(0, 20, 1000),
                       max_step=0.05)
        ax.plot(sol.y[0], sol.y[1], 'b-', linewidth=0.3, alpha=0.4)

# Mark equilibria
for n in range(-3, 4):
    if n % 2 == 0:
        ax.plot(n*np.pi, 0, 'go', markersize=8)  # stable
    else:
        ax.plot(n*np.pi, 0, 'rx', markersize=10, markeredgewidth=2)  # unstable

ax.set_xlabel('$\\theta$ (radians)')
ax.set_ylabel('$\\dot{\\theta}$ (rad/s)')
ax.set_title(f'Damped Pendulum Phase Portrait ($\\omega_0={omega0}$, $\\beta={beta}$)\n'
             f'Green circles = stable equilibria, Red crosses = saddle points')
ax.set_xlim(-3*np.pi, 3*np.pi)
ax.set_ylim(-10, 10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 6. The Matrix Exponential (Advanced)

For completeness, the general solution of $\mathbf{X}' = A\mathbf{X}$ can be written compactly as:

$$\mathbf{X}(t) = e^{At}\mathbf{X}(0)$$

where the **matrix exponential** is defined by the power series:

$$e^{At} = I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + \cdots$$

For diagonalizable $A = PDP^{-1}$ (where $D$ is diagonal):

$$e^{At} = P\,e^{Dt}\,P^{-1} = P\begin{pmatrix} e^{\lambda_1 t} & 0 \\ 0 & e^{\lambda_2 t} \end{pmatrix}P^{-1}$$

This connects the eigenvalue method to the matrix exponential framework.

```python
import numpy as np
from scipy.linalg import expm

# --- Matrix exponential solution ---
A = np.array([[1, 3], [1, -1]])
X0 = np.array([1, 0])

t_vals = np.linspace(0, 1, 5)
for t in t_vals:
    X_t = expm(A * t) @ X0
    print(f"t = {t:.2f}: X = [{X_t[0]:.4f}, {X_t[1]:.4f}]")

# Verify: eigendecomposition approach
evals, P = np.linalg.eig(A)
P_inv = np.linalg.inv(P)

t = 1.0
D_exp = np.diag(np.exp(evals * t))
X_eigen = P @ D_exp @ P_inv @ X0
X_expm = expm(A * t) @ X0

print(f"\nAt t=1.0:")
print(f"  Matrix exponential: {X_expm}")
print(f"  Eigendecomposition: {X_eigen}")
print(f"  Match: {np.allclose(X_expm, X_eigen)}")
```

---

## 7. Cross-References

- **Mathematical Methods Lesson 10** covers higher-order systems, boundary value problems, and Sturm-Liouville theory.
- **Lesson 13 (Second-Order ODE)** introduced the spring-mass system whose phase plane analysis is a special case of this lesson.
- **Control Theory Lessons 06-09** use state-space analysis ($\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u}$), which is a direct extension of the linear systems theory presented here, with feedback control.
- **Deep Learning Lesson 34** (Optimizers) relates to gradient flow systems $\dot{\mathbf{x}} = -\nabla f(\mathbf{x})$, where the loss landscape plays the role of a potential.

---

## Practice Problems

**1.** Convert the third-order ODE $y''' - 6y'' + 11y' - 6y = 0$ into a first-order system. Find the eigenvalues and general solution.

**2.** Solve the system $\mathbf{X}' = \begin{pmatrix} 3 & -2 \\ 4 & -1 \end{pmatrix}\mathbf{X}$ with $\mathbf{X}(0) = (1, 1)^T$. Classify the equilibrium and sketch the phase portrait.

**3.** For the competing species model:
   $$\dot{x} = x(3 - x - 2y), \quad \dot{y} = y(2 - y - x)$$
   - (a) Find all equilibrium points.
   - (b) Linearize at each equilibrium and classify stability.
   - (c) Draw the phase portrait numerically and interpret: which species survives?

**4.** A damped pendulum with $\omega_0 = 3$ and $\beta = 0.5$ starts at $\theta(0) = \pi - 0.1$ (nearly upside-down), $\dot{\theta}(0) = 0$.
   - (a) Near which equilibrium will the pendulum end up?
   - (b) Simulate and plot both $\theta(t)$ and the phase portrait trajectory.
   - (c) How does the behavior change if $\beta = 0$ (no damping)?

**5.** Use `scipy.linalg.expm` to compute $e^{At}$ for $A = \begin{pmatrix} 0 & 1 \\ -4 & 0 \end{pmatrix}$ at $t = 0, \pi/4, \pi/2, \pi$. Verify that the solutions form closed orbits (center). What is the period?

---

## References

- **William E. Boyce & Richard C. DiPrima**, *Elementary Differential Equations*, 11th Edition, Chapters 7-9
- **Steven H. Strogatz**, *Nonlinear Dynamics and Chaos*, 2nd Edition (excellent for qualitative analysis and applications)
- **Erwin Kreyszig**, *Advanced Engineering Mathematics*, 10th Edition, Chapter 4
- **Lawrence Perko**, *Differential Equations and Dynamical Systems*, 3rd Edition
- **SciPy expm**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html

---

[Previous: Second-Order Ordinary Differential Equations](./13_Second_Order_ODE.md) | [Next: Laplace Transform for ODE](./15_Laplace_Transform_for_ODE.md)
