# 10. Higher-Order ODE and Systems

## Learning Objectives

- Formulate the **characteristic equation** for **nth-order constant-coefficient linear ODEs** and find the general solution
- Find particular solutions for non-homogeneous ODEs using the **method of undetermined coefficients** and **variation of parameters**
- Express systems of ODEs in **vector-matrix form** and solve using **eigenvalues/eigenvectors** and **matrix exponentials**
- Classify equilibrium points in the **phase plane** and determine **stability**
- Apply linearization techniques to **nonlinear systems** and analyze representative physical/biological models
- Find normal modes of **coupled oscillators** and understand their connection to Lagrangian mechanics

---

## 1. Higher-Order Linear ODEs

### 1.1 nth-Order Constant-Coefficient ODEs

The general form of an nth-order constant-coefficient linear ODE is:

$$a_n y^{(n)} + a_{n-1} y^{(n-1)} + \cdots + a_1 y' + a_0 y = f(x)$$

where $a_0, a_1, \ldots, a_n$ are constants. If $f(x) = 0$, it's **homogeneous**, and if $f(x) \neq 0$, it's **non-homogeneous**.

**Key Principle - Superposition**: Solutions of homogeneous equations form a **linear space**, so the **general solution** with $n$ linearly independent solutions $y_1, y_2, \ldots, y_n$ is:

$$y_h = c_1 y_1 + c_2 y_2 + \cdots + c_n y_n$$

The general solution of a non-homogeneous equation is:

$$y = y_h + y_p$$

where $y_p$ is one particular solution.

### 1.2 Characteristic Equation and General Solution

Assuming a solution of the form $y = e^{rx}$ for the homogeneous equation gives the **characteristic equation**:

$$a_n r^n + a_{n-1} r^{n-1} + \cdots + a_1 r + a_0 = 0$$

The general solution is determined by the types of characteristic roots:

| Type of Characteristic Roots | Solution Form |
|---|---|
| Distinct real roots $r_1, r_2, \ldots, r_n$ | $c_1 e^{r_1 x} + c_2 e^{r_2 x} + \cdots$ |
| Repeated root $r$ (multiplicity $m$) | $(c_1 + c_2 x + \cdots + c_m x^{m-1}) e^{rx}$ |
| Complex roots $\alpha \pm i\beta$ | $e^{\alpha x}(c_1 \cos\beta x + c_2 \sin\beta x)$ |
| Repeated complex roots ($\alpha \pm i\beta$, multiplicity $m$) | $e^{\alpha x}\sum_{k=0}^{m-1} x^k (a_k \cos\beta x + b_k \sin\beta x)$ |

**Example**: 4th-order ODE $y^{(4)} - 5y'' + 4y = 0$

```python
import numpy as np
import sympy as sp

# --- Solve characteristic equation with SymPy ---
r = sp.Symbol('r')
char_eq = r**4 - 5*r**2 + 4  # characteristic equation
roots = sp.solve(char_eq, r)
print(f"Characteristic roots: {roots}")
# Output: Characteristic roots: [-2, -1, 1, 2]

# --- Find general solution ---
x = sp.Symbol('x')
y = sp.Function('y')
ode = sp.Eq(y(x).diff(x, 4) - 5*y(x).diff(x, 2) + 4*y(x), 0)
general_sol = sp.dsolve(ode, y(x))
print(f"General solution: {general_sol}")
# Output: y(x) = C1*exp(-2*x) + C2*exp(-x) + C3*exp(x) + C4*exp(2*x)
```

**Example with repeated roots**: $y''' - 3y'' + 3y' - y = 0$

Characteristic equation $r^3 - 3r^2 + 3r - 1 = (r-1)^3 = 0$, so $r = 1$ (multiplicity 3)

$$y = (c_1 + c_2 x + c_3 x^2) e^x$$

```python
# 3rd-order ODE with repeated roots
ode2 = sp.Eq(y(x).diff(x, 3) - 3*y(x).diff(x, 2) + 3*y(x).diff(x) - y(x), 0)
sol2 = sp.dsolve(ode2, y(x))
print(f"Repeated-root general solution: {sol2}")
# Output: y(x) = (C1 + C2*x + C3*x**2)*exp(x)
```

### 1.3 Particular Solutions for Non-Homogeneous Problems

#### Method of Undetermined Coefficients

Applicable when $f(x)$ consists of polynomials, exponentials, trigonometric functions, or their products.

| Form of $f(x)$ | Assumed form of $y_p$ |
|---|---|
| $P_n(x)$ (nth-degree polynomial) | $A_n x^n + A_{n-1} x^{n-1} + \cdots + A_0$ |
| $e^{\alpha x}$ | $A e^{\alpha x}$ |
| $\cos\beta x$ or $\sin\beta x$ | $A \cos\beta x + B \sin\beta x$ |
| $e^{\alpha x} P_n(x)$ | $e^{\alpha x}(A_n x^n + \cdots + A_0)$ |

If $\alpha$ is a characteristic root, multiply by $x^s$ ($s$ is the multiplicity of $\alpha$).

#### Variation of Parameters

A general method applicable for any form of $f(x)$. For second-order ODE $y'' + p(x)y' + q(x)y = f(x)$:

$$y_p = -y_1 \int \frac{y_2 f}{W} dx + y_2 \int \frac{y_1 f}{W} dx$$

where $W = y_1 y_2' - y_2 y_1'$ is the **Wronskian**.

```python
# --- Non-homogeneous ODE: y'' + y = sec(x) ---
# Cannot be solved by undetermined coefficients → use variation of parameters
x = sp.Symbol('x')
y = sp.Function('y')

ode_nh = sp.Eq(y(x).diff(x, 2) + y(x), sp.sec(x))
sol_nh = sp.dsolve(ode_nh, y(x))
print(f"Variation of parameters result: {sol_nh}")

# Direct Wronskian calculation
y1 = sp.cos(x)
y2 = sp.sin(x)
W = y1 * sp.diff(y2, x) - y2 * sp.diff(y1, x)
print(f"Wronskian: W = {sp.simplify(W)}")
# Output: W = 1

# Particular solution calculation
f_x = sp.sec(x)
yp = -y1 * sp.integrate(y2 * f_x / W, x) + y2 * sp.integrate(y1 * f_x / W, x)
yp_simplified = sp.simplify(yp)
print(f"Particular solution: y_p = {yp_simplified}")
```

---

## 2. Systems of ODEs

### 2.1 Vector-Matrix Representation

A system of first-order ODEs is expressed concisely in vector-matrix form:

$$\frac{d\mathbf{x}}{dt} = A\mathbf{x} + \mathbf{f}(t)$$

where $\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}$, and $A$ is an $n \times n$ coefficient matrix.

**Important**: Any **nth-order ODE** can be converted to a system of first-order ODEs. For example, $y'' + 3y' + 2y = 0$ becomes:

$$x_1 = y, \quad x_2 = y'$$

$$\frac{d}{dt}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ -2 & -3 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Convert 2nd-order ODE to system of 1st-order ODEs and solve numerically ---
# y'' + 3y' + 2y = 0, y(0) = 1, y'(0) = 0
def system(t, x):
    """x[0] = y, x[1] = y'"""
    return [x[1], -2*x[0] - 3*x[1]]

t_span = (0, 5)
x0 = [1.0, 0.0]
sol = solve_ivp(system, t_span, x0, t_eval=np.linspace(0, 5, 200), method='RK45')

# Compare with analytical solution
t_exact = np.linspace(0, 5, 200)
# Characteristic roots: r = -1, -2 → y = 2e^{-t} - e^{-2t}
y_exact = 2*np.exp(-t_exact) - np.exp(-2*t_exact)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(sol.t, sol.y[0], 'b-', label='Numerical solution (RK45)')
axes[0].plot(t_exact, y_exact, 'r--', label='Analytical solution')
axes[0].set_xlabel('t')
axes[0].set_ylabel('y(t)')
axes[0].set_title('Solution Comparison')
axes[0].legend()

axes[1].plot(sol.y[0], sol.y[1], 'b-')
axes[1].set_xlabel('y')
axes[1].set_ylabel("y'")
axes[1].set_title('Phase Plane Trajectory')
axes[1].plot(x0[0], x0[1], 'ro', markersize=8, label='Initial value')
axes[1].legend()

plt.tight_layout()
plt.savefig('second_order_ode_solution.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 2.2 Solution Using Eigenvalues/Eigenvectors

For the homogeneous system $\mathbf{x}' = A\mathbf{x}$, assuming a solution $\mathbf{x} = \mathbf{v} e^{\lambda t}$ gives:

$$A\mathbf{v} = \lambda \mathbf{v}$$

This reduces to finding the **eigenvalues** $\lambda$ and **eigenvectors** $\mathbf{v}$ of $A$.

**Case 1: Distinct real eigenvalues** $\lambda_1, \lambda_2$

$$\mathbf{x}(t) = c_1 \mathbf{v}_1 e^{\lambda_1 t} + c_2 \mathbf{v}_2 e^{\lambda_2 t}$$

**Case 2: Complex eigenvalues** $\lambda = \alpha \pm i\beta$

$$\mathbf{x}(t) = e^{\alpha t}\left[c_1(\mathbf{a}\cos\beta t - \mathbf{b}\sin\beta t) + c_2(\mathbf{a}\sin\beta t + \mathbf{b}\cos\beta t)\right]$$

where $\mathbf{v} = \mathbf{a} + i\mathbf{b}$.

**Case 3: Repeated eigenvalue** ($\lambda$, multiplicity 2, 1 eigenvector) - requires generalized eigenvector $\mathbf{w}$:

$$(A - \lambda I)\mathbf{w} = \mathbf{v}$$

$$\mathbf{x}(t) = c_1 \mathbf{v} e^{\lambda t} + c_2 (\mathbf{v} t + \mathbf{w}) e^{\lambda t}$$

```python
import numpy as np
from scipy.linalg import eig

# --- Eigenvalue/eigenvector solution for system of ODEs ---
# x' = Ax, A = [[1, 1], [4, -2]]
A = np.array([[1, 1], [4, -2]])
eigenvalues, eigenvectors = eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors (column vectors):")
print(eigenvectors)
# Eigenvalues: [2, -3]
# Eigenvectors: v1 = [1, 1], v2 = [1, -4] (normalized)

# Construct general solution
t = np.linspace(0, 3, 200)
# Determine c1, c2 from initial condition x(0) = [3, 2]
x0 = np.array([3, 2])
# c1*v1 + c2*v2 = x0 → [c1, c2] = V^{-1} x0
V = eigenvectors
c = np.linalg.solve(V, x0)
print(f"Coefficients: c1 = {c[0]:.4f}, c2 = {c[1]:.4f}")

# Compute analytical solution
x1_sol = c[0] * V[0, 0] * np.exp(eigenvalues[0].real * t) + \
         c[1] * V[0, 1] * np.exp(eigenvalues[1].real * t)
x2_sol = c[0] * V[1, 0] * np.exp(eigenvalues[0].real * t) + \
         c[1] * V[1, 1] * np.exp(eigenvalues[1].real * t)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Solution over time
axes[0].plot(t, x1_sol.real, 'b-', label='$x_1(t)$')
axes[0].plot(t, x2_sol.real, 'r-', label='$x_2(t)$')
axes[0].set_xlabel('t')
axes[0].set_title('Solution of System of ODEs')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Phase plane
axes[1].plot(x1_sol.real, x2_sol.real, 'b-', linewidth=2)
axes[1].plot(x0[0], x0[1], 'ro', markersize=8, label='Initial value')
axes[1].set_xlabel('$x_1$')
axes[1].set_ylabel('$x_2$')
axes[1].set_title('Phase Plane Trajectory')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('system_ode_eigenvalue.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 2.3 Matrix Exponential

The solution of the system $\mathbf{x}' = A\mathbf{x}$, $\mathbf{x}(0) = \mathbf{x}_0$ is expressed using the **matrix exponential**:

$$\mathbf{x}(t) = e^{At} \mathbf{x}_0$$

The matrix exponential extends the series definition of the scalar exponential function to matrices:

$$e^{At} = I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + \cdots = \sum_{k=0}^{\infty} \frac{(At)^k}{k!}$$

**Properties:**
- $e^{0} = I$ (identity matrix)
- $\frac{d}{dt} e^{At} = A e^{At}$
- If $A$ is diagonalizable: $A = PDP^{-1}$ → $e^{At} = P e^{Dt} P^{-1}$

```python
from scipy.linalg import expm

# --- Solve system of ODEs using matrix exponential ---
A = np.array([[1, 1], [4, -2]])
x0 = np.array([3, 2])

t_vals = np.linspace(0, 3, 200)
solutions = np.array([expm(A * t) @ x0 for t in t_vals])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_vals, solutions[:, 0], 'b-', linewidth=2, label='$x_1(t)$')
ax.plot(t_vals, solutions[:, 1], 'r-', linewidth=2, label='$x_2(t)$')
ax.set_xlabel('t', fontsize=12)
ax.set_ylabel('x(t)', fontsize=12)
ax.set_title('Solution Using Matrix Exponential', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('matrix_exponential.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Compute symbolic matrix exponential with SymPy ---
t_sym = sp.Symbol('t')
A_sym = sp.Matrix([[1, 1], [4, -2]])
exp_At = sp.exp(A_sym * t_sym)  # symbolic matrix exponential
print("e^{At} =")
sp.pprint(exp_At)
```

**Non-homogeneous system** $\mathbf{x}' = A\mathbf{x} + \mathbf{f}(t)$ has solution:

$$\mathbf{x}(t) = e^{At}\mathbf{x}_0 + \int_0^t e^{A(t-s)} \mathbf{f}(s) \, ds$$

This is called **Duhamel's integral** or the **variation of constants formula**.

---

## 3. Phase Plane Analysis

For a 2D autonomous system $\mathbf{x}' = A\mathbf{x}$, the eigenvalues of $A$ determine the qualitative behavior of the system.

### 3.1 Classification of Equilibrium Points (Nodes, Saddles, Spirals, Centers)

An **equilibrium point** $\mathbf{x}^*$ satisfies $A\mathbf{x}^* = \mathbf{0}$. If $A$ is invertible, the origin $\mathbf{x}^* = \mathbf{0}$ is the unique equilibrium.

Classification by eigenvalues $\lambda_1, \lambda_2$ of $A$:

| Type of Eigenvalues | Equilibrium Type | Stability |
|---|---|---|
| $\lambda_1 < \lambda_2 < 0$ (real, negative) | **Stable node** | Asymptotically stable |
| $\lambda_1 > \lambda_2 > 0$ (real, positive) | **Unstable node** | Unstable |
| $\lambda_1 < 0 < \lambda_2$ (real, opposite signs) | **Saddle point** | Unstable |
| $\alpha \pm i\beta$, $\alpha < 0$ | **Stable spiral** | Asymptotically stable |
| $\alpha \pm i\beta$, $\alpha > 0$ | **Unstable spiral** | Unstable |
| $\pm i\beta$ (purely imaginary) | **Center** | Stable (not asymptotically) |

**Trace-determinant plane**: Using $\tau = \text{tr}(A) = \lambda_1 + \lambda_2$, $\Delta = \det(A) = \lambda_1 \lambda_2$:
- $\Delta < 0$: Saddle point
- $\Delta > 0$, $\tau < 0$: Stable (node or spiral)
- $\Delta > 0$, $\tau > 0$: Unstable (node or spiral)
- $\Delta > 0$, $\tau = 0$: Center
- $\tau^2 - 4\Delta > 0$: Node, $\tau^2 - 4\Delta < 0$: Spiral

### 3.2 Stability Criteria

**Definition**: An equilibrium point $\mathbf{x}^*$ is
- **Stable**: For all $\epsilon > 0$, if $\|\mathbf{x}(0) - \mathbf{x}^*\| < \delta$, then $\|\mathbf{x}(t) - \mathbf{x}^*\| < \epsilon$ for all $t > 0$
- **Asymptotically stable**: Stable and $\mathbf{x}(t) \to \mathbf{x}^*$ as $t \to \infty$
- **Unstable**: Not stable

**Stability criteria for linear systems**:
- All eigenvalues have negative real parts → **Asymptotically stable**
- At least one eigenvalue has positive real part → **Unstable**
- All eigenvalues have real parts ≤ 0, with at least one zero → Additional analysis needed

### 3.3 Drawing Phase Portraits

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def plot_phase_portrait(A, title, ax, xlim=(-3, 3), ylim=(-3, 3)):
    """Draw phase portrait for a 2D linear system."""
    # Vector field (streamplot)
    x1 = np.linspace(xlim[0], xlim[1], 20)
    x2 = np.linspace(ylim[0], ylim[1], 20)
    X1, X2 = np.meshgrid(x1, x2)
    U = A[0, 0] * X1 + A[0, 1] * X2
    V = A[1, 0] * X1 + A[1, 1] * X2

    ax.streamplot(X1, X2, U, V, color='steelblue', density=1.5, linewidth=0.8,
                  arrowsize=1.2)

    # Add trajectories from multiple initial conditions
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        r0 = 2.5
        ic = [r0 * np.cos(angle), r0 * np.sin(angle)]

        def rhs(t, x):
            return A @ x

        sol = solve_ivp(rhs, [0, 10], ic, t_eval=np.linspace(0, 10, 500),
                        method='RK45')
        ax.plot(sol.y[0], sol.y[1], 'b-', alpha=0.4, linewidth=0.8)

    # Eigenvalues/eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    eigval_str = ", ".join([f"{ev:.2f}" for ev in eigenvalues])
    ax.set_title(f"{title}\n$\\lambda = {eigval_str}$", fontsize=11)

    # Show eigenvector directions (for real eigenvalues)
    for i in range(2):
        if np.isreal(eigenvalues[i]):
            v = eigenvectors[:, i].real
            ax.arrow(0, 0, v[0]*1.5, v[1]*1.5, head_width=0.15,
                     head_length=0.1, fc='red', ec='red', linewidth=1.5)
            ax.arrow(0, 0, -v[0]*1.5, -v[1]*1.5, head_width=0.15,
                     head_length=0.1, fc='red', ec='red', linewidth=1.5)

    ax.plot(0, 0, 'ko', markersize=6)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


# --- Phase portraits for 4 equilibrium point types ---
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# (a) Stable node: all eigenvalues negative
A_stable_node = np.array([[-2, 0], [0, -1]])
plot_phase_portrait(A_stable_node, 'Stable Node', axes[0, 0])

# (b) Saddle point: eigenvalues of opposite signs
A_saddle = np.array([[1, 0], [0, -2]])
plot_phase_portrait(A_saddle, 'Saddle Point', axes[0, 1])

# (c) Stable spiral: complex eigenvalues, negative real part
A_spiral = np.array([[-0.5, 2], [-2, -0.5]])
plot_phase_portrait(A_spiral, 'Stable Spiral', axes[1, 0])

# (d) Center: purely imaginary eigenvalues
A_center = np.array([[0, 1], [-4, 0]])
plot_phase_portrait(A_center, 'Center', axes[1, 1])

plt.tight_layout()
plt.savefig('phase_portraits.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 4. Introduction to Nonlinear Systems

### 4.1 Linearization

The behavior near an equilibrium point $\mathbf{x}^*$ of a nonlinear autonomous system $\mathbf{x}' = \mathbf{F}(\mathbf{x})$ is analyzed through linearization using the **Jacobian matrix**:

$$\mathbf{x}' \approx J(\mathbf{x}^*) (\mathbf{x} - \mathbf{x}^*)$$

where the Jacobian matrix is:

$$J = \begin{pmatrix} \frac{\partial F_1}{\partial x_1} & \frac{\partial F_1}{\partial x_2} \\ \frac{\partial F_2}{\partial x_1} & \frac{\partial F_2}{\partial x_2} \end{pmatrix}_{\mathbf{x} = \mathbf{x}^*}$$

**Hartman-Grobman theorem**: If an equilibrium point is **hyperbolic** (i.e., all eigenvalues of the Jacobian have nonzero real parts), the topological behavior of the nonlinear system is **topologically equivalent** to the linearized system.

> **Caution**: Centers are not hyperbolic, so linearization alone cannot determine the exact behavior of the nonlinear system.

### 4.2 Lotka-Volterra Equations

A classical nonlinear system modeling predator-prey interaction:

$$\frac{dx}{dt} = \alpha x - \beta x y \quad \text{(prey)}$$

$$\frac{dy}{dt} = -\gamma y + \delta x y \quad \text{(predator)}$$

**Equilibrium points**:
1. $(0, 0)$ - Extinction (trivial)
2. $(\gamma/\delta, \alpha/\beta)$ - Coexistence

Jacobian at the coexistence equilibrium:

$$J = \begin{pmatrix} 0 & -\beta\gamma/\delta \\ \delta\alpha/\beta & 0 \end{pmatrix}$$

Eigenvalues are $\lambda = \pm i\sqrt{\alpha\gamma}$ (purely imaginary), so linear analysis gives a **center**. Nonlinear analysis also confirms closed orbits (periodic motion) due to conserved quantity $V = \delta x - \gamma \ln x + \beta y - \alpha \ln y$.

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Lotka-Volterra simulation ---
alpha, beta, gamma, delta = 1.0, 0.5, 0.75, 0.25

def lotka_volterra(t, z):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = -gamma * y + delta * x * y
    return [dxdt, dydt]

# Multiple initial conditions
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

initial_conditions = [[2, 1], [4, 2], [1, 3], [6, 1]]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for ic, color in zip(initial_conditions, colors):
    sol = solve_ivp(lotka_volterra, [0, 30], ic,
                    t_eval=np.linspace(0, 30, 1000), method='RK45',
                    rtol=1e-10, atol=1e-12)

    # Time domain
    axes[0].plot(sol.t, sol.y[0], '-', color=color, alpha=0.8,
                 label=f'Prey ({ic[0]},{ic[1]})')
    axes[0].plot(sol.t, sol.y[1], '--', color=color, alpha=0.8,
                 label=f'Predator ({ic[0]},{ic[1]})')

    # Phase plane
    axes[1].plot(sol.y[0], sol.y[1], '-', color=color, linewidth=1.5,
                 label=f'IC=({ic[0]},{ic[1]})')
    axes[1].plot(ic[0], ic[1], 'o', color=color, markersize=6)

# Mark equilibrium point
x_eq, y_eq = gamma / delta, alpha / beta
axes[1].plot(x_eq, y_eq, 'k*', markersize=15, zorder=5, label='Equilibrium')

axes[0].set_xlabel('Time t')
axes[0].set_ylabel('Population')
axes[0].set_title('Lotka-Volterra: Time Domain')
axes[0].legend(fontsize=7, ncol=2)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Prey x')
axes[1].set_ylabel('Predator y')
axes[1].set_title('Lotka-Volterra: Phase Plane')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# Vector field (streamplot)
x_range = np.linspace(0.2, 8, 20)
y_range = np.linspace(0.2, 5, 20)
X, Y = np.meshgrid(x_range, y_range)
U = alpha * X - beta * X * Y
V = -gamma * Y + delta * X * Y
speed = np.sqrt(U**2 + V**2)

axes[2].streamplot(X, Y, U, V, color=speed, cmap='coolwarm', density=1.5,
                   linewidth=0.8)
axes[2].plot(x_eq, y_eq, 'k*', markersize=15, label='Equilibrium')
axes[2].set_xlabel('Prey x')
axes[2].set_ylabel('Predator y')
axes[2].set_title('Lotka-Volterra: Vector Field')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lotka_volterra.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 4.3 Van der Pol Oscillator

An oscillator with nonlinear damping, widely used in electronics and biological rhythm modeling:

$$\ddot{x} - \mu(1 - x^2)\dot{x} + x = 0$$

- When $|x| < 1$: Negative damping (energy supply) → Amplitude increases
- When $|x| > 1$: Positive damping (energy dissipation) → Amplitude decreases

This competition leads to a **limit cycle** when $\mu > 0$ (around $|x| \approx 2$).

System form: $x_1 = x$, $x_2 = \dot{x}$

$$\dot{x}_1 = x_2, \quad \dot{x}_2 = \mu(1 - x_1^2) x_2 - x_1$$

Jacobian at the origin $(0, 0)$:

$$J = \begin{pmatrix} 0 & 1 \\ -1 & \mu \end{pmatrix}$$

If $\mu > 0$, then $\text{tr}(J) = \mu > 0$, so the origin is **unstable**.

```python
# --- Van der Pol oscillator simulation ---
def van_der_pol(t, z, mu):
    x, xdot = z
    return [xdot, mu * (1 - x**2) * xdot - x]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
mu_values = [0.1, 1.0, 5.0]

for idx, mu in enumerate(mu_values):
    # Multiple initial conditions
    ics = [[0.1, 0], [4, 0], [0, 5], [2, -3]]

    for ic in ics:
        sol = solve_ivp(van_der_pol, [0, 50], ic,
                        args=(mu,), t_eval=np.linspace(0, 50, 2000),
                        method='RK45', rtol=1e-10, atol=1e-12)

        # Time domain
        axes[0, idx].plot(sol.t, sol.y[0], linewidth=0.8, alpha=0.8)
        # Phase plane
        axes[1, idx].plot(sol.y[0], sol.y[1], linewidth=0.8, alpha=0.8)

    axes[0, idx].set_xlabel('t')
    axes[0, idx].set_ylabel('x(t)')
    axes[0, idx].set_title(f'Van der Pol ($\\mu$ = {mu}): Time Domain')
    axes[0, idx].grid(True, alpha=0.3)

    axes[1, idx].set_xlabel('x')
    axes[1, idx].set_ylabel('$\\dot{x}$')
    axes[1, idx].set_title(f'Van der Pol ($\\mu$ = {mu}): Phase Plane')
    axes[1, idx].plot(0, 0, 'ro', markersize=5)
    axes[1, idx].grid(True, alpha=0.3)
    axes[1, idx].set_aspect('equal')

plt.tight_layout()
plt.savefig('van_der_pol.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Observations**:
- $\mu \to 0$: Nearly sinusoidal oscillation (weak nonlinearity)
- $\mu = 1$: Clear convergence to limit cycle
- $\mu \gg 1$: **Relaxation oscillation** - Alternating slow/fast segments

---

## 5. Physics Applications

### 5.1 Coupled Oscillators

Consider the motion of two masses connected by springs:

```
Wall ─── k ─── [m₁] ─── k_c ─── [m₂] ─── k ─── Wall
```

Newton's equations of motion:

$$m_1 \ddot{x}_1 = -k x_1 - k_c (x_1 - x_2)$$

$$m_2 \ddot{x}_2 = -k x_2 - k_c (x_2 - x_1)$$

For the symmetric case $m_1 = m_2 = m$, in matrix form:

$$m \begin{pmatrix} \ddot{x}_1 \\ \ddot{x}_2 \end{pmatrix} = -\begin{pmatrix} k + k_c & -k_c \\ -k_c & k + k_c \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

### 5.2 Normal Modes

Normal modes are special motion patterns where all particles oscillate at the **same frequency**.

Substituting $\mathbf{x}(t) = \mathbf{u} e^{i\omega t}$ gives an **eigenvalue problem**:

$$K\mathbf{u} = \omega^2 M\mathbf{u}$$

or $M^{-1}K\mathbf{u} = \omega^2 \mathbf{u}$

Normal modes for the symmetric case:

| Mode | Frequency | Pattern | Physical Meaning |
|---|---|---|---|
| Mode 1 (in-phase) | $\omega_1 = \sqrt{k/m}$ | $\mathbf{u}_1 = (1, 1)^T$ | Both masses move together |
| Mode 2 (out-of-phase) | $\omega_2 = \sqrt{(k + 2k_c)/m}$ | $\mathbf{u}_2 = (1, -1)^T$ | Masses move opposite |

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Coupled oscillators and normal modes ---
m = 1.0     # mass
k = 1.0     # wall spring constant
kc = 0.5    # coupling spring constant

# Stiffness matrix and mass matrix
K = np.array([[k + kc, -kc],
              [-kc, k + kc]])
M = np.array([[m, 0],
              [0, m]])

# Normal modes (generalized eigenvalue problem)
from scipy.linalg import eigh
omega_sq, modes = eigh(K, M)
omega = np.sqrt(omega_sq)

print("Normal frequencies:")
for i, w in enumerate(omega):
    print(f"  omega_{i+1} = {w:.4f} rad/s (f = {w/(2*np.pi):.4f} Hz)")
print(f"\nNormal mode vectors:")
print(f"  Mode 1 (in-phase): {modes[:, 0]}")
print(f"  Mode 2 (out-of-phase): {modes[:, 1]}")

# Simulation: only mass 1 displaced initially
def coupled_oscillator(t, z):
    x1, x2, v1, v2 = z
    a1 = (-k * x1 - kc * (x1 - x2)) / m
    a2 = (-k * x2 - kc * (x2 - x1)) / m
    return [v1, v2, a1, a2]

# Initial condition: x1(0) = 1, x2(0) = 0 (to observe beating)
sol = solve_ivp(coupled_oscillator, [0, 60], [1, 0, 0, 0],
                t_eval=np.linspace(0, 60, 2000), method='RK45',
                rtol=1e-12, atol=1e-14)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Displacement of each mass
axes[0].plot(sol.t, sol.y[0], 'b-', linewidth=1, label='$x_1(t)$ (mass 1)')
axes[0].plot(sol.t, sol.y[1], 'r-', linewidth=1, label='$x_2(t)$ (mass 2)')
axes[0].set_xlabel('Time t')
axes[0].set_ylabel('Displacement')
axes[0].set_title('Coupled Oscillators: Energy Transfer (Beating)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Normal coordinates
q1 = (sol.y[0] + sol.y[1]) / np.sqrt(2)  # in-phase mode
q2 = (sol.y[0] - sol.y[1]) / np.sqrt(2)  # out-of-phase mode

axes[1].plot(sol.t, q1, 'g-', linewidth=1, label='$q_1(t)$ (in-phase mode)')
axes[1].plot(sol.t, q2, 'm-', linewidth=1, label='$q_2(t)$ (out-of-phase mode)')
axes[1].set_xlabel('Time t')
axes[1].set_ylabel('Normal coordinates')
axes[1].set_title('Normal Coordinates: Independent Oscillation of Each Mode')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Energy exchange
E1 = 0.5 * m * sol.y[2]**2 + 0.5 * k * sol.y[0]**2
E2 = 0.5 * m * sol.y[3]**2 + 0.5 * k * sol.y[1]**2
E_coupling = 0.5 * kc * (sol.y[0] - sol.y[1])**2

axes[2].plot(sol.t, E1, 'b-', linewidth=1, label='Mass 1 energy')
axes[2].plot(sol.t, E2, 'r-', linewidth=1, label='Mass 2 energy')
axes[2].plot(sol.t, E1 + E2 + E_coupling, 'k--', linewidth=0.8,
             label='Total energy', alpha=0.5)
axes[2].set_xlabel('Time t')
axes[2].set_ylabel('Energy')
axes[2].set_title('Energy Exchange: Beat frequency = $|\\omega_2 - \\omega_1|/2$')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('coupled_oscillators.png', dpi=150, bbox_inches='tight')
plt.show()

# Beat frequency calculation
omega_beat = abs(omega[1] - omega[0]) / 2
T_beat = 2 * np.pi / omega_beat if omega_beat > 0 else float('inf')
print(f"\nBeat frequency: {omega_beat:.4f} rad/s")
print(f"Beat period: {T_beat:.2f} s")
```

### 5.3 Systems of ODEs from Lagrangian Mechanics

The **double pendulum** is a representative nonlinear system from Lagrangian mechanics.

From the Lagrangian $L = T - V$, the Euler-Lagrange equations are:

$$\frac{d}{dt} \frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = 0$$

**Small angle approximation** linearizes the double pendulum equations:

$$\begin{pmatrix} (m_1 + m_2) l_1 & m_2 l_2 \\ l_1 & l_2 \end{pmatrix} \begin{pmatrix} \ddot{\theta}_1 \\ \ddot{\theta}_2 \end{pmatrix} = -g \begin{pmatrix} (m_1 + m_2) \theta_1 \\ \theta_2 \end{pmatrix}$$

This reduces to a generalized eigenvalue problem $K\mathbf{u} = \omega^2 M\mathbf{u}$.

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Double pendulum (full nonlinear equations) ---
g = 9.81
m1, m2 = 1.0, 1.0
l1, l2 = 1.0, 1.0

def double_pendulum(t, z):
    th1, th2, w1, w2 = z
    delta = th1 - th2

    den1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta)**2
    den2 = l2 / l1 * den1

    dw1 = (-m2 * l1 * w1**2 * np.sin(delta) * np.cos(delta)
            - m2 * l2 * w2**2 * np.sin(delta)
            - (m1 + m2) * g * np.sin(th1)
            + m2 * g * np.sin(th2) * np.cos(delta)) / den1

    dw2 = (m2 * l2 * w2**2 * np.sin(delta) * np.cos(delta)
            + (m1 + m2) * l1 * w1**2 * np.sin(delta)
            + (m1 + m2) * g * np.sin(th1) * np.cos(delta)
            - (m1 + m2) * g * np.sin(th2)) / den2

    return [w1, w2, dw1, dw2]

# Two slightly different initial conditions → sensitivity to chaos
t_span = (0, 20)
t_eval = np.linspace(0, 20, 5000)

ic1 = [np.pi/2, np.pi/2, 0, 0]
ic2 = [np.pi/2 + 0.001, np.pi/2, 0, 0]  # theta1 differs by 0.001 rad

sol1 = solve_ivp(double_pendulum, t_span, ic1, t_eval=t_eval,
                 method='RK45', rtol=1e-12, atol=1e-14)
sol2 = solve_ivp(double_pendulum, t_span, ic2, t_eval=t_eval,
                 method='RK45', rtol=1e-12, atol=1e-14)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# theta1 comparison
axes[0, 0].plot(sol1.t, sol1.y[0], 'b-', linewidth=0.8, label='IC 1')
axes[0, 0].plot(sol2.t, sol2.y[0], 'r--', linewidth=0.8, label='IC 2')
axes[0, 0].set_xlabel('t')
axes[0, 0].set_ylabel('$\\theta_1$')
axes[0, 0].set_title('$\\theta_1(t)$: Sensitivity to initial conditions (chaos)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# theta2 comparison
axes[0, 1].plot(sol1.t, sol1.y[1], 'b-', linewidth=0.8, label='IC 1')
axes[0, 1].plot(sol2.t, sol2.y[1], 'r--', linewidth=0.8, label='IC 2')
axes[0, 1].set_xlabel('t')
axes[0, 1].set_ylabel('$\\theta_2$')
axes[0, 1].set_title('$\\theta_2(t)$: Sensitivity to initial conditions (chaos)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Phase space (theta1 vs omega1)
axes[1, 0].plot(sol1.y[0], sol1.y[2], 'b-', linewidth=0.3, alpha=0.7)
axes[1, 0].set_xlabel('$\\theta_1$')
axes[1, 0].set_ylabel('$\\dot{\\theta}_1$')
axes[1, 0].set_title('Phase space: $(\\theta_1, \\dot{\\theta}_1)$')
axes[1, 0].grid(True, alpha=0.3)

# Pendulum tip trajectory
x1 = l1 * np.sin(sol1.y[0])
y1 = -l1 * np.cos(sol1.y[0])
x2 = x1 + l2 * np.sin(sol1.y[1])
y2 = y1 - l2 * np.cos(sol1.y[1])

axes[1, 1].plot(x2, y2, 'b-', linewidth=0.2, alpha=0.5)
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('y')
axes[1, 1].set_title('Double Pendulum Tip Trajectory')
axes[1, 1].set_aspect('equal')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Double Pendulum - Chaotic Dynamics', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('double_pendulum.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Small-amplitude normal mode analysis ---
M_mat = np.array([[(m1 + m2) * l1, m2 * l2],
                  [l1, l2]])
K_mat = np.array([[(m1 + m2) * g, 0],
                  [0, g]])

# Generalized eigenvalue problem: K u = omega^2 M u
from scipy.linalg import eig
eigenvalues, eigenvectors = eig(K_mat, M_mat)
omega_normal = np.sqrt(eigenvalues.real)
omega_normal = np.sort(omega_normal)

print("\nDouble pendulum (small amplitude) normal frequencies:")
print(f"  omega_1 = {omega_normal[0]:.4f} rad/s (in-phase mode)")
print(f"  omega_2 = {omega_normal[1]:.4f} rad/s (out-of-phase mode)")
print(f"  Ratio omega_2/omega_1 = {omega_normal[1]/omega_normal[0]:.4f}")
```

---

## Exercises

### Basic Problems

**Problem 1**: Find the general solution of the following ODEs.
- (a) $y^{(4)} + 4y'' = 0$
- (b) $y''' - y = 0$
- (c) $y'' + 4y' + 13y = 0$

<details>
<summary>Solution Hint</summary>

(a) Characteristic equation: $r^4 + 4r^2 = r^2(r^2 + 4) = 0$ → $r = 0, 0, \pm 2i$

General solution: $y = c_1 + c_2 x + c_3 \cos 2x + c_4 \sin 2x$

(b) $r^3 - 1 = (r-1)(r^2+r+1) = 0$ → $r = 1, -\frac{1}{2} \pm \frac{\sqrt{3}}{2}i$

(c) $r^2 + 4r + 13 = 0$ → $r = -2 \pm 3i$

```python
# Verification
import sympy as sp
x = sp.Symbol('x')
y = sp.Function('y')

# (a)
sol_a = sp.dsolve(y(x).diff(x, 4) + 4*y(x).diff(x, 2), y(x))
print(f"(a): {sol_a}")

# (b)
sol_b = sp.dsolve(y(x).diff(x, 3) - y(x), y(x))
print(f"(b): {sol_b}")

# (c)
sol_c = sp.dsolve(y(x).diff(x, 2) + 4*y(x).diff(x) + 13*y(x), y(x))
print(f"(c): {sol_c}")
```

</details>

**Problem 2**: Find the general solution of the following system using eigenvalues/eigenvectors.

$$\frac{d}{dt}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} 3 & -2 \\ 2 & -2 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

<details>
<summary>Solution Hint</summary>

Characteristic equation: $\lambda^2 - \lambda - 2 = 0$ → $\lambda_1 = 2$, $\lambda_2 = -1$

Finding eigenvectors for each eigenvalue and constructing the general solution:

$$\mathbf{x}(t) = c_1 \begin{pmatrix} 2 \\ 1 \end{pmatrix} e^{2t} + c_2 \begin{pmatrix} 1 \\ 2 \end{pmatrix} e^{-t}$$

</details>

### Applied Problems

**Problem 3** (Coupled oscillators): Three masses $m$ are connected by identical springs with spring constant $k$ (both ends fixed to walls). Find the normal frequencies and normal modes.

<details>
<summary>Solution Hint</summary>

Stiffness matrix:

$$K = k \begin{pmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{pmatrix}$$

Eigenvalues: $\omega_n^2 = \frac{k}{m}(2 - 2\cos\frac{n\pi}{4})$, $n = 1, 2, 3$

```python
import numpy as np
from scipy.linalg import eigh

k, m_val = 1.0, 1.0
K = k * np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
M = m_val * np.eye(3)

omega_sq, modes = eigh(K, M)
print("Normal frequencies:", np.sqrt(omega_sq))
print("Normal modes:\n", modes)
```

</details>

**Problem 4** (Nonlinear analysis): Find all equilibrium points of the following nonlinear system and classify the stability at each point.

$$\dot{x} = x(3 - x - 2y), \quad \dot{y} = y(2 - x - y)$$

<details>
<summary>Solution Hint</summary>

Equilibrium points: $(0,0)$, $(3,0)$, $(0,2)$, $(1,1)$ — points where $\dot{x}=0$, $\dot{y}=0$ simultaneously.

Jacobian at each equilibrium:

$$J = \begin{pmatrix} 3 - 2x - 2y & -2x \\ -y & 2 - x - 2y \end{pmatrix}$$

```python
import sympy as sp

x, y = sp.symbols('x y')
F1 = x * (3 - x - 2*y)
F2 = y * (2 - x - y)

# Equilibrium points
eq_pts = sp.solve([F1, F2], [x, y])
print("Equilibrium points:", eq_pts)

# Jacobian matrix
J = sp.Matrix([[sp.diff(F1, x), sp.diff(F1, y)],
               [sp.diff(F2, x), sp.diff(F2, y)]])

for pt in eq_pts:
    J_at = J.subs([(x, pt[0]), (y, pt[1])])
    eigenvals = J_at.eigenvals()
    print(f"\nEquilibrium {pt}: eigenvalues = {eigenvals}")
```

</details>

**Problem 5** (Phase portraits): Analyze how the phase portrait of the following system changes with parameter $\mu$ (**Hopf bifurcation**):

$$\dot{x} = \mu x - y - x(x^2 + y^2), \quad \dot{y} = x + \mu y - y(x^2 + y^2)$$

<details>
<summary>Solution Hint</summary>

Eigenvalues of the Jacobian at the origin: $\lambda = \mu \pm i$

- $\mu < 0$: Stable spiral
- $\mu = 0$: Bifurcation point (non-hyperbolic)
- $\mu > 0$: Unstable spiral + stable limit cycle (radius $r = \sqrt{\mu}$)

In polar coordinates $(r, \theta)$: $\dot{r} = r(\mu - r^2)$, $\dot{\theta} = 1$

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def hopf_system(t, z, mu):
    x, y = z
    r_sq = x**2 + y**2
    return [mu*x - y - x*r_sq, x + mu*y - y*r_sq]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
mu_vals = [-0.5, 0.0, 0.5]

for idx, mu in enumerate(mu_vals):
    for r0 in [0.1, 0.3, 0.8, 1.2]:
        for theta0 in [0, np.pi/2, np.pi]:
            ic = [r0*np.cos(theta0), r0*np.sin(theta0)]
            sol = solve_ivp(hopf_system, [0, 30], ic, args=(mu,),
                            t_eval=np.linspace(0, 30, 2000),
                            method='RK45', rtol=1e-10)
            axes[idx].plot(sol.y[0], sol.y[1], linewidth=0.5, alpha=0.6)

    axes[idx].set_title(f'$\\mu = {mu}$')
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('y')
    axes[idx].set_aspect('equal')
    axes[idx].grid(True, alpha=0.3)
    if mu > 0:
        theta = np.linspace(0, 2*np.pi, 100)
        axes[idx].plot(np.sqrt(mu)*np.cos(theta), np.sqrt(mu)*np.sin(theta),
                       'r-', linewidth=2, label=f'Limit cycle $r=\\sqrt{{{mu}}}$')
        axes[idx].legend()

plt.suptitle('Hopf Bifurcation', fontsize=14)
plt.tight_layout()
plt.savefig('hopf_bifurcation.png', dpi=150, bbox_inches='tight')
plt.show()
```

</details>

---

## References

### Textbooks
1. **Boas, M. L.** (2005). *Mathematical Methods in the Physical Sciences*, 3rd ed., Chapter 8. Wiley.
2. **Strogatz, S. H.** (2018). *Nonlinear Dynamics and Chaos*, 2nd ed. CRC Press.
   - Standard textbook for nonlinear dynamics and phase plane analysis
3. **Hirsch, M. W., Smale, S., & Devaney, R. L.** (2013). *Differential Equations, Dynamical Systems, and an Introduction to Chaos*, 3rd ed. Academic Press.
4. **Arfken, G. B. et al.** (2012). *Mathematical Methods for Physicists*, 7th ed., Chapter 10. Academic Press.

### Online Resources
1. **MIT OCW 18.03**: Differential Equations (Arthur Mattuck)
2. **3Blue1Brown**: Differential equations, studying the unsolvable
3. **Steve Brunton (YouTube)**: Data-Driven Dynamical Systems series

### Core Library Documentation
1. **SciPy `solve_ivp`**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
2. **SymPy `dsolve`**: https://docs.sympy.org/latest/modules/solvers/ode.html
3. **Matplotlib `streamplot`**: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.streamplot.html

---

## Next Lesson

In [09. Series Solutions and Special Functions](09_Series_Solutions_Special_Functions.md), we'll cover **series solutions** to ODEs using the **Frobenius method**, and study the properties and applications of the most important **special functions** in physics (Bessel functions, Legendre polynomials, Hermite functions, Laguerre functions).
