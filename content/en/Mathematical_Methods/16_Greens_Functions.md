# 16. Green's Functions

## Learning Objectives

- Understand the definition, properties, and various representations of the **Dirac delta function** $\delta(x)$ and apply them
- Express solutions to inhomogeneous differential equations $L[u] = f(x)$ in integral form using **Green's functions**
- Directly construct Green's functions for **boundary value problems** and apply matching conditions
- Represent Green's functions in series form using **eigenfunction expansion** and understand connections to Sturm-Liouville theory
- Obtain and physically interpret Green's functions for **partial differential equations** (Poisson, heat, wave equations)
- Apply Green's functions to solve real problems in **physical applications** such as electrostatics, quantum mechanics, and acoustics

> **Importance in Physics**: Green's functions are universal tools describing "response to a point source." Once the response to a point source (Green's function) is known, the solution for **arbitrary source distributions** can be obtained by a single integral via the superposition principle. Key problems in modern physics—electrostatic potential of point charges, propagators in quantum mechanics, radiation from point sources in acoustics—all reduce to Green's functions.

---

## 1. Dirac Delta Function

### 1.1 Definition and Basic Properties

The **Dirac delta function** $\delta(x)$ is not a function in the strict sense but a **generalized function** or **distribution**. It is defined by two properties:

$$\delta(x) = 0 \quad (x \neq 0), \qquad \int_{-\infty}^{\infty} \delta(x) \, dx = 1$$

**Sifting property**: The most important property of the delta function.

$$\int_{-\infty}^{\infty} f(x) \delta(x - a) \, dx = f(a)$$

This means $\delta(x-a)$ "extracts" the function value at $x = a$.

**Additional properties**:

$$\delta(-x) = \delta(x) \quad \text{(even function)}$$

$$\delta(ax) = \frac{1}{|a|}\delta(x) \quad (a \neq 0)$$

$$x\delta(x) = 0$$

$$\delta(g(x)) = \sum_i \frac{\delta(x - x_i)}{|g'(x_i)|} \quad (g(x_i) = 0, \; g'(x_i) \neq 0)$$

### 1.2 Representations of Delta Function

$\delta(x)$ can be represented as limits of regular functions:

**Gaussian representation**:
$$\delta(x) = \lim_{\epsilon \to 0} \frac{1}{\epsilon\sqrt{\pi}} e^{-x^2/\epsilon^2}$$

**Lorentzian representation**:
$$\delta(x) = \lim_{\epsilon \to 0} \frac{1}{\pi} \frac{\epsilon}{x^2 + \epsilon^2}$$

**sinc representation (Fourier representation)**:
$$\delta(x) = \frac{1}{2\pi} \int_{-\infty}^{\infty} e^{ikx} dk = \lim_{N \to \infty} \frac{\sin(Nx)}{\pi x}$$

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 1000)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Gaussian representation
for eps in [1.0, 0.5, 0.2, 0.05]:
    delta_gauss = np.exp(-x**2 / eps**2) / (eps * np.sqrt(np.pi))
    axes[0].plot(x, delta_gauss, label=f'$\\epsilon={eps}$')
axes[0].set_title('Gaussian representation')
axes[0].legend(); axes[0].set_ylim(0, 10); axes[0].grid(True, alpha=0.3)

# Lorentzian representation
for eps in [1.0, 0.5, 0.2, 0.05]:
    delta_lorentz = (1/np.pi) * eps / (x**2 + eps**2)
    axes[1].plot(x, delta_lorentz, label=f'$\\epsilon={eps}$')
axes[1].set_title('Lorentzian representation')
axes[1].legend(); axes[1].set_ylim(0, 10); axes[1].grid(True, alpha=0.3)

# sinc representation
for N in [5, 20, 50, 200]:
    delta_sinc = np.sin(N * x) / (np.pi * x + 1e-30)
    axes[2].plot(x, delta_sinc, label=f'$N={N}$', alpha=0.8)
axes[2].set_title('sinc representation (Fourier)')
axes[2].legend(); axes[2].set_ylim(-5, 70); axes[2].grid(True, alpha=0.3)

plt.suptitle('Various representations of the Dirac delta function', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
```

### 1.3 Derivatives of Delta Function

$\delta'(x)$ is defined by the following integration property:

$$\int_{-\infty}^{\infty} f(x) \delta'(x-a) \, dx = -f'(a)$$

Generally for $n$-th derivative:

$$\int_{-\infty}^{\infty} f(x) \delta^{(n)}(x-a) \, dx = (-1)^n f^{(n)}(a)$$

### 1.4 Multidimensional Delta Functions

3D Dirac delta:

$$\delta^3(\mathbf{r} - \mathbf{r}') = \delta(x - x')\delta(y - y')\delta(z - z')$$

**Sifting property**: $\int f(\mathbf{r}) \delta^3(\mathbf{r} - \mathbf{r}') \, d^3r = f(\mathbf{r}')$

**In spherical coordinates**: $\delta^3(\mathbf{r} - \mathbf{r}') = \frac{\delta(r-r')}{r^2} \frac{\delta(\theta-\theta')}{\sin\theta} \delta(\phi-\phi')$

Important relation (core of electrostatics):

$$\nabla^2 \left(\frac{1}{|\mathbf{r} - \mathbf{r}'|}\right) = -4\pi \delta^3(\mathbf{r} - \mathbf{r}')$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Numerical verification of the sifting property
# For f(x) = cos(x), a = 1.0: integral f(x) delta_eps(x-a) dx ~ f(a)
a = 1.0
f = lambda t: np.cos(t)
x = np.linspace(-10, 10, 100000)

print("=== Numerical verification of the sifting property ===")
print(f"f(a) = cos({a}) = {f(a):.8f}")
print()

for eps in [1.0, 0.1, 0.01, 0.001]:
    # Use Gaussian approximation of delta
    delta_approx = np.exp(-(x - a)**2 / eps**2) / (eps * np.sqrt(np.pi))
    integral = np.trapz(f(x) * delta_approx, x)
    error = abs(integral - f(a))
    print(f"  eps = {eps:.3f}: integral = {integral:.8f}, error = {error:.2e}")
```

---

## 2. Concept of Green's Functions

### 2.1 Inhomogeneous Differential Equations and Superposition Principle

For a **linear differential operator** $L$, we wish to solve the inhomogeneous equation:

$$L[u(x)] = f(x)$$

If $L$ is linear, the **superposition principle** holds: if $L[u_1] = f_1$ and $L[u_2] = f_2$, then $L[\alpha u_1 + \beta u_2] = \alpha f_1 + \beta f_2$.

### 2.2 Green's Function as Point Source Response

Expressing source $f(x)$ as a superposition of delta functions:

$$f(x) = \int f(x') \delta(x - x') \, dx'$$

Define the **Green's function** $G(x, x')$ as "response to point source $\delta(x - x')$":

$$L[G(x, x')] = \delta(x - x')$$

Then by the superposition principle, the solution to the original equation is:

$$\boxed{u(x) = \int G(x, x') f(x') \, dx'}$$

This is the essence of the Green's function method. Once the point source response $G$ is found, the solution for arbitrary source $f$ is immediately obtained by integration.

### 2.3 Physical Intuition

| Physical System | Operator $L$ | Source $f$ | Green's Function $G$ |
|---|---|---|---|
| Electrostatics | $\nabla^2$ | $-\rho/\epsilon_0$ | Potential of point charge |
| Heat conduction | $\partial_t - \alpha^2\nabla^2$ | Heat source | Temperature response to point heat source |
| Wave | $\partial_t^2 - c^2\nabla^2$ | External force | Wave from point impulse |
| Quantum mechanics | $i\hbar\partial_t - H$ | — | Propagator |

```python
import numpy as np
import matplotlib.pyplot as plt

# Conceptual demo: response of a 1D string to a point load
# L[u] = u'' = f(x), u(0) = u(1) = 0
# Green's function: G(x, x') = x'(1-x) for x > x', x(1-x') for x < x'

def greens_function_string(x, xp):
    """Green's function for a string fixed at both ends"""
    return np.where(x < xp, x * (1 - xp), xp * (1 - x))

x = np.linspace(0, 1, 500)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Green's function for various point source positions
for xp in [0.2, 0.4, 0.5, 0.6, 0.8]:
    G = greens_function_string(x, xp)
    ax1.plot(x, G, linewidth=2, label=f"$x' = {xp}$")
    ax1.plot(xp, greens_function_string(xp, xp), 'ko', markersize=5)

ax1.set_xlabel('$x$'); ax1.set_ylabel("$G(x, x')$")
ax1.set_title("Green's function $G(x, x')$ by point source position")
ax1.legend(); ax1.grid(True, alpha=0.3)

# Right: superposition principle - solution for arbitrary source
f_source = lambda t: np.sin(2 * np.pi * t)  # arbitrary source f(x)
u_solution = np.array([np.trapz(greens_function_string(xi, x) * f_source(x), x)
                        for xi in x])

ax2.plot(x, f_source(x), 'r--', linewidth=1.5, label='source $f(x) = \\sin(2\\pi x)$')
ax2.plot(x, u_solution, 'b-', linewidth=2.5, label='solution $u(x) = \\int G f \\, dx\'$')
ax2.set_xlabel('$x$'); ax2.set_title('Solution construction by superposition principle')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.suptitle("Green's function concept: superposition of point source responses", fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
```

---

## 3. Green's Functions for Boundary Value Problems

### 3.1 Construction Method

For second-order ODE $L[y] = y'' + p(x)y' + q(x)y = f(x)$ with homogeneous boundary conditions $y(a) = 0$, $y(b) = 0$, construct the Green's function:

**Step 1**: Find two independent solutions of the homogeneous equation $L[y] = 0$:
- $y_1(x)$: satisfies $y_1(a) = 0$
- $y_2(x)$: satisfies $y_2(b) = 0$

**Step 2**: The Green's function is defined piecewise:

$$G(x, x') = \begin{cases} A \, y_1(x) y_2(x') & x < x' \\ A \, y_1(x') y_2(x) & x > x' \end{cases}$$

**Step 3**: Apply matching conditions:

(1) **Continuity**: $G$ is continuous at $x = x'$

$$G(x'^-, x') = G(x'^+, x')$$

(2) **Jump discontinuity in derivative**: $G'$ has a jump discontinuity at $x = x'$

$$\left.\frac{\partial G}{\partial x}\right|_{x'^+} - \left.\frac{\partial G}{\partial x}\right|_{x'^-} = \frac{1}{p(x')}$$

where $p(x')$ is the coefficient of the highest-order term of $L$ (for standard form $y''$, $p=1$).

Constant $A$ is determined from these conditions:

$$A = \frac{1}{p(x') W(y_1, y_2)(x')}$$

where $W$ is the **Wronskian**: $W = y_1 y_2' - y_1' y_2$.

### 3.2 Symmetry

**Theorem**: Green's functions for self-adjoint operators are symmetric:

$$G(x, x') = G(x', x)$$

This mathematically expresses the **reciprocity theorem**: the response at point $x$ due to a source at $x'$ equals the response at $x'$ due to a source at $x$.

### 3.3 Example: $y'' = f(x)$, $y(0) = y(1) = 0$

Homogeneous solutions: $y_1 = x$, $y_2 = 1 - x$. $W = y_1 y_2' - y_1'y_2 = -x - (1-x) = -1$.

$$G(x, x') = \begin{cases} x(1-x') & x < x' \\ x'(1-x) & x > x' \end{cases}$$

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Construct and verify Green's function using SymPy
x_sym, xp_sym = sp.symbols('x xp')

# y'' = f(x), y(0) = y(1) = 0
# Homogeneous solutions: y1 = x (y1(0)=0), y2 = 1-x (y2(1)=0)
y1 = x_sym
y2 = 1 - x_sym
W = y1 * sp.diff(y2, x_sym) - sp.diff(y1, x_sym) * y2
print(f"Wronskian W = {W}")  # -1

# Green's function (x < x' region)
G_left = -y1 * y2.subs(x_sym, xp_sym) / W  # x < x'
G_right = -y1.subs(x_sym, xp_sym) * y2 / W  # x > x'
print(f"G(x, x') = {G_left} (x < x')")
print(f"G(x, x') = {G_right} (x > x')")

# Verify matching conditions
print("\n=== Matching condition verification (x = x') ===")
# Continuity
G_left_at_xp = G_left.subs(x_sym, xp_sym)
G_right_at_xp = G_right.subs(x_sym, xp_sym)
print(f"Continuity: G(x'^-, x') = {G_left_at_xp}, G(x'^+, x') = {G_right_at_xp}")
print(f"  difference = {sp.simplify(G_left_at_xp - G_right_at_xp)}")

# Derivative jump
dG_left = sp.diff(G_left, x_sym).subs(x_sym, xp_sym)
dG_right = sp.diff(G_right, x_sym).subs(x_sym, xp_sym)
jump = sp.simplify(dG_right - dG_left)
print(f"Derivative jump: G'(x'^+) - G'(x'^-) = {jump}")  # 1 (= 1/p(x'))

# Numerical visualization
x_num = np.linspace(0, 1, 500)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Contour plot of G(x, x') (checking symmetry)
xp_vals = np.linspace(0, 1, 500)
X, XP = np.meshgrid(x_num, xp_vals)
G_vals = np.where(X < XP, X * (1 - XP), XP * (1 - X))

c = ax1.contourf(X, XP, G_vals, levels=30, cmap='viridis')
plt.colorbar(c, ax=ax1)
ax1.set_xlabel("$x$"); ax1.set_ylabel("$x'$")
ax1.set_title("$G(x, x')$ contour — symmetry $G(x,x') = G(x',x)$")

# Cross-sections of G(x, x') at specific x'
for xp in [0.25, 0.5, 0.75]:
    G_section = np.where(x_num < xp, x_num * (1 - xp), xp * (1 - x_num))
    ax2.plot(x_num, G_section, linewidth=2, label=f"$x' = {xp}$")
    # Mark kink point (derivative discontinuity)
    ax2.plot(xp, xp * (1 - xp), 'ko', markersize=6)

ax2.set_xlabel('$x$'); ax2.set_ylabel("$G(x, x')$")
ax2.set_title("Green's function cross-section: derivative discontinuity")
ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

---

## 4. Sturm-Liouville Problems and Eigenfunction Expansion

### 4.1 Eigenfunction Expansion Method

Using the completeness of eigenfunctions from [Sturm-Liouville theory (Lesson 10)](10_Sturm_Liouville_Theory.md), the Green's function can be expanded in a series of eigenfunctions.

Eigenvalue problem for self-adjoint operator $L$:

$$L[\phi_n] = \lambda_n w(x) \phi_n, \quad n = 1, 2, 3, \ldots$$

If eigenfunctions $\{\phi_n\}$ form a complete orthogonal set in $L^2_w[a,b]$, the Green's function is:

$$\boxed{G(x, x') = \sum_{n=1}^{\infty} \frac{\phi_n(x) \phi_n(x')}{\lambda_n \|\phi_n\|_w^2}}$$

where $\|\phi_n\|_w^2 = \int_a^b w(x) |\phi_n(x)|^2 dx$.

### 4.2 Derivation

Expand $G(x, x')$ in eigenfunctions:

$$G(x, x') = \sum_n c_n(x') \phi_n(x)$$

Substitute into $L[G] = \delta(x - x')$, multiply both sides by $\phi_m(x)$ and integrate:

$$\lambda_m c_m(x') \|\phi_m\|_w^2 = \phi_m(x')$$

Therefore $c_m(x') = \phi_m(x') / (\lambda_m \|\phi_m\|_w^2)$.

### 4.3 Example: $y'' = f(x)$, $y(0) = y(\pi) = 0$

Eigenvalue problem: $\phi_n'' = -\lambda_n \phi_n$, $\phi_n(0) = \phi_n(\pi) = 0$

$\phi_n = \sin(nx)$, $\lambda_n = n^2$, $\|\phi_n\|^2 = \pi/2$

$$G(x, x') = \frac{2}{\pi} \sum_{n=1}^{\infty} \frac{\sin(nx)\sin(nx')}{n^2}$$

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, np.pi, 500)

def G_eigenfunction(x_val, xp, N_terms):
    """Green's function from eigenfunction expansion"""
    G = np.zeros_like(x_val)
    for n in range(1, N_terms + 1):
        G += (2 / np.pi) * np.sin(n * x_val) * np.sin(n * xp) / n**2
    return G

def G_exact(x_val, xp):
    """Exact Green's function (closed form)"""
    return np.where(x_val < xp,
                    x_val * (np.pi - xp) / np.pi,
                    xp * (np.pi - x_val) / np.pi)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
xp = np.pi / 2  # x' = pi/2

# Left: series convergence visualization
ax1.plot(x, G_exact(x, xp), 'k-', linewidth=3, label='Exact solution')
for N in [1, 3, 10, 50]:
    ax1.plot(x, G_eigenfunction(x, xp, N), '--', linewidth=1.5, label=f'$N = {N}$')
ax1.set_title(f"Convergence of eigenfunction expansion ($x' = \\pi/2$)")
ax1.set_xlabel('$x$'); ax1.legend(); ax1.grid(True, alpha=0.3)

# Right: error vs number of terms
N_values = np.arange(1, 101)
errors = []
for N in N_values:
    G_approx = G_eigenfunction(x, xp, N)
    G_true = G_exact(x, xp)
    errors.append(np.max(np.abs(G_approx - G_true)))

ax2.semilogy(N_values, errors, 'b-', linewidth=2)
ax2.set_xlabel('Number of terms $N$'); ax2.set_ylabel('Maximum error')
ax2.set_title('Convergence rate of eigenfunction expansion')
ax2.grid(True, alpha=0.3)

plt.suptitle("Green's function by eigenfunction expansion", fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
```

---

## 5. Green's Functions for Ordinary Differential Equations

### 5.1 General Second-Order ODE

$$y'' + p(x)y' + q(x)y = f(x), \quad y(a) = y(b) = 0$$

The solution to this problem is closely connected to **variation of parameters**. In fact, the particular solution obtained by variation of parameters is equivalent to the integral representation using Green's functions.

### 5.2 Constant Coefficient Equations

For $y'' + k^2 y = f(x)$, $y(0) = y(L) = 0$:

Homogeneous solutions: $y_1 = \sin(kx)$, $y_2 = \sin(k(L-x))$

$$G(x, x') = \frac{1}{k\sin(kL)} \begin{cases} \sin(kx)\sin(k(L-x')) & x < x' \\ \sin(kx')\sin(k(L-x)) & x > x' \end{cases}$$

### 5.3 Example: Green's Function for Harmonic Oscillator

**Damped harmonic oscillator**: $\ddot{x} + 2\gamma\dot{x} + \omega_0^2 x = f(t)$

**Causal Green's function** or **retarded Green's function** with initial conditions $x(0) = \dot{x}(0) = 0$:

$$G_R(t, t') = \begin{cases} \frac{1}{\omega_d} e^{-\gamma(t-t')} \sin(\omega_d(t-t')) & t > t' \\ 0 & t < t' \end{cases}$$

where $\omega_d = \sqrt{\omega_0^2 - \gamma^2}$ (underdamped, $\gamma < \omega_0$).

Solution: $x(t) = \int_0^t G_R(t, t') f(t') \, dt'$

```python
import numpy as np
import matplotlib.pyplot as plt

# Green's function of the damped harmonic oscillator
omega0, gamma = 5.0, 0.5
omega_d = np.sqrt(omega0**2 - gamma**2)

def G_retarded(t, tp):
    """Retarded Green's function"""
    dt = t - tp
    return np.where(dt > 0,
                    np.exp(-gamma * dt) * np.sin(omega_d * dt) / omega_d,
                    0.0)

t = np.linspace(0, 10, 2000)

# Compute responses to various source functions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sources = {
    'Impulse': lambda tp: np.where(np.abs(tp - 1.0) < 0.05, 1.0/0.1, 0.0),
    'Step function': lambda tp: np.where(tp > 1.0, 1.0, 0.0),
    'Sinusoid': lambda tp: np.sin(3.0 * tp),
    'Double pulse': lambda tp: (np.where(np.abs(tp-1)<0.05, 1/0.1, 0.0) +
                             np.where(np.abs(tp-3)<0.05, -1/0.1, 0.0))
}

for ax, (name, f_source) in zip(axes.flat, sources.items()):
    # Convolution integral using Green's function
    f_vals = f_source(t)
    x_response = np.array([np.trapz(G_retarded(ti, t[:i+1]) * f_vals[:i+1], t[:i+1])
                           if i > 0 else 0.0 for i, ti in enumerate(t)])

    ax.plot(t, f_vals * 0.1, 'r--', alpha=0.5, label='source $f(t)$ (scaled)')
    ax.plot(t, x_response, 'b-', linewidth=2, label='response $x(t)$')
    ax.set_title(f'Source: {name}'); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlabel('$t$')

plt.suptitle(f'Damped harmonic oscillator ($\\omega_0={omega0}, \\gamma={gamma}$)',
             fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
```

---

## 6. Green's Functions for Partial Differential Equations

### 6.1 Free Space Green's Function for Poisson's Equation

**Poisson's equation**: $\nabla^2 \phi = -\rho/\epsilon_0$

Green's function definition: $\nabla^2 G(\mathbf{r}, \mathbf{r}') = \delta^3(\mathbf{r} - \mathbf{r}')$

**3D free space**:

$$G(\mathbf{r}, \mathbf{r}') = -\frac{1}{4\pi|\mathbf{r} - \mathbf{r}'|}$$

**2D free space**:

$$G(\mathbf{r}, \mathbf{r}') = \frac{1}{2\pi}\ln|\mathbf{r} - \mathbf{r}'|$$

Solution: $\phi(\mathbf{r}) = \frac{1}{4\pi\epsilon_0}\int \frac{\rho(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} d^3r'$ — exactly **Coulomb potential**!

### 6.2 Method of Images

With finite boundaries, the free space Green's function cannot be used directly. The **method of images** constructs a Green's function satisfying boundary conditions by adding "fictitious sources."

**Example**: Point charge above grounded infinite plane ($z = 0$)

For real charge $q$ at $(0, 0, d)$, place image charge $-q$ at $(0, 0, -d)$:

$$G(\mathbf{r}, \mathbf{r}') = -\frac{1}{4\pi}\left(\frac{1}{|\mathbf{r} - \mathbf{r}'|} - \frac{1}{|\mathbf{r} - \mathbf{r}''|}\right)$$

where $\mathbf{r}'' = (x', y', -z')$ is the image point.

### 6.3 Green's Function for Heat Equation

**Heat equation**: $\frac{\partial u}{\partial t} = \alpha^2 \nabla^2 u$

Free space Green's function (1D):

$$G(x, t; x', t') = \frac{1}{\sqrt{4\pi\alpha^2(t-t')}} \exp\left(-\frac{(x-x')^2}{4\alpha^2(t-t')}\right), \quad t > t'$$

This starts as $\delta(x - x')$ at $t = t'$ and spreads as a Gaussian over time.

### 6.4 Retarded Green's Function for Wave Equation

**Wave equation**: $\nabla^2 G - \frac{1}{c^2}\frac{\partial^2 G}{\partial t^2} = \delta^3(\mathbf{r} - \mathbf{r}')\delta(t - t')$

3D **retarded Green's function**:

$$G_R(\mathbf{r}, t; \mathbf{r}', t') = -\frac{\delta(t - t' - |\mathbf{r}-\mathbf{r}'|/c)}{4\pi|\mathbf{r}-\mathbf{r}'|}$$

This reflects causality: signals propagate at speed $c$, arriving after time $|\mathbf{r}-\mathbf{r}'|/c$.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Heat equation Green's function visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

alpha2 = 0.01
x = np.linspace(-2, 2, 1000)
xp = 0.0  # source position

for t_val in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
    G_heat = np.exp(-(x - xp)**2 / (4 * alpha2 * t_val)) / np.sqrt(4 * np.pi * alpha2 * t_val)
    ax1.plot(x, G_heat, linewidth=2, label=f'$t = {t_val}$')

ax1.set_xlabel('$x$'); ax1.set_ylabel("$G(x, t; 0, 0)$")
ax1.set_title('Heat equation Green\'s function (1D)')
ax1.legend(); ax1.grid(True, alpha=0.3)

# --- 2D Poisson equation Green's function visualization ---
x2d = np.linspace(-2, 2, 300)
y2d = np.linspace(-2, 2, 300)
X, Y = np.meshgrid(x2d, y2d)

# Point source position
xp2, yp2 = 0.5, 0.3
R = np.sqrt((X - xp2)**2 + (Y - yp2)**2)
R = np.maximum(R, 0.01)  # avoid singularity
G_2d = np.log(R) / (2 * np.pi)

c = ax2.contourf(X, Y, G_2d, levels=30, cmap='RdBu_r')
ax2.plot(xp2, yp2, 'k*', markersize=15, label="source position $(x', y')$")
plt.colorbar(c, ax=ax2)
ax2.set_xlabel('$x$'); ax2.set_ylabel('$y$')
ax2.set_title("2D Poisson Green's function $G = \\frac{1}{2\\pi}\\ln r$")
ax2.legend(); ax2.set_aspect('equal')

plt.suptitle("Green's functions for partial differential equations", fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
```

---

## 7. Physical Applications

### 7.1 Electrostatics: Potential of Charge Distribution

Given charge density $\rho(\mathbf{r})$, the potential is:

$$\phi(\mathbf{r}) = \frac{1}{4\pi\epsilon_0} \int \frac{\rho(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} d^3r'$$

This is exactly the Green's function solution to $\nabla^2\phi = -\rho/\epsilon_0$.

### 7.2 Quantum Mechanics: Propagator

The Green's function of the time-dependent Schrödinger equation is the **propagator** $K(\mathbf{r}, t; \mathbf{r}', t')$:

$$\psi(\mathbf{r}, t) = \int K(\mathbf{r}, t; \mathbf{r}', t_0) \psi(\mathbf{r}', t_0) \, d^3r'$$

Free particle propagator: $K = \left(\frac{m}{2\pi i\hbar(t-t')}\right)^{3/2} \exp\left(\frac{im|\mathbf{r}-\mathbf{r}'|^2}{2\hbar(t-t')}\right)$

### 7.3 Acoustics: Radiation from Point Source

For wave equation $\nabla^2 p - \frac{1}{c^2}\ddot{p} = -S(\mathbf{r}, t)$ with monochromatic point source $S = \delta^3(\mathbf{r})e^{-i\omega t}$, the pressure is:

$$p(\mathbf{r}) = -\frac{e^{ikr}}{4\pi r} \quad (k = \omega/c)$$

This is a **spherical wave**.

```python
import numpy as np
import matplotlib.pyplot as plt

# === Electrostatics application: potential of 2D charge distribution ===
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

x = np.linspace(-3, 3, 300)
y = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, y)

# (a) Point charge
q1, x1, y1 = 1.0, 0.0, 0.0
R1 = np.sqrt((X - x1)**2 + (Y - y1)**2)
phi_point = q1 / (2 * np.pi * np.maximum(R1, 0.05))

axes[0].contourf(X, Y, phi_point, levels=30, cmap='hot_r')
axes[0].plot(x1, y1, 'k+', markersize=15, markeredgewidth=3)
axes[0].set_title('(a) Point charge $+q$'); axes[0].set_aspect('equal')

# (b) Dipole
q2, d = 1.0, 0.5
R_plus = np.sqrt((X - d)**2 + Y**2)
R_minus = np.sqrt((X + d)**2 + Y**2)
phi_dipole = q2 / (2*np.pi*np.maximum(R_plus, 0.05)) - q2 / (2*np.pi*np.maximum(R_minus, 0.05))

axes[1].contourf(X, Y, phi_dipole, levels=np.linspace(-3, 3, 31), cmap='RdBu_r')
axes[1].plot(d, 0, 'r+', markersize=12, markeredgewidth=3)
axes[1].plot(-d, 0, 'b_', markersize=12, markeredgewidth=3)
axes[1].set_title('(b) Electric dipole $+q, -q$'); axes[1].set_aspect('equal')

# (c) Method of images: point charge near grounded plane
q3, d3 = 1.0, 1.0
R_real = np.sqrt(X**2 + (Y - d3)**2)
R_image = np.sqrt(X**2 + (Y + d3)**2)  # image charge
phi_image = q3/(2*np.pi*np.maximum(R_real, 0.05)) - q3/(2*np.pi*np.maximum(R_image, 0.05))
phi_image[Y < 0] = 0  # zero below the grounded plane

axes[2].contourf(X, Y, phi_image, levels=30, cmap='hot_r')
axes[2].axhline(0, color='green', linewidth=3, label='ground plane')
axes[2].plot(0, d3, 'k+', markersize=12, markeredgewidth=3)
axes[2].plot(0, -d3, 'kx', markersize=12, markeredgewidth=3, alpha=0.4)
axes[2].set_title('(c) Method of images: point charge above ground plane')
axes[2].legend(); axes[2].set_aspect('equal')

plt.suptitle("Green's function applications in electrostatics (2D)", fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# Electric field vectors (dipole)
fig, ax = plt.subplots(figsize=(8, 6))
Ex = q2*(X-d)/(2*np.pi*np.maximum(R_plus,0.05)**2) - q2*(X+d)/(2*np.pi*np.maximum(R_minus,0.05)**2)
Ey = q2*Y/(2*np.pi*np.maximum(R_plus,0.05)**2) - q2*Y/(2*np.pi*np.maximum(R_minus,0.05)**2)
E_mag = np.sqrt(Ex**2 + Ey**2)

ax.streamplot(X, Y, Ex, Ey, color=np.log10(E_mag+1e-3), cmap='inferno',
              density=2, linewidth=1)
ax.plot(d, 0, 'ro', markersize=10, label='$+q$')
ax.plot(-d, 0, 'bo', markersize=10, label='$-q$')
ax.set_title("Electric dipole field lines (Green's function superposition)")
ax.legend(); ax.set_aspect('equal'); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
plt.tight_layout(); plt.show()
```

---

## 8. Green's Identities and Integral Representations

### 8.1 Green's Identities

**Green's first identity**: If $u$, $v$ are sufficiently smooth in domain $\Omega$:

$$\int_\Omega (u \nabla^2 v + \nabla u \cdot \nabla v) \, dV = \oint_{\partial\Omega} u \frac{\partial v}{\partial n} \, dS$$

**Green's second identity** (symmetric form):

$$\int_\Omega (u \nabla^2 v - v \nabla^2 u) \, dV = \oint_{\partial\Omega} \left(u \frac{\partial v}{\partial n} - v \frac{\partial u}{\partial n}\right) dS$$

### 8.2 Integral Representation

In Green's second identity, let $v = G$ (where $\nabla^2 G = \delta^3(\mathbf{r} - \mathbf{r}')$):

$$u(\mathbf{r}) = \int_\Omega G(\mathbf{r}, \mathbf{r}') f(\mathbf{r}') \, dV' + \oint_{\partial\Omega} \left(G \frac{\partial u}{\partial n'} - u \frac{\partial G}{\partial n'}\right) dS'$$

**Dirichlet boundary condition** ($u = h$ on $\partial\Omega$): Choose $G = 0$ on $\partial\Omega$:

$$u(\mathbf{r}) = \int_\Omega G f \, dV' - \oint_{\partial\Omega} h \frac{\partial G}{\partial n'} dS'$$

**Neumann boundary condition** ($\partial u/\partial n = g$ on $\partial\Omega$): Choose $\partial G/\partial n' = -1/|\partial\Omega|$ (constant).

```python
import numpy as np
import matplotlib.pyplot as plt

# Numerical verification of Green's identities (1D version)
# integral_0^1 (u v'' + u'v') dx = [u v']_0^1
# u(x) = sin(pi*x), v(x) = x^2

x = np.linspace(0, 1, 10000)

u = np.sin(np.pi * x)
u_prime = np.pi * np.cos(np.pi * x)
v = x**2
v_prime = 2 * x
v_double_prime = 2 * np.ones_like(x)

# Left-hand side
lhs = np.trapz(u * v_double_prime + u_prime * v_prime, x)

# Right-hand side: [u v']_0^1 = u(1)v'(1) - u(0)v'(0)
rhs = u[-1] * v_prime[-1] - u[0] * v_prime[0]

print("=== Numerical verification of Green's first identity (1D) ===")
print(f"u(x) = sin(pi*x), v(x) = x^2")
print(f"LHS: integral(u v'' + u'v') dx = {lhs:.8f}")
print(f"RHS: [u v']_0^1               = {rhs:.8f}")
print(f"Difference: {abs(lhs - rhs):.2e}")

# Green's second identity
u_dbl_prime = -np.pi**2 * np.sin(np.pi * x)

lhs2 = np.trapz(u * v_double_prime - v * u_dbl_prime, x)
rhs2 = (u[-1]*v_prime[-1] - v[-1]*u_prime[-1]) - (u[0]*v_prime[0] - v[0]*u_prime[0])

print("\n=== Numerical verification of Green's second identity (1D) ===")
print(f"LHS: integral(u v'' - v u'') dx = {lhs2:.8f}")
print(f"RHS: [uv' - vu']_0^1           = {rhs2:.8f}")
print(f"Difference: {abs(lhs2 - rhs2):.2e}")

# Verification of integral representation using Green's function
# Problem: u'' = f(x), u(0) = u(1) = 0, f(x) = -pi^2 sin(pi*x)
# Exact solution: u(x) = sin(pi*x)
print("\n=== Integral representation verification ===")
f_rhs = -np.pi**2 * np.sin(np.pi * x)
u_green = np.array([np.trapz(np.where(x < xi, x*(1-xi), xi*(1-x)) * f_rhs, x)
                     for xi in x])
u_exact = np.sin(np.pi * x)

print(f"Maximum error |u_Green - u_exact| = {np.max(np.abs(u_green - u_exact)):.6e}")
```

---

## Practice Problems

### Basic Problems

**Problem 1.** Calculate the following Dirac delta function integrals.

(a) $\int_{-\infty}^{\infty} (x^3 + 2x + 1)\delta(x - 2) \, dx$

(b) $\int_0^5 e^{-x}\delta(x - 3) \, dx$

(c) $\int_{-\infty}^{\infty} \cos(x)\delta'(x) \, dx$

**Problem 2.** Show that $\delta(x^2 - a^2) = \frac{1}{2|a|}[\delta(x-a) + \delta(x+a)]$ ($a > 0$), and calculate $\int_{-\infty}^{\infty} e^{x}\delta(x^2 - 4)\,dx$.

**Problem 3.** Directly construct the Green's function for $y'' = f(x)$, $y(0) = y(L) = 0$, and verify $G(x, x') = G(x', x)$.

**Problem 4.** Find the Green's function for $y'' + y = f(x)$, $y(0) = y(\pi/2) = 0$. (Hint: Use homogeneous solutions $\sin x$, $\cos x$)

### Advanced Problems

**Problem 5.** Find the Green's function for $y'' = f(x)$, $y(0) = y(\pi) = 0$ using eigenfunction expansion, and compare with closed form $G(x,x') = \frac{1}{\pi}[x(\pi-x') \text{ or } x'(\pi-x)]$ to derive the series:

$$\sum_{n=1}^{\infty} \frac{\sin(nx)\sin(nx')}{n^2} = \frac{\pi}{2} \begin{cases} x(1-x'/\pi) & x < x' \\ x'(1-x/\pi) & x > x' \end{cases}$$

**Problem 6.** Use method of images to find the Green's function for $y > 0$ half-plane with $\nabla^2 G = \delta^2(\mathbf{r} - \mathbf{r}')$, $G(x, 0) = 0$ (Dirichlet).

**Problem 7.** Show that the 1D heat equation Green's function $G(x, t; 0, 0) = \frac{1}{\sqrt{4\pi\alpha^2 t}}e^{-x^2/(4\alpha^2 t)}$ satisfies:

(a) $\partial_t G = \alpha^2 \partial_{xx} G$ ($t > 0$)

(b) $\int_{-\infty}^{\infty} G \, dx = 1$ (all $t > 0$)

(c) $\lim_{t \to 0^+} G(x, t; 0, 0) = \delta(x)$

**Problem 8.** For damped harmonic oscillator $\ddot{x} + 2\gamma\dot{x} + \omega_0^2 x = \delta(t)$, find Green's functions for three cases: $\gamma > \omega_0$ (overdamped), $\gamma = \omega_0$ (critically damped), $\gamma < \omega_0$ (underdamped).

**Problem 9.** Solve the following 2D Poisson equation:
$$\nabla^2 \phi = -\delta(\mathbf{r} - \mathbf{r}_1) + \delta(\mathbf{r} - \mathbf{r}_2)$$
$\mathbf{r}_1 = (1, 0)$, $\mathbf{r}_2 = (-1, 0)$. Visualize potential $\phi$ and electric field $\mathbf{E} = -\nabla\phi$.

**Problem 10.** Use Green's second identity to prove $G(x, x') = G(x', x)$ (symmetry of Green's functions for self-adjoint operators).

---

## Advanced Topics

### Dyadic Green's Functions

For inhomogeneous problems involving **vector fields** (e.g., Maxwell equations), Green's functions take tensor (dyadic) form:

$$\mathbf{E}(\mathbf{r}) = \int \overleftrightarrow{G}(\mathbf{r}, \mathbf{r}') \cdot \mathbf{J}(\mathbf{r}') \, d^3r'$$

### Frequency Domain Green's Functions

For time-dependent problems, Fourier transform gives:

$$G(\mathbf{r}, \mathbf{r}'; \omega) = \int_{-\infty}^{\infty} G(\mathbf{r}, t; \mathbf{r}', t') e^{i\omega(t-t')} d(t-t')$$

**Helmholtz equation** Green's function: $(\nabla^2 + k^2)G = \delta^3(\mathbf{r} - \mathbf{r}')$ $\rightarrow$ $G = -\frac{e^{ik|\mathbf{r}-\mathbf{r}'|}}{4\pi|\mathbf{r}-\mathbf{r}'|}$

### Numerical Green's Functions and Boundary Element Method (BEM)

For complex geometries where analytical Green's functions are difficult to obtain, the **Boundary Element Method** is used. Knowing the free space Green's function allows solutions via **boundary integrals** only instead of volume integrals, reducing dimensionality by one.

### References

- **Boas, M. L.** *Mathematical Methods in the Physical Sciences*, 3rd Ed., Ch. 13
- **Arfken, Weber, Harris.** *Mathematical Methods for Physicists*, 7th Ed., Ch. 10
- **Jackson, J. D.** *Classical Electrodynamics*, 3rd Ed., Ch. 1-2 (electrostatics Green's functions)
- **Stakgold, I., Holst, M.** *Green's Functions and Boundary Value Problems*, 3rd Ed. (2011)
- **Duffy, D. G.** *Green's Functions with Applications*, 2nd Ed. (2015)

---

**Previous**: [15. Laplace Transform](15_Laplace_Transform.md)
**Next**: [17. Calculus of Variations](17_Calculus_of_Variations.md)
