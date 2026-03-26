# 13. Partial Differential Equations

## Learning Objectives

- Classify partial differential equations (PDEs) as **elliptic, parabolic, or hyperbolic**, and set appropriate **boundary conditions and initial conditions** for each type
- Apply the **separation of variables** method in Cartesian, cylindrical, and spherical coordinates to decompose PDEs into ODEs
- Obtain analytical solutions of the **heat equation**, **wave equation**, and **Laplace's equation**, and interpret their physical meanings
- Describe wave propagation using **d'Alembert's solution**
- Apply PDE solution methods to **potential problems in electromagnetism** and the **infinite potential well in quantum mechanics**

---

## 1. Classification of PDEs

### 1.1 Elliptic, Parabolic, and Hyperbolic

General form of a second-order linear PDE:

$$A \frac{\partial^2 u}{\partial x^2} + 2B \frac{\partial^2 u}{\partial x \partial y} + C \frac{\partial^2 u}{\partial y^2} + \text{(lower order terms)} = 0$$

Based on the **discriminant** $\Delta = B^2 - AC$, PDEs are classified into three types:

| Discriminant | Type | Representative Equation | Physical Meaning |
|--------------|------|------------------------|------------------|
| $\Delta < 0$ | **Elliptic** | Laplace's equation $\nabla^2 u = 0$ | Steady state (equilibrium) |
| $\Delta = 0$ | **Parabolic** | Heat equation $u_t = \alpha^2 u_{xx}$ | Diffusion process |
| $\Delta > 0$ | **Hyperbolic** | Wave equation $u_{tt} = c^2 u_{xx}$ | Wave propagation |

This classification shares the same mathematical structure as the classification of conic sections.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Comparison of solution characteristics by PDE type ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Elliptic: harmonic function of Laplace's equation u = x² - y²
x, y = np.linspace(-2, 2, 100), np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
axes[0].contourf(X, Y, X**2 - Y**2, levels=20, cmap='RdBu_r')
axes[0].set_title('Elliptic: $\\nabla^2 u = 0$ (steady state)')

# Parabolic: time snapshots of heat equation solution
x_h = np.linspace(0, np.pi, 100)
for t in [0.0, 0.05, 0.2, 0.5, 1.0]:
    axes[1].plot(x_h, np.sin(x_h) * np.exp(-t), label=f't={t}')
axes[1].set_title('Parabolic: $u_t = \\alpha^2 u_{xx}$ (heat diffusion)')
axes[1].legend(fontsize=8)

# Hyperbolic: time snapshots of wave equation solution
x_w = np.linspace(0, 2*np.pi, 200)
for t in [0.0, 0.5, 1.0, 1.5]:
    axes[2].plot(x_w, np.sin(x_w - t) + np.sin(x_w + t), label=f't={t:.1f}')
axes[2].set_title('Hyperbolic: $u_{tt} = c^2 u_{xx}$ (wave propagation)')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()
```

### 1.2 Boundary and Initial Conditions

To determine a unique solution to a PDE, appropriate **boundary conditions (BC)** and **initial conditions (IC)** are required.

| Type | Name | Condition | Physical Meaning |
|------|------|-----------|------------------|
| First kind | **Dirichlet** | $u = f$ on $\partial\Omega$ | Specify temperature/potential on boundary |
| Second kind | **Neumann** | $\partial u / \partial n = g$ on $\partial\Omega$ | Specify heat flux/electric field on boundary |
| Third kind | **Robin** | $\alpha u + \beta \partial u / \partial n = h$ | Convective heat transfer (Newton cooling) |

**Hadamard's well-posed problem** conditions: a solution must (1) exist, (2) be unique, and (3) depend continuously on the data.

- **Elliptic**: Boundary conditions only (BVP)
- **Parabolic**: Initial conditions + boundary conditions (IBVP)
- **Hyperbolic**: Initial conditions (both $u$ and $u_t$) + boundary conditions

---

## 2. Separation of Variables

### 2.1 Separation in Cartesian Coordinates

Core idea: Assume the solution as a product of functions of each variable $u(x, t) = X(x) T(t)$.

**Example - 1D Heat Equation** ($u(0,t) = u(L,t) = 0$, $u(x,0) = f(x)$):

$$\frac{\partial u}{\partial t} = \alpha^2 \frac{\partial^2 u}{\partial x^2}$$

Substituting $u = X(x)T(t)$ and dividing both sides by $\alpha^2 XT$:

$$\frac{T'}{\alpha^2 T} = \frac{X''}{X} = -\lambda \quad (\text{separation constant})$$

This separates into two ODEs: $X'' + \lambda X = 0$ (with boundary conditions) and $T' + \alpha^2 \lambda T = 0$.

Eigenvalues: $\lambda_n = (n\pi/L)^2$, eigenfunctions: $X_n = \sin(n\pi x/L)$. General solution:

$$u(x, t) = \sum_{n=1}^{\infty} b_n \sin\left(\frac{n\pi x}{L}\right) e^{-\alpha^2 (n\pi/L)^2 t}, \quad b_n = \frac{2}{L}\int_0^L f(x)\sin\left(\frac{n\pi x}{L}\right)dx$$

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Separation of variables: 1D heat equation solution ---
L, alpha, N_terms = np.pi, 1.0, 50

def initial_condition(x):
    return x * (np.pi - x)

# Fourier sine coefficients (numerical integration)
def compute_bn(n):
    x = np.linspace(0, L, 1000)
    return (2/L) * np.trapz(initial_condition(x) * np.sin(n*np.pi*x/L), x)

coeffs = [compute_bn(n) for n in range(1, N_terms + 1)]

def heat_solution(x, t):
    u = np.zeros_like(x, dtype=float)
    for n in range(1, N_terms + 1):
        u += coeffs[n-1] * np.sin(n*np.pi*x/L) * np.exp(-alpha**2*(n*np.pi/L)**2*t)
    return u

x = np.linspace(0, L, 200)
plt.figure(figsize=(10, 6))
for t in [0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]:
    plt.plot(x, heat_solution(x, t), label=f't = {t}')
plt.xlabel('x'); plt.ylabel('u(x, t)')
plt.title('1D heat equation solution (separation of variables)')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

print("=== Fourier sine coefficients (first 5) ===")
for n in range(1, 6):
    print(f"  b_{n} = {coeffs[n-1]:.6f}")
```

### 2.2 Separation in Cylindrical Coordinates

Substituting $u = R(r)\Phi(\phi)Z(z)$ into Laplace's equation in cylindrical coordinates $(r, \phi, z)$ separates it into three ODEs:

$$Z'' - k^2 Z = 0, \qquad \Phi'' + m^2 \Phi = 0, \qquad r^2 R'' + rR' + (k^2r^2 - m^2)R = 0$$

The last equation is the **Bessel equation**, whose solutions are $J_m(kr)$ and $Y_m(kr)$. In regions including the origin, $Y_m$ diverges, so only $R(r) = J_m(kr)$ is taken.

### 2.3 Separation in Spherical Coordinates

In spherical coordinates $(r, \theta, \phi)$, setting $u = R(r)\Theta(\theta)\Phi(\phi)$:

- $\Phi'' + m^2\Phi = 0$ $\Rightarrow$ $\Phi = e^{im\phi}$
- $\Theta$: **Associated Legendre equation** $\Rightarrow$ $P_l^m(\cos\theta)$
- $R$: $r^2R'' + 2rR' - l(l+1)R = 0$ $\Rightarrow$ $R = Ar^l + Br^{-(l+1)}$

The general solution is expressed in terms of **spherical harmonics** $Y_l^m(\theta,\phi)$. For axisymmetric cases ($m=0$):

$$u(r, \theta) = \sum_{l=0}^{\infty}\left(A_l r^l + B_l r^{-(l+1)}\right)P_l(\cos\theta)$$

---

## 3. Heat/Diffusion Equation

### 3.1 Solution of the 1D Heat Equation

$$\frac{\partial u}{\partial t} = \alpha^2 \frac{\partial^2 u}{\partial x^2}, \qquad \alpha^2 = \frac{k}{\rho c_p}$$

**Physical meaning**: The rate of temperature change is proportional to the spatial curvature. Convex regions ($u_{xx} < 0$) cool, while concave regions ($u_{xx} > 0$) heat.

### 3.2 Nonhomogeneous Boundary Conditions

For $u(0,t) = T_1$, $u(L,t) = T_2$, separate out the **steady-state solution**:

$$u(x, t) = \underbrace{T_1 + \frac{T_2 - T_1}{L}x}_{v(x) \text{ (steady state)}} + w(x, t)$$

$w$ satisfies homogeneous BC ($w(0,t) = w(L,t) = 0$) and is determined by the initial condition $w(x,0) = f(x) - v(x)$.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Heat equation with nonhomogeneous boundary conditions ---
L, alpha, T1, T2, N_terms = 1.0, 0.1, 100.0, 50.0, 30

steady = lambda x: T1 + (T2 - T1) * x / L
w_init = lambda x: -steady(x)  # initial: uniform 0°C

# Fourier coefficients of transient solution
cn = [(2/L) * np.trapz(w_init(np.linspace(0,L,1000)) * np.sin(n*np.pi*np.linspace(0,L,1000)/L),
      np.linspace(0,L,1000)) for n in range(1, N_terms+1)]

def full_solution(x, t):
    u = steady(x)
    for n in range(1, N_terms+1):
        u += cn[n-1] * np.sin(n*np.pi*x/L) * np.exp(-alpha**2*(n*np.pi/L)**2*t)
    return u

x = np.linspace(0, L, 200)
plt.figure(figsize=(10, 6))
for t in [0.0, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0]:
    plt.plot(x, full_solution(x, t), label=f't = {t}')
plt.plot(x, steady(x), 'k--', lw=2, label='steady state')
plt.xlabel('x'); plt.ylabel('u(x,t) [°C]')
plt.title('Heat equation with nonhomogeneous boundary conditions'); plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
```

### 3.3 Time Evolution and Steady State

**Time constant** of the $n$-th mode: $\tau_n = 1 / [\alpha^2(n\pi/L)^2]$. Higher-frequency modes decay faster, so after sufficient time, only the $n=1$ mode remains, and as $t \to \infty$, the steady state $u \to v(x)$ is reached.

---

## 4. Wave Equation

### 4.1 1D Wave Equation and d'Alembert's Solution

$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$$

**D'Alembert's solution**: For an infinite domain with $u(x,0)=f(x)$, $u_t(x,0)=g(x)$:

$$u(x,t) = \frac{1}{2}[f(x-ct) + f(x+ct)] + \frac{1}{2c}\int_{x-ct}^{x+ct} g(s)\,ds$$

**Finite interval** ($u(0,t) = u(L,t) = 0$) separation of variables:

$$u(x,t) = \sum_{n=1}^{\infty}\sin\left(\frac{n\pi x}{L}\right)\left[A_n\cos(\omega_n t) + B_n\sin(\omega_n t)\right], \quad \omega_n = \frac{n\pi c}{L}$$

```python
import numpy as np
import matplotlib.pyplot as plt

# --- D'Alembert solution: left-right propagation of Gaussian pulse ---
c = 1.0
f = lambda x: np.exp(-10 * x**2)
dalembert = lambda x, t: 0.5 * (f(x - c*t) + f(x + c*t))

x = np.linspace(-5, 5, 500)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for idx, t in enumerate([0.0, 0.5, 1.0, 1.5, 2.0, 3.0]):
    ax = axes[idx//3, idx%3]
    ax.plot(x, dalembert(x, t), 'b-', lw=2)
    ax.fill_between(x, dalembert(x, t), alpha=0.2)
    ax.set_xlim(-5, 5); ax.set_ylim(-0.3, 1.1)
    ax.set_title(f't = {t:.1f}'); ax.grid(True, alpha=0.3)
fig.suptitle("D'Alembert solution: left-right propagation of Gaussian pulse", fontsize=14)
plt.tight_layout(); plt.show()
```

**Guitar string vibration** (triangular initial displacement with center plucked):

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Guitar string vibration (separation of variables) ---
L, c, N = 1.0, 1.0, 20
An = lambda n: 8/(n**2 * np.pi**2) * np.sin(n*np.pi/2)

def string_sol(x, t):
    u = np.zeros_like(x, dtype=float)
    for n in range(1, N+1):
        u += An(n) * np.sin(n*np.pi*x/L) * np.cos(n*np.pi*c*t/L)
    return u

x = np.linspace(0, L, 200)
T_period = 2*L/c
plt.figure(figsize=(10, 6))
for t in [0, T_period/8, T_period/4, 3*T_period/8, T_period/2]:
    plt.plot(x, string_sol(x, t), label=f't = {t:.3f} ({t/T_period:.0%} T)')
plt.xlabel('x'); plt.ylabel('u(x,t)')
plt.title(f'Guitar string vibration (fundamental period T = {T_period:.2f})')
plt.legend(); plt.grid(True, alpha=0.3); plt.axhline(0, color='k', lw=0.5)
plt.show()

print("=== Natural frequencies ===")
for n in range(1, 6):
    print(f"  n={n}: f_{n} = {n*c/(2*L):.2f} Hz")
```

### 4.2 2D Circular Membrane Vibration (Drum)

The axisymmetric modes of a circular membrane are given by $u(r,t) = J_0(\lambda r)[A\cos(\lambda ct) + B\sin(\lambda ct)]$ from the cylindrical wave equation. From the boundary condition $u(a,t) = 0$, $\lambda_{0n} = j_{0n}/a$ ($j_{0n}$: zeros of $J_0$).

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn_zeros, j0

# --- Axisymmetric modes of circular membrane vibration ---
a = 1.0
zeros_J0 = jn_zeros(0, 3)  # [2.4048, 5.5201, 8.6537]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
r = np.linspace(0, a, 100)
for idx, j0n in enumerate(zeros_J0):
    R = j0(j0n * r / a)
    axes[idx].plot(r, R, 'b-', lw=2)
    axes[idx].fill_between(r, R, alpha=0.2)
    axes[idx].axhline(0, color='k', lw=0.5)
    axes[idx].set_title(f'Mode (0,{idx+1}): $j_{{0,{idx+1}}}$ = {j0n:.4f}')
    axes[idx].set_xlabel('r'); axes[idx].grid(True, alpha=0.3)
plt.suptitle('Axisymmetric vibration modes of circular membrane', fontsize=14)
plt.tight_layout(); plt.show()
```

---

## 5. Laplace's Equation

### 5.1 Rectangular Domain

$\nabla^2 u = 0$, three sides grounded, top side $u(x,b) = f(x)$:

$$u(x,y) = \sum_{n=1}^{\infty} c_n \sin\left(\frac{n\pi x}{a}\right)\sinh\left(\frac{n\pi y}{a}\right), \quad c_n = \frac{2}{a\sinh(n\pi b/a)}\int_0^a f(x)\sin\left(\frac{n\pi x}{a}\right)dx$$

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Laplace's equation on rectangular domain ---
a, b, N = 2.0, 1.0, 30
f_top = lambda x: np.sin(np.pi * x / a)

xq = np.linspace(0, a, 1000)
cn = [(2/(a*np.sinh(n*np.pi*b/a))) * np.trapz(f_top(xq)*np.sin(n*np.pi*xq/a), xq)
      for n in range(1, N+1)]

def laplace_rect(x, y):
    u = np.zeros_like(x, dtype=float)
    for n in range(1, N+1):
        u += cn[n-1] * np.sin(n*np.pi*x/a) * np.sinh(n*np.pi*y/a)
    return u

x, y = np.linspace(0, a, 100), np.linspace(0, b, 100)
X, Y = np.meshgrid(x, y)
fig, ax = plt.subplots(figsize=(10, 5))
cs = ax.contourf(X, Y, laplace_rect(X, Y), levels=30, cmap='hot')
plt.colorbar(cs, label='u(x,y)')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
ax.set_title('Laplace equation solution on rectangular domain')
plt.tight_layout(); plt.show()
```

### 5.2 Circular/Spherical Domains

**Circular domain** ($r < a$): $u(r,\theta) = a_0/2 + \sum_{n=1}^{\infty}(r/a)^n(a_n\cos n\theta + b_n\sin n\theta)$

**Spherical domain** (axisymmetric): $u(r,\theta) = \sum_{l=0}^{\infty} A_l (r/a)^l P_l(\cos\theta)$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

# --- Laplace's equation on spherical domain ---
a, L_max = 1.0, 10
f_bdy = lambda th: np.cos(th)**2  # u(a,θ) = cos²θ

theta_q = np.linspace(0, np.pi, 1000)
Al = [(2*l+1)/2 * np.trapz(f_bdy(theta_q)*legendre(l)(np.cos(theta_q))*np.sin(theta_q), theta_q)
      for l in range(L_max+1)]

print("=== Legendre coefficients ===")
for l in range(5): print(f"  A_{l} = {Al[l]:.6f}")  # A_0≈1/3, A_2≈2/3

# Visualization (cross-section)
r_v, th_v = np.linspace(0.01, a, 50), np.linspace(0, np.pi, 100)
R, Th = np.meshgrid(r_v, th_v)
U = sum(Al[l]*(R/a)**l * legendre(l)(np.cos(Th)) for l in range(L_max+1))

fig, ax = plt.subplots(figsize=(6, 8))
cs = ax.contourf(R*np.sin(Th), R*np.cos(Th), U, levels=30, cmap='coolwarm')
plt.colorbar(cs, label='u(r,θ)')
th_c = np.linspace(0, np.pi, 100)
ax.plot(a*np.sin(th_c), a*np.cos(th_c), 'k-', lw=2)
ax.set_xlabel('r sin(θ)'); ax.set_ylabel('r cos(θ)'); ax.set_aspect('equal')
ax.set_title('Laplace equation solution inside sphere: $u(a,\\theta)=\\cos^2\\theta$')
plt.tight_layout(); plt.show()
```

### 5.3 Poisson's Equation

**Poisson's equation** $\nabla^2 u = f(\mathbf{r})$ is the nonhomogeneous form of Laplace's equation. In electromagnetism, it appears as $\nabla^2\phi = -\rho/\epsilon_0$.

Solution: $u = u_h + u_p$ ($\nabla^2 u_h = 0$, $\nabla^2 u_p = f$). The particular solution is obtained using **Green's function** (see Lesson 16):

$$u_p(\mathbf{r}) = \int G(\mathbf{r}, \mathbf{r}') f(\mathbf{r}') \, d^3r'$$

---

## 6. Helmholtz Equation

### 6.1 Definition and Physical Background

Separating the time-dependent part from the wave equation yields the **Helmholtz equation**:

$$\nabla^2 u + k^2 u = 0$$

Here $k = \omega/c$ is the wavenumber. This is a generalization of Laplace's equation ($k = 0$).

**Physical applications**:

| Field | Equation | Meaning of $k$ |
|-------|----------|---------------|
| Acoustics | $\nabla^2 p + k^2 p = 0$ | $k = \omega/c_s$ (sound wave wavenumber) |
| Electromagnetism | $\nabla^2 \mathbf{E} + k^2 \mathbf{E} = 0$ | $k = \omega\sqrt{\mu\epsilon}$ |
| Quantum mechanics | $\nabla^2 \psi + \frac{2mE}{\hbar^2}\psi = 0$ | $k^2 = 2mE/\hbar^2$ |
| Diffusion (steady) | $\nabla^2 c - \alpha^2 c = 0$ | $k^2 = -\alpha^2 < 0$ (modified Helmholtz) |

### 6.2 Solution in Cartesian Coordinates

For a rectangular domain $[0, a] \times [0, b]$ with Dirichlet boundary conditions:

$$u_{n_x, n_y}(x, y) = \sin\frac{n_x \pi x}{a} \sin\frac{n_y \pi y}{b}$$

Eigenvalues: $k^2_{n_x, n_y} = \left(\frac{n_x \pi}{a}\right)^2 + \left(\frac{n_y \pi}{b}\right)^2$

### 6.3 Helmholtz Equation in Cylindrical Coordinates

$\nabla^2 u + k^2 u = 0$ in cylindrical coordinates $(r, \phi, z)$ with separation $u = R(r)\Phi(\phi)Z(z)$:

$$R'' + \frac{1}{r}R' + \left(\kappa^2 - \frac{m^2}{r^2}\right)R = 0$$

where $\kappa^2 = k^2 - k_z^2$. This is the **Bessel equation** with solution $J_m(\kappa r)$.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, jn_zeros

# --- Helmholtz equation modes of circular waveguide ---
a = 1.0  # waveguide radius
modes = [(0, 1), (1, 1), (2, 1), (0, 2)]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for idx, (m, n) in enumerate(modes):
    ax = axes[idx // 2, idx % 2]
    alpha_mn = jn_zeros(m, n)[-1]

    r = np.linspace(0, a, 100)
    theta = np.linspace(0, 2 * np.pi, 200)
    R, Theta = np.meshgrid(r, theta)
    X, Y = R * np.cos(Theta), R * np.sin(Theta)

    Z = jv(m, alpha_mn * R / a) * np.cos(m * Theta)
    ax.contourf(X, Y, Z, levels=20, cmap='RdBu_r')
    ax.set_title(f'TM$_{{{m}{n}}}$ mode, $k_c a$ = {alpha_mn:.3f}')
    ax.set_aspect('equal')
    circle = plt.Circle((0, 0), a, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)

plt.suptitle('Circular waveguide Helmholtz modes', fontsize=14)
plt.tight_layout()
plt.show()
```

### 6.4 Helmholtz Equation in Spherical Coordinates

Separation in spherical coordinates leads to the **spherical Bessel equation** for the radial part:

$$r^2 R'' + 2rR' + [k^2 r^2 - l(l+1)]R = 0$$

Solution: Spherical Bessel function $j_l(kr) = \sqrt{\frac{\pi}{2kr}} J_{l+1/2}(kr)$

$$u_{lm}(r, \theta, \phi) = j_l(kr) Y_l^m(\theta, \phi)$$

---

## 7. Nonhomogeneous PDEs and Eigenfunction Expansion

### 7.1 Nonhomogeneous Heat Equation

$$\frac{\partial u}{\partial t} = \alpha^2 \frac{\partial^2 u}{\partial x^2} + F(x, t)$$

where $F(x, t)$ is a source term representing heat source/sink. With homogeneous boundary conditions $u(0,t) = u(L,t) = 0$, expand the solution in eigenfunctions:

$$u(x, t) = \sum_{n=1}^{\infty} T_n(t) \sin\frac{n\pi x}{L}$$

Also expand the source term:

$$F(x, t) = \sum_{n=1}^{\infty} F_n(t) \sin\frac{n\pi x}{L}$$

Substituting yields a first-order ODE for each mode:

$$T_n'(t) + \alpha^2\left(\frac{n\pi}{L}\right)^2 T_n(t) = F_n(t)$$

This can be solved using the integrating factor method.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Nonhomogeneous heat equation: sinusoidal heat source ---
L, alpha2, N = np.pi, 1.0, 30

# Heat source: F(x,t) = sin(x) (spatially nonuniform steady heat source)
# F_n = (2/L) * ∫₀ᴸ sin(x)sin(nx) dx = δ_{n,1}
# T_1' + T_1 = 1, T_1(0) = 0 → T_1(t) = 1 - e^{-t}
# Remaining T_n(0) = 0, F_n = 0 → T_n(t) = 0

x = np.linspace(0, L, 200)
plt.figure(figsize=(10, 6))
for t in [0.0, 0.1, 0.3, 0.5, 1.0, 3.0]:
    T1 = 1 - np.exp(-t)
    u = T1 * np.sin(x)
    plt.plot(x, u, label=f't = {t}')

plt.plot(x, np.sin(x), 'k--', lw=2, label='steady state')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('Nonhomogeneous heat equation: sinusoidal heat source')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 7.2 Nonhomogeneous Wave Equation and Resonance

In a nonhomogeneous wave equation, when the external force frequency matches a natural frequency, **resonance** occurs. The amplitude grows linearly in time:

$$u_{tt} = c^2 u_{xx} + F_0 \sin\left(\frac{n\pi x}{L}\right)\cos(\omega_n t)$$

At exact resonance ($\omega = \omega_n$), the particular solution is proportional to $t\sin(\omega_n t)$, showing linear growth.

---

## 8. Uniqueness Theorems

### 8.1 Energy Method

**Theorem**: The solution of the Dirichlet problem for the heat equation $u_t = \alpha^2 \nabla^2 u$ in a bounded domain $\Omega$ is unique.

**Proof**: If two solutions $u_1, u_2$ exist, their difference $w = u_1 - u_2$ satisfies the homogeneous equation with zero boundary/initial conditions.

$$E(t) = \int_\Omega w^2 \, dV \geq 0$$

$$\frac{dE}{dt} = 2\int_\Omega w \, w_t \, dV = 2\alpha^2 \int_\Omega w \nabla^2 w \, dV = -2\alpha^2 \int_\Omega |\nabla w|^2 \, dV \leq 0$$

Thus $E(t) \leq E(0) = 0$, so $w \equiv 0$, i.e., $u_1 = u_2$. $\blacksquare$

### 8.2 Uniqueness for Laplace's Equation

**Dirichlet problem**: $\nabla^2 u = 0$ in $\Omega$, $u = f$ on $\partial\Omega$ — the solution is unique.

**Neumann problem**: $\nabla^2 u = 0$ in $\Omega$, $\partial u/\partial n = g$ on $\partial\Omega$ — the solution is unique up to an additive constant.

### 8.3 Maximum Principle

**Theorem**: For a harmonic function ($\nabla^2 u = 0$) in a domain $\Omega$, the maximum and minimum values are attained on the **boundary** $\partial\Omega$.

**Physical meaning**: In a steady state with no heat sources/sinks, the interior temperature always lies between the boundary temperatures. This is the physical reason why boundary conditions alone can determine the entire interior temperature.

---

## 9. Applications in Physics

### 9.1 Electromagnetism: Potential Problems

Fundamental equation of electrostatics $\nabla^2\phi = -\rho/\epsilon_0$. In the absence of charges, it becomes Laplace's equation.

**Example**: Interior potential of a rectangular conductor box with one face at $V_0$ and other faces grounded:

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Interior potential of rectangular conductor box ---
a, b, c_box, V0, N = 1.0, 1.0, 1.0, 100.0, 15

def potential_box(x, y, z_val):
    u = np.zeros_like(x, dtype=float)
    for m in range(1, N+1, 2):     # odd terms only
        for n in range(1, N+1, 2):
            gamma = np.pi * np.sqrt((m/a)**2 + (n/b)**2)
            Amn = 16*V0 / (m*n*np.pi**2 * np.sinh(gamma*c_box))
            u += Amn * np.sin(m*np.pi*x/a) * np.sin(n*np.pi*y/b) * np.sinh(gamma*z_val)
    return u

x, y = np.linspace(0, a, 100), np.linspace(0, b, 100)
X, Y = np.meshgrid(x, y)
fig, ax = plt.subplots(figsize=(8, 6))
cs = ax.contourf(X, Y, potential_box(X, Y, 0.5), levels=30, cmap='RdYlBu_r')
plt.colorbar(cs, label='$\\phi$ [V]')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
ax.set_title('Interior potential of rectangular conductor box (z = 0.5)')
plt.tight_layout(); plt.show()
```

### 9.2 Quantum Mechanics: Infinite Potential Well

**Time-independent Schrödinger equation**: $-(\hbar^2/2m)\nabla^2\psi + V\psi = E\psi$

**1D infinite well** ($0 < x < L$):

$$\psi_n(x) = \sqrt{\frac{2}{L}}\sin\left(\frac{n\pi x}{L}\right), \quad E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}$$

**2D square well** ($a = b$): $E_{n_x,n_y} = \frac{\pi^2\hbar^2}{2m}(n_x^2 + n_y^2)/a^2$. **Degeneracy** occurs where $(1,2)$ and $(2,1)$ have the same energy.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- 2D infinite potential well ---
a = 1.0  # natural units: ℏ = m = 1
psi = lambda x, y, nx, ny: 2/a * np.sin(nx*np.pi*x/a) * np.sin(ny*np.pi*y/a)
E = lambda nx, ny: np.pi**2/2 * (nx**2 + ny**2) / a**2

x, y = np.linspace(0, a, 100), np.linspace(0, a, 100)
X, Y = np.meshgrid(x, y)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for idx, (nx, ny) in enumerate([(1,1),(1,2),(2,1),(2,2),(1,3),(3,1)]):
    ax = axes[idx//3, idx%3]
    prob = np.abs(psi(X, Y, nx, ny))**2
    cs = ax.contourf(X, Y, prob, levels=30, cmap='inferno')
    plt.colorbar(cs, ax=ax, label='$|\\psi|^2$')
    ax.set_title(f'$(n_x,n_y)=({nx},{ny})$, $E={E(nx,ny)/E(1,1):.1f}\\,E_{{11}}$')
    ax.set_aspect('equal')
fig.suptitle('Probability density distribution of 2D infinite potential well', fontsize=14)
plt.tight_layout(); plt.show()

# Check degeneracy
print(f"E(1,2) = {E(1,2):.4f}, E(2,1) = {E(2,1):.4f}")
print(f"Degenerate: E(1,2) = E(2,1) → {'Yes' if abs(E(1,2)-E(2,1))<1e-10 else 'No'}")
```

The **time-dependent Schrödinger equation** is a parabolic PDE. Separation of variables yields $\Psi = \psi(\mathbf{r})e^{-iEt/\hbar}$, and the general solution is a superposition of stationary states: $\Psi(\mathbf{r},t) = \sum_n c_n \psi_n(\mathbf{r})e^{-iE_n t/\hbar}$.

---

## Practice Problems

### Problem 1: PDE Classification

Classify the following PDEs: (a) $u_{xx}+4u_{xy}+u_{yy}=0$ (b) $u_{xx}+2u_{xy}+u_{yy}=0$ (c) $3u_{xx}-6u_{yy}=0$

**Hint**: $\Delta = B^2 - AC$. Note that the cross-term coefficient is $2B$.

### Problem 2: Heat Equation

For $L=\pi$, $\alpha=1$, both ends at 0°C, initial $f(x)=\sin x + 3\sin 2x$:
(a) Find $u(x,t)$. (b) Plot temperature at $t=0.5$. (c) How many times faster does $n=2$ decay than $n=1$?

### Problem 3: Wave Equation

For $L=1$, $c=2$, $u(x,0)=0$, $u_t(x,0)=\sin(\pi x)$: (a) Find $u(x,t)$. (b) Find the fundamental frequency.

### Problem 4: Laplace's Equation

In a unit disk with $\nabla^2 u = 0$, $u(1,\theta) = \cos^2\theta$:
(a) Fourier expansion of $\cos^2\theta$. (b) Find $u(r,\theta)$. (c) Temperature at the center and its physical meaning.

### Problem 5: Quantum Mechanics

For a 2D square well: (a) First 5 energy levels and their degeneracies. (b) Compare $|\psi|^2$ for $(1,2)$ and $(2,1)$.

### Problem 6: Helmholtz Equation

In a circular region of radius $a$ with $\nabla^2 u + k^2 u = 0$, $u(a, \theta) = 0$. Find the axisymmetric ($m=0$) eigenvalues $k_{0n}$ and eigenfunctions.

### Problem 7: Nonhomogeneous PDE

Solve $u_t = u_{xx} + 2\sin(3\pi x)$, $u(0,t) = u(1,t) = 0$, $u(x,0) = \sin(\pi x)$ using the eigenfunction expansion method.

### Problem 8: Uniqueness

Using the energy method, prove that the solution to the wave equation $u_{tt} = c^2 u_{xx}$ ($0 < x < L$) with Dirichlet boundary conditions and initial conditions is unique. (Hint: Use $E(t) = \int_0^L (w_t^2 + c^2 w_x^2) dx$)

---

## References

### Textbooks
1. **Boas, M. L.** (2005). *Mathematical Methods in the Physical Sciences*, 3rd ed., Ch. 13. Wiley.
2. **Haberman, R.** (2013). *Applied Partial Differential Equations*, 5th ed. Pearson.
3. **Strauss, W. A.** (2008). *Partial Differential Equations: An Introduction*, 2nd ed. Wiley.
4. **Jackson, J. D.** (1999). *Classical Electrodynamics*, 3rd ed. Wiley. — Electromagnetism applications of the Helmholtz equation

### Related Lessons
- [07. Fourier Series](07_Fourier_Series.md): Core tool for separation of variables
- [11. Series Solutions and Special Functions](11_Series_Solutions_Special_Functions.md): Bessel and Legendre functions
- [12. Sturm-Liouville Theory](12_Sturm_Liouville_Theory.md): General framework for eigenvalue problems
- [16. Green's Functions](16_Greens_Functions.md): Systematic solution method for nonhomogeneous PDEs

---

## Next Lesson

[14. Complex Analysis](14_Complex_Analysis.md) covers differentiation and integration of complex-variable functions, the Cauchy-Riemann conditions, conformal mapping, and the residue theorem. We will also explore techniques for transforming boundary value problems of PDEs using conformal mapping.
