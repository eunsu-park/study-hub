# Fourier Series and PDE

## Learning Objectives

- Compute Fourier coefficients for periodic functions and interpret them as projections onto orthogonal basis functions
- Determine when to use full Fourier series, cosine series, or sine series based on boundary conditions
- Solve the heat equation on a finite interval using separation of variables combined with Fourier series
- Solve the wave equation using both the Fourier method and D'Alembert's formula
- Implement Fourier series computations and animate PDE solutions in Python

## Prerequisites

Before studying this lesson, you should be comfortable with:
- Introduction to PDE, classification, and boundary conditions (Lesson 17)
- Integration techniques including integration by parts (Lessons 1-4)
- Basic trigonometric identities and orthogonality

## The Big Idea: Decomposition into Simple Pieces

The strategy for solving PDE on finite domains has three steps:

1. **Separation of variables**: Assume $u(x,t) = X(x)T(t)$ and split the PDE into two ODE
2. **Solve the spatial ODE**: The boundary conditions force specific eigenvalues and eigenfunctions
3. **Superpose**: Use Fourier series to match the initial condition as a sum of eigenfunctions

This works because the PDE is linear: the sum of solutions is again a solution. Think of it like decomposing white light into its spectral colors (Fourier modes), solving for how each color evolves separately, then recombining.

## Fourier Series: The Mathematical Foundation

### Periodic Functions and Orthogonality

A function $f(x)$ with period $2L$ can be represented as:

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[a_n \cos\left(\frac{n\pi x}{L}\right) + b_n \sin\left(\frac{n\pi x}{L}\right)\right]$$

The key property making this work is **orthogonality** on $[-L, L]$:

$$\int_{-L}^{L} \cos\left(\frac{m\pi x}{L}\right) \cos\left(\frac{n\pi x}{L}\right) dx = \begin{cases} 0 & m \neq n \\ L & m = n \neq 0 \\ 2L & m = n = 0 \end{cases}$$

$$\int_{-L}^{L} \sin\left(\frac{m\pi x}{L}\right) \sin\left(\frac{n\pi x}{L}\right) dx = \begin{cases} 0 & m \neq n \\ L & m = n \end{cases}$$

$$\int_{-L}^{L} \cos\left(\frac{m\pi x}{L}\right) \sin\left(\frac{n\pi x}{L}\right) dx = 0 \quad \text{for all } m, n$$

Think of this like vector decomposition in $\mathbb{R}^3$: to find the component of a vector along $\hat{e}_1$, you take the dot product with $\hat{e}_1$. Here, the "dot product" is the integral, and the "basis vectors" are sines and cosines.

### Fourier Coefficients

The coefficients are computed by "projecting" $f$ onto each basis function:

$$a_0 = \frac{1}{L} \int_{-L}^{L} f(x) \, dx$$

$$a_n = \frac{1}{L} \int_{-L}^{L} f(x) \cos\left(\frac{n\pi x}{L}\right) dx, \quad n \geq 1$$

$$b_n = \frac{1}{L} \int_{-L}^{L} f(x) \sin\left(\frac{n\pi x}{L}\right) dx, \quad n \geq 1$$

The factor $a_0/2$ in the series (rather than $a_0$) is a convention that makes the formula for $a_0$ consistent with the general $a_n$ formula.

### Convergence: Dirichlet Conditions

The Fourier series converges to $f(x)$ at points of continuity if $f$ is piecewise smooth (finitely many discontinuities and corners on each period). At a discontinuity, the series converges to the average of the left and right limits:

$$\frac{f(x^-) + f(x^+)}{2}$$

Near discontinuities, the partial sums exhibit **Gibbs phenomenon**: an overshoot of about 9% that does not diminish as more terms are added (though the overshoot region shrinks).

### Even and Odd Functions: Half-Range Expansions

If $f(x)$ is defined only on $[0, L]$ (as is typical for PDE on a finite interval), we can extend it to get either:

**Cosine series** (even extension, $b_n = 0$):

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} a_n \cos\left(\frac{n\pi x}{L}\right), \quad a_n = \frac{2}{L}\int_0^L f(x) \cos\left(\frac{n\pi x}{L}\right) dx$$

Appropriate when the boundary conditions involve zero derivatives (Neumann): $u_x(0,t) = u_x(L,t) = 0$.

**Sine series** (odd extension, $a_n = 0$):

$$f(x) = \sum_{n=1}^{\infty} b_n \sin\left(\frac{n\pi x}{L}\right), \quad b_n = \frac{2}{L}\int_0^L f(x) \sin\left(\frac{n\pi x}{L}\right) dx$$

Appropriate when the boundary conditions prescribe zero values (Dirichlet): $u(0,t) = u(L,t) = 0$.

### Worked Example: Square Wave

Find the Fourier sine series of $f(x) = 1$ on $[0, \pi]$.

$$b_n = \frac{2}{\pi}\int_0^{\pi} 1 \cdot \sin(nx) \, dx = \frac{2}{\pi}\left[-\frac{\cos(nx)}{n}\right]_0^{\pi} = \frac{2}{n\pi}(1 - \cos(n\pi)) = \frac{2}{n\pi}(1 - (-1)^n)$$

So $b_n = 0$ for even $n$ and $b_n = \frac{4}{n\pi}$ for odd $n$:

$$f(x) = \frac{4}{\pi}\left(\sin x + \frac{\sin 3x}{3} + \frac{\sin 5x}{5} + \cdots\right)$$

The slow $1/n$ decay of coefficients reflects the discontinuity in the square wave. Smooth functions have rapidly decaying coefficients.

## Solving the Heat Equation

### Problem Setup

$$u_t = \alpha u_{xx}, \quad 0 < x < L, \quad t > 0$$
$$u(0, t) = 0, \quad u(L, t) = 0 \quad \text{(Dirichlet BCs)}$$
$$u(x, 0) = f(x) \quad \text{(initial condition)}$$

### Step 1: Separation of Variables

Assume $u(x, t) = X(x)T(t)$. Substituting:

$$X(x)T'(t) = \alpha X''(x)T(t)$$

Divide by $\alpha X T$:

$$\frac{T'(t)}{\alpha T(t)} = \frac{X''(x)}{X(x)}$$

The left side depends only on $t$ and the right side only on $x$. Since they are equal for all $x$ and $t$, both must equal a constant, say $-\lambda$:

$$\frac{X''}{X} = -\lambda \quad \Longrightarrow \quad X'' + \lambda X = 0$$

$$\frac{T'}{\alpha T} = -\lambda \quad \Longrightarrow \quad T' + \alpha\lambda T = 0$$

### Step 2: Solve the Spatial Problem (Eigenvalue Problem)

With boundary conditions $X(0) = 0$ and $X(L) = 0$:

For $\lambda > 0$ (writing $\lambda = \mu^2$): $X = A\cos(\mu x) + B\sin(\mu x)$.

$X(0) = 0$ gives $A = 0$. $X(L) = 0$ gives $B\sin(\mu L) = 0$.

For nontrivial solutions ($B \neq 0$): $\sin(\mu L) = 0$, so $\mu_n = \frac{n\pi}{L}$.

**Eigenvalues**: $\lambda_n = \left(\frac{n\pi}{L}\right)^2$, $n = 1, 2, 3, \ldots$

**Eigenfunctions**: $X_n(x) = \sin\left(\frac{n\pi x}{L}\right)$

### Step 3: Solve the Time Problem

For each $n$: $T_n' + \alpha\lambda_n T_n = 0$, giving:

$$T_n(t) = e^{-\alpha \lambda_n t} = e^{-\alpha n^2 \pi^2 t / L^2}$$

This is an exponential decay. Higher modes (larger $n$) decay **faster** because $\lambda_n \propto n^2$. This is why the heat equation smooths out sharp features quickly: the high-frequency components of the initial temperature die away first.

### Step 4: Superpose and Match Initial Condition

The general solution is:

$$u(x, t) = \sum_{n=1}^{\infty} B_n \sin\left(\frac{n\pi x}{L}\right) e^{-\alpha n^2 \pi^2 t / L^2}$$

At $t = 0$: $u(x, 0) = \sum_{n=1}^{\infty} B_n \sin\left(\frac{n\pi x}{L}\right) = f(x)$.

The $B_n$ are the Fourier sine coefficients of $f(x)$:

$$B_n = \frac{2}{L}\int_0^L f(x) \sin\left(\frac{n\pi x}{L}\right) dx$$

**Complete solution**: The temperature at any point $(x, t)$ is determined by the Fourier decomposition of the initial temperature, with each mode decaying exponentially.

## Solving the Wave Equation

### Fourier Method

$$u_{tt} = c^2 u_{xx}, \quad u(0,t) = u(L,t) = 0, \quad u(x,0) = f(x), \quad u_t(x,0) = g(x)$$

Separation of variables gives the same eigenfunctions $\sin(n\pi x / L)$, but the time equation is now:

$$T_n'' + c^2 \lambda_n T_n = 0 \quad \Longrightarrow \quad T_n(t) = A_n \cos(\omega_n t) + B_n \sin(\omega_n t)$$

where $\omega_n = cn\pi/L$ are the **natural frequencies** of the string. Unlike the heat equation, these modes oscillate rather than decay -- the wave equation conserves energy.

$$u(x,t) = \sum_{n=1}^{\infty} \left[A_n \cos(\omega_n t) + B_n \sin(\omega_n t)\right] \sin\left(\frac{n\pi x}{L}\right)$$

The coefficients are determined by the initial conditions:

$$A_n = \frac{2}{L}\int_0^L f(x)\sin\left(\frac{n\pi x}{L}\right) dx, \quad B_n = \frac{2}{\omega_n L}\int_0^L g(x)\sin\left(\frac{n\pi x}{L}\right) dx$$

### D'Alembert's Solution

For the wave equation on an infinite domain ($-\infty < x < \infty$):

$$u(x,t) = \frac{1}{2}[f(x - ct) + f(x + ct)] + \frac{1}{2c}\int_{x-ct}^{x+ct} g(\xi) \, d\xi$$

This elegant formula reveals the physics: the initial displacement splits into two waves traveling in opposite directions at speed $c$. The term involving $g$ accounts for the initial velocity.

## Solving Laplace's Equation on a Rectangle

$$u_{xx} + u_{yy} = 0, \quad 0 < x < a, \quad 0 < y < b$$

With boundary conditions: $u(x, 0) = 0$, $u(x, b) = f(x)$, $u(0, y) = 0$, $u(a, y) = 0$.

Separation $u = X(x)Y(y)$ gives:

$$\frac{X''}{X} = -\frac{Y''}{Y} = -\lambda$$

The $x$-problem with homogeneous BCs gives $X_n = \sin(n\pi x / a)$, $\lambda_n = (n\pi/a)^2$.

The $y$-problem: $Y'' - \lambda_n Y = 0$ with $Y(0) = 0$ gives $Y_n(y) = \sinh(n\pi y / a)$.

$$u(x, y) = \sum_{n=1}^{\infty} C_n \sin\left(\frac{n\pi x}{a}\right) \sinh\left(\frac{n\pi y}{a}\right)$$

Matching $u(x, b) = f(x)$:

$$C_n = \frac{2}{a \sinh(n\pi b/a)} \int_0^a f(x) \sin\left(\frac{n\pi x}{a}\right) dx$$

## Python Implementation

```python
"""
Fourier Series and PDE Solutions.

This script demonstrates:
1. Computing and visualizing Fourier coefficients
2. Animated heat equation solution via Fourier series
3. Wave equation with D'Alembert and Fourier solutions
4. Laplace equation on a rectangle
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# ── 1. Fourier Series Approximation ─────────────────────────
def fourier_sine_coefficients(f, L, N):
    """
    Compute the first N Fourier sine coefficients of f on [0, L].

    b_n = (2/L) * integral_0^L f(x) sin(n*pi*x/L) dx

    We use numerical integration (trapezoidal rule) for generality.
    """
    x = np.linspace(0, L, 1000)
    dx = x[1] - x[0]
    coeffs = []
    for n in range(1, N + 1):
        integrand = f(x) * np.sin(n * np.pi * x / L)
        # Trapezoidal integration — simple but effective for smooth integrands
        b_n = (2.0 / L) * np.trapz(integrand, x)
        coeffs.append(b_n)
    return coeffs


def fourier_sine_series(coeffs, L, x, t=None, alpha=None, mode='static'):
    """
    Evaluate the Fourier sine series at points x.

    Modes:
        'static': sum of b_n * sin(n*pi*x/L)
        'heat':   sum of b_n * sin(n*pi*x/L) * exp(-alpha*(n*pi/L)^2 * t)
        'wave':   sum of b_n * sin(n*pi*x/L) * cos(c*n*pi*t/L)
    """
    result = np.zeros_like(x)
    for n, b_n in enumerate(coeffs, start=1):
        spatial = np.sin(n * np.pi * x / L)
        if mode == 'static':
            result += b_n * spatial
        elif mode == 'heat':
            decay = np.exp(-alpha * (n * np.pi / L)**2 * t)
            result += b_n * spatial * decay
        elif mode == 'wave':
            # alpha parameter reused as wave speed c here
            oscillation = np.cos(alpha * n * np.pi * t / L)
            result += b_n * spatial * oscillation
    return result


# Example: Fourier series of a triangle wave
L = 1.0
f_triangle = lambda x: np.where(x <= L/2, 2*x/L, 2*(L-x)/L)

x_plot = np.linspace(0, L, 500)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot convergence of Fourier series
for N in [1, 3, 5, 15, 50]:
    coeffs = fourier_sine_coefficients(f_triangle, L, N)
    y_approx = fourier_sine_series(coeffs, L, x_plot)
    axes[0, 0].plot(x_plot, y_approx, label=f'N={N}', alpha=0.8)

axes[0, 0].plot(x_plot, f_triangle(x_plot), 'k--', linewidth=2, label='Exact')
axes[0, 0].set_title('Fourier Sine Series Convergence (Triangle Wave)')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('f(x)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Coefficient magnitudes — shows how quickly coefficients decay
N_show = 20
coeffs_20 = fourier_sine_coefficients(f_triangle, L, N_show)
axes[0, 1].stem(range(1, N_show + 1), np.abs(coeffs_20), basefmt='k-')
axes[0, 1].set_title('|b_n| vs n (Triangle Wave)')
axes[0, 1].set_xlabel('n')
axes[0, 1].set_ylabel('|b_n|')
axes[0, 1].grid(True, alpha=0.3)

# ── 2. Heat Equation via Fourier Series ──────────────────────
alpha_heat = 0.01  # thermal diffusivity
N_modes = 50
coeffs_heat = fourier_sine_coefficients(f_triangle, L, N_modes)

times = [0, 0.5, 2.0, 5.0, 15.0]
colors_heat = plt.cm.hot(np.linspace(0.1, 0.9, len(times)))

for i, t in enumerate(times):
    u = fourier_sine_series(coeffs_heat, L, x_plot, t=t,
                            alpha=alpha_heat, mode='heat')
    axes[1, 0].plot(x_plot, u, color=colors_heat[i], linewidth=2,
                    label=f't = {t}')

axes[1, 0].set_title('Heat Equation: Fourier Series Solution')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('u(x, t)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# ── 3. Wave Equation via Fourier Series ──────────────────────
c_wave = 1.0  # wave speed
N_modes_wave = 30
# Initial condition: plucked string (triangle)
coeffs_wave = fourier_sine_coefficients(f_triangle, L, N_modes_wave)

times_wave = [0, 0.1, 0.25, 0.5, 0.75, 1.0]
colors_wave = plt.cm.viridis(np.linspace(0, 1, len(times_wave)))

for i, t in enumerate(times_wave):
    u = fourier_sine_series(coeffs_wave, L, x_plot, t=t,
                            alpha=c_wave, mode='wave')
    axes[1, 1].plot(x_plot, u, color=colors_wave[i], linewidth=1.5,
                    label=f't = {t:.2f}')

axes[1, 1].set_title('Wave Equation: Plucked String')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('u(x, t)')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fourier_series_pde.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to fourier_series_pde.png")

# ── 4. Laplace Equation on a Rectangle ──────────────────────
print("\n=== Laplace Equation on [0,a] x [0,b] ===")
a, b_rect = 2.0, 1.0  # rectangle dimensions
N_laplace = 30

# Boundary condition at y = b: f(x) = sin(pi*x/a)
# This is already the n=1 Fourier mode, so only C_1 is nonzero
# C_1 = 1 / sinh(pi * b / a)
x_rect = np.linspace(0, a, 100)
y_rect = np.linspace(0, b_rect, 50)
X_rect, Y_rect = np.meshgrid(x_rect, y_rect)

# For general f(x), compute Fourier coefficients
f_boundary = lambda x: np.sin(np.pi * x / a) + 0.5 * np.sin(3 * np.pi * x / a)

U = np.zeros_like(X_rect)
for n in range(1, N_laplace + 1):
    # Numerical integration for C_n
    integrand = f_boundary(x_rect) * np.sin(n * np.pi * x_rect / a)
    coeff = (2.0 / a) * np.trapz(integrand, x_rect)
    C_n = coeff / np.sinh(n * np.pi * b_rect / a)

    # Add this mode's contribution
    U += C_n * np.sin(n * np.pi * X_rect / a) * np.sinh(n * np.pi * Y_rect / a)

fig2, ax2 = plt.subplots(figsize=(10, 5))
contour = ax2.contourf(X_rect, Y_rect, U, levels=30, cmap='RdBu_r')
plt.colorbar(contour, ax=ax2, label='u(x, y)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Laplace Equation: Steady-State Temperature on Rectangle')
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('laplace_rectangle.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to laplace_rectangle.png")
```

## Summary

| Method | Equation | Solution Form |
|--------|----------|--------------|
| Fourier + heat | $u_t = \alpha u_{xx}$ | $\sum B_n \sin(n\pi x/L) e^{-\alpha(n\pi/L)^2 t}$ (exponential decay) |
| Fourier + wave | $u_{tt} = c^2 u_{xx}$ | $\sum [A_n \cos + B_n \sin](\omega_n t) \sin(n\pi x/L)$ (oscillation) |
| D'Alembert | $u_{tt} = c^2 u_{xx}$ | $\frac{1}{2}[f(x-ct) + f(x+ct)]$ (traveling waves) |
| Fourier + Laplace | $u_{xx} + u_{yy} = 0$ | $\sum C_n \sin(n\pi x/a) \sinh(n\pi y/a)$ |

The choice of sine vs cosine series is dictated by the boundary conditions: Dirichlet (zero value) gives sine, Neumann (zero derivative) gives cosine. The physical distinction between the heat and wave equations appears in the time factor: exponential decay vs oscillation.

For a more thorough treatment of Fourier analysis including the Fourier transform on infinite domains and Parseval's theorem, see [Mathematical Methods - Fourier Series](../Mathematical_Methods/07_Fourier_Series.md) and [Fourier Transform](../Mathematical_Methods/08_Fourier_Transform.md).

## Practice Problems

1. **Fourier coefficients**: Compute the Fourier sine series of $f(x) = x(1-x)$ on $[0, 1]$. Show that $b_n = 0$ for even $n$ and find a closed form for odd $n$. How does the decay rate of $b_n$ compare with the square wave example?

2. **Heat equation**: Solve $u_t = u_{xx}$ on $[0, \pi]$ with $u(0,t) = u(\pi,t) = 0$ and $u(x,0) = 100$. Find the Fourier series solution and determine how long it takes for the maximum temperature to drop to $50°$. (Keep only the first term for an estimate.)

3. **Wave equation**: A guitar string of length $L = 0.65$ m is plucked at $x = L/4$ with initial displacement $f(x) = \begin{cases} 4x/L & 0 \leq x \leq L/4 \\ 4(L-x)/(3L) & L/4 < x \leq L \end{cases}$ and released from rest. If $c = 300$ m/s, find the first four nonzero terms of the Fourier series solution. What are the corresponding frequencies in Hz?

4. **Laplace equation**: Solve $u_{xx} + u_{yy} = 0$ on $[0, 1] \times [0, 1]$ with $u(x,0) = 0$, $u(x,1) = \sin(\pi x)$, $u(0,y) = u(1,y) = 0$. Verify that the solution at the center point $(1/2, 1/2)$ satisfies the mean value property: compare $u(1/2, 1/2)$ with the average of $u$ over a small circle centered there.

5. **Gibbs phenomenon**: Write a Python script that plots the partial Fourier sine series of $f(x) = 1$ on $[0, \pi]$ for $N = 10, 50, 200$. Measure the overshoot near $x = 0$ and $x = \pi$. Verify that the overshoot converges to approximately $9\%$ regardless of $N$.

---

*Previous: [Introduction to Partial Differential Equations](./17_Introduction_to_PDE.md) | Next: [Numerical Methods for Differential Equations](./19_Numerical_Methods_for_DE.md)*
