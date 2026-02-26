# Spectral Methods

## Learning Objectives
- Understand the mathematical foundation of spectral methods
- Master Fourier spectral methods and FFT-based differentiation
- Learn Chebyshev collocation and pseudospectral techniques
- Apply dealiasing strategies (3/2 rule) to prevent aliasing errors
- Implement spectral solvers for PDEs (Burgers equation, KdV equation)

## Table of Contents
1. [Introduction to Spectral Methods](#1-introduction-to-spectral-methods)
2. [Fourier Spectral Method](#2-fourier-spectral-method)
3. [Discrete Fourier Transform and FFT](#3-discrete-fourier-transform-and-fft)
4. [Spectral Differentiation](#4-spectral-differentiation)
5. [Chebyshev Polynomials](#5-chebyshev-polynomials)
6. [Pseudospectral Methods](#6-pseudospectral-methods)
7. [Dealiasing](#7-dealiasing)
8. [Applications: Solving PDEs Spectrally](#8-applications-solving-pdes-spectrally)
9. [Practice Problems](#9-practice-problems)

---

## 1. Introduction to Spectral Methods

### 1.1 What are Spectral Methods?

**Why Spectral Methods?** Finite difference methods converge as $O(h^p)$ -- doubling the resolution only reduces the error by a fixed factor. Spectral methods converge as $O(e^{-cN})$ for smooth problems -- doubling the resolution can reduce the error by orders of magnitude. This makes spectral methods vastly more efficient when the solution is smooth, which is why they are the method of choice for turbulence simulation (DNS), weather prediction, and quantum mechanics.

Spectral methods approximate solutions to differential equations using global basis functions (e.g., Fourier series, Chebyshev polynomials). Unlike finite difference or finite element methods that use local approximations, spectral methods achieve **exponential convergence** for smooth problems.

```
┌─────────────────────────────────────────────────────────────┐
│             Comparison: Finite Difference vs Spectral       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Finite Difference:          Spectral Methods:             │
│  ┌─┬─┬─┬─┬─┬─┬─┐             ┌──────────────┐              │
│  │ │ │ │ │ │ │ │             │ ∑ aₖ φₖ(x)   │              │
│  └─┴─┴─┴─┴─┴─┴─┘             └──────────────┘              │
│  Local stencils             Global basis functions         │
│  O(h^p) convergence         O(e^(-cn)) convergence         │
│  (p = order)                (exponential!)                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Advantages:**
- Exponential accuracy for smooth solutions
- Natural handling of derivatives (multiplication in spectral space)
- Efficient with FFT (O(N log N) operations)

**Limitations:**
- Requires smooth, periodic (Fourier) or well-behaved (Chebyshev) functions
- Complex geometries are challenging
- Non-periodic boundaries require special treatment

### 1.2 Types of Spectral Methods

```python
"""
Three main types of spectral methods:

1. Galerkin Method:
   - Weighted residual approach
   - Projects PDE onto basis functions
   - Example: u(x) = Σ aₙ φₙ(x), minimize ∫R(u)φₘ dx

2. Collocation Method (Pseudospectral):
   - Enforces PDE exactly at collocation points
   - Faster than Galerkin, slightly less accurate
   - Example: R(u(xⱼ)) = 0 at grid points xⱼ

3. Tau Method:
   - Similar to Galerkin, but allows residual at boundaries
   - Useful for non-periodic problems
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

np.random.seed(42)
```

---

## 2. Fourier Spectral Method

### 2.1 Fourier Series Representation

For periodic functions on [0, 2π], we expand:

```
u(x) = Σ ûₖ e^(ikx)
       k=-∞ to ∞

where ûₖ = (1/2π) ∫₀^(2π) u(x) e^(-ikx) dx
```

In practice, we truncate to N modes:

```python
def fourier_series_example():
    """
    Demonstrate Fourier series approximation of a smooth periodic function.
    """
    N = 64
    x = np.linspace(0, 2*np.pi, N, endpoint=False)

    # Original function: u(x) = sin(x) + 0.5*sin(2x) + 0.1*cos(5x)
    u = np.sin(x) + 0.5*np.sin(2*x) + 0.1*np.cos(5*x)

    # Compute Fourier coefficients using FFT
    u_hat = fft(u)

    # Reconstruct function from spectral coefficients
    u_reconstructed = np.real(ifft(u_hat))

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(x, u, 'b-', label='Original', linewidth=2)
    plt.plot(x, u_reconstructed, 'r--', label='Reconstructed', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Fourier Series Representation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fourier_series.png', dpi=150)
    plt.close()

    # Print spectral coefficients (magnitude)
    k = fftfreq(N, 1/N)
    plt.figure(figsize=(10, 4))
    plt.stem(k, np.abs(u_hat), basefmt=' ')
    plt.xlabel('Wavenumber k')
    plt.ylabel('|û_k|')
    plt.title('Fourier Spectrum')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fourier_spectrum.png', dpi=150)
    plt.close()

    print(f"L2 error: {np.linalg.norm(u - u_reconstructed):.2e}")

fourier_series_example()
```

**Output:**
```
L2 error: 1.34e-14
```

The reconstruction is essentially exact (up to machine precision).

---

## 3. Discrete Fourier Transform and FFT

### 3.1 DFT Definition

For N discrete points, the DFT is:

```
ûₖ = Σ_{j=0}^{N-1} uⱼ e^(-2πijk/N),  k = 0, 1, ..., N-1

Inverse: uⱼ = (1/N) Σ_{k=0}^{N-1} ûₖ e^(2πijk/N)
```

### 3.2 FFT Algorithm

The Fast Fourier Transform reduces complexity from O(N²) to O(N log N):

```
┌────────────────────────────────────────────────────────────┐
│              Cooley-Tukey FFT Algorithm                    │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  DFT[u₀, u₁, ..., u_{N-1}]                                │
│       = DFT[u₀, u₂, u₄, ...] (even indices)               │
│         + W * DFT[u₁, u₃, u₅, ...] (odd indices)          │
│                                                            │
│  where W = e^(-2πi/N) (twiddle factors)                   │
│                                                            │
│  Recursively split until base case (N=1)                  │
│  Complexity: O(N log N)                                    │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

```python
from scipy.fft import fft, ifft, fftfreq
import time

def fft_performance_test():
    """
    Compare naive DFT vs FFT performance.
    """
    def naive_dft(u):
        N = len(u)
        u_hat = np.zeros(N, dtype=complex)
        for k in range(N):
            for j in range(N):
                u_hat[k] += u[j] * np.exp(-2j * np.pi * k * j / N)
        return u_hat

    sizes = [32, 64, 128, 256, 512]
    naive_times = []
    fft_times = []

    for N in sizes:
        u = np.random.randn(N)

        # Naive DFT
        start = time.time()
        u_hat_naive = naive_dft(u)
        naive_times.append(time.time() - start)

        # FFT
        start = time.time()
        u_hat_fft = fft(u)
        fft_times.append(time.time() - start)

        # Verify equivalence
        error = np.linalg.norm(u_hat_naive - u_hat_fft)
        print(f"N={N:4d}: Naive={naive_times[-1]:.4f}s, FFT={fft_times[-1]:.6f}s, Error={error:.2e}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.loglog(sizes, naive_times, 'o-', label='Naive DFT O(N²)', linewidth=2)
    plt.loglog(sizes, fft_times, 's-', label='FFT O(N log N)', linewidth=2)
    plt.xlabel('N')
    plt.ylabel('Time (s)')
    plt.title('DFT vs FFT Performance')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('fft_performance.png', dpi=150)
    plt.close()

fft_performance_test()
```

---

## 4. Spectral Differentiation

### 4.1 Derivatives in Spectral Space

The key advantage of spectral methods: **differentiation becomes multiplication in spectral space**.

For Fourier basis:
```
u(x) = Σ ûₖ e^(ikx)
du/dx = Σ (ik) ûₖ e^(ikx)
```

In code:
```python
def spectral_derivative(u, L=2*np.pi):
    """
    Compute derivative using spectral differentiation.

    Parameters:
    -----------
    u : array, function values on uniform grid
    L : float, domain length (default 2π)

    Returns:
    --------
    du_dx : array, derivative
    """
    N = len(u)
    u_hat = fft(u)
    k = fftfreq(N, L/N) * 2 * np.pi  # Wavenumbers

    # Derivative in spectral space: multiply by ik
    du_hat = 1j * k * u_hat

    # Transform back to physical space
    du_dx = np.real(ifft(du_hat))

    return du_dx

# Example: differentiate u(x) = sin(x) + 0.5*sin(2x)
N = 128
x = np.linspace(0, 2*np.pi, N, endpoint=False)
u = np.sin(x) + 0.5*np.sin(2*x)
du_dx_exact = np.cos(x) + np.cos(2*x)
du_dx_spectral = spectral_derivative(u)

error = np.linalg.norm(du_dx_exact - du_dx_spectral, np.inf)
print(f"Max error in derivative: {error:.2e}")
```

**Output:**
```
Max error in derivative: 1.78e-14
```

### 4.2 Higher-Order Derivatives

```python
def spectral_derivative_n(u, n, L=2*np.pi):
    """
    Compute n-th derivative using spectral differentiation.
    """
    N = len(u)
    u_hat = fft(u)
    k = fftfreq(N, L/N) * 2 * np.pi

    # n-th derivative: multiply by (ik)^n
    du_hat = (1j * k)**n * u_hat

    du_dx = np.real(ifft(du_hat))
    return du_dx

# Test second derivative
u = np.sin(x)
d2u_dx2_exact = -np.sin(x)
d2u_dx2_spectral = spectral_derivative_n(u, n=2)

error = np.linalg.norm(d2u_dx2_exact - d2u_dx2_spectral, np.inf)
print(f"Max error in 2nd derivative: {error:.2e}")
```

---

## 5. Chebyshev Polynomials

### 5.1 Chebyshev Polynomials of the First Kind

For non-periodic problems on [-1, 1], Chebyshev polynomials are optimal:

```
T₀(x) = 1
T₁(x) = x
T₂(x) = 2x² - 1
T₃(x) = 4x³ - 3x
...
Tₙ(x) = cos(n arccos(x))
```

**Recurrence relation:**
```
T_{n+1}(x) = 2x Tₙ(x) - T_{n-1}(x)
```

```python
def chebyshev_polynomials(n, x):
    """
    Evaluate Chebyshev polynomials T₀, T₁, ..., Tₙ at points x.

    Returns:
    --------
    T : array of shape (n+1, len(x))
    """
    T = np.zeros((n+1, len(x)))
    T[0] = 1
    if n >= 1:
        T[1] = x
    for k in range(2, n+1):
        T[k] = 2*x*T[k-1] - T[k-2]
    return T

# Plot first 5 Chebyshev polynomials
x = np.linspace(-1, 1, 200)
T = chebyshev_polynomials(4, x)

plt.figure(figsize=(10, 5))
for k in range(5):
    plt.plot(x, T[k], label=f'T_{k}(x)', linewidth=2)
plt.xlabel('x')
plt.ylabel('Tₖ(x)')
plt.title('Chebyshev Polynomials')
plt.legend()
plt.grid(True)
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.tight_layout()
plt.savefig('chebyshev_polynomials.png', dpi=150)
plt.close()
```

### 5.2 Chebyshev-Gauss-Lobatto Points

Optimal collocation points (include endpoints):

```
xⱼ = cos(πj/N),  j = 0, 1, ..., N
```

```python
def chebyshev_points(N):
    """
    Compute Chebyshev-Gauss-Lobatto points.
    """
    j = np.arange(N+1)
    x = np.cos(np.pi * j / N)
    return x

N = 16
x_cheb = chebyshev_points(N)

plt.figure(figsize=(10, 3))
plt.plot(x_cheb, np.zeros_like(x_cheb), 'ro', markersize=8)
plt.xlim(-1.1, 1.1)
plt.ylim(-0.5, 0.5)
plt.xlabel('x')
plt.title(f'Chebyshev-Gauss-Lobatto Points (N={N})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('chebyshev_points.png', dpi=150)
plt.close()

print("Chebyshev points (clustered near boundaries):")
print(x_cheb)
```

### 5.3 Chebyshev Differentiation Matrix

```python
def chebyshev_diff_matrix(N):
    """
    Compute Chebyshev differentiation matrix D.

    u'(xⱼ) ≈ Σ Dⱼₖ u(xₖ)
    """
    x = chebyshev_points(N)
    D = np.zeros((N+1, N+1))

    c = np.ones(N+1)
    c[0] = 2
    c[N] = 2

    for i in range(N+1):
        for j in range(N+1):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1)**(i+j)) / (x[i] - x[j])

    # Diagonal elements
    for i in range(N+1):
        D[i, i] = -np.sum(D[i, :])

    return D, x

# Test: differentiate u(x) = exp(x)
N = 16
D, x = chebyshev_diff_matrix(N)
u = np.exp(x)
du_dx_exact = np.exp(x)
du_dx_cheb = D @ u

error = np.linalg.norm(du_dx_exact - du_dx_cheb, np.inf)
print(f"Chebyshev differentiation error: {error:.2e}")
```

---

## 6. Pseudospectral Methods

### 6.1 Galerkin vs Pseudospectral

```
┌────────────────────────────────────────────────────────────┐
│           Galerkin vs Pseudospectral (Collocation)         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Galerkin:                                                 │
│    ∫ R(u) φₘ(x) dx = 0  for all basis functions φₘ        │
│    → Requires integration (inner products)                │
│    → Higher accuracy, more expensive                      │
│                                                            │
│  Pseudospectral (Collocation):                            │
│    R(u(xⱼ)) = 0  at collocation points xⱼ                 │
│    → No integration, evaluate at grid points             │
│    → Slightly lower accuracy, much faster                │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

```python
def pseudospectral_example():
    """
    Solve u''(x) = -π² sin(πx), u(0)=u(1)=0 using pseudospectral method.

    Exact solution: u(x) = sin(πx)
    """
    N = 16

    # Map Chebyshev points from [-1,1] to [0,1]
    x_cheb = (chebyshev_points(N) + 1) / 2

    # Build differentiation matrix (need to scale for [0,1] domain)
    D, _ = chebyshev_diff_matrix(N)
    D = D * 2  # Scale for [0,1] domain
    D2 = D @ D  # Second derivative

    # Boundary conditions: u(0) = u(N) = 0
    # Interior points: indices 1 to N-1
    D2_interior = D2[1:N, 1:N]

    # Right-hand side: f(x) = -π² sin(πx)
    f = -np.pi**2 * np.sin(np.pi * x_cheb[1:N])

    # Solve linear system
    u_interior = np.linalg.solve(D2_interior, f)

    # Add boundary values
    u = np.zeros(N+1)
    u[1:N] = u_interior

    # Compare with exact solution
    u_exact = np.sin(np.pi * x_cheb)

    plt.figure(figsize=(10, 5))
    plt.plot(x_cheb, u_exact, 'b-', label='Exact', linewidth=2)
    plt.plot(x_cheb, u, 'ro', label='Pseudospectral', markersize=8)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title("Pseudospectral Solution: u'' = -π² sin(πx)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pseudospectral_example.png', dpi=150)
    plt.close()

    error = np.linalg.norm(u_exact - u, np.inf)
    print(f"Pseudospectral error: {error:.2e}")

pseudospectral_example()
```

---

## 7. Dealiasing

### 7.1 Aliasing Problem

Nonlinear terms in PDEs (e.g., u ∂u/∂x in Burgers equation) produce high-frequency modes that can **alias** onto low frequencies:

```
┌────────────────────────────────────────────────────────────┐
│                    Aliasing Example                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  If u has max wavenumber k_max = N/2,                     │
│  then u² has modes up to 2*k_max = N                      │
│                                                            │
│  But DFT only resolves modes up to N/2!                   │
│  → High frequencies wrap around (alias)                   │
│                                                            │
│  Solution: 3/2 rule (zero-padding)                        │
│    1. Pad u to 3N/2 modes                                 │
│    2. Compute nonlinearity in extended space             │
│    3. Truncate back to N modes                            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 7.2 3/2 Rule Implementation

```python
def dealias_product_3_2_rule(u, v):
    """
    Compute dealiased product w = u*v using 3/2 rule.

    Parameters:
    -----------
    u, v : arrays of length N

    Returns:
    --------
    w : dealiased product (length N)
    """
    N = len(u)
    M = 3 * N // 2  # Extended grid

    # Fourier transform
    u_hat = fft(u)
    v_hat = fft(v)

    # Zero-pad to 3N/2
    u_hat_padded = np.zeros(M, dtype=complex)
    v_hat_padded = np.zeros(M, dtype=complex)

    # Low frequencies
    u_hat_padded[:N//2] = u_hat[:N//2]
    u_hat_padded[-N//2:] = u_hat[-N//2:]

    v_hat_padded[:N//2] = v_hat[:N//2]
    v_hat_padded[-N//2:] = v_hat[-N//2:]

    # Inverse transform to extended grid
    u_extended = ifft(u_hat_padded)
    v_extended = ifft(v_hat_padded)

    # Multiply in physical space
    w_extended = u_extended * v_extended

    # Transform back
    w_hat_extended = fft(w_extended)

    # Truncate to original resolution
    w_hat = np.zeros(N, dtype=complex)
    w_hat[:N//2] = w_hat_extended[:N//2]
    w_hat[-N//2:] = w_hat_extended[-N//2:]

    # Scale (account for extended grid)
    w_hat *= M / N

    w = ifft(w_hat)
    return np.real(w)

# Test: compare aliased vs dealiased product
N = 64
x = np.linspace(0, 2*np.pi, N, endpoint=False)
u = np.sin(3*x)
v = np.cos(4*x)

# Aliased product (direct multiplication)
w_aliased = u * v

# Dealiased product (3/2 rule)
w_dealiased = dealias_product_3_2_rule(u, v)

# Exact product
w_exact = u * v  # For this example, exact = 0.5*(sin(7x) - sin(x))

print(f"Aliased error:   {np.linalg.norm(w_exact - w_aliased):.2e}")
print(f"Dealiased error: {np.linalg.norm(w_exact - w_dealiased):.2e}")
```

---

## 8. Applications: Solving PDEs Spectrally

### 8.1 Burgers Equation

The viscous Burgers equation is a fundamental nonlinear PDE:

```
∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²

Initial condition: u(x,0) = sin(x)
Periodic boundary: u(0,t) = u(2π,t)
```

```python
def burgers_spectral(nu=0.01, T=2.0, N=128, dt=0.001):
    """
    Solve Burgers equation using Fourier spectral method with dealiasing.

    ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
    """
    # Grid
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    k = fftfreq(N, L/N) * 2 * np.pi

    # Initial condition
    u = np.sin(x)

    # Time integration (RK4)
    nt = int(T / dt)
    time = 0.0

    # Store solution at select times
    t_save = [0, 0.5, 1.0, 1.5, 2.0]
    u_save = []

    def rhs(u):
        """Right-hand side of Burgers equation in spectral space."""
        u_hat = fft(u)

        # Linear term: ν ∂²u/∂x²
        linear = -nu * k**2 * u_hat

        # Nonlinear term: -u ∂u/∂x (compute in physical space with dealiasing)
        ux = np.real(ifft(1j * k * u_hat))
        nonlinear_phys = -u * ux
        nonlinear_phys = dealias_product_3_2_rule(u, ux)
        nonlinear = fft(nonlinear_phys)

        du_dt_hat = linear + nonlinear
        return np.real(ifft(du_dt_hat))

    for n in range(nt):
        # RK4 time stepping
        k1 = rhs(u)
        k2 = rhs(u + 0.5*dt*k1)
        k3 = rhs(u + 0.5*dt*k2)
        k4 = rhs(u + dt*k3)

        u = u + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        time += dt

        # Save solution
        if any(abs(time - t) < dt/2 for t in t_save):
            u_save.append(u.copy())

    # Plot
    plt.figure(figsize=(10, 6))
    for i, t in enumerate(t_save[:len(u_save)]):
        plt.plot(x, u_save[i], label=f't = {t:.1f}', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'Burgers Equation (ν={nu})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('burgers_spectral.png', dpi=150)
    plt.close()

    return x, u_save

x, u_save = burgers_spectral(nu=0.01, T=2.0)
print("Burgers equation solved successfully.")
```

### 8.2 Korteweg-de Vries (KdV) Equation

The KdV equation models soliton waves:

```
∂u/∂t + u ∂u/∂x + ∂³u/∂x³ = 0

Initial condition: u(x,0) = -6κ² sech²(κx)  (single soliton)
```

```python
def kdv_spectral(kappa=0.5, T=10.0, N=256, dt=0.01):
    """
    Solve KdV equation using Fourier spectral method.

    ∂u/∂t + u ∂u/∂x + ∂³u/∂x³ = 0
    """
    # Grid (larger domain for soliton propagation)
    L = 40 * np.pi
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    k = fftfreq(N, L/N) * 2 * np.pi

    # Initial condition: single soliton
    u = -6 * kappa**2 / np.cosh(kappa * x)**2

    # Time integration
    nt = int(T / dt)

    def rhs(u):
        """Right-hand side of KdV equation."""
        u_hat = fft(u)

        # Dispersion: ∂³u/∂x³
        dispersion = -(1j * k)**3 * u_hat

        # Nonlinearity: -u ∂u/∂x
        ux = np.real(ifft(1j * k * u_hat))
        nonlinear_phys = -u * ux
        nonlinear = fft(nonlinear_phys)

        du_dt_hat = dispersion + nonlinear
        return np.real(ifft(du_dt_hat))

    # RK4 time stepping
    u_save = [u.copy()]
    for n in range(nt):
        k1 = rhs(u)
        k2 = rhs(u + 0.5*dt*k1)
        k3 = rhs(u + 0.5*dt*k2)
        k4 = rhs(u + dt*k3)

        u = u + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        if n % 100 == 0:
            u_save.append(u.copy())

    # Plot soliton propagation
    plt.figure(figsize=(12, 6))
    for i, u_snap in enumerate(u_save[::5]):
        plt.plot(x, u_snap, label=f't = {i*5*dt:.1f}', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('KdV Equation: Soliton Propagation')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('kdv_spectral.png', dpi=150)
    plt.close()

    return x, u_save

x, u_save = kdv_spectral(kappa=0.5, T=10.0)
print("KdV equation solved successfully.")
```

---

## Exercises

### Exercise 1: Exponential Convergence
Demonstrate that spectral methods achieve exponential convergence for smooth periodic functions. Approximate u(x) = e^(sin(x)) on [0, 2π] using N Fourier modes. Compute the L∞ error for N = 4, 8, 16, 32, 64 and verify exponential decay on a log-scale plot.

<details><summary>Show Answer</summary>

```python
def test_exponential_convergence():
    def u_exact(x):
        return np.exp(np.sin(x))

    N_values = [4, 8, 16, 32, 64]
    errors = []

    x_fine = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    u_fine = u_exact(x_fine)

    for N in N_values:
        x = np.linspace(0, 2*np.pi, N, endpoint=False)
        u = u_exact(x)

        # Interpolate using spectral method
        u_hat = fft(u)

        # Evaluate on fine grid
        k = fftfreq(N, 2*np.pi/N) * 2 * np.pi
        u_interp = np.zeros(len(x_fine))
        for j, xj in enumerate(x_fine):
            u_interp[j] = np.real(np.sum(u_hat * np.exp(1j * k * xj)) / N)

        error = np.linalg.norm(u_fine - u_interp, np.inf)
        errors.append(error)
        print(f"N={N:3d}: Error = {error:.2e}")

    plt.figure(figsize=(8, 5))
    plt.semilogy(N_values, errors, 'o-', linewidth=2, markersize=8)
    plt.xlabel('N (number of modes)')
    plt.ylabel('L∞ error')
    plt.title('Exponential Convergence of Spectral Method')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('exponential_convergence.png', dpi=150)
    plt.close()

test_exponential_convergence()
```

The error drops dramatically (from ~1e-1 at N=4 to ~1e-14 at N=64), confirming exponential (spectral) convergence. A polynomial method would show algebraic decay O(N^{-p}) instead.
</details>

### Exercise 2: Chebyshev Interpolation vs Runge Phenomenon
Interpolate the Runge function f(x) = 1/(1 + 25x²) on [-1, 1] using (a) N+1 uniformly spaced points and (b) Chebyshev-Gauss-Lobatto points for N = 10, 20. Compute the L∞ error on a fine grid and explain why Chebyshev nodes avoid the Runge phenomenon.

<details><summary>Show Answer</summary>

```python
def runge_vs_chebyshev(N=16):
    f = lambda x: 1.0 / (1 + 25 * x**2)
    x_fine = np.linspace(-1, 1, 500)
    f_fine = f(x_fine)

    # Uniform nodes
    x_uni = np.linspace(-1, 1, N+1)
    p_uni = np.polyfit(x_uni, f(x_uni), N)
    err_uni = np.max(np.abs(np.polyval(p_uni, x_fine) - f_fine))

    # Chebyshev-Gauss-Lobatto nodes
    j = np.arange(N+1)
    x_cheb = np.cos(np.pi * j / N)
    # Barycentric interpolation at fine points
    # (simple approach: use numpy polynomial fit on Chebyshev nodes)
    p_cheb = np.polyfit(x_cheb, f(x_cheb), N)
    err_cheb = np.max(np.abs(np.polyval(p_cheb, x_fine) - f_fine))

    print(f"N={N}: Uniform L∞ error = {err_uni:.4e}, Chebyshev L∞ error = {err_cheb:.4e}")

runge_vs_chebyshev(N=10)
runge_vs_chebyshev(N=20)
```

Uniform nodes produce large oscillations near the endpoints (Runge phenomenon), yielding errors that grow with N. Chebyshev nodes cluster near ±1, minimizing the Lebesgue constant and suppressing oscillations. The Chebyshev interpolant converges rapidly to the true function.
</details>

### Exercise 3: Spectral Heat Equation
Solve the 1D heat equation ∂u/∂t = ∂²u/∂x² on [0, 2π] with periodic boundary conditions and initial condition u(x,0) = sin(x). Use the Fourier spectral method with exact time integration (integrating factor). Compare the numerical solution at t = 1 with the exact solution u(x,t) = e^(-t) sin(x).

<details><summary>Show Answer</summary>

```python
def spectral_heat_equation(T=1.0, N=64, dt=0.01):
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    k = fftfreq(N, 1.0/N)  # integer wavenumbers

    # Initial condition
    u = np.sin(x)
    u_hat = fft(u)

    # Exact time integration: u_hat(k, t) = u_hat(k, 0) * exp(-k^2 * t)
    t_final = T
    u_hat_final = u_hat * np.exp(-k**2 * t_final)
    u_spectral = np.real(ifft(u_hat_final))

    u_exact = np.exp(-T) * np.sin(x)
    error = np.linalg.norm(u_spectral - u_exact, np.inf)
    print(f"Spectral heat equation error at t={T}: {error:.2e}")

    plt.figure(figsize=(8, 4))
    plt.plot(x, u_exact, 'b-', label='Exact', linewidth=2)
    plt.plot(x, u_spectral, 'r--', label='Spectral', linewidth=2)
    plt.xlabel('x'); plt.ylabel('u(x,t)')
    plt.title(f'Heat Equation at t={T}')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig('spectral_heat.png', dpi=150)
    plt.close()

spectral_heat_equation(T=1.0)
```

The integrating factor method solves the linear part exactly in spectral space. The error is at machine precision level (~1e-15), demonstrating the power of spectral methods for smooth problems with periodic boundaries.
</details>

### Exercise 4: Two-Soliton Collision in KdV
Modify the KdV spectral solver to initialize two solitons with different speeds. Use the initial condition:
```
u(x,0) = -6κ₁² sech²(κ₁(x+10)) - 6κ₂² sech²(κ₂(x-10))
```
with κ₁ = 0.5, κ₂ = 0.8. Simulate until T = 20 and verify that after the collision, both solitons emerge with their original shape and speed (an elastic collision).

<details><summary>Show Answer</summary>

```python
def kdv_two_soliton(T=20.0, N=512, dt=0.005):
    L = 60 * np.pi
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    k = fftfreq(N, L/N) * 2 * np.pi

    kappa1, kappa2 = 0.5, 0.8
    # Two-soliton initial condition (solitons at x=-10 and x=+10)
    u = (-6 * kappa1**2 / np.cosh(kappa1 * (x + 10))**2
         - 6 * kappa2**2 / np.cosh(kappa2 * (x - 10))**2)

    def rhs(u):
        u_hat = fft(u)
        dispersion = -(1j * k)**3 * u_hat
        ux = np.real(ifft(1j * k * u_hat))
        nonlinear = fft(-u * ux)
        return np.real(ifft(dispersion + nonlinear))

    u_initial = u.copy()
    nt = int(T / dt)
    for n in range(nt):
        k1 = rhs(u); k2 = rhs(u + 0.5*dt*k1)
        k3 = rhs(u + 0.5*dt*k2); k4 = rhs(u + dt*k3)
        u = u + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    plt.figure(figsize=(10, 4))
    plt.plot(x, u_initial, 'b-', label='t=0', linewidth=2)
    plt.plot(x, u, 'r-', label=f't={T}', linewidth=2)
    plt.xlabel('x'); plt.ylabel('u')
    plt.title('KdV Two-Soliton Collision')
    plt.legend(); plt.grid(True); plt.xlim(-L/4, L/4)
    plt.tight_layout()
    plt.savefig('kdv_two_soliton.png', dpi=150)
    plt.close()
    print(f"Mass conservation: initial={np.sum(u_initial):.6f}, final={np.sum(u):.6f}")

kdv_two_soliton()
```

After the collision, both solitons re-emerge with their original amplitudes and speeds, only shifted in position. This elastic collision (no energy loss) is a hallmark of the KdV soliton, arising from the integrable structure of the equation. Mass (and other conserved quantities) should be preserved to high accuracy.
</details>

### Exercise 5: Aliasing Error Demonstration
Show the effect of aliasing in a concrete example. For N = 32, compute u(x) = sin(Nx/2 · x) (a high-frequency mode) and then compute sin(3x) · u(x) without dealiasing. Compare the result in spectral space to the correctly dealiased version using the 3/2 rule.

<details><summary>Show Answer</summary>

```python
def aliasing_demonstration(N=32):
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    k = fftfreq(N, 1.0/N)

    # Two modes that when multiplied produce frequencies beyond N/2
    u = np.sin(int(N/4) * x)   # mode N/4
    v = np.sin(int(N/4) * x)   # same mode

    # Naive (aliased) product
    w_naive = u * v
    w_naive_hat = np.abs(fft(w_naive))

    # Dealiased product using 3/2 rule
    w_dealiased = dealias_product_3_2_rule(u, v)
    w_dealiased_hat = np.abs(fft(w_dealiased))

    # Exact: sin(k0*x)*sin(k0*x) = 0.5*(1 - cos(2*k0*x))
    # Mode 0 and mode 2*k0 should appear
    k0 = int(N/4)
    w_exact_hat = np.zeros(N)
    w_exact_hat[0] = N/2           # DC component
    w_exact_hat[min(2*k0, N-1)] = N/4  # frequency 2*k0

    print(f"Mode 0:    naive={w_naive_hat[0]:.2f}, dealiased={w_dealiased_hat[0]:.2f}, exact={N/2:.2f}")
    print(f"Mode {2*k0}: naive={w_naive_hat[2*k0 % N]:.2f}, dealiased={w_dealiased_hat[2*k0 % N]:.2f}, exact={N/4:.2f}")
    print("Aliasing causes spurious energy at wrong wavenumbers in the naive case.")

aliasing_demonstration(N=32)
```

When mode N/4 is squared, the product contains mode N/2 (the Nyquist frequency). Without dealiasing, this aliases back to mode 0, contaminating low-frequency content. The 3/2 rule zero-pads the spectrum before multiplication, correctly capturing the N/2 mode and returning the right answer.
</details>

---

## Navigation
- Previous: [20. Monte Carlo Simulation](20_Monte_Carlo_Simulation.md)
- Next: [22. Finite Element Method](22_Finite_Element_Method.md)
- [Back to Overview](00_Overview.md)
