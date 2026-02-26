# 08. Finite Difference Method Basics

## Learning Objectives
- Understand the basic principles of finite difference methods
- Learn grid/mesh generation techniques
- Derive forward/backward/central difference formulas and analyze accuracy
- Understand the concept of truncation error
- Learn CFL condition and von Neumann stability analysis

---

## 1. Introduction to Finite Difference Methods

### 1.1 Basic Idea

The Finite Difference Method (FDM) is a method of approximating derivatives with finite differences.

```
Definition of derivative:
f'(x) = lim[h→0] (f(x+h) - f(x)) / h

Finite difference approximation (h is a small finite value):
f'(x) ≈ (f(x+h) - f(x)) / h
```

```python
import numpy as np
import matplotlib.pyplot as plt

def finite_difference_demo():
    """Demonstrate basic finite difference concept"""

    # Test function: f(x) = sin(x)
    # Exact derivative: f'(x) = cos(x)
    f = lambda x: np.sin(x)
    f_exact = lambda x: np.cos(x)

    x = np.pi / 4  # Test point
    h_values = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

    print(f"Test function: f(x) = sin(x), x = π/4")
    print(f"Exact derivative: f'(π/4) = cos(π/4) = {f_exact(x):.10f}")
    print()
    print(f"{'h':<10} {'Forward':<15} {'Backward':<15} {'Central':<15}")
    print("-" * 55)

    for h in h_values:
        # Forward Difference
        forward = (f(x + h) - f(x)) / h

        # Backward Difference
        backward = (f(x) - f(x - h)) / h

        # Central Difference
        central = (f(x + h) - f(x - h)) / (2 * h)

        print(f"{h:<10.4f} {forward:<15.10f} {backward:<15.10f} {central:<15.10f}")

    print()
    print("Observation: Central difference is most accurate (2nd order accuracy)")

finite_difference_demo()
```

### 1.2 Why Finite Difference Methods?

| Advantages | Disadvantages |
|------|------|
| Simple implementation | Unsuitable for complex geometries |
| Intuitive understanding | Difficult to handle irregular grids |
| Computationally efficient | Limited local resolution control |
| High-order accuracy possible | Boundary condition handling can be complex |

---

## 2. Grid/Mesh Generation

### 2.1 1D Uniform Grid

```python
import numpy as np

def create_1d_grid(x_min, x_max, nx):
    """
    Create 1D uniform grid

    Parameters:
    -----------
    x_min : float - starting point
    x_max : float - ending point
    nx : int - number of grid points

    Returns:
    --------
    x : array - grid point coordinates
    dx : float - grid spacing
    """
    x = np.linspace(x_min, x_max, nx)
    dx = (x_max - x_min) / (nx - 1)
    return x, dx

# Example
x, dx = create_1d_grid(0, 1, 11)
print(f"Grid points: {x}")
print(f"Grid spacing dx = {dx}")
print(f"Number of interior points: {len(x) - 2}")  # Excluding boundaries
```

### 2.2 2D Uniform Grid

```python
def create_2d_grid(x_range, y_range, nx, ny):
    """
    Create 2D uniform grid

    Parameters:
    -----------
    x_range : tuple - (x_min, x_max)
    y_range : tuple - (y_min, y_max)
    nx, ny : int - number of grid points in each direction

    Returns:
    --------
    X, Y : 2D arrays - grid point coordinates
    dx, dy : float - grid spacing
    """
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)

    dx = (x_range[1] - x_range[0]) / (nx - 1)
    dy = (y_range[1] - y_range[0]) / (ny - 1)

    X, Y = np.meshgrid(x, y)

    return X, Y, dx, dy

# Example
X, Y, dx, dy = create_2d_grid((0, 1), (0, 1), 11, 11)
print(f"Grid size: {X.shape}")
print(f"dx = {dx}, dy = {dy}")
```

### 2.3 Spatiotemporal Grid

```python
def create_spacetime_grid(x_range, t_range, nx, nt):
    """
    Create spatiotemporal grid (1D space + time)

    Parameters:
    -----------
    x_range : tuple - (x_min, x_max)
    t_range : tuple - (t_start, t_end)
    nx : int - number of spatial grid points
    nt : int - number of time steps

    Returns:
    --------
    x, t : arrays - coordinates
    dx, dt : float - spacing
    """
    x = np.linspace(x_range[0], x_range[1], nx)
    t = np.linspace(t_range[0], t_range[1], nt + 1)

    dx = (x_range[1] - x_range[0]) / (nx - 1)
    dt = (t_range[1] - t_range[0]) / nt

    return x, t, dx, dt

# Example
x, t, dx, dt = create_spacetime_grid((0, 1), (0, 0.5), 51, 100)
print(f"Spatial grid points: {len(x)}, dx = {dx:.4f}")
print(f"Time steps: {len(t)-1}, dt = {dt:.6f}")
```

### 2.4 Grid Visualization

```python
import matplotlib.pyplot as plt

def visualize_grids():
    """Visualize grids"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1D grid
    ax1 = axes[0]
    x, dx = create_1d_grid(0, 1, 11)
    ax1.scatter(x, np.zeros_like(x), s=50, c='blue')
    for i, xi in enumerate(x):
        ax1.axvline(x=xi, color='gray', linestyle='--', alpha=0.3)
        ax1.annotate(f'$x_{{{i}}}$', (xi, 0.02), ha='center', fontsize=8)
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 0.2)
    ax1.set_title('1D Uniform Grid', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_yticks([])
    ax1.annotate(f'dx = {dx:.2f}', (0.5, 0.1), ha='center', fontsize=10)

    # 2D grid
    ax2 = axes[1]
    X, Y, dx, dy = create_2d_grid((0, 1), (0, 1), 6, 6)
    ax2.scatter(X, Y, s=30, c='blue')
    for i in range(X.shape[0]):
        ax2.axhline(y=Y[i, 0], color='gray', linestyle='--', alpha=0.3)
    for j in range(X.shape[1]):
        ax2.axvline(x=X[0, j], color='gray', linestyle='--', alpha=0.3)
    ax2.set_title('2D Uniform Grid', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')

    # Spatiotemporal grid
    ax3 = axes[2]
    x_st = np.linspace(0, 1, 6)
    t_st = np.linspace(0, 0.5, 4)
    X_st, T_st = np.meshgrid(x_st, t_st)
    ax3.scatter(X_st, T_st, s=30, c='blue')
    for i in range(len(t_st)):
        ax3.axhline(y=t_st[i], color='gray', linestyle='--', alpha=0.3)
    for j in range(len(x_st)):
        ax3.axvline(x=x_st[j], color='gray', linestyle='--', alpha=0.3)
    ax3.set_title('Spatiotemporal Grid', fontsize=12)
    ax3.set_xlabel('x (space)')
    ax3.set_ylabel('t (time)')

    # Show time progression direction
    ax3.annotate('', xy=(1.1, 0.4), xytext=(1.1, 0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax3.text(1.15, 0.25, 'Time', rotation=90, va='center', color='red')

    plt.tight_layout()
    plt.savefig('grids.png', dpi=150, bbox_inches='tight')
    plt.show()

# visualize_grids()
```

---

## 3. Deriving Difference Formulas

### 3.1 Taylor Series-Based Derivation

Difference formulas are derived using Taylor series.

```
f(x+h) = f(x) + h·f'(x) + (h²/2!)·f''(x) + (h³/3!)·f'''(x) + O(h⁴)
f(x-h) = f(x) - h·f'(x) + (h²/2!)·f''(x) - (h³/3!)·f'''(x) + O(h⁴)
```

```python
def derive_difference_formulas():
    """Derivation process for difference formulas"""

    print("=" * 60)
    print("Deriving Difference Formulas (Taylor Expansion)")
    print("=" * 60)

    # Forward difference derivation
    print("\n[1] Forward Difference")
    print("    f(x+h) = f(x) + h·f'(x) + (h²/2)·f''(x) + O(h³)")
    print("    Rearranging:")
    print("    f'(x) = [f(x+h) - f(x)] / h - (h/2)·f''(ξ)")
    print("    ≈ [f(x+h) - f(x)] / h + O(h)")
    print("    → First-order accurate")

    # Backward difference derivation
    print("\n[2] Backward Difference")
    print("    f(x-h) = f(x) - h·f'(x) + (h²/2)·f''(x) + O(h³)")
    print("    Rearranging:")
    print("    f'(x) = [f(x) - f(x-h)] / h + (h/2)·f''(ξ)")
    print("    ≈ [f(x) - f(x-h)] / h + O(h)")
    print("    → First-order accurate")

    # Central difference derivation
    print("\n[3] Central Difference")
    print("    f(x+h) - f(x-h) = 2h·f'(x) + (2h³/6)·f'''(x) + O(h⁵)")
    print("    Rearranging:")
    print("    f'(x) = [f(x+h) - f(x-h)] / 2h - (h²/6)·f'''(ξ)")
    print("    ≈ [f(x+h) - f(x-h)] / 2h + O(h²)")
    print("    → Second-order accurate")

    # Second derivative central difference
    print("\n[4] Second Derivative Central Difference")
    print("    f(x+h) + f(x-h) = 2f(x) + h²·f''(x) + (h⁴/12)·f''''(x) + O(h⁶)")
    print("    Rearranging:")
    print("    f''(x) = [f(x+h) - 2f(x) + f(x-h)] / h² + O(h²)")
    print("    → Second-order accurate")

derive_difference_formulas()
```

### 3.2 Summary of Difference Formulas

#### First Derivative (∂u/∂x)

| Name | Formula | Accuracy | Stencil |
|------|------|--------|--------|
| Forward | (u_{i+1} - u_i) / Δx | O(Δx) | [i, i+1] |
| Backward | (u_i - u_{i-1}) / Δx | O(Δx) | [i-1, i] |
| Central | (u_{i+1} - u_{i-1}) / 2Δx | O(Δx²) | [i-1, i+1] |

#### Second Derivative (∂²u/∂x²)

| Name | Formula | Accuracy |
|------|------|--------|
| Central | (u_{i+1} - 2u_i + u_{i-1}) / Δx² | O(Δx²) |

```python
def difference_operators():
    """Implement difference operators"""

    def forward_diff(u, dx, i):
        """Forward difference: ∂u/∂x ≈ (u[i+1] - u[i]) / dx"""
        return (u[i+1] - u[i]) / dx

    def backward_diff(u, dx, i):
        """Backward difference: ∂u/∂x ≈ (u[i] - u[i-1]) / dx"""
        return (u[i] - u[i-1]) / dx

    def central_diff_1st(u, dx, i):
        """Central difference (1st derivative): ∂u/∂x ≈ (u[i+1] - u[i-1]) / (2*dx)"""
        return (u[i+1] - u[i-1]) / (2 * dx)

    def central_diff_2nd(u, dx, i):
        """Central difference (2nd derivative): ∂²u/∂x² ≈ (u[i+1] - 2*u[i] + u[i-1]) / dx²"""
        return (u[i+1] - 2*u[i] + u[i-1]) / dx**2

    return forward_diff, backward_diff, central_diff_1st, central_diff_2nd

# Vectorized version
def apply_diff_operators(u, dx):
    """Apply difference operators to entire array"""

    # 1st derivative (central difference, interior points)
    du_dx = np.zeros_like(u)
    du_dx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
    # Boundaries: forward/backward difference
    du_dx[0] = (u[1] - u[0]) / dx
    du_dx[-1] = (u[-1] - u[-2]) / dx

    # 2nd derivative (central difference, interior points)
    d2u_dx2 = np.zeros_like(u)
    d2u_dx2[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2

    return du_dx, d2u_dx2

# Test
x = np.linspace(0, np.pi, 101)
dx = x[1] - x[0]
u = np.sin(x)  # u = sin(x)

du_dx, d2u_dx2 = apply_diff_operators(u, dx)

# Compare
print(f"At position x = π/2:")
print(f"  Numerical ∂u/∂x = {du_dx[50]:.6f}, exact = {np.cos(x[50]):.6f}")
print(f"  Numerical ∂²u/∂x² = {d2u_dx2[50]:.6f}, exact = {-np.sin(x[50]):.6f}")
```

---

## 4. Truncation Error Analysis

### 4.1 What is Truncation Error?

Truncation error consists of the higher-order terms that are truncated when approximating a derivative with a difference.

```python
def truncation_error_analysis():
    """Analyze truncation error"""

    # f(x) = sin(x), f''(x) = -sin(x)
    f = lambda x: np.sin(x)
    f_exact = lambda x: np.cos(x)
    f_2nd = lambda x: -np.sin(x)

    x = np.pi / 4
    h_values = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])

    errors_forward = []
    errors_central = []

    for h in h_values:
        # Forward difference error
        forward = (f(x + h) - f(x)) / h
        err_forward = abs(forward - f_exact(x))
        errors_forward.append(err_forward)

        # Central difference error
        central = (f(x + h) - f(x - h)) / (2 * h)
        err_central = abs(central - f_exact(x))
        errors_central.append(err_central)

    errors_forward = np.array(errors_forward)
    errors_central = np.array(errors_central)

    # Calculate convergence order
    order_forward = np.log(errors_forward[:-1] / errors_forward[1:]) / np.log(2)
    order_central = np.log(errors_central[:-1] / errors_central[1:]) / np.log(2)

    print("Truncation Error Analysis")
    print("=" * 70)
    print(f"{'h':<12} {'Forward Error':<18} {'Order':<8} {'Central Error':<18} {'Order':<8}")
    print("-" * 70)

    for i, h in enumerate(h_values):
        order_f = order_forward[i-1] if i > 0 else '-'
        order_c = order_central[i-1] if i > 0 else '-'
        if i > 0:
            print(f"{h:<12.4f} {errors_forward[i]:<18.2e} {order_f:<8.2f} {errors_central[i]:<18.2e} {order_c:<8.2f}")
        else:
            print(f"{h:<12.4f} {errors_forward[i]:<18.2e} {'-':<8} {errors_central[i]:<18.2e} {'-':<8}")

    print()
    print("Conclusion:")
    print(f"  Forward difference: O(h) - 1st order accuracy (halving h halves error)")
    print(f"  Central difference: O(h²) - 2nd order accuracy (halving h quarters error)")

    return h_values, errors_forward, errors_central

h_values, errors_forward, errors_central = truncation_error_analysis()
```

### 4.2 Error Convergence Visualization

```python
def plot_convergence():
    """Visualize error convergence"""
    fig, ax = plt.subplots(figsize=(10, 6))

    h_values, errors_forward, errors_central = truncation_error_analysis()

    # Log-log plot
    ax.loglog(h_values, errors_forward, 'o-', label='Forward (1st order)', linewidth=2, markersize=8)
    ax.loglog(h_values, errors_central, 's-', label='Central (2nd order)', linewidth=2, markersize=8)

    # Reference lines
    ax.loglog(h_values, h_values * 0.5, 'k--', alpha=0.5, label='O(h)')
    ax.loglog(h_values, h_values**2 * 2, 'k:', alpha=0.5, label='O(h²)')

    ax.set_xlabel('h (grid spacing)', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Finite Difference Truncation Error Convergence', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('truncation_error.png', dpi=150, bbox_inches='tight')
    plt.show()

# plot_convergence()
```

---

## 5. CFL Condition (Courant-Friedrichs-Lewy Condition)

### 5.1 What is the CFL Condition?

The CFL condition is a condition that ensures stability in numerical methods for time-dependent PDEs.

```
Physical meaning:
- Numerical information propagation speed ≥ Physical information propagation speed

Mathematical conditions:
- Advection equation: c·Δt/Δx ≤ 1 (Courant number ≤ 1)
- Heat equation: α·Δt/Δx² ≤ 1/2
- Wave equation: c·Δt/Δx ≤ 1
```

### The CFL Condition: Physical Intuition

The Courant-Friedrichs-Lewy (CFL) condition is the most important stability constraint in explicit finite difference methods:

```
c · Δt / Δx ≤ 1     (Courant number C ≤ 1)
```

The physical meaning is elegant: **the numerical domain of dependence must contain the physical domain of dependence**. Information in a PDE propagates at speed c. In one time step Δt, physical information travels a distance c·Δt. The numerical stencil reaches Δx in one step. If c·Δt > Δx, the physical wave outruns the numerical grid, and the scheme misses critical information — leading to instability.

```python
def cfl_condition_demo():
    """Demonstrate CFL condition"""

    print("CFL Condition (Courant-Friedrichs-Lewy)")
    print("=" * 60)

    # Advection equation
    print("\n[1] Advection equation: ∂u/∂t + c·∂u/∂x = 0")
    print("    CFL condition: C = c·Δt/Δx ≤ 1")
    print("    → Courant number C must be less than or equal to 1 for stability")

    c = 1.0  # Propagation speed
    dx = 0.1
    dt_max = dx / c
    print(f"    Example: c = {c}, Δx = {dx}")
    print(f"    Maximum allowed Δt = {dt_max}")

    # Heat equation
    print("\n[2] Heat equation: ∂u/∂t = α·∂²u/∂x² (FTCS)")
    print("    CFL condition: r = α·Δt/Δx² ≤ 1/2")

    alpha = 0.01  # Thermal diffusivity
    dx = 0.1
    dt_max = 0.5 * dx**2 / alpha
    print(f"    Example: α = {alpha}, Δx = {dx}")
    print(f"    Maximum allowed Δt = {dt_max}")

    # Wave equation
    print("\n[3] Wave equation: ∂²u/∂t² = c²·∂²u/∂x²")
    print("    CFL condition: C = c·Δt/Δx ≤ 1")

    c = 1.0
    dx = 0.1
    dt_max = dx / c
    print(f"    Example: c = {c}, Δx = {dx}")
    print(f"    Maximum allowed Δt = {dt_max}")

cfl_condition_demo()
```

### 5.2 CFL Calculator

```python
class CFLCalculator:
    """CFL condition calculation and verification"""

    @staticmethod
    def advection(c, dx, dt):
        """
        Calculate CFL number for advection equation

        Parameters:
        -----------
        c : float - propagation speed
        dx : float - spatial grid spacing
        dt : float - time step

        Returns:
        --------
        C : float - Courant number
        stable : bool - stability status
        """
        C = abs(c) * dt / dx
        stable = C <= 1.0
        return C, stable

    @staticmethod
    def heat_ftcs(alpha, dx, dt):
        """
        Calculate CFL number for heat equation (FTCS)

        Parameters:
        -----------
        alpha : float - thermal diffusivity
        """
        r = alpha * dt / dx**2
        stable = r <= 0.5
        return r, stable

    @staticmethod
    def wave(c, dx, dt):
        """Calculate CFL number for wave equation"""
        C = c * dt / dx
        stable = C <= 1.0
        return C, stable

    @staticmethod
    def max_dt_advection(c, dx, safety=0.9):
        """Calculate maximum allowed dt for advection equation"""
        return safety * dx / abs(c)

    @staticmethod
    def max_dt_heat(alpha, dx, safety=0.9):
        """Calculate maximum allowed dt for heat equation (FTCS)"""
        return safety * 0.5 * dx**2 / alpha

    @staticmethod
    def max_dt_wave(c, dx, safety=0.9):
        """Calculate maximum allowed dt for wave equation"""
        return safety * dx / c

# Usage example
cfl = CFLCalculator()

# Heat equation example
alpha = 0.01
dx = 0.02
dt = 0.001

r, stable = cfl.heat_ftcs(alpha, dx, dt)
print(f"Heat equation CFL analysis:")
print(f"  α = {alpha}, Δx = {dx}, Δt = {dt}")
print(f"  r = α·Δt/Δx² = {r:.4f}")
print(f"  Stable: {stable}")
print(f"  Recommended maximum Δt = {cfl.max_dt_heat(alpha, dx):.6f}")
```

---

## 6. von Neumann Stability Analysis

### 6.1 Analysis Principle

von Neumann stability analysis examines the growth of Fourier modes over time.

```
Assumption: u_j^n = G^n · e^{i·k·j·Δx}

Where:
- G: amplification factor
- k: wave number
- j: spatial index
- n: time step

Stability condition: |G| ≤ 1 (for all k)
```

```python
def von_neumann_analysis():
    """von Neumann stability analysis"""

    print("von Neumann Stability Analysis")
    print("=" * 60)

    # FTCS heat equation analysis
    print("\n[Example] FTCS Heat Equation")
    print("    u_j^{n+1} = u_j^n + r·(u_{j+1}^n - 2·u_j^n + u_{j-1}^n)")
    print("    where r = α·Δt/Δx²")
    print()
    print("    Substitute u_j^n = G^n·e^{ikjΔx}:")
    print("    G·e^{ikjΔx} = e^{ikjΔx} + r·(e^{ik(j+1)Δx} - 2·e^{ikjΔx} + e^{ik(j-1)Δx})")
    print()
    print("    Divide both sides by e^{ikjΔx}:")
    print("    G = 1 + r·(e^{ikΔx} + e^{-ikΔx} - 2)")
    print("      = 1 + r·(2cos(kΔx) - 2)")
    print("      = 1 - 2r·(1 - cos(kΔx))")
    print("      = 1 - 4r·sin²(kΔx/2)")
    print()
    print("    Stability condition |G| ≤ 1:")
    print("    -1 ≤ 1 - 4r·sin²(kΔx/2) ≤ 1")
    print()
    print("    From left inequality:")
    print("    -2 ≤ -4r·sin²(kΔx/2)")
    print("    4r·sin²(kΔx/2) ≤ 2")
    print()
    print("    Maximum value of sin²(kΔx/2) = 1 (at kΔx = π):")
    print("    4r ≤ 2")
    print("    r ≤ 1/2")
    print()
    print("    Conclusion: FTCS heat equation is stable when r = α·Δt/Δx² ≤ 1/2")

von_neumann_analysis()
```

### 6.2 Amplification Factor Visualization

```python
def plot_amplification_factor():
    """Visualize amplification factor"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # k*dx range
    kh = np.linspace(0, np.pi, 100)

    # FTCS heat equation amplification factor
    ax1 = axes[0]
    r_values = [0.1, 0.25, 0.5, 0.6, 0.8]

    for r in r_values:
        G = 1 - 4 * r * np.sin(kh / 2)**2
        label = f'r = {r}' + (' (unstable)' if r > 0.5 else '')
        linestyle = '--' if r > 0.5 else '-'
        ax1.plot(kh, G, label=label, linestyle=linestyle, linewidth=2)

    ax1.axhline(y=1, color='red', linestyle=':', alpha=0.7)
    ax1.axhline(y=-1, color='red', linestyle=':', alpha=0.7)
    ax1.fill_between(kh, -1, 1, alpha=0.1, color='green', label='Stable region')
    ax1.set_xlabel('kΔx', fontsize=12)
    ax1.set_ylabel('G (amplification factor)', fontsize=12)
    ax1.set_title('FTCS Heat Equation Amplification Factor\nG = 1 - 4r·sin²(kΔx/2)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, np.pi)
    ax1.set_ylim(-2, 1.5)

    # Compare multiple schemes (advection equation)
    ax2 = axes[1]
    C = 0.8  # Courant number

    # FTCS (unstable)
    G_ftcs = 1 - 1j * C * np.sin(kh)
    ax2.plot(kh, np.abs(G_ftcs), label='FTCS (unstable)', linewidth=2)

    # Upwind method
    G_upwind = 1 - C * (1 - np.cos(kh)) - 1j * C * np.sin(kh)
    ax2.plot(kh, np.abs(G_upwind), label='Upwind', linewidth=2)

    # Lax-Friedrichs
    G_lax = np.cos(kh) - 1j * C * np.sin(kh)
    ax2.plot(kh, np.abs(G_lax), label='Lax-Friedrichs', linewidth=2)

    ax2.axhline(y=1, color='red', linestyle=':', alpha=0.7, label='Stability limit')
    ax2.set_xlabel('kΔx', fontsize=12)
    ax2.set_ylabel('|G| (amplification factor magnitude)', fontsize=12)
    ax2.set_title(f'Advection Equation Amplification Factor Comparison (C = {C})', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, np.pi)
    ax2.set_ylim(0, 2)

    plt.tight_layout()
    plt.savefig('amplification_factor.png', dpi=150, bbox_inches='tight')
    plt.show()

# plot_amplification_factor()
```

### 6.3 Verification by Numerical Experiment

```python
def stability_experiment():
    """Stability numerical experiment"""

    # Parameters
    L = 1.0
    nx = 51
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)

    alpha = 0.01

    # Initial condition
    u0 = np.sin(np.pi * x)

    # FTCS scheme
    def ftcs_step(u, r):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
        u_new[0] = 0  # Boundary condition
        u_new[-1] = 0
        return u_new

    # Experiment with different r values
    r_values = [0.4, 0.5, 0.6]
    n_steps = 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, r in enumerate(r_values):
        dt = r * dx**2 / alpha
        u = u0.copy()

        ax = axes[idx]
        ax.plot(x, u0, 'b--', label='Initial', alpha=0.5)

        for step in range(n_steps):
            u = ftcs_step(u, r)
            if step in [20, 50, 99]:
                ax.plot(x, u, label=f'step {step+1}')

        stable = r <= 0.5
        status = "Stable" if stable else "Unstable"
        ax.set_title(f'r = {r} ({status})\ndt = {dt:.6f}', fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if stable:
            ax.set_ylim(-1.5, 1.5)
        else:
            ax.set_ylim(-10, 10)

    plt.tight_layout()
    plt.savefig('stability_experiment.png', dpi=150, bbox_inches='tight')
    plt.show()

# stability_experiment()
```

---

## 7. High-Order Accuracy Difference Formulas

### 7.1 Fourth-Order Accuracy Formulas

```python
def high_order_formulas():
    """High-order accuracy difference formulas"""

    print("High-Order Accuracy Difference Formulas")
    print("=" * 60)

    # 1st derivative 4th order accuracy
    print("\n[1] First Derivative (4th order accuracy)")
    print("    f'(x) ≈ [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / 12h")
    print("    Truncation error: O(h⁴)")

    # 2nd derivative 4th order accuracy
    print("\n[2] Second Derivative (4th order accuracy)")
    print("    f''(x) ≈ [-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)] / 12h²")
    print("    Truncation error: O(h⁴)")

    # Numerical verification
    f = lambda x: np.sin(x)
    f_1 = lambda x: np.cos(x)
    f_2 = lambda x: -np.sin(x)

    x = np.pi / 4
    h = 0.1

    # 1st derivative
    d1_2nd = (f(x + h) - f(x - h)) / (2 * h)  # 2nd order accuracy
    d1_4th = (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12 * h)  # 4th order accuracy

    print(f"\nNumerical verification (h = {h}):")
    print(f"  Exact f'(π/4) = {f_1(x):.10f}")
    print(f"  2nd order: {d1_2nd:.10f}, error: {abs(d1_2nd - f_1(x)):.2e}")
    print(f"  4th order: {d1_4th:.10f}, error: {abs(d1_4th - f_1(x)):.2e}")

    # 2nd derivative
    d2_2nd = (f(x + h) - 2*f(x) + f(x - h)) / h**2
    d2_4th = (-f(x + 2*h) + 16*f(x + h) - 30*f(x) + 16*f(x - h) - f(x - 2*h)) / (12 * h**2)

    print(f"\n  Exact f''(π/4) = {f_2(x):.10f}")
    print(f"  2nd order: {d2_2nd:.10f}, error: {abs(d2_2nd - f_2(x)):.2e}")
    print(f"  4th order: {d2_4th:.10f}, error: {abs(d2_4th - f_2(x)):.2e}")

high_order_formulas()
```

### 7.2 Difference Coefficient Generator

```python
from scipy.special import factorial

def compute_fd_coefficients(derivative_order, accuracy_order, positions=None):
    """
    Compute finite difference coefficients

    Parameters:
    -----------
    derivative_order : int - order of derivative (1=1st derivative, 2=2nd derivative, ...)
    accuracy_order : int - order of accuracy
    positions : list - stencil positions (default: central difference)

    Returns:
    --------
    coeffs : array - difference coefficients
    positions : array - stencil positions
    """
    import numpy as np

    if positions is None:
        # Central difference stencil
        n_points = derivative_order + accuracy_order
        if n_points % 2 == 0:
            n_points += 1
        half = n_points // 2
        positions = np.arange(-half, half + 1)

    n = len(positions)
    positions = np.array(positions, dtype=float)

    # Construct Vandermonde matrix
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = positions[j] ** i

    # Right-hand side vector
    b = np.zeros(n)
    b[derivative_order] = factorial(derivative_order)

    # Calculate coefficients
    coeffs = np.linalg.solve(A, b)

    return coeffs, positions

# Example: 1st derivative, 2nd order accuracy
coeffs, pos = compute_fd_coefficients(1, 2)
print("1st derivative (2nd order accuracy):")
print(f"  Positions: {pos}")
print(f"  Coefficients: {coeffs}")
print(f"  Formula: ({coeffs[0]:.1f}·f[i-1] + {coeffs[1]:.1f}·f[i] + {coeffs[2]:.1f}·f[i+1]) / h")

# 2nd derivative, 2nd order accuracy
coeffs, pos = compute_fd_coefficients(2, 2)
print("\n2nd derivative (2nd order accuracy):")
print(f"  Positions: {pos}")
print(f"  Coefficients: {coeffs}")
```

---

## 8. Efficient Implementation Using Sparse Matrices

### 8.1 Using scipy.sparse

```python
from scipy import sparse
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

def create_laplacian_1d(nx, dx, bc_type='dirichlet'):
    """
    Create 1D Laplacian matrix (sparse matrix)

    Parameters:
    -----------
    nx : int - number of grid points
    dx : float - grid spacing
    bc_type : str - boundary condition type

    Returns:
    --------
    L : sparse matrix - Laplacian matrix
    """
    n = nx - 2  # Number of interior points (Dirichlet BC)

    # Diagonal elements
    main_diag = -2.0 * np.ones(n)
    off_diag = np.ones(n - 1)

    # Create sparse matrix
    L = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')
    L = L / dx**2

    return L

# Example
nx = 101
dx = 1.0 / (nx - 1)
L = create_laplacian_1d(nx, dx)

print(f"Laplacian matrix size: {L.shape}")
print(f"Number of non-zero elements: {L.nnz}")
print(f"Density: {L.nnz / (L.shape[0] * L.shape[1]) * 100:.2f}%")
print(f"\nMatrix excerpt:")
print(L.toarray()[:5, :5])
```

### 8.2 2D Laplacian Matrix

```python
def create_laplacian_2d(nx, ny, dx, dy):
    """
    Create 2D Laplacian matrix (5-point stencil)

    d²u/dx² + d²u/dy² ≈ (u_{i+1,j} - 2u_{i,j} + u_{i-1,j})/dx²
                       + (u_{i,j+1} - 2u_{i,j} + u_{i,j-1})/dy²

    Matrix indexing: Flatten interior points in row-major order to 1D
    k = (j-1)*(nx-2) + (i-1)  (i, j are 1-based interior point indices)
    """
    mx = nx - 2  # Number of interior points in x direction
    my = ny - 2  # Number of interior points in y direction
    n = mx * my  # Total interior points

    # Coefficients
    cx = 1.0 / dx**2
    cy = 1.0 / dy**2
    cc = -2.0 * (cx + cy)

    # Construct diagonals
    main_diag = cc * np.ones(n)
    x_diag = cx * np.ones(n - 1)
    y_diag = cy * np.ones(n - mx)

    # x-direction neighbor connections (break at row boundaries)
    for j in range(my):
        if j < my - 1:
            x_diag[j * mx + mx - 1] = 0

    # Create sparse matrix
    diagonals = [y_diag, x_diag, main_diag, x_diag, y_diag]
    offsets = [-mx, -1, 0, 1, mx]

    L = diags(diagonals, offsets, shape=(n, n), format='csr')

    return L

# Example
nx, ny = 11, 11
dx = dy = 1.0 / (nx - 1)
L = create_laplacian_2d(nx, ny, dx, dy)

print(f"2D Laplacian matrix size: {L.shape}")
print(f"Number of non-zero elements: {L.nnz}")
print(f"Density: {L.nnz / (L.shape[0] * L.shape[1]) * 100:.2f}%")
```

### 8.3 Sparse Matrix Visualization

```python
def visualize_sparse_matrices():
    """Visualize sparse matrix structure"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1D Laplacian
    L1d = create_laplacian_1d(21, 0.05)
    ax1 = axes[0]
    ax1.spy(L1d, markersize=3)
    ax1.set_title(f'1D Laplacian (n={L1d.shape[0]})\nTridiagonal Structure', fontsize=12)

    # 2D Laplacian (small size)
    L2d = create_laplacian_2d(7, 7, 0.1, 0.1)
    ax2 = axes[1]
    ax2.spy(L2d, markersize=5)
    ax2.set_title(f'2D Laplacian ({L2d.shape[0]}x{L2d.shape[0]})\nPentadiagonal Structure', fontsize=12)

    # Sparsity comparison
    ax3 = axes[2]
    sizes = [11, 21, 41, 81, 161]
    densities_1d = []
    densities_2d = []

    for s in sizes:
        L1 = create_laplacian_1d(s, 1.0/(s-1))
        densities_1d.append(L1.nnz / (L1.shape[0]**2) * 100)

        L2 = create_laplacian_2d(s, s, 1.0/(s-1), 1.0/(s-1))
        densities_2d.append(L2.nnz / (L2.shape[0]**2) * 100)

    ax3.semilogy(sizes, densities_1d, 'o-', label='1D Laplacian', linewidth=2)
    ax3.semilogy(sizes, densities_2d, 's-', label='2D Laplacian', linewidth=2)
    ax3.set_xlabel('Grid size n', fontsize=12)
    ax3.set_ylabel('Matrix density (%)', fontsize=12)
    ax3.set_title('Sparse Matrix Density\n(Change with Grid Size)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sparse_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()

# visualize_sparse_matrices()
```

---

## 9. Comprehensive Example: Solving Poisson Equation

```python
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
import matplotlib.pyplot as plt

def solve_poisson_1d():
    """
    Solve 1D Poisson equation

    -d²u/dx² = f(x), 0 < x < 1
    Boundary conditions: u(0) = 0, u(1) = 0
    Source term: f(x) = π²·sin(πx)
    Analytical solution: u(x) = sin(πx)
    """
    # Parameters
    nx = 101
    L = 1.0
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)

    # Source term (interior points only)
    x_inner = x[1:-1]
    f = np.pi**2 * np.sin(np.pi * x_inner)

    # Laplacian matrix (-d²/dx²)
    n = nx - 2
    main_diag = 2.0 * np.ones(n)
    off_diag = -1.0 * np.ones(n - 1)
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')
    A = A / dx**2

    # Solve linear system
    u_inner = spsolve(A, f)

    # Full solution (including boundary conditions)
    u = np.zeros(nx)
    u[1:-1] = u_inner
    u[0] = 0  # Dirichlet BC
    u[-1] = 0

    # Analytical solution
    u_exact = np.sin(np.pi * x)

    # Error
    error = np.max(np.abs(u - u_exact))
    print(f"1D Poisson Equation Solution")
    print(f"  Grid points: {nx}")
    print(f"  Maximum error: {error:.2e}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(x, u_exact, 'b-', label='Analytical', linewidth=2)
    ax1.plot(x, u, 'ro', label='Numerical', markersize=4, markevery=5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x)')
    ax1.set_title('1D Poisson Equation Solution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, np.abs(u - u_exact), 'g-', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('|Error|')
    ax2.set_title(f'Numerical Solution Error (max: {error:.2e})')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('poisson_1d.png', dpi=150, bbox_inches='tight')
    plt.show()

    return x, u, u_exact

# x, u, u_exact = solve_poisson_1d()
```

---

## 10. Summary

### Key Concepts Review

| Concept | Description |
|------|------|
| Finite Difference Method | Approximate derivatives with finite differences |
| Forward difference | (u_{i+1} - u_i)/Δx, O(Δx) |
| Backward difference | (u_i - u_{i-1})/Δx, O(Δx) |
| Central difference | (u_{i+1} - u_{i-1})/(2Δx), O(Δx²) |
| Truncation error | Higher-order terms truncated from Taylor expansion |
| CFL condition | Δt/Δx constraint for numerical stability |
| von Neumann analysis | Analyze Fourier mode amplification factor |
| Sparse matrices | Efficient storage/computation for large systems |

### Key Formulas

```
1st derivative (central): f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
2nd derivative (central): f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²

CFL conditions:
- Advection equation: c·Δt/Δx ≤ 1
- Heat equation (FTCS): α·Δt/Δx² ≤ 0.5
- Wave equation: c·Δt/Δx ≤ 1
```

### Next Steps

1. **Chapter 09**: Heat Equation - FTCS, BTCS, Crank-Nicolson
2. **Chapter 10**: Wave Equation - CTCS, boundary condition handling
3. **Chapter 11**: Laplace/Poisson - iterative methods
4. **Chapter 12**: Advection Equation - upwind method, numerical diffusion

---

## Exercises

### Exercise 1: Verify Difference Accuracy
For f(x) = e^x at x = 1, calculate forward/backward/central differences and compare errors at h = 0.1, 0.01, 0.001.

### Exercise 2: Numerical Second Derivative
For f(x) = x⁴, calculate f''(x) using central differences and compare with the exact value 12x².

### Exercise 3: Calculate CFL Condition
For thermal diffusivity α = 0.05 and grid spacing Δx = 0.01, find the maximum time step Δt for FTCS method stability.

### Exercise 4: Sparse Matrix Poisson Equation
Modify the 1D Poisson example above to solve for f(x) = 1 (constant). (Analytical solution: u(x) = x(1-x)/2)

---

## References

1. **Textbooks**:
   - LeVeque, "Finite Difference Methods for Ordinary and Partial Differential Equations"
   - Strikwerda, "Finite Difference Schemes and Partial Differential Equations"

2. **Python Libraries**:
   - scipy.sparse: Sparse matrix operations
   - numpy: Array operations

3. **Online**:
   - MIT OCW 18.336: Numerical Methods for PDEs
