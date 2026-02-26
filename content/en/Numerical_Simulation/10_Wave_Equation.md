# 10. Wave Equation

## Learning Objectives
- Understand the physical meaning of 1D/2D wave equations
- Implement the CTCS (Central Time Central Space) method
- Handle various boundary conditions (fixed, free, absorbing)
- Visualize wave propagation animations

---

**Why does the wave equation require a different approach than the heat equation?** The heat equation is **dissipative** -- it smooths out disturbances over time, and energy monotonically decreases. The wave equation is **dispersive** -- it propagates disturbances without damping, and energy is conserved. This fundamental difference means numerical schemes that work well for heat (like implicit methods that damp high-frequency modes) are unnecessary for waves, while schemes that preserve wave amplitude and minimize numerical dispersion become essential.

## 1. Wave Equation Theory

### 1.1 Physical Background

The wave equation is a hyperbolic PDE that describes wave phenomena.

```
1D Wave Equation:
d²u/dt² = c² · d²u/dx²

where:
- u(x,t): displacement
- c: wave propagation speed
- x: spatial coordinate
- t: time
```

### 1.2 Application Fields

| Field | Physical quantity u | Propagation speed c |
|-------|---------------------|---------------------|
| String vibration | displacement | sqrt(T/rho) (T: tension, rho: linear density) |
| Sound waves | pressure | sqrt(gamma*P/rho) (~340 m/s in air) |
| Electromagnetic waves | electric/magnetic field | speed of light (~3x10^8 m/s) |
| Seismic waves | ground displacement | several km/s |

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Wave speeds for various materials (examples)
wave_speeds = {
    'Guitar string (E)': 329.0,   # Hz converted to m/s
    'Air (sound, 20°C)': 343.0,
    'Water (sound)': 1481.0,
    'Steel (longitudinal)': 5960.0,
    'Light (vacuum)': 299792458.0,
}

for material, c in wave_speeds.items():
    print(f"{material}: c = {c:.0f} m/s")
```

### 1.3 D'Alembert's Solution

D'Alembert showed that the general solution in an infinite domain decomposes into two counter-propagating waves:

$$u(x,t) = f(x - ct) + g(x + ct)$$

where $f$ travels to the right and $g$ travels to the left, both at speed $c$. The wave shapes are preserved exactly -- the wave equation has no dispersion or dissipation. The functions $f$ and $g$ are determined by the initial displacement $u(x,0)$ and initial velocity $u_t(x,0)$.

```
u(x,t) = f(x - ct) + g(x + ct)

- f(x - ct): wave traveling to the right
- g(x + ct): wave traveling to the left
```

```python
import numpy as np
import matplotlib.pyplot as plt

def dalembert_demo():
    """D'Alembert solution visualization"""
    c = 1.0  # wave speed
    x = np.linspace(-5, 5, 500)

    # Initial condition: Gaussian pulse
    def f(x):
        return np.exp(-x**2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    times = [0, 0.5, 1.0, 1.5, 2.0, 2.5]

    for idx, t in enumerate(times):
        ax = axes[idx // 3, idx % 3]

        # Right-traveling wave
        u_right = 0.5 * f(x - c*t)
        # Left-traveling wave
        u_left = 0.5 * f(x + c*t)
        # Total solution
        u_total = u_right + u_left

        ax.plot(x, u_right, 'b--', alpha=0.5, label='Right-traveling')
        ax.plot(x, u_left, 'r--', alpha=0.5, label='Left-traveling')
        ax.plot(x, u_total, 'k-', linewidth=2, label='Total')

        ax.set_xlim(-5, 5)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title(f't = {t:.1f}')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()

    plt.suptitle("D'Alembert Solution: u = f(x-ct) + f(x+ct)", fontsize=14)
    plt.tight_layout()
    plt.savefig('wave_dalembert.png', dpi=150, bbox_inches='tight')
    plt.show()

# dalembert_demo()
```

---

## 2. CTCS Method (Central Time Central Space)

**Why CTCS for waves?** The wave equation is second-order in both time and space, so central differences (which are second-order accurate) are the natural choice for both. CTCS is explicit (no equation solving needed), time-reversible (matching the physics), and energy-conserving when $C = 1$. It is the gold standard for wave propagation problems.

### 2.1 Discretization

CTCS uses central differencing for both time and space.

```
Time: Central difference
d²u/dt² ≈ (u_i^{n+1} - 2u_i^n + u_i^{n-1}) / dt²

Space: Central difference
d²u/dx² ≈ (u_{i+1}^n - 2u_i^n + u_{i-1}^n) / dx²

Combined (rearranged):
u_i^{n+1} = 2u_i^n - u_i^{n-1} + C² · (u_{i+1}^n - 2u_i^n + u_{i-1}^n)

where C = c·dt/dx (Courant number)
```

### 2.2 CTCS Stencil

```
Time n+1:         [i]
                   |
Time n:    [i-1]--[i]--[i+1]
                   |
Time n-1:         [i]
```

### 2.3 Stability Condition

The CFL (Courant-Friedrichs-Lewy) condition is one of the most important results in computational physics:

$$C = \frac{c \cdot \Delta t}{\Delta x} \leq 1$$

The Courant number $C$ compares two speeds: the numerical domain of dependence ($\Delta x / \Delta t$) and the physical propagation speed ($c$). If $C > 1$, physical signals travel faster than numerical information can propagate on the grid, and the scheme becomes unstable. When $C = 1$ exactly, CTCS reproduces the exact solution (no numerical dispersion), because the grid perfectly matches the wave's characteristic lines.

```
CFL Condition: C = c·dt/dx <= 1

Physical meaning:
- Numerical information propagation speed (dx/dt) >= physical propagation speed (c)
- Most accurate when C = 1 (no numerical dispersion)
```

### 2.4 CTCS Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class WaveEquation1D:
    """
    1D Wave Equation CTCS Method

    d²u/dt² = c² · d²u/dx²
    """

    def __init__(self, L=1.0, c=1.0, nx=101, T=2.0, courant=0.9):
        """
        Parameters:
        -----------
        L : float - domain length
        c : float - wave propagation speed
        nx : int - number of spatial grid points
        T : float - final time
        courant : float - Courant number (<= 1)
        """
        self.L = L
        self.c = c
        self.nx = nx
        self.T = T

        # Grid generation
        self.dx = L / (nx - 1)
        self.x = np.linspace(0, L, nx)

        # Time step based on CFL condition: dt = C * dx / c
        # Choosing courant close to 1 minimizes numerical dispersion
        self.dt = courant * self.dx / c
        self.nt = int(np.ceil(T / self.dt))
        self.dt = T / self.nt

        self.C = c * self.dt / self.dx  # Courant number
        self.C2 = self.C ** 2

        print(f"1D Wave Equation CTCS Setup")
        print(f"  L = {L}, c = {c}")
        print(f"  nx = {nx}, dx = {self.dx:.4f}")
        print(f"  dt = {self.dt:.6f}, nt = {self.nt}")
        print(f"  Courant number C = {self.C:.4f}")
        print(f"  Stability: {'OK' if self.C <= 1 else 'WARNING!'}")

    def set_initial_conditions(self, u0_func, v0_func=None):
        """
        Set initial conditions

        Parameters:
        -----------
        u0_func : callable - u(x, 0) = u0_func(x)
        v0_func : callable - du/dt(x, 0) = v0_func(x)
        """
        self.u = u0_func(self.x)
        self.u0 = self.u.copy()

        if v0_func is None:
            v0_func = lambda x: np.zeros_like(x)

        # CTCS is a 3-level scheme (uses u at times n-1, n, n+1).
        # The first step needs special treatment because u^{-1} doesn't exist.
        # We use a Taylor expansion: u^1 ≈ u^0 + dt·v^0 + (dt²/2)·c²·u_xx^0
        # This preserves second-order accuracy at the first step.
        self.u_prev = self.u.copy()

        # Spatial second derivative
        d2u = np.zeros_like(self.u)
        d2u[1:-1] = (self.u[2:] - 2*self.u[1:-1] + self.u[:-2]) / self.dx**2

        self.u = self.u_prev + self.dt * v0_func(self.x) + \
                 0.5 * self.dt**2 * self.c**2 * d2u

    def set_boundary_conditions(self, bc_type='fixed', left_value=0, right_value=0):
        """
        Set boundary conditions

        Parameters:
        -----------
        bc_type : str - 'fixed', 'free', 'absorbing'
        """
        self.bc_type = bc_type
        self.bc_left = left_value
        self.bc_right = right_value

    def apply_bc(self, u, u_prev=None):
        """Apply boundary conditions"""
        if self.bc_type == 'fixed':
            # Dirichlet: u = constant
            u[0] = self.bc_left
            u[-1] = self.bc_right

        elif self.bc_type == 'free':
            # Neumann: du/dx = 0
            u[0] = u[1]
            u[-1] = u[-2]

        elif self.bc_type == 'absorbing':
            # Absorbing boundary (1st order Sommerfeld)
            # du/dt +/- c·du/dx = 0
            if u_prev is not None:
                # Left: du/dt - c·du/dx = 0 (outgoing to the right)
                u[0] = u_prev[0] + self.C * (u[1] - u_prev[1])
                # Right: du/dt + c·du/dx = 0 (outgoing to the left)
                u[-1] = u_prev[-1] - self.C * (u[-1] - u_prev[-2])

        return u

    def step(self):
        """Advance one time step (CTCS)"""
        u_new = np.zeros_like(self.u)

        # Interior point update
        u_new[1:-1] = (2*self.u[1:-1] - self.u_prev[1:-1] +
                       self.C2 * (self.u[2:] - 2*self.u[1:-1] + self.u[:-2]))

        # Apply boundary conditions
        u_new = self.apply_bc(u_new, self.u_prev)

        # Update
        self.u_prev = self.u.copy()
        self.u = u_new

    def solve(self, save_interval=None):
        """Solve over entire time interval"""
        if save_interval is None:
            save_interval = max(1, self.nt // 200)

        history = [self.u0.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), np.array(history)


def demo_wave_1d():
    """1D wave equation demo"""
    L = 1.0
    c = 1.0

    # Compare three boundary conditions
    bc_types = ['fixed', 'free', 'absorbing']
    results = {}

    for bc in bc_types:
        solver = WaveEquation1D(L=L, c=c, nx=201, T=3.0, courant=0.9)

        # Initial condition: Gaussian pulse
        def u0(x):
            x0 = 0.3
            sigma = 0.05
            return np.exp(-(x - x0)**2 / (2 * sigma**2))

        solver.set_initial_conditions(u0)
        solver.set_boundary_conditions(bc_type=bc)

        times, history = solver.solve(save_interval=10)
        results[bc] = (solver, times, history)

        print(f"\n{bc} boundary condition completed")

    # Visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))

    for row, bc in enumerate(bc_types):
        solver, times, history = results[bc]

        # Multiple time snapshots
        time_indices = [0, len(times)//4, len(times)//2, len(times)-1]

        for col, ti in enumerate(time_indices):
            ax = axes[row, col]
            ax.plot(solver.x, history[ti], 'b-', linewidth=1.5)
            ax.set_xlim(0, L)
            ax.set_ylim(-1.5, 1.5)
            ax.set_xlabel('x')
            ax.set_ylabel('u')
            ax.set_title(f't = {times[ti]:.2f}')
            ax.grid(True, alpha=0.3)

            if col == 0:
                ax.set_ylabel(f'{bc}\nu')

    plt.suptitle('1D Wave Equation: Boundary Condition Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('wave_1d_bc_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results

# results = demo_wave_1d()
```

---

## 3. Boundary Conditions in Detail

### 3.1 Fixed Boundary (Fixed/Dirichlet)

```
u(0, t) = 0, u(L, t) = 0

Physical meaning: Both ends of the string are fixed
Result: Wave reflects at boundary with phase inversion
```

### 3.2 Free Boundary (Free/Neumann)

```
du/dx(0, t) = 0, du/dx(L, t) = 0

Physical meaning: No force at boundary (free to move)
Result: Wave reflects at boundary with same phase
```

### 3.3 Absorbing Boundary (Absorbing/Sommerfeld)

Absorbing boundaries approximate an infinite domain by letting waves pass through without reflection. The first-order Sommerfeld condition assumes waves hit the boundary at normal incidence:

$$\frac{\partial u}{\partial t} + c\frac{\partial u}{\partial x} = 0 \quad \text{(right boundary, outgoing wave)}$$

This works perfectly for normally-incident plane waves, but reflects waves arriving at oblique angles. Higher-order absorbing conditions (or Perfectly Matched Layers) handle oblique incidence better.

```
du/dt + c·du/dx = 0 (right boundary)
du/dt - c·du/dx = 0 (left boundary)

Physical meaning: Wave exits through boundary (infinite domain approximation)
```

```python
def boundary_condition_comparison():
    """Boundary condition effect comparison"""
    L = 1.0
    c = 1.0
    nx = 201
    T = 4.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    bc_names = {'fixed': 'Fixed Boundary (Inverted Reflection)',
                'free': 'Free Boundary (Same Phase Reflection)',
                'absorbing': 'Absorbing Boundary (No Reflection)'}

    for idx, bc_type in enumerate(['fixed', 'free', 'absorbing']):
        solver = WaveEquation1D(L=L, c=c, nx=nx, T=T, courant=0.95)

        # Initial condition: Gaussian pulse moving toward center
        def u0(x):
            x0 = 0.2
            sigma = 0.05
            return np.exp(-(x - x0)**2 / (2 * sigma**2))

        solver.set_initial_conditions(u0)
        solver.set_boundary_conditions(bc_type=bc_type)

        times, history = solver.solve(save_interval=5)

        # Space-time contour plot
        ax = axes[idx]
        X, T_grid = np.meshgrid(solver.x, times)
        c_plot = ax.contourf(X, T_grid, history, levels=30, cmap='RdBu_r')
        plt.colorbar(c_plot, ax=ax)

        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title(bc_names[bc_type])

    plt.tight_layout()
    plt.savefig('wave_bc_spacetime.png', dpi=150, bbox_inches='tight')
    plt.show()

# boundary_condition_comparison()
```

---

## 4. Standing Waves and Normal Modes

Standing waves arise when two counter-propagating waves of equal amplitude interfere. They are the natural vibration patterns (eigenmodes) of bounded systems. Any vibration of a string fixed at both ends can be decomposed into a sum of these modes -- this is the physical basis of Fourier series.

### 4.1 Analytical Standing Wave Solution

Normal modes with fixed boundary conditions:

```
u_n(x, t) = sin(n*pi*x/L) · cos(n*pi*c*t/L)

Natural frequency: f_n = n*c/(2L)
```

```python
def standing_waves():
    """Standing wave normal modes"""
    L = 1.0
    c = 1.0
    x = np.linspace(0, L, 200)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for n in range(1, 7):
        ax = axes[(n-1)//3, (n-1)%3]

        # Standing wave at multiple times
        for phase in np.linspace(0, np.pi, 5):
            u = np.sin(n * np.pi * x / L) * np.cos(phase)
            alpha = 0.2 + 0.8 * (1 - abs(np.cos(phase)))
            ax.plot(x, u, 'b-', alpha=alpha)

        # Envelope
        ax.plot(x, np.sin(n * np.pi * x / L), 'r--', linewidth=2, label='Envelope')
        ax.plot(x, -np.sin(n * np.pi * x / L), 'r--', linewidth=2)

        f_n = n * c / (2 * L)
        ax.set_title(f'Mode {n}: f = {f_n:.2f} Hz')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_xlim(0, L)
        ax.set_ylim(-1.3, 1.3)
        ax.grid(True, alpha=0.3)

        # Mark nodes
        nodes = np.linspace(0, L, n+1)
        for node in nodes:
            ax.axvline(x=node, color='green', linestyle=':', alpha=0.5)

    plt.suptitle('Standing Wave Normal Modes (Fixed-Fixed Boundary)', fontsize=14)
    plt.tight_layout()
    plt.savefig('wave_standing_modes.png', dpi=150, bbox_inches='tight')
    plt.show()

# standing_waves()
```

### 4.2 Numerical vs Analytical Solution Comparison

```python
def compare_with_exact():
    """Compare numerical and analytical solutions"""
    L = 1.0
    c = 1.0

    # Analytical solution: u(x,t) = sin(pi*x)·cos(pi*c*t)
    def exact_solution(x, t):
        return np.sin(np.pi * x / L) * np.cos(np.pi * c * t / L)

    # Compare for different Courant numbers
    courant_values = [0.5, 0.8, 1.0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    T = 2.0

    for idx, C in enumerate(courant_values):
        solver = WaveEquation1D(L=L, c=c, nx=51, T=T, courant=C)

        # Initial condition: first normal mode
        solver.set_initial_conditions(
            u0_func=lambda x: np.sin(np.pi * x / L),
            v0_func=lambda x: np.zeros_like(x)
        )
        solver.set_boundary_conditions(bc_type='fixed')

        times, history = solver.solve()

        # Compare at final time
        u_exact = exact_solution(solver.x, T)
        u_numerical = history[-1]

        ax = axes[idx]
        ax.plot(solver.x, u_exact, 'b-', label='Analytical', linewidth=2)
        ax.plot(solver.x, u_numerical, 'ro', label='Numerical', markersize=4)

        error = np.max(np.abs(u_numerical - u_exact))
        ax.set_title(f'C = {C}\nMax Error: {error:.2e}')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Analytical Solution Comparison at t = {T}', fontsize=14)
    plt.tight_layout()
    plt.savefig('wave_exact_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# compare_with_exact()
```

---

## 5. 2D Wave Equation

### 5.1 2D Wave Equation

```
d²u/dt² = c² · (d²u/dx² + d²u/dy²) = c² · nabla²u
```

### 5.2 CTCS 2D Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class WaveEquation2D:
    """
    2D Wave Equation CTCS Method

    d²u/dt² = c² · (d²u/dx² + d²u/dy²)
    """

    def __init__(self, Lx=1.0, Ly=1.0, c=1.0, nx=101, ny=101, T=2.0, courant=0.5):
        """
        Parameters:
        -----------
        Lx, Ly : float - domain size
        c : float - wave speed
        nx, ny : int - number of grid points
        T : float - final time
        courant : float - Courant number (C <= 1/sqrt(2) in 2D)
        """
        self.Lx = Lx
        self.Ly = Ly
        self.c = c
        self.nx = nx
        self.ny = ny
        self.T = T

        # Grid generation
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # 2D CFL: C² <= 1/(1/dx² + 1/dy²) · dt²
        # Simplified: C_x² + C_y² <= 1, for uniform grid C <= 1/sqrt(2)
        dt_max = courant / (c * np.sqrt(1/self.dx**2 + 1/self.dy**2))
        self.dt = dt_max
        self.nt = int(np.ceil(T / self.dt))
        self.dt = T / self.nt

        self.Cx = c * self.dt / self.dx
        self.Cy = c * self.dt / self.dy
        self.Cx2 = self.Cx ** 2
        self.Cy2 = self.Cy ** 2

        print(f"2D Wave Equation CTCS Setup")
        print(f"  Grid: {nx} x {ny}")
        print(f"  Cx = {self.Cx:.4f}, Cy = {self.Cy:.4f}")
        print(f"  Cx² + Cy² = {self.Cx2 + self.Cy2:.4f} (must be <= 1 for stability)")

    def set_initial_conditions(self, u0_func, v0_func=None):
        """Set initial conditions"""
        self.u = u0_func(self.X, self.Y)
        self.u0 = self.u.copy()

        if v0_func is None:
            v0_func = lambda X, Y: np.zeros_like(X)

        # First time step
        self.u_prev = self.u.copy()

        d2u_dx2 = np.zeros_like(self.u)
        d2u_dy2 = np.zeros_like(self.u)
        d2u_dx2[:, 1:-1] = (self.u[:, 2:] - 2*self.u[:, 1:-1] + self.u[:, :-2]) / self.dx**2
        d2u_dy2[1:-1, :] = (self.u[2:, :] - 2*self.u[1:-1, :] + self.u[:-2, :]) / self.dy**2

        self.u = (self.u_prev + self.dt * v0_func(self.X, self.Y) +
                  0.5 * self.dt**2 * self.c**2 * (d2u_dx2 + d2u_dy2))

    def set_boundary_conditions(self, bc_type='fixed'):
        """Set boundary conditions"""
        self.bc_type = bc_type

    def apply_bc(self, u):
        """Apply boundary conditions"""
        if self.bc_type == 'fixed':
            u[0, :] = 0
            u[-1, :] = 0
            u[:, 0] = 0
            u[:, -1] = 0
        elif self.bc_type == 'free':
            u[0, :] = u[1, :]
            u[-1, :] = u[-2, :]
            u[:, 0] = u[:, 1]
            u[:, -1] = u[:, -2]
        return u

    def step(self):
        """Advance one time step"""
        u_new = np.zeros_like(self.u)

        # Interior point update
        u_new[1:-1, 1:-1] = (
            2*self.u[1:-1, 1:-1] - self.u_prev[1:-1, 1:-1] +
            self.Cx2 * (self.u[1:-1, 2:] - 2*self.u[1:-1, 1:-1] + self.u[1:-1, :-2]) +
            self.Cy2 * (self.u[2:, 1:-1] - 2*self.u[1:-1, 1:-1] + self.u[:-2, 1:-1])
        )

        u_new = self.apply_bc(u_new)

        self.u_prev = self.u.copy()
        self.u = u_new

    def solve(self, save_interval=None):
        """Solve over entire time interval"""
        if save_interval is None:
            save_interval = max(1, self.nt // 100)

        history = [self.u0.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), history


def demo_wave_2d():
    """2D wave equation demo"""
    solver = WaveEquation2D(Lx=1.0, Ly=1.0, c=1.0, nx=101, ny=101, T=2.0, courant=0.4)

    # Initial condition: Gaussian pulse at center
    def u0(X, Y):
        x0, y0 = 0.5, 0.5
        sigma = 0.1
        return np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

    solver.set_initial_conditions(u0)
    solver.set_boundary_conditions('fixed')

    times, history = solver.solve(save_interval=10)

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    plot_times = [0, len(times)//5, 2*len(times)//5,
                  3*len(times)//5, 4*len(times)//5, len(times)-1]

    vmax = np.max(np.abs(history[0]))

    for idx, ti in enumerate(plot_times):
        ax = axes[idx // 3, idx % 3]
        c_plot = ax.contourf(solver.X, solver.Y, history[ti],
                            levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(c_plot, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f't = {times[ti]:.3f}')
        ax.set_aspect('equal')

    plt.suptitle('2D Wave Equation (Fixed Boundary)', fontsize=14)
    plt.tight_layout()
    plt.savefig('wave_2d.png', dpi=150, bbox_inches='tight')
    plt.show()

    return solver, times, history

# solver, times, history = demo_wave_2d()
```

---

## 6. Animation Visualization

### 6.1 1D Wave Animation

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_wave_animation_1d():
    """Create 1D wave animation"""
    # Simulation
    solver = WaveEquation1D(L=1.0, c=1.0, nx=201, T=4.0, courant=0.95)

    def u0(x):
        # Two Gaussian pulses
        return (np.exp(-(x - 0.3)**2 / 0.01) +
                0.5 * np.exp(-(x - 0.7)**2 / 0.01))

    solver.set_initial_conditions(u0)
    solver.set_boundary_conditions('fixed')

    times, history = solver.solve(save_interval=2)

    # Create animation
    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot([], [], 'b-', linewidth=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    ax.set_xlim(0, solver.L)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('1D Wave Equation (Fixed Boundary)')
    ax.grid(True, alpha=0.3)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(frame):
        line.set_data(solver.x, history[frame])
        time_text.set_text(f't = {times[frame]:.3f}')
        return line, time_text

    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(times), interval=30, blit=True)

    # Save as GIF
    # anim.save('wave_1d_animation.gif', writer='pillow', fps=30)
    plt.show()

    return anim

# anim = create_wave_animation_1d()
```

### 6.2 2D Wave Animation

```python
def create_wave_animation_2d():
    """Create 2D wave animation"""
    solver = WaveEquation2D(Lx=1.0, Ly=1.0, c=1.0, nx=81, ny=81, T=2.0, courant=0.4)

    def u0(X, Y):
        # Initial pulse offset from center
        x0, y0 = 0.3, 0.3
        sigma = 0.08
        return np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

    solver.set_initial_conditions(u0)
    solver.set_boundary_conditions('fixed')

    times, history = solver.solve(save_interval=5)

    # Animation
    fig, ax = plt.subplots(figsize=(8, 8))
    vmax = np.max(np.abs(history[0]))

    c_plot = ax.contourf(solver.X, solver.Y, history[0],
                        levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(c_plot, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    title = ax.set_title(f't = 0.000')

    def animate(frame):
        ax.clear()
        c_plot = ax.contourf(solver.X, solver.Y, history[frame],
                            levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f't = {times[frame]:.3f}')
        ax.set_aspect('equal')
        return c_plot.collections

    anim = FuncAnimation(fig, animate, frames=len(times), interval=50)

    # Save as GIF
    # anim.save('wave_2d_animation.gif', writer='pillow', fps=20)
    plt.show()

    return anim

# anim = create_wave_animation_2d()
```

---

## 7. Numerical Dispersion Analysis

Numerical dispersion is a purely artificial effect where different wavelengths travel at different speeds on the discrete grid, even though the continuous wave equation is non-dispersive ($\omega = ck$ for all $k$). Short wavelengths (comparable to the grid spacing) suffer the most distortion. Understanding this effect is critical for choosing grid resolution: you need enough points per wavelength to keep dispersion errors acceptable.

### 7.1 Dispersion Relation

The CTCS numerical dispersion relation replaces the exact $\omega = ck$ with:

$$\sin\!\left(\frac{\omega \Delta t}{2}\right) = C \cdot \sin\!\left(\frac{k \Delta x}{2}\right)$$

For small $k\Delta x$ (long wavelengths), $\sin \approx$ argument and the exact relation is recovered. For large $k\Delta x$ (short wavelengths), the numerical phase velocity is less than $c$, causing trailing artifacts.

```python
def dispersion_analysis():
    """Numerical dispersion analysis"""
    # Continuous dispersion relation: omega = c·k
    # CTCS dispersion relation: sin(omega·dt/2) = C·sin(k·dx/2)

    k_dx = np.linspace(0.01, np.pi, 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Dispersion for various Courant numbers
    ax1 = axes[0]
    courant_values = [0.5, 0.8, 0.95, 1.0]

    for C in courant_values:
        # Calculate omega from numerical dispersion relation
        # omega_num · dt/2 = arcsin(C · sin(k·dx/2))
        arg = C * np.sin(k_dx / 2)
        arg = np.clip(arg, -1, 1)  # Keep within arcsin range
        omega_num = 2 * np.arcsin(arg)

        # Normalized phase velocity
        c_phase = omega_num / k_dx

        ax1.plot(k_dx, c_phase / C, label=f'C = {C}', linewidth=2)

    ax1.axhline(y=1, color='r', linestyle='--', label='Exact (c_num/c = 1)')
    ax1.set_xlabel('k*dx')
    ax1.set_ylabel('c_numerical / c_exact')
    ax1.set_title('Numerical Phase Velocity (Dispersion)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, np.pi)
    ax1.set_ylim(0.5, 1.1)

    # Group velocity
    ax2 = axes[1]
    for C in courant_values:
        # Group velocity = d(omega)/dk
        arg = C * np.sin(k_dx / 2)
        arg = np.clip(arg, -1, 1)

        # Numerical differentiation
        omega_num = 2 * np.arcsin(arg)
        d_omega = np.gradient(omega_num, k_dx[1] - k_dx[0])

        ax2.plot(k_dx, d_omega / C, label=f'C = {C}', linewidth=2)

    ax2.axhline(y=1, color='r', linestyle='--', label='Exact')
    ax2.set_xlabel('k*dx')
    ax2.set_ylabel('c_group / c_exact')
    ax2.set_title('Numerical Group Velocity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, np.pi)
    ax2.set_ylim(0, 1.5)

    plt.tight_layout()
    plt.savefig('wave_dispersion.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Observations:")
    print("1. Numerical dispersion is minimized when C = 1.0")
    print("2. All C values are accurate for small k (long wavelength)")
    print("3. Dispersion error increases for large k (short wavelength)")

# dispersion_analysis()
```

---

## 8. Application Examples

### 8.1 Guitar String Vibration

```python
def guitar_string_simulation():
    """Guitar string vibration simulation"""
    L = 0.65  # Guitar string length [m]
    T = 73.0  # Tension [N]
    mu = 3.75e-4  # Linear density [kg/m]

    c = np.sqrt(T / mu)  # Wave speed
    f1 = c / (2 * L)  # Fundamental frequency

    print(f"Guitar string parameters:")
    print(f"  Length: {L} m")
    print(f"  Tension: {T} N")
    print(f"  Linear density: {mu} kg/m")
    print(f"  Wave speed: {c:.1f} m/s")
    print(f"  Fundamental frequency: {f1:.1f} Hz (Note: {freq_to_note(f1)})")

    # Simulation
    solver = WaveEquation1D(L=L, c=c, nx=201, T=0.01, courant=0.9)

    # Initial condition: plucked position (triangular)
    pluck_position = 0.2  # Relative position from L

    def u0(x):
        peak = L * pluck_position
        amplitude = 0.005  # 5mm
        return np.where(x < peak,
                       amplitude * x / peak,
                       amplitude * (L - x) / (L - peak))

    solver.set_initial_conditions(u0)
    solver.set_boundary_conditions('fixed')

    times, history = solver.solve(save_interval=2)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # String shape at multiple times
    ax1 = axes[0, 0]
    for i in range(0, len(times), len(times)//6):
        ax1.plot(solver.x * 1000, history[i] * 1000, label=f't = {times[i]*1000:.2f} ms')
    ax1.set_xlabel('Position (mm)')
    ax1.set_ylabel('Displacement (mm)')
    ax1.set_title('String Displacement')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Time evolution at specific position
    ax2 = axes[0, 1]
    monitor_position = int(0.25 * solver.nx)
    displacements = [h[monitor_position] for h in history]
    ax2.plot(np.array(times) * 1000, np.array(displacements) * 1000, 'b-')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Displacement (mm)')
    ax2.set_title(f'Vibration at x = {solver.x[monitor_position]*1000:.1f} mm')
    ax2.grid(True, alpha=0.3)

    # Frequency spectrum
    ax3 = axes[1, 0]
    from scipy.fft import fft, fftfreq

    signal = np.array(displacements)
    n = len(signal)
    dt = times[1] - times[0]

    freqs = fftfreq(n, dt)
    spectrum = np.abs(fft(signal))

    positive_mask = freqs > 0
    ax3.plot(freqs[positive_mask], spectrum[positive_mask])
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Frequency Spectrum')
    ax3.set_xlim(0, 5000)
    ax3.grid(True, alpha=0.3)

    # Mark expected harmonics
    for n in range(1, 6):
        f_n = n * f1
        if f_n < 5000:
            ax3.axvline(x=f_n, color='r', linestyle='--', alpha=0.5)
            ax3.text(f_n, ax3.get_ylim()[1]*0.9, f'{n}f1', rotation=90)

    # Space-time plot
    ax4 = axes[1, 1]
    X, T_grid = np.meshgrid(solver.x * 1000, np.array(times) * 1000)
    c_plot = ax4.contourf(X, T_grid, np.array(history) * 1000,
                         levels=30, cmap='RdBu_r')
    plt.colorbar(c_plot, ax=ax4, label='Displacement (mm)')
    ax4.set_xlabel('Position (mm)')
    ax4.set_ylabel('Time (ms)')
    ax4.set_title('Space-Time Displacement')

    plt.tight_layout()
    plt.savefig('guitar_string.png', dpi=150, bbox_inches='tight')
    plt.show()


def freq_to_note(freq):
    """Convert frequency to note name"""
    A4 = 440.0
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

    if freq <= 0:
        return "N/A"

    semitones = 12 * np.log2(freq / A4)
    note_idx = int(round(semitones)) % 12
    octave = 4 + int(round(semitones + 9) // 12)

    return f"{notes[note_idx]}{octave}"

# guitar_string_simulation()
```

### 8.2 Drum Head Vibration (2D)

```python
def drum_head_simulation():
    """Drum head vibration simulation (circular membrane)"""
    # Use circular mask on square domain
    R = 0.15  # Radius [m]
    c = 100.0  # Wave speed [m/s]

    solver = WaveEquation2D(Lx=2*R, Ly=2*R, c=c, nx=101, ny=101, T=0.02, courant=0.4)

    # Create circular mask
    center_x, center_y = R, R
    distance = np.sqrt((solver.X - center_x)**2 + (solver.Y - center_y)**2)
    mask = distance <= R

    # Initial condition: center push
    def u0(X, Y):
        r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        u = np.exp(-(r**2) / (0.03**2)) * 0.005  # 5mm amplitude
        u[~mask] = 0
        return u

    solver.set_initial_conditions(u0)

    # Custom boundary condition: fixed at circular boundary
    def circular_bc(u):
        u[~mask] = 0
        # Also set near circular boundary to 0
        boundary = (distance > R * 0.95) & (distance <= R)
        u[boundary] = 0
        return u

    # Solve (modify boundary condition)
    original_apply_bc = solver.apply_bc
    solver.apply_bc = circular_bc

    times, history = solver.solve(save_interval=5)

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    plot_times = [0, len(times)//5, 2*len(times)//5,
                  3*len(times)//5, 4*len(times)//5, len(times)-1]

    vmax = np.max(np.abs(history[0]))

    for idx, ti in enumerate(plot_times):
        ax = axes[idx // 3, idx % 3]

        # Data with circular mask applied
        data = history[ti].copy()
        data[~mask] = np.nan

        c_plot = ax.contourf(solver.X * 1000, solver.Y * 1000, data * 1000,
                            levels=30, cmap='RdBu_r', vmin=-vmax*1000, vmax=vmax*1000)

        # Circle boundary
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(center_x*1000 + R*1000*np.cos(theta),
               center_y*1000 + R*1000*np.sin(theta), 'k-', linewidth=2)

        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(f't = {times[ti]*1000:.2f} ms')
        ax.set_aspect('equal')

    plt.suptitle('Drum Head Vibration (Circular Membrane)', fontsize=14)
    plt.tight_layout()
    plt.savefig('drum_head.png', dpi=150, bbox_inches='tight')
    plt.show()

# drum_head_simulation()
```

---

## 9. Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| Wave equation | d²u/dt² = c²·nabla²u, hyperbolic PDE |
| CTCS | Central difference in both time/space, 2nd order accuracy |
| Courant number | C = c·dt/dx, stability: C <= 1 |
| Fixed boundary | Wave reflects with phase inversion |
| Free boundary | Wave reflects with same phase |
| Absorbing boundary | Wave transmits (no reflection) |

### CFL Condition

```
1D: C = c·dt/dx <= 1
2D: Cx² + Cy² <= 1 (uniform grid: C <= 1/sqrt(2))
```

### Next Steps

1. **Chapter 11**: Laplace/Poisson Equation (elliptic)
2. **Chapter 12**: Advection Equation (first-order hyperbolic)

---

## Exercises

### Exercise 1: Courant Number Experiment
Verify numerical solution stability for C = 0.5, 0.8, 1.0, 1.1.

### Exercise 2: Standing Wave Mode
Starting from initial condition u(x,0) = sin(2*pi*x/L), verify the second standing wave mode.

### Exercise 3: Improved Absorbing Boundary
Implement 2nd order Sommerfeld absorbing boundary condition and compare with 1st order.

### Exercise 4: Circular Membrane Normal Mode
Compare the first normal mode of a circular membrane (Bessel function) with numerical solution.

---

## References

1. **Textbook**: LeVeque, "Finite Difference Methods"
2. **Physics**: Morse & Ingard, "Theoretical Acoustics"
3. **Numerical Dispersion**: Trefethen, "Finite Difference and Spectral Methods"
