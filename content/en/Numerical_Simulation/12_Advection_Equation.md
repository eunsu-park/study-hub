# 12. Advection Equation

## Learning Objectives
- Understand the advection equation as a first-order hyperbolic PDE
- Implement the Upwind Scheme
- Analyze FTCS instability
- Learn Lax-Friedrichs and Lax-Wendroff methods
- Understand numerical dispersion and numerical diffusion
- Grasp the importance of the Courant number

---

## 1. Advection Equation Theory

### 1.1 Definition and Physical Meaning

```
1D Linear Advection Equation:
du/dt + c · du/dx = 0

where:
- u(x,t): advected quantity (concentration, temperature, etc.)
- c: advection velocity (positive means moving right)
```

### 1.2 Analytical Solution

The solution to the advection equation is the initial profile moving at velocity c:

```
u(x, t) = u_0(x - ct)

where u_0(x) is the initial condition
```

```python
import numpy as np
import matplotlib.pyplot as plt

def exact_advection():
    """Advection equation analytical solution visualization"""
    c = 1.0  # advection velocity
    L = 4.0
    x = np.linspace(0, L, 500)

    # Initial condition: Gaussian pulse
    def u0(x):
        x0 = 1.0
        sigma = 0.2
        return np.exp(-(x - x0)**2 / (2 * sigma**2))

    fig, ax = plt.subplots(figsize=(12, 5))

    times = [0, 0.5, 1.0, 1.5, 2.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))

    for t, color in zip(times, colors):
        u = u0(x - c * t)  # analytical solution
        ax.plot(x, u, color=color, linewidth=2, label=f't = {t}')

    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title(f'Advection Equation Analytical Solution: u(x,t) = u_0(x - ct), c = {c}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L)

    # Direction indicator
    ax.annotate('', xy=(3, 0.8), xytext=(2, 0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(2.5, 0.85, f'c = {c}', color='red', ha='center')

    plt.tight_layout()
    plt.savefig('advection_exact.png', dpi=150, bbox_inches='tight')
    plt.show()

# exact_advection()
```

### 1.3 Characteristic Lines

The characteristic lines of the advection equation are straight lines x - ct = const.

```python
def characteristic_lines():
    """Characteristic lines visualization"""
    c = 1.0
    L = 2.0
    T = 2.0

    fig, ax = plt.subplots(figsize=(10, 6))

    # Characteristic lines (x - ct = const)
    for x0 in np.linspace(0, L, 11):
        t = np.linspace(0, T, 100)
        x = x0 + c * t
        ax.plot(x, t, 'b-', alpha=0.7)

    # Domain boundaries
    ax.axvline(x=0, color='k', linestyle='-', linewidth=2)
    ax.axvline(x=L, color='k', linestyle='-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=2)
    ax.axhline(y=T, color='k', linestyle='--', linewidth=1)

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title(f'Characteristic Lines: dx/dt = c = {c}')
    ax.set_xlim(-0.5, L + 1)
    ax.set_ylim(-0.1, T + 0.1)
    ax.grid(True, alpha=0.3)

    # Direction indicator
    ax.annotate('', xy=(1.5, 1.5), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(1.2, 0.8, 'Information propagation', color='red', fontsize=12)

    plt.tight_layout()
    plt.savefig('characteristic_lines.png', dpi=150, bbox_inches='tight')
    plt.show()

# characteristic_lines()
```

---

## 2. FTCS Instability

### 2.1 FTCS Scheme

```
(u_i^{n+1} - u_i^n) / dt + c · (u_{i+1}^n - u_{i-1}^n) / (2*dx) = 0

Rearranged:
u_i^{n+1} = u_i^n - (C/2) · (u_{i+1}^n - u_{i-1}^n)

where C = c·dt/dx (Courant number)
```

### 2.2 Cause of Instability

von Neumann analysis:
```
Amplification factor G = 1 - i*C·sin(k*dx)
|G|² = 1 + C²·sin²(k*dx) > 1  (always!)

-> FTCS is unconditionally unstable for the advection equation!
```

```python
class AdvectionFTCS:
    """
    Advection equation FTCS (unstable)

    Warning: For educational purposes only. Actually unstable!
    """

    def __init__(self, L=4.0, c=1.0, nx=101, T=1.0, courant=0.5):
        self.L = L
        self.c = c
        self.nx = nx
        self.T = T

        self.dx = L / (nx - 1)
        self.dt = courant * self.dx / abs(c)
        self.nt = int(np.ceil(T / self.dt))
        self.dt = T / self.nt

        self.x = np.linspace(0, L, nx)
        self.C = c * self.dt / self.dx

        print(f"FTCS Advection Equation (unstable)")
        print(f"  C = {self.C:.4f}")

    def set_initial_condition(self, func):
        self.u = func(self.x)
        self.u0 = self.u.copy()

    def step(self):
        """FTCS step (unstable)"""
        u_new = self.u.copy()

        # Interior points
        u_new[1:-1] = self.u[1:-1] - (self.C / 2) * (self.u[2:] - self.u[:-2])

        # Periodic boundary conditions
        u_new[0] = self.u[0] - (self.C / 2) * (self.u[1] - self.u[-2])
        u_new[-1] = u_new[0]

        self.u = u_new

    def solve(self, save_interval=1):
        history = [self.u0.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), np.array(history)


def demo_ftcs_instability():
    """FTCS instability demonstration"""
    solver = AdvectionFTCS(L=4.0, c=1.0, nx=101, T=1.0, courant=0.5)

    def u0(x):
        return np.exp(-(x - 1.0)**2 / 0.08)

    solver.set_initial_condition(u0)
    times, history = solver.solve(save_interval=5)

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    time_indices = [0, 5, 10, 15, 20, min(25, len(times)-1)]

    for idx, ti in enumerate(time_indices):
        ax = axes[idx // 3, idx % 3]

        # Numerical solution
        ax.plot(solver.x, history[ti], 'b-', label='Numerical', linewidth=2)

        # Analytical solution
        u_exact = u0(solver.x - solver.c * times[ti])
        ax.plot(solver.x, u_exact, 'r--', label='Analytical', linewidth=2)

        ax.set_xlim(0, solver.L)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title(f't = {times[ti]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('FTCS Advection Equation: Unconditionally Unstable!', fontsize=14, color='red')
    plt.tight_layout()
    plt.savefig('advection_ftcs_unstable.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nMaximum value at final time: {np.max(np.abs(history[-1])):.2e}")
    print("-> Numerical solution grows explosively!")

# demo_ftcs_instability()
```

---

## 3. Upwind Scheme

### 3.1 Upwind Principle

Approximate spatial derivative from the upwind direction (where information comes from):

```
c > 0 (moving right):
du/dx ≈ (u_i - u_{i-1}) / dx  (backward difference)

c < 0 (moving left):
du/dx ≈ (u_{i+1} - u_i) / dx  (forward difference)
```

### Why Upwind? Following the Physics

The advection equation describes quantities carried by a flow at speed c. **Information travels in the direction of the flow** — a wave moving rightward depends only on what lies upstream (to the left). The upwind scheme respects this by using a one-sided difference that looks in the upstream direction:

- If c > 0 (rightward flow): use backward difference (u_i - u_{i-1})/Δx
- If c < 0 (leftward flow): use forward difference (u_{i+1} - u_i)/Δx

This is why central differences (FTCS) fail for pure advection: they look equally in both directions and don't respect the flow's directionality, leading to unconditional instability. The amplification factor for FTCS on advection is |G|² = 1 + C²·sin²(kΔx) > 1 for all wavenumbers — the scheme *always* amplifies every mode.

### 3.2 Scheme

```
c > 0:
u_i^{n+1} = u_i^n - C · (u_i^n - u_{i-1}^n)
          = (1-C)·u_i^n + C·u_{i-1}^n

Stability condition: 0 <= C <= 1
```

```python
class AdvectionUpwind:
    """
    Advection equation Upwind scheme

    Conditionally stable: 0 <= C <= 1
    First-order accuracy
    """

    def __init__(self, L=4.0, c=1.0, nx=101, T=1.0, courant=0.8):
        self.L = L
        self.c = c
        self.nx = nx
        self.T = T

        self.dx = L / (nx - 1)
        self.dt = courant * self.dx / abs(c)
        self.nt = int(np.ceil(T / self.dt))
        self.dt = T / self.nt

        self.x = np.linspace(0, L, nx)
        self.C = c * self.dt / self.dx

        print(f"Upwind Advection Equation")
        print(f"  C = {self.C:.4f}")
        print(f"  Stability: {'OK' if 0 <= self.C <= 1 else 'WARNING!'}")

    def set_initial_condition(self, func):
        self.u = func(self.x)
        self.u0 = self.u.copy()

    def step(self):
        """Upwind step"""
        u_new = self.u.copy()

        if self.c > 0:
            # Backward difference (information comes from left)
            u_new[1:] = self.u[1:] - self.C * (self.u[1:] - self.u[:-1])
            # Left boundary: inflow condition (value from outside)
            u_new[0] = self.u0[0]  # or specific value
        else:
            # Forward difference (information comes from right)
            u_new[:-1] = self.u[:-1] - self.C * (self.u[1:] - self.u[:-1])
            # Right boundary
            u_new[-1] = self.u0[-1]

        self.u = u_new

    def solve(self, save_interval=1):
        history = [self.u0.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), np.array(history)


def demo_upwind():
    """Upwind scheme demo"""
    L = 4.0
    c = 1.0
    T = 2.0

    def u0(x):
        return np.exp(-(x - 1.0)**2 / 0.08)

    # Compare various Courant numbers
    courant_values = [0.5, 0.8, 0.95]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, C in enumerate(courant_values):
        solver = AdvectionUpwind(L=L, c=c, nx=101, T=T, courant=C)
        solver.set_initial_condition(u0)
        times, history = solver.solve()

        ax = axes[idx]

        # Initial, final
        ax.plot(solver.x, u0(solver.x), 'k--', label='Initial', alpha=0.5)
        ax.plot(solver.x, history[-1], 'b-', label='Numerical', linewidth=2)
        ax.plot(solver.x, u0(solver.x - c * T), 'r--', label='Analytical', linewidth=2)

        error = np.max(np.abs(history[-1] - u0(solver.x - c * T)))
        ax.set_title(f'C = {C}\nMax Error: {error:.4f}')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, L)

    plt.suptitle('Upwind Scheme: Accuracy vs Courant Number', fontsize=12)
    plt.tight_layout()
    plt.savefig('advection_upwind.png', dpi=150, bbox_inches='tight')
    plt.show()

# demo_upwind()
```

### 3.3 Numerical Diffusion

The upwind scheme is stable but introduces numerical diffusion.

```
Modified equation:
du/dt + c·du/dx = (c·dx/2)·(1 - C)·d²u/dx²

-> Artificial diffusion coefficient: D_num = (c·dx/2)·(1 - C)
-> No diffusion when C = 1 (exact solution)
```

```python
def numerical_diffusion_demo():
    """Numerical diffusion demonstration"""
    L = 4.0
    c = 1.0
    T = 2.0

    def u0(x):
        # Discontinuous initial condition (step function)
        return np.where((x > 0.5) & (x < 1.5), 1.0, 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    courant_values = [0.3, 0.6, 0.95]

    for idx, C in enumerate(courant_values):
        solver = AdvectionUpwind(L=L, c=c, nx=201, T=T, courant=C)
        solver.set_initial_condition(u0)
        times, history = solver.solve()

        ax = axes[idx]

        ax.plot(solver.x, u0(solver.x), 'k--', label='Initial', alpha=0.5)
        ax.plot(solver.x, history[-1], 'b-', label='Numerical', linewidth=2)
        ax.plot(solver.x, u0(solver.x - c * T), 'r--', label='Analytical', linewidth=2)

        # Artificial diffusion coefficient
        D_num = (c * solver.dx / 2) * (1 - C)
        ax.set_title(f'C = {C}\nNumerical Diffusion: D = {D_num:.4f}')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, L)
        ax.set_ylim(-0.2, 1.4)

    plt.suptitle('Numerical Diffusion of Upwind Scheme (Discontinuous Initial Condition)', fontsize=12)
    plt.tight_layout()
    plt.savefig('numerical_diffusion.png', dpi=150, bbox_inches='tight')
    plt.show()

# numerical_diffusion_demo()
```

---

## 4. Lax-Friedrichs Scheme

### 4.1 Scheme

Stabilized modification of FTCS:

```
u_i^{n+1} = (1/2)·(u_{i+1}^n + u_{i-1}^n) - (C/2)·(u_{i+1}^n - u_{i-1}^n)

= (1/2)·(1+C)·u_{i-1}^n + (1/2)·(1-C)·u_{i+1}^n
```

### 4.2 Implementation

```python
class AdvectionLaxFriedrichs:
    """
    Advection equation Lax-Friedrichs scheme

    Conditionally stable: |C| <= 1
    First-order accuracy
    """

    def __init__(self, L=4.0, c=1.0, nx=101, T=1.0, courant=0.8):
        self.L = L
        self.c = c
        self.nx = nx
        self.T = T

        self.dx = L / (nx - 1)
        self.dt = courant * self.dx / abs(c)
        self.nt = int(np.ceil(T / self.dt))
        self.dt = T / self.nt

        self.x = np.linspace(0, L, nx)
        self.C = c * self.dt / self.dx

        print(f"Lax-Friedrichs Advection Equation")
        print(f"  C = {self.C:.4f}")

    def set_initial_condition(self, func):
        self.u = func(self.x)
        self.u0 = self.u.copy()

    def step(self):
        """Lax-Friedrichs step"""
        u_new = np.zeros_like(self.u)

        # Interior points
        u_new[1:-1] = (0.5 * (self.u[2:] + self.u[:-2]) -
                       (self.C / 2) * (self.u[2:] - self.u[:-2]))

        # Boundaries: extrapolation or fixed
        u_new[0] = self.u[0]
        u_new[-1] = self.u[-1]

        self.u = u_new

    def solve(self, save_interval=1):
        history = [self.u0.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), np.array(history)
```

---

## 5. Lax-Wendroff Scheme

### 5.1 Theory

Use Taylor expansion for second-order accuracy:

```
u^{n+1} = u^n + dt·du/dt + (dt²/2)·d²u/dt²

For advection equation:
du/dt = -c·du/dx
d²u/dt² = c²·d²u/dx²

Result:
u_i^{n+1} = u_i^n - (C/2)·(u_{i+1}^n - u_{i-1}^n)
          + (C²/2)·(u_{i+1}^n - 2u_i^n + u_{i-1}^n)
```

### 5.2 Implementation

```python
class AdvectionLaxWendroff:
    """
    Advection equation Lax-Wendroff scheme

    Conditionally stable: |C| <= 1
    Second-order accuracy
    """

    def __init__(self, L=4.0, c=1.0, nx=101, T=1.0, courant=0.8):
        self.L = L
        self.c = c
        self.nx = nx
        self.T = T

        self.dx = L / (nx - 1)
        self.dt = courant * self.dx / abs(c)
        self.nt = int(np.ceil(T / self.dt))
        self.dt = T / self.nt

        self.x = np.linspace(0, L, nx)
        self.C = c * self.dt / self.dx
        self.C2 = self.C ** 2

        print(f"Lax-Wendroff Advection Equation")
        print(f"  C = {self.C:.4f}")

    def set_initial_condition(self, func):
        self.u = func(self.x)
        self.u0 = self.u.copy()

    def step(self):
        """Lax-Wendroff step"""
        u_new = np.zeros_like(self.u)

        # Interior points
        u_new[1:-1] = (self.u[1:-1]
                       - (self.C / 2) * (self.u[2:] - self.u[:-2])
                       + (self.C2 / 2) * (self.u[2:] - 2*self.u[1:-1] + self.u[:-2]))

        # Boundaries
        u_new[0] = self.u[0]
        u_new[-1] = self.u[-1]

        self.u = u_new

    def solve(self, save_interval=1):
        history = [self.u0.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), np.array(history)


def compare_all_schemes():
    """Compare all schemes"""
    L = 4.0
    c = 1.0
    T = 2.0
    C = 0.8

    def u0(x):
        return np.exp(-(x - 1.0)**2 / 0.08)

    schemes = {
        'Upwind': AdvectionUpwind(L, c, nx=101, T=T, courant=C),
        'Lax-Friedrichs': AdvectionLaxFriedrichs(L, c, nx=101, T=T, courant=C),
        'Lax-Wendroff': AdvectionLaxWendroff(L, c, nx=101, T=T, courant=C),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (name, solver) in enumerate(schemes.items()):
        solver.set_initial_condition(u0)
        times, history = solver.solve()

        ax = axes[idx]

        ax.plot(solver.x, u0(solver.x), 'k--', alpha=0.5, label='Initial')
        ax.plot(solver.x, history[-1], 'b-', linewidth=2, label='Numerical')
        ax.plot(solver.x, u0(solver.x - c * T), 'r--', linewidth=2, label='Analytical')

        error = np.max(np.abs(history[-1] - u0(solver.x - c * T)))
        ax.set_title(f'{name}\nMax Error: {error:.4f}')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, L)

    plt.suptitle(f'Advection Scheme Comparison (C = {C})', fontsize=12)
    plt.tight_layout()
    plt.savefig('advection_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# compare_all_schemes()
```

---

## 6. Numerical Dispersion Analysis

### 6.1 Dispersion Relation

```python
def dispersion_analysis():
    """Numerical dispersion relation analysis"""
    k_dx = np.linspace(0.01, np.pi, 200)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    C = 0.8

    # Exact dispersion relation: omega = c·k
    # Normalized: omega·dt = C·k·dx
    omega_exact = C * k_dx

    # Calculate omega from amplification factor for each scheme

    # Upwind: G = 1 - C + C·exp(-i·k·dx)
    G_upwind = 1 - C + C * np.exp(-1j * k_dx)
    omega_upwind = -np.angle(G_upwind)

    # Lax-Friedrichs: G = cos(k·dx) - i·C·sin(k·dx)
    G_lf = np.cos(k_dx) - 1j * C * np.sin(k_dx)
    omega_lf = -np.angle(G_lf)

    # Lax-Wendroff: G = 1 - i·C·sin(k·dx) - C²·(1 - cos(k·dx))
    G_lw = 1 - 1j * C * np.sin(k_dx) - C**2 * (1 - np.cos(k_dx))
    omega_lw = -np.angle(G_lw)

    # Phase velocity comparison
    ax1 = axes[0]
    ax1.plot(k_dx, omega_exact / k_dx / C, 'k-', linewidth=2, label='Exact')
    ax1.plot(k_dx, omega_upwind / k_dx / C, 'b-', linewidth=2, label='Upwind')
    ax1.plot(k_dx, omega_lf / k_dx / C, 'g-', linewidth=2, label='Lax-Friedrichs')
    ax1.plot(k_dx, omega_lw / k_dx / C, 'r-', linewidth=2, label='Lax-Wendroff')

    ax1.set_xlabel('k*dx')
    ax1.set_ylabel('c_num / c')
    ax1.set_title(f'Phase Velocity Error (C = {C})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, np.pi)
    ax1.set_ylim(0, 1.2)

    # Amplitude damping (amplification factor magnitude)
    ax2 = axes[1]
    ax2.axhline(y=1, color='k', linestyle='-', linewidth=2, label='Exact')
    ax2.plot(k_dx, np.abs(G_upwind), 'b-', linewidth=2, label='Upwind')
    ax2.plot(k_dx, np.abs(G_lf), 'g-', linewidth=2, label='Lax-Friedrichs')
    ax2.plot(k_dx, np.abs(G_lw), 'r-', linewidth=2, label='Lax-Wendroff')

    ax2.set_xlabel('k*dx')
    ax2.set_ylabel('|G|')
    ax2.set_title(f'Amplification Factor Magnitude (Amplitude Damping)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, np.pi)
    ax2.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig('advection_dispersion.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Observations:")
    print("1. Upwind, Lax-Friedrichs: Large damping (numerical diffusion)")
    print("2. Lax-Wendroff: Small damping, phase error (numerical dispersion)")
    print("3. All schemes: Error increases for short wavelengths (large k)")

# dispersion_analysis()
```

### 6.2 Dispersion vs Diffusion Effects

```python
def dispersion_diffusion_demo():
    """Numerical dispersion vs numerical diffusion visualization"""
    L = 8.0
    c = 1.0
    T = 4.0
    C = 0.8

    # Initial condition: square pulse (discontinuous)
    def u0_square(x):
        return np.where((x > 1.0) & (x < 2.0), 1.0, 0.0)

    # Initial condition: Gaussian (smooth)
    def u0_gauss(x):
        return np.exp(-(x - 1.5)**2 / 0.1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for row, (u0, name) in enumerate([(u0_square, 'Square Pulse'), (u0_gauss, 'Gaussian')]):
        # Upwind (diffusion dominant)
        solver1 = AdvectionUpwind(L, c, nx=201, T=T, courant=C)
        solver1.set_initial_condition(u0)
        _, hist1 = solver1.solve()

        ax1 = axes[row, 0]
        ax1.plot(solver1.x, u0(solver1.x), 'k--', alpha=0.5, label='Initial')
        ax1.plot(solver1.x, hist1[-1], 'b-', linewidth=2, label='Numerical')
        ax1.plot(solver1.x, u0(solver1.x - c * T), 'r--', linewidth=2, label='Analytical')
        ax1.set_title(f'Upwind ({name})\nNumerical Diffusion')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, L)
        ax1.set_ylim(-0.3, 1.3)

        # Lax-Wendroff (dispersion dominant)
        solver2 = AdvectionLaxWendroff(L, c, nx=201, T=T, courant=C)
        solver2.set_initial_condition(u0)
        _, hist2 = solver2.solve()

        ax2 = axes[row, 1]
        ax2.plot(solver2.x, u0(solver2.x), 'k--', alpha=0.5, label='Initial')
        ax2.plot(solver2.x, hist2[-1], 'b-', linewidth=2, label='Numerical')
        ax2.plot(solver2.x, u0(solver2.x - c * T), 'r--', linewidth=2, label='Analytical')
        ax2.set_title(f'Lax-Wendroff ({name})\nNumerical Dispersion (Oscillations)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, L)
        ax2.set_ylim(-0.3, 1.3)

        # Error comparison
        ax3 = axes[row, 2]
        exact = u0(solver1.x - c * T)
        ax3.plot(solver1.x, hist1[-1] - exact, 'b-', label='Upwind Error')
        ax3.plot(solver2.x, hist2[-1] - exact, 'r-', label='Lax-Wendroff Error')
        ax3.set_title(f'Error Comparison ({name})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, L)
        ax3.set_ylim(-0.5, 0.5)

        for ax in axes[row]:
            ax.set_xlabel('x')
            ax.set_ylabel('u')

    plt.tight_layout()
    plt.savefig('dispersion_vs_diffusion.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Key observations:")
    print("- Upwind: Smooth but spreads out (diffusion)")
    print("- Lax-Wendroff: Sharp but oscillates (dispersion)")
    print("- Both effects are problematic for discontinuous solutions")

# dispersion_diffusion_demo()
```

---

## 7. Importance of Courant Number

### 7.1 Stability and Accuracy

```python
def courant_number_study():
    """Stability and accuracy vs Courant number"""
    L = 4.0
    c = 1.0
    T = 1.0

    def u0(x):
        return np.exp(-(x - 1.0)**2 / 0.08)

    courant_values = [0.2, 0.5, 0.8, 0.95, 1.0, 1.05]
    errors = {'Upwind': [], 'Lax-Wendroff': []}
    stable = {'Upwind': [], 'Lax-Wendroff': []}

    for C in courant_values:
        try:
            # Upwind
            solver1 = AdvectionUpwind(L, c, nx=101, T=T, courant=C)
            solver1.set_initial_condition(u0)
            _, hist1 = solver1.solve()
            err1 = np.max(np.abs(hist1[-1] - u0(solver1.x - c * T)))
            errors['Upwind'].append(err1)
            stable['Upwind'].append(err1 < 10)

            # Lax-Wendroff
            solver2 = AdvectionLaxWendroff(L, c, nx=101, T=T, courant=C)
            solver2.set_initial_condition(u0)
            _, hist2 = solver2.solve()
            err2 = np.max(np.abs(hist2[-1] - u0(solver2.x - c * T)))
            errors['Lax-Wendroff'].append(err2)
            stable['Lax-Wendroff'].append(err2 < 10)

        except:
            errors['Upwind'].append(np.nan)
            errors['Lax-Wendroff'].append(np.nan)
            stable['Upwind'].append(False)
            stable['Lax-Wendroff'].append(False)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(courant_values))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, errors['Upwind'], width, label='Upwind', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, errors['Lax-Wendroff'], width, label='Lax-Wendroff', alpha=0.8)

    # Mark unstable
    for i, C in enumerate(courant_values):
        if not stable['Upwind'][i]:
            bars1[i].set_color('red')
        if not stable['Lax-Wendroff'][i]:
            bars2[i].set_color('red')

    ax.set_xlabel('Courant Number C')
    ax.set_ylabel('Maximum Error')
    ax.set_title('Accuracy vs Courant Number (Red = Unstable)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(courant_values)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Mark stable region
    ax.axvline(x=4.5, color='red', linestyle='--', alpha=0.5)
    ax.text(4.7, ax.get_ylim()[1]*0.9, 'C > 1\nUnstable', color='red', fontsize=10)

    plt.tight_layout()
    plt.savefig('courant_study.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nResults summary:")
    for C, e1, e2 in zip(courant_values, errors['Upwind'], errors['Lax-Wendroff']):
        status = 'UNSTABLE' if C > 1 else ''
        print(f"C = {C}: Upwind = {e1:.4f}, L-W = {e2:.4f} {status}")

# courant_number_study()
```

---

## 8. Comprehensive Example: Pollutant Transport

```python
def pollution_transport():
    """Pollutant transport simulation"""
    L = 10.0  # River length [km]
    c = 1.0   # Flow velocity [km/h]
    T = 8.0   # Simulation time [h]

    # Initial pollution distribution: high concentration in 0-2 km section
    def u0(x):
        return np.where((x > 0) & (x < 2), np.sin(np.pi * x / 2)**2, 0)

    # High-resolution reference solution
    solver_ref = AdvectionUpwind(L, c, nx=1001, T=T, courant=0.99)
    solver_ref.set_initial_condition(u0)
    _, hist_ref = solver_ref.solve()

    # Low-resolution comparison
    solvers = {
        'Upwind (coarse grid)': AdvectionUpwind(L, c, nx=51, T=T, courant=0.8),
        'Lax-Wendroff (coarse grid)': AdvectionLaxWendroff(L, c, nx=51, T=T, courant=0.8),
        'High-resolution reference': solver_ref,
    }

    results = {}
    for name, solver in solvers.items():
        if name != 'High-resolution reference':
            solver.set_initial_condition(u0)
            _, hist = solver.solve()
            results[name] = (solver, hist)
        else:
            results[name] = (solver, hist_ref)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # t = 0
    ax1 = axes[0, 0]
    ax1.fill_between(solver_ref.x, 0, hist_ref[0], alpha=0.3, color='blue')
    ax1.plot(solver_ref.x, hist_ref[0], 'b-', linewidth=2)
    ax1.set_title('Initial Pollution Distribution (t = 0)')
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Pollution Concentration')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, L)

    # t = T/2
    ax2 = axes[0, 1]
    mid_idx = len(hist_ref) // 2
    ax2.plot(solver_ref.x, hist_ref[mid_idx], 'k--', linewidth=2, label='Reference')
    for name, (solver, hist) in results.items():
        if name != 'High-resolution reference':
            mid = len(hist) // 2
            ax2.plot(solver.x, hist[mid], linewidth=2, label=name)
    ax2.set_title(f't = {T/2} hours')
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Pollution Concentration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, L)

    # t = T
    ax3 = axes[1, 0]
    ax3.plot(solver_ref.x, hist_ref[-1], 'k--', linewidth=2, label='Reference')
    for name, (solver, hist) in results.items():
        if name != 'High-resolution reference':
            ax3.plot(solver.x, hist[-1], linewidth=2, label=name)
    ax3.set_title(f't = {T} hours (Final)')
    ax3.set_xlabel('Distance (km)')
    ax3.set_ylabel('Pollution Concentration')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, L)

    # Space-time plot
    ax4 = axes[1, 1]
    solver = results['Upwind (coarse grid)'][0]
    hist = results['Upwind (coarse grid)'][1]
    times = np.linspace(0, T, len(hist))
    X, Time = np.meshgrid(solver.x, times)
    c = ax4.contourf(X, Time, hist, levels=20, cmap='YlOrRd')
    plt.colorbar(c, ax=ax4, label='Concentration')
    ax4.set_xlabel('Distance (km)')
    ax4.set_ylabel('Time (h)')
    ax4.set_title('Space-Time Pollution Distribution (Upwind)')

    plt.suptitle('River Pollutant Transport Simulation', fontsize=14)
    plt.tight_layout()
    plt.savefig('pollution_transport.png', dpi=150, bbox_inches='tight')
    plt.show()

# pollution_transport()
```

---

## 9. Summary

### Scheme Comparison Table

| Scheme | Accuracy | Stability | Characteristics |
|--------|----------|-----------|-----------------|
| FTCS | O(dt, dx²) | **Unconditionally unstable** | Do not use |
| Upwind | O(dt, dx) | C <= 1 | Numerical diffusion |
| Lax-Friedrichs | O(dt, dx) | C <= 1 | Large numerical diffusion |
| Lax-Wendroff | O(dt², dx²) | C <= 1 | Numerical dispersion (oscillations) |

### CFL Condition

```
C = c·dt/dx <= 1

Physical meaning:
- Numerical information propagation speed >= physical propagation speed
- C = 1: Upwind gives exact solution
```

### Types of Numerical Error

| Type | Cause | Effect | Representative Scheme |
|------|-------|--------|----------------------|
| Numerical diffusion | Odd-order truncation error | Solution spreads | Upwind |
| Numerical dispersion | Even-order truncation error | Oscillations | Lax-Wendroff |

---

## Exercises

### Exercise 1: Confirm FTCS Instability
Run FTCS at various Courant numbers and confirm instability.

### Exercise 2: Reverse Advection
Modify the Upwind scheme for c < 0 and test.

### Exercise 3: Beam-Warming Scheme
Implement second-order upwind (Beam-Warming) and compare with Lax-Wendroff.

### Exercise 4: 2D Advection
Solve du/dt + c_x·du/dx + c_y·du/dy = 0.

---

## References

1. **Textbook**: LeVeque, "Numerical Methods for Conservation Laws"
2. **CFD**: Versteeg & Malalasekera, "An Introduction to CFD"
3. **Numerical Analysis**: Strikwerda, "Finite Difference Schemes and PDEs"
