# 13. CFD Basics (Computational Fluid Dynamics Basics)

## Learning Objectives
- Understand fundamental principles of fluid mechanics and governing equations
- Grasp the relationship between Reynolds number and flow characteristics
- Understand the derivation and meaning of Navier-Stokes equations
- Distinguish between compressible and incompressible flows
- Learn the boundary layer concept
- Implement simple channel flow CFD

**Why This Lesson Matters:** Fluid mechanics governs an enormous range of phenomena -- from blood flow in arteries to weather prediction to aircraft design. The Navier-Stokes equations that describe fluid motion are nonlinear PDEs that rarely admit analytical solutions. CFD (Computational Fluid Dynamics) bridges this gap by solving these equations numerically, making it one of the most impactful applications of numerical simulation.

---

## 1. Fluid Mechanics Fundamentals

### 1.1 Continuum Hypothesis

The continuum hypothesis is the foundational assumption that enables us to use PDEs for fluid modeling. It states that we can treat the fluid as a continuous medium rather than tracking individual molecules, provided the flow length scales are much larger than the molecular mean free path.

```
Continuum Hypothesis:
- Treat fluid as a continuous medium
- Use fluid particle concept instead of individual molecules
- Valid when Knudsen number Kn = lambda/L << 1
  (lambda: mean free path, L: characteristic length)
```

### 1.2 Basic Material Properties

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Fluid properties example
class FluidProperties:
    """Fluid properties class"""

    def __init__(self, name, rho, mu, k=None, cp=None):
        """
        Parameters:
        - name: fluid name
        - rho: density [kg/m^3]
        - mu: dynamic viscosity [Pa·s]
        - k: thermal conductivity [W/(m·K)]
        - cp: specific heat [J/(kg·K)]
        """
        self.name = name
        self.rho = rho      # density
        self.mu = mu        # dynamic viscosity
        self.k = k          # thermal conductivity
        self.cp = cp        # specific heat at constant pressure

    @property
    def nu(self):
        """Kinematic viscosity"""
        return self.mu / self.rho

    @property
    def alpha(self):
        """Thermal diffusivity"""
        if self.k and self.cp:
            return self.k / (self.rho * self.cp)
        return None

    @property
    def Pr(self):
        """Prandtl number"""
        if self.cp:
            return self.mu * self.cp / self.k
        return None

    def __repr__(self):
        return f"FluidProperties({self.name}): rho={self.rho}, mu={self.mu}, nu={self.nu:.2e}"

# Common fluids
water = FluidProperties("Water (20 deg C)", rho=998, mu=1.002e-3, k=0.598, cp=4182)
air = FluidProperties("Air (20 deg C)", rho=1.204, mu=1.825e-5, k=0.0257, cp=1007)
oil = FluidProperties("Engine Oil (20 deg C)", rho=880, mu=0.29, k=0.145, cp=1880)

print(water)
print(f"  Prandtl number: {water.Pr:.2f}")
print(air)
print(f"  Prandtl number: {air.Pr:.2f}")
```

### 1.3 Reynolds Number

The Reynolds number is perhaps the most important dimensionless parameter in fluid mechanics. It measures the ratio of inertial forces (which cause turbulent mixing) to viscous forces (which damp out disturbances):

$$Re = \frac{\rho U L}{\mu} = \frac{U L}{\nu}$$

At low $Re$, viscosity dominates and flow is smooth (laminar). At high $Re$, inertia dominates and flow becomes chaotic (turbulent). This single number determines the qualitative character of the flow and dictates the numerical resolution required.

```
Reynolds Number Definition:
Re = rho*U*L/mu = U*L/nu = (Inertial forces)/(Viscous forces)

where:
- rho: fluid density
- U: characteristic velocity
- L: characteristic length
- mu: dynamic viscosity
- nu = mu/rho: kinematic viscosity

Flow Characteristics:
- Re < 2300: Laminar flow
- 2300 < Re < 4000: Transition region
- Re > 4000: Turbulent flow
```

```python
def reynolds_number_analysis():
    """Reynolds number and flow characteristics analysis"""

    # Pipe flow example
    D = 0.05  # pipe diameter [m]
    U_range = np.linspace(0.01, 2.0, 100)  # velocity range [m/s]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reynolds number in water
    ax1 = axes[0]
    Re_water = water.rho * U_range * D / water.mu

    ax1.plot(U_range, Re_water, 'b-', linewidth=2, label='Water')
    ax1.axhline(y=2300, color='orange', linestyle='--', label='Transition start')
    ax1.axhline(y=4000, color='red', linestyle='--', label='Turbulent')

    ax1.fill_between(U_range, 0, 2300, alpha=0.2, color='green', label='Laminar')
    ax1.fill_between(U_range, 2300, 4000, alpha=0.2, color='orange')
    ax1.fill_between(U_range, 4000, max(Re_water), alpha=0.2, color='red')

    ax1.set_xlabel('Flow Velocity U [m/s]')
    ax1.set_ylabel('Reynolds Number Re')
    ax1.set_title(f'Reynolds Number in Pipe Flow (D = {D*100} cm, Water)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Compare different fluids
    ax2 = axes[1]
    fluids = [water, air, oil]
    colors = ['blue', 'cyan', 'brown']

    for fluid, color in zip(fluids, colors):
        Re = fluid.rho * U_range * D / fluid.mu
        ax2.plot(U_range, Re, color=color, linewidth=2, label=fluid.name)

    ax2.axhline(y=2300, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Flow Velocity U [m/s]')
    ax2.set_ylabel('Reynolds Number Re')
    ax2.set_title('Reynolds Number Comparison for Different Fluids')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('reynolds_number.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Practical calculation examples
    print("\n=== Practical Reynolds Number Calculation Examples ===")
    print(f"Pipe diameter D = {D*100} cm")

    test_velocities = [0.1, 0.5, 1.0]
    for U in test_velocities:
        Re = water.rho * U * D / water.mu
        regime = "Laminar" if Re < 2300 else ("Transition" if Re < 4000 else "Turbulent")
        print(f"  U = {U} m/s -> Re = {Re:.0f} ({regime})")

# reynolds_number_analysis()
```

---

## 2. Governing Equations

### 2.1 Continuity Equation

```
Mass Conservation Law:
d(rho)/dt + nabla·(rho*u) = 0

Tensor notation:
d(rho)/dt + d(rho*u_i)/dx_i = 0

Incompressible flow (rho = const):
nabla·u = 0
or: du/dx + dv/dy + dw/dz = 0
```

```python
def visualize_continuity():
    """Continuity equation visualization - control volume approach"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Control volume concept
    ax1 = axes[0]

    # Control volume (rectangle)
    cv = plt.Rectangle((0.3, 0.3), 0.4, 0.4, fill=False,
                       edgecolor='black', linewidth=2)
    ax1.add_patch(cv)

    # Mass inflow/outflow arrows
    # x direction
    ax1.annotate('', xy=(0.3, 0.5), xytext=(0.1, 0.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax1.text(0.15, 0.55, r'$\rho u A$', fontsize=12, color='blue')

    ax1.annotate('', xy=(0.9, 0.5), xytext=(0.7, 0.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax1.text(0.75, 0.55, r'$\rho u A + \frac{\partial(\rho u)}{\partial x}\Delta x A$', fontsize=10, color='blue')

    # y direction
    ax1.annotate('', xy=(0.5, 0.3), xytext=(0.5, 0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.text(0.55, 0.15, r'$\rho v A$', fontsize=12, color='red')

    ax1.annotate('', xy=(0.5, 0.9), xytext=(0.5, 0.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.text(0.55, 0.8, r'$\rho v A + ...$', fontsize=10, color='red')

    ax1.text(0.5, 0.5, r'$\Delta V$', fontsize=14, ha='center', va='center')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.set_title('Control Volume and Mass Flux')
    ax1.axis('off')

    # Incompressible flow field example (div u = 0)
    ax2 = axes[1]
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)

    # Flow field: u = y, v = -x (rotational flow around point, div=0)
    U = Y
    V = -X

    # Divergence calculation (numerical)
    div = np.zeros_like(X)

    ax2.streamplot(X, Y, U, V, color='blue', density=1.5, linewidth=1)
    ax2.quiver(X[::2, ::2], Y[::2, ::2], U[::2, ::2], V[::2, ::2],
              color='red', alpha=0.7)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(r'Incompressible Flow Field Example: $u=y, v=-x$ ($\nabla \cdot \mathbf{u} = 0$)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('continuity_equation.png', dpi=150, bbox_inches='tight')
    plt.show()

# visualize_continuity()
```

### 2.2 Momentum Equation

```
Newton's Second Law Applied:
rho(Du/Dt) = -nabla*p + mu*nabla²u + rho*g + f

where:
- Du/Dt = du/dt + (u·nabla)u : material derivative
- nabla*p: pressure gradient
- mu*nabla²u: viscous forces
- rho*g: gravity
- f: external body forces

Component form (incompressible, 2D):
x: rho(du/dt + u*du/dx + v*du/dy) = -dp/dx + mu(d²u/dx² + d²u/dy²)
y: rho(dv/dt + u*dv/dx + v*dv/dy) = -dp/dy + mu(d²v/dx² + d²v/dy²)
```

### 2.3 Navier-Stokes Equation Derivation

```python
def navier_stokes_derivation():
    """Visualize meaning of each term in Navier-Stokes equations"""

    print("=" * 60)
    print("Navier-Stokes Equation Derivation")
    print("=" * 60)

    derivation = """
    1. Mass Conservation (Continuity):
       d(rho)/dt + nabla·(rho*u) = 0

       Incompressible: nabla·u = 0

    2. Momentum Conservation (Newton's 2nd Law):

       Material Derivative (Lagrangian):
       Du/Dt = du/dt + (u·nabla)u
                |         |
           local      convective
           acceleration  acceleration

       Force Balance:
       rho(Du/Dt) = sum(F) = -nabla*p + nabla·tau + rho*g
                     |       |          |          |
                  inertia  pressure  viscous     body
                  force    force     force       force

       Newtonian Fluid Assumption (tau = mu(nabla*u + nabla*u^T)):
       rho(Du/Dt) = -nabla*p + mu*nabla²u + rho*g

    3. Incompressible Navier-Stokes Equations:

       du/dt + (u·nabla)u = -1/rho * nabla*p + nu*nabla²u + g

       where nu = mu/rho (kinematic viscosity)

    4. Dimensionless Form:

       Characteristic scales: L(length), U(velocity), T=L/U(time), P=rho*U²(pressure)

       du*/dt* + (u*·nabla*)u* = -nabla*p* + (1/Re)*nabla*²u*

       where Re = U*L/nu (Reynolds number)
    """
    print(derivation)

    # Visualize relative magnitudes of each term
    fig, ax = plt.subplots(figsize=(12, 6))

    Re_range = np.logspace(0, 6, 100)

    # Dimensionless magnitude comparison
    inertia = np.ones_like(Re_range)  # O(1)
    viscous = 1 / Re_range            # O(1/Re)
    pressure = np.ones_like(Re_range) # O(1)

    ax.loglog(Re_range, inertia, 'b-', linewidth=2, label='Inertia term O(1)')
    ax.loglog(Re_range, viscous, 'r-', linewidth=2, label='Viscous term O(1/Re)')
    ax.loglog(Re_range, pressure, 'g--', linewidth=2, label='Pressure term O(1)')

    ax.axvline(x=2300, color='gray', linestyle=':', label='Laminar-turbulent transition')
    ax.fill_between(Re_range, 1e-6, 1, where=Re_range < 2300,
                   alpha=0.2, color='green', label='Viscous dominated')
    ax.fill_between(Re_range, 1e-6, 1, where=Re_range >= 2300,
                   alpha=0.2, color='blue', label='Inertia dominated')

    ax.set_xlabel('Reynolds Number Re')
    ax.set_ylabel('Relative Magnitude')
    ax.set_title('Relative Magnitude of Navier-Stokes Equation Terms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-6, 10)

    plt.tight_layout()
    plt.savefig('navier_stokes_terms.png', dpi=150, bbox_inches='tight')
    plt.show()

# navier_stokes_derivation()
```

---

## 3. Compressible vs Incompressible Flow

### 3.1 Mach Number and Compressibility

```
Mach Number Definition:
Ma = U/a

where:
- U: fluid velocity
- a: speed of sound (ideal gas: a = sqrt(gamma*R*T))

Classification:
- Ma < 0.3: Can be treated as incompressible (density change < 5%)
- 0.3 < Ma < 0.8: Subsonic
- 0.8 < Ma < 1.2: Transonic
- 1.2 < Ma < 5: Supersonic
- Ma > 5: Hypersonic
```

```python
def compressibility_analysis():
    """Compressibility effect analysis"""

    # Isentropic density ratio
    def density_ratio(Ma, gamma=1.4):
        """rho/rho_0 = (1 + (gamma-1)/2 Ma²)^(-1/(gamma-1))"""
        return (1 + (gamma - 1) / 2 * Ma**2) ** (-1 / (gamma - 1))

    # Pressure ratio
    def pressure_ratio(Ma, gamma=1.4):
        """p/p_0 = (1 + (gamma-1)/2 Ma²)^(-gamma/(gamma-1))"""
        return (1 + (gamma - 1) / 2 * Ma**2) ** (-gamma / (gamma - 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    Ma = np.linspace(0, 3, 200)

    # Density ratio
    ax1 = axes[0]
    rho_ratio = density_ratio(Ma)
    ax1.plot(Ma, rho_ratio, 'b-', linewidth=2)
    ax1.axvline(x=0.3, color='green', linestyle='--', label='Ma=0.3 (Incompressible limit)')
    ax1.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5)
    ax1.fill_between(Ma, 0, rho_ratio, where=Ma < 0.3, alpha=0.3, color='green',
                    label='Incompressible region')

    ax1.set_xlabel('Mach Number Ma')
    ax1.set_ylabel(r'Density Ratio $\rho/\rho_0$')
    ax1.set_title('Density Change in Isentropic Flow')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, 1.1)

    # Compressible/Incompressible equation comparison
    ax2 = axes[1]

    equations = {
        'Incompressible': [
            r'$\nabla \cdot \mathbf{u} = 0$',
            r'$\rho \frac{D\mathbf{u}}{Dt} = -\nabla p + \mu \nabla^2 \mathbf{u}$',
            '3 equations, 4 unknowns (u,v,w,p)',
            'Pressure: determined by Poisson equation'
        ],
        'Compressible': [
            r'$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0$',
            r'$\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot (\rho \mathbf{u} \otimes \mathbf{u}) = -\nabla p + \nabla \cdot \tau$',
            r'$\frac{\partial E}{\partial t} + \nabla \cdot ((E+p)\mathbf{u}) = \nabla \cdot (k\nabla T + \tau \cdot \mathbf{u})$',
            '5 equations, 5 unknowns (rho,u,v,w,E) + equation of state'
        ]
    }

    ax2.text(0.25, 0.95, 'Incompressible (Ma < 0.3)', fontsize=14, fontweight='bold',
            ha='center', transform=ax2.transAxes)
    ax2.text(0.75, 0.95, 'Compressible (Ma > 0.3)', fontsize=14, fontweight='bold',
            ha='center', transform=ax2.transAxes)

    y_pos = 0.85
    for eq in equations['Incompressible']:
        ax2.text(0.25, y_pos, eq, fontsize=10, ha='center', transform=ax2.transAxes)
        y_pos -= 0.12

    y_pos = 0.85
    for eq in equations['Compressible']:
        ax2.text(0.75, y_pos, eq, fontsize=10, ha='center', transform=ax2.transAxes)
        y_pos -= 0.12

    ax2.axvline(x=0.5, color='black', linestyle='-', linewidth=2,
               transform=ax2.transAxes)
    ax2.axis('off')
    ax2.set_title('Governing Equations Comparison')

    plt.tight_layout()
    plt.savefig('compressibility.png', dpi=150, bbox_inches='tight')
    plt.show()

# compressibility_analysis()
```

---

## 4. Boundary Layer Theory

### 4.1 Boundary Layer Concept

```
Boundary Layer:
- Region near wall where viscous effects dominate
- Wall: no-slip condition (u = 0)
- Outside boundary layer: free stream velocity U_infinity

Boundary Layer Thickness delta:
- Position where velocity reaches 99% of free stream
- Laminar flat plate: delta ~ x/sqrt(Re_x) (Blasius)
- Turbulent flat plate: delta ~ x/Re_x^(1/5)
```

```python
def boundary_layer_theory():
    """Boundary layer theory visualization"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Blasius laminar boundary layer profile
    ax1 = axes[0, 0]

    # Blasius similarity solution (approximation)
    eta = np.linspace(0, 8, 100)  # Dimensionless coordinate eta = y*sqrt(U_inf/(nu*x))

    # f'(eta) approximation (actual is numerical solution of Blasius equation)
    # Using simple approximation here
    u_U = np.tanh(eta / 2.5) ** 1.5  # Approximation

    ax1.plot(u_U, eta, 'b-', linewidth=2)
    ax1.axhline(y=5.0, color='red', linestyle='--', label=r'$\delta_{99}$ (eta approx 5)')
    ax1.fill_betweenx(eta, 0, u_U, alpha=0.3)

    ax1.set_xlabel(r'$u/U_\infty$')
    ax1.set_ylabel(r'$\eta = y\sqrt{U_\infty/\nu x}$')
    ax1.set_title('Blasius Laminar Boundary Layer Velocity Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.1)
    ax1.set_ylim(0, 8)

    # (2) Boundary layer thickness growth
    ax2 = axes[0, 1]

    nu = 1.5e-5  # Air kinematic viscosity
    U_inf = 10   # Free stream velocity [m/s]
    x = np.linspace(0.01, 1, 100)  # Position on plate [m]

    Re_x = U_inf * x / nu

    # Laminar boundary layer thickness (Blasius)
    delta_lam = 5.0 * x / np.sqrt(Re_x)

    # Turbulent boundary layer thickness (1/7 power law)
    delta_turb = 0.37 * x / Re_x ** 0.2

    # Transition location (Re_x ~ 5x10^5)
    x_trans = 5e5 * nu / U_inf

    ax2.plot(x * 1000, delta_lam * 1000, 'b-', linewidth=2, label='Laminar')
    ax2.plot(x * 1000, delta_turb * 1000, 'r-', linewidth=2, label='Turbulent')
    ax2.axvline(x=x_trans * 1000, color='green', linestyle='--', label=f'Transition point (x approx {x_trans*1000:.0f} mm)')

    ax2.set_xlabel('x [mm]')
    ax2.set_ylabel(r'$\delta$ [mm]')
    ax2.set_title(f'Boundary Layer Thickness Growth (U_inf = {U_inf} m/s, Air)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # (3) Wall shear stress
    ax3 = axes[1, 0]

    # Laminar wall shear stress coefficient
    Cf_lam = 0.664 / np.sqrt(Re_x)

    # Turbulent wall shear stress coefficient
    Cf_turb = 0.027 / Re_x ** (1/7)

    ax3.loglog(Re_x, Cf_lam, 'b-', linewidth=2, label='Laminar (Blasius)')
    ax3.loglog(Re_x, Cf_turb, 'r-', linewidth=2, label='Turbulent (1/7 power law)')
    ax3.axvline(x=5e5, color='green', linestyle='--', alpha=0.5)

    ax3.set_xlabel(r'$Re_x$')
    ax3.set_ylabel(r'$C_f = \tau_w / (0.5\rho U_\infty^2)$')
    ax3.set_title('Wall Friction Coefficient')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # (4) Boundary layer concept diagram
    ax4 = axes[1, 1]

    # Flat plate
    ax4.fill_between([0, 5], [-0.1, -0.1], [0, 0], color='gray', alpha=0.5)

    # Boundary layer
    x_plate = np.linspace(0, 5, 50)
    delta_vis = 0.5 * np.sqrt(x_plate)  # Simplified boundary layer
    ax4.fill_between(x_plate, 0, delta_vis, alpha=0.3, color='blue', label='Boundary layer')
    ax4.plot(x_plate, delta_vis, 'b-', linewidth=2)

    # Velocity profile arrows
    for x0 in [0.5, 1.5, 3.0, 4.5]:
        y_arrows = np.linspace(0, 0.5 * np.sqrt(x0) * 1.2, 6)
        for y in y_arrows:
            u = min(1, y / (0.5 * np.sqrt(x0))) if x0 > 0 else 0
            ax4.annotate('', xy=(x0 + u * 0.3, y), xytext=(x0, y),
                        arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

    # Free stream
    ax4.annotate('', xy=(5, 1.5), xytext=(0, 1.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax4.text(2.5, 1.7, r'$U_\infty$', fontsize=14)

    ax4.text(2.5, 0.3, 'Boundary Layer', fontsize=12, color='blue')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Boundary Layer Development on Flat Plate')
    ax4.set_xlim(-0.5, 5.5)
    ax4.set_ylim(-0.2, 2)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('boundary_layer.png', dpi=150, bbox_inches='tight')
    plt.show()

# boundary_layer_theory()
```

---

## 5. Simple CFD Example: Poiseuille Flow

### 5.1 2D Channel Flow (Poiseuille Flow)

```
Problem Setup:
- Steady laminar flow between two parallel plates
- Driven by pressure gradient
- Analytical solution exists (for verification)

Governing Equation (steady, fully developed):
d²u/dy² = (1/mu)(dp/dx) = const

Boundary Conditions:
- y = 0: u = 0 (no-slip)
- y = H: u = 0 (no-slip)

Analytical Solution:
u(y) = -(1/2mu)(dp/dx)y(H-y)

Maximum velocity (center):
u_max = (H²/8mu)|dp/dx|
```

```python
def poiseuille_flow_exact():
    """Poiseuille flow analytical solution"""

    H = 1.0       # channel height [m]
    mu = 0.01     # dynamic viscosity [Pa·s]
    dpdx = -1.0   # pressure gradient [Pa/m] (negative = positive x direction flow)

    y = np.linspace(0, H, 100)
    u_exact = -(1 / (2 * mu)) * dpdx * y * (H - y)

    u_max = H**2 / (8 * mu) * abs(dpdx)
    u_avg = 2 / 3 * u_max

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Velocity profile
    ax1 = axes[0]
    ax1.plot(u_exact, y, 'b-', linewidth=2, label='Analytical')
    ax1.axhline(y=H/2, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(x=u_max, color='green', linestyle='--', alpha=0.5, label=f'u_max = {u_max:.2f}')
    ax1.axvline(x=u_avg, color='orange', linestyle='--', alpha=0.5, label=f'u_avg = {u_avg:.2f}')

    ax1.set_xlabel('u [m/s]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('Poiseuille Flow Velocity Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Shear stress profile
    ax2 = axes[1]
    tau = mu * np.gradient(u_exact, y)
    ax2.plot(tau, y, 'r-', linewidth=2)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    ax2.set_xlabel(r'$\tau$ [Pa]')
    ax2.set_ylabel('y [m]')
    ax2.set_title('Shear Stress Distribution')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('poiseuille_exact.png', dpi=150, bbox_inches='tight')
    plt.show()

    return u_exact, y

# u_exact, y = poiseuille_flow_exact()
```

### 5.2 CFD Implementation with Finite Differences

```python
def cfd_channel_flow():
    """
    2D Channel Flow CFD Simulation
    - Unsteady Navier-Stokes equations
    - Time-march to steady state
    """

    # Grid setup
    Nx = 50       # x-direction grid points
    Ny = 30       # y-direction grid points
    Lx = 2.0      # channel length [m]
    Ly = 1.0      # channel height [m]

    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    # Fluid properties
    rho = 1.0     # density [kg/m^3]
    mu = 0.01     # dynamic viscosity [Pa·s]
    nu = mu / rho

    # Pressure gradient (modeled as body force)
    dpdx = -1.0   # [Pa/m]

    # Time settings
    dt = 0.001
    n_steps = 2000

    # Initialize
    u = np.zeros((Ny, Nx))  # x-velocity
    v = np.zeros((Ny, Nx))  # y-velocity
    p = np.zeros((Ny, Nx))  # pressure

    # CFL check
    u_max_expected = Ly**2 / (8 * mu) * abs(dpdx)
    CFL = u_max_expected * dt / dx
    print(f"Expected maximum velocity: {u_max_expected:.4f} m/s")
    print(f"CFL number: {CFL:.4f}")

    # Analytical solution (for verification)
    u_exact = -(1 / (2 * mu)) * dpdx * y * (Ly - y)

    def apply_boundary_conditions(u, v):
        """Apply boundary conditions"""
        # Walls (no-slip)
        u[0, :] = 0    # bottom wall
        u[-1, :] = 0   # top wall
        v[0, :] = 0
        v[-1, :] = 0

        # Inlet/outlet (periodic or Neumann)
        u[:, 0] = u[:, 1]    # inlet
        u[:, -1] = u[:, -2]  # outlet
        v[:, 0] = 0
        v[:, -1] = v[:, -2]

        return u, v

    def compute_rhs(u, v, p, nu, dx, dy, dpdx, rho):
        """Compute right-hand side (momentum equations)"""
        Ny, Nx = u.shape
        rhs_u = np.zeros_like(u)
        rhs_v = np.zeros_like(v)

        for i in range(1, Ny-1):
            for j in range(1, Nx-1):
                # Convection terms (central difference)
                duudx = (u[i, j+1]**2 - u[i, j-1]**2) / (2 * dx)
                duvdy = (u[i+1, j] * v[i+1, j] - u[i-1, j] * v[i-1, j]) / (2 * dy)

                dvudx = (v[i, j+1] * u[i, j+1] - v[i, j-1] * u[i, j-1]) / (2 * dx)
                dvvdy = (v[i+1, j]**2 - v[i-1, j]**2) / (2 * dy)

                # Diffusion terms (central difference)
                d2udx2 = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dx**2
                d2udy2 = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dy**2

                d2vdx2 = (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dx**2
                d2vdy2 = (v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dy**2

                # Pressure terms
                dpdx_local = (p[i, j+1] - p[i, j-1]) / (2 * dx) if j > 0 and j < Nx-1 else 0
                dpdy_local = (p[i+1, j] - p[i-1, j]) / (2 * dy) if i > 0 and i < Ny-1 else 0

                # Momentum equation RHS
                rhs_u[i, j] = -duudx - duvdy - dpdx_local/rho + nu * (d2udx2 + d2udy2) - dpdx/rho
                rhs_v[i, j] = -dvudx - dvvdy - dpdy_local/rho + nu * (d2vdx2 + d2vdy2)

        return rhs_u, rhs_v

    # Time marching
    history = []

    for n in range(n_steps):
        # Boundary conditions
        u, v = apply_boundary_conditions(u, v)

        # RHS calculation
        rhs_u, rhs_v = compute_rhs(u, v, p, nu, dx, dy, dpdx, rho)

        # Time advancement (Euler)
        u = u + dt * rhs_u
        v = v + dt * rhs_v

        # Convergence check
        if n % 200 == 0:
            u_center = u[:, Nx//2]
            error = np.max(np.abs(u_center - u_exact))
            history.append((n, error, np.max(u)))
            print(f"Step {n}: max error = {error:.6f}, max u = {np.max(u):.6f}")

    # Results visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Velocity field (vectors)
    ax1 = axes[0, 0]
    skip = 2
    ax1.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              u[::skip, ::skip], v[::skip, ::skip],
              color='blue', scale=30)
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('Velocity Vector Field')
    ax1.set_aspect('equal')

    # (2) u-velocity contours
    ax2 = axes[0, 1]
    cf = ax2.contourf(X, Y, u, levels=20, cmap='jet')
    plt.colorbar(cf, ax=ax2, label='u [m/s]')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_title('u-Velocity Distribution')
    ax2.set_aspect('equal')

    # (3) Comparison with analytical solution
    ax3 = axes[1, 0]
    u_center = u[:, Nx//2]
    ax3.plot(u_center, y, 'bo-', markersize=4, label='CFD')
    ax3.plot(u_exact, y, 'r-', linewidth=2, label='Analytical')
    ax3.set_xlabel('u [m/s]')
    ax3.set_ylabel('y [m]')
    ax3.set_title('Velocity Profile Comparison (x = L/2)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # (4) Convergence history
    ax4 = axes[1, 1]
    steps, errors, u_maxs = zip(*history)
    ax4.semilogy(steps, errors, 'b-o', label='Error')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Max Error')
    ax4.set_title('Convergence History')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cfd_channel_flow.png', dpi=150, bbox_inches='tight')
    plt.show()

    return u, v, p, X, Y

# u, v, p, X, Y = cfd_channel_flow()
```

---

## 6. Key Challenges in CFD

### 6.1 Mesh Generation

```python
def mesh_types_visualization():
    """CFD mesh types visualization"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (1) Structured mesh
    ax1 = axes[0, 0]
    nx, ny = 10, 8
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 1, ny)

    for i in range(ny):
        ax1.plot(x, np.full_like(x, y[i]), 'b-', linewidth=0.5)
    for j in range(nx):
        ax1.plot(np.full_like(y, x[j]), y, 'b-', linewidth=0.5)

    X, Y = np.meshgrid(x, y)
    ax1.plot(X, Y, 'ko', markersize=3)
    ax1.set_title('Structured Mesh')
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.1, 2.1)
    ax1.set_ylim(-0.1, 1.1)

    # (2) Unstructured mesh
    ax2 = axes[0, 1]
    from scipy.spatial import Delaunay

    # Random points
    np.random.seed(42)
    points = np.random.rand(30, 2)
    points[:, 0] *= 2

    # Add boundary points
    boundary = np.array([[0, 0], [2, 0], [2, 1], [0, 1]])
    for i in range(4):
        edge = np.linspace(boundary[i], boundary[(i+1)%4], 8)[1:-1]
        points = np.vstack([points, edge])

    tri = Delaunay(points)
    ax2.triplot(points[:, 0], points[:, 1], tri.simplices, 'b-', linewidth=0.5)
    ax2.plot(points[:, 0], points[:, 1], 'ko', markersize=3)
    ax2.set_title('Unstructured Mesh')
    ax2.set_aspect('equal')
    ax2.set_xlim(-0.1, 2.1)
    ax2.set_ylim(-0.1, 1.1)

    # (3) O-mesh (around cylinder)
    ax3 = axes[1, 0]
    r_inner = 0.3
    r_outer = 1.0
    n_r = 8
    n_theta = 24

    r = np.linspace(r_inner, r_outer, n_r)
    theta = np.linspace(0, 2*np.pi, n_theta)

    for ri in r:
        x_circle = ri * np.cos(theta)
        y_circle = ri * np.sin(theta)
        ax3.plot(x_circle, y_circle, 'b-', linewidth=0.5)

    for ti in theta[:-1]:
        x_radial = r * np.cos(ti)
        y_radial = r * np.sin(ti)
        ax3.plot(x_radial, y_radial, 'b-', linewidth=0.5)

    R, Theta = np.meshgrid(r, theta)
    X_o = R * np.cos(Theta)
    Y_o = R * np.sin(Theta)
    ax3.plot(X_o, Y_o, 'ko', markersize=2)

    ax3.set_title('O-Mesh (Around Cylinder)')
    ax3.set_aspect('equal')
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)

    # (4) Boundary layer prism mesh
    ax4 = axes[1, 1]

    # Boundary layer region (prism)
    x_wall = np.linspace(0, 2, 20)
    y_layers = [0, 0.02, 0.05, 0.1, 0.2, 0.4]

    for yl in y_layers:
        ax4.plot(x_wall, np.full_like(x_wall, yl), 'b-', linewidth=0.5)

    # Outer unstructured mesh (triangles)
    np.random.seed(123)
    outer_points = np.random.rand(20, 2)
    outer_points[:, 0] *= 2
    outer_points[:, 1] = outer_points[:, 1] * 0.5 + 0.4

    tri_outer = Delaunay(outer_points)
    ax4.triplot(outer_points[:, 0], outer_points[:, 1], tri_outer.simplices,
               'g-', linewidth=0.5)

    ax4.fill_between(x_wall, 0, y_layers[-1], alpha=0.2, color='blue',
                    label='Boundary layer (prism)')
    ax4.set_title('Hybrid Mesh (Prism + Triangle)')
    ax4.set_aspect('equal')
    ax4.set_xlim(-0.1, 2.1)
    ax4.set_ylim(-0.05, 1)
    ax4.legend()

    plt.tight_layout()
    plt.savefig('mesh_types.png', dpi=150, bbox_inches='tight')
    plt.show()

# mesh_types_visualization()
```

### 6.2 Turbulence Modeling

```
Major Turbulence Models:

1. RANS (Reynolds-Averaged Navier-Stokes):
   - k-epsilon model: general-purpose, industry standard
   - k-omega model: accurate near walls
   - SST model: combines k-epsilon + k-omega advantages

2. LES (Large Eddy Simulation):
   - Large eddies computed directly
   - Small eddies modeled (SGS model)

3. DNS (Direct Numerical Simulation):
   - All turbulence scales computed directly
   - Computational cost proportional to Re^3

Selection Criteria:
- Accuracy: DNS > LES > RANS
- Computational cost: DNS > LES > RANS
- Practicality: RANS > LES > DNS
```

---

## 7. Exercises

### Exercise 1: Reynolds Number Calculation
Calculate the Reynolds number for water (20 deg C) flowing at average velocity 2 m/s through a 5 cm diameter pipe. Determine the flow regime.

### Exercise 2: Poiseuille Flow
Derive the relationship between average velocity and maximum velocity in Poiseuille flow.

### Exercise 3: Boundary Layer Thickness
Calculate the laminar boundary layer thickness at 10 cm from the leading edge for air (20 deg C) flowing at 5 m/s over a flat plate.

### Exercise 4: CFD Grid Dependence
Modify the channel flow CFD code to perform a grid convergence test by varying the grid size.

---

## 8. References

### Key Textbooks
- Versteeg & Malalasekera, "An Introduction to Computational Fluid Dynamics"
- Anderson, "Computational Fluid Dynamics: The Basics with Applications"
- Ferziger & Peric, "Computational Methods for Fluid Dynamics"

### CFD Software
- OpenFOAM (open source)
- ANSYS Fluent (commercial)
- COMSOL Multiphysics (commercial)
- SU2 (open source)

### Online Resources
- NASA CFD Resources
- CFD Online (forums, tutorials)
- LearnCAx (free courses)

---

## Summary

```
CFD Fundamentals Summary:

1. Governing Equations:
   - Continuity: nabla·u = 0 (incompressible)
   - Momentum: rho(Du/Dt) = -nabla*p + mu*nabla²u
   - Energy: (compressible flow)

2. Dimensionless Numbers:
   - Re = rho*U*L/mu (inertia/viscous)
   - Ma = U/a (compressibility)
   - Pr = nu/alpha (momentum/thermal diffusion)

3. CFD Workflow:
   (1) Mesh generation (pre-processing)
   (2) Discretization and solution (solver)
   (3) Post-processing (visualization, analysis)

4. Key Considerations:
   - Grid quality and convergence
   - Boundary condition setup
   - Turbulence model selection
   - Numerical stability (CFL condition)
```

---

## Exercises

### Exercise 1: Reynolds Number and Flow Regime
Water at 20°C (density 998 kg/m³, dynamic viscosity 1.002×10⁻³ Pa·s) flows through a 5 cm diameter pipe at an average velocity of 2 m/s. Calculate the Reynolds number and determine the flow regime. Then find the minimum velocity needed to achieve turbulent flow in the same pipe.

### Exercise 2: Navier-Stokes Term Analysis
For a 2D incompressible flow with velocity field u = sin(x)cos(y) and v = -cos(x)sin(y), verify that the continuity equation is satisfied. Then compute the vorticity ω = ∂v/∂x - ∂u/∂y and describe what it represents physically.

### Exercise 3: Poiseuille Flow Comparison
Implement the analytical Poiseuille flow solution and the finite difference CFD solution for a channel with height H = 1 m, dynamic viscosity μ = 0.01 Pa·s, and pressure gradient dp/dx = -1 Pa/m. Use grid sizes Ny = 10, 20, and 40, and plot the L2 error between the numerical and analytical solutions as a function of grid spacing. What convergence order do you observe?

### Exercise 4: Boundary Layer Scaling
For air at 20°C (kinematic viscosity ν = 1.5×10⁻⁵ m²/s) flowing at U∞ = 10 m/s over a flat plate, compute and plot: (a) the laminar boundary layer thickness δ(x) for 0 < x < 0.5 m using the Blasius formula δ = 5x/√(Re_x), (b) the local friction coefficient Cf(x) = 0.664/√(Re_x), and (c) the transition location where Re_x = 5×10⁵. At what flow speed does the transition point move to x = 0.1 m?

### Exercise 5: CFD Mesh Sensitivity
Modify the channel flow CFD code to perform a systematic grid convergence study. Run simulations with Nx×Ny = 25×15, 50×30, and 100×60 grids. For each grid, compare the centerline velocity profile against the analytical Poiseuille solution and compute the maximum absolute error. Does the error halve when the grid spacing is halved (first-order convergence) or reduce by a factor of four (second-order)?

---

In the next lesson, we will cover the Lid-Driven Cavity problem and the SIMPLE algorithm, which are representative problems for incompressible flow.
