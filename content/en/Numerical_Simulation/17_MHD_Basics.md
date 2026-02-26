# 17. MHD Basics (Magnetohydrodynamics Basics)

## Learning Objectives
- Understand basic concepts of Magnetohydrodynamics (MHD)
- Grasp MHD assumptions and applicability
- Derive ideal MHD equations
- Understand Alfven velocity and MHD waves
- Learn concepts of magnetic pressure and magnetic tension

**Why This Lesson Matters:** Magnetohydrodynamics (MHD) governs some of the most energetic phenomena in the universe: solar flares, astrophysical jets, stellar dynamos, and fusion reactor plasmas. By coupling fluid dynamics with Maxwell's equations, MHD captures how conducting fluids and magnetic fields interact -- the magnetic field exerts forces on the fluid, while the fluid's motion stretches and amplifies the field. Understanding MHD is essential for space weather prediction, controlled fusion, and astrophysical modeling.

---

## 1. Introduction to MHD

### 1.1 Definition and Applications

```
Magnetohydrodynamics (MHD):
- Interaction between electrically conducting fluids and electromagnetic fields
- Combination of fluid dynamics + electromagnetism

Application Areas:
1. Astrophysics: Sun, stars, galaxies, interstellar medium
2. Nuclear fusion: Tokamak, stellarator plasma confinement
3. Geophysics: Earth's magnetic field dynamo
4. Engineering: MHD generators, electromagnetic pumps, metal casting
5. Space physics: Solar wind, magnetosphere, space weather

History:
- Alfven (1942): Discovery of MHD waves -> 1970 Nobel Prize in Physics
```

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
mu0 = 4 * np.pi * 1e-7  # Vacuum permeability [H/m]
eps0 = 8.854e-12        # Vacuum permittivity [F/m]
c = 299792458           # Speed of light [m/s]
kB = 1.381e-23          # Boltzmann constant [J/K]
me = 9.109e-31          # Electron mass [kg]
mp = 1.673e-27          # Proton mass [kg]
e = 1.602e-19           # Elementary charge [C]

def mhd_introduction():
    """MHD Overview"""

    print("=" * 60)
    print("Magnetohydrodynamics (MHD) Overview")
    print("=" * 60)

    intro = """
    Core Concepts of MHD:

    1. Conducting Fluids:
       - Plasma, liquid metals, salt water
       - Free charges respond to electromagnetic fields
       - Electrical conductivity sigma > 0

    2. Fluid-Magnetic Field Interaction:
       - Magnetic field exerts force on fluid motion (Lorentz force)
       - Fluid motion changes magnetic field (induction)

    3. Coupled Equations:
       - Fluid dynamics: Continuity, momentum, energy
       - Electromagnetism: Maxwell's equations (subset)

    +-----------------------------------------------------+
    |                   MHD Domain                        |
    |                                                     |
    |    [Fluid Dynamics]  <---- Coupling --->  [EM]      |
    |                                                     |
    |    rho, v, p          J = sigma(E + v x B)   E, B   |
    |    Continuity/         Ohm's law           Maxwell  |
    |    Momentum                                (subset) |
    |    Energy                                           |
    +-----------------------------------------------------+
    """
    print(intro)

mhd_introduction()
```

### 1.2 MHD Time/Space Scales

```python
def mhd_scales():
    """Comparison of MHD-related time/space scales"""

    # Example: Solar corona plasma
    n = 1e15       # Density [m^-3]
    T = 1e6        # Temperature [K]
    B = 1e-2       # Magnetic field [T]
    L = 1e8        # Characteristic length [m]

    # Plasma frequency
    omega_pe = np.sqrt(n * e**2 / (eps0 * me))
    omega_pi = np.sqrt(n * e**2 / (eps0 * mp))

    # Cyclotron frequency
    omega_ce = e * B / me
    omega_ci = e * B / mp

    # Debye length
    lambda_D = np.sqrt(eps0 * kB * T / (n * e**2))

    # Thermal velocity
    v_te = np.sqrt(2 * kB * T / me)
    v_ti = np.sqrt(2 * kB * T / mp)

    # Alfven velocity
    rho = n * mp
    v_A = B / np.sqrt(mu0 * rho)

    # Sound speed
    gamma = 5/3
    p = n * kB * T
    c_s = np.sqrt(gamma * p / rho)

    print("=" * 60)
    print("Plasma Scales (Solar Corona Example)")
    print("=" * 60)
    print(f"\nInput Parameters:")
    print(f"  Density n = {n:.2e} m^-3")
    print(f"  Temperature T = {T:.2e} K")
    print(f"  Magnetic field B = {B*1000:.1f} mT")
    print(f"  Characteristic length L = {L/1e6:.0f} Mm")

    print(f"\nFrequencies:")
    print(f"  Electron plasma frequency omega_pe = {omega_pe:.2e} rad/s")
    print(f"  Ion plasma frequency omega_pi = {omega_pi:.2e} rad/s")
    print(f"  Electron cyclotron omega_ce = {omega_ce:.2e} rad/s")
    print(f"  Ion cyclotron omega_ci = {omega_ci:.2e} rad/s")

    print(f"\nVelocities:")
    print(f"  Electron thermal velocity v_te = {v_te/1e6:.2f} Mm/s")
    print(f"  Ion thermal velocity v_ti = {v_ti/1e3:.2f} km/s")
    print(f"  Alfven velocity v_A = {v_A/1e3:.2f} km/s")
    print(f"  Sound speed c_s = {c_s/1e3:.2f} km/s")

    print(f"\nLength Scales:")
    print(f"  Debye length lambda_D = {lambda_D:.4f} m")
    print(f"  Electron inertial length c/omega_pe = {c/omega_pe:.4f} m")
    print(f"  Ion inertial length c/omega_pi = {c/omega_pi:.2f} m")

    # Check MHD validity conditions
    print(f"\nMHD Validity Conditions:")
    print(f"  L >> lambda_D: {L:.2e} >> {lambda_D:.4f} {'OK' if L > 1000*lambda_D else 'Check needed'}")
    print(f"  L >> c/omega_pi: {L:.2e} >> {c/omega_pi:.2f} {'OK' if L > 100*c/omega_pi else 'Check needed'}")

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    scales = {
        'lambda_D': lambda_D,
        'c/omega_pe': c/omega_pe,
        'c/omega_pi': c/omega_pi,
        'v_A/omega_ci': v_A/omega_ci,
        'L (MHD)': L
    }

    y_pos = np.arange(len(scales))
    values = list(scales.values())
    labels = list(scales.keys())

    ax.barh(y_pos, np.log10(values), color=['red', 'orange', 'yellow', 'green', 'blue'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('log10(Length [m])')
    ax.set_title('Plasma Length Scale Comparison')
    ax.axvline(x=np.log10(L), color='black', linestyle='--', label='MHD scale')
    ax.grid(True, alpha=0.3)

    for i, v in enumerate(values):
        ax.text(np.log10(v) + 0.1, i, f'{v:.2e} m', va='center')

    plt.tight_layout()
    plt.savefig('mhd_scales.png', dpi=150, bbox_inches='tight')
    plt.show()

# mhd_scales()
```

---

## 2. MHD Assumptions

### 2.1 Basic Assumptions

```
Core MHD Assumptions:

1. Quasi-neutrality:
   n_i ~ n_e = n
   - Valid at scales larger than Debye length
   - Charge separation neglected

2. Low-frequency approximation:
   omega << omega_ci << omega_ce
   - Displacement current neglected: dD/dt ~ 0
   - Electron inertia neglected

3. Fluid approximation:
   L >> lambda_mfp (mean free path)
   - Local thermal equilibrium assumed
   - Kinetic effects neglected

4. Non-relativistic:
   v << c
   - Relativistic corrections unnecessary

Results:
- Maxwell's equations simplified
- Electric field derived from magnetic field and velocity
- Magnetic induction equation derived
```

```python
def mhd_assumptions():
    """Physical meaning of MHD assumptions"""

    print("=" * 60)
    print("MHD Assumptions and Maxwell's Equations Simplification")
    print("=" * 60)

    assumptions = """
    Maxwell's Equations:
    (1) div E = rho_c/eps0      <- MHD: quasi-neutrality, rho_c ~ 0
    (2) div B = 0               <- Unchanged
    (3) curl E = -dB/dt         <- Unchanged
    (4) curl B = mu0 J + mu0 eps0 dE/dt  <- Displacement current neglected

    Simplified Maxwell's Equations (MHD):
    (1') div E ~ 0 (quasi-neutrality)
    (2') div B = 0
    (3') curl E = -dB/dt
    (4') curl B = mu0 J  (Ampere's law)

    Generalized Ohm's Law:
    E + v x B = eta J + (J x B)/ne - grad(p_e)/ne + (m_e/ne^2) dJ/dt
                 |       |            |               |
              Resistive  Hall    Electron pressure  Electron inertia

    Ideal MHD:
    E + v x B = 0  (all RHS terms neglected)
    -> Magnetic field "frozen" to fluid

    Resistive MHD:
    E + v x B = eta J  (only resistive effect included)
    -> Magnetic diffusion and reconnection possible
    """
    print(assumptions)

mhd_assumptions()
```

---

## 3. Ideal MHD Equations

### 3.1 Governing Equations

```
Ideal MHD Equation System:

1. Mass Conservation:
   d rho/dt + div(rho v) = 0

2. Momentum Conservation:
   rho(dv/dt + (v . grad)v) = -grad p + J x B + rho g
                                   |
                               Lorentz force

   J x B = (curl B) x B / mu0 = (B . grad)B/mu0 - grad(B^2/2mu0)
                                    |               |
                              Magnetic tension  Magnetic pressure

3. Energy Conservation (adiabatic):
   d/dt(p/rho^gamma) + v . grad(p/rho^gamma) = 0

   Or: dp/dt + v . grad p + gamma p div v = 0

4. Induction Equation:
   dB/dt = curl(v x B)

   (E = -v x B substituted)

5. Divergence Constraint:
   div B = 0 (always maintained)
```

```python
def ideal_mhd_equations():
    """Ideal MHD equations visualization"""

    print("=" * 60)
    print("Ideal MHD Equation System")
    print("=" * 60)

    equations = """
    Conservative Form:

    dU/dt + div F = S

    Where:
    +-------------------------------------------------------------+
    | Conserved Variables U:                                       |
    |   U = [rho, rho v, B, E]^T                                  |
    |   E = p/(gamma-1) + rho v^2/2 + B^2/2mu0 (total energy)     |
    +-------------------------------------------------------------+
    | Flux F (x direction):                                        |
    |   F1 = rho vx                        (mass)                 |
    |   F2 = rho vx v - Bx B/mu0 + P* I   (momentum)              |
    |   F3 = vx B - Bx v                   (magnetic field)       |
    |   F4 = (E + P*) vx - Bx(v . B)/mu0  (energy)                |
    |                                                              |
    |   P* = p + B^2/2mu0 (total pressure)                        |
    +-------------------------------------------------------------+
    | 8 variables: rho, vx, vy, vz, Bx, By, Bz, p                 |
    | 8 equations: continuity(1), momentum(3), induction(3),      |
    |              energy(1)                                       |
    | + constraint: div B = 0                                      |
    +-------------------------------------------------------------+
    """
    print(equations)

    # Lorentz force visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) Magnetic pressure
    ax1 = axes[0]

    # Uniform magnetic field region and exterior
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)

    # Magnetic field (z direction, strong in central region)
    Bz = np.exp(-(X**2 + Y**2))

    # Magnetic pressure grad(B^2/2mu0)
    B_pressure = Bz**2 / (2 * mu0)
    grad_Bp_x, grad_Bp_y = np.gradient(B_pressure, x[1]-x[0])

    im = ax1.pcolormesh(X, Y, B_pressure * 1e6, cmap='hot', shading='auto')
    plt.colorbar(im, ax=ax1, label=r'$B^2/2\mu_0$ [uPa]')

    # Show magnetic pressure gradient (force)
    skip = 5
    ax1.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              -grad_Bp_x[::skip, ::skip], -grad_Bp_y[::skip, ::skip],
              color='white', alpha=0.8)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(r'Magnetic Pressure $-\nabla(B^2/2\mu_0)$: Outward Force')
    ax1.set_aspect('equal')

    # (2) Magnetic tension
    ax2 = axes[1]

    # Curved magnetic field lines
    theta = np.linspace(0, 2*np.pi, 100)
    for r in [0.5, 1.0, 1.5]:
        x_line = r * (1 + 0.3 * np.sin(2*theta)) * np.cos(theta)
        y_line = r * (1 + 0.3 * np.sin(2*theta)) * np.sin(theta)
        ax2.plot(x_line, y_line, 'b-', linewidth=1.5)

    # Show tension direction (toward center of curvature)
    for r in [1.0]:
        theta_arrows = np.linspace(0, 2*np.pi, 8, endpoint=False)
        for t in theta_arrows:
            x0 = r * (1 + 0.3 * np.sin(2*t)) * np.cos(t)
            y0 = r * (1 + 0.3 * np.sin(2*t)) * np.sin(t)

            # Center of curvature direction (approximate)
            dx = -x0 * 0.3
            dy = -y0 * 0.3
            ax2.annotate('', xy=(x0+dx, y0+dy), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(r'Magnetic Tension $(B\cdot\nabla)B/\mu_0$: Toward Curvature Center')
    ax2.set_aspect('equal')
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.grid(True, alpha=0.3)

    # Legend
    ax2.plot([], [], 'b-', linewidth=2, label='Magnetic field lines')
    ax2.plot([], [], 'r-', linewidth=2, label='Tension direction')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('mhd_lorentz_force.png', dpi=150, bbox_inches='tight')
    plt.show()

# ideal_mhd_equations()
```

---

## 4. Alfven Velocity

### 4.1 Definition and Physical Meaning

The Alfven velocity is the fundamental speed scale of MHD, analogous to the speed of sound in fluid mechanics. It represents the speed at which transverse perturbations propagate along magnetic field lines:

$$v_A = \frac{B}{\sqrt{\mu_0 \rho}}$$

where $B$ is the magnetic field strength, $\mu_0$ is the vacuum permeability, and $\rho$ is the mass density. The key insight: it arises from the balance between magnetic tension (the restoring force, $\sim B^2/\mu_0$) and inertia ($\sim \rho v^2$). The plasma beta $\beta = 2\mu_0 p / B^2$ compares thermal to magnetic pressure and determines which force dominates the dynamics.

```
Alfven Velocity:

v_A = B / sqrt(mu0 rho)

Physical Meaning:
- Speed of transverse waves propagating along magnetic field lines
- Equipartition of magnetic and kinetic energy
- B^2/2mu0 ~ rho v_A^2/2

Dimensionless Parameters:
- Alfven Mach number: M_A = v/v_A
- Plasma beta: beta = 2 mu0 p / B^2 = (c_s/v_A)^2 * 2/gamma

  beta << 1: Magnetic pressure dominated (solar corona)
  beta >> 1: Thermal pressure dominated (solar convection zone)
  beta ~ 1: Both important
```

```python
def alfven_velocity_analysis():
    """Alfven velocity analysis"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) Alfven velocity in various environments
    ax1 = axes[0]

    # Environment parameters
    environments = {
        'Solar corona': {'n': 1e14, 'B': 0.01, 'T': 1e6},
        'Solar wind (1AU)': {'n': 5e6, 'B': 5e-9, 'T': 1e5},
        'Interstellar medium': {'n': 1e6, 'B': 3e-10, 'T': 1e4},
        'Tokamak': {'n': 1e20, 'B': 5, 'T': 1e8},
        'Liquid sodium': {'n': 2.5e28, 'B': 0.1, 'T': 400}  # Density converted to particle number
    }

    names = []
    v_A_values = []
    v_s_values = []

    for name, params in environments.items():
        n = params['n']
        B = params['B']
        T = params['T']

        # Ion mass (proton for plasma, Na for sodium)
        if 'sodium' in name:
            m_ion = 23 * mp  # Na mass
            rho = n * m_ion
        else:
            m_ion = mp
            rho = n * m_ion

        # Alfven velocity
        v_A = B / np.sqrt(mu0 * rho)

        # Sound speed
        gamma = 5/3
        p = n * kB * T
        c_s = np.sqrt(gamma * p / rho)

        names.append(name)
        v_A_values.append(v_A)
        v_s_values.append(c_s)

    y_pos = np.arange(len(names))

    ax1.barh(y_pos - 0.2, np.log10(v_A_values), 0.4, label=r'$v_A$', color='blue')
    ax1.barh(y_pos + 0.2, np.log10(v_s_values), 0.4, label=r'$c_s$', color='red')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names)
    ax1.set_xlabel('log10(velocity [m/s])')
    ax1.set_title('Alfven Velocity and Sound Speed in Various Environments')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2) Plasma beta regime
    ax2 = axes[1]

    B_range = np.logspace(-10, 1, 100)  # Magnetic field range
    n_values = [1e6, 1e14, 1e20]  # Density
    T = 1e6  # Fixed temperature

    colors = ['blue', 'green', 'red']
    for n, color in zip(n_values, colors):
        rho = n * mp
        p = n * kB * T

        # Plasma beta
        beta = 2 * mu0 * p / B_range**2

        ax2.loglog(B_range, beta, color=color, linewidth=2, label=f'n = {n:.0e} m^-3')

    ax2.axhline(y=1, color='black', linestyle='--', label=r'$\beta = 1$')
    ax2.fill_between([1e-10, 1e1], 1e-6, 1, alpha=0.2, color='blue', label=r'$\beta < 1$ (magnetic pressure dominated)')
    ax2.fill_between([1e-10, 1e1], 1, 1e6, alpha=0.2, color='red', label=r'$\beta > 1$ (thermal pressure dominated)')

    ax2.set_xlabel('B [T]')
    ax2.set_ylabel(r'$\beta = 2\mu_0 p / B^2$')
    ax2.set_title(f'Plasma Beta (T = {T:.0e} K)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(1e-6, 1e6)
    ax2.set_xlim(1e-10, 1e1)

    plt.tight_layout()
    plt.savefig('alfven_velocity.png', dpi=150, bbox_inches='tight')
    plt.show()

# alfven_velocity_analysis()
```

---

## 5. MHD Waves

### 5.1 Wave Types

```
Three Wave Modes in Ideal MHD:

1. Alfven Wave (Shear Alfven Wave):
   - Velocity: v_A = B0/sqrt(mu0 rho)
   - Direction: Propagates only along magnetic field
   - Characteristics: Transverse, incompressible, magnetic field line oscillation
   - Velocity perturbation: delta v perpendicular to B0, k

2. Fast Magnetosonic Wave:
   - Velocity: v_f = sqrt[(v_A^2 + c_s^2)/2 + sqrt((v_A^2 + c_s^2)^2 - 4 v_A^2 c_s^2 cos^2(theta))/2]
   - Characteristics: Magnetic + thermal pressure restoring force
   - Isotropic propagation (all directions)

3. Slow Magnetosonic Wave:
   - Velocity: v_s = sqrt[(v_A^2 + c_s^2)/2 - sqrt((v_A^2 + c_s^2)^2 - 4 v_A^2 c_s^2 cos^2(theta))/2]
   - Characteristics: Magnetic and thermal pressure oppose each other
   - Propagates near magnetic field direction

Where theta = angle(k, B0)

Special Cases:
- theta = 0 (parallel): v_f = max(v_A, c_s), v_s = min(v_A, c_s)
- theta = pi/2 (perpendicular): v_f = sqrt(v_A^2 + c_s^2), v_s = 0
```

```python
def mhd_wave_speeds():
    """MHD wave speed dispersion relation"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) Wave speed vs propagation angle
    ax1 = axes[0]

    theta = np.linspace(0, np.pi/2, 100)

    # v_A/c_s ratio (related to plasma beta)
    ratios = [0.5, 1.0, 2.0]  # v_A/c_s

    for ratio in ratios:
        vA = ratio
        cs = 1.0

        # Fast/slow waves
        term1 = (vA**2 + cs**2) / 2
        term2 = np.sqrt((vA**2 + cs**2)**2 - 4 * vA**2 * cs**2 * np.cos(theta)**2) / 2

        v_fast = np.sqrt(term1 + term2)
        v_slow = np.sqrt(np.maximum(term1 - term2, 0))

        # Alfven wave (component)
        v_alfven = np.abs(vA * np.cos(theta))

        ax1.plot(np.degrees(theta), v_fast, '-', linewidth=2, label=f'Fast (vA/cs={ratio})')
        ax1.plot(np.degrees(theta), v_slow, '--', linewidth=2, label=f'Slow (vA/cs={ratio})')
        ax1.plot(np.degrees(theta), v_alfven, ':', linewidth=2, label=f'Alfven (vA/cs={ratio})')

    ax1.set_xlabel('theta [degrees]')
    ax1.set_ylabel('Phase velocity / cs')
    ax1.set_title('MHD Wave Speed vs Propagation Angle')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 90)

    # (2) Friedrichs diagram (polar coordinates)
    ax2 = axes[1]

    theta_full = np.linspace(0, 2*np.pi, 360)

    vA = 2.0
    cs = 1.0

    term1 = (vA**2 + cs**2) / 2
    term2 = np.sqrt((vA**2 + cs**2)**2 - 4 * vA**2 * cs**2 * np.cos(theta_full)**2) / 2

    v_fast = np.sqrt(term1 + term2)
    v_slow = np.sqrt(np.maximum(term1 - term2, 0))
    v_alfven = np.abs(vA * np.cos(theta_full))

    # Polar -> Cartesian
    x_fast = v_fast * np.sin(theta_full)
    y_fast = v_fast * np.cos(theta_full)

    x_slow = v_slow * np.sin(theta_full)
    y_slow = v_slow * np.cos(theta_full)

    x_alf = v_alfven * np.sin(theta_full)
    y_alf = v_alfven * np.cos(theta_full)

    ax2.plot(x_fast, y_fast, 'b-', linewidth=2, label='Fast')
    ax2.plot(x_slow, y_slow, 'r-', linewidth=2, label='Slow')
    ax2.plot(x_alf, y_alf, 'g--', linewidth=2, label='Alfven')

    # Show B0 direction
    ax2.annotate('', xy=(0, 3), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax2.text(0.2, 2.8, r'$B_0$', fontsize=14)

    ax2.set_xlabel(r'$v_\perp / c_s$')
    ax2.set_ylabel(r'$v_\parallel / c_s$')
    ax2.set_title(f'Friedrichs Diagram (vA/cs = {vA/cs})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)

    plt.tight_layout()
    plt.savefig('mhd_waves.png', dpi=150, bbox_inches='tight')
    plt.show()

# mhd_wave_speeds()
```

### 5.2 Alfven Wave Visualization

```python
def alfven_wave_visualization():
    """Alfven wave visualization"""

    fig = plt.figure(figsize=(14, 10))

    # (1) Alfven wave concept (3D)
    ax1 = fig.add_subplot(221, projection='3d')

    z = np.linspace(0, 4*np.pi, 100)
    t = 0

    # Equilibrium magnetic field direction (z)
    B0 = 1.0

    # Perturbation (y direction oscillation)
    k = 1
    omega = k  # Normalized vA = 1
    By = 0.3 * np.sin(k*z - omega*t)

    # Field line position
    x_line = np.zeros_like(z)
    y_line = By

    ax1.plot(x_line, y_line, z, 'b-', linewidth=2, label='Magnetic field line')
    ax1.plot([0]*len(z), [0]*len(z), z, 'k--', alpha=0.5, label='Equilibrium position')

    # Velocity perturbation
    vy = -0.3 * np.sin(k*z - omega*t)  # v proportional to -B (Alfven relation)
    skip = 10
    ax1.quiver(x_line[::skip], y_line[::skip], z[::skip],
              np.zeros(len(z[::skip])), vy[::skip], np.zeros(len(z[::skip])),
              color='red', length=0.5, arrow_length_ratio=0.3, label='Velocity perturbation')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z (B0 direction)')
    ax1.set_title('Alfven Wave: Transverse Magnetic Field Line Oscillation')
    ax1.legend()

    # (2) Time evolution
    ax2 = fig.add_subplot(222)

    z = np.linspace(0, 4*np.pi, 200)
    times = [0, 0.5, 1.0, 1.5, 2.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))

    for t, color in zip(times, colors):
        By = 0.3 * np.sin(k*z - omega*t)
        ax2.plot(z, By, color=color, linewidth=1.5, label=f't = {t:.1f}')

    ax2.set_xlabel('z')
    ax2.set_ylabel(r'$\delta B_y$')
    ax2.set_title('Alfven Wave Propagation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # (3) Energy partition
    ax3 = fig.add_subplot(223)

    # Magnetic energy and kinetic energy
    z = np.linspace(0, 4*np.pi, 200)
    t = 0.5

    B_pert = 0.3 * np.sin(k*z - omega*t)
    v_pert = -0.3 * np.sin(k*z - omega*t)

    # Energy density (ignoring units, proportional relationship only)
    E_mag = B_pert**2 / 2  # proportional to B^2/2mu0
    E_kin = v_pert**2 / 2  # proportional to rho v^2/2

    ax3.plot(z, E_mag, 'b-', linewidth=2, label=r'$\delta B^2/2\mu_0$ (magnetic)')
    ax3.plot(z, E_kin, 'r--', linewidth=2, label=r'$\rho\delta v^2/2$ (kinetic)')
    ax3.plot(z, E_mag + E_kin, 'k-', linewidth=2, label='Total')

    ax3.set_xlabel('z')
    ax3.set_ylabel('Energy density')
    ax3.set_title('Alfven Wave Energy Equipartition')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # (4) Perturbation relations
    ax4 = fig.add_subplot(224)

    # delta v = -delta B/sqrt(mu0 rho) (Alfven relation)
    info_text = """
Alfven Wave Characteristics:

1. Propagation direction: B0 direction (k parallel to B0)
2. Polarization: Transverse (delta v perpendicular to B0, delta B perpendicular to B0)
3. Velocity: v_A = B0/sqrt(mu0 rho)

Perturbation Relation:
delta v = +/- delta B/sqrt(mu0 rho)
(+/- for k . B0 greater/less than 0)

Features:
- Incompressible: div(delta v) = 0
- No density/pressure perturbation
- Magnetic field lines oscillate like "guitar strings"
- Tension provides restoring force

Alfven Theorem (Frozen-in Condition):
In ideal MHD, magnetic field lines
move with the fluid
("frozen-in" condition)
    """
    ax4.text(0.1, 0.95, info_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig('alfven_wave_detail.png', dpi=150, bbox_inches='tight')
    plt.show()

# alfven_wave_visualization()
```

---

## 6. Magnetic Pressure and Magnetic Tension

### 6.1 Force Decomposition

```
Lorentz Force Decomposition:

J x B = (1/mu0)(curl B) x B

Using vector identity:
(curl B) x B = (B . grad)B - grad(B^2/2)

Therefore:
J x B = (B . grad)B/mu0 - grad(B^2/2mu0)
            |                |
       Magnetic tension  Magnetic pressure gradient

1. Magnetic Pressure:
   p_m = B^2/2mu0

   - Isotropic (same in all directions)
   - Force from high B to low B regions
   - Dense field lines -> high pressure

2. Magnetic Tension:
   T = (B . grad)B/mu0 = (B^2/mu0) kappa

   - Direction toward curvature center kappa
   - Force to straighten curved field lines
   - Similar to "guitar string tension"
```

```python
def magnetic_pressure_tension():
    """Magnetic pressure and tension equilibrium examples"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (1) Magnetic pressure equilibrium (Z-pinch)
    ax1 = axes[0]

    r = np.linspace(0.1, 2, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    R, Theta = np.meshgrid(r, theta)

    # Magnetic field (theta direction field from axial current)
    # B_theta proportional to 1/r (exterior), B_theta proportional to r (interior)
    r_plasma = 1.0
    B_theta = np.where(R < r_plasma, R, 1/R)

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    # Magnetic pressure
    P_mag = B_theta**2 / 2

    im = ax1.pcolormesh(X, Y, P_mag, cmap='hot', shading='auto')
    plt.colorbar(im, ax=ax1, label=r'$B^2/2\mu_0$')

    # Magnetic field lines (concentric circles)
    for r_line in [0.3, 0.6, 0.9, 1.2, 1.5]:
        circle = plt.Circle((0, 0), r_line, fill=False, color='blue', linewidth=1)
        ax1.add_patch(circle)

    # Pressure gradient direction
    ax1.annotate('', xy=(0.7, 0), xytext=(0.3, 0),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax1.annotate('', xy=(1.7, 0), xytext=(1.3, 0),
                arrowprops=dict(arrowstyle='<-', color='white', lw=2))

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Z-pinch: Magnetic Pressure Compresses Plasma')
    ax1.set_aspect('equal')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)

    # (2) Magnetic tension (curved field lines)
    ax2 = axes[1]

    # Curved magnetic field lines
    x = np.linspace(-2, 2, 100)
    y_lines = [0.3 * np.sin(np.pi * x / 2),
              0.6 * np.sin(np.pi * x / 2),
              0.9 * np.sin(np.pi * x / 2)]

    for y in y_lines:
        ax2.plot(x, y, 'b-', linewidth=2)

    # Tension direction (toward center of curvature = downward)
    x_arrows = [-1, 0, 1]
    for xa in x_arrows:
        idx = np.argmin(np.abs(x - xa))
        y_arrow = 0.6 * np.sin(np.pi * xa / 2)
        # If curvature is positive, tension is downward
        tension_dir = -1 if xa == 0 else (-0.5 if xa > 0 else 0.5)
        ax2.annotate('', xy=(xa, y_arrow + tension_dir * 0.3),
                    xytext=(xa, y_arrow),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Magnetic Tension: Force to Straighten Curved Field Lines')
    ax2.set_aspect('equal')
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, alpha=0.3)

    # Legend
    ax2.plot([], [], 'b-', linewidth=2, label='Magnetic field lines')
    ax2.plot([], [], 'r-', linewidth=2, label='Tension direction')
    ax2.legend()

    # (3) Equilibrium example
    ax3 = axes[2]

    info_text = """
MHD Equilibrium Condition:

In static equilibrium:
grad p = J x B = (B . grad)B/mu0 - grad(B^2/2mu0)

Rearranged:
grad(p + B^2/2mu0) = (B . grad)B/mu0
       |                   |
   Total pressure    Magnetic tension

Application Examples:

1. Theta-pinch:
   - Only Bz exists (straight)
   - No tension, pressure equilibrium
   - d/dz(p + B^2/2mu0) = 0

2. Z-pinch:
   - Only B_theta exists (circular)
   - Pressure + tension balance thermal pressure
   - (1/r) d/dr[r(p + B^2/2mu0)] = B_theta^2/mu0 r

3. Screw pinch:
   - Bz + B_theta combination
   - Complex equilibrium conditions
   - Basic form of tokamak
    """
    ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig('magnetic_pressure_tension.png', dpi=150, bbox_inches='tight')
    plt.show()

# magnetic_pressure_tension()
```

---

## 7. Frozen-in Theorem

### 7.1 Magnetic Field Frozen-in Condition

```
Magnetic Field Line Frozen-in (Frozen-in Theorem):

In ideal MHD:
E + v x B = 0  (no resistance)

Induction equation:
dB/dt = curl(v x B)

Physical Meaning:
1. Magnetic field lines move together with fluid elements
2. If you "tag" a field line, it moves with the fluid
3. If two fluid elements are on the same field line,
   they remain on the same field line forever

Magnetic Flux Conservation:
d/dt integral B . dS = 0  (for moving surface)

Violation Condition (Resistive MHD):
E + v x B = eta J

Induction equation:
dB/dt = curl(v x B) + eta/mu0 grad^2 B
                           |
                    Magnetic diffusion

Magnetic Reynolds Number:
Rm = mu0 v L / eta

Rm >> 1: Frozen-in condition valid (most astrophysical plasmas)
Rm ~ 1: Diffusion and convection compete
Rm << 1: Diffusion dominated
```

```python
def frozen_in_theorem():
    """Frozen-in theorem visualization"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (1) Ideal MHD: Field lines move with fluid
    ax1 = axes[0, 0]

    # Initial grid and field lines
    x = np.linspace(0, 2, 6)
    y = np.linspace(0, 1, 6)

    # Initial state
    for xi in x:
        ax1.plot([xi, xi], [0, 1], 'b-', linewidth=1, alpha=0.5)
    for yi in y:
        ax1.plot([0, 2], [yi, yi], 'b-', linewidth=1, alpha=0.5)

    # After deformation (shear flow)
    # v = (y, 0) -> x' = x + t*y
    t = 0.5
    for yi in y:
        x_new = x + t * yi
        ax1.plot(x_new, np.full_like(x_new, yi), 'r--', linewidth=1, alpha=0.7)

    for xi in x:
        y_line = np.linspace(0, 1, 20)
        x_line = xi + t * y_line
        ax1.plot(x_line, y_line, 'r--', linewidth=1, alpha=0.7)

    ax1.plot([], [], 'b-', linewidth=2, label='Initial (field lines)')
    ax1.plot([], [], 'r--', linewidth=2, label='After deformation')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Ideal MHD: Field Lines Frozen to Fluid')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # (2) Magnetic flux conservation
    ax2 = axes[0, 1]

    theta = np.linspace(0, 2*np.pi, 100)

    # Initial circular loop
    r0 = 1
    x0 = r0 * np.cos(theta)
    y0 = r0 * np.sin(theta)
    ax2.plot(x0, y0, 'b-', linewidth=2, label='Initial loop')
    ax2.fill(x0, y0, alpha=0.2, color='blue')

    # Deformed loop (compression)
    rx, ry = 0.5, 2.0  # Compression/extension
    x1 = rx * np.cos(theta)
    y1 = ry * np.sin(theta)
    ax2.plot(x1, y1, 'r-', linewidth=2, label='Deformed loop')
    ax2.fill(x1, y1, alpha=0.2, color='red')

    # Area comparison
    A0 = np.pi * r0**2
    A1 = np.pi * rx * ry

    ax2.text(0, 0, 'Phi = int B.dA\nConserved!', ha='center', va='center', fontsize=11)
    ax2.text(1.5, 0, f'A0 = {A0:.2f}', fontsize=10)
    ax2.text(0.3, 1.5, f'A1 = {A1:.2f}', fontsize=10)

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Magnetic Flux Conservation')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.grid(True, alpha=0.3)

    # (3) Magnetic Reynolds number
    ax3 = axes[1, 0]

    # Rm for various environments
    environments = {
        'Laboratory\nplasma': 1e2,
        'Solar\nphotosphere': 1e6,
        'Solar\ncorona': 1e12,
        'Interstellar\nmedium': 1e18,
        'Liquid\nmetal': 1e1
    }

    names = list(environments.keys())
    Rm_values = list(environments.values())

    y_pos = np.arange(len(names))
    colors = ['red' if Rm < 100 else 'green' for Rm in Rm_values]

    ax3.barh(y_pos, np.log10(Rm_values), color=colors)
    ax3.axvline(x=np.log10(100), color='black', linestyle='--', label=r'$R_m = 100$')

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names)
    ax3.set_xlabel(r'log10($R_m$)')
    ax3.set_title('Magnetic Reynolds Number Comparison')
    ax3.grid(True, alpha=0.3)

    # Legend
    ax3.plot([], [], 'g-', linewidth=10, label='Frozen-in valid')
    ax3.plot([], [], 'r-', linewidth=10, label='Diffusion important')
    ax3.legend()

    # (4) Concept summary
    ax4 = axes[1, 1]

    info_text = """
Frozen-in Theorem Summary:

Condition: Ideal MHD (eta = 0, E + v x B = 0)

Results:
1. dB/dt = curl(v x B)
2. d/dt integral B . dS = 0 (moving surface)
3. Field lines move with fluid

Physical Interpretation:
- Fluid elements "frozen" to field lines
- Magnetic field compression <-> density increase
- B/rho proportional to constant (1D compression)

Violation (eta != 0):
- Magnetic diffusion: tau_diff = mu0 L^2/eta
- Magnetic reconnection possible
- Energy conversion (magnetic -> kinetic/thermal)

Importance:
- Solar flares: reconnection
- Tokamak: frozen-in important
- Magnetosphere: reconnection phenomena
    """
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig('frozen_in_theorem.png', dpi=150, bbox_inches='tight')
    plt.show()

# frozen_in_theorem()
```

---

## 8. Practice Problems

### Exercise 1: Alfven Velocity
Calculate the Alfven velocity for the solar corona with B = 10 G, n = 10^8 cm^-3. Compare with the sound speed (T = 10^6 K) and find the plasma beta.

### Exercise 2: MHD Wave Speed
For v_A = 2c_s, find the phase velocity of the fast magnetosonic wave propagating perpendicular to the magnetic field (theta = 90 degrees).

### Exercise 3: Magnetic Pressure Equilibrium
Find the pressure equilibrium condition at the boundary between a uniform magnetic field Bz region and a field-free region.

### Exercise 4: Frozen-in
Calculate the magnetic diffusion time for a plasma with length L = 1 Mm and conductivity sigma = 10^6 S/m. What is the magnetic Reynolds number when velocity v = 100 km/s?

---

## 9. References

### Core Textbooks
- Goedbloed & Poedts, "Principles of Magnetohydrodynamics"
- Kulsrud, "Plasma Physics for Astrophysics"
- Freidberg, "Ideal MHD"

### Papers/Reviews
- Alfven (1942) original paper (MHD waves)
- Priest & Forbes, "Magnetic Reconnection" (reconnection)

### Online Resources
- Thorne & Blandford, "Modern Classical Physics" (Ch. 19)
- Chen, "Introduction to Plasma Physics" (MHD chapter)

---

## Summary

```
MHD Basics Key Points:

1. MHD Assumptions:
   - Quasi-neutrality, low-frequency, fluid approximation
   - L >> lambda_D, c/omega_pi
   - Maxwell simplification

2. Ideal MHD Equations:
   - Continuity: d rho/dt + div(rho v) = 0
   - Momentum: rho Dv/Dt = -grad p + J x B
   - Energy: D(p/rho^gamma)/Dt = 0
   - Induction: dB/dt = curl(v x B)
   - Constraint: div B = 0

3. Key Velocities:
   - Alfven: v_A = B/sqrt(mu0 rho)
   - Sound: c_s = sqrt(gamma p/rho)
   - Plasma beta: beta = 2 mu0 p/B^2

4. MHD Waves:
   - Alfven wave: v_A (along field)
   - Fast magnetosonic: sqrt(v_A^2 + c_s^2) (perpendicular)
   - Slow magnetosonic: min(v_A, c_s) (parallel)

5. Lorentz Force:
   J x B = -grad(B^2/2mu0) + (B . grad)B/mu0
          Magnetic pressure   Magnetic tension

6. Frozen-in:
   - Ideal MHD: E + v x B = 0
   - Field lines frozen to fluid
   - Valid when Rm >> 1
```

---

## Exercises

### Exercise 1: Alfven Velocity and Plasma Beta
For the solar corona with magnetic field B = 10 G = 10⁻³ T and number density n = 10⁸ cm⁻³ = 10¹⁴ m⁻³ (proton plasma), calculate: (a) the Alfven velocity vA = B/√(μ₀ρ) where ρ = n·mp, (b) the sound speed cs = √(γp/ρ) at T = 10⁶ K, and (c) the plasma beta β = 2μ₀p/B². Classify this environment as magnetically dominated (β < 1) or thermally dominated (β > 1) and explain the physical implication for wave propagation.

### Exercise 2: MHD Wave Phase Velocities
For a plasma with vA = 2cs, compute the phase velocities of all three MHD wave modes (fast magnetosonic, Alfven, slow magnetosonic) at propagation angles θ = 0°, 30°, 60°, and 90° relative to the background magnetic field B₀. Use the dispersion relations given in Section 5.1. Plot the Friedrichs diagram showing all three wave mode surfaces in polar coordinates (v⊥ vs v‖). At which angle does the fast wave achieve its maximum speed?

### Exercise 3: Magnetic Pressure Equilibrium
A current sheet separates a region with magnetic field B₁ = 10 G and thermal pressure p₁ = 0.5 × 10⁻³ Pa from a field-free region (B₂ = 0) with thermal pressure p₂. Using total pressure balance p + B²/(2μ₀) = constant across the current sheet, calculate p₂. Then verify that the jump in total pressure is zero, which is the equilibrium condition.

### Exercise 4: Frozen-in Theorem and Magnetic Diffusion
For a resistive plasma with electrical conductivity σ = 10⁶ S/m and characteristic length L = 1 Mm = 10⁶ m, compute: (a) the magnetic diffusivity η = 1/(μ₀σ), (b) the magnetic diffusion timescale τdiff = μ₀σL², and (c) the magnetic Reynolds number Rm = μ₀σvL for a flow velocity v = 100 km/s. Is the frozen-in condition valid? Compare τdiff to the Alfven crossing time τA = L/vA using B = 10 G, n = 10¹⁴ m⁻³.

### Exercise 5: Ideal MHD Conservative Form
Write out explicitly all seven components of the conservative variable vector U and flux vector F(U) for 1D ideal MHD. Then verify that your expressions are dimensionally consistent by checking the units of each component. Starting from the induction equation dB/dt = ∇×(v×B), show algebraically that this is equivalent to the flux form used in the conservative MHD system.

---

The next lesson covers numerical methods for MHD equations.
