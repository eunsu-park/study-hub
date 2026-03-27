# 16. Projects

## Learning Objectives

- Implement a full 3D particle orbit integrator using the Boris algorithm with various field geometries
- Develop a general dispersion relation solver for cold and warm plasmas with visualization tools
- Create a 1D Vlasov-Poisson solver using semi-Lagrangian methods to study kinetic plasma phenomena
- Synthesize knowledge from single-particle orbits, wave theory, and kinetic theory
- Verify numerical results against analytical predictions and explore nonlinear physics beyond linear theory
- Gain practical experience with computational plasma physics methods used in research

## Project 1: Particle Orbit Simulator

### 1.1 Overview

**Goal**: Build a versatile 3D particle orbit integrator that can handle various electromagnetic field configurations and visualize particle trajectories, drifts, and invariants.

**Difficulty**: ⭐⭐⭐

**Time estimate**: 10–15 hours

**Skills developed**:
- Numerical integration of equations of motion
- Boris algorithm (leap-frog method for particles in E&M fields)
- 3D visualization
- Verification against analytical drift velocities
- Understanding of particle confinement and loss mechanisms

### 1.2 Physics Background

The relativistic equation of motion for a charged particle is:

$$\frac{d\mathbf{p}}{dt} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

where $\mathbf{p} = \gamma m \mathbf{v}$ is the momentum and $\gamma = 1/\sqrt{1 - v^2/c^2}$ is the Lorentz factor.

For non-relativistic particles ($v \ll c$):

$$m \frac{d\mathbf{v}}{dt} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

The **Boris algorithm** is a second-order accurate, time-reversible scheme that conserves energy in static fields. It's the workhorse of particle-in-cell (PIC) codes.

**Boris algorithm** (one time step $\Delta t$):

1. Half-accelerate by electric field:
   $$\mathbf{v}^{-} = \mathbf{v}^n + \frac{q \Delta t}{2m} \mathbf{E}$$

2. Rotate by magnetic field:
   $$\mathbf{v}^{+} = \mathbf{v}^{-} + \mathbf{v}^{-} \times \mathbf{t} + (\mathbf{v}^{-} \times \mathbf{t}) \times \mathbf{s}$$
   where:
   $$\mathbf{t} = \frac{q \Delta t}{2m} \mathbf{B}, \quad \mathbf{s} = \frac{2\mathbf{t}}{1 + t^2}$$

3. Half-accelerate by electric field:
   $$\mathbf{v}^{n+1} = \mathbf{v}^{+} + \frac{q \Delta t}{2m} \mathbf{E}$$

4. Update position:
   $$\mathbf{x}^{n+1} = \mathbf{x}^n + \mathbf{v}^{n+1} \Delta t$$

### 1.3 Implementation Guide

**Step 1: Basic Infrastructure**

Create a `Particle` class with attributes:
- `q`, `m`: charge and mass
- `r`: position vector [x, y, z]
- `v`: velocity vector [vx, vy, vz]
- `history`: lists to store trajectory

**Step 2: Field Configurations**

Implement functions to return $\mathbf{E}(\mathbf{r}, t)$ and $\mathbf{B}(\mathbf{r}, t)$ for various geometries:

1. **Uniform B**: $\mathbf{B} = B_0 \hat{\mathbf{z}}$ (gyration)
2. **Uniform E + B**: $\mathbf{E} = E_0 \hat{\mathbf{x}}$, $\mathbf{B} = B_0 \hat{\mathbf{z}}$ (E×B drift)
3. **Gradient B**: $\mathbf{B} = B_0 (1 + \alpha x) \hat{\mathbf{z}}$ (grad-B drift)
4. **Curved B**: $\mathbf{B} = \frac{B_0 R_0}{R_0 + x} \hat{\mathbf{z}}$ (curvature drift, simplified)
5. **Magnetic mirror**: $\mathbf{B} = B_0 (1 + (z/L)^2) \hat{\mathbf{z}}$ (mirror force)
6. **Tokamak (simplified)**: Toroidal + poloidal field

**Step 3: Boris Integrator**

Implement the Boris algorithm as a function:

```python
def boris_step(r, v, E, B, q, m, dt):
    """
    One step of Boris algorithm.

    Parameters:
    r, v: position and velocity (3D arrays)
    E, B: electric and magnetic fields at position r (3D arrays)
    q, m: charge and mass
    dt: time step

    Returns:
    r_new, v_new
    """
    # Half electric acceleration
    v_minus = v + (q * dt / (2 * m)) * E

    # Magnetic rotation
    t = (q * dt / (2 * m)) * B
    s = 2 * t / (1 + np.dot(t, t))
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)

    # Half electric acceleration
    v_new = v_plus + (q * dt / (2 * m)) * E

    # Position update
    r_new = r + v_new * dt

    return r_new, v_new
```

**Step 4: Integration Loop**

```python
def integrate_orbit(particle, E_func, B_func, t_final, dt):
    """
    Integrate particle orbit from t=0 to t=t_final.

    Parameters:
    particle: Particle object
    E_func, B_func: functions returning E(r, t) and B(r, t)
    t_final: final time
    dt: time step
    """
    t = 0
    while t < t_final:
        E = E_func(particle.r, t)
        B = B_func(particle.r, t)

        particle.r, particle.v = boris_step(particle.r, particle.v, E, B,
                                             particle.q, particle.m, dt)

        particle.history['t'].append(t)
        particle.history['r'].append(particle.r.copy())
        particle.history['v'].append(particle.v.copy())

        t += dt
```

**Step 5: Visualization**

Plot 3D trajectories using `matplotlib`:

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

r_history = np.array(particle.history['r'])
ax.plot(r_history[:, 0], r_history[:, 1], r_history[:, 2], linewidth=1)

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('Particle Trajectory')
plt.show()
```

**Step 6: Analysis and Verification**

For each field configuration, calculate:

1. **Gyroradius**: $\rho = m v_\perp / (q B)$
2. **Drift velocities**: Compare numerical with analytical:
   - E×B drift: $\mathbf{v}_E = \mathbf{E} \times \mathbf{B} / B^2$
   - Grad-B drift: $\mathbf{v}_{\nabla B} = \pm \frac{m v_\perp^2}{2 q B^3} \mathbf{B} \times \nabla B$
   - Curvature drift: $\mathbf{v}_\kappa = \frac{m v_\parallel^2}{q B^3} \mathbf{B} \times (\mathbf{B} \cdot \nabla)\mathbf{B}$
3. **Adiabatic invariants**:
   - Magnetic moment: $\mu = m v_\perp^2 / (2B)$
   - Longitudinal action: $J = \oint v_\parallel ds$ (for bounce motion)
4. **Energy conservation**: Check that $E_{kin} = \frac{1}{2} m v^2$ is conserved in static fields

### 1.4 Complete Code

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
e = 1.6e-19
m_e = 9.11e-31
m_p = 1.67e-27

class Particle:
    def __init__(self, q, m, r0, v0):
        self.q = q
        self.m = m
        self.r = np.array(r0, dtype=float)
        self.v = np.array(v0, dtype=float)
        self.history = {'t': [], 'r': [], 'v': []}

    def kinetic_energy(self):
        return 0.5 * self.m * np.dot(self.v, self.v)

    def magnetic_moment(self, B):
        B_mag = np.linalg.norm(B)
        v_perp = np.linalg.norm(self.v - np.dot(self.v, B) * B / B_mag**2 * B_mag)
        return self.m * v_perp**2 / (2 * B_mag)

def boris_step(r, v, E, B, q, m, dt):
    """Boris algorithm: one time step."""
    # Half electric push
    v_minus = v + (q * dt / (2 * m)) * E

    # Magnetic rotation
    # t_vec = (q dt/2m) B is half the rotation angle vector; the Boris trick
    # decomposes the full rotation into two cross-products to avoid computing
    # a full rotation matrix while remaining exactly time-reversible.
    t_vec = (q * dt / (2 * m)) * B
    t_mag_sq = np.dot(t_vec, t_vec)
    # s_vec = 2t/(1+|t|²) is the double-angle formula for the rotation:
    # this exact form conserves kinetic energy in a static B field to machine
    # precision, which is why Boris is preferred over simple Euler integration.
    s_vec = 2 * t_vec / (1 + t_mag_sq)

    v_prime = v_minus + np.cross(v_minus, t_vec)
    v_plus = v_minus + np.cross(v_prime, s_vec)

    # Half electric push
    v_new = v_plus + (q * dt / (2 * m)) * E

    # Position update using the already-updated velocity (leap-frog ordering)
    # ensures second-order accuracy in both position and velocity.
    r_new = r + v_new * dt

    return r_new, v_new

def integrate_orbit(particle, E_func, B_func, t_final, dt):
    """Integrate particle orbit."""
    t = 0
    while t < t_final:
        E = E_func(particle.r, t)
        B = B_func(particle.r, t)

        particle.r, particle.v = boris_step(particle.r, particle.v, E, B,
                                             particle.q, particle.m, dt)

        particle.history['t'].append(t)
        particle.history['r'].append(particle.r.copy())
        particle.history['v'].append(particle.v.copy())

        t += dt

# Field configurations
def uniform_B(r, t, B0=0.1):
    """Uniform magnetic field in z direction."""
    return np.array([0, 0, 0]), np.array([0, 0, B0])

def E_cross_B(r, t, E0=1000, B0=0.1):
    """Uniform E and B fields (E×B drift)."""
    return np.array([E0, 0, 0]), np.array([0, 0, B0])

def gradient_B(r, t, B0=0.1, alpha=0.1):
    """Gradient in B field."""
    x, y, z = r
    B = B0 * (1 + alpha * x)
    return np.array([0, 0, 0]), np.array([0, 0, B])

def magnetic_mirror(r, t, B0=0.1, L=1.0):
    """Magnetic mirror field."""
    x, y, z = r
    B_mag = B0 * (1 + (z / L)**2)
    # Simplified: B field in z direction with magnitude varying
    # For full mirror: need radial component too
    Bz = B_mag
    Br = -B0 * (z / L**2) * np.sqrt(x**2 + y**2)  # from div B = 0
    theta = np.arctan2(y, x)
    Bx = Br * np.cos(theta)
    By = Br * np.sin(theta)
    return np.array([0, 0, 0]), np.array([Bx, By, Bz])

# Test Case 1: Gyration in uniform B
print("Test 1: Gyration in uniform B field")
print("=" * 50)

B0 = 0.1  # T
v_perp = 1e6  # m/s
electron = Particle(q=-e, m=m_e, r0=[0, 0, 0], v0=[v_perp, 0, 0])

omega_c = e * B0 / m_e
rho_c = m_e * v_perp / (e * B0)
T_c = 2 * np.pi / omega_c

print(f"Cyclotron frequency: {omega_c:.3e} rad/s")
print(f"Gyroradius: {rho_c * 1e3:.3f} mm")
print(f"Cyclotron period: {T_c * 1e9:.3f} ns")

dt = T_c / 100
t_final = 3 * T_c

integrate_orbit(electron, lambda r, t: uniform_B(r, t, B0), lambda r, t: (np.zeros(3), np.zeros(3)),
                t_final, dt)

r_hist = np.array(electron.history['r'])

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121)
ax1.plot(r_hist[:, 0] * 1e3, r_hist[:, 1] * 1e3, 'b-', linewidth=1)
ax1.plot(r_hist[0, 0] * 1e3, r_hist[0, 1] * 1e3, 'go', markersize=8, label='Start')
ax1.set_xlabel('x (mm)')
ax1.set_ylabel('y (mm)')
ax1.set_title('Electron Gyration in Uniform B')
ax1.set_aspect('equal')
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(r_hist[:, 0] * 1e3, r_hist[:, 1] * 1e3, r_hist[:, 2] * 1e3, 'b-', linewidth=1)
ax2.set_xlabel('x (mm)')
ax2.set_ylabel('y (mm)')
ax2.set_zlabel('z (mm)')
ax2.set_title('3D View')

plt.tight_layout()
plt.savefig('project1_gyration.png', dpi=150)
plt.show()

# Verify gyroradius
x_max = np.max(np.abs(r_hist[:, 0]))
print(f"\nNumerical gyroradius: {x_max * 1e3:.3f} mm")
print(f"Theoretical: {rho_c * 1e3:.3f} mm")
print(f"Relative error: {100 * (x_max - rho_c) / rho_c:.2f}%")

# Test Case 2: E×B drift
print("\n\nTest 2: E×B drift")
print("=" * 50)

E0 = 1000  # V/m
B0 = 0.1   # T

v_ExB = E0 / B0
print(f"E×B drift velocity: {v_ExB:.2f} m/s")

proton = Particle(q=e, m=m_p, r0=[0, 0, 0], v0=[0, 1e5, 0])

t_final = 1e-4
dt = 1e-7

integrate_orbit(proton, lambda r, t: E_cross_B(r, t, E0, B0),
                lambda r, t: (np.zeros(3), np.zeros(3)), t_final, dt)

r_hist = np.array(proton.history['r'])
t_hist = np.array(proton.history['t'])

# Calculate drift velocity
drift_y = (r_hist[-1, 1] - r_hist[0, 1]) / t_final

print(f"\nNumerical drift velocity: {drift_y:.2f} m/s")
print(f"Theoretical E×B drift: {v_ExB:.2f} m/s")
print(f"Relative error: {100 * (drift_y - v_ExB) / v_ExB:.2f}%")

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(r_hist[:, 0] * 1e3, r_hist[:, 1] * 1e3, 'b-', linewidth=1)
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('E×B Drift (x-y plane)')
plt.grid(alpha=0.3)

plt.subplot(122)
plt.plot(t_hist * 1e6, r_hist[:, 1] * 1e3, 'b-', linewidth=2)
plt.xlabel('Time (μs)')
plt.ylabel('y position (mm)')
plt.title('Drift Motion')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('project1_ExB_drift.png', dpi=150)
plt.show()

# Test Case 3: Magnetic Mirror
print("\n\nTest 3: Magnetic Mirror")
print("=" * 50)

B0 = 0.1
L = 0.5
v_parallel = 1e5
v_perp = 5e4

electron_mirror = Particle(q=-e, m=m_e, r0=[0, 0, -0.4], v0=[v_perp, 0, v_parallel])

t_final = 5e-6
dt = 1e-9

integrate_orbit(electron_mirror, lambda r, t: magnetic_mirror(r, t, B0, L),
                lambda r, t: (np.zeros(3), np.zeros(3)), t_final, dt)

r_hist = np.array(electron_mirror.history['r'])
v_hist = np.array(electron_mirror.history['v'])

# Calculate magnetic moment
B_field = np.array([magnetic_mirror(r, 0, B0, L)[1] for r in r_hist])
B_mag = np.linalg.norm(B_field, axis=1)

mu_values = []
for i, (v, B) in enumerate(zip(v_hist, B_field)):
    B_unit = B / B_mag[i]
    v_par = np.dot(v, B_unit)
    v_perp_mag = np.sqrt(np.dot(v, v) - v_par**2)
    mu = m_e * v_perp_mag**2 / (2 * B_mag[i])
    mu_values.append(mu)

mu_values = np.array(mu_values)

print(f"Magnetic moment μ:")
print(f"  Mean: {np.mean(mu_values):.3e} J/T")
print(f"  Std: {np.std(mu_values):.3e} J/T")
print(f"  Variation: {100 * np.std(mu_values) / np.mean(mu_values):.2f}%")

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121)
ax1.plot(r_hist[:, 2], r_hist[:, 0] * 1e3, 'b-', linewidth=1)
ax1.set_xlabel('z (m)')
ax1.set_ylabel('x (mm)')
ax1.set_title('Mirror Bounce Motion (side view)')
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(122)
ax2.plot(np.array(electron_mirror.history['t']) * 1e6, mu_values / mu_values[0], 'b-', linewidth=1)
ax2.axhline(1.0, color='r', linestyle='--', label='Perfect conservation')
ax2.set_xlabel('Time (μs)')
ax2.set_ylabel('μ / μ₀')
ax2.set_title('Magnetic Moment Conservation')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('project1_mirror.png', dpi=150)
plt.show()

print("\nProject 1 complete!")
```

### 1.5 Extensions

1. **Loss cone**: In the mirror configuration, vary the pitch angle and determine the loss cone angle.
2. **Tokamak orbits**: Implement a toroidal field $B_\phi \propto 1/R$ and poloidal field $B_\theta$ to see banana orbits.
3. **Poincaré sections**: For periodic orbits, plot the phase space (e.g., $v_\parallel$ vs. $z$ at $x=0$ crossings).
4. **Relativistic particles**: Extend to relativistic energies and compare with non-relativistic.
5. **Statistical ensemble**: Run many particles with different initial conditions and compute drift statistics.

---

## Project 2: Dispersion Relation Solver

### 2.1 Overview

**Goal**: Create a general solver for electrostatic and electromagnetic wave dispersion relations in plasmas. Generate dispersion diagrams, CMA diagrams, and identify wave modes.

**Difficulty**: ⭐⭐⭐⭐

**Time estimate**: 12–20 hours

**Skills developed**:
- Root-finding for complex dispersion relations
- Understanding of cold and warm plasma wave theory
- Visualization of multi-dimensional data (ω-k diagrams, CMA diagrams)
- Interpretation of wave modes and resonances/cutoffs

### 2.2 Physics Background

The dispersion relation for electromagnetic waves in a plasma is derived from Maxwell's equations with the plasma current $\mathbf{J}$:

$$\nabla \times \nabla \times \mathbf{E} + \frac{\omega^2}{c^2} \overleftrightarrow{\epsilon} \cdot \mathbf{E} = 0$$

where $\overleftrightarrow{\epsilon}$ is the dielectric tensor.

For a **cold, magnetized plasma**, the dielectric tensor in the principal frame is:

$$\overleftrightarrow{\epsilon} = \begin{pmatrix}
S & -iD & 0 \\
iD & S & 0 \\
0 & 0 & P
\end{pmatrix}$$

where the **Stix parameters** are:

$$S = 1 - \sum_s \frac{\omega_{ps}^2}{\omega^2 - \omega_{cs}^2}$$
$$D = \sum_s \frac{\omega_{cs}}{\omega} \frac{\omega_{ps}^2}{\omega^2 - \omega_{cs}^2}$$
$$P = 1 - \sum_s \frac{\omega_{ps}^2}{\omega^2}$$

For propagation at angle $\theta$ to the magnetic field, the dispersion relation is:

$$A n^4 - B n^2 + C = 0$$

where $n = ck/\omega$ is the refractive index and:

$$A = S \sin^2\theta + P \cos^2\theta$$
$$B = (S^2 - D^2) \sin^2\theta + PS(1 + \cos^2\theta)$$
$$C = P(S^2 - D^2)$$

**Warm plasma** corrections add thermal terms (Bohm-Gross, Landau damping, etc.).

### 2.3 Implementation Guide

**Step 1: Stix Parameters**

```python
def stix_parameters(omega, omega_ps, omega_cs):
    """
    Calculate Stix parameters S, D, P.

    Parameters:
    omega: wave frequency (rad/s)
    omega_ps: plasma frequencies (array for each species)
    omega_cs: cyclotron frequencies (array, signed)

    Returns:
    S, D, P
    """
    S = 1 - np.sum(omega_ps**2 / (omega**2 - omega_cs**2))
    D = np.sum((omega_cs / omega) * omega_ps**2 / (omega**2 - omega_cs**2))
    P = 1 - np.sum(omega_ps**2 / omega**2)

    return S, D, P
```

**Step 2: Dispersion Relation**

```python
def cold_plasma_dispersion(omega, k, theta, omega_ps, omega_cs):
    """
    Cold plasma dispersion relation: A n^4 - B n^2 + C = 0.

    Returns the LHS (should be zero for a wave mode).
    """
    S, D, P = stix_parameters(omega, omega_ps, omega_cs)

    c = 3e8
    n = c * k / omega  # refractive index

    sin2 = np.sin(theta)**2
    cos2 = np.cos(theta)**2

    A = S * sin2 + P * cos2
    B = (S**2 - D**2) * sin2 + P * S * (1 + cos2)
    C = P * (S**2 - D**2)

    return A * n**4 - B * n**2 + C
```

**Step 3: Root Finding**

For a given $k$ and $\theta$, find $\omega$ such that the dispersion relation is satisfied:

```python
from scipy.optimize import fsolve, brentq

def find_omega(k, theta, omega_guess, omega_ps, omega_cs):
    """
    Find wave frequency ω for given k and θ.
    """
    def dispersion_func(omega):
        if omega <= 0:
            return 1e10  # penalize negative frequencies
        return cold_plasma_dispersion(omega, k, theta, omega_ps, omega_cs)

    omega_solution = fsolve(dispersion_func, omega_guess)[0]

    return omega_solution
```

**Step 4: Generate Dispersion Diagram**

```python
def dispersion_diagram(k_range, theta, omega_ps, omega_cs, omega_guesses):
    """
    Generate ω(k) dispersion diagram for multiple modes.

    Parameters:
    k_range: array of wavenumbers
    theta: propagation angle
    omega_guesses: list of initial guesses for different modes

    Returns:
    omegas: list of arrays, one per mode
    """
    omegas = [[] for _ in omega_guesses]

    for k in k_range:
        for i, omega_guess in enumerate(omega_guesses):
            try:
                omega_sol = find_omega(k, theta, omega_guess, omega_ps, omega_cs)
                if omega_sol > 0 and np.abs(cold_plasma_dispersion(omega_sol, k, theta,
                                                                     omega_ps, omega_cs)) < 1e-3:
                    omegas[i].append(omega_sol)
                else:
                    omegas[i].append(np.nan)
            except:
                omegas[i].append(np.nan)

    return [np.array(omega_list) for omega_list in omegas]
```

**Step 5: CMA Diagram**

The **Clemmow-Mullaly-Allis (CMA) diagram** shows regions in $(\omega_{pe}^2/\omega_{ce}^2, \omega^2/\omega_{ce}^2)$ space where different wave modes exist.

Cutoffs and resonances:
- **R cutoff**: $\omega = \omega_R = \frac{1}{2}(\omega_{ce} + \sqrt{\omega_{ce}^2 + 4\omega_{pe}^2})$
- **L cutoff**: $\omega = \omega_L = \frac{1}{2}(-\omega_{ce} + \sqrt{\omega_{ce}^2 + 4\omega_{pe}^2})$
- **P cutoff**: $\omega = \omega_{pe}$
- **Upper hybrid**: $\omega_{UH}^2 = \omega_{pe}^2 + \omega_{ce}^2$
- **Lower hybrid**: $\omega_{LH}^2 = \frac{\omega_{pi}^2}{1 + \omega_{pe}^2 / \omega_{ce}^2}$

### 2.4 Complete Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq

# Constants
e = 1.6e-19
m_e = 9.11e-31
m_p = 1.67e-27
c = 3e8

def stix_parameters(omega, omega_ps, omega_cs):
    """Calculate Stix parameters S, D, P."""
    S = 1 - np.sum(omega_ps**2 / (omega**2 - omega_cs**2 + 1e-10))
    D = np.sum((omega_cs / omega) * omega_ps**2 / (omega**2 - omega_cs**2 + 1e-10))
    P = 1 - np.sum(omega_ps**2 / (omega**2 + 1e-10))

    return S, D, P

def cold_plasma_dispersion(omega, k, theta, omega_ps, omega_cs):
    """Cold plasma dispersion relation."""
    S, D, P = stix_parameters(omega, omega_ps, omega_cs)

    n = c * k / (omega + 1e-10)

    sin2 = np.sin(theta)**2
    cos2 = np.cos(theta)**2

    A = S * sin2 + P * cos2
    B = (S**2 - D**2) * sin2 + P * S * (1 + cos2)
    C = P * (S**2 - D**2)

    return A * n**4 - B * n**2 + C

# Plasma parameters
n = 1e18  # m^-3
B = 0.05  # T

omega_pe = np.sqrt(n * e**2 / (m_e * 8.85e-12))
omega_pi = np.sqrt(n * e**2 / (m_p * 8.85e-12))
omega_ce = e * B / m_e
omega_ci = e * B / m_p

omega_ps = np.array([omega_pe, omega_pi])
omega_cs = np.array([omega_ce, -omega_ci])  # electrons +, ions -

print("Plasma parameters:")
print(f"  ω_pe / (2π) = {omega_pe / (2 * np.pi):.3e} Hz")
print(f"  ω_ce / (2π) = {omega_ce / (2 * np.pi):.3e} Hz")
print(f"  ω_pi / (2π) = {omega_pi / (2 * np.pi):.3e} Hz")
print(f"  ω_ci / (2π) = {omega_ci / (2 * np.pi):.3e} Hz")
print()

# Cutoff frequencies
omega_R = 0.5 * (omega_ce + np.sqrt(omega_ce**2 + 4 * omega_pe**2))
omega_L = 0.5 * (-omega_ce + np.sqrt(omega_ce**2 + 4 * omega_pe**2))
omega_UH = np.sqrt(omega_pe**2 + omega_ce**2)
omega_LH = omega_pi / np.sqrt(1 + omega_pe**2 / omega_ce**2)

print("Characteristic frequencies:")
print(f"  R cutoff: {omega_R / (2 * np.pi):.3e} Hz")
print(f"  L cutoff: {omega_L / (2 * np.pi):.3e} Hz")
print(f"  Upper hybrid: {omega_UH / (2 * np.pi):.3e} Hz")
print(f"  Lower hybrid: {omega_LH / (2 * np.pi):.3e} Hz")
print()

# Dispersion diagram: parallel propagation (θ = 0)
k_range = np.linspace(1, 1000, 300)
theta = 0  # parallel

# Find O-mode (P = 0 → ω = ω_pe) and X-mode (more complex)
omega_O = []
omega_R_mode = []
omega_L_mode = []

for k in k_range:
    # O-mode: ω² = ω_pe² + k² c²
    omega_O.append(np.sqrt(omega_pe**2 + (k * c)**2))

    # R-mode: solve S - n² = 0
    def R_dispersion(omega):
        S, D, P = stix_parameters(omega, omega_ps, omega_cs)
        n = c * k / omega
        return S - n**2

    # L-mode: solve S - n² = 0 (but different branch)

    try:
        omega_R_sol = fsolve(R_dispersion, omega_R * 1.5)[0]
        if omega_R_sol > omega_R and omega_R_sol < 10 * omega_ce:
            omega_R_mode.append(omega_R_sol)
        else:
            omega_R_mode.append(np.nan)
    except:
        omega_R_mode.append(np.nan)

    try:
        omega_L_sol = fsolve(R_dispersion, omega_L * 0.9)[0]
        if omega_L_sol > 0 and omega_L_sol < omega_L * 1.5:
            omega_L_mode.append(omega_L_sol)
        else:
            omega_L_mode.append(np.nan)
    except:
        omega_L_mode.append(np.nan)

omega_O = np.array(omega_O)
omega_R_mode = np.array(omega_R_mode)
omega_L_mode = np.array(omega_L_mode)

# Plot dispersion diagram
plt.figure(figsize=(12, 7))

plt.plot(k_range, omega_O / omega_ce, 'b-', linewidth=2, label='O-mode')
plt.plot(k_range, omega_R_mode / omega_ce, 'r-', linewidth=2, label='R-mode (X-mode branch)')
plt.plot(k_range, omega_L_mode / omega_ce, 'g-', linewidth=2, label='L-mode (X-mode branch)')

# Light line
plt.plot(k_range, k_range * c / omega_ce, 'k--', linewidth=1, alpha=0.5, label='Light line ω = ck')

# Cutoffs and resonances
plt.axhline(omega_R / omega_ce, color='r', linestyle=':', alpha=0.7, label=f'R cutoff')
plt.axhline(omega_L / omega_ce, color='g', linestyle=':', alpha=0.7, label=f'L cutoff')
plt.axhline(omega_pe / omega_ce, color='b', linestyle=':', alpha=0.7, label=f'ω_pe')
plt.axhline(omega_UH / omega_ce, color='m', linestyle=':', alpha=0.7, label=f'Upper hybrid')
plt.axhline(1.0, color='orange', linestyle=':', alpha=0.7, label=f'ω_ce')

plt.xlabel('Wavenumber k (m⁻¹)', fontsize=12)
plt.ylabel('ω / ω_ce', fontsize=12)
plt.title('Cold Plasma Dispersion: Parallel Propagation (θ = 0)', fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.grid(alpha=0.3)
plt.xlim(0, 1000)
plt.ylim(0, 20)
plt.tight_layout()
plt.savefig('project2_dispersion_parallel.png', dpi=150)
plt.show()

print("Project 2: Dispersion diagram complete!")
```

### 2.5 Extensions

1. **Oblique propagation**: Vary $\theta$ from 0 to $\pi/2$ and plot $\omega(k, \theta)$ as a 3D surface.
2. **Warm plasma**: Add thermal corrections (Bohm-Gross: $\omega^2 = \omega_{pe}^2 + 3k^2 v_{te}^2$).
3. **Ion acoustic waves**: Include ion response, plot IA dispersion with Landau damping.
4. **CMA diagram**: Plot regions of propagation/evanescence in $(X, Y)$ space where $X = \omega_{pe}^2/\omega^2$, $Y = \omega_{ce}/\omega$.
5. **Resonance cones**: For whistler waves, plot the resonance cone angle vs. frequency.

---

## Project 3: 1D Vlasov-Poisson Solver

### 3.1 Overview

**Goal**: Implement a 1D-1V Vlasov-Poisson solver using the semi-Lagrangian (splitting) method. Simulate Langmuir oscillations, Landau damping, two-stream instability, and bump-on-tail instability.

**Difficulty**: ⭐⭐⭐⭐⭐

**Time estimate**: 20–30 hours

**Skills developed**:
- Advanced numerical methods for PDEs
- Kinetic plasma simulations
- FFT-based Poisson solver
- Phase space visualization
- Comparison with linear theory (Landau damping rate, growth rates)

### 3.2 Physics Background

The **1D Vlasov equation** describes the evolution of the electron distribution function $f(x, v, t)$:

$$\frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x} + \frac{q E}{m} \frac{\partial f}{\partial v} = 0$$

coupled with **Poisson's equation** for the electric field:

$$\frac{\partial E}{\partial x} = \frac{q}{\epsilon_0} (n_i - n_e)$$

where the electron density is:
$$n_e(x, t) = \int f(x, v, t) \, dv$$

Ions are assumed to be a fixed neutralizing background: $n_i = n_0 = \text{const}$.

**Splitting method**: Split the Vlasov equation into two steps:

1. **Advection in x** (free streaming):
   $$\frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x} = 0$$

2. **Advection in v** (acceleration):
   $$\frac{\partial f}{\partial t} + \frac{q E}{m} \frac{\partial f}{\partial v} = 0$$

Each step is solved exactly by **backward tracing** of characteristics.

### 3.3 Implementation Guide

**Step 1: Initialization**

Set up the phase space grid:
- $N_x$ grid points in $x \in [0, L]$
- $N_v$ grid points in $v \in [v_{min}, v_{max}]$
- Initial distribution: $f_0(x, v) = f_0(v) (1 + \alpha \cos(kx))$ for Landau damping

**Step 2: Poisson Solver (FFT)**

```python
def solve_poisson_fft(rho, dx, L):
    """
    Solve Poisson equation: d²φ/dx² = -ρ/ε₀ using FFT.

    Returns electric field E = -dφ/dx.
    """
    epsilon_0 = 8.85e-12
    Nx = len(rho)

    # Fourier transform of rho
    rho_k = np.fft.fft(rho)

    # Wavenumbers
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    # k[0] = 0 (DC mode) would produce division by zero in φ_k = -ρ_k/(ε₀k²).
    # Setting k[0] = 1 temporarily makes the denominator finite; the resulting
    # phi_k[0] is immediately overridden below so it has no physical effect.
    k[0] = 1  # avoid division by zero (DC component is arbitrary for periodic)

    # Fourier transform of potential: φ_k = -ρ_k / (ε₀ k²)
    phi_k = -rho_k / (epsilon_0 * k**2)
    # Setting the DC potential to zero fixes the additive gauge freedom: in a
    # periodic domain only differences in potential are physical, and a non-zero
    # mean would shift all particle energies without affecting the dynamics.
    phi_k[0] = 0  # set DC component to zero

    # Differentiate in Fourier space: exact spectral differentiation avoids the
    # truncation error of finite-difference stencils for dφ/dx.
    # Electric field: E = -dφ/dx → E_k = i k φ_k
    E_k = 1j * k * phi_k

    # Inverse FFT
    E = np.real(np.fft.ifft(E_k))

    return E
```

**Step 3: Advection in x**

Solve $\partial f / \partial t + v \partial f / \partial x = 0$:

For each $v_j$, the characteristic is $x(t) = x_0 + v_j \Delta t$. To update $f$, trace back:

$$f^{n+1}(x_i, v_j) = f^n(x_i - v_j \Delta t, v_j)$$

Use interpolation (e.g., cubic spline) for non-grid values.

**Step 4: Advection in v**

Solve $\partial f / \partial t + (qE/m) \partial f / \partial v = 0$:

For each $x_i$, the characteristic is $v(t) = v_0 + (q E_i / m) \Delta t$. Update:

$$f^{n+1}(x_i, v_j) = f^n(x_i, v_j - \frac{qE_i}{m} \Delta t)$$

Again, use interpolation.

**Step 5: Time-stepping Loop**

```python
for n in range(Nt):
    # 1. Compute density
    n_e = np.trapz(f, v_grid, axis=1)

    # 2. Solve Poisson
    rho = e * (n_0 - n_e)
    E = solve_poisson_fft(rho, dx, L)

    # 3. Advection in x (half step)
    f = advect_x(f, v_grid, dt/2)

    # 4. Advection in v (full step)
    f = advect_v(f, E, dt)

    # 5. Advection in x (half step)
    f = advect_x(f, v_grid, dt/2)

    # 6. Diagnostics
    energy[n] = compute_energy(f, E)
```

### 3.4 Test Cases

**Test 1: Plasma Oscillation**

Initialize with a small sinusoidal density perturbation:
$$n_e(x, 0) = n_0 (1 + \epsilon \cos(kx)), \quad f(x, v, 0) = \frac{n_e(x, 0)}{\sqrt{2\pi v_{th}^2}} e^{-v^2/(2v_{th}^2)}$$

The electric field should oscillate at $\omega_{pe}$. Verify:
$$\omega_{pe} = \sqrt{\frac{n_0 e^2}{\epsilon_0 m_e}}$$

**Test 2: Landau Damping**

Use the same initial condition. The electric field amplitude should decay exponentially:
$$E(t) \propto e^{-\gamma_L t}$$

where the Landau damping rate is (for $k\lambda_D \ll 1$):
$$\gamma_L \approx \sqrt{\frac{\pi}{8}} \omega_{pe} (k\lambda_D)^{-3} e^{-1/(2k^2\lambda_D^2)}$$

**Test 3: Two-Stream Instability**

Initialize with two counter-streaming beams:
$$f(x, v, 0) = \frac{n_0}{2} \left[ \frac{1}{\sqrt{2\pi v_{th}^2}} e^{-(v - v_0)^2 / (2v_{th}^2)} + \frac{1}{\sqrt{2\pi v_{th}^2}} e^{-(v + v_0)^2 / (2v_{th}^2)} \right]$$

with a small perturbation in $x$. The instability grows exponentially. Measure growth rate and compare with theory.

**Test 4: Bump-on-Tail**

Initialize with a bulk Maxwellian plus a small bump at higher velocity:
$$f(x, v, 0) = \frac{n_b}{\sqrt{2\pi v_{th}^2}} e^{-v^2/(2v_{th}^2)} + \frac{n_{bump}}{\sqrt{2\pi v_{bump}^2}} e^{-(v - v_{bump})^2/(2v_{bump}^2)}$$

This is unstable and drives waves that flatten the bump (quasi-linear diffusion).

### 3.5 Complete Code (Simplified)

Due to length, here's a streamlined version:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Constants
e = 1.6e-19
m_e = 9.11e-31
epsilon_0 = 8.85e-12

# Grid parameters
Nx = 128
Nv = 128
L = 2 * np.pi / 1e5  # spatial domain (one wavelength)
v_max = 5e6  # m/s

x_grid = np.linspace(0, L, Nx, endpoint=False)
v_grid = np.linspace(-v_max, v_max, Nv)
dx = x_grid[1] - x_grid[0]
dv = v_grid[1] - v_grid[0]

# Plasma parameters
n_0 = 1e16  # m^-3
T_e = 1  # eV
v_th = np.sqrt(e * T_e / m_e)
omega_pe = np.sqrt(n_0 * e**2 / (epsilon_0 * m_e))

# Time step (CFL condition)
dt = 0.1 * min(dx / v_max, dv / (e * 1e3 / m_e))  # conservative
Nt = 500

print(f"ω_pe = {omega_pe:.3e} rad/s")
print(f"T_pe = {2 * np.pi / omega_pe:.3e} s")
print(f"dt = {dt:.3e} s")
print(f"Total time = {Nt * dt:.3e} s ({Nt * dt * omega_pe / (2 * np.pi):.2f} plasma periods)")

# Initial distribution: Maxwellian with perturbation
k_pert = 2 * np.pi / L
alpha = 0.01

f = np.zeros((Nx, Nv))
for i, x in enumerate(x_grid):
    n_local = n_0 * (1 + alpha * np.cos(k_pert * x))
    f[i, :] = n_local / (np.sqrt(2 * np.pi) * v_th) * np.exp(-v_grid**2 / (2 * v_th**2))

# Advection functions (using linear interpolation for simplicity)
def advect_x(f, v_grid, dt):
    """Advect in x: f(x - v*dt, v)."""
    f_new = np.zeros_like(f)
    for j, v in enumerate(v_grid):
        shift = v * dt
        x_old = (x_grid - shift) % L  # periodic boundary
        f_new[:, j] = np.interp(x_old, x_grid, f[:, j], period=L)
    return f_new

def advect_v(f, E, dt):
    """Advect in v: f(x, v - (qE/m)*dt)."""
    f_new = np.zeros_like(f)
    for i, x_i in enumerate(x_grid):
        accel = -e * E[i] / m_e
        v_old = v_grid - accel * dt
        # Interpolate (with extrapolation for boundary)
        f_new[i, :] = np.interp(v_old, v_grid, f[i, :], left=0, right=0)
    return f_new

def solve_poisson_fft(rho, dx):
    """Solve Poisson equation using FFT."""
    Nx = len(rho)
    rho_k = np.fft.fft(rho)
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    # k[0] = 0 (the DC/zero-frequency mode) would cause division by zero in
    # phi_k = -rho_k / (ε₀ k²). Setting k[0] = 1 is a placeholder that makes
    # the division well-defined; the resulting phi_k[0] is overwritten next.
    k[0] = 1  # avoid division by zero
    phi_k = -rho_k / (epsilon_0 * k**2)
    # Force the DC component of the potential to zero: with periodic boundaries,
    # the absolute potential is arbitrary (only ∇φ matters physically), and
    # a non-zero DC term would accumulate numerical drift. Setting phi_k[0] = 0
    # fixes the gauge and ensures the mean electric field is zero, consistent
    # with overall charge neutrality (∫ρ dx = 0 for a quasi-neutral plasma).
    phi_k[0] = 0
    # Differentiate in Fourier space: E = -dφ/dx → E_k = ik φ_k.
    # Multiplication by ik is exact (no finite-difference error), which is one
    # reason spectral methods are preferred for the Poisson step.
    E_k = 1j * k * phi_k
    E = np.real(np.fft.ifft(E_k))
    return E

# Diagnostics
E_history = []
energy_history = []

# Time-stepping loop
for n in range(Nt):
    # Compute density by integrating f over velocity space: n_e = ∫f dv.
    # This moment reduction is exact and is why kinetic Vlasov simulations
    # can recover fluid moments without any closure assumption.
    n_e = np.trapz(f, v_grid, axis=1)

    # Solve Poisson
    # rho = e(n_i - n_e): ions are a fixed background (n_i = n_0), so only
    # electron density fluctuations drive the self-consistent electric field.
    rho = e * (n_0 - n_e)
    E = solve_poisson_fft(rho, dx)

    # Store diagnostics before advancing f so we record the state that
    # corresponds to the electric field just computed (consistent snapshot).
    E_history.append(np.max(np.abs(E)))
    field_energy = 0.5 * epsilon_0 * np.sum(E**2) * dx
    kinetic_energy = 0.5 * m_e * np.sum(f * (v_grid[np.newaxis, :]**2) * dx * dv)
    energy_history.append(field_energy + kinetic_energy)

    # Strang splitting (half-x, full-v, half-x) achieves second-order accuracy
    # in time: splitting at first order would give only O(dt) accuracy, while
    # this symmetric arrangement cancels the leading error term, matching the
    # accuracy of Runge-Kutta 2 at the same cost.
    f = advect_x(f, v_grid, dt / 2)
    f = advect_v(f, E, dt)
    f = advect_x(f, v_grid, dt / 2)

    # Print progress
    if n % 50 == 0:
        print(f"Step {n}/{Nt}, E_max = {E_history[-1]:.3e} V/m")

# Plot results
t_grid = np.arange(Nt) * dt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Electric field amplitude vs. time
axes[0, 0].semilogy(t_grid * omega_pe / (2 * np.pi), E_history, 'b-', linewidth=1)
axes[0, 0].set_xlabel('Time (plasma periods)', fontsize=11)
axes[0, 0].set_ylabel('Max |E| (V/m)', fontsize=11)
axes[0, 0].set_title('Electric Field Amplitude (Landau Damping)', fontsize=12)
axes[0, 0].grid(alpha=0.3)

# Fit exponential decay to measure damping rate
t_fit = t_grid[50:200]
E_fit = np.array(E_history[50:200])
log_E = np.log(E_fit)
p = np.polyfit(t_fit, log_E, 1)
gamma_numerical = -p[0]

# Theoretical Landau damping rate
k = k_pert
lambda_D = v_th / omega_pe
kLD = k * lambda_D
gamma_theory = np.sqrt(np.pi / 8) * omega_pe * (kLD)**(-3) * np.exp(-1 / (2 * kLD**2))

axes[0, 0].plot(t_fit * omega_pe / (2 * np.pi), np.exp(p[0] * t_fit + p[1]), 'r--',
                linewidth=2, label=f'Fit: γ = {gamma_numerical:.2e} s⁻¹')
axes[0, 0].legend(fontsize=10)

print(f"\nLandau damping:")
print(f"  Numerical γ = {gamma_numerical:.3e} s⁻¹ ({gamma_numerical/omega_pe:.3e} ω_pe)")
print(f"  Theoretical γ = {gamma_theory:.3e} s⁻¹ ({gamma_theory/omega_pe:.3e} ω_pe)")
print(f"  Relative error = {100 * (gamma_numerical - gamma_theory) / gamma_theory:.1f}%")

# Energy conservation
axes[0, 1].plot(t_grid * omega_pe / (2 * np.pi), np.array(energy_history) / energy_history[0], 'g-',
                linewidth=2)
axes[0, 1].axhline(1.0, color='k', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Time (plasma periods)', fontsize=11)
axes[0, 1].set_ylabel('Total energy / E(0)', fontsize=11)
axes[0, 1].set_title('Energy Conservation', fontsize=12)
axes[0, 1].grid(alpha=0.3)

# Phase space at initial time
axes[1, 0].contourf(x_grid, v_grid / 1e6, f.T, levels=20, cmap='viridis')
axes[1, 0].set_xlabel('x (m)', fontsize=11)
axes[1, 0].set_ylabel('v (10⁶ m/s)', fontsize=11)
axes[1, 0].set_title('Phase Space f(x, v) at Final Time', fontsize=12)

# Density profile
n_e_final = np.trapz(f, v_grid, axis=1)
axes[1, 1].plot(x_grid, n_e_final / n_0, 'b-', linewidth=2, label='Final')
axes[1, 1].axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Equilibrium')
axes[1, 1].set_xlabel('x (m)', fontsize=11)
axes[1, 1].set_ylabel('n_e / n_0', fontsize=11)
axes[1, 1].set_title('Density Profile', fontsize=12)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('project3_vlasov_landau.png', dpi=150)
plt.show()

print("\nProject 3: Vlasov-Poisson simulation complete!")
```

### 3.6 Extensions

1. **Higher-order interpolation**: Use cubic or quintic splines for better accuracy.
2. **Two-stream instability**: Modify initial condition, measure growth rate, compare with theory.
3. **Bump-on-tail**: Observe plateau formation, measure wave energy saturation.
4. **Electron trapping**: Plot phase space vortices formed by trapped particles.
5. **2D Vlasov**: Extend to 2D-2V (computationally intensive!).
6. **PIC comparison**: Implement a simple 1D PIC code and compare with Vlasov.

---

## Conclusion

These three projects synthesize the material from the entire course:

- **Project 1** (Particle Orbits) brings to life the single-particle theory from Lessons 3–4.
- **Project 2** (Dispersion Solver) implements the wave theory from Lessons 9–10.
- **Project 3** (Vlasov-Poisson) tackles the kinetic theory from Lessons 6–8 and demonstrates fundamental plasma phenomena like Landau damping.

By completing these projects, you will have hands-on experience with the computational tools used in modern plasma physics research. These methods scale up to:

- **Particle-in-Cell (PIC)** codes for collisionless plasma simulations (e.g., laser-plasma interaction, space physics)
- **Gyrokinetic** codes for tokamak turbulence (e.g., GENE, GYRO, GS2)
- **MHD** codes for fusion equilibrium and stability (e.g., NIMROD, M3D-C1)

Congratulations on completing the Plasma Physics course! You now have a solid foundation in both the theory and computation of plasma physics.

---

## Exercises

### Exercise 1: Boris Algorithm Convergence Study

The Boris algorithm is second-order accurate in time. Verify this empirically by measuring how the gyroradius error scales with the time step.

**Steps**:
1. Set up an electron gyrating in a uniform magnetic field $\mathbf{B} = 0.1\,\text{T}\,\hat{\mathbf{z}}$ with $v_\perp = 10^6\,\text{m/s}$ and no initial parallel velocity.
2. Compute the analytical gyroradius $\rho_c = m_e v_\perp / (eB)$ and cyclotron period $T_c = 2\pi m_e / (eB)$.
3. Run the simulation for exactly 10 cyclotron periods using time steps $\Delta t = T_c / N$ for $N \in \{10, 20, 50, 100, 200, 500\}$.
4. After 10 periods, measure the positional error: the deviation of the particle's position from its starting point (the orbit should close exactly).
5. Plot the positional error vs. $\Delta t$ on a log-log scale and fit the slope. Confirm the slope is close to 2, consistent with second-order convergence.
6. Also measure the energy drift $\Delta E_{kin} / E_{kin,0}$ at each time step size and verify it remains bounded (not growing) for all $N$.

**Expected result**: Positional error $\propto (\Delta t)^2$; energy is conserved to machine precision regardless of $\Delta t$.

---

### Exercise 2: Loss Cone in a Magnetic Mirror

A magnetic mirror confines particles only if their pitch angle exceeds the loss cone angle. Derive the loss cone angle analytically and then verify it numerically using the particle orbit simulator.

**Steps**:
1. For a magnetic mirror with mirror ratio $R_m = B_{max}/B_{min}$ (use $B_{min} = 0.1\,\text{T}$ at $z=0$ and $B_{max} = 0.5\,\text{T}$ at the mirror points), derive the loss cone half-angle:
   $$\sin^2\alpha_{lc} = \frac{B_{min}}{B_{max}} = \frac{1}{R_m}$$
2. Launch 50 electrons from the midplane ($z=0$) with the same speed $v = 5 \times 10^6\,\text{m/s}$ but with pitch angles $\alpha$ uniformly sampled from $0°$ to $90°$ (where $\alpha$ is the angle between $\mathbf{v}$ and $\mathbf{B}$).
3. Integrate each orbit for $t_{final} = 50\,\mu\text{s}$ using the magnetic mirror field from Section 1.3.
4. Classify each particle as confined (bouncing) or lost (reaching $|z| > z_{mirror}$) and record its initial pitch angle.
5. Plot the confinement outcome vs. pitch angle. Mark the theoretical loss cone boundary and compare with your numerical results.
6. Estimate the fraction of an isotropic distribution that would be confined.

**Hint**: The adiabatic invariant $\mu = m v_\perp^2 / (2B)$ is conserved. Use this to derive the pitch angle at the mirror point and determine the condition for reflection.

---

### Exercise 3: Cutoff and Resonance Identification in the CMA Diagram

The Clemmow-Mullaly-Allis (CMA) diagram organizes all cold plasma wave modes by their cutoffs and resonances. Build this diagram numerically and identify the named wave modes.

**Steps**:
1. Choose a fixed magnetic field $B_0 = 0.05\,\text{T}$. Define dimensionless axes:
   - $X = \omega_{pe}^2 / \omega^2$ (density parameter, varied by changing $n$ or $\omega$)
   - $Y = \omega_{ce} / \omega$ (magnetization parameter)
2. On a grid of $(X, Y)$ values with $X \in [0, 4]$ and $Y \in [0, 3]$, compute the Stix parameters $S$, $D$, $P$ for each point (treat $\omega$ as fixed and vary $n$ to change $X$).
3. Draw the cutoff lines:
   - $P = 0$ (O-mode cutoff: $\omega = \omega_{pe}$, i.e., $X = 1$)
   - $R = S + D = 0$ (R cutoff)
   - $L = S - D = 0$ (L cutoff)
4. Draw the resonance lines:
   - $S = 0$ (upper and lower hybrid resonances)
   - $\tan^2\theta = -P/S$ for $\theta = 0$ and $\theta = \pi/2$ (parallel and perpendicular resonances)
5. Color the regions by which wave modes propagate (both $n_\pm^2 > 0$, one positive, or both negative/evanescent).
6. Label the regions with their standard names: O-mode, X-mode, R-wave, L-wave, whistler mode, lower hybrid wave.

**Reference**: The diagram should reproduce Figure 1-8 in T. H. Stix, *Waves in Plasmas* (AIP, 1992).

---

### Exercise 4: Two-Stream Instability Growth Rate Measurement

The two-stream instability is one of the most important kinetic plasma instabilities. Measure its linear growth rate numerically and compare with the analytical prediction.

**Steps**:
1. Modify the Vlasov-Poisson solver from Project 3 to accept a two-beam initial condition:
   $$f_0(x, v) = \frac{n_0}{2}\left[\mathcal{M}(v - v_0) + \mathcal{M}(v + v_0)\right](1 + \alpha\cos(k_0 x))$$
   where $\mathcal{M}(v) = (2\pi v_{th}^2)^{-1/2}\exp(-v^2/2v_{th}^2)$, beam speed $v_0 = 3v_{th}$, $v_{th} = 10^6\,\text{m/s}$, $\alpha = 0.01$.
2. Choose the perturbation wavenumber $k_0$ to lie in the unstable band: for symmetric beams the instability is strongest near $k_0 \approx \omega_{pe} / v_0$. Verify this is in the unstable region using the cold-beam dispersion relation:
   $$1 = \frac{\omega_{pe}^2/2}{(\omega - kv_0)^2} + \frac{\omega_{pe}^2/2}{(\omega + kv_0)^2}$$
3. Run the simulation and record $\max_x |E(x, t)|$ at each time step.
4. Identify the linear growth phase (where $\ln|E|$ grows linearly in time) and measure the numerical growth rate $\gamma_{num}$ by fitting a straight line.
5. Compare $\gamma_{num}$ with the analytical imaginary part of $\omega$ obtained by solving the cold-beam dispersion relation numerically (using `numpy.roots` on the polynomial form).
6. Plot the phase space $f(x, v)$ at three times: initial, mid-saturation, and after saturation. Describe the vortex structures (phase space holes) that appear at saturation.

**Expected result**: During the linear phase, $|E| \propto e^{\gamma t}$ with $\gamma \approx \omega_{pe}/2\sqrt{2}$ for $v_0 \gg v_{th}$ and $k = \omega_{pe}/v_0$.

---

### Exercise 5: Integrated Mini-Project — Whistler Wave Propagation and Dispersion

Whistler waves are right-hand circularly polarized electromagnetic waves that propagate below the electron cyclotron frequency. They play an important role in radiation belt dynamics and ionospheric communication. Design and carry out an integrated mini-project that connects all three main projects.

**Part A — Dispersion diagram** (using Project 2 code):
1. Compute and plot the whistler wave branch ($\omega < \omega_{ce}$, parallel propagation, R-mode) on a $\omega$-$k$ diagram.
2. Overplot the group velocity $v_g = d\omega/dk$ as a function of frequency and identify the frequency of maximum group velocity.
3. Show that the group velocity is approximately $v_g \approx 2c\sqrt{\omega/\omega_{pe}^2 \cdot \omega_{ce}}$ in the limit $\omega_{ce} \ll \omega_{pe}$ and verify numerically.

**Part B — Temporal dispersion** (analytical calculation):
1. A lightning stroke generates a broadband impulse. Consider two frequencies $f_1 = 5\,\text{kHz}$ and $f_2 = 10\,\text{kHz}$ propagating through the ionosphere (use $n_e = 10^{10}\,\text{m}^{-3}$, $B_0 = 5 \times 10^{-5}\,\text{T}$, $L = 1000\,\text{km}$).
2. Calculate the travel time $t = L/v_g(\omega)$ for each frequency.
3. The time delay between arrivals is the "whistler dispersion." Estimate the time separation $\Delta t = t_1 - t_2$ and compare with observed whistler dispersions (typically 1–10 seconds).

**Part C — Particle resonance** (analytical):
1. Electrons can resonate with whistler waves via the cyclotron resonance condition: $\omega - k v_\parallel = \omega_{ce}$ (for electrons). Given the whistler wave at $f = 5\,\text{kHz}$, find the resonant electron energy (in keV) for this ionospheric plasma.
2. Discuss how this resonance leads to pitch angle scattering and loss of radiation belt electrons into the atmosphere (wave-particle interaction(파동-입자 상호작용) in action).

**Deliverable**: A single Python script that generates three figures (dispersion diagram, group velocity curve, travel time vs. frequency) with appropriate annotations, plus a short written discussion of the wave-particle resonance.

---

**Previous**: [Plasma Diagnostics](./15_Plasma_Diagnostics.md) | **Next**: None (last lesson)
