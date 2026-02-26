# 5. Single Particle Motion II

## Learning Objectives

- Derive the general drift velocity formula for arbitrary forces in magnetized plasmas
- Understand grad-B drift and its physical origin from non-uniform magnetic fields
- Analyze curvature drift and its role in curved magnetic field geometries
- Calculate polarization drift from time-varying electric fields and its mass dependence
- Compare and contrast all drift types with their charge and mass dependencies
- Simulate particle orbits in spatially varying magnetic fields using Python

## 1. General Drift Velocity Formula

### 1.1 Derivation from First Principles

In Lesson 4, we studied $\mathbf{E}\times\mathbf{B}$ drift. Now we generalize to arbitrary forces. Consider a particle in a uniform magnetic field $\mathbf{B}$ subject to an additional force $\mathbf{F}$.

The equation of motion is:

$$
m\frac{d\mathbf{v}}{dt} = q(\mathbf{E} + \mathbf{v}\times\mathbf{B}) + \mathbf{F}
$$

We decompose the velocity into:
- Gyromotion around field lines
- Drift perpendicular to $\mathbf{B}$
- Motion parallel to $\mathbf{B}$

For a slowly varying force (slow compared to gyrofrequency $\omega_c$), we can average over a gyroperiod. The perpendicular drift velocity is:

$$
\mathbf{v}_D = \frac{\mathbf{F}\times\mathbf{B}}{qB^2}
$$

**Key insight**: Any force $\mathbf{F}$ perpendicular to $\mathbf{B}$ causes drift in the direction $\mathbf{F}\times\mathbf{B}$.

### 1.2 Physical Interpretation

The drift arises because the force $\mathbf{F}$ modifies the particle's velocity differently at different points in the gyro-orbit:

```
     Gyro-orbit viewed from above (B pointing up)

     F → (force to the right)

        v increases
           ↑
     ←-----o-----→  Direction of drift: ⊙ (into page)
           ↓         F × B points into page
        v decreases

     Radius larger where v is larger
     → net displacement perpendicular to both F and B
```

The drift direction is:
- Independent of charge sign if $\mathbf{F}$ is charge-independent (e.g., gravity)
- Opposite for opposite charges if $\mathbf{F}$ depends on charge (e.g., electric force)

### 1.3 Comparison with E×B Drift

For electric force $\mathbf{F} = q\mathbf{E}$:

$$
\mathbf{v}_E = \frac{q\mathbf{E}\times\mathbf{B}}{qB^2} = \frac{\mathbf{E}\times\mathbf{B}}{B^2}
$$

This recovers the $\mathbf{E}\times\mathbf{B}$ drift — independent of charge and mass, as expected.

## 2. Grad-B Drift

### 2.1 Physical Origin

In a non-uniform magnetic field where $|\mathbf{B}|$ varies in space, a gyrating particle samples different field strengths during its orbit. Since the Larmor radius depends on $B$:

$$
r_L = \frac{mv_\perp}{|q|B}
$$

the radius is larger where $B$ is weaker. This asymmetry produces a net drift.

### 2.2 Derivation Using Guiding Center Expansion

Consider a magnetic field varying in the x-direction: $\mathbf{B} = B(x)\hat{z}$. Expand around the guiding center position $x_0$:

$$
B(x) \approx B_0 + \left(\frac{\partial B}{\partial x}\right)_0 (x - x_0)
$$

During gyration, the particle position is:

$$
x = x_0 + r_L\cos(\omega_c t)
$$

The effective force from the spatial variation is:

$$
F_x = -\mu\frac{\partial B}{\partial x}
$$

where the magnetic moment $\mu = \frac{mv_\perp^2}{2B}$ is the first adiabatic invariant (conserved for slowly varying fields).

Averaging over a gyroperiod and using the general drift formula:

$$
\mathbf{v}_{\nabla B} = \frac{\mathbf{F}_{\nabla B}\times\mathbf{B}}{qB^2} = \frac{-\mu\nabla B\times\mathbf{B}}{qB^2}
$$

Since $\mu = \frac{mv_\perp^2}{2B}$:

$$
\boxed{\mathbf{v}_{\nabla B} = \pm\frac{mv_\perp^2}{2qB^3}(\mathbf{B}\times\nabla B)}
$$

The sign depends on charge: ions and electrons drift in **opposite** directions.

### 2.3 Alternative Form

Using $\nabla B = (\nabla B)$ in the direction perpendicular to $\mathbf{B}$:

$$
\mathbf{v}_{\nabla B} = \pm\frac{v_\perp^2}{2\omega_c}\frac{\mathbf{B}\times\nabla B}{B^2}
$$

where $\omega_c = |q|B/m$ is the gyrofrequency.

### 2.4 Current from Grad-B Drift

Since ions and electrons drift in opposite directions, grad-B drift produces a **diamagnetic current**:

$$
\mathbf{J}_{\nabla B} = n(q_i\mathbf{v}_{\nabla B,i} + q_e\mathbf{v}_{\nabla B,e})
$$

For a plasma with $q_i = e$, $q_e = -e$, and assuming $T_i = T_e = T$:

$$
\mathbf{J}_{\nabla B} = -\frac{2nkT}{B^2}\mathbf{B}\times\nabla B
$$

This current creates a magnetic field opposing the gradient, hence "diamagnetic".

### 2.5 Numerical Example

**Example**: Proton in Earth's magnetosphere.
- $B_0 = 10^{-5}$ T, $\nabla B \sim 10^{-7}$ T/m (dipole field gradient)
- $v_\perp = 10^5$ m/s, $m = 1.67\times 10^{-27}$ kg, $q = 1.6\times 10^{-19}$ C

$$
v_{\nabla B} = \frac{mv_\perp^2}{2qB^2}\frac{\nabla B}{B} \approx \frac{(1.67\times 10^{-27})(10^5)^2}{2(1.6\times 10^{-19})(10^{-5})^2} \cdot 10^{-2} \approx 5.2\times 10^3 \text{ m/s}
$$

Much smaller than gyration speed but significant over long timescales.

## 3. Curvature Drift

### 3.1 Curved Magnetic Field Lines

When magnetic field lines are curved, a particle moving along the field line experiences a centrifugal force in the guiding center frame. The radius of curvature $R_c$ is defined by:

$$
\frac{\mathbf{B}}{B}\cdot\nabla\frac{\mathbf{B}}{B} = -\frac{\mathbf{R}_c}{R_c^2}
$$

where $\mathbf{R}_c$ points toward the center of curvature.

### 3.2 Centrifugal Force

A particle with parallel velocity $v_\parallel$ following a curved field line feels:

$$
\mathbf{F}_c = \frac{mv_\parallel^2}{R_c}\hat{R}_c = \frac{mv_\parallel^2}{R_c^2}\mathbf{R}_c
$$

Using the general drift formula:

$$
\boxed{\mathbf{v}_R = \frac{mv_\parallel^2}{qB^2}\frac{\mathbf{R}_c\times\mathbf{B}}{R_c^2}}
$$

This is the **curvature drift**.

### 3.3 Combined Grad-B and Curvature Drift

In most realistic geometries (e.g., tokamaks, dipole fields), curvature and field gradients occur together. For a curved field with varying magnitude:

$$
\nabla B \approx \frac{B}{R_c}
$$

The total drift is:

$$
\mathbf{v}_{gc} = \mathbf{v}_{\nabla B} + \mathbf{v}_R = \frac{m}{qB^2}\left(\frac{v_\perp^2}{2} + v_\parallel^2\right)\frac{\mathbf{B}\times\nabla B}{B}
$$

This can also be written using the total kinetic energy:

$$
\mathbf{v}_{gc} = \frac{m}{qB^3}\left(\frac{v_\perp^2}{2} + v_\parallel^2\right)(\mathbf{B}\times\nabla B)
$$

### 3.4 Toroidal Field Geometry

In a tokamak, the toroidal field $B_\phi \propto 1/R$ creates both curvature and gradient:

```
    Tokamak cross-section (poloidal plane)

         Weak field              Strong field
         (large R)                (small R)
            |                         |
            |                         |
         ───┼────────────o────────────┼───
            |        plasma           |
            |         center          |
            |                         |

    Grad-B drift: ∇B points inward (toward major axis)
    For ions (+): drift downward (in poloidal plane)
    For electrons (−): drift upward

    → Charge separation → vertical electric field
    → E×B drift outward → particle loss

    (Solved by magnetic shear and toroidal rotation)
```

## 4. Polarization Drift

### 4.1 Time-Varying Electric Field

When the electric field varies in time, the $\mathbf{E}\times\mathbf{B}$ drift velocity changes, and the particle must accelerate. This creates an additional drift.

Starting from:

$$
m\frac{d\mathbf{v}}{dt} = q(\mathbf{E} + \mathbf{v}\times\mathbf{B})
$$

Decompose $\mathbf{v} = \mathbf{v}_E + \mathbf{v}_\perp + \mathbf{v}_\parallel$ where $\mathbf{v}_E = \mathbf{E}\times\mathbf{B}/B^2$ is the $\mathbf{E}\times\mathbf{B}$ drift.

Taking the time derivative of $\mathbf{v}_E$:

$$
\frac{d\mathbf{v}_E}{dt} = \frac{1}{B^2}\frac{d\mathbf{E}}{dt}\times\mathbf{B}
$$

The equation of motion for the perpendicular velocity becomes:

$$
m\frac{d\mathbf{v}_\perp}{dt} = q\mathbf{v}_\perp\times\mathbf{B} - m\frac{d\mathbf{v}_E}{dt}
$$

The last term acts like an additional force. Averaging over the gyroperiod and applying the general drift formula:

$$
\boxed{\mathbf{v}_P = \frac{m}{qB^2}\frac{d\mathbf{E}_\perp}{dt}}
$$

This is the **polarization drift**.

### 4.2 Mass Dependence

Notice that $\mathbf{v}_P \propto m/q$:
- For ions (heavy): significant drift
- For electrons (light): negligible drift (by factor $m_e/m_i \sim 1/1836$)

This creates a **polarization current**:

$$
\mathbf{J}_P = \sum_s n_s q_s \mathbf{v}_{P,s} \approx n_i q_i \mathbf{v}_{P,i} = \frac{n_i m_i}{B^2}\frac{d\mathbf{E}_\perp}{dt}
$$

The electron contribution is negligible.

### 4.3 Physical Interpretation

When $\mathbf{E}$ changes, the $\mathbf{E}\times\mathbf{B}$ drift changes. Ions, being heavy, cannot respond instantly and "overshoot" or "lag behind" the new drift velocity. This transient motion is the polarization drift.

```
    E increases suddenly

    Before:  vE (small)
             ───→

    After:   vE (large)
             ─────────→

    Ions lag: ──→  (smaller than new vE initially)
    Electrons respond instantly: ─────────→

    → Net current during transient
```

### 4.4 Polarization Drift in Waves

In plasma waves with frequency $\omega$, if $\mathbf{E} = \mathbf{E}_0 e^{-i\omega t}$:

$$
\frac{d\mathbf{E}}{dt} = -i\omega\mathbf{E}
$$

The polarization drift becomes:

$$
\mathbf{v}_P = -i\frac{m\omega}{qB^2}\mathbf{E}_\perp
$$

This is crucial in deriving the dielectric tensor for electromagnetic waves in magnetized plasmas.

## 5. Gravitational Drift

### 5.1 Drift in Gravitational Field

For a gravitational force $\mathbf{F}_g = m\mathbf{g}$, the drift is:

$$
\boxed{\mathbf{v}_g = \frac{m}{q}\frac{\mathbf{g}\times\mathbf{B}}{B^2}}
$$

This drift:
- Depends on sign of charge (ions and electrons drift in opposite directions)
- Proportional to mass (heavier particles drift faster)
- Independent of particle energy

### 5.2 Current from Gravitational Drift

Since ions and electrons drift in opposite directions:

$$
\mathbf{J}_g = n(q_i\mathbf{v}_{g,i} + q_e\mathbf{v}_{g,e}) = n(m_i - m_e)\frac{\mathbf{g}\times\mathbf{B}}{B^2} \approx n m_i\frac{\mathbf{g}\times\mathbf{B}}{B^2}
$$

In laboratory plasmas, gravity is usually negligible. But in astrophysical plasmas (e.g., solar atmosphere), gravitational drift can be important.

### 5.3 Numerical Example

**Example**: Proton in solar corona.
- $B = 10^{-3}$ T (strong active region), $g = 274$ m/s² (solar surface gravity)
- $m = 1.67\times 10^{-27}$ kg, $q = 1.6\times 10^{-19}$ C

$$
v_g = \frac{mg}{qB} = \frac{(1.67\times 10^{-27})(274)}{(1.6\times 10^{-19})(10^{-3})} \approx 2.9 \text{ m/s}
$$

Small but not negligible over solar timescales (hours to days).

## 6. Summary of All Drifts

### 6.1 Comprehensive Drift Table

| Drift Type | Formula | Charge Dependence | Mass Dependence | Physical Origin |
|------------|---------|-------------------|-----------------|-----------------|
| **General** | $\mathbf{F}\times\mathbf{B}/(qB^2)$ | Depends on $\mathbf{F}$ | Depends on $\mathbf{F}$ | Arbitrary force |
| **E×B** | $\mathbf{E}\times\mathbf{B}/B^2$ | Independent | Independent | Electric force |
| **Grad-B** | $\pm\frac{mv_\perp^2}{2qB^3}(\mathbf{B}\times\nabla B)$ | Opposite signs | $\propto m$ | Non-uniform $\|\mathbf{B}\|$ |
| **Curvature** | $\frac{mv_\parallel^2}{qB^2R_c^2}(\mathbf{R}_c\times\mathbf{B})$ | Opposite signs | $\propto m$ | Curved field lines |
| **Combined GC** | $\frac{m}{qB^2}(\frac{v_\perp^2}{2}+v_\parallel^2)\frac{\mathbf{B}\times\nabla B}{B}$ | Opposite signs | $\propto m$ | Grad-B + curvature |
| **Polarization** | $\frac{m}{qB^2}\frac{d\mathbf{E}_\perp}{dt}$ | Opposite signs | $\propto m$ | Time-varying E |
| **Gravitational** | $\frac{m}{q}\frac{\mathbf{g}\times\mathbf{B}}{B^2}$ | Opposite signs | $\propto m$ | Gravity |

### 6.2 Key Observations

1. **Charge-independent drifts**: Only $\mathbf{E}\times\mathbf{B}$ drift (no current)
2. **Charge-dependent drifts**: All others (produce currents)
3. **Mass-independent drift**: Only $\mathbf{E}\times\mathbf{B}$
4. **Mass-proportional drifts**: Grad-B, curvature, polarization, gravitational

The combination of these drifts determines:
- Particle confinement in magnetic traps
- Cross-field transport
- Current generation in plasmas
- Stability properties

### 6.3 Ordering of Drift Velocities

Typically, in magnetized plasmas:

$$
v_\parallel \sim v_{th} \gg v_E \sim \frac{E}{B}v_{th} \gg v_{\nabla B} \sim \frac{r_L}{L}v_{th} \gg v_P
$$

where $L$ is the gradient scale length. The hierarchy depends on:
- $E/B$ ratio
- Gradient scale $L/r_L$
- Time variation rate $\omega/\omega_c$

## 7. Python Implementations

### 7.1 Grad-B Drift Simulation

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
m_p = 1.67e-27  # proton mass (kg)
m_e = 9.11e-31  # electron mass (kg)
q_p = 1.6e-19   # proton charge (C)
q_e = -1.6e-19  # electron charge (C)

def magnetic_field_gradient(x, y, z):
    """
    Non-uniform magnetic field: B = B0(1 + alpha*x)*z_hat
    Creates a gradient in x-direction
    """
    B0 = 1e-3  # Tesla
    # alpha = 0.1 means B doubles over ~10 m — small enough that the guiding-center
    # approximation holds (r_L << gradient scale L = B/|∇B| = 1/alpha = 10 m)
    alpha = 0.1  # gradient parameter (1/m)

    Bx = 0
    By = 0
    Bz = B0 * (1 + alpha * x)

    return np.array([Bx, By, Bz])

def equations_of_motion_gradb(t, state, q, m):
    """
    Equations of motion in non-uniform B field
    state = [x, y, z, vx, vy, vz]
    """
    x, y, z, vx, vy, vz = state

    # Magnetic field at current position
    B = magnetic_field_gradient(x, y, z)

    # Lorentz force: F = q(v × B)
    v = np.array([vx, vy, vz])
    F = q * np.cross(v, B)

    # Acceleration
    ax, ay, az = F / m

    return np.array([vx, vy, vz, ax, ay, az])

def rk4_step(f, t, y, dt, q, m):
    # RK4 is chosen over simpler schemes (Euler, RK2) because the Lorentz force is
    # stiff on the gyration timescale; RK4's O(dt^5) local error keeps the gyro-orbit
    # from artificially gaining or losing energy over the many cycles we simulate.
    """4th-order Runge-Kutta step"""
    k1 = f(t, y, q, m)
    k2 = f(t + dt/2, y + dt*k1/2, q, m)
    k3 = f(t + dt/2, y + dt*k2/2, q, m)
    k4 = f(t + dt, y + dt*k3, q, m)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

def simulate_gradb_drift(particle_type='proton', v_perp=1e5, v_para=1e4,
                         duration=1e-3, dt=1e-8):
    """
    Simulate particle motion in grad-B field
    """
    # Particle properties
    if particle_type == 'proton':
        q, m = q_p, m_p
    else:
        q, m = q_e, m_e

    # Initial conditions
    x0, y0, z0 = 0.0, 0.0, 0.0
    # Start velocity in the x-z plane: v_perp drives gyration, v_para gives a helical
    # orbit. Starting vx = v_perp (not vz) so the initial gyration plane is x-y,
    # making the grad-B drift clearly visible in the y-direction.
    vx0, vy0, vz0 = v_perp, 0.0, v_para
    state = np.array([x0, y0, z0, vx0, vy0, vz0])

    # Time array
    # dt = 1e-8 s is chosen to be ~1/10 of the proton gyroperiod at B0 = 1e-3 T
    # (T_gyro = 2π m/qB ~ 6.5e-8 s), ensuring accurate resolution of gyration.
    num_steps = int(duration / dt)
    times = np.linspace(0, duration, num_steps)

    # Storage
    trajectory = np.zeros((num_steps, 6))
    trajectory[0] = state

    # Integration
    for i in range(1, num_steps):
        state = rk4_step(equations_of_motion_gradb, times[i-1], state, dt, q, m)
        trajectory[i] = state

    return times, trajectory

# Simulate proton and electron
print("Simulating grad-B drift...")
t_p, traj_p = simulate_gradb_drift('proton', v_perp=5e4, v_para=1e4,
                                    duration=5e-4, dt=1e-8)
t_e, traj_e = simulate_gradb_drift('electron', v_perp=5e6, v_para=1e6,
                                    duration=5e-7, dt=1e-11)

# Plotting
fig = plt.figure(figsize=(15, 5))

# 3D trajectory - Proton
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(traj_p[:, 0], traj_p[:, 1], traj_p[:, 2], 'b-', linewidth=0.5)
ax1.scatter([traj_p[0, 0]], [traj_p[0, 1]], [traj_p[0, 2]],
            color='green', s=50, label='Start')
ax1.scatter([traj_p[-1, 0]], [traj_p[-1, 1]], [traj_p[-1, 2]],
            color='red', s=50, label='End')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_zlabel('z (m)')
ax1.set_title('Proton Trajectory (Grad-B Drift)')
ax1.legend()
ax1.grid(True)

# 3D trajectory - Electron
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot(traj_e[:, 0], traj_e[:, 1], traj_e[:, 2], 'r-', linewidth=0.5)
ax2.scatter([traj_e[0, 0]], [traj_e[0, 1]], [traj_e[0, 2]],
            color='green', s=50, label='Start')
ax2.scatter([traj_e[-1, 0]], [traj_e[-1, 1]], [traj_e[-1, 2]],
            color='red', s=50, label='End')
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_zlabel('z (m)')
ax2.set_title('Electron Trajectory (Grad-B Drift)')
ax2.legend()
ax2.grid(True)

# XY projection showing drift
ax3 = fig.add_subplot(133)
ax3.plot(traj_p[:, 0], traj_p[:, 1], 'b-', linewidth=1, label='Proton')
ax3.plot(traj_e[:, 0]*1e3, traj_e[:, 1]*1e3, 'r-', linewidth=1, label='Electron (scaled 1000x)')
ax3.set_xlabel('x (m)')
ax3.set_ylabel('y (m)')
ax3.set_title('XY Projection: Opposite Drift Directions')
ax3.legend()
ax3.grid(True)
ax3.axis('equal')

plt.tight_layout()
plt.savefig('gradb_drift_simulation.png', dpi=150)
print("Saved: gradb_drift_simulation.png")

# Calculate drift velocity by measuring net y-displacement over the full simulation.
# Dividing total displacement by total time gives the average drift speed, filtering
# out the oscillatory gyration which sums to zero over complete cycles.
y_drift_p = traj_p[-1, 1] - traj_p[0, 1]
y_drift_e = traj_e[-1, 1] - traj_e[0, 1]
v_drift_p = y_drift_p / t_p[-1]
v_drift_e = y_drift_e / t_e[-1]

print(f"\nProton drift velocity: {v_drift_p:.2e} m/s (y-direction)")
print(f"Electron drift velocity: {v_drift_e:.2e} m/s (y-direction)")
print(f"Opposite directions: {np.sign(v_drift_p) != np.sign(v_drift_e)}")
```

### 7.2 Curvature Drift in Dipole Field

```python
def magnetic_field_dipole(r, theta, phi, M=1e15):
    """
    Dipole magnetic field in spherical coordinates
    B_r = (2M/r^3) cos(theta)
    B_theta = (M/r^3) sin(theta)
    B_phi = 0
    M: magnetic moment (A·m^2)
    """
    # Clamp r to avoid 1/r^3 singularity at the origin; physically, the dipole
    # formula is only valid outside the source (r >> source radius), so r_safe
    # just prevents a numerical crash if the integrator steps inside r = 0.
    r_safe = max(r, 0.1)  # avoid singularity

    B_r = (2 * M / r_safe**3) * np.cos(theta)
    B_theta = (M / r_safe**3) * np.sin(theta)
    B_phi = 0.0

    return np.array([B_r, B_theta, B_phi])

def spherical_to_cartesian_vector(v_sph, theta, phi):
    """Convert vector from spherical to Cartesian coordinates"""
    v_r, v_theta, v_phi = v_sph

    # Transformation matrix
    v_x = v_r * np.sin(theta) * np.cos(phi) + v_theta * np.cos(theta) * np.cos(phi) - v_phi * np.sin(phi)
    v_y = v_r * np.sin(theta) * np.sin(phi) + v_theta * np.cos(theta) * np.sin(phi) + v_phi * np.cos(phi)
    v_z = v_r * np.cos(theta) - v_theta * np.sin(theta)

    return np.array([v_x, v_y, v_z])

def cartesian_to_spherical(x, y, z):
    """Convert Cartesian to spherical coordinates"""
    r = np.sqrt(x**2 + y**2 + z**2)
    r = max(r, 1e-10)
    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.arctan2(y, x)
    return r, theta, phi

def equations_of_motion_dipole(t, state, q, m, M):
    """
    Equations of motion in dipole field (Cartesian coordinates)
    """
    x, y, z, vx, vy, vz = state

    # Convert position to spherical
    r, theta, phi = cartesian_to_spherical(x, y, z)

    # Magnetic field in spherical coordinates
    B_sph = magnetic_field_dipole(r, theta, phi, M)

    # Convert B to Cartesian
    B = spherical_to_cartesian_vector(B_sph, theta, phi)

    # Lorentz force
    v = np.array([vx, vy, vz])
    F = q * np.cross(v, B)

    # Acceleration
    ax, ay, az = F / m

    return np.array([vx, vy, vz, ax, ay, az])

def simulate_dipole_drift(particle_type='proton', r0=1e6, theta0=np.pi/4,
                         v_perp=1e5, v_para=5e4, duration=10.0, dt=1e-3):
    """
    Simulate particle in dipole field (e.g., Earth's magnetosphere)
    """
    # Particle properties
    if particle_type == 'proton':
        q, m = q_p, m_p
    else:
        q, m = q_e, m_e

    # Earth's magnetic moment
    M = 8e15  # A·m^2 (approximate)

    # Initial position (spherical)
    phi0 = 0.0
    x0 = r0 * np.sin(theta0) * np.cos(phi0)
    y0 = r0 * np.sin(theta0) * np.sin(phi0)
    z0 = r0 * np.cos(theta0)

    # Initial velocity (perpendicular and parallel to B)
    B_sph = magnetic_field_dipole(r0, theta0, phi0, M)
    B_cart = spherical_to_cartesian_vector(B_sph, theta0, phi0)
    B_mag = np.linalg.norm(B_cart)
    b_hat = B_cart / B_mag

    # Perpendicular direction: project x-hat onto the plane perpendicular to B,
    # then normalise. This gives a well-defined perpendicular basis vector without
    # requiring the full cross-product formalism, as long as B is not exactly along x.
    perp1 = np.array([1, 0, 0]) - np.dot([1, 0, 0], b_hat) * b_hat
    perp1 /= np.linalg.norm(perp1)

    # Decompose initial velocity into guiding-center components so the pitch angle
    # at t=0 is physically meaningful: v_para along B, v_perp initiates gyration.
    v0 = v_para * b_hat + v_perp * perp1
    vx0, vy0, vz0 = v0

    state = np.array([x0, y0, z0, vx0, vy0, vz0])

    # Time array
    num_steps = int(duration / dt)
    times = np.linspace(0, duration, num_steps)

    # Storage
    trajectory = np.zeros((num_steps, 6))
    trajectory[0] = state

    # Integration
    for i in range(1, num_steps):
        state = rk4_step(lambda t, s, q, m: equations_of_motion_dipole(t, s, q, m, M),
                        times[i-1], state, dt, q, m)
        trajectory[i] = state

    return times, trajectory

# Simulate proton in dipole field
print("\nSimulating curvature drift in dipole field...")
t_dip, traj_dip = simulate_dipole_drift('proton', r0=2e6, theta0=np.pi/3,
                                        v_perp=5e4, v_para=3e4,
                                        duration=100.0, dt=0.01)

# Plot
fig = plt.figure(figsize=(12, 10))

# 3D trajectory
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot(traj_dip[:, 0]/1e6, traj_dip[:, 1]/1e6, traj_dip[:, 2]/1e6,
         'b-', linewidth=0.8)
ax1.scatter([0], [0], [0], color='cyan', s=200, marker='o', label='Earth')
ax1.set_xlabel('x (Mm)')
ax1.set_ylabel('y (Mm)')
ax1.set_zlabel('z (Mm)')
ax1.set_title('Proton Trajectory in Dipole Field')
ax1.legend()
ax1.grid(True)

# XY projection
ax2 = fig.add_subplot(222)
ax2.plot(traj_dip[:, 0]/1e6, traj_dip[:, 1]/1e6, 'b-', linewidth=0.8)
ax2.scatter([0], [0], color='cyan', s=200, marker='o', label='Earth')
ax2.set_xlabel('x (Mm)')
ax2.set_ylabel('y (Mm)')
ax2.set_title('XY Projection')
ax2.axis('equal')
ax2.grid(True)
ax2.legend()

# XZ projection
ax3 = fig.add_subplot(223)
ax3.plot(traj_dip[:, 0]/1e6, traj_dip[:, 2]/1e6, 'b-', linewidth=0.8)
ax3.scatter([0], [0], color='cyan', s=200, marker='o', label='Earth')
ax3.set_xlabel('x (Mm)')
ax3.set_ylabel('z (Mm)')
ax3.set_title('XZ Projection (Meridional Plane)')
ax3.axis('equal')
ax3.grid(True)
ax3.legend()

# Radial distance vs time
r_traj = np.sqrt(traj_dip[:, 0]**2 + traj_dip[:, 1]**2 + traj_dip[:, 2]**2)
ax4 = fig.add_subplot(224)
ax4.plot(t_dip, r_traj/1e6, 'b-', linewidth=1)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Radial Distance (Mm)')
ax4.set_title('Radial Distance vs Time')
ax4.grid(True)

plt.tight_layout()
plt.savefig('curvature_drift_dipole.png', dpi=150)
print("Saved: curvature_drift_dipole.png")
```

### 7.3 Comparison of All Drifts

```python
def compute_drift_velocities(B=1e-3, E=1e-2, grad_B=1e-5, R_c=1.0,
                            dE_dt=1.0, g=9.8, v_perp=1e5, v_para=1e5,
                            particle='proton'):
    """
    Compute magnitudes of all drift velocities for comparison
    """
    if particle == 'proton':
        q, m = q_p, m_p
    else:
        q, m = q_e, m_e

    # E×B drift — independent of charge and mass, so both species drift identically
    # and no current is generated; E and B must be perpendicular for this formula.
    v_ExB = E / B

    # Grad-B drift: m*v_perp^2 / (2|q|B^2) * (|∇B|/B).
    # The ratio ∇B/B = 1/L is the inverse gradient scale length; small L (strong
    # gradient) gives larger drift — consistent with the guiding-center expansion.
    v_gradB = (m * v_perp**2) / (2 * abs(q) * B**2) * (grad_B / B)

    # Curvature drift: uses v_para^2 (not v_perp^2) because centrifugal force depends
    # on parallel motion along the curved field line, not perpendicular gyration.
    v_curv = (m * v_para**2) / (abs(q) * B**2 * R_c)

    # Combined grad-B + curvature: in realistic geometries the two drifts always
    # co-exist (toroidal 1/R field has both curvature and ∇B), so summing them gives
    # the physically correct guiding-center drift used in tokamak transport analysis.
    v_gc = (m / (abs(q) * B**2)) * (v_perp**2/2 + v_para**2) * (grad_B / B)

    # Polarization drift
    v_pol = (m / (abs(q) * B**2)) * dE_dt

    # Gravitational drift
    v_grav = (m * g) / (abs(q) * B)

    return {
        'E×B': v_ExB,
        'Grad-B': v_gradB,
        'Curvature': v_curv,
        'GC (combined)': v_gc,
        'Polarization': v_pol,
        'Gravitational': v_grav,
        'Parallel': v_para,
        'Perpendicular': v_perp
    }

# Compute for typical tokamak parameters
print("\n=== Drift Velocity Comparison ===")
print("\nTokamak Parameters:")
print("B = 2 T, E = 1 kV/m, ∇B/B = 0.1 m⁻¹, Rc = 3 m")
print("v_perp = v_para = 1e5 m/s")

drifts_p = compute_drift_velocities(B=2.0, E=1e3, grad_B=0.2, R_c=3.0,
                                    dE_dt=1e4, g=9.8,
                                    v_perp=1e5, v_para=1e5,
                                    particle='proton')

drifts_e = compute_drift_velocities(B=2.0, E=1e3, grad_B=0.2, R_c=3.0,
                                    dE_dt=1e4, g=9.8,
                                    v_perp=1e5, v_para=1e5,
                                    particle='electron')

print("\nProton drifts:")
for name, value in drifts_p.items():
    if name not in ['Parallel', 'Perpendicular']:
        print(f"  {name:20s}: {value:12.3e} m/s")

print("\nElectron drifts:")
for name, value in drifts_e.items():
    if name not in ['Parallel', 'Perpendicular']:
        print(f"  {name:20s}: {value:12.3e} m/s")

# Visualization
drift_names = ['E×B', 'Grad-B', 'Curv.', 'GC', 'Polar.', 'Grav.']
drift_values_p = [drifts_p['E×B'], drifts_p['Grad-B'], drifts_p['Curvature'],
                  drifts_p['GC (combined)'], drifts_p['Polarization'],
                  drifts_p['Gravitational']]
drift_values_e = [drifts_e['E×B'], drifts_e['Grad-B'], drifts_e['Curvature'],
                  drifts_e['GC (combined)'], drifts_e['Polarization'],
                  drifts_e['Gravitational']]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Proton drifts
x_pos = np.arange(len(drift_names))
bars1 = ax1.bar(x_pos, drift_values_p, color='blue', alpha=0.7)
ax1.set_yscale('log')
ax1.set_ylabel('Drift Velocity (m/s)', fontsize=12)
ax1.set_xlabel('Drift Type', fontsize=12)
ax1.set_title('Proton Drift Velocities (Tokamak)', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(drift_names, rotation=45, ha='right')
ax1.grid(True, alpha=0.3, which='both')
ax1.axhline(y=drifts_p['Parallel'], color='red', linestyle='--',
            linewidth=2, label=f"v_parallel = {drifts_p['Parallel']:.1e} m/s")
ax1.legend()

# Add values on bars
for i, (bar, val) in enumerate(zip(bars1, drift_values_p)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1e}', ha='center', va='bottom', fontsize=8, rotation=0)

# Electron drifts
bars2 = ax2.bar(x_pos, drift_values_e, color='red', alpha=0.7)
ax2.set_yscale('log')
ax2.set_ylabel('Drift Velocity (m/s)', fontsize=12)
ax2.set_xlabel('Drift Type', fontsize=12)
ax2.set_title('Electron Drift Velocities (Tokamak)', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(drift_names, rotation=45, ha='right')
ax2.grid(True, alpha=0.3, which='both')
ax2.axhline(y=drifts_e['Parallel'], color='blue', linestyle='--',
            linewidth=2, label=f"v_parallel = {drifts_e['Parallel']:.1e} m/s")
ax2.legend()

# Add values on bars
for i, (bar, val) in enumerate(zip(bars2, drift_values_e)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1e}', ha='center', va='bottom', fontsize=8, rotation=0)

plt.tight_layout()
plt.savefig('drift_comparison.png', dpi=150)
print("\nSaved: drift_comparison.png")
```

### 7.4 Drift Current Visualization

```python
def calculate_drift_currents(n=1e19, B=2.0, T=1e3, grad_B=0.2,
                             E_field=1e3, dE_dt=1e4, g=9.8):
    """
    Calculate current densities from different drifts
    n: plasma density (m^-3)
    T: temperature (eV)
    B: magnetic field (T)
    """
    # Convert temperature to SI
    T_J = T * q_p  # Joules

    # Thermal velocity — computed here for completeness but not used in current
    # formulas because the drift current expressions are already averaged over the
    # Maxwellian (v_perp^2 → k_B T / m), making explicit v_th unnecessary.
    v_th_p = np.sqrt(2 * T_J / m_p)
    v_th_e = np.sqrt(2 * T_J / m_e)

    # E×B drift produces no current because both species drift in the same direction
    # and at the same speed (charge/mass cancel), so opposite charges cancel out.
    J_ExB = 0.0

    # Grad-B current: J = n * (q_i v_gradB,i + q_e v_gradB,e).
    # Each species contributes nkT/B^2 * ∇B (same magnitude, opposite sign of charge,
    # but opposite drift direction), so they ADD rather than cancel — diamagnetic current.
    J_gradB = n * 2 * T_J / B**2 * grad_B

    # Polarization current comes almost entirely from ions: m_i/m_e ~ 1836, so the
    # electron contribution is negligible. Only ion mass enters the formula.
    J_pol = n * m_p / B**2 * dE_dt

    # Gravitational current: ions dominate (m_i >> m_e); the formula uses m_p
    # because even for heavier ions the proton approximation is illustrative here.
    J_grav = n * m_p * g / B

    return {
        'E×B': J_ExB,
        'Grad-B': J_gradB,
        'Polarization': J_pol,
        'Gravitational': J_grav
    }

# Calculate currents
print("\n=== Drift Current Densities ===")
print("Plasma: n = 1e19 m^-3, T = 1 keV, B = 2 T")

currents = calculate_drift_currents(n=1e19, B=2.0, T=1e3, grad_B=0.2,
                                    E_field=1e3, dE_dt=1e4, g=9.8)

print("\nCurrent densities:")
for name, value in currents.items():
    print(f"  J_{name:20s}: {value:12.3e} A/m²")

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))

current_names = list(currents.keys())
current_values = list(currents.values())
colors = ['gray', 'blue', 'orange', 'green']

bars = ax.bar(current_names, current_values, color=colors, alpha=0.7, edgecolor='black')

ax.set_ylabel('Current Density (A/m²)', fontsize=14, fontweight='bold')
ax.set_xlabel('Drift Type', fontsize=14, fontweight='bold')
ax.set_title('Current Densities from Different Drifts', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, val in zip(bars, current_values):
    height = bar.get_height()
    if val != 0:
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2e}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2., 0.1*max(current_values),
               'No current', ha='center', va='bottom', fontsize=10,
               fontweight='bold', style='italic')

plt.tight_layout()
plt.savefig('drift_currents.png', dpi=150)
print("\nSaved: drift_currents.png")
```

## Summary

In this lesson, we explored the rich variety of drift motions in magnetized plasmas:

1. **General drift formula**: $\mathbf{v}_D = \mathbf{F}\times\mathbf{B}/(qB^2)$ applies to any force perpendicular to $\mathbf{B}$.

2. **Grad-B drift**: Non-uniform magnetic field strength causes particles to drift perpendicular to both $\mathbf{B}$ and $\nabla B$. Direction depends on charge sign → produces current.

3. **Curvature drift**: Centrifugal force in curved field lines causes drift. Combined with grad-B drift gives the total guiding center drift.

4. **Polarization drift**: Time-varying electric fields cause mass-dependent drift. Important for ions, negligible for electrons → polarization current.

5. **Gravitational drift**: Gravity (or any mass-proportional force) causes charge-dependent drift.

6. **Drift currents**: Most drifts (except $\mathbf{E}\times\mathbf{B}$) produce currents because ions and electrons drift differently.

7. **Plasma confinement**: Understanding these drifts is crucial for:
   - Magnetic fusion (tokamaks, stellarators)
   - Space plasmas (magnetosphere, solar wind)
   - Astrophysical plasmas

The hierarchy of timescales — gyration ($\omega_c^{-1}$), bounce ($\omega_b^{-1}$), drift ($\omega_d^{-1}$) — allows systematic perturbation theory and the guiding center approximation.

## Practice Problems

### Problem 1: Grad-B Drift in a Mirror

A proton with energy $W = 10$ keV and pitch angle $\alpha = 60°$ (at the midplane) is confined in a magnetic mirror with $B_{min} = 0.5$ T and $B_{max} = 2$ T.

(a) Calculate the magnetic moment $\mu$ and verify it is conserved.

(b) Compute the grad-B drift velocity at the midplane if the gradient scale length is $L = \frac{B}{|\nabla B|} = 1$ m.

(c) Estimate the drift orbit circumference and the time for one complete drift orbit.

(d) Compare the drift period with the bounce period.

**Hint**: Use $v_\perp = v\sin\alpha$, $v_\parallel = v\cos\alpha$, and the mirror force $F_\parallel = -\mu\nabla_\parallel B$.

---

### Problem 2: Tokamak Vertical Drift

In a tokamak with major radius $R_0 = 2$ m, the toroidal field is $B_\phi(R) = B_0 R_0/R$ where $B_0 = 3$ T. Consider a deuteron with $v_\perp = 1\times 10^5$ m/s and $v_\parallel = 5\times 10^5$ m/s at the outboard midplane ($R = R_0 + a$ where $a = 0.5$ m is the minor radius).

(a) Calculate the grad-B drift velocity (vertical direction).

(b) Calculate the curvature drift velocity.

(c) Compute the total guiding center drift and compare with the individual contributions.

(d) In a simple model without toroidal rotation, this vertical drift causes charge separation and a vertical electric field. Estimate the $\mathbf{E}\times\mathbf{B}$ drift this field produces (radial direction). This is a particle loss mechanism — why?

**Hint**: For a toroidal field, $\nabla B \approx -\frac{B}{R}\hat{R}$ and $R_c \approx R_0$.

---

### Problem 3: Polarization Current in a Wave

An ion acoustic wave in a plasma has electric field $\mathbf{E} = E_0 \cos(kx - \omega t)\hat{y}$ in a magnetic field $\mathbf{B} = B_0\hat{z}$. The wave frequency is $\omega = 10^5$ rad/s, $E_0 = 100$ V/m, $B_0 = 1$ T, and the ion density is $n_i = 10^{18}$ m$^{-3}$.

(a) Calculate the $\mathbf{E}\times\mathbf{B}$ drift velocity (time-dependent).

(b) Calculate the polarization drift velocity for ions (deuterons).

(c) Compute the polarization current density $\mathbf{J}_P$.

(d) Compare the magnitudes of $\mathbf{v}_E$ and $\mathbf{v}_P$. Under what condition is polarization drift important?

**Hint**: $\frac{d\mathbf{E}}{dt} = -\frac{\partial \mathbf{E}}{\partial t} - \mathbf{v}\cdot\nabla\mathbf{E}$. For slow drifts, approximate $\frac{d\mathbf{E}}{dt} \approx -\frac{\partial \mathbf{E}}{\partial t}$.

---

### Problem 4: Loss Cone and Grad-B Drift

In a magnetic mirror device, particles with small pitch angles escape through the loss cone. Grad-B drift causes particles to drift azimuthally around the axis.

(a) For a mirror with circular symmetry (axisymmetric), explain why grad-B drift does not cause particle loss.

(b) Consider a "minimum-B" configuration where $\mathbf{B}$ has a local minimum (quadrupole field). Show that the grad-B drift direction reverses compared to a simple mirror.

(c) In a real mirror, asymmetries break the symmetry. If the field has a small non-axisymmetric component $\delta B/B \sim 0.01$, estimate the time for a particle to drift into the loss cone due to this asymmetry.

**Hint**: In minimum-B, $\nabla B$ points outward from the minimum, opposite to a mirror maximum.

---

### Problem 5: Gravitational Sedimentation in Solar Prominences

Solar prominences are cool, dense plasma structures suspended in the hot corona by magnetic fields. Estimate the gravitational drift of protons in a prominence.

Parameters:
- Solar surface gravity: $g = 274$ m/s²
- Magnetic field: $B = 5\times 10^{-3}$ T (typical prominence field)
- Assume the field is horizontal and gravity is vertical (toward solar surface)

(a) Calculate the gravitational drift velocity for protons and electrons.

(b) Estimate the current density if the prominence density is $n = 10^{16}$ m$^{-3}$.

(c) This current creates a magnetic field (Ampère's law). Estimate the magnitude of this induced field if the prominence thickness is $L = 10^7$ m.

(d) Discuss whether this self-generated field can support the prominence against gravity (compare magnetic pressure $B^2/(2\mu_0)$ with gravitational pressure $\rho g L$).

**Hint**: The gravitational current is $\mathbf{J}_g \approx n m_p \mathbf{g}\times\mathbf{B}/B^2$. Use Ampère's law $\nabla\times\mathbf{B} = \mu_0\mathbf{J}$ to estimate $\delta B \sim \mu_0 J L$.

---

## Navigation

- **Previous**: [Single Particle Motion I](./04_Single_Particle_Motion_I.md)
- **Next**: [Magnetic Mirrors and Adiabatic Invariants](./06_Magnetic_Mirrors_Adiabatic_Invariants.md)
