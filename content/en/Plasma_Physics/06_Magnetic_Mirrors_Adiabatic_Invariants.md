# 6. Magnetic Mirrors and Adiabatic Invariants

## Learning Objectives

- Understand the magnetic mirror force and its role in particle confinement
- Derive the first adiabatic invariant (magnetic moment) and prove its conservation
- Analyze magnetic mirror geometry, loss cone, and trapped vs. passing particles
- Explore the hierarchy of adiabatic invariants and their associated timescales
- Study tokamak orbits (banana orbits and passing particles)
- Simulate magnetic mirror confinement and loss cone dynamics using Python

## 1. Magnetic Mirror Force

### 1.1 Converging Field Lines

In a region where magnetic field lines converge (field strength increases), a gyrating particle experiences a force opposing the motion toward higher field regions. This is the **magnetic mirror force**.

Consider a particle moving along a field line where $B = B(s)$ varies with arc length $s$:

```
    Weak field              Strong field
    B_low                   B_high
      ↓                       ↓
      |                       |||
      |        particle       |||
      |  ←──── moving ────→   |||
      |          →            |||
      |                       |||

    Large r_L              Small r_L
    (gyro-radius)          (gyro-radius)
```

The particle's perpendicular velocity component gyrates with radius $r_L = mv_\perp/(|q|B)$. As $B$ increases, $r_L$ decreases, but the perpendicular kinetic energy changes due to the work done by the Lorentz force.

### 1.2 Derivation of Mirror Force

The magnetic moment is defined as:

$$
\mu = \frac{mv_\perp^2}{2B} = \frac{W_\perp}{B}
$$

where $W_\perp = \frac{1}{2}mv_\perp^2$ is the perpendicular kinetic energy.

For slowly varying fields (adiabatic limit), $\mu$ is conserved (we'll prove this rigorously in Section 2). Assuming $\mu$ = constant:

$$
W_\perp = \mu B = \text{constant} \times B
$$

The total energy is conserved:

$$
W = W_\perp + W_\parallel = \frac{1}{2}m(v_\perp^2 + v_\parallel^2) = \text{constant}
$$

Therefore:

$$
\frac{dW_\parallel}{ds} = -\frac{dW_\perp}{ds} = -\mu\frac{dB}{ds}
$$

The parallel force is:

$$
F_\parallel = \frac{dW_\parallel}{ds} = -\mu\frac{dB}{ds} = -\mu\nabla_\parallel B
$$

**This is the magnetic mirror force**: it opposes motion toward regions of higher $B$.

### 1.3 Physical Interpretation

The mirror force arises because the Lorentz force $\mathbf{F} = q\mathbf{v}\times\mathbf{B}$ has a component along the field line when $\mathbf{B}$ is non-uniform. Consider a particle gyrating in a converging field:

```
    Top of gyro-orbit (moving toward high B)
         ↑ v_perp
         |
    ────┼──── B field line
         |
         ↓ Lorentz force has component opposing motion

    Bottom of gyro-orbit (moving toward low B)
         |
    ────┼──── B field line
         ↓ v_perp
         ↑ Lorentz force has component aiding motion

    Net effect: force opposing motion toward high B
```

Averaging over a gyroperiod gives the mirror force $F_\parallel = -\mu\nabla_\parallel B$.

## 2. First Adiabatic Invariant: Magnetic Moment μ

### 2.1 Conservation Proof via Action-Angle Variables

The magnetic moment $\mu$ is the first adiabatic invariant. To prove its conservation, we use the action-angle formalism from classical mechanics.

The action associated with gyration is:

$$
J_\perp = \oint p_\perp \, dq_\perp
$$

where the integral is over one gyroperiod. For circular motion in a magnetic field:

$$
J_\perp = m v_\perp \cdot 2\pi r_L = m v_\perp \cdot 2\pi \frac{mv_\perp}{|q|B} = \frac{2\pi m^2 v_\perp^2}{|q|B}
$$

The adiabatic invariant is:

$$
I = \frac{J_\perp}{2\pi} = \frac{m^2 v_\perp^2}{|q|B} = \frac{2m}{|q|}\frac{mv_\perp^2}{2B} = \frac{2m}{|q|}\mu
$$

Thus, $\mu$ is conserved when the field varies slowly compared to the gyroperiod:

$$
\frac{1}{\omega_c}\frac{dB}{dt} \ll B
$$

This is the **adiabatic condition** or **slow variation condition**.

### 2.2 Physical Meaning: Flux Conservation

The magnetic moment can also be interpreted as the magnetic flux through the gyro-orbit:

$$
\Phi_\text{gyro} = \pi r_L^2 B = \pi \left(\frac{mv_\perp}{|q|B}\right)^2 B = \frac{\pi m^2 v_\perp^2}{q^2 B}
$$

This is proportional to $\mu$:

$$
\Phi_\text{gyro} = \frac{\pi m^2 v_\perp^2}{q^2 B} = \frac{\pi m}{q^2} \cdot \frac{mv_\perp^2}{B} = \frac{2\pi m}{q^2}\mu
$$

**Conservation of $\mu$ means the magnetic flux through the gyro-orbit is conserved** (for slowly varying fields).

### 2.3 Breakdown of Adiabaticity

The adiabatic invariance breaks down when:

1. **Rapid field variation**: $\frac{1}{\omega_c}\frac{dB}{dt} \sim B$ (comparable to gyroperiod)
2. **Strong gradients**: $r_L |\nabla B| / B \sim 1$ (gradient scale comparable to gyro-radius)
3. **Resonances**: External perturbations at frequency $\omega \approx n\omega_c$ (gyro-resonance)

Example: In magnetic reconnection regions, $B$ can change rapidly, violating adiabaticity and allowing particles to change $\mu$.

## 3. Magnetic Mirror and Bottle

### 3.1 Mirror Geometry

A magnetic mirror consists of two regions of strong field (mirrors) separated by a weak field region:

```
    Mirror 1              Midplane           Mirror 2
    B_max                   B_min             B_max
      |||                     |                 |||
      |||                     |                 |||
      ||| ←─── particle ───→  |  ←────────────→ |||
      |||      trapped        |                 |||
      |||                     |                 |||
       ↑                      ↑                  ↑
    Reflection            Equator          Reflection
    point                                   point
```

The mirror ratio is:

$$
R = \frac{B_{\text{max}}}{B_{\text{min}}}
$$

### 3.2 Pitch Angle and Loss Cone

The pitch angle $\alpha$ is the angle between the velocity and the magnetic field:

$$
\alpha = \arctan\left(\frac{v_\perp}{v_\parallel}\right)
$$

At the midplane (where $B = B_0 = B_{\text{min}}$), the pitch angle is $\alpha_0$:

$$
\tan\alpha_0 = \frac{v_{\perp,0}}{v_{\parallel,0}}
$$

Using conservation of $\mu$ and total energy:

$$
\mu = \frac{mv_\perp^2}{2B} = \text{constant}
$$

$$
v^2 = v_\perp^2 + v_\parallel^2 = \text{constant}
$$

At the mirror point (where $v_\parallel = 0$), $v_\perp = v$ and $B = B_{\text{mirror}}$:

$$
\frac{mv^2}{2B_{\text{mirror}}} = \frac{mv_{\perp,0}^2}{2B_0}
$$

Therefore:

$$
\frac{v_\perp^2}{v^2}\bigg|_{\text{mirror}} = 1 = \frac{v_{\perp,0}^2}{v^2} \cdot \frac{B_{\text{mirror}}}{B_0}
$$

Since $v_{\perp,0}^2/v^2 = \sin^2\alpha_0$:

$$
\sin^2\alpha_0 = \frac{B_0}{B_{\text{mirror}}}
$$

For the particle to be **trapped** (reflected before reaching $B_{\text{max}}$):

$$
\sin^2\alpha_0 > \frac{B_0}{B_{\text{max}}} = \frac{1}{R}
$$

Or equivalently:

$$
\boxed{\alpha_0 > \alpha_c = \arcsin\left(\frac{1}{\sqrt{R}}\right)}
$$

This defines the **loss cone**: particles with $\alpha_0 < \alpha_c$ are not trapped and escape.

### 3.3 Loss Cone Solid Angle

The loss cone in velocity space is a cone around the $v_\parallel$ axis:

```
    Velocity space

         v_parallel
            ↑
            |       Passing particles
            |      (escape)
            |     /
            |    / α_c (loss cone angle)
            |   /
            |  /_______________
            | /               /
            |/_______________/ ← Loss cone
           /|
          / |
         /  |
        /   |
    v_perp  | Trapped particles
            | (confined)
```

The solid angle of the loss cone is:

$$
\Delta\Omega = 2\pi(1 - \cos\alpha_c) = 2\pi\left(1 - \sqrt{1 - \frac{1}{R}}\right)
$$

For large mirror ratio $R \gg 1$:

$$
\Delta\Omega \approx \frac{\pi}{R}
$$

The fraction of particles in the loss cone (assuming isotropic distribution):

$$
f_{\text{loss}} = \frac{\Delta\Omega}{4\pi} \approx \frac{1}{4R}
$$

### 3.4 Magnetic Bottle

A **magnetic bottle** is a mirror configuration with closed field lines, providing confinement in all directions perpendicular to the field. Examples:
- Simple mirror devices
- Biconic cusp
- Minimum-B configurations (quadrupole fields)

The main loss mechanism is:
1. **Scattering into loss cone**: Collisions change pitch angle, scattering trapped particles into the loss cone.
2. **End losses**: Particles in the loss cone escape through the mirror throats.

The confinement time is:

$$
\tau_{\text{conf}} \sim \frac{R}{\nu_c}\frac{1}{f_{\text{loss}}}
$$

where $\nu_c$ is the collision frequency.

## 4. Bounce Motion

### 4.1 Bounce Trajectory

A trapped particle bounces between the two mirror points, oscillating in the parallel direction while drifting perpendicular to $\mathbf{B}$ (from grad-B and curvature drifts).

The parallel velocity is:

$$
v_\parallel = \pm\sqrt{v^2 - v_\perp^2} = \pm v\sqrt{1 - \frac{\mu B}{W}}
$$

where the $\pm$ depends on direction of motion. At the mirror point, $v_\parallel = 0$, so:

$$
B_{\text{mirror}} = \frac{W}{\mu} = \frac{mv^2}{2\mu}
$$

### 4.2 Bounce Frequency

The bounce period is:

$$
\tau_b = 2\int_0^{s_{\text{mirror}}} \frac{ds}{v_\parallel(s)}
$$

where $s$ is arc length along the field line and $s_{\text{mirror}}$ is the distance to the mirror point.

For a parabolic mirror with $B(z) = B_0(1 + z^2/L^2)$:

$$
\tau_b \approx \frac{4L}{v_\parallel}
$$

The bounce frequency is:

$$
\omega_b = \frac{2\pi}{\tau_b} \approx \frac{\pi v_\parallel}{2L}
$$

For typical parameters:
- $v_\parallel \sim 10^5$ m/s
- $L \sim 10$ m (mirror length)
- $\omega_b \sim 10^4$ rad/s

This is much slower than the gyrofrequency $\omega_c \sim 10^8$ rad/s.

### 4.3 Bounce-Averaged Drift

The grad-B and curvature drifts vary along the field line. For long-term behavior, we average over a bounce period:

$$
\langle\mathbf{v}_D\rangle_b = \frac{1}{\tau_b}\int_0^{\tau_b} \mathbf{v}_D(s(t)) \, dt
$$

This bounce-averaged drift determines the slow evolution of the particle orbit.

## 5. Second and Third Adiabatic Invariants

### 5.1 Hierarchy of Timescales

There are three natural timescales in magnetized plasmas:

$$
\omega_c \gg \omega_b \gg \omega_d
$$

where:
- $\omega_c = |q|B/m$: gyrofrequency
- $\omega_b \sim v_\parallel/L$: bounce frequency
- $\omega_d \sim v_D/R$: drift frequency

Each timescale is associated with an adiabatic invariant.

### 5.2 Second Adiabatic Invariant: J

The second invariant is associated with bounce motion:

$$
\boxed{J = \oint m v_\parallel \, ds}
$$

where the integral is along the field line between the two mirror points. This is the action for bounce motion.

**Conservation**: $J$ is conserved when the magnetic field varies slowly compared to the bounce period:

$$
\frac{1}{\omega_b}\frac{\partial B}{\partial t} \ll B
$$

### 5.3 Physical Meaning of J

$J$ is related to the area enclosed in $(s, p_\parallel)$ phase space:

```
    p_parallel = m v_parallel
         ↑
         |
         |    /\
         |   /  \    Particle bounces
         |  /    \   between mirror points
         | /      \
         |/        \___
        ─┼──────────────→ s (position along field line)
         0       s_mirror

    Area enclosed = J
```

Conservation of $J$ means particles remain on the same bounce orbit if the field changes slowly.

### 5.4 Third Adiabatic Invariant: Φ

The third invariant is associated with drift motion around the axis:

$$
\boxed{\Phi = \oint \mathbf{A}\cdot d\mathbf{l}}
$$

where the integral is around the drift orbit and $\mathbf{A}$ is the vector potential ($\mathbf{B} = \nabla\times\mathbf{A}$).

Using Stokes' theorem:

$$
\Phi = \int_S (\nabla\times\mathbf{A})\cdot d\mathbf{S} = \int_S \mathbf{B}\cdot d\mathbf{S}
$$

**$\Phi$ is the magnetic flux enclosed by the drift orbit.**

**Conservation**: $\Phi$ is conserved when the magnetic field varies slowly compared to the drift period:

$$
\frac{1}{\omega_d}\frac{\partial B}{\partial t} \ll B
$$

### 5.5 Summary of Adiabatic Invariants

| Invariant | Formula | Associated Motion | Timescale | Conservation Condition |
|-----------|---------|-------------------|-----------|----------------------|
| **μ** | $\frac{mv_\perp^2}{2B}$ | Gyration | $\omega_c^{-1}$ | $\tau_c \ll \tau_{\text{var}}$ |
| **J** | $\oint mv_\parallel ds$ | Bounce | $\omega_b^{-1}$ | $\tau_b \ll \tau_{\text{var}}$ |
| **Φ** | $\oint \mathbf{A}\cdot d\mathbf{l}$ | Drift | $\omega_d^{-1}$ | $\tau_d \ll \tau_{\text{var}}$ |

where $\tau_{\text{var}}$ is the characteristic time for field variation.

**Key principle**: Each invariant is conserved when the field varies slowly compared to the associated motion period.

## 6. Tokamak Orbits

### 6.1 Tokamak Geometry Review

A tokamak has:
- **Toroidal field** $B_\phi \propto 1/R$ (decreases with major radius)
- **Poloidal field** $B_\theta$ (from plasma current)
- **Total field**: $\mathbf{B} = B_\phi\hat{\phi} + B_\theta\hat{\theta}$

The field is stronger on the inboard side (small $R$) than the outboard side (large $R$):

```
    Top view of tokamak

         Weak field
         (outboard)
             |
        ─────┼─────
       /     |     \
      /      o      \   ← Plasma
     │  (magnetic   │
     │    axis)     │
      \            /
       \          /
        ──────────
             |
         Strong field
         (inboard)
```

This creates a magnetic mirror effect in the toroidal direction.

### 6.2 Trapped and Passing Particles

Due to the $1/R$ variation of $B_\phi$, particles can be:

1. **Passing particles**: Complete full poloidal circuits around the torus.
2. **Trapped particles**: Reflect between the inboard side (high $B$) and cannot complete the circuit.

The trapping condition is similar to a magnetic mirror. At the outboard midplane ($\theta = 0$, $R = R_0 + a$):

$$
\sin^2\alpha_0 < \frac{B_{\text{out}}}{B_{\text{in}}} \approx \frac{R_{\text{in}}}{R_{\text{out}}} = \frac{R_0 - a}{R_0 + a} \approx 1 - \frac{2a}{R_0} = 1 - 2\epsilon
$$

where $\epsilon = a/R_0$ is the inverse aspect ratio.

For small $\epsilon$:

$$
\alpha_c \approx \sqrt{2\epsilon}
$$

The fraction of trapped particles is:

$$
f_{\text{trap}} \approx \sqrt{2\epsilon} \approx \sqrt{\frac{a}{R_0}}
$$

For typical tokamaks with $\epsilon \sim 0.3$:

$$
f_{\text{trap}} \approx 0.77 \approx 77\%
$$

### 6.3 Banana Orbits

Trapped particles drift vertically (grad-B + curvature drift) and trace out "banana-shaped" orbits in the poloidal plane:

```
    Poloidal cross-section

         Top
          |
      ────┼────
     /    |    \
    │  /──┴──\  │  ← Banana orbit
    │ │       │ │     (trapped particle)
    │  \─────/  │
     \         /
      ─────────
          |
        Bottom

    Passing orbit: complete circle around
    Banana orbit: reflected, drift creates banana shape
```

The banana width (radial excursion) is:

$$
\Delta r_b \sim q\rho_i\sqrt{\epsilon}
$$

where:
- $q = rB_\phi/(R_0 B_\theta)$ is the safety factor
- $\rho_i = m_i v_{th,i}/(eB)$ is the ion Larmor radius
- $\epsilon = a/R_0$

**Importance**: Banana orbits reduce confinement by:
1. Increasing effective step size for radial transport
2. Creating neoclassical transport (collisional detrapping)

### 6.4 Passing Particles

Passing particles have large $v_\parallel$ and complete full poloidal circuits. They also drift vertically but remain on closed flux surfaces (in ideal axisymmetric geometry).

The orbit shift from the flux surface is:

$$
\Delta r_{\text{pass}} \sim q\rho_i
$$

Much smaller than the banana width.

### 6.5 Neoclassical Transport

The combination of trapped particle orbits and collisions leads to **neoclassical transport**, which exceeds classical (collision-based) transport:

$$
D_{\text{neo}} = \frac{q^2\rho_i^2}{\tau_e}\epsilon^{3/2}
$$

where $\tau_e$ is the electron collision time.

This is larger than classical diffusion $D_{\text{class}} \sim \rho_i^2/\tau_e$ by a factor $\sim q^2\epsilon^{3/2}$.

## 7. Van Allen Radiation Belts

### 7.1 Earth's Magnetosphere as a Magnetic Mirror

Earth's dipole magnetic field forms a natural magnetic bottle:

```
    Solar wind
    ────────→

         Magnetopause
           ____
          /    \
         /      \
    ────┤ Earth ├────  Equatorial plane
         \      /
          \____/

    Magnetic field lines:
    - Compressed on dayside (solar wind pressure)
    - Extended on nightside (magnetotail)
    - Particles trapped on closed field lines
```

Charged particles (electrons and protons from solar wind and cosmic rays) are trapped in the dipole field, forming the Van Allen radiation belts.

### 7.2 Two-Belt Structure

There are two main belts:

1. **Inner belt** ($1.2 - 2$ Earth radii):
   - Mostly high-energy protons (10-100 MeV)
   - Source: cosmic ray albedo neutron decay (CRAND)
   - Relatively stable

2. **Outer belt** ($4 - 6$ Earth radii):
   - Mostly electrons (0.1-10 MeV)
   - Source: solar wind injection during geomagnetic storms
   - Highly variable

### 7.3 Particle Motion in Dipole Field

Particles undergo three motions:
1. **Gyration** around field lines ($\omega_c \sim 10^4$ rad/s for protons)
2. **Bounce** between mirror points near the poles ($\omega_b \sim 1$ rad/s)
3. **Drift** around Earth ($\omega_d \sim 10^{-3}$ rad/s):
   - Protons drift westward
   - Electrons drift eastward

The drift creates a **ring current** during geomagnetic storms.

### 7.4 Loss Mechanisms

Particles are lost through:
1. **Atmospheric scattering**: Collisions with neutrals at low altitudes
2. **Charge exchange**: Protons capture electrons, become neutral, escape
3. **Wave-particle interactions**: VLF/ELF waves scatter particles into loss cone
4. **Magnetopause shadowing**: Field lines intersect magnetopause, particles escape

Lifetimes range from days (outer belt electrons) to years (inner belt protons).

## 8. Python Implementations

### 8.1 Magnetic Mirror Simulation

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
m_p = 1.67e-27  # kg
q_p = 1.6e-19   # C

def mirror_field(z, B0=1e-3, L=10.0, R=5.0):
    """
    Parabolic mirror field: B(z) = B0 * (1 + (z/L)^2 * (R-1))

    B0: midplane field (T)
    L: mirror length scale (m)
    R: mirror ratio B_max/B_min
    """
    # Parabolic profile is chosen because it is analytically simple AND gives a
    # smooth, monotonic increase from B_min at z=0 to B_max at |z|=L, closely
    # approximating the on-axis field of real Helmholtz-coil mirror devices.
    return B0 * (1 + (z / L)**2 * (R - 1))

def mirror_grad(z, B0=1e-3, L=10.0, R=5.0):
    """Gradient of mirror field dB/dz"""
    return B0 * 2 * (R - 1) * z / L**2

def equations_of_motion_mirror(t, state, q, m, B0, L, R):
    """
    Equations of motion in mirror field
    state = [x, y, z, vx, vy, vz]
    """
    x, y, z, vx, vy, vz = state

    # Magnetic field (z-component only, aligned with z-axis)
    B = mirror_field(z, B0, L, R)
    Bx, By, Bz = 0, 0, B

    # Lorentz force
    v = np.array([vx, vy, vz])
    B_vec = np.array([Bx, By, Bz])
    F_lorentz = q * np.cross(v, B_vec)

    # Mirror force (adiabatic approximation)
    # Recompute μ at every step from current vx,vy rather than using a stored
    # constant: this lets us monitor how well μ is actually conserved numerically,
    # while still using the guiding-center approximation for the parallel force.
    # mu = m * (vx^2 + vy^2) / (2 * B)
    v_perp_sq = vx**2 + vy**2
    mu = m * v_perp_sq / (2 * B)
    # F_mirror = -μ ∂B/∂z: this is the gradient force derived from the adiabatic
    # invariant. It is negative (opposing increasing B) because a trapped particle
    # is reflected before all its parallel kinetic energy converts to perpendicular.
    F_mirror = -mu * mirror_grad(z, B0, L, R)

    # Total force (Lorentz + mirror force in z-direction)
    Fx, Fy, Fz_lorentz = F_lorentz / m
    Fz = Fz_lorentz + F_mirror / m

    return np.array([vx, vy, vz, Fx, Fy, Fz])

def rk4_step(f, t, y, dt, *args):
    """4th-order Runge-Kutta"""
    k1 = f(t, y, *args)
    k2 = f(t + dt/2, y + dt*k1/2, *args)
    k3 = f(t + dt/2, y + dt*k2/2, *args)
    k4 = f(t + dt, y + dt*k3, *args)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

def simulate_mirror(v_perp, v_para, B0=1e-3, L=10.0, R=5.0,
                   duration=1.0, dt=1e-4):
    """
    Simulate particle in magnetic mirror

    v_perp: initial perpendicular velocity (m/s)
    v_para: initial parallel velocity (m/s)
    """
    # Initial conditions
    x0, y0, z0 = 0.0, 0.0, 0.0
    vx0, vy0, vz0 = v_perp, 0.0, v_para
    state = np.array([x0, y0, z0, vx0, vy0, vz0])

    # Time array
    num_steps = int(duration / dt)
    times = np.linspace(0, duration, num_steps)

    # Storage
    trajectory = np.zeros((num_steps, 6))
    trajectory[0] = state

    # Integration
    for i in range(1, num_steps):
        state = rk4_step(equations_of_motion_mirror, times[i-1], state,
                        dt, q_p, m_p, B0, L, R)
        trajectory[i] = state

    return times, trajectory

# Calculate loss cone angle
B0 = 1e-3  # Tesla
L = 10.0   # meters
R = 5.0    # mirror ratio

# Loss cone formula: sin^2(alpha_c) = B_min/B_max = 1/R.
# This comes from μ conservation: at the mirror point v_perp = v_total, so
# (v_perp,0)^2 / v^2 = B_min/B_max. Particles with smaller pitch angle have
# insufficient v_perp to be reflected and escape — they are "in the loss cone".
alpha_c = np.arcsin(1/np.sqrt(R)) * 180/np.pi  # degrees

print(f"Mirror ratio R = {R}")
print(f"Loss cone angle: {alpha_c:.2f}°")
print(f"Trapping condition: α > {alpha_c:.2f}°\n")

# Simulate trapped and lost particles
v_total = 1e5  # m/s

# Trapped particle (α = 60° > α_c)
alpha_trapped = 60 * np.pi/180
v_perp_trapped = v_total * np.sin(alpha_trapped)
v_para_trapped = v_total * np.cos(alpha_trapped)

print(f"Trapped particle: α = 60°")
print(f"  v_perp = {v_perp_trapped:.2e} m/s")
print(f"  v_para = {v_para_trapped:.2e} m/s")

t_trap, traj_trap = simulate_mirror(v_perp_trapped, v_para_trapped,
                                     B0, L, R, duration=2.0, dt=1e-4)

# Lost particle (α = 20° < α_c)
alpha_lost = 20 * np.pi/180
v_perp_lost = v_total * np.sin(alpha_lost)
v_para_lost = v_total * np.cos(alpha_lost)

print(f"\nLost particle: α = 20°")
print(f"  v_perp = {v_perp_lost:.2e} m/s")
print(f"  v_para = {v_para_lost:.2e} m/s")

t_lost, traj_lost = simulate_mirror(v_perp_lost, v_para_lost,
                                     B0, L, R, duration=1.0, dt=1e-4)

# Plotting
fig = plt.figure(figsize=(16, 10))

# 3D trajectories
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.plot(traj_trap[:, 0], traj_trap[:, 1], traj_trap[:, 2],
        'b-', linewidth=0.5, label='Trapped')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_zlabel('z (m)')
ax1.set_title(f'Trapped Particle (α = 60° > α_c = {alpha_c:.1f}°)')
ax1.legend()
ax1.grid(True)

ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax2.plot(traj_lost[:, 0], traj_lost[:, 1], traj_lost[:, 2],
        'r-', linewidth=0.5, label='Lost')
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_zlabel('z (m)')
ax2.set_title(f'Lost Particle (α = 20° < α_c = {alpha_c:.1f}°)')
ax2.legend()
ax2.grid(True)

# Z vs time (bounce motion)
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(t_trap, traj_trap[:, 2], 'b-', linewidth=1, label='Trapped')
ax3.axhline(y=L, color='k', linestyle='--', label=f'Mirror at ±{L} m')
ax3.axhline(y=-L, color='k', linestyle='--')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('z (m)')
ax3.set_title('Bounce Motion')
ax3.legend()
ax3.grid(True)

# Magnetic field along z
z_array = np.linspace(-L*1.5, L*1.5, 1000)
B_array = mirror_field(z_array, B0, L, R)

ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(z_array, B_array * 1e3, 'k-', linewidth=2)
ax4.set_xlabel('z (m)')
ax4.set_ylabel('B (mT)')
ax4.set_title('Magnetic Field Profile')
ax4.grid(True)

# Velocity components vs time
ax5 = fig.add_subplot(2, 3, 5)
v_perp_trap = np.sqrt(traj_trap[:, 3]**2 + traj_trap[:, 4]**2)
v_para_trap = traj_trap[:, 5]
ax5.plot(t_trap, v_perp_trap/1e3, 'b-', linewidth=1, label='v_perp')
ax5.plot(t_trap, v_para_trap/1e3, 'r-', linewidth=1, label='v_para')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Velocity (km/s)')
ax5.set_title('Velocity Components (Trapped)')
ax5.legend()
ax5.grid(True)

# Adiabatic invariant μ — plot normalised to μ(0) so deviations from 1 directly
# show the fractional numerical error. A well-resolved simulation keeps μ/μ(0)
# within ~1% throughout the bounce; larger drift reveals dt is too large or that
# the field gradient is too steep for the adiabatic approximation to hold.
B_traj_trap = mirror_field(traj_trap[:, 2], B0, L, R)
mu_trap = m_p * v_perp_trap**2 / (2 * B_traj_trap)

ax6 = fig.add_subplot(2, 3, 6)
ax6.plot(t_trap, mu_trap / mu_trap[0], 'b-', linewidth=1)
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('μ(t) / μ(0)')
ax6.set_title('Conservation of Magnetic Moment μ')
ax6.grid(True)
ax6.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='μ = const')
ax6.legend()

plt.tight_layout()
plt.savefig('magnetic_mirror.png', dpi=150)
print("\nSaved: magnetic_mirror.png")

# Calculate bounce period by detecting sign changes in v_z (the parallel velocity).
# A sign change means v_z passed through zero — the particle reversed direction at a
# mirror point. The time between consecutive reversals is half a bounce period.
z_positions = traj_trap[:, 2]
# Find turning points (where v_z changes sign)
v_z = traj_trap[:, 5]
sign_changes = np.diff(np.sign(v_z))
bounce_indices = np.where(sign_changes != 0)[0]

if len(bounce_indices) >= 2:
    tau_bounce = t_trap[bounce_indices[1]] - t_trap[bounce_indices[0]]
    omega_bounce = 2 * np.pi / tau_bounce
    print(f"\nBounce period: {tau_bounce:.4f} s")
    print(f"Bounce frequency: {omega_bounce:.2f} rad/s")

    # Compare with gyrofrequency
    omega_c = q_p * B0 / m_p
    print(f"Gyrofrequency: {omega_c:.2e} rad/s")
    print(f"Ratio ω_c/ω_b: {omega_c/omega_bounce:.2e}")
```

### 8.2 Loss Cone Visualization

```python
# Loss cone in velocity space
fig = plt.figure(figsize=(12, 5))

# 3D velocity space
ax1 = fig.add_subplot(121, projection='3d')

# Generate loss cone
theta = np.linspace(0, 2*np.pi, 50)
alpha_cone = np.linspace(0, alpha_c * np.pi/180, 20)
theta_grid, alpha_grid = np.meshgrid(theta, alpha_cone)

v = v_total
vx_cone = v * np.sin(alpha_grid) * np.cos(theta_grid)
vy_cone = v * np.sin(alpha_grid) * np.sin(theta_grid)
vz_cone = v * np.cos(alpha_grid)

ax1.plot_surface(vx_cone/1e3, vy_cone/1e3, vz_cone/1e3,
                alpha=0.3, color='red', label='Loss cone')

# Generate trapped region (example particles)
np.random.seed(42)
n_particles = 500
alpha_samples = np.arccos(np.random.uniform(-1, 1, n_particles))
theta_samples = np.random.uniform(0, 2*np.pi, n_particles)

# Separate trapped and lost
trapped_mask = alpha_samples > alpha_c * np.pi/180
lost_mask = ~trapped_mask

vx_trapped = v * np.sin(alpha_samples[trapped_mask]) * np.cos(theta_samples[trapped_mask])
vy_trapped = v * np.sin(alpha_samples[trapped_mask]) * np.sin(theta_samples[trapped_mask])
vz_trapped = v * np.cos(alpha_samples[trapped_mask])

vx_lost = v * np.sin(alpha_samples[lost_mask]) * np.cos(theta_samples[lost_mask])
vy_lost = v * np.sin(alpha_samples[lost_mask]) * np.sin(theta_samples[lost_mask])
vz_lost = v * np.cos(alpha_samples[lost_mask])

ax1.scatter(vx_trapped/1e3, vy_trapped/1e3, vz_trapped/1e3,
           c='blue', s=1, alpha=0.5, label='Trapped')
ax1.scatter(vx_lost/1e3, vy_lost/1e3, vz_lost/1e3,
           c='red', s=2, alpha=0.8, label='Lost')

ax1.set_xlabel('vx (km/s)')
ax1.set_ylabel('vy (km/s)')
ax1.set_zlabel('vz (km/s)')
ax1.set_title(f'Loss Cone in Velocity Space\nα_c = {alpha_c:.1f}°, R = {R}')
ax1.legend()

# 2D pitch angle distribution
ax2 = fig.add_subplot(122)
alpha_deg = np.linspace(0, 180, 1000)
alpha_rad = alpha_deg * np.pi / 180

# Distribution function (isotropic) proportional to sin(α)
f_alpha = np.sin(alpha_rad)

ax2.fill_between(alpha_deg, 0, f_alpha, where=(alpha_deg < alpha_c),
                color='red', alpha=0.5, label='Loss cone (escaping)')
ax2.fill_between(alpha_deg, 0, f_alpha, where=(alpha_deg >= alpha_c),
                color='blue', alpha=0.5, label='Trapped')
ax2.plot(alpha_deg, f_alpha, 'k-', linewidth=2)
ax2.axvline(x=alpha_c, color='red', linestyle='--', linewidth=2,
           label=f'α_c = {alpha_c:.1f}°')

ax2.set_xlabel('Pitch Angle α (degrees)', fontsize=12)
ax2.set_ylabel('f(α) ∝ sin(α)', fontsize=12)
ax2.set_title('Pitch Angle Distribution', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('loss_cone.png', dpi=150)
print("Saved: loss_cone.png")

# Calculate fraction in loss cone
solid_angle_loss = 2 * np.pi * (1 - np.cos(alpha_c * np.pi/180))
solid_angle_total = 4 * np.pi
frac_loss = solid_angle_loss / solid_angle_total

print(f"\nLoss cone solid angle: {solid_angle_loss:.4f} sr")
print(f"Fraction in loss cone: {frac_loss:.4f} = {frac_loss*100:.2f}%")
print(f"Fraction trapped: {1-frac_loss:.4f} = {(1-frac_loss)*100:.2f}%")
```

### 8.3 Banana Orbit Simulation (Simplified Tokamak)

```python
def tokamak_field(R, Z, R0=2.0, a=0.5, B0=3.0, q=2.0):
    """
    Simplified tokamak field

    R, Z: cylindrical coordinates (m)
    R0: major radius (m)
    a: minor radius (m)
    B0: field at magnetic axis (T)
    q: safety factor
    """
    # Toroidal field decreases as 1/R because the field is produced by straight
    # central solenoid coils; by Ampere's law B_phi * 2πR = const → B_phi ∝ 1/R.
    # This 1/R variation creates both the field gradient and curvature that cause
    # the banana orbit drift.
    B_phi = B0 * R0 / R

    # Poloidal field model: linear inside the plasma (uniform current density),
    # dipole-like outside. This gives a safety factor q = rB_phi/(R0 B_theta)
    # that is constant inside — a simplification avoiding the full Grad-Shafranov.
    r = np.sqrt((R - R0)**2 + Z**2)
    if r < a:
        B_theta = B0 * r / (q * R0)
    else:
        B_theta = B0 * a**2 / (q * R0 * r)

    # Convert poloidal field to cylindrical (R,Z) components using the local
    # poloidal angle; the sign convention ensures B_theta circles the magnetic axis.
    B_R = -B_theta * Z / r if r > 1e-6 else 0
    B_Z = B_theta * (R - R0) / r if r > 1e-6 else 0

    return B_R, B_Z, B_phi

def tokamak_magnitude(R, Z, R0=2.0, a=0.5, B0=3.0, q=2.0):
    """Total field magnitude"""
    B_R, B_Z, B_phi = tokamak_field(R, Z, R0, a, B0, q)
    return np.sqrt(B_R**2 + B_Z**2 + B_phi**2)

# Plot tokamak field magnitude
R_grid = np.linspace(1.0, 3.0, 100)
Z_grid = np.linspace(-1.0, 1.0, 100)
R_mesh, Z_mesh = np.meshgrid(R_grid, Z_grid)

B_mag = np.zeros_like(R_mesh)
for i in range(len(Z_grid)):
    for j in range(len(R_grid)):
        B_mag[i, j] = tokamak_magnitude(R_mesh[i, j], Z_mesh[i, j])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Field contours
contour = ax1.contourf(R_mesh, Z_mesh, B_mag, levels=20, cmap='viridis')
ax1.contour(R_mesh, Z_mesh, B_mag, levels=10, colors='white',
           linewidths=0.5, alpha=0.5)
plt.colorbar(contour, ax=ax1, label='B (T)')

# Mark regions
R0, a = 2.0, 0.5
circle = plt.Circle((R0, 0), a, fill=False, color='red',
                    linewidth=2, label='Last closed flux surface')
ax1.add_patch(circle)
ax1.plot([R0], [0], 'r*', markersize=15, label='Magnetic axis')

ax1.set_xlabel('R (m)', fontsize=12)
ax1.set_ylabel('Z (m)', fontsize=12)
ax1.set_title('Tokamak Magnetic Field |B|', fontsize=14, fontweight='bold')
ax1.set_aspect('equal')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Field along midplane
R_midplane = np.linspace(1.0, 3.0, 200)
B_midplane = [tokamak_magnitude(R, 0) for R in R_midplane]

ax2.plot(R_midplane, B_midplane, 'b-', linewidth=2)
ax2.axvline(x=R0-a, color='r', linestyle='--', label='Inboard edge')
ax2.axvline(x=R0+a, color='r', linestyle='--', label='Outboard edge')
ax2.axvline(x=R0, color='g', linestyle='--', label='Magnetic axis')

ax2.set_xlabel('R (m)', fontsize=12)
ax2.set_ylabel('B (T)', fontsize=12)
ax2.set_title('Field Along Midplane (Z=0)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('tokamak_field.png', dpi=150)
print("Saved: tokamak_field.png")

# Calculate trapping fraction using the small-ε approximation f_trap ≈ sqrt(2ε).
# This estimate comes from the loss-cone condition for the tokamak mirror: particles
# with sin^2(alpha) < 1 - B_out/B_in ≈ 2ε are trapped. For ε ~ 0.3 (typical
# aspect ratio), about 77% of particles are on banana orbits — a dominant effect
# in neoclassical transport.
epsilon = a / R0
f_trapped = np.sqrt(2 * epsilon)
print(f"\nTokamak parameters:")
print(f"  Major radius R0 = {R0} m")
print(f"  Minor radius a = {a} m")
print(f"  Inverse aspect ratio ε = {epsilon:.3f}")
print(f"  Trapped fraction ≈ sqrt(2ε) = {f_trapped:.3f} = {f_trapped*100:.1f}%")

# Banana width estimate
T_keV = 10  # keV
T_J = T_keV * 1e3 * q_p
v_th_i = np.sqrt(2 * T_J / m_p)
rho_i = m_p * v_th_i / (q_p * B0)
q_safety = 2.0
banana_width = q_safety * rho_i * np.sqrt(epsilon)

print(f"\nBanana orbit:")
print(f"  Ion temperature T_i = {T_keV} keV")
print(f"  Thermal velocity v_th = {v_th_i:.2e} m/s")
print(f"  Ion Larmor radius ρ_i = {rho_i*1e3:.2f} mm")
print(f"  Banana width Δr_b ≈ {banana_width*1e3:.2f} mm")
```

## Summary

In this lesson, we explored magnetic mirrors and adiabatic invariants:

1. **Magnetic mirror force**: $F_\parallel = -\mu\nabla_\parallel B$ reflects particles from high-field regions, enabling confinement in magnetic bottles.

2. **First adiabatic invariant μ**: The magnetic moment $\mu = mv_\perp^2/(2B)$ is conserved when fields vary slowly compared to the gyroperiod. This corresponds to conservation of flux through the gyro-orbit.

3. **Loss cone**: Particles with pitch angle $\alpha < \alpha_c = \arcsin(1/\sqrt{R})$ escape through mirror throats. The fraction in the loss cone is $\sim 1/(4R)$ for large $R$.

4. **Bounce motion**: Trapped particles oscillate between mirror points with frequency $\omega_b \sim v_\parallel/L$, much slower than gyrofrequency.

5. **Hierarchy of invariants**:
   - $\mu$: gyration (fastest)
   - $J = \oint mv_\parallel ds$: bounce (intermediate)
   - $\Phi = \oint \mathbf{A}\cdot d\mathbf{l}$: drift (slowest)

6. **Tokamak orbits**: The $1/R$ variation creates trapped (banana) and passing particles. Trapped fraction $\sim\sqrt{\epsilon}$ where $\epsilon = a/R_0$.

7. **Van Allen belts**: Earth's dipole field forms a natural magnetic bottle, trapping charged particles from space.

Understanding adiabatic invariants is crucial for:
- Designing magnetic confinement devices
- Predicting particle transport
- Analyzing space plasma dynamics

## Practice Problems

### Problem 1: Mirror Confinement Time

A simple magnetic mirror has $B_{\text{min}} = 0.2$ T, $B_{\text{max}} = 1.0$ T, and length $L = 5$ m. The plasma density is $n = 10^{18}$ m$^{-3}$ and temperature $T = 100$ eV. The collision frequency is $\nu_c = 10^4$ s$^{-1}$.

(a) Calculate the mirror ratio $R$ and loss cone angle $\alpha_c$.

(b) Estimate the fraction of particles in the loss cone (assume isotropic distribution).

(c) Calculate the confinement time $\tau_{\text{conf}} \sim (R/\nu_c)(1/f_{\text{loss}})$.

(d) Compare with the collision time $\tau_c = 1/\nu_c$. Is this a well-confined plasma?

---

### Problem 2: Conservation of μ in a Slowly Varying Field

A proton with initial energy $W = 1$ keV and pitch angle $\alpha_0 = 45°$ moves in a magnetic field that increases from $B_0 = 0.1$ T to $B_1 = 0.5$ T over a distance $L = 10$ m.

(a) Calculate the initial magnetic moment $\mu$.

(b) Assuming $\mu$ is conserved, find the final perpendicular and parallel velocities.

(c) What is the final pitch angle $\alpha_1$?

(d) Verify that total energy is conserved.

(e) Check the adiabatic condition: is $r_L |\nabla B|/B \ll 1$?

---

### Problem 3: Bounce Frequency in a Parabolic Mirror

In a parabolic mirror with $B(z) = B_0(1 + z^2/L^2)$ where $B_0 = 0.5$ T and $L = 10$ m, a deuteron has energy $W = 10$ keV and pitch angle $\alpha_0 = 60°$ at the midplane ($z = 0$).

(a) Find the mirror point $z_{\text{mirror}}$ where $v_\parallel = 0$.

(b) Estimate the bounce period $\tau_b$ using $\tau_b \approx 4z_{\text{mirror}}/\langle v_\parallel\rangle$ where $\langle v_\parallel\rangle$ is the average parallel velocity.

(c) Calculate the bounce frequency $\omega_b = 2\pi/\tau_b$.

(d) Compare with the gyrofrequency $\omega_c$ at the midplane. Verify $\omega_c \gg \omega_b$.

**Hint**: Use energy conservation: $\frac{1}{2}mv^2\sin^2\alpha_0 = \frac{1}{2}mv^2\frac{B(z_m)}{B_0}$.

---

### Problem 4: Tokamak Banana Width

In a tokamak with $R_0 = 3$ m, $a = 1$ m, $B_0 = 5$ T, and safety factor $q = 2$, calculate the banana width for deuterium ions with temperature $T_i = 20$ keV.

(a) Compute the inverse aspect ratio $\epsilon = a/R_0$.

(b) Calculate the ion Larmor radius $\rho_i = m_i v_{th,i}/(eB_0)$ where $v_{th,i} = \sqrt{2k_BT_i/m_i}$.

(c) Estimate the banana width $\Delta r_b \approx q\rho_i\sqrt{\epsilon}$.

(d) What fraction of the minor radius is the banana width? Is this significant for confinement?

(e) Calculate the trapped particle fraction $f_{\text{trap}} \approx \sqrt{2\epsilon}$.

---

### Problem 5: Second Adiabatic Invariant in a Dipole Field

An electron is trapped on a dipole field line at $L = 4$ Earth radii ($L$ is the equatorial crossing distance in units of $R_E = 6.37\times 10^6$ m). The magnetic field on this line varies as:

$$
B(s) = B_{\text{eq}}\sqrt{1 + 3\sin^2\lambda}/\cos^6\lambda
$$

where $\lambda$ is magnetic latitude and $B_{\text{eq}} = 10^{-6}$ T is the equatorial field.

(a) The electron has energy $W = 100$ keV and equatorial pitch angle $\alpha_{\text{eq}} = 45°$. Find the mirror latitude $\lambda_m$ where the particle reflects.

(b) Estimate the arc length $s_m$ from equator to mirror point (use $s \approx LR_E\lambda$ for small $\lambda$).

(c) Calculate the second invariant $J = \oint mv_\parallel ds \approx 4m\langle v_\parallel\rangle s_m$ (approximate).

(d) If the field slowly decreases (e.g., during a geomagnetic storm), $J$ remains constant. Explain qualitatively how the mirror latitude changes.

**Hint**: For $\lambda_m$, use $\sin^2\alpha_{\text{eq}} = B_{\text{eq}}/B(\lambda_m)$.

---

## Navigation

- **Previous**: [Single Particle Motion II](./05_Single_Particle_Motion_II.md)
- **Next**: [Vlasov Equation](./07_Vlasov_Equation.md)
