# 4. Single Particle Motion I

## Learning Objectives

- Derive the equations of motion for a charged particle in electromagnetic fields
- Analyze cyclotron motion in uniform magnetic fields and calculate Larmor radius and gyrofrequency
- Understand the E×B drift and its physical origin from asymmetric gyration
- Introduce the guiding center approximation for separating fast gyration from slow drift
- Implement the Boris algorithm for accurate particle orbit integration
- Visualize particle trajectories in various field configurations using Python

## 1. Equation of Motion

### 1.1 The Lorentz Force

A charged particle with charge $q$ and mass $m$ moving with velocity $\mathbf{v}$ in electric field $\mathbf{E}$ and magnetic field $\mathbf{B}$ experiences the **Lorentz force**:

$$\mathbf{F} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

Newton's second law gives the **equation of motion**:

$$m\frac{d\mathbf{v}}{dt} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

$$\frac{d\mathbf{x}}{dt} = \mathbf{v}$$

This is a system of 6 first-order ODEs for $(\mathbf{x}, \mathbf{v})$.

### 1.2 Key Properties

**1. Magnetic force does no work:**

$$\frac{dE_{kin}}{dt} = \mathbf{F} \cdot \mathbf{v} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B}) \cdot \mathbf{v} = q\mathbf{E} \cdot \mathbf{v}$$

Since $(\mathbf{v} \times \mathbf{B}) \cdot \mathbf{v} = 0$, the magnetic force is always perpendicular to velocity and does no work.

**Energy conservation:** In static fields with $\mathbf{E} = -\nabla \phi$:

$$E_{total} = \frac{1}{2}m v^2 + q\phi = \text{const}$$

**2. Magnetic force causes deflection, not acceleration:**

The magnetic force changes the **direction** of $\mathbf{v}$ but not its **magnitude**.

**3. Parallel and perpendicular decomposition:**

Define $\hat{\mathbf{b}} = \mathbf{B}/B$ as the unit vector along $\mathbf{B}$. Decompose velocity:

$$\mathbf{v} = v_\parallel \hat{\mathbf{b}} + \mathbf{v}_\perp$$

where $v_\parallel = \mathbf{v} \cdot \hat{\mathbf{b}}$ and $\mathbf{v}_\perp = \mathbf{v} - v_\parallel \hat{\mathbf{b}}$.

The equations of motion become:

$$\frac{dv_\parallel}{dt} = \frac{q}{m} E_\parallel$$

$$\frac{d\mathbf{v}_\perp}{dt} = \frac{q}{m}(\mathbf{E}_\perp + \mathbf{v}_\perp \times \mathbf{B})$$

Parallel motion is **decoupled** from perpendicular motion (assuming uniform $\mathbf{B}$).

## 2. Uniform Magnetic Field: Cyclotron Motion

### 2.1 Solution for B-field Only

Consider a particle in a uniform magnetic field $\mathbf{B} = B\hat{\mathbf{z}}$ with no electric field.

**Parallel motion:**

$$\frac{dv_z}{dt} = 0 \quad \Rightarrow \quad v_z = v_{z,0} = \text{const}$$

The particle moves freely along $\mathbf{B}$.

**Perpendicular motion:**

In the $(x, y)$ plane:

$$m\frac{dv_x}{dt} = qv_y B$$

$$m\frac{dv_y}{dt} = -qv_x B$$

Define the **gyrofrequency** (cyclotron frequency):

$$\omega_c = \frac{|q|B}{m}$$

**Sign convention:** We use $|q|$ so that $\omega_c > 0$. The actual signed frequency is:

$$\Omega_c = \frac{qB}{m} = \begin{cases}
\omega_c & \text{for } q > 0 \\
-\omega_c & \text{for } q < 0
\end{cases}$$

The equations become:

$$\frac{dv_x}{dt} = \pm \omega_c v_y, \quad \frac{dv_y}{dt} = \mp \omega_c v_x$$

where the upper sign is for positive charge, lower for negative.

### 2.2 Circular Motion

Combining:

$$\frac{d^2 v_x}{dt^2} = \pm \omega_c \frac{dv_y}{dt} = -\omega_c^2 v_x$$

This is simple harmonic motion with angular frequency $\omega_c$:

$$v_x(t) = v_\perp \cos(\omega_c t + \phi_0)$$

$$v_y(t) = \mp v_\perp \sin(\omega_c t + \phi_0)$$

where $v_\perp = \sqrt{v_x^2 + v_y^2}$ is the constant perpendicular speed.

Integrating for position (assuming particle starts at origin):

$$x(t) = \frac{v_\perp}{\omega_c} \sin(\omega_c t + \phi_0) = r_L \sin(\omega_c t + \phi_0)$$

$$y(t) = \pm \frac{v_\perp}{\omega_c} \cos(\omega_c t + \phi_0) = \pm r_L \cos(\omega_c t + \phi_0)$$

This describes **circular motion** with radius:

$$r_L = \frac{v_\perp}{\omega_c} = \frac{m v_\perp}{|q|B}$$

This is the **Larmor radius** (or gyroradius).

### 2.3 Sense of Rotation

```
Gyration Direction:

B field into page (⊗):

Positive charge (e.g., proton):          Negative charge (e.g., electron):
        ↑                                         ↑
        │                                         │
    ←───●───→  Counterclockwise               ───●───→  Clockwise
        │      (right-hand rule)                  │      (opposite)
        ↓                                         ↓

For B = B ẑ (out of page):
- Ions (q > 0): rotate counterclockwise (viewed from +z)
- Electrons (q < 0): rotate clockwise
```

**Right-hand rule:** Point thumb along $\mathbf{B}$; fingers curl in the direction of **positive** charge gyration.

### 2.4 Helical Trajectory

Combining parallel and perpendicular motion:

$$\mathbf{r}(t) = \begin{pmatrix} r_L \sin(\omega_c t) \\ \pm r_L \cos(\omega_c t) \\ v_z t \end{pmatrix}$$

This is a **helix** with:
- Radius $r_L$
- Pitch $p = 2\pi v_z / \omega_c$
- Chirality determined by sign of $q$

```
Helical Trajectory:

        z (along B)
        ↑
        │     ╱╲
        │    ╱  ╲
        │   ╱    ╲
        │  ╱      ╲
        │ ╱        ╲
        │╱__________╲___→ y
       ╱│            ╲
      ╱ │             ╲
     ╱  │              ╲
    ╱   │               ╲
   ╱    │                ╲
  x     │
       (gyration in xy-plane)

Particle spirals around field line with:
- Gyroradius r_L
- Parallel velocity v_z
```

### 2.5 Characteristic Scales

For electrons:

$$\omega_{ce} = \frac{eB}{m_e} \approx 1.76 \times 10^{11} B[\text{T}] \quad [\text{rad/s}]$$

$$r_{Le} = \frac{v_\perp}{\omega_{ce}} \approx 5.69 \times 10^{-6} \frac{v_\perp[\text{m/s}]}{B[\text{T}]} \quad [\text{m}]$$

For thermal electrons at $T_e = 100$ eV:

$$v_{te} \approx 4.2 \times 10^6 \text{ m/s}$$

In $B = 1$ T:

$$\omega_{ce} \approx 1.76 \times 10^{11} \text{ rad/s}, \quad r_{Le} \approx 24 \text{ μm}$$

For protons ($m_p = 1836 m_e$):

$$\omega_{ci} = \frac{\omega_{ce}}{1836} \approx 9.6 \times 10^7 \text{ rad/s}$$

$$r_{Li} = \sqrt{1836} \, r_{Le} \approx 43 r_{Le} \approx 1 \text{ mm}$$

**Ordering:**
$$r_{Le} \ll r_{Li}, \quad \omega_{ci} \ll \omega_{ce}$$

Electrons gyrate much faster and with smaller radius than ions.

## 3. Uniform E and B Fields: E×B Drift

### 3.1 Perpendicular Electric Field

Now add a uniform electric field $\mathbf{E}$ perpendicular to $\mathbf{B}$. The equation of motion is:

$$m\frac{d\mathbf{v}}{dt} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

**Key observation:** The $\mathbf{E}$ term breaks the symmetry of the circular orbit.

### 3.2 Drift Velocity

We can show (derivation below) that the solution consists of:
1. **Gyration** at frequency $\omega_c$ with radius $r_L$
2. **Drift** perpendicular to both $\mathbf{E}$ and $\mathbf{B}$

The **E×B drift velocity** is:

$$\mathbf{v}_E = \frac{\mathbf{E} \times \mathbf{B}}{B^2}$$

**Key properties:**
- Independent of charge $q$ and mass $m$!
- Perpendicular to both $\mathbf{E}$ and $\mathbf{B}$
- Magnitude: $v_E = E/B$ (if $\mathbf{E} \perp \mathbf{B}$)

### 3.3 Derivation

Write $\mathbf{v} = \mathbf{v}_g + \mathbf{v}'$, where $\mathbf{v}_g$ is a constant drift velocity (to be determined) and $\mathbf{v}'$ is the gyration velocity in the drifting frame.

Substitute into the equation of motion:

$$m\frac{d\mathbf{v}'}{dt} = q[\mathbf{E} + (\mathbf{v}_g + \mathbf{v}') \times \mathbf{B}]$$

$$= q[\mathbf{E} + \mathbf{v}_g \times \mathbf{B}] + q\mathbf{v}' \times \mathbf{B}$$

**Choose $\mathbf{v}_g$ such that the constant term vanishes:**

$$\mathbf{E} + \mathbf{v}_g \times \mathbf{B} = 0$$

Cross both sides with $\mathbf{B}$:

$$\mathbf{E} \times \mathbf{B} + (\mathbf{v}_g \times \mathbf{B}) \times \mathbf{B} = 0$$

Use the vector identity $(\mathbf{A} \times \mathbf{B}) \times \mathbf{C} = \mathbf{B}(\mathbf{A} \cdot \mathbf{C}) - \mathbf{A}(\mathbf{B} \cdot \mathbf{C})$:

$$\mathbf{E} \times \mathbf{B} + \mathbf{B}(\mathbf{v}_g \cdot \mathbf{B}) - \mathbf{v}_g B^2 = 0$$

If $\mathbf{v}_g \perp \mathbf{B}$, then $\mathbf{v}_g \cdot \mathbf{B} = 0$:

$$\mathbf{v}_g = \frac{\mathbf{E} \times \mathbf{B}}{B^2}$$

This is the E×B drift.

In the drifting frame, the equation becomes:

$$m\frac{d\mathbf{v}'}{dt} = q\mathbf{v}' \times \mathbf{B}$$

which is just gyration (solved in Section 2).

### 3.4 Physical Picture

Why does E×B drift occur?

```
Physical Mechanism of E×B Drift:

Consider E pointing in +y direction, B in +z (into page).

Electron orbit:

    Point A: v pointing +x        Point C: v pointing -x
             E accelerates up               E accelerates down
             ↑                              ↓
    ╭───────●───────╮            ╭─────────●─────────╮
   ╱        │        ╲          ╱          │          ╲
  │    D    │    B    │        │      D    │    B      │
  │         ●         │        │           ●           │
  │    ←─── ● ───→   │        │      ←─── ● ───→     │
  │         E         │        │           E           │
   ╲                 ╱          ╲                     ╱
    ╰───────────────╯            ╰───────────────────╯

At A: E accelerates particle upward → larger v_y → larger r_L
At C: E decelerates particle → smaller v_y → smaller r_L

Asymmetry: Top of orbit has larger radius than bottom
Result: Net drift to the right (−x direction)

Direction: E×B/B² = (E_y ŷ) × (B_z ẑ) / B² = -E_y/B x̂ (−x direction)

Both electrons and ions drift in the same direction!
(ions gyrate opposite direction but E affects them oppositely)
```

### 3.5 Example: Magnetron

In a magnetron, electrons drift in crossed $\mathbf{E}$ and $\mathbf{B}$ fields:

- Radial electric field: $E_r$ (outward)
- Axial magnetic field: $B_z$
- E×B drift: azimuthal, $v_\theta = E_r/B_z$

Electrons orbit around the anode in a "spoke" pattern, interacting with RF cavities to generate microwaves.

## 4. Parallel Electric Field

### 4.1 Acceleration Along B

If $\mathbf{E}$ has a component parallel to $\mathbf{B}$:

$$\frac{dv_\parallel}{dt} = \frac{q}{m} E_\parallel$$

The particle accelerates (or decelerates) along the field line.

**Energy gain:**

$$\frac{d}{dt}\left(\frac{1}{2}mv^2\right) = q\mathbf{E} \cdot \mathbf{v} = q E_\parallel v_\parallel$$

### 4.2 Combined Motion

With both $E_\parallel$ and $E_\perp$:

- **Parallel:** Accelerated motion along $\mathbf{B}$
- **Perpendicular:** Gyration + E×B drift
- Result: Helical path with changing pitch and drifting guiding center

```
Combined E∥ and E⊥:

        z (B direction)
        ↑
        │   ___
        │  /   \___
        │ /        \___
        │/  increasing v_z
       /│   (E∥ accelerates)
      / │
     /  │            ←─ E⊥ causes E×B drift
    /   │               (guiding center moves)
   /    │
  x     │

Particle spirals with:
- Increasing v_z (if qE∥ > 0)
- Drifting guiding center (E×B)
- Approximately constant r_L (v_⊥ constant if E⊥ constant)
```

## 5. Guiding Center Approximation

### 5.1 Motivation

In many plasmas, the gyroradius $r_L$ is much smaller than the scale length $L$ of field variations:

$$r_L \ll L$$

It's inefficient to resolve the fast gyration when we're interested in slower, larger-scale dynamics.

**Guiding center approximation:** Separate fast gyration from slow drift.

### 5.2 Guiding Center Position

Define the **guiding center position** $\mathbf{R}$ as the center of the gyro-orbit:

$$\mathbf{R} = \mathbf{r} - \frac{\mathbf{v}_\perp \times \mathbf{B}}{\omega_c B}$$

For a particle at position $\mathbf{r}$ with perpendicular velocity $\mathbf{v}_\perp$:

$$\mathbf{R} = \mathbf{r} - r_L \hat{\mathbf{e}}_\perp$$

where $\hat{\mathbf{e}}_\perp$ points from the guiding center to the particle position.

```
Guiding Center:

         B (out of page)
              ⊙

        Particle orbit
          ╭───╮
         ╱     ╲
        │   ●   │  ← Guiding center R
        │  ╱│╲  │
         ╲╱ │ ╲╱
          ╰─●─╯
            ↑
       Particle position r

r_L = |r - R|
```

### 5.3 Guiding Center Velocity

The velocity of the guiding center is the **drift velocity**, obtained by averaging over a gyro-period:

$$\mathbf{V}_g = \langle \mathbf{v} \rangle_\text{gyro} = \mathbf{v}_E + \mathbf{v}_\text{other drifts}$$

For uniform $\mathbf{E}$ and $\mathbf{B}$:

$$\mathbf{V}_g = \frac{\mathbf{E} \times \mathbf{B}}{B^2} + v_\parallel \hat{\mathbf{b}}$$

The gyration motion is "averaged out" in this description.

### 5.4 Adiabatic Invariants

When fields vary slowly in space and time (compared to gyration), certain quantities are **adiabatic invariants**—they are approximately conserved.

The most important is the **magnetic moment**:

$$\mu = \frac{m v_\perp^2}{2B} = \frac{E_\perp}{B}$$

where $E_\perp = \frac{1}{2}m v_\perp^2$ is the perpendicular kinetic energy.

**Conservation:** If $B$ changes slowly along the particle trajectory, $\mu$ remains nearly constant.

We'll explore this in the next lesson (Single Particle Motion II).

## 6. Computational Implementation: Boris Algorithm

### 6.1 Challenges of Particle Pushing

Integrating the Lorentz force equation is challenging because:

1. **Gyration timescale:** $\tau_c = 2\pi/\omega_c$ can be very short ($\sim 10^{-11}$ s for electrons in 1 T)
2. **Stiffness:** Explicit methods (Euler, RK4) require $\Delta t \ll \tau_c$ for stability
3. **Energy conservation:** Naive methods accumulate error, causing unphysical energy drift

### 6.2 The Boris Algorithm

The **Boris algorithm** (Boris, 1970) is the gold standard for particle pushing in plasma simulations. It:

- Is **second-order accurate**
- **Conserves energy** (for constant fields)
- Is **efficient** and **stable**

**Algorithm:**

Given $\mathbf{v}^n$ and $\mathbf{x}^n$ at time $t^n$, compute $\mathbf{v}^{n+1}$ and $\mathbf{x}^{n+1}$ at $t^{n+1} = t^n + \Delta t$:

1. **Half electric push:**
   $$\mathbf{v}^- = \mathbf{v}^n + \frac{q\mathbf{E}}{m} \frac{\Delta t}{2}$$

2. **Magnetic rotation:**
   Define $\mathbf{t} = \frac{q\mathbf{B}}{m} \frac{\Delta t}{2}$ and $s = \frac{2\mathbf{t}}{1 + t^2}$.

   $$\mathbf{v}' = \mathbf{v}^- + \mathbf{v}^- \times \mathbf{t}$$

   $$\mathbf{v}^+ = \mathbf{v}^- + \mathbf{v}' \times \mathbf{s}$$

3. **Half electric push:**
   $$\mathbf{v}^{n+1} = \mathbf{v}^+ + \frac{q\mathbf{E}}{m} \frac{\Delta t}{2}$$

4. **Position update:**
   $$\mathbf{x}^{n+1} = \mathbf{x}^n + \mathbf{v}^{n+1} \Delta t$$

**Why it works:** The rotation step exactly conserves $|\mathbf{v}|$ (magnetic force does no work), and the half-step electric acceleration minimizes energy error.

### 6.3 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
e = 1.602176634e-19      # Elementary charge [C]
m_e = 9.1093837015e-31   # Electron mass [kg]
m_p = 1.672621898e-27    # Proton mass [kg]

class ParticleTracer:
    """
    Trace charged particle trajectories using the Boris algorithm.
    """

    def __init__(self, q, m, x0, v0):
        """
        Parameters:
        -----------
        q : float
            Charge [C]
        m : float
            Mass [kg]
        x0 : array_like
            Initial position [m] (3D)
        v0 : array_like
            Initial velocity [m/s] (3D)
        """
        self.q = q
        self.m = m
        self.x = np.array(x0, dtype=float)
        self.v = np.array(v0, dtype=float)

        # History
        self.x_history = [self.x.copy()]
        self.v_history = [self.v.copy()]
        self.t_history = [0.0]

    def boris_step(self, E, B, dt):
        """
        Advance one timestep using Boris algorithm.

        Parameters:
        -----------
        E : array_like
            Electric field [V/m] (3D)
        B : array_like
            Magnetic field [T] (3D)
        dt : float
            Timestep [s]
        """
        q_over_m = self.q / self.m

        # Half electric push: apply E acceleration for dt/2 before the rotation.
        # Splitting E and B this way isolates the purely rotational magnetic step,
        # which lets us handle the stiff B-field term exactly without any
        # stability constraint on dt (unlike a naive Euler step for the full force).
        v_minus = self.v + 0.5 * q_over_m * E * dt

        # Construct the rotation vectors t and s for the magnetic half-step.
        # t = (q/m) * B * (dt/2) is the half-angle of rotation in the plane
        # perpendicular to B. The Cayley-Klein (tan half-angle) parameterization
        # is chosen because it maps exactly onto a unit-magnitude rotation:
        # |v_plus| == |v_minus| is guaranteed algebraically, not just
        # approximately, regardless of how large |t| is. Naive cross-product
        # rotation (v += v x omega * dt) would shrink or grow |v| by (1 ± |t|^2).
        t = 0.5 * q_over_m * B * dt
        t_mag_sq = np.dot(t, t)
        # s = 2t / (1 + |t|^2) is the second half of the Cayley-Klein map.
        # Together, v_prime = v + v x t and v_plus = v_minus + v_prime x s
        # implement an exact rotation by angle 2*arctan(|t|) around B.
        s = 2 * t / (1 + t_mag_sq)

        v_prime = v_minus + np.cross(v_minus, t)
        v_plus = v_minus + np.cross(v_prime, s)

        # Second half electric push: the symmetric half-step structure (E/2 →
        # rotate → E/2) makes the algorithm time-reversible (replacing dt → -dt
        # and reversing v recovers the initial state exactly). Time-reversibility
        # is what prevents the secular energy drift that plagues one-sided methods
        # like the simple Euler integrator over many gyroperiods.
        self.v = v_plus + 0.5 * q_over_m * E * dt

        # Position update
        self.x += self.v * dt

    def trace(self, E_func, B_func, t_max, dt):
        """
        Trace particle trajectory.

        Parameters:
        -----------
        E_func : callable
            Function E(x, t) returning electric field
        B_func : callable
            Function B(x, t) returning magnetic field
        t_max : float
            Maximum simulation time [s]
        dt : float
            Timestep [s]
        """
        t = 0.0
        while t < t_max:
            # Re-evaluate fields at the current position each step to support
            # spatially and temporally varying fields (non-uniform B, oscillating E).
            # For uniform static fields, this is redundant but keeps the interface
            # general without a significant performance cost.
            E = E_func(self.x, t)
            B = B_func(self.x, t)

            self.boris_step(E, B, dt)

            t += dt
            self.t_history.append(t)
            # Store copies rather than references so that subsequent in-place
            # updates to self.x and self.v do not overwrite the recorded history.
            self.x_history.append(self.x.copy())
            self.v_history.append(self.v.copy())

        # Convert to arrays only at the end to avoid O(N^2) memory reallocation
        # that would occur if we appended rows to a numpy array inside the loop.
        self.x_history = np.array(self.x_history)
        self.v_history = np.array(self.v_history)
        self.t_history = np.array(self.t_history)


# Example 1: Gyration in uniform B field
def example_gyration():
    """Pure gyration in uniform magnetic field."""

    # B in the z-direction with E = 0: the simplest setup that produces pure
    # circular (gyro) motion without any drift. Choosing B along z lets us
    # directly read off the gyration in the xy-plane in the trajectory plot.
    B0 = 0.1  # Tesla
    B_func = lambda x, t: np.array([0, 0, B0])
    E_func = lambda x, t: np.array([0, 0, 0])

    # Initial velocity purely in x so that the initial Larmor radius is r_L = v_x/omega_c,
    # making it trivial to verify against the analytic formula without resolving
    # a vector decomposition into parallel and perpendicular components.
    v_perp = 1e6  # m/s
    x0 = [0, 0, 0]
    v0 = [v_perp, 0, 0]

    # Create tracer
    electron = ParticleTracer(q=-e, m=m_e, x0=x0, v0=v0)

    # Gyrofrequency and period
    omega_ce = e * B0 / m_e
    T_gyro = 2 * np.pi / omega_ce
    r_L = v_perp / omega_ce

    print(f"Electron gyration:")
    print(f"  Gyrofrequency: ω_ce = {omega_ce:.3e} rad/s")
    print(f"  Gyroperiod: T = {T_gyro:.3e} s")
    print(f"  Larmor radius: r_L = {r_L:.3e} m = {r_L*1e6:.2f} μm")

    # Trace for 3 periods
    dt = T_gyro / 100
    electron.trace(E_func, B_func, t_max=3*T_gyro, dt=dt)

    # Plot
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(121)
    ax1.plot(electron.x_history[:, 0] * 1e6, electron.x_history[:, 1] * 1e6,
             'b-', linewidth=1.5)
    ax1.scatter([0], [0], c='r', s=100, marker='x', label='Guiding center')
    ax1.set_xlabel('x [μm]', fontsize=11)
    ax1.set_ylabel('y [μm]', fontsize=11)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Electron Gyration (top view)', fontsize=12, fontweight='bold')

    ax2 = fig.add_subplot(122)
    ax2.plot(electron.t_history / T_gyro,
             np.sqrt(electron.v_history[:, 0]**2 + electron.v_history[:, 1]**2),
             'r-', linewidth=1.5, label='$v_\\perp$')
    ax2.axhline(v_perp, color='k', linestyle='--', alpha=0.5, label='Expected')
    ax2.set_xlabel('Time [gyroperiods]', fontsize=11)
    ax2.set_ylabel('Perpendicular velocity [m/s]', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Velocity Conservation', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('gyration_uniform_B.png', dpi=150)
    plt.show()

example_gyration()


# Example 2: Helical trajectory
def example_helix():
    """Helical trajectory with parallel velocity."""

    B0 = 0.5  # Tesla
    B_func = lambda x, t: np.array([0, 0, B0])
    E_func = lambda x, t: np.array([0, 0, 0])

    # Proton with both perpendicular and parallel velocity
    v_perp = 1e5  # m/s
    v_para = 2e5  # m/s
    x0 = [0, 0, 0]
    v0 = [v_perp, 0, v_para]

    proton = ParticleTracer(q=e, m=m_p, x0=x0, v0=v0)

    # Parameters
    omega_ci = e * B0 / m_p
    T_gyro = 2 * np.pi / omega_ci
    r_L = v_perp / omega_ci
    pitch = 2 * np.pi * v_para / omega_ci

    print(f"\nProton helix:")
    print(f"  Gyroperiod: T = {T_gyro:.3e} s")
    print(f"  Larmor radius: r_L = {r_L:.3e} m = {r_L*1e3:.2f} mm")
    print(f"  Pitch: {pitch:.3e} m = {pitch*100:.2f} cm")

    dt = T_gyro / 100
    proton.trace(E_func, B_func, t_max=5*T_gyro, dt=dt)

    # 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = proton.x_history[:, 0] * 1e3
    y = proton.x_history[:, 1] * 1e3
    z = proton.x_history[:, 2]

    ax.plot(x, y, z, 'b-', linewidth=1.5, alpha=0.8)
    ax.scatter([x[0]], [y[0]], [z[0]], c='g', s=100, marker='o', label='Start')
    ax.scatter([x[-1]], [y[-1]], [z[-1]], c='r', s=100, marker='s', label='End')

    ax.set_xlabel('x [mm]', fontsize=11)
    ax.set_ylabel('y [mm]', fontsize=11)
    ax.set_zlabel('z [m]', fontsize=11)
    ax.set_title('Proton Helical Trajectory', fontsize=13, fontweight='bold')
    ax.legend()

    plt.savefig('helical_trajectory.png', dpi=150)
    plt.show()

example_helix()


# Example 3: E×B drift
def example_ExB_drift():
    """E×B drift in crossed electric and magnetic fields."""

    # Fields
    E0 = 1e3  # V/m (in x-direction)
    B0 = 0.1  # T (in z-direction)
    E_func = lambda x, t: np.array([E0, 0, 0])
    B_func = lambda x, t: np.array([0, 0, B0])

    # E×B drift velocity (should be in -y direction)
    v_ExB = E0 / B0  # m/s

    print(f"\nE×B drift:")
    print(f"  E = {E0} V/m (x-direction)")
    print(f"  B = {B0} T (z-direction)")
    print(f"  v_E = E/B = {v_ExB:.3e} m/s (-y direction)")

    # Electron and proton with small initial perpendicular velocity
    v_perp = 1e5  # m/s
    x0 = [0, 0, 0]
    v0_e = [v_perp, 0, 0]
    v0_p = [v_perp, 0, 0]

    electron = ParticleTracer(q=-e, m=m_e, x0=x0, v0=v0_e)
    proton = ParticleTracer(q=e, m=m_p, x0=x0, v0=v0_p)

    # Gyroperiods
    T_e = 2 * np.pi * m_e / (e * B0)
    T_p = 2 * np.pi * m_p / (e * B0)

    dt_e = T_e / 100
    dt_p = T_p / 100
    t_max = 20 * T_p  # Simulate long enough to see drift

    electron.trace(E_func, B_func, t_max, dt_e)
    proton.trace(E_func, B_func, t_max, dt_p)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(electron.x_history[:, 0] * 1e3, electron.x_history[:, 1] * 1e6,
            'b-', linewidth=1, alpha=0.7, label='Electron')
    ax.plot(proton.x_history[:, 0] * 1e3, proton.x_history[:, 1] * 1e6,
            'r-', linewidth=1, alpha=0.7, label='Proton')

    # Expected drift
    y_drift = -v_ExB * t_max
    ax.axhline(y_drift * 1e6, color='k', linestyle='--', linewidth=2,
               label=f'Expected drift: {y_drift*1e6:.1f} μm')

    ax.set_xlabel('x [mm]', fontsize=11)
    ax.set_ylabel('y [μm]', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('E×B Drift (both species drift together)', fontsize=12, fontweight='bold')

    # Guiding center positions
    ax = axes[1]
    # Average over gyration
    window = 20
    y_e_gc = np.convolve(electron.x_history[:, 1], np.ones(window)/window, mode='valid')
    y_p_gc = np.convolve(proton.x_history[:, 1], np.ones(window)/window, mode='valid')
    t_gc = electron.t_history[:len(y_e_gc)]

    ax.plot(t_gc * 1e6, y_e_gc * 1e6, 'b-', linewidth=2, label='Electron GC')
    ax.plot(proton.t_history[:len(y_p_gc)] * 1e6, y_p_gc * 1e6,
            'r-', linewidth=2, label='Proton GC')

    # Expected
    ax.plot(t_gc * 1e6, -v_ExB * t_gc * 1e6, 'k--', linewidth=2,
            label=f'Expected: $-v_E t$')

    ax.set_xlabel('Time [μs]', fontsize=11)
    ax.set_ylabel('Guiding Center y [μm]', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Guiding Center Drift', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('ExB_drift.png', dpi=150)
    plt.show()

example_ExB_drift()
```

### 6.4 Validation: Energy Conservation

```python
def validate_energy_conservation():
    """Check energy conservation in various field configurations."""

    B0 = 1.0
    v0 = 1e6

    # Three configs test different energy-conservation scenarios:
    # "B only" — total KE must be strictly constant (B does no work);
    # "E⊥ + B" — E×B drift does no work, so KE is again conserved;
    # "E∥ + B" — E accelerates along B, so KE grows while PE decreases;
    # tracking all three reveals which parts of the Boris algorithm preserve
    # which conservation laws, and demonstrates where naive integrators fail.
    configs = [
        ("B only", lambda x, t: np.array([0, 0, 0]),
                    lambda x, t: np.array([0, 0, B0])),
        ("E⊥ + B", lambda x, t: np.array([1e3, 0, 0]),
                   lambda x, t: np.array([0, 0, B0])),
        ("E∥ + B", lambda x, t: np.array([0, 0, 1e3]),
                   lambda x, t: np.array([0, 0, B0])),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (name, E_func, B_func) in zip(axes, configs):
        electron = ParticleTracer(q=-e, m=m_e, x0=[0, 0, 0], v0=[v0, 0, 0])

        T_gyro = 2 * np.pi * m_e / (e * B0)
        # dt = T_gyro / 100 is a commonly used rule of thumb: 100 steps per
        # gyroperiod keeps the Boris rotation small enough that the discrete
        # orbit closely tracks the analytic circle, while still being 10x
        # coarser than what an explicit Euler integrator would require.
        dt = T_gyro / 100
        electron.trace(E_func, B_func, t_max=10*T_gyro, dt=dt)

        # Compute kinetic energy
        KE = 0.5 * m_e * np.sum(electron.v_history**2, axis=1)

        # For E∥, add electrostatic potential energy so the conserved quantity
        # is the total mechanical energy KE + PE, not just KE. Omitting PE in
        # this case would show a growing KE and falsely suggest non-conservation.
        if name == "E∥ + B":
            PE = -(-e) * 1e3 * electron.x_history[:, 2]
            total_E = KE + PE
        else:
            total_E = KE

        # Normalize
        total_E /= total_E[0]

        ax.plot(electron.t_history / T_gyro, total_E, 'b-', linewidth=2)
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time [gyroperiods]', fontsize=11)
        ax.set_ylabel('Normalized Total Energy', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # y-limits ±0.001 reveal sub-0.1% energy drift that distinguishes
        # the Boris algorithm's near-perfect conservation from methods that
        # accumulate secular errors.
        ax.set_ylim(0.999, 1.001)

    plt.tight_layout()
    plt.savefig('energy_conservation_boris.png', dpi=150)
    plt.show()

validate_energy_conservation()
```

## Summary

Single particle motion in electromagnetic fields forms the foundation of plasma kinetic theory:

1. **Lorentz force** governs charged particle trajectories: $m d\mathbf{v}/dt = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$, with magnetic force doing no work.

2. **Cyclotron motion** in uniform $\mathbf{B}$ produces helical trajectories with gyrofrequency $\omega_c = |q|B/m$ and Larmor radius $r_L = mv_\perp/(|q|B)$.

3. **E×B drift** arises from perpendicular electric fields, giving drift velocity $\mathbf{v}_E = (\mathbf{E} \times \mathbf{B})/B^2$ that is independent of charge and mass.

4. **Guiding center approximation** separates fast gyration from slow drift, enabling efficient tracking of particle motion when $r_L \ll L$.

5. **Boris algorithm** provides a robust, energy-conserving numerical method for integrating particle trajectories in arbitrary electromagnetic fields.

In the next lesson, we'll extend these concepts to non-uniform fields, introducing grad-B drift, curvature drift, and magnetic mirror effects.

## Practice Problems

### Problem 1: Cyclotron Frequency and Radius

An electron with kinetic energy $E_k = 100$ eV enters a uniform magnetic field $B = 0.5$ T perpendicularly.

(a) Calculate the cyclotron frequency $\omega_{ce}$ and gyroperiod $T$.

(b) Find the Larmor radius $r_L$.

(c) If the electron also has a parallel velocity $v_\parallel = 10^7$ m/s, compute the pitch of the helix.

(d) Repeat (a)-(c) for a proton with the same energy.

### Problem 2: E×B Drift Calculation

A plasma is in a magnetic field $\mathbf{B} = 2\hat{\mathbf{z}}$ T. An electric field $\mathbf{E} = 500\hat{\mathbf{x}}$ V/m is applied.

(a) Calculate the E×B drift velocity vector.

(b) Verify that this drift is the same for electrons and protons.

(c) If the electric field oscillates at $\omega = 2\pi \times 10^6$ rad/s, does the drift velocity follow instantaneously? (Compare $\omega$ to $\omega_{ce}$ and $\omega_{ci}$.)

(d) Estimate the time-averaged drift if $\mathbf{E} = E_0 \cos(\omega t) \hat{\mathbf{x}}$.

### Problem 3: Magnetron Configuration

In a cylindrical magnetron, an electron at radius $r$ experiences:
- Radial electric field: $E_r = V_0/r$ (where $V_0$ is constant)
- Axial magnetic field: $B_z = B_0$

(a) Compute the azimuthal E×B drift velocity $v_\theta(r)$.

(b) Find the radius $r_0$ where $v_\theta$ is maximum.

(c) For $V_0 = 10$ kV and $B_0 = 0.1$ T, plot $v_\theta(r)$ for $r \in [1, 10]$ cm.

(d) At $r = 5$ cm, calculate the electron Larmor radius. Is $r_L \ll r$?

### Problem 4: Boris Algorithm Analysis

Implement the Boris algorithm for a particle in $\mathbf{B} = B_0 \hat{\mathbf{z}}$ with initial velocity $\mathbf{v}_0 = v_0 \hat{\mathbf{x}}$.

(a) Simulate for 10 gyroperiods with timesteps $\Delta t = T/N$ for $N = 10, 20, 50, 100$.

(b) For each $N$, compute the final kinetic energy and compare to the initial value. Plot the relative error vs. $N$.

(c) Measure the position error after 10 gyroperiods (the particle should return to the starting point). How does it scale with $N$?

(d) Repeat with a simple Euler method and compare accuracy and energy conservation.

### Problem 5: Combined Fields

A particle with charge $q = -e$, mass $m = m_e$, starts at rest at the origin in fields:
- $\mathbf{E} = E_0 \hat{\mathbf{z}}$ (constant)
- $\mathbf{B} = B_0 \hat{\mathbf{z}}$ (constant)

(a) Write down the equations of motion for $v_x(t)$, $v_y(t)$, $v_z(t)$.

(b) Solve for $v_z(t)$. Describe the parallel motion.

(c) For the perpendicular motion, show that the guiding center drifts with the E×B velocity even though $\mathbf{E} \times \mathbf{B} = 0$ in this case. (Hint: This is a trick question—what is E×B when E and B are parallel?)

(d) Numerically integrate the trajectory for $E_0 = 1000$ V/m, $B_0 = 0.1$ T for 100 gyroperiods. Plot $z(t)$ and verify your analytical solution.

---

**Previous:** [Plasma Description Hierarchy](./03_Plasma_Description_Hierarchy.md) | **Next:** [Single Particle Motion II](./05_Single_Particle_Motion_II.md)
