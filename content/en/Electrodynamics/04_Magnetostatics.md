# Magnetostatics

[← Previous: 03. Conductors and Dielectrics](03_Conductors_and_Dielectrics.md) | [Next: 05. Magnetic Vector Potential →](05_Magnetic_Vector_Potential.md)

---

## Learning Objectives

1. State the Lorentz force law and compute the magnetic force on moving charges and current-carrying wires
2. Apply the Biot-Savart law to calculate magnetic fields from current distributions
3. Derive and apply Ampere's law in both integral and differential forms
4. Explain why $\nabla \cdot \mathbf{B} = 0$ (no magnetic monopoles) and its physical consequences
5. Define the magnetic dipole moment and compute the field of a magnetic dipole
6. Calculate the force and torque on a magnetic dipole in an external field
7. Implement Biot-Savart calculations numerically using Python

---

We now cross from the world of electric charges at rest to the world of charges in motion — electric currents. Moving charges produce a fundamentally new kind of field: the magnetic field $\mathbf{B}$. Unlike the electric field, which points radially toward or away from charges, the magnetic field curls around currents in closed loops. This difference in topology — radial vs. circulating — is encoded in the mathematics: while $\nabla \cdot \mathbf{E} = \rho/\epsilon_0$ and $\nabla \times \mathbf{E} = 0$ for electrostatics, for magnetostatics we have $\nabla \cdot \mathbf{B} = 0$ and $\nabla \times \mathbf{B} = \mu_0 \mathbf{J}$. Everything is "swapped."

---

## The Lorentz Force Law

The magnetic force on a charge $q$ moving with velocity $\mathbf{v}$ in a magnetic field $\mathbf{B}$ is:

$$\mathbf{F}_{\text{mag}} = q\mathbf{v} \times \mathbf{B}$$

Combined with the electric force, this gives the full **Lorentz force**:

$$\mathbf{F} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

Key properties of the magnetic force:
- It is always **perpendicular** to $\mathbf{v}$ — it never does work on the charge
- It changes the **direction** of motion, not the **speed**
- A charged particle in a uniform $\mathbf{B}$ field moves in a **circle** (or helix if there is a component of $\mathbf{v}$ parallel to $\mathbf{B}$)

### Cyclotron Motion

For a charge $q$ moving perpendicular to a uniform field $\mathbf{B}$, the circular orbit has:

$$r = \frac{mv}{qB} \qquad \text{(cyclotron radius)}$$

$$\omega_c = \frac{qB}{m} \qquad \text{(cyclotron frequency)}$$

The cyclotron frequency is independent of speed — this remarkable fact is the basis of the cyclotron particle accelerator.

> **Analogy**: The magnetic force is like a hand that always pushes perpendicular to your direction of travel. If you are walking straight and someone constantly pushes you sideways (perpendicular to your velocity), you end up walking in a circle. The push never speeds you up or slows you down — it only bends your path.

### Force on a Current-Carrying Wire

For a wire carrying current $I$ in a magnetic field:

$$\mathbf{F} = I \int d\mathbf{l} \times \mathbf{B}$$

For a straight wire of length $L$ in a uniform field:

$$F = BIL\sin\theta$$

where $\theta$ is the angle between the wire and the field.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Simulate charged particle motion in a magnetic field
# Why 3D: the helical motion requires three spatial dimensions

q = 1.6e-19    # proton charge (C)
m = 1.67e-27   # proton mass (kg)
B = np.array([0, 0, 1e-3])  # magnetic field in z-direction (1 mT)

# Initial conditions: velocity has both perpendicular and parallel components
v0 = np.array([1e5, 0, 3e4])  # m/s — mostly perpendicular, some parallel

# Time parameters
# Why ω_c: we need the cyclotron period to choose appropriate time steps
omega_c = q * np.linalg.norm(B) / m
T_c = 2 * np.pi / omega_c   # cyclotron period
dt = T_c / 200               # 200 steps per period for smooth curves
N_steps = 1000

# Integrate equations of motion: dv/dt = (q/m)(v × B)
# Why Euler-Cromer: it preserves energy better than simple Euler for oscillatory motion
pos = np.zeros((N_steps, 3))
vel = np.zeros((N_steps, 3))
pos[0] = np.array([0, 0, 0])
vel[0] = v0

for i in range(N_steps - 1):
    # Lorentz force (no electric field)
    a = (q / m) * np.cross(vel[i], B)
    vel[i+1] = vel[i] + a * dt
    pos[i+1] = pos[i] + vel[i+1] * dt  # Euler-Cromer: use updated velocity

# Analytical cyclotron radius
r_c = m * np.sqrt(v0[0]**2 + v0[1]**2) / (q * np.linalg.norm(B))
print(f"Cyclotron frequency: ω_c = {omega_c:.4e} rad/s")
print(f"Cyclotron period:    T_c = {T_c:.4e} s")
print(f"Cyclotron radius:    r_c = {r_c:.4f} m")

fig = plt.figure(figsize=(12, 5))

# 3D helical trajectory
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', linewidth=0.8)
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_zlabel('z (m)')
ax1.set_title('Helical Motion in Uniform B')

# xy projection — circular motion
ax2 = fig.add_subplot(122)
ax2.plot(pos[:, 0], pos[:, 1], 'b-', linewidth=0.8)
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_title('xy-Projection (Circular)')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# Draw expected cyclotron circle for comparison
theta_circ = np.linspace(0, 2*np.pi, 200)
ax2.plot(r_c * np.cos(theta_circ) + r_c, r_c * np.sin(theta_circ),
         'r--', alpha=0.5, label=f'r_c = {r_c:.3f} m')
ax2.legend()

plt.tight_layout()
plt.savefig('cyclotron_motion.png', dpi=150)
plt.show()
```

---

## The Biot-Savart Law

The Biot-Savart law gives the magnetic field produced by a steady current:

$$\mathbf{B}(\mathbf{r}) = \frac{\mu_0}{4\pi} \int \frac{I \, d\mathbf{l}' \times \hat{\boldsymbol{\mathscr{r}}}}{{|\boldsymbol{\mathscr{r}}|}^2}$$

where:
- $\mu_0 = 4\pi \times 10^{-7}$ T$\cdot$m/A is the permeability of free space
- $d\mathbf{l}'$ is a current element along the wire
- $\boldsymbol{\mathscr{r}} = \mathbf{r} - \mathbf{r}'$ is the vector from source to field point

For a volume current density $\mathbf{J}$:

$$\mathbf{B}(\mathbf{r}) = \frac{\mu_0}{4\pi} \int \frac{\mathbf{J}(\mathbf{r}') \times \hat{\boldsymbol{\mathscr{r}}}}{|\boldsymbol{\mathscr{r}}|^2} \, d\tau'$$

### Example: Magnetic Field of a Circular Loop

For a circular loop of radius $R$ carrying current $I$, the field on the axis at distance $z$ from the center:

$$B_z = \frac{\mu_0 I R^2}{2(R^2 + z^2)^{3/2}}$$

At the center of the loop ($z = 0$):

$$B_{\text{center}} = \frac{\mu_0 I}{2R}$$

### Example: Infinite Straight Wire

An infinitely long straight wire carrying current $I$. By the Biot-Savart law (or more easily by Ampere's law):

$$B = \frac{\mu_0 I}{2\pi s}$$

where $s$ is the perpendicular distance from the wire. The field circles the wire — the direction is given by the right-hand rule.

```python
import numpy as np
import matplotlib.pyplot as plt

# Biot-Savart law: magnetic field of a circular current loop
# Why numerical: the off-axis field has no simple closed form

mu_0 = 4 * np.pi * 1e-7   # permeability of free space (T·m/A)

def biot_savart_loop(R, I, field_points, N_segments=1000):
    """
    Compute B field from a circular loop of radius R carrying current I.

    Parameters:
        R: loop radius (m)
        I: current (A)
        field_points: array of shape (M, 3) — points where B is evaluated
        N_segments: number of segments to discretize the loop

    Returns:
        B: array of shape (M, 3) — magnetic field at each point
    """
    # Discretize the loop into small current elements
    # Why many segments: accuracy improves with finer discretization
    phi = np.linspace(0, 2 * np.pi, N_segments, endpoint=False)
    dphi = 2 * np.pi / N_segments

    # Position of each current element on the loop (in xy-plane)
    loop_x = R * np.cos(phi)
    loop_y = R * np.sin(phi)
    loop_z = np.zeros_like(phi)

    # Current element direction: dl' = R dphi * (-sin φ, cos φ, 0)
    dl_x = -R * np.sin(phi) * dphi
    dl_y = R * np.cos(phi) * dphi
    dl_z = np.zeros_like(phi)

    B = np.zeros_like(field_points)

    for i in range(len(field_points)):
        # Separation vector: r - r'
        rx = field_points[i, 0] - loop_x
        ry = field_points[i, 1] - loop_y
        rz = field_points[i, 2] - loop_z
        r_mag = np.sqrt(rx**2 + ry**2 + rz**2)
        r_mag = np.maximum(r_mag, 1e-10)

        # Cross product dl' × r_hat = dl' × (r/|r|)
        # Why cross product: this is the core of Biot-Savart
        cross_x = dl_y * rz - dl_z * ry
        cross_y = dl_z * rx - dl_x * rz
        cross_z = dl_x * ry - dl_y * rx

        # Sum contributions from all segments: B = (μ₀I/4π) Σ (dl'×r̂)/r²
        B[i, 0] = (mu_0 * I / (4 * np.pi)) * np.sum(cross_x / r_mag**3)
        B[i, 1] = (mu_0 * I / (4 * np.pi)) * np.sum(cross_y / r_mag**3)
        B[i, 2] = (mu_0 * I / (4 * np.pi)) * np.sum(cross_z / r_mag**3)

    return B

# Compute on-axis field and compare with analytic formula
R = 0.1    # 10 cm radius
I = 1.0    # 1 A current

z_vals = np.linspace(-0.5, 0.5, 200)
field_pts = np.column_stack([np.zeros_like(z_vals), np.zeros_like(z_vals), z_vals])
B = biot_savart_loop(R, I, field_pts)

# Analytic on-axis formula
B_analytic = mu_0 * I * R**2 / (2 * (R**2 + z_vals**2)**1.5)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(z_vals * 100, B[:, 2] * 1e6, 'b-', linewidth=2, label='Biot-Savart (numerical)')
axes[0].plot(z_vals * 100, B_analytic * 1e6, 'r--', linewidth=2, label='Analytic')
axes[0].set_xlabel('z (cm)')
axes[0].set_ylabel('$B_z$ ($\\mu$T)')
axes[0].set_title('On-Axis Field of Circular Loop')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Compute field in the xz-plane for a 2D cross-section
x_grid = np.linspace(-0.3, 0.3, 40)
z_grid = np.linspace(-0.3, 0.3, 40)
XG, ZG = np.meshgrid(x_grid, z_grid)

pts = np.column_stack([XG.ravel(), np.zeros(XG.size), ZG.ravel()])
B_grid = biot_savart_loop(R, I, pts, N_segments=500)
Bx = B_grid[:, 0].reshape(XG.shape)
Bz = B_grid[:, 2].reshape(ZG.shape)
B_mag = np.sqrt(Bx**2 + Bz**2)

axes[1].streamplot(XG, ZG, Bx, Bz, color=np.log10(B_mag + 1e-12),
                   cmap='viridis', density=2)
axes[1].plot([-R, R], [0, 0], 'ro', markersize=8)
axes[1].set_xlabel('x (m)')
axes[1].set_ylabel('z (m)')
axes[1].set_title('B Field of Circular Loop (xz-plane)')
axes[1].set_aspect('equal')

plt.suptitle('Biot-Savart Law: Current Loop', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('biot_savart_loop.png', dpi=150)
plt.show()
```

---

## Ampere's Law

Ampere's law relates the circulation of $\mathbf{B}$ around a closed loop to the enclosed current:

### Integral Form

$$\oint \mathbf{B} \cdot d\mathbf{l} = \mu_0 I_{\text{enc}}$$

### Differential Form

$$\nabla \times \mathbf{B} = \mu_0 \mathbf{J}$$

Ampere's law is to magnetostatics what Gauss's law is to electrostatics — it is always true, but most useful when symmetry allows us to pull $\mathbf{B}$ out of the integral.

### Applications

**Long straight wire**: Circular Amperian loop of radius $s$:
$$B(2\pi s) = \mu_0 I \implies B = \frac{\mu_0 I}{2\pi s}$$

**Infinite solenoid** (n turns per unit length, current I): The field is uniform inside and zero outside:
$$B = \mu_0 n I \quad \text{(inside)}, \qquad B = 0 \quad \text{(outside)}$$

**Toroidal solenoid** (N total turns, radius $R$): Inside the torus at distance $s$ from the center:
$$B = \frac{\mu_0 N I}{2\pi s}$$

> **Analogy**: Ampere's law says that if you walk around a closed path and measure how much $\mathbf{B}$ "follows along" with you (the line integral), the total is proportional to the current threading through your path — like counting how many water pipes pass through a hoop by measuring the total flow around the hoop's circumference.

---

## No Magnetic Monopoles

The divergence of the magnetic field is always zero:

$$\nabla \cdot \mathbf{B} = 0$$

In integral form (via the divergence theorem):

$$\oint \mathbf{B} \cdot d\mathbf{a} = 0$$

This means:
- **Magnetic field lines have no beginning or end** — they always form closed loops
- **There are no magnetic monopoles** (no isolated north or south poles)
- $\mathbf{B}$ is a **solenoidal** field

This is in stark contrast to $\mathbf{E}$, whose divergence is $\rho/\epsilon_0$. If magnetic monopoles existed, we would have $\nabla \cdot \mathbf{B} = \mu_0 \rho_m$ for magnetic charge density $\rho_m$. Despite extensive searches, no monopole has ever been found (though they are predicted by some grand unified theories).

---

## Magnetic Dipole

### Magnetic Dipole Moment

A small current loop of area $A$ carrying current $I$ has a magnetic dipole moment:

$$\mathbf{m} = I \mathbf{A} = IA\hat{n}$$

where $\hat{n}$ is the normal to the loop (direction given by the right-hand rule with current direction).

### Field of a Magnetic Dipole

Far from the loop ($r \gg \sqrt{A}$), the field has the characteristic dipole pattern:

$$\mathbf{B}_{\text{dip}} = \frac{\mu_0}{4\pi r^3}\left[3(\mathbf{m} \cdot \hat{r})\hat{r} - \mathbf{m}\right] = \frac{\mu_0 m}{4\pi r^3}\left(2\cos\theta\,\hat{r} + \sin\theta\,\hat{\theta}\right)$$

This has exactly the same angular structure as the electric dipole field! The field falls off as $1/r^3$.

### Torque and Force on a Dipole

A magnetic dipole $\mathbf{m}$ in an external field $\mathbf{B}$ experiences:

**Torque**:
$$\boldsymbol{\tau} = \mathbf{m} \times \mathbf{B}$$

This torque tends to align $\mathbf{m}$ with $\mathbf{B}$ — this is how compass needles work.

**Potential energy**:
$$U = -\mathbf{m} \cdot \mathbf{B}$$

Minimum energy when $\mathbf{m} \parallel \mathbf{B}$ (aligned), maximum when anti-aligned.

**Force** (in a non-uniform field):
$$\mathbf{F} = \nabla(\mathbf{m} \cdot \mathbf{B})$$

A dipole is attracted toward regions of stronger field — this is why magnets attract iron filings.

```python
import numpy as np
import matplotlib.pyplot as plt

# Compare electric and magnetic dipole fields — they have the same structure!
# Why compare: seeing the structural identity deepens understanding of both

mu_0 = 4 * np.pi * 1e-7
epsilon_0 = 8.854e-12

# Both dipoles pointing in the z-direction
m = 1.0   # magnetic dipole moment (A·m²)
p = 1.0   # electric dipole moment (C·m) — normalized for visual comparison

# Grid in the xz-plane
x = np.linspace(-2, 2, 40)
z = np.linspace(-2, 2, 40)
X, Z = np.meshgrid(x, z)

r = np.sqrt(X**2 + Z**2)
r = np.maximum(r, 0.3)  # exclude region near origin

# Angles: cos θ = z/r, sin θ = x/r (in xz-plane, θ measured from z-axis)
cos_theta = Z / r
sin_theta = X / r

# Magnetic dipole field: B = (μ₀m/4πr³)(2cosθ r̂ + sinθ θ̂)
# Convert to Cartesian: r̂ = sinθ x̂ + cosθ ẑ, θ̂ = cosθ x̂ - sinθ ẑ
prefactor_B = mu_0 * m / (4 * np.pi * r**3)
Br = prefactor_B * 2 * cos_theta     # radial component
Bt = prefactor_B * sin_theta          # theta component

# To Cartesian
Bx = Br * sin_theta + Bt * cos_theta
Bz = Br * cos_theta - Bt * sin_theta

# Electric dipole field (same structure with different prefactor)
prefactor_E = p / (4 * np.pi * epsilon_0 * r**3)
Er = prefactor_E * 2 * cos_theta
Et = prefactor_E * sin_theta
Ex = Er * sin_theta + Et * cos_theta
Ez = Er * cos_theta - Et * sin_theta

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

B_mag = np.sqrt(Bx**2 + Bz**2)
axes[0].streamplot(X, Z, Bx, Bz, color=np.log10(B_mag), cmap='plasma', density=2)
axes[0].set_xlabel('x')
axes[0].set_ylabel('z')
axes[0].set_title('Magnetic Dipole Field')
axes[0].set_aspect('equal')

E_mag = np.sqrt(Ex**2 + Ez**2)
axes[1].streamplot(X, Z, Ex, Ez, color=np.log10(E_mag), cmap='plasma', density=2)
axes[1].set_xlabel('x')
axes[1].set_ylabel('z')
axes[1].set_title('Electric Dipole Field')
axes[1].set_aspect('equal')

plt.suptitle('Dipole Fields: Magnetic vs Electric (Same Structure!)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('dipole_comparison.png', dpi=150)
plt.show()
```

---

## Magnetic Materials (Brief Overview)

Just as dielectrics respond to electric fields with polarization $\mathbf{P}$, magnetic materials respond to magnetic fields with **magnetization** $\mathbf{M}$ (magnetic dipole moment per unit volume):

$$\mathbf{B} = \mu_0(\mathbf{H} + \mathbf{M})$$

where $\mathbf{H}$ is the magnetic field intensity (analogous to $\mathbf{D}$ for electrostatics).

For **linear** magnetic materials: $\mathbf{M} = \chi_m \mathbf{H}$ and $\mathbf{B} = \mu_0(1+\chi_m)\mathbf{H} = \mu\mathbf{H}$.

There are three classes:

| Type | $\chi_m$ | Examples | Mechanism |
|---|---|---|---|
| **Diamagnetic** | $\sim -10^{-5}$ | Cu, Ag, H$_2$O | Orbital electron response |
| **Paramagnetic** | $\sim 10^{-5}$ to $10^{-3}$ | Al, O$_2$, Pt | Alignment of permanent moments |
| **Ferromagnetic** | $\sim 10^2$ to $10^5$ | Fe, Co, Ni | Domain alignment + exchange interaction |

Ferromagnetic materials exhibit **hysteresis** — their magnetization depends on their history, not just the current field. This is why permanent magnets exist and why magnetic data storage works.

---

## Force Between Parallel Wires

Two parallel wires separated by distance $d$, carrying currents $I_1$ and $I_2$:

$$\frac{F}{L} = \frac{\mu_0 I_1 I_2}{2\pi d}$$

- **Parallel currents attract** (same direction)
- **Anti-parallel currents repel** (opposite direction)

This force defines the SI unit of current: the ampere is defined such that two wires 1 m apart carrying 1 A each experience a force of $2 \times 10^{-7}$ N/m.

---

## Comparison: Electrostatics vs. Magnetostatics

| Property | Electrostatics | Magnetostatics |
|---|---|---|
| Source | Charges $\rho$ | Currents $\mathbf{J}$ |
| Field | $\mathbf{E}$ | $\mathbf{B}$ |
| Force law | $\mathbf{F} = q\mathbf{E}$ | $\mathbf{F} = q\mathbf{v} \times \mathbf{B}$ |
| Divergence | $\nabla \cdot \mathbf{E} = \rho/\epsilon_0$ | $\nabla \cdot \mathbf{B} = 0$ |
| Curl | $\nabla \times \mathbf{E} = 0$ | $\nabla \times \mathbf{B} = \mu_0\mathbf{J}$ |
| Source law | Gauss's law | Ampere's law |
| Potential | $V$ (scalar) | $\mathbf{A}$ (vector) |
| Dipole moment | $\mathbf{p} = q\mathbf{d}$ | $\mathbf{m} = I\mathbf{A}$ |

---

## Summary

| Concept | Key Equation |
|---|---|
| Lorentz force | $\mathbf{F} = q(\mathbf{E} + \mathbf{v}\times\mathbf{B})$ |
| Cyclotron radius | $r = mv/(qB)$ |
| Biot-Savart | $\mathbf{B} = \frac{\mu_0}{4\pi}\int \frac{I\,d\mathbf{l}'\times\hat{\boldsymbol{\mathscr{r}}}}{|\boldsymbol{\mathscr{r}}|^2}$ |
| Ampere's law (integral) | $\oint \mathbf{B}\cdot d\mathbf{l} = \mu_0 I_{\text{enc}}$ |
| Ampere's law (differential) | $\nabla\times\mathbf{B} = \mu_0\mathbf{J}$ |
| No monopoles | $\nabla \cdot \mathbf{B} = 0$ |
| Solenoid | $B = \mu_0 n I$ |
| Dipole moment | $\mathbf{m} = IA\hat{n}$ |
| Torque on dipole | $\boldsymbol{\tau} = \mathbf{m}\times\mathbf{B}$ |
| Force on dipole | $\mathbf{F} = \nabla(\mathbf{m}\cdot\mathbf{B})$ |

---

## Exercises

### Exercise 1: Cyclotron Simulation
Modify the cyclotron simulation to include a uniform electric field $\mathbf{E} = E_0 \hat{x}$. Observe the **$\mathbf{E} \times \mathbf{B}$ drift** — the particle drifts perpendicular to both $\mathbf{E}$ and $\mathbf{B}$ with velocity $v_d = E/B$. Verify this numerically.

### Exercise 2: Helmholtz Coils
Two coaxial circular loops of radius $R$, separated by distance $R$, each carrying current $I$ in the same direction. Compute and plot $B_z$ along the axis. Show that the field is remarkably uniform near the midpoint (the first and second derivatives of $B_z$ vanish there).

### Exercise 3: Biot-Savart for a Finite Wire
A straight wire of length $2L$ carries current $I$. Use the Biot-Savart law to derive the field at perpendicular distance $s$ from the wire. Show that for $L \to \infty$, you recover $B = \mu_0 I/(2\pi s)$.

### Exercise 4: Force Between Current Loops
Two coaxial circular loops of radius $R$ carry currents $I_1$ and $I_2$. Numerically compute the force between them as a function of their separation $d$. Use the fact that $F = \nabla(\mathbf{m} \cdot \mathbf{B})$ for large separations and compare with the full numerical result.

### Exercise 5: Magnetic Field of a Solenoid (Numerical)
Model a solenoid as $N$ circular loops uniformly spaced along the z-axis. Use the Biot-Savart law to compute the field everywhere. Plot $B_z$ along the axis and verify the interior field approaches $\mu_0 n I$. Examine the fringing field near the ends.

---

[← Previous: 03. Conductors and Dielectrics](03_Conductors_and_Dielectrics.md) | [Next: 05. Magnetic Vector Potential →](05_Magnetic_Vector_Potential.md)
