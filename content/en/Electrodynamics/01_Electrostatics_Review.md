# Electrostatics Review

[Next: 02. Electric Potential and Energy →](02_Electric_Potential_and_Energy.md)

---

## Learning Objectives

1. State Coulomb's law and compute the force between point charges in vector form
2. Define the electric field and apply the superposition principle to multiple charges
3. Compute electric fields from continuous charge distributions (line, surface, volume)
4. Derive and apply Gauss's law in both integral and differential forms
5. Use Gauss's law to find electric fields with high symmetry (spherical, cylindrical, planar)
6. Understand the divergence and curl of the electrostatic field
7. Implement numerical calculations of electric fields from charge distributions in Python

---

Electrostatics is the bedrock upon which all of electrodynamics is built. Before we can understand electromagnetic waves, radiation, or the full glory of Maxwell's equations, we must first master the physics of charges at rest. This lesson revisits the fundamental laws governing static electric fields — Coulomb's law, the superposition principle, and Gauss's law — and develops them into the precise mathematical language of vector calculus. Every equation here will reappear, generalized and deepened, as we progress through the course.

---

## Coulomb's Law

The starting point of electrostatics is the experimental observation that two point charges exert forces on each other. Coulomb's law gives the force on charge $q_2$ due to charge $q_1$:

$$\mathbf{F}_{12} = \frac{1}{4\pi\epsilon_0} \frac{q_1 q_2}{|\mathbf{r}_2 - \mathbf{r}_1|^2} \hat{\mathbf{r}}_{12}$$

where:
- $\epsilon_0 \approx 8.854 \times 10^{-12}$ C$^2$/(N$\cdot$m$^2$) is the permittivity of free space
- $\mathbf{r}_1, \mathbf{r}_2$ are the position vectors of the two charges
- $\hat{\mathbf{r}}_{12} = \frac{\mathbf{r}_2 - \mathbf{r}_1}{|\mathbf{r}_2 - \mathbf{r}_1|}$ is the unit vector from $q_1$ to $q_2$

The constant $k_e = \frac{1}{4\pi\epsilon_0} \approx 8.988 \times 10^9$ N$\cdot$m$^2$/C$^2$ is often used for brevity.

### Historical Context

Charles-Augustin de Coulomb established this law experimentally in 1785 using a torsion balance. The precision of his measurements was remarkable for the era, confirming the inverse-square dependence to within a fraction of a percent. Modern experiments constrain the exponent to $2 \pm 10^{-16}$ — making Coulomb's law one of the most precisely verified laws in all of physics.

### Comparison with Gravity

| Property | Coulomb | Newton (gravity) |
|---|---|---|
| Force law | $F \propto q_1 q_2/r^2$ | $F \propto m_1 m_2/r^2$ |
| Sign of source | Both $+$ and $-$ | Only $+$ (mass > 0) |
| Attraction/repulsion | Both | Attraction only |
| Relative strength | $\sim 10^{36}$ times stronger | 1 (baseline) |
| Mediating field | Electric field $\mathbf{E}$ | Gravitational field $\mathbf{g}$ |

The ratio of electric to gravitational force between a proton and an electron is $F_e/F_g \approx 2.3 \times 10^{39}$ — electromagnetism utterly dominates at the atomic scale.

**Key properties:**
- The force is **central** — it acts along the line joining the two charges
- It obeys **Newton's third law**: $\mathbf{F}_{12} = -\mathbf{F}_{21}$
- Like charges repel ($F > 0$), opposite charges attract ($F < 0$)
- It follows an **inverse-square law**, just like gravity

> **Analogy**: Think of Coulomb's law as gravity's electric cousin. Both are inverse-square forces between "sources" (mass for gravity, charge for electricity). The crucial difference is that electric charge comes in two signs — positive and negative — allowing both attraction and repulsion, while gravity only attracts.

---

## The Electric Field

Rather than thinking about forces between pairs of charges, we define the **electric field** $\mathbf{E}$ as the force per unit positive test charge:

$$\mathbf{E}(\mathbf{r}) = \frac{\mathbf{F}}{q_{\text{test}}} = \frac{1}{4\pi\epsilon_0} \frac{q}{|\mathbf{r} - \mathbf{r}'|^2} \hat{\mathbf{r}}$$

where $\mathbf{r}'$ is the source charge position and $\mathbf{r}$ is the field point.

The electric field is a **vector field** — it assigns a vector to every point in space. This is a profound conceptual shift: the field exists in space whether or not a test charge is present to "feel" it.

### Superposition Principle

For a collection of $N$ point charges $q_1, q_2, \ldots, q_N$ at positions $\mathbf{r}_1', \mathbf{r}_2', \ldots, \mathbf{r}_N'$, the total electric field at point $\mathbf{r}$ is the vector sum:

$$\mathbf{E}(\mathbf{r}) = \frac{1}{4\pi\epsilon_0} \sum_{i=1}^{N} \frac{q_i}{|\mathbf{r} - \mathbf{r}_i'|^2} \hat{\boldsymbol{\mathscr{r}}}_i$$

where $\hat{\boldsymbol{\mathscr{r}}}_i = \frac{\mathbf{r} - \mathbf{r}_i'}{|\mathbf{r} - \mathbf{r}_i'|}$.

Superposition is **exact** — there are no corrections, no higher-order terms. The field from each charge is completely independent of all others.

```python
import numpy as np
import matplotlib.pyplot as plt

# Compute electric field from multiple point charges using superposition
# Why NumPy: vectorized operations let us evaluate the field on a grid efficiently

k_e = 8.988e9  # Coulomb constant (N*m^2/C^2)

# Define charges: (x, y, charge) — a dipole-like configuration
charges = [
    (−0.5, 0.0, 1e-9),   # +1 nC at (-0.5, 0)
    ( 0.5, 0.0, -1e-9),   # -1 nC at (+0.5, 0)
]

# Create a 2D grid of field points
# Why meshgrid: we need E evaluated at every (x,y) point for visualization
x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)

Ex = np.zeros_like(X)
Ey = np.zeros_like(Y)

# Superposition: sum contributions from each charge
for (qx, qy, q) in charges:
    dx = X - qx                       # displacement vectors (x-component)
    dy = Y - qy                       # displacement vectors (y-component)
    r_sq = dx**2 + dy**2              # squared distance
    r_sq = np.maximum(r_sq, 1e-6)     # avoid division by zero near charges
    r = np.sqrt(r_sq)
    # Why we divide by r^3: E ~ q*r_hat/r^2 = q*(r_vec/r)/r^2 = q*r_vec/r^3
    Ex += k_e * q * dx / r_sq**(3/2)
    Ey += k_e * q * dy / r_sq**(3/2)

# Visualize with streamlines — they follow the direction of E
E_mag = np.sqrt(Ex**2 + Ey**2)
fig, ax = plt.subplots(figsize=(8, 8))
ax.streamplot(X, Y, Ex, Ey, color=np.log10(E_mag), cmap='inferno', density=2)
for (qx, qy, q) in charges:
    color = 'red' if q > 0 else 'blue'
    ax.plot(qx, qy, 'o', color=color, markersize=12)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Electric Field Lines of a Dipole')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('dipole_field.png', dpi=150)
plt.show()
```

---

## Continuous Charge Distributions

Real charge distributions are often continuous. We replace the discrete sum with an integral. There are three cases depending on dimensionality:

| Distribution | Charge element | Electric field |
|---|---|---|
| **Line** (charge per length $\lambda$) | $dq = \lambda \, dl'$ | $\mathbf{E} = \frac{1}{4\pi\epsilon_0}\int \frac{\lambda(\mathbf{r}') \, dl'}{|\mathbf{r}-\mathbf{r}'|^2}\hat{\boldsymbol{\mathscr{r}}}$ |
| **Surface** (charge per area $\sigma$) | $dq = \sigma \, da'$ | $\mathbf{E} = \frac{1}{4\pi\epsilon_0}\int \frac{\sigma(\mathbf{r}') \, da'}{|\mathbf{r}-\mathbf{r}'|^2}\hat{\boldsymbol{\mathscr{r}}}$ |
| **Volume** (charge per volume $\rho$) | $dq = \rho \, d\tau'$ | $\mathbf{E} = \frac{1}{4\pi\epsilon_0}\int \frac{\rho(\mathbf{r}') \, d\tau'}{|\mathbf{r}-\mathbf{r}'|^2}\hat{\boldsymbol{\mathscr{r}}}$ |

### Example: Field of a Uniformly Charged Ring

A ring of radius $R$ carries total charge $Q$ uniformly distributed. On the axis (at distance $z$ from the center):

$$E_z = \frac{1}{4\pi\epsilon_0} \frac{Qz}{(z^2 + R^2)^{3/2}}$$

This follows from symmetry — the transverse components cancel by pairs, and only the axial component survives.

```python
import numpy as np
import matplotlib.pyplot as plt

# Electric field on axis of a uniformly charged ring
# Why analytic + numerical: comparing them verifies our integration approach

epsilon_0 = 8.854e-12
Q = 1e-9       # total charge (1 nC)
R = 0.1        # ring radius (10 cm)

z = np.linspace(-0.5, 0.5, 500)

# Analytic formula — derived from symmetry and direct integration
# Why (z^2 + R^2)^(3/2): this comes from the geometry of the separation vector
E_analytic = Q * z / (4 * np.pi * epsilon_0 * (z**2 + R**2)**1.5)

# Numerical integration — discretize ring into N small charges
N = 1000
theta = np.linspace(0, 2*np.pi, N, endpoint=False)
dq = Q / N  # each small segment carries charge dq

E_numerical = np.zeros_like(z)
for i, zi in enumerate(z):
    # Sum contributions from each segment; only z-component survives
    for th in theta:
        rx = -R * np.cos(th)       # vector from segment to axis point (x)
        ry = -R * np.sin(th)       # vector from segment to axis point (y)
        rz = zi                     # z-component of separation
        r_mag = np.sqrt(rx**2 + ry**2 + rz**2)
        # Why only z: by symmetry, x and y components cancel over the full ring
        E_numerical[i] += dq * rz / (4 * np.pi * epsilon_0 * r_mag**3)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(z * 100, E_analytic, 'b-', linewidth=2, label='Analytic')
ax.plot(z * 100, E_numerical, 'r--', linewidth=2, label=f'Numerical (N={N})')
ax.set_xlabel('z (cm)')
ax.set_ylabel('$E_z$ (V/m)')
ax.set_title('Electric Field on Axis of Charged Ring')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ring_field.png', dpi=150)
plt.show()
```

---

## Divergence and Curl of E

The electric field of a static charge distribution has two fundamental vector-calculus properties:

### Divergence of E

Apply the divergence theorem to Gauss's law (derived below) to obtain:

$$\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}$$

This is the **differential form of Gauss's law**. It tells us that electric field lines **originate** on positive charges and **terminate** on negative charges. In charge-free regions, $\nabla \cdot \mathbf{E} = 0$ — the field is solenoidal there.

### Curl of E

For electrostatics (static charges, no time-varying magnetic fields):

$$\nabla \times \mathbf{E} = 0$$

A zero curl means the electric field is **conservative** — the work done moving a charge around any closed loop is zero:

$$\oint \mathbf{E} \cdot d\mathbf{l} = 0$$

This is what allows us to define a scalar potential $V$ such that $\mathbf{E} = -\nabla V$ (covered in the next lesson).

> **Analogy**: Think of the divergence as measuring whether a point is a "source" or "sink" of field lines — like a faucet (+) or drain (-) in a bathtub of water. The zero curl means the field never "swirls" — unlike a whirlpool, the electrostatic field only points radially toward or away from charges.

---

## Gauss's Law

Gauss's law is one of the four Maxwell equations and is the most powerful tool in electrostatics for problems with high symmetry.

### Integral Form

$$\oint_S \mathbf{E} \cdot d\mathbf{a} = \frac{Q_{\text{enc}}}{\epsilon_0}$$

The **electric flux** through any closed surface $S$ equals the enclosed charge divided by $\epsilon_0$.

**What this means physically**: No matter how complicated the charge distribution inside a closed surface, the total flux through that surface depends only on the total enclosed charge. Charges outside the surface contribute zero net flux.

### Differential Form

$$\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}$$

These two forms are equivalent — connected by the **divergence theorem**:

$$\oint_S \mathbf{E} \cdot d\mathbf{a} = \int_V (\nabla \cdot \mathbf{E}) \, d\tau = \int_V \frac{\rho}{\epsilon_0} \, d\tau = \frac{Q_{\text{enc}}}{\epsilon_0}$$

### Proof Sketch for a Point Charge

Consider a point charge $q$ at the origin and a spherical Gaussian surface of radius $r$ centered on it:

$$\oint \mathbf{E} \cdot d\mathbf{a} = \oint \frac{q}{4\pi\epsilon_0 r^2} \hat{r} \cdot r^2 \sin\theta \, d\theta \, d\phi \, \hat{r} = \frac{q}{4\pi\epsilon_0} \cdot 4\pi = \frac{q}{\epsilon_0}$$

Since any charge distribution is a superposition of point charges, and the integral is linear, Gauss's law extends to arbitrary distributions.

---

## Applications of Gauss's Law

Gauss's law is most useful when the charge distribution has enough symmetry that $\mathbf{E}$ can be pulled out of the flux integral. The three classic symmetries are:

### 1. Spherical Symmetry — Uniformly Charged Sphere

For a sphere of radius $R$ carrying total charge $Q$ uniformly distributed throughout its volume ($\rho = \frac{3Q}{4\pi R^3}$):

**Outside** ($r > R$): Choose a spherical Gaussian surface of radius $r$:

$$E(4\pi r^2) = \frac{Q}{\epsilon_0} \implies E = \frac{Q}{4\pi\epsilon_0 r^2}$$

The field is identical to that of a point charge — a beautiful result.

**Inside** ($r < R$): The enclosed charge is $Q_{\text{enc}} = Q\left(\frac{r}{R}\right)^3$:

$$E(4\pi r^2) = \frac{Q}{\epsilon_0}\left(\frac{r}{R}\right)^3 \implies E = \frac{Q r}{4\pi\epsilon_0 R^3}$$

The field grows linearly with $r$ inside the sphere.

### 2. Cylindrical Symmetry — Infinite Line Charge

An infinitely long line with uniform charge per unit length $\lambda$. Choose a cylindrical Gaussian surface of radius $s$ and length $L$:

$$E(2\pi s L) = \frac{\lambda L}{\epsilon_0} \implies E = \frac{\lambda}{2\pi\epsilon_0 s}$$

The field falls off as $1/s$ (not $1/s^2$!) — a signature of cylindrical geometry.

### 3. Planar Symmetry — Infinite Plane of Charge

An infinite plane with uniform surface charge density $\sigma$. A pillbox Gaussian surface gives:

$$2EA = \frac{\sigma A}{\epsilon_0} \implies E = \frac{\sigma}{2\epsilon_0}$$

The field is **uniform** — independent of distance from the plane. This is why parallel-plate capacitors produce nearly uniform fields between their plates.

```python
import numpy as np
import matplotlib.pyplot as plt

# Gauss's law applications: E vs distance for three geometries
# Why plot all three together: comparing their distance dependence is instructive

epsilon_0 = 8.854e-12

# --- Spherical: uniformly charged solid sphere ---
Q = 1e-9      # total charge (1 nC)
R_sphere = 0.1  # radius (10 cm)

r = np.linspace(0.001, 0.4, 500)
E_sphere = np.where(
    r < R_sphere,
    Q * r / (4 * np.pi * epsilon_0 * R_sphere**3),        # inside: linear
    Q / (4 * np.pi * epsilon_0 * r**2)                     # outside: 1/r^2
)

# --- Cylindrical: infinite line charge ---
lam = 1e-9     # linear charge density (1 nC/m)
s = r          # use same radial array
E_line = lam / (2 * np.pi * epsilon_0 * s)                 # 1/s dependence

# --- Planar: infinite sheet ---
sigma = 1e-9   # surface charge density (1 nC/m^2)
E_plane = sigma / (2 * epsilon_0) * np.ones_like(r)        # constant

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(r * 100, E_sphere, 'b-', linewidth=2)
axes[0].axvline(x=R_sphere*100, color='gray', linestyle='--', label=f'R={R_sphere*100} cm')
axes[0].set_xlabel('r (cm)')
axes[0].set_ylabel('E (V/m)')
axes[0].set_title('Sphere (Q = 1 nC, R = 10 cm)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(s * 100, E_line, 'r-', linewidth=2)
axes[1].set_xlabel('s (cm)')
axes[1].set_ylabel('E (V/m)')
axes[1].set_title(r'Line Charge ($\lambda$ = 1 nC/m)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(r * 100, E_plane, 'g-', linewidth=2)
axes[2].set_xlabel('distance (cm)')
axes[2].set_ylabel('E (V/m)')
axes[2].set_title(r'Plane ($\sigma$ = 1 nC/m$^2$)')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(0, E_plane[0] * 1.5)

plt.suptitle("Gauss's Law: Three Classic Symmetries", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('gauss_three_symmetries.png', dpi=150)
plt.show()
```

---

## The Field of a Dipole

A classic and important configuration is the **electric dipole**: two equal and opposite charges $\pm q$ separated by distance $d$. The **dipole moment** is:

$$\mathbf{p} = q\mathbf{d}$$

where $\mathbf{d}$ points from the negative to the positive charge.

Far from the dipole ($r \gg d$), the electric field in spherical coordinates is:

$$\mathbf{E}_{\text{dip}}(r, \theta) = \frac{p}{4\pi\epsilon_0 r^3}\left(2\cos\theta \, \hat{r} + \sin\theta \, \hat{\theta}\right)$$

Key features:
- The field falls off as $1/r^3$ (faster than a point charge's $1/r^2$)
- On the axis ($\theta = 0$): $\mathbf{E} = \frac{2p}{4\pi\epsilon_0 r^3}\hat{r}$
- On the equatorial plane ($\theta = \pi/2$): $\mathbf{E} = \frac{p}{4\pi\epsilon_0 r^3}\hat{\theta}$

The dipole is the leading term in the **multipole expansion** of any charge distribution whose total charge is zero.

### Torque and Energy of a Dipole in an External Field

A dipole $\mathbf{p}$ in an external field $\mathbf{E}$ experiences:

**Torque**: $\boldsymbol{\tau} = \mathbf{p} \times \mathbf{E}$ — tends to align $\mathbf{p}$ with $\mathbf{E}$

**Potential energy**: $U = -\mathbf{p} \cdot \mathbf{E}$

- Minimum energy when $\mathbf{p} \parallel \mathbf{E}$ (aligned)
- Maximum energy when $\mathbf{p}$ and $\mathbf{E}$ are anti-parallel

**Force** (in a non-uniform field): $\mathbf{F} = (\mathbf{p} \cdot \nabla)\mathbf{E}$

A dipole in a uniform field feels torque but no net force. Only a non-uniform field can exert a net translational force on a dipole — this is the basis of dielectrophoresis, used in microfluidics to manipulate biological cells.

---

## Boundary Conditions for E

At an interface carrying surface charge density $\sigma$, the electric field satisfies:

**Normal component is discontinuous**:
$$E_{\text{above}}^{\perp} - E_{\text{below}}^{\perp} = \frac{\sigma}{\epsilon_0}$$

**Tangential component is continuous**:
$$E_{\text{above}}^{\parallel} = E_{\text{below}}^{\parallel}$$

These boundary conditions follow from applying Gauss's law (for the normal component) and the vanishing curl of $\mathbf{E}$ (for the tangential component) to infinitesimally thin surfaces and loops.

Understanding these conditions is crucial for solving problems at interfaces — conductor surfaces, dielectric boundaries, and charge sheets — which we will explore extensively in Lesson 3.

---

## Numerical Gauss's Law Verification

We can numerically verify Gauss's law by computing the flux through a closed surface:

```python
import numpy as np

# Numerically verify Gauss's law: compute flux through a spherical surface
# Why numerical verification: builds confidence that the math works in code

epsilon_0 = 8.854e-12
k_e = 1 / (4 * np.pi * epsilon_0)

# Point charge at origin
q = 1e-9  # 1 nC

# Create a spherical Gaussian surface (radius = 0.2 m)
R = 0.2
N_theta = 200       # polar angle resolution
N_phi = 400         # azimuthal angle resolution

theta = np.linspace(0, np.pi, N_theta)
phi = np.linspace(0, 2 * np.pi, N_phi)
THETA, PHI = np.meshgrid(theta, phi)

# Surface element dA = R^2 sin(theta) dtheta dphi * r_hat
# Why R^2 sin(theta): this is the Jacobian for spherical coordinates
dtheta = theta[1] - theta[0]
dphi = phi[1] - phi[0]
dA = R**2 * np.sin(THETA) * dtheta * dphi

# E on the surface: for a point charge at origin, E_r = kq/R^2
E_r = k_e * q / R**2

# Total flux = sum of E_r * dA over the surface
flux_numerical = np.sum(E_r * dA)
flux_exact = q / epsilon_0

print(f"Numerical flux:  {flux_numerical:.6f} V·m")
print(f"Exact (q/ε₀):   {flux_exact:.6f} V·m")
print(f"Relative error:  {abs(flux_numerical - flux_exact)/flux_exact:.2e}")

# Verify with an off-center charge — charge OUTSIDE the surface
# Gauss's law predicts zero enclosed charge => zero net flux
q2_pos = np.array([0.5, 0.0, 0.0])  # charge at (0.5, 0, 0), outside R=0.2

# Points on the Gaussian surface
X_s = R * np.sin(THETA) * np.cos(PHI)
Y_s = R * np.sin(THETA) * np.sin(PHI)
Z_s = R * np.cos(THETA)

# Separation vectors from external charge to surface points
dx = X_s - q2_pos[0]
dy = Y_s - q2_pos[1]
dz = Z_s - q2_pos[2]
r_mag = np.sqrt(dx**2 + dy**2 + dz**2)

# E field at each surface point due to external charge
Ex = k_e * q * dx / r_mag**3
Ey = k_e * q * dy / r_mag**3
Ez = k_e * q * dz / r_mag**3

# Outward normal = (X_s, Y_s, Z_s)/R on a sphere centered at origin
# Why dot product with normal: flux = E · n dA
E_dot_n = (Ex * X_s + Ey * Y_s + Ez * Z_s) / R
flux_external = np.sum(E_dot_n * dA)

print(f"\nFlux from external charge: {flux_external:.6e} V·m  (should be ~0)")
```

---

## Summary

| Concept | Key Equation |
|---|---|
| Coulomb's law | $\mathbf{F} = \frac{1}{4\pi\epsilon_0}\frac{q_1 q_2}{r^2}\hat{r}$ |
| Electric field (point charge) | $\mathbf{E} = \frac{1}{4\pi\epsilon_0}\frac{q}{r^2}\hat{r}$ |
| Superposition | $\mathbf{E}_{\text{total}} = \sum_i \mathbf{E}_i$ |
| Gauss's law (integral) | $\oint \mathbf{E}\cdot d\mathbf{a} = Q_{\text{enc}}/\epsilon_0$ |
| Gauss's law (differential) | $\nabla \cdot \mathbf{E} = \rho/\epsilon_0$ |
| Curl of E (statics) | $\nabla \times \mathbf{E} = 0$ |
| Sphere ($r > R$) | $E = Q/(4\pi\epsilon_0 r^2)$ |
| Line charge | $E = \lambda/(2\pi\epsilon_0 s)$ |
| Infinite plane | $E = \sigma/(2\epsilon_0)$ |
| Dipole (far field) | $E \sim p/(4\pi\epsilon_0 r^3)$ |

---

## Exercises

### Exercise 1: Superposition Practice
Two charges $q_1 = +3\,\mu\text{C}$ at $(0, 0, 0)$ and $q_2 = -5\,\mu\text{C}$ at $(1, 0, 0)$ m. Find the electric field (magnitude and direction) at $(0.5, 0.5, 0)$ m. Compute both analytically and numerically in Python.

### Exercise 2: Gauss's Law — Spherical Shell
A thin spherical shell of radius $R$ carries total charge $Q$. Using Gauss's law, prove that:
- $\mathbf{E} = 0$ everywhere inside the shell
- $\mathbf{E} = \frac{Q}{4\pi\epsilon_0 r^2}\hat{r}$ outside the shell

Then write Python code to numerically verify the discontinuity at $r = R$.

### Exercise 3: Non-Uniform Charge Distribution
A solid sphere of radius $R$ has a charge density that varies as $\rho(r) = \rho_0(1 - r/R)$ for $r \leq R$. Use Gauss's law to find $\mathbf{E}(r)$ for both $r < R$ and $r > R$. Plot the result.

### Exercise 4: Numerical Flux Calculation
Place three point charges at arbitrary positions inside a cube. Numerically compute the electric flux through each face of the cube and verify that the total flux equals $Q_{\text{enc}}/\epsilon_0$.

### Exercise 5: Dipole Field Visualization
Write a Python program that computes and visualizes the electric field of a quadrupole (four charges: $+q, -q, -q, +q$ at the corners of a square). How does the far-field behavior differ from a dipole?

---

[Next: 02. Electric Potential and Energy →](02_Electric_Potential_and_Energy.md)
