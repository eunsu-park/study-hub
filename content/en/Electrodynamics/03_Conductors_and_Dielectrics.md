# Conductors and Dielectrics

[← Previous: 02. Electric Potential and Energy](02_Electric_Potential_and_Energy.md) | [Next: 04. Magnetostatics →](04_Magnetostatics.md)

---

## Learning Objectives

1. Describe the electrostatic properties of conductors and derive boundary conditions at conducting surfaces
2. Apply the method of image charges to solve problems with grounded conductors
3. Explain dielectric polarization and distinguish between free and bound charges
4. Define the displacement field $\mathbf{D}$ and relate it to $\mathbf{E}$ and $\mathbf{P}$
5. Derive and apply boundary conditions at a dielectric interface
6. Calculate capacitance for standard geometries and with dielectric filling
7. Solve image-charge and capacitance problems numerically in Python

---

Real electrostatic problems rarely involve charges floating in empty space. Charges live on conductors (metals, wires, electrodes) and fields pass through dielectrics (glass, plastic, biological tissue). Understanding how materials respond to electric fields is essential for everything from circuit design to understanding how cell membranes maintain voltage. This lesson introduces the two great classes of materials in electrostatics — conductors that redistribute charge freely, and dielectrics that polarize but do not conduct — and develops the mathematical tools to handle both.

---

## Conductors in Electrostatic Equilibrium

A conductor contains charges (typically electrons) that are free to move. When a conductor reaches electrostatic equilibrium:

1. **$\mathbf{E} = 0$ inside the conductor.** If there were a field, free charges would move until they cancelled it. Equilibrium demands zero internal field.

2. **$\rho = 0$ inside the conductor.** From Gauss's law: $\nabla \cdot \mathbf{E} = \rho/\epsilon_0 = 0$.

3. **All excess charge resides on the surface.** Since there is no volume charge, any net charge must sit on the conductor's surface.

4. **The conductor is an equipotential.** Since $\mathbf{E} = -\nabla V = 0$ inside, $V$ must be constant throughout.

5. **$\mathbf{E}$ just outside is perpendicular to the surface.** Any tangential component would cause surface charges to flow.

> **Analogy**: A conductor in an electric field is like a pool of water on a tilted table. The water (free charge) redistributes itself until its surface is perfectly level (equipotential). Just as water flows to eliminate any slope, charges flow to eliminate any tangential electric field.

### Surface Charge and the Field Just Outside

At the surface of a conductor carrying surface charge density $\sigma$:

$$\mathbf{E}_{\text{just outside}} = \frac{\sigma}{\epsilon_0} \hat{n}$$

where $\hat{n}$ points outward. This follows from a pillbox Gaussian surface straddling the conductor surface (the field inside is zero).

The **force per unit area** on the surface charge is:

$$\mathbf{f} = \frac{\sigma^2}{2\epsilon_0} \hat{n}$$

This is the **electrostatic pressure** — it always pushes outward, regardless of the sign of $\sigma$.

---

## Method of Image Charges

The method of images is a clever trick: we replace a complicated boundary-value problem (charge near a conductor) with a simpler one (charge plus a fictitious "image" charge) that satisfies the same boundary conditions. By uniqueness, this must be the correct solution.

### Point Charge Above a Grounded Plane

A charge $+q$ sits at height $d$ above an infinite grounded conducting plane ($V = 0$ on the plane).

**Image solution**: Remove the conductor and place an image charge $-q$ at distance $d$ below the plane (at the mirror position). The potential from the pair $+q$ and $-q$ automatically satisfies $V = 0$ on the plane.

The potential at point $(x, y, z)$ with $z > 0$:

$$V = \frac{q}{4\pi\epsilon_0}\left[\frac{1}{\sqrt{x^2 + y^2 + (z-d)^2}} - \frac{1}{\sqrt{x^2 + y^2 + (z+d)^2}}\right]$$

**Induced surface charge density** on the plane ($z = 0$):

$$\sigma = -\epsilon_0 \frac{\partial V}{\partial z}\bigg|_{z=0} = \frac{-qd}{2\pi(x^2 + y^2 + d^2)^{3/2}}$$

The total induced charge is $Q_{\text{induced}} = \int \sigma \, da = -q$, as expected.

**Force on the charge** toward the plane:

$$F = \frac{1}{4\pi\epsilon_0}\frac{q^2}{(2d)^2} = \frac{q^2}{16\pi\epsilon_0 d^2}$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Image charge method: point charge above a grounded conducting plane
# Why image charges: they replace complex boundary conditions with simple geometry

epsilon_0 = 8.854e-12
k_e = 1 / (4 * np.pi * epsilon_0)

q = 1e-9    # charge (1 nC)
d = 0.3     # height above plane (30 cm)

# Grid in the xz-plane (y=0)
x = np.linspace(-1, 1, 300)
z = np.linspace(-0.5, 1.0, 300)
X, Z = np.meshgrid(x, z)

# Real charge at (0, 0, d) and image charge at (0, 0, -d)
r_real = np.sqrt(X**2 + (Z - d)**2)
r_image = np.sqrt(X**2 + (Z + d)**2)
r_real = np.maximum(r_real, 1e-4)
r_image = np.maximum(r_image, 1e-4)

# Potential is valid only above the plane (z > 0)
V = k_e * q / r_real - k_e * q / r_image

# Mask the region below the conductor (z < 0) — V=0 there physically
V[Z < 0] = 0

# Why clip: avoid extreme values near the charge
V_clipped = np.clip(V, -200, 200)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Equipotential lines
levels = np.linspace(-150, 150, 31)
cs = axes[0].contour(X, Z, V_clipped, levels=levels, cmap='RdBu_r')
axes[0].axhline(y=0, color='gray', linewidth=3, label='Conductor')
axes[0].plot(0, d, 'ro', markersize=10, label=f'+q at z={d}')
axes[0].plot(0, -d, 'bx', markersize=10, label=f'-q (image)')
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('z (m)')
axes[0].set_title('Equipotentials')
axes[0].legend()
axes[0].set_aspect('equal')

# Induced surface charge density on the plane
x_surf = np.linspace(-1, 1, 500)
# Why this formula: σ = -qd / (2π(x²+d²)^(3/2)), derived from -ε₀ ∂V/∂z|_{z=0}
sigma = -q * d / (2 * np.pi * (x_surf**2 + d**2)**1.5)

axes[1].plot(x_surf * 100, sigma * 1e9, 'b-', linewidth=2)
axes[1].fill_between(x_surf * 100, sigma * 1e9, alpha=0.3)
axes[1].set_xlabel('x (cm)')
axes[1].set_ylabel(r'$\sigma$ (nC/m$^2$)')
axes[1].set_title('Induced Surface Charge Density')
axes[1].grid(True, alpha=0.3)

# Verify total induced charge equals -q
total_sigma = np.trapz(sigma * 2 * np.pi * np.abs(x_surf), x_surf)
axes[1].text(0.95, 0.95, f'Total induced charge: {total_sigma/q:.3f}q',
             transform=axes[1].transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat'))

plt.suptitle('Method of Image Charges', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('image_charges.png', dpi=150)
plt.show()
```

### Image Charge for a Grounded Sphere

A charge $q$ at distance $a$ from the center of a grounded sphere of radius $R$ ($a > R$). The image charge is:

$$q' = -\frac{R}{a}q, \quad \text{located at } b = \frac{R^2}{a} \text{ from center}$$

This is a non-trivial result — the image charge has a different magnitude and sits inside the sphere.

---

## Dielectrics

### Polarization

When an electric field is applied to a dielectric material, the atomic or molecular charges shift slightly, creating tiny dipoles. This is **polarization**. The polarization vector $\mathbf{P}$ is the dipole moment per unit volume:

$$\mathbf{P} = \frac{\text{dipole moment}}{\text{volume}} = n \langle \mathbf{p} \rangle$$

where $n$ is the number density of molecules and $\langle \mathbf{p} \rangle$ is the average molecular dipole moment.

### Bound Charges

Polarization produces effective charges called **bound charges**:

$$\rho_b = -\nabla \cdot \mathbf{P} \qquad \text{(volume bound charge)}$$
$$\sigma_b = \mathbf{P} \cdot \hat{n} \qquad \text{(surface bound charge)}$$

The physical picture: when dipoles align, their internal charges cancel in pairs, but at the surfaces and wherever $\mathbf{P}$ is non-uniform, uncancelled charge appears.

> **Analogy**: Imagine a long line of people holding hands. Each person is electrically neutral, but when they all lean slightly to the right (polarize), the rightmost person's right hand has no partner — it sticks out like an uncompensated surface charge. Inside the line, every left hand holds a right hand, so there is no net charge.

### The Displacement Field D

Total charge = free charge + bound charge: $\rho = \rho_f + \rho_b$. Gauss's law becomes:

$$\nabla \cdot \mathbf{E} = \frac{\rho_f + \rho_b}{\epsilon_0} = \frac{\rho_f - \nabla \cdot \mathbf{P}}{\epsilon_0}$$

Define the **displacement field**:

$$\mathbf{D} = \epsilon_0 \mathbf{E} + \mathbf{P}$$

Then Gauss's law for $\mathbf{D}$ involves only free charges:

$$\nabla \cdot \mathbf{D} = \rho_f$$
$$\oint \mathbf{D} \cdot d\mathbf{a} = Q_{f,\text{enc}}$$

### Linear Dielectrics

For most materials (at ordinary field strengths), $\mathbf{P}$ is proportional to $\mathbf{E}$:

$$\mathbf{P} = \epsilon_0 \chi_e \mathbf{E}$$

where $\chi_e$ is the **electric susceptibility**. Then:

$$\mathbf{D} = \epsilon_0(1 + \chi_e)\mathbf{E} = \epsilon \mathbf{E}$$

where $\epsilon = \epsilon_0 \epsilon_r$ is the **permittivity** and $\epsilon_r = 1 + \chi_e$ is the **relative permittivity** (dielectric constant).

| Material | $\epsilon_r$ |
|---|---|
| Vacuum | 1 |
| Air | 1.0006 |
| Paper | 3.7 |
| Glass | 4-10 |
| Silicon | 11.7 |
| Water | 80 |
| Barium titanate | ~1000-10000 |

---

## Boundary Conditions at a Dielectric Interface

At the interface between two dielectrics ($\epsilon_1$ and $\epsilon_2$), with free surface charge $\sigma_f$:

**Normal component** (from a pillbox):
$$D_1^{\perp} - D_2^{\perp} = \sigma_f$$
$$\epsilon_1 E_1^{\perp} - \epsilon_2 E_2^{\perp} = \sigma_f$$

If there is no free surface charge ($\sigma_f = 0$):
$$\epsilon_1 E_1^{\perp} = \epsilon_2 E_2^{\perp}$$

**Tangential component** (from a loop, since $\nabla \times \mathbf{E} = 0$):
$$E_1^{\parallel} = E_2^{\parallel}$$

These conditions determine how field lines bend at a dielectric interface:

$$\frac{\tan\theta_1}{\tan\theta_2} = \frac{\epsilon_1}{\epsilon_2}$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstrate field line bending at a dielectric interface
# Why this visualization: seeing refraction of E-field lines builds intuition

epsilon_r1 = 1.0    # air
epsilon_r2 = 4.0    # glass

# Incident angles (measured from the normal)
theta1 = np.linspace(5, 85, 100)
theta1_rad = np.radians(theta1)

# Snell's-law analogue for E-field: tan(θ₂)/tan(θ₁) = ε₂/ε₁
# Why tangent (not sine): this comes from continuity of E∥ and ε E⊥
tan_theta2 = (epsilon_r2 / epsilon_r1) * np.tan(theta1_rad)
theta2_rad = np.arctan(tan_theta2)
theta2 = np.degrees(theta2_rad)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot refraction angles
axes[0].plot(theta1, theta2, 'b-', linewidth=2)
axes[0].plot(theta1, theta1, 'k--', alpha=0.5, label='No refraction')
axes[0].set_xlabel(r'$\theta_1$ (degrees) — angle in medium 1')
axes[0].set_ylabel(r'$\theta_2$ (degrees) — angle in medium 2')
axes[0].set_title(f'E-field Refraction (εᵣ₁={epsilon_r1}, εᵣ₂={epsilon_r2})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Draw field lines crossing the interface
axes[1].axhline(y=0, color='gray', linewidth=3, label='Interface')
axes[1].fill_between([-2, 2], [-2, -2], [0, 0], alpha=0.1, color='blue',
                      label=f'Medium 2 (εᵣ={epsilon_r2})')
axes[1].fill_between([-2, 2], [0, 0], [2, 2], alpha=0.1, color='yellow',
                      label=f'Medium 1 (εᵣ={epsilon_r1})')

# Draw a few representative field lines
for theta1_deg in [20, 40, 60]:
    t1 = np.radians(theta1_deg)
    t2 = np.arctan((epsilon_r2 / epsilon_r1) * np.tan(t1))

    # Line in medium 1 (above interface)
    L = 1.5
    x_start = -L * np.sin(t1)
    y_start = L * np.cos(t1)
    axes[1].annotate('', xy=(0, 0), xytext=(x_start, y_start),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Line in medium 2 (below interface)
    x_end = L * np.sin(t2)
    y_end = -L * np.cos(t2)
    axes[1].annotate('', xy=(x_end, y_end), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))

    axes[1].text(x_start - 0.1, y_start, f'{theta1_deg}°', fontsize=9)

axes[1].set_xlim(-2, 2)
axes[1].set_ylim(-2, 2)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Field Lines at Dielectric Interface')
axes[1].legend(loc='lower right')
axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig('dielectric_refraction.png', dpi=150)
plt.show()
```

---

## Capacitance

A **capacitor** is a device that stores charge and energy in the electric field between conductors. For two conductors carrying charges $+Q$ and $-Q$:

$$C = \frac{Q}{V} \qquad [F = C/V]$$

where $V = V_+ - V_-$ is the potential difference.

### Standard Geometries

**Parallel plates** (area $A$, separation $d$):
$$C = \frac{\epsilon_0 A}{d}$$

**Coaxial cylinders** (inner radius $a$, outer radius $b$, length $L$):
$$C = \frac{2\pi\epsilon_0 L}{\ln(b/a)}$$

**Concentric spheres** (inner radius $a$, outer radius $b$):
$$C = 4\pi\epsilon_0 \frac{ab}{b-a}$$

### Effect of Dielectrics

Filling a capacitor with a dielectric of constant $\epsilon_r$ multiplies the capacitance by $\epsilon_r$:

$$C_{\text{dielectric}} = \epsilon_r C_{\text{vacuum}}$$

This is the main reason dielectrics are used — they increase capacitance while maintaining insulation.

### Energy Stored in a Capacitor

$$W = \frac{1}{2}CV^2 = \frac{Q^2}{2C} = \frac{1}{2}QV$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Capacitance calculations for three standard geometries
# Why compare geometries: each has different scaling behavior

epsilon_0 = 8.854e-12

# --- Parallel Plate ---
A = 0.01                  # plate area (100 cm²)
d = np.linspace(0.001, 0.01, 100)  # separation (1 mm to 1 cm)
C_pp = epsilon_0 * A / d

# --- Coaxial Cable ---
a = 0.001                 # inner radius (1 mm)
b = np.linspace(0.002, 0.02, 100)  # outer radius
L = 1.0                   # length (1 m)
C_coax = 2 * np.pi * epsilon_0 * L / np.log(b / a)

# --- Concentric Spheres ---
a_s = 0.05                # inner radius (5 cm)
b_s = np.linspace(0.06, 0.5, 100)  # outer radius
C_sphere = 4 * np.pi * epsilon_0 * a_s * b_s / (b_s - a_s)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(d * 1000, C_pp * 1e12, 'b-', linewidth=2)
axes[0].set_xlabel('Separation d (mm)')
axes[0].set_ylabel('C (pF)')
axes[0].set_title('Parallel Plate')
axes[0].grid(True, alpha=0.3)

axes[1].plot(b * 1000, C_coax * 1e12, 'r-', linewidth=2)
axes[1].set_xlabel('Outer radius b (mm)')
axes[1].set_ylabel('C (pF)')
axes[1].set_title(f'Coaxial (a={a*1000:.0f} mm, L={L} m)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(b_s * 100, C_sphere * 1e12, 'g-', linewidth=2)
axes[2].set_xlabel('Outer radius b (cm)')
axes[2].set_ylabel('C (pF)')
axes[2].set_title(f'Concentric Spheres (a={a_s*100:.0f} cm)')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Capacitance of Standard Geometries', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('capacitance_geometries.png', dpi=150)
plt.show()

# Dielectric effect on parallel plate capacitor
print("Effect of dielectric on parallel plate capacitor:")
print(f"{'Material':<20} {'εᵣ':>6} {'C (pF)':>10}")
print("-" * 38)
d_fixed = 0.001  # 1 mm gap
for name, eps_r in [('Vacuum', 1), ('Paper', 3.7), ('Glass', 5),
                     ('Silicon', 11.7), ('Water', 80)]:
    C = eps_r * epsilon_0 * A / d_fixed
    print(f"{name:<20} {eps_r:>6.1f} {C*1e12:>10.2f}")
```

---

## Combinations of Capacitors

Capacitors can be combined in series and parallel configurations:

**Parallel** (same voltage across each):
$$C_{\text{total}} = C_1 + C_2 + C_3 + \cdots$$

**Series** (same charge on each):
$$\frac{1}{C_{\text{total}}} = \frac{1}{C_1} + \frac{1}{C_2} + \frac{1}{C_3} + \cdots$$

Note the pattern: capacitors combine oppositely to resistors. Parallel capacitors add directly (like series resistors), and series capacitors add reciprocally (like parallel resistors).

### Why?

In parallel, each capacitor independently stores charge from the same voltage source, so the total stored charge (and hence total capacitance) is the sum.

In series, the same charge $Q$ appears on each capacitor (by conservation of charge on the isolated conductors between them), but the voltages add: $V_{\text{total}} = V_1 + V_2 + \cdots = Q/C_1 + Q/C_2 + \cdots$

---

## Force on a Dielectric

A dielectric slab is pulled into a parallel-plate capacitor by the fringing fields at the edge. For a capacitor with plates of width $w$, separation $d$, and a dielectric slab of thickness $d$ and permittivity $\epsilon_r$ inserted a distance $x$:

At constant voltage $V$:

$$F = \frac{\epsilon_0(\epsilon_r - 1)wV^2}{2d}$$

The force is independent of $x$ — it is constant as the slab slides in. This is because the energy change per unit displacement is constant.

---

## Capacitors in Practice

### Energy Density in a Capacitor

Between the plates of a parallel-plate capacitor with field $E = V/d$:

$$u = \frac{1}{2}\epsilon_0 E^2 = \frac{1}{2}\epsilon_0\frac{V^2}{d^2}$$

For a 1 mm gap at 1000 V: $u \approx 4.4 \times 10^{-3}$ J/m$^3$. With a dielectric ($\epsilon_r = 1000$, as in barium titanate): $u \approx 4.4$ J/m$^3$.

### Dielectric Breakdown

Every dielectric has a maximum field strength it can withstand before it breaks down (becomes conducting). Typical values:

| Material | Breakdown field (MV/m) |
|---|---|
| Air | 3 |
| Paper | 16 |
| Glass | 10-40 |
| Teflon | 60 |
| Mica | 100-200 |

This limits the maximum voltage across a capacitor. For a 1 mm air gap: $V_{\max} = 3 \times 10^6 \times 10^{-3} = 3000$ V.

The maximum energy density is determined by the dielectric constant AND the breakdown field — materials with both high $\epsilon_r$ and high breakdown field are prized for compact energy storage.

---

## Summary

| Concept | Key Equation |
|---|---|
| E inside conductor | $\mathbf{E} = 0$ |
| Surface charge field | $E = \sigma/\epsilon_0$ |
| Image charge (plane) | $q' = -q$ at mirror position |
| Image charge (sphere) | $q' = -(R/a)q$ at $b = R^2/a$ |
| Polarization | $\mathbf{P} = \epsilon_0 \chi_e \mathbf{E}$ |
| Bound charges | $\rho_b = -\nabla \cdot \mathbf{P}$, $\sigma_b = \mathbf{P}\cdot\hat{n}$ |
| Displacement field | $\mathbf{D} = \epsilon_0\mathbf{E} + \mathbf{P} = \epsilon\mathbf{E}$ |
| Gauss's law for D | $\nabla \cdot \mathbf{D} = \rho_f$ |
| BC (normal) | $\epsilon_1 E_1^\perp - \epsilon_2 E_2^\perp = \sigma_f$ |
| BC (tangential) | $E_1^\parallel = E_2^\parallel$ |
| Parallel plate C | $C = \epsilon_0 A/d$ |
| Capacitor energy | $W = \frac{1}{2}CV^2$ |

---

## Exercises

### Exercise 1: Image Charge — Grounded Sphere
A charge $q = 5$ nC is located at distance $a = 0.5$ m from the center of a grounded conducting sphere of radius $R = 0.2$ m. Find the image charge magnitude and position. Compute and plot the potential in the plane containing both charges and the sphere center.

### Exercise 2: Dielectric Sphere
A dielectric sphere of radius $R$ and dielectric constant $\epsilon_r$ is placed in a uniform external field $\mathbf{E}_0 = E_0 \hat{z}$. Show that the field inside the sphere is uniform and find its magnitude: $E_{\text{inside}} = \frac{3}{\epsilon_r + 2}E_0$. Plot the field lines inside and outside.

### Exercise 3: Multi-Layer Capacitor
Three dielectric slabs of thicknesses $d_1, d_2, d_3$ and dielectric constants $\epsilon_1, \epsilon_2, \epsilon_3$ are stacked between the plates of a parallel-plate capacitor. Derive the effective capacitance. Verify numerically for $d_1=d_2=d_3=1$ mm and $\epsilon_1=2, \epsilon_2=5, \epsilon_3=10$.

### Exercise 4: Energy of a Charged Conductor
A conducting sphere of radius $R$ carries total charge $Q$. Compute the electrostatic energy stored in the field. How does this compare to the energy of a uniformly charged insulating sphere of the same $R$ and $Q$?

### Exercise 5: Polarization Bound Charges
A dielectric cylinder of radius $R$ and length $L$ has polarization $\mathbf{P} = P_0 \hat{z}$ (uniform). Compute the bound volume and surface charges. Show that the total bound charge is zero. Sketch the bound charge distribution.

---

[← Previous: 02. Electric Potential and Energy](02_Electric_Potential_and_Energy.md) | [Next: 04. Magnetostatics →](04_Magnetostatics.md)
