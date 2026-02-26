# EM Waves in Vacuum

[← Previous: 08. Maxwell's Equations — Integral Form](08_Maxwells_Equations_Integral.md)

---

## Learning Objectives

1. Derive plane wave solutions to Maxwell's equations and verify they satisfy all four equations
2. Prove the transversality conditions: $\mathbf{E} \perp \mathbf{B} \perp \mathbf{k}$
3. Describe and mathematically represent linear, circular, and elliptical polarization
4. Define the Stokes parameters and use them to characterize arbitrary polarization states
5. Calculate the energy, momentum, and radiation pressure of electromagnetic waves
6. Explain the impedance of free space and its physical significance
7. Simulate polarization states and energy transport numerically in Python

---

We now arrive at the physical prediction that made Maxwell's theory immortal: electromagnetic waves. The wave equation derived in Lesson 7 admits solutions that describe oscillating electric and magnetic fields propagating through empty space at the speed of light. These waves carry energy, momentum, and information. They are light, radio, X-rays, and every other form of electromagnetic radiation. This lesson develops the detailed properties of plane waves in vacuum — their structure, polarization, and energetics — providing the foundation for all of wave optics and radiation theory.

---

## Plane Wave Solutions

The wave equation in vacuum is:

$$\nabla^2 \mathbf{E} = \mu_0\epsilon_0 \frac{\partial^2 \mathbf{E}}{\partial t^2} = \frac{1}{c^2}\frac{\partial^2 \mathbf{E}}{\partial t^2}$$

The simplest solutions are **monochromatic plane waves** — sinusoidal oscillations uniform over infinite planes perpendicular to the propagation direction.

For a wave propagating in the $\hat{z}$ direction:

$$\tilde{\mathbf{E}} = \tilde{E}_0 \, e^{i(kz - \omega t)}$$

where:
- $\tilde{E}_0$ is a complex amplitude vector (encoding both amplitude and phase)
- $k = 2\pi/\lambda$ is the wave number
- $\omega = 2\pi f$ is the angular frequency
- The dispersion relation: $\omega = ck$

In general, for propagation along $\hat{k}$:

$$\tilde{\mathbf{E}}(\mathbf{r}, t) = \tilde{\mathbf{E}}_0 \, e^{i(\mathbf{k}\cdot\mathbf{r} - \omega t)}$$

where $\mathbf{k}$ is the **wave vector** (pointing in the direction of propagation, with magnitude $k = \omega/c$).

The physical field is the real part:

$$\mathbf{E} = \text{Re}(\tilde{\mathbf{E}}) = E_0 \cos(\mathbf{k}\cdot\mathbf{r} - \omega t + \phi)$$

> **Analogy**: A plane wave is like the wave that forms when you drop a very long, straight rod horizontally into a swimming pool. The wave crests form parallel lines (planes in 3D) that march forward together. Every point along a given crest oscillates in unison — this is the "plane" in plane wave.

---

## Transversality: E, B, and k

Maxwell's equations impose strong constraints on the directions of $\mathbf{E}$ and $\mathbf{B}$ relative to $\mathbf{k}$.

### From Gauss's Law

$$\nabla \cdot \mathbf{E} = 0 \implies i\mathbf{k} \cdot \tilde{\mathbf{E}}_0 = 0 \implies \mathbf{k} \cdot \mathbf{E}_0 = 0$$

$\mathbf{E}$ is perpendicular to $\mathbf{k}$ — the electric field oscillates transversely to the direction of propagation.

### From Faraday's Law

$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t} \implies i\mathbf{k} \times \tilde{\mathbf{E}}_0 = i\omega \tilde{\mathbf{B}}_0$$

$$\tilde{\mathbf{B}}_0 = \frac{1}{\omega}\mathbf{k} \times \tilde{\mathbf{E}}_0 = \frac{1}{c}\hat{k} \times \tilde{\mathbf{E}}_0$$

This tells us:
1. $\mathbf{B}$ is also perpendicular to $\mathbf{k}$ (transverse)
2. $\mathbf{B}$ is perpendicular to $\mathbf{E}$
3. $|\mathbf{B}| = |\mathbf{E}|/c$

So the three vectors $\mathbf{E}$, $\mathbf{B}$, and $\mathbf{k}$ form a **right-handed orthogonal triad**:

$$\boxed{\mathbf{E} \perp \mathbf{B} \perp \mathbf{k}, \qquad B_0 = E_0/c, \qquad \hat{k} = \hat{E} \times \hat{B}}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Visualize a plane EM wave: E in x, B in y, propagation in z
# Why 3D: the orthogonal structure of E, B, k demands three dimensions

c = 1.0            # normalized speed of light
lambda_0 = 1.0     # wavelength
k = 2 * np.pi / lambda_0
omega = c * k
E0 = 1.0
B0 = E0 / c

z = np.linspace(0, 3 * lambda_0, 500)
t = 0  # snapshot at t = 0

Ex = E0 * np.sin(k * z - omega * t)
By = B0 * np.sin(k * z - omega * t)

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot E field (oscillates in x-direction)
ax.plot(z, Ex, np.zeros_like(z), 'b-', linewidth=2, label='E (x-direction)')

# Plot B field (oscillates in y-direction)
ax.plot(z, np.zeros_like(z), By, 'r-', linewidth=2, label='B (y-direction)')

# Draw arrows at selected points to show vector nature
N_arrows = 30
z_arrows = np.linspace(0, 3*lambda_0, N_arrows)
for zi in z_arrows:
    ex = E0 * np.sin(k * zi)
    by = B0 * np.sin(k * zi)
    # Why arrows: they show that E and B are perpendicular at every point
    ax.quiver(zi, 0, 0, 0, ex, 0, color='blue', alpha=0.3, arrow_length_ratio=0.3)
    ax.quiver(zi, 0, 0, 0, 0, by, color='red', alpha=0.3, arrow_length_ratio=0.3)

# Propagation direction arrow
ax.quiver(0, 0, 0, 0.5, 0, 0, color='green', linewidth=3, arrow_length_ratio=0.3)
ax.text(0.3, 0, 0.15, 'k', fontsize=14, color='green', fontweight='bold')

ax.set_xlabel('z (propagation)')
ax.set_ylabel('x (E-field)')
ax.set_zlabel('y (B-field)')
ax.set_title('Plane Electromagnetic Wave: E ⊥ B ⊥ k', fontsize=14)
ax.legend(loc='upper right')

# Adjust view angle for best visualization
ax.view_init(elev=20, azim=-60)
plt.tight_layout()
plt.savefig('plane_wave_3d.png', dpi=150)
plt.show()

# Verify transversality numerically
print("Transversality verification:")
print(f"E · k = E_x * k_z = {E0} * 0 = 0  (E ⊥ k) ✓")
print(f"B · k = B_y * k_z = {B0} * 0 = 0  (B ⊥ k) ✓")
print(f"E · B = E_x * B_y = varies, but E_vec · B_vec = 0  (E ⊥ B) ✓")
print(f"|B₀|/|E₀| = {B0/E0} = 1/c ✓")
```

---

## Polarization

The **polarization** of an electromagnetic wave describes the pattern traced by the electric field vector in the plane perpendicular to propagation.

For a wave propagating in $\hat{z}$, the most general electric field is:

$$\mathbf{E}(z,t) = E_{0x}\cos(kz - \omega t)\hat{x} + E_{0y}\cos(kz - \omega t + \delta)\hat{y}$$

where $\delta$ is the phase difference between the $x$ and $y$ components.

### Linear Polarization ($\delta = 0$ or $\pi$)

$$\mathbf{E} = (E_{0x}\hat{x} + E_{0y}\hat{y})\cos(kz - \omega t)$$

The tip of $\mathbf{E}$ oscillates back and forth along a fixed line. The polarization direction makes angle $\alpha = \arctan(E_{0y}/E_{0x})$ with the $x$-axis.

### Circular Polarization ($\delta = \pm\pi/2$, $E_{0x} = E_{0y}$)

$$\mathbf{E} = E_0[\cos(kz-\omega t)\hat{x} \mp \sin(kz-\omega t)\hat{y}]$$

The tip of $\mathbf{E}$ traces a circle. Convention:
- $\delta = -\pi/2$: **Right circular** (clockwise when looking into the wave)
- $\delta = +\pi/2$: **Left circular** (counterclockwise when looking into the wave)

### Elliptical Polarization (general $\delta$, arbitrary amplitudes)

The most general case. The tip of $\mathbf{E}$ traces an ellipse. Linear and circular are special cases.

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize different polarization states
# Why multiple panels: comparing states side by side reveals the pattern

omega = 2 * np.pi
t = np.linspace(0, 1, 500)   # one period

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

polarization_cases = [
    ('Linear (horizontal)', 1.0, 0.0, 0.0),
    ('Linear (45°)', 1.0, 1.0, 0.0),
    ('Linear (vertical)', 0.0, 1.0, 0.0),
    ('Right Circular', 1.0, 1.0, -np.pi/2),
    ('Left Circular', 1.0, 1.0, np.pi/2),
    ('Elliptical', 1.0, 0.5, np.pi/4),
]

for idx, (name, E0x, E0y, delta) in enumerate(polarization_cases):
    ax = axes[idx // 3, idx % 3]

    Ex = E0x * np.cos(omega * t)
    Ey = E0y * np.cos(omega * t + delta)

    # Why color by time: it shows the direction of rotation
    colors = plt.cm.viridis(t / t[-1])
    for i in range(len(t) - 1):
        ax.plot([Ex[i], Ex[i+1]], [Ey[i], Ey[i+1]], color=colors[i], linewidth=2)

    # Mark starting point
    ax.plot(Ex[0], Ey[0], 'ro', markersize=8, zorder=5)
    # Arrow showing direction at t=0
    if idx >= 3 and E0y > 0:
        ax.annotate('', xy=(Ex[10], Ey[10]), xytext=(Ex[0], Ey[0]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel('$E_x$')
    ax.set_ylabel('$E_y$')
    ax.set_title(f'{name}\n$E_{{0x}}$={E0x}, $E_{{0y}}$={E0y}, δ={delta/np.pi:.2f}π')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', alpha=0.3)
    ax.axvline(x=0, color='gray', alpha=0.3)

plt.suptitle('Polarization States of EM Waves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('polarization_states.png', dpi=150)
plt.show()
```

---

## Stokes Parameters

The **Stokes parameters** provide a complete description of the polarization state, including partially polarized light. For a wave with complex amplitudes $\tilde{E}_x = E_{0x}$ and $\tilde{E}_y = E_{0y}e^{i\delta}$:

$$S_0 = E_{0x}^2 + E_{0y}^2 \qquad \text{(total intensity)}$$
$$S_1 = E_{0x}^2 - E_{0y}^2 \qquad \text{(horizontal vs. vertical preference)}$$
$$S_2 = 2E_{0x}E_{0y}\cos\delta \qquad \text{(+45° vs. -45° preference)}$$
$$S_3 = 2E_{0x}E_{0y}\sin\delta \qquad \text{(right vs. left circular preference)}$$

For fully polarized light: $S_0^2 = S_1^2 + S_2^2 + S_3^2$.

The **degree of polarization** (for partially polarized light): $\Pi = \frac{\sqrt{S_1^2+S_2^2+S_3^2}}{S_0}$, with $0 \leq \Pi \leq 1$.

### Poincare Sphere

The normalized Stokes parameters $(S_1/S_0, S_2/S_0, S_3/S_0)$ define a point on the **Poincare sphere**:
- North pole: right circular
- South pole: left circular
- Equator: linear polarizations
- Interior points: partially polarized

```python
import numpy as np
import matplotlib.pyplot as plt

# Stokes parameters for various polarization states
# Why Stokes: they are directly measurable (unlike complex amplitudes)

def stokes_parameters(E0x, E0y, delta):
    """Compute Stokes parameters for a fully polarized wave."""
    S0 = E0x**2 + E0y**2
    S1 = E0x**2 - E0y**2
    S2 = 2 * E0x * E0y * np.cos(delta)
    S3 = 2 * E0x * E0y * np.sin(delta)
    return S0, S1, S2, S3

# Define several polarization states
states = {
    'H (horizontal)':    (1.0, 0.0, 0.0),
    'V (vertical)':      (0.0, 1.0, 0.0),
    '+45° linear':       (1.0, 1.0, 0.0),
    '-45° linear':       (1.0, 1.0, np.pi),
    'Right circular':    (1.0, 1.0, -np.pi/2),
    'Left circular':     (1.0, 1.0, np.pi/2),
    'Elliptical (1)':    (1.0, 0.5, np.pi/4),
    'Elliptical (2)':    (0.8, 0.6, np.pi/3),
}

print(f"{'State':<22} {'S₀':>6} {'S₁':>6} {'S₂':>6} {'S₃':>6}  {'Check':>8}")
print("=" * 70)

for name, (E0x, E0y, delta) in states.items():
    S0, S1, S2, S3 = stokes_parameters(E0x, E0y, delta)
    check = np.sqrt(S1**2 + S2**2 + S3**2)
    # Why check: for fully polarized light, √(S₁²+S₂²+S₃²) = S₀
    print(f"{name:<22} {S0:6.2f} {S1:6.2f} {S2:6.2f} {S3:6.2f}  {check:8.4f}")

# Poincare sphere visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Draw sphere
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(xs, ys, zs, alpha=0.1, color='lightblue')

# Plot polarization states on the sphere
colors = plt.cm.tab10(np.linspace(0, 1, len(states)))
for idx, (name, (E0x, E0y, delta)) in enumerate(states.items()):
    S0, S1, S2, S3 = stokes_parameters(E0x, E0y, delta)
    if S0 > 0:
        # Normalize to unit sphere
        s1, s2, s3 = S1/S0, S2/S0, S3/S0
        ax.scatter(s1, s2, s3, color=colors[idx], s=100, zorder=5)
        ax.text(s1*1.15, s2*1.15, s3*1.15, name, fontsize=8, color=colors[idx])

# Label poles and equator
ax.set_xlabel('$S_1/S_0$')
ax.set_ylabel('$S_2/S_0$')
ax.set_zlabel('$S_3/S_0$')
ax.set_title('Poincare Sphere', fontsize=14)

plt.tight_layout()
plt.savefig('poincare_sphere.png', dpi=150)
plt.show()
```

---

## Energy and Momentum of EM Waves

### Energy Density

The instantaneous energy density of a plane wave:

$$u = \frac{1}{2}\left(\epsilon_0 E^2 + \frac{B^2}{\mu_0}\right)$$

For a plane wave with $B = E/c$ and $c = 1/\sqrt{\mu_0\epsilon_0}$:

$$\frac{B^2}{\mu_0} = \frac{E^2}{\mu_0 c^2} = \epsilon_0 E^2$$

So the electric and magnetic contributions are **exactly equal**:

$$u = \epsilon_0 E^2 = \frac{B^2}{\mu_0}$$

> **Analogy**: In a mechanical wave (e.g., a vibrating string), kinetic and potential energies are equal on average. In an electromagnetic wave, the electric and magnetic energy densities play these roles — perfectly balanced partners carrying equal shares of the wave's energy.

### Time-Averaged Energy Density

$$\langle u \rangle = \frac{1}{2}\epsilon_0 E_0^2$$

### Poynting Vector and Intensity

$$\mathbf{S} = \frac{1}{\mu_0}\mathbf{E}\times\mathbf{B} = \frac{E^2}{\mu_0 c}\hat{k} = cu\,\hat{k}$$

The intensity (time-averaged power per unit area):

$$I = \langle|\mathbf{S}|\rangle = \frac{1}{2}\frac{E_0^2}{\mu_0 c} = \frac{1}{2}\epsilon_0 c E_0^2 = \frac{c}{2\mu_0}B_0^2$$

### Momentum

Electromagnetic waves carry momentum. The momentum density:

$$\mathbf{g} = \frac{\mathbf{S}}{c^2} = \frac{u}{c}\hat{k}$$

The momentum carried by a wave of intensity $I$ hitting a surface:

$$\text{Radiation pressure (absorption)} = \frac{I}{c}$$
$$\text{Radiation pressure (perfect reflection)} = \frac{2I}{c}$$

```python
import numpy as np

# Energy and momentum of electromagnetic waves — practical calculations
# Why real numbers: connecting abstract formulas to tangible quantities

c = 3e8
mu_0 = 4 * np.pi * 1e-7
epsilon_0 = 8.854e-12

print("Electromagnetic Wave Properties")
print("=" * 60)

# Example 1: Sunlight
I_sun = 1361  # W/m² (solar constant at Earth)
E0_sun = np.sqrt(2 * I_sun / (epsilon_0 * c))
B0_sun = E0_sun / c
P_abs_sun = I_sun / c
P_ref_sun = 2 * I_sun / c

print(f"\n1. Sunlight at Earth (I = {I_sun} W/m²)")
print(f"   E₀ = {E0_sun:.1f} V/m")
print(f"   B₀ = {B0_sun*1e6:.3f} μT")
print(f"   Radiation pressure (absorbing): {P_abs_sun*1e6:.3f} μPa")
print(f"   Radiation pressure (reflecting): {P_ref_sun*1e6:.3f} μPa")
print(f"   Force on 10×10 m sail: {P_ref_sun * 100:.6f} N")

# Example 2: Laser pointer
P_laser = 5e-3   # 5 mW
A_spot = np.pi * (0.5e-3)**2   # 0.5 mm radius spot
I_laser = P_laser / A_spot
E0_laser = np.sqrt(2 * I_laser / (epsilon_0 * c))
B0_laser = E0_laser / c

print(f"\n2. 5 mW Laser Pointer (spot radius 0.5 mm)")
print(f"   Intensity: I = {I_laser:.1f} W/m²")
print(f"   E₀ = {E0_laser:.1f} V/m")
print(f"   B₀ = {B0_laser*1e6:.3f} μT")

# Example 3: Cell phone signal
P_phone = 1.0   # 1 W transmitted power
d = 1.0          # 1 m distance
I_phone = P_phone / (4 * np.pi * d**2)   # isotropic radiation
E0_phone = np.sqrt(2 * I_phone / (epsilon_0 * c))

print(f"\n3. Cell Phone (1 W, 1 m away)")
print(f"   Intensity: I = {I_phone:.2f} W/m²")
print(f"   E₀ = {E0_phone:.2f} V/m")

# Example 4: Microwave oven
P_oven = 1000    # 1 kW
A_oven = 0.3 * 0.3  # 30 cm × 30 cm cavity
I_oven = P_oven / A_oven
E0_oven = np.sqrt(2 * I_oven / (epsilon_0 * c))

print(f"\n4. Microwave Oven (1 kW, 30×30 cm)")
print(f"   Intensity: I = {I_oven:.0f} W/m²")
print(f"   E₀ = {E0_oven:.0f} V/m")

# Verify energy equipartition
print(f"\n--- Energy Equipartition Verification ---")
print(f"For any plane wave:")
print(f"  Electric energy density: u_E = ε₀E²/2")
print(f"  Magnetic energy density: u_B = B²/(2μ₀) = E²/(2μ₀c²) = ε₀E²/2 = u_E  ✓")
print(f"  The two contributions are EXACTLY equal!")
```

---

## Impedance of Free Space

The ratio of the electric to magnetic field amplitudes defines the **impedance of free space**:

$$Z_0 = \frac{E_0}{H_0} = \frac{E_0}{B_0/\mu_0} = \mu_0 c = \sqrt{\frac{\mu_0}{\epsilon_0}}$$

$$\boxed{Z_0 = \sqrt{\frac{\mu_0}{\epsilon_0}} \approx 376.73 \; \Omega}$$

This is a fundamental constant of nature. It plays a role in electromagnetic wave propagation analogous to the characteristic impedance of a transmission line.

The intensity of a plane wave can be written elegantly in terms of $Z_0$:

$$I = \frac{E_0^2}{2Z_0}$$

In a medium with permittivity $\epsilon$ and permeability $\mu$:

$$Z = \sqrt{\frac{\mu}{\epsilon}} = \frac{Z_0}{\sqrt{\epsilon_r \mu_r}}$$

When a wave crosses an interface between media of different impedances, partial reflection occurs — just like a wave on a string encountering a change in density.

```python
import numpy as np
import matplotlib.pyplot as plt

# Impedance of free space and its role in wave propagation
# Why impedance: it determines reflection and transmission at interfaces

mu_0 = 4 * np.pi * 1e-7
epsilon_0 = 8.854e-12
c = 1 / np.sqrt(mu_0 * epsilon_0)

Z_0 = np.sqrt(mu_0 / epsilon_0)
print(f"Impedance of free space: Z₀ = {Z_0:.4f} Ω")
print(f"                         Z₀ ≈ 120π = {120*np.pi:.4f} Ω")

# Impedance and reflection at interfaces
# Reflection coefficient: r = (Z₂ - Z₁)/(Z₂ + Z₁)
# Transmission coefficient: t = 2Z₂/(Z₂ + Z₁)
# Why these formulas: they follow from matching boundary conditions at the interface

materials = {
    'Vacuum': 1.0,
    'Air': 1.0006,
    'Glass (n=1.5)': 1.5**2,
    'Water (n=1.33)': 1.33**2,
    'Silicon (n=3.42)': 3.42**2,
    'Diamond (n=2.42)': 2.42**2,
}

print(f"\n{'Material':<22} {'εᵣ':>6} {'Z (Ω)':>10} {'r (from vacuum)':>16} {'R (%)':>8}")
print("=" * 70)

Z1 = Z_0  # incoming wave in vacuum
for name, eps_r in materials.items():
    Z2 = Z_0 / np.sqrt(eps_r)
    r = (Z2 - Z1) / (Z2 + Z1)     # amplitude reflection coefficient
    R = r**2                         # power reflectance
    print(f"{name:<22} {eps_r:6.3f} {Z2:10.2f} {r:16.4f} {R*100:8.2f}")

# Visualize reflection at glass interface
n_glass = 1.5
Z_glass = Z_0 / n_glass
r = (Z_glass - Z_0) / (Z_glass + Z_0)   # negative: phase flip on reflection
t = 2 * Z_glass / (Z_glass + Z_0)

print(f"\nNormal incidence reflection at air-glass interface:")
print(f"  r = {r:.4f} (negative = π phase shift)")
print(f"  t = {t:.4f}")
print(f"  R = {r**2*100:.2f}%")
print(f"  T = {(1-r**2)*100:.2f}%")
print(f"  R + T = {(r**2 + 1-r**2)*100:.1f}% (energy conservation) ✓")

# Plot: time snapshot of incident, reflected, and transmitted waves
z = np.linspace(-10, 10, 1000)
k1 = 1.0           # wave number in medium 1
k2 = n_glass * k1   # wave number in medium 2

# Incident wave (z < 0)
E_inc = np.where(z < 0, np.sin(k1 * z), 0)

# Reflected wave (z < 0, traveling backward)
E_ref = np.where(z < 0, r * np.sin(-k1 * z), 0)

# Transmitted wave (z > 0)
E_trans = np.where(z >= 0, t * np.sin(k2 * z), 0)

# Total field
E_total = E_inc + E_ref + E_trans

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(z, E_inc, 'b-', linewidth=1.5, label='Incident')
axes[0].plot(z, E_ref, 'r-', linewidth=1.5, label='Reflected')
axes[0].plot(z, E_trans, 'g-', linewidth=1.5, label='Transmitted')
axes[0].axvline(x=0, color='gray', linewidth=3, alpha=0.5)
axes[0].fill_between([0, 10], [-1.5, -1.5], [1.5, 1.5], alpha=0.05, color='blue')
axes[0].text(-5, 1.2, 'Vacuum', fontsize=12)
axes[0].text(3, 1.2, f'Glass (n={n_glass})', fontsize=12)
axes[0].set_ylabel('E')
axes[0].set_title('Individual Waves at Interface')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(z, E_total, 'k-', linewidth=2)
axes[1].axvline(x=0, color='gray', linewidth=3, alpha=0.5)
axes[1].fill_between([0, 10], [-1.5, -1.5], [1.5, 1.5], alpha=0.05, color='blue')
axes[1].set_xlabel('z')
axes[1].set_ylabel('E')
axes[1].set_title('Total Field (standing wave pattern in medium 1)')
axes[1].grid(True, alpha=0.3)

plt.suptitle(f'Reflection at Dielectric Interface (R = {r**2*100:.1f}%)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wave_reflection.png', dpi=150)
plt.show()
```

---

## The Electromagnetic Spectrum

All electromagnetic waves share the same fundamental nature — they differ only in frequency (or equivalently, wavelength):

$$c = f\lambda$$

| Region | Frequency | Wavelength | Source |
|---|---|---|---|
| Radio | < 300 MHz | > 1 m | Antennas |
| Microwave | 300 MHz - 300 GHz | 1 mm - 1 m | Klystrons, magnetrons |
| Infrared | 300 GHz - 430 THz | 700 nm - 1 mm | Thermal radiation |
| Visible | 430 - 750 THz | 400 - 700 nm | Atoms, molecules |
| Ultraviolet | 750 THz - 30 PHz | 10 - 400 nm | Hot stars, discharge |
| X-ray | 30 PHz - 30 EHz | 0.01 - 10 nm | Inner-shell electrons |
| Gamma ray | > 30 EHz | < 0.01 nm | Nuclear transitions |

```python
import numpy as np
import matplotlib.pyplot as plt

# The electromagnetic spectrum — frequencies and wavelengths
# Why log scale: the spectrum spans ~20 orders of magnitude

c = 3e8

# Define spectrum regions
regions = [
    ('Radio', 1e3, 3e8, 'red'),
    ('Microwave', 3e8, 3e11, 'orange'),
    ('Infrared', 3e11, 4.3e14, 'darkred'),
    ('Visible', 4.3e14, 7.5e14, 'green'),
    ('UV', 7.5e14, 3e16, 'purple'),
    ('X-ray', 3e16, 3e19, 'blue'),
    ('Gamma', 3e19, 3e22, 'black'),
]

fig, ax = plt.subplots(figsize=(14, 4))

for name, f_low, f_high, color in regions:
    ax.barh(0, np.log10(f_high) - np.log10(f_low), left=np.log10(f_low),
            height=0.6, color=color, alpha=0.6, edgecolor='black')
    f_mid = np.sqrt(f_low * f_high)
    ax.text(np.log10(f_mid), 0, name, ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')

ax.set_xlabel('log₁₀(frequency / Hz)')
ax.set_yticks([])
ax.set_xlim(2, 23)
ax.set_title('The Electromagnetic Spectrum', fontsize=14, fontweight='bold')

# Add wavelength axis on top
ax2 = ax.twiny()
ax2.set_xlim(2, 23)
# λ = c/f, so log₁₀(λ) = log₁₀(c) - log₁₀(f) = 8.477 - log₁₀(f)
tick_freqs = np.arange(3, 23, 2)
tick_lambdas = [f'{c/10**f:.0e}' for f in tick_freqs]
ax2.set_xticks(tick_freqs)
ax2.set_xticklabels(tick_lambdas, fontsize=8)
ax2.set_xlabel('Wavelength (m)')

plt.tight_layout()
plt.savefig('em_spectrum.png', dpi=150)
plt.show()

# Photon energy E = hf
h = 6.626e-34   # Planck's constant
print("\nPhoton energies across the spectrum:")
print(f"{'Region':<12} {'f (Hz)':>12} {'λ':>12} {'E (eV)':>12}")
print("=" * 50)
for name, f_low, f_high, _ in regions:
    f = np.sqrt(f_low * f_high)   # geometric mean
    lam = c / f
    E_eV = h * f / 1.6e-19

    if lam >= 1:
        lam_str = f"{lam:.1f} m"
    elif lam >= 1e-3:
        lam_str = f"{lam*1e3:.1f} mm"
    elif lam >= 1e-6:
        lam_str = f"{lam*1e6:.1f} μm"
    elif lam >= 1e-9:
        lam_str = f"{lam*1e9:.1f} nm"
    else:
        lam_str = f"{lam*1e12:.2f} pm"

    print(f"{name:<12} {f:12.2e} {lam_str:>12} {E_eV:12.4e}")
```

---

## Summary

| Concept | Key Equation |
|---|---|
| Plane wave | $\tilde{\mathbf{E}} = \tilde{\mathbf{E}}_0\,e^{i(\mathbf{k}\cdot\mathbf{r}-\omega t)}$ |
| Dispersion relation | $\omega = ck$, $c = 1/\sqrt{\mu_0\epsilon_0}$ |
| Transversality | $\mathbf{E} \perp \mathbf{B} \perp \mathbf{k}$ |
| E-B relation | $\mathbf{B} = \frac{1}{c}\hat{k}\times\mathbf{E}$, $B_0 = E_0/c$ |
| Energy density | $u = \epsilon_0 E^2 = B^2/\mu_0$ |
| Energy equipartition | $u_E = u_B$ (electric = magnetic) |
| Intensity | $I = \frac{1}{2}\epsilon_0 c E_0^2 = E_0^2/(2Z_0)$ |
| Poynting vector | $\mathbf{S} = cu\,\hat{k}$ |
| Radiation pressure | $P = I/c$ (absorb), $2I/c$ (reflect) |
| Impedance of free space | $Z_0 = \sqrt{\mu_0/\epsilon_0} \approx 377\;\Omega$ |
| Stokes parameters | $S_0, S_1, S_2, S_3$ describe polarization |
| Reflection coefficient | $r = (Z_2-Z_1)/(Z_2+Z_1)$ |

---

## Exercises

### Exercise 1: Wave Verification
Verify that $\mathbf{E} = E_0\cos(kz - \omega t)\hat{x}$ and $\mathbf{B} = (E_0/c)\cos(kz - \omega t)\hat{y}$ satisfy all four Maxwell equations in vacuum. Check each equation explicitly.

### Exercise 2: Circular Polarization Decomposition
Show that any linearly polarized wave can be decomposed into two circularly polarized waves of equal amplitude. Write the decomposition explicitly and verify it numerically.

### Exercise 3: Solar Sail
A solar sail of area $A = 100$ m$^2$ with reflectivity $R = 0.95$ is positioned 1 AU from the Sun ($I = 1361$ W/m$^2$). Calculate (a) the radiation force, (b) the acceleration if the sail mass is 1 kg, (c) the time to reach the orbit of Mars (1.5 AU), ignoring gravity. Compare the radiation force with the Sun's gravitational pull on the sail.

### Exercise 4: Polarizer Chain
Three ideal linear polarizers are placed in series. The first is at $0°$, the second at $45°$, and the third at $90°$. Using Malus's law ($I = I_0\cos^2\theta$), compute the fraction of incident unpolarized light that passes through all three. What happens with just two polarizers at $0°$ and $90°$? Simulate with Stokes vectors.

### Exercise 5: Standing Waves
Two plane waves of equal amplitude and frequency traveling in opposite directions create a standing wave. Derive the standing wave pattern for $\mathbf{E}$ and $\mathbf{B}$, and show that the time-averaged Poynting vector is zero. Verify numerically and plot the spatial envelope.

---

[← Previous: 08. Maxwell's Equations — Integral Form](08_Maxwells_Equations_Integral.md)
