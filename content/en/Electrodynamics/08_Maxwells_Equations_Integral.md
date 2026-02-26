# Maxwell's Equations — Integral Form

[← Previous: 07. Maxwell's Equations — Differential Form](07_Maxwells_Equations_Differential.md) | [Next: 09. EM Waves in Vacuum →](09_EM_Waves_Vacuum.md)

---

## Learning Objectives

1. Convert Maxwell's equations between differential and integral forms using Stokes' and divergence theorems
2. Derive the continuity equation and explain conservation of charge
3. State and derive Poynting's theorem for electromagnetic energy conservation
4. Compute the Poynting vector and interpret energy flow in electromagnetic systems
5. Introduce the Maxwell stress tensor and explain electromagnetic momentum
6. Apply energy and momentum conservation to practical problems
7. Numerically compute energy flow and verify Poynting's theorem

---

The differential form of Maxwell's equations is elegant and compact, but the integral form connects directly to measurements: voltmeters measure line integrals of $\mathbf{E}$, and fluxmeters measure surface integrals of $\mathbf{B}$. More importantly, the integral form naturally leads to conservation laws — the conservation of charge and the conservation of electromagnetic energy. Poynting's theorem, which we derive in this lesson, reveals that electromagnetic fields carry energy and momentum, flowing through space as a tangible physical quantity. The Poynting vector $\mathbf{S} = \frac{1}{\mu_0}\mathbf{E} \times \mathbf{B}$ tells us exactly where energy is going, answering questions like "how does energy get from a battery to a light bulb?"

---

## Integral Forms of Maxwell's Equations

The differential and integral forms are connected by two fundamental theorems of vector calculus:

- **Divergence theorem**: $\int_V (\nabla \cdot \mathbf{F}) \, d\tau = \oint_S \mathbf{F} \cdot d\mathbf{a}$
- **Stokes' theorem**: $\int_S (\nabla \times \mathbf{F}) \cdot d\mathbf{a} = \oint_C \mathbf{F} \cdot d\mathbf{l}$

Applying these to Maxwell's equations:

### (i) Gauss's Law

$$\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0} \xrightarrow{\text{divergence thm}} \boxed{\oint_S \mathbf{E} \cdot d\mathbf{a} = \frac{Q_{\text{enc}}}{\epsilon_0}}$$

The total electric flux through a closed surface equals the enclosed charge divided by $\epsilon_0$.

### (ii) No Magnetic Monopoles

$$\nabla \cdot \mathbf{B} = 0 \xrightarrow{\text{divergence thm}} \boxed{\oint_S \mathbf{B} \cdot d\mathbf{a} = 0}$$

The total magnetic flux through any closed surface is zero. Every magnetic field line that enters a volume must also exit it.

### (iii) Faraday's Law

$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t} \xrightarrow{\text{Stokes' thm}} \boxed{\oint_C \mathbf{E} \cdot d\mathbf{l} = -\frac{d}{dt}\int_S \mathbf{B} \cdot d\mathbf{a}}$$

The EMF around a loop equals the negative rate of change of magnetic flux through the loop.

### (iv) Ampere-Maxwell Law

$$\nabla \times \mathbf{B} = \mu_0\mathbf{J} + \mu_0\epsilon_0\frac{\partial \mathbf{E}}{\partial t} \xrightarrow{\text{Stokes' thm}} \boxed{\oint_C \mathbf{B} \cdot d\mathbf{l} = \mu_0 I_{\text{enc}} + \mu_0\epsilon_0\frac{d}{dt}\int_S \mathbf{E} \cdot d\mathbf{a}}$$

The circulation of $\mathbf{B}$ around a loop equals $\mu_0$ times the total current (real + displacement) through the loop.

> **Analogy**: Think of the differential form as a microscope — it describes what happens at a single point. The integral form is the view from a satellite — it describes the cumulative effect over a region. The divergence theorem and Stokes' theorem are the "zoom" buttons that connect the two views.

---

## Conservation of Charge: The Continuity Equation

Taking the divergence of the Ampere-Maxwell equation:

$$\nabla \cdot (\nabla \times \mathbf{B}) = \mu_0\nabla \cdot \mathbf{J} + \mu_0\epsilon_0\frac{\partial}{\partial t}(\nabla \cdot \mathbf{E})$$

The left side vanishes (divergence of curl is zero). Using Gauss's law on the right:

$$0 = \mu_0\nabla \cdot \mathbf{J} + \mu_0\frac{\partial \rho}{\partial t}$$

$$\boxed{\nabla \cdot \mathbf{J} + \frac{\partial \rho}{\partial t} = 0} \qquad \text{(Continuity equation)}$$

In integral form (using the divergence theorem):

$$\oint_S \mathbf{J} \cdot d\mathbf{a} = -\frac{dQ_{\text{enc}}}{dt}$$

**Physical meaning**: The net current flowing out of a volume equals the rate of decrease of charge inside. Charge is neither created nor destroyed — it is **locally conserved**. This is the most fundamental conservation law in electrodynamics.

```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstrate charge conservation: expanding charged sphere
# Why this model: it cleanly illustrates the continuity equation

# A uniformly charged sphere expands radially
# ρ(r,t) = ρ₀(R₀/R(t))³ for r < R(t), where R(t) = R₀ + vt
# J(r,t) = ρ(r,t) v r̂

R0 = 1.0         # initial radius
rho_0 = 1.0      # initial charge density
v = 0.5           # expansion velocity

t_values = [0, 0.5, 1.0, 1.5, 2.0]
r = np.linspace(0, 4, 1000)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

total_charge_values = []

for t_val in t_values:
    R_t = R0 + v * t_val
    # Charge density: total charge Q = ρ₀(4π/3)R₀³ is conserved
    # So ρ(t) = ρ₀(R₀/R(t))³
    rho_t = rho_0 * (R0 / R_t)**3

    rho_profile = np.where(r < R_t, rho_t, 0)
    axes[0].plot(r, rho_profile, linewidth=2, label=f't = {t_val:.1f}')

    # Current density J = ρv at r < R(t)
    J_profile = np.where(r < R_t, rho_t * v, 0)
    axes[1].plot(r, J_profile, linewidth=2, label=f't = {t_val:.1f}')

    # Verify total charge is conserved
    # Q = ∫ρ 4πr² dr
    Q = np.trapz(rho_profile * 4 * np.pi * r**2, r)
    total_charge_values.append(Q)

Q_exact = rho_0 * (4/3) * np.pi * R0**3
axes[0].set_xlabel('r')
axes[0].set_ylabel(r'$\rho(r)$')
axes[0].set_title('Charge Density (expanding sphere)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('r')
axes[1].set_ylabel(r'$J_r(r)$')
axes[1].set_title('Current Density')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Continuity Equation: Charge Conservation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('continuity_equation.png', dpi=150)
plt.show()

# Verify charge conservation
print("Charge conservation verification:")
print(f"Exact total charge: Q = {Q_exact:.4f}")
for t_val, Q_num in zip(t_values, total_charge_values):
    print(f"  t = {t_val:.1f}: Q = {Q_num:.4f} (error: {abs(Q_num-Q_exact)/Q_exact:.2e})")
```

---

## Poynting's Theorem: Energy Conservation

How much energy is stored in an electromagnetic field, and how does it flow? Poynting's theorem answers both questions.

### Derivation

Start from the rate at which the electromagnetic field does work on charges. The force per unit volume is $\mathbf{f} = \rho\mathbf{E} + \mathbf{J}\times\mathbf{B}$. The power delivered per unit volume is:

$$\frac{dW_{\text{mech}}}{dt \, d\tau} = \mathbf{f} \cdot \mathbf{v} = \rho\mathbf{v}\cdot\mathbf{E} + (\mathbf{J}\times\mathbf{B})\cdot\mathbf{v}$$

Since $\mathbf{J} = \rho\mathbf{v}$ and $(\mathbf{J}\times\mathbf{B})\cdot\mathbf{v} = (\mathbf{v}\times\mathbf{B})\cdot\mathbf{J} = 0$ (magnetic force does no work):

$$\frac{dW_{\text{mech}}}{dt \, d\tau} = \mathbf{J} \cdot \mathbf{E}$$

Now eliminate $\mathbf{J}$ using the Ampere-Maxwell law and use Faraday's law to simplify. After algebra involving the vector identity $\nabla\cdot(\mathbf{E}\times\mathbf{B}) = \mathbf{B}\cdot(\nabla\times\mathbf{E}) - \mathbf{E}\cdot(\nabla\times\mathbf{B})$:

$$\boxed{-\frac{\partial u}{\partial t} = \nabla \cdot \mathbf{S} + \mathbf{J} \cdot \mathbf{E}}$$

where:

$$u = \frac{1}{2}\left(\epsilon_0 E^2 + \frac{B^2}{\mu_0}\right) \qquad \text{(energy density)}$$

$$\mathbf{S} = \frac{1}{\mu_0}\mathbf{E} \times \mathbf{B} \qquad \text{(Poynting vector)}$$

### Integral Form

$$\boxed{-\frac{dU_{\text{em}}}{dt} = \oint_S \mathbf{S} \cdot d\mathbf{a} + \int_V \mathbf{J}\cdot\mathbf{E} \, d\tau}$$

**Physical meaning**: The rate of decrease of electromagnetic energy in a volume equals the energy radiated out through the surface (Poynting flux) plus the energy delivered to charges inside (Joule heating).

> **Analogy**: Poynting's theorem is like a bank statement for electromagnetic energy. The energy stored in the field is your account balance ($U_{\text{em}}$). The Poynting vector is money flowing through the walls of the bank ($\mathbf{S}$). The $\mathbf{J}\cdot\mathbf{E}$ term is money being withdrawn by charges (spent on mechanical work or heat). The theorem says: balance decreases = outflow + withdrawals.

---

## The Poynting Vector

The Poynting vector $\mathbf{S} = \frac{1}{\mu_0}\mathbf{E}\times\mathbf{B}$ has units of W/m$^2$ and represents the **energy flux** — the rate of energy flow per unit area.

### Direction of Energy Flow

$\mathbf{S}$ points in the direction of energy propagation. For a plane electromagnetic wave propagating in the $\hat{z}$ direction with $\mathbf{E} = E_0\hat{x}$ and $\mathbf{B} = B_0\hat{y}$:

$$\mathbf{S} = \frac{E_0 B_0}{\mu_0}\hat{z} = \frac{E_0^2}{\mu_0 c}\hat{z}$$

Energy flows in the direction of wave propagation, as expected.

### Time-Averaged Poynting Vector

For oscillating fields, the time average is often more useful:

$$\langle\mathbf{S}\rangle = \frac{1}{2\mu_0}\text{Re}(\tilde{\mathbf{E}} \times \tilde{\mathbf{B}}^*)$$

where $\tilde{\mathbf{E}}$ and $\tilde{\mathbf{B}}$ are complex amplitudes.

### Intensity

The **intensity** of an electromagnetic wave is the magnitude of the time-averaged Poynting vector:

$$I = |\langle\mathbf{S}\rangle| = \frac{1}{2}\epsilon_0 c E_0^2 = \frac{E_0^2}{2\mu_0 c}$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Poynting vector of a plane electromagnetic wave
# Why visualize: seeing E, B, and S together clarifies energy flow

c = 3e8               # speed of light (m/s)
mu_0 = 4 * np.pi * 1e-7
epsilon_0 = 8.854e-12
f = 1e9               # frequency (1 GHz)
omega = 2 * np.pi * f
k = omega / c
E0 = 100              # electric field amplitude (V/m)
B0 = E0 / c           # magnetic field amplitude (T)

# Spatial profile at t = 0
z = np.linspace(0, 3 * c / f, 1000)  # 3 wavelengths

E_x = E0 * np.sin(k * z)           # E polarized in x
B_y = B0 * np.sin(k * z)           # B polarized in y
S_z = (1 / mu_0) * E_x * B_y       # Poynting vector in z

# Time-averaged intensity
I_avg = E0**2 / (2 * mu_0 * c)

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axes[0].plot(z * 1e2, E_x, 'b-', linewidth=2)
axes[0].set_ylabel('$E_x$ (V/m)')
axes[0].set_title('Electric Field')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='gray', alpha=0.3)

axes[1].plot(z * 1e2, B_y * 1e6, 'r-', linewidth=2)
axes[1].set_ylabel('$B_y$ ($\\mu$T)')
axes[1].set_title('Magnetic Field')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='gray', alpha=0.3)

axes[2].plot(z * 1e2, S_z, 'g-', linewidth=2, label='$S_z$ (instantaneous)')
axes[2].axhline(y=I_avg, color='orange', linestyle='--', linewidth=2,
                label=f'$\\langle S \\rangle$ = {I_avg:.2f} W/m²')
axes[2].set_ylabel('$S_z$ (W/m$^2$)')
axes[2].set_xlabel('z (cm)')
axes[2].set_title('Poynting Vector (Energy Flux)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('Plane EM Wave: E, B, and Poynting Vector',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('poynting_vector.png', dpi=150)
plt.show()

print(f"Wavelength:          λ = {c/f*100:.2f} cm")
print(f"E₀ = {E0} V/m")
print(f"B₀ = {B0*1e6:.4f} μT")
print(f"Time-averaged intensity: ⟨S⟩ = {I_avg:.4f} W/m²")
```

---

## Energy Flow in a Resistive Wire

A surprising application of the Poynting vector: how does energy get from a battery to a light bulb?

For a long straight wire carrying current $I$ with resistance per unit length $R/L$:

- **Inside the wire**: $\mathbf{E}$ is along the wire (driving the current), $\mathbf{B}$ circles the wire
- **The Poynting vector** $\mathbf{S} = \frac{1}{\mu_0}\mathbf{E}\times\mathbf{B}$ points **radially inward**!

Energy does not flow through the wire — it flows through the electromagnetic field surrounding the wire, entering the wire radially. The wire acts as a "sink" that absorbs energy from the field and converts it to heat.

The total power entering a cylindrical surface of radius $a$ (the wire surface) and length $l$:

$$P = \oint \mathbf{S}\cdot d\mathbf{a} = \frac{E \cdot B}{2\mu_0}(2\pi a l) = \frac{(V/l)(\mu_0 I/2\pi a)}{\mu_0}(2\pi a l) = VI = I^2R$$

This is Joule's law — confirmed by Poynting's theorem.

```python
import numpy as np
import matplotlib.pyplot as plt

# Poynting vector around a current-carrying resistive wire
# Why this example: it reveals that energy flows through the field, not the wire

mu_0 = 4 * np.pi * 1e-7

I = 1.0          # current (A)
V_per_m = 0.1    # voltage drop per meter (V/m) — determines E inside
a_wire = 0.001   # wire radius (1 mm)

# Grid around the wire (cross-section in xy-plane, wire along z)
x = np.linspace(-0.02, 0.02, 40)
y = np.linspace(-0.02, 0.02, 40)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)
r = np.maximum(r, a_wire)   # clip to wire surface

# E field: along z everywhere (uniform in this simple model)
# B field: circles the wire, B = μ₀I/(2πr)
# Why E along z: driven by the battery/voltage source
E_z = V_per_m   # uniform E field along wire

B_phi = mu_0 * I / (2 * np.pi * r)  # magnitude
# Convert B_φ to Cartesian: B_x = -B_φ sin(φ), B_y = B_φ cos(φ)
phi = np.arctan2(Y, X)
B_x = -B_phi * np.sin(phi)
B_y = B_phi * np.cos(phi)

# Poynting vector: S = (1/μ₀)(E × B)
# E = E_z ẑ, B = B_x x̂ + B_y ŷ
# Why cross product: S = (E_z/μ₀)(ẑ × (B_x x̂ + B_y ŷ)) = (E_z/μ₀)(B_x ŷ - B_y x̂)
# Wait — let's be careful: ẑ × x̂ = ŷ, ẑ × ŷ = -x̂
S_x = -(E_z / mu_0) * B_y   # radially inward component (x)
S_y = (E_z / mu_0) * B_x    # radially inward component (y)

# Verify: S should point radially INWARD (toward the wire)
S_r = (S_x * X + S_y * Y) / r   # radial component

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Poynting vector field
S_mag = np.sqrt(S_x**2 + S_y**2)
axes[0].streamplot(X, Y, S_x, S_y, color=np.log10(S_mag + 1e-10),
                   cmap='viridis', density=2)
circle = plt.Circle((0, 0), a_wire, color='gray', fill=True, alpha=0.5)
axes[0].add_patch(circle)
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('y (m)')
axes[0].set_title('Poynting Vector S (Energy flows INTO the wire)')
axes[0].set_aspect('equal')

# Radial component of S
r_line = np.linspace(a_wire, 0.02, 100)
S_r_line = -E_z * mu_0 * I / (2 * np.pi * r_line * mu_0)
# Simplified: S_r = -E_z * I / (2π r) — negative means inward
axes[1].plot(r_line * 1000, -S_r_line, 'r-', linewidth=2)
axes[1].set_xlabel('r (mm)')
axes[1].set_ylabel('|S_r| (W/m²)')
axes[1].set_title('Radial Energy Flux (inward)')
axes[1].grid(True, alpha=0.3)

# Total power entering the wire (per meter length)
P_per_m = I * V_per_m  # = I²R per meter
axes[1].text(0.95, 0.95, f'Power/m = IV = {P_per_m:.3f} W/m\n= I²R (Joule heating)',
             transform=axes[1].transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat'))

plt.suptitle('Energy Flow Around a Resistive Wire', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('poynting_wire.png', dpi=150)
plt.show()
```

---

## Maxwell Stress Tensor

Electromagnetic fields carry **momentum** as well as energy. The momentum density is:

$$\mathbf{g} = \mu_0\epsilon_0 \mathbf{S} = \epsilon_0(\mathbf{E}\times\mathbf{B})$$

The **Maxwell stress tensor** $\overleftrightarrow{T}$ describes the electromagnetic force per unit area on a surface:

$$T_{ij} = \epsilon_0\left(E_i E_j - \frac{1}{2}\delta_{ij}E^2\right) + \frac{1}{\mu_0}\left(B_i B_j - \frac{1}{2}\delta_{ij}B^2\right)$$

The force on charges in a volume $V$ bounded by surface $S$ is:

$$\mathbf{F} = \oint_S \overleftrightarrow{T} \cdot d\mathbf{a} - \mu_0\epsilon_0\frac{d}{dt}\int_V \mathbf{S} \, d\tau$$

In static fields, the second term vanishes and:

$$F_i = \oint_S \sum_j T_{ij} \, da_j$$

### Physical Interpretation

The stress tensor tells us:
- **Diagonal elements** ($T_{xx}, T_{yy}, T_{zz}$): pressure (tension along field lines, compression perpendicular to them)
- **Off-diagonal elements**: shear stress

Electric and magnetic field lines behave like **rubber bands under tension** — they pull along their length and push sideways. This is the mechanical origin of electromagnetic forces.

### Radiation Pressure

When an electromagnetic wave hits a surface, it transfers momentum and exerts pressure:

**Perfect absorption**:
$$P_{\text{rad}} = \frac{I}{c} = \frac{\langle S \rangle}{c}$$

**Perfect reflection**:
$$P_{\text{rad}} = \frac{2I}{c}$$

For sunlight at Earth ($I = 1361$ W/m$^2$): $P_{\text{rad}} \approx 4.5\,\mu$Pa for absorption, $9\,\mu$Pa for reflection. Tiny — but enough to measurably affect the orbits of small asteroids (the Yarkovsky effect) and to propel solar sails.

### Angular Momentum

Circularly polarized light also carries angular momentum. The angular momentum density is:

$$\ell_z = \pm \frac{u}{\omega}$$

where the $\pm$ corresponds to right/left circular polarization. Each photon carries angular momentum $\pm\hbar$ — this is the spin-1 nature of the photon.

```python
import numpy as np

# Maxwell stress tensor: compute force between two parallel charged plates
# Why stress tensor: it gives the force without knowing the charge distribution in detail

epsilon_0 = 8.854e-12
mu_0 = 4 * np.pi * 1e-7

# Parallel plate capacitor
sigma = 1e-6      # surface charge density (μC/m²)
A = 0.01          # plate area (100 cm²)

# Electric field between plates
E = sigma / epsilon_0   # V/m (field in gap, no B field)

# Maxwell stress tensor between the plates
# With E = E_z ẑ only:
# T_zz = ε₀(E_z² - E²/2) = ε₀ E²/2
# This is the pressure (force per unit area) on the plate
T_zz = epsilon_0 * E**2 / 2

# Force on one plate
F = T_zz * A

# Compare with direct Coulomb force: F = σ²A/(2ε₀)
F_coulomb = sigma**2 * A / (2 * epsilon_0)

print("Maxwell Stress Tensor: Force Between Capacitor Plates")
print("=" * 55)
print(f"Surface charge density: σ = {sigma*1e6:.1f} μC/m²")
print(f"Plate area:             A = {A*1e4:.0f} cm²")
print(f"Electric field:         E = {E:.2f} V/m")
print(f"")
print(f"Stress T_zz = ε₀E²/2 = {T_zz:.4f} Pa")
print(f"Force (stress tensor): F = {F:.6f} N")
print(f"Force (Coulomb):       F = {F_coulomb:.6f} N")
print(f"Agreement:             {abs(F - F_coulomb)/F_coulomb:.2e} relative error")
print(f"")
print(f"Electromagnetic momentum density:")
print(f"  g = ε₀(E × B) = 0 (no B field in this static case)")
```

---

## Summary of Conservation Laws

| Conservation Law | Differential Form | Integral Form |
|---|---|---|
| **Charge** | $\nabla\cdot\mathbf{J} + \partial\rho/\partial t = 0$ | $\oint \mathbf{J}\cdot d\mathbf{a} = -dQ_{\text{enc}}/dt$ |
| **Energy** | $\nabla\cdot\mathbf{S} + \partial u/\partial t = -\mathbf{J}\cdot\mathbf{E}$ | $\oint \mathbf{S}\cdot d\mathbf{a} = -dU_{\text{em}}/dt - \int\mathbf{J}\cdot\mathbf{E}\,d\tau$ |
| **Momentum** | $\nabla\cdot\overleftrightarrow{T} - \mu_0\epsilon_0\partial\mathbf{S}/\partial t = \mathbf{f}$ | $\oint \overleftrightarrow{T}\cdot d\mathbf{a} = \mathbf{F}_{\text{mech}} + d\mathbf{p}_{\text{em}}/dt$ |

Each conservation law has the same structure: a flux through a surface balanced by a rate of change inside the volume.

---

## Maxwell's Equations: Complete Summary

| # | Differential | Integral | Name |
|---|---|---|---|
| I | $\nabla\cdot\mathbf{E} = \rho/\epsilon_0$ | $\oint\mathbf{E}\cdot d\mathbf{a} = Q_{\text{enc}}/\epsilon_0$ | Gauss |
| II | $\nabla\cdot\mathbf{B} = 0$ | $\oint\mathbf{B}\cdot d\mathbf{a} = 0$ | No monopoles |
| III | $\nabla\times\mathbf{E} = -\partial\mathbf{B}/\partial t$ | $\oint\mathbf{E}\cdot d\mathbf{l} = -d\Phi_B/dt$ | Faraday |
| IV | $\nabla\times\mathbf{B} = \mu_0\mathbf{J}+\mu_0\epsilon_0\partial\mathbf{E}/\partial t$ | $\oint\mathbf{B}\cdot d\mathbf{l} = \mu_0(I_{\text{enc}}+\epsilon_0 d\Phi_E/dt)$ | Ampere-Maxwell |

**Auxiliary**: $\mathbf{F} = q(\mathbf{E}+\mathbf{v}\times\mathbf{B})$ (Lorentz force), $\nabla\cdot\mathbf{J}+\partial\rho/\partial t = 0$ (continuity)

---

## Summary

| Concept | Key Equation |
|---|---|
| Continuity equation | $\nabla\cdot\mathbf{J} + \partial\rho/\partial t = 0$ |
| Energy density | $u = \frac{1}{2}(\epsilon_0 E^2 + B^2/\mu_0)$ |
| Poynting vector | $\mathbf{S} = \frac{1}{\mu_0}\mathbf{E}\times\mathbf{B}$ |
| Poynting's theorem | $-\partial u/\partial t = \nabla\cdot\mathbf{S} + \mathbf{J}\cdot\mathbf{E}$ |
| Wave intensity | $I = \frac{1}{2}\epsilon_0 c E_0^2$ |
| EM momentum density | $\mathbf{g} = \epsilon_0(\mathbf{E}\times\mathbf{B}) = \mathbf{S}/c^2$ |
| Stress tensor | $T_{ij} = \epsilon_0(E_iE_j - \frac{1}{2}\delta_{ij}E^2) + \frac{1}{\mu_0}(B_iB_j - \frac{1}{2}\delta_{ij}B^2)$ |
| Radiation pressure | $P = I/c$ (absorption), $P = 2I/c$ (reflection) |

---

## Exercises

### Exercise 1: Energy in a Capacitor
A parallel-plate capacitor ($A = 100$ cm$^2$, $d = 1$ mm) is charged to $V = 1000$ V. Compute the total energy stored using (a) $W = \frac{1}{2}CV^2$ and (b) $W = \frac{\epsilon_0}{2}\int E^2 \, d\tau$. Show they agree.

### Exercise 2: Poynting Vector of a Charging Capacitor
During charging, the Poynting vector between the plates of a circular capacitor points radially inward. Compute $\mathbf{S}$ at the edge of the plates and verify that the total inward flux equals the rate of increase of field energy between the plates.

### Exercise 3: Solar Radiation Pressure
The Sun delivers about $I = 1361$ W/m$^2$ at Earth's distance. Compute (a) the radiation pressure on a perfectly absorbing surface and a perfectly reflecting surface, (b) the total force on a $10 \times 10$ m solar sail, (c) the electric and magnetic field amplitudes.

### Exercise 4: Stress Tensor Force
Use the Maxwell stress tensor to compute the force per unit length between two parallel infinite wires carrying currents $I_1$ and $I_2$ separated by distance $d$. Verify it agrees with $F/L = \mu_0 I_1 I_2/(2\pi d)$.

### Exercise 5: Numerical Poynting Theorem Verification
For the 1D FDTD simulation from Lesson 7, compute the energy density $u(x,t)$ and Poynting vector $S(x,t)$ at each time step. Verify Poynting's theorem: the total energy in the domain plus the energy that has left through the boundaries should equal the initial energy.

---

[← Previous: 07. Maxwell's Equations — Differential Form](07_Maxwells_Equations_Differential.md) | [Next: 09. EM Waves in Vacuum →](09_EM_Waves_Vacuum.md)
