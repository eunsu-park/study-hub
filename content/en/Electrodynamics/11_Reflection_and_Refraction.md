# 11. Reflection and Refraction

[← Previous: 10. EM Waves in Matter](10_EM_Waves_Matter.md) | [Next: 12. Waveguides and Cavities →](12_Waveguides_and_Cavities.md)

## Learning Objectives

1. Derive Snell's law from the boundary conditions of Maxwell's equations
2. Derive the Fresnel equations for s- and p-polarized light
3. Compute reflectance and transmittance and verify energy conservation
4. Understand Brewster's angle and its applications in polarization optics
5. Analyze total internal reflection and the physics of evanescent waves
6. Design single-layer anti-reflection coatings using thin-film interference
7. Implement numerical calculations of Fresnel coefficients using Python

When light hits the interface between two different media, part of the wave reflects and part transmits. The precise fractions depend on the angle of incidence, the polarization, and the refractive indices of the two materials. These phenomena, described quantitatively by Fresnel's equations, underpin everything from eyeglass coatings and fiber optic couplers to the glare off a lake and the sparkle of a diamond. In this lesson, we derive the Fresnel equations directly from Maxwell's boundary conditions, gaining insight into why polarized sunglasses work and how to eliminate reflections with thin-film coatings.

> **Analogy**: Imagine a rope wave hitting a junction where the rope changes thickness. At the junction, part of the wave bounces back and part continues into the thicker rope (at a different speed). The "impedance mismatch" between the two rope segments determines how much energy is reflected vs. transmitted. For electromagnetic waves, the refractive index plays the role of rope thickness, and the angle of incidence adds geometric richness to the problem.

---

## 1. Boundary Conditions at an Interface

### 1.1 Maxwell's Boundary Conditions

At a planar interface between media 1 and 2 (with no free surface charges or currents), Maxwell's equations require:

$$\epsilon_1 E_{1\perp} = \epsilon_2 E_{2\perp} \quad \text{(normal } D \text{ continuous)}$$

$$B_{1\perp} = B_{2\perp} \quad \text{(normal } B \text{ continuous)}$$

$$E_{1\parallel} = E_{2\parallel} \quad \text{(tangential } E \text{ continuous)}$$

$$\frac{1}{\mu_1} B_{1\parallel} = \frac{1}{\mu_2} B_{2\parallel} \quad \text{(tangential } H \text{ continuous)}$$

These conditions must hold at every point on the interface and at all times.

### 1.2 Phase Matching — Deriving Snell's Law

Consider a plane wave $\mathbf{E}_I = \mathbf{E}_0 \, e^{i(\mathbf{k}_I \cdot \mathbf{r} - \omega t)}$ incident on a planar interface at $z = 0$. Let the reflected wave have wave vector $\mathbf{k}_R$ and the transmitted wave $\mathbf{k}_T$.

The boundary conditions must hold for **all** $x$ and $t$ on the interface. This requires the phase of all three waves to match at $z = 0$:

$$\mathbf{k}_I \cdot \mathbf{r}\big|_{z=0} = \mathbf{k}_R \cdot \mathbf{r}\big|_{z=0} = \mathbf{k}_T \cdot \mathbf{r}\big|_{z=0}$$

This yields two results:

**Law of reflection**: $\theta_R = \theta_I$ (angle of incidence equals angle of reflection)

**Snell's law**:

$$\boxed{n_1 \sin\theta_I = n_2 \sin\theta_T}$$

where $\theta_I$ is the angle of incidence (measured from the surface normal) and $\theta_T$ is the angle of transmission (refraction).

Note that Snell's law is not an independent postulate — it emerges naturally from requiring that Maxwell's boundary conditions are satisfiable.

---

## 2. The Fresnel Equations

### 2.1 Polarization Convention

We decompose the incident field into two polarization components:

- **s-polarization** (from German *senkrecht* = perpendicular): $\mathbf{E}$ perpendicular to the plane of incidence
- **p-polarization** (parallel): $\mathbf{E}$ in the plane of incidence

The plane of incidence is defined by the incident wave vector $\mathbf{k}_I$ and the surface normal $\hat{n}$.

### 2.2 s-Polarization (TE)

For s-polarization, the electric field points out of the plane of incidence (say, along $\hat{y}$). Applying the tangential $E$ and $H$ boundary conditions:

$$E_I + E_R = E_T$$

$$\frac{1}{\mu_1}(E_I - E_R)\cos\theta_I = \frac{1}{\mu_2} E_T \cos\theta_T$$

For non-magnetic materials ($\mu_1 = \mu_2 = \mu_0$), the **Fresnel reflection coefficient** for s-polarization is:

$$\boxed{r_s = \frac{n_1 \cos\theta_I - n_2 \cos\theta_T}{n_1 \cos\theta_I + n_2 \cos\theta_T}}$$

and the **transmission coefficient**:

$$\boxed{t_s = \frac{2 n_1 \cos\theta_I}{n_1 \cos\theta_I + n_2 \cos\theta_T}}$$

### 2.3 p-Polarization (TM)

For p-polarization, the magnetic field is perpendicular to the plane of incidence. The boundary conditions give:

$$\boxed{r_p = \frac{n_2 \cos\theta_I - n_1 \cos\theta_T}{n_2 \cos\theta_I + n_1 \cos\theta_T}}$$

$$\boxed{t_p = \frac{2 n_1 \cos\theta_I}{n_2 \cos\theta_I + n_1 \cos\theta_T}}$$

### 2.4 Normal Incidence

At $\theta_I = 0$ (normal incidence), both polarizations give the same result:

$$r = \frac{n_1 - n_2}{n_1 + n_2}, \quad t = \frac{2n_1}{n_1 + n_2}$$

For an air-glass interface ($n_1 = 1.0$, $n_2 = 1.5$): $r = -0.2$, meaning 4% of the intensity is reflected. The negative sign indicates a $\pi$ phase shift.

```python
import numpy as np
import matplotlib.pyplot as plt

def fresnel_coefficients(n1, n2, theta_i):
    """
    Compute Fresnel reflection and transmission coefficients.

    Parameters:
        n1, n2   : refractive indices (can be complex for absorbing media)
        theta_i  : angle of incidence (radians), array

    Returns:
        r_s, r_p, t_s, t_p (complex amplitude coefficients)

    Why complex: the coefficients carry phase information, which is
    crucial for thin-film interference calculations and for understanding
    evanescent waves in total internal reflection.
    """
    # Snell's law to find transmission angle
    # Use complex sqrt to handle total internal reflection gracefully
    cos_i = np.cos(theta_i)
    sin_i = np.sin(theta_i)
    cos_t = np.sqrt(1 - (n1 / n2 * sin_i)**2 + 0j)

    # s-polarization (TE)
    r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    t_s = 2 * n1 * cos_i / (n1 * cos_i + n2 * cos_t)

    # p-polarization (TM)
    r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    t_p = 2 * n1 * cos_i / (n2 * cos_i + n1 * cos_t)

    return r_s, r_p, t_s, t_p

# Air to glass (external reflection)
n1, n2 = 1.0, 1.5
theta = np.linspace(0, np.pi / 2 - 0.001, 500)

r_s, r_p, t_s, t_p = fresnel_coefficients(n1, n2, theta)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Amplitude coefficients
axes[0].plot(np.degrees(theta), r_s.real, 'b-', linewidth=2, label='$r_s$')
axes[0].plot(np.degrees(theta), r_p.real, 'r-', linewidth=2, label='$r_p$')
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Angle of incidence (degrees)')
axes[0].set_ylabel('Amplitude coefficient')
axes[0].set_title(f'Fresnel Coefficients: Air ($n_1$={n1}) → Glass ($n_2$={n2})')
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)

# Reflectance and Transmittance (intensity)
R_s = np.abs(r_s)**2
R_p = np.abs(r_p)**2

axes[1].plot(np.degrees(theta), R_s, 'b-', linewidth=2, label='$R_s$')
axes[1].plot(np.degrees(theta), R_p, 'r-', linewidth=2, label='$R_p$')
axes[1].plot(np.degrees(theta), 0.5 * (R_s + R_p), 'k--', linewidth=1.5,
             label='Unpolarized')

# Mark Brewster angle
theta_B = np.arctan(n2 / n1)
axes[1].axvline(x=np.degrees(theta_B), color='green', linestyle=':',
                label=f"Brewster's angle = {np.degrees(theta_B):.1f}°")

axes[1].set_xlabel('Angle of incidence (degrees)')
axes[1].set_ylabel('Reflectance')
axes[1].set_title('Reflectance vs. Angle')
axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fresnel_air_glass.png", dpi=150)
plt.show()
```

---

## 3. Reflectance and Transmittance

### 3.1 Definitions

The **reflectance** $R$ and **transmittance** $T$ describe the fraction of incident *power* that is reflected and transmitted:

$$R_s = |r_s|^2, \quad R_p = |r_p|^2$$

$$T_s = \frac{n_2 \cos\theta_T}{n_1 \cos\theta_I} |t_s|^2, \quad T_p = \frac{n_2 \cos\theta_T}{n_1 \cos\theta_I} |t_p|^2$$

The factor $n_2 \cos\theta_T / (n_1 \cos\theta_I)$ accounts for the change in beam cross-section and wave speed upon refraction.

### 3.2 Energy Conservation

Energy conservation requires:

$$R + T = 1$$

for each polarization separately. This is a useful check on any Fresnel calculation.

```python
def verify_energy_conservation(n1, n2, theta_i):
    """
    Verify R + T = 1 for both polarizations.

    Why verify: this catch numerical errors and ensures that
    the Fresnel equations are implemented correctly — a common
    source of sign and factor-of-2 mistakes.
    """
    r_s, r_p, t_s, t_p = fresnel_coefficients(n1, n2, theta_i)

    cos_i = np.cos(theta_i)
    sin_t = n1 / n2 * np.sin(theta_i)
    cos_t = np.sqrt(1 - sin_t**2 + 0j)

    R_s = np.abs(r_s)**2
    R_p = np.abs(r_p)**2

    # Transmittance includes the beam area and velocity correction
    T_s = (n2 * cos_t.real) / (n1 * cos_i) * np.abs(t_s)**2
    T_p = (n2 * cos_t.real) / (n1 * cos_i) * np.abs(t_p)**2

    print(f"At theta_i = {np.degrees(theta_i):.1f}°:")
    print(f"  R_s + T_s = {(R_s + T_s).real:.6f}")
    print(f"  R_p + T_p = {(R_p + T_p).real:.6f}")

# Check at several angles
for angle_deg in [0, 30, 56.3, 80]:
    verify_energy_conservation(1.0, 1.5, np.radians(angle_deg))
```

---

## 4. Brewster's Angle

### 4.1 Derivation

At Brewster's angle $\theta_B$, the reflected p-polarization vanishes: $r_p = 0$. From the Fresnel equation:

$$n_2 \cos\theta_B = n_1 \cos\theta_T$$

Combined with Snell's law, this gives:

$$\boxed{\tan\theta_B = \frac{n_2}{n_1}}$$

At Brewster's angle, $\theta_I + \theta_T = 90°$. The reflected and refracted rays are perpendicular.

### 4.2 Physical Interpretation

The reflected wave is generated by the oscillating dipoles induced in the second medium. At Brewster's angle, the refracted wave travels perpendicular to the would-be reflected wave. Since dipoles do not radiate along their oscillation axis, the p-polarized dipoles cannot emit radiation in the reflection direction. Only s-polarized light is reflected.

### 4.3 Applications

- **Polarized sunglasses**: Glare from horizontal surfaces (roads, water) is predominantly s-polarized at angles near Brewster's angle. Polarized lenses block s-polarized light.
- **Brewster windows**: Laser cavities use windows tilted at Brewster's angle to eliminate reflection loss for p-polarized light, forcing the laser to emit linearly polarized light.
- **Pseudo-Brewster angle**: For absorbing media (complex $n_2$), $R_p$ reaches a minimum but does not vanish. The angle of minimum $R_p$ is the pseudo-Brewster angle.

---

## 5. Total Internal Reflection

### 5.1 Critical Angle

When light travels from a denser medium to a less dense one ($n_1 > n_2$), Snell's law gives:

$$\sin\theta_T = \frac{n_1}{n_2} \sin\theta_I$$

When $\sin\theta_T = 1$ (i.e., $\theta_T = 90°$), we reach the **critical angle**:

$$\boxed{\theta_c = \arcsin\left(\frac{n_2}{n_1}\right)}$$

For $\theta_I > \theta_c$, Snell's law has no real solution for $\theta_T$, and the wave is **totally internally reflected**.

### 5.2 What Happens During TIR?

Despite the name "total" reflection, the field does not abruptly stop at the interface. Instead:

- $|r_s| = |r_p| = 1$ — all incident power is reflected
- The transmission coefficient $t$ is not zero; there is an **evanescent wave** in medium 2
- The Fresnel coefficients become complex, introducing a **phase shift** on reflection

### 5.3 Evanescent Waves

Beyond the critical angle, $\cos\theta_T$ becomes purely imaginary:

$$\cos\theta_T = i\sqrt{\left(\frac{n_1}{n_2}\right)^2 \sin^2\theta_I - 1}$$

The transmitted field becomes:

$$\mathbf{E}_T \propto e^{ik_x x} e^{-\kappa z}$$

where $\kappa = k_2\sqrt{(n_1/n_2)^2\sin^2\theta_I - 1}$ is the decay constant. This is an **evanescent wave**: it propagates along the interface ($x$-direction) but decays exponentially away from it ($z$-direction).

The penetration depth is:

$$d_p = \frac{1}{\kappa} = \frac{\lambda_2}{2\pi\sqrt{(n_1/n_2)^2\sin^2\theta_I - 1}}$$

Typically $d_p \sim \lambda$ — the evanescent field extends about one wavelength into medium 2.

> **Analogy**: Evanescent waves are like the sound that leaks through a wall. If the wall is thin enough (comparable to the penetration depth), some energy can tunnel through to the other side. This is **frustrated total internal reflection**, the optical analog of quantum tunneling.

```python
def plot_evanescent_wave(n1, n2, theta_i_deg, wavelength=500e-9):
    """
    Visualize the evanescent wave field beyond the critical angle.

    Why this matters: evanescent waves are the basis of
    TIRF microscopy, fiber optic sensors, and near-field optics.
    """
    theta_c = np.degrees(np.arcsin(n2 / n1))
    theta_i = np.radians(theta_i_deg)
    k2 = 2 * np.pi * n2 / wavelength

    # Decay constant in medium 2
    kappa = k2 * np.sqrt((n1 / n2)**2 * np.sin(theta_i)**2 - 1)
    # Lateral wave number
    kx = k2 * (n1 / n2) * np.sin(theta_i)

    # Penetration depth
    d_p = 1.0 / kappa

    # Create spatial grid
    x = np.linspace(0, 3 * wavelength, 300)
    z = np.linspace(-2 * wavelength, 3 * wavelength, 400)
    X, Z = np.meshgrid(x, z)

    # Field in medium 1 (z < 0): incident + reflected
    # For simplicity, show only the evanescent field in medium 2
    E = np.zeros_like(X, dtype=complex)

    # Medium 2 (z > 0): evanescent wave
    mask2 = Z > 0
    E[mask2] = np.exp(1j * kx * X[mask2]) * np.exp(-kappa * Z[mask2])

    # Medium 1 (z < 0): simple standing wave pattern
    k1 = 2 * np.pi * n1 / wavelength
    kz1 = k1 * np.cos(theta_i)
    mask1 = Z <= 0
    E[mask1] = (np.exp(1j * (kx * X[mask1] + kz1 * Z[mask1])) +
                np.exp(1j * (kx * X[mask1] - kz1 * Z[mask1])))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Real part of E-field
    im = axes[0].pcolormesh(X * 1e9, Z * 1e9, E.real, cmap='RdBu_r',
                             shading='auto', vmin=-2, vmax=2)
    axes[0].axhline(y=0, color='white', linewidth=2)
    axes[0].set_xlabel('x (nm)')
    axes[0].set_ylabel('z (nm)')
    axes[0].set_title(f'E-field (TIR at {theta_i_deg}°, $\\theta_c$ = {theta_c:.1f}°)')
    plt.colorbar(im, ax=axes[0], label='Re(E)')

    # Intensity vs z at x=0
    z_line = np.linspace(-wavelength, 3 * wavelength, 500)
    I_med1 = np.ones_like(z_line[z_line <= 0])
    I_med2 = np.exp(-2 * kappa * z_line[z_line > 0])

    axes[1].plot(z_line[z_line <= 0] * 1e9, I_med1, 'b-', linewidth=2,
                 label='Medium 1')
    axes[1].plot(z_line[z_line > 0] * 1e9, I_med2, 'r-', linewidth=2,
                 label='Medium 2 (evanescent)')
    axes[1].axvline(x=0, color='black', linewidth=2, label='Interface')
    axes[1].axhline(y=np.exp(-2), color='gray', linestyle='--', alpha=0.5,
                     label=f'$1/e^2$ depth = {d_p*1e9:.0f} nm')
    axes[1].set_xlabel('z (nm)')
    axes[1].set_ylabel('Intensity (normalized)')
    axes[1].set_title('Evanescent Field Decay')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("evanescent_wave.png", dpi=150)
    plt.show()

    print(f"Critical angle: {theta_c:.1f}°")
    print(f"Penetration depth: {d_p*1e9:.1f} nm")

# Glass to air, beyond critical angle
plot_evanescent_wave(n1=1.5, n2=1.0, theta_i_deg=45, wavelength=500e-9)
```

---

## 6. Anti-Reflection Coatings

### 6.1 The Problem

A single air-glass interface reflects about 4% of the incident light. A camera lens with 10 surfaces loses nearly 34% of the light to reflections (plus ghost images from multiple reflections). Anti-reflection (AR) coatings dramatically reduce these losses.

### 6.2 Single-Layer Quarter-Wave Coating

The simplest AR coating is a thin film of thickness $d$ and refractive index $n_c$ on a substrate of index $n_s$. Destructive interference between reflections from the top and bottom of the film cancels the reflected wave when:

**Thickness condition** (quarter-wave):

$$d = \frac{\lambda}{4 n_c}$$

**Index matching condition** (zero reflection at normal incidence):

$$n_c = \sqrt{n_1 \cdot n_s}$$

For air ($n_1 = 1$) on glass ($n_s = 1.5$): $n_c = \sqrt{1.5} \approx 1.225$. MgF$_2$ ($n = 1.38$) is the closest common material — not perfect, but it reduces reflectance from 4% to about 1.3%.

### 6.3 Transfer Matrix Method

For multilayer coatings, the transfer matrix method is the standard tool. Each layer $j$ of thickness $d_j$ and refractive index $n_j$ contributes a 2x2 matrix:

$$M_j = \begin{pmatrix} \cos\delta_j & -i\sin\delta_j / \eta_j \\ -i\eta_j \sin\delta_j & \cos\delta_j \end{pmatrix}$$

where $\delta_j = 2\pi n_j d_j \cos\theta_j / \lambda$ is the phase thickness, and $\eta_j = n_j \cos\theta_j$ for s-polarization ($\eta_j = n_j / \cos\theta_j$ for p-polarization).

The total system matrix is $M = M_1 M_2 \cdots M_N$, and the reflection coefficient is:

$$r = \frac{(M_{11} + M_{12}\eta_s)\eta_0 - (M_{21} + M_{22}\eta_s)}{(M_{11} + M_{12}\eta_s)\eta_0 + (M_{21} + M_{22}\eta_s)}$$

```python
def transfer_matrix_reflectance(n_layers, d_layers, n_substrate,
                                 wavelengths, theta_i=0, polarization='s'):
    """
    Compute reflectance of a multilayer thin film using the transfer matrix method.

    Parameters:
        n_layers    : list of refractive indices [n_1, n_2, ...] (can be complex)
        d_layers    : list of layer thicknesses [d_1, d_2, ...] in meters
        n_substrate : substrate refractive index
        wavelengths : array of wavelengths (m)
        theta_i     : angle of incidence (rad)
        polarization: 's' or 'p'

    Why transfer matrices: they compose naturally for any number of layers,
    handle interference exactly, and extend to complex refractive indices
    for absorbing films.
    """
    n0 = 1.0  # ambient medium (air)
    R = np.zeros(len(wavelengths))

    for idx, lam in enumerate(wavelengths):
        # Snell's law in each layer
        sin_i = n0 * np.sin(theta_i)

        # Admittances depend on polarization
        cos_i = np.sqrt(1 - (sin_i / n0)**2 + 0j)
        cos_sub = np.sqrt(1 - (sin_i / n_substrate)**2 + 0j)

        if polarization == 's':
            eta_0 = n0 * cos_i
            eta_s = n_substrate * cos_sub
        else:
            eta_0 = n0 / cos_i
            eta_s = n_substrate / cos_sub

        # Build total transfer matrix
        M = np.eye(2, dtype=complex)
        for n_j, d_j in zip(n_layers, d_layers):
            cos_j = np.sqrt(1 - (sin_i / n_j)**2 + 0j)
            delta_j = 2 * np.pi * n_j * d_j * cos_j / lam

            if polarization == 's':
                eta_j = n_j * cos_j
            else:
                eta_j = n_j / cos_j

            layer_matrix = np.array([
                [np.cos(delta_j), -1j * np.sin(delta_j) / eta_j],
                [-1j * eta_j * np.sin(delta_j), np.cos(delta_j)]
            ])
            M = M @ layer_matrix

        # Reflection coefficient
        r = ((M[0, 0] + M[0, 1] * eta_s) * eta_0 -
             (M[1, 0] + M[1, 1] * eta_s)) / \
            ((M[0, 0] + M[0, 1] * eta_s) * eta_0 +
             (M[1, 0] + M[1, 1] * eta_s))

        R[idx] = np.abs(r)**2

    return R

# Compare: uncoated glass vs single-layer MgF2 vs ideal quarter-wave
wavelengths = np.linspace(350e-9, 800e-9, 500)
n_glass = 1.52
lambda_design = 550e-9  # design wavelength (green)

# Uncoated
R_bare = np.abs((1 - n_glass) / (1 + n_glass))**2 * np.ones(len(wavelengths))

# MgF2 coating (n=1.38)
n_MgF2 = 1.38
d_MgF2 = lambda_design / (4 * n_MgF2)
R_MgF2 = transfer_matrix_reflectance([n_MgF2], [d_MgF2], n_glass, wavelengths)

# Ideal quarter-wave (n = sqrt(n_glass))
n_ideal = np.sqrt(n_glass)
d_ideal = lambda_design / (4 * n_ideal)
R_ideal = transfer_matrix_reflectance([n_ideal], [d_ideal], n_glass, wavelengths)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(wavelengths * 1e9, R_bare * 100, 'k--', linewidth=1.5, label='Uncoated')
ax.plot(wavelengths * 1e9, R_MgF2 * 100, 'b-', linewidth=2,
        label=f'MgF$_2$ (n={n_MgF2})')
ax.plot(wavelengths * 1e9, R_ideal * 100, 'r-', linewidth=2,
        label=f'Ideal (n={n_ideal:.3f})')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Reflectance (%)')
ax.set_title('Anti-Reflection Coating Performance')
ax.set_ylim(0, 5)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Shade visible spectrum
ax.axvspan(380, 750, alpha=0.05, color='yellow')
ax.text(565, 4.5, 'Visible', ha='center', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig("antireflection_coating.png", dpi=150)
plt.show()

print(f"Design wavelength: {lambda_design*1e9:.0f} nm")
print(f"MgF2 thickness: {d_MgF2*1e9:.1f} nm")
print(f"MgF2 reflectance at design λ: {R_MgF2[len(wavelengths)//2]*100:.2f}%")
```

---

## 7. Comprehensive Visualization

Let us create a complete visualization showing all the phenomena discussed in this lesson.

```python
def comprehensive_fresnel_plot():
    """
    Complete Fresnel analysis for both external and internal reflection.

    Why both cases matter: external reflection (air→glass) is relevant for
    camera lenses and windows, while internal reflection (glass→air) gives
    total internal reflection, essential for fiber optics and prisms.
    """
    theta = np.linspace(0, np.pi / 2 - 0.001, 1000)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- External reflection: air → glass ---
    n1, n2 = 1.0, 1.5
    r_s, r_p, _, _ = fresnel_coefficients(n1, n2, theta)
    R_s, R_p = np.abs(r_s)**2, np.abs(r_p)**2
    theta_B = np.arctan(n2 / n1)

    axes[0, 0].plot(np.degrees(theta), R_s, 'b-', linewidth=2, label='$R_s$')
    axes[0, 0].plot(np.degrees(theta), R_p, 'r-', linewidth=2, label='$R_p$')
    axes[0, 0].axvline(x=np.degrees(theta_B), color='green', linestyle=':',
                        label=f'$\\theta_B$ = {np.degrees(theta_B):.1f}°')
    axes[0, 0].set_title(f'External: Air → Glass ($n_2/n_1$ = {n2/n1:.1f})')
    axes[0, 0].set_ylabel('Reflectance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)

    # Phase on reflection (external)
    axes[0, 1].plot(np.degrees(theta), np.angle(r_s) / np.pi, 'b-', linewidth=2,
                     label='$\\phi_s / \\pi$')
    axes[0, 1].plot(np.degrees(theta), np.angle(r_p) / np.pi, 'r-', linewidth=2,
                     label='$\\phi_p / \\pi$')
    axes[0, 1].set_title('Phase Shift on External Reflection')
    axes[0, 1].set_ylabel('Phase / $\\pi$')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # --- Internal reflection: glass → air ---
    n1, n2 = 1.5, 1.0
    r_s, r_p, _, _ = fresnel_coefficients(n1, n2, theta)
    R_s, R_p = np.abs(r_s)**2, np.abs(r_p)**2
    theta_c = np.arcsin(n2 / n1)
    theta_B = np.arctan(n2 / n1)

    axes[1, 0].plot(np.degrees(theta), R_s, 'b-', linewidth=2, label='$R_s$')
    axes[1, 0].plot(np.degrees(theta), R_p, 'r-', linewidth=2, label='$R_p$')
    axes[1, 0].axvline(x=np.degrees(theta_c), color='purple', linestyle='--',
                        label=f'$\\theta_c$ = {np.degrees(theta_c):.1f}°')
    axes[1, 0].axvline(x=np.degrees(theta_B), color='green', linestyle=':',
                        label=f'$\\theta_B$ = {np.degrees(theta_B):.1f}°')
    axes[1, 0].set_title(f'Internal: Glass → Air ($n_1/n_2$ = {n1/n2:.1f})')
    axes[1, 0].set_xlabel('Angle of incidence (degrees)')
    axes[1, 0].set_ylabel('Reflectance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)

    # Phase on reflection (internal) — shows Goos-Hanchen-like phase shifts
    axes[1, 1].plot(np.degrees(theta), np.angle(r_s) / np.pi, 'b-', linewidth=2,
                     label='$\\phi_s / \\pi$')
    axes[1, 1].plot(np.degrees(theta), np.angle(r_p) / np.pi, 'r-', linewidth=2,
                     label='$\\phi_p / \\pi$')
    axes[1, 1].axvline(x=np.degrees(theta_c), color='purple', linestyle='--',
                        label=f'$\\theta_c$ = {np.degrees(theta_c):.1f}°')
    axes[1, 1].set_title('Phase Shift on Internal Reflection')
    axes[1, 1].set_xlabel('Angle of incidence (degrees)')
    axes[1, 1].set_ylabel('Phase / $\\pi$')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fresnel_comprehensive.png", dpi=150)
    plt.show()

comprehensive_fresnel_plot()
```

---

## Summary

| Concept | Key Formula | Physical Significance |
|---------|-------------|----------------------|
| Snell's law | $n_1 \sin\theta_I = n_2 \sin\theta_T$ | Phase matching at interface |
| Fresnel $r_s$ | $(n_1\cos\theta_I - n_2\cos\theta_T)/(n_1\cos\theta_I + n_2\cos\theta_T)$ | s-polarized reflection amplitude |
| Fresnel $r_p$ | $(n_2\cos\theta_I - n_1\cos\theta_T)/(n_2\cos\theta_I + n_1\cos\theta_T)$ | p-polarized reflection amplitude |
| Brewster's angle | $\tan\theta_B = n_2/n_1$ | Zero p-polarized reflection |
| Critical angle | $\sin\theta_c = n_2/n_1$ | Onset of total internal reflection |
| Penetration depth | $d_p = \lambda/(2\pi\sqrt{(n_1/n_2)^2\sin^2\theta_I - 1})$ | Evanescent wave extent |
| Quarter-wave coating | $d = \lambda/(4n_c)$, $n_c = \sqrt{n_s}$ | Zero reflection at design $\lambda$ |

---

## Exercises

### Exercise 1: Diamond Brilliance
Diamond has $n = 2.42$. (a) Calculate the critical angle for diamond-air. (b) Compute the Brewster angle from air into diamond. (c) Plot $R_s$ and $R_p$ for 0-90 degrees. (d) Explain why diamonds appear so brilliant (hint: the small critical angle means most light entering the top is totally internally reflected by the facets).

### Exercise 2: Multilayer Dielectric Mirror
Design a 5-layer dielectric mirror (alternating high/low index layers: $n_H = 2.3$, $n_L = 1.38$, on glass $n_s = 1.52$) that achieves $R > 99\%$ at $\lambda = 1064$ nm (Nd:YAG laser). Use quarter-wave thicknesses for each layer. Plot $R(\lambda)$ from 800 to 1300 nm using the transfer matrix method.

### Exercise 3: Frustrated Total Internal Reflection
Two glass prisms ($n = 1.5$) are separated by an air gap of variable width $d$. A beam hits the first prism at 45 degrees (above the critical angle). Using the transfer matrix method (two interfaces with an air gap layer), compute the transmittance $T$ as a function of $d/\lambda$ from 0 to 3. Verify that $T \to 1$ when $d \to 0$ and $T \to 0$ when $d \gg \lambda$.

### Exercise 4: Polarization by Reflection
Unpolarized light reflects off a glass window ($n = 1.5$) at various angles. (a) Plot the degree of polarization $P = (R_s - R_p)/(R_s + R_p)$ as a function of angle. (b) At what angle is $P$ maximized? (c) Compute $P$ at 60 degrees and explain why polarized sunglasses are effective.

---

[← Previous: 10. EM Waves in Matter](10_EM_Waves_Matter.md) | [Next: 12. Waveguides and Cavities →](12_Waveguides_and_Cavities.md)
