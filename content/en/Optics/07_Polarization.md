# 07. Polarization

[← Previous: 06. Diffraction](06_Diffraction.md) | [Next: 08. Laser Fundamentals →](08_Laser_Fundamentals.md)

---

## Learning Objectives

1. Describe the three types of polarization — linear, circular, and elliptical — and their mathematical representations
2. Apply Malus's law to calculate the transmitted intensity through a sequence of polarizers
3. Explain polarization by reflection (Brewster's angle), scattering, birefringence, and dichroism
4. Analyze the effects of quarter-wave and half-wave plates on polarization states
5. Use Jones vectors and Jones matrices to solve problems with polarizing optical elements
6. Describe optical activity, the Faraday effect, and their physical origins
7. Explain how LCD displays, polarizing sunglasses, and 3D cinema exploit polarization

---

## Why This Matters

Polarization is the "hidden dimension" of light that is invisible to the naked eye but profoundly useful in technology. Liquid crystal displays (LCDs), 3D cinema glasses, optical fiber communications, laser radar (lidar), stress analysis of transparent materials, and astronomical instruments for measuring cosmic magnetic fields all rely on controlling or measuring polarization. Understanding polarization also deepens your grasp of light as a transverse electromagnetic wave — the direction of the electric field oscillation is not just a theoretical detail but a physical property that can be manipulated, measured, and exploited.

> **Analogy**: Think of a rope threaded through a picket fence. If you shake the rope up-and-down, the vertical oscillation passes through the fence's vertical slats. But if you shake it side-to-side, the horizontal oscillation is blocked. A polarizer works the same way for light: it transmits the component of the electric field aligned with its transmission axis and blocks the perpendicular component. The "picket fence" for light is made of aligned molecules or metal nanowires.

---

## 1. What Is Polarization?

### 1.1 Transverse Nature of Light

Light is a transverse electromagnetic wave: the electric field $\mathbf{E}$ oscillates perpendicular to the direction of propagation. For a wave propagating in the $z$-direction:

$$\mathbf{E}(z, t) = E_x(z, t)\,\hat{\mathbf{x}} + E_y(z, t)\,\hat{\mathbf{y}}$$

The **polarization** of light describes the trajectory that the tip of the $\mathbf{E}$ vector traces as the wave passes a fixed point. This trajectory depends on the amplitudes and phases of $E_x$ and $E_y$.

### 1.2 The General Case: Elliptical Polarization

In general:

$$E_x = E_{0x}\cos(\omega t - kz)$$
$$E_y = E_{0y}\cos(\omega t - kz + \delta)$$

where $E_{0x}$ and $E_{0y}$ are the amplitudes and $\delta$ is the **phase difference** between the $x$ and $y$ components. The tip of $\mathbf{E}$ traces an ellipse — hence **elliptical polarization** is the most general polarization state.

Special cases of the phase difference $\delta$:

| $\delta$ | Polarization |
|----------|-------------|
| $0$ or $\pi$ | **Linear** (along a fixed direction) |
| $\pm\pi/2$ with $E_{0x} = E_{0y}$ | **Circular** |
| Any other $\delta$ | **Elliptical** (general case) |

### 1.3 Types of Polarization

**Linear polarization**: $\mathbf{E}$ oscillates along a fixed line in the $xy$-plane. The orientation angle $\alpha$:

$$\tan\alpha = \frac{E_{0y}}{E_{0x}} \quad (\text{when } \delta = 0)$$

**Circular polarization**: The $\mathbf{E}$ vector rotates at constant magnitude, tracing a circle. Two handedness conventions:
- **Right-circular (RCP)**: $\delta = -\pi/2$ — $\mathbf{E}$ rotates clockwise when viewed facing the oncoming wave (IEEE convention)
- **Left-circular (LCP)**: $\delta = +\pi/2$ — $\mathbf{E}$ rotates counterclockwise

**Unpolarized light**: The polarization direction changes randomly on timescales much shorter than the measurement time. Natural light (sunlight, incandescent bulbs, LEDs) is typically unpolarized.

**Partially polarized light**: A mixture of polarized and unpolarized components, characterized by the **degree of polarization** $P$ ($0 \leq P \leq 1$).

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize different polarization states by tracing the E-field vector tip

omega_t = np.linspace(0, 2*np.pi, 500)  # one full cycle

# Define polarization states: (E_0x, E_0y, delta, label)
states = [
    (1.0, 0.0, 0, 'Linear (horizontal)'),
    (1.0, 1.0, 0, 'Linear (45°)'),
    (1.0, 1.0, -np.pi/2, 'Right circular'),
    (1.0, 1.0, np.pi/2, 'Left circular'),
    (1.0, 0.5, np.pi/4, 'Elliptical'),
    (1.0, 0.7, np.pi/3, 'Elliptical (general)'),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, (E0x, E0y, delta, label) in zip(axes.flat, states):
    # Electric field components
    Ex = E0x * np.cos(omega_t)
    Ey = E0y * np.cos(omega_t + delta)

    # Plot the polarization ellipse (trajectory of E-vector tip)
    ax.plot(Ex, Ey, 'b-', linewidth=2)

    # Mark the starting point and direction
    ax.plot(Ex[0], Ey[0], 'ro', markersize=8, zorder=5, label='Start')
    # Arrow showing direction of rotation
    mid = len(omega_t) // 8
    ax.annotate('', xy=(Ex[mid], Ey[mid]), xytext=(Ex[mid-5], Ey[mid-5]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_xlabel('$E_x$', fontsize=11)
    ax.set_ylabel('$E_y$', fontsize=11)
    ax.set_title(f'{label}\n$E_{{0x}}$={E0x}, $E_{{0y}}$={E0y}, $\\delta$={delta/np.pi:.2f}π',
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

plt.suptitle('Polarization States of Light', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('polarization_states.png', dpi=150)
plt.show()
```

---

## 2. Polarizers and Malus's Law

### 2.1 Ideal Linear Polarizer

An ideal linear polarizer transmits only the component of $\mathbf{E}$ along its **transmission axis** and completely absorbs the perpendicular component.

If linearly polarized light with intensity $I_0$ and polarization angle $\theta_0$ strikes a polarizer with transmission axis at angle $\theta$, the transmitted intensity is:

$$I = I_0 \cos^2(\theta - \theta_0)$$

This is **Malus's law** (Etienne-Louis Malus, 1809).

**Special cases**:
- Parallel ($\theta = \theta_0$): $I = I_0$ (full transmission)
- Perpendicular ($\theta - \theta_0 = 90°$): $I = 0$ (full extinction)
- 45°: $I = I_0/2$

### 2.2 Unpolarized Light Through a Polarizer

For unpolarized light, the polarization angle is random. Averaging Malus's law over all angles:

$$I = I_0 \langle\cos^2\theta\rangle = \frac{I_0}{2}$$

An ideal polarizer transmits exactly half the intensity of unpolarized light.

### 2.3 Multiple Polarizers

The "crossed polarizer" paradox: Two crossed polarizers (transmission axes at 90°) block all light. But inserting a third polarizer at 45° between them allows some light through:

1. Unpolarized light $I_0$ → first polarizer (0°): $I_1 = I_0/2$ (polarized at 0°)
2. Through 45° polarizer: $I_2 = I_1\cos^2 45° = I_0/4$ (polarized at 45°)
3. Through 90° polarizer: $I_3 = I_2\cos^2 45° = I_0/8$ (polarized at 90°)

The 45° polarizer "rotates" the polarization by 45° at each step, allowing transmission through the formerly opaque pair. This works because measurement (projection) alters the polarization state.

```python
import numpy as np
import matplotlib.pyplot as plt

# Malus's Law: transmitted intensity through two polarizers
# as a function of the angle between their transmission axes

theta_deg = np.linspace(0, 360, 500)
theta_rad = np.deg2rad(theta_deg)

# Malus's law for initially polarized light
I_malus = np.cos(theta_rad)**2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Malus's law
ax1.plot(theta_deg, I_malus, 'b-', linewidth=2, label="$I = I_0 \\cos^2\\theta$")
ax1.set_xlabel('Angle between polarizers $\\theta$ (degrees)', fontsize=12)
ax1.set_ylabel('Transmitted Intensity $I / I_0$', fontsize=12)
ax1.set_title("Malus's Law", fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(np.arange(0, 361, 45))

# Right: N polarizers at equal angular increments from 0 to 90 degrees
# As N increases, more light gets through!
N_values = range(1, 11)
transmission = []

for N in N_values:
    # N polarizers equally spaced from 0 to 90 degrees
    # Each step rotates by 90/N degrees
    step_angle = np.deg2rad(90 / N)
    # Starting with polarized light (after first polarizer from unpolarized: I_0/2)
    I = 0.5  # after first polarizer
    for _ in range(N):
        I *= np.cos(step_angle)**2
    transmission.append(I)

ax2.bar(list(N_values), transmission, color='steelblue', edgecolor='navy', alpha=0.8)
ax2.set_xlabel('Number of intermediate polarizers N', fontsize=12)
ax2.set_ylabel('Final transmitted intensity $I / I_0$', fontsize=12)
ax2.set_title('Transmission through N+1 polarizers\n(spanning 0° to 90° in equal steps)', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

# Annotate key values
for N, T in zip(N_values, transmission):
    ax2.text(N, T + 0.01, f'{T:.3f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('malus_law.png', dpi=150)
plt.show()
```

---

## 3. Polarization Mechanisms

### 3.1 Polarization by Reflection: Brewster's Angle

When unpolarized light strikes a dielectric surface, the reflected light is partially polarized. At a specific angle — **Brewster's angle** $\theta_B$ — the reflected light is completely polarized (with $\mathbf{E}$ perpendicular to the plane of incidence, i.e., $s$-polarized):

$$\tan\theta_B = \frac{n_2}{n_1}$$

At Brewster's angle, the reflected and refracted rays are perpendicular ($\theta_B + \theta_t = 90°$).

| Interface | Brewster's Angle |
|-----------|-----------------|
| Air → Glass ($n=1.5$) | 56.3° |
| Air → Water ($n=1.33$) | 53.1° |
| Air → Diamond ($n=2.42$) | 67.5° |

**Why it works**: At Brewster's angle, the oscillating dipoles in the surface are oriented along the reflected ray direction. Since dipoles do not radiate along their oscillation axis, the $p$-polarized (parallel) component has zero reflection.

This is why polarizing sunglasses are effective at reducing glare from roads and water surfaces — the reflected light from these surfaces is strongly $s$-polarized, and polarizing sunglasses are oriented to block $s$-polarization.

### 3.2 Polarization by Scattering

When light scatters from particles much smaller than the wavelength (Rayleigh scattering), the scattered light is polarized. The degree of polarization depends on the scattering angle:
- 0° and 180° (forward/backward): unpolarized
- 90° (perpendicular): fully polarized

The sky is partially polarized, with maximum polarization at 90° from the sun. Bees and many other insects can detect this polarization and use it for navigation.

### 3.3 Polarization by Birefringence

**Birefringent** (doubly refracting) crystals have different refractive indices for different polarization directions. When unpolarized light enters such a crystal, it splits into two beams:

- **Ordinary ray** ($o$-ray): Obeys Snell's law normally, refractive index $n_o$
- **Extraordinary ray** ($e$-ray): Does not obey Snell's law in general, refractive index $n_e$ (which varies with direction)

The two rays are orthogonally polarized.

| Crystal | $n_o$ | $n_e$ | $\Delta n = n_e - n_o$ | Type |
|---------|-------|-------|----------------------|------|
| Calcite (CaCO$_3$) | 1.658 | 1.486 | -0.172 | Negative |
| Quartz (SiO$_2$) | 1.544 | 1.553 | +0.009 | Positive |
| Mica | 1.599 | 1.594 | -0.005 | Negative |
| Rutile (TiO$_2$) | 2.616 | 2.903 | +0.287 | Positive |

The direction in which $n_o = n_e$ is called the **optic axis**. Light propagating along the optic axis experiences no birefringence.

### 3.4 Polarization by Dichroism

**Dichroic** materials absorb one polarization more strongly than the other. Polaroid film (invented by Edwin Land in 1928) contains aligned polyvinyl alcohol molecules that absorb light polarized along the molecular chains while transmitting the perpendicular polarization.

Modern wire-grid polarizers use nanoscale metal wires on a substrate — the $\mathbf{E}$ component parallel to the wires drives currents and is absorbed; the perpendicular component passes through.

---

## 4. Wave Plates (Retarders)

### 4.1 Principle

A wave plate is a birefringent crystal cut so that the optic axis lies in the plane of the plate. Light passing through it experiences different phase velocities for the two polarization components, introducing a controlled phase difference:

$$\delta = \frac{2\pi}{\lambda}(n_e - n_o) \cdot t$$

where $t$ is the plate thickness. The two polarization directions in the plate are called the **fast axis** (lower $n$) and **slow axis** (higher $n$).

### 4.2 Quarter-Wave Plate ($\lambda/4$)

A quarter-wave plate introduces a phase difference of $\delta = \pi/2$ (90°):

$$t = \frac{\lambda}{4|n_e - n_o|}$$

**Key transformations**:
- Linear (45° to fast axis) → Circular: Input at 45° gives equal amplitudes on both axes with $\pi/2$ phase difference = circular polarization
- Circular → Linear: Conversely, circular polarization becomes linear after a $\lambda/4$ plate

### 4.3 Half-Wave Plate ($\lambda/2$)

A half-wave plate introduces $\delta = \pi$ (180°):

$$t = \frac{\lambda}{2|n_e - n_o|}$$

**Key transformations**:
- Rotates linear polarization: If input polarization makes angle $\alpha$ with the fast axis, the output polarization makes angle $-\alpha$ — the polarization is "reflected" about the fast axis, effectively rotated by $2\alpha$
- Reverses circular handedness: RCP → LCP and vice versa

```python
import numpy as np
import matplotlib.pyplot as plt

# Effect of wave plates on polarization state
# Visualize the transformation of the E-field tip trajectory

omega_t = np.linspace(0, 2*np.pi, 500)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Define input and output states for different wave plate configurations
configs = [
    # (title, E0x_in, E0y_in, delta_in, retardation, description)
    ('Input: Linear 45°', 1, 1, 0, 0, 'No plate'),
    ('λ/4 plate → RCP', 1, 1, 0, -np.pi/2, 'Quarter-wave'),
    ('λ/2 plate → Linear -45°', 1, 1, 0, -np.pi, 'Half-wave'),
    ('Input: RCP', 1, 1, -np.pi/2, 0, 'No plate'),
    ('λ/4 plate → Linear', 1, 1, -np.pi/2, -np.pi/2, 'Quarter-wave'),
    ('λ/2 plate → LCP', 1, 1, -np.pi/2, -np.pi, 'Half-wave'),
]

for ax, (title, E0x, E0y, delta_in, retard, desc) in zip(axes.flat, configs):
    # Input state
    Ex_in = E0x * np.cos(omega_t)
    Ey_in = E0y * np.cos(omega_t + delta_in)

    # Output state: add retardation to the y-component (assuming fast axis = x)
    delta_out = delta_in + retard
    Ex_out = E0x * np.cos(omega_t)
    Ey_out = E0y * np.cos(omega_t + delta_out)

    # Plot input (gray dashed) and output (blue solid)
    ax.plot(Ex_in, Ey_in, 'gray', linewidth=1, linestyle='--', alpha=0.5, label='Input')
    ax.plot(Ex_out, Ey_out, 'b-', linewidth=2, label='Output')

    # Direction arrow on output
    mid = len(omega_t) // 8
    ax.annotate('', xy=(Ex_out[mid], Ey_out[mid]),
                xytext=(Ex_out[mid-5], Ey_out[mid-5]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_xlabel('$E_x$', fontsize=10)
    ax.set_ylabel('$E_y$', fontsize=10)
    ax.set_title(f'{title}\n({desc})', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.legend(fontsize=8, loc='upper right')

plt.suptitle('Wave Plate Effects on Polarization', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wave_plate_effects.png', dpi=150)
plt.show()
```

---

## 5. Jones Vector and Jones Matrix Formalism

### 5.1 Jones Vectors

The **Jones vector** represents the polarization state of a fully polarized monochromatic wave as a 2D complex vector:

$$\mathbf{J} = \begin{pmatrix} E_{0x} \\ E_{0y} e^{i\delta} \end{pmatrix}$$

Common Jones vectors (normalized to unit intensity):

| Polarization State | Jones Vector |
|-------------------|-------------|
| Horizontal ($x$) | $\begin{pmatrix} 1 \\ 0 \end{pmatrix}$ |
| Vertical ($y$) | $\begin{pmatrix} 0 \\ 1 \end{pmatrix}$ |
| Linear at $+45°$ | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$ |
| Linear at $-45°$ | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$ |
| Right circular (RCP) | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -i \end{pmatrix}$ |
| Left circular (LCP) | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ i \end{pmatrix}$ |

The intensity is $I = |\mathbf{J}|^2 = |E_{0x}|^2 + |E_{0y}|^2$.

### 5.2 Jones Matrices

Each polarizing optical element is represented by a $2 \times 2$ **Jones matrix** $\mathbf{M}$. The output Jones vector is:

$$\mathbf{J}_{\text{out}} = \mathbf{M} \cdot \mathbf{J}_{\text{in}}$$

For a sequence of elements, multiply the matrices from right to left:

$$\mathbf{J}_{\text{out}} = \mathbf{M}_N \cdots \mathbf{M}_2 \cdot \mathbf{M}_1 \cdot \mathbf{J}_{\text{in}}$$

**Common Jones matrices** (for elements with principal axes aligned with $x$ and $y$):

| Element | Jones Matrix |
|---------|-------------|
| Horizontal polarizer | $\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$ |
| Vertical polarizer | $\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$ |
| Quarter-wave plate (fast axis horizontal) | $e^{-i\pi/4}\begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$ |
| Half-wave plate (fast axis horizontal) | $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ |
| General wave plate (retardation $\Gamma$) | $\begin{pmatrix} 1 & 0 \\ 0 & e^{i\Gamma} \end{pmatrix}$ |

### 5.3 Rotation of Elements

To represent an element rotated by angle $\alpha$ from the $x$-axis, apply the rotation:

$$\mathbf{M}(\alpha) = \mathbf{R}(-\alpha) \cdot \mathbf{M}(0) \cdot \mathbf{R}(\alpha)$$

where the rotation matrix is:

$$\mathbf{R}(\alpha) = \begin{pmatrix} \cos\alpha & \sin\alpha \\ -\sin\alpha & \cos\alpha \end{pmatrix}$$

For a **linear polarizer at angle** $\alpha$:

$$\mathbf{M}_{\text{pol}}(\alpha) = \begin{pmatrix} \cos^2\alpha & \sin\alpha\cos\alpha \\ \sin\alpha\cos\alpha & \sin^2\alpha \end{pmatrix}$$

```python
import numpy as np

# Jones calculus: solve a multi-element polarization problem
# Example: Light through polarizer → quarter-wave plate → analyzer

def jones_polarizer(angle_deg):
    """Jones matrix for a linear polarizer at angle alpha from horizontal."""
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c**2, s*c],
                     [s*c, s**2]])

def jones_waveplate(retardation, angle_deg):
    """
    Jones matrix for a wave plate with given retardation (radians),
    fast axis at angle_deg from horizontal.
    """
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), np.sin(a)],
                  [-np.sin(a), np.cos(a)]])
    R_inv = np.array([[np.cos(a), -np.sin(a)],
                      [np.sin(a), np.cos(a)]])
    # Wave plate with fast axis along x
    W = np.array([[1, 0],
                  [0, np.exp(1j * retardation)]])
    return R_inv @ W @ R

def jones_intensity(J):
    """Calculate intensity from a Jones vector."""
    return float(np.abs(J[0])**2 + np.abs(J[1])**2)

# Problem: Unpolarized light approximated as horizontal
# → Vertical polarizer → Quarter-wave plate at 45° → Horizontal analyzer

print("Jones Calculus Example")
print("=" * 50)

# Step 1: Start with horizontally polarized light
J_in = np.array([1.0, 0.0], dtype=complex)
print(f"Input: {J_in}  (horizontal, I = {jones_intensity(J_in):.3f})")

# Step 2: Pass through vertical polarizer
M_vpol = jones_polarizer(90)
J_after_vpol = M_vpol @ J_in
print(f"After V-polarizer: {J_after_vpol}  (I = {jones_intensity(J_after_vpol):.3f})")
print("  → Blocked! (crossed polarizers)")

print()
print("Now insert a quarter-wave plate at 45° between crossed polarizers:")

# Step 1: Horizontal polarizer
M_hpol = jones_polarizer(0)
J1 = M_hpol @ np.array([1.0, 0.0], dtype=complex)
print(f"After H-polarizer: {J1}  (I = {jones_intensity(J1):.3f})")

# Step 2: Quarter-wave plate at 45°
M_qwp_45 = jones_waveplate(np.pi/2, 45)
J2 = M_qwp_45 @ J1
print(f"After QWP at 45°: [{J2[0]:.4f}, {J2[1]:.4f}]  (I = {jones_intensity(J2):.3f})")
# This should be circularly polarized

# Step 3: Vertical polarizer (analyzer)
M_vpol = jones_polarizer(90)
J3 = M_vpol @ J2
print(f"After V-polarizer: [{J3[0]:.4f}, {J3[1]:.4f}]  (I = {jones_intensity(J3):.3f})")
print(f"  → Transmission: {jones_intensity(J3)*100:.1f}% of input")

print()
print("Verify Malus's law with rotation:")
print("-" * 40)
# Rotate the analyzer and check intensity
for angle in range(0, 181, 15):
    M_analyzer = jones_polarizer(angle)
    J_out = M_analyzer @ M_qwp_45 @ J1
    I = jones_intensity(J_out)
    # After QWP, light is circularly polarized → intensity should be constant!
    print(f"  Analyzer at {angle:3d}°: I = {I:.4f}")
```

---

## 6. Optical Activity and Faraday Effect

### 6.1 Optical Activity

Certain materials rotate the plane of linearly polarized light as it propagates through them. This phenomenon is called **optical activity** or **optical rotation**.

The rotation angle:

$$\phi = [\alpha] \cdot \ell \cdot c$$

where $[\alpha]$ is the **specific rotation** (depends on wavelength, temperature), $\ell$ is the path length, and $c$ is the concentration (for solutions).

| Substance | Specific Rotation $[\alpha]_D$ (deg/dm per g/mL) |
|-----------|--------------------------------------------------|
| Sucrose (sugar) | +66.5° |
| Glucose | +52.7° |
| Fructose | -92.0° |
| Quartz (crystalline, per mm) | +21.7° |

**Physical origin**: Optical activity arises from the chirality (handedness) of the molecular structure. A chiral molecule and its mirror image rotate light in opposite directions. This is the basis for **polarimetry** in chemistry — measuring sugar concentration in food processing, determining enantiomeric purity of pharmaceuticals.

### 6.2 The Faraday Effect

Michael Faraday discovered (1845) that a magnetic field applied along the direction of light propagation in certain materials rotates the plane of polarization:

$$\phi = V \cdot B \cdot \ell$$

where $V$ is the **Verdet constant** (material-dependent), $B$ is the magnetic field strength, and $\ell$ is the path length.

| Material | Verdet Constant (rad/T/m) at 589 nm |
|----------|--------------------------------------|
| Water | 1.309 |
| Glass (flint) | 3.17 |
| Terbium gallium garnet (TGG) | 40 |

**Key difference from optical activity**: The Faraday rotation is **non-reciprocal** — it accumulates in the same direction regardless of the light's propagation direction. This makes it invaluable for building **optical isolators** (devices that allow light to pass in one direction only), which are essential for protecting lasers from back-reflections.

### 6.3 Optical Isolator

An optical isolator combines a Faraday rotator and polarizers:

1. Input polarizer (0°): Transmits horizontally polarized light
2. Faraday rotator (45°): Rotates polarization by 45°
3. Output polarizer (45°): Transmits the 45°-rotated light

Forward direction: 0° → (rotate +45°) → 45° → **transmitted** ✓

Backward (reflected) direction: 45° → (rotate +45° again, not -45°!) → 90° → **blocked by input polarizer** ✗

The non-reciprocal nature of the Faraday effect is crucial — a half-wave plate would undo the rotation on the return path, but a Faraday rotator adds to it.

---

## 7. Fresnel Equations and Polarization at Interfaces

### 7.1 The Fresnel Equations

The reflection and transmission coefficients at a dielectric interface depend on the polarization:

**s-polarization** (perpendicular to plane of incidence):

$$r_s = \frac{n_1\cos\theta_i - n_2\cos\theta_t}{n_1\cos\theta_i + n_2\cos\theta_t}$$

**p-polarization** (parallel to plane of incidence):

$$r_p = \frac{n_2\cos\theta_i - n_1\cos\theta_t}{n_2\cos\theta_i + n_1\cos\theta_t}$$

The reflectances are $R_s = |r_s|^2$ and $R_p = |r_p|^2$.

At Brewster's angle, $r_p = 0$ (no reflected $p$-component), which is why the reflected light is purely $s$-polarized.

```python
import numpy as np
import matplotlib.pyplot as plt

# Fresnel equations: reflectance vs angle for s and p polarization
# Shows Brewster's angle where R_p = 0

def fresnel_reflectance(theta_i_deg, n1, n2):
    """
    Calculate s and p reflectances using Fresnel equations.
    theta_i_deg: angle of incidence in degrees
    n1, n2: refractive indices of the two media
    Returns: (R_s, R_p)
    """
    theta_i = np.deg2rad(theta_i_deg)

    # Snell's law: sin(theta_t) = n1/n2 * sin(theta_i)
    sin_theta_t = (n1 / n2) * np.sin(theta_i)

    # Handle total internal reflection
    mask = sin_theta_t <= 1.0
    cos_theta_t = np.where(mask, np.sqrt(1 - sin_theta_t**2), 0)

    # Fresnel coefficients
    r_s = np.where(mask,
                   (n1 * np.cos(theta_i) - n2 * cos_theta_t) /
                   (n1 * np.cos(theta_i) + n2 * cos_theta_t + 1e-30),
                   1.0)

    r_p = np.where(mask,
                   (n2 * np.cos(theta_i) - n1 * cos_theta_t) /
                   (n2 * np.cos(theta_i) + n1 * cos_theta_t + 1e-30),
                   1.0)

    R_s = r_s**2
    R_p = r_p**2

    return R_s, R_p

theta = np.linspace(0, 89.9, 500)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: external reflection (air → glass)
n1, n2 = 1.0, 1.5
R_s, R_p = fresnel_reflectance(theta, n1, n2)
theta_B = np.rad2deg(np.arctan(n2/n1))

ax1.plot(theta, R_s * 100, 'b-', linewidth=2, label='$R_s$ (s-polarization)')
ax1.plot(theta, R_p * 100, 'r-', linewidth=2, label='$R_p$ (p-polarization)')
ax1.plot(theta, (R_s + R_p) / 2 * 100, 'k--', linewidth=1, label='Unpolarized (average)')
ax1.axvline(theta_B, color='green', linestyle=':', linewidth=1.5,
            label=f'Brewster angle = {theta_B:.1f}°')
ax1.set_xlabel('Angle of Incidence (degrees)', fontsize=12)
ax1.set_ylabel('Reflectance (%)', fontsize=12)
ax1.set_title(f'External Reflection: Air (n={n1}) → Glass (n={n2})', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 90)
ax1.set_ylim(0, 100)

# Right: internal reflection (glass → air) — shows TIR and Brewster's angle
n1, n2 = 1.5, 1.0
R_s_int, R_p_int = fresnel_reflectance(theta, n1, n2)
theta_B_int = np.rad2deg(np.arctan(n2/n1))
theta_c = np.rad2deg(np.arcsin(n2/n1))

ax2.plot(theta, R_s_int * 100, 'b-', linewidth=2, label='$R_s$')
ax2.plot(theta, R_p_int * 100, 'r-', linewidth=2, label='$R_p$')
ax2.axvline(theta_B_int, color='green', linestyle=':', linewidth=1.5,
            label=f'Brewster = {theta_B_int:.1f}°')
ax2.axvline(theta_c, color='orange', linestyle='--', linewidth=1.5,
            label=f'Critical angle = {theta_c:.1f}°')
ax2.set_xlabel('Angle of Incidence (degrees)', fontsize=12)
ax2.set_ylabel('Reflectance (%)', fontsize=12)
ax2.set_title(f'Internal Reflection: Glass (n={n1}) → Air (n={n2})', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 90)
ax2.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('fresnel_equations.png', dpi=150)
plt.show()
```

---

## 8. Applications of Polarization

### 8.1 Liquid Crystal Displays (LCDs)

An LCD pixel works by controlling the polarization of light:

1. **Backlight** produces unpolarized light
2. **First polarizer** transmits one linear polarization
3. **Liquid crystal layer**: Normally, the LC molecules are twisted by 90° (twisted nematic mode), which rotates the polarization by 90°. An applied voltage untwists the molecules, leaving the polarization unchanged.
4. **Second polarizer** (crossed with the first): Transmits the 90°-rotated light (bright pixel). When voltage is applied, the unrotated light is blocked (dark pixel).

In-plane switching (IPS), vertical alignment (VA), and other LC modes offer improved viewing angles and contrast ratios.

### 8.2 Polarizing Sunglasses

Glare from horizontal surfaces (roads, water, snow) is predominantly $s$-polarized (horizontal). Polarizing sunglasses have their transmission axis oriented vertically, blocking the glare while transmitting vertically polarized and unpolarized light from the environment.

### 8.3 3D Cinema

**Passive 3D (RealD)**: Two projectors (or one with rapid switching) project the left-eye and right-eye images with opposite circular polarizations (RCP and LCP). The audience wears glasses with LCP and RCP filters for each eye. Using circular rather than linear polarization allows head tilting without losing the 3D effect.

### 8.4 Photoelasticity

Transparent materials under mechanical stress become birefringent (stress-induced birefringence). Viewing a stressed sample between crossed polarizers reveals colorful fringe patterns that map the stress distribution. This technique is widely used in mechanical engineering for testing structural components.

### 8.5 Ellipsometry

Ellipsometry measures the change in polarization state upon reflection from a surface to determine film thickness and optical constants with sub-nanometer precision. It is a standard technique in semiconductor fabrication for monitoring thin film deposition.

---

## Exercises

### Exercise 1: Malus's Law Chain

Three ideal linear polarizers are arranged in sequence with transmission axes at 0°, 30°, and 75° relative to the horizontal.

(a) If unpolarized light of intensity $I_0$ enters the first polarizer, what is the intensity after each polarizer?

(b) What is the final polarization direction?

(c) If the middle polarizer (30°) is removed, what is the transmitted intensity? Explain the seemingly paradoxical result.

### Exercise 2: Brewster's Angle

Light is incident on a glass plate ($n = 1.52$) from air.

(a) Calculate Brewster's angle.

(b) At Brewster's angle, what fraction of the incident unpolarized light is reflected? (Hint: use the Fresnel equation for $R_s$ at Brewster's angle.)

(c) If 10 glass plates are stacked at Brewster's angle (Brewster stack), what is the degree of polarization of the transmitted beam?

### Exercise 3: Wave Plate Analysis

A half-wave plate is oriented with its fast axis at 22.5° from the horizontal.

(a) If horizontally polarized light enters, what is the output polarization direction?

(b) Express the input and output as Jones vectors.

(c) Write the Jones matrix for this half-wave plate and verify your answer by matrix multiplication.

### Exercise 4: Circular Polarizer Design

Design an optical system that converts unpolarized light into right-circularly polarized light.

(a) Specify the required elements and their orientations.

(b) What fraction of the input intensity is transmitted?

(c) Verify your design using Jones calculus.

### Exercise 5: Optical Isolator

An optical isolator uses a Faraday rotator with Verdet constant $V = 40$ rad/(T$\cdot$m) at 1064 nm.

(a) What magnetic field strength is needed for 45° rotation in a 2 cm crystal?

(b) Explain step-by-step why back-reflected light is blocked but forward light is transmitted.

(c) If the Faraday rotator's rotation angle deviates by $\pm 1°$ from the ideal 45°, what is the isolation ratio (ratio of forward to backward transmitted intensity)?

---

## Summary

| Concept | Key Formula / Fact |
|---------|-------------------|
| Polarization types | Linear ($\delta = 0, \pi$), circular ($\delta = \pm\pi/2$, equal amplitudes), elliptical (general) |
| Malus's law | $I = I_0\cos^2\theta$ (angle between polarization and transmission axis) |
| Unpolarized through polarizer | $I = I_0/2$ |
| Brewster's angle | $\tan\theta_B = n_2/n_1$; reflected light is $s$-polarized |
| Birefringence | $n_o \neq n_e$; splits light into ordinary and extraordinary rays |
| Quarter-wave plate | $\delta = \pi/2$; converts linear ↔ circular |
| Half-wave plate | $\delta = \pi$; rotates linear by $2\alpha$; flips circular handedness |
| Jones vector | $\mathbf{J} = \begin{pmatrix} E_{0x} \\ E_{0y}e^{i\delta} \end{pmatrix}$ |
| Fresnel equations | $r_s, r_p$ depend on angle; $R_p = 0$ at Brewster's angle |
| Optical activity | $\phi = [\alpha] \ell c$ (rotation of plane of polarization) |
| Faraday effect | $\phi = VB\ell$ (non-reciprocal rotation; basis for optical isolators) |
| LCD operation | Crossed polarizers + twisted nematic LC; voltage controls twist → brightness |

---

[← Previous: 06. Diffraction](06_Diffraction.md) | [Next: 08. Laser Fundamentals →](08_Laser_Fundamentals.md)
