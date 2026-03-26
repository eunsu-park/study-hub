# 12. Waveguides and Cavities

[← Previous: 11. Reflection and Refraction](11_Reflection_and_Refraction.md) | [Next: 13. Radiation and Antennas →](13_Radiation_and_Antennas.md)

## Learning Objectives

1. Derive the transverse field equations for guided waves from Maxwell's equations
2. Solve for TE and TM modes in a rectangular waveguide and identify cutoff frequencies
3. Understand the waveguide dispersion relation and the concepts of phase and group velocity
4. Analyze rectangular cavity resonators and compute their resonant frequencies and quality factor Q
5. Describe the basic principles of circular waveguides and optical fibers
6. Compute waveguide parameters numerically and visualize mode patterns using Python
7. Connect waveguide theory to practical applications in microwave engineering and photonics

A waveguide is a hollow or dielectric structure that confines and directs electromagnetic waves, much like a pipe channels water. Unlike free-space propagation, waveguides impose boundary conditions that quantize the transverse field pattern into discrete **modes**, each with a minimum operating frequency. This mode structure is the key to understanding microwave transmission, radar plumbing, particle accelerators, and optical fiber communication. In this lesson, we derive the mode structure from first principles, explore the rich physics of waveguide dispersion, and extend the analysis to resonant cavities where waves are trapped in three dimensions.

> **Analogy**: A waveguide is like a hallway for sound. If you shout in a narrow hallway, only certain spatial patterns (modes) of the sound wave can propagate without destructive interference from the walls. Very low-frequency sounds (wavelengths much larger than the hallway width) cannot propagate at all — they are "below cutoff." Similarly, a microwave waveguide only transmits frequencies above the cutoff frequency of its dominant mode.

---

## 1. General Theory of Guided Waves

### 1.1 Setup

Consider a waveguide extending along the $z$-axis with a uniform cross-section in the $xy$-plane. We seek solutions of the form:

$$\mathbf{E}(x, y, z, t) = \mathbf{E}(x, y) \, e^{i(k_z z - \omega t)}$$

$$\mathbf{H}(x, y, z, t) = \mathbf{H}(x, y) \, e^{i(k_z z - \omega t)}$$

where $k_z$ is the propagation constant along the guide.

### 1.2 Transverse and Longitudinal Decomposition

We decompose the fields into transverse ($\mathbf{E}_T$, $\mathbf{H}_T$) and longitudinal ($E_z$, $H_z$) components. Maxwell's curl equations then give the **transverse fields entirely in terms of** $E_z$ and $H_z$:

$$\mathbf{E}_T = \frac{i}{k_c^2}\left(k_z \nabla_T E_z - \omega\mu \, \hat{z} \times \nabla_T H_z\right)$$

$$\mathbf{H}_T = \frac{i}{k_c^2}\left(k_z \nabla_T H_z + \omega\epsilon \, \hat{z} \times \nabla_T E_z\right)$$

where $k_c^2 = k^2 - k_z^2$ is the **cutoff wave number** and $k = \omega\sqrt{\mu\epsilon}$.

This is a powerful result: once we solve for $E_z$ and $H_z$ (scalar problems), all six field components are determined.

### 1.3 Mode Classification

- **TE modes** (Transverse Electric): $E_z = 0$, fields determined by $H_z$
- **TM modes** (Transverse Magnetic): $H_z = 0$, fields determined by $E_z$
- **TEM modes**: $E_z = H_z = 0$, require two or more conductors (e.g., coaxial cable)

For a hollow waveguide, TEM modes are impossible — at least one of $E_z$ or $H_z$ must be nonzero.

---

## 2. Rectangular Waveguide

### 2.1 Geometry

Consider a rectangular waveguide with width $a$ (along $x$) and height $b$ (along $y$), where $a > b$ by convention. The walls are perfect conductors.

### 2.2 TM Modes ($H_z = 0$)

The longitudinal electric field satisfies:

$$\left(\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + k_c^2\right) E_z = 0$$

with the boundary condition $E_z = 0$ on all walls (tangential $E$ vanishes on a perfect conductor). The solution is:

$$E_z^{mn} = E_0 \sin\left(\frac{m\pi x}{a}\right) \sin\left(\frac{n\pi y}{b}\right)$$

with $m = 1, 2, 3, \ldots$ and $n = 1, 2, 3, \ldots$ (both must be at least 1 for a nontrivial solution).

The cutoff wave number is:

$$k_c^{mn} = \sqrt{\left(\frac{m\pi}{a}\right)^2 + \left(\frac{n\pi}{b}\right)^2}$$

### 2.3 TE Modes ($E_z = 0$)

The longitudinal magnetic field satisfies the same Helmholtz equation, but with the Neumann boundary condition $\partial H_z / \partial n = 0$ on the walls:

$$H_z^{mn} = H_0 \cos\left(\frac{m\pi x}{a}\right) \cos\left(\frac{n\pi y}{b}\right)$$

with $m = 0, 1, 2, \ldots$ and $n = 0, 1, 2, \ldots$ (but not $m = n = 0$). The cosine functions naturally satisfy the Neumann condition.

### 2.4 Cutoff Frequency

The cutoff frequency for mode $(m, n)$ is:

$$\boxed{f_c^{mn} = \frac{1}{2\sqrt{\mu\epsilon}} \sqrt{\left(\frac{m}{a}\right)^2 + \left(\frac{n}{b}\right)^2}}$$

Below cutoff ($f < f_c$), $k_z$ becomes imaginary and the wave is evanescent. For $a > b$:

- **Dominant mode**: TE$_{10}$ with $f_c = 1/(2a\sqrt{\mu\epsilon})$ — the lowest cutoff
- **First TM mode**: TM$_{11}$ with higher cutoff than TE$_{10}$

The **single-mode bandwidth** is the frequency range $[f_c^{10}, f_c^{20}]$ (or $[f_c^{10}, f_c^{01}]$ depending on $a/b$) where only the dominant mode propagates. For the standard $a = 2b$ ratio, this is an octave: $f_c^{20} = 2f_c^{10}$.

```python
import numpy as np
import matplotlib.pyplot as plt

def cutoff_frequencies(a, b, max_m=5, max_n=5, mu=4*np.pi*1e-7, eps=8.854e-12):
    """
    Compute cutoff frequencies for rectangular waveguide modes.

    Parameters:
        a, b  : waveguide dimensions (m), a > b
        max_m, max_n : maximum mode indices

    Why mode ordering matters: in practice, we want single-mode operation
    (only the dominant mode propagating). The cutoff frequencies determine
    the usable bandwidth of the waveguide.
    """
    modes = []
    for m in range(max_m + 1):
        for n in range(max_n + 1):
            if m == 0 and n == 0:
                continue  # no TEM mode in hollow waveguide

            fc = 0.5 / np.sqrt(mu * eps) * np.sqrt((m / a)**2 + (n / b)**2)

            # Determine mode type
            if m == 0 or n == 0:
                mode_type = 'TE'  # TE modes allow m=0 or n=0
            else:
                mode_type = 'TE/TM'  # both exist for m,n >= 1

            modes.append({
                'm': m, 'n': n,
                'fc_GHz': fc / 1e9,
                'type': mode_type
            })

    # Sort by cutoff frequency
    modes.sort(key=lambda x: x['fc_GHz'])
    return modes

# WR-90 waveguide (X-band): a = 22.86 mm, b = 10.16 mm
a = 22.86e-3  # m
b = 10.16e-3  # m

modes = cutoff_frequencies(a, b)

print("Rectangular Waveguide WR-90 Mode Table")
print("=" * 55)
print(f"{'Mode':<12} {'Type':<8} {'f_c (GHz)':>10}")
print("-" * 55)
for mode in modes[:12]:
    name = f"({mode['m']},{mode['n']})"
    print(f"{name:<12} {mode['type']:<8} {mode['fc_GHz']:>10.3f}")

print(f"\nDominant mode: TE10, f_c = {modes[0]['fc_GHz']:.3f} GHz")
print(f"Single-mode band: {modes[0]['fc_GHz']:.3f} - {modes[1]['fc_GHz']:.3f} GHz")
```

---

## 3. Dispersion Relation

### 3.1 Waveguide Dispersion

The propagation constant for a propagating mode is:

$$k_z = \sqrt{k^2 - k_c^2} = \frac{\omega}{c}\sqrt{1 - \left(\frac{f_c}{f}\right)^2}$$

This gives the **waveguide dispersion relation**:

$$\boxed{\omega^2 = \omega_c^2 + k_z^2 c^2}$$

This has the same form as a relativistic energy-momentum relation $E^2 = (mc^2)^2 + (pc)^2$, with $\omega_c$ playing the role of a "rest mass." The cutoff frequency acts as an effective mass for the guided photon.

### 3.2 Phase and Group Velocity

**Phase velocity** (speed of constant-phase surfaces along the guide):

$$v_p = \frac{\omega}{k_z} = \frac{c}{\sqrt{1 - (f_c/f)^2}} > c$$

**Group velocity** (speed of energy transport):

$$v_g = \frac{d\omega}{dk_z} = c\sqrt{1 - \left(\frac{f_c}{f}\right)^2} < c$$

Note the remarkable relation:

$$\boxed{v_p \cdot v_g = c^2}$$

The phase velocity exceeds $c$, but the group velocity (which carries information and energy) is always less than $c$. At cutoff, $v_g \to 0$ and $v_p \to \infty$.

```python
def waveguide_dispersion(a, b, m, n, f_max_GHz=20):
    """
    Plot the dispersion relation and velocities for a waveguide mode.

    Why dispersion matters: the frequency-dependent group velocity causes
    pulse broadening in waveguide communication systems, analogous to
    chromatic dispersion in optical fibers.
    """
    c = 3e8  # speed of light (m/s)
    fc = 0.5 * c * np.sqrt((m / a)**2 + (n / b)**2)
    fc_GHz = fc / 1e9

    f = np.linspace(fc * 1.01, f_max_GHz * 1e9, 1000)
    omega = 2 * np.pi * f
    omega_c = 2 * np.pi * fc

    # Propagation constant
    kz = (omega / c) * np.sqrt(1 - (fc / f)**2)

    # Phase and group velocity
    vp = omega / kz
    vg = c**2 / vp  # from vp * vg = c^2

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Dispersion diagram (omega vs kz)
    axes[0].plot(kz, f / 1e9, 'b-', linewidth=2)
    axes[0].plot(kz, kz * c / (2 * np.pi * 1e9), 'k--', alpha=0.5,
                 label='Light line ($\\omega = ck_z$)')
    axes[0].axhline(y=fc_GHz, color='r', linestyle=':', label=f'$f_c$ = {fc_GHz:.2f} GHz')
    axes[0].set_xlabel('$k_z$ (rad/m)')
    axes[0].set_ylabel('Frequency (GHz)')
    axes[0].set_title(f'Dispersion: TE$_{{{m}{n}}}$')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Phase and group velocity
    axes[1].plot(f / 1e9, vp / c, 'b-', linewidth=2, label='$v_p / c$')
    axes[1].plot(f / 1e9, vg / c, 'r-', linewidth=2, label='$v_g / c$')
    axes[1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(x=fc_GHz, color='green', linestyle=':', label=f'$f_c$')
    axes[1].set_xlabel('Frequency (GHz)')
    axes[1].set_ylabel('Velocity / c')
    axes[1].set_title('Phase and Group Velocity')
    axes[1].set_ylim(0, 5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Guide wavelength vs free-space wavelength
    lambda_g = 2 * np.pi / kz
    lambda_0 = c / f

    axes[2].plot(lambda_0 * 1e3, lambda_g * 1e3, 'b-', linewidth=2)
    axes[2].plot(lambda_0 * 1e3, lambda_0 * 1e3, 'k--', alpha=0.5,
                 label='$\\lambda_g = \\lambda_0$')
    axes[2].set_xlabel('Free-space wavelength (mm)')
    axes[2].set_ylabel('Guide wavelength (mm)')
    axes[2].set_title('Guide Wavelength')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'WR-90 Rectangular Waveguide: TE$_{{{m}{n}}}$ Mode', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("waveguide_dispersion.png", dpi=150)
    plt.show()

waveguide_dispersion(a=22.86e-3, b=10.16e-3, m=1, n=0)
```

---

## 4. Mode Patterns

### 4.1 Field Distribution of TE$_{mn}$ Modes

For the TE$_{mn}$ mode, the transverse fields are:

$$H_x = -\frac{ik_z}{k_c^2} \frac{m\pi}{a} H_0 \sin\left(\frac{m\pi x}{a}\right)\cos\left(\frac{n\pi y}{b}\right)$$

$$H_y = -\frac{ik_z}{k_c^2} \frac{n\pi}{b} H_0 \cos\left(\frac{m\pi x}{a}\right)\sin\left(\frac{n\pi y}{b}\right)$$

$$E_x = \frac{i\omega\mu}{k_c^2} \frac{n\pi}{b} H_0 \cos\left(\frac{m\pi x}{a}\right)\sin\left(\frac{n\pi y}{b}\right)$$

$$E_y = -\frac{i\omega\mu}{k_c^2} \frac{m\pi}{a} H_0 \sin\left(\frac{m\pi x}{a}\right)\cos\left(\frac{n\pi y}{b}\right)$$

The dominant mode TE$_{10}$ has a particularly simple pattern: $E_y = E_0 \sin(\pi x / a)$, a single half-sine variation across the wide dimension.

```python
def plot_waveguide_modes(a, b, modes_to_plot=None):
    """
    Visualize electric field patterns for rectangular waveguide modes.

    Why visualize: the mode pattern determines coupling efficiency
    to antennas, the location of slots for radiation, and the
    current distribution on the waveguide walls.
    """
    if modes_to_plot is None:
        modes_to_plot = [('TE', 1, 0), ('TE', 2, 0), ('TE', 0, 1),
                         ('TE', 1, 1), ('TM', 1, 1), ('TE', 2, 1)]

    x = np.linspace(0, a, 100)
    y = np.linspace(0, b, 80)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, (mode_type, m, n) in enumerate(modes_to_plot):
        if idx >= len(axes):
            break

        kc2 = (m * np.pi / a)**2 + (n * np.pi / b)**2

        if mode_type == 'TE':
            # Electric field components for TE_mn
            if kc2 == 0:
                continue
            Ex = (n * np.pi / b) * np.cos(m * np.pi * X / a) * np.sin(n * np.pi * Y / b)
            Ey = -(m * np.pi / a) * np.sin(m * np.pi * X / a) * np.cos(n * np.pi * Y / b)
        else:  # TM
            # Electric field components for TM_mn
            Ex = (m * np.pi / a) * np.cos(m * np.pi * X / a) * np.sin(n * np.pi * Y / b)
            Ey = (n * np.pi / b) * np.sin(m * np.pi * X / a) * np.cos(n * np.pi * Y / b)

        E_mag = np.sqrt(Ex**2 + Ey**2)

        ax = axes[idx]
        im = ax.pcolormesh(X * 1e3, Y * 1e3, E_mag, cmap='hot', shading='auto')

        # Add vector arrows (subsample for clarity)
        skip = 8
        scale = E_mag.max() if E_mag.max() > 0 else 1
        ax.quiver(X[::skip, ::skip] * 1e3, Y[::skip, ::skip] * 1e3,
                  Ex[::skip, ::skip] / scale, Ey[::skip, ::skip] / scale,
                  color='cyan', alpha=0.7, scale=15)

        c_val = 3e8
        fc = 0.5 * c_val * np.sqrt((m / a)**2 + (n / b)**2)
        ax.set_title(f'{mode_type}$_{{{m}{n}}}$ ($f_c$ = {fc/1e9:.2f} GHz)')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_aspect('equal')

    plt.suptitle('Rectangular Waveguide Mode Patterns (Electric Field)', fontsize=14)
    plt.tight_layout()
    plt.savefig("waveguide_modes.png", dpi=150)
    plt.show()

plot_waveguide_modes(a=22.86e-3, b=10.16e-3)
```

---

## 5. Rectangular Cavity Resonator

### 5.1 From Waveguide to Cavity

A resonant cavity is formed by closing both ends of a waveguide with conducting walls (at $z = 0$ and $z = d$). The additional boundary conditions quantize $k_z$:

$$k_z = \frac{p\pi}{d}, \quad p = 0, 1, 2, \ldots$$

(For TM modes, $p = 0$ is allowed; for TE modes, $p \geq 1$ for a nontrivial solution with the specific boundary conditions on $H_z$.)

### 5.2 Resonant Frequencies

The resonant frequencies of a rectangular cavity are:

$$\boxed{f_{mnp} = \frac{1}{2\sqrt{\mu\epsilon}}\sqrt{\left(\frac{m}{a}\right)^2 + \left(\frac{n}{b}\right)^2 + \left(\frac{p}{d}\right)^2}}$$

The **dominant mode** of the cavity depends on the dimensions. For $a > d > b$, the lowest resonance is typically TE$_{101}$.

### 5.3 Quality Factor Q

The quality factor measures how well the cavity stores energy:

$$\boxed{Q = \omega_0 \frac{W_{\text{stored}}}{P_{\text{loss}}} = \frac{f_0}{\Delta f}}$$

where $W_{\text{stored}}$ is the time-averaged stored energy and $P_{\text{loss}}$ is the power dissipated in the walls.

For a TE$_{101}$ mode in a rectangular cavity with wall conductivity $\sigma$:

$$Q = \frac{(k a)^2 b d \, \mu \omega}{4 R_s \left[b d (a^2 + d^2)/a^2 d^2 \cdot \pi^2/2 + a(b/d + d/b) + \ldots\right]}$$

where $R_s = \sqrt{\omega\mu / 2\sigma}$ is the surface resistance. Typical values: $Q \sim 10^3$ to $10^5$ for copper cavities.

Superconducting cavities (used in particle accelerators) achieve $Q > 10^{10}$.

```python
def cavity_resonances(a, b, d, max_mode=4):
    """
    Compute resonant frequencies of a rectangular cavity.

    Why cavities matter: microwave ovens use cavity resonances to
    heat food, particle accelerators use superconducting cavities
    to accelerate charged particles, and radar systems use cavity
    filters for frequency selection.
    """
    c = 3e8
    resonances = []

    for m in range(max_mode + 1):
        for n in range(max_mode + 1):
            for p in range(max_mode + 1):
                if m == 0 and n == 0:
                    continue

                # TE modes: need at least two nonzero indices
                # TM modes: m >= 1 and n >= 1, p can be 0
                if m >= 1 and n >= 1:
                    f = 0.5 * c * np.sqrt((m/a)**2 + (n/b)**2 + (p/d)**2)
                    resonances.append(('TM' if p == 0 or True else 'TM',
                                       m, n, p, f / 1e9))

                if (m >= 1 or n >= 1) and p >= 1:
                    f = 0.5 * c * np.sqrt((m/a)**2 + (n/b)**2 + (p/d)**2)
                    resonances.append(('TE', m, n, p, f / 1e9))

    # Remove duplicates (TE and TM can share the same frequency)
    seen = set()
    unique = []
    for mode in resonances:
        key = (mode[1], mode[2], mode[3])
        if key not in seen:
            seen.add(key)
            unique.append(mode)

    unique.sort(key=lambda x: x[4])
    return unique

# Cavity dimensions
a_cav = 30e-3   # 30 mm
b_cav = 15e-3   # 15 mm
d_cav = 25e-3   # 25 mm

resonances = cavity_resonances(a_cav, b_cav, d_cav)

print(f"Rectangular Cavity: {a_cav*1e3:.0f} x {b_cav*1e3:.0f} x {d_cav*1e3:.0f} mm")
print("=" * 50)
print(f"{'Mode (m,n,p)':<15} {'f (GHz)':>10}")
print("-" * 50)
for mode in resonances[:10]:
    print(f"({mode[1]},{mode[2]},{mode[3]}){'':<10} {mode[4]:>10.3f}")
```

---

## 6. Optical Fibers (Brief Introduction)

### 6.1 Step-Index Fiber

An optical fiber consists of a core (refractive index $n_1$) surrounded by a cladding ($n_2 < n_1$). Light is confined by total internal reflection. The key parameter is the **numerical aperture**:

$$\text{NA} = \sqrt{n_1^2 - n_2^2} = n_1 \sin\theta_{\max}$$

where $\theta_{\max}$ is the maximum acceptance angle.

### 6.2 V-Number and Number of Modes

The **V-number** determines how many modes a fiber supports:

$$V = \frac{2\pi a}{\lambda} \text{NA} = \frac{2\pi a}{\lambda}\sqrt{n_1^2 - n_2^2}$$

where $a$ is the core radius. For $V < 2.405$, only the fundamental LP$_{01}$ mode propagates (single-mode fiber). The number of modes for large $V$ is approximately:

$$N_{\text{modes}} \approx \frac{V^2}{2}$$

### 6.3 Fiber Dispersion

In optical fibers, the total dispersion has two contributions:

- **Material dispersion**: the refractive index varies with wavelength (Lorentz model from Lesson 10)
- **Waveguide dispersion**: the mode's effective index depends on the ratio $a/\lambda$

The zero-dispersion wavelength of standard silica fiber is near 1.3 $\mu$m, which is why early fiber systems operated at this wavelength.

```python
def fiber_modes_and_na(n_core, n_clad, core_radius, wavelength):
    """
    Compute fiber parameters: NA, V-number, and number of modes.

    Why these parameters: NA determines coupling efficiency from a source,
    V-number determines single-mode vs multimode operation, and the
    number of modes affects bandwidth (modal dispersion).
    """
    NA = np.sqrt(n_core**2 - n_clad**2)
    V = 2 * np.pi * core_radius / wavelength * NA
    N_modes = max(1, int(V**2 / 2))

    theta_max = np.arcsin(NA)

    print(f"Optical Fiber Parameters")
    print(f"========================")
    print(f"Core index:    {n_core:.4f}")
    print(f"Cladding index: {n_clad:.4f}")
    print(f"Core radius:   {core_radius*1e6:.1f} μm")
    print(f"Wavelength:    {wavelength*1e9:.0f} nm")
    print(f"NA:            {NA:.4f}")
    print(f"V-number:      {V:.3f}")
    print(f"Single-mode:   {'Yes' if V < 2.405 else 'No'}")
    print(f"Approx modes:  {N_modes}")
    print(f"Max acceptance angle: {np.degrees(theta_max):.1f}°")

    return NA, V, N_modes

# Standard single-mode fiber (SMF-28)
fiber_modes_and_na(n_core=1.4681, n_clad=1.4629,
                   core_radius=4.1e-6, wavelength=1550e-9)

print()

# Multimode fiber
fiber_modes_and_na(n_core=1.480, n_clad=1.460,
                   core_radius=25e-6, wavelength=850e-9)
```

---

## 7. Circular Waveguide (Brief Overview)

### 7.1 Mode Structure

In cylindrical coordinates $(r, \phi, z)$, the solutions involve Bessel functions. The TE$_{mn}$ and TM$_{mn}$ mode cutoff frequencies are:

$$f_c^{\text{TE}} = \frac{x'_{mn}}{2\pi a \sqrt{\mu\epsilon}}, \quad f_c^{\text{TM}} = \frac{x_{mn}}{2\pi a \sqrt{\mu\epsilon}}$$

where $x_{mn}$ is the $n$-th zero of $J_m(x)$ and $x'_{mn}$ is the $n$-th zero of $J'_m(x)$.

| Mode | $x_{mn}$ or $x'_{mn}$ | Relative cutoff |
|------|------------------------|-----------------|
| TE$_{11}$ | $x'_{11} = 1.841$ | 1.000 (dominant) |
| TM$_{01}$ | $x_{01} = 2.405$ | 1.306 |
| TE$_{21}$ | $x'_{21} = 3.054$ | 1.659 |
| TM$_{11}$ | $x_{11} = 3.832$ | 2.081 |
| TE$_{01}$ | $x'_{01} = 3.832$ | 2.081 |

The TE$_{01}$ mode is special: its attenuation **decreases** with frequency, making it attractive for long-distance microwave transmission (though practical issues with mode conversion limited its use).

---

## 8. Waveguide Attenuation

### 8.1 Sources of Loss

Real waveguides have finite wall conductivity, introducing ohmic losses. The attenuation constant is:

$$\alpha = \frac{P_{\text{loss per unit length}}}{2 P_{\text{transmitted}}}$$

For the TE$_{10}$ mode:

$$\alpha = \frac{R_s}{a^3 b k \eta} \left(2b\pi^2 + a^3 k^2\right) \cdot \frac{1}{\sqrt{1 - (f_c/f)^2}}$$

where $R_s = \sqrt{\pi f \mu / \sigma}$ is the surface resistance and $\eta = \sqrt{\mu/\epsilon}$ is the intrinsic impedance.

Key features:
- Attenuation is infinite at cutoff ($f \to f_c$)
- Attenuation has a minimum at an intermediate frequency
- Higher-order modes generally have higher attenuation

---

## Summary

| Concept | Key Formula | Physical Meaning |
|---------|-------------|------------------|
| Cutoff frequency | $f_c^{mn} = \frac{c}{2}\sqrt{(m/a)^2 + (n/b)^2}$ | Minimum frequency for mode propagation |
| Dispersion relation | $\omega^2 = \omega_c^2 + k_z^2 c^2$ | Massive-particle-like dispersion |
| Phase velocity | $v_p = c/\sqrt{1 - (f_c/f)^2}$ | Always $> c$ |
| Group velocity | $v_g = c\sqrt{1 - (f_c/f)^2}$ | Always $< c$, carries energy |
| $v_p \cdot v_g$ relation | $v_p \cdot v_g = c^2$ | Geometric mean equals $c$ |
| Cavity resonance | $f_{mnp} = \frac{c}{2}\sqrt{(m/a)^2 + (n/b)^2 + (p/d)^2}$ | 3D standing wave |
| Quality factor | $Q = \omega_0 W / P_{\text{loss}}$ | Energy storage efficiency |
| Fiber V-number | $V = (2\pi a/\lambda)\sqrt{n_1^2 - n_2^2}$ | Mode count parameter |

---

## Exercises

### Exercise 1: WR-284 Waveguide
The WR-284 waveguide has dimensions $a = 72.14$ mm and $b = 34.04$ mm. (a) Calculate the cutoff frequency of the first 8 modes and identify each as TE or TM. (b) Determine the usable single-mode frequency range. (c) At 3 GHz, compute the phase velocity, group velocity, and guide wavelength for the TE$_{10}$ mode.

### Exercise 2: Microwave Cavity Design
Design a rectangular cavity resonator with fundamental TE$_{101}$ mode at exactly 2.45 GHz (microwave oven frequency). Choose dimensions $a$, $b$, $d$ with $a > d > b$. (a) Compute the Q-factor assuming copper walls ($\sigma = 5.96 \times 10^7$ S/m). (b) How long does energy remain in the cavity after the source is turned off ($\tau = Q / \omega_0$)?

### Exercise 3: Single-Mode Fiber Design
Design a step-index single-mode fiber for operation at 1310 nm. The core and cladding indices are $n_1 = 1.468$ and $n_2 = 1.463$. (a) Find the maximum core radius for single-mode operation ($V < 2.405$). (b) Compute the NA and acceptance angle. (c) If the fiber is used at 850 nm instead, how many modes would it support?

### Exercise 4: Mode Visualization
Write a Python program to visualize the TE$_{21}$ and TM$_{21}$ modes in a rectangular waveguide with $a = 2b$. Plot both the electric field vector pattern and the magnetic field lines. Identify the locations of maximum and zero field.

### Exercise 5: Circular Waveguide Modes
For a circular waveguide of radius $a = 15$ mm, compute the cutoff frequencies of the first 6 modes. Plot the radial field pattern of the TE$_{11}$ mode using Bessel functions. Compare the single-mode bandwidth with that of a rectangular waveguide of comparable area.

---

[← Previous: 11. Reflection and Refraction](11_Reflection_and_Refraction.md) | [Next: 13. Radiation and Antennas →](13_Radiation_and_Antennas.md)
