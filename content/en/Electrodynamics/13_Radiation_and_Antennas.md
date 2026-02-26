# 13. Radiation and Antennas

[← Previous: 12. Waveguides and Cavities](12_Waveguides_and_Cavities.md) | [Next: 14. Relativistic Electrodynamics →](14_Relativistic_Electrodynamics.md)

## Learning Objectives

1. Derive the retarded potentials from Maxwell's equations and understand the concept of retarded time
2. Compute the radiation fields of an oscillating electric dipole in the far-field limit
3. Apply the Larmor formula to calculate radiated power from an accelerating charge
4. Understand radiation resistance and its role in antenna efficiency
5. Analyze the half-wave dipole antenna and compute its radiation pattern
6. Design simple antenna arrays and understand beam steering through phase control
7. Quantify antenna performance using directivity, gain, and effective area

Every radio station, cell tower, Wi-Fi router, and star in the sky is a source of electromagnetic radiation. Radiation occurs whenever charges accelerate, and the emitted fields carry energy and momentum to infinity. The challenge is to compute the fields far from the source and to shape the radiation pattern for practical purposes. This lesson takes us from the fundamental retarded potentials — which encode the finite speed of light into the potential formalism — through dipole radiation, the Larmor formula, and into the engineering of antennas that shape and direct electromagnetic waves with remarkable precision.

> **Analogy**: Drop a stone in a pond and watch the ripples spread outward. The ripples carry information about when and where the stone hit — but they arrive at distant points with a delay proportional to the distance. This is exactly what retarded potentials describe: the electromagnetic "ripples" from an accelerating charge reach a distant observer at the retarded time $t_r = t - |\mathbf{r} - \mathbf{r}'|/c$, encoding the source's past state.

---

## 1. Retarded Potentials

### 1.1 The Problem of Causality

In electrostatics, the scalar potential is:

$$V(\mathbf{r}) = \frac{1}{4\pi\epsilon_0}\int \frac{\rho(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} \, d^3r'$$

But for time-varying sources, this cannot be correct — it implies that a change in $\rho$ at $\mathbf{r}'$ instantaneously affects $V$ at $\mathbf{r}$, violating causality.

### 1.2 Retarded Time

The correct solution uses the **retarded time**:

$$t_r = t - \frac{|\mathbf{r} - \mathbf{r}'|}{c}$$

This is the time at which a signal must leave the source at $\mathbf{r}'$ to reach the observation point $\mathbf{r}$ at time $t$, traveling at the speed of light.

### 1.3 Retarded Potentials

The retarded potentials are:

$$\boxed{V(\mathbf{r}, t) = \frac{1}{4\pi\epsilon_0}\int \frac{\rho(\mathbf{r}', t_r)}{|\mathbf{r} - \mathbf{r}'|} \, d^3r'}$$

$$\boxed{\mathbf{A}(\mathbf{r}, t) = \frac{\mu_0}{4\pi}\int \frac{\mathbf{J}(\mathbf{r}', t_r)}{|\mathbf{r} - \mathbf{r}'|} \, d^3r'}$$

These satisfy the Lorenz gauge condition $\nabla \cdot \mathbf{A} + \mu_0\epsilon_0 \partial V/\partial t = 0$ and the inhomogeneous wave equations:

$$\nabla^2 V - \frac{1}{c^2}\frac{\partial^2 V}{\partial t^2} = -\frac{\rho}{\epsilon_0}$$

$$\nabla^2 \mathbf{A} - \frac{1}{c^2}\frac{\partial^2 \mathbf{A}}{\partial t^2} = -\mu_0 \mathbf{J}$$

### 1.4 Lienard-Wiechert Potentials (Overview)

For a single point charge $q$ moving along a trajectory $\mathbf{r}_0(t)$ with velocity $\mathbf{v}(t)$, the retarded potentials become:

$$V(\mathbf{r}, t) = \frac{q}{4\pi\epsilon_0} \frac{1}{|\boldsymbol{\mathscr{r}}|(1 - \hat{\boldsymbol{\mathscr{r}}} \cdot \boldsymbol{\beta})}\bigg|_{t_r}$$

$$\mathbf{A}(\mathbf{r}, t) = \frac{\mu_0 q}{4\pi} \frac{\mathbf{v}}{|\boldsymbol{\mathscr{r}}|(1 - \hat{\boldsymbol{\mathscr{r}}} \cdot \boldsymbol{\beta})}\bigg|_{t_r}$$

where $\boldsymbol{\mathscr{r}} = \mathbf{r} - \mathbf{r}_0(t_r)$ is the vector from the retarded position to the field point, and $\boldsymbol{\beta} = \mathbf{v}/c$. The factor $(1 - \hat{\boldsymbol{\mathscr{r}}} \cdot \boldsymbol{\beta})$ in the denominator is responsible for relativistic beaming effects.

---

## 2. Oscillating Electric Dipole Radiation

### 2.1 Setup

The simplest radiating system is an oscillating electric dipole:

$$\mathbf{p}(t) = p_0 \cos(\omega t) \, \hat{z} = \text{Re}[p_0 e^{-i\omega t}] \, \hat{z}$$

This represents two charges $+q$ and $-q$ oscillating about a common center with separation $d(t)$, so $p_0 = qd_0$.

### 2.2 Far-Field Approximation

In the **far field** (or radiation zone), where $r \gg \lambda \gg d$, the vector potential simplifies to:

$$\mathbf{A}(\mathbf{r}, t) \approx -\frac{\mu_0 p_0 \omega}{4\pi r} \sin(\omega t_r) \, \hat{z}$$

where $t_r = t - r/c$.

### 2.3 Radiation Fields

The electric and magnetic fields in the far zone are:

$$\mathbf{E} = -\frac{\mu_0 p_0 \omega^2}{4\pi c}\frac{\sin\theta}{r}\cos\left[\omega\left(t - \frac{r}{c}\right)\right] \hat{\theta}$$

$$\mathbf{B} = \frac{1}{c}\hat{r} \times \mathbf{E} = -\frac{\mu_0 p_0 \omega^2}{4\pi c^2}\frac{\sin\theta}{r}\cos\left[\omega\left(t - \frac{r}{c}\right)\right] \hat{\phi}$$

Key features of dipole radiation:
- Fields fall as $1/r$ (not $1/r^2$) — this is the hallmark of radiation
- $\mathbf{E} \perp \mathbf{B} \perp \hat{r}$ — the fields are transverse
- The $\sin\theta$ pattern means **no radiation along the dipole axis** ($\theta = 0$) and **maximum radiation in the equatorial plane** ($\theta = \pi/2$)

### 2.4 Radiated Power

The time-averaged Poynting vector is:

$$\langle \mathbf{S} \rangle = \frac{\mu_0 p_0^2 \omega^4}{32\pi^2 c} \frac{\sin^2\theta}{r^2} \hat{r}$$

Integrating over all solid angles gives the total radiated power:

$$\boxed{P = \frac{\mu_0 p_0^2 \omega^4}{12\pi c} = \frac{p_0^2 \omega^4}{12\pi \epsilon_0 c^3}}$$

This $\omega^4$ dependence explains why the sky is blue (Rayleigh scattering) — higher-frequency blue light is scattered much more than lower-frequency red light.

```python
import numpy as np
import matplotlib.pyplot as plt

def dipole_radiation_pattern(theta, power_pattern='sin2'):
    """
    Compute the radiation pattern of an oscillating electric dipole.

    The sin^2(theta) pattern is the defining signature of electric
    dipole radiation — it shows zero radiation along the dipole axis
    and maximum radiation perpendicular to it.
    """
    if power_pattern == 'sin2':
        return np.sin(theta)**2
    return np.ones_like(theta)

# Create polar radiation pattern plot
theta = np.linspace(0, 2 * np.pi, 500)
r_pattern = dipole_radiation_pattern(theta)

fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                          subplot_kw={'projection': 'polar'})

# Power pattern
axes[0].plot(theta, r_pattern, 'b-', linewidth=2)
axes[0].fill(theta, r_pattern, alpha=0.2, color='blue')
axes[0].set_title('Dipole Power Pattern $\\sim \\sin^2\\theta$', pad=20)
axes[0].set_theta_zero_location('N')
axes[0].set_theta_direction(-1)

# Field pattern (sqrt of power)
axes[1].plot(theta, np.abs(np.sin(theta)), 'r-', linewidth=2)
axes[1].fill(theta, np.abs(np.sin(theta)), alpha=0.2, color='red')
axes[1].set_title('Dipole Field Pattern $\\sim |\\sin\\theta|$', pad=20)
axes[1].set_theta_zero_location('N')
axes[1].set_theta_direction(-1)

plt.tight_layout()
plt.savefig("dipole_radiation_pattern.png", dpi=150)
plt.show()
```

---

## 3. The Larmor Formula

### 3.1 Radiated Power from an Accelerating Charge

A point charge $q$ with acceleration $\mathbf{a}$ radiates power:

$$\boxed{P = \frac{q^2 a^2}{6\pi\epsilon_0 c^3} = \frac{\mu_0 q^2 a^2}{6\pi c}}$$

This is the **Larmor formula**, valid for non-relativistic motion ($v \ll c$).

The angular distribution of radiated power is:

$$\frac{dP}{d\Omega} = \frac{q^2 a^2}{16\pi^2 \epsilon_0 c^3} \sin^2\Theta$$

where $\Theta$ is the angle between the acceleration vector and the observation direction.

### 3.2 Connection to Dipole Radiation

For a dipole $\mathbf{p}(t) = q \mathbf{d}(t)$, the acceleration of the charge is related to $\ddot{\mathbf{p}}$:

$$P = \frac{|\ddot{\mathbf{p}}|^2}{6\pi\epsilon_0 c^3}$$

For harmonic oscillation $\ddot{\mathbf{p}} = -\omega^2 p_0 \hat{z}$, this reproduces the dipole formula.

### 3.3 Relativistic Generalization

For a charge moving at relativistic speeds, the Larmor formula generalizes to:

$$P = \frac{q^2 \gamma^6}{6\pi\epsilon_0 c}\left(a^2 - \frac{|\mathbf{v} \times \mathbf{a}|^2}{c^2}\right)$$

The $\gamma^6$ factor makes synchrotron radiation (from charges moving in circles at relativistic speeds) extremely intense.

```python
def larmor_power(q, a, eps0=8.854e-12, c=3e8):
    """
    Compute power radiated by an accelerating charge (Larmor formula).

    Why Larmor: this formula is the foundation of all radiation physics.
    Every antenna, every synchrotron, every X-ray tube operates on the
    principle that accelerating charges radiate.
    """
    return q**2 * a**2 / (6 * np.pi * eps0 * c**3)

# Example: electron in a TV tube (CRT)
q_e = 1.602e-19   # electron charge (C)
m_e = 9.109e-31   # electron mass (kg)

# Electron accelerated through 25 kV over 10 cm
V_accel = 25e3  # volts
d_accel = 0.1   # meters
a_crt = q_e * V_accel / (m_e * d_accel)

P_crt = larmor_power(q_e, a_crt)
print(f"CRT electron acceleration: {a_crt:.2e} m/s²")
print(f"Radiated power: {P_crt:.2e} W")
print(f"This is negligible compared to kinetic energy gain!")

print()

# Synchrotron electron (v ~ c, circular orbit)
gamma = 1000  # Lorentz factor for ~500 MeV electron
v = 3e8 * np.sqrt(1 - 1/gamma**2)
R = 10  # orbit radius (m)
a_sync = v**2 / R  # centripetal acceleration

# Relativistic Larmor: multiply by gamma^4 for circular motion
P_sync = larmor_power(q_e, a_sync) * gamma**4
print(f"Synchrotron electron (γ={gamma}):")
print(f"  Centripetal acceleration: {a_sync:.2e} m/s²")
print(f"  Radiated power: {P_sync:.2e} W = {P_sync*1e6:.1f} μW per electron")
```

---

## 4. Radiation Resistance

### 4.1 Definition

When a current flows in an antenna, the radiated power can be expressed as:

$$P_{\text{rad}} = \frac{1}{2} I_0^2 R_{\text{rad}}$$

where $I_0$ is the peak current and $R_{\text{rad}}$ is the **radiation resistance** — a fictitious resistance that accounts for the power radiated away. It is "fictitious" because no actual resistor dissipates the energy; instead, it is carried away by electromagnetic waves.

### 4.2 Short Dipole Radiation Resistance

For a short dipole antenna of length $\ell \ll \lambda$, carrying a uniform current $I_0$:

$$R_{\text{rad}} = \frac{2\pi}{3}\eta_0 \left(\frac{\ell}{\lambda}\right)^2 \approx 790 \left(\frac{\ell}{\lambda}\right)^2 \, \Omega$$

where $\eta_0 = \sqrt{\mu_0/\epsilon_0} \approx 377 \, \Omega$ is the impedance of free space.

For $\ell = 0.01\lambda$: $R_{\text{rad}} \approx 0.079 \, \Omega$ — very low, making short antennas inefficient.

### 4.3 Antenna Efficiency

The efficiency of an antenna is:

$$\eta = \frac{R_{\text{rad}}}{R_{\text{rad}} + R_{\text{loss}}}$$

where $R_{\text{loss}}$ is the ohmic resistance of the antenna conductor. For short antennas, $R_{\text{rad}}$ is small and efficiency is poor. This is why AM radio towers (operating at wavelengths of hundreds of meters) must be very tall.

---

## 5. The Half-Wave Dipole

### 5.1 Current Distribution

A center-fed dipole antenna of total length $L = \lambda/2$ carries a sinusoidal current distribution:

$$I(z) = I_0 \cos\left(\frac{\pi z}{L}\right) = I_0 \cos(k z)$$

where $z$ is measured from the center and $|z| \leq L/2$.

### 5.2 Radiation Pattern

The far-field radiation pattern is found by integrating the contributions of all current elements. The result is:

$$E_\theta \propto \frac{\cos\left(\frac{\pi}{2}\cos\theta\right)}{\sin\theta}$$

This pattern is slightly more directional than the short dipole's $\sin\theta$ pattern.

### 5.3 Radiation Resistance and Directivity

For the half-wave dipole:

$$R_{\text{rad}} = 73.1 \, \Omega$$

This is much larger than the short dipole's radiation resistance, making the half-wave dipole practical. The input impedance is $Z_{\text{in}} \approx 73.1 + j42.5 \, \Omega$ (slightly inductive).

The **directivity** is:

$$D = \frac{U_{\max}}{\bar{U}} = 1.64 = 2.15 \, \text{dBi}$$

where dBi means "dB relative to an isotropic radiator."

```python
def half_wave_dipole_pattern(theta):
    """
    Radiation pattern of a half-wave dipole antenna.

    Why the half-wave dipole: it's the most fundamental practical antenna.
    Its ~73 ohm radiation resistance matches common transmission lines,
    and its pattern is the building block for more complex antenna arrays.
    """
    # Avoid division by zero at theta = 0 and pi
    sin_theta = np.sin(theta)
    sin_theta = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)
    return (np.cos(np.pi / 2 * np.cos(theta)) / sin_theta)**2

def compare_antenna_patterns():
    """Compare short dipole vs half-wave dipole radiation patterns."""
    theta = np.linspace(0, 2 * np.pi, 1000)

    # Patterns (normalized)
    short_dipole = np.sin(theta)**2
    half_wave = half_wave_dipole_pattern(theta)
    half_wave_norm = half_wave / half_wave.max()

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    ax.plot(theta, short_dipole, 'b-', linewidth=2, label='Short dipole')
    ax.plot(theta, half_wave_norm, 'r-', linewidth=2, label='Half-wave dipole')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.legend(loc='lower right')
    ax.set_title('Antenna Radiation Patterns', pad=20)

    plt.tight_layout()
    plt.savefig("antenna_patterns.png", dpi=150)
    plt.show()

compare_antenna_patterns()
```

---

## 6. Antenna Arrays

### 6.1 Array Factor

When $N$ identical antennas are arranged along the $z$-axis with spacing $d$, each fed with the same amplitude but with a progressive phase shift $\alpha$, the total radiation pattern is:

$$\text{Array Pattern} = \text{Element Pattern} \times \text{Array Factor}$$

The **array factor** for a uniform linear array is:

$$\text{AF}(\theta) = \sum_{n=0}^{N-1} e^{in(kd\cos\theta + \alpha)} = \frac{\sin\left(\frac{N\psi}{2}\right)}{\sin\left(\frac{\psi}{2}\right)}$$

where $\psi = kd\cos\theta + \alpha$.

### 6.2 Beam Steering

The main beam direction $\theta_0$ occurs when $\psi = 0$:

$$kd\cos\theta_0 + \alpha = 0 \implies \theta_0 = \arccos\left(-\frac{\alpha}{kd}\right)$$

By varying the phase shift $\alpha$, we can **electronically steer the beam** without physically moving the antenna — the principle behind phased array radar.

### 6.3 Array Properties

- **Beamwidth**: Decreases as $\sim 1/N$ (more elements = narrower beam)
- **Sidelobes**: The first sidelobe is about 13.2 dB below the main beam for a uniform array
- **Grating lobes**: Additional main beams appear when $d > \lambda/2$, which is usually undesirable

```python
def array_factor(theta, N, d_lambda, alpha_deg=0):
    """
    Compute array factor for a uniform linear antenna array.

    Parameters:
        theta     : angle from array axis (radians)
        N         : number of elements
        d_lambda  : element spacing in wavelengths (d/lambda)
        alpha_deg : progressive phase shift (degrees)

    Why phased arrays: they enable electronic beam steering in
    microseconds (vs. seconds for mechanical rotation), making them
    essential for modern radar, 5G, and satellite communications.
    """
    k = 2 * np.pi  # k*lambda = 2*pi, so k*d = 2*pi*d/lambda
    alpha = np.radians(alpha_deg)
    psi = k * d_lambda * np.cos(theta) + alpha

    # Array factor (handle the sin(0)/sin(0) case)
    numerator = np.sin(N * psi / 2)
    denominator = np.sin(psi / 2)

    # Avoid division by zero
    af = np.where(np.abs(denominator) < 1e-10, N, numerator / denominator)

    return np.abs(af)**2 / N**2  # normalized

def plot_array_patterns():
    """Demonstrate beam steering and array factor properties."""
    theta = np.linspace(0, np.pi, 1000)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Effect of number of elements (broadside, alpha=0)
    ax = axes[0, 0]
    for N in [2, 4, 8, 16]:
        af = array_factor(theta, N, d_lambda=0.5, alpha_deg=0)
        ax.plot(np.degrees(theta), 10 * np.log10(af + 1e-10),
                linewidth=1.5, label=f'N = {N}')
    ax.set_xlabel('Angle from array axis (degrees)')
    ax.set_ylabel('Array Factor (dB)')
    ax.set_title('Effect of Number of Elements (d = λ/2, broadside)')
    ax.set_ylim(-30, 3)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Beam steering
    ax = axes[0, 1]
    N = 8
    for alpha in [0, -45, -90, -135]:
        af = array_factor(theta, N, d_lambda=0.5, alpha_deg=alpha)
        ax.plot(np.degrees(theta), 10 * np.log10(af + 1e-10),
                linewidth=1.5, label=f'$\\alpha$ = {alpha}°')
    ax.set_xlabel('Angle from array axis (degrees)')
    ax.set_ylabel('Array Factor (dB)')
    ax.set_title(f'Beam Steering (N={N}, d = λ/2)')
    ax.set_ylim(-30, 3)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Polar plot: broadside array
    ax = axes[1, 0]
    ax_polar = fig.add_subplot(2, 2, 3, projection='polar')
    axes[1, 0].remove()

    for N in [4, 8, 16]:
        theta_full = np.linspace(0, 2 * np.pi, 2000)
        af = array_factor(theta_full, N, d_lambda=0.5)
        af_db = 10 * np.log10(af + 1e-10)
        af_db_plot = np.clip(af_db + 30, 0, 30)  # shift for plotting
        ax_polar.plot(theta_full, af_db_plot, linewidth=1.5, label=f'N={N}')

    ax_polar.set_title('Polar Pattern (broadside)', pad=20)
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)
    ax_polar.legend(loc='lower right')

    # Grating lobes: effect of spacing
    ax = axes[1, 1]
    N = 8
    for d_lam in [0.25, 0.5, 0.75, 1.0]:
        af = array_factor(theta, N, d_lambda=d_lam)
        ax.plot(np.degrees(theta), 10 * np.log10(af + 1e-10),
                linewidth=1.5, label=f'd = {d_lam}λ')
    ax.set_xlabel('Angle from array axis (degrees)')
    ax.set_ylabel('Array Factor (dB)')
    ax.set_title(f'Grating Lobes: Effect of Spacing (N={N})')
    ax.set_ylim(-30, 3)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("antenna_arrays.png", dpi=150)
    plt.show()

plot_array_patterns()
```

---

## 7. Directivity and Gain

### 7.1 Definitions

**Directivity** is the ratio of peak radiation intensity to the average:

$$D = \frac{U_{\max}}{\bar{U}} = \frac{4\pi U_{\max}}{P_{\text{rad}}}$$

where $U(\theta, \phi) = r^2 |\langle \mathbf{S} \rangle|$ is the radiation intensity (W/sr).

**Gain** includes antenna efficiency:

$$G = \eta \, D$$

**Effective area** relates the power captured by a receiving antenna to the incident power density:

$$A_e = \frac{\lambda^2}{4\pi} G$$

### 7.2 Standard Antenna Parameters

| Antenna | Directivity | Gain (typical) | Radiation Resistance |
|---------|-------------|-----------------|---------------------|
| Isotropic | 1 (0 dBi) | — | — |
| Short dipole ($\ell \ll \lambda$) | 1.5 (1.76 dBi) | $\sim$0.5–1.0 dBi | $790(\ell/\lambda)^2$ $\Omega$ |
| Half-wave dipole | 1.64 (2.15 dBi) | $\sim$2.0 dBi | 73.1 $\Omega$ |
| Quarter-wave monopole | 3.28 (5.15 dBi) | $\sim$4.5 dBi | 36.5 $\Omega$ |

### 7.3 Friis Transmission Equation

For two antennas separated by distance $R$, the received power is:

$$\frac{P_r}{P_t} = G_t G_r \left(\frac{\lambda}{4\pi R}\right)^2$$

The factor $(\lambda / 4\pi R)^2$ is the **free-space path loss**.

```python
def friis_link_budget(Pt_dBm, Gt_dBi, Gr_dBi, freq_GHz, distance_km):
    """
    Compute received power using the Friis equation.

    Why Friis: this is the fundamental equation for all wireless
    link design — from satellite communications to Bluetooth.
    """
    c = 3e8
    lam = c / (freq_GHz * 1e9)
    R = distance_km * 1e3

    # Free space path loss in dB
    FSPL_dB = 20 * np.log10(4 * np.pi * R / lam)

    Pr_dBm = Pt_dBm + Gt_dBi + Gr_dBi - FSPL_dB

    print(f"Friis Link Budget")
    print(f"=================")
    print(f"Frequency:    {freq_GHz} GHz (λ = {lam*1e3:.1f} mm)")
    print(f"Distance:     {distance_km} km")
    print(f"Tx power:     {Pt_dBm} dBm")
    print(f"Tx gain:      {Gt_dBi} dBi")
    print(f"Rx gain:      {Gr_dBi} dBi")
    print(f"Path loss:    {FSPL_dB:.1f} dB")
    print(f"Rx power:     {Pr_dBm:.1f} dBm = {10**(Pr_dBm/10) * 1e-3:.2e} W")

    return Pr_dBm

# Example: Wi-Fi link
friis_link_budget(Pt_dBm=20, Gt_dBi=3, Gr_dBi=3,
                  freq_GHz=5.0, distance_km=0.05)

print()

# Example: Satellite link
friis_link_budget(Pt_dBm=43, Gt_dBi=40, Gr_dBi=35,
                  freq_GHz=12.0, distance_km=36000)
```

---

## 8. Visualization: Dipole Radiation in Time

```python
def animate_dipole_radiation(save=False):
    """
    Visualize the time evolution of electric field lines from
    an oscillating electric dipole.

    Why visualize: static patterns do not convey the dynamic nature
    of radiation — seeing the fields propagate outward gives
    physical intuition about how energy leaves the source.
    """
    # Create field on a grid
    N = 200
    x = np.linspace(-5, 5, N)
    z = np.linspace(-5, 5, N)
    X, Z = np.meshgrid(x, z)
    R = np.sqrt(X**2 + Z**2)
    R = np.where(R < 0.3, 0.3, R)  # avoid singularity at origin

    theta = np.arctan2(X, Z)  # angle from z-axis (dipole axis)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    phases = [0, np.pi / 3, 2 * np.pi / 3]

    for ax, phase in zip(axes, phases):
        # Far-field radiation pattern: E ~ sin(theta)/r * cos(kr - wt)
        k = 2 * np.pi  # wavelength = 1 unit
        E_theta = np.sin(theta) / R * np.cos(k * R - phase)

        # Convert to Cartesian components for vector plot
        Ex = E_theta * np.cos(theta)
        Ez = -E_theta * np.sin(theta)
        E_mag = np.sqrt(Ex**2 + Ez**2)

        # Intensity plot
        im = ax.pcolormesh(X, Z, E_theta, cmap='RdBu_r', shading='auto',
                           vmin=-2, vmax=2)

        # Mark the dipole
        ax.plot(0, 0, 'ko', markersize=10)
        ax.arrow(0, -0.15, 0, 0.3, head_width=0.1, head_length=0.05,
                 fc='yellow', ec='yellow')

        ax.set_xlabel('x / λ')
        ax.set_ylabel('z / λ')
        ax.set_title(f'ωt = {phase/np.pi:.1f}π')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='$E_\\theta$', shrink=0.7)

    plt.suptitle('Oscillating Dipole Radiation (Electric Field)', fontsize=14)
    plt.tight_layout()
    plt.savefig("dipole_radiation_time.png", dpi=150)
    plt.show()

animate_dipole_radiation()
```

---

## Summary

| Concept | Key Formula | Physical Meaning |
|---------|-------------|------------------|
| Retarded potential | $V = \frac{1}{4\pi\epsilon_0}\int \frac{\rho(\mathbf{r}', t_r)}{|\mathbf{r}-\mathbf{r}'|} d^3r'$ | Causality: fields encode past source state |
| Dipole radiation | $P = \frac{p_0^2\omega^4}{12\pi\epsilon_0 c^3}$ | Power scales as $\omega^4$ |
| Larmor formula | $P = \frac{q^2 a^2}{6\pi\epsilon_0 c^3}$ | Accelerating charges radiate |
| Radiation resistance | $P_{\text{rad}} = \frac{1}{2}I_0^2 R_{\text{rad}}$ | Equivalent resistance for radiated power |
| Half-wave dipole | $R_{\text{rad}} = 73.1\,\Omega$ | Practical match to transmission lines |
| Array factor | $\text{AF} = \frac{\sin(N\psi/2)}{\sin(\psi/2)}$ | Pattern multiplication |
| Friis equation | $P_r/P_t = G_t G_r (\lambda/4\pi R)^2$ | Wireless link power budget |

---

## Exercises

### Exercise 1: Retarded Potential of a Switched Current
A long wire carries zero current for $t < 0$ and current $I_0$ for $t > 0$. Compute the magnetic field at a distance $s$ from the wire as a function of time, using retarded potentials. Show that the field "turns on" from the nearest point of the wire and gradually extends as more of the wire contributes.

### Exercise 2: Magnetic Dipole Radiation
An oscillating magnetic dipole $\mathbf{m}(t) = m_0 \cos(\omega t) \hat{z}$ radiates with a pattern similar to the electric dipole but with $\mathbf{E}$ and $\mathbf{B}$ swapped. (a) Write down the far-field electric field. (b) Compute the total radiated power. (c) Show that the ratio of magnetic to electric dipole radiation power is $(m_0 \omega / p_0 c^2)^2$.

### Exercise 3: Phased Array Design
Design an 8-element phased array operating at 10 GHz with half-wavelength spacing. (a) Plot the array factor for beam steering angles of 0, 30, 45, and 60 degrees from broadside. (b) Determine the 3 dB beamwidth for each case. (c) What is the maximum steering angle before a grating lobe appears?

### Exercise 4: Satellite Link Budget
A geostationary satellite (altitude 36,000 km) transmits at 12 GHz with 20 W power through a 1-meter dish (gain $\approx$ 40 dBi). (a) Calculate the free-space path loss. (b) If the ground station has a 3-meter dish (gain $\approx$ 50 dBi), what is the received power? (c) Is this above the thermal noise floor of a 30 MHz bandwidth receiver at 290 K?

---

[← Previous: 12. Waveguides and Cavities](12_Waveguides_and_Cavities.md) | [Next: 14. Relativistic Electrodynamics →](14_Relativistic_Electrodynamics.md)
