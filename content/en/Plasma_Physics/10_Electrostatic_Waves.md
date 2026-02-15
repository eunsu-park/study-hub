# 10. Electrostatic Waves

## Learning Objectives

- Understand the general framework for electrostatic wave dispersion relations
- Derive the Bohm-Gross dispersion for Langmuir waves in warm plasmas
- Analyze ion acoustic waves and their damping conditions
- Compute upper hybrid and lower hybrid resonances in magnetized plasmas
- Learn about Bernstein modes and their role in wave heating
- Solve electrostatic dispersion relations numerically for various plasma parameters

## Introduction

In the previous lessons, we studied MHD and electromagnetic waves, which involve magnetic field perturbations. Now we focus on **electrostatic waves**, where:

$$\mathbf{B}_1 = 0, \quad \mathbf{E}_1 = -\nabla\phi_1$$

These waves involve only electric field fluctuations and can be derived from the scalar potential. They arise from charge separation and are governed by Poisson's equation:

$$\nabla \cdot \mathbf{E}_1 = \frac{\rho_1}{\epsilon_0}$$

Electrostatic waves are important for:
- Plasma heating and current drive (electron cyclotron, lower hybrid)
- Diagnostics (Thomson scattering, collective modes)
- Understanding instabilities (two-stream, drift waves)
- Wave-particle interactions (Landau damping, quasilinear theory)

The key tool is the **dielectric function** $\epsilon(\mathbf{k}, \omega)$, which encodes the plasma's response to electromagnetic perturbations.

## 1. General Electrostatic Dispersion Framework

### 1.1 Dielectric Function and Susceptibility

For a perturbation $\propto e^{i(\mathbf{k}\cdot\mathbf{x} - \omega t)}$, Poisson's equation becomes:

$$\mathbf{k} \cdot \mathbf{E}_1 = \frac{\rho_1}{\epsilon_0}$$

The induced charge density $\rho_1$ depends on the electric field through the plasma response. Define the **dielectric function** $\epsilon(\mathbf{k}, \omega)$ such that:

$$\mathbf{k} \cdot \mathbf{D} = \mathbf{k} \cdot (\epsilon_0 \epsilon \mathbf{E}_1) = 0$$

For electrostatic waves, this gives:

$$\boxed{\epsilon(\mathbf{k}, \omega) = 0}$$

This is the **electrostatic dispersion relation**.

The dielectric function is related to the **susceptibility** $\chi_s$ of each species:

$$\epsilon = 1 + \sum_s \chi_s(\mathbf{k}, \omega)$$

The susceptibility describes how species $s$ responds to the electric field:

$$\chi_s = \frac{n_{1s}/n_0}{\epsilon_0 E_1 / (e_s n_0)}$$

From the linearized Vlasov equation, the susceptibility for species $s$ is:

$$\chi_s(\mathbf{k}, \omega) = -\frac{\omega_{ps}^2}{k^2} \int \frac{\mathbf{k} \cdot \partial f_0/\partial \mathbf{v}}{\omega - \mathbf{k}\cdot\mathbf{v}} d^3v$$

where $\omega_{ps}^2 = n_0 e_s^2 / (\epsilon_0 m_s)$ is the plasma frequency.

### 1.2 Cold Plasma Limit

For a **cold plasma** ($T = 0$, $f_0 = n_0 \delta(\mathbf{v})$), the integral simplifies. Particles respond to the wave electric field via:

$$m \frac{d\mathbf{v}_1}{dt} = e \mathbf{E}_1$$

For harmonic perturbations $\propto e^{-i\omega t}$:
$$-i\omega m \mathbf{v}_1 = e \mathbf{E}_1$$

The density perturbation from continuity is:
$$-i\omega n_1 + n_0 i\mathbf{k}\cdot\mathbf{v}_1 = 0$$

Combining:
$$n_1 = \frac{n_0 \mathbf{k}\cdot\mathbf{v}_1}{\omega} = \frac{n_0 e \mathbf{k}\cdot\mathbf{E}_1}{m\omega^2}$$

The susceptibility is:
$$\chi_s = \frac{n_1 e_s}{\epsilon_0 E_1 n_0} = -\frac{\omega_{ps}^2}{\omega^2}$$

The dielectric function for multiple cold species:

$$\epsilon = 1 - \sum_s \frac{\omega_{ps}^2}{\omega^2}$$

### 1.3 Warm Plasma: Kinetic Effects

For a **warm plasma** with thermal velocity $v_{th}$, kinetic effects become important when:

$$\frac{\omega}{k} \sim v_{th}$$

The phase velocity $v_\phi = \omega/k$ is comparable to particle velocities. This leads to:
- **Landau damping**: resonant wave-particle interaction
- **Thermal dispersion**: wave frequency depends on $k v_{th}$

For a Maxwellian distribution:
$$f_0(\mathbf{v}) = n_0 \left(\frac{m}{2\pi k_B T}\right)^{3/2} \exp\left(-\frac{m v^2}{2k_B T}\right)$$

the 1D susceptibility (for $\mathbf{k} = k \hat{z}$) involves the **plasma dispersion function** $Z(\zeta)$:

$$\chi_s = -\frac{\omega_{ps}^2}{k^2 v_{th,s}^2} \left[1 + \zeta_s Z(\zeta_s)\right]$$

where $\zeta_s = \omega/(k v_{th,s})$ and:

$$Z(\zeta) = \frac{1}{\sqrt{\pi}} \int_{-\infty}^{\infty} \frac{e^{-x^2}}{x - \zeta} dx$$

For large $|\zeta|$ (phase velocity $\gg v_{th}$):
$$Z(\zeta) \approx -\frac{1}{\zeta}\left(1 + \frac{1}{2\zeta^2} + \frac{3}{4\zeta^4} + \cdots\right)$$

This gives thermal corrections to the cold plasma result.

## 2. Langmuir Waves (Electron Plasma Oscillations)

### 2.1 Cold Plasma: Pure Oscillation

In an unmagnetized plasma with stationary ions, the cold plasma dispersion is:

$$\epsilon = 1 - \frac{\omega_{pe}^2}{\omega^2} = 0$$

This gives:
$$\boxed{\omega = \omega_{pe}}$$

These are **Langmuir waves** (or electron plasma oscillations), first observed by Langmuir and Tonks in 1929.

Properties:
- **Dispersionless**: $\omega$ independent of $k$
- **No propagation**: group velocity $v_g = d\omega/dk = 0$
- **Standing oscillation**: electrons oscillate collectively
- **High frequency**: $\omega_{pe} \sim 10^{9}-10^{11}$ rad/s for typical plasmas

Physical picture:
```
Time t=0:           Time t=T/4:         Time t=T/2:
-  +  -  +  -       - - -  +  +  +      -  +  -  +  -
Ions (stationary)   Electrons          Back to equilibrium
                    displaced
```

### 2.2 Warm Plasma: Bohm-Gross Dispersion

Including electron thermal motion, the dispersion becomes:

$$1 - \frac{\omega_{pe}^2}{k^2 v_{th,e}^2}[1 + \zeta_e Z(\zeta_e)] = 0$$

For $\omega \gg k v_{th,e}$ (weak thermal effects), expand to first order:

$$1 - \frac{\omega_{pe}^2}{\omega^2}\left(1 + \frac{3k^2 v_{th,e}^2}{\omega^2}\right) = 0$$

Solving for $\omega^2$:

$$\boxed{\omega^2 = \omega_{pe}^2 + 3k^2 v_{th,e}^2}$$

This is the **Bohm-Gross dispersion relation** (1949), where $v_{th,e} = \sqrt{k_B T_e / m_e}$.

Properties:
- **Dispersive**: $\omega$ depends on $k$
- **Phase velocity**: $v_\phi = \omega/k = \sqrt{\omega_{pe}^2/k^2 + 3v_{th,e}^2}$
- **Group velocity**: $v_g = d\omega/dk = 3k v_{th,e}^2/\omega$
- **Cutoff**: minimum frequency $\omega_{pe}$ at $k = 0$

The thermal correction is important when:
$$k\lambda_D \sim 1$$

where $\lambda_D = v_{th,e}/\omega_{pe}$ is the Debye length.

### 2.3 Landau Damping of Langmuir Waves

For $\omega/k \sim v_{th,e}$, the imaginary part of $Z(\zeta)$ gives damping:

$$\text{Im}\,\omega = -\sqrt{\frac{\pi}{8}} \frac{\omega_{pe}}{k^3 \lambda_D^3} \exp\left(-\frac{\omega_{pe}^2}{2k^2 v_{th,e}^2} - \frac{3}{2}\right)$$

The damping rate decreases exponentially for $k\lambda_D \ll 1$.

Physical picture: particles moving at $v \approx \omega/k$ can exchange energy with the wave resonantly. If there are more slow particles than fast particles (Maxwellian has negative slope at high $v$), the wave loses energy.

## 3. Ion Acoustic Waves

### 3.1 Derivation: Two-Fluid Model

Consider **low-frequency** oscillations where:
- Electrons respond adiabatically (fast, isothermal)
- Ions respond dynamically (slow, inertial)

**Electron response** (Boltzmann relation):
$$n_e = n_0 \exp\left(\frac{e\phi}{k_B T_e}\right) \approx n_0\left(1 + \frac{e\phi}{k_B T_e}\right)$$

**Ion continuity and momentum**:
$$\frac{\partial n_i}{\partial t} + \nabla\cdot(n_i \mathbf{v}_i) = 0$$

$$m_i n_i \frac{\partial \mathbf{v}_i}{\partial t} = -e n_i \nabla\phi - \nabla(n_i k_B T_i)$$

**Poisson equation**:
$$\epsilon_0 \nabla^2 \phi = e(n_e - n_i)$$

Linearizing with $n_e = n_0 + n_{e1}$, $n_i = n_0 + n_{i1}$, $\phi = \phi_1$, and assuming $e^{i(kx - \omega t)}$:

$$-\omega^2 m_i n_{i1} = -k^2 e n_{i1} \phi_1 - k^2 n_{i1} k_B T_i$$

$$-k^2 \epsilon_0 \phi_1 = e\left(\frac{e n_0 \phi_1}{k_B T_e} - n_{i1}\right)$$

Combining:
$$-k^2 \epsilon_0 \phi_1 = e\left(\frac{e n_0 \phi_1}{k_B T_e} - \frac{k^2 e \phi_1 (n_0 + k_B T_i k^2/\omega^2 m_i)}{\omega^2 m_i - k^2 k_B T_i}\right)$$

After algebra, the dispersion relation is:

$$\omega^2 = \frac{k^2 k_B T_e}{m_i (1 + k^2 \lambda_D^2)} \left(\frac{1 + 3T_i/T_e}{1 + k^2\lambda_D^2}\right)$$

For $k\lambda_D \ll 1$:

$$\boxed{\omega \approx k c_s}$$

where the **ion acoustic speed** is:

$$\boxed{c_s = \sqrt{\frac{k_B T_e}{m_i}}}$$

assuming $T_i \ll T_e$.

### 3.2 Physical Picture

Ion acoustic waves are analogous to sound waves in a neutral gas, but:
- Restoring force: electron pressure (not ion pressure)
- Inertia: ion mass
- Electrons act as a thermodynamic reservoir

```
Ion density:     High    Low     High    Low
                  |       |       |       |
                  v       ^       v       ^
                 +++     ---     +++     ---
Electron cloud: (responds adiabatically)

Wave propagates at c_s ~ √(Te/mi)
```

Typical speeds:
- $T_e = 10$ eV, hydrogen: $c_s \approx 50$ km/s
- $T_e = 1$ keV, deuterium: $c_s \approx 200$ km/s

Much slower than electron thermal velocity ($v_{th,e} \sim 10^3$ km/s), but faster than ion thermal velocity ($v_{th,i} \sim 10$ km/s).

### 3.3 Ion Landau Damping

For $\omega/k \sim v_{th,i}$, ions moving near the wave phase velocity can damp the wave:

$$\gamma \approx -\sqrt{\frac{\pi}{8}} \frac{c_s}{k\lambda_D} \left(\frac{T_i}{T_e}\right)^{3/2} \exp\left(-\frac{1}{2k^2\lambda_D^2} - \frac{3}{2}\right)$$

The condition for **weak damping** is:

$$T_e \gg T_i$$

If $T_e \sim T_i$, the waves are heavily damped and don't propagate effectively.

This is observed in lab plasmas:
- Hot electrons, cold ions: strong ion acoustic waves
- Comparable temperatures: waves damped out

### 3.4 Role in Turbulence

Ion acoustic turbulence is ubiquitous in plasmas:
- Driven by current (ion acoustic instability)
- Nonlinear interactions create spectrum of modes
- Leads to anomalous resistivity, heating

Observed in:
- Ionosphere (radar scatter)
- Solar wind
- Tokamak edge plasmas

## 4. Waves in Magnetized Plasmas

### 4.1 Upper Hybrid Resonance

In a magnetized plasma, electrons gyrate at $\omega_{ce}$ while also oscillating at $\omega_{pe}$. For waves propagating perpendicular to $\mathbf{B}$ ($\mathbf{k} \perp \mathbf{B}$), the two motions couple.

The dispersion relation for electrostatic waves across $\mathbf{B}$ is:

$$\epsilon_{\perp} = 1 - \frac{\omega_{pe}^2}{\omega^2 - \omega_{ce}^2} = 0$$

This gives:

$$\boxed{\omega_{UH}^2 = \omega_{pe}^2 + \omega_{ce}^2}$$

This is the **upper hybrid frequency**.

Physical picture:
- Electrons oscillate at $\omega_{pe}$
- Magnetic field adds restoring force from $\mathbf{v} \times \mathbf{B}$
- Effective frequency is geometric mean of the two

The upper hybrid layer (where $\omega = \omega_{UH}$) is used for:
- Plasma heating (upper hybrid resonance heating)
- Current drive
- Diagnostics (reflectometry)

### 4.2 Lower Hybrid Resonance

For lower frequencies involving ion motion, the **lower hybrid frequency** arises from coupling of ion cyclotron and ion plasma oscillations:

$$\frac{1}{\omega_{LH}^2} = \frac{1}{\omega_{ci}\omega_{ce}} + \frac{1}{\omega_{pi}^2 + \omega_{ci}^2}$$

For $\omega_{pi}^2 \gg \omega_{ci}^2$:

$$\boxed{\omega_{LH} \approx \sqrt{\frac{\omega_{ci}\omega_{ce}}{1 + \omega_{pe}^2/\omega_{ce}^2}} \approx \sqrt{\omega_{ci}\omega_{ce}}}$$

(assuming $\omega_{pe}^2 \ll \omega_{ce}^2$).

Typical values:
- $B = 5$ T, $n = 10^{20}$ m$^{-3}$, hydrogen
- $f_{ce} \approx 140$ GHz, $f_{ci} \approx 76$ MHz
- $f_{LH} \approx 3.3$ GHz

Lower hybrid waves are used extensively for:
- **Current drive**: can drive currents efficiently via Landau damping
- Heating (though less efficient than ECRH or ICRH)
- Plasma startup in tokamaks

The waves can propagate at specific angles to $\mathbf{B}$, allowing localized deposition.

### 4.3 Warm Plasma Corrections

Including finite temperature, the cutoff and resonances shift:

$$\omega_{UH}^2 = \omega_{pe}^2 + \omega_{ce}^2 + 3k^2 v_{th,e}^2$$

For lower hybrid:
$$\omega_{LH}^2 \approx \omega_{ci}\omega_{ce} \left(1 + \frac{\omega_{pe}^2}{\omega_{ce}^2}\right)^{-1}\left(1 + k^2\lambda_D^2\right)$$

The thermal corrections become important when $k\lambda_D \sim 1$ or $k\rho_L \sim 1$ (where $\rho_L$ is Larmor radius).

## 5. Bernstein Modes

### 5.1 Electron Bernstein Modes

At harmonics of the electron cyclotron frequency, **electrostatic** modes can propagate even when electromagnetic waves are cutoff. These are **Bernstein modes**.

The dispersion relation involves Bessel functions. Near the $n$-th harmonic:

$$\omega \approx n\omega_{ce}\left(1 + \frac{k^2 v_{th,e}^2}{2\omega_{ce}^2}\right)$$

These modes exist for $\omega \approx n\omega_{ce}$, $n = 1, 2, 3, \ldots$

Properties:
- Electrostatic (no $\mathbf{B}_1$)
- Can propagate where EM waves cannot
- Strong interaction with resonant electrons
- Used for heating and current drive

The hot plasma dielectric for $\mathbf{k} \perp \mathbf{B}$ is:

$$\epsilon_{\perp} = 1 + \frac{\omega_{pe}^2}{k^2 v_{th,e}^2} \sum_{n=-\infty}^{\infty} \frac{I_n(\lambda) e^{-\lambda}}{\omega/\omega_{ce} - n} \left(\frac{\omega}{k v_{th,e}}\right)Z\left(\frac{\omega - n\omega_{ce}}{k v_{th,e}}\right)$$

where $\lambda = k^2 \rho_L^2$ and $I_n$ is the modified Bessel function.

### 5.2 Ion Bernstein Modes

Similarly, ion Bernstein modes exist near harmonics of $\omega_{ci}$:

$$\omega \approx n\omega_{ci}\left(1 + \frac{k^2 v_{th,i}^2}{2\omega_{ci}^2}\right)$$

These are important for:
- Ion heating in dense plasmas
- Mode conversion heating (EM wave converts to electrostatic Bernstein mode)
- Understanding kinetic Alfvén waves at small scales

### 5.3 Mode Conversion

In inhomogeneous plasmas, electromagnetic waves can **mode convert** to electrostatic waves at specific locations:

```
EM wave (O-mode or X-mode)
       |
       v
  Cutoff/Resonance layer
       |
       v
Bernstein wave (electrostatic)
       |
       v
   Absorption via Landau/cyclotron damping
```

This is used in **mode conversion heating** schemes, avoiding cutoff issues with direct EM wave injection.

## 6. Python Implementation

### 6.1 Langmuir Wave Dispersion

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz

def plasma_dispersion_Z(zeta):
    """
    Plasma dispersion function Z(ζ).

    Z(ζ) = (1/√π) ∫ exp(-x²)/(x - ζ) dx

    Uses Faddeeva function: Z(ζ) = i√π w(ζ) where w(z) = exp(-z²)erfc(-iz)
    """
    return 1j * np.sqrt(np.pi) * wofz(zeta)

def langmuir_dispersion_cold(k, omega_pe):
    """
    Cold plasma dispersion: ω = ω_pe (independent of k).
    """
    return omega_pe * np.ones_like(k)

def langmuir_dispersion_warm(k, omega_pe, v_th):
    """
    Warm plasma dispersion (Bohm-Gross): ω² = ω_pe² + 3k²v_th².
    """
    return np.sqrt(omega_pe**2 + 3 * k**2 * v_th**2)

def langmuir_dispersion_kinetic(k, omega_pe, v_th, num_points=100):
    """
    Solve full kinetic dispersion using plasma dispersion function.

    ε = 1 - (ω_pe²/k²v_th²)[1 + ζZ(ζ)] = 0
    where ζ = ω/(kv_th)
    """
    omega_solutions = []

    for k_val in k:
        if k_val == 0:
            omega_solutions.append(omega_pe)
            continue

        # Search for real part of ω
        omega_range = np.linspace(omega_pe, omega_pe * 1.5, num_points)

        epsilon = []
        for omega in omega_range:
            zeta = omega / (k_val * v_th)
            Z_val = plasma_dispersion_Z(zeta)
            eps = 1 - (omega_pe**2 / (k_val**2 * v_th**2)) * (1 + zeta * Z_val)
            epsilon.append(np.real(eps))

        epsilon = np.array(epsilon)

        # Find zero crossing
        sign_changes = np.where(np.diff(np.sign(epsilon)))[0]
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            omega_solutions.append(omega_range[idx])
        else:
            omega_solutions.append(omega_range[-1])

    return np.array(omega_solutions)

# Parameters
n = 1e19  # m^-3
T_e = 10  # eV
m_e = 9.109e-31  # kg
e = 1.602e-19  # C
epsilon_0 = 8.854e-12  # F/m
k_B = 1.381e-23  # J/K

omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
v_th = np.sqrt(2 * k_B * T_e * e / m_e)
lambda_D = v_th / omega_pe

print(f"Plasma frequency: f_pe = {omega_pe / (2*np.pi) / 1e9:.2f} GHz")
print(f"Thermal velocity: v_th = {v_th / 1e6:.2f} Mm/s")
print(f"Debye length: λ_D = {lambda_D * 1e6:.2f} μm")

# Wavenumber array
k = np.linspace(0.01, 2, 100) / lambda_D  # Normalize to 1/λ_D

# Compute dispersions
omega_cold = langmuir_dispersion_cold(k, omega_pe)
omega_warm = langmuir_dispersion_warm(k, omega_pe, v_th)
omega_kinetic = langmuir_dispersion_kinetic(k, omega_pe, v_th)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Dispersion relation
ax1.plot(k * lambda_D, omega_cold / omega_pe, 'k--', linewidth=2, label='Cold plasma')
ax1.plot(k * lambda_D, omega_warm / omega_pe, 'b-', linewidth=2, label='Bohm-Gross')
ax1.plot(k * lambda_D, omega_kinetic / omega_pe, 'r:', linewidth=2, label='Kinetic (full)')
ax1.set_xlabel('$k\\lambda_D$', fontsize=13)
ax1.set_ylabel('$\\omega / \\omega_{pe}$', fontsize=13)
ax1.set_title('Langmuir Wave Dispersion', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.95, 1.4])

# Phase and group velocity
v_phase_warm = omega_warm / k
v_group_warm = 3 * k * v_th**2 / omega_warm

ax2.plot(k * lambda_D, v_phase_warm / v_th, 'b-', linewidth=2, label='$v_\\phi$ (Bohm-Gross)')
ax2.plot(k * lambda_D, v_group_warm / v_th, 'r--', linewidth=2, label='$v_g$ (Bohm-Gross)')
ax2.axhline(1, color='k', linestyle=':', alpha=0.5, label='$v_{th}$')
ax2.set_xlabel('$k\\lambda_D$', fontsize=13)
ax2.set_ylabel('Velocity / $v_{th}$', fontsize=13)
ax2.set_title('Phase and Group Velocity', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('langmuir_dispersion.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 6.2 Ion Acoustic Wave Dispersion

```python
def ion_acoustic_dispersion(k, n, T_e, T_i, m_i, Z=1):
    """
    Ion acoustic wave dispersion with damping.

    Parameters:
    -----------
    k : array
        Wavenumber (m^-1)
    n : float
        Density (m^-3)
    T_e : float
        Electron temperature (eV)
    T_i : float
        Ion temperature (eV)
    m_i : float
        Ion mass (kg)
    Z : int
        Ion charge number

    Returns:
    --------
    omega_real : array
        Real frequency
    gamma : array
        Damping rate
    """
    T_e_J = T_e * e
    T_i_J = T_i * e

    # Ion acoustic speed
    c_s = np.sqrt(k_B * T_e_J / m_i)

    # Debye length
    v_th_e = np.sqrt(2 * k_B * T_e_J / m_e)
    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    lambda_D = v_th_e / omega_pe

    # Real frequency
    omega_real = k * c_s / np.sqrt(1 + k**2 * lambda_D**2)

    # Damping rate (approximate, for T_e >> T_i)
    v_th_i = np.sqrt(2 * k_B * T_i_J / m_i)
    zeta_i = omega_real / (k * v_th_i)

    # Landau damping (asymptotic form for large ζ)
    gamma = -np.sqrt(np.pi/8) * (omega_real / (k * lambda_D)) * \
            (T_i / T_e)**(3/2) * np.exp(-1/(2*k**2*lambda_D**2) - 3/2)

    return omega_real, gamma

# Parameters
n = 1e19  # m^-3
T_e = 10  # eV
m_i = 1.673e-27  # kg (proton)
T_i_array = np.array([0.1, 0.5, 1.0, 5.0])  # eV

# Compute Debye length
v_th_e = np.sqrt(2 * k_B * T_e * e / m_e)
omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
lambda_D = v_th_e / omega_pe
c_s = np.sqrt(k_B * T_e * e / m_i)

print(f"Ion acoustic speed: c_s = {c_s / 1e3:.1f} km/s")

k = np.linspace(0.01, 3, 200) / lambda_D

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = ['blue', 'green', 'orange', 'red']

for i, T_i in enumerate(T_i_array):
    omega_real, gamma = ion_acoustic_dispersion(k, n, T_e, T_i, m_i)

    ax1.plot(k * lambda_D, omega_real * lambda_D / c_s, color=colors[i],
             linewidth=2, label=f'$T_i = {T_i}$ eV')

    ax2.semilogy(k * lambda_D, np.abs(gamma) / omega_real, color=colors[i],
                 linewidth=2, label=f'$T_i = {T_i}$ eV')

# Add linear dispersion line
k_linear = k[k * lambda_D < 0.5]
ax1.plot(k_linear * lambda_D, k_linear * lambda_D, 'k--', alpha=0.5,
         label='$\\omega = kc_s$')

ax1.set_xlabel('$k\\lambda_D$', fontsize=13)
ax1.set_ylabel('$\\omega\\lambda_D / c_s$', fontsize=13)
ax1.set_title('Ion Acoustic Dispersion', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('$k\\lambda_D$', fontsize=13)
ax2.set_ylabel('$|\\gamma| / \\omega$', fontsize=13)
ax2.set_title('Landau Damping Rate', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([1e-6, 1])

plt.tight_layout()
plt.savefig('ion_acoustic_dispersion.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 6.3 Upper and Lower Hybrid Frequencies

```python
def hybrid_frequencies(n, B, Z=1, A=1):
    """
    Compute upper and lower hybrid frequencies.

    Parameters:
    -----------
    n : array
        Density (m^-3)
    B : array
        Magnetic field (T)
    Z : int
        Ion charge
    A : float
        Ion mass number

    Returns:
    --------
    dict with frequencies
    """
    m_i = A * 1.673e-27

    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    omega_pi = np.sqrt(n * Z**2 * e**2 / (epsilon_0 * m_i))
    omega_ce = e * B / m_e
    omega_ci = Z * e * B / m_i

    omega_UH = np.sqrt(omega_pe**2 + omega_ce**2)

    # Lower hybrid
    term1 = 1 / (omega_ci * omega_ce)
    term2 = 1 / (omega_pi**2 + omega_ci**2)
    omega_LH = 1 / np.sqrt(term1 + term2)

    return {
        'omega_pe': omega_pe,
        'omega_pi': omega_pi,
        'omega_ce': omega_ce,
        'omega_ci': omega_ci,
        'omega_UH': omega_UH,
        'omega_LH': omega_LH
    }

# Parameter ranges
B_array = np.linspace(0.5, 10, 100)  # T
n_ref = 1e20  # m^-3

freq = hybrid_frequencies(n_ref, B_array, Z=1, A=2)  # Deuterium

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Upper hybrid
ax1.plot(B_array, freq['omega_pe'] / (2*np.pi) / 1e9, 'b--',
         linewidth=2, label='$f_{pe}$')
ax1.plot(B_array, freq['omega_ce'] / (2*np.pi) / 1e9, 'r--',
         linewidth=2, label='$f_{ce}$')
ax1.plot(B_array, freq['omega_UH'] / (2*np.pi) / 1e9, 'k-',
         linewidth=2.5, label='$f_{UH}$')
ax1.set_xlabel('Magnetic Field (T)', fontsize=13)
ax1.set_ylabel('Frequency (GHz)', fontsize=13)
ax1.set_title(f'Upper Hybrid Frequency ($n = {n_ref:.0e}$ m$^{{-3}}$)', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Lower hybrid
ax2.plot(B_array, freq['omega_ci'] / (2*np.pi) / 1e6, 'r--',
         linewidth=2, label='$f_{ci}$')
ax2.plot(B_array, freq['omega_LH'] / (2*np.pi) / 1e6, 'k-',
         linewidth=2.5, label='$f_{LH}$')
ax2.set_xlabel('Magnetic Field (T)', fontsize=13)
ax2.set_ylabel('Frequency (MHz)', fontsize=13)
ax2.set_title(f'Lower Hybrid Frequency ($n = {n_ref:.0e}$ m$^{{-3}}$)', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hybrid_frequencies.png', dpi=150, bbox_inches='tight')
plt.show()

# Print typical ITER values
B_ITER = 5.3  # T
n_ITER = 1e20  # m^-3
freq_ITER = hybrid_frequencies(n_ITER, B_ITER, Z=1, A=2)

print(f"\nITER-like parameters (B = {B_ITER} T, n = {n_ITER:.0e} m^-3):")
print(f"  f_pe = {freq_ITER['omega_pe'][0] / (2*np.pi) / 1e9:.1f} GHz")
print(f"  f_ce = {freq_ITER['omega_ce'][0] / (2*np.pi) / 1e9:.1f} GHz")
print(f"  f_UH = {freq_ITER['omega_UH'][0] / (2*np.pi) / 1e9:.1f} GHz")
print(f"  f_ci = {freq_ITER['omega_ci'][0] / (2*np.pi) / 1e6:.1f} MHz")
print(f"  f_LH = {freq_ITER['omega_LH'][0] / (2*np.pi) / 1e6:.1f} MHz")
```

### 6.4 Bernstein Mode Dispersion

```python
from scipy.special import iv  # Modified Bessel function

def bernstein_mode_approximate(n_harmonic, k_perp, rho_L, omega_c):
    """
    Approximate Bernstein mode dispersion near n-th harmonic.

    ω ≈ nω_c(1 + k²v_th²/(2ω_c²))
    """
    lambda_val = (k_perp * rho_L)**2
    omega = n_harmonic * omega_c * (1 + lambda_val / 2)
    return omega

# Parameters
B = 2.0  # T
T_e = 5e3  # eV (5 keV)
n = 5e19  # m^-3

omega_ce = e * B / m_e
v_th_e = np.sqrt(2 * k_B * T_e * e / m_e)
rho_L = v_th_e / omega_ce

print(f"Electron cyclotron frequency: f_ce = {omega_ce / (2*np.pi) / 1e9:.2f} GHz")
print(f"Larmor radius: ρ_L = {rho_L * 1e3:.2f} mm")

# Wavenumber
k_perp = np.linspace(0, 100, 500) / rho_L

fig, ax = plt.subplots(figsize=(10, 7))

harmonics = [1, 2, 3, 4, 5]
colors = plt.cm.viridis(np.linspace(0, 1, len(harmonics)))

for i, n_harm in enumerate(harmonics):
    omega = bernstein_mode_approximate(n_harm, k_perp, rho_L, omega_ce)
    ax.plot(k_perp * rho_L, omega / omega_ce, color=colors[i],
            linewidth=2, label=f'n = {n_harm}')

    # Mark the harmonic
    ax.axhline(n_harm, color=colors[i], linestyle='--', alpha=0.3)

ax.set_xlabel('$k_\\perp \\rho_L$', fontsize=13)
ax.set_ylabel('$\\omega / \\omega_{ce}$', fontsize=13)
ax.set_title('Electron Bernstein Mode Dispersion', fontsize=14)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 10])
ax.set_ylim([0, 6])

plt.tight_layout()
plt.savefig('bernstein_modes.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Summary

Electrostatic waves are fundamental modes in plasmas, characterized by:

**General framework**:
- Dispersion relation: $\epsilon(\mathbf{k}, \omega) = 0$
- Dielectric function from susceptibilities: $\epsilon = 1 + \sum_s \chi_s$
- Cold plasma: algebraic dispersion
- Warm plasma: kinetic effects via plasma dispersion function $Z(\zeta)$

**Langmuir waves**:
- Cold plasma: $\omega = \omega_{pe}$ (no propagation)
- Warm plasma (Bohm-Gross): $\omega^2 = \omega_{pe}^2 + 3k^2v_{th}^2$
- Group velocity: $v_g = 3kv_{th}^2/\omega$
- Landau damping when $\omega/k \sim v_{th}$

**Ion acoustic waves**:
- Dispersion: $\omega = kc_s$ where $c_s = \sqrt{k_B T_e/m_i}$
- Electron pressure restoring force, ion inertia
- Weak damping requires $T_e \gg T_i$
- Important for turbulence and transport

**Magnetized plasma resonances**:
- Upper hybrid: $\omega_{UH}^2 = \omega_{pe}^2 + \omega_{ce}^2$
- Lower hybrid: $\omega_{LH} \approx \sqrt{\omega_{ci}\omega_{ce}}$
- Used for plasma heating and current drive

**Bernstein modes**:
- Electrostatic modes near cyclotron harmonics: $\omega \approx n\omega_c$
- Exist where EM waves are cutoff
- Mode conversion allows heating in overdense plasmas

Applications include:
- Plasma diagnostics (Thomson scattering, reflectometry)
- Heating and current drive (ECRH, LHCD)
- Understanding instabilities and turbulence
- Astrophysical phenomena (solar radio bursts, pulsar emissions)

## Practice Problems

### Problem 1: Langmuir Wave Properties
A plasma has $n = 5 \times 10^{18}$ m$^{-3}$ and $T_e = 5$ eV.

(a) Calculate the plasma frequency $f_{pe}$ and Debye length $\lambda_D$.

(b) For a Langmuir wave with $k = 0.1/\lambda_D$, calculate the frequency using the Bohm-Gross relation.

(c) Compute the phase velocity $v_\phi$ and group velocity $v_g$. How do they compare to $v_{th,e}$?

(d) At what wavenumber does the thermal correction to $\omega$ become 10% of $\omega_{pe}$?

### Problem 2: Ion Acoustic Wave Damping
An argon plasma ($A = 40$) has $n = 10^{19}$ m$^{-3}$, $T_e = 20$ eV, $T_i = 2$ eV.

(a) Calculate the ion acoustic speed $c_s$.

(b) For $k\lambda_D = 0.3$, find the real frequency $\omega$ and the Landau damping rate $\gamma$.

(c) Calculate the damping length $L_d = v_g/|\gamma|$ where $v_g = d\omega/dk$.

(d) If the ion temperature increases to $T_i = 10$ eV, how does the damping change? Is the wave still observable?

### Problem 3: Hybrid Resonances
A tokamak has $B = 3$ T and a density that varies from $n = 10^{20}$ m$^{-3}$ (core) to $n = 10^{18}$ m$^{-3}$ (edge).

(a) Calculate $f_{UH}$ at the core and edge.

(b) A heating system operates at 110 GHz. Where is the upper hybrid resonance layer located?

(c) Calculate $f_{LH}$ at the core and edge (assume deuterium).

(d) If a lower hybrid system operates at 5 GHz, sketch where the resonance layer is located as a function of radius.

### Problem 4: Bernstein Wave Heating
Consider a dense plasma where $\omega_{pe} > \omega_{ce}$, making the plasma overdense for electron cyclotron waves.

(a) For $B = 1.5$ T, $T_e = 3$ keV, find the maximum density for which $\omega_{pe} < \omega_{ce}$.

(b) If $n = 5 \times 10^{20}$ m$^{-3}$ (overdense), calculate $\omega_{UH}/\omega_{ce}$.

(c) An O-mode wave at $\omega = \omega_{ce}$ cannot penetrate. Explain how mode conversion to a Bernstein wave at the upper hybrid layer can allow heating.

(d) Estimate the perpendicular wavenumber $k_\perp \rho_L$ of the Bernstein wave needed for efficient electron heating at the first harmonic.

### Problem 5: Dispersion Relation Analysis
The electrostatic dispersion relation for an electron-ion plasma is:

$$1 - \frac{\omega_{pe}^2}{k^2 v_{th,e}^2}[1 + \zeta_e Z(\zeta_e)] - \frac{\omega_{pi}^2}{k^2 v_{th,i}^2}[1 + \zeta_i Z(\zeta_i)] = 0$$

where $\zeta_s = \omega/(k v_{th,s})$.

(a) In the limit $\omega \gg k v_{th,e}$, expand to show this gives the Bohm-Gross relation.

(b) In the limit $\omega \ll k v_{th,e}$ but $\omega \sim k v_{th,i}$, show this gives ion acoustic waves with electron Boltzmann response.

(c) For $m_i/m_e = 1836$, $T_e = T_i = 1$ keV, numerically solve for $\omega(k)$ over the range $0.01 < k\lambda_D < 10$. Identify the Langmuir and ion acoustic branches.

(d) At what value of $k\lambda_D$ do the two branches have comparable frequencies? What physics occurs in this regime?

---

**Previous**: [9. Collisional Kinetics](./09_Collisional_Kinetics.md)
**Next**: [11. Electromagnetic Waves](./11_Electromagnetic_Waves.md)
