# 2. Coulomb Collisions

## Learning Objectives

- Understand the physics of Coulomb scattering between charged particles and derive the Rutherford cross-section
- Calculate collision frequencies for electron-electron, ion-ion, and electron-ion interactions
- Derive the Coulomb logarithm and understand its role in determining collision rates
- Compute Spitzer resistivity and understand its temperature dependence
- Distinguish between collisional and collisionless plasma regimes using the Knudsen number
- Apply Python tools to analyze collision dynamics and transport properties

## 1. Coulomb Scattering

### 1.1 The Two-Body Problem

Consider two charged particles with charges $q_1$ and $q_2$, masses $m_1$ and $m_2$, interacting via the Coulomb force. In the center-of-mass frame, this reduces to a single-particle problem with reduced mass:

$$\mu = \frac{m_1 m_2}{m_1 + m_2}$$

The particle moves in the Coulomb potential:

$$V(r) = \frac{1}{4\pi\epsilon_0} \frac{q_1 q_2}{r}$$

### 1.2 Scattering Geometry

```
Scattering Geometry (Lab Frame):

              b (impact parameter)
              ↓
    ●→→→→→→→→→●→→→→→→→
   particle 1  ↑  scattered particle
            target
         (particle 2)

              χ = scattering angle

Classical orbit:
- Hyperbolic trajectory for repulsive force
- Deflection angle χ depends on impact parameter b
- Small b → large deflection
- Large b → small deflection
```

For a **repulsive** Coulomb interaction ($q_1 q_2 > 0$), the scattering angle $\chi$ is related to the impact parameter $b$ by:

$$\tan\left(\frac{\chi}{2}\right) = \frac{b_{90}}{b}$$

where $b_{90}$ is the **impact parameter for 90° scattering**:

$$b_{90} = \frac{1}{4\pi\epsilon_0} \frac{q_1 q_2}{\mu v_0^2} = \frac{q_1 q_2}{4\pi\epsilon_0 E_{cm}}$$

Here, $v_0$ is the initial relative velocity and $E_{cm} = \frac{1}{2}\mu v_0^2$ is the center-of-mass kinetic energy.

**Physical interpretation:** $b_{90}$ is the impact parameter at which the Coulomb potential energy equals the kinetic energy:

$$\frac{1}{4\pi\epsilon_0}\frac{q_1 q_2}{b_{90}} = \frac{1}{2}\mu v_0^2$$

### 1.3 Rutherford Cross-Section

The **differential cross-section** describes the probability of scattering into a solid angle $d\Omega$ at angle $\chi$:

$$\frac{d\sigma}{d\Omega} = \left(\frac{b_{90}}{2}\right)^2 \frac{1}{\sin^4(\chi/2)}$$

This is the **Rutherford scattering formula**, one of the most important results in classical scattering theory.

**Key features:**
1. Strong forward-bias: $d\sigma/d\Omega \to \infty$ as $\chi \to 0$ (small-angle scattering dominates)
2. Symmetry: depends only on $|\chi|$
3. Divergence: total cross-section $\sigma_{total} = \int (d\sigma/d\Omega) d\Omega$ diverges!

The divergence arises because:
- Small-angle scattering ($\chi \ll 1$) dominates
- Long-range Coulomb force allows arbitrarily large impact parameters
- Many weak deflections are more important than rare large deflections

### 1.4 Momentum Transfer Cross-Section

For transport properties, we need the **momentum transfer cross-section**, weighted by $(1 - \cos\chi)$:

$$\sigma_m = \int (1 - \cos\chi) \frac{d\sigma}{d\Omega} d\Omega$$

This integral also diverges, but the divergence is milder ($\ln b_{max}$). We'll address this shortly with Debye screening.

## 2. The Coulomb Logarithm

### 2.1 Cutoffs for the Coulomb Interaction

The Rutherford formula assumes an infinite-range unscreened Coulomb potential. In a plasma, two physical effects provide cutoffs:

**1. Maximum impact parameter ($b_{max}$):** Debye shielding

At distances $b > \lambda_D$, the Coulomb potential is exponentially screened. Thus:

$$b_{max} \sim \lambda_D = \sqrt{\frac{\epsilon_0 k_B T}{n e^2}}$$

**2. Minimum impact parameter ($b_{min}$):** Classical distance of closest approach or quantum uncertainty

The minimum impact parameter is the larger of:

(a) **Classical closest approach** $b_{90}$ (when potential energy equals kinetic energy)

(b) **Quantum de Broglie wavelength** $\lambda_{dB} = \hbar/(mv_{thermal})$ (when wave effects matter)

For most plasmas, the classical limit dominates:

$$b_{min} \sim b_{90} = \frac{q_1 q_2}{4\pi\epsilon_0 \mu v_{th}^2}$$

where $v_{th} = \sqrt{k_B T/m}$ is the thermal velocity.

### 2.2 Definition of the Coulomb Logarithm

The **Coulomb logarithm** is defined as:

$$\ln\Lambda = \ln\left(\frac{b_{max}}{b_{min}}\right)$$

**For electron-ion collisions:**

$$\ln\Lambda_{ei} \approx \ln\left(\frac{12\pi n_e \lambda_D^3}{Z}\right)$$

Using $\lambda_D = \sqrt{\epsilon_0 k_B T_e/(n_e e^2)}$:

$$\ln\Lambda_{ei} \approx \begin{cases}
23 - \ln\left(\sqrt{n_e/10^6} \, Z \, T_e^{-3/2}\right) & T_e < 10 Z^2 \text{ eV} \\
24 - \ln\left(\sqrt{n_e/10^6} \, T_e^{-1}\right) & T_e > 10 Z^2 \text{ eV}
\end{cases}$$

where $n_e$ is in cm$^{-3}$ and $T_e$ is in eV.

**Typical values:**
- Laboratory plasmas: $\ln\Lambda \approx 10 - 15$
- Fusion plasmas: $\ln\Lambda \approx 15 - 20$
- Astrophysical plasmas: $\ln\Lambda \approx 20 - 30$

```
Physical Meaning of ln Λ:

ln Λ ≈ ln(number of particles in Debye sphere) ≈ ln(N_D)

    b_max ~ λ_D
       ↓
  ●───●───●───●     Debye sphere
  ●───●───●───●     contains ~N_D particles
  ●───●───●───●
       ↑
    b_min ~ b_90

ln Λ counts the "effective range" of Coulomb interactions
in units of logarithms (weak dependence on plasma parameters)
```

### 2.3 Weak Logarithmic Dependence

The remarkable feature of $\ln\Lambda$ is its **weak dependence** on plasma parameters. Even though $n$ and $T$ vary by many orders of magnitude across different plasmas, $\ln\Lambda$ only varies by a factor of 2-3.

| Plasma | $n$ [m$^{-3}$] | $T$ [eV] | $\ln\Lambda$ |
|--------|----------------|----------|--------------|
| Tokamak core | $10^{20}$ | 10,000 | 17 |
| Tokamak edge | $10^{19}$ | 100 | 15 |
| Solar corona | $10^{14}$ | 100 | 19 |
| Ionosphere |  $10^{12}$ | 0.1 | 12 |
| Glow discharge | $10^{16}$ | 2 | 10 |

This weak dependence allows us to treat $\ln\Lambda \approx 15$ as a constant in many estimates.

## 3. Collision Frequencies

### 3.1 Momentum Transfer Collision Frequency

The **collision frequency** $\nu$ is the rate at which a particle undergoes momentum-changing collisions. It's defined as:

$$\nu = n \sigma_m v_{th}$$

where $\sigma_m$ is the momentum transfer cross-section.

For Coulomb collisions, integrating the Rutherford formula with cutoffs:

$$\sigma_m \sim \pi b_{90}^2 \ln\Lambda$$

### 3.2 Electron-Ion Collision Frequency

The **electron-ion collision frequency** is:

$$\nu_{ei} = \frac{n_i Z^2 e^4 \ln\Lambda}{4\pi\epsilon_0^2 m_e^2 v_e^3}$$

Using the electron thermal velocity $v_e = \sqrt{k_B T_e/m_e}$:

$$\nu_{ei} = \frac{n_i Z^2 e^4 \ln\Lambda}{4\pi\epsilon_0^2 m_e^{1/2} (k_B T_e)^{3/2}}$$

**Numerical formula:**

$$\nu_{ei} \approx 2.91 \times 10^{-6} \, \frac{n_e[\text{m}^{-3}] \, Z \, \ln\Lambda}{T_e[\text{eV}]^{3/2}} \quad [\text{s}^{-1}]$$

**Key scaling:** $\nu_{ei} \propto n T^{-3/2}$

- Collisions increase with density (more targets)
- Collisions decrease rapidly with temperature (faster particles spend less time in collision region)

### 3.3 Electron-Electron Collision Frequency

For **like-particle collisions** (electron-electron or ion-ion), the kinematics differ. The collision frequency is:

$$\nu_{ee} \approx \nu_{ei}$$

(same order of magnitude, with a numerical factor of order unity).

More precisely:

$$\nu_{ee} = \frac{n_e e^4 \ln\Lambda}{8\pi\epsilon_0^2 m_e^{1/2} (k_B T_e)^{3/2}} \approx 1.45 \times 10^{-6} \, \frac{n_e[\text{m}^{-3}] \, \ln\Lambda}{T_e[\text{eV}]^{3/2}} \quad [\text{s}^{-1}]$$

### 3.4 Ion-Ion Collision Frequency

Similarly, for ions:

$$\nu_{ii} = \frac{n_i Z^4 e^4 \ln\Lambda}{8\pi\epsilon_0^2 m_i^{1/2} (k_B T_i)^{3/2}}$$

**Comparison with electron collision frequency:**

$$\frac{\nu_{ii}}{\nu_{ei}} \sim \sqrt{\frac{m_e}{m_i}} \left(\frac{T_e}{T_i}\right)^{3/2}$$

For $T_e \sim T_i$ and hydrogen plasma:

$$\nu_{ii} \sim \frac{\nu_{ei}}{43}$$

Ions collide much less frequently than electrons (slower thermal velocities).

### 3.5 Ordering of Collision Frequencies

For typical plasmas with $T_e \sim T_i$ and $m_i \gg m_e$:

$$\nu_{ee} \sim \nu_{ei} \gg \nu_{ie} \gg \nu_{ii}$$

where $\nu_{ie}$ is the ion-electron collision frequency.

**Physical interpretation:**
- Electrons collide frequently with both electrons and ions
- Ions collide primarily with ions; electrons are too fast and light to deflect ions significantly
- Momentum and energy exchange between species occurs on the slower ion timescale

## 4. Spitzer Resistivity

### 4.1 Derivation from Collision Frequency

Electrical resistivity arises from momentum transfer between electrons (carrying current) and ions (stationary).

In a simple Drude model, the conductivity is:

$$\sigma_{\parallel} = \frac{n_e e^2}{m_e \nu_{ei}}$$

The **Spitzer resistivity** is:

$$\eta = \frac{1}{\sigma_{\parallel}} = \frac{m_e \nu_{ei}}{n_e e^2}$$

Substituting $\nu_{ei}$:

$$\eta = \frac{Z \, m_e e^2 \ln\Lambda}{4\pi\epsilon_0^2 (k_B T_e)^{3/2}}$$

**Numerical formula:**

$$\eta \approx 5.2 \times 10^{-5} \, \frac{Z \, \ln\Lambda}{T_e[\text{eV}]^{3/2}} \quad [\Omega \cdot \text{m}]$$

### 4.2 Temperature Dependence

The key result is the **strong temperature dependence**:

$$\eta \propto T_e^{-3/2}$$

**Implications:**
- Hot plasmas are excellent conductors (low resistivity)
- Resistivity decreases rapidly with heating
- For fusion plasmas ($T_e \sim 10$ keV), $\eta \sim 10^{-8}$ Ω·m (comparable to room-temperature copper!)

```
Resistivity vs Temperature:

η [Ω⋅m]
 ↑
10⁻⁴│         .
     │       .
10⁻⁵│      .
     │    .
10⁻⁶│   .
     │  .
10⁻⁷│ .
     │.
10⁻⁸│
     └─────────────────→ T_e [eV]
     10   100   1000  10000

Spitzer: η ∝ T^(-3/2)
```

### 4.3 Collisional Heating

Resistivity dissipates electrical energy as heat. The power density is:

$$P_{Ohmic} = \eta J^2 = \eta \frac{I^2}{A^2}$$

where $J$ is current density and $I$ is total current.

In tokamaks, **Ohmic heating** dominates at low temperatures but becomes ineffective at high $T$ due to the $T^{-3/2}$ scaling.

### 4.4 Comparison with Classical Resistivity

Compare Spitzer resistivity to classical metals:

| Material | $\eta$ [Ω·m] at 300 K |
|----------|------------------------|
| Copper | $1.7 \times 10^{-8}$ |
| Aluminum | $2.7 \times 10^{-8}$ |
| Plasma ($T_e=10$ keV) | $\sim 10^{-8}$ |

Hot plasmas are as conductive as metals! However, the physical mechanism is different:
- Metals: electron-phonon scattering
- Plasmas: electron-ion Coulomb collisions

## 5. Mean Free Path and Collisionality

### 5.1 Mean Free Path

The **mean free path** $\lambda_{mfp}$ is the average distance a particle travels between collisions:

$$\lambda_{mfp} = \frac{v_{th}}{\nu}$$

**For electrons:**

$$\lambda_{mfp,e} = \frac{v_{te}}{\nu_{ei}} = \frac{\sqrt{k_B T_e/m_e}}{\nu_{ei}}$$

**Numerical estimate:**

$$\lambda_{mfp,e} \approx 3.44 \times 10^{11} \, \frac{T_e[\text{eV}]^2}{n_e[\text{m}^{-3}] \, Z \, \ln\Lambda} \quad [\text{m}]$$

### 5.2 Knudsen Number

The **Knudsen number** compares the mean free path to the system size $L$:

$$Kn = \frac{\lambda_{mfp}}{L}$$

**Collisionality regimes:**

- **Collisional (fluid-like):** $Kn \ll 1$
  - Many collisions occur within system size
  - Local thermodynamic equilibrium (LTE)
  - Fluid (MHD) description valid

- **Collisionless (kinetic):** $Kn \gg 1$
  - Few or no collisions within system size
  - Distribution function non-Maxwellian
  - Kinetic (Vlasov) description required

- **Transitional:** $Kn \sim 1$
  - Neither limit applicable
  - Most difficult regime to model

### 5.3 Examples

**Tokamak core:**
- $n_e = 10^{20}$ m$^{-3}$, $T_e = 10$ keV, $L = 1$ m, $\ln\Lambda = 17$
- $\nu_{ei} \approx 1.7 \times 10^4$ s$^{-1}$
- $v_{te} \approx 4.2 \times 10^7$ m/s
- $\lambda_{mfp} \approx 2500$ m $\gg L$
- $Kn \approx 2500$ → **collisionless**

**Magnetospheric plasma:**
- $n \sim 10^6$ m$^{-3}$, $T \sim 1$ keV, $L \sim 10^7$ m
- $\lambda_{mfp} \sim 10^{15}$ m $\gg L$
- Extremely collisionless

**Glow discharge:**
- $n \sim 10^{16}$ m$^{-3}$, $T \sim 2$ eV, $L \sim 0.1$ m
- $\lambda_{mfp} \sim 1$ m $\gtrsim L$
- Transitional regime

## 6. Energy Equipartition

### 6.1 Energy Exchange Between Species

When $T_e \ne T_i$, energy is transferred between species via collisions. The **energy exchange frequency** is:

$$\nu_{E,ei} = \frac{2m_e}{m_i} \nu_{ei}$$

The factor $2m_e/m_i \ll 1$ reflects inefficient energy transfer in collisions between particles of very different masses.

### 6.2 Equipartition Time

The **equipartition time** is the time for temperatures to equilibrate:

$$\tau_{eq} = \frac{1}{\nu_{E,ei}} = \frac{m_i}{2m_e \nu_{ei}}$$

**Numerical estimate:**

$$\tau_{eq} \approx 1.09 \times 10^{13} \, \frac{A \, T_e[\text{eV}]^{3/2}}{n_e[\text{m}^{-3}] \, Z \, \ln\Lambda} \quad [\text{s}]$$

where $A$ is the ion mass number.

For hydrogen plasma ($A=1$) in a tokamak:
- $n_e = 10^{20}$ m$^{-3}$, $T_e = 10$ keV
- $\tau_{eq} \approx 1$ s

This is **long** compared to energy confinement times ($\sim 0.1$ s), so $T_e$ and $T_i$ can differ significantly.

### 6.3 Electron vs Ion Heating

In fusion experiments:
- **Electron heating** (e.g., ECRH, Ohmic): heats $T_e$ directly
- **Ion heating** (e.g., neutral beam injection, ICRH): heats $T_i$ directly

Due to slow equipartition ($\tau_{eq} \gg \tau_E$), separate control of $T_e$ and $T_i$ is possible. Typical tokamak profiles show $T_e \gtrsim T_i$ in the core.

## 7. Computational Examples

### 7.1 Collision Frequency Calculator

```python
import numpy as np
import matplotlib.pyplot as plt

# Constants
e = 1.602176634e-19
m_e = 9.1093837015e-31
m_p = 1.672621898e-27
epsilon_0 = 8.8541878128e-12
k_B = 1.380649e-23
eV_to_K = 11604.518

def coulomb_logarithm(n_e, T_e, Z=1):
    """
    Calculate Coulomb logarithm.

    Parameters:
    -----------
    n_e : float
        Electron density [m^-3]
    T_e : float
        Electron temperature [eV]
    Z : int
        Ion charge state

    Returns:
    --------
    ln_Lambda : float
        Coulomb logarithm
    """
    n_e_cgs = n_e * 1e-6  # Convert to cm^-3

    if T_e < 10 * Z**2:
        ln_Lambda = 23 - np.log(np.sqrt(n_e_cgs) * Z * T_e**(-1.5))
    else:
        ln_Lambda = 24 - np.log(np.sqrt(n_e_cgs) * T_e**(-1))

    return ln_Lambda

def nu_ei(n_e, T_e, Z=1, ln_Lambda=None):
    """
    Electron-ion collision frequency.

    Returns: frequency [s^-1]
    """
    if ln_Lambda is None:
        ln_Lambda = coulomb_logarithm(n_e, T_e, Z)

    return 2.91e-6 * n_e * Z * ln_Lambda / T_e**1.5

def nu_ee(n_e, T_e, ln_Lambda=None):
    """
    Electron-electron collision frequency.

    Returns: frequency [s^-1]
    """
    if ln_Lambda is None:
        ln_Lambda = coulomb_logarithm(n_e, T_e)

    return 1.45e-6 * n_e * ln_Lambda / T_e**1.5

def nu_ii(n_i, T_i, Z=1, A=1, ln_Lambda=None):
    """
    Ion-ion collision frequency.

    Returns: frequency [s^-1]
    """
    if ln_Lambda is None:
        ln_Lambda = coulomb_logarithm(n_i, T_i, Z)

    # Conversion factor
    factor = 1.45e-6 * np.sqrt(m_e / (A * m_p))
    return factor * n_i * Z**4 * ln_Lambda / T_i**1.5

def spitzer_resistivity(T_e, Z=1, ln_Lambda=None):
    """
    Spitzer resistivity.

    Parameters:
    -----------
    T_e : float
        Electron temperature [eV]
    Z : int
        Ion charge
    ln_Lambda : float, optional
        Coulomb logarithm

    Returns:
    --------
    eta : float
        Resistivity [Ohm*m]
    """
    if ln_Lambda is None:
        ln_Lambda = 15  # Typical value

    return 5.2e-5 * Z * ln_Lambda / T_e**1.5

def mean_free_path(n_e, T_e, Z=1, ln_Lambda=None):
    """
    Electron mean free path.

    Returns: lambda_mfp [m]
    """
    if ln_Lambda is None:
        ln_Lambda = coulomb_logarithm(n_e, T_e, Z)

    return 3.44e11 * T_e**2 / (n_e * Z * ln_Lambda)

# Demonstration
if __name__ == "__main__":
    print("="*70)
    print("COLLISION FREQUENCY ANALYSIS")
    print("="*70)

    # Example: Tokamak parameters
    n_e = 1e20  # m^-3
    T_e = 10000  # eV
    T_i = 8000   # eV
    Z = 1
    A = 2  # Deuterium

    ln_Lambda = coulomb_logarithm(n_e, T_e, Z)

    print(f"\nPlasma Parameters:")
    print(f"  n_e = {n_e:.2e} m^-3")
    print(f"  T_e = {T_e:.0f} eV")
    print(f"  T_i = {T_i:.0f} eV")
    print(f"  Z   = {Z}, A = {A}")
    print(f"  ln Λ = {ln_Lambda:.2f}")
    print("-"*70)

    nu_ei_val = nu_ei(n_e, T_e, Z, ln_Lambda)
    nu_ee_val = nu_ee(n_e, T_e, ln_Lambda)
    nu_ii_val = nu_ii(n_e, T_i, Z, A, ln_Lambda)

    print(f"\nCollision Frequencies:")
    print(f"  ν_ei = {nu_ei_val:.3e} s^-1  (period: {1/nu_ei_val:.3e} s)")
    print(f"  ν_ee = {nu_ee_val:.3e} s^-1  (period: {1/nu_ee_val:.3e} s)")
    print(f"  ν_ii = {nu_ii_val:.3e} s^-1  (period: {1/nu_ii_val:.3e} s)")
    print(f"  Ratio ν_ei/ν_ii = {nu_ei_val/nu_ii_val:.1f}")
    print("-"*70)

    eta = spitzer_resistivity(T_e, Z, ln_Lambda)
    print(f"\nSpitzer Resistivity:")
    print(f"  η = {eta:.3e} Ω·m")
    print(f"  (Copper at 300 K: 1.7e-8 Ω·m)")
    print("-"*70)

    lambda_mfp = mean_free_path(n_e, T_e, Z, ln_Lambda)
    v_te = np.sqrt(k_B * T_e * eV_to_K / m_e)

    print(f"\nMean Free Path:")
    print(f"  λ_mfp = {lambda_mfp:.2e} m")
    print(f"  v_te  = {v_te:.3e} m/s")
    print(f"  For system size L = 1 m:")
    print(f"    Knudsen number Kn = {lambda_mfp/1:.0f}")
    print(f"    Regime: {'Collisionless' if lambda_mfp > 1 else 'Collisional'}")
    print("-"*70)

    # Energy equipartition time
    tau_eq = (A * m_p) / (2 * m_e * nu_ei_val)
    print(f"\nEnergy Equipartition:")
    print(f"  τ_eq = {tau_eq:.3e} s = {tau_eq*1000:.1f} ms")
    print("="*70)
```

### 7.2 Resistivity vs Temperature

```python
def plot_resistivity_vs_temperature():
    """Plot Spitzer resistivity as a function of temperature."""

    T_vals = np.logspace(0, 4, 100)  # 1 eV to 10 keV
    Z_vals = [1, 2, 6]  # H, He, C

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Linear-log plot
    for Z in Z_vals:
        eta_vals = [spitzer_resistivity(T, Z, ln_Lambda=15) for T in T_vals]
        ax1.loglog(T_vals, eta_vals, linewidth=2, label=f'Z={Z}')

    # Add reference: T^(-3/2) scaling
    eta_ref = spitzer_resistivity(10, Z=1) * (T_vals/10)**(-1.5)
    ax1.loglog(T_vals, eta_ref, 'k--', alpha=0.5, linewidth=1.5,
               label=r'$\propto T^{-3/2}$')

    # Copper resistivity (room temp)
    ax1.axhline(y=1.7e-8, color='brown', linestyle=':', linewidth=2,
                label='Copper (300 K)')

    ax1.set_xlabel(r'Temperature $T_e$ [eV]', fontsize=12)
    ax1.set_ylabel(r'Resistivity $\eta$ [Ω·m]', fontsize=12)
    ax1.set_title('Spitzer Resistivity vs Temperature', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 1e4)
    ax1.set_ylim(1e-9, 1e-4)

    # Conductivity plot
    for Z in Z_vals:
        sigma_vals = [1/spitzer_resistivity(T, Z, ln_Lambda=15) for T in T_vals]
        ax2.loglog(T_vals, sigma_vals, linewidth=2, label=f'Z={Z}')

    ax2.set_xlabel(r'Temperature $T_e$ [eV]', fontsize=12)
    ax2.set_ylabel(r'Conductivity $\sigma$ [S/m]', fontsize=12)
    ax2.set_title('Electrical Conductivity vs Temperature', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, 1e4)

    plt.tight_layout()
    plt.savefig('spitzer_resistivity.png', dpi=150)
    plt.show()

plot_resistivity_vs_temperature()
```

### 7.3 Collision Frequency Scaling

```python
def plot_collision_frequency_scaling():
    """Visualize scaling of collision frequencies with n and T."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Scan 1: Vary density at fixed T
    n_vals = np.logspace(14, 22, 100)
    T_fixed = 100  # eV

    nu_ei_vals = [nu_ei(n, T_fixed) for n in n_vals]
    nu_ee_vals = [nu_ee(n, T_fixed) for n in n_vals]

    ax = axes[0, 0]
    ax.loglog(n_vals, nu_ei_vals, 'b-', linewidth=2, label=r'$\nu_{ei}$')
    ax.loglog(n_vals, nu_ee_vals, 'r--', linewidth=2, label=r'$\nu_{ee}$')
    ax.set_xlabel(r'Density $n_e$ [m$^{-3}$]', fontsize=11)
    ax.set_ylabel(r'Collision Frequency [s$^{-1}$]', fontsize=11)
    ax.set_title(f'Collision Frequency vs Density (T={T_fixed} eV)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Scan 2: Vary temperature at fixed n
    T_vals = np.logspace(0, 4, 100)
    n_fixed = 1e19  # m^-3

    nu_ei_vals = [nu_ei(n_fixed, T) for T in T_vals]
    nu_ee_vals = [nu_ee(n_fixed, T) for T in T_vals]

    ax = axes[0, 1]
    ax.loglog(T_vals, nu_ei_vals, 'b-', linewidth=2, label=r'$\nu_{ei}$')
    ax.loglog(T_vals, nu_ee_vals, 'r--', linewidth=2, label=r'$\nu_{ee}$')

    # Reference line: T^(-3/2)
    nu_ref = nu_ei(n_fixed, 100) * (T_vals/100)**(-1.5)
    ax.loglog(T_vals, nu_ref, 'k:', linewidth=1.5, alpha=0.7,
              label=r'$\propto T^{-3/2}$')

    ax.set_xlabel(r'Temperature $T_e$ [eV]', fontsize=11)
    ax.set_ylabel(r'Collision Frequency [s$^{-1}$]', fontsize=11)
    ax.set_title(f'Collision Frequency vs Temperature (n={n_fixed:.0e} m⁻³)',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Scan 3: Mean free path vs temperature
    lambda_vals = [mean_free_path(n_fixed, T) for T in T_vals]

    ax = axes[1, 0]
    ax.loglog(T_vals, lambda_vals, 'g-', linewidth=2)
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5,
               label='L = 1 m (device size)')
    ax.set_xlabel(r'Temperature $T_e$ [eV]', fontsize=11)
    ax.set_ylabel(r'Mean Free Path $\lambda_{mfp}$ [m]', fontsize=11)
    ax.set_title(f'Mean Free Path vs Temperature (n={n_fixed:.0e} m⁻³)',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Scan 4: Coulomb logarithm landscape
    n_range = np.logspace(14, 22, 50)
    T_range = np.logspace(0, 4, 50)
    N, T = np.meshgrid(n_range, T_range)

    ln_Lambda_map = np.zeros_like(N)
    for i in range(len(T_range)):
        for j in range(len(n_range)):
            ln_Lambda_map[i, j] = coulomb_logarithm(N[i, j], T[i, j])

    ax = axes[1, 1]
    contour = ax.contourf(N, T, ln_Lambda_map, levels=20, cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax, label=r'$\ln\Lambda$')

    cs = ax.contour(N, T, ln_Lambda_map, levels=[10, 15, 20, 25],
                    colors='white', linewidths=1.5, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=9)

    ax.set_xlabel(r'Density $n_e$ [m$^{-3}$]', fontsize=11)
    ax.set_ylabel(r'Temperature $T_e$ [eV]', fontsize=11)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Coulomb Logarithm Landscape', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('collision_frequency_scaling.png', dpi=150)
    plt.show()

plot_collision_frequency_scaling()
```

### 7.4 Collisionality Map

```python
def plot_collisionality_map():
    """
    Create a map showing collisional vs collisionless regimes
    for various system sizes.
    """
    n_range = np.logspace(14, 24, 100)
    T_range = np.logspace(0, 4, 100)
    N, T = np.meshgrid(n_range, T_range)

    # Mean free path
    lambda_mfp_map = 3.44e11 * T**2 / (N * 15)  # ln Λ ≈ 15

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Mean free path contours
    ax = axes[0]
    levels = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1e4]
    contour = ax.contourf(N, T, lambda_mfp_map, levels=levels,
                          cmap='RdYlGn', norm=plt.matplotlib.colors.LogNorm())
    cbar = plt.colorbar(contour, ax=ax, label=r'Mean Free Path $\lambda_{mfp}$ [m]')

    cs = ax.contour(N, T, lambda_mfp_map, levels=levels,
                    colors='black', linewidths=1, alpha=0.4)
    ax.clabel(cs, inline=True, fontsize=9, fmt='%g m')

    # Mark typical system sizes
    system_sizes = {
        'Tokamak': 1,
        'Lab device': 0.1,
        'Magnetosphere': 1e7,
    }

    for name, L in system_sizes.items():
        # Line where lambda_mfp = L (Kn = 1)
        T_Kn1 = np.sqrt(N * 15 * L / 3.44e11)
        valid = (T_Kn1 >= T_range.min()) & (T_Kn1 <= T_range.max())
        ax.plot(N[valid], T_Kn1[valid], 'r--', linewidth=2.5, alpha=0.8)

        # Label
        idx = len(N) // 2
        if valid[idx]:
            ax.annotate(f'Kn=1 (L={L}m)', (N[idx], T_Kn1[idx]),
                       fontsize=10, color='red', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(r'Density $n_e$ [m$^{-3}$]', fontsize=12)
    ax.set_ylabel(r'Temperature $T_e$ [eV]', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Mean Free Path Landscape', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 2: Knudsen number for L=1m
    L_ref = 1.0  # m
    Kn_map = lambda_mfp_map / L_ref

    ax = axes[1]
    levels_Kn = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    contour = ax.contourf(N, T, Kn_map, levels=levels_Kn,
                          cmap='coolwarm', norm=plt.matplotlib.colors.LogNorm())
    cbar = plt.colorbar(contour, ax=ax, label=f'Knudsen Number (L={L_ref}m)')

    # Mark Kn = 1 (boundary)
    cs_boundary = ax.contour(N, T, Kn_map, levels=[1],
                             colors='black', linewidths=3)
    ax.clabel(cs_boundary, inline=True, fontsize=12, fmt='Kn=1')

    # Shade regions
    ax.text(1e15, 1e3, 'Collisionless\n(Kn >> 1)',
           fontsize=14, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(1e23, 10, 'Collisional\n(Kn << 1)',
           fontsize=14, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    ax.set_xlabel(r'Density $n_e$ [m$^{-3}$]', fontsize=12)
    ax.set_ylabel(r'Temperature $T_e$ [eV]', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'Collisionality Regime (L={L_ref} m)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('collisionality_map.png', dpi=150)
    plt.show()

plot_collisionality_map()
```

## Summary

Coulomb collisions arise from the long-range electromagnetic interaction between charged particles and profoundly influence plasma transport and dynamics:

1. **Rutherford scattering** describes binary Coulomb collisions, with a differential cross-section strongly peaked in the forward direction.

2. **The Coulomb logarithm** $\ln\Lambda \approx 10-20$ regularizes divergences by accounting for Debye screening (maximum impact parameter) and quantum/classical effects (minimum impact parameter).

3. **Collision frequencies** scale as $\nu \propto n T^{-3/2}$, with electrons colliding much more frequently than ions due to their lower mass.

4. **Spitzer resistivity** $\eta \propto T_e^{-3/2}$ decreases rapidly with temperature, making hot plasmas excellent conductors.

5. **Collisionality regimes** are characterized by the Knudsen number $Kn = \lambda_{mfp}/L$, with fusion plasmas typically collisionless ($Kn \gg 1$) and requiring kinetic descriptions.

6. **Energy equipartition** between electrons and ions occurs on slow timescales $\tau_{eq} \sim (m_i/m_e)\nu_{ei}^{-1}$, allowing separate temperature evolution in auxiliary-heated plasmas.

Understanding collision dynamics is essential for modeling transport, heating, current drive, and the transition between kinetic and fluid descriptions.

## Practice Problems

### Problem 1: Collision Frequency in a Glow Discharge

A neon glow discharge has $n_e = 10^{16}$ m$^{-3}$, $T_e = 2$ eV, $T_i = 0.05$ eV.

(a) Calculate the Coulomb logarithm.

(b) Compute the electron-ion collision frequency $\nu_{ei}$ and the collision period.

(c) Find the mean free path. Compare to a typical discharge tube diameter of 3 cm. Is this a collisional or collisionless plasma?

(d) Calculate the electron-electron collision frequency and compare to $\nu_{ei}$.

### Problem 2: Spitzer Resistivity in a Tokamak

Consider a deuterium tokamak with:
- Core: $n_e = 5 \times 10^{19}$ m$^{-3}$, $T_e = 12$ keV
- Edge: $n_e = 2 \times 10^{18}$ m$^{-3}$, $T_e = 100$ eV

(a) Calculate the Spitzer resistivity at both locations.

(b) If a current density $J = 1$ MA/m$^2$ flows through the core, compute the Ohmic heating power density $P = \eta J^2$.

(c) The edge carries the same total current but through a smaller cross-section with $J_{edge} = 3$ MA/m$^2$. Compare the Ohmic heating power densities. Where is resistive heating more important?

(d) Estimate the voltage drop along a 10 m toroidal path for both core and edge.

### Problem 3: Temperature Equilibration

An electron cyclotron resonance heating (ECRH) system deposits 1 MW into electrons in a deuterium plasma with $n_e = 10^{20}$ m$^{-3}$, $T_e = 5$ keV, $T_i = 3$ keV, and volume $V = 10$ m$^3$.

(a) Calculate the energy equipartition time $\tau_{eq}$.

(b) Estimate the rate of energy transfer from electrons to ions (in watts).

(c) If the energy confinement time is $\tau_E = 0.1$ s, compare $\tau_{eq}$ and $\tau_E$. Will the temperatures equilibrate?

(d) Find the steady-state electron and ion temperatures assuming all input power is lost through transport (neglect radiation and other losses). Use the fact that power balance gives:
   $$P_{ECRH} = P_{ei} + P_{loss,e}$$
   $$P_{ei} = P_{loss,i}$$
   where $P_{ei} \propto (T_e - T_i)/\tau_{eq}$ and $P_{loss} \propto 3nT/\tau_E$.

### Problem 4: Impact Parameter Estimates

For an electron-ion collision in a hydrogen plasma at $T_e = 10$ eV and $n_e = 10^{18}$ m$^{-3}$:

(a) Calculate the Debye length $\lambda_D$.

(b) Compute the 90° scattering impact parameter $b_{90}$ using the thermal velocity.

(c) Find the de Broglie wavelength $\lambda_{dB} = \hbar/(m_e v_{th})$ and compare to $b_{90}$. Which determines $b_{min}$?

(d) Calculate $\ln\Lambda = \ln(b_{max}/b_{min})$ and compare to the standard formula.

### Problem 5: Collisionality Regimes

A magnetized plasma column has $B = 0.5$ T along the axis, with length $L_\parallel = 2$ m and radius $r = 0.1$ m.

(a) For $n_e = 10^{18}$ m$^{-3}$ and $T_e = 50$ eV, calculate the parallel and perpendicular mean free paths. (Hint: $\lambda_{\parallel} = v_{th,\parallel}/\nu$ and $\lambda_\perp \sim r_L$ if $r_L \ll \lambda_{mfp}$.)

(b) Compute the Knudsen numbers $Kn_\parallel = \lambda_{mfp}/L_\parallel$ and $Kn_\perp = \lambda_{mfp}/r$.

(c) Is the plasma collisional or collisionless along field lines? Across field lines?

(d) Repeat for $n_e = 10^{20}$ m$^{-3}$ and $T_e = 1$ keV. How does the collisionality change?

---

**Previous:** [Introduction to Plasma](./01_Introduction_to_Plasma.md) | **Next:** [Plasma Description Hierarchy](./03_Plasma_Description_Hierarchy.md)
