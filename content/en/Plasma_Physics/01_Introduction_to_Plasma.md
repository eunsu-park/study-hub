# 1. Introduction to Plasma

## Learning Objectives

- Understand plasma as the fourth state of matter and identify natural and laboratory examples
- Derive and apply the Debye shielding concept and calculate the Debye length for various plasmas
- Explain the plasma criteria for collective behavior and quasi-neutrality conditions
- Calculate characteristic plasma frequencies (plasma frequency and gyrofrequency) from first principles
- Compute the plasma beta parameter and interpret the relative importance of thermal versus magnetic pressure
- Apply Python tools to compute and compare plasma parameters across different physical systems

## 1. The Fourth State of Matter

### 1.1 What is Plasma?

Plasma is often called the **fourth state of matter**, following solid, liquid, and gas. It consists of a collection of free charged particles—electrons and ions—that exhibit collective behavior due to long-range electromagnetic interactions.

```
State Transitions:

Solid → Liquid → Gas → Plasma
  ↑         ↑        ↑       ↑
 Heat     Heat    Heat   Ionization
            (add energy →)

Key characteristics:
- Solid:  molecules in fixed lattice, short-range forces
- Liquid: molecules mobile, short-range forces
- Gas:    molecules independent, rare collisions
- Plasma: ions + electrons, long-range EM forces, collective behavior
```

The transition from gas to plasma occurs through **ionization**—the process where neutral atoms or molecules lose electrons due to:
- Thermal energy (high temperature)
- Electromagnetic radiation (photoionization)
- Collisions with energetic particles
- Electric fields (avalanche breakdown)

### 1.2 Ionization and Degree of Ionization

The **degree of ionization** $\alpha$ is defined as:

$$\alpha = \frac{n_i}{n_i + n_n}$$

where $n_i$ is the ion density and $n_n$ is the neutral density.

- **Fully ionized plasma**: $\alpha \approx 1$ (nearly all particles are charged)
- **Partially ionized plasma**: $0 < \alpha < 1$ (mixture of charged and neutral particles)
- **Weakly ionized plasma**: $\alpha \ll 1$ (most particles remain neutral, but collective effects dominate)

Even weakly ionized plasmas ($\alpha \sim 10^{-6}$ in some cases) can exhibit plasma behavior if the plasma criteria (discussed below) are satisfied.

### 1.3 Examples of Plasmas

Plasmas are the most common state of matter in the universe, comprising over 99% of visible matter.

**Natural Plasmas:**

| System | Temperature (eV) | Density (m⁻³) | Magnetic Field (T) |
|--------|------------------|---------------|-------------------|
| Interstellar medium | 1 | 10⁶ | 10⁻¹⁰ |
| Solar wind (1 AU) | 10 | 10⁷ | 10⁻⁹ |
| Solar corona | 100 | 10¹⁴ | 10⁻² |
| Solar core | 1000 | 10³² | — |
| Lightning | 2-3 | 10²² | — |
| Ionosphere (F-layer) | 0.1 | 10¹² | 10⁻⁵ |

**Laboratory Plasmas:**

| System | Temperature (eV) | Density (m⁻³) | Magnetic Field (T) |
|--------|------------------|---------------|-------------------|
| Tokamak core | 10,000 | 10²⁰ | 5 |
| Tokamak edge | 100 | 10¹⁹ | 5 |
| Fluorescent lamp | 1-2 | 10¹⁶ | — |
| Neon sign | 1-3 | 10¹⁴ | — |
| Arc discharge | 1-2 | 10²² | — |
| Plasma processing | 3-5 | 10¹⁶ | 10⁻² |

*Note: 1 eV ≈ 11,600 K. Plasma temperature is conventionally expressed in electron-volts.*

## 2. Debye Shielding

### 2.1 The Problem: Long-Range Coulomb Force

The Coulomb potential from a point charge $q$ is:

$$\phi(r) = \frac{1}{4\pi\epsilon_0} \frac{q}{r}$$

This $1/r$ dependence means the force is **long-range**—it extends to infinity. If every particle in a plasma interacted with every other particle via unshielded Coulomb forces, the system would be intractable.

However, plasmas exhibit a remarkable collective phenomenon: **Debye shielding**.

### 2.2 Derivation of the Debye Length

Consider a test charge $Q$ inserted into a plasma. Mobile electrons and ions will rearrange to shield this charge.

**Assumptions:**
1. Plasma is in thermal equilibrium (Boltzmann distribution)
2. Electrostatic potential is small: $|e\phi| \ll k_B T$
3. No external fields, no magnetic field

The electron and ion densities respond to the potential $\phi$ via Boltzmann relations:

$$n_e = n_0 \exp\left(\frac{e\phi}{k_B T_e}\right) \approx n_0 \left(1 + \frac{e\phi}{k_B T_e}\right)$$

$$n_i = n_0 \exp\left(-\frac{Ze\phi}{k_B T_i}\right) \approx n_0 \left(1 - \frac{Ze\phi}{k_B T_i}\right)$$

where $n_0$ is the background density, $Z$ is the ion charge state, and we've linearized for small $\phi$.

The charge density is:

$$\rho = e(Zn_i - n_e) = -en_0\left(\frac{e}{k_B T_e} + \frac{Ze}{k_B T_i}\right)\phi$$

Poisson's equation relates potential to charge density:

$$\nabla^2 \phi = -\frac{\rho}{\epsilon_0} = \frac{n_0 e}{\epsilon_0}\left(\frac{e}{k_B T_e} + \frac{Ze}{k_B T_i}\right)\phi$$

Define the **Debye length** $\lambda_D$:

$$\frac{1}{\lambda_D^2} = \frac{n_0 e^2}{\epsilon_0 k_B T_e} + \frac{Zn_0 e^2}{\epsilon_0 k_B T_i} = \frac{n_0 e^2}{\epsilon_0 k_B}\left(\frac{1}{T_e} + \frac{Z}{T_i}\right)$$

For a **single-species** case (or when $T_e \approx T_i = T$ and $Z=1$):

$$\lambda_D = \sqrt{\frac{\epsilon_0 k_B T}{n_0 e^2}}$$

For **electrons only** (often the dominant contribution due to higher mobility):

$$\lambda_{De} = \sqrt{\frac{\epsilon_0 k_B T_e}{n_e e^2}} \approx 7.43 \times 10^3 \sqrt{\frac{T_e[\text{eV}]}{n_e[\text{m}^{-3}]}} \quad [\text{m}]$$

**Physical interpretation:** The Debye length is the distance over which charge imbalances are shielded by mobile charge carriers.

Poisson's equation becomes:

$$\nabla^2 \phi = \frac{\phi}{\lambda_D^2}$$

For spherical symmetry (test charge at origin):

$$\frac{1}{r^2}\frac{d}{dr}\left(r^2 \frac{d\phi}{dr}\right) = \frac{\phi}{\lambda_D^2}$$

Solution:

$$\phi(r) = \frac{Q}{4\pi\epsilon_0 r} e^{-r/\lambda_D}$$

This is the **Debye-Hückel potential**—the Coulomb potential screened by an exponential factor.

```
Shielded Potential:

φ(r) ∝ (1/r) × exp(-r/λ_D)

         |
    Q    |    ___
  -----  |   /   \___
    |    |  /        \____
    |    | /              \______
----+----+-------------------------- r
    0   λ_D

Unshielded: 1/r (dashed)
Shielded:   (1/r)exp(-r/λ_D) (solid)

For r ≪ λ_D: full Coulomb force
For r ≫ λ_D: exponentially suppressed
```

### 2.3 The Plasma Parameter

The number of particles within a Debye sphere is:

$$N_D = n \cdot \frac{4}{3}\pi\lambda_D^3 = n\lambda_D^3 \cdot \frac{4\pi}{3}$$

For a system to exhibit **collective plasma behavior**, we require:

$$n\lambda_D^3 \gg 1$$

**Interpretation:** Many particles must be within a Debye sphere so that collective shielding dominates over individual particle interactions.

When $n\lambda_D^3 \ll 1$, the system behaves as a **weakly coupled gas** where binary collisions dominate (not a true plasma).

**Plasma coupling parameter:**

$$\Gamma = \frac{\text{potential energy}}{\text{kinetic energy}} = \frac{e^2}{4\pi\epsilon_0 a k_B T}$$

where $a = (3/4\pi n)^{1/3}$ is the mean inter-particle spacing.

- **Weakly coupled plasma**: $\Gamma \ll 1$ (most plasmas, including fusion devices)
- **Strongly coupled plasma**: $\Gamma \gtrsim 1$ (e.g., white dwarf interiors, dusty plasmas)

For weakly coupled plasmas:

$$\Gamma \sim \frac{1}{N_D^{1/3}}$$

So $n\lambda_D^3 \gg 1$ is equivalent to $\Gamma \ll 1$.

## 3. Quasi-Neutrality

### 3.1 Charge Neutrality Condition

At length scales much larger than the Debye length ($L \gg \lambda_D$), plasmas are **quasi-neutral**:

$$n_e \approx Z_i n_i$$

**Why?** Any charge imbalance creates large electric fields:

$$E \sim \frac{e(n_i - n_e)L}{\epsilon_0}$$

These fields accelerate particles to restore neutrality on timescales of order the plasma period $\omega_{pe}^{-1}$ (defined below).

### 3.2 Plasma Approximation

In most plasma calculations, we can assume:
1. **Quasi-neutrality**: $n_e = Zn_i$ for $L \gg \lambda_D$
2. **Collective behavior**: $n\lambda_D^3 \gg 1$

These two conditions define the **plasma regime**:

```
Plasma Criteria:

    n λ_D³ ≫ 1         (many particles in Debye sphere)
    L ≫ λ_D            (system size ≫ Debye length)

Together these imply:
    - Long-range collective interactions
    - Quasi-neutrality
    - Plasma oscillations and waves
```

## 4. Plasma Frequency

### 4.1 Electron Plasma Oscillations

Consider a cold, uniform plasma ($T=0$) with immobile ions. Displace all electrons by a small distance $\delta x$ relative to ions.

**Charge separation creates an electric field:**

Using Gauss's law for a slab of displaced electrons:

$$E = \frac{n_e e \delta x}{\epsilon_0}$$

**Equation of motion for an electron:**

$$m_e \frac{d^2(\delta x)}{dt^2} = -eE = -\frac{n_e e^2}{\epsilon_0} \delta x$$

This is simple harmonic motion with angular frequency:

$$\omega_{pe}^2 = \frac{n_e e^2}{\epsilon_0 m_e}$$

Define the **electron plasma frequency**:

$$\omega_{pe} = \sqrt{\frac{n_e e^2}{\epsilon_0 m_e}} \approx 5.64 \times 10^4 \sqrt{n_e[\text{m}^{-3}]} \quad [\text{rad/s}]$$

**Physical interpretation:** The natural oscillation frequency of electrons about the ion background. This is the fastest timescale in most plasmas.

### 4.2 Ion Plasma Frequency

Similarly, for ions:

$$\omega_{pi} = \sqrt{\frac{n_i Z^2 e^2}{\epsilon_0 m_i}}$$

Since $m_i \gg m_e$ (typically $m_i/m_e \sim 1836$ for hydrogen), we have:

$$\frac{\omega_{pi}}{\omega_{pe}} = \sqrt{\frac{m_e}{m_i}} \approx \frac{1}{43} \quad \text{(for protons)}$$

Ion oscillations are much slower than electron oscillations.

### 4.3 Plasma Period

The **plasma period** is:

$$\tau_{pe} = \frac{2\pi}{\omega_{pe}} \approx 1.11 \times 10^{-4} \frac{1}{\sqrt{n_e[\text{m}^{-3}]}} \quad [\text{s}]$$

For typical fusion plasmas ($n_e \sim 10^{20}$ m$^{-3}$):

$$\tau_{pe} \sim 10^{-14} \text{ s} = 10 \text{ fs}$$

This sets the fundamental timescale for electrostatic phenomena.

## 5. Gyrofrequency and Larmor Radius

### 5.1 Cyclotron Motion

A charged particle in a magnetic field $\mathbf{B}$ experiences the Lorentz force:

$$\mathbf{F} = q\mathbf{v} \times \mathbf{B}$$

For motion perpendicular to $\mathbf{B}$, the force is perpendicular to velocity, causing circular motion.

The **gyrofrequency** (or cyclotron frequency) is:

$$\omega_c = \frac{|q|B}{m}$$

**For electrons:**

$$\omega_{ce} = \frac{eB}{m_e} \approx 1.76 \times 10^{11} B[\text{T}] \quad [\text{rad/s}]$$

**For ions:**

$$\omega_{ci} = \frac{ZeB}{m_i}$$

For protons ($Z=1$, $m_i = 1836 m_e$):

$$\omega_{ci} \approx 9.58 \times 10^7 B[\text{T}] \quad [\text{rad/s}]$$

Note: $\omega_{ci}/\omega_{ce} = m_e/m_i \ll 1$ — ions gyrate much slower than electrons.

### 5.2 Larmor Radius

The radius of the circular orbit (gyroradius or Larmor radius) is found by balancing centripetal acceleration:

$$m\frac{v_\perp^2}{r_L} = qv_\perp B$$

$$r_L = \frac{mv_\perp}{qB} = \frac{v_\perp}{\omega_c}$$

**Thermal Larmor radius:** Using thermal velocity $v_{th} = \sqrt{k_B T/m}$:

$$r_{L,thermal} = \frac{v_{th}}{\omega_c} = \frac{\sqrt{k_B T m}}{qB}$$

**For electrons:**

$$r_{Le} = \frac{\sqrt{2k_B T_e m_e}}{eB} \approx 2.28 \times 10^{-6} \frac{\sqrt{T_e[\text{eV}]}}{B[\text{T}]} \quad [\text{m}]$$

**For ions (protons):**

$$r_{Li} = \frac{\sqrt{2k_B T_i m_p}}{eB} \approx 9.77 \times 10^{-5} \frac{\sqrt{T_i[\text{eV}]}}{B[\text{T}]} \quad [\text{m}]$$

The Larmor radius sets the scale for perpendicular particle transport and determines when magnetic field effects are important.

## 6. Plasma Beta

### 6.1 Definition

The **plasma beta** is the ratio of thermal pressure to magnetic pressure:

$$\beta = \frac{p}{p_B} = \frac{2\mu_0 nk_B T}{B^2}$$

where:
- Thermal pressure: $p = nk_B T$ (summed over species)
- Magnetic pressure: $p_B = B^2/(2\mu_0)$

For a plasma with electrons and ions:

$$\beta = \frac{2\mu_0 (n_e k_B T_e + n_i k_B T_i)}{B^2}$$

Assuming quasi-neutrality ($n_e = n_i = n$) and $T_e \approx T_i = T$:

$$\beta = \frac{4\mu_0 n k_B T}{B^2}$$

### 6.2 Physical Interpretation

- **$\beta \ll 1$**: Magnetic pressure dominates (magnetized plasma)
  - Particles are tightly bound to field lines
  - Field lines are approximately rigid
  - Examples: tokamak core, solar corona, magnetosphere

- **$\beta \sim 1$**: Thermal and magnetic pressures comparable
  - Field lines can be distorted by plasma pressure
  - Examples: tokamak edge, reversed-field pinch

- **$\beta \gg 1$**: Thermal pressure dominates (unmagnetized plasma)
  - Magnetic field has little effect on dynamics
  - Examples: early universe, some laser plasmas

### 6.3 Critical Beta and MHD Stability

In magnetic confinement fusion, there is a **critical beta** $\beta_c$ above which the plasma becomes MHD unstable. For tokamaks:

$$\beta_c \sim \frac{I_p}{aB_0}$$

where $I_p$ is plasma current, $a$ is minor radius, $B_0$ is toroidal field. Typical values: $\beta_c \sim 0.02 - 0.05$ (2-5%).

High-$\beta$ plasmas are desirable for fusion (higher pressure → higher fusion power), but stability limits exist.

## 7. Characteristic Scales and Orderings

### 7.1 Summary of Plasma Parameters

For a plasma characterized by density $n$, temperature $T$, and magnetic field $B$:

| Parameter | Symbol | Formula | Units |
|-----------|--------|---------|-------|
| Debye length | $\lambda_D$ | $\sqrt{\epsilon_0 k_B T/(n e^2)}$ | m |
| Plasma frequency | $\omega_{pe}$ | $\sqrt{n e^2/(\epsilon_0 m_e)}$ | rad/s |
| Electron gyrofrequency | $\omega_{ce}$ | $eB/m_e$ | rad/s |
| Ion gyrofrequency | $\omega_{ci}$ | $eB/m_i$ | rad/s |
| Electron Larmor radius | $r_{Le}$ | $v_{te}/\omega_{ce}$ | m |
| Ion Larmor radius | $r_{Li}$ | $v_{ti}/\omega_{ci}$ | m |
| Plasma beta | $\beta$ | $2\mu_0 nk_B T/B^2$ | — |

where $v_{te} = \sqrt{k_B T_e/m_e}$ and $v_{ti} = \sqrt{k_B T_i/m_i}$ are thermal speeds.

### 7.2 Typical Orderings

For magnetized plasmas:

**Length scales:**
$$r_{Le} \ll r_{Li} \ll L_\parallel \sim L_\perp$$

where $L_\parallel$ and $L_\perp$ are characteristic parallel and perpendicular scale lengths.

**Frequency orderings:**
$$\omega_{ci} \ll \omega_{ce} \ll \omega_{pe}$$

**Time scales:**
$$\tau_{pe} \ll \tau_{ce} \ll \tau_{ci}$$

These orderings are exploited in **asymptotic expansions** to simplify plasma models:
- **Drift-kinetic theory**: average over fast gyro-motion
- **Gyrokinetic theory**: average over gyro-phase
- **MHD**: assume time scales $\gg \tau_{ce}$, length scales $\gg r_{Li}$

## 8. Computational Tools

### 8.1 Plasma Parameter Calculator

Here is a Python function to compute all basic plasma parameters:

```python
import numpy as np
import matplotlib.pyplot as plt

# Physical constants
e = 1.602176634e-19      # Elementary charge [C]
m_e = 9.1093837015e-31   # Electron mass [kg]
m_p = 1.672621898e-27    # Proton mass [kg]
epsilon_0 = 8.8541878128e-12  # Permittivity [F/m]
k_B = 1.380649e-23       # Boltzmann constant [J/K]
mu_0 = 4e-7 * np.pi      # Permeability [H/m]
eV_to_K = 11604.518      # Conversion factor eV to K

class PlasmaParameters:
    """Compute and store fundamental plasma parameters."""

    def __init__(self, n_e, T_e, B=0, T_i=None, Z=1, A=1):
        """
        Parameters:
        -----------
        n_e : float
            Electron density [m^-3]
        T_e : float
            Electron temperature [eV]
        B : float, optional
            Magnetic field strength [T]
        T_i : float, optional
            Ion temperature [eV]. If None, assumes T_i = T_e
        Z : int, optional
            Ion charge state (default: 1)
        A : int, optional
            Ion mass number (default: 1 for hydrogen)
        """
        self.n_e = n_e
        self.T_e = T_e
        self.B = B
        self.T_i = T_i if T_i is not None else T_e
        self.Z = Z
        self.A = A
        self.m_i = A * m_p

        # Compute all parameters
        self._compute_parameters()

    def _compute_parameters(self):
        """Compute all plasma parameters."""

        # Debye length [m]
        # T_e is given in eV; multiply by eV_to_K to get Kelvin, then by k_B to
        # get Joules. This avoids propagating a separate unit throughout the class.
        self.lambda_D = np.sqrt(epsilon_0 * k_B * self.T_e * eV_to_K /
                                (self.n_e * e**2))

        # Plasma parameter (number of particles in Debye sphere)
        # N_D >> 1 is the fundamental criterion for collective plasma behavior;
        # grouping it here alongside lambda_D makes it trivial to check the
        # plasma criterion immediately after calculating the shielding scale.
        self.N_D = self.n_e * (4*np.pi/3) * self.lambda_D**3

        # Electron plasma frequency [rad/s]
        # omega_pe depends only on density (not temperature), because it arises
        # from the restoring force of the ion background, not thermal pressure.
        self.omega_pe = np.sqrt(self.n_e * e**2 / (epsilon_0 * m_e))
        self.f_pe = self.omega_pe / (2*np.pi)  # [Hz]

        # Ion plasma frequency [rad/s]
        # omega_pi << omega_pe (by sqrt(m_e/m_i)); keeping both lets users
        # verify the frequency ordering that underpins many fluid approximations.
        self.omega_pi = np.sqrt(self.n_e * self.Z**2 * e**2 /
                                (epsilon_0 * self.m_i))
        self.f_pi = self.omega_pi / (2*np.pi)  # [Hz]

        # Thermal velocities [m/s]
        # Using the same eV_to_K * k_B pattern as lambda_D keeps unit conversion
        # centralized; v_te and v_ti set the speed scales for all subsequent drifts.
        self.v_te = np.sqrt(k_B * self.T_e * eV_to_K / m_e)
        self.v_ti = np.sqrt(k_B * self.T_i * eV_to_K / self.m_i)

        if self.B > 0:
            # Gyrofrequency and Larmor radius are only meaningful when B > 0;
            # guarding with this condition prevents division-by-zero and signals
            # to the caller that magnetization effects do not apply for B = 0.
            self.omega_ce = e * self.B / m_e
            self.f_ce = self.omega_ce / (2*np.pi)  # [Hz]

            # Ion gyrofrequency [rad/s]
            self.omega_ci = self.Z * e * self.B / self.m_i
            self.f_ci = self.omega_ci / (2*np.pi)  # [Hz]

            # Larmor radii [m]
            # r_Le = v_te / omega_ce rather than the exact formula mv/qB because
            # the thermal speed already encodes the temperature dependence cleanly.
            self.r_Le = self.v_te / self.omega_ce
            self.r_Li = self.v_ti / self.omega_ci

            # Plasma beta
            # Sum T_e + T_i in the thermal pressure so beta reflects both species;
            # beta < 1 means magnetic pressure dominates and confines the plasma.
            p_thermal = self.n_e * k_B * (self.T_e + self.T_i) * eV_to_K
            p_magnetic = self.B**2 / (2 * mu_0)
            self.beta = p_thermal / p_magnetic
        else:
            self.omega_ce = None
            self.omega_ci = None
            self.r_Le = None
            self.r_Li = None
            self.beta = None

    def print_summary(self):
        """Print a formatted summary of plasma parameters."""
        print("="*60)
        print("PLASMA PARAMETERS")
        print("="*60)
        print(f"Input Parameters:")
        print(f"  Electron density:     n_e = {self.n_e:.3e} m^-3")
        # Show temperature in both eV and K so the reader can compare directly
        # with tables that use either convention (plasma literature uses eV,
        # thermodynamics literature uses K).
        print(f"  Electron temperature: T_e = {self.T_e:.3f} eV ({self.T_e*eV_to_K:.3e} K)")
        print(f"  Ion temperature:      T_i = {self.T_i:.3f} eV ({self.T_i*eV_to_K:.3e} K)")
        print(f"  Magnetic field:       B   = {self.B:.3f} T")
        print(f"  Ion charge/mass:      Z   = {self.Z}, A = {self.A}")
        print("-"*60)

        print(f"Debye Shielding:")
        print(f"  Debye length:         λ_D = {self.lambda_D:.3e} m")
        print(f"  Plasma parameter:     N_D = {self.N_D:.3e}")
        # Threshold of 100 (not 1) is conservative: N_D >> 1 is the theoretical
        # requirement, but N_D ~ 100 ensures the statistical average is reliable.
        print(f"  Plasma criterion:     N_D >> 1? {self.N_D > 100}")
        print("-"*60)

        print(f"Plasma Frequencies:")
        print(f"  Electron plasma freq: ω_pe = {self.omega_pe:.3e} rad/s ({self.f_pe:.3e} Hz)")
        print(f"  Ion plasma freq:      ω_pi = {self.omega_pi:.3e} rad/s ({self.f_pi:.3e} Hz)")
        # Printing the ratio makes it immediately clear how much faster electrons
        # oscillate than ions, motivating the separation of electron/ion timescales
        # used throughout plasma theory.
        print(f"  Ratio:                ω_pe/ω_pi = {self.omega_pe/self.omega_pi:.2f}")
        print("-"*60)

        print(f"Thermal Velocities:")
        print(f"  Electron thermal vel: v_te = {self.v_te:.3e} m/s")
        print(f"  Ion thermal vel:      v_ti = {self.v_ti:.3e} m/s")
        print(f"  Ratio:                v_te/v_ti = {self.v_te/self.v_ti:.2f}")
        print("-"*60)

        if self.B > 0:
            print(f"Magnetic Field Effects:")
            print(f"  Electron gyrofreq:    ω_ce = {self.omega_ce:.3e} rad/s ({self.f_ce:.3e} Hz)")
            print(f"  Ion gyrofreq:         ω_ci = {self.omega_ci:.3e} rad/s ({self.f_ci:.3e} Hz)")
            print(f"  Electron Larmor:      r_Le = {self.r_Le:.3e} m")
            print(f"  Ion Larmor:           r_Li = {self.r_Li:.3e} m")
            print(f"  Plasma beta:          β    = {self.beta:.3e}")
            print(f"  Regime:               ", end="")
            # Beta thresholds (0.01, 0.1, 10) correspond to practically meaningful
            # boundaries: β < 0.01 is typical of deep magnetosphere and tokamak
            # cores where field-line bending is negligible; β > 10 means the
            # magnetic field is dynamically irrelevant.
            if self.beta < 0.01:
                print("Strongly magnetized (β << 1)")
            elif self.beta < 0.1:
                print("Magnetized (β < 1)")
            elif self.beta < 10:
                print("Moderate β")
            else:
                print("Weakly magnetized (β > 1)")
            print("-"*60)

            print(f"Frequency Orderings:")
            print(f"  ω_ci/ω_ce = {self.omega_ci/self.omega_ce:.3e}")
            print(f"  ω_ce/ω_pe = {self.omega_ce/self.omega_pe:.3e}")
            print(f"  ω_pi/ω_pe = {self.omega_pi/self.omega_pe:.3e}")
            print("-"*60)

            print(f"Length Scale Orderings:")
            print(f"  r_Le/λ_D  = {self.r_Le/self.lambda_D:.3e}")
            print(f"  r_Li/r_Le = {self.r_Li/self.r_Le:.3e}")

        print("="*60)


# Example usage
if __name__ == "__main__":
    # Three examples span ~13 orders of magnitude in density and 4 in temperature,
    # chosen to cover qualitatively distinct regimes: fusion (strongly magnetized,
    # collisionless), space (weakly magnetized, collisionless), and industrial
    # plasma (unmagnetized, partially collisional).
    print("\n### Example 1: Tokamak Core ###\n")
    tokamak = PlasmaParameters(n_e=1e20, T_e=10000, B=5, T_i=10000, Z=1, A=2)
    tokamak.print_summary()

    print("\n### Example 2: Solar Wind (1 AU) ###\n")
    solar_wind = PlasmaParameters(n_e=1e7, T_e=10, B=1e-9, T_i=10, Z=1, A=1)
    solar_wind.print_summary()

    print("\n### Example 3: Fluorescent Lamp ###\n")
    fluorescent = PlasmaParameters(n_e=1e16, T_e=2, B=0, T_i=0.1, Z=1, A=40)
    fluorescent.print_summary()
```

### 8.2 Plasma Parameter Space Visualization

Let's visualize the parameter space of different plasmas:

```python
def plot_plasma_parameter_space():
    """Plot various plasmas in (n, T) space with characteristic regions."""

    # Define various plasmas
    plasmas = {
        'Interstellar medium': (1e6, 1),
        'Solar wind': (1e7, 10),
        'Ionosphere': (1e12, 0.1),
        'Solar corona': (1e14, 100),
        'Glow discharge': (1e16, 3),
        'Tokamak edge': (1e19, 100),
        'Tokamak core': (1e20, 10000),
        'Laser plasma': (1e26, 1000),
        'White dwarf': (1e36, 10000),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: n-T space with Debye length contours
    n_range = np.logspace(4, 38, 100)
    T_range = np.logspace(-2, 5, 100)
    N, T = np.meshgrid(n_range, T_range)

    # Debye length [m]
    lambda_D = np.sqrt(epsilon_0 * k_B * T * eV_to_K / (N * e**2))

    contour1 = ax1.contour(N, T, lambda_D,
                           levels=[1e-12, 1e-9, 1e-6, 1e-3, 1, 1e3],
                           colors='gray', alpha=0.5, linewidths=0.8)
    ax1.clabel(contour1, inline=True, fontsize=8, fmt='λ_D=%gm')

    # Plot plasmas
    for name, (n, T_eV) in plasmas.items():
        ax1.scatter(n, T_eV, s=100, alpha=0.7, edgecolors='black', linewidths=1.5)
        ax1.annotate(name, (n, T_eV), fontsize=9,
                    xytext=(5, 5), textcoords='offset points')

    # Plasma parameter N_D = 1 line
    T_for_ND1 = 3/(4*np.pi) * (e**2 / (epsilon_0 * k_B * eV_to_K)) * n_range**(-2/3)
    ax1.plot(n_range, T_for_ND1, 'r--', linewidth=2,
             label=r'$n\lambda_D^3 = 1$ (plasma boundary)')

    ax1.set_xlabel(r'Electron Density $n_e$ [m$^{-3}$]', fontsize=12)
    ax1.set_ylabel(r'Temperature $T$ [eV]', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1e4, 1e38)
    ax1.set_ylim(1e-2, 1e5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_title('Plasma Parameter Space', fontsize=14, fontweight='bold')

    # Plot 2: Coupling parameter space
    # Compute coupling parameter Γ
    a = (3/(4*np.pi*N))**(1/3)  # mean inter-particle spacing
    Gamma = e**2 / (4*np.pi*epsilon_0 * a * k_B * T * eV_to_K)

    contour2 = ax2.contourf(N, T, Gamma,
                            levels=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
                            cmap='RdYlBu_r', alpha=0.7)
    cbar = plt.colorbar(contour2, ax=ax2, label=r'Coupling Parameter $\Gamma$')

    # Mark Γ = 1 (strongly coupled boundary)
    cs = ax2.contour(N, T, Gamma, levels=[1], colors='red', linewidths=3)
    ax2.clabel(cs, inline=True, fontsize=12, fmt=r'$\Gamma=1$')

    # Plot plasmas
    for name, (n, T_eV) in plasmas.items():
        ax2.scatter(n, T_eV, s=100, alpha=0.9, edgecolors='black',
                   linewidths=1.5, zorder=10)
        ax2.annotate(name, (n, T_eV), fontsize=9,
                    xytext=(5, 5), textcoords='offset points')

    ax2.set_xlabel(r'Electron Density $n_e$ [m$^{-3}$]', fontsize=12)
    ax2.set_ylabel(r'Temperature $T$ [eV]', fontsize=12)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(1e4, 1e38)
    ax2.set_ylim(1e-2, 1e5)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Plasma Coupling Parameter Space', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('plasma_parameter_space.png', dpi=150, bbox_inches='tight')
    plt.show()

# Run visualization
plot_plasma_parameter_space()
```

### 8.3 Frequency and Length Scale Comparison

```python
def compare_plasma_scales():
    """Compare characteristic frequencies and length scales for various plasmas."""

    plasma_configs = {
        'Tokamak core': {'n_e': 1e20, 'T_e': 10000, 'B': 5, 'T_i': 10000},
        'Tokamak edge': {'n_e': 1e19, 'T_e': 100, 'B': 5, 'T_i': 100},
        'Solar wind': {'n_e': 1e7, 'T_e': 10, 'B': 1e-9, 'T_i': 10},
        'Ionosphere': {'n_e': 1e12, 'T_e': 0.1, 'B': 5e-5, 'T_i': 0.1},
        'Fluorescent': {'n_e': 1e16, 'T_e': 2, 'B': 0, 'T_i': 0.1},
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = list(plasma_configs.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    # Compute parameters
    results = {}
    for name, config in plasma_configs.items():
        pp = PlasmaParameters(**config)
        results[name] = pp

    # Plot 1: Frequencies
    ax = axes[0, 0]
    x = np.arange(len(names))
    width = 0.2

    omega_pe_vals = [results[n].omega_pe for n in names]
    omega_pi_vals = [results[n].omega_pi for n in names]
    omega_ce_vals = [results[n].omega_ce if results[n].omega_ce else 0 for n in names]
    omega_ci_vals = [results[n].omega_ci if results[n].omega_ci else 0 for n in names]

    ax.bar(x - 1.5*width, omega_pe_vals, width, label=r'$\omega_{pe}$', alpha=0.8)
    ax.bar(x - 0.5*width, omega_pi_vals, width, label=r'$\omega_{pi}$', alpha=0.8)
    ax.bar(x + 0.5*width, omega_ce_vals, width, label=r'$\omega_{ce}$', alpha=0.8)
    ax.bar(x + 1.5*width, omega_ci_vals, width, label=r'$\omega_{ci}$', alpha=0.8)

    ax.set_ylabel('Frequency [rad/s]', fontsize=11)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Characteristic Frequencies', fontsize=12, fontweight='bold')

    # Plot 2: Length scales
    ax = axes[0, 1]
    lambda_D_vals = [results[n].lambda_D for n in names]
    r_Le_vals = [results[n].r_Le if results[n].r_Le else np.nan for n in names]
    r_Li_vals = [results[n].r_Li if results[n].r_Li else np.nan for n in names]

    ax.bar(x - width, lambda_D_vals, width, label=r'$\lambda_D$', alpha=0.8)
    ax.bar(x, r_Le_vals, width, label=r'$r_{Le}$', alpha=0.8)
    ax.bar(x + width, r_Li_vals, width, label=r'$r_{Li}$', alpha=0.8)

    ax.set_ylabel('Length [m]', fontsize=11)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Characteristic Length Scales', fontsize=12, fontweight='bold')

    # Plot 3: Plasma parameter and Beta
    ax = axes[1, 0]
    N_D_vals = [results[n].N_D for n in names]

    ax.bar(x, N_D_vals, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, label=r'$N_D = 1$')
    ax.set_ylabel(r'Plasma Parameter $N_D$', fontsize=11)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Plasma Parameter (Debye sphere population)', fontsize=12, fontweight='bold')

    # Plot 4: Beta
    ax = axes[1, 1]
    beta_vals = [results[n].beta if results[n].beta else np.nan for n in names]

    ax.bar(x, beta_vals, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, label=r'$\beta = 1$')
    ax.set_ylabel(r'Plasma Beta $\beta$', fontsize=11)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Plasma Beta (thermal/magnetic pressure)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('plasma_scales_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# Run comparison
compare_plasma_scales()
```

### 8.4 Interactive Parameter Explorer

```python
def interactive_parameter_scan():
    """
    Scan plasma parameters to understand scaling relationships.
    """
    # Vary density at fixed temperature
    n_vals = np.logspace(14, 22, 50)
    T_fixed = 1000  # eV
    B_fixed = 2     # T

    lambda_D = []
    omega_pe = []
    r_Le = []
    beta = []

    for n in n_vals:
        pp = PlasmaParameters(n_e=n, T_e=T_fixed, B=B_fixed)
        lambda_D.append(pp.lambda_D)
        omega_pe.append(pp.omega_pe)
        r_Le.append(pp.r_Le)
        beta.append(pp.beta)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.loglog(n_vals, lambda_D, 'b-', linewidth=2)
    ax.set_xlabel(r'Density $n_e$ [m$^{-3}$]', fontsize=11)
    ax.set_ylabel(r'Debye Length $\lambda_D$ [m]', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(r'$\lambda_D \propto n^{-1/2}$', fontsize=12)

    ax = axes[0, 1]
    ax.loglog(n_vals, omega_pe, 'r-', linewidth=2)
    ax.set_xlabel(r'Density $n_e$ [m$^{-3}$]', fontsize=11)
    ax.set_ylabel(r'Plasma Frequency $\omega_{pe}$ [rad/s]', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(r'$\omega_{pe} \propto n^{1/2}$', fontsize=12)

    ax = axes[1, 0]
    ax.loglog(n_vals, r_Le, 'g-', linewidth=2)
    ax.set_xlabel(r'Density $n_e$ [m$^{-3}$]', fontsize=11)
    ax.set_ylabel(r'Electron Larmor Radius $r_{Le}$ [m]', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(r'$r_{Le}$ independent of $n$ (fixed $T, B$)', fontsize=12)

    ax = axes[1, 1]
    ax.loglog(n_vals, beta, 'm-', linewidth=2)
    ax.set_xlabel(r'Density $n_e$ [m$^{-3}$]', fontsize=11)
    ax.set_ylabel(r'Plasma Beta $\beta$', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(r'$\beta \propto n$ (fixed $T, B$)', fontsize=12)

    plt.suptitle(f'Parameter Scaling with Density (T={T_fixed} eV, B={B_fixed} T)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('parameter_scaling_density.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Now vary temperature at fixed density
    T_vals = np.logspace(0, 4, 50)
    n_fixed = 1e19  # m^-3

    lambda_D = []
    v_te = []
    r_Le = []
    beta = []

    for T in T_vals:
        pp = PlasmaParameters(n_e=n_fixed, T_e=T, B=B_fixed)
        lambda_D.append(pp.lambda_D)
        v_te.append(pp.v_te)
        r_Le.append(pp.r_Le)
        beta.append(pp.beta)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.loglog(T_vals, lambda_D, 'b-', linewidth=2)
    ax.set_xlabel(r'Temperature $T_e$ [eV]', fontsize=11)
    ax.set_ylabel(r'Debye Length $\lambda_D$ [m]', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(r'$\lambda_D \propto T^{1/2}$', fontsize=12)

    ax = axes[0, 1]
    ax.loglog(T_vals, v_te, 'r-', linewidth=2)
    ax.set_xlabel(r'Temperature $T_e$ [eV]', fontsize=11)
    ax.set_ylabel(r'Thermal Velocity $v_{te}$ [m/s]', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(r'$v_{te} \propto T^{1/2}$', fontsize=12)

    ax = axes[1, 0]
    ax.loglog(T_vals, r_Le, 'g-', linewidth=2)
    ax.set_xlabel(r'Temperature $T_e$ [eV]', fontsize=11)
    ax.set_ylabel(r'Electron Larmor Radius $r_{Le}$ [m]', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(r'$r_{Le} \propto T^{1/2}$', fontsize=12)

    ax = axes[1, 1]
    ax.loglog(T_vals, beta, 'm-', linewidth=2)
    ax.set_xlabel(r'Temperature $T_e$ [eV]', fontsize=11)
    ax.set_ylabel(r'Plasma Beta $\beta$', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(r'$\beta \propto T$ (fixed $n, B$)', fontsize=12)

    plt.suptitle(f'Parameter Scaling with Temperature (n={n_fixed:.0e} m⁻³, B={B_fixed} T)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('parameter_scaling_temperature.png', dpi=150, bbox_inches='tight')
    plt.show()

# Run interactive scans
interactive_parameter_scan()
```

## Summary

In this lesson, we introduced plasma as the fourth state of matter and derived the fundamental parameters that characterize plasma behavior:

1. **Debye shielding**: Plasmas screen charge imbalances over the Debye length $\lambda_D = \sqrt{\epsilon_0 k_B T/(ne^2)}$, converting the long-range Coulomb force into an exponentially screened interaction.

2. **Plasma criteria**: For collective behavior, we require $n\lambda_D^3 \gg 1$ (many particles in Debye sphere) and quasi-neutrality at scales $L \gg \lambda_D$.

3. **Plasma frequency**: The natural electrostatic oscillation frequency $\omega_{pe} = \sqrt{ne^2/(\epsilon_0 m_e)}$ sets the fundamental timescale for electron dynamics.

4. **Gyrofrequency and Larmor radius**: In magnetic fields, particles gyrate at $\omega_c = |q|B/m$ with radius $r_L = v_\perp/\omega_c$, introducing anisotropy and new length/time scales.

5. **Plasma beta**: The ratio $\beta = 2\mu_0 nk_B T/B^2$ determines whether thermal or magnetic pressure dominates, with profound implications for confinement and stability.

These parameters form the foundation for understanding plasma behavior across 40 orders of magnitude in density and temperature—from the interstellar medium to fusion devices.

## Practice Problems

### Problem 1: Plasma Parameter Calculation

Consider a low-temperature plasma with $n_e = 10^{17}$ m$^{-3}$, $T_e = 3$ eV, and $B = 0.01$ T.

(a) Calculate the Debye length, plasma frequency, electron gyrofrequency, and electron Larmor radius.

(b) Verify that this system satisfies the plasma criteria $n\lambda_D^3 \gg 1$.

(c) Compare the timescales $\omega_{pe}^{-1}$ and $\omega_{ce}^{-1}$. Which is faster?

(d) If ions are argon (A=40), compute the ion gyrofrequency and Larmor radius. Compare to electron values.

### Problem 2: Debye Shielding in a Tokamak

A tokamak has core parameters: $n_e = 10^{20}$ m$^{-3}$, $T_e = 15$ keV, $T_i = 12$ keV.

(a) Calculate the electron and ion contributions to the Debye length separately, then find the effective Debye length.

(b) At what distance is the potential from a test charge reduced to $1/e$ of its Coulomb value?

(c) If the tokamak has a minor radius $a = 1$ m, verify that quasi-neutrality is an excellent approximation.

(d) How many electrons are within one Debye sphere?

### Problem 3: Scaling Analysis

For a hydrogen plasma in a magnetic field $B = 2$ T:

(a) Find the density $n$ at which the electron Larmor radius equals the Debye length, assuming $T_e = 1$ keV.

(b) Derive a general expression for the ratio $r_{Le}/\lambda_D$ in terms of $n$, $T_e$, and $B$.

(c) For the density found in (a), compute the plasma beta. Is this a magnetized or unmagnetized plasma?

### Problem 4: Solar Corona Parameters

The solar corona has $n_e \sim 10^{14}$ m$^{-3}$, $T_e \sim 100$ eV, and $B \sim 10^{-2}$ T.

(a) Compute all characteristic plasma parameters.

(b) Determine the plasma beta. What does this tell you about magnetic confinement of the corona?

(c) The Alfvén velocity is $v_A = B/\sqrt{\mu_0 n_i m_i}$ (for protons). Compare $v_A$ to the electron thermal velocity. What does this imply for wave propagation?

(d) At what distance from the Sun (in solar radii $R_\odot$) would the plasma become unmagnetized ($\beta \sim 1$), assuming density falls as $n \propto r^{-2}$ and $B \propto r^{-2}$?

### Problem 5: Laboratory Plasma Regimes

You are designing a plasma experiment and can vary $n$ and $T$ independently, with $10^{16} \le n \le 10^{21}$ m$^{-3}$ and $1 \le T_e \le 1000$ eV. A magnetic field of $B = 1$ T is available.

(a) Create a plot in $(n, T)$ space showing contours of constant $\lambda_D$, $\omega_{pe}$, $r_{Le}$, and $\beta$.

(b) Identify regions where:
   - The plasma is weakly coupled ($\Gamma < 0.1$)
   - The electron Larmor radius is smaller than 1 mm
   - The plasma beta is less than 0.01

(c) For $n = 10^{19}$ m$^{-3}$ and $T_e = 100$ eV, compute the collision frequency (you'll need concepts from Lesson 2—use $\nu_{ei} \sim 10^{-6} n_e \ln\Lambda / T_e^{3/2}$ in SI units with $\ln\Lambda \approx 15$ as an estimate).

(d) Determine whether this plasma is collisional or collisionless by comparing $\nu_{ei}$ to $\omega_{ce}$.

---

**Previous:** [Overview](./00_Overview.md) | **Next:** [Coulomb Collisions](./02_Coulomb_Collisions.md)
