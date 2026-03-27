# 12. Nonlinear Optics

[← Previous: 11. Holography](11_Holography.md) | [Next: 13. Quantum Optics Primer →](13_Quantum_Optics_Primer.md)

---

In everyday experience, light behaves linearly: shining two flashlights into a glass does not create new colors. The beams pass through each other without interaction. But at the extreme intensities produced by lasers — electric fields approaching the binding fields of atoms ($\sim 10^{11}\,\text{V/m}$) — the optical response of materials becomes nonlinear, and extraordinary things happen. Light can change color, beams can interact with each other, and materials can become transparent or opaque depending on the light intensity.

Nonlinear optics, born in 1961 when Franken and colleagues observed second harmonic generation just one year after the invention of the laser, has grown into a vast field with applications from green laser pointers (frequency-doubled infrared) to ultrafast pulse compression, optical parametric amplifiers, and quantum entangled photon sources. This lesson develops the theory from the nonlinear polarization, derives the key second- and third-order effects, and explores their applications.

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

1. Explain the origin of optical nonlinearity through the anharmonic oscillator model and express the nonlinear polarization as a power series
2. Identify and describe second-order ($\chi^{(2)}$) processes: second harmonic generation (SHG), sum/difference frequency generation (SFG/DFG)
3. Derive the phase matching condition and explain birefringent and quasi-phase matching techniques
4. Identify and describe third-order ($\chi^{(3)}$) processes: Kerr effect, self-phase modulation (SPM), four-wave mixing (FWM)
5. Explain the operating principle of optical parametric oscillators and amplifiers
6. Analyze applications including frequency conversion, ultrafast pulse generation, and entangled photon sources
7. Perform numerical simulations of second harmonic generation with phase matching

---

## Table of Contents

1. [Origin of Optical Nonlinearity](#1-origin-of-optical-nonlinearity)
2. [Nonlinear Polarization and Susceptibility](#2-nonlinear-polarization-and-susceptibility)
3. [Second Harmonic Generation (SHG)](#3-second-harmonic-generation-shg)
4. [Sum and Difference Frequency Generation](#4-sum-and-difference-frequency-generation)
5. [Phase Matching](#5-phase-matching)
6. [Third-Order Nonlinear Effects](#6-third-order-nonlinear-effects)
7. [Self-Phase Modulation and Solitons](#7-self-phase-modulation-and-solitons)
8. [Optical Parametric Processes](#8-optical-parametric-processes)
9. [Applications](#9-applications)
10. [Python Examples](#10-python-examples)
11. [Summary](#11-summary)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. Origin of Optical Nonlinearity

### 1.1 The Linear Regime

In linear optics, the polarization $\mathbf{P}$ (electric dipole moment per unit volume) of a material is proportional to the applied electric field:

$$\mathbf{P} = \epsilon_0 \chi^{(1)} \mathbf{E}$$

where $\chi^{(1)}$ is the linear susceptibility. This gives rise to the refractive index $n = \sqrt{1 + \chi^{(1)}}$ and linear absorption. In this regime, the superposition principle holds perfectly: light beams pass through each other without interaction.

### 1.2 The Anharmonic Oscillator Model

Consider an electron bound to a nucleus. In the linear regime, the restoring force is harmonic: $F = -\kappa x$. But at large displacements (strong fields), the restoring force becomes anharmonic:

$$F = -\kappa x - \kappa_2 x^2 - \kappa_3 x^3 - \cdots$$

The $x^2$ term (in non-centrosymmetric materials) gives rise to $\chi^{(2)}$ effects, and the $x^3$ term gives $\chi^{(3)}$ effects. The nonlinear terms are small corrections — typically $x_{\text{NL}}/x_{\text{L}} \sim E/E_{\text{atomic}} \sim 10^{-8}$ at moderate intensities — but they are detectable with coherent laser sources.

> **Analogy**: Imagine pushing a child on a swing (harmonic oscillator). At small amplitudes, the restoring force is perfectly proportional to displacement — push gently, get a gentle swing back. But push hard enough, and the swing reaches extreme angles where the motion is no longer simple harmonic — the period changes, the motion distorts. In materials, the "swing" is the electron's displacement from equilibrium, and the "nonlinear response" is the distortion that occurs when the driving electric field becomes comparable to internal atomic fields.

### 1.3 How Strong Must the Field Be?

The characteristic atomic electric field is:

$$E_{\text{atom}} = \frac{e}{4\pi\epsilon_0 a_0^2} \approx 5 \times 10^{11}\,\text{V/m}$$

A 1 W laser focused to a 10 $\mu$m spot has an intensity of $I \approx 3 \times 10^9\,\text{W/m}^2$ and a field strength of $E \approx 5 \times 10^7\,\text{V/m}$ — about $10^{-4}$ of $E_{\text{atom}}$. This seems tiny, but the $\chi^{(2)}$ processes involve the product of two such fields, and with careful phase matching over centimeters of crystal, conversion efficiencies of 50-80% are routinely achieved.

---

## 2. Nonlinear Polarization and Susceptibility

### 2.1 Power Series Expansion

The general nonlinear polarization is expanded as:

$$P_i = \epsilon_0\left[\chi^{(1)}_{ij}E_j + \chi^{(2)}_{ijk}E_jE_k + \chi^{(3)}_{ijkl}E_jE_kE_l + \cdots\right]$$

where:
- $\chi^{(1)}_{ij}$: linear susceptibility (rank-2 tensor) — refractive index, absorption
- $\chi^{(2)}_{ijk}$: second-order susceptibility (rank-3 tensor) — SHG, SFG, DFG, Pockels effect
- $\chi^{(3)}_{ijkl}$: third-order susceptibility (rank-4 tensor) — Kerr effect, THG, FWM

### 2.2 Symmetry Constraints on $\chi^{(2)}$

A crucial symmetry principle: **$\chi^{(2)}$ vanishes in centrosymmetric media** (materials with inversion symmetry). If the crystal has a center of symmetry, inverting all coordinates ($\mathbf{E} \to -\mathbf{E}$, $\mathbf{P} \to -\mathbf{P}$) requires:

$$-P = \chi^{(2)}(-E)(-E) = \chi^{(2)}E^2 = P$$

which implies $P = 0$. Therefore:
- **$\chi^{(2)} \neq 0$**: Non-centrosymmetric crystals (KDP, BBO, LiNbO$_3$, KTP), surfaces/interfaces
- **$\chi^{(2)} = 0$**: Centrosymmetric materials (glass, liquids, gases, Si, most metals)
- **$\chi^{(3)} \neq 0$**: All materials (no symmetry restriction)

### 2.3 Typical Values

| Material | $\chi^{(2)}$ (pm/V) | $\chi^{(3)}$ (m$^2$/V$^2$) | Primary use |
|----------|---------------------|----------------------------|-------------|
| BBO ($\beta$-BaB$_2$O$_4$) | 2.2 | — | UV SHG, OPO |
| LiNbO$_3$ | 27 | — | SHG, EO modulation, QPM |
| KTP (KTiOPO$_4$) | 16 | — | SHG (green laser pointers) |
| Fused silica | 0 (centrosymmetric) | $2.5 \times 10^{-22}$ | Fiber Kerr effect |
| CS$_2$ | 0 (liquid) | $3 \times 10^{-20}$ | Kerr gating |

---

## 3. Second Harmonic Generation (SHG)

### 3.1 The Process

When an intense beam at frequency $\omega$ passes through a $\chi^{(2)}$ crystal, the nonlinear polarization contains a term at $2\omega$:

$$P^{(2)}(2\omega) = \epsilon_0 \chi^{(2)} E(\omega)^2$$

This oscillating polarization at $2\omega$ radiates a new electromagnetic wave at twice the frequency (half the wavelength). A 1064 nm infrared laser produces 532 nm green light.

### 3.2 Historical Context

Peter Franken's group at the University of Michigan demonstrated SHG in 1961 — just one year after the first laser. They focused a ruby laser (694.3 nm, red) into a quartz crystal and detected a faint signal at 347.15 nm (UV). The signal was so weak that the journal editor reportedly thought the faint dot on the film plate was a blemish and had it removed from the published photograph.

### 3.3 Coupled Wave Equations

The fundamental ($\omega$) and second harmonic ($2\omega$) fields evolve according to:

$$\frac{dA_{2\omega}}{dz} = -i\kappa_1 A_\omega^2 e^{i\Delta kz}$$

$$\frac{dA_\omega}{dz} = -i\kappa_2 A_{2\omega}A_\omega^* e^{-i\Delta kz}$$

where:
- $\kappa_1, \kappa_2$ are coupling coefficients proportional to $\chi^{(2)}$
- $\Delta k = k_{2\omega} - 2k_\omega$ is the **phase mismatch**

### 3.4 Undepleted Pump Approximation

When the conversion efficiency is low, $A_\omega \approx$ const, and:

$$A_{2\omega}(L) = -i\kappa_1 A_\omega^2 \frac{e^{i\Delta kL} - 1}{i\Delta k}$$

The SHG intensity is:

$$I_{2\omega} \propto \chi^{(2)2} I_\omega^2 L^2 \text{sinc}^2\!\left(\frac{\Delta kL}{2}\right)$$

The $\text{sinc}^2$ factor shows that efficient SHG requires $\Delta k \approx 0$ — **phase matching**.

### 3.5 The Coherence Length

Without phase matching, the SH signal oscillates as a function of crystal length with period:

$$L_c = \frac{\pi}{|\Delta k|} = \frac{\lambda}{4(n_{2\omega} - n_\omega)}$$

Typically $L_c \sim 10\text{-}100\,\mu\text{m}$ — far too short for practical SHG. Phase matching overcomes this limitation.

---

## 4. Sum and Difference Frequency Generation

### 4.1 Sum Frequency Generation (SFG)

Two beams at frequencies $\omega_1$ and $\omega_2$ mix in a $\chi^{(2)}$ crystal to produce a beam at $\omega_3 = \omega_1 + \omega_2$:

$$\omega_1 + \omega_2 \to \omega_3$$

Phase matching condition: $\mathbf{k}_1 + \mathbf{k}_2 = \mathbf{k}_3$ (energy and momentum conservation).

SHG is the special case $\omega_1 = \omega_2$.

### 4.2 Difference Frequency Generation (DFG)

A strong pump at $\omega_3$ and a weaker signal at $\omega_1$ produce an idler at $\omega_2 = \omega_3 - \omega_1$:

$$\omega_3 - \omega_1 \to \omega_2$$

Each photon at $\omega_3$ that is destroyed creates one photon at $\omega_1$ and one at $\omega_2$. The signal at $\omega_1$ is amplified (optical parametric amplification) — this is the basis of OPOs and OPAs.

### 4.3 Energy and Momentum Conservation

All $\chi^{(2)}$ processes obey:

**Energy conservation** (photon picture):
$$\hbar\omega_3 = \hbar\omega_1 + \hbar\omega_2$$

**Momentum conservation** (phase matching):
$$\hbar\mathbf{k}_3 = \hbar\mathbf{k}_1 + \hbar\mathbf{k}_2$$

These are exact quantum-mechanical conservation laws. The phase matching condition $\Delta\mathbf{k} = 0$ is simply momentum conservation for the participating photons.

---

## 5. Phase Matching

### 5.1 The Problem

In a dispersive medium, $n(\omega)$ increases with frequency (normal dispersion). Therefore:

$$k_{2\omega} = \frac{2\omega n(2\omega)}{c} \neq 2k_\omega = \frac{2\omega n(\omega)}{c}$$

because $n(2\omega) > n(\omega)$. The phase mismatch $\Delta k = k_{2\omega} - 2k_\omega > 0$ destroys the constructive buildup.

### 5.2 Birefringent Phase Matching

The solution exploits **birefringence**: in anisotropic crystals, the refractive index depends on polarization (ordinary $n_o$ and extraordinary $n_e$). By choosing the crystal orientation (angle $\theta$ relative to the optic axis), we can make:

$$n_e(2\omega, \theta) = n_o(\omega)$$

This is **Type I phase matching**: both fundamental photons have the same polarization (ordinary), and the SH photon has the orthogonal polarization (extraordinary).

For the extraordinary index at angle $\theta$:

$$\frac{1}{n_e^2(\theta)} = \frac{\cos^2\theta}{n_o^2} + \frac{\sin^2\theta}{n_e^2}$$

**Type II phase matching**: the two fundamental photons have orthogonal polarizations (one ordinary, one extraordinary):

$$n_e(\omega, \theta) + n_o(\omega) = 2n_e(2\omega, \theta)$$

### 5.3 Quasi-Phase Matching (QPM)

An alternative approach: periodically invert the sign of $\chi^{(2)}$ with period $\Lambda$, creating a **periodically poled** crystal:

$$\chi^{(2)}(z) = d_{\text{eff}}\,\text{sign}\!\left[\cos\!\left(\frac{2\pi z}{\Lambda}\right)\right]$$

The periodic inversion adds a grating vector $K_G = 2\pi/\Lambda$ that compensates the phase mismatch:

$$\Delta k = k_{2\omega} - 2k_\omega - K_G = 0$$

$$\boxed{\Lambda = \frac{2\pi}{\Delta k} = \frac{2L_c}{\pi}}$$

> **Analogy**: Phase matching is like pushing a child on a swing at just the right timing. If you push every time the swing reaches you (in phase), the swing goes higher and higher — this is phase-matched SHG, where the generated SH signal builds up constructively. If your pushes are randomly timed, sometimes you push forward and sometimes backward, and the swing barely moves — this is the phase-mismatched case. Quasi-phase matching is like pushing forward for half a swing cycle, then stepping aside for half a cycle, then pushing again — not as efficient as perfect timing, but the swing still builds up.

**Advantages of QPM**:
- Use the **largest** $\chi^{(2)}$ component ($d_{33}$ in LiNbO$_3$, typically 5x larger than the component used in birefringent PM)
- Engineer any phase matching wavelength by choosing $\Lambda$
- No walk-off (beams propagate along the crystal axis)
- Periodically poled lithium niobate (PPLN) is the workhorse material

---

## 6. Third-Order Nonlinear Effects

### 6.1 The Kerr Effect (Optical Kerr Effect)

The third-order polarization at frequency $\omega$ includes a term:

$$P^{(3)}(\omega) = 3\epsilon_0\chi^{(3)}|E(\omega)|^2 E(\omega)$$

This causes an intensity-dependent refractive index:

$$\boxed{n = n_0 + n_2 I}$$

where $n_2 = \frac{3\chi^{(3)}}{4n_0^2\epsilon_0 c}$ is the nonlinear refractive index. For silica glass: $n_2 \approx 2.6 \times 10^{-20}\,\text{m}^2/\text{W}$.

### 6.2 Self-Focusing

A beam with a Gaussian intensity profile experiences a stronger refractive index increase at the center (high intensity) than at the edges (low intensity), creating a positive lens. If the beam power exceeds the **critical power**:

$$P_{\text{cr}} = \frac{3.77\lambda^2}{8\pi n_0 n_2} \approx 3\,\text{MW} \quad (\text{in silica at 800 nm})$$

the beam collapses to a focus — **self-focusing**. This can lead to catastrophic damage in high-power laser systems and is responsible for filamentation in the atmosphere.

### 6.3 Four-Wave Mixing (FWM)

Three waves at $\omega_1, \omega_2, \omega_3$ interact through $\chi^{(3)}$ to generate a fourth wave at:

$$\omega_4 = \omega_1 + \omega_2 - \omega_3$$

with phase matching $\mathbf{k}_4 = \mathbf{k}_1 + \mathbf{k}_2 - \mathbf{k}_3$.

**Degenerate FWM** ($\omega_1 = \omega_2 = \omega_3 = \omega$): generates $\omega_4 = \omega$ — phase conjugation (optical phase conjugate mirror). This can undo wavefront distortion.

**Non-degenerate FWM**: Creates new frequency components. In fiber optics, FWM between WDM channels causes crosstalk — a significant impairment that drove the development of non-zero dispersion-shifted fiber.

---

## 7. Self-Phase Modulation and Solitons

### 7.1 Self-Phase Modulation (SPM)

A pulse propagating through a Kerr medium acquires an intensity-dependent phase:

$$\phi_{\text{NL}}(t) = -n_2 I(t) \frac{\omega_0 L}{c}$$

For a Gaussian pulse, the instantaneous frequency shifts:

$$\Delta\omega(t) = -\frac{d\phi_{\text{NL}}}{dt} = n_2\frac{\omega_0 L}{c}\frac{dI}{dt}$$

- **Leading edge** ($dI/dt > 0$): frequency decreases (redshift)
- **Trailing edge** ($dI/dt < 0$): frequency increases (blueshift)

SPM generates new frequencies, broadening the pulse spectrum without changing its temporal shape (in the absence of dispersion). This spectral broadening is the basis of **supercontinuum generation** — a single ultrashort pulse can generate a spectrum spanning the entire visible range.

### 7.2 Optical Solitons

In an optical fiber with anomalous dispersion ($D > 0$ at 1550 nm), the interplay between SPM (which broadens the spectrum) and dispersion (which compresses red-shifted leading edges and blue-shifted trailing edges) can balance perfectly. The result is a **soliton** — a pulse that propagates without changing its shape:

$$A(z, t) = A_0\,\text{sech}\!\left(\frac{t}{\tau_0}\right)e^{i\gamma P_0 z/2}$$

The soliton condition requires a specific relationship between peak power $P_0$ and pulse width $\tau_0$:

$$P_0 = \frac{|{\beta_2}|}{\gamma \tau_0^2}$$

where $\beta_2$ is the group velocity dispersion and $\gamma = n_2\omega/(cA_{\text{eff}})$ is the nonlinear parameter.

Solitons were proposed for long-distance fiber communication (soliton transmission) but have been largely superseded by coherent detection with digital signal processing.

---

## 8. Optical Parametric Processes

### 8.1 Optical Parametric Amplification (OPA)

In DFG, the signal wave at $\omega_1$ is amplified while a new idler wave at $\omega_2 = \omega_3 - \omega_1$ is generated. The signal gain is:

$$G = 1 + \left(\frac{\Gamma}{\kappa}\right)^2\sinh^2(\kappa L)$$

where $\Gamma \propto \sqrt{I_{\text{pump}}}\,\chi^{(2)}$ is the parametric gain coefficient and $\kappa = \sqrt{\Gamma^2 - (\Delta k/2)^2}$.

At perfect phase matching: $G \approx \frac{1}{4}e^{2\Gamma L}$ — exponential gain, similar to a laser amplifier.

### 8.2 Optical Parametric Oscillator (OPO)

Place the parametric gain medium inside an optical cavity:

```
Mirror ──── χ⁽²⁾ crystal ──── Mirror
   ↑ pump (ω₃)
   ← signal (ω₁) resonates in cavity
   → idler (ω₂) may also resonate
```

When the parametric gain exceeds the cavity losses, the OPO oscillates — generating signal and idler beams from a single pump. The output frequencies are determined by phase matching and cavity resonance.

### 8.3 Tunability

The great advantage of OPOs: **broad tunability**. By changing the crystal angle, temperature, or poling period, the signal and idler wavelengths can be continuously tuned across a very wide range.

Example: A PPLN OPO pumped at 1064 nm can produce signal output from 1.4-4.5 $\mu$m, covering the entire mid-infrared — a spectral range critical for molecular spectroscopy, gas sensing, and defense applications.

### 8.4 Spontaneous Parametric Down-Conversion (SPDC)

When no signal wave is present, quantum vacuum fluctuations at $\omega_1$ can seed the parametric process, causing spontaneous splitting of pump photons:

$$\omega_{\text{pump}} \to \omega_{\text{signal}} + \omega_{\text{idler}}$$

The signal and idler photons are created simultaneously and are **quantum-mechanically entangled** in energy, momentum, polarization, and time. SPDC is the most widely used source of entangled photon pairs for quantum optics experiments (see Lesson 13).

---

## 9. Applications

### 9.1 Frequency Conversion

- **Green laser pointers**: Nd:YVO$_4$ (1064 nm) → KTP SHG → 532 nm green
- **UV lasers**: Multiple SHG stages: 1064 → 532 → 266 nm (fourth harmonic)
- **Deep UV**: SFG and SHG in BBO crystals for lithography, spectroscopy
- **Mid-IR generation**: DFG and OPO for molecular spectroscopy ($3\text{-}20\,\mu\text{m}$)

### 9.2 Ultrafast Pulse Technology

- **SPM spectral broadening** + chirped mirror compression: few-femtosecond pulses
- **Optical parametric chirped-pulse amplification (OPCPA)**: Extremely high peak powers (PW-class lasers)
- **Frequency combs**: Mode-locked laser + SPM broadening → phase-stabilized comb (Nobel Prize 2005, Hall & Hansch)

### 9.3 Quantum Optics

- **Entangled photon pairs** via SPDC: foundation of quantum communication and quantum computing experiments
- **Squeezed light** via parametric processes: used in gravitational wave detectors (LIGO)
- **Heralded single photon sources**: detecting one photon of an SPDC pair "heralds" the presence of its partner

### 9.4 Telecommunications

- **Wavelength conversion** in fiber networks using FWM or DFG
- **All-optical switching** using the Kerr effect
- **Supercontinuum sources** for WDM channel testing and spectroscopy

---

## 10. Python Examples

### 10.1 Second Harmonic Generation with Phase Matching

```python
import numpy as np
import matplotlib.pyplot as plt

def shg_coupled_equations(L, N, d_eff, wavelength, n_omega, n_2omega,
                           I_pump, delta_k=0):
    """
    Solve the coupled wave equations for SHG.

    We integrate the pair of ODEs that describe energy exchange between
    the fundamental (ω) and second harmonic (2ω) fields. The phase
    mismatch Δk controls whether the exchange is constructive (Δk=0)
    or oscillatory (Δk≠0). This is a direct numerical demonstration
    of why phase matching is so critical.
    """
    c = 3e8
    eps_0 = 8.854e-12
    omega = 2 * np.pi * c / wavelength

    # Convert intensity to field amplitude: I = n*eps_0*c*|E|^2 / 2
    E_pump = np.sqrt(2 * I_pump / (n_omega * eps_0 * c))

    # Coupling coefficients
    kappa_1 = d_eff * omega / (n_2omega * c)  # for SH growth
    kappa_2 = d_eff * omega / (n_omega * c)   # for fundamental depletion

    # Initialize fields
    dz = L / N
    z = np.linspace(0, L, N + 1)
    A_omega = np.zeros(N + 1, dtype=complex)
    A_2omega = np.zeros(N + 1, dtype=complex)
    A_omega[0] = E_pump
    A_2omega[0] = 0.0

    # Runge-Kutta 4th order integration
    for i in range(N):
        zi = z[i]
        Aw = A_omega[i]
        A2w = A_2omega[i]

        def dA2w_dz(Aw_val, zi_val):
            return -1j * kappa_1 * Aw_val**2 * np.exp(1j * delta_k * zi_val)

        def dAw_dz(Aw_val, A2w_val, zi_val):
            return -1j * kappa_2 * A2w_val * np.conj(Aw_val) * np.exp(-1j * delta_k * zi_val)

        # RK4 for coupled system
        k1_2w = dz * dA2w_dz(Aw, zi)
        k1_w = dz * dAw_dz(Aw, A2w, zi)

        k2_2w = dz * dA2w_dz(Aw + k1_w/2, zi + dz/2)
        k2_w = dz * dAw_dz(Aw + k1_w/2, A2w + k1_2w/2, zi + dz/2)

        k3_2w = dz * dA2w_dz(Aw + k2_w/2, zi + dz/2)
        k3_w = dz * dAw_dz(Aw + k2_w/2, A2w + k2_2w/2, zi + dz/2)

        k4_2w = dz * dA2w_dz(Aw + k3_w, zi + dz)
        k4_w = dz * dAw_dz(Aw + k3_w, A2w + k3_2w, zi + dz)

        A_2omega[i+1] = A2w + (k1_2w + 2*k2_2w + 2*k3_2w + k4_2w) / 6
        A_omega[i+1] = Aw + (k1_w + 2*k2_w + 2*k3_w + k4_w) / 6

    # Convert to intensities
    I_omega = 0.5 * n_omega * eps_0 * c * np.abs(A_omega)**2
    I_2omega = 0.5 * n_2omega * eps_0 * c * np.abs(A_2omega)**2

    return z, I_omega, I_2omega

# Parameters for KTP crystal, SHG of 1064 nm → 532 nm
wavelength = 1064e-9
d_eff = 3.18e-12  # Effective nonlinear coefficient for KTP (m/V)
n_omega = 1.740    # Refractive index at 1064 nm
n_2omega = 1.779   # Refractive index at 532 nm
L = 0.02           # 20 mm crystal length
I_pump = 1e12      # 1 GW/m² (moderately focused pulsed laser)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Left: Phase-matched vs. mismatched ---
for dk_label, dk_val in [('Δk = 0 (perfect PM)', 0),
                          ('Δk = 100 /m', 100),
                          ('Δk = 1000 /m', 1000)]:
    z, I_w, I_2w = shg_coupled_equations(L, 5000, d_eff, wavelength,
                                          n_omega, n_2omega, I_pump, dk_val)
    efficiency = I_2w / I_pump
    axes[0].plot(z * 1e3, efficiency * 100, linewidth=2, label=dk_label)

axes[0].set_xlabel('Crystal length (mm)', fontsize=11)
axes[0].set_ylabel('SHG conversion efficiency (%)', fontsize=11)
axes[0].set_title('SHG Efficiency: Phase Matching Matters', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# --- Right: Quasi-phase matching ---
# Demonstrate the sinc² dependence on Δk*L
dk_range = np.linspace(-2000, 2000, 1000)
L_crystal = 0.01  # 10 mm

# Undepleted pump approximation: η ∝ sinc²(ΔkL/2)
eta_approx = np.sinc(dk_range * L_crystal / (2 * np.pi))**2

axes[1].plot(dk_range, eta_approx, 'b-', linewidth=2)
axes[1].set_xlabel('Phase mismatch Δk (1/m)', fontsize=11)
axes[1].set_ylabel('Normalized efficiency (sinc²)', fontsize=11)
axes[1].set_title(f'SHG Efficiency vs. Phase Mismatch (L = {L_crystal*1e3:.0f} mm)',
                   fontsize=13)
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Perfect PM')
axes[1].legend()

plt.tight_layout()
plt.savefig('shg_phase_matching.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 10.2 Self-Phase Modulation

```python
import numpy as np
import matplotlib.pyplot as plt

def self_phase_modulation(t, pulse, n2, omega0, L, I_peak):
    """
    Compute self-phase modulation of an optical pulse.

    SPM arises because the Kerr effect makes the refractive index
    intensity-dependent: n = n0 + n2*I. A pulse with time-varying
    intensity acquires a time-varying phase, which generates new
    frequency components and broadens the spectrum — without
    changing the pulse shape in time (when dispersion is negligible).
    """
    c = 3e8
    # Normalize pulse intensity
    I_t = I_peak * np.abs(pulse)**2 / np.max(np.abs(pulse)**2)

    # Nonlinear phase
    phi_nl = n2 * I_t * omega0 * L / c

    # Apply SPM
    pulse_spm = pulse * np.exp(1j * phi_nl)

    # Instantaneous frequency shift
    dt = t[1] - t[0]
    delta_omega = -np.gradient(phi_nl, dt)

    return pulse_spm, phi_nl, delta_omega

# Gaussian pulse parameters
c = 3e8
wavelength = 800e-9  # Ti:Sapphire
omega0 = 2 * np.pi * c / wavelength
tau_fwhm = 50e-15  # 50 fs pulse
tau = tau_fwhm / (2 * np.sqrt(np.log(2)))  # Gaussian 1/e width

# Time grid
t = np.linspace(-200e-15, 200e-15, 4096)
pulse_in = np.exp(-t**2 / (2 * tau**2))

# Material parameters
n2 = 2.6e-20  # Silica glass
I_peak = 5e13  # W/m² (moderately intense)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Different propagation lengths to show SPM evolution
lengths = [0.001, 0.005, 0.01, 0.02]  # meters
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(lengths)))

for L, color in zip(lengths, colors):
    pulse_out, phi_nl, delta_omega = self_phase_modulation(
        t, pulse_in, n2, omega0, L, I_peak
    )

    max_phi = np.max(phi_nl)
    label = f'L = {L*1e3:.0f} mm (φ_max = {max_phi:.1f} rad)'

    # Temporal intensity (unchanged by SPM alone)
    axes[0, 0].plot(t * 1e15, np.abs(pulse_out)**2, color=color,
                     linewidth=1.5, label=label)

    # Nonlinear phase
    axes[0, 1].plot(t * 1e15, phi_nl, color=color, linewidth=1.5)

    # Spectrum (broadened by SPM)
    spectrum_out = np.fft.fftshift(np.abs(np.fft.fft(pulse_out))**2)
    freq = np.fft.fftshift(np.fft.fftfreq(len(t), t[1] - t[0]))
    # Convert to wavelength-like relative frequency
    axes[1, 0].plot(freq * 1e-12, spectrum_out / spectrum_out.max(),
                     color=color, linewidth=1.5, label=label)

    # Chirp (instantaneous frequency)
    axes[1, 1].plot(t * 1e15, delta_omega * 1e-12, color=color, linewidth=1.5)

axes[0, 0].set_xlabel('Time (fs)')
axes[0, 0].set_ylabel('|E|² (normalized)')
axes[0, 0].set_title('Temporal Intensity')
axes[0, 0].legend(fontsize=8)

axes[0, 1].set_xlabel('Time (fs)')
axes[0, 1].set_ylabel('Nonlinear phase (rad)')
axes[0, 1].set_title('SPM Phase φ_NL(t)')

axes[1, 0].set_xlabel('Frequency (THz)')
axes[1, 0].set_ylabel('Spectral intensity')
axes[1, 0].set_title('Spectrum (broadened by SPM)')
axes[1, 0].set_xlim(-30, 30)

axes[1, 1].set_xlabel('Time (fs)')
axes[1, 1].set_ylabel('Δω (THz)')
axes[1, 1].set_title('Chirp (Instantaneous Frequency Shift)')

for ax in axes.flat:
    ax.grid(True, alpha=0.3)

plt.suptitle('Self-Phase Modulation of a 50 fs Pulse', fontsize=14)
plt.tight_layout()
plt.savefig('self_phase_modulation.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 10.3 Quasi-Phase Matching Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

def qpm_comparison(L, coherence_length, N_points=10000):
    """
    Compare SHG growth for perfect PM, no PM, and quasi-PM.

    Perfect phase matching: SH grows quadratically with z.
    No phase matching: SH oscillates between 0 and a small maximum
    with period 2*L_c — energy sloshes back and forth.
    QPM: periodically flipping χ⁽²⁾ with period 2*L_c prevents
    the back-conversion, giving steady (though slower) growth.
    """
    z = np.linspace(0, L, N_points)
    dz = z[1] - z[0]

    # Perfect phase matching: Δk = 0, SH grows linearly with z
    # (undepleted pump: A_2w ∝ z, I_2w ∝ z²)
    shg_perfect = (z / L)**2

    # Phase mismatched: oscillates with coherence length
    dk = np.pi / coherence_length  # Δk = π/L_c
    shg_mismatched = np.sin(dk * z / 2)**2 / (dk * L / 2)**2

    # Quasi-phase matched: flip χ⁽²⁾ every L_c
    # The SH field grows in a staircase pattern
    period = 2 * coherence_length
    # Effective Δk after QPM: reduced by factor (2/π)
    # SH grows as (2z/(πL))² compared to (z/L)² for perfect PM
    shg_qpm = (2 * z / (np.pi * L))**2

    # For detailed visualization: actual staircase growth
    shg_qpm_exact = np.zeros_like(z)
    A_2w = 0.0  # SH field amplitude
    for i in range(1, len(z)):
        # Determine sign of χ⁽²⁾ based on QPM period
        domain = int(z[i] / coherence_length) % 2
        sign = 1.0 if domain == 0 else -1.0

        # Growth with phase mismatch, but χ⁽²⁾ flips to compensate
        A_2w += sign * np.exp(1j * dk * z[i]) * dz
        shg_qpm_exact[i] = np.abs(A_2w)**2

    # Normalize
    shg_qpm_exact /= shg_qpm_exact[-1] if shg_qpm_exact[-1] > 0 else 1

    return z, shg_perfect, shg_mismatched, shg_qpm, shg_qpm_exact

L = 10e-3  # 10 mm crystal
L_c = 0.5e-3  # 0.5 mm coherence length

z, perfect, mismatched, qpm_approx, qpm_exact = qpm_comparison(L, L_c, 50000)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(z * 1e3, perfect, 'g-', linewidth=2, label='Perfect phase matching')
ax.plot(z * 1e3, qpm_exact * (2/np.pi)**2, 'b-', linewidth=1.5,
        label=f'Quasi-phase matching (Λ = {2*L_c*1e3:.1f} mm)')
ax.plot(z * 1e3, mismatched * (L_c/L)**2 * 4, 'r-', linewidth=1.5,
        label='No phase matching (oscillating)')

# Mark coherence length
for i in range(int(L / L_c)):
    if i < 3:  # Only mark first few
        ax.axvline(x=(i + 0.5) * 2 * L_c * 1e3, color='gray',
                   linestyle=':', alpha=0.3)

ax.set_xlabel('Crystal length (mm)', fontsize=12)
ax.set_ylabel('SHG intensity (normalized)', fontsize=12)
ax.set_title('Comparison: Perfect PM vs. QPM vs. No PM', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('qpm_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 11. Summary

| Concept | Key Formula / Idea |
|---------|--------------------|
| Nonlinear polarization | $P = \epsilon_0(\chi^{(1)}E + \chi^{(2)}E^2 + \chi^{(3)}E^3 + \cdots)$ |
| $\chi^{(2)}$ symmetry | Vanishes in centrosymmetric media |
| SHG | $\omega + \omega \to 2\omega$; $I_{2\omega} \propto \chi^{(2)2}I_\omega^2 L^2\text{sinc}^2(\Delta kL/2)$ |
| Phase matching | $\Delta k = k_{2\omega} - 2k_\omega = 0$; momentum conservation |
| Birefringent PM | $n_e(2\omega, \theta) = n_o(\omega)$ |
| Quasi-PM | Periodically poled crystal; $\Lambda = 2L_c$ |
| Coherence length | $L_c = \pi/\|\Delta k\|$ |
| Kerr effect | $n = n_0 + n_2 I$ |
| Self-phase modulation | Intensity-dependent phase → spectral broadening |
| Self-focusing | $P > P_{\text{cr}} = 3.77\lambda^2/(8\pi n_0 n_2)$ → beam collapse |
| FWM | $\omega_4 = \omega_1 + \omega_2 - \omega_3$ (four-wave mixing) |
| OPO | $\chi^{(2)}$ parametric oscillator; broadly tunable |
| SPDC | Spontaneous photon pair generation; entangled photons |

---

## 12. Exercises

### Exercise 1: SHG Efficiency

A KTP crystal ($d_{\text{eff}} = 3.18\,\text{pm/V}$, $n_\omega = 1.740$, $n_{2\omega} = 1.779$) is phase-matched for SHG of 1064 nm light.

(a) Calculate the coherence length without phase matching.
(b) For a crystal length of 10 mm and pump intensity $I = 100\,\text{MW/cm}^2$, estimate the SHG efficiency in the undepleted pump approximation.
(c) At what pump intensity does the undepleted pump approximation break down (say, when efficiency exceeds 10%)?

### Exercise 2: QPM Design

Design a PPLN (periodically poled lithium niobate) crystal for SHG of 1550 nm → 775 nm at room temperature. Given: $n(1550\,\text{nm}) = 2.211$, $n(775\,\text{nm}) = 2.259$.

(a) Calculate the phase mismatch $\Delta k$ without QPM.
(b) Calculate the required QPM period $\Lambda$.
(c) If $d_{33} = 27\,\text{pm/V}$, how does the effective $d_{\text{eff}}$ for first-order QPM compare?
(d) How long must the crystal be for 50% conversion efficiency at $I = 1\,\text{GW/m}^2$?

### Exercise 3: Self-Phase Modulation

A 100 fs pulse at 800 nm with peak power 1 MW propagates through 1 cm of fused silica ($n_2 = 2.6 \times 10^{-20}\,\text{m}^2/\text{W}$, beam area = 100 $\mu$m$^2$).

(a) Calculate the peak intensity and the maximum nonlinear phase $\phi_{\text{NL,max}}$.
(b) Estimate the spectral broadening factor.
(c) Is self-focusing a concern? Calculate $P/P_{\text{cr}}$.
(d) Modify the Python SPM code to visualize this case.

### Exercise 4: Phase Matching Angles

For Type I SHG in BBO crystal at 800 nm ($n_o(\text{800}) = 1.6609$, $n_e(\text{800}) = 1.5426$, $n_o(\text{400}) = 1.6924$, $n_e(\text{400}) = 1.5667$):

(a) Calculate the phase matching angle $\theta_{\text{PM}}$ where $n_e(400, \theta) = n_o(800)$.
(b) Calculate the angular acceptance bandwidth $\Delta\theta$.
(c) Why is BBO preferred over KTP for ultrashort pulse SHG?

---

## 13. References

1. Boyd, R. W. (2020). *Nonlinear Optics* (4th ed.). Academic Press. — The standard reference.
2. Saleh, B. E. A., & Teich, M. C. (2019). *Fundamentals of Photonics* (3rd ed.). Wiley. — Chapter 21.
3. Shen, Y. R. (2002). *The Principles of Nonlinear Optics*. Wiley Classics.
4. Agrawal, G. P. (2019). *Nonlinear Fiber Optics* (6th ed.). Academic Press.
5. Franken, P. A., et al. (1961). "Generation of optical harmonics." *Physical Review Letters*, 7, 118.
6. Armstrong, J. A., et al. (1962). "Interactions between light waves in a nonlinear dielectric." *Physical Review*, 127, 1918.

---

[← Previous: 11. Holography](11_Holography.md) | [Next: 13. Quantum Optics Primer →](13_Quantum_Optics_Primer.md)
