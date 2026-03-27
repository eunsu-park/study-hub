# 01. Nature of Light

[Next: 02. Geometric Optics Fundamentals →](02_Geometric_Optics_Fundamentals.md)

---

## Learning Objectives

1. Describe the wave-particle duality of light and explain how both models complement each other
2. Relate the electromagnetic spectrum to wavelength, frequency, and photon energy
3. Calculate the speed of light in various media using the refractive index
4. Explain dispersion and its physical origin in terms of frequency-dependent refractive index
5. Trace the historical development of our understanding of light from Newton through Einstein
6. Apply the photon energy relation $E = h\nu$ to practical problems in spectroscopy and photonics
7. Distinguish between phase velocity, group velocity, and signal velocity in dispersive media

---

## Why This Matters

Light is the primary messenger of the universe. Nearly everything we know about distant stars, molecular structures, and the quantum world comes from analyzing light. Understanding the nature of light is the gateway to optics, photonics, telecommunications, medical imaging, and quantum computing. Whether you are designing a camera lens, building a fiber-optic network, or interpreting an astronomical spectrum, it all begins here — with the question that puzzled humanity for millennia: *What is light?*

> **Analogy**: Think of light as a coin with two faces. One face — the *wave* face — explains interference patterns, diffraction, and polarization. The other face — the *particle* face — explains the photoelectric effect and photon counting. You can never see both faces at the same time, but both are equally real. This complementarity is not a limitation of our understanding; it is a fundamental feature of nature.

---

## 1. Historical Development

### 1.1 The Corpuscular Theory (Newton, 1704)

Isaac Newton proposed that light consists of tiny particles ("corpuscles") that travel in straight lines. This explained reflection (particles bouncing off surfaces) and refraction (particles accelerating into denser media). Newton's authority made the corpuscular theory dominant for over a century.

**Strengths**: Explained rectilinear propagation, sharp shadows, reflection.
**Weaknesses**: Could not explain interference or diffraction patterns.

### 1.2 The Wave Theory (Huygens, 1690; Young, 1801; Fresnel, 1818)

Christiaan Huygens proposed that light is a wave propagating through a hypothetical medium called the "luminiferous aether." Thomas Young's double-slit experiment (1801) provided dramatic evidence for wave behavior, and Augustin-Jean Fresnel developed a rigorous mathematical wave theory that explained diffraction.

**Key prediction**: Waves should travel *slower* in denser media (opposite to Newton's prediction). Foucault's measurement (1850) confirmed this, decisively favoring the wave theory.

### 1.3 Electromagnetic Theory (Maxwell, 1865)

James Clerk Maxwell unified electricity and magnetism into four equations and showed that electromagnetic waves propagate at speed:

$$c = \frac{1}{\sqrt{\mu_0 \epsilon_0}} \approx 3 \times 10^8 \text{ m/s}$$

This matched the measured speed of light, leading Maxwell to conclude: *"Light is an electromagnetic disturbance."* Heinrich Hertz confirmed electromagnetic waves experimentally in 1887.

### 1.4 The Quantum Revolution (Planck, 1900; Einstein, 1905)

Max Planck resolved the ultraviolet catastrophe by proposing that energy is emitted in discrete quanta: $E = h\nu$. Albert Einstein extended this to light itself in 1905, explaining the photoelectric effect by treating light as a stream of photons — each carrying energy $E = h\nu$ and momentum $p = h/\lambda$.

### 1.5 Modern Synthesis: Quantum Electrodynamics (QED)

Richard Feynman, Julian Schwinger, and Sin-Itiro Tomonaga developed QED in the 1940s-50s, providing the most complete description of light-matter interactions. In QED, photons are excitations of the quantized electromagnetic field. The theory agrees with experiment to extraordinary precision (better than 1 part in $10^{12}$).

---

## 2. Light as an Electromagnetic Wave

### 2.1 Maxwell's Equations and Wave Solutions

In free space, Maxwell's equations yield the wave equation:

$$\nabla^2 \mathbf{E} = \mu_0 \epsilon_0 \frac{\partial^2 \mathbf{E}}{\partial t^2}$$

A monochromatic plane wave solution takes the form:

$$\mathbf{E}(\mathbf{r}, t) = \mathbf{E}_0 \cos(\mathbf{k} \cdot \mathbf{r} - \omega t + \phi)$$

where:
- $\mathbf{E}_0$ is the amplitude vector (determines polarization direction)
- $\mathbf{k}$ is the wave vector ($|\mathbf{k}| = 2\pi/\lambda$, points in the propagation direction)
- $\omega = 2\pi\nu$ is the angular frequency
- $\phi$ is the initial phase

The magnetic field $\mathbf{B}$ is perpendicular to $\mathbf{E}$ and to $\mathbf{k}$:

$$\mathbf{B} = \frac{1}{c} \hat{\mathbf{k}} \times \mathbf{E}$$

### 2.2 The Electromagnetic Spectrum

The electromagnetic spectrum spans an enormous range of wavelengths:

| Region | Wavelength | Frequency | Photon Energy |
|--------|-----------|-----------|---------------|
| Radio | > 1 m | < 300 MHz | < 1.24 $\mu$eV |
| Microwave | 1 mm – 1 m | 300 MHz – 300 GHz | 1.24 $\mu$eV – 1.24 meV |
| Infrared | 700 nm – 1 mm | 300 GHz – 430 THz | 1.24 meV – 1.77 eV |
| **Visible** | **400 – 700 nm** | **430 – 750 THz** | **1.77 – 3.10 eV** |
| Ultraviolet | 10 – 400 nm | 750 THz – 30 PHz | 3.10 – 124 eV |
| X-ray | 0.01 – 10 nm | 30 PHz – 30 EHz | 124 eV – 124 keV |
| Gamma ray | < 0.01 nm | > 30 EHz | > 124 keV |

The visible spectrum — the narrow band our eyes can detect — is a tiny sliver of this range:

| Color | Wavelength (nm) | Frequency (THz) |
|-------|-----------------|-----------------|
| Red | 620 – 700 | 430 – 484 |
| Orange | 590 – 620 | 484 – 508 |
| Yellow | 570 – 590 | 508 – 526 |
| Green | 495 – 570 | 526 – 606 |
| Blue | 450 – 495 | 606 – 668 |
| Violet | 380 – 450 | 668 – 789 |

### 2.3 Fundamental Relations

The wavelength $\lambda$, frequency $\nu$, and speed $c$ are related by:

$$c = \lambda \nu$$

In vacuum, $c = 299\,792\,458$ m/s (exact, by definition of the meter since 1983).

The wave number $k$ and angular frequency $\omega$:

$$k = \frac{2\pi}{\lambda}, \qquad \omega = 2\pi\nu, \qquad c = \frac{\omega}{k}$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize an electromagnetic plane wave propagating in the z-direction
# E oscillates in the x-direction, B oscillates in the y-direction

z = np.linspace(0, 4 * np.pi, 500)  # spatial coordinate (in units of wavelength/2pi)
t = 0  # snapshot at t = 0

# Normalized fields: E_x and B_y for a plane wave
E_x = np.sin(z)       # electric field component
B_y = np.sin(z)       # magnetic field component (in phase, perpendicular)

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111, projection='3d')

# Plot E-field (red) oscillating in the x-z plane
ax.plot(z, E_x, np.zeros_like(z), color='red', linewidth=2, label='E-field (x)')
# Plot B-field (blue) oscillating in the y-z plane
ax.plot(z, np.zeros_like(z), B_y, color='blue', linewidth=2, label='B-field (y)')

ax.set_xlabel('z (propagation)')
ax.set_ylabel('E_x')
ax.set_zlabel('B_y')
ax.set_title('Electromagnetic Plane Wave')
ax.legend()
plt.tight_layout()
plt.savefig('em_plane_wave.png', dpi=150)
plt.show()
```

---

## 3. Light as Particles: Photons

### 3.1 Photon Energy

Each photon carries energy:

$$E = h\nu = \frac{hc}{\lambda}$$

where $h = 6.626 \times 10^{-34}$ J$\cdot$s is Planck's constant.

A convenient form for calculations:

$$E \text{ (eV)} = \frac{1240}{\lambda \text{ (nm)}}$$

This means a green photon ($\lambda = 550$ nm) carries about 2.25 eV of energy — enough to trigger photochemical reactions in your retina but not enough to ionize most atoms.

### 3.2 Photon Momentum

Despite being massless, photons carry momentum:

$$p = \frac{E}{c} = \frac{h}{\lambda} = \frac{h\nu}{c}$$

This is the basis for **radiation pressure** — sunlight exerts about 4.6 $\mu$Pa on a perfectly absorbing surface. Solar sails for spacecraft exploit this tiny but persistent force.

### 3.3 The Photoelectric Effect

Einstein's explanation of the photoelectric effect (1905) was the first direct evidence for photons:

$$E_k = h\nu - \phi$$

where $E_k$ is the maximum kinetic energy of ejected electrons and $\phi$ is the work function of the material. Key observations:
- Below a threshold frequency $\nu_0 = \phi/h$, no electrons are emitted regardless of intensity
- Above the threshold, electron energy increases linearly with frequency
- Emission is instantaneous — no time delay, even at very low intensities

```python
import numpy as np
import matplotlib.pyplot as plt

# Photoelectric effect: kinetic energy vs. photon frequency
# Demonstrates that E_k depends linearly on frequency, not intensity

h = 6.626e-34         # Planck's constant (J·s)
eV = 1.602e-19        # electron-volt to Joules conversion

# Work functions for common metals (in eV)
metals = {
    'Cesium (Cs)': 2.1,
    'Sodium (Na)': 2.28,
    'Zinc (Zn)': 4.33,
    'Platinum (Pt)': 5.64,
}

# Frequency range: 0 to 2000 THz
nu = np.linspace(0, 2000e12, 500)
E_photon_eV = h * nu / eV  # photon energy in eV

fig, ax = plt.subplots(figsize=(10, 6))

for metal, phi in metals.items():
    # Kinetic energy is max(0, E_photon - phi)
    # We only plot the region where E_k > 0 (above threshold)
    E_k = E_photon_eV - phi
    mask = E_k > 0
    ax.plot(nu[mask] / 1e12, E_k[mask], linewidth=2, label=f'{metal}, $\\phi$ = {phi} eV')
    # Mark the threshold frequency with a vertical dotted line
    nu_threshold = phi * eV / h
    ax.axvline(nu_threshold / 1e12, linestyle=':', alpha=0.5)

ax.set_xlabel('Frequency (THz)', fontsize=12)
ax.set_ylabel('Max Kinetic Energy (eV)', fontsize=12)
ax.set_title('Photoelectric Effect: $E_k = h\\nu - \\phi$', fontsize=14)
ax.legend(fontsize=10)
ax.set_xlim(0, 2000)
ax.set_ylim(0, 6)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('photoelectric_effect.png', dpi=150)
plt.show()
```

### 3.4 Wave-Particle Duality: The de Broglie Relation

Louis de Broglie (1924) proposed that *all* matter has wave-like properties:

$$\lambda = \frac{h}{p} = \frac{h}{mv}$$

For photons this is consistent with $p = h/\lambda$. For electrons, neutrons, and even large molecules, matter-wave interference has been experimentally confirmed. The complementarity principle (Niels Bohr) states that the wave and particle descriptions are complementary — the experimental setup determines which aspect is revealed.

---

## 4. Speed of Light in Media

### 4.1 Refractive Index

When light enters a material, it interacts with the electrons in the medium. The speed of light in a medium is:

$$v = \frac{c}{n}$$

where $n$ is the **refractive index** of the medium. Since $v \leq c$ for normal materials, we have $n \geq 1$.

| Material | Refractive Index $n$ |
|----------|---------------------|
| Vacuum | 1 (exact) |
| Air (STP) | 1.000293 |
| Water | 1.333 |
| Glass (crown) | 1.52 |
| Glass (flint) | 1.62 |
| Diamond | 2.417 |
| Silicon | 3.48 (at 1550 nm) |

> **Analogy**: Imagine a marching band crossing from pavement (fast medium) onto a muddy field (slow medium) at an angle. The side that hits the mud first slows down, causing the entire line to pivot — this is refraction. The "refractive index" is like the ratio of walking speed on pavement to walking speed in mud.

### 4.2 Microscopic Origin of the Refractive Index

At the atomic level, the refractive index arises from the polarization of bound electrons by the electromagnetic field. The Lorentz oscillator model treats each electron as a damped harmonic oscillator driven by the electric field:

$$n^2(\omega) = 1 + \frac{Nq^2}{\epsilon_0 m_e} \sum_j \frac{f_j}{\omega_{0j}^2 - \omega^2 - i\gamma_j \omega}$$

where $f_j$ are oscillator strengths, $\omega_{0j}$ are resonance frequencies, and $\gamma_j$ are damping constants. This model captures:
- **Normal dispersion**: $n$ increases with $\omega$ (away from resonances)
- **Anomalous dispersion**: $n$ decreases with $\omega$ (near resonances)
- **Absorption**: The imaginary part of $n$ gives the absorption coefficient

### 4.3 Phase Velocity and Group Velocity

In a dispersive medium, different frequency components travel at different speeds:

- **Phase velocity**: $v_p = \frac{\omega}{k} = \frac{c}{n(\omega)}$ — speed of a single frequency component
- **Group velocity**: $v_g = \frac{d\omega}{dk}$ — speed of the envelope of a wave packet

These are related by:

$$v_g = v_p - \lambda \frac{dv_p}{d\lambda} = \frac{c}{n + \omega \frac{dn}{d\omega}}$$

The group velocity is typically the speed at which energy and information propagate.

```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstrate the difference between phase velocity and group velocity
# by showing a wave packet in a dispersive medium

x = np.linspace(0, 100, 2000)  # spatial coordinate

# Central wave number and frequency
k0 = 1.0      # central wave number
omega0 = 1.0   # central frequency (v_p = omega/k = 1.0)

# Dispersion: omega = omega0 + v_g * (k - k0) + 0.5 * beta2 * (k - k0)^2
# For demonstration: v_p = 1.0, v_g = 0.6 (group slower than phase)
v_p = 1.0
v_g = 0.6

# Build a Gaussian wave packet at t=0 and t=30
sigma = 5.0  # width of the envelope
times = [0, 30]

fig, axes = plt.subplots(len(times), 1, figsize=(12, 6), sharex=True)

for ax, t in zip(axes, times):
    # Envelope moves at group velocity
    envelope = np.exp(-((x - 50 - v_g * t) ** 2) / (2 * sigma ** 2))
    # Carrier wave moves at phase velocity
    carrier = np.cos(k0 * (x - v_p * t))
    # The wave packet is the product of envelope and carrier
    wave_packet = envelope * carrier

    ax.plot(x, wave_packet, 'b-', linewidth=1, label='Wave packet')
    ax.plot(x, envelope, 'r--', linewidth=1.5, label='Envelope ($v_g$)')
    ax.plot(x, -envelope, 'r--', linewidth=1.5)
    ax.set_ylabel('Amplitude')
    ax.set_title(f't = {t} (phase moves at $v_p$ = {v_p}, envelope at $v_g$ = {v_g})')
    ax.legend(loc='upper right')
    ax.set_ylim(-1.3, 1.3)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Position x')
plt.tight_layout()
plt.savefig('phase_vs_group_velocity.png', dpi=150)
plt.show()
```

---

## 5. Dispersion

### 5.1 Cauchy's Equation and Sellmeier Equation

For transparent materials far from absorption resonances, the refractive index varies with wavelength. Two empirical models:

**Cauchy's equation** (approximate):

$$n(\lambda) = A + \frac{B}{\lambda^2} + \frac{C}{\lambda^4} + \cdots$$

**Sellmeier equation** (more accurate, physics-based):

$$n^2(\lambda) = 1 + \sum_i \frac{B_i \lambda^2}{\lambda^2 - C_i}$$

where $B_i$ and $C_i$ are empirically determined constants related to the resonance frequencies and oscillator strengths.

### 5.2 Chromatic Dispersion and Prisms

A prism separates white light into its constituent colors because the refractive index varies with wavelength. Violet light ($n$ larger) is bent more than red light ($n$ smaller).

The **angular dispersion** of a prism is:

$$\frac{d\theta}{d\lambda} = \frac{d\theta}{dn} \cdot \frac{dn}{d\lambda}$$

The **Abbe number** $V_d$ characterizes the dispersion of an optical glass:

$$V_d = \frac{n_d - 1}{n_F - n_C}$$

where $n_d$, $n_F$, $n_C$ are refractive indices at specific wavelengths (587.6 nm, 486.1 nm, 656.3 nm). A high Abbe number means low dispersion.

```python
import numpy as np
import matplotlib.pyplot as plt

# Sellmeier equation for BK7 glass (common optical glass)
# Demonstrates normal dispersion: n increases as wavelength decreases

def sellmeier_bk7(wavelength_um):
    """
    Compute refractive index of BK7 glass using the Sellmeier equation.
    wavelength_um: wavelength in micrometers
    Returns: refractive index n
    """
    # Sellmeier coefficients for Schott BK7 (sourced from Schott catalog)
    B1, B2, B3 = 1.03961212, 0.231792344, 1.01046945
    C1, C2, C3 = 0.00600069867, 0.0200179144, 103.560653  # in um^2

    lam2 = wavelength_um ** 2
    n_sq = 1 + (B1 * lam2 / (lam2 - C1)
                + B2 * lam2 / (lam2 - C2)
                + B3 * lam2 / (lam2 - C3))
    return np.sqrt(n_sq)

# Wavelength range: 300 nm to 2000 nm
wavelengths_nm = np.linspace(300, 2000, 500)
wavelengths_um = wavelengths_nm / 1000.0  # convert to micrometers

n_bk7 = sellmeier_bk7(wavelengths_um)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: n vs wavelength — shows normal dispersion (n decreases with wavelength)
ax1.plot(wavelengths_nm, n_bk7, 'b-', linewidth=2)
ax1.set_xlabel('Wavelength (nm)', fontsize=12)
ax1.set_ylabel('Refractive Index n', fontsize=12)
ax1.set_title('BK7 Glass: Sellmeier Dispersion Curve', fontsize=13)
ax1.axvspan(380, 700, alpha=0.15, color='yellow', label='Visible range')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Right: group index n_g = n - lambda * dn/dlambda
# Group index determines the group velocity: v_g = c / n_g
dn_dlam = np.gradient(n_bk7, wavelengths_um)  # dn/d(lambda in um)
n_group = n_bk7 - wavelengths_um * dn_dlam

ax2.plot(wavelengths_nm, n_group, 'r-', linewidth=2, label='Group index $n_g$')
ax2.plot(wavelengths_nm, n_bk7, 'b--', linewidth=1.5, label='Phase index $n$')
ax2.set_xlabel('Wavelength (nm)', fontsize=12)
ax2.set_ylabel('Index', fontsize=12)
ax2.set_title('Phase Index vs. Group Index', fontsize=13)
ax2.axvspan(380, 700, alpha=0.15, color='yellow')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bk7_dispersion.png', dpi=150)
plt.show()
```

### 5.3 Rainbows: Dispersion in Nature

Rainbows are a beautiful manifestation of dispersion. Sunlight enters a water droplet, refracts at the surface, reflects internally, and refracts again upon exit. Because $n_{\text{water}}$ varies with wavelength, different colors emerge at slightly different angles:

- **Primary rainbow**: One internal reflection, seen at about 42$^\circ$ from the antisolar point. Red on the outside, violet on the inside.
- **Secondary rainbow**: Two internal reflections, seen at about 51$^\circ$. Colors reversed (red inside, violet outside). Fainter due to additional reflection loss.
- **Alexander's dark band**: The region between primary and secondary rainbows appears darker because no light is directed into this angular range.

---

## 6. Energy and Intensity

### 6.1 Poynting Vector

The energy flow of an electromagnetic wave is described by the **Poynting vector**:

$$\mathbf{S} = \frac{1}{\mu_0} \mathbf{E} \times \mathbf{B}$$

The time-averaged intensity (power per unit area) is:

$$I = \langle |\mathbf{S}| \rangle = \frac{1}{2} c \epsilon_0 E_0^2 = \frac{E_0^2}{2\mu_0 c}$$

For sunlight at Earth's surface, $I \approx 1000$ W/m$^2$ on a clear day.

### 6.2 Photon Flux

The intensity can also be expressed in terms of photon flux $\Phi$ (photons per second per unit area):

$$I = \Phi \cdot h\nu$$

A 1 mW red laser pointer ($\lambda = 650$ nm) emits about $3.3 \times 10^{15}$ photons per second — a staggering number, which is why classical wave optics works so well for most practical purposes.

```python
import numpy as np

# Calculate photon flux for a laser beam
# This helps build intuition for why classical optics works at everyday power levels

h = 6.626e-34       # Planck's constant (J·s)
c = 3e8             # speed of light (m/s)

# Laser parameters
power_mW = 1.0                      # laser power in milliwatts
wavelength_nm = 650                  # red laser pointer
beam_diameter_mm = 1.0               # typical beam diameter

# Derived quantities
power_W = power_mW * 1e-3
wavelength_m = wavelength_nm * 1e-9
beam_area = np.pi * (beam_diameter_mm * 1e-3 / 2) ** 2  # beam cross-section area

# Photon energy
E_photon = h * c / wavelength_m
print(f"Photon energy: {E_photon:.3e} J = {E_photon / 1.602e-19:.3f} eV")

# Photon flux (total photons per second)
photon_rate = power_W / E_photon
print(f"Photon emission rate: {photon_rate:.3e} photons/s")

# Intensity (power per unit area)
intensity = power_W / beam_area
print(f"Beam intensity: {intensity:.1f} W/m^2")

# Photon flux density (photons per second per unit area)
flux_density = photon_rate / beam_area
print(f"Photon flux density: {flux_density:.3e} photons/(s·m^2)")

# At this rate, individual photon granularity is utterly undetectable
# by any classical measurement — validating the wave description
```

---

## 7. Optical Path Length and Fermat's Principle (Preview)

The **optical path length** (OPL) through a medium is:

$$\text{OPL} = \int_A^B n(s) \, ds$$

where $n(s)$ is the refractive index along the path and $ds$ is the differential path element.

**Fermat's Principle** states that light follows the path for which the optical path length is *stationary* (usually a minimum). This single principle unifies:
- **Reflection**: angle of incidence equals angle of reflection
- **Refraction**: Snell's law
- **Curved paths** in graded-index media (e.g., mirages, GRIN lenses)

We will explore Fermat's principle in depth in [Lesson 02](02_Geometric_Optics_Fundamentals.md).

---

## Exercises

### Exercise 1: Photon Energy Calculations

A hydrogen atom emits a photon during the transition from $n=3$ to $n=2$ (the H-alpha line).

(a) Calculate the wavelength of the emitted photon using $\frac{1}{\lambda} = R_H \left(\frac{1}{n_1^2} - \frac{1}{n_2^2}\right)$ where $R_H = 1.097 \times 10^7$ m$^{-1}$.

(b) What color is this light?

(c) Calculate the photon's energy in eV and its momentum.

### Exercise 2: Speed in a Medium

Light enters a diamond ($n = 2.417$) from air.

(a) What is the speed of light inside the diamond?

(b) What is the wavelength of 589 nm (sodium D-line) light inside the diamond?

(c) Does the frequency change when light enters the diamond? Explain why or why not.

### Exercise 3: Dispersion Analysis

Using the Sellmeier equation for BK7 glass (coefficients given in the code above):

(a) Calculate the refractive index at 486.1 nm (F line), 587.6 nm (d line), and 656.3 nm (C line).

(b) Compute the Abbe number $V_d$.

(c) Calculate the group velocity at 500 nm. Is it faster or slower than the phase velocity?

### Exercise 4: Photon Counting

A CCD camera sensor has pixels of area 5 $\mu$m $\times$ 5 $\mu$m. During a 1-second exposure, the intensity of light at 550 nm reaching one pixel is $10^{-6}$ W/m$^2$.

(a) How many photons hit this pixel during the exposure?

(b) If the quantum efficiency is 70%, how many photoelectrons are generated?

(c) At what light level (intensity) does photon shot noise become significant — say, when $\sqrt{N} / N > 10\%$?

### Exercise 5: Phase vs. Group Velocity

In a medium where $n(\lambda) = 1.5 + \frac{3 \times 10^4}{\lambda^2}$ (with $\lambda$ in nm):

(a) Calculate the phase velocity at $\lambda = 500$ nm.

(b) Calculate $dn/d\lambda$ at this wavelength.

(c) Determine the group velocity. Is this medium normally or anomalously dispersive at 500 nm?

---

## Summary

| Concept | Key Formula / Fact |
|---------|-------------------|
| Wave-particle duality | Light behaves as both wave (interference, diffraction) and particle (photoelectric effect, Compton scattering) |
| Photon energy | $E = h\nu = hc/\lambda$; convenient form: $E$(eV) = 1240/$\lambda$(nm) |
| Photon momentum | $p = h/\lambda = E/c$ |
| EM wave relation | $c = \lambda\nu = \omega/k$ |
| Speed in medium | $v = c/n$, where $n$ is the refractive index |
| Phase velocity | $v_p = \omega/k = c/n(\omega)$ |
| Group velocity | $v_g = d\omega/dk = c/(n + \omega \cdot dn/d\omega)$ |
| Poynting vector | $\mathbf{S} = (\mathbf{E} \times \mathbf{B})/\mu_0$; intensity $I = \frac{1}{2}c\epsilon_0 E_0^2$ |
| Sellmeier equation | $n^2(\lambda) = 1 + \sum_i B_i\lambda^2/(\lambda^2 - C_i)$ |
| Abbe number | $V_d = (n_d - 1)/(n_F - n_C)$; measures dispersion |
| Historical progression | Newton (corpuscles) → Huygens/Young/Fresnel (waves) → Maxwell (EM) → Einstein/Planck (photons) → QED |

---

[Next: 02. Geometric Optics Fundamentals →](02_Geometric_Optics_Fundamentals.md)
