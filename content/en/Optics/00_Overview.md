# Optics

## Overview

Optics is the study of light — its generation, propagation, and interaction with matter. From the ancient observation that light travels in straight lines to modern quantum optics and photonics, optics provides the foundation for technologies ranging from eyeglasses and cameras to fiber-optic communications, lasers, and holographic displays. This topic bridges electrodynamics with practical applications in imaging, telecommunications, and photonics.

## Prerequisites

- **Electrodynamics**: Maxwell's equations, electromagnetic waves (Electrodynamics L07-L11)
- **Mathematical Methods**: Fourier transforms, complex analysis (Mathematical_Methods L06-L08)
- **Python**: NumPy, Matplotlib (Python L01-L08)
- **Signal Processing** (optional): Fourier analysis background (Signal_Processing L03-L05)

## Learning Path

```
Foundations (L01-L04)
├── L01: Nature of Light
├── L02: Geometric Optics Fundamentals
├── L03: Mirrors and Lenses
└── L04: Optical Instruments

Wave Optics (L05-L07)
├── L05: Wave Optics — Interference
├── L06: Diffraction
└── L07: Polarization

Modern & Advanced Optics (L08-L14)
├── L08: Laser Fundamentals
├── L09: Fiber Optics
├── L10: Fourier Optics
├── L11: Holography
├── L12: Nonlinear Optics
├── L13: Quantum Optics Primer
└── L14: Computational Optics

Applied & Measurement Optics (L15-L17)
├── L15: Zernike Polynomials
├── L16: Adaptive Optics
└── L17: Spectroscopy
```

## Lessons

| # | Lesson | Description |
|---|--------|-------------|
| 01 | [Nature of Light](01_Nature_of_Light.md) | Wave-particle duality, EM spectrum, refractive index, dispersion |
| 02 | [Geometric Optics Fundamentals](02_Geometric_Optics_Fundamentals.md) | Fermat's principle, Snell's law, total internal reflection, prisms |
| 03 | [Mirrors and Lenses](03_Mirrors_and_Lenses.md) | Mirror/lens equations, magnification, aberrations |
| 04 | [Optical Instruments](04_Optical_Instruments.md) | Microscope, telescope, camera, human eye, Rayleigh criterion |
| 05 | [Wave Optics — Interference](05_Wave_Optics_Interference.md) | Young's double slit, thin films, Michelson interferometer, coherence |
| 06 | [Diffraction](06_Diffraction.md) | Single slit, Airy pattern, diffraction grating, Fresnel diffraction |
| 07 | [Polarization](07_Polarization.md) | Malus's law, wave plates, Jones calculus, birefringence, optical activity |
| 08 | [Laser Fundamentals](08_Laser_Fundamentals.md) | Stimulated emission, laser types, Gaussian beams, ABCD matrices |
| 09 | [Fiber Optics](09_Fiber_Optics.md) | Step/graded index, NA, dispersion, EDFA, WDM, fiber Bragg gratings |
| 10 | [Fourier Optics](10_Fourier_Optics.md) | Angular spectrum, lens as FT, 4f system, OTF/MTF/PSF, phase contrast |
| 11 | [Holography](11_Holography.md) | Recording/reconstruction, volume holograms, digital holography |
| 12 | [Nonlinear Optics](12_Nonlinear_Optics.md) | SHG, phase matching, Kerr effect, four-wave mixing, OPO |
| 13 | [Quantum Optics Primer](13_Quantum_Optics_Primer.md) | Photon states, squeezed light, entanglement, QKD (BB84) |
| 14 | [Computational Optics](14_Computational_Optics.md) | Ray tracing, BPM, phase retrieval, computational photography |
| 15 | [Zernike Polynomials](15_Zernike_Polynomials.md) | Wavefront analysis, Noll indexing, orthogonality, Kolmogorov turbulence |
| 16 | [Adaptive Optics](16_Adaptive_Optics.md) | Shack-Hartmann, deformable mirrors, closed-loop control, laser guide stars |
| 17 | [Spectroscopy](17_Spectroscopy.md) | Line broadening, gratings, Fabry-Pérot, Beer-Lambert, Raman |

## Relationship to Other Topics

| Topic | Connection |
|-------|-----------|
| Electrodynamics | Maxwell's equations as the foundation of light propagation |
| Signal_Processing | Fourier analysis, spatial filtering, transfer functions |
| Computer_Vision | Image formation, lens models, camera calibration |
| Quantum_Computing | Photonic quantum computing, entangled photon sources |
| Mathematical_Methods | Fourier transforms, Bessel functions, complex analysis |
| Numerical_Simulation | FDTD and BPM methods for photonic device simulation |

## Example Files

Located in `examples/Optics/`:

| File | Description |
|------|-------------|
| `01_snells_law.py` | Snell's law visualization, total internal reflection, prism dispersion |
| `02_thin_lens.py` | Thin lens ray tracing, image formation, aberrations |
| `03_interference.py` | Young's double slit, thin film coatings, Michelson interferometer |
| `04_diffraction.py` | Single slit, Airy pattern, diffraction grating |
| `05_polarization.py` | Jones calculus, Malus's law, wave plates, Brewster angle |
| `06_gaussian_beam.py` | Gaussian beam propagation, ABCD matrices, beam quality |
| `07_fiber_optics.py` | Fiber modes, dispersion, attenuation budget |
| `08_fourier_optics.py` | 2D Fourier transforms, 4f filtering, PSF/OTF |
| `09_ray_tracing.py` | Sequential ray tracer, spot diagrams, aberration analysis |
| `10_holography_sim.py` | Hologram recording/reconstruction, phase retrieval |
| `11_zernike_polynomials.py` | Zernike mode gallery, wavefront fitting, Kolmogorov phase screens |
| `12_adaptive_optics.py` | Shack-Hartmann WFS, deformable mirror, closed-loop AO simulation |
| `13_spectroscopy.py` | Line profiles (Voigt), grating spectrometer, Fabry-Pérot, Beer-Lambert |
