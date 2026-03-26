# Electrodynamics

## Overview

Electrodynamics is the study of electromagnetic fields and their interactions with charged matter. Building on Coulomb's law and magnetostatics, it culminates in Maxwell's equations — a unified framework that describes all classical electromagnetic phenomena, from static charges to light propagation. This topic bridges fundamental physics with practical applications in antennas, waveguides, and computational electromagnetics.

## Prerequisites

- **Mathematical Methods**: Vector calculus, complex analysis, Fourier transforms (Mathematical_Methods L05-L08)
- **Python**: NumPy, Matplotlib (Python L01-L08)
- **Linear Algebra**: Eigenvalues, matrix operations (Math_for_AI L01-L03)
- **Signal Processing** (optional): Fourier analysis background (Signal_Processing L03-L05)

## Learning Path

```
Foundations (L01-L06)
├── L01: Electrostatics Review
├── L02: Electric Potential and Energy
├── L03: Conductors and Dielectrics
├── L04: Magnetostatics
├── L05: Magnetic Vector Potential
└── L06: Electromagnetic Induction

Maxwell's Equations (L07-L11)
├── L07: Maxwell's Equations — Differential Form
├── L08: Maxwell's Equations — Integral Form
├── L09: Electromagnetic Waves in Vacuum
├── L10: Electromagnetic Waves in Matter
└── L11: Reflection and Refraction

Advanced Topics (L12-L18)
├── L12: Waveguides and Cavities
├── L13: Radiation and Antennas
├── L14: Relativistic Electrodynamics
├── L15: Multipole Expansion
├── L16: Computational Electrodynamics (FDTD)
├── L17: Electromagnetic Scattering
└── L18: Applications — Plasmonics and Metamaterials
```

## Lessons

| # | Lesson | Description |
|---|--------|-------------|
| 01 | [Electrostatics Review](01_Electrostatics_Review.md) | Coulomb's law, Gauss's law, electric field, superposition |
| 02 | [Electric Potential and Energy](02_Electric_Potential_and_Energy.md) | Scalar potential, Poisson/Laplace equations, energy density |
| 03 | [Conductors and Dielectrics](03_Conductors_and_Dielectrics.md) | Boundary conditions, polarization, capacitance, dielectric constant |
| 04 | [Magnetostatics](04_Magnetostatics.md) | Biot-Savart law, Ampère's law, magnetic dipole, vector potential |
| 05 | [Magnetic Vector Potential](05_Magnetic_Vector_Potential.md) | Gauge freedom, Coulomb gauge, multipole expansion |
| 06 | [Electromagnetic Induction](06_Electromagnetic_Induction.md) | Faraday's law, Lenz's law, inductance, mutual inductance |
| 07 | [Maxwell's Equations — Differential Form](07_Maxwells_Equations_Differential.md) | Displacement current, full Maxwell equations, wave equation derivation |
| 08 | [Maxwell's Equations — Integral Form](08_Maxwells_Equations_Integral.md) | Stokes/divergence theorems, conservation laws, Poynting vector |
| 09 | [EM Waves in Vacuum](09_EM_Waves_Vacuum.md) | Plane waves, polarization, energy transport, impedance of free space |
| 10 | [EM Waves in Matter](10_EM_Waves_Matter.md) | Dispersion, absorption, complex refractive index, skin depth |
| 11 | [Reflection and Refraction](11_Reflection_and_Refraction.md) | Fresnel equations, Brewster angle, total internal reflection, coatings |
| 12 | [Waveguides and Cavities](12_Waveguides_and_Cavities.md) | TE/TM modes, cutoff frequency, rectangular waveguide, resonant cavities |
| 13 | [Radiation and Antennas](13_Radiation_and_Antennas.md) | Retarded potentials, Larmor formula, dipole radiation, antenna arrays |
| 14 | [Relativistic Electrodynamics](14_Relativistic_Electrodynamics.md) | Lorentz transformation of fields, electromagnetic tensor, covariant formulation |
| 15 | [Multipole Expansion](15_Multipole_Expansion.md) | Monopole, dipole, quadrupole, spherical harmonics, radiation patterns |
| 16 | [Computational Electrodynamics](16_Computational_Electrodynamics.md) | FDTD method, Yee grid, absorbing boundaries (PML), simulation examples |
| 17 | [Electromagnetic Scattering](17_Electromagnetic_Scattering.md) | Rayleigh scattering, Mie theory, Born approximation, cross sections |
| 18 | [Applications — Plasmonics and Metamaterials](18_Plasmonics_and_Metamaterials.md) | Surface plasmons, negative refraction, cloaking, photonic crystals |

## Relationship to Other Topics

| Topic | Connection |
|-------|-----------|
| Mathematical_Methods | Vector calculus, Fourier transforms, Green's functions used throughout |
| Plasma_Physics | Maxwell's equations couple with fluid equations in plasma |
| MHD | Low-frequency limit of electrodynamics in conducting fluids |
| Signal_Processing | Wave propagation, Fourier analysis, filter design connections |
| Numerical_Simulation | FDTD and FEM methods for field computation |
| Computer_Vision | Optics foundations for image formation |

## Example Files

Located in `examples/Electrodynamics/`:

| File | Description |
|------|-------------|
| `01_electrostatics.py` | Electric field visualization, Gauss's law verification |
| `02_potential_laplace.py` | Laplace equation solver, equipotential lines |
| `03_capacitor_sim.py` | Parallel plate capacitor, dielectric effects |
| `04_magnetostatics.py` | Biot-Savart field computation, magnetic dipole |
| `05_faraday_induction.py` | Faraday's law simulation, EMF computation |
| `06_maxwell_waves.py` | Plane wave propagation, Poynting vector |
| `07_fresnel.py` | Fresnel coefficients, Brewster angle plot |
| `08_waveguide_modes.py` | TE/TM mode patterns, cutoff frequencies |
| `09_dipole_radiation.py` | Oscillating dipole radiation pattern |
| `10_fdtd_1d.py` | 1D FDTD simulation with absorbing boundaries |
| `11_fdtd_2d.py` | 2D FDTD simulation, waveguide, scattering |
| `12_mie_scattering.py` | Mie theory computation, scattering cross sections |
