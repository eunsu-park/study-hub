# Plasma Physics

## Overview

This topic covers the fundamental physics of plasmas — the fourth state of matter — from single particle dynamics to kinetic theory and fluid descriptions. These lessons bridge the gap between basic electromagnetism and advanced topics like Magnetohydrodynamics (MHD), providing the physical foundation needed for fusion, space, and astrophysical plasma research.

## Prerequisites

- Vector calculus (Mathematical_Methods L05)
- Partial differential equations (Mathematical_Methods L13)
- Basic electromagnetism (Maxwell's equations, Lorentz force)
- Python intermediate level (NumPy, SciPy, Matplotlib)

## Lesson Plan

### Fundamentals

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [01_Introduction_to_Plasma.md](./01_Introduction_to_Plasma.md) | ⭐ | Debye shielding, plasma frequency, gyrofrequency, plasma beta, quasi-neutrality | Conceptual foundation |
| [02_Coulomb_Collisions.md](./02_Coulomb_Collisions.md) | ⭐⭐ | Coulomb scattering, collision frequencies, Spitzer resistivity, mean free path | Collisionality regimes |
| [03_Plasma_Description_Hierarchy.md](./03_Plasma_Description_Hierarchy.md) | ⭐⭐ | Klimontovich → Vlasov → Fluid hierarchy, model selection criteria | Framework overview |

### Single Particle Motion

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [04_Single_Particle_Motion_I.md](./04_Single_Particle_Motion_I.md) | ⭐⭐ | Gyration, Larmor radius, E×B drift, guiding center | Uniform fields |
| [05_Single_Particle_Motion_II.md](./05_Single_Particle_Motion_II.md) | ⭐⭐⭐ | Grad-B drift, curvature drift, polarization drift, general force drift | Non-uniform fields |
| [06_Magnetic_Mirrors_Adiabatic_Invariants.md](./06_Magnetic_Mirrors_Adiabatic_Invariants.md) | ⭐⭐⭐ | Magnetic mirror, μ/J/Φ invariants, loss cone, banana orbits | Trapped particles |

### Kinetic Theory

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [07_Vlasov_Equation.md](./07_Vlasov_Equation.md) | ⭐⭐⭐ | Phase space, distribution functions, Vlasov equation, BGK modes | Collisionless kinetics |
| [08_Landau_Damping.md](./08_Landau_Damping.md) | ⭐⭐⭐⭐ | Landau contour, wave-particle resonance, inverse Landau damping, particle trapping | Key kinetic effect |
| [09_Collisional_Kinetics.md](./09_Collisional_Kinetics.md) | ⭐⭐⭐⭐ | Fokker-Planck, Rosenbluth potentials, Braginskii transport, neoclassical | Collisional effects |

### Plasma Waves

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [10_Electrostatic_Waves.md](./10_Electrostatic_Waves.md) | ⭐⭐⭐ | Langmuir waves, ion acoustic waves, Bernstein modes | Electrostatic dispersion |
| [11_Electromagnetic_Waves.md](./11_Electromagnetic_Waves.md) | ⭐⭐⭐ | R/L/O/X modes, whistler waves, CMA diagram, Faraday rotation | EM wave propagation |
| [12_Wave_Heating_and_Instabilities.md](./12_Wave_Heating_and_Instabilities.md) | ⭐⭐⭐⭐ | ECRH, ICRH, beam-plasma, Weibel, firehose, mirror instabilities | Heating and stability |

### Fluid Description

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [13_Two_Fluid_Model.md](./13_Two_Fluid_Model.md) | ⭐⭐⭐ | Moment equations, generalized Ohm's law, Hall effect, diamagnetic drift | Bridge to MHD |
| [14_From_Kinetic_to_MHD.md](./14_From_Kinetic_to_MHD.md) | ⭐⭐⭐⭐ | CGL model, MHD validity conditions, drift/gyrokinetic theory overview | Systematic reduction |

### Applications and Projects

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [15_Plasma_Diagnostics.md](./15_Plasma_Diagnostics.md) | ⭐⭐⭐ | Langmuir probe, Thomson scattering, interferometry, spectroscopy | Experimental methods |
| [16_Projects.md](./16_Projects.md) | ⭐⭐⭐⭐ | Orbit simulator, dispersion solver, 1D Vlasov-Poisson solver | Three full projects |

## Recommended Learning Path

```
Fundamentals (L01-L03)          Single Particle Motion (L04-L06)
       │                                │
       ▼                                ▼
  Plasma parameters            Gyration, drifts, mirrors
  Collisions, models           Adiabatic invariants
       │                                │
       └────────────┬───────────────────┘
                    │
                    ▼
          Kinetic Theory (L07-L09)
          Vlasov, Landau damping
          Fokker-Planck, transport
                    │
                    ▼
          Plasma Waves (L10-L12)
          ES/EM waves, CMA diagram
          Heating, instabilities
                    │
                    ▼
          Fluid Description (L13-L14)
          Two-fluid, Ohm's law
          MHD derivation, gyrokinetics
                    │
            ┌───────┴───────┐
            ▼               ▼
    Diagnostics (L15)   Projects (L16)
    Probes, scattering  Orbit sim, Vlasov
            │
            ▼
    → MHD Topic (advanced)
```

## Related Topics

- **Numerical_Simulation L17-L18**: MHD basics and numerical methods (prerequisite for MHD topic, complementary to L13-L14)
- **Numerical_Simulation L19**: PIC simulation method (computational complement to L04-L06)
- **Mathematical_Methods L05**: Vector analysis (used throughout)
- **Mathematical_Methods L13**: PDE methods (used in wave theory)
- **MHD Topic**: Advanced magnetohydrodynamics (builds on L04-L06, L13-L14)

## Example Code

Example code for this topic is available in `examples/Plasma_Physics/`.

## Total

- **16 lessons** (3 fundamentals + 3 particle motion + 3 kinetic + 3 waves + 2 fluid + 2 applications/projects)
- **Difficulty range**: ⭐ to ⭐⭐⭐⭐
- **Languages**: Python (primary)
- **Key libraries**: NumPy, SciPy, Matplotlib, Numba (optional for Vlasov solver)

## References

### Textbooks
- F.F. Chen, *Introduction to Plasma Physics and Controlled Fusion* (Vol. 1, 3rd ed.)
- R.J. Goldston & P.H. Rutherford, *Introduction to Plasma Physics*
- D.R. Nicholson, *Introduction to Plasma Theory*
- T.J.M. Boyd & J.J. Sanderson, *The Physics of Plasmas*
- J.A. Bittencourt, *Fundamentals of Plasma Physics*

### Online
- MIT OCW 22.611J: Introduction to Plasma Physics I
- Princeton Plasma Physics Laboratory educational resources
