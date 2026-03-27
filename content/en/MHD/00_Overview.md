# Magnetohydrodynamics (MHD)

## Overview

This topic covers advanced magnetohydrodynamics — the physics of electrically conducting fluids interacting with magnetic fields. Building on the MHD foundations in Numerical Simulation (L17-L18) and plasma physics fundamentals (Plasma_Physics), these lessons explore equilibrium theory, stability, magnetic reconnection, turbulence, dynamo action, and astrophysical/fusion applications with comprehensive computational examples.

## Prerequisites

- **Plasma_Physics** L04-L06 (single particle motion, drifts, adiabatic invariants)
- **Plasma_Physics** L13-L14 (two-fluid model, kinetic-to-MHD derivation)
- **Numerical_Simulation** L17-L18 (ideal MHD equations, 1D MHD numerics)
- **Mathematical_Methods** L05 (vector analysis), L13 (PDE methods)
- Python intermediate level (NumPy, SciPy, Matplotlib)

## Lesson Plan

### Equilibrium and Stability (L01-L04)

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [01_MHD_Equilibria.md](./01_MHD_Equilibria.md) | ⭐⭐ | Force balance, Z-pinch, θ-pinch, Grad-Shafranov, safety factor, flux surfaces | Equilibrium theory |
| [02_Linear_Stability.md](./02_Linear_Stability.md) | ⭐⭐⭐ | Linearized MHD, energy principle (δW), Kruskal-Shafranov, Suydam criterion | Stability framework |
| [03_Pressure_Driven_Instabilities.md](./03_Pressure_Driven_Instabilities.md) | ⭐⭐⭐ | Rayleigh-Taylor, Parker instability, interchange, ballooning, Mercier | Pressure-driven modes |
| [04_Current_Driven_Instabilities.md](./04_Current_Driven_Instabilities.md) | ⭐⭐⭐ | Kink (m=1), sausage (m=0), tearing mode, NTM, resistive wall mode | Current-driven modes |

### Magnetic Reconnection (L05-L07)

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [05_Reconnection_Theory.md](./05_Reconnection_Theory.md) | ⭐⭐⭐⭐ | Sweet-Parker, Petschek, Hall MHD reconnection, X-point geometry | Theory fundamentals |
| [06_Reconnection_Applications.md](./06_Reconnection_Applications.md) | ⭐⭐⭐⭐ | Solar flares, CME, substorms, sawtooth crashes, island coalescence | Astrophysical/fusion |
| [07_Advanced_Reconnection.md](./07_Advanced_Reconnection.md) | ⭐⭐⭐⭐ | Plasmoid instability, turbulent reconnection, guide field, relativistic | Cutting-edge topics |

### MHD Turbulence and Dynamo (L08-L10)

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [08_MHD_Turbulence.md](./08_MHD_Turbulence.md) | ⭐⭐⭐⭐ | IK vs GS95 spectra, Elsässer variables, critical balance, anisotropy | Turbulence theory |
| [09_Dynamo_Theory.md](./09_Dynamo_Theory.md) | ⭐⭐⭐⭐ | Cowling theorem, mean-field theory, α-Ω dynamo, Earth/solar dynamo | Field generation |
| [10_Turbulent_Dynamo.md](./10_Turbulent_Dynamo.md) | ⭐⭐⭐⭐ | Small-scale (Kazantsev), large-scale dynamo, helicity, DNS/LES | Advanced dynamo |

### Astrophysical and Fusion Applications (L11-L14)

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [11_Solar_MHD.md](./11_Solar_MHD.md) | ⭐⭐⭐ | Flux tubes, sunspots, solar dynamo, coronal heating, Parker wind | Solar physics |
| [12_Accretion_Disk_MHD.md](./12_Accretion_Disk_MHD.md) | ⭐⭐⭐⭐ | MRI, angular momentum transport, α-disk, disk winds/jets | Accretion physics |
| [13_Fusion_MHD.md](./13_Fusion_MHD.md) | ⭐⭐⭐ | Tokamak/stellarator, disruptions, ELM, sawtooth, beta limits | Fusion plasma |
| [14_Space_Weather.md](./14_Space_Weather.md) | ⭐⭐⭐ | Magnetosphere, Dungey cycle, storms, CME propagation, GIC | Space weather |

### Advanced Computational Methods and Projects (L15-L18)

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [15_2D_MHD_Solver.md](./15_2D_MHD_Solver.md) | ⭐⭐⭐⭐ | 2D finite volume, Constrained Transport, WENO, Orszag-Tang vortex | 2D solver |
| [16_Relativistic_MHD.md](./16_Relativistic_MHD.md) | ⭐⭐⭐⭐ | SRMHD, GRMHD basics, relativistic jets, black hole accretion | Relativistic regime |
| [17_Spectral_Methods.md](./17_Spectral_Methods.md) | ⭐⭐⭐⭐ | Pseudo-spectral, Chebyshev, MHD-PIC hybrid, AMR, SPH-MHD | Advanced methods |
| [18_Projects.md](./18_Projects.md) | ⭐⭐⭐⭐ | Solar flare sim, disruption prediction, spherical dynamo | Three full projects |

## Recommended Learning Path

```
Equilibrium & Stability (L01-L04)
         │
         ├──→ Reconnection (L05-L07)
         │           │
         │           ▼
         ├──→ Turbulence & Dynamo (L08-L10)
         │           │
         │           ▼
         ├──→ Applications (L11-L14)
         │    Solar, Accretion, Fusion, Space Weather
         │           │
         └───────────┘
                     │
                     ▼
         Advanced Methods & Projects (L15-L18)
         2D Solver, Relativistic, Spectral, Projects
```

### Focused Paths

| Path | Lessons | Duration |
|------|---------|----------|
| **Fusion focus** | L01-L04 → L13 → L15 | 4 weeks |
| **Astrophysics focus** | L01-L04 → L05-L07 → L08-L10 → L11-L12 → L15-L16 | 8 weeks |
| **Computational focus** | L01-L02 → L15 → L17 → L18 | 4 weeks |
| **Full course** | L01-L18 in order | 12 weeks |

## Example Code

Example code for this topic is available in `examples/MHD/`.

## Total

- **18 lessons** (4 equilibrium/stability + 3 reconnection + 3 turbulence/dynamo + 4 applications + 4 advanced/projects)
- **Difficulty range**: ⭐⭐ to ⭐⭐⭐⭐
- **Languages**: Python (primary)
- **Key libraries**: NumPy, SciPy, Matplotlib, Numba (for 2D solvers)

## References

### Textbooks
- J.P. Freidberg, *Ideal MHD* (Cambridge, 2014)
- D. Biskamp, *Nonlinear Magnetohydrodynamics* (Cambridge, 1993)
- E. Priest, *Magnetohydrodynamics of the Sun* (Cambridge, 2014)
- J.P. Goedbloed, R. Keppens, S. Poedts, *Magnetohydrodynamics of Laboratory and Astrophysical Plasmas* (Cambridge, 2019)
- A. Brandenburg & A. Nordlund, "Astrophysical turbulence modeling" (Rep. Prog. Phys., 2011)

### Online
- NCAR HAO MHD tutorial
- Athena++ documentation: https://www.athena-astro.app/
