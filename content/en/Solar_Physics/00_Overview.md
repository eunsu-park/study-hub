# Solar Physics

## Overview

Solar physics studies the Sun — our nearest star — from its nuclear-burning core to the outer reaches of the heliosphere. As the only star close enough to observe in spatially resolved detail, the Sun serves as a fundamental laboratory for stellar astrophysics, plasma physics, and magnetohydrodynamics. Understanding the Sun is not merely an academic exercise: solar activity drives space weather that affects satellite operations, power grids, communications, and human spaceflight.

This topic covers the Sun's internal structure and energy generation, the techniques of helioseismology that reveal the solar interior, the layered solar atmosphere from the photosphere through the corona, the magnetic phenomena that drive solar activity (active regions, flares, coronal mass ejections), and the solar wind that fills interplanetary space. We also explore modern observational techniques, space weather applications, and current research missions. Throughout, we emphasize the physical principles — fluid dynamics, radiative transfer, MHD — that unify these diverse phenomena into a coherent picture of our nearest star.

## Prerequisites

- **Plasma_Physics** (L04-L06, L13-L14): Single particle motion, fluid description, plasma waves
- **MHD** (L05-L06, L09, L11): Magnetic reconnection, dynamo theory, solar MHD overview
- **Electrodynamics** (L04-L06, L10, L13): Magnetostatics, electromagnetic waves, radiation
- **Mathematical_Methods** (L04-L06, L10): Vector analysis, Fourier methods, ODEs/PDEs
- **Python**: NumPy, Matplotlib, SciPy (intermediate level)

## Learning Path

```
Solar Interior (L01-L04)
├── L01: Solar Interior — structure, hydrostatic equilibrium, energy transport
├── L02: Nuclear Energy Generation — pp chain, CNO cycle, neutrinos
├── L03: Helioseismology — oscillation modes, inversion, internal rotation
└── L04: Photosphere — radiative transfer, granulation, spectral lines

Solar Atmosphere (L05-L07)
├── L05: Chromosphere and Transition Region — spicules, network, UV emission
├── L06: Corona — heating problem, loops, X-ray emission
└── L07: Solar Magnetic Fields — flux tubes, magnetograph, force-free fields

Solar Activity (L08-L12)
├── L08: Active Regions and Sunspots — structure, evolution, magnetic classification
├── L09: Solar Dynamo and Activity Cycle — alpha-omega dynamo, butterfly diagram
├── L10: Solar Flares — magnetic reconnection, particle acceleration, emission
├── L11: Coronal Mass Ejections — initiation, propagation, interplanetary CMEs
└── L12: Solar Wind — Parker model, fast/slow wind, heliospheric structure

Observations & Applications (L13-L16)
├── L13: Solar Spectroscopy and Instruments — spectral diagnostics, coronagraphs, EUV imaging
├── L14: Solar Energetic Particles — acceleration, transport, SEP events
├── L15: Space Weather and Modern Missions — forecasting, SDO, Parker Solar Probe, Solar Orbiter
└── L16: Capstone Projects — integrated modeling and analysis exercises
```

## Lesson List

| # | Lesson | Description |
|---|--------|-------------|
| 01 | [Solar Interior](01_Solar_Interior.md) | Hydrostatic equilibrium, radiative/convective energy transport, tachocline, standard solar model |
| 02 | [Nuclear Energy Generation](02_Nuclear_Energy_Generation.md) | Thermonuclear reactions, Gamow peak, pp chain, CNO cycle, solar neutrino problem |
| 03 | [Helioseismology](03_Helioseismology.md) | Solar oscillations, p/g/f modes, l-nu diagram, inversion techniques, internal rotation |
| 04 | [Photosphere](04_Photosphere.md) | Radiative transfer, limb darkening, granulation, supergranulation, spectral line formation |
| 05 | [Chromosphere and Transition Region](05_Chromosphere_and_TR.md) | Chromospheric structure, spicules, transition region, UV/EUV emission |
| 06 | [Corona](06_Corona.md) | Coronal heating problem, coronal loops, X-ray/EUV observations, solar wind origin |
| 07 | [Solar Magnetic Fields](07_Solar_Magnetic_Fields.md) | Flux tubes, magnetographs, potential and force-free field models, magnetic helicity |
| 08 | [Active Regions and Sunspots](08_Active_Regions_and_Sunspots.md) | Sunspot structure, penumbral fine structure, magnetic classification, active region evolution |
| 09 | [Solar Dynamo and Activity Cycle](09_Solar_Dynamo_and_Cycle.md) | Alpha-omega dynamo, Babcock-Leighton mechanism, butterfly diagram, grand minima |
| 10 | [Solar Flares](10_Solar_Flares.md) | Standard flare model, magnetic reconnection, particle acceleration, multi-wavelength emission |
| 11 | [Coronal Mass Ejections](11_CMEs.md) | CME initiation mechanisms, propagation models, interplanetary CMEs, geomagnetic storms |
| 12 | [Solar Wind](12_Solar_Wind.md) | Parker spiral, fast/slow wind sources, heliospheric current sheet, termination shock |
| 13 | [Solar Spectroscopy and Instruments](13_Spectroscopy_Instruments.md) | Emission/absorption diagnostics, coronagraphs, EUV/X-ray imagers, radio observations |
| 14 | [Solar Energetic Particles](14_SEPs.md) | Impulsive vs gradual events, diffusive shock acceleration, particle transport |
| 15 | [Space Weather and Modern Missions](15_Modern_Solar_Missions.md) | Forecasting models, SDO, Parker Solar Probe, Solar Orbiter, DKIST |
| 16 | [Capstone Projects](16_Projects.md) | Integrated modeling: solar interior model, flare analysis, CME propagation, helioseismology pipeline |

## Related Topics

| Topic | Connection |
|-------|------------|
| Plasma_Physics | Single-particle motion, Vlasov equation, plasma waves — foundation for coronal and heliospheric physics |
| MHD | Magnetic reconnection, dynamo theory, MHD instabilities — essential for flares, CMEs, solar wind |
| Electrodynamics | Electromagnetic wave propagation, radiation theory — basis for solar radio and spectral observations |
| Signal_Processing | Fourier analysis, spectral methods — core tools for helioseismology and time series analysis |
| Optics | Spectroscopy, diffraction, polarimetry — instrument design and observational techniques |
| Numerical_Simulation | ODE/PDE solvers, MHD codes — computational solar physics |

## Example Files

Located in `examples/Solar_Physics/`:

| File | Description |
|------|-------------|
| `01_solar_interior_model.py` | Numerical integration of hydrostatic equilibrium and temperature profile |
| `02_nuclear_reactions.py` | Gamow peak calculation, pp chain energy generation rate, neutrino flux |
| `03_helioseismology.py` | Spherical harmonic decomposition, l-nu diagram, acoustic ray tracing |
| `04_radiative_transfer.py` | Eddington-Barbier solution, limb darkening curves, spectral line profiles |
| `05_chromosphere_tr.py` | Temperature-height model, DEM analysis, transition region emission |
| `06_coronal_loop.py` | Hydrostatic coronal loop model, scaling laws, energy balance |
| `07_magnetic_field.py` | Potential field source surface (PFSS) extrapolation, magnetic topology |
| `08_sunspot_model.py` | Sunspot cooling model, Wilson depression, penumbral flow simulation |
| `09_dynamo_model.py` | Babcock-Leighton flux transport dynamo, butterfly diagram generation |
| `10_flare_reconnection.py` | Sweet-Parker and Petschek reconnection rates, flare energy release |
| `11_cme_propagation.py` | Drag-based CME propagation model, arrival time prediction |
| `12_parker_wind.py` | Parker solar wind equation solver, spiral magnetic field, wind speed profile |
