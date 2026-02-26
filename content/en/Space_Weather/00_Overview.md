# Space Weather

## Overview

Space weather encompasses the dynamic conditions in the space environment — from the Sun through interplanetary space to Earth's magnetosphere, ionosphere, and thermosphere — that can affect technological systems and human activity. Unlike terrestrial weather, which is driven by differential solar heating of the atmosphere, space weather is driven by the Sun's magnetic activity: solar flares, coronal mass ejections (CMEs), and the ever-flowing solar wind create a chain of disturbances that propagate outward through the heliosphere and interact with planetary magnetic fields.

Understanding space weather requires synthesizing knowledge from solar physics, plasma physics, magnetohydrodynamics, and electrodynamics. The field has grown from a scientific curiosity into an operational necessity as modern society has become increasingly dependent on space-based and ground-based technological infrastructure vulnerable to geomagnetic disturbances. This topic traces the full chain from solar wind arrival at Earth through magnetospheric dynamics, ionospheric effects, and practical impacts, culminating in modern forecasting approaches including machine learning methods.

## Prerequisites

| Topic | Lessons | Concepts |
|-------|---------|----------|
| Plasma_Physics | L04-L06, L10-L12 | Particle motion, adiabatic invariants, plasma waves, kinetic theory |
| MHD | L05-L06, L14 | Magnetic reconnection, MHD equilibria, space weather MHD overview |
| Electrodynamics | L04-L06, L10 | Magnetostatics, Maxwell's equations, EM waves in matter |
| Mathematical_Methods | L04-L06, L10 | Vector analysis, Fourier methods, ODEs |
| Python | — | NumPy, Matplotlib, SciPy (intermediate level) |

## Learning Path

```
                        Space Weather Learning Path
                        ===========================

Block 1: Magnetosphere          Block 2: Geomagnetic Activity
(L01-L04)                       (L05-L07)
┌─────────────────────┐         ┌─────────────────────┐
│ L01 Introduction    │         │ L05 Geomagnetic     │
│ L02 Magnetosphere   │────────▶│     Storms          │
│     Structure       │         │ L06 Substorms       │
│ L03 Current Systems │         │ L07 Radiation Belts │
│ L04 SW-M Coupling   │         │                     │
└─────────────────────┘         └────────┬────────────┘
                                         │
                                         ▼
Block 3: Near-Earth Environment  Block 4: Impacts & Forecasting
(L08-L10)                       (L11-L16)
┌─────────────────────┐         ┌─────────────────────────┐
│ L08 Ionosphere      │         │ L11 GICs & Power Grids  │
│ L09 Thermosphere &  │────────▶│ L12 Satellite & Tech    │
│     Drag            │         │ L13 Geomagnetic Indices │
│ L10 SEP Events      │         │ L14 Forecasting Methods │
└─────────────────────┘         │ L15 AI/ML for SW       │
                                │ L16 Capstone Projects   │
                                └─────────────────────────┘
```

## Lesson List

| # | Lesson | Description |
|---|--------|-------------|
| 01 | [Introduction to Space Weather](./01_Introduction_to_Space_Weather.md) | Definition, Sun-Earth connection chain, historical events, socioeconomic impacts |
| 02 | [Magnetosphere Structure](./02_Magnetosphere_Structure.md) | Dipole field, magnetopause, bow shock, magnetosheath, plasmasphere, L-shells |
| 03 | [Magnetospheric Current Systems](./03_Magnetospheric_Current_Systems.md) | Chapman-Ferraro, ring current, tail current, Birkeland currents, ionospheric currents |
| 04 | [Solar Wind-Magnetosphere Coupling](./04_Solar_Wind_Magnetosphere_Coupling.md) | Dayside reconnection, coupling functions, polar cap potential, viscous interaction |
| 05 | [Geomagnetic Storms](./05_Geomagnetic_Storms.md) | Storm phases, Dst development, CME- vs CIR-driven storms, ring current injection |
| 06 | [Substorms](./06_Magnetospheric_Substorms.md) | Growth/expansion/recovery phases, current disruption, plasmoid formation |
| 07 | [Radiation Belts](./07_Radiation_Belts.md) | Inner/outer belts, trapped particle dynamics, wave-particle interactions, slot region |
| 08 | [Ionospheric Space Weather](./08_Ionosphere.md) | Ionospheric layers, storms, scintillation, TEC variations, GNSS effects |
| 09 | [Thermosphere and Satellite Drag](./09_Thermosphere_and_Satellite_Drag.md) | Thermospheric heating, density enhancements, drag modeling, orbit prediction |
| 10 | [Solar Energetic Particle Events](./10_Solar_Energetic_Particle_Events.md) | SEP acceleration, transport, GLE events, radiation hazards |
| 11 | [Geomagnetically Induced Currents](./11_Geomagnetically_Induced_Currents.md) | GIC physics, power grid impacts, pipeline effects, mitigation strategies |
| 12 | [Satellite and Technology Impacts](./12_Technological_Impacts.md) | Surface/deep charging, single-event effects, HF blackouts, aviation impacts |
| 13 | [Geomagnetic Indices](./13_Space_Weather_Indices.md) | Dst, Kp, AE, SYM-H, Ap, F10.7, NOAA scales, derivation and interpretation |
| 14 | [Space Weather Forecasting](./14_Forecasting_Models.md) | Empirical, physics-based, and ensemble models; ENLIL, WSA, real-time operations |
| 15 | [AI and Machine Learning for Space Weather](./15_AI_ML_for_Space_Weather.md) | Neural networks for Dst, CNN for solar images, LSTM for time series, transfer learning |
| 16 | [Capstone Projects](./16_Projects.md) | End-to-end storm analysis, Dst prediction pipeline, radiation belt modeling |

## Related Topics

| Topic | Connection |
|-------|------------|
| Plasma_Physics | Fundamental plasma processes: particle motion, waves, kinetic theory |
| MHD | Magnetic reconnection, MHD equilibria, large-scale plasma dynamics |
| Electrodynamics | Maxwell's equations, EM wave propagation, magnetostatics |
| Deep_Learning | Neural network architectures for space weather prediction |
| Machine_Learning | Classical ML approaches for geomagnetic index forecasting |
| Signal_Processing | Time series analysis of geomagnetic data, spectral methods |

## Example Files

| File | Description |
|------|-------------|
| `dipole_field.py` | Earth's magnetic dipole field visualization and L-shell mapping |
| `magnetopause_standoff.py` | Magnetopause standoff distance calculation (Shue model) |
| `ring_current_dst.py` | Ring current energy and Dst depression (Dessler-Parker-Sckopke) |
| `coupling_functions.py` | Akasofu epsilon, Newell, and Borovsky coupling functions |
| `burton_equation.py` | Burton equation Dst prediction model |
| `substorm_current_wedge.py` | Substorm current wedge magnetic perturbation model |
| `radiation_belt_diffusion.py` | Radial diffusion equation for radiation belt electrons |
| `ionospheric_conductivity.py` | Height-dependent Pedersen and Hall conductivities |
| `gic_calculation.py` | GIC calculation from geoelectric field and network model |
| `geomagnetic_indices.py` | Kp, Dst, AE index calculation from magnetometer data |
| `dst_neural_network.py` | LSTM-based Dst index prediction |
| `storm_analysis_pipeline.py` | End-to-end geomagnetic storm event analysis |
