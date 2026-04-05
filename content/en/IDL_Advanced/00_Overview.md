# IDL Advanced — Study Guide

## Introduction

This folder provides a **comprehensive advanced IDL (Interactive Data Language) curriculum** focused on scientific data analysis with SolarSoft (SSW). While IDL Basics covers the language fundamentals, this course dives into advanced array manipulation, publication-quality visualization, solar instrument pipelines (SDO/AIA, SDO/HMI, GOES, RHESSI), spectral analysis, image processing, curve fitting, interoperability with Python, and performance optimization for large-scale data.

IDL remains the workhorse language for solar physics and space science. The SolarSoft library suite provides calibrated instrument pipelines, coordinate transforms, and data access utilities that have been refined over decades. Mastering these tools is essential for productive research in heliophysics.

## What You'll Learn

- Advanced array manipulation and multi-dimensional data techniques
- Publication-quality plotting: multi-panel, contour, surface, and PostScript output
- Map projections and heliographic coordinate systems
- Object-oriented IDL and widget programming
- SolarSoft framework: installation, instrument trees, utility routines
- SDO/AIA calibration, multi-wavelength analysis, and DEM estimation
- SDO/HMI magnetic field analysis and Carrington maps
- GOES X-ray light curves and RHESSI spectral/image analysis
- Spectral analysis: FFT, wavelets, Lomb-Scargle
- Image processing: filtering, morphology, feature detection
- Curve fitting: CURVEFIT, MPFIT, Gaussian fitting, chi-square analysis
- Scientific file formats: NetCDF, HDF5, CDF
- IDL-Python bridge and migration strategies
- Performance optimization and large dataset handling
- Capstone: end-to-end solar flare event analysis

## Prerequisites

| Topic | Required Level |
|-------|---------------|
| **[IDL Basics](../IDL_Basics/00_Overview.md)** | Proficient — variables, arrays, control flow, basic plotting, FITS I/O |
| **[Solar_Physics](../Solar_Physics/00_Overview.md)** | Familiar — solar atmosphere, flares, CMEs |
| Linux/Shell | Familiar — command line, environment variables |

## Learning Roadmap

```
┌─────────────────────────────────┐
│  Block 1: Advanced Foundations  │  L01–L04
│  Arrays, Plotting, Maps, OOP   │
└──────────┬──────────────────────┘
           │
┌──────────▼──────────────────────┐
│  Block 2: SolarSoft & Solar    │  L05–L08
│  Instruments                    │  SSW, AIA, HMI, GOES, RHESSI
└──────────┬──────────────────────┘
           │
┌──────────▼──────────────────────┐
│  Block 3: Analysis Techniques  │  L09–L12
│  Spectral, Image, Fitting, I/O │
└──────────┬──────────────────────┘
           │
┌──────────▼──────────────────────┐
│  Block 4: Integration          │  L13–L15
│  Python Bridge, Performance,   │
│  Capstone Project              │
└─────────────────────────────────┘
```

## Lessons

| # | Filename | Description |
|---|----------|-------------|
| **Block 1: Advanced Foundations** |
| 01 | `01_Advanced_Array_Techniques.md` | REFORM, REBIN, CONGRID, TOTAL, MEDIAN, SMOOTH, CONVOL, IMAGE_STATISTICS |
| 02 | `02_Advanced_Plotting.md` | Multi-panel plots, CONTOUR, SURFACE, PLOTS, PostScript output |
| 03 | `03_Map_Projections.md` | MAP_SET, MAP_CONTINENTS, coordinate transforms, WCS |
| 04 | `04_Object_Oriented_IDL.md` | Class definitions, inheritance, widget programming |
| **Block 2: SolarSoft & Solar Instruments** |
| 05 | `05_SolarSoft_Framework.md` | SSW installation, instrument trees, utility routines |
| 06 | `06_SDO_AIA_Analysis.md` | AIA_PREP, multi-wavelength analysis, DEM basics |
| 07 | `07_SDO_HMI_Analysis.md` | Magnetograms, vector fields, Carrington maps |
| 08 | `08_GOES_and_RHESSI.md` | GOES light curves, RHESSI imaging and spectroscopy |
| **Block 3: Analysis Techniques** |
| 09 | `09_Spectral_Analysis.md` | FFT, wavelets, Lomb-Scargle, spectral filtering |
| 10 | `10_Image_Processing.md` | Filtering, morphology, edge detection, feature tracking |
| 11 | `11_Curve_Fitting.md` | CURVEFIT, MPFIT, GAUSSFIT, chi-square analysis |
| 12 | `12_NetCDF_and_HDF5.md` | NetCDF, HDF5, CDF file I/O |
| **Block 4: Integration** |
| 13 | `13_IDL_Python_Bridge.md` | Python bridge, pIDLy, hissw, migration strategies |
| 14 | `14_Performance_and_Large_Data.md` | ASSOC, vectorization, memory management, profiling |
| 15 | `15_Capstone_Solar_Event_Analysis.md` | End-to-end solar flare analysis project |

## Environment Setup

### SolarSoft Installation

SolarSoft (SSW) is required for Lessons 05-08 and 15:

```bash
# Download SolarSoft
export SSW=/usr/local/ssw
mkdir -p $SSW
cd $SSW
wget https://www.lmsal.com/solarsoft/ssw_install.tar
tar xf ssw_install.tar

# Install instrument packages
ssw_install, /sdo, /aia, /hmi, /goes, /hessi

# Set environment variables
export SSW=/usr/local/ssw
export SSW_INSTR="aia hmi goes hessi"
source $SSW/gen/setup/setup.ssw
```

### Starting SolarSoft IDL

```bash
sswidl          # Launches IDL with SolarSoft environment loaded
```

## Related Materials

- **[IDL Basics](../IDL_Basics/00_Overview.md)** — Language fundamentals, prerequisite for this course
- **[Solar_Physics](../Solar_Physics/00_Overview.md)** — Solar atmosphere, flares, CMEs
- **[Space_Weather](../Space_Weather/00_Overview.md)** — Geomagnetic storms, forecasting
- **[Plasma_Physics](../Plasma_Physics/00_Overview.md)** — MHD, plasma waves, reconnection

---

*Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)*
