# 06. SDO/AIA Analysis

**Previous**: [SolarSoft Framework](./05_SolarSoft_Framework.md) | **Next**: [SDO/HMI Analysis](./07_SDO_HMI_Analysis.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand SDO/AIA instrument characteristics (channels, cadence, resolution)
2. Read and calibrate AIA data with `read_sdo` and `aia_prep`
3. Work with AIA response functions for temperature diagnostics
4. Create multi-wavelength composite images
5. Perform basic Differential Emission Measure (DEM) analysis

---

## 1. SDO/AIA Overview

The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO) provides full-disk solar images in 10 wavelength channels.

### AIA EUV/UV Channels

| Channel (A) | Ion | log T (K) | Region | Cadence |
|-------------|-----|-----------|--------|---------|
| 94 | Fe XVIII | 6.8 | Flaring plasma | 12 s |
| 131 | Fe VIII, XXI | 5.6, 7.0 | Transition region / Flares | 12 s |
| 171 | Fe IX | 5.8 | Quiet corona, loops | 12 s |
| 193 | Fe XII, XXIV | 6.2, 7.3 | Corona / Hot flare | 12 s |
| 211 | Fe XIV | 6.3 | Active region corona | 12 s |
| 304 | He II | 4.7 | Chromosphere / TR | 12 s |
| 335 | Fe XVI | 6.4 | Active region corona | 12 s |
| 1600 | C IV + cont. | 5.0 | TR / Upper photosphere | 24 s |
| 1700 | Continuum | 3.7 | Photosphere | 24 s |
| 4500 | Continuum | 3.7 | Photosphere | 3600 s |

### AIA Specifications

- **Spatial resolution**: 0.6 arcsec/pixel (4096 x 4096 pixels)
- **Field of view**: 41 arcmin (1.3 solar diameters)
- **Temporal cadence**: 12 s (EUV), 24 s (UV), 3600 s (visible)
- **Data rate**: ~1.5 TB/day

---

## 2. Reading AIA Data

### Using read_sdo

```idl
; Read a single AIA FITS file
file = 'aia.lev1.171A_2024-01-15T120000Z.image_lev1.fits'
read_sdo, file, index, data

; Examine header
HELP, index, /STRUCTURE
PRINT, 'Channel:   ', index.wavelnth, ' A'
PRINT, 'Date:      ', index.date_obs
PRINT, 'Exptime:   ', index.exptime, ' s'
PRINT, 'Image size:', index.naxis1, ' x ', index.naxis2
PRINT, 'Pixel size:', index.cdelt1, ' arcsec/pixel'
PRINT, 'Sun center:', index.crpix1, ', ', index.crpix2
```

### Reading Multiple Files

```idl
; Read a time series
files = FILE_SEARCH('/data/aia/171/2024/01/15/*.fits', COUNT=nf)
PRINT, nf, ' files found'

; Read all at once (memory-intensive for 4k images)
read_sdo, files, index, data

; Or read one at a time (memory-efficient)
FOR i = 0, nf-1 DO BEGIN
    read_sdo, files[i], idx, dat
    ; Process each image...
    IF i EQ 0 THEN all_index = idx ELSE $
        all_index = [all_index, idx]
ENDFOR
```

---

## 3. AIA Calibration with aia_prep

`aia_prep` performs standard calibration: flat-fielding, spike removal, exposure normalization, roll correction, and co-registration.

```idl
; Basic calibration
read_sdo, file, index, data
aia_prep, index, data, oindex, odata

; With common options
aia_prep, index, data, oindex, odata, $
    /NORMALIZE, $      ; Normalize by exposure time (DN/s)
    /REGISTER, $       ; Co-register to common pointing
    /CUTOUT, $         ; Trim to 4096x4096 if oversized
    /UNKILL_PIX        ; Interpolate over hot/dead pixels

; Check calibration status
PRINT, 'Level:    ', oindex.lvl_num    ; Should be 1.5
PRINT, 'Normalized:', oindex.datamean   ; In DN/s
PRINT, 'Pixel size:', oindex.cdelt1     ; 0.6 arcsec exactly after /REGISTER
```

### What aia_prep Does

| Step | Description |
|------|-------------|
| Dark/pedestal subtraction | Remove CCD bias |
| Flat field correction | Correct pixel-to-pixel sensitivity |
| Spike removal | Remove cosmic ray hits |
| Bad pixel interpolation | Fix known hot/dead pixels |
| Exposure normalization | Convert to DN/s (with /NORMALIZE) |
| Roll correction | Correct spacecraft roll angle |
| Plate scale alignment | Set exact 0.6 arcsec/pixel (with /REGISTER) |
| Pointing update | Apply latest pointing calibration |

### Batch Calibration

```idl
; Calibrate many files efficiently
files = FILE_SEARCH('/data/aia/171/*.fits', COUNT=nf)
out_dir = '/data/aia/171/prepped/'
FILE_MKDIR, out_dir

FOR i = 0, nf-1 DO BEGIN
    read_sdo, files[i], index, data
    aia_prep, index, data, oindex, odata, /NORMALIZE, /REGISTER

    ; Write calibrated file
    outfile = out_dir + 'prepped_' + FILE_BASENAME(files[i])
    mwritefits, oindex, odata, OUTFILE=outfile

    IF (i MOD 100) EQ 0 THEN $
        PRINT, STRTRIM(i, 2) + '/' + STRTRIM(nf, 2) + ' done'
ENDFOR
```

---

## 4. AIA Response Functions

AIA response functions describe the sensitivity of each channel as a function of temperature. They are essential for temperature diagnostics.

### Loading Response Functions

```idl
; Get AIA temperature response functions
tresp = AIA_GET_RESPONSE(/TEMPERATURE, /DN)

; tresp structure:
;   tresp.logte  — log10(T) array (e.g., 4.0 to 8.0)
;   tresp.a94    — 94 A response [DN cm^5 s^-1 pixel^-1]
;   tresp.a131   — 131 A response
;   tresp.a171   — 171 A response
;   tresp.a193   — 193 A response
;   tresp.a211   — 211 A response
;   tresp.a304   — 304 A response
;   tresp.a335   — 335 A response

HELP, tresp, /STRUCTURE

; Plot response functions
WINDOW, 0, XSIZE=800, YSIZE=500
PLOT, tresp.logte, tresp.a171, /YLOG, $
    XTITLE='log T (K)', YTITLE='Response (DN cm!U5!N s!U-1!N pix!U-1!N)', $
    TITLE='AIA Temperature Response Functions', $
    YRANGE=[1e-28, 1e-23], XRANGE=[5.0, 8.0]
OPLOT, tresp.logte, tresp.a94, COLOR=50
OPLOT, tresp.logte, tresp.a131, COLOR=100
OPLOT, tresp.logte, tresp.a193, COLOR=150
OPLOT, tresp.logte, tresp.a211, COLOR=200
OPLOT, tresp.logte, tresp.a335, COLOR=250

; Add legend
XYOUTS, 7.5, 1e-24, '171', COLOR=255
XYOUTS, 7.5, 5e-25, '94', COLOR=50
XYOUTS, 7.5, 2e-25, '131', COLOR=100
XYOUTS, 7.5, 1e-25, '193', COLOR=150
XYOUTS, 7.5, 5e-26, '211', COLOR=200
XYOUTS, 7.5, 2e-26, '335', COLOR=250
```

### Time-Dependent Response

```idl
; AIA response degrades over time (UV contamination, CCD aging)
; Get response for a specific date
tresp_2024 = AIA_GET_RESPONSE(/TEMPERATURE, /DN, TIMEDEPEND_DATE='2024-01-15')
tresp_2011 = AIA_GET_RESPONSE(/TEMPERATURE, /DN, TIMEDEPEND_DATE='2011-01-01')

; Compare — 2024 response is lower due to degradation
PRINT, 'Peak 171 response (2011): ', MAX(tresp_2011.a171)
PRINT, 'Peak 171 response (2024): ', MAX(tresp_2024.a171)
PRINT, 'Ratio: ', MAX(tresp_2024.a171) / MAX(tresp_2011.a171)
```

---

## 5. Multi-Wavelength Analysis

### Channel Ratios for Temperature Estimation

```idl
; Quick temperature estimate from channel ratio
; Using 193/171 ratio (sensitive to ~1-3 MK plasma)

; Read co-temporal images
read_sdo, 'aia_171.fits', idx171, dat171
read_sdo, 'aia_193.fits', idx193, dat193

; Calibrate
aia_prep, idx171, dat171, oi171, od171, /NORMALIZE, /REGISTER
aia_prep, idx193, dat193, oi193, od193, /NORMALIZE, /REGISTER

; Compute ratio (avoid division by zero)
threshold = 10.0  ; DN/s minimum
mask = (od171 GT threshold) AND (od193 GT threshold)
ratio = FLTARR(4096, 4096)
good = WHERE(mask, ngood)
IF ngood GT 0 THEN ratio[good] = od193[good] / od171[good]

; Display
WINDOW, 0, XSIZE=512, YSIZE=512
LOADCT, 33
TV, BYTSCL(REBIN(ALOG10(ratio > 0.01), 512, 512), MIN=-1, MAX=2)
```

### Creating Composite Images

```idl
; Three-color composite: 171 (green), 193 (blue), 304 (red)
read_sdo, 'aia_171.fits', i171, d171
read_sdo, 'aia_193.fits', i193, d193
read_sdo, 'aia_304.fits', i304, d304

aia_prep, i171, d171, oi171, od171, /NORMALIZE, /REGISTER
aia_prep, i193, d193, oi193, od193, /NORMALIZE, /REGISTER
aia_prep, i304, d304, oi304, od304, /NORMALIZE, /REGISTER

; Scale each channel (adjust for dynamic range)
r = BYTSCL(ALOG10(od304 > 1), MIN=0, MAX=3.5)   ; 304 -> Red
g = BYTSCL(ALOG10(od171 > 1), MIN=0, MAX=3.5)   ; 171 -> Green
b = BYTSCL(ALOG10(od193 > 1), MIN=0, MAX=3.5)   ; 193 -> Blue

; Rebin for display
sz = 1024
r = REBIN(r, sz, sz)
g = REBIN(g, sz, sz)
b = REBIN(b, sz, sz)

; Display as true-color image
WINDOW, 0, XSIZE=sz, YSIZE=sz
TV, [[[r]], [[g]], [[b]]], TRUE=3

; Save to PNG
WRITE_PNG, 'aia_composite.png', r, g, b
```

### AIA Color Tables

```idl
; AIA provides instrument-specific color tables
aia_lct, WAVE=171, /LOAD    ; Green (Fe IX)
aia_lct, WAVE=193, /LOAD    ; Bronze (Fe XII)
aia_lct, WAVE=304, /LOAD    ; Red (He II)
aia_lct, WAVE=94, /LOAD     ; Green (Fe XVIII)
aia_lct, WAVE=131, /LOAD    ; Teal (Fe VIII/XXI)
aia_lct, WAVE=211, /LOAD    ; Purple (Fe XIV)
aia_lct, WAVE=335, /LOAD    ; Blue (Fe XVI)
```

---

## 6. Differential Emission Measure (DEM) Basics

The DEM describes the amount of plasma at each temperature along the line of sight. It connects the observed intensities to the underlying temperature distribution.

### DEM Concept

```
I_channel = integral{ R_channel(T) * DEM(T) * dT }

where:
  I_channel = observed intensity (DN/s/pixel)
  R_channel(T) = temperature response function
  DEM(T) = differential emission measure [cm^-5 K^-1]
  T = temperature
```

### Simple Two-Temperature Model

```idl
; Estimate DEM at two temperatures using two channels
; This is a very simplified approach

tresp = AIA_GET_RESPONSE(/TEMPERATURE, /DN)

; Find response at specific temperatures
t1_idx = VALUE_LOCATE(tresp.logte, 5.9)  ; ~0.8 MK (171 peak)
t2_idx = VALUE_LOCATE(tresp.logte, 6.2)  ; ~1.6 MK (193 peak)

; Response matrix: R[channel, temperature]
R = DBLARR(2, 2)
R[0, 0] = tresp.a171[t1_idx]  ; 171 response at T1
R[0, 1] = tresp.a171[t2_idx]  ; 171 response at T2
R[1, 0] = tresp.a193[t1_idx]  ; 193 response at T1
R[1, 1] = tresp.a193[t2_idx]  ; 193 response at T2

; For a single pixel with observed intensities:
I_obs = [500.0D, 800.0D]  ; [171 DN/s, 193 DN/s]

; Solve: I = R * DEM  ->  DEM = R^(-1) * I
R_inv = INVERT(R)
DEM = R_inv ## I_obs

PRINT, 'DEM at log T=5.9: ', DEM[0], ' cm^-5 K^-1'
PRINT, 'DEM at log T=6.2: ', DEM[1], ' cm^-5 K^-1'
```

### Using SSW DEM Routines

```idl
; Several DEM inversion methods are available in SSW:

; 1. xrt_dem_iterative2 (from Hinode/XRT but works with AIA)
; 2. aia_sparse_em_solve (Cheung et al.)
; 3. demreg (Hannah & Kontar regularized DEM)

; Example with AIA data:
; Prepare 6-channel data cube for one pixel or region
channels = [94, 131, 171, 193, 211, 335]
n_chan = N_ELEMENTS(channels)
I_obs = DBLARR(n_chan)
I_err = DBLARR(n_chan)

; Fill with observed values (DN/s) and uncertainties
I_obs = [5.0, 20.0, 500.0, 800.0, 200.0, 30.0]
I_err = SQRT(I_obs > 1.0)  ; Poisson noise estimate

; Get response functions
tresp = AIA_GET_RESPONSE(/TEMPERATURE, /DN)
logte = tresp.logte
nte = N_ELEMENTS(logte)

; Build response matrix
resp_matrix = DBLARR(n_chan, nte)
resp_matrix[0, *] = tresp.a94
resp_matrix[1, *] = tresp.a131
resp_matrix[2, *] = tresp.a171
resp_matrix[3, *] = tresp.a193
resp_matrix[4, *] = tresp.a211
resp_matrix[5, *] = tresp.a335

; Simple positive least-squares DEM inversion
; (More sophisticated methods use regularization)
```

### AIA_BP_ESTIMATE — Quick Temperature Estimate

```idl
; aia_bp_estimate provides a quick "best-fit" temperature
; using the ratio of observed to predicted intensities

; For a set of 6-channel intensities:
I_obs = [5.0, 20.0, 500.0, 800.0, 200.0, 30.0]

; This routine finds the isothermal temperature that best
; matches the observed channel ratios
aia_bp_estimate, I_obs, tmap, emmap

PRINT, 'Best-fit log T: ', ALOG10(tmap)
PRINT, 'Best-fit EM:    ', emmap, ' cm^-5'
```

---

## 7. Region-of-Interest Analysis

```idl
; Extract and analyze a sub-region of an AIA image

; Read and calibrate
read_sdo, 'aia_171.fits', index, data
aia_prep, index, data, oindex, odata, /NORMALIZE, /REGISTER

; Define ROI in arcsec
xcen = -200.0   ; arcsec (active region)
ycen = 300.0
half_fov = 150.0  ; arcsec

; Convert to pixels
wcs = FITSHEAD2WCS(oindex)
pix_ll = WCS_GET_PIXEL(wcs, [xcen-half_fov, ycen-half_fov])
pix_ur = WCS_GET_PIXEL(wcs, [xcen+half_fov, ycen+half_fov])

x0 = ROUND(pix_ll[0]) > 0
y0 = ROUND(pix_ll[1]) > 0
x1 = ROUND(pix_ur[0]) < (oindex.naxis1-1)
y1 = ROUND(pix_ur[1]) < (oindex.naxis2-1)

; Extract sub-image
subimg = odata[x0:x1, y0:y1]
PRINT, 'ROI size: ', SIZE(subimg, /DIMENSIONS)

; Compute light curve for ROI over a time series
files = FILE_SEARCH('/data/aia/171/*.fits', COUNT=nf)
lightcurve = DBLARR(nf)
times = DBLARR(nf)

FOR i = 0, nf-1 DO BEGIN
    read_sdo, files[i], idx, dat
    aia_prep, idx, dat, oi, od, /NORMALIZE, /REGISTER
    lightcurve[i] = MEAN(od[x0:x1, y0:y1])
    times[i] = ANYTIM(oi.date_obs)
ENDFOR

; Plot light curve
t_min = (times - times[0]) / 60.0  ; minutes
PLOT, t_min, lightcurve, $
    XTITLE='Time (min)', YTITLE='Mean Intensity (DN/s)', $
    TITLE='AIA 171 ROI Light Curve'
```

---

## Summary

| Topic | Key Routines | Purpose |
|-------|-------------|---------|
| Data I/O | `read_sdo` | Read AIA FITS files |
| Calibration | `aia_prep` | Standard pipeline calibration |
| Response | `aia_get_response` | Temperature response functions |
| Color tables | `aia_lct` | Channel-specific color maps |
| Composites | Manual RGB | Multi-wavelength visualization |
| DEM | `aia_bp_estimate`, DEM codes | Temperature diagnostics |
| ROI | WCS pixel conversion | Sub-region extraction |

---

**Previous**: [SolarSoft Framework](./05_SolarSoft_Framework.md) | **Next**: [SDO/HMI Analysis](./07_SDO_HMI_Analysis.md)
