# 07. SDO/HMI Analysis

**Previous**: [SDO/AIA Analysis](./06_SDO_AIA_Analysis.md) | **Next**: [GOES and RHESSI](./08_GOES_and_RHESSI.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand SDO/HMI data products (magnetograms, Dopplergrams, continuum)
2. Read and calibrate HMI data with `read_sdo` and `hmi_prep`
3. Analyze line-of-sight and vector magnetic field data
4. Compute total unsigned magnetic flux and flux imbalance
5. Create Carrington synoptic maps from HMI data
6. Apply coordinate transforms for magnetic field projection corrections

---

## 1. SDO/HMI Overview

The Helioseismic and Magnetic Imager (HMI) on SDO measures the Sun's photospheric magnetic field and velocity field.

### HMI Data Products

| Product | Keyword | Cadence | Resolution | Description |
|---------|---------|---------|------------|-------------|
| Line-of-sight magnetogram | `hmi.M_720s` | 720 s | 0.5"/pix | LOS magnetic field (Gauss) |
| LOS magnetogram (45s) | `hmi.M_45s` | 45 s | 0.5"/pix | High-cadence LOS B |
| Continuum intensity | `hmi.Ic_720s` | 720 s | 0.5"/pix | White-light intensity |
| Dopplergram | `hmi.V_720s` | 720 s | 0.5"/pix | LOS velocity (m/s) |
| Vector magnetogram | `hmi.B_720s` | 720 s | 0.5"/pix | Full B vector (Br, Btheta, Bphi) |
| SHARP data | `hmi.sharp_720s` | 720 s | 0.5"/pix | Active region patches with keywords |

### HMI Specifications

- **Spatial resolution**: 0.505 arcsec/pixel (4096 x 4096 pixels)
- **Field of view**: Full solar disk
- **Spectral line**: Fe I 6173 A
- **Noise level**: ~7 G (LOS), ~100 G (transverse)

---

## 2. Reading HMI Data

```idl
; Read HMI line-of-sight magnetogram
file = 'hmi.M_720s_2024.01.15_12_00_00_TAI.fits'
read_sdo, file, index, data

; Examine header
PRINT, 'Type:      ', index.content     ; 'MAGNETOGRAM'
PRINT, 'Date:      ', index.date_obs
PRINT, 'Size:      ', index.naxis1, ' x ', index.naxis2
PRINT, 'Pixel:     ', index.cdelt1, ' arcsec/pixel'
PRINT, 'Units:     ', index.bunit       ; 'Gauss'

; Data range
PRINT, 'Min B_LOS: ', MIN(data), ' G'
PRINT, 'Max B_LOS: ', MAX(data), ' G'
```

### Reading Multiple HMI Products

```idl
; Read magnetogram, Dopplergram, and continuum
read_sdo, 'hmi_mag.fits', idx_mag, mag
read_sdo, 'hmi_dop.fits', idx_dop, dop
read_sdo, 'hmi_ic.fits', idx_ic, ic

; Quick display
WINDOW, 0, XSIZE=1536, YSIZE=512
!P.MULTI = [0, 3, 1]
LOADCT, 0
TV, BYTSCL(REBIN(mag, 512, 512), MIN=-200, MAX=200), 0
TV, BYTSCL(REBIN(dop, 512, 512), MIN=-3000, MAX=3000), 1
TV, BYTSCL(REBIN(ic, 512, 512)), 2
!P.MULTI = 0
```

---

## 3. HMI Calibration with hmi_prep

```idl
; Basic HMI calibration
read_sdo, file, index, data
hmi_prep, index, data, oindex, odata

; HMI_PREP does:
; - Bad pixel correction
; - Cosmic ray removal
; - Roll angle correction
; - Plate scale normalization (exact 0.505"/pixel)
; - Pointing update

; For magnetic field analysis, often minimal prep is needed
; since HMI data products are already pipeline-processed
```

---

## 4. Line-of-Sight Magnetic Field Analysis

### Display Magnetogram

```idl
; Read and display magnetogram
read_sdo, 'hmi_mag.fits', index, mag

; Bipolar display: blue=negative, red=positive
LOADCT, 33  ; Blue-Red color table
WINDOW, 0, XSIZE=1024, YSIZE=1024
TV, BYTSCL(REBIN(mag, 1024, 1024), MIN=-500, MAX=500)

; Add solar limb
theta = FINDGEN(361) * !DTOR
rsun_pix = index.rsun_obs / index.cdelt1  ; Solar radius in pixels
xc = index.crpix1 / 4.0  ; Scaled for display
yc = index.crpix2 / 4.0
PLOTS, xc + rsun_pix/4.0*COS(theta), $
       yc + rsun_pix/4.0*SIN(theta), /DEVICE, COLOR=255
```

### Total Unsigned Magnetic Flux

```idl
; Compute total unsigned magnetic flux for an active region
; Phi = sum(|B_LOS| * dA)

; Solar parameters
rsun_cm = 6.957e10  ; Solar radius in cm
dist_sun = index.dsun_obs * 100.0  ; Sun-Earth distance in cm

; Pixel area in cm^2
cdelt_rad = index.cdelt1 * !DTOR / 3600.0  ; arcsec to radians
pixel_area = (cdelt_rad * dist_sun)^2  ; cm^2

; Define active region ROI (in pixels)
x0 = 1800 & x1 = 2200
y0 = 2000 & y1 = 2400
roi_mag = mag[x0:x1, y0:y1]

; Total unsigned flux
flux_unsigned = TOTAL(ABS(roi_mag)) * pixel_area
PRINT, 'Total unsigned flux: ', flux_unsigned, ' Mx'

; Positive and negative flux
pos_mask = WHERE(roi_mag GT 0, n_pos)
neg_mask = WHERE(roi_mag LT 0, n_neg)

flux_pos = (n_pos GT 0) ? TOTAL(roi_mag[pos_mask]) * pixel_area : 0.0
flux_neg = (n_neg GT 0) ? TOTAL(ABS(roi_mag[neg_mask])) * pixel_area : 0.0

PRINT, 'Positive flux: ', flux_pos, ' Mx'
PRINT, 'Negative flux: ', flux_neg, ' Mx'
PRINT, 'Imbalance:     ', (flux_pos - flux_neg) / (flux_pos + flux_neg)
```

### Flux Time Series

```idl
; Track magnetic flux evolution of an active region
files = FILE_SEARCH('/data/hmi/mag/*.fits', COUNT=nf)

flux_pos = DBLARR(nf)
flux_neg = DBLARR(nf)
times = DBLARR(nf)

FOR i = 0, nf-1 DO BEGIN
    read_sdo, files[i], idx, dat

    ; Extract ROI
    roi = dat[x0:x1, y0:y1]
    times[i] = ANYTIM(idx.date_obs)

    ; Compute flux
    pos = WHERE(roi GT 10, np)   ; Threshold to reduce noise
    neg = WHERE(roi LT -10, nn)
    IF np GT 0 THEN flux_pos[i] = TOTAL(roi[pos]) * pixel_area
    IF nn GT 0 THEN flux_neg[i] = TOTAL(ABS(roi[neg])) * pixel_area
ENDFOR

; Plot
t_hr = (times - times[0]) / 3600.0
PLOT, t_hr, flux_pos * 1e-22, $
    XTITLE='Time (hours)', $
    YTITLE='Flux (10!U22!N Mx)', $
    TITLE='Magnetic Flux Evolution', $
    YRANGE=[0, MAX([flux_pos, flux_neg]) * 1.1e-22]
OPLOT, t_hr, flux_neg * 1e-22, LINESTYLE=2
```

---

## 5. Vector Magnetic Field Data

### Reading Vector Magnetograms

```idl
; HMI vector magnetic field: hmi.B_720s
; Three components: Br (radial), Btheta (colatitudinal), Bphi (azimuthal)
; Or equivalently: field strength, inclination, azimuth

; Read SHARP (Space-weather HMI Active Region Patches) data
; SHARP data provides pre-cut AR patches with derived keywords
file_br = 'hmi.sharp_720s_Br.fits'
file_bt = 'hmi.sharp_720s_Bt.fits'
file_bp = 'hmi.sharp_720s_Bp.fits'

read_sdo, file_br, idx_br, Br
read_sdo, file_bt, idx_bt, Bt
read_sdo, file_bp, idx_bp, Bp

PRINT, 'AR NOAA: ', idx_br.noaa_ar
PRINT, 'HARPNUM: ', idx_br.harpnum
```

### HMI Disambiguation

```idl
; The transverse field has a 180-degree ambiguity in azimuth
; HMI pipeline resolves this using the Minimum Energy method
; The disambiguated data is in hmi.B_720s (already resolved)

; For manual disambiguation (research-level):
; - Potential field comparison
; - Minimum energy algorithm
; - Acute angle method

; The SHARP data products are already disambiguated
```

### Magnetic Field Vector Display

```idl
; Display vector magnetogram with arrows
; Br: background image, Bt/Bp: arrow overlay

; Display radial field
WINDOW, 0, XSIZE=800, YSIZE=600
LOADCT, 33
TV, BYTSCL(CONGRID(Br, 800, 600), MIN=-1000, MAX=1000)

; Overlay transverse field arrows
; Subsample for clarity
step = 20
nx = (SIZE(Br, /DIMENSIONS))[0]
ny = (SIZE(Br, /DIMENSIONS))[1]

FOR ix = 0, nx-1, step DO BEGIN
    FOR iy = 0, ny-1, step DO BEGIN
        bt_val = Bt[ix, iy]
        bp_val = Bp[ix, iy]
        b_trans = SQRT(bt_val^2 + bp_val^2)

        IF b_trans GT 100 THEN BEGIN  ; Threshold
            ; Normalize arrow length
            scale = 0.5 * step
            dx = bp_val / b_trans * scale
            dy = -bt_val / b_trans * scale

            ; Convert to display coordinates
            x_disp = FLOAT(ix) / nx * 800
            y_disp = FLOAT(iy) / ny * 600

            ARROW, x_disp, y_disp, x_disp + dx, y_disp + dy, $
                /DEVICE, HSIZE=5, COLOR=255, THICK=1
        ENDIF
    ENDFOR
ENDFOR
```

---

## 6. Projection Corrections

### LOS to Radial Field Correction

```idl
; The LOS magnetogram measures B_LOS = B_r * cos(theta) + B_h * sin(theta)
; Near disk center, B_LOS ~ B_r
; Near the limb, the correction becomes important

; Simple mu-correction: B_r ~ B_LOS / mu
; where mu = cos(theta) = sqrt(1 - rho^2)

; Create mu map
nx = index.naxis1 & ny = index.naxis2
x = (FINDGEN(nx) - index.crpix1) * index.cdelt1  ; arcsec
y = (FINDGEN(ny) - index.crpix2) * index.cdelt2  ; arcsec
xx = x # REPLICATE(1.0, ny)
yy = REPLICATE(1.0, nx) # y

rho = SQRT(xx^2 + yy^2) / index.rsun_obs  ; Normalized distance from center
mu = SQRT(1.0 - rho^2 < 1.0)              ; cos(heliocentric angle)

; Apply correction (only where mu > some threshold)
Br_corrected = mag * 0.0  ; Initialize
good = WHERE(mu GT 0.3, ngood)  ; Avoid limb
IF ngood GT 0 THEN Br_corrected[good] = mag[good] / mu[good]

; Mask off-disk
disk = WHERE(rho GE 1.0, ndisk)
IF ndisk GT 0 THEN Br_corrected[disk] = 0.0
```

---

## 7. Carrington Synoptic Maps

### Building a Synoptic Map

```idl
; A Carrington synoptic map combines central-meridian strips from
; full-disk magnetograms over one solar rotation (~27.3 days)

; Parameters
n_lon = 3600   ; 0.1 degree resolution
n_lat = 1800   ; 0.1 degree resolution
synoptic = FLTARR(n_lon, n_lat)

; For each magnetogram in a Carrington rotation:
files = FILE_SEARCH('/data/hmi/mag/CR2277/*.fits', COUNT=nf)

FOR i = 0, nf-1 DO BEGIN
    read_sdo, files[i], idx, dat

    ; Get Carrington longitude of central meridian
    carr_lon_cm = TIM2CARR(idx.date_obs, /DC)
    lon_idx = ROUND(carr_lon_cm * 10) MOD n_lon  ; 0.1 deg bins

    ; Extract central meridian strip
    cx = ROUND(idx.crpix1)
    strip = dat[cx-2:cx+2, *]  ; 5-pixel wide strip
    strip_avg = MEAN(strip, DIMENSION=1)  ; Average across strip

    ; Map to latitude
    y_pix = FINDGEN(idx.naxis2)
    y_arcsec = (y_pix - idx.crpix2) * idx.cdelt2
    lat_rad = ASIN(y_arcsec / idx.rsun_obs < 1.0 > (-1.0))
    lat_deg = lat_rad * !RADEG

    ; Interpolate to synoptic grid
    lat_grid = FINDGEN(n_lat) * 180.0 / n_lat - 90.0
    strip_interp = INTERPOL(strip_avg, lat_deg, lat_grid)

    ; Apply mu-correction
    mu = COS(lat_grid * !DTOR)
    good = WHERE(ABS(mu) GT 0.2)
    IF N_ELEMENTS(good) GT 1 THEN $
        strip_interp[good] = strip_interp[good] / mu[good]

    ; Insert into synoptic map
    synoptic[lon_idx, *] = strip_interp
ENDFOR

; Display synoptic map
WINDOW, 0, XSIZE=900, YSIZE=450
LOADCT, 33
TV, BYTSCL(CONGRID(synoptic, 900, 450), MIN=-50, MAX=50)
```

### Using Pre-Made Synoptic Maps

```idl
; JSOC provides pre-computed synoptic maps:
; hmi.Synoptic_Mr_720s — radial field synoptic
; hmi.Synoptic_Ml_720s — LOS field synoptic

; Download from JSOC
; http://jsoc.stanford.edu/ajax/lookdata.html?ds=hmi.Synoptic_Mr_720s[2277]

; Read synoptic FITS file
synoptic = READFITS('hmi_synoptic_mr_2277.fits', header)
PRINT, SIZE(synoptic, /DIMENSIONS)  ; 3600 x 1440 typically
```

---

## 8. SHARP Keywords for Space Weather

```idl
; SHARP data includes pre-computed magnetic field parameters
; useful for space weather analysis

read_sdo, 'hmi_sharp.fits', idx, data

; Key SHARP keywords:
PRINT, 'USFLUX:  ', idx.usflux    ; Total unsigned flux (Mx)
PRINT, 'MEANGBT: ', idx.meangbt   ; Mean gradient of total field (G/Mm)
PRINT, 'MEANJZD: ', idx.meanjzd   ; Mean vertical current density (mA/m^2)
PRINT, 'TOTUSJZ: ', idx.totusjz   ; Total unsigned vertical current (A)
PRINT, 'MEANALP: ', idx.meanalp   ; Mean twist parameter alpha
PRINT, 'MEANSHR: ', idx.meanshr   ; Mean shear angle (degrees)
PRINT, 'SHRGT45: ', idx.shrgt45   ; Fraction of pixels with shear > 45 deg
PRINT, 'AREA:    ', idx.area_acr  ; Active region area (uH)

; These parameters are used in flare prediction models
```

---

## Summary

| Topic | Key Routines | Purpose |
|-------|-------------|---------|
| Data I/O | `read_sdo` | Read HMI FITS files |
| Calibration | `hmi_prep` | Standard calibration |
| LOS field | Direct analysis | Magnetogram analysis |
| Vector field | SHARP data | Full magnetic vector |
| Flux calculation | `TOTAL`, pixel area | Unsigned flux |
| Projection | mu-correction | LOS-to-radial conversion |
| Synoptic maps | Central meridian strips | Carrington maps |
| Space weather | SHARP keywords | AR characterization |

---

**Previous**: [SDO/AIA Analysis](./06_SDO_AIA_Analysis.md) | **Next**: [GOES and RHESSI](./08_GOES_and_RHESSI.md)
