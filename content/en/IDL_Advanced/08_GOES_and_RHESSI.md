# 08. GOES and RHESSI

**Previous**: [SDO/HMI Analysis](./07_SDO_HMI_Analysis.md) | **Next**: [Spectral Analysis](./09_Spectral_Analysis.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Read and plot GOES X-ray light curves with flare class labels
2. Work with the GOES event list and flare catalog
3. Understand RHESSI imaging and spectroscopy concepts
4. Create RHESSI images with `hsi_image`
5. Perform RHESSI spectral analysis with OSPEX

---

## 1. GOES X-ray Data

The GOES (Geostationary Operational Environmental Satellite) series provides continuous monitoring of solar X-ray flux in two channels.

### GOES X-ray Channels

| Channel | Wavelength | Energy | Sensitivity |
|---------|-----------|--------|-------------|
| Long | 1-8 A | 1.55-12.4 keV | B-class and above |
| Short | 0.5-4 A | 3.1-24.8 keV | C-class and above |

### Flare Classification

| Class | Flux (W/m^2, 1-8 A) | Typical Duration |
|-------|---------------------|------------------|
| A | < 1e-7 | — |
| B | 1e-7 to 1e-6 | — |
| C | 1e-6 to 1e-5 | Minutes |
| M | 1e-5 to 1e-4 | Tens of minutes |
| X | > 1e-4 | Hours |

---

## 2. Reading GOES Data

### Using rd_goes

```idl
; Read GOES data for a specific time range
rd_goes, data, tarray, $
    TSTART='2024-01-01T00:00:00', $
    TEND='2024-01-01T23:59:59', $
    SAT=16, $              ; GOES-16 (or 17, 18)
    /ONE_MINUTE            ; 1-minute averaged data

; data: structure with flux arrays
; data.lo — long channel (1-8 A) flux [W/m^2]
; data.hi — short channel (0.5-4 A) flux [W/m^2]
; tarray  — time array

PRINT, 'Time range: ', ANYTIM(tarray[0], /CCSDS), $
       ' to ', ANYTIM(tarray[-1], /CCSDS)
PRINT, 'Max 1-8 A flux: ', MAX(data.lo), ' W/m^2'
```

### Alternative: GOES_CHIANTI_TEM

```idl
; GOES temperature and emission measure from two-channel analysis
goes_chianti_tem, date='2024-01-01', $
    tstart='2024-01-01 10:00', tend='2024-01-01 14:00', $
    tem=tem, em=em, fl_gos=fl, $
    /ONE_MINUTE

PRINT, 'Temperature: ', tem, ' MK'
PRINT, 'Emission measure: ', em, ' cm^-3'
```

---

## 3. Plotting GOES Light Curves

### Basic Light Curve

```idl
; Read data
rd_goes, data, tarray, $
    TSTART='2024-01-01', TEND='2024-01-02', /ONE_MINUTE

; Convert time to hours for plotting
t_hours = (tarray - tarray[0]) / 3600.0

; Plot
WINDOW, 0, XSIZE=900, YSIZE=500
PLOT, t_hours, data.lo, /YLOG, $
    XTITLE='Time (hours from start)', $
    YTITLE='Flux (W m!U-2!N)', $
    TITLE='GOES 1-8 A X-ray Flux', $
    YRANGE=[1e-9, 1e-3], YSTYLE=1, $
    XSTYLE=1
OPLOT, t_hours, data.hi, LINESTYLE=2, COLOR=200
```

### With Flare Class Labels

```idl
; Add flare class labels on the right axis
WINDOW, 0, XSIZE=900, YSIZE=500

; Plot with room for labels
PLOT, t_hours, data.lo, /YLOG, $
    YRANGE=[1e-9, 1e-3], YSTYLE=9, $  ; YSTYLE=9: exact range, suppress right axis
    XSTYLE=1, $
    XTITLE='Time (hours from start)', $
    YTITLE='GOES 1-8 A Flux (W m!U-2!N)', $
    TITLE='GOES X-ray Light Curve', $
    POSITION=[0.12, 0.12, 0.88, 0.92]

; Draw flare class boundaries and labels
classes = ['A', 'B', 'C', 'M', 'X']
levels = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]

FOR i = 0, N_ELEMENTS(levels)-1 DO BEGIN
    PLOTS, [0, MAX(t_hours)], [levels[i], levels[i]], $
        LINESTYLE=1, COLOR=150
    XYOUTS, MAX(t_hours)*1.02, levels[i], classes[i], $
        CHARSIZE=1.2, COLOR=200
ENDFOR

; Add short channel
OPLOT, t_hours, data.hi, COLOR=100, LINESTYLE=2

; Legend
XYOUTS, 0.15, 0.88, '1-8 A', /NORMAL, CHARSIZE=0.9
XYOUTS, 0.15, 0.85, '0.5-4 A', /NORMAL, CHARSIZE=0.9, COLOR=100
```

### Publication-Quality GOES Plot

```idl
SET_PLOT, 'PS'
DEVICE, FILENAME='goes_lightcurve.eps', /ENCAPSULATED, $
    XSIZE=18, YSIZE=10, /COLOR, BITS=8

!P.THICK = 3
!P.CHARTHICK = 2
!P.CHARSIZE = 1.0
!X.THICK = 2
!Y.THICK = 2

; Use UTC time axis with UTPLOT
utplot, tarray, data.lo, $
    YTITLE='GOES 1-8 A Flux (W m!U-2!N)', $
    TITLE='GOES X-ray Light Curve', $
    /YLOG, YRANGE=[1e-8, 1e-3], $
    TIMERANGE=[tarray[0], tarray[-1]]

; utplot is an SSW routine that formats the x-axis with UT times

DEVICE, /CLOSE
SET_PLOT, 'X'
!P.THICK = 0 & !P.CHARTHICK = 0 & !P.CHARSIZE = 0
!X.THICK = 0 & !Y.THICK = 0
```

---

## 4. GOES Flare Catalog

```idl
; Access the GOES flare event list
; Using SSW event list tools

; Search for flares in a time range
search_result = SSW_HER_QUERY( $
    '2024-01-01T00:00:00', '2024-01-31T23:59:59', $
    /GOES_FLARE)

; Or use the older event list
rd_gev, gev, TSTART='2024-01-01', TEND='2024-01-31'

; gev structure contains:
;   gev.st$date, gev.st$time  — start time
;   gev.en$date, gev.en$time  — end time
;   gev.pk$date, gev.pk$time  — peak time
;   gev.class                  — flare class (e.g., 'M1.5')
;   gev.noaa                  — NOAA AR number
;   gev.loc                   — heliographic location

; Print flare list
FOR i = 0, N_ELEMENTS(gev)-1 DO BEGIN
    PRINT, gev[i].st$date, ' ', gev[i].st$time, $
           ' Class: ', gev[i].class, $
           ' AR: ', gev[i].noaa, $
           ' Loc: ', gev[i].loc
ENDFOR
```

### Flare Detection from GOES Data

```idl
; Simple flare detection algorithm
; Detect when flux exceeds C-class threshold (1e-6 W/m^2)

rd_goes, data, tarray, TSTART='2024-01-15', TEND='2024-01-16', /ONE_MIN

threshold = 1e-6  ; C-class
above = WHERE(data.lo GT threshold, n_above)

IF n_above GT 0 THEN BEGIN
    ; Find contiguous intervals (flares)
    breaks = WHERE(above[1:*] - above[0:n_above-2] GT 1, n_breaks)

    IF n_breaks GT 0 THEN BEGIN
        starts = [above[0], above[breaks+1]]
        ends = [above[breaks], above[n_above-1]]

        PRINT, n_breaks+1, ' flares detected above C-class'
        FOR i = 0, N_ELEMENTS(starts)-1 DO BEGIN
            peak_idx = starts[i] + $
                WHERE(data.lo[starts[i]:ends[i]] EQ $
                MAX(data.lo[starts[i]:ends[i]]))
            peak_idx = peak_idx[0]
            PRINT, 'Flare ', i+1, ': Peak at ', $
                ANYTIM(tarray[peak_idx], /CCSDS), $
                ' Flux: ', data.lo[peak_idx], ' W/m^2'
        ENDFOR
    ENDIF
ENDIF
```

---

## 5. RHESSI Overview

The Reuven Ramaty High Energy Solar Spectroscopic Imager (RHESSI, 2002-2018) provided the first high-resolution hard X-ray and gamma-ray imaging of solar flares.

### RHESSI Capabilities

| Feature | Value |
|---------|-------|
| Energy range | 3 keV - 17 MeV |
| Spectral resolution | ~1 keV (FWHM at 100 keV) |
| Angular resolution | ~2.3 arcsec (finest grids) |
| Imaging method | Rotating Modulation Collimators (RMC) |
| Detectors | 9 Ge detectors |

### RHESSI Data Products

```idl
; RHESSI data is accessed through SSW
; Key data types:
; - Eventlist: photon-by-photon data
; - Spectrum: energy-binned count rates
; - Image: reconstructed images
; - Lightcurve: time profiles in energy bands

; Check data availability
hsi_obs_summary, '2024-01-15'  ; Show observation summary
```

---

## 6. RHESSI Imaging

### Creating RHESSI Images

```idl
; RHESSI imaging using the object-oriented framework
; Create an image object
obj = HSI_IMAGE()

; Set parameters
obj->SET, OBS_TIME_INTERVAL = $
    ANYTIM(['2024-01-15 12:00:00', '2024-01-15 12:02:00'])
obj->SET, IMAGE_DIM = [128, 128]     ; Image size in pixels
obj->SET, PIXEL_SIZE = [2.0, 2.0]    ; Arcsec per pixel
obj->SET, ENERGY_BAND = [6.0, 12.0]  ; Energy range in keV

; Select imaging algorithm
; Options: CLEAN, MEM_NJIT, PIXON, BACK_PROJECTION, VIS_FWDFIT
obj->SET, IMAGE_ALGORITHM = 'CLEAN'

; Select detectors and grids
obj->SET, DET_INDEX_MASK = $
    REPLICATE(1, 9)  ; Use all 9 detectors

; Create the image
image = obj->GETDATA()
PRINT, SIZE(image, /DIMENSIONS)

; Display
LOADCT, 3
WINDOW, 0, XSIZE=512, YSIZE=512
TV, BYTSCL(CONGRID(image, 512, 512))

; Get coordinate information
xyoffset = obj->GET(/XYOFFSET)
pixel_size = obj->GET(/PIXEL_SIZE)
PRINT, 'Image center: ', xyoffset, ' arcsec'

; Clean up
OBJ_DESTROY, obj
```

### RHESSI Image Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| BACK_PROJECTION | Simple back-projection | Quick-look |
| CLEAN | Iterative deconvolution | General use |
| MEM_NJIT | Maximum Entropy | Extended sources |
| PIXON | Pixon method | Complex morphology |
| VIS_FWDFIT | Visibility forward fitting | Simple sources |

### Multi-Energy Band Images

```idl
; Create images at multiple energy bands
energy_bands = [[6, 12], [12, 25], [25, 50], [50, 100]]
n_bands = (SIZE(energy_bands, /DIMENSIONS))[1]

!P.MULTI = [0, 2, 2]
FOR i = 0, n_bands-1 DO BEGIN
    obj = HSI_IMAGE()
    obj->SET, OBS_TIME_INTERVAL = $
        ANYTIM(['2024-01-15 12:00:00', '2024-01-15 12:02:00'])
    obj->SET, ENERGY_BAND = REFORM(energy_bands[*, i])
    obj->SET, IMAGE_ALGORITHM = 'CLEAN'
    obj->SET, IMAGE_DIM = [64, 64]
    obj->SET, PIXEL_SIZE = [2.0, 2.0]

    image = obj->GETDATA()

    LOADCT, 3
    TV, BYTSCL(CONGRID(image, 256, 256))
    XYOUTS, 0.1, 0.9 - i*0.5, $
        STRING(energy_bands[0,i], energy_bands[1,i], $
        FORMAT='(I0, "-", I0, " keV")'), /NORMAL

    OBJ_DESTROY, obj
ENDFOR
!P.MULTI = 0
```

---

## 7. RHESSI Spectral Analysis with OSPEX

OSPEX (Object Spectral Executive) is the standard tool for RHESSI spectral fitting.

### Basic OSPEX Usage

```idl
; Create an OSPEX object
o = OSPEX()

; Load RHESSI spectral data
o->SET, SPEX_SPECFILE = 'hsi_spectrum_20240115_120000.fits'
o->SET, SPEX_DRMFILE = 'hsi_drm_20240115_120000.fits'

; Set time interval
o->SET, SPEX_ERANGE = [6.0, 100.0]  ; Energy range (keV)
o->SET, SPEX_TBAND = $
    ANYTIM(['2024-01-15 12:00:00', '2024-01-15 12:02:00'])

; Display the spectrum
o->PLOT_SPECTRUM

; Fit the spectrum
; Thermal + power-law model
o->SET, FIT_FUNCTION = 'vth+thick2'  ; Thermal + thick-target
o->SET, FIT_COMP_PARAMS = $
    [1.0, 1.5, 1.0, $  ; vth: EM, kT, abundance
     0.5, 5.0, 20.0, 100.0, 3e33]  ; thick2: parameters

; Perform the fit
o->DOFIT

; Get results
fit_params = o->GET(/SPEX_SUMM_PARAMS)
fit_chi2 = o->GET(/SPEX_SUMM_CHISQ)
PRINT, 'Temperature: ', fit_params[1], ' keV'
PRINT, 'EM:          ', fit_params[0], ' x 10^49 cm^-3'
PRINT, 'Chi-square:  ', fit_chi2

; Plot fit
o->PLOT_SPECTRUM, /FIT
```

### OSPEX Fit Functions

| Function | Description | Parameters |
|----------|-------------|------------|
| `vth` | Isothermal (CHIANTI) | EM, kT, abundance |
| `thick2` | Thick-target bremsstrahlung | Flux, spectral index, low/high E cutoff |
| `thin2` | Thin-target bremsstrahlung | Flux, spectral index, low/high E cutoff |
| `bpow` | Broken power-law | Flux, indices, break energy |
| `line` | Gaussian line | Area, center, width |

### Time-Resolved Spectroscopy

```idl
; Fit spectra at multiple time intervals
o = OSPEX()
o->SET, SPEX_SPECFILE = 'hsi_spectrum.fits'
o->SET, SPEX_DRMFILE = 'hsi_drm.fits'
o->SET, FIT_FUNCTION = 'vth+thick2'

; Define time bins
t_start = ANYTIM('2024-01-15 12:00:00')
dt = 20.0  ; 20-second intervals
n_intervals = 30

temperatures = FLTARR(n_intervals)
em_values = FLTARR(n_intervals)
chi2_values = FLTARR(n_intervals)

FOR i = 0, n_intervals-1 DO BEGIN
    t0 = t_start + i * dt
    t1 = t0 + dt

    o->SET, SPEX_TBAND = [t0, t1]

    ; Initial guess (or use previous fit as seed)
    IF i EQ 0 THEN $
        o->SET, FIT_COMP_PARAMS = [1.0, 1.5, 1.0, 0.5, 5.0, 20, 100, 3e33]

    o->DOFIT, /NOPLOT

    params = o->GET(/SPEX_SUMM_PARAMS)
    temperatures[i] = params[1]
    em_values[i] = params[0]
    chi2_values[i] = o->GET(/SPEX_SUMM_CHISQ)
ENDFOR

; Plot temperature evolution
time_axis = FINDGEN(n_intervals) * dt
PLOT, time_axis, temperatures, $
    XTITLE='Time (s)', YTITLE='Temperature (keV)', $
    TITLE='RHESSI Temperature Evolution'

OBJ_DESTROY, o
```

---

## 8. Combining GOES and RHESSI

```idl
; Plot GOES light curve with RHESSI energy bands overlaid

; Read GOES
rd_goes, goes_data, goes_time, $
    TSTART='2024-01-15 11:50', TEND='2024-01-15 12:30', /ONE_MIN

; Read RHESSI light curve
hsi_obj = HSI_LIGHTCURVE()
hsi_obj->SET, OBS_TIME_INTERVAL = $
    ANYTIM(['2024-01-15 11:50:00', '2024-01-15 12:30:00'])
hsi_obj->SET, ENERGY_BAND = [[6, 12], [12, 25], [25, 50]]
hsi_lc = hsi_obj->GETDATA()
hsi_time = hsi_obj->GETDATA(/TIME)

; Multi-panel plot
!P.MULTI = [0, 1, 2]

; GOES panel
utplot, goes_time, goes_data.lo, /YLOG, $
    YTITLE='Flux (W m!U-2!N)', $
    TITLE='GOES 1-8 A', $
    YRANGE=[1e-7, 1e-3]
OPLOT, goes_time, goes_data.hi, LINESTYLE=2

; RHESSI panel
; (Exact implementation depends on data format)

!P.MULTI = 0
OBJ_DESTROY, hsi_obj
```

---

## Summary

| Topic | Key Routines | Purpose |
|-------|-------------|---------|
| GOES data | `rd_goes` | Read X-ray light curves |
| GOES plotting | `utplot` | Time-axis formatted plots |
| Flare catalog | `rd_gev` | Access event list |
| GOES T/EM | `goes_chianti_tem` | Temperature analysis |
| RHESSI imaging | `hsi_image` | X-ray image reconstruction |
| RHESSI spectra | OSPEX | Spectral fitting |
| RHESSI light curves | `hsi_lightcurve` | Energy-resolved time profiles |

---

**Previous**: [SDO/HMI Analysis](./07_SDO_HMI_Analysis.md) | **Next**: [Spectral Analysis](./09_Spectral_Analysis.md)
