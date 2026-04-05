# 15. Capstone: Solar Event Analysis

**Previous**: [Performance and Large Data](./14_Performance_and_Large_Data.md)

---

## Learning Objectives

In this capstone project, you will:

1. Download SDO/AIA multi-wavelength data for a solar flare event
2. Calibrate all images with AIA_PREP
3. Create running difference images to visualize eruption dynamics
4. Build multi-wavelength composite images
5. Extract and analyze time series of flare region intensity
6. Detect flare onset from the light curve
7. Produce a publication-quality figure set and output to PostScript

This lesson integrates techniques from all previous lessons into a complete, realistic analysis workflow.

---

## Project Overview

We will analyze a solar flare event step by step. The workflow follows the standard solar physics analysis pipeline:

```
Data Acquisition → Calibration → Visualization → Time Series → Detection → Publication Figures
```

---

## Step 1: Data Acquisition

```idl
; Define the event
; Example: M-class flare on 2024-01-15 around 12:00 UT
event_date = '2024-01-15'
flare_time = '2024-01-15T12:00:00'
t_start = '2024-01-15T11:30:00'
t_end = '2024-01-15T13:00:00'

; Channels to analyze
channels = [94, 131, 171, 193, 211, 304, 335]
n_channels = N_ELEMENTS(channels)

; Data directory
data_dir = '/data/flare_analysis/'
FILE_MKDIR, data_dir

; Option A: Download from JSOC using sswidl
FOR ic = 0, n_channels-1 DO BEGIN
    wave = channels[ic]
    wave_dir = data_dir + STRTRIM(wave, 2) + '/'
    FILE_MKDIR, wave_dir

    ; Search for files (using VSO or JSOC)
    vso_search, t_start, t_end, $
        INSTRUMENT='aia', WAVE=STRTRIM(wave, 2), $
        results, /FLAT

    ; Download (select subset — 1 image per minute for speed)
    ; In practice, select ~60 images spanning the event
    vso_get, results[0:59], OUT_DIR=wave_dir, FILENAMES=fnames
    PRINT, 'Downloaded ', N_ELEMENTS(fnames), ' files for ', wave, ' A'
ENDFOR

; Option B: If files already exist on disk
FOR ic = 0, n_channels-1 DO BEGIN
    wave = channels[ic]
    wave_dir = data_dir + STRTRIM(wave, 2) + '/'
    files = FILE_SEARCH(wave_dir + '*.fits', COUNT=nf)
    PRINT, 'Channel ', wave, ' A: ', nf, ' files'
ENDFOR
```

---

## Step 2: Calibration

```idl
; Calibrate all channels
PRINT, '=== Calibration ==='
prep_dir = data_dir + 'prepped/'
FILE_MKDIR, prep_dir

FOR ic = 0, n_channels-1 DO BEGIN
    wave = channels[ic]
    wave_dir = data_dir + STRTRIM(wave, 2) + '/'
    out_dir = prep_dir + STRTRIM(wave, 2) + '/'
    FILE_MKDIR, out_dir

    files = FILE_SEARCH(wave_dir + '*.fits', COUNT=nf)
    IF nf EQ 0 THEN CONTINUE

    PRINT, 'Calibrating ', wave, ' A (', nf, ' files)...'
    t0 = SYSTIME(1)

    FOR i = 0, nf-1 DO BEGIN
        read_sdo, files[i], index, data
        aia_prep, index, data, oindex, odata, $
            /NORMALIZE, /REGISTER, /CUTOUT

        outfile = out_dir + 'prep_' + FILE_BASENAME(files[i])
        mwritefits, oindex, odata, OUTFILE=outfile

        ; Free memory
        data = 0 & odata = 0
    ENDFOR

    elapsed = SYSTIME(1) - t0
    PRINT, '  Done in ', elapsed, ' s (', elapsed/nf, ' s/file)'
ENDFOR
```

---

## Step 3: Running Difference Images

```idl
; Create running difference images for 171 A
PRINT, '=== Running Differences ==='

wave = 171
wave_dir = prep_dir + STRTRIM(wave, 2) + '/'
files = FILE_SEARCH(wave_dir + '*.fits', COUNT=nf)

; Read all calibrated images
read_sdo, files, index, data_cube
; data_cube: [4096, 4096, nf]

; Define active region sub-field (in pixels)
; Convert from arcsec using WCS
wcs = FITSHEAD2WCS(index[0])
xcen = -200.0  ; arcsec (example AR location)
ycen = 300.0
half_fov = 250.0  ; arcsec

pix_ll = WCS_GET_PIXEL(wcs, [xcen-half_fov, ycen-half_fov])
pix_ur = WCS_GET_PIXEL(wcs, [xcen+half_fov, ycen+half_fov])
x0 = ROUND(pix_ll[0]) > 0
y0 = ROUND(pix_ll[1]) > 0
x1 = ROUND(pix_ur[0]) < (index[0].naxis1-1)
y1 = ROUND(pix_ur[1]) < (index[0].naxis2-1)

; Extract sub-field
sub_cube = data_cube[x0:x1, y0:y1, *]
sub_nx = x1 - x0 + 1
sub_ny = y1 - y0 + 1
data_cube = 0  ; Free full-disk data

; Compute running differences
PRINT, 'Computing running differences...'
run_diff = sub_cube[*, *, 1:nf-1] - sub_cube[*, *, 0:nf-2]

; Save a movie of running differences
diff_dir = data_dir + 'run_diff/'
FILE_MKDIR, diff_dir

vmax = 100.0  ; DN/s scaling
LOADCT, 33    ; Blue-Red for +/- differences

FOR t = 0, nf-2 DO BEGIN
    frame = run_diff[*, *, t]
    img = BYTSCL(frame, MIN=-vmax, MAX=vmax)

    ; Save as PNG
    outfile = diff_dir + STRING(t, FORMAT='("diff_", I04, ".png")')
    WRITE_PNG, outfile, CONGRID(img, 512, 512)
ENDFOR

PRINT, 'Saved ', nf-1, ' running difference images'
```

---

## Step 4: Multi-Wavelength Composite

```idl
; Create a three-color composite image at the flare peak time
PRINT, '=== Multi-Wavelength Composite ==='

; Find the frame closest to flare peak
flare_peak_time = '2024-01-15T12:10:00'
peak_tai = ANYTIM(flare_peak_time)
times_171 = ANYTIM(index.date_obs)
peak_idx = (WHERE(ABS(times_171 - peak_tai) EQ MIN(ABS(times_171 - peak_tai))))[0]
PRINT, 'Peak frame: ', peak_idx, ' at ', ANYTIM(times_171[peak_idx], /CCSDS)

; Read the three channels at peak time
channels_rgb = [304, 171, 193]  ; Red, Green, Blue
colors = STRARR(3)
rgb_data = FLTARR(sub_nx, sub_ny, 3)

FOR ic = 0, 2 DO BEGIN
    wave = channels_rgb[ic]
    wave_dir = prep_dir + STRTRIM(wave, 2) + '/'
    files_w = FILE_SEARCH(wave_dir + '*.fits', COUNT=nf_w)

    ; Find matching time
    read_sdo, files_w, idx_w
    times_w = ANYTIM(idx_w.date_obs)
    match_idx = (WHERE(ABS(times_w - peak_tai) EQ $
        MIN(ABS(times_w - peak_tai))))[0]

    read_sdo, files_w[match_idx], idx_tmp, dat_tmp
    rgb_data[*, *, ic] = dat_tmp[x0:x1, y0:y1]
    dat_tmp = 0
ENDFOR

; Scale each channel
r = BYTSCL(ALOG10(rgb_data[*, *, 0] > 1), MIN=0.5, MAX=3.5)  ; 304
g = BYTSCL(ALOG10(rgb_data[*, *, 1] > 1), MIN=0.5, MAX=3.5)  ; 171
b = BYTSCL(ALOG10(rgb_data[*, *, 2] > 1), MIN=0.5, MAX=3.5)  ; 193

; Display
sz = 512
r_disp = CONGRID(r, sz, sz)
g_disp = CONGRID(g, sz, sz)
b_disp = CONGRID(b, sz, sz)

WINDOW, 0, XSIZE=sz, YSIZE=sz
TV, [[[r_disp]], [[g_disp]], [[b_disp]]], TRUE=3

; Save composite
WRITE_PNG, data_dir + 'composite_peak.png', r_disp, g_disp, b_disp
PRINT, 'Saved composite image'
```

---

## Step 5: Time Series Analysis

```idl
; Extract light curves from the flare region
PRINT, '=== Light Curve Extraction ==='

; Define flare core region (smaller box within the sub-field)
; In relative pixel coordinates within the sub-field
fx0 = sub_nx/2 - 30
fx1 = sub_nx/2 + 30
fy0 = sub_ny/2 - 30
fy1 = sub_ny/2 + 30

; Extract light curves for all channels
n_time_pts = N_ELEMENTS(files)  ; Number of time steps for 171
lightcurves = DBLARR(n_channels, n_time_pts)
times_sec = DBLARR(n_time_pts)

; 171 A light curve from already-loaded data
FOR t = 0, n_time_pts-1 DO BEGIN
    lightcurves[2, t] = MEAN(sub_cube[fx0:fx1, fy0:fy1, t])
    times_sec[t] = ANYTIM(index[t].date_obs)
ENDFOR

; Other channels
FOR ic = 0, n_channels-1 DO BEGIN
    IF ic EQ 2 THEN CONTINUE  ; Already did 171

    wave = channels[ic]
    wave_dir = prep_dir + STRTRIM(wave, 2) + '/'
    files_w = FILE_SEARCH(wave_dir + '*.fits', COUNT=nf_w)

    FOR t = 0, (nf_w < n_time_pts)-1 DO BEGIN
        read_sdo, files_w[t], idx_w, dat_w
        roi = dat_w[x0+fx0:x0+fx1, y0+fy0:y0+fy1]
        lightcurves[ic, t] = MEAN(roi)
        dat_w = 0
    ENDFOR

    PRINT, 'Extracted light curve for ', wave, ' A'
ENDFOR

; Time in minutes from start
t_min = (times_sec - times_sec[0]) / 60.0
```

---

## Step 6: Flare Onset Detection

```idl
; Detect flare onset from the 171 A light curve
PRINT, '=== Flare Onset Detection ==='

lc = lightcurves[2, *]  ; 171 A
lc = REFORM(lc)

; Pre-flare baseline (first 10 minutes)
pre_flare_idx = WHERE(t_min LT 10.0, n_pre)
baseline_mean = MEAN(lc[pre_flare_idx])
baseline_std = STDDEV(lc[pre_flare_idx])

PRINT, 'Pre-flare baseline: ', baseline_mean, ' +/- ', baseline_std, ' DN/s'

; Onset: first time the light curve exceeds baseline + 3*sigma
threshold = baseline_mean + 3.0 * baseline_std
onset_candidates = WHERE(lc GT threshold AND t_min GT 10.0, n_onset)

IF n_onset GT 0 THEN BEGIN
    onset_idx = onset_candidates[0]
    onset_time = times_sec[onset_idx]
    PRINT, 'Flare onset detected at: ', ANYTIM(onset_time, /CCSDS)
    PRINT, '  t = ', t_min[onset_idx], ' minutes from start'
    PRINT, '  Intensity: ', lc[onset_idx], ' DN/s'
ENDIF

; Peak
peak_idx = WHERE(lc EQ MAX(lc))
peak_idx = peak_idx[0]
peak_time = times_sec[peak_idx]
PRINT, 'Flare peak at: ', ANYTIM(peak_time, /CCSDS)
PRINT, '  t = ', t_min[peak_idx], ' minutes from start'
PRINT, '  Peak intensity: ', lc[peak_idx], ' DN/s'
PRINT, '  Enhancement: ', lc[peak_idx] / baseline_mean, 'x baseline'

; Rise time and decay time
half_max = (MAX(lc) + baseline_mean) / 2.0
rise_idx = WHERE(lc[0:peak_idx] GT half_max)
decay_idx = WHERE(lc[peak_idx:*] LT half_max)

IF N_ELEMENTS(rise_idx) GT 0 AND N_ELEMENTS(decay_idx) GT 0 THEN BEGIN
    rise_time = t_min[peak_idx] - t_min[rise_idx[0]]
    decay_time = t_min[peak_idx + decay_idx[0]] - t_min[peak_idx]
    PRINT, 'Rise time:  ', rise_time, ' minutes'
    PRINT, 'Decay time: ', decay_time, ' minutes'
ENDIF
```

---

## Step 7: Publication-Quality Figure Set

```idl
; Generate a multi-panel publication figure
PRINT, '=== Publication Figures ==='

; === Figure 1: Multi-panel light curves ===
SET_PLOT, 'PS'
DEVICE, FILENAME=data_dir + 'figure1_lightcurves.eps', $
    /ENCAPSULATED, /COLOR, BITS=8, $
    XSIZE=18, YSIZE=22

!P.THICK = 3
!P.CHARTHICK = 2
!P.CHARSIZE = 0.9
!X.THICK = 2
!Y.THICK = 2
!P.MULTI = [0, 1, n_channels]

channel_names = ['94 A (Fe XVIII)', '131 A (Fe VIII/XXI)', $
    '171 A (Fe IX)', '193 A (Fe XII)', '211 A (Fe XIV)', $
    '304 A (He II)', '335 A (Fe XVI)']

FOR ic = 0, n_channels-1 DO BEGIN
    lc_plot = REFORM(lightcurves[ic, *])
    valid = WHERE(lc_plot GT 0, nvalid)
    IF nvalid EQ 0 THEN CONTINUE

    PLOT, t_min, lc_plot, $
        XTITLE=(ic EQ n_channels-1) ? 'Time (minutes)' : '', $
        YTITLE='DN/s', $
        TITLE=channel_names[ic], $
        XSTYLE=1, XRANGE=[0, MAX(t_min)]

    ; Mark onset and peak
    IF N_ELEMENTS(onset_idx) GT 0 THEN $
        PLOTS, [t_min[onset_idx], t_min[onset_idx]], !Y.CRANGE, $
            LINESTYLE=2, COLOR=200
    PLOTS, [t_min[peak_idx], t_min[peak_idx]], !Y.CRANGE, $
        LINESTYLE=1, COLOR=150
ENDFOR

DEVICE, /CLOSE

; === Figure 2: Image evolution ===
DEVICE, FILENAME=data_dir + 'figure2_evolution.eps', $
    /ENCAPSULATED, /COLOR, BITS=8, $
    XSIZE=18, YSIZE=12

!P.MULTI = [0, 4, 2]

; Select 4 time steps: pre-flare, onset, peak, decay
time_labels = ['Pre-flare', 'Onset', 'Peak', 'Decay']
time_indices = [5, onset_idx, peak_idx, peak_idx + 20 < (n_time_pts-1)]

; Row 1: Direct images (171 A)
LOADCT, 3
FOR it = 0, 3 DO BEGIN
    img = REFORM(sub_cube[*, *, time_indices[it]])
    TV, BYTSCL(CONGRID(ALOG10(img > 1), 200, 200), MIN=0.5, MAX=3.5)
    XYOUTS, 0.5, 0.95, time_labels[it] + ' (' + $
        ANYTIM(times_sec[time_indices[it]], /CCSDS) + ')', $
        /NORMAL, ALIGNMENT=0.5, CHARSIZE=0.7
ENDFOR

; Row 2: Running difference images
LOADCT, 33
FOR it = 0, 3 DO BEGIN
    ti = time_indices[it]
    IF ti GT 0 AND ti LT nf THEN BEGIN
        diff = REFORM(run_diff[*, *, (ti-1) > 0])
        TV, BYTSCL(CONGRID(diff, 200, 200), MIN=-vmax, MAX=vmax)
    ENDIF
ENDFOR

DEVICE, /CLOSE

; === Figure 3: Composite + light curve ===
DEVICE, FILENAME=data_dir + 'figure3_summary.eps', $
    /ENCAPSULATED, /COLOR, BITS=8, $
    XSIZE=18, YSIZE=8

; Left panel: composite image
; (Need to write true-color to PostScript differently)
; Use indexed-color approximation
LOADCT, 0
TV, BYTSCL(CONGRID(ALOG10(sub_cube[*, *, peak_idx] > 1), 300, 300), $
    MIN=0.5, MAX=3.5), 20, 50

; Right panel: main light curve with annotations
PLOT, t_min, REFORM(lightcurves[2, *]), $
    POSITION=[0.55, 0.15, 0.95, 0.90], /NOERASE, $
    XTITLE='Time (minutes from start)', $
    YTITLE='AIA 171 Mean Intensity (DN/s)', $
    TITLE='Flare Light Curve', $
    XSTYLE=1

; Shade pre-flare region
POLYFILL, [0, 10, 10, 0], $
    [!Y.CRANGE[0], !Y.CRANGE[0], !Y.CRANGE[1], !Y.CRANGE[1]], $
    COLOR=240

; Re-plot light curve on top
OPLOT, t_min, REFORM(lightcurves[2, *]), THICK=3

; Mark onset
PLOTS, [t_min[onset_idx], t_min[onset_idx]], !Y.CRANGE, $
    LINESTYLE=2, THICK=2, COLOR=100
XYOUTS, t_min[onset_idx]+0.5, !Y.CRANGE[1]*0.9, 'Onset', $
    CHARSIZE=0.7

; Mark peak
PLOTS, [t_min[peak_idx], t_min[peak_idx]], !Y.CRANGE, $
    LINESTYLE=1, THICK=2, COLOR=50
XYOUTS, t_min[peak_idx]+0.5, !Y.CRANGE[1]*0.85, 'Peak', $
    CHARSIZE=0.7

; Baseline + threshold
PLOTS, [0, MAX(t_min)], [baseline_mean, baseline_mean], $
    LINESTYLE=3, COLOR=150
PLOTS, [0, MAX(t_min)], [threshold, threshold], $
    LINESTYLE=1, COLOR=200

DEVICE, /CLOSE
SET_PLOT, 'X'

; Reset system variables
!P.THICK = 0 & !P.CHARTHICK = 0 & !P.CHARSIZE = 0
!X.THICK = 0 & !Y.THICK = 0 & !P.MULTI = 0

PRINT, '=== Publication figures saved ==='
PRINT, '  figure1_lightcurves.eps'
PRINT, '  figure2_evolution.eps'
PRINT, '  figure3_summary.eps'
```

---

## Step 8: Summary Report

```idl
; Print analysis summary
PRINT, ''
PRINT, '========================================='
PRINT, '  Solar Flare Analysis Summary'
PRINT, '========================================='
PRINT, 'Event date:     ', event_date
PRINT, 'Time range:     ', t_start, ' to ', t_end
PRINT, 'Channels:       ', STRJOIN(STRTRIM(channels, 2), ', '), ' A'
PRINT, ''
PRINT, 'Pre-flare mean: ', baseline_mean, ' DN/s (171 A)'
PRINT, 'Flare onset:    ', ANYTIM(onset_time, /CCSDS)
PRINT, 'Flare peak:     ', ANYTIM(peak_time, /CCSDS)
PRINT, 'Peak intensity: ', MAX(lc), ' DN/s (171 A)'
PRINT, 'Enhancement:    ', MAX(lc)/baseline_mean, 'x'
PRINT, 'Rise time:      ', rise_time, ' minutes'
PRINT, 'Decay time:     ', decay_time, ' minutes'
PRINT, '========================================='

; Save results
SAVE, lightcurves, times_sec, t_min, channels, $
    baseline_mean, baseline_std, $
    onset_time, peak_time, rise_time, decay_time, $
    FILENAME=data_dir + 'analysis_results.sav'

PRINT, 'Results saved to: ', data_dir + 'analysis_results.sav'
```

---

## Project Extensions

1. **DEM Analysis**: Use the 6 EUV channels to compute DEM maps at the flare peak, compare pre-flare and peak DEM profiles
2. **Magnetic Context**: Overlay HMI magnetogram contours on the AIA images to identify magnetic topology
3. **GOES Correlation**: Plot GOES 1-8 A light curve alongside AIA light curves, verify flare class
4. **EUV Wave Detection**: Use running differences over the full disk to detect EUV waves (coronal bright fronts)
5. **CME Association**: Check LASCO coronagraph data for associated CME
6. **Spectral Analysis**: Apply FFT to the pre-flare light curve to search for quasi-periodic pulsations (QPPs)

---

## Checklist

- [ ] Data downloaded for all channels covering the event
- [ ] All images calibrated with aia_prep (/NORMALIZE, /REGISTER)
- [ ] Running difference movie created
- [ ] Multi-wavelength composite image at flare peak
- [ ] Light curves extracted for all channels
- [ ] Flare onset and peak times identified
- [ ] Rise time and decay time measured
- [ ] Publication-quality PostScript figures generated
- [ ] Analysis results saved (.sav file)

---

**Previous**: [Performance and Large Data](./14_Performance_and_Large_Data.md)
