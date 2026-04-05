# Project: Solar Light Curve

**Previous**: [Debugging and Best Practices](./14_Debugging_and_Best_Practices.md)

## Learning Objectives

After completing this project, you will be able to:

1. Read FITS time series data (GOES X-ray flux)
2. Extract time and flux arrays from FITS files
3. Handle date/time conversion for scientific data
4. Create publication-quality light curve plots
5. Add proper axis labels, legends, and annotations
6. Output figures in PostScript format for journal submission
7. Apply all IDL skills learned in this course to a real-world problem

---

In this capstone project, we bring together everything from the previous 14 lessons to build a complete solar physics data analysis workflow. We will read GOES (Geostationary Operational Environmental Satellite) X-ray flux data, process it, and produce a publication-quality light curve plot.

## Project Overview

GOES X-ray Sensor (XRS) data is the standard dataset for monitoring solar flare activity. The sensor measures solar X-ray flux in two wavelength bands:

- **Short channel (0.5-4 A)**: Sensitive to hot flare plasma
- **Long channel (1-8 A)**: Standard flare classification channel

Solar flares are classified by the peak flux in the 1-8 A channel:

| Class | Flux Range (W/m^2) |
|-------|-------------------|
| A | < 10^-7 |
| B | 10^-7 to 10^-6 |
| C | 10^-6 to 10^-5 |
| M | 10^-5 to 10^-4 |
| X | >= 10^-4 |

Our goal is to create a program that:
1. Reads GOES XRS data from a FITS file (or simulated data)
2. Extracts time and flux arrays
3. Identifies flare peaks
4. Creates a publication-quality light curve
5. Outputs PostScript for inclusion in a paper

---

## Step 1: Generate Simulated GOES Data

Since we may not have actual GOES FITS files available, let us first create realistic simulated data:

```idl
;+
; NAME:
;   generate_goes_data
;
; PURPOSE:
;   Generate simulated GOES X-ray flux data with realistic flare events
;
; OUTPUTS:
;   Returns a structure with time (Julian dates), short and long channel flux
;-
FUNCTION generate_goes_data, DURATION_HOURS=duration, CADENCE_SEC=cadence, $
                              SEED=seed
  IF N_ELEMENTS(duration) EQ 0 THEN duration = 24.0D0
  IF N_ELEMENTS(cadence) EQ 0 THEN cadence = 60.0D0  ; 1-minute cadence
  IF N_ELEMENTS(seed) EQ 0 THEN seed = 42L

  ; Time array
  n_points = LONG(duration * 3600.0D0 / cadence)
  start_jd = JULDAY(7, 15, 2024, 0, 0, 0)
  time_jd = start_jd + DINDGEN(n_points) * cadence / 86400.0D0

  ; Background flux (quiet Sun)
  ; Long channel background: ~5e-8 W/m^2 (A-class)
  background_long = 5.0D-8 + RANDOMN(seed, n_points) * 5.0D-9
  background_long = background_long > 1.0D-9  ; Floor

  ; Short channel is ~10x weaker
  background_short = background_long * 0.1D0

  ; Add flare events
  flux_long = background_long
  flux_short = background_short

  ; Flare 1: C-class (peak ~3e-6)
  t_peak1 = start_jd + 4.0D0 / 24.0D0    ; 04:00 UT
  flare1 = flare_profile(time_jd, t_peak1, 3.0D-6, 10.0D0, 30.0D0)
  flux_long = flux_long + flare1
  flux_short = flux_short + flare1 * 0.15D0

  ; Flare 2: M-class (peak ~2e-5)
  t_peak2 = start_jd + 10.5D0 / 24.0D0   ; 10:30 UT
  flare2 = flare_profile(time_jd, t_peak2, 2.0D-5, 8.0D0, 45.0D0)
  flux_long = flux_long + flare2
  flux_short = flux_short + flare2 * 0.12D0

  ; Flare 3: B-class (peak ~5e-7)
  t_peak3 = start_jd + 16.0D0 / 24.0D0   ; 16:00 UT
  flare3 = flare_profile(time_jd, t_peak3, 5.0D-7, 5.0D0, 20.0D0)
  flux_long = flux_long + flare3
  flux_short = flux_short + flare3 * 0.1D0

  ; Flare 4: X-class (peak ~1.5e-4)
  t_peak4 = start_jd + 19.0D0 / 24.0D0   ; 19:00 UT
  flare4 = flare_profile(time_jd, t_peak4, 1.5D-4, 6.0D0, 60.0D0)
  flux_long = flux_long + flare4
  flux_short = flux_short + flare4 * 0.08D0

  RETURN, {time_jd: time_jd, $
           flux_long: flux_long, $
           flux_short: flux_short, $
           n_points: n_points, $
           start_jd: start_jd, $
           cadence: cadence, $
           duration: duration}
END

;+
; Helper: generate a flare temporal profile (fast rise, slow decay)
;-
FUNCTION flare_profile, time_jd, peak_jd, peak_flux, rise_min, decay_min
  dt = (time_jd - peak_jd) * 1440.0D0    ; Time in minutes from peak
  rise = EXP(-(dt < 0)^2 / (2.0D0 * rise_min^2))    ; Gaussian rise
  decay = EXP(-(dt > 0) / decay_min)                  ; Exponential decay
  profile = peak_flux * (rise * (dt LE 0) + decay * (dt GT 0))
  RETURN, profile
END
```

---

## Step 2: Save Simulated Data as FITS

```idl
PRO save_goes_fits, goes_data, filename
  ; Create primary HDU with long channel data
  MKHDR, header, goes_data.flux_long

  ; Add metadata
  SXADDPAR, header, 'TELESCOP', 'GOES-16', 'Satellite name'
  SXADDPAR, header, 'INSTRUME', 'XRS', 'Instrument name'
  SXADDPAR, header, 'OBJECT', 'Sun', 'Target'
  SXADDPAR, header, 'BUNIT', 'W/m^2', 'Flux units'
  SXADDPAR, header, 'CHANNEL', '1-8 A', 'Wavelength band'

  CALDAT, goes_data.start_jd, m, d, y, h, mn, s
  date_str = STRING(FORMAT='(I4, "-", I02, "-", I02, "T", I02, ":", I02, ":", I02)', $
    y, m, d, h, mn, FIX(s))
  SXADDPAR, header, 'DATE-OBS', date_str, 'Observation start'
  SXADDPAR, header, 'CADENCE', goes_data.cadence, 'Time cadence (seconds)'
  SXADDPAR, header, 'NPOINTS', goes_data.n_points, 'Number of data points'
  SXADDPAR, header, 'DURATION', goes_data.duration, 'Duration (hours)'

  SXADDPAR, header, 'COMMENT', 'Simulated GOES XRS data for IDL tutorial'
  SXADDPAR, header, 'HISTORY', 'Generated by generate_goes_data.pro'
  SXADDPAR, header, 'HISTORY', 'Created: ' + SYSTIME()

  ; Write primary HDU (long channel flux)
  WRITEFITS, filename, goes_data.flux_long, header

  ; Write short channel as extension
  MKHDR, ext_header, goes_data.flux_short, /EXTEND
  SXADDPAR, ext_header, 'CHANNEL', '0.5-4 A', 'Wavelength band'
  SXADDPAR, ext_header, 'BUNIT', 'W/m^2', 'Flux units'
  WRITEFITS, filename, goes_data.flux_short, ext_header, /APPEND

  ; Write time array as extension
  MKHDR, time_header, goes_data.time_jd, /EXTEND
  SXADDPAR, time_header, 'TTYPE', 'TIME_JD', 'Julian Date'
  WRITEFITS, filename, goes_data.time_jd, time_header, /APPEND

  PRINT, 'FITS file saved: ' + filename
END
```

---

## Step 3: Read the FITS Data

```idl
FUNCTION read_goes_fits, filename
  ; Check file
  IF ~FILE_TEST(filename) THEN BEGIN
    PRINT, 'File not found: ' + filename
    RETURN, !NULL
  ENDIF

  ; Read primary HDU (long channel)
  flux_long = READFITS(filename, header)

  ; Read metadata
  telescop = STRTRIM(SXPAR(header, 'TELESCOP'), 2)
  date_obs = STRTRIM(SXPAR(header, 'DATE-OBS'), 2)
  cadence = SXPAR(header, 'CADENCE')
  n_points = SXPAR(header, 'NPOINTS')
  duration = SXPAR(header, 'DURATION')

  PRINT, 'Telescope: ' + telescop
  PRINT, 'Date: ' + date_obs
  PRINT, FORMAT='("Points: ", I0, "  Cadence: ", F5.1, "s  Duration: ", F5.1, "h")', $
    n_points, cadence, duration

  ; Read extensions
  flux_short = READFITS(filename, ext_hdr, EXTEN_NO=1)
  time_jd = READFITS(filename, time_hdr, EXTEN_NO=2)

  RETURN, {time_jd: time_jd, $
           flux_long: flux_long, $
           flux_short: flux_short, $
           header: header, $
           date_obs: date_obs, $
           cadence: cadence, $
           n_points: n_points}
END
```

---

## Step 4: Identify Flare Events

```idl
FUNCTION find_flares, time_jd, flux, THRESHOLD=threshold, MIN_SEPARATION=min_sep
  IF N_ELEMENTS(threshold) EQ 0 THEN threshold = 1.0D-6   ; C1.0
  IF N_ELEMENTS(min_sep) EQ 0 THEN min_sep = 30.0D0       ; 30 minutes

  n = N_ELEMENTS(flux)

  ; Find local maxima above threshold
  ; A point is a local max if it's greater than both neighbors
  is_peak = BYTARR(n)
  FOR i = 1L, n - 2 DO BEGIN
    IF flux[i] GT flux[i-1] AND flux[i] GT flux[i+1] AND $
       flux[i] GT threshold THEN is_peak[i] = 1B
  ENDFOR

  peak_idx = WHERE(is_peak, n_peaks)
  IF n_peaks EQ 0 THEN BEGIN
    PRINT, 'No flares found above threshold'
    RETURN, !NULL
  ENDIF

  ; Merge nearby peaks (keep the largest within min_separation)
  min_sep_jd = min_sep / 1440.0D0
  merged = BYTARR(n_peaks) + 1B
  FOR i = 0, n_peaks - 2 DO BEGIN
    IF ~merged[i] THEN CONTINUE
    FOR j = i + 1, n_peaks - 1 DO BEGIN
      IF (time_jd[peak_idx[j]] - time_jd[peak_idx[i]]) LT min_sep_jd THEN BEGIN
        ; Keep the larger peak
        IF flux[peak_idx[j]] GT flux[peak_idx[i]] THEN $
          merged[i] = 0B ELSE merged[j] = 0B
      ENDIF
    ENDFOR
  ENDFOR

  keep = WHERE(merged, n_flares)
  peak_idx = peak_idx[keep]

  ; Classify flares
  flares = REPLICATE({peak_time: 0.0D0, peak_flux: 0.0D0, class: '', $
                       index: 0L}, n_flares)

  FOR i = 0, n_flares - 1 DO BEGIN
    flares[i].peak_time = time_jd[peak_idx[i]]
    flares[i].peak_flux = flux[peak_idx[i]]
    flares[i].index = peak_idx[i]

    ; Classify
    f = flux[peak_idx[i]]
    IF f GE 1.0D-4 THEN BEGIN
      flares[i].class = 'X' + STRTRIM(STRING(f / 1.0D-4, FORMAT='(F4.1)'), 2)
    ENDIF ELSE IF f GE 1.0D-5 THEN BEGIN
      flares[i].class = 'M' + STRTRIM(STRING(f / 1.0D-5, FORMAT='(F4.1)'), 2)
    ENDIF ELSE IF f GE 1.0D-6 THEN BEGIN
      flares[i].class = 'C' + STRTRIM(STRING(f / 1.0D-6, FORMAT='(F4.1)'), 2)
    ENDIF ELSE IF f GE 1.0D-7 THEN BEGIN
      flares[i].class = 'B' + STRTRIM(STRING(f / 1.0D-7, FORMAT='(F4.1)'), 2)
    ENDIF ELSE BEGIN
      flares[i].class = 'A'
    ENDELSE
  ENDFOR

  ; Print summary
  PRINT, FORMAT='("Found ", I0, " flare events:")', n_flares
  FOR i = 0, n_flares - 1 DO BEGIN
    CALDAT, flares[i].peak_time, m, d, y, h, mn, s
    PRINT, FORMAT='("  ", A6, " at ", I02, ":", I02, " UT  (", E9.2, " W/m^2)")', $
      flares[i].class, h, mn, flares[i].peak_flux
  ENDFOR

  RETURN, flares
END
```

---

## Step 5: Create the Light Curve Plot (Screen)

```idl
PRO plot_light_curve, goes_data, flares, TO_FILE=to_file
  ; Extract data
  time = goes_data.time_jd
  flux_long = goes_data.flux_long
  flux_short = goes_data.flux_short

  ; Time as hours from start
  start_jd = MIN(time)
  time_hours = (time - start_jd) * 24.0D0

  ; Get date string for title
  CALDAT, start_jd, m, d, y
  months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
  date_label = STRTRIM(d, 2) + ' ' + months[m-1] + ' ' + STRTRIM(y, 2)

  ;--- Screen Plot ---
  IF ~KEYWORD_SET(to_file) THEN BEGIN
    DEVICE, DECOMPOSED=0
    LOADCT, 0, /SILENT
    WINDOW, 0, XSIZE=900, YSIZE=600, TITLE='GOES X-ray Light Curve'
  ENDIF

  ;--- PostScript Output ---
  IF KEYWORD_SET(to_file) THEN BEGIN
    original_device = !D.NAME
    SET_PLOT, 'PS'
    DEVICE, FILENAME=to_file, $
      /COLOR, /ENCAPSULATED, $
      XSIZE=20, YSIZE=14, $      ; cm
      BITS_PER_PIXEL=8, $
      /PORTRAIT
    LOADCT, 0, /SILENT
  ENDIF

  ; Plot parameters
  thick = KEYWORD_SET(to_file) ? 3 : 2
  charthick = KEYWORD_SET(to_file) ? 2 : 1
  charsize = KEYWORD_SET(to_file) ? 1.0 : 1.3

  ; Main plot: Long channel (1-8 A) with log Y axis
  PLOT, time_hours, flux_long, $
    /YLOG, $
    YRANGE=[1e-8, 1e-3], $
    XRANGE=[0, MAX(time_hours)], $
    XSTYLE=1, YSTYLE=9, $        ; YSTYLE=9: exact + suppress right axis
    TITLE='GOES X-ray Flux (' + date_label + ')', $
    XTITLE='Time (UT hours)', $
    YTITLE='Flux (W m!U-2!N)', $
    THICK=thick, $
    CHARSIZE=charsize, $
    CHARTHICK=charthick, $
    XTHICK=thick, YTHICK=thick, $
    POSITION=[0.12, 0.12, 0.88, 0.90]

  ; Overlay short channel (0.5-4 A)
  OPLOT, time_hours, flux_short, LINESTYLE=2, THICK=thick

  ; Draw flare classification lines
  class_levels = [1e-7, 1e-6, 1e-5, 1e-4]
  class_labels = ['B', 'C', 'M', 'X']
  FOR i = 0, N_ELEMENTS(class_levels) - 1 DO BEGIN
    OPLOT, [0, MAX(time_hours)], REPLICATE(class_levels[i], 2), $
      LINESTYLE=1, THICK=1
    ; Label on right side
    XYOUTS, MAX(time_hours) * 1.02, class_levels[i], $
      class_labels[i], CHARSIZE=charsize * 0.8, CHARTHICK=charthick
  ENDFOR

  ; Mark flare peaks
  IF N_ELEMENTS(flares) GT 0 THEN BEGIN
    FOR i = 0, N_ELEMENTS(flares) - 1 DO BEGIN
      peak_hour = (flares[i].peak_time - start_jd) * 24.0D0
      ; Draw vertical marker
      OPLOT, [peak_hour, peak_hour], [flares[i].peak_flux * 0.5, flares[i].peak_flux * 2], $
        THICK=1
      ; Label the flare
      XYOUTS, peak_hour, flares[i].peak_flux * 2.5, $
        flares[i].class, ALIGNMENT=0.5, $
        CHARSIZE=charsize * 0.7, CHARTHICK=charthick
    ENDFOR
  ENDIF

  ; Legend
  legend_x = 0.15
  legend_y = 0.85
  XYOUTS, legend_x + 0.07, legend_y, '1-8 '+STRING(197B), $       ; Angstrom symbol
    /NORMAL, CHARSIZE=charsize * 0.8, CHARTHICK=charthick
  PLOTS, [legend_x, legend_x + 0.06], [legend_y + 0.01, legend_y + 0.01], $
    /NORMAL, THICK=thick, LINESTYLE=0
  XYOUTS, legend_x + 0.07, legend_y - 0.04, '0.5-4 '+STRING(197B), $
    /NORMAL, CHARSIZE=charsize * 0.8, CHARTHICK=charthick
  PLOTS, [legend_x, legend_x + 0.06], [legend_y - 0.03, legend_y - 0.03], $
    /NORMAL, THICK=thick, LINESTYLE=2

  ; Close PostScript if needed
  IF KEYWORD_SET(to_file) THEN BEGIN
    DEVICE, /CLOSE
    SET_PLOT, original_device
    PRINT, 'PostScript saved: ' + to_file
  ENDIF
END
```

---

## Step 6: Complete Main Program

```idl
;+
; NAME:
;   goes_light_curve
;
; PURPOSE:
;   Main program for the Solar Light Curve project.
;   Generates (or reads) GOES X-ray data, identifies flares,
;   and creates publication-quality light curve plots.
;
; CALLING SEQUENCE:
;   goes_light_curve [, /FROM_FILE] [, FITS_FILE=filename]
;-
PRO goes_light_curve, FROM_FILE=from_file, FITS_FILE=fits_file
  PRINT, '============================================='
  PRINT, '   GOES Solar X-ray Light Curve Analysis'
  PRINT, '============================================='
  PRINT, ''

  IF ~KEYWORD_SET(fits_file) THEN fits_file = 'goes_xrs_sim.fits'

  ;--- Step 1: Get data ---
  IF KEYWORD_SET(from_file) THEN BEGIN
    ; Read from FITS file
    PRINT, 'Reading data from: ' + fits_file
    goes_data = read_goes_fits(fits_file)
    IF ~ISA(goes_data) THEN RETURN
  ENDIF ELSE BEGIN
    ; Generate simulated data
    PRINT, 'Generating simulated GOES data...'
    goes_data = generate_goes_data(DURATION_HOURS=24.0D0, CADENCE_SEC=60.0D0)

    ; Save to FITS
    PRINT, 'Saving to FITS: ' + fits_file
    save_goes_fits, goes_data, fits_file
  ENDELSE

  ;--- Step 2: Print statistics ---
  PRINT, ''
  PRINT, '--- Data Summary ---'
  PRINT, FORMAT='("  Data points: ", I0)', goes_data.n_points
  PRINT, FORMAT='("  Long channel range: ", E9.2, " to ", E9.2, " W/m^2")', $
    MIN(goes_data.flux_long), MAX(goes_data.flux_long)
  PRINT, FORMAT='("  Short channel range: ", E9.2, " to ", E9.2, " W/m^2")', $
    MIN(goes_data.flux_short), MAX(goes_data.flux_short)

  ;--- Step 3: Find flares ---
  PRINT, ''
  PRINT, '--- Flare Detection ---'
  flares = find_flares(goes_data.time_jd, goes_data.flux_long, $
                        THRESHOLD=1.0D-7)

  ;--- Step 4: Screen plot ---
  PRINT, ''
  PRINT, '--- Creating Screen Plot ---'
  plot_light_curve, goes_data, flares

  ;--- Step 5: PostScript output ---
  PRINT, ''
  PRINT, '--- Creating PostScript ---'
  ps_file = 'goes_light_curve.eps'
  plot_light_curve, goes_data, flares, TO_FILE=ps_file

  ;--- Summary ---
  PRINT, ''
  PRINT, '============================================='
  PRINT, '   Analysis Complete'
  PRINT, '============================================='
  PRINT, 'FITS data: ' + fits_file
  PRINT, 'PostScript: ' + ps_file
  IF N_ELEMENTS(flares) GT 0 THEN BEGIN
    PRINT, FORMAT='("Flares detected: ", I0)', N_ELEMENTS(flares)
    max_flare = flares[WHERE(flares.peak_flux EQ MAX(flares.peak_flux))]
    PRINT, 'Largest flare: ' + max_flare[0].class
  ENDIF
END
```

---

## Step 7: Run the Project

```idl
; Method 1: Generate data and plot
IDL> .RUN goes_light_curve
IDL> goes_light_curve

; Method 2: Read from existing FITS file
IDL> goes_light_curve, /FROM_FILE, FITS_FILE='goes_xrs_sim.fits'
```

Expected output:

```
=============================================
   GOES Solar X-ray Light Curve Analysis
=============================================

Generating simulated GOES data...
Saving to FITS: goes_xrs_sim.fits
FITS file saved: goes_xrs_sim.fits

--- Data Summary ---
  Data points: 1440
  Long channel range:  3.20E-08 to  1.52E-04 W/m^2
  Short channel range:  2.99E-09 to  1.25E-05 W/m^2

--- Flare Detection ---
Found 4 flare events:
  C3.0   at 04:00 UT  (3.00E-06 W/m^2)
  M2.0   at 10:30 UT  (2.00E-05 W/m^2)
  B5.0   at 16:00 UT  (5.00E-07 W/m^2)
  X1.5   at 19:00 UT  (1.50E-04 W/m^2)

--- Creating Screen Plot ---
--- Creating PostScript ---
PostScript saved: goes_light_curve.eps

=============================================
   Analysis Complete
=============================================
FITS data: goes_xrs_sim.fits
PostScript: goes_light_curve.eps
Flares detected: 4
Largest flare: X1.5
```

---

## Extensions and Exercises

1. **Multi-Day Plot**: Modify the program to handle multiple days of data and plot with date tick labels on the X axis.
2. **Peak Detection Refinement**: Improve the flare detection by computing the rise time and decay time for each flare.
3. **Background Subtraction**: Implement a running median background subtraction to isolate flare emission above the quiet-Sun level.
4. **Derivative Plot**: Add a second panel showing the time derivative of the flux (dF/dt), useful for identifying impulsive phase of flares.
5. **Energy Estimation**: Integrate the flux above background to estimate the total radiated energy of each flare event.

---

## Summary

This project integrated the following IDL skills:

| Skill | How It Was Used |
|-------|----------------|
| Variables & Data Types | Double-precision time and flux arrays |
| Arrays & Operations | Array creation, WHERE filtering, array math |
| Operators | Relational operators for flare classification |
| Control Flow | FOR loops, IF/THEN/ELSE, CASE for classification |
| Procedures & Functions | Modular design with separate routines |
| String Processing | Formatting labels and FITS header values |
| File I/O | FITS reading/writing with READFITS/WRITEFITS |
| Structures | Flare records as structure arrays |
| Plotting | PLOT, OPLOT, XYOUTS for publication figures |
| Date & Time | Julian dates, CALDAT, time axis formatting |
| Best Practices | Vectorization, error checking, documentation |

Congratulations on completing IDL Basics! You now have the foundation to work with scientific data in IDL/GDL for solar physics, space science, and beyond.

---

**Previous**: [Debugging and Best Practices](./14_Debugging_and_Best_Practices.md)
