; Exercise 15: Project - Solar Light Curve
;
; Complete project exercises combining all skills learned.

; Exercise 1: Write a flare profile function that generates a
; realistic flare temporal profile (fast rise, exponential decay).
FUNCTION exercise_15a, time_minutes, peak_time, peak_flux, rise_time, decay_time
  ; TODO: Compute dt = time - peak_time (in minutes)
  ; TODO: For dt <= 0: Gaussian rise = peak_flux * exp(-dt^2 / (2*rise^2))
  ; TODO: For dt > 0: exponential decay = peak_flux * exp(-dt / decay_time)
  ; TODO: Combine and return the profile array
  ; Hint: Use (dt LE 0) and (dt GT 0) as boolean masks
  RETURN, DBLARR(N_ELEMENTS(time_minutes))
END

; Exercise 2: Write a flare classifier function that takes a peak
; flux value and returns the GOES class string (e.g., 'M2.5').
FUNCTION exercise_15b, peak_flux
  ; TODO: Classify based on flux thresholds:
  ;   X >= 1e-4, M >= 1e-5, C >= 1e-6, B >= 1e-7, A < 1e-7
  ; TODO: Compute the sub-class number (e.g., M2.5 = 2.5e-5)
  ; TODO: Return formatted string like 'M2.5' or 'X1.0'
  ; Hint: class_num = peak_flux / threshold for that class
  RETURN, ''
END

; Exercise 3: Write a procedure that creates a complete light curve
; plot with:
; - Logarithmic Y axis
; - Flare classification level lines (B, C, M, X)
; - Time in UT hours on X axis
; - Title with date
; - Legend for channels
PRO exercise_15c, time_hours, flux_long, flux_short
  ; TODO: PLOT flux_long with /YLOG, YRANGE=[1e-8, 1e-3]
  ; TODO: OPLOT flux_short with different linestyle
  ; TODO: Draw horizontal lines at 1e-7, 1e-6, 1e-5, 1e-4
  ; TODO: Label each line on the right side
  ; TODO: Add legend, title, and axis labels
END

; Exercise 4: Put it all together. Generate 48 hours of simulated
; GOES data with at least 3 flares of different classes, detect the
; flares, classify them, and produce both screen and PostScript plots.
PRO exercise_15d
  ; TODO: Generate time array (48 hours, 1-minute cadence)
  ; TODO: Create background flux (~5e-8 with noise)
  ; TODO: Add 3+ flares using exercise_15a
  ; TODO: Detect peaks above C1.0 threshold
  ; TODO: Classify each using exercise_15b
  ; TODO: Print flare summary table
  ; TODO: Create screen plot using exercise_15c
  ; TODO: Create PostScript output
  ; TODO: Clean up PostScript file
END
