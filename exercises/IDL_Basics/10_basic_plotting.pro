; Exercise 10: Basic Plotting
;
; Practice creating plots with various styles and annotations.

; Exercise 1: Create a plot of sin(x), cos(x), and sin(x)*cos(x)
; on the same axes with different line styles, a legend, and title.
PRO exercise_10a
  ; TODO: Create x = FINDGEN(360) * !DTOR
  ; TODO: PLOT sin(x) with solid line
  ; TODO: OPLOT cos(x) with dashed line
  ; TODO: OPLOT sin(x)*cos(x) with dash-dot line
  ; TODO: Add legend using XYOUTS and PLOTS
  ; TODO: Add title and axis labels
END

; Exercise 2: Create a 2x2 multi-panel plot showing:
; top-left: sin(x), top-right: cos(x),
; bottom-left: exp(-x/5), bottom-right: histogram of random data.
PRO exercise_10b
  ; TODO: Set !P.MULTI = [0, 2, 2]
  ; TODO: Create each panel with PLOT
  ; TODO: For histogram, use HISTOGRAM function and PLOT with PSYM=10
  ; TODO: Reset !P.MULTI = 0
END

; Exercise 3: Create a log-log plot of frequency vs. power
; for a synthetic power spectrum: P(f) = A * f^(-alpha).
PRO exercise_10c
  ; TODO: Create frequency array from 0.01 to 100 (log-spaced)
  ; TODO: Compute power with alpha = 1.5 and A = 100
  ; TODO: Plot with /XLOG, /YLOG
  ; TODO: Add proper axis labels
END

; Exercise 4: Create a PostScript file containing a publication-quality
; plot with thick lines, large fonts, and proper formatting.
PRO exercise_10d
  ; TODO: SET_PLOT, 'PS'
  ; TODO: DEVICE, FILENAME='exercise10.eps', /COLOR, /ENCAPSULATED
  ; TODO: Plot some data with THICK=3, CHARSIZE=1.2
  ; TODO: DEVICE, /CLOSE
  ; TODO: SET_PLOT back to original device
  ; TODO: Delete the temp file
END
