; 10 Basic Plotting
; =================
; Demonstrates PLOT, OPLOT, XYOUTS, multi-panel plots,
; and logarithmic axes.

PRO example_10_plotting

  x = FINDGEN(360) * !DTOR
  y_sin = SIN(x)
  y_cos = COS(x)

  ; Basic plot with title and labels
  PLOT, x / !DTOR, y_sin, $
    TITLE='Trigonometric Functions', $
    XTITLE='Angle (degrees)', $
    YTITLE='Value', $
    XSTYLE=1, YSTYLE=1, $
    YRANGE=[-1.5, 1.5], $
    CHARSIZE=1.3, THICK=2

  ; Overlay cosine
  OPLOT, x / !DTOR, y_cos, LINESTYLE=2, THICK=2

  ; Text annotations
  XYOUTS, 90, 1.1, 'sin(x)', CHARSIZE=1.1
  XYOUTS, 0, 1.1, 'cos(x)', CHARSIZE=1.1

  ; Wait for user
  PRINT, 'Press Enter for multi-panel plot...'
  tmp = ''

  ; Multi-panel plot
  !P.MULTI = [0, 2, 2]

  PLOT, x / !DTOR, SIN(x), TITLE='sin(x)', CHARSIZE=1.5
  PLOT, x / !DTOR, COS(x), TITLE='cos(x)', CHARSIZE=1.5
  PLOT, x / !DTOR, SIN(2*x), TITLE='sin(2x)', CHARSIZE=1.5
  PLOT, x / !DTOR, COS(2*x), TITLE='cos(2x)', CHARSIZE=1.5

  !P.MULTI = 0

  ; Log plot
  freq = 10.0^(FINDGEN(40) / 10.0)
  power = 1.0 / freq^1.5
  PLOT, freq, power, /XLOG, /YLOG, $
    TITLE='Power Spectrum', $
    XTITLE='Frequency', YTITLE='Power', $
    XSTYLE=1, THICK=2, CHARSIZE=1.3

  PRINT, 'Example 10 complete.'
END
