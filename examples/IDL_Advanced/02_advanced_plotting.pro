;+
; 02_advanced_plotting.pro — Lesson 02: Advanced Plotting
;
; Demonstrates:
;   - Multi-panel plots with !P.MULTI
;   - Contour plots (filled and line)
;   - Surface plots (wireframe and shaded)
;   - PostScript output
;-

PRO advanced_plotting_demo

    x = FINDGEN(200) * 0.05
    y1 = SIN(x) & y2 = COS(x) & y3 = EXP(-x*0.3) & y4 = ALOG(x+1)

    ; --- Multi-panel ---
    WINDOW, 0, XSIZE=800, YSIZE=600
    !P.MULTI = [0, 2, 2]
    PLOT, x, y1, TITLE='Sine'
    PLOT, x, y2, TITLE='Cosine'
    PLOT, x, y3, TITLE='Exp Decay'
    PLOT, x, y4, TITLE='Logarithm'
    !P.MULTI = 0

    ; --- Contour plot ---
    data = DIST(100)
    WINDOW, 1, XSIZE=500, YSIZE=500
    LOADCT, 33
    CONTOUR, data, NLEVELS=15, /FILL, TITLE='Filled Contour'
    CONTOUR, data, NLEVELS=15, /OVERPLOT

    ; --- Surface plot ---
    WINDOW, 2, XSIZE=500, YSIZE=500
    SHADE_SURF, DIST(50), SHADES=BYTSCL(DIST(50)), $
        AX=45, AZ=30, TITLE='Shaded Surface'

    ; --- PostScript output ---
    SET_PLOT, 'PS'
    DEVICE, FILENAME='example_plot.ps', /ENCAPSULATED, $
        XSIZE=18, YSIZE=12, /COLOR, BITS=8
    !P.THICK = 3 & !P.CHARSIZE = 1.2
    PLOT, x, y1, XTITLE='X', YTITLE='Y', TITLE='Publication Plot'
    OPLOT, x, y2, LINESTYLE=2
    DEVICE, /CLOSE
    SET_PLOT, 'X'
    !P.THICK = 0 & !P.CHARSIZE = 0

    PRINT, 'Saved example_plot.ps'
END

advanced_plotting_demo
END
