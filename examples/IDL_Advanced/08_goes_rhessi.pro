;+
; 08_goes_rhessi.pro — Lesson 08: GOES and RHESSI
;
; Demonstrates GOES X-ray light curve plotting with flare class labels.
; Requires: SolarSoft with GOES package
;-

PRO goes_demo
    rd_goes, data, tarray, TSTART='2024-01-15', TEND='2024-01-16', /ONE_MINUTE

    t_hours = (tarray - tarray[0]) / 3600.0

    WINDOW, 0, XSIZE=900, YSIZE=500
    PLOT, t_hours, data.lo, /YLOG, $
        XTITLE='Time (hours)', YTITLE='Flux (W m!U-2!N)', $
        TITLE='GOES 1-8 A X-ray Flux', $
        YRANGE=[1e-9, 1e-3], YSTYLE=1

    ; Flare class labels
    classes = ['A', 'B', 'C', 'M', 'X']
    levels = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    FOR i = 0, 4 DO BEGIN
        PLOTS, [0, MAX(t_hours)], [levels[i], levels[i]], LINESTYLE=1, COLOR=150
        XYOUTS, MAX(t_hours)*0.95, levels[i]*1.2, classes[i], CHARSIZE=1.0
    ENDFOR

    ; Overplot short channel
    OPLOT, t_hours, data.hi, LINESTYLE=2, COLOR=200

    PRINT, 'Max 1-8A flux: ', MAX(data.lo), ' W/m^2'
END

goes_demo
END
