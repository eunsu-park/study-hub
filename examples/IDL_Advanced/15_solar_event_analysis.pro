;+
; 15_solar_event_analysis.pro — Lesson 15: Capstone Solar Event Analysis
;
; Demonstrates end-to-end flare analysis workflow with synthetic data.
; (Uses synthetic data so it runs without real FITS files)
;-

PRO solar_event_demo
    PRINT, '=== Capstone: Solar Event Analysis (Synthetic Data) ==='

    ; Synthetic parameters
    nx = 128 & ny = 128 & nt = 120
    cadence = 60.0  ; seconds
    t = FINDGEN(nt) * cadence

    ; Simulate a flare in 171 A
    cube = FLTARR(nx, ny, nt)
    flare_x = 64 & flare_y = 64

    FOR i = 0, nt-1 DO BEGIN
        ; Background
        frame = DIST(nx) * 10.0 + 100.0

        ; Flare: Gaussian brightening peaking at t=50 min
        t_min = t[i] / 60.0
        flare_amp = 500.0 * EXP(-0.5*((t_min - 50.0)/10.0)^2)
        IF flare_amp GT 5 THEN BEGIN
            FOR ix = -15, 15 DO FOR iy = -15, 15 DO BEGIN
                r2 = ix^2 + iy^2
                IF r2 LT 225 THEN $
                    frame[(flare_x+ix)>0<(nx-1), (flare_y+iy)>0<(ny-1)] += $
                        flare_amp * EXP(-r2/50.0)
            ENDFOR
        ENDIF

        cube[*, *, i] = frame + RANDOMN(seed, nx, ny) * 10.0
    ENDFOR

    ; Extract light curve
    hw = 10
    roi = cube[flare_x-hw:flare_x+hw, flare_y-hw:flare_y+hw, *]
    lightcurve = REFORM(MEAN(MEAN(roi, DIMENSION=1), DIMENSION=1))
    t_min = t / 60.0

    ; Flare onset detection
    pre = WHERE(t_min LT 20.0)
    baseline = MEAN(lightcurve[pre])
    bstd = STDDEV(lightcurve[pre])
    thresh = baseline + 3.0 * bstd
    onset = (WHERE(lightcurve GT thresh AND t_min GT 20.0))[0]
    peak = (WHERE(lightcurve EQ MAX(lightcurve)))[0]

    PRINT, 'Baseline:    ', baseline, ' DN/s'
    PRINT, 'Onset:       t = ', t_min[onset], ' min'
    PRINT, 'Peak:        t = ', t_min[peak], ' min'
    PRINT, 'Enhancement: ', MAX(lightcurve)/baseline, 'x'

    ; Running difference
    run_diff = cube[*, *, 1:nt-1] - cube[*, *, 0:nt-2]

    ; Multi-panel plot
    WINDOW, 0, XSIZE=800, YSIZE=800
    !P.MULTI = [0, 2, 2]
    LOADCT, 3
    TV, BYTSCL(CONGRID(cube[*, *, peak], 400, 400), MIN=50, MAX=800), 0
    LOADCT, 33
    TV, BYTSCL(CONGRID(run_diff[*, *, (peak-1)>0], 400, 400), MIN=-200, MAX=200), 1
    LOADCT, 0
    PLOT, t_min, lightcurve, XTITLE='Time (min)', YTITLE='DN/s', $
        TITLE='Flare Light Curve'
    PLOTS, [t_min[onset], t_min[onset]], !Y.CRANGE, LINESTYLE=2, COLOR=200
    PLOTS, [t_min[peak], t_min[peak]], !Y.CRANGE, LINESTYLE=1, COLOR=150
    PLOTS, [0, MAX(t_min)], [thresh, thresh], LINESTYLE=3

    ; Frame statistics
    means = FLTARR(nt)
    FOR i = 0, nt-1 DO means[i] = MEAN(cube[*, *, i])
    PLOT, t_min, means, XTITLE='Time (min)', YTITLE='Mean DN/s', $
        TITLE='Full-Frame Mean'
    !P.MULTI = 0

    PRINT, '=== Demo Complete ==='
END

solar_event_demo
END
