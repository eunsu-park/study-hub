;+
; 11_curve_fitting.pro — Lesson 11: Curve Fitting
;
; Demonstrates LINFIT, POLY_FIT, GAUSSFIT, and CURVEFIT.
;-

PRO curve_fitting_demo
    ; Linear fit
    x = FINDGEN(30) + 1.0
    y = 2.5*x + 10.0 + RANDOMN(seed, 30)*3.0
    result = LINFIT(x, y, SIGMA=sigma)
    PRINT, '--- LINFIT ---'
    PRINT, 'Intercept: ', result[0], ' +/- ', sigma[0]
    PRINT, 'Slope:     ', result[1], ' +/- ', sigma[1]

    ; Polynomial fit (quadratic)
    x2 = FINDGEN(50) * 0.1
    y2 = 3.0 - 2.0*x2 + 0.5*x2^2 + RANDOMN(seed, 50)*0.3
    coeffs = POLY_FIT(x2, y2, 2, SIGMA=sig2)
    PRINT, '--- POLY_FIT (deg 2) ---'
    PRINT, 'Coefficients: ', coeffs

    ; Gaussian fit
    x3 = FINDGEN(200)*0.1 - 10.0
    y3 = 5.0*EXP(-0.5*((x3-1.5)/2.0)^2) + 0.5 + RANDOMN(seed, 200)*0.2
    yfit = GAUSSFIT(x3, y3, gcoeffs, NTERMS=4)
    PRINT, '--- GAUSSFIT ---'
    PRINT, 'Amplitude: ', gcoeffs[0]
    PRINT, 'Center:    ', gcoeffs[1]
    PRINT, 'Sigma:     ', gcoeffs[2]
    PRINT, 'FWHM:      ', 2.354 * gcoeffs[2]

    ; Plot results
    WINDOW, 0, XSIZE=800, YSIZE=600
    !P.MULTI = [0, 1, 3]
    PLOT, x, y, PSYM=4, TITLE='Linear Fit'
    OPLOT, x, result[0]+result[1]*x, THICK=2

    PLOT, x2, y2, PSYM=1, TITLE='Quadratic Fit'
    OPLOT, x2, coeffs[0]+coeffs[1]*x2+coeffs[2]*x2^2, THICK=2

    PLOT, x3, y3, PSYM=3, TITLE='Gaussian Fit'
    OPLOT, x3, yfit, THICK=2, COLOR=250
    !P.MULTI = 0
END

curve_fitting_demo
END
