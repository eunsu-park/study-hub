; 15 Project: Solar Light Curve
; =============================
; Complete example: generate simulated GOES X-ray data,
; detect flares, and plot a publication-quality light curve.

FUNCTION ex15_flare_profile, time_jd, peak_jd, peak_flux, rise_min, decay_min
  dt = (time_jd - peak_jd) * 1440.0D0
  rise = EXP(-(dt < 0)^2 / (2.0D * rise_min^2))
  decay = EXP(-(dt > 0) / decay_min)
  RETURN, peak_flux * (rise * (dt LE 0) + decay * (dt GT 0))
END

PRO example_15_solar_light_curve

  PRINT, '===== Solar Light Curve Example ====='

  ; Generate simulated data
  n = 1440L
  start_jd = JULDAY(7, 15, 2024, 0, 0, 0)
  time_jd = start_jd + DINDGEN(n) * 60.0D0 / 86400.0D0
  seed = 42L

  ; Background + flares
  flux = 5.0D-8 + RANDOMN(seed, n) * 5.0D-9
  flux = flux > 1.0D-9
  flux += ex15_flare_profile(time_jd, start_jd + 4.0D/24, 3.0D-6, 10.0D, 30.0D)
  flux += ex15_flare_profile(time_jd, start_jd + 10.5D/24, 2.0D-5, 8.0D, 45.0D)
  flux += ex15_flare_profile(time_jd, start_jd + 19.0D/24, 1.5D-4, 6.0D, 60.0D)

  ; Statistics
  PRINT, FORMAT='("Data points: ", I0)', n
  PRINT, FORMAT='("Flux range: ", E9.2, " to ", E9.2, " W/m^2")', MIN(flux), MAX(flux)

  ; Plot
  time_hours = (time_jd - start_jd) * 24.0D0

  PLOT, time_hours, flux, /YLOG, $
    YRANGE=[1e-8, 1e-3], XRANGE=[0, 24], $
    XSTYLE=1, YSTYLE=1, $
    TITLE='GOES X-ray Flux (15 Jul 2024)', $
    XTITLE='Time (UT hours)', $
    YTITLE='Flux (W m!U-2!N)', $
    THICK=2, CHARSIZE=1.3

  ; Flare class lines
  levels = [1e-7, 1e-6, 1e-5, 1e-4]
  labels = ['B', 'C', 'M', 'X']
  FOR i = 0, 3 DO BEGIN
    OPLOT, [0, 24], [levels[i], levels[i]], LINESTYLE=1
    XYOUTS, 24.3, levels[i], labels[i], CHARSIZE=1.0
  ENDFOR

  PRINT, '===== Example 15 complete ====='
END
