;+
; Exercise 08: GOES and RHESSI
; Requires: SolarSoft with GOES package
;-

PRO exercise_08

    ; === Exercise 1: GOES light curve ===
    ; Plot GOES 1-8A data for any day, add flare class lines and labels
    ; TODO: rd_goes, PLOT /YLOG, add horizontal lines at class boundaries

    ; === Exercise 2: Flare detection ===
    ; Implement a simple flare detector: find all intervals where
    ; GOES flux exceeds C1.0 level (1e-6 W/m^2)
    ; Report start time, peak time, peak flux for each flare
    ; TODO: Use WHERE to find above-threshold intervals
    ; TODO: Group contiguous intervals
    ; Hint: breaks = WHERE(idx[1:*] - idx[0:n-2] GT 1)

    ; === Exercise 3: GOES temperature ===
    ; Estimate plasma temperature from the 0.5-4A / 1-8A flux ratio
    ; T ~ 0.09 * (ratio)^(-0.45) keV (approximate formula)
    ; Plot temperature vs time during a flare
    ; TODO: rd_goes, compute ratio, apply formula, plot

    ; === Exercise 4: GOES event list ===
    ; Query the flare event list for one month
    ; Count flares by class (A, B, C, M, X)
    ; TODO: rd_gev, parse class field, count each category

    ; === Exercise 5: Multi-instrument plot ===
    ; Create a 2-panel plot:
    ; Top: GOES 1-8A light curve
    ; Bottom: Derivative of GOES flux (proxy for hard X-ray emission)
    ; TODO: Compute dI/dt using DERIV or manual differencing

END
