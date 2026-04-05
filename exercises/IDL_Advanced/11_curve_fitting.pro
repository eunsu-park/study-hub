;+
; Exercise 11: Curve Fitting
;-

PRO exercise_11

    ; === Exercise 1: Linear regression ===
    ; Generate data: y = 3.5*x - 2.0 + noise(sigma=1.0), n=50
    ; Fit with LINFIT, print parameters and uncertainties
    ; Compute reduced chi-square
    ; TODO: Generate data, LINFIT, compute chi2_red

    ; === Exercise 2: Gaussian fitting ===
    ; Generate a Gaussian with A=10, center=5, sigma=1.5, background=2
    ; Fit with GAUSSFIT(NTERMS=4)
    ; Compute and print the FWHM
    ; TODO: Generate, GAUSSFIT, print results

    ; === Exercise 3: Double Gaussian ===
    ; Generate a signal with two overlapping Gaussians
    ; (peaks at x=3 and x=7, different amplitudes)
    ; Define a double-Gaussian function and fit with CURVEFIT
    ; TODO: Define function, set initial guesses, CURVEFIT

    ; === Exercise 4: MPFITFUN with constraints ===
    ; Fit an exponential decay y = A*exp(-t/tau) + C
    ; Constrain A > 0, tau > 0, C >= 0
    ; Use PARINFO to set constraints
    ; TODO: Define function, set PARINFO, MPFITFUN
    ; Hint: parinfo[i].limited = [1, 0] for lower bound only

    ; === Exercise 5: Spectral line fitting ===
    ; Generate a synthetic emission line at 195.12 A with:
    ; width = 0.05 A, amplitude = 80 DN, background = 15 DN
    ; Fit the line and compute:
    ; (a) Doppler velocity if measured center is 195.15 A
    ; (b) Thermal ion temperature from the width
    ; TODO: MPFITFUN, compute v = (lambda - lambda0)/lambda0 * c

END
