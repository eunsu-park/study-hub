# 11. Curve Fitting

**Previous**: [Image Processing](./10_Image_Processing.md) | **Next**: [NetCDF and HDF5](./12_NetCDF_and_HDF5.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Fit polynomials with POLY_FIT and lines with LINFIT
2. Fit Gaussians with GAUSSFIT
3. Use CURVEFIT for nonlinear least-squares fitting
4. Apply MPFIT (Markwardt) for robust parameter estimation
5. Perform chi-square analysis and compute confidence intervals
6. Define custom fit functions for scientific models

---

## 1. Linear and Polynomial Fitting

### LINFIT — Linear Regression

```idl
; Simple linear fit: y = a + b*x
x = FINDGEN(20) + 1.0
y = 2.5 * x + 10.0 + RANDOMN(seed, 20) * 3.0

; Fit
result = LINFIT(x, y, SIGMA=sigma)
; result[0] = intercept, result[1] = slope

PRINT, 'Intercept: ', result[0], ' +/- ', sigma[0]
PRINT, 'Slope:     ', result[1], ' +/- ', sigma[1]

; With weights (inverse variance)
errors = REPLICATE(3.0, 20)
weights = 1.0 / errors^2
result = LINFIT(x, y, MEASURE_ERRORS=errors, SIGMA=sigma)

; Plot
PLOT, x, y, PSYM=4, TITLE='Linear Fit'
OPLOT, x, result[0] + result[1]*x, THICK=2
```

### POLY_FIT — Polynomial Regression

```idl
; Polynomial fit: y = c0 + c1*x + c2*x^2 + ...
x = FINDGEN(50) * 0.1
y = 3.0 - 2.0*x + 0.5*x^2 + RANDOMN(seed, 50) * 0.5

; Quadratic fit (degree 2)
coeffs = POLY_FIT(x, y, 2, SIGMA=sigma, CHISQ=chisq, YFIT=yfit)
PRINT, 'Coefficients: ', coeffs
PRINT, 'Uncertainties: ', sigma
PRINT, 'Chi-square:    ', chisq

; Plot
PLOT, x, y, PSYM=1, TITLE='Polynomial Fit (degree 2)'
OPLOT, x, yfit, THICK=2, COLOR=250

; Higher degree: check for overfitting
FOR deg = 1, 5 DO BEGIN
    c = POLY_FIT(x, y, deg, CHISQ=chi2)
    PRINT, 'Degree ', deg, ': chi2 = ', chi2
ENDFOR
```

### SVDFIT — SVD-Based Fitting

```idl
; SVDFIT uses Singular Value Decomposition for robust fitting
; Better numerical stability for ill-conditioned problems

; Fit with Legendre polynomials
result = SVDFIT(x, y, 3, SIGMA=sigma, CHISQ=chisq, $
    FUNCTION_NAME='LEGENDRE')

; Custom basis functions using SVDFIT
; Define a function that returns basis vectors
; SVDFIT calls this function with x and returns [n_data, n_basis] array
```

---

## 2. Gaussian Fitting

### GAUSSFIT

```idl
; Fit a Gaussian: y = A0 * exp(-((x-A1)/A2)^2 / 2) + A3 + A4*x + A5*x^2

; Generate test data
x = FINDGEN(200) * 0.1 - 10.0
amp = 5.0  & center = 1.5  & width = 2.0
y = amp * EXP(-0.5*((x - center)/width)^2) + $
    0.5 + RANDOMN(seed, 200) * 0.2

; Fit Gaussian (with linear background)
yfit = GAUSSFIT(x, y, coeffs, NTERMS=4)
; coeffs = [amplitude, center, sigma, constant_bg]

PRINT, 'Amplitude: ', coeffs[0]
PRINT, 'Center:    ', coeffs[1]
PRINT, 'Sigma:     ', coeffs[2]
PRINT, 'Background:', coeffs[3]
PRINT, 'FWHM:      ', 2.354 * coeffs[2]  ; FWHM = 2*sqrt(2*ln(2)) * sigma

; Plot
PLOT, x, y, PSYM=3, TITLE='Gaussian Fit'
OPLOT, x, yfit, COLOR=250, THICK=2
```

### Multiple Gaussian Fitting

```idl
; For multiple Gaussians, use CURVEFIT or MPFIT with a custom function
; (GAUSSFIT only fits a single Gaussian)

; Define double Gaussian function
PRO double_gauss, x, p, ymod, pder
    ; p = [A1, c1, w1, A2, c2, w2, bg]
    g1 = p[0] * EXP(-0.5*((x - p[1])/p[2])^2)
    g2 = p[3] * EXP(-0.5*((x - p[4])/p[5])^2)
    ymod = g1 + g2 + p[6]

    ; Partial derivatives (optional, for faster convergence)
    IF N_PARAMS() GE 4 THEN BEGIN
        pder = FLTARR(N_ELEMENTS(x), 7)
        pder[*, 0] = g1 / p[0]
        pder[*, 1] = g1 * (x - p[1]) / p[2]^2
        pder[*, 2] = g1 * (x - p[1])^2 / p[2]^3
        pder[*, 3] = g2 / p[3]
        pder[*, 4] = g2 * (x - p[4]) / p[5]^2
        pder[*, 5] = g2 * (x - p[4])^2 / p[5]^3
        pder[*, 6] = REPLICATE(1.0, N_ELEMENTS(x))
    ENDIF
END
```

---

## 3. CURVEFIT — Nonlinear Least-Squares

### Basic Usage

```idl
; CURVEFIT minimizes chi-square for a user-defined function
; The function must have the signature: PRO func, x, params, ymodel, pder

; Define an exponential decay function
PRO exp_decay, x, p, ymod, pder
    ; p = [amplitude, decay_rate, offset]
    ymod = p[0] * EXP(-p[1] * x) + p[2]

    IF N_PARAMS() GE 4 THEN BEGIN
        pder = FLTARR(N_ELEMENTS(x), 3)
        pder[*, 0] = EXP(-p[1] * x)
        pder[*, 1] = -p[0] * x * EXP(-p[1] * x)
        pder[*, 2] = 1.0
    ENDIF
END

; Generate test data
x = FINDGEN(100) * 0.1
y_true = 10.0 * EXP(-0.3 * x) + 2.0
y = y_true + RANDOMN(seed, 100) * 0.5

; Initial guess
params = [8.0, 0.2, 1.5]

; Fit
weights = REPLICATE(1.0/0.5^2, 100)  ; 1/sigma^2
yfit = CURVEFIT(x, y, weights, params, sigma, $
    FUNCTION_NAME='exp_decay', $
    CHISQ=chisq, /NODERIVATIVE)

PRINT, 'Amplitude:  ', params[0], ' +/- ', sigma[0]
PRINT, 'Decay rate: ', params[1], ' +/- ', sigma[1]
PRINT, 'Offset:     ', params[2], ' +/- ', sigma[2]
PRINT, 'Chi-sq/dof: ', chisq / (100 - 3)

; Plot
PLOT, x, y, PSYM=3, TITLE='Exponential Decay Fit'
OPLOT, x, yfit, COLOR=250, THICK=2
```

### CURVEFIT Keywords

| Keyword | Description |
|---------|-------------|
| `FUNCTION_NAME` | Name of the fit function procedure |
| `CHISQ` | Output chi-square value |
| `SIGMA` | Output parameter uncertainties |
| `ITMAX` | Maximum iterations (default 20) |
| `TOL` | Convergence tolerance |
| `/NODERIVATIVE` | Compute derivatives numerically |
| `STATUS` | 0=converged, 1=chi-sq increasing, 2=max iterations |

---

## 4. MPFIT — Markwardt Levenberg-Marquardt

MPFIT is the gold standard for curve fitting in IDL. It provides parameter constraints, fixed parameters, and better convergence than CURVEFIT.

### Installation

```idl
; MPFIT is available from:
; https://pages.physics.wisc.edu/~craigm/idl/fitting.html
; Or included in SolarSoft: $SSW/gen/idl/fitting/

; Key routines:
; MPFIT      — General minimization
; MPFITFUN   — Function fitting (easiest to use)
; MPFITPEAK  — Peak (Gaussian, Lorentzian) fitting
; MPFITEXPR  — Fit an expression string
```

### MPFITFUN — Function Fitting

```idl
; Define fit function (different signature from CURVEFIT)
FUNCTION my_model, x, p
    ; p = [amplitude, center, width, background]
    RETURN, p[0] * EXP(-0.5*((x - p[1])/p[2])^2) + p[3]
END

; Generate data
x = FINDGEN(200) * 0.1 - 10.0
y_true = 5.0 * EXP(-0.5*((x - 2.0)/1.5)^2) + 1.0
yerr = REPLICATE(0.3, 200)
y = y_true + RANDOMN(seed, 200) * 0.3

; Initial guess
p0 = [4.0, 1.0, 2.0, 0.5]

; Fit with MPFITFUN
params = MPFITFUN('my_model', x, y, yerr, p0, $
    PERROR=perror, BESTNORM=bestnorm, DOF=dof, $
    STATUS=status)

PRINT, 'Status: ', status  ; 1-4 = success
PRINT, 'Reduced chi-sq: ', bestnorm / dof
PRINT, 'Parameters:'
PRINT, '  Amplitude: ', params[0], ' +/- ', perror[0]
PRINT, '  Center:    ', params[1], ' +/- ', perror[1]
PRINT, '  Width:     ', params[2], ' +/- ', perror[2]
PRINT, '  Background:', params[3], ' +/- ', perror[3]
```

### Parameter Constraints with PARINFO

```idl
; PARINFO structure controls parameter behavior
n_params = 4
parinfo = REPLICATE({value: 0.D, fixed: 0, limited: [0, 0], $
    limits: [0.D, 0.D], step: 0.D, tied: ''}, n_params)

; Set initial values
parinfo[0].value = 4.0   ; Amplitude
parinfo[1].value = 1.0   ; Center
parinfo[2].value = 2.0   ; Width
parinfo[3].value = 0.5   ; Background

; Constrain amplitude > 0
parinfo[0].limited = [1, 0]  ; Lower limit active
parinfo[0].limits[0] = 0.0

; Constrain width > 0
parinfo[2].limited = [1, 0]
parinfo[2].limits[0] = 0.01

; Fix the background parameter
; parinfo[3].fixed = 1
; parinfo[3].value = 1.0

; Tie parameters (e.g., param 2 = 2 * param 0)
; parinfo[2].tied = '2.0 * P[0]'

; Fit with constraints
params = MPFITFUN('my_model', x, y, yerr, $
    PARINFO=parinfo, PERROR=perror, STATUS=status)
```

### MPFITPEAK — Specialized Peak Fitting

```idl
; Fit a Gaussian peak with MPFITPEAK
yfit = MPFITPEAK(x, y, params, NTERMS=4, /GAUSSIAN, $
    PERROR=perror, SIGMA=yerr)

; params = [peak, center, width, background]
PRINT, 'Peak:       ', params[0]
PRINT, 'Center:     ', params[1]
PRINT, 'Width:      ', params[2]
PRINT, 'Background: ', params[3]

; Also supports Lorentzian and Moffat profiles
yfit_lor = MPFITPEAK(x, y, params_lor, NTERMS=4, /LORENTZIAN)
```

---

## 5. Chi-Square Analysis

### Goodness of Fit

```idl
; Reduced chi-square: chi2_red = chi2 / dof
; dof = n_data - n_params

n_data = N_ELEMENTS(x)
n_params = 4
dof = n_data - n_params

; Compute chi-square manually
residuals = y - my_model(x, params)
chi2 = TOTAL((residuals / yerr)^2)
chi2_red = chi2 / dof

PRINT, 'Chi-square:         ', chi2
PRINT, 'Degrees of freedom: ', dof
PRINT, 'Reduced chi-square: ', chi2_red
; chi2_red ~ 1.0 indicates a good fit
; chi2_red >> 1 indicates poor fit or underestimated errors
; chi2_red << 1 indicates overestimated errors
```

### Confidence Intervals

```idl
; 1-sigma confidence interval from MPFIT: PERROR
; These are formal 1-sigma errors assuming chi2_red ~ 1

; Scale errors if chi2_red != 1
scaled_errors = perror * SQRT(chi2_red)

; For 2-sigma (95.4%): multiply by 2
; For 3-sigma (99.7%): multiply by 3

PRINT, '1-sigma errors: ', perror
PRINT, '2-sigma errors: ', 2.0 * perror
PRINT, '3-sigma errors: ', 3.0 * perror

; Delta-chi-square for confidence contours
; 1-param: delta_chi2 = 1.0 (68.3%), 4.0 (95.4%), 9.0 (99.7%)
; 2-param: delta_chi2 = 2.30 (68.3%), 6.17 (95.4%), 11.8 (99.7%)
```

### Parameter Correlation

```idl
; Covariance matrix from MPFIT
params = MPFITFUN('my_model', x, y, yerr, p0, $
    COVAR=covar, PERROR=perror)

; Covariance matrix
PRINT, 'Covariance matrix:'
PRINT, covar

; Correlation matrix
n_p = N_ELEMENTS(params)
corr = FLTARR(n_p, n_p)
FOR i = 0, n_p-1 DO $
    FOR j = 0, n_p-1 DO $
        corr[i, j] = covar[i, j] / (perror[i] * perror[j])

PRINT, 'Correlation matrix:'
PRINT, corr
```

---

## 6. Practical: Fitting Solar Spectral Lines

```idl
; Fit an emission line profile (e.g., from EIS or IRIS)
; Gaussian line + linear background

; Simulated spectral line
wavelength = FINDGEN(100) * 0.01 + 195.0  ; Angstroms
line_center = 195.12
line_width = 0.05  ; Angstroms
line_amp = 100.0
continuum = 10.0 + 0.5 * (wavelength - 195.0)
line_profile = line_amp * EXP(-0.5*((wavelength - line_center)/line_width)^2)
spectrum = line_profile + continuum + RANDOMN(seed, 100) * 5.0

; Error estimate
spec_err = SQRT(ABS(spectrum) > 1.0)

; Fit function: Gaussian + linear background
FUNCTION spectral_line, x, p
    ; p = [amp, center, width, bg_const, bg_slope]
    gaussian = p[0] * EXP(-0.5*((x - p[1])/p[2])^2)
    background = p[3] + p[4] * (x - MEAN(x))
    RETURN, gaussian + background
END

; Initial guess
p0 = [80.0, 195.1, 0.06, 10.0, 0.5]

; Constraints
parinfo = REPLICATE({value: 0.D, fixed: 0, limited: [0,0], $
    limits: [0.D, 0.D]}, 5)
parinfo[*].value = p0
parinfo[0].limited = [1, 0] & parinfo[0].limits[0] = 0  ; Amp > 0
parinfo[2].limited = [1, 0] & parinfo[2].limits[0] = 0.01  ; Width > 0

; Fit
params = MPFITFUN('spectral_line', wavelength, spectrum, spec_err, $
    PARINFO=parinfo, PERROR=perror, BESTNORM=chi2, DOF=dof)

; Results
PRINT, 'Line center:    ', params[1], ' +/- ', perror[1], ' A'
PRINT, 'Line width:     ', params[2], ' +/- ', perror[2], ' A'
PRINT, 'FWHM:           ', 2.354 * params[2], ' A'
PRINT, 'Line intensity: ', params[0], ' +/- ', perror[0]

; Doppler velocity from line shift
c = 3e5  ; km/s
rest_wavelength = 195.12  ; A
v_doppler = (params[1] - rest_wavelength) / rest_wavelength * c
v_err = perror[1] / rest_wavelength * c
PRINT, 'Doppler velocity: ', v_doppler, ' +/- ', v_err, ' km/s'

; Thermal width -> temperature
m_ion = 56.0 * 1.67e-24  ; Fe mass in grams
k_B = 1.38e-16           ; erg/K
lambda0_cm = rest_wavelength * 1e-8
sigma_cm = params[2] * 1e-8
T_ion = m_ion * (c * 1e5)^2 * (sigma_cm / lambda0_cm)^2 / k_B
PRINT, 'Ion temperature: ', T_ion, ' K'

; Plot
PLOT, wavelength, spectrum, PSYM=10, $
    XTITLE='Wavelength (A)', YTITLE='Intensity', $
    TITLE='Spectral Line Fit'
OPLOT, wavelength, spectral_line(wavelength, params), COLOR=250, THICK=2
```

---

## Summary

| Method | IDL Function | Best For |
|--------|-------------|----------|
| Linear | `LINFIT` | y = a + bx |
| Polynomial | `POLY_FIT` | y = sum(c_i * x^i) |
| SVD | `SVDFIT` | Ill-conditioned linear |
| Gaussian | `GAUSSFIT` | Single Gaussian peak |
| Nonlinear | `CURVEFIT` | General nonlinear models |
| MPFIT | `MPFITFUN` | Constrained nonlinear (recommended) |
| Peak | `MPFITPEAK` | Gaussian/Lorentzian peaks |

---

**Previous**: [Image Processing](./10_Image_Processing.md) | **Next**: [NetCDF and HDF5](./12_NetCDF_and_HDF5.md)
