;+
; Exercise 07: SDO/HMI Analysis
; Requires: SolarSoft with HMI, and HMI FITS file(s)
;-

PRO exercise_07

    ; === Exercise 1: Magnetogram display ===
    ; Read an HMI magnetogram and display with blue-red color table
    ; Draw the solar limb circle on top
    ; TODO: read_sdo, LOADCT 33, TV/BYTSCL, PLOTS for circle

    ; === Exercise 2: Flux calculation ===
    ; Compute positive, negative, and total unsigned flux for a 400x400 pixel ROI
    ; Convert pixel area to cm^2 using header keywords
    ; TODO: Extract ROI, compute pixel_area, use WHERE and TOTAL

    ; === Exercise 3: Mu-correction ===
    ; Apply cos(theta) correction to convert B_LOS to B_radial
    ; Create a mu map and display it
    ; TODO: Compute rho, mu, B_r = B_LOS / mu (where mu > 0.3)

    ; === Exercise 4: Flux time series ===
    ; Given 10 HMI magnetogram files, compute the unsigned flux of a
    ; fixed ROI at each time step. Plot flux vs time.
    ; TODO: Loop, extract ROI, compute flux, plot

    ; === Exercise 5: Histogram of field strengths ===
    ; Compute and plot a histogram of magnetic field strengths for the full disk
    ; Mask off-disk pixels. Use log scale for the y-axis.
    ; TODO: Compute rho mask, use HISTOGRAM, PLOT /YLOG

END
