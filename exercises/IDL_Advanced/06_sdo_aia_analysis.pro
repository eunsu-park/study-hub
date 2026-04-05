;+
; Exercise 06: SDO/AIA Analysis
; Requires: SolarSoft with AIA, and AIA FITS file(s)
;-

PRO exercise_06

    ; === Exercise 1: Read and display ===
    ; Read an AIA 171 FITS file, print all header keywords, display the image
    ; TODO: Use read_sdo, HELP /STRUCTURE, aia_lct, TV/BYTSCL

    ; === Exercise 2: Calibrate and compare ===
    ; Compare raw vs calibrated (aia_prep) pixel values for the same image
    ; (a) Print mean DN before and after aia_prep
    ; (b) Print pixel scale before and after /REGISTER
    ; TODO: read_sdo, aia_prep with /NORMALIZE /REGISTER

    ; === Exercise 3: AIA response functions ===
    ; Plot all 7 EUV channel response functions on a single graph
    ; Mark the peak temperature for each channel
    ; TODO: aia_get_response, PLOT/OPLOT, find peak with WHERE/MAX

    ; === Exercise 4: Channel ratio map ===
    ; Compute the 193/171 ratio for a co-temporal pair of images
    ; Display the ratio map with a diverging color table
    ; TODO: Read two files, calibrate, compute ratio, display

    ; === Exercise 5: Light curve extraction ===
    ; Given a series of AIA files, extract the mean intensity of a
    ; 100x100 pixel box centered at (2048, 2048) vs time
    ; Plot the light curve with time in minutes on the x-axis
    ; TODO: Loop over files, read_sdo, extract ROI, collect means

END
