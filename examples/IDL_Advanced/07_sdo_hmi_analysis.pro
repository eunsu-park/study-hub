;+
; 07_sdo_hmi_analysis.pro — Lesson 07: SDO/HMI Analysis
;
; Demonstrates HMI magnetogram reading, display, and flux calculation.
; Requires: SolarSoft with SDO/HMI package
;-

PRO hmi_analysis_demo, hmi_file
    IF N_ELEMENTS(hmi_file) EQ 0 THEN BEGIN
        PRINT, 'Usage: hmi_analysis_demo, "hmi_mag.fits"'
        RETURN
    ENDIF

    read_sdo, hmi_file, index, mag
    PRINT, 'Date:  ', index.date_obs
    PRINT, 'Range: ', MIN(mag), ' to ', MAX(mag), ' G'

    ; Display magnetogram (blue=neg, red=pos)
    LOADCT, 33
    WINDOW, 0, XSIZE=512, YSIZE=512
    TV, BYTSCL(REBIN(mag, 512, 512), MIN=-500, MAX=500)

    ; Compute total unsigned flux for center region
    cx = index.naxis1/2 & cy = index.naxis2/2
    hw = 200  ; half-width in pixels
    roi = mag[cx-hw:cx+hw, cy-hw:cy+hw]

    cdelt_rad = index.cdelt1 * !DTOR / 3600.0
    pixel_area = (cdelt_rad * index.dsun_obs * 100.0)^2

    pos = WHERE(roi GT 10, np)
    neg = WHERE(roi LT -10, nn)
    flux_pos = (np GT 0) ? TOTAL(roi[pos]) * pixel_area : 0.0D
    flux_neg = (nn GT 0) ? TOTAL(ABS(roi[neg])) * pixel_area : 0.0D

    PRINT, 'Positive flux: ', flux_pos, ' Mx'
    PRINT, 'Negative flux: ', flux_neg, ' Mx'
    PRINT, 'Imbalance:     ', (flux_pos-flux_neg)/(flux_pos+flux_neg)
END

; Uncomment to run:
; hmi_analysis_demo, 'hmi_mag.fits'
END
