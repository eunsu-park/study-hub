;+
; 06_sdo_aia_analysis.pro — Lesson 06: SDO/AIA Analysis
;
; Demonstrates AIA data reading, calibration, and composite creation.
; Requires: SolarSoft with SDO/AIA package
;-

PRO aia_analysis_demo, aia_file
    IF N_ELEMENTS(aia_file) EQ 0 THEN BEGIN
        PRINT, 'Usage: aia_analysis_demo, "aia_171.fits"'
        RETURN
    ENDIF

    ; Read
    read_sdo, aia_file, index, data
    PRINT, 'Channel:   ', index.wavelnth, ' A'
    PRINT, 'Date:      ', index.date_obs
    PRINT, 'Image size:', index.naxis1, ' x ', index.naxis2

    ; Calibrate
    aia_prep, index, data, oindex, odata, /NORMALIZE, /REGISTER
    PRINT, 'Calibrated to level: ', oindex.lvl_num

    ; Display with AIA color table
    aia_lct, WAVE=index.wavelnth, /LOAD
    WINDOW, 0, XSIZE=512, YSIZE=512
    TV, BYTSCL(ALOG10(REBIN(odata, 512, 512) > 1), MIN=0.5, MAX=3.5)

    ; Basic statistics
    IMAGE_STATISTICS, odata, MEAN=mn, STDDEV=sd, MINIMUM=mi, MAXIMUM=mx
    PRINT, 'Mean: ', mn, ' Std: ', sd, ' Min: ', mi, ' Max: ', mx
END

; Uncomment to run:
; aia_analysis_demo, 'aia_171.fits'
END
