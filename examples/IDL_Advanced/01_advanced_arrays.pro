;+
; 01_advanced_arrays.pro — Lesson 01: Advanced Array Techniques
;
; Demonstrates:
;   - REFORM for reshaping arrays
;   - REBIN for rebinning (up/down sampling)
;   - CONGRID for arbitrary resizing
;   - TOTAL for dimensional reduction
;   - MEDIAN filtering and SMOOTH
;   - CONVOL with custom kernels
;
; Run: IDL> .run 01_advanced_arrays
;-

PRO advanced_arrays_demo

    ; --- REFORM ---
    arr = INDGEN(12)
    arr2d = REFORM(arr, 3, 4)
    PRINT, '--- REFORM ---'
    PRINT, 'Original: ', SIZE(arr, /DIMENSIONS)
    PRINT, 'Reshaped: ', SIZE(arr2d, /DIMENSIONS)

    ; Remove degenerate dimension
    img = FLTARR(64, 64, 1)
    img = REFORM(img)
    PRINT, 'After REFORM: ', SIZE(img, /DIMENSIONS)

    ; --- REBIN ---
    small = DIST(64)
    big = REBIN(small, 256, 256)
    shrunk = REBIN(big, 64, 64)
    PRINT, '--- REBIN ---'
    PRINT, 'Original:  ', SIZE(small, /DIMENSIONS)
    PRINT, 'Expanded:  ', SIZE(big, /DIMENSIONS)
    PRINT, 'Shrunk:    ', SIZE(shrunk, /DIMENSIONS)

    ; --- CONGRID ---
    resized = CONGRID(small, 100, 100, /INTERP)
    PRINT, '--- CONGRID ---'
    PRINT, 'Resized to arbitrary: ', SIZE(resized, /DIMENSIONS)

    ; --- TOTAL along dimension ---
    cube = RANDOMU(seed, 32, 32, 10)
    time_sum = TOTAL(cube, 3)
    PRINT, '--- TOTAL ---'
    PRINT, 'Cube: ', SIZE(cube, /DIMENSIONS)
    PRINT, 'Sum over dim 3: ', SIZE(time_sum, /DIMENSIONS)

    ; --- Smoothing and filtering ---
    noisy = DIST(128) + RANDOMN(seed, 128, 128) * 20.0
    smoothed = SMOOTH(noisy, 5, /EDGE_TRUNCATE)
    med_filt = MEDIAN(noisy, 5)

    ; --- CONVOL with Gaussian kernel ---
    sigma = 2.0
    ksize = 11
    x = FINDGEN(ksize) - ksize/2
    kernel = EXP(-x^2 / (2.0*sigma^2))
    kernel = kernel / TOTAL(kernel)
    kernel_2d = kernel # kernel
    gauss_smooth = CONVOL(noisy, kernel_2d, /EDGE_TRUNCATE)

    ; Display comparison
    WINDOW, 0, XSIZE=512, YSIZE=512
    !P.MULTI = [0, 2, 2]
    LOADCT, 0
    TV, BYTSCL(CONGRID(noisy, 256, 256)), 0
    TV, BYTSCL(CONGRID(smoothed, 256, 256)), 1
    TV, BYTSCL(CONGRID(med_filt, 256, 256)), 2
    TV, BYTSCL(CONGRID(gauss_smooth, 256, 256)), 3
    !P.MULTI = 0

    PRINT, 'Demo complete. Window shows: Original / Boxcar / Median / Gaussian'
END

advanced_arrays_demo
END
