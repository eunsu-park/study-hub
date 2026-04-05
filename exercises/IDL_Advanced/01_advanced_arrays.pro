;+
; Exercise 01: Advanced Array Techniques
;
; Complete the TODO sections to practice array manipulation.
;-

PRO exercise_01

    ; === Exercise 1: REFORM ===
    ; Create a 1D array of 24 elements and reshape it to 4x6, then to 2x3x4
    arr = FINDGEN(24)
    ; TODO: Reshape arr to 4x6 and print dimensions
    ; TODO: Reshape arr to 2x3x4 and print dimensions
    ; TODO: Extract the element at [1, 2, 3] and print it

    ; === Exercise 2: REBIN ===
    ; Downsample a 256x256 image to 64x64 using averaging,
    ; then upsample back to 256x256. Compare with original.
    img = DIST(256)
    ; TODO: Downsample to 64x64
    ; TODO: Upsample back to 256x256
    ; TODO: Compute the RMS difference between original and round-tripped image
    ; Hint: RMS = SQRT(MEAN((original - roundtrip)^2))

    ; === Exercise 3: TOTAL with dimensions ===
    ; Given a 3D cube [50, 50, 20], compute:
    ; (a) the total over all elements
    ; (b) the mean image (average over time, dimension 3)
    ; (c) the light curve (sum over spatial dimensions 1 and 2)
    cube = RANDOMU(seed, 50, 50, 20) * 100.0
    ; TODO: Compute (a), (b), (c) and print their sizes

    ; === Exercise 4: Smoothing comparison ===
    ; Apply SMOOTH (width=7), MEDIAN (width=7), and Gaussian (sigma=2)
    ; to a noisy image. Compute the RMS error vs the clean image.
    clean = DIST(128)
    noisy = clean + RANDOMN(seed, 128, 128) * 20.0
    ; TODO: Apply three filters
    ; TODO: Compute and print RMS error for each
    ; Hint: For Gaussian, create a 1D kernel, form 2D via outer product, use CONVOL

    ; === Exercise 5: WHERE + ARRAY_INDICES ===
    ; Find all pixels in a 100x100x50 cube that are above the 99th percentile
    ; Report how many there are and the coordinates of the maximum
    cube2 = RANDOMU(seed, 100, 100, 50)
    ; TODO: Find 99th percentile threshold
    ; Hint: sorted = cube2[SORT(cube2)], threshold = sorted[0.99 * N_ELEMENTS(sorted)]
    ; TODO: Find pixels above threshold using WHERE
    ; TODO: Convert to subscripts using ARRAY_INDICES
    ; TODO: Find and print coordinates of the maximum value

END
