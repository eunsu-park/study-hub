;+
; Exercise 13: IDL-Python Bridge
; Requires: IDL 8.5+ with Python bridge
;-

PRO exercise_13

    ; === Exercise 1: numpy operations ===
    ; Use numpy from IDL to:
    ; (a) Create a 100x100 array of random numbers
    ; (b) Compute mean, std, min, max
    ; (c) Compare with IDL's MEAN, STDDEV, MIN, MAX
    ; TODO: PYTHON.IMPORT('numpy'), call functions, compare

    ; === Exercise 2: scipy curve fitting ===
    ; Use scipy.optimize.curve_fit from IDL to fit a Gaussian
    ; Generate data in IDL, pass to Python for fitting, get results back
    ; TODO: Create data in IDL, import scipy, call curve_fit

    ; === Exercise 3: matplotlib plot ===
    ; Use matplotlib from IDL to create a plot that IDL cannot easily make
    ; (e.g., a violin plot, hexbin, or styled errorbar plot)
    ; Save to PNG
    ; TODO: import matplotlib.pyplot, create plot, savefig

    ; === Exercise 4: Data type round-trip ===
    ; Create each IDL data type (BYTE, INT, LONG, FLOAT, DOUBLE, STRING)
    ; Pass each to Python, check the numpy dtype, pass back to IDL
    ; Verify the values match
    ; TODO: Test each type, print numpy dtype

    ; === Exercise 5: SunPy integration ===
    ; (If SunPy is installed) Use SunPy from IDL to:
    ; (a) Create a SunPy Map from an AIA FITS file
    ; (b) Print the map metadata (date, wavelength, scale)
    ; (c) Get the data array back into IDL
    ; TODO: PYTHON.IMPORT('sunpy.map'), create Map, extract data

END
