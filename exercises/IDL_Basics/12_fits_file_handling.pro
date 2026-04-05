; Exercise 12: FITS File Handling
;
; Practice reading and writing FITS files with headers.

; Exercise 1: Create a FITS file with a simulated solar image
; and proper WCS (World Coordinate System) keywords.
PRO exercise_12a
  ; TODO: Create a 512x512 float image (e.g., DIST + noise)
  ; TODO: Create header with MKHDR
  ; TODO: Add keywords: OBJECT, TELESCOP, WAVELNTH, DATE-OBS, EXPTIME
  ; TODO: Add WCS: CRPIX1/2, CRVAL1/2, CDELT1/2, CTYPE1/2
  ; TODO: Write with WRITEFITS
  ; TODO: Read back and verify keywords with SXPAR
  ; TODO: Clean up the file
END

; Exercise 2: Write a function that reads a FITS file and returns
; a structure containing the data and key metadata.
FUNCTION exercise_12b, filename
  ; TODO: Read with READFITS
  ; TODO: Extract NAXIS1, NAXIS2, BITPIX, DATE-OBS, EXPTIME
  ; TODO: Return {data: data, nx: ..., ny: ..., date: ..., exptime: ...}
  ; TODO: Handle missing keywords gracefully
  RETURN, !NULL
END

; Exercise 3: Write a multi-extension FITS file containing:
; Extension 0: a 256x256 image
; Extension 1: a 1D flux array
; Extension 2: a binary table with columns (time, flux, error)
PRO exercise_12c
  ; TODO: Create primary image data
  ; TODO: Write with MWRFITS, /CREATE
  ; TODO: Create 1D array and MWRFITS append
  ; TODO: Create structure array and MWRFITS append
  ; TODO: Read back each extension with MRDFITS
  ; TODO: Clean up the file
END

; Exercise 4: Write a procedure that prints a formatted summary
; of any FITS file: dimensions, type, and selected keywords.
PRO exercise_12d, filename
  ; TODO: Read header with HEADFITS
  ; TODO: Extract and print: NAXIS, NAXIS1, NAXIS2, BITPIX
  ; TODO: Print DATE-OBS, TELESCOP, OBJECT if present
  ; TODO: Use SXPAR with COUNT to check keyword existence
END
