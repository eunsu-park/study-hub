; 12 FITS File Handling
; =====================
; Demonstrates creating, writing, and reading FITS files,
; header manipulation with SXPAR/SXADDPAR.

PRO example_12_fits

  ; Create test data
  nx = 256L & ny = 256L
  data = DIST(nx, ny) + RANDOMN(seed, nx, ny) * 10.0

  ; Create FITS header
  MKHDR, header, data
  SXADDPAR, header, 'OBJECT', 'Test Image', 'Target'
  SXADDPAR, header, 'TELESCOP', 'Simulated', 'Telescope'
  SXADDPAR, header, 'WAVELNTH', 171, 'Wavelength (Angstroms)'
  SXADDPAR, header, 'EXPTIME', 2.0, 'Exposure time (s)'
  SXADDPAR, header, 'DATE-OBS', '2024-07-15T12:00:00', 'Observation date'
  SXADDPAR, header, 'CRPIX1', nx/2.0 + 0.5, 'Reference pixel X'
  SXADDPAR, header, 'CRPIX2', ny/2.0 + 0.5, 'Reference pixel Y'
  SXADDPAR, header, 'CDELT1', 0.6, 'Plate scale (arcsec/pix)'
  SXADDPAR, header, 'HISTORY', 'Created by IDL example'

  ; Write FITS file
  filename = 'example_image.fits'
  WRITEFITS, filename, data, header
  PRINT, 'Wrote:', filename

  ; Read it back
  read_data = READFITS(filename, read_header)
  PRINT, 'Read back:', SIZE(read_data, /DIMENSIONS)

  ; Extract keywords
  PRINT, 'OBJECT:', STRTRIM(SXPAR(read_header, 'OBJECT'), 2)
  PRINT, 'WAVELNTH:', SXPAR(read_header, 'WAVELNTH')
  PRINT, 'EXPTIME:', SXPAR(read_header, 'EXPTIME')
  PRINT, 'DATE-OBS:', STRTRIM(SXPAR(read_header, 'DATE-OBS'), 2)
  PRINT, 'Data range:', MIN(read_data), MAX(read_data)

  ; Clean up
  FILE_DELETE, filename, /ALLOW_NONEXISTENT

  PRINT, 'Example 12 complete.'
END
