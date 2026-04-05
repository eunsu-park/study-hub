; 11 Image Display
; ================
; Demonstrates TV, TVSCL, BYTSCL, LOADCT,
; image resizing, and color tables.

PRO example_11_image_display

  ; Create a test image
  image = DIST(256)
  PRINT, 'Image dimensions:', SIZE(image, /DIMENSIONS)
  PRINT, 'Image range:', MIN(image), MAX(image)

  ; Display with auto-scaling
  WINDOW, 0, XSIZE=256, YSIZE=256, TITLE='TVSCL'
  TVSCL, image

  ; BYTSCL with custom range
  mean_val = MEAN(image)
  sigma = STDDEV(image)
  scaled = BYTSCL(image, MIN=mean_val - 2*sigma, MAX=mean_val + 2*sigma)
  WINDOW, 1, XSIZE=256, YSIZE=256, TITLE='BYTSCL Custom Range'
  TV, scaled

  ; Color tables
  PRINT, '--- Color Tables ---'
  LOADCT, 39, /SILENT    ; Rainbow+White
  WINDOW, 2, XSIZE=256, YSIZE=256, TITLE='Rainbow'
  TV, scaled

  ; Resize with CONGRID
  small = DIST(64)
  big = CONGRID(BYTSCL(small), 256, 256, /INTERP)
  WINDOW, 3, XSIZE=256, YSIZE=256, TITLE='CONGRID Interpolated'
  TV, big

  ; Create an RGB image
  DEVICE, DECOMPOSED=1
  nx = 256 & ny = 256
  rgb = BYTARR(3, nx, ny)
  rgb[0, *, *] = BYTSCL(DIST(nx, ny))
  rgb[1, *, *] = BYTSCL(SHIFT(DIST(nx, ny), 80, 0))
  rgb[2, *, *] = BYTSCL(SHIFT(DIST(nx, ny), 0, 80))
  WINDOW, 4, XSIZE=nx, YSIZE=ny, TITLE='RGB Image'
  TV, rgb, TRUE=1

  PRINT, 'Example 11 complete.'
END
