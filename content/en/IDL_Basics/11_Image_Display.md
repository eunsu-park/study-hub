# Image Display

**Previous**: [Basic Plotting](./10_Basic_Plotting.md) | **Next**: [FITS File Handling](./12_FITS_File_Handling.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Display images with TV and TVSCL
2. Create and manage graphics windows for images
3. Load and use color tables with LOADCT
4. Scale image data with BYTSCL
5. Resize images with CONGRID and REBIN
6. Work with the RGB color model and decomposed color
7. Understand device-independent graphics with !D system variable

---

Image display is fundamental to scientific visualization, whether you are viewing solar disk images, spectrograms, or simulation output. IDL provides powerful tools for rendering 2D data as images with full control over color mapping and scaling.

## TV and TVSCL

### TV — Display an Image

`TV` displays a byte array as an image. The input must be a BYTE array (0-255):

```idl
; Create a sample image
image = BYTSCL(DIST(256))    ; 256x256 gradient image

; Display it
WINDOW, 0, XSIZE=256, YSIZE=256
TV, image
```

### TVSCL — Display with Automatic Scaling

`TVSCL` automatically scales the data to the byte range (0-255):

```idl
; Display floating-point data directly
data = DIST(200)    ; 200x200 float array
WINDOW, 0, XSIZE=200, YSIZE=200
TVSCL, data

; TVSCL is equivalent to:
; TV, BYTSCL(data)
```

### Position Control

```idl
; TV with position (pixel coordinates)
image = BYTSCL(DIST(100))

WINDOW, 0, XSIZE=500, YSIZE=400
TV, image, 50, 100           ; Place at pixel (50, 100)
TV, image, 200, 100          ; Place another copy

; TV with channel number (position index)
WINDOW, 0, XSIZE=600, YSIZE=200
TV, image, 0                  ; First position
TV, image, 1                  ; Second position (right of first)
TV, image, 2                  ; Third position
```

---

## BYTSCL — Byte Scaling

`BYTSCL` linearly scales data to the range [0, 255] (or [MIN, MAX] with TOP keyword):

```idl
; Basic scaling
data = FINDGEN(10) * 100.0    ; [0, 100, 200, ..., 900]
scaled = BYTSCL(data)
PRINT, scaled                  ; [0, 28, 57, 85, 113, 142, 170, 198, 227, 255]

; Scale with custom range
data = RANDOMN(seed, 256, 256) * 100.0
scaled = BYTSCL(data, MIN=-200, MAX=200)    ; Clip to [-200, 200]

; Scale with TOP keyword (max output value)
scaled = BYTSCL(data, TOP=200)    ; Output range [0, 200]

; Common pattern: display with clipping
image = RANDOMN(seed, 512, 512) + 5.0
mean_val = MEAN(image)
sigma = STDDEV(image)
display = BYTSCL(image, MIN=mean_val - 3*sigma, MAX=mean_val + 3*sigma)
WINDOW, XSIZE=512, YSIZE=512
TV, display
```

---

## Color Tables with LOADCT

IDL provides 41 built-in color tables:

```idl
; Load a color table
LOADCT, 0      ; B-W Linear (grayscale)
LOADCT, 1      ; Blue-White
LOADCT, 3      ; Red Temperature
LOADCT, 5      ; Standard Gamma-II
LOADCT, 13     ; Rainbow
LOADCT, 33     ; Blue-Red
LOADCT, 39     ; Rainbow + White

; Display available color tables
LOADCT              ; Interactive: shows all tables and lets you choose

; Load with /SILENT to suppress messages
LOADCT, 39, /SILENT
```

### Custom Color Tables

```idl
; Modify individual colors
; RGB vectors (256 entries each)
TVLCT, r, g, b, /GET          ; Get current color table

; Set a specific color entry
r[255] = 255 & g[255] = 0 & b[255] = 0    ; Entry 255 = red
TVLCT, r, g, b

; Create a simple blue-to-red color table
n = 256
r = BINDGEN(n)                  ; 0 to 255
g = BYTARR(n)                   ; All zeros
b = REVERSE(BINDGEN(n))         ; 255 to 0
TVLCT, r, g, b

; Load a colorbar to visualize the table
bar = BINDGEN(256) # REPLICATE(1B, 20)    ; 256x20 image
WINDOW, XSIZE=256, YSIZE=20
TV, bar
```

### LOADCT with Specific Range

```idl
; Load color table only for a range of indices
LOADCT, 39, NCOLORS=200, BOTTOM=10
; Fills indices 10-209 with Rainbow+White table
; Leaves 0-9 and 210-255 unchanged
```

---

## Window Management for Images

```idl
; Create a window sized to the image
nx = 512
ny = 512
image = BYTSCL(DIST(nx, ny))

WINDOW, 0, XSIZE=nx, YSIZE=ny, TITLE='Image Display'
TV, image

; Resize image to fit a larger window
WINDOW, 1, XSIZE=800, YSIZE=600, TITLE='Resized'
TV, CONGRID(image, 800, 600)

; Check current window properties
PRINT, 'Device:', !D.NAME
PRINT, 'Window:', !D.WINDOW
PRINT, 'Size:', !D.X_SIZE, 'x', !D.Y_SIZE
```

---

## Image Resizing

### CONGRID — Arbitrary Resize

```idl
; Resize to any dimensions (interpolation)
small = DIST(64)
big = CONGRID(small, 512, 512)           ; Nearest neighbor (default)
big_interp = CONGRID(small, 512, 512, /INTERP)  ; Bilinear interpolation

WINDOW, 0, XSIZE=1024, YSIZE=512
TV, BYTSCL(big), 0, 0
TV, BYTSCL(big_interp), 512, 0
```

### REBIN — Integer Factor Resize

```idl
; Expand by integer multiples
small = DIST(64)
big = REBIN(small, 256, 256)       ; 4x expansion (averaging)
bigger = REBIN(small, 512, 512)    ; 8x expansion

; Shrink by integer factors
big = DIST(512)
small = REBIN(big, 128, 128)      ; 4x reduction (averaging)
; Note: 512/128 = 4, must be exact integer factor
```

---

## RGB Color Model

### Decomposed vs. Indexed Color

```idl
; Check current color mode
DEVICE, GET_DECOMPOSED=mode
PRINT, 'Decomposed:', mode   ; 1 = true color (24-bit), 0 = indexed (8-bit)

; Indexed color mode (classic)
DEVICE, DECOMPOSED=0
LOADCT, 39
image = BYTSCL(DIST(256))
TV, image    ; Colors come from the current color table

; Decomposed (true color) mode
DEVICE, DECOMPOSED=1
; Each pixel is a 24-bit value: B*65536L + G*256L + R
```

### True Color Images

```idl
; Create an RGB image (3 x nx x ny) or (nx x ny x 3)
nx = 256
ny = 256

; Create color channels
red_channel = BYTSCL(DIST(nx, ny))
green_channel = BYTSCL(SHIFT(DIST(nx, ny), 100, 0))
blue_channel = BYTSCL(SHIFT(DIST(nx, ny), 0, 100))

; Compose RGB image (pixel-interleaved: [3, nx, ny])
rgb = BYTARR(3, nx, ny)
rgb[0, *, *] = red_channel
rgb[1, *, *] = green_channel
rgb[2, *, *] = blue_channel

; Display true color image
DEVICE, DECOMPOSED=1
WINDOW, 0, XSIZE=nx, YSIZE=ny
TV, rgb, TRUE=1    ; TRUE=1 means first dimension is color channel
; TRUE=2 for [nx, 3, ny], TRUE=3 for [nx, ny, 3]
```

---

## Image with Plot Overlay

Combining images with plot annotations:

```idl
; Display image with axes and colorbar
image = DIST(256)
scaled = BYTSCL(image)

DEVICE, DECOMPOSED=0
LOADCT, 39, /SILENT

; Use PLOT to establish coordinate system, then TV
WINDOW, 0, XSIZE=400, YSIZE=350

; Create coordinate system with invisible plot
PLOT, [0, 256], [0, 256], /NODATA, $
  XSTYLE=1, YSTYLE=1, $
  POSITION=[0.15, 0.15, 0.85, 0.9], $
  XTITLE='Pixel X', YTITLE='Pixel Y', $
  TITLE='Image with Axes'

; Get the plot region in device coordinates
px = !X.WINDOW * !D.X_SIZE
py = !Y.WINDOW * !D.Y_SIZE
sx = (px[1] - px[0])
sy = (py[1] - py[0])

; Resize and display image in the plot region
TV, CONGRID(scaled, FIX(sx), FIX(sy)), px[0], py[0]

; Redraw axes on top
PLOT, [0, 256], [0, 256], /NODATA, /NOERASE, $
  XSTYLE=1, YSTYLE=1, $
  POSITION=[0.15, 0.15, 0.85, 0.9], $
  XTITLE='Pixel X', YTITLE='Pixel Y', $
  TITLE='Image with Axes'
```

---

## Device-Independent Graphics

### The !D System Variable

```idl
; !D contains device information
PRINT, !D.NAME          ; Current device (X, WIN, PS, Z)
PRINT, !D.X_SIZE        ; Device width in pixels
PRINT, !D.Y_SIZE        ; Device height in pixels
PRINT, !D.X_VSIZE       ; Visible area width
PRINT, !D.Y_VSIZE       ; Visible area height
PRINT, !D.N_COLORS      ; Number of available colors
PRINT, !D.TABLE_SIZE    ; Color table size
PRINT, !D.WINDOW        ; Current window number (-1 if none)
```

### The Z-Buffer Device

The Z-buffer renders images in memory without a display, useful for batch processing:

```idl
; Switch to Z-buffer
original = !D.NAME
SET_PLOT, 'Z'
DEVICE, SET_RESOLUTION=[800, 600], Z_BUFFERING=0

; Create plot in memory
LOADCT, 0, /SILENT
x = FINDGEN(100) / 10.0
PLOT, x, SIN(x), TITLE='Z-Buffer Plot'

; Capture the rendered image
snapshot = TVRD()

; Switch back to screen
SET_PLOT, original

; Display or save the snapshot
TV, snapshot
; Or write to a file:
WRITE_PNG, 'plot.png', snapshot
```

---

## Saving Images to Files

```idl
; Save as PNG
image = BYTSCL(DIST(256))
WRITE_PNG, 'image.png', image

; Save with color table
LOADCT, 39, /SILENT
TVLCT, r, g, b, /GET
WRITE_PNG, 'image_color.png', image, r, g, b

; Save as JPEG
WRITE_JPEG, 'image.jpg', image, QUALITY=95

; Save as TIFF
WRITE_TIFF, 'image.tif', image

; Read images
png_data = READ_PNG('image.png')
HELP, png_data

jpeg_data = READ_JPEG('image.jpg')
```

---

## Practical Example: Solar Image Display

```idl
PRO display_solar_image, image, TITLE=title, WAVELENGTH=wave
  IF ~KEYWORD_SET(title) THEN title = 'Solar Image'
  IF ~KEYWORD_SET(wave) THEN wave = 0

  ; Get image dimensions
  sz = SIZE(image, /DIMENSIONS)
  nx = sz[0]
  ny = sz[1]

  ; Scale based on wavelength
  CASE wave OF
    171: BEGIN
      LOADCT, 1, /SILENT    ; Blue-white for 171A
      scaled = BYTSCL(ALOG10(image > 1), MIN=0, MAX=4)
    END
    193: BEGIN
      LOADCT, 3, /SILENT    ; Red temperature for 193A
      scaled = BYTSCL(ALOG10(image > 1), MIN=0, MAX=4)
    END
    304: BEGIN
      LOADCT, 8, /SILENT    ; Red for 304A
      scaled = BYTSCL(ALOG10(image > 1), MIN=0, MAX=3.5)
    END
    ELSE: BEGIN
      LOADCT, 0, /SILENT    ; Grayscale default
      mean_val = MEAN(image)
      sigma = STDDEV(image)
      scaled = BYTSCL(image, MIN=mean_val - 3*sigma, MAX=mean_val + 3*sigma)
    END
  ENDCASE

  ; Display
  win_size = 512
  WINDOW, /FREE, XSIZE=win_size, YSIZE=win_size + 30, TITLE=title
  TV, CONGRID(scaled, win_size, win_size), 0, 30

  ; Add title
  XYOUTS, 0.5, 0.97, title, /NORMAL, ALIGNMENT=0.5, $
    CHARSIZE=1.3, CHARTHICK=1.5
END
```

---

## Summary

| Procedure/Function | Description |
|-------------------|-------------|
| `TV, image` | Display byte image |
| `TVSCL, data` | Display with auto-scaling |
| `BYTSCL(data)` | Scale data to 0-255 byte range |
| `LOADCT, n` | Load color table number n |
| `TVLCT, r, g, b` | Set/get RGB color table |
| `CONGRID(img, nx, ny)` | Resize to arbitrary dimensions |
| `REBIN(img, nx, ny)` | Resize by integer factors |
| `WINDOW, n, XSIZE=, YSIZE=` | Create graphics window |
| `WSET, n` | Select active window |
| `TVRD()` | Read image from display |
| `WRITE_PNG` | Save as PNG file |
| `WRITE_JPEG` | Save as JPEG file |
| `DEVICE, DECOMPOSED=` | Set color mode |
| `SET_PLOT, 'Z'` | Switch to Z-buffer (off-screen) |

---

**Previous**: [Basic Plotting](./10_Basic_Plotting.md) | **Next**: [FITS File Handling](./12_FITS_File_Handling.md)
