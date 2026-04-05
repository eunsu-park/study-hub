# 02. Advanced Plotting

**Previous**: [Advanced Array Techniques](./01_Advanced_Array_Techniques.md) | **Next**: [Map Projections](./03_Map_Projections.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Create multi-panel plots using `!P.MULTI`
2. Generate contour plots with custom levels and fill colors
3. Render 3D surface and shaded surface plots
4. Overlay annotations with PLOTS, XYOUTS, and drawing primitives
5. Manage color tables for scientific visualization
6. Produce publication-quality PostScript output

---

## 1. Multi-Panel Plots with !P.MULTI

The `!P.MULTI` system variable controls multi-panel layout.

```idl
; !P.MULTI = [remaining, ncols, nrows, nz, order]
; remaining = 0 means start fresh
; order = 0 (row-major, default), 1 (column-major)

; 2 columns x 2 rows layout
!P.MULTI = [0, 2, 2]

; Plot 1 (top-left)
PLOT, FINDGEN(100), SIN(FINDGEN(100)*0.1), TITLE='Sine'

; Plot 2 (top-right)
PLOT, FINDGEN(100), COS(FINDGEN(100)*0.1), TITLE='Cosine'

; Plot 3 (bottom-left)
PLOT, FINDGEN(100), EXP(-FINDGEN(100)*0.05), TITLE='Exponential Decay'

; Plot 4 (bottom-right)
PLOT, FINDGEN(100), ALOG(FINDGEN(100)+1), TITLE='Logarithm'

!P.MULTI = 0  ; Reset to single-panel
```

### Custom Panel Positioning with POSITION

For finer control, use the `POSITION` keyword instead of `!P.MULTI`:

```idl
; POSITION = [x0, y0, x1, y1] in normalized coordinates (0-1)
WINDOW, 0, XSIZE=800, YSIZE=600

; Top panel (full width, top half)
PLOT, x, y1, POSITION=[0.1, 0.55, 0.95, 0.95], $
    TITLE='Light Curve', /NOERASE

; Bottom-left panel
PLOT, x, y2, POSITION=[0.1, 0.08, 0.5, 0.45], $
    TITLE='Spectrum', /NOERASE

; Bottom-right panel
PLOT, x, y3, POSITION=[0.55, 0.08, 0.95, 0.45], $
    TITLE='Phase', /NOERASE
```

---

## 2. Contour Plots

### Basic Contour

```idl
; Create sample 2D data
data = DIST(100)

; Basic contour plot
CONTOUR, data, TITLE='Contour Plot'

; With specific contour levels
CONTOUR, data, LEVELS=[10, 20, 30, 40, 50, 60, 70]

; With NLEVELS
CONTOUR, data, NLEVELS=15, TITLE='15 Levels'
```

### Filled Contours

```idl
; Filled contour plot
levels = FINDGEN(10) * 8
LOADCT, 33  ; Blue-Red color table
CONTOUR, data, LEVELS=levels, /FILL, $
    C_COLORS=BYTSCL(INDGEN(10), TOP=254), $
    TITLE='Filled Contour'

; Overlay contour lines on filled contour
CONTOUR, data, LEVELS=levels, /OVERPLOT, $
    C_LABELS=REPLICATE(1, N_ELEMENTS(levels))  ; Label every level
```

### Contour with Axes

```idl
; Generate coordinate arrays
nx = 100 & ny = 100
x = FINDGEN(nx) * 0.1         ; 0.0 to 9.9
y = FINDGEN(ny) * 0.1         ; 0.0 to 9.9
z = SIN(x # REPLICATE(1, ny)) * COS(REPLICATE(1, nx) # y)

CONTOUR, z, x, y, $
    NLEVELS=20, /FILL, $
    XTITLE='X (arcsec)', YTITLE='Y (arcsec)', $
    TITLE='2D Wave Pattern'

; Overlay specific level
CONTOUR, z, x, y, LEVELS=[0.0], /OVERPLOT, $
    C_THICK=2, C_LINESTYLE=2, C_ANNOTATION='Zero'
```

### Contour Keywords Reference

| Keyword | Description |
|---------|-------------|
| `LEVELS` | Array of contour level values |
| `NLEVELS` | Number of equally-spaced levels |
| `/FILL` | Fill between contour levels |
| `C_COLORS` | Color index for each level |
| `C_THICK` | Line thickness for each level |
| `C_LINESTYLE` | Line style (0=solid, 1=dotted, 2=dashed, ...) |
| `C_LABELS` | 0/1 array: which levels to label |
| `C_ANNOTATION` | String labels for levels |
| `/OVERPLOT` | Draw on existing plot |
| `C_CHARSIZE` | Character size for labels |

---

## 3. Surface Plots

### SURFACE — Wireframe

```idl
data = DIST(50)

; Basic wireframe surface
SURFACE, data, TITLE='Wireframe Surface'

; With viewing angle
SURFACE, data, AX=45, AZ=30, $
    XTITLE='X', YTITLE='Y', ZTITLE='Z', $
    TITLE='Rotated Surface'

; Skirt and bottom
SURFACE, data, /SKIRT, ZVALUE=0, $
    TITLE='Surface with Skirt'
```

### SHADE_SURF — Shaded Surface

```idl
; Shaded surface (Gouraud shading)
SHADE_SURF, data, AX=45, AZ=30, $
    TITLE='Shaded Surface'

; With custom shading
; SHADES keyword accepts a byte array for per-pixel coloring
LOADCT, 33
SHADE_SURF, data, SHADES=BYTSCL(data), $
    AX=50, AZ=45, TITLE='Color-Shaded Surface'
```

### Combining Surface and Contour

```idl
; Surface with contour projection at the bottom
SURFACE, data, AX=45, AZ=30, TITLE='Surface + Contour'
CONTOUR, data, /T3D, ZVALUE=0, /NOERASE, NLEVELS=10
```

---

## 4. Overplotting and Annotations

### PLOTS — Drawing on the Plot Window

```idl
; Draw lines and symbols on an existing plot
x = FINDGEN(100) * 0.1
y = SIN(x)
PLOT, x, y, TITLE='Annotated Plot'

; Draw a horizontal line at y=0
PLOTS, [0, 10], [0, 0], LINESTYLE=2, COLOR=200

; Draw a filled circle at the peak
peak_x = !PI / 2
PLOTS, peak_x, 1.0, PSYM=8, SYMSIZE=2, COLOR=250

; Define custom symbol (filled circle)
A = FINDGEN(17) * (!PI * 2 / 16.0)
USERSYM, COS(A), SIN(A), /FILL
PLOTS, peak_x, 1.0, PSYM=8, SYMSIZE=1.5, COLOR=250

; Draw a box around a region
PLOTS, [2, 4, 4, 2, 2], [0.5, 0.5, 1.0, 1.0, 0.5], $
    LINESTYLE=0, THICK=2, COLOR=150
```

### XYOUTS — Text Annotations

```idl
; Add text at data coordinates
XYOUTS, peak_x, 1.05, 'Peak', ALIGNMENT=0.5, CHARSIZE=1.2

; Add text at normalized coordinates
XYOUTS, 0.5, 0.02, 'Generated with IDL', /NORMAL, $
    ALIGNMENT=0.5, CHARSIZE=0.8

; Rotated text
XYOUTS, 0.02, 0.5, 'Intensity', /NORMAL, $
    ALIGNMENT=0.5, ORIENTATION=90, CHARSIZE=1.0
```

### ARROW

```idl
; Draw an arrow from (x0,y0) to (x1,y1) in data coordinates
ARROW, 3.0, 0.8, peak_x, 1.0, /DATA, THICK=2, HSIZE=10
```

---

## 5. Color Table Management

### Loading Color Tables

```idl
; Built-in color tables (0-74)
LOADCT, 0     ; B-W Linear (grayscale)
LOADCT, 1     ; Blue/White
LOADCT, 3     ; Red Temperature
LOADCT, 13    ; Rainbow
LOADCT, 33    ; Blue-Red
LOADCT, 39    ; Rainbow+White

; List available color tables
LOADCT, /GET_NAMES, NAMES=ct_names
FOR i = 0, N_ELEMENTS(ct_names)-1 DO PRINT, i, ': ', ct_names[i]
```

### Custom Color Tables

```idl
; Create a custom color table
r = BYTARR(256) & g = BYTARR(256) & b = BYTARR(256)

; Blue-White-Red diverging colormap
; Blue (0-127) -> White (128) -> Red (129-255)
r[0:127] = BINDGEN(128) * 2
g[0:127] = BINDGEN(128) * 2
b[0:127] = 255
r[128:255] = 255
g[128:255] = REVERSE(BINDGEN(128) * 2)
b[128:255] = REVERSE(BINDGEN(128) * 2)

TVLCT, r, g, b  ; Load into color table
```

### Color Bar

```idl
; Simple color bar using TV
LOADCT, 33
bar = BINDGEN(256) # REPLICATE(1B, 20)  ; 256 x 20 bar
TV, bar, 0.15, 0.02, XSIZE=0.7, YSIZE=0.03, /NORMAL

; Or use the COLORBAR procedure (from Coyote library or SSW)
; COLORBAR, RANGE=[vmin, vmax], TITLE='Intensity', /VERTICAL
```

---

## 6. Compound Visualization Plots

### Image with Contour Overlay

```idl
; Display image with contour overlay
img = DIST(256)
LOADCT, 3
WINDOW, 0, XSIZE=600, YSIZE=600
TV, BYTSCL(img)
CONTOUR, img, /NOERASE, NLEVELS=10, $
    POSITION=[0.0, 0.0, 1.0, 1.0], $
    XSTYLE=1, YSTYLE=1, COLOR=255
```

### Time-Distance Plot

```idl
; Solar slit analysis: intensity along a slit vs time
; data: [distance, time] or [slit_pixels, n_frames]
nslit = 200 & ntime = 300
td_map = FLTARR(nslit, ntime)
FOR t = 0, ntime-1 DO $
    td_map[*, t] = 100 + 50 * SIN(2*!PI*(FINDGEN(nslit)/50.0 - t/30.0))

LOADCT, 3
WINDOW, 0, XSIZE=800, YSIZE=400
TV, BYTSCL(td_map, MIN=30, MAX=170), 50, 50, $
    XSIZE=700, YSIZE=300
PLOT, FINDGEN(nslit), FINDGEN(ntime), /NODATA, /NOERASE, $
    POSITION=[50./800, 50./400, 750./800, 350./400], $
    XTITLE='Distance (pixels)', YTITLE='Time (frames)', $
    TITLE='Time-Distance Diagram', XSTYLE=1, YSTYLE=1, $
    XRANGE=[0, nslit-1], YRANGE=[0, ntime-1]
```

---

## 7. Publication-Quality PostScript Output

### Basic PostScript

```idl
; Open PostScript device
SET_PLOT, 'PS'
DEVICE, FILENAME='figure1.ps', /ENCAPSULATED, $
    XSIZE=18, YSIZE=12, /COLOR, BITS_PER_PIXEL=8

; Set thick lines and large text for publication
!P.THICK = 3
!X.THICK = 2
!Y.THICK = 2
!P.CHARSIZE = 1.2
!P.CHARTHICK = 2

; Create the plot
LOADCT, 0
x = FINDGEN(200) * 0.05
PLOT, x, SIN(x), XTITLE='Time (s)', YTITLE='Amplitude', $
    TITLE='Publication Plot', XSTYLE=1

; Close PostScript
DEVICE, /CLOSE
SET_PLOT, 'X'  ; Return to screen display

; Reset plot parameters
!P.THICK = 0
!X.THICK = 0
!Y.THICK = 0
!P.CHARSIZE = 0
!P.CHARTHICK = 0
```

### Multi-Page PostScript

```idl
; Multi-page PS file
SET_PLOT, 'PS'
DEVICE, FILENAME='multi_page.ps', /COLOR, BITS_PER_PIXEL=8, $
    XSIZE=20, YSIZE=25

; Page 1
!P.MULTI = [0, 2, 3]
FOR i = 0, 5 DO $
    PLOT, FINDGEN(100), RANDOMN(seed, 100), TITLE='Panel '+STRTRIM(i+1,2)

; New page
DEVICE, /ADVANCE  ; or EJECT

; Page 2
!P.MULTI = [0, 1, 2]
CONTOUR, DIST(100), NLEVELS=10, /FILL, TITLE='Contour'
SURFACE, DIST(50), AX=45, AZ=30, TITLE='Surface'

DEVICE, /CLOSE
SET_PLOT, 'X'
!P.MULTI = 0
```

### PostScript to PDF

```bash
# Convert PostScript to PDF using Ghostscript
ps2pdf figure1.ps figure1.pdf

# Or with specific settings
gs -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sOutputFile=figure1.pdf figure1.ps
```

### Tips for Publication Figures

```idl
; Standard two-column figure width: 8.5 cm (single) or 18 cm (double)
; Standard fonts for journals: Helvetica, Times

; Use DEVICE keywords for font control in PS
DEVICE, SET_FONT='Helvetica', /TT_FONT
; or
DEVICE, /HELVETICA  ; PostScript built-in font

; Line style reference:
; 0 = Solid  ___________
; 1 = Dotted ...........
; 2 = Dashed -----------
; 3 = Dash-dot -.-.-.-.-
; 4 = Dash-dot-dot -..-..
; 5 = Long dash -- -- --

; Symbol reference (PSYM):
; 1 = Plus (+)
; 2 = Asterisk (*)
; 3 = Period (.)
; 4 = Diamond
; 5 = Triangle
; 6 = Square
; 7 = X
; 8 = User-defined (USERSYM)
; Negative PSYM: connect symbols with lines
```

---

## 8. Advanced Plot Customization

### System Variables for Global Control

```idl
; !P — Plot parameters
!P.BACKGROUND = 255    ; White background (for PostScript)
!P.COLOR = 0           ; Black foreground
!P.FONT = -1           ; Hershey vector fonts (device-independent)
!P.FONT = 0            ; Device font (PostScript: scalable)
!P.FONT = 1            ; TrueType font

; !X, !Y, !Z — Axis parameters
!X.STYLE = 1           ; Exact axis range (no padding)
!Y.STYLE = 1
!X.MARGIN = [10, 3]    ; Left/right margin in character widths
!Y.MARGIN = [4, 2]     ; Bottom/top margin

; Save and restore system variables
saved_p = !P
saved_x = !X
saved_y = !Y
; ... do custom plotting ...
!P = saved_p
!X = saved_x
!Y = saved_y
```

### Logarithmic Axes

```idl
; Log-log plot
x = FINDGEN(100) + 1
y = x^2.5
PLOT, x, y, /XLOG, /YLOG, $
    XTITLE='Frequency (Hz)', YTITLE='Power', $
    TITLE='Power Spectrum (Log-Log)'

; Semi-log plot (y-axis only)
PLOT, x, y, /YLOG, XTITLE='Channel', YTITLE='Counts'
```

### Error Bars with ERRPLOT and OPLOTERR

```idl
; Manual error bars
x = FINDGEN(20) * 0.5
y = SIN(x) + RANDOMN(seed, 20) * 0.1
yerr = REPLICATE(0.1, 20)

PLOT, x, y, PSYM=4, TITLE='Data with Error Bars'

; Draw error bars manually
FOR i = 0, N_ELEMENTS(x)-1 DO $
    PLOTS, [x[i], x[i]], [y[i]-yerr[i], y[i]+yerr[i]]

; Using ERRPLOT (overplots error bars)
ERRPLOT, x, y-yerr, y+yerr
```

---

## Summary

| Technique | Key Functions/Keywords | Purpose |
|-----------|----------------------|---------|
| Multi-panel | `!P.MULTI`, `POSITION` | Layout multiple plots |
| Contour | `CONTOUR`, `/FILL`, `C_COLORS` | 2D field visualization |
| Surface | `SURFACE`, `SHADE_SURF` | 3D visualization |
| Annotations | `PLOTS`, `XYOUTS`, `ARROW` | Labels and markers |
| Color tables | `LOADCT`, `TVLCT` | Color management |
| PostScript | `SET_PLOT, 'PS'`, `DEVICE` | Publication output |
| Error bars | `ERRPLOT`, manual `PLOTS` | Uncertainty display |

---

**Previous**: [Advanced Array Techniques](./01_Advanced_Array_Techniques.md) | **Next**: [Map Projections](./03_Map_Projections.md)
