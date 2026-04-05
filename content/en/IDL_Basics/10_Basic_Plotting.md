# Basic Plotting

**Previous**: [Structures](./09_Structures.md) | **Next**: [Image Display](./11_Image_Display.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Create line plots with the PLOT procedure
2. Overlay data with OPLOT
3. Add text annotations with XYOUTS
4. Customize axes with AXIS and plot keywords
5. Control line styles, colors, symbols, and thickness
6. Use the !P system variable for multi-panel plots
7. Create publication-quality PostScript output

---

IDL's built-in plotting system is one of its greatest strengths. With a few commands, you can produce publication-quality figures for scientific papers. The plotting system is device-independent — the same commands produce output on screen, in PostScript, or in other formats.

## The PLOT Procedure

### Basic Line Plot

```idl
; Simple plot
x = FINDGEN(100) / 10.0    ; 0.0 to 9.9
y = SIN(x)
PLOT, x, y

; Plot with only y values (x defaults to index)
data = RANDOMN(seed, 50)
PLOT, data
```

### Plot with Title and Labels

```idl
x = FINDGEN(360) * !DTOR    ; 0 to 2*pi in radians
y = SIN(x)

PLOT, x / !DTOR, y, $
  TITLE='Sine Function', $
  XTITLE='Angle (degrees)', $
  YTITLE='sin(x)', $
  CHARSIZE=1.5
```

### Axis Range Control

```idl
x = FINDGEN(100)
y = EXP(-x / 20.0) * SIN(x / 5.0)

PLOT, x, y, $
  XRANGE=[0, 80], $
  YRANGE=[-1, 1], $
  XSTYLE=1, $         ; Exact X range (no extension)
  YSTYLE=1             ; Exact Y range (no extension)

; XSTYLE/YSTYLE values:
; 0 = default (IDL extends range slightly)
; 1 = exact range
; 2 = extended range
; 4 = suppress axis
; 8 = suppress box axis (top/right)
; Values can be combined: XSTYLE=1+8 = exact range + no box
```

---

## Line Styles, Symbols, and Colors

### LINESTYLE

```idl
x = FINDGEN(100) / 10.0

; LINESTYLE values:
; 0 = Solid (default)
; 1 = Dotted
; 2 = Dashed
; 3 = Dash-dot
; 4 = Dash-dot-dot
; 5 = Long dashes

PLOT, x, SIN(x), LINESTYLE=0, TITLE='Line Styles'
OPLOT, x, COS(x), LINESTYLE=2
OPLOT, x, SIN(2*x)*0.5, LINESTYLE=3
```

### PSYM — Plot Symbols

```idl
x = FINDGEN(20)
y = SQRT(x)

; PSYM values:
; 0 = Line (default)
; 1 = Plus (+)
; 2 = Asterisk (*)
; 3 = Period (.)
; 4 = Diamond
; 5 = Triangle
; 6 = Square
; 7 = X
; 8 = User-defined (with USERSYM)
; 10 = Histogram mode

; Positive PSYM: symbols only
PLOT, x, y, PSYM=4, SYMSIZE=1.5

; Negative PSYM: symbols connected by lines
PLOT, x, y, PSYM=-4, SYMSIZE=1.5

; Histogram style
PLOT, x, y, PSYM=10
```

### THICK — Line Thickness

```idl
x = FINDGEN(100) / 10.0
PLOT, x, SIN(x), THICK=3, TITLE='Thick Lines'
OPLOT, x, COS(x), THICK=1, LINESTYLE=2
```

### COLOR

```idl
; In IDL's default 8-bit color model:
; Colors depend on the loaded color table
; 0 = background (usually black or white)
; 255 = foreground

; Load a color table
LOADCT, 0    ; Black and white (grayscale)

; In decomposed color mode (24-bit, default in modern IDL):
; Color is specified as a 24-bit RGB value
; or using COLOR = R + 256L*G + 256L^2*B

; Check current mode
PRINT, !D.NAME            ; Current device
DEVICE, GET_DECOMPOSED=decomp
PRINT, 'Decomposed:', decomp

; Example with decomposed color
DEVICE, DECOMPOSED=1
red   = '0000FF'XL      ; Red in BGR format
green = '00FF00'XL      ; Green
blue  = 'FF0000'XL      ; Blue
white = 'FFFFFF'XL

PLOT, x, SIN(x), COLOR=red, BACKGROUND=white
OPLOT, x, COS(x), COLOR=blue

; Or use cgColor from Coyote Graphics library (common in community)
; PLOT, x, SIN(x), COLOR=cgColor('red')
```

---

## OPLOT — Overplotting

```idl
; OPLOT draws on the existing plot without erasing
x = FINDGEN(100) / 10.0

; First plot sets up the axes
PLOT, x, SIN(x), $
  TITLE='Trigonometric Functions', $
  XTITLE='x (radians)', $
  YTITLE='f(x)', $
  YRANGE=[-1.5, 1.5], $
  YSTYLE=1

; Overlay additional curves
OPLOT, x, COS(x), LINESTYLE=2
OPLOT, x, SIN(x) + COS(x), LINESTYLE=3
OPLOT, x, SIN(x) * COS(x), LINESTYLE=4
```

### ERRPLOT — Error Bars

```idl
; Plot data with error bars
n = 20
x = FINDGEN(n)
y = 2.0 * x + RANDOMN(seed, n) * 3.0
err = REPLICATE(1.5, n) + RANDOMU(seed, n) * 2.0

PLOT, x, y, PSYM=4, SYMSIZE=1.5, $
  TITLE='Data with Error Bars', $
  XTITLE='X', YTITLE='Y'

; ERRPLOT draws vertical error bars
ERRPLOT, x, y - err, y + err
```

---

## XYOUTS — Text Annotations

```idl
x = FINDGEN(100) / 10.0
y = SIN(x)
PLOT, x, y, TITLE='Annotated Plot'

; Add text at data coordinates
XYOUTS, 1.57, 1.0, 'Maximum', CHARSIZE=1.2, ALIGNMENT=0.5
XYOUTS, 4.71, -1.0, 'Minimum', CHARSIZE=1.2, ALIGNMENT=0.5

; Add text at normalized coordinates (0-1 range)
XYOUTS, 0.15, 0.85, 'y = sin(x)', /NORMAL, CHARSIZE=1.5

; Text formatting
XYOUTS, 0.5, 0.05, 'Figure 1: Sine Wave', /NORMAL, $
  ALIGNMENT=0.5, CHARSIZE=1.3

; Greek letters and special characters (using !C for newline, !U for superscript)
XYOUTS, 0.5, 0.80, '!7a!3 = 0.05', /NORMAL, CHARSIZE=1.5
; !7 switches to Symbol font (alpha), !3 back to Helvetica

; Superscripts and subscripts
XYOUTS, 0.5, 0.70, 'x!U2!N + y!U2!N = r!U2!N', /NORMAL, CHARSIZE=1.5
; !U = start superscript, !N = return to normal
; !D = start subscript, !N = return to normal
```

---

## AXIS — Custom Axes

```idl
; Add or modify axes
x = FINDGEN(100) / 10.0
y = SIN(x)
PLOT, x, y, YSTYLE=8+1, $     ; Suppress right Y axis
  YTITLE='sin(x)'

; Add a right-side Y axis with different scaling
y2 = y * 180.0 / !PI   ; Convert to degrees
AXIS, YAXIS=1, YRANGE=[MIN(y2), MAX(y2)], $
  YTITLE='Angle (degrees)', YSTYLE=1

; X axis at Y=0
AXIS, XAXIS=0, XTICKFORMAT='(A1)'    ; Bottom axis, no labels
AXIS, YAXIS=0, YTICKFORMAT='(A1)'    ; Left axis, no labels
```

---

## Multi-Panel Plots with !P.MULTI

```idl
; Create a 2x2 grid of plots
!P.MULTI = [0, 2, 2]    ; [remaining, columns, rows]

x = FINDGEN(100) / 10.0

PLOT, x, SIN(x), TITLE='sin(x)'
PLOT, x, COS(x), TITLE='cos(x)'
PLOT, x, TAN(x) < 5 > (-5), TITLE='tan(x)', YRANGE=[-5,5], YSTYLE=1
PLOT, x, EXP(-x/5.0), TITLE='exp(-x/5)'

; Reset to single panel
!P.MULTI = 0

; More complex layout: 3 rows, 1 column
!P.MULTI = [0, 1, 3]
PLOT, x, SIN(x), TITLE='Panel 1'
PLOT, x, COS(x), TITLE='Panel 2'
PLOT, x, SIN(x)*COS(x), TITLE='Panel 3'
!P.MULTI = 0
```

---

## Logarithmic Plots

```idl
; Semi-log plot (Y-axis logarithmic)
x = FINDGEN(100) + 1.0
y = EXP(x / 20.0)
PLOT, x, y, /YLOG, TITLE='Semi-Log Plot', $
  XTITLE='X', YTITLE='Y (log scale)'

; Log-log plot
x = FINDGEN(100) + 1.0
y = x^2.5
PLOT, x, y, /XLOG, /YLOG, TITLE='Log-Log Plot'

; Multiple decades
freq = 10.0^(FINDGEN(50)/10.0)      ; 1 to 100000
power = 1.0 / freq^1.5
PLOT, freq, power, /XLOG, /YLOG, $
  TITLE='Power Spectrum', $
  XTITLE='Frequency (Hz)', $
  YTITLE='Power'
```

---

## Graphics Windows

```idl
; Create a new window
WINDOW, 0, XSIZE=800, YSIZE=600, TITLE='Main Plot'

; Create a second window
WINDOW, 1, XSIZE=400, YSIZE=400, TITLE='Detail View'

; Switch between windows
WSET, 0    ; Make window 0 active
PLOT, FINDGEN(100), SIN(FINDGEN(100)/10.0), TITLE='Window 0'

WSET, 1    ; Make window 1 active
PLOT, FINDGEN(50), COS(FINDGEN(50)/5.0), TITLE='Window 1'

; Delete a window
WDELETE, 1

; Get current window info
PRINT, !D.WINDOW        ; Current window number
PRINT, !D.X_SIZE, !D.Y_SIZE  ; Current window size
```

---

## PostScript Output

Creating PostScript files is essential for publication-quality figures:

```idl
; Save the current device
original_device = !D.NAME

; Switch to PostScript device
SET_PLOT, 'PS'
DEVICE, FILENAME='figure1.ps', $
  /PORTRAIT, $              ; or /LANDSCAPE
  XSIZE=18, YSIZE=12, $    ; Size in cm
  /COLOR, $                 ; Enable color
  BITS_PER_PIXEL=8, $       ; Color depth
  /ENCAPSULATED             ; Create EPS file

; Create the plot
x = FINDGEN(360) * !DTOR
PLOT, x / !DTOR, SIN(x), $
  TITLE='Sine Function', $
  XTITLE='Angle (degrees)', $
  YTITLE='sin(x)', $
  THICK=3, CHARTHICK=2, CHARSIZE=1.2, $
  XTHICK=2, YTHICK=2, $
  XSTYLE=1, YSTYLE=1

OPLOT, x / !DTOR, COS(x), LINESTYLE=2, THICK=3

; Add legend
XYOUTS, 280, 0.8, 'sin(x)', CHARSIZE=1.0, CHARTHICK=2
PLOTS, [250, 275], [0.82, 0.82], THICK=3
XYOUTS, 280, 0.6, 'cos(x)', CHARSIZE=1.0, CHARTHICK=2
PLOTS, [250, 275], [0.62, 0.62], THICK=3, LINESTYLE=2

; Close the file
DEVICE, /CLOSE

; Return to screen display
SET_PLOT, original_device

PRINT, 'PostScript file saved: figure1.ps'
```

### Converting PostScript to PDF

```bash
# Using ps2pdf (from Ghostscript)
ps2pdf figure1.ps figure1.pdf

# Using epstopdf (for EPS files)
epstopdf figure1.ps
```

---

## Adding a Legend

IDL does not have a built-in legend command, but creating one is straightforward:

```idl
PRO add_legend, labels, linestyles, colors, x0, y0, $
                CHARSIZE=charsize, THICK=thick
  IF ~KEYWORD_SET(charsize) THEN charsize = 1.0
  IF ~KEYWORD_SET(thick) THEN thick = 2

  n = N_ELEMENTS(labels)
  dy = 0.04  ; Vertical spacing in normal coordinates

  FOR i = 0, n - 1 DO BEGIN
    ; Draw line segment
    PLOTS, [x0, x0 + 0.06], [y0 - i*dy, y0 - i*dy], $
      /NORMAL, LINESTYLE=linestyles[i], $
      COLOR=colors[i], THICK=thick

    ; Draw label
    XYOUTS, x0 + 0.08, y0 - i*dy - 0.008, labels[i], $
      /NORMAL, CHARSIZE=charsize, COLOR=colors[i]
  ENDFOR
END
```

---

## Practical Example: Solar Activity Plot

```idl
PRO plot_solar_activity
  ; Simulated sunspot number data (monthly)
  n_months = 132    ; 11 years
  time = FINDGEN(n_months) / 12.0    ; Years
  ; Simulate solar cycle
  sunspots = 100.0 * SIN(2.0 * !PI * time / 11.0)^2 + $
             RANDOMN(seed, n_months) * 20.0
  sunspots = sunspots > 0    ; No negative sunspot numbers

  ; 13-month smoothed
  smoothed = SMOOTH(sunspots, 13, /EDGE_TRUNCATE)

  ; Plot
  PLOT, time, sunspots, $
    TITLE='Solar Activity (Simulated)', $
    XTITLE='Years', $
    YTITLE='Sunspot Number', $
    PSYM=3, $    ; Dots
    XSTYLE=1, $
    CHARSIZE=1.3

  OPLOT, time, smoothed, THICK=3, COLOR='0000FF'XL

  ; Annotations
  max_idx = WHERE(smoothed EQ MAX(smoothed))
  XYOUTS, time[max_idx[0]], MAX(smoothed) + 10, 'Solar Maximum', $
    ALIGNMENT=0.5, CHARSIZE=1.1

  min_idx = WHERE(smoothed EQ MIN(smoothed[13:n_months-14])) + 13
  XYOUTS, time[min_idx[0]], MIN(smoothed) - 15, 'Solar Minimum', $
    ALIGNMENT=0.5, CHARSIZE=1.1
END
```

---

## Summary

| Procedure | Description |
|-----------|-------------|
| `PLOT, x, y` | Create a new plot |
| `OPLOT, x, y` | Overlay on existing plot |
| `XYOUTS, x, y, text` | Add text annotation |
| `AXIS` | Add/modify axes |
| `ERRPLOT` | Add error bars |
| `PLOTS, x, y` | Draw line segments |
| `WINDOW, n` | Create graphics window |
| `WSET, n` | Select active window |
| `SET_PLOT, 'PS'` | Switch to PostScript device |
| `DEVICE, FILENAME=f` | Configure output device |
| `!P.MULTI` | Multi-panel layout |

| Keyword | Description |
|---------|-------------|
| TITLE, XTITLE, YTITLE | Plot and axis titles |
| XRANGE, YRANGE | Axis ranges |
| XSTYLE, YSTYLE | Axis style flags |
| LINESTYLE | Line pattern (0-5) |
| PSYM | Plot symbol (1-10) |
| SYMSIZE | Symbol size multiplier |
| THICK | Line thickness |
| COLOR | Line/symbol color |
| CHARSIZE | Character size multiplier |
| /YLOG, /XLOG | Logarithmic axes |

---

**Previous**: [Structures](./09_Structures.md) | **Next**: [Image Display](./11_Image_Display.md)
