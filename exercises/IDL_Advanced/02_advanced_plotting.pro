;+
; Exercise 02: Advanced Plotting
;
; Complete the TODO sections to practice advanced visualization.
;-

PRO exercise_02

    ; === Exercise 1: Multi-panel layout ===
    ; Create a 3x2 panel layout showing 6 different mathematical functions
    ; TODO: Set !P.MULTI for 3 columns, 2 rows
    ; TODO: Plot sin, cos, tan, exp, log, sqrt over [0, 10]
    ; TODO: Reset !P.MULTI

    ; === Exercise 2: Filled contour with color bar ===
    ; Create a filled contour plot of z = sin(x)*cos(y) with 20 levels
    ; Add a color bar at the bottom
    nx = 100 & ny = 100
    x = FINDGEN(nx) * 0.1 - 5.0
    y = FINDGEN(ny) * 0.1 - 5.0
    z = SIN(x # REPLICATE(1, ny)) * COS(REPLICATE(1, nx) # y)
    ; TODO: Create filled contour with 20 levels
    ; TODO: Overlay contour lines
    ; TODO: Add axis labels and title

    ; === Exercise 3: Publication PostScript ===
    ; Create an EPS file with two panels:
    ; (a) A line plot with error bars
    ; (b) A contour plot
    ; Use thick lines, large fonts, and proper axis labels
    ; TODO: SET_PLOT, 'PS' and configure DEVICE
    ; TODO: Create the two panels
    ; TODO: Close DEVICE and restore SET_PLOT, 'X'

    ; === Exercise 4: Annotated plot ===
    ; Create a sine wave plot and annotate:
    ; (a) Mark the first peak with a filled circle
    ; (b) Draw a horizontal line at y=0
    ; (c) Add a text label at the peak
    ; (d) Draw an arrow pointing to the first zero crossing
    ; TODO: Implement using PLOTS, XYOUTS, USERSYM, ARROW

    ; === Exercise 5: Surface + contour combo ===
    ; Create a surface plot of DIST(40) with a contour projection below
    ; TODO: Use SURFACE with AX, AZ keywords
    ; TODO: Overlay CONTOUR with /T3D

END
