;+
; Exercise 15: Capstone — Solar Event Analysis
;
; Complete this mini-project using synthetic data.
;-

PRO exercise_15

    ; === Setup: Generate synthetic flare data ===
    ; Create a 64x64x80 cube simulating a flare at pixel (32,32)
    ; Cadence: 30 seconds, Flare peak at frame 40
    ; Background: 200 DN, Peak flare: 1000 DN above background
    nx = 64 & ny = 64 & nt = 80
    cadence = 30.0
    ; TODO: Create cube with background + Gaussian flare brightening

    ; === Exercise 1: Running difference ===
    ; Compute running difference images
    ; Find the frame with the largest absolute difference (most dynamic)
    ; Display that frame with a blue-red color table
    ; TODO: Subtract consecutive frames, find MAX, display

    ; === Exercise 2: Light curve extraction ===
    ; Extract the mean intensity in a 10x10 box centered on the flare
    ; Plot the light curve with time in minutes on the x-axis
    ; TODO: Define ROI, MEAN over spatial dims, PLOT

    ; === Exercise 3: Flare onset detection ===
    ; Determine the pre-flare baseline (first 10 frames)
    ; Detect onset: first time intensity > baseline + 3*sigma
    ; Detect peak: frame with maximum intensity
    ; Print onset time, peak time, rise time, enhancement factor
    ; TODO: MEAN/STDDEV of baseline, WHERE for threshold crossing

    ; === Exercise 4: Multi-panel summary figure ===
    ; Create a 2x2 plot with:
    ; (a) Pre-flare image (frame 5)
    ; (b) Peak image (frame at peak)
    ; (c) Running difference at peak
    ; (d) Light curve with onset and peak marked
    ; Save as PostScript
    ; TODO: SET_PLOT 'PS', !P.MULTI, TV, PLOT, PLOTS for markers

    ; === Exercise 5: Save results ===
    ; Save the following to an IDL .sav file:
    ; cube, lightcurve, onset_time, peak_time, rise_time, enhancement
    ; Then RESTORE and verify all variables are present
    ; TODO: SAVE, RESTORE, HELP

END
