; Exercise 11: Image Display
;
; Practice image display, scaling, and color tables.

; Exercise 1: Create a 256x256 test image with a Gaussian blob
; at the center and display it with BYTSCL and different color tables.
PRO exercise_11a
  ; TODO: Create x and y index arrays
  ; TODO: Compute Gaussian: A * exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))
  ; TODO: Display with LOADCT, 0 (grayscale) using BYTSCL
  ; TODO: Display with LOADCT, 39 (rainbow) using BYTSCL
  ; Hint: Use DIST(256) as a starting point, or build from scratch
END

; Exercise 2: Display the same image with three different scaling
; strategies: linear, sqrt, and log.
PRO exercise_11b
  image = DIST(256) + 1.0
  ; TODO: Linear: BYTSCL(image)
  ; TODO: Sqrt: BYTSCL(SQRT(image))
  ; TODO: Log: BYTSCL(ALOG10(image))
  ; TODO: Display all three side by side in one window
  ; Hint: Create window with XSIZE=768, YSIZE=256, use TV with position
END

; Exercise 3: Create an RGB false-color composite from three
; grayscale "channels" (simulating multi-wavelength solar images).
PRO exercise_11c
  ; TODO: Create three channels (e.g., shifted DIST images)
  ; TODO: Scale each to byte range
  ; TODO: Compose into [3, nx, ny] RGB array
  ; TODO: Display with TV, rgb, TRUE=1
  ; Hint: DEVICE, DECOMPOSED=1 for true-color display
END

; Exercise 4: Write a procedure that displays an image with
; a colorbar below it using PLOT coordinates.
PRO exercise_11d, image
  ; TODO: If no argument, create image = DIST(128)
  ; TODO: Display image in top portion of window
  ; TODO: Create a colorbar (1D gradient) and display below
  ; TODO: Add min/max labels to the colorbar
  ; Hint: colorbar = BINDGEN(256) # REPLICATE(1B, 10)
END
