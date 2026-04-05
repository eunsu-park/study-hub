;+
; Exercise 10: Image Processing
;-

PRO exercise_10

    ; === Exercise 1: Edge detection comparison ===
    ; Apply Sobel, Roberts, and Laplacian to DIST(256)
    ; Display all three results side by side
    ; TODO: SOBEL, ROBERTS, CONVOL with Laplacian kernel

    ; === Exercise 2: Morphological cleaning ===
    ; Create a binary image with 5 circles (different sizes) + random noise pixels
    ; Use morphological opening to remove noise, then label the circles
    ; Report the area (in pixels) of each detected region
    ; TODO: Create binary, add noise, MORPH_OPEN, LABEL_REGION

    ; === Exercise 3: Running difference movie ===
    ; Generate a synthetic 64x64x50 cube with a moving bright spot
    ; Compute running differences and find the frame with maximum change
    ; TODO: Create cube, compute diff, find MAX frame

    ; === Exercise 4: Feature tracking ===
    ; Create a synthetic image cube with a bright point moving diagonally
    ; Track its position using intensity-weighted centroid
    ; Plot the track (x vs y) and velocity (dx/dt, dy/dt)
    ; TODO: Create moving feature, implement centroid tracking

    ; === Exercise 5: Sunspot detection ===
    ; Given a synthetic continuum image (DIST as background, dark circles as spots):
    ; Threshold to create a binary mask, clean with morphology, label regions
    ; Compute area and centroid of each detected sunspot
    ; TODO: Create synthetic image, threshold, MORPH_OPEN, LABEL_REGION

END
