# 10. Image Processing

**Previous**: [Spectral Analysis](./09_Spectral_Analysis.md) | **Next**: [Curve Fitting](./11_Curve_Fitting.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply smoothing and median filters to solar images
2. Use morphological operations (open, close, dilate, erode)
3. Detect edges with Sobel, Roberts, and Laplacian operators
4. Identify and label connected regions with LABEL_REGION
5. Create running difference and base difference images
6. Track features across a time series of solar images

---

## 1. Image Filtering

### Smoothing Filters

```idl
; Read a solar image (or create synthetic data)
img = DIST(512) + RANDOMN(seed, 512, 512) * 30.0
img = FLOAT(img)

; Boxcar smoothing
img_boxcar = SMOOTH(img, 5, /EDGE_TRUNCATE)

; Gaussian smoothing
sigma = 2.0
ksize = 11
x = FINDGEN(ksize) - ksize/2
kernel = EXP(-x^2 / (2*sigma^2))
kernel = kernel / TOTAL(kernel)
kernel_2d = kernel # kernel
img_gauss = CONVOL(img, kernel_2d, /EDGE_TRUNCATE)

; Median filter (good for salt-and-pepper noise)
img_median = MEDIAN(img, 5)
```

### Custom Convolution Kernels

```idl
; Unsharp masking (sharpening)
img_smooth = SMOOTH(img, 11, /EDGE_TRUNCATE)
img_unsharp = img + 2.0 * (img - img_smooth)

; High-pass filter
; HPF = Original - LPF
img_highpass = img - SMOOTH(img, 21, /EDGE_TRUNCATE)

; Laplacian of Gaussian (LoG) for blob detection
; First smooth, then apply Laplacian
img_log = CONVOL(SMOOTH(img, 3), $
    [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], /EDGE_TRUNCATE)
```

---

## 2. Edge Detection

### Sobel Operator

```idl
; SOBEL function — built-in edge detection
img_edges = SOBEL(img)

; Manual Sobel with directional components
sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

grad_x = CONVOL(FLOAT(img), sobel_x, /EDGE_TRUNCATE)
grad_y = CONVOL(FLOAT(img), sobel_y, /EDGE_TRUNCATE)
gradient_mag = SQRT(grad_x^2 + grad_y^2)
gradient_dir = ATAN(grad_y, grad_x)  ; Direction in radians
```

### Roberts Cross

```idl
; ROBERTS function — 2x2 edge detection (faster, less noise-robust)
img_roberts = ROBERTS(img)

; Manual Roberts
roberts_1 = [[1, 0], [0, -1]]
roberts_2 = [[0, 1], [-1, 0]]
r1 = CONVOL(FLOAT(img), roberts_1)
r2 = CONVOL(FLOAT(img), roberts_2)
roberts_mag = SQRT(r1^2 + r2^2)
```

### Laplacian

```idl
; Laplacian (second derivative) — detects zero crossings
laplacian_kernel = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
img_laplacian = CONVOL(FLOAT(img), laplacian_kernel, /EDGE_TRUNCATE)

; 8-connected Laplacian
laplacian_8 = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
img_lap8 = CONVOL(FLOAT(img), laplacian_8, /EDGE_TRUNCATE)
```

### Canny-Like Edge Detection

```idl
; Approximate Canny edge detection:
; 1. Gaussian smooth
; 2. Compute gradient magnitude and direction
; 3. Non-maximum suppression
; 4. Hysteresis thresholding

; Step 1: Smooth
img_smooth = CONVOL(FLOAT(img), kernel_2d, /EDGE_TRUNCATE)

; Step 2: Gradient
gx = CONVOL(img_smooth, sobel_x, /EDGE_TRUNCATE)
gy = CONVOL(img_smooth, sobel_y, /EDGE_TRUNCATE)
gmag = SQRT(gx^2 + gy^2)

; Step 3-4: Threshold (simplified)
threshold_high = 0.3 * MAX(gmag)
threshold_low = 0.1 * MAX(gmag)
edges = BYTARR(512, 512)
strong = WHERE(gmag GT threshold_high)
edges[strong] = 255B
weak = WHERE(gmag GT threshold_low AND gmag LE threshold_high)
; Connect weak edges to strong ones (simplified: just include them)
edges[weak] = 128B
```

---

## 3. Morphological Operations

### Structuring Elements

```idl
; Create structuring elements
; Disk-shaped
radius = 3
se_disk = SHIFT(DIST(2*radius+1), radius, radius) LE radius

; Square
se_square = REPLICATE(1B, 5, 5)

; Cross-shaped
se_cross = BYTARR(5, 5)
se_cross[2, *] = 1B
se_cross[*, 2] = 1B
```

### Erosion and Dilation

```idl
; Create a binary image
binary = BYTARR(256, 256)
binary[50:100, 50:100] = 1B     ; Square
binary[150:200, 150:200] = 1B   ; Another square
binary[120:130, 80:90] = 1B     ; Small bridge

; Erosion — shrinks bright regions
eroded = ERODE(binary, se_disk)

; Dilation — expands bright regions
dilated = DILATE(binary, se_disk)

; Display
WINDOW, 0, XSIZE=768, YSIZE=256
!P.MULTI = [0, 3, 1]
TV, binary * 255B, 0
TV, eroded * 255B, 1
TV, dilated * 255B, 2
!P.MULTI = 0
```

### Opening and Closing

```idl
; Opening = Erosion then Dilation (removes small bright features)
opened = MORPH_OPEN(binary, se_disk)

; Closing = Dilation then Erosion (fills small dark holes)
closed = MORPH_CLOSE(binary, se_disk)

; Top-hat transform: original - opening (extracts small bright features)
tophat = binary - MORPH_OPEN(binary, se_disk)

; Black-hat transform: closing - original (extracts small dark features)
blackhat = MORPH_CLOSE(binary, se_disk) - binary
```

### Application: Cleaning Solar Magnetograms

```idl
; Remove small-scale noise from a thresholded magnetogram
; Threshold to create binary active region mask
threshold = 50.0  ; Gauss
ar_mask = BYTE(ABS(magnetogram) GT threshold)

; Remove isolated pixels (noise)
se = SHIFT(DIST(5), 2, 2) LE 2  ; Circular SE, radius 2
ar_clean = MORPH_OPEN(ar_mask, se)

; Fill holes in active regions
ar_filled = MORPH_CLOSE(ar_clean, REPLICATE(1B, 7, 7))
```

---

## 4. Connected Component Labeling

### LABEL_REGION

```idl
; Label connected components in a binary image
binary = BYTARR(256, 256)
binary[20:50, 30:60] = 1B
binary[80:120, 100:150] = 1B
binary[180:220, 50:80] = 1B
binary[150:170, 180:210] = 1B

; Label regions
labels = LABEL_REGION(binary)
; labels: integer array, each connected region gets a unique label
; Background = 0, first region = 1, second = 2, etc.

n_regions = MAX(labels)
PRINT, 'Number of regions: ', n_regions

; Analyze each region
FOR i = 1, n_regions DO BEGIN
    region_pixels = WHERE(labels EQ i, n_pix)
    xy = ARRAY_INDICES(binary, region_pixels)
    x_center = MEAN(xy[0, *])
    y_center = MEAN(xy[1, *])
    PRINT, 'Region ', i, ': ', n_pix, ' pixels, center = (', $
        x_center, ', ', y_center, ')'
ENDFOR
```

### Application: Active Region Detection

```idl
; Detect and characterize active regions in a magnetogram
; 1. Threshold
ar_binary = BYTE(ABS(magnetogram) GT 100)

; 2. Clean morphologically
se = SHIFT(DIST(7), 3, 3) LE 3
ar_binary = MORPH_CLOSE(MORPH_OPEN(ar_binary, se), se)

; 3. Label regions
labels = LABEL_REGION(ar_binary)
n_ar = MAX(labels)

; 4. Compute properties for each region
PRINT, 'Detected ', n_ar, ' active regions'
FOR i = 1, n_ar DO BEGIN
    pixels = WHERE(labels EQ i, area)
    IF area LT 100 THEN CONTINUE  ; Skip tiny regions

    xy = ARRAY_INDICES(magnetogram, pixels)
    flux_values = magnetogram[pixels]

    PRINT, '--- AR #', i, ' ---'
    PRINT, '  Area:          ', area, ' pixels'
    PRINT, '  Centroid:      (', MEAN(xy[0,*]), ', ', MEAN(xy[1,*]), ')'
    PRINT, '  Max |B|:       ', MAX(ABS(flux_values)), ' G'
    PRINT, '  Mean B:        ', MEAN(flux_values), ' G'
    PRINT, '  Unsigned flux:  ', TOTAL(ABS(flux_values)), ' G*pix'
ENDFOR
```

---

## 5. Running Difference Images

Running differences highlight temporal changes and are widely used in solar physics to detect eruptions, waves, and flows.

### Simple Running Difference

```idl
; data_cube: [nx, ny, nt]
nx = 512 & ny = 512 & nt = 100
cube = FLTARR(nx, ny, nt)
; (in practice, read from calibrated FITS files)

; Running difference: diff[t] = image[t] - image[t-1]
diff_cube = cube[*, *, 1:nt-1] - cube[*, *, 0:nt-2]
; Result: [nx, ny, nt-1]

; Display a running difference image
; Use symmetric scaling around zero
diff_img = diff_cube[*, *, 50]
vmax = 3.0 * STDDEV(diff_img)
LOADCT, 33  ; Blue-Red
TV, BYTSCL(diff_img, MIN=-vmax, MAX=vmax)
```

### Base Difference

```idl
; Base difference: diff[t] = image[t] - image[0]
; Better for tracking cumulative changes
base_image = cube[*, *, 0]
base_3d = REBIN(base_image, nx, ny, nt)
base_diff = cube - base_3d
```

### Percentage Running Difference

```idl
; Percentage difference: (image[t] - image[t-1]) / image[t-1]
; Normalizes for intensity variations across the image
pct_diff = FLTARR(nx, ny, nt-1)
FOR t = 0, nt-2 DO BEGIN
    denom = cube[*, *, t]
    valid = WHERE(denom GT 10.0, nvalid)  ; Avoid division by near-zero
    frame = FLTARR(nx, ny)
    IF nvalid GT 0 THEN $
        frame[valid] = (cube[valid + LONG(t+1)*LONG(nx)*LONG(ny)] - $
                        cube[valid + LONG(t)*LONG(nx)*LONG(ny)]) / denom[valid]
    pct_diff[*, *, t] = frame
ENDFOR
```

---

## 6. Feature Detection and Tracking

### Sunspot Detection

```idl
; Detect sunspots in a continuum image
; Sunspots are dark regions (< threshold of quiet-Sun intensity)

; Estimate quiet-Sun intensity
qs_intensity = MEDIAN(continuum_img)
threshold = 0.9 * qs_intensity  ; Umbra: < 0.5, Penumbra: 0.5-0.9

; Threshold
sunspot_mask = continuum_img LT threshold

; Clean and label
se = SHIFT(DIST(5), 2, 2) LE 2
sunspot_mask = MORPH_OPEN(sunspot_mask, se)
sunspot_labels = LABEL_REGION(sunspot_mask)

n_spots = MAX(sunspot_labels)
PRINT, 'Detected ', n_spots, ' sunspot regions'
```

### Coronal Loop Tracing

```idl
; Simple ridge detection for coronal loops
; 1. Smooth the image
img_smooth = CONVOL(FLOAT(euv_image), kernel_2d, /EDGE_TRUNCATE)

; 2. Compute Hessian matrix eigenvalues
; Second derivatives
d2x = CONVOL(img_smooth, [[1, -2, 1]], /EDGE_TRUNCATE)
d2y = CONVOL(img_smooth, [[1], [-2], [1]], /EDGE_TRUNCATE)
dxy = CONVOL(CONVOL(img_smooth, [[-1, 0, 1]], /EDGE_TRUNCATE), $
             [[-1], [0], [1]], /EDGE_TRUNCATE)

; Eigenvalues of Hessian
trace_h = d2x + d2y
det_h = d2x * d2y - dxy^2
discriminant = SQRT((d2x - d2y)^2 + 4*dxy^2)

lambda1 = 0.5 * (trace_h + discriminant)
lambda2 = 0.5 * (trace_h - discriminant)

; Ridges: one eigenvalue near zero, the other strongly negative
; (bright ridges = loops)
ridge_strength = ABS(lambda2) * (lambda1 GT -0.1 * ABS(lambda2))
```

### Feature Tracking Across Frames

```idl
; Track a bright feature (e.g., coronal bright point) across time
; Simple centroid tracking

; Define initial position
x_track = FLTARR(nt)
y_track = FLTARR(nt)
box_half = 20  ; Tracking box half-width

; Initial position
x_track[0] = 256.0
y_track[0] = 256.0

FOR t = 1, nt-1 DO BEGIN
    ; Extract search box around previous position
    x0 = ROUND(x_track[t-1]) - box_half > 0
    x1 = ROUND(x_track[t-1]) + box_half < (nx-1)
    y0 = ROUND(y_track[t-1]) - box_half > 0
    y1 = ROUND(y_track[t-1]) + box_half < (ny-1)

    subimg = cube[x0:x1, y0:y1, t]

    ; Intensity-weighted centroid
    total_intensity = TOTAL(subimg)
    IF total_intensity GT 0 THEN BEGIN
        xx = FINDGEN(x1-x0+1) + x0
        yy = FINDGEN(y1-y0+1) + y0
        xx2d = xx # REPLICATE(1.0, y1-y0+1)
        yy2d = REPLICATE(1.0, x1-x0+1) # yy
        x_track[t] = TOTAL(xx2d * subimg) / total_intensity
        y_track[t] = TOTAL(yy2d * subimg) / total_intensity
    ENDIF ELSE BEGIN
        x_track[t] = x_track[t-1]
        y_track[t] = y_track[t-1]
    ENDELSE
ENDFOR

; Plot track
PLOT, x_track, y_track, PSYM=-1, $
    XTITLE='X (pixels)', YTITLE='Y (pixels)', $
    TITLE='Feature Track'
```

---

## 7. Practical: Solar Flare Ribbon Detection

```idl
; Detect flare ribbons using running difference in AIA 1600 A

; Step 1: Compute running difference
diff = cube[*, *, 1:nt-1] - cube[*, *, 0:nt-2]

; Step 2: Threshold for bright enhancements (flare ribbons)
FOR t = 0, nt-2 DO BEGIN
    frame = diff[*, *, t]
    thresh = 5.0 * STDDEV(frame)  ; 5-sigma above background

    ribbon_mask = frame GT thresh

    ; Step 3: Clean morphologically
    se = REPLICATE(1B, 3, 3)
    ribbon_mask = MORPH_CLOSE(ribbon_mask, se)

    ; Step 4: Label ribbons
    labels = LABEL_REGION(ribbon_mask)
    n_ribbons = MAX(labels)

    IF n_ribbons GT 0 THEN BEGIN
        ; Compute total ribbon area
        total_area = TOTAL(ribbon_mask)
        PRINT, 'Frame ', t, ': ', n_ribbons, ' ribbon segments, ', $
            total_area, ' pixels'
    ENDIF
ENDFOR
```

---

## Summary

| Technique | Key Functions | Purpose |
|-----------|-------------|---------|
| Smoothing | `SMOOTH`, `MEDIAN`, `CONVOL` | Noise reduction |
| Edge detection | `SOBEL`, `ROBERTS`, Laplacian | Boundary finding |
| Morphology | `ERODE`, `DILATE`, `MORPH_OPEN`, `MORPH_CLOSE` | Shape cleaning |
| Labeling | `LABEL_REGION` | Connected components |
| Running difference | Array subtraction | Temporal changes |
| Feature tracking | Centroid computation | Motion analysis |

---

**Previous**: [Spectral Analysis](./09_Spectral_Analysis.md) | **Next**: [Curve Fitting](./11_Curve_Fitting.md)
