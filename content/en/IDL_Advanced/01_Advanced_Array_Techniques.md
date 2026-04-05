# 01. Advanced Array Techniques

**Next**: [Advanced Plotting](./02_Advanced_Plotting.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Reshape arrays with REFORM and understand memory layout implications
2. Rebin data for up-sampling and down-sampling with REBIN and CONGRID
3. Perform dimension-wise reduction with TOTAL, MEAN, and MEDIAN
4. Apply smoothing filters with SMOOTH, MEDIAN, and CONVOL
5. Extract statistical summaries with IMAGE_STATISTICS
6. Write efficient multi-dimensional array manipulation patterns

---

## 1. Array Reshaping with REFORM

`REFORM` changes an array's dimensions without copying data. The total number of elements must remain the same.

```idl
; Create a 1D array of 12 elements
arr = INDGEN(12)
PRINT, SIZE(arr, /DIMENSIONS)   ; 12

; Reshape to 3x4
arr2d = REFORM(arr, 3, 4)
PRINT, SIZE(arr2d, /DIMENSIONS) ; 3 4

; Reshape to 2x2x3
arr3d = REFORM(arr, 2, 2, 3)
PRINT, SIZE(arr3d, /DIMENSIONS) ; 2 2 3

; Flatten back to 1D
flat = REFORM(arr3d, 12)
```

### Common Use Cases

```idl
; Remove degenerate dimensions (trailing dimensions of size 1)
; e.g., 512 x 512 x 1 -> 512 x 512
img = FLTARR(512, 512, 1)
img = REFORM(img)               ; Now 512 x 512
PRINT, SIZE(img, /DIMENSIONS)   ; 512 512

; Reorganize a data cube for time-series extraction
; datacube: [nx, ny, nt] -> extract spatial pixel as 1D time series
datacube = RANDOMU(seed, 256, 256, 100)
; Extract pixel (128, 128) across all time steps
pixel_ts = REFORM(datacube[128, 128, *])  ; [100]
PRINT, SIZE(pixel_ts, /DIMENSIONS)        ; 100
```

### Memory Layout: Column-Major Order

IDL uses **column-major** order (Fortran convention). The first subscript varies fastest in memory:

```idl
; For arr[3, 4], memory layout is:
;   arr[0,0], arr[1,0], arr[2,0],   <- first column
;   arr[0,1], arr[1,1], arr[2,1],   <- second column
;   ...
; This is the OPPOSITE of C/Python (row-major)

; REFORM does not reorder data, it just changes the shape metadata
; So REFORM(arr, 4, 3) gives a DIFFERENT logical view than TRANSPOSE(arr)
arr = INDGEN(3, 4)
PRINT, arr
;        0       1       2
;        3       4       5
;        6       7       8
;        9      10      11

PRINT, REFORM(arr, 4, 3)
;        0       1       2       3
;        4       5       6       7
;        8       9      10      11

PRINT, TRANSPOSE(arr)
;        0       3       6       9
;        1       4       7      10
;        2       5       8      11
```

---

## 2. Rebinning with REBIN

`REBIN` resizes an array by integer factors. It averages when shrinking and interpolates when expanding. The new dimensions must be integer multiples (or divisors) of the original.

```idl
; 2x2 -> 4x4 (expand by factor 2)
small = [[1.0, 2.0], [3.0, 4.0]]
big = REBIN(small, 4, 4)
PRINT, big
;       1.00000      1.50000      2.00000      2.00000
;       2.00000      2.50000      3.00000      3.00000
;       3.00000      3.50000      4.00000      4.00000
;       3.00000      3.50000      4.00000      4.00000

; 1024x1024 -> 256x256 (shrink by factor 4, averaging)
big_img = DIST(1024)
small_img = REBIN(big_img, 256, 256)
PRINT, SIZE(small_img, /DIMENSIONS)  ; 256 256
```

### REBIN with /SAMPLE

By default, REBIN uses bilinear interpolation (expansion) or averaging (contraction). Use `/SAMPLE` for nearest-neighbor:

```idl
; Nearest-neighbor expansion (no interpolation)
small = BYTARR(4, 4) + 128B
small[1:2, 1:2] = 255B
big = REBIN(small, 16, 16, /SAMPLE)  ; Each pixel replicated 4x4
```

### Rebinning Multi-Dimensional Data

```idl
; Rebin a 3D data cube: [512, 512, 100] -> [128, 128, 100]
; Only rebin spatial dimensions, keep time dimension
cube = FLTARR(512, 512, 100)
cube_small = REBIN(cube, 128, 128, 100)  ; Factor 4 spatial downsample

; Rebin time: [128, 128, 100] -> [128, 128, 20] (5-step average)
cube_tavg = REBIN(cube_small, 128, 128, 20)
```

---

## 3. Resizing with CONGRID

Unlike REBIN, `CONGRID` can resize to **any** dimension (not just integer multiples). It uses interpolation.

```idl
; Resize from 100x100 to 256x256
img = DIST(100)
img_resized = CONGRID(img, 256, 256)

; Cubic interpolation for higher quality
img_cubic = CONGRID(img, 256, 256, /INTERP, CUBIC=-0.5)

; Compare methods:
;   CONGRID - arbitrary resize, interpolation
;   REBIN   - integer factor only, averaging (better for downsampling)
```

### CONGRID vs REBIN

| Feature | REBIN | CONGRID |
|---------|-------|---------|
| Resize factor | Integer multiples only | Any size |
| Downsampling | Averaging (flux-conserving) | Interpolation (not flux-conserving) |
| Upsampling | Bilinear interpolation | Nearest-neighbor or cubic |
| Speed | Faster | Slower for large arrays |
| Best for | Scientific data reduction | Display/visualization |

---

## 4. Dimensional Reduction with TOTAL

`TOTAL` sums array elements, optionally along a specific dimension.

```idl
; Sum all elements
arr = FINDGEN(3, 4) + 1
PRINT, TOTAL(arr)    ; 78.0000  (sum of 1..12)

; Sum along dimension 1 (columns) -> result is [4] array
col_sum = TOTAL(arr, 1)
PRINT, col_sum       ; 6.00000  15.0000  24.0000  33.0000

; Sum along dimension 2 (rows) -> result is [3] array
row_sum = TOTAL(arr, 2)
PRINT, row_sum       ; 22.0000  26.0000  30.0000

; Cumulative sum
cum = TOTAL(FINDGEN(5)+1, /CUMULATIVE)
PRINT, cum           ; 1.00000  3.00000  6.00000  10.0000  15.0000
```

### Practical: Column-Density Calculation

```idl
; Compute column density from a 3D density cube
; density_cube: [nx, ny, nz] in particles/cm^3
; dz: cell height in cm
density_cube = RANDOMU(seed, 256, 256, 100) * 1e10  ; particles/cm^3
dz = 1e8  ; 1000 km per cell

; Column density = integral(n * dz) along z-axis (dimension 3)
column_density = TOTAL(density_cube, 3) * dz  ; [256, 256] in particles/cm^2
PRINT, SIZE(column_density, /DIMENSIONS)      ; 256 256
```

---

## 5. MEAN, MEDIAN, and Statistical Functions

### MEAN

```idl
arr = FINDGEN(100) + 1
PRINT, MEAN(arr)        ; 50.5000
PRINT, MEAN(arr, /NAN)  ; Ignore NaN values

; Dimension-wise mean
cube = RANDOMU(seed, 100, 100, 50)
time_mean = MEAN(cube, DIMENSION=3)  ; [100, 100] temporal mean
```

### MEDIAN

`MEDIAN` can compute the scalar median or apply a median filter.

```idl
; Scalar median
arr = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]
PRINT, MEDIAN(arr)   ; 3.00000

; Median filter (1D) — window width must be odd
signal = FINDGEN(100) + RANDOMN(seed, 100) * 5
filtered = MEDIAN(signal, 5)  ; 5-point median filter

; 2D median filter
img = DIST(256) + RANDOMN(seed, 256, 256) * 20
img_filtered = MEDIAN(img, 3)  ; 3x3 median filter

; MEDIAN is excellent for removing salt-and-pepper noise
; while preserving edges (unlike SMOOTH)
```

### MOMENT

```idl
; Compute mean, variance, skewness, kurtosis in one call
data = RANDOMN(seed, 10000)
result = MOMENT(data)
PRINT, 'Mean:     ', result[0]
PRINT, 'Variance: ', result[1]
PRINT, 'Skewness: ', result[2]
PRINT, 'Kurtosis: ', result[3]
```

---

## 6. Smoothing with SMOOTH

`SMOOTH` applies a boxcar (running-average) filter.

```idl
; 1D smoothing
signal = SIN(FINDGEN(200) * 0.1) + RANDOMN(seed, 200) * 0.3
smoothed = SMOOTH(signal, 11)  ; 11-point boxcar average

; 2D smoothing
img = DIST(256) + RANDOMN(seed, 256, 256) * 10
img_smooth = SMOOTH(img, 5)    ; 5x5 boxcar average

; Edge treatment: /EDGE_TRUNCATE prevents edge artifacts
img_smooth = SMOOTH(img, 5, /EDGE_TRUNCATE)

; /NAN keyword: propagate or ignore NaN
data_with_nan = FINDGEN(100)
data_with_nan[50] = !VALUES.F_NAN
smoothed = SMOOTH(data_with_nan, 5, /NAN)  ; Ignore NaN in average
```

### Gaussian Smoothing

For Gaussian smoothing, convolve with a Gaussian kernel:

```idl
; Create a Gaussian kernel
kernel_size = 11
sigma = 2.0
x = FINDGEN(kernel_size) - kernel_size/2
kernel_1d = EXP(-x^2 / (2.0 * sigma^2))
kernel_1d = kernel_1d / TOTAL(kernel_1d)  ; Normalize

; 2D Gaussian kernel via outer product
kernel_2d = kernel_1d # kernel_1d

; Apply
img_gauss = CONVOL(img, kernel_2d, /EDGE_TRUNCATE)
```

---

## 7. Convolution with CONVOL

`CONVOL` performs general N-dimensional convolution with a user-defined kernel.

```idl
; Sharpening kernel (Laplacian + identity)
sharpen_kernel = [[-1, -1, -1], $
                  [-1,  9, -1], $
                  [-1, -1, -1]]
img_sharp = CONVOL(FLOAT(img), sharpen_kernel, /EDGE_TRUNCATE)

; Edge detection kernel (horizontal Sobel)
sobel_h = [[-1, -2, -1], $
           [ 0,  0,  0], $
           [ 1,  2,  1]]
edges = CONVOL(FLOAT(img), sobel_h)

; Custom kernel with normalization
; The third argument is the scale factor (sum of kernel)
blur_kernel = FLTARR(5, 5) + 1.0
img_blur = CONVOL(FLOAT(img), blur_kernel, TOTAL(blur_kernel), /EDGE_TRUNCATE)
```

### Convolution Keywords

```idl
; /CENTER — center the kernel (default for odd-sized kernels)
; /EDGE_TRUNCATE — extend edge pixels
; /EDGE_WRAP — wrap around (periodic boundary)
; /EDGE_ZERO — zero-pad edges (default)
; /NAN — handle NaN values

result = CONVOL(data, kernel, /CENTER, /EDGE_TRUNCATE, /NAN)
```

---

## 8. IMAGE_STATISTICS

`IMAGE_STATISTICS` computes comprehensive statistics for an image array in one call.

```idl
img = DIST(512) + RANDOMN(seed, 512, 512) * 10

IMAGE_STATISTICS, img, $
    COUNT=count, $
    MEAN=img_mean, $
    STDDEV=img_stddev, $
    VARIANCE=img_var, $
    MINIMUM=img_min, $
    MAXIMUM=img_max, $
    DATA_SUM=img_sum

PRINT, 'Pixel count: ', count
PRINT, 'Mean:        ', img_mean
PRINT, 'Std Dev:     ', img_stddev
PRINT, 'Min:         ', img_min
PRINT, 'Max:         ', img_max
```

### Masked Statistics

```idl
; Compute statistics only within a region of interest
mask = BYTARR(512, 512)
mask[100:400, 100:400] = 1B

IMAGE_STATISTICS, img, MASK=mask, $
    MEAN=roi_mean, STDDEV=roi_stddev

PRINT, 'ROI Mean:    ', roi_mean
PRINT, 'ROI Std Dev: ', roi_stddev
```

---

## 9. Multi-Dimensional Array Operations

### Broadcasting-Style Operations

IDL does not have NumPy-style broadcasting, but you can use `REBIN` and `REFORM` to achieve similar results:

```idl
; Subtract row-wise mean from a 2D array
; data: [nx, nt]
data = RANDOMU(seed, 100, 50)  ; 100 spatial pixels, 50 time steps

; Compute mean over time (dimension 2) -> [100]
time_mean = MEAN(data, DIMENSION=2)

; Need to replicate time_mean to [100, 50] for subtraction
; Method 1: REBIN after REFORM
time_mean_2d = REBIN(time_mean, 100, 50)
data_detrended = data - time_mean_2d

; Method 2: Using ## (matrix multiply with column vector)
; time_mean is [100], REPLICATE(1.0, 50) is [50]
; time_mean # REPLICATE(1.0, 1, 50) gives [100, 50]
time_mean_2d = time_mean # REPLICATE(1.0, 1, 50)
data_detrended = data - time_mean_2d
```

### WHERE for Multi-Dimensional Indexing

```idl
; Find all pixels above a threshold in a data cube
cube = RANDOMU(seed, 100, 100, 50)
threshold = 0.95

; WHERE returns 1D indices into the flattened array
idx = WHERE(cube GT threshold, count)
PRINT, 'Pixels above threshold: ', count

; Convert 1D index to multi-dimensional subscripts
subscripts = ARRAY_INDICES(cube, idx)
; subscripts: [3, count] — each column is [x, y, t]
PRINT, 'First pixel: x=', subscripts[0,0], ' y=', subscripts[1,0], $
       ' t=', subscripts[2,0]

; Set all above-threshold pixels to NaN
cube[idx] = !VALUES.F_NAN
```

### Efficient Accumulation Patterns

```idl
; Running difference of a time series
; data: [nx, ny, nt]
nt = 100
data = RANDOMU(seed, 256, 256, nt)

; Running difference: diff[t] = data[t] - data[t-1]
diff = data[*, *, 1:nt-1] - data[*, *, 0:nt-2]
PRINT, SIZE(diff, /DIMENSIONS)  ; 256 256 99

; Base-difference: diff[t] = data[t] - data[0]
base = REBIN(data[*, *, 0], 256, 256, nt-1)
base_diff = data[*, *, 1:nt-1] - base
```

---

## 10. Practical Example: Time-Series Analysis of a Solar Image Cube

```idl
; Simulated solar EUV image cube: [512, 512, 200] at 12s cadence
nx = 512 & ny = 512 & nt = 200
cadence = 12.0  ; seconds

; Generate synthetic data (in practice, read from FITS files)
cube = FLTARR(nx, ny, nt)
FOR t = 0, nt-1 DO BEGIN
    cube[*, *, t] = DIST(nx) * (1.0 + 0.1 * SIN(2*!PI*t/50.0))
ENDFOR

; Add noise
cube += RANDOMN(seed, nx, ny, nt) * 10.0

; Step 1: Temporal smoothing (5-frame boxcar)
cube_smooth = FLTARR(nx, ny, nt)
FOR i = 0, nx-1 DO FOR j = 0, ny-1 DO $
    cube_smooth[i, j, *] = SMOOTH(REFORM(cube[i, j, *]), 5, /EDGE_TRUNCATE)

; Step 2: Spatial rebinning to 128x128
cube_rebin = REBIN(cube_smooth, 128, 128, nt)

; Step 3: Compute light curve for a region of interest
x1 = 50 & x2 = 70 & y1 = 50 & y2 = 70
roi = cube_rebin[x1:x2, y1:y2, *]
lightcurve = TOTAL(TOTAL(roi, 1), 1)  ; Sum over x, then y -> [nt]
lightcurve = REFORM(lightcurve)

; Step 4: Compute running difference images
run_diff = cube_rebin[*, *, 1:nt-1] - cube_rebin[*, *, 0:nt-2]

; Step 5: Basic statistics of each frame
frame_means = FLTARR(nt)
frame_stddevs = FLTARR(nt)
FOR t = 0, nt-1 DO BEGIN
    IMAGE_STATISTICS, cube_rebin[*, *, t], MEAN=m, STDDEV=s
    frame_means[t] = m
    frame_stddevs[t] = s
ENDFOR

; Plot results
time = FINDGEN(nt) * cadence / 60.0  ; minutes
WINDOW, 0, XSIZE=800, YSIZE=600
!P.MULTI = [0, 1, 3]
PLOT, time, lightcurve, XTITLE='Time (min)', YTITLE='Intensity', $
    TITLE='ROI Light Curve'
PLOT, time, frame_means, XTITLE='Time (min)', YTITLE='Mean DN', $
    TITLE='Frame Mean'
PLOT, time, frame_stddevs, XTITLE='Time (min)', YTITLE='Std Dev', $
    TITLE='Frame Standard Deviation'
!P.MULTI = 0
```

---

## Summary

| Function | Purpose | Key Keywords |
|----------|---------|-------------|
| `REFORM` | Reshape without copying | — |
| `REBIN` | Integer-factor resize (average/interpolate) | `/SAMPLE` |
| `CONGRID` | Arbitrary resize (interpolation) | `/INTERP`, `CUBIC=` |
| `TOTAL` | Sum along dimension | `/CUMULATIVE`, `/NAN` |
| `MEAN` | Mean along dimension | `DIMENSION=`, `/NAN` |
| `MEDIAN` | Median scalar or filter | width argument |
| `SMOOTH` | Boxcar average | `/EDGE_TRUNCATE`, `/NAN` |
| `CONVOL` | N-D convolution | `/EDGE_TRUNCATE`, `/EDGE_WRAP` |
| `IMAGE_STATISTICS` | Comprehensive image stats | `MASK=` |
| `WHERE` + `ARRAY_INDICES` | Multi-dimensional indexing | — |

---

**Next**: [Advanced Plotting](./02_Advanced_Plotting.md)
