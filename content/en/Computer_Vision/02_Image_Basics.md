# Image Basic Operations

## Overview

The foundation of image processing is reading, displaying, and saving image files. This document covers OpenCV's basic I/O functions, pixel-level access, and Region of Interest (ROI) setup.

**Difficulty**: ⭐ (Beginner)

## Learning Objectives

After completing this lesson, you will be able to:

1. Master `cv2.imread()`, `cv2.imshow()`, `cv2.imwrite()` functions
2. Understand and utilize IMREAD flags
3. Understand image coordinate system (y, x order)
4. Access and modify pixels
5. Set ROI (Region of Interest) and copy images

---

## Table of Contents

Before the OpenCV reference, read [**Theory & Principles**](#theory--principles) — how a continuous scene becomes a discrete array of integers, the pixel coordinate convention OpenCV uses, and why BGR (not RGB) is the default channel order.

1. [Reading Images - imread()](#1-reading-images---imread)
2. [Displaying Images - imshow()](#2-displaying-images---imshow)
3. [Saving Images - imwrite()](#3-saving-images---imwrite)
4. [Checking Image Properties](#4-checking-image-properties)
5. [Coordinate System and Pixel Access](#5-coordinate-system-and-pixel-access)
6. [ROI and Image Copying](#6-roi-and-image-copying)
7. [Practice Problems](#7-practice-problems)
8. [Next Steps](#8-next-steps)
9. [References](#9-references)

---

## Theory & Principles

A "digital image" in OpenCV is a 3D NumPy array of unsigned 8-bit integers. Understanding how a real scene turns into that array — and what the three indices, the integer values, and the channel order actually mean — is the prerequisite for every other operation in this track. This section covers:

- **(A) From continuous scene to discrete array** — sampling, quantization, and where image-processing artifacts ultimately come from.
- **(B) The pixel coordinate system** — why `img[y, x]` has `y` first, and what the axis directions are.
- **(C) Bit depth and dynamic range** — why 8-bit is the default, when it hurts, and what `float32` / 16-bit images are for.
- **(D) Color channels and why OpenCV uses BGR** — the historical accident that keeps biting users.

### A. From a Scene to an Array: Sampling and Quantization

#### A.1 The continuous image model

Before a camera is involved, an image is a **continuous 2D function**

```
f : ℝ² → ℝᶜ          (grayscale: c = 1; color: c = 3)
```

that assigns a light intensity (or a color triple) to every point in some region of the image plane. The real world is continuous in both position and intensity. Converting this to something a computer can store requires two discretization steps.

#### A.2 Sampling — making position discrete

A camera's sensor consists of a finite grid of photosites. Each photosite integrates the light arriving over a small rectangular footprint and reports a single number. Mathematically, this is **sampling** the continuous function `f(x, y)` at integer grid points:

```
I[i, j] = ∫∫ f(x, y) · w(x - j·Δ, y - i·Δ) dx dy
```

where `Δ` is the pixel pitch and `w` is the sensor's point-spread function. The result is an `H × W` array of grid-aligned samples.

What gets lost in this step: **frequencies above the Nyquist rate** (`1 / 2Δ`). Any detail finer than two pixels cannot be represented faithfully by the samples. If the lens fails to remove these high frequencies before they hit the sensor, they **alias** into visible low-frequency moiré patterns. This is why proper downsampling needs a low-pass filter first (the `INTER_AREA` rule from lesson 04) — and why photographers care about Bayer demosaicing and anti-alias filters.

#### A.3 Quantization — making intensity discrete

The sensor output is still a (noisy) real number. Storing it requires choosing a finite set of levels. For an 8-bit unsigned image, the real-valued signal `s ∈ [0, 1]` is mapped to

```
I[i, j] = round(s · 255)   ∈   {0, 1, 2, ..., 255}
```

That round-off is the **quantization error**, bounded by `±0.5 / 255 ≈ ±0.2%`. For most natural-scene photography it is invisible. It becomes visible (as banding) in very smooth gradients — a clear sky, a studio backdrop — where the human eye can detect transitions between adjacent levels.

Two strategies break the banding:

- **Increase bit depth**: 10, 12, or 16 bits per channel. A 16-bit PNG has 65,536 levels per channel instead of 256.
- **Dither**: add sub-pixel noise before quantization so the banding pattern is broken up into texture (imperceptible at normal viewing distance).

### B. The Pixel Coordinate System

OpenCV — and NumPy, and almost every array-based image library — indexes pixels as `img[row, col]`, with the origin `(0, 0)` at the **top-left** corner:

```
      col →  0    1    2    ...    W-1
row    ┌─────────────────────────────┐
  ↓    │                             │
  0    │ ●───────→ x                 │
       │ │                           │
  1    │ │                           │
       │ ▼                           │
  2    │ y                           │
       │                             │
  ...  │                             │
       │                             │
 H-1   │                             │
       └─────────────────────────────┘
```

A few important consequences:

- **`img[y, x]` has `y` first** because `y` is the row index. This reverses from standard mathematical `f(x, y)` notation — a constant source of confusion and bugs. Inside OpenCV function calls that take `(x, y)` tuples (like `cv2.circle(img, (x, y), ...)`), the order is swapped back. **Rule of thumb: arrays are `[y, x]`, function arguments are `(x, y)`.**
- **`y` increases downward.** This is the opposite of math textbooks but matches raster scan order and screen coordinates. A "positive rotation angle" in OpenCV is therefore counter-clockwise as viewed on screen but clockwise in pure mathematical terms.
- **Pixel centers vs pixel corners.** Different conventions treat the coordinate `(0, 0)` as the center of the top-left pixel or its top-left corner. OpenCV uses the *center* convention — important when doing sub-pixel interpolation or geometric transforms (see lesson 04).

### C. Bit Depth and Dynamic Range

#### C.1 The default: `uint8`

OpenCV's default image dtype is `numpy.uint8` — 8 bits per channel, 256 levels per channel, range `[0, 255]`. It matches display hardware (most monitors only show 256 levels per channel) and is what every standard image format (JPEG, PNG-8, BMP) stores.

Consequences to remember:

- **Arithmetic saturates at 255.** `cv2.add(200, 100) = 255`, not 300. If you use `+` instead of `cv2.add` on `uint8` arrays, NumPy wraps around: `np.uint8(200) + np.uint8(100) = 44`. Neither is usually what you want.
- **Subtraction saturates at 0 — or wraps, depending on the operator.** Same rule.
- **Float operations must be converted back carefully.** Divide by 255 to get `[0, 1]` floats, do the math, then multiply by 255 and clip before casting back to `uint8`. Forgetting the clip produces inverted-looking outputs where pixels wrap past 255.

#### C.2 When 8 bits is not enough

- **Raw camera data** (12-14 bits) contains detail in shadows and highlights that gets clipped by 8-bit conversion.
- **Medical images** (CT, MRI) are routinely 12-16 bits per channel — the diagnostic information is in the extra bits.
- **HDR photography** stores linear intensities in `float32` so the tonal range spans multiple orders of magnitude (darkest shadows to brightest sky).
- **Intermediate computation** (filtering, accumulating Laplacians, etc.) should often be done in `float32` or `float64` even if the input and output are `uint8`, to avoid intermediate overflow and quantization error.

OpenCV supports these via dtypes `uint16`, `int16`, `float32`, `float64`. The convention is that a `float32` image in the `[0, 1]` range represents the same intensity as a `uint8` image in the `[0, 255]` range.

### D. Color Channels — and Why OpenCV Uses BGR

#### D.1 A color image is three grayscale images

A color image stores three values per pixel — typically one each for red, green, and blue channels. These three values form three parallel "grayscale" 2D arrays, stacked along a third axis:

```
img.shape == (H, W, 3)     # H rows × W cols × 3 channels
img[y, x]                 # → array of length 3, one value per channel
img[:, :, 0]              # → the full blue channel  (under OpenCV's BGR order)
img[:, :, 1]              # → the full green channel
img[:, :, 2]              # → the full red channel
```

The idea generalizes: 4-channel images add an alpha channel for transparency (`shape == (H, W, 4)`), and multispectral / hyperspectral images can have dozens or hundreds of channels.

#### D.2 Why BGR instead of RGB

OpenCV's `imread` and `imshow` use **BGR** (blue-green-red) channel order, while almost every other library (PIL, matplotlib, TensorFlow, PyTorch, web standards) uses RGB. The reason is historical: early 24-bit Windows bitmap files stored pixels in the `blue, green, red` byte order (little-endian representation of the 3-byte pixel), and OpenCV mirrored that layout to allow zero-copy loading of BMPs on Windows. Once code depended on it, the convention froze.

The practical result is a bug class:

```python
img = cv2.imread('cat.jpg')          # BGR
plt.imshow(img)                      # expects RGB — colors look wrong
plt.imshow(img[:, :, ::-1])          # fix: reverse channel axis
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))   # explicit fix, preferred
```

The matplotlib-looks-wrong symptom (reds become blues, skies become red-orange) is so common that it is worth internalizing the rule: **cross the OpenCV / non-OpenCV boundary with `cv2.cvtColor`.**

### From Theory to the Functions Below

- `cv2.imread(path, flag)` — reads from disk, producing the `uint8` BGR array described in §D.
- `cv2.imshow(window, img)` — displays BGR directly (consistent with `imread`), which is why matplotlib requires the swap and `imshow` does not.
- `cv2.imwrite(path, img)` — writes with the same BGR convention and applies the quantization step of §A.3 when the format demands it (JPEG compression is lossy; PNG is lossless).
- `img.shape`, `img.dtype`, `img.size` — the dimensions and bit depth of §C.
- `img[y, x]`, `img[y1:y2, x1:x2]` — NumPy indexing using the `(row, col)` convention of §B.
- `img[:, :, c]` — channel extraction from §D.1.

---

## 1. Reading Images - imread()

### Basic Usage

```python
import cv2

# imread returns None silently on failure (no exception) — always guard against
# this; skipping the check leads to cryptic AttributeError crashes later
img = cv2.imread('image.jpg')

if img is None:
    print("Error: Cannot read image.")
else:
    print(f"Image loaded successfully: {img.shape}")
```

### IMREAD Flags

```
┌─────────────────────────────────────────────────────────────────┐
│                       IMREAD Flag Comparison                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Original Image (PNG with alpha channel)                      │
│   ┌─────────────────────────────────────────────────────┐      │
│   │  R   G   B   A  │  R   G   B   A  │  R   G   B   A  │      │
│   │ 255 100  50 200 │ 255 100  50 200 │ 255 100  50 200 │      │
│   └─────────────────────────────────────────────────────┘      │
│                          │                                     │
│        ┌─────────────────┼─────────────────┐                   │
│        ▼                 ▼                 ▼                   │
│                                                                │
│   IMREAD_COLOR       IMREAD_GRAYSCALE  IMREAD_UNCHANGED        │
│   ┌───────────┐      ┌───────────┐     ┌───────────────┐       │
│   │ B  G  R   │      │   Gray    │     │ B  G  R  A    │       │
│   │ 50 100 255│      │    123    │     │ 50 100 255 200│       │
│   └───────────┘      └───────────┘     └───────────────┘       │
│   shape: (H,W,3)     shape: (H,W)      shape: (H,W,4)          │
│   3-channel BGR      2D, single value  Alpha channel preserved  │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

### Flag Details

```python
import cv2

# IMREAD_COLOR: always produces a 3-channel BGR array regardless of source format
# (even for grayscale JPEGs) — this consistency simplifies downstream processing
img_color = cv2.imread('image.png', cv2.IMREAD_COLOR)
img_color = cv2.imread('image.png', 1)  # Same
img_color = cv2.imread('image.png')     # Can omit (default)

# IMREAD_GRAYSCALE: returns a 2D array — saves 2/3 of memory vs COLOR for tasks
# that don't need color (edge detection, thresholding, template matching)
img_gray = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
img_gray = cv2.imread('image.png', 0)  # Same

# IMREAD_UNCHANGED: the only flag that preserves the alpha channel —
# essential when you need transparency data (compositing, masking operations)
img_unchanged = cv2.imread('image.png', cv2.IMREAD_UNCHANGED)
img_unchanged = cv2.imread('image.png', -1)  # Same

# Compare results
print(f"COLOR: {img_color.shape}")        # (H, W, 3)
print(f"GRAYSCALE: {img_gray.shape}")     # (H, W)
print(f"UNCHANGED: {img_unchanged.shape}") # (H, W, 4) - for PNG
```

### Additional Flags

```python
import cv2

# IMREAD_ANYDEPTH: Load 16-bit/32-bit images as is
img_depth = cv2.imread('depth_map.png', cv2.IMREAD_ANYDEPTH)

# IMREAD_ANYCOLOR: Maintain possible color formats
img_any = cv2.imread('image.jpg', cv2.IMREAD_ANYCOLOR)

# Combining flags
# 16-bit grayscale + maintain color format
img_combined = cv2.imread('image.tiff',
                          cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
```

### Various Image Formats

```python
import cv2

# Supported major formats
formats = [
    'image.jpg',   # JPEG
    'image.png',   # PNG (alpha channel supported)
    'image.bmp',   # BMP
    'image.tiff',  # TIFF
    'image.webp',  # WebP
    'image.ppm',   # PPM/PGM/PBM
]

# Read by format
for filepath in formats:
    img = cv2.imread(filepath)
    if img is not None:
        print(f"{filepath}: {img.shape}")
```

---

## 2. Displaying Images - imshow()

### Basic Usage

```python
import cv2

img = cv2.imread('image.jpg')

# Display image in window
cv2.imshow('Window Name', img)

# Wait for key press
key = cv2.waitKey(0)  # 0 = wait indefinitely

# Close all windows
cv2.destroyAllWindows()
```

### waitKey() Details

```
┌─────────────────────────────────────────────────────────────────┐
│                      waitKey() Behavior                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   cv2.waitKey(delay)                                            │
│                                                                 │
│   delay = 0   → Wait indefinitely until key press               │
│   delay > 0   → Wait delay milliseconds then proceed            │
│   delay = 1   → Minimum wait (often used for video playback)    │
│                                                                 │
│   Return value: ASCII code of pressed key (-1 = timeout)        │
│                                                                 │
│   Examples:                                                     │
│   key = cv2.waitKey(0)                                          │
│   if key == 27:        # ESC key                                │
│       break                                                     │
│   elif key == ord('q'):  # 'q' key                              │
│       break                                                     │
│   elif key == ord('s'):  # 's' key                              │
│       cv2.imwrite('saved.jpg', img)                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Managing Multiple Windows

```python
import cv2

img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# Display multiple windows
cv2.imshow('Image 1', img1)
cv2.imshow('Image 2', img2)

# Set window position
cv2.namedWindow('Positioned', cv2.WINDOW_NORMAL)
cv2.moveWindow('Positioned', 100, 100)  # x=100, y=100 position
cv2.imshow('Positioned', img1)

# Make window resizable
cv2.namedWindow('Resizable', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resizable', 800, 600)
cv2.imshow('Resizable', img1)

cv2.waitKey(0)

# Close specific window
cv2.destroyWindow('Image 1')

# Close all windows
cv2.destroyAllWindows()
```

### Key Input Handling Pattern

```python
import cv2

img = cv2.imread('image.jpg')
original = img.copy()  # Keep a pristine copy — img will be modified in-loop

while True:
    cv2.imshow('Interactive', img)
    # & 0xFF masks the return value to 8 bits: on Linux, waitKey() can return
    # values > 255 due to keyboard modifier flags; masking ensures reliable comparison
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('r'):  # 'r' - restore original
        img = original.copy()
        print("Restored to original")
    elif key == ord('g'):  # 'g' - grayscale
        img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        print("Applied grayscale")
    elif key == ord('s'):  # 's' - save
        cv2.imwrite('output.jpg', img)
        print("Saved")

cv2.destroyAllWindows()
```

### Displaying Images in Jupyter Notebook

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')

# Using matplotlib (need BGR → RGB conversion)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 6))
plt.imshow(img_rgb)
plt.title('Image Display in Jupyter')
plt.axis('off')
plt.show()

# Display multiple images simultaneously
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_rgb)
axes[0].set_title('Original')
axes[0].axis('off')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
axes[1].imshow(gray, cmap='gray')
axes[1].set_title('Grayscale')
axes[1].axis('off')

# Split B, G, R channels
b, g, r = cv2.split(img)
axes[2].imshow(r, cmap='gray')
axes[2].set_title('Red Channel')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

---

## 3. Saving Images - imwrite()

### Basic Usage

```python
import cv2

img = cv2.imread('input.jpg')

# Basic save
success = cv2.imwrite('output.jpg', img)

if success:
    print("Save successful!")
else:
    print("Save failed!")

# Save with format conversion
cv2.imwrite('output.png', img)   # JPEG → PNG
cv2.imwrite('output.bmp', img)   # JPEG → BMP
```

### Setting Compression Quality

```python
import cv2

img = cv2.imread('input.jpg')

# JPEG is lossy: quality=95 is near-lossless (good for archiving); quality=30
# cuts file size dramatically at the cost of visible artifacts — use for thumbnails
cv2.imwrite('high_quality.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
cv2.imwrite('low_quality.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 30])

# PNG is lossless — compression only affects speed/file size, never quality
# Use compression=0 when writing many frames in a loop (speed matters more)
cv2.imwrite('fast_compress.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv2.imwrite('max_compress.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

# WebP offers better compression than JPEG at equivalent visual quality
cv2.imwrite('output.webp', img, [cv2.IMWRITE_WEBP_QUALITY, 80])
```

### Comparing File Sizes

```python
import cv2
import os

img = cv2.imread('input.jpg')

# Save with various qualities
qualities = [10, 30, 50, 70, 90]
for q in qualities:
    filename = f'quality_{q}.jpg'
    cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, q])
    size_kb = os.path.getsize(filename) / 1024
    print(f"Quality {q}: {size_kb:.1f} KB")
```

---

## 4. Checking Image Properties

### shape, dtype, size

```python
import cv2

img = cv2.imread('image.jpg')

# shape: (height, width, channels)
print(f"Shape: {img.shape}")
height, width, channels = img.shape
print(f"Height: {height}px")
print(f"Width: {width}px")
print(f"Channels: {channels}")

# dtype: data type
print(f"Data type: {img.dtype}")  # uint8

# size: total number of elements
print(f"Total elements: {img.size}")  # H * W * C

# Grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"Gray Shape: {gray.shape}")  # (height, width) - no channels

# Safely check channel count
if len(img.shape) == 3:
    h, w, c = img.shape
else:
    h, w = img.shape
    c = 1
```

### Image Info Utility Function

```python
import cv2
import os

def get_image_info(filepath):
    """Returns detailed image file information as dictionary"""
    info = {'filepath': filepath}

    # Check file exists
    if not os.path.exists(filepath):
        info['error'] = 'File does not exist'
        return info

    # File size
    info['file_size_kb'] = os.path.getsize(filepath) / 1024

    # Load image
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        info['error'] = 'Cannot read image'
        return info

    # Basic info
    info['shape'] = img.shape
    info['dtype'] = str(img.dtype)
    info['height'] = img.shape[0]
    info['width'] = img.shape[1]
    info['channels'] = img.shape[2] if len(img.shape) == 3 else 1

    # Statistics
    info['min_value'] = int(img.min())
    info['max_value'] = int(img.max())
    info['mean_value'] = float(img.mean())

    return info

# Usage example
info = get_image_info('sample.jpg')
for key, value in info.items():
    print(f"{key}: {value}")
```

---

## 5. Coordinate System and Pixel Access

### OpenCV Coordinate System

```
┌─────────────────────────────────────────────────────────────────┐
│                     OpenCV Coordinate System                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   (0,0) ────────────────────────────────▶ x (width, columns)    │
│     │                                                           │
│     │    ┌───────────────────────────┐                         │
│     │    │ (0,0)  (1,0)  (2,0)  ...  │                         │
│     │    │ (0,1)  (1,1)  (2,1)  ...  │                         │
│     │    │ (0,2)  (1,2)  (2,2)  ...  │                         │
│     │    │  ...    ...    ...   ...  │                         │
│     │    └───────────────────────────┘                         │
│     ▼                                                           │
│   y (height, rows)                                              │
│                                                                 │
│   Important! Array indexing: img[y, x] or img[row, column]     │
│              OpenCV functions: (x, y) order                     │
│                                                                 │
│   e.g.: img[100, 200]     → pixel at y=100, x=200              │
│         cv2.circle(img, (200, 100), ...)  → at x=200, y=100    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Pixel Access

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# Read single pixel (y, x order!)
pixel = img[100, 200]  # position y=100, x=200
print(f"Pixel value (BGR): {pixel}")  # [B, G, R]

# Access individual channels
b = img[100, 200, 0]  # Blue
g = img[100, 200, 1]  # Green
r = img[100, 200, 2]  # Red
print(f"B={b}, G={g}, R={r}")

# Grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
pixel_gray = gray[100, 200]  # single value
print(f"Grayscale value: {pixel_gray}")
```

### Modifying Pixels

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# Modify single pixel
img[100, 200] = [255, 0, 0]  # Change to blue

# Modify region (100x100 region to red)
img[0:100, 0:100] = [0, 0, 255]  # Red in BGR

# Modify specific channel only
img[0:100, 100:200, 0] = 0    # Blue channel to 0
img[0:100, 100:200, 1] = 0    # Green channel to 0
img[0:100, 100:200, 2] = 255  # Red channel to 255

cv2.imshow('Modified', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### item() and itemset() (for single pixel, faster)

```python
import cv2

img = cv2.imread('image.jpg')

# item(): access single value (faster)
b = img.item(100, 200, 0)
g = img.item(100, 200, 1)
r = img.item(100, 200, 2)

# itemset(): modify single value (faster)
img.itemset((100, 200, 0), 255)  # Blue = 255
img.itemset((100, 200, 1), 0)    # Green = 0
img.itemset((100, 200, 2), 0)    # Red = 0

# Performance comparison
import time

# Regular indexing
start = time.time()
for i in range(10000):
    val = img[100, 200, 0]
print(f"Regular indexing: {time.time() - start:.4f}s")

# Using item()
start = time.time()
for i in range(10000):
    val = img.item(100, 200, 0)
print(f"item(): {time.time() - start:.4f}s")
```

---

## 6. ROI and Image Copying

Processing only the relevant sub-region of an image instead of the full frame is one of the most effective ways to reduce compute cost. ROI also makes operations like copy-paste compositing, face blurring, and targeted color correction straightforward.

### ROI (Region of Interest)

```
┌─────────────────────────────────────────────────────────────────┐
│                       ROI Concept                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Original Image (img)                                          │
│   ┌────────────────────────────────────┐                        │
│   │                                    │                        │
│   │      y1──────────────┐             │                        │
│   │       │    ROI       │             │                        │
│   │       │              │             │                        │
│   │       │              │             │                        │
│   │      y2──────────────┘             │                        │
│   │      x1             x2             │                        │
│   │                                    │                        │
│   └────────────────────────────────────┘                        │
│                                                                 │
│   roi = img[y1:y2, x1:x2]                                       │
│                                                                 │
│   Note: NumPy slicing returns a view!                           │
│         roi modification → original also modified               │
│         Use .copy() if copy is needed                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Setting and Using ROI

```python
import cv2

img = cv2.imread('image.jpg')

# Extract ROI (y1:y2, x1:x2)
# From top-left (100, 50) to bottom-right (300, 250)
roi = img[50:250, 100:300]

print(f"Original size: {img.shape}")
print(f"ROI size: {roi.shape}")  # (200, 200, 3)

# Display ROI
cv2.imshow('Original', img)
cv2.imshow('ROI', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Copying and Pasting ROI

```python
import cv2

img = cv2.imread('image.jpg')

# .copy() creates an independent array — without it, roi is a view into img,
# and modifying it would unexpectedly change the source region too
roi = img[50:150, 100:200].copy()

# Paste to another location — NumPy assigns by value, so this is a true copy
img[200:300, 300:400] = roi  # Sizes must match!

# Copy region within image — copy() is critical here: without it, reading the
# source and writing to the destination could overlap and corrupt the result
src_region = img[0:100, 0:100].copy()
img[-100:, -100:] = src_region

cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### View vs Copy

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')
original_value = img[100, 100, 0]

# View - shares memory with original
roi_view = img[50:150, 50:150]
roi_view[:] = 0  # Make ROI black
print(f"Original modified: {img[100, 100, 0]}")  # 0

# Restore original
img = cv2.imread('image.jpg')

# Copy - independent memory
roi_copy = img[50:150, 50:150].copy()
roi_copy[:] = 0  # Only copy becomes black
print(f"Original preserved: {img[100, 100, 0]}")  # Original value
```

### Copying Entire Image

```python
import cv2

img = cv2.imread('image.jpg')

# Method 1: .copy() method
img_copy1 = img.copy()

# Method 2: NumPy copy
import numpy as np
img_copy2 = np.copy(img)

# Method 3: Slicing then copy (not recommended)
img_copy3 = img[:].copy()

# Wrong copy (creates view)
img_wrong = img  # Same object reference!
img_wrong[0, 0] = [0, 0, 0]
print(f"Original also changed: {img[0, 0]}")  # [0, 0, 0]
```

### Practical ROI Examples

```python
import cv2

def extract_face_region(img, x, y, w, h):
    """Extract face region (with boundary check)"""
    h_img, w_img = img.shape[:2]

    # Boundary check
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)

    return img[y1:y2, x1:x2].copy()


def apply_mosaic(img, x, y, w, h, ratio=0.1):
    """Apply mosaic to specific region"""
    roi = img[y:y+h, x:x+w]

    # Shrink then enlarge (mosaic effect)
    small = cv2.resize(roi, None, fx=ratio, fy=ratio,
                       interpolation=cv2.INTER_NEAREST)
    mosaic = cv2.resize(small, (w, h),
                        interpolation=cv2.INTER_NEAREST)

    img[y:y+h, x:x+w] = mosaic
    return img


# Usage example
img = cv2.imread('image.jpg')
img = apply_mosaic(img, 100, 100, 200, 200, ratio=0.05)
cv2.imshow('Mosaic', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 7. Practice Problems

### Exercise 1: Compare Image Reading Modes

Read one image in three modes (COLOR, GRAYSCALE, UNCHANGED) and compare their shapes. Test with both PNG (with transparency) and JPEG files.

```python
# Hint
import cv2

filepath = 'test.png'
# Read in COLOR, GRAYSCALE, UNCHANGED
# Compare shapes
```

### Exercise 2: Image Quality Analyzer

Save a JPEG image at various qualities (10, 30, 50, 70, 90) and calculate file size and PSNR (Peak Signal-to-Noise Ratio) for each.

```python
# Hint: PSNR calculation
def calculate_psnr(original, compressed):
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
```

### Exercise 3: Create Color Grid

Create a 400x400 image and divide it into 16 cells of 100x100 size, filling each with a different color. Use ROI.

```
┌────┬────┬────┬────┐
│Red │Yell│Gren│Cyan│
├────┼────┼────┼────┤
│Blue│Prpl│Wht │Blck│
├────┼────┼────┼────┤
│... │... │... │... │
└────┴────┴────┴────┘
```

### Exercise 4: Add Image Border

Write a function to add a 10-pixel thick border around an image. (Image size should increase)

```python
def add_border(img, thickness=10, color=(0, 0, 255)):
    """Add border to image"""
    # Hint: use numpy.pad or cv2.copyMakeBorder
    pass
```

### Exercise 5: Pixel-Based Gradient

Create a 300x300 image with a horizontal gradient from black (left) to white (right). Use NumPy broadcasting without loops.

```python
# Hint
import numpy as np
gradient = np.linspace(0, 255, 300)  # 300 values from 0~255
```

---

## 8. Next Steps

In [03_Color_Spaces.md](./03_Color_Spaces.md), you'll learn about various color spaces like BGR, RGB, HSV, LAB and color-based object tracking!

**Topics to Learn Next**:
- BGR vs RGB differences
- Understanding HSV color space
- Color space conversion with `cv2.cvtColor()`
- Color-based object tracking

---

## 9. References

### Official Documentation

- [imread() documentation](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56)
- [imshow() documentation](https://docs.opencv.org/4.x/d7/dfc/group__highgui.html#ga453d42fe4cb60e5723281a89973ee563)
- [imwrite() documentation](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce)

### Related Learning Materials

| Folder | Related Content |
|--------|-----------------|
| [Python/](../Python/) | NumPy slicing, array operations |
| [01_Environment_Setup.md](./01_Environment_Setup.md) | Installation and basic concepts |
