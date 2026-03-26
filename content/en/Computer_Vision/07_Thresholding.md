# Binarization and Thresholding

## Overview

Binarization is the process of converting a grayscale image into a black and white image. Pixels are classified as either 0 or 255 based on a threshold value. This document covers various thresholding methods and practical application techniques.

**Difficulty**: ⭐⭐ (Beginner-Intermediate)

## Learning Objectives

After completing this lesson, you will be able to:

1. `cv2.threshold()` function and various flags
2. OTSU automatic threshold determination
3. Adaptive Thresholding
4. Multi-level thresholding
5. HSV color-based thresholding
6. Document binarization and shadow handling

---

## Table of Contents

1. [Binarization Overview](#1-binarization-overview)
2. [Global Thresholding - threshold()](#2-global-thresholding---threshold)
3. [OTSU Automatic Threshold](#3-otsu-automatic-threshold)
4. [Adaptive Thresholding - adaptiveThreshold()](#4-adaptive-thresholding---adaptivethreshold)
5. [Multi-level Thresholding](#5-multi-level-thresholding)
6. [HSV Color-based Thresholding](#6-hsv-color-based-thresholding)
7. [Document Binarization and Shadow Handling](#7-document-binarization-and-shadow-handling)
8. [Exercises](#8-exercises)
9. [Next Steps](#9-next-steps)
10. [References](#10-references)

---

## 1. Binarization Overview

### What is Binarization?

```
┌─────────────────────────────────────────────────────────────────┐
│                      Binarization Concept                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Grayscale Image (0-255)         Binary Image (0 or 255)      │
│   ┌─────────────────────┐        ┌─────────────────────┐       │
│   │░░░▒▒▒▓▓▓███████████│  ───▶  │     █████████████████│       │
│   │░░░░▒▒▒▓▓▓██████████│        │     █████████████████│       │
│   │░░░░░▒▒▒▓▓▓█████████│        │     █████████████████│       │
│   └─────────────────────┘        └─────────────────────┘       │
│                                                                 │
│   Based on Threshold (T):                                      │
│   - Pixel value > T → White (255)                              │
│   - Pixel value ≤ T → Black (0)                                │
│                                                                 │
│   Use Cases:                                                    │
│   - Object-background separation                               │
│   - Document scanning                                          │
│   - Preprocessing for contour detection                        │
│   - Mask generation                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Types of Thresholding

```
┌─────────────────────────────────────────────────────────────────┐
│                     Thresholding Types                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Global Thresholding                                          │
│   - Apply single threshold to entire image                     │
│   - Suitable for uniformly lit images                          │
│   - cv2.threshold()                                             │
│                                                                 │
│   Adaptive Thresholding                                        │
│   - Apply different thresholds to different regions            │
│   - Suitable for unevenly lit images                           │
│   - cv2.adaptiveThreshold()                                     │
│                                                                 │
│   Example:                                                      │
│   ┌────────────────┐      ┌────────────────┐                   │
│   │ Bright  Dark   │      │ Bright  Dark   │                   │
│   │  ██      ██    │      │  ██      ██    │                   │
│   │  ██      ██    │      │  ██      ██    │                   │
│   └────────────────┘      └────────────────┘                   │
│   Original with shadow     Global: Partial loss                │
│                           Adaptive: Full detection             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Global Thresholding - threshold()

### Basic Usage

```python
import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# threshold(src, thresh, maxval, type)
# src: Input image (grayscale)
# thresh: Threshold value
# maxval: Maximum value (usually 255)
# type: Thresholding type
# Returns: (threshold used, result image)

ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

print(f"Threshold used: {ret}")

cv2.imshow('Original', img)
cv2.imshow('Binary', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Thresholding Types

```
┌─────────────────────────────────────────────────────────────────┐
│                     Thresholding Types                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input pixel value distribution:                              │
│   ▲                                                            │
│   │     ░░░░░▒▒▒▒▒▓▓▓▓▓███████                                │
│   │     ░░░░░░▒▒▒▒▒▓▓▓▓▓██████                                │
│   └──────────────┬───────────────▶ Pixel value                │
│                  T (Threshold)                                 │
│                                                                 │
│   THRESH_BINARY:          dst = maxval if src > T else 0       │
│   value > T → 255, value ≤ T → 0                              │
│                                                                 │
│   THRESH_BINARY_INV:      dst = 0 if src > T else maxval       │
│   value > T → 0, value ≤ T → 255 (inverted)                   │
│                                                                 │
│   THRESH_TRUNC:           dst = T if src > T else src          │
│   value > T → T, value ≤ T → keep                             │
│                                                                 │
│   THRESH_TOZERO:          dst = src if src > T else 0          │
│   value > T → keep, value ≤ T → 0                             │
│                                                                 │
│   THRESH_TOZERO_INV:      dst = 0 if src > T else src          │
│   value > T → 0, value ≤ T → keep                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Comparison of Type Results

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
thresh = 127

threshold_types = [
    ('BINARY', cv2.THRESH_BINARY),
    ('BINARY_INV', cv2.THRESH_BINARY_INV),
    ('TRUNC', cv2.THRESH_TRUNC),
    ('TOZERO', cv2.THRESH_TOZERO),
    ('TOZERO_INV', cv2.THRESH_TOZERO_INV),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].imshow(img, cmap='gray')
axes[0].set_title(f'Original')

for ax, (name, thresh_type) in zip(axes[1:], threshold_types):
    _, result = cv2.threshold(img, thresh, 255, thresh_type)
    ax.imshow(result, cmap='gray')
    ax.set_title(f'{name}')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
```

### Threshold Selection Guide

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_optimal_threshold(img):
    """Find appropriate threshold through histogram analysis"""
    # Calculate histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.flatten()

    # Test with various thresholds
    thresholds = [64, 96, 127, 160, 192]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Display histogram
    axes[0, 0].plot(hist)
    axes[0, 0].set_title('Histogram')
    axes[0, 0].axvline(x=127, color='r', linestyle='--', label='T=127')
    axes[0, 0].legend()

    # Original
    axes[0, 1].imshow(img, cmap='gray')
    axes[0, 1].set_title('Original')

    # Results with various thresholds
    for ax, t in zip(axes.flatten()[2:], thresholds):
        _, binary = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
        ax.imshow(binary, cmap='gray')
        ax.set_title(f'Threshold = {t}')

    for ax in axes.flatten():
        ax.axis('off')
    axes[0, 0].axis('on')

    plt.tight_layout()
    plt.show()


img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
find_optimal_threshold(img)
```

---

## 3. OTSU Automatic Threshold

Choosing a threshold manually requires inspecting each image, which is impractical for batch processing. Otsu's method solves this by treating threshold selection as an optimization problem: find the value that best separates the histogram into two compact, well-separated clusters. The result is a data-driven threshold that works consistently across images with varying exposure.

### OTSU Algorithm

```
┌─────────────────────────────────────────────────────────────────┐
│                       OTSU Algorithm                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   The OTSU method automatically finds the optimal threshold    │
│   by analyzing the histogram.                                  │
│                                                                 │
│   Principle:                                                    │
│   - Separate histogram into two classes                        │
│   - Maximize between-class variance                            │
│   - Or minimize within-class variance                          │
│                                                                 │
│   Histogram Example:                                            │
│   ▲                                                            │
│   │   ████                    ████                             │
│   │  ██████                 ████████                           │
│   │ ████████               ██████████                          │
│   └────────────────┬───────────────────▶                       │
│                    T (Threshold found by OTSU)                 │
│    Background class     Foreground class                       │
│                                                                 │
│   Suitable for:                                                 │
│   - Bimodal histogram (two peaks)                              │
│   - Clear separation between background and foreground         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### OTSU Usage

```python
import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Why thresh=0 when using OTSU: the value is ignored — OpenCV overwrites it with the
# computed optimal threshold; passing 0 signals intent and avoids confusion
ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ret is the threshold Otsu found; inspecting it tells you about the image contrast
print(f"Threshold determined by OTSU: {ret}")

cv2.imshow('Original', img)
cv2.imshow('OTSU Binary', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### OTSU vs Fixed Threshold Comparison

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('document.jpg', cv2.IMREAD_GRAYSCALE)

# Fixed threshold
_, fixed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# OTSU automatic threshold
ret_otsu, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')

axes[1].imshow(fixed, cmap='gray')
axes[1].set_title('Fixed (T=127)')

axes[2].imshow(otsu, cmap='gray')
axes[2].set_title(f'OTSU (T={ret_otsu:.0f})')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
```

### Gaussian Blur + OTSU (Noise Handling)

```python
import cv2

img = cv2.imread('noisy_image.jpg', cv2.IMREAD_GRAYSCALE)

# Direct OTSU
_, otsu_direct = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Why blur before OTSU: noise creates many tiny histogram spikes that can shift
# Otsu's variance calculation toward a wrong valley; blurring merges these spikes
# back into the two main peaks, making the bimodal structure clearer
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret, otsu_blur = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"OTSU threshold after blur: {ret}")

cv2.imshow('Direct OTSU', otsu_direct)
cv2.imshow('Blur + OTSU', otsu_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 4. Adaptive Thresholding - adaptiveThreshold()

Global thresholding (including Otsu) uses one threshold for the entire image, which fails when illumination is uneven — the shadow side of a document may be darker than the brightest foreground pixels on the bright side. Adaptive thresholding computes a separate threshold for each pixel based on its local neighborhood, making it robust to gradients in lighting and the standard choice for document scanning.

### What is Adaptive Thresholding?

```
┌─────────────────────────────────────────────────────────────────┐
│                   Adaptive Thresholding                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Problem: Unevenly lit image                                  │
│   ┌─────────────────────────────────────────┐                   │
│   │ ████████           ░░░░░░░░             │                   │
│   │ Bright area        Dark area            │                   │
│   │ (with text)        (with text)          │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                 │
│   Global thresholding:                                          │
│   - Process entire image with one threshold                    │
│   - Bright area OK, dark area text lost (or vice versa)        │
│                                                                 │
│   Adaptive thresholding:                                        │
│   - Determine local threshold by analyzing surrounding area    │
│     for each pixel                                             │
│   - Robust to lighting changes                                 │
│                                                                 │
│   ┌─────────────────────────────────────────┐                   │
│   │ Local area 1      Local area 2          │                   │
│   │ T = 200           T = 100               │                   │
│   │ (bright area)     (dark area)           │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Usage

```python
import cv2

img = cv2.imread('document.jpg', cv2.IMREAD_GRAYSCALE)

# adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType,
#                   blockSize, C)
# adaptiveMethod: ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C
# blockSize: Local area size (must be odd — required so there is a single center pixel)
# C: Constant subtracted from calculated mean/weighted mean

# MEAN_C: treats all neighbors equally — fast but can be noisy at edges
adaptive_mean = cv2.adaptiveThreshold(
    img, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    11, 2  # blockSize=11 captures ~11px lighting variation; C=2 prevents noise pixels
           # from flipping to white (background subtraction)
)

# Why GAUSSIAN_C preferred: pixels near the center of the block are more likely to
# share the same illumination as the target pixel; down-weighting distant neighbors
# produces a smoother, less noisy threshold map than MEAN_C
adaptive_gaussian = cv2.adaptiveThreshold(
    img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)

cv2.imshow('Original', img)
cv2.imshow('Adaptive Mean', adaptive_mean)
cv2.imshow('Adaptive Gaussian', adaptive_gaussian)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Parameter Adjustment

```
┌─────────────────────────────────────────────────────────────────┐
│                  adaptiveThreshold Parameters                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   blockSize (local area size):                                 │
│   - Small values (e.g., 3, 5): Preserve fine details,          │
│     sensitive to noise                                         │
│   - Large values (e.g., 31, 51): Smooth results, may lose      │
│     detail                                                     │
│   - Typically use 11 ~ 31                                      │
│                                                                 │
│   C (constant):                                                 │
│   - Value subtracted from calculated threshold                 │
│   - Positive: More pixels become white                         │
│   - Negative: More pixels become black                         │
│   - Typically use 2 ~ 10                                       │
│                                                                 │
│   Threshold calculation:                                        │
│   T(x,y) = mean(blockSize × blockSize area) - C               │
│                                                                 │
│   Geometric intuition: the local mean estimates the background  │
│   brightness around pixel (x,y); subtracting C lowers the bar  │
│   so that only pixels noticeably brighter than their surroundings│
│   (i.e., ink on paper) pass the test                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('document_shadow.jpg', cv2.IMREAD_GRAYSCALE)

# Test various parameter combinations
params = [
    (11, 2),
    (11, 5),
    (21, 2),
    (21, 5),
    (31, 2),
    (31, 10),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for ax, (block_size, c) in zip(axes, params):
    result = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c
    )
    ax.imshow(result, cmap='gray')
    ax.set_title(f'blockSize={block_size}, C={c}')
    ax.axis('off')

plt.tight_layout()
plt.show()
```

### Global vs Adaptive Comparison

```python
import cv2
import matplotlib.pyplot as plt

# Document image with shadow
img = cv2.imread('document_with_shadow.jpg', cv2.IMREAD_GRAYSCALE)

# Global thresholding
_, global_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# OTSU
_, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive
adaptive = cv2.adaptiveThreshold(
    img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    21, 10
)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original')

axes[0, 1].imshow(global_thresh, cmap='gray')
axes[0, 1].set_title('Global (T=127)')

axes[1, 0].imshow(otsu, cmap='gray')
axes[1, 0].set_title('OTSU')

axes[1, 1].imshow(adaptive, cmap='gray')
axes[1, 1].set_title('Adaptive Gaussian')

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## 5. Multi-level Thresholding

### Multi-level Thresholding

```python
import cv2
import numpy as np

def multi_threshold(img, thresholds):
    """
    Multi-level thresholding

    Parameters:
    - img: Grayscale image
    - thresholds: List of threshold values [T1, T2, T3, ...]

    Returns:
    - Labeled image (0, 1, 2, 3, ...)
    """
    result = np.zeros_like(img)
    thresholds = sorted(thresholds)

    for i, t in enumerate(thresholds):
        result[img > t] = (i + 1) * (255 // (len(thresholds)))

    return result


img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 3-level separation (dark, medium, bright)
result = multi_threshold(img, [85, 170])

# 4-level separation
result4 = multi_threshold(img, [64, 128, 192])

cv2.imshow('Original', img)
cv2.imshow('3 Levels', result)
cv2.imshow('4 Levels', result4)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Apply Colormap

```python
import cv2
import numpy as np

def quantize_colors(img, levels=4):
    """Quantize image into n levels"""
    # Calculate step value
    step = 256 // levels
    quantized = (img // step) * step

    return quantized


img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Quantization
quantized = quantize_colors(img, levels=8)

# Apply colormap
colored = cv2.applyColorMap(quantized, cv2.COLORMAP_JET)

cv2.imshow('Original', img)
cv2.imshow('Quantized', quantized)
cv2.imshow('Colored', colored)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 6. HSV Color-based Thresholding

### Color Range Masking

```python
import cv2
import numpy as np

img = cv2.imread('colorful_image.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define blue color range
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])

# Create mask with inRange
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Apply mask
result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('Original', img)
cv2.imshow('Mask', mask)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Real-time Color Range Adjustment

```python
import cv2
import numpy as np

def nothing(x):
    pass

# Create window and trackbars
cv2.namedWindow('Controls')
cv2.createTrackbar('H_Low', 'Controls', 0, 179, nothing)
cv2.createTrackbar('H_High', 'Controls', 179, 179, nothing)
cv2.createTrackbar('S_Low', 'Controls', 0, 255, nothing)
cv2.createTrackbar('S_High', 'Controls', 255, 255, nothing)
cv2.createTrackbar('V_Low', 'Controls', 0, 255, nothing)
cv2.createTrackbar('V_High', 'Controls', 255, 255, nothing)

img = cv2.imread('colorful_image.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

while True:
    h_low = cv2.getTrackbarPos('H_Low', 'Controls')
    h_high = cv2.getTrackbarPos('H_High', 'Controls')
    s_low = cv2.getTrackbarPos('S_Low', 'Controls')
    s_high = cv2.getTrackbarPos('S_High', 'Controls')
    v_low = cv2.getTrackbarPos('V_Low', 'Controls')
    v_high = cv2.getTrackbarPos('V_High', 'Controls')

    lower = np.array([h_low, s_low, v_low])
    upper = np.array([h_high, s_high, v_high])

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('Original', img)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

### Primary Color Ranges

```
┌─────────────────────────────────────────────────────────────────┐
│                    HSV Color Range Guide                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Color        H (Hue)        S (Saturation)   V (Value)       │
│   ───────────────────────────────────────────────────────      │
│   Red          0-10           100-255          100-255         │
│   (wrapping)   160-179        100-255          100-255         │
│                                                                 │
│   Orange       10-25          100-255          100-255         │
│                                                                 │
│   Yellow       25-35          100-255          100-255         │
│                                                                 │
│   Green        35-85          100-255          100-255         │
│                                                                 │
│   Cyan         85-95          100-255          100-255         │
│                                                                 │
│   Blue         95-130         100-255          100-255         │
│                                                                 │
│   Purple       130-160        100-255          100-255         │
│                                                                 │
│   White        0-179          0-30             200-255         │
│                                                                 │
│   Black        0-179          0-255            0-50            │
│                                                                 │
│   Gray         0-179          0-30             50-200          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Document Binarization and Shadow Handling

### Document Binarization Pipeline

```python
import cv2
import numpy as np

def binarize_document(img, method='adaptive'):
    """
    Document image binarization

    Parameters:
    - img: Input image (color or grayscale)
    - method: 'adaptive', 'otsu', 'combined'
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if method == 'otsu':
        # Why blur before OTSU: smoothing collapses noise spikes in the histogram so
        # Otsu finds the true valley between background and foreground peaks
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method == 'adaptive':
        # Why blockSize=21, C=15: a 21px block handles typical shadow gradients in
        # A4 scans; C=15 is aggressive enough to suppress paper texture noise
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21, 15
        )

    elif method == 'combined':
        # Combine OTSU + Adaptive
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu = cv2.threshold(blur, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21, 15
        )

        # Why AND: a pixel is kept only if *both* methods agree it is foreground;
        # this intersection removes false positives that each method produces alone
        binary = cv2.bitwise_and(otsu, adaptive)

    return binary


img = cv2.imread('document.jpg')
binary = binarize_document(img, method='adaptive')
```

### Shadow Removal

```python
import cv2
import numpy as np

def remove_shadow(img):
    """
    Remove shadows from document image
    """
    # Split RGB
    rgb_planes = cv2.split(img)
    result_planes = []

    for plane in rgb_planes:
        # Estimate background with dilation
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))

        # Remove noise with medianBlur
        bg = cv2.medianBlur(dilated, 21)

        # Calculate difference and normalize
        diff = 255 - cv2.absdiff(plane, bg)

        # Enhance contrast
        normalized = cv2.normalize(diff, None, alpha=0, beta=255,
                                    norm_type=cv2.NORM_MINMAX)
        result_planes.append(normalized)

    result = cv2.merge(result_planes)
    return result


def binarize_with_shadow_removal(img):
    """Binarize after shadow removal"""
    # Remove shadow
    no_shadow = remove_shadow(img)

    # Convert to grayscale
    gray = cv2.cvtColor(no_shadow, cv2.COLOR_BGR2GRAY)

    # Adaptive binarization
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 10
    )

    return binary, no_shadow


img = cv2.imread('document_with_shadow.jpg')
binary, no_shadow = binarize_with_shadow_removal(img)

cv2.imshow('Original', img)
cv2.imshow('Shadow Removed', no_shadow)
cv2.imshow('Binary', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Divide Technique (Background Division)

```python
import cv2
import numpy as np

def divide_binarization(img, blur_kernel=21):
    """
    Binarization after correcting uneven illumination with divide technique

    Principle: original / background = uniform image
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Why strong blur (blur_kernel=21): we want the background illumination map,
    # not any text content; the kernel must be large enough that all text is blurred away
    bg = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Why divide: gray/bg normalizes each pixel by local brightness; a dark ink pixel
    # on a shadowed background divides to a low ratio just like on a bright background,
    # making the result illumination-independent
    divided = cv2.divide(gray, bg, scale=255)

    # Why OTSU on the divided image: after division the histogram is reliably bimodal
    # (ink vs paper), so Otsu finds a stable threshold without manual tuning
    _, binary = cv2.threshold(divided, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary, divided


img = cv2.imread('document_uneven_lighting.jpg')
binary, divided = divide_binarization(img)

cv2.imshow('Original', img)
cv2.imshow('Divided', divided)
cv2.imshow('Binary', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 8. Exercises

### Exercise 1: Automatic Optimal Threshold Search

Implement a function that analyzes the histogram to find the optimal threshold between two peaks in a bimodal distribution. Compare with OTSU results.

```python
def find_valley_threshold(img):
    """
    Find the valley between two peaks in the histogram
    and return it as the threshold
    """
    # Hint: Use scipy.signal.find_peaks or
    # Smooth histogram and find minimum
    pass
```

### Exercise 2: Adaptive Thresholding Parameter Tuning GUI

Create a program that allows real-time adjustment of `blockSize` and `C` values using trackbars while viewing the results.

### Exercise 3: Business Card Scanner

Create a program that receives a business card image and performs the following:
1. Shadow/uneven lighting correction
2. Binarization
3. Noise removal (morphological operations)
4. Save result

### Exercise 4: Color Separation Tool

Write a function that extracts specific color regions from an image and calculates the area of the extracted regions. Example: "Red area occupies 15% of total"

### Exercise 5: Hysteresis Thresholding

Implement hysteresis thresholding used in Canny edge detection:
- Above high threshold: Definite edge
- Below low threshold: Definitely not an edge
- In between: Edge only if connected to definite edge

```python
def hysteresis_threshold(img, low_thresh, high_thresh):
    """
    Implement hysteresis thresholding
    """
    pass
```

---

## 9. Next Steps

Learn various edge detection techniques such as Sobel and Canny in [08_Edge_Detection.md](./08_Edge_Detection.md)!

**What you'll learn next**:
- Sobel and Scharr derivative operators
- Laplacian edge detection
- Canny edge detection algorithm
- Edge-based object detection

---

## 10. References

### Official Documentation

- [threshold() documentation](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57)
- [adaptiveThreshold() documentation](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3)
- [inRange() documentation](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981)

### Related Learning Materials

| File | Related Content |
|------|----------|
| [03_Color_Spaces.md](./03_Color_Spaces.md) | HSV color space |
| [06_Morphology.md](./06_Morphology.md) | Noise removal after binarization |

### Additional References

- [OTSU Algorithm Explanation](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
- [Document Binarization Techniques](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_niblack_sauvola.html)
