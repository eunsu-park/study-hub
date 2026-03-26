# Histogram Analysis

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what an image histogram is and how it represents pixel intensity distributions
2. Calculate and visualize histograms for grayscale and multi-channel images using OpenCV
3. Implement histogram equalization and CLAHE to enhance image contrast
4. Compare histograms using different distance metrics to measure image similarity
5. Apply histogram backprojection to locate regions of interest based on color distributions
6. Analyze histogram shape to diagnose image quality issues such as overexposure and low contrast

---

## Overview

A histogram is a graph representing the brightness distribution of an image. It is used for image analysis, contrast enhancement, color comparison, and more. In this lesson, we will learn about histogram calculation, equalization, CLAHE, comparison, backprojection, and other techniques.

---

## Table of Contents

1. [Histogram Basics](#1-histogram-basics)
2. [Histogram Calculation](#2-histogram-calculation)
3. [Histogram Equalization](#3-histogram-equalization)
4. [CLAHE](#4-clahe)
5. [Histogram Comparison](#5-histogram-comparison)
6. [Backprojection](#6-backprojection)
7. [Practice Problems](#7-practice-problems)

---

## 1. Histogram Basics

### What is a Histogram?

A histogram reduces an entire image to a compact statistical summary of its intensity distribution, independent of spatial layout. This makes it a powerful "fingerprint" for an image: two photos of the same scene under the same lighting will share very similar histograms even if they are slightly shifted or rotated, while a drastically different scene will have a distinct histogram shape. This property is what makes histograms useful for rapid image retrieval, exposure diagnosis, and color-based object tracking.

```
Histogram:
A graph representing the distribution of pixel brightness values in an image

X-axis: Brightness value (0-255)
Y-axis: Number of pixels with that brightness value

Dark Image                Bright Image            High Contrast Image
    │                        │                      │
Freq│█                       │       █              │   █   █
uenc│██                      │      ██              │  ███ ███
y   │███                     │     ███              │ █████████
    └────────────           └────────────          └────────────
    0          255          0          255         0          255
     Brightness               Brightness              Brightness
```

### Applications of Histograms

```
1. Image Analysis
   - Check exposure status (overexposed, underexposed)
   - Assess contrast level

2. Image Enhancement
   - Histogram equalization
   - Contrast adjustment

3. Image Comparison
   - Similar image search
   - Color-based matching

4. Object Tracking
   - Color histogram backprojection
   - CamShift/MeanShift algorithms
```

---

## 2. Histogram Calculation

### cv2.calcHist() Function

```python
hist = cv2.calcHist(images, channels, mask, histSize, ranges)
```

| Parameter | Description |
|-----------|-------------|
| images | List of input images [img] |
| channels | Channel indices [0], [1], [2] or [0, 1], etc. |
| mask | Mask (None = entire image) |
| histSize | Number of bins [256] |
| ranges | Value range [0, 256] |

### Grayscale Histogram

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calc_gray_histogram(image_path):
    """Calculate and visualize grayscale histogram"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate histogram
    hist = cv2.calcHist(
        [img],           # Image (passed as list)
        [0],             # Channel (0 for grayscale)
        None,            # Mask (entire image)
        [256],           # 256 bins — one per possible 8-bit intensity level (0–255).
                         # This gives maximum precision. Fewer bins (e.g., 64) would
                         # merge adjacent intensities, speeding up comparison at the
                         # cost of discriminative power; useful for retrieval at scale.
        [0, 256]         # Value range (upper bound is exclusive, so this covers 0–255)
    )

    # Visualize with Matplotlib
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.plot(hist, color='black')
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()

    return hist

hist = calc_gray_histogram('image.jpg')
```

### Color Histogram

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calc_color_histogram(image_path):
    """RGB channel-wise histogram"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    colors = ('r', 'g', 'b')
    channel_names = ('Red', 'Green', 'Blue')

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    for i, (color, name) in enumerate(zip(colors, channel_names)):
        # BGR order, so adjust index: R=2, G=1, B=0
        channel_idx = 2 - i
        hist = cv2.calcHist([img], [channel_idx], None, [256], [0, 256])
        plt.plot(hist, color=color, label=name)

    plt.title('Color Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.legend()

    plt.tight_layout()
    plt.show()

calc_color_histogram('colorful.jpg')
```

### 2D Histogram (Hue-Saturation)

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calc_2d_histogram(image_path):
    """Hue-Saturation 2D histogram"""
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # H: 0-180, S: 0-256
    hist = cv2.calcHist(
        [hsv],
        [0, 1],          # H and S channels
        None,
        [30, 32],        # Number of bins (H: 30, S: 32)
        [0, 180, 0, 256] # Ranges (H: 0-180, S: 0-256)
    )

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(hist, interpolation='nearest')
    plt.title('2D Histogram (H-S)')
    plt.xlabel('Saturation')
    plt.ylabel('Hue')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    return hist

hist_2d = calc_2d_histogram('colorful.jpg')
```

### Histogram with Mask

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_with_mask(image_path):
    """Calculate histogram for specific region only"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape

    # Create circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w//2, h//2), min(h, w)//3, 255, -1)

    # Full histogram
    hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Masked region histogram
    hist_masked = cv2.calcHist([img], [0], mask, [256], [0, 256])

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.plot(hist_full, label='Full', alpha=0.7)
    plt.plot(hist_masked, label='Masked', alpha=0.7)
    plt.legend()
    plt.title('Histograms')

    plt.tight_layout()
    plt.show()

histogram_with_mask('image.jpg')
```

---

## 3. Histogram Equalization

### Concept

```
Histogram Equalization:
Makes the brightness distribution of an image uniform to enhance contrast

Original Histogram            Equalized Histogram
    │                              │
    │█                             │   █ █ █
    │███                           │ █ █ █ █ █
    │█████                         │█ █ █ █ █ █ █
    └────────────                  └────────────────
    0          255                 0              255

Transformation Process:
1. Calculate histogram
2. Calculate cumulative distribution function (CDF)
3. Normalize CDF
4. Map pixel values
```

The CDF mapping works on a simple geometric principle: if you stretch an image's intensity range so that the cumulative count of pixels is evenly spread across 0–255, the output histogram becomes as flat (uniform) as possible. Mathematically, the mapping is `out = round(CDF(in) × 255)`. A pixel value where many pixels are clustered (a tall histogram spike) gets spread over a wide output range, separating formerly indistinguishable gray levels and revealing hidden detail.

### cv2.equalizeHist()

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalize_histogram_demo(image_path):
    """Histogram equalization demo"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Histogram equalization
    equalized = cv2.equalizeHist(img)

    # Calculate histograms
    hist_before = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_after = cv2.calcHist([equalized], [0], None, [256], [0, 256])

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(equalized, cmap='gray')
    axes[0, 1].set_title('Equalized')
    axes[0, 1].axis('off')

    axes[1, 0].plot(hist_before)
    axes[1, 0].set_title('Original Histogram')
    axes[1, 0].set_xlim([0, 256])

    axes[1, 1].plot(hist_after)
    axes[1, 1].set_title('Equalized Histogram')
    axes[1, 1].set_xlim([0, 256])

    plt.tight_layout()
    plt.show()

    return equalized

equalized = equalize_histogram_demo('dark_image.jpg')
```

### Color Image Equalization

```python
import cv2
import numpy as np

def equalize_color_image(image_path):
    """Histogram equalization for color images"""
    img = cv2.imread(image_path)

    # Method 1: Use YCrCb color space (recommended)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])  # Equalize Y channel only
    result_ycrcb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Method 2: Use HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])  # Equalize V channel only
    result_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Method 3: Equalize each channel individually (may cause color distortion)
    b, g, r = cv2.split(img)
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)
    result_rgb = cv2.merge([b_eq, g_eq, r_eq])

    cv2.imshow('Original', img)
    cv2.imshow('YCrCb Equalization', result_ycrcb)
    cv2.imshow('HSV Equalization', result_hsv)
    cv2.imshow('RGB Equalization', result_rgb)
    cv2.waitKey(0)

    return result_ycrcb

equalize_color_image('dark_color.jpg')
```

---

## 4. CLAHE

### Concept

```
CLAHE (Contrast Limited Adaptive Histogram Equalization):
Adaptive histogram equalization

Problem: Global equalization can amplify noise
Solution: Divide image into tiles and equalize locally

┌────┬────┬────┬────┐
│    │    │    │    │
│ T1 │ T2 │ T3 │ T4 │   Apply equalization
├────┼────┼────┼────┤   to each tile
│    │    │    │    │
│ T5 │ T6 │ T7 │ T8 │   Smooth boundaries
├────┼────┼────┼────┤   with interpolation
│ T9 │T10 │T11 │T12 │
└────┴────┴────┴────┘

Features:
- clipLimit: Contrast limit (higher = stronger contrast)
- tileGridSize: Tile size (smaller = more detailed)
```

### cv2.createCLAHE()

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def clahe_demo(image_path):
    """CLAHE application demo"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Standard equalization
    equalized = cv2.equalizeHist(img)

    # Create and apply CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,      # Contrast amplification cap per tile. When any histogram
                            # bin would exceed this limit after equalization, the excess
                            # votes are redistributed uniformly — this prevents runaway
                            # noise amplification in flat (low-texture) regions.
                            # 2.0 is a conservative default; raise to 4–8 for very dark
                            # medical images, but expect more noise at high values.
        tileGridSize=(8, 8) # Divides the image into an 8×8 grid of tiles; equalization
                            # is applied independently within each tile, then boundaries
                            # are blended with bilinear interpolation. Smaller tiles
                            # (e.g., 4×4) enhance local detail more aggressively;
                            # larger tiles (16×16) behave closer to global equalization.
    )
    clahe_result = clahe.apply(img)

    # Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(equalized, cmap='gray')
    axes[1].set_title('Standard Equalization')
    axes[1].axis('off')

    axes[2].imshow(clahe_result, cmap='gray')
    axes[2].set_title('CLAHE')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return clahe_result

clahe_demo('low_contrast.jpg')
```

### CLAHE Parameter Comparison

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_clahe_params(image_path):
    """Compare CLAHE with different parameters"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    clip_limits = [1.0, 2.0, 4.0, 8.0]
    tile_sizes = [(4, 4), (8, 8), (16, 16)]

    fig, axes = plt.subplots(len(tile_sizes), len(clip_limits) + 1,
                              figsize=(15, 10))

    for i, tile_size in enumerate(tile_sizes):
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'Original\nTile: {tile_size}')
        axes[i, 0].axis('off')

        for j, clip_limit in enumerate(clip_limits):
            clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                     tileGridSize=tile_size)
            result = clahe.apply(img)

            axes[i, j + 1].imshow(result, cmap='gray')
            axes[i, j + 1].set_title(f'clip={clip_limit}')
            axes[i, j + 1].axis('off')

    plt.tight_layout()
    plt.show()

compare_clahe_params('low_contrast.jpg')
```

### Applying CLAHE to Color Images

```python
import cv2
import numpy as np

def clahe_color(image_path, clip_limit=2.0, tile_size=(8, 8)):
    """Apply CLAHE to color image"""
    img = cv2.imread(image_path)

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convert back to BGR
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    cv2.imshow('Original', img)
    cv2.imshow('CLAHE', result)
    cv2.waitKey(0)

    return result

clahe_color('dark_scene.jpg')
```

---

## 5. Histogram Comparison

### cv2.compareHist()

```python
similarity = cv2.compareHist(hist1, hist2, method)
```

| Method | Description | Range | Interpretation |
|--------|-------------|-------|----------------|
| cv2.HISTCMP_CORREL | Correlation | -1 ~ 1 | 1: Perfect match |
| cv2.HISTCMP_CHISQR | Chi-Square | 0 ~ ∞ | 0: Perfect match |
| cv2.HISTCMP_INTERSECT | Intersection | 0 ~ min(sum) | Higher = more similar |
| cv2.HISTCMP_BHATTACHARYYA | Bhattacharyya distance | 0 ~ 1 | 0: Perfect match |

### Histogram Comparison Example

```python
import cv2
import numpy as np

def compare_histograms(image_paths):
    """Compare histograms of multiple images"""
    # Base image
    base_img = cv2.imread(image_paths[0])
    base_hsv = cv2.cvtColor(base_img, cv2.COLOR_BGR2HSV)

    # Calculate histogram (H-S 2D)
    base_hist = cv2.calcHist(
        [base_hsv], [0, 1], None,
        [50, 60], [0, 180, 0, 256]
    )
    cv2.normalize(base_hist, base_hist, 0, 1, cv2.NORM_MINMAX)

    print(f"Base image: {image_paths[0]}")
    print("-" * 50)

    methods = [
        (cv2.HISTCMP_CORREL, 'Correlation'),
        (cv2.HISTCMP_CHISQR, 'Chi-Square'),
        (cv2.HISTCMP_INTERSECT, 'Intersection'),
        (cv2.HISTCMP_BHATTACHARYYA, 'Bhattacharyya')
    ]

    for path in image_paths[1:]:
        img = cv2.imread(path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [50, 60], [0, 180, 0, 256]
        )
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        print(f"\nComparing: {path}")
        for method, name in methods:
            result = cv2.compareHist(base_hist, hist, method)
            print(f"  {name}: {result:.4f}")

# Usage example
image_files = ['ref.jpg', 'similar1.jpg', 'similar2.jpg', 'different.jpg']
compare_histograms(image_files)
```

### Similar Image Search

```python
import cv2
import numpy as np
import os

def find_similar_images(query_path, search_dir, top_k=5):
    """Histogram-based similar image search"""
    # Query image histogram
    query = cv2.imread(query_path)
    query_hsv = cv2.cvtColor(query, cv2.COLOR_BGR2HSV)
    query_hist = cv2.calcHist([query_hsv], [0, 1], None,
                               [50, 60], [0, 180, 0, 256])
    cv2.normalize(query_hist, query_hist, 0, 1, cv2.NORM_MINMAX)

    results = []

    # Compare with all images in search directory
    for filename in os.listdir(search_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        filepath = os.path.join(search_dir, filename)
        img = cv2.imread(filepath)
        if img is None:
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None,
                             [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        # Calculate correlation (higher = more similar)
        similarity = cv2.compareHist(query_hist, hist, cv2.HISTCMP_CORREL)
        results.append((filename, similarity))

    # Sort by similarity
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"Query: {query_path}")
    print(f"\nTop {top_k} similar images:")
    for filename, sim in results[:top_k]:
        print(f"  {filename}: {sim:.4f}")

    return results[:top_k]

# Usage example
find_similar_images('query.jpg', './image_database/', top_k=5)
```

---

## 6. Backprojection

### Concept

```
Backprojection:
Detect specific color regions using histograms

Process:
1. Calculate color histogram of object of interest (ROI)
2. Replace each pixel in the entire image with its histogram value
3. High value = similar to color of interest

Applications:
- Color-based object tracking
- Core of CamShift/MeanShift algorithms

Example:
┌─────────────┐       ┌─────────────┐
│   🟡 ROI    │       │ ■ ■ □ □ □ │
│  (Yellow)   │  ──▶  │ ■ ■ ■ □ □ │  High value = Yellow
│             │       │ □ ■ ■ ■ □ │
└─────────────┘       └─────────────┘
  Color Histogram      Backprojection Result
```

### cv2.calcBackProject()

```python
import cv2
import numpy as np

def backprojection_demo(image_path, roi_coords):
    """Backprojection demo"""
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Set ROI region
    x, y, w, h = roi_coords
    roi = hsv[y:y+h, x:x+w]

    # Calculate ROI histogram
    roi_hist = cv2.calcHist(
        [roi], [0, 1], None,
        [180, 256], [0, 180, 0, 256]
    )
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Backprojection
    backproj = cv2.calcBackProject(
        [hsv], [0, 1], roi_hist,
        [0, 180, 0, 256], 1
    )

    # Remove noise with filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(backproj, -1, kernel, backproj)
    _, backproj = cv2.threshold(backproj, 50, 255, cv2.THRESH_BINARY)

    # Visualization
    result = img.copy()
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mask detected region
    mask = cv2.merge([backproj, backproj, backproj])
    detected = cv2.bitwise_and(img, mask)

    cv2.imshow('Original with ROI', result)
    cv2.imshow('Back Projection', backproj)
    cv2.imshow('Detected', detected)
    cv2.waitKey(0)

    return backproj

# Usage example (x, y, width, height)
backprojection_demo('scene.jpg', (100, 100, 50, 50))
```

### Skin Color Detection

```python
import cv2
import numpy as np

def detect_skin(image_path):
    """Skin color detection (using backprojection)"""
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Skin color range (HSV)
    # H: 0-20, S: 48-255, V: 80-255 (typical skin color)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Skin color mask
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Generate histogram of skin region
    skin_region = cv2.bitwise_and(hsv, hsv, mask=skin_mask)
    skin_hist = cv2.calcHist([skin_region], [0, 1], skin_mask,
                              [180, 256], [0, 180, 0, 256])
    cv2.normalize(skin_hist, skin_hist, 0, 255, cv2.NORM_MINMAX)

    # Backprojection
    backproj = cv2.calcBackProject([hsv], [0, 1], skin_hist,
                                    [0, 180, 0, 256], 1)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    backproj = cv2.morphologyEx(backproj, cv2.MORPH_OPEN, kernel)
    backproj = cv2.morphologyEx(backproj, cv2.MORPH_CLOSE, kernel)

    # Result
    result = cv2.bitwise_and(img, img, mask=backproj)

    cv2.imshow('Original', img)
    cv2.imshow('Skin Mask', backproj)
    cv2.imshow('Detected Skin', result)
    cv2.waitKey(0)

    return backproj

detect_skin('person.jpg')
```

### Object Tracking with CamShift

```python
import cv2
import numpy as np

def camshift_tracking(video_path):
    """Object tracking using CamShift"""
    cap = cv2.VideoCapture(video_path)

    # Select ROI from first frame
    ret, frame = cap.read()
    if not ret:
        return

    # Select ROI (select with mouse or specify directly)
    roi = cv2.selectROI('Select ROI', frame, False)
    cv2.destroyWindow('Select ROI')

    x, y, w, h = roi
    track_window = (x, y, w, h)

    # Calculate ROI histogram
    roi_frame = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]),
                       np.array([180, 255, 255]))

    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # CamShift termination criteria
    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Backprojection
        backproj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Apply CamShift
        ret, track_window = cv2.CamShift(backproj, track_window, term_criteria)

        # Draw result (rotated rectangle)
        pts = cv2.boxPoints(ret)
        pts = np.int_(pts)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        cv2.imshow('CamShift Tracking', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# camshift_tracking('video.mp4')
```

---

## 7. Practice Problems

### Problem 1: Automatic Contrast Adjustment

Analyze the histogram of an image and automatically perform optimal contrast adjustment.

<details>
<summary>Solution Code</summary>

```python
import cv2
import numpy as np

def auto_contrast(image):
    """Automatic contrast adjustment (histogram stretching)"""
    if len(image.shape) == 3:
        # Color image: LAB conversion
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Histogram stretching on L channel
        l_min = np.min(l)
        l_max = np.max(l)
        l_stretched = ((l - l_min) * 255 / (l_max - l_min)).astype(np.uint8)

        lab_stretched = cv2.merge([l_stretched, a, b])
        result = cv2.cvtColor(lab_stretched, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale
        img_min = np.min(image)
        img_max = np.max(image)
        result = ((image - img_min) * 255 / (img_max - img_min)).astype(np.uint8)

    return result

# Test
img = cv2.imread('low_contrast.jpg')
result = auto_contrast(img)
cv2.imshow('Original', img)
cv2.imshow('Auto Contrast', result)
cv2.waitKey(0)
```

</details>

### Problem 2: Color Distribution Analysis

Extract the top 3 dominant colors from an image.

<details>
<summary>Solution Code</summary>

```python
import cv2
import numpy as np
from collections import Counter

def find_dominant_colors(image, k=3):
    """Extract dominant colors using K-means"""
    # Convert image to 1D array
    pixels = image.reshape(-1, 3).astype(np.float32)

    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Count pixels in each cluster
    label_counts = Counter(labels.flatten())

    # Return colors and ratios
    colors = []
    total = len(labels)
    for idx, count in label_counts.most_common(k):
        color = centers[idx].astype(int)
        percentage = count / total * 100
        colors.append((color, percentage))

    # Visualize results
    result = np.zeros((100, 300, 3), dtype=np.uint8)
    x = 0
    for color, pct in colors:
        width = int(pct * 3)
        result[:, x:x+width] = color
        x += width
        print(f"BGR: {color}, Ratio: {pct:.1f}%")

    cv2.imshow('Dominant Colors', result)
    cv2.waitKey(0)

    return colors

# Test
img = cv2.imread('colorful.jpg')
colors = find_dominant_colors(img, k=5)
```

</details>

### Problem 3: Illumination Normalization

Normalize a document image with uneven illumination.

<details>
<summary>Solution Code</summary>

```python
import cv2
import numpy as np

def normalize_illumination(image):
    """Illumination normalization"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Estimate background (large blur)
    background = cv2.GaussianBlur(gray, (101, 101), 0)

    # Remove background (original / background)
    normalized = cv2.divide(gray, background, scale=255)

    # Apply additional CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    cv2.imshow('Original', gray)
    cv2.imshow('Background', background)
    cv2.imshow('Normalized', normalized)
    cv2.imshow('Enhanced', enhanced)
    cv2.waitKey(0)

    return enhanced

# Test
img = cv2.imread('uneven_document.jpg')
result = normalize_illumination(img)
```

</details>

### Recommended Problems

| Difficulty | Topic | Description |
|------------|-------|-------------|
| ⭐ | Histogram Plotting | Visualize RGB channel histograms |
| ⭐⭐ | Contrast Enhancement | Compare equalizeHist vs CLAHE |
| ⭐⭐ | Image Similarity | Find similar images using histograms |
| ⭐⭐⭐ | Object Tracking | Track colored objects with CamShift |
| ⭐⭐⭐ | HDR Tone Mapping | Merge multi-exposure images |

---

## Next Steps

- [13_Feature_Detection.md](./13_Feature_Detection.md) - Harris, FAST, SIFT, ORB

---

## References

- [OpenCV Histograms](https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html)
- [Histogram Equalization](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)
- [Histogram Backprojection](https://docs.opencv.org/4.x/dc/df6/tutorial_py_histogram_backprojection.html)
