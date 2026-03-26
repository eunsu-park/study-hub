# Edge Detection

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the concept of image gradients and how they represent intensity changes in an image
2. Implement edge detection using Sobel, Scharr, and Laplacian operators with OpenCV
3. Apply the Canny edge detection algorithm and tune its hysteresis thresholds
4. Compare the strengths and limitations of first-order versus second-order derivative operators
5. Analyze gradient magnitude and direction to characterize edge properties
6. Design a preprocessing pipeline that selects appropriate edge detection methods for different image types

---

## Overview

An edge is a region in an image where brightness changes rapidly, representing object boundaries or structures. This lesson covers the concept of image gradients and various edge detection techniques including Sobel, Scharr, Laplacian, and Canny.

---

## Table of Contents

1. [Image Gradient Concept](#1-image-gradient-concept)
2. [Sobel Operator](#2-sobel-operator)
3. [Scharr Operator](#3-scharr-operator)
4. [Laplacian Operator](#4-laplacian-operator)
5. [Canny Edge Detection](#5-canny-edge-detection)
6. [Gradient Magnitude and Direction](#6-gradient-magnitude-and-direction)
7. [Exercises](#7-exercises)

---

## 1. Image Gradient Concept

### What is Gradient?

```
Gradient: Rate of change in image brightness

Mathematical Definition:
∇f = (∂f/∂x, ∂f/∂y)

- ∂f/∂x: Rate of change in x direction (horizontal)
- ∂f/∂y: Rate of change in y direction (vertical)

Gradient Magnitude:
|∇f| = √((∂f/∂x)² + (∂f/∂y)²)

Gradient Direction:
θ = arctan(∂f/∂y / ∂f/∂x)
```

The gradient vector (∂f/∂x, ∂f/∂y) always points in the direction of steepest brightness increase, like water flowing uphill. Its magnitude |∇f| tells you how sharp the edge is; its direction θ tells you which way the brightness rises — perpendicular to the edge boundary itself. For example, a vertical edge (left side dark, right side bright) produces a large ∂f/∂x and near-zero ∂f/∂y, so θ ≈ 0° and the gradient points horizontally.

### Types of Edges

```
1. Step Edge
   Brightness ──┐
                │
                └── Brightness
   → Ideal edge, abrupt change

2. Ramp Edge
   Brightness ──╲
                 ╲
                  ╲── Brightness
   → Gradual change, blurred boundary

3. Roof Edge
   Brightness ──╱╲
               ╱  ╲
              ╱    ╲── Brightness
   → Line structure

4. Ridge Edge
          ╱╲
         ╱  ╲
      ──╱    ╲──
   → Thin line structure
```

### Edge Detection Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Input    │     │    Noise    │     │  Gradient   │     │    Edge     │
│    Image    │ ──▶ │   Removal   │ ──▶ │ Calculation │ ──▶ │ Extraction  │
│             │     │  (Gaussian) │     │ (Sobel etc) │     │ (Threshold) │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## 2. Sobel Operator

### Concept

```
Sobel Operator: First derivative-based edge detection
→ Calculate gradients in x and y directions separately

3x3 Sobel Kernels:

Gx (Horizontal edge detection):   Gy (Vertical edge detection):
┌────┬────┬────┐                  ┌────┬────┬────┐
│ -1 │  0 │ +1 │                  │ -1 │ -2 │ -1 │
├────┼────┼────┤                  ├────┼────┼────┤
│ -2 │  0 │ +2 │                  │  0 │  0 │  0 │
├────┼────┼────┤                  ├────┼────┼────┤
│ -1 │  0 │ +1 │                  │ +1 │ +2 │ +1 │
└────┴────┴────┘                  └────┴────┴────┘

→ Gx: Detect vertical edges (left-right brightness difference)
→ Gy: Detect horizontal edges (top-bottom brightness difference)
```

### cv2.Sobel() Function

```python
cv2.Sobel(src, ddepth, dx, dy, ksize=3, scale=1, delta=0)
```

| Parameter | Description |
|----------|------|
| src | Input image |
| ddepth | Output image depth (cv2.CV_64F recommended) |
| dx | Derivative order in x direction (0 or 1) |
| dy | Derivative order in y direction (0 or 1) |
| ksize | Kernel size (1, 3, 5, 7) |
| scale | Scale factor |
| delta | Value added to result |

### Basic Usage

```python
import cv2
import numpy as np

# Read image
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel operation
# CV_64F (float64) is required because gradients can be negative —
# a dark-to-bright transition gives a positive value, bright-to-dark gives negative.
# Using uint8 would silently clip all negative values to 0, missing half the edges.
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # x direction
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # y direction

# Convert to absolute value and then to 8-bit
# We take the absolute value so that both directions of contrast
# (bright→dark and dark→bright) map to the same edge strength.
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

# Combine x, y gradients
# Equal weighting (0.5 each) avoids overflow while preserving both edge orientations.
sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# Display results
cv2.imshow('Original', img)
cv2.imshow('Sobel X', sobel_x)
cv2.imshow('Sobel Y', sobel_y)
cv2.imshow('Sobel Combined', sobel_combined)
cv2.waitKey(0)
```

### Calculate Gradient Magnitude

```python
import cv2
import numpy as np

def sobel_magnitude(image):
    """Calculate Sobel gradient magnitude"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Gaussian blur before Sobel: the derivative operator amplifies noise
    # (differentiation is a high-pass filter), so smoothing first is essential
    # to distinguish real edges from noise spikes.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Sobel operation (calculate in float64)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Gradient magnitude: sqrt(Gx² + Gy²)
    # This is the Euclidean length of the gradient vector (Gx, Gy), representing
    # the steepness of the brightness ramp at each pixel — large at sharp edges.
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize to 0-255 range
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    return magnitude

# Usage example
img = cv2.imread('image.jpg')
edges = sobel_magnitude(img)
cv2.imshow('Sobel Magnitude', edges)
cv2.waitKey(0)
```

### Differences by Kernel Size

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_sobel_ksize(image_path):
    """Compare Sobel kernel sizes"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    ksizes = [1, 3, 5, 7]

    for ax, ksize in zip(axes.flatten(), ksizes):
        # When ksize=1, use 3x1 or 1x3 filter
        if ksize == 1:
            sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
            sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)
        else:
            sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
            sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)

        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

        ax.imshow(magnitude, cmap='gray')
        ax.set_title(f'Sobel ksize={ksize}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# ksize comparison:
# - ksize=1: Most sensitive, vulnerable to noise
# - ksize=3: Standard, balanced results
# - ksize=5, 7: Smoother edges, more robust to noise
```

---

## 3. Scharr Operator

### Concept

```
Scharr Operator: More accurate 3x3 kernel than Sobel
→ Better rotational symmetry

Scharr Kernels:

Gx:                         Gy:
┌────┬────┬────┐           ┌────┬────┬────┐
│ -3 │  0 │ +3 │           │ -3 │-10 │ -3 │
├────┼────┼────┤           ├────┼────┼────┤
│-10 │  0 │+10 │           │  0 │  0 │  0 │
├────┼────┼────┤           ├────┼────┼────┤
│ -3 │  0 │ +3 │           │ +3 │+10 │ +3 │
└────┴────┴────┘           └────┴────┴────┘

Sobel vs Scharr:
- Sobel: [-1, 0, 1] × [-1, -2, -1]ᵀ
- Scharr: [-3, 0, 3] × [-3, -10, -3]ᵀ
→ Scharr is more accurate in diagonal directions
```

### cv2.Scharr() Function

```python
cv2.Scharr(src, ddepth, dx, dy, scale=1, delta=0)
```

```python
import cv2
import numpy as np

def compare_sobel_scharr(image):
    """Compare Sobel and Scharr"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sobel (ksize=3)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # Scharr (fixed 3x3)
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    scharr_mag = np.sqrt(scharr_x**2 + scharr_y**2)

    # Normalize
    sobel_mag = np.clip(sobel_mag, 0, 255).astype(np.uint8)
    scharr_mag = np.clip(scharr_mag, 0, 255).astype(np.uint8)

    return sobel_mag, scharr_mag

# Scharr usage example
img = cv2.imread('image.jpg')
sobel, scharr = compare_sobel_scharr(img)

cv2.imshow('Sobel', sobel)
cv2.imshow('Scharr', scharr)
cv2.waitKey(0)
```

### Using Scharr with Sobel

```python
# Use ksize=-1 or ksize=cv2.FILTER_SCHARR in cv2.Sobel()
scharr_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=-1)  # Use Scharr kernel
scharr_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=-1)

# Above code is equivalent to
scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
```

---

## 4. Laplacian Operator

### Concept

```
Laplacian Operator: Second derivative-based edge detection
→ Zero-crossing at points where brightness changes rapidly

Mathematical Definition:
∇²f = ∂²f/∂x² + ∂²f/∂y²

Laplacian Kernels:

4-connectivity:             8-connectivity:
┌────┬────┬────┐           ┌────┬────┬────┐
│  0 │  1 │  0 │           │  1 │  1 │  1 │
├────┼────┼────┤           ├────┼────┼────┤
│  1 │ -4 │  1 │           │  1 │ -8 │  1 │
├────┼────┼────┤           ├────┼────┼────┤
│  0 │  1 │  0 │           │  1 │  1 │  1 │
└────┴────┴────┘           └────┴────┴────┘

Characteristics:
- Detects edges regardless of direction
- Very sensitive to noise (second derivative)
- Zero-crossing points are edges
```

### First Derivative vs Second Derivative

```
Original Signal (Step Edge):
       ────────────┐
                   │
                   └────────────

First Derivative (Sobel):
                  ╱╲
                 ╱  ╲
       ─────────╱    ╲─────────
       → Peak point is edge

Second Derivative (Laplacian):
            ╱╲
           ╱  ╲
       ───╱    ╲───
              ╱  ╲
             ╱    ╲
       → Zero-crossing point is edge
```

### cv2.Laplacian() Function

```python
cv2.Laplacian(src, ddepth, ksize=1, scale=1, delta=0)
```

| Parameter | Description |
|----------|------|
| src | Input image |
| ddepth | Output image depth |
| ksize | Kernel size (1, 3, 5, 7) |
| scale | Scale factor |
| delta | Value added to result |

### Basic Usage

```python
import cv2
import numpy as np

def laplacian_edge(image):
    """Laplacian edge detection"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Remove noise (Laplacian is sensitive to noise)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Laplacian operation
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)

    # Convert to absolute value
    laplacian = cv2.convertScaleAbs(laplacian)

    return laplacian

# Usage example
img = cv2.imread('image.jpg')
edges = laplacian_edge(img)
cv2.imshow('Laplacian', edges)
cv2.waitKey(0)
```

### LoG (Laplacian of Gaussian)

```python
import cv2
import numpy as np

def log_edge_detection(image, sigma=1.0):
    """
    LoG (Laplacian of Gaussian) edge detection
    1. Remove noise with Gaussian blur
    2. Detect edges with Laplacian
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur (kernel size based on sigma)
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1

    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)

    # Laplacian
    log = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)

    # Absolute value
    log = cv2.convertScaleAbs(log)

    return log

# Use LoG
img = cv2.imread('image.jpg')
edges = log_edge_detection(img, sigma=1.5)
cv2.imshow('LoG', edges)
cv2.waitKey(0)
```

---

## 5. Canny Edge Detection

### Concept

```
Canny Edge Detection: Multi-stage edge detection algorithm
→ Most widely used edge detection method

Canny's 3 Goals:
1. Low error rate: Detect only real edges
2. Accurate localization: Edges at precise locations
3. Single response: One line for one edge

4-Stage Processing:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Gaussian   │     │    Sobel    │     │     Non-    │     │  Hysteresis │
│    Blur     │ ──▶ │  Gradient   │ ──▶ │   Maximum   │ ──▶ │ Thresholding│
│             │     │             │     │ Suppression │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### Canny Algorithm Details

```
Step 1: Noise Removal (Gaussian Blur)
- Apply 5x5 Gaussian filter
- Remove high-frequency noise

Step 2: Gradient Calculation
- Calculate Gx, Gy with Sobel operation
- Magnitude: G = √(Gx² + Gy²)
- Direction: θ = arctan(Gy/Gx)

Step 3: Non-Maximum Suppression (NMS)
┌─────────────────────────────────────┐
│  Keep only maximum values along     │
│  gradient direction                 │
│  → Make edges 1 pixel thin          │
└─────────────────────────────────────┘

Direction Quantization (4 directions):
        90°
         │
  135° ──┼── 45°
         │
        0° (180°)

Example:
When direction θ = 45°, compare along diagonal
┌───┬───┬───┐
│   │ q │   │
├───┼───┼───┤
│   │ p │   │  Keep p if p > q and p > r
├───┼───┼───┤
│   │ r │   │
└───┴───┴───┘

Step 4: Hysteresis Thresholding
┌─────────────────────────────────────┐
│  high_threshold: Strong edges       │
│  low_threshold: Weak edges          │
│                                     │
│  Strong edges: Always include       │
│  Weak edges: Include if connected   │
│                to strong edge       │
│  Others: Remove                     │
└─────────────────────────────────────┘

Example:
high = 100, low = 50

Pixel value 120 → Strong edge (include)
Pixel value 70  → Weak edge (check connection)
Pixel value 30  → Remove
```

### cv2.Canny() Function

```python
cv2.Canny(image, threshold1, threshold2, apertureSize=3, L2gradient=False)
```

| Parameter | Description |
|----------|------|
| image | Input image (grayscale) |
| threshold1 | Low threshold |
| threshold2 | High threshold |
| apertureSize | Sobel kernel size (3, 5, 7) |
| L2gradient | True: L2 norm, False: L1 norm |

### Basic Usage

```python
import cv2

def canny_edge(image, low=50, high=150):
    """Canny edge detection"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Pre-blurring before Canny is optional but recommended:
    # Canny's internal Gaussian (apertureSize-derived) is fixed, while this
    # external blur lets you control smoothing scale independently of edge precision.
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Hysteresis thresholding uses two thresholds rather than one to solve the
    # "weak edge" problem: a single threshold either breaks continuous edges
    # (too high) or includes noise (too low). High marks definite edges;
    # low admits uncertain pixels only when they connect to a definite edge.
    edges = cv2.Canny(blurred, low, high)

    return edges

# Usage example
img = cv2.imread('image.jpg')
edges = canny_edge(img, 50, 150)

cv2.imshow('Original', img)
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)
```

### Threshold Tuning

```python
import cv2
import numpy as np

def canny_with_trackbar(image_path):
    """Adjust Canny thresholds with trackbar"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    cv2.namedWindow('Canny')

    def nothing(x):
        pass

    cv2.createTrackbar('Low', 'Canny', 50, 255, nothing)
    cv2.createTrackbar('High', 'Canny', 150, 255, nothing)

    while True:
        low = cv2.getTrackbarPos('Low', 'Canny')
        high = cv2.getTrackbarPos('High', 'Canny')

        # Ensure low is not greater than high
        if low >= high:
            low = high - 1

        edges = cv2.Canny(blurred, low, high)

        cv2.imshow('Canny', edges)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cv2.destroyAllWindows()

# Execute
canny_with_trackbar('image.jpg')
```

### Automatic Threshold Setting

```python
import cv2
import numpy as np

def auto_canny(image, sigma=0.33):
    """
    Automatic threshold Canny
    Calculate low and high based on median value
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Calculate median
    median = np.median(blurred)

    # Calculate thresholds
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))

    print(f"Auto threshold: low={low}, high={high}")

    edges = cv2.Canny(blurred, low, high)

    return edges

# Usage example
img = cv2.imread('image.jpg')
edges = auto_canny(img)
cv2.imshow('Auto Canny', edges)
cv2.waitKey(0)
```

### Canny on Color Images

```python
import cv2
import numpy as np

def canny_color(image, low=50, high=150):
    """
    Canny edge detection on color images
    Detect edges on each channel and combine
    """
    # Method 1: Convert to grayscale then process
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges_gray = cv2.Canny(gray, low, high)

    # Method 2: Process each channel then combine
    b, g, r = cv2.split(image)
    edges_b = cv2.Canny(b, low, high)
    edges_g = cv2.Canny(g, low, high)
    edges_r = cv2.Canny(r, low, high)

    # Combine with OR operation
    edges_color = cv2.bitwise_or(edges_b, edges_g)
    edges_color = cv2.bitwise_or(edges_color, edges_r)

    return edges_gray, edges_color

# Usage example
img = cv2.imread('image.jpg')
edges_gray, edges_color = canny_color(img)

cv2.imshow('Edges (Gray)', edges_gray)
cv2.imshow('Edges (Color)', edges_color)
cv2.waitKey(0)
```

---

## 6. Gradient Magnitude and Direction

### Calculate Gradient Magnitude

```python
import cv2
import numpy as np

def gradient_magnitude_direction(image):
    """Calculate gradient magnitude and direction"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Sobel gradient
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude
    magnitude = np.sqrt(gx**2 + gy**2)

    # arctan2(gy, gx) gives the full 360° direction of the gradient vector;
    # we reduce to 0-180° because edge orientation is undirected — an edge
    # running NE-SW is the same as SW-NE (opposite gradient directions).
    direction = np.arctan2(gy, gx)

    # Convert direction to degrees (0-180)
    direction_deg = np.degrees(direction) % 180

    return magnitude, direction_deg

# Usage example
img = cv2.imread('image.jpg')
mag, dir = gradient_magnitude_direction(img)

# Normalize and display
mag_display = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
dir_display = (dir / 180 * 255).astype(np.uint8)

cv2.imshow('Magnitude', mag_display)
cv2.imshow('Direction', dir_display)
cv2.waitKey(0)
```

### Visualize Gradient Direction

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_gradient_direction(image, step=20):
    """
    Visualize gradient direction with arrows
    step: Sampling interval
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(gx**2 + gy**2)

    # Draw arrows
    result = image.copy()
    h, w = gray.shape

    for y in range(step, h - step, step):
        for x in range(step, w - step, step):
            if magnitude[y, x] > 50:  # Display only above certain magnitude
                # Normalize direction vector
                dx = gx[y, x]
                dy = gy[y, x]
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    dx = int(dx / length * 10)
                    dy = int(dy / length * 10)

                    cv2.arrowedLine(
                        result,
                        (x, y),
                        (x + dx, y + dy),
                        (0, 255, 0),
                        1,
                        tipLength=0.3
                    )

    return result

# Usage example
img = cv2.imread('image.jpg')
vis = visualize_gradient_direction(img, step=15)
cv2.imshow('Gradient Direction', vis)
cv2.waitKey(0)
```

### Compare Edge Detection Algorithms

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_edge_detectors(image_path):
    """Compare various edge detection algorithms"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # 1. Sobel
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = np.clip(sobel, 0, 255).astype(np.uint8)

    # 2. Scharr
    scharr_x = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
    scharr = np.sqrt(scharr_x**2 + scharr_y**2)
    scharr = np.clip(scharr, 0, 255).astype(np.uint8)

    # 3. Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    laplacian = cv2.convertScaleAbs(laplacian)

    # 4. Canny
    canny = cv2.Canny(blurred, 50, 150)

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')

    axes[0, 1].imshow(sobel, cmap='gray')
    axes[0, 1].set_title('Sobel')

    axes[0, 2].imshow(scharr, cmap='gray')
    axes[0, 2].set_title('Scharr')

    axes[1, 0].imshow(laplacian, cmap='gray')
    axes[1, 0].set_title('Laplacian')

    axes[1, 1].imshow(canny, cmap='gray')
    axes[1, 1].set_title('Canny')

    axes[1, 2].axis('off')

    for ax in axes.flatten():
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Run comparison
compare_edge_detectors('image.jpg')
```

---

## 7. Exercises

### Problem 1: Implement Adaptive Canny

Implement a Canny function that automatically adjusts thresholds based on the brightness distribution of the image.

<details>
<summary>Hint</summary>

Calculate low and high thresholds based on the median value of the image.

</details>

<details>
<summary>Solution Code</summary>

```python
import cv2
import numpy as np

def adaptive_canny(image, sigma=0.33):
    """
    Adaptive Canny edge detection
    Automatically set thresholds based on median brightness
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Calculate median
    median = np.median(blurred)

    # Calculate thresholds (adjust range with sigma)
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))

    edges = cv2.Canny(blurred, low, high)

    return edges, low, high

# Test
img = cv2.imread('image.jpg')
edges, low, high = adaptive_canny(img)
print(f"Adaptive thresholds: low={low}, high={high}")
cv2.imshow('Adaptive Canny', edges)
cv2.waitKey(0)
```

</details>

### Problem 2: Separate Edges by Direction

Implement a function that separates and displays horizontal and vertical edges.

<details>
<summary>Hint</summary>

Calculate gradient direction and classify as horizontal (near 0 degrees) or vertical (near 90 degrees) based on angle.

</details>

<details>
<summary>Solution Code</summary>

```python
import cv2
import numpy as np

def separate_edges_by_direction(image, angle_threshold=30):
    """
    Separate horizontal/vertical edges
    angle_threshold: Allowed angle range
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel gradient
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude and direction
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.degrees(np.arctan2(gy, gx)) % 180

    # Apply threshold
    _, edges = cv2.threshold(magnitude.astype(np.uint8), 50, 255, cv2.THRESH_BINARY)

    # Horizontal edges (direction near 0 or 180 degrees)
    # Strong Sobel gy means horizontal edge
    horizontal_mask = ((direction < angle_threshold) |
                       (direction > 180 - angle_threshold))
    horizontal_edges = np.zeros_like(edges)
    horizontal_edges[horizontal_mask & (edges > 0)] = 255

    # Vertical edges (direction near 90 degrees)
    vertical_mask = ((direction > 90 - angle_threshold) &
                     (direction < 90 + angle_threshold))
    vertical_edges = np.zeros_like(edges)
    vertical_edges[vertical_mask & (edges > 0)] = 255

    return horizontal_edges, vertical_edges

# Test
img = cv2.imread('image.jpg')
h_edges, v_edges = separate_edges_by_direction(img)

cv2.imshow('Horizontal Edges', h_edges)
cv2.imshow('Vertical Edges', v_edges)
cv2.waitKey(0)
```

</details>

### Problem 3: Multi-Scale Edge Detection

Implement a function that detects edges at multiple scales and combines them.

<details>
<summary>Hint</summary>

Apply Gaussian blur with various sigma values, then apply Canny and combine the results.

</details>

<details>
<summary>Solution Code</summary>

```python
import cv2
import numpy as np

def multi_scale_canny(image, scales=[1.0, 2.0, 4.0], low=50, high=150):
    """
    Multi-scale Canny edge detection
    scales: Gaussian blur sigma values
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    combined_edges = np.zeros(gray.shape, dtype=np.uint8)

    for sigma in scales:
        # Kernel size based on scale
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)

        # Canny edge detection
        edges = cv2.Canny(blurred, low, high)

        # Combine (OR operation)
        combined_edges = cv2.bitwise_or(combined_edges, edges)

    return combined_edges

# Test
img = cv2.imread('image.jpg')
edges = multi_scale_canny(img, scales=[1.0, 2.0, 3.0])
cv2.imshow('Multi-scale Canny', edges)
cv2.waitKey(0)
```

</details>

### Recommended Problems

| Difficulty | Topic | Description |
|--------|------|------|
| ⭐ | Basic Canny | Apply Canny to various images |
| ⭐⭐ | Threshold Experiment | Find optimal thresholds with trackbar |
| ⭐⭐ | Preprocessing Comparison | Compare edge quality by blur type |
| ⭐⭐⭐ | Document Scanning | Detect document contours |
| ⭐⭐⭐ | Coin Detection | Find coin boundaries using edges |

---

## Next Steps

- [09_Contours.md](./09_Contours.md) - findContours, drawContours, hierarchy structure

---

## References

- [OpenCV Edge Detection Tutorial](https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html)
- [Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
- [Image Gradients](https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html)
