# Hough Transform

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the Hough Transform principle and how a point in image space maps to a curve in parameter space
2. Implement standard and probabilistic Hough line detection using OpenCV's HoughLines() and HoughLinesP()
3. Apply Hough circle detection with HoughCircles() and tune its accumulator and threshold parameters
4. Analyze the trade-offs between standard and probabilistic Hough transforms in terms of speed and accuracy
5. Design a lane detection pipeline that combines edge detection with Hough line filtering
6. Evaluate parameter sensitivity and optimize Hough Transform settings for specific image domains

---

## Overview

The Hough Transform is an algorithm for detecting geometric shapes such as lines and circles in images. It is applied to the output of an edge detector to find specific shapes, and has many uses including lane detection and coin counting.

---

## Table of Contents

1. [Hough Transform Concepts](#1-hough-transform-concepts)
2. [Hough Line Transform](#2-hough-line-transform)
3. [Probabilistic Hough Line Transform](#3-probabilistic-hough-line-transform)
4. [Hough Circle Transform](#4-hough-circle-transform)
5. [Parameter Tuning Strategy](#5-parameter-tuning-strategy)
6. [Lane Detection Basics](#6-lane-detection-basics)
7. [Practice Problems](#7-practice-problems)

---

## 1. Hough Transform Concepts

### Theory: Image-Space / Parameter-Space Duality

A line in 2D image space can be written in many forms, but all require **two parameters** — a 2D line has 2 degrees of freedom. Pick any parametrization that uses a pair `(a, b)`. Then:

- Fixing `(a, b)` defines one line in image space (one curve).
- Fixing a single image point `(x, y)` and asking "which `(a, b)` pairs produce a line through this point?" gives a *curve in parameter space* — the locus of all lines through that point.

These are dual viewpoints of the same geometry. The Hough insight: this duality can be used to detect lines.

1. For each edge point in the image, plot the *curve* of parameter values consistent with it.
2. Where many curves intersect in parameter space, many image edge points lie on the *same line*.
3. Find the peaks in parameter space. They are the lines.

Instead of searching over continuous parameter space (infinite possibilities), quantize it into a grid — an **accumulator array** — and have each edge point increment the bins whose parameters pass through it. Peaks in the accumulator correspond to lines in the image.

### Theory: The Voting Procedure

With `(ρ, θ)` parametrization:

1. **Quantize** the parameter space. Typical resolution: `ρ` bin width = 1 pixel, `θ` bin width = 1°. For an image with diagonal `D`, the `ρ` range is `[-D, D]`, so the accumulator has `~2D × 180` bins.
2. **Initialize** the accumulator to zero.
3. **For each edge pixel** `(x, y)` in the binary edge map, loop over all `θ` bins, compute `ρ = x cos θ + y sin θ`, and **increment** the corresponding `(ρ, θ)` bin.
4. **Find peaks** in the accumulator. Each peak above a threshold corresponds to a line with at least that many supporting edge pixels.

After all voting, the value of an accumulator bin equals the number of edge pixels that lie on the exact line defined by that `(ρ, θ)` — a direct measure of how strongly the data supports that line.

#### Why this works even with missing/noisy data

Voting is **robust to occlusion and noise** because each edge point contributes independently. If half the pixels of a line are missing, the peak in parameter space is smaller but still exists. If a point comes from noise rather than a real line, it contributes to *some* bin but not the peak. Many unstructured noise points contribute to a low flat background — only genuine lines produce concentrated peaks.

### Theory: Generalizations

#### Probabilistic Hough Transform (`HoughLinesP`)

Standard Hough is `O(#edge_pixels × #θ_bins)` per image and returns only `(ρ, θ)` — lines, but no endpoints. The Probabilistic Hough Transform trades a little precision for speed, and recovers endpoints as a bonus:

- Process only a **random subset** of edge pixels (much faster).
- Stop voting for a line once it has enough support.
- Trace along the detected line to find the **actual endpoints** in the edge image.

OpenCV's `HoughLinesP` returns `(x1, y1, x2, y2)` segments, which is usually what you want in practice.

#### Generalized Hough Transform

The same voting idea works for **any** parametric shape, not just lines and circles. For a general shape, build an **R-table** that encodes the offset from each boundary point to a reference point on the shape. At detection time, each edge point votes for the possible reference-point locations. This handles arbitrary shape templates but at much higher memory cost, which is why it is rarely used today — deep-learning detectors have largely replaced it.

### Hough Space

```
Core idea:
A point in image space → a curve in Hough space
A line in image space  → a point in Hough space

Image space (x, y)               Hough space (ρ, θ)
┌─────────────────┐            ┌─────────────────┐
│                 │            │                 │
│    •            │            │      ╱╲         │
│      ╲          │    ──▶     │     ╱  ╲        │
│        ╲        │            │    ╱ •  ╲       │
│          •      │            │   ╱      ╲      │
│                 │            │                 │
└─────────────────┘            └─────────────────┘
Points on a line                Represented by a single point

Line representations:
y = mx + b  (slope, intercept)  → cannot represent vertical lines
ρ = x·cos(θ) + y·sin(θ)         → polar form (preferred)

ρ: perpendicular distance from origin to the line
θ: angle the perpendicular makes with the x-axis
```

Geometrically, every edge point (x, y) satisfies this equation for some (ρ, θ) pair. The key insight is that a single image-space point maps to a sinusoidal *curve* of possible (ρ, θ) values, and points lying on the same line all produce curves that intersect at the same (ρ, θ) — that intersection is the detected line. The slope-intercept form (y = mx + b) cannot handle vertical lines (infinite slope) without special-casing, while the polar form represents every line uniformly without exceptions, which is why it is preferred.

### Hough Transform Pipeline

```
1. Edge detection (Canny, etc.)
         │
         ▼
2. For each edge point, enumerate all candidate lines
   (sweep θ from 0° to 180° and compute ρ)
         │
         ▼
3. Vote into the accumulator array
         │
         ▼
4. Bins with votes above threshold = detected lines

Accumulator visualization:
        θ
      0° ────────────────────▶ 180°
    ρ │  ·  ·  ·  ·  ·  ·  ·  ·
  -max│  ·  ·  ★  ·  ·  ·  ·  ·   ★: many votes
      │  ·  ·  ·  ·  ·  ★  ·  ·      = a line exists
      │  ·  ·  ·  ·  ·  ·  ·  ·
   max│  ·  ·  ·  ·  ·  ·  ·  ·
      ▼
```

### Simple Example

```python
import cv2
import numpy as np

# Visualize the Hough Transform
def visualize_hough_space(image_path):
    """Visualize Hough space"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 50, 150)

    # Standard Hough line transform (returns accumulator peaks)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    # Visualization
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # Draw line (extended in both directions)
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Edges', edges)
    cv2.imshow('Hough Lines', result)
    cv2.waitKey(0)

visualize_hough_space('lines.jpg')
```

### Pros and Cons of the Hough Transform

**Pros**
- Robust to noise
- Detects incomplete shapes
- Detects multiple instances simultaneously

**Cons**
- High computational cost
- Memory grows with parameter-space size
- Sensitive to parameter choice

---

## 2. Hough Line Transform

Simple line fitting (least-squares regression) breaks down the moment edges have gaps, occlusions, or noisy outlier pixels — because it minimizes total error, a few bad points can pull the estimated line far off. The Hough Transform sidesteps this with a *voting* mechanism: each edge pixel independently votes for every line it could lie on, and only lines with many independent votes survive. This makes it inherently robust to gaps and noise without requiring connected contours.

### Theory: Parametrization — Why `(ρ, θ)` and Not `(m, b)`

The slope-intercept form `y = mx + b` has two parameters `(m, b)`. But:

- **Vertical lines have infinite slope.** A purely vertical edge cannot be expressed with a finite `m`. Splitting parameters into two cases (vertical vs non-vertical) is ugly and error-prone.
- **The slope range is unbounded.** A 45° line has `m = 1`, a 89° line has `m ≈ 57` — they are "close" in image-space angle but far apart in `m`. Uniform binning of `m` gives wildly non-uniform angular coverage.

The fix is the **normal form** (or polar form):

```
ρ = x cos θ + y sin θ
```

Where:

- **`ρ`** (rho) is the perpendicular distance from the origin to the line.
- **`θ`** (theta) is the angle that this perpendicular makes with the `x` axis.

Every line in the plane has a unique `(ρ, θ)` with `θ ∈ [0, π)` and `ρ ∈ ℝ` (negative `ρ` corresponds to lines on the "other side" of the origin, but is conventionally mapped by extending `θ` into `[0, 2π)` or flipping the sign of `ρ`). Vertical lines have `θ = 0`; horizontal lines have `θ = π/2`. All parameters are bounded, and uniform binning gives uniform angular coverage.

For a fixed image point `(x₀, y₀)`, the set of `(ρ, θ)` values for lines through that point is a **sinusoid** in parameter space: `ρ = x₀ cos θ + y₀ sin θ`. The accumulator curve for each edge point is therefore a sine wave.

### cv2.HoughLines() Function

```python
lines = cv2.HoughLines(image, rho, theta, threshold)
```

| Parameter | Description |
|-----------|-------------|
| image | Input image (8-bit, single channel, binary edge image) |
| rho | ρ resolution (pixels, typically 1) |
| theta | θ resolution (radians, typically np.pi/180) |
| threshold | Minimum vote count to qualify as a line |
| lines | Detected lines [(ρ, θ), ...] |

### Basic Usage

```python
import cv2
import numpy as np

def hough_lines_example(image_path):
    """Standard Hough line detection"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Hough line transform
    lines = cv2.HoughLines(
        edges,
        rho=1,              # ρ resolution: 1 pixel
        theta=np.pi/180,    # θ resolution: 1 degree
        threshold=100       # minimum number of votes — this is the key quality gate:
                            #   too low → many spurious lines from noise; too high → real lines missed.
                            #   Each edge pixel that lies on a candidate line casts one vote, so threshold
                            #   approximates the minimum pixel length of a line you want to detect.
    )

    result = img.copy()

    if lines is not None:
        print(f"Lines detected: {len(lines)}")

        for line in lines:
            rho, theta = line[0]

            # Polar -> Cartesian
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # Draw infinite line
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Original', img)
    cv2.imshow('Edges', edges)
    cv2.imshow('Hough Lines', result)
    cv2.waitKey(0)

hough_lines_example('building.jpg')
```

### Detecting Only Horizontal/Vertical Lines

```python
import cv2
import numpy as np

def detect_horizontal_vertical_lines(image_path):
    """Detect only horizontal and vertical lines"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    result = img.copy()
    horizontal = []
    vertical = []

    if lines is not None:
        for line in lines:
            rho, theta = line[0]

            # Classify by angle (5° tolerance)
            angle_deg = np.degrees(theta)

            if 85 < angle_deg < 95:  # vertical (θ ≈ 90°)
                vertical.append((rho, theta))
                color = (255, 0, 0)  # blue
            elif angle_deg < 5 or angle_deg > 175:  # horizontal (θ ≈ 0° or 180°)
                horizontal.append((rho, theta))
                color = (0, 255, 0)  # green
            else:
                continue

            # Draw the line
            a = np.cos(theta)
            b = np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
            x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), color, 2)

    print(f"Horizontal: {len(horizontal)}")
    print(f"Vertical:   {len(vertical)}")

    cv2.imshow('H/V Lines', result)
    cv2.waitKey(0)

detect_horizontal_vertical_lines('grid.jpg')
```

---

## 3. Probabilistic Hough Line Transform

### Theory: Circle Hough Transform

A circle requires **three parameters**: center and radius `(x_c, y_c, r)`. Everything from the line case generalizes:

- Each edge point `(x, y)` is consistent with a **3D surface** in parameter space: all `(x_c, y_c, r)` such that `(x - x_c)² + (y - y_c)² = r²`. That surface is the cone with apex at `(x, y, 0)` in `(x_c, y_c, r)` space.
- The accumulator is now 3D, typically of size `W × H × R_max`.
- Voting a cone surface for each edge point is expensive.

#### Gradient-direction optimization

Here is the crucial speedup: at a circle edge pixel, the **gradient direction points along the radius** — toward or away from the center. So if you know the gradient direction at `(x, y)` (from Sobel), the center `(x_c, y_c)` must lie on that gradient line. Instead of voting for every possible center at every possible radius, vote only for centers on the line through `(x, y)` in the gradient direction.

This reduces the per-edge-pixel vote from a 2D surface to a 1D set (the gradient line, parametrized by distance, i.e. radius). OpenCV's `HoughCircles` implementation uses this trick — that is why you must pass `HOUGH_GRADIENT` as the method.

#### Two-stage accumulator

OpenCV's implementation goes further: first accumulate in 2D `(x_c, y_c)` to find centers, then in 1D `r` at each detected center to find radii. This turns a 3D accumulator search into a 2D + 1D search — a much smaller problem.

### cv2.HoughLinesP() Function

```
Standard Hough vs Probabilistic Hough:

Standard Hough (HoughLines):
- Returns infinite lines (ρ, θ)
- Examines every edge point
- Slower, more accurate

Probabilistic Hough (HoughLinesP):
- Returns line segments (x1, y1, x2, y2)
- Random subsampling of edge points
- Faster, more practical
```

```python
lines = cv2.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap)
```

| Parameter | Description |
|-----------|-------------|
| image | Input edge image |
| rho | ρ resolution |
| theta | θ resolution |
| threshold | Minimum vote count |
| minLineLength | Minimum segment length |
| maxLineGap | Maximum allowed gap inside a segment |
| lines | Detected segments [(x1, y1, x2, y2), ...] |

### Basic Usage

```python
import cv2
import numpy as np

def hough_lines_p_example(image_path):
    """Probabilistic Hough line detection"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Probabilistic Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,    # minimum line length — rejects short noise fragments; increase
                             #   for road lanes (want long continuous marks), decrease for short dashes.
        maxLineGap=10        # maximum pixel gap allowed inside a single segment — setting this
                             #   higher "bridges" dashed lines into one segment, which is useful for lane
                             #   detection where paint markings have regular gaps.
    )

    result = img.copy()

    if lines is not None:
        print(f"Segments detected: {len(lines)}")

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Mark segment endpoints
            cv2.circle(result, (x1, y1), 5, (255, 0, 0), -1)
            cv2.circle(result, (x2, y2), 5, (0, 0, 255), -1)

    cv2.imshow('HoughLinesP', result)
    cv2.waitKey(0)

hough_lines_p_example('document.jpg')
```

### Segment Filtering

```python
import cv2
import numpy as np

def filter_lines(image_path, angle_threshold=30):
    """Filter segments by angle and length"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)

    result = img.copy()

    if lines is None:
        return result

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Segment length
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Angle relative to horizontal
        if x2 - x1 != 0:
            angle = np.degrees(np.arctan(abs(y2 - y1) / abs(x2 - x1)))
        else:
            angle = 90

        # Filter: keep only near-horizontal or near-vertical
        if angle < angle_threshold:
            color = (0, 255, 0)  # near horizontal
        elif angle > 90 - angle_threshold:
            color = (255, 0, 0)  # near vertical
        else:
            continue  # ignore diagonals

        cv2.line(result, (x1, y1), (x2, y2), color, 2)

    cv2.imshow('Filtered Lines', result)
    cv2.waitKey(0)

    return result

filter_lines('building.jpg', angle_threshold=20)
```

### Segment Merging

```python
import cv2
import numpy as np
from collections import defaultdict

def merge_lines(lines, angle_threshold=10, distance_threshold=20):
    """Merge similar segments"""
    if lines is None or len(lines) == 0:
        return []

    # Group segments by angle
    groups = defaultdict(list)

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Compute angle
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

        # Quantize angle into bins of angle_threshold width
        angle_group = round(angle / angle_threshold) * angle_threshold
        groups[angle_group].append(line[0])

    merged = []

    for angle, group_lines in groups.items():
        if len(group_lines) == 1:
            merged.append(group_lines[0])
            continue

        # Within the group, merge nearby segments
        # Simple strategy: take min/max along the dominant axis
        all_points = []
        for x1, y1, x2, y2 in group_lines:
            all_points.extend([(x1, y1), (x2, y2)])

        all_points = np.array(all_points)

        # Sort along the dominant axis to pick the two extreme endpoints
        if abs(np.cos(np.radians(angle))) > 0.5:
            # Mostly horizontal: sort by x
            sorted_pts = sorted(all_points, key=lambda p: p[0])
        else:
            # Mostly vertical: sort by y
            sorted_pts = sorted(all_points, key=lambda p: p[1])

        start = sorted_pts[0]
        end = sorted_pts[-1]
        merged.append([start[0], start[1], end[0], end[1]])

    return merged
```

---

## 4. Hough Circle Transform

### cv2.HoughCircles() Function

```
Hough Circle Transform:
Detects circles in images

Circle equation: (x - a)² + (y - b)² = r²
Parameters: center (a, b), radius r

A 3D accumulator is required → inefficient
→ Use the gradient-based method (cv2.HOUGH_GRADIENT)

cv2.HOUGH_GRADIENT pipeline:
1. Edge detection
2. Vote along gradient direction at each edge point
3. Select candidate centers
4. Estimate radii
```

```python
circles = cv2.HoughCircles(image, method, dp, minDist, param1, param2, minRadius, maxRadius)
```

| Parameter | Description |
|-----------|-------------|
| image | Input grayscale image |
| method | Detection method (cv2.HOUGH_GRADIENT or cv2.HOUGH_GRADIENT_ALT) |
| dp | Accumulator resolution ratio (1 = same as input) |
| minDist | Minimum distance between detected circle centers |
| param1 | Upper Canny threshold |
| param2 | Circle detection threshold (lower → more detections) |
| minRadius | Minimum radius (0 = unbounded) |
| maxRadius | Maximum radius (0 = unbounded) |

### Basic Usage

```python
import cv2
import numpy as np

def hough_circles_example(image_path):
    """Hough circle detection"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction (important for circle detection)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Hough circle transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,              # inverse ratio of accumulator resolution — dp=1 means full resolution
                           #   (more memory, more precise); dp=2 halves the accumulator size (faster).
        minDist=50,        # minimum distance between circle centers — prevents the algorithm
                           #   from returning many overlapping circles for the same coin/object.
        param1=100,        # upper Canny threshold; the lower is automatically set to half.
        param2=30,         # accumulator threshold for circle centers — the most sensitive tuning
                           #   knob. Lower values detect more circles (including false positives from noise);
                           #   higher values require a stronger consensus of edge points around the center,
                           #   yielding fewer but more confident detections.
        minRadius=10,      # minimum radius
        maxRadius=100      # maximum radius
    )

    result = img.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for circle in circles[0, :]:
            cx, cy, r = circle

            # Draw circle
            cv2.circle(result, (cx, cy), r, (0, 255, 0), 2)

            # Center point
            cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)

            print(f"Circle: center=({cx}, {cy}), radius={r}")

        print(f"Circles detected: {len(circles[0])}")

    cv2.imshow('Circles', result)
    cv2.waitKey(0)

hough_circles_example('coins.jpg')
```

### Coin Detection

```python
import cv2
import numpy as np

def detect_coins(image_path):
    """Coin detection and classification"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Hough circle transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=100,
        param2=35,
        minRadius=30,
        maxRadius=80
    )

    result = img.copy()
    coin_count = 0
    total_value = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for circle in circles[0, :]:
            cx, cy, r = circle
            coin_count += 1

            # Estimate denomination from radius (illustrative)
            if r < 40:
                value = 10
                color = (255, 0, 0)    # blue
            elif r < 55:
                value = 50
                color = (0, 255, 0)    # green
            else:
                value = 100
                color = (0, 0, 255)    # red

            total_value += value

            # Draw
            cv2.circle(result, (cx, cy), r, color, 2)
            cv2.circle(result, (cx, cy), 3, (0, 0, 0), -1)
            cv2.putText(result, f'{value}', (cx - 15, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    print(f"Coin count: {coin_count}")
    print(f"Total value: {total_value} won")

    cv2.imshow('Coins', result)
    cv2.waitKey(0)

    return coin_count, total_value

detect_coins('coins.jpg')
```

### HOUGH_GRADIENT_ALT (OpenCV 4.3+)

```python
import cv2
import numpy as np

def hough_circles_alt(image_path):
    """Use HOUGH_GRADIENT_ALT (more accurate)"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # HOUGH_GRADIENT_ALT: more accurate but slower
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT_ALT,  # alternative algorithm
        dp=1.5,
        minDist=50,
        param1=300,    # edge gradient threshold
        param2=0.9,    # circularity threshold (0-1, higher = stricter)
        minRadius=20,
        maxRadius=100
    )

    result = img.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for cx, cy, r in circles[0, :]:
            cv2.circle(result, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)

    cv2.imshow('HOUGH_GRADIENT_ALT', result)
    cv2.waitKey(0)

hough_circles_alt('circles.jpg')
```

---

## 5. Parameter Tuning Strategy

### Line Detection Parameters

```
┌────────────────────────────────────────────────────────────────┐
│                    HoughLines parameters                        │
├────────────────────────────────────────────────────────────────┤
│ rho (ρ resolution)                                              │
│ - Smaller: more precise, more memory, slower                    │
│ - Recommended: 1 (1 pixel)                                      │
│                                                                │
│ theta (θ resolution)                                            │
│ - Smaller: more precise angles                                  │
│ - Recommended: np.pi/180 (1°)                                   │
│                                                                │
│ threshold (minimum vote count)                                  │
│ - Higher: only strong (long) lines                              │
│ - Lower: also weak (short) lines, more noise                    │
│ - Tip: scale with image size and expected line length           │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                   HoughLinesP parameters                        │
├────────────────────────────────────────────────────────────────┤
│ minLineLength (minimum segment length)                          │
│ - Higher: only long segments                                    │
│ - Effective at suppressing noise                                │
│                                                                │
│ maxLineGap (maximum gap)                                        │
│ - Higher: connects broken segments into one                     │
│ - Useful for dashed-line detection                              │
└────────────────────────────────────────────────────────────────┘
```

### Circle Detection Parameters

```
┌────────────────────────────────────────────────────────────────┐
│                   HoughCircles parameters                       │
├────────────────────────────────────────────────────────────────┤
│ dp (resolution ratio)                                           │
│ - 1: full resolution → accurate but slow                        │
│ - 2: half resolution → fast but less accurate                   │
│ - Recommended: 1 ~ 1.5                                          │
│                                                                │
│ minDist (minimum center distance)                               │
│ - Too small: same circle detected multiple times                │
│ - Too large: nearby circles missed                              │
│ - Recommended: ≥ 2 × expected radius                            │
│                                                                │
│ param1 (upper Canny threshold)                                  │
│ - Higher: only strong edges used                                │
│ - Recommended: 100 ~ 200                                        │
│                                                                │
│ param2 (accumulator threshold)                                  │
│ - Higher: only confident circles                                │
│ - Lower: also incomplete circles                                │
│ - Recommended: 20 ~ 50                                          │
│                                                                │
│ minRadius, maxRadius                                            │
│ - Bound the expected radius range                               │
│ - Wrong values cause detection to fail                          │
└────────────────────────────────────────────────────────────────┘
```

### Trackbar-Based Parameter Tuning

```python
import cv2
import numpy as np

def tune_hough_circles(image_path):
    """Tune HoughCircles parameters with trackbars"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    cv2.namedWindow('Circles')

    def nothing(x):
        pass

    cv2.createTrackbar('minDist', 'Circles', 50, 200, nothing)
    cv2.createTrackbar('param1', 'Circles', 100, 300, nothing)
    cv2.createTrackbar('param2', 'Circles', 30, 100, nothing)
    cv2.createTrackbar('minRadius', 'Circles', 10, 100, nothing)
    cv2.createTrackbar('maxRadius', 'Circles', 100, 200, nothing)

    while True:
        minDist = cv2.getTrackbarPos('minDist', 'Circles')
        param1 = cv2.getTrackbarPos('param1', 'Circles')
        param2 = cv2.getTrackbarPos('param2', 'Circles')
        minRadius = cv2.getTrackbarPos('minRadius', 'Circles')
        maxRadius = cv2.getTrackbarPos('maxRadius', 'Circles')

        # Validation
        if minDist < 1:
            minDist = 1
        if param2 < 1:
            param2 = 1

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius
        )

        result = img.copy()

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for cx, cy, r in circles[0, :]:
                cv2.circle(result, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)

            # Display number of detected circles
            cv2.putText(result, f'Circles: {len(circles[0])}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Circles', result)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

tune_hough_circles('coins.jpg')
```

---

## 6. Lane Detection Basics

### Lane Detection Pipeline

```
1. Define region of interest (ROI)
         │
         ▼
2. Convert to grayscale
         │
         ▼
3. Gaussian blur
         │
         ▼
4. Canny edge detection
         │
         ▼
5. Mask the ROI
         │
         ▼
6. Hough line transform
         │
         ▼
7. Filter and average segments
         │
         ▼
8. Compose with the original image
```

### Basic Lane Detection

```python
import cv2
import numpy as np

def detect_lane_lines(image):
    """Basic lane detection"""
    height, width = image.shape[:2]

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edges
    edges = cv2.Canny(blurred, 50, 150)

    # Region of interest (trapezoid)
    mask = np.zeros_like(edges)
    vertices = np.array([[
        (0, height),
        (width * 0.45, height * 0.6),
        (width * 0.55, height * 0.6),
        (width, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough line transform
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )

    # Result image
    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Compose with original
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    return result

# Example
img = cv2.imread('road.jpg')
result = detect_lane_lines(img)
cv2.imshow('Lane Detection', result)
cv2.waitKey(0)
```

### Left/Right Lane Separation

```python
import cv2
import numpy as np

def separate_lanes(image):
    """Separately detect left and right lanes"""
    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # ROI mask
    mask = np.zeros_like(edges)
    vertices = np.array([[
        (50, height),
        (width * 0.45, height * 0.6),
        (width * 0.55, height * 0.6),
        (width - 50, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked, 1, np.pi/180, 30,
                             minLineLength=30, maxLineGap=100)

    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Slope
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)

            # Classify into left/right by slope
            # Image coordinates: y axis points downward
            # Left lane:  negative slope (/)
            # Right lane: positive slope (\)
            if slope < -0.5:
                left_lines.append(line[0])
            elif slope > 0.5:
                right_lines.append(line[0])

    result = image.copy()

    # Draw left/right lanes
    for x1, y1, x2, y2 in left_lines:
        cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 3)  # blue

    for x1, y1, x2, y2 in right_lines:
        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 3)  # red

    return result, left_lines, right_lines

# Example
img = cv2.imread('road.jpg')
result, left, right = separate_lanes(img)
print(f"Left lane segments:  {len(left)}")
print(f"Right lane segments: {len(right)}")
cv2.imshow('Lanes', result)
cv2.waitKey(0)
```

### Lane Averaging

```python
import cv2
import numpy as np

def average_lane_lines(lines, height):
    """Average segments into a single line"""
    if len(lines) == 0:
        return None

    # Collect all points
    x_coords = []
    y_coords = []

    for x1, y1, x2, y2 in lines:
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])

    # First-degree polynomial fit (line)
    poly = np.polyfit(y_coords, x_coords, 1)

    # Compute the start and end of the averaged line
    y1 = height
    y2 = int(height * 0.6)
    x1 = int(np.polyval(poly, y1))
    x2 = int(np.polyval(poly, y2))

    return (x1, y1, x2, y2)

def detect_lanes_averaged(image):
    """Lane detection with averaged segments"""
    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # ROI
    mask = np.zeros_like(edges)
    vertices = np.array([[
        (50, height),
        (width * 0.45, height * 0.6),
        (width * 0.55, height * 0.6),
        (width - 50, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked, 1, np.pi/180, 30,
                             minLineLength=30, maxLineGap=100)

    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)

            if slope < -0.5:
                left_lines.append(line[0])
            elif slope > 0.5:
                right_lines.append(line[0])

    result = image.copy()

    # Draw averaged lanes
    left_avg = average_lane_lines(left_lines, height)
    right_avg = average_lane_lines(right_lines, height)

    if left_avg is not None:
        cv2.line(result, (left_avg[0], left_avg[1]),
                 (left_avg[2], left_avg[3]), (255, 0, 0), 5)

    if right_avg is not None:
        cv2.line(result, (right_avg[0], right_avg[1]),
                 (right_avg[2], right_avg[3]), (0, 0, 255), 5)

    # Fill the lane region
    if left_avg is not None and right_avg is not None:
        pts = np.array([
            [left_avg[0], left_avg[1]],
            [left_avg[2], left_avg[3]],
            [right_avg[2], right_avg[3]],
            [right_avg[0], right_avg[1]]
        ], np.int32)

        overlay = result.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        result = cv2.addWeighted(overlay, 0.3, result, 0.7, 0)

    return result

# Example
img = cv2.imread('road.jpg')
result = detect_lanes_averaged(img)
cv2.imshow('Averaged Lanes', result)
cv2.waitKey(0)
```

### Parking Space Detection

```python
import cv2
import numpy as np

class ParkingDetector:
    """Parking space detection system"""

    def __init__(self):
        self.parking_spaces = []

    def detect_parking_lines(self, image_path):
        """Detect parking lines"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binarization
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Edge detection
        edges = cv2.Canny(binary, 50, 150)

        # Line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                               minLineLength=50, maxLineGap=10)

        # Draw lines
        result = img.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return result, lines

    def find_parking_spaces(self, lines, img_shape):
        """Find parking spaces from lines"""
        if lines is None:
            return []

        # Group parallel lines
        vertical_lines = []
        horizontal_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            if abs(angle) < 45:  # horizontal
                horizontal_lines.append(line[0])
            else:  # vertical
                vertical_lines.append(line[0])

        # Find rectangular parking spaces
        spaces = []
        for v_line in vertical_lines:
            for h_line in horizontal_lines:
                space = self.calculate_space(v_line, h_line)
                if space is not None:
                    spaces.append(space)

        return spaces

    def calculate_space(self, v_line, h_line):
        """Compute a parking space (simplified)"""
        return None

# Example
detector = ParkingDetector()
result, lines = detector.detect_parking_lines('parking.jpg')
cv2.imshow('Parking Line Detection', result)
cv2.waitKey(0)
```

### Document Edge Detection

```python
import cv2
import numpy as np

def detect_document_edges(image_path):
    """Detect document edges"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preprocessing
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                           minLineLength=100, maxLineGap=10)

    # Draw lines
    result = img.copy()
    if lines is not None:
        # Group by angle
        horizontal = []
        vertical = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            if abs(angle) < 45:
                horizontal.append(line[0])
                cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
            else:
                vertical.append(line[0])
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Original', img)
    cv2.imshow('Document Edges', result)
    cv2.waitKey(0)

    return result

# Example
result = detect_document_edges('document.jpg')
```

---

## 7. Practice Problems

### Problem 1: Chessboard Detection

Detect every line in a chessboard image and find the intersections.

<details>
<summary>Solution</summary>

```python
import cv2
import numpy as np

def detect_chessboard_lines(image_path):
    """Detect chessboard lines and their intersections"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    result = img.copy()
    horizontal = []
    vertical = []

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta)

            a = np.cos(theta)
            b = np.sin(theta)

            # Classify into horizontal / vertical
            if 80 < angle < 100:  # vertical
                vertical.append((rho, theta))
            elif angle < 10 or angle > 170:  # horizontal
                horizontal.append((rho, theta))

    # Compute intersections
    intersections = []
    for h_rho, h_theta in horizontal:
        for v_rho, v_theta in vertical:
            # Intersection of two lines
            A = np.array([
                [np.cos(h_theta), np.sin(h_theta)],
                [np.cos(v_theta), np.sin(v_theta)]
            ])
            b = np.array([h_rho, v_rho])

            try:
                x, y = np.linalg.solve(A, b)
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    intersections.append((int(x), int(y)))
            except:
                pass

    # Draw
    for x, y in intersections:
        cv2.circle(result, (x, y), 5, (0, 0, 255), -1)

    print(f"Intersections: {len(intersections)}")
    cv2.imshow('Chessboard', result)
    cv2.waitKey(0)

detect_chessboard_lines('chessboard.jpg')
```

</details>

### Problem 2: Iris Detection

Detect the iris circle in an eye image.

<details>
<summary>Solution</summary>

```python
import cv2
import numpy as np

def detect_iris(image_path):
    """Detect the iris in an eye image"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Equalize brightness
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Iris detection (a dark circle)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=100,
        param2=25,
        minRadius=20,
        maxRadius=60
    )

    result = img.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))

        # Pick the largest circle (the iris)
        for cx, cy, r in sorted(circles[0], key=lambda x: -x[2])[:1]:
            # Iris
            cv2.circle(result, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(result, (cx, cy), 2, (0, 0, 255), 3)

    cv2.imshow('Iris', result)
    cv2.waitKey(0)

detect_iris('eye.jpg')
```

</details>

### Problem 3: Circular Road Sign Detection

Detect red, circular traffic signs.

<details>
<summary>Solution</summary>

```python
import cv2
import numpy as np

def detect_red_signs(image_path):
    """Detect red, circular traffic signs"""
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red mask (red wraps around at 0° and 180° in HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Circle detection
    circles = cv2.HoughCircles(
        red_mask,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=100
    )

    result = img.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for cx, cy, r in circles[0]:
            cv2.circle(result, (cx, cy), r, (0, 255, 0), 3)
            cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)

    cv2.imshow('Red Signs', result)
    cv2.imshow('Mask', red_mask)
    cv2.waitKey(0)

detect_red_signs('traffic_sign.jpg')
```

</details>

### Recommended Exercises

| Difficulty | Topic | Description |
|------------|-------|-------------|
| Easy | Line detection | Detect horizontal/vertical lines on a building |
| Medium | Coin counting | Count coins and total their values |
| Medium | Document detection | Detect the four edges of a document |
| Hard | Lane detection | Real-time lane detection from a road video |
| Hard | Dashboard | Read tick marks on an analog gauge |

---

## Summary

### Key Concepts
1. **Hough Transform principles**
   - Mapping from image space to parameter space
   - Voting mechanism
   - Local maxima detection

2. **Line detection**
   - Standard Hough Transform (HoughLines)
   - Probabilistic Hough Transform (HoughLinesP)
   - Parameter tuning

3. **Circle detection**
   - 3D parameter space (x, y, r)
   - Parameter optimization
   - Detecting multiple circles

4. **Practical applications**
   - Lane detection
   - Parking space detection
   - Document edge detection
   - Object counting

5. **Performance optimization**
   - Parameter optimization
   - ROI processing
   - Multi-scale processing

### Parameter Tuning Guide
- **rho, theta**: Higher resolution = more accurate but slower
- **threshold**: Higher = fewer but stronger lines
- **minLineLength**: Minimum line-length threshold
- **maxLineGap**: Maximum gap allowed inside a segment
- **param1**: Edge detection threshold
- **param2**: Accumulator threshold

### Important Notes
- Preprocessing is crucial (edge detection, noise removal)
- Parameter values vary greatly with image characteristics
- Real-time processing requires performance optimization
- ROI cuts down compute cost
- Multi-scale processing helps detect objects across sizes

---

## Next Steps

- [Histogram Analysis](./12_Histogram_Analysis.md) - calcHist, equalizeHist, CLAHE

---

## References

- [OpenCV Hough Line Transform](https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html)
- [OpenCV Hough Circle Transform](https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html)
- [Lane Detection Tutorial](https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132)
