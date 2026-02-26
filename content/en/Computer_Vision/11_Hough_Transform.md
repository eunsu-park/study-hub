# Hough Transform

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the Hough Transform principle and how it maps image space points to curves in parameter space
2. Implement standard and probabilistic Hough line detection using OpenCV's HoughLines() and HoughLinesP()
3. Apply Hough circle detection with HoughCircles() and tune its accumulator and threshold parameters
4. Analyze the trade-offs between standard and probabilistic Hough transforms in terms of speed and accuracy
5. Design a lane detection pipeline that combines edge detection with Hough line filtering
6. Evaluate parameter sensitivity and optimize Hough Transform settings for specific image domains

---

## Overview
The Hough Transform is a classical technique used to detect shapes in images. It is primarily used to find lines and circles but can theoretically detect any parametric shape. It works by transforming the image space into parameter space and finding points where many lines intersect.

## Table of Contents
1. [Hough Transform Principles](#1-hough-transform-principles)
2. [Line Detection](#2-line-detection)
3. [Circle Detection](#3-circle-detection)
4. [Practical Applications](#4-practical-applications)
5. [Performance Optimization](#5-performance-optimization)
6. [Practice Problems](#6-practice-problems)

---

## 1. Hough Transform Principles

### 1.1 Basic Concept

Lines in image space are represented as points in parameter space.

**Line Representation in Image Space**
```
y = mx + b  # slope-intercept form
```

**Line Representation in Hough Space (Polar Coordinates)**
```
ρ = x cos(θ) + y sin(θ)
```

- ρ: distance from origin to the line
- θ: angle the perpendicular from the origin makes with the x-axis

Geometrically, every point (x, y) on an edge satisfies this equation for some (ρ, θ) pair. The key insight is that a single image-space point maps to a sinusoidal *curve* of possible (ρ, θ) values, and collinear points produce curves that all intersect at the same (ρ, θ) — that intersection is the detected line. Polar form is preferred over slope-intercept (y = mx + b) because it handles vertical lines (infinite slope) without special cases.

### 1.2 Algorithm Flow

```
1. Edge Detection (Canny, etc.)
2. Parameter Space Generation
3. Voting Process
   - Each edge point votes for all possible lines passing through it
4. Find Local Maxima
   - Points with many votes become detected lines
5. Extract Lines/Circles
```

### 1.3 Advantages and Disadvantages

**Advantages**
- Robust to noise
- Can detect incomplete shapes
- Can detect multiple instances simultaneously

**Disadvantages**
- High computational cost
- Requires memory proportional to parameter space size
- Sensitive to parameter selection

---

## 2. Line Detection

Simple line fitting (least-squares regression) breaks down the moment an edge has gaps, occlusions, or noisy outlier pixels — it minimizes total error, so a few bad points can pull the estimated line far off. The Hough Transform sidesteps this by using a *voting* mechanism: each edge pixel independently votes for every line it could belong to, and only lines with many independent votes survive. This makes it inherently robust to gaps and noise without requiring a connected contour.

### 2.1 Standard Hough Transform (cv2.HoughLines)

Detects all possible lines and returns them in (ρ, θ) format.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def hough_lines_demo(image_path):
    """Standard Hough Transform Demo"""
    # Read image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Hough Transform
    # HoughLines(image, rho, theta, threshold)
    # rho: distance resolution (pixels)
    # theta: angle resolution (radians)
    # threshold: minimum number of votes — this is the key quality gate:
    #   too low → many spurious lines from noise; too high → real lines missed.
    #   200 works well for images with strong, long edges (e.g., sudoku grids).
    #   Each edge pixel that lies on a candidate line casts one vote, so threshold
    #   approximates the minimum pixel length of a line you want to detect.
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    # Draw results
    result = img.copy()
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # Extend line beyond image boundaries
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original'), plt.axis('off')
    plt.subplot(132), plt.imshow(edges, cmap='gray')
    plt.title('Edges'), plt.axis('off')
    plt.subplot(133), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Detected Lines'), plt.axis('off')
    plt.tight_layout()
    plt.show()

    return lines

# Example usage
lines = hough_lines_demo('sudoku.jpg')
print(f"Number of lines detected: {len(lines) if lines is not None else 0}")
```

### 2.2 Probabilistic Hough Transform (cv2.HoughLinesP)

Detects only line segments, returning them as (x1, y1, x2, y2) coordinates.

```python
def hough_lines_p_demo(image_path):
    """Probabilistic Hough Transform Demo"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Probabilistic Hough Transform
    # HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap)
    # minLineLength: minimum line length — rejects short noise fragments; increase
    #   for road lanes (want long continuous marks), decrease for short dashes.
    # maxLineGap: maximum pixel gap allowed inside a single segment — setting this
    #   higher "bridges" dashed lines into one segment, which is useful for lane
    #   detection where paint markings have regular gaps.
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                            minLineLength=100, maxLineGap=10)

    # Draw results
    result = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original'), plt.axis('off')
    plt.subplot(132), plt.imshow(edges, cmap='gray')
    plt.title('Edges'), plt.axis('off')
    plt.subplot(133), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Line Segments'), plt.axis('off')
    plt.tight_layout()
    plt.show()

    return lines

# Example usage
lines = hough_lines_p_demo('road.jpg')
print(f"Number of line segments detected: {len(lines) if lines is not None else 0}")
```

### 2.3 Comparison Between HoughLines and HoughLinesP

```python
def compare_hough_methods(image_path):
    """Compare Standard and Probabilistic Hough Transforms"""
    import time

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Standard Hough Transform
    start_time = time.time()
    lines1 = cv2.HoughLines(edges, 1, np.pi/180, 200)
    time1 = time.time() - start_time

    result1 = img.copy()
    if lines1 is not None:
        for line in lines1[:20]:  # Display only first 20 lines
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(result1, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Probabilistic Hough Transform
    start_time = time.time()
    lines2 = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                             minLineLength=100, maxLineGap=10)
    time2 = time.time() - start_time

    result2 = img.copy()
    if lines2 is not None:
        for line in lines2:
            x1, y1, x2, y2 = line[0]
            cv2.line(result2, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display results
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original'), plt.axis('off')
    plt.subplot(132), plt.imshow(cv2.cvtColor(result1, cv2.COLOR_BGR2RGB))
    plt.title(f'HoughLines ({time1:.3f}s)'), plt.axis('off')
    plt.subplot(133), plt.imshow(cv2.cvtColor(result2, cv2.COLOR_BGR2RGB))
    plt.title(f'HoughLinesP ({time2:.3f}s)'), plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"\n=== Comparison Results ===")
    print(f"HoughLines: {len(lines1) if lines1 is not None else 0} lines, {time1:.3f}s")
    print(f"HoughLinesP: {len(lines2) if lines2 is not None else 0} segments, {time2:.3f}s")

# Example usage
compare_hough_methods('building.jpg')
```

---

## 3. Circle Detection

### 3.1 Hough Circle Transform (cv2.HoughCircles)

Detects circles using the Hough Transform. Each circle is defined by three parameters (center x, y and radius r).

```python
def hough_circles_demo(image_path):
    """Hough Circle Transform Demo"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    gray = cv2.medianBlur(gray, 5)

    # Circle detection
    # HoughCircles(image, method, dp, minDist, param1, param2, minRadius, maxRadius)
    # dp: inverse ratio of accumulator resolution — dp=1 means full resolution
    #   (more memory, more precise); dp=2 halves the accumulator size (faster).
    # minDist: minimum distance between circle centers — prevents the algorithm
    #   from returning many overlapping circles for the same coin/object.
    # param1: upper Canny threshold; the lower is automatically set to half.
    # param2: accumulator threshold for circle centers — the most sensitive tuning
    #   knob. Lower values detect more circles (including false positives from noise);
    #   higher values require a stronger consensus of edge points around the center,
    #   yielding fewer but more confident detections.
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30,
                               minRadius=10, maxRadius=100)

    # Draw results
    result = img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw outer circle
            cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw center
            cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Display
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original'), plt.axis('off')
    plt.subplot(122), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected Circles ({len(circles[0]) if circles is not None else 0})'), plt.axis('off')
    plt.tight_layout()
    plt.show()

    return circles

# Example usage
circles = hough_circles_demo('coins.jpg')
if circles is not None:
    print(f"Number of circles detected: {len(circles[0])}")
    for i, circle in enumerate(circles[0][:5], 1):
        print(f"Circle {i}: center=({circle[0]}, {circle[1]}), radius={circle[2]}")
```

### 3.2 Circle Detection with Parameter Adjustment

```python
def interactive_circle_detection(image_path):
    """Interactive Circle Detection Parameter Adjustment"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Test multiple parameter combinations
    param_sets = [
        {'param1': 50, 'param2': 30, 'minRadius': 10, 'maxRadius': 100},
        {'param1': 100, 'param2': 30, 'minRadius': 10, 'maxRadius': 100},
        {'param1': 50, 'param2': 20, 'minRadius': 10, 'maxRadius': 100},
        {'param1': 50, 'param2': 30, 'minRadius': 20, 'maxRadius': 80},
    ]

    plt.figure(figsize=(16, 8))
    for idx, params in enumerate(param_sets, 1):
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, **params)

        result = img.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)

        plt.subplot(2, 2, idx)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f"param1={params['param1']}, param2={params['param2']}\n"
                 f"({len(circles[0]) if circles is not None else 0} circles)")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
interactive_circle_detection('coins.jpg')
```

### 3.3 Coin Counting Application

```python
class CoinCounter:
    """Coin Counting System"""

    def __init__(self):
        # Coin size settings (in pixels)
        self.coin_sizes = {
            'large': (40, 60),    # e.g., 500 won
            'medium': (30, 40),   # e.g., 100 won
            'small': (20, 30)     # e.g., 10 won
        }

        self.coin_values = {
            'large': 500,
            'medium': 100,
            'small': 10
        }

    def classify_coin(self, radius):
        """Classify coin based on radius"""
        if self.coin_sizes['large'][0] <= radius <= self.coin_sizes['large'][1]:
            return 'large'
        elif self.coin_sizes['medium'][0] <= radius <= self.coin_sizes['medium'][1]:
            return 'medium'
        elif self.coin_sizes['small'][0] <= radius <= self.coin_sizes['small'][1]:
            return 'small'
        return None

    def count_coins(self, image_path):
        """Count coins"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        # Detect circles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 30,
                                   param1=50, param2=30,
                                   minRadius=20, maxRadius=60)

        result = img.copy()
        coin_counts = {'large': 0, 'medium': 0, 'small': 0}
        total_value = 0

        if circles is not None:
            circles = np.uint16(np.around(circles))

            for i in circles[0, :]:
                x, y, r = i[0], i[1], i[2]
                coin_type = self.classify_coin(r)

                if coin_type:
                    coin_counts[coin_type] += 1
                    value = self.coin_values[coin_type]
                    total_value += value

                    # Draw with different colors for each type
                    if coin_type == 'large':
                        color = (0, 0, 255)  # Red
                    elif coin_type == 'medium':
                        color = (0, 255, 0)  # Green
                    else:
                        color = (255, 0, 0)  # Blue

                    cv2.circle(result, (x, y), r, color, 2)
                    cv2.circle(result, (x, y), 2, color, 3)
                    cv2.putText(result, f'{value}', (x-10, y+5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display results
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original'), plt.axis('off')
        plt.subplot(122), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f'Total: {total_value} won'), plt.axis('off')
        plt.tight_layout()
        plt.show()

        return coin_counts, total_value

# Example usage
counter = CoinCounter()
counts, total = counter.count_coins('coins.jpg')
print(f"\n=== Coin Counting Results ===")
print(f"Large coins (500 won): {counts['large']}")
print(f"Medium coins (100 won): {counts['medium']}")
print(f"Small coins (10 won): {counts['small']}")
print(f"Total value: {total} won")
```

---

## 4. Practical Applications

### 4.1 Lane Detection

```python
class LaneDetector:
    """Lane Detection System"""

    def __init__(self):
        self.roi_vertices = None

    def region_of_interest(self, img, vertices):
        """Apply region of interest mask"""
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def draw_lines(self, img, lines):
        """Draw lane lines"""
        if lines is None:
            return img

        # Separate left and right lanes
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0

            # Filter by slope
            if abs(slope) < 0.5:  # Ignore nearly horizontal lines
                continue

            if slope < 0:
                left_lines.append(line[0])
            else:
                right_lines.append(line[0])

        # Calculate average lines
        line_img = np.zeros_like(img)

        if left_lines:
            left_line = self.average_lines(left_lines, img.shape)
            if left_line is not None:
                cv2.line(line_img, left_line[0], left_line[1], (255, 0, 0), 10)

        if right_lines:
            right_line = self.average_lines(right_lines, img.shape)
            if right_line is not None:
                cv2.line(line_img, right_line[0], right_line[1], (0, 0, 255), 10)

        return cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

    def average_lines(self, lines, shape):
        """Calculate average line from multiple lines"""
        if not lines:
            return None

        lines = np.array(lines)
        x1s, y1s, x2s, y2s = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]

        # Linear regression
        fit = np.polyfit(np.concatenate([x1s, x2s]),
                        np.concatenate([y1s, y2s]), 1)

        # Extend line to ROI boundaries
        y1 = shape[0]
        y2 = int(shape[0] * 0.6)
        x1 = int((y1 - fit[1]) / fit[0])
        x2 = int((y2 - fit[1]) / fit[0])

        return [(x1, y1), (x2, y2)]

    def detect_lanes(self, image_path):
        """Detect lanes"""
        img = cv2.imread(image_path)
        height, width = img.shape[:2]

        # Define ROI
        vertices = np.array([[
            (width * 0.1, height),
            (width * 0.4, height * 0.6),
            (width * 0.6, height * 0.6),
            (width * 0.9, height)
        ]], dtype=np.int32)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blur, 50, 150)

        # Apply ROI
        roi_edges = self.region_of_interest(edges, vertices)

        # Line detection
        lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, 50,
                               minLineLength=100, maxLineGap=50)

        # Draw results
        result = self.draw_lines(img.copy(), lines)

        # Display
        plt.figure(figsize=(15, 10))
        plt.subplot(231), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original'), plt.axis('off')
        plt.subplot(232), plt.imshow(gray, cmap='gray')
        plt.title('Grayscale'), plt.axis('off')
        plt.subplot(233), plt.imshow(edges, cmap='gray')
        plt.title('Edges'), plt.axis('off')
        plt.subplot(234), plt.imshow(roi_edges, cmap='gray')
        plt.title('ROI'), plt.axis('off')
        plt.subplot(235), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Lane Detection Result'), plt.axis('off')
        plt.tight_layout()
        plt.show()

        return result

# Example usage
detector = LaneDetector()
result = detector.detect_lanes('road.jpg')
```

### 4.2 Parking Space Detection

```python
class ParkingDetector:
    """Parking Space Detection System"""

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

            if abs(angle) < 45:  # Horizontal
                horizontal_lines.append(line[0])
            else:  # Vertical
                vertical_lines.append(line[0])

        # Find rectangular parking spaces
        spaces = []
        for v_line in vertical_lines:
            for h_line in horizontal_lines:
                # Calculate intersection
                space = self.calculate_space(v_line, h_line)
                if space is not None:
                    spaces.append(space)

        return spaces

    def calculate_space(self, v_line, h_line):
        """Calculate parking space"""
        # Simplified implementation
        # In practice, more complex geometry calculations needed
        return None

# Example usage
detector = ParkingDetector()
result, lines = detector.detect_parking_lines('parking.jpg')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Parking Line Detection')
plt.axis('off')
plt.show()
```

### 4.3 Document Edge Detection

```python
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

        # Find corner points
        if horizontal and vertical:
            # Simplified corner detection
            corners = find_corners(horizontal, vertical)
            for corner in corners:
                cv2.circle(result, corner, 10, (0, 255, 0), -1)

    # Display
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original'), plt.axis('off')
    plt.subplot(132), plt.imshow(edges, cmap='gray')
    plt.title('Edges'), plt.axis('off')
    plt.subplot(133), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Detected Edges'), plt.axis('off')
    plt.tight_layout()
    plt.show()

    return result

def find_corners(horizontal_lines, vertical_lines):
    """Find corner points"""
    corners = []
    # In practice, calculate line intersections
    return corners

# Example usage
result = detect_document_edges('document.jpg')
```

---

## 5. Performance Optimization

### 5.1 Parameter Optimization

```python
class HoughOptimizer:
    """Hough Transform Parameter Optimizer"""

    def __init__(self, image):
        self.image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def optimize_line_detection(self):
        """Find optimal parameters for line detection"""
        edges = cv2.Canny(self.gray, 50, 150)

        # Test parameter combinations
        thresholds = [50, 100, 150, 200]
        min_lengths = [50, 100, 150]
        max_gaps = [5, 10, 20]

        results = []

        for threshold in thresholds:
            for min_length in min_lengths:
                for max_gap in max_gaps:
                    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold,
                                           minLineLength=min_length,
                                           maxLineGap=max_gap)

                    num_lines = len(lines) if lines is not None else 0
                    results.append({
                        'threshold': threshold,
                        'min_length': min_length,
                        'max_gap': max_gap,
                        'num_lines': num_lines
                    })

        # Sort and display results
        results.sort(key=lambda x: abs(x['num_lines'] - 10))  # Find ~10 lines

        print("=== Top 5 Parameter Combinations ===")
        for i, result in enumerate(results[:5], 1):
            print(f"{i}. threshold={result['threshold']}, "
                  f"min_length={result['min_length']}, "
                  f"max_gap={result['max_gap']}, "
                  f"lines={result['num_lines']}")

        return results[0]

    def optimize_circle_detection(self):
        """Find optimal parameters for circle detection"""
        gray_blur = cv2.medianBlur(self.gray, 5)

        # Test parameter combinations
        param1_values = [30, 50, 100]
        param2_values = [20, 30, 40]

        results = []

        for param1 in param1_values:
            for param2 in param2_values:
                circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, 20,
                                          param1=param1, param2=param2,
                                          minRadius=10, maxRadius=100)

                num_circles = len(circles[0]) if circles is not None else 0
                results.append({
                    'param1': param1,
                    'param2': param2,
                    'num_circles': num_circles
                })

        print("\n=== Circle Detection Parameter Results ===")
        for result in results:
            print(f"param1={result['param1']}, param2={result['param2']}, "
                  f"circles={result['num_circles']}")

        return results

# Example usage
img = cv2.imread('test.jpg')
optimizer = HoughOptimizer(img)
best_params = optimizer.optimize_line_detection()
circle_results = optimizer.optimize_circle_detection()
```

### 5.2 ROI Processing

```python
def hough_with_roi(image_path, roi_rect):
    """Hough Transform with ROI"""
    img = cv2.imread(image_path)
    x, y, w, h = roi_rect

    # Extract ROI
    roi = img[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Line detection only in ROI
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                           minLineLength=30, maxLineGap=10)

    # Draw on original image
    result = img.copy()
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Adjust coordinates to original image
            cv2.line(result, (x+x1, y+y1), (x+x2, y+y2), (0, 0, 255), 2)

    # Display
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original'), plt.axis('off')
    plt.subplot(122), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('ROI Line Detection'), plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
hough_with_roi('image.jpg', roi_rect=(100, 100, 300, 200))
```

### 5.3 Multi-Scale Processing

```python
def multiscale_hough(image_path, scales=[1.0, 0.75, 0.5]):
    """Multi-scale Hough Transform"""
    img = cv2.imread(image_path)

    all_lines = []

    plt.figure(figsize=(15, 5))

    for idx, scale in enumerate(scales, 1):
        # Resize image
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        resized = cv2.resize(img, (width, height))

        # Line detection
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                               minLineLength=30, maxLineGap=10)

        # Adjust coordinates back to original size
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                scaled_line = [int(x1/scale), int(y1/scale),
                             int(x2/scale), int(y2/scale)]
                all_lines.append(scaled_line)

        # Draw results
        result = resized.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        plt.subplot(1, len(scales), idx)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f'Scale: {scale}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    return all_lines

# Example usage
lines = multiscale_hough('complex_scene.jpg')
print(f"Total lines detected across all scales: {len(lines)}")
```

---

## 6. Practice Problems

### Problem 1: Chessboard Detection
Create a program that detects lines in a chessboard image and finds intersection points.

```python
def detect_chessboard(image_path):
    """
    TODO: Implement the following
    1. Detect horizontal and vertical lines
    2. Calculate line intersections
    3. Visualize 64 intersection points (8x8)
    """
    pass
```

### Problem 2: Real-time Lane Detection
Implement real-time lane detection from a video file.

```python
def video_lane_detection(video_path):
    """
    TODO: Implement the following
    1. Read video frame by frame
    2. Detect lanes in each frame
    3. Display smoothed results
    4. Calculate average lane width
    """
    pass
```

### Problem 3: Pizza Slice Counter
Count the number of slices by detecting triangular divisions in a pizza image.

```python
def count_pizza_slices(image_path):
    """
    TODO: Implement the following
    1. Detect lines from center
    2. Calculate angles between lines
    3. Count slices
    4. Visualize results
    """
    pass
```

### Problem 4: Stopwatch Detection
Detect hour, minute, and second hands in an analog clock image and read the time.

```python
def read_analog_clock(image_path):
    """
    TODO: Implement the following
    1. Detect clock circle
    2. Detect three hands
    3. Calculate angles
    4. Convert to time
    """
    pass
```

### Problem 5: Building Window Counter
Count the number of windows in a building image.

```python
def count_building_windows(image_path):
    """
    TODO: Implement the following
    1. Detect horizontal and vertical lines
    2. Find rectangular regions
    3. Filter regions matching window size
    4. Count and visualize
    """
    pass
```

---

## Summary

### Key Concepts
1. **Hough Transform Principles**
   - Conversion from image space to parameter space
   - Voting mechanism
   - Local maxima detection

2. **Line Detection**
   - Standard Hough Transform (HoughLines)
   - Probabilistic Hough Transform (HoughLinesP)
   - Parameter adjustment

3. **Circle Detection**
   - 3-parameter space (x, y, r)
   - Parameter optimization
   - Multiple circle detection

4. **Practical Applications**
   - Lane detection
   - Parking space detection
   - Document edge detection
   - Object counting

5. **Performance Optimization**
   - Parameter optimization
   - ROI processing
   - Multi-scale processing

### Parameter Tuning Guide
- **rho, theta**: Higher resolution = more accurate but slower
- **threshold**: Higher = fewer but stronger lines
- **minLineLength**: Minimum line length threshold
- **maxLineGap**: Maximum gap in line segments
- **param1**: Edge detection threshold
- **param2**: Accumulator threshold

### Important Notes
- Preprocessing is crucial (edge detection, noise removal)
- Parameter values vary greatly by image characteristics
- Real-time processing requires performance optimization
- ROI reduces computational cost
- Multi-scale processing detects objects of various sizes

---

## References
- OpenCV Documentation: https://docs.opencv.org/
- Hough Transform Theory: https://en.wikipedia.org/wiki/Hough_transform
- Lane Detection Tutorial: https://towardsdatascience.com/lane-detection-with-opencv/
