# Contour Detection

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what contours are and how they represent object boundaries in binary images
2. Implement contour detection using OpenCV's findContours() with different retrieval and approximation modes
3. Analyze contour hierarchy to distinguish outer and inner contours in complex scenes
4. Calculate contour properties such as area, perimeter, bounding box, and moments
5. Apply contour approximation methods to simplify shape representations
6. Design a workflow to count and separate objects in an image using contour analysis

---

## Overview

A contour is a curve of continuous points having the same color or brightness, representing the shape of an object. This lesson covers contour detection using findContours(), hierarchy structure, approximation, area/perimeter calculation, and more.

---

## Table of Contents

1. [Contour Basics](#1-contour-basics)
2. [findContours() Function](#2-findcontours-function)
3. [Contour Hierarchy](#3-contour-hierarchy)
4. [Drawing and Approximating Contours](#4-drawing-and-approximating-contours)
5. [Calculating Contour Properties](#5-calculating-contour-properties)
6. [Object Counting and Separation](#6-object-counting-and-separation)
7. [Exercises](#7-exercises)

---

## 1. Contour Basics

### What is a Contour?

```
Contour:
- A curve of continuous points with the same color/brightness
- Represents object boundaries
- Extracted from binary images

Original Image     Binarization      Contour Detection
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  ┌───┐      │     │  ■■■■■      │     │  ┌───┐      │
│  │ ● │      │     │  ■■■■■      │     │  │   │      │
│  └───┘      │ ──▶ │  ■■■■■      │ ──▶ │  └───┘      │
│        ┌──┐ │     │        ■■■ │     │        ┌──┐ │
│        └──┘ │     │        ■■■ │     │        └──┘ │
└─────────────┘     └─────────────┘     └─────────────┘
                      (White regions)    (Boundaries only)
```

### Contour Detection Process

```
1. Read image
      │
      ▼
2. Convert to grayscale
      │
      ▼
3. Binarization (threshold)
      │
      ▼
4. Detect contours (findContours)
      │
      ▼
5. Analyze/draw contours
```

### Basic Example

```python
import cv2
import numpy as np

# Read image
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Binarization
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Detect contours
# RETR_EXTERNAL: only retrieves the outermost contours — ideal when objects
# don't have holes and you want to avoid counting inner boundaries as separate objects.
# For nested shapes (e.g., donut, letter "O"), use RETR_CCOMP or RETR_TREE instead.
# CHAIN_APPROX_SIMPLE: compresses collinear points, storing only endpoints.
# A rectangle's 4 sides collapse from ~hundreds of pixels to just 4 points,
# saving significant memory and speeding up downstream processing.
contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,      # External contours only
    cv2.CHAIN_APPROX_SIMPLE  # Compression
)

print(f"Number of detected contours: {len(contours)}")

# Draw contours
result = img.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

cv2.imshow('Contours', result)
cv2.waitKey(0)
```

---

## 2. findContours() Function

### Function Signature

```python
contours, hierarchy = cv2.findContours(image, mode, method)
```

| Parameter | Description |
|----------|------|
| image | Input binary image (8-bit single channel) |
| mode | Contour retrieval mode (RETR_*) |
| method | Contour approximation method (CHAIN_*) |
| contours | List of detected contours |
| hierarchy | Contour hierarchy structure |

### Retrieval Mode

```
┌────────────────────────────────────────────────────────────────────┐
│                         RETR_EXTERNAL                              │
├────────────────────────────────────────────────────────────────────┤
│  Detect only outermost contours                                    │
│                                                                    │
│  ┌──────────────┐                                                  │
│  │  ┌────────┐  │   → Detect only outer rectangle                 │
│  │  │ ┌────┐ │  │                                                  │
│  │  │ └────┘ │  │                                                  │
│  │  └────────┘  │                                                  │
│  └──────────────┘                                                  │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                           RETR_LIST                                │
├────────────────────────────────────────────────────────────────────┤
│  Detect all contours, no hierarchy (same level)                    │
│                                                                    │
│  ┌──────────────┐                                                  │
│  │  ┌────────┐  │   → Detect all 3, no parent-child relationship  │
│  │  │ ┌────┐ │  │                                                  │
│  │  │ └────┘ │  │                                                  │
│  │  └────────┘  │                                                  │
│  └──────────────┘                                                  │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                           RETR_CCOMP                               │
├────────────────────────────────────────────────────────────────────┤
│  2-level hierarchy                                                 │
│  - Level 1: Outer contours                                         │
│  - Level 2: Holes (inner contours)                                 │
│                                                                    │
│  ┌──────────────┐   Level 1 (outer)                                │
│  │  ┌────────┐  │   Level 2 (holes)                                │
│  │  │ ■■■■■■ │  │   (White areas inside are Level 2)               │
│  │  └────────┘  │                                                  │
│  └──────────────┘                                                  │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                           RETR_TREE                                │
├────────────────────────────────────────────────────────────────────┤
│  Complete hierarchy (parent-child relationship)                    │
│                                                                    │
│  ┌──────────────┐   Level 0 (outermost)                            │
│  │  ┌────────┐  │   Level 1                                        │
│  │  │ ┌────┐ │  │   Level 2                                        │
│  │  │ │ ■■ │ │  │   Level 3                                        │
│  │  │ └────┘ │  │                                                  │
│  │  └────────┘  │                                                  │
│  └──────────────┘                                                  │
└────────────────────────────────────────────────────────────────────┘
```

### Approximation Method

```
┌────────────────────────────────────────────────────────────────────┐
│                      CHAIN_APPROX_NONE                             │
├────────────────────────────────────────────────────────────────────┤
│  Store all contour points                                          │
│                                                                    │
│      • • • • • •                                                   │
│    •           •    → Store all boundary pixels                   │
│    •           •       High memory usage                          │
│    •           •                                                   │
│      • • • • • •                                                   │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                     CHAIN_APPROX_SIMPLE                            │
├────────────────────────────────────────────────────────────────────┤
│  Store only endpoints of straight segments (compression)           │
│                                                                    │
│      •         •                                                   │
│                      → Store only 4 vertices                      │
│                         Memory efficient                          │
│                                                                    │
│      •         •                                                   │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                    CHAIN_APPROX_TC89_L1                            │
│                    CHAIN_APPROX_TC89_KCOS                          │
├────────────────────────────────────────────────────────────────────┤
│  Teh-Chin chain approximation algorithm                            │
│  → More aggressive compression                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Mode Examples

```python
import cv2
import numpy as np

def compare_retrieval_modes(image_path):
    """Compare contour retrieval modes"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    modes = [
        (cv2.RETR_EXTERNAL, 'RETR_EXTERNAL'),
        (cv2.RETR_LIST, 'RETR_LIST'),
        (cv2.RETR_CCOMP, 'RETR_CCOMP'),
        (cv2.RETR_TREE, 'RETR_TREE')
    ]

    for mode, name in modes:
        contours, hierarchy = cv2.findContours(
            binary.copy(),
            mode,
            cv2.CHAIN_APPROX_SIMPLE
        )

        result = img.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        print(f"{name}: {len(contours)} contours")
        cv2.imshow(name, result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

compare_retrieval_modes('nested_shapes.jpg')
```

---

## 3. Contour Hierarchy

Real images contain nested shapes — a donut has an outer ring and an inner hole; a document scanner needs to distinguish the page boundary from text blocks inside it. Hierarchy captures these parent-child containment relationships so you can query "give me only the outermost shape" or "give me all holes inside object #2" without resorting to manual spatial intersection tests.

### hierarchy Structure

```
hierarchy[i] = [Next, Previous, First_Child, Parent]

Next:        Index of next contour at same level (-1: none)
Previous:    Index of previous contour at same level (-1: none)
First_Child: Index of first child contour (-1: none)
Parent:      Index of parent contour (-1: none)

Example:
┌───────────────────────────────────┐
│ ┌─────────────┐ ┌─────────────┐  │
│ │   ┌───┐     │ │             │  │
│ │   │ A │     │ │      B      │  │
│ │   └───┘     │ │             │  │
│ │      C      │ │             │  │
│ └─────────────┘ └─────────────┘  │
│                  D                │
└───────────────────────────────────┘

RETR_TREE result:
Index 0 (D): Next=-1, Prev=-1, Child=1, Parent=-1  (outermost)
Index 1 (C): Next=2,  Prev=-1, Child=3, Parent=0
Index 2 (B): Next=-1, Prev=1,  Child=-1, Parent=0
Index 3 (A): Next=-1, Prev=-1, Child=-1, Parent=1
```

### Traversing Hierarchy

```python
import cv2
import numpy as np

def analyze_hierarchy(image_path):
    """Analyze contour hierarchy"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # RETR_TREE is chosen here because we want the full parent-child tree,
    # not just two levels — essential when shapes are nested more than one level deep.
    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        print("No contours found.")
        return

    # OpenCV returns hierarchy as shape (1, N, 4); squeeze the leading dimension
    # so we can index it as hierarchy[i] = [next, prev, first_child, parent].
    hierarchy = hierarchy[0]  # (1, N, 4) -> (N, 4)

    print("Hierarchy analysis:")
    print("-" * 50)

    for i, h in enumerate(hierarchy):
        next_c, prev_c, first_child, parent = h

        # Calculate level
        level = 0
        p = parent
        while p != -1:
            level += 1
            p = hierarchy[p][3]  # Parent's parent

        indent = "  " * level
        print(f"{indent}Contour {i}:")
        print(f"{indent}  Level: {level}")
        print(f"{indent}  Parent: {parent}")
        print(f"{indent}  Child: {first_child}")
        print(f"{indent}  Area: {cv2.contourArea(contours[i]):.0f}")

analyze_hierarchy('nested_shapes.jpg')
```

### Extract Contours at Specific Level

```python
import cv2
import numpy as np

def get_contours_at_level(contours, hierarchy, level):
    """Return contours at specific level only"""
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    result = []

    for i in range(len(contours)):
        # Calculate current contour's level
        current_level = 0
        parent = hierarchy[i][3]
        while parent != -1:
            current_level += 1
            parent = hierarchy[parent][3]

        if current_level == level:
            result.append(contours[i])

    return result

def get_outer_contours(contours, hierarchy):
    """Return only outermost contours (those without parent)"""
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    result = []

    for i in range(len(contours)):
        if hierarchy[i][3] == -1:  # No parent
            result.append(contours[i])

    return result

def get_inner_contours(contours, hierarchy, parent_idx):
    """Return child (inner) contours of specific contour"""
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    result = []

    # First child
    child = hierarchy[parent_idx][2]

    while child != -1:
        result.append(contours[child])
        child = hierarchy[child][0]  # Next sibling

    return result

# Usage example
img = cv2.imread('nested.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(
    binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)

# Level 0 contours
level0 = get_contours_at_level(contours, hierarchy, 0)

# Outermost contours
outer = get_outer_contours(contours, hierarchy)

result = img.copy()
cv2.drawContours(result, outer, -1, (0, 255, 0), 2)
cv2.imshow('Outer Contours', result)
cv2.waitKey(0)
```

---

## 4. Drawing and Approximating Contours

### cv2.drawContours() Function

```python
cv2.drawContours(image, contours, contourIdx, color, thickness)
```

| Parameter | Description |
|----------|------|
| image | Image to draw on |
| contours | List of contours |
| contourIdx | Index of contour to draw (-1: all) |
| color | Color (B, G, R) |
| thickness | Line thickness (-1: fill) |

```python
import cv2
import numpy as np

def draw_contours_examples(image, contours):
    """Draw contours in various ways"""

    # Draw all contours
    result1 = image.copy()
    cv2.drawContours(result1, contours, -1, (0, 255, 0), 2)

    # Draw specific contour only
    result2 = image.copy()
    if len(contours) > 0:
        cv2.drawContours(result2, contours, 0, (255, 0, 0), 3)

    # Fill contours
    result3 = image.copy()
    cv2.drawContours(result3, contours, -1, (0, 0, 255), -1)

    # Each contour different color
    result4 = image.copy()
    for i, contour in enumerate(contours):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.drawContours(result4, [contour], 0, color, 2)

    return result1, result2, result3, result4
```

### cv2.approxPolyDP() - Polygon Approximation

```
Douglas-Peucker Algorithm:
Approximate contour with fewer points

epsilon (precision):
- Smaller: Closer to original (more points)
- Larger: Simplified (fewer points)

Example:
Original (many points)   epsilon=0.01         epsilon=0.05
      •  •  •                 •                     •
   •        •              •     •                •   •
  •          •            •       •              •     •
  •          •             •     •                  •
   •        •               •   •
      •  •  •                 •                     •
```

```python
import cv2
import numpy as np

def approximate_contour(contour, epsilon_ratio=0.02):
    """
    Polygon approximation of contour
    epsilon_ratio: Allowed error ratio relative to perimeter
    """
    # Calculate contour perimeter
    perimeter = cv2.arcLength(contour, True)

    # epsilon = perimeter * ratio
    epsilon = epsilon_ratio * perimeter

    # Approximation
    approx = cv2.approxPolyDP(contour, epsilon, True)

    return approx

def compare_approximations(image, contour):
    """Compare approximations with various epsilon values"""
    epsilons = [0.001, 0.01, 0.02, 0.05, 0.1]

    for eps in epsilons:
        result = image.copy()
        approx = approximate_contour(contour, eps)

        cv2.drawContours(result, [approx], 0, (0, 255, 0), 2)

        # Mark vertices
        for point in approx:
            x, y = point[0]
            cv2.circle(result, (x, y), 5, (0, 0, 255), -1)

        cv2.putText(result, f'epsilon={eps}, points={len(approx)}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow(f'Approximation {eps}', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage example
img = cv2.imread('shape.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    compare_approximations(img, contours[0])
```

### Shape Identification (by vertex count)

```python
import cv2
import numpy as np

def identify_shape(contour):
    """Identify shape by vertex count"""
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    vertices = len(approx)

    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        # Distinguish square vs rectangle
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            return "Square"
        else:
            return "Rectangle"
    elif vertices == 5:
        return "Pentagon"
    elif vertices == 6:
        return "Hexagon"
    elif vertices > 6:
        # Check if circular
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity > 0.8:
            return "Circle"
        else:
            return f"Polygon ({vertices} vertices)"
    else:
        return "Unknown"

def label_shapes(image_path):
    """Identify and label all shapes in image"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()

    for contour in contours:
        # Ignore very small contours
        if cv2.contourArea(contour) < 100:
            continue

        # Identify shape
        shape = identify_shape(contour)

        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Draw contour
        cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)

        # Display label
        cv2.putText(result, shape, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Shapes', result)
    cv2.waitKey(0)

label_shapes('shapes.jpg')
```

---

## 5. Calculating Contour Properties

### Perimeter and Area

```python
import cv2
import numpy as np

def contour_properties(contour):
    """Calculate basic contour properties"""

    # Area
    area = cv2.contourArea(contour)

    # Perimeter (closed=True: closed curve)
    perimeter = cv2.arcLength(contour, True)

    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    bounding_area = w * h

    # Area ratio (Extent)
    extent = area / bounding_area if bounding_area > 0 else 0

    # Circularity
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

    # Convex hull
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    # Solidity
    solidity = area / hull_area if hull_area > 0 else 0

    return {
        'area': area,
        'perimeter': perimeter,
        'extent': extent,
        'circularity': circularity,
        'solidity': solidity
    }

# Usage example
img = cv2.imread('shape.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i, contour in enumerate(contours):
    props = contour_properties(contour)
    print(f"Contour {i}:")
    print(f"  Area: {props['area']:.0f}")
    print(f"  Perimeter: {props['perimeter']:.1f}")
    print(f"  Extent: {props['extent']:.2f}")
    print(f"  Circularity: {props['circularity']:.2f}")
    print(f"  Solidity: {props['solidity']:.2f}")
```

### Bounding Shapes

```python
import cv2
import numpy as np

def bounding_shapes(image, contour):
    """Various bounding shapes for contour"""
    result = image.copy()

    # 1. Bounding Rectangle
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 2. Rotated Rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int_(box)
    cv2.drawContours(result, [box], 0, (255, 0, 0), 2)

    # 3. Minimum Enclosing Circle
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    cv2.circle(result, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)

    # 4. Fitting Ellipse
    if len(contour) >= 5:  # Minimum 5 points required
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(result, ellipse, (255, 255, 0), 2)

    # 5. Fitting Line
    rows, cols = image.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(result, (cols-1, righty), (0, lefty), (0, 255, 255), 2)

    return result

# Usage example
img = cv2.imread('shape.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    result = bounding_shapes(img, contours[0])
    cv2.imshow('Bounding Shapes', result)
    cv2.waitKey(0)
```

### Convex Hull

```
Convex Hull:
The smallest convex polygon that encloses a set of points

      •  •
    •      •          ┌──────────┐
  •          •   →   │          │
    •  •   •         │          │
        • •          └──────────┘
   Original contour    Convex hull

Convexity Defects:
Deepest points between contour and convex hull
→ Used for finger detection etc.
```

```python
import cv2
import numpy as np

def convex_hull_analysis(image, contour):
    """Convex hull analysis"""
    result = image.copy()

    # Calculate convex hull
    hull = cv2.convexHull(contour)

    # Original contour
    cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)

    # Convex hull
    cv2.drawContours(result, [hull], 0, (0, 0, 255), 2)

    # Convexity defects (useful for finger detection etc.)
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if len(hull_indices) > 3 and len(contour) > 3:
        defects = cv2.convexityDefects(contour, hull_indices)

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                # Display only defects with certain depth
                if d / 256 > 10:  # Depth threshold
                    cv2.circle(result, far, 5, (255, 0, 255), -1)
                    cv2.line(result, start, far, (255, 0, 255), 1)
                    cv2.line(result, far, end, (255, 0, 255), 1)

    return result

# Usage example (hand image)
img = cv2.imread('hand.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # Select largest contour
    largest = max(contours, key=cv2.contourArea)
    result = convex_hull_analysis(img, largest)
    cv2.imshow('Convex Hull', result)
    cv2.waitKey(0)
```

---

## 6. Object Counting and Separation

### Object Counting

```python
import cv2
import numpy as np

def count_objects(image_path, min_area=100):
    """Count objects in image"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive binarization
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Remove noise with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Detect contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Area filtering removes noise contours caused by thresholding artifacts.
    # Without this step, every small speckle becomes a counted "object."
    # The threshold (min_area) should be set relative to the smallest real object
    # you expect — not a fixed constant, as image resolution and scale vary.
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    result = img.copy()
    for i, contour in enumerate(valid_contours):
        # Centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Display number
            cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)
            cv2.putText(result, str(i + 1), (cx - 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    print(f"Detected objects: {len(valid_contours)}")

    cv2.imshow('Counted Objects', result)
    cv2.waitKey(0)

    return len(valid_contours)

# Coin counting example
count_objects('coins.jpg', min_area=500)
```

### Object Separation and Extraction

```python
import cv2
import numpy as np

def extract_objects(image_path, output_dir='objects/'):
    """Separate and save individual objects"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    objects = []
    for i, contour in enumerate(contours):
        # Ignore very small objects
        if cv2.contourArea(contour) < 100:
            continue

        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Add padding
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)

        # Extract object region
        roi = img[y1:y2, x1:x2].copy()
        objects.append(roi)

        # Save
        cv2.imwrite(f'{output_dir}object_{i:03d}.jpg', roi)

    print(f"Extracted {len(objects)} objects")
    return objects

# Usage example
objects = extract_objects('multiple_objects.jpg')
```

### Detect Specific Shapes Only

```python
import cv2
import numpy as np

def find_circles(image_path):
    """Detect only circular objects"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()
    circles = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        # Calculate circularity
        circularity = 4 * np.pi * area / (perimeter ** 2)

        # Consider as circle if circularity >= 0.8
        if circularity > 0.8 and area > 100:
            circles.append(contour)
            cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)

            # Mark center point
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)

    print(f"Circular objects: {len(circles)}")
    cv2.imshow('Circles', result)
    cv2.waitKey(0)

    return circles

def find_rectangles(image_path):
    """Detect only rectangular objects"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()
    rectangles = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue

        # Polygon approximation
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # Rectangle if 4 vertices
        if len(approx) == 4:
            rectangles.append(contour)
            cv2.drawContours(result, [approx], 0, (0, 255, 0), 2)

    print(f"Rectangular objects: {len(rectangles)}")
    cv2.imshow('Rectangles', result)
    cv2.waitKey(0)

    return rectangles

# Usage example
find_circles('shapes.jpg')
find_rectangles('shapes.jpg')
```

---

## 7. Exercises

### Problem 1: Coin Counter

Count coins in a coin image and calculate total amount (distinguish by size).

<details>
<summary>Hint</summary>

Classify coin types based on coin size (area or radius).

</details>

<details>
<summary>Solution Code</summary>

```python
import cv2
import numpy as np

def count_coins_by_size(image_path):
    """Classify and count coins by size"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Canny edge detection + closing operation
    edges = cv2.Canny(blurred, 30, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Detect contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()

    # Classify by size (radius-based)
    small_coins = []   # 10 won
    medium_coins = []  # 50 won
    large_coins = []   # 100 won

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # Ignore noise
            continue

        # Minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)

        # Check circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.7:  # Not circular
                continue

        # Classify by size (example thresholds)
        if radius < 30:
            small_coins.append((int(x), int(y), int(radius)))
            color = (255, 0, 0)  # Blue - 10 won
        elif radius < 40:
            medium_coins.append((int(x), int(y), int(radius)))
            color = (0, 255, 0)  # Green - 50 won
        else:
            large_coins.append((int(x), int(y), int(radius)))
            color = (0, 0, 255)  # Red - 100 won

        cv2.circle(result, (int(x), int(y)), int(radius), color, 2)

    # Output results
    total = (len(small_coins) * 10 +
             len(medium_coins) * 50 +
             len(large_coins) * 100)

    print(f"10 won: {len(small_coins)}")
    print(f"50 won: {len(medium_coins)}")
    print(f"100 won: {len(large_coins)}")
    print(f"Total: {total} won")

    cv2.imshow('Coins', result)
    cv2.waitKey(0)

count_coins_by_size('coins.jpg')
```

</details>

### Problem 2: Document Rectangle Detection

Find document (paper) contour in image and return 4 vertices.

<details>
<summary>Hint</summary>

Find the largest 4-sided contour. Approximate to 4 points using approxPolyDP.

</details>

<details>
<summary>Solution Code</summary>

```python
import cv2
import numpy as np

def find_document(image_path):
    """Find 4 vertices of document area"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Detect contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    document_corners = None

    for contour in contours[:5]:  # Check top 5 only
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # If 4 vertices, it's a document
        if len(approx) == 4:
            document_corners = approx
            break

    if document_corners is not None:
        result = img.copy()
        cv2.drawContours(result, [document_corners], 0, (0, 255, 0), 3)

        # Mark vertices
        for point in document_corners:
            x, y = point[0]
            cv2.circle(result, (x, y), 10, (0, 0, 255), -1)

        cv2.imshow('Document', result)
        cv2.waitKey(0)

        return document_corners.reshape(4, 2)
    else:
        print("Document not found.")
        return None

corners = find_document('document.jpg')
if corners is not None:
    print("Document vertices:", corners)
```

</details>

### Problem 3: Detect Empty Spaces

Count the number of holes (empty spaces) in a binary image.

<details>
<summary>Hint</summary>

Use RETR_CCOMP or RETR_TREE to find inner contours (holes).

</details>

<details>
<summary>Solution Code</summary>

```python
import cv2
import numpy as np

def count_holes(image_path):
    """Count holes inside objects"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # RETR_CCOMP: 2-level hierarchy (outer + holes)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        return 0

    hierarchy = hierarchy[0]

    result = img.copy()
    holes = []

    for i, h in enumerate(hierarchy):
        # Contour with parent = hole
        if h[3] != -1:  # Has parent
            area = cv2.contourArea(contours[i])
            if area > 50:  # Ignore noise
                holes.append(contours[i])
                cv2.drawContours(result, [contours[i]], 0, (0, 0, 255), 2)

    print(f"Number of holes: {len(holes)}")

    cv2.imshow('Holes', result)
    cv2.waitKey(0)

    return len(holes)

count_holes('donut.jpg')
```

</details>

### Recommended Problems

| Difficulty | Topic | Description |
|--------|------|------|
| ⭐ | Basic Detection | Count objects with findContours |
| ⭐⭐ | Area Filter | Detect only objects in specific size range |
| ⭐⭐ | Shape Classification | Distinguish triangles, rectangles, circles |
| ⭐⭐⭐ | Document Scanner | Detect document then perspective transform |
| ⭐⭐⭐ | Finger Counter | Count fingers using convexity defects |

---

## Next Steps

- [10_Shape_Analysis.md](./10_Shape_Analysis.md) - moments, boundingRect, convexHull, matchShapes

---

## References

- [OpenCV Contour Features](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)
- [Contour Hierarchy](https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html)
- [Contours in OpenCV](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
