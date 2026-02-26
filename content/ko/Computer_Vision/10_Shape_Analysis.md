# 도형 분석 (Shape Analysis)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 이미지 모멘트(image moments)를 정의하고 면적, 무게중심, 방향 등의 형상 정보를 어떻게 인코딩하는지 설명할 수 있다
2. OpenCV를 사용하여 윤곽선 모멘트(contour moments)에서 무게중심(centroid)과 방향(orientation)을 계산하여 구현할 수 있다
3. 검출된 윤곽선을 감싸는 경계 도형(bounding shapes)(사각형, 원, 타원)을 적용할 수 있다
4. 컨벡스 헐(convex hull)을 계산하고 형상 특성 파악을 위해 볼록성 결함(convexity defects)을 식별할 수 있다
5. 후 모멘트(Hu moments)와 형상 매칭(shape matching) 기법을 사용하여 도형을 비교할 수 있다
6. 여러 기하학적 디스크립터(geometric descriptor)를 결합한 도형 분류 시스템(shape classification system)을 설계할 수 있다

---

## 개요

윤곽선에서 추출한 도형의 특성을 분석하고 분류하는 방법을 학습합니다. 모멘트, 무게중심, 경계 도형, 컨벡스 헐, 형상 매칭 등 다양한 도형 분석 기법을 다룹니다.

도형 분석은 근본적인 인식 문제를 해결합니다. 동일한 물체도 위치, 크기, 회전이 바뀌면 이미지마다 다르게 보입니다. 왼쪽에서 촬영한 동전과 오른쪽에서 촬영한 동전은 다르게 보입니다. 이동(translation), 스케일(scale), 회전(rotation)에 불변인 디스크립터(descriptor)를 계산함으로써, 도형 분석은 장면에서 어떻게 보이든 관계없이 객체를 인식하고 비교할 수 있게 해줍니다.

---

## 목차

1. [이미지 모멘트](#1-이미지-모멘트)
2. [무게중심 계산](#2-무게중심-계산)
3. [경계 사각형](#3-경계-사각형)
4. [최소 외접 도형](#4-최소-외접-도형)
5. [컨벡스 헐](#5-컨벡스-헐)
6. [형상 매칭](#6-형상-매칭)
7. [도형 분류 시스템](#7-도형-분류-시스템)
8. [연습 문제](#8-연습-문제)

---

## 1. 이미지 모멘트

### 모멘트란?

```
Image Moments:
Feature values calculated as weighted averages of image pixel values

Mathematical Definition:
Mij = Σ Σ x^i × y^j × I(x, y)

- M00: Area (0th moment)
- M10, M01: 1st moments (for centroid calculation)
- M20, M02, M11: 2nd moments (orientation, variance)

Applications:
- Area and perimeter calculation
- Centroid
- Orientation
- Ellipse fitting
- Hu Moments - invariant features
```

### cv2.moments() 함수

```python
import cv2
import numpy as np

def calculate_moments(image_path):
    """Calculate image moments"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for i, contour in enumerate(contours):
        # Calculate moments
        M = cv2.moments(contour)

        print(f"Contour {i}:")
        print(f"  M00 (area): {M['m00']:.0f}")
        print(f"  M10: {M['m10']:.0f}")
        print(f"  M01: {M['m01']:.0f}")

        # Centroid
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            print(f"  Centroid: ({cx}, {cy})")

        # Central Moments
        print(f"  mu20: {M['mu20']:.0f}")
        print(f"  mu11: {M['mu11']:.0f}")
        print(f"  mu02: {M['mu02']:.0f}")

        # Normalized Central Moments
        print(f"  nu20: {M['nu20']:.6f}")
        print(f"  nu11: {M['nu11']:.6f}")
        print(f"  nu02: {M['nu02']:.6f}")

calculate_moments('shapes.jpg')
```

### 모멘트 종류

```
Spatial Moments:
m00, m10, m01, m20, m11, m02, m30, m21, m12, m03

┌─────────────────────────────────────────────────────────┐
│  m00 = Σ I(x,y)           → Area (white pixels count)   │
│  m10 = Σ x × I(x,y)       → Sum of x coordinates        │
│  m01 = Σ y × I(x,y)       → Sum of y coordinates        │
└─────────────────────────────────────────────────────────┘

Central Moments:
mu20, mu11, mu02, mu30, mu21, mu12, mu03

┌─────────────────────────────────────────────────────────┐
│  Calculated relative to centroid                        │
│  mu20 = Σ (x - cx)² × I(x,y)                           │
│  → Translation Invariant                                │
└─────────────────────────────────────────────────────────┘

Normalized Central Moments:
nu20, nu11, nu02, nu30, nu21, nu12, nu03

┌─────────────────────────────────────────────────────────┐
│  nuij = muij / m00^((i+j)/2 + 1)                       │
│  → Translation + Scale Invariant                        │
└─────────────────────────────────────────────────────────┘
```

### 휴 모멘트 (Hu Moments)

**모멘트가 형상 인식에 유용한 이유는?** 중심(centroid, M10/M00, M01/M00)을 핀 위에서 형상이 균형을 이루는 점으로 생각하세요 — 물리학의 질량 중심(center of mass)과 직접적인 비유입니다. 중심 모멘트(central moment)는 중심 기준으로 측정되므로 위치에 관계없이 형상의 분포를 설명하여 이동 불변(translation-invariant)입니다. 정규화된 중심 모멘트(M00^((i+j)/2+1)로 나눔)는 정규화 인자가 형상의 면적에 비례하므로 추가로 스케일 불변(scale-invariant)입니다. Hu는 이를 회전에도 불변인 7개의 특정 대수적 표현으로 결합했습니다. 로그 변환 `−sign(h)·log10(|h|)`은 7개의 값이 여러 자릿수에 걸쳐 있어 유클리드 거리에서 비교 가능하도록 로그 스케일을 사용합니다.

```python
import cv2
import numpy as np

def hu_moments_analysis(contour):
    """Calculate and analyze Hu moments"""
    # Regular moments
    M = cv2.moments(contour)

    # Hu's 7 moments are specific algebraic combinations of normalized central
    # moments that cancel out any rotation effect. They are invariant to
    # translation (central moments), scale (normalized), and rotation (Hu's algebra).
    huMoments = cv2.HuMoments(M)

    # The 7 Hu values span ~15 orders of magnitude (e.g., h1≈1e-1, h7≈1e-15).
    # Log scale makes them comparable for Euclidean distance matching;
    # the sign is preserved so the direction of asymmetry is not lost.
    huMoments_log = -np.sign(huMoments) * np.log10(np.abs(huMoments) + 1e-10)

    print("Hu Moments:")
    for i, h in enumerate(huMoments_log.flatten()):
        print(f"  h{i+1}: {h:.4f}")

    return huMoments

def compare_shapes_hu(contour1, contour2):
    """Compare two shapes using Hu moments"""
    hu1 = cv2.HuMoments(cv2.moments(contour1)).flatten()
    hu2 = cv2.HuMoments(cv2.moments(contour2)).flatten()

    # Log scale transformation
    hu1_log = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-10)
    hu2_log = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-10)

    # Euclidean distance
    distance = np.linalg.norm(hu1_log - hu2_log)

    return distance

# Usage example
img = cv2.imread('shapes.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) >= 2:
    dist = compare_shapes_hu(contours[0], contours[1])
    print(f"Shape similarity distance: {dist:.4f}")
    # The smaller the distance, the more similar the shapes
```

---

## 2. 무게중심 계산

### 무게중심 공식

무게중심 공식은 가중 평균(weighted-average) 원리의 직접적인 적용입니다. M10 = Σ x·I(x,y)는 픽셀 밝기로 가중된 x좌표를 누적하고(각 픽셀이 "얼마나 오른쪽으로 당기는지" 합산), M00 = Σ I(x,y)는 총 가중치(면적)입니다. M10 / M00은 픽셀 질량으로 가중된 평균 x위치를 제공합니다. 이는 밀도 I(x,y)를 가진 납작한 판의 질량 중심과 정확히 동일합니다. cy = M01 / M00에도 같은 논리가 적용됩니다.

```
Centroid:
The center of mass of a shape

cx = M10 / M00
cy = M01 / M00

          (x1,y1)
             *
            / \
           /   \
          /  •  \    ← (cx, cy) centroid
         /       \
        *---------*
    (x2,y2)    (x3,y3)

Characteristics:
- Always located inside the shape
- Maintains relative position regardless of rotation or scale
```

### 무게중심 계산 및 시각화

```python
import cv2
import numpy as np

def find_centroids(image_path):
    """Find centroids of all contours"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()
    centroids = []

    for contour in contours:
        # Calculate moments
        M = cv2.moments(contour)

        # Guard against zero area (degenerate contour with no enclosed pixels).
        # m10/m00 is the x-weighted average (center of mass in x);
        # m01/m00 is the y-weighted average. Together they give the centroid.
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append((cx, cy))

            # Draw contour
            cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)

            # Mark centroid
            cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(result, f'({cx},{cy})', (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow('Centroids', result)
    cv2.waitKey(0)

    return centroids

centroids = find_centroids('shapes.jpg')
print(f"Centroid coordinates: {centroids}")
```

### 도형 방향 계산

```python
import cv2
import numpy as np

def calculate_orientation(contour):
    """Calculate the major axis orientation of a shape"""
    M = cv2.moments(contour)

    if M['m00'] == 0:
        return None, None

    # Centroid
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']

    # Calculate orientation using 2nd central moments
    # theta = 0.5 * arctan(2 * mu11 / (mu20 - mu02))
    if (M['mu20'] - M['mu02']) != 0:
        theta = 0.5 * np.arctan2(2 * M['mu11'], (M['mu20'] - M['mu02']))
    else:
        theta = 0

    return (cx, cy), theta

def draw_orientation(image, contour):
    """Display shape orientation with an arrow"""
    result = image.copy()

    center, theta = calculate_orientation(contour)
    if center is None:
        return result

    cx, cy = int(center[0]), int(center[1])

    # Major axis arrow
    length = 50
    dx = int(length * np.cos(theta))
    dy = int(length * np.sin(theta))

    # Contour
    cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)

    # Centroid
    cv2.circle(result, (cx, cy), 5, (255, 0, 0), -1)

    # Direction arrow
    cv2.arrowedLine(result, (cx, cy), (cx + dx, cy + dy),
                    (0, 0, 255), 2, tipLength=0.3)

    # Display angle
    angle_deg = np.degrees(theta)
    cv2.putText(result, f'{angle_deg:.1f} deg', (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return result

# Usage example
img = cv2.imread('elongated_shape.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    result = draw_orientation(img, contours[0])
    cv2.imshow('Orientation', result)
    cv2.waitKey(0)
```

---

## 3. 경계 사각형

### cv2.boundingRect()

```
Bounding Rectangle:
Minimum upright rectangle that completely encloses a contour

    ┌───────────────┐
    │   ╱╲          │
    │  ╱  ╲         │  (x, y): top-left
    │ ╱    ╲        │  w: width
    │ ╲    ╱        │  h: height
    │  ╲  ╱         │
    │   ╲╱          │
    └───────────────┘
```

```python
import cv2
import numpy as np

def bounding_rect_example(image_path):
    """Bounding rectangle example"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()

    for contour in contours:
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Draw rectangle
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display information
        aspect_ratio = w / h
        extent = cv2.contourArea(contour) / (w * h)

        print(f"Position: ({x}, {y})")
        print(f"Size: {w} x {h}")
        print(f"Aspect ratio: {aspect_ratio:.2f}")
        print(f"Extent: {extent:.2f}")  # Actual area / rectangle area ratio

    cv2.imshow('Bounding Rectangle', result)
    cv2.waitKey(0)

bounding_rect_example('shapes.jpg')
```

### cv2.minAreaRect() - 회전된 경계 사각형

```
Rotated Bounding Rectangle:
Smallest rotated rectangle that encloses the shape

            ╱╲
           ╱  ╲
          ╱    ╲
         ╱──────╲
        ╱        ╲
       ╲──────────╱

Return value: ((cx, cy), (w, h), angle)
- (cx, cy): center point
- (w, h): width, height
- angle: rotation angle
```

```python
import cv2
import numpy as np

def min_area_rect_example(image_path):
    """Minimum area rotated rectangle"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()

    for contour in contours:
        # Minimum area rotated rectangle
        rect = cv2.minAreaRect(contour)
        center, size, angle = rect

        print(f"Center: {center}")
        print(f"Size: {size}")
        print(f"Angle: {angle:.1f}")

        # Calculate corner coordinates
        box = cv2.boxPoints(rect)
        box = np.int_(box)

        # Draw rectangle
        cv2.drawContours(result, [box], 0, (0, 255, 0), 2)

        # Mark center point
        cv2.circle(result, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)

    cv2.imshow('Min Area Rect', result)
    cv2.waitKey(0)

min_area_rect_example('rotated_shapes.jpg')
```

---

## 4. 최소 외접 도형

### cv2.minEnclosingCircle()

```python
import cv2
import numpy as np

def min_enclosing_circle(image_path):
    """Minimum enclosing circle"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()

    for contour in contours:
        # Minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Draw circle
        cv2.circle(result, center, radius, (0, 255, 0), 2)
        cv2.circle(result, center, 3, (0, 0, 255), -1)

        # Area ratio (indirect circularity measure)
        contour_area = cv2.contourArea(contour)
        circle_area = np.pi * radius * radius
        fill_ratio = contour_area / circle_area

        print(f"Center: {center}, Radius: {radius}")
        print(f"Fill ratio: {fill_ratio:.2f}")

    cv2.imshow('Min Enclosing Circle', result)
    cv2.waitKey(0)

min_enclosing_circle('shapes.jpg')
```

### cv2.fitEllipse()

```python
import cv2
import numpy as np

def fit_ellipse_example(image_path):
    """Ellipse fitting"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()

    for contour in contours:
        # fitEllipse requires at least 5 points
        if len(contour) >= 5:
            # Fit ellipse
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse

            print(f"Center: {center}")
            print(f"Axes: {axes}")  # (major axis, minor axis)
            print(f"Angle: {angle:.1f}")

            # Draw ellipse
            cv2.ellipse(result, ellipse, (0, 255, 0), 2)

            # Center point
            cv2.circle(result, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)

    cv2.imshow('Fitted Ellipse', result)
    cv2.waitKey(0)

fit_ellipse_example('ellipse_shapes.jpg')
```

### 경계 도형 비교

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_bounding_shapes(image_path):
    """Compare various bounding shapes"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        # 1. Bounding rectangle (blue)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 2. Rotated bounding rectangle (green)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int_(box)
        cv2.drawContours(result, [box], 0, (0, 255, 0), 2)

        # 3. Minimum enclosing circle (red)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        cv2.circle(result, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)

        # 4. Fitted ellipse (yellow)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(result, ellipse, (0, 255, 255), 2)

    # Legend
    cv2.putText(result, 'Blue: Bounding Rect', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(result, 'Green: Min Area Rect', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(result, 'Red: Min Enclosing Circle', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(result, 'Yellow: Fitted Ellipse', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow('Bounding Shapes', result)
    cv2.waitKey(0)

compare_bounding_shapes('shapes.jpg')
```

---

## 5. 컨벡스 헐

### cv2.convexHull()

```
Convex Hull:
Smallest convex polygon that encloses a set of points
→ Like wrapping with a rubber band

       •  •  •
     •        •
   •   Original  •
     •        •
   •  •    •  •

       ┌──────┐
      │       │
     │ Convex  │
    │   Hull   │
     └────────┘
```

```python
import cv2
import numpy as np

def convex_hull_example(image_path):
    """Convex hull example"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()

    for contour in contours:
        # Convex hull
        hull = cv2.convexHull(contour)

        # Original contour (green)
        cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)

        # Convex hull (red)
        cv2.drawContours(result, [hull], 0, (0, 0, 255), 2)

        # Compare areas (Solidity)
        contour_area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / hull_area if hull_area > 0 else 0

        print(f"Contour area: {contour_area:.0f}")
        print(f"Convex hull area: {hull_area:.0f}")
        print(f"Solidity: {solidity:.2f}")
        # Solidity close to 1 indicates convex shape

    cv2.imshow('Convex Hull', result)
    cv2.waitKey(0)

convex_hull_example('star_shape.jpg')
```

### 컨벡시티 결함 (Convexity Defects)

```
Convexity Defects:
Concave parts between contour and convex hull
→ Used for finger detection

            ╱╲
           ╱  ╲
        start  end
          ╲  ╱
           ╲╱ ← far (deepest point)

Return value: [start, end, far, depth]
- start: starting point index
- end: ending point index
- far: deepest point index
- depth: depth (divide by 256 for use)
```

```python
import cv2
import numpy as np

def convexity_defects_example(image_path):
    """Detect convexity defects (finger counting)"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return

    # Select largest contour
    contour = max(contours, key=cv2.contourArea)

    result = img.copy()

    # Convex hull (return indices)
    hull = cv2.convexHull(contour, returnPoints=False)

    # Convexity defects
    defects = cv2.convexityDefects(contour, hull)

    if defects is None:
        return

    # Analyze defects
    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]

        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        # Convert depth (divide by 256)
        depth = d / 256.0

        # Only count if depth is above threshold (space between fingers)
        if depth > 20:
            finger_count += 1

            # Visualization
            cv2.circle(result, far, 5, (0, 0, 255), -1)
            cv2.line(result, start, far, (0, 255, 0), 2)
            cv2.line(result, far, end, (0, 255, 0), 2)

    # Number of fingers = defects count + 1
    print(f"Finger count: {finger_count + 1}")

    # Contour
    cv2.drawContours(result, [contour], 0, (255, 0, 0), 2)

    cv2.imshow('Convexity Defects', result)
    cv2.waitKey(0)

convexity_defects_example('hand.jpg')
```

---

## 6. 형상 매칭

### cv2.matchShapes()

```
Shape Matching:
Compare similarity of two contours (based on Hu moments)

cv2.matchShapes(contour1, contour2, method, parameter)

method:
- cv2.CONTOURS_MATCH_I1: Σ|1/mA - 1/mB|
- cv2.CONTOURS_MATCH_I2: Σ|mA - mB|
- cv2.CONTOURS_MATCH_I3: Σ|mA - mB| / |mA|

Return value: smaller is more similar (0 = identical)
```

```python
import cv2
import numpy as np

def shape_matching_example():
    """Shape matching example"""
    # Create template shape
    template = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(template, (100, 100), 80, 255, -1)

    # Test shapes
    shapes = {
        'circle': cv2.circle(np.zeros((200, 200), dtype=np.uint8),
                             (100, 100), 60, 255, -1),
        'ellipse': cv2.ellipse(np.zeros((200, 200), dtype=np.uint8),
                               (100, 100), (80, 50), 0, 0, 360, 255, -1),
        'square': cv2.rectangle(np.zeros((200, 200), dtype=np.uint8),
                                (30, 30), (170, 170), 255, -1),
    }

    # Template contour
    contours_t, _ = cv2.findContours(template, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    template_contour = contours_t[0]

    print("Shape matching results (lower is more similar):")
    for name, shape in shapes.items():
        contours_s, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        if contours_s:
            match = cv2.matchShapes(template_contour, contours_s[0],
                                     cv2.CONTOURS_MATCH_I1, 0)
            print(f"  {name}: {match:.4f}")

shape_matching_example()
```

### 템플릿 기반 도형 검출

```python
import cv2
import numpy as np

def find_similar_shapes(image_path, template_path, threshold=0.1):
    """Find shapes similar to template"""
    # Load template
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    _, template_bin = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
    template_contours, _ = cv2.findContours(
        template_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    template_contour = max(template_contours, key=cv2.contourArea)

    # Target image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()
    similar_shapes = []

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        # matchShapes compares Hu moments of both contours using method I1,
        # which normalizes each moment before differencing (Σ|1/mA - 1/mB|),
        # making it robust to scale differences between template and candidate.
        # A score of 0 means identical shape; scores below ~0.1 are usually
        # visually very similar.
        match = cv2.matchShapes(
            template_contour, contour, cv2.CONTOURS_MATCH_I1, 0
        )

        if match < threshold:
            similar_shapes.append(contour)
            cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)

            # Display matching score
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(result, f'{match:.3f}', (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    print(f"Number of similar shapes: {len(similar_shapes)}")
    cv2.imshow('Similar Shapes', result)
    cv2.waitKey(0)

    return similar_shapes

# Usage example
find_similar_shapes('shapes.jpg', 'template_circle.jpg', threshold=0.15)
```

---

## 7. 도형 분류 시스템

### 종합 도형 분류기

```python
import cv2
import numpy as np

class ShapeClassifier:
    """Shape classifier"""

    def __init__(self):
        self.shape_names = {
            3: 'Triangle',
            4: 'Rectangle',
            5: 'Pentagon',
            6: 'Hexagon'
        }

    def classify(self, contour):
        """Classify shape from contour"""
        # Basic properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < 100 or perimeter == 0:
            return None, {}

        # Polygon approximation
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        vertices = len(approx)

        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h

        # Circularity
        circularity = 4 * np.pi * area / (perimeter ** 2)

        # Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Classification
        properties = {
            'vertices': vertices,
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'solidity': solidity
        }

        # Shape identification
        # Polygon vertex count comes first because polygons can have circularity
        # close to 1 (a regular hexagon has ~0.91), so checking circularity first
        # would misclassify them as circles.
        if vertices == 3:
            shape = 'Triangle'
        elif vertices == 4:
            if 0.95 <= aspect_ratio <= 1.05:
                shape = 'Square'
            else:
                shape = 'Rectangle'
        elif vertices == 5:
            shape = 'Pentagon'
        elif vertices == 6:
            shape = 'Hexagon'
        elif circularity > 0.85:
            # 0.85 is empirical: a perfect circle scores 1.0; digitization and
            # contour approximation degrade it slightly, so 0.85 is a practical
            # threshold that excludes ellipses and rounded rectangles.
            shape = 'Circle'
        elif 0.6 < circularity < 0.85 and solidity > 0.9:
            shape = 'Ellipse'
        elif solidity < 0.7:
            # Low solidity means the convex hull is much larger than the contour,
            # indicating significant concavities — the hallmark of a star shape.
            shape = 'Star' if vertices > 6 else 'Irregular'
        else:
            shape = f'Polygon-{vertices}'

        return shape, properties

    def process_image(self, image_path):
        """Classify all shapes in image"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        result = img.copy()
        classifications = []

        for contour in contours:
            shape, props = self.classify(contour)
            if shape is None:
                continue

            classifications.append((shape, props))

            # Centroid via moments: placing the label at the center of mass
            # ensures it stays inside the shape regardless of the shape's position
            # or size — more reliable than using the bounding rect center.
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Draw contour
                color = self._get_shape_color(shape)
                cv2.drawContours(result, [contour], 0, color, 2)

                # Label
                cv2.putText(result, shape, (cx - 30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Classified Shapes', result)
        cv2.waitKey(0)

        return classifications

    def _get_shape_color(self, shape):
        """Color for each shape type"""
        colors = {
            'Circle': (0, 0, 255),      # Red
            'Ellipse': (0, 128, 255),   # Orange
            'Triangle': (0, 255, 0),    # Green
            'Square': (255, 0, 0),      # Blue
            'Rectangle': (255, 128, 0), # Sky blue
            'Pentagon': (255, 0, 255),  # Purple
            'Hexagon': (128, 0, 128),   # Purple
            'Star': (0, 255, 255),      # Yellow
        }
        return colors.get(shape, (128, 128, 128))

# Usage example
classifier = ShapeClassifier()
results = classifier.process_image('various_shapes.jpg')

print("\nClassification results:")
for shape, props in results:
    print(f"  {shape}:")
    print(f"    Area: {props['area']:.0f}")
    print(f"    Circularity: {props['circularity']:.2f}")
    print(f"    Vertices: {props['vertices']}")
```

### 실시간 도형 검출

```python
import cv2
import numpy as np

def realtime_shape_detection():
    """Real-time shape detection with webcam"""
    classifier = ShapeClassifier()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphology operations
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            shape, props = classifier.classify(contour)
            if shape is None:
                continue

            # Centroid
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Draw
                cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
                cv2.putText(frame, shape, (cx - 30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('Shape Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# realtime_shape_detection()
```

---

## 8. 연습 문제

### 문제 1: 도형 정렬

이미지에서 검출된 도형들을 면적 기준으로 정렬하고 순위를 표시하세요.

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def rank_shapes_by_area(image_path):
    """Sort shapes by area"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Store with area
    contour_areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            contour_areas.append((contour, area))

    # Sort by area (descending)
    contour_areas.sort(key=lambda x: x[1], reverse=True)

    result = img.copy()

    for rank, (contour, area) in enumerate(contour_areas, 1):
        # Centroid
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Draw
        cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)
        cv2.putText(result, f'#{rank}', (cx - 15, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(result, f'{area:.0f}', (cx - 25, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    cv2.imshow('Ranked Shapes', result)
    cv2.waitKey(0)

rank_shapes_by_area('shapes.jpg')
```

</details>

### 문제 2: 특정 비율의 사각형 찾기

가로세로 비율이 2:1인 사각형만 검출하세요.

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def find_2to1_rectangles(image_path, tolerance=0.2):
    """Find 2:1 aspect ratio rectangles"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()
    found = []

    target_ratio = 2.0

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        # Polygon approximation
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # Check if it has 4 vertices
        if len(approx) != 4:
            continue

        # Check ratio using bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h)

        # Check 2:1 ratio (with tolerance)
        if abs(aspect_ratio - target_ratio) < tolerance:
            found.append(contour)
            cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)

            # Display ratio
            cv2.putText(result, f'{aspect_ratio:.2f}:1', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    print(f"2:1 rectangle count: {len(found)}")
    cv2.imshow('2:1 Rectangles', result)
    cv2.waitKey(0)

find_2to1_rectangles('rectangles.jpg')
```

</details>

### 문제 3: 가장 원형에 가까운 도형 찾기

이미지에서 원형도가 가장 높은 도형을 찾아 표시하세요.

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def find_most_circular(image_path):
    """Find the most circular shape"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_circularity = 0
    best_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < 100 or perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)

        if circularity > best_circularity:
            best_circularity = circularity
            best_contour = contour

    result = img.copy()

    if best_contour is not None:
        # All contours (gray)
        cv2.drawContours(result, contours, -1, (128, 128, 128), 1)

        # Most circular (green)
        cv2.drawContours(result, [best_contour], 0, (0, 255, 0), 3)

        # Display information
        M = cv2.moments(best_contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(result, f'Circularity: {best_circularity:.3f}',
                    (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    print(f"Maximum circularity: {best_circularity:.4f}")
    cv2.imshow('Most Circular', result)
    cv2.waitKey(0)

find_most_circular('shapes.jpg')
```

</details>

### 추천 문제

| 난이도 | 주제 | 설명 |
|--------|------|------|
| ⭐ | 무게중심 | 모든 도형의 무게중심 표시 |
| ⭐⭐ | 방향 | 긴 도형의 주축 방향 표시 |
| ⭐⭐ | 유사도 | matchShapes로 도형 분류 |
| ⭐⭐⭐ | 손가락 세기 | 컨벡시티 결함 활용 |
| ⭐⭐⭐ | 카드 인식 | 사각형 검출 + 분류 |

---

## 다음 단계

- [허프 변환 (Hough Transform)](./11_Hough_Transform.md) - HoughLines, HoughLinesP, HoughCircles

---

## 참고 자료

- [OpenCV Contour Features](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)
- [Contour Properties](https://docs.opencv.org/4.x/d1/d32/tutorial_py_contour_properties.html)
- [Image Moments](https://docs.opencv.org/4.x/d8/d23/classcv_1_1Moments.html)
