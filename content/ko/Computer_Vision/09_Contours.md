# 윤곽선 검출 (Contour Detection)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 윤곽선(contour)이 무엇이며 이진 이미지(binary image)에서 객체 경계를 어떻게 표현하는지 설명할 수 있다
2. 다양한 검색 모드와 근사화 모드로 OpenCV의 findContours()를 사용해 윤곽선 검출을 구현할 수 있다
3. 복잡한 장면에서 외부 윤곽선과 내부 윤곽선을 구분하기 위해 윤곽선 계층 구조(contour hierarchy)를 분석할 수 있다
4. 면적(area), 둘레(perimeter), 경계 상자(bounding box), 모멘트(moments) 등 윤곽선 속성을 계산할 수 있다
5. 형상 표현을 단순화하기 위해 윤곽선 근사화(contour approximation) 방법을 적용할 수 있다
6. 윤곽선 분석(contour analysis)을 사용하여 이미지에서 객체를 카운팅하고 분리하는 워크플로우를 설계할 수 있다

---

## 개요

윤곽선(Contour)은 동일한 색상이나 밝기를 가진 연속적인 점들의 곡선으로, 객체의 형태를 나타냅니다. 이 레슨에서는 findContours()를 사용한 윤곽선 검출, 계층 구조, 근사화, 면적/둘레 계산 등을 학습합니다.

---

## 목차

1. [윤곽선 기초](#1-윤곽선-기초)
2. [findContours() 함수](#2-findcontours-함수)
3. [윤곽선 계층 구조](#3-윤곽선-계층-구조)
4. [윤곽선 그리기와 근사화](#4-윤곽선-그리기와-근사화)
5. [윤곽선 속성 계산](#5-윤곽선-속성-계산)
6. [객체 카운팅과 분리](#6-객체-카운팅과-분리)
7. [연습 문제](#7-연습-문제)

---

## 1. 윤곽선 기초

### 윤곽선이란?

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

### 윤곽선 검출 과정

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

### 기본 예제

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

## 2. findContours() 함수

### 함수 시그니처

```python
contours, hierarchy = cv2.findContours(image, mode, method)
```

| 파라미터 | 설명 |
|----------|------|
| image | 입력 이진 이미지 (8비트 단일 채널) |
| mode | 윤곽선 검색 모드 (RETR_*) |
| method | 윤곽선 근사화 방법 (CHAIN_*) |
| contours | 검출된 윤곽선 리스트 |
| hierarchy | 윤곽선 계층 구조 |

### 검색 모드 (Retrieval Mode)

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

### 근사화 방법 (Approximation Method)

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

### 모드별 예제

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

## 3. 윤곽선 계층 구조

실제 이미지에는 중첩된 형상이 포함됩니다. 도넛은 바깥쪽 링과 안쪽 구멍을 가지고, 문서 스캐너는 텍스트 블록 안의 페이지 경계를 구분해야 합니다. 계층 구조(hierarchy)는 이러한 부모-자식 포함 관계를 포착하므로, 수동으로 공간 교차 테스트를 하지 않고도 "가장 바깥쪽 형상만 가져와" 또는 "객체 #2 안의 구멍 모두 가져와" 같은 쿼리를 수행할 수 있습니다.

### hierarchy 구조

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

### 계층 구조 탐색

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

### 특정 레벨의 윤곽선만 추출

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

## 4. 윤곽선 그리기와 근사화

### cv2.drawContours() 함수

```python
cv2.drawContours(image, contours, contourIdx, color, thickness)
```

| 파라미터 | 설명 |
|----------|------|
| image | 그릴 이미지 |
| contours | 윤곽선 리스트 |
| contourIdx | 그릴 윤곽선 인덱스 (-1: 모두) |
| color | 색상 (B, G, R) |
| thickness | 선 두께 (-1: 채우기) |

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

### cv2.approxPolyDP() - 다각형 근사화

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

### 도형 식별 (꼭짓점 수로)

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

## 5. 윤곽선 속성 계산

### 둘레와 면적

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

### 경계 도형

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

### 컨벡스 헐

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

## 6. 객체 카운팅과 분리

### 객체 카운팅

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

### 객체 분리 및 추출

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

### 특정 모양만 검출

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

## 7. 연습 문제

### 문제 1: 동전 카운터

동전 이미지에서 동전 수를 세고 총 금액을 계산하세요 (크기로 구분).

<details>
<summary>힌트</summary>

동전 크기(면적 또는 반지름)를 기준으로 동전 종류를 분류합니다.

</details>

<details>
<summary>정답 코드</summary>

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

### 문제 2: 문서 사각형 검출

이미지에서 문서(종이)의 윤곽선을 찾고 4개의 꼭짓점을 반환하세요.

<details>
<summary>힌트</summary>

가장 큰 4각형 윤곽선을 찾습니다. approxPolyDP로 4개 점으로 근사화합니다.

</details>

<details>
<summary>정답 코드</summary>

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

### 문제 3: 빈 공간 검출

이진 이미지에서 구멍(빈 공간)의 수를 세세요.

<details>
<summary>힌트</summary>

RETR_CCOMP 또는 RETR_TREE를 사용하여 내부 윤곽선(구멍)을 찾습니다.

</details>

<details>
<summary>정답 코드</summary>

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

### 추천 문제

| 난이도 | 주제 | 설명 |
|--------|------|------|
| ⭐ | 기본 검출 | findContours로 객체 개수 세기 |
| ⭐⭐ | 면적 필터 | 특정 크기 범위의 객체만 검출 |
| ⭐⭐ | 도형 분류 | 삼각형, 사각형, 원 구분 |
| ⭐⭐⭐ | 문서 스캐너 | 문서 검출 후 투시 변환 |
| ⭐⭐⭐ | 손가락 카운터 | 컨벡시티 결함으로 손가락 세기 |

---

## 다음 단계

- [도형 분석 (Shape Analysis)](./10_Shape_Analysis.md) - moments, boundingRect, convexHull, matchShapes

---

## 참고 자료

- [OpenCV Contour Features](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)
- [Contour Hierarchy](https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html)
- [Contours in OpenCV](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
