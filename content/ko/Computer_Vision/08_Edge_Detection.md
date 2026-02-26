# 엣지 검출 (Edge Detection)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 이미지 그래디언트(image gradient) 개념과 그것이 밝기 변화를 나타내는 방식을 설명할 수 있다
2. OpenCV를 사용하여 소벨(Sobel), 샤르(Scharr), 라플라시안(Laplacian) 연산자로 엣지 검출(edge detection)을 구현할 수 있다
3. 캐니 엣지 검출(Canny edge detection) 알고리즘을 적용하고 이력 임계값(hysteresis threshold)을 조정할 수 있다
4. 1차 미분 연산자와 2차 미분 연산자의 장단점을 비교할 수 있다
5. 그래디언트 크기(gradient magnitude)와 방향(direction)을 분석하여 엣지 특성을 파악할 수 있다
6. 다양한 이미지 유형에 적합한 엣지 검출 방법을 선택하는 전처리 파이프라인(preprocessing pipeline)을 설계할 수 있다

---

## 개요

엣지(Edge)는 이미지에서 밝기가 급격하게 변하는 영역으로, 객체의 경계나 구조를 나타냅니다. 이 레슨에서는 이미지 그래디언트 개념과 Sobel, Scharr, Laplacian, Canny 등 다양한 엣지 검출 기법을 학습합니다.

---

## 목차

1. [이미지 그래디언트 개념](#1-이미지-그래디언트-개념)
2. [Sobel 연산자](#2-sobel-연산자)
3. [Scharr 연산자](#3-scharr-연산자)
4. [Laplacian 연산자](#4-laplacian-연산자)
5. [Canny 엣지 검출](#5-canny-엣지-검출)
6. [그래디언트 크기와 방향](#6-그래디언트-크기와-방향)
7. [연습 문제](#7-연습-문제)

---

## 1. 이미지 그래디언트 개념

### 그래디언트란?

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

그래디언트 벡터(∂f/∂x, ∂f/∂y)는 항상 밝기가 가장 급격하게 증가하는 방향을 가리킵니다. 마치 물이 언덕을 올라가는 것과 같습니다. 크기 |∇f|는 엣지가 얼마나 날카로운지를, 방향 θ는 밝기가 증가하는 방향(엣지 경계 자체에 수직)을 알려줍니다. 예를 들어, 수직 엣지(왼쪽이 어둡고 오른쪽이 밝은 경우)는 ∂f/∂x가 크고 ∂f/∂y는 거의 0에 가까우므로, θ ≈ 0°이고 그래디언트(gradient)는 수평 방향을 가리킵니다.

### 엣지의 종류

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

### 엣지 검출 파이프라인

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Input    │     │    Noise    │     │  Gradient   │     │    Edge     │
│    Image    │ ──▶ │   Removal   │ ──▶ │ Calculation │ ──▶ │ Extraction  │
│             │     │  (Gaussian) │     │ (Sobel etc) │     │ (Threshold) │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## 2. Sobel 연산자

### 개념

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

### cv2.Sobel() 함수

```python
cv2.Sobel(src, ddepth, dx, dy, ksize=3, scale=1, delta=0)
```

| 파라미터 | 설명 |
|----------|------|
| src | 입력 이미지 |
| ddepth | 출력 이미지 깊이 (cv2.CV_64F 권장) |
| dx | x 방향 미분 차수 (0 또는 1) |
| dy | y 방향 미분 차수 (0 또는 1) |
| ksize | 커널 크기 (1, 3, 5, 7) |
| scale | 스케일 팩터 |
| delta | 결과에 더할 값 |

### 기본 사용법

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

### 그래디언트 크기 계산

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

### 커널 크기에 따른 차이

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

## 3. Scharr 연산자

### 개념

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

### cv2.Scharr() 함수

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

### Sobel에서 Scharr 사용하기

```python
# Use ksize=-1 or ksize=cv2.FILTER_SCHARR in cv2.Sobel()
scharr_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=-1)  # Use Scharr kernel
scharr_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=-1)

# Above code is equivalent to
scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
```

---

## 4. Laplacian 연산자

### 개념

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

### 1차 미분 vs 2차 미분

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

### cv2.Laplacian() 함수

```python
cv2.Laplacian(src, ddepth, ksize=1, scale=1, delta=0)
```

| 파라미터 | 설명 |
|----------|------|
| src | 입력 이미지 |
| ddepth | 출력 이미지 깊이 |
| ksize | 커널 크기 (1, 3, 5, 7) |
| scale | 스케일 팩터 |
| delta | 결과에 더할 값 |

### 기본 사용법

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

## 5. Canny 엣지 검출

### 개념

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

### Canny 알고리즘 상세

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

### cv2.Canny() 함수

```python
cv2.Canny(image, threshold1, threshold2, apertureSize=3, L2gradient=False)
```

| 파라미터 | 설명 |
|----------|------|
| image | 입력 이미지 (그레이스케일) |
| threshold1 | 낮은 임계값 (low) |
| threshold2 | 높은 임계값 (high) |
| apertureSize | Sobel 커널 크기 (3, 5, 7) |
| L2gradient | True: L2 norm, False: L1 norm |

### 기본 사용법

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

### 임계값 튜닝

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

### 자동 임계값 설정

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

### 컬러 이미지에서 Canny

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

## 6. 그래디언트 크기와 방향

### 그래디언트 크기 계산

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

### 그래디언트 방향 시각화

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

### 엣지 검출 알고리즘 비교

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

## 7. 연습 문제

### 문제 1: 적응형 Canny 구현

이미지의 밝기 분포에 따라 자동으로 임계값을 조절하는 Canny 함수를 구현하세요.

<details>
<summary>힌트</summary>

이미지의 중간값(median)을 기준으로 낮은 임계값과 높은 임계값을 계산합니다.

</details>

<details>
<summary>정답 코드</summary>

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

### 문제 2: 방향별 엣지 분리

수평 엣지와 수직 엣지를 분리하여 표시하는 함수를 구현하세요.

<details>
<summary>힌트</summary>

그래디언트 방향을 계산하고, 각도에 따라 수평(0도 근처)과 수직(90도 근처)을 분류합니다.

</details>

<details>
<summary>정답 코드</summary>

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

### 문제 3: 다중 스케일 엣지 검출

여러 스케일에서 엣지를 검출하고 합성하는 함수를 구현하세요.

<details>
<summary>힌트</summary>

다양한 sigma 값으로 Gaussian blur를 적용한 후 Canny를 적용하고, 결과를 합성합니다.

</details>

<details>
<summary>정답 코드</summary>

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

### 추천 문제

| 난이도 | 주제 | 설명 |
|--------|------|------|
| ⭐ | 기본 Canny | 다양한 이미지에 Canny 적용 |
| ⭐⭐ | 임계값 실험 | 트랙바로 최적 임계값 찾기 |
| ⭐⭐ | 전처리 비교 | blur 종류에 따른 엣지 품질 비교 |
| ⭐⭐⭐ | 문서 스캔 | 문서 윤곽선 검출 |
| ⭐⭐⭐ | 동전 검출 | 엣지로 동전 경계 찾기 |

---

## 다음 단계

- [윤곽선 검출 (Contour Detection)](./09_Contours.md) - findContours, drawContours, 계층 구조

---

## 참고 자료

- [OpenCV Edge Detection Tutorial](https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html)
- [Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
- [Image Gradients](https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html)
