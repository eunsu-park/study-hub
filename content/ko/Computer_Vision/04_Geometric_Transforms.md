# 기하학적 변환

## 개요

기하학적 변환(Geometric Transformation)은 이미지의 공간적 위치를 변경하는 작업입니다. 크기 조절, 회전, 이동, 뒤집기, 원근 변환 등이 포함됩니다. 이 문서에서는 OpenCV의 기하학적 변환 함수들과 실제 활용 예제를 학습합니다.

**난이도**: ⭐⭐ (초급-중급)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `cv2.resize()`와 보간법(interpolation) 이해
2. 회전, 뒤집기 함수 사용
3. 어파인 변환 (warpAffine) 활용
4. 원근 변환 (warpPerspective) 활용
5. 문서 스캔/교정 예제 구현

---

## 목차

1. [이미지 크기 조절 - resize()](#1-이미지-크기-조절---resize)
2. [뒤집기와 회전 - flip(), rotate()](#2-뒤집기와-회전---flip-rotate)
3. [어파인 변환 - warpAffine()](#3-어파인-변환---warpaffine)
4. [원근 변환 - warpPerspective()](#4-원근-변환---warpperspective)
5. [문서 교정 예제](#5-문서-교정-예제)
6. [연습 문제](#6-연습-문제)
7. [다음 단계](#7-다음-단계)
8. [참고 자료](#8-참고-자료)

---

## 1. 이미지 크기 조절 - resize()

### 기본 사용법

```python
import cv2

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# dsize argument is (width, height) — the reverse of img.shape which is (height, width)
# This is a common source of bugs: always double-check the axis order
resized = cv2.resize(img, (640, 480))

# fx/fy are scale factors; passing dsize=None tells resize to compute size from them
resized = cv2.resize(img, None, fx=0.5, fy=0.5)  # Reduce to 50%

# Computing the height from ratio ensures the aspect ratio is maintained exactly
# (integer rounding can introduce a 1px error, but this is usually acceptable)
new_width = 800
ratio = new_width / w
new_height = int(h * ratio)
resized = cv2.resize(img, (new_width, new_height))
```

### 보간법 (Interpolation Methods)

```
┌─────────────────────────────────────────────────────────────────┐
│                       Interpolation Comparison                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Method                   Features              Use Cases      │
│   ───────────────────────────────────────────────────────────   │
│   INTER_NEAREST           Nearest neighbor      Fast, low qual. │
│   (Nearest interpolation) Blocky artifacts      Real-time proc. │
│                                                                │
│   INTER_LINEAR            Linear (default)      Balanced choice │
│   (Bilinear interpolation) Smooth results      General resizing │
│                                                                │
│   INTER_AREA              Area interpolation    Best for shrink │
│   (Area-based)             Prevents moiré       Downsampling    │
│                                                                │
│   INTER_CUBIC             Cubic interpolation   Good for enlarg.│
│   (Bicubic)                Smooth and sharp     Quality focus   │
│                                                                │
│   INTER_LANCZOS4          Lanczos interpolation Best quality    │
│   (8x8 neighbors)          Sharpest            Slow speed      │
│                                                                │
│   Recommendations:                                              │
│   - Shrinking: INTER_AREA                                       │
│   - Enlarging: INTER_CUBIC or INTER_LANCZOS4                    │
│   - Real-time: INTER_LINEAR or INTER_NEAREST                    │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 보간법 비교 예제

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Shrink to 10% then scale back up to full size — this stress-tests each
# interpolation method by forcing it to reconstruct information that was discarded
small = cv2.resize(img, None, fx=0.1, fy=0.1)

interpolations = [
    ('NEAREST', cv2.INTER_NEAREST),   # Blocky but fast — good for pixel art or masks
    ('LINEAR', cv2.INTER_LINEAR),     # Default: smooth, fast, good general choice
    ('AREA', cv2.INTER_AREA),         # Best for downscaling — averages pixel blocks
    ('CUBIC', cv2.INTER_CUBIC),       # Better upscaling quality; uses 4x4 neighbors
    ('LANCZOS4', cv2.INTER_LANCZOS4), # Best quality; uses 8x8 neighbors — slowest
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].imshow(img)
axes[0].set_title('Original')

for ax, (name, interp) in zip(axes[1:], interpolations):
    enlarged = cv2.resize(small, img.shape[:2][::-1], interpolation=interp)
    ax.imshow(enlarged)
    ax.set_title(f'{name}')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
```

### 비율 유지 리사이즈 함수

```python
import cv2

def resize_with_aspect_ratio(img, width=None, height=None, inter=cv2.INTER_AREA):
    """Resize while maintaining aspect ratio"""
    h, w = img.shape[:2]

    if width is None and height is None:
        return img

    if width is None:
        ratio = height / h
        new_size = (int(w * ratio), height)
    else:
        ratio = width / w
        new_size = (width, int(h * ratio))

    return cv2.resize(img, new_size, interpolation=inter)


def resize_to_fit(img, max_width, max_height, inter=cv2.INTER_AREA):
    """Fit within maximum size while maintaining aspect ratio"""
    h, w = img.shape[:2]

    ratio_w = max_width / w
    ratio_h = max_height / h
    ratio = min(ratio_w, ratio_h)

    if ratio >= 1:  # Already small enough
        return img

    new_size = (int(w * ratio), int(h * ratio))
    return cv2.resize(img, new_size, interpolation=inter)


# Usage example
img = cv2.imread('large_image.jpg')
img_fit = resize_to_fit(img, 800, 600)
img_width = resize_with_aspect_ratio(img, width=640)
```

---

## 2. 뒤집기와 회전 - flip(), rotate()

### cv2.flip()

```
┌─────────────────────────────────────────────────────────────────┐
│                        flip() Operation                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   flipCode = 1 (horizontal)   flipCode = 0 (vertical)  flipCode = -1│
│                                                                 │
│   Original  Result            Original  Result         Original Result│
│   ┌───┐   ┌───┐           ┌───┐   ┌───┐         ┌───┐  ┌───┐  │
│   │1 2│   │2 1│           │1 2│   │3 4│         │1 2│  │4 3│  │
│   │3 4│   │4 3│           │3 4│   │1 2│         │3 4│  │2 1│  │
│   └───┘   └───┘           └───┘   └───┘         └───┘  └───┘  │
│                                                                 │
│   Left-right flip         Top-bottom flip         Both flips   │
│   (Mirror effect)         (Water reflection)      (180° rotation)│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```python
import cv2

img = cv2.imread('image.jpg')

# Horizontal flip (left-right)
flipped_h = cv2.flip(img, 1)

# Vertical flip (top-bottom)
flipped_v = cv2.flip(img, 0)

# Both directions (equivalent to 180° rotation)
flipped_both = cv2.flip(img, -1)

# Also possible with NumPy
import numpy as np
flipped_h_np = img[:, ::-1]      # Horizontal
flipped_v_np = img[::-1, :]      # Vertical
flipped_both_np = img[::-1, ::-1]  # Both
```

### cv2.rotate()

```python
import cv2

img = cv2.imread('image.jpg')

# 90 degrees clockwise
rotated_90_cw = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# 90 degrees counter-clockwise
rotated_90_ccw = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# 180 degrees
rotated_180 = cv2.rotate(img, cv2.ROTATE_180)

# Check image size changes
print(f"Original: {img.shape}")           # (H, W, C)
print(f"90°: {rotated_90_cw.shape}") # (W, H, C) - swapped
print(f"180°: {rotated_180.shape}")  # (H, W, C) - same
```

### 임의 각도 회전

```python
import cv2

def rotate_image(img, angle, center=None, scale=1.0):
    """Rotate image by arbitrary angle"""
    h, w = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    # Create rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # Apply rotation
    rotated = cv2.warpAffine(img, M, (w, h))

    return rotated


def rotate_image_full(img, angle):
    """Rotate image without cropping (expand canvas)"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new bounds after rotation
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust translation
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(img, M, (new_w, new_h))

    return rotated


# Usage examples
img = cv2.imread('image.jpg')
rotated_30 = rotate_image(img, 30)       # 30° rotation (partially cropped)
rotated_45_full = rotate_image_full(img, 45)  # 45° rotation (fully preserved)
```

---

## 3. 어파인 변환 - warpAffine()

어파인 변환(Affine Transformation)은 카메라 기울기 보정, 이미지 스티칭 전 정렬, 객체 포즈 정규화에 핵심적으로 사용됩니다. 핵심 개념은 회전, 스케일, 이동, 전단(Shear)의 어떠한 조합도 단일 2×3 행렬 곱셈으로 표현할 수 있다는 것입니다. 따라서 여러 변환을 합성할 때는 단순히 행렬을 곱하면 되고, 변환을 하나씩 순서대로 적용할 때의 정밀도 손실도 없습니다.

### 어파인 변환이란?

```
┌─────────────────────────────────────────────────────────────────┐
│                      Affine Transformation                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Affine transformation preserves lines as lines and parallel   │
│   lines as parallel lines                                       │
│                                                                 │
│   Included transformations:                                     │
│   - Translation                                                 │
│   - Rotation                                                    │
│   - Scale                                                       │
│   - Shear                                                       │
│                                                                 │
│   Transformation matrix (2x3):                                  │
│   ┌         ┐   ┌                    ┐                         │
│   │ a  b  tx│   │ scale*cos  -sin  tx│                         │
│   │ c  d  ty│ = │ sin   scale*cos  ty│                         │
│   └         ┘   └                    ┘                         │
│                                                                 │
│   [x']   [a b tx]   [x]                                         │
│   [y'] = [c d ty] × [y]                                         │
│                     [1]                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

행렬의 `[a, b; c, d]` 부분은 회전과 스케일을 인코딩합니다. 순수 회전(각도 θ)의 경우 `a=cos θ, b=-sin θ, c=sin θ, d=cos θ`입니다. `[tx, ty]` 열은 이동 벡터로, 선형 변환이 적용된 후 이미지를 이동시킵니다. 그래서 `getRotationMatrix2D`가 2×3 행렬을 반환하는 것입니다. 즉, 회전과 이동을 하나의 단계로 합칩니다.

### 이동 (Translation)

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# Identity (a=d=1, b=c=0) + pure translation: the simplest affine transform
# tx, ty move the image right/down; negative values move it left/up
tx, ty = 100, 50
M = np.float32([
    [1, 0, tx],
    [0, 1, ty]
])

# Output size (w, h) must be specified explicitly — pixels shifted outside
# this canvas are simply discarded (no automatic padding)
translated = cv2.warpAffine(img, M, (w, h))

cv2.imshow('Original', img)
cv2.imshow('Translated', translated)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 회전 + 스케일

```python
import cv2

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

center = (w // 2, h // 2)
angle = 45   # Positive = counter-clockwise (OpenCV convention)
scale = 0.7  # Combining scale here avoids a separate resize call

# getRotationMatrix2D builds the 2x3 matrix automatically, placing center at origin,
# applying rotation+scale, then translating back — saves manual matrix construction
M = cv2.getRotationMatrix2D(center, angle, scale)

# Keeping (w, h) as output size means corners may be cropped; use rotate_image_full
# (below) when you need the entire rotated image to remain visible
rotated = cv2.warpAffine(img, M, (w, h))
```

### 전단 변환 (Shear)

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# Horizontal shear
shear_x = 0.3
M_shear_x = np.float32([
    [1, shear_x, 0],
    [0, 1, 0]
])
sheared_x = cv2.warpAffine(img, M_shear_x, (int(w + h * shear_x), h))

# Vertical shear
shear_y = 0.3
M_shear_y = np.float32([
    [1, 0, 0],
    [shear_y, 1, 0]
])
sheared_y = cv2.warpAffine(img, M_shear_y, (w, int(h + w * shear_y)))
```

### 3점을 이용한 어파인 변환

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# 3 points from source
src_pts = np.float32([
    [0, 0],      # Top-left
    [w-1, 0],    # Top-right
    [0, h-1]     # Bottom-left
])

# 3 points after transformation
dst_pts = np.float32([
    [50, 50],    # Top-left
    [w-50, 30],  # Top-right
    [30, h-50]   # Bottom-left
])

# Calculate affine transformation matrix
M = cv2.getAffineTransform(src_pts, dst_pts)

# Apply transformation
result = cv2.warpAffine(img, M, (w, h))

# Mark points
for pt in src_pts.astype(int):
    cv2.circle(img, tuple(pt), 5, (0, 0, 255), -1)

for pt in dst_pts.astype(int):
    cv2.circle(result, tuple(pt), 5, (0, 255, 0), -1)
```

---

## 4. 원근 변환 - warpPerspective()

어파인 변환이 평행선을 평행하게 유지하는 반면, 원근 변환(Perspective Transformation)은 카메라의 투영 기하학 전체를 모델링할 수 있습니다. 즉, 평행선이 소실점에서 수렴하는 현상까지 표현합니다. 기울어진 문서를 "펼치거나", 차선 감지를 위해 조감도(Bird's Eye View)를 생성하거나, 비스듬히 찍힌 이미지를 정면으로 정렬하는 데 사용하는 도구입니다.

### 원근 변환이란?

```
┌─────────────────────────────────────────────────────────────────┐
│                       Perspective Transformation                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Perspective transformation converts trapezoid to rectangle    │
│   (or vice versa). Transforms images captured in 3D space as if │
│   viewed from the front                                         │
│                                                                 │
│   Practical applications:                                       │
│   - Document scanning (tilted document → front view)            │
│   - Lane detection (Bird's eye view)                            │
│   - QR code recognition                                         │
│   - Image rectification                                         │
│                                                                 │
│   Transformation matrix (3x3):                                  │
│   ┌             ┐                                               │
│   │ h11 h12 h13 │                                               │
│   │ h21 h22 h23 │                                               │
│   │ h31 h32 h33 │                                               │
│   └             ┘                                               │
│                                                                 │
│   Source (trapezoid)           Result (rectangle)               │
│   ┌─────────────┐         ┌─────────────────┐                   │
│   │ ┌─────────┐ │         │ ┌─────────────┐ │                   │
│   │ │         │ │   ───▶  │ │             │ │                   │
│   │ │ Document│ │         │ │   Document  │ │                   │
│   │ │         │ │         │ │             │ │                   │
│   │ └───────────┘│         │ └─────────────┘ │                   │
│   └─────────────┘         └─────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

3×3 단응사상(Homography) 행렬 H는 소스의 임의 점 (x, y)를 목적지 (x', y')로 동차 좌표계(Homogeneous Coordinates)를 통해 매핑합니다. [x', y', w'] = H·[x, y, 1] 이후 w'로 나눕니다. w'(세 번째 요소)로 나누는 연산이 어파인 변환에서는 불가능한 수렴선 효과를 가능하게 합니다. H는 자유도가 8개(9개 요소 중 스케일 1개 제외)이므로 정확히 4쌍의 점 대응이 필요합니다.

### 4점을 이용한 원근 변환

```python
import cv2
import numpy as np

img = cv2.imread('tilted_document.jpg')
h, w = img.shape[:2]

# Corners must be ordered consistently (e.g., TL, TR, BR, BL) to match dst_pts
# A mislabelled point swaps two corners and produces a twisted "butterfly" warp
src_pts = np.float32([
    [100, 50],    # Top-left
    [500, 80],    # Top-right
    [550, 400],   # Bottom-right
    [50, 380]     # Bottom-left
])

# dst_pts defines a perfect rectangle — this is what makes the output "front-facing"
# The output size (500, 400) should match these rectangle dimensions
dst_pts = np.float32([
    [0, 0],
    [500, 0],
    [500, 400],
    [0, 400]
])

# getPerspectiveTransform solves for the 8 degrees of freedom using the 4 point pairs
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# warpPerspective applies the homography pixel-by-pixel using inverse mapping
# (samples source coordinates for each destination pixel to avoid holes)
result = cv2.warpPerspective(img, M, (500, 400))

# Mark points
img_with_pts = img.copy()
for i, pt in enumerate(src_pts.astype(int)):
    cv2.circle(img_with_pts, tuple(pt), 10, (0, 0, 255), -1)
    cv2.putText(img_with_pts, str(i+1), tuple(pt),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow('Original with points', img_with_pts)
cv2.imshow('Warped', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Bird's Eye View (조감도)

```python
import cv2
import numpy as np

def get_birds_eye_view(img, src_pts, width, height):
    """
    Create bird's eye view using perspective transformation

    Parameters:
    - img: Input image
    - src_pts: 4 points from source (top-left, top-right, bottom-right, bottom-left)
    - width, height: Output image size
    """
    dst_pts = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))

    return warped, M


# Example for lane detection
img = cv2.imread('road.jpg')
h, w = img.shape[:2]

# 4 points of road area (trapezoid)
road_pts = np.float32([
    [w * 0.4, h * 0.6],   # Top-left
    [w * 0.6, h * 0.6],   # Top-right
    [w * 0.9, h * 0.95],  # Bottom-right
    [w * 0.1, h * 0.95]   # Bottom-left
])

birds_eye, M = get_birds_eye_view(img, road_pts, 400, 600)
```

---

## 5. 문서 교정 예제

### 자동 문서 스캔 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                   Document Scan Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input image                                                   │
│       │                                                         │
│       ▼                                                         │
│   Preprocessing (grayscale, blur, edge)                         │
│       │                                                         │
│       ▼                                                         │
│   Contour detection (findContours)                              │
│       │                                                         │
│       ▼                                                         │
│   Rectangle detection (approximate to 4 points with approxPolyDP)│
│       │                                                         │
│       ▼                                                         │
│   Order corners (top-left, top-right, bottom-right, bottom-left)│
│       │                                                         │
│       ▼                                                         │
│   Perspective transformation (warpPerspective)                  │
│       │                                                         │
│       ▼                                                         │
│   Post-processing (binarization, sharpening)                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 구현 코드

```python
import cv2
import numpy as np

def order_points(pts):
    """Order 4 points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype=np.float32)

    # Point with smallest sum = top-left
    # Point with largest sum = bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Point with smallest difference = top-right
    # Point with largest difference = bottom-left
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def four_point_transform(img, pts):
    """Perspective transformation using 4 points"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Measure both top and bottom edges: a tilted document will have different
    # projected widths; taking the max preserves all content without cropping
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    # Same logic for height: left and right edges may project to different lengths
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    # Destination points
    dst = np.float32([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ])

    # Perspective transformation
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    return warped


def find_document(img):
    """Automatically detect document region in image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur before Canny: Gaussian smoothing suppresses pixel-level noise that
    # would otherwise create spurious edges and fragmented contours
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # RETR_EXTERNAL retrieves only outermost contours — sufficient for finding
    # the document border without getting confused by text/graphics inside it
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    # Only examine the 5 largest contours — a document is almost always the
    # biggest rectangular region in the frame
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    doc_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        # 0.02 * peri is the epsilon tolerance: 2% of perimeter — small enough
        # to ignore fine detail, large enough to collapse near-rectangular edges to 4 pts
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:  # Only a 4-vertex polygon is a candidate document
            doc_contour = approx
            break

    return doc_contour


def scan_document(img):
    """Main document scan function"""
    # Save original size
    orig = img.copy()
    ratio = img.shape[0] / 500.0

    # Resize (improve processing speed)
    img = cv2.resize(img, (int(img.shape[1] / ratio), 500))

    # Detect document
    doc_contour = find_document(img)

    if doc_contour is None:
        print("Document not found.")
        return None

    # Adjust coordinates to original size
    doc_contour = doc_contour.reshape(4, 2) * ratio

    # Perspective transformation
    warped = four_point_transform(orig, doc_contour)

    # Post-processing (optional)
    # Grayscale + adaptive binarization
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_binary = cv2.adaptiveThreshold(
        warped_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 10
    )

    return warped, warped_binary


# Usage example
img = cv2.imread('document_photo.jpg')
result_color, result_binary = scan_document(img)

if result_color is not None:
    cv2.imshow('Original', img)
    cv2.imshow('Scanned (Color)', result_color)
    cv2.imshow('Scanned (Binary)', result_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### 수동 4점 선택 (마우스 클릭)

```python
import cv2
import numpy as np

points = []

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Select 4 corners', param)

            if len(points) == 4:
                print("4 points selected! Press 's' to transform.")


def manual_perspective_transform(img):
    """Select 4 points with mouse for perspective transformation"""
    global points
    points = []

    img_display = img.copy()
    cv2.imshow('Select 4 corners', img_display)
    cv2.setMouseCallback('Select 4 corners', click_event, img_display)

    print("Click 4 corners of document clockwise (starting from top-left)")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and len(points) == 4:
            break
        elif key == ord('r'):  # Reset
            points = []
            img_display = img.copy()
            cv2.imshow('Select 4 corners', img_display)
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()

    pts = np.array(points, dtype=np.float32)
    result = four_point_transform(img, pts)

    return result


# Usage example
img = cv2.imread('document.jpg')
result = manual_perspective_transform(img)

if result is not None:
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

---

## 6. 연습 문제

### 연습 1: 배치 리사이즈

폴더 내의 모든 이미지를 가로 800px로 리사이즈하고 (비율 유지), 품질 90%의 JPEG로 저장하는 스크립트를 작성하세요.

```python
# Hint
import os
import glob

def batch_resize(input_folder, output_folder, max_width=800):
    # Use os.listdir or glob.glob
    pass
```

### 연습 2: 이미지 회전 애니메이션

이미지를 0도부터 360도까지 5도씩 회전하면서 애니메이션으로 보여주는 프로그램을 작성하세요. 이미지가 잘리지 않도록 캔버스를 확장하세요.

### 연습 3: 신분증 스캐너

다음 기능을 가진 신분증 스캐너를 구현하세요:
1. 마우스로 4점 선택
2. 원근 변환으로 정면 뷰 생성
3. 표준 신분증 크기(85.6mm x 54mm) 비율로 출력

### 연습 4: 이미지 모자이크

여러 이미지를 받아서 N x M 그리드로 배치하는 함수를 작성하세요. 각 이미지는 동일한 크기로 리사이즈되어야 합니다.

```python
def create_mosaic(images, rows, cols, cell_size=(200, 200)):
    """Arrange images in rows x cols grid"""
    pass
```

### 연습 5: AR 카드 효과

이미지에서 직사각형 카드를 검출하고, 그 위에 다른 이미지를 오버레이하는 간단한 AR 효과를 구현하세요.

```python
# Hint: Use reverse perspective transformation
# 1. Detect card region
# 2. Transform overlay image to fit card region
# 3. Composite with original
```

---

## 7. 다음 단계

[이미지 필터링](./05_Image_Filtering.md)에서 블러, 샤프닝, 커스텀 필터 등 이미지 필터링 기법을 학습합니다!

**다음에 배울 내용**:
- 커널과 컨볼루션 개념
- 블러 필터 (평균, 가우시안, 중앙값, 양방향)
- 샤프닝 필터
- 커스텀 필터 (filter2D)

---

## 8. 참고 자료

### 공식 문서

- [resize() 문서](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d)
- [warpAffine() 문서](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983)
- [warpPerspective() 문서](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87)

### 관련 학습 자료

| 폴더 | 관련 내용 |
|------|----------|
| [색상 공간](./03_Color_Spaces.md) | 색상 변환, 엣지 검출 전처리 |
| [윤곽선 검출 (Contour Detection)](./09_Contours.md) | 문서 영역 검출에 활용 |

### 추가 참고

- [PyImageSearch - 4-point Transform](https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/)
- [OpenCV 보간법 가이드](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121)

