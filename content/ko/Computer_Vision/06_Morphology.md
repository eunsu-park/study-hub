# 모폴로지 연산

## 개요

모폴로지 연산(Morphological Operations)은 이진 이미지나 그레이스케일 이미지의 형태를 기반으로 하는 연산입니다. 주로 노이즈 제거, 객체 분리, 홀 채우기 등에 사용됩니다. 이 문서에서는 구조 요소의 개념부터 다양한 모폴로지 연산의 활용까지 학습합니다.

**난이도**: ⭐⭐ (초급-중급)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 구조 요소(Structuring Element) 이해
2. 침식(Erosion)과 팽창(Dilation) 연산
3. 열기(Opening)와 닫기(Closing) 연산
4. 그래디언트, 탑햇, 블랙햇 연산
5. 노이즈 제거 및 객체 분리 응용

---

## 목차

1. [모폴로지 연산 개요](#1-모폴로지-연산-개요)
2. [구조 요소 - getStructuringElement()](#2-구조-요소---getstructuringelement)
3. [침식 - erode()](#3-침식---erode)
4. [팽창 - dilate()](#4-팽창---dilate)
5. [열기와 닫기 - morphologyEx()](#5-열기와-닫기---morphologyex)
6. [그래디언트, 탑햇, 블랙햇](#6-그래디언트-탑햇-블랙햇)
7. [실전 응용](#7-실전-응용)
8. [연습 문제](#8-연습-문제)
9. [다음 단계](#9-다음-단계)
10. [참고 자료](#10-참고-자료)

---

## 1. 모폴로지 연산 개요

가우시안 블러와 같은 픽셀 단위 필터는 모든 픽셀을 동등하게 취급합니다. 하지만 현실에서는 객체 경계를 흐리게 하지 않고 스펙클 노이즈를 제거하거나, 붙어 있는 두 세포를 분리해야 하는 경우처럼 객체의 *형태*를 기반으로 처리해야 하는 경우가 많습니다. 모폴로지 연산은 형태를 가진 마스크로 이미지 구조를 탐색함으로써 이 간극을 채웁니다. 이진 이미지 정리와 형태 분석을 위한 표준 도구입니다.

### 모폴로지란?

```
┌─────────────────────────────────────────────────────────────────┐
│                  Morphological Operations Overview               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Morphology = Study of shape                                   │
│   Operations based on the shape of images                       │
│                                                                 │
│   Main Uses:                                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  1. Noise removal     - Remove small noise dots          │   │
│   │  2. Hole filling      - Fill holes inside objects        │   │
│   │  3. Object separation - Separate connected objects       │   │
│   │  4. Object connection - Connect disconnected parts       │   │
│   │  5. Edge detection    - Morphological gradient           │   │
│   │  6. Skeletonization  - Extract object skeleton           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Basic Operations:                                             │
│   - Erosion: Shrink objects                                     │
│   - Dilation: Expand objects                                    │
│                                                                 │
│   Combined Operations:                                          │
│   - Opening = Erosion → Dilation                                │
│   - Closing = Dilation → Erosion                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 작동 원리

모폴로지 연산은 **구조 요소(Structuring Element)**라는 작은 마스크를 이미지 위로 이동시키며 픽셀 값을 결정합니다.

---

## 2. 구조 요소 - getStructuringElement()

### 구조 요소란?

```
┌─────────────────────────────────────────────────────────────────┐
│                        Structuring Element                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Structuring Element = A small binary matrix used in operations│
│                                                                 │
│   Main Shapes:                                                  │
│                                                                 │
│   MORPH_RECT (Rectangle)   MORPH_CROSS (Cross)    MORPH_ELLIPSE │
│   ┌───┬───┬───┐           ┌───┬───┬───┐        ┌───┬───┬───┐  │
│   │ 1 │ 1 │ 1 │           │ 0 │ 1 │ 0 │        │ 0 │ 1 │ 0 │  │
│   ├───┼───┼───┤           ├───┼───┼───┤        ├───┼───┼───┤  │
│   │ 1 │ 1 │ 1 │           │ 1 │ 1 │ 1 │        │ 1 │ 1 │ 1 │  │
│   ├───┼───┼───┤           ├───┼───┼───┤        ├───┼───┼───┤  │
│   │ 1 │ 1 │ 1 │           │ 0 │ 1 │ 0 │        │ 0 │ 1 │ 0 │  │
│   └───┴───┴───┘           └───┴───┴───┘        └───┴───┴───┘  │
│   All directions          Vertical/Horizontal   Elliptical      │
│                                                                 │
│   Effect by Size:                                               │
│   - Small size (3x3): Fine processing                           │
│   - Large size (7x7, 9x9): Strong effect                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 구조 요소 생성

```python
import cv2
import numpy as np

# getStructuringElement(shape, ksize, anchor=(-1,-1))
# shape: Structuring element shape
# ksize: (width, height) size
# anchor: Reference point (default: center)

# Why MORPH_RECT: treats all directions equally — use when objects have
# roughly rectangular or straight edges (text, PCB traces)
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
print("RECT (5x5):\n", rect_kernel)

# Cross
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
print("\nCROSS (5x5):\n", cross_kernel)

# Why MORPH_ELLIPSE: approximates a disk — preferred for circular/rounded
# objects (cells, coins) because it avoids introducing rectangular artifacts
# at diagonals that MORPH_RECT would create
ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
print("\nELLIPSE (5x5):\n", ellipse_kernel)

# Custom structuring element
custom_kernel = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.uint8)
```

### 구조 요소 시각화

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

shapes = [
    ('RECT', cv2.MORPH_RECT),
    ('CROSS', cv2.MORPH_CROSS),
    ('ELLIPSE', cv2.MORPH_ELLIPSE)
]

sizes = [(5, 5), (7, 7), (11, 11)]

fig, axes = plt.subplots(len(shapes), len(sizes), figsize=(12, 10))

for i, (name, shape) in enumerate(shapes):
    for j, size in enumerate(sizes):
        kernel = cv2.getStructuringElement(shape, size)
        axes[i, j].imshow(kernel, cmap='gray')
        axes[i, j].set_title(f'{name} {size}')
        axes[i, j].axis('off')

plt.tight_layout()
plt.show()
```

---

## 3. 침식 - erode()

침식(Erosion)은 "전경이 이 커널 형태의 영역을 완전히 덮는가?"라는 질문에 답합니다. 그렇지 않으면 해당 픽셀은 제거됩니다. 이 덕분에 커널을 완전히 덮을 수 없는 고립된 노이즈 점을 제거하고, 접촉된 객체들 사이의 얇은 연결을 끊어 개별 계수가 가능하도록 만드는 데 최적의 도구입니다.

### 침식 연산 원리

```
┌─────────────────────────────────────────────────────────────────┐
│                         Erosion                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Principle:                                                    │
│   - Move the structuring element across the image               │
│   - Set center pixel to 1 only if all pixels under the         │
│     structuring element are 1                                   │
│   - If any pixel is 0, center pixel becomes 0                   │
│                                                                 │
│   Effect:                                                       │
│   - Shrinks foreground (white) area                             │
│   - Removes small noise                                         │
│   - Separates connected objects                                 │
│   - Smooths boundaries                                          │
│                                                                 │
│   Example:                                                      │
│   Original:           After Erosion (3x3):                      │
│   ┌─────────────┐     ┌─────────────┐                          │
│   │ ████████████│     │   ████████  │                          │
│   │ ████████████│ ──▶ │   ████████  │                          │
│   │ ████████████│     │   ████████  │                          │
│   │ ████████████│     │             │                          │
│   └─────────────┘     └─────────────┘                          │
│   Borders shrink by 1 pixel                                     │
│                                                                 │
│   Noise Removal:                                                │
│   ┌─────────────┐     ┌─────────────┐                          │
│   │ ██  ■  ████ │     │ ██     ███  │                          │
│   │ ████  ████  │ ──▶ │  ██    ██   │  Small dots (■) removed  │
│   │    ■  ████  │     │       ███   │                          │
│   └─────────────┘     └─────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 침식 사용법

```python
import cv2
import numpy as np

# Prepare binary image
img = cv2.imread('binary_image.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Why MORPH_RECT (3x3): the smallest kernel that still has a meaningful neighborhood;
# larger kernels erode more aggressively and may destroy the objects you want to keep
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# erode(src, kernel, iterations=1)
# Why iterations: repeating erosion N times is equivalent to eroding with a larger
# kernel but cheaper to compute — use iterations to tune removal strength incrementally
eroded_1 = cv2.erode(binary, kernel, iterations=1)
eroded_2 = cv2.erode(binary, kernel, iterations=2)
eroded_3 = cv2.erode(binary, kernel, iterations=3)

cv2.imshow('Original', binary)
cv2.imshow('Eroded 1x', eroded_1)
cv2.imshow('Eroded 2x', eroded_2)
cv2.imshow('Eroded 3x', eroded_3)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 침식 테스트 이미지 생성

```python
import cv2
import numpy as np

# Create test image
img = np.zeros((300, 400), dtype=np.uint8)

# Large rectangle
cv2.rectangle(img, (50, 50), (150, 150), 255, -1)

# Small noise dots
for _ in range(50):
    x, y = np.random.randint(200, 350), np.random.randint(50, 250)
    cv2.circle(img, (x, y), 2, 255, -1)

# Connected circles
cv2.circle(img, (280, 150), 40, 255, -1)
cv2.circle(img, (320, 150), 40, 255, -1)

# Apply erosion
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
eroded = cv2.erode(img, kernel, iterations=1)

cv2.imshow('Original', img)
cv2.imshow('Eroded', eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 4. 팽창 - dilate()

팽창(Dilation)은 침식의 쌍대(Dual) 연산입니다. "전경이 이 커널 영역의 어느 부분이라도 닿는가?"라고 묻습니다. 그렇다면 픽셀이 설정됩니다. 따라서 끊어진 획을 다시 연결하고 작은 틈을 채우는 데 이상적입니다. 전체 객체 크기가 유지되도록 항상 침식과 짝을 이루어 사용합니다(열기 또는 닫기 형태로).

### 팽창 연산 원리

```
┌─────────────────────────────────────────────────────────────────┐
│                         Dilation                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Principle:                                                    │
│   - Move the structuring element across the image               │
│   - Set center pixel to 1 if any pixel under the               │
│     structuring element is 1                                    │
│   - Opposite of erosion                                         │
│                                                                 │
│   Effect:                                                       │
│   - Expands foreground (white) area                             │
│   - Fills holes                                                 │
│   - Connects broken parts                                       │
│   - Emphasizes objects                                          │
│                                                                 │
│   Example:                                                      │
│   Original:           After Dilation (3x3):                     │
│   ┌─────────────┐     ┌─────────────┐                          │
│   │   ██████    │     │ ████████████│                          │
│   │   ██████    │ ──▶ │ ████████████│                          │
│   │   ██████    │     │ ████████████│                          │
│   └─────────────┘     └─────────────┘                          │
│   Borders expand by 1 pixel                                     │
│                                                                 │
│   Connect Broken Parts:                                         │
│   ┌─────────────┐     ┌─────────────┐                          │
│   │ ██      ██  │     │ ████    ████│                          │
│   │ ██  ..  ██  │ ──▶ │ ██████████  │  Dotted line connected  │
│   │ ██      ██  │     │ ████    ████│                          │
│   └─────────────┘     └─────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 팽창 사용법

```python
import cv2
import numpy as np

# Prepare binary image
img = cv2.imread('binary_image.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# dilate(src, kernel, iterations=1)
dilated_1 = cv2.dilate(binary, kernel, iterations=1)
dilated_2 = cv2.dilate(binary, kernel, iterations=2)
dilated_3 = cv2.dilate(binary, kernel, iterations=3)

cv2.imshow('Original', binary)
cv2.imshow('Dilated 1x', dilated_1)
cv2.imshow('Dilated 2x', dilated_2)
cv2.imshow('Dilated 3x', dilated_3)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 침식과 팽창 비교

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Test image
img = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
cv2.circle(img, (100, 100), 20, 0, -1)  # Inner hole

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

eroded = cv2.erode(img, kernel, iterations=1)
dilated = cv2.dilate(img, kernel, iterations=1)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')

axes[1].imshow(eroded, cmap='gray')
axes[1].set_title('Eroded (Shrink)')

axes[2].imshow(dilated, cmap='gray')
axes[2].set_title('Dilated (Expand)')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## 5. 열기와 닫기 - morphologyEx()

단순 침식은 객체를 영구적으로 축소하고, 단순 팽창은 객체를 팽창시킵니다. 열기(Opening)와 닫기(Closing)는 두 연산을 결합하여 객체 크기를 대략 유지하면서 *특정 유형의 결함*(노이즈 점 또는 홀)만을 대상으로 합니다. 이 대칭성 때문에 실제 처리 파이프라인에서는 단독 침식/팽창보다 열기/닫기가 선호됩니다.

### 열기 (Opening)

```
┌─────────────────────────────────────────────────────────────────┐
│                      Opening                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Opening = Erosion → Dilation                                  │
│                                                                 │
│   Effect:                                                       │
│   - Removes small noise (dots)                                  │
│   - Maintains overall object size approximately                 │
│   - Breaks thin connections                                     │
│                                                                 │
│   Original    Erosion      Dilation (Opening result)            │
│   ┌──────┐    ┌──────┐    ┌──────┐                              │
│   │██ ■ █│    │█     │    │██   █│                              │
│   │██████│ ─▶ │ ████ │ ─▶ │██████│                              │
│   │  ■ ██│    │    █ │    │    ██│                              │
│   └──────┘    └──────┘    └──────┘                              │
│   Small dots (■) removed                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 닫기 (Closing)

```
┌─────────────────────────────────────────────────────────────────┐
│                      Closing                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Closing = Dilation → Erosion                                  │
│                                                                 │
│   Effect:                                                       │
│   - Fills small holes                                           │
│   - Maintains overall object size approximately                 │
│   - Connects broken parts                                       │
│                                                                 │
│   Original    Dilation     Erosion (Closing result)             │
│   ┌──────┐    ┌──────┐    ┌──────┐                              │
│   │██████│    │██████│    │██████│                              │
│   │██○ ██│ ─▶ │██████│ ─▶ │██████│                              │
│   │██████│    │██████│    │██████│                              │
│   └──────┘    └──────┘    └──────┘                              │
│   Inner hole (○) filled                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### morphologyEx() 사용법

```python
import cv2
import numpy as np

img = cv2.imread('binary_image.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Why (5,5): kernel must be larger than the noise/holes you want to remove;
# a 5x5 kernel removes features smaller than ~5 pixels in diameter
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# morphologyEx(src, op, kernel, iterations=1)
# op: Operation type

# Why MORPH_OPEN first: erosion removes small noise dots; the subsequent dilation
# restores the larger objects to their original size (erode → dilate = opening)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Closing: Hole filling (dilate → erode; expands to fill holes, then contracts back)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Why open before close: opening on the raw image avoids noise dots being "healed"
# into the object by the closing step — order matters
clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Original', binary)
cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)
cv2.imshow('Open + Close', clean)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 열기와 닫기 비교 테스트

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Test image: Rectangle with noise + holes
img = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), 255, -1)

# Add noise (small dots)
noise = img.copy()
for _ in range(30):
    x, y = np.random.randint(10, 45), np.random.randint(10, 190)
    cv2.circle(noise, (x, y), 2, 255, -1)
for _ in range(30):
    x, y = np.random.randint(155, 190), np.random.randint(10, 190)
    cv2.circle(noise, (x, y), 2, 255, -1)

# Add holes (inside object)
holes = noise.copy()
for _ in range(10):
    x, y = np.random.randint(60, 140), np.random.randint(60, 140)
    cv2.circle(holes, (x, y), 3, 0, -1)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

opening = cv2.morphologyEx(holes, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(holes, cv2.MORPH_CLOSE, kernel)
both = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(holes, cmap='gray')
axes[0, 0].set_title('Original (Noise + Holes)')

axes[0, 1].imshow(opening, cmap='gray')
axes[0, 1].set_title('Opening (Noise Removed)')

axes[1, 0].imshow(closing, cmap='gray')
axes[1, 0].set_title('Closing (Holes Filled)')

axes[1, 1].imshow(both, cmap='gray')
axes[1, 1].set_title('Open + Close')

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## 6. 그래디언트, 탑햇, 블랙햇

### 모폴로지 그래디언트

```
┌─────────────────────────────────────────────────────────────────┐
│                   Morphological Gradient                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Gradient = Dilation - Erosion                                 │
│                                                                 │
│   Effect: Extract object outline (boundary)                     │
│                                                                 │
│   Original          Dilation           Erosion                  │
│   ┌──────┐         ┌──────┐         ┌──────┐                   │
│   │ ████ │         │██████│         │  ██  │                   │
│   │ ████ │    -    │██████│    =    │  ██  │                   │
│   │ ████ │         │██████│         │  ██  │                   │
│   └──────┘         └──────┘         └──────┘                   │
│                                                                 │
│   Gradient Result:                                              │
│   ┌──────┐                                                      │
│   │ ████ │  → Only outline remains                              │
│   │ █  █ │                                                      │
│   │ ████ │                                                      │
│   └──────┘                                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 탑햇과 블랙햇

```
┌─────────────────────────────────────────────────────────────────┐
│                    Top-hat / Black-hat                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Top-hat = Original - Opening                                  │
│   - Extract small bright parts from bright areas                │
│   - Detect small objects brighter than background               │
│                                                                 │
│   Black-hat = Closing - Original                                │
│   - Extract small dark parts from dark areas                    │
│   - Detect small holes/objects darker than background           │
│                                                                 │
│   Applications:                                                 │
│   - Correct images with uneven illumination                     │
│   - Remove shadows from document images                         │
│   - Detect small defects                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 구현 및 사용

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

# Morphological gradient
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# Top-hat
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

# Black-hat
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

# Manual calculation (for verification)
dilated = cv2.dilate(img, kernel)
eroded = cv2.erode(img, kernel)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

gradient_manual = dilated - eroded
tophat_manual = img - opening
blackhat_manual = closing - img

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original')

axes[0, 1].imshow(gradient, cmap='gray')
axes[0, 1].set_title('Gradient (Edge)')

axes[0, 2].imshow(tophat, cmap='gray')
axes[0, 2].set_title('Top Hat (Bright spots)')

axes[1, 0].imshow(blackhat, cmap='gray')
axes[1, 0].set_title('Black Hat (Dark spots)')

# Enhance contrast using top-hat + black-hat
enhanced = cv2.add(img, tophat)
enhanced = cv2.subtract(enhanced, blackhat)
axes[1, 1].imshow(enhanced, cmap='gray')
axes[1, 1].set_title('Enhanced (Top+Black Hat)')

for ax in axes.flatten():
    ax.axis('off')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
```

### 모든 모폴로지 연산 정리

```python
import cv2

# List of operations available in morphologyEx()
operations = {
    cv2.MORPH_ERODE: "Erode",
    cv2.MORPH_DILATE: "Dilate",
    cv2.MORPH_OPEN: "Open (Erode + Dilate)",
    cv2.MORPH_CLOSE: "Close (Dilate + Erode)",
    cv2.MORPH_GRADIENT: "Gradient (Dilate - Erode)",
    cv2.MORPH_TOPHAT: "Top Hat (Src - Open)",
    cv2.MORPH_BLACKHAT: "Black Hat (Close - Src)",
    cv2.MORPH_HITMISS: "Hit-Miss (Pattern Matching)"
}

for op, name in operations.items():
    print(f"{op}: {name}")
```

---

## 7. 실전 응용

### 노이즈 제거 파이프라인

```python
import cv2
import numpy as np

def remove_noise_morphology(binary_img, noise_size=3):
    """
    Remove noise using morphological operations

    Parameters:
    - binary_img: Binary image
    - noise_size: Maximum size of noise to remove
    """
    # Why noise_size * 2 + 1: the kernel must fully contain the largest noise dot
    # (radius noise_size → diameter noise_size*2) and be odd for a centered anchor
    kernel_size = noise_size * 2 + 1
    # Why MORPH_ELLIPSE: circular objects (cells, blobs) are better modeled with a
    # disk-shaped kernel — avoids introducing rectangular bias at diagonals
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    # Opening to remove small noise dots (erode kills noise, dilate restores objects)
    cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # Closing to fill small holes (dilate bridges gaps, erode restores boundaries)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned


# Usage example
img = cv2.imread('noisy_document.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
cleaned = remove_noise_morphology(binary, noise_size=2)
```

### 객체 분리

```python
import cv2
import numpy as np

def separate_objects(binary_img, erosion_iterations=3):
    """
    Separate connected objects
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Why multiple erosion iterations instead of a large kernel: iterating with a
    # small kernel is cheaper and lets you tune separation strength without rebuilding
    # the structuring element; each pass peels off one layer from every boundary
    eroded = cv2.erode(binary_img, kernel, iterations=erosion_iterations)

    # Distance transform to find center points — the peak of the distance map is
    # the point farthest from any background pixel, i.e., the object center
    dist_transform = cv2.distanceTransform(eroded, cv2.DIST_L2, 5)
    # Why 0.5 * max: keeps only the top half of distance values, retaining confident
    # object cores while discarding ambiguous border regions
    _, sure_fg = cv2.threshold(
        dist_transform, 0.5 * dist_transform.max(), 255, 0
    )
    sure_fg = np.uint8(sure_fg)

    return eroded, sure_fg


# Usage example
img = cv2.imread('connected_circles.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
separated, centers = separate_objects(binary)
```

### 문서 이미지 전처리

```python
import cv2
import numpy as np

def preprocess_document(img):
    """
    Document image preprocessing (shadow removal + binarization)
    """
    # Grayscale conversion
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Top-hat to extract bright background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    # Black-hat to correct shadows/dark areas
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Subtract black-hat from original (shadow removal effect)
    no_shadow = cv2.add(gray, blackhat)

    # Adaptive binarization
    binary = cv2.adaptiveThreshold(
        no_shadow, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 15
    )

    # Noise removal
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)

    return binary


# Usage example
img = cv2.imread('document_with_shadow.jpg')
result = preprocess_document(img)
```

### 스켈레톤화 (Skeletonization)

```python
import cv2
import numpy as np

def skeletonize(img):
    """
    Extract skeleton using morphological operations
    """
    skeleton = np.zeros_like(img)
    temp = img.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        # Opening operation
        opened = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)

        # Calculate difference
        diff = cv2.subtract(temp, opened)

        # Erosion
        temp = cv2.erode(temp, kernel)

        # Add to skeleton
        skeleton = cv2.bitwise_or(skeleton, diff)

        # Stop if no more white pixels
        if cv2.countNonZero(temp) == 0:
            break

    return skeleton


# Usage example
img = cv2.imread('character.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
skeleton = skeletonize(binary)
```

---

## 8. 연습 문제

### 연습 1: 구조 요소 효과 비교

동일한 이진 이미지에 대해 세 가지 구조 요소(RECT, CROSS, ELLIPSE)를 사용하여 침식과 팽창을 적용하고, 결과의 차이를 분석하세요.

### 연습 2: 문자 두께 조절

손글씨 이미지에서 문자의 두께를 조절하는 함수를 작성하세요:
- 양수 값: 팽창으로 두껍게
- 음수 값: 침식으로 얇게

```python
def adjust_stroke_width(img, amount):
    """
    amount > 0: Thicken
    amount < 0: Thin
    """
    pass
```

### 연습 3: 경계 추출 비교

다음 세 가지 방법으로 객체의 경계를 추출하고 비교하세요:
1. 모폴로지 그래디언트
2. Canny 엣지 검출
3. findContours

### 연습 4: 점자 인식 전처리

점자 이미지에서 각 점을 개별적으로 검출하기 위한 전처리 파이프라인을 설계하세요. (힌트: 침식으로 점들을 분리)

### 연습 5: 세포 분리 (Watershed 전처리)

현미경 세포 이미지에서 붙어있는 세포들을 분리하기 위한 전처리를 구현하세요:
1. 이진화
2. 노이즈 제거 (열기/닫기)
3. 확실한 배경 영역 찾기 (팽창)
4. 확실한 전경 영역 찾기 (거리 변환 + 임계값)

---

## 9. 다음 단계

[이진화 및 임계처리](./07_Thresholding.md)에서 다양한 이진화 방법과 임계처리 기법을 학습합니다!

**다음에 배울 내용**:
- 전역 임계처리 (`cv2.threshold`)
- OTSU 자동 임계값
- 적응형 임계처리
- HSV 기반 임계처리

---

## 10. 참고 자료

### 공식 문서

- [erode() 문서](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511f2fb1c)
- [dilate() 문서](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c)
- [morphologyEx() 문서](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f)
- [getStructuringElement() 문서](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac342a1bb6eabf6f55c803b09268e36dc)

### 관련 학습 자료

| 폴더 | 관련 내용 |
|------|----------|
| [이미지 필터링](./05_Image_Filtering.md) | 필터링 기초 |
| [윤곽선 검출 (Contour Detection)](./09_Contours.md) | 전처리 후 윤곽선 검출 |

### 추가 참고

- [모폴로지 연산 튜토리얼](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
- [수학적 모폴로지 이론](https://en.wikipedia.org/wiki/Mathematical_morphology)

