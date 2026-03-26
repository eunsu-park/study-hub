# 색상 공간

## 개요

컴퓨터 비전에서 색상 공간(Color Space)은 색상을 표현하는 방법입니다. OpenCV는 기본적으로 BGR 색상 공간을 사용하지만, 특정 작업에는 HSV, LAB 등 다른 색상 공간이 더 효과적입니다. 이 문서에서는 다양한 색상 공간의 특성과 변환 방법, 그리고 색상 기반 객체 추적을 학습합니다.

**난이도**: ⭐⭐ (초급-중급)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. BGR과 RGB의 차이 이해
2. HSV 색상 공간의 원리와 활용
3. `cv2.cvtColor()`를 사용한 색상 공간 변환
4. 채널 분리/병합
5. 색상 기반 객체 추적 구현

---

## 목차

1. [BGR vs RGB](#1-bgr-vs-rgb)
2. [cv2.cvtColor()와 색상 변환 상수](#2-cv2cvtcolor와-색상-변환-상수)
3. [HSV 색상 공간](#3-hsv-색상-공간)
4. [LAB 색상 공간](#4-lab-색상-공간)
5. [그레이스케일 변환](#5-그레이스케일-변환)
6. [채널 분리와 병합](#6-채널-분리와-병합)
7. [색상 기반 객체 추적](#7-색상-기반-객체-추적)
8. [연습 문제](#8-연습-문제)
9. [다음 단계](#9-다음-단계)
10. [참고 자료](#10-참고-자료)

---

## 1. BGR vs RGB

### OpenCV의 기본 색상 순서

```
┌─────────────────────────────────────────────────────────────────┐
│                    BGR vs RGB Comparison                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   OpenCV (BGR)                 Most Libraries (RGB)             │
│   ┌─────────────┐              ┌─────────────┐                 │
│   │ B │ G │ R │               │ R │ G │ B │                   │
│   │[0]│[1]│[2]│               │[0]│[1]│[2]│                   │
│   └─────────────┘              └─────────────┘                 │
│                                                                 │
│   Pure red:                    Pure red:                        │
│   [0, 0, 255]                  [255, 0, 0]                      │
│                                                                 │
│   Pure blue:                   Pure blue:                       │
│   [255, 0, 0]                  [0, 0, 255]                      │
│                                                                 │
│   OpenCV libraries:            RGB libraries:                   │
│   - cv2.imread()               - matplotlib                     │
│   - cv2.imshow()               - PIL/Pillow                     │
│   - cv2.imwrite()              - Tkinter                        │
│                                - Web browsers (CSS/HTML)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### BGR을 사용하는 이유

역사적인 이유입니다. 초기 카메라와 디스플레이 하드웨어가 BGR 순서로 데이터를 저장했고, OpenCV는 이 관례를 따랐습니다.

### BGR ↔ RGB 변환

```python
import cv2
import numpy as np

img_bgr = cv2.imread('image.jpg')

# cvtColor is the safest and most readable approach — explicitly declares intent
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

img_bgr_back = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# [:, :, ::-1] reverses the channel axis in-place (zero-copy view) — faster
# than cvtColor but less readable; use when performance matters
img_rgb_np = img_bgr[:, :, ::-1]  # Reverse channel order
img_rgb_np = img_bgr[..., ::-1]   # Same result

# cv2.split + cv2.merge is slower than slicing but makes the intent explicit
# and is easier to extend (e.g., inserting a new channel between them)
b, g, r = cv2.split(img_bgr)
img_rgb_split = cv2.merge([r, g, b])
```

### matplotlib과 함께 사용하기

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')

# Wrong display (BGR as-is → colors are swapped)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img)  # BGR as-is → red and blue swapped
plt.title('Wrong (BGR)')
plt.axis('off')

# Correct display (convert to RGB)
plt.subplot(1, 3, 2)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title('Correct (RGB)')
plt.axis('off')

# Grayscale
plt.subplot(1, 3, 3)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale')
plt.axis('off')

plt.tight_layout()
plt.show()
```

---

## 2. cv2.cvtColor()와 색상 변환 상수

### 기본 사용법

```python
import cv2

img = cv2.imread('image.jpg')

# cv2.cvtColor(src, code) - color space conversion
dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

### 주요 변환 코드

```
┌─────────────────────────────────────────────────────────────────┐
│                     Major Color Conversion Codes                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   BGR ↔ Other Color Spaces                                      │
│   ├── COLOR_BGR2RGB / COLOR_RGB2BGR                             │
│   ├── COLOR_BGR2GRAY / COLOR_GRAY2BGR                           │
│   ├── COLOR_BGR2HSV / COLOR_HSV2BGR                             │
│   ├── COLOR_BGR2LAB / COLOR_LAB2BGR                             │
│   ├── COLOR_BGR2YCrCb / COLOR_YCrCb2BGR                         │
│   └── COLOR_BGR2HLS / COLOR_HLS2BGR                             │
│                                                                 │
│   RGB ↔ Other Color Spaces                                      │
│   ├── COLOR_RGB2GRAY / COLOR_GRAY2RGB                           │
│   ├── COLOR_RGB2HSV / COLOR_HSV2RGB                             │
│   ├── COLOR_RGB2LAB / COLOR_LAB2RGB                             │
│   └── COLOR_RGB2HLS / COLOR_HLS2RGB                             │
│                                                                 │
│   Special Conversions                                           │
│   ├── COLOR_BGR2HSV_FULL  (H: 0-255)                            │
│   ├── COLOR_BGR2HSV       (H: 0-179)                            │
│   └── COLOR_BayerBG2BGR   (Bayer → BGR)                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 변환 예시

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to various color spaces
conversions = {
    'Original (RGB)': img_rgb,
    'Grayscale': cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
    'HSV': cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
    'LAB': cv2.cvtColor(img, cv2.COLOR_BGR2LAB),
    'YCrCb': cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb),
    'HLS': cv2.cvtColor(img, cv2.COLOR_BGR2HLS),
}

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for ax, (name, converted) in zip(axes, conversions.items()):
    if len(converted.shape) == 2:
        ax.imshow(converted, cmap='gray')
    else:
        ax.imshow(converted)
    ax.set_title(name)
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## 3. HSV 색상 공간

RGB와 BGR은 색상과 밝기가 섞여 있어 조명이 변할 때 특정 색상을 분리하기 어렵습니다. HSV는 이 두 요소를 분리합니다. Hue 채널 하나만으로 색상을 나타낼 수 있으므로, 밝은 곳이든 어두운 곳이든 간단한 범위 임계처리만으로 "빨간 물체"를 감지할 수 있습니다.

### HSV란?

HSV는 색상(Hue), 채도(Saturation), 명도(Value)로 색을 표현합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      HSV Color Space                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   H (Hue) - Color                                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  0°    60°   120°   180°   240°   300°   360°          │   │
│   │  Red   Yellow Green  Cyan   Blue  Magenta Red          │   │
│   │  ├──────┼──────┼──────┼──────┼──────┼──────┤            │   │
│   │  0     30     60     90    120    150    179            │   │
│   │      (OpenCV H range: 0-179)                            │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   S (Saturation) - Saturation (0-255)                           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  0 (grayscale/gray)  ──────────────▶  255 (pure color)  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   V (Value) - Brightness (0-255)                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  0 (black)  ──────────────────▶  255 (bright)           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│                        V (Brightness)                           │
│                          ▲                                       │
│                          │    White                              │
│                          │   /                                   │
│                          │  /                                    │
│                          │ /     Pure color                      │
│                          │/───────●                              │
│                          │        ╲                              │
│                          │         ╲  S (Saturation)             │
│                          │          ╲                            │
│                          ●───────────╲───▶ H (Hue, circular)     │
│                        Black                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### HSV 변환 및 채널 확인

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')

# BGR → HSV conversion
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Split channels
h, s, v = cv2.split(hsv)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original')

axes[0, 1].imshow(h, cmap='hsv')  # Use hsv colormap for Hue
axes[0, 1].set_title('H (Hue)')

axes[1, 0].imshow(s, cmap='gray')
axes[1, 0].set_title('S (Saturation)')

axes[1, 1].imshow(v, cmap='gray')
axes[1, 1].set_title('V (Value)')

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()
```

### HSV의 장점

```python
import cv2
import numpy as np

# In HSV, lighting changes mainly affect V (brightness); H stays stable.
# That's why HSV works far better than BGR for robust color detection.

img = cv2.imread('red_objects.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Red wraps around the Hue circle: it appears near H=0 AND near H=180.
# Two separate ranges are needed because OpenCV's H axis is 0-179, not circular.
lower_red1 = np.array([0, 100, 100])    # S>100 and V>100 exclude near-gray pixels
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

# Bitwise OR merges both masks into one — pixels belonging to either range pass
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 | mask2

# bitwise_and zeroes out pixels where mask=0, keeping only the detected color
result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('Original', img)
cv2.imshow('Mask', mask)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 주요 색상의 HSV 범위

```
┌─────────────────────────────────────────────────────────────────┐
│                    Common Color HSV Ranges (OpenCV)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Color      H (Hue)        S (Saturation)   V (Value)          │
│   ──────────────────────────────────────────────────────────    │
│   Red        0-10, 160-179   100-255         100-255            │
│   Orange     10-25           100-255         100-255            │
│   Yellow     25-35           100-255         100-255            │
│   Green      35-85           100-255         100-255            │
│   Cyan       85-95           100-255         100-255            │
│   Blue       95-130          100-255         100-255            │
│   Magenta    130-160         100-255         100-255            │
│                                                                 │
│   White      0-179           0-30            200-255            │
│   Black      0-179           0-255           0-50               │
│   Gray       0-179           0-30            50-200             │
│                                                                 │
│   Note: Ranges need adjustment based on lighting conditions     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. LAB 색상 공간

LAB은 RGB와 HSV가 공통으로 가진 문제를 해결합니다. 즉, 수치상의 동일한 차이가 인간이 느끼는 동일한 차이와 일치하지 않는다는 문제입니다. LAB에서는 두 색상 벡터 간의 유클리드 거리가 인간의 눈에 보이는 색상 차이와 근접하게 대응합니다. 따라서 지각적 색상 비교와 전문적인 색상 보정에 이상적인 색상 공간입니다.

### LAB이란?

LAB(또는 CIELAB)은 인간의 색상 인지에 기반한 색상 공간입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      LAB Color Space                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   L (Lightness) - Brightness                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  0 (black)  ──────────────────────▶  255 (white)        │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   A - Green(-) ↔ Red(+)                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  0 (green)  ────── 128 (neutral) ──────  255 (red)      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   B - Blue(-) ↔ Yellow(+)                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  0 (blue)  ────── 128 (neutral) ──────  255 (yellow)    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│                     +B (Yellow)                                  │
│                        ▲                                        │
│                        │                                        │
│            -A ◀────────┼────────▶ +A                            │
│          (Green)       │        (Red)                           │
│                        │                                        │
│                        ▼                                        │
│                     -B (Blue)                                    │
│                                                                 │
│   Advantages:                                                   │
│   - Color distance calculation similar to human vision          │
│   - Brightness and color are separated                          │
│   - Useful for color correction and color transfer              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### LAB 변환 및 활용

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(lab)

# Modifying only L leaves the color (a, b) untouched — this is the key advantage
# over adjusting brightness in BGR, where adding a constant shifts all three channels
# and inadvertently changes the hue
l_adjusted = cv2.add(l, 30)  # cv2.add saturates at 255, avoiding overflow wrapping
l_adjusted = np.clip(l_adjusted, 0, 255).astype(np.uint8)

# Reassemble: a and b unchanged, so colors remain perceptually identical to the original
lab_adjusted = cv2.merge([l_adjusted, a, b])
result = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original')

axes[0, 1].imshow(l, cmap='gray')
axes[0, 1].set_title('L (Lightness)')

axes[0, 2].imshow(a, cmap='RdYlGn_r')
axes[0, 2].set_title('A (Green-Red)')

axes[1, 0].imshow(b, cmap='YlGnBu_r')
axes[1, 0].set_title('B (Blue-Yellow)')

axes[1, 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('Brightness Adjusted')

for ax in axes.flatten():
    ax.axis('off')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
```

### CLAHE로 LAB 밝기 보정

```python
import cv2

img = cv2.imread('dark_image.jpg')

# Working in LAB is crucial here: CLAHE must be applied only to lightness (L),
# not to color channels — otherwise it would create color distortions
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# CLAHE enhances local contrast adaptively per tile rather than globally,
# preventing over-brightening bright regions while lifting dark ones.
# clipLimit=2.0 caps the amplification to avoid amplifying noise.
# tileGridSize=(8,8) is a good balance: coarser → more global; finer → more local
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)

# a and b carry the color; only L was modified, so hues are preserved
lab_clahe = cv2.merge([l_clahe, a, b])
result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

cv2.imshow('Original', img)
cv2.imshow('CLAHE Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 5. 그레이스케일 변환

### 변환 원리

```
┌─────────────────────────────────────────────────────────────────┐
│                   Grayscale Conversion Principle                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   BGR → Grayscale conversion formula:                           │
│                                                                 │
│   Gray = 0.114 × B + 0.587 × G + 0.299 × R                     │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   Why not simple average?                               │   │
│   │                                                         │   │
│   │   Human eyes are most sensitive to green and least to blue │
│   │   Therefore, green (G) has the highest weight (0.587)  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Color image                     Grayscale                     │
│   ┌───────────────┐              ┌───────────────┐             │
│   │ B │ G │ R │               │     Gray      │             │
│   │200│100│ 50│    ───▶       │      121      │             │
│   └───────────────┘              └───────────────┘             │
│   0.114×200 + 0.587×100 + 0.299×50 = 121.45                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

`Gray = 0.114·B + 0.587·G + 0.299·R` 공식은 인간의 광수용체 민감도에 따라 채널에 가중치를 부여합니다. 눈은 녹색에 가장 민감(~55%)하고, 빨간색에 중간 정도(~30%), 파란색에 가장 덜 민감(~11%)합니다. 단순 평균(각 0.333)을 사용하면 파란색 영역이 너무 밝고 녹색 영역이 너무 어두운 그레이스케일이 만들어집니다.

### 그레이스케일 변환 방법

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# cvtColor uses the luminosity-weighted formula above — preferred over imread grayscale
# because it works on an already-loaded image without re-reading from disk
gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Method 2: Read directly with imread
gray2 = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Method 3: Manual calculation with NumPy (for learning)
b, g, r = cv2.split(img)
gray3 = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)

# Method 4: Simple average (not recommended - visually unnatural)
gray4 = np.mean(img, axis=2).astype(np.uint8)

# Compare results
print(f"cvtColor result: {gray1.shape}")
print(f"Manual calculation result: {gray3.shape}")
print(f"Max difference: {np.max(np.abs(gray1.astype(int) - gray3.astype(int)))}")
```

### 그레이스케일 → 컬러 (의사 컬러)

```python
import cv2

gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Grayscale → 3 channels (still grayscale)
gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Apply colormap (heatmap, etc.)
# COLORMAP_JET, COLORMAP_HOT, COLORMAP_RAINBOW, etc.
colormap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

cv2.imshow('Grayscale', gray)
cv2.imshow('Colormap', colormap)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 6. 채널 분리와 병합

### cv2.split()과 cv2.merge()

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# Split channels
b, g, r = cv2.split(img)

# Or use NumPy indexing (faster)
b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]

# Merge channels
merged = cv2.merge([b, g, r])  # BGR order

# Change channel order when merging (BGR → RGB)
rgb = cv2.merge([r, g, b])

# Combine with empty channels (display single channel only)
zeros = np.zeros_like(b)
only_blue = cv2.merge([b, zeros, zeros])
only_green = cv2.merge([zeros, g, zeros])
only_red = cv2.merge([zeros, zeros, r])
```

### 채널별 시각화

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')
b, g, r = cv2.split(img)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Original
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original')

# Each channel (as grayscale)
axes[0, 1].imshow(r, cmap='gray')
axes[0, 1].set_title('Red Channel')

axes[0, 2].imshow(g, cmap='gray')
axes[0, 2].set_title('Green Channel')

axes[1, 0].imshow(b, cmap='gray')
axes[1, 0].set_title('Blue Channel')

# Each channel (in color)
zeros = np.zeros_like(b)
axes[1, 1].imshow(cv2.merge([zeros, zeros, r]))  # RGB order
axes[1, 1].set_title('Red Only')

axes[1, 2].imshow(cv2.merge([zeros, g, zeros]))
axes[1, 2].set_title('Green Only')

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()
```

### 채널 조작 예제

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 1. Boost red channel: cast to int16 first to avoid uint8 overflow, then clip
b, g, r = cv2.split(img)
r_boost = np.clip(r.astype(np.int16) + 50, 0, 255).astype(np.uint8)
warm = cv2.merge([b, g, r_boost])  # Higher R relative to B gives a warm/sunset feel

# 2. Swapping R and B produces a "cool" or infrared-like look — useful for artistic effects
b, g, r = cv2.split(img)
swapped = cv2.merge([r, g, b])

# 3. Simple average is visually inaccurate (ignores perceptual weights) but useful
# as a fast approximation when exact luminance doesn't matter
b, g, r = cv2.split(img)
gray_avg = ((b.astype(np.int16) + g + r) // 3).astype(np.uint8)

# 4. zeros_like preserves the same shape and dtype as b — safer than np.zeros((h,w))
b, g, r = cv2.split(img)
only_r = cv2.merge([np.zeros_like(b), np.zeros_like(g), r])
```

---

## 7. 색상 기반 객체 추적

### inRange()를 사용한 색상 필터링

```
┌─────────────────────────────────────────────────────────────────┐
│                   Color-Based Object Tracking Pipeline          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input image (BGR)                                             │
│        │                                                        │
│        ▼                                                        │
│   HSV conversion                                                │
│        │                                                        │
│        ▼                                                        │
│   cv2.inRange(hsv, lower, upper) ──▶ Binary mask               │
│        │                                                        │
│        ▼                                                        │
│   Noise removal (morphological operations)                      │
│        │                                                        │
│        ▼                                                        │
│   Contour detection                                             │
│        │                                                        │
│        ▼                                                        │
│   Extract object position/size                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 색상 추적 구현

```python
import cv2
import numpy as np

def track_color(img, lower_hsv, upper_hsv):
    """Track objects in a specific color range"""
    # HSV conversion
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create mask
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Detect contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Draw results
    result = img.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area filter
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Center point
            cx, cy = x + w//2, y + h//2
            cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)

    return result, mask


# Example usage: Track blue
img = cv2.imread('blue_objects.jpg')

lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])

result, mask = track_color(img, lower_blue, upper_blue)

cv2.imshow('Original', img)
cv2.imshow('Mask', mask)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 실시간 색상 추적 (웹캠)

```python
import cv2
import numpy as np

def nothing(x):
    pass

# Create trackbars
cv2.namedWindow('Trackbars')
cv2.createTrackbar('H_Low', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('H_High', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('S_Low', 'Trackbars', 100, 255, nothing)
cv2.createTrackbar('S_High', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('V_Low', 'Trackbars', 100, 255, nothing)
cv2.createTrackbar('V_High', 'Trackbars', 255, 255, nothing)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Read trackbar values
    h_low = cv2.getTrackbarPos('H_Low', 'Trackbars')
    h_high = cv2.getTrackbarPos('H_High', 'Trackbars')
    s_low = cv2.getTrackbarPos('S_Low', 'Trackbars')
    s_high = cv2.getTrackbarPos('S_High', 'Trackbars')
    v_low = cv2.getTrackbarPos('V_Low', 'Trackbars')
    v_high = cv2.getTrackbarPos('V_High', 'Trackbars')

    lower = np.array([h_low, s_low, v_low])
    upper = np.array([h_high, s_high, v_high])

    # HSV conversion and mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 다중 색상 추적

```python
import cv2
import numpy as np

# Define multiple colors
colors = {
    'red': {
        'lower1': np.array([0, 100, 100]),
        'upper1': np.array([10, 255, 255]),
        'lower2': np.array([160, 100, 100]),
        'upper2': np.array([179, 255, 255]),
        'color': (0, 0, 255)
    },
    'green': {
        'lower': np.array([35, 100, 100]),
        'upper': np.array([85, 255, 255]),
        'color': (0, 255, 0)
    },
    'blue': {
        'lower': np.array([100, 100, 100]),
        'upper': np.array([130, 255, 255]),
        'color': (255, 0, 0)
    }
}

def track_multiple_colors(img, colors):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    result = img.copy()

    for name, params in colors.items():
        # Create mask
        if 'lower1' in params:  # For colors like red with two ranges
            mask1 = cv2.inRange(hsv, params['lower1'], params['upper1'])
            mask2 = cv2.inRange(hsv, params['lower2'], params['upper2'])
            mask = mask1 | mask2
        else:
            mask = cv2.inRange(hsv, params['lower'], params['upper'])

        # Detect contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x+w, y+h), params['color'], 2)
                cv2.putText(result, name, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, params['color'], 2)

    return result
```

---

## 8. 연습 문제

### 연습 1: 색상 팔레트 생성

16가지 주요 색상(빨강, 주황, 노랑, 초록, 청록, 파랑, 보라, 분홍, 흰색, 검정, 회색 등)을 BGR 값으로 정의하고, 100x100 크기의 색상 칩을 4x4 격자로 배치한 팔레트 이미지를 생성하세요.

### 연습 2: HSV 색상 선택기

마우스로 이미지를 클릭하면 해당 픽셀의 HSV 값을 출력하고, 그 색상과 유사한 모든 영역을 하이라이트하는 프로그램을 작성하세요.

```python
# Hint: use cv2.setMouseCallback()
def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Output HSV value of clicked position
        pass
```

### 연습 3: 채널 스왑 효과

이미지의 채널을 다양하게 조합하여 6가지 효과(BGR, BRG, GBR, GRB, RBG, RGB)를 만들고 비교하세요.

### 연습 4: 피부색 검출

HSV와 YCrCb 색상 공간을 사용하여 이미지에서 피부색 영역을 검출하세요. 두 방법의 결과를 비교하세요.

```python
# Example HSV ranges for skin color
# H: 0-50, S: 20-150, V: 70-255

# Example YCrCb ranges for skin color
# Y: 0-255, Cr: 135-180, Cb: 85-135
```

### 연습 5: 색상 전이 애니메이션

H 채널을 점진적으로 증가시켜 이미지의 색상이 무지개처럼 변하는 애니메이션을 만드세요.

```python
# Hint
for h_shift in range(0, 180, 5):
    h_channel = (original_h + h_shift) % 180
    # ...
```

---

## 9. 다음 단계

[기하학적 변환](./04_Geometric_Transforms.md)에서 이미지 크기 조절, 회전, 뒤집기, 어파인/원근 변환 등을 학습합니다!

**다음에 배울 내용**:
- `cv2.resize()`와 보간법
- 회전, 뒤집기 함수
- 어파인 변환 (이동, 회전, 스케일)
- 원근 변환 (문서 스캔)

---

## 10. 참고 자료

### 공식 문서

- [cvtColor() 문서](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html)
- [색상 공간 변환](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html)
- [inRange() 문서](https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html)

### 관련 학습 자료

| 폴더 | 관련 내용 |
|------|----------|
| [이미지 기초 연산](./02_Image_Basics.md) | 이미지 읽기, 픽셀 접근 |
| [이진화 및 임계처리](./07_Thresholding.md) | HSV 기반 임계처리 |

### 색상 공간 참고

- [색상 공간 위키피디아](https://en.wikipedia.org/wiki/Color_space)
- [HSV 색상 모델](https://en.wikipedia.org/wiki/HSL_and_HSV)
- [CIELAB 색상 공간](https://en.wikipedia.org/wiki/CIELAB_color_space)

