# 이진화 및 임계처리

## 개요

이진화(Binarization)는 그레이스케일 이미지를 흑백 이미지로 변환하는 과정입니다. 임계값(Threshold)을 기준으로 픽셀을 0 또는 255로 분류합니다. 이 문서에서는 다양한 임계처리 방법과 실전 활용 기법을 학습합니다.

**난이도**: ⭐⭐ (초급-중급)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `cv2.threshold()` 함수와 다양한 플래그
2. OTSU 자동 임계값 결정
3. 적응형 임계처리 (Adaptive Threshold)
4. 다중 임계처리
5. HSV 색상 기반 임계처리
6. 문서 이진화 및 그림자 처리

---

## 목차

1. [이진화 개요](#1-이진화-개요)
2. [전역 임계처리 - threshold()](#2-전역-임계처리---threshold)
3. [OTSU 자동 임계값](#3-otsu-자동-임계값)
4. [적응형 임계처리 - adaptiveThreshold()](#4-적응형-임계처리---adaptivethreshold)
5. [다중 임계처리](#5-다중-임계처리)
6. [HSV 색상 기반 임계처리](#6-hsv-색상-기반-임계처리)
7. [문서 이진화와 그림자 처리](#7-문서-이진화와-그림자-처리)
8. [연습 문제](#8-연습-문제)
9. [다음 단계](#9-다음-단계)
10. [참고 자료](#10-참고-자료)

---

## 1. 이진화 개요

### 이진화란?

```
┌─────────────────────────────────────────────────────────────────┐
│                      Binarization Concept                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Grayscale Image (0-255)         Binary Image (0 or 255)      │
│   ┌─────────────────────┐        ┌─────────────────────┐       │
│   │░░░▒▒▒▓▓▓███████████│  ───▶  │     █████████████████│       │
│   │░░░░▒▒▒▓▓▓██████████│        │     █████████████████│       │
│   │░░░░░▒▒▒▓▓▓█████████│        │     █████████████████│       │
│   └─────────────────────┘        └─────────────────────┘       │
│                                                                 │
│   Based on Threshold (T):                                      │
│   - Pixel value > T → White (255)                              │
│   - Pixel value ≤ T → Black (0)                                │
│                                                                 │
│   Use Cases:                                                    │
│   - Object-background separation                               │
│   - Document scanning                                          │
│   - Preprocessing for contour detection                        │
│   - Mask generation                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 임계처리 유형

```
┌─────────────────────────────────────────────────────────────────┐
│                     Thresholding Types                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Global Thresholding                                          │
│   - Apply single threshold to entire image                     │
│   - Suitable for uniformly lit images                          │
│   - cv2.threshold()                                             │
│                                                                 │
│   Adaptive Thresholding                                        │
│   - Apply different thresholds to different regions            │
│   - Suitable for unevenly lit images                           │
│   - cv2.adaptiveThreshold()                                     │
│                                                                 │
│   Example:                                                      │
│   ┌────────────────┐      ┌────────────────┐                   │
│   │ Bright  Dark   │      │ Bright  Dark   │                   │
│   │  ██      ██    │      │  ██      ██    │                   │
│   │  ██      ██    │      │  ██      ██    │                   │
│   └────────────────┘      └────────────────┘                   │
│   Original with shadow     Global: Partial loss                │
│                           Adaptive: Full detection             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 전역 임계처리 - threshold()

### 기본 사용법

```python
import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# threshold(src, thresh, maxval, type)
# src: Input image (grayscale)
# thresh: Threshold value
# maxval: Maximum value (usually 255)
# type: Thresholding type
# Returns: (threshold used, result image)

ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

print(f"Threshold used: {ret}")

cv2.imshow('Original', img)
cv2.imshow('Binary', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 임계처리 타입

```
┌─────────────────────────────────────────────────────────────────┐
│                     Thresholding Types                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input pixel value distribution:                              │
│   ▲                                                            │
│   │     ░░░░░▒▒▒▒▒▓▓▓▓▓███████                                │
│   │     ░░░░░░▒▒▒▒▒▓▓▓▓▓██████                                │
│   └──────────────┬───────────────▶ Pixel value                │
│                  T (Threshold)                                 │
│                                                                 │
│   THRESH_BINARY:          dst = maxval if src > T else 0       │
│   value > T → 255, value ≤ T → 0                              │
│                                                                 │
│   THRESH_BINARY_INV:      dst = 0 if src > T else maxval       │
│   value > T → 0, value ≤ T → 255 (inverted)                   │
│                                                                 │
│   THRESH_TRUNC:           dst = T if src > T else src          │
│   value > T → T, value ≤ T → keep                             │
│                                                                 │
│   THRESH_TOZERO:          dst = src if src > T else 0          │
│   value > T → keep, value ≤ T → 0                             │
│                                                                 │
│   THRESH_TOZERO_INV:      dst = 0 if src > T else src          │
│   value > T → 0, value ≤ T → keep                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 타입별 결과 비교

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
thresh = 127

threshold_types = [
    ('BINARY', cv2.THRESH_BINARY),
    ('BINARY_INV', cv2.THRESH_BINARY_INV),
    ('TRUNC', cv2.THRESH_TRUNC),
    ('TOZERO', cv2.THRESH_TOZERO),
    ('TOZERO_INV', cv2.THRESH_TOZERO_INV),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].imshow(img, cmap='gray')
axes[0].set_title(f'Original')

for ax, (name, thresh_type) in zip(axes[1:], threshold_types):
    _, result = cv2.threshold(img, thresh, 255, thresh_type)
    ax.imshow(result, cmap='gray')
    ax.set_title(f'{name}')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
```

### 임계값 선택 가이드

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_optimal_threshold(img):
    """Find appropriate threshold through histogram analysis"""
    # Calculate histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.flatten()

    # Test with various thresholds
    thresholds = [64, 96, 127, 160, 192]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Display histogram
    axes[0, 0].plot(hist)
    axes[0, 0].set_title('Histogram')
    axes[0, 0].axvline(x=127, color='r', linestyle='--', label='T=127')
    axes[0, 0].legend()

    # Original
    axes[0, 1].imshow(img, cmap='gray')
    axes[0, 1].set_title('Original')

    # Results with various thresholds
    for ax, t in zip(axes.flatten()[2:], thresholds):
        _, binary = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
        ax.imshow(binary, cmap='gray')
        ax.set_title(f'Threshold = {t}')

    for ax in axes.flatten():
        ax.axis('off')
    axes[0, 0].axis('on')

    plt.tight_layout()
    plt.show()


img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
find_optimal_threshold(img)
```

---

## 3. OTSU 자동 임계값

임계값을 수동으로 선택하려면 각 이미지를 일일이 검사해야 하므로 배치 처리에는 비실용적입니다. Otsu(오츠) 방법은 임계값 선택을 최적화 문제로 접근하여 이를 해결합니다. 히스토그램을 두 개의 간결하고 잘 분리된 클러스터로 가장 잘 나누는 값을 찾습니다. 결과적으로 다양한 노출 조건의 이미지에서 일관되게 동작하는 데이터 기반 임계값을 얻을 수 있습니다.

### OTSU 알고리즘

```
┌─────────────────────────────────────────────────────────────────┐
│                       OTSU Algorithm                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   The OTSU method automatically finds the optimal threshold    │
│   by analyzing the histogram.                                  │
│                                                                 │
│   Principle:                                                    │
│   - Separate histogram into two classes                        │
│   - Maximize between-class variance                            │
│   - Or minimize within-class variance                          │
│                                                                 │
│   Histogram Example:                                            │
│   ▲                                                            │
│   │   ████                    ████                             │
│   │  ██████                 ████████                           │
│   │ ████████               ██████████                          │
│   └────────────────┬───────────────────▶                       │
│                    T (Threshold found by OTSU)                 │
│    Background class     Foreground class                       │
│                                                                 │
│   Suitable for:                                                 │
│   - Bimodal histogram (two peaks)                              │
│   - Clear separation between background and foreground         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### OTSU 사용법

```python
import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Why thresh=0 when using OTSU: the value is ignored — OpenCV overwrites it with the
# computed optimal threshold; passing 0 signals intent and avoids confusion
ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ret is the threshold Otsu found; inspecting it tells you about the image contrast
print(f"Threshold determined by OTSU: {ret}")

cv2.imshow('Original', img)
cv2.imshow('OTSU Binary', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### OTSU vs 고정 임계값 비교

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('document.jpg', cv2.IMREAD_GRAYSCALE)

# Fixed threshold
_, fixed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# OTSU automatic threshold
ret_otsu, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')

axes[1].imshow(fixed, cmap='gray')
axes[1].set_title('Fixed (T=127)')

axes[2].imshow(otsu, cmap='gray')
axes[2].set_title(f'OTSU (T={ret_otsu:.0f})')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
```

### 가우시안 블러 + OTSU (노이즈 처리)

```python
import cv2

img = cv2.imread('noisy_image.jpg', cv2.IMREAD_GRAYSCALE)

# Direct OTSU
_, otsu_direct = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Why blur before OTSU: noise creates many tiny histogram spikes that can shift
# Otsu's variance calculation toward a wrong valley; blurring merges these spikes
# back into the two main peaks, making the bimodal structure clearer
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret, otsu_blur = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"OTSU threshold after blur: {ret}")

cv2.imshow('Direct OTSU', otsu_direct)
cv2.imshow('Blur + OTSU', otsu_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 4. 적응형 임계처리 - adaptiveThreshold()

전역 임계처리(Otsu 포함)는 이미지 전체에 하나의 임계값을 사용하는데, 조명이 불균일한 경우에는 실패합니다. 예를 들어 문서의 그림자 부분은 밝은 쪽의 가장 밝은 전경 픽셀보다도 어두울 수 있습니다. 적응형 임계처리(Adaptive Thresholding)는 각 픽셀의 주변 영역을 기반으로 별도의 임계값을 계산하므로, 조명 기울기(Gradient)에 강인하며 문서 스캐닝의 표준 방법으로 자리잡고 있습니다.

### 적응형 임계처리란?

```
┌─────────────────────────────────────────────────────────────────┐
│                   Adaptive Thresholding                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Problem: Unevenly lit image                                  │
│   ┌─────────────────────────────────────────┐                   │
│   │ ████████           ░░░░░░░░             │                   │
│   │ Bright area        Dark area            │                   │
│   │ (with text)        (with text)          │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                 │
│   Global thresholding:                                          │
│   - Process entire image with one threshold                    │
│   - Bright area OK, dark area text lost (or vice versa)        │
│                                                                 │
│   Adaptive thresholding:                                        │
│   - Determine local threshold by analyzing surrounding area    │
│     for each pixel                                             │
│   - Robust to lighting changes                                 │
│                                                                 │
│   ┌─────────────────────────────────────────┐                   │
│   │ Local area 1      Local area 2          │                   │
│   │ T = 200           T = 100               │                   │
│   │ (bright area)     (dark area)           │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 사용법

```python
import cv2

img = cv2.imread('document.jpg', cv2.IMREAD_GRAYSCALE)

# adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType,
#                   blockSize, C)
# adaptiveMethod: ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C
# blockSize: Local area size (must be odd — required so there is a single center pixel)
# C: Constant subtracted from calculated mean/weighted mean

# MEAN_C: treats all neighbors equally — fast but can be noisy at edges
adaptive_mean = cv2.adaptiveThreshold(
    img, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    11, 2  # blockSize=11 captures ~11px lighting variation; C=2 prevents noise pixels
           # from flipping to white (background subtraction)
)

# Why GAUSSIAN_C preferred: pixels near the center of the block are more likely to
# share the same illumination as the target pixel; down-weighting distant neighbors
# produces a smoother, less noisy threshold map than MEAN_C
adaptive_gaussian = cv2.adaptiveThreshold(
    img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)

cv2.imshow('Original', img)
cv2.imshow('Adaptive Mean', adaptive_mean)
cv2.imshow('Adaptive Gaussian', adaptive_gaussian)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 파라미터 조정

```
┌─────────────────────────────────────────────────────────────────┐
│                  adaptiveThreshold Parameters                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   blockSize (local area size):                                 │
│   - Small values (e.g., 3, 5): Preserve fine details,          │
│     sensitive to noise                                         │
│   - Large values (e.g., 31, 51): Smooth results, may lose      │
│     detail                                                     │
│   - Typically use 11 ~ 31                                      │
│                                                                 │
│   C (constant):                                                 │
│   - Value subtracted from calculated threshold                 │
│   - Positive: More pixels become white                         │
│   - Negative: More pixels become black                         │
│   - Typically use 2 ~ 10                                       │
│                                                                 │
│   Threshold calculation:                                        │
│   T(x,y) = mean(blockSize × blockSize area) - C               │
│                                                                 │
│   Geometric intuition: the local mean estimates the background  │
│   brightness around pixel (x,y); subtracting C lowers the bar  │
│   so that only pixels noticeably brighter than their surroundings│
│   (i.e., ink on paper) pass the test                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('document_shadow.jpg', cv2.IMREAD_GRAYSCALE)

# Test various parameter combinations
params = [
    (11, 2),
    (11, 5),
    (21, 2),
    (21, 5),
    (31, 2),
    (31, 10),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for ax, (block_size, c) in zip(axes, params):
    result = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c
    )
    ax.imshow(result, cmap='gray')
    ax.set_title(f'blockSize={block_size}, C={c}')
    ax.axis('off')

plt.tight_layout()
plt.show()
```

### 전역 vs 적응형 비교

```python
import cv2
import matplotlib.pyplot as plt

# Document image with shadow
img = cv2.imread('document_with_shadow.jpg', cv2.IMREAD_GRAYSCALE)

# Global thresholding
_, global_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# OTSU
_, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive
adaptive = cv2.adaptiveThreshold(
    img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    21, 10
)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original')

axes[0, 1].imshow(global_thresh, cmap='gray')
axes[0, 1].set_title('Global (T=127)')

axes[1, 0].imshow(otsu, cmap='gray')
axes[1, 0].set_title('OTSU')

axes[1, 1].imshow(adaptive, cmap='gray')
axes[1, 1].set_title('Adaptive Gaussian')

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## 5. 다중 임계처리

### 다중 레벨 임계처리

```python
import cv2
import numpy as np

def multi_threshold(img, thresholds):
    """
    Multi-level thresholding

    Parameters:
    - img: Grayscale image
    - thresholds: List of threshold values [T1, T2, T3, ...]

    Returns:
    - Labeled image (0, 1, 2, 3, ...)
    """
    result = np.zeros_like(img)
    thresholds = sorted(thresholds)

    for i, t in enumerate(thresholds):
        result[img > t] = (i + 1) * (255 // (len(thresholds)))

    return result


img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 3-level separation (dark, medium, bright)
result = multi_threshold(img, [85, 170])

# 4-level separation
result4 = multi_threshold(img, [64, 128, 192])

cv2.imshow('Original', img)
cv2.imshow('3 Levels', result)
cv2.imshow('4 Levels', result4)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 컬러맵 적용

```python
import cv2
import numpy as np

def quantize_colors(img, levels=4):
    """Quantize image into n levels"""
    # Calculate step value
    step = 256 // levels
    quantized = (img // step) * step

    return quantized


img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Quantization
quantized = quantize_colors(img, levels=8)

# Apply colormap
colored = cv2.applyColorMap(quantized, cv2.COLORMAP_JET)

cv2.imshow('Original', img)
cv2.imshow('Quantized', quantized)
cv2.imshow('Colored', colored)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 6. HSV 색상 기반 임계처리

### 색상 범위 마스킹

```python
import cv2
import numpy as np

img = cv2.imread('colorful_image.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define blue color range
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])

# Create mask with inRange
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Apply mask
result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('Original', img)
cv2.imshow('Mask', mask)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 실시간 색상 범위 조정

```python
import cv2
import numpy as np

def nothing(x):
    pass

# Create window and trackbars
cv2.namedWindow('Controls')
cv2.createTrackbar('H_Low', 'Controls', 0, 179, nothing)
cv2.createTrackbar('H_High', 'Controls', 179, 179, nothing)
cv2.createTrackbar('S_Low', 'Controls', 0, 255, nothing)
cv2.createTrackbar('S_High', 'Controls', 255, 255, nothing)
cv2.createTrackbar('V_Low', 'Controls', 0, 255, nothing)
cv2.createTrackbar('V_High', 'Controls', 255, 255, nothing)

img = cv2.imread('colorful_image.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

while True:
    h_low = cv2.getTrackbarPos('H_Low', 'Controls')
    h_high = cv2.getTrackbarPos('H_High', 'Controls')
    s_low = cv2.getTrackbarPos('S_Low', 'Controls')
    s_high = cv2.getTrackbarPos('S_High', 'Controls')
    v_low = cv2.getTrackbarPos('V_Low', 'Controls')
    v_high = cv2.getTrackbarPos('V_High', 'Controls')

    lower = np.array([h_low, s_low, v_low])
    upper = np.array([h_high, s_high, v_high])

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('Original', img)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

### 주요 색상 범위

```
┌─────────────────────────────────────────────────────────────────┐
│                    HSV Color Range Guide                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Color        H (Hue)        S (Saturation)   V (Value)       │
│   ───────────────────────────────────────────────────────      │
│   Red          0-10           100-255          100-255         │
│   (wrapping)   160-179        100-255          100-255         │
│                                                                 │
│   Orange       10-25          100-255          100-255         │
│                                                                 │
│   Yellow       25-35          100-255          100-255         │
│                                                                 │
│   Green        35-85          100-255          100-255         │
│                                                                 │
│   Cyan         85-95          100-255          100-255         │
│                                                                 │
│   Blue         95-130         100-255          100-255         │
│                                                                 │
│   Purple       130-160        100-255          100-255         │
│                                                                 │
│   White        0-179          0-30             200-255         │
│                                                                 │
│   Black        0-179          0-255            0-50            │
│                                                                 │
│   Gray         0-179          0-30             50-200          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 문서 이진화와 그림자 처리

### 문서 이진화 파이프라인

```python
import cv2
import numpy as np

def binarize_document(img, method='adaptive'):
    """
    Document image binarization

    Parameters:
    - img: Input image (color or grayscale)
    - method: 'adaptive', 'otsu', 'combined'
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if method == 'otsu':
        # Why blur before OTSU: smoothing collapses noise spikes in the histogram so
        # Otsu finds the true valley between background and foreground peaks
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method == 'adaptive':
        # Why blockSize=21, C=15: a 21px block handles typical shadow gradients in
        # A4 scans; C=15 is aggressive enough to suppress paper texture noise
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21, 15
        )

    elif method == 'combined':
        # Combine OTSU + Adaptive
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu = cv2.threshold(blur, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21, 15
        )

        # Why AND: a pixel is kept only if *both* methods agree it is foreground;
        # this intersection removes false positives that each method produces alone
        binary = cv2.bitwise_and(otsu, adaptive)

    return binary


img = cv2.imread('document.jpg')
binary = binarize_document(img, method='adaptive')
```

### 그림자 제거

```python
import cv2
import numpy as np

def remove_shadow(img):
    """
    Remove shadows from document image
    """
    # Split RGB
    rgb_planes = cv2.split(img)
    result_planes = []

    for plane in rgb_planes:
        # Estimate background with dilation
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))

        # Remove noise with medianBlur
        bg = cv2.medianBlur(dilated, 21)

        # Calculate difference and normalize
        diff = 255 - cv2.absdiff(plane, bg)

        # Enhance contrast
        normalized = cv2.normalize(diff, None, alpha=0, beta=255,
                                    norm_type=cv2.NORM_MINMAX)
        result_planes.append(normalized)

    result = cv2.merge(result_planes)
    return result


def binarize_with_shadow_removal(img):
    """Binarize after shadow removal"""
    # Remove shadow
    no_shadow = remove_shadow(img)

    # Convert to grayscale
    gray = cv2.cvtColor(no_shadow, cv2.COLOR_BGR2GRAY)

    # Adaptive binarization
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 10
    )

    return binary, no_shadow


img = cv2.imread('document_with_shadow.jpg')
binary, no_shadow = binarize_with_shadow_removal(img)

cv2.imshow('Original', img)
cv2.imshow('Shadow Removed', no_shadow)
cv2.imshow('Binary', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Divide 기법 (배경 나누기)

```python
import cv2
import numpy as np

def divide_binarization(img, blur_kernel=21):
    """
    Binarization after correcting uneven illumination with divide technique

    Principle: original / background = uniform image
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Why strong blur (blur_kernel=21): we want the background illumination map,
    # not any text content; the kernel must be large enough that all text is blurred away
    bg = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Why divide: gray/bg normalizes each pixel by local brightness; a dark ink pixel
    # on a shadowed background divides to a low ratio just like on a bright background,
    # making the result illumination-independent
    divided = cv2.divide(gray, bg, scale=255)

    # Why OTSU on the divided image: after division the histogram is reliably bimodal
    # (ink vs paper), so Otsu finds a stable threshold without manual tuning
    _, binary = cv2.threshold(divided, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary, divided


img = cv2.imread('document_uneven_lighting.jpg')
binary, divided = divide_binarization(img)

cv2.imshow('Original', img)
cv2.imshow('Divided', divided)
cv2.imshow('Binary', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 8. 연습 문제

### 연습 1: 최적 임계값 자동 탐색

히스토그램을 분석하여 바이모달 분포의 두 봉우리 사이 최적 임계값을 찾는 함수를 구현하세요. OTSU 결과와 비교해보세요.

```python
def find_valley_threshold(img):
    """
    Find the valley between two peaks in the histogram
    and return it as the threshold
    """
    # Hint: Use scipy.signal.find_peaks or
    # Smooth histogram and find minimum
    pass
```

### 연습 2: 적응형 임계처리 파라미터 튜닝 GUI

트랙바를 사용하여 `blockSize`와 `C` 값을 실시간으로 조정하면서 결과를 확인할 수 있는 프로그램을 작성하세요.

### 연습 3: 명함 스캐너

명함 이미지를 입력받아 다음 과정을 수행하는 프로그램을 작성하세요:
1. 그림자/조명 불균일 보정
2. 이진화
3. 노이즈 제거 (모폴로지 연산)
4. 결과 저장

### 연습 4: 색상 분리 도구

이미지에서 특정 색상 영역을 추출하고, 추출된 영역의 면적을 계산하는 함수를 작성하세요. 예: "빨간색 영역이 전체의 15%를 차지함"

### 연습 5: 히스테리시스 임계처리

Canny 엣지 검출에서 사용되는 히스테리시스 임계처리를 직접 구현하세요:
- 높은 임계값 이상: 확실한 엣지
- 낮은 임계값 이하: 확실히 비엣지
- 중간: 확실한 엣지와 연결된 경우만 엣지

```python
def hysteresis_threshold(img, low_thresh, high_thresh):
    """
    Implement hysteresis thresholding
    """
    pass
```

---

## 9. 다음 단계

[엣지 검출 (Edge Detection)](./08_Edge_Detection.md)에서 Sobel, Canny 등 다양한 엣지 검출 기법을 학습합니다!

**다음에 배울 내용**:
- Sobel, Scharr 미분 연산자
- Laplacian 엣지 검출
- Canny 엣지 검출 알고리즘
- 엣지 기반 객체 검출

---

## 10. 참고 자료

### 공식 문서

- [threshold() 문서](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57)
- [adaptiveThreshold() 문서](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3)
- [inRange() 문서](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981)

### 관련 학습 자료

| 폴더 | 관련 내용 |
|------|----------|
| [색상 공간](./03_Color_Spaces.md) | HSV 색상 공간 |
| [모폴로지 연산](./06_Morphology.md) | 이진화 후 노이즈 제거 |

### 추가 참고

- [OTSU 알고리즘 설명](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
- [문서 이진화 기법](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_niblack_sauvola.html)

