# 이진화 및 임계처리

## 개요

이진화(Binarization)는 그레이스케일 이미지를 흑백 이미지로 변환하는 과정입니다. 임계값(Threshold)을 기준으로 픽셀을 0 또는 255로 분류합니다. 이 문서에서는 다양한 임계처리 방법과 실전 활용 기법을 학습합니다.

**난이도**: ⭐⭐ (초급-중급)

**학습 목표**:
- `cv2.threshold()` 함수와 다양한 플래그
- OTSU 자동 임계값 결정
- 적응형 임계처리 (Adaptive Threshold)
- 다중 임계처리
- HSV 색상 기반 임계처리
- 문서 이진화 및 그림자 처리

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
│                         이진화 개념                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   그레이스케일 이미지 (0-255)      이진 이미지 (0 또는 255)       │
│   ┌─────────────────────┐        ┌─────────────────────┐       │
│   │░░░▒▒▒▓▓▓███████████│  ───▶  │     █████████████████│       │
│   │░░░░▒▒▒▓▓▓██████████│        │     █████████████████│       │
│   │░░░░░▒▒▒▓▓▓█████████│        │     █████████████████│       │
│   └─────────────────────┘        └─────────────────────┘       │
│                                                                 │
│   임계값(T) 기준:                                                │
│   - 픽셀 값 > T → 흰색 (255)                                    │
│   - 픽셀 값 ≤ T → 검정 (0)                                      │
│                                                                 │
│   사용 목적:                                                     │
│   - 객체와 배경 분리                                            │
│   - 문서 스캔                                                   │
│   - 윤곽선 검출 전처리                                          │
│   - 마스크 생성                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 임계처리 유형

```
┌─────────────────────────────────────────────────────────────────┐
│                      임계처리 유형                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   전역 임계처리 (Global Thresholding)                            │
│   - 이미지 전체에 단일 임계값 적용                                │
│   - 조명이 균일한 이미지에 적합                                  │
│   - cv2.threshold()                                             │
│                                                                 │
│   적응형 임계처리 (Adaptive Thresholding)                        │
│   - 영역별로 다른 임계값 적용                                    │
│   - 조명이 불균일한 이미지에 적합                                │
│   - cv2.adaptiveThreshold()                                     │
│                                                                 │
│   예시:                                                         │
│   ┌────────────────┐      ┌────────────────┐                   │
│   │ 밝음   어두움   │      │ 밝음   어두움   │                   │
│   │  ██      ██    │      │  ██      ██    │                   │
│   │  ██      ██    │      │  ██      ██    │                   │
│   └────────────────┘      └────────────────┘                   │
│      그림자 있는 원본           전역: 일부 손실                   │
│                              적응형: 전체 검출                   │
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
# src: 입력 이미지 (그레이스케일)
# thresh: 임계값
# maxval: 최대값 (보통 255)
# type: 임계처리 타입
# 반환: (사용된 임계값, 결과 이미지)

ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

print(f"사용된 임계값: {ret}")

cv2.imshow('Original', img)
cv2.imshow('Binary', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 임계처리 타입

```
┌─────────────────────────────────────────────────────────────────┐
│                      임계처리 타입                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   입력 픽셀 값 분포:                                             │
│   ▲                                                            │
│   │     ░░░░░▒▒▒▒▒▓▓▓▓▓███████                                │
│   │     ░░░░░░▒▒▒▒▒▓▓▓▓▓██████                                │
│   └──────────────┬───────────────▶ 픽셀값                      │
│                  T (임계값)                                     │
│                                                                 │
│   THRESH_BINARY:          dst = maxval if src > T else 0       │
│   값 > T → 255, 값 ≤ T → 0                                     │
│                                                                 │
│   THRESH_BINARY_INV:      dst = 0 if src > T else maxval       │
│   값 > T → 0, 값 ≤ T → 255 (반전)                              │
│                                                                 │
│   THRESH_TRUNC:           dst = T if src > T else src          │
│   값 > T → T, 값 ≤ T → 유지                                    │
│                                                                 │
│   THRESH_TOZERO:          dst = src if src > T else 0          │
│   값 > T → 유지, 값 ≤ T → 0                                    │
│                                                                 │
│   THRESH_TOZERO_INV:      dst = 0 if src > T else src          │
│   값 > T → 0, 값 ≤ T → 유지                                    │
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
    """히스토그램 분석으로 적절한 임계값 찾기"""
    # 히스토그램 계산
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.flatten()

    # 다양한 임계값으로 테스트
    thresholds = [64, 96, 127, 160, 192]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 히스토그램 표시
    axes[0, 0].plot(hist)
    axes[0, 0].set_title('Histogram')
    axes[0, 0].axvline(x=127, color='r', linestyle='--', label='T=127')
    axes[0, 0].legend()

    # 원본
    axes[0, 1].imshow(img, cmap='gray')
    axes[0, 1].set_title('Original')

    # 다양한 임계값 결과
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

### OTSU 알고리즘

```
┌─────────────────────────────────────────────────────────────────┐
│                      OTSU 알고리즘                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   OTSU 방법은 히스토그램을 분석하여 최적의 임계값을 자동으로        │
│   찾아주는 알고리즘입니다.                                        │
│                                                                 │
│   원리:                                                         │
│   - 히스토그램을 두 클래스로 분리                                │
│   - 클래스 간 분산(between-class variance) 최대화                │
│   - 또는 클래스 내 분산(within-class variance) 최소화            │
│                                                                 │
│   히스토그램 예시:                                               │
│   ▲                                                            │
│   │   ████                    ████                             │
│   │  ██████                 ████████                           │
│   │ ████████               ██████████                          │
│   └────────────────┬───────────────────▶                       │
│                    T (OTSU가 찾은 임계값)                        │
│       배경 클래스        전경 클래스                              │
│                                                                 │
│   적합한 경우:                                                   │
│   - 바이모달(bimodal) 히스토그램 (두 개의 봉우리)                 │
│   - 배경과 전경이 명확히 구분되는 경우                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### OTSU 사용법

```python
import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# OTSU 플래그 추가 (비트 OR 연산)
# thresh 값은 0으로 설정 (OTSU가 자동으로 결정)
ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"OTSU가 결정한 임계값: {ret}")

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

# 고정 임계값
_, fixed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# OTSU 자동 임계값
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

# 직접 OTSU
_, otsu_direct = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 가우시안 블러 후 OTSU (권장)
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret, otsu_blur = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"블러 후 OTSU 임계값: {ret}")

cv2.imshow('Direct OTSU', otsu_direct)
cv2.imshow('Blur + OTSU', otsu_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 4. 적응형 임계처리 - adaptiveThreshold()

### 적응형 임계처리란?

```
┌─────────────────────────────────────────────────────────────────┐
│                    적응형 임계처리                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   문제 상황: 조명이 불균일한 이미지                               │
│   ┌─────────────────────────────────────────┐                   │
│   │ ████████           ░░░░░░░░             │                   │
│   │ 밝은 영역           어두운 영역           │                   │
│   │ (텍스트 있음)       (텍스트 있음)         │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                 │
│   전역 임계처리:                                                 │
│   - 하나의 임계값으로 전체 처리                                  │
│   - 밝은 영역 OK, 어두운 영역 텍스트 손실 (또는 그 반대)         │
│                                                                 │
│   적응형 임계처리:                                               │
│   - 각 픽셀마다 주변 영역을 분석하여 로컬 임계값 결정             │
│   - 조명 변화에 강건함                                          │
│                                                                 │
│   ┌─────────────────────────────────────────┐                   │
│   │ 로컬 영역 1       로컬 영역 2             │                   │
│   │ T = 200          T = 100                │                   │
│   │ (밝은 영역)       (어두운 영역)           │                   │
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
# adaptiveMethod: ADAPTIVE_THRESH_MEAN_C 또는 ADAPTIVE_THRESH_GAUSSIAN_C
# blockSize: 로컬 영역 크기 (홀수)
# C: 계산된 평균/가중평균에서 빼는 상수

# MEAN_C: 로컬 영역의 평균
adaptive_mean = cv2.adaptiveThreshold(
    img, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    11, 2
)

# GAUSSIAN_C: 로컬 영역의 가우시안 가중 평균 (중심에 더 큰 가중치)
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
│                   adaptiveThreshold 파라미터                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   blockSize (로컬 영역 크기):                                    │
│   - 작은 값 (예: 3, 5): 세밀한 디테일 보존, 노이즈에 민감         │
│   - 큰 값 (예: 31, 51): 부드러운 결과, 세부 정보 손실 가능       │
│   - 보통 11 ~ 31 사용                                           │
│                                                                 │
│   C (상수):                                                      │
│   - 계산된 임계값에서 빼는 값                                    │
│   - 양수: 더 많은 픽셀이 흰색이 됨                               │
│   - 음수: 더 많은 픽셀이 검정이 됨                               │
│   - 보통 2 ~ 10 사용                                            │
│                                                                 │
│   임계값 계산:                                                   │
│   T(x,y) = mean(blockSize × blockSize 영역) - C                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('document_shadow.jpg', cv2.IMREAD_GRAYSCALE)

# 다양한 파라미터 조합 테스트
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

# 그림자가 있는 문서 이미지
img = cv2.imread('document_with_shadow.jpg', cv2.IMREAD_GRAYSCALE)

# 전역 임계처리
_, global_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# OTSU
_, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 적응형
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
    다중 임계처리

    Parameters:
    - img: 그레이스케일 이미지
    - thresholds: 임계값 리스트 [T1, T2, T3, ...]

    Returns:
    - 레이블된 이미지 (0, 1, 2, 3, ...)
    """
    result = np.zeros_like(img)
    thresholds = sorted(thresholds)

    for i, t in enumerate(thresholds):
        result[img > t] = (i + 1) * (255 // (len(thresholds)))

    return result


img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 3단계 분리 (어두움, 중간, 밝음)
result = multi_threshold(img, [85, 170])

# 4단계 분리
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
    """이미지를 n단계로 양자화"""
    # 단계별 값 계산
    step = 256 // levels
    quantized = (img // step) * step

    return quantized


img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 양자화
quantized = quantize_colors(img, levels=8)

# 컬러맵 적용
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

# 파란색 범위 정의
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])

# inRange로 마스크 생성
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 마스크 적용
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

# 윈도우와 트랙바 생성
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
│                    HSV 색상 범위 가이드                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   색상         H (Hue)        S (Saturation)   V (Value)        │
│   ───────────────────────────────────────────────────────       │
│   빨간색       0-10           100-255          100-255          │
│   (래핑)       160-179        100-255          100-255          │
│                                                                 │
│   주황색       10-25          100-255          100-255          │
│                                                                 │
│   노란색       25-35          100-255          100-255          │
│                                                                 │
│   초록색       35-85          100-255          100-255          │
│                                                                 │
│   청록색       85-95          100-255          100-255          │
│                                                                 │
│   파란색       95-130         100-255          100-255          │
│                                                                 │
│   보라색       130-160        100-255          100-255          │
│                                                                 │
│   흰색         0-179          0-30             200-255          │
│                                                                 │
│   검정색       0-179          0-255            0-50             │
│                                                                 │
│   회색         0-179          0-30             50-200           │
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
    문서 이미지 이진화

    Parameters:
    - img: 입력 이미지 (컬러 또는 그레이스케일)
    - method: 'adaptive', 'otsu', 'combined'
    """
    # 그레이스케일 변환
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if method == 'otsu':
        # OTSU
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method == 'adaptive':
        # 적응형
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21, 15
        )

    elif method == 'combined':
        # OTSU + 적응형 결합
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu = cv2.threshold(blur, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21, 15
        )

        # 두 결과의 AND 연산
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
    문서 이미지에서 그림자 제거
    """
    # RGB 분리
    rgb_planes = cv2.split(img)
    result_planes = []

    for plane in rgb_planes:
        # dilate로 배경 추정
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))

        # medianBlur로 노이즈 제거
        bg = cv2.medianBlur(dilated, 21)

        # 차이 계산 및 정규화
        diff = 255 - cv2.absdiff(plane, bg)

        # 대비 향상
        normalized = cv2.normalize(diff, None, alpha=0, beta=255,
                                    norm_type=cv2.NORM_MINMAX)
        result_planes.append(normalized)

    result = cv2.merge(result_planes)
    return result


def binarize_with_shadow_removal(img):
    """그림자 제거 후 이진화"""
    # 그림자 제거
    no_shadow = remove_shadow(img)

    # 그레이스케일 변환
    gray = cv2.cvtColor(no_shadow, cv2.COLOR_BGR2GRAY)

    # 적응형 이진화
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
    Divide 기법으로 조명 불균일 보정 후 이진화

    원리: 원본 / 배경 = 균일한 이미지
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 배경 추정 (강한 블러)
    bg = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # 나누기 연산 (255를 곱해서 범위 유지)
    divided = cv2.divide(gray, bg, scale=255)

    # 이진화
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
    히스토그램의 두 봉우리 사이 골짜기(valley)를 찾아
    임계값으로 반환
    """
    # 힌트: scipy.signal.find_peaks 또는
    # 히스토그램 스무딩 후 최솟값 찾기
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
    히스테리시스 임계처리 구현
    """
    pass
```

---

## 9. 다음 단계

[08_Edge_Detection.md](./08_Edge_Detection.md)에서 Sobel, Canny 등 다양한 엣지 검출 기법을 학습합니다!

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
| [03_Color_Spaces.md](./03_Color_Spaces.md) | HSV 색상 공간 |
| [06_Morphology.md](./06_Morphology.md) | 이진화 후 노이즈 제거 |

### 추가 참고

- [OTSU 알고리즘 설명](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
- [문서 이진화 기법](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_niblack_sauvola.html)

