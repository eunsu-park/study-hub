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

### 이론: 분류 문제로서의 임계처리

그레이스케일 이미지 `I(x, y)`와 임계값 `T`가 주어지면, 전역 임계처리는 이진 이미지를 생성합니다:

```
B(x, y) = 1  if I(x, y) > T
B(x, y) = 0  otherwise
```

이것은 1차원 특징(픽셀 밝기)과 단일 결정 경계 `T`를 가진 분류입니다. 이 모델이 올바른 경우:

1. 전경과 배경이 **명백히 다른 밝기**를 가진다.
2. 그 밝기가 **공간적으로 일관**되어 있다 — 전경이 이미지 전체에서 같은 밝기 범위를 가지고, 배경도 마찬가지다.

둘 다 성립하면 이미지 히스토그램은 **쌍봉(bimodal)** — 클래스당 한 봉우리씩, 두 봉우리가 분리됨 — 이고, 이상적인 임계값은 두 봉우리 사이의 골짜기에 있습니다. 이 레슨의 방법들은 대부분 그 골짜기를 어떻게 찾는지에서 차이가 납니다.

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

### 이론: Otsu 방법: 자동 임계값 선택

Otsu 방법(1979)은 *히스토그램만 이용해* 두 클래스를 가장 잘 분리하는 임계값 `T*`를 고릅니다. 파라미터가 필요 없으며, 쌍봉 히스토그램 가정 하에서 거의 최적의 결과를 줍니다.

#### B.1 목적 함수

후보 임계값 `t`에 대해 픽셀을 두 클래스 `C₀ = { p : I(p) ≤ t }`, `C₁ = { p : I(p) > t }`로 분할합니다. 다음을 정의하면:

- `ω₀(t)`, `ω₁(t)` = 각 클래스의 픽셀 비율 (`ω₀ + ω₁ = 1`).
- `μ₀(t)`, `μ₁(t)` = 각 클래스의 평균 밝기.
- `σ₀²(t)`, `σ₁²(t)` = 각 클래스의 분산.

Otsu는 **클래스 내 분산**(within-class spread)을 정의합니다:

```
σ_W²(t) = ω₀(t) · σ₀²(t) + ω₁(t) · σ₁²(t)
```

좋은 임계값은 각 클래스를 빽빽하게 만듭니다 — 각 그룹 내 분산이 작아야 합니다. Otsu는 `T* = argmin_t σ_W²(t)`를 고릅니다.

#### B.2 클래스 간 분산 단축

모든 `t`에서 `σ_W²`를 계산하는 것은 비용이 큽니다(두 분산을 모두 다시 계산). 핵심 항등식이 이미지 전체에 대해 상수인 전체 분산 `σ²`를 다음과 같이 분해합니다:

```
σ² = σ_W²(t) + σ_B²(t)
```

여기서 `σ_B²(t) = ω₀(t) · ω₁(t) · (μ₁(t) - μ₀(t))²`가 **클래스 간 분산**(inter-class variance)입니다. `σ²`가 상수이므로 **`σ_W²` 최소화는 `σ_B²` 최대화와 동등**합니다. 그리고 `σ_B²`는 클래스 비율과 평균만 필요하므로, `t`가 0에서 255로 훑어갈 때 점진적으로 갱신할 수 있습니다 — 256-빈 히스토그램을 한 번 훑는 것으로 충분합니다.

#### B.3 Otsu가 실패할 때

이 유도는 히스토그램이 쌍봉이라 가정합니다. 그렇지 않으면 — 전경만 있고 배경이 없는 이미지, 명확한 봉우리 없는 노이즈성 그라데이션, 삼봉 장면 — Otsu는 여전히 *어떤* 임계값을 반환하지만, 무의미할 수 있습니다. 좋은 실천: 자동 선택을 믿기 전에 히스토그램을 그려 쌍봉성을 시각적으로 확인하세요.

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

### 이론: 단일 전역 임계값이 실패하는 이유: 불균일 조명

불균일한 조명 아래 촬영된 문서를 생각해 봅시다 — 왼쪽은 창가 쪽이라 밝고, 오른쪽은 그늘져 어둡습니다. 왼쪽의 *잉크 픽셀*이 오른쪽의 *빈 종이 픽셀*보다 밝을 수 있습니다. 단일 전역 임계값은:

- 밝은 쪽의 잉크를 올바로 분류하지만 그늘의 종이를 잉크로 분류(임계값이 너무 낮음), 또는
- 그늘의 종이를 올바로 분류하지만 바랜 잉크를 종이로 분류(임계값이 너무 높음).

히스토그램은 분리된 두 봉우리 대신 흐릿한 단일 뭉치가 되어 쌍봉 가정이 깨집니다. 해결책은 **임계값이 이미지 내에서 변하도록** 하는 것입니다.

### 이론: 적응형 임계처리

적응형 임계처리는 작은 주변 창의 통계에 기반해 **픽셀마다 다른 임계값**을 고릅니다:

```
T(x, y) = f({ I(x', y') : (x', y') ∈ (x, y) 주변 창 }) - C
B(x, y) = 1  if I(x, y) > T(x, y)
```

통계 `f`와 상수 `C`가 동작을 제어합니다.

#### D.1 평균 적응형 (`ADAPTIVE_THRESH_MEAN_C`)

`f`는 창의 산술 평균. 국소 임계값 = 국소 평균 − `C`. 가장 저렴한 옵션 — 내부적으로 box filter(§05)를 사용합니다.

효과: 임계값이 국소 배경 밝기를 따라갑니다. 어두운 영역은 낮은 임계값, 밝은 영역은 높은 임계값. "주변보다 최소 `C`만큼 어두운" 잉크는 절대 밝기와 무관하게 전경이 됩니다. 전역 조명이 크게 변해도 작동합니다.

#### D.2 가우시안 적응형 (`ADAPTIVE_THRESH_GAUSSIAN_C`)

`f`는 가우시안 가중 평균 — 가까운 픽셀이 먼 픽셀보다 더 기여합니다. 내부적으로 가우시안 블러를 사용. 창이 밝은 영역과 어두운 영역을 걸칠 때 가중치가 가장자리 아티팩트를 줄여주어 평면 평균 버전보다 정확합니다. 조금 더 비쌈.

#### D.3 창 크기와 `C` 선택

- **창 크기**는 검출하려는 전경 특징보다 커야 하지만, 조명 변화의 스케일보다는 작아야 합니다. 글자의 경우 글자 높이의 몇 배 정도가 잘 작동합니다. 창이 너무 작으면 임계값이 전경 자체를 따라가고(모든 것이 회색), 창이 너무 크면 전역 임계처리로 돌아갑니다.
- **`C`**는 임계값을 국소 평균에서 "약간 어두운 쪽"으로 옮기는 편향. `C`가 클수록 → 배경보다 *확연히* 어두운 픽셀만 전경(깨끗하지만 희미한 특징을 놓칠 수 있음). 일반 값: 8비트 이미지에서 2–15.

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

### 이론: 이진 전역 임계처리를 넘어

#### E.1 다중 임계처리

어떤 장면은 본질적으로 3개 이상의 클래스를 가집니다(예: 하늘, 잎사귀, 건물). Multi-Otsu는 §B를 일반화해 쌍별 클래스 간 분산의 합을 최대화함으로써 히스토그램을 `k`개 클래스로 분할합니다. OpenCV는 직접 포함하지 않지만, `skimage.filters.threshold_multiotsu`가 제공합니다.

#### E.2 `inRange`를 사용한 색상 임계처리

구별 특징이 밝기가 아니라 색일 때, `cv2.inRange(img, lower, upper)`가 각 채널을 독립적으로 임계처리하고 결과를 AND합니다 — 값이 색 공간의 박스 안에 떨어지는 픽셀만 유지. HSV에서(§03) 이 박스는 "충분한 채도와 밝기에서의 hue 슬랩"이 되며, 이것이 색상 기반 객체 분할의 전형적 방식입니다.

#### E.3 히스테리시스 임계처리

단일 임계값은 모든 픽셀을 강한 분류로 강제하는데, `T` 바로 근처의 픽셀은 임의적입니다. 히스테리시스는 *두* 임계값 `T_low < T_high`를 씁니다. `T_high` 이상의 픽셀은 확실히 전경, `T_low` 이하의 픽셀은 확실히 배경, 둘 사이의 픽셀은 확실한-전경 픽셀에 연결된 경우에만 전경입니다. Canny(§08)가 최종 에지 연결 단계에서 이를 다루는 방식이며, 중간 값에서의 결정이 이웃에 의존해야 하는 모든 영역에서 임계처리로 일반화됩니다.

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

