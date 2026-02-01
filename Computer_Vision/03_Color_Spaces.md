# 색상 공간

## 개요

컴퓨터 비전에서 색상 공간(Color Space)은 색상을 표현하는 방법입니다. OpenCV는 기본적으로 BGR 색상 공간을 사용하지만, 특정 작업에는 HSV, LAB 등 다른 색상 공간이 더 효과적입니다. 이 문서에서는 다양한 색상 공간의 특성과 변환 방법, 그리고 색상 기반 객체 추적을 학습합니다.

**난이도**: ⭐⭐ (초급-중급)

**학습 목표**:
- BGR과 RGB의 차이 이해
- HSV 색상 공간의 원리와 활용
- `cv2.cvtColor()`를 사용한 색상 공간 변환
- 채널 분리/병합
- 색상 기반 객체 추적 구현

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
│                    BGR vs RGB 비교                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   OpenCV (BGR)                 대부분의 라이브러리 (RGB)         │
│   ┌─────────────┐              ┌─────────────┐                 │
│   │ B │ G │ R │               │ R │ G │ B │                   │
│   │[0]│[1]│[2]│               │[0]│[1]│[2]│                   │
│   └─────────────┘              └─────────────┘                 │
│                                                                 │
│   순수한 빨간색:               순수한 빨간색:                    │
│   [0, 0, 255]                  [255, 0, 0]                      │
│                                                                 │
│   순수한 파란색:               순수한 파란색:                    │
│   [255, 0, 0]                  [0, 0, 255]                      │
│                                                                 │
│   OpenCV 사용 라이브러리:       RGB 사용 라이브러리:             │
│   - cv2.imread()               - matplotlib                     │
│   - cv2.imshow()               - PIL/Pillow                     │
│   - cv2.imwrite()              - Tkinter                        │
│                                - 웹 브라우저 (CSS/HTML)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### BGR을 사용하는 이유

역사적인 이유입니다. 초기 카메라와 디스플레이 하드웨어가 BGR 순서로 데이터를 저장했고, OpenCV는 이 관례를 따랐습니다.

### BGR ↔ RGB 변환

```python
import cv2
import numpy as np

# 이미지 읽기 (BGR)
img_bgr = cv2.imread('image.jpg')

# BGR → RGB 변환
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# RGB → BGR 변환
img_bgr_back = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# NumPy로 직접 변환 (슬라이싱)
img_rgb_np = img_bgr[:, :, ::-1]  # 채널 순서 뒤집기
img_rgb_np = img_bgr[..., ::-1]   # 동일한 결과

# 채널별 스왑
b, g, r = cv2.split(img_bgr)
img_rgb_split = cv2.merge([r, g, b])
```

### matplotlib과 함께 사용하기

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')

# 잘못된 표시 (BGR 그대로 → 색상이 뒤바뀜)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img)  # BGR 그대로 → 빨강과 파랑이 뒤바뀜
plt.title('Wrong (BGR)')
plt.axis('off')

# 올바른 표시 (RGB로 변환)
plt.subplot(1, 3, 2)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title('Correct (RGB)')
plt.axis('off')

# 그레이스케일
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

# cv2.cvtColor(src, code) - 색상 공간 변환
dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

### 주요 변환 코드

```
┌─────────────────────────────────────────────────────────────────┐
│                     주요 색상 변환 코드                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   BGR ↔ 기타 색상 공간                                          │
│   ├── COLOR_BGR2RGB / COLOR_RGB2BGR                             │
│   ├── COLOR_BGR2GRAY / COLOR_GRAY2BGR                           │
│   ├── COLOR_BGR2HSV / COLOR_HSV2BGR                             │
│   ├── COLOR_BGR2LAB / COLOR_LAB2BGR                             │
│   ├── COLOR_BGR2YCrCb / COLOR_YCrCb2BGR                         │
│   └── COLOR_BGR2HLS / COLOR_HLS2BGR                             │
│                                                                 │
│   RGB ↔ 기타 색상 공간                                          │
│   ├── COLOR_RGB2GRAY / COLOR_GRAY2RGB                           │
│   ├── COLOR_RGB2HSV / COLOR_HSV2RGB                             │
│   ├── COLOR_RGB2LAB / COLOR_LAB2RGB                             │
│   └── COLOR_RGB2HLS / COLOR_HLS2RGB                             │
│                                                                 │
│   특수 변환                                                      │
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

# 다양한 색상 공간으로 변환
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

### HSV란?

HSV는 색상(Hue), 채도(Saturation), 명도(Value)로 색을 표현합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      HSV 색상 공간                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   H (Hue) - 색상                                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  0°    60°   120°   180°   240°   300°   360°          │   │
│   │  빨강   노랑   초록   청록   파랑   보라   빨강          │   │
│   │  ├──────┼──────┼──────┼──────┼──────┼──────┤            │   │
│   │  0     30     60     90    120    150    179            │   │
│   │      (OpenCV에서 H 범위: 0-179)                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   S (Saturation) - 채도 (0-255)                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  0 (무채색/회색)  ──────────────▶  255 (순수한 색)       │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   V (Value) - 명도 (0-255)                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  0 (검은색)  ──────────────────▶  255 (밝은 색)          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│                        V (밝기)                                  │
│                          ▲                                       │
│                          │    흰색                               │
│                          │   /                                   │
│                          │  /                                    │
│                          │ /     순수한 색                       │
│                          │/───────●                              │
│                          │        ╲                              │
│                          │         ╲  S (채도)                   │
│                          │          ╲                            │
│                          ●───────────╲───▶ H (색상, 원형)        │
│                        검은색                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### HSV 변환 및 채널 확인

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')

# BGR → HSV 변환
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 채널 분리
h, s, v = cv2.split(hsv)

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original')

axes[0, 1].imshow(h, cmap='hsv')  # Hue는 hsv 컬러맵 사용
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

# RGB/BGR에서는 조명 변화에 민감
# HSV에서는 V 채널만 영향받음 → 색상 검출에 유리

# 예: 빨간색 검출
img = cv2.imread('red_objects.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 빨간색 범위 정의 (Hue가 0 또는 180 근처)
# 빨간색은 Hue 범위의 양 끝에 있음
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

# 마스크 생성
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 | mask2  # 두 마스크 합치기

# 결과 표시
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
│                    주요 색상 HSV 범위 (OpenCV)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   색상       H (Hue)        S (Saturation)   V (Value)          │
│   ──────────────────────────────────────────────────────────    │
│   빨강       0-10, 160-179   100-255         100-255            │
│   주황       10-25           100-255         100-255            │
│   노랑       25-35           100-255         100-255            │
│   초록       35-85           100-255         100-255            │
│   청록       85-95           100-255         100-255            │
│   파랑       95-130          100-255         100-255            │
│   보라       130-160         100-255         100-255            │
│                                                                 │
│   흰색       0-179           0-30            200-255            │
│   검정       0-179           0-255           0-50               │
│   회색       0-179           0-30            50-200             │
│                                                                 │
│   주의: 조명 조건에 따라 범위 조정 필요                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. LAB 색상 공간

### LAB이란?

LAB(또는 CIELAB)은 인간의 색상 인지에 기반한 색상 공간입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      LAB 색상 공간                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   L (Lightness) - 밝기                                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  0 (검정)  ──────────────────────▶  255 (흰색)          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   A - 초록(-) ↔ 빨강(+)                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  0 (초록)  ────── 128 (중립) ──────  255 (빨강)          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   B - 파랑(-) ↔ 노랑(+)                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  0 (파랑)  ────── 128 (중립) ──────  255 (노랑)          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│                     +B (노랑)                                    │
│                        ▲                                        │
│                        │                                        │
│            -A ◀────────┼────────▶ +A                            │
│          (초록)        │        (빨강)                          │
│                        │                                        │
│                        ▼                                        │
│                     -B (파랑)                                    │
│                                                                 │
│   장점:                                                         │
│   - 인간 시각과 유사한 색상 거리 계산                             │
│   - 밝기와 색상이 분리됨                                         │
│   - 색상 보정, 색상 전이에 유용                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### LAB 변환 및 활용

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')

# BGR → LAB 변환
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# 채널 분리
l, a, b = cv2.split(lab)

# L 채널 조정으로 밝기 보정
l_adjusted = cv2.add(l, 30)  # 밝기 증가
l_adjusted = np.clip(l_adjusted, 0, 255).astype(np.uint8)

# 다시 합치기
lab_adjusted = cv2.merge([l_adjusted, a, b])
result = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)

# 시각화
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

# LAB 변환
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# CLAHE 적용 (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)

# 다시 합치기
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
│                   그레이스케일 변환 원리                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   BGR → Grayscale 변환 공식:                                    │
│                                                                 │
│   Gray = 0.114 × B + 0.587 × G + 0.299 × R                     │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   왜 단순 평균이 아닐까?                                  │   │
│   │                                                         │   │
│   │   인간의 눈은 녹색에 가장 민감하고, 파란색에 가장 둔감함    │   │
│   │   따라서 녹색(G)의 가중치가 가장 높음 (0.587)              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   컬러 이미지                     그레이스케일                   │
│   ┌───────────────┐              ┌───────────────┐             │
│   │ B │ G │ R │               │     Gray      │             │
│   │200│100│ 50│    ───▶       │      121      │             │
│   └───────────────┘              └───────────────┘             │
│   0.114×200 + 0.587×100 + 0.299×50 = 121.45                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 그레이스케일 변환 방법

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 방법 1: cvtColor (권장)
gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 방법 2: imread로 직접 읽기
gray2 = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 방법 3: NumPy로 직접 계산 (학습용)
b, g, r = cv2.split(img)
gray3 = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)

# 방법 4: 단순 평균 (비추천 - 시각적으로 부자연스러움)
gray4 = np.mean(img, axis=2).astype(np.uint8)

# 결과 비교
print(f"cvtColor 결과: {gray1.shape}")
print(f"직접 계산 결과: {gray3.shape}")
print(f"차이 최대값: {np.max(np.abs(gray1.astype(int) - gray3.astype(int)))}")
```

### 그레이스케일 → 컬러 (의사 컬러)

```python
import cv2

gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 그레이스케일 → 3채널 (여전히 흑백)
gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# 컬러맵 적용 (히트맵 등)
# COLORMAP_JET, COLORMAP_HOT, COLORMAP_RAINBOW 등
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

# 채널 분리
b, g, r = cv2.split(img)

# 또는 NumPy 인덱싱 사용 (더 빠름)
b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]

# 채널 병합
merged = cv2.merge([b, g, r])  # BGR 순서

# 채널 순서 변경하여 병합 (BGR → RGB)
rgb = cv2.merge([r, g, b])

# 빈 채널과 조합 (단일 채널만 표시)
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

# 원본
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original')

# 각 채널 (그레이스케일로)
axes[0, 1].imshow(r, cmap='gray')
axes[0, 1].set_title('Red Channel')

axes[0, 2].imshow(g, cmap='gray')
axes[0, 2].set_title('Green Channel')

axes[1, 0].imshow(b, cmap='gray')
axes[1, 0].set_title('Blue Channel')

# 각 채널 (컬러로)
zeros = np.zeros_like(b)
axes[1, 1].imshow(cv2.merge([zeros, zeros, r]))  # RGB 순서
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

# 1. 특정 채널 증폭
b, g, r = cv2.split(img)
r_boost = np.clip(r.astype(np.int16) + 50, 0, 255).astype(np.uint8)
warm = cv2.merge([b, g, r_boost])  # 따뜻한 색조

# 2. 채널 스왑
b, g, r = cv2.split(img)
swapped = cv2.merge([r, g, b])  # R과 B 교환

# 3. 채널 평균으로 그레이스케일
b, g, r = cv2.split(img)
gray_avg = ((b.astype(np.int16) + g + r) // 3).astype(np.uint8)

# 4. 특정 채널만 유지 (나머지 0으로)
b, g, r = cv2.split(img)
only_r = cv2.merge([np.zeros_like(b), np.zeros_like(g), r])
```

---

## 7. 색상 기반 객체 추적

### inRange()를 사용한 색상 필터링

```
┌─────────────────────────────────────────────────────────────────┐
│                   색상 기반 객체 추적 파이프라인                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   입력 이미지 (BGR)                                              │
│        │                                                        │
│        ▼                                                        │
│   HSV 변환                                                      │
│        │                                                        │
│        ▼                                                        │
│   cv2.inRange(hsv, lower, upper) ──▶ 이진 마스크                 │
│        │                                                        │
│        ▼                                                        │
│   노이즈 제거 (모폴로지 연산)                                     │
│        │                                                        │
│        ▼                                                        │
│   윤곽선 검출                                                    │
│        │                                                        │
│        ▼                                                        │
│   객체 위치/크기 추출                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 색상 추적 구현

```python
import cv2
import numpy as np

def track_color(img, lower_hsv, upper_hsv):
    """특정 색상 범위의 객체를 추적"""
    # HSV 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 마스크 생성
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 윤곽선 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # 결과 그리기
    result = img.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # 최소 면적 필터
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # 중심점
            cx, cy = x + w//2, y + h//2
            cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)

    return result, mask


# 사용 예: 파란색 추적
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

# 트랙바 생성
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

    # 트랙바 값 읽기
    h_low = cv2.getTrackbarPos('H_Low', 'Trackbars')
    h_high = cv2.getTrackbarPos('H_High', 'Trackbars')
    s_low = cv2.getTrackbarPos('S_Low', 'Trackbars')
    s_high = cv2.getTrackbarPos('S_High', 'Trackbars')
    v_low = cv2.getTrackbarPos('V_Low', 'Trackbars')
    v_high = cv2.getTrackbarPos('V_High', 'Trackbars')

    lower = np.array([h_low, s_low, v_low])
    upper = np.array([h_high, s_high, v_high])

    # HSV 변환 및 마스크
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

# 여러 색상 정의
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
        # 마스크 생성
        if 'lower1' in params:  # 빨간색처럼 범위가 두 개인 경우
            mask1 = cv2.inRange(hsv, params['lower1'], params['upper1'])
            mask2 = cv2.inRange(hsv, params['lower2'], params['upper2'])
            mask = mask1 | mask2
        else:
            mask = cv2.inRange(hsv, params['lower'], params['upper'])

        # 윤곽선 검출
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
# 힌트: cv2.setMouseCallback() 사용
def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 클릭한 위치의 HSV 값 출력
        pass
```

### 연습 3: 채널 스왑 효과

이미지의 채널을 다양하게 조합하여 6가지 효과(BGR, BRG, GBR, GRB, RBG, RGB)를 만들고 비교하세요.

### 연습 4: 피부색 검출

HSV와 YCrCb 색상 공간을 사용하여 이미지에서 피부색 영역을 검출하세요. 두 방법의 결과를 비교하세요.

```python
# 피부색 HSV 범위 예시
# H: 0-50, S: 20-150, V: 70-255

# 피부색 YCrCb 범위 예시
# Y: 0-255, Cr: 135-180, Cb: 85-135
```

### 연습 5: 색상 전이 애니메이션

H 채널을 점진적으로 증가시켜 이미지의 색상이 무지개처럼 변하는 애니메이션을 만드세요.

```python
# 힌트
for h_shift in range(0, 180, 5):
    h_channel = (original_h + h_shift) % 180
    # ...
```

---

## 9. 다음 단계

[04_Geometric_Transforms.md](./04_Geometric_Transforms.md)에서 이미지 크기 조절, 회전, 뒤집기, 어파인/원근 변환 등을 학습합니다!

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
| [02_Image_Basics.md](./02_Image_Basics.md) | 이미지 읽기, 픽셀 접근 |
| [07_Thresholding.md](./07_Thresholding.md) | HSV 기반 임계처리 |

### 색상 공간 참고

- [색상 공간 위키피디아](https://en.wikipedia.org/wiki/Color_space)
- [HSV 색상 모델](https://en.wikipedia.org/wiki/HSL_and_HSV)
- [CIELAB 색상 공간](https://en.wikipedia.org/wiki/CIELAB_color_space)

