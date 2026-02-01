# 이미지 필터링

## 개요

이미지 필터링(Image Filtering)은 이미지의 픽셀 값을 주변 픽셀을 고려하여 변환하는 작업입니다. 노이즈 제거, 블러, 샤프닝 등 다양한 효과를 낼 수 있습니다. 이 문서에서는 커널과 컨볼루션의 개념부터 OpenCV의 다양한 필터 함수까지 학습합니다.

**난이도**: ⭐⭐ (초급-중급)

**학습 목표**:
- 커널(Kernel)과 컨볼루션(Convolution) 개념 이해
- 다양한 블러 필터 (`blur`, `GaussianBlur`, `medianBlur`, `bilateralFilter`)
- 엣지 보존 스무딩
- 커스텀 필터와 샤프닝 구현

---

## 목차

1. [커널과 컨볼루션](#1-커널과-컨볼루션)
2. [평균 블러 - blur()](#2-평균-블러---blur)
3. [가우시안 블러 - GaussianBlur()](#3-가우시안-블러---gaussianblur)
4. [중앙값 블러 - medianBlur()](#4-중앙값-블러---medianblur)
5. [양방향 필터 - bilateralFilter()](#5-양방향-필터---bilateralfilter)
6. [커스텀 필터 - filter2D()](#6-커스텀-필터---filter2d)
7. [샤프닝 필터](#7-샤프닝-필터)
8. [연습 문제](#8-연습-문제)
9. [다음 단계](#9-다음-단계)
10. [참고 자료](#10-참고-자료)

---

## 1. 커널과 컨볼루션

### 커널(Kernel)이란?

```
┌─────────────────────────────────────────────────────────────────┐
│                        커널 (Kernel)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   커널(또는 필터, 마스크)은 이미지에 적용할 연산을 정의하는        │
│   작은 행렬입니다. 일반적으로 3x3, 5x5, 7x7 크기를 사용합니다.    │
│                                                                 │
│   예: 3x3 평균 필터 커널                                         │
│                                                                 │
│        1/9   1/9   1/9         ┌───┬───┬───┐                   │
│                                │1/9│1/9│1/9│                   │
│        1/9   1/9   1/9    =    ├───┼───┼───┤                   │
│                                │1/9│1/9│1/9│                   │
│        1/9   1/9   1/9         ├───┼───┼───┤                   │
│                                │1/9│1/9│1/9│                   │
│                                └───┴───┴───┘                   │
│                                                                 │
│   커널 크기의 의미:                                              │
│   - 크기가 클수록 더 넓은 영역을 고려                            │
│   - 큰 커널 = 강한 효과, 느린 처리                               │
│   - 작은 커널 = 약한 효과, 빠른 처리                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 컨볼루션(Convolution) 연산

```
┌─────────────────────────────────────────────────────────────────┐
│                      컨볼루션 연산                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   입력 이미지의 각 픽셀에 대해 커널을 적용하여 새 값을 계산         │
│                                                                 │
│   입력 이미지          3x3 커널            출력                  │
│   ┌───┬───┬───┬───┐   ┌───┬───┬───┐                           │
│   │ 1 │ 2 │ 3 │ 4 │   │1/9│1/9│1/9│                           │
│   ├───┼───┼───┼───┤   ├───┼───┼───┤      결과 픽셀:            │
│   │ 5 │ 6 │ 7 │ 8 │   │1/9│1/9│1/9│   (1+2+3+5+6+7+9+10+11)/9 │
│   ├───┼───┼───┼───┤   ├───┼───┼───┤      = 54/9 = 6            │
│   │ 9 │10 │11 │12 │   │1/9│1/9│1/9│                           │
│   ├───┼───┼───┼───┤   └───┴───┴───┘                           │
│   │13 │14 │15 │16 │                                            │
│   └───┴───┴───┴───┘                                            │
│                                                                 │
│   과정:                                                         │
│   1. 커널을 이미지 위에 놓음                                     │
│   2. 대응하는 픽셀끼리 곱함                                      │
│   3. 모든 결과를 더함                                           │
│   4. 다음 픽셀로 이동하여 반복                                   │
│                                                                 │
│   경계 처리:                                                     │
│   - BORDER_CONSTANT: 상수 값으로 채움 (기본값 0)                 │
│   - BORDER_REPLICATE: 경계 픽셀 복제                            │
│   - BORDER_REFLECT: 경계에서 반사                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 컨볼루션 시각화

```python
import cv2
import numpy as np

def visualize_convolution(img, kernel):
    """컨볼루션 과정을 시각화 (학습용)"""
    h, w = img.shape
    kh, kw = kernel.shape
    pad = kh // 2

    # 패딩 추가
    padded = np.pad(img, pad, mode='constant', constant_values=0)

    # 결과 배열
    result = np.zeros_like(img, dtype=np.float64)

    # 컨볼루션 (느린 버전 - 학습용)
    for y in range(h):
        for x in range(w):
            region = padded[y:y+kh, x:x+kw]
            result[y, x] = np.sum(region * kernel)

    return result


# 예제
img = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
], dtype=np.float64)

kernel = np.ones((3, 3)) / 9  # 평균 필터

result = visualize_convolution(img, kernel)
print("입력:\n", img)
print("\n결과:\n", result)
```

---

## 2. 평균 블러 - blur()

### 기본 사용법

평균 블러는 가장 단순한 블러 필터로, 커널 영역의 평균값을 사용합니다.

```python
import cv2

img = cv2.imread('image.jpg')

# blur(src, ksize)
# ksize: (width, height) 형태의 커널 크기

blur_3x3 = cv2.blur(img, (3, 3))
blur_5x5 = cv2.blur(img, (5, 5))
blur_7x7 = cv2.blur(img, (7, 7))
blur_15x15 = cv2.blur(img, (15, 15))

cv2.imshow('Original', img)
cv2.imshow('3x3 Blur', blur_3x3)
cv2.imshow('5x5 Blur', blur_5x5)
cv2.imshow('15x15 Blur', blur_15x15)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 평균 블러 커널

```
┌─────────────────────────────────────────────────────────────────┐
│                      평균 블러 커널                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   3x3 평균 커널:                                                │
│   ┌─────┬─────┬─────┐                                          │
│   │ 1/9 │ 1/9 │ 1/9 │                                          │
│   ├─────┼─────┼─────┤                                          │
│   │ 1/9 │ 1/9 │ 1/9 │  =  1/9 × [[1, 1, 1],                   │
│   ├─────┼─────┼─────┤           [1, 1, 1],                    │
│   │ 1/9 │ 1/9 │ 1/9 │           [1, 1, 1]]                    │
│   └─────┴─────┴─────┘                                          │
│                                                                 │
│   5x5 평균 커널:                                                │
│   모든 값이 1/25                                                │
│                                                                 │
│   특징:                                                         │
│   - 단순하고 빠름                                               │
│   - 엣지도 함께 흐려짐                                          │
│   - 균일한 노이즈 제거에 효과적                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### boxFilter()

`blur()`의 일반화된 버전입니다.

```python
import cv2

img = cv2.imread('image.jpg')

# normalize=True (기본): 커널 정규화 (평균 필터)
# normalize=False: 합계 필터
blur_normalized = cv2.boxFilter(img, -1, (5, 5), normalize=True)
sum_filter = cv2.boxFilter(img, -1, (5, 5), normalize=False)

# blur(img, (5, 5))와 동일
print(f"차이: {np.sum(np.abs(cv2.blur(img, (5, 5)) - blur_normalized))}")  # 0
```

---

## 3. 가우시안 블러 - GaussianBlur()

### 가우시안 필터란?

가우시안 필터는 중심에 더 큰 가중치를 주는 블러 필터입니다. 자연스러운 블러 효과를 만들어냅니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      가우시안 커널                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   가우시안 분포 (정규 분포, 종 모양):                            │
│                                                                 │
│          ▲                                                      │
│          │     ████                                             │
│          │   ████████                                           │
│          │  ██████████                                          │
│          │ ████████████                                         │
│          │██████████████                                        │
│          └──────────────────▶                                   │
│                   중심에서 멀어질수록 가중치 감소                 │
│                                                                 │
│   3x3 가우시안 커널 (근사값):                                    │
│   ┌─────┬─────┬─────┐                                          │
│   │ 1   │ 2   │ 1   │                                          │
│   ├─────┼─────┼─────┤  ×  1/16                                 │
│   │ 2   │ 4   │ 2   │                                          │
│   ├─────┼─────┼─────┤                                          │
│   │ 1   │ 2   │ 1   │                                          │
│   └─────┴─────┴─────┘                                          │
│                                                                 │
│   특징:                                                         │
│   - 평균 블러보다 자연스러운 결과                                │
│   - 엣지 검출 전처리에 자주 사용                                 │
│   - 시그마(σ) 값으로 블러 강도 조절                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 사용법

```python
import cv2

img = cv2.imread('image.jpg')

# GaussianBlur(src, ksize, sigmaX, sigmaY=0)
# ksize: 커널 크기 (홀수여야 함)
# sigmaX: X방향 표준편차 (0이면 커널 크기에서 자동 계산)
# sigmaY: Y방향 표준편차 (0이면 sigmaX와 같은 값)

# 커널 크기로 지정 (sigma 자동 계산)
blur1 = cv2.GaussianBlur(img, (5, 5), 0)

# sigma 지정 (커널 크기는 적절히 자동 조정)
blur2 = cv2.GaussianBlur(img, (0, 0), 3)  # sigma=3

# 커널 크기와 sigma 모두 지정
blur3 = cv2.GaussianBlur(img, (7, 7), 1.5)
```

### sigma와 커널 크기의 관계

```python
import cv2
import numpy as np

# 가우시안 커널 직접 생성하여 확인
def show_gaussian_kernel(ksize, sigma):
    kernel = cv2.getGaussianKernel(ksize, sigma)
    kernel_2d = kernel @ kernel.T  # 1D를 2D로
    print(f"Kernel ({ksize}x{ksize}, sigma={sigma}):")
    print(np.round(kernel_2d, 4))
    print(f"합계: {np.sum(kernel_2d):.4f}\n")


show_gaussian_kernel(3, 0)   # sigma 자동 계산
show_gaussian_kernel(5, 0)
show_gaussian_kernel(5, 1.0)
show_gaussian_kernel(5, 2.0)

# 권장: sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
```

### 평균 블러 vs 가우시안 블러

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 동일한 커널 크기로 비교
ksize = 15
avg_blur = cv2.blur(img, (ksize, ksize))
gauss_blur = cv2.GaussianBlur(img, (ksize, ksize), 0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_rgb)
axes[0].set_title('Original')

axes[1].imshow(cv2.cvtColor(avg_blur, cv2.COLOR_BGR2RGB))
axes[1].set_title('Average Blur')

axes[2].imshow(cv2.cvtColor(gauss_blur, cv2.COLOR_BGR2RGB))
axes[2].set_title('Gaussian Blur')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## 4. 중앙값 블러 - medianBlur()

### 중앙값 필터란?

중앙값 필터는 커널 영역의 중앙값(median)을 사용합니다. Salt-and-pepper 노이즈 제거에 매우 효과적입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                     중앙값 필터 동작                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   입력 영역:                                                     │
│   ┌────┬────┬────┐                                             │
│   │ 10 │ 20 │ 30 │                                             │
│   ├────┼────┼────┤                                             │
│   │ 40 │255 │ 60 │   ← 중앙의 255는 노이즈 (salt)               │
│   ├────┼────┼────┤                                             │
│   │ 70 │ 80 │ 90 │                                             │
│   └────┴────┴────┘                                             │
│                                                                 │
│   값 정렬: 10, 20, 30, 40, 60, 70, 80, 90, 255                  │
│   중앙값: 60 (5번째 값)                                          │
│                                                                 │
│   결과:                                                         │
│   ┌────┬────┬────┐                                             │
│   │    │    │    │                                             │
│   ├────┼────┼────┤                                             │
│   │    │ 60 │    │   ← 노이즈가 제거됨                          │
│   ├────┼────┼────┤                                             │
│   │    │    │    │                                             │
│   └────┴────┴────┘                                             │
│                                                                 │
│   특징:                                                         │
│   - Salt-and-pepper 노이즈에 매우 효과적                        │
│   - 엣지를 비교적 잘 보존                                       │
│   - 평균/가우시안보다 느림                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 사용법

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# Salt-and-pepper 노이즈 추가 (테스트용)
def add_salt_pepper_noise(img, amount=0.05):
    noisy = img.copy()
    h, w = img.shape[:2]
    num_pixels = int(amount * h * w)

    # Salt (흰색)
    for _ in range(num_pixels):
        y, x = np.random.randint(0, h), np.random.randint(0, w)
        noisy[y, x] = 255

    # Pepper (검은색)
    for _ in range(num_pixels):
        y, x = np.random.randint(0, h), np.random.randint(0, w)
        noisy[y, x] = 0

    return noisy


noisy_img = add_salt_pepper_noise(img, 0.02)

# medianBlur(src, ksize)
# ksize: 홀수만 가능 (3, 5, 7, ...)
median_3 = cv2.medianBlur(noisy_img, 3)
median_5 = cv2.medianBlur(noisy_img, 5)

# 비교: 평균 블러, 가우시안 블러
avg_blur = cv2.blur(noisy_img, (5, 5))
gauss_blur = cv2.GaussianBlur(noisy_img, (5, 5), 0)

cv2.imshow('Noisy', noisy_img)
cv2.imshow('Average Blur', avg_blur)
cv2.imshow('Gaussian Blur', gauss_blur)
cv2.imshow('Median Blur', median_5)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 5. 양방향 필터 - bilateralFilter()

### 양방향 필터란?

양방향 필터(Bilateral Filter)는 엣지를 보존하면서 스무딩하는 필터입니다. 피부 보정, 그림 효과 등에 사용됩니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                     양방향 필터 원리                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   일반 가우시안 필터:                                            │
│   - 거리만 고려 → 엣지도 흐려짐                                  │
│                                                                 │
│   양방향 필터:                                                   │
│   - 거리(공간) + 색상 차이 모두 고려                              │
│   - 색상이 유사한 픽셀만 평균에 포함                              │
│   - 엣지(색상 차이가 큰 곳)는 보존                                │
│                                                                 │
│   예시:                                                         │
│   ┌─────────────────────────────────────────┐                   │
│   │ 100  100  100 │ 200  200  200 │          │                   │
│   │ 100  100  100 │ 200  200  200 │  ← 엣지  │                   │
│   │ 100  100  100 │ 200  200  200 │          │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                 │
│   가우시안: 100과 200이 섞여서 150 부근으로                       │
│   양방향: 100 영역은 100으로, 200 영역은 200으로 유지             │
│                                                                 │
│   가중치 = 공간 가우시안 × 색상 가우시안                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 사용법

```python
import cv2

img = cv2.imread('portrait.jpg')

# bilateralFilter(src, d, sigmaColor, sigmaSpace)
# d: 필터 크기 (-1이면 sigmaSpace에서 자동 계산)
# sigmaColor: 색상 공간에서의 시그마 (높을수록 더 넓은 색상 범위 평균)
# sigmaSpace: 좌표 공간에서의 시그마 (높을수록 더 넓은 영역 고려)

# 약한 효과
bilateral_weak = cv2.bilateralFilter(img, 9, 50, 50)

# 중간 효과
bilateral_medium = cv2.bilateralFilter(img, 9, 75, 75)

# 강한 효과 (그림 같은 효과)
bilateral_strong = cv2.bilateralFilter(img, 15, 100, 100)

# 매우 강한 효과
bilateral_extreme = cv2.bilateralFilter(img, 15, 150, 150)
```

### 피부 스무딩 예제

```python
import cv2
import numpy as np

def skin_smoothing(img, strength='medium'):
    """피부 스무딩 효과"""
    params = {
        'weak': (5, 30, 30),
        'medium': (9, 75, 75),
        'strong': (15, 100, 100),
        'extreme': (20, 150, 150)
    }

    d, sigmaColor, sigmaSpace = params.get(strength, params['medium'])

    # 양방향 필터 적용
    smooth = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

    # 원본과 블렌딩 (자연스러운 효과)
    alpha = 0.7  # 블렌딩 비율
    result = cv2.addWeighted(smooth, alpha, img, 1 - alpha, 0)

    return result


img = cv2.imread('portrait.jpg')
result = skin_smoothing(img, 'medium')

cv2.imshow('Original', img)
cv2.imshow('Smoothed', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 블러 필터 비교

```python
import cv2
import time
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')

# 처리 시간 비교
filters = []

start = time.time()
avg = cv2.blur(img, (9, 9))
filters.append(('Average', avg, time.time() - start))

start = time.time()
gauss = cv2.GaussianBlur(img, (9, 9), 0)
filters.append(('Gaussian', gauss, time.time() - start))

start = time.time()
median = cv2.medianBlur(img, 9)
filters.append(('Median', median, time.time() - start))

start = time.time()
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
filters.append(('Bilateral', bilateral, time.time() - start))

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for ax, (name, result, elapsed) in zip(axes, filters):
    ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax.set_title(f'{name} ({elapsed*1000:.1f}ms)')
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## 6. 커스텀 필터 - filter2D()

### filter2D() 사용법

`filter2D()`를 사용하면 직접 정의한 커널로 컨볼루션을 수행할 수 있습니다.

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# filter2D(src, ddepth, kernel)
# ddepth: 출력 이미지의 깊이 (-1 = 입력과 동일)
# kernel: 사용자 정의 커널

# 평균 필터를 직접 만들어 적용
kernel_avg = np.ones((5, 5), np.float32) / 25
avg_custom = cv2.filter2D(img, -1, kernel_avg)

# blur()와 동일한 결과
avg_builtin = cv2.blur(img, (5, 5))
print(f"차이: {np.sum(np.abs(avg_custom - avg_builtin))}")  # 0
```

### 다양한 커스텀 커널

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 1. 엠보스 (Emboss) 효과
kernel_emboss = np.array([
    [-2, -1, 0],
    [-1,  1, 1],
    [ 0,  1, 2]
])
emboss = cv2.filter2D(img, -1, kernel_emboss) + 128

# 2. 윤곽 검출 (라플라시안)
kernel_laplacian = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
])
laplacian = cv2.filter2D(img, -1, kernel_laplacian)

# 3. Sobel X (수직 엣지)
kernel_sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
sobel_x = cv2.filter2D(img, -1, kernel_sobel_x)

# 4. Sobel Y (수평 엣지)
kernel_sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])
sobel_y = cv2.filter2D(img, -1, kernel_sobel_y)
```

### 커널 시각화 도구

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_and_show_kernel(img, kernel, title):
    """커널 적용 결과와 커널 시각화"""
    result = cv2.filter2D(img, -1, kernel)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 원본
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')

    # 커널 시각화
    im = axes[1].imshow(kernel, cmap='RdBu_r', vmin=-2, vmax=2)
    axes[1].set_title(f'Kernel ({kernel.shape[0]}x{kernel.shape[1]})')
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            axes[1].text(j, i, f'{kernel[i,j]:.1f}',
                        ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=axes[1])

    # 결과
    axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[2].set_title(title)
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


img = cv2.imread('image.jpg')

# 예제: 엠보스 커널
kernel_emboss = np.array([
    [-2, -1, 0],
    [-1,  1, 1],
    [ 0,  1, 2]
], dtype=np.float32)

apply_and_show_kernel(img, kernel_emboss, 'Emboss')
```

---

## 7. 샤프닝 필터

### 샤프닝 원리

```
┌─────────────────────────────────────────────────────────────────┐
│                      샤프닝 원리                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   샤프닝 = 원본 + (원본 - 블러)                                  │
│         = 원본 + 고주파 성분                                     │
│         = 엣지 강조                                             │
│                                                                 │
│   또는 커널로 직접:                                              │
│                                                                 │
│   기본 샤프닝 커널:                                              │
│   ┌────┬────┬────┐                                             │
│   │  0 │ -1 │  0 │                                             │
│   ├────┼────┼────┤                                             │
│   │ -1 │  5 │ -1 │   중앙 = 5 (원본 가중치)                     │
│   ├────┼────┼────┤   주변 = -1 (블러 빼기)                     │
│   │  0 │ -1 │  0 │   합 = 1 (밝기 유지)                         │
│   └────┴────┴────┘                                             │
│                                                                 │
│   강한 샤프닝 커널:                                              │
│   ┌────┬────┬────┐                                             │
│   │ -1 │ -1 │ -1 │                                             │
│   ├────┼────┼────┤                                             │
│   │ -1 │  9 │ -1 │   중앙 = 9                                  │
│   ├────┼────┼────┤   주변 = -1 × 8 = -8                        │
│   │ -1 │ -1 │ -1 │   합 = 1                                    │
│   └────┴────┴────┘                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 샤프닝 구현

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 방법 1: 커널 사용
kernel_sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
sharpened1 = cv2.filter2D(img, -1, kernel_sharpen)

# 방법 2: 강한 샤프닝 커널
kernel_sharpen_strong = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
])
sharpened2 = cv2.filter2D(img, -1, kernel_sharpen_strong)

# 방법 3: Unsharp Masking (언샤프 마스크)
def unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """
    Unsharp masking으로 샤프닝

    amount: 샤프닝 강도 (1.0 = 표준)
    threshold: 엣지 검출 임계값 (노이즈 방지)
    """
    # 블러 이미지
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)

    # 원본 - 블러 = 엣지/디테일
    # sharpened = 원본 + amount × (원본 - 블러)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)

    if threshold > 0:
        # 변화량이 threshold 이하인 픽셀은 원본 유지
        diff = cv2.absdiff(img, blurred)
        mask = (diff < threshold).astype(np.uint8) * 255
        sharpened = np.where(mask == 255, img, sharpened)

    return sharpened


sharpened3 = unsharp_mask(img, amount=1.5)
```

### 적응형 샤프닝

```python
import cv2
import numpy as np

def adaptive_sharpening(img, amount=1.0):
    """
    적응형 샤프닝 - 엣지 영역에만 샤프닝 적용
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 엣지 검출
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # 블러
    blurred = cv2.GaussianBlur(img, (5, 5), 1)

    # 샤프닝
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)

    # 엣지 영역에만 샤프닝 적용
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) / 255.0
    result = (sharpened * edges_3ch + img * (1 - edges_3ch)).astype(np.uint8)

    return result


img = cv2.imread('image.jpg')
result = adaptive_sharpening(img, amount=2.0)
```

---

## 8. 연습 문제

### 연습 1: 노이즈 제거 비교

가우시안 노이즈와 Salt-and-pepper 노이즈를 각각 생성하고, 세 가지 블러 필터(평균, 가우시안, 중앙값)로 제거 효과를 비교하세요. PSNR 값으로 정량적 비교도 수행하세요.

```python
# 힌트: 가우시안 노이즈 추가
def add_gaussian_noise(img, mean=0, var=100):
    noise = np.random.normal(mean, var**0.5, img.shape)
    noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy
```

### 연습 2: 실시간 블러 강도 조절

웹캠 영상에 트랙바로 블러 강도(커널 크기)를 조절할 수 있는 프로그램을 작성하세요. 가우시안 블러와 양방향 필터 중 선택할 수 있게 하세요.

### 연습 3: 커스텀 엠보스 방향

8방향(상, 하, 좌, 우, 대각선 4방향)으로 다른 엠보스 효과를 내는 커널들을 설계하고 테스트하세요.

### 연습 4: 고급 샤프닝

다음 기능을 가진 고급 샤프닝 함수를 구현하세요:
1. 샤프닝 강도 조절 (amount)
2. 블러 반경 조절 (radius)
3. 임계값 적용 (threshold) - 작은 변화는 무시
4. 하이라이트/섀도우 별도 처리

### 연습 5: 미니어처 효과 (틸트 시프트)

가우시안 블러와 마스크를 사용하여 틸트 시프트(tilt-shift) 미니어처 효과를 구현하세요. 이미지 중앙 부분은 선명하게, 위아래는 점진적으로 블러 처리합니다.

```python
# 힌트
def tilt_shift(img, focus_y, focus_height, blur_amount):
    # 그라디언트 마스크 생성
    # 블러 이미지와 원본을 마스크로 블렌딩
    pass
```

---

## 9. 다음 단계

[06_Morphology.md](./06_Morphology.md)에서 침식, 팽창, 열기/닫기 등 형태학적 연산을 학습합니다!

**다음에 배울 내용**:
- 구조 요소 (Structuring Element)
- 침식 (Erosion)과 팽창 (Dilation)
- 열기 (Opening)와 닫기 (Closing)
- 노이즈 제거와 객체 분리

---

## 10. 참고 자료

### 공식 문서

- [blur() 문서](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga8c45db9afe636703801b0b2e440fce37)
- [GaussianBlur() 문서](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1)
- [medianBlur() 문서](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9)
- [bilateralFilter() 문서](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed)

### 관련 학습 자료

| 폴더 | 관련 내용 |
|------|----------|
| [04_Geometric_Transforms.md](./04_Geometric_Transforms.md) | 이미지 전처리 |
| [08_Edge_Detection.md](./08_Edge_Detection.md) | 필터링 후 엣지 검출 |

### 추가 참고

- [이미지 필터링 이론](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)
- [컨볼루션 시각화](https://setosa.io/ev/image-kernels/)

