# 이미지 필터링

## 개요

이미지 필터링(Image Filtering)은 이미지의 픽셀 값을 주변 픽셀을 고려하여 변환하는 작업입니다. 노이즈 제거, 블러, 샤프닝 등 다양한 효과를 낼 수 있습니다. 이 문서에서는 커널과 컨볼루션의 개념부터 OpenCV의 다양한 필터 함수까지 학습합니다.

**난이도**: ⭐⭐ (초급-중급)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 커널(Kernel)과 컨볼루션(Convolution) 개념 이해
2. 다양한 블러 필터 (`blur`, `GaussianBlur`, `medianBlur`, `bilateralFilter`)
3. 엣지 보존 스무딩
4. 커스텀 필터와 샤프닝 구현

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
│                        Kernel                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   A kernel (or filter, mask) is a small matrix that defines    │
│   the operation to apply to an image. Typically 3x3, 5x5, 7x7. │
│                                                                 │
│   Example: 3x3 average filter kernel                            │
│                                                                 │
│        1/9   1/9   1/9         ┌───┬───┬───┐                   │
│                                │1/9│1/9│1/9│                   │
│        1/9   1/9   1/9    =    ├───┼───┼───┤                   │
│                                │1/9│1/9│1/9│                   │
│        1/9   1/9   1/9         ├───┼───┼───┤                   │
│                                │1/9│1/9│1/9│                   │
│                                └───┴───┴───┘                   │
│                                                                 │
│   Kernel size meaning:                                          │
│   - Larger size considers wider area                            │
│   - Large kernel = strong effect, slow processing              │
│   - Small kernel = weak effect, fast processing                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 컨볼루션(Convolution) 연산

```
┌─────────────────────────────────────────────────────────────────┐
│                      Convolution Operation                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Apply kernel to each pixel of input image to calculate new value│
│                                                                 │
│   Input image          3x3 kernel           Output              │
│   ┌───┬───┬───┬───┐   ┌───┬───┬───┐                           │
│   │ 1 │ 2 │ 3 │ 4 │   │1/9│1/9│1/9│                           │
│   ├───┼───┼───┼───┤   ├───┼───┼───┤      Result pixel:         │
│   │ 5 │ 6 │ 7 │ 8 │   │1/9│1/9│1/9│   (1+2+3+5+6+7+9+10+11)/9 │
│   ├───┼───┼───┼───┤   ├───┼───┼───┤      = 54/9 = 6            │
│   │ 9 │10 │11 │12 │   │1/9│1/9│1/9│                           │
│   ├───┼───┼───┼───┤   └───┴───┴───┘                           │
│   │13 │14 │15 │16 │                                            │
│   └───┴───┴───┴───┘                                            │
│                                                                 │
│   Process:                                                      │
│   1. Place kernel over image                                    │
│   2. Multiply corresponding pixels                              │
│   3. Sum all results                                            │
│   4. Move to next pixel and repeat                              │
│                                                                 │
│   Border handling:                                              │
│   - BORDER_CONSTANT: Fill with constant value (default 0)       │
│   - BORDER_REPLICATE: Replicate border pixels                   │
│   - BORDER_REFLECT: Reflect at border                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 컨볼루션 시각화

```python
import cv2
import numpy as np

def visualize_convolution(img, kernel):
    """Visualize convolution process (for learning)"""
    h, w = img.shape
    kh, kw = kernel.shape
    pad = kh // 2  # Why pad = kh//2: ensures the output has the same size as the input

    # Why zero-padding: border pixels need neighbors; padding with 0 is neutral for average kernels
    padded = np.pad(img, pad, mode='constant', constant_values=0)

    # Why float64: intermediate sums can exceed uint8 range (0-255); promotes before clipping
    result = np.zeros_like(img, dtype=np.float64)

    # Slow explicit loop — used here to make each step visible; use cv2.filter2D in production
    for y in range(h):
        for x in range(w):
            region = padded[y:y+kh, x:x+kw]
            result[y, x] = np.sum(region * kernel)

    return result


# Example
img = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
], dtype=np.float64)

kernel = np.ones((3, 3)) / 9  # Average filter: weights sum to 1 to preserve overall brightness

result = visualize_convolution(img, kernel)
print("Input:\n", img)
print("\nResult:\n", result)
```

**블러링이 작동하는 이유는?** 이미지의 노이즈(noise)는 실제 장면 내용을 반영하지 않으면서 인접 픽셀이 급격히 다른, 빠른 고주파 픽셀 변동으로 나타납니다. 블러 커널이 이웃 픽셀을 평균하면 이러한 급격한 무작위 변동이 상쇄되는 반면, 실제 이미지 특징(여러 픽셀에 걸쳐 점진적으로 변하는)은 보존됩니다. 신호 처리 관점에서 블러 커널은 고주파 성분을 억제하는 **저역 통과 필터(low-pass filter)**입니다. 이것이 가우시안 블러가 엣지 검출 전처리 단계로 선호되는 이유이기도 합니다: 노이즈로 인한 거짓 엣지를 제거하면서 실제 구조적 경계는 보존합니다.

---

## 2. 평균 블러 - blur()

### 기본 사용법

평균 블러는 가장 단순한 블러 필터로, 커널 영역의 평균값을 사용합니다.

```python
import cv2

img = cv2.imread('image.jpg')

# blur(src, ksize)
# ksize: kernel size in (width, height) format

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
│                      Average Blur Kernel                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   3x3 average kernel:                                           │
│   ┌─────┬─────┬─────┐                                          │
│   │ 1/9 │ 1/9 │ 1/9 │                                          │
│   ├─────┼─────┼─────┤                                          │
│   │ 1/9 │ 1/9 │ 1/9 │  =  1/9 × [[1, 1, 1],                   │
│   ├─────┼─────┼─────┤           [1, 1, 1],                    │
│   │ 1/9 │ 1/9 │ 1/9 │           [1, 1, 1]]                    │
│   └─────┴─────┴─────┘                                          │
│                                                                 │
│   5x5 average kernel:                                           │
│   All values are 1/25                                           │
│                                                                 │
│   Features:                                                     │
│   - Simple and fast                                             │
│   - Edges also get blurred                                      │
│   - Effective for uniform noise removal                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### boxFilter()

`blur()`의 일반화된 버전입니다.

```python
import cv2

img = cv2.imread('image.jpg')

# normalize=True (default): Normalize kernel (average filter)
# normalize=False: Sum filter
blur_normalized = cv2.boxFilter(img, -1, (5, 5), normalize=True)
sum_filter = cv2.boxFilter(img, -1, (5, 5), normalize=False)

# Same as blur(img, (5, 5))
print(f"Difference: {np.sum(np.abs(cv2.blur(img, (5, 5)) - blur_normalized))}")  # 0
```

---

## 3. 가우시안 블러 - GaussianBlur()

### 가우시안 필터란?

가우시안 필터는 중심에 더 큰 가중치를 주는 블러 필터입니다. 자연스러운 블러 효과를 만들어냅니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Gaussian Kernel                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Gaussian distribution (normal distribution, bell shape):      │
│                                                                 │
│          ▲                                                      │
│          │     ████                                             │
│          │   ████████                                           │
│          │  ██████████                                          │
│          │ ████████████                                         │
│          │██████████████                                        │
│          └──────────────────▶                                   │
│                   Weight decreases away from center             │
│                                                                 │
│   3x3 Gaussian kernel (approximate):                            │
│   ┌─────┬─────┬─────┐                                          │
│   │ 1   │ 2   │ 1   │                                          │
│   ├─────┼─────┼─────┤  ×  1/16                                 │
│   │ 2   │ 4   │ 2   │                                          │
│   ├─────┼─────┼─────┤                                          │
│   │ 1   │ 2   │ 1   │                                          │
│   └─────┴─────┴─────┘                                          │
│                                                                 │
│   Features:                                                     │
│   - More natural result than average blur                       │
│   - Often used for edge detection preprocessing                │
│   - Control blur strength with sigma (σ) value                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 사용법

```python
import cv2

img = cv2.imread('image.jpg')

# GaussianBlur(src, ksize, sigmaX, sigmaY=0)
# ksize: Kernel size (must be odd)
# sigmaX: Standard deviation in X direction (0 = auto-calculate from kernel size)
# sigmaY: Standard deviation in Y direction (0 = same as sigmaX)

# Why ksize=(5,5) and sigmaX=0: letting OpenCV derive sigma from kernel size is the
# recommended default — it ties blur strength to a single intuitive parameter (kernel size)
# rather than requiring you to keep ksize and sigma in sync manually
blur1 = cv2.GaussianBlur(img, (5, 5), 0)

# Why (0,0) with explicit sigma: when you reason in terms of sigma (e.g., "blur ~3 pixels"),
# letting OpenCV pick the minimal sufficient kernel size avoids unnecessary computation
blur2 = cv2.GaussianBlur(img, (0, 0), 3)  # sigma=3

# Specify both kernel size and sigma
blur3 = cv2.GaussianBlur(img, (7, 7), 1.5)
```

### sigma와 커널 크기의 관계

```python
import cv2
import numpy as np

# Generate Gaussian kernel directly to check
def show_gaussian_kernel(ksize, sigma):
    kernel = cv2.getGaussianKernel(ksize, sigma)
    kernel_2d = kernel @ kernel.T  # 1D to 2D
    print(f"Kernel ({ksize}x{ksize}, sigma={sigma}):")
    print(np.round(kernel_2d, 4))
    print(f"Sum: {np.sum(kernel_2d):.4f}\n")


show_gaussian_kernel(3, 0)   # sigma auto-calculated
show_gaussian_kernel(5, 0)
show_gaussian_kernel(5, 1.0)
show_gaussian_kernel(5, 2.0)

# Recommended: sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
```

### 평균 블러 vs 가우시안 블러

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Compare with same kernel size
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
│                     Median Filter Operation                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input region:                                                 │
│   ┌────┬────┬────┐                                             │
│   │ 10 │ 20 │ 30 │                                             │
│   ├────┼────┼────┤                                             │
│   │ 40 │255 │ 60 │   ← Center 255 is noise (salt)              │
│   ├────┼────┼────┤                                             │
│   │ 70 │ 80 │ 90 │                                             │
│   └────┴────┴────┘                                             │
│                                                                 │
│   Sort values: 10, 20, 30, 40, 60, 70, 80, 90, 255             │
│   Median: 60 (5th value)                                        │
│                                                                 │
│   Result:                                                       │
│   ┌────┬────┬────┐                                             │
│   │    │    │    │                                             │
│   ├────┼────┼────┤                                             │
│   │    │ 60 │    │   ← Noise removed                           │
│   ├────┼────┼────┤                                             │
│   │    │    │    │                                             │
│   └────┴────┴────┘                                             │
│                                                                 │
│   Features:                                                     │
│   - Very effective for salt-and-pepper noise                   │
│   - Preserves edges relatively well                            │
│   - Slower than average/Gaussian                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 사용법

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# Add salt-and-pepper noise (for testing)
def add_salt_pepper_noise(img, amount=0.05):
    noisy = img.copy()
    h, w = img.shape[:2]
    num_pixels = int(amount * h * w)

    # Salt (white)
    for _ in range(num_pixels):
        y, x = np.random.randint(0, h), np.random.randint(0, w)
        noisy[y, x] = 255

    # Pepper (black)
    for _ in range(num_pixels):
        y, x = np.random.randint(0, h), np.random.randint(0, w)
        noisy[y, x] = 0

    return noisy


noisy_img = add_salt_pepper_noise(img, 0.02)

# medianBlur(src, ksize)
# ksize: Only odd numbers allowed (3, 5, 7, ...)
median_3 = cv2.medianBlur(noisy_img, 3)
median_5 = cv2.medianBlur(noisy_img, 5)

# Compare: average blur, Gaussian blur
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
│                     Bilateral Filter Principle                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Regular Gaussian filter:                                      │
│   - Only considers distance → edges also blurred                │
│                                                                 │
│   Bilateral filter:                                             │
│   - Considers both distance (spatial) + color difference        │
│   - Only includes similar-colored pixels in average             │
│   - Preserves edges (where color difference is large)           │
│                                                                 │
│   Example:                                                      │
│   ┌─────────────────────────────────────────┐                   │
│   │ 100  100  100 │ 200  200  200 │          │                   │
│   │ 100  100  100 │ 200  200  200 │  ← Edge  │                   │
│   │ 100  100  100 │ 200  200  200 │          │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                 │
│   Gaussian: 100 and 200 mix to around 150                       │
│   Bilateral: 100 area stays 100, 200 area stays 200             │
│                                                                 │
│   Weight = spatial Gaussian × color Gaussian                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 사용법

```python
import cv2

img = cv2.imread('portrait.jpg')

# bilateralFilter(src, d, sigmaColor, sigmaSpace)
# d: Filter size (-1 = auto-calculate from sigmaSpace)
# sigmaColor: Sigma in color space (higher = average wider color range)
# sigmaSpace: Sigma in coordinate space (higher = consider wider area)

# Weak effect
bilateral_weak = cv2.bilateralFilter(img, 9, 50, 50)

# Medium effect
bilateral_medium = cv2.bilateralFilter(img, 9, 75, 75)

# Strong effect (painting-like)
bilateral_strong = cv2.bilateralFilter(img, 15, 100, 100)

# Very strong effect
bilateral_extreme = cv2.bilateralFilter(img, 15, 150, 150)
```

### 피부 스무딩 예제

```python
import cv2
import numpy as np

def skin_smoothing(img, strength='medium'):
    """Skin smoothing effect"""
    params = {
        'weak': (5, 30, 30),
        'medium': (9, 75, 75),
        'strong': (15, 100, 100),
        'extreme': (20, 150, 150)
    }

    d, sigmaColor, sigmaSpace = params.get(strength, params['medium'])

    # Apply bilateral filter
    smooth = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

    # Blend with original (natural effect)
    alpha = 0.7  # Blending ratio
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

# Compare processing time
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

# Visualization
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
# ddepth: Output image depth (-1 = same as input)
# kernel: User-defined kernel

# Create and apply average filter manually
kernel_avg = np.ones((5, 5), np.float32) / 25
avg_custom = cv2.filter2D(img, -1, kernel_avg)

# Same result as blur()
avg_builtin = cv2.blur(img, (5, 5))
print(f"Difference: {np.sum(np.abs(avg_custom - avg_builtin))}")  # 0
```

### 다양한 커스텀 커널

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 1. Emboss effect
kernel_emboss = np.array([
    [-2, -1, 0],
    [-1,  1, 1],
    [ 0,  1, 2]
])
emboss = cv2.filter2D(img, -1, kernel_emboss) + 128

# 2. Edge detection (Laplacian)
kernel_laplacian = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
])
laplacian = cv2.filter2D(img, -1, kernel_laplacian)

# 3. Sobel X (vertical edges)
kernel_sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
sobel_x = cv2.filter2D(img, -1, kernel_sobel_x)

# 4. Sobel Y (horizontal edges)
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
    """Visualize kernel application result and kernel"""
    result = cv2.filter2D(img, -1, kernel)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Kernel visualization
    im = axes[1].imshow(kernel, cmap='RdBu_r', vmin=-2, vmax=2)
    axes[1].set_title(f'Kernel ({kernel.shape[0]}x{kernel.shape[1]})')
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            axes[1].text(j, i, f'{kernel[i,j]:.1f}',
                        ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=axes[1])

    # Result
    axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[2].set_title(title)
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


img = cv2.imread('image.jpg')

# Example: Emboss kernel
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
│                      Sharpening Principle                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Sharpening = Original + (Original - Blur)                     │
│              = Original + High-frequency component              │
│              = Edge enhancement                                 │
│                                                                 │
│   Or directly with kernel:                                      │
│                                                                 │
│   Basic sharpening kernel:                                      │
│   ┌────┬────┬────┐                                             │
│   │  0 │ -1 │  0 │                                             │
│   ├────┼────┼────┤                                             │
│   │ -1 │  5 │ -1 │   Center = 5 (original weight)              │
│   ├────┼────┼────┤   Surrounding = -1 (subtract blur)          │
│   │  0 │ -1 │  0 │   Sum = 1 (preserve brightness)             │
│   └────┴────┴────┘                                             │
│                                                                 │
│   Strong sharpening kernel:                                     │
│   ┌────┬────┬────┐                                             │
│   │ -1 │ -1 │ -1 │                                             │
│   ├────┼────┼────┤                                             │
│   │ -1 │  9 │ -1 │   Center = 9                                │
│   ├────┼────┼────┤   Surrounding = -1 × 8 = -8                │
│   │ -1 │ -1 │ -1 │   Sum = 1                                   │
│   └────┴────┴────┘                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 샤프닝 구현

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# Method 1: Using kernel
kernel_sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
sharpened1 = cv2.filter2D(img, -1, kernel_sharpen)

# Method 2: Strong sharpening kernel
kernel_sharpen_strong = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
])
sharpened2 = cv2.filter2D(img, -1, kernel_sharpen_strong)

# Method 3: Unsharp Masking
def unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """
    Sharpening with unsharp masking

    amount: Sharpening strength (1.0 = standard)
    threshold: Edge detection threshold (noise prevention)
    """
    # Why Gaussian blur here: isolates low-frequency content; subtracting it leaves only
    # high-frequency detail (edges, texture) which we then amplify
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)

    # Original - Blur = Edges/Details
    # sharpened = Original + amount × (Original - Blur)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)

    if threshold > 0:
        # Why threshold: prevents amplifying flat-region noise — only sharpen where
        # there is already a meaningful intensity difference between original and blur
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
    Adaptive sharpening - apply sharpening only to edge regions
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    # Why dilate edges: the 1-pixel Canny edge is too narrow; dilation creates a soft
    # transition zone so sharpening doesn't produce hard halos at region boundaries
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # Why blur before sharpening: we need the low-frequency baseline to subtract from
    blurred = cv2.GaussianBlur(img, (5, 5), 1)

    # Sharpening
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)

    # Why blend instead of hard mask: keeps flat areas completely unchanged while
    # concentrating sharpening where edges already exist, avoiding noise amplification
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
# Hint: Add Gaussian noise
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
# Hint
def tilt_shift(img, focus_y, focus_height, blur_amount):
    # Create gradient mask
    # Blend blurred and original images using mask
    pass
```

---

## 9. 다음 단계

[모폴로지 연산](./06_Morphology.md)에서 침식, 팽창, 열기/닫기 등 형태학적 연산을 학습합니다!

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
| [기하학적 변환](./04_Geometric_Transforms.md) | 이미지 전처리 |
| [엣지 검출 (Edge Detection)](./08_Edge_Detection.md) | 필터링 후 엣지 검출 |

### 추가 참고

- [이미지 필터링 이론](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)
- [컨볼루션 시각화](https://setosa.io/ev/image-kernels/)

