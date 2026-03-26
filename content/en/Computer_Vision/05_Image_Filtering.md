# Image Filtering

## Overview

Image filtering is the process of transforming pixel values by considering neighboring pixels. Various effects such as noise removal, blur, and sharpening can be achieved. In this document, we'll learn from the concepts of kernels and convolution to various filter functions in OpenCV.

**Difficulty**: ⭐⭐ (Beginner-Intermediate)

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand kernel and convolution concepts
2. Learn various blur filters (`blur`, `GaussianBlur`, `medianBlur`, `bilateralFilter`)
3. Edge-preserving smoothing
4. Implement custom filters and sharpening

---

## Table of Contents

1. [Kernels and Convolution](#1-kernels-and-convolution)
2. [Average Blur - blur()](#2-average-blur---blur)
3. [Gaussian Blur - GaussianBlur()](#3-gaussian-blur---gaussianblur)
4. [Median Blur - medianBlur()](#4-median-blur---medianblur)
5. [Bilateral Filter - bilateralFilter()](#5-bilateral-filter---bilateralfilter)
6. [Custom Filter - filter2D()](#6-custom-filter---filter2d)
7. [Sharpening Filter](#7-sharpening-filter)
8. [Practice Problems](#8-practice-problems)
9. [Next Steps](#9-next-steps)
10. [References](#10-references)

---

## 1. Kernels and Convolution

### What is a Kernel?

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

### Convolution Operation

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

### Convolution Visualization

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

**Why does blurring work?** Noise in an image manifests as rapid, high-frequency pixel variations — adjacent pixels differ sharply without reflecting actual scene content. When a blur kernel averages neighboring pixels, these sharp random fluctuations cancel out, while genuine image features (which vary gradually across many pixels) are preserved. In signal processing terms, a blur kernel is a **low-pass filter** that suppresses high-frequency components. This is also why Gaussian blur is preferred as a preprocessing step before edge detection: it removes noise-induced false edges while preserving true structural boundaries.

---

## 2. Average Blur - blur()

### Basic Usage

Average blur is the simplest blur filter, using the average value of the kernel area.

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

### Average Blur Kernel

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

A generalized version of `blur()`.

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

## 3. Gaussian Blur - GaussianBlur()

### What is Gaussian Filter?

Gaussian filter is a blur filter that gives more weight to the center. It produces a natural blur effect.

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

### Usage

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

### Relationship Between Sigma and Kernel Size

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

### Average Blur vs Gaussian Blur

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

## 4. Median Blur - medianBlur()

### What is Median Filter?

Median filter uses the median value of the kernel area. Very effective for salt-and-pepper noise removal.

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

### Usage

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

## 5. Bilateral Filter - bilateralFilter()

### What is Bilateral Filter?

Bilateral filter smooths while preserving edges. Used for skin retouching, artistic effects, etc.

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

### Usage

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

### Skin Smoothing Example

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

### Blur Filter Comparison

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

## 6. Custom Filter - filter2D()

### filter2D() Usage

`filter2D()` allows performing convolution with custom-defined kernels.

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

### Various Custom Kernels

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

### Kernel Visualization Tool

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

## 7. Sharpening Filter

### Sharpening Principle

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

### Sharpening Implementation

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

### Adaptive Sharpening

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

## 8. Practice Problems

### Exercise 1: Noise Removal Comparison

Generate Gaussian noise and salt-and-pepper noise separately, and compare the removal effects with three blur filters (average, Gaussian, median). Perform quantitative comparison using PSNR values.

```python
# Hint: Add Gaussian noise
def add_gaussian_noise(img, mean=0, var=100):
    noise = np.random.normal(mean, var**0.5, img.shape)
    noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy
```

### Exercise 2: Real-Time Blur Intensity Control

Write a program that can adjust blur intensity (kernel size) with a trackbar on webcam video. Allow selection between Gaussian blur and bilateral filter.

### Exercise 3: Custom Emboss Directions

Design and test kernels that produce different emboss effects in 8 directions (up, down, left, right, and 4 diagonals).

### Exercise 4: Advanced Sharpening

Implement an advanced sharpening function with the following features:
1. Sharpening strength control (amount)
2. Blur radius control (radius)
3. Threshold application - ignore small changes
4. Separate handling for highlights/shadows

### Exercise 5: Miniature Effect (Tilt Shift)

Implement tilt-shift miniature effect using Gaussian blur and masks. Keep the center of the image sharp, with progressively more blur at top and bottom.

```python
# Hint
def tilt_shift(img, focus_y, focus_height, blur_amount):
    # Create gradient mask
    # Blend blurred and original images using mask
    pass
```

---

## 9. Next Steps

In [06_Morphology.md](./06_Morphology.md), you'll learn morphological operations such as erosion, dilation, opening/closing!

**Next topics**:
- Structuring Element
- Erosion and Dilation
- Opening and Closing
- Noise removal and object separation

---

## 10. References

### Official Documentation

- [blur() documentation](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga8c45db9afe636703801b0b2e440fce37)
- [GaussianBlur() documentation](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1)
- [medianBlur() documentation](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9)
- [bilateralFilter() documentation](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed)

### Related Learning Materials

| Folder | Related Content |
|--------|----------------|
| [04_Geometric_Transforms.md](./04_Geometric_Transforms.md) | Image preprocessing |
| [08_Edge_Detection.md](./08_Edge_Detection.md) | Edge detection after filtering |

### Additional References

- [Image Filtering Theory](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)
- [Convolution Visualization](https://setosa.io/ev/image-kernels/)
