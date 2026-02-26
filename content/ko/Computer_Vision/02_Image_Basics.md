# 이미지 기초 연산

## 개요

이미지 처리의 기본은 이미지 파일을 읽고, 표시하고, 저장하는 것입니다. 이 문서에서는 OpenCV의 기본 I/O 함수와 픽셀 단위 접근, 관심 영역(ROI) 설정 방법을 학습합니다.

**난이도**: ⭐ (입문)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `cv2.imread()`, `cv2.imshow()`, `cv2.imwrite()` 함수 마스터
2. IMREAD 플래그 이해 및 활용
3. 이미지 좌표 시스템 이해 (y, x 순서)
4. 픽셀 단위 접근 및 수정
5. ROI(관심 영역) 설정과 이미지 복사

---

## 목차

1. [이미지 읽기 - imread()](#1-이미지-읽기---imread)
2. [이미지 표시 - imshow()](#2-이미지-표시---imshow)
3. [이미지 저장 - imwrite()](#3-이미지-저장---imwrite)
4. [이미지 속성 확인](#4-이미지-속성-확인)
5. [좌표 시스템과 픽셀 접근](#5-좌표-시스템과-픽셀-접근)
6. [ROI와 이미지 복사](#6-roi와-이미지-복사)
7. [연습 문제](#7-연습-문제)
8. [다음 단계](#8-다음-단계)
9. [참고 자료](#9-참고-자료)

---

## 1. 이미지 읽기 - imread()

### 기본 사용법

```python
import cv2

# imread returns None silently on failure (no exception) — always guard against
# this; skipping the check leads to cryptic AttributeError crashes later
img = cv2.imread('image.jpg')

if img is None:
    print("Error: Cannot read image.")
else:
    print(f"Image loaded successfully: {img.shape}")
```

### IMREAD 플래그

```
┌─────────────────────────────────────────────────────────────────┐
│                       IMREAD Flag Comparison                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Original Image (PNG with alpha channel)                      │
│   ┌─────────────────────────────────────────────────────┐      │
│   │  R   G   B   A  │  R   G   B   A  │  R   G   B   A  │      │
│   │ 255 100  50 200 │ 255 100  50 200 │ 255 100  50 200 │      │
│   └─────────────────────────────────────────────────────┘      │
│                          │                                     │
│        ┌─────────────────┼─────────────────┐                   │
│        ▼                 ▼                 ▼                   │
│                                                                │
│   IMREAD_COLOR       IMREAD_GRAYSCALE  IMREAD_UNCHANGED        │
│   ┌───────────┐      ┌───────────┐     ┌───────────────┐       │
│   │ B  G  R   │      │   Gray    │     │ B  G  R  A    │       │
│   │ 50 100 255│      │    123    │     │ 50 100 255 200│       │
│   └───────────┘      └───────────┘     └───────────────┘       │
│   shape: (H,W,3)     shape: (H,W)      shape: (H,W,4)          │
│   3-channel BGR      2D, single value  Alpha channel preserved  │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 플래그 상세

```python
import cv2

# IMREAD_COLOR: always produces a 3-channel BGR array regardless of source format
# (even for grayscale JPEGs) — this consistency simplifies downstream processing
img_color = cv2.imread('image.png', cv2.IMREAD_COLOR)
img_color = cv2.imread('image.png', 1)  # Same
img_color = cv2.imread('image.png')     # Can omit (default)

# IMREAD_GRAYSCALE: returns a 2D array — saves 2/3 of memory vs COLOR for tasks
# that don't need color (edge detection, thresholding, template matching)
img_gray = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
img_gray = cv2.imread('image.png', 0)  # Same

# IMREAD_UNCHANGED: the only flag that preserves the alpha channel —
# essential when you need transparency data (compositing, masking operations)
img_unchanged = cv2.imread('image.png', cv2.IMREAD_UNCHANGED)
img_unchanged = cv2.imread('image.png', -1)  # Same

# Compare results
print(f"COLOR: {img_color.shape}")        # (H, W, 3)
print(f"GRAYSCALE: {img_gray.shape}")     # (H, W)
print(f"UNCHANGED: {img_unchanged.shape}") # (H, W, 4) - for PNG
```

### 추가 플래그

```python
import cv2

# IMREAD_ANYDEPTH: Load 16-bit/32-bit images as is
img_depth = cv2.imread('depth_map.png', cv2.IMREAD_ANYDEPTH)

# IMREAD_ANYCOLOR: Maintain possible color formats
img_any = cv2.imread('image.jpg', cv2.IMREAD_ANYCOLOR)

# Combining flags
# 16-bit grayscale + maintain color format
img_combined = cv2.imread('image.tiff',
                          cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
```

### 다양한 이미지 포맷

```python
import cv2

# Supported major formats
formats = [
    'image.jpg',   # JPEG
    'image.png',   # PNG (alpha channel supported)
    'image.bmp',   # BMP
    'image.tiff',  # TIFF
    'image.webp',  # WebP
    'image.ppm',   # PPM/PGM/PBM
]

# Read by format
for filepath in formats:
    img = cv2.imread(filepath)
    if img is not None:
        print(f"{filepath}: {img.shape}")
```

---

## 2. 이미지 표시 - imshow()

### 기본 사용법

```python
import cv2

img = cv2.imread('image.jpg')

# Display image in window
cv2.imshow('Window Name', img)

# Wait for key press
key = cv2.waitKey(0)  # 0 = wait indefinitely

# Close all windows
cv2.destroyAllWindows()
```

### waitKey() 상세

```
┌─────────────────────────────────────────────────────────────────┐
│                      waitKey() Behavior                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   cv2.waitKey(delay)                                            │
│                                                                 │
│   delay = 0   → Wait indefinitely until key press               │
│   delay > 0   → Wait delay milliseconds then proceed            │
│   delay = 1   → Minimum wait (often used for video playback)    │
│                                                                 │
│   Return value: ASCII code of pressed key (-1 = timeout)        │
│                                                                 │
│   Examples:                                                     │
│   key = cv2.waitKey(0)                                          │
│   if key == 27:        # ESC key                                │
│       break                                                     │
│   elif key == ord('q'):  # 'q' key                              │
│       break                                                     │
│   elif key == ord('s'):  # 's' key                              │
│       cv2.imwrite('saved.jpg', img)                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 여러 창 관리

```python
import cv2

img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# Display multiple windows
cv2.imshow('Image 1', img1)
cv2.imshow('Image 2', img2)

# Set window position
cv2.namedWindow('Positioned', cv2.WINDOW_NORMAL)
cv2.moveWindow('Positioned', 100, 100)  # x=100, y=100 position
cv2.imshow('Positioned', img1)

# Make window resizable
cv2.namedWindow('Resizable', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resizable', 800, 600)
cv2.imshow('Resizable', img1)

cv2.waitKey(0)

# Close specific window
cv2.destroyWindow('Image 1')

# Close all windows
cv2.destroyAllWindows()
```

### 키 입력 처리 패턴

```python
import cv2

img = cv2.imread('image.jpg')
original = img.copy()  # Keep a pristine copy — img will be modified in-loop

while True:
    cv2.imshow('Interactive', img)
    # & 0xFF masks the return value to 8 bits: on Linux, waitKey() can return
    # values > 255 due to keyboard modifier flags; masking ensures reliable comparison
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('r'):  # 'r' - restore original
        img = original.copy()
        print("Restored to original")
    elif key == ord('g'):  # 'g' - grayscale
        img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        print("Applied grayscale")
    elif key == ord('s'):  # 's' - save
        cv2.imwrite('output.jpg', img)
        print("Saved")

cv2.destroyAllWindows()
```

### Jupyter Notebook에서 이미지 표시

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')

# Using matplotlib (need BGR → RGB conversion)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 6))
plt.imshow(img_rgb)
plt.title('Image Display in Jupyter')
plt.axis('off')
plt.show()

# Display multiple images simultaneously
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_rgb)
axes[0].set_title('Original')
axes[0].axis('off')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
axes[1].imshow(gray, cmap='gray')
axes[1].set_title('Grayscale')
axes[1].axis('off')

# Split B, G, R channels
b, g, r = cv2.split(img)
axes[2].imshow(r, cmap='gray')
axes[2].set_title('Red Channel')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

---

## 3. 이미지 저장 - imwrite()

### 기본 사용법

```python
import cv2

img = cv2.imread('input.jpg')

# Basic save
success = cv2.imwrite('output.jpg', img)

if success:
    print("Save successful!")
else:
    print("Save failed!")

# Save with format conversion
cv2.imwrite('output.png', img)   # JPEG → PNG
cv2.imwrite('output.bmp', img)   # JPEG → BMP
```

### 압축 품질 설정

```python
import cv2

img = cv2.imread('input.jpg')

# JPEG is lossy: quality=95 is near-lossless (good for archiving); quality=30
# cuts file size dramatically at the cost of visible artifacts — use for thumbnails
cv2.imwrite('high_quality.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
cv2.imwrite('low_quality.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 30])

# PNG is lossless — compression only affects speed/file size, never quality
# Use compression=0 when writing many frames in a loop (speed matters more)
cv2.imwrite('fast_compress.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv2.imwrite('max_compress.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

# WebP offers better compression than JPEG at equivalent visual quality
cv2.imwrite('output.webp', img, [cv2.IMWRITE_WEBP_QUALITY, 80])
```

### 파일 크기 비교

```python
import cv2
import os

img = cv2.imread('input.jpg')

# Save with various qualities
qualities = [10, 30, 50, 70, 90]
for q in qualities:
    filename = f'quality_{q}.jpg'
    cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, q])
    size_kb = os.path.getsize(filename) / 1024
    print(f"Quality {q}: {size_kb:.1f} KB")
```

---

## 4. 이미지 속성 확인

### shape, dtype, size

```python
import cv2

img = cv2.imread('image.jpg')

# shape: (height, width, channels)
print(f"Shape: {img.shape}")
height, width, channels = img.shape
print(f"Height: {height}px")
print(f"Width: {width}px")
print(f"Channels: {channels}")

# dtype: data type
print(f"Data type: {img.dtype}")  # uint8

# size: total number of elements
print(f"Total elements: {img.size}")  # H * W * C

# Grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"Gray Shape: {gray.shape}")  # (height, width) - no channels

# Safely check channel count
if len(img.shape) == 3:
    h, w, c = img.shape
else:
    h, w = img.shape
    c = 1
```

### 이미지 정보 유틸리티 함수

```python
import cv2
import os

def get_image_info(filepath):
    """Returns detailed image file information as dictionary"""
    info = {'filepath': filepath}

    # Check file exists
    if not os.path.exists(filepath):
        info['error'] = 'File does not exist'
        return info

    # File size
    info['file_size_kb'] = os.path.getsize(filepath) / 1024

    # Load image
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        info['error'] = 'Cannot read image'
        return info

    # Basic info
    info['shape'] = img.shape
    info['dtype'] = str(img.dtype)
    info['height'] = img.shape[0]
    info['width'] = img.shape[1]
    info['channels'] = img.shape[2] if len(img.shape) == 3 else 1

    # Statistics
    info['min_value'] = int(img.min())
    info['max_value'] = int(img.max())
    info['mean_value'] = float(img.mean())

    return info

# Usage example
info = get_image_info('sample.jpg')
for key, value in info.items():
    print(f"{key}: {value}")
```

---

## 5. 좌표 시스템과 픽셀 접근

### OpenCV 좌표 시스템

```
┌─────────────────────────────────────────────────────────────────┐
│                     OpenCV Coordinate System                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   (0,0) ────────────────────────────────▶ x (width, columns)    │
│     │                                                           │
│     │    ┌───────────────────────────┐                         │
│     │    │ (0,0)  (1,0)  (2,0)  ...  │                         │
│     │    │ (0,1)  (1,1)  (2,1)  ...  │                         │
│     │    │ (0,2)  (1,2)  (2,2)  ...  │                         │
│     │    │  ...    ...    ...   ...  │                         │
│     │    └───────────────────────────┘                         │
│     ▼                                                           │
│   y (height, rows)                                              │
│                                                                 │
│   Important! Array indexing: img[y, x] or img[row, column]     │
│              OpenCV functions: (x, y) order                     │
│                                                                 │
│   e.g.: img[100, 200]     → pixel at y=100, x=200              │
│         cv2.circle(img, (200, 100), ...)  → at x=200, y=100    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 픽셀 접근

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# Read single pixel (y, x order!)
pixel = img[100, 200]  # position y=100, x=200
print(f"Pixel value (BGR): {pixel}")  # [B, G, R]

# Access individual channels
b = img[100, 200, 0]  # Blue
g = img[100, 200, 1]  # Green
r = img[100, 200, 2]  # Red
print(f"B={b}, G={g}, R={r}")

# Grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
pixel_gray = gray[100, 200]  # single value
print(f"Grayscale value: {pixel_gray}")
```

### 픽셀 수정

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# Modify single pixel
img[100, 200] = [255, 0, 0]  # Change to blue

# Modify region (100x100 region to red)
img[0:100, 0:100] = [0, 0, 255]  # Red in BGR

# Modify specific channel only
img[0:100, 100:200, 0] = 0    # Blue channel to 0
img[0:100, 100:200, 1] = 0    # Green channel to 0
img[0:100, 100:200, 2] = 255  # Red channel to 255

cv2.imshow('Modified', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### item()과 itemset() (단일 픽셀용, 더 빠름)

```python
import cv2

img = cv2.imread('image.jpg')

# item(): access single value (faster)
b = img.item(100, 200, 0)
g = img.item(100, 200, 1)
r = img.item(100, 200, 2)

# itemset(): modify single value (faster)
img.itemset((100, 200, 0), 255)  # Blue = 255
img.itemset((100, 200, 1), 0)    # Green = 0
img.itemset((100, 200, 2), 0)    # Red = 0

# Performance comparison
import time

# Regular indexing
start = time.time()
for i in range(10000):
    val = img[100, 200, 0]
print(f"Regular indexing: {time.time() - start:.4f}s")

# Using item()
start = time.time()
for i in range(10000):
    val = img.item(100, 200, 0)
print(f"item(): {time.time() - start:.4f}s")
```

---

## 6. ROI와 이미지 복사

### ROI (Region of Interest)

```
┌─────────────────────────────────────────────────────────────────┐
│                       ROI Concept                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Original Image (img)                                          │
│   ┌────────────────────────────────────┐                        │
│   │                                    │                        │
│   │      y1──────────────┐             │                        │
│   │       │    ROI       │             │                        │
│   │       │              │             │                        │
│   │       │              │             │                        │
│   │      y2──────────────┘             │                        │
│   │      x1             x2             │                        │
│   │                                    │                        │
│   └────────────────────────────────────┘                        │
│                                                                 │
│   roi = img[y1:y2, x1:x2]                                       │
│                                                                 │
│   Note: NumPy slicing returns a view!                           │
│         roi modification → original also modified               │
│         Use .copy() if copy is needed                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### ROI 설정 및 사용

```python
import cv2

img = cv2.imread('image.jpg')

# Extract ROI (y1:y2, x1:x2)
# From top-left (100, 50) to bottom-right (300, 250)
roi = img[50:250, 100:300]

print(f"Original size: {img.shape}")
print(f"ROI size: {roi.shape}")  # (200, 200, 3)

# Display ROI
cv2.imshow('Original', img)
cv2.imshow('ROI', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### ROI 복사와 붙여넣기

```python
import cv2

img = cv2.imread('image.jpg')

# .copy() creates an independent array — without it, roi is a view into img,
# and modifying it would unexpectedly change the source region too
roi = img[50:150, 100:200].copy()

# Paste to another location — NumPy assigns by value, so this is a true copy
img[200:300, 300:400] = roi  # Sizes must match!

# Copy region within image — copy() is critical here: without it, reading the
# source and writing to the destination could overlap and corrupt the result
src_region = img[0:100, 0:100].copy()
img[-100:, -100:] = src_region

cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 뷰(View) vs 복사(Copy)

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')
original_value = img[100, 100, 0]

# View - shares memory with original
roi_view = img[50:150, 50:150]
roi_view[:] = 0  # Make ROI black
print(f"Original modified: {img[100, 100, 0]}")  # 0

# Restore original
img = cv2.imread('image.jpg')

# Copy - independent memory
roi_copy = img[50:150, 50:150].copy()
roi_copy[:] = 0  # Only copy becomes black
print(f"Original preserved: {img[100, 100, 0]}")  # Original value
```

### 전체 이미지 복사

```python
import cv2

img = cv2.imread('image.jpg')

# Method 1: .copy() method
img_copy1 = img.copy()

# Method 2: NumPy copy
import numpy as np
img_copy2 = np.copy(img)

# Method 3: Slicing then copy (not recommended)
img_copy3 = img[:].copy()

# Wrong copy (creates view)
img_wrong = img  # Same object reference!
img_wrong[0, 0] = [0, 0, 0]
print(f"Original also changed: {img[0, 0]}")  # [0, 0, 0]
```

### 실용적인 ROI 예제

```python
import cv2

def extract_face_region(img, x, y, w, h):
    """Extract face region (with boundary check)"""
    h_img, w_img = img.shape[:2]

    # Boundary check
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)

    return img[y1:y2, x1:x2].copy()


def apply_mosaic(img, x, y, w, h, ratio=0.1):
    """Apply mosaic to specific region"""
    roi = img[y:y+h, x:x+w]

    # Shrink then enlarge (mosaic effect)
    small = cv2.resize(roi, None, fx=ratio, fy=ratio,
                       interpolation=cv2.INTER_NEAREST)
    mosaic = cv2.resize(small, (w, h),
                        interpolation=cv2.INTER_NEAREST)

    img[y:y+h, x:x+w] = mosaic
    return img


# Usage example
img = cv2.imread('image.jpg')
img = apply_mosaic(img, 100, 100, 200, 200, ratio=0.05)
cv2.imshow('Mosaic', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 7. 연습 문제

### 연습 1: 이미지 읽기 모드 비교

하나의 이미지를 세 가지 모드(COLOR, GRAYSCALE, UNCHANGED)로 읽고 각각의 shape를 비교하세요. PNG 파일(투명도 포함)과 JPEG 파일로 테스트해보세요.

```python
# Hint
import cv2

filepath = 'test.png'
# Read in COLOR, GRAYSCALE, UNCHANGED
# Compare shapes
```

### 연습 2: 이미지 품질 분석기

JPEG 이미지를 다양한 품질(10, 30, 50, 70, 90)로 저장하고, 각각의 파일 크기와 PSNR(Peak Signal-to-Noise Ratio)을 계산하세요.

```python
# Hint: PSNR calculation
def calculate_psnr(original, compressed):
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
```

### 연습 3: 색상 격자 만들기

400x400 이미지를 만들고 100x100 크기의 16개 셀로 나누어 각각 다른 색상으로 채우세요. ROI를 사용하세요.

```
┌────┬────┬────┬────┐
│Red │Yell│Gren│Cyan│
├────┼────┼────┼────┤
│Blue│Prpl│Wht │Blck│
├────┼────┼────┼────┤
│... │... │... │... │
└────┴────┴────┴────┘
```

### 연습 4: 이미지 테두리 추가

이미지 주변에 10픽셀 두께의 테두리를 추가하는 함수를 작성하세요. (이미지 크기가 증가해야 함)

```python
def add_border(img, thickness=10, color=(0, 0, 255)):
    """Add border to image"""
    # Hint: use numpy.pad or cv2.copyMakeBorder
    pass
```

### 연습 5: 픽셀 기반 그라디언트

300x300 이미지를 만들고 왼쪽에서 오른쪽으로 검은색에서 흰색으로 변하는 수평 그라디언트를 만드세요. 반복문 없이 NumPy 브로드캐스팅을 사용하세요.

```python
# Hint
import numpy as np
gradient = np.linspace(0, 255, 300)  # 300 values from 0~255
```

---

## 8. 다음 단계

[색상 공간](./03_Color_Spaces.md)에서 BGR, RGB, HSV, LAB 등 다양한 색상 공간과 색상 기반 객체 추적을 학습합니다!

**다음에 배울 내용**:
- BGR vs RGB 차이점
- HSV 색상 공간의 이해
- `cv2.cvtColor()`로 색상 공간 변환
- 색상 기반 객체 추적

---

## 9. 참고 자료

### 공식 문서

- [imread() 문서](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56)
- [imshow() 문서](https://docs.opencv.org/4.x/d7/dfc/group__highgui.html#ga453d42fe4cb60e5723281a89973ee563)
- [imwrite() 문서](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce)

### 관련 학습 자료

| 폴더 | 관련 내용 |
|------|----------|
| [Python/](../Python/) | NumPy 슬라이싱, 배열 연산 |
| [환경 설정 및 기초](./01_Environment_Setup.md) | 설치 및 기본 개념 |

