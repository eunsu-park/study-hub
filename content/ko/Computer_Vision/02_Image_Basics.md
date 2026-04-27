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

### 이론: 장면에서 배열로: 샘플링과 양자화

#### A.1 연속 이미지 모델

카메라가 개입하기 전, 이미지는 **연속적인 2D 함수**입니다:

```
f : ℝ² → ℝᶜ          (흑백: c = 1; 컬러: c = 3)
```

이 함수는 이미지 평면의 어떤 영역 내 모든 점에 광도(또는 색 세 쌍)를 할당합니다. 실세계는 위치와 밝기 모두에서 연속적입니다. 이를 컴퓨터가 저장할 수 있는 형태로 바꾸려면 두 단계의 이산화가 필요합니다.

#### A.2 샘플링 — 위치를 이산화하기

카메라 센서는 유한한 수의 포토사이트(photosite) 격자로 구성됩니다. 각 포토사이트는 자신의 작은 직사각형 풋프린트에 도달한 빛을 적분하여 하나의 수를 보고합니다. 수학적으로, 이것은 연속 함수 `f(x, y)`를 정수 격자점에서 **샘플링**하는 것입니다:

```
I[i, j] = ∫∫ f(x, y) · w(x - j·Δ, y - i·Δ) dx dy
```

여기서 `Δ`는 픽셀 간격(pixel pitch)이고 `w`는 센서의 점 퍼짐 함수(PSF)입니다. 결과는 격자 정렬된 `H × W` 배열입니다.

이 단계에서 잃어버리는 것: **Nyquist 속도(`1 / 2Δ`) 이상의 주파수**. 두 픽셀보다 가는 디테일은 샘플들로 충실하게 표현될 수 없습니다. 렌즈가 이 고주파를 센서에 도달하기 전에 제거하지 못하면, 이들이 **앨리어싱(aliasing)**되어 눈에 보이는 저주파 무아레 패턴이 됩니다. 그래서 제대로 된 다운샘플링에는 먼저 저역 통과 필터가 필요하고(04 레슨의 `INTER_AREA` 규칙), 사진가들이 Bayer demosaicing과 앤티앨리어스 필터에 신경을 쓰는 것입니다.

#### A.3 양자화 — 밝기를 이산화하기

센서의 출력은 여전히 (노이즈가 있는) 실수입니다. 저장하려면 유한한 레벨 집합을 골라야 합니다. 8비트 unsigned 이미지의 경우, 실수 신호 `s ∈ [0, 1]`은 다음과 같이 매핑됩니다:

```
I[i, j] = round(s · 255)   ∈   {0, 1, 2, ..., 255}
```

이 반올림이 **양자화 오차(quantization error)**이며, 크기는 `±0.5 / 255 ≈ ±0.2%`로 경계가 지워집니다. 대부분의 자연 장면 사진에서는 보이지 않습니다. 매우 부드러운 그라데이션에서 띠(banding)로 보이기 시작합니다 — 맑은 하늘, 스튜디오 배경처럼 사람의 눈이 인접 레벨 간 전이를 감지할 수 있는 경우입니다.

띠를 깨는 두 가지 전략:

- **비트 깊이 증가**: 채널당 10, 12, 16비트. 16비트 PNG는 채널당 256레벨이 아닌 65,536레벨을 가집니다.
- **디더(Dither)**: 양자화 전에 서브 픽셀 노이즈를 추가해 띠 패턴을 텍스처로 흩뿌립니다(정상 시거리에서는 감지되지 않음).

### 이론: 컬러 채널 — 그리고 OpenCV가 BGR을 쓰는 이유

#### D.1 컬러 이미지는 세 개의 흑백 이미지

컬러 이미지는 픽셀당 세 개의 값을 저장합니다 — 보통 빨강, 초록, 파랑 각 채널 하나씩입니다. 이 세 값이 평행한 세 개의 "흑백" 2D 배열을 형성하며, 세 번째 축을 따라 쌓입니다:

```
img.shape == (H, W, 3)     # H행 × W열 × 3채널
img[y, x]                 # → 길이 3의 배열, 채널당 한 값씩
img[:, :, 0]              # → 전체 blue 채널   (OpenCV의 BGR 순서에서)
img[:, :, 1]              # → 전체 green 채널
img[:, :, 2]              # → 전체 red 채널
```

아이디어는 일반화됩니다. 4채널 이미지는 투명도를 위한 알파 채널을 추가하고(`shape == (H, W, 4)`), 다중 스펙트럼 / 초분광(hyperspectral) 이미지는 수십 또는 수백 개의 채널을 가질 수 있습니다.

#### D.2 RGB 대신 BGR인 이유

OpenCV의 `imread`와 `imshow`는 **BGR**(blue-green-red) 채널 순서를 사용합니다. 반면 거의 모든 다른 라이브러리(PIL, matplotlib, TensorFlow, PyTorch, 웹 표준)는 RGB를 사용합니다. 이유는 역사적입니다. 초기 24비트 Windows 비트맵 파일은 픽셀을 `blue, green, red` 바이트 순서로 저장했고(3바이트 픽셀의 리틀 엔디언 표현), OpenCV는 Windows에서 BMP를 zero-copy 로드할 수 있도록 이 레이아웃을 반영했습니다. 한 번 코드가 이에 의존하자 관례가 굳어졌습니다.

실용적 결과는 흔한 버그 한 부류입니다:

```python
img = cv2.imread('cat.jpg')          # BGR
plt.imshow(img)                      # RGB를 기대 — 색이 이상해짐
plt.imshow(img[:, :, ::-1])          # 수정: 채널 축을 뒤집기
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))   # 명시적 수정, 선호됨
```

matplotlib에서 이상하게 보이는 증상(빨강이 파랑이 되고, 하늘이 주황-빨강이 되는)은 너무 흔해서 이 규칙을 내면화할 가치가 있습니다: **OpenCV / 비-OpenCV 경계를 넘을 때는 `cv2.cvtColor`로 건너세요.**

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

### 이론: 비트 깊이와 동적 범위

#### C.1 기본값: `uint8`

OpenCV의 기본 이미지 dtype은 `numpy.uint8` — 채널당 8비트, 256레벨, 범위 `[0, 255]`입니다. 디스플레이 하드웨어와 일치하고(대부분의 모니터는 채널당 256레벨만 표시), 모든 표준 이미지 포맷(JPEG, PNG-8, BMP)이 저장하는 형식입니다.

기억해야 할 결과들:

- **산술이 255에서 포화(saturate)됩니다.** `cv2.add(200, 100) = 255`, 300이 아닙니다. `uint8` 배열에 `cv2.add` 대신 `+`를 쓰면 NumPy가 래핑(wraparound)합니다: `np.uint8(200) + np.uint8(100) = 44`. 둘 다 대개 원하는 결과가 아닙니다.
- **뺄셈은 0에서 포화 — 또는 연산자에 따라 래핑.** 같은 규칙.
- **Float 연산은 조심스럽게 변환해야 합니다.** 255로 나눠 `[0, 1]` float를 만들고 연산한 뒤 255를 곱하고 clip한 후 `uint8`로 다시 캐스팅하세요. clip을 잊으면 값이 255를 넘어 래핑되어 뒤집힌 것처럼 보이는 출력이 나옵니다.

#### C.2 8비트로 충분하지 않을 때

- **Raw 카메라 데이터**(12-14비트)에는 8비트 변환 시 잘려나가는 그림자와 하이라이트의 디테일이 담겨 있습니다.
- **의료 영상**(CT, MRI)은 일반적으로 채널당 12-16비트 — 진단 정보가 그 여분의 비트에 있습니다.
- **HDR 사진**은 선형 광도를 `float32`로 저장해 톤 범위가 여러 자릿수에 걸칩니다(가장 어두운 그림자부터 가장 밝은 하늘까지).
- **중간 연산**(필터링, Laplacian 누적 등)은 입력과 출력이 `uint8`이더라도 `float32`나 `float64`로 수행해 중간 오버플로와 양자화 오차를 피해야 합니다.

OpenCV는 `uint16`, `int16`, `float32`, `float64` dtype을 지원합니다. 관례는 `[0, 1]` 범위의 `float32` 이미지가 `[0, 255]` 범위의 `uint8` 이미지와 같은 밝기를 나타낸다는 것입니다.

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

### 이론: 픽셀 좌표계

OpenCV는 — 그리고 NumPy와 거의 모든 배열 기반 이미지 라이브러리는 — 픽셀을 `img[row, col]`로 인덱싱하며, 원점 `(0, 0)`은 **왼쪽 위** 모서리에 있습니다:

```
      col →  0    1    2    ...    W-1
row    ┌─────────────────────────────┐
  ↓    │                             │
  0    │ ●───────→ x                 │
       │ │                           │
  1    │ │                           │
       │ ▼                           │
  2    │ y                           │
       │                             │
  ...  │                             │
       │                             │
 H-1   │                             │
       └─────────────────────────────┘
```

몇 가지 중요한 결과:

- **`img[y, x]`에서 `y`가 먼저 옵니다**. `y`가 행 인덱스이기 때문입니다. 이는 표준 수학 표기 `f(x, y)`와 반대이며, 끊임없이 혼란과 버그의 원인이 됩니다. OpenCV 함수 호출에서 `(x, y)` 튜플을 받는 경우(예: `cv2.circle(img, (x, y), ...)`)는 순서가 다시 뒤바뀝니다. **경험 법칙: 배열은 `[y, x]`, 함수 인자는 `(x, y)`.**
- **`y`는 아래로 증가합니다.** 수학 교과서와는 반대이지만, 래스터 스캔 순서와 화면 좌표와는 일치합니다. 따라서 OpenCV의 "양의 회전 각"은 화면에서 봤을 때는 반시계 방향이지만, 순수 수학적으로는 시계 방향입니다.
- **픽셀 중심 vs 픽셀 모서리.** 어떤 관례는 좌표 `(0, 0)`을 왼쪽 위 픽셀의 중심으로 보고, 어떤 관례는 모서리로 봅니다. OpenCV는 *중심* 관례를 사용합니다 — 서브 픽셀 보간이나 기하 변환(04 레슨)을 할 때 중요합니다.

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

이미지 전체 대신 관련 하위 영역만 처리하는 것은 연산 비용을 줄이는 가장 효과적인 방법 중 하나입니다. ROI를 사용하면 복사-붙여넣기 합성, 얼굴 흐림 처리, 특정 영역 색상 보정 등의 작업도 간결하게 구현할 수 있습니다.

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

