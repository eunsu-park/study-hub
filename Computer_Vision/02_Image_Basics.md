# 이미지 기초 연산

## 개요

이미지 처리의 기본은 이미지 파일을 읽고, 표시하고, 저장하는 것입니다. 이 문서에서는 OpenCV의 기본 I/O 함수와 픽셀 단위 접근, 관심 영역(ROI) 설정 방법을 학습합니다.

**난이도**: ⭐ (입문)

**학습 목표**:
- `cv2.imread()`, `cv2.imshow()`, `cv2.imwrite()` 함수 마스터
- IMREAD 플래그 이해 및 활용
- 이미지 좌표 시스템 이해 (y, x 순서)
- 픽셀 단위 접근 및 수정
- ROI(관심 영역) 설정과 이미지 복사

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

# 기본 사용 (컬러로 읽기)
img = cv2.imread('image.jpg')

# 읽기 실패 확인 (항상 해야 함!)
if img is None:
    print("Error: 이미지를 읽을 수 없습니다.")
else:
    print(f"이미지 로드 성공: {img.shape}")
```

### IMREAD 플래그

```
┌─────────────────────────────────────────────────────────────────┐
│                       IMREAD 플래그 비교                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   원본 이미지 (PNG, 알파 채널 포함)                             │
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
│   3채널 BGR          2차원, 단일값     알파 채널 보존            │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 플래그 상세

```python
import cv2

# 1. IMREAD_COLOR (기본값, 1)
# - 컬러로 읽기 (알파 채널 무시)
# - 항상 3채널 BGR로 변환
img_color = cv2.imread('image.png', cv2.IMREAD_COLOR)
img_color = cv2.imread('image.png', 1)  # 동일
img_color = cv2.imread('image.png')     # 기본값이므로 생략 가능

# 2. IMREAD_GRAYSCALE (0)
# - 그레이스케일로 읽기
# - 2차원 배열 반환
img_gray = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
img_gray = cv2.imread('image.png', 0)  # 동일

# 3. IMREAD_UNCHANGED (-1)
# - 원본 그대로 읽기 (알파 채널 포함)
# - PNG의 투명도 정보 필요할 때 사용
img_unchanged = cv2.imread('image.png', cv2.IMREAD_UNCHANGED)
img_unchanged = cv2.imread('image.png', -1)  # 동일

# 결과 비교
print(f"COLOR: {img_color.shape}")        # (H, W, 3)
print(f"GRAYSCALE: {img_gray.shape}")     # (H, W)
print(f"UNCHANGED: {img_unchanged.shape}") # (H, W, 4) - PNG의 경우
```

### 추가 플래그

```python
import cv2

# IMREAD_ANYDEPTH: 16비트/32비트 이미지 그대로 로드
img_depth = cv2.imread('depth_map.png', cv2.IMREAD_ANYDEPTH)

# IMREAD_ANYCOLOR: 가능한 컬러 포맷 유지
img_any = cv2.imread('image.jpg', cv2.IMREAD_ANYCOLOR)

# 플래그 조합
# 16비트 그레이스케일 + 컬러 형식 유지
img_combined = cv2.imread('image.tiff',
                          cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
```

### 다양한 이미지 포맷

```python
import cv2

# 지원되는 주요 포맷
formats = [
    'image.jpg',   # JPEG
    'image.png',   # PNG (알파 채널 지원)
    'image.bmp',   # BMP
    'image.tiff',  # TIFF
    'image.webp',  # WebP
    'image.ppm',   # PPM/PGM/PBM
]

# 포맷별 읽기
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

# 창에 이미지 표시
cv2.imshow('Window Name', img)

# 키 입력 대기
key = cv2.waitKey(0)  # 0 = 무한 대기

# 모든 창 닫기
cv2.destroyAllWindows()
```

### waitKey() 상세

```
┌─────────────────────────────────────────────────────────────────┐
│                      waitKey() 동작                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   cv2.waitKey(delay)                                            │
│                                                                 │
│   delay = 0   → 키 입력까지 무한 대기                           │
│   delay > 0   → delay 밀리초 대기 후 자동 진행                   │
│   delay = 1   → 최소 대기 (비디오 재생에 자주 사용)              │
│                                                                 │
│   반환값: 눌린 키의 ASCII 코드 (-1 = 시간 초과)                  │
│                                                                 │
│   예시:                                                         │
│   key = cv2.waitKey(0)                                          │
│   if key == 27:        # ESC 키                                 │
│       break                                                     │
│   elif key == ord('q'):  # 'q' 키                               │
│       break                                                     │
│   elif key == ord('s'):  # 's' 키                               │
│       cv2.imwrite('saved.jpg', img)                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 여러 창 관리

```python
import cv2

img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# 여러 창 표시
cv2.imshow('Image 1', img1)
cv2.imshow('Image 2', img2)

# 창 위치 지정
cv2.namedWindow('Positioned', cv2.WINDOW_NORMAL)
cv2.moveWindow('Positioned', 100, 100)  # x=100, y=100 위치
cv2.imshow('Positioned', img1)

# 창 크기 조절 가능하게 설정
cv2.namedWindow('Resizable', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resizable', 800, 600)
cv2.imshow('Resizable', img1)

cv2.waitKey(0)

# 특정 창만 닫기
cv2.destroyWindow('Image 1')

# 모든 창 닫기
cv2.destroyAllWindows()
```

### 키 입력 처리 패턴

```python
import cv2

img = cv2.imread('image.jpg')
original = img.copy()

while True:
    cv2.imshow('Interactive', img)
    key = cv2.waitKey(1) & 0xFF  # 하위 8비트만 사용

    if key == 27:  # ESC
        break
    elif key == ord('r'):  # 'r' - 원본 복원
        img = original.copy()
        print("원본으로 복원")
    elif key == ord('g'):  # 'g' - 그레이스케일
        img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        print("그레이스케일 적용")
    elif key == ord('s'):  # 's' - 저장
        cv2.imwrite('output.jpg', img)
        print("저장 완료")

cv2.destroyAllWindows()
```

### Jupyter Notebook에서 이미지 표시

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')

# matplotlib 사용 (BGR → RGB 변환 필요)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 6))
plt.imshow(img_rgb)
plt.title('Image Display in Jupyter')
plt.axis('off')
plt.show()

# 여러 이미지 동시 표시
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_rgb)
axes[0].set_title('Original')
axes[0].axis('off')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
axes[1].imshow(gray, cmap='gray')
axes[1].set_title('Grayscale')
axes[1].axis('off')

# B, G, R 채널 분리
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

# 기본 저장
success = cv2.imwrite('output.jpg', img)

if success:
    print("저장 성공!")
else:
    print("저장 실패!")

# 포맷 변환하여 저장
cv2.imwrite('output.png', img)   # JPEG → PNG
cv2.imwrite('output.bmp', img)   # JPEG → BMP
```

### 압축 품질 설정

```python
import cv2

img = cv2.imread('input.jpg')

# JPEG 품질 (0-100, 기본값 95)
cv2.imwrite('high_quality.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
cv2.imwrite('low_quality.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 30])

# PNG 압축 레벨 (0-9, 기본값 3)
# 0 = 압축 없음 (빠름, 큰 파일)
# 9 = 최대 압축 (느림, 작은 파일)
cv2.imwrite('fast_compress.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv2.imwrite('max_compress.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

# WebP 품질 (0-100)
cv2.imwrite('output.webp', img, [cv2.IMWRITE_WEBP_QUALITY, 80])
```

### 파일 크기 비교

```python
import cv2
import os

img = cv2.imread('input.jpg')

# 다양한 품질로 저장
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
print(f"높이: {height}px")
print(f"너비: {width}px")
print(f"채널: {channels}")

# dtype: 데이터 타입
print(f"데이터 타입: {img.dtype}")  # uint8

# size: 전체 원소 개수
print(f"전체 원소: {img.size}")  # H * W * C

# 그레이스케일 이미지
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"그레이 Shape: {gray.shape}")  # (height, width) - 채널 없음

# 안전하게 채널 수 확인
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
    """이미지 파일의 상세 정보를 딕셔너리로 반환"""
    info = {'filepath': filepath}

    # 파일 존재 확인
    if not os.path.exists(filepath):
        info['error'] = '파일이 존재하지 않습니다'
        return info

    # 파일 크기
    info['file_size_kb'] = os.path.getsize(filepath) / 1024

    # 이미지 로드
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        info['error'] = '이미지를 읽을 수 없습니다'
        return info

    # 기본 정보
    info['shape'] = img.shape
    info['dtype'] = str(img.dtype)
    info['height'] = img.shape[0]
    info['width'] = img.shape[1]
    info['channels'] = img.shape[2] if len(img.shape) == 3 else 1

    # 통계 정보
    info['min_value'] = int(img.min())
    info['max_value'] = int(img.max())
    info['mean_value'] = float(img.mean())

    return info

# 사용 예
info = get_image_info('sample.jpg')
for key, value in info.items():
    print(f"{key}: {value}")
```

---

## 5. 좌표 시스템과 픽셀 접근

### OpenCV 좌표 시스템

```
┌─────────────────────────────────────────────────────────────────┐
│                     OpenCV 좌표 시스템                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   (0,0) ────────────────────────────────▶ x (width, 열)         │
│     │                                                           │
│     │    ┌───────────────────────────┐                         │
│     │    │ (0,0)  (1,0)  (2,0)  ...  │                         │
│     │    │ (0,1)  (1,1)  (2,1)  ...  │                         │
│     │    │ (0,2)  (1,2)  (2,2)  ...  │                         │
│     │    │  ...    ...    ...   ...  │                         │
│     │    └───────────────────────────┘                         │
│     ▼                                                           │
│   y (height, 행)                                                │
│                                                                 │
│   중요! 배열 인덱싱: img[y, x] 또는 img[행, 열]                  │
│         OpenCV 함수: (x, y) 순서 사용                           │
│                                                                 │
│   예: img[100, 200]     → y=100, x=200 위치의 픽셀              │
│       cv2.circle(img, (200, 100), ...)  → x=200, y=100 위치     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 픽셀 접근

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 단일 픽셀 읽기 (y, x 순서!)
pixel = img[100, 200]  # y=100, x=200 위치
print(f"픽셀 값 (BGR): {pixel}")  # [B, G, R]

# 개별 채널 접근
b = img[100, 200, 0]  # Blue
g = img[100, 200, 1]  # Green
r = img[100, 200, 2]  # Red
print(f"B={b}, G={g}, R={r}")

# 그레이스케일 이미지
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
pixel_gray = gray[100, 200]  # 단일 값
print(f"그레이스케일 값: {pixel_gray}")
```

### 픽셀 수정

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 단일 픽셀 수정
img[100, 200] = [255, 0, 0]  # 파란색으로 변경

# 영역 수정 (100x100 영역을 빨간색으로)
img[0:100, 0:100] = [0, 0, 255]  # BGR에서 빨간색

# 특정 채널만 수정
img[0:100, 100:200, 0] = 0    # Blue 채널을 0으로
img[0:100, 100:200, 1] = 0    # Green 채널을 0으로
img[0:100, 100:200, 2] = 255  # Red 채널을 255로

cv2.imshow('Modified', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### item()과 itemset() (단일 픽셀용, 더 빠름)

```python
import cv2

img = cv2.imread('image.jpg')

# item(): 단일 값 접근 (더 빠름)
b = img.item(100, 200, 0)
g = img.item(100, 200, 1)
r = img.item(100, 200, 2)

# itemset(): 단일 값 수정 (더 빠름)
img.itemset((100, 200, 0), 255)  # Blue = 255
img.itemset((100, 200, 1), 0)    # Green = 0
img.itemset((100, 200, 2), 0)    # Red = 0

# 성능 비교
import time

# 일반 인덱싱
start = time.time()
for i in range(10000):
    val = img[100, 200, 0]
print(f"일반 인덱싱: {time.time() - start:.4f}초")

# item() 사용
start = time.time()
for i in range(10000):
    val = img.item(100, 200, 0)
print(f"item(): {time.time() - start:.4f}초")
```

---

## 6. ROI와 이미지 복사

### ROI (Region of Interest)

```
┌─────────────────────────────────────────────────────────────────┐
│                       ROI 개념                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   원본 이미지 (img)                                              │
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
│   주의: NumPy 슬라이싱은 뷰(view)를 반환!                         │
│         roi 수정 → 원본도 수정됨                                 │
│         복사가 필요하면 .copy() 사용                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### ROI 설정 및 사용

```python
import cv2

img = cv2.imread('image.jpg')

# ROI 추출 (y1:y2, x1:x2)
# 좌상단 (100, 50)부터 우하단 (300, 250)까지
roi = img[50:250, 100:300]

print(f"원본 크기: {img.shape}")
print(f"ROI 크기: {roi.shape}")  # (200, 200, 3)

# ROI 표시
cv2.imshow('Original', img)
cv2.imshow('ROI', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### ROI 복사와 붙여넣기

```python
import cv2

img = cv2.imread('image.jpg')

# ROI 복사 (중요: .copy() 사용)
roi = img[50:150, 100:200].copy()

# 다른 위치에 붙여넣기
img[200:300, 300:400] = roi  # 크기가 같아야 함!

# 이미지 내 영역 복사
# 좌상단 100x100을 우하단에 복사
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

# 뷰 (View) - 원본과 메모리 공유
roi_view = img[50:150, 50:150]
roi_view[:] = 0  # ROI를 검은색으로
print(f"원본 변경됨: {img[100, 100, 0]}")  # 0

# 원본 복원
img = cv2.imread('image.jpg')

# 복사 (Copy) - 독립적인 메모리
roi_copy = img[50:150, 50:150].copy()
roi_copy[:] = 0  # 복사본만 검은색으로
print(f"원본 유지됨: {img[100, 100, 0]}")  # 원래 값
```

### 전체 이미지 복사

```python
import cv2

img = cv2.imread('image.jpg')

# 방법 1: .copy() 메서드
img_copy1 = img.copy()

# 방법 2: NumPy copy
import numpy as np
img_copy2 = np.copy(img)

# 방법 3: 슬라이싱 후 copy (권장하지 않음)
img_copy3 = img[:].copy()

# 잘못된 복사 (뷰 생성)
img_wrong = img  # 같은 객체 참조!
img_wrong[0, 0] = [0, 0, 0]
print(f"원본도 변경됨: {img[0, 0]}")  # [0, 0, 0]
```

### 실용적인 ROI 예제

```python
import cv2

def extract_face_region(img, x, y, w, h):
    """얼굴 영역 추출 (경계 체크 포함)"""
    h_img, w_img = img.shape[:2]

    # 경계 체크
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)

    return img[y1:y2, x1:x2].copy()


def apply_mosaic(img, x, y, w, h, ratio=0.1):
    """특정 영역에 모자이크 적용"""
    roi = img[y:y+h, x:x+w]

    # 축소 후 확대 (모자이크 효과)
    small = cv2.resize(roi, None, fx=ratio, fy=ratio,
                       interpolation=cv2.INTER_NEAREST)
    mosaic = cv2.resize(small, (w, h),
                        interpolation=cv2.INTER_NEAREST)

    img[y:y+h, x:x+w] = mosaic
    return img


# 사용 예
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
# 힌트
import cv2

filepath = 'test.png'
# COLOR, GRAYSCALE, UNCHANGED로 읽기
# shape 비교
```

### 연습 2: 이미지 품질 분석기

JPEG 이미지를 다양한 품질(10, 30, 50, 70, 90)로 저장하고, 각각의 파일 크기와 PSNR(Peak Signal-to-Noise Ratio)을 계산하세요.

```python
# 힌트: PSNR 계산
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
│빨강│노랑│초록│청록│
├────┼────┼────┼────┤
│파랑│보라│흰색│검정│
├────┼────┼────┼────┤
│... │... │... │... │
└────┴────┴────┴────┘
```

### 연습 4: 이미지 테두리 추가

이미지 주변에 10픽셀 두께의 테두리를 추가하는 함수를 작성하세요. (이미지 크기가 증가해야 함)

```python
def add_border(img, thickness=10, color=(0, 0, 255)):
    """이미지에 테두리 추가"""
    # 힌트: numpy.pad 또는 cv2.copyMakeBorder 사용
    pass
```

### 연습 5: 픽셀 기반 그라디언트

300x300 이미지를 만들고 왼쪽에서 오른쪽으로 검은색에서 흰색으로 변하는 수평 그라디언트를 만드세요. 반복문 없이 NumPy 브로드캐스팅을 사용하세요.

```python
# 힌트
import numpy as np
gradient = np.linspace(0, 255, 300)  # 0~255 값 300개
```

---

## 8. 다음 단계

[03_Color_Spaces.md](./03_Color_Spaces.md)에서 BGR, RGB, HSV, LAB 등 다양한 색상 공간과 색상 기반 객체 추적을 학습합니다!

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
| [01_Environment_Setup.md](./01_Environment_Setup.md) | 설치 및 기본 개념 |

