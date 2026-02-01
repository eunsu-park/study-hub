# 기하학적 변환

## 개요

기하학적 변환(Geometric Transformation)은 이미지의 공간적 위치를 변경하는 작업입니다. 크기 조절, 회전, 이동, 뒤집기, 원근 변환 등이 포함됩니다. 이 문서에서는 OpenCV의 기하학적 변환 함수들과 실제 활용 예제를 학습합니다.

**난이도**: ⭐⭐ (초급-중급)

**학습 목표**:
- `cv2.resize()`와 보간법(interpolation) 이해
- 회전, 뒤집기 함수 사용
- 어파인 변환 (warpAffine) 활용
- 원근 변환 (warpPerspective) 활용
- 문서 스캔/교정 예제 구현

---

## 목차

1. [이미지 크기 조절 - resize()](#1-이미지-크기-조절---resize)
2. [뒤집기와 회전 - flip(), rotate()](#2-뒤집기와-회전---flip-rotate)
3. [어파인 변환 - warpAffine()](#3-어파인-변환---warpaffine)
4. [원근 변환 - warpPerspective()](#4-원근-변환---warpperspective)
5. [문서 교정 예제](#5-문서-교정-예제)
6. [연습 문제](#6-연습-문제)
7. [다음 단계](#7-다음-단계)
8. [참고 자료](#8-참고-자료)

---

## 1. 이미지 크기 조절 - resize()

### 기본 사용법

```python
import cv2

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# 방법 1: 직접 크기 지정 (width, height 순서!)
resized = cv2.resize(img, (640, 480))

# 방법 2: 비율로 지정
resized = cv2.resize(img, None, fx=0.5, fy=0.5)  # 50%로 축소

# 방법 3: 한 쪽 기준으로 비율 유지
new_width = 800
ratio = new_width / w
new_height = int(h * ratio)
resized = cv2.resize(img, (new_width, new_height))
```

### 보간법 (Interpolation Methods)

```
┌─────────────────────────────────────────────────────────────────┐
│                       보간법 비교                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   보간법                   특징                   사용 상황      │
│   ───────────────────────────────────────────────────────────   │
│   INTER_NEAREST           최근접 이웃            빠름, 저품질   │
│   (최근접 보간)            블록 현상 발생         실시간 처리    │
│                                                                │
│   INTER_LINEAR            선형 보간 (기본값)     균형 잡힌 선택 │
│   (양선형 보간)            부드러운 결과          일반적 리사이즈│
│                                                                │
│   INTER_AREA              영역 보간              축소에 최적   │
│   (영역 기반)              모아레 현상 방지       다운샘플링     │
│                                                                │
│   INTER_CUBIC             3차 보간               확대에 좋음   │
│   (바이큐빅)               부드럽고 선명          품질 중시      │
│                                                                │
│   INTER_LANCZOS4          란초스 보간            최고 품질     │
│   (8x8 이웃)               가장 선명              속도 느림     │
│                                                                │
│   권장:                                                        │
│   - 축소: INTER_AREA                                           │
│   - 확대: INTER_CUBIC 또는 INTER_LANCZOS4                      │
│   - 실시간: INTER_LINEAR 또는 INTER_NEAREST                    │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 보간법 비교 예제

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 먼저 축소 후 확대하여 차이 비교
small = cv2.resize(img, None, fx=0.1, fy=0.1)  # 10%로 축소

interpolations = [
    ('NEAREST', cv2.INTER_NEAREST),
    ('LINEAR', cv2.INTER_LINEAR),
    ('AREA', cv2.INTER_AREA),
    ('CUBIC', cv2.INTER_CUBIC),
    ('LANCZOS4', cv2.INTER_LANCZOS4),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].imshow(img)
axes[0].set_title('Original')

for ax, (name, interp) in zip(axes[1:], interpolations):
    enlarged = cv2.resize(small, img.shape[:2][::-1], interpolation=interp)
    ax.imshow(enlarged)
    ax.set_title(f'{name}')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
```

### 비율 유지 리사이즈 함수

```python
import cv2

def resize_with_aspect_ratio(img, width=None, height=None, inter=cv2.INTER_AREA):
    """비율을 유지하면서 리사이즈"""
    h, w = img.shape[:2]

    if width is None and height is None:
        return img

    if width is None:
        ratio = height / h
        new_size = (int(w * ratio), height)
    else:
        ratio = width / w
        new_size = (width, int(h * ratio))

    return cv2.resize(img, new_size, interpolation=inter)


def resize_to_fit(img, max_width, max_height, inter=cv2.INTER_AREA):
    """최대 크기 내에 맞추면서 비율 유지"""
    h, w = img.shape[:2]

    ratio_w = max_width / w
    ratio_h = max_height / h
    ratio = min(ratio_w, ratio_h)

    if ratio >= 1:  # 이미 작으면 그대로
        return img

    new_size = (int(w * ratio), int(h * ratio))
    return cv2.resize(img, new_size, interpolation=inter)


# 사용 예
img = cv2.imread('large_image.jpg')
img_fit = resize_to_fit(img, 800, 600)
img_width = resize_with_aspect_ratio(img, width=640)
```

---

## 2. 뒤집기와 회전 - flip(), rotate()

### cv2.flip()

```
┌─────────────────────────────────────────────────────────────────┐
│                        flip() 동작                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   flipCode = 1 (수평)     flipCode = 0 (수직)    flipCode = -1  │
│                                                                 │
│   원본     결과            원본     결과          원본    결과   │
│   ┌───┐   ┌───┐           ┌───┐   ┌───┐         ┌───┐  ┌───┐  │
│   │1 2│   │2 1│           │1 2│   │3 4│         │1 2│  │4 3│  │
│   │3 4│   │4 3│           │3 4│   │1 2│         │3 4│  │2 1│  │
│   └───┘   └───┘           └───┘   └───┘         └───┘  └───┘  │
│                                                                 │
│   좌우 반전              상하 반전              둘 다 반전      │
│   (거울 효과)            (물에 비친 효과)       (180도 회전)    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```python
import cv2

img = cv2.imread('image.jpg')

# 수평 뒤집기 (좌우 반전)
flipped_h = cv2.flip(img, 1)

# 수직 뒤집기 (상하 반전)
flipped_v = cv2.flip(img, 0)

# 양방향 뒤집기 (180도 회전과 동일)
flipped_both = cv2.flip(img, -1)

# NumPy로도 가능
import numpy as np
flipped_h_np = img[:, ::-1]      # 수평
flipped_v_np = img[::-1, :]      # 수직
flipped_both_np = img[::-1, ::-1]  # 양방향
```

### cv2.rotate()

```python
import cv2

img = cv2.imread('image.jpg')

# 90도 시계방향
rotated_90_cw = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# 90도 반시계방향
rotated_90_ccw = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# 180도
rotated_180 = cv2.rotate(img, cv2.ROTATE_180)

# 이미지 크기 변화 확인
print(f"원본: {img.shape}")           # (H, W, C)
print(f"90도: {rotated_90_cw.shape}") # (W, H, C) - 가로세로 교환
print(f"180도: {rotated_180.shape}")  # (H, W, C) - 동일
```

### 임의 각도 회전

```python
import cv2

def rotate_image(img, angle, center=None, scale=1.0):
    """임의 각도로 이미지 회전"""
    h, w = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    # 회전 행렬 생성
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # 회전 적용
    rotated = cv2.warpAffine(img, M, (w, h))

    return rotated


def rotate_image_full(img, angle):
    """이미지가 잘리지 않도록 회전 (캔버스 확장)"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # 회전 행렬
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 회전 후 새 경계 계산
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # 이동량 조정
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(img, M, (new_w, new_h))

    return rotated


# 사용 예
img = cv2.imread('image.jpg')
rotated_30 = rotate_image(img, 30)       # 30도 회전 (일부 잘림)
rotated_45_full = rotate_image_full(img, 45)  # 45도 회전 (전체 보존)
```

---

## 3. 어파인 변환 - warpAffine()

### 어파인 변환이란?

```
┌─────────────────────────────────────────────────────────────────┐
│                      어파인 변환                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   어파인 변환은 직선을 직선으로, 평행선을 평행선으로 유지하는 변환    │
│                                                                 │
│   포함되는 변환:                                                 │
│   - 이동 (Translation)                                          │
│   - 회전 (Rotation)                                             │
│   - 스케일 (Scale)                                              │
│   - 전단 (Shear)                                                │
│                                                                 │
│   변환 행렬 (2x3):                                               │
│   ┌         ┐   ┌                    ┐                         │
│   │ a  b  tx│   │ scale*cos  -sin  tx│                         │
│   │ c  d  ty│ = │ sin   scale*cos  ty│                         │
│   └         ┘   └                    ┘                         │
│                                                                 │
│   [x']   [a b tx]   [x]                                         │
│   [y'] = [c d ty] × [y]                                         │
│                     [1]                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 이동 (Translation)

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# 이동 행렬: x방향 100, y방향 50 이동
tx, ty = 100, 50
M = np.float32([
    [1, 0, tx],
    [0, 1, ty]
])

translated = cv2.warpAffine(img, M, (w, h))

cv2.imshow('Original', img)
cv2.imshow('Translated', translated)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 회전 + 스케일

```python
import cv2

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# getRotationMatrix2D(center, angle, scale)
center = (w // 2, h // 2)
angle = 45  # 반시계방향 45도
scale = 0.7  # 70% 크기

M = cv2.getRotationMatrix2D(center, angle, scale)

rotated = cv2.warpAffine(img, M, (w, h))
```

### 전단 변환 (Shear)

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# 수평 전단
shear_x = 0.3
M_shear_x = np.float32([
    [1, shear_x, 0],
    [0, 1, 0]
])
sheared_x = cv2.warpAffine(img, M_shear_x, (int(w + h * shear_x), h))

# 수직 전단
shear_y = 0.3
M_shear_y = np.float32([
    [1, 0, 0],
    [shear_y, 1, 0]
])
sheared_y = cv2.warpAffine(img, M_shear_y, (w, int(h + w * shear_y)))
```

### 3점을 이용한 어파인 변환

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# 원본의 3점
src_pts = np.float32([
    [0, 0],      # 좌상단
    [w-1, 0],    # 우상단
    [0, h-1]     # 좌하단
])

# 변환 후 3점
dst_pts = np.float32([
    [50, 50],    # 좌상단
    [w-50, 30],  # 우상단
    [30, h-50]   # 좌하단
])

# 어파인 변환 행렬 계산
M = cv2.getAffineTransform(src_pts, dst_pts)

# 변환 적용
result = cv2.warpAffine(img, M, (w, h))

# 점 표시
for pt in src_pts.astype(int):
    cv2.circle(img, tuple(pt), 5, (0, 0, 255), -1)

for pt in dst_pts.astype(int):
    cv2.circle(result, tuple(pt), 5, (0, 255, 0), -1)
```

---

## 4. 원근 변환 - warpPerspective()

### 원근 변환이란?

```
┌─────────────────────────────────────────────────────────────────┐
│                       원근 변환                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   원근 변환은 사다리꼴을 직사각형으로 (또는 그 반대로) 변환         │
│   3D 공간에서 촬영된 이미지를 정면에서 본 것처럼 변환               │
│                                                                 │
│   실제 활용:                                                     │
│   - 문서 스캔 (기울어진 문서 → 정면)                             │
│   - 차선 검출 (Bird's eye view)                                 │
│   - QR 코드 인식                                                │
│   - 이미지 교정                                                  │
│                                                                 │
│   변환 행렬 (3x3):                                               │
│   ┌             ┐                                               │
│   │ h11 h12 h13 │                                               │
│   │ h21 h22 h23 │                                               │
│   │ h31 h32 h33 │                                               │
│   └             ┘                                               │
│                                                                 │
│   원본 (사다리꼴)          결과 (직사각형)                        │
│   ┌─────────────┐         ┌─────────────────┐                   │
│   │ ┌─────────┐ │         │ ┌─────────────┐ │                   │
│   │ │         │ │   ───▶  │ │             │ │                   │
│   │ │ 문서    │ │         │ │    문서     │ │                   │
│   │ │         │ │         │ │             │ │                   │
│   │ └───────────┘│         │ └─────────────┘ │                   │
│   └─────────────┘         └─────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4점을 이용한 원근 변환

```python
import cv2
import numpy as np

img = cv2.imread('tilted_document.jpg')
h, w = img.shape[:2]

# 원본의 4점 (문서의 네 꼭짓점)
src_pts = np.float32([
    [100, 50],    # 좌상단
    [500, 80],    # 우상단
    [550, 400],   # 우하단
    [50, 380]     # 좌하단
])

# 변환 후 4점 (정면 직사각형)
dst_pts = np.float32([
    [0, 0],
    [500, 0],
    [500, 400],
    [0, 400]
])

# 원근 변환 행렬 계산
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# 변환 적용
result = cv2.warpPerspective(img, M, (500, 400))

# 점 표시
img_with_pts = img.copy()
for i, pt in enumerate(src_pts.astype(int)):
    cv2.circle(img_with_pts, tuple(pt), 10, (0, 0, 255), -1)
    cv2.putText(img_with_pts, str(i+1), tuple(pt),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow('Original with points', img_with_pts)
cv2.imshow('Warped', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Bird's Eye View (조감도)

```python
import cv2
import numpy as np

def get_birds_eye_view(img, src_pts, width, height):
    """
    원근 변환으로 조감도(위에서 본 시점) 생성

    Parameters:
    - img: 입력 이미지
    - src_pts: 원본의 4점 (좌상, 우상, 우하, 좌하)
    - width, height: 출력 이미지 크기
    """
    dst_pts = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))

    return warped, M


# 차선 검출용 예시
img = cv2.imread('road.jpg')
h, w = img.shape[:2]

# 도로 영역 4점 (사다리꼴)
road_pts = np.float32([
    [w * 0.4, h * 0.6],   # 좌상단
    [w * 0.6, h * 0.6],   # 우상단
    [w * 0.9, h * 0.95],  # 우하단
    [w * 0.1, h * 0.95]   # 좌하단
])

birds_eye, M = get_birds_eye_view(img, road_pts, 400, 600)
```

---

## 5. 문서 교정 예제

### 자동 문서 스캔 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                   문서 스캔 파이프라인                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   입력 이미지                                                    │
│       │                                                         │
│       ▼                                                         │
│   전처리 (그레이스케일, 블러, 엣지)                               │
│       │                                                         │
│       ▼                                                         │
│   윤곽선 검출 (findContours)                                     │
│       │                                                         │
│       ▼                                                         │
│   사각형 검출 (approxPolyDP로 4점 근사)                          │
│       │                                                         │
│       ▼                                                         │
│   꼭짓점 정렬 (좌상, 우상, 우하, 좌하)                            │
│       │                                                         │
│       ▼                                                         │
│   원근 변환 (warpPerspective)                                    │
│       │                                                         │
│       ▼                                                         │
│   후처리 (이진화, 선명화)                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 구현 코드

```python
import cv2
import numpy as np

def order_points(pts):
    """4점을 좌상, 우상, 우하, 좌하 순으로 정렬"""
    rect = np.zeros((4, 2), dtype=np.float32)

    # 합이 가장 작은 점 = 좌상단
    # 합이 가장 큰 점 = 우하단
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 차가 가장 작은 점 = 우상단
    # 차가 가장 큰 점 = 좌하단
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def four_point_transform(img, pts):
    """4점을 이용한 원근 변환"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 새 이미지의 너비 계산
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    # 새 이미지의 높이 계산
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    # 목적지 점
    dst = np.float32([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ])

    # 원근 변환
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    return warped


def find_document(img):
    """이미지에서 문서 영역을 자동 검출"""
    # 전처리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # 윤곽선 검출
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    doc_contour = None
    for contour in contours:
        # 윤곽선 근사
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # 4점이면 문서로 간주
        if len(approx) == 4:
            doc_contour = approx
            break

    return doc_contour


def scan_document(img):
    """문서 스캔 메인 함수"""
    # 원본 크기 저장
    orig = img.copy()
    ratio = img.shape[0] / 500.0

    # 리사이즈 (처리 속도 향상)
    img = cv2.resize(img, (int(img.shape[1] / ratio), 500))

    # 문서 검출
    doc_contour = find_document(img)

    if doc_contour is None:
        print("문서를 찾을 수 없습니다.")
        return None

    # 원본 크기에 맞게 좌표 조정
    doc_contour = doc_contour.reshape(4, 2) * ratio

    # 원근 변환
    warped = four_point_transform(orig, doc_contour)

    # 후처리 (선택적)
    # 그레이스케일 + 적응형 이진화
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_binary = cv2.adaptiveThreshold(
        warped_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 10
    )

    return warped, warped_binary


# 사용 예
img = cv2.imread('document_photo.jpg')
result_color, result_binary = scan_document(img)

if result_color is not None:
    cv2.imshow('Original', img)
    cv2.imshow('Scanned (Color)', result_color)
    cv2.imshow('Scanned (Binary)', result_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### 수동 4점 선택 (마우스 클릭)

```python
import cv2
import numpy as np

points = []

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Select 4 corners', param)

            if len(points) == 4:
                print("4점 선택 완료! 's' 키를 눌러 변환하세요.")


def manual_perspective_transform(img):
    """마우스로 4점을 선택하여 원근 변환"""
    global points
    points = []

    img_display = img.copy()
    cv2.imshow('Select 4 corners', img_display)
    cv2.setMouseCallback('Select 4 corners', click_event, img_display)

    print("문서의 4개 꼭짓점을 시계방향으로 클릭하세요 (좌상단부터)")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and len(points) == 4:
            break
        elif key == ord('r'):  # 리셋
            points = []
            img_display = img.copy()
            cv2.imshow('Select 4 corners', img_display)
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()

    pts = np.array(points, dtype=np.float32)
    result = four_point_transform(img, pts)

    return result


# 사용 예
img = cv2.imread('document.jpg')
result = manual_perspective_transform(img)

if result is not None:
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

---

## 6. 연습 문제

### 연습 1: 배치 리사이즈

폴더 내의 모든 이미지를 가로 800px로 리사이즈하고 (비율 유지), 품질 90%의 JPEG로 저장하는 스크립트를 작성하세요.

```python
# 힌트
import os
import glob

def batch_resize(input_folder, output_folder, max_width=800):
    # os.listdir 또는 glob.glob 사용
    pass
```

### 연습 2: 이미지 회전 애니메이션

이미지를 0도부터 360도까지 5도씩 회전하면서 애니메이션으로 보여주는 프로그램을 작성하세요. 이미지가 잘리지 않도록 캔버스를 확장하세요.

### 연습 3: 신분증 스캐너

다음 기능을 가진 신분증 스캐너를 구현하세요:
1. 마우스로 4점 선택
2. 원근 변환으로 정면 뷰 생성
3. 표준 신분증 크기(85.6mm x 54mm) 비율로 출력

### 연습 4: 이미지 모자이크

여러 이미지를 받아서 N x M 그리드로 배치하는 함수를 작성하세요. 각 이미지는 동일한 크기로 리사이즈되어야 합니다.

```python
def create_mosaic(images, rows, cols, cell_size=(200, 200)):
    """이미지들을 rows x cols 그리드로 배치"""
    pass
```

### 연습 5: AR 카드 효과

이미지에서 직사각형 카드를 검출하고, 그 위에 다른 이미지를 오버레이하는 간단한 AR 효과를 구현하세요.

```python
# 힌트: 원근 변환의 역방향 사용
# 1. 카드 영역 검출
# 2. 오버레이할 이미지를 카드 영역에 맞게 변환
# 3. 원본에 합성
```

---

## 7. 다음 단계

[05_Image_Filtering.md](./05_Image_Filtering.md)에서 블러, 샤프닝, 커스텀 필터 등 이미지 필터링 기법을 학습합니다!

**다음에 배울 내용**:
- 커널과 컨볼루션 개념
- 블러 필터 (평균, 가우시안, 중앙값, 양방향)
- 샤프닝 필터
- 커스텀 필터 (filter2D)

---

## 8. 참고 자료

### 공식 문서

- [resize() 문서](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d)
- [warpAffine() 문서](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983)
- [warpPerspective() 문서](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87)

### 관련 학습 자료

| 폴더 | 관련 내용 |
|------|----------|
| [03_Color_Spaces.md](./03_Color_Spaces.md) | 색상 변환, 엣지 검출 전처리 |
| [09_Contours.md](./09_Contours.md) | 문서 영역 검출에 활용 |

### 추가 참고

- [PyImageSearch - 4-point Transform](https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/)
- [OpenCV 보간법 가이드](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121)

