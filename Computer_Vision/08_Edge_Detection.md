# 엣지 검출 (Edge Detection)

## 개요

엣지(Edge)는 이미지에서 밝기가 급격하게 변하는 영역으로, 객체의 경계나 구조를 나타냅니다. 이 레슨에서는 이미지 그래디언트 개념과 Sobel, Scharr, Laplacian, Canny 등 다양한 엣지 검출 기법을 학습합니다.

---

## 목차

1. [이미지 그래디언트 개념](#1-이미지-그래디언트-개념)
2. [Sobel 연산자](#2-sobel-연산자)
3. [Scharr 연산자](#3-scharr-연산자)
4. [Laplacian 연산자](#4-laplacian-연산자)
5. [Canny 엣지 검출](#5-canny-엣지-검출)
6. [그래디언트 크기와 방향](#6-그래디언트-크기와-방향)
7. [연습 문제](#7-연습-문제)

---

## 1. 이미지 그래디언트 개념

### 그래디언트란?

```
그래디언트(Gradient): 이미지 밝기의 변화율

수학적 정의:
∇f = (∂f/∂x, ∂f/∂y)

- ∂f/∂x: x 방향(수평) 밝기 변화율
- ∂f/∂y: y 방향(수직) 밝기 변화율

그래디언트 크기 (Magnitude):
|∇f| = √((∂f/∂x)² + (∂f/∂y)²)

그래디언트 방향 (Direction):
θ = arctan(∂f/∂y / ∂f/∂x)
```

### 엣지의 종류

```
1. 스텝 엣지 (Step Edge)
   밝기 ──┐
         │
         └── 밝기
   → 이상적인 엣지, 급격한 변화

2. 램프 엣지 (Ramp Edge)
   밝기 ──╲
          ╲
           ╲── 밝기
   → 점진적인 변화, 흐릿한 경계

3. 지붕 엣지 (Roof Edge)
   밝기 ──╱╲
        ╱  ╲
       ╱    ╲── 밝기
   → 선(line) 구조

4. 리지 엣지 (Ridge Edge)
       ╱╲
      ╱  ╲
   ──╱    ╲──
   → 얇은 선 구조
```

### 엣지 검출 파이프라인

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   입력      │     │   노이즈    │     │  그래디언트 │     │    엣지     │
│  이미지     │ ──▶ │   제거      │ ──▶ │    계산     │ ──▶ │   추출      │
│             │     │ (Gaussian)  │     │ (Sobel 등)  │     │ (임계값)    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## 2. Sobel 연산자

### 개념

```
Sobel 연산자: 1차 미분 기반 엣지 검출
→ x, y 방향의 그래디언트를 각각 계산

3x3 Sobel 커널:

Gx (수평 엣지 검출):        Gy (수직 엣지 검출):
┌────┬────┬────┐           ┌────┬────┬────┐
│ -1 │  0 │ +1 │           │ -1 │ -2 │ -1 │
├────┼────┼────┤           ├────┼────┼────┤
│ -2 │  0 │ +2 │           │  0 │  0 │  0 │
├────┼────┼────┤           ├────┼────┼────┤
│ -1 │  0 │ +1 │           │ +1 │ +2 │ +1 │
└────┴────┴────┘           └────┴────┴────┘

→ Gx: 세로 엣지 검출 (좌우 밝기 차이)
→ Gy: 가로 엣지 검출 (상하 밝기 차이)
```

### cv2.Sobel() 함수

```python
cv2.Sobel(src, ddepth, dx, dy, ksize=3, scale=1, delta=0)
```

| 파라미터 | 설명 |
|----------|------|
| src | 입력 이미지 |
| ddepth | 출력 이미지 깊이 (cv2.CV_64F 권장) |
| dx | x 방향 미분 차수 (0 또는 1) |
| dy | y 방향 미분 차수 (0 또는 1) |
| ksize | 커널 크기 (1, 3, 5, 7) |
| scale | 스케일 팩터 |
| delta | 결과에 더할 값 |

### 기본 사용법

```python
import cv2
import numpy as np

# 이미지 읽기
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel 연산
# ddepth를 CV_64F로 설정하여 음수 값도 처리
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # x 방향
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # y 방향

# 절대값 변환 후 8비트로 변환
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

# x, y 그래디언트 합성
sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# 결과 표시
cv2.imshow('Original', img)
cv2.imshow('Sobel X', sobel_x)
cv2.imshow('Sobel Y', sobel_y)
cv2.imshow('Sobel Combined', sobel_combined)
cv2.waitKey(0)
```

### 그래디언트 크기 계산

```python
import cv2
import numpy as np

def sobel_magnitude(image):
    """Sobel 그래디언트 크기 계산"""
    # 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Sobel 연산 (float64로 계산)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # 그래디언트 크기: sqrt(Gx² + Gy²)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # 0-255 범위로 정규화
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    return magnitude

# 사용 예
img = cv2.imread('image.jpg')
edges = sobel_magnitude(img)
cv2.imshow('Sobel Magnitude', edges)
cv2.waitKey(0)
```

### 커널 크기에 따른 차이

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_sobel_ksize(image_path):
    """Sobel 커널 크기 비교"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    ksizes = [1, 3, 5, 7]

    for ax, ksize in zip(axes.flatten(), ksizes):
        # ksize=1일 때는 3x1 또는 1x3 필터 사용
        if ksize == 1:
            sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
            sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)
        else:
            sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
            sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)

        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

        ax.imshow(magnitude, cmap='gray')
        ax.set_title(f'Sobel ksize={ksize}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# ksize 비교:
# - ksize=1: 가장 민감, 노이즈에 취약
# - ksize=3: 표준, 균형 잡힌 결과
# - ksize=5, 7: 더 부드러운 엣지, 노이즈에 강함
```

---

## 3. Scharr 연산자

### 개념

```
Scharr 연산자: Sobel보다 더 정확한 3x3 커널
→ 회전 대칭성이 더 좋음

Scharr 커널:

Gx:                         Gy:
┌────┬────┬────┐           ┌────┬────┬────┐
│ -3 │  0 │ +3 │           │ -3 │-10 │ -3 │
├────┼────┼────┤           ├────┼────┼────┤
│-10 │  0 │+10 │           │  0 │  0 │  0 │
├────┼────┼────┤           ├────┼────┼────┤
│ -3 │  0 │ +3 │           │ +3 │+10 │ +3 │
└────┴────┴────┘           └────┴────┴────┘

Sobel vs Scharr:
- Sobel: [-1, 0, 1] × [-1, -2, -1]ᵀ
- Scharr: [-3, 0, 3] × [-3, -10, -3]ᵀ
→ Scharr가 대각선 방향에서 더 정확
```

### cv2.Scharr() 함수

```python
cv2.Scharr(src, ddepth, dx, dy, scale=1, delta=0)
```

```python
import cv2
import numpy as np

def compare_sobel_scharr(image):
    """Sobel과 Scharr 비교"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sobel (ksize=3)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # Scharr (3x3 고정)
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    scharr_mag = np.sqrt(scharr_x**2 + scharr_y**2)

    # 정규화
    sobel_mag = np.clip(sobel_mag, 0, 255).astype(np.uint8)
    scharr_mag = np.clip(scharr_mag, 0, 255).astype(np.uint8)

    return sobel_mag, scharr_mag

# Scharr 사용 예
img = cv2.imread('image.jpg')
sobel, scharr = compare_sobel_scharr(img)

cv2.imshow('Sobel', sobel)
cv2.imshow('Scharr', scharr)
cv2.waitKey(0)
```

### Sobel에서 Scharr 사용하기

```python
# cv2.Sobel()에서 ksize=-1 또는 ksize=cv2.FILTER_SCHARR 사용
scharr_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=-1)  # Scharr 커널 사용
scharr_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=-1)

# 위 코드는 아래와 동일
scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
```

---

## 4. Laplacian 연산자

### 개념

```
Laplacian 연산자: 2차 미분 기반 엣지 검출
→ 밝기가 급격히 변하는 지점에서 0을 교차

수학적 정의:
∇²f = ∂²f/∂x² + ∂²f/∂y²

Laplacian 커널:

4-연결성:                   8-연결성:
┌────┬────┬────┐           ┌────┬────┬────┐
│  0 │  1 │  0 │           │  1 │  1 │  1 │
├────┼────┼────┤           ├────┼────┼────┤
│  1 │ -4 │  1 │           │  1 │ -8 │  1 │
├────┼────┼────┤           ├────┼────┼────┤
│  0 │  1 │  0 │           │  1 │  1 │  1 │
└────┴────┴────┘           └────┴────┴────┘

특징:
- 방향에 무관하게 엣지 검출
- 노이즈에 매우 민감 (2차 미분)
- Zero-crossing 지점이 엣지
```

### 1차 미분 vs 2차 미분

```
원본 신호 (스텝 엣지):
       ────────────┐
                   │
                   └────────────

1차 미분 (Sobel):
                  ╱╲
                 ╱  ╲
       ─────────╱    ╲─────────
       → 피크 지점이 엣지

2차 미분 (Laplacian):
            ╱╲
           ╱  ╲
       ───╱    ╲───
              ╱  ╲
             ╱    ╲
       → Zero-crossing 지점이 엣지
```

### cv2.Laplacian() 함수

```python
cv2.Laplacian(src, ddepth, ksize=1, scale=1, delta=0)
```

| 파라미터 | 설명 |
|----------|------|
| src | 입력 이미지 |
| ddepth | 출력 이미지 깊이 |
| ksize | 커널 크기 (1, 3, 5, 7) |
| scale | 스케일 팩터 |
| delta | 결과에 더할 값 |

### 기본 사용법

```python
import cv2
import numpy as np

def laplacian_edge(image):
    """Laplacian 엣지 검출"""
    # 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 노이즈 제거 (Laplacian은 노이즈에 민감)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Laplacian 연산
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)

    # 절대값 변환
    laplacian = cv2.convertScaleAbs(laplacian)

    return laplacian

# 사용 예
img = cv2.imread('image.jpg')
edges = laplacian_edge(img)
cv2.imshow('Laplacian', edges)
cv2.waitKey(0)
```

### LoG (Laplacian of Gaussian)

```python
import cv2
import numpy as np

def log_edge_detection(image, sigma=1.0):
    """
    LoG (Laplacian of Gaussian) 엣지 검출
    1. Gaussian 블러로 노이즈 제거
    2. Laplacian으로 엣지 검출
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian 블러 (sigma에 따른 커널 크기)
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1

    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)

    # Laplacian
    log = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)

    # 절대값
    log = cv2.convertScaleAbs(log)

    return log

# LoG 사용
img = cv2.imread('image.jpg')
edges = log_edge_detection(img, sigma=1.5)
cv2.imshow('LoG', edges)
cv2.waitKey(0)
```

---

## 5. Canny 엣지 검출

### 개념

```
Canny 엣지 검출: 다단계 엣지 검출 알고리즘
→ 가장 널리 사용되는 엣지 검출 방법

Canny의 3가지 목표:
1. 낮은 오류율: 실제 엣지만 검출
2. 정확한 위치: 엣지가 정확한 위치에
3. 단일 응답: 하나의 엣지에 하나의 선

4단계 처리:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Gaussian   │     │   Sobel     │     │   비최대    │     │  이력       │
│   Blur      │ ──▶ │  Gradient   │ ──▶ │   억제      │ ──▶ │  임계값     │
│             │     │             │     │   (NMS)     │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### Canny 알고리즘 상세

```
단계 1: 노이즈 제거 (Gaussian Blur)
- 5x5 Gaussian 필터 적용
- 고주파 노이즈 제거

단계 2: 그래디언트 계산
- Sobel 연산으로 Gx, Gy 계산
- 크기: G = √(Gx² + Gy²)
- 방향: θ = arctan(Gy/Gx)

단계 3: 비최대 억제 (Non-Maximum Suppression)
┌─────────────────────────────────────┐
│  그래디언트 방향을 따라 최댓값만 유지  │
│  → 엣지를 1픽셀 두께로 얇게 만듦      │
└─────────────────────────────────────┘

방향 양자화 (4방향):
        90°
         │
  135° ──┼── 45°
         │
        0° (180°)

예시:
방향 θ = 45°일 때, 대각선 방향으로 비교
┌───┬───┬───┐
│   │ q │   │
├───┼───┼───┤
│   │ p │   │  p가 q, r보다 크면 유지
├───┼───┼───┤
│   │ r │   │
└───┴───┴───┘

단계 4: 이력 임계값 (Hysteresis Thresholding)
┌─────────────────────────────────────┐
│  high_threshold: 강한 엣지           │
│  low_threshold: 약한 엣지            │
│                                     │
│  강한 엣지: 무조건 포함              │
│  약한 엣지: 강한 엣지와 연결되면 포함 │
│  나머지: 제거                        │
└─────────────────────────────────────┘

예시:
high = 100, low = 50

픽셀 값 120 → 강한 엣지 (포함)
픽셀 값 70  → 약한 엣지 (연결 확인)
픽셀 값 30  → 제거
```

### cv2.Canny() 함수

```python
cv2.Canny(image, threshold1, threshold2, apertureSize=3, L2gradient=False)
```

| 파라미터 | 설명 |
|----------|------|
| image | 입력 이미지 (그레이스케일) |
| threshold1 | 낮은 임계값 (low) |
| threshold2 | 높은 임계값 (high) |
| apertureSize | Sobel 커널 크기 (3, 5, 7) |
| L2gradient | True: L2 norm, False: L1 norm |

### 기본 사용법

```python
import cv2

def canny_edge(image, low=50, high=150):
    """Canny 엣지 검출"""
    # 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 노이즈 제거 (선택적 - Canny 내부에서도 수행)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Canny 엣지 검출
    edges = cv2.Canny(blurred, low, high)

    return edges

# 사용 예
img = cv2.imread('image.jpg')
edges = canny_edge(img, 50, 150)

cv2.imshow('Original', img)
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)
```

### 임계값 튜닝

```python
import cv2
import numpy as np

def canny_with_trackbar(image_path):
    """트랙바로 Canny 임계값 조절"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    cv2.namedWindow('Canny')

    def nothing(x):
        pass

    cv2.createTrackbar('Low', 'Canny', 50, 255, nothing)
    cv2.createTrackbar('High', 'Canny', 150, 255, nothing)

    while True:
        low = cv2.getTrackbarPos('Low', 'Canny')
        high = cv2.getTrackbarPos('High', 'Canny')

        # low가 high보다 크지 않도록
        if low >= high:
            low = high - 1

        edges = cv2.Canny(blurred, low, high)

        cv2.imshow('Canny', edges)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cv2.destroyAllWindows()

# 실행
canny_with_trackbar('image.jpg')
```

### 자동 임계값 설정

```python
import cv2
import numpy as np

def auto_canny(image, sigma=0.33):
    """
    자동 임계값 Canny
    중간값 기준으로 low, high 계산
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # 중간값 계산
    median = np.median(blurred)

    # 임계값 계산
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))

    print(f"Auto threshold: low={low}, high={high}")

    edges = cv2.Canny(blurred, low, high)

    return edges

# 사용 예
img = cv2.imread('image.jpg')
edges = auto_canny(img)
cv2.imshow('Auto Canny', edges)
cv2.waitKey(0)
```

### 컬러 이미지에서 Canny

```python
import cv2
import numpy as np

def canny_color(image, low=50, high=150):
    """
    컬러 이미지에서 Canny 엣지 검출
    각 채널별로 엣지 검출 후 합성
    """
    # 방법 1: 그레이스케일 변환 후 처리
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges_gray = cv2.Canny(gray, low, high)

    # 방법 2: 각 채널별 처리 후 합성
    b, g, r = cv2.split(image)
    edges_b = cv2.Canny(b, low, high)
    edges_g = cv2.Canny(g, low, high)
    edges_r = cv2.Canny(r, low, high)

    # OR 연산으로 합성
    edges_color = cv2.bitwise_or(edges_b, edges_g)
    edges_color = cv2.bitwise_or(edges_color, edges_r)

    return edges_gray, edges_color

# 사용 예
img = cv2.imread('image.jpg')
edges_gray, edges_color = canny_color(img)

cv2.imshow('Edges (Gray)', edges_gray)
cv2.imshow('Edges (Color)', edges_color)
cv2.waitKey(0)
```

---

## 6. 그래디언트 크기와 방향

### 그래디언트 크기 계산

```python
import cv2
import numpy as np

def gradient_magnitude_direction(image):
    """그래디언트 크기와 방향 계산"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Sobel 그래디언트
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 크기 (Magnitude)
    magnitude = np.sqrt(gx**2 + gy**2)

    # 방향 (Direction) - 라디안
    direction = np.arctan2(gy, gx)

    # 방향을 도(degree)로 변환 (0-180)
    direction_deg = np.degrees(direction) % 180

    return magnitude, direction_deg

# 사용 예
img = cv2.imread('image.jpg')
mag, dir = gradient_magnitude_direction(img)

# 정규화하여 표시
mag_display = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
dir_display = (dir / 180 * 255).astype(np.uint8)

cv2.imshow('Magnitude', mag_display)
cv2.imshow('Direction', dir_display)
cv2.waitKey(0)
```

### 그래디언트 방향 시각화

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_gradient_direction(image, step=20):
    """
    그래디언트 방향을 화살표로 시각화
    step: 샘플링 간격
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(gx**2 + gy**2)

    # 화살표 그리기
    result = image.copy()
    h, w = gray.shape

    for y in range(step, h - step, step):
        for x in range(step, w - step, step):
            if magnitude[y, x] > 50:  # 일정 크기 이상만 표시
                # 방향 벡터 정규화
                dx = gx[y, x]
                dy = gy[y, x]
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    dx = int(dx / length * 10)
                    dy = int(dy / length * 10)

                    cv2.arrowedLine(
                        result,
                        (x, y),
                        (x + dx, y + dy),
                        (0, 255, 0),
                        1,
                        tipLength=0.3
                    )

    return result

# 사용 예
img = cv2.imread('image.jpg')
vis = visualize_gradient_direction(img, step=15)
cv2.imshow('Gradient Direction', vis)
cv2.waitKey(0)
```

### 엣지 검출 알고리즘 비교

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_edge_detectors(image_path):
    """다양한 엣지 검출 알고리즘 비교"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # 1. Sobel
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = np.clip(sobel, 0, 255).astype(np.uint8)

    # 2. Scharr
    scharr_x = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
    scharr = np.sqrt(scharr_x**2 + scharr_y**2)
    scharr = np.clip(scharr, 0, 255).astype(np.uint8)

    # 3. Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    laplacian = cv2.convertScaleAbs(laplacian)

    # 4. Canny
    canny = cv2.Canny(blurred, 50, 150)

    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')

    axes[0, 1].imshow(sobel, cmap='gray')
    axes[0, 1].set_title('Sobel')

    axes[0, 2].imshow(scharr, cmap='gray')
    axes[0, 2].set_title('Scharr')

    axes[1, 0].imshow(laplacian, cmap='gray')
    axes[1, 0].set_title('Laplacian')

    axes[1, 1].imshow(canny, cmap='gray')
    axes[1, 1].set_title('Canny')

    axes[1, 2].axis('off')

    for ax in axes.flatten():
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# 비교 실행
compare_edge_detectors('image.jpg')
```

---

## 7. 연습 문제

### 문제 1: 적응형 Canny 구현

이미지의 밝기 분포에 따라 자동으로 임계값을 조절하는 Canny 함수를 구현하세요.

<details>
<summary>힌트</summary>

이미지의 중간값(median)을 기준으로 낮은 임계값과 높은 임계값을 계산합니다.

</details>

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def adaptive_canny(image, sigma=0.33):
    """
    적응형 Canny 엣지 검출
    이미지 밝기의 중간값을 기준으로 임계값 자동 설정
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 중간값 계산
    median = np.median(blurred)

    # 임계값 계산 (sigma로 범위 조절)
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))

    edges = cv2.Canny(blurred, low, high)

    return edges, low, high

# 테스트
img = cv2.imread('image.jpg')
edges, low, high = adaptive_canny(img)
print(f"Adaptive thresholds: low={low}, high={high}")
cv2.imshow('Adaptive Canny', edges)
cv2.waitKey(0)
```

</details>

### 문제 2: 방향별 엣지 분리

수평 엣지와 수직 엣지를 분리하여 표시하는 함수를 구현하세요.

<details>
<summary>힌트</summary>

그래디언트 방향을 계산하고, 각도에 따라 수평(0도 근처)과 수직(90도 근처)을 분류합니다.

</details>

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def separate_edges_by_direction(image, angle_threshold=30):
    """
    수평/수직 엣지 분리
    angle_threshold: 허용 각도 범위
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel 그래디언트
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # 크기와 방향
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.degrees(np.arctan2(gy, gx)) % 180

    # 임계값 적용
    _, edges = cv2.threshold(magnitude.astype(np.uint8), 50, 255, cv2.THRESH_BINARY)

    # 수평 엣지 (방향이 0 또는 180도 근처)
    # Sobel gy가 강하면 수평 엣지
    horizontal_mask = ((direction < angle_threshold) |
                       (direction > 180 - angle_threshold))
    horizontal_edges = np.zeros_like(edges)
    horizontal_edges[horizontal_mask & (edges > 0)] = 255

    # 수직 엣지 (방향이 90도 근처)
    vertical_mask = ((direction > 90 - angle_threshold) &
                     (direction < 90 + angle_threshold))
    vertical_edges = np.zeros_like(edges)
    vertical_edges[vertical_mask & (edges > 0)] = 255

    return horizontal_edges, vertical_edges

# 테스트
img = cv2.imread('image.jpg')
h_edges, v_edges = separate_edges_by_direction(img)

cv2.imshow('Horizontal Edges', h_edges)
cv2.imshow('Vertical Edges', v_edges)
cv2.waitKey(0)
```

</details>

### 문제 3: 다중 스케일 엣지 검출

여러 스케일에서 엣지를 검출하고 합성하는 함수를 구현하세요.

<details>
<summary>힌트</summary>

다양한 sigma 값으로 Gaussian blur를 적용한 후 Canny를 적용하고, 결과를 합성합니다.

</details>

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def multi_scale_canny(image, scales=[1.0, 2.0, 4.0], low=50, high=150):
    """
    다중 스케일 Canny 엣지 검출
    scales: Gaussian blur sigma 값들
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    combined_edges = np.zeros(gray.shape, dtype=np.uint8)

    for sigma in scales:
        # 스케일에 따른 커널 크기
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1

        # Gaussian blur 적용
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)

        # Canny 엣지 검출
        edges = cv2.Canny(blurred, low, high)

        # 합성 (OR 연산)
        combined_edges = cv2.bitwise_or(combined_edges, edges)

    return combined_edges

# 테스트
img = cv2.imread('image.jpg')
edges = multi_scale_canny(img, scales=[1.0, 2.0, 3.0])
cv2.imshow('Multi-scale Canny', edges)
cv2.waitKey(0)
```

</details>

### 추천 문제

| 난이도 | 주제 | 설명 |
|--------|------|------|
| ⭐ | 기본 Canny | 다양한 이미지에 Canny 적용 |
| ⭐⭐ | 임계값 실험 | 트랙바로 최적 임계값 찾기 |
| ⭐⭐ | 전처리 비교 | blur 종류에 따른 엣지 품질 비교 |
| ⭐⭐⭐ | 문서 스캔 | 문서 윤곽선 검출 |
| ⭐⭐⭐ | 동전 검출 | 엣지로 동전 경계 찾기 |

---

## 다음 단계

- [09_Contours.md](./09_Contours.md) - findContours, drawContours, 계층 구조

---

## 참고 자료

- [OpenCV Edge Detection Tutorial](https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html)
- [Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
- [Image Gradients](https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html)
