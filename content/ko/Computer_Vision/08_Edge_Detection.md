# 엣지 검출 (Edge Detection)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 이미지 그래디언트(image gradient) 개념과 그것이 밝기 변화를 나타내는 방식을 설명할 수 있다
2. OpenCV를 사용하여 소벨(Sobel), 샤르(Scharr), 라플라시안(Laplacian) 연산자로 엣지 검출(edge detection)을 구현할 수 있다
3. 캐니 엣지 검출(Canny edge detection) 알고리즘을 적용하고 이력 임계값(hysteresis threshold)을 조정할 수 있다
4. 1차 미분 연산자와 2차 미분 연산자의 장단점을 비교할 수 있다
5. 그래디언트 크기(gradient magnitude)와 방향(direction)을 분석하여 엣지 특성을 파악할 수 있다
6. 다양한 이미지 유형에 적합한 엣지 검출 방법을 선택하는 전처리 파이프라인(preprocessing pipeline)을 설계할 수 있다

---

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

### 이론: 미분으로서의 에지

에지를 가로지르는 1D 밝기 프로파일을 평활화된 계단 함수로 모델링합니다. 이상적인 날카로운 에지:

```
I(x) = { I_dark   if x < x_edge
       { I_bright otherwise
```

는 `x_edge`에서 불연속입니다. 실세계(회절, 렌즈 블러, 센서 PSF)는 이를 시그모이드 램프로 부드럽게 하지만, 구조는 동일합니다. 미분에 관한 세 가지 사실:

- **1차 미분** `I'(x)`는 에지에서 피크를 가집니다 — 상승 에지는 밝은 스파이크, 하강 에지는 어두운 스파이크. 에지 = `|I'|`의 국소 극값.
- **2차 미분** `I''(x)`는 에지에서 제로를 가로지릅니다 — 한쪽은 양, 다른 한쪽은 음. 에지 = `I''`의 제로 크로싱.

두 성질 모두 2D로 일반화됩니다. 2D 이미지 `I(x, y)`의 기울기는 벡터:

```
∇I = (∂I/∂x, ∂I/∂y)
```

이며 크기 `|∇I|`(에지 강도)와 방향 `θ = atan2(∂I/∂y, ∂I/∂x)`(에지에 수직)을 가집니다. Laplacian `∇²I = ∂²I/∂x² + ∂²I/∂y²`이 2D 2차 미분 연산자입니다.

### 그래디언트란?

```
Gradient: Rate of change in image brightness

Mathematical Definition:
∇f = (∂f/∂x, ∂f/∂y)

- ∂f/∂x: Rate of change in x direction (horizontal)
- ∂f/∂y: Rate of change in y direction (vertical)

Gradient Magnitude:
|∇f| = √((∂f/∂x)² + (∂f/∂y)²)

Gradient Direction:
θ = arctan(∂f/∂y / ∂f/∂x)
```

그래디언트 벡터(∂f/∂x, ∂f/∂y)는 항상 밝기가 가장 급격하게 증가하는 방향을 가리킵니다. 마치 물이 언덕을 올라가는 것과 같습니다. 크기 |∇f|는 엣지가 얼마나 날카로운지를, 방향 θ는 밝기가 증가하는 방향(엣지 경계 자체에 수직)을 알려줍니다. 예를 들어, 수직 엣지(왼쪽이 어둡고 오른쪽이 밝은 경우)는 ∂f/∂x가 크고 ∂f/∂y는 거의 0에 가까우므로, θ ≈ 0°이고 그래디언트(gradient)는 수평 방향을 가리킵니다.

### 엣지의 종류

```
1. Step Edge
   Brightness ──┐
                │
                └── Brightness
   → Ideal edge, abrupt change

2. Ramp Edge
   Brightness ──╲
                 ╲
                  ╲── Brightness
   → Gradual change, blurred boundary

3. Roof Edge
   Brightness ──╱╲
               ╱  ╲
              ╱    ╲── Brightness
   → Line structure

4. Ridge Edge
          ╱╲
         ╱  ╲
      ──╱    ╲──
   → Thin line structure
```

### 엣지 검출 파이프라인

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Input    │     │    Noise    │     │  Gradient   │     │    Edge     │
│    Image    │ ──▶ │   Removal   │ ──▶ │ Calculation │ ──▶ │ Extraction  │
│             │     │  (Gaussian) │     │ (Sobel etc) │     │ (Threshold) │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## 2. Sobel 연산자

### 이론: 1차 연산자: Sobel과 그 가족

#### B.1 순진한 전방 차분과 그것이 쓰이지 않는 이유

가장 단순한 이산 미분은 `I'(x) ≈ I(x+1) - I(x-1)`(중심 차분). 1×3 커널 `[-1 0 1]`로 구현됩니다. 깨끗한 램프에서는 올바른 답을 주지만, 잡음에 극도로 민감합니다 — 각 출력 픽셀이 두 입력 픽셀에만 의존합니다.

#### B.2 Sobel: 평활화 내장 기울기

Sobel은 미분과 수직 방향의 저역 평활화를 결합합니다. 수평 Sobel 커널:

```
S_x = [-1  0  +1]       분리 가능한 곱으로 읽으면:
      [-2  0  +2]       S_x = [1]         · [-1  0  +1]
      [-1  0  +1]             [2]
                              [1]
                        (y 방향 평활화)     (x 방향 차분)
```

수직 Sobel은 그 전치. 이 평활화-후-미분 구조가 Sobel을 픽셀 단위 잡음에 강하게 만듭니다(단순 중심 차분에는 치명적인 잡음). 응답 풋프린트가 약간 커지는 것이 대가입니다.

왜 하필 `[1, 2, 1]` 평활기일까요? 가우시안의 3-탭 근사입니다(`[1,1] * [1,1]`과 같음, 가우시안 이항 근사의 첫 단계). 미분 `[-1, 0, +1]`은 중심 차분입니다.

#### B.3 Scharr: 더 등방적인 Sobel

Sobel의 등방성 — 모든 각도의 에지에 동등하게 반응하는지 — 은 3×3 커널로는 완벽하지 않습니다. Scharr는 3×3 계수를 등방성 기준으로 최적화합니다:

```
Sc_x = [ -3   0   +3]
       [-10   0  +10]
       [ -3   0   +3]
```

3×3 커널에서 각도 정확도가 중요하면(예: HOG 특징의 기울기 방향 히스토그램) Scharr를 쓰세요. 더 큰 커널에서는 Sobel-3이면 충분합니다.

#### B.4 기울기 크기와 방향

Sobel이 `G_x = ∂I/∂x`와 `G_y = ∂I/∂y`를 별도로 줍니다. 결합:

```
|∇I| = sqrt(G_x² + G_y²)        (크기)
θ    = atan2(G_y, G_x)           (방향, 에지에 수직)
```

속도를 위해 `|∇I| ≈ |G_x| + |G_y|`가 자주 쓰입니다 — 회전 대칭이 아니지만 빠릅니다. 크기가 후단(예: Canny, HOG)에서 쓰일 때는 제대로 된 `sqrt` 형식이 필수입니다.

### 개념

```
Sobel Operator: First derivative-based edge detection
→ Calculate gradients in x and y directions separately

3x3 Sobel Kernels:

Gx (Horizontal edge detection):   Gy (Vertical edge detection):
┌────┬────┬────┐                  ┌────┬────┬────┐
│ -1 │  0 │ +1 │                  │ -1 │ -2 │ -1 │
├────┼────┼────┤                  ├────┼────┼────┤
│ -2 │  0 │ +2 │                  │  0 │  0 │  0 │
├────┼────┼────┤                  ├────┼────┼────┤
│ -1 │  0 │ +1 │                  │ +1 │ +2 │ +1 │
└────┴────┴────┘                  └────┴────┴────┘

→ Gx: Detect vertical edges (left-right brightness difference)
→ Gy: Detect horizontal edges (top-bottom brightness difference)
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

# Read image
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel operation
# CV_64F (float64) is required because gradients can be negative —
# a dark-to-bright transition gives a positive value, bright-to-dark gives negative.
# Using uint8 would silently clip all negative values to 0, missing half the edges.
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # x direction
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # y direction

# Convert to absolute value and then to 8-bit
# We take the absolute value so that both directions of contrast
# (bright→dark and dark→bright) map to the same edge strength.
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

# Combine x, y gradients
# Equal weighting (0.5 each) avoids overflow while preserving both edge orientations.
sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# Display results
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
    """Calculate Sobel gradient magnitude"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Gaussian blur before Sobel: the derivative operator amplifies noise
    # (differentiation is a high-pass filter), so smoothing first is essential
    # to distinguish real edges from noise spikes.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Sobel operation (calculate in float64)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Gradient magnitude: sqrt(Gx² + Gy²)
    # This is the Euclidean length of the gradient vector (Gx, Gy), representing
    # the steepness of the brightness ramp at each pixel — large at sharp edges.
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize to 0-255 range
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    return magnitude

# Usage example
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
    """Compare Sobel kernel sizes"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    ksizes = [1, 3, 5, 7]

    for ax, ksize in zip(axes.flatten(), ksizes):
        # When ksize=1, use 3x1 or 1x3 filter
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

# ksize comparison:
# - ksize=1: Most sensitive, vulnerable to noise
# - ksize=3: Standard, balanced results
# - ksize=5, 7: Smoother edges, more robust to noise
```

---

## 3. Scharr 연산자

### 개념

```
Scharr Operator: More accurate 3x3 kernel than Sobel
→ Better rotational symmetry

Scharr Kernels:

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
→ Scharr is more accurate in diagonal directions
```

### cv2.Scharr() 함수

```python
cv2.Scharr(src, ddepth, dx, dy, scale=1, delta=0)
```

```python
import cv2
import numpy as np

def compare_sobel_scharr(image):
    """Compare Sobel and Scharr"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sobel (ksize=3)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # Scharr (fixed 3x3)
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    scharr_mag = np.sqrt(scharr_x**2 + scharr_y**2)

    # Normalize
    sobel_mag = np.clip(sobel_mag, 0, 255).astype(np.uint8)
    scharr_mag = np.clip(scharr_mag, 0, 255).astype(np.uint8)

    return sobel_mag, scharr_mag

# Scharr usage example
img = cv2.imread('image.jpg')
sobel, scharr = compare_sobel_scharr(img)

cv2.imshow('Sobel', sobel)
cv2.imshow('Scharr', scharr)
cv2.waitKey(0)
```

### Sobel에서 Scharr 사용하기

```python
# Use ksize=-1 or ksize=cv2.FILTER_SCHARR in cv2.Sobel()
scharr_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=-1)  # Use Scharr kernel
scharr_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=-1)

# Above code is equivalent to
scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
```

---

## 4. Laplacian 연산자

### 이론: 2차 연산자: Laplacian

Laplacian `∇²I = ∂²I/∂x² + ∂²I/∂y²`의 3×3 이산 근사:

```
L = [ 0  1  0]     또는     [ 1  1  1]     (대각선 포함, 약간 덜 등방적)
    [ 1 -4  1]              [ 1 -8  1]
    [ 0  1  0]              [ 1  1  1]
```

이상적인 계단 에지에서 `∇²I`는 부호를 바꿉니다 — 에지 바로 앞에서 양, 에지에서 0, 바로 뒤에서 음(또는 반대). 에지는 `∇²I`의 **제로 크로싱**이지, *극값이 아닙니다*.

#### C.1 Laplacian이 잡음에 민감한 이유

2차 미분은 1차보다 잡음을 더 증폭합니다. 한 픽셀의 무작위 변동이 커널 내에서 이웃의 네 배 가중치를 받습니다. 잡음이 있는 이미지에 원시 Laplacian을 쓰면 가짜 제로 크로싱의 바다가 생깁니다.

#### C.2 Laplacian-of-Gaussian (LoG)

먼저 평활화해 잡음 문제를 해결:

```
LoG(x, y; σ) = ∇²[G(x, y; σ) * I](x, y)  =  [∇²G(x, y; σ)] * I(x, y)
```

Laplacian은 선형 컨볼루션과 교환하므로, 평활화 커널에 미리 구워 넣을 수 있습니다. LoG 커널은 특유의 "멕시칸 햇" 모양 — 중심이 양, 그 주위 고리가 음, 가장자리는 0으로 접근. LoG 필터링된 이미지의 제로 크로싱이 스케일 `σ`의 에지입니다.

#### C.3 Difference-of-Gaussians (DoG) ≈ LoG

LoG는 직접 계산하기에 비쌉니다. 핵심 근사:

```
DoG(x, y; σ) = G(x, y; k·σ) - G(x, y; σ)  ≈  (k-1) · σ² · LoG(x, y; σ)
```

여기서 `k ≈ 1.6`. 두 번의 분리 가능 가우시안 블러와 뺄셈이 한 번의 분리 불가능한 LoG와 같은 결과를 줍니다. SIFT가 DoG를 사용하는 이유이자(§13), 대부분의 스케일-공간 방법이 직접 LoG 대신 가우시안 차이 위에 구축되는 이유입니다.

### 개념

```
Laplacian Operator: Second derivative-based edge detection
→ Zero-crossing at points where brightness changes rapidly

Mathematical Definition:
∇²f = ∂²f/∂x² + ∂²f/∂y²

Laplacian Kernels:

4-connectivity:             8-connectivity:
┌────┬────┬────┐           ┌────┬────┬────┐
│  0 │  1 │  0 │           │  1 │  1 │  1 │
├────┼────┼────┤           ├────┼────┼────┤
│  1 │ -4 │  1 │           │  1 │ -8 │  1 │
├────┼────┼────┤           ├────┼────┼────┤
│  0 │  1 │  0 │           │  1 │  1 │  1 │
└────┴────┴────┘           └────┴────┴────┘

Characteristics:
- Detects edges regardless of direction
- Very sensitive to noise (second derivative)
- Zero-crossing points are edges
```

### 1차 미분 vs 2차 미분

```
Original Signal (Step Edge):
       ────────────┐
                   │
                   └────────────

First Derivative (Sobel):
                  ╱╲
                 ╱  ╲
       ─────────╱    ╲─────────
       → Peak point is edge

Second Derivative (Laplacian):
            ╱╲
           ╱  ╲
       ───╱    ╲───
              ╱  ╲
             ╱    ╲
       → Zero-crossing point is edge
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
    """Laplacian edge detection"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Remove noise (Laplacian is sensitive to noise)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Laplacian operation
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)

    # Convert to absolute value
    laplacian = cv2.convertScaleAbs(laplacian)

    return laplacian

# Usage example
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
    LoG (Laplacian of Gaussian) edge detection
    1. Remove noise with Gaussian blur
    2. Detect edges with Laplacian
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur (kernel size based on sigma)
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1

    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)

    # Laplacian
    log = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)

    # Absolute value
    log = cv2.convertScaleAbs(log)

    return log

# Use LoG
img = cv2.imread('image.jpg')
edges = log_edge_detection(img, sigma=1.5)
cv2.imshow('LoG', edges)
cv2.waitKey(0)
```

---

## 5. Canny 엣지 검출

### 이론: Canny 에지 검출기: 5단계 파이프라인

John Canny(1986)는 세 가지 최적성 기준으로부터 그의 알고리즘을 유도했습니다:

1. **좋은 검출(Good detection)**: 진짜 에지는 찾아져야 하고, 에지가 아닌 것은 플래그되지 말아야 한다.
2. **좋은 국소화(Good localization)**: 검출된 에지는 실제 에지에 가까워야 한다.
3. **단일 응답(Single response)**: 각 에지는 두꺼운 응답이 아니라 정확히 하나의 검출된 선을 만들어야 한다.

가우시안 잡음 모델 아래에서 이 기준들을 최적화하면 특정 필터 모양 — 놀랍게도 `∂G/∂x`에 매우 가까움 — 과 특정 후처리 파이프라인이 유도됩니다. 다섯 단계:

#### D.1 가우시안 평활화

미분 전에 고주파 잡음을 억제하기 위해 `GaussianBlur(σ)` 블러링(Sobel이 이미 일부 수행하지만, Canny는 선택한 스케일에서 명시적으로 수행).

#### D.2 기울기 계산

Sobel로 `G_x`, `G_y`를 얻고 크기 `|∇I|`와 방향 `θ`를 계산. `θ`를 네 빈으로 양자화: 0°(수평), 45°, 90°(수직), 135°.

#### D.3 비최대 억제 (Non-maximum suppression)

에지를 얇게 만듭니다. 각 픽셀에 대해 기울기 방향(에지에 수직) 따라 두 이웃을 봅니다. 현재 픽셀의 크기가 그 셋 중 최대가 아니면 0으로. "단일 응답" 기준을 다룹니다 — 이 단계가 없으면 강한 에지가 넓은 높은-크기 능선을 만듭니다.

#### D.4 이중 임계값

살아남은 픽셀을 분류:

- **강함(Strong)** (`|∇I| ≥ T_high`): 확실히 에지.
- **약함(Weak)** (`T_low ≤ |∇I| < T_high`): 아마도 에지.
- **0** (`|∇I| < T_low`): 에지 아님.

Canny의 경험적 지침: `T_high ≈ 2 · T_low`이 좋은 시작점. 두 임계값이 Canny가 강하고 명확한 에지와 미묘한 연속부 모두에서 강건하게 작동하는 이유입니다.

#### D.5 히스테리시스 추적

약한 픽셀이 강한 픽셀에 (8-이웃, 전이적으로도) 연결되어 있을 때만 강함으로 바꿉니다. 최종 에지 맵은 원래 강한 픽셀 + 그로부터 도달 가능한 모든 약한 픽셀. §07.E.3과 같은 히스테리시스 아이디어 — 중간 경우의 결정은 이웃 기반.

결과: 얇은(§D.3 덕분) 에지가 실제 에지 위치 가까이(최적화된 필터 덕분) 있고, 잡음 영역을 가로질러 연결됨(§D.5 덕분).

### 개념

```
Canny Edge Detection: Multi-stage edge detection algorithm
→ Most widely used edge detection method

Canny's 3 Goals:
1. Low error rate: Detect only real edges
2. Accurate localization: Edges at precise locations
3. Single response: One line for one edge

4-Stage Processing:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Gaussian   │     │    Sobel    │     │     Non-    │     │  Hysteresis │
│    Blur     │ ──▶ │  Gradient   │ ──▶ │   Maximum   │ ──▶ │ Thresholding│
│             │     │             │     │ Suppression │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### Canny 알고리즘 상세

```
Step 1: Noise Removal (Gaussian Blur)
- Apply 5x5 Gaussian filter
- Remove high-frequency noise

Step 2: Gradient Calculation
- Calculate Gx, Gy with Sobel operation
- Magnitude: G = √(Gx² + Gy²)
- Direction: θ = arctan(Gy/Gx)

Step 3: Non-Maximum Suppression (NMS)
┌─────────────────────────────────────┐
│  Keep only maximum values along     │
│  gradient direction                 │
│  → Make edges 1 pixel thin          │
└─────────────────────────────────────┘

Direction Quantization (4 directions):
        90°
         │
  135° ──┼── 45°
         │
        0° (180°)

Example:
When direction θ = 45°, compare along diagonal
┌───┬───┬───┐
│   │ q │   │
├───┼───┼───┤
│   │ p │   │  Keep p if p > q and p > r
├───┼───┼───┤
│   │ r │   │
└───┴───┴───┘

Step 4: Hysteresis Thresholding
┌─────────────────────────────────────┐
│  high_threshold: Strong edges       │
│  low_threshold: Weak edges          │
│                                     │
│  Strong edges: Always include       │
│  Weak edges: Include if connected   │
│                to strong edge       │
│  Others: Remove                     │
└─────────────────────────────────────┘

Example:
high = 100, low = 50

Pixel value 120 → Strong edge (include)
Pixel value 70  → Weak edge (check connection)
Pixel value 30  → Remove
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
    """Canny edge detection"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Pre-blurring before Canny is optional but recommended:
    # Canny's internal Gaussian (apertureSize-derived) is fixed, while this
    # external blur lets you control smoothing scale independently of edge precision.
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Hysteresis thresholding uses two thresholds rather than one to solve the
    # "weak edge" problem: a single threshold either breaks continuous edges
    # (too high) or includes noise (too low). High marks definite edges;
    # low admits uncertain pixels only when they connect to a definite edge.
    edges = cv2.Canny(blurred, low, high)

    return edges

# Usage example
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
    """Adjust Canny thresholds with trackbar"""
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

        # Ensure low is not greater than high
        if low >= high:
            low = high - 1

        edges = cv2.Canny(blurred, low, high)

        cv2.imshow('Canny', edges)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cv2.destroyAllWindows()

# Execute
canny_with_trackbar('image.jpg')
```

### 자동 임계값 설정

```python
import cv2
import numpy as np

def auto_canny(image, sigma=0.33):
    """
    Automatic threshold Canny
    Calculate low and high based on median value
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Calculate median
    median = np.median(blurred)

    # Calculate thresholds
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))

    print(f"Auto threshold: low={low}, high={high}")

    edges = cv2.Canny(blurred, low, high)

    return edges

# Usage example
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
    Canny edge detection on color images
    Detect edges on each channel and combine
    """
    # Method 1: Convert to grayscale then process
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges_gray = cv2.Canny(gray, low, high)

    # Method 2: Process each channel then combine
    b, g, r = cv2.split(image)
    edges_b = cv2.Canny(b, low, high)
    edges_g = cv2.Canny(g, low, high)
    edges_r = cv2.Canny(r, low, high)

    # Combine with OR operation
    edges_color = cv2.bitwise_or(edges_b, edges_g)
    edges_color = cv2.bitwise_or(edges_color, edges_r)

    return edges_gray, edges_color

# Usage example
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
    """Calculate gradient magnitude and direction"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Sobel gradient
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude
    magnitude = np.sqrt(gx**2 + gy**2)

    # arctan2(gy, gx) gives the full 360° direction of the gradient vector;
    # we reduce to 0-180° because edge orientation is undirected — an edge
    # running NE-SW is the same as SW-NE (opposite gradient directions).
    direction = np.arctan2(gy, gx)

    # Convert direction to degrees (0-180)
    direction_deg = np.degrees(direction) % 180

    return magnitude, direction_deg

# Usage example
img = cv2.imread('image.jpg')
mag, dir = gradient_magnitude_direction(img)

# Normalize and display
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
    Visualize gradient direction with arrows
    step: Sampling interval
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(gx**2 + gy**2)

    # Draw arrows
    result = image.copy()
    h, w = gray.shape

    for y in range(step, h - step, step):
        for x in range(step, w - step, step):
            if magnitude[y, x] > 50:  # Display only above certain magnitude
                # Normalize direction vector
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

# Usage example
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
    """Compare various edge detection algorithms"""
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

    # Visualization
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

# Run comparison
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
    Adaptive Canny edge detection
    Automatically set thresholds based on median brightness
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Calculate median
    median = np.median(blurred)

    # Calculate thresholds (adjust range with sigma)
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))

    edges = cv2.Canny(blurred, low, high)

    return edges, low, high

# Test
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
    Separate horizontal/vertical edges
    angle_threshold: Allowed angle range
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel gradient
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude and direction
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.degrees(np.arctan2(gy, gx)) % 180

    # Apply threshold
    _, edges = cv2.threshold(magnitude.astype(np.uint8), 50, 255, cv2.THRESH_BINARY)

    # Horizontal edges (direction near 0 or 180 degrees)
    # Strong Sobel gy means horizontal edge
    horizontal_mask = ((direction < angle_threshold) |
                       (direction > 180 - angle_threshold))
    horizontal_edges = np.zeros_like(edges)
    horizontal_edges[horizontal_mask & (edges > 0)] = 255

    # Vertical edges (direction near 90 degrees)
    vertical_mask = ((direction > 90 - angle_threshold) &
                     (direction < 90 + angle_threshold))
    vertical_edges = np.zeros_like(edges)
    vertical_edges[vertical_mask & (edges > 0)] = 255

    return horizontal_edges, vertical_edges

# Test
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
    Multi-scale Canny edge detection
    scales: Gaussian blur sigma values
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    combined_edges = np.zeros(gray.shape, dtype=np.uint8)

    for sigma in scales:
        # Kernel size based on scale
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)

        # Canny edge detection
        edges = cv2.Canny(blurred, low, high)

        # Combine (OR operation)
        combined_edges = cv2.bitwise_or(combined_edges, edges)

    return combined_edges

# Test
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

- [윤곽선 검출 (Contour Detection)](./09_Contours.md) - findContours, drawContours, 계층 구조

---

## 참고 자료

- [OpenCV Edge Detection Tutorial](https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html)
- [Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
- [Image Gradients](https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html)
