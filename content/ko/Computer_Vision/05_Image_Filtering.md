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

이미지 필터링은 그 수학적 핵심에서 **2D 신호에 적용되는 컨볼루션**입니다. 이 레슨의 모든 선형 필터 — 평균, 가우시안, Sobel, Laplacian, 샤프닝, 엠보스 — 는 같은 연산에 서로 다른 커널을 꽂은 것뿐입니다. 두 필터(중앙값, 양방향)는 컨볼루션 틀 *바깥에* 있으며, 그것이 바로 그들에게 특별한 경계 보존 동작을 부여합니다.

### 이론: 컨볼루션이란 실제로 무엇인가

1D 연속 신호 `f(t)`와 커널 `h(t)`의 컨볼루션:

```
(f * h)(t) = ∫ f(τ) · h(t - τ) dτ
```

이산 2D 이미지의 대응:

```
(I * K)(x, y) = Σᵢ Σⱼ  I(x - i, y - j) · K(i, j)
```

각 출력 픽셀은 국소 이웃 픽셀들의 가중합이며, 가중치는 커널이 결정합니다. **서로 다른 필터 = 서로 다른 가중치 패턴** — 이것이 선형 필터링의 전부입니다.

#### 교차 상관 vs 진짜 컨볼루션

위 교과서적 정의는 커널을 뒤집습니다 — `K(i, j)`가 오프셋 `-i, -j`에서 평가됨. 반면 `cv2.filter2D`가 실제로 계산하는 것은 **교차 상관(cross-correlation)** — 뒤집기 없음:

```
(I ⋆ K)(x, y) = Σᵢ Σⱼ  I(x + i, y + j) · K(i, j)
```

**대칭 커널**(평균, 가우시안, Laplacian, 대부분의 샤프닝)에서는 뒤집기가 무의미하므로 둘이 동일. **비대칭 커널**(Sobel)에서는 180° 회전만큼 다릅니다. 실용 규칙: 비대칭 커널로 진짜 컨볼루션이 필요하면 `cv2.flip(kernel, -1)`로 먼저 뒤집고 `filter2D`. 이 레슨에서는 두 용어를 혼용합니다.

#### 선형 평행이동 불변성 (LSI)

고정 커널과의 컨볼루션은 **Linear Shift-Invariant**:

- **선형**: `I1 + I2`에 필터링 = 각각에 필터링한 결과의 합. 입력 2배 = 출력 2배.
- **평행이동 불변**: 입력을 `(a, b)`만큼 이동하면 출력도 정확히 `(a, b)`만큼 이동. 같은 필터가 이미지 전체에 동일 적용.

`cv2.filter2D`로 만든 모든 필터(가우시안, Sobel, Laplacian, 샤프닝, 엠보스, 커스텀)는 LSI — 즉 §1–3에서 다루는 선형 필터 전부에 해당. **중앙값 필터**(중앙값 블러 절)와 **양방향 필터**(양방향 필터 절)는 선형이 *아닙니다* — 이 비선형성이 선형 필터로는 불가능한 일을 가능하게 합니다.

#### 분리 가능한 커널

2D 커널 `K`가 `K(i, j) = u(i) · v(j)`로 분해되면 **분리 가능(separable)**. 이때 2D 컨볼루션은 두 1D 컨볼루션이 됩니다:

```
I * K  =  (I *row u) *col v
```

`k × k` 커널의 픽셀당 비용이 `k²` 대신 `2k`. 가우시안, 평균, Sobel 커널은 모두 분리 가능하며, OpenCV의 `GaussianBlur`, `boxFilter`, `Sobel`이 내부적으로 이를 활용. 직접 `cv2.sepFilter2D`도 사용 가능.

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

### 이론: 경계 처리

커널이 이미지 가장자리를 넘어가면 존재하지 않는 픽셀을 참조하게 됩니다. 모든 필터는 이 가상 픽셀에 어떤 값을 쓸지 제어하는 `borderType` 파라미터를 노출합니다. 1D 행 `abcdefgh` 양쪽에 가상 픽셀 3개씩:

```
BORDER_CONSTANT     |  0  0  0 | a b c d e f g h |  0  0  0 |   (상수 채움)
BORDER_REPLICATE    |  a  a  a | a b c d e f g h |  h  h  h |   (가장자리 픽셀 복제)
BORDER_REFLECT      |  c  b  a | a b c d e f g h |  h  g  f |   (가장자리 픽셀 포함 반사)
BORDER_REFLECT_101  |  d  c  b | a b c d e f g h |  g  f  e |   (가장자리 픽셀 제외 반사) ← OpenCV 기본
BORDER_WRAP         |  f  g  h | a b c d e f g h |  a  b  c |   (주기적 — 이미지가 반복)
```

효과:

- **CONSTANT**는 0 평균 커널(기울기 연산자)에는 OK이지만, 평균 커널에는 가장자리에 어두운 후광 생성 — 0과의 평균이 값을 끌어내림.
- **REPLICATE**는 자연 이미지에 가장 저렴하고 안전한 기본값 — 새 값을 도입하지 않음.
- **REFLECT_101**(대부분 필터의 OpenCV 기본값)은 경계를 가로질러 미분이 연속이 되도록 가장 매끄러운 확장 생성 — 이미지 가장자리에서 에지 검출 커널의 가짜 강한 응답이 없음. Sobel, Laplacian, Canny 전처리, 다중 스케일 피라미드에 선호.
- **WRAP**은 DFT 주기성을 가정하는 주파수 영역 연산에 의미.

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

### 이론: 블러링이 작동하는 이유 (주파수 영역 관점)

어떠한 이미지든 2D 푸리에 변환을 통해 서로 다른 공간 주파수의 사인파 분해로 표현 가능. 저주파 = 매끄럽고 천천히 변하는 내용(넓은 영역, 그라데이션). 고주파 = 급격한 변화(에지, 텍스처, 잡음).

**컨볼루션 정리**가 필터링에 적용된 푸리에 분석의 중심 결과:

```
F(I * K) = F(I) · F(K)
```

커널의 효과는 그 푸리에 변환 `F(K)`로 완전히 기술됩니다. `F(K) ≈ 1`인 주파수는 통과, `F(K) ≈ 0`인 주파수는 억제됨. 이것이 모든 선형 필터를 분류:

- **저역 통과(Low-pass)**: `F(K)`가 0 근처에서 크고 고주파에서 작음. 매끄러운 내용 통과, 급변 차단. 이것이 블러. (평균, 가우시안.)
- **고역 통과(High-pass)**: `F(K)`가 0에서 ~0이고 고주파에서 큼. 매끄러운 내용 차단, 에지 강조. (Laplacian.)
- **대역 통과(Band-pass)**: 특정 주파수 링만 통과. (Difference-of-Gaussians, §08 에지 검출에서 사용.)

**블러링이 잡음을 제거하는 이유 — 그리고 에지도 흐리는 이유**: 센서 잡음과 소금-후추 잡음은 픽셀 간 급격한 변화로 나타남, 거의 정의상 고주파 성분. 실제 장면 특징은 더 천천히 변하고 저주파에 집중. 저역 통과 커널이 잡음이 사는 고주파 대역을 억제하면서 신호가 사는 저주파 대역을 보존. 걸림돌: **날카로운 에지도 고주파 성분**이라, 선형 저역 통과 필터는 "잡음 때문의 고주파"와 "에지 때문의 고주파"를 구별할 수 없어 — 둘 다 약화시킴. 그래서 모든 선형 블러는 에지를 흐리고, 진짜 경계 보존 평활화에는 비선형이 필요(양방향 필터 절에서 다룸).

---

## 2. 평균 블러 - blur()

### 이론: 저역 통과로서의 박스 필터

박스("균일 평균") 커널 — 모든 항목이 `1/k²` — 은 가장 단순한 저역 통과 필터입니다. 주파수 영역에서 그 푸리에 변환은 2D `sinc` 함수: 0 주파수의 주 lobe + 진동하는 side-lobe. 이 side-lobe는 박스 필터가 고주파를 깨끗하게 억제하지 *않음*을 의미 — 부호가 교대하며 일부가 새어 나옵니다. 그래서 평균된 이미지가 에지 근처에서 약간 "ringy"하게 보이고, 자연 이미지 평활화에 가우시안(§3)보다 일반적으로 열등한 이유.

박스 커널의 유일한 장점: 극단적 단순성과 **분리 가능성**(2D 박스 = 1D 행 평균 × 1D 열 평균), 모든 블러 중 가장 빠름. `cv2.boxFilter`는 `cv2.blur`의 부모 — 같은 연산, 적분 이미지 빌딩 블록으로 유용한 비정규화 변형 옵션 포함.

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

### 이론: 가우시안 커널 심층

2D 가우시안 필터:

```
G(x, y; σ) = (1 / (2π σ²)) · exp( -(x² + y²) / (2σ²) )
```

- **`σ`**는 픽셀 단위 표준편차 — 블러 강도 제어. 큰 `σ`는 가중치를 멀리 퍼뜨려 더 강한 블러.
- **`1 / (2π σ²)`**는 적분이 1이 되도록 정규화, 전체 밝기 보존.

#### 가우시안이 지배적인 이유

가우시안은 다음을 모두 만족하는 (스케일 차이 제외) 유일한 커널:

- **등방(Isotropic)**: 반지름 대칭. 블러된 점이 어느 방향에서 봐도 같음.
- **분리 가능(Separable)**: `G(x, y; σ) = G(x; σ) · G(y; σ)` — 2D 블러 = 1D 블러 두 번(`O(k²) → O(2k)` 가속이 공짜).
- **비음수(Non-negative)**: 음수 가중치 없음, 모든 출력은 진짜 가중 평균 — 입력 범위 밖으로 못 나감.
- **자기 컨볼루션에 닫힘**: `GaussianBlur(σ₁) ∘ GaussianBlur(σ₂) = GaussianBlur(√(σ₁² + σ₂²))`. 이미지 피라미드와 스케일-공간이 이 항등식에 의존.

다른 어떤 커널도 네 가지를 동시에 만족하지 않으므로, 가우시안이 컴퓨터 비전 전반(SIFT의 스케일-공간 §13, Canny의 전처리 §08, 특징 검출, 신경망 사전분포)에 등장.

#### σ vs 커널 크기

경험 법칙: **`ksize ≥ 6σ + 1`**(가우시안 질량의 ±3σ ≈ 99.7% 포함). `sigma=0`을 설정하면 OpenCV가 `ksize`로부터 σ 유도:

```
σ = 0.3 · ((ksize - 1) · 0.5 - 1) + 0.8
```

블러 강도가 관심사라면 `σ`를 명시적으로 설정하고 `ksize=(0, 0)`을 전달 — OpenCV가 충분히 큰 커널을 자동 선택.

#### 확산과의 연결

분산 `σ²`인 `GaussianBlur` 적용은 수학적으로 2D **열 방정식** `∂I/∂t = ∇²I`를 시간 `t = σ²/2`만큼 전진시키는 것과 같습니다. 블러링은 말 그대로 픽셀 값이 확산되는 것 — 금속판에서 열이 퍼지듯이. 스케일-공간 방법과 *비등방 확산*(확산 속도가 국소 기울기에 의존하는 변형, 경계 보존 평활화에 사용)의 기초.

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

### 이론: 비선형 순서 통계 필터로서의 중앙값

중앙값 필터는 컨볼루션이 *아닙니다* — **순서 통계(rank-order) 연산**입니다. 각 픽셀에서 이웃 값을 정렬하고 중간 값을 선택. 이 단순한 변경이 깊은 결과를 가져옵니다:

- **선형 필터는 이상치 제거에 최적이 될 수 없습니다.** 선형 필터의 출력은 가중합이므로, 단일 salt 또는 pepper 이상치(값 0 또는 255)가 그 가중치에 비례해 결과에 직접 영향. 작은 가중치라도 가시적 아티팩트.
- **중앙값은 50%까지의 이상치에 강건합니다.** 정렬된 이웃의 중간 값은 절반 미만이 이상치인 한 변하지 않음. 3×3 이웃(9개 값) 중 단일 이상치는 중앙값에 도달할 수 없습니다.

이 강건성은 잡음 과정이 희소 이상치를 생성한다는 가정 아래 수학적으로 최적 — 정확히 소금-후추 잡음의 모델. 가우시안(가산) 잡음에는 중앙값이 *평균*보다 나쁨(평균은 가우시안 가정 아래 최대우도 추정량) — 그래서 임펄스 잡음에는 중앙값이 이기지만 센서 잡음에는 가우시안 블러가 선호되는 이유.

비용: 정렬은 O(k² log k), 컨볼루션의 O(k²)보다 나쁨. 그리고 비선형이라 LSI 도구(푸리에 분석, 분리 가능성)가 적용 불가.

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
    np.random.seed(0)  # Deterministic noise so the figure is reproducible
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

### 이론: 선형성을 깨서 경계 보존하기

선형 저역 통과 필터는 "잡음으로 인한 고주파"와 "에지로 인한 고주파"를 구별할 수 없음 — `F(K)`에게 둘은 동일. 양방향 필터는 커널을 이웃 픽셀 값에 **의존하게 만들어** 이를 우회. 이 데이터 의존성이 선형성을 깨고, 그것이 바로 목표.

#### 공식

```
BF[I](x) = (1 / W(x)) · Σ_{y ∈ Ω}   Gσ_s(‖x - y‖) · Gσ_r(|I(x) - I(y)|) · I(y)
```

- **Gσ_s** (공간 커널)는 이웃을 *거리*로 가중 — 일반 가우시안 블러와 같은 역할. `sigmaSpace`로 제어.
- **Gσ_r** (range 커널)은 이웃을 *밝기 유사도*로 가중 — 비슷한 색의 픽셀이 더 기여, 에지 건너편 픽셀은 거의 0. `sigmaColor`로 제어.
- **W(x)**는 가중치 합을 1로 만드는 픽셀별 정규화. `x`에 의존하므로 비선형의 일부.

#### 에지가 살아남는 이유

평탄 영역 내부에서는 모든 이웃이 비슷한 밝기를 가져 `Gσ_r ≈ 1`, 양방향 필터가 보통 공간 가우시안으로 퇴화. 에지 근처에서는 반대편 이웃의 밝기가 매우 달라 range 가중치 ≈ 0 — 사실상 무시. 평균은 중심 픽셀이 속한 에지의 한쪽에서만 계산. 평탄 영역은 평활화, 에지 전이는 보존.

비용은 계산적/분석적: 데이터 의존이라 분리 가능성 적용 불가, 푸리에 도구도 직접 적용 불가. 순진한 구현은 픽셀당 O(k²); 빠른 근사(예: permutohedral lattice)가 실시간 사용을 가능하게 함.

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

### 이론: 일반 LSI 필터로서의 filter2D

`cv2.filter2D`는 OpenCV에서 가장 일반적인 선형 필터 원시 — 사용자 지정 커널로 임의의 2D 교차 상관을 구현. §1-3의 모든 필터(그리고 §7의 샤프닝 커널)는 특정 커널의 `filter2D` 호출 하나로 표현 가능; 전용 함수(`blur`, `GaussianBlur`, `Sobel`)는 본질적으로 최적화된 내부를 가진 편의 래퍼.

커스텀 커널 설계 시 두 가지 고려:

- **밝기 보존은 합 1, 에지 검출은 합 0.** 합이 1인 커널(평균, 가우시안, `[[0,-1,0],[-1,5,-1],[0,-1,0]]` 같은 샤프닝)은 DC 성분(평균 밝기)을 보존. 합이 0인 커널(Laplacian, Sobel, 엠보스)은 DC 성분을 제거하고 변화에만 반응 — 에지/특징 추출에 유용.
- **분리 가능 커널에는 `cv2.sepFilter2D` 사용.** 커널이 `K(i, j) = u(i) · v(j)`로 분해되면 `sepFilter2D(img, ddepth, u, v)`가 `filter2D(img, ddepth, K)`보다 훨씬 빠름. 대부분의 유용한 커널(가우시안, 박스, Sobel)은 분리 가능; 가속을 위해 몇 줄 더 작성할 가치가 있음.

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

### 이론: 주파수 분해에서 유도되는 언샤프 마스킹

어떤 이미지든 저주파와 고주파 부분으로 분해:

```
I = I_low + I_high
여기서 I_low = GaussianBlur(I), I_high = I - I_low
```

**샤프닝은 고주파 부분을 증폭**:

```
I_sharp = I_low + (1 + α) · I_high
        = I_low + (1 + α) · (I - I_low)
        = I + α · (I - I_low)
        = I + α · (I - GaussianBlur(I))
```

마지막 형식이 정확히 이 섹션 후반의 `unsharp_mask` 함수가 계산하는 것. `α`는 `amount` 파라미터 — 클수록 고주파 성분을 더 공격적으로 증폭.

**threshold** 파라미터는 평탄 영역 보호: 어떤 `|I - I_low|` 이하의 차이는 실제 디테일 대신 잡음으로 간주되어 원본 픽셀 유지. 없으면 평탄해야 할 영역에서 센서 잡음이 증폭 — 거칠고 알갱이 진 피부나 하늘로 보임.

#### 한 번에 동등한 커널

표준 `[[0,-1,0],[-1,5,-1],[0,-1,0]]` 샤프닝 커널은 같은 연산을 단일 3×3 컨볼루션으로 압축:

- 중심 값 `5` = `1 (원본) + 4 (α=1로 고주파 증폭)`
- 네 `-1`의 합 `-4` (암묵적 블러 빼기)
- 커널 합 `1`, 전체 밝기 보존

강한 샤프닝 커널 `[[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]`은 모든 8개 이웃을 블러 추정(균일 3×3 박스)으로 취급한 `α = 8`에 대응.

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

