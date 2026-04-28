# 특징점 검출 (Feature Detection)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 특징점(feature point)이 무엇이며 좋은 키포인트(keypoint)를 구성하는 속성(반복성, 변별력, 지역성)을 정의할 수 있다
2. 해리스 코너 검출(Harris corner detection)을 구현하고 코너 응답 함수(corner response function)를 해석할 수 있다
3. OpenCV를 사용하여 이미지에서 키포인트를 찾기 위해 FAST, SIFT, ORB 검출기를 적용할 수 있다
4. 속도, 정확도, 특허/라이선스 측면에서 Harris, FAST, SIFT, ORB의 장단점을 비교할 수 있다
5. 키포인트 디스크립터(keypoint descriptor)를 추출하고 시각화하며, 지역 이미지 구조를 어떻게 인코딩하는지 이해할 수 있다
6. 변환된 이미지 버전에 걸쳐 키포인트 반복성(keypoint repeatability)을 분석하여 검출기 성능을 평가할 수 있다

---

## 개요

특징점(Feature)은 이미지에서 고유하고 반복적으로 검출 가능한 지점입니다. 코너, 블롭, 엣지 교차점 등이 있으며, 이미지 매칭, 객체 인식, 3D 재구성 등에 활용됩니다. 이 레슨에서는 Harris, FAST, SIFT, ORB 등 다양한 특징점 검출 알고리즘을 학습합니다.

---

## 목차

1. [특징점 기초 개념](#1-특징점-기초-개념)
2. [코너 검출 - Harris](#2-코너-검출---harris)
3. [좋은 특징점](#3-좋은-특징점)
4. [FAST 검출기](#4-fast-검출기)
5. [SIFT 검출기](#5-sift-검출기)
6. [ORB 검출기](#6-orb-검출기)
7. [키포인트와 디스크립터](#7-키포인트와-디스크립터)
8. [연습 문제](#8-연습-문제)

---

## 1. 특징점 기초 개념

### 이론: 좋은 키포인트의 조건

유용한 키포인트는 세 가지 속성을 만족해야 합니다 — 셋은 서로 긴장 관계이며, 모든 검출기가 다르게 타협합니다.

#### 반복성 (Repeatability)

같은 장면의 다른 뷰에서 **같은 점**이 발견되어야 합니다. 창문 모서리는 정면에서 찍든 각도에서 찍든, 밝은 빛이든 어두운 빛이든, 압축된 후든, 다른 카메라로든 검출되어야 합니다. 없으면 같은 것의 두 이미지에서 나온 키포인트들이 겹치지 않고, 매칭이 불가능해집니다.

#### 변별력 (Distinctiveness)

각 키포인트는 **이미지의 대부분의 다른 점들과 달라 보여야** 합니다. 모든 키포인트가 같은 국소 외관을 가지면(예: 반복 텍스처 위) 이미지 A의 어느 키포인트가 B의 어느 키포인트에 대응하는지 알 수 없어 — 매칭이 모호해집니다.

#### 지역성 (Locality)

키포인트는 **작은 영역**을 묘사해야 합니다. 부분적 가림이 키포인트를 파괴하지 않고, 디스크립터 계산이 빠르도록. 큰 범위의 "특징"은 디스크립터가 모든 이미지에 존재하지 않을 내용에 의존하게 만듭니다.

이 레슨의 세 검출기는 이를 다르게 타협합니다:

- **Harris**는 코너에 대한 반복성과 변별력에 집중하지만 스케일 불변성이 없습니다.
- **SIFT**는 훨씬 높은 계산 비용으로 스케일 및 회전 불변성을 추가합니다.
- **FAST/ORB**는 실시간 사용이 가능할 정도로 큰 속도 향상을 위해 반복성 일부를 희생합니다.

### 특징점이란?

```
Feature Point / Keypoint:
A uniquely identifiable point in an image

Requirements for Good Features:
1. Repeatability: Same object should produce same features
2. Distinctiveness: Different features should be distinguishable
3. Invariance: Robust to rotation, scale, and lighting changes
4. Accuracy: Precise location detection

Types of Features:
+-------------------------------------------------------------+
|  Corner                    Blob                              |
|                                                              |
|       +------              *****                             |
|       |                    *******                           |
|    ---+                    ********                          |
|                            *******                           |
|   Change in two            *****                             |
|   directions               Specific region size              |
+-------------------------------------------------------------+
```

### 특징점 검출 파이프라인

```
1. Feature Detection
   - Find keypoint locations in image
   - Harris, FAST, SIFT, ORB, etc.
         |
         v
2. Feature Description
   - Generate feature vector around each keypoint
   - SIFT descriptor, ORB descriptor, BRIEF, etc.
         |
         v
3. Feature Matching
   - Compare descriptors with other images
   - BFMatcher, FLANN, etc.
```

### 검출기 비교

```
+----------------+-----------+-----------+-----------+----------+
| Algorithm      | Speed     | Rotation  | Scale     | Patent   |
|                |           | Invariant | Invariant |          |
+----------------+-----------+-----------+-----------+----------+
| Harris         | Fast      | O         | X         | None     |
| FAST           | Very Fast | X         | X         | None     |
| SIFT           | Slow      | O         | O         | Expired  |
| SURF           | Medium    | O         | O         | Yes      |
| ORB            | Fast      | O         | O         | None     |
| AKAZE          | Medium    | O         | O         | None     |
+----------------+-----------+-----------+-----------+----------+
```

---

## 2. 코너 검출 - Harris

### 이론: 구조 텐서 (Structure Tensor)

#### 중심 아이디어

코너는 밝기가 **둘 이상의 방향으로** 빠르게 변하는 점입니다. 작은 창을 어느 방향으로든 살짝 이동하면 내용에 큰 변화가 생겨야 합니다. 형식적으로, 픽셀 `(x, y)` 주변 창을 `(u, v)`만큼 이동해 제곱차 합을 측정합니다:

```
E(u, v) = Σ_{(x',y') ∈ W}  [ I(x' + u, y' + v) - I(x', y') ]²
```

`(x', y')` 주변에서 `I(x' + u, y' + v)`의 1차 Taylor 전개:

```
I(x' + u, y' + v) ≈ I(x', y') + u · I_x(x', y') + v · I_y(x', y')
```

`E(u, v)`에 대입:

```
E(u, v) ≈ Σ [ u · I_x + v · I_y ]²

        = [u v] · M · [u; v]            where M = Σ [ I_x²     I_x·I_y ]
                                                 [ I_x·I_y  I_y²    ]
```

2×2 행렬 `M`이 **구조 텐서**(second-moment matrix 또는 autocorrelation matrix로도 불림). 이웃에서 이미지 밝기가 어떻게 변하는지를 인코딩하며, 거의 모든 코너 검출기의 기초입니다.

#### 고유값 읽기

`M`의 고유값 `λ₁ ≥ λ₂`는 창의 주축 방향을 따라 `E`가 얼마나 빨리 성장하는지를 측정합니다:

| `λ₁`, `λ₂` | `E`의 모양 | 해석 |
|------------|------------|-------|
| 둘 다 작음 | 평평 | 균일 영역 — 구조 없음 |
| 하나 크고 하나 작음 | 능선 | 에지 — 한 방향으로만 밝기 변화 |
| 둘 다 큼 | 그릇 | 코너 — 두 방향 모두 밝기 변화 |

따라서 **코너 검출은 두 고유값 모두가 큰 픽셀을 찾는 것으로 축소됩니다**. `λ₁`, `λ₂`의 서로 다른 스칼라 조합이 서로 다른 검출기를 만듭니다.

#### 계산 세부

실전에서 `M`은 `I_x`, `I_y`에 Sobel을, 합에 가우시안 가중 창을 써서 계산됩니다. (균일 가중치가 아닌) 가우시안 가중이 더 매끄러운 응답을 주고 검출기를 정확한 창 크기에 덜 민감하게 만듭니다.

### 이론: Harris 코너 응답

고유값 계산은 비쌉니다. Harris(1988)는 `min(λ₁, λ₂)`처럼 반응하지만 행렬식과 트레이스만 쓰는 스칼라를 제안했습니다:

```
R = det(M) - k · trace(M)²  =  λ₁·λ₂ - k·(λ₁ + λ₂)²
```

여기서 `k ≈ 0.04–0.06`은 경험적 상수. 동작:

- 평평(`λ₁, λ₂` 둘 다 작음): `R`은 작고 양수나 음수일 수 있음.
- 에지(한 `λ`가 훨씬 큼): `det(M) ≈ 0`이지만 `trace(M)²`는 큼, 따라서 `R`은 강하게 음수.
- 코너(둘 다 큼): `det(M)`이 크고 `R`은 강하게 양수.

`R`이 어떤 값 `R_thresh` 이상이고 국소 최대이면 유지. 두 조건을 통과하는 픽셀이 Harris 코너.

**Shi-Tomasi**(1994)는 실전에서 더 단순한 스칼라 `R = min(λ₁, λ₂)`가 약간 더 잘 작동함을 관찰했습니다 — 더 작은 고유값이 임계값을 넘기만 하면 됩니다. 이것이 `goodFeaturesToTrack`이 쓰는 것입니다. Harris와의 차이는 보통 작지만 일관되게 Shi-Tomasi가 더 좋습니다.

### 개념

```
Harris Corner Detection:
Analyzes intensity changes when shifting an image patch

- Flat region: No change in any direction
- Edge: No change along edge direction, large change perpendicular
- Corner: Large change in all directions

Auto-correlation Matrix M:
M = sum [Ix^2    IxIy]
        [IxIy   Iy^2 ]

Ix, Iy: Derivatives in x, y directions

Corner Response Function:
R = det(M) - k * (trace(M))^2
R = lambda1*lambda2 - k(lambda1 + lambda2)^2

- R > threshold: Corner
- R ~ 0: Flat
- R < 0: Edge
```

**기하학적 직관**: M의 고유값(eigenvalue) λ1과 λ2는 픽셀 주변의 두 가지 주요 밝기 변화 방향을 나타냅니다. **코너(corner)**에서는 모든 방향으로 이미지가 강하게 변하므로 두 고유값이 모두 커서 det(M) = λ1·λ2가 크고 R이 양수가 됩니다. **엣지(edge)**에서는 한 방향으로만 밝기가 강하게 변하므로 하나의 고유값은 크고 다른 하나는 작아 det(M)는 작지만 trace(M)는 여전히 커서 R이 음수가 됩니다. **평탄 영역(flat)**에서는 두 고유값 모두 미미하여 R이 0에 가깝습니다. 파라미터 k(일반적으로 0.04–0.06)는 민감도를 제어합니다: 작은 k는 더 많은 코너를 검출하지만 거짓 양성도 증가합니다.

### cv2.cornerHarris()

```python
import cv2
import numpy as np

def harris_corner_detection(image_path):
    """Harris corner detection"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Harris corner detection
    dst = cv2.cornerHarris(
        gray,
        blockSize=2,     # Small patch (2px) captures fine structure; larger values
                         # smooth over detail and miss tight corners
        ksize=3,         # Sobel kernel size: 3 is the smallest that gives a
                         # good gradient estimate without excessive blurring
        k=0.04           # Harris sensitivity: 0.04–0.06 is the empirically safe
                         # range — smaller k detects more corners but adds false
                         # positives; larger k misses weak corners
    )

    # Dilate result to expand corner peaks so they are visible as distinct
    # bright regions — without dilation, corners appear as single-pixel dots
    dst = cv2.dilate(dst, None)

    # Threshold at 1% of peak response: scales automatically to image content
    # so the same code works on both high-contrast and low-contrast images
    result = img.copy()
    result[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imshow('Harris Corners', result)
    cv2.waitKey(0)

    return dst

harris_corner_detection('chessboard.jpg')
```

### 서브픽셀 정확도

```python
import cv2
import numpy as np

def harris_subpixel(image_path):
    """Harris corners with sub-pixel accuracy"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_float = np.float32(gray)

    # Harris corners — same parameters as above (blockSize=2, ksize=3, k=0.04)
    dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)

    # Dilate then threshold: dilation merges nearby high-response pixels into
    # connected blobs, making connectedComponentsWithStats find cleaner centroids
    dst = cv2.dilate(dst, None)
    ret, dst_thresh = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst_thresh = np.uint8(dst_thresh)

    # Find corner centroids using connected components
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst_thresh)

    # Refine to sub-pixel precision
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(
        gray,
        np.float32(centroids),
        (5, 5),      # Window size
        (-1, -1),    # Zero zone
        criteria
    )

    result = img.copy()
    for i, corner in enumerate(corners):
        x, y = corner.ravel()
        if i == 0:  # First one is background
            continue
        cv2.circle(result, (int(x), int(y)), 5, (0, 255, 0), -1)

    cv2.imshow('SubPixel Corners', result)
    cv2.waitKey(0)

    return corners

harris_subpixel('chessboard.jpg')
```

---

## 3. 좋은 특징점

### cv2.goodFeaturesToTrack()

```
Shi-Tomasi Corner Detection (Harris improvement):
R = min(lambda1, lambda2)

More stable corner detection than Harris
-> Suitable for optical flow tracking
```

```python
import cv2
import numpy as np

def good_features_demo(image_path):
    """Good features detection"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect good features
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=100,     # Cap output count — unlimited corners overwhelm
                            # downstream trackers (e.g., Lucas-Kanade)
        qualityLevel=0.01,  # Accept corners with response >= 1% of the strongest;
                            # low value keeps weaker but still useful corners
        minDistance=10,     # Enforce spatial spread: prevents clusters of
                            # redundant corners from the same region
        blockSize=3,        # Slightly larger patch than Harris (2) for more
                            # stable gradient estimates in noisy images
        useHarrisDetector=False,  # Use Shi-Tomasi (min eigenvalue) — it avoids
                                  # the k trade-off and is more numerically stable
        k=0.04              # Harris parameter (only active when useHarrisDetector=True)
    )

    result = img.copy()

    if corners is not None:
        corners = np.int_(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(result, (x, y), 5, (0, 255, 0), -1)

        print(f"Detected corners: {len(corners)}")

    cv2.imshow('Good Features', result)
    cv2.waitKey(0)

    return corners

good_features_demo('building.jpg')
```

### 마스크를 이용한 영역 제한

```python
import cv2
import numpy as np

def features_with_mask(image_path):
    """Detect features only in specific region"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Create ROI mask (center region only)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)

    # Detect features only in masked region
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=50,
        qualityLevel=0.01,
        minDistance=10,
        mask=mask
    )

    result = img.copy()

    # Show mask region
    cv2.rectangle(result, (w//4, h//4), (3*w//4, 3*h//4), (128, 128, 128), 2)

    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(result, (int(x), int(y)), 5, (0, 255, 0), -1)

    cv2.imshow('Features with Mask', result)
    cv2.waitKey(0)

features_with_mask('scene.jpg')
```

---

## 4. FAST 검출기

### 개념

```
FAST (Features from Accelerated Segment Test):
Very fast corner detection algorithm

Principle:
Examine circular pattern (16 pixels) around center pixel P

        1  2  3
     16           4
   15               5
  14        P        6
   13               7
     12          8
        11 10 9

Decision criterion (N=12):
- If N consecutive pixels are brighter than P: Corner
- If N consecutive pixels are darker than P: Corner

Characteristics:
- Very fast (real-time processing)
- No rotation invariance
- No scale invariance
- Non-maximum suppression (NMS) prevents multiple detections
```

### cv2.FastFeatureDetector

```python
import cv2
import numpy as np

def fast_detection(image_path):
    """FAST feature detection"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create FAST detector
    fast = cv2.FastFeatureDetector_create(
        threshold=20,           # Intensity threshold
        nonmaxSuppression=True  # Non-maximum suppression
    )

    # Detect features
    keypoints = fast.detect(gray, None)

    # Draw results
    result = cv2.drawKeypoints(
        img, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print(f"Detected features: {len(keypoints)}")

    cv2.imshow('FAST', result)
    cv2.waitKey(0)

    return keypoints

fast_detection('building.jpg')
```

### FAST 파라미터 비교

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_fast_thresholds(image_path):
    """Compare FAST thresholds"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresholds = [10, 20, 30, 50]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, thresh in zip(axes, thresholds):
        fast = cv2.FastFeatureDetector_create(
            threshold=thresh,
            nonmaxSuppression=True
        )
        kps = fast.detect(gray, None)
        result = cv2.drawKeypoints(img, kps, None, color=(0, 255, 0))

        ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Threshold={thresh}, Points={len(kps)}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

compare_fast_thresholds('building.jpg')
```

---

## 5. SIFT 검출기

### 이론: 스케일-공간과 DoG: 스케일 불변 키포인트 만들기

Harris 코너는 **Sobel 미분 창이 암시하는 스케일**의 코너에 반응합니다. 이미지를 줌인하면 "코너"가 고정된 창 크기에서 더 이상 코너처럼 보이지 않을 수 있습니다 — 서로 다른 스케일의 같은 장면이 서로 다른 검출기 출력을 냅니다. 서로 다른 거리에서 찍은 사진을 매칭하려면 이는 치명적입니다.

#### 스케일-공간 아이디어

이미지를 **연속 3D 스택** `L(x, y, σ)`로 표현 — 원본을 증가하는 `σ`의 가우시안 커널과 컨볼루션한 것. 키포인트는 이 스택 내 어떤 스케일 특성 응답이 최대가 되는 위치 `(x, y, σ)`. 피크의 `σ`가 키포인트의 **고유 스케일** — 이미지 내 특징의 실제 크기.

#### 왜 Laplacian-of-Gaussian

Lindeberg는 `σ²·∇²L(x, y, σ)`(정규화된 LoG)의 극값이 블롭 같은 구조의 자연스러운 스케일에 대응함을 보였습니다. 밝은 배경의 어두운 블롭을 가지고 `σ`를 증가시키면, LoG 응답은 `σ`가 블롭 크기와 일치할 때 정확히 피크에 도달 — 위치와 스케일 모두를 줍니다.

#### 왜 실전에서 DoG

LoG는 직접 계산이 비쌉니다. Lesson 08(엣지 검출)에서 유도한 대로, **Difference-of-Gaussians**가 스케일된 LoG를 근사합니다:

```
DoG(x, y; σ) = L(x, y; k·σ) - L(x, y; σ)  ≈  (k-1) · σ² · ∇²L(x, y, σ)
```

SIFT(Lowe, 2004)는 서로 다른 스케일의 가우시안 블러 이미지의 **피라미드**를 만들고, 인접 스케일 간 DoG를 취하고, 3D (x, y, σ) 공간에서 국소 극값을 찾습니다. 각 극값이 위치와 스케일을 가진 SIFT 키포인트가 됩니다.

SIFT는 또한 이웃의 지배적 기울기 방향을 계산해 키포인트에 붙여서 회전 불변성을 줍니다. 디스크립터(아래)는 이 방향 기준으로 계산되므로, 이미지를 회전하면 디스크립터도 예측 가능하게 회전합니다.

### 개념

```
SIFT (Scale-Invariant Feature Transform):
Feature detection and description invariant to scale and rotation

Steps:
1. Scale-space extrema detection (DoG: Difference of Gaussians)
2. Keypoint localization (sub-pixel accuracy, edge removal)
3. Orientation assignment (gradient histogram)
4. Descriptor computation (4x4 grid x 8 directions = 128 dimensions)

Scale Space:
+-------------------------------------------------+
|  Octave 0    Octave 1    Octave 2               |
|  +-------+   +-----+    +---+                   |
|  | s=1.6|   | s=1.6|   |s=1.6|  -> Scale-wise   |
|  | s=2.0|   | s=2.0|   |s=2.0|     Gaussian     |
|  | s=2.5|   | s=2.5|   |s=2.5|     blur         |
|  | s=3.2|   | s=3.2|   |s=3.2|                  |
|  +-------+   +-----+    +---+                   |
|  Original    1/2 size   1/4 size                |
+-------------------------------------------------+

DoG (Difference of Gaussians):
D(x, y, sigma) = L(x, y, k*sigma) - L(x, y, sigma)
-> Blob detection via difference between adjacent scales
```

### cv2.SIFT_create()

```python
import cv2
import numpy as np

def sift_detection(image_path):
    """SIFT feature detection"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create(
        nfeatures=0,          # 0 = unlimited; set a cap (e.g. 500) to bound
                              # descriptor computation time at inference
        nOctaveLayers=3,      # Layers between octaves: 3 gives good coverage
                              # of the scale space without redundant computation
        contrastThreshold=0.04,  # Reject low-contrast keypoints (texture-less areas)
                                 # — lower values keep more but noisier keypoints
        edgeThreshold=10,     # Reject edge responses; higher values are more
                              # permissive and allow more edge-like keypoints
        sigma=1.6             # Initial Gaussian blur matches the assumed camera
                              # smoothing; Lowe's original paper validated 1.6
    )

    # Compute keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Draw results
    result = cv2.drawKeypoints(
        img, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print(f"Detected features: {len(keypoints)}")
    if descriptors is not None:
        print(f"Descriptor size: {descriptors.shape}")

    cv2.imshow('SIFT', result)
    cv2.waitKey(0)

    return keypoints, descriptors

kps, descs = sift_detection('object.jpg')
```

### SIFT 키포인트 분석

```python
import cv2
import numpy as np

def analyze_sift_keypoints(image_path):
    """Detailed analysis of SIFT keypoints"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    print("SIFT Keypoint Analysis:")
    print("-" * 50)

    # Keypoint attributes
    for i, kp in enumerate(keypoints[:5]):
        print(f"Keypoint {i}:")
        print(f"  Location (x, y): ({kp.pt[0]:.1f}, {kp.pt[1]:.1f})")
        print(f"  Size (scale): {kp.size:.1f}")
        print(f"  Angle: {kp.angle:.1f} degrees")
        print(f"  Response: {kp.response:.4f}")
        print(f"  Octave: {kp.octave}")

    # Scale distribution
    scales = [kp.size for kp in keypoints]
    print(f"\nScale range: {min(scales):.1f} ~ {max(scales):.1f}")

    # Descriptor analysis
    if descriptors is not None:
        print(f"\nDescriptors:")
        print(f"  Dimensions: {descriptors.shape[1]}")
        print(f"  Value range: {descriptors.min():.1f} ~ {descriptors.max():.1f}")

analyze_sift_keypoints('object.jpg')
```

---

## 6. ORB 검출기

### 개념

```
ORB (Oriented FAST and Rotated BRIEF):
Improved version of FAST + BRIEF, patent-free

Components:
1. oFAST: FAST with orientation information
   - Computes orientation for rotation invariance
   - Image pyramid for scale invariance

2. rBRIEF: Rotated BRIEF
   - BRIEF: Binary descriptor (256 bits)
   - Learned comparison patterns for better discrimination
   - Fast matching with Hamming distance

Characteristics:
- Much faster than SIFT/SURF
- Patent-free
- Suitable for real-time processing
- Binary descriptor -> Fast matching

BRIEF Descriptor:
Compare intensities of two points in a patch
tau(P; x, y) = { 1 if P(x) < P(y)
               { 0 otherwise
-> n comparisons yield n-bit binary string
```

### cv2.ORB_create()

```python
import cv2
import numpy as np

def orb_detection(image_path):
    """ORB feature detection"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create ORB detector
    orb = cv2.ORB_create(
        nfeatures=500,        # Practical limit for real-time use: 500 gives good
                              # coverage without overwhelming the matcher
        scaleFactor=1.2,      # Mild downscale per pyramid level; smaller values
                              # build finer-grained scale space but cost more memory
        nlevels=8,            # 8 levels gives ~3.6× scale range (1.2^8), enough
                              # to handle typical viewpoint scale changes
        edgeThreshold=31,     # Matches patchSize: keeps FAST away from borders
                              # so BRIEF can sample a full 31×31 patch
        firstLevel=0,         # Start detection at full resolution for small features
        WTA_K=2,              # Default binary comparison (pairs); use 3 or 4 for
                              # more discriminative but larger descriptors
        scoreType=cv2.ORB_HARRIS_SCORE,  # Harris score is more accurate than FAST
                                         # score for ranking keypoints by quality
        patchSize=31,         # BRIEF patch size: larger patches are more
                              # discriminative but slower to compute
        fastThreshold=20      # FAST intensity threshold: 20 balances sensitivity
                              # and false positive rate for typical images
    )

    # Compute keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Draw results
    result = cv2.drawKeypoints(
        img, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print(f"Detected features: {len(keypoints)}")
    if descriptors is not None:
        print(f"Descriptor size: {descriptors.shape}")
        print(f"Descriptor type: {descriptors.dtype}")  # uint8 (binary)

    cv2.imshow('ORB', result)
    cv2.waitKey(0)

    return keypoints, descriptors

kps, descs = orb_detection('object.jpg')
```

### SIFT vs ORB 비교

```python
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def compare_sift_orb(image_path):
    """Compare SIFT and ORB performance"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # SIFT
    sift = cv2.SIFT_create()
    start = time.time()
    kps_sift, descs_sift = sift.detectAndCompute(gray, None)
    sift_time = time.time() - start

    # ORB
    orb = cv2.ORB_create(nfeatures=len(kps_sift))
    start = time.time()
    kps_orb, descs_orb = orb.detectAndCompute(gray, None)
    orb_time = time.time() - start

    print("Performance Comparison:")
    print("-" * 50)
    print(f"SIFT: {len(kps_sift)} points, {sift_time*1000:.1f}ms")
    print(f"ORB:  {len(kps_orb)} points, {orb_time*1000:.1f}ms")
    print(f"Speed ratio: ORB is {sift_time/orb_time:.1f}x faster")

    if descs_sift is not None and descs_orb is not None:
        print(f"\nSIFT descriptor: {descs_sift.shape}, {descs_sift.dtype}")
        print(f"ORB descriptor: {descs_orb.shape}, {descs_orb.dtype}")

    # Visualization
    result_sift = cv2.drawKeypoints(img, kps_sift, None, color=(0, 255, 0))
    result_orb = cv2.drawKeypoints(img, kps_orb, None, color=(0, 0, 255))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(cv2.cvtColor(result_sift, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'SIFT: {len(kps_sift)} points')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(result_orb, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'ORB: {len(kps_orb)} points')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

compare_sift_orb('object.jpg')
```

---

## 7. 키포인트와 디스크립터

### 이론: 디스크립터: 이미지 간 키포인트 매칭

키포인트 검출은 문제의 절반 — 나머지 절반은 다른 이미지에서 식별 가능하게 해주는 벡터를 붙이는 것입니다. 이것이 **디스크립터**. 두 가족이 지배적입니다:

#### 기울기 히스토그램 디스크립터 (SIFT, SURF, HOG)

각 키포인트 주변에 고정 크기 패치(예: 16×16)에서 기울기 크기와 방향을 계산. 패치를 하위 셀로 나눔(예: 4×4 픽셀 셀의 4×4 격자). 각 하위 셀에서 기울기 방향의 히스토그램을 크기로 가중해 예: 8 각도 빈에 비닝. 연결: `4×4×8 = 128` 숫자. 이것이 SIFT 디스크립터.

속성:

- **변별력**: 128차원이면 수백만 가지의 서로 다른 국소 외관을 구별할 공간이 있음.
- **작은 어긋남에 강건**: 히스토그램은 기울기가 어느 정확한 픽셀에 떨어지는지가 아니라 어느 빈인지에만 민감.
- **느림**: 키포인트당 128개의 `float32`, 유클리드 거리로 매칭 — 대규모에서 비쌈.

#### 이진 디스크립터 (BRIEF, ORB, BRISK)

각 키포인트 주변에 미리 선택된 픽셀 쌍 집합(예: 256 쌍). 각 쌍 `(p₁, p₂)`에 대해 디스크립터 비트는 `1 if I(p₁) < I(p₂) else 0`. 256비트를 32바이트 문자열로 연결.

속성:

- **컴팩트**: 256비트(32바이트) vs SIFT의 512바이트.
- **매칭 빠름**: Hamming 거리(비트 XOR + popcount)가 float 유클리드 거리보다 ~100배 빠름.
- **덜 강건**: 이진 비교는 히스토그램 집계보다 잡음과 조명 변화에 더 민감. ORB가 학습된 방향 샘플링 패턴으로 이를 완화.

ORB(Oriented FAST and Rotated BRIEF) = FAST 키포인트 + 회전 및 학습된 BRIEF 디스크립터 + 다중 스케일 피라미드. SIFT의 특허 없는 대안이며, 많은 응용에서 SIFT와 비슷한 품질로 10–100배 빠릅니다.

### KeyPoint 구조

```python
import cv2
import numpy as np

def keypoint_structure():
    """Understanding keypoint structure"""
    # Manually create keypoint
    kp = cv2.KeyPoint(
        x=100.5,        # x coordinate
        y=200.5,        # y coordinate
        size=20,        # Feature size (diameter)
        angle=45,       # Orientation (degrees)
        response=0.8,   # Response strength
        octave=0,       # Octave (scale)
        class_id=-1     # Class ID
    )

    print("KeyPoint Attributes:")
    print(f"  Location: ({kp.pt[0]}, {kp.pt[1]})")
    print(f"  Size: {kp.size}")
    print(f"  Angle: {kp.angle}")
    print(f"  Response: {kp.response}")
    print(f"  Octave: {kp.octave}")
    print(f"  Class ID: {kp.class_id}")

keypoint_structure()
```

### 디스크립터 이해

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_descriptors(image_path):
    """Visualize descriptors"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # SIFT descriptor (128-dimensional float)
    sift = cv2.SIFT_create()
    kps_sift, descs_sift = sift.detectAndCompute(img, None)

    # ORB descriptor (32 bytes = 256 bits)
    orb = cv2.ORB_create()
    kps_orb, descs_orb = orb.detectAndCompute(img, None)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # SIFT descriptor histogram
    if descs_sift is not None and len(descs_sift) > 0:
        axes[0, 0].bar(range(128), descs_sift[0])
        axes[0, 0].set_title('SIFT Descriptor (128D)')
        axes[0, 0].set_xlabel('Dimension')

        axes[0, 1].imshow(descs_sift[:50], aspect='auto', cmap='viridis')
        axes[0, 1].set_title('SIFT Descriptors (first 50)')
        axes[0, 1].set_xlabel('Dimension')
        axes[0, 1].set_ylabel('Keypoint')

    # ORB descriptor (binary)
    if descs_orb is not None and len(descs_orb) > 0:
        # Convert binary to bits
        bits = np.unpackbits(descs_orb[0])
        axes[1, 0].bar(range(256), bits)
        axes[1, 0].set_title('ORB Descriptor (256 bits)')
        axes[1, 0].set_xlabel('Bit')

        # Multiple descriptors
        bits_all = np.unpackbits(descs_orb[:50], axis=1)
        axes[1, 1].imshow(bits_all, aspect='auto', cmap='binary')
        axes[1, 1].set_title('ORB Descriptors (first 50)')
        axes[1, 1].set_xlabel('Bit')
        axes[1, 1].set_ylabel('Keypoint')

    plt.tight_layout()
    plt.show()

visualize_descriptors('object.jpg')
```

### 다양한 검출기 사용

```python
import cv2
import numpy as np

def use_various_detectors(image_path):
    """Use various feature detectors"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detectors = {
        'SIFT': cv2.SIFT_create(),
        'ORB': cv2.ORB_create(),
        'BRISK': cv2.BRISK_create(),
        'AKAZE': cv2.AKAZE_create(),
        # 'KAZE': cv2.KAZE_create(),  # Slow
    }

    results = {}

    for name, detector in detectors.items():
        kps, descs = detector.detectAndCompute(gray, None)
        results[name] = {
            'keypoints': kps,
            'descriptors': descs,
            'count': len(kps),
            'desc_size': descs.shape[1] if descs is not None else 0
        }

        print(f"{name}:")
        print(f"  Feature count: {len(kps)}")
        if descs is not None:
            print(f"  Descriptor: {descs.shape}, {descs.dtype}")
        print()

    return results

results = use_various_detectors('object.jpg')
```

---

## 8. 연습 문제

### 문제 1: 최적 특징점 선택

이미지에서 가장 강한 50개의 특징점만 선택하세요.

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def select_best_keypoints(image_path, n=50):
    """Select N strongest keypoints"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect many keypoints with ORB
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Sort by response strength
    keypoints_sorted = sorted(keypoints, key=lambda x: x.response, reverse=True)

    # Select top N
    best_keypoints = keypoints_sorted[:n]

    # Select corresponding descriptors
    indices = [keypoints.index(kp) for kp in best_keypoints]
    best_descriptors = descriptors[indices] if descriptors is not None else None

    result = cv2.drawKeypoints(
        img, best_keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imshow(f'Best {n} Keypoints', result)
    cv2.waitKey(0)

    return best_keypoints, best_descriptors

kps, descs = select_best_keypoints('building.jpg', n=50)
```

</details>

### 문제 2: 균일 분포 특징점

이미지를 그리드로 나누어 각 셀에서 하나씩 특징점을 선택하세요.

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def uniform_keypoints(image_path, grid_size=(8, 8)):
    """Select keypoints uniformly per grid cell"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    orb = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Calculate grid size
    cell_h = h // grid_size[0]
    cell_w = w // grid_size[1]

    # Select strongest keypoint per cell
    selected_kps = []
    selected_indices = []

    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            # Cell region
            x_min = col * cell_w
            x_max = (col + 1) * cell_w
            y_min = row * cell_h
            y_max = (row + 1) * cell_h

            # Filter keypoints in cell
            cell_kps = []
            for i, kp in enumerate(keypoints):
                if x_min <= kp.pt[0] < x_max and y_min <= kp.pt[1] < y_max:
                    cell_kps.append((i, kp))

            if cell_kps:
                # Select strongest keypoint
                best_idx, best_kp = max(cell_kps, key=lambda x: x[1].response)
                selected_kps.append(best_kp)
                selected_indices.append(best_idx)

    # Descriptors
    selected_descs = descriptors[selected_indices] if descriptors is not None else None

    result = cv2.drawKeypoints(
        img, selected_kps, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Draw grid
    for row in range(1, grid_size[0]):
        cv2.line(result, (0, row * cell_h), (w, row * cell_h), (128, 128, 128), 1)
    for col in range(1, grid_size[1]):
        cv2.line(result, (col * cell_w, 0), (col * cell_w, h), (128, 128, 128), 1)

    cv2.imshow('Uniform Keypoints', result)
    cv2.waitKey(0)

    return selected_kps, selected_descs

kps, descs = uniform_keypoints('building.jpg', grid_size=(6, 8))
```

</details>

### 문제 3: 회전 불변성 테스트

이미지를 회전시킨 후 동일한 특징점이 검출되는지 확인하세요.

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def test_rotation_invariance(image_path, angle=45):
    """Test rotation invariance"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Rotate image
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h))

    # SIFT (rotation invariant)
    sift = cv2.SIFT_create(nfeatures=100)

    kps1, descs1 = sift.detectAndCompute(gray, None)
    kps2, descs2 = sift.detectAndCompute(rotated, None)

    # Feature matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descs1, descs2, k=2)

    # Filter good matches (Lowe's ratio test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Original features: {len(kps1)}")
    print(f"Rotated image features: {len(kps2)}")
    print(f"Matched features: {len(good_matches)}")
    print(f"Match rate: {len(good_matches) / len(kps1) * 100:.1f}%")

    # Visualization
    result = cv2.drawMatches(
        gray, kps1, rotated, kps2,
        good_matches, None,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow('Rotation Invariance Test', result)
    cv2.waitKey(0)

test_rotation_invariance('object.jpg', angle=30)
```

</details>

### 추천 문제

| 난이도 | 주제 | 설명 |
|--------|------|------|
| ⭐ | 기본 검출 | Harris, FAST, ORB 비교 |
| ⭐⭐ | 성능 비교 | 검출 속도와 개수 비교 |
| ⭐⭐ | 파라미터 튜닝 | 최적 파라미터 찾기 |
| ⭐⭐⭐ | 스케일 불변성 | 크기 변화에 대한 테스트 |
| ⭐⭐⭐ | 실시간 검출 | 웹캠으로 실시간 특징점 표시 |

---

## 다음 단계

- [특징점 매칭 (Feature Matching)](./14_Feature_Matching.md) - BFMatcher, FLANN, Homography

---

## 참고 자료

- [OpenCV Feature Detection](https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html)
- [SIFT Paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
- [ORB Paper](https://www.willowgarage.com/sites/default/files/orb_final.pdf)
