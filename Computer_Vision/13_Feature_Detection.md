# 특징점 검출 (Feature Detection)

## 개요

특징점(Feature)은 이미지에서 고유하고 반복적으로 검출 가능한 지점입니다. 코너, 블롭, 엣지 교차점 등이 있으며, 이미지 매칭, 객체 인식, 3D 재구성 등에 활용됩니다. 이 레슨에서는 Harris, FAST, SIFT, ORB 등 다양한 특징점 검출 알고리즘을 학습합니다.

---

## 목차

1. [특징점 기초 개념](#1-특징점-기초-개념)
2. [코너 검출 - Harris](#2-코너-검출---harris)
3. [좋은 특징점 - goodFeaturesToTrack](#3-좋은-특징점---goodfeaturestotrack)
4. [FAST 검출기](#4-fast-검출기)
5. [SIFT 검출기](#5-sift-검출기)
6. [ORB 검출기](#6-orb-검출기)
7. [키포인트와 디스크립터](#7-키포인트와-디스크립터)
8. [연습 문제](#8-연습-문제)

---

## 1. 특징점 기초 개념

### 특징점이란?

```
특징점 (Feature Point / Keypoint):
이미지에서 고유하게 식별 가능한 지점

좋은 특징점의 조건:
1. 반복성 (Repeatability): 같은 물체는 항상 같은 특징점
2. 구별성 (Distinctiveness): 서로 다른 특징점은 구별 가능
3. 불변성 (Invariance): 회전, 크기, 조명 변화에 강건
4. 정확성 (Accuracy): 정확한 위치 검출

특징점의 종류:
┌─────────────────────────────────────────────────────────────┐
│  코너 (Corner)              블롭 (Blob)                      │
│                                                             │
│       ┌──────              ●●●●●                            │
│       │                    ●●●●●●●                          │
│    ───┘                    ●●●●●●●●                         │
│                            ●●●●●●●                          │
│   두 방향으로 변화           ●●●●●                            │
│                          특정 크기의 영역                     │
└─────────────────────────────────────────────────────────────┘
```

### 특징점 검출 파이프라인

```
1. 특징점 검출 (Detection)
   - 이미지에서 키포인트 위치 찾기
   - Harris, FAST, SIFT, ORB 등
         │
         ▼
2. 특징점 기술 (Description)
   - 각 키포인트 주변의 특징 벡터 생성
   - SIFT descriptor, ORB descriptor, BRIEF 등
         │
         ▼
3. 특징점 매칭 (Matching)
   - 다른 이미지의 디스크립터와 비교
   - BFMatcher, FLANN 등
```

### 검출기 비교

```
┌────────────────┬───────────┬───────────┬───────────┬──────────┐
│ 알고리즘       │ 속도      │ 회전 불변 │ 크기 불변 │ 특허     │
├────────────────┼───────────┼───────────┼───────────┼──────────┤
│ Harris         │ 빠름      │ ○         │ ✗         │ 없음     │
│ FAST           │ 매우 빠름 │ ✗         │ ✗         │ 없음     │
│ SIFT           │ 느림      │ ○         │ ○         │ 만료     │
│ SURF           │ 보통      │ ○         │ ○         │ 있음     │
│ ORB            │ 빠름      │ ○         │ ○         │ 없음     │
│ AKAZE          │ 보통      │ ○         │ ○         │ 없음     │
└────────────────┴───────────┴───────────┴───────────┴──────────┘
```

---

## 2. 코너 검출 - Harris

### 개념

```
Harris 코너 검출:
이미지 패치를 이동시켰을 때 밝기 변화 분석

- 평탄 영역: 모든 방향으로 변화 없음
- 엣지: 엣지 방향은 변화 없음, 수직 방향은 변화 큼
- 코너: 모든 방향으로 변화 큼

자기상관 행렬 M:
M = Σ [Ix²    IxIy]
    [IxIy   Iy² ]

Ix, Iy: x, y 방향 미분

코너 응답 함수:
R = det(M) - k × (trace(M))²
R = λ1×λ2 - k(λ1 + λ2)²

- R > threshold: 코너
- R ≈ 0: 평탄
- R < 0: 엣지
```

### cv2.cornerHarris()

```python
import cv2
import numpy as np

def harris_corner_detection(image_path):
    """Harris 코너 검출"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Harris 코너 검출
    dst = cv2.cornerHarris(
        gray,
        blockSize=2,     # 이웃 크기
        ksize=3,         # Sobel 커널 크기
        k=0.04           # Harris 파라미터
    )

    # 결과 팽창 (코너 강조)
    dst = cv2.dilate(dst, None)

    # 임계값 이상의 점을 코너로 표시
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
    """서브픽셀 정확도의 Harris 코너"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_float = np.float32(gray)

    # Harris 코너
    dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)

    # 코너 위치 추출
    dst = cv2.dilate(dst, None)
    ret, dst_thresh = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst_thresh = np.uint8(dst_thresh)

    # 연결 요소로 코너 중심 찾기
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst_thresh)

    # 서브픽셀 정밀도로 개선
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(
        gray,
        np.float32(centroids),
        (5, 5),      # 윈도우 크기
        (-1, -1),    # 제로 존
        criteria
    )

    result = img.copy()
    for i, corner in enumerate(corners):
        x, y = corner.ravel()
        if i == 0:  # 첫 번째는 배경
            continue
        cv2.circle(result, (int(x), int(y)), 5, (0, 255, 0), -1)

    cv2.imshow('SubPixel Corners', result)
    cv2.waitKey(0)

    return corners

harris_subpixel('chessboard.jpg')
```

---

## 3. 좋은 특징점 - goodFeaturesToTrack

### cv2.goodFeaturesToTrack()

```
Shi-Tomasi 코너 검출 (Harris 개선):
R = min(λ1, λ2)

Harris보다 더 안정적인 코너 검출
→ 광학 흐름(Optical Flow) 추적에 적합
```

```python
import cv2
import numpy as np

def good_features_demo(image_path):
    """좋은 특징점 검출"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 좋은 특징점 검출
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=100,     # 최대 코너 수
        qualityLevel=0.01,  # 품질 수준 (최대 응답의 비율)
        minDistance=10,     # 코너 간 최소 거리
        blockSize=3,        # 이웃 크기
        useHarrisDetector=False,  # Shi-Tomasi 사용
        k=0.04              # Harris 파라미터 (Harris 사용시)
    )

    result = img.copy()

    if corners is not None:
        corners = np.int_(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(result, (x, y), 5, (0, 255, 0), -1)

        print(f"검출된 코너 수: {len(corners)}")

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
    """특정 영역에서만 특징점 검출"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ROI 마스크 생성 (중앙 영역만)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)

    # 마스크 영역에서만 특징점 검출
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=50,
        qualityLevel=0.01,
        minDistance=10,
        mask=mask
    )

    result = img.copy()

    # 마스크 영역 표시
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
매우 빠른 코너 검출 알고리즘

원리:
중심 픽셀 P 주변의 원형 패턴(16픽셀) 검사

        1  2  3
     16           4
   15               5
  14        P        6
   13               7
     12          8
        11 10 9

판단 기준 (N=12):
- 연속된 N개 픽셀이 P보다 밝으면: 코너
- 연속된 N개 픽셀이 P보다 어두우면: 코너

특징:
- 매우 빠름 (실시간 처리)
- 회전 불변성 없음
- 크기 불변성 없음
- 비최대 억제(NMS)로 다중 검출 방지
```

### cv2.FastFeatureDetector

```python
import cv2
import numpy as np

def fast_detection(image_path):
    """FAST 특징점 검출"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # FAST 검출기 생성
    fast = cv2.FastFeatureDetector_create(
        threshold=20,           # 밝기 임계값
        nonmaxSuppression=True  # 비최대 억제
    )

    # 특징점 검출
    keypoints = fast.detect(gray, None)

    # 결과 그리기
    result = cv2.drawKeypoints(
        img, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print(f"검출된 특징점 수: {len(keypoints)}")

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
    """FAST 임계값 비교"""
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

### 개념

```
SIFT (Scale-Invariant Feature Transform):
크기와 회전에 불변한 특징점 검출 및 기술

단계:
1. 스케일 공간 극값 검출 (DoG: Difference of Gaussians)
2. 키포인트 정밀화 (서브픽셀 정확도, 엣지 제거)
3. 방향 할당 (그래디언트 히스토그램)
4. 디스크립터 계산 (4x4 그리드 x 8방향 = 128차원)

스케일 공간:
┌─────────────────────────────────────────────────┐
│  Octave 0    Octave 1    Octave 2              │
│  ┌───────┐   ┌─────┐    ┌───┐                  │
│  │ σ=1.6│   │ σ=1.6│   │σ=1.6│  → 스케일별     │
│  │ σ=2.0│   │ σ=2.0│   │σ=2.0│     가우시안    │
│  │ σ=2.5│   │ σ=2.5│   │σ=2.5│     블러        │
│  │ σ=3.2│   │ σ=3.2│   │σ=3.2│                 │
│  └───────┘   └─────┘    └───┘                  │
│  원본 크기    1/2 축소   1/4 축소              │
└─────────────────────────────────────────────────┘

DoG (Difference of Gaussians):
D(x, y, σ) = L(x, y, kσ) - L(x, y, σ)
→ 인접한 스케일 간의 차이로 블롭 검출
```

### cv2.SIFT_create()

```python
import cv2
import numpy as np

def sift_detection(image_path):
    """SIFT 특징점 검출"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # SIFT 검출기 생성
    sift = cv2.SIFT_create(
        nfeatures=0,          # 최대 특징점 수 (0=무제한)
        nOctaveLayers=3,      # 옥타브당 레이어 수
        contrastThreshold=0.04,  # 대비 임계값
        edgeThreshold=10,     # 엣지 임계값
        sigma=1.6             # 초기 가우시안 시그마
    )

    # 특징점과 디스크립터 계산
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # 결과 그리기
    result = cv2.drawKeypoints(
        img, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print(f"검출된 특징점 수: {len(keypoints)}")
    if descriptors is not None:
        print(f"디스크립터 크기: {descriptors.shape}")

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
    """SIFT 키포인트 상세 분석"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    print("SIFT 키포인트 분석:")
    print("-" * 50)

    # 키포인트 속성
    for i, kp in enumerate(keypoints[:5]):
        print(f"키포인트 {i}:")
        print(f"  위치 (x, y): ({kp.pt[0]:.1f}, {kp.pt[1]:.1f})")
        print(f"  크기 (scale): {kp.size:.1f}")
        print(f"  방향 (angle): {kp.angle:.1f}도")
        print(f"  응답 (response): {kp.response:.4f}")
        print(f"  옥타브: {kp.octave}")

    # 스케일 분포
    scales = [kp.size for kp in keypoints]
    print(f"\n스케일 범위: {min(scales):.1f} ~ {max(scales):.1f}")

    # 디스크립터 분석
    if descriptors is not None:
        print(f"\n디스크립터:")
        print(f"  차원: {descriptors.shape[1]}")
        print(f"  값 범위: {descriptors.min():.1f} ~ {descriptors.max():.1f}")

analyze_sift_keypoints('object.jpg')
```

---

## 6. ORB 검출기

### 개념

```
ORB (Oriented FAST and Rotated BRIEF):
FAST + BRIEF의 개선 버전, 특허 무료

구성:
1. oFAST: 방향 정보가 추가된 FAST
   - 회전 불변성을 위해 방향(orientation) 계산
   - 이미지 피라미드로 스케일 불변성

2. rBRIEF: 회전된 BRIEF
   - BRIEF: 바이너리 디스크립터 (256비트)
   - 학습된 비교 패턴으로 구별력 향상
   - Hamming 거리로 빠른 매칭

특징:
- SIFT/SURF보다 훨씬 빠름
- 특허 무료
- 실시간 처리에 적합
- 바이너리 디스크립터 → 빠른 매칭

BRIEF 디스크립터:
패치 내 두 점의 밝기 비교
τ(P; x, y) = { 1 if P(x) < P(y)
             { 0 otherwise
→ n개 비교로 n비트 바이너리 문자열
```

### cv2.ORB_create()

```python
import cv2
import numpy as np

def orb_detection(image_path):
    """ORB 특징점 검출"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ORB 검출기 생성
    orb = cv2.ORB_create(
        nfeatures=500,        # 최대 특징점 수
        scaleFactor=1.2,      # 피라미드 스케일 팩터
        nlevels=8,            # 피라미드 레벨 수
        edgeThreshold=31,     # 엣지 임계값
        firstLevel=0,         # 첫 피라미드 레벨
        WTA_K=2,              # BRIEF에서 비교할 점 수 (2, 3, 4)
        scoreType=cv2.ORB_HARRIS_SCORE,  # 점수 유형
        patchSize=31,         # BRIEF 패치 크기
        fastThreshold=20      # FAST 임계값
    )

    # 특징점과 디스크립터 계산
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # 결과 그리기
    result = cv2.drawKeypoints(
        img, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print(f"검출된 특징점 수: {len(keypoints)}")
    if descriptors is not None:
        print(f"디스크립터 크기: {descriptors.shape}")
        print(f"디스크립터 타입: {descriptors.dtype}")  # uint8 (바이너리)

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
    """SIFT와 ORB 성능 비교"""
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

    print("성능 비교:")
    print("-" * 50)
    print(f"SIFT: {len(kps_sift)} points, {sift_time*1000:.1f}ms")
    print(f"ORB:  {len(kps_orb)} points, {orb_time*1000:.1f}ms")
    print(f"속도 비율: ORB가 {sift_time/orb_time:.1f}배 빠름")

    if descs_sift is not None and descs_orb is not None:
        print(f"\nSIFT 디스크립터: {descs_sift.shape}, {descs_sift.dtype}")
        print(f"ORB 디스크립터: {descs_orb.shape}, {descs_orb.dtype}")

    # 시각화
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

### KeyPoint 구조

```python
import cv2
import numpy as np

def keypoint_structure():
    """키포인트 구조 이해"""
    # 키포인트 수동 생성
    kp = cv2.KeyPoint(
        x=100.5,        # x 좌표
        y=200.5,        # y 좌표
        size=20,        # 특징점 크기 (직경)
        angle=45,       # 방향 (도)
        response=0.8,   # 응답 강도
        octave=0,       # 옥타브 (스케일)
        class_id=-1     # 분류 ID
    )

    print("KeyPoint 속성:")
    print(f"  위치: ({kp.pt[0]}, {kp.pt[1]})")
    print(f"  크기: {kp.size}")
    print(f"  방향: {kp.angle}")
    print(f"  응답: {kp.response}")
    print(f"  옥타브: {kp.octave}")
    print(f"  클래스 ID: {kp.class_id}")

keypoint_structure()
```

### 디스크립터 이해

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_descriptors(image_path):
    """디스크립터 시각화"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # SIFT 디스크립터 (128차원 float)
    sift = cv2.SIFT_create()
    kps_sift, descs_sift = sift.detectAndCompute(img, None)

    # ORB 디스크립터 (32바이트 = 256비트)
    orb = cv2.ORB_create()
    kps_orb, descs_orb = orb.detectAndCompute(img, None)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # SIFT 디스크립터 히스토그램
    if descs_sift is not None and len(descs_sift) > 0:
        axes[0, 0].bar(range(128), descs_sift[0])
        axes[0, 0].set_title('SIFT Descriptor (128D)')
        axes[0, 0].set_xlabel('Dimension')

        axes[0, 1].imshow(descs_sift[:50], aspect='auto', cmap='viridis')
        axes[0, 1].set_title('SIFT Descriptors (first 50)')
        axes[0, 1].set_xlabel('Dimension')
        axes[0, 1].set_ylabel('Keypoint')

    # ORB 디스크립터 (바이너리)
    if descs_orb is not None and len(descs_orb) > 0:
        # 바이너리를 비트로 변환
        bits = np.unpackbits(descs_orb[0])
        axes[1, 0].bar(range(256), bits)
        axes[1, 0].set_title('ORB Descriptor (256 bits)')
        axes[1, 0].set_xlabel('Bit')

        # 여러 디스크립터
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
    """다양한 특징점 검출기 사용"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detectors = {
        'SIFT': cv2.SIFT_create(),
        'ORB': cv2.ORB_create(),
        'BRISK': cv2.BRISK_create(),
        'AKAZE': cv2.AKAZE_create(),
        # 'KAZE': cv2.KAZE_create(),  # 느림
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
        print(f"  특징점 수: {len(kps)}")
        if descs is not None:
            print(f"  디스크립터: {descs.shape}, {descs.dtype}")
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
    """가장 강한 N개의 특징점 선택"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ORB로 특징점 검출 (많이 검출)
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # 응답 강도로 정렬
    keypoints_sorted = sorted(keypoints, key=lambda x: x.response, reverse=True)

    # 상위 N개 선택
    best_keypoints = keypoints_sorted[:n]

    # 해당 디스크립터도 선택
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
    """그리드별로 균일하게 특징점 선택"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    orb = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # 그리드 크기 계산
    cell_h = h // grid_size[0]
    cell_w = w // grid_size[1]

    # 각 셀별로 가장 강한 특징점 선택
    selected_kps = []
    selected_indices = []

    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            # 셀 영역
            x_min = col * cell_w
            x_max = (col + 1) * cell_w
            y_min = row * cell_h
            y_max = (row + 1) * cell_h

            # 셀 내의 특징점 필터링
            cell_kps = []
            for i, kp in enumerate(keypoints):
                if x_min <= kp.pt[0] < x_max and y_min <= kp.pt[1] < y_max:
                    cell_kps.append((i, kp))

            if cell_kps:
                # 가장 강한 특징점 선택
                best_idx, best_kp = max(cell_kps, key=lambda x: x[1].response)
                selected_kps.append(best_kp)
                selected_indices.append(best_idx)

    # 디스크립터
    selected_descs = descriptors[selected_indices] if descriptors is not None else None

    result = cv2.drawKeypoints(
        img, selected_kps, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # 그리드 표시
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
    """회전 불변성 테스트"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 이미지 회전
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h))

    # SIFT (회전 불변)
    sift = cv2.SIFT_create(nfeatures=100)

    kps1, descs1 = sift.detectAndCompute(gray, None)
    kps2, descs2 = sift.detectAndCompute(rotated, None)

    # 특징점 매칭
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descs1, descs2, k=2)

    # 좋은 매칭 필터링 (Lowe's ratio test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"원본 특징점: {len(kps1)}")
    print(f"회전 이미지 특징점: {len(kps2)}")
    print(f"매칭된 특징점: {len(good_matches)}")
    print(f"매칭률: {len(good_matches) / len(kps1) * 100:.1f}%")

    # 시각화
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

- [14_Feature_Matching.md](./14_Feature_Matching.md) - BFMatcher, FLANN, Homography

---

## 참고 자료

- [OpenCV Feature Detection](https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html)
- [SIFT Paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
- [ORB Paper](https://www.willowgarage.com/sites/default/files/orb_final.pdf)
