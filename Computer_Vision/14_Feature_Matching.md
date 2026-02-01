# 특징점 매칭 (Feature Matching)

## 개요

특징점 매칭은 두 이미지에서 동일한 특징점을 찾아 연결하는 과정입니다. 객체 인식, 이미지 스티칭, 3D 재구성, 객체 추적 등에 활용됩니다. 이 레슨에서는 BFMatcher, FLANN, 거리 메트릭, Lowe's ratio test, Homography, RANSAC 등을 학습합니다.

---

## 목차

1. [특징점 매칭 기초](#1-특징점-매칭-기초)
2. [BFMatcher](#2-bfmatcher)
3. [FLANN 기반 매처](#3-flann-기반-매처)
4. [거리 메트릭](#4-거리-메트릭)
5. [매칭 필터링](#5-매칭-필터링)
6. [Homography와 RANSAC](#6-homography와-ransac)
7. [이미지 스티칭 기초](#7-이미지-스티칭-기초)
8. [연습 문제](#8-연습-문제)

---

## 1. 특징점 매칭 기초

### 매칭 과정

```
┌─────────────────────────────────────────────────────────────────────┐
│                     특징점 매칭 파이프라인                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   이미지 1                        이미지 2                          │
│   ┌─────────┐                     ┌─────────┐                       │
│   │ ●  ●    │                     │   ●  ●  │                       │
│   │    ●  ● │                     │ ●    ●  │                       │
│   │  ●      │                     │   ●     │                       │
│   └─────────┘                     └─────────┘                       │
│       │                               │                             │
│       ▼                               ▼                             │
│  ┌──────────┐                   ┌──────────┐                        │
│  │ 특징점   │                   │ 특징점   │                        │
│  │ 검출     │                   │ 검출     │                        │
│  └────┬─────┘                   └────┬─────┘                        │
│       │                               │                             │
│       ▼                               ▼                             │
│  ┌──────────┐                   ┌──────────┐                        │
│  │디스크립터│                   │디스크립터│                        │
│  │ 계산     │                   │ 계산     │                        │
│  └────┬─────┘                   └────┬─────┘                        │
│       │                               │                             │
│       └──────────┬───────────────────┘                              │
│                  ▼                                                  │
│           ┌──────────────┐                                          │
│           │   매칭       │                                          │
│           │ (BFMatcher   │                                          │
│           │  or FLANN)   │                                          │
│           └──────┬───────┘                                          │
│                  ▼                                                  │
│           ┌──────────────┐                                          │
│           │ 필터링       │                                          │
│           │ (Ratio Test, │                                          │
│           │  RANSAC)     │                                          │
│           └──────────────┘                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### DMatch 구조

```python
import cv2

# DMatch 속성
# match.queryIdx  : 쿼리(첫 번째) 이미지의 디스크립터 인덱스
# match.trainIdx  : 훈련(두 번째) 이미지의 디스크립터 인덱스
# match.imgIdx    : 훈련 이미지의 인덱스 (여러 이미지 매칭 시)
# match.distance  : 디스크립터 간의 거리 (유사도)
```

---

## 2. BFMatcher

### 개념

```
BFMatcher (Brute-Force Matcher):
모든 디스크립터 쌍의 거리를 계산하여 최소 거리 찾기

장점:
- 구현 간단
- 항상 최적 매칭 보장

단점:
- O(N × M) 복잡도 (N, M: 디스크립터 개수)
- 대량 특징점에서 느림

                Query Descriptors
                d1   d2   d3   d4
            ┌────┬────┬────┬────┐
Train   d1' │ 10 │ 25 │ 15 │ 30 │
Desc    d2' │ 20 │  5 │ 35 │ 12 │  ← 각 셀: 거리
        d3' │ 30 │ 18 │  8 │ 22 │
            └────┴────┴────┴────┘

매칭: d1↔d1'(10), d2↔d2'(5), d3↔d3'(8), d4↔d2'(12)
```

### cv2.BFMatcher

```python
import cv2
import numpy as np

def bf_matching_demo(img1_path, img2_path):
    """BFMatcher 기본 사용법"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # ORB 검출기 (바이너리 디스크립터)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # BFMatcher 생성 (Hamming 거리 - 바이너리용)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 매칭
    matches = bf.match(des1, des2)

    # 거리순 정렬
    matches = sorted(matches, key=lambda x: x.distance)

    # 상위 30개 매칭 그리기
    result = cv2.drawMatches(
        img1, kp1, img2, kp2,
        matches[:30], None,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    )

    print(f"총 매칭 수: {len(matches)}")
    print(f"최소 거리: {matches[0].distance:.2f}")
    print(f"최대 거리: {matches[-1].distance:.2f}")

    cv2.imshow('BF Matches', result)
    cv2.waitKey(0)

    return matches

matches = bf_matching_demo('query.jpg', 'train.jpg')
```

### crossCheck 옵션

```python
import cv2

def bf_crosscheck_comparison(img1_path, img2_path):
    """crossCheck 옵션 비교"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # crossCheck=False
    bf_no_cross = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_no_cross = bf_no_cross.match(des1, des2)

    # crossCheck=True
    # A→B와 B→A 모두 일치해야 매칭
    bf_cross = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_cross = bf_cross.match(des1, des2)

    print(f"crossCheck=False: {len(matches_no_cross)} matches")
    print(f"crossCheck=True:  {len(matches_cross)} matches")

    # crossCheck=True가 더 신뢰할 수 있는 매칭

bf_crosscheck_comparison('query.jpg', 'train.jpg')
```

### knnMatch

```python
import cv2
import numpy as np

def bf_knn_matching(img1_path, img2_path, k=2):
    """k-최근접 이웃 매칭"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # SIFT 검출기 (float 디스크립터)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher (L2 거리 - float용)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # k개의 최근접 이웃 반환
    matches = bf.knnMatch(des1, des2, k=k)

    # 각 쿼리 디스크립터에 대해 k개의 매칭
    print(f"쿼리 디스크립터 수: {len(des1)}")
    print(f"각 쿼리당 {k}개 매칭")

    # 첫 번째 쿼리의 매칭 확인
    if len(matches) > 0:
        print(f"\n첫 번째 쿼리의 매칭:")
        for i, m in enumerate(matches[0]):
            print(f"  매칭 {i+1}: trainIdx={m.trainIdx}, distance={m.distance:.2f}")

    return matches

matches = bf_knn_matching('query.jpg', 'train.jpg', k=2)
```

---

## 3. FLANN 기반 매처

### 개념

```
FLANN (Fast Library for Approximate Nearest Neighbors):
근사 최근접 이웃 검색을 위한 라이브러리

특징:
- BFMatcher보다 빠름 (대규모 데이터)
- 근사 알고리즘 (100% 정확하지 않음)
- KD-Tree, K-Means Tree 등 사용

인덱스 유형:
1. FLANN_INDEX_KDTREE (0): float 디스크립터용
2. FLANN_INDEX_LSH (6): 바이너리 디스크립터용
```

### FLANN 사용법

```python
import cv2
import numpy as np

def flann_matching_sift(img1_path, img2_path):
    """FLANN 매칭 (SIFT - float 디스크립터)"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN 파라미터 설정 (KD-Tree)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(
        algorithm=FLANN_INDEX_KDTREE,
        trees=5
    )
    search_params = dict(
        checks=50  # 검색 횟수 (높을수록 정확, 느림)
    )

    # FLANN 매처 생성
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # k-최근접 이웃 매칭
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"총 매칭: {len(matches)}")
    print(f"좋은 매칭: {len(good_matches)}")

    # 결과 그리기
    result = cv2.drawMatches(
        img1, kp1, img2, kp2,
        good_matches, None,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow('FLANN Matches', result)
    cv2.waitKey(0)

    return good_matches, kp1, kp2

matches, kp1, kp2 = flann_matching_sift('query.jpg', 'train.jpg')
```

### FLANN for ORB (바이너리)

```python
import cv2
import numpy as np

def flann_matching_orb(img1_path, img2_path):
    """FLANN 매칭 (ORB - 바이너리 디스크립터)"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # FLANN 파라미터 (LSH - 바이너리용)
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,        # 해시 테이블 수
        key_size=12,           # 키 크기
        multi_probe_level=1    # 다중 프로브 레벨
    )
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 디스크립터를 float32로 변환 (FLANN 요구사항)
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)

    matches = flann.knnMatch(des1, des2, k=2)

    # Ratio test
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    result = cv2.drawMatches(
        img1, kp1, img2, kp2,
        good_matches, None,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow('FLANN ORB Matches', result)
    cv2.waitKey(0)

    return good_matches

flann_matching_orb('query.jpg', 'train.jpg')
```

---

## 4. 거리 메트릭

### 거리 유형

```
┌────────────────────────────────────────────────────────────────────┐
│                        거리 메트릭 비교                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  cv2.NORM_L1 (Manhattan Distance)                                 │
│  d = Σ|a_i - b_i|                                                  │
│  → 잘 사용하지 않음                                                 │
│                                                                    │
│  cv2.NORM_L2 (Euclidean Distance)                                 │
│  d = √(Σ(a_i - b_i)²)                                             │
│  → SIFT, SURF 등 float 디스크립터용                                │
│                                                                    │
│  cv2.NORM_HAMMING                                                  │
│  d = Σ(a_i XOR b_i)                                               │
│  → ORB, BRIEF 등 바이너리 디스크립터용 (256비트)                   │
│                                                                    │
│  cv2.NORM_HAMMING2                                                 │
│  → ORB (WTA_K=3,4) 용                                              │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 디스크립터별 추천 메트릭

```python
import cv2

# 디스크립터별 권장 거리 메트릭
descriptor_distance = {
    'SIFT': cv2.NORM_L2,
    'SURF': cv2.NORM_L2,
    'KAZE': cv2.NORM_L2,
    'ORB': cv2.NORM_HAMMING,
    'BRISK': cv2.NORM_HAMMING,
    'AKAZE': cv2.NORM_HAMMING,  # 바이너리 모드
    'BRIEF': cv2.NORM_HAMMING,
    'FREAK': cv2.NORM_HAMMING,
}

def get_matcher(descriptor_type):
    """디스크립터 유형에 맞는 매처 반환"""
    norm_type = descriptor_distance.get(descriptor_type, cv2.NORM_L2)
    return cv2.BFMatcher(norm_type, crossCheck=True)
```

---

## 5. 매칭 필터링

### Lowe's Ratio Test

```
Lowe's Ratio Test:
최근접 이웃과 두 번째 최근접 이웃의 거리 비율로 필터링

원리:
좋은 매칭 → 최근접이 확실히 가까움 (비율 작음)
나쁜 매칭 → 여러 후보가 비슷한 거리 (비율 큼)

distance(best) / distance(second_best) < threshold

권장 threshold: 0.7 ~ 0.8
```

```python
import cv2
import numpy as np

def lowe_ratio_test(img1_path, img2_path, ratio_thresh=0.75):
    """Lowe's ratio test 적용"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio test
    good_matches = []
    for m, n in matches:
        ratio = m.distance / n.distance
        if ratio < ratio_thresh:
            good_matches.append(m)

    print(f"전체 매칭: {len(matches)}")
    print(f"Ratio test 통과: {len(good_matches)}")
    print(f"필터링 비율: {len(good_matches)/len(matches)*100:.1f}%")

    # 매칭 품질 분석
    if good_matches:
        distances = [m.distance for m in good_matches]
        print(f"평균 거리: {np.mean(distances):.2f}")
        print(f"거리 표준편차: {np.std(distances):.2f}")

    return good_matches, kp1, kp2

matches, kp1, kp2 = lowe_ratio_test('query.jpg', 'train.jpg')
```

### 거리 기반 필터링

```python
import cv2
import numpy as np

def distance_based_filtering(matches, threshold_factor=2.0):
    """거리 기반 매칭 필터링"""
    if not matches:
        return []

    distances = [m.distance for m in matches]
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    # 평균 + k*표준편차 이하만 유지
    threshold = mean_dist + threshold_factor * std_dist

    good_matches = [m for m in matches if m.distance < threshold]

    print(f"거리 평균: {mean_dist:.2f}")
    print(f"거리 표준편차: {std_dist:.2f}")
    print(f"임계값: {threshold:.2f}")
    print(f"필터링 결과: {len(matches)} → {len(good_matches)}")

    return good_matches
```

### 대칭 매칭 (Symmetric Matching)

```python
import cv2

def symmetric_matching(des1, des2, norm_type=cv2.NORM_L2):
    """대칭 매칭 (A→B와 B→A 모두 확인)"""
    bf = cv2.BFMatcher(norm_type)

    # A → B 매칭
    matches_ab = bf.knnMatch(des1, des2, k=1)

    # B → A 매칭
    matches_ba = bf.knnMatch(des2, des1, k=1)

    # 양방향 일치하는 것만 선택
    symmetric = []
    for m_ab in matches_ab:
        if len(m_ab) == 0:
            continue

        query_idx = m_ab[0].queryIdx
        train_idx = m_ab[0].trainIdx

        # B→A 매칭에서 역방향 확인
        for m_ba in matches_ba:
            if len(m_ba) == 0:
                continue

            if m_ba[0].queryIdx == train_idx and m_ba[0].trainIdx == query_idx:
                symmetric.append(m_ab[0])
                break

    return symmetric
```

---

## 6. Homography와 RANSAC

### Homography 개념

```
Homography (호모그래피):
평면 간의 투시 변환을 나타내는 3x3 행렬

┌     ┐   ┌           ┐ ┌   ┐
│ x'  │   │ h11 h12 h13 │ │ x │
│ y'  │ = │ h21 h22 h23 │ │ y │
│  1  │   │ h31 h32 h33 │ │ 1 │
└     ┘   └           ┘ └   ┘

x' = (h11*x + h12*y + h13) / (h31*x + h32*y + h33)
y' = (h21*x + h22*y + h23) / (h31*x + h32*y + h33)

활용:
- 객체 위치 추정
- 이미지 정합
- 파노라마 스티칭
- AR 마커 검출
```

### cv2.findHomography()

```python
import cv2
import numpy as np

def find_object_homography(img1_path, img2_path, min_matches=10):
    """호모그래피로 객체 찾기"""
    img1 = cv2.imread(img1_path)  # 쿼리 (찾을 객체)
    img2 = cv2.imread(img2_path)  # 타겟 (장면)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT 특징점 및 매칭
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"좋은 매칭 수: {len(good_matches)}")

    if len(good_matches) >= min_matches:
        # 매칭된 점 좌표 추출
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 호모그래피 계산 (RANSAC)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is not None:
            # 쿼리 이미지의 코너를 변환
            h, w = gray1.shape
            corners = np.float32([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ]).reshape(-1, 1, 2)

            transformed_corners = cv2.perspectiveTransform(corners, H)

            # 타겟 이미지에 객체 위치 표시
            result = img2.copy()
            cv2.polylines(
                result,
                [np.int32(transformed_corners)],
                True,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )

            # 매칭 시각화
            matches_mask = mask.ravel().tolist()
            draw_params = dict(
                matchColor=(0, 255, 0),
                singlePointColor=None,
                matchesMask=matches_mask,
                flags=2
            )

            match_img = cv2.drawMatches(
                img1, kp1, img2, kp2,
                good_matches, None, **draw_params
            )

            cv2.imshow('Object Detection', result)
            cv2.imshow('Matches', match_img)
            cv2.waitKey(0)

            # 인라이어 비율
            inliers = np.sum(mask)
            print(f"인라이어: {inliers}/{len(good_matches)}")
            print(f"인라이어 비율: {inliers/len(good_matches)*100:.1f}%")

            return H, transformed_corners
    else:
        print(f"매칭 부족: {len(good_matches)} < {min_matches}")
        return None, None

H, corners = find_object_homography('book_cover.jpg', 'scene.jpg')
```

### RANSAC 이해

```
RANSAC (RANdom SAmple Consensus):
아웃라이어가 있는 데이터에서 모델 추정

알고리즘:
1. 최소 샘플 무작위 선택 (호모그래피: 4점)
2. 모델 계산
3. 모든 점에 대해 에러 계산
4. 임계값 이내의 점(인라이어) 수 계산
5. 반복하여 인라이어가 가장 많은 모델 선택
6. 인라이어로 모델 재계산 (옵션)

┌────────────────────────────────────────┐
│  ●  ●  ●  ●  ●                         │
│     ●  ●  ●        ← 인라이어 (직선 근처) │
│        ●  ●  ●                         │
│  ×                                     │
│           ×        ← 아웃라이어          │
│     ×          ×                       │
└────────────────────────────────────────┘

findHomography 파라미터:
- cv2.RANSAC: RANSAC 사용
- ransacReprojThreshold: 인라이어 판정 임계값 (픽셀)
```

```python
import cv2
import numpy as np

def homography_methods_comparison(src_pts, dst_pts):
    """다양한 호모그래피 계산 방법 비교"""

    methods = [
        (0, 'Regular (LS)'),
        (cv2.RANSAC, 'RANSAC'),
        (cv2.LMEDS, 'Least-Median'),
        (cv2.RHO, 'PROSAC'),
    ]

    for method, name in methods:
        try:
            H, mask = cv2.findHomography(
                src_pts, dst_pts,
                method,
                ransacReprojThreshold=5.0
            )

            if H is not None and mask is not None:
                inliers = np.sum(mask)
                print(f"{name}: {inliers}/{len(src_pts)} inliers")
            else:
                print(f"{name}: Failed")
        except Exception as e:
            print(f"{name}: Error - {e}")
```

---

## 7. 이미지 스티칭 기초

### 간단한 파노라마

```python
import cv2
import numpy as np

def simple_panorama(img1_path, img2_path):
    """간단한 파노라마 스티칭"""
    img1 = cv2.imread(img1_path)  # 왼쪽 이미지
    img2 = cv2.imread(img2_path)  # 오른쪽 이미지

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 특징점 검출 및 매칭
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print(f"매칭 수: {len(good)}")

    if len(good) < 4:
        print("매칭이 부족합니다.")
        return None

    # 호모그래피 계산
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("호모그래피 계산 실패")
        return None

    # 이미지 워핑
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 결과 이미지 크기 계산
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners1_transformed = cv2.perspectiveTransform(corners1, H)

    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)

    all_corners = np.concatenate([corners1_transformed, corners2], axis=0)

    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

    # 이동 변환
    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ], dtype=np.float32)

    # 이미지 1 워핑
    result_width = x_max - x_min
    result_height = y_max - y_min

    warped1 = cv2.warpPerspective(
        img1,
        translation @ H,
        (result_width, result_height)
    )

    # 이미지 2 복사
    warped1[-y_min:-y_min+h2, -x_min:-x_min+w2] = img2

    cv2.imshow('Panorama', warped1)
    cv2.waitKey(0)

    return warped1

panorama = simple_panorama('left.jpg', 'right.jpg')
```

### OpenCV Stitcher 사용

```python
import cv2
import numpy as np

def opencv_stitcher(image_paths):
    """OpenCV Stitcher 클래스 사용"""
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)

    if len(images) < 2:
        print("최소 2개의 이미지가 필요합니다.")
        return None

    # Stitcher 생성
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    # 또는: cv2.Stitcher_SCANS (문서 스캔용)

    # 스티칭 수행
    status, result = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        print("스티칭 성공!")
        cv2.imshow('Stitched', result)
        cv2.waitKey(0)
        return result
    elif status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        print("더 많은 이미지가 필요합니다.")
    elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        print("호모그래피 추정 실패")
    elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
        print("카메라 파라미터 조정 실패")

    return None

# 사용 예
image_files = ['pano1.jpg', 'pano2.jpg', 'pano3.jpg']
result = opencv_stitcher(image_files)
```

---

## 8. 연습 문제

### 문제 1: 최적 매칭 파라미터 찾기

다양한 ratio threshold 값을 테스트하여 최적의 값을 찾으세요.

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_optimal_ratio(img1_path, img2_path):
    """최적의 ratio threshold 찾기"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    ratios = np.arange(0.5, 1.0, 0.05)
    results = []

    for ratio in ratios:
        good = [m for m, n in matches if m.distance < ratio * n.distance]
        results.append(len(good))

    # 그래프
    plt.figure(figsize=(10, 5))
    plt.plot(ratios, results, 'b-o')
    plt.xlabel('Ratio Threshold')
    plt.ylabel('Number of Matches')
    plt.title('Ratio Threshold vs Match Count')
    plt.grid(True)
    plt.show()

    # 기울기 변화 분석
    gradients = np.gradient(results)
    optimal_idx = np.argmax(np.abs(gradients))
    optimal_ratio = ratios[optimal_idx]

    print(f"권장 ratio threshold: {optimal_ratio:.2f}")

    return optimal_ratio

optimal = find_optimal_ratio('query.jpg', 'train.jpg')
```

</details>

### 문제 2: 다중 객체 검출

한 장면에서 같은 객체가 여러 개 있을 때 모두 검출하세요.

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def detect_multiple_objects(template_path, scene_path, threshold=10):
    """여러 개의 동일 객체 검출"""
    template = cv2.imread(template_path)
    scene = cv2.imread(scene_path)

    gray_t = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_s = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp_t, des_t = sift.detectAndCompute(gray_t, None)
    kp_s, des_s = sift.detectAndCompute(gray_s, None)

    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(des_t, des_s, k=2)

    # Ratio test
    good_matches = []
    for m, n in all_matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < threshold:
        print("매칭 부족")
        return []

    # 클러스터링으로 여러 인스턴스 찾기
    scene_pts = np.array([kp_s[m.trainIdx].pt for m in good_matches])

    # K-means 클러스터링
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = min(5, len(good_matches) // threshold)  # 최대 5개 객체

    if k < 1:
        k = 1

    _, labels, centers = cv2.kmeans(
        np.float32(scene_pts),
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    result = scene.copy()
    detected = []

    for cluster_id in range(k):
        cluster_mask = labels.ravel() == cluster_id
        cluster_matches = [m for m, is_in in zip(good_matches, cluster_mask) if is_in]

        if len(cluster_matches) >= threshold // 2:
            # 각 클러스터에서 호모그래피 계산
            src_pts = np.float32([kp_t[m.queryIdx].pt for m in cluster_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_s[m.trainIdx].pt for m in cluster_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                h, w = gray_t.shape
                corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(corners, H)

                cv2.polylines(result, [np.int32(transformed)], True, (0, 255, 0), 3)
                detected.append(transformed)

    print(f"검출된 객체 수: {len(detected)}")
    cv2.imshow('Multiple Objects', result)
    cv2.waitKey(0)

    return detected

detect_multiple_objects('coin.jpg', 'coins.jpg')
```

</details>

### 문제 3: 실시간 객체 추적

웹캠에서 템플릿 객체를 실시간으로 추적하세요.

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def realtime_object_tracking(template_path):
    """실시간 객체 추적"""
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    h, w = template.shape

    # ORB 사용 (빠름)
    orb = cv2.ORB_create(nfeatures=500)
    kp_t, des_t = orb.detectAndCompute(template, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_f, des_f = orb.detectAndCompute(gray, None)

        if des_f is not None and len(des_f) > 10:
            matches = bf.knnMatch(des_t, des_f, k=2)

            # Ratio test
            good = []
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            if len(good) >= 10:
                src_pts = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if H is not None:
                    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    transformed = cv2.perspectiveTransform(corners, H)
                    cv2.polylines(frame, [np.int32(transformed)], True, (0, 255, 0), 3)

                    # 매칭 수 표시
                    cv2.putText(frame, f'Matches: {len(good)}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# realtime_object_tracking('logo.jpg')
```

</details>

### 추천 문제

| 난이도 | 주제 | 설명 |
|--------|------|------|
| ⭐ | 기본 매칭 | 두 이미지 간 특징점 매칭 |
| ⭐⭐ | 필터링 | Ratio test, 거리 필터링 |
| ⭐⭐ | 객체 검출 | 호모그래피로 객체 찾기 |
| ⭐⭐⭐ | 파노라마 | 2장 이상 이미지 스티칭 |
| ⭐⭐⭐ | 실시간 추적 | 웹캠으로 객체 추적 |

---

## 다음 단계

- [15_Object_Detection_Basics.md](./15_Object_Detection_Basics.md) - Template Matching, Haar Cascade, HOG+SVM

---

## 참고 자료

- [OpenCV Feature Matching](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
- [Homography Tutorial](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html)
- [Image Stitching](https://docs.opencv.org/4.x/d8/d19/tutorial_stitcher.html)
