# 도형 분석 (Shape Analysis)

## 개요

윤곽선에서 추출한 도형의 특성을 분석하고 분류하는 방법을 학습합니다. 모멘트, 무게중심, 경계 도형, 컨벡스 헐, 형상 매칭 등 다양한 도형 분석 기법을 다룹니다.

---

## 목차

1. [이미지 모멘트](#1-이미지-모멘트)
2. [무게중심 계산](#2-무게중심-계산)
3. [경계 사각형](#3-경계-사각형)
4. [최소 외접 도형](#4-최소-외접-도형)
5. [컨벡스 헐](#5-컨벡스-헐)
6. [형상 매칭](#6-형상-매칭)
7. [도형 분류 시스템](#7-도형-분류-시스템)
8. [연습 문제](#8-연습-문제)

---

## 1. 이미지 모멘트

### 모멘트란?

```
이미지 모멘트 (Image Moments):
이미지 픽셀 값의 가중 평균으로 계산되는 특징값

수학적 정의:
Mij = Σ Σ x^i × y^j × I(x, y)

- M00: 면적 (0차 모멘트)
- M10, M01: 1차 모멘트 (무게중심 계산용)
- M20, M02, M11: 2차 모멘트 (방향, 분산)

활용:
- 면적, 둘레 계산
- 무게중심 (Centroid)
- 방향 (Orientation)
- 타원 피팅
- 휴 모멘트 (Hu Moments) - 불변 특징
```

### cv2.moments() 함수

```python
import cv2
import numpy as np

def calculate_moments(image_path):
    """이미지 모멘트 계산"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for i, contour in enumerate(contours):
        # 모멘트 계산
        M = cv2.moments(contour)

        print(f"윤곽선 {i}:")
        print(f"  M00 (면적): {M['m00']:.0f}")
        print(f"  M10: {M['m10']:.0f}")
        print(f"  M01: {M['m01']:.0f}")

        # 무게중심
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            print(f"  무게중심: ({cx}, {cy})")

        # 중심 모멘트 (Central Moments)
        print(f"  mu20: {M['mu20']:.0f}")
        print(f"  mu11: {M['mu11']:.0f}")
        print(f"  mu02: {M['mu02']:.0f}")

        # 정규화된 중심 모멘트
        print(f"  nu20: {M['nu20']:.6f}")
        print(f"  nu11: {M['nu11']:.6f}")
        print(f"  nu02: {M['nu02']:.6f}")

calculate_moments('shapes.jpg')
```

### 모멘트 종류

```
공간 모멘트 (Spatial Moments):
m00, m10, m01, m20, m11, m02, m30, m21, m12, m03

┌─────────────────────────────────────────────────────────┐
│  m00 = Σ I(x,y)           → 면적 (흰색 픽셀 수)        │
│  m10 = Σ x × I(x,y)       → x 좌표의 총합              │
│  m01 = Σ y × I(x,y)       → y 좌표의 총합              │
└─────────────────────────────────────────────────────────┘

중심 모멘트 (Central Moments):
mu20, mu11, mu02, mu30, mu21, mu12, mu03

┌─────────────────────────────────────────────────────────┐
│  무게중심 기준으로 계산                                  │
│  mu20 = Σ (x - cx)² × I(x,y)                           │
│  → 이동 불변 (Translation Invariant)                    │
└─────────────────────────────────────────────────────────┘

정규화된 중심 모멘트 (Normalized Central Moments):
nu20, nu11, nu02, nu30, nu21, nu12, nu03

┌─────────────────────────────────────────────────────────┐
│  nuij = muij / m00^((i+j)/2 + 1)                       │
│  → 이동 + 크기 불변 (Scale Invariant)                   │
└─────────────────────────────────────────────────────────┘
```

### 휴 모멘트 (Hu Moments)

```python
import cv2
import numpy as np

def hu_moments_analysis(contour):
    """휴 모멘트 계산 및 분석"""
    # 일반 모멘트
    M = cv2.moments(contour)

    # 휴 모멘트 (7개의 불변 특징)
    huMoments = cv2.HuMoments(M)

    # 로그 스케일 변환 (비교 용이)
    huMoments_log = -np.sign(huMoments) * np.log10(np.abs(huMoments) + 1e-10)

    print("휴 모멘트:")
    for i, h in enumerate(huMoments_log.flatten()):
        print(f"  h{i+1}: {h:.4f}")

    return huMoments

def compare_shapes_hu(contour1, contour2):
    """휴 모멘트로 두 도형 비교"""
    hu1 = cv2.HuMoments(cv2.moments(contour1)).flatten()
    hu2 = cv2.HuMoments(cv2.moments(contour2)).flatten()

    # 로그 스케일 변환
    hu1_log = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-10)
    hu2_log = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-10)

    # 유클리드 거리
    distance = np.linalg.norm(hu1_log - hu2_log)

    return distance

# 사용 예
img = cv2.imread('shapes.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) >= 2:
    dist = compare_shapes_hu(contours[0], contours[1])
    print(f"두 도형의 유사도 거리: {dist:.4f}")
    # 거리가 작을수록 비슷한 모양
```

---

## 2. 무게중심 계산

### 무게중심 공식

```
무게중심 (Centroid):
도형의 질량 중심점

cx = M10 / M00
cy = M01 / M00

          (x1,y1)
             *
            / \
           /   \
          /  •  \    ← (cx, cy) 무게중심
         /       \
        *---------*
    (x2,y2)    (x3,y3)

특징:
- 도형 내부에 항상 위치
- 회전, 크기 변환에 상관없이 상대적 위치 유지
```

### 무게중심 계산 및 시각화

```python
import cv2
import numpy as np

def find_centroids(image_path):
    """모든 윤곽선의 무게중심 찾기"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()
    centroids = []

    for contour in contours:
        # 모멘트 계산
        M = cv2.moments(contour)

        # 면적이 0이 아닌 경우만
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append((cx, cy))

            # 윤곽선 그리기
            cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)

            # 무게중심 표시
            cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(result, f'({cx},{cy})', (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow('Centroids', result)
    cv2.waitKey(0)

    return centroids

centroids = find_centroids('shapes.jpg')
print(f"무게중심 좌표: {centroids}")
```

### 도형 방향 계산

```python
import cv2
import numpy as np

def calculate_orientation(contour):
    """도형의 주축 방향 계산"""
    M = cv2.moments(contour)

    if M['m00'] == 0:
        return None, None

    # 무게중심
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']

    # 2차 중심 모멘트로 방향 계산
    # theta = 0.5 * arctan(2 * mu11 / (mu20 - mu02))
    if (M['mu20'] - M['mu02']) != 0:
        theta = 0.5 * np.arctan2(2 * M['mu11'], (M['mu20'] - M['mu02']))
    else:
        theta = 0

    return (cx, cy), theta

def draw_orientation(image, contour):
    """도형의 방향을 화살표로 표시"""
    result = image.copy()

    center, theta = calculate_orientation(contour)
    if center is None:
        return result

    cx, cy = int(center[0]), int(center[1])

    # 주축 방향 화살표
    length = 50
    dx = int(length * np.cos(theta))
    dy = int(length * np.sin(theta))

    # 윤곽선
    cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)

    # 무게중심
    cv2.circle(result, (cx, cy), 5, (255, 0, 0), -1)

    # 방향 화살표
    cv2.arrowedLine(result, (cx, cy), (cx + dx, cy + dy),
                    (0, 0, 255), 2, tipLength=0.3)

    # 각도 표시
    angle_deg = np.degrees(theta)
    cv2.putText(result, f'{angle_deg:.1f} deg', (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return result

# 사용 예
img = cv2.imread('elongated_shape.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    result = draw_orientation(img, contours[0])
    cv2.imshow('Orientation', result)
    cv2.waitKey(0)
```

---

## 3. 경계 사각형

### cv2.boundingRect()

```
경계 사각형 (Bounding Rectangle):
윤곽선을 완전히 감싸는 최소 수직 사각형

    ┌───────────────┐
    │   ╱╲          │
    │  ╱  ╲         │  (x, y): 좌상단
    │ ╱    ╲        │  w: 너비
    │ ╲    ╱        │  h: 높이
    │  ╲  ╱         │
    │   ╲╱          │
    └───────────────┘
```

```python
import cv2
import numpy as np

def bounding_rect_example(image_path):
    """경계 사각형 예제"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()

    for contour in contours:
        # 경계 사각형
        x, y, w, h = cv2.boundingRect(contour)

        # 사각형 그리기
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 정보 표시
        aspect_ratio = w / h
        extent = cv2.contourArea(contour) / (w * h)

        print(f"위치: ({x}, {y})")
        print(f"크기: {w} x {h}")
        print(f"가로세로비: {aspect_ratio:.2f}")
        print(f"Extent: {extent:.2f}")  # 사각형 대비 실제 면적 비율

    cv2.imshow('Bounding Rectangle', result)
    cv2.waitKey(0)

bounding_rect_example('shapes.jpg')
```

### cv2.minAreaRect() - 회전된 경계 사각형

```
회전된 경계 사각형:
도형을 가장 작게 감싸는 회전된 사각형

            ╱╲
           ╱  ╲
          ╱    ╲
         ╱──────╲
        ╱        ╲
       ╲──────────╱

반환값: ((cx, cy), (w, h), angle)
- (cx, cy): 중심점
- (w, h): 너비, 높이
- angle: 회전 각도
```

```python
import cv2
import numpy as np

def min_area_rect_example(image_path):
    """최소 면적 회전 사각형"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()

    for contour in contours:
        # 최소 면적 회전 사각형
        rect = cv2.minAreaRect(contour)
        center, size, angle = rect

        print(f"중심: {center}")
        print(f"크기: {size}")
        print(f"각도: {angle:.1f}")

        # 꼭짓점 좌표 계산
        box = cv2.boxPoints(rect)
        box = np.int_(box)

        # 사각형 그리기
        cv2.drawContours(result, [box], 0, (0, 255, 0), 2)

        # 중심점 표시
        cv2.circle(result, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)

    cv2.imshow('Min Area Rect', result)
    cv2.waitKey(0)

min_area_rect_example('rotated_shapes.jpg')
```

---

## 4. 최소 외접 도형

### cv2.minEnclosingCircle()

```python
import cv2
import numpy as np

def min_enclosing_circle(image_path):
    """최소 외접원"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()

    for contour in contours:
        # 최소 외접원
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # 원 그리기
        cv2.circle(result, center, radius, (0, 255, 0), 2)
        cv2.circle(result, center, 3, (0, 0, 255), -1)

        # 면적 비율 (원형도 간접 측정)
        contour_area = cv2.contourArea(contour)
        circle_area = np.pi * radius * radius
        fill_ratio = contour_area / circle_area

        print(f"중심: {center}, 반지름: {radius}")
        print(f"채움 비율: {fill_ratio:.2f}")

    cv2.imshow('Min Enclosing Circle', result)
    cv2.waitKey(0)

min_enclosing_circle('shapes.jpg')
```

### cv2.fitEllipse()

```python
import cv2
import numpy as np

def fit_ellipse_example(image_path):
    """타원 피팅"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()

    for contour in contours:
        # fitEllipse는 최소 5개 점 필요
        if len(contour) >= 5:
            # 타원 피팅
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse

            print(f"중심: {center}")
            print(f"축 길이: {axes}")  # (장축, 단축)
            print(f"각도: {angle:.1f}")

            # 타원 그리기
            cv2.ellipse(result, ellipse, (0, 255, 0), 2)

            # 중심점
            cv2.circle(result, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)

    cv2.imshow('Fitted Ellipse', result)
    cv2.waitKey(0)

fit_ellipse_example('ellipse_shapes.jpg')
```

### 경계 도형 비교

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_bounding_shapes(image_path):
    """다양한 경계 도형 비교"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        # 1. 경계 사각형 (파랑)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 2. 회전된 경계 사각형 (녹색)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int_(box)
        cv2.drawContours(result, [box], 0, (0, 255, 0), 2)

        # 3. 최소 외접원 (빨강)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        cv2.circle(result, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)

        # 4. 타원 피팅 (노랑)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(result, ellipse, (0, 255, 255), 2)

    # 범례
    cv2.putText(result, 'Blue: Bounding Rect', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(result, 'Green: Min Area Rect', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(result, 'Red: Min Enclosing Circle', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(result, 'Yellow: Fitted Ellipse', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow('Bounding Shapes', result)
    cv2.waitKey(0)

compare_bounding_shapes('shapes.jpg')
```

---

## 5. 컨벡스 헐

### cv2.convexHull()

```
컨벡스 헐 (Convex Hull):
점 집합을 감싸는 가장 작은 볼록 다각형
→ 고무줄로 감싼 모양

       •  •  •
     •        •
   •    원본    •
     •        •
   •  •    •  •

       ┌──────┐
      │       │
     │  컨벡스 │
    │   헐    │
     └────────┘
```

```python
import cv2
import numpy as np

def convex_hull_example(image_path):
    """컨벡스 헐 예제"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()

    for contour in contours:
        # 컨벡스 헐
        hull = cv2.convexHull(contour)

        # 원본 윤곽선 (녹색)
        cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)

        # 컨벡스 헐 (빨강)
        cv2.drawContours(result, [hull], 0, (0, 0, 255), 2)

        # 면적 비교 (Solidity)
        contour_area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / hull_area if hull_area > 0 else 0

        print(f"윤곽선 면적: {contour_area:.0f}")
        print(f"컨벡스 헐 면적: {hull_area:.0f}")
        print(f"Solidity: {solidity:.2f}")
        # Solidity가 1에 가까울수록 볼록한 도형

    cv2.imshow('Convex Hull', result)
    cv2.waitKey(0)

convex_hull_example('star_shape.jpg')
```

### 컨벡시티 결함 (Convexity Defects)

```
컨벡시티 결함:
윤곽선과 컨벡스 헐 사이의 오목한 부분
→ 손가락 검출에 활용

            ╱╲
           ╱  ╲
        start  end
          ╲  ╱
           ╲╱ ← far (가장 깊은 점)

반환값: [start, end, far, depth]
- start: 시작점 인덱스
- end: 끝점 인덱스
- far: 가장 깊은 점 인덱스
- depth: 깊이 (256으로 나눠서 사용)
```

```python
import cv2
import numpy as np

def convexity_defects_example(image_path):
    """컨벡시티 결함 검출 (손가락 세기)"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return

    # 가장 큰 윤곽선 선택
    contour = max(contours, key=cv2.contourArea)

    result = img.copy()

    # 컨벡스 헐 (인덱스 반환)
    hull = cv2.convexHull(contour, returnPoints=False)

    # 컨벡시티 결함
    defects = cv2.convexityDefects(contour, hull)

    if defects is None:
        return

    # 결함 분석
    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]

        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        # 깊이 변환 (256으로 나눔)
        depth = d / 256.0

        # 깊이가 일정 이상인 경우만 (손가락 사이 공간)
        if depth > 20:
            finger_count += 1

            # 시각화
            cv2.circle(result, far, 5, (0, 0, 255), -1)
            cv2.line(result, start, far, (0, 255, 0), 2)
            cv2.line(result, far, end, (0, 255, 0), 2)

    # 손가락 개수 = 결함 개수 + 1
    print(f"손가락 개수: {finger_count + 1}")

    # 윤곽선
    cv2.drawContours(result, [contour], 0, (255, 0, 0), 2)

    cv2.imshow('Convexity Defects', result)
    cv2.waitKey(0)

convexity_defects_example('hand.jpg')
```

---

## 6. 형상 매칭

### cv2.matchShapes()

```
형상 매칭:
두 윤곽선의 유사도 비교 (휴 모멘트 기반)

cv2.matchShapes(contour1, contour2, method, parameter)

method:
- cv2.CONTOURS_MATCH_I1: Σ|1/mA - 1/mB|
- cv2.CONTOURS_MATCH_I2: Σ|mA - mB|
- cv2.CONTOURS_MATCH_I3: Σ|mA - mB| / |mA|

반환값: 작을수록 유사함 (0 = 동일)
```

```python
import cv2
import numpy as np

def shape_matching_example():
    """형상 매칭 예제"""
    # 템플릿 도형 생성
    template = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(template, (100, 100), 80, 255, -1)

    # 테스트 도형들
    shapes = {
        'circle': cv2.circle(np.zeros((200, 200), dtype=np.uint8),
                             (100, 100), 60, 255, -1),
        'ellipse': cv2.ellipse(np.zeros((200, 200), dtype=np.uint8),
                               (100, 100), (80, 50), 0, 0, 360, 255, -1),
        'square': cv2.rectangle(np.zeros((200, 200), dtype=np.uint8),
                                (30, 30), (170, 170), 255, -1),
    }

    # 템플릿 윤곽선
    contours_t, _ = cv2.findContours(template, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    template_contour = contours_t[0]

    print("형상 매칭 결과 (낮을수록 유사):")
    for name, shape in shapes.items():
        contours_s, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        if contours_s:
            match = cv2.matchShapes(template_contour, contours_s[0],
                                     cv2.CONTOURS_MATCH_I1, 0)
            print(f"  {name}: {match:.4f}")

shape_matching_example()
```

### 템플릿 기반 도형 검출

```python
import cv2
import numpy as np

def find_similar_shapes(image_path, template_path, threshold=0.1):
    """템플릿과 유사한 도형 찾기"""
    # 템플릿 로드
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    _, template_bin = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
    template_contours, _ = cv2.findContours(
        template_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    template_contour = max(template_contours, key=cv2.contourArea)

    # 대상 이미지
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()
    similar_shapes = []

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        # 형상 매칭
        match = cv2.matchShapes(
            template_contour, contour, cv2.CONTOURS_MATCH_I1, 0
        )

        if match < threshold:
            similar_shapes.append(contour)
            cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)

            # 매칭 점수 표시
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(result, f'{match:.3f}', (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    print(f"유사한 도형 수: {len(similar_shapes)}")
    cv2.imshow('Similar Shapes', result)
    cv2.waitKey(0)

    return similar_shapes

# 사용 예
find_similar_shapes('shapes.jpg', 'template_circle.jpg', threshold=0.15)
```

---

## 7. 도형 분류 시스템

### 종합 도형 분류기

```python
import cv2
import numpy as np

class ShapeClassifier:
    """도형 분류기"""

    def __init__(self):
        self.shape_names = {
            3: 'Triangle',
            4: 'Rectangle',
            5: 'Pentagon',
            6: 'Hexagon'
        }

    def classify(self, contour):
        """윤곽선으로 도형 분류"""
        # 기본 속성
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < 100 or perimeter == 0:
            return None, {}

        # 다각형 근사화
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        vertices = len(approx)

        # 경계 사각형
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h

        # 원형도
        circularity = 4 * np.pi * area / (perimeter ** 2)

        # 솔리디티
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # 분류
        properties = {
            'vertices': vertices,
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'solidity': solidity
        }

        # 도형 판별
        if vertices == 3:
            shape = 'Triangle'
        elif vertices == 4:
            if 0.95 <= aspect_ratio <= 1.05:
                shape = 'Square'
            else:
                shape = 'Rectangle'
        elif vertices == 5:
            shape = 'Pentagon'
        elif vertices == 6:
            shape = 'Hexagon'
        elif circularity > 0.85:
            shape = 'Circle'
        elif 0.6 < circularity < 0.85 and solidity > 0.9:
            shape = 'Ellipse'
        elif solidity < 0.7:
            shape = 'Star' if vertices > 6 else 'Irregular'
        else:
            shape = f'Polygon-{vertices}'

        return shape, properties

    def process_image(self, image_path):
        """이미지의 모든 도형 분류"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        result = img.copy()
        classifications = []

        for contour in contours:
            shape, props = self.classify(contour)
            if shape is None:
                continue

            classifications.append((shape, props))

            # 무게중심
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # 윤곽선 그리기
                color = self._get_shape_color(shape)
                cv2.drawContours(result, [contour], 0, color, 2)

                # 라벨
                cv2.putText(result, shape, (cx - 30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Classified Shapes', result)
        cv2.waitKey(0)

        return classifications

    def _get_shape_color(self, shape):
        """도형별 색상"""
        colors = {
            'Circle': (0, 0, 255),      # 빨강
            'Ellipse': (0, 128, 255),   # 주황
            'Triangle': (0, 255, 0),    # 녹색
            'Square': (255, 0, 0),      # 파랑
            'Rectangle': (255, 128, 0), # 하늘색
            'Pentagon': (255, 0, 255),  # 보라
            'Hexagon': (128, 0, 128),   # 자주
            'Star': (0, 255, 255),      # 노랑
        }
        return colors.get(shape, (128, 128, 128))

# 사용 예
classifier = ShapeClassifier()
results = classifier.process_image('various_shapes.jpg')

print("\n분류 결과:")
for shape, props in results:
    print(f"  {shape}:")
    print(f"    면적: {props['area']:.0f}")
    print(f"    원형도: {props['circularity']:.2f}")
    print(f"    꼭짓점: {props['vertices']}")
```

### 실시간 도형 검출

```python
import cv2
import numpy as np

def realtime_shape_detection():
    """웹캠으로 실시간 도형 검출"""
    classifier = ShapeClassifier()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 적응형 이진화
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # 모폴로지 연산
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            shape, props = classifier.classify(contour)
            if shape is None:
                continue

            # 무게중심
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # 그리기
                cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
                cv2.putText(frame, shape, (cx - 30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('Shape Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# realtime_shape_detection()
```

---

## 8. 연습 문제

### 문제 1: 도형 정렬

이미지에서 검출된 도형들을 면적 기준으로 정렬하고 순위를 표시하세요.

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def rank_shapes_by_area(image_path):
    """도형을 면적순으로 정렬"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 면적과 함께 저장
    contour_areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            contour_areas.append((contour, area))

    # 면적 기준 정렬 (내림차순)
    contour_areas.sort(key=lambda x: x[1], reverse=True)

    result = img.copy()

    for rank, (contour, area) in enumerate(contour_areas, 1):
        # 무게중심
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # 그리기
        cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)
        cv2.putText(result, f'#{rank}', (cx - 15, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(result, f'{area:.0f}', (cx - 25, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    cv2.imshow('Ranked Shapes', result)
    cv2.waitKey(0)

rank_shapes_by_area('shapes.jpg')
```

</details>

### 문제 2: 특정 비율의 사각형 찾기

가로세로 비율이 2:1인 사각형만 검출하세요.

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def find_2to1_rectangles(image_path, tolerance=0.2):
    """2:1 비율 사각형 찾기"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()
    found = []

    target_ratio = 2.0

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        # 다각형 근사화
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # 4개 꼭짓점인지 확인
        if len(approx) != 4:
            continue

        # 경계 사각형으로 비율 확인
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h)

        # 2:1 비율 확인 (허용 오차 포함)
        if abs(aspect_ratio - target_ratio) < tolerance:
            found.append(contour)
            cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)

            # 비율 표시
            cv2.putText(result, f'{aspect_ratio:.2f}:1', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    print(f"2:1 사각형 수: {len(found)}")
    cv2.imshow('2:1 Rectangles', result)
    cv2.waitKey(0)

find_2to1_rectangles('rectangles.jpg')
```

</details>

### 문제 3: 가장 원형에 가까운 도형 찾기

이미지에서 원형도가 가장 높은 도형을 찾아 표시하세요.

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def find_most_circular(image_path):
    """가장 원형에 가까운 도형 찾기"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_circularity = 0
    best_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < 100 or perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)

        if circularity > best_circularity:
            best_circularity = circularity
            best_contour = contour

    result = img.copy()

    if best_contour is not None:
        # 모든 윤곽선 (회색)
        cv2.drawContours(result, contours, -1, (128, 128, 128), 1)

        # 가장 원형인 것 (녹색)
        cv2.drawContours(result, [best_contour], 0, (0, 255, 0), 3)

        # 정보 표시
        M = cv2.moments(best_contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(result, f'Circularity: {best_circularity:.3f}',
                    (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    print(f"최대 원형도: {best_circularity:.4f}")
    cv2.imshow('Most Circular', result)
    cv2.waitKey(0)

find_most_circular('shapes.jpg')
```

</details>

### 추천 문제

| 난이도 | 주제 | 설명 |
|--------|------|------|
| ⭐ | 무게중심 | 모든 도형의 무게중심 표시 |
| ⭐⭐ | 방향 | 긴 도형의 주축 방향 표시 |
| ⭐⭐ | 유사도 | matchShapes로 도형 분류 |
| ⭐⭐⭐ | 손가락 세기 | 컨벡시티 결함 활용 |
| ⭐⭐⭐ | 카드 인식 | 사각형 검출 + 분류 |

---

## 다음 단계

- [11_Hough_Transform.md](./11_Hough_Transform.md) - HoughLines, HoughLinesP, HoughCircles

---

## 참고 자료

- [OpenCV Contour Features](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)
- [Contour Properties](https://docs.opencv.org/4.x/d1/d32/tutorial_py_contour_properties.html)
- [Image Moments](https://docs.opencv.org/4.x/d8/d23/classcv_1_1Moments.html)
