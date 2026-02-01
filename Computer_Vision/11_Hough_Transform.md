# 허프 변환 (Hough Transform)

## 개요

허프 변환은 이미지에서 직선, 원 등의 기하학적 형태를 검출하는 알고리즘입니다. 엣지 검출 결과에서 특정 모양을 찾는 데 사용되며, 차선 검출, 동전 검출 등 다양한 응용 분야가 있습니다.

---

## 목차

1. [허프 변환 개념](#1-허프-변환-개념)
2. [허프 직선 변환](#2-허프-직선-변환)
3. [확률적 허프 직선 변환](#3-확률적-허프-직선-변환)
4. [허프 원 변환](#4-허프-원-변환)
5. [파라미터 튜닝 전략](#5-파라미터-튜닝-전략)
6. [차선 검출 기초](#6-차선-검출-기초)
7. [연습 문제](#7-연습-문제)

---

## 1. 허프 변환 개념

### 허프 공간 (Hough Space)

```
기본 아이디어:
이미지 공간의 점 → 허프 공간의 곡선
이미지 공간의 직선 → 허프 공간의 점

이미지 공간 (x, y)              허프 공간 (ρ, θ)
┌─────────────────┐            ┌─────────────────┐
│                 │            │                 │
│    •            │            │      ╱╲         │
│      ╲          │    ──▶     │     ╱  ╲        │
│        ╲        │            │    ╱ •  ╲       │
│          •      │            │   ╱      ╲      │
│                 │            │                 │
└─────────────────┘            └─────────────────┘
직선 위의 점들                   점 하나로 표현

직선의 표현:
y = mx + b  (기울기, y절편) → 수직선 표현 불가
ρ = x·cos(θ) + y·sin(θ)    → 극좌표 표현 (선호)

ρ: 원점에서 직선까지의 수직 거리
θ: 수직선과 x축이 이루는 각도
```

### 허프 변환 과정

```
1. 엣지 검출 (Canny 등)
         │
         ▼
2. 각 엣지 점에 대해 가능한 모든 직선 계산
   (θ를 0°~180° 변화시키며 ρ 계산)
         │
         ▼
3. 누적 배열(Accumulator)에 투표
         │
         ▼
4. 임계값 이상의 투표를 받은 점 = 직선

누적 배열 시각화:
        θ
      0° ────────────────────▶ 180°
    ρ │  ·  ·  ·  ·  ·  ·  ·  ·
  -max│  ·  ·  ★  ·  ·  ·  ·  ·   ★: 많은 투표
      │  ·  ·  ·  ·  ·  ★  ·  ·      = 직선 존재
      │  ·  ·  ·  ·  ·  ·  ·  ·
   max│  ·  ·  ·  ·  ·  ·  ·  ·
      ▼
```

### 간단한 예제

```python
import cv2
import numpy as np

# 허프 변환 시각화
def visualize_hough_space(image_path):
    """허프 공간 시각화"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 50, 150)

    # 허프 직선 변환 (누적 배열 반환)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    # 시각화
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # 직선 그리기 (양 방향으로 길게)
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Edges', edges)
    cv2.imshow('Hough Lines', result)
    cv2.waitKey(0)

visualize_hough_space('lines.jpg')
```

---

## 2. 허프 직선 변환

### cv2.HoughLines() 함수

```python
lines = cv2.HoughLines(image, rho, theta, threshold)
```

| 파라미터 | 설명 |
|----------|------|
| image | 입력 이미지 (8비트, 단일 채널, 이진화된 엣지 이미지) |
| rho | ρ 해상도 (픽셀 단위, 보통 1) |
| theta | θ 해상도 (라디안 단위, 보통 np.pi/180) |
| threshold | 직선으로 인정할 최소 투표 수 |
| lines | 검출된 직선 [(ρ, θ), ...] |

### 기본 사용법

```python
import cv2
import numpy as np

def hough_lines_example(image_path):
    """표준 허프 직선 검출"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 엣지 검출
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 허프 직선 변환
    lines = cv2.HoughLines(
        edges,
        rho=1,              # ρ 해상도: 1 픽셀
        theta=np.pi/180,    # θ 해상도: 1도
        threshold=100       # 최소 투표 수
    )

    result = img.copy()

    if lines is not None:
        print(f"검출된 직선 수: {len(lines)}")

        for line in lines:
            rho, theta = line[0]

            # 극좌표 → 직교좌표 변환
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # 직선 그리기 (무한 직선)
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Original', img)
    cv2.imshow('Edges', edges)
    cv2.imshow('Hough Lines', result)
    cv2.waitKey(0)

hough_lines_example('building.jpg')
```

### 수평선/수직선만 검출

```python
import cv2
import numpy as np

def detect_horizontal_vertical_lines(image_path):
    """수평선과 수직선만 검출"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    result = img.copy()
    horizontal = []
    vertical = []

    if lines is not None:
        for line in lines:
            rho, theta = line[0]

            # 각도로 분류 (허용 오차 5도)
            angle_deg = np.degrees(theta)

            if 85 < angle_deg < 95:  # 수직선 (θ ≈ 90°)
                vertical.append((rho, theta))
                color = (255, 0, 0)  # 파랑
            elif angle_deg < 5 or angle_deg > 175:  # 수평선 (θ ≈ 0° 또는 180°)
                horizontal.append((rho, theta))
                color = (0, 255, 0)  # 녹색
            else:
                continue

            # 직선 그리기
            a = np.cos(theta)
            b = np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
            x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), color, 2)

    print(f"수평선: {len(horizontal)}개")
    print(f"수직선: {len(vertical)}개")

    cv2.imshow('H/V Lines', result)
    cv2.waitKey(0)

detect_horizontal_vertical_lines('grid.jpg')
```

---

## 3. 확률적 허프 직선 변환

### cv2.HoughLinesP() 함수

```
표준 허프 vs 확률적 허프:

표준 허프 (HoughLines):
- 무한 직선 반환 (ρ, θ)
- 모든 점 검사
- 느림, 정확

확률적 허프 (HoughLinesP):
- 선분 반환 (x1, y1, x2, y2)
- 무작위 점 샘플링
- 빠름, 실용적
```

```python
lines = cv2.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap)
```

| 파라미터 | 설명 |
|----------|------|
| image | 입력 엣지 이미지 |
| rho | ρ 해상도 |
| theta | θ 해상도 |
| threshold | 최소 투표 수 |
| minLineLength | 최소 선분 길이 |
| maxLineGap | 선분 사이 최대 허용 간격 |
| lines | 검출된 선분 [(x1, y1, x2, y2), ...] |

### 기본 사용법

```python
import cv2
import numpy as np

def hough_lines_p_example(image_path):
    """확률적 허프 직선 검출"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 확률적 허프 변환
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,    # 최소 50픽셀 이상
        maxLineGap=10        # 10픽셀 이내 간격은 연결
    )

    result = img.copy()

    if lines is not None:
        print(f"검출된 선분 수: {len(lines)}")

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 선분 끝점 표시
            cv2.circle(result, (x1, y1), 5, (255, 0, 0), -1)
            cv2.circle(result, (x2, y2), 5, (0, 0, 255), -1)

    cv2.imshow('HoughLinesP', result)
    cv2.waitKey(0)

hough_lines_p_example('document.jpg')
```

### 선분 필터링

```python
import cv2
import numpy as np

def filter_lines(image_path, angle_threshold=30):
    """각도와 길이로 선분 필터링"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)

    result = img.copy()

    if lines is None:
        return result

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # 선분 길이 계산
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # 각도 계산 (수평 기준)
        if x2 - x1 != 0:
            angle = np.degrees(np.arctan(abs(y2 - y1) / abs(x2 - x1)))
        else:
            angle = 90

        # 필터링: 특정 각도 이하만
        if angle < angle_threshold:
            color = (0, 255, 0)  # 거의 수평
        elif angle > 90 - angle_threshold:
            color = (255, 0, 0)  # 거의 수직
        else:
            continue  # 대각선은 무시

        cv2.line(result, (x1, y1), (x2, y2), color, 2)

    cv2.imshow('Filtered Lines', result)
    cv2.waitKey(0)

    return result

filter_lines('building.jpg', angle_threshold=20)
```

### 선분 병합

```python
import cv2
import numpy as np
from collections import defaultdict

def merge_lines(lines, angle_threshold=10, distance_threshold=20):
    """유사한 선분들 병합"""
    if lines is None or len(lines) == 0:
        return []

    # 선분을 각도별로 그룹화
    groups = defaultdict(list)

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # 각도 계산
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

        # 각도 그룹 (angle_threshold 단위로 양자화)
        angle_group = round(angle / angle_threshold) * angle_threshold
        groups[angle_group].append(line[0])

    merged = []

    for angle, group_lines in groups.items():
        if len(group_lines) == 1:
            merged.append(group_lines[0])
            continue

        # 같은 그룹 내에서 가까운 선분들 병합
        # 간단히: 전체 점들의 최소/최대 좌표로 하나의 선분 생성
        all_points = []
        for x1, y1, x2, y2 in group_lines:
            all_points.extend([(x1, y1), (x2, y2)])

        all_points = np.array(all_points)

        # 주 방향으로 정렬하여 양 끝점 선택
        if abs(np.cos(np.radians(angle))) > 0.5:
            # 수평에 가까움: x로 정렬
            sorted_pts = sorted(all_points, key=lambda p: p[0])
        else:
            # 수직에 가까움: y로 정렬
            sorted_pts = sorted(all_points, key=lambda p: p[1])

        start = sorted_pts[0]
        end = sorted_pts[-1]
        merged.append([start[0], start[1], end[0], end[1]])

    return merged
```

---

## 4. 허프 원 변환

### cv2.HoughCircles() 함수

```
허프 원 변환:
이미지에서 원 검출

원의 방정식: (x - a)² + (y - b)² = r²
파라미터: 중심 (a, b), 반지름 r

3차원 누적 배열 필요 → 비효율적
→ 그래디언트 기반 방법 사용 (cv2.HOUGH_GRADIENT)

cv2.HOUGH_GRADIENT 동작:
1. 엣지 검출
2. 각 엣지 점에서 그래디언트 방향으로 투표
3. 중심 후보 선정
4. 반지름 추정
```

```python
circles = cv2.HoughCircles(image, method, dp, minDist, param1, param2, minRadius, maxRadius)
```

| 파라미터 | 설명 |
|----------|------|
| image | 입력 그레이스케일 이미지 |
| method | 검출 방법 (cv2.HOUGH_GRADIENT 또는 cv2.HOUGH_GRADIENT_ALT) |
| dp | 누적 배열 해상도 비율 (1 = 원본과 동일) |
| minDist | 검출된 원 중심 간 최소 거리 |
| param1 | Canny 엣지의 상위 임계값 |
| param2 | 원 검출 임계값 (낮을수록 많이 검출) |
| minRadius | 최소 반지름 (0 = 무제한) |
| maxRadius | 최대 반지름 (0 = 무제한) |

### 기본 사용법

```python
import cv2
import numpy as np

def hough_circles_example(image_path):
    """허프 원 검출"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 노이즈 제거 (원 검출에 중요)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 허프 원 변환
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,              # 원본 해상도
        minDist=50,        # 원 중심 간 최소 거리
        param1=100,        # Canny 상위 임계값
        param2=30,         # 원 검출 임계값
        minRadius=10,      # 최소 반지름
        maxRadius=100      # 최대 반지름
    )

    result = img.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for circle in circles[0, :]:
            cx, cy, r = circle

            # 원 그리기
            cv2.circle(result, (cx, cy), r, (0, 255, 0), 2)

            # 중심점
            cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)

            print(f"원: 중심({cx}, {cy}), 반지름={r}")

        print(f"검출된 원 수: {len(circles[0])}")

    cv2.imshow('Circles', result)
    cv2.waitKey(0)

hough_circles_example('coins.jpg')
```

### 동전 검출

```python
import cv2
import numpy as np

def detect_coins(image_path):
    """동전 검출 및 분류"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # 허프 원 변환
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=100,
        param2=35,
        minRadius=30,
        maxRadius=80
    )

    result = img.copy()
    coin_count = 0
    total_value = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for circle in circles[0, :]:
            cx, cy, r = circle
            coin_count += 1

            # 크기로 동전 종류 추정 (예시)
            if r < 40:
                value = 10
                color = (255, 0, 0)    # 파랑
            elif r < 55:
                value = 50
                color = (0, 255, 0)    # 녹색
            else:
                value = 100
                color = (0, 0, 255)    # 빨강

            total_value += value

            # 그리기
            cv2.circle(result, (cx, cy), r, color, 2)
            cv2.circle(result, (cx, cy), 3, (0, 0, 0), -1)
            cv2.putText(result, f'{value}', (cx - 15, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    print(f"동전 개수: {coin_count}")
    print(f"총액: {total_value}원")

    cv2.imshow('Coins', result)
    cv2.waitKey(0)

    return coin_count, total_value

detect_coins('coins.jpg')
```

### HOUGH_GRADIENT_ALT (OpenCV 4.3+)

```python
import cv2
import numpy as np

def hough_circles_alt(image_path):
    """HOUGH_GRADIENT_ALT 사용 (더 정확)"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # HOUGH_GRADIENT_ALT: 더 정확하지만 느림
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT_ALT,  # 대체 알고리즘
        dp=1.5,
        minDist=50,
        param1=300,    # 엣지 그래디언트 임계값
        param2=0.9,    # 원형도 임계값 (0-1, 높을수록 엄격)
        minRadius=20,
        maxRadius=100
    )

    result = img.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for cx, cy, r in circles[0, :]:
            cv2.circle(result, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)

    cv2.imshow('HOUGH_GRADIENT_ALT', result)
    cv2.waitKey(0)

hough_circles_alt('circles.jpg')
```

---

## 5. 파라미터 튜닝 전략

### 직선 검출 파라미터

```
┌────────────────────────────────────────────────────────────────┐
│                    HoughLines 파라미터                          │
├────────────────────────────────────────────────────────────────┤
│ rho (ρ 해상도)                                                  │
│ - 작을수록: 더 정밀, 더 많은 메모리, 더 느림                     │
│ - 권장: 1 (1픽셀)                                               │
│                                                                │
│ theta (θ 해상도)                                                │
│ - 작을수록: 더 정밀한 각도                                       │
│ - 권장: np.pi/180 (1도)                                         │
│                                                                │
│ threshold (최소 투표 수)                                        │
│ - 높을수록: 더 강한(긴) 직선만 검출                             │
│ - 낮을수록: 약한(짧은) 직선도 검출, 노이즈 증가                 │
│ - 튜닝 방법: 이미지 크기와 예상 직선 길이에 따라 조정           │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                   HoughLinesP 파라미터                          │
├────────────────────────────────────────────────────────────────┤
│ minLineLength (최소 선분 길이)                                  │
│ - 높을수록: 긴 선분만 검출                                      │
│ - 노이즈 감소에 효과적                                          │
│                                                                │
│ maxLineGap (최대 간격)                                         │
│ - 높을수록: 끊어진 선분도 하나로 연결                           │
│ - 점선 검출 시 유용                                             │
└────────────────────────────────────────────────────────────────┘
```

### 원 검출 파라미터

```
┌────────────────────────────────────────────────────────────────┐
│                   HoughCircles 파라미터                         │
├────────────────────────────────────────────────────────────────┤
│ dp (해상도 비율)                                                │
│ - 1: 원본 해상도 → 정확하지만 느림                              │
│ - 2: 1/2 해상도 → 빠르지만 덜 정확                              │
│ - 권장: 1 ~ 1.5                                                 │
│                                                                │
│ minDist (최소 중심 거리)                                        │
│ - 너무 작으면: 같은 원을 여러 번 검출                           │
│ - 너무 크면: 가까운 원 놓침                                     │
│ - 권장: 예상 원 반지름 * 2 이상                                 │
│                                                                │
│ param1 (Canny 상위 임계값)                                      │
│ - 높을수록: 강한 엣지만 사용                                    │
│ - 권장: 100 ~ 200                                               │
│                                                                │
│ param2 (누적 임계값)                                            │
│ - 높을수록: 확실한 원만 검출                                    │
│ - 낮을수록: 불완전한 원도 검출                                  │
│ - 권장: 20 ~ 50                                                 │
│                                                                │
│ minRadius, maxRadius                                            │
│ - 예상 원 크기 범위 지정                                        │
│ - 잘못 설정하면 검출 실패                                       │
└────────────────────────────────────────────────────────────────┘
```

### 트랙바로 파라미터 튜닝

```python
import cv2
import numpy as np

def tune_hough_circles(image_path):
    """트랙바로 HoughCircles 파라미터 튜닝"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    cv2.namedWindow('Circles')

    def nothing(x):
        pass

    cv2.createTrackbar('minDist', 'Circles', 50, 200, nothing)
    cv2.createTrackbar('param1', 'Circles', 100, 300, nothing)
    cv2.createTrackbar('param2', 'Circles', 30, 100, nothing)
    cv2.createTrackbar('minRadius', 'Circles', 10, 100, nothing)
    cv2.createTrackbar('maxRadius', 'Circles', 100, 200, nothing)

    while True:
        minDist = cv2.getTrackbarPos('minDist', 'Circles')
        param1 = cv2.getTrackbarPos('param1', 'Circles')
        param2 = cv2.getTrackbarPos('param2', 'Circles')
        minRadius = cv2.getTrackbarPos('minRadius', 'Circles')
        maxRadius = cv2.getTrackbarPos('maxRadius', 'Circles')

        # 유효성 검사
        if minDist < 1:
            minDist = 1
        if param2 < 1:
            param2 = 1

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius
        )

        result = img.copy()

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for cx, cy, r in circles[0, :]:
                cv2.circle(result, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)

            # 검출된 원 수 표시
            cv2.putText(result, f'Circles: {len(circles[0])}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Circles', result)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

tune_hough_circles('coins.jpg')
```

---

## 6. 차선 검출 기초

### 차선 검출 파이프라인

```
1. 관심 영역(ROI) 설정
         │
         ▼
2. 그레이스케일 변환
         │
         ▼
3. 가우시안 블러
         │
         ▼
4. Canny 엣지 검출
         │
         ▼
5. 관심 영역 마스킹
         │
         ▼
6. 허프 직선 변환
         │
         ▼
7. 선분 필터링 및 평균화
         │
         ▼
8. 결과 합성
```

### 기본 차선 검출

```python
import cv2
import numpy as np

def detect_lane_lines(image):
    """기본 차선 검출"""
    height, width = image.shape[:2]

    # 그레이스케일
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny 엣지
    edges = cv2.Canny(blurred, 50, 150)

    # 관심 영역 (사다리꼴)
    mask = np.zeros_like(edges)
    vertices = np.array([[
        (0, height),
        (width * 0.45, height * 0.6),
        (width * 0.55, height * 0.6),
        (width, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # 허프 직선 변환
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )

    # 결과 이미지
    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # 원본과 합성
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    return result

# 사용 예
img = cv2.imread('road.jpg')
result = detect_lane_lines(img)
cv2.imshow('Lane Detection', result)
cv2.waitKey(0)
```

### 좌/우 차선 분리

```python
import cv2
import numpy as np

def separate_lanes(image):
    """좌/우 차선 분리 검출"""
    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # ROI 마스크
    mask = np.zeros_like(edges)
    vertices = np.array([[
        (50, height),
        (width * 0.45, height * 0.6),
        (width * 0.55, height * 0.6),
        (width - 50, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked, 1, np.pi/180, 30,
                             minLineLength=30, maxLineGap=100)

    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 기울기 계산
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)

            # 기울기로 좌/우 분류
            # 이미지 좌표계: y축이 아래로 증가
            # 왼쪽 차선: 음의 기울기 (/)
            # 오른쪽 차선: 양의 기울기 (\)
            if slope < -0.5:
                left_lines.append(line[0])
            elif slope > 0.5:
                right_lines.append(line[0])

    result = image.copy()

    # 좌/우 차선 그리기
    for x1, y1, x2, y2 in left_lines:
        cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 3)  # 파랑

    for x1, y1, x2, y2 in right_lines:
        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 3)  # 빨강

    return result, left_lines, right_lines

# 사용 예
img = cv2.imread('road.jpg')
result, left, right = separate_lanes(img)
print(f"왼쪽 차선: {len(left)}개")
print(f"오른쪽 차선: {len(right)}개")
cv2.imshow('Lanes', result)
cv2.waitKey(0)
```

### 차선 평균화

```python
import cv2
import numpy as np

def average_lane_lines(lines, height):
    """선분들을 평균하여 하나의 직선으로"""
    if len(lines) == 0:
        return None

    # 모든 점 수집
    x_coords = []
    y_coords = []

    for x1, y1, x2, y2 in lines:
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])

    # 1차 다항식 피팅 (직선)
    poly = np.polyfit(y_coords, x_coords, 1)

    # 직선의 시작점과 끝점 계산
    y1 = height
    y2 = int(height * 0.6)
    x1 = int(np.polyval(poly, y1))
    x2 = int(np.polyval(poly, y2))

    return (x1, y1, x2, y2)

def detect_lanes_averaged(image):
    """평균화된 차선 검출"""
    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # ROI
    mask = np.zeros_like(edges)
    vertices = np.array([[
        (50, height),
        (width * 0.45, height * 0.6),
        (width * 0.55, height * 0.6),
        (width - 50, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked, 1, np.pi/180, 30,
                             minLineLength=30, maxLineGap=100)

    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)

            if slope < -0.5:
                left_lines.append(line[0])
            elif slope > 0.5:
                right_lines.append(line[0])

    result = image.copy()

    # 평균화된 차선 그리기
    left_avg = average_lane_lines(left_lines, height)
    right_avg = average_lane_lines(right_lines, height)

    if left_avg is not None:
        cv2.line(result, (left_avg[0], left_avg[1]),
                 (left_avg[2], left_avg[3]), (255, 0, 0), 5)

    if right_avg is not None:
        cv2.line(result, (right_avg[0], right_avg[1]),
                 (right_avg[2], right_avg[3]), (0, 0, 255), 5)

    # 차선 영역 채우기
    if left_avg is not None and right_avg is not None:
        pts = np.array([
            [left_avg[0], left_avg[1]],
            [left_avg[2], left_avg[3]],
            [right_avg[2], right_avg[3]],
            [right_avg[0], right_avg[1]]
        ], np.int32)

        overlay = result.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        result = cv2.addWeighted(overlay, 0.3, result, 0.7, 0)

    return result

# 사용 예
img = cv2.imread('road.jpg')
result = detect_lanes_averaged(img)
cv2.imshow('Averaged Lanes', result)
cv2.waitKey(0)
```

---

## 7. 연습 문제

### 문제 1: 체스판 검출

체스판 이미지에서 모든 직선을 검출하고 교차점을 찾으세요.

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def detect_chessboard_lines(image_path):
    """체스판 직선과 교차점 검출"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    result = img.copy()
    horizontal = []
    vertical = []

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta)

            a = np.cos(theta)
            b = np.sin(theta)

            # 수평선/수직선 분류
            if 80 < angle < 100:  # 수직
                vertical.append((rho, theta))
            elif angle < 10 or angle > 170:  # 수평
                horizontal.append((rho, theta))

    # 교차점 계산
    intersections = []
    for h_rho, h_theta in horizontal:
        for v_rho, v_theta in vertical:
            # 두 직선의 교차점
            A = np.array([
                [np.cos(h_theta), np.sin(h_theta)],
                [np.cos(v_theta), np.sin(v_theta)]
            ])
            b = np.array([h_rho, v_rho])

            try:
                x, y = np.linalg.solve(A, b)
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    intersections.append((int(x), int(y)))
            except:
                pass

    # 그리기
    for x, y in intersections:
        cv2.circle(result, (x, y), 5, (0, 0, 255), -1)

    print(f"교차점 수: {len(intersections)}")
    cv2.imshow('Chessboard', result)
    cv2.waitKey(0)

detect_chessboard_lines('chessboard.jpg')
```

</details>

### 문제 2: 아이리스 검출

눈 이미지에서 홍채 원을 검출하세요.

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def detect_iris(image_path):
    """눈에서 홍채 검출"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 밝기 균일화
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 홍채 검출 (어두운 원)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=100,
        param2=25,
        minRadius=20,
        maxRadius=60
    )

    result = img.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))

        # 가장 큰 원 선택 (홍채)
        for cx, cy, r in sorted(circles[0], key=lambda x: -x[2])[:1]:
            # 홍채
            cv2.circle(result, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(result, (cx, cy), 2, (0, 0, 255), 3)

    cv2.imshow('Iris', result)
    cv2.waitKey(0)

detect_iris('eye.jpg')
```

</details>

### 문제 3: 원형 도로 표지판 검출

빨간색 원형 교통 표지판을 검출하세요.

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def detect_red_signs(image_path):
    """빨간 원형 표지판 검출"""
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 빨간색 마스크 (HSV에서 빨강은 0도와 180도 부근)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 모폴로지 연산
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # 원 검출
    circles = cv2.HoughCircles(
        red_mask,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=100
    )

    result = img.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for cx, cy, r in circles[0]:
            cv2.circle(result, (cx, cy), r, (0, 255, 0), 3)
            cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)

    cv2.imshow('Red Signs', result)
    cv2.imshow('Mask', red_mask)
    cv2.waitKey(0)

detect_red_signs('traffic_sign.jpg')
```

</details>

### 추천 문제

| 난이도 | 주제 | 설명 |
|--------|------|------|
| ⭐ | 직선 검출 | 건물 사진에서 수평/수직선 검출 |
| ⭐⭐ | 동전 세기 | 동전 사진에서 개수와 금액 계산 |
| ⭐⭐ | 문서 검출 | 문서 경계선 4개 검출 |
| ⭐⭐⭐ | 차선 검출 | 도로 영상에서 실시간 차선 검출 |
| ⭐⭐⭐ | 계기판 | 아날로그 게이지 눈금 읽기 |

---

## 다음 단계

- [12_Histogram_Analysis.md](./12_Histogram_Analysis.md) - calcHist, equalizeHist, CLAHE

---

## 참고 자료

- [OpenCV Hough Line Transform](https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html)
- [OpenCV Hough Circle Transform](https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html)
- [Lane Detection Tutorial](https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132)
