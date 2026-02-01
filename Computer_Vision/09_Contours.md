# 윤곽선 검출 (Contour Detection)

## 개요

윤곽선(Contour)은 동일한 색상이나 밝기를 가진 연속적인 점들의 곡선으로, 객체의 형태를 나타냅니다. 이 레슨에서는 findContours()를 사용한 윤곽선 검출, 계층 구조, 근사화, 면적/둘레 계산 등을 학습합니다.

---

## 목차

1. [윤곽선 기초](#1-윤곽선-기초)
2. [findContours() 함수](#2-findcontours-함수)
3. [윤곽선 계층 구조](#3-윤곽선-계층-구조)
4. [윤곽선 그리기와 근사화](#4-윤곽선-그리기와-근사화)
5. [윤곽선 속성 계산](#5-윤곽선-속성-계산)
6. [객체 카운팅과 분리](#6-객체-카운팅과-분리)
7. [연습 문제](#7-연습-문제)

---

## 1. 윤곽선 기초

### 윤곽선이란?

```
윤곽선 (Contour):
- 동일한 색상/밝기를 가진 연속적인 점들의 곡선
- 객체의 경계를 나타냄
- 이진 이미지에서 추출

원본 이미지           이진화              윤곽선 검출
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  ┌───┐      │     │  ■■■■■      │     │  ┌───┐      │
│  │ ● │      │     │  ■■■■■      │     │  │   │      │
│  └───┘      │ ──▶ │  ■■■■■      │ ──▶ │  └───┘      │
│        ┌──┐ │     │        ■■■ │     │        ┌──┐ │
│        └──┘ │     │        ■■■ │     │        └──┘ │
└─────────────┘     └─────────────┘     └─────────────┘
                         (흰색 영역)        (경계선만)
```

### 윤곽선 검출 과정

```
1. 이미지 읽기
      │
      ▼
2. 그레이스케일 변환
      │
      ▼
3. 이진화 (threshold)
      │
      ▼
4. 윤곽선 검출 (findContours)
      │
      ▼
5. 윤곽선 분석/그리기
```

### 기본 예제

```python
import cv2
import numpy as np

# 이미지 읽기
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 이진화
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 윤곽선 검출
contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,      # 외곽 윤곽선만
    cv2.CHAIN_APPROX_SIMPLE  # 압축
)

print(f"검출된 윤곽선 수: {len(contours)}")

# 윤곽선 그리기
result = img.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

cv2.imshow('Contours', result)
cv2.waitKey(0)
```

---

## 2. findContours() 함수

### 함수 시그니처

```python
contours, hierarchy = cv2.findContours(image, mode, method)
```

| 파라미터 | 설명 |
|----------|------|
| image | 입력 이진 이미지 (8비트 단일 채널) |
| mode | 윤곽선 검색 모드 (RETR_*) |
| method | 윤곽선 근사화 방법 (CHAIN_*) |
| contours | 검출된 윤곽선 리스트 |
| hierarchy | 윤곽선 계층 구조 |

### 검색 모드 (Retrieval Mode)

```
┌────────────────────────────────────────────────────────────────────┐
│                         RETR_EXTERNAL                              │
├────────────────────────────────────────────────────────────────────┤
│  가장 바깥쪽 윤곽선만 검출                                          │
│                                                                    │
│  ┌──────────────┐                                                  │
│  │  ┌────────┐  │   → 외곽 사각형만 검출                           │
│  │  │ ┌────┐ │  │                                                  │
│  │  │ └────┘ │  │                                                  │
│  │  └────────┘  │                                                  │
│  └──────────────┘                                                  │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                           RETR_LIST                                │
├────────────────────────────────────────────────────────────────────┤
│  모든 윤곽선 검출, 계층 구조 없음 (동일 레벨)                       │
│                                                                    │
│  ┌──────────────┐                                                  │
│  │  ┌────────┐  │   → 3개 모두 검출, 부모-자식 관계 없음           │
│  │  │ ┌────┐ │  │                                                  │
│  │  │ └────┘ │  │                                                  │
│  │  └────────┘  │                                                  │
│  └──────────────┘                                                  │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                           RETR_CCOMP                               │
├────────────────────────────────────────────────────────────────────┤
│  2레벨 계층 구조                                                   │
│  - 레벨 1: 외곽 윤곽선                                             │
│  - 레벨 2: 구멍 (내부 윤곽선)                                      │
│                                                                    │
│  ┌──────────────┐   레벨 1 (외곽)                                  │
│  │  ┌────────┐  │   레벨 2 (구멍)                                  │
│  │  │ ■■■■■■ │  │   (내부의 흰색 영역은 레벨 2)                    │
│  │  └────────┘  │                                                  │
│  └──────────────┘                                                  │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                           RETR_TREE                                │
├────────────────────────────────────────────────────────────────────┤
│  완전한 계층 구조 (부모-자식 관계)                                  │
│                                                                    │
│  ┌──────────────┐   레벨 0 (최외곽)                                │
│  │  ┌────────┐  │   레벨 1                                         │
│  │  │ ┌────┐ │  │   레벨 2                                         │
│  │  │ │ ■■ │ │  │   레벨 3                                         │
│  │  │ └────┘ │  │                                                  │
│  │  └────────┘  │                                                  │
│  └──────────────┘                                                  │
└────────────────────────────────────────────────────────────────────┘
```

### 근사화 방법 (Approximation Method)

```
┌────────────────────────────────────────────────────────────────────┐
│                      CHAIN_APPROX_NONE                             │
├────────────────────────────────────────────────────────────────────┤
│  모든 윤곽선 점 저장                                               │
│                                                                    │
│      • • • • • •                                                   │
│    •           •    → 모든 경계 픽셀 저장                          │
│    •           •       메모리 많이 사용                            │
│    •           •                                                   │
│      • • • • • •                                                   │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                     CHAIN_APPROX_SIMPLE                            │
├────────────────────────────────────────────────────────────────────┤
│  직선 부분은 끝점만 저장 (압축)                                    │
│                                                                    │
│      •         •                                                   │
│                      → 4개 꼭짓점만 저장                           │
│                         메모리 효율적                              │
│                                                                    │
│      •         •                                                   │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                    CHAIN_APPROX_TC89_L1                            │
│                    CHAIN_APPROX_TC89_KCOS                          │
├────────────────────────────────────────────────────────────────────┤
│  Teh-Chin 체인 근사화 알고리즘                                     │
│  → 더 공격적인 압축                                                │
└────────────────────────────────────────────────────────────────────┘
```

### 모드별 예제

```python
import cv2
import numpy as np

def compare_retrieval_modes(image_path):
    """윤곽선 검색 모드 비교"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    modes = [
        (cv2.RETR_EXTERNAL, 'RETR_EXTERNAL'),
        (cv2.RETR_LIST, 'RETR_LIST'),
        (cv2.RETR_CCOMP, 'RETR_CCOMP'),
        (cv2.RETR_TREE, 'RETR_TREE')
    ]

    for mode, name in modes:
        contours, hierarchy = cv2.findContours(
            binary.copy(),
            mode,
            cv2.CHAIN_APPROX_SIMPLE
        )

        result = img.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        print(f"{name}: {len(contours)} contours")
        cv2.imshow(name, result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

compare_retrieval_modes('nested_shapes.jpg')
```

---

## 3. 윤곽선 계층 구조

### hierarchy 구조

```
hierarchy[i] = [Next, Previous, First_Child, Parent]

Next:        같은 레벨의 다음 윤곽선 인덱스 (-1: 없음)
Previous:    같은 레벨의 이전 윤곽선 인덱스 (-1: 없음)
First_Child: 첫 번째 자식 윤곽선 인덱스 (-1: 없음)
Parent:      부모 윤곽선 인덱스 (-1: 없음)

예시:
┌───────────────────────────────────┐
│ ┌─────────────┐ ┌─────────────┐  │
│ │   ┌───┐     │ │             │  │
│ │   │ A │     │ │      B      │  │
│ │   └───┘     │ │             │  │
│ │      C      │ │             │  │
│ └─────────────┘ └─────────────┘  │
│                  D                │
└───────────────────────────────────┘

RETR_TREE 결과:
Index 0 (D): Next=-1, Prev=-1, Child=1, Parent=-1  (최외곽)
Index 1 (C): Next=2,  Prev=-1, Child=3, Parent=0
Index 2 (B): Next=-1, Prev=1,  Child=-1, Parent=0
Index 3 (A): Next=-1, Prev=-1, Child=-1, Parent=1
```

### 계층 구조 탐색

```python
import cv2
import numpy as np

def analyze_hierarchy(image_path):
    """윤곽선 계층 구조 분석"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        print("윤곽선이 없습니다.")
        return

    hierarchy = hierarchy[0]  # (1, N, 4) -> (N, 4)

    print("계층 구조 분석:")
    print("-" * 50)

    for i, h in enumerate(hierarchy):
        next_c, prev_c, first_child, parent = h

        # 레벨 계산
        level = 0
        p = parent
        while p != -1:
            level += 1
            p = hierarchy[p][3]  # 부모의 부모

        indent = "  " * level
        print(f"{indent}윤곽선 {i}:")
        print(f"{indent}  레벨: {level}")
        print(f"{indent}  부모: {parent}")
        print(f"{indent}  자식: {first_child}")
        print(f"{indent}  면적: {cv2.contourArea(contours[i]):.0f}")

analyze_hierarchy('nested_shapes.jpg')
```

### 특정 레벨의 윤곽선만 추출

```python
import cv2
import numpy as np

def get_contours_at_level(contours, hierarchy, level):
    """특정 레벨의 윤곽선만 반환"""
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    result = []

    for i in range(len(contours)):
        # 현재 윤곽선의 레벨 계산
        current_level = 0
        parent = hierarchy[i][3]
        while parent != -1:
            current_level += 1
            parent = hierarchy[parent][3]

        if current_level == level:
            result.append(contours[i])

    return result

def get_outer_contours(contours, hierarchy):
    """최외곽 윤곽선만 반환 (부모가 없는 것)"""
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    result = []

    for i in range(len(contours)):
        if hierarchy[i][3] == -1:  # 부모가 없음
            result.append(contours[i])

    return result

def get_inner_contours(contours, hierarchy, parent_idx):
    """특정 윤곽선의 자식(내부) 윤곽선 반환"""
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    result = []

    # 첫 번째 자식
    child = hierarchy[parent_idx][2]

    while child != -1:
        result.append(contours[child])
        child = hierarchy[child][0]  # 다음 형제

    return result

# 사용 예
img = cv2.imread('nested.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(
    binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)

# 레벨 0 윤곽선
level0 = get_contours_at_level(contours, hierarchy, 0)

# 최외곽 윤곽선
outer = get_outer_contours(contours, hierarchy)

result = img.copy()
cv2.drawContours(result, outer, -1, (0, 255, 0), 2)
cv2.imshow('Outer Contours', result)
cv2.waitKey(0)
```

---

## 4. 윤곽선 그리기와 근사화

### cv2.drawContours() 함수

```python
cv2.drawContours(image, contours, contourIdx, color, thickness)
```

| 파라미터 | 설명 |
|----------|------|
| image | 그릴 이미지 |
| contours | 윤곽선 리스트 |
| contourIdx | 그릴 윤곽선 인덱스 (-1: 모두) |
| color | 색상 (B, G, R) |
| thickness | 선 두께 (-1: 채우기) |

```python
import cv2
import numpy as np

def draw_contours_examples(image, contours):
    """다양한 방식으로 윤곽선 그리기"""

    # 모든 윤곽선 그리기
    result1 = image.copy()
    cv2.drawContours(result1, contours, -1, (0, 255, 0), 2)

    # 특정 윤곽선만 그리기
    result2 = image.copy()
    if len(contours) > 0:
        cv2.drawContours(result2, contours, 0, (255, 0, 0), 3)

    # 윤곽선 채우기
    result3 = image.copy()
    cv2.drawContours(result3, contours, -1, (0, 0, 255), -1)

    # 각 윤곽선 다른 색으로
    result4 = image.copy()
    for i, contour in enumerate(contours):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.drawContours(result4, [contour], 0, color, 2)

    return result1, result2, result3, result4
```

### cv2.approxPolyDP() - 다각형 근사화

```
Douglas-Peucker 알고리즘:
윤곽선을 더 적은 점으로 근사화

epsilon (정밀도):
- 작을수록: 원본에 가까움 (점 많음)
- 클수록: 단순화 (점 적음)

예시:
원본 (많은 점)          epsilon=0.01         epsilon=0.05
      •  •  •                 •                     •
   •        •              •     •                •   •
  •          •            •       •              •     •
  •          •             •     •                  •
   •        •               •   •
      •  •  •                 •                     •
```

```python
import cv2
import numpy as np

def approximate_contour(contour, epsilon_ratio=0.02):
    """
    윤곽선 다각형 근사화
    epsilon_ratio: 둘레 대비 허용 오차 비율
    """
    # 윤곽선 둘레 계산
    perimeter = cv2.arcLength(contour, True)

    # epsilon = 둘레 * 비율
    epsilon = epsilon_ratio * perimeter

    # 근사화
    approx = cv2.approxPolyDP(contour, epsilon, True)

    return approx

def compare_approximations(image, contour):
    """다양한 epsilon으로 근사화 비교"""
    epsilons = [0.001, 0.01, 0.02, 0.05, 0.1]

    for eps in epsilons:
        result = image.copy()
        approx = approximate_contour(contour, eps)

        cv2.drawContours(result, [approx], 0, (0, 255, 0), 2)

        # 꼭짓점 표시
        for point in approx:
            x, y = point[0]
            cv2.circle(result, (x, y), 5, (0, 0, 255), -1)

        cv2.putText(result, f'epsilon={eps}, points={len(approx)}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow(f'Approximation {eps}', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 사용 예
img = cv2.imread('shape.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    compare_approximations(img, contours[0])
```

### 도형 식별 (꼭짓점 수로)

```python
import cv2
import numpy as np

def identify_shape(contour):
    """꼭짓점 수로 도형 식별"""
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    vertices = len(approx)

    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        # 정사각형 vs 직사각형 구분
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            return "Square"
        else:
            return "Rectangle"
    elif vertices == 5:
        return "Pentagon"
    elif vertices == 6:
        return "Hexagon"
    elif vertices > 6:
        # 원형 여부 확인
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity > 0.8:
            return "Circle"
        else:
            return f"Polygon ({vertices} vertices)"
    else:
        return "Unknown"

def label_shapes(image_path):
    """이미지의 모든 도형 식별 및 라벨링"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()

    for contour in contours:
        # 너무 작은 윤곽선 무시
        if cv2.contourArea(contour) < 100:
            continue

        # 도형 식별
        shape = identify_shape(contour)

        # 무게중심 계산
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # 윤곽선 그리기
        cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)

        # 라벨 표시
        cv2.putText(result, shape, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Shapes', result)
    cv2.waitKey(0)

label_shapes('shapes.jpg')
```

---

## 5. 윤곽선 속성 계산

### 둘레와 면적

```python
import cv2
import numpy as np

def contour_properties(contour):
    """윤곽선의 기본 속성 계산"""

    # 면적
    area = cv2.contourArea(contour)

    # 둘레 (closed=True: 닫힌 곡선)
    perimeter = cv2.arcLength(contour, True)

    # 경계 사각형
    x, y, w, h = cv2.boundingRect(contour)
    bounding_area = w * h

    # 면적 비율 (Extent)
    extent = area / bounding_area if bounding_area > 0 else 0

    # 원형도 (Circularity)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

    # 컨벡스 헐
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    # 솔리디티 (Solidity)
    solidity = area / hull_area if hull_area > 0 else 0

    return {
        'area': area,
        'perimeter': perimeter,
        'extent': extent,
        'circularity': circularity,
        'solidity': solidity
    }

# 사용 예
img = cv2.imread('shape.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i, contour in enumerate(contours):
    props = contour_properties(contour)
    print(f"윤곽선 {i}:")
    print(f"  면적: {props['area']:.0f}")
    print(f"  둘레: {props['perimeter']:.1f}")
    print(f"  Extent: {props['extent']:.2f}")
    print(f"  원형도: {props['circularity']:.2f}")
    print(f"  Solidity: {props['solidity']:.2f}")
```

### 경계 도형

```python
import cv2
import numpy as np

def bounding_shapes(image, contour):
    """윤곽선의 다양한 경계 도형"""
    result = image.copy()

    # 1. 경계 사각형 (Bounding Rectangle)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 2. 회전된 경계 사각형 (Rotated Rectangle)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int_(box)
    cv2.drawContours(result, [box], 0, (255, 0, 0), 2)

    # 3. 최소 외접원 (Minimum Enclosing Circle)
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    cv2.circle(result, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)

    # 4. 타원 피팅 (Fitting Ellipse)
    if len(contour) >= 5:  # 최소 5개 점 필요
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(result, ellipse, (255, 255, 0), 2)

    # 5. 직선 피팅 (Fitting Line)
    rows, cols = image.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(result, (cols-1, righty), (0, lefty), (0, 255, 255), 2)

    return result

# 사용 예
img = cv2.imread('shape.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    result = bounding_shapes(img, contours[0])
    cv2.imshow('Bounding Shapes', result)
    cv2.waitKey(0)
```

### 컨벡스 헐

```
컨벡스 헐 (Convex Hull):
점 집합을 감싸는 가장 작은 볼록 다각형

      •  •
    •      •          ┌──────────┐
  •          •   →   │          │
    •  •   •         │          │
        • •          └──────────┘
   원본 윤곽선          컨벡스 헐

컨벡시티 결함 (Convexity Defects):
윤곽선과 컨벡스 헐 사이의 가장 깊은 점
→ 손가락 검출 등에 사용
```

```python
import cv2
import numpy as np

def convex_hull_analysis(image, contour):
    """컨벡스 헐 분석"""
    result = image.copy()

    # 컨벡스 헐 계산
    hull = cv2.convexHull(contour)

    # 원본 윤곽선
    cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)

    # 컨벡스 헐
    cv2.drawContours(result, [hull], 0, (0, 0, 255), 2)

    # 컨벡시티 결함 (손가락 검출 등에 유용)
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if len(hull_indices) > 3 and len(contour) > 3:
        defects = cv2.convexityDefects(contour, hull_indices)

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                # 결함 깊이가 일정 이상인 경우만 표시
                if d / 256 > 10:  # 깊이 임계값
                    cv2.circle(result, far, 5, (255, 0, 255), -1)
                    cv2.line(result, start, far, (255, 0, 255), 1)
                    cv2.line(result, far, end, (255, 0, 255), 1)

    return result

# 사용 예 (손 이미지)
img = cv2.imread('hand.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # 가장 큰 윤곽선 선택
    largest = max(contours, key=cv2.contourArea)
    result = convex_hull_analysis(img, largest)
    cv2.imshow('Convex Hull', result)
    cv2.waitKey(0)
```

---

## 6. 객체 카운팅과 분리

### 객체 카운팅

```python
import cv2
import numpy as np

def count_objects(image_path, min_area=100):
    """이미지에서 객체 수 세기"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 적응형 이진화
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # 모폴로지 연산으로 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 윤곽선 검출
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 크기 필터링
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    result = img.copy()
    for i, contour in enumerate(valid_contours):
        # 무게중심
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 번호 표시
            cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)
            cv2.putText(result, str(i + 1), (cx - 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    print(f"검출된 객체 수: {len(valid_contours)}")

    cv2.imshow('Counted Objects', result)
    cv2.waitKey(0)

    return len(valid_contours)

# 동전 세기 예제
count_objects('coins.jpg', min_area=500)
```

### 객체 분리 및 추출

```python
import cv2
import numpy as np

def extract_objects(image_path, output_dir='objects/'):
    """개별 객체를 분리하여 저장"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    objects = []
    for i, contour in enumerate(contours):
        # 너무 작은 객체 무시
        if cv2.contourArea(contour) < 100:
            continue

        # 경계 사각형
        x, y, w, h = cv2.boundingRect(contour)

        # 여백 추가
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)

        # 객체 영역 추출
        roi = img[y1:y2, x1:x2].copy()
        objects.append(roi)

        # 저장
        cv2.imwrite(f'{output_dir}object_{i:03d}.jpg', roi)

    print(f"{len(objects)}개 객체 추출 완료")
    return objects

# 사용 예
objects = extract_objects('multiple_objects.jpg')
```

### 특정 모양만 검출

```python
import cv2
import numpy as np

def find_circles(image_path):
    """원형 객체만 검출"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()
    circles = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        # 원형도 계산
        circularity = 4 * np.pi * area / (perimeter ** 2)

        # 원형도가 0.8 이상이면 원으로 판단
        if circularity > 0.8 and area > 100:
            circles.append(contour)
            cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)

            # 중심점 표시
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)

    print(f"원형 객체 수: {len(circles)}")
    cv2.imshow('Circles', result)
    cv2.waitKey(0)

    return circles

def find_rectangles(image_path):
    """사각형 객체만 검출"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()
    rectangles = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue

        # 다각형 근사화
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # 꼭짓점이 4개면 사각형
        if len(approx) == 4:
            rectangles.append(contour)
            cv2.drawContours(result, [approx], 0, (0, 255, 0), 2)

    print(f"사각형 객체 수: {len(rectangles)}")
    cv2.imshow('Rectangles', result)
    cv2.waitKey(0)

    return rectangles

# 사용 예
find_circles('shapes.jpg')
find_rectangles('shapes.jpg')
```

---

## 7. 연습 문제

### 문제 1: 동전 카운터

동전 이미지에서 동전 수를 세고 총 금액을 계산하세요 (크기로 구분).

<details>
<summary>힌트</summary>

동전 크기(면적 또는 반지름)를 기준으로 동전 종류를 분류합니다.

</details>

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def count_coins_by_size(image_path):
    """동전 크기별로 분류하고 개수 세기"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Canny 엣지 검출 + 닫힘 연산
    edges = cv2.Canny(blurred, 30, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    # 윤곽선 검출
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()

    # 크기별 분류 (반지름 기준)
    small_coins = []   # 10원
    medium_coins = []  # 50원
    large_coins = []   # 100원

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # 노이즈 무시
            continue

        # 최소 외접원
        (x, y), radius = cv2.minEnclosingCircle(contour)

        # 원형도 확인
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.7:  # 원형 아님
                continue

        # 크기별 분류 (예시 임계값)
        if radius < 30:
            small_coins.append((int(x), int(y), int(radius)))
            color = (255, 0, 0)  # 파랑 - 10원
        elif radius < 40:
            medium_coins.append((int(x), int(y), int(radius)))
            color = (0, 255, 0)  # 녹색 - 50원
        else:
            large_coins.append((int(x), int(y), int(radius)))
            color = (0, 0, 255)  # 빨강 - 100원

        cv2.circle(result, (int(x), int(y)), int(radius), color, 2)

    # 결과 출력
    total = (len(small_coins) * 10 +
             len(medium_coins) * 50 +
             len(large_coins) * 100)

    print(f"10원: {len(small_coins)}개")
    print(f"50원: {len(medium_coins)}개")
    print(f"100원: {len(large_coins)}개")
    print(f"총액: {total}원")

    cv2.imshow('Coins', result)
    cv2.waitKey(0)

count_coins_by_size('coins.jpg')
```

</details>

### 문제 2: 문서 사각형 검출

이미지에서 문서(종이)의 윤곽선을 찾고 4개의 꼭짓점을 반환하세요.

<details>
<summary>힌트</summary>

가장 큰 4각형 윤곽선을 찾습니다. approxPolyDP로 4개 점으로 근사화합니다.

</details>

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def find_document(image_path):
    """문서 영역의 4개 꼭짓점 찾기"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 엣지 검출
    edges = cv2.Canny(blurred, 50, 150)

    # 윤곽선 검출
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 면적순 정렬
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    document_corners = None

    for contour in contours[:5]:  # 상위 5개만 확인
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # 4개 꼭짓점이면 문서
        if len(approx) == 4:
            document_corners = approx
            break

    if document_corners is not None:
        result = img.copy()
        cv2.drawContours(result, [document_corners], 0, (0, 255, 0), 3)

        # 꼭짓점 표시
        for point in document_corners:
            x, y = point[0]
            cv2.circle(result, (x, y), 10, (0, 0, 255), -1)

        cv2.imshow('Document', result)
        cv2.waitKey(0)

        return document_corners.reshape(4, 2)
    else:
        print("문서를 찾지 못했습니다.")
        return None

corners = find_document('document.jpg')
if corners is not None:
    print("문서 꼭짓점:", corners)
```

</details>

### 문제 3: 빈 공간 검출

이진 이미지에서 구멍(빈 공간)의 수를 세세요.

<details>
<summary>힌트</summary>

RETR_CCOMP 또는 RETR_TREE를 사용하여 내부 윤곽선(구멍)을 찾습니다.

</details>

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def count_holes(image_path):
    """객체 내부의 구멍 수 세기"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # RETR_CCOMP: 2레벨 계층 (외곽 + 구멍)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        return 0

    hierarchy = hierarchy[0]

    result = img.copy()
    holes = []

    for i, h in enumerate(hierarchy):
        # 부모가 있는 윤곽선 = 구멍
        if h[3] != -1:  # 부모가 있음
            area = cv2.contourArea(contours[i])
            if area > 50:  # 노이즈 무시
                holes.append(contours[i])
                cv2.drawContours(result, [contours[i]], 0, (0, 0, 255), 2)

    print(f"구멍 수: {len(holes)}")

    cv2.imshow('Holes', result)
    cv2.waitKey(0)

    return len(holes)

count_holes('donut.jpg')
```

</details>

### 추천 문제

| 난이도 | 주제 | 설명 |
|--------|------|------|
| ⭐ | 기본 검출 | findContours로 객체 개수 세기 |
| ⭐⭐ | 면적 필터 | 특정 크기 범위의 객체만 검출 |
| ⭐⭐ | 도형 분류 | 삼각형, 사각형, 원 구분 |
| ⭐⭐⭐ | 문서 스캐너 | 문서 검출 후 투시 변환 |
| ⭐⭐⭐ | 손가락 카운터 | 컨벡시티 결함으로 손가락 세기 |

---

## 다음 단계

- [10_Shape_Analysis.md](./10_Shape_Analysis.md) - moments, boundingRect, convexHull, matchShapes

---

## 참고 자료

- [OpenCV Contour Features](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)
- [Contour Hierarchy](https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html)
- [Contours in OpenCV](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
