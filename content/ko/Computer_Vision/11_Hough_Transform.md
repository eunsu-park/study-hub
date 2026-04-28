# 허프 변환 (Hough Transform)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 허프 변환(Hough Transform) 원리와 이미지 공간의 점이 파라미터 공간(parameter space)의 곡선으로 어떻게 매핑되는지 설명할 수 있다
2. OpenCV의 HoughLines()와 HoughLinesP()를 사용하여 표준 및 확률적(probabilistic) 허프 직선 검출을 구현할 수 있다
3. HoughCircles()로 허프 원 검출을 적용하고 어큐뮬레이터(accumulator) 및 임계값(threshold) 파라미터를 조정할 수 있다
4. 속도와 정확도 측면에서 표준 허프 변환과 확률적 허프 변환의 장단점을 분석할 수 있다
5. 엣지 검출과 허프 직선 필터링을 결합한 차선 검출(lane detection) 파이프라인을 설계할 수 있다
6. 파라미터 민감도를 평가하고 특정 이미지 도메인에 맞게 허프 변환 설정을 최적화할 수 있다

---

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

### 이론: 이미지 공간 / 파라미터 공간 쌍대성

2D 이미지 공간의 직선은 여러 형식으로 쓸 수 있지만, 모두 **두 파라미터**가 필요합니다 — 2D 직선의 자유도는 2. 한 쌍 `(a, b)`를 쓰는 임의의 매개변수화를 선택하세요. 그러면:

- `(a, b)`를 고정하면 이미지 공간의 직선 하나를 정의합니다(한 곡선).
- 이미지 점 `(x, y)` 하나를 고정하고 "이 점을 지나는 직선들의 `(a, b)` 쌍은 무엇인가?"를 물으면 *파라미터 공간의 곡선* — 그 점을 지나는 모든 직선의 궤적.

같은 기하의 쌍대 관점입니다. 허프의 통찰: 이 쌍대성을 직선 검출에 이용할 수 있다.

1. 이미지의 각 에지 점에 대해, 그와 일치하는 파라미터 값의 *곡선*을 그린다.
2. 파라미터 공간에서 많은 곡선이 교차하는 곳에서, 많은 이미지 에지 점이 *같은 직선* 위에 놓여 있다.
3. 파라미터 공간의 피크를 찾는다. 그것들이 직선이다.

연속 파라미터 공간(무한한 가능성)을 탐색하는 대신, 격자로 양자화 — **어큐뮬레이터 배열**(accumulator) — 하고 각 에지 점이 자신의 파라미터가 통과하는 빈을 증가시키게 합니다. 어큐뮬레이터의 피크가 이미지의 직선에 대응합니다.

### 이론: 투표 절차

`(ρ, θ)` 매개변수화로:

1. 파라미터 공간을 **양자화**. 전형 해상도: `ρ` 빈 너비 = 1 픽셀, `θ` 빈 너비 = 1°. 대각선 `D`의 이미지에서 `ρ` 범위는 `[-D, D]`이므로 어큐뮬레이터는 `~2D × 180` 빈.
2. 어큐뮬레이터를 0으로 **초기화**.
3. **이진 에지 맵의 각 에지 픽셀** `(x, y)`에 대해 모든 `θ` 빈을 순회하며 `ρ = x cos θ + y sin θ`를 계산하고 해당 `(ρ, θ)` 빈을 **증가**.
4. 어큐뮬레이터에서 **피크 찾기**. 임계값을 넘는 각 피크는 적어도 그만큼의 지지 에지 픽셀을 가진 직선에 대응.

모든 투표 후 어큐뮬레이터 빈의 값은, 그 `(ρ, θ)`가 정의하는 정확한 직선 위에 놓일 에지 픽셀의 수와 같습니다 — 데이터가 그 직선을 얼마나 강하게 지지하는지의 직접 측정.

#### 누락/잡음 데이터에도 작동하는 이유

투표는 각 에지 점이 독립적으로 기여하므로 **가림과 잡음에 강합니다**. 직선의 픽셀 절반이 누락되어도 파라미터 공간의 피크는 작아지지만 여전히 존재합니다. 한 점이 진짜 직선이 아니라 잡음이면 *어떤* 빈에 기여하지만 피크에는 기여하지 않습니다. 구조 없는 잡음이 많은 점은 낮고 평평한 배경에 기여할 뿐 — 진짜 직선만이 집중된 피크를 만듭니다.

### 이론: 일반화

#### 확률적 허프 변환 (`HoughLinesP`)

표준 허프는 이미지당 `O(#에지_픽셀 × #θ_빈)`이며 `(ρ, θ)`만 반환합니다 — 직선이지만 끝점은 없습니다. 확률적 허프 변환은 정확도를 약간 양보하는 대신 속도를 얻고, 끝점도 함께 반환합니다:

- 에지 픽셀의 **무작위 부분집합**만 처리(훨씬 빠름).
- 한 직선이 충분한 지지를 얻으면 더 이상 투표하지 않음.
- 검출된 직선을 따라가며 에지 이미지에서 **실제 끝점** 추적.

OpenCV의 `HoughLinesP`는 `(x1, y1, x2, y2)` 세그먼트를 반환하는데, 실전에서 대개 원하는 것은 이것입니다.

#### 일반 허프 변환

같은 투표 아이디어는 직선과 원뿐 아니라 **임의** 매개변수 형태에도 작동합니다. 일반 형태의 경우, 각 경계점에서 형태의 참조점으로의 오프셋을 인코딩하는 **R-table**을 만듭니다. 검출 시 각 에지 점이 가능한 참조점 위치에 투표합니다. 임의 형태 템플릿을 다룰 수 있지만 메모리 비용이 훨씬 높아 오늘날 거의 쓰이지 않습니다 — 딥러닝 검출기가 이를 대부분 대체했습니다.

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

기하학적으로, 엣지 위의 각 점 (x, y)는 어떤 (ρ, θ) 쌍에 대해 이 방정식을 만족합니다. 핵심 통찰은, 이미지 공간의 한 점이 가능한 (ρ, θ) 값들의 사인파형 *곡선(curve)*으로 매핑되며, 동일 직선 위의 점들은 모두 같은 (ρ, θ)에서 교차하는 곡선을 생성한다는 것입니다 — 그 교차점이 바로 검출된 직선입니다. 기울기-절편 표현(y = mx + b)이 수직선(무한대 기울기)을 특수 처리 없이는 다룰 수 없는 반면, 극좌표 표현은 이러한 예외 없이 모든 직선을 처리할 수 있어 선호됩니다.

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

# Visualize the Hough Transform
def visualize_hough_space(image_path):
    """허프 공간 시각화"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 50, 150)

    # Standard Hough line transform (returns accumulator peaks)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    # Visualization
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # Draw line (extended in both directions)
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

### 허프 변환의 장단점

**장점**
- 노이즈에 강인 (robust to noise)
- 불완전한 형상도 검출 가능
- 여러 인스턴스를 동시에 검출 가능

**단점**
- 계산 비용이 높음
- 파라미터 공간 크기에 비례하는 메모리 필요
- 파라미터 선택에 민감

---

## 2. 허프 직선 변환

단순한 직선 피팅(최소제곱 회귀)은 엣지에 끊김, 가려짐(occlusion), 또는 노이즈 아웃라이어 픽셀이 존재하는 순간 무너집니다 — 총 오차를 최소화하는 방식이기 때문에, 몇 개의 잘못된 점만으로도 추정 직선이 크게 틀어질 수 있습니다. 허프 변환은 *투표(voting)* 메커니즘으로 이 문제를 우회합니다: 각 엣지 픽셀이 속할 수 있는 모든 직선에 독립적으로 투표하고, 많은 독립 투표를 받은 직선만 살아남습니다. 덕분에 연결된 윤곽선이 없어도 끊김과 노이즈에 본질적으로 강인합니다.

### 이론: 매개변수화: 왜 `(m, b)`가 아닌 `(ρ, θ)`인가

기울기-절편 형식 `y = mx + b`는 두 파라미터 `(m, b)`를 가집니다. 하지만:

- **수직선은 무한 기울기를 가집니다.** 완전히 수직인 에지는 유한한 `m`으로 표현할 수 없습니다. 파라미터를 두 경우(수직 vs 비수직)로 나누는 것은 보기 싫고 오류 유발적.
- **기울기 범위가 무한합니다.** 45° 직선은 `m = 1`, 89° 직선은 `m ≈ 57`이며, 이미지 공간 각도에서는 "가깝지만" `m`에서는 멀리 떨어져 있습니다. `m`의 균일 비닝은 각도 커버리지를 극도로 비균일하게 만듭니다.

해결책은 **법선 형식**(또는 극형식):

```
ρ = x cos θ + y sin θ
```

여기서:

- **`ρ`**(rho)는 원점에서 직선까지의 수직 거리.
- **`θ`**(theta)는 이 수직선이 `x` 축과 이루는 각도.

평면 내 모든 직선은 `θ ∈ [0, π)`와 `ρ ∈ ℝ`에서 유일한 `(ρ, θ)`를 가집니다(음수 `ρ` 값은 원점의 "반대편" 직선에 대응하지만 관례상 `θ`를 `[0, 2π)`로 확장하거나 `ρ` 부호를 뒤집어 매핑). 수직선은 `θ = 0`, 수평선은 `θ = π/2`. 모든 파라미터가 유계이며, 균일 비닝이 균일한 각도 커버리지를 줍니다.

고정된 이미지 점 `(x₀, y₀)`에 대해, 그 점을 지나는 직선들의 `(ρ, θ)` 값 집합은 파라미터 공간에서 **사인 곡선**입니다: `ρ = x₀ cos θ + y₀ sin θ`. 각 에지 점의 어큐뮬레이터 곡선은 따라서 사인파입니다.

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

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Hough line transform
    lines = cv2.HoughLines(
        edges,
        rho=1,              # ρ resolution: 1 pixel
        theta=np.pi/180,    # θ resolution: 1 degree
        threshold=100       # minimum number of votes — this is the key quality gate:
                            #   too low → many spurious lines from noise; too high → real lines missed.
                            #   Each edge pixel that lies on a candidate line casts one vote, so threshold
                            #   approximates the minimum pixel length of a line you want to detect.
    )

    result = img.copy()

    if lines is not None:
        print(f"검출된 직선 수: {len(lines)}")

        for line in lines:
            rho, theta = line[0]

            # Polar -> Cartesian
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # Draw infinite line
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

            # Classify by angle (5° tolerance)
            angle_deg = np.degrees(theta)

            if 85 < angle_deg < 95:  # vertical (θ ≈ 90°)
                vertical.append((rho, theta))
                color = (255, 0, 0)  # blue
            elif angle_deg < 5 or angle_deg > 175:  # horizontal (θ ≈ 0° or 180°)
                horizontal.append((rho, theta))
                color = (0, 255, 0)  # green
            else:
                continue

            # Draw the line
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

### 이론: 원 허프 변환

원은 **세 파라미터**가 필요합니다: 중심과 반지름 `(x_c, y_c, r)`. 직선 변환의 모든 내용이 일반화됩니다:

- 각 에지 점 `(x, y)`는 파라미터 공간의 **3D 표면**과 일치합니다: `(x - x_c)² + (y - y_c)² = r²`을 만족하는 모든 `(x_c, y_c, r)`. 그 표면은 `(x_c, y_c, r)` 공간에서 꼭짓점이 `(x, y, 0)`인 원뿔.
- 어큐뮬레이터는 이제 3D, 보통 크기 `W × H × R_max`.
- 각 에지 점마다 원뿔 표면을 투표하는 것은 비쌉니다.

#### 기울기 방향 최적화

핵심 가속: 원의 에지 픽셀에서 **기울기 방향은 반지름을 따라** 중심 쪽 또는 반대쪽을 가리킵니다. 따라서 `(x, y)`에서 기울기 방향(Sobel로부터)을 안다면, 중심 `(x_c, y_c)`는 그 기울기선 위에 있어야 합니다. 모든 가능한 반지름에서 모든 가능한 중심에 투표하는 대신, 기울기 방향으로 `(x, y)`를 통과하는 선 위의 중심에만 투표합니다.

이렇게 하면 에지 픽셀당 투표가 2D 표면에서 1D 집합으로(거리, 즉 반지름으로 매개변수화된 기울기선) 줄어듭니다. OpenCV의 `HoughCircles` 구현이 이 트릭을 사용합니다 — 그래서 방법으로 `HOUGH_GRADIENT`를 전달해야 합니다.

#### 2단계 어큐뮬레이터

OpenCV의 구현은 더 최적화되어 있습니다: 먼저 2D `(x_c, y_c)`에서 누적해 중심을 찾고, 그다음 검출된 각 중심에서 1D `r`로 반지름을 찾습니다. 3D 어큐뮬레이터 탐색을 2D + 1D 탐색으로 바꿔 훨씬 작은 문제로 만듭니다.

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

    # Probabilistic Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,    # minimum line length — rejects short noise fragments; increase
                             #   for road lanes (want long continuous marks), decrease for short dashes.
        maxLineGap=10        # maximum pixel gap allowed inside a single segment — setting this
                             #   higher "bridges" dashed lines into one segment, which is useful for lane
                             #   detection where paint markings have regular gaps.
    )

    result = img.copy()

    if lines is not None:
        print(f"검출된 선분 수: {len(lines)}")

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Mark segment endpoints
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

        # Segment length
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Angle relative to horizontal
        if x2 - x1 != 0:
            angle = np.degrees(np.arctan(abs(y2 - y1) / abs(x2 - x1)))
        else:
            angle = 90

        # Filter: keep only near-horizontal or near-vertical
        if angle < angle_threshold:
            color = (0, 255, 0)  # near horizontal
        elif angle > 90 - angle_threshold:
            color = (255, 0, 0)  # near vertical
        else:
            continue  # ignore diagonals

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

    # Group segments by angle
    groups = defaultdict(list)

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Compute angle
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

        # Quantize angle into bins of angle_threshold width
        angle_group = round(angle / angle_threshold) * angle_threshold
        groups[angle_group].append(line[0])

    merged = []

    for angle, group_lines in groups.items():
        if len(group_lines) == 1:
            merged.append(group_lines[0])
            continue

        # Within the group, merge nearby segments
        # Simple strategy: take min/max along the dominant axis
        all_points = []
        for x1, y1, x2, y2 in group_lines:
            all_points.extend([(x1, y1), (x2, y2)])

        all_points = np.array(all_points)

        # Sort along the dominant axis to pick the two extreme endpoints
        if abs(np.cos(np.radians(angle))) > 0.5:
            # Mostly horizontal: sort by x
            sorted_pts = sorted(all_points, key=lambda p: p[0])
        else:
            # Mostly vertical: sort by y
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

    # Noise reduction (important for circle detection)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Hough circle transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,              # inverse ratio of accumulator resolution — dp=1 means full resolution
                           #   (more memory, more precise); dp=2 halves the accumulator size (faster).
        minDist=50,        # minimum distance between circle centers — prevents the algorithm
                           #   from returning many overlapping circles for the same coin/object.
        param1=100,        # upper Canny threshold; the lower is automatically set to half.
        param2=30,         # accumulator threshold for circle centers — the most sensitive tuning
                           #   knob. Lower values detect more circles (including false positives from noise);
                           #   higher values require a stronger consensus of edge points around the center,
                           #   yielding fewer but more confident detections.
        minRadius=10,      # minimum radius
        maxRadius=100      # maximum radius
    )

    result = img.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for circle in circles[0, :]:
            cx, cy, r = circle

            # Draw circle
            cv2.circle(result, (cx, cy), r, (0, 255, 0), 2)

            # Center point
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

    # Hough circle transform
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

            # Estimate denomination from radius (illustrative)
            if r < 40:
                value = 10
                color = (255, 0, 0)    # blue
            elif r < 55:
                value = 50
                color = (0, 255, 0)    # green
            else:
                value = 100
                color = (0, 0, 255)    # red

            total_value += value

            # Draw
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

    # HOUGH_GRADIENT_ALT: more accurate but slower
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT_ALT,  # alternative algorithm
        dp=1.5,
        minDist=50,
        param1=300,    # edge gradient threshold
        param2=0.9,    # circularity threshold (0-1, higher = stricter)
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

        # Validation
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

            # Display number of detected circles
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

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edges
    edges = cv2.Canny(blurred, 50, 150)

    # Region of interest (trapezoid)
    mask = np.zeros_like(edges)
    vertices = np.array([[
        (0, height),
        (width * 0.45, height * 0.6),
        (width * 0.55, height * 0.6),
        (width, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough line transform
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )

    # Result image
    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Compose with original
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    return result

# Example
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

    # ROI mask
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

            # Slope
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)

            # Classify into left/right by slope
            # Image coordinates: y axis points downward
            # Left lane:  negative slope (/)
            # Right lane: positive slope (\)
            if slope < -0.5:
                left_lines.append(line[0])
            elif slope > 0.5:
                right_lines.append(line[0])

    result = image.copy()

    # Draw left/right lanes
    for x1, y1, x2, y2 in left_lines:
        cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 3)  # blue

    for x1, y1, x2, y2 in right_lines:
        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 3)  # red

    return result, left_lines, right_lines

# Example
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

    # Collect all points
    x_coords = []
    y_coords = []

    for x1, y1, x2, y2 in lines:
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])

    # First-degree polynomial fit (line)
    poly = np.polyfit(y_coords, x_coords, 1)

    # Compute the start and end of the averaged line
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

    # Draw averaged lanes
    left_avg = average_lane_lines(left_lines, height)
    right_avg = average_lane_lines(right_lines, height)

    if left_avg is not None:
        cv2.line(result, (left_avg[0], left_avg[1]),
                 (left_avg[2], left_avg[3]), (255, 0, 0), 5)

    if right_avg is not None:
        cv2.line(result, (right_avg[0], right_avg[1]),
                 (right_avg[2], right_avg[3]), (0, 0, 255), 5)

    # Fill the lane region
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

# Example
img = cv2.imread('road.jpg')
result = detect_lanes_averaged(img)
cv2.imshow('Averaged Lanes', result)
cv2.waitKey(0)
```

### 주차 공간 검출

```python
import cv2
import numpy as np

class ParkingDetector:
    """주차 공간 검출 시스템"""

    def __init__(self):
        self.parking_spaces = []

    def detect_parking_lines(self, image_path):
        """주차선 검출"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binarization
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Edge detection
        edges = cv2.Canny(binary, 50, 150)

        # Line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                               minLineLength=50, maxLineGap=10)

        # Draw lines
        result = img.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return result, lines

    def find_parking_spaces(self, lines, img_shape):
        """직선에서 주차 공간 찾기"""
        if lines is None:
            return []

        # Group parallel lines
        vertical_lines = []
        horizontal_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            if abs(angle) < 45:  # horizontal
                horizontal_lines.append(line[0])
            else:  # vertical
                vertical_lines.append(line[0])

        # Find rectangular parking spaces
        spaces = []
        for v_line in vertical_lines:
            for h_line in horizontal_lines:
                space = self.calculate_space(v_line, h_line)
                if space is not None:
                    spaces.append(space)

        return spaces

    def calculate_space(self, v_line, h_line):
        """주차 공간 계산 (단순화된 구현)"""
        return None

# Example
detector = ParkingDetector()
result, lines = detector.detect_parking_lines('parking.jpg')
cv2.imshow('Parking Line Detection', result)
cv2.waitKey(0)
```

### 문서 엣지 검출

```python
import cv2
import numpy as np

def detect_document_edges(image_path):
    """문서 엣지 검출"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preprocessing
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                           minLineLength=100, maxLineGap=10)

    # Draw lines
    result = img.copy()
    if lines is not None:
        # Group by angle
        horizontal = []
        vertical = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            if abs(angle) < 45:
                horizontal.append(line[0])
                cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
            else:
                vertical.append(line[0])
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Original', img)
    cv2.imshow('Document Edges', result)
    cv2.waitKey(0)

    return result

# Example
result = detect_document_edges('document.jpg')
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

            # Classify into horizontal / vertical
            if 80 < angle < 100:  # vertical
                vertical.append((rho, theta))
            elif angle < 10 or angle > 170:  # horizontal
                horizontal.append((rho, theta))

    # Compute intersections
    intersections = []
    for h_rho, h_theta in horizontal:
        for v_rho, v_theta in vertical:
            # Intersection of two lines
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

    # Draw
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

    # Equalize brightness
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Iris detection (a dark circle)
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

        # Pick the largest circle (the iris)
        for cx, cy, r in sorted(circles[0], key=lambda x: -x[2])[:1]:
            # Iris
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

    # Red mask (red wraps around at 0° and 180° in HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Circle detection
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

## 요약

### 핵심 개념
1. **허프 변환 원리**
   - 이미지 공간에서 파라미터 공간으로의 변환
   - 투표(voting) 메커니즘
   - 극값(local maxima) 검출

2. **직선 검출**
   - 표준 허프 변환 (HoughLines)
   - 확률적 허프 변환 (HoughLinesP)
   - 파라미터 조정

3. **원 검출**
   - 3차원 파라미터 공간 (x, y, r)
   - 파라미터 최적화
   - 다중 원 검출

4. **실용 응용**
   - 차선 검출
   - 주차 공간 검출
   - 문서 엣지 검출
   - 객체 카운팅

5. **성능 최적화**
   - 파라미터 최적화
   - ROI 처리
   - 다중 스케일 처리

### 파라미터 튜닝 가이드
- **rho, theta**: 해상도가 높을수록 더 정확하지만 느림
- **threshold**: 높을수록 직선 수가 줄어들지만 더 강한 직선만 검출
- **minLineLength**: 최소 직선 길이 임계값
- **maxLineGap**: 선분 내 최대 허용 간격
- **param1**: 엣지 검출 임계값
- **param2**: 누적 배열 임계값

### 주요 사항
- 전처리가 핵심 (엣지 검출, 노이즈 제거)
- 이미지 특성에 따라 파라미터 값이 크게 달라짐
- 실시간 처리에는 성능 최적화 필요
- ROI로 계산 비용 절감 가능
- 다중 스케일 처리로 다양한 크기의 객체 탐지 가능

---

## 다음 단계

- [히스토그램 분석 (Histogram Analysis)](./12_Histogram_Analysis.md) - calcHist, equalizeHist, CLAHE

---

## 참고 자료

- [OpenCV Hough Line Transform](https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html)
- [OpenCV Hough Circle Transform](https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html)
- [Lane Detection Tutorial](https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132)
