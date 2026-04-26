# 모폴로지 연산

## 개요

모폴로지 연산(Morphological Operations)은 이진 이미지나 그레이스케일 이미지의 형태를 기반으로 하는 연산입니다. 주로 노이즈 제거, 객체 분리, 홀 채우기 등에 사용됩니다. 이 문서에서는 구조 요소의 개념부터 다양한 모폴로지 연산의 활용까지 학습합니다.

**난이도**: ⭐⭐ (초급-중급)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 구조 요소(Structuring Element) 이해
2. 침식(Erosion)과 팽창(Dilation) 연산
3. 열기(Opening)와 닫기(Closing) 연산
4. 그래디언트, 탑햇, 블랙햇 연산
5. 노이즈 제거 및 객체 분리 응용

---

## 목차

OpenCV 함수 참조에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. 침식과 팽창의 집합론적 정의, "열기 = 침식 후 팽창"이 작은 객체를 제거하는 이유, 그리고 이 연산들을 완전한 대수로 만드는 쌍대성을 다룹니다.

1. [모폴로지 연산 개요](#1-모폴로지-연산-개요)
2. [구조 요소 - getStructuringElement()](#2-구조-요소---getstructuringelement)
3. [침식 - erode()](#3-침식---erode)
4. [팽창 - dilate()](#4-팽창---dilate)
5. [열기와 닫기 - morphologyEx()](#5-열기와-닫기---morphologyex)
6. [그래디언트, 탑햇, 블랙햇](#6-그래디언트-탑햇-블랙햇)
7. [실전 응용](#7-실전-응용)
8. [연습 문제](#8-연습-문제)
9. [다음 단계](#9-다음-단계)
10. [참고 자료](#10-참고-자료)

---

## 이론과 원리

모폴로지는 1960년대에 Matheron과 Serra가 형태를 분석하기 위한 **집합론적** 틀로 개발했습니다. 컨볼루션 필터가 이미지를 신호로 보고 가중 평균을 적용하는 것과 달리, 모폴로지 연산자는 이진 이미지를 **픽셀 위치의 집합**으로 다루고, *구조 요소(structuring element)*라 불리는 작은 참조 형태로 그 집합을 조작합니다. 결과로 나오는 대수는 단순하고 증명 가능하며, 소수의 빌딩 블록이 놀라울 정도로 다양한 형태 처리 도구로 조합됩니다.

이 섹션은 다음을 다룹니다:

- **(A) 집합으로서의 이미지** — 이진 모폴로지가 컨볼루션과 근본적으로 다른 이유.
- **(B) 두 기본 연산: 침식과 팽창** — 정의, 기하학적 의미, 쌍대성.
- **(C) 열기와 닫기** — 실제로 대개 필요한 두 복합 연산.
- **(D) 그레이스케일 확장** — `{0, 1}` 이미지에서 `[0, 255]` 이미지로 같은 아이디어가 일반화되는 방법.
- **(E) 그래디언트, 탑햇, 블랙햇** — 특정 구조 특징을 분리하는 재조합 연산.

### A. 집합으로서의 이미지

이진 모폴로지에서 이미지 `A`는 "전경(foreground)" 픽셀 위치의 집합으로 취급됩니다:

```
A = { (x, y) : I(x, y) = 1 }
```

**구조 요소** `B`는 프로브 모양을 나타내는 작은 집합(보통 3×3 또는 5×5)입니다 — 원, 십자, 수평선 등. `B`의 원점은 그 픽셀 중 하나입니다(보통 중심). 모폴로지 연산자는 기본 집합 연산을 써서 `A`와 `B`를 조합합니다.

이 틀은 컨볼루션과의 결정적 차이를 분명히 합니다. 컨볼루션은 평균을 계산합니다 — 각 픽셀의 결과는 이웃의 *모든* 값에 의존합니다. 모폴로지는 기하학적 질문을 합니다 — 주어진 위치에서 `B`가 `A` 내에 *들어가는가*, 또는 `B`가 `A`를 *건드리는가*. 답은 이진이고, 연산은 **비선형**입니다: 05 레슨의 LSI 도구들은 여기서 적용되지 않습니다.

### B. 침식과 팽창: 기본 연산

#### B.1 침식 (`⊖`)

```
A ⊖ B = { x : B_x ⊆ A }
```

여기서 `B_x`는 `B`의 원점을 `x`에 두도록 평행이동한 것. 말로 풀면: `A`를 `B`로 침식한 것은, `B`가 `A` 내부에 *완전히* 들어가는 모든 위치의 집합입니다. 경계 근처의 픽셀 중 `B`의 어떤 구성원이라도 `A` 바깥에 놓이는 곳은 제거됩니다.

기하학적 직관:

- 침식은 객체를 **축소**합니다. `B`의 반지름만큼 한 겹 벗겨냅니다.
- `B`보다 작은 객체는 **완전히 사라집니다** — 침식이 노이즈 점을 제거.
- 블롭 사이의 좁은 연결은 `B`가 그 연결보다 넓을 때 **끊어집니다** — 침식이 맞닿은 객체를 분리.

두 번째 동등한 정의는 침식을 OpenCV의 국소 최솟값 구현과 직접 연결합니다:

```
(A ⊖ B)(x) = min_{b ∈ B}  A(x + b)
```

구조 요소 아래의 모든 픽셀이 1일 때만 출력 픽셀이 1 — `min`이 바로 이를 포착합니다.

#### B.2 팽창 (`⊕`)

```
A ⊕ B = { x : B̂_x ∩ A ≠ ∅ }
```

여기서 `B̂`은 `B`의 반사(대칭 구조 요소에서는 `B`와 같음, 일반적인 경우). 말로 풀면: 팽창은 `B`가 `A`를 *건드리는* 위치의 집합 — `B`의 구성원 중 적어도 하나가 `A` 내부에 놓이는 곳.

기하학적 직관:

- 팽창은 객체를 `B`의 반지름만큼 **성장**시킵니다.
- 객체 내부의 작은 구멍이 `B`보다 작으면 **채워집니다**.
- 객체 간 좁은 간격이 **메워집니다**.

침식과 마찬가지로 국소 최댓값 형식:

```
(A ⊕ B)(x) = max_{b ∈ B}  A(x - b)
```

#### B.3 쌍대성 (Duality)

침식과 팽창은 독립적이지 않습니다 — 여집합 아래에서 **쌍대**입니다:

```
(A ⊖ B)ᶜ = Aᶜ ⊕ B̂
```

전경을 침식하는 것은 배경을 팽창하는 것과 같습니다. 그래서 팽창은 반전 → 침식 → 반전으로 구현할 수 있습니다. 또한 침식으로 만든 모든 형태 제거 기법은 팽창을 이용한 형태 채움 대응물을 가진다는 의미이기도 합니다.

### C. 열기와 닫기: 대개 진짜 원하는 것

원시 침식은 모든 것을 줄이고, 원시 팽창은 모든 것을 키웁니다. 둘 중 어느 것도 대개 단독으로는 원하는 것이 아닙니다. *큰* 객체(실제 특징)를 줄이지 않고 *작은* 객체(노이즈)만 제거하거나, 객체를 키우지 않으면서 *작은* 구멍만 채우길 원합니다. 그것이 복합 연산자가 하는 일입니다.

#### C.1 열기 (`∘`)

```
A ∘ B = (A ⊖ B) ⊕ B
```

"침식 후 같은 구조 요소로 팽창." 직관:

- 침식이 `B`보다 작은 모든 것을 삭제하고 생존자에게서 한 겹을 벗깁니다.
- 팽창이 같은 양만큼 생존자를 다시 키웁니다 — 그래서 침식을 견딘 특징은 거의 원래 크기로 돌아옵니다.
- 하지만 침식에서 죽은 객체는 되살릴 수 없습니다.

순 효과: **작은 객체는 사라지고, 큰 객체는 보존(경계는 매끄럽게)**. 소금 노이즈 제거와 분할 마스크 정리를 위한 표준 도구. 수학적으로 `A ∘ B`는 `B`의 평행이동 복사본들의 합집합으로 쓸 수 있는 `A`의 가장 큰 부분집합 — `A`의 "`B`를 존중하는" 부분.

#### C.2 닫기 (`•`)

```
A • B = (A ⊕ B) ⊖ B
```

"팽창 후 침식." 열기와의 쌍대성으로:

- 팽창이 `B`보다 작은 간격과 구멍을 채웁니다.
- 이어지는 침식이 객체 경계를 거의 원래 위치로 복원 — 하지만 채워진 영역은 유지.

순 효과: **객체 내부의 작은 구멍은 채워지고, 객체 간 좁은 간격은 닫히며, 주요 형태는 보존**. 후추 노이즈 제거와 조각난 객체를 잇기 위한 표준 도구. 닫기와 열기도 쌍대입니다: `(A ∘ B)ᶜ = Aᶜ • B̂`.

#### C.3 반복과 멱등성 (Idempotence)

핵심 성질: `(A ∘ B) ∘ B = A ∘ B`. 열기를 두 번 적용해도 한 번과 같은 결과 — **멱등**합니다. 닫기도 마찬가지. 이 성질 덕분에 한 번이면 충분하고, 반복은 도움이 되지 않는다는 것을 알 수 있습니다.

반면 `(A ⊖ B) ⊖ B = A ⊖ (B ⊕ B)`는 멱등이 *아닙니다*. 침식을 반복하면 계속 축소됩니다. 그래서 OpenCV가 `erode`/`dilate`에는 `iterations` 파라미터를 노출하지만 열기/닫기의 `morphologyEx`에는 없습니다 — 멱등 연산을 반복할 이유가 없습니다.

### D. 그레이스케일 모폴로지

그레이스케일 이미지의 경우 집합론적 관점이 일반화됩니다: 이미지를 3D 속의 *표면* `I(x, y)`로 보고, 구조 요소가 이 표면을 아래에서 떠받치거나(침식) 위에서 덮어씌웁니다(팽창). 실용적 정의는 §B의 국소 최소/최대 형식입니다:

```
(I ⊖ B)(x, y) = min_{(i,j) ∈ B}  I(x + i, y + j)
(I ⊕ B)(x, y) = max_{(i,j) ∈ B}  I(x + i, y + j)
```

이것들을 때로 "min 필터", "max 필터"라 부릅니다. §B–§C의 모든 내용이 그대로 이어집니다. 열기는 여전히 `B`보다 작은 밝은 구조를 제거하고, 닫기는 `B`보다 작은 어두운 구조를 채우며, 둘 다 멱등입니다.

### E. 그래디언트, 탑햇, 블랙햇: 파생 연산자

기본 연산으로 만든 세 복합 연산자가 특정 구조 특징을 추출합니다:

#### E.1 모폴로지 그래디언트

```
gradient(A) = (A ⊕ B) - (A ⊖ B)
```

팽창이 객체를 "두껍게", 침식이 "얇게" 하고, 그 차이가 정확히 경계층입니다. 이진 이미지에서는 에지 픽셀을 주고, 그레이스케일에서는 대략적인 기울기 크기를 줍니다. 균일한 배경 위의 날카로운 경계라는 특정 경우에 Sobel 기반 에지의 더 단순한 대안으로 유용합니다.

#### E.2 탑햇 (white top-hat)

```
tophat(A) = A - (A ∘ B)
```

원본 이미지에서 그 열기를 뺀 것. 열기가 작은 밝은 특징을 제거하므로, 이 차이는 **작은 밝은 특징 자체를 분리**합니다 — 열기가 버린 것들. 탑햇은 구조 요소보다 작은 밝은 객체를 검출하는 데 이상적입니다(종이 위 글자, 이미지 속 별, 어두운 배경 위 밝은 점). 특히 배경이 균일하지 않을 때 — 탑햇이 암묵적으로 배경 변화를 정규화합니다.

#### E.3 블랙햇

```
blackhat(A) = (A • B) - A
```

이미지의 닫기에서 원본을 뺀 것 — 쌍대에 의해, **밝은 배경 위의 작은 어두운 특징**을 분리합니다. 밝기가 다양한 문서에서 어두운 글자를 추출하거나, 밝은 표면의 어두운 결함, 또는 "`B`보다 작은 어두운 것" 검출 문제에 유용합니다.

### 이론에서 아래 함수들로

- `cv2.getStructuringElement(shape, size)` — `B`를 만듦. Shape 선택(`MORPH_RECT`, `MORPH_CROSS`, `MORPH_ELLIPSE`)은 프로브가 어떤 기하를 존중하는지를 결정 — 등방적 동작을 원하면 `MORPH_ELLIPSE`, 속도는 `MORPH_RECT`, 방향 프로빙은 cross 또는 line.
- `cv2.erode(img, kernel, iterations=n)` / `cv2.dilate` — §B.1 / §B.2 기본 연산. `iterations=n`은 커널을 `n`번 자기 팽창으로 키운 것과 동등.
- `cv2.morphologyEx(img, op, kernel)` — 복합 연산을 위한 통합 진입점. `op` 플래그 선택: `MORPH_OPEN` (§C.1), `MORPH_CLOSE` (§C.2), `MORPH_GRADIENT` (§E.1), `MORPH_TOPHAT` (§E.2), `MORPH_BLACKHAT` (§E.3), `MORPH_HITMISS` (hit-or-miss 변환, 패턴 검출기).

---

## 1. 모폴로지 연산 개요

가우시안 블러와 같은 픽셀 단위 필터는 모든 픽셀을 동등하게 취급합니다. 하지만 현실에서는 객체 경계를 흐리게 하지 않고 스펙클 노이즈를 제거하거나, 붙어 있는 두 세포를 분리해야 하는 경우처럼 객체의 *형태*를 기반으로 처리해야 하는 경우가 많습니다. 모폴로지 연산은 형태를 가진 마스크로 이미지 구조를 탐색함으로써 이 간극을 채웁니다. 이진 이미지 정리와 형태 분석을 위한 표준 도구입니다.

### 모폴로지란?

```
┌─────────────────────────────────────────────────────────────────┐
│                  Morphological Operations Overview               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Morphology = Study of shape                                   │
│   Operations based on the shape of images                       │
│                                                                 │
│   Main Uses:                                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  1. Noise removal     - Remove small noise dots          │   │
│   │  2. Hole filling      - Fill holes inside objects        │   │
│   │  3. Object separation - Separate connected objects       │   │
│   │  4. Object connection - Connect disconnected parts       │   │
│   │  5. Edge detection    - Morphological gradient           │   │
│   │  6. Skeletonization  - Extract object skeleton           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Basic Operations:                                             │
│   - Erosion: Shrink objects                                     │
│   - Dilation: Expand objects                                    │
│                                                                 │
│   Combined Operations:                                          │
│   - Opening = Erosion → Dilation                                │
│   - Closing = Dilation → Erosion                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 작동 원리

모폴로지 연산은 **구조 요소(Structuring Element)**라는 작은 마스크를 이미지 위로 이동시키며 픽셀 값을 결정합니다.

---

## 2. 구조 요소 - getStructuringElement()

### 구조 요소란?

```
┌─────────────────────────────────────────────────────────────────┐
│                        Structuring Element                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Structuring Element = A small binary matrix used in operations│
│                                                                 │
│   Main Shapes:                                                  │
│                                                                 │
│   MORPH_RECT (Rectangle)   MORPH_CROSS (Cross)    MORPH_ELLIPSE │
│   ┌───┬───┬───┐           ┌───┬───┬───┐        ┌───┬───┬───┐  │
│   │ 1 │ 1 │ 1 │           │ 0 │ 1 │ 0 │        │ 0 │ 1 │ 0 │  │
│   ├───┼───┼───┤           ├───┼───┼───┤        ├───┼───┼───┤  │
│   │ 1 │ 1 │ 1 │           │ 1 │ 1 │ 1 │        │ 1 │ 1 │ 1 │  │
│   ├───┼───┼───┤           ├───┼───┼───┤        ├───┼───┼───┤  │
│   │ 1 │ 1 │ 1 │           │ 0 │ 1 │ 0 │        │ 0 │ 1 │ 0 │  │
│   └───┴───┴───┘           └───┴───┴───┘        └───┴───┴───┘  │
│   All directions          Vertical/Horizontal   Elliptical      │
│                                                                 │
│   Effect by Size:                                               │
│   - Small size (3x3): Fine processing                           │
│   - Large size (7x7, 9x9): Strong effect                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 구조 요소 생성

```python
import cv2
import numpy as np

# getStructuringElement(shape, ksize, anchor=(-1,-1))
# shape: Structuring element shape
# ksize: (width, height) size
# anchor: Reference point (default: center)

# Why MORPH_RECT: treats all directions equally — use when objects have
# roughly rectangular or straight edges (text, PCB traces)
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
print("RECT (5x5):\n", rect_kernel)

# Cross
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
print("\nCROSS (5x5):\n", cross_kernel)

# Why MORPH_ELLIPSE: approximates a disk — preferred for circular/rounded
# objects (cells, coins) because it avoids introducing rectangular artifacts
# at diagonals that MORPH_RECT would create
ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
print("\nELLIPSE (5x5):\n", ellipse_kernel)

# Custom structuring element
custom_kernel = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.uint8)
```

### 구조 요소 시각화

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

shapes = [
    ('RECT', cv2.MORPH_RECT),
    ('CROSS', cv2.MORPH_CROSS),
    ('ELLIPSE', cv2.MORPH_ELLIPSE)
]

sizes = [(5, 5), (7, 7), (11, 11)]

fig, axes = plt.subplots(len(shapes), len(sizes), figsize=(12, 10))

for i, (name, shape) in enumerate(shapes):
    for j, size in enumerate(sizes):
        kernel = cv2.getStructuringElement(shape, size)
        axes[i, j].imshow(kernel, cmap='gray')
        axes[i, j].set_title(f'{name} {size}')
        axes[i, j].axis('off')

plt.tight_layout()
plt.show()
```

---

## 3. 침식 - erode()

침식(Erosion)은 "전경이 이 커널 형태의 영역을 완전히 덮는가?"라는 질문에 답합니다. 그렇지 않으면 해당 픽셀은 제거됩니다. 이 덕분에 커널을 완전히 덮을 수 없는 고립된 노이즈 점을 제거하고, 접촉된 객체들 사이의 얇은 연결을 끊어 개별 계수가 가능하도록 만드는 데 최적의 도구입니다.

### 침식 연산 원리

```
┌─────────────────────────────────────────────────────────────────┐
│                         Erosion                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Principle:                                                    │
│   - Move the structuring element across the image               │
│   - Set center pixel to 1 only if all pixels under the         │
│     structuring element are 1                                   │
│   - If any pixel is 0, center pixel becomes 0                   │
│                                                                 │
│   Effect:                                                       │
│   - Shrinks foreground (white) area                             │
│   - Removes small noise                                         │
│   - Separates connected objects                                 │
│   - Smooths boundaries                                          │
│                                                                 │
│   Example:                                                      │
│   Original:           After Erosion (3x3):                      │
│   ┌─────────────┐     ┌─────────────┐                          │
│   │ ████████████│     │   ████████  │                          │
│   │ ████████████│ ──▶ │   ████████  │                          │
│   │ ████████████│     │   ████████  │                          │
│   │ ████████████│     │             │                          │
│   └─────────────┘     └─────────────┘                          │
│   Borders shrink by 1 pixel                                     │
│                                                                 │
│   Noise Removal:                                                │
│   ┌─────────────┐     ┌─────────────┐                          │
│   │ ██  ■  ████ │     │ ██     ███  │                          │
│   │ ████  ████  │ ──▶ │  ██    ██   │  Small dots (■) removed  │
│   │    ■  ████  │     │       ███   │                          │
│   └─────────────┘     └─────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 침식 사용법

```python
import cv2
import numpy as np

# Prepare binary image
img = cv2.imread('binary_image.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Why MORPH_RECT (3x3): the smallest kernel that still has a meaningful neighborhood;
# larger kernels erode more aggressively and may destroy the objects you want to keep
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# erode(src, kernel, iterations=1)
# Why iterations: repeating erosion N times is equivalent to eroding with a larger
# kernel but cheaper to compute — use iterations to tune removal strength incrementally
eroded_1 = cv2.erode(binary, kernel, iterations=1)
eroded_2 = cv2.erode(binary, kernel, iterations=2)
eroded_3 = cv2.erode(binary, kernel, iterations=3)

cv2.imshow('Original', binary)
cv2.imshow('Eroded 1x', eroded_1)
cv2.imshow('Eroded 2x', eroded_2)
cv2.imshow('Eroded 3x', eroded_3)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 침식 테스트 이미지 생성

```python
import cv2
import numpy as np

# Create test image
img = np.zeros((300, 400), dtype=np.uint8)

# Large rectangle
cv2.rectangle(img, (50, 50), (150, 150), 255, -1)

# Small noise dots
for _ in range(50):
    x, y = np.random.randint(200, 350), np.random.randint(50, 250)
    cv2.circle(img, (x, y), 2, 255, -1)

# Connected circles
cv2.circle(img, (280, 150), 40, 255, -1)
cv2.circle(img, (320, 150), 40, 255, -1)

# Apply erosion
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
eroded = cv2.erode(img, kernel, iterations=1)

cv2.imshow('Original', img)
cv2.imshow('Eroded', eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 4. 팽창 - dilate()

팽창(Dilation)은 침식의 쌍대(Dual) 연산입니다. "전경이 이 커널 영역의 어느 부분이라도 닿는가?"라고 묻습니다. 그렇다면 픽셀이 설정됩니다. 따라서 끊어진 획을 다시 연결하고 작은 틈을 채우는 데 이상적입니다. 전체 객체 크기가 유지되도록 항상 침식과 짝을 이루어 사용합니다(열기 또는 닫기 형태로).

### 팽창 연산 원리

```
┌─────────────────────────────────────────────────────────────────┐
│                         Dilation                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Principle:                                                    │
│   - Move the structuring element across the image               │
│   - Set center pixel to 1 if any pixel under the               │
│     structuring element is 1                                    │
│   - Opposite of erosion                                         │
│                                                                 │
│   Effect:                                                       │
│   - Expands foreground (white) area                             │
│   - Fills holes                                                 │
│   - Connects broken parts                                       │
│   - Emphasizes objects                                          │
│                                                                 │
│   Example:                                                      │
│   Original:           After Dilation (3x3):                     │
│   ┌─────────────┐     ┌─────────────┐                          │
│   │   ██████    │     │ ████████████│                          │
│   │   ██████    │ ──▶ │ ████████████│                          │
│   │   ██████    │     │ ████████████│                          │
│   └─────────────┘     └─────────────┘                          │
│   Borders expand by 1 pixel                                     │
│                                                                 │
│   Connect Broken Parts:                                         │
│   ┌─────────────┐     ┌─────────────┐                          │
│   │ ██      ██  │     │ ████    ████│                          │
│   │ ██  ..  ██  │ ──▶ │ ██████████  │  Dotted line connected  │
│   │ ██      ██  │     │ ████    ████│                          │
│   └─────────────┘     └─────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 팽창 사용법

```python
import cv2
import numpy as np

# Prepare binary image
img = cv2.imread('binary_image.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# dilate(src, kernel, iterations=1)
dilated_1 = cv2.dilate(binary, kernel, iterations=1)
dilated_2 = cv2.dilate(binary, kernel, iterations=2)
dilated_3 = cv2.dilate(binary, kernel, iterations=3)

cv2.imshow('Original', binary)
cv2.imshow('Dilated 1x', dilated_1)
cv2.imshow('Dilated 2x', dilated_2)
cv2.imshow('Dilated 3x', dilated_3)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 침식과 팽창 비교

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Test image
img = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
cv2.circle(img, (100, 100), 20, 0, -1)  # Inner hole

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

eroded = cv2.erode(img, kernel, iterations=1)
dilated = cv2.dilate(img, kernel, iterations=1)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')

axes[1].imshow(eroded, cmap='gray')
axes[1].set_title('Eroded (Shrink)')

axes[2].imshow(dilated, cmap='gray')
axes[2].set_title('Dilated (Expand)')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## 5. 열기와 닫기 - morphologyEx()

단순 침식은 객체를 영구적으로 축소하고, 단순 팽창은 객체를 팽창시킵니다. 열기(Opening)와 닫기(Closing)는 두 연산을 결합하여 객체 크기를 대략 유지하면서 *특정 유형의 결함*(노이즈 점 또는 홀)만을 대상으로 합니다. 이 대칭성 때문에 실제 처리 파이프라인에서는 단독 침식/팽창보다 열기/닫기가 선호됩니다.

### 열기 (Opening)

```
┌─────────────────────────────────────────────────────────────────┐
│                      Opening                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Opening = Erosion → Dilation                                  │
│                                                                 │
│   Effect:                                                       │
│   - Removes small noise (dots)                                  │
│   - Maintains overall object size approximately                 │
│   - Breaks thin connections                                     │
│                                                                 │
│   Original    Erosion      Dilation (Opening result)            │
│   ┌──────┐    ┌──────┐    ┌──────┐                              │
│   │██ ■ █│    │█     │    │██   █│                              │
│   │██████│ ─▶ │ ████ │ ─▶ │██████│                              │
│   │  ■ ██│    │    █ │    │    ██│                              │
│   └──────┘    └──────┘    └──────┘                              │
│   Small dots (■) removed                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 닫기 (Closing)

```
┌─────────────────────────────────────────────────────────────────┐
│                      Closing                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Closing = Dilation → Erosion                                  │
│                                                                 │
│   Effect:                                                       │
│   - Fills small holes                                           │
│   - Maintains overall object size approximately                 │
│   - Connects broken parts                                       │
│                                                                 │
│   Original    Dilation     Erosion (Closing result)             │
│   ┌──────┐    ┌──────┐    ┌──────┐                              │
│   │██████│    │██████│    │██████│                              │
│   │██○ ██│ ─▶ │██████│ ─▶ │██████│                              │
│   │██████│    │██████│    │██████│                              │
│   └──────┘    └──────┘    └──────┘                              │
│   Inner hole (○) filled                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### morphologyEx() 사용법

```python
import cv2
import numpy as np

img = cv2.imread('binary_image.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Why (5,5): kernel must be larger than the noise/holes you want to remove;
# a 5x5 kernel removes features smaller than ~5 pixels in diameter
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# morphologyEx(src, op, kernel, iterations=1)
# op: Operation type

# Why MORPH_OPEN first: erosion removes small noise dots; the subsequent dilation
# restores the larger objects to their original size (erode → dilate = opening)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Closing: Hole filling (dilate → erode; expands to fill holes, then contracts back)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Why open before close: opening on the raw image avoids noise dots being "healed"
# into the object by the closing step — order matters
clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Original', binary)
cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)
cv2.imshow('Open + Close', clean)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 열기와 닫기 비교 테스트

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Test image: Rectangle with noise + holes
img = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), 255, -1)

# Add noise (small dots)
noise = img.copy()
for _ in range(30):
    x, y = np.random.randint(10, 45), np.random.randint(10, 190)
    cv2.circle(noise, (x, y), 2, 255, -1)
for _ in range(30):
    x, y = np.random.randint(155, 190), np.random.randint(10, 190)
    cv2.circle(noise, (x, y), 2, 255, -1)

# Add holes (inside object)
holes = noise.copy()
for _ in range(10):
    x, y = np.random.randint(60, 140), np.random.randint(60, 140)
    cv2.circle(holes, (x, y), 3, 0, -1)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

opening = cv2.morphologyEx(holes, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(holes, cv2.MORPH_CLOSE, kernel)
both = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(holes, cmap='gray')
axes[0, 0].set_title('Original (Noise + Holes)')

axes[0, 1].imshow(opening, cmap='gray')
axes[0, 1].set_title('Opening (Noise Removed)')

axes[1, 0].imshow(closing, cmap='gray')
axes[1, 0].set_title('Closing (Holes Filled)')

axes[1, 1].imshow(both, cmap='gray')
axes[1, 1].set_title('Open + Close')

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## 6. 그래디언트, 탑햇, 블랙햇

### 모폴로지 그래디언트

```
┌─────────────────────────────────────────────────────────────────┐
│                   Morphological Gradient                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Gradient = Dilation - Erosion                                 │
│                                                                 │
│   Effect: Extract object outline (boundary)                     │
│                                                                 │
│   Original          Dilation           Erosion                  │
│   ┌──────┐         ┌──────┐         ┌──────┐                   │
│   │ ████ │         │██████│         │  ██  │                   │
│   │ ████ │    -    │██████│    =    │  ██  │                   │
│   │ ████ │         │██████│         │  ██  │                   │
│   └──────┘         └──────┘         └──────┘                   │
│                                                                 │
│   Gradient Result:                                              │
│   ┌──────┐                                                      │
│   │ ████ │  → Only outline remains                              │
│   │ █  █ │                                                      │
│   │ ████ │                                                      │
│   └──────┘                                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 탑햇과 블랙햇

```
┌─────────────────────────────────────────────────────────────────┐
│                    Top-hat / Black-hat                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Top-hat = Original - Opening                                  │
│   - Extract small bright parts from bright areas                │
│   - Detect small objects brighter than background               │
│                                                                 │
│   Black-hat = Closing - Original                                │
│   - Extract small dark parts from dark areas                    │
│   - Detect small holes/objects darker than background           │
│                                                                 │
│   Applications:                                                 │
│   - Correct images with uneven illumination                     │
│   - Remove shadows from document images                         │
│   - Detect small defects                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 구현 및 사용

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

# Morphological gradient
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# Top-hat
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

# Black-hat
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

# Manual calculation (for verification)
dilated = cv2.dilate(img, kernel)
eroded = cv2.erode(img, kernel)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

gradient_manual = dilated - eroded
tophat_manual = img - opening
blackhat_manual = closing - img

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original')

axes[0, 1].imshow(gradient, cmap='gray')
axes[0, 1].set_title('Gradient (Edge)')

axes[0, 2].imshow(tophat, cmap='gray')
axes[0, 2].set_title('Top Hat (Bright spots)')

axes[1, 0].imshow(blackhat, cmap='gray')
axes[1, 0].set_title('Black Hat (Dark spots)')

# Enhance contrast using top-hat + black-hat
enhanced = cv2.add(img, tophat)
enhanced = cv2.subtract(enhanced, blackhat)
axes[1, 1].imshow(enhanced, cmap='gray')
axes[1, 1].set_title('Enhanced (Top+Black Hat)')

for ax in axes.flatten():
    ax.axis('off')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
```

### 모든 모폴로지 연산 정리

```python
import cv2

# List of operations available in morphologyEx()
operations = {
    cv2.MORPH_ERODE: "Erode",
    cv2.MORPH_DILATE: "Dilate",
    cv2.MORPH_OPEN: "Open (Erode + Dilate)",
    cv2.MORPH_CLOSE: "Close (Dilate + Erode)",
    cv2.MORPH_GRADIENT: "Gradient (Dilate - Erode)",
    cv2.MORPH_TOPHAT: "Top Hat (Src - Open)",
    cv2.MORPH_BLACKHAT: "Black Hat (Close - Src)",
    cv2.MORPH_HITMISS: "Hit-Miss (Pattern Matching)"
}

for op, name in operations.items():
    print(f"{op}: {name}")
```

---

## 7. 실전 응용

### 노이즈 제거 파이프라인

```python
import cv2
import numpy as np

def remove_noise_morphology(binary_img, noise_size=3):
    """
    Remove noise using morphological operations

    Parameters:
    - binary_img: Binary image
    - noise_size: Maximum size of noise to remove
    """
    # Why noise_size * 2 + 1: the kernel must fully contain the largest noise dot
    # (radius noise_size → diameter noise_size*2) and be odd for a centered anchor
    kernel_size = noise_size * 2 + 1
    # Why MORPH_ELLIPSE: circular objects (cells, blobs) are better modeled with a
    # disk-shaped kernel — avoids introducing rectangular bias at diagonals
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    # Opening to remove small noise dots (erode kills noise, dilate restores objects)
    cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # Closing to fill small holes (dilate bridges gaps, erode restores boundaries)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned


# Usage example
img = cv2.imread('noisy_document.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
cleaned = remove_noise_morphology(binary, noise_size=2)
```

### 객체 분리

```python
import cv2
import numpy as np

def separate_objects(binary_img, erosion_iterations=3):
    """
    Separate connected objects
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Why multiple erosion iterations instead of a large kernel: iterating with a
    # small kernel is cheaper and lets you tune separation strength without rebuilding
    # the structuring element; each pass peels off one layer from every boundary
    eroded = cv2.erode(binary_img, kernel, iterations=erosion_iterations)

    # Distance transform to find center points — the peak of the distance map is
    # the point farthest from any background pixel, i.e., the object center
    dist_transform = cv2.distanceTransform(eroded, cv2.DIST_L2, 5)
    # Why 0.5 * max: keeps only the top half of distance values, retaining confident
    # object cores while discarding ambiguous border regions
    _, sure_fg = cv2.threshold(
        dist_transform, 0.5 * dist_transform.max(), 255, 0
    )
    sure_fg = np.uint8(sure_fg)

    return eroded, sure_fg


# Usage example
img = cv2.imread('connected_circles.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
separated, centers = separate_objects(binary)
```

### 문서 이미지 전처리

```python
import cv2
import numpy as np

def preprocess_document(img):
    """
    Document image preprocessing (shadow removal + binarization)
    """
    # Grayscale conversion
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Top-hat to extract bright background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    # Black-hat to correct shadows/dark areas
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Subtract black-hat from original (shadow removal effect)
    no_shadow = cv2.add(gray, blackhat)

    # Adaptive binarization
    binary = cv2.adaptiveThreshold(
        no_shadow, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 15
    )

    # Noise removal
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)

    return binary


# Usage example
img = cv2.imread('document_with_shadow.jpg')
result = preprocess_document(img)
```

### 스켈레톤화 (Skeletonization)

```python
import cv2
import numpy as np

def skeletonize(img):
    """
    Extract skeleton using morphological operations
    """
    skeleton = np.zeros_like(img)
    temp = img.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        # Opening operation
        opened = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)

        # Calculate difference
        diff = cv2.subtract(temp, opened)

        # Erosion
        temp = cv2.erode(temp, kernel)

        # Add to skeleton
        skeleton = cv2.bitwise_or(skeleton, diff)

        # Stop if no more white pixels
        if cv2.countNonZero(temp) == 0:
            break

    return skeleton


# Usage example
img = cv2.imread('character.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
skeleton = skeletonize(binary)
```

---

## 8. 연습 문제

### 연습 1: 구조 요소 효과 비교

동일한 이진 이미지에 대해 세 가지 구조 요소(RECT, CROSS, ELLIPSE)를 사용하여 침식과 팽창을 적용하고, 결과의 차이를 분석하세요.

### 연습 2: 문자 두께 조절

손글씨 이미지에서 문자의 두께를 조절하는 함수를 작성하세요:
- 양수 값: 팽창으로 두껍게
- 음수 값: 침식으로 얇게

```python
def adjust_stroke_width(img, amount):
    """
    amount > 0: Thicken
    amount < 0: Thin
    """
    pass
```

### 연습 3: 경계 추출 비교

다음 세 가지 방법으로 객체의 경계를 추출하고 비교하세요:
1. 모폴로지 그래디언트
2. Canny 엣지 검출
3. findContours

### 연습 4: 점자 인식 전처리

점자 이미지에서 각 점을 개별적으로 검출하기 위한 전처리 파이프라인을 설계하세요. (힌트: 침식으로 점들을 분리)

### 연습 5: 세포 분리 (Watershed 전처리)

현미경 세포 이미지에서 붙어있는 세포들을 분리하기 위한 전처리를 구현하세요:
1. 이진화
2. 노이즈 제거 (열기/닫기)
3. 확실한 배경 영역 찾기 (팽창)
4. 확실한 전경 영역 찾기 (거리 변환 + 임계값)

---

## 9. 다음 단계

[이진화 및 임계처리](./07_Thresholding.md)에서 다양한 이진화 방법과 임계처리 기법을 학습합니다!

**다음에 배울 내용**:
- 전역 임계처리 (`cv2.threshold`)
- OTSU 자동 임계값
- 적응형 임계처리
- HSV 기반 임계처리

---

## 10. 참고 자료

### 공식 문서

- [erode() 문서](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511f2fb1c)
- [dilate() 문서](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c)
- [morphologyEx() 문서](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f)
- [getStructuringElement() 문서](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac342a1bb6eabf6f55c803b09268e36dc)

### 관련 학습 자료

| 폴더 | 관련 내용 |
|------|----------|
| [이미지 필터링](./05_Image_Filtering.md) | 필터링 기초 |
| [윤곽선 검출 (Contour Detection)](./09_Contours.md) | 전처리 후 윤곽선 검출 |

### 추가 참고

- [모폴로지 연산 튜토리얼](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
- [수학적 모폴로지 이론](https://en.wikipedia.org/wiki/Mathematical_morphology)

