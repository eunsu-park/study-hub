# 02. 2D 변환

[← 이전: 01. 그래픽스 파이프라인 개요](01_Graphics_Pipeline_Overview.md) | [다음: 03. 3D 변환과 투영 →](03_3D_Transformations_and_Projections.md)

---

## 학습 목표

1. 2D 이동(translation), 회전(rotation), 스케일링(scaling)을 행렬로 표현
2. 2D 변환에 3x3 행렬이 필요한 이유로서 동차 좌표계(homogeneous coordinates) 이해
3. 행렬 곱셈으로 여러 변환을 합성하고 순서가 중요함을 인식
4. 반사(reflection)와 전단(shear) 변환 적용
5. 아핀 변환(affine transformation)과 사영 변환(projective transformation) 구분
6. NumPy를 사용하여 모든 2D 변환을 처음부터 구현
7. 각 변환이 도형에 미치는 기하학적 효과 시각화
8. "명령 카드(instruction card)" 비유를 통해 변환 합성에 대한 직관 구축

---

## 왜 이것이 중요한가

변환은 컴퓨터 그래픽스에서 움직임과 변화의 수학적 언어다. 데스크탑에서 아이콘을 드래그하거나, 사진 편집기에서 이미지를 회전하거나, 2D 게임 캐릭터가 화면을 가로질러 달리는 것을 볼 때마다 2D 변환이 작동하고 있다. 더 복잡한 3D 경우를 다루기 전에 먼저 2D 변환을 마스터하는 것은 견고한 토대를 만든다 — 같은 원리(행렬 표현, 합성, 좌표계)가 3D에도 직접 적용되기 때문이다.

각 변환을 **모든 점에게 어디로 이동할지 알려주는 명령 카드(instruction card)**로 생각하라. 회전 카드는 "모든 점이여, 원점 주위로 45도 회전하라"고 말한다. 스케일링 카드는 "모든 점이여, 원점에서 두 배 멀리 이동하라"고 말한다. 명령 카드를 쌓으면 변환을 합성하게 되며 — 카드를 쌓는 순서가 결과에 심대한 영향을 미친다.

---

## 1. 2D에서의 점과 벡터

2D에서 **점(point)**은 위치다: $\mathbf{p} = (x, y)$.
2D에서 **벡터(vector)**는 변위(displacement)다: $\mathbf{v} = (v_x, v_y)$.

그래픽스에서는 두 가지 모두 열 벡터로 취급하여 행렬 변환을 적용한다:

$$\mathbf{p} = \begin{bmatrix} x \\ y \end{bmatrix}$$

**변환(transformation)** $T$는 점을 새로운 위치로 매핑한다:

$$T: \mathbb{R}^2 \rightarrow \mathbb{R}^2, \quad \mathbf{p}' = T(\mathbf{p})$$

$T$가 행렬 곱셈으로 표현될 수 있다면, **선형 변환(linear transformation)**이라고 한다:

$$\mathbf{p}' = \mathbf{M} \cdot \mathbf{p}$$

---

## 2. 기본 2D 변환

### 2.1 스케일링(Scaling)

스케일링은 객체의 크기를 변경한다. **비균일 스케일(non-uniform scale)**은 각 축마다 다른 계수를 허용한다:

$$\mathbf{S}(s_x, s_y) = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$$

스케일 적용:

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} s_x \cdot x \\ s_y \cdot y \end{bmatrix}$$

- **균일 스케일(Uniform scaling)**: $s_x = s_y$ (형태 유지, 크기 변경)
- **비균일 스케일(Non-uniform scaling)**: $s_x \neq s_y$ (축을 따라 늘리기/압축)
- $s = 1$: 변화 없음; $s > 1$: 확대; $0 < s < 1$: 축소; $s < 0$: 반사 + 스케일

### 2.2 회전(Rotation)

각도 $\theta$만큼의 회전 (원점 기준 반시계 방향):

$$\mathbf{R}(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

**유도**: $(r, 0)$에 있는 점을 $\theta$만큼 회전하면 $(r\cos\theta, r\sin\theta)$로 이동한다. $(0, r)$에 있는 점을 $\theta$만큼 회전하면 $(-r\sin\theta, r\cos\theta)$로 이동한다. 이것들이 회전 행렬의 열을 형성한다.

회전 행렬의 성질:
- **직교 행렬(Orthogonal)**: $\mathbf{R}^T = \mathbf{R}^{-1}$ (역행렬은 단순히 전치 행렬)
- **행렬식(Determinant)** = 1 (넓이와 방향 보존)
- **회전 합성(Rotation composition)**: $\mathbf{R}(\alpha) \cdot \mathbf{R}(\beta) = \mathbf{R}(\alpha + \beta)$

### 2.3 이동(Translation)

이동은 모든 점을 고정된 오프셋만큼 이동시킨다:

$$\mathbf{p}' = \mathbf{p} + \mathbf{t} = \begin{bmatrix} x + t_x \\ y + t_y \end{bmatrix}$$

**문제점**: 이동은 *선형 변환이 아니다* — 2x2 행렬 곱셈으로 표현될 수 없다. 선형 변환은 항상 원점을 원점에 매핑하기 때문이다: $\mathbf{M} \cdot \mathbf{0} = \mathbf{0}$. 그러나 이동은 원점을 이동시킨다.

이것이 **동차 좌표계(homogeneous coordinates)**를 도입하는 동기가 된다.

---

## 3. 동차 좌표계(Homogeneous Coordinates)

### 3.1 핵심 아이디어

이동을 포함한 모든 아핀 변환을 행렬 곱셈으로 통일하기 위해 세 번째 좌표를 추가한다. 2D 점 $(x, y)$는 다음이 된다:

$$\mathbf{p}_h = \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

이제 이동을 3x3 행렬로 표현할 수 있다:

$$\mathbf{T}(t_x, t_y) = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}$$

검증:

$$\begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} x + t_x \\ y + t_y \\ 1 \end{bmatrix}$$

### 3.2 왜 2D에 3x3인가?

핵심 통찰: **$n$차원 공간에서의 이동은 $(n+1)$차원 동차 공간에서의 선형 변환이다**. 2D 점을 3D 동차 공간에 임베딩함으로써, 모든 아핀 변환을 행렬 곱셈으로 표현하는 능력을 얻게 된다.

동차 좌표계에서 2D 아핀 변환의 일반 형식:

$$\mathbf{M} = \begin{bmatrix} a & b & t_x \\ c & d & t_y \\ 0 & 0 & 1 \end{bmatrix}$$

여기서:
- $\begin{bmatrix} a & b \\ c & d \end{bmatrix}$는 회전, 스케일링, 전단을 인코딩
- $\begin{bmatrix} t_x \\ t_y \end{bmatrix}$는 이동을 인코딩
- 하단 행 $[0, 0, 1]$은 $w$ 좌표가 1로 유지되도록 보장

### 3.3 동차 좌표계에서의 점과 벡터

| 엔티티 | 동차 형식 | 이유 |
|--------|-----------------|------|
| 점(Point) | $(x, y, 1)$ | 점은 이동되어야 함 |
| 벡터(Vector) | $(v_x, v_y, 0)$ | 벡터는 이동되어선 안 됨 |

이동 행렬에 벡터 $(v_x, v_y, 0)^T$를 곱하면 변화 없이 그대로 남는다 ($t_x, t_y$ 항이 세 번째 성분의 0과 곱해진다). 이것은 벡터가 위치와 무관한 방향을 나타낸다는 것을 올바르게 모델링한다.

### 3.4 동차 형식의 모든 변환

**이동(Translation)**:
$$\mathbf{T}(t_x, t_y) = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}$$

**스케일링(Scaling)** (원점 기준):
$$\mathbf{S}(s_x, s_y) = \begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**회전(Rotation)** (원점 기준):
$$\mathbf{R}(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

---

## 4. 변환의 합성(Composition of Transformations)

### 4.1 행렬 곱셈 = 변환 합성

행렬 표현의 아름다움: 변환을 합성하는 것은 단순히 행렬 곱셈이다. 먼저 회전하고, 그 다음 이동하려면:

$$\mathbf{p}' = \mathbf{T} \cdot \mathbf{R} \cdot \mathbf{p}$$

> **읽는 순서**: 변환은 **오른쪽에서 왼쪽으로** 적용된다. $\mathbf{T} \cdot \mathbf{R} \cdot \mathbf{p}$에서 회전 $\mathbf{R}$이 먼저 적용되고, 그 다음 이동 $\mathbf{T}$가 적용된다.

### 4.2 순서가 중요하다!

변환 합성은 **교환 법칙이 성립하지 않는다**: 일반적으로 $\mathbf{T} \cdot \mathbf{R} \neq \mathbf{R} \cdot \mathbf{T}$.

**예시**: 90도 회전하고 $(5, 0)$만큼 이동하는 경우를 생각해보자.

**회전 후 이동** ($\mathbf{T} \cdot \mathbf{R}$):
1. 원점 주위로 점을 회전
2. 그 다음 모든 것을 오른쪽으로 5만큼 이동

**이동 후 회전** ($\mathbf{R} \cdot \mathbf{T}$):
1. 점을 오른쪽으로 5만큼 이동
2. 그 다음 원점 주위로 회전 (이동된 점이 원을 그리며 움직임)

이 두 경우는 매우 다른 결과를 만든다! 두 번째 경우는 이동된 점을 원점 주위로 공전시킨다.

```python
import numpy as np

def make_translation(tx, ty):
    """Create a 2D translation matrix in homogeneous coordinates."""
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0,  1]
    ], dtype=float)

def make_rotation(theta_deg):
    """Create a 2D rotation matrix (counterclockwise) in homogeneous coordinates."""
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=float)

# Demonstrate that order matters
T = make_translation(5, 0)
R = make_rotation(90)
point = np.array([1, 0, 1])  # Point at (1, 0)

# Rotate first, then translate
result_RT = T @ R @ point
print(f"Rotate then Translate: ({result_RT[0]:.2f}, {result_RT[1]:.2f})")
# Output: Rotate then Translate: (5.00, 1.00)

# Translate first, then rotate
result_TR = R @ T @ point
print(f"Translate then Rotate: ({result_TR[0]:.2f}, {result_TR[1]:.2f})")
# Output: Translate then Rotate: (0.00, 6.00)
```

### 4.3 임의의 점을 중심으로 한 회전

원점 대신 점 $\mathbf{c} = (c_x, c_y)$ 주위로 회전하려면:

1. $\mathbf{c}$가 원점에 오도록 이동: $\mathbf{T}(-c_x, -c_y)$
2. 회전: $\mathbf{R}(\theta)$
3. 다시 이동: $\mathbf{T}(c_x, c_y)$

$$\mathbf{M} = \mathbf{T}(c_x, c_y) \cdot \mathbf{R}(\theta) \cdot \mathbf{T}(-c_x, -c_y)$$

이 패턴 — "원점으로 이동, 변환 적용, 다시 이동" — 은 그래픽스에서 자주 등장한다.

```python
def make_rotation_about_point(theta_deg, cx, cy):
    """
    Rotate around an arbitrary center point.

    Why this pattern? Rotation matrices rotate about the origin.
    To rotate about (cx, cy), we temporarily move (cx, cy) to the origin,
    perform the rotation, and move back.
    """
    T_to_origin = make_translation(-cx, -cy)
    R = make_rotation(theta_deg)
    T_back = make_translation(cx, cy)

    # Compose: apply right to left (to_origin first, rotate, then back)
    return T_back @ R @ T_to_origin
```

### 4.4 임의의 점을 중심으로 한 스케일링

마찬가지로, 점 $\mathbf{c}$를 중심으로 스케일링하려면:

$$\mathbf{M} = \mathbf{T}(c_x, c_y) \cdot \mathbf{S}(s_x, s_y) \cdot \mathbf{T}(-c_x, -c_y)$$

---

## 5. 반사(Reflection)

반사는 축을 기준으로 지오메트리를 대칭 이동한다.

### 5.1 축에 대한 반사

**x축에 대한 반사** (y 뒤집기):
$$\mathbf{M}_x = \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**y축에 대한 반사** (x 뒤집기):
$$\mathbf{M}_y = \begin{bmatrix} -1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**원점에 대한 반사** (둘 다 뒤집기):
$$\mathbf{M}_o = \begin{bmatrix} -1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

### 5.2 임의의 직선에 대한 반사

원점을 통과하는 각도 $\alpha$의 직선에 대한 반사:

$$\mathbf{M}_\alpha = \begin{bmatrix} \cos 2\alpha & \sin 2\alpha & 0 \\ \sin 2\alpha & -\cos 2\alpha & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

원점을 통과하지 않는 직선에 대해서는: 이동, 반사, 다시 이동 (임의의 점을 중심으로 한 회전과 같은 패턴).

---

## 6. 전단(Shear)

전단은 다른 축에 비례하여 한 축을 따라 객체를 "기울인다".

### 6.1 X축 전단

$$\mathbf{Sh}_x(k) = \begin{bmatrix} 1 & k & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

이것은 각 점의 $x$ 좌표를 $k \cdot y$만큼 이동시킨다:

$$x' = x + k \cdot y, \quad y' = y$$

카드 덱을 상상해보라: 맨 아래 카드는 그대로 있고, 각 위쪽 카드는 조금씩 오른쪽으로 밀린다.

### 6.2 Y축 전단

$$\mathbf{Sh}_y(k) = \begin{bmatrix} 1 & 0 & 0 \\ k & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

이것은 각 점의 $y$ 좌표를 $k \cdot x$만큼 이동시킨다.

### 6.3 분해 통찰

모든 2D 아핀 변환은 다음의 연속으로 분해할 수 있다:
- 회전
- (비균일) 스케일링
- 이동

대안적으로, 전단은 때로 회전보다 하드웨어에서 구현하기 더 단순하기 때문에 계산적 구성 요소로 사용된다.

---

## 7. 아핀 변환(Affine) vs 사영 변환(Projective Transformation)

### 7.1 아핀 변환(Affine Transformation)

**아핀 변환(affine transformation)**은 다음을 보존한다:
- **공선성(Collinearity)**: 직선 위의 점들은 직선 위에 남는다
- **거리 비율(Ratios of distances)**: 선분의 중점은 중점으로 남는다
- **평행성(Parallelism)**: 평행선은 평행하게 유지된다

지금까지 본 모든 변환(이동, 회전, 스케일링, 반사, 전단)은 아핀이다. 동차 좌표계에서 아핀 변환은 다음 형식을 가진다:

$$\mathbf{M}_{\text{affine}} = \begin{bmatrix} a & b & t_x \\ c & d & t_y \\ 0 & 0 & 1 \end{bmatrix}$$

하단 행은 항상 $[0, 0, 1]$이다.

### 7.2 사영 변환(Projective Transformation)

**사영 변환(projective transformation)**(**단응사상(homography)** 또는 **원근 변환(perspective transformation)**이라고도 함)은 하단 행이 단순하지 않을 수 있다:

$$\mathbf{M}_{\text{proj}} = \begin{bmatrix} a & b & c \\ d & e & f \\ g & h & 1 \end{bmatrix}$$

곱셈 후, $w$ 성분으로 나누어 2D로 돌아와야 한다:

$$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = \mathbf{M}_{\text{proj}} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}, \quad \text{결과} = \left(\frac{x'}{w'}, \frac{y'}{w'}\right)$$

사영 변환은:
- 평행성을 **보존하지 않는다** (평행선이 수렴할 수 있다)
- 공선성을 **보존한다** (직선은 직선으로 유지된다)
- 3D 그래픽스의 원근 투영에 사용된다 (레슨 03)
- 이미지 스티칭(image stitching)과 증강 현실(augmented reality)에 사용된다

### 7.3 변환 계층 구조

```
Rigid (Euclidean)  ⊂  Similarity  ⊂  Affine  ⊂  Projective
     │                    │              │            │
  Rotation +          Rigid +        Similarity +   Affine +
  Translation        Uniform Scale   Non-uniform    Perspective
                                     Scale + Shear
     │                    │              │            │
  Preserves:          Preserves:     Preserves:    Preserves:
  - Distances         - Angles       - Parallelism - Collinearity
  - Angles            - Ratios       - Ratios      (only)
  - Ratios
```

---

## 8. 구현: 완전한 2D 변환 라이브러리

```python
"""
Complete 2D transformation library using NumPy.

All transformations operate in homogeneous coordinates (3x3 matrices).
Points are represented as (x, y, 1) column vectors.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


# ═══════════════════════════════════════════════════════════════
# Core Transformation Matrices
# ═══════════════════════════════════════════════════════════════

def translate(tx, ty):
    """
    Create a 2D translation matrix.

    Why homogeneous coords? Without them, translation requires addition
    (p' = p + t), making it incompatible with other transforms that use
    multiplication. Homogeneous coords unify everything as multiplication.
    """
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0,  1]
    ], dtype=float)


def rotate(theta_deg):
    """
    Create a 2D rotation matrix (counterclockwise about origin).

    The rotation matrix is orthogonal: R^T = R^{-1}.
    This means the inverse rotation is simply the transpose,
    which is computationally cheaper than a general matrix inverse.
    """
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=float)


def scale(sx, sy=None):
    """
    Create a 2D scaling matrix.

    If sy is not given, uniform scaling is applied (sx = sy).
    Negative scale values produce reflection + scaling.
    """
    if sy is None:
        sy = sx  # Uniform scaling
    return np.array([
        [sx,  0, 0],
        [ 0, sy, 0],
        [ 0,  0, 1]
    ], dtype=float)


def reflect_x():
    """Reflect across the x-axis (y coordinates are negated)."""
    return scale(1, -1)


def reflect_y():
    """Reflect across the y-axis (x coordinates are negated)."""
    return scale(-1, 1)


def reflect_line(angle_deg):
    """
    Reflect across a line through the origin at the given angle.

    Derivation: rotate so the line aligns with x-axis, reflect across
    x-axis, rotate back. This simplifies to the formula below.
    """
    a = np.radians(angle_deg)
    c2, s2 = np.cos(2 * a), np.sin(2 * a)
    return np.array([
        [ c2, s2, 0],
        [ s2, -c2, 0],
        [  0,   0, 1]
    ], dtype=float)


def shear_x(k):
    """
    Shear along x-axis: x' = x + k*y, y' = y.

    Visualize a deck of cards: each card slides horizontally
    proportional to its height.
    """
    return np.array([
        [1, k, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=float)


def shear_y(k):
    """Shear along y-axis: x' = x, y' = y + k*x."""
    return np.array([
        [1, 0, 0],
        [k, 1, 0],
        [0, 0, 1]
    ], dtype=float)


# ═══════════════════════════════════════════════════════════════
# Compound Transformations
# ═══════════════════════════════════════════════════════════════

def rotate_about(theta_deg, cx, cy):
    """
    Rotate about an arbitrary center point (cx, cy).

    Pattern: translate center to origin -> rotate -> translate back.
    This "sandwich" pattern is fundamental in graphics.
    """
    return translate(cx, cy) @ rotate(theta_deg) @ translate(-cx, -cy)


def scale_about(sx, sy, cx, cy):
    """Scale about an arbitrary center point (cx, cy)."""
    return translate(cx, cy) @ scale(sx, sy) @ translate(-cx, -cy)


# ═══════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════

def transform_points(matrix, points):
    """
    Apply a 3x3 transformation matrix to an array of 2D points.

    Parameters:
        matrix: 3x3 transformation matrix
        points: Nx2 array of (x, y) coordinates

    Returns:
        Nx2 array of transformed (x', y') coordinates

    Why we add w=1: each point (x,y) becomes (x,y,1) in homogeneous
    coordinates. After transformation, we extract (x', y') and discard w'.
    For affine transforms w' is always 1; for projective transforms,
    we would need to divide by w'.
    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]

    # Convert to homogeneous: (x, y) -> (x, y, 1)
    ones = np.ones((n, 1))
    homogeneous = np.hstack([points, ones])  # Nx3

    # Apply transformation: each row is a point, so we transpose
    # M @ p for column vectors = (p^T @ M^T)^T for row vectors
    transformed = (matrix @ homogeneous.T).T  # Nx3

    # Convert back from homogeneous: divide by w (handles projective case)
    w = transformed[:, 2:3]
    return transformed[:, :2] / w


def compose(*transforms):
    """
    Compose multiple transformations (applied right to left).

    compose(A, B, C) produces A @ B @ C, meaning C is applied first.
    This matches the mathematical convention p' = A * B * C * p.
    """
    result = np.eye(3)
    for t in transforms:
        result = result @ t
    return result


# ═══════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════

def plot_shape(ax, points, color='blue', alpha=0.3, label=None):
    """Plot a 2D polygon defined by its vertices."""
    polygon = Polygon(points, closed=True)
    p = PatchCollection([polygon], alpha=alpha, facecolors=[color],
                        edgecolors=[color], linewidths=2)
    ax.add_collection(p)
    if label:
        centroid = points.mean(axis=0)
        ax.annotate(label, centroid, fontsize=10, ha='center',
                    fontweight='bold', color=color)


def demo_transformations():
    """Demonstrate key 2D transformations visually."""
    # Define a simple house shape
    house = np.array([
        [0, 0], [2, 0], [2, 2], [1, 3], [0, 2]  # Square base + triangle roof
    ])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('2D Transformations', fontsize=16, fontweight='bold')

    transforms = [
        ("Translation (3, 1)", translate(3, 1)),
        ("Rotation (45 deg)", rotate(45)),
        ("Scale (1.5, 0.8)", scale(1.5, 0.8)),
        ("Reflect (y-axis)", reflect_y()),
        ("Shear X (k=0.5)", shear_x(0.5)),
        ("Rotate about (1,1) 45 deg", rotate_about(45, 1, 1)),
    ]

    for ax, (title, matrix) in zip(axes.flat, transforms):
        ax.set_xlim(-4, 6)
        ax.set_ylim(-4, 6)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_title(title)

        # Draw original shape
        plot_shape(ax, house, color='blue', alpha=0.2, label='Original')

        # Draw transformed shape
        transformed = transform_points(matrix, house)
        plot_shape(ax, transformed, color='red', alpha=0.4, label='Transformed')

    plt.tight_layout()
    plt.savefig('2d_transforms_demo.png', dpi=150, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════════
# Demonstration
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # --- Basic transforms ---
    print("=== 2D Transformation Examples ===\n")

    point = np.array([[3, 2]])  # A single point at (3, 2)

    # Translation
    T = translate(5, -1)
    print(f"Original: {point[0]}")
    print(f"After translate(5, -1): {transform_points(T, point)[0]}")
    # Expected: (8, 1)

    # Rotation by 90 degrees
    R = rotate(90)
    print(f"After rotate(90): {transform_points(R, point)[0]}")
    # Expected: (-2, 3) -- 90 deg CCW

    # Composition: rotate then translate
    M = compose(translate(5, -1), rotate(90))
    print(f"After rotate(90) then translate(5,-1): {transform_points(M, point)[0]}")
    # Expected: (3, 2) -- rotate to (-2,3), then translate to (3, 2)

    # --- Order matters ---
    print("\n=== Order Matters ===")
    p = np.array([[1, 0]])
    M1 = compose(translate(5, 0), rotate(90))  # Rotate first, then translate
    M2 = compose(rotate(90), translate(5, 0))  # Translate first, then rotate

    r1 = transform_points(M1, p)[0]
    r2 = transform_points(M2, p)[0]
    print(f"Rotate then Translate: ({r1[0]:.2f}, {r1[1]:.2f})")
    print(f"Translate then Rotate: ({r2[0]:.2f}, {r2[1]:.2f})")
    print(f"Same result? {np.allclose(r1, r2)}")  # False!

    # --- Transform a triangle ---
    print("\n=== Transforming a Triangle ===")
    triangle = np.array([[0, 0], [1, 0], [0.5, 1]])
    M = compose(
        translate(2, 3),       # 3. Move to final position
        rotate(45),            # 2. Rotate 45 degrees
        scale(2, 2)            # 1. Double the size
    )
    transformed = transform_points(M, triangle)
    print(f"Original triangle:\n{triangle}")
    print(f"After scale(2) -> rotate(45) -> translate(2,3):\n"
          f"{np.round(transformed, 3)}")

    # --- Visualization ---
    print("\n=== Generating visualization... ===")
    demo_transformations()
```

---

## 9. 행렬 성질과 역변환(Inverses)

역변환을 이해하는 것은 연산을 "되돌리는" 데 필수적이다.

### 9.1 일반적인 변환의 역행렬

| 변환 | 정방향 행렬 | 역행렬 |
|---------------|---------------|---------|
| 이동 $\mathbf{T}(t_x, t_y)$ | $\begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}$ | $\mathbf{T}(-t_x, -t_y)$ |
| 회전 $\mathbf{R}(\theta)$ | $\theta$만큼 회전 | $\mathbf{R}(-\theta) = \mathbf{R}(\theta)^T$ |
| 스케일 $\mathbf{S}(s_x, s_y)$ | $(s_x, s_y)$만큼 스케일 | $\mathbf{S}(1/s_x, 1/s_y)$ |
| 합성 $\mathbf{A} \cdot \mathbf{B}$ | B 적용 후 A | $\mathbf{B}^{-1} \cdot \mathbf{A}^{-1}$ |

### 9.2 합성의 역변환

합성된 변환의 역변환은 순서를 반대로 한다:

$$(\mathbf{A} \cdot \mathbf{B} \cdot \mathbf{C})^{-1} = \mathbf{C}^{-1} \cdot \mathbf{B}^{-1} \cdot \mathbf{A}^{-1}$$

옷을 입고 벗는 것과 같다: 양말을 신고, 그 다음 신발을 신는다 (합성). 되돌리려면 신발을 먼저 벗고, 그 다음 양말을 벗는다 (역순).

### 9.3 행렬식(Determinant)과 넓이 변화

2D 변환 행렬의 **행렬식(determinant)** (왼쪽 위 2x2 블록)은 넓이가 어떻게 변하는지 알려준다:

$$\text{넓이 비율} = |\det(\mathbf{M}_{2 \times 2})|$$

- $|\det| = 1$: 넓이 보존 (회전, 반사)
- $|\det| > 1$: 넓이 증가 (확대)
- $|\det| < 1$: 넓이 감소 (축소)
- $\det < 0$: 방향이 반전됨 (반사)

```python
def analyze_transform(matrix):
    """Analyze properties of a 2D transformation matrix."""
    # Extract the linear part (upper-left 2x2)
    linear = matrix[:2, :2]
    det = np.linalg.det(linear)

    # Extract translation
    tx, ty = matrix[0, 2], matrix[1, 2]

    print(f"Matrix:\n{matrix}")
    print(f"Translation: ({tx:.3f}, {ty:.3f})")
    print(f"Determinant: {det:.3f}")
    print(f"Area scale factor: {abs(det):.3f}")
    print(f"Preserves orientation: {det > 0}")
    print(f"Is orthogonal: {np.allclose(linear @ linear.T, np.eye(2))}")
    print()

# Example analyses
print("Pure rotation (45 deg):")
analyze_transform(rotate(45))

print("Non-uniform scale (2, 0.5):")
analyze_transform(scale(2, 0.5))

print("Reflection across y-axis:")
analyze_transform(reflect_y())
```

---

## 10. 흔한 함정(Pitfalls)

### 10.1 회전 방향 규약

시스템마다 다른 규약을 사용한다:
- **수학 규약**: 반시계 방향이 양수 (우리가 사용하는 것)
- **화면 규약**: y축이 아래를 향하는 경우가 많아, 수학에서의 "반시계"가 화면에서는 시계 방향으로 나타남

항상 회전 방향이 좌표계와 일치하는지 확인하라!

### 10.2 원점 기준의 변환

스케일링과 회전 행렬은 기본적으로 **원점 기준**으로 동작한다. 객체가 원점에 중심이 없으면 결과가 예상치 못할 수 있다:

```python
# A square centered at (5, 5), not the origin
square = np.array([[4, 4], [6, 4], [6, 6], [4, 6]])

# Scaling by 2 about the origin moves the square away!
S_origin = scale(2)
wrong = transform_points(S_origin, square)
print(f"Scale about origin: {wrong}")  # Points at (8,8) to (12,12)

# Correct: scale about the square's center (5, 5)
S_center = scale_about(2, 2, 5, 5)
correct = transform_points(S_center, square)
print(f"Scale about center: {correct}")  # Points at (3,3) to (7,7)
```

### 10.3 부동소수점 누적 오차

회전 행렬을 반복해서 합성하면 부동소수점 오차가 누적되어, 행렬이 적절한 회전 행렬(직교 행렬)에서 "벗어날" 수 있다. 장시간 실행되는 애니메이션에서는 주기적으로 재직교화(re-orthogonalize)하라:

```python
def orthogonalize(matrix):
    """
    Re-orthogonalize a rotation matrix that has accumulated floating-point errors.

    Uses the Gram-Schmidt process on the linear part.
    Without this, after thousands of small rotations, the matrix
    may introduce slight scaling or skew artifacts.
    """
    linear = matrix[:2, :2].copy()

    # Normalize first column
    col0 = linear[:, 0]
    col0 = col0 / np.linalg.norm(col0)

    # Make second column orthogonal to first, then normalize
    col1 = linear[:, 1]
    col1 = col1 - np.dot(col1, col0) * col0
    col1 = col1 / np.linalg.norm(col1)

    result = matrix.copy()
    result[:2, 0] = col0
    result[:2, 1] = col1
    return result
```

---

## 요약

| 변환 | 행렬 (동차 형식) | 보존하는 것 |
|---------------|---------------------|-----------|
| 이동 $(t_x, t_y)$ | $\begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}$ | 형태, 크기, 방향 |
| 회전 $\theta$ | $\begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$ | 형태, 크기 |
| 스케일 $(s_x, s_y)$ | $\begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1 \end{bmatrix}$ | 형태 (균일 스케일인 경우) |
| 전단 $k$ (x축) | $\begin{bmatrix} 1 & k & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$ | 넓이 |
| 반사 (y축) | $\begin{bmatrix} -1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$ | 형태, 크기 |

**핵심 요점**:
- 동차 좌표계 (2D를 위한 3x3 행렬)는 모든 아핀 변환을 행렬 곱셈으로 통일한다
- 변환 합성은 행렬 곱셈이며, **오른쪽에서 왼쪽으로** 적용된다
- **순서가 중요하다**: 일반적으로 $\mathbf{T} \cdot \mathbf{R} \neq \mathbf{R} \cdot \mathbf{T}$
- 원점이 아닌 점을 기준으로 변환하려면: 원점으로 이동, 변환, 다시 이동
- 아핀 변환은 평행성을 보존하지만, 사영 변환은 그렇지 않다

---

## 연습 문제

1. **행렬 구성**: 다음을 그 순서대로 적용하는 하나의 3x3 행렬을 작성하라: (a) 인수 2로 스케일, (b) 반시계 방향으로 30도 회전, (c) $(3, -1)$만큼 이동. 점 $(1, 0)$에 적용하여 검증하라.

2. **순서 조사**: 모서리가 $(0,0), (1,0), (1,1), (0,1)$인 정사각형에 대해 다음을 계산하라: (a) 45도 회전 후 2배 스케일, (b) 2배 스케일 후 45도 회전. 두 결과를 플로팅하고 시각적 차이를 설명하라.

3. **임의의 회전**: 점 $(3, 4)$를 중심으로 60도 회전하는 합성 행렬을 유도하라. 이 변환 하에서 점 $(3, 4)$가 자기 자신으로 매핑되는지 검증하라.

4. **역변환**: 객체가 $\mathbf{M} = \mathbf{T}(2, 3) \cdot \mathbf{R}(45) \cdot \mathbf{S}(2, 1)$로 변환되었다. 이 변환을 되돌리는 역행렬 $\mathbf{M}^{-1}$을 작성하라. $\mathbf{M}^{-1} \cdot \mathbf{M} = \mathbf{I}$임을 검증하라.

5. **전단 분해**: 각도 $\theta$의 회전이 세 개의 전단으로 분해될 수 있음을 보여라. 이를 구현하고 결과가 표준 회전 행렬과 일치하는지 검증하라.

6. **사영 변환**: 사영 변환이 단위 정사각형 $\{(0,0), (1,0), (1,1), (0,1)\}$을 사각형 $\{(0,0), (2,0), (1.5, 1), (0.5, 1)\}$에 매핑한다. 3x3 사영 행렬을 구하는 연립 방정식을 세워라 (8개의 미지수, $h_{33} = 1$로 정규화).

---

## 더 읽을거리

1. Marschner, S. & Shirley, P. *Fundamentals of Computer Graphics* (5th ed.), Ch. 6 -- "Transformation Matrices"
2. Hughes, J.F. et al. *Computer Graphics: Principles and Practice* (3rd ed.), Ch. 11 -- "2D Transformations"
3. [3Blue1Brown - Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) -- 행렬과 변환에 대한 아름다운 시각적 직관
4. Strang, G. *Introduction to Linear Algebra* -- 선형 변환에 관한 챕터들
5. [Immersive Linear Algebra](http://immersivemath.com/ila/) -- 2D/3D 시각화를 포함한 인터랙티브 온라인 교재
