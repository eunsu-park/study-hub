# 03. 3D 변환과 투영

[&larr; 이전: 02. 2D 변환](02_2D_Transformations.md) | [다음: 04. 래스터화 →](04_Rasterization.md)

---

## 학습 목표

1. 2D 변환 개념을 3D 동차 좌표계용 4x4 행렬로 확장한다
2. 오브젝트를 월드 공간에 배치하는 모델 행렬(Model matrix)을 구성한다
3. 카메라의 시점을 나타내는 뷰 행렬(View matrix, lookAt)을 유도한다
4. 원근 투영(Perspective projection)과 직교 투영(Orthographic projection) 행렬을 이해하고 유도한다
5. 오브젝트 공간(Object space)에서 스크린 공간(Screen space)까지 이어지는 완전한 MVP(Model-View-Projection) 파이프라인을 설명한다
6. NDC를 픽셀 좌표로 매핑하는 뷰포트 변환(Viewport transform)을 설명한다
7. 오일러 각도(Euler angles)의 문제점(짐벌 락(Gimbal lock))을 인식하고, 대안으로서 쿼터니언(Quaternion)을 이해한다
8. Python으로 완전한 MVP 행렬 체인을 구현한다

---

## 왜 이것이 중요한가

레슨 02에서 우리는 3x3 행렬을 이용한 2D 변환을 마스터했다. 이제 이 개념을 3D로 확장한다. 3D에서는 해결해야 할 과제가 더 많다. 오브젝트를 3D 월드에 배치하고, 카메라를 시뮬레이션하며, 3D 장면을 2D 화면에 투영해야 한다. **MVP(Model-View-Projection)** 행렬 체인은 실시간 3D 그래픽스에서 가장 중요한 개념이라 해도 과언이 아니다. 모든 3D 애플리케이션의 모든 정점(vertex)은 예외 없이 이 파이프라인을 통과한다. 3D 점이 2D 픽셀이 되는 과정을 이해하는 것은 카메라 조작부터 그림자 매핑(Shadow mapping), VR 렌더링까지 모든 것을 이해하는 핵심이다.

---

## 1. 3D 동차 좌표계(Homogeneous Coordinates)

2D에서 3x3 행렬을 사용했던 것처럼 ($w$ 성분 추가), 3D에서는 **4x4 행렬**을 사용한다.

$$\mathbf{p} = \begin{bmatrix} x \\ y \\ z \end{bmatrix} \quad \rightarrow \quad \mathbf{p}_h = \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$$

일반적인 3D 아핀 변환(Affine transformation):

$$\mathbf{M} = \begin{bmatrix} & & & t_x \\ & \mathbf{R}_{3\times3} & & t_y \\ & & & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

여기서 $\mathbf{R}_{3\times3}$은 회전/스케일/전단(Shear)을 담고, $(t_x, t_y, t_z)$는 이동(Translation)이다.

---

## 2. 기본 3D 변환

### 2.1 이동(Translation)

$$\mathbf{T}(t_x, t_y, t_z) = \begin{bmatrix} 1 & 0 & 0 & t_x \\ 0 & 1 & 0 & t_y \\ 0 & 0 & 1 & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

### 2.2 스케일(Scaling)

$$\mathbf{S}(s_x, s_y, s_z) = \begin{bmatrix} s_x & 0 & 0 & 0 \\ 0 & s_y & 0 & 0 \\ 0 & 0 & s_z & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

### 2.3 회전(Rotation)

2D는 회전축이 하나지만, 3D에는 세 가지 주축 회전이 있다.

**x축 기준 $\theta$ 회전**:

$$\mathbf{R}_x(\theta) = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta & 0 \\ 0 & \sin\theta & \cos\theta & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

**y축 기준 $\theta$ 회전**:

$$\mathbf{R}_y(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta & 0 \\ 0 & 1 & 0 & 0 \\ -\sin\theta & 0 & \cos\theta & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

**z축 기준 $\theta$ 회전**:

$$\mathbf{R}_z(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 & 0 \\ \sin\theta & \cos\theta & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

> **참고**: $\mathbf{R}_y$의 부호 패턴이 $\mathbf{R}_x$, $\mathbf{R}_z$와 "뒤바뀐" 것처럼 보이는 이유는, $y$축 회전이 순환 순서 $y \rightarrow z \rightarrow x$에 따른 오른손 법칙(Right-hand rule)을 따르기 때문이다.

### 2.4 임의 축 기준 회전

단위 벡터 $\hat{\mathbf{u}} = (u_x, u_y, u_z)$를 기준으로 각도 $\theta$만큼 회전할 때, **로드리게스 회전 공식(Rodrigues' rotation formula)**은 다음과 같다.

$$\mathbf{R}(\hat{\mathbf{u}}, \theta) = \cos\theta \cdot \mathbf{I} + (1 - \cos\theta)(\hat{\mathbf{u}} \otimes \hat{\mathbf{u}}) + \sin\theta \cdot [\hat{\mathbf{u}}]_\times$$

여기서 $[\hat{\mathbf{u}}]_\times$는 반대칭 외적 행렬(Skew-symmetric cross-product matrix):

$$[\hat{\mathbf{u}}]_\times = \begin{bmatrix} 0 & -u_z & u_y \\ u_z & 0 & -u_x \\ -u_y & u_x & 0 \end{bmatrix}$$

그리고 $\hat{\mathbf{u}} \otimes \hat{\mathbf{u}}$는 외적(Outer product):

$$\hat{\mathbf{u}} \otimes \hat{\mathbf{u}} = \begin{bmatrix} u_x^2 & u_x u_y & u_x u_z \\ u_x u_y & u_y^2 & u_y u_z \\ u_x u_z & u_y u_z & u_z^2 \end{bmatrix}$$

---

## 3. 모델 행렬(Model Matrix)

**모델 행렬** $\mathbf{M}_{\text{model}}$은 정점을 **오브젝트 공간(Object space)**(3D 모델에 지역적인 공간)에서 **월드 공간(World space)**(장면 공유 좌표계)으로 변환한다.

일반적으로 스케일, 회전, 이동의 합성으로 구성된다.

$$\mathbf{M}_{\text{model}} = \mathbf{T} \cdot \mathbf{R} \cdot \mathbf{S}$$

> **순서 규약**: 먼저 스케일(지역 공간에서 크기 변경), 그 다음 회전(오브젝트 방향 설정), 마지막으로 이동(월드 내 위치 배치). 오른쪽에서 왼쪽으로 적용: $\mathbf{p}_{\text{world}} = \mathbf{T} \cdot \mathbf{R} \cdot \mathbf{S} \cdot \mathbf{p}_{\text{object}}$.

```python
import numpy as np

def make_translation(tx, ty, tz):
    """Create a 4x4 translation matrix."""
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0,  1]
    ], dtype=float)

def make_scale(sx, sy, sz):
    """Create a 4x4 scaling matrix."""
    return np.array([
        [sx,  0,  0, 0],
        [ 0, sy,  0, 0],
        [ 0,  0, sz, 0],
        [ 0,  0,  0, 1]
    ], dtype=float)

def make_rotation_y(theta_deg):
    """Create a 4x4 rotation matrix about the y-axis."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1]
    ], dtype=float)

# Example: place a cube at position (5, 0, -3), rotated 45 deg around Y,
# scaled to half size
model_matrix = (make_translation(5, 0, -3)
                @ make_rotation_y(45)
                @ make_scale(0.5, 0.5, 0.5))
print("Model matrix:\n", model_matrix)
```

---

## 4. 뷰 행렬(View Matrix, 카메라)

**뷰 행렬** $\mathbf{M}_{\text{view}}$은 **월드 공간**을 **카메라(눈) 공간(Camera/Eye space)**으로 변환한다. 카메라 공간에서는 카메라가 원점에 위치하고 음의 z축 방향을 바라본다(OpenGL 규약).

### 4.1 LookAt 구성

주어진 값:
- $\mathbf{eye}$: 월드 공간에서의 카메라 위치
- $\mathbf{target}$: 카메라가 바라보는 점
- $\mathbf{up}$: 월드의 "위쪽" 방향 (보통 $(0, 1, 0)$)

카메라를 위한 정규직교 기저(Orthonormal basis)를 구성한다.

$$\mathbf{f} = \text{normalize}(\mathbf{target} - \mathbf{eye}) \quad \text{(앞 방향, Forward)}$$

$$\mathbf{r} = \text{normalize}(\mathbf{f} \times \mathbf{up}) \quad \text{(오른쪽 방향, Right)}$$

$$\mathbf{u} = \mathbf{r} \times \mathbf{f} \quad \text{(실제 위쪽 방향, True up)}$$

뷰 행렬은 회전(카메라 축을 월드 축에 정렬)과 이동(카메라를 원점으로 이동)을 결합한다.

$$\mathbf{M}_{\text{view}} = \begin{bmatrix} r_x & r_y & r_z & -\mathbf{r} \cdot \mathbf{eye} \\ u_x & u_y & u_z & -\mathbf{u} \cdot \mathbf{eye} \\ -f_x & -f_y & -f_z & \mathbf{f} \cdot \mathbf{eye} \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

> **왜 $\mathbf{f}$를 부정하는가?** OpenGL 규약에서 카메라는 눈 공간(Eye space)에서 $-z$ 방향을 바라본다. 따라서 앞 방향은 $-z$로 매핑되고, 세 번째 행에서 $\mathbf{f}$를 부정한다.

```python
def normalize(v):
    """Normalize a vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-10:
        return v  # Avoid division by zero
    return v / n

def look_at(eye, target, up):
    """
    Construct a view (camera) matrix.

    Why this works: the view matrix is the INVERSE of the camera's
    model matrix. Instead of computing a 4x4 inverse, we exploit the
    fact that rotation matrices are orthogonal (inverse = transpose)
    and combine with translation analytically.

    Parameters:
        eye: camera position (3D)
        target: point the camera looks at (3D)
        up: world up vector (3D)

    Returns:
        4x4 view matrix
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    # Camera basis vectors
    f = normalize(target - eye)     # Forward (into the screen)
    r = normalize(np.cross(f, up))  # Right
    u = np.cross(r, f)              # True up (may differ from input 'up')

    # Build view matrix: rotation part transposes the camera basis,
    # translation part dots with -eye to account for camera position
    view = np.array([
        [r[0],  r[1],  r[2],  -np.dot(r, eye)],
        [u[0],  u[1],  u[2],  -np.dot(u, eye)],
        [-f[0], -f[1], -f[2],  np.dot(f, eye)],
        [0,     0,     0,     1]
    ], dtype=float)

    return view

# Camera at (0, 2, 5), looking at origin, world up is +Y
view_matrix = look_at(
    eye=[0, 2, 5],
    target=[0, 0, 0],
    up=[0, 1, 0]
)
print("View matrix:\n", np.round(view_matrix, 4))
```

### 4.2 직관: "카메라를 이동"하는 것 vs "월드를 이동"하는 것

카메라를 오른쪽으로 이동시키는 것은 전체 월드를 왼쪽으로 이동시키는 것과 동일하다. 뷰 행렬은 후자를 수행한다. 카메라가 원점에 있는 것처럼 보이도록 모든 월드 공간 정점을 변환하여, 이후의 투영 계산을 단순화한다.

---

## 5. 투영 행렬(Projection Matrices)

투영은 3D 눈 공간(Eye-space) 좌표를 2D로 변환한다. 두 가지 주요 유형이 있다.

### 5.1 직교 투영(Orthographic Projection)

직교 투영은 직사각형 상자(뷰 볼륨)를 NDC 큐브 $[-1, 1]^3$으로 매핑한다. **원근 단축(Perspective foreshortening)**이 없으므로, 멀리 있는 오브젝트도 가까이 있는 것과 같은 크기로 보인다.

뷰 볼륨이 left $l$, right $r$, bottom $b$, top $t$, near $n$, far $f$로 정의될 때:

$$\mathbf{P}_{\text{ortho}} = \begin{bmatrix} \frac{2}{r-l} & 0 & 0 & -\frac{r+l}{r-l} \\ 0 & \frac{2}{t-b} & 0 & -\frac{t+b}{t-b} \\ 0 & 0 & \frac{-2}{f-n} & -\frac{f+n}{f-n} \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

이것은 단순히 이동(상자를 원점 중심으로 배치)에 이어 스케일(단위 크기로 만들기)을 수행하는 것이다.

```python
def orthographic(left, right, bottom, top, near, far):
    """
    Create an orthographic projection matrix (OpenGL convention).

    Why no perspective? In orthographic projection, parallel lines
    remain parallel. This is useful for CAD, 2D games, isometric views,
    and shadow maps (directional lights use ortho projection).
    """
    return np.array([
        [2/(right-left), 0, 0, -(right+left)/(right-left)],
        [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
        [0, 0, -2/(far-near),  -(far+near)/(far-near)],
        [0, 0, 0, 1]
    ], dtype=float)
```

### 5.2 원근 투영(Perspective Projection)

원근 투영은 인간의 눈과 카메라가 보는 방식을 시뮬레이션한다. **멀리 있는 오브젝트는 더 작게 보인다.** 절두체(Frustum, 잘린 피라미드 형태)를 NDC 큐브로 매핑한다.

```
       Near Plane          Far Plane
       ┌───────┐          ┌─────────────┐
       │       │         ╱│             │╲
       │  eye  │────────╱ │   Frustum   │ ╲
       │ (0,0) │────────╲ │   Volume    │ ╱
       │       │         ╲│             │╱
       └───────┘          └─────────────┘
       z = -n              z = -f
```

수직 시야각(Field of View) $\text{fov}$, 종횡비(Aspect ratio) $a = \frac{w}{h}$, 근평면 $n$, 원평면 $f$가 주어질 때:

$$t = n \cdot \tan\left(\frac{\text{fov}}{2}\right), \quad r = t \cdot a$$

원근 투영 행렬 (OpenGL 규약, $z \in [-1, 1]$로 매핑):

$$\mathbf{P}_{\text{persp}} = \begin{bmatrix} \frac{n}{r} & 0 & 0 & 0 \\ 0 & \frac{n}{t} & 0 & 0 \\ 0 & 0 & \frac{-(f+n)}{f-n} & \frac{-2fn}{f-n} \\ 0 & 0 & -1 & 0 \end{bmatrix}$$

### 5.3 원근 행렬 유도

핵심 통찰은, 원근 투영이 $x$와 $y$를 $-z$로 나눈다는 것이다 (멀리 있는 오브젝트는 $|z|$가 더 크므로 작아진다). 단계별로 유도해 보자.

**1단계**: 눈 공간에서 $(x, y, z)$에 위치한 점은 다음 스크린 좌표로 투영되어야 한다.

$$x_{\text{proj}} = \frac{n \cdot x}{-z}, \quad y_{\text{proj}} = \frac{n \cdot y}{-z}$$

(카메라가 $-z$ 방향을 바라보므로 $-z$를 사용한다. 근평면은 $z = -n$에 위치한다.)

**2단계**: 이것을 행렬 곱셈과 원근 나눗셈(Perspective division)으로 표현하려 한다. 트릭은 동차 좌표의 $w$ 성분을 사용하는 것이다. $w' = -z$로 설정하면:

$$\begin{bmatrix} x' \\ y' \\ z' \\ w' \end{bmatrix} = \mathbf{P} \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$$

원근 나눗셈 후: $\left(\frac{x'}{w'}, \frac{y'}{w'}, \frac{z'}{w'}\right)$.

**3단계**: $\frac{x'}{w'} = \frac{n \cdot x}{-z}$가 되어야 한다. $w' = -z$이면 $x' = n \cdot x$이다. 이로써 첫 번째 행을 얻는다: $[n, 0, 0, 0]$.

절두체 범위로 정규화(Normalization, $r$과 $t$로 나누기)하고 $z$-매핑을 $[-1, 1]$로 처리하면 전체 행렬이 완성된다.

**4단계**: $z$-매핑은 다음 조건을 만족해야 한다.
- $z = -n$이 $z_{\text{NDC}} = -1$로 매핑
- $z = -f$가 $z_{\text{NDC}} = +1$로 매핑

$1/z$의 선형 시스템을 풀면 세 번째 행을 얻는다: $[0, 0, -(f+n)/(f-n), -2fn/(f-n)]$.

```python
def perspective(fov_deg, aspect, near, far):
    """
    Create a perspective projection matrix (OpenGL convention).

    Why fov and aspect instead of l/r/b/t? This is the more intuitive
    parameterization: fov controls "zoom level" and aspect matches
    the screen's width/height ratio.

    Parameters:
        fov_deg: vertical field of view in degrees
        aspect: width / height ratio
        near: distance to near clipping plane (positive)
        far: distance to far clipping plane (positive)
    """
    fov = np.radians(fov_deg)
    t = near * np.tan(fov / 2)     # Half-height of near plane
    r = t * aspect                  # Half-width of near plane

    return np.array([
        [near/r, 0,      0,                     0],
        [0,      near/t, 0,                     0],
        [0,      0,      -(far+near)/(far-near), -2*far*near/(far-near)],
        [0,      0,      -1,                     0]
    ], dtype=float)
```

### 5.4 깊이 정밀도(Depth Precision)

원근 행렬은 $z$를 비선형적으로 매핑한다. 즉, 근평면 근처에 훨씬 더 많은 정밀도가 집중되고, 원평면 근처는 정밀도가 떨어진다. NDC 깊이(Depth)는 다음과 같다.

$$z_{\text{NDC}} = \frac{-(f+n)}{f-n} + \frac{2fn}{(f-n)(-z)}$$

이것은 $-z$의 쌍곡선 함수(Hyperbolic function)다. 그 결과:
- 근평면/원평면 비율이 1:1000이면 깊이 범위의 처음 10% 내에서 대부분의 깊이 정밀도가 낭비된다
- **실용적 조언**: 근평면을 가능한 한 멀리, 원평면을 가능한 한 가까이 유지하라
- **역 Z(Reversed-Z)**: 깊이 매핑을 뒤집어 부동소수점 정밀도를 더 균일하게 사용하는 현대적 기법

---

## 6. MVP 파이프라인

오브젝트 공간에서 클립 공간(Clip space)까지의 완전한 변환 체인:

$$\mathbf{p}_{\text{clip}} = \mathbf{P} \cdot \mathbf{V} \cdot \mathbf{M} \cdot \mathbf{p}_{\text{object}}$$

각 행렬의 역할:
- $\mathbf{M}$ = 모델 행렬 (오브젝트 &rarr; 월드)
- $\mathbf{V}$ = 뷰 행렬 (월드 &rarr; 눈/카메라)
- $\mathbf{P}$ = 투영 행렬 (눈 &rarr; 클립)

클리핑(Clipping) 후, **원근 나눗셈(Perspective division)**으로 클립 공간을 NDC로 변환한다.

$$\mathbf{p}_{\text{NDC}} = \left(\frac{x_c}{w_c}, \frac{y_c}{w_c}, \frac{z_c}{w_c}\right)$$

```
Object      Model       World      View       Eye       Projection    Clip
Space  ────────────▶  Space  ────────────▶  Space  ──────────────▶  Space
                                                                       │
                                                              Perspective
                                                              Division
                                                                       │
                                                                       ▼
Screen     Viewport      NDC
Space  ◀────────────   Space
(pixels)              [-1,1]^3
```

---

## 7. 뷰포트 변환(Viewport Transform)

**뷰포트 변환**은 NDC 좌표 $[-1, 1]^2$를 스크린 픽셀 좌표로 매핑한다.

$$x_{\text{screen}} = \frac{w}{2} \cdot x_{\text{NDC}} + \frac{w}{2} + x_0$$

$$y_{\text{screen}} = \frac{h}{2} \cdot y_{\text{NDC}} + \frac{h}{2} + y_0$$

여기서 $(x_0, y_0)$는 뷰포트 오프셋(보통 $(0, 0)$)이고, $(w, h)$는 픽셀 단위의 뷰포트 크기다.

행렬 형태:

$$\mathbf{M}_{\text{viewport}} = \begin{bmatrix} \frac{w}{2} & 0 & 0 & x_0 + \frac{w}{2} \\ 0 & \frac{h}{2} & 0 & y_0 + \frac{h}{2} \\ 0 & 0 & \frac{1}{2} & \frac{1}{2} \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

```python
def viewport(x0, y0, width, height):
    """
    Create the viewport transformation matrix.

    Maps NDC [-1,1]^2 to screen coordinates [x0, x0+width] x [y0, y0+height].
    Depth is mapped from [-1,1] to [0,1] for the depth buffer.
    """
    return np.array([
        [width/2,  0,        0,   x0 + width/2],
        [0,        height/2, 0,   y0 + height/2],
        [0,        0,        0.5, 0.5],
        [0,        0,        0,   1]
    ], dtype=float)
```

---

## 8. 오일러 각도(Euler Angles)와 짐벌 락(Gimbal Lock)

### 8.1 오일러 각도

**오일러 각도**는 3D 방향을 세 번의 순차적 회전으로 표현한다.
- **요(Yaw)** ($\psi$): y축 기준 회전 (좌우 시선 이동)
- **피치(Pitch)** ($\theta$): x축 기준 회전 (상하 시선 이동)
- **롤(Roll)** ($\phi$): z축 기준 회전 (머리 기울이기)

$$\mathbf{R} = \mathbf{R}_y(\psi) \cdot \mathbf{R}_x(\theta) \cdot \mathbf{R}_z(\phi)$$

(회전 순서는 규약에 따라 다르며, 위는 일반적인 선택 중 하나다.)

### 8.2 짐벌 락(Gimbal Lock)

**짐벌 락**은 두 회전 축이 정렬될 때 발생하며, 자유도(Degree of freedom) 하나를 잃게 된다. 요-피치-롤(Yaw-Pitch-Roll) 규약에서 피치가 $\pm 90°$가 되면, 요와 롤 축이 평행해져 어느 쪽을 변경해도 같은 회전이 발생한다.

**예시**: 비행기에서 피치를 90도 위로 올리면(기수가 수직 위를 향하면), 요와 롤이 모두 같은 축을 기준으로 회전한다. 회전 축 하나를 독립적으로 제어하는 능력을 잃게 된 것이다.

```
Normal state:              Gimbal lock (pitch = 90°):
  Yaw  ↻  (Y axis)           Yaw  ↻  (Y axis)
  Pitch ↻ (X axis)           Pitch ↻ (Z axis aligned!)
  Roll  ↻ (Z axis)           Roll  ↻ (Z axis)
  [3 independent axes]       [Yaw and Roll = same axis!]
```

### 8.3 쿼터니언(Quaternions): 해결책

**쿼터니언**은 짐벌 락 없이 3D 회전을 우아하게 표현하는 4차원 수다.

$$q = w + xi + yj + zk$$

또는 동치 표현: $q = (w, \mathbf{v})$, 여기서 $\mathbf{v} = (x, y, z)$.

단위 축 $\hat{\mathbf{u}}$를 기준으로 각도 $\theta$만큼의 회전은 다음과 같이 표현된다.

$$q = \left(\cos\frac{\theta}{2}, \sin\frac{\theta}{2} \cdot \hat{\mathbf{u}}\right)$$

**주요 특성**:
- **단위 쿼터니언(Unit quaternion)** ($|q| = 1$)이 회전을 표현한다
- **합성(Composition)**: $q_{\text{combined}} = q_2 \cdot q_1$ (쿼터니언 곱셈)
- **보간(Interpolation)**: SLERP(구면 선형 보간, Spherical Linear Interpolation)으로 방향 간 부드러운 블렌딩
- **짐벌 락 없음**: 쿼터니언에는 특이점(Singularity)이 없다
- **간결함**: 9개(3x3 행렬) 대신 4개의 숫자

```python
def quaternion_from_axis_angle(axis, angle_deg):
    """
    Create a unit quaternion from an axis-angle rotation.

    Why quaternions over Euler angles?
    1. No gimbal lock -- all orientations are reachable
    2. Smooth interpolation (SLERP)
    3. Compact (4 floats vs 9 for matrix)
    4. Numerically stable composition
    """
    axis = normalize(np.asarray(axis, dtype=float))
    half_angle = np.radians(angle_deg) / 2
    w = np.cos(half_angle)
    xyz = np.sin(half_angle) * axis
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def quaternion_to_matrix(q):
    """
    Convert a unit quaternion to a 4x4 rotation matrix.

    This avoids trigonometric functions entirely -- the conversion
    uses only multiplications and additions, which is more efficient.
    """
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y),   0],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x),   0],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y), 0],
        [0,             0,             0,              1]
    ], dtype=float)


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions (compose rotations).

    q1 * q2 applies q2 first, then q1 (same as matrix convention).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def slerp(q1, q2, t):
    """
    Spherical Linear Interpolation between two quaternions.

    SLERP produces constant-speed rotation along the shortest path
    between two orientations. This is essential for smooth animations.

    Parameters:
        q1, q2: unit quaternions (start and end orientations)
        t: interpolation parameter in [0, 1]
    """
    dot = np.dot(q1, q2)

    # If dot product is negative, negate one quaternion to take the shorter path
    # (q and -q represent the same rotation)
    if dot < 0:
        q2 = -q2
        dot = -dot

    # If quaternions are very close, use linear interpolation to avoid division by zero
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta

    return w1 * q1 + w2 * q2
```

---

## 9. 완전한 MVP 구현

```python
"""
Complete Model-View-Projection pipeline demonstration.

Transforms a 3D cube from object space through every coordinate space
to final screen pixel positions.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════
# Helper functions (from earlier sections)
# ═══════════════════════════════════════════════════════════════

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v

def make_translation(tx, ty, tz):
    return np.array([[1,0,0,tx],[0,1,0,ty],[0,0,1,tz],[0,0,0,1]], dtype=float)

def make_rotation_y(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=float)

def make_scale(sx, sy, sz):
    return np.array([[sx,0,0,0],[0,sy,0,0],[0,0,sz,0],[0,0,0,1]], dtype=float)

def look_at(eye, target, up):
    eye, target, up = [np.asarray(v, dtype=float) for v in [eye, target, up]]
    f = normalize(target - eye)
    r = normalize(np.cross(f, up))
    u = np.cross(r, f)
    return np.array([
        [r[0], r[1], r[2], -np.dot(r, eye)],
        [u[0], u[1], u[2], -np.dot(u, eye)],
        [-f[0],-f[1],-f[2], np.dot(f, eye)],
        [0, 0, 0, 1]
    ], dtype=float)

def perspective(fov_deg, aspect, near, far):
    fov = np.radians(fov_deg)
    t = near * np.tan(fov / 2)
    r = t * aspect
    return np.array([
        [near/r, 0, 0, 0],
        [0, near/t, 0, 0],
        [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
        [0, 0, -1, 0]
    ], dtype=float)

# ═══════════════════════════════════════════════════════════════
# Define scene
# ═══════════════════════════════════════════════════════════════

# A unit cube centered at origin (8 vertices)
cube_vertices = np.array([
    [-0.5, -0.5, -0.5],
    [ 0.5, -0.5, -0.5],
    [ 0.5,  0.5, -0.5],
    [-0.5,  0.5, -0.5],
    [-0.5, -0.5,  0.5],
    [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5,  0.5],
    [-0.5,  0.5,  0.5],
], dtype=float)

# ═══════════════════════════════════════════════════════════════
# Build MVP matrices
# ═══════════════════════════════════════════════════════════════

# Model: scale by 2, rotate 30 deg around Y, translate to (0, 1, -5)
M = make_translation(0, 1, -5) @ make_rotation_y(30) @ make_scale(2, 2, 2)

# View: camera at (0, 3, 5) looking at origin
V = look_at(eye=[0, 3, 5], target=[0, 0, 0], up=[0, 1, 0])

# Projection: 60 degree FOV, 16:9 aspect, near=0.1, far=100
P = perspective(fov_deg=60, aspect=16/9, near=0.1, far=100)

# Combined MVP
MVP = P @ V @ M

# ═══════════════════════════════════════════════════════════════
# Transform vertices through the pipeline
# ═══════════════════════════════════════════════════════════════

print("=== MVP Pipeline Demonstration ===\n")
print(f"Processing {len(cube_vertices)} vertices of a unit cube...\n")

# Screen dimensions
screen_w, screen_h = 1920, 1080

for i, v in enumerate(cube_vertices):
    # Step 1: Object space -> Clip space (via MVP)
    p_obj = np.array([v[0], v[1], v[2], 1.0])
    p_clip = MVP @ p_obj

    # Step 2: Perspective division -> NDC
    w = p_clip[3]
    p_ndc = p_clip[:3] / w

    # Step 3: Viewport transform -> Screen coordinates
    sx = (p_ndc[0] + 1) * 0.5 * screen_w
    sy = (1 - p_ndc[1]) * 0.5 * screen_h  # Flip Y (screen Y goes down)
    depth = (p_ndc[2] + 1) * 0.5  # Depth in [0, 1]

    if i < 4:  # Print first 4 vertices as examples
        print(f"Vertex {i}: object={v}")
        print(f"  clip=({p_clip[0]:.3f}, {p_clip[1]:.3f}, "
              f"{p_clip[2]:.3f}, {p_clip[3]:.3f})")
        print(f"  NDC=({p_ndc[0]:.3f}, {p_ndc[1]:.3f}, {p_ndc[2]:.3f})")
        print(f"  screen=({sx:.1f}, {sy:.1f}), depth={depth:.4f}")
        print()

print("... (remaining vertices follow the same process)")
```

---

## 10. 법선 변환(Normal Transformation)

기하 형태를 변환할 때, 법선 벡터(Normal vector)는 특별한 처리가 필요하다. 모델 행렬에 비균일 스케일(Non-uniform scaling)이 포함된 경우, 법선에 단순히 $\mathbf{M}$을 적용하면 잘못된 결과가 나온다.

올바른 **법선 행렬(Normal matrix)**은 모델 행렬의 왼쪽 상단 3x3 부분의 역행렬 전치(Inverse-transpose)다.

$$\mathbf{N} = (\mathbf{M}_{3\times3}^{-1})^T$$

**왜 그럴까?** 법선은 위치(Position)가 아니라 표면에 수직인 벡터다. 비균일 스케일은 표면을 변형시키지만 수직성(Perpendicularity)은 보존해야 한다. 역행렬 전치를 사용하면 변환된 법선이 변환된 표면에 여전히 수직임을 보장한다.

**증명 스케치**: $\mathbf{t}$가 접선 벡터(Tangent vector, 표면 평면에 놓인 벡터)라면 $\mathbf{n} \cdot \mathbf{t} = 0$이다. 변환 후: $\mathbf{n}' \cdot \mathbf{t}' = (\mathbf{N}\mathbf{n})^T (\mathbf{M}\mathbf{t}) = \mathbf{n}^T \mathbf{N}^T \mathbf{M} \mathbf{t}$. 이것이 0이 되려면 $\mathbf{N}^T \mathbf{M} = \mathbf{I}$이어야 하므로, $\mathbf{N} = (\mathbf{M}^{-1})^T$임을 얻는다.

```python
def compute_normal_matrix(model_matrix):
    """
    Compute the normal transformation matrix.

    If the model matrix has only rotation and uniform scale,
    the normal matrix equals the model's upper-left 3x3 (since
    rotation is orthogonal and uniform scale cancels out).
    But for non-uniform scaling, we MUST use the inverse-transpose.
    """
    linear = model_matrix[:3, :3]
    return np.linalg.inv(linear).T
```

---

## 요약

| 좌표 공간 | 범위 | 이전 단계 |
|-----------|------|-----------|
| 오브젝트(지역) | 임의 | 모델 정의 |
| 월드(World) | 임의 | 모델 행렬 $\mathbf{M}$ |
| 눈/카메라(Eye) | 카메라가 원점, $-z$ 방향 | 뷰 행렬 $\mathbf{V}$ |
| 클립(Clip) | $-w \leq x,y,z \leq w$ | 투영 행렬 $\mathbf{P}$ |
| NDC | $[-1, 1]^3$ | 원근 나눗셈 ($\div w$) |
| 스크린(Screen) | $[0, W] \times [0, H]$ 픽셀 | 뷰포트 변환 |

**핵심 정리**:
- MVP 체인 $\mathbf{P} \cdot \mathbf{V} \cdot \mathbf{M}$은 모든 3D 렌더링의 중심 변환이다
- 뷰 행렬은 카메라 위치와 바라보는 방향으로부터 **lookAt** 공식을 사용해 구성한다
- 원근 투영은 $w = -z$로 나눔으로써 원근 단축 효과를 만든다
- 깊이 정밀도는 비선형이며 근평면 근처에 집중된다
- 오일러 각도는 **짐벌 락** 문제가 있으며, 쿼터니언이 견고한 대안을 제공한다
- 법선은 반드시 모델 행렬의 **역행렬 전치**로 변환해야 한다

---

## 연습 문제

1. **MVP 구성**: $(3, 3, 3)$에 위치한 카메라가 up 벡터 $(0, 1, 0)$으로 원점을 바라볼 때, $(1, 0, -2)$에 위치한 단위 큐브에 대한 전체 MVP 행렬을 구성하라. 8개의 정점 모두를 스크린 좌표(1920x1080 뷰포트)로 변환하라.

2. **직교 투영 vs 원근 투영**: 직교 투영과 원근 투영으로 같은 장면을 렌더링하라. 시각적 차이를 설명하라. 어떤 경우에 직교 투영이 더 적합한가?

3. **깊이 버퍼 값**: $n = 0.1$, $f = 100$인 원근 투영에서, $z = -1$, $z = -10$, $z = -50$, $z = -100$에 있는 오브젝트의 NDC 깊이를 계산하라. $z_{\text{NDC}}(z)$ 함수를 그래프로 그리고, 이 비선형 분포가 왜 정밀도 문제를 일으키는지 논의하라.

4. **짐벌 락 시연**: 오일러 각도 회전 시스템을 구현하라. 피치가 90도일 때 요와 롤이 같은 회전을 만들어냄을 보여라. 그런 다음 쿼터니언으로 동일한 회전을 수행할 때 락이 발생하지 않음을 보여라.

5. **SLERP 애니메이션**: 쿼터니언을 사용해 "정면 바라보기"에서 "오른쪽으로 180도, 위로 45도"로의 부드러운 회전을 구현하라. SLERP로 10개의 중간 방향을 샘플링하고 각각을 회전 행렬로 변환하라.

6. **법선 변환**: 비균일 스케일 $(2, 1, 0.5)$과 회전이 포함된 모델 행렬을 만들라. 모델 행렬을 법선 벡터에 직접 적용하면 잘못된 결과가 나옴을, 역행렬 전치를 적용하면 올바른 수직 법선이 나옴을 보여라.

---

## 추가 자료

1. Marschner, S. & Shirley, P. *Fundamentals of Computer Graphics* (5th ed.), Ch. 7 -- "Viewing"
2. Akenine-Moller, T. et al. *Real-Time Rendering* (4th ed.), Ch. 4 -- "Transforms"
3. [Learn OpenGL -- Coordinate Systems](https://learnopengl.com/Getting-started/Coordinate-Systems) -- MVP 파이프라인 인터랙티브 설명
4. [Quaternions and Spatial Rotation](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation) -- 수학적 세부 사항
5. [The Depth Buffer Explained (blog)](https://developer.nvidia.com/content/depth-precision-visualized) -- NVIDIA의 깊이 정밀도 시각화
