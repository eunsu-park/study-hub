# 10. 레이 트레이싱 기초

[← 이전: 09. 씬 그래프와 공간 자료구조](09_Scene_Graphs_and_Spatial_Data_Structures.md) | [다음: 11. 패스 트레이싱과 전역 조명 →](11_Path_Tracing_and_Global_Illumination.md)

---

## 학습 목표

1. 레이(Ray)를 수학적으로 정의하고 가상 카메라에서 1차 레이를 생성한다
2. 레이-구, 레이-평면, 레이-삼각형 교차 테스트를 유도하고 구현한다
3. 효율적인 레이-삼각형 교차를 위한 Moller-Trumbore 알고리즘을 이해한다
4. 반사와 굴절을 포함한 재귀적(Whitted 방식) 레이 트레이싱을 구현한다
5. 굴절에 스넬의 법칙(Snell's Law)을, 반사 강도에 프레넬 방정식(Fresnel Equations)을 적용한다
6. 그림자 레이(Shadow Ray)가 광원의 가시성을 어떻게 결정하는지 설명한다
7. 가속 자료구조(L09의 BVH)를 실제 레이 트레이싱 성능과 연결한다
8. Python으로 구, 그림자, 반사를 렌더링하는 완전한 소프트웨어 레이 트레이서를 구축한다

---

## 왜 중요한가

레이 트레이싱은 **눈에서 역방향으로 빛을 추적하는** 것이라 생각하면 된다. 물리 세계에서 빛은 광원을 떠나 씬을 돌아다니다가 결국 눈에 들어온다. 이 순방향 과정을 시뮬레이션하는 것은 광선의 대부분이 시청자에게 도달하지 않으므로 지독히 낭비적이다. 레이 트레이싱은 이 과정을 역전시킨다: 카메라에서 씬으로 레이를 쏘아 "이 픽셀이 무엇을 보았는가?"를 묻는다.

1980년 Turner Whitted가 처음 공식화한 이 우아한 역전은 래스터화만으로는 어렵거나 불가능한 올바른 그림자, 반사, 굴절 효과를 포함한 놀랍도록 사실적인 이미지를 만들어 낸다. 오늘날 하드웨어 레이 트레이싱(NVIDIA RTX, AMD RDNA)은 실시간 레이 트레이싱을 실용화했으며, 현대 그래픽스를 다루는 누구에게나 기본 알고리즘의 이해는 필수적이다.

---

## 1. 레이 정의와 1차 레이 생성

### 1.1 수학적 레이

**레이(Ray)**는 원점 $\mathbf{o}$와 방향 $\mathbf{d}$로 정의된 반직선이다:

$$\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}, \quad t \ge 0$$

$t$ 값을 선택하면 레이 위의 임의의 점을 찾을 수 있다. 파라미터 $t$는 레이를 따라가는 "거리" 역할을 한다($\|\mathbf{d}\| = 1$일 때 정확히 거리와 같다).

### 1.2 카메라 모델

1차 레이를 생성하려면 다음을 정의한 가상 카메라가 필요하다:
- **시점(Eye Position)** $\mathbf{e}$
- **바라보는 방향** (또는 목표 점)
- **위 벡터(Up Vector)** $\mathbf{up}$
- **시야각(FOV, Field of View)** $\theta$
- **이미지 해상도** $W \times H$

정규 직교 카메라 기저를 구성한다:

$$\mathbf{w} = \frac{\mathbf{e} - \mathbf{target}}{\|\mathbf{e} - \mathbf{target}\|}, \quad
\mathbf{u} = \frac{\mathbf{up} \times \mathbf{w}}{\|\mathbf{up} \times \mathbf{w}\|}, \quad
\mathbf{v} = \mathbf{w} \times \mathbf{u}$$

여기서 $\mathbf{w}$는 씬에서 멀어지는 방향(오른손 좌표계 규약), $\mathbf{u}$는 카메라의 오른쪽, $\mathbf{v}$는 카메라의 실제 위쪽을 가리킨다.

### 1.3 픽셀 레이 생성

크기 $W \times H$인 이미지의 픽셀 $(i, j)$에 대해:

$$\text{aspect} = W / H$$

$$s = 2 \tan(\theta / 2)$$

$$u_{\text{pixel}} = s \cdot \text{aspect} \cdot \left(\frac{i + 0.5}{W} - 0.5\right)$$

$$v_{\text{pixel}} = s \cdot \left(0.5 - \frac{j + 0.5}{H}\right)$$

레이 방향(아직 정규화되지 않음)은:

$$\mathbf{d} = u_{\text{pixel}} \cdot \mathbf{u} + v_{\text{pixel}} \cdot \mathbf{v} - \mathbf{w}$$

$+0.5$ 오프셋은 레이를 픽셀의 모서리가 아닌 **중심**에 배치하여 앨리어싱을 줄인다.

---

## 2. 레이-오브젝트 교차

### 2.1 레이-구 교차

중심 $\mathbf{c}$, 반지름 $r$의 구는 다음을 만족한다:

$$\|\mathbf{p} - \mathbf{c}\|^2 = r^2$$

레이 방정식 $\mathbf{p} = \mathbf{o} + t\mathbf{d}$를 대입하면:

$$\|\mathbf{o} + t\mathbf{d} - \mathbf{c}\|^2 = r^2$$

$\boldsymbol{\ell} = \mathbf{o} - \mathbf{c}$로 놓고 전개하면:

$$(\mathbf{d} \cdot \mathbf{d})t^2 + 2(\boldsymbol{\ell} \cdot \mathbf{d})t + (\boldsymbol{\ell} \cdot \boldsymbol{\ell} - r^2) = 0$$

이는 이차방정식 $at^2 + bt + c = 0$으로:
- $a = \mathbf{d} \cdot \mathbf{d}$
- $b = 2(\boldsymbol{\ell} \cdot \mathbf{d})$
- $c = \boldsymbol{\ell} \cdot \boldsymbol{\ell} - r^2$

판별식 $\Delta = b^2 - 4ac$:
- $\Delta < 0$: 레이가 구를 빗나감
- $\Delta = 0$: 레이가 접선 방향 (한 점에서 교차)
- $\Delta > 0$: 레이가 관통 (두 점에서 교차)

가장 작은 양의 $t$가 보이는 교점을 준다. 교점 $\mathbf{p}$에서의 표면 법선은:

$$\mathbf{n} = \frac{\mathbf{p} - \mathbf{c}}{r}$$

### 2.2 레이-평면 교차

법선 $\mathbf{n}$과 점 $\mathbf{q}$로 정의된 평면 (또는 동등하게 $\mathbf{n} \cdot \mathbf{p} = d$, 여기서 $d = \mathbf{n} \cdot \mathbf{q}$):

$$\mathbf{n} \cdot (\mathbf{o} + t\mathbf{d}) = d$$

$$t = \frac{d - \mathbf{n} \cdot \mathbf{o}}{\mathbf{n} \cdot \mathbf{d}}$$

$\mathbf{n} \cdot \mathbf{d} = 0$이면 레이가 평면에 평행하다(교점 없음). $t < 0$이면 평면이 레이 원점 뒤에 있다.

### 2.3 레이-삼각형 교차: Moller-Trumbore 알고리즘

삼각형은 꼭짓점 $\mathbf{v}_0, \mathbf{v}_1, \mathbf{v}_2$로 정의된다. 삼각형 위의 임의의 점은 **무게중심 좌표(Barycentric Coordinates)**로 표현할 수 있다:

$$\mathbf{p} = (1 - u - v)\mathbf{v}_0 + u\mathbf{v}_1 + v\mathbf{v}_2, \quad u \ge 0, \; v \ge 0, \; u + v \le 1$$

이를 레이 방정식 $\mathbf{o} + t\mathbf{d}$와 같다고 놓으면:

$$\mathbf{o} + t\mathbf{d} = (1 - u - v)\mathbf{v}_0 + u\mathbf{v}_1 + v\mathbf{v}_2$$

선형 시스템으로 정리하면:

$$\begin{bmatrix} -\mathbf{d} & \mathbf{v}_1 - \mathbf{v}_0 & \mathbf{v}_2 - \mathbf{v}_0 \end{bmatrix} \begin{bmatrix} t \\ u \\ v \end{bmatrix} = \mathbf{o} - \mathbf{v}_0$$

**Moller-Trumbore 알고리즘**은 명시적 행렬 구성 없이 외적(Cross Product)을 이용한 크래머 공식(Cramer's Rule)으로 이를 풀어낸다:

$\mathbf{e}_1 = \mathbf{v}_1 - \mathbf{v}_0$, $\mathbf{e}_2 = \mathbf{v}_2 - \mathbf{v}_0$, $\mathbf{T} = \mathbf{o} - \mathbf{v}_0$이라 하면:

$$\mathbf{P} = \mathbf{d} \times \mathbf{e}_2, \quad \text{det} = \mathbf{e}_1 \cdot \mathbf{P}$$

$$u = \frac{\mathbf{T} \cdot \mathbf{P}}{\text{det}}, \quad \mathbf{Q} = \mathbf{T} \times \mathbf{e}_1$$

$$v = \frac{\mathbf{d} \cdot \mathbf{Q}}{\text{det}}, \quad t = \frac{\mathbf{e}_2 \cdot \mathbf{Q}}{\text{det}}$$

$u \ge 0$, $v \ge 0$, $u + v \le 1$, $t > 0$이면 교차가 유효하다.

**왜 Moller-Trumbore인가?** 나눗셈 1번, 외적 2번, 여러 번의 내적만 사용하며 역행렬이 불필요하다. 또한 꼭짓점 속성(법선, 텍스처 좌표) 보간에 필수적인 무게중심 좌표 $(u, v)$를 무료로 제공한다.

### 2.4 구현

```python
import numpy as np

def ray_sphere(origin, direction, center, radius):
    """
    Ray-sphere intersection.
    Returns (hit, t, normal) where t is the nearest positive intersection.
    """
    oc = origin - center
    a = np.dot(direction, direction)
    b = 2.0 * np.dot(oc, direction)
    c = np.dot(oc, oc) - radius * radius

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False, float('inf'), None

    sqrt_disc = np.sqrt(discriminant)
    # Why we try the smaller root first: it's the nearer intersection
    t = (-b - sqrt_disc) / (2.0 * a)
    if t < 1e-4:  # Avoid self-intersection
        t = (-b + sqrt_disc) / (2.0 * a)
    if t < 1e-4:
        return False, float('inf'), None

    hit_point = origin + t * direction
    normal = (hit_point - center) / radius
    return True, t, normal


def ray_plane(origin, direction, plane_normal, plane_d):
    """
    Ray-plane intersection.
    Plane equation: dot(normal, p) = plane_d.
    """
    denom = np.dot(plane_normal, direction)
    if abs(denom) < 1e-8:  # Ray parallel to plane
        return False, float('inf'), None

    t = (plane_d - np.dot(plane_normal, origin)) / denom
    if t < 1e-4:
        return False, float('inf'), None

    return True, t, plane_normal


def ray_triangle_moller_trumbore(origin, direction, v0, v1, v2):
    """
    Moller-Trumbore ray-triangle intersection.
    Returns (hit, t, u, v) where u, v are barycentric coordinates.
    """
    EPSILON = 1e-8
    e1 = v1 - v0
    e2 = v2 - v0

    # P = d x e2 -- this vector is reused for both det and u
    P = np.cross(direction, e2)
    det = np.dot(e1, P)

    # Why we check det near zero: ray is parallel to triangle plane
    if abs(det) < EPSILON:
        return False, float('inf'), 0, 0

    inv_det = 1.0 / det
    T = origin - v0

    # u is the first barycentric coordinate
    u = np.dot(T, P) * inv_det
    if u < 0.0 or u > 1.0:
        return False, float('inf'), 0, 0

    Q = np.cross(T, e1)

    # v is the second barycentric coordinate
    v = np.dot(direction, Q) * inv_det
    if v < 0.0 or u + v > 1.0:
        return False, float('inf'), 0, 0

    t = np.dot(e2, Q) * inv_det
    if t < EPSILON:
        return False, float('inf'), 0, 0

    return True, t, u, v
```

---

## 3. 그림자 레이 (Shadow Rays)

표면 점 $\mathbf{p}$가 그림자 안에 있는지 결정하려면, $\mathbf{p}$에서 각 광원을 향해 **그림자 레이(Shadow Ray)**를 쏜다:

$$\mathbf{r}_{\text{shadow}}(t) = \mathbf{p} + t\mathbf{L}$$

여기서 $\mathbf{L}$은 광원을 향하는 방향이다. 그림자 레이가 광원에 도달하기 전에 어떤 오브젝트를 맞추면, 해당 점은 그림자 안에 있으며 그 광원으로부터 직접 조명을 받지 않는다.

**자기 교차 문제**: 그림자 레이의 원점 $\mathbf{p}$는 표면 위에 있다. 부동소수점 오차로 인해 레이가 같은 표면과 즉시 교차할 수 있다. 표준 수정 방법은 법선을 따라 원점을 약간 오프셋하는 것이다:

$$\mathbf{p}_{\text{offset}} = \mathbf{p} + \epsilon \cdot \mathbf{n}$$

여기서 $\epsilon \approx 10^{-4}$는 작은 편향값이다. 또는 최소 $t$ 임계값을 사용할 수 있다.

---

## 4. 재귀적 레이 트레이싱 (Whitted 방식)

### 4.1 Whitted 모델

Turner Whitted의 1980년 논문은 **재귀적 레이 트레이싱(Recursive Ray Tracing)**을 소개했으며, 각 레이 충돌이 추가 레이를 생성할 수 있다:

1. **1차 레이(Primary Ray)**: 카메라에서 각 픽셀을 통해
2. **그림자 레이(Shadow Ray)**: 충돌 점에서 각 광원을 향해 (그림자 결정)
3. **반사 레이(Reflection Ray)**: 표면이 반사성이면 반사 방향으로 레이를 생성
4. **굴절 레이(Refraction Ray)**: 표면이 투명하면 굴절된 방향으로 레이를 생성

픽셀의 색상은 재귀적으로 계산된다:

$$L(\mathbf{p}) = L_{\text{ambient}} + L_{\text{direct}} + k_r \cdot L(\mathbf{p}_{\text{reflect}}) + k_t \cdot L(\mathbf{p}_{\text{refract}})$$

여기서 $k_r$은 반사 계수, $k_t$는 투과(굴절) 계수다.

재귀는 다음 조건에서 종료된다:
- 최대 깊이에 도달 (예: 5번 바운스)
- 레이가 모든 오브젝트를 빗나감 (배경색 반환)
- 기여도가 무시할 수 있을 만큼 작아짐

### 4.2 반사 방향

법선 $\mathbf{n}$을 가진 표면에 입사 레이 방향 $\mathbf{d}$가 충돌할 때, **완전 반사(Perfect Reflection)** 방향은:

$$\mathbf{r} = \mathbf{d} - 2(\mathbf{d} \cdot \mathbf{n})\mathbf{n}$$

이 공식은 $\mathbf{d}$를 $\mathbf{n}$ 주위로 "미러링"한다. $\mathbf{d}$는 표면을 *향해* 가리키므로, 앞면 충돌에서는 $\mathbf{d} \cdot \mathbf{n} < 0$임을 유의하라.

### 4.3 굴절: 스넬의 법칙

빛이 굴절률 $\eta_1$의 매질에서 굴절률 $\eta_2$의 매질로 이동할 때, 굴절된 방향은 **스넬의 법칙(Snell's Law)**을 따른다:

$$\eta_1 \sin\theta_1 = \eta_2 \sin\theta_2$$

여기서 $\theta_1$은 입사각, $\theta_2$는 굴절각이다.

굴절된 방향 벡터는:

$$\mathbf{t} = \frac{\eta_1}{\eta_2}\mathbf{d} + \left(\frac{\eta_1}{\eta_2}\cos\theta_1 - \cos\theta_2\right)\mathbf{n}$$

여기서:
$$\cos\theta_1 = -\mathbf{d} \cdot \mathbf{n}$$
$$\cos^2\theta_2 = 1 - \left(\frac{\eta_1}{\eta_2}\right)^2(1 - \cos^2\theta_1)$$

$\cos^2\theta_2 < 0$이면 **전반사(Total Internal Reflection)**가 발생한다(굴절 레이 없음).

일반적인 굴절률: 공기 $\approx 1.0$, 물 $= 1.33$, 유리 $= 1.5$, 다이아몬드 $= 2.42$.

### 4.4 프레넬 방정식

**프레넬 방정식(Fresnel Equations)**은 계면(Interface)에서 얼마나 많은 빛이 반사되고 굴절되는지 결정한다. 그래픽스에서는 Schlick 근사가 일반적으로 사용된다:

$$R(\theta) \approx R_0 + (1 - R_0)(1 - \cos\theta)^5$$

여기서 $R_0 = \left(\frac{\eta_1 - \eta_2}{\eta_1 + \eta_2}\right)^2$은 수직 입사(Normal Incidence)에서의 반사율이다.

경사 각도($\theta \to 90°$)에서 $R \to 1$ -- 거의 모든 빛이 반사된다. 이것이 얕은 각도에서 수면을 볼 때 선명한 반사가 보이는 이유다.

---

## 5. 레이 트레이싱을 위한 가속 자료구조

모든 레이를 모든 오브젝트에 대해 테스트하면 레이당 $O(N)$이다. 수백만 개의 삼각형과 수백만 개의 픽셀이 있는 씬에서 이것은 너무 느리다.

**가속 자료구조**([레슨 9](09_Scene_Graphs_and_Spatial_Data_Structures.md)에서 자세히 다룸)는 이를 레이당 $O(\log N)$으로 줄인다:

- **BVH(경계 볼륨 계층 구조)**: 가장 일반적인 선택. AABB 트리를 구성하고 하향식으로 순회하며 레이가 빗나가는 서브트리를 건너뛴다.
- **kd-트리**: 축 정렬 평면으로 공간을 분할한다. 역사적으로 인기 있었지만 현재는 BVH가 일반적으로 선호된다.
- **균등 격자(Uniform Grid)**: 구성이 단순하지만 오브젝트 분포가 불균등할 경우 성능이 저하된다.

현대 GPU 레이 트레이싱(RTX, DXR, Vulkan RT)은 하드웨어 가속 BVH 순회를 사용한다.

---

## 6. Python 구현: 간단한 레이 트레이서

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

# --- Scene Definition ---

@dataclass
class Material:
    """Surface material properties."""
    color: np.ndarray           # Diffuse color (RGB, 0-1)
    ambient: float = 0.1        # Ambient coefficient
    diffuse: float = 0.7        # Diffuse coefficient
    specular: float = 0.3       # Specular coefficient
    shininess: float = 50.0     # Specular exponent
    reflectivity: float = 0.0   # 0 = no reflection, 1 = perfect mirror
    transparency: float = 0.0   # 0 = opaque, 1 = fully transparent
    ior: float = 1.5            # Index of refraction (glass)


@dataclass
class SceneSphere:
    """A sphere in the scene."""
    center: np.ndarray
    radius: float
    material: Material

    def intersect(self, origin, direction):
        """Ray-sphere intersection. Returns (t, normal) or (inf, None)."""
        oc = origin - self.center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - self.radius ** 2
        disc = b * b - 4 * a * c
        if disc < 0:
            return float('inf'), None
        sqrt_disc = np.sqrt(disc)
        t = (-b - sqrt_disc) / (2.0 * a)
        if t < 1e-4:
            t = (-b + sqrt_disc) / (2.0 * a)
        if t < 1e-4:
            return float('inf'), None
        hit = origin + t * direction
        normal = (hit - self.center) / self.radius
        return t, normal


@dataclass
class Plane:
    """An infinite plane."""
    point: np.ndarray       # A point on the plane
    normal: np.ndarray      # Surface normal (unit)
    material: Material

    def intersect(self, origin, direction):
        """Ray-plane intersection."""
        denom = np.dot(self.normal, direction)
        if abs(denom) < 1e-8:
            return float('inf'), None
        t = np.dot(self.point - origin, self.normal) / denom
        if t < 1e-4:
            return float('inf'), None
        return t, self.normal.copy()


@dataclass
class Light:
    """A point light source."""
    position: np.ndarray
    color: np.ndarray       # Light color/intensity
    intensity: float = 1.0


def normalize(v):
    """Safely normalize a vector."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def reflect(d, n):
    """Compute reflection direction: d is incoming (toward surface)."""
    return d - 2 * np.dot(d, n) * n


def refract(d, n, eta_ratio):
    """
    Compute refraction direction using Snell's law.
    d: incoming direction (toward surface, unit)
    n: surface normal (unit, pointing outward)
    eta_ratio: eta_i / eta_t
    Returns refracted direction or None (total internal reflection).
    """
    cos_i = -np.dot(d, n)
    sin2_t = eta_ratio ** 2 * (1.0 - cos_i ** 2)
    if sin2_t > 1.0:
        # Total internal reflection
        return None
    cos_t = np.sqrt(1.0 - sin2_t)
    return eta_ratio * d + (eta_ratio * cos_i - cos_t) * n


def fresnel_schlick(cos_theta, eta1, eta2):
    """
    Schlick's approximation for Fresnel reflectance.
    Returns fraction of light that is reflected.
    """
    r0 = ((eta1 - eta2) / (eta1 + eta2)) ** 2
    return r0 + (1 - r0) * (1 - cos_theta) ** 5


# --- Ray Tracer Core ---

class RayTracer:
    def __init__(self, objects, lights, bg_color=np.array([0.1, 0.1, 0.2]),
                 max_depth=5):
        self.objects = objects
        self.lights = lights
        self.bg_color = bg_color
        self.max_depth = max_depth

    def find_nearest(self, origin, direction):
        """Find the nearest intersection among all objects."""
        nearest_t = float('inf')
        nearest_obj = None
        nearest_normal = None

        for obj in self.objects:
            t, normal = obj.intersect(origin, direction)
            if t < nearest_t:
                nearest_t = t
                nearest_obj = obj
                nearest_normal = normal

        return nearest_t, nearest_obj, nearest_normal

    def shade(self, origin, direction, depth=0):
        """
        Recursively trace a ray and compute the color.
        This is the heart of Whitted-style ray tracing.
        """
        if depth >= self.max_depth:
            return self.bg_color.copy()

        t, obj, normal = self.find_nearest(origin, direction)
        if obj is None:
            return self.bg_color.copy()

        hit_point = origin + t * direction
        mat = obj.material

        # Ensure normal faces the incoming ray
        # Why: for transparent objects, we may be inside the surface
        if np.dot(normal, direction) > 0:
            normal = -normal

        color = np.zeros(3)

        # --- Ambient term ---
        color += mat.ambient * mat.color

        # --- Direct illumination (diffuse + specular) for each light ---
        for light in self.lights:
            light_dir = normalize(light.position - hit_point)
            light_dist = np.linalg.norm(light.position - hit_point)

            # Shadow test: cast ray toward light, check for occlusion
            shadow_origin = hit_point + 1e-4 * normal
            shadow_t, shadow_obj, _ = self.find_nearest(shadow_origin, light_dir)

            if shadow_t < light_dist:
                # Point is in shadow for this light
                continue

            # Diffuse (Lambertian)
            n_dot_l = max(0.0, np.dot(normal, light_dir))
            color += mat.diffuse * n_dot_l * mat.color * light.color * light.intensity

            # Specular (Blinn-Phong)
            view_dir = normalize(-direction)
            half_vec = normalize(light_dir + view_dir)
            n_dot_h = max(0.0, np.dot(normal, half_vec))
            color += mat.specular * (n_dot_h ** mat.shininess) * light.color * light.intensity

        # --- Reflection ---
        if mat.reflectivity > 0.0 and depth < self.max_depth:
            reflect_dir = normalize(reflect(direction, normal))
            reflect_origin = hit_point + 1e-4 * normal
            reflect_color = self.shade(reflect_origin, reflect_dir, depth + 1)
            color += mat.reflectivity * reflect_color

        # --- Refraction (transparency) ---
        if mat.transparency > 0.0 and depth < self.max_depth:
            # Determine if we're entering or leaving the object
            entering = np.dot(direction, normal) < 0
            if entering:
                eta_ratio = 1.0 / mat.ior  # Air -> Material
                refract_normal = normal
            else:
                eta_ratio = mat.ior / 1.0   # Material -> Air
                refract_normal = -normal

            cos_i = abs(np.dot(direction, refract_normal))
            kr = fresnel_schlick(cos_i, 1.0, mat.ior)

            refract_dir = refract(normalize(direction), refract_normal, eta_ratio)
            if refract_dir is not None:
                refract_origin = hit_point - 1e-4 * refract_normal
                refract_color = self.shade(refract_origin, normalize(refract_dir),
                                           depth + 1)
                # Mix reflection and refraction using Fresnel
                color = color * (1 - mat.transparency) \
                      + mat.transparency * (kr * reflect(direction, normal) is not None
                                            and self.shade(hit_point + 1e-4 * normal,
                                                          normalize(reflect(direction, normal)),
                                                          depth + 1) * kr
                                            + (1 - kr) * refract_color
                                            if True else refract_color)
                # Simplified: blend refracted and reflected
                reflect_dir = normalize(reflect(direction, normal))
                reflect_origin = hit_point + 1e-4 * normal
                reflect_color = self.shade(reflect_origin, reflect_dir, depth + 1)

                color = color * (1 - mat.transparency) \
                      + mat.transparency * (kr * reflect_color + (1 - kr) * refract_color)

        # Clamp to [0, 1]
        return np.clip(color, 0.0, 1.0)

    def render(self, width, height, fov_deg=60.0,
               eye=np.array([0, 1, 5.0]),
               target=np.array([0, 0, 0])):
        """
        Render the scene to an image array (H, W, 3).
        """
        image = np.zeros((height, width, 3))
        fov_rad = np.radians(fov_deg)
        aspect = width / height

        # Build camera coordinate frame
        forward = normalize(target - eye)
        right = normalize(np.cross(forward, np.array([0, 1, 0])))
        up = np.cross(right, forward)

        # Image plane half-dimensions
        half_h = np.tan(fov_rad / 2)
        half_w = half_h * aspect

        for j in range(height):
            for i in range(width):
                # Map pixel to [-1, 1] range
                u = (2 * (i + 0.5) / width - 1) * half_w
                v = (1 - 2 * (j + 0.5) / height) * half_h

                direction = normalize(forward + u * right + v * up)
                color = self.shade(eye, direction)
                image[j, i] = color

            # Progress indicator
            if (j + 1) % (height // 10) == 0:
                print(f"  Row {j+1}/{height} ({100*(j+1)//height}%)")

        return image


# --- Build Scene ---

# Materials
red_mat = Material(color=np.array([0.9, 0.1, 0.1]), reflectivity=0.2)
green_mat = Material(color=np.array([0.1, 0.9, 0.1]), reflectivity=0.1)
blue_mat = Material(color=np.array([0.1, 0.1, 0.9]), reflectivity=0.3)
mirror_mat = Material(color=np.array([0.9, 0.9, 0.9]),
                      reflectivity=0.8, diffuse=0.2, specular=0.8)
floor_mat = Material(color=np.array([0.5, 0.5, 0.5]), reflectivity=0.1)

# Objects
objects = [
    SceneSphere(center=np.array([-2.0, 0.5, -1.0]), radius=1.0, material=red_mat),
    SceneSphere(center=np.array([0.0, 0.7, 0.0]),   radius=1.2, material=mirror_mat),
    SceneSphere(center=np.array([2.0, 0.5, -0.5]),   radius=1.0, material=blue_mat),
    SceneSphere(center=np.array([0.5, 0.3, 2.0]),    radius=0.6, material=green_mat),
    Plane(point=np.array([0, -0.5, 0]), normal=np.array([0, 1, 0]),
          material=floor_mat),
]

# Lights
lights = [
    Light(position=np.array([-5, 8, 5]),  color=np.array([1, 1, 1]),    intensity=0.8),
    Light(position=np.array([5, 6, -3]),   color=np.array([0.8, 0.9, 1]), intensity=0.6),
]

# --- Render ---
tracer = RayTracer(objects, lights, max_depth=4)
print("Rendering 320x240 image...")
image = tracer.render(320, 240, fov_deg=60,
                      eye=np.array([0, 2, 6]),
                      target=np.array([0, 0, 0]))

# Save result (requires matplotlib or PIL)
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7.5))
    plt.imshow(image)
    plt.axis('off')
    plt.title('Whitted-Style Ray Tracer')
    plt.tight_layout()
    plt.savefig('ray_trace_result.png', dpi=150)
    plt.close()
    print("Saved ray_trace_result.png")
except ImportError:
    # Fallback: save as PPM
    with open('ray_trace_result.ppm', 'w') as f:
        f.write(f'P3\n{320} {240}\n255\n')
        for j in range(240):
            for i in range(320):
                r, g, b = (image[j, i] * 255).astype(int)
                f.write(f'{r} {g} {b} ')
            f.write('\n')
    print("Saved ray_trace_result.ppm")
```

---

## 7. 레이 트레이싱 대 래스터화

| 측면 | 래스터화(Rasterization) | 레이 트레이싱(Ray Tracing) |
|------|------------------------|--------------------------|
| 접근 방식 | 오브젝트 순서: 각 삼각형을 화면에 투영 | 이미지 순서: 각 픽셀에 대해 레이를 추적 |
| 그림자 | 섀도우 맵 필요 (근사치) | 자연스러움 (그림자 레이가 정확) |
| 반사 | 화면 공간 또는 큐브 맵 (근사치) | 자연스러움 (반사 레이가 정확) |
| 굴절 | 매우 어려움 | 자연스러움 (굴절 레이) |
| 성능 | 삼각형 수에서 $O(N)$; GPU 최적화 | BVH 적용 시 $O(N \log N)$; 역사적으로 느림 |
| 실시간 | 게임의 기본 방식 | 이제 RTX 하드웨어로 가능 |
| 전역 조명 | 근사치 (SSAO, 프로브) | 패스 트레이싱으로 확장 (L11) |

현대 렌더러는 두 방식을 종종 결합한다: 1차 가시성에는 래스터화를, 그림자, 반사, 앰비언트 오클루전에는 레이 트레이싱을 사용한다.

---

## 8. 실용적 고려사항

### 8.1 수치 안정성

부동소수점 오차는 아티팩트를 유발한다:
- **섀도우 에크니(Shadow Acne)**: 그림자 레이의 자기 교차. 수정: 레이 원점을 $\epsilon \cdot \mathbf{n}$만큼 오프셋
- **피터 패닝(Peter Panning)**: 과도한 편향이 그림자를 오브젝트로부터 멀어지게 한다. 수정: 작은 $\epsilon$ 사용
- **물샘 없는 교차(Watertight Intersections)**: 공유 모서리 삼각형 사이의 틈. 현대 알고리즘(예: Woop et al. 2013)은 물샘 없는 테스트를 보장한다

### 8.2 안티앨리어싱

픽셀당 하나의 레이는 앨리어스된(계단 현상이 있는) 모서리를 만든다. **수퍼샘플링(Supersampling)**은 픽셀당 여러 레이를 작은 무작위 오프셋(층화 또는 지터 샘플링)으로 쏘고 결과를 평균한다:

$$\text{픽셀 색상} = \frac{1}{N}\sum_{k=1}^{N} \text{shade}(\mathbf{r}_k)$$

$N = 4$(2x2 격자)만으로도 계단 현상을 크게 줄인다.

### 8.3 감마 보정

원시 선형 공간 색상은 화면에 표시하기 전에 **감마 보정(Gamma Correction)**이 필요하다:

$$c_{\text{display}} = c_{\text{linear}}^{1/\gamma}, \quad \gamma = 2.2$$

감마 보정 없이는 이미지가 너무 어둡게 보인다.

---

## 요약

| 개념 | 핵심 공식 / 아이디어 |
|------|-------------------|
| 레이 | $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ |
| 레이-구 | 이차방정식: $at^2 + bt + c = 0$; 판별식이 교차 횟수 결정 |
| 레이-평면 | $t = (d - \mathbf{n} \cdot \mathbf{o}) / (\mathbf{n} \cdot \mathbf{d})$ |
| Moller-Trumbore | 외적을 이용한 크래머 공식; $t$, $u$, $v$ 제공 |
| 그림자 레이 | 충돌 점에서 광원을 향해 쏨; 더 가까운 교차 = 그림자 |
| 반사 | $\mathbf{r} = \mathbf{d} - 2(\mathbf{d} \cdot \mathbf{n})\mathbf{n}$ |
| 스넬의 법칙 | $\eta_1 \sin\theta_1 = \eta_2 \sin\theta_2$ |
| 프레넬 (Schlick) | $R(\theta) \approx R_0 + (1 - R_0)(1 - \cos\theta)^5$ |
| Whitted 트레이싱 | 재귀적: 1차 + 그림자 + 반사 + 굴절 레이 |
| BVH 가속 | 레이당 비용을 $O(N)$에서 $O(\log N)$으로 감소 |

## 연습문제

1. **손으로 레이-구**: 레이 원점 $(0, 0, 5)$, 방향 $(0, 0, -1)$, 구 중심 $(0, 0, 0)$, 반지름 1이 주어졌을 때, 두 교차 $t$ 값을 구하라. 더 가까운 교점에서의 표면 법선은 무엇인가?

2. **Moller-Trumbore**: 알고리즘을 구현하고, 삼각형 $\mathbf{v}_0 = (0, 0, 0)$, $\mathbf{v}_1 = (1, 0, 0)$, $\mathbf{v}_2 = (0, 1, 0)$과 레이 원점 $(0.2, 0.2, 1)$, 방향 $(0, 0, -1)$로 테스트하라. 무게중심 좌표를 검증하라.

3. **굴절 시각화**: 레이 트레이서에 유리 구(IOR = 1.5)를 추가하도록 수정하라. 구를 통해 볼 때 배경 오브젝트가 어떻게 왜곡되는지 관찰하라. 다른 IOR 값으로 실험해 보라.

4. **그림자 비교**: 같은 씬을 광원 1개, 그 다음 다른 위치의 광원 3개로 렌더링하라. 그림자 패턴을 비교하라. 왜 다수의 광원이 더 부드러운 그림자처럼 보이는가?

5. **안티앨리어싱**: $4\times$ 수퍼샘플링(픽셀당 2x2 지터 격자)을 구현하라. 단일 샘플 렌더링과 결과를 비교하라. 성능 차이를 측정하라.

6. **BVH 통합**: [레슨 9](09_Scene_Graphs_and_Spatial_Data_Structures.md)의 BVH를 이 레이 트레이서에 연결하라. 100개의 무작위 구를 생성하고 BVH 가속 유무에 따른 렌더링 시간을 비교하라.

## 더 읽을거리

- Whitted, T. "An Improved Illumination Model for Shaded Display." *Communications of the ACM*, 1980. (재귀적 레이 트레이싱의 선구적 논문)
- Pharr, M., Jakob, W., Humphreys, G. *Physically Based Rendering*, 4th ed. MIT Press, 2023. (2-4장: 레이-오브젝트 교차, 가속 구조)
- Shirley, P. *Ray Tracing in One Weekend*. Online, 2020. (훌륭한 실습 튜토리얼)
- Moller, T., Trumbore, B. "Fast, Minimum Storage Ray/Triangle Intersection." *Journal of Graphics Tools*, 1997. (표준 레이-삼각형 알고리즘)
- Akenine-Moller, T. et al. *Real-Time Rendering*, 4th ed. CRC Press, 2018. (26장: 실시간 레이 트레이싱)
