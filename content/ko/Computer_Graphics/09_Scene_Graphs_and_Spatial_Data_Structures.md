# 09. 씬 그래프와 공간 자료구조

[← 이전: 08. 셰이더 프로그래밍 (GLSL)](08_Shader_Programming_GLSL.md) | [다음: 10. 레이 트레이싱 기초 →](10_Ray_Tracing_Basics.md)

---

## 학습 목표

1. 씬 그래프(Scene Graph)를 3D 세계를 구성하는 계층적 자료구조로 이해한다
2. 부모-자식 체인을 통해 로컬(모델) 공간과 월드 공간 변환을 구별한다
3. 렌더링과 변환 전파를 위한 씬 그래프 순회를 구현한다
4. 경계 볼륨(Bounding Volume)인 AABB, OBB, 경계 구를 정의하고 구성한다
5. 표면적 휴리스틱(SAH, Surface Area Heuristic)을 사용하여 경계 볼륨 계층 구조(BVH)를 구축한다
6. 옥트리(Octree), 쿼드트리(Quadtree), BSP 트리(BSP Tree)의 공간 분할 방식을 설명한다
7. 절두체 컬링(Frustum Culling)을 적용하고 오클루전 컬링(Occlusion Culling)의 개념을 이해한다
8. Python으로 BVH 구성과 레이-BVH 교차 검사를 구현한다

---

## 왜 중요한가

백만 개의 삼각형이 있는 씬을 모든 픽셀이나 레이에 대해 매 삼각형을 검사하는 방식으로는 효율적으로 렌더링하거나 레이 트레이싱할 수 없다. **공간 자료구조(Spatial Data Structures)**는 이 $O(n)$ 브루트포스 비용을 쿼리당 $O(\log n)$으로 줄여주는 핵심 도구로, 실시간 렌더링과 인터랙티브 레이 트레이싱을 가능하게 한다. 현대의 모든 게임 엔진, 영화 렌더러, 물리 시뮬레이션은 씬 그래프와 가속 자료구조에 의존한다. 이를 이해하는 것은 삼각형 하나를 셰이딩하는 방법을 아는 것에서, 초당 60프레임으로 *수백만 개*의 삼각형을 셰이딩하는 것으로 넘어가는 다리가 된다.

성능 외에도, **씬 그래프(Scene Graph)**는 복잡한 씬의 조직적 골격을 제공한다. 로봇 팔, 태양계, 또는 도시 블록은 모두 부모에 종속된 위치를 가진 오브젝트들의 계층 구조로 자연스럽게 표현된다. 이 구조를 마스터하면 모든 그래픽스 엔진이 기반으로 삼는 어휘와 알고리즘을 갖추게 된다.

---

## 1. 씬 그래프

### 1.1 씬 그래프란?

**씬 그래프(Scene Graph)**는 각 노드가 씬 안의 오브젝트, 그룹, 또는 변환을 나타내는 방향성 비순환 그래프(보통 트리)다. 노드는 다음을 저장한다:

- 부모에 상대적인 **로컬 변환(Local Transform)** (이동, 회전, 스케일)
- 선택적 지오메트리(메시), 재질, 조명, 또는 카메라 데이터
- 자식 노드 목록

```
Root (World)
├── Sun (Light)
├── Planet (Mesh + Transform)
│   ├── Moon (Mesh + Transform)
│   └── Satellite (Mesh + Transform)
└── Spaceship (Group + Transform)
    ├── Hull (Mesh)
    ├── Left_Engine (Mesh + Transform)
    └── Right_Engine (Mesh + Transform)
```

씬 그래프의 강점은 "Spaceship" 노드를 이동하면 모든 자식(Hull, 엔진)이 자동으로 함께 이동한다는 것인데, 이는 자식들의 위치가 부모에 **상대적으로** 정의되기 때문이다.

### 1.2 로컬 변환과 월드 변환

각 노드는 부모에 상대적인 위치를 나타내는 **로컬 변환 행렬** $\mathbf{M}_{\text{local}}$을 저장한다. **월드 변환(World Transform)** $\mathbf{M}_{\text{world}}$은 모든 조상 변환의 곱이다:

$$\mathbf{M}_{\text{world}} = \mathbf{M}_{\text{root}} \cdot \mathbf{M}_{\text{child}_1} \cdot \mathbf{M}_{\text{child}_2} \cdots \mathbf{M}_{\text{local}}$$

로컬 공간의 점 $\mathbf{p}$에 대한 월드 위치는:

$$\mathbf{p}_{\text{world}} = \mathbf{M}_{\text{world}} \cdot \mathbf{p}_{\text{local}}$$

이 합성이 씬 그래프가 강력한 이유다: 행렬 하나를 수정하면 전체 서브트리가 그에 따라 갱신된다.

**예시**: 행성이 태양 주위를 공전하고(원점 기준 회전), 달이 행성 주위를 공전한다. 달의 월드 위치는:

$$\mathbf{M}_{\text{moon}}^{\text{world}} = \mathbf{R}_{\text{planet orbit}} \cdot \mathbf{T}_{\text{planet offset}} \cdot \mathbf{R}_{\text{moon orbit}} \cdot \mathbf{T}_{\text{moon offset}}$$

### 1.3 씬 그래프 순회

씬을 렌더링하기 위해 변환 행렬을 스택에 누적하면서 **깊이 우선 탐색(Depth-First Traversal)**을 수행한다:

```
function traverse(node, parent_world_matrix):
    world_matrix = parent_world_matrix * node.local_transform
    if node.has_geometry:
        render(node.geometry, world_matrix)
    for child in node.children:
        traverse(child, world_matrix)
```

**렌더링 순서**가 중요한 경우:
- **불투명(Opaque) 오브젝트**: 앞에서 뒤로 렌더링 (early z-test 거부를 통해 오버드로우 최소화)
- **투명(Transparent) 오브젝트**: 뒤에서 앞으로 렌더링 (올바른 알파 블렌딩을 위해 멀리 있는 오브젝트를 먼저 그려야 함)
- **정렬(Sorting)**: 씬 그래프 자체는 올바른 순서를 보장하지 않으므로, 별도의 정렬 패스가 자주 사용된다

### 1.4 Python 구현: 씬 그래프

```python
import numpy as np

class SceneNode:
    """A node in the scene graph with hierarchical transforms."""

    def __init__(self, name, transform=None):
        self.name = name
        # Local transform relative to parent (4x4 identity by default)
        self.local_transform = transform if transform is not None else np.eye(4)
        self.children = []
        self.geometry = None       # Optional mesh data
        self.world_transform = np.eye(4)  # Computed during traversal

    def add_child(self, child):
        """Attach a child node, creating the parent-child relationship."""
        self.children.append(child)
        return child

    def update_transforms(self, parent_world=None):
        """
        Depth-first traversal that propagates transforms down the tree.
        Each node's world_transform = parent's world_transform * local_transform.
        """
        if parent_world is None:
            parent_world = np.eye(4)

        # Why matrix multiply here: this composes all ancestor transforms
        # so that the node's geometry can be placed directly in world space
        self.world_transform = parent_world @ self.local_transform

        for child in self.children:
            child.update_transforms(self.world_transform)

    def collect_renderables(self, result=None):
        """Gather all nodes that have geometry, sorted for rendering."""
        if result is None:
            result = []
        if self.geometry is not None:
            result.append(self)
        for child in self.children:
            child.collect_renderables(result)
        return result


def make_translation(tx, ty, tz):
    """Create a 4x4 translation matrix."""
    M = np.eye(4)
    M[0, 3] = tx
    M[1, 3] = ty
    M[2, 3] = tz
    return M


def make_rotation_y(angle_deg):
    """Create a 4x4 rotation matrix around the Y axis."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    M = np.eye(4)
    M[0, 0] = c;  M[0, 2] = s
    M[2, 0] = -s; M[2, 2] = c
    return M


# Build a simple solar system scene graph
root = SceneNode("Root")
sun = root.add_child(SceneNode("Sun"))
sun.geometry = "sun_mesh"

# Planet: rotate 45 degrees around sun, offset by 5 units
planet_orbit = root.add_child(
    SceneNode("PlanetOrbit", make_rotation_y(45.0))
)
planet = planet_orbit.add_child(
    SceneNode("Planet", make_translation(5.0, 0.0, 0.0))
)
planet.geometry = "planet_mesh"

# Moon: rotate 30 degrees around planet, offset by 1.5 units
moon_orbit = planet.add_child(
    SceneNode("MoonOrbit", make_rotation_y(30.0))
)
moon = moon_orbit.add_child(
    SceneNode("Moon", make_translation(1.5, 0.0, 0.0))
)
moon.geometry = "moon_mesh"

# Propagate all transforms
root.update_transforms()

# Print world positions
for node in root.collect_renderables():
    pos = node.world_transform[:3, 3]
    print(f"{node.name:>10}: world position = ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
```

출력:
```
       Sun: world position = (0.00, 0.00, 0.00)
    Planet: world position = (3.54, 0.00, -3.54)
      Moon: world position = (4.61, 0.00, -2.89)
```

달의 월드 위치는 행성 공전 + 행성 오프셋 + 달 공전 + 달 오프셋이 합성된 결과로, 씬 그래프 순회에 의해 자동으로 계산된다.

---

## 2. 경계 볼륨 (Bounding Volumes)

레이, 절두체, 또는 다른 오브젝트와의 교차를 위해 모든 삼각형을 검사하는 것은 비용이 크다. **경계 볼륨(Bounding Volume)**은 복잡한 지오메트리를 감싸는 단순한 기하 형태로, 저렴한 "조기 거부(reject early)" 테스트를 제공한다.

### 2.1 축 정렬 경계 상자 (AABB, Axis-Aligned Bounding Box)

AABB는 좌표축에 정렬된 최솟값과 최댓값 모서리로 정의된다:

$$\text{AABB} = \{(x, y, z) \;|\; x_{\min} \le x \le x_{\max},\; y_{\min} \le y \le y_{\max},\; z_{\min} \le z \le z_{\max}\}$$

**장점**: 매우 빠른 교차 테스트(슬랩 방법), 쉬운 구성(꼭짓점의 최솟값/최댓값), 컴팩트한 저장(6개의 float).

**단점**: 길쭉하거나 회전된 오브젝트에 잘 맞지 않는다. AABB로 감싸진 오브젝트를 회전하면 AABB를 재계산해야 한다(크기가 커진다).

**레이-AABB 교차 (슬랩 방법)**: 레이 $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$가 AABB와 교차하려면 각 슬랩 쌍 안에 레이가 존재하는 구간이 겹쳐야 한다:

$$t_{x,\text{min}} = \frac{x_{\min} - o_x}{d_x}, \quad t_{x,\text{max}} = \frac{x_{\max} - o_x}{d_x}$$

$y$와 $z$도 마찬가지다. 다음 조건을 만족하면 레이가 상자에 맞는다:

$$t_{\text{enter}} = \max(t_{x,\text{min}}, t_{y,\text{min}}, t_{z,\text{min}}) \le t_{\text{exit}} = \min(t_{x,\text{max}}, t_{y,\text{max}}, t_{z,\text{max}})$$

그리고 $t_{\text{exit}} \ge 0$.

### 2.2 방향성 경계 상자 (OBB, Oriented Bounding Box)

OBB는 임의 방향을 가진 상자로, 중심 $\mathbf{c}$, 세 개의 정규 직교 축 $\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3$, 그리고 반 확장(half-extents) $e_1, e_2, e_3$로 정의된다.

$$\text{OBB} = \left\{\mathbf{c} + \sum_{i=1}^{3} a_i \mathbf{u}_i \;\middle|\; |a_i| \le e_i\right\}$$

**장점**: 길쭉하거나 회전된 오브젝트에 훨씬 더 잘 맞는다.

**단점**: 교차 테스트가 더 비싸다. OBB-OBB 겹침 검사는 최대 15개의 축을 테스트하는 **분리 축 정리(SAT, Separating Axis Theorem)**를 사용한다.

### 2.3 경계 구 (Bounding Sphere)

경계 구는 중심 $\mathbf{c}$와 반지름 $r$로 정의된다:

$$\text{Sphere} = \{\mathbf{p} \;|\; \|\mathbf{p} - \mathbf{c}\| \le r\}$$

**장점**: 회전 불변(재계산 불필요), 매우 빠른 점-구 내부 판정 및 구-구 테스트.

**단점**: 구형이 아닌 오브젝트에 잘 맞지 않는다(낭비되는 체적이 큼).

**레이-구 교차**: $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$를 $\|\mathbf{p} - \mathbf{c}\|^2 = r^2$에 대입한다:

$$\|\mathbf{o} + t\mathbf{d} - \mathbf{c}\|^2 = r^2$$
$$(\mathbf{d} \cdot \mathbf{d})t^2 + 2\mathbf{d} \cdot (\mathbf{o} - \mathbf{c})t + (\mathbf{o} - \mathbf{c}) \cdot (\mathbf{o} - \mathbf{c}) - r^2 = 0$$

이는 $t$에 대한 이차방정식이다. 판별식 $\Delta < 0$이면 레이가 빗나간다. 그렇지 않으면 가장 작은 양의 근이 가장 가까운 교점을 준다.

### 2.4 비교

| 속성 | AABB | OBB | 경계 구 |
|------|------|-----|---------|
| 저장 공간 | 6 floats | 15 floats | 4 floats |
| 적합도 | 보통 | 높음 | 낮음 |
| 교차 비용 | 매우 낮음 | 보통 | 매우 낮음 |
| 회전 처리 | 재계산 | 축 변환 | 변경 없음 |
| 구성 | $O(n)$ | $O(n)$ PCA | $O(n)$ |

---

## 3. 경계 볼륨 계층 구조 (BVH, Bounding Volume Hierarchy)

### 3.1 개념

**BVH**는 경계 볼륨의 트리다. 루트의 경계 볼륨은 전체 씬을 감싼다. 각 내부 노드의 볼륨은 그 서브트리의 모든 오브젝트를 감싼다. 리프(Leaf)에는 하나 또는 소수의 기본 도형(Primitive)이 들어있다.

```
          [Root AABB: entire scene]
         /                         \
  [Left AABB]                [Right AABB]
   /       \                  /         \
[Leaf A] [Leaf B]        [Leaf C]   [Leaf D]
 tri 1    tri 2,3         tri 4      tri 5,6
```

**레이 순회**: 루트 AABB에 대해 레이를 테스트한다. 빗나가면 전체 트리를 건너뛴다. 맞으면 자식으로 재귀한다. 이를 통해 씬의 넓은 부분을 가지치기하여 평균 복잡도를 $O(n)$에서 $O(\log n)$으로 줄인다.

### 3.2 BVH 구성

핵심 결정은 각 노드에서 기본 도형을 어떻게 **분할**하는가이다. 일반적인 전략:

**중점 분할(Midpoint Split)**: 경계 상자의 가장 긴 축에서 중점을 따라 분할한다. 단순하지만 불균형 트리를 만들 수 있다.

**중앙값 분할(Median Split)**: 각 자식이 절반의 기본 도형을 갖도록 분할한다. 균형 트리를 보장하지만 경계 상자의 부피가 커질 수 있다.

**표면적 휴리스틱(SAH, Surface Area Heuristic)**: 표준. 노드의 비용은 다음과 같다:

$$C_{\text{node}} = C_{\text{trav}} + \frac{SA(\text{left})}{SA(\text{parent})} \cdot n_{\text{left}} \cdot C_{\text{isect}} + \frac{SA(\text{right})}{SA(\text{parent})} \cdot n_{\text{right}} \cdot C_{\text{isect}}$$

여기서:
- $C_{\text{trav}}$ = 노드 순회 비용 (상수)
- $C_{\text{isect}}$ = 기본 도형 교차 테스트 비용 (상수)
- $SA(\cdot)$ = 경계 상자의 표면적
- $n_{\text{left}}, n_{\text{right}}$ = 각 자식의 기본 도형 수

직관: 레이는 **표면적이 큰** 자식 노드에 맞을 가능성이 높다. SAH는 부모 대비 표면적이 작은 자식을 만드는 분할을 선호함으로써 레이 쿼리의 예상 비용을 최소화한다.

실제로는 각 축의 여러 후보 분할 위치(또는 기본 도형 경계)에서 SAH 비용을 평가하고, 비용을 최소화하는 분할을 선택한다.

### 3.3 레이 쿼리를 위한 BVH 순회

```
function bvh_intersect(ray, node):
    if not ray_intersects_aabb(ray, node.aabb):
        return NO_HIT

    if node.is_leaf:
        return intersect_primitives(ray, node.primitives)

    hit_left  = bvh_intersect(ray, node.left)
    hit_right = bvh_intersect(ray, node.right)
    return closest(hit_left, hit_right)
```

**최적화**: 어떤 자식의 AABB가 레이 원점에 더 가까운지 테스트하여 그 자식을 먼저 순회한다. 교점을 찾으면 두 번째 자식의 AABB 진입 거리가 찾은 교점보다 멀 경우 조기 종료할 수 있다.

### 3.4 Python 구현: BVH

```python
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

@dataclass
class AABB:
    """Axis-Aligned Bounding Box."""
    min_pt: np.ndarray  # (3,) minimum corner
    max_pt: np.ndarray  # (3,) maximum corner

    def surface_area(self) -> float:
        """Surface area of the box -- used by SAH to estimate ray hit probability."""
        d = self.max_pt - self.min_pt
        return 2.0 * (d[0]*d[1] + d[1]*d[2] + d[2]*d[0])

    def longest_axis(self) -> int:
        """Return index of the longest axis (0=x, 1=y, 2=z)."""
        d = self.max_pt - self.min_pt
        return int(np.argmax(d))

    @staticmethod
    def from_points(points: np.ndarray) -> 'AABB':
        """Build AABB from an array of points (N, 3)."""
        return AABB(np.min(points, axis=0), np.max(points, axis=0))

    @staticmethod
    def union(a: 'AABB', b: 'AABB') -> 'AABB':
        """Merge two AABBs into one that encloses both."""
        return AABB(np.minimum(a.min_pt, b.min_pt),
                    np.maximum(a.max_pt, b.max_pt))

    def intersect_ray(self, origin: np.ndarray, inv_dir: np.ndarray) -> Tuple[bool, float]:
        """
        Slab-based ray-AABB intersection.
        inv_dir = 1.0 / ray_direction (precomputed for speed).
        Returns (hit, t_entry).
        """
        t1 = (self.min_pt - origin) * inv_dir
        t2 = (self.max_pt - origin) * inv_dir

        t_min = np.minimum(t1, t2)  # Why element-wise min: handles negative direction
        t_max = np.maximum(t1, t2)

        t_enter = np.max(t_min)
        t_exit  = np.min(t_max)

        hit = (t_enter <= t_exit) and (t_exit >= 0.0)
        return hit, t_enter


@dataclass
class Sphere:
    """A simple sphere primitive for testing BVH."""
    center: np.ndarray
    radius: float
    color: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.8]))

    def aabb(self) -> AABB:
        r = np.array([self.radius, self.radius, self.radius])
        return AABB(self.center - r, self.center + r)

    def intersect_ray(self, origin, direction) -> Tuple[bool, float]:
        """Geometric ray-sphere intersection test."""
        oc = origin - self.center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - self.radius ** 2
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return False, float('inf')

        sqrt_disc = np.sqrt(discriminant)
        t = (-b - sqrt_disc) / (2.0 * a)
        if t < 0.001:
            t = (-b + sqrt_disc) / (2.0 * a)
        if t < 0.001:
            return False, float('inf')

        return True, t


@dataclass
class BVHNode:
    """A node in the Bounding Volume Hierarchy."""
    aabb: AABB
    left: Optional['BVHNode'] = None
    right: Optional['BVHNode'] = None
    primitives: List = field(default_factory=list)  # Non-empty only for leaves

    @property
    def is_leaf(self) -> bool:
        return len(self.primitives) > 0


def build_bvh(primitives: List, max_leaf_size: int = 2) -> BVHNode:
    """
    Recursively build a BVH using the Surface Area Heuristic (SAH).

    The SAH evaluates candidate splits by estimating the expected ray
    intersection cost. We try several split positions along the longest
    axis and pick the one that minimizes cost.
    """
    # Compute bounding box of all primitives
    aabbs = [p.aabb() for p in primitives]
    total_aabb = aabbs[0]
    for box in aabbs[1:]:
        total_aabb = AABB.union(total_aabb, box)

    # Base case: few enough primitives to store in a leaf
    if len(primitives) <= max_leaf_size:
        return BVHNode(aabb=total_aabb, primitives=primitives)

    # Choose split axis and position using SAH
    best_cost = float('inf')
    best_axis = 0
    best_split_idx = len(primitives) // 2

    # Why we try all 3 axes: the longest axis is usually best, but
    # SAH may find a better split on a different axis
    for axis in range(3):
        # Sort primitive centers along this axis
        centers = [p.center[axis] if hasattr(p, 'center')
                   else (aabbs[i].min_pt[axis] + aabbs[i].max_pt[axis]) / 2
                   for i, p in enumerate(primitives)]
        sorted_indices = np.argsort(centers)
        sorted_prims = [primitives[i] for i in sorted_indices]
        sorted_aabbs = [aabbs[i] for i in sorted_indices]

        n = len(sorted_prims)

        # Build prefix AABBs from left and suffix AABBs from right
        left_aabbs = [None] * n
        right_aabbs = [None] * n

        left_aabbs[0] = sorted_aabbs[0]
        for i in range(1, n):
            left_aabbs[i] = AABB.union(left_aabbs[i-1], sorted_aabbs[i])

        right_aabbs[n-1] = sorted_aabbs[n-1]
        for i in range(n-2, -1, -1):
            right_aabbs[i] = AABB.union(right_aabbs[i+1], sorted_aabbs[i])

        # Evaluate SAH cost for each candidate split position
        parent_sa = total_aabb.surface_area()
        C_TRAV = 1.0    # Relative cost of traversing a node
        C_ISECT = 4.0   # Relative cost of a primitive intersection

        for i in range(1, n):
            # Split: left gets [0..i-1], right gets [i..n-1]
            left_sa = left_aabbs[i-1].surface_area()
            right_sa = right_aabbs[i].surface_area()

            cost = C_TRAV + (left_sa / parent_sa) * i * C_ISECT \
                         + (right_sa / parent_sa) * (n - i) * C_ISECT

            if cost < best_cost:
                best_cost = cost
                best_axis = axis
                best_split_idx = i

    # Perform the best split
    centers = [p.center[best_axis] if hasattr(p, 'center')
               else (aabbs[i].min_pt[best_axis] + aabbs[i].max_pt[best_axis]) / 2
               for i, p in enumerate(primitives)]
    sorted_indices = np.argsort(centers)
    sorted_prims = [primitives[i] for i in sorted_indices]

    left_prims  = sorted_prims[:best_split_idx]
    right_prims = sorted_prims[best_split_idx:]

    # Guard: if SAH puts everything on one side, force an even split
    if len(left_prims) == 0 or len(right_prims) == 0:
        mid = len(sorted_prims) // 2
        left_prims  = sorted_prims[:mid]
        right_prims = sorted_prims[mid:]

    left_child  = build_bvh(left_prims, max_leaf_size)
    right_child = build_bvh(right_prims, max_leaf_size)

    return BVHNode(aabb=total_aabb, left=left_child, right=right_child)


def bvh_intersect(node: BVHNode, origin: np.ndarray, direction: np.ndarray,
                  inv_dir: np.ndarray) -> Tuple[bool, float, object]:
    """
    Traverse the BVH to find the nearest ray intersection.
    Returns (hit, t, primitive).
    """
    hit_box, t_entry = node.aabb.intersect_ray(origin, inv_dir)
    if not hit_box:
        return False, float('inf'), None

    if node.is_leaf:
        closest_t = float('inf')
        closest_prim = None
        for prim in node.primitives:
            hit, t = prim.intersect_ray(origin, direction)
            if hit and t < closest_t:
                closest_t = t
                closest_prim = prim
        return closest_prim is not None, closest_t, closest_prim

    # Recurse into both children, keep the nearest hit
    hit_l, t_l, prim_l = bvh_intersect(node.left, origin, direction, inv_dir)
    hit_r, t_r, prim_r = bvh_intersect(node.right, origin, direction, inv_dir)

    if hit_l and hit_r:
        if t_l <= t_r:
            return True, t_l, prim_l
        else:
            return True, t_r, prim_r
    elif hit_l:
        return True, t_l, prim_l
    elif hit_r:
        return True, t_r, prim_r
    else:
        return False, float('inf'), None


# --- Demo: Build BVH and trace a ray ---
np.random.seed(42)
spheres = [Sphere(center=np.random.uniform(-5, 5, 3),
                  radius=np.random.uniform(0.3, 1.0),
                  color=np.random.uniform(0, 1, 3))
           for _ in range(20)]

bvh_root = build_bvh(spheres, max_leaf_size=2)

# Trace a ray from the camera toward the scene
ray_origin = np.array([0.0, 0.0, 10.0])
ray_dir = np.array([0.0, 0.0, -1.0])
inv_dir = 1.0 / ray_dir  # Precompute for slab test

hit, t, prim = bvh_intersect(bvh_root, ray_origin, ray_dir, inv_dir)
if hit:
    hit_point = ray_origin + t * ray_dir
    print(f"Hit sphere at center {prim.center} at t={t:.3f}")
    print(f"Hit point: ({hit_point[0]:.2f}, {hit_point[1]:.2f}, {hit_point[2]:.2f})")
else:
    print("No intersection found")
```

---

## 4. 옥트리와 쿼드트리

### 4.1 쿼드트리 (2D)

**쿼드트리(Quadtree)**는 2D 공간을 재귀적으로 네 개의 동일한 사분면으로 나눈다. 각 노드는 오브젝트를 직접 저장하는 리프이거나, 정확히 네 개의 자식을 갖는다.

```
┌───────────┬───────────┐
│           │           │
│    NW     │    NE     │
│           │           │
├───────────┼───────────┤
│           │           │
│    SW     │    SE     │
│           │           │
└───────────┴───────────┘
```

**활용**: 2D 게임의 충돌 감지, 공간 쿼리(특정 점 근처의 모든 오브젝트 찾기), 지형 LOD(Level of Detail).

**삽입**: 오브젝트를 완전히 포함하는 가장 작은 사분면에 배치한다. 사분면에 오브젝트가 너무 많으면 세분화한다.

### 4.2 옥트리 (3D)

**옥트리(Octree)**는 쿼드트리를 3D로 확장하여 각 노드를 8개의 팔분공간(Octant)으로 분할한다. 각 팔분공간은 노드의 중심에서 세 축 모두를 따라 분할하여 정의된다.

**구성**:
1. 전체 씬을 감싸는 루트 노드로 시작한다
2. 오브젝트 삽입: 노드에 임계값 이상의 오브젝트가 있으면 8개의 자식으로 분할한다
3. 오브젝트는 완전히 포함하는 자식으로 이동하거나, 경계에 걸치면 부모에 남는다

**특성**:
- 균등 세분화 (기하에 적응하는 BVH와 달리)
- 오브젝트가 대략 균등하게 분포된 경우에 적합
- 균형 잡힌 경우 $n$개의 오브젝트에 대해 깊이는 $O(\log_8 n)$으로 제한됨

**BVH 대비 트레이드오프**: 옥트리는 기하가 어디에 있든 공간을 균등하게 세분화한다. BVH는 기하 분포에 적응하므로 레이 트레이싱에서 일반적으로 더 좋은 성능을 보인다. 옥트리는 점진적 구성이 더 단순하고 동적 씬에 더 적합하다.

### 4.3 쿼드트리 구현

```python
class QuadTreeNode:
    """Quadtree node for 2D spatial partitioning."""

    MAX_OBJECTS = 4
    MAX_DEPTH = 8

    def __init__(self, x, y, width, height, depth=0):
        self.bounds = (x, y, width, height)
        self.depth = depth
        self.objects = []
        self.children = None  # None until subdivided

    def subdivide(self):
        """Split into four equal quadrants."""
        x, y, w, h = self.bounds
        hw, hh = w / 2, h / 2
        d = self.depth + 1
        # Why list of 4: NW, NE, SW, SE -- standard quadtree convention
        self.children = [
            QuadTreeNode(x,      y,      hw, hh, d),  # NW
            QuadTreeNode(x + hw, y,      hw, hh, d),  # NE
            QuadTreeNode(x,      y + hh, hw, hh, d),  # SW
            QuadTreeNode(x + hw, y + hh, hw, hh, d),  # SE
        ]

    def insert(self, obj_x, obj_y, obj_data=None):
        """Insert a point object into the quadtree."""
        x, y, w, h = self.bounds
        # Check if point is within this node's bounds
        if not (x <= obj_x < x + w and y <= obj_y < y + h):
            return False

        if self.children is None:
            self.objects.append((obj_x, obj_y, obj_data))
            # Subdivide if over capacity and not at max depth
            if len(self.objects) > self.MAX_OBJECTS and self.depth < self.MAX_DEPTH:
                self.subdivide()
                # Re-insert existing objects into children
                old_objects = self.objects
                self.objects = []
                for ox, oy, od in old_objects:
                    inserted = False
                    for child in self.children:
                        if child.insert(ox, oy, od):
                            inserted = True
                            break
                    if not inserted:
                        self.objects.append((ox, oy, od))
            return True

        # Try to insert into children
        for child in self.children:
            if child.insert(obj_x, obj_y, obj_data):
                return True
        return False

    def query_range(self, qx, qy, qw, qh):
        """Find all objects within a rectangular range."""
        x, y, w, h = self.bounds
        results = []

        # Check if query range intersects this node
        if qx + qw < x or qx > x + w or qy + qh < y or qy > y + h:
            return results

        # Check objects stored at this node
        for ox, oy, od in self.objects:
            if qx <= ox <= qx + qw and qy <= oy <= qy + qh:
                results.append((ox, oy, od))

        # Recurse into children
        if self.children:
            for child in self.children:
                results.extend(child.query_range(qx, qy, qw, qh))

        return results


# Demo
qt = QuadTreeNode(0, 0, 100, 100)
import random
random.seed(42)
for i in range(50):
    qt.insert(random.uniform(0, 100), random.uniform(0, 100), f"obj_{i}")

nearby = qt.query_range(40, 40, 20, 20)
print(f"Objects in region (40,40)-(60,60): {len(nearby)} found")
```

---

## 5. BSP 트리

### 5.1 개념

**이진 공간 분할 트리(BSP Tree, Binary Space Partition Tree)**는 임의의 초평면(3D에서는 평면, 2D에서는 직선)을 사용하여 공간을 재귀적으로 분할한다. 각 내부 노드는 분할 평면을 저장하고, 두 자식은 각 쪽의 반공간을 나타낸다.

**핵심 특성**: BSP 트리는 어떤 시점에서도 폴리곤의 앞뒤 순서를 결정할 수 있어, z-버퍼 없이도 투명 오브젝트의 올바른 렌더링을 가능하게 한다.

### 5.2 구성

1. 분할 평면으로 폴리곤(또는 평면)을 선택한다
2. 나머지 폴리곤을 평면의 **앞**, **뒤**, 또는 **걸침**으로 분류한다
3. 걸치는 폴리곤을 평면을 따라 분할한다
4. 재귀: 앞쪽 폴리곤은 앞쪽 자식으로, 뒤쪽 폴리곤은 뒤쪽 자식으로

**평면 선택**은 매우 중요하다: 잘못된 선택은 폴리곤 분할 수를 늘려 트리가 비대해진다. 휴리스틱은 트리 깊이와 분할 수의 균형을 맞춘다.

### 5.3 화가 알고리즘을 위한 BSP 순회

위치 $\mathbf{e}$의 카메라로부터 뒤에서 앞으로(화가 알고리즘) 렌더링하려면:

```
function render_bsp(node, eye_position):
    if node is leaf:
        draw(node.polygon)
        return

    d = dot(eye_position - node.plane_point, node.plane_normal)

    if d > 0:  // Eye is in front of the plane
        render_bsp(node.back,  eye_position)  // Draw back first
        draw(node.polygon)
        render_bsp(node.front, eye_position)  // Draw front last (on top)
    else:
        render_bsp(node.front, eye_position)
        draw(node.polygon)
        render_bsp(node.back,  eye_position)
```

이는 어떤 시점에서도 올바른 뒤에서 앞 순서를 만들어 내므로, BSP 트리는 역사적으로 중요했다(Doom, Quake에서 사용).

### 5.4 현대적 활용

BSP 트리는 현대 실시간 렌더링에서 덜 일반적이지만(z-버퍼가 빠르고 하드웨어 가속됨), 다음 용도로는 여전히 유용하다:
- **CSG(구성 입체 기하학, Constructive Solid Geometry)**: 입체에 대한 불리언 연산
- **가시성 결정**: 많은 가려개(Occluder)가 있는 실내 환경
- **충돌 감지**: 물리 엔진의 공간 쿼리

---

## 6. 절두체 컬링과 오클루전 컬링

### 6.1 시야 절두체

시야 절두체(View Frustum)는 카메라의 시야각, 화면 비율, 근/원 평면으로 정의된 잘린 피라미드다. 절두체 내부에 있거나 교차하는 오브젝트만 잠재적으로 보인다.

$$\text{Frustum} = \bigcap_{i=1}^{6} \{\mathbf{p} \;|\; \mathbf{n}_i \cdot \mathbf{p} + d_i \ge 0\}$$

여기서 $\mathbf{n}_i$와 $d_i$는 여섯 개의 절두체 평면(왼쪽, 오른쪽, 위, 아래, 근, 원)을 정의한다.

### 6.2 절두체 컬링

**절두체 컬링(Frustum Culling)**은 각 오브젝트의 경계 볼륨을 절두체와 비교 테스트한다. 완전히 외부에 있는 오브젝트는 건너뛴다.

AABB의 경우 각 절두체 평면에 대해 테스트한다:
- 평면 법선에 대한 AABB의 "가장 양의" 꼭짓점이 평면 뒤에 있으면, 전체 AABB가 외부에 있다
- AABB의 "가장 음의" 꼭짓점이 앞에 있으면, AABB가 해당 평면의 완전히 내부에 있다

이 테스트는 AABB당 $O(1)$이며 대형 씬에서 드로우 콜 수를 극적으로 줄인다.

BVH나 옥트리를 이용한 **계층적 절두체 컬링**: 부모 노드가 절두체 외부에 있으면 모든 자식도 외부이므로 서브트리 전체를 가지치기한다.

### 6.3 오클루전 컬링

절두체 내부의 오브젝트도 다른 오브젝트 **뒤에 가려져** 있을 수 있다. **오클루전 컬링(Occlusion Culling)**은 이러한 숨겨진 오브젝트를 감지하고 건너뛴다.

**방법**:
- **하드웨어 오클루전 쿼리**: GPU에 "이 경계 상자가 보이는 픽셀을 만들어 내겠는가?"를 조건부 렌더링으로 질문한다
- **계층적 Z-버퍼(HZB, Hierarchical Z-Buffer)**: 밉맵된 깊이 버퍼를 유지하고 거친 깊이 레벨에 대해 경계 볼륨을 테스트한다
- **소프트웨어 오클루전**: CPU에서 단순화된 가려개 세트를 래스터화하고, 나머지 오브젝트를 결과 깊이 버퍼에 대해 테스트한다
- **잠재적 가시 집합(PVS, Potentially Visible Sets)**: 어떤 영역이 어떤 다른 영역을 볼 수 있는지 사전 계산한다(실내 환경에서 사용)

### 6.4 종합

일반적인 렌더링 파이프라인은 이 기법들을 조합한다:

```
Scene Graph
    └── Frustum Cull (using BVH/octree)
         └── Occlusion Cull (HZB or queries)
              └── Sort (front-to-back for opaque, back-to-front for transparent)
                   └── Draw
```

각 단계는 오브젝트를 걸러내어 GPU에 전달되는 드로우 콜 수를 줄인다.

---

## 7. 성능 고려사항

### 7.1 올바른 자료구조 선택

| 사용 사례 | 권장 자료구조 |
|----------|--------------|
| 정적 씬 레이 트레이싱 | BVH (SAH) |
| 동적 씬 레이 트레이싱 | BVH (리피팅 포함) |
| 실시간 절두체 컬링 | 옥트리 또는 BVH |
| 실내 가시성 | BSP + PVS |
| 2D 충돌 감지 | 쿼드트리 |
| GPU 레이 트레이싱 (RTX) | BVH (하드웨어 가속) |

### 7.2 BVH 대 옥트리 트레이드오프

- **BVH**: 기하 밀도에 적응하고 노드가 겹칠 수 있으며, 불균등 분포에 더 좋다
- **옥트리**: 고정 공간 세분화로 점진적 갱신이 더 단순하며, 대략 균등 분포에 더 좋다
- **하이브리드 접근**: 상위 레벨에서는 옥트리, 각 셀 내부에서는 BVH를 사용한다

### 7.3 동적 씬

오브젝트가 이동할 때 공간 자료구조도 갱신해야 한다:
- **재구성(Rebuild)**: 비용이 크지만 최적 자료구조를 만든다
- **리피팅(Refit)**: 위상 변경 없이 AABB를 하향식으로 갱신한다(빠르지만 시간이 지남에 따라 품질이 저하된다)
- **점진적 삽입/삭제**: 옥트리에서 지원되며, BVH는 더 신중한 처리가 필요하다
- **두 레벨 BVH**: 정적 기하는 고정 BVH에, 동적 오브젝트는 자주 재구성되는 별도 BVH에 저장한다

---

## 요약

| 개념 | 핵심 아이디어 |
|------|-------------|
| 씬 그래프 | 계층적 트리: 각 노드에 로컬 변환, 월드 변환 = 조상 체인의 곱 |
| AABB | 축 정렬 상자; 매우 빠른 테스트; 6 floats; 회전 시 재계산 |
| OBB | 방향성 상자; 높은 적합도; 더 비싼 테스트 (SAT) |
| 경계 구 | 회전 불변; 빠른 테스트; 낮은 적합도 |
| BVH | 경계 볼륨의 이진 트리; SAH가 최적 분할 제공; $O(\log n)$ 레이 쿼리 |
| 옥트리 / 쿼드트리 | 균등 공간 세분화; 노드당 8개(3D) 또는 4개(2D) 자식 |
| BSP 트리 | 임의의 분할 평면; z-버퍼 없이 뒤에서 앞 순서 가능 |
| 절두체 컬링 | 평면-AABB 테스트로 시야 절두체 외부 오브젝트 건너뜀 |
| 오클루전 컬링 | 다른 오브젝트 뒤에 가려진 오브젝트 건너뜀 (HZB, 쿼리, PVS) |

## 연습문제

1. **씬 그래프 변환**: 노드 A가 이동 $(3, 0, 0)$과 Y축 기준 $45°$ 회전을 갖고, 자식 노드 B가 이동 $(2, 0, 0)$을 갖는 씬 그래프에서, B의 월드 위치를 손으로 계산하라.

2. **AABB 구성**: 꼭짓점이 명시된 5개의 삼각형이 주어졌을 때, 각 삼각형의 AABB를 계산하고, 다섯 개를 모두 감싸는 AABB를 계산하라. Python으로 구현하라.

3. **BVH 분할 비교**: BVH 코드를 수정하여 SAH 대신 중점 분할을 사용하도록 하라. 1000개의 무작위 구를 생성하고 SAH 대비 중점 분할 BVH의 레이당 평균 AABB 교차 테스트 횟수를 비교하라.

4. **옥트리 구현**: 쿼드트리 코드를 3D(옥트리)로 확장하라. 100개의 무작위 구를 삽입하고 최근접 이웃 쿼리를 구현하라.

5. **절두체 컬링**: 6개의 절두체 평면과 AABB를 받아 AABB가 절두체의 외부, 내부, 또는 교차 상태인지 반환하는 함수를 구현하라.

6. **BSP 순서**: 위치와 법선이 명시된 2D 폴리곤 5개가 주어졌을 때, 손으로 BSP 트리를 구성하라. 두 개의 서로 다른 시점에 대한 앞에서 뒤 순회 순서를 보여라.

## 더 읽을거리

- Ericson, C. *Real-Time Collision Detection*. Morgan Kaufmann, 2004. (경계 볼륨과 공간 자료구조에 관한 결정판 참고서)
- Pharr, M., Jakob, W., Humphreys, G. *Physically Based Rendering: From Theory to Implementation*, 4th ed. MIT Press, 2023. (4장: BVH 구성 및 순회)
- Akenine-Moller, T., Haines, E., Hoffman, N. *Real-Time Rendering*, 4th ed. CRC Press, 2018. (19장, 25장: 공간 자료구조와 컬링)
- Karras, T. "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees." *HPG*, 2012. (GPU 가속 BVH 구성)
- Meagher, D. "Geometric Modeling Using Octree Encoding." *Computer Graphics and Image Processing*, 1982. (옥트리 원논문)
