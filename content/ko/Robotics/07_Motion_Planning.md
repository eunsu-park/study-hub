# 운동 계획(Motion Planning)

[← 이전: 로봇 동역학](06_Robot_Dynamics.md) | [다음: 궤적 계획 →](08_Trajectory_Planning.md)

## 학습 목표

1. 구성 공간(C-space, Configuration Space)을 정의하고, 작업 공간(workspace)의 장애물이 C-space 장애물로 어떻게 매핑되는지 설명한다
2. 인공 퍼텐셜 필드(Artificial Potential Field) 방법을 구현하고 그 지역 최솟값(local minima) 문제를 파악한다
3. 다중 쿼리 계획 시나리오를 위한 확률적 로드맵(PRM, Probabilistic Roadmap)을 구성한다
4. 단일 쿼리 계획을 위한 급속 탐색 랜덤 트리(RRT, Rapidly-exploring Random Tree)를 구축하고, RRT*의 최적성 개선 방식을 이해한다
5. 완전성(completeness), 최적성(optimality), 계산 복잡도 측면에서 플래너들을 비교한다
6. 경로 단축(path shortcutting) 및 경로 스무딩(path smoothing)을 포함한 실용적인 후처리 기법을 적용한다

---

## 왜 중요한가

기구학(Kinematics)과 동역학(Dynamics)은 "로봇이 이 자세에 도달할 수 있는가?"와 "어떤 토크가 필요한가?"를 다룬다. 하지만 *아무것도 부딪히지 않고 어떻게 그곳에 도달하는가*에 대해서는 아무것도 말하지 않는다. 운동 계획(Motion Planning)은 시작 구성(start configuration)에서 목표 구성(goal configuration)까지 충돌 없는 경로를 찾는 문제다 — 고수준 작업 명령("컵을 집어라")과 저수준 관절 궤적 사이의 다리 역할을 한다.

구조화된 장애물 없는 환경에서 운동 계획은 사소한 문제다. 그러나 로봇이 사람, 가구, 다른 로봇, 그리고 자기 자신의 몸과 공간을 공유하는 현실 세계에서는 로보틱스에서 가장 어려운 계산 문제 중 하나다. 6-자유도 로봇의 구성 공간은 6차원이며, 이 공간에서 가능한 모든 경로를 확인하는 것은 계산적으로 불가능하다. 이 레슨에서 배우는 샘플링 기반 방법(PRM, RRT, RRT*)은 공간을 지능적으로 탐색함으로써 — 전수 탐색 없이 — 이 문제를 실용적으로 만든다.

> **비유**: RRT는 마치 그래플링 훅을 무작위로 던져 미로를 탐색하는 것과 같다 — 각 던짐은 탐색되지 않은 영역으로 도달 범위를 넓힌다. 미로의 완전한 지도가 필요하지 않다. 시작점을 출구와 연결하기에 충분한 무작위 "던짐"만 있으면 된다.

---

## 구성 공간(C-Space)

### 작업 공간에서 C-Space로

**작업 공간(workspace)**은 로봇과 장애물이 존재하는 물리적 3D 공간이다. **구성 공간(C-space)**은 가능한 모든 로봇 구성의 공간 — C-space의 각 점이 모든 관절의 상태를 완전히 지정한다.

$n$-자유도 로봇의 경우, C-space는 $n$차원이다:
- 2링크 평면 로봇: C-space는 $\mathbb{R}^2$ (또는 관절 각도가 감기는(wrap around) 경우 2-원환면(2-torus) $T^2$)
- 6-자유도 매니퓰레이터: C-space는 $\mathbb{R}^6$
- 평면 위의 이동 로봇: C-space는 $\mathbb{R}^2 \times S^1$ (위치 + 방향)

### C-Space 장애물

작업 공간의 장애물은 **C-space 장애물** — 로봇이 장애물과 충돌하는 모든 구성의 집합 — 이 된다. **자유 C-space** $\mathcal{C}_{free}$는 그 여집합이다: 충돌 없는 모든 구성.

```python
import numpy as np

class PlanarEnvironment:
    """2D workspace with circular obstacles for a 2-link planar robot.

    Why circles? Because collision checking with circles is fast
    (just distance comparison), and any obstacle can be over-approximated
    by a set of circles. For real robots, bounding spheres and mesh
    collision checkers are used, but circles capture the essence.
    """
    def __init__(self, l1, l2, obstacles):
        """
        obstacles: list of (cx, cy, radius) tuples
        """
        self.l1 = l1
        self.l2 = l2
        self.obstacles = obstacles

    def fk(self, q):
        """Forward kinematics: returns joint positions and end-effector."""
        t1, t2 = q
        # Joint 1 position (at origin)
        p0 = np.array([0, 0])
        # Joint 2 position
        p1 = np.array([self.l1 * np.cos(t1),
                        self.l1 * np.sin(t1)])
        # End-effector position
        p2 = p1 + np.array([self.l2 * np.cos(t1 + t2),
                             self.l2 * np.sin(t1 + t2)])
        return p0, p1, p2

    def check_collision(self, q, n_samples=10):
        """Check if configuration q is collision-free.

        Why sample along the links? Because a link is not a point —
        it has length. Even if the joints are clear, the middle of a link
        might pass through an obstacle. We check multiple points along
        each link to catch this.
        """
        p0, p1, p2 = self.fk(q)

        # Sample points along link 1 and link 2
        for t in np.linspace(0, 1, n_samples):
            # Point on link 1
            pt1 = p0 + t * (p1 - p0)
            # Point on link 2
            pt2 = p1 + t * (p2 - p1)

            for cx, cy, r in self.obstacles:
                center = np.array([cx, cy])
                if np.linalg.norm(pt1 - center) < r:
                    return True  # collision
                if np.linalg.norm(pt2 - center) < r:
                    return True  # collision

        return False  # no collision

    def is_free(self, q):
        """Is configuration q in the free C-space?"""
        return not self.check_collision(q)

    def check_path(self, q_start, q_end, resolution=0.05):
        """Check if a straight-line path in C-space is collision-free.

        Why check the path, not just the endpoints? Because two
        collision-free configurations can be connected by a path that
        passes through an obstacle. We discretize the path and check
        intermediate configurations.
        """
        distance = np.linalg.norm(q_end - q_start)
        n_steps = max(int(distance / resolution), 2)

        for t in np.linspace(0, 1, n_steps):
            q = q_start + t * (q_end - q_start)
            if self.check_collision(q):
                return False

        return True


# Create an environment with obstacles
env = PlanarEnvironment(
    l1=1.0, l2=0.8,
    obstacles=[
        (0.8, 0.8, 0.3),   # obstacle 1
        (-0.5, 1.0, 0.25),  # obstacle 2
        (0.3, -0.5, 0.2),   # obstacle 3
    ]
)

# Test some configurations
test_configs = [
    np.radians([0, 0]),
    np.radians([45, 30]),
    np.radians([90, 0]),
    np.radians([30, 90]),
]

for q in test_configs:
    status = "FREE" if env.is_free(q) else "COLLISION"
    _, _, p_ee = env.fk(q)
    print(f"q=({np.degrees(q[0]):>6.1f}, {np.degrees(q[1]):>6.1f}) deg -> "
          f"ee=({p_ee[0]:.2f}, {p_ee[1]:.2f}) : {status}")
```

### C-Space 장애물 시각화

```python
def compute_cspace_map(env, resolution=2.0):
    """Compute a discretized C-space occupancy grid.

    Why visualize C-space? Because planning happens in C-space, not
    workspace. A narrow corridor in workspace might be a wide passage
    in C-space (or vice versa). Understanding the C-space topology
    is essential for choosing the right planner.

    resolution: degrees per grid cell
    """
    n = int(360 / resolution)
    cspace = np.zeros((n, n), dtype=bool)

    for i, t1 in enumerate(np.linspace(-np.pi, np.pi, n)):
        for j, t2 in enumerate(np.linspace(-np.pi, np.pi, n)):
            q = np.array([t1, t2])
            cspace[j, i] = not env.is_free(q)  # True = obstacle

    obstacle_fraction = np.sum(cspace) / cspace.size * 100
    print(f"C-space resolution: {n}x{n} ({n*n} cells)")
    print(f"Obstacle fraction: {obstacle_fraction:.1f}%")

    return cspace

# This would be slow for high resolution; use low resolution for demonstration
cspace = compute_cspace_map(env, resolution=5.0)
```

---

## 퍼텐셜 필드(Potential Field) 방법

### 아이디어

로봇을 인공 퍼텐셜 필드(artificial potential field) 안의 입자로 취급한다:
- **인력 퍼텐셜(Attractive potential)**: 로봇을 목표를 향해 당긴다
- **척력 퍼텐셜(Repulsive potential)**: 로봇을 장애물에서 밀어낸다

로봇은 전체 퍼텐셜의 음의 기울기(gradient descent)를 따른다.

### 수학적 공식화

**인력 퍼텐셜** (이차식):
$$U_{att}(\mathbf{q}) = \frac{1}{2} k_{att} \|\mathbf{q} - \mathbf{q}_{goal}\|^2$$

**척력 퍼텐셜** (역거리):
$$U_{rep}(\mathbf{q}) = \begin{cases} \frac{1}{2} k_{rep} \left(\frac{1}{\rho(\mathbf{q})} - \frac{1}{\rho_0}\right)^2 & \text{if } \rho(\mathbf{q}) \leq \rho_0 \\ 0 & \text{if } \rho(\mathbf{q}) > \rho_0 \end{cases}$$

여기서 $\rho(\mathbf{q})$는 가장 가까운 장애물까지의 거리이고, $\rho_0$는 영향 범위이다.

**전체 퍼텐셜**: $U(\mathbf{q}) = U_{att}(\mathbf{q}) + U_{rep}(\mathbf{q})$

**제어 법칙**: $\dot{\mathbf{q}} = -\nabla U(\mathbf{q})$

```python
class PotentialFieldPlanner:
    """Artificial potential field planner for a point robot in 2D.

    Why start with a point robot? Because the potential field concept
    is clearest without the complexity of robot geometry. For real
    manipulators, we compute workspace distances from each link to
    obstacles and sum the repulsive potentials.
    """
    def __init__(self, obstacles, k_att=1.0, k_rep=100.0, rho_0=0.5):
        """
        obstacles: list of (cx, cy, radius) tuples
        k_att: attractive gain
        k_rep: repulsive gain
        rho_0: obstacle influence distance
        """
        self.obstacles = obstacles
        self.k_att = k_att
        self.k_rep = k_rep
        self.rho_0 = rho_0

    def attractive_force(self, q, q_goal):
        """Negative gradient of attractive potential.

        Why quadratic potential? Because the gradient is linear in distance,
        giving a constant-direction, proportional-magnitude force.
        Alternative: conic potential (constant magnitude, direction only).
        """
        return -self.k_att * (q - q_goal)

    def repulsive_force(self, q):
        """Negative gradient of repulsive potential.

        Why inverse-distance? Because the repulsive force grows
        rapidly near obstacles (preventing collision) but vanishes
        far from obstacles (not interfering with goal-seeking).
        """
        f_rep = np.zeros_like(q)

        for cx, cy, r in self.obstacles:
            center = np.array([cx, cy])
            diff = q[:2] - center  # works for 2D configurations
            dist = np.linalg.norm(diff)
            rho = max(dist - r, 0.01)  # distance to obstacle surface

            if rho < self.rho_0:
                # Magnitude: k_rep * (1/rho - 1/rho_0) * 1/rho^2
                magnitude = self.k_rep * (1.0/rho - 1.0/self.rho_0) / (rho**2)
                # Direction: away from obstacle
                direction = diff / max(dist, 0.01)
                f_rep[:2] += magnitude * direction

        return f_rep

    def plan(self, q_start, q_goal, step_size=0.01, max_steps=5000, tol=0.05):
        """Follow the negative gradient of the total potential.

        Returns a path (list of configurations) from start toward goal.
        May get stuck in a local minimum!
        """
        q = q_start.copy()
        path = [q.copy()]

        for step in range(max_steps):
            if np.linalg.norm(q - q_goal) < tol:
                print(f"  Reached goal in {step} steps")
                return np.array(path), True

            f_att = self.attractive_force(q, q_goal)
            f_rep = self.repulsive_force(q)
            f_total = f_att + f_rep

            # Normalize to prevent huge steps
            f_norm = np.linalg.norm(f_total)
            if f_norm > 1e-10:
                q = q + step_size * f_total / f_norm
            else:
                print(f"  Stuck at local minimum (step {step})")
                return np.array(path), False

            path.append(q.copy())

        print(f"  Max steps reached (distance to goal: "
              f"{np.linalg.norm(q - q_goal):.3f})")
        return np.array(path), False


# Example: navigate around obstacles
planner = PotentialFieldPlanner(
    obstacles=[
        (1.0, 1.0, 0.3),
        (0.5, -0.5, 0.25),
    ],
    k_att=1.0,
    k_rep=50.0,
    rho_0=0.5
)

q_start = np.array([0.0, 0.0])
q_goal = np.array([1.5, 1.5])

print("=== Potential Field Planner ===")
path, success = planner.plan(q_start, q_goal)
print(f"Path length: {len(path)} steps")
if success:
    print(f"Final position: {path[-1]}")
```

### 지역 최솟값(Local Minima) 문제

퍼텐셜 필드의 치명적 약점: 척력과 인력이 목표가 *아닌* 지점에서 균형을 이루어 로봇이 갇히는 **지역 최솟값**이 생길 수 있다.

```python
# Demonstration: local minimum
# Place an obstacle directly between start and goal
planner_trap = PotentialFieldPlanner(
    obstacles=[
        (0.75, 0.75, 0.3),  # obstacle right on the straight-line path
    ],
    k_att=1.0,
    k_rep=100.0,
    rho_0=0.6
)

q_start = np.array([0.0, 0.0])
q_goal = np.array([1.5, 1.5])

print("\n=== Local Minimum Demonstration ===")
path, success = planner_trap.plan(q_start, q_goal, max_steps=10000)
if not success:
    dist = np.linalg.norm(path[-1] - q_goal)
    print(f"Stuck at: {path[-1].round(3)}, distance to goal: {dist:.3f}")
```

**지역 최솟값 해결책**:
1. **무작위 이탈(Random walk escape)**: 갇혔을 때 무작위 교란을 추가
2. **내비게이션 함수(Navigation functions)**: 지역 최솟값이 없는 특수 구성 퍼텐셜 (이론적으로 가능하지만 계산 비용이 높음)
3. **샘플링 기반 플래너 사용** (PRM, RRT) — 지역 최솟값 문제가 없음

---

## 샘플링 기반 플래너(Sampling-Based Planners)

샘플링 기반 플래너는 무작위 샘플링을 통해 충돌 없는 구성의 그래프나 트리를 구축함으로써 지역 최솟값 문제를 피한다. 이는 현대 운동 계획에서 지배적인 패러다임이다.

### 핵심 개념

- **샘플링(Sampling)**: C-space에서 무작위 구성을 생성
- **충돌 검사(Collision checking)**: 구성(또는 경로 세그먼트)이 충돌 없는지 확인
- **최근접 이웃(Nearest neighbor)**: 새로운 샘플에 가장 가까운 기존 노드를 찾음
- **로컬 플래너(Local planner)**: 두 구성을 단순한 경로(보통 C-space에서 직선)로 연결

---

## 확률적 로드맵(PRM, Probabilistic Roadmap)

### 개요

PRM은 **다중 쿼리(multi-query)** 플래너다: 전처리 단계에서 자유 C-space의 로드맵(그래프)을 구축하고, 이후 로드맵에 시작과 목표를 연결함으로써 여러 시작-목표 쿼리에 답한다.

### 알고리즘

**전처리 (로드맵 구축)**:
1. C-space에서 $N$개의 무작위 구성 샘플링
2. 충돌 중인 구성 제거
3. 각 자유 구성에 대해 로컬 플래너를 사용하여 $k$개의 최근접 이웃에 연결 시도
4. 성공한 연결을 로드맵 그래프의 엣지로 추가

**쿼리**:
1. 시작과 목표를 로드맵에 연결
2. 그래프 탐색(A*, Dijkstra)으로 경로 탐색

```python
from collections import defaultdict
import heapq

class PRM:
    """Probabilistic Roadmap planner.

    Why PRM? When you need to answer many queries in the same environment
    (e.g., a robot repeatedly picking objects from different locations
    in the same workspace), PRM amortizes the exploration cost. Build
    the roadmap once, answer many queries fast.

    Limitation: PRM requires the environment to be static during the
    preprocessing phase. Dynamic obstacles require replanning.
    """
    def __init__(self, env, n_samples=500, k_neighbors=10):
        self.env = env
        self.n_samples = n_samples
        self.k_neighbors = k_neighbors
        self.nodes = []
        self.edges = defaultdict(list)  # adjacency list
        self.roadmap_built = False

    def build_roadmap(self):
        """Build the PRM roadmap by sampling and connecting.

        Phase 1: Sample free configurations
        Phase 2: Connect nearby configurations
        """
        print(f"Building PRM roadmap ({self.n_samples} samples, "
              f"k={self.k_neighbors})...")

        # Phase 1: Sampling
        n_free = 0
        n_attempts = 0
        while n_free < self.n_samples:
            q = np.random.uniform(-np.pi, np.pi, 2)
            n_attempts += 1
            if self.env.is_free(q):
                self.nodes.append(q)
                n_free += 1

        print(f"  Sampled {n_free} free configs in {n_attempts} attempts")
        print(f"  Free space fraction: {n_free/n_attempts*100:.1f}%")

        # Phase 2: Connect neighbors
        n_edges = 0
        for i, q_i in enumerate(self.nodes):
            # Find k nearest neighbors
            distances = [(np.linalg.norm(q_i - q_j), j)
                        for j, q_j in enumerate(self.nodes) if j != i]
            distances.sort()
            neighbors = distances[:self.k_neighbors]

            for dist, j in neighbors:
                # Check if edge already exists
                if j not in [e[0] for e in self.edges[i]]:
                    # Try to connect
                    if self.env.check_path(q_i, self.nodes[j]):
                        self.edges[i].append((j, dist))
                        self.edges[j].append((i, dist))
                        n_edges += 1

        print(f"  Created {n_edges} edges")
        self.roadmap_built = True

    def query(self, q_start, q_goal):
        """Find a path from start to goal using the roadmap.

        Why A*? Because it's the optimal graph search algorithm for
        finding shortest paths with an admissible heuristic. The
        Euclidean distance in C-space is admissible (never overestimates).
        """
        if not self.roadmap_built:
            self.build_roadmap()

        # Add start and goal to roadmap temporarily
        start_idx = len(self.nodes)
        goal_idx = start_idx + 1
        temp_nodes = self.nodes + [q_start, q_goal]

        # Connect start and goal to nearest roadmap nodes
        temp_edges = defaultdict(list, {k: list(v) for k, v in self.edges.items()})

        for idx, q in [(start_idx, q_start), (goal_idx, q_goal)]:
            distances = [(np.linalg.norm(q - q_j), j)
                        for j, q_j in enumerate(self.nodes)]
            distances.sort()

            connected = 0
            for dist, j in distances[:self.k_neighbors * 2]:
                if self.env.check_path(q, self.nodes[j]):
                    temp_edges[idx].append((j, dist))
                    temp_edges[j].append((idx, dist))
                    connected += 1
                    if connected >= self.k_neighbors:
                        break

            if connected == 0:
                print(f"  Cannot connect {'start' if idx == start_idx else 'goal'} "
                      f"to roadmap")
                return None

        # A* search
        path = self._astar(temp_edges, temp_nodes, start_idx, goal_idx)

        if path is None:
            print("  No path found in roadmap")
            return None

        # Convert indices to configurations
        config_path = [temp_nodes[i] for i in path]
        total_length = sum(np.linalg.norm(config_path[i+1] - config_path[i])
                          for i in range(len(config_path)-1))
        print(f"  Path found: {len(path)} waypoints, length={total_length:.3f}")

        return np.array(config_path)

    def _astar(self, edges, nodes, start, goal):
        """A* graph search."""
        open_set = [(0, start)]  # (f_score, node)
        came_from = {}
        g_score = {start: 0}
        f_score = {start: np.linalg.norm(nodes[start] - nodes[goal])}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for neighbor, dist in edges[current]:
                tentative_g = g_score[current] + dist

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + np.linalg.norm(nodes[neighbor] - nodes[goal])
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))

        return None  # no path found


# Build and query PRM
print("=== PRM Planner ===")
prm = PRM(env, n_samples=300, k_neighbors=8)
prm.build_roadmap()

q_start = np.radians([-30, 60])
q_goal = np.radians([120, -45])

if env.is_free(q_start) and env.is_free(q_goal):
    path = prm.query(q_start, q_goal)
else:
    print("Start or goal in collision!")
```

---

## 급속 탐색 랜덤 트리(RRT, Rapidly-Exploring Random Tree)

### 개요

RRT는 **단일 쿼리(single-query)** 플래너다: 시작 구성을 루트로 하는 트리를 구축하고, 무작위 탐색을 통해 목표를 향해 성장한다. 가장 널리 사용되는 샘플링 기반 플래너다.

### 알고리즘

1. 시작 구성을 루트로 하여 트리 $T$를 초기화
2. 목표에 도달하거나 타임아웃까지 **반복**:
   a. 무작위 구성 $\mathbf{q}_{rand}$ 샘플링
   b. $T$에서 $\mathbf{q}_{rand}$에 가장 가까운 노드 찾기: $\mathbf{q}_{near}$
   c. $\mathbf{q}_{near}$에서 $\mathbf{q}_{rand}$ 방향으로 스텝 크기 $\delta$만큼 확장: $\mathbf{q}_{new}$
   d. $\mathbf{q}_{near}$에서 $\mathbf{q}_{new}$까지의 경로가 충돌 없으면, $\mathbf{q}_{new}$를 $T$에 추가
   e. $\mathbf{q}_{new}$가 $\mathbf{q}_{goal}$에 충분히 가까우면, 연결하고 경로를 반환
3. 경로를 찾지 못하면 실패 반환

```python
class RRT:
    """Rapidly-exploring Random Tree planner.

    Why RRT? It's the workhorse of modern motion planning:
    - Works in high-dimensional C-spaces (6+ DOF)
    - No local minima (random exploration escapes traps)
    - Biased toward unexplored regions (Voronoi bias)
    - Simple to implement and extend

    The key insight: by always extending toward a random sample from
    the nearest existing node, RRT naturally explores outward from
    dense regions into sparse ones. This 'Voronoi bias' means the tree
    spreads rapidly to fill the free space.
    """
    def __init__(self, env, step_size=0.1, goal_bias=0.1,
                 max_iter=5000, goal_threshold=0.15):
        self.env = env
        self.step_size = step_size
        self.goal_bias = goal_bias      # probability of sampling goal
        self.max_iter = max_iter
        self.goal_threshold = goal_threshold

    def plan(self, q_start, q_goal):
        """Build an RRT from start toward goal.

        Returns path (list of configs) or None if planning fails.
        """
        # Tree: list of (config, parent_index)
        tree = [(q_start.copy(), -1)]

        for iteration in range(self.max_iter):
            # Step 1: Sample (with goal bias)
            if np.random.random() < self.goal_bias:
                q_rand = q_goal.copy()
            else:
                q_rand = np.random.uniform(-np.pi, np.pi, 2)

            # Step 2: Find nearest node
            distances = [np.linalg.norm(q_rand - node[0]) for node in tree]
            nearest_idx = np.argmin(distances)
            q_near = tree[nearest_idx][0]

            # Step 3: Extend toward random sample
            direction = q_rand - q_near
            dist = np.linalg.norm(direction)
            if dist < 1e-10:
                continue

            if dist > self.step_size:
                q_new = q_near + self.step_size * direction / dist
            else:
                q_new = q_rand.copy()

            # Step 4: Collision check
            if not self.env.is_free(q_new):
                continue
            if not self.env.check_path(q_near, q_new):
                continue

            # Add to tree
            tree.append((q_new.copy(), nearest_idx))

            # Step 5: Goal check
            if np.linalg.norm(q_new - q_goal) < self.goal_threshold:
                # Try to connect directly to goal
                if self.env.check_path(q_new, q_goal):
                    tree.append((q_goal.copy(), len(tree) - 1))
                    path = self._extract_path(tree)
                    print(f"  RRT found path in {iteration+1} iterations, "
                          f"{len(tree)} nodes")
                    return path

        print(f"  RRT failed after {self.max_iter} iterations "
              f"({len(tree)} nodes)")
        return None

    def _extract_path(self, tree):
        """Trace back from goal to start through parent pointers."""
        path = []
        idx = len(tree) - 1

        while idx != -1:
            path.append(tree[idx][0])
            idx = tree[idx][1]

        path.reverse()
        return np.array(path)


# Run RRT
print("\n=== RRT Planner ===")
rrt = RRT(env, step_size=0.15, goal_bias=0.1, max_iter=5000)

q_start = np.radians([-30, 60])
q_goal = np.radians([120, -45])

if env.is_free(q_start) and env.is_free(q_goal):
    path_rrt = rrt.plan(q_start, q_goal)
    if path_rrt is not None:
        total_length = sum(np.linalg.norm(path_rrt[i+1] - path_rrt[i])
                          for i in range(len(path_rrt)-1))
        print(f"  Path length: {total_length:.3f} rad")
        print(f"  Waypoints: {len(path_rrt)}")
```

### 목표 편향(Goal Bias)

`goal_bias` 매개변수는 무작위 점 대신 목표를 샘플링할 확률을 제어한다. 너무 낮으면 (예: 0.0): 트리가 균일하게 탐색하지만 목표에 도달하는 데 오래 걸릴 수 있다. 너무 높으면 (예: 0.5): 트리가 목표를 향해 탐욕적으로 성장하여 장애물 뒤에 갇힐 수 있다. 일반적인 값은 0.05에서 0.15이다.

---

## RRT*(최적 RRT)

### RRT의 문제점

기본 RRT는 *어떤* 경로를 찾지만, *최단* 경로는 찾지 못한다. 각 노드가 트리에서 최선의 부모가 아닌 최근접 이웃에만 연결되기 때문에, 경로가 일반적으로 불규칙하고 준최적(suboptimal)이다.

### RRT* 개선 사항

RRT*는 **점근적 최적성(asymptotic optimality)** 을 보장하는 두 가지 연산을 추가한다 — 샘플 수가 늘어남에 따라 경로가 최적 해로 수렴한다:

1. **재연결(Rewiring)**: 새 노드를 추가한 후, 인근 노드들이 새 노드를 거쳐 재라우팅할 경우 이득이 있는지 확인
2. **최선 부모 선택(Choose best parent)**: 새 노드를 단순히 가장 가까운 노드가 아닌, 도달 비용(cost-to-come)이 가장 낮은 인근 노드에 연결

```python
class RRTStar:
    """RRT* — asymptotically optimal variant of RRT.

    Why RRT* over RRT? Because RRT paths can be arbitrarily bad —
    they zigzag, take detours, and waste energy. RRT* paths converge
    to the optimal solution given enough samples. The computational
    overhead is modest (neighbor search + rewiring).

    The key theoretical result (Karaman & Frazzoli, 2011):
    - RRT is NOT asymptotically optimal
    - RRT* IS asymptotically optimal
    - The connection radius should be r = gamma * (log(n)/n)^(1/d)
      where d is the C-space dimension
    """
    def __init__(self, env, step_size=0.15, goal_bias=0.1,
                 max_iter=3000, goal_threshold=0.15, gamma=1.5):
        self.env = env
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.max_iter = max_iter
        self.goal_threshold = goal_threshold
        self.gamma = gamma

    def plan(self, q_start, q_goal):
        """RRT* planning with rewiring."""
        # Tree storage: parallel arrays for efficiency
        nodes = [q_start.copy()]
        parents = [-1]
        costs = [0.0]  # cost from start to each node

        best_goal_idx = None
        best_goal_cost = np.inf

        for iteration in range(self.max_iter):
            # Sample
            if np.random.random() < self.goal_bias:
                q_rand = q_goal.copy()
            else:
                q_rand = np.random.uniform(-np.pi, np.pi, 2)

            # Nearest neighbor
            distances = [np.linalg.norm(q_rand - node) for node in nodes]
            nearest_idx = np.argmin(distances)
            q_near = nodes[nearest_idx]

            # Extend
            direction = q_rand - q_near
            dist = np.linalg.norm(direction)
            if dist < 1e-10:
                continue

            if dist > self.step_size:
                q_new = q_near + self.step_size * direction / dist
            else:
                q_new = q_rand.copy()

            if not self.env.is_free(q_new):
                continue

            # === RRT* specific: choose best parent ===
            n = len(nodes)
            d = len(q_start)
            # Connection radius (from theory)
            r = min(self.gamma * (np.log(n + 1) / (n + 1)) ** (1.0 / d),
                    self.step_size * 3)

            # Find nearby nodes
            nearby = []
            for i, node in enumerate(nodes):
                if np.linalg.norm(q_new - node) < r:
                    nearby.append(i)

            # Choose parent with minimum cost
            best_parent = nearest_idx
            best_cost = costs[nearest_idx] + np.linalg.norm(q_new - q_near)

            for idx in nearby:
                new_cost = costs[idx] + np.linalg.norm(q_new - nodes[idx])
                if new_cost < best_cost:
                    if self.env.check_path(nodes[idx], q_new):
                        best_parent = idx
                        best_cost = new_cost

            # Check if connection to best parent is valid
            if not self.env.check_path(nodes[best_parent], q_new):
                continue

            # Add new node
            new_idx = len(nodes)
            nodes.append(q_new.copy())
            parents.append(best_parent)
            costs.append(best_cost)

            # === RRT* specific: rewire nearby nodes ===
            for idx in nearby:
                rewire_cost = best_cost + np.linalg.norm(q_new - nodes[idx])
                if rewire_cost < costs[idx]:
                    if self.env.check_path(q_new, nodes[idx]):
                        parents[idx] = new_idx
                        costs[idx] = rewire_cost
                        # Propagate cost changes to children
                        self._propagate_cost(nodes, parents, costs, idx)

            # Goal check
            if np.linalg.norm(q_new - q_goal) < self.goal_threshold:
                if self.env.check_path(q_new, q_goal):
                    goal_cost = best_cost + np.linalg.norm(q_new - q_goal)
                    if goal_cost < best_goal_cost:
                        # Add/update goal node
                        if best_goal_idx is None:
                            best_goal_idx = len(nodes)
                            nodes.append(q_goal.copy())
                            parents.append(new_idx)
                            costs.append(goal_cost)
                        else:
                            parents[best_goal_idx] = new_idx
                            costs[best_goal_idx] = goal_cost
                        best_goal_cost = goal_cost

        if best_goal_idx is not None:
            path = self._extract_path(nodes, parents, best_goal_idx)
            print(f"  RRT* found path: cost={best_goal_cost:.3f}, "
                  f"{len(path)} waypoints, {len(nodes)} total nodes")
            return np.array(path)
        else:
            print(f"  RRT* failed after {self.max_iter} iterations")
            return None

    def _propagate_cost(self, nodes, parents, costs, idx):
        """Propagate cost change to all descendants (BFS)."""
        queue = [idx]
        while queue:
            current = queue.pop(0)
            for i, parent in enumerate(parents):
                if parent == current:
                    costs[i] = costs[current] + np.linalg.norm(
                        nodes[i] - nodes[current])
                    queue.append(i)

    def _extract_path(self, nodes, parents, goal_idx):
        path = []
        idx = goal_idx
        while idx != -1:
            path.append(nodes[idx])
            idx = parents[idx]
        path.reverse()
        return path


# Run RRT*
print("\n=== RRT* Planner ===")
rrt_star = RRTStar(env, step_size=0.15, goal_bias=0.1, max_iter=3000)

if env.is_free(q_start) and env.is_free(q_goal):
    path_star = rrt_star.plan(q_start, q_goal)
```

---

## 완전성과 최적성

### 이론적 보장

| 플래너 | 완전한가? | 최적인가? | 복잡도 |
|---------|-----------|----------|------------|
| 퍼텐셜 필드(Potential Field) | 아니오 (지역 최솟값) | 아니오 | 스텝당 $O(1)$ |
| PRM | 확률적 완전 | 점근적 최적 (PRM*) | $O(n \log n)$ |
| RRT | 확률적 완전 | **아니오** | $O(n \log n)$ |
| RRT* | 확률적 완전 | **점근적 최적** | $O(n \log n)$ |

**확률적 완전(Probabilistically complete)**: 해가 존재하면, 샘플 수가 무한에 가까워질수록 해를 찾을 확률이 1에 수렴한다.

**점근적 최적(Asymptotically optimal)**: 샘플 수가 무한에 가까워질수록, 반환된 경로의 비용이 최적 비용으로 수렴한다.

```python
def compare_planners(env, q_start, q_goal, n_trials=5):
    """Compare RRT and RRT* on the same problem.

    Why compare? Because the theoretical guarantees don't tell you
    about practical performance — how many samples are needed, how
    good the path is with a finite budget, and how long it takes.
    """
    print("\n=== Planner Comparison ===")
    print(f"Running {n_trials} trials each...\n")

    rrt_results = []
    rrt_star_results = []

    for trial in range(n_trials):
        np.random.seed(trial * 42)

        # RRT
        rrt = RRT(env, step_size=0.15, goal_bias=0.1, max_iter=5000)
        path = rrt.plan(q_start, q_goal)
        if path is not None:
            length = sum(np.linalg.norm(path[i+1] - path[i])
                        for i in range(len(path)-1))
            rrt_results.append(length)

        # RRT*
        rrt_s = RRTStar(env, step_size=0.15, goal_bias=0.1, max_iter=5000)
        path_s = rrt_s.plan(q_start, q_goal)
        if path_s is not None:
            length_s = sum(np.linalg.norm(path_s[i+1] - path_s[i])
                          for i in range(len(path_s)-1))
            rrt_star_results.append(length_s)

    if rrt_results:
        print(f"\nRRT:  avg length = {np.mean(rrt_results):.3f}, "
              f"std = {np.std(rrt_results):.3f}, "
              f"success = {len(rrt_results)}/{n_trials}")
    if rrt_star_results:
        print(f"RRT*: avg length = {np.mean(rrt_star_results):.3f}, "
              f"std = {np.std(rrt_star_results):.3f}, "
              f"success = {len(rrt_star_results)}/{n_trials}")

# Uncomment to run comparison (takes time):
# compare_planners(env, q_start, q_goal, n_trials=3)
```

---

## 경로 후처리(Path Post-Processing)

샘플링 기반 플래너의 원시 경로는 일반적으로 불규칙하고 준최적이다. 후처리를 통해 경로 품질을 향상시킨다.

### 경로 단축(Shortcutting)

인접하지 않은 경유점(waypoint)을 직접 경로로 연결하는 시도를 반복하여 중간 경유점을 제거한다:

```python
def shortcut_path(env, path, max_attempts=200):
    """Remove unnecessary waypoints by direct connection.

    Why shortcutting? Sampling-based planners create paths that follow
    the tree structure, which is rarely the most direct route. Shortcutting
    is like straightening a tangled rope — you pull the ends and let the
    middle straighten out wherever possible.
    """
    if len(path) < 3:
        return path

    improved_path = list(path)

    for _ in range(max_attempts):
        if len(improved_path) < 3:
            break

        # Pick two random non-adjacent indices
        i = np.random.randint(0, len(improved_path) - 2)
        j = np.random.randint(i + 2, len(improved_path))

        # Try to connect directly
        if env.check_path(improved_path[i], improved_path[j]):
            # Remove intermediate waypoints
            improved_path = improved_path[:i+1] + improved_path[j:]

    return np.array(improved_path)

def smooth_path(path, weight_smooth=0.5, weight_data=0.5, n_iterations=100,
                tolerance=1e-6):
    """Smooth path using gradient descent.

    Minimize: sum of (smoothed - original)^2 + (smoothed[i+1] - smoothed[i])^2

    Why smooth? Even after shortcutting, the path has sharp corners
    at waypoints. Smoothing reduces these corners, producing a path
    that's easier for trajectory planning (Lesson 8) to follow.

    Note: this does NOT check collisions! After smoothing, verify
    the path is still collision-free.
    """
    smoothed = path.copy()
    fixed_start = path[0].copy()
    fixed_end = path[-1].copy()

    for _ in range(n_iterations):
        change = 0
        for i in range(1, len(smoothed) - 1):
            old = smoothed[i].copy()
            smoothed[i] += weight_data * (path[i] - smoothed[i]) + \
                           weight_smooth * (smoothed[i-1] + smoothed[i+1] - 2*smoothed[i])
            change += np.linalg.norm(smoothed[i] - old)

        # Fix endpoints
        smoothed[0] = fixed_start
        smoothed[-1] = fixed_end

        if change < tolerance:
            break

    return smoothed
```

---

## 실용적 고려사항

### 충돌 검사 효율성

충돌 검사가 계획 시간을 지배한다 (종종 계산의 90% 이상). 최적화 방법:
- **경계 볼륨(Bounding volumes)**: 정밀한 메시 검사 전에 경계 구(sphere)/박스 먼저 확인
- **공간 해싱(Spatial hashing)**: 빠른 근접 쿼리를 위해 장애물 조직화
- **지연 평가(Lazy evaluation)**: 충돌 검사를 미루고 실제 사용되는 경로만 검증 (Lazy PRM, Lazy RRT)

### 고차원 C-Space

많은 자유도를 가진 로봇(예: 14+ 자유도의 양팔 조작)에서는 C-space의 부피가 지수적으로 커지기 때문에 샘플링 기반 플래너가 느려진다. 전략:
- **태스크 공간 샘플링(Task-space sampling)**: 모든 관절을 샘플링하는 대신 엔드 이펙터 자세를 샘플링하고 역기구학(IK) 사용
- **제약 다양체 샘플링(Constraint manifold sampling)**: 제약이 있는 작업(컵을 수평으로 유지)에서 샘플을 제약 다양체에 투영
- **정보 기반 샘플링(Informed sampling)**: 현재 최선 경로 비용을 사용하여 타원형 영역에 집중적으로 샘플링 (Informed RRT*)

---

## 요약

- **구성 공간(C-space)**은 운동 계획 문제를 물리 공간에서 관절 공간으로 매핑하며, 각 점이 완전한 로봇 구성을 나타낸다
- **퍼텐셜 필드(Potential field)** 방법은 단순하지만 **지역 최솟값(local minima)** 문제가 있어 로봇이 갇힐 수 있다
- **PRM**은 여러 쿼리에 재사용 가능한 로드맵을 구축한다; 다수의 계획 요청이 있는 정적 환경에 적합
- **RRT**는 시작에서 목표를 향해 트리를 성장시킨다; 고차원 공간에서의 단일 쿼리에 적합
- **RRT***는 재연결(rewiring)과 최선 부모 선택을 추가하여 더 많은 계산 비용으로 **점근적 최적성**을 보장한다
- **후처리** (단축, 스무딩)는 샘플링 기반 플래너의 경로 품질을 크게 향상시킨다
- 실제로 충돌 검사가 계산의 병목이다; 효율적인 자료구조와 지연 평가가 필수적이다

---

## 연습 문제

### 연습 1: C-Space 시각화

$l_1 = 1.0$, $l_2 = 0.8$인 2링크 평면 로봇에 대해:
1. $(0.8, 0.8)$에 반지름 $0.3$의 단일 원형 장애물을 배치한다
2. $2°$ 해상도로 C-space 장애물 맵을 계산한다
3. 자유 C-space를 시각화한다 (연결된 영역과 분리된 영역이 보여야 한다)
4. 물리적 장애물이 $(0, 1.2)$로 이동하면 C-space 장애물 형태는 어떻게 변하는가?

### 연습 2: 퍼텐셜 필드 튜닝

1. 2링크 로봇에 대한 퍼텐셜 필드 플래너를 구현한다 (태스크 공간 좌표 사용)
2. 플래너가 성공하는 시작/목표 쌍을 찾는다
3. 플래너가 지역 최솟값에 갇히는 시작/목표 쌍을 찾는다
4. 무작위 이탈 전략을 구현한다: 기울기 크기가 임계값 아래로 떨어지면 무작위 변위를 추가한다. 이것이 지역 최솟값 문제를 해결하는가?

### 연습 3: RRT 구현

1. C-space에서 2링크 평면 로봇에 대한 RRT를 구현한다
2. 같은 시작/목표 사이에서 계획을 10번 실행한다. 경로 길이의 분산은 얼마인가?
3. 다양한 반복 횟수 (100, 500, 2000)에서 RRT 트리를 그려 탐색을 시각화한다
4. 스텝 크기를 변경하면 성공률과 경로 품질에 어떤 영향을 미치는가?

### 연습 4: RRT vs RRT*

1. 같은 문제에서 $N = 1000, 2000, 5000$ 샘플로 RRT와 RRT* 모두 실행한다
2. 두 알고리즘에 대해 샘플 수 대 경로 비용을 그래프로 그린다
3. RRT* 경로는 샘플이 많아질수록 개선되지만 RRT 경로는 체계적으로 개선되지 않음을 검증한다

### 연습 5: 후처리

1. RRT 경로를 생성한다 (불규칙할 것이다)
2. 경로 단축을 적용하고 경로 길이 감소를 측정한다
3. 단축된 경로에 스무딩을 적용한다
4. 스무딩된 경로가 여전히 충돌 없음을 검증한다
5. 총 경로 길이를 비교한다: 원시 RRT, 단축 후, 스무딩 후

---

[← 이전: 로봇 동역학](06_Robot_Dynamics.md) | [다음: 궤적 계획 →](08_Trajectory_Planning.md)
