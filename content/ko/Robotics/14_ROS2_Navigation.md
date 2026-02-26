# 14. ROS2 내비게이션 스택(Nav2)

[← 이전: ROS2 기초](13_ROS2_Fundamentals.md) | [다음: 로봇공학을 위한 강화학습 →](15_RL_for_Robotics.md)

---

## 학습 목표

1. Nav2(Navigation2) 스택의 전체 아키텍처와 주요 구성 요소를 설명한다
2. 코스트맵(costmap) 레이어(정적, 장애물, 팽창)와 이를 내비게이션 코스트맵으로 합성하는 방법을 이해한다
3. 전역 계획 알고리즘(NavFn, Theta*, Smac 계획기)을 비교한다
4. 지역 계획 접근법(DWB, MPPI, 규제 순수 추종)을 설명한다
5. 강건한 자율 내비게이션을 위해 복구 행동과 행동 트리(behavior tree)를 설정한다
6. 다중 목표 자율 임무를 위한 웨이포인트 추종을 구현한다

---

이전 단원에서 우리는 ROS2의 기초인 노드(node), 토픽(topic), 서비스(service), 액션(action), 그리고 모듈화된 로봇 소프트웨어를 만드는 방법을 배웠다. 이제 그 인프라를 이동 로봇이 가질 수 있는 가장 중요한 능력 중 하나인 자율 내비게이션에 활용해보자. 내비게이션은 이 질문에 답한다: *로봇이 어떻게 A 지점에서 B 지점까지 안전하고, 효율적이며, 안정적으로 이동하는가?*

**Nav2**(Navigation2) 스택은 ROS2의 표준 내비게이션 프레임워크이다. ROS1 내비게이션 스택의 후속이지만 더 모듈화되고 플러그인 기반의 아키텍처를 갖춘 완전한 재작성 버전이다. Nav2는 코스트맵 구성에서 전역 경로 계획, 지역 궤적 추종, 막힌 상황에서의 복구, 다중 웨이포인트 임무까지 모든 것을 처리한다. 창고 로봇부터 야외 배달 차량에 이르기까지 모든 이동 로봇을 배포하려면 Nav2 이해가 필수적이다.

> **비유**: Nav2의 행동 트리(behavior tree)는 조종사의 의사결정 순서도와 같다 — 주요 경로가 막히면 대안을 시도하고, 모든 대안이 실패하면 복구 절차(회전, 후진, 대기)를 실행한다. 조종사가 즉흥적으로 행동하는 것이 아니라 구조화된 프로토콜을 따르듯, Nav2는 모든 상황에서 정확히 무엇을 해야 하는지 정의하는 행동 트리를 따르므로 문제가 발생해도 안전하고 예측 가능한 동작을 보장한다.

---

## 1. Nav2 아키텍처 개요

### 1.1 전체 그림

Nav2는 여러 주요 서버로 구성되며, 각 서버는 라이프사이클 노드로 실행되고 ROS2 액션과 토픽을 통해 통신한다:

```
┌──────────────────────────────────────────────────────────┐
│                    BT Navigator                           │
│            (Behavior Tree orchestrator)                    │
├──────────┬───────────┬───────────┬───────────────────────┤
│ Planner  │ Controller│ Smoother  │ Recovery              │
│ Server   │ Server    │ Server    │ Server                │
│ (global  │ (local    │ (path     │ (spin, backup,        │
│  path)   │  control) │  smooth)  │  wait, clear)         │
├──────────┴───────────┴───────────┴───────────────────────┤
│                    Costmap 2D                             │
│              (Global + Local costmaps)                    │
├──────────────────────────────────────────────────────────┤
│          AMCL / SLAM        │        TF2                  │
│       (Localization)        │   (Transform tree)          │
├──────────────────────────────────────────────────────────┤
│                  Sensor Drivers                           │
│            (LiDAR, cameras, IMU, etc.)                    │
└──────────────────────────────────────────────────────────┘
```

### 1.2 내비게이션 흐름

일반적인 내비게이션 요청은 다음 단계를 거친다:

1. **목표 수신**: 사용자 또는 상위 수준 계획기가 `NavigateToPose` 액션 목표를 전송
2. **전역 계획**: 계획기 서버(Planner Server)가 전역 코스트맵에서 시작점부터 목표점까지의 경로를 계산
3. **경로 스무딩**: 스무더 서버(Smoother Server, 선택 사항)가 부드러움을 위해 경로를 정제
4. **지역 제어**: 컨트롤러 서버(Controller Server)가 지역 코스트맵을 사용하여 20+ Hz로 속도 명령을 계산하면서 경로를 추종
5. **복구**: 로봇이 막히면 복구 행동이 발동됨 (회전, 후진, 대기)
6. **조율**: BT 내비게이터(BT Navigator)가 행동 트리를 사용하여 흐름을 관리

### 1.3 내비게이션의 핵심 프레임

```
map                    Global frame — SLAM or static map
 └── odom              Local frame — odometry (smooth but drifts)
      └── base_link    Robot body frame
           └── ...     Sensor frames
```

**왜 두 개의 프레임인가?** `odom` 프레임은 부드럽고 높은 빈도의 자세 추정값을 제공하지만(휠 엔코더로부터) 시간이 지나면 드리프트가 발생한다. `map` 프레임은 전역적으로 일관된 위치를 제공하지만(SLAM 또는 AMCL로부터) 수정이 발생할 때 불연속적인 점프가 생길 수 있다. 내비게이션은 전역 계획에는 `map`을, 지역 제어에는 `odom`을 사용한다.

---

## 2. 코스트맵(Costmap)

### 2.1 코스트맵이란?

**코스트맵**은 각 셀에 로봇이 통과하기에 얼마나 위험하거나 바람직하지 않은지를 나타내는 비용 값을 가진 2D 격자이다:

| 값 | 의미 | 셀 상태 |
|----|------|---------|
| 0 | 자유 공간(Free space) | 안전하게 통과 가능 |
| 1-252 | 증가하는 비용 | 장애물에 근접 |
| 253 | 내접 장애물(Inscribed obstacle) | 로봇 풋프린트가 장애물에 접촉 |
| 254 | 치명적 장애물(Lethal obstacle) | 장애물이 이 셀을 점유 |
| 255 | 알 수 없음(Unknown) | 이 셀에 대한 정보 없음 |

### 2.2 코스트맵 레이어

Nav2는 **레이어드 코스트맵(layered costmap)** 아키텍처를 사용한다. 각 레이어가 정보를 추가하며, 레이어들이 마스터 코스트맵으로 합성된다:

```
┌───────────────────────┐
│   Master Costmap      │  ← Combined result (max of all layers)
├───────────────────────┤
│   Inflation Layer     │  ← Adds cost gradient around obstacles
├───────────────────────┤
│   Obstacle Layer      │  ← Dynamic obstacles from sensors
├───────────────────────┤
│   Static Layer        │  ← Known map (from SLAM or file)
└───────────────────────┘
```

#### 정적 레이어(Static Layer)

사전에 만들어진 지도(SLAM 또는 지도 파일로부터)를 로드한다. 셀은 지도 데이터를 기반으로 자유, 점유, 알 수 없음으로 표시된다.

#### 장애물 레이어(Obstacle Layer)

실시간 센서 데이터(LiDAR, 깊이 카메라)를 통합하여 정적 지도에 없는 동적 장애물을 감지한다. **복셀 격자(voxel grid)** (3D) 또는 **광선 추적(raytracing)**을 사용하여 센서 빔이 통과하는 셀을 지우고(장애물 없음) 빔이 닿는 셀을 표시한다(장애물).

#### 팽창 레이어(Inflation Layer)

팽창 레이어는 안전한 내비게이션에 중요하다. 장애물 주변에 비용 기울기를 추가한다:

$$\text{cost}(d) = \begin{cases} 254 & \text{if } d \leq r_{inscribed} \\ \alpha \cdot e^{-\beta (d - r_{inscribed})} & \text{if } r_{inscribed} < d \leq r_{inflation} \\ 0 & \text{if } d > r_{inflation} \end{cases}$$

여기서:
- $d$는 가장 가까운 장애물까지의 거리
- $r_{inscribed}$는 로봇 풋프린트(footprint)의 내접 반지름
- $r_{inflation}$은 팽창 반지름(비용이 퍼지는 거리)
- $\alpha$와 $\beta$는 비용 감쇠율 제어

```python
import numpy as np

def compute_inflation_costmap(obstacle_grid, inscribed_radius, inflation_radius,
                               cost_scaling_factor, resolution):
    """Compute inflation layer from obstacle positions.

    Why inflate obstacles? The robot is not a point — it has a physical
    footprint. A path planner working on a point model would plan paths
    that clip the robot's edges through obstacles. The inflation layer
    effectively 'grows' obstacles by the robot's radius, so the planner
    can treat the robot as a point and still produce collision-free paths.

    Why a cost gradient (not binary)? The gradient pushes planned paths
    away from obstacles, creating smoother, safer trajectories. Without
    it, the planner might route the robot right along the wall edge.
    """
    rows, cols = obstacle_grid.shape
    costmap = np.zeros_like(obstacle_grid, dtype=float)

    # Find all obstacle cells
    obstacles = np.argwhere(obstacle_grid == 254)

    # For each cell, compute distance to nearest obstacle
    for r in range(rows):
        for c in range(cols):
            if obstacle_grid[r, c] == 254:
                costmap[r, c] = 254
                continue

            # Distance to nearest obstacle (in meters)
            if len(obstacles) > 0:
                dists = np.sqrt(((obstacles - [r, c]) * resolution)**2).sum(axis=1)
                min_dist = dists.min()
            else:
                min_dist = float('inf')

            if min_dist <= inscribed_radius:
                costmap[r, c] = 253  # Robot footprint touches obstacle
            elif min_dist <= inflation_radius:
                # Exponential cost decay
                costmap[r, c] = int(252 * np.exp(
                    -cost_scaling_factor * (min_dist - inscribed_radius)))
            # else: 0 (free space)

    return costmap.astype(np.uint8)
```

### 2.3 전역 코스트맵과 지역 코스트맵

| 속성 | 전역 코스트맵 | 지역 코스트맵 |
|------|-------------|-------------|
| 프레임 | `map` | `odom` |
| 크기 | 전체 지도 (또는 매우 큰 범위) | 로봇 주변 작은 창 (3-10 m) |
| 업데이트 빈도 | 낮음 (0.5-2 Hz) | 높음 (5-20 Hz) |
| 포함 내용 | 정적 지도 + 팽창 | 동적 장애물 + 팽창 |
| 사용처 | 전역 계획기 | 지역 컨트롤러 |

```yaml
# Example Nav2 costmap configuration (params.yaml)
global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      resolution: 0.05
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      plugins: ["obstacle_layer", "inflation_layer"]
```

---

## 3. 전역 계획(Global Planning)

### 3.1 NavFn (내비게이션 함수)

**NavFn**은 기본 전역 계획기이다. 코스트맵에서 최단 경로를 찾기 위해 **다익스트라(Dijkstra) 알고리즘** 또는 **A***를 사용한다:

$$g(n) = \min_{p \in \text{predecessors}} \left[ g(p) + \text{cost}(p, n) \right]$$

NavFn은 코스트맵 격자에서 동작하며, 셀 비용을 엣지 가중치로 취급한다. 비용이 높은 셀(장애물 근처)은 통과하기 더 비싸므로 계획기는 자연스럽게 장애물 근접을 피하는 경로를 찾는다.

### 3.2 Theta* 계획기

**Theta***는 격자 기반 A*보다 더 부드러운 경로를 생성하는 임의 각도(any-angle) 계획기이다:

- A*는 경로를 격자 엣지(8방향 연결)로 제한하여 들쭉날쭉한 경로를 생성
- Theta*는 인접하지 않은 격자 셀 간의 가시선(line-of-sight) 단축을 허용
- 결과: 불필요한 회전이 적은, 더 짧고 부드러운 경로

**핵심 확인**: 각 노드를 확장할 때, Theta*는 부모의 부모에서 현재 노드까지 명확한 가시선이 있는지 테스트한다. 그렇다면 중간 노드를 건너뛴다.

### 3.3 Smac 계획기

**Smac**(State Machine-based A*-Complemented) 계획기들은 Nav2의 현대적 계획 도구 모음이다:

| 계획기 | 유형 | 최적 용도 |
|--------|------|----------|
| SmacPlanner2D | 2D 격자 탐색 | 원형 로봇 (전방향 또는 차동 구동) |
| SmacPlannerHybrid | 하이브리드 A* (SE(2)) | 자동차형 로봇 (아커만, 비전방향) |
| SmacPlannerLattice | 상태 격자(state lattice) | 복잡한 운동학적 제약이 있는 로봇 |

**하이브리드 A***는 로봇의 회전 반지름을 고려한 경로를 생성하기 위해 $(x, y, \theta)$ 공간을 탐색한다:

$$\mathbf{x}_{k+1} = \mathbf{x}_k + \begin{bmatrix} v \cos\theta_k \cdot \Delta t \\ v \sin\theta_k \cdot \Delta t \\ v \tan\delta / L \cdot \Delta t \end{bmatrix}$$

여기서 $\delta$는 조향각(steering angle), $L$은 축간거리(wheelbase)이다.

```python
def global_planning_comparison():
    """Conceptual comparison of global planning approaches.

    Why multiple planners? Different robots have different kinematics:
    - A round differential-drive robot can turn in place → 2D grid is fine
    - An Ackermann-steered vehicle cannot turn in place → needs Hybrid A*
    - A multi-trailer vehicle has complex constraints → needs lattice planner

    Nav2's plugin architecture lets you swap planners without changing
    the rest of the navigation stack.
    """

    # Simplified A* on a costmap grid
    import heapq

    def astar_grid(costmap, start, goal):
        """A* search on a 2D costmap.

        Why A* over Dijkstra? The heuristic (Euclidean distance) guides
        the search toward the goal, expanding far fewer nodes than
        Dijkstra's uniform expansion. In a large costmap (1000x1000),
        this can be 10-100x faster.
        """
        rows, cols = costmap.shape
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),
                           (-1,-1),(-1,1),(1,-1),(1,1)]:
                nr, nc = current[0]+dr, current[1]+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if costmap[nr, nc] >= 253:  # Lethal or inscribed
                        continue

                    # Movement cost = distance * (1 + cell cost / 252)
                    dist = 1.414 if dr != 0 and dc != 0 else 1.0
                    move_cost = dist * (1 + costmap[nr, nc] / 252.0)
                    new_g = g_score[current] + move_cost

                    if (nr, nc) not in g_score or new_g < g_score[(nr, nc)]:
                        g_score[(nr, nc)] = new_g
                        # Heuristic: Euclidean distance to goal
                        h = ((nr-goal[0])**2 + (nc-goal[1])**2)**0.5
                        f = new_g + h
                        heapq.heappush(open_set, (f, (nr, nc)))
                        came_from[(nr, nc)] = current

        return None  # No path found

    return astar_grid
```

---

## 4. 지역 계획(Local Planning) — 컨트롤러 플러그인

### 4.1 컨트롤러의 역할

지역 컨트롤러(Nav2에서 "controller"라 부름)는 전역 경로와 지역 코스트맵을 받아 높은 빈도(20-100 Hz)로 속도 명령 ($v$, $\omega$)을 생성한다. 컨트롤러는 다음을 수행해야 한다:

- 전역 경로를 대략적으로 추종
- 지역 코스트맵에서 보이는 동적 장애물 회피
- 로봇의 운동학적(kinematic) 및 동역학적(dynamic) 제약 준수
- 부드럽고 실현 가능한 속도 명령 생성

### 4.2 DWB (동적 창 기반, Dynamic Window Based)

**DWB**(Dynamic Window approach version B)는 동적 창 접근법(DWA)의 Nav2 구현이다:

1. 동적 창 제한 내의 허용 속도 공간 $(v, \omega)$에서 **속도 샘플링**
2. 각 속도 샘플에 대해 앞으로 **궤적을 시뮬레이션** (일반적으로 1-3초)
3. 여러 평가 기준(경로 정렬, 목표 거리, 장애물 근접도)으로 **각 궤적을 채점**
4. **최선의** 궤적을 선택하고 그 첫 번째 속도 명령을 실행

```python
def dwb_concept(robot_state, global_path, local_costmap,
                v_range, omega_range, dt=0.1, horizon=2.0):
    """Conceptual DWB local planner.

    Why sample-and-score instead of optimization? Sampling is robust
    to local minima and easy to implement. The critics (scoring
    functions) can encode complex behaviors without needing gradients.
    The downside: computational cost scales with the number of samples.
    """
    n_v_samples = 20
    n_omega_samples = 40
    best_score = -float('inf')
    best_v, best_omega = 0.0, 0.0

    v_min, v_max = v_range
    omega_min, omega_max = omega_range

    for v in np.linspace(v_min, v_max, n_v_samples):
        for omega in np.linspace(omega_min, omega_max, n_omega_samples):
            # Simulate trajectory
            traj = simulate_trajectory(robot_state, v, omega, dt, horizon)

            # Score trajectory using multiple critics
            score = 0.0

            # Critic 1: Path alignment — how close is the trajectory to the global path?
            path_dist = compute_path_distance(traj, global_path)
            score -= 5.0 * path_dist

            # Critic 2: Goal distance — how close does the trajectory end to the goal?
            goal_dist = compute_goal_distance(traj[-1], global_path[-1])
            score -= 3.0 * goal_dist

            # Critic 3: Obstacle proximity — how close to obstacles?
            obs_cost = compute_obstacle_cost(traj, local_costmap)
            if obs_cost >= 253:  # Lethal collision
                continue  # Skip this trajectory entirely
            score -= 2.0 * obs_cost

            # Critic 4: Forward progress — prefer moving forward
            score += 1.0 * v

            if score > best_score:
                best_score = score
                best_v, best_omega = v, omega

    return best_v, best_omega


def simulate_trajectory(state, v, omega, dt, horizon):
    """Forward-simulate a constant-velocity trajectory."""
    x, y, theta = state
    trajectory = [(x, y, theta)]
    t = 0.0

    while t < horizon:
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt
        trajectory.append((x, y, theta))
        t += dt

    return trajectory
```

### 4.3 MPPI (모델 예측 경로 적분, Model Predictive Path Integral)

**MPPI**는 확률론적 최적화를 사용하는 현대적인 샘플링 기반 컨트롤러이다:

1. $N$개의 무작위 제어 시퀀스(속도 궤적) 샘플링
2. 각각을 동역학 모델을 통해 앞으로 시뮬레이션
3. 비용 함수를 사용하여 각 궤적 채점
4. 제어 시퀀스의 **가중 평균** 계산 (가중치 = $e^{-\text{cost}/\lambda}$)
5. 가중 평균 시퀀스의 첫 번째 제어를 실행

$$\mathbf{u}^* = \frac{\sum_{i=1}^N w_i \mathbf{u}^{(i)}}{\sum_{i=1}^N w_i}, \qquad w_i = \exp\left(-\frac{1}{\lambda} S(\mathbf{x}^{(i)})\right)$$

여기서 $S(\mathbf{x}^{(i)})$는 궤적 비용, $\lambda$는 온도 파라미터이다.

**DWB 대비 장점**:
- 더 풍부한 궤적 집합 탐색 (일정 속도만이 아님)
- 복잡한 비용 함수를 자연스럽게 처리
- GPU 가속 가능 (샘플링이 병렬 독립적)
- 좁은 공간과 동적 환경에서 더 우수한 성능

### 4.4 규제 순수 추종(Regulated Pure Pursuit)

**규제 순수 추종**은 부드러운 경로 추종에 적합한 더 단순한 컨트롤러이다:

1. 로봇 앞 거리 $L$에서 전역 경로의 **전방주시점(lookahead point)** 찾기
2. 그 지점에 도달하기 위한 곡률(curvature) 계산:
$$\kappa = \frac{2 \sin(\alpha)}{L}$$
여기서 $\alpha$는 로봇의 진행 방향과 전방주시점 사이의 각도
3. 곡률과 장애물 근접도에 기반하여 **속도를 규제**

```python
class RegulatedPurePursuit:
    """Regulated Pure Pursuit controller.

    Why 'regulated'? Classic pure pursuit drives at a fixed speed and
    only controls steering. Regulated pure pursuit also adapts the speed:
    slower near obstacles, slower in tight turns, and approaching the
    goal. This makes it much safer and more practical than classic
    pure pursuit.
    """

    def __init__(self, lookahead_dist=0.6, max_speed=0.5,
                 min_speed=0.1, max_angular_vel=1.0):
        self.lookahead_dist = lookahead_dist
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_angular_vel = max_angular_vel

    def compute_velocity(self, robot_pose, path, costmap=None):
        """Compute velocity command from path and robot pose.

        Why use a lookahead distance? Looking ahead smooths out small
        path deviations and prevents oscillation. Too short a lookahead
        causes jerky behavior; too long causes corner cutting. The
        optimal lookahead depends on speed and path curvature.
        """
        # Find lookahead point on path
        lookahead_pt = self._find_lookahead(robot_pose, path)
        if lookahead_pt is None:
            return 0.0, 0.0  # No path to follow

        rx, ry, rtheta = robot_pose
        lx, ly = lookahead_pt

        # Transform lookahead to robot frame
        dx = lx - rx
        dy = ly - ry
        local_x = np.cos(rtheta)*dx + np.sin(rtheta)*dy
        local_y = -np.sin(rtheta)*dx + np.cos(rtheta)*dy

        # Compute curvature
        L_sq = local_x**2 + local_y**2
        if L_sq < 1e-6:
            return 0.0, 0.0
        curvature = 2.0 * local_y / L_sq

        # Regulate speed based on curvature
        # Higher curvature = lower speed (tighter turn)
        curvature_speed = self.max_speed / (1.0 + abs(curvature) * 5.0)

        # Regulate speed based on obstacle proximity (if costmap available)
        obstacle_speed = self.max_speed
        if costmap is not None:
            # Check cost along path ahead
            max_cost = self._check_path_cost(robot_pose, path, costmap)
            if max_cost > 128:
                obstacle_speed = self.min_speed

        # Final speed is minimum of all regulations
        speed = max(self.min_speed, min(curvature_speed, obstacle_speed))

        # Angular velocity from curvature and speed
        angular_vel = np.clip(speed * curvature,
                               -self.max_angular_vel, self.max_angular_vel)

        return speed, angular_vel

    def _find_lookahead(self, robot_pose, path):
        """Find the point on the path at lookahead distance."""
        rx, ry = robot_pose[0], robot_pose[1]

        for i in range(len(path) - 1):
            px, py = path[i]
            dist = np.sqrt((px - rx)**2 + (py - ry)**2)
            if dist >= self.lookahead_dist:
                return (px, py)

        # Return last point if path is shorter than lookahead
        return tuple(path[-1]) if len(path) > 0 else None

    def _check_path_cost(self, robot_pose, path, costmap):
        """Check maximum costmap cost along upcoming path."""
        # Simplified: check a few points ahead
        return 0  # Placeholder
```

### 4.5 컨트롤러 비교

| 컨트롤러 | 장점 | 단점 | 최적 용도 |
|---------|------|------|----------|
| DWB | 검증된 성능, 설정 가능한 평가 기준 | 제한된 궤적 다양성 | 범용 목적 |
| MPPI | 풍부한 궤적, 복잡한 비용 처리 가능 | 계산 비용이 높음 | 동적 환경, 좁은 공간 |
| 규제 PP | 단순하고, 부드럽고, 예측 가능 | 장애물 회피 없음 (전역 경로에 의존) | 개방된 환경, 부드러운 경로 |

---

## 5. 복구 행동(Recovery Behaviors)

### 5.1 복구가 필수적인 이유

로봇은 막힌다. 전역 경로가 지도에 없는 장애물에 의해 차단될 수 있다. 지역 계획기가 코너에서 진동할 수 있다. 사람이 앞에 서 있을 수 있다. 복구 행동은 로봇이 영구적으로 막히지 않도록 하는 안전망이다.

### 5.2 내장 복구 행동

| 복구 | 설명 | 사용 시기 |
|------|------|----------|
| **Spin** (회전) | 제자리에서 회전 (360도) | 장애물 판독값을 지우고 새 경로 탐색 |
| **BackUp** (후진) | 짧은 거리를 후진 | 회전하기엔 너무 가까운 장애물에서 후진 |
| **Wait** (대기) | 멈추고 장애물이 이동할 때까지 대기 | 동적 장애물 (사람, 다른 로봇) |
| **ClearCostmap** (코스트맵 초기화) | 지역 코스트맵의 장애물 레이어 초기화 | 오래된 장애물 데이터가 거짓 차단을 유발할 때 |

### 5.3 복구 설정

```yaml
# Recovery behavior configuration
recoveries_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    recovery_plugins: ["spin", "backup", "wait"]
    spin:
      plugin: "nav2_recoveries/Spin"
    backup:
      plugin: "nav2_recoveries/BackUp"
    wait:
      plugin: "nav2_recoveries/Wait"
```

---

## 6. 내비게이션을 위한 행동 트리(Behavior Tree)

### 6.1 행동 트리를 사용하는 이유

Nav2는 내비게이션 파이프라인을 조율하기 위해 **행동 트리(BT, Behavior Trees)**를 사용한다. 행동 트리는 다음과 같은 계층적 의사결정 구조이다:

- **모듈화**: 각 행동은 독립적인 노드
- **반응성**: 트리가 고정 빈도로 틱(tick)되며, 행동을 중단할 수 있음
- **가독성**: 트리 구조가 제어 로직을 명확하고 검사 가능하게 만듦
- **확장성**: 새로운 행동을 BT 노드로 쉽게 추가 가능

### 6.2 BT 노드 타입

| 노드 타입 | 기호 | 동작 |
|----------|------|------|
| **시퀀스(Sequence)** | → | 자식을 좌에서 우로 실행; 자식 중 하나라도 실패하면 실패 |
| **폴백(Fallback)** | ? | 자식을 좌에서 우로 시도; 자식 중 하나라도 성공하면 성공 |
| **액션(Action)** | [박스] | ROS2 액션 실행 (계획, 추종, 회전 등) |
| **조건(Condition)** | (타원) | 조건 확인 (목표 업데이트됨? 경로 유효?) |
| **데코레이터(Decorator)** | ◇ | 자식 동작 수정 (재시도, 반전, 속도 제한) |

### 6.3 기본 내비게이션 BT

기본 Nav2 행동 트리(단순화):

```
[Sequence] NavigateWithRecovery
├── [RateController 1Hz]
│   └── [Sequence] Navigate
│       ├── [Action] ComputePathToPose          ← Global planning
│       ├── [Action] SmoothPath (optional)       ← Path smoothing
│       └── [Action] FollowPath                  ← Local control
└── [Fallback] RecoveryFallback
    ├── [Action] ClearLocalCostmap
    ├── [Action] Spin
    ├── [Action] Wait
    └── [Action] BackUp
```

동작:
1. **시퀀스**가 내비게이션을 시도한다: 경로 계산, 스무딩, 추종
2. 시퀀스가 **실패**하면 (계획기 실패, 컨트롤러 실패 등), 실행이 복구 폴백으로 이동
3. **폴백**이 하나가 성공할 때까지 순서대로 각 복구를 시도
4. 성공적인 복구 후, 시퀀스가 내비게이션을 재시도

### 6.4 사용자 정의 행동 트리

직접 BT XML을 작성하여 내비게이션 행동을 커스터마이즈할 수 있다:

```xml
<!-- Custom navigation behavior tree -->
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <PipelineSequence name="NavigateWithRetry">
      <!-- Retry navigation up to 3 times -->
      <RetryNode num_attempts="3" name="RetryNav">
        <PipelineSequence name="NavigateSequence">
          <!-- Compute global path -->
          <ComputePathToPose goal="{goal}" path="{path}"
                             planner_id="GridBased"/>

          <!-- Follow the path -->
          <FollowPath path="{path}" controller_id="FollowPath"/>
        </PipelineSequence>
      </RetryNode>

      <!-- If navigation fails after retries, try recovery -->
      <RecoveryNode name="RecoveryActions" number_of_retries="2">
        <Sequence name="RecoverySequence">
          <ClearEntireCostmap name="ClearLocal"
                              service_name="local_costmap/clear_entirely_local_costmap"/>
          <Spin spin_dist="1.57"/>
        </Sequence>
      </RecoveryNode>
    </PipelineSequence>
  </BehaviorTree>
</root>
```

```python
def behavior_tree_concept():
    """Simplified behavior tree execution engine.

    Why behavior trees over state machines? State machines become
    unwieldy as complexity grows — the number of transitions explodes
    combinatorially. Behavior trees scale better: adding a new behavior
    is just adding a new node, without rewiring the entire graph.
    They also handle interrupts naturally (reactive ticking).
    """

    class BTNode:
        """Base class for behavior tree nodes."""
        SUCCESS = 'SUCCESS'
        FAILURE = 'FAILURE'
        RUNNING = 'RUNNING'

        def tick(self):
            raise NotImplementedError

    class Sequence(BTNode):
        """Executes children in order. Fails if any child fails."""
        def __init__(self, children):
            self.children = children
            self.current = 0

        def tick(self):
            while self.current < len(self.children):
                status = self.children[self.current].tick()
                if status == self.RUNNING:
                    return self.RUNNING
                if status == self.FAILURE:
                    self.current = 0
                    return self.FAILURE
                self.current += 1
            self.current = 0
            return self.SUCCESS

    class Fallback(BTNode):
        """Tries children in order. Succeeds if any child succeeds."""
        def __init__(self, children):
            self.children = children

        def tick(self):
            for child in self.children:
                status = child.tick()
                if status == self.SUCCESS:
                    return self.SUCCESS
                if status == self.RUNNING:
                    return self.RUNNING
            return self.FAILURE

    return Sequence, Fallback
```

---

## 7. 웨이포인트 추종과 자율 내비게이션

### 7.1 웨이포인트 추종

다중 목표 임무(순찰, 배달, 점검)를 위해 Nav2는 `NavigateThroughPoses` 액션을 통한 웨이포인트 추종을 제공한다:

```python
from nav2_msgs.action import NavigateThroughPoses
from geometry_msgs.msg import PoseStamped


class WaypointMission(Node):
    """Multi-waypoint autonomous navigation.

    Why waypoint following instead of single goals? Real missions
    involve multiple stops: a delivery robot visits stations A, B, C.
    A patrol robot follows a predefined route. Waypoint following
    handles the multi-goal orchestration, including what to do at
    each waypoint (wait, perform action, take photo).
    """

    def __init__(self):
        super().__init__('waypoint_mission')
        self.nav_client = ActionClient(
            self, NavigateThroughPoses, 'navigate_through_poses')

    def create_mission(self, waypoints):
        """Create a navigation mission from a list of (x, y, yaw) waypoints."""
        goal = NavigateThroughPoses.Goal()

        for x, y, yaw in waypoints:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.z = np.sin(yaw / 2)
            pose.pose.orientation.w = np.cos(yaw / 2)
            goal.poses.append(pose)

        return goal

    def execute_mission(self, waypoints):
        """Send waypoint mission to Nav2."""
        self.nav_client.wait_for_server()
        goal = self.create_mission(waypoints)
        future = self.nav_client.send_goal_async(
            goal, feedback_callback=self.feedback_callback)
        return future

    def feedback_callback(self, feedback_msg):
        """Monitor navigation progress."""
        feedback = feedback_msg.feedback
        current_pose = feedback.current_pose
        # n_remaining = feedback.number_of_poses_remaining
        self.get_logger().info(
            f'Navigating: at ({current_pose.pose.position.x:.1f}, '
            f'{current_pose.pose.position.y:.1f})')
```

### 7.2 웨이포인트에서의 작업 실행기

Nav2는 각 웨이포인트에서 실행되는 **웨이포인트 작업 실행기(waypoint task executor)** 플러그인을 지원한다:

```python
def waypoint_task_executor_concept():
    """Concept: execute a task at each waypoint.

    Examples of waypoint tasks:
    - Take a photo (inspection robot)
    - Wait for loading/unloading (delivery robot)
    - Collect sensor data (environmental monitoring)
    - Announce arrival (service robot)

    Nav2 supports custom task executor plugins that are called
    when the robot arrives at each waypoint.
    """
    pass
```

---

## 8. 전체 통합

### 8.1 최소 Nav2 설정

```yaml
# Minimal Nav2 params.yaml for a differential-drive robot
bt_navigator:
  ros__parameters:
    global_frame: map
    robot_base_frame: base_link
    default_bt_xml_filename: "navigate_w_replanning_and_recovery.xml"

planner_server:
  ros__parameters:
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: true
      allow_unknown: true

controller_server:
  ros__parameters:
    controller_frequency: 20.0
    controller_plugins: ["FollowPath"]
    FollowPath:
      plugin: "dwb_core::DWBLocalPlanner"
      min_vel_x: 0.0
      max_vel_x: 0.5
      max_vel_theta: 1.0
      min_speed_xy: 0.0
      max_speed_xy: 0.5
      acc_lim_x: 2.5
      acc_lim_theta: 3.2
      decel_lim_x: -2.5
```

### 8.2 일반적인 문제 해결

| 문제 | 가능한 원인 | 해결책 |
|------|-----------|--------|
| 로봇이 움직이지 않음 | TF 트리 오류 | `map→odom→base_link` 연결 확인 |
| "유효한 경로 없음" | 코스트맵이 너무 많이 팽창됨 | `inflation_radius` 줄이기 |
| 로봇이 진동함 | 컨트롤러 게인이 너무 공격적 | 속도 줄이기, 스무딩 늘리기 |
| 로봇이 장애물에 부딪힘 | 지역 코스트맵이 업데이트되지 않음 | 센서 토픽, QoS 호환성 확인 |
| 복구가 계속 발동됨 | 경로가 벽에 너무 가까움 | 코스트맵 팽창 늘리기 |

---

## 요약

| 개념 | 핵심 내용 |
|------|----------|
| Nav2 | 완전한 내비게이션 스택: 계획, 제어, 복구, 행동 트리 |
| 코스트맵 레이어 | 정적 지도 + 동적 장애물 + 팽창 = 내비게이션 격자 |
| 팽창(Inflation) | 로봇 반지름으로 장애물 확장 + 안전 여유를 위한 비용 기울기 |
| 전역 계획기 | A*, Theta*, 하이브리드 A*; 충돌 없는 경로를 위한 전역 코스트맵 탐색 |
| 지역 컨트롤러 | DWB (샘플 & 채점), MPPI (확률론적 최적), 순수 추종 (기하학적) |
| 복구 행동 | 회전, 후진, 대기, 코스트맵 초기화 — 내비게이션 실패 시 발동 |
| 행동 트리 | 계층적, 반응형 작업 조율; 상태 머신을 대체 |
| 웨이포인트 추종 | 각 웨이포인트에서 선택적 작업 실행이 가능한 다중 목표 임무 |

---

## 연습 문제

1. **코스트맵 분석**: 5개의 직사각형 장애물이 있는 100x100 점유 격자에서, 내접 반지름 0.2 m, 팽창 반지름 0.6 m, 비용 스케일링 팩터 3.0으로 팽창 레이어를 구현하라. 결과 코스트맵을 히트맵(heatmap)으로 시각화하라. 팽창된 코스트맵에서 계획된 경로가 장애물 근접을 어떻게 피하는지 보여라.

2. **전역 계획기 비교**: 2D 코스트맵에서 A*와 단순화된 Theta*를 구현하라. 세 가지 시나리오에 대한 결과 경로를 비교하라: (a) 개방된 공간, (b) 좁은 통로, (c) 막다른 길이 있는 미로. 각각의 경로 길이와 회전 수를 측정하라.

3. **DWB 구현**: 위에서 설명한 샘플-채점 지역 계획기를 구현하라. 차동 구동 로봇이 전역 지도에 없는 장애물 근처를 통과하는 전역 경로를 추종하는 시나리오를 만들어라. 지역 계획기가 지역 코스트맵에 대한 궤적 채점을 통해 장애물을 회피하는 것을 보여라.

4. **행동 트리 시뮬레이터**: 단순화된 행동 트리 실행 엔진(시퀀스, 폴백, 액션 노드)을 구현하라. 다음 내비게이션 BT를 만들어라: 목표까지 내비게이션을 시도하고, 실패하면 회전 복구를 시도한 다음 후진을 시도하고, 그래도 실패하면 실패를 선언한다. 다음 시나리오를 시뮬레이션하라: (a) 내비게이션 성공, (b) 회전 복구로 문제 해결, (c) 모든 복구가 실패하는 경우.

5. **웨이포인트 임무**: 6개의 웨이포인트로 루프를 형성하는 창고 로봇의 순찰 임무를 설계하라. 로봇은 각 웨이포인트를 방문하고, 5초 대기한 후(점검 시뮬레이션) 다음 웨이포인트로 이동해야 한다. 내비게이션 구간이 실패하면 해당 웨이포인트를 건너뛰고 다음으로 진행해야 한다. 임무 로직을 의사코드 또는 Python으로 구현하라.

---

## 더 읽을거리

- Macenski, S. et al. "The Marathon 2: A Navigation System." *IEEE/RSJ IROS*, 2020. (Nav2 아키텍처 및 설계)
- Nav2 공식 문서: [navigation.ros.org](https://navigation.ros.org/) (설정 가이드, 튜토리얼)
- Colledanchise, M. and Ogren, P. *Behavior Trees in Robotics and AI*. CRC Press, 2018. (행동 트리 이론 및 실습)
- Fox, D. et al. "The Dynamic Window Approach to Collision Avoidance." *IEEE Robotics & Automation Magazine*, 1997. (DWA 기초)
- Williams, G. et al. "Information Theoretic MPC for Model-Based Reinforcement Learning." *ICRA*, 2017. (MPPI 이론)

---

[← 이전: ROS2 기초](13_ROS2_Fundamentals.md) | [다음: 로봇공학을 위한 강화학습 →](15_RL_for_Robotics.md)
