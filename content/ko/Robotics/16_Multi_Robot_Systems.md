# 16. 다중 로봇 시스템과 군집(Multi-Robot Systems and Swarms)

[← 이전: 로보틱스를 위한 강화학습](15_RL_for_Robotics.md)

---

## 학습 목표

1. 단일 로봇과 다중 로봇 시스템 패러다임을 구분하고 다중 로봇 접근 방식이 유리한 경우를 식별한다
2. 경매 기반 및 최적화 기반 방법을 사용하여 작업 할당(task allocation) 문제를 공식화하고 해결한다
3. 대형 제어(formation control) 전략을 구현한다: 선두-추종(leader-follower), 가상 구조(virtual structure), 행동 기반(behavior-based)
4. 합의 알고리즘(consensus algorithms)과 분산 협조(distributed coordination)에서의 역할을 이해한다
5. 군집 지능(swarm intelligence) 원리를 설명한다: Reynolds 군집 비행(flocking), 개미 군집 최적화(ACO), 입자 군집 최적화(PSO)
6. 다중 로봇 시스템을 위한 통신 아키텍처와 그 트레이드오프를 분석한다

---

이 강좌 전반에 걸쳐 우리는 단일 로봇에 집중해 왔다 — 하나의 조작기, 하나의 이동 플랫폼, 하나의 센서 집합. 하지만 많은 실제 세계 작업들은 함께 일하는 로봇 팀에 의해 더 잘 수행된다. 지진 후 붕괴된 건물을 수색하는 단일 로봇은 몇 시간이 걸릴 것이다; 50대의 소형 로봇 군집은 같은 면적을 몇 분 만에 커버할 수 있다. 단일 창고 로봇은 한 번에 하나의 패키지를 이동하지만; Amazon의 Kiva 시스템처럼 500대의 로봇 함대는 하루에 수백만 개의 패키지를 처리한다.

다중 로봇 시스템은 근본적으로 새로운 과제들을 도입한다: 로봇들이 어떻게 일을 나누는가? 서로 충돌을 어떻게 피하는가? 통신이 제한적일 때 어떻게 정보를 공유하는가? 그리고 아마도 가장 매혹적으로, 간단한 개별 규칙에서 복잡하게 조율된 행동이 어떻게 출현할 수 있는가?

이 레슨은 공식적인 작업 할당 알고리즘에서부터 군집 지능의 아름다운 단순함까지 이러한 질문들을 탐구하며, 중앙 조정자 없이 집단적 행동이 출현하는 곳까지 다룬다.

> **비유**: 로봇 군집은 새 떼와 같다 — 단일 리더 없이도 단순한 지역 규칙에서 복잡하게 조율된 행동이 출현한다. 각 새는 세 가지 규칙을 따른다: 이웃과 가까이 있기(응집력), 이웃과 충돌하지 않기(분리), 이웃과 같은 방향으로 이동하기(정렬). 이 세 가지 간단한 규칙으로부터, 수천 마리의 새들이 어떤 새도 대형을 "계획"하지 않고 숨막히는 대형을 만들어낸다. 군집 로보틱스(swarm robotics)는 동일한 원리를 적용한다: 간단한 지역 규칙, 복잡한 전역 행동.

---

## 1. 단일 로봇 vs. 다중 로봇 시스템

### 1.1 왜 여러 대의 로봇이 필요한가?

| 장점 | 설명 | 예시 |
|------|------|------|
| **병렬성(parallelism)** | 여러 로봇이 동시에 작업 | 50대의 로봇이 건물을 1/50의 시간에 청소 |
| **강건성(robustness)** | 시스템이 개별 장애를 허용 | 탐색 드론 한 대가 충돌해도 나머지 19대가 계속 |
| **공간 분산(spatial distribution)** | 대규모 면적 동시 커버 | 100 km²에 걸친 환경 모니터링 |
| **보완적 역량(complementary capabilities)** | 다른 하위 작업을 위한 다른 로봇 | 검사를 위한 지상 로봇 + 공중 드론 |
| **비용 효율성(cost efficiency)** | 많은 단순 로봇이 하나의 복잡한 로봇보다 저렴 | $100 드론 군집 vs. $50,000 로봇 하나 |

### 1.2 과제

| 과제 | 설명 |
|------|------|
| **조율(coordination)** | 로봇들이 충돌하는 행동을 피해야 함 |
| **통신(communication)** | 대역폭 제한, 지연, 메시지 손실 |
| **작업 할당(task allocation)** | 누가 무엇을 하는가? 일반적으로 NP-난해 |
| **충돌 회피(collision avoidance)** | 로봇들이 서로 충돌하지 않아야 함 |
| **확장성(scalability)** | 알고리즘이 10대와 10,000대 로봇 모두에서 작동해야 함 |
| **이질성(heterogeneity)** | 다른 로봇들이 다른 역량을 가질 수 있음 |

### 1.3 다중 로봇 시스템의 분류 체계

```
Multi-Robot Systems
├── Cooperative (shared goal)
│   ├── Centralized (single planner)
│   ├── Decentralized (distributed decision-making)
│   └── Hybrid
├── Competitive (conflicting goals)
│   └── Adversarial (pursuit-evasion, etc.)
└── By structure
    ├── Homogeneous (identical robots)
    └── Heterogeneous (different capabilities)
```

---

## 2. 작업 할당(Task Allocation)

### 2.1 문제 정의

주어진 것:
- 역량 $C_1, C_2, \ldots, C_N$을 가진 $N$대의 로봇
- 요구 사항 $T_1, T_2, \ldots, T_M$을 가진 $M$개의 작업
- 비용 함수 $c_{ij}$ = 로봇 $i$가 작업 $j$를 수행하는 비용

찾아야 할 것: 모든 제약을 만족하면서 총 비용을 최소화하는 할당.

이것은 **할당 문제(assignment problem)**의 변형으로, 다중 로봇 다중 작업 시나리오에서는 일반적으로 NP-난해다.

### 2.2 경매 기반 할당(Auction-Based Allocation)

**시장 기반 접근 방식(market-based approaches)**은 작업 할당을 로봇들이 작업에 입찰하는 경매로 취급한다:

```python
import numpy as np

class AuctionAllocator:
    """Sequential single-item auction for task allocation.

    Why auction-based? Auctions are naturally decentralized — each robot
    computes its own bid based on local information (distance to task,
    current battery, capability). The auctioneer only needs to collect
    bids and assign winners. This scales better than centralized
    optimization and degrades gracefully when communication is lossy.
    """

    def __init__(self, n_robots, n_tasks):
        self.n_robots = n_robots
        self.n_tasks = n_tasks

    def compute_bids(self, robot_positions, task_positions, robot_capabilities=None):
        """Each robot bids on each task based on estimated cost.

        Why use negative distance as bid (higher = better)?
        In auctions, the highest bidder wins. We want the closest
        robot to win the task, so we use -distance as the bid.
        More sophisticated bids could factor in battery level,
        current workload, and capability match.
        """
        bids = np.zeros((self.n_robots, self.n_tasks))

        for i in range(self.n_robots):
            for j in range(self.n_tasks):
                distance = np.linalg.norm(
                    robot_positions[i] - task_positions[j])
                # Bid = negative distance (closer = higher bid)
                bids[i, j] = -distance

                # Capability check: zero bid if robot can't do the task
                if robot_capabilities is not None:
                    if not robot_capabilities[i].get(j, True):
                        bids[i, j] = -np.inf

        return bids

    def sequential_auction(self, bids):
        """Assign tasks one by one to the highest bidder.

        Why sequential instead of simultaneous? Sequential auctions are
        simpler and avoid the combinatorial explosion of simultaneous
        auctions. The trade-off: sequential auctions may not find the
        globally optimal assignment, but they are fast and produce
        reasonable solutions.
        """
        n_robots, n_tasks = bids.shape
        assignments = {}  # task_id -> robot_id
        available_robots = set(range(n_robots))

        for task in range(min(n_tasks, n_robots)):
            # Find the best available robot for this task
            best_robot = None
            best_bid = -np.inf

            for robot in available_robots:
                if bids[robot, task] > best_bid:
                    best_bid = bids[robot, task]
                    best_robot = robot

            if best_robot is not None:
                assignments[task] = best_robot
                available_robots.remove(best_robot)

        return assignments

    def consensus_auction(self, bids, n_rounds=10):
        """Consensus-Based Bundle Algorithm (CBBA) — decentralized auction.

        Why consensus? In a fully decentralized system, there is no
        auctioneer. Each robot maintains its own view of the assignment
        and communicates with neighbors to reach agreement. CBBA
        alternates: (1) each robot bids on its best available task,
        (2) robots exchange bids and the highest bid wins each task.
        Converges to a conflict-free assignment if the graph is connected.
        """
        # Simplified CBBA: iteratively bid and resolve conflicts
        # Phase 1: each robot selects its best task
        # Phase 2: robots exchange bids, highest bidder wins
        # Repeat until convergence
        # Full implementation uses local_assignments + local_bids matrices
        pass  # See exercises for full implementation
```

### 2.3 최적화 기반 할당(Optimization-Based Allocation)

계산이 감당 가능한 소규모 팀의 경우 최적 할당을 풀 수 있다:

$$\min \sum_{i=1}^{N} \sum_{j=1}^{M} c_{ij} x_{ij}$$

제약 조건:
$$\sum_{j=1}^{M} x_{ij} \leq 1 \quad \forall i \quad \text{(each robot does at most one task)}$$
$$\sum_{i=1}^{N} x_{ij} = 1 \quad \forall j \quad \text{(each task assigned to exactly one robot)}$$
$$x_{ij} \in \{0, 1\}$$

**헝가리안 알고리즘(Hungarian algorithm)**은 이것을 $O(\max(N,M)^3)$ 시간에 최적으로 해결한다.

```python
from scipy.optimize import linear_sum_assignment

def optimal_assignment(cost_matrix):
    """Solve the assignment problem using the Hungarian algorithm.

    Why optimal assignment for small teams? For 10 robots and 10 tasks,
    the Hungarian algorithm runs in milliseconds and finds the globally
    optimal assignment. For 1000 robots, it takes seconds — still
    feasible for one-time allocation. For dynamic re-allocation at
    high frequency, auction-based methods are more practical.
    """
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    assignments = {col: row for row, col in zip(row_indices, col_indices)}
    total_cost = cost_matrix[row_indices, col_indices].sum()
    return assignments, total_cost
```

---

## 3. 대형 제어(Formation Control)

### 3.1 왜 대형이 필요한가?

대형으로 움직이는 로봇들은 다음을 할 수 있다:
- 대형 물체를 협력하여 운반
- 알려진 기하학을 가진 센서 배열 생성
- 통신 네트워크 토폴로지 유지
- 영역 커버리지 보장 제공

### 3.2 선두-추종(Leader-Follower)

가장 단순한 대형 전략: 한 로봇이 선두가 되고, 나머지는 선두에 대한 원하는 오프셋을 유지한다.

```python
class LeaderFollowerFormation:
    """Leader-follower formation control.

    Why leader-follower? It's the simplest formation strategy.
    One robot (the leader) follows a path or is teleoperated.
    Followers maintain a fixed offset relative to the leader.
    The leader doesn't need to know about the followers.

    Limitation: single point of failure — if the leader fails,
    the entire formation breaks down. No true distributed control.
    """

    def __init__(self, n_followers, desired_offsets):
        """
        desired_offsets: list of (dx, dy) relative to leader
        Example: [(-1, 1), (-1, -1), (-2, 0)] for a V-formation
        """
        self.n_followers = n_followers
        self.offsets = np.array(desired_offsets)

    def compute_follower_goals(self, leader_pose):
        """Compute desired positions for each follower.

        Why rotate offsets by the leader's heading? The formation
        should maintain its shape relative to the leader's direction
        of travel, not in a fixed global frame. If the leader turns
        left, the formation turns left with it.
        """
        lx, ly, ltheta = leader_pose
        c, s = np.cos(ltheta), np.sin(ltheta)

        goals = []
        for dx, dy in self.offsets:
            # Rotate offset by leader heading
            gx = lx + c * dx - s * dy
            gy = ly + s * dx + c * dy
            goals.append((gx, gy, ltheta))

        return goals

    def follower_control(self, follower_pose, goal_pose, kp=2.0, kd=0.5):
        """Simple proportional control for a follower.

        Each follower independently tracks its goal position using
        a simple proportional controller. In practice, you'd use
        a more sophisticated controller (MPC, DWA) that also
        considers inter-robot collision avoidance.
        """
        fx, fy, ftheta = follower_pose
        gx, gy, gtheta = goal_pose

        # Distance and angle to goal
        dx = gx - fx
        dy = gy - fy
        dist = np.sqrt(dx**2 + dy**2)
        angle_to_goal = np.arctan2(dy, dx)

        # Heading error
        heading_error = angle_to_goal - ftheta
        heading_error = (heading_error + np.pi) % (2*np.pi) - np.pi

        # Control commands
        linear_vel = kp * dist
        angular_vel = kd * heading_error

        return linear_vel, angular_vel
```

### 3.3 가상 구조(Virtual Structure)

**가상 구조** 접근 방식은 대형을 강체(rigid body)로 취급한다. 가상 구조가 이동하고 각 로봇은 구조의 할당된 점을 추적한다.

```python
class VirtualStructureFormation:
    """Virtual structure formation control.

    Why virtual structure? It ensures the formation moves as a
    rigid body — all robots accelerate, turn, and stop together.
    The formation shape is perfectly maintained (in theory).

    The virtual structure is an imaginary rigid body that exists
    only in software. Each robot is assigned a point on this
    structure and tracks it using its own controller.
    """

    def __init__(self, formation_points):
        """
        formation_points: list of (dx, dy) relative to virtual structure center
        """
        self.structure_points = np.array(formation_points)
        self.center = np.array([0.0, 0.0])
        self.heading = 0.0
        self.velocity = np.array([0.0, 0.0])

    def update_structure(self, desired_center, desired_heading, dt):
        """Move the virtual structure toward its goal.

        Why move the structure smoothly? Abrupt changes would require
        all robots to make sudden movements simultaneously. Smooth
        virtual structure motion produces smooth robot motion.
        """
        # Simple proportional control of the virtual structure
        kp = 1.0
        self.center += kp * (desired_center - self.center) * dt
        heading_error = desired_heading - self.heading
        heading_error = (heading_error + np.pi) % (2*np.pi) - np.pi
        self.heading += kp * heading_error * dt

    def get_robot_goals(self):
        """Compute goal positions for all robots on the virtual structure."""
        c, s = np.cos(self.heading), np.sin(self.heading)
        goals = []
        for dx, dy in self.structure_points:
            gx = self.center[0] + c*dx - s*dy
            gy = self.center[1] + s*dx + c*dy
            goals.append((gx, gy, self.heading))
        return goals
```

### 3.4 행동 기반 대형(Behavior-Based Formation)

강체 구조 대신, 각 로봇이 출현하는 대형 패턴을 만들어내는 지역 행동 규칙을 따른다:

```python
class BehaviorBasedFormation:
    """Behavior-based formation using potential fields.

    Why behavior-based? It's fully decentralized — no leader, no
    virtual structure, no central coordinator. Each robot makes
    decisions based only on local information (positions of nearby
    robots). This makes it robust to robot failures and communication
    disruptions. If a robot is lost, the others naturally close the gap.
    """

    def __init__(self, desired_distance=2.0, n_robots=5):
        self.desired_distance = desired_distance
        self.n_robots = n_robots

    def compute_velocity(self, robot_idx, all_positions, goal_position):
        """Compute velocity for one robot based on local behaviors.

        Three behaviors, weighted and summed:
        1. Goal attraction: move toward the mission goal
        2. Inter-robot repulsion: maintain minimum distance from neighbors
        3. Inter-robot attraction: stay within communication range
        """
        my_pos = all_positions[robot_idx]
        velocity = np.zeros(2)

        # Behavior 1: Goal attraction
        goal_vec = goal_position - my_pos
        goal_dist = np.linalg.norm(goal_vec)
        if goal_dist > 0.01:
            velocity += 1.0 * goal_vec / goal_dist

        # Behavior 2 & 3: Inter-robot forces
        for j in range(len(all_positions)):
            if j == robot_idx:
                continue

            diff = all_positions[j] - my_pos
            dist = np.linalg.norm(diff)
            if dist < 0.01:
                continue

            direction = diff / dist

            if dist < self.desired_distance * 0.8:
                # Too close — repel
                repulsion_strength = 3.0 * (1.0 - dist / (self.desired_distance * 0.8))
                velocity -= repulsion_strength * direction
            elif dist > self.desired_distance * 1.2:
                # Too far — attract
                attraction_strength = 1.0 * (dist / self.desired_distance - 1.2)
                velocity += attraction_strength * direction
            # In the "sweet spot": no force

        # Limit velocity
        speed = np.linalg.norm(velocity)
        max_speed = 1.0
        if speed > max_speed:
            velocity = velocity / speed * max_speed

        return velocity
```

---

## 4. 합의 알고리즘(Consensus Algorithms)

### 4.1 합의 문제(The Consensus Problem)

**합의(consensus)**는 모든 로봇들이 공통 값(위치, 방향, 결정)에 동의함을 의미한다. 각 로봇은 자체 추정값에서 시작하고 이웃과 통신하여 반복적으로 업데이트한다.

### 4.2 선형 합의 프로토콜(Linear Consensus Protocol)

통신 그래프 $\mathcal{G}$를 가진 $N$대의 로봇 네트워크에 대해:

$$x_i(k+1) = x_i(k) + \epsilon \sum_{j \in \mathcal{N}_i} (x_j(k) - x_i(k))$$

여기서 $\mathcal{N}_i$는 로봇 $i$의 이웃 집합이고 $\epsilon > 0$은 스텝 크기다.

이것은 모든 초기값의 **평균**으로 수렴한다: $x^* = \frac{1}{N}\sum_{i=1}^N x_i(0)$.

**수렴 조건**: 통신 그래프가 **연결(connected)**되어 있어야 하고 $\epsilon < 1/d_{max}$ 이어야 한다 (여기서 $d_{max}$는 최대 차수(degree)). 왜 그래프가 연결되어야 하는가? 합의는 이웃 간 통신을 통해 정보를 전파한다. 그래프가 끊어져 있으면(두 개 이상의 고립된 그룹) 한 그룹의 정보가 다른 그룹에 절대 도달할 수 없어, 전역 평균이 아닌 각 그룹 내 평균으로 수렴한다. 연결성은 모든 로봇의 초기값이 결국 다른 모든 로봇에 영향을 미치도록 보장하여, 진정한 전역 합의를 가능하게 한다.

```python
class ConsensusProtocol:
    """Distributed average consensus.

    Why consensus? Many multi-robot tasks require agreement:
    - Average position (rendezvous point)
    - Average sensor reading (distributed estimation)
    - Binary decision (go left or go right?)
    - Formation center computation (without central coordinator)

    Consensus achieves this without any central computer — each robot
    only talks to its immediate neighbors, yet the entire network
    converges to agreement.
    """

    def __init__(self, adjacency_matrix, epsilon=0.1):
        self.adj = np.array(adjacency_matrix)
        self.n = len(self.adj)
        self.epsilon = epsilon

        # Verify convergence condition
        max_degree = self.adj.sum(axis=1).max()
        if epsilon >= 1.0 / max_degree:
            print(f"Warning: epsilon={epsilon} may not converge. "
                  f"Max degree = {max_degree}, need epsilon < {1.0/max_degree:.3f}")

    def step(self, values):
        """One consensus iteration.

        Each robot updates its value by moving toward the average
        of its neighbors' values. After enough iterations, all
        values converge to the global average.
        """
        new_values = values.copy()
        for i in range(self.n):
            neighbor_sum = 0.0
            n_neighbors = 0
            for j in range(self.n):
                if self.adj[i, j] > 0:
                    neighbor_sum += values[j] - values[i]
                    n_neighbors += 1
            new_values[i] += self.epsilon * neighbor_sum
        return new_values

    def run(self, initial_values, max_iterations=100, tolerance=1e-6):
        """Run consensus until convergence.

        Why iterate? In a fully connected graph, one step would
        suffice. But in sparse graphs (each robot talks to 2-3
        neighbors), information propagates slowly across the network.
        The number of iterations needed scales with the graph diameter
        (longest shortest path between any two nodes).
        """
        values = np.array(initial_values, dtype=float)
        history = [values.copy()]

        for k in range(max_iterations):
            new_values = self.step(values)

            if np.max(np.abs(new_values - values)) < tolerance:
                break

            values = new_values
            history.append(values.copy())

        return values, history
```

### 4.3 랑데부를 위한 합의(Consensus for Rendezvous)

자연스러운 응용: 로봇들이 자신의 위치에 대해 합의를 실행하여 만남의 장소에 동의한다.

```python
def consensus_rendezvous(positions, adjacency_matrix, n_steps=50):
    """Robots converge to their centroid using consensus.

    Why not just compute the centroid directly? That would require
    one robot to know ALL positions — a centralized approach. With
    consensus, each robot only needs to communicate with neighbors.
    The centroid emerges from local interactions, making the system
    robust to communication failures and scalable to large teams.
    """
    consensus = ConsensusProtocol(adjacency_matrix, epsilon=0.1)

    x_values = positions[:, 0].copy()
    y_values = positions[:, 1].copy()

    x_history = [x_values.copy()]
    y_history = [y_values.copy()]

    for step in range(n_steps):
        x_values = consensus.step(x_values)
        y_values = consensus.step(y_values)
        x_history.append(x_values.copy())
        y_history.append(y_values.copy())

    # All robots converge to the centroid
    final_positions = np.stack([x_values, y_values], axis=1)
    return final_positions, x_history, y_history
```

---

## 5. 군집 지능(Swarm Intelligence)

### 5.1 Reynolds 군집 비행(Reynolds Flocking)

Craig Reynolds의 1986년 boids 모델은 세 가지 간단한 규칙으로 놀랍도록 사실적인 군집 비행 행동을 만들어낸다:

1. **분리(separation)**: 근처의 군집 동료와 붐비지 않도록 회피
2. **정렬(alignment)**: 근처 군집 동료의 평균 방향으로 조종
3. **응집력(cohesion)**: 근처 군집 동료의 평균 위치로 조종

```python
class ReynoldsFlocking:
    """Reynolds flocking model for swarm robot coordination.

    Why local rules? Each robot only needs information about its
    immediate neighbors (within sensor range). There is no central
    controller, no global communication, no designated leader.
    Yet the swarm exhibits complex collective behavior: synchronized
    motion, obstacle avoidance, splitting and merging around barriers.

    This is the defining characteristic of swarm intelligence:
    global complexity from local simplicity.
    """

    def __init__(self, n_robots, perception_radius=3.0,
                 separation_dist=1.0, max_speed=1.0):
        self.n = n_robots
        self.perception_radius = perception_radius
        self.separation_dist = separation_dist
        self.max_speed = max_speed

        # 행동 가중치 — 분리(separation)가 의도적으로 가장 높다 (2.0 vs 1.0).
        # 충돌 방지는 하드 안전 제약이고, 정렬과 응집력은 소프트 선호이기
        # 때문이다. 모든 가중치가 동일하면 응집력이 로봇을 가까이 당기는 힘이
        # 근거리에서 분리를 극복하여 충돌을 유발할 수 있다. 2:1 비율은 로봇이
        # 가까울 때 회피가 항상 지배하도록 보장한다.
        self.w_separation = 2.0   # 최우선: 충돌 회피
        self.w_alignment = 1.0    # 이웃 속도 매칭
        self.w_cohesion = 1.0     # 그룹 유지
        self.w_goal = 0.5         # 선택사항: 목표를 향해 조종

    def compute_velocity(self, positions, velocities, robot_idx, goal=None):
        """Compute velocity for one robot using Reynolds rules.

        Why weight separation highest? Collision avoidance is the
        most critical behavior. Without strong separation, robots
        crash into each other, especially when the cohesion force
        pulls them together. The balance of weights determines the
        flock's 'personality' — tight vs. loose, cautious vs. bold.
        """
        my_pos = positions[robot_idx]
        my_vel = velocities[robot_idx]

        # Find neighbors within perception radius
        neighbors = []
        for j in range(self.n):
            if j == robot_idx:
                continue
            dist = np.linalg.norm(positions[j] - my_pos)
            if dist < self.perception_radius:
                neighbors.append(j)

        if len(neighbors) == 0:
            # No neighbors — just steer toward goal if available
            if goal is not None:
                return self._steer_toward(my_pos, goal, my_vel)
            return my_vel

        # Rule 1: Separation — steer away from close neighbors
        separation = np.zeros(2)
        for j in neighbors:
            diff = my_pos - positions[j]
            dist = np.linalg.norm(diff)
            if dist < self.separation_dist and dist > 0.01:
                # Inversely proportional to distance (closer = stronger repulsion)
                separation += diff / (dist**2)

        # Rule 2: Alignment — match average velocity of neighbors
        avg_vel = np.mean([velocities[j] for j in neighbors], axis=0)
        alignment = avg_vel - my_vel

        # Rule 3: Cohesion — steer toward center of neighbors
        center = np.mean([positions[j] for j in neighbors], axis=0)
        cohesion = center - my_pos

        # Combine behaviors
        velocity = (self.w_separation * separation
                    + self.w_alignment * alignment
                    + self.w_cohesion * cohesion)

        # Optional goal attraction
        if goal is not None:
            goal_vec = goal - my_pos
            goal_dist = np.linalg.norm(goal_vec)
            if goal_dist > 0.1:
                velocity += self.w_goal * goal_vec / goal_dist

        # Limit speed
        speed = np.linalg.norm(velocity)
        if speed > self.max_speed:
            velocity = velocity / speed * self.max_speed

        return velocity

    def _steer_toward(self, pos, target, current_vel):
        """Steer toward a target position."""
        desired = target - pos
        dist = np.linalg.norm(desired)
        if dist < 0.01:
            return np.zeros(2)
        desired = desired / dist * self.max_speed
        steer = desired - current_vel
        return steer

    def simulate(self, initial_positions, goal=None, dt=0.05, n_steps=500):
        """Run flocking simulation. Swarm behavior is emergent — simulation
        is the only way to understand what the collective will do."""
        positions = np.array(initial_positions, dtype=float)
        velocities = np.random.randn(self.n, 2) * 0.1
        history = [positions.copy()]
        for _ in range(n_steps):
            new_vel = np.array([self.compute_velocity(positions, velocities, i, goal)
                                for i in range(self.n)])
            velocities = new_vel
            positions += velocities * dt
            history.append(positions.copy())
        return np.array(history)
```

### 5.2 개미 군집 최적화(Ant Colony Optimization, ACO)

**ACO**는 개미들이 페로몬(pheromone) 흔적을 사용하여 먹이로의 최단 경로를 찾는 방식에서 영감을 받았다:

1. 개미들이 무작위로 탐색하며 경로에 페로몬을 분비한다
2. 더 짧은 경로에 더 많은 페로몬이 쌓인다 (개미들이 더 빨리 돌아옴)
3. 미래의 개미들은 페로몬이 더 많은 경로를 선호한다
4. 양성 피드백 루프(positive feedback loop)가 최단 경로에 트래픽을 집중시킨다

```python
class AntColonyOptimization:
    """Ant Colony Optimization for multi-robot path planning.

    Why ACO for robots? In environments where the optimal path changes
    frequently (warehouse with moving obstacles, road network with
    traffic), ACO provides continuous, adaptive path optimization.
    Robot 'pheromones' can be implemented as shared information in a
    distributed database or ad-hoc wireless network.
    """

    def __init__(self, n_nodes, n_ants=20, alpha=1.0, beta=2.0, rho=0.1):
        self.n_nodes = n_nodes
        self.n_ants = n_ants
        self.alpha = alpha   # Pheromone importance
        self.beta = beta     # Distance importance
        self.rho = rho       # Evaporation rate

        # Pheromone matrix (initially uniform)
        self.pheromone = np.ones((n_nodes, n_nodes))

    def select_next(self, current, visited, distances):
        """Probabilistic selection of next node.

        Why probabilistic? Deterministic greedy selection (always follow
        strongest pheromone) would converge prematurely to a suboptimal
        path. The randomness ensures exploration of alternative routes.
        The balance between pheromone (exploitation) and distance
        (greedy heuristic) is controlled by alpha and beta.
        """
        unvisited = [j for j in range(self.n_nodes) if j not in visited]
        if not unvisited:
            return None

        probabilities = np.zeros(len(unvisited))
        for idx, j in enumerate(unvisited):
            tau = self.pheromone[current, j] ** self.alpha
            eta = (1.0 / max(distances[current, j], 1e-10)) ** self.beta
            probabilities[idx] = tau * eta

        probabilities /= probabilities.sum()
        chosen_idx = np.random.choice(len(unvisited), p=probabilities)
        return unvisited[chosen_idx]

    def run(self, distances, start, goal, n_iterations=100):
        """Run ACO: for each iteration, ants construct paths, then
        pheromone evaporates and successful ants deposit new pheromone.
        Over iterations, pheromone concentrates on shorter paths.
        """
        best_path, best_cost = None, float('inf')

        for _ in range(n_iterations):
            paths, costs = [], []
            for _ in range(self.n_ants):
                path, visited, current = [start], {start}, start
                while current != goal:
                    nxt = self.select_next(current, visited, distances)
                    if nxt is None: break
                    path.append(nxt); visited.add(nxt); current = nxt
                if current == goal:
                    cost = sum(distances[path[k], path[k+1]] for k in range(len(path)-1))
                    paths.append(path); costs.append(cost)
                    if cost < best_cost: best_cost, best_path = cost, path

            self.pheromone *= (1 - self.rho)  # Evaporation
            for path, cost in zip(paths, costs):
                deposit = 1.0 / cost
                for k in range(len(path) - 1):
                    self.pheromone[path[k], path[k+1]] += deposit

        return best_path, best_cost
```

### 5.3 입자 군집 최적화(Particle Swarm Optimization, PSO)

**PSO**는 새 떼와 물고기 떼의 사회적 행동에서 영감을 받은 연속 최적화(continuous optimization)를 위한 방법이다:

$$v_i(t+1) = w \cdot v_i(t) + c_1 r_1 (p_i - x_i(t)) + c_2 r_2 (g - x_i(t))$$
$$x_i(t+1) = x_i(t) + v_i(t+1)$$

여기서:
- $p_i$는 입자 $i$의 개인 최적 위치(personal best position)
- $g$는 모든 입자에 걸친 전역 최적 위치(global best position)
- $w$는 관성 가중치(inertia weight), $c_1, c_2$는 가속 계수(acceleration coefficients)
- $r_1, r_2$는 $[0, 1]$의 무작위 값

```python
def pso_step(positions, velocities, personal_best_pos, global_best_pos,
             w=0.7, c1=1.5, c2=1.5):
    """One step of Particle Swarm Optimization.

    Why PSO for robotics? PSO can solve optimization problems in
    multi-robot systems: coverage optimization, parameter tuning,
    and environmental monitoring. Each robot acts as a 'particle'
    exploring the solution space.

    The velocity update balances three forces:
    1. Inertia (w): continue in current direction (exploration)
    2. Cognitive (c1): attraction to personal best (individual memory)
    3. Social (c2): attraction to global best (collective knowledge)
    """
    n, dim = positions.shape
    r1 = np.random.rand(n, dim)
    r2 = np.random.rand(n, dim)

    cognitive = c1 * r1 * (personal_best_pos - positions)
    social = c2 * r2 * (global_best_pos - positions)

    new_velocities = w * velocities + cognitive + social
    new_positions = positions + new_velocities

    return new_positions, new_velocities
```

---

## 6. 통신 아키텍처(Communication Architectures)

### 6.1 중앙 집중식 통신(Centralized Communication)

모든 로봇이 중앙 서버를 통해 통신한다:

```
       ┌─────────┐
       │ Central  │
       │ Server   │
       └─┬─┬─┬─┬─┘
         │ │ │ │
    ┌────┘ │ │ └────┐
    R1    R2 R3    R4
```

**장점**: 전체 정보 이용 가능, 전역 최적 결정
**단점**: 단일 장애 지점(single point of failure), 통신 병목(bottleneck), 확장 불가

### 6.2 분산식 통신(Decentralized Communication)

로봇들이 범위 내의 이웃과만 통신한다:

```
    R1 ── R2 ── R3
     \         /
      R4 ── R5
```

**장점**: 단일 장애 지점 없음, 자연스럽게 확장, 제한된 범위에서 작동
**단점**: 정보가 느리게 전파, 지역 결정이 차선책일 수 있음

### 6.3 애드혹 메시 네트워크(Ad-Hoc Mesh Networks)

로봇들이 자기 조직화(self-organizing) 통신 네트워크를 형성한다. 핵심 요구 사항은 **연결성(connectivity)**이다 — 분산 알고리즘(합의, 대형 제어, 작업 할당)이 작동하려면 그래프가 연결되어 있어야 한다. 그래프가 끊어지면 군집이 조율할 수 없는 독립적인 하위 그룹으로 분리된다.

```python
def communication_graph(positions, comm_range):
    """Build adjacency matrix from robot positions and comm range.

    Why model communication explicitly? In real deployments,
    robots may be out of range, signals blocked by obstacles,
    and bandwidth limited. Algorithms must work within these constraints.
    """
    n = len(positions)
    adjacency = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(positions[i] - positions[j]) <= comm_range:
                adjacency[i, j] = adjacency[j, i] = 1
    return adjacency
```

**연결성 제약 계획(connectivity-constrained planning)**은 이동 중에 통신 그래프가 연결된 상태를 유지해야 한다는 제약을 추가한다 — 로봇은 그렇게 하는 것이 네트워크를 끊지 않을 경우에만 이동할 수 있다.

---

## 7. 응용 사례

| 응용 | 핵심 과제 | 접근 방식 |
|------|----------|----------|
| **수색 및 구조(search and rescue)** | 재난 현장 빠르게 커버, 통신 유지 | 프런티어 탐색(frontier exploration), 릴레이 체인, 정보 기반 탐색 |
| **창고 물류(warehouse logistics)** | 500대 이상 로봇 교통 관리 | 다중 에이전트 경로 찾기(MAPF), 지역 충돌 해결이 있는 중앙 집중식 계획 |
| **환경 모니터링(environmental monitoring)** | 대규모 면적의 적응형 커버리지 | 로봇들이 정보가 많은 지역에 집중; 출처 탐색을 위한 기울기 추적 |
| **협력 조작(cooperative manipulation)** | 여러 로봇이 큰 물체 운반 | 대형 제어 + 분산 힘 제어 + 원하는 궤적에 대한 합의 |

---

## 8. 확장성 및 실용적 고려 사항

### 8.1 확장 법칙(Scaling Laws)

| 알고리즘 | 통신 | 로봇당 계산 | 확장 한계 |
|---------|------|-----------|----------|
| 중앙 집중식 계획 | $O(N^2)$ | $O(1)$ | 약 50대 |
| 경매 기반 | $O(N)$ | $O(M)$ | 약 500대 |
| 합의 | 스텝당 $O(d)$ | $O(d)$ | 약 10,000대 |
| Reynolds 군집 비행 | $O(k)$ ($k$명의 이웃) | $O(k)$ | 1,000,000대 이상 |

여기서 $N$ = 로봇 수, $M$ = 작업 수, $d$ = 노드 차수(degree).

### 8.2 이질적 팀(Heterogeneous Teams)

실제 배포에서는 종종 다른 유형의 로봇들이 함께 작동한다: 항공 드론은 전체 조사용, 지상 로봇은 건물 진입용, 통신 릴레이는 연결성 유지용. **역량 인식 작업 할당(capability-aware task allocation)**은 각 로봇의 강점(감지 범위, 적재 용량, 이동성)에 기반하여 작업을 할당한다.

---

## 요약

| 개념 | 핵심 아이디어 |
|------|--------------|
| 다중 로봇 장점 | 병렬성, 강건성, 공간 분산, 보완적 역량 |
| 작업 할당(task allocation) | 최적으로 로봇에게 작업 할당; 확장성을 위한 경매 기반, 최적성을 위한 헝가리안 |
| 선두-추종(leader-follower) | 가장 단순한 대형; 단일 장애 지점 |
| 가상 구조(virtual structure) | 대형이 강체로 이동; 조율되지만 중앙 집중식 |
| 행동 기반 대형(behavior-based formation) | 지역 규칙이 출현하는 대형 만들어냄; 강건하고 분산식 |
| 합의(consensus) | 이웃 통신을 통한 분산 동의; 평균으로 수렴 |
| Reynolds 군집 비행 | 세 가지 규칙 (분리, 정렬, 응집력)이 복잡한 군집 행동 만들어냄 |
| ACO | 페로몬 기반 최단 경로 찾기; 적응적, 분산식 |
| PSO | 군집 기반 연속 최적화; 개인 및 전역 최적 끌개 |
| 통신 아키텍처 | 중앙 집중식 (최적이지만 취약) vs. 분산식 (강건하지만 차선책) |
| 연결성(connectivity) | 분산 알고리즘이 작동하려면 통신 그래프가 연결되어야 함 |

---

## 연습문제

1. **경매 기반 할당**: 순차 경매 할당기를 구현하라. 무작위 위치에 5대의 로봇과 8개의 작업이 있는 시나리오를 생성하라. 경매를 실행하고 할당을 시각화하라. 총 비용 (거리의 합)을 헝가리안 알고리즘의 최적 할당과 비교하라. 경매 결과가 최적에 얼마나 가까운가?

2. **선두-추종 대형**: 다이아몬드 패턴의 4대 로봇에 대한 선두-추종 대형 제어를 구현하라. 선두는 8자 궤적을 따른다. 모든 로봇의 궤적을 그리고 시간에 따른 대형 오차 (원하는 오프셋에서의 편차)를 측정하라. 선두가 급격한 방향 전환을 할 때 무슨 일이 일어나는가?

3. **Reynolds 군집 비행**: 30대의 로봇에 대한 Reynolds 군집 비행 모델을 구현하라. 20x20 면적의 무작위 위치에서 시작하라. 시뮬레이션을 실행하고 시간에 따른 군집을 시각화하라. 세 가지 가중치 매개변수를 실험하라: (a) 분리 가중치 증가 — 무슨 일이 일어나는가? (b) 응집력 가중치 증가 — 무슨 일이 일어나는가? (c) 정렬 가중치를 0으로 설정 — 무엇이 변하는가?

4. **합의 수렴**: 10대의 로봇에 대한 선형 합의 프로토콜을 구현하라. 세 가지 통신 토폴로지로 테스트하라: (a) 완전 연결(fully connected), (b) 링(ring), (c) 스타(star). 각 토폴로지에 대한 수렴까지의 반복 횟수를 비교하라. 그래프 연결성과 수렴 속도 사이의 관계는 무엇인가?

5. **다중 로봇 커버리지**: 6대의 로봇이 20x20 면적을 최적으로 커버하기 위한 분산 알고리즘을 설계하라. 각 로봇의 감지 반경은 3 m다. Voronoi 기반 접근 방식을 사용하라: 각 로봇이 자신의 Voronoi 셀의 무게 중심으로 이동한다. 시뮬레이션하고 시간에 따른 커버리지 비율을 측정하라.

---

## 참고 문헌

- Bullo, F. et al. *Distributed Control of Robotic Networks*. Princeton University Press, 2009. (수학적 기초)
- Brambilla, M. et al. "Swarm Robotics: A Review from the Swarm Engineering Perspective." *Swarm Intelligence*, 2013. (포괄적인 군집 로보틱스 서베이)
- Reynolds, C. "Flocks, Herds, and Schools: A Distributed Behavioral Model." *SIGGRAPH*, 1987. (원본 군집 비행 논문)
- Dorigo, M. et al. "Ant Colony Optimization: A New Meta-Heuristic." *IEEE CEC*, 1999. (ACO)
- Ren, W. and Beard, R. *Distributed Consensus in Multi-vehicle Cooperative Control*. Springer, 2008. (합의 알고리즘)
- Khamis, A. et al. "Multi-robot Task Allocation: A Review." *Robotics and Autonomous Systems*, 2015. (작업 할당 서베이)

---

[← 이전: 로보틱스를 위한 강화학습](15_RL_for_Robotics.md)
