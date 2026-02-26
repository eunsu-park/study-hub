# 16. Multi-Robot Systems and Swarms

[← Previous: Reinforcement Learning for Robotics](15_RL_for_Robotics.md)

---

## Learning Objectives

1. Distinguish between single-robot and multi-robot system paradigms and identify when multi-robot approaches are beneficial
2. Formulate and solve task allocation problems using auction-based and optimization-based methods
3. Implement formation control strategies: leader-follower, virtual structure, and behavior-based
4. Understand consensus algorithms and their role in distributed coordination
5. Describe swarm intelligence principles: Reynolds flocking, ant colony optimization, and particle swarm optimization
6. Analyze communication architectures and their trade-offs for multi-robot systems

---

Throughout this course, we have focused on a single robot — one manipulator, one mobile platform, one set of sensors. But many real-world tasks are better served by teams of robots working together. A single robot searching a collapsed building after an earthquake would take hours; a swarm of 50 small robots could cover the same area in minutes. A single warehouse robot moves one package at a time; a fleet of 500 robots (like Amazon's Kiva system) handles millions of packages per day.

Multi-robot systems introduce fundamentally new challenges: How do robots divide work? How do they avoid collisions with each other? How do they share information when communication is limited? And perhaps most fascinatingly, how can complex coordinated behavior emerge from simple individual rules?

This lesson explores these questions, from formal task allocation algorithms to the beautiful simplicity of swarm intelligence, where collective behavior emerges without any central coordinator.

> **Analogy**: A swarm of robots is like a flock of birds — no single leader, yet complex coordinated behavior emerges from simple local rules. Each bird follows three rules: stay close to neighbors (cohesion), don't collide with neighbors (separation), and move in the same direction as neighbors (alignment). From these three simple rules, thousands of birds create breathtaking formations without any bird "planning" the formation. Swarm robotics applies the same principle: simple local rules, complex global behavior.

---

## 1. Single-Robot vs. Multi-Robot Systems

### 1.1 Why Multiple Robots?

| Advantage | Description | Example |
|-----------|-------------|---------|
| **Parallelism** | Multiple robots work simultaneously | 50 robots clean a building in 1/50th the time |
| **Robustness** | System tolerates individual failures | If one search drone crashes, 19 others continue |
| **Spatial distribution** | Cover large areas simultaneously | Environmental monitoring across 100 km² |
| **Complementary capabilities** | Different robots for different subtasks | Ground robot + aerial drone for inspection |
| **Cost efficiency** | Many simple robots cheaper than one complex robot | Swarm of $100 drones vs. one $50,000 robot |

### 1.2 Challenges

| Challenge | Description |
|-----------|-------------|
| **Coordination** | Robots must avoid conflicting actions |
| **Communication** | Bandwidth limits, latency, message loss |
| **Task allocation** | Who does what? NP-hard in general |
| **Collision avoidance** | Robots must not collide with each other |
| **Scalability** | Algorithms must work for 10 robots and 10,000 |
| **Heterogeneity** | Different robots may have different capabilities |

### 1.3 Taxonomy of Multi-Robot Systems

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

## 2. Task Allocation

### 2.1 The Problem

Given:
- $N$ robots with capabilities $C_1, C_2, \ldots, C_N$
- $M$ tasks with requirements $T_1, T_2, \ldots, T_M$
- Cost function $c_{ij}$ = cost for robot $i$ to perform task $j$

Find: Assignment that minimizes total cost while satisfying all constraints.

This is a variant of the **assignment problem**, which is NP-hard in general for multi-robot, multi-task scenarios.

### 2.2 Auction-Based Allocation

**Market-based approaches** treat task allocation as an auction where robots bid on tasks:

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

### 2.3 Optimization-Based Allocation

For smaller teams where computation is affordable, we can solve the optimal assignment:

$$\min \sum_{i=1}^{N} \sum_{j=1}^{M} c_{ij} x_{ij}$$

subject to:
$$\sum_{j=1}^{M} x_{ij} \leq 1 \quad \forall i \quad \text{(each robot does at most one task)}$$
$$\sum_{i=1}^{N} x_{ij} = 1 \quad \forall j \quad \text{(each task assigned to exactly one robot)}$$
$$x_{ij} \in \{0, 1\}$$

The **Hungarian algorithm** solves this optimally in $O(\max(N,M)^3)$ time.

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

## 3. Formation Control

### 3.1 Why Formations?

Robots in formation can:
- Transport large objects cooperatively
- Create sensor arrays with known geometry
- Maintain communication network topology
- Provide area coverage guarantees

### 3.2 Leader-Follower

The simplest formation strategy: one robot leads, others maintain a desired offset relative to the leader.

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

### 3.3 Virtual Structure

The **virtual structure** approach treats the formation as a rigid body. The virtual structure moves, and each robot tracks its assigned point on the structure.

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

### 3.4 Behavior-Based Formation

Instead of a rigid structure, each robot follows local behavioral rules that produce emergent formation patterns:

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

## 4. Consensus Algorithms

### 4.1 The Consensus Problem

**Consensus** means all robots agree on a common value (position, heading, decision). Each robot starts with its own estimate and iteratively updates by communicating with neighbors.

### 4.2 Linear Consensus Protocol

For a network of $N$ robots with communication graph $\mathcal{G}$:

$$x_i(k+1) = x_i(k) + \epsilon \sum_{j \in \mathcal{N}_i} (x_j(k) - x_i(k))$$

where $\mathcal{N}_i$ is the set of neighbors of robot $i$ and $\epsilon > 0$ is the step size.

This converges to the **average** of all initial values: $x^* = \frac{1}{N}\sum_{i=1}^N x_i(0)$.

**Convergence condition**: The communication graph must be **connected** and $\epsilon < 1/d_{max}$ where $d_{max}$ is the maximum degree. Why must the graph be connected? Consensus propagates information through neighbor-to-neighbor communication. If the graph is disconnected (two or more isolated groups), information from one group can never reach the other, so they converge to different values — the average within each group, not the global average. Connectivity guarantees that every robot's initial value eventually influences every other robot, enabling true global agreement.

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

### 4.3 Consensus for Rendezvous

A natural application: robots agree on a meeting point by running consensus on their positions.

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

## 5. Swarm Intelligence

### 5.1 Reynolds Flocking

Craig Reynolds' 1986 boids model produces remarkably realistic flocking behavior from three simple rules:

1. **Separation**: Avoid crowding nearby flockmates
2. **Alignment**: Steer toward the average heading of nearby flockmates
3. **Cohesion**: Steer toward the average position of nearby flockmates

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

        # Behavior weights — separation is intentionally highest (2.0 vs 1.0)
        # because collision prevention is a hard safety constraint, while
        # alignment and cohesion are soft preferences. If all weights were
        # equal, the cohesion force pulling robots together could overcome
        # separation at close range, causing collisions. The 2:1 ratio
        # ensures that avoidance always dominates when robots are close.
        self.w_separation = 2.0   # Highest priority: avoid collisions
        self.w_alignment = 1.0    # Match neighbor velocities
        self.w_cohesion = 1.0     # Stay with the group
        self.w_goal = 0.5         # Optional: steer toward a goal

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

### 5.2 Ant Colony Optimization (ACO)

**ACO** is inspired by how ants find shortest paths to food using pheromone trails:

1. Ants randomly explore, depositing pheromone on their path
2. Shorter paths accumulate more pheromone (ants return faster)
3. Future ants prefer paths with more pheromone
4. Positive feedback loop concentrates traffic on the shortest path

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

### 5.3 Particle Swarm Optimization (PSO)

**PSO** is inspired by the social behavior of bird flocking and fish schooling, used for continuous optimization:

$$v_i(t+1) = w \cdot v_i(t) + c_1 r_1 (p_i - x_i(t)) + c_2 r_2 (g - x_i(t))$$
$$x_i(t+1) = x_i(t) + v_i(t+1)$$

where:
- $p_i$ is the personal best position of particle $i$
- $g$ is the global best position across all particles
- $w$ is the inertia weight, $c_1, c_2$ are acceleration coefficients
- $r_1, r_2$ are random values in $[0, 1]$

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

## 6. Communication Architectures

### 6.1 Centralized Communication

All robots communicate through a central server:

```
       ┌─────────┐
       │ Central  │
       │ Server   │
       └─┬─┬─┬─┬─┘
         │ │ │ │
    ┌────┘ │ │ └────┐
    R1    R2 R3    R4
```

**Pros**: Full information available, globally optimal decisions
**Cons**: Single point of failure, communication bottleneck, does not scale

### 6.2 Decentralized Communication

Robots communicate only with neighbors within range:

```
    R1 ── R2 ── R3
     \         /
      R4 ── R5
```

**Pros**: No single point of failure, scales naturally, works with limited range
**Cons**: Information propagates slowly, local decisions may be suboptimal

### 6.3 Ad-Hoc Mesh Networks

Robots form a self-organizing communication network. A key requirement is **connectivity** — the graph must be connected for distributed algorithms (consensus, formation control, task allocation) to work. If the graph disconnects, the swarm splits into independent sub-groups that cannot coordinate.

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

**Connectivity-constrained planning** adds the constraint that the communication graph must remain connected during motion — robots can only move if doing so does not disconnect the network.

---

## 7. Applications

| Application | Key Challenge | Approach |
|-------------|--------------|----------|
| **Search and rescue** | Cover disaster site quickly, maintain comms | Frontier exploration, relay chains, information-driven search |
| **Warehouse logistics** | Traffic management for 500+ robots | Multi-Agent Path Finding (MAPF), centralized planning with local conflict resolution |
| **Environmental monitoring** | Adaptive coverage of large areas | Robots concentrate in high-information areas; gradient-following for source seeking |
| **Cooperative manipulation** | Multiple robots carrying a large object | Formation control + distributed force control + consensus on desired trajectory |

---

## 8. Scalability and Practical Considerations

### 8.1 Scaling Laws

| Algorithm | Communication | Computation per Robot | Scales To |
|-----------|--------------|----------------------|-----------|
| Centralized planning | $O(N^2)$ | $O(1)$ | ~50 robots |
| Auction-based | $O(N)$ | $O(M)$ | ~500 robots |
| Consensus | $O(d)$ per step | $O(d)$ | ~10,000 robots |
| Reynolds flocking | $O(k)$ (k neighbors) | $O(k)$ | ~1,000,000+ |

where $N$ = number of robots, $M$ = number of tasks, $d$ = node degree.

### 8.2 Heterogeneous Teams

Real-world deployments often use different robot types working together: aerial drones for overhead survey, ground robots for building entry, and communication relays to maintain connectivity. **Capability-aware task allocation** assigns tasks based on each robot's strengths (sensing range, payload capacity, mobility).

---

## Summary

| Concept | Key Idea |
|---------|----------|
| Multi-robot advantages | Parallelism, robustness, spatial distribution, complementary capabilities |
| Task allocation | Assign tasks to robots optimally; auction-based for scalability, Hungarian for optimality |
| Leader-follower | Simplest formation; single point of failure |
| Virtual structure | Formation moves as rigid body; coordinated but centralized |
| Behavior-based formation | Local rules produce emergent formations; robust and decentralized |
| Consensus | Distributed agreement through neighbor communication; converges to average |
| Reynolds flocking | Three rules (separation, alignment, cohesion) produce complex swarm behavior |
| ACO | Pheromone-based shortest path finding; adaptive, distributed |
| PSO | Swarm-based continuous optimization; personal and global best attractors |
| Communication architecture | Centralized (optimal but fragile) vs. decentralized (robust but suboptimal) |
| Connectivity | Communication graph must be connected for distributed algorithms to work |

---

## Exercises

1. **Auction-based allocation**: Implement the sequential auction allocator. Create a scenario with 5 robots and 8 tasks at random positions. Run the auction and visualize the assignments. Compare the total cost (sum of distances) with the optimal assignment from the Hungarian algorithm. How close is the auction result to optimal?

2. **Leader-follower formation**: Implement leader-follower formation control for 4 robots in a diamond pattern. The leader follows a figure-eight trajectory. Plot the trajectories of all robots and measure the formation error (deviation from desired offsets) over time. What happens when the leader makes sharp turns?

3. **Reynolds flocking**: Implement the Reynolds flocking model for 30 robots. Start with random positions in a 20x20 area. Run the simulation and visualize the swarm over time. Experiment with the three weight parameters: (a) increase separation weight — what happens? (b) increase cohesion weight — what happens? (c) set alignment weight to zero — what changes?

4. **Consensus convergence**: Implement the linear consensus protocol for 10 robots. Test with three communication topologies: (a) fully connected, (b) ring, (c) star. Compare the number of iterations to converge for each topology. What is the relationship between graph connectivity and convergence speed?

5. **Multi-robot coverage**: Design a distributed algorithm for 6 robots to achieve optimal coverage of a 20x20 area. Each robot has a sensing radius of 3 m. Use a Voronoi-based approach: each robot moves to the centroid of its Voronoi cell. Simulate and measure the coverage percentage over time.

---

## Further Reading

- Bullo, F. et al. *Distributed Control of Robotic Networks*. Princeton University Press, 2009. (Mathematical foundations)
- Brambilla, M. et al. "Swarm Robotics: A Review from the Swarm Engineering Perspective." *Swarm Intelligence*, 2013. (Comprehensive swarm robotics survey)
- Reynolds, C. "Flocks, Herds, and Schools: A Distributed Behavioral Model." *SIGGRAPH*, 1987. (Original flocking paper)
- Dorigo, M. et al. "Ant Colony Optimization: A New Meta-Heuristic." *IEEE CEC*, 1999. (ACO)
- Ren, W. and Beard, R. *Distributed Consensus in Multi-vehicle Cooperative Control*. Springer, 2008. (Consensus algorithms)
- Khamis, A. et al. "Multi-robot Task Allocation: A Review." *Robotics and Autonomous Systems*, 2015. (Task allocation survey)

---

[← Previous: Reinforcement Learning for Robotics](15_RL_for_Robotics.md)
