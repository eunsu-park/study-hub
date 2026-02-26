# Motion Planning

[← Previous: Robot Dynamics](06_Robot_Dynamics.md) | [Next: Trajectory Planning →](08_Trajectory_Planning.md)

## Learning Objectives

1. Define the configuration space (C-space) and explain how obstacles in workspace map to C-space obstacles
2. Implement the artificial potential field method and identify its local minima problem
3. Construct a Probabilistic Roadmap (PRM) for multi-query planning scenarios
4. Build a Rapidly-exploring Random Tree (RRT) for single-query planning, and understand the RRT* optimality improvement
5. Compare planners on completeness, optimality, and computational complexity
6. Apply practical post-processing techniques including path smoothing and shortcutting

---

## Why This Matters

Kinematics and dynamics answer "can the robot reach this pose?" and "what torques are needed?" But they say nothing about *how to get there without hitting anything*. Motion planning is the problem of finding a collision-free path from a start configuration to a goal configuration — the bridge between high-level task commands ("pick up the cup") and low-level joint trajectories.

In structured, obstacle-free environments, motion planning is trivial. In the real world — where robots share space with humans, furniture, other robots, and their own bodies — it is one of the hardest computational problems in robotics. The configuration space of a 6-DOF robot is 6-dimensional; checking every possible path in this space is computationally intractable. The sampling-based methods we study in this lesson (PRM, RRT, RRT*) make this problem practical by exploring the space intelligently without exhaustively searching it.

> **Analogy**: RRT is like exploring a maze by randomly throwing a grappling hook — each toss extends your reach into unexplored territory. You don't need a complete map of the maze; you just need enough random "throws" to connect your starting point to the exit.

---

## Configuration Space (C-Space)

### From Workspace to C-Space

The **workspace** is the physical 3D space where the robot and obstacles exist. The **configuration space (C-space)** is the space of all possible robot configurations — each point in C-space completely specifies the state of every joint.

For an $n$-DOF robot, C-space is $n$-dimensional:
- 2-link planar robot: C-space is $\mathbb{R}^2$ (or $T^2$, the 2-torus, if joint angles wrap around)
- 6-DOF manipulator: C-space is $\mathbb{R}^6$
- Mobile robot on a plane: C-space is $\mathbb{R}^2 \times S^1$ (position + heading)

### C-Space Obstacles

An obstacle in workspace becomes a **C-space obstacle** — the set of all configurations where the robot collides with the obstacle. The **free C-space** $\mathcal{C}_{free}$ is the complement: all collision-free configurations.

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

### Visualizing C-Space Obstacles

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

## Potential Field Methods

### The Idea

Treat the robot as a particle in an artificial potential field:
- **Attractive potential**: pulls the robot toward the goal
- **Repulsive potential**: pushes the robot away from obstacles

The robot follows the negative gradient of the total potential (gradient descent).

### Mathematical Formulation

**Attractive potential** (quadratic):
$$U_{att}(\mathbf{q}) = \frac{1}{2} k_{att} \|\mathbf{q} - \mathbf{q}_{goal}\|^2$$

**Repulsive potential** (inverse distance):
$$U_{rep}(\mathbf{q}) = \begin{cases} \frac{1}{2} k_{rep} \left(\frac{1}{\rho(\mathbf{q})} - \frac{1}{\rho_0}\right)^2 & \text{if } \rho(\mathbf{q}) \leq \rho_0 \\ 0 & \text{if } \rho(\mathbf{q}) > \rho_0 \end{cases}$$

where $\rho(\mathbf{q})$ is the distance to the nearest obstacle and $\rho_0$ is the influence range.

**Total potential**: $U(\mathbf{q}) = U_{att}(\mathbf{q}) + U_{rep}(\mathbf{q})$

**Control law**: $\dot{\mathbf{q}} = -\nabla U(\mathbf{q})$

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

### The Local Minima Problem

The critical weakness of potential fields: the repulsive and attractive forces can balance at points that are *not* the goal, creating **local minima** where the robot gets stuck.

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

**Solutions to local minima**:
1. **Random walk escape**: Add random perturbation when stuck
2. **Navigation functions**: Specially constructed potentials with no local minima (theoretical, expensive)
3. **Use sampling-based planners instead** (PRM, RRT) — they do not suffer from local minima

---

## Sampling-Based Planners

Sampling-based planners avoid the local minima problem by building a graph or tree of collision-free configurations through random sampling. They are the dominant paradigm in modern motion planning.

### Key Concepts

- **Sampling**: Generate random configurations in C-space
- **Collision checking**: Verify if a configuration (or path segment) is collision-free
- **Nearest neighbor**: Find the closest existing node to a new sample
- **Local planner**: Connect two configurations by a simple path (usually straight line in C-space)

---

## Probabilistic Roadmap (PRM)

### Overview

PRM is a **multi-query** planner: it builds a roadmap (graph) of the free C-space during a preprocessing phase, then answers multiple start-goal queries by connecting them to the roadmap.

### Algorithm

**Preprocessing (Roadmap Construction)**:
1. Sample $N$ random configurations in C-space
2. Discard configurations that are in collision
3. For each free configuration, try to connect it to its $k$ nearest neighbors using the local planner
4. Add successful connections as edges in the roadmap graph

**Query**:
1. Connect start and goal to the roadmap
2. Search for a path using graph search (A*, Dijkstra)

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

## Rapidly-Exploring Random Tree (RRT)

### Overview

RRT is a **single-query** planner: it builds a tree rooted at the start configuration, growing toward the goal through random exploration. It is the most widely used sampling-based planner.

### Algorithm

1. Initialize tree $T$ with the start configuration as root
2. **Repeat** until goal is reached or timeout:
   a. Sample a random configuration $\mathbf{q}_{rand}$
   b. Find the nearest node in $T$ to $\mathbf{q}_{rand}$: $\mathbf{q}_{near}$
   c. Extend from $\mathbf{q}_{near}$ toward $\mathbf{q}_{rand}$ by step size $\delta$: $\mathbf{q}_{new}$
   d. If the path from $\mathbf{q}_{near}$ to $\mathbf{q}_{new}$ is collision-free, add $\mathbf{q}_{new}$ to $T$
   e. If $\mathbf{q}_{new}$ is close enough to $\mathbf{q}_{goal}$, connect and return path
3. If no path found, return failure

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

### Goal Bias

The `goal_bias` parameter controls the probability of sampling the goal instead of a random point. Too low (e.g., 0.0): the tree explores uniformly but may take long to reach the goal. Too high (e.g., 0.5): the tree grows greedily toward the goal and may get stuck behind obstacles. A typical value is 0.05 to 0.15.

---

## RRT* (Optimal RRT)

### The Problem with RRT

Basic RRT finds *a* path, but not the *shortest* path. The path is typically jerky and suboptimal because each node connects only to its nearest neighbor in the tree, not the best parent.

### RRT* Improvements

RRT* adds two operations that guarantee **asymptotic optimality** — as the number of samples grows, the path converges to the optimal solution:

1. **Rewiring**: After adding a new node, check if nearby nodes would benefit from rerouting through the new node
2. **Choose best parent**: Connect the new node to the nearby node that gives the lowest cost-to-come (not just the nearest node)

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

## Completeness and Optimality

### Theoretical Guarantees

| Planner | Complete? | Optimal? | Complexity |
|---------|-----------|----------|------------|
| Potential Field | No (local minima) | No | $O(1)$ per step |
| PRM | Probabilistically complete | Asymptotically optimal (PRM*) | $O(n \log n)$ |
| RRT | Probabilistically complete | **No** | $O(n \log n)$ |
| RRT* | Probabilistically complete | **Asymptotically optimal** | $O(n \log n)$ |

**Probabilistically complete**: If a solution exists, the probability of finding it approaches 1 as the number of samples approaches infinity.

**Asymptotically optimal**: As the number of samples approaches infinity, the cost of the returned path converges to the optimal cost.

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

## Path Post-Processing

Raw paths from sampling-based planners are typically jerky and suboptimal. Post-processing improves path quality.

### Shortcutting

Repeatedly try to connect non-adjacent waypoints by a direct path, eliminating intermediate waypoints:

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

## Practical Considerations

### Collision Checking Efficiency

Collision checking dominates planning time (often 90%+ of computation). Optimizations include:
- **Bounding volumes**: Check bounding spheres/boxes before detailed mesh checks
- **Spatial hashing**: Organize obstacles for fast proximity queries
- **Lazy evaluation**: Defer collision checks and only verify paths that are actually used (Lazy PRM, Lazy RRT)

### High-Dimensional C-Spaces

For robots with many DOF (e.g., dual-arm manipulation with 14+ DOF), sampling-based planners become slower because the volume of C-space grows exponentially. Strategies:
- **Task-space sampling**: Sample end-effector poses and use IK, rather than sampling all joints
- **Constraint manifold sampling**: For constrained tasks (keeping a glass upright), project samples onto the constraint manifold
- **Informed sampling**: Use the current best path cost to focus sampling in ellipsoidal regions (Informed RRT*)

---

## Summary

- **Configuration space (C-space)** maps the motion planning problem from physical space to joint space, where each point is a complete robot configuration
- **Potential field** methods are simple but suffer from **local minima** — the robot can get stuck
- **PRM** builds a reusable roadmap for multiple queries; suitable for static environments with many planning requests
- **RRT** grows a tree from start toward goal; good for single queries in high-dimensional spaces
- **RRT*** adds rewiring and best-parent selection, guaranteeing **asymptotic optimality** at the cost of more computation
- **Post-processing** (shortcutting, smoothing) significantly improves path quality from sampling-based planners
- In practice, collision checking is the computational bottleneck; efficient data structures and lazy evaluation are essential

---

## Exercises

### Exercise 1: C-Space Visualization

For the 2-link planar robot with $l_1 = 1.0$, $l_2 = 0.8$:
1. Place a single circular obstacle at $(0.8, 0.8)$ with radius $0.3$
2. Compute the C-space obstacle map at $2°$ resolution
3. Visualize the free C-space (you should see connected and disconnected regions)
4. How does the C-space obstacle shape change if the physical obstacle moves to $(0, 1.2)$?

### Exercise 2: Potential Field Tuning

1. Implement the potential field planner for the 2-link robot (using task-space coordinates)
2. Find a start/goal pair where the planner succeeds
3. Find a start/goal pair where the planner gets stuck in a local minimum
4. Implement a random escape strategy: when the gradient magnitude drops below a threshold, add a random displacement. Does this fix the local minimum problem?

### Exercise 3: RRT Implementation

1. Implement RRT for the 2-link planar robot in C-space
2. Run 10 planning attempts between the same start/goal. What is the variance in path length?
3. Plot the RRT tree at different iteration counts (100, 500, 2000) to visualize exploration
4. How does changing the step size affect success rate and path quality?

### Exercise 4: RRT vs RRT*

1. Run both RRT and RRT* on the same problem with $N = 1000, 2000, 5000$ samples
2. Plot path cost vs number of samples for both algorithms
3. Verify that RRT* paths improve with more samples while RRT paths do not systematically improve

### Exercise 5: Post-Processing

1. Generate an RRT path (which will be jerky)
2. Apply shortcutting and measure the path length reduction
3. Apply smoothing to the shortcut path
4. Verify that the smoothed path is still collision-free
5. Compare total path lengths: raw RRT, after shortcutting, after smoothing

---

[← Previous: Robot Dynamics](06_Robot_Dynamics.md) | [Next: Trajectory Planning →](08_Trajectory_Planning.md)
