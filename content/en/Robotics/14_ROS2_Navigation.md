# 14. ROS2 Navigation Stack (Nav2)

[← Previous: ROS2 Fundamentals](13_ROS2_Fundamentals.md) | [Next: Reinforcement Learning for Robotics →](15_RL_for_Robotics.md)

---

## Learning Objectives

1. Describe the overall architecture of the Nav2 (Navigation2) stack and its major components
2. Understand costmap layers (static, obstacle, inflation) and how they compose into a navigation costmap
3. Compare global planning algorithms: NavFn, Theta*, and Smac planners
4. Explain local planning approaches: DWB, MPPI, and regulated pure pursuit
5. Configure recovery behaviors and behavior trees for robust autonomous navigation
6. Implement waypoint following for multi-goal autonomous missions

---

In the previous lesson, we learned the fundamentals of ROS2: nodes, topics, services, actions, and how to build modular robot software. Now we put that infrastructure to work for one of the most important capabilities a mobile robot can have — autonomous navigation. Navigation answers the question: *How does a robot get from point A to point B safely, efficiently, and reliably?*

The **Nav2** (Navigation2) stack is the standard navigation framework for ROS2. It is the successor to the ROS1 navigation stack but is a complete rewrite with a more modular, plugin-based architecture. Nav2 handles everything from costmap construction to global path planning, local trajectory following, recovery from stuck situations, and multi-waypoint missions. Understanding Nav2 is essential for deploying any mobile robot — from warehouse robots to outdoor delivery vehicles.

> **Analogy**: Nav2's behavior tree is like a pilot's decision flowchart — if the main route is blocked, try alternatives; if all alternatives fail, execute recovery procedures (circle, back up, wait). Just as a pilot follows a structured protocol rather than improvising, Nav2 follows a behavior tree that defines exactly what to do in every situation, ensuring safe and predictable behavior even when things go wrong.

---

## 1. Nav2 Architecture Overview

### 1.1 The Big Picture

Nav2 consists of several major servers, each running as a lifecycle node and communicating via ROS2 actions and topics:

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

### 1.2 Navigation Flow

A typical navigation request flows through these stages:

1. **Goal received**: User or higher-level planner sends a `NavigateToPose` action goal
2. **Global planning**: The Planner Server computes a path from start to goal on the global costmap
3. **Path smoothing**: The Smoother Server (optional) refines the path for smoothness
4. **Local control**: The Controller Server follows the path using the local costmap, computing velocity commands at 20+ Hz
5. **Recovery**: If the robot gets stuck, recovery behaviors are triggered (spin, backup, wait)
6. **Orchestration**: The BT Navigator manages the flow using a behavior tree

### 1.3 Key Frames in Navigation

```
map                    Global frame — SLAM or static map
 └── odom              Local frame — odometry (smooth but drifts)
      └── base_link    Robot body frame
           └── ...     Sensor frames
```

**Why two frames?** The `odom` frame provides smooth, high-rate pose estimates (from wheel encoders) but drifts over time. The `map` frame provides globally consistent positions (from SLAM or AMCL) but can have discontinuous jumps when corrections occur. Navigation uses `map` for global planning and `odom` for local control.

---

## 2. Costmaps

### 2.1 What Is a Costmap?

A **costmap** is a 2D grid where each cell has a cost value indicating how dangerous or undesirable it is for the robot to pass through:

| Value | Meaning | Cell State |
|-------|---------|------------|
| 0 | Free space | Safe to traverse |
| 1-252 | Increasing cost | Proximity to obstacles |
| 253 | Inscribed obstacle | Robot footprint touches obstacle |
| 254 | Lethal obstacle | Obstacle occupies this cell |
| 255 | Unknown | No information about this cell |

### 2.2 Costmap Layers

Nav2 uses a **layered costmap** architecture. Each layer adds information, and layers are combined into the master costmap:

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

#### Static Layer

Loads the pre-built map (from SLAM or a map file). Cells are marked as free, occupied, or unknown based on the map data.

#### Obstacle Layer

Integrates real-time sensor data (LiDAR, depth camera) to detect dynamic obstacles not in the static map. Uses a **voxel grid** (3D) or **raytracing** to clear cells where the sensor beam passes through (no obstacle) and mark cells where it hits (obstacle).

#### Inflation Layer

The inflation layer is crucial for safe navigation. It adds a cost gradient around obstacles:

$$\text{cost}(d) = \begin{cases} 254 & \text{if } d \leq r_{inscribed} \\ \alpha \cdot e^{-\beta (d - r_{inscribed})} & \text{if } r_{inscribed} < d \leq r_{inflation} \\ 0 & \text{if } d > r_{inflation} \end{cases}$$

where:
- $d$ is the distance from the nearest obstacle
- $r_{inscribed}$ is the inscribed radius of the robot footprint
- $r_{inflation}$ is the inflation radius (how far the cost extends)
- $\alpha$ and $\beta$ control the cost decay rate

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

### 2.3 Global vs. Local Costmap

| Property | Global Costmap | Local Costmap |
|----------|---------------|---------------|
| Frame | `map` | `odom` |
| Size | Full map (or very large) | Small window around robot (3-10 m) |
| Update rate | Low (0.5-2 Hz) | High (5-20 Hz) |
| Contains | Static map + inflation | Dynamic obstacles + inflation |
| Used by | Global planner | Local controller |

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

## 3. Global Planning

### 3.1 NavFn (Navigation Function)

**NavFn** is the default global planner. It uses **Dijkstra's algorithm** or **A*** to find the shortest path on the costmap:

$$g(n) = \min_{p \in \text{predecessors}} \left[ g(p) + \text{cost}(p, n) \right]$$

NavFn operates on the costmap grid, treating cell costs as edge weights. Higher-cost cells (near obstacles) are more expensive to traverse, so the planner naturally finds paths that avoid obstacle proximity.

### 3.2 Theta* Planner

**Theta*** is an any-angle planner that produces smoother paths than grid-based A*:

- A* restricts paths to grid edges (8-connected), producing jagged paths
- Theta* allows line-of-sight shortcuts between non-adjacent grid cells
- Result: shorter, smoother paths with fewer unnecessary turns

**Key check**: For each node expansion, Theta* tests if there is a clear line-of-sight from the parent's parent to the current node. If so, it skips the intermediate node.

### 3.3 Smac Planners

The **Smac** (State Machine-based A*-Complemented) planners are Nav2's modern planning suite:

| Planner | Type | Best For |
|---------|------|----------|
| SmacPlanner2D | 2D grid search | Circular robots (holonomic or differential) |
| SmacPlannerHybrid | Hybrid A* (SE(2)) | Car-like robots (Ackermann, non-holonomic) |
| SmacPlannerLattice | State lattice | Robots with complex kinematic constraints |

**Hybrid A*** searches in $(x, y, \theta)$ space, producing paths that respect the robot's turning radius:

$$\mathbf{x}_{k+1} = \mathbf{x}_k + \begin{bmatrix} v \cos\theta_k \cdot \Delta t \\ v \sin\theta_k \cdot \Delta t \\ v \tan\delta / L \cdot \Delta t \end{bmatrix}$$

where $\delta$ is the steering angle and $L$ is the wheelbase.

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

## 4. Local Planning (Controller Plugins)

### 4.1 The Controller's Role

The local controller (called "controller" in Nav2) takes the global path and the local costmap and produces velocity commands ($v$, $\omega$) at high rate (20-100 Hz). It must:

- Follow the global path approximately
- Avoid dynamic obstacles visible in the local costmap
- Respect the robot's kinematic and dynamic constraints
- Produce smooth, feasible velocity commands

### 4.2 DWB (Dynamic Window Based)

**DWB** (Dynamic Window approach version B) is the Nav2 implementation of the Dynamic Window Approach (DWA):

1. **Sample velocities** in the admissible velocity space $(v, \omega)$ within dynamic window limits
2. **Simulate trajectories** forward for each velocity sample (typically 1-3 seconds)
3. **Score each trajectory** using multiple critics (path alignment, goal distance, obstacle proximity)
4. **Select the best** trajectory and execute its first velocity command

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

### 4.3 MPPI (Model Predictive Path Integral)

**MPPI** is a modern, sampling-based controller that uses stochastic optimization:

1. Sample $N$ random control sequences (velocity trajectories)
2. Simulate each forward through the dynamics model
3. Score each trajectory using a cost function
4. Compute a **weighted average** of the control sequences (weight = $e^{-\text{cost}/\lambda}$)
5. Execute the first control of the weighted-average sequence

$$\mathbf{u}^* = \frac{\sum_{i=1}^N w_i \mathbf{u}^{(i)}}{\sum_{i=1}^N w_i}, \qquad w_i = \exp\left(-\frac{1}{\lambda} S(\mathbf{x}^{(i)})\right)$$

where $S(\mathbf{x}^{(i)})$ is the trajectory cost and $\lambda$ is a temperature parameter.

**Advantages over DWB**:
- Explores a richer set of trajectories (not just constant velocity)
- Handles complex cost functions naturally
- GPU-acceleratable (embarrassingly parallel sampling)
- Better at navigating tight spaces and dynamic environments

### 4.4 Regulated Pure Pursuit

**Regulated Pure Pursuit** is a simpler controller suitable for following smooth paths:

1. Find the **lookahead point** on the global path at distance $L$ ahead of the robot
2. Compute the curvature to reach that point:
$$\kappa = \frac{2 \sin(\alpha)}{L}$$
where $\alpha$ is the angle between the robot's heading and the lookahead point
3. **Regulate** the speed based on curvature and obstacle proximity

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

### 4.5 Controller Comparison

| Controller | Strengths | Weaknesses | Best For |
|-----------|-----------|------------|----------|
| DWB | Well-tested, configurable critics | Limited trajectory diversity | General-purpose |
| MPPI | Rich trajectories, handles complex costs | Computationally expensive | Dynamic environments, tight spaces |
| Regulated PP | Simple, smooth, predictable | No obstacle avoidance (relies on global path) | Open environments, smooth paths |

---

## 5. Recovery Behaviors

### 5.1 Why Recoveries Are Essential

Robots get stuck. The global path might be blocked by an unmapped obstacle. The local planner might oscillate in a corner. A person might stand in the way. Recovery behaviors are the safety net that keeps the robot from getting permanently stuck.

### 5.2 Built-in Recovery Behaviors

| Recovery | Description | When to Use |
|----------|-------------|-------------|
| **Spin** | Rotate in place (360 degrees) | Clear obstacle readings, find new path |
| **BackUp** | Drive backward a short distance | Back away from obstacle too close to turn |
| **Wait** | Stop and wait for obstacle to move | Dynamic obstacles (people, other robots) |
| **ClearCostmap** | Reset obstacle layer in local costmap | Stale obstacle data causing false blocks |

### 5.3 Recovery Configuration

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

## 6. Behavior Trees for Navigation

### 6.1 Why Behavior Trees?

Nav2 uses **behavior trees** (BTs) to orchestrate the navigation pipeline. A behavior tree is a hierarchical decision-making structure that is:

- **Modular**: Each behavior is a self-contained node
- **Reactive**: The tree is ticked at a fixed rate; behaviors can be interrupted
- **Readable**: The tree structure makes the control logic explicit and inspectable
- **Extensible**: New behaviors are easily added as BT nodes

### 6.2 BT Node Types

| Node Type | Symbol | Behavior |
|-----------|--------|----------|
| **Sequence** | → | Execute children left-to-right; fail if any child fails |
| **Fallback** | ? | Try children left-to-right; succeed if any child succeeds |
| **Action** | [box] | Execute a ROS2 action (plan, follow, spin, etc.) |
| **Condition** | (oval) | Check a condition (goal updated? path valid?) |
| **Decorator** | ◇ | Modify child behavior (retry, invert, rate limit) |

### 6.3 Default Navigation BT

The default Nav2 behavior tree (simplified):

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

The behavior:
1. **Sequence** tries to navigate: compute path, smooth it, follow it
2. If the sequence **fails** (planner fails, controller fails, etc.), execution moves to the recovery fallback
3. **Fallback** tries each recovery in order until one succeeds
4. After a successful recovery, the sequence retries navigation

### 6.4 Custom Behavior Trees

You can customize the navigation behavior by writing your own BT XML:

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

## 7. Waypoint Following and Autonomous Navigation

### 7.1 Waypoint Following

For multi-goal missions (patrol, delivery, inspection), Nav2 provides waypoint following through the `NavigateThroughPoses` action:

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

### 7.2 Task Executor at Waypoints

Nav2 supports **waypoint task executors** — plugins that run at each waypoint:

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

## 8. Putting It All Together

### 8.1 Minimal Nav2 Configuration

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

### 8.2 Common Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| Robot doesn't move | TF tree broken | Check `map→odom→base_link` chain |
| "No valid path" | Costmap inflated too much | Reduce `inflation_radius` |
| Robot oscillates | Controller gains too aggressive | Reduce speed, increase smoothing |
| Robot hits obstacles | Local costmap not updating | Check sensor topics, QoS compatibility |
| Recovery keeps triggering | Path too close to walls | Increase costmap inflation |

---

## Summary

| Concept | Key Idea |
|---------|----------|
| Nav2 | Complete navigation stack: planning, control, recovery, behavior trees |
| Costmap layers | Static map + dynamic obstacles + inflation = navigation grid |
| Inflation | Grow obstacles by robot radius + cost gradient for safety margin |
| Global planner | A*, Theta*, Hybrid A*; search on global costmap for collision-free path |
| Local controller | DWB (sample & score), MPPI (stochastic optimal), Pure Pursuit (geometric) |
| Recovery behaviors | Spin, backup, wait, clear costmap — activated when navigation fails |
| Behavior trees | Hierarchical, reactive task orchestration; replace state machines |
| Waypoint following | Multi-goal missions with optional task execution at each waypoint |

---

## Exercises

1. **Costmap analysis**: Given a 100x100 occupancy grid with 5 rectangular obstacles, implement the inflation layer with inscribed radius 0.2 m, inflation radius 0.6 m, and cost scaling factor 3.0. Visualize the resulting costmap as a heatmap. Show how paths planned on the inflated costmap avoid obstacle proximity.

2. **Global planner comparison**: Implement A* and a simplified Theta* on a 2D costmap. Compare the resulting paths for three scenarios: (a) open space, (b) narrow corridor, (c) maze with dead ends. Measure path length and number of turns for each.

3. **DWB implementation**: Implement the sample-and-score local planner described above. Create a scenario where a differential-drive robot follows a global path that passes near an obstacle not on the global map. Show that the local planner avoids the obstacle by scoring trajectories against the local costmap.

4. **Behavior tree simulator**: Implement the simplified behavior tree engine (Sequence, Fallback, Action nodes). Create a navigation BT that: attempts to navigate to a goal, and if navigation fails, tries spin recovery, then backup, then declares failure. Simulate scenarios where (a) navigation succeeds, (b) spin recovery resolves the issue, and (c) all recoveries fail.

5. **Waypoint mission**: Design a patrol mission for a warehouse robot with 6 waypoints forming a loop. The robot should visit each waypoint, wait 5 seconds (simulating an inspection), then proceed to the next. If any navigation segment fails, the robot should skip that waypoint and proceed to the next. Implement the mission logic as pseudocode or Python.

---

## Further Reading

- Macenski, S. et al. "The Marathon 2: A Navigation System." *IEEE/RSJ IROS*, 2020. (Nav2 architecture and design)
- Nav2 Official Documentation: [navigation.ros.org](https://navigation.ros.org/) (Configuration guides, tutorials)
- Colledanchise, M. and Ogren, P. *Behavior Trees in Robotics and AI*. CRC Press, 2018. (Behavior trees theory and practice)
- Fox, D. et al. "The Dynamic Window Approach to Collision Avoidance." *IEEE Robotics & Automation Magazine*, 1997. (DWA foundation)
- Williams, G. et al. "Information Theoretic MPC for Model-Based Reinforcement Learning." *ICRA*, 2017. (MPPI theory)

---

[← Previous: ROS2 Fundamentals](13_ROS2_Fundamentals.md) | [Next: Reinforcement Learning for Robotics →](15_RL_for_Robotics.md)
