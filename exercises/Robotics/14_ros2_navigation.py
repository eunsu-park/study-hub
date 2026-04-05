"""
Exercises for Lesson 14: ROS2 Navigation Stack
Topic: Robotics
Solutions to practice problems from the lesson.

Note: These exercises simulate Nav2 concepts in pure Python since
actual Nav2 requires a full ROS2 installation.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple


def exercise_1():
    """
    Exercise 1: Costmap Analysis
    100x100 occupancy grid with inflation layer.
    """
    grid_size = 100
    resolution = 0.1  # m/cell (10x10m world)
    inscribed_radius = 0.2   # m
    inflation_radius = 0.6   # m
    cost_scaling = 3.0

    # Create base occupancy grid (0=free, 100=occupied)
    occupancy = np.zeros((grid_size, grid_size), dtype=float)

    # Add 5 rectangular obstacles
    obstacles = [
        (20, 20, 10, 5),   # (row, col, height, width)
        (50, 10, 8, 12),
        (30, 60, 15, 5),
        (70, 40, 5, 20),
        (10, 80, 12, 8),
    ]
    for r, c, h, w in obstacles:
        occupancy[r:r+h, c:c+w] = 100

    # Inflate costmap
    costmap = occupancy.copy()
    inscribed_cells = int(inscribed_radius / resolution)
    inflation_cells = int(inflation_radius / resolution)

    # For each occupied cell, inflate
    occ_rows, occ_cols = np.where(occupancy == 100)
    for oi, oj in zip(occ_rows, occ_cols):
        for di in range(-inflation_cells, inflation_cells + 1):
            for dj in range(-inflation_cells, inflation_cells + 1):
                ni, nj = oi + di, oj + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    dist = np.sqrt(di**2 + dj**2) * resolution
                    if dist <= inscribed_radius:
                        costmap[ni, nj] = max(costmap[ni, nj], 254)  # lethal
                    elif dist <= inflation_radius:
                        cost = 253 * np.exp(-cost_scaling * (dist - inscribed_radius))
                        costmap[ni, nj] = max(costmap[ni, nj], cost)

    n_lethal = np.sum(costmap >= 254)
    n_inflated = np.sum((costmap > 0) & (costmap < 254))
    n_free = np.sum(costmap == 0)

    print("Costmap Inflation Analysis")
    print(f"  Grid: {grid_size}x{grid_size} ({grid_size*resolution}m x {grid_size*resolution}m)")
    print(f"  Inscribed radius: {inscribed_radius}m ({inscribed_cells} cells)")
    print(f"  Inflation radius: {inflation_radius}m ({inflation_cells} cells)")
    print(f"  Cost scaling: {cost_scaling}")
    print(f"\n  Cells: lethal={n_lethal}, inflated={n_inflated}, free={n_free}")
    print(f"  Coverage: {(n_lethal + n_inflated) / (grid_size**2) * 100:.1f}% non-free")
    print(f"\n  The inflation layer creates a gradient around obstacles.")
    print(f"  Paths planned on inflated costmap naturally avoid obstacle proximity")
    print(f"  because the planner penalizes high-cost cells.")


def exercise_2():
    """
    Exercise 2: Global Planner Comparison (A* vs Theta*)
    """
    grid_size = 50
    grid = np.zeros((grid_size, grid_size))

    # Add obstacles for different scenarios
    scenarios = {
        "open_space": [(20, 20, 3, 3), (30, 35, 4, 4)],
        "narrow_corridor": [(10, 0, 30, 20), (10, 22, 30, 28)],  # narrow gap
        "maze": [(5, 5, 2, 30), (15, 10, 2, 30), (25, 5, 2, 30), (35, 10, 2, 30)],
    }

    def a_star(grid, start, goal):
        """A* on grid (4-connected or 8-connected)."""
        rows, cols = grid.shape
        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: np.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2)}

        neighbors_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

        while open_set:
            current = min(open_set, key=lambda n: f_score.get(n, float('inf')))
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            open_set.remove(current)
            for dr, dc in neighbors_8:
                nr, nc = current[0]+dr, current[1]+dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                    neighbor = (nr, nc)
                    cost = np.sqrt(dr**2 + dc**2)
                    tentative_g = g_score[current] + cost
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + np.sqrt(
                            (goal[0]-nr)**2 + (goal[1]-nc)**2)
                        open_set.add(neighbor)

        return None  # no path

    def theta_star(grid, start, goal):
        """Simplified Theta* (any-angle A*)."""
        rows, cols = grid.shape
        open_set = {start}
        came_from = {start: start}
        g_score = {start: 0}
        f_score = {start: np.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2)}
        neighbors_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

        def line_of_sight(p1, p2):
            """Bresenham-like check for collision-free line."""
            r0, c0 = p1
            r1, c1 = p2
            n = max(abs(r1-r0), abs(c1-c0))
            if n == 0:
                return True
            for i in range(n + 1):
                t = i / n
                r = int(round(r0 + t * (r1 - r0)))
                c = int(round(c0 + t * (c1 - c0)))
                if 0 <= r < rows and 0 <= c < cols:
                    if grid[r, c] != 0:
                        return False
                else:
                    return False
            return True

        while open_set:
            current = min(open_set, key=lambda n: f_score.get(n, float('inf')))
            if current == goal:
                path = [current]
                while came_from[current] != current:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            open_set.remove(current)
            for dr, dc in neighbors_8:
                nr, nc = current[0]+dr, current[1]+dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                    neighbor = (nr, nc)
                    parent = came_from[current]

                    # Theta* key idea: try to connect neighbor to current's parent
                    if line_of_sight(parent, neighbor):
                        new_g = g_score[parent] + np.sqrt(
                            (nr-parent[0])**2 + (nc-parent[1])**2)
                        if new_g < g_score.get(neighbor, float('inf')):
                            came_from[neighbor] = parent
                            g_score[neighbor] = new_g
                            f_score[neighbor] = new_g + np.sqrt(
                                (goal[0]-nr)**2 + (goal[1]-nc)**2)
                            open_set.add(neighbor)
                    else:
                        cost = np.sqrt(dr**2 + dc**2)
                        new_g = g_score[current] + cost
                        if new_g < g_score.get(neighbor, float('inf')):
                            came_from[neighbor] = current
                            g_score[neighbor] = new_g
                            f_score[neighbor] = new_g + np.sqrt(
                                (goal[0]-nr)**2 + (goal[1]-nc)**2)
                            open_set.add(neighbor)

        return None

    def count_turns(path):
        if len(path) < 3:
            return 0
        turns = 0
        for i in range(2, len(path)):
            d1 = (path[i-1][0]-path[i-2][0], path[i-1][1]-path[i-2][1])
            d2 = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
            if d1 != d2:
                turns += 1
        return turns

    def path_length(path):
        return sum(np.sqrt((path[i+1][0]-path[i][0])**2 +
                           (path[i+1][1]-path[i][1])**2)
                   for i in range(len(path)-1))

    start = (2, 2)
    goal = (47, 47)

    print("Global Planner Comparison: A* vs Theta*")
    print(f"  Grid: {grid_size}x{grid_size}, Start: {start}, Goal: {goal}")
    print(f"\n  {'Scenario':>18} | {'Algo':>7} | {'Length':>8} | {'Turns':>6} | {'Nodes':>6}")
    print("  " + "-" * 58)

    for scenario_name, obs_list in scenarios.items():
        g = np.zeros((grid_size, grid_size))
        for r, c, h, w in obs_list:
            r2 = min(r+h, grid_size)
            c2 = min(c+w, grid_size)
            g[r:r2, c:c2] = 1

        for algo_name, algo_fn in [("A*", a_star), ("Theta*", theta_star)]:
            path = algo_fn(g, start, goal)
            if path:
                length = path_length(path)
                turns = count_turns(path)
                print(f"  {scenario_name:>18} | {algo_name:>7} | {length:>8.2f} | "
                      f"{turns:>6} | {len(path):>6}")
            else:
                print(f"  {scenario_name:>18} | {algo_name:>7} | {'N/A':>8} | {'N/A':>6} | {'N/A':>6}")

    print(f"\n  Theta* produces shorter paths with fewer turns because it allows")
    print(f"  any-angle connections, not just 8-connected grid moves.")


def exercise_3():
    """
    Exercise 3: DWB (Dynamic Window) Local Planner
    """
    # Robot parameters
    max_v = 0.5       # m/s
    max_w = 1.0       # rad/s
    max_acc_v = 0.5   # m/s^2
    max_acc_w = 1.0   # rad/s^2
    dt = 0.1
    sim_time = 2.0
    n_v_samples = 11
    n_w_samples = 21

    # Current state
    x, y, theta = 0.0, 0.0, 0.0
    v_curr, w_curr = 0.2, 0.0

    # Global path (waypoints)
    global_path = [(1, 0), (2, 0.5), (3, 1.0), (4, 1.0)]

    # Unexpected obstacle (not on global map)
    obstacle = (1.5, 0.1, 0.2)  # x, y, radius

    def simulate_trajectory(v, w, x0, y0, th0, steps=20):
        """Simulate forward for sim_time with constant v, w."""
        traj = [(x0, y0, th0)]
        x_, y_, th_ = x0, y0, th0
        dt_sim = sim_time / steps
        for _ in range(steps):
            x_ += v * np.cos(th_) * dt_sim
            y_ += v * np.sin(th_) * dt_sim
            th_ += w * dt_sim
            traj.append((x_, y_, th_))
        return traj

    def score_trajectory(traj, global_path, obstacles):
        """Score = path_alignment - obstacle_proximity + velocity."""
        # Path alignment: distance to nearest global path point at end
        end = np.array([traj[-1][0], traj[-1][1]])
        min_dist = min(np.linalg.norm(end - np.array(wp)) for wp in global_path)
        path_score = -min_dist

        # Obstacle clearance
        obs_score = 0
        for tx, ty, _ in traj:
            for ox, oy, or_ in [obstacles]:
                d = np.sqrt((tx - ox)**2 + (ty - oy)**2)
                if d < or_:
                    return -1000  # collision
                if d < or_ + 0.3:
                    obs_score -= 1.0 / d

        # Velocity score (prefer faster motion)
        v_end = np.sqrt((traj[-1][0] - traj[-2][0])**2 +
                         (traj[-1][1] - traj[-2][1])**2) / (sim_time / 20)

        return 3.0 * path_score + 2.0 * obs_score + 1.0 * v_end

    # Dynamic window
    v_min = max(0, v_curr - max_acc_v * dt)
    v_max = min(max_v, v_curr + max_acc_v * dt)
    w_min = max(-max_w, w_curr - max_acc_w * dt)
    w_max = min(max_w, w_curr + max_acc_w * dt)

    best_score = -float('inf')
    best_v, best_w = 0, 0
    n_feasible = 0

    v_samples = np.linspace(v_min, v_max, n_v_samples)
    w_samples = np.linspace(w_min, w_max, n_w_samples)

    for v in v_samples:
        for w in w_samples:
            traj = simulate_trajectory(v, w, x, y, theta)
            score = score_trajectory(traj, global_path, obstacle)
            if score > -999:
                n_feasible += 1
            if score > best_score:
                best_score = score
                best_v, best_w = v, w

    best_traj = simulate_trajectory(best_v, best_w, x, y, theta)

    print("DWB Local Planner Simulation")
    print(f"  Obstacle at ({obstacle[0]}, {obstacle[1]}), r={obstacle[2]}m")
    print(f"  Dynamic window: v=[{v_min:.2f}, {v_max:.2f}], w=[{w_min:.2f}, {w_max:.2f}]")
    print(f"  Samples: {n_v_samples} x {n_w_samples} = {n_v_samples * n_w_samples}")
    print(f"  Feasible trajectories: {n_feasible}")
    print(f"  Best: v={best_v:.3f} m/s, w={best_w:.3f} rad/s, score={best_score:.3f}")
    print(f"  Trajectory endpoint: ({best_traj[-1][0]:.3f}, {best_traj[-1][1]:.3f})")
    print(f"  The local planner avoids the unexpected obstacle by steering around it.")


def exercise_4():
    """
    Exercise 4: Behavior Tree Simulator
    """
    class NodeStatus(Enum):
        SUCCESS = "SUCCESS"
        FAILURE = "FAILURE"
        RUNNING = "RUNNING"

    class BTNode:
        def __init__(self, name):
            self.name = name

        def tick(self):
            raise NotImplementedError

    class ActionNode(BTNode):
        def __init__(self, name, action_fn):
            super().__init__(name)
            self.action_fn = action_fn

        def tick(self):
            return self.action_fn()

    class SequenceNode(BTNode):
        def __init__(self, name, children):
            super().__init__(name)
            self.children = children

        def tick(self):
            for child in self.children:
                status = child.tick()
                if status != NodeStatus.SUCCESS:
                    return status
            return NodeStatus.SUCCESS

    class FallbackNode(BTNode):
        def __init__(self, name, children):
            super().__init__(name)
            self.children = children

        def tick(self):
            for child in self.children:
                status = child.tick()
                if status != NodeStatus.FAILURE:
                    return status
            return NodeStatus.FAILURE

    # Scenario simulation state
    scenario_state = {"nav_success": True, "spin_helps": False}
    log = []

    def navigate():
        log.append("  Attempting navigation...")
        if scenario_state["nav_success"]:
            log.append("  Navigation: SUCCESS")
            return NodeStatus.SUCCESS
        log.append("  Navigation: FAILURE")
        return NodeStatus.FAILURE

    def spin_recovery():
        log.append("  Attempting spin recovery...")
        if scenario_state["spin_helps"]:
            log.append("  Spin recovery: SUCCESS")
            return NodeStatus.SUCCESS
        log.append("  Spin recovery: FAILURE")
        return NodeStatus.FAILURE

    def backup_recovery():
        log.append("  Attempting backup recovery...")
        log.append("  Backup recovery: FAILURE")
        return NodeStatus.FAILURE

    def declare_failure():
        log.append("  Declaring FAILURE")
        return NodeStatus.FAILURE

    # Build BT: Fallback(Navigate, Sequence(Spin, Navigate), Sequence(Backup, Navigate), Fail)
    bt = FallbackNode("root", [
        ActionNode("Navigate", navigate),
        SequenceNode("SpinRecover", [
            ActionNode("Spin", spin_recovery),
            ActionNode("RetryNav", navigate),
        ]),
        SequenceNode("BackupRecover", [
            ActionNode("Backup", backup_recovery),
            ActionNode("RetryNav2", navigate),
        ]),
        ActionNode("Fail", declare_failure),
    ])

    print("Behavior Tree Navigation Simulator")

    # Scenario A: Navigation succeeds
    scenario_state = {"nav_success": True, "spin_helps": False}
    log = []
    result = bt.tick()
    print(f"\n  Scenario A: Navigation succeeds")
    for l in log:
        print(l)
    print(f"  Final result: {result.value}")

    # Scenario B: Nav fails, spin helps
    scenario_state = {"nav_success": False, "spin_helps": True}
    log = []

    # Need to make navigate succeed after spin
    call_count = [0]
    def navigate_retry():
        call_count[0] += 1
        if call_count[0] <= 1:
            log.append("  Attempting navigation...")
            log.append("  Navigation: FAILURE")
            return NodeStatus.FAILURE
        log.append("  Attempting navigation (retry)...")
        log.append("  Navigation: SUCCESS")
        return NodeStatus.SUCCESS

    bt_b = FallbackNode("root", [
        ActionNode("Navigate", navigate_retry),
        SequenceNode("SpinRecover", [
            ActionNode("Spin", spin_recovery),
            ActionNode("RetryNav", navigate_retry),
        ]),
    ])
    call_count[0] = 0
    result = bt_b.tick()
    print(f"\n  Scenario B: Nav fails, spin recovery helps")
    for l in log:
        print(l)
    print(f"  Final result: {result.value}")

    # Scenario C: All recoveries fail
    scenario_state = {"nav_success": False, "spin_helps": False}
    log = []
    result = bt.tick()
    print(f"\n  Scenario C: All recoveries fail")
    for l in log:
        print(l)
    print(f"  Final result: {result.value}")


def exercise_5():
    """
    Exercise 5: Waypoint Patrol Mission
    6 waypoints in a warehouse loop, 5s wait at each.
    """
    waypoints = [
        (1, 1, "Loading dock"),
        (5, 1, "Aisle A start"),
        (5, 8, "Aisle A end"),
        (9, 8, "Aisle B end"),
        (9, 1, "Aisle B start"),
        (1, 1, "Back to loading dock"),
    ]

    # Simulate navigation with some failures
    np.random.seed(42)
    fail_prob = 0.15  # 15% chance of navigation failure

    print("Waypoint Patrol Mission")
    print(f"  {len(waypoints)} waypoints, 5s inspection at each")
    print(f"  Navigation failure probability: {fail_prob*100:.0f}%")
    print()

    total_time = 0
    visited = 0
    skipped = 0

    for i, (wx, wy, label) in enumerate(waypoints):
        nav_success = np.random.random() > fail_prob

        if nav_success:
            # Simulate travel time (1m/s speed)
            if i > 0:
                dx = wx - waypoints[i-1][0]
                dy = wy - waypoints[i-1][1]
                travel_time = np.sqrt(dx**2 + dy**2) / 1.0
                total_time += travel_time

            # Inspection
            inspection_time = 5.0
            total_time += inspection_time
            visited += 1
            print(f"  WP {i+1} ({label:>22}): REACHED at t={total_time:.1f}s "
                  f"- inspecting for {inspection_time}s")
        else:
            skipped += 1
            print(f"  WP {i+1} ({label:>22}): NAVIGATION FAILED - skipping")

    print(f"\n  Mission summary:")
    print(f"    Waypoints visited: {visited}/{len(waypoints)}")
    print(f"    Waypoints skipped: {skipped}")
    print(f"    Total mission time: {total_time:.1f}s")
    print(f"\n  Mission logic: on failure, skip waypoint and proceed to next.")
    print(f"  This ensures the patrol continues even with partial failures.")


if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 14: ROS2 Navigation Stack — Exercise Solutions")
    print("=" * 70)

    print("\n--- Exercise 1: Costmap Analysis ---")
    exercise_1()

    print("\n--- Exercise 2: Global Planner Comparison ---")
    exercise_2()

    print("\n--- Exercise 3: DWB Implementation ---")
    exercise_3()

    print("\n--- Exercise 4: Behavior Tree Simulator ---")
    exercise_4()

    print("\n--- Exercise 5: Waypoint Mission ---")
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
