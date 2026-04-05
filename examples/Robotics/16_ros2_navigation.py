"""
ROS2 Navigation Stack (Nav2) Concepts — Pure-Python Simulation
===============================================================
Demonstrate the core algorithms behind Nav2 without requiring a ROS2
installation. We simulate the three pillars of autonomous navigation:

  1. **Costmap**: A 2D occupancy grid with obstacle, inflation, and free
     layers. Nav2 uses Costmap2D with pluggable layers; here we build a
     simplified version from scratch to show the principles.

  2. **Global Planner (A*)**: Finds the shortest obstacle-free path from
     start to goal on the costmap. Nav2 offers NavFn (Dijkstra/A*),
     Theta* (any-angle), and Smac (lattice-based) planners.

  3. **Local Controller (Pure Pursuit)**: Tracks the global path in real
     time by steering toward a lookahead point. Nav2 offers DWB (Dynamic
     Window), MPPI, and Regulated Pure Pursuit controllers.

Additionally, we simulate a **recovery behavior** (backup-and-rotate)
triggered when the robot gets stuck — mirroring Nav2's behavior tree
recovery actions (spin, backup, wait, clear costmap).

This example is purely educational — no ROS2 dependencies required.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq


# ---------------------------------------------------------------------------
# Costmap generation
# ---------------------------------------------------------------------------
class Costmap:
    """A 2D costmap with obstacle and inflation layers.

    In Nav2, the costmap is built from multiple layers:
      - Static layer: from a pre-built map (SLAM output)
      - Obstacle layer: from live sensor data (LiDAR, depth camera)
      - Inflation layer: expands obstacles by the robot radius + safety margin

    Cost values (matching Nav2 convention):
      - 0: free space
      - 1-252: increasing cost (inflation gradient)
      - 253: inscribed (robot center would touch obstacle)
      - 254: lethal (occupied cell)
      - 255: unknown

    Parameters:
        width, height: Grid dimensions in cells.
        resolution: Meters per cell.
        inflation_radius: Inflation distance in meters.
    """

    FREE = 0
    LETHAL = 254
    INSCRIBED = 253

    def __init__(self, width: int = 100, height: int = 100,
                 resolution: float = 0.1, inflation_radius: float = 0.3):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.inflation_radius = inflation_radius
        self.grid = np.zeros((height, width), dtype=np.uint8)

    def add_rectangular_obstacle(self, x: int, y: int, w: int, h: int):
        """Add a rectangular obstacle (in grid coordinates)."""
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.width, x + w)
        y2 = min(self.height, y + h)
        self.grid[y1:y2, x1:x2] = self.LETHAL

    def inflate(self):
        """Apply inflation layer around all lethal cells.

        Inflation creates a cost gradient around obstacles. The cost
        decreases with distance from the obstacle, reaching zero at the
        inflation radius. This keeps the robot away from obstacles even
        if the global path technically has clearance.

        Uses an efficient distance-transform approach.
        """
        lethal_mask = (self.grid == self.LETHAL)
        inflation_cells = int(self.inflation_radius / self.resolution)

        for iy in range(self.height):
            for ix in range(self.width):
                if lethal_mask[iy, ix]:
                    continue

                # Find distance to nearest lethal cell (brute force for clarity)
                y_min = max(0, iy - inflation_cells)
                y_max = min(self.height, iy + inflation_cells + 1)
                x_min = max(0, ix - inflation_cells)
                x_max = min(self.width, ix + inflation_cells + 1)

                local = lethal_mask[y_min:y_max, x_min:x_max]
                if not np.any(local):
                    continue

                ys, xs = np.where(local)
                dists = np.sqrt((xs + x_min - ix)**2 + (ys + y_min - iy)**2)
                min_dist = dists.min() * self.resolution

                if min_dist < self.inflation_radius:
                    # Linear decay from INSCRIBED to 1
                    cost = int(self.INSCRIBED * (1.0 - min_dist / self.inflation_radius))
                    self.grid[iy, ix] = max(self.grid[iy, ix], cost)

    def world_to_grid(self, wx: float, wy: float) -> tuple:
        """Convert world coordinates to grid indices."""
        gx = int(wx / self.resolution)
        gy = int(wy / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> tuple:
        """Convert grid indices to world coordinates (cell center)."""
        wx = (gx + 0.5) * self.resolution
        wy = (gy + 0.5) * self.resolution
        return wx, wy

    def is_free(self, gx: int, gy: int) -> bool:
        """Check if a cell is traversable (below inscribed cost)."""
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return self.grid[gy, gx] < self.INSCRIBED
        return False


# ---------------------------------------------------------------------------
# A* global planner
# ---------------------------------------------------------------------------
def astar(costmap: Costmap, start: tuple, goal: tuple) -> list:
    """A* search on the costmap grid.

    Nav2's NavFn planner uses a wavefront (Dijkstra) or A* algorithm on
    the costmap. A* is Dijkstra with a heuristic that guides the search
    toward the goal, typically reducing the number of cells expanded.

    The cost to traverse a cell includes both the travel distance and the
    costmap value (inflation cost), so paths naturally stay away from
    obstacles even when a shorter path exists near walls.

    Args:
        costmap: The Costmap object.
        start: (gx, gy) start cell.
        goal: (gx, gy) goal cell.

    Returns:
        List of (gx, gy) waypoints from start to goal, or empty if no path.
    """
    open_set = [(0, start)]
    came_from = {}
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0

    # 8-connected neighbors
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dx, dy in neighbors:
            nx, ny = current[0] + dx, current[1] + dy

            if not costmap.is_free(nx, ny):
                continue

            # Movement cost: diagonal = sqrt(2), cardinal = 1
            move_cost = np.sqrt(dx**2 + dy**2) * costmap.resolution
            # Add costmap penalty (normalized)
            cell_cost = costmap.grid[ny, nx] / 252.0 * costmap.resolution
            tentative_g = g_score[current] + move_cost + cell_cost

            if tentative_g < g_score[(nx, ny)]:
                g_score[(nx, ny)] = tentative_g
                # Euclidean heuristic (admissible for 8-connected grid)
                h = np.sqrt((nx - goal[0])**2 + (ny - goal[1])**2) * costmap.resolution
                f = tentative_g + h
                came_from[(nx, ny)] = current
                heapq.heappush(open_set, (f, (nx, ny)))

    return []  # No path found


# ---------------------------------------------------------------------------
# Pure Pursuit local controller
# ---------------------------------------------------------------------------
class PurePursuitController:
    """Regulated Pure Pursuit path-tracking controller.

    Pure pursuit finds a "lookahead point" on the path at a fixed distance
    ahead of the robot, then computes the curvature needed to reach it.
    This produces smooth, stable tracking behavior.

    Nav2's RegulatedPurePursuit adds speed regulation near obstacles and
    curvature-based slowdown. We implement the basic version here.

    Parameters:
        lookahead_dist: Distance ahead on the path to target (meters).
        max_linear_vel: Maximum forward speed (m/s).
        max_angular_vel: Maximum turning rate (rad/s).
    """

    def __init__(self, lookahead_dist: float = 0.5,
                 max_linear_vel: float = 0.5,
                 max_angular_vel: float = 1.5):
        self.lookahead_dist = lookahead_dist
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel

    def compute_velocity(self, robot_x: float, robot_y: float,
                         robot_theta: float,
                         path: list) -> tuple:
        """Compute velocity command to follow the path.

        Returns:
            (v, omega): Linear and angular velocity commands.
            lookahead: The (x, y) lookahead point (for visualization).
        """
        # Find the lookahead point on the path
        lookahead_pt = path[-1]  # Default to goal
        for i in range(len(path) - 1, -1, -1):
            dx = path[i][0] - robot_x
            dy = path[i][1] - robot_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist >= self.lookahead_dist:
                lookahead_pt = path[i]
                break

        # Compute curvature to reach the lookahead point
        dx = lookahead_pt[0] - robot_x
        dy = lookahead_pt[1] - robot_y

        # Transform to robot frame
        local_x = np.cos(robot_theta) * dx + np.sin(robot_theta) * dy
        local_y = -np.sin(robot_theta) * dx + np.cos(robot_theta) * dy

        # Distance to goal for speed regulation
        goal_dx = path[-1][0] - robot_x
        goal_dy = path[-1][1] - robot_y
        goal_dist = np.sqrt(goal_dx**2 + goal_dy**2)

        if goal_dist < 0.1:
            return 0.0, 0.0, lookahead_pt

        # Pure pursuit curvature: kappa = 2 * y / L^2
        L_sq = local_x**2 + local_y**2
        if L_sq < 1e-6:
            return 0.0, 0.0, lookahead_pt

        curvature = 2.0 * local_y / L_sq

        # Speed regulation: slow down near goal and at high curvature
        v = self.max_linear_vel
        v = min(v, goal_dist)  # Slow near goal
        v = min(v, self.max_linear_vel / (1.0 + abs(curvature)))  # Slow at curves

        omega = v * curvature
        omega = np.clip(omega, -self.max_angular_vel, self.max_angular_vel)

        return v, omega, lookahead_pt


# ---------------------------------------------------------------------------
# Recovery behavior
# ---------------------------------------------------------------------------
def backup_and_rotate(robot_pose: np.ndarray, backup_dist: float = 0.3,
                      rotate_angle: float = np.pi / 2) -> list:
    """Generate a recovery maneuver: back up, then rotate.

    Nav2 recovery behaviors are triggered by the behavior tree when the
    robot gets stuck (controller fails, planner fails, or progress is
    stalled). Common recoveries: Spin, BackUp, Wait, ClearCostmap.

    Returns a list of intermediate poses for the maneuver.
    """
    x, y, theta = robot_pose
    poses = []

    # Phase 1: Back up
    n_backup = 10
    for i in range(1, n_backup + 1):
        frac = i / n_backup
        bx = x - backup_dist * frac * np.cos(theta)
        by = y - backup_dist * frac * np.sin(theta)
        poses.append(np.array([bx, by, theta]))

    # Phase 2: Rotate in place
    n_rotate = 10
    bx, by = poses[-1][0], poses[-1][1]
    for i in range(1, n_rotate + 1):
        frac = i / n_rotate
        new_theta = theta + rotate_angle * frac
        poses.append(np.array([bx, by, new_theta]))

    return poses


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo_nav2_concepts():
    """Run a Nav2-style navigation simulation: costmap + A* + pure pursuit."""
    print("=" * 60)
    print("ROS2 Navigation (Nav2) Concepts Demo")
    print("=" * 60)

    np.random.seed(42)

    # --- Build costmap with obstacles ---
    costmap = Costmap(width=80, height=80, resolution=0.1,
                      inflation_radius=0.25)

    # Add walls and obstacles (furniture-like layout)
    costmap.add_rectangular_obstacle(20, 15, 3, 25)  # Wall 1
    costmap.add_rectangular_obstacle(40, 30, 20, 3)  # Wall 2
    costmap.add_rectangular_obstacle(55, 10, 3, 20)  # Wall 3
    costmap.add_rectangular_obstacle(10, 50, 25, 3)  # Wall 4
    costmap.add_rectangular_obstacle(45, 55, 3, 20)  # Wall 5

    print("Inflating costmap...")
    costmap.inflate()

    # --- Plan global path ---
    start_world = (1.0, 1.0)
    goal_world = (7.0, 7.0)
    start_grid = costmap.world_to_grid(*start_world)
    goal_grid = costmap.world_to_grid(*goal_world)

    print(f"Planning path from {start_world} to {goal_world}...")
    grid_path = astar(costmap, start_grid, goal_grid)

    if not grid_path:
        print("ERROR: No path found! Check obstacle configuration.")
        return

    # Convert to world coordinates
    world_path = [costmap.grid_to_world(gx, gy) for gx, gy in grid_path]
    print(f"Global path: {len(world_path)} waypoints")

    # --- Follow path with Pure Pursuit ---
    controller = PurePursuitController(lookahead_dist=0.4, max_linear_vel=0.4)

    robot = np.array([start_world[0], start_world[1], 0.0])
    dt = 0.1
    max_steps = 500

    trajectory = [robot.copy()]
    lookahead_points = []

    print("Following path with Pure Pursuit controller...")
    for step in range(max_steps):
        v, omega, la_pt = controller.compute_velocity(
            robot[0], robot[1], robot[2], world_path)
        lookahead_points.append(la_pt)

        if v == 0.0 and omega == 0.0:
            print(f"Goal reached at step {step}!")
            break

        # Simple kinematic update
        robot[0] += v * np.cos(robot[2]) * dt
        robot[1] += v * np.sin(robot[2]) * dt
        robot[2] += omega * dt
        trajectory.append(robot.copy())

    trajectory = np.array(trajectory)

    # --- Simulate a recovery behavior at an arbitrary pose ---
    recovery_start = np.array([3.0, 3.5, np.pi / 4])
    recovery_poses = backup_and_rotate(recovery_start)
    recovery_arr = np.array(recovery_poses)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Costmap with path
    ax1 = axes[0]
    ax1.imshow(costmap.grid, origin='lower', cmap='RdYlGn_r',
               extent=[0, costmap.width * costmap.resolution,
                       0, costmap.height * costmap.resolution],
               vmin=0, vmax=254, alpha=0.8)
    path_arr = np.array(world_path)
    ax1.plot(path_arr[:, 0], path_arr[:, 1], 'b-', lw=2,
             label='A* global path')
    ax1.plot(*start_world, 'go', markersize=10, label='Start')
    ax1.plot(*goal_world, 'r*', markersize=15, label='Goal')
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Costmap + A* Global Path")
    ax1.legend(fontsize=8)
    ax1.set_aspect('equal')

    # 2. Pure Pursuit tracking
    ax2 = axes[1]
    ax2.imshow(costmap.grid, origin='lower', cmap='RdYlGn_r',
               extent=[0, costmap.width * costmap.resolution,
                       0, costmap.height * costmap.resolution],
               vmin=0, vmax=254, alpha=0.3)
    ax2.plot(path_arr[:, 0], path_arr[:, 1], 'b--', lw=1, alpha=0.5,
             label='Global path')
    ax2.plot(trajectory[:, 0], trajectory[:, 1], 'g-', lw=2,
             label='Pure Pursuit trajectory')
    ax2.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8)
    ax2.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=12)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("Pure Pursuit Path Tracking")
    ax2.legend(fontsize=8)
    ax2.set_aspect('equal')

    # 3. Recovery behavior
    ax3 = axes[2]
    ax3.plot(recovery_arr[:, 0], recovery_arr[:, 1], 'o-', color='orange',
             markersize=4, label='Recovery maneuver')
    ax3.plot(recovery_start[0], recovery_start[1], 'rs', markersize=10,
             label='Stuck pose')
    # Draw heading arrows
    for i in range(0, len(recovery_arr), 3):
        dx = 0.15 * np.cos(recovery_arr[i, 2])
        dy = 0.15 * np.sin(recovery_arr[i, 2])
        ax3.arrow(recovery_arr[i, 0], recovery_arr[i, 1], dx, dy,
                  head_width=0.05, head_length=0.03, fc='orange', ec='orange')
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_title("Recovery: Backup + Rotate")
    ax3.legend(fontsize=8)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    plt.suptitle("ROS2 Nav2 Navigation Concepts", fontsize=14)
    plt.tight_layout()
    plt.savefig("16_ros2_navigation.png", dpi=120)
    plt.show()

    # Summary
    final_err = np.sqrt((trajectory[-1, 0] - goal_world[0])**2
                        + (trajectory[-1, 1] - goal_world[1])**2)
    print(f"\nPath length (A*): {len(world_path)} waypoints")
    print(f"Tracking steps: {len(trajectory)}")
    print(f"Final distance to goal: {final_err:.3f} m")
    print(f"Recovery maneuver: {len(recovery_poses)} poses "
          f"(backup {0.3:.1f} m + rotate {90:.0f} deg)")


if __name__ == "__main__":
    demo_nav2_concepts()
