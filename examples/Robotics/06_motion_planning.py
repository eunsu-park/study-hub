"""
Motion Planning: RRT and RRT* with Obstacle Avoidance
=====================================================
Sampling-based planning algorithms for navigating cluttered environments.

Motion planning answers: "How do I get from A to B without hitting obstacles?"

Grid-based methods (A*, Dijkstra) discretize the space and become impractical
in high dimensions. Sampling-based planners work by randomly exploring the
continuous space, building a tree of collision-free paths. They scale well
to high-dimensional configuration spaces (6+ DOF manipulators).

RRT (Rapidly-exploring Random Tree):
  - Grows a tree by randomly sampling points and extending toward them
  - Probabilistically complete: will find a path if one exists (given enough time)
  - Fast to find *a* path, but the path is usually suboptimal (jagged)

RRT* (asymptotically optimal):
  - Adds a rewiring step: when a new node is added, check if nearby nodes
    would benefit from routing through the new node
  - Converges to the optimal path as the number of samples → infinity
  - Slower per iteration than RRT, but produces much better paths
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class Obstacle:
    """Axis-aligned rectangular obstacle."""

    def __init__(self, x: float, y: float, w: float, h: float):
        self.x, self.y = x, y  # Bottom-left corner
        self.w, self.h = w, h

    def contains(self, point: np.ndarray) -> bool:
        """Check if a point is inside this obstacle (with small margin)."""
        margin = 0.05
        return (self.x - margin <= point[0] <= self.x + self.w + margin and
                self.y - margin <= point[1] <= self.y + self.h + margin)


class Environment:
    """2D planning environment with rectangular obstacles."""

    def __init__(self, x_range: Tuple[float, float] = (0, 10),
                 y_range: Tuple[float, float] = (0, 10)):
        self.x_range = x_range
        self.y_range = y_range
        self.obstacles: List[Obstacle] = []

    def add_obstacle(self, x: float, y: float, w: float, h: float):
        self.obstacles.append(Obstacle(x, y, w, h))

    def is_free(self, point: np.ndarray) -> bool:
        """Check if a point is in free space (not inside any obstacle)."""
        if not (self.x_range[0] <= point[0] <= self.x_range[1] and
                self.y_range[0] <= point[1] <= self.y_range[1]):
            return False
        return not any(obs.contains(point) for obs in self.obstacles)

    def is_edge_free(self, p1: np.ndarray, p2: np.ndarray,
                      resolution: float = 0.05) -> bool:
        """Check if a straight line between p1 and p2 is collision-free.

        We discretize the edge and check each point. The resolution determines
        how finely we sample — too coarse and we miss thin obstacles,
        too fine and it is slow. A good rule: resolution < min obstacle width / 2.
        """
        dist = np.linalg.norm(p2 - p1)
        n_checks = max(int(dist / resolution), 2)
        for i in range(n_checks + 1):
            t = i / n_checks
            point = p1 + t * (p2 - p1)
            if not self.is_free(point):
                return False
        return True


def create_test_environment() -> Environment:
    """Create a test environment with several obstacles."""
    env = Environment()
    env.add_obstacle(2, 2, 1, 4)    # Tall wall
    env.add_obstacle(5, 0, 1, 5)    # Bottom wall
    env.add_obstacle(5, 6, 1, 4)    # Top wall
    env.add_obstacle(7, 3, 2, 1)    # Horizontal block
    env.add_obstacle(1, 7, 3, 0.5)  # Shelf
    return env


# ---------------------------------------------------------------------------
# RRT
# ---------------------------------------------------------------------------
class RRTNode:
    """A node in the RRT tree."""

    def __init__(self, position: np.ndarray, parent: Optional[int] = None,
                 cost: float = 0.0):
        self.position = position
        self.parent = parent  # Index of parent node in the tree list
        self.cost = cost      # Cost from start (for RRT*)


def rrt(env: Environment, start: np.ndarray, goal: np.ndarray,
        max_iter: int = 2000, step_size: float = 0.5,
        goal_bias: float = 0.1, goal_tolerance: float = 0.3) -> Tuple[List[RRTNode], Optional[List[np.ndarray]]]:
    """Standard RRT algorithm.

    The core loop:
      1. Sample a random point (with goal_bias probability, sample the goal)
      2. Find the nearest node in the tree
      3. Extend toward the sample by step_size
      4. If collision-free, add the new node
      5. Check if we reached the goal

    goal_bias: probability of sampling the goal directly. This "pulls" the
    tree toward the goal. Too high → greedy and may get stuck; too low → slow.
    A typical value is 0.05-0.15.
    """
    tree = [RRTNode(start)]

    for i in range(max_iter):
        # Step 1: Random sampling with goal bias
        if np.random.random() < goal_bias:
            sample = goal.copy()
        else:
            sample = np.array([
                np.random.uniform(*env.x_range),
                np.random.uniform(*env.y_range)
            ])

        # Step 2: Find nearest node
        distances = [np.linalg.norm(node.position - sample) for node in tree]
        nearest_idx = int(np.argmin(distances))
        nearest = tree[nearest_idx]

        # Step 3: Steer toward sample
        direction = sample - nearest.position
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            continue
        direction = direction / dist
        new_pos = nearest.position + min(step_size, dist) * direction

        # Step 4: Collision check
        if env.is_edge_free(nearest.position, new_pos):
            new_cost = nearest.cost + np.linalg.norm(new_pos - nearest.position)
            new_node = RRTNode(new_pos, parent=nearest_idx, cost=new_cost)
            tree.append(new_node)

            # Step 5: Goal check
            if np.linalg.norm(new_pos - goal) < goal_tolerance:
                # Connect to goal
                if env.is_edge_free(new_pos, goal):
                    goal_node = RRTNode(goal, parent=len(tree) - 1,
                                         cost=new_cost + np.linalg.norm(goal - new_pos))
                    tree.append(goal_node)
                    path = extract_path(tree, len(tree) - 1)
                    return tree, path

    return tree, None  # No path found


def rrt_star(env: Environment, start: np.ndarray, goal: np.ndarray,
             max_iter: int = 3000, step_size: float = 0.5,
             goal_bias: float = 0.1, goal_tolerance: float = 0.3,
             rewire_radius: float = 1.5) -> Tuple[List[RRTNode], Optional[List[np.ndarray]]]:
    """RRT* — asymptotically optimal variant of RRT.

    The key difference from RRT is the rewiring step:
    After adding a new node, we check all nodes within rewire_radius.
    If routing through the new node gives a shorter path to any neighbor,
    we update that neighbor's parent.

    This gradually improves the tree quality toward optimality.
    The rewire_radius should shrink as gamma * (log(n)/n)^(1/d) for
    theoretical guarantees, but a fixed radius works well in practice.
    """
    tree = [RRTNode(start)]
    best_goal_idx = None
    best_goal_cost = float('inf')

    for i in range(max_iter):
        # Sample
        if np.random.random() < goal_bias:
            sample = goal.copy()
        else:
            sample = np.array([
                np.random.uniform(*env.x_range),
                np.random.uniform(*env.y_range)
            ])

        # Find nearest
        distances = [np.linalg.norm(node.position - sample) for node in tree]
        nearest_idx = int(np.argmin(distances))
        nearest = tree[nearest_idx]

        # Steer
        direction = sample - nearest.position
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            continue
        direction = direction / dist
        new_pos = nearest.position + min(step_size, dist) * direction

        if not env.is_edge_free(nearest.position, new_pos):
            continue

        # --- RRT* addition: choose best parent from nearby nodes ---
        new_idx = len(tree)
        near_indices = [j for j, node in enumerate(tree)
                        if np.linalg.norm(node.position - new_pos) < rewire_radius]

        best_parent = nearest_idx
        best_cost = nearest.cost + np.linalg.norm(new_pos - nearest.position)

        for j in near_indices:
            candidate_cost = tree[j].cost + np.linalg.norm(new_pos - tree[j].position)
            if candidate_cost < best_cost and env.is_edge_free(tree[j].position, new_pos):
                best_parent = j
                best_cost = candidate_cost

        new_node = RRTNode(new_pos, parent=best_parent, cost=best_cost)
        tree.append(new_node)

        # --- RRT* addition: rewire nearby nodes ---
        for j in near_indices:
            new_cost_via = best_cost + np.linalg.norm(tree[j].position - new_pos)
            if new_cost_via < tree[j].cost and env.is_edge_free(new_pos, tree[j].position):
                tree[j].parent = new_idx
                tree[j].cost = new_cost_via

        # Goal check
        if np.linalg.norm(new_pos - goal) < goal_tolerance:
            if env.is_edge_free(new_pos, goal):
                goal_cost = best_cost + np.linalg.norm(goal - new_pos)
                if goal_cost < best_goal_cost:
                    goal_node = RRTNode(goal, parent=new_idx, cost=goal_cost)
                    tree.append(goal_node)
                    best_goal_idx = len(tree) - 1
                    best_goal_cost = goal_cost

    if best_goal_idx is not None:
        path = extract_path(tree, best_goal_idx)
        return tree, path
    return tree, None


def extract_path(tree: List[RRTNode], goal_idx: int) -> List[np.ndarray]:
    """Trace back from goal to start through parent pointers."""
    path = []
    idx = goal_idx
    while idx is not None:
        path.append(tree[idx].position)
        idx = tree[idx].parent
    path.reverse()
    return path


def smooth_path(env: Environment, path: List[np.ndarray],
                iterations: int = 100) -> List[np.ndarray]:
    """Shortcut-based path smoothing.

    RRT paths are typically jagged because each segment follows the random
    sampling pattern. We improve the path by repeatedly:
      1. Pick two random waypoints along the path
      2. If the straight line between them is collision-free, remove all
         intermediate waypoints

    This is a simple but effective post-processing step.
    """
    path = [p.copy() for p in path]
    for _ in range(iterations):
        if len(path) <= 2:
            break
        i = np.random.randint(0, len(path) - 2)
        j = np.random.randint(i + 2, len(path))
        if env.is_edge_free(path[i], path[j]):
            path = path[:i + 1] + path[j:]
    return path


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_environment(ax, env: Environment):
    """Draw obstacles."""
    for obs in env.obstacles:
        rect = Rectangle((obs.x, obs.y), obs.w, obs.h,
                          facecolor='gray', edgecolor='black', alpha=0.7)
        ax.add_patch(rect)


def plot_tree(ax, tree: List[RRTNode], color='lightblue', alpha=0.3):
    """Draw the RRT tree edges."""
    for node in tree:
        if node.parent is not None:
            parent = tree[node.parent]
            ax.plot([node.position[0], parent.position[0]],
                    [node.position[1], parent.position[1]],
                    color=color, linewidth=0.5, alpha=alpha)


def demo_motion_planning():
    """Demonstrate RRT and RRT* path planning."""
    print("=" * 60)
    print("Motion Planning (RRT / RRT*) Demo")
    print("=" * 60)

    env = create_test_environment()
    start = np.array([1.0, 1.0])
    goal = np.array([9.0, 9.0])

    np.random.seed(42)  # For reproducibility

    # Run RRT
    print("\nRunning RRT...")
    tree_rrt, path_rrt = rrt(env, start, goal, max_iter=3000)
    rrt_found = path_rrt is not None
    print(f"  RRT: {'Found' if rrt_found else 'No'} path, "
          f"tree size = {len(tree_rrt)} nodes")
    if path_rrt:
        cost_rrt = sum(np.linalg.norm(np.array(path_rrt[i+1]) - np.array(path_rrt[i]))
                        for i in range(len(path_rrt) - 1))
        print(f"  RRT path cost: {cost_rrt:.2f}")

    # Run RRT*
    print("\nRunning RRT*...")
    np.random.seed(42)
    tree_rrt_star, path_star = rrt_star(env, start, goal, max_iter=3000)
    star_found = path_star is not None
    print(f"  RRT*: {'Found' if star_found else 'No'} path, "
          f"tree size = {len(tree_rrt_star)} nodes")
    if path_star:
        cost_star = sum(np.linalg.norm(np.array(path_star[i+1]) - np.array(path_star[i]))
                         for i in range(len(path_star) - 1))
        print(f"  RRT* path cost: {cost_star:.2f}")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # RRT
    ax = axes[0]
    plot_environment(ax, env)
    plot_tree(ax, tree_rrt, color='lightblue')
    if path_rrt:
        path_arr = np.array(path_rrt)
        ax.plot(path_arr[:, 0], path_arr[:, 1], 'b-', linewidth=2.5, label='RRT path')
    ax.plot(*start, 'go', markersize=12, label='Start')
    ax.plot(*goal, 'r*', markersize=15, label='Goal')
    ax.set_xlim(env.x_range)
    ax.set_ylim(env.y_range)
    ax.set_aspect('equal')
    ax.set_title(f"RRT (cost: {cost_rrt:.2f})" if rrt_found else "RRT (no path)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # RRT*
    ax = axes[1]
    plot_environment(ax, env)
    plot_tree(ax, tree_rrt_star, color='lightyellow')
    if path_star:
        path_arr = np.array(path_star)
        ax.plot(path_arr[:, 0], path_arr[:, 1], 'r-', linewidth=2.5, label='RRT* path')
    ax.plot(*start, 'go', markersize=12, label='Start')
    ax.plot(*goal, 'r*', markersize=15, label='Goal')
    ax.set_xlim(env.x_range)
    ax.set_ylim(env.y_range)
    ax.set_aspect('equal')
    ax.set_title(f"RRT* (cost: {cost_star:.2f})" if star_found else "RRT* (no path)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Smoothed path
    ax = axes[2]
    plot_environment(ax, env)
    best_path = path_star if star_found else path_rrt
    if best_path:
        smoothed = smooth_path(env, best_path, iterations=200)
        path_arr = np.array(best_path)
        smooth_arr = np.array(smoothed)
        ax.plot(path_arr[:, 0], path_arr[:, 1], '--', color='gray',
                linewidth=1.5, alpha=0.5, label='Original')
        ax.plot(smooth_arr[:, 0], smooth_arr[:, 1], 'g-', linewidth=2.5,
                label='Smoothed')
        cost_smooth = sum(np.linalg.norm(smooth_arr[i+1] - smooth_arr[i])
                          for i in range(len(smooth_arr) - 1))
        print(f"  Smoothed path cost: {cost_smooth:.2f}")
    ax.plot(*start, 'go', markersize=12, label='Start')
    ax.plot(*goal, 'r*', markersize=15, label='Goal')
    ax.set_xlim(env.x_range)
    ax.set_ylim(env.y_range)
    ax.set_aspect('equal')
    ax.set_title("Path Smoothing")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.suptitle("RRT vs RRT* Motion Planning", fontsize=14)
    plt.tight_layout()
    plt.savefig("06_motion_planning.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    demo_motion_planning()
