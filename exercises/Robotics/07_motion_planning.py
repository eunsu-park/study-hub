"""
Exercises for Lesson 07: Motion Planning
Topic: Robotics
Solutions to practice problems from the lesson.
"""

import numpy as np


def fk_2link(q, l1, l2):
    """Forward kinematics for 2-link planar robot. Returns list of link endpoints."""
    x0, y0 = 0, 0
    x1 = l1 * np.cos(q[0])
    y1 = l1 * np.sin(q[0])
    x2 = x1 + l2 * np.cos(q[0] + q[1])
    y2 = y1 + l2 * np.sin(q[0] + q[1])
    return [(x0, y0), (x1, y1), (x2, y2)]


def arm_collides(q, l1, l2, obstacles, n_samples=10):
    """Check if 2-link arm collides with circular obstacles."""
    points = fk_2link(q, l1, l2)
    for seg_start, seg_end in [(points[0], points[1]), (points[1], points[2])]:
        for t in np.linspace(0, 1, n_samples):
            px = seg_start[0] + t * (seg_end[0] - seg_start[0])
            py = seg_start[1] + t * (seg_end[1] - seg_start[1])
            for ox, oy, r in obstacles:
                if (px - ox)**2 + (py - oy)**2 < r**2:
                    return True
    return False


def exercise_1():
    """
    Exercise 1: C-Space Visualization
    2-link robot, l1=1.0, l2=0.8, circular obstacle at (0.8, 0.8) r=0.3.
    """
    l1, l2 = 1.0, 0.8
    obstacles = [(0.8, 0.8, 0.3)]
    resolution = 2  # degrees

    n = int(360 / resolution)
    theta1_range = np.linspace(-np.pi, np.pi, n, endpoint=False)
    theta2_range = np.linspace(-np.pi, np.pi, n, endpoint=False)

    # Build C-space obstacle map
    c_space = np.zeros((n, n), dtype=bool)
    for i, t1 in enumerate(theta1_range):
        for j, t2 in enumerate(theta2_range):
            c_space[i, j] = arm_collides([t1, t2], l1, l2, obstacles)

    n_collision = np.sum(c_space)
    n_free = np.sum(~c_space)
    total = n * n

    print(f"C-Space obstacle map (resolution={resolution}°)")
    print(f"  Grid size: {n} x {n} = {total} cells")
    print(f"  Collision cells: {n_collision} ({100 * n_collision / total:.1f}%)")
    print(f"  Free cells: {n_free} ({100 * n_free / total:.1f}%)")

    # Repeat with obstacle at (0, 1.2)
    obstacles2 = [(0.0, 1.2, 0.3)]
    c_space2 = np.zeros((n, n), dtype=bool)
    for i, t1 in enumerate(theta1_range):
        for j, t2 in enumerate(theta2_range):
            c_space2[i, j] = arm_collides([t1, t2], l1, l2, obstacles2)

    n_collision2 = np.sum(c_space2)
    print(f"\n  With obstacle at (0, 1.2):")
    print(f"  Collision cells: {n_collision2} ({100 * n_collision2 / total:.1f}%)")
    print(f"  The C-space obstacle shape changes with physical obstacle position.")
    print(f"  At (0.8, 0.8), the obstacle is in the common workspace of both links.")
    print(f"  At (0, 1.2), it's above the base — different arm configurations collide.")


def exercise_2():
    """
    Exercise 2: Potential Field Planner for 2-link robot (task space).
    """
    l1, l2 = 1.0, 0.8

    def fk_ee(q):
        x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
        y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
        return np.array([x, y])

    def jacobian(q):
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
        return np.array([
            [-l1*s1 - l2*s12, -l2*s12],
            [ l1*c1 + l2*c12,  l2*c12]
        ])

    def potential_field_planner(q_start, x_goal, obstacles, k_att=1.0, k_rep=0.5,
                                 rho_0=0.5, max_iter=5000, alpha=0.01):
        q = q_start.copy()
        path = [q.copy()]

        for i in range(max_iter):
            x_ee = fk_ee(q)
            J = jacobian(q)

            # Attractive force (in task space)
            diff = x_goal - x_ee
            dist_goal = np.linalg.norm(diff)
            if dist_goal < 0.01:
                return path, True

            F_att = k_att * diff

            # Repulsive force (from obstacles)
            F_rep = np.zeros(2)
            for ox, oy, r in obstacles:
                obs = np.array([ox, oy])
                d = np.linalg.norm(x_ee - obs) - r
                if d < rho_0 and d > 0.01:
                    grad = (x_ee - obs) / np.linalg.norm(x_ee - obs)
                    F_rep += k_rep * (1.0/d - 1.0/rho_0) * (1.0/d**2) * grad

            F_total = F_att + F_rep
            # Map to joint space: dq = J^T * F
            dq = alpha * J.T @ F_total
            q = q + dq
            path.append(q.copy())

        return path, False

    # Case 1: successful
    q_start = np.radians([30, 60])
    x_goal = np.array([0.5, 1.0])
    obstacles = [(0.8, 0.8, 0.15)]

    path, success = potential_field_planner(q_start, x_goal, obstacles)
    print(f"Case 1: start=(30°, 60°), goal at (0.5, 1.0)")
    print(f"  Success: {success}, Steps: {len(path)}")
    if success:
        x_final = fk_ee(path[-1])
        print(f"  Final EE position: ({x_final[0]:.4f}, {x_final[1]:.4f})")

    # Case 2: local minimum (obstacle between start and goal)
    q_start2 = np.radians([0, 30])
    x_goal2 = np.array([-0.5, 0.8])
    obstacles2 = [(0.0, 1.0, 0.25)]

    path2, success2 = potential_field_planner(q_start2, x_goal2, obstacles2, max_iter=3000)
    print(f"\nCase 2: potential local minimum scenario")
    print(f"  Success: {success2}, Steps: {len(path2)}")

    # Case 3: random escape
    def potential_field_with_escape(q_start, x_goal, obstacles, k_att=1.0, k_rep=0.5,
                                     rho_0=0.5, max_iter=5000, alpha=0.01):
        q = q_start.copy()
        path = [q.copy()]
        stuck_count = 0

        for i in range(max_iter):
            x_ee = fk_ee(q)
            J = jacobian(q)
            diff = x_goal - x_ee
            dist_goal = np.linalg.norm(diff)
            if dist_goal < 0.01:
                return path, True

            F_att = k_att * diff
            F_rep = np.zeros(2)
            for ox, oy, r in obstacles:
                obs = np.array([ox, oy])
                d = np.linalg.norm(x_ee - obs) - r
                if d < rho_0 and d > 0.01:
                    grad = (x_ee - obs) / np.linalg.norm(x_ee - obs)
                    F_rep += k_rep * (1.0/d - 1.0/rho_0) * (1.0/d**2) * grad

            F_total = F_att + F_rep
            grad_mag = np.linalg.norm(F_total)

            if grad_mag < 0.1:
                stuck_count += 1
                if stuck_count > 50:
                    # Random escape
                    q += np.random.randn(2) * 0.3
                    stuck_count = 0
            else:
                stuck_count = 0

            dq = alpha * J.T @ F_total
            q = q + dq
            path.append(q.copy())

        return path, False

    path3, success3 = potential_field_with_escape(q_start2, x_goal2, obstacles2, max_iter=5000)
    print(f"\nCase 3: with random escape strategy")
    print(f"  Success: {success3}, Steps: {len(path3)}")
    print(f"  Random escape helps escape local minima but is not guaranteed.")


def exercise_3():
    """
    Exercise 3: RRT Implementation in C-space.
    """
    l1, l2 = 1.0, 0.8
    obstacles = [(0.8, 0.8, 0.3)]
    step_size = 0.1
    max_iter = 2000

    class RRT:
        def __init__(self, q_start, q_goal, step_size=0.1, max_iter=2000):
            self.q_start = q_start
            self.q_goal = q_goal
            self.step_size = step_size
            self.max_iter = max_iter
            self.tree = [q_start.copy()]
            self.parent = [-1]

        def random_config(self, goal_bias=0.1):
            if np.random.random() < goal_bias:
                return self.q_goal.copy()
            return np.random.uniform(-np.pi, np.pi, 2)

        def nearest(self, q):
            dists = [np.linalg.norm(q - node) for node in self.tree]
            return np.argmin(dists)

        def steer(self, q_near, q_rand):
            diff = q_rand - q_near
            dist = np.linalg.norm(diff)
            if dist <= self.step_size:
                return q_rand
            return q_near + (diff / dist) * self.step_size

        def collision_free(self, q1, q2, n_check=10):
            for t in np.linspace(0, 1, n_check):
                q = q1 + t * (q2 - q1)
                if arm_collides(q, l1, l2, obstacles, n_samples=5):
                    return False
            return True

        def plan(self):
            for i in range(self.max_iter):
                q_rand = self.random_config()
                idx_near = self.nearest(q_rand)
                q_near = self.tree[idx_near]
                q_new = self.steer(q_near, q_rand)

                if self.collision_free(q_near, q_new):
                    self.tree.append(q_new)
                    self.parent.append(idx_near)

                    if np.linalg.norm(q_new - self.q_goal) < self.step_size:
                        return self.extract_path(len(self.tree) - 1), i + 1
            return None, self.max_iter

        def extract_path(self, idx):
            path = []
            while idx != -1:
                path.append(self.tree[idx])
                idx = self.parent[idx]
            return path[::-1]

    q_start = np.radians([30, 60])
    q_goal = np.radians([-45, -30])

    print(f"RRT in C-space for 2-link robot")
    print(f"  Step size: {step_size}, Max iterations: {max_iter}")

    np.random.seed(42)
    n_trials = 10
    path_lengths = []
    iterations_list = []

    for trial in range(n_trials):
        np.random.seed(trial)
        rrt = RRT(q_start, q_goal, step_size, max_iter)
        path, iters = rrt.plan()

        if path is not None:
            # Compute path length
            length = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
                         for i in range(len(path) - 1))
            path_lengths.append(length)
            iterations_list.append(iters)

    success_rate = len(path_lengths) / n_trials
    print(f"\n  Results over {n_trials} trials:")
    print(f"    Success rate: {success_rate * 100:.0f}%")
    if path_lengths:
        print(f"    Path length: mean={np.mean(path_lengths):.3f}, "
              f"std={np.std(path_lengths):.3f}")
        print(f"    Iterations: mean={np.mean(iterations_list):.0f}")
        print(f"    Path length variance: {np.var(path_lengths):.6f}")
        print(f"    (High variance is expected — RRT is randomized)")


def exercise_4():
    """
    Exercise 4: RRT vs RRT*
    """
    l1, l2 = 1.0, 0.8
    obstacles = [(0.8, 0.8, 0.3)]

    class RRTStar:
        def __init__(self, q_start, q_goal, step_size=0.1, max_iter=2000, gamma=1.0):
            self.q_start = q_start
            self.q_goal = q_goal
            self.step_size = step_size
            self.max_iter = max_iter
            self.gamma = gamma
            self.tree = [q_start.copy()]
            self.parent = [-1]
            self.cost = [0.0]

        def plan(self):
            best_path = None
            best_cost = float('inf')

            for i in range(self.max_iter):
                q_rand = (self.q_goal.copy() if np.random.random() < 0.1
                          else np.random.uniform(-np.pi, np.pi, 2))

                idx_near = self._nearest(q_rand)
                q_near = self.tree[idx_near]
                q_new = self._steer(q_near, q_rand)

                if not self._collision_free(q_near, q_new):
                    continue

                # Find nearby nodes
                n = len(self.tree)
                r = min(self.gamma * np.sqrt(np.log(n + 1) / (n + 1)), self.step_size * 3)
                near_indices = [j for j, node in enumerate(self.tree)
                                if np.linalg.norm(node - q_new) < r]

                # Choose best parent
                best_parent = idx_near
                best_parent_cost = self.cost[idx_near] + np.linalg.norm(q_new - q_near)

                for j in near_indices:
                    new_cost = self.cost[j] + np.linalg.norm(q_new - self.tree[j])
                    if new_cost < best_parent_cost and self._collision_free(self.tree[j], q_new):
                        best_parent = j
                        best_parent_cost = new_cost

                self.tree.append(q_new)
                self.parent.append(best_parent)
                self.cost.append(best_parent_cost)

                # Rewire
                idx_new = len(self.tree) - 1
                for j in near_indices:
                    new_cost = self.cost[idx_new] + np.linalg.norm(self.tree[j] - q_new)
                    if new_cost < self.cost[j] and self._collision_free(q_new, self.tree[j]):
                        self.parent[j] = idx_new
                        self.cost[j] = new_cost

                # Check goal
                if np.linalg.norm(q_new - self.q_goal) < self.step_size:
                    path_cost = self.cost[-1]
                    if path_cost < best_cost:
                        best_cost = path_cost
                        best_path = self._extract_path(len(self.tree) - 1)

            return best_path, best_cost

        def _nearest(self, q):
            return min(range(len(self.tree)),
                       key=lambda i: np.linalg.norm(q - self.tree[i]))

        def _steer(self, q_near, q_rand):
            diff = q_rand - q_near
            dist = np.linalg.norm(diff)
            if dist <= self.step_size:
                return q_rand
            return q_near + (diff / dist) * self.step_size

        def _collision_free(self, q1, q2, n_check=10):
            for t in np.linspace(0, 1, n_check):
                q = q1 + t * (q2 - q1)
                if arm_collides(q, l1, l2, obstacles, n_samples=5):
                    return False
            return True

        def _extract_path(self, idx):
            path = []
            while idx != -1:
                path.append(self.tree[idx])
                idx = self.parent[idx]
            return path[::-1]

    q_start = np.radians([30, 60])
    q_goal = np.radians([-45, -30])

    print("RRT vs RRT* path cost comparison:")
    print(f"{'N samples':>10} | {'RRT cost':>10} | {'RRT* cost':>10}")
    print("-" * 40)

    for N in [500, 1000, 2000]:
        np.random.seed(42)

        # RRT (simple version — just use RRTStar without rewiring, or quick RRT)
        # Simple RRT cost
        class SimpleRRT:
            def __init__(self):
                self.tree = [q_start.copy()]
                self.parent = [-1]
                self.cost = [0.0]

            def plan(self, max_iter):
                for _ in range(max_iter):
                    q_rand = (q_goal.copy() if np.random.random() < 0.1
                              else np.random.uniform(-np.pi, np.pi, 2))
                    dists = [np.linalg.norm(q_rand - n) for n in self.tree]
                    idx = np.argmin(dists)
                    q_near = self.tree[idx]
                    diff = q_rand - q_near
                    dist = np.linalg.norm(diff)
                    q_new = q_near + (diff / dist) * min(dist, 0.1)
                    collides = False
                    for t in np.linspace(0, 1, 10):
                        q = q_near + t * (q_new - q_near)
                        if arm_collides(q, l1, l2, obstacles, 5):
                            collides = True
                            break
                    if not collides:
                        self.tree.append(q_new)
                        self.parent.append(idx)
                        self.cost.append(self.cost[idx] + np.linalg.norm(q_new - q_near))
                        if np.linalg.norm(q_new - q_goal) < 0.1:
                            return self.cost[-1]
                return float('inf')

        np.random.seed(42)
        rrt = SimpleRRT()
        rrt_cost = rrt.plan(N)

        np.random.seed(42)
        rrt_star = RRTStar(q_start, q_goal, max_iter=N)
        _, rrt_star_cost = rrt_star.plan()

        print(f"{N:>10} | {rrt_cost:>10.4f} | {rrt_star_cost:>10.4f}")

    print(f"\nRRT* paths improve with more samples (asymptotic optimality)")
    print(f"RRT paths do not systematically improve — first-found path is returned.")


def exercise_5():
    """
    Exercise 5: Path Post-Processing (shortcutting + smoothing).
    """
    l1, l2 = 1.0, 0.8
    obstacles = [(0.8, 0.8, 0.3)]

    def collision_free_line(q1, q2, n_check=20):
        for t in np.linspace(0, 1, n_check):
            q = q1 + t * (q2 - q1)
            if arm_collides(q, l1, l2, obstacles, 5):
                return False
        return True

    # Generate a jerky path (simulate RRT-like path)
    np.random.seed(42)
    raw_path = [
        np.radians([30, 60]),
        np.radians([25, 50]),
        np.radians([20, 45]),
        np.radians([10, 35]),
        np.radians([5, 25]),
        np.radians([-5, 15]),
        np.radians([-15, 5]),
        np.radians([-25, -5]),
        np.radians([-35, -15]),
        np.radians([-40, -25]),
        np.radians([-45, -30]),
    ]

    def path_length(path):
        return sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path) - 1))

    raw_len = path_length(raw_path)
    print(f"Raw path: {len(raw_path)} waypoints, length = {raw_len:.4f} rad")

    # Shortcutting
    shortcut_path = raw_path.copy()
    n_attempts = 100
    for _ in range(n_attempts):
        if len(shortcut_path) <= 2:
            break
        i = np.random.randint(0, len(shortcut_path) - 2)
        j = np.random.randint(i + 2, len(shortcut_path))
        if collision_free_line(shortcut_path[i], shortcut_path[j]):
            shortcut_path = shortcut_path[:i+1] + shortcut_path[j:]

    shortcut_len = path_length(shortcut_path)
    print(f"After shortcutting: {len(shortcut_path)} waypoints, length = {shortcut_len:.4f} rad")
    print(f"  Reduction: {(1 - shortcut_len / raw_len) * 100:.1f}%")

    # Smoothing (simple averaging)
    smooth_path = [p.copy() for p in shortcut_path]
    for iteration in range(10):
        new_path = [smooth_path[0].copy()]
        for i in range(1, len(smooth_path) - 1):
            avg = 0.25 * smooth_path[i-1] + 0.5 * smooth_path[i] + 0.25 * smooth_path[i+1]
            new_path.append(avg)
        new_path.append(smooth_path[-1].copy())
        smooth_path = new_path

    smooth_len = path_length(smooth_path)
    print(f"After smoothing: {len(smooth_path)} waypoints, length = {smooth_len:.4f} rad")

    # Verify collision-free
    all_free = True
    for i in range(len(smooth_path) - 1):
        if not collision_free_line(smooth_path[i], smooth_path[i + 1], 20):
            all_free = False
            break
    print(f"Smoothed path collision-free: {all_free}")

    print(f"\nSummary:")
    print(f"  Raw RRT:       length = {raw_len:.4f}")
    print(f"  Shortcutted:   length = {shortcut_len:.4f} ({(1-shortcut_len/raw_len)*100:.1f}% reduction)")
    print(f"  Smoothed:      length = {smooth_len:.4f} ({(1-smooth_len/raw_len)*100:.1f}% total reduction)")


if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 07: Motion Planning — Exercise Solutions")
    print("=" * 70)

    print("\n--- Exercise 1: C-Space Visualization ---")
    exercise_1()

    print("\n--- Exercise 2: Potential Field Tuning ---")
    exercise_2()

    print("\n--- Exercise 3: RRT Implementation ---")
    exercise_3()

    print("\n--- Exercise 4: RRT vs RRT* ---")
    exercise_4()

    print("\n--- Exercise 5: Post-Processing ---")
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
