"""
Exercises for Lesson 16: Multi-Robot Systems and Swarms
Topic: Robotics
Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def exercise_1():
    """
    Exercise 1: Auction-Based Task Allocation
    5 robots, 8 tasks. Compare sequential auction vs Hungarian algorithm.
    """
    np.random.seed(42)
    n_robots = 5
    n_tasks = 8

    robot_positions = np.random.uniform(0, 10, (n_robots, 2))
    task_positions = np.random.uniform(0, 10, (n_tasks, 2))

    # Cost matrix: distance from each robot to each task
    cost_matrix = np.zeros((n_robots, n_tasks))
    for i in range(n_robots):
        for j in range(n_tasks):
            cost_matrix[i, j] = np.linalg.norm(robot_positions[i] - task_positions[j])

    # Sequential auction
    def sequential_auction(cost_matrix):
        n_r, n_t = cost_matrix.shape
        assignments = {}
        assigned_tasks = set()
        total_cost = 0

        for _ in range(min(n_r, n_t)):
            best_bid = float('inf')
            best_robot = -1
            best_task = -1

            for r in range(n_r):
                if r in assignments:
                    continue
                for t in range(n_t):
                    if t in assigned_tasks:
                        continue
                    if cost_matrix[r, t] < best_bid:
                        best_bid = cost_matrix[r, t]
                        best_robot = r
                        best_task = t

            if best_robot >= 0:
                assignments[best_robot] = best_task
                assigned_tasks.add(best_task)
                total_cost += best_bid

        return assignments, total_cost

    # Hungarian algorithm (optimal)
    def hungarian_assignment(cost_matrix):
        # Pad if non-square
        n_r, n_t = cost_matrix.shape
        if n_r < n_t:
            padded = np.zeros((n_t, n_t))
            padded[:n_r, :] = cost_matrix
            padded[n_r:, :] = 1e6  # dummy robots with high cost
        else:
            padded = cost_matrix

        row_ind, col_ind = linear_sum_assignment(padded)
        assignments = {}
        total_cost = 0
        for r, t in zip(row_ind, col_ind):
            if r < n_r:
                assignments[r] = t
                total_cost += cost_matrix[r, t]
        return assignments, total_cost

    auction_assign, auction_cost = sequential_auction(cost_matrix)
    hungarian_assign, hungarian_cost = hungarian_assignment(cost_matrix)

    print("Auction-Based Task Allocation")
    print(f"  {n_robots} robots, {n_tasks} tasks")

    print(f"\n  Sequential Auction:")
    for r, t in sorted(auction_assign.items()):
        print(f"    Robot {r} → Task {t} (dist={cost_matrix[r, t]:.2f})")
    print(f"  Total cost: {auction_cost:.4f}")

    print(f"\n  Hungarian (Optimal):")
    for r, t in sorted(hungarian_assign.items()):
        print(f"    Robot {r} → Task {t} (dist={cost_matrix[r, t]:.2f})")
    print(f"  Total cost: {hungarian_cost:.4f}")

    ratio = auction_cost / hungarian_cost
    print(f"\n  Auction / Optimal ratio: {ratio:.3f}")
    print(f"  Auction is {(ratio - 1) * 100:.1f}% above optimal")
    print(f"  Note: Auction is greedy and suboptimal but distributed;")
    print(f"  Hungarian is centralized and optimal.")


def exercise_2():
    """
    Exercise 2: Leader-Follower Formation Control
    4 robots in diamond pattern, leader follows figure-eight.
    """
    np.random.seed(42)
    dt = 0.05
    T = 20.0
    steps = int(T / dt)

    # Desired formation: diamond pattern (offsets from leader)
    # Leader at center, 3 followers
    offsets = {
        "F1": np.array([-1.0, 0.0]),   # left
        "F2": np.array([1.0, 0.0]),    # right
        "F3": np.array([0.0, -1.0]),   # behind
    }

    # Leader trajectory: figure-eight
    def leader_pos(t):
        x = 3 * np.sin(2 * np.pi * t / T)
        y = 1.5 * np.sin(4 * np.pi * t / T)
        return np.array([x, y])

    def leader_vel(t):
        x = 3 * 2 * np.pi / T * np.cos(2 * np.pi * t / T)
        y = 1.5 * 4 * np.pi / T * np.cos(4 * np.pi * t / T)
        return np.array([x, y])

    # Leader heading (for rotating formation)
    def leader_heading(t):
        v = leader_vel(t)
        return np.arctan2(v[1], v[0])

    # Follower control: PD tracking of desired position
    Kp, Kd = 3.0, 2.0
    followers = {name: leader_pos(0) + offset for name, offset in offsets.items()}
    follower_vels = {name: np.zeros(2) for name in offsets}

    formation_errors = {name: [] for name in offsets}

    for step in range(steps):
        t = step * dt
        l_pos = leader_pos(t)
        l_heading = leader_heading(t)
        l_vel = leader_vel(t)

        # Rotation matrix for formation alignment with leader heading
        c, s = np.cos(l_heading), np.sin(l_heading)
        R = np.array([[c, -s], [s, c]])

        for name, offset in offsets.items():
            # Desired position in world frame
            desired = l_pos + R @ offset
            desired_vel = l_vel  # approximate

            # PD control
            pos_error = desired - followers[name]
            vel_error = desired_vel - follower_vels[name]
            acc = Kp * pos_error + Kd * vel_error

            follower_vels[name] += acc * dt
            followers[name] += follower_vels[name] * dt

            formation_errors[name].append(np.linalg.norm(pos_error))

    print("Leader-Follower Formation Control")
    print(f"  Diamond formation, figure-eight trajectory")
    print(f"  Duration: {T}s, dt={dt}s")

    print(f"\n  {'Follower':>10} | {'Mean error (m)':>14} | {'Max error (m)':>14}")
    print("  " + "-" * 45)
    for name in offsets:
        errs = formation_errors[name]
        print(f"  {name:>10} | {np.mean(errs):>14.4f} | {np.max(errs):>14.4f}")

    print(f"\n  Sharp turns cause increased formation error because followers")
    print(f"  must rapidly change direction. The tracking delay creates")
    print(f"  transient deformation of the formation pattern.")
    print(f"  At the figure-eight crossing point, leader changes direction")
    print(f"  quickly, resulting in the largest formation errors.")


def exercise_3():
    """
    Exercise 3: Reynolds Flocking for 30 robots.
    """
    np.random.seed(42)
    n_robots = 30
    dt = 0.05
    T = 15.0
    steps = int(T / dt)
    area_size = 20.0

    # Initial random positions and velocities
    positions = np.random.uniform(0, area_size, (n_robots, 2))
    velocities = np.random.uniform(-0.5, 0.5, (n_robots, 2))
    max_speed = 2.0
    max_force = 1.0

    def reynolds_flocking(positions, velocities, w_sep=1.5, w_ali=1.0, w_coh=1.0,
                           r_sep=1.5, r_neighbor=5.0):
        """Compute Reynolds flocking forces for all agents."""
        n = len(positions)
        forces = np.zeros((n, 2))

        for i in range(n):
            sep_force = np.zeros(2)
            ali_force = np.zeros(2)
            coh_force = np.zeros(2)
            n_neighbors = 0

            for j in range(n):
                if i == j:
                    continue
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)

                if dist < r_neighbor:
                    n_neighbors += 1

                    # Separation: repel from too-close neighbors
                    if dist < r_sep and dist > 0.01:
                        sep_force += diff / dist / dist  # inverse distance

                    # Alignment: match velocity of neighbors
                    ali_force += velocities[j]

                    # Cohesion: move toward centroid of neighbors
                    coh_force += positions[j]

            if n_neighbors > 0:
                ali_force = ali_force / n_neighbors - velocities[i]
                coh_force = coh_force / n_neighbors - positions[i]

            total = w_sep * sep_force + w_ali * ali_force + w_coh * coh_force
            mag = np.linalg.norm(total)
            if mag > max_force:
                total = total / mag * max_force
            forces[i] = total

        return forces

    # Run three experiments
    experiments = [
        ("Baseline", {"w_sep": 1.5, "w_ali": 1.0, "w_coh": 1.0}),
        ("High separation", {"w_sep": 5.0, "w_ali": 1.0, "w_coh": 1.0}),
        ("High cohesion", {"w_sep": 1.5, "w_ali": 1.0, "w_coh": 5.0}),
        ("No alignment", {"w_sep": 1.5, "w_ali": 0.0, "w_coh": 1.0}),
    ]

    print("Reynolds Flocking: 30 Robots")
    print(f"  Area: {area_size}x{area_size}m, Duration: {T}s")

    for exp_name, weights in experiments:
        pos = positions.copy()
        vel = velocities.copy()

        for step in range(steps):
            forces = reynolds_flocking(pos, vel, **weights)
            vel += forces * dt
            # Limit speed
            speeds = np.linalg.norm(vel, axis=1, keepdims=True)
            vel = np.where(speeds > max_speed, vel / speeds * max_speed, vel)
            pos += vel * dt
            # Wrap around
            pos = pos % area_size

        # Metrics
        centroid = np.mean(pos, axis=0)
        spread = np.mean(np.linalg.norm(pos - centroid, axis=1))
        mean_speed = np.mean(np.linalg.norm(vel, axis=1))
        # Alignment: average cosine similarity of velocities
        mean_vel = np.mean(vel, axis=0)
        mean_vel_norm = np.linalg.norm(mean_vel)
        alignment = mean_vel_norm / (mean_speed + 1e-10)

        print(f"\n  {exp_name}:")
        print(f"    Spread (avg dist from centroid): {spread:.2f} m")
        print(f"    Alignment (0=random, 1=perfect): {alignment:.3f}")
        print(f"    Mean speed: {mean_speed:.2f} m/s")

    print(f"\n  (a) High separation: robots spread apart, less cohesive group")
    print(f"  (b) High cohesion: robots cluster tightly, may collide")
    print(f"  (c) No alignment: robots move toward centroid but without")
    print(f"      coordinated direction — more chaotic, less flocking behavior")


def exercise_4():
    """
    Exercise 4: Consensus Convergence
    10 robots, three topologies: fully connected, ring, star.
    """
    np.random.seed(42)
    n = 10
    x0 = np.random.uniform(0, 10, n)  # initial values
    epsilon = 0.1  # consensus step size
    tol = 0.01
    max_iter = 1000

    def build_adjacency(topology, n):
        """Build adjacency matrix for given topology."""
        A = np.zeros((n, n))
        if topology == "fully_connected":
            A = np.ones((n, n)) - np.eye(n)
        elif topology == "ring":
            for i in range(n):
                A[i, (i + 1) % n] = 1
                A[i, (i - 1) % n] = 1
        elif topology == "star":
            # Node 0 is the hub
            for i in range(1, n):
                A[0, i] = 1
                A[i, 0] = 1
        return A

    def run_consensus(A, x0, epsilon, max_iter, tol):
        """Run linear consensus protocol."""
        x = x0.copy()
        n = len(x)
        history = [x.copy()]

        for iteration in range(max_iter):
            x_new = x.copy()
            for i in range(n):
                neighbors = np.where(A[i] > 0)[0]
                if len(neighbors) > 0:
                    for j in neighbors:
                        x_new[i] += epsilon * (x[j] - x[i])
            x = x_new
            history.append(x.copy())

            # Check convergence
            if np.max(x) - np.min(x) < tol:
                return iteration + 1, x, np.array(history)

        return max_iter, x, np.array(history)

    print("Consensus Protocol: 10 Robots")
    print(f"  Initial values: {x0.round(2)}")
    print(f"  Target consensus: mean = {np.mean(x0):.4f}")
    print(f"  epsilon = {epsilon}, tolerance = {tol}")

    topologies = ["fully_connected", "ring", "star"]

    print(f"\n  {'Topology':>18} | {'Iterations':>10} | {'Final range':>12} | {'Final mean':>11}")
    print("  " + "-" * 60)

    for topo in topologies:
        A = build_adjacency(topo, n)
        iters, x_final, history = run_consensus(A, x0, epsilon, max_iter, tol)
        final_range = np.max(x_final) - np.min(x_final)
        final_mean = np.mean(x_final)

        print(f"  {topo:>18} | {iters:>10} | {final_range:>12.6f} | {final_mean:>11.4f}")

    # Graph connectivity analysis
    print(f"\n  Graph connectivity and convergence:")
    for topo in topologies:
        A = build_adjacency(topo, n)
        D = np.diag(np.sum(A, axis=1))
        L = D - A  # Laplacian
        eigvals = np.sort(np.real(np.linalg.eigvals(L)))
        lambda2 = eigvals[1]  # algebraic connectivity
        print(f"    {topo:>18}: lambda_2 (algebraic connectivity) = {lambda2:.4f}")

    print(f"\n  Higher algebraic connectivity => faster convergence.")
    print(f"  Fully connected has highest lambda_2 (fastest).")
    print(f"  Ring has lowest lambda_2 (slowest — information must propagate around).")


def exercise_5():
    """
    Exercise 5: Multi-Robot Voronoi Coverage
    6 robots covering a 20x20 area with sensing radius 3m.
    """
    np.random.seed(42)
    n_robots = 6
    area_size = 20.0
    sensing_radius = 3.0
    dt = 0.5
    n_steps = 100

    # Initial random positions
    positions = np.random.uniform(2, area_size - 2, (n_robots, 2))

    # Grid for coverage measurement
    grid_res = 0.5
    grid_x = np.arange(0, area_size, grid_res)
    grid_y = np.arange(0, area_size, grid_res)
    grid_points = np.array([(x, y) for x in grid_x for y in grid_y])
    n_grid = len(grid_points)

    def compute_voronoi_centroids(positions, grid_points):
        """Compute Voronoi cell centroids for each robot using a grid."""
        n_r = len(positions)
        centroids = np.zeros((n_r, 2))
        cell_sizes = np.zeros(n_r)

        # Assign each grid point to nearest robot
        for gp in grid_points:
            dists = np.linalg.norm(positions - gp, axis=1)
            nearest = np.argmin(dists)
            centroids[nearest] += gp
            cell_sizes[nearest] += 1

        for i in range(n_r):
            if cell_sizes[i] > 0:
                centroids[i] /= cell_sizes[i]
            else:
                centroids[i] = positions[i]

        return centroids

    def compute_coverage(positions, grid_points, sensing_radius):
        """Compute percentage of grid points covered by at least one robot."""
        covered = 0
        for gp in grid_points:
            dists = np.linalg.norm(positions - gp, axis=1)
            if np.min(dists) <= sensing_radius:
                covered += 1
        return covered / len(grid_points) * 100

    # Initial coverage
    coverage_initial = compute_coverage(positions, grid_points, sensing_radius)

    # Lloyd's algorithm: move to Voronoi centroid
    coverage_history = [coverage_initial]
    Kp_coverage = 0.5

    for step in range(n_steps):
        centroids = compute_voronoi_centroids(positions, grid_points)

        # Move toward centroid
        for i in range(n_robots):
            direction = centroids[i] - positions[i]
            positions[i] += Kp_coverage * direction * dt

        # Keep in bounds
        positions = np.clip(positions, 0.5, area_size - 0.5)

        coverage = compute_coverage(positions, grid_points, sensing_radius)
        coverage_history.append(coverage)

    print("Voronoi-Based Multi-Robot Coverage")
    print(f"  {n_robots} robots, {area_size}x{area_size}m area, "
          f"sensing radius = {sensing_radius}m")
    print(f"\n  Coverage over time:")
    print(f"  {'Step':>5} | {'Coverage (%)':>12}")
    print("  " + "-" * 22)
    for step in [0, 10, 20, 50, 100]:
        if step < len(coverage_history):
            print(f"  {step:>5} | {coverage_history[step]:>12.2f}")

    print(f"\n  Initial coverage: {coverage_initial:.2f}%")
    print(f"  Final coverage:   {coverage_history[-1]:.2f}%")
    print(f"  Improvement:      {coverage_history[-1] - coverage_initial:.2f}%")

    # Final robot positions
    print(f"\n  Final robot positions:")
    for i in range(n_robots):
        print(f"    Robot {i}: ({positions[i, 0]:.2f}, {positions[i, 1]:.2f})")

    # Theoretical maximum coverage
    single_area = np.pi * sensing_radius**2
    total_sensing = n_robots * single_area
    max_possible = min(100.0, total_sensing / area_size**2 * 100)
    print(f"\n  Theoretical max coverage (no overlap): {max_possible:.1f}%")
    print(f"  Actual coverage: {coverage_history[-1]:.1f}%")
    print(f"  (Overlap between sensors reduces effective coverage below theoretical max)")


if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 16: Multi-Robot Systems — Exercise Solutions")
    print("=" * 70)

    print("\n--- Exercise 1: Auction-Based Allocation ---")
    exercise_1()

    print("\n--- Exercise 2: Leader-Follower Formation ---")
    exercise_2()

    print("\n--- Exercise 3: Reynolds Flocking ---")
    exercise_3()

    print("\n--- Exercise 4: Consensus Convergence ---")
    exercise_4()

    print("\n--- Exercise 5: Multi-Robot Coverage ---")
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
