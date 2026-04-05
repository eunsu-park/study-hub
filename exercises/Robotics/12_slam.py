"""
Exercises for Lesson 12: SLAM
Topic: Robotics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: EKF-SLAM Simulation
    10 landmarks, 20x20m environment, robot driving in a square path.
    """
    np.random.seed(42)

    # Environment
    env_size = 20.0
    n_landmarks = 10
    landmarks = np.random.uniform(2, env_size - 2, (n_landmarks, 2))
    sensor_range = 8.0
    sigma_r, sigma_b = 0.3, np.radians(5)
    R = np.diag([sigma_r**2, sigma_b**2])

    # Robot follows square path
    dt = 0.1
    v = 2.0   # m/s
    path_segments = [
        (v, 0.0, 40),        # right
        (v, np.pi/2, 5),     # turn left 90
        (v, 0.0, 40),        # up
        (v, np.pi/2, 5),     # turn
        (v, 0.0, 40),        # left
        (v, np.pi/2, 5),     # turn
        (v, 0.0, 40),        # down
        (v, np.pi/2, 5),     # turn
    ]

    # EKF-SLAM state: [x, y, theta, m1x, m1y, m2x, m2y, ...]
    state = np.array([2.0, 2.0, 0.0])  # robot state
    P = np.diag([0.01, 0.01, 0.001])
    landmark_map = {}  # id -> index in state

    sigma_v = 0.1
    sigma_w = 0.05

    # Simulate
    x_true = state[:3].copy()
    true_path = [x_true.copy()]
    est_path = [state[:3].copy()]
    landmark_first_seen = {}

    total_steps = 0
    for seg_v, seg_w_rate, seg_steps in path_segments:
        for _ in range(seg_steps):
            # True motion
            w = seg_w_rate
            x_true[0] += seg_v * np.cos(x_true[2]) * dt
            x_true[1] += seg_v * np.sin(x_true[2]) * dt
            x_true[2] += w * dt

            # Noisy control
            v_noisy = seg_v + np.random.normal(0, sigma_v)
            w_noisy = w + np.random.normal(0, sigma_w)

            # Predict
            theta = state[2]
            state[0] += v_noisy * np.cos(theta) * dt
            state[1] += v_noisy * np.sin(theta) * dt
            state[2] += w_noisy * dt

            n = len(state)
            F = np.eye(n)
            F[0, 2] = -v_noisy * np.sin(theta) * dt
            F[1, 2] = v_noisy * np.cos(theta) * dt

            Q = np.zeros((n, n))
            Q[0, 0] = (sigma_v * dt)**2
            Q[1, 1] = (sigma_v * dt)**2
            Q[2, 2] = (sigma_w * dt)**2
            P = F @ P @ F.T + Q

            # Observe landmarks
            for lm_id, lm_pos in enumerate(landmarks):
                dx = lm_pos[0] - x_true[0]
                dy = lm_pos[1] - x_true[1]
                r_true = np.sqrt(dx**2 + dy**2)
                if r_true > sensor_range:
                    continue

                r_meas = r_true + np.random.normal(0, sigma_r)
                b_meas = np.arctan2(dy, dx) - x_true[2] + np.random.normal(0, sigma_b)

                if lm_id not in landmark_map:
                    # Initialize new landmark
                    idx = len(state)
                    lx = state[0] + r_meas * np.cos(state[2] + b_meas)
                    ly = state[1] + r_meas * np.sin(state[2] + b_meas)
                    state = np.append(state, [lx, ly])
                    landmark_map[lm_id] = idx
                    landmark_first_seen[lm_id] = total_steps

                    # Augment P
                    n_old = len(P)
                    P_new = np.zeros((n_old + 2, n_old + 2))
                    P_new[:n_old, :n_old] = P
                    P_new[n_old, n_old] = 10.0
                    P_new[n_old+1, n_old+1] = 10.0
                    P = P_new
                else:
                    # Update existing landmark
                    idx = landmark_map[lm_id]
                    dx_e = state[idx] - state[0]
                    dy_e = state[idx + 1] - state[1]
                    r_exp = np.sqrt(dx_e**2 + dy_e**2)
                    b_exp = np.arctan2(dy_e, dx_e) - state[2]

                    n = len(state)
                    H = np.zeros((2, n))
                    H[0, 0] = -dx_e / r_exp
                    H[0, 1] = -dy_e / r_exp
                    H[0, idx] = dx_e / r_exp
                    H[0, idx+1] = dy_e / r_exp
                    H[1, 0] = dy_e / r_exp**2
                    H[1, 1] = -dx_e / r_exp**2
                    H[1, 2] = -1
                    H[1, idx] = -dy_e / r_exp**2
                    H[1, idx+1] = dx_e / r_exp**2

                    inn = np.array([r_meas - r_exp, b_meas - b_exp])
                    inn[1] = (inn[1] + np.pi) % (2*np.pi) - np.pi

                    S = H @ P @ H.T + R
                    K = P @ H.T @ np.linalg.inv(S)
                    state = state + K @ inn
                    P = (np.eye(n) - K @ H) @ P

            true_path.append(x_true.copy())
            est_path.append(state[:3].copy())
            total_steps += 1

    # Results
    print("EKF-SLAM Simulation")
    print(f"  Environment: {env_size}x{env_size}m, {n_landmarks} landmarks")
    print(f"  Sensor range: {sensor_range}m")
    print(f"  Total steps: {total_steps}")
    print(f"  Landmarks discovered: {len(landmark_map)}")

    # Landmark position errors and uncertainty
    print(f"\n  Landmark estimation results:")
    print(f"  {'ID':>4} | {'True pos':>16} | {'Est pos':>16} | {'Error (m)':>10} | {'Sigma (m)':>10}")
    print("  " + "-" * 65)
    for lm_id in sorted(landmark_map.keys()):
        idx = landmark_map[lm_id]
        true_pos = landmarks[lm_id]
        est_pos = state[idx:idx+2]
        error = np.linalg.norm(true_pos - est_pos)
        sigma = np.sqrt(max(P[idx, idx], P[idx+1, idx+1]))
        print(f"  {lm_id:>4} | ({true_pos[0]:6.2f},{true_pos[1]:6.2f}) | "
              f"({est_pos[0]:6.2f},{est_pos[1]:6.2f}) | {error:>10.4f} | {sigma:>10.4f}")

    final_pos_err = np.linalg.norm(true_path[-1][:2] - est_path[-1][:2])
    print(f"\n  Final robot position error: {final_pos_err:.4f} m")


def exercise_2():
    """
    Exercise 2: Pose Graph Optimization
    Robot drives in a square loop with odometry drift + loop closure.
    """
    np.random.seed(42)

    # Create square trajectory (true)
    n_poses = 20  # 5 per side
    poses_true = []
    side_length = 10.0
    poses_per_side = 5

    for side in range(4):
        angle = side * np.pi / 2
        for j in range(poses_per_side):
            t = j / poses_per_side
            x = side_length * t * np.cos(angle) + (side_length if side >= 2 else 0) * (-1 if side >= 2 else 1)
            y = side_length * t * np.sin(angle)
            if side == 0:
                x, y = t * side_length, 0
            elif side == 1:
                x, y = side_length, t * side_length
            elif side == 2:
                x, y = side_length - t * side_length, side_length
            elif side == 3:
                x, y = 0, side_length - t * side_length
            poses_true.append([x, y, angle])

    poses_true = np.array(poses_true[:n_poses])

    # Create noisy odometry
    sigma_odom = np.array([0.05, 0.05, np.radians(2)])
    poses_odom = [poses_true[0].copy()]

    edges = []
    info_odom = np.diag([100, 100, 400])  # information matrix for odometry
    info_loop = np.diag([50, 50, 200])    # weaker for loop closure

    for i in range(1, n_poses):
        # True relative transform
        dx = poses_true[i, 0] - poses_true[i-1, 0]
        dy = poses_true[i, 1] - poses_true[i-1, 1]
        dtheta = poses_true[i, 2] - poses_true[i-1, 2]

        # Add noise
        z_ij = np.array([dx, dy, dtheta]) + np.random.normal(0, sigma_odom)
        edges.append((i-1, i, z_ij, info_odom))

        # Propagate noisy odometry
        prev = poses_odom[-1]
        new_pose = prev + z_ij
        poses_odom.append(new_pose)

    poses_odom = np.array(poses_odom)

    # Error before optimization (drift)
    drift = np.linalg.norm(poses_odom[-1, :2] - poses_odom[0, :2])
    print("Pose Graph Optimization")
    print(f"  {n_poses} poses in a square loop")
    print(f"  Drift without loop closure: {drift:.4f} m")
    print(f"  (Last pose should be near first pose but is displaced by odometry drift)")

    # Add loop closure edge (last to first)
    # True: relative is close to zero (same position)
    z_loop = np.array([0.0, 0.0, 0.0]) + np.random.normal(0, sigma_odom * 0.5)
    edges.append((n_poses - 1, 0, z_loop, info_loop))

    # Simple Gauss-Newton optimization
    x = poses_odom.flatten()

    for iteration in range(20):
        dim = 3 * n_poses
        H = np.zeros((dim, dim))
        b = np.zeros(dim)
        total_error = 0

        for i, j, z_ij, omega in edges:
            xi = x[3*i:3*i+3]
            xj = x[3*j:3*j+3]
            dx = xj[0] - xi[0]
            dy = xj[1] - xi[1]
            c = np.cos(xi[2])
            s = np.sin(xi[2])
            e = np.array([
                c*dx + s*dy - z_ij[0],
                -s*dx + c*dy - z_ij[1],
                (xj[2] - xi[2] - z_ij[2] + np.pi) % (2*np.pi) - np.pi
            ])
            total_error += e.T @ omega @ e

            # Jacobians
            Ai = np.array([
                [-c, -s, -s*dx + c*dy],
                [s, -c, -c*dx - s*dy],
                [0, 0, -1]
            ])
            Bi = np.array([
                [c, s, 0],
                [-s, c, 0],
                [0, 0, 1]
            ])

            H[3*i:3*i+3, 3*i:3*i+3] += Ai.T @ omega @ Ai
            H[3*i:3*i+3, 3*j:3*j+3] += Ai.T @ omega @ Bi
            H[3*j:3*j+3, 3*i:3*i+3] += Bi.T @ omega @ Ai
            H[3*j:3*j+3, 3*j:3*j+3] += Bi.T @ omega @ Bi
            b[3*i:3*i+3] += Ai.T @ omega @ e
            b[3*j:3*j+3] += Bi.T @ omega @ e

        # Fix first pose (gauge freedom)
        H[:3, :] = 0
        H[:, :3] = 0
        H[0, 0] = H[1, 1] = H[2, 2] = 1e6
        b[:3] = 0

        try:
            delta = np.linalg.solve(H, -b)
            x += delta
            if np.linalg.norm(delta) < 1e-6:
                break
        except np.linalg.LinAlgError:
            break

    poses_opt = x.reshape(-1, 3)

    # Results
    print(f"\n  After optimization ({iteration + 1} iterations):")
    drift_opt = np.linalg.norm(poses_opt[-1, :2] - poses_opt[0, :2])
    print(f"  Drift after optimization: {drift_opt:.4f} m (was {drift:.4f} m)")

    # Per-pose error
    errors_before = np.sqrt(np.sum((poses_odom[:, :2] - poses_true[:, :2])**2, axis=1))
    errors_after = np.sqrt(np.sum((poses_opt[:, :2] - poses_true[:, :2])**2, axis=1))
    print(f"  Mean position error before: {np.mean(errors_before):.4f} m")
    print(f"  Mean position error after:  {np.mean(errors_after):.4f} m")
    print(f"\n  The loop closure redistributes accumulated drift across the trajectory.")


def exercise_3():
    """
    Exercise 3: Data Association Challenge
    Nearest-neighbor vs Mahalanobis gating with close landmarks.
    """
    np.random.seed(42)
    sigma_r, sigma_b = 0.5, np.radians(5)

    # Two close landmarks
    lm1 = np.array([5.0, 5.0])
    lm2 = np.array([5.3, 5.1])  # only 0.32m apart

    # Robot at origin
    robot = np.array([0.0, 0.0, 0.0])

    # Generate observations
    n_obs = 20
    correct_nn = 0
    correct_maha = 0

    # Simplified covariance for Mahalanobis
    P_lm = np.diag([0.3**2, 0.3**2])  # landmark position uncertainty

    for _ in range(n_obs):
        # Observe landmark 1
        dx = lm1[0] - robot[0]
        dy = lm1[1] - robot[1]
        r_true = np.sqrt(dx**2 + dy**2)
        b_true = np.arctan2(dy, dx) - robot[2]

        r_obs = r_true + np.random.normal(0, sigma_r)
        b_obs = b_true + np.random.normal(0, sigma_b)

        # Observed position
        obs_x = robot[0] + r_obs * np.cos(robot[2] + b_obs)
        obs_y = robot[1] + r_obs * np.sin(robot[2] + b_obs)
        obs_pos = np.array([obs_x, obs_y])

        # Nearest neighbor: closest landmark
        d1 = np.linalg.norm(obs_pos - lm1)
        d2 = np.linalg.norm(obs_pos - lm2)
        nn_match = 0 if d1 <= d2 else 1
        if nn_match == 0:
            correct_nn += 1

        # Mahalanobis distance
        R_obs = np.diag([sigma_r**2 * np.cos(b_obs)**2 + (r_obs * sigma_b * np.sin(b_obs))**2,
                          sigma_r**2 * np.sin(b_obs)**2 + (r_obs * sigma_b * np.cos(b_obs))**2])
        S1 = P_lm + R_obs
        S2 = P_lm + R_obs
        inn1 = obs_pos - lm1
        inn2 = obs_pos - lm2
        d_maha1 = np.sqrt(inn1 @ np.linalg.inv(S1) @ inn1)
        d_maha2 = np.sqrt(inn2 @ np.linalg.inv(S2) @ inn2)
        maha_match = 0 if d_maha1 <= d_maha2 else 1
        if maha_match == 0:
            correct_maha += 1

    print("Data Association: Nearest-Neighbor vs Mahalanobis")
    print(f"  Landmark 1: {lm1}, Landmark 2: {lm2}")
    print(f"  Distance between landmarks: {np.linalg.norm(lm1 - lm2):.3f} m")
    print(f"  Range noise: sigma_r = {sigma_r} m")
    print(f"\n  Over {n_obs} observations of landmark 1:")
    print(f"    Nearest neighbor correct: {correct_nn}/{n_obs} ({100*correct_nn/n_obs:.0f}%)")
    print(f"    Mahalanobis correct:      {correct_maha}/{n_obs} ({100*correct_maha/n_obs:.0f}%)")
    print(f"\n  When landmarks are close together relative to sensor noise,")
    print(f"  nearest-neighbor matching can misassociate observations.")
    print(f"  Mahalanobis distance accounts for uncertainty, improving robustness.")


def exercise_4():
    """
    Exercise 4: Visual Feature Matching
    Simulated ORB-like feature matching between two views.
    """
    np.random.seed(42)

    # Simulate 3D points
    n_points = 30
    points_3d = np.random.uniform(-2, 2, (n_points, 3))
    points_3d[:, 2] += 5  # shift forward

    # Camera intrinsics
    fx, fy = 500, 500
    cx, cy = 320, 240
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Camera 1: at origin
    R1 = np.eye(3)
    t1 = np.zeros(3)

    # Camera 2: small translation and rotation
    angle = np.radians(5)
    R2 = np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])
    t2 = np.array([0.3, 0.0, 0.0])

    # Project points to both cameras
    def project(P, R, t, K):
        p_cam = R @ P + t
        if p_cam[2] <= 0:
            return None
        px = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
        py = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
        return np.array([px, py])

    pts1, pts2, pts3d_matched = [], [], []
    for p in points_3d:
        p1 = project(p, R1, t1, K)
        p2 = project(p, R2, t2, K)
        if p1 is not None and p2 is not None:
            if 0 <= p1[0] < 640 and 0 <= p1[1] < 480:
                if 0 <= p2[0] < 640 and 0 <= p2[1] < 480:
                    # Add noise
                    pts1.append(p1 + np.random.normal(0, 0.5, 2))
                    pts2.append(p2 + np.random.normal(0, 0.5, 2))
                    pts3d_matched.append(p)

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    pts3d_matched = np.array(pts3d_matched)

    print("Visual Feature Matching Simulation")
    print(f"  {len(pts1)} matched features between two views")
    print(f"  Camera translation: {t2}")
    print(f"  Camera rotation: {np.degrees(angle):.1f}° about y-axis")

    # Triangulate points
    triangulated = []
    for i in range(len(pts1)):
        # Simple triangulation using DLT
        u1, v1 = pts1[i]
        u2, v2 = pts2[i]

        P1 = K @ np.hstack([R1, t1.reshape(-1, 1)])
        P2 = K @ np.hstack([R2, t2.reshape(-1, 1)])

        A = np.array([
            u1 * P1[2] - P1[0],
            v1 * P1[2] - P1[1],
            u2 * P2[2] - P2[0],
            v2 * P2[2] - P2[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X[:3] / X[3]
        triangulated.append(X)

    triangulated = np.array(triangulated)
    errors = np.linalg.norm(triangulated - pts3d_matched, axis=1)

    print(f"\n  Triangulation results:")
    print(f"    Mean 3D error: {np.mean(errors):.4f} m")
    print(f"    Max 3D error:  {np.max(errors):.4f} m")
    print(f"    Median error:  {np.median(errors):.4f} m")


def exercise_5():
    """
    Exercise 5: Loop Closure Impact on Pose Graph SLAM.
    Long corridor (50 poses) with return trip.
    """
    np.random.seed(42)
    n_poses = 50
    step_size = 1.0
    sigma_odom = np.array([0.02, 0.02, np.radians(1)])

    # True trajectory: straight corridor and back
    poses_true = np.zeros((n_poses, 3))
    for i in range(25):
        poses_true[i] = [i * step_size, 0, 0]
    for i in range(25):
        poses_true[25 + i] = [(24 - i) * step_size, 0.5, np.pi]

    # Noisy odometry
    poses_noisy = [poses_true[0].copy()]
    edges = []
    info_odom = np.diag([1.0 / sigma_odom[0]**2,
                          1.0 / sigma_odom[1]**2,
                          1.0 / sigma_odom[2]**2])

    for i in range(1, n_poses):
        dpose = poses_true[i] - poses_true[i-1]
        dpose_noisy = dpose + np.random.normal(0, sigma_odom)
        edges.append((i-1, i, dpose_noisy, info_odom))
        poses_noisy.append(poses_noisy[-1] + dpose_noisy)

    poses_noisy = np.array(poses_noisy)

    # Without loop closure
    err_no_loop = np.linalg.norm(poses_noisy[-1, :2] - poses_true[-1, :2])
    mean_err_no_loop = np.mean(np.sqrt(np.sum((poses_noisy[:, :2] - poses_true[:, :2])**2, axis=1)))

    # With loop closure (connect last to first)
    sigma_loop = sigma_odom * 0.5
    info_loop = np.diag([1.0/sigma_loop[0]**2, 1.0/sigma_loop[1]**2, 1.0/sigma_loop[2]**2])
    z_loop = poses_true[0] - poses_true[-1] + np.random.normal(0, sigma_loop * 0.1)
    edges_with_loop = edges + [(n_poses - 1, 0, z_loop, info_loop)]

    # Optimize
    def optimize_graph(edges_list, init_poses, n_iter=30):
        x = init_poses.flatten().copy()
        n = len(init_poses)

        for iteration in range(n_iter):
            dim = 3 * n
            H = np.zeros((dim, dim))
            b = np.zeros(dim)

            for i, j, z_ij, omega in edges_list:
                xi = x[3*i:3*i+3]
                xj = x[3*j:3*j+3]
                dx_v = xj[0] - xi[0]
                dy_v = xj[1] - xi[1]
                c = np.cos(xi[2])
                s = np.sin(xi[2])
                e = np.array([
                    c*dx_v + s*dy_v - z_ij[0],
                    -s*dx_v + c*dy_v - z_ij[1],
                    (xj[2] - xi[2] - z_ij[2] + np.pi) % (2*np.pi) - np.pi
                ])

                Ai = np.array([[-c, -s, -s*dx_v + c*dy_v],
                                [s, -c, -c*dx_v - s*dy_v],
                                [0, 0, -1]])
                Bi = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

                H[3*i:3*i+3, 3*i:3*i+3] += Ai.T @ omega @ Ai
                H[3*i:3*i+3, 3*j:3*j+3] += Ai.T @ omega @ Bi
                H[3*j:3*j+3, 3*i:3*i+3] += Bi.T @ omega @ Ai
                H[3*j:3*j+3, 3*j:3*j+3] += Bi.T @ omega @ Bi
                b[3*i:3*i+3] += Ai.T @ omega @ e
                b[3*j:3*j+3] += Bi.T @ omega @ e

            # Fix first pose
            H[:3, :] = 0; H[:, :3] = 0
            H[0,0] = H[1,1] = H[2,2] = 1e6
            b[:3] = 0

            try:
                delta = np.linalg.solve(H, -b)
                x += delta
                if np.linalg.norm(delta) < 1e-8:
                    break
            except np.linalg.LinAlgError:
                break

        return x.reshape(-1, 3)

    poses_opt = optimize_graph(edges_with_loop, poses_noisy)
    err_with_loop = np.linalg.norm(poses_opt[-1, :2] - poses_true[-1, :2])
    mean_err_with_loop = np.mean(np.sqrt(np.sum((poses_opt[:, :2] - poses_true[:, :2])**2, axis=1)))

    print("Loop Closure Impact on Pose Graph SLAM")
    print(f"  {n_poses} poses, corridor trajectory")
    print(f"\n  {'Metric':>30} | {'No loop':>10} | {'With loop':>10}")
    print("  " + "-" * 55)
    print(f"  {'Final position error (m)':>30} | {err_no_loop:>10.4f} | {err_with_loop:>10.4f}")
    print(f"  {'Mean trajectory error (m)':>30} | {mean_err_no_loop:>10.4f} | {mean_err_with_loop:>10.4f}")
    improvement = (1 - mean_err_with_loop / mean_err_no_loop) * 100
    print(f"\n  Improvement: {improvement:.1f}% reduction in mean trajectory error")
    print(f"  Loop closure distributes accumulated drift across entire trajectory.")


if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 12: SLAM — Exercise Solutions")
    print("=" * 70)

    print("\n--- Exercise 1: EKF-SLAM Simulation ---")
    exercise_1()

    print("\n--- Exercise 2: Pose Graph Optimization ---")
    exercise_2()

    print("\n--- Exercise 3: Data Association Challenge ---")
    exercise_3()

    print("\n--- Exercise 4: Visual Feature Matching ---")
    exercise_4()

    print("\n--- Exercise 5: Loop Closure Impact ---")
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
