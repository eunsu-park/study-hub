"""
Exercise Solutions for Lesson 23: SLAM Introduction
Computer Vision - Visual Odometry, Loop Closure, ICP, Occupancy Grid, SLAM

Topics covered:
- Visual Odometry (monocular VO pipeline)
- Loop closure detection (BoW-based)
- ICP (Iterative Closest Point) algorithm
- Occupancy grid mapping from LiDAR scans
- Complete SLAM system integration
"""

import numpy as np


# =============================================================================
# Helper: Feature extraction and matching
# =============================================================================

def extract_features(img, n_features=50):
    """
    Simple corner-based feature extraction.
    Returns list of (x, y) keypoints and patch descriptors.
    """
    h, w = img.shape
    img_f = img.astype(np.float64)

    # Compute corner response
    Ix = np.zeros_like(img_f)
    Iy = np.zeros_like(img_f)
    Ix[:, 1:-1] = (img_f[:, 2:] - img_f[:, :-2]) / 2
    Iy[1:-1, :] = (img_f[2:, :] - img_f[:-2, :]) / 2

    Ixx, Iyy, Ixy = Ix * Ix, Iy * Iy, Ix * Iy

    response = np.zeros_like(img_f)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            sxx = np.sum(Ixx[i-1:i+2, j-1:j+2])
            syy = np.sum(Iyy[i-1:i+2, j-1:j+2])
            sxy = np.sum(Ixy[i-1:i+2, j-1:j+2])
            det = sxx * syy - sxy**2
            trace = sxx + syy
            response[i, j] = det - 0.04 * trace**2

    # Extract keypoints (local maxima)
    thresh = max(response.max() * 0.01, 1e-6)
    kps = []
    for i in range(2, h - 2):
        for j in range(2, w - 2):
            if response[i, j] > thresh:
                local = response[i-1:i+2, j-1:j+2]
                if response[i, j] == local.max():
                    kps.append((j, i, response[i, j]))

    kps.sort(key=lambda x: x[2], reverse=True)
    kps = kps[:n_features]

    # Extract patch descriptors
    keypoints = []
    descriptors = []
    patch = 4
    for x, y, _ in kps:
        if patch <= y < h - patch and patch <= x < w - patch:
            desc = img_f[y-patch:y+patch, x-patch:x+patch].flatten()
            std = desc.std()
            if std > 1:
                desc = (desc - desc.mean()) / std
            descriptors.append(desc)
            keypoints.append((x, y))

    return keypoints, np.array(descriptors) if descriptors else np.array([])


def match_features(desc1, desc2, ratio_thresh=0.8):
    """Match features using ratio test. Returns list of (idx1, idx2) pairs."""
    if len(desc1) == 0 or len(desc2) == 0:
        return []

    matches = []
    for i in range(len(desc1)):
        dists = np.sqrt(np.sum((desc2 - desc1[i])**2, axis=1))
        sorted_idx = np.argsort(dists)
        if len(sorted_idx) >= 2:
            if dists[sorted_idx[1]] > 0:
                ratio = dists[sorted_idx[0]] / dists[sorted_idx[1]]
                if ratio < ratio_thresh:
                    matches.append((i, sorted_idx[0]))

    return matches


# =============================================================================
# Exercise 1: Visual Odometry
# =============================================================================

def exercise_1_visual_odometry():
    """
    Implement monocular Visual Odometry.

    Pipeline:
    1. Detect features in consecutive frames
    2. Match features between frames
    3. Estimate Essential Matrix
    4. Recover pose (R, t)
    5. Accumulate trajectory

    Returns:
        list of (R, t) poses and trajectory
    """
    np.random.seed(42)

    # Camera intrinsics
    f = 300.0
    K = np.array([[f, 0, 80], [0, f, 60], [0, 0, 1]], dtype=np.float64)

    # Generate synthetic frames with known camera motion
    h, w = 120, 160
    n_frames = 8

    # Ground truth trajectory: move forward with slight turns
    gt_poses = []
    cur_R = np.eye(3)
    cur_t = np.zeros(3)
    gt_poses.append((cur_R.copy(), cur_t.copy()))

    for i in range(1, n_frames):
        # Small rotation + forward motion
        angle = np.radians(5 * np.sin(i * 0.5))
        dR = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        dt = np.array([0.2 * np.sin(i * 0.3), 0.0, 1.0])

        cur_R = dR @ cur_R
        cur_t = cur_t + cur_R @ dt
        gt_poses.append((cur_R.copy(), cur_t.copy()))

    # Generate frames (textured pattern that shifts)
    frames = []
    for i in range(n_frames):
        frame = np.random.randint(50, 80, (h, w), dtype=np.uint8)
        # Add features (bright spots at known 3D positions projected)
        for fi in range(20):
            fx = int(40 + 80 * np.sin(fi * 0.7) - i * 5) % w
            fy = int(30 + 60 * np.cos(fi * 0.5)) % h
            size = 3
            y1, y2 = max(0, fy-size), min(h, fy+size)
            x1, x2 = max(0, fx-size), min(w, fx+size)
            frame[y1:y2, x1:x2] = np.random.randint(160, 220)
        frames.append(frame)

    print("Visual Odometry")
    print(f"  Frames: {n_frames}")
    print(f"  Image size: {w}x{h}")
    print(f"  Focal length: {f:.0f}")
    print("=" * 60)

    # Process consecutive frame pairs
    estimated_trajectory = [(np.eye(3), np.zeros(3))]

    for i in range(1, n_frames):
        kps1, desc1 = extract_features(frames[i-1], n_features=50)
        kps2, desc2 = extract_features(frames[i], n_features=50)

        matches = match_features(desc1, desc2)

        if len(matches) < 5:
            print(f"  Frame {i}: insufficient matches ({len(matches)})")
            estimated_trajectory.append(estimated_trajectory[-1])
            continue

        # Get matched points
        pts1 = np.array([kps1[m[0]] for m in matches], dtype=np.float64)
        pts2 = np.array([kps2[m[1]] for m in matches], dtype=np.float64)

        # Normalize
        pts1_n = np.column_stack([
            (pts1[:, 0] - K[0, 2]) / K[0, 0],
            (pts1[:, 1] - K[1, 2]) / K[1, 1]
        ])
        pts2_n = np.column_stack([
            (pts2[:, 0] - K[0, 2]) / K[0, 0],
            (pts2[:, 1] - K[1, 2]) / K[1, 1]
        ])

        # Essential matrix (8-point)
        n_match = len(matches)
        A = np.zeros((n_match, 9))
        for j in range(n_match):
            x1, y1 = pts1_n[j]
            x2, y2 = pts2_n[j]
            A[j] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

        _, _, Vt = np.linalg.svd(A)
        E = Vt[-1].reshape(3, 3)

        # Enforce rank-2
        U, S, Vt2 = np.linalg.svd(E)
        S_new = np.array([(S[0]+S[1])/2, (S[0]+S[1])/2, 0])
        E = U @ np.diag(S_new) @ Vt2

        # Recover pose (simplified: use first valid decomposition)
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        R_est = U @ W @ Vt2
        if np.linalg.det(R_est) < 0:
            R_est = -R_est
        t_est = U[:, 2]

        # Accumulate
        prev_R, prev_t = estimated_trajectory[-1]
        new_t = prev_t + prev_R @ t_est
        new_R = R_est @ prev_R
        estimated_trajectory.append((new_R, new_t))

        # Compare with ground truth
        gt_t = gt_poses[i][1]
        error = np.linalg.norm(new_t - gt_t)
        print(f"  Frame {i}: matches={len(matches)}, "
              f"est_pos=({new_t[0]:.2f},{new_t[1]:.2f},{new_t[2]:.2f}), "
              f"error={error:.2f}")

    # Final trajectory comparison
    print(f"\n  Trajectory Summary:")
    total_gt_dist = np.linalg.norm(gt_poses[-1][1])
    total_est_dist = np.linalg.norm(estimated_trajectory[-1][1])
    print(f"    GT total distance:  {total_gt_dist:.2f}")
    print(f"    Est total distance: {total_est_dist:.2f}")

    return estimated_trajectory


# =============================================================================
# Exercise 2: Loop Closure Detection
# =============================================================================

def exercise_2_loop_closure():
    """
    Detect loop closures using Bag-of-Words (BoW) similarity.

    Steps:
    1. Build visual vocabulary (cluster descriptors)
    2. Compute BoW vectors for each frame
    3. Detect candidates via cosine similarity
    4. Geometric verification (inlier count)

    Returns:
        list of detected loop closures
    """
    np.random.seed(42)

    # Generate frames: a trajectory that revisits earlier positions
    h, w = 80, 100
    n_frames = 20

    # Frames 0-7: forward, Frames 8-15: new area, Frames 16-19: back to start
    frames = []
    for i in range(n_frames):
        base = np.random.randint(50, 80, (h, w), dtype=np.uint8)

        # Add consistent features based on "location"
        if i < 8 or i >= 16:
            # Similar features for start area
            loc_seed = i if i < 8 else (i - 16)
            np.random.seed(100 + loc_seed % 8)
        else:
            np.random.seed(200 + i)

        for fi in range(15):
            fx = np.random.randint(5, w - 5)
            fy = np.random.randint(5, h - 5)
            base[fy-2:fy+2, fx-2:fx+2] = np.random.randint(150, 220)

        np.random.seed(42 + i)  # Reset
        frames.append(base)

    print("Loop Closure Detection")
    print(f"  Frames: {n_frames}")
    print(f"  Expected loops: frames 16-19 should match frames 0-3")
    print("=" * 60)

    # Step 1: Extract features and build vocabulary
    all_descriptors = []
    frame_features = []

    for i, frame in enumerate(frames):
        kps, descs = extract_features(frame, n_features=30)
        frame_features.append((kps, descs))
        if len(descs) > 0:
            all_descriptors.append(descs)

    # Simple vocabulary: cluster all descriptors (k-means with k=20)
    if all_descriptors:
        all_desc = np.vstack(all_descriptors)
        k = min(20, len(all_desc) // 2)

        # K-means clustering
        np.random.seed(42)
        centroids = all_desc[np.random.choice(len(all_desc), k, replace=False)]

        for _ in range(10):
            labels = np.argmin(
                np.array([np.sum((all_desc - c)**2, axis=1) for c in centroids]).T,
                axis=1
            )
            for c in range(k):
                mask = labels == c
                if np.sum(mask) > 0:
                    centroids[c] = np.mean(all_desc[mask], axis=0)

        print(f"\n  Vocabulary: {k} words from {len(all_desc)} descriptors")

    # Step 2: Compute BoW vectors
    bow_vectors = []
    for i, (kps, descs) in enumerate(frame_features):
        bow = np.zeros(k)
        if len(descs) > 0:
            for desc in descs:
                dists = np.sum((centroids - desc)**2, axis=1)
                word = np.argmin(dists)
                bow[word] += 1
            # L2 normalize
            norm = np.linalg.norm(bow)
            if norm > 0:
                bow /= norm
        bow_vectors.append(bow)

    # Step 3: Detect loop closures
    min_frame_gap = 10  # Minimum frame gap to consider as loop
    similarity_threshold = 0.3

    loop_closures = []
    print(f"\n  Similarity Matrix (selected pairs):")

    for i in range(min_frame_gap, n_frames):
        for j in range(0, i - min_frame_gap):
            sim = np.dot(bow_vectors[i], bow_vectors[j])

            if sim > similarity_threshold:
                # Step 4: Geometric verification
                kps_i, desc_i = frame_features[i]
                kps_j, desc_j = frame_features[j]
                matches = match_features(desc_i, desc_j, ratio_thresh=0.85)

                if len(matches) >= 5:
                    loop_closures.append({
                        'frame_i': i,
                        'frame_j': j,
                        'similarity': sim,
                        'matches': len(matches),
                    })
                    print(f"    Frame {i} <-> Frame {j}: "
                          f"sim={sim:.3f}, matches={len(matches)} "
                          f"** LOOP DETECTED **")

    print(f"\n  Loop closures detected: {len(loop_closures)}")
    for lc in loop_closures:
        print(f"    {lc['frame_i']} <-> {lc['frame_j']}: "
              f"sim={lc['similarity']:.3f}")

    return loop_closures


# =============================================================================
# Exercise 3: ICP (Iterative Closest Point)
# =============================================================================

def exercise_3_icp():
    """
    Implement the ICP algorithm for point cloud alignment.

    Steps:
    1. Find nearest correspondences
    2. Compute optimal R, t using SVD
    3. Apply transformation
    4. Iterate until convergence

    Returns:
        (R_final, t_final, aligned_points, error_history)
    """
    np.random.seed(42)

    # Source point cloud (2D for simplicity)
    n_points = 50
    source = np.column_stack([
        np.random.uniform(-3, 3, n_points),
        np.random.uniform(-3, 3, n_points),
    ])

    # Apply known transformation to create target
    true_angle = np.radians(15)
    true_t = np.array([1.5, 0.8])
    R_true = np.array([
        [np.cos(true_angle), -np.sin(true_angle)],
        [np.sin(true_angle), np.cos(true_angle)]
    ])

    target = (R_true @ source.T).T + true_t
    # Add noise to target
    target += np.random.randn(n_points, 2) * 0.05

    print("ICP (Iterative Closest Point)")
    print(f"  Points: {n_points}")
    print(f"  True rotation: {np.degrees(true_angle):.1f} deg")
    print(f"  True translation: ({true_t[0]:.2f}, {true_t[1]:.2f})")
    print("=" * 60)

    # ICP algorithm
    max_iterations = 50
    tolerance = 1e-6
    current = source.copy()
    R_total = np.eye(2)
    t_total = np.zeros(2)
    error_history = []

    print(f"\n  {'Iter':>5} | {'Error':>10} | {'dError':>10}")
    print(f"  {'-'*35}")

    for iteration in range(max_iterations):
        # Step 1: Find nearest correspondences
        correspondences = np.zeros(n_points, dtype=int)
        for i in range(n_points):
            dists = np.sum((target - current[i])**2, axis=1)
            correspondences[i] = np.argmin(dists)

        # Step 2: Compute centroids
        matched_target = target[correspondences]
        src_centroid = np.mean(current, axis=0)
        tgt_centroid = np.mean(matched_target, axis=0)

        # Step 3: Compute R, t using SVD
        src_centered = current - src_centroid
        tgt_centered = matched_target - tgt_centroid

        H = src_centered.T @ tgt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T

        t = tgt_centroid - R @ src_centroid

        # Step 4: Apply transformation
        current = (R @ current.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t

        # Compute error
        error = np.mean(np.sqrt(np.sum(
            (current - target[correspondences])**2, axis=1)))
        error_history.append(error)

        if iteration < 5 or iteration % 10 == 0 or error < tolerance:
            d_error = (error_history[-2] - error
                       if len(error_history) > 1 else 0)
            print(f"  {iteration:>5} | {error:>10.6f} | {d_error:>10.6f}")

        # Check convergence
        if len(error_history) > 1:
            if abs(error_history[-1] - error_history[-2]) < tolerance:
                print(f"\n  Converged at iteration {iteration}")
                break

    # Recover estimated angle
    est_angle = np.arctan2(R_total[1, 0], R_total[0, 0])

    print(f"\n  Results:")
    print(f"    Estimated rotation: {np.degrees(est_angle):.2f} deg "
          f"(true: {np.degrees(true_angle):.1f})")
    print(f"    Estimated translation: ({t_total[0]:.3f}, {t_total[1]:.3f}) "
          f"(true: ({true_t[0]:.2f}, {true_t[1]:.2f}))")
    print(f"    Final error: {error_history[-1]:.6f}")
    print(f"    Iterations: {len(error_history)}")

    return R_total, t_total, current, error_history


# =============================================================================
# Exercise 4: Occupancy Grid Map
# =============================================================================

def exercise_4_occupancy_grid():
    """
    Create an occupancy grid map from simulated LiDAR scans.

    Uses log-odds representation for probabilistic updates.

    Parameters:
        - Grid resolution: 0.1m per cell
        - Log-odds: occupied +0.5, free -0.2

    Returns:
        occupancy grid (probability map)
    """
    np.random.seed(42)

    # Map parameters
    grid_size = 100  # 100x100 cells
    resolution = 0.1  # meters per cell
    map_size = grid_size * resolution  # 10m x 10m

    # Initialize log-odds map (0 = unknown = 0.5 probability)
    log_odds = np.zeros((grid_size, grid_size), dtype=np.float64)

    # Log-odds update values
    l_occ = 0.5   # Log-odds for occupied
    l_free = -0.2  # Log-odds for free

    # Define environment: walls and obstacles
    # Wall segments (in meters): (x1, y1, x2, y2)
    walls = [
        (1, 1, 9, 1),   # Bottom wall
        (1, 9, 9, 9),   # Top wall
        (1, 1, 1, 9),   # Left wall
        (9, 1, 9, 9),   # Right wall
        (4, 3, 4, 7),   # Internal wall
        (6, 2, 8, 2),   # Obstacle
    ]

    def ray_cast(robot_pos, angle, max_range=8.0, walls=walls):
        """Cast a ray and find intersection with walls."""
        x0, y0 = robot_pos
        dx = np.cos(angle)
        dy = np.sin(angle)

        min_t = max_range
        for wx1, wy1, wx2, wy2 in walls:
            # Line-segment intersection
            # Parametric: ray = (x0 + t*dx, y0 + t*dy)
            # Wall: (wx1 + s*(wx2-wx1), wy1 + s*(wy2-wy1))
            dwx = wx2 - wx1
            dwy = wy2 - wy1

            denom = dx * dwy - dy * dwx
            if abs(denom) < 1e-10:
                continue

            t = ((wx1 - x0) * dwy - (wy1 - y0) * dwx) / denom
            s = ((wx1 - x0) * dy - (wy1 - y0) * dx) / denom

            if 0 <= s <= 1 and 0 < t < min_t:
                min_t = t

        return min_t

    def update_grid(log_odds, robot_pos, scan_ranges, scan_angles, resolution):
        """Update occupancy grid with a single scan."""
        rx = int(robot_pos[0] / resolution)
        ry = int(robot_pos[1] / resolution)

        for angle, rng in zip(scan_angles, scan_ranges):
            # Cells along the ray
            n_steps = int(rng / resolution)

            for step in range(n_steps):
                t = step * resolution
                x = robot_pos[0] + t * np.cos(angle)
                y = robot_pos[1] + t * np.sin(angle)
                gx = int(x / resolution)
                gy = int(y / resolution)

                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    log_odds[gy, gx] += l_free  # Free space

            # Endpoint = occupied
            ex = robot_pos[0] + rng * np.cos(angle)
            ey = robot_pos[1] + rng * np.sin(angle)
            gex = int(ex / resolution)
            gey = int(ey / resolution)

            if 0 <= gex < grid_size and 0 <= gey < grid_size:
                log_odds[gey, gex] += l_occ  # Occupied

        return log_odds

    # Simulate robot trajectory and scans
    robot_positions = [
        (3, 3), (3, 5), (3, 7),   # Left side
        (5, 5),                     # Center
        (7, 3), (7, 5), (7, 7),   # Right side
    ]

    n_rays = 36  # 10-degree resolution
    scan_angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

    print("Occupancy Grid Mapping")
    print(f"  Grid: {grid_size}x{grid_size}, resolution={resolution}m")
    print(f"  Map size: {map_size}x{map_size}m")
    print(f"  Robot poses: {len(robot_positions)}")
    print(f"  Rays per scan: {n_rays}")
    print("=" * 60)

    for pos_idx, robot_pos in enumerate(robot_positions):
        # Perform scan
        ranges = []
        for angle in scan_angles:
            r = ray_cast(robot_pos, angle)
            ranges.append(r)

        # Update grid
        log_odds = update_grid(log_odds, robot_pos, ranges, scan_angles,
                               resolution)

        avg_range = np.mean(ranges)
        print(f"  Pose {pos_idx}: ({robot_pos[0]:.1f}, {robot_pos[1]:.1f}), "
              f"avg_range={avg_range:.2f}m")

    # Convert log-odds to probability
    prob_map = 1.0 / (1.0 + np.exp(-log_odds))

    # Statistics
    occupied = np.sum(prob_map > 0.7)
    free = np.sum(prob_map < 0.3)
    unknown = np.sum((prob_map >= 0.3) & (prob_map <= 0.7))

    print(f"\n  Map Statistics:")
    print(f"    Occupied cells (>0.7): {occupied} ({100*occupied/grid_size**2:.1f}%)")
    print(f"    Free cells (<0.3):     {free} ({100*free/grid_size**2:.1f}%)")
    print(f"    Unknown cells:         {unknown} ({100*unknown/grid_size**2:.1f}%)")
    print(f"    Probability range: [{prob_map.min():.4f}, {prob_map.max():.4f}]")

    # Visualize as ASCII art (downsampled)
    print(f"\n  Map Visualization (downsampled to 20x20):")
    ds = grid_size // 20
    for i in range(0, grid_size, ds):
        row = ""
        for j in range(0, grid_size, ds):
            p = prob_map[i, j]
            if p > 0.7:
                row += "##"
            elif p < 0.3:
                row += "  "
            else:
                row += ".."
        print(f"    {row}")

    return prob_map


# =============================================================================
# Exercise 5: Complete SLAM System
# =============================================================================

def exercise_5_complete_slam():
    """
    Integrate Visual Odometry, loop closure, and mapping into a SLAM system.

    Components:
    1. Keyframe management
    2. Visual odometry for pose estimation
    3. Loop closure detection
    4. Pose graph with simple optimization
    5. 3D map generation

    Returns:
        SLAM results dict
    """
    np.random.seed(42)

    class SimpleSLAM:
        def __init__(self):
            self.keyframes = []       # (frame_idx, pose, features)
            self.poses = []           # List of (R, t)
            self.map_points = []      # 3D map points
            self.loop_closures = []   # Detected loops
            self.keyframe_interval = 3

        def is_keyframe(self, frame_idx):
            """Decide if current frame should be a keyframe."""
            if not self.keyframes:
                return True
            return frame_idx - self.keyframes[-1][0] >= self.keyframe_interval

        def track(self, frame, frame_idx):
            """Estimate pose from features (simplified)."""
            kps, descs = extract_features(frame, n_features=30)

            if not self.keyframes or len(descs) == 0:
                return np.eye(3), np.zeros(3), len(kps)

            # Match with last keyframe
            last_kps, last_descs = (self.keyframes[-1][2][0],
                                    self.keyframes[-1][2][1])
            matches = match_features(last_descs, descs, ratio_thresh=0.85)

            # Estimate relative motion (simplified)
            if len(matches) >= 3:
                # Use average displacement as proxy for motion
                displacements = []
                for m1, m2 in matches:
                    if m1 < len(last_kps) and m2 < len(kps):
                        dx = kps[m2][0] - last_kps[m1][0]
                        dy = kps[m2][1] - last_kps[m1][1]
                        displacements.append((dx, dy))

                if displacements:
                    avg_dx = np.mean([d[0] for d in displacements])
                    avg_dy = np.mean([d[1] for d in displacements])
                    t_est = np.array([avg_dx * 0.01, avg_dy * 0.01, 1.0])
                else:
                    t_est = np.array([0, 0, 1.0])
            else:
                t_est = np.array([0, 0, 1.0])

            R_est = np.eye(3)
            return R_est, t_est, len(matches)

        def detect_loop(self, frame_idx, features):
            """Check for loop closure with earlier keyframes."""
            if len(self.keyframes) < 5:
                return None

            kps, descs = features
            if len(descs) == 0:
                return None

            best_match = None
            best_count = 0

            # Check against non-recent keyframes
            for kf_idx, kf_pose, kf_features in self.keyframes[:-4]:
                kf_kps, kf_descs = kf_features
                if len(kf_descs) == 0:
                    continue

                matches = match_features(descs, kf_descs, ratio_thresh=0.85)
                if len(matches) > best_count and len(matches) >= 5:
                    best_count = len(matches)
                    best_match = kf_idx

            if best_match is not None:
                return {'current': frame_idx, 'match': best_match,
                        'strength': best_count}
            return None

        def optimize_poses(self):
            """Simple pose graph optimization (average correction)."""
            if not self.loop_closures:
                return

            # For each loop closure, compute correction
            for lc in self.loop_closures:
                curr_idx = lc['current']
                match_idx = lc['match']

                # Find poses for these frames
                curr_pose_idx = None
                match_pose_idx = None

                for i, (kf_idx, _, _) in enumerate(self.keyframes):
                    if kf_idx == curr_idx:
                        curr_pose_idx = i
                    if kf_idx == match_idx:
                        match_pose_idx = i

                if curr_pose_idx is not None and match_pose_idx is not None:
                    # Distribute correction across intermediate poses
                    n_between = curr_pose_idx - match_pose_idx
                    if n_between > 0:
                        # Correction per step (simplified)
                        correction = 0.1 / n_between
                        for k in range(match_pose_idx + 1, curr_pose_idx + 1):
                            if k < len(self.poses):
                                self.poses[k] = (
                                    self.poses[k][0],
                                    self.poses[k][1] * (1 - correction)
                                )

        def process(self, frames):
            """Main SLAM processing loop."""
            for i, frame in enumerate(frames):
                # Track
                R, t, n_matches = self.track(frame, i)

                # Accumulate pose
                if self.poses:
                    prev_R, prev_t = self.poses[-1]
                    new_t = prev_t + prev_R @ t
                    new_R = R @ prev_R
                else:
                    new_R, new_t = R, t

                self.poses.append((new_R, new_t.copy()))

                # Keyframe management
                if self.is_keyframe(i):
                    kps, descs = extract_features(frame, n_features=30)
                    features = (kps, descs)
                    self.keyframes.append((i, (new_R, new_t.copy()), features))

                    # Loop closure detection
                    loop = self.detect_loop(i, features)
                    if loop:
                        self.loop_closures.append(loop)

                    # Add map points (simplified)
                    for x, y in kps[:10]:
                        pt_3d = new_R.T @ np.array([
                            (x - 80) * 0.01,
                            (y - 60) * 0.01,
                            1.0
                        ]) + new_t
                        self.map_points.append(pt_3d)

            # Optimize
            self.optimize_poses()

    # Generate synthetic frames (loop trajectory)
    h, w = 120, 160
    n_frames = 25

    frames = []
    for i in range(n_frames):
        frame = np.random.randint(50, 80, (h, w), dtype=np.uint8)

        # Add location-specific features
        # Frames 0-7 and 18-24 should have similar features (loop)
        if i < 8 or i >= 18:
            loc = i if i < 8 else (i - 18)
            np.random.seed(300 + loc % 8)
        else:
            np.random.seed(400 + i)

        for fi in range(20):
            fx = np.random.randint(8, w - 8)
            fy = np.random.randint(8, h - 8)
            frame[fy-3:fy+3, fx-3:fx+3] = np.random.randint(150, 230)

        np.random.seed(42 + i)
        frames.append(frame)

    print("Complete SLAM System")
    print(f"  Frames: {n_frames}")
    print(f"  Image size: {w}x{h}")
    print("=" * 60)

    # Run SLAM
    slam = SimpleSLAM()
    slam.process(frames)

    print(f"\n  Results:")
    print(f"    Keyframes: {len(slam.keyframes)}")
    print(f"    Map points: {len(slam.map_points)}")
    print(f"    Loop closures: {len(slam.loop_closures)}")

    for lc in slam.loop_closures:
        print(f"      Frame {lc['current']} <-> Frame {lc['match']} "
              f"(strength={lc['strength']})")

    # Trajectory summary
    if slam.poses:
        trajectory = np.array([p[1] for p in slam.poses])
        print(f"\n  Trajectory:")
        print(f"    Start: ({trajectory[0][0]:.2f}, "
              f"{trajectory[0][1]:.2f}, {trajectory[0][2]:.2f})")
        print(f"    End:   ({trajectory[-1][0]:.2f}, "
              f"{trajectory[-1][1]:.2f}, {trajectory[-1][2]:.2f})")
        print(f"    Total distance: "
              f"{np.sum(np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))):.2f}")

    # Map statistics
    if slam.map_points:
        map_pts = np.array(slam.map_points)
        print(f"\n  Map Bounds:")
        for axis, name in zip(range(3), ['X', 'Y', 'Z']):
            print(f"    {name}: [{map_pts[:, axis].min():.2f}, "
                  f"{map_pts[:, axis].max():.2f}]")

    return {
        'keyframes': len(slam.keyframes),
        'map_points': len(slam.map_points),
        'loop_closures': slam.loop_closures,
        'trajectory': slam.poses,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Visual Odometry")
    exercise_1_visual_odometry()

    print("\n>>> Exercise 2: Loop Closure Detection")
    exercise_2_loop_closure()

    print("\n>>> Exercise 3: ICP Algorithm")
    exercise_3_icp()

    print("\n>>> Exercise 4: Occupancy Grid Map")
    exercise_4_occupancy_grid()

    print("\n>>> Exercise 5: Complete SLAM System")
    exercise_5_complete_slam()

    print("\nAll exercises completed successfully.")
