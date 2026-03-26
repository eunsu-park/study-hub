"""
Exercise Solutions for Lesson 21: 3D Vision Basics
Computer Vision - Stereo Vision, Point Clouds, 3D Reconstruction

Topics covered:
- Stereo depth estimation (disparity map)
- Point cloud filtering (outlier removal, downsampling, plane extraction)
- 3D reconstruction from two views (Essential Matrix, triangulation)
- Mesh reconstruction from point cloud
- Real-time stereo vision simulation
"""

import numpy as np


# =============================================================================
# Helper: Stereo image generation
# =============================================================================

def generate_stereo_pair(h=100, w=150, baseline=10, focal_length=200):
    """
    Generate synthetic stereo image pair with known depth.

    Returns:
        (left_img, right_img, depth_map, K)
    """
    np.random.seed(42)

    # Camera intrinsics
    K = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    # Create depth map with objects at different distances
    depth_map = np.ones((h, w), dtype=np.float64) * 50  # Background at 50 units

    # Near object (box)
    depth_map[30:60, 20:50] = 15

    # Medium object (circle)
    yy, xx = np.ogrid[:h, :w]
    circle = ((xx - 100)**2 + (yy - 50)**2) <= 20**2
    depth_map[circle] = 25

    # Far object
    depth_map[10:25, 110:140] = 40

    # Generate left image (texture based on depth)
    left = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            d = depth_map[i, j]
            # Closer objects are brighter
            brightness = int(255 * (1 - d / 60))
            left[i, j] = max(30, min(220, brightness + np.random.randint(-10, 10)))

    # Generate right image (shifted by disparity = baseline * f / depth)
    right = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            d = depth_map[i, j]
            disparity = baseline * focal_length / d
            src_j = int(j + disparity)
            if 0 <= src_j < w:
                right[i, j] = left[i, src_j]

    return left, right, depth_map, K


# =============================================================================
# Exercise 1: Stereo Depth Estimation
# =============================================================================

def exercise_1_stereo_depth():
    """
    Compute depth from stereo images using block matching.

    Implements:
    - Block matching (SAD - Sum of Absolute Differences)
    - Disparity to depth conversion
    - Quality comparison at different block sizes

    Returns:
        (disparity_map, depth_estimate)
    """
    left, right, gt_depth, K = generate_stereo_pair()
    h, w = left.shape
    baseline = 10
    focal = K[0, 0]

    def block_matching(left, right, block_size=5, max_disparity=30):
        """Compute disparity map using SAD block matching."""
        h, w = left.shape
        half = block_size // 2
        disparity = np.zeros((h, w), dtype=np.float64)

        left_f = left.astype(np.float64)
        right_f = right.astype(np.float64)

        for i in range(half, h - half):
            for j in range(half + max_disparity, w - half):
                left_block = left_f[i-half:i+half+1, j-half:j+half+1]
                best_d = 0
                best_sad = float('inf')

                for d in range(max_disparity):
                    rj = j - d
                    if rj - half < 0:
                        continue
                    right_block = right_f[i-half:i+half+1,
                                          rj-half:rj+half+1]
                    sad = np.sum(np.abs(left_block - right_block))
                    if sad < best_sad:
                        best_sad = sad
                        best_d = d

                disparity[i, j] = best_d

        return disparity

    print("Stereo Depth Estimation")
    print(f"  Image size: {w}x{h}")
    print(f"  Baseline: {baseline}, Focal: {focal:.0f}")
    print("=" * 60)

    # Test different block sizes
    block_sizes = [3, 5, 9]
    best_error = float('inf')
    best_result = None

    for bs in block_sizes:
        disp = block_matching(left, right, block_size=bs, max_disparity=25)

        # Convert disparity to depth
        depth_est = np.zeros_like(disp)
        valid = disp > 0
        depth_est[valid] = baseline * focal / disp[valid]

        # Compare with ground truth
        valid_gt = (gt_depth > 0) & valid
        if np.sum(valid_gt) > 0:
            errors = np.abs(depth_est[valid_gt] - gt_depth[valid_gt])
            mean_error = np.mean(errors)
            median_error = np.median(errors)
        else:
            mean_error = float('inf')
            median_error = float('inf')

        print(f"\n  Block size {bs}:")
        print(f"    Valid disparities: {np.sum(valid)} / {h*w}")
        print(f"    Disparity range: [{disp[valid].min():.1f}, "
              f"{disp[valid].max():.1f}]" if np.any(valid) else
              "    No valid disparities")
        print(f"    Depth error (mean): {mean_error:.2f}")
        print(f"    Depth error (median): {median_error:.2f}")

        if mean_error < best_error:
            best_error = mean_error
            best_result = (disp, depth_est, bs)

    if best_result:
        print(f"\n  Best block size: {best_result[2]} "
              f"(error={best_error:.2f})")

    return best_result[0], best_result[1] if best_result else (None, None)


# =============================================================================
# Exercise 2: Point Cloud Filtering
# =============================================================================

def exercise_2_point_cloud_filtering():
    """
    Clean and process a noisy 3D point cloud.

    Operations:
    1. Statistical outlier removal
    2. Voxel downsampling
    3. Plane extraction (RANSAC)

    Returns:
        (cleaned_points, plane_model)
    """
    np.random.seed(42)

    # Generate noisy point cloud
    n_points = 500

    # Ground plane (z ~ 0)
    plane_pts = np.column_stack([
        np.random.uniform(-5, 5, 300),
        np.random.uniform(-5, 5, 300),
        np.random.randn(300) * 0.1  # Small z variation
    ])

    # Object points (cube above plane)
    obj_pts = np.column_stack([
        np.random.uniform(1, 3, 150),
        np.random.uniform(-1, 1, 150),
        np.random.uniform(0.5, 2, 150),
    ])

    # Outlier points
    outlier_pts = np.column_stack([
        np.random.uniform(-10, 10, 50),
        np.random.uniform(-10, 10, 50),
        np.random.uniform(-5, 10, 50),
    ])

    points = np.vstack([plane_pts, obj_pts, outlier_pts])
    n_total = len(points)

    print("Point Cloud Filtering")
    print(f"  Total points: {n_total}")
    print(f"  Plane: {len(plane_pts)}, Object: {len(obj_pts)}, "
          f"Outliers: {len(outlier_pts)}")
    print("=" * 60)

    # Step 1: Statistical Outlier Removal
    print("\n  [1] Statistical Outlier Removal:")
    k_neighbors = 20
    std_ratio = 2.0

    # Compute mean distance to k nearest neighbors
    mean_dists = np.zeros(n_total)
    for i in range(n_total):
        dists = np.sqrt(np.sum((points - points[i])**2, axis=1))
        dists_sorted = np.sort(dists)[1:k_neighbors+1]  # Exclude self
        mean_dists[i] = np.mean(dists_sorted)

    # Remove outliers
    global_mean = np.mean(mean_dists)
    global_std = np.std(mean_dists)
    threshold = global_mean + std_ratio * global_std

    inlier_mask = mean_dists < threshold
    clean_pts = points[inlier_mask]

    print(f"    k={k_neighbors}, std_ratio={std_ratio}")
    print(f"    Threshold: {threshold:.3f}")
    print(f"    Removed: {n_total - len(clean_pts)} points")
    print(f"    Remaining: {len(clean_pts)} points")

    # Step 2: Voxel Downsampling
    print("\n  [2] Voxel Downsampling:")
    voxel_size = 0.5

    # Hash points to voxel grid
    voxel_indices = np.floor(clean_pts / voxel_size).astype(int)
    voxel_dict = {}

    for i, key in enumerate(map(tuple, voxel_indices)):
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(i)

    # Average points within each voxel
    downsampled = np.zeros((len(voxel_dict), 3))
    for i, (key, indices) in enumerate(voxel_dict.items()):
        downsampled[i] = np.mean(clean_pts[indices], axis=0)

    print(f"    Voxel size: {voxel_size}")
    print(f"    Before: {len(clean_pts)} points")
    print(f"    After:  {len(downsampled)} points")
    print(f"    Reduction: {100*(1-len(downsampled)/len(clean_pts)):.1f}%")

    # Step 3: Plane Extraction (RANSAC)
    print("\n  [3] Plane Extraction (RANSAC):")
    n_iterations = 100
    dist_threshold = 0.15
    best_plane = None
    best_inliers = 0

    for _ in range(n_iterations):
        # Sample 3 random points
        idx = np.random.choice(len(downsampled), 3, replace=False)
        p1, p2, p3 = downsampled[idx]

        # Compute plane: ax + by + cz + d = 0
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal /= norm_len
        d = -np.dot(normal, p1)

        # Count inliers
        distances = np.abs(downsampled @ normal + d)
        n_inliers = np.sum(distances < dist_threshold)

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_plane = np.append(normal, d)

    if best_plane is not None:
        a, b, c, d = best_plane
        print(f"    Plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
        print(f"    Inliers: {best_inliers} / {len(downsampled)}")
        print(f"    Inlier ratio: {100*best_inliers/len(downsampled):.1f}%")

        # The dominant normal should be close to [0, 0, 1] (z-up)
        z_alignment = abs(best_plane[2])
        print(f"    Z-alignment: {z_alignment:.3f} "
              f"({'horizontal plane' if z_alignment > 0.8 else 'non-horizontal'})")

    return clean_pts, best_plane


# =============================================================================
# Exercise 3: 3D Reconstruction from Two Views
# =============================================================================

def exercise_3_two_view_reconstruction():
    """
    Reconstruct 3D points from two camera views.

    Steps:
    1. Generate corresponding points
    2. Compute Essential Matrix
    3. Recover camera pose (R, t)
    4. Triangulate 3D points

    Returns:
        (points_3d, R, t)
    """
    np.random.seed(42)

    # Camera intrinsics
    f = 500.0
    K = np.array([[f, 0, 320], [0, f, 240], [0, 0, 1]], dtype=np.float64)

    # Ground truth 3D points
    n_points = 30
    pts_3d_true = np.column_stack([
        np.random.uniform(-2, 2, n_points),
        np.random.uniform(-2, 2, n_points),
        np.random.uniform(3, 8, n_points),  # In front of cameras
    ])

    # Camera 1 at origin
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))

    # Camera 2 with known relative pose
    angle = np.radians(10)
    R2_true = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    t2_true = np.array([[0.5], [0.0], [0.1]])

    # Project points to both cameras
    P1 = K @ np.hstack([R1, t1])
    P2 = K @ np.hstack([R2_true, t2_true])

    pts1 = np.zeros((n_points, 2))
    pts2 = np.zeros((n_points, 2))

    for i in range(n_points):
        pt_h = np.append(pts_3d_true[i], 1)
        p1 = P1 @ pt_h
        p2 = P2 @ pt_h
        pts1[i] = p1[:2] / p1[2]
        pts2[i] = p2[:2] / p2[2]

    # Add noise
    pts1 += np.random.randn(n_points, 2) * 0.5
    pts2 += np.random.randn(n_points, 2) * 0.5

    print("3D Reconstruction from Two Views")
    print(f"  Points: {n_points}")
    print(f"  Focal length: {f:.0f}")
    print("=" * 60)

    # Step 1: Compute Essential Matrix using 8-point algorithm
    print("\n  [1] Essential Matrix Estimation:")

    # Normalize points
    pts1_n = np.column_stack([
        (pts1[:, 0] - K[0, 2]) / K[0, 0],
        (pts1[:, 1] - K[1, 2]) / K[1, 1]
    ])
    pts2_n = np.column_stack([
        (pts2[:, 0] - K[0, 2]) / K[0, 0],
        (pts2[:, 1] - K[1, 2]) / K[1, 1]
    ])

    # Build constraint matrix for Essential Matrix
    A = np.zeros((n_points, 9))
    for i in range(n_points):
        x1, y1 = pts1_n[i]
        x2, y2 = pts2_n[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

    _, _, Vt = np.linalg.svd(A)
    E_est = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint
    U, S, Vt2 = np.linalg.svd(E_est)
    S_corrected = np.array([(S[0] + S[1]) / 2, (S[0] + S[1]) / 2, 0])
    E_est = U @ np.diag(S_corrected) @ Vt2

    print(f"    Singular values: {S_corrected}")

    # Step 2: Recover pose
    print("\n  [2] Pose Recovery:")
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)

    # Two possible rotations
    R_candidates = [U @ W @ Vt2, U @ W.T @ Vt2]
    t_candidates = [U[:, 2:3], -U[:, 2:3]]

    # Select correct solution (positive depth for most points)
    best_R = None
    best_t = None
    best_count = 0

    for R_cand in R_candidates:
        # Ensure proper rotation (det = 1)
        if np.linalg.det(R_cand) < 0:
            R_cand = -R_cand

        for t_cand in t_candidates:
            # Count points with positive depth in both cameras
            positive = 0
            for i in range(n_points):
                p1_h = np.append(pts1_n[i], 1)
                # Simple depth check
                p_cam2 = R_cand @ np.append(pts1_n[i], 1.0) + t_cand.flatten()
                if p_cam2[2] > 0:
                    positive += 1

            if positive > best_count:
                best_count = positive
                best_R = R_cand
                best_t = t_cand

    print(f"    Estimated R:\n{best_R}")
    print(f"    Estimated t: {best_t.flatten()}")
    print(f"    Points with positive depth: {best_count}/{n_points}")

    # Step 3: Triangulation
    print("\n  [3] Triangulation:")
    P1_est = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2_est = K @ np.hstack([best_R, best_t]) if best_R is not None else P2

    reconstructed = np.zeros((n_points, 3))
    for i in range(n_points):
        # DLT triangulation
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        A_tri = np.array([
            x1 * P1_est[2] - P1_est[0],
            y1 * P1_est[2] - P1_est[1],
            x2 * P2_est[2] - P2_est[0],
            y2 * P2_est[2] - P2_est[1],
        ])

        _, _, Vt_tri = np.linalg.svd(A_tri)
        X = Vt_tri[-1]
        reconstructed[i] = X[:3] / X[3]

    # Compute reconstruction error
    # Scale ambiguity: normalize by mean distance
    scale_true = np.mean(np.sqrt(np.sum(pts_3d_true**2, axis=1)))
    scale_est = np.mean(np.sqrt(np.sum(reconstructed**2, axis=1)))

    if scale_est > 1e-6:
        reconstructed_scaled = reconstructed * (scale_true / scale_est)
        errors = np.sqrt(np.sum(
            (reconstructed_scaled - pts_3d_true)**2, axis=1))
        print(f"    Mean 3D error (after scaling): {np.mean(errors):.3f}")
        print(f"    Median 3D error: {np.median(errors):.3f}")
    else:
        print(f"    Scale estimation failed")

    print(f"    Reconstructed points range:")
    print(f"      X: [{reconstructed[:, 0].min():.2f}, "
          f"{reconstructed[:, 0].max():.2f}]")
    print(f"      Y: [{reconstructed[:, 1].min():.2f}, "
          f"{reconstructed[:, 1].max():.2f}]")
    print(f"      Z: [{reconstructed[:, 2].min():.2f}, "
          f"{reconstructed[:, 2].max():.2f}]")

    return reconstructed, best_R, best_t


# =============================================================================
# Exercise 4: Mesh Reconstruction
# =============================================================================

def exercise_4_mesh_reconstruction():
    """
    Generate a 3D mesh from a point cloud using Delaunay-like triangulation.

    Steps:
    1. Point cloud preprocessing (normal estimation)
    2. 2D Delaunay triangulation (project to XY plane)
    3. Mesh quality metrics

    Returns:
        (vertices, triangles, normals)
    """
    np.random.seed(42)

    # Generate surface point cloud (hemisphere)
    n_points = 200
    theta = np.random.uniform(0, np.pi / 2, n_points)  # Elevation
    phi = np.random.uniform(0, 2 * np.pi, n_points)    # Azimuth
    r = 5.0 + np.random.randn(n_points) * 0.1  # Slight noise

    points = np.column_stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ])

    print("Mesh Reconstruction")
    print(f"  Input points: {n_points}")
    print(f"  Point cloud bounds:")
    for axis, name in zip(range(3), ['X', 'Y', 'Z']):
        print(f"    {name}: [{points[:, axis].min():.2f}, "
              f"{points[:, axis].max():.2f}]")
    print("=" * 60)

    # Step 1: Normal estimation
    print("\n  [1] Normal Estimation:")
    k_nn = 10
    normals = np.zeros_like(points)

    for i in range(n_points):
        # Find k nearest neighbors
        dists = np.sqrt(np.sum((points - points[i])**2, axis=1))
        nn_idx = np.argsort(dists)[1:k_nn+1]
        neighbors = points[nn_idx]

        # PCA to estimate normal (eigenvector of smallest eigenvalue)
        centered = neighbors - np.mean(neighbors, axis=0)
        cov = centered.T @ centered / len(neighbors)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Smallest eigenvalue's eigenvector = normal
        normal = eigenvectors[:, 0]

        # Orient normal outward (away from center)
        if np.dot(normal, points[i]) < 0:
            normal = -normal

        normals[i] = normal / np.linalg.norm(normal)

    # Check normal consistency
    normal_consistency = np.mean([
        np.dot(normals[i], points[i] / np.linalg.norm(points[i]))
        for i in range(n_points)
    ])
    print(f"    Normal consistency (outward): {normal_consistency:.3f} "
          f"(1.0 = perfect)")

    # Step 2: Simple triangulation (project to XY and use grid-based approach)
    print("\n  [2] Triangulation:")

    # Project to XY plane for 2D triangulation
    pts_2d = points[:, :2]

    # Simple grid-based triangulation
    # Sort points and connect nearest neighbors
    triangles = []
    k_tri = 3  # Connect each point to its nearest neighbors

    for i in range(n_points):
        dists = np.sqrt(np.sum((pts_2d - pts_2d[i])**2, axis=1))
        nn = np.argsort(dists)[1:k_tri+1]

        for j in range(len(nn)):
            for k_idx in range(j + 1, len(nn)):
                tri = sorted([i, nn[j], nn[k_idx]])
                tri_tuple = tuple(tri)
                if tri_tuple not in [tuple(t) for t in triangles]:
                    triangles.append(tri)

    triangles = np.array(triangles[:min(len(triangles), 500)])

    print(f"    Triangles generated: {len(triangles)}")

    # Step 3: Mesh quality metrics
    print("\n  [3] Mesh Quality:")

    if len(triangles) > 0:
        # Edge lengths
        edge_lengths = []
        for tri in triangles:
            for e1, e2 in [(0, 1), (1, 2), (0, 2)]:
                length = np.linalg.norm(points[tri[e1]] - points[tri[e2]])
                edge_lengths.append(length)

        edge_lengths = np.array(edge_lengths)
        print(f"    Edge length: mean={np.mean(edge_lengths):.3f}, "
              f"std={np.std(edge_lengths):.3f}")
        print(f"    Edge range: [{edge_lengths.min():.3f}, "
              f"{edge_lengths.max():.3f}]")

        # Triangle areas
        areas = []
        for tri in triangles:
            v0 = points[tri[0]]
            v1 = points[tri[1]]
            v2 = points[tri[2]]
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            areas.append(area)

        areas = np.array(areas)
        print(f"    Triangle area: mean={np.mean(areas):.4f}, "
              f"total={np.sum(areas):.2f}")

        # Euler's formula check: V - E + F = 2 (for closed mesh)
        edges = set()
        for tri in triangles:
            for e1, e2 in [(0, 1), (1, 2), (0, 2)]:
                edge = tuple(sorted([tri[e1], tri[e2]]))
                edges.add(edge)

        V = n_points
        E = len(edges)
        F = len(triangles)
        euler = V - E + F
        print(f"    Vertices={V}, Edges={E}, Faces={F}")
        print(f"    Euler characteristic: {euler} "
              f"(2 for closed, 1 for open)")

    return points, triangles, normals


# =============================================================================
# Exercise 5: Real-time Stereo Vision
# =============================================================================

def exercise_5_realtime_stereo():
    """
    Simulate real-time stereo depth estimation pipeline.

    Measures:
    - Processing time per stage
    - FPS at different resolutions
    - Depth accuracy vs speed tradeoff

    Returns:
        performance metrics dict
    """
    import time

    print("Real-time Stereo Vision Simulation")
    print("=" * 60)

    # Test at different resolutions
    resolutions = [
        (80, 60, "Low"),
        (160, 120, "Medium"),
        (320, 240, "High"),
    ]

    block_sizes = [3, 5, 9]
    results = {}

    for res_w, res_h, label in resolutions:
        print(f"\n  Resolution: {res_w}x{res_h} ({label})")

        # Generate stereo pair at this resolution
        np.random.seed(42)
        left = np.random.randint(40, 200, (res_h, res_w), dtype=np.uint8)
        right = np.random.randint(40, 200, (res_h, res_w), dtype=np.uint8)

        # Add correlated features
        for y in range(0, res_h - 10, 15):
            for x in range(0, res_w - 10, 20):
                val = np.random.randint(100, 220)
                left[y:y+8, x:x+8] = val
                disp = np.random.randint(2, 10)
                if x - disp >= 0:
                    right[y:y+8, x-disp:x-disp+8] = val

        for bs in block_sizes:
            half = bs // 2
            max_d = min(20, res_w // 4)

            start = time.perf_counter()

            # Rectification (simulated - just copy)
            rect_left = left.copy()
            rect_right = right.copy()

            t_rect = time.perf_counter()

            # Block matching
            disp = np.zeros((res_h, res_w), dtype=np.float64)
            left_f = rect_left.astype(np.float64)
            right_f = rect_right.astype(np.float64)

            # Process only a subset for speed
            sample_step = max(1, res_h // 30)
            for i in range(half, res_h - half, sample_step):
                for j in range(half + max_d, res_w - half):
                    lb = left_f[i-half:i+half+1, j-half:j+half+1]
                    best_d = 0
                    best_sad = float('inf')
                    for d in range(0, max_d, 2):  # Step by 2 for speed
                        rj = j - d
                        if rj - half < 0:
                            continue
                        rb = right_f[i-half:i+half+1, rj-half:rj+half+1]
                        sad = np.sum(np.abs(lb - rb))
                        if sad < best_sad:
                            best_sad = sad
                            best_d = d
                    disp[i, j] = best_d

            t_match = time.perf_counter()

            # Depth conversion
            focal = 200.0
            baseline = 10.0
            depth = np.zeros_like(disp)
            valid = disp > 0
            depth[valid] = baseline * focal / disp[valid]

            t_total = time.perf_counter()

            elapsed = t_total - start
            fps = 1.0 / elapsed if elapsed > 0 else float('inf')
            rect_time = (t_rect - start) * 1000
            match_time = (t_match - t_rect) * 1000
            total_time = elapsed * 1000

            valid_pct = 100 * np.sum(valid) / (res_h * res_w)

            if bs == 5:  # Only print for middle block size
                print(f"    Block={bs}: {total_time:.1f}ms "
                      f"(rect={rect_time:.1f}, match={match_time:.1f}), "
                      f"FPS={fps:.1f}, valid={valid_pct:.0f}%")

            key = f"{label}_{bs}"
            results[key] = {
                'resolution': (res_w, res_h),
                'block_size': bs,
                'total_ms': total_time,
                'fps': fps,
                'valid_pct': valid_pct,
            }

    # Summary
    print(f"\n  Performance Summary:")
    print(f"    {'Config':>15} | {'Time (ms)':>10} | {'FPS':>8} | "
          f"{'Valid':>6}")
    print(f"    {'-'*50}")

    for key in sorted(results.keys()):
        r = results[key]
        res = f"{r['resolution'][0]}x{r['resolution'][1]}_b{r['block_size']}"
        print(f"    {res:>15} | {r['total_ms']:>10.1f} | "
              f"{r['fps']:>8.1f} | {r['valid_pct']:>5.0f}%")

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Stereo Depth Estimation")
    exercise_1_stereo_depth()

    print("\n>>> Exercise 2: Point Cloud Filtering")
    exercise_2_point_cloud_filtering()

    print("\n>>> Exercise 3: 3D Reconstruction from Two Views")
    exercise_3_two_view_reconstruction()

    print("\n>>> Exercise 4: Mesh Reconstruction")
    exercise_4_mesh_reconstruction()

    print("\n>>> Exercise 5: Real-time Stereo Vision")
    exercise_5_realtime_stereo()

    print("\nAll exercises completed successfully.")
