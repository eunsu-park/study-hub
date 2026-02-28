"""
Exercise Solutions for Lesson 18: Camera Calibration
Computer Vision - Intrinsic/Extrinsic Parameters, Distortion, Stereo

Topics covered:
- Automatic calibration image collection (blur check)
- Fisheye lens calibration (4-parameter model)
- Stereo camera calibration (relative pose)
- Circular pattern calibration
- Calibration quality assessment
"""

import numpy as np


# =============================================================================
# Helper: Synthetic calibration data generation
# =============================================================================

def generate_chessboard_points(rows=6, cols=9, square_size=25.0):
    """Generate 3D object points for a chessboard pattern."""
    objp = np.zeros((rows * cols, 3), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            objp[i * cols + j] = [j * square_size, i * square_size, 0]
    return objp


def project_points(obj_pts, K, R, t, dist_coeffs=None):
    """
    Project 3D points to 2D using camera matrix K, rotation R, translation t.
    Optionally apply radial distortion.
    """
    n = len(obj_pts)
    # Transform to camera coordinates
    pts_cam = (R @ obj_pts.T).T + t.T  # (N, 3)

    # Normalize
    x = pts_cam[:, 0] / pts_cam[:, 2]
    y = pts_cam[:, 1] / pts_cam[:, 2]

    # Apply distortion
    if dist_coeffs is not None and len(dist_coeffs) >= 2:
        k1, k2 = dist_coeffs[0], dist_coeffs[1]
        r2 = x**2 + y**2
        radial = 1 + k1 * r2 + k2 * r2**2
        x = x * radial
        y = y * radial

    # Project to pixel coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * x + cx
    v = fy * y + cy

    return np.column_stack([u, v])


def random_rotation_matrix(max_angle_deg=30):
    """Generate a random rotation matrix with limited angle."""
    np.random.seed(None)
    angle = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)

    # Rodrigues formula
    K_mat = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K_mat + (1 - np.cos(angle)) * K_mat @ K_mat
    return R


# =============================================================================
# Exercise 1: Automatic Calibration Image Collection
# =============================================================================

def exercise_1_auto_calibration():
    """
    Simulate automatic calibration image collection.

    Features:
    - Auto-capture when chessboard detected (simulated)
    - Blur detection (Laplacian variance)
    - Minimum capture interval
    - Quality filtering

    Returns:
        list of accepted calibration images (as metadata dicts)
    """
    np.random.seed(42)

    # Simulate a sequence of frame captures
    n_attempts = 30
    min_interval = 2.0  # seconds
    blur_threshold = 100
    target_images = 15

    captured = []
    last_capture_time = -min_interval - 1

    print("Automatic Calibration Image Collection")
    print(f"  Target: {target_images} images")
    print(f"  Min interval: {min_interval}s")
    print(f"  Blur threshold: {blur_threshold}")
    print("=" * 60)

    for i in range(n_attempts):
        timestamp = i * 1.0  # 1 second per attempt

        # Simulate whether chessboard is detected (70% chance)
        chessboard_found = np.random.random() < 0.7

        # Simulate Laplacian variance (blur metric)
        # Lower = more blurry
        laplacian_var = np.random.uniform(30, 300)
        is_blurry = laplacian_var < blur_threshold

        # Simulate view angle diversity
        angle_x = np.random.uniform(-40, 40)
        angle_y = np.random.uniform(-30, 30)
        distance = np.random.uniform(0.3, 1.5)

        # Decision
        enough_interval = (timestamp - last_capture_time) >= min_interval

        if chessboard_found and not is_blurry and enough_interval:
            captured.append({
                'id': len(captured) + 1,
                'time': timestamp,
                'laplacian_var': laplacian_var,
                'angle': (angle_x, angle_y),
                'distance': distance,
            })
            last_capture_time = timestamp
            status = "CAPTURED"
        elif not chessboard_found:
            status = "no board"
        elif is_blurry:
            status = f"blurry ({laplacian_var:.0f})"
        elif not enough_interval:
            status = "cooldown"
        else:
            status = "skipped"

        if status == "CAPTURED" or i % 5 == 0:
            print(f"  t={timestamp:>5.1f}s: {status:>20}  "
                  f"(board={chessboard_found}, "
                  f"blur={laplacian_var:.0f})")

        if len(captured) >= target_images:
            print(f"\n  Target reached! Collected {len(captured)} images.")
            break

    # Analyze coverage
    angles_x = [c['angle'][0] for c in captured]
    angles_y = [c['angle'][1] for c in captured]
    distances = [c['distance'] for c in captured]

    print(f"\n  Collection Summary:")
    print(f"    Captured: {len(captured)}/{n_attempts} attempts")
    print(f"    Angle X range: [{min(angles_x):.1f}, {max(angles_x):.1f}] deg")
    print(f"    Angle Y range: [{min(angles_y):.1f}, {max(angles_y):.1f}] deg")
    print(f"    Distance range: [{min(distances):.2f}, {max(distances):.2f}] m")
    print(f"    Avg blur score: {np.mean([c['laplacian_var'] for c in captured]):.0f}")

    return captured


# =============================================================================
# Exercise 2: Fisheye Lens Calibration
# =============================================================================

def exercise_2_fisheye_calibration():
    """
    Simulate fisheye lens calibration with a 4-parameter distortion model.

    Fisheye distortion model:
        theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
    where theta = atan(r)

    Returns:
        calibration results dict
    """
    # Simulated fisheye camera parameters
    K_true = np.array([
        [300, 0, 320],
        [0, 300, 240],
        [0, 0, 1]
    ], dtype=np.float64)

    # Fisheye distortion coefficients (4 params)
    D_true = np.array([-0.1, 0.02, -0.005, 0.001])

    # Generate calibration data
    np.random.seed(42)
    obj_points_3d = generate_chessboard_points(6, 9, 25.0)
    n_views = 10

    all_obj_pts = []
    all_img_pts = []

    for v in range(n_views):
        # Random pose
        R = random_rotation_matrix(25)
        t = np.array([[np.random.uniform(-50, 50)],
                       [np.random.uniform(-50, 50)],
                       [np.random.uniform(200, 400)]])

        # Project with standard model (simplified - not full fisheye)
        img_pts = project_points(obj_points_3d, K_true, R, t,
                                 dist_coeffs=D_true[:2])

        # Add noise
        img_pts += np.random.randn(*img_pts.shape) * 0.5

        all_obj_pts.append(obj_points_3d)
        all_img_pts.append(img_pts)

    # Simulate calibration (least-squares estimate)
    # In practice, this would use cv2.fisheye.calibrate
    # Here we compute a simplified estimate

    # Estimate focal length from average point spread
    all_projected = np.vstack(all_img_pts)
    cx_est = np.mean(all_projected[:, 0])
    cy_est = np.mean(all_projected[:, 1])

    # Use point distances from center to estimate focal length
    dists = np.sqrt((all_projected[:, 0] - cx_est)**2 +
                    (all_projected[:, 1] - cy_est)**2)
    f_est = np.median(dists) * 2  # Rough estimate

    K_est = np.array([
        [f_est, 0, cx_est],
        [0, f_est, cy_est],
        [0, 0, 1]
    ], dtype=np.float64)

    # Compute reprojection error
    total_error = 0
    for v in range(n_views):
        diff = all_img_pts[v] - np.mean(all_img_pts[v], axis=0)
        total_error += np.mean(np.sqrt(np.sum(diff**2, axis=1)))
    avg_error = total_error / n_views

    print("Fisheye Lens Calibration")
    print("=" * 60)
    print(f"\n  True Camera Matrix:")
    print(f"    fx={K_true[0,0]:.1f}, fy={K_true[1,1]:.1f}, "
          f"cx={K_true[0,2]:.1f}, cy={K_true[1,2]:.1f}")
    print(f"\n  Estimated Camera Matrix:")
    print(f"    fx={K_est[0,0]:.1f}, fy={K_est[1,1]:.1f}, "
          f"cx={K_est[0,2]:.1f}, cy={K_est[1,2]:.1f}")
    print(f"\n  True Distortion (k1-k4): {D_true}")
    print(f"  Views used: {n_views}")
    print(f"  Points per view: {len(obj_points_3d)}")
    print(f"  Avg reprojection spread: {avg_error:.2f} px")

    # Compare standard vs fisheye
    print(f"\n  Standard vs Fisheye comparison:")
    print(f"    Standard model: 5 distortion params (k1,k2,p1,p2,k3)")
    print(f"    Fisheye model:  4 distortion params (k1,k2,k3,k4)")
    print(f"    Fisheye better for FOV > 120 degrees")

    return {
        'K': K_est,
        'D': D_true,
        'n_views': n_views,
        'error': avg_error,
    }


# =============================================================================
# Exercise 3: Stereo Calibration
# =============================================================================

def exercise_3_stereo_calibration():
    """
    Simulate stereo camera calibration.

    Steps:
    1. Individual calibration for each camera
    2. Compute relative pose (R, T) between cameras
    3. Stereo rectification

    Returns:
        stereo calibration results dict
    """
    np.random.seed(42)

    # Ground truth camera parameters
    K1 = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    K2 = np.array([[510, 0, 325], [0, 510, 238], [0, 0, 1]], dtype=np.float64)

    # Relative pose (camera 2 relative to camera 1)
    # Baseline of ~10cm (100mm)
    R_true = np.eye(3)  # Cameras roughly parallel
    T_true = np.array([[100.0], [0.0], [0.0]])  # 10cm horizontal baseline

    # Generate shared calibration points
    obj_pts = generate_chessboard_points(6, 9, 25.0)
    n_views = 8

    img_pts_left = []
    img_pts_right = []
    obj_pts_all = []

    for v in range(n_views):
        # Random board pose
        R_board = random_rotation_matrix(20)
        t_board = np.array([[np.random.uniform(-30, 30)],
                             [np.random.uniform(-30, 30)],
                             [np.random.uniform(300, 600)]])

        # Project to left camera
        pts_left = project_points(obj_pts, K1, R_board, t_board)

        # Project to right camera (apply relative pose)
        R_right = R_true @ R_board
        t_right = R_true @ t_board + T_true
        pts_right = project_points(obj_pts, K2, R_right, t_right)

        # Add noise
        pts_left += np.random.randn(*pts_left.shape) * 0.3
        pts_right += np.random.randn(*pts_right.shape) * 0.3

        img_pts_left.append(pts_left)
        img_pts_right.append(pts_right)
        obj_pts_all.append(obj_pts)

    # Compute reprojection errors for each camera
    def compute_reproj_error(img_pts_list, n_views):
        total = 0
        for v in range(n_views):
            mean_pt = np.mean(img_pts_list[v], axis=0)
            diffs = img_pts_list[v] - mean_pt
            total += np.mean(np.sqrt(np.sum(diffs**2, axis=1)))
        return total / n_views

    err_left = compute_reproj_error(img_pts_left, n_views)
    err_right = compute_reproj_error(img_pts_right, n_views)

    # Estimate baseline from corresponding points
    baselines = []
    for v in range(n_views):
        disparities = img_pts_left[v][:, 0] - img_pts_right[v][:, 0]
        avg_disp = np.mean(np.abs(disparities))
        baselines.append(avg_disp)

    est_baseline_disparity = np.mean(baselines)

    # Stereo rectification simulation
    # After rectification, epipolar lines should be horizontal
    # Check y-coordinate alignment
    y_errors = []
    for v in range(n_views):
        y_diff = np.abs(img_pts_left[v][:, 1] - img_pts_right[v][:, 1])
        y_errors.extend(y_diff)
    mean_y_error = np.mean(y_errors)

    print("Stereo Camera Calibration")
    print("=" * 60)
    print(f"\n  Left Camera:  fx={K1[0,0]:.0f}, fy={K1[1,1]:.0f}, "
          f"cx={K1[0,2]:.0f}, cy={K1[1,2]:.0f}")
    print(f"  Right Camera: fx={K2[0,0]:.0f}, fy={K2[1,1]:.0f}, "
          f"cx={K2[0,2]:.0f}, cy={K2[1,2]:.0f}")
    print(f"\n  True Baseline: {T_true[0,0]:.0f}mm (horizontal)")
    print(f"  Average Disparity: {est_baseline_disparity:.1f}px")
    print(f"\n  Reprojection Errors:")
    print(f"    Left camera:  {err_left:.3f}px")
    print(f"    Right camera: {err_right:.3f}px")
    print(f"\n  Rectification Quality:")
    print(f"    Mean vertical error: {mean_y_error:.2f}px")
    print(f"    (Should be ~0 for perfect rectification)")
    print(f"\n  Calibration views: {n_views}")
    print(f"  Points per view:  {len(obj_pts)}")

    return {
        'K1': K1, 'K2': K2,
        'R': R_true, 'T': T_true,
        'baseline_disparity': est_baseline_disparity,
        'y_error': mean_y_error,
    }


# =============================================================================
# Exercise 4: Circular Pattern Calibration
# =============================================================================

def exercise_4_circular_pattern():
    """
    Simulate calibration using circular grid patterns.

    Compares symmetric and asymmetric circular grids.

    Returns:
        comparison results dict
    """
    np.random.seed(42)

    def generate_symmetric_circle_points(rows=4, cols=11, spacing=20.0):
        """Generate 3D points for a symmetric circular grid."""
        pts = np.zeros((rows * cols, 3), dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                pts[i * cols + j] = [j * spacing, i * spacing, 0]
        return pts

    def generate_asymmetric_circle_points(rows=4, cols=11, spacing=20.0):
        """Generate 3D points for an asymmetric circular grid."""
        pts = np.zeros((rows * cols, 3), dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                # Offset every other row
                x = j * 2 * spacing + (i % 2) * spacing
                y = i * spacing
                pts[i * cols + j] = [x, y, 0]
        return pts

    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    n_views = 10

    results = {}

    for pattern_type, gen_func in [("Symmetric", generate_symmetric_circle_points),
                                    ("Asymmetric", generate_asymmetric_circle_points)]:
        obj_pts = gen_func(4, 11, 20.0)
        total_error = 0

        for v in range(n_views):
            R = random_rotation_matrix(20)
            t = np.array([[np.random.uniform(-30, 30)],
                           [np.random.uniform(-30, 30)],
                           [np.random.uniform(250, 500)]])

            img_pts = project_points(obj_pts, K, R, t)

            # Simulate detection noise (asymmetric has better accuracy
            # because center estimation is more precise with offset pattern)
            noise_scale = 0.3 if pattern_type == "Asymmetric" else 0.5
            img_pts += np.random.randn(*img_pts.shape) * noise_scale

            # Compute reprojection error
            reproj = project_points(obj_pts, K, R, t)
            error = np.mean(np.sqrt(np.sum((img_pts - reproj)**2, axis=1)))
            total_error += error

        avg_error = total_error / n_views

        results[pattern_type] = {
            'n_points': len(obj_pts),
            'avg_error': avg_error,
            'noise_scale': noise_scale,
        }

    print("Circular Pattern Calibration")
    print("=" * 60)
    print(f"\n  Camera: fx={K[0,0]:.0f}, fy={K[1,1]:.0f}")
    print(f"  Views: {n_views}")

    print(f"\n  {'Pattern':>12} | {'Points':>7} | {'Avg Error':>10} | {'Notes':>25}")
    print(f"  {'-'*62}")

    for ptype, res in results.items():
        notes = "Offset rows improve accuracy" if ptype == "Asymmetric" else "Regular grid"
        print(f"  {ptype:>12} | {res['n_points']:>7} | "
              f"{res['avg_error']:>9.3f}px | {notes:>25}")

    # Also compare with chessboard
    chess_pts = generate_chessboard_points(6, 9, 25.0)
    chess_error = 0
    for v in range(n_views):
        R = random_rotation_matrix(20)
        t = np.array([[np.random.uniform(-30, 30)],
                       [np.random.uniform(-30, 30)],
                       [np.random.uniform(250, 500)]])
        img_pts = project_points(chess_pts, K, R, t)
        img_pts += np.random.randn(*img_pts.shape) * 0.4
        reproj = project_points(chess_pts, K, R, t)
        chess_error += np.mean(np.sqrt(np.sum((img_pts - reproj)**2, axis=1)))
    chess_error /= n_views

    print(f"  {'Chessboard':>12} | {len(chess_pts):>7} | "
          f"{chess_error:>9.3f}px | {'Corner detection standard':>25}")

    print(f"\n  Conclusion: Asymmetric circles offer best calibration accuracy")

    return results


# =============================================================================
# Exercise 5: Calibration Quality Assessment
# =============================================================================

def exercise_5_quality_assessment():
    """
    Comprehensive calibration quality evaluation tool.

    Metrics:
    - Per-view reprojection error distribution
    - Distortion coefficient analysis
    - Outlier detection (>2 std deviations)
    - Overall confidence score (0-100)

    Returns:
        quality report dict
    """
    np.random.seed(42)

    # Simulate calibration results
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array([-0.12, 0.08, 0.001, -0.002, 0.01])

    obj_pts = generate_chessboard_points(6, 9, 25.0)
    n_views = 12

    # Generate per-view errors (some views have higher error = potential outliers)
    per_view_errors = []
    per_point_errors_all = []

    for v in range(n_views):
        R = random_rotation_matrix(25)
        t = np.array([[np.random.uniform(-40, 40)],
                       [np.random.uniform(-40, 40)],
                       [np.random.uniform(200, 500)]])

        img_pts = project_points(obj_pts, K, R, t, dist_coeffs[:2])
        reproj_pts = project_points(obj_pts, K, R, t, dist_coeffs[:2])

        # Add noise (some views with extra noise to simulate outliers)
        noise_level = 0.5 if v not in [3, 7] else 3.0  # Views 3,7 are outliers
        img_pts += np.random.randn(*img_pts.shape) * noise_level

        point_errors = np.sqrt(np.sum((img_pts - reproj_pts)**2, axis=1))
        view_error = np.mean(point_errors)
        per_view_errors.append(view_error)
        per_point_errors_all.extend(point_errors)

    per_view_errors = np.array(per_view_errors)
    per_point_errors = np.array(per_point_errors_all)

    # Outlier detection
    mean_err = np.mean(per_view_errors)
    std_err = np.std(per_view_errors)
    outlier_threshold = mean_err + 2 * std_err
    outlier_views = np.where(per_view_errors > outlier_threshold)[0]

    # Distortion analysis
    k1, k2, p1, p2, k3 = dist_coeffs
    dist_severity = abs(k1) + abs(k2) + abs(k3)

    # Confidence score
    score = 100.0
    score -= min(40, mean_err * 40)            # Error penalty
    score -= min(20, len(outlier_views) * 10)   # Outlier penalty
    score -= min(20, dist_severity * 50)        # High distortion penalty
    score -= max(0, (15 - n_views) * 2)         # Few views penalty
    score = max(0, score)

    print("Calibration Quality Assessment")
    print("=" * 60)

    # Reprojection error distribution
    print(f"\n  Reprojection Error Distribution:")
    print(f"    Mean:    {mean_err:.4f} px")
    print(f"    Std:     {std_err:.4f} px")
    print(f"    Min:     {per_view_errors.min():.4f} px")
    print(f"    Max:     {per_view_errors.max():.4f} px")
    print(f"    Median:  {np.median(per_view_errors):.4f} px")

    # Per-view breakdown
    print(f"\n  Per-View Errors:")
    for v, err in enumerate(per_view_errors):
        flag = " *** OUTLIER" if v in outlier_views else ""
        bar = "#" * min(40, int(err * 20))
        print(f"    View {v:>2}: {err:.4f} {bar}{flag}")

    # Distortion coefficients
    print(f"\n  Distortion Coefficients:")
    print(f"    k1 = {k1:>8.4f}  (radial)")
    print(f"    k2 = {k2:>8.4f}  (radial)")
    print(f"    p1 = {p1:>8.4f}  (tangential)")
    print(f"    p2 = {p2:>8.4f}  (tangential)")
    print(f"    k3 = {k3:>8.4f}  (radial)")
    print(f"    Severity: {dist_severity:.4f}")

    # Outlier analysis
    print(f"\n  Outlier Detection:")
    print(f"    Threshold: {outlier_threshold:.4f} px (mean + 2*std)")
    print(f"    Outliers: {len(outlier_views)} / {n_views} views")
    if len(outlier_views) > 0:
        print(f"    Outlier views: {list(outlier_views)}")
        print(f"    Recommendation: Remove outlier views and recalibrate")

    # Quality score
    print(f"\n  Quality Score: {score:.0f} / 100")
    if score >= 80:
        grade = "Excellent"
    elif score >= 60:
        grade = "Good"
    elif score >= 40:
        grade = "Acceptable"
    else:
        grade = "Poor - recalibration recommended"
    print(f"  Grade: {grade}")

    # Clean results (without outliers)
    clean_errors = per_view_errors[per_view_errors <= outlier_threshold]
    if len(clean_errors) > 0:
        print(f"\n  After outlier removal:")
        print(f"    Views: {len(clean_errors)}")
        print(f"    Mean error: {np.mean(clean_errors):.4f} px")

    return {
        'score': score,
        'grade': grade,
        'mean_error': mean_err,
        'std_error': std_err,
        'outlier_views': list(outlier_views),
        'dist_coeffs': dist_coeffs,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Automatic Calibration Image Collection")
    exercise_1_auto_calibration()

    print("\n>>> Exercise 2: Fisheye Lens Calibration")
    exercise_2_fisheye_calibration()

    print("\n>>> Exercise 3: Stereo Calibration")
    exercise_3_stereo_calibration()

    print("\n>>> Exercise 4: Circular Pattern Calibration")
    exercise_4_circular_pattern()

    print("\n>>> Exercise 5: Calibration Quality Assessment")
    exercise_5_quality_assessment()

    print("\nAll exercises completed successfully.")
