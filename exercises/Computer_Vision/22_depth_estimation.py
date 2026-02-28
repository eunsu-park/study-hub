"""
Exercise Solutions for Lesson 22: Depth Estimation
Computer Vision - Monocular Depth, MiDaS/DPT Simulation, SfM, 3D Viewer

Topics covered:
- MiDaS-style monocular depth estimation (simulated)
- Depth-based background blur (portrait mode)
- Structure from Motion (SfM) reconstruction
- Real-time depth estimation pipeline
- Depth-to-point-cloud 3D viewer
"""

import numpy as np


# =============================================================================
# Helper: Simulated depth estimation
# =============================================================================

def estimate_depth_monocular(img):
    """
    Simulate monocular depth estimation.

    Uses simple heuristics as a proxy for a DNN model:
    - Lower region = closer (floor assumption)
    - Brighter objects = closer (simplistic)
    - Texture density correlates with distance

    Returns:
        depth_map (H, W) with relative depth values (higher = farther)
    """
    h, w = img.shape[:2]

    if len(img.shape) == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img.astype(np.float64)

    depth = np.zeros((h, w), dtype=np.float64)

    # Vertical position prior (bottom = close, top = far)
    for i in range(h):
        depth[i, :] = 0.3 + 0.7 * (1 - i / h)

    # Brightness modulation (brighter objects appear closer)
    bright_norm = gray / 255.0
    depth *= (0.5 + 0.5 * (1 - bright_norm))

    # Edge density modulation (more texture = closer)
    edges = np.zeros((h, w), dtype=np.float64)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            gx = float(gray[i, j+1]) - float(gray[i, j-1])
            gy = float(gray[i+1, j]) - float(gray[i-1, j])
            edges[i, j] = np.sqrt(gx**2 + gy**2)

    edge_norm = edges / max(edges.max(), 1)
    depth *= (0.7 + 0.3 * (1 - edge_norm))

    # Normalize to [0, 1] range
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)

    return depth


# =============================================================================
# Exercise 1: MiDaS Depth Estimation (Simulated)
# =============================================================================

def exercise_1_midas_depth():
    """
    Simulate MiDaS-style depth estimation on multiple test images.

    Pipeline:
    1. Load/generate test images
    2. Run depth estimation
    3. Apply colormap visualization
    4. Compare across images

    Returns:
        list of (image_name, depth_map, statistics) tuples
    """
    np.random.seed(42)

    # Create diverse test images
    test_images = {}

    # Scene 1: Indoor (bright foreground object, dark background)
    indoor = np.ones((100, 150), dtype=np.uint8) * 60
    indoor[50:90, 40:110] = 180  # Bright table
    indoor[55:75, 60:90] = 200   # Object on table
    test_images['Indoor'] = indoor

    # Scene 2: Outdoor (gradient sky, textured ground)
    outdoor = np.zeros((100, 150), dtype=np.uint8)
    for i in range(50):
        outdoor[i, :] = int(100 + i * 2)  # Sky gradient
    for i in range(50, 100):
        outdoor[i, :] = np.random.randint(60, 120)  # Ground texture
    # Add tree
    outdoor[20:70, 30:45] = 50
    test_images['Outdoor'] = outdoor

    # Scene 3: Portrait (person in center)
    portrait = np.ones((100, 150), dtype=np.uint8) * 80
    # Body
    yy, xx = np.ogrid[:100, :150]
    body_mask = ((xx - 75)**2 / 20**2 + (yy - 65)**2 / 30**2) <= 1
    portrait[body_mask] = 160
    # Head
    head_mask = ((xx - 75)**2 + (yy - 30)**2) <= 15**2
    portrait[head_mask] = 180
    test_images['Portrait'] = portrait

    results = []

    print("MiDaS Depth Estimation (Simulated)")
    print("=" * 60)

    for name, img in test_images.items():
        depth = estimate_depth_monocular(img)

        stats = {
            'mean': np.mean(depth),
            'std': np.std(depth),
            'min': np.min(depth),
            'max': np.max(depth),
        }

        # Apply pseudo-colormap (near=red, far=blue)
        color_depth = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                d = depth[i, j]
                # Blue (far) to Red (near)
                color_depth[i, j, 0] = int(255 * (1 - d))  # B: far=bright
                color_depth[i, j, 2] = int(255 * d)         # R: near=bright

        results.append((name, depth, stats))

        print(f"\n  {name}:")
        print(f"    Image size: {img.shape[1]}x{img.shape[0]}")
        print(f"    Depth range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"    Depth mean: {stats['mean']:.3f}, std: {stats['std']:.3f}")
        print(f"    Colormap: shape={color_depth.shape}")

    return results


# =============================================================================
# Exercise 2: Depth-based Background Blur
# =============================================================================

def exercise_2_background_blur():
    """
    Apply depth-dependent blur to create portrait mode effect.

    Steps:
    1. Estimate depth
    2. Separate foreground/background using depth threshold
    3. Apply variable blur to background
    4. Smooth boundaries for natural transition

    Returns:
        (original, blurred, mask)
    """
    np.random.seed(42)

    # Create portrait image
    h, w = 100, 120
    img = np.random.randint(60, 90, (h, w), dtype=np.uint8)

    # Person (foreground)
    yy, xx = np.ogrid[:h, :w]
    body = ((xx - 60)**2 / 18**2 + (yy - 65)**2 / 28**2) <= 1
    head = ((xx - 60)**2 + (yy - 30)**2) <= 14**2
    person = body | head
    img[person] = np.random.randint(150, 190, np.sum(person))

    # Background objects
    img[5:20, 90:115] = 70   # Dark window
    img[40:60, 5:20] = 100   # Shelf

    # Estimate depth
    depth = estimate_depth_monocular(img)

    # Threshold: closest 35% = foreground
    threshold = np.percentile(depth, 35)
    fg_mask = depth > threshold  # Higher depth = closer in our model

    # Smooth the mask for natural boundaries
    mask_float = fg_mask.astype(np.float64)

    # Simple box blur on mask (5x5)
    blur_size = 5
    pad = blur_size // 2
    mask_smooth = np.zeros_like(mask_float)
    padded_mask = np.pad(mask_float, pad, mode='reflect')
    for i in range(h):
        for j in range(w):
            mask_smooth[i, j] = np.mean(
                padded_mask[i:i+blur_size, j:j+blur_size])

    # Apply blur to background
    # Create blurred version (box blur with large kernel)
    blur_kernel = 11
    pad_b = blur_kernel // 2
    padded_img = np.pad(img.astype(np.float64), pad_b, mode='reflect')
    blurred_bg = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            blurred_bg[i, j] = np.mean(
                padded_img[i:i+blur_kernel, j:j+blur_kernel])

    # Composite: foreground sharp + background blurred
    result = (img.astype(np.float64) * mask_smooth +
              blurred_bg * (1 - mask_smooth))
    result = np.clip(result, 0, 255).astype(np.uint8)

    print("Depth-based Background Blur")
    print("=" * 60)
    print(f"  Image size: {w}x{h}")
    print(f"  Depth threshold: {threshold:.3f}")
    print(f"  Foreground pixels: {np.sum(fg_mask)} ({100*np.mean(fg_mask):.1f}%)")
    print(f"  Background pixels: {np.sum(~fg_mask)} ({100*np.mean(~fg_mask):.1f}%)")
    print(f"  Blur kernel: {blur_kernel}x{blur_kernel}")

    # Quality metrics
    fg_sharpness = np.std(result[fg_mask])
    bg_sharpness = np.std(result[~fg_mask])
    print(f"\n  Sharpness (std):")
    print(f"    Foreground: {fg_sharpness:.1f}")
    print(f"    Background: {bg_sharpness:.1f}")
    print(f"    Ratio: {fg_sharpness/max(bg_sharpness,1):.2f}x "
          f"(higher = better separation)")

    return img, result, mask_smooth


# =============================================================================
# Exercise 3: Structure from Motion (SfM)
# =============================================================================

def exercise_3_sfm():
    """
    Reconstruct 3D point cloud from two views using SfM pipeline.

    Pipeline:
    1. Feature matching between views
    2. Essential Matrix estimation
    3. Camera pose recovery
    4. Triangulation of matched points

    Returns:
        (points_3d, camera_poses)
    """
    np.random.seed(42)

    # Camera intrinsics
    f = 500.0
    K = np.array([[f, 0, 320], [0, f, 240], [0, 0, 1]], dtype=np.float64)

    # Generate 3D scene (random points in a box)
    n_pts = 40
    scene_pts = np.column_stack([
        np.random.uniform(-3, 3, n_pts),
        np.random.uniform(-2, 2, n_pts),
        np.random.uniform(4, 10, n_pts),
    ])

    # Two camera poses
    R1 = np.eye(3)
    t1 = np.zeros(3)

    angle = np.radians(8)
    R2 = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    t2 = np.array([0.8, 0.0, 0.1])

    # Project to both views
    def project(pts, K, R, t):
        projected = []
        for p in pts:
            p_cam = R @ p + t
            if p_cam[2] > 0:
                px = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
                py = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
                projected.append((px, py))
            else:
                projected.append(None)
        return projected

    proj1 = project(scene_pts, K, R1, t1)
    proj2 = project(scene_pts, K, R2, t2)

    # Filter valid correspondences
    valid_pairs = []
    for i in range(n_pts):
        if proj1[i] is not None and proj2[i] is not None:
            valid_pairs.append((
                np.array(proj1[i]) + np.random.randn(2) * 0.3,
                np.array(proj2[i]) + np.random.randn(2) * 0.3,
                i
            ))

    print("Structure from Motion (SfM)")
    print(f"  Scene points: {n_pts}")
    print(f"  Valid correspondences: {len(valid_pairs)}")
    print("=" * 60)

    # Essential Matrix using normalized 8-point algorithm
    pts1 = np.array([p[0] for p in valid_pairs])
    pts2 = np.array([p[1] for p in valid_pairs])

    # Normalize
    pts1_n = np.column_stack([
        (pts1[:, 0] - K[0, 2]) / K[0, 0],
        (pts1[:, 1] - K[1, 2]) / K[1, 1]
    ])
    pts2_n = np.column_stack([
        (pts2[:, 0] - K[0, 2]) / K[0, 0],
        (pts2[:, 1] - K[1, 2]) / K[1, 1]
    ])

    n_valid = len(valid_pairs)
    A = np.zeros((n_valid, 9))
    for i in range(n_valid):
        x1, y1 = pts1_n[i]
        x2, y2 = pts2_n[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

    _, _, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)

    # Enforce rank-2
    U, S, Vt2 = np.linalg.svd(E)
    S_new = np.array([(S[0]+S[1])/2, (S[0]+S[1])/2, 0])
    E = U @ np.diag(S_new) @ Vt2

    print(f"\n  Essential Matrix singular values: "
          f"[{S_new[0]:.4f}, {S_new[1]:.4f}, {S_new[2]:.4f}]")

    # Triangulate points
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R2, t2.reshape(3, 1)])

    reconstructed = []
    for i in range(n_valid):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        A_tri = np.array([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1],
        ])

        _, _, Vt_t = np.linalg.svd(A_tri)
        X = Vt_t[-1]
        pt_3d = X[:3] / X[3]
        reconstructed.append(pt_3d)

    reconstructed = np.array(reconstructed)
    gt_pts = scene_pts[[p[2] for p in valid_pairs]]

    # Compute error (with scale normalization)
    scale_gt = np.mean(np.linalg.norm(gt_pts, axis=1))
    scale_est = np.mean(np.linalg.norm(reconstructed, axis=1))
    if scale_est > 1e-6:
        reconstructed_scaled = reconstructed * (scale_gt / scale_est)
        errors = np.linalg.norm(reconstructed_scaled - gt_pts, axis=1)
        print(f"\n  Reconstruction Results:")
        print(f"    Mean error: {np.mean(errors):.4f}")
        print(f"    Median error: {np.median(errors):.4f}")
        print(f"    Max error: {np.max(errors):.4f}")
    else:
        print(f"    Scale estimation failed")
        errors = np.zeros(n_valid)

    print(f"    Reconstructed points: {len(reconstructed)}")
    print(f"    Point cloud bounds:")
    for axis, name in zip(range(3), ['X', 'Y', 'Z']):
        print(f"      {name}: [{reconstructed[:, axis].min():.2f}, "
              f"{reconstructed[:, axis].max():.2f}]")

    return reconstructed, [{'R': R1, 't': t1}, {'R': R2, 't': t2}]


# =============================================================================
# Exercise 4: Real-time Depth Estimation
# =============================================================================

def exercise_4_realtime_depth():
    """
    Simulate real-time monocular depth estimation pipeline.

    Measures performance at different scales and evaluates
    speed vs quality tradeoff.

    Returns:
        performance results dict
    """
    import time

    print("Real-time Depth Estimation")
    print("=" * 60)

    # Test configurations (scale factor, name)
    configs = [
        (1.0, "Full resolution"),
        (0.75, "75% scale"),
        (0.5, "50% scale"),
        (0.25, "25% scale"),
    ]

    base_h, base_w = 120, 160
    n_frames = 5

    results = {}

    for scale, name in configs:
        h = int(base_h * scale)
        w = int(base_w * scale)

        times = []
        depth_stats = []

        for f in range(n_frames):
            np.random.seed(42 + f)
            frame = np.random.randint(40, 200, (h, w), dtype=np.uint8)
            # Add structure
            frame[h//3:2*h//3, w//4:3*w//4] = np.random.randint(150, 220)

            start = time.perf_counter()
            depth = estimate_depth_monocular(frame)
            elapsed = time.perf_counter() - start

            times.append(elapsed)
            depth_stats.append({
                'mean': np.mean(depth),
                'std': np.std(depth),
            })

        avg_time = np.mean(times) * 1000  # ms
        fps = 1000 / avg_time if avg_time > 0 else float('inf')
        avg_std = np.mean([s['std'] for s in depth_stats])

        results[name] = {
            'resolution': (w, h),
            'time_ms': avg_time,
            'fps': fps,
            'depth_detail': avg_std,
        }

        print(f"\n  {name} ({w}x{h}):")
        print(f"    Avg time: {avg_time:.1f}ms")
        print(f"    FPS: {fps:.1f}")
        print(f"    Depth detail (std): {avg_std:.4f}")

    # Summary comparison
    print(f"\n  Speed vs Quality Tradeoff:")
    print(f"    {'Config':>20} | {'FPS':>8} | {'Detail':>8} | "
          f"{'Quality/FPS':>12}")
    print(f"    {'-'*55}")

    for name, r in results.items():
        qps = r['depth_detail'] * r['fps']
        print(f"    {name:>20} | {r['fps']:>8.1f} | "
              f"{r['depth_detail']:>.4f} | {qps:>12.2f}")

    # Recommendation
    best = max(results.items(),
               key=lambda x: x[1]['depth_detail'] * min(x[1]['fps'], 30))
    print(f"\n  Recommendation: '{best[0]}' for best quality-speed balance")

    return results


# =============================================================================
# Exercise 5: Depth-to-Point-Cloud 3D Viewer
# =============================================================================

def exercise_5_depth_to_pointcloud():
    """
    Convert a depth map to a 3D point cloud.

    Steps:
    1. Back-project depth pixels to 3D
    2. Assign colors from original image
    3. Filter invalid points
    4. Compute point cloud statistics

    Returns:
        (points_3d, colors, statistics)
    """
    np.random.seed(42)

    # Create test image and depth map
    h, w = 80, 100
    img = np.random.randint(50, 180, (h, w, 3), dtype=np.uint8)

    # Foreground object
    yy, xx = np.ogrid[:h, :w]
    fg = ((xx - 50)**2 + (yy - 40)**2) <= 20**2
    img[fg] = [180, 100, 80]

    # Depth map
    depth = np.ones((h, w), dtype=np.float64) * 5.0  # Background
    depth[fg] = 2.0  # Foreground closer
    depth += np.random.randn(h, w) * 0.1
    depth = np.clip(depth, 0.5, 10)

    # Camera intrinsics
    fx, fy = 200.0, 200.0
    cx, cy = w / 2.0, h / 2.0

    print("Depth-to-Point-Cloud Conversion")
    print(f"  Image: {w}x{h}x3")
    print(f"  Depth range: [{depth.min():.2f}, {depth.max():.2f}]")
    print(f"  Camera: fx={fx:.0f}, fy={fy:.0f}, cx={cx:.0f}, cy={cy:.0f}")
    print("=" * 60)

    # Back-project to 3D
    points_3d = []
    colors = []
    valid_count = 0

    for i in range(h):
        for j in range(w):
            d = depth[i, j]
            if d <= 0 or d > 10:
                continue

            # Back-projection: (u, v, d) -> (X, Y, Z)
            X = (j - cx) * d / fx
            Y = (i - cy) * d / fy
            Z = d

            points_3d.append([X, Y, Z])
            colors.append(img[i, j] / 255.0)  # Normalize to [0, 1]
            valid_count += 1

    points_3d = np.array(points_3d)
    colors = np.array(colors)

    print(f"\n  Point Cloud:")
    print(f"    Total pixels: {h * w}")
    print(f"    Valid points: {valid_count}")
    print(f"    Invalid (filtered): {h * w - valid_count}")

    # Point cloud statistics
    stats = {}
    for axis, name in zip(range(3), ['X', 'Y', 'Z']):
        vals = points_3d[:, axis]
        stats[name] = {
            'min': vals.min(),
            'max': vals.max(),
            'mean': vals.mean(),
            'std': vals.std(),
        }
        print(f"    {name}: [{vals.min():.3f}, {vals.max():.3f}], "
              f"mean={vals.mean():.3f}")

    # Bounding box
    bbox_min = points_3d.min(axis=0)
    bbox_max = points_3d.max(axis=0)
    bbox_size = bbox_max - bbox_min
    print(f"\n  Bounding Box: {bbox_size[0]:.2f} x {bbox_size[1]:.2f} x "
          f"{bbox_size[2]:.2f}")

    # Estimate surface area (convex hull approximation)
    # Simple: bounding box surface area
    sa = 2 * (bbox_size[0] * bbox_size[1] +
              bbox_size[1] * bbox_size[2] +
              bbox_size[0] * bbox_size[2])
    print(f"  Bounding box surface area: {sa:.2f}")

    # Point density
    volume = np.prod(bbox_size) if np.all(bbox_size > 0) else 1
    density = valid_count / volume
    print(f"  Point density: {density:.1f} pts/unit^3")

    # Foreground/background separation in 3D
    fg_mask_3d = points_3d[:, 2] < 3.0  # Close points
    n_fg = np.sum(fg_mask_3d)
    n_bg = np.sum(~fg_mask_3d)
    print(f"\n  Segmentation:")
    print(f"    Foreground (Z<3): {n_fg} points")
    print(f"    Background (Z>=3): {n_bg} points")

    return points_3d, colors, stats


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: MiDaS Depth Estimation")
    exercise_1_midas_depth()

    print("\n>>> Exercise 2: Depth-based Background Blur")
    exercise_2_background_blur()

    print("\n>>> Exercise 3: Structure from Motion (SfM)")
    exercise_3_sfm()

    print("\n>>> Exercise 4: Real-time Depth Estimation")
    exercise_4_realtime_depth()

    print("\n>>> Exercise 5: Depth-to-Point-Cloud 3D Viewer")
    exercise_5_depth_to_pointcloud()

    print("\nAll exercises completed successfully.")
