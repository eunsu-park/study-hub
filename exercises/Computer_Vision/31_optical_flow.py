"""
Exercise Solutions for Lesson 31: Optical Flow
Computer Vision - Lucas-Kanade, Horn-Schunck, Flow Visualization, Motion Estimation

Topics covered:
- Brightness constancy and optical flow equation
- Lucas-Kanade sparse optical flow
- Horn-Schunck dense optical flow
- Flow visualization (color coding)
- End-point error evaluation
- Motion segmentation from flow fields
- Video stabilization using optical flow
"""

import numpy as np


# =============================================================================
# Helper: Synthetic frame pair generation
# =============================================================================

def generate_frame_pair(h=60, w=80, motion_type="translate"):
    """
    Generate two frames with known ground-truth optical flow.

    Motion types:
    - "translate": uniform translation
    - "rotate": rotation about center
    - "diverge": expansion from center
    - "mixed": combination of motions with moving objects

    Returns:
        (frame1, frame2, gt_flow) where gt_flow is (h, w, 2) [u, v]
    """
    np.random.seed(42)

    # Create textured frame 1
    frame1 = np.zeros((h, w), dtype=np.float64)
    # Random texture patches
    for _ in range(30):
        py = np.random.randint(0, h - 5)
        px = np.random.randint(0, w - 5)
        sz = np.random.randint(3, 8)
        val = np.random.uniform(80, 220)
        frame1[py:min(py+sz, h), px:min(px+sz, w)] = val
    # Add gradient background
    for i in range(h):
        for j in range(w):
            frame1[i, j] += 30 + 20 * np.sin(i * 0.15) * np.cos(j * 0.12)

    gt_flow = np.zeros((h, w, 2), dtype=np.float64)

    if motion_type == "translate":
        dx, dy = 3.0, 1.5
        gt_flow[:, :, 0] = dx
        gt_flow[:, :, 1] = dy

    elif motion_type == "rotate":
        angle = np.radians(3)
        cy, cx = h / 2, w / 2
        for i in range(h):
            for j in range(w):
                x_off = j - cx
                y_off = i - cy
                gt_flow[i, j, 0] = x_off * (np.cos(angle) - 1) - y_off * np.sin(angle)
                gt_flow[i, j, 1] = x_off * np.sin(angle) + y_off * (np.cos(angle) - 1)

    elif motion_type == "diverge":
        cy, cx = h / 2, w / 2
        expansion = 0.05
        for i in range(h):
            for j in range(w):
                gt_flow[i, j, 0] = (j - cx) * expansion
                gt_flow[i, j, 1] = (i - cy) * expansion

    elif motion_type == "mixed":
        # Background translation
        gt_flow[:, :, 0] = 2.0
        gt_flow[:, :, 1] = 0.5
        # Moving object (different velocity)
        for i in range(20, 40):
            for j in range(30, 55):
                gt_flow[i, j, 0] = -3.0
                gt_flow[i, j, 1] = 1.0
                frame1[i, j] = 180

    # Generate frame 2 by warping frame 1
    frame2 = np.zeros_like(frame1)
    for i in range(h):
        for j in range(w):
            src_x = j - gt_flow[i, j, 0]
            src_y = i - gt_flow[i, j, 1]
            si, sj = int(round(src_y)), int(round(src_x))
            if 0 <= si < h and 0 <= sj < w:
                frame2[i, j] = frame1[si, sj]

    return frame1, frame2, gt_flow


# =============================================================================
# Exercise 1: Optical Flow Equation
# =============================================================================

def exercise_1_flow_equation():
    """
    Derive and verify the optical flow constraint equation.

    Ix*u + Iy*v + It = 0

    Steps:
    1. Compute spatial gradients (Ix, Iy)
    2. Compute temporal gradient (It)
    3. Verify constraint with known flow
    4. Analyze aperture problem

    Returns:
        (Ix, Iy, It, residuals)
    """
    h, w = 60, 80
    frame1, frame2, gt_flow = generate_frame_pair(h, w, "translate")

    print("Optical Flow Constraint Equation")
    print(f"  Image size: {w}x{h}")
    print("=" * 60)

    # Compute spatial gradients (central differences)
    Ix = np.zeros((h, w), dtype=np.float64)
    Iy = np.zeros((h, w), dtype=np.float64)
    Ix[:, 1:-1] = (frame1[:, 2:] - frame1[:, :-2]) / 2.0
    Iy[1:-1, :] = (frame1[2:, :] - frame1[:-2, :]) / 2.0

    # Temporal gradient
    It = frame2 - frame1

    # Verify: Ix*u + Iy*v + It should be ~0
    u_gt = gt_flow[:, :, 0]
    v_gt = gt_flow[:, :, 1]
    residual = Ix * u_gt + Iy * v_gt + It

    print(f"\n  Gradient Statistics:")
    print(f"    Ix range: [{Ix.min():.2f}, {Ix.max():.2f}]")
    print(f"    Iy range: [{Iy.min():.2f}, {Iy.max():.2f}]")
    print(f"    It range: [{It.min():.2f}, {It.max():.2f}]")

    print(f"\n  Constraint Verification (Ix*u + Iy*v + It = 0):")
    # Exclude border pixels
    inner = residual[2:-2, 2:-2]
    print(f"    Residual mean: {inner.mean():.4f}")
    print(f"    Residual std:  {inner.std():.4f}")
    print(f"    Residual |max|: {np.abs(inner).max():.4f}")

    # Aperture problem analysis
    print(f"\n  Aperture Problem Analysis:")
    # Find regions with strong gradient in only one direction
    gradient_mag = np.sqrt(Ix**2 + Iy**2)
    strong_grad = gradient_mag > 10

    if strong_grad.sum() > 0:
        # Ratio of gradients indicates edge orientation
        angle = np.arctan2(Iy[strong_grad], Ix[strong_grad])
        print(f"    Strong gradient pixels: {strong_grad.sum()}")
        print(f"    Gradient angle range: [{np.degrees(angle.min()):.0f}, "
              f"{np.degrees(angle.max()):.0f}] degrees")

        # At edges, we can only determine normal flow
        normal_flow = np.abs(It[strong_grad]) / gradient_mag[strong_grad]
        print(f"    Normal flow range: [{normal_flow.min():.3f}, "
              f"{normal_flow.max():.3f}]")
        print(f"    Full flow is underdetermined along edges!")
    else:
        print(f"    No strong gradient pixels found")

    # Regions with texture (both Ix and Iy are significant)
    textured = (np.abs(Ix) > 5) & (np.abs(Iy) > 5)
    print(f"\n  Textured pixels (solvable): {textured.sum()} / {h*w}")
    print(f"  Under-constrained pixels: {(h*w - textured.sum())}")

    return Ix, Iy, It, residual


# =============================================================================
# Exercise 2: Lucas-Kanade Sparse Flow
# =============================================================================

def exercise_2_lucas_kanade():
    """
    Implement Lucas-Kanade optical flow estimation.

    Assumes constant flow within a local window.
    Solves: [Ix Iy]^T [Ix Iy] [u v]^T = -[Ix Iy]^T [It]

    Returns:
        (flow_u, flow_v, keypoints)
    """
    h, w = 60, 80
    frame1, frame2, gt_flow = generate_frame_pair(h, w, "translate")

    print("Lucas-Kanade Sparse Optical Flow")
    print(f"  Image size: {w}x{h}")
    print("=" * 60)

    # Compute gradients
    Ix = np.zeros((h, w), dtype=np.float64)
    Iy = np.zeros((h, w), dtype=np.float64)
    Ix[:, 1:-1] = (frame1[:, 2:] - frame1[:, :-2]) / 2.0
    Iy[1:-1, :] = (frame1[2:, :] - frame1[:-2, :]) / 2.0
    It = frame2 - frame1

    # Find good features to track (Harris corners)
    win = 3
    corner_response = np.zeros((h, w), dtype=np.float64)
    for i in range(win, h - win):
        for j in range(win, w - win):
            Ixx = np.sum(Ix[i-win:i+win+1, j-win:j+win+1] ** 2)
            Iyy = np.sum(Iy[i-win:i+win+1, j-win:j+win+1] ** 2)
            Ixy = np.sum(Ix[i-win:i+win+1, j-win:j+win+1] *
                        Iy[i-win:i+win+1, j-win:j+win+1])
            det = Ixx * Iyy - Ixy ** 2
            trace = Ixx + Iyy
            corner_response[i, j] = det - 0.04 * trace ** 2

    # Select top corners
    n_features = 30
    threshold = max(corner_response.max() * 0.01, 1e-6)
    keypoints = []

    for i in range(win + 1, h - win - 1):
        for j in range(win + 1, w - win - 1):
            if corner_response[i, j] > threshold:
                local = corner_response[i-1:i+2, j-1:j+2]
                if corner_response[i, j] == local.max():
                    keypoints.append((j, i, corner_response[i, j]))

    keypoints.sort(key=lambda x: x[2], reverse=True)
    keypoints = keypoints[:n_features]

    print(f"  Features detected: {len(keypoints)}")

    # LK flow estimation at each keypoint
    lk_window = 7
    half_w = lk_window // 2
    flow_results = []

    for kp_x, kp_y, _ in keypoints:
        if (half_w <= kp_y < h - half_w and half_w <= kp_x < w - half_w):
            # Extract local window
            win_Ix = Ix[kp_y-half_w:kp_y+half_w+1, kp_x-half_w:kp_x+half_w+1].flatten()
            win_Iy = Iy[kp_y-half_w:kp_y+half_w+1, kp_x-half_w:kp_x+half_w+1].flatten()
            win_It = It[kp_y-half_w:kp_y+half_w+1, kp_x-half_w:kp_x+half_w+1].flatten()

            # Build A matrix and b vector
            A = np.column_stack([win_Ix, win_Iy])
            b = -win_It

            # Solve A^T A v = A^T b
            ATA = A.T @ A
            ATb = A.T @ b

            # Check if well-conditioned
            eigenvalues = np.linalg.eigvalsh(ATA)
            if eigenvalues.min() > 1.0:  # Sufficient texture
                flow_vec = np.linalg.solve(ATA, ATb)
                u, v = flow_vec

                # Compare with ground truth
                gt_u = gt_flow[kp_y, kp_x, 0]
                gt_v = gt_flow[kp_y, kp_x, 1]
                error = np.sqrt((u - gt_u)**2 + (v - gt_v)**2)

                flow_results.append({
                    'x': kp_x, 'y': kp_y,
                    'u': u, 'v': v,
                    'gt_u': gt_u, 'gt_v': gt_v,
                    'error': error,
                    'min_eigval': eigenvalues.min(),
                })

    print(f"  Flow computed for: {len(flow_results)} points")

    if flow_results:
        errors = [r['error'] for r in flow_results]
        print(f"\n  End-Point Error:")
        print(f"    Mean EPE: {np.mean(errors):.4f}")
        print(f"    Median EPE: {np.median(errors):.4f}")
        print(f"    Max EPE: {np.max(errors):.4f}")

        # Show some results
        print(f"\n  Sample Results:")
        print(f"    {'Point':>8} | {'Est (u,v)':>14} | {'GT (u,v)':>14} | {'EPE':>6}")
        print(f"    {'-'*50}")
        for r in flow_results[:8]:
            print(f"    ({r['x']:2d},{r['y']:2d}) | "
                  f"({r['u']:+6.2f},{r['v']:+6.2f}) | "
                  f"({r['gt_u']:+6.2f},{r['gt_v']:+6.2f}) | "
                  f"{r['error']:6.3f}")

    return flow_results


# =============================================================================
# Exercise 3: Horn-Schunck Dense Flow
# =============================================================================

def exercise_3_horn_schunck():
    """
    Implement Horn-Schunck dense optical flow with smoothness regularization.

    Minimizes: sum(Ix*u + Iy*v + It)^2 + alpha * (|grad(u)|^2 + |grad(v)|^2)

    Returns:
        (flow_u, flow_v)
    """
    h, w = 60, 80
    frame1, frame2, gt_flow = generate_frame_pair(h, w, "translate")

    print("Horn-Schunck Dense Optical Flow")
    print(f"  Image size: {w}x{h}")
    print("=" * 60)

    # Compute gradients
    Ix = np.zeros((h, w), dtype=np.float64)
    Iy = np.zeros((h, w), dtype=np.float64)
    Ix[:, 1:-1] = (frame1[:, 2:] - frame1[:, :-2]) / 2.0
    Iy[1:-1, :] = (frame1[2:, :] - frame1[:-2, :]) / 2.0
    It = frame2 - frame1

    # Initialize flow
    u = np.zeros((h, w), dtype=np.float64)
    v = np.zeros((h, w), dtype=np.float64)

    alpha = 50.0  # Smoothness weight
    n_iterations = 100

    print(f"  Alpha (smoothness): {alpha}")
    print(f"  Iterations: {n_iterations}")

    errors_over_iter = []

    for iteration in range(n_iterations):
        # Compute local average of flow (Laplacian approximation)
        u_avg = np.zeros_like(u)
        v_avg = np.zeros_like(v)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                u_avg[i, j] = (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1]) / 4.0
                v_avg[i, j] = (v[i-1, j] + v[i+1, j] + v[i, j-1] + v[i, j+1]) / 4.0

        # Update flow
        denom = alpha + Ix**2 + Iy**2
        numerator = Ix * u_avg + Iy * v_avg + It

        u = u_avg - Ix * numerator / denom
        v = v_avg - Iy * numerator / denom

        # Compute error (inside border)
        epe = np.sqrt((u[5:-5, 5:-5] - gt_flow[5:-5, 5:-5, 0])**2 +
                      (v[5:-5, 5:-5] - gt_flow[5:-5, 5:-5, 1])**2)
        mean_epe = epe.mean()
        errors_over_iter.append(mean_epe)

        if (iteration + 1) % 25 == 0:
            print(f"  Iter {iteration+1:>3d}: mean EPE = {mean_epe:.4f}")

    print(f"\n  Convergence:")
    print(f"    Initial EPE: {errors_over_iter[0]:.4f}")
    print(f"    Final EPE:   {errors_over_iter[-1]:.4f}")
    print(f"    Improvement: {errors_over_iter[0] - errors_over_iter[-1]:.4f}")

    # Flow statistics
    flow_mag = np.sqrt(u**2 + v**2)
    inner = flow_mag[5:-5, 5:-5]
    print(f"\n  Estimated Flow:")
    print(f"    Magnitude range: [{inner.min():.3f}, {inner.max():.3f}]")
    print(f"    Mean magnitude: {inner.mean():.3f}")
    print(f"    GT magnitude: {np.sqrt(gt_flow[:,:,0]**2 + gt_flow[:,:,1]**2).mean():.3f}")

    # Alpha sensitivity
    print(f"\n  Smoothness (alpha) Effect:")
    for test_alpha in [1, 10, 50, 200]:
        u_t = np.zeros((h, w), dtype=np.float64)
        v_t = np.zeros((h, w), dtype=np.float64)
        for _ in range(50):
            u_avg_t = np.zeros_like(u_t)
            v_avg_t = np.zeros_like(v_t)
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    u_avg_t[i, j] = (u_t[i-1,j]+u_t[i+1,j]+u_t[i,j-1]+u_t[i,j+1])/4.0
                    v_avg_t[i, j] = (v_t[i-1,j]+v_t[i+1,j]+v_t[i,j-1]+v_t[i,j+1])/4.0
            denom_t = test_alpha + Ix**2 + Iy**2
            num_t = Ix * u_avg_t + Iy * v_avg_t + It
            u_t = u_avg_t - Ix * num_t / denom_t
            v_t = v_avg_t - Iy * num_t / denom_t

        epe_t = np.sqrt((u_t[5:-5,5:-5] - gt_flow[5:-5,5:-5,0])**2 +
                        (v_t[5:-5,5:-5] - gt_flow[5:-5,5:-5,1])**2).mean()
        smoothness = np.sqrt(np.gradient(u_t)[0]**2 + np.gradient(v_t)[0]**2).mean()
        print(f"    alpha={test_alpha:>4}: EPE={epe_t:.4f}, smoothness={smoothness:.4f}")

    return u, v


# =============================================================================
# Exercise 4: Flow Visualization
# =============================================================================

def exercise_4_flow_visualization():
    """
    Implement optical flow visualization using color coding.

    Color coding:
    - Hue = flow direction (angle)
    - Value/Saturation = flow magnitude

    Returns:
        (flow_color_image, flow_arrows)
    """
    h, w = 60, 80

    print("Optical Flow Visualization")
    print(f"  Image size: {w}x{h}")
    print("=" * 60)

    motion_types = ["translate", "rotate", "diverge", "mixed"]
    results = {}

    for motion in motion_types:
        _, _, gt_flow = generate_frame_pair(h, w, motion)

        u = gt_flow[:, :, 0]
        v = gt_flow[:, :, 1]

        # Magnitude and angle
        magnitude = np.sqrt(u**2 + v**2)
        angle = np.arctan2(v, u)  # Range [-pi, pi]

        # Normalize magnitude for visualization
        max_mag = magnitude.max() + 1e-10

        # HSV-like color coding (compute RGB directly)
        # Hue from angle, value from magnitude
        hue = (angle + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
        sat = np.ones_like(hue)
        val = np.minimum(magnitude / max_mag, 1.0)

        # Convert HSV to RGB
        color = np.zeros((h, w, 3), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                h_val = hue[i, j] * 6.0
                hi = int(h_val) % 6
                f = h_val - int(h_val)
                p = val[i, j] * (1 - sat[i, j])
                q = val[i, j] * (1 - f * sat[i, j])
                t_val = val[i, j] * (1 - (1 - f) * sat[i, j])
                v_val = val[i, j]

                if hi == 0:
                    color[i, j] = [v_val, t_val, p]
                elif hi == 1:
                    color[i, j] = [q, v_val, p]
                elif hi == 2:
                    color[i, j] = [p, v_val, t_val]
                elif hi == 3:
                    color[i, j] = [p, q, v_val]
                elif hi == 4:
                    color[i, j] = [t_val, p, v_val]
                else:
                    color[i, j] = [v_val, p, q]

        results[motion] = {
            'color': color,
            'magnitude': magnitude,
            'angle': angle,
        }

        print(f"\n  {motion}:")
        print(f"    Flow magnitude: [{magnitude.min():.3f}, {magnitude.max():.3f}], "
              f"mean={magnitude.mean():.3f}")
        print(f"    Flow angle: [{np.degrees(angle.min()):.0f}, "
              f"{np.degrees(angle.max()):.0f}] degrees")
        print(f"    Color range: [{color.min():.3f}, {color.max():.3f}]")

    # Arrow-based visualization (sample sparse arrows)
    print(f"\n  Arrow Visualization (translate flow):")
    _, _, flow = generate_frame_pair(h, w, "translate")
    arrow_step = 10
    arrows = []
    for i in range(arrow_step, h - arrow_step, arrow_step):
        for j in range(arrow_step, w - arrow_step, arrow_step):
            dx = flow[i, j, 0]
            dy = flow[i, j, 1]
            mag = np.sqrt(dx**2 + dy**2)
            arrows.append((j, i, dx, dy, mag))
            print(f"    ({j:2d},{i:2d}) -> ({dx:+.1f},{dy:+.1f}) |{mag:.2f}|")

    return results


# =============================================================================
# Exercise 5: Motion Segmentation
# =============================================================================

def exercise_5_motion_segmentation():
    """
    Segment moving objects using optical flow.

    Steps:
    1. Compute dense flow
    2. Estimate dominant (background) motion
    3. Identify outlier flow vectors (moving objects)
    4. Generate binary motion mask

    Returns:
        (motion_mask, background_flow)
    """
    h, w = 60, 80
    frame1, frame2, gt_flow = generate_frame_pair(h, w, "mixed")

    print("Motion Segmentation from Optical Flow")
    print(f"  Image size: {w}x{h}")
    print("=" * 60)

    # Use GT flow (in practice, would estimate with LK or Horn-Schunck)
    flow = gt_flow.copy()
    u, v = flow[:, :, 0], flow[:, :, 1]

    # Step 1: Estimate dominant motion (median)
    bg_u = np.median(u)
    bg_v = np.median(v)
    print(f"\n  Estimated background motion:")
    print(f"    u = {bg_u:.2f}, v = {bg_v:.2f}")

    # Step 2: Compute flow deviation from background
    deviation_u = u - bg_u
    deviation_v = v - bg_v
    deviation_mag = np.sqrt(deviation_u**2 + deviation_v**2)

    print(f"\n  Deviation from background:")
    print(f"    Range: [{deviation_mag.min():.3f}, {deviation_mag.max():.3f}]")
    print(f"    Mean: {deviation_mag.mean():.3f}")

    # Step 3: Threshold to get motion mask
    thresholds = [0.5, 1.0, 2.0, 3.0]
    print(f"\n  Threshold Analysis:")
    print(f"    {'Threshold':>10} | {'Moving pixels':>14} | {'Coverage':>8}")
    print(f"    {'-'*40}")

    best_mask = None
    best_iou = 0
    gt_moving = (gt_flow[:, :, 0] != gt_flow[0, 0, 0]) | (gt_flow[:, :, 1] != gt_flow[0, 0, 1])

    for thresh in thresholds:
        motion_mask = deviation_mag > thresh
        n_moving = motion_mask.sum()
        coverage = n_moving / (h * w)

        # Compare with GT moving region
        intersection = np.sum(motion_mask & gt_moving)
        union = np.sum(motion_mask | gt_moving)
        iou = intersection / (union + 1e-10)

        if iou > best_iou:
            best_iou = iou
            best_mask = motion_mask.copy()

        print(f"    {thresh:>10.1f} | {n_moving:>14} | {coverage:>8.3f} | IoU={iou:.3f}")

    # Step 4: Connected component analysis (simple)
    print(f"\n  Best segmentation (IoU={best_iou:.3f}):")
    if best_mask is not None:
        ys, xs = np.where(best_mask)
        if len(ys) > 0:
            bbox = (xs.min(), ys.min(), xs.max(), ys.max())
            centroid = (xs.mean(), ys.mean())
            area = best_mask.sum()

            print(f"    Moving region area: {area} pixels")
            print(f"    Centroid: ({centroid[0]:.1f}, {centroid[1]:.1f})")
            print(f"    BBox: ({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})")

            # Motion of the moving object
            obj_u = u[best_mask].mean()
            obj_v = v[best_mask].mean()
            obj_speed = np.sqrt(obj_u**2 + obj_v**2)
            print(f"    Object motion: ({obj_u:.2f}, {obj_v:.2f}), speed={obj_speed:.2f}")

            # Relative motion
            rel_u = obj_u - bg_u
            rel_v = obj_v - bg_v
            rel_speed = np.sqrt(rel_u**2 + rel_v**2)
            print(f"    Relative motion: ({rel_u:.2f}, {rel_v:.2f}), speed={rel_speed:.2f}")

    return best_mask, (bg_u, bg_v)


# =============================================================================
# Exercise 6: Video Stabilization
# =============================================================================

def exercise_6_video_stabilization():
    """
    Implement basic video stabilization using optical flow.

    Steps:
    1. Generate shaky video (random camera jitter)
    2. Estimate global motion between frames
    3. Compute cumulative trajectory
    4. Smooth trajectory
    5. Apply corrective transformation

    Returns:
        (original_trajectory, smoothed_trajectory)
    """
    np.random.seed(42)
    n_frames = 30
    h, w = 40, 50

    print("Video Stabilization via Optical Flow")
    print(f"  Frames: {n_frames}, Size: {w}x{h}")
    print("=" * 60)

    # Generate camera jitter (simulated shake)
    jitter_x = np.cumsum(np.random.randn(n_frames) * 1.5)
    jitter_y = np.cumsum(np.random.randn(n_frames) * 1.0)

    # Ground truth global motion (what we want to remove)
    gt_dx = np.diff(jitter_x)
    gt_dy = np.diff(jitter_y)

    print(f"\n  Camera Jitter:")
    print(f"    X range: [{jitter_x.min():.2f}, {jitter_x.max():.2f}]")
    print(f"    Y range: [{jitter_y.min():.2f}, {jitter_y.max():.2f}]")
    print(f"    Max displacement: {max(abs(jitter_x.max()-jitter_x.min()), abs(jitter_y.max()-jitter_y.min())):.2f}")

    # Estimate global motion from optical flow (simulated: use GT + noise)
    est_dx = gt_dx + np.random.randn(len(gt_dx)) * 0.2
    est_dy = gt_dy + np.random.randn(len(gt_dy)) * 0.2

    # Cumulative trajectory
    cum_x = np.concatenate([[0], np.cumsum(est_dx)])
    cum_y = np.concatenate([[0], np.cumsum(est_dy)])

    print(f"\n  Estimated Trajectory:")
    print(f"    X range: [{cum_x.min():.2f}, {cum_x.max():.2f}]")
    print(f"    Y range: [{cum_y.min():.2f}, {cum_y.max():.2f}]")

    # Smooth trajectory (moving average)
    window_sizes = [3, 5, 9, 15]
    smoothed_results = {}

    for ws in window_sizes:
        half = ws // 2
        smooth_x = np.copy(cum_x)
        smooth_y = np.copy(cum_y)

        for i in range(half, n_frames - half):
            smooth_x[i] = np.mean(cum_x[i-half:i+half+1])
            smooth_y[i] = np.mean(cum_y[i-half:i+half+1])

        # Correction = original - smoothed
        correction_x = cum_x - smooth_x
        correction_y = cum_y - smooth_y

        # Residual jitter after correction
        residual_x = jitter_x - correction_x
        residual_y = jitter_y - correction_y
        residual_jitter = np.sqrt(
            np.diff(residual_x)**2 + np.diff(residual_y)**2
        ).mean()
        original_jitter = np.sqrt(gt_dx**2 + gt_dy**2).mean()

        reduction = 1 - residual_jitter / (original_jitter + 1e-10)

        smoothed_results[ws] = {
            'smooth_x': smooth_x,
            'smooth_y': smooth_y,
            'residual_jitter': residual_jitter,
            'reduction': reduction,
        }

    print(f"\n  Smoothing Analysis:")
    print(f"    Original jitter: {np.sqrt(gt_dx**2 + gt_dy**2).mean():.4f} px/frame")
    print(f"    {'Window':>8} | {'Residual':>10} | {'Reduction':>10}")
    print(f"    {'-'*35}")

    for ws in window_sizes:
        r = smoothed_results[ws]
        print(f"    {ws:>8} | {r['residual_jitter']:>10.4f} | "
              f"{r['reduction']:>9.1%}")

    # Show trajectory at a few time steps
    best_ws = min(smoothed_results, key=lambda k: smoothed_results[k]['residual_jitter'])
    best = smoothed_results[best_ws]
    print(f"\n  Best window size: {best_ws}")
    print(f"  Trajectory comparison (best smoothing):")
    print(f"    {'Frame':>6} | {'Original X':>10} | {'Smoothed X':>10} | {'Correction':>10}")
    print(f"    {'-'*45}")
    for t in range(0, n_frames, 5):
        print(f"    {t:>6} | {cum_x[t]:>10.2f} | "
              f"{best['smooth_x'][t]:>10.2f} | "
              f"{cum_x[t] - best['smooth_x'][t]:>10.2f}")

    return {'original': (cum_x, cum_y),
            'smoothed': (best['smooth_x'], best['smooth_y'])}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Optical Flow Equation")
    exercise_1_flow_equation()

    print("\n>>> Exercise 2: Lucas-Kanade Sparse Flow")
    exercise_2_lucas_kanade()

    print("\n>>> Exercise 3: Horn-Schunck Dense Flow")
    exercise_3_horn_schunck()

    print("\n>>> Exercise 4: Flow Visualization")
    exercise_4_flow_visualization()

    print("\n>>> Exercise 5: Motion Segmentation")
    exercise_5_motion_segmentation()

    print("\n>>> Exercise 6: Video Stabilization")
    exercise_6_video_stabilization()

    print("\nAll exercises completed successfully.")
