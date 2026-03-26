"""
Exercise Solutions for Lesson 29: 3D Gaussian Splatting
Computer Vision - Gaussian Primitives, Differentiable Rasterization, Density Control

Topics covered:
- 3D Gaussian representation (position, covariance, color, opacity)
- Quaternion to rotation matrix conversion
- 2D Gaussian projection and evaluation
- Alpha compositing for splatting
- Adaptive density control (split, clone, prune)
- Quality metrics (PSNR, SSIM)
"""

import numpy as np


# =============================================================================
# Helper: Gaussian utilities
# =============================================================================

def quaternion_to_rotation(q):
    """Convert unit quaternion (w, x, y, z) to 3x3 rotation matrix."""
    q = q / (np.linalg.norm(q) + 1e-10)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


def build_covariance(scale, rotation_quat):
    """Build 3D covariance matrix from scale and rotation quaternion."""
    S = np.diag(scale)
    R = quaternion_to_rotation(rotation_quat)
    L = R @ S
    return L @ L.T


# =============================================================================
# Exercise 1: 3D Gaussian Representation
# =============================================================================

def exercise_1_gaussian_representation():
    """
    Explore the 3D Gaussian primitive representation.

    Each Gaussian has:
    - Position (mean): 3D center
    - Covariance: 3x3 matrix (from scale + rotation)
    - Color: RGB (or spherical harmonics)
    - Opacity: alpha in [0, 1]

    Returns:
        list of Gaussian parameter dicts
    """
    np.random.seed(42)
    n_gaussians = 8

    print("3D Gaussian Representation")
    print(f"  Gaussians: {n_gaussians}")
    print("=" * 60)

    gaussians = []
    for i in range(n_gaussians):
        # Random position
        position = np.random.uniform(-2, 2, 3)

        # Scale (log-space, then exponentiate)
        log_scale = np.random.uniform(-2, 0, 3)
        scale = np.exp(log_scale)

        # Rotation (random quaternion)
        q = np.random.randn(4)
        q /= np.linalg.norm(q)

        # Color (RGB)
        color = np.random.uniform(0.1, 0.9, 3)

        # Opacity (logit-space, then sigmoid)
        logit_opacity = np.random.uniform(-2, 2)
        opacity = 1.0 / (1.0 + np.exp(-logit_opacity))

        # Build covariance
        cov = build_covariance(scale, q)

        # Eigendecomposition for analysis
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        g = {
            'position': position,
            'scale': scale,
            'rotation': q,
            'color': color,
            'opacity': opacity,
            'covariance': cov,
            'eigenvalues': eigenvalues,
        }
        gaussians.append(g)

        print(f"\n  Gaussian {i}:")
        print(f"    Position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
        print(f"    Scale: ({scale[0]:.3f}, {scale[1]:.3f}, {scale[2]:.3f})")
        print(f"    Color: ({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
        print(f"    Opacity: {opacity:.3f}")
        print(f"    Cov eigenvalues: ({eigenvalues[0]:.4f}, "
              f"{eigenvalues[1]:.4f}, {eigenvalues[2]:.4f})")
        vol = np.prod(scale) * (4/3) * np.pi
        print(f"    Approx volume: {vol:.4f}")

    # Memory analysis
    params_per_gaussian = 3 + 3 + 4 + 3 + 1  # pos + scale + quat + color + opacity
    # With SH: 3 + 3 + 4 + 48 + 1 = 59
    params_sh = 3 + 3 + 4 + 48 + 1

    print(f"\n  Memory Analysis:")
    print(f"    Params per Gaussian (RGB): {params_per_gaussian}")
    print(f"    Params per Gaussian (SH degree 3): {params_sh}")
    for n in [100000, 1000000, 5000000]:
        mem_rgb = n * params_per_gaussian * 4 / (1024 * 1024)
        mem_sh = n * params_sh * 4 / (1024 * 1024)
        count_str = f"{n:>10,}"
        print(f"    {count_str} Gaussians: {mem_rgb:.1f} MB (RGB), "
              f"{mem_sh:.1f} MB (SH)")

    return gaussians


# =============================================================================
# Exercise 2: 2D Gaussian Projection
# =============================================================================

def exercise_2_gaussian_projection():
    """
    Project 3D Gaussians to 2D screen space.

    Steps:
    1. Apply world-to-camera transform
    2. Project 3D covariance to 2D using Jacobian
    3. Evaluate 2D Gaussian at pixel locations

    Returns:
        (means_2d, covs_2d)
    """
    np.random.seed(42)
    h, w = 60, 80

    # Camera
    f = 200.0
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)

    # Camera extrinsics (at z=5 looking at origin)
    R_cam = np.eye(3)
    t_cam = np.array([0, 0, 5], dtype=np.float64)

    print("2D Gaussian Projection")
    print(f"  Image: {w}x{h}")
    print(f"  Focal length: {f:.0f}")
    print("=" * 60)

    # Create some 3D Gaussians
    positions = np.array([
        [0, 0, 0],
        [-1, 0.5, 0.5],
        [1.2, -0.3, -0.5],
        [0.5, 0.8, 1.0],
    ], dtype=np.float64)

    n_gaussians = len(positions)
    scales = np.array([
        [0.3, 0.2, 0.15],
        [0.15, 0.4, 0.1],
        [0.2, 0.2, 0.3],
        [0.1, 0.1, 0.1],
    ], dtype=np.float64)

    means_2d = []
    covs_2d = []
    depths = []

    for i in range(n_gaussians):
        # Transform to camera space
        p_cam = R_cam @ positions[i] + t_cam
        depth = p_cam[2]
        depths.append(depth)

        if depth <= 0:
            means_2d.append(None)
            covs_2d.append(None)
            continue

        # Project center to 2D
        px = f * p_cam[0] / p_cam[2] + w / 2
        py = f * p_cam[1] / p_cam[2] + h / 2
        means_2d.append(np.array([px, py]))

        # Jacobian of perspective projection
        J = np.array([
            [f / p_cam[2], 0, -f * p_cam[0] / (p_cam[2]**2)],
            [0, f / p_cam[2], -f * p_cam[1] / (p_cam[2]**2)],
        ], dtype=np.float64)

        # 3D covariance
        q = np.array([1, 0, 0, 0], dtype=np.float64)  # Identity rotation
        cov_3d = build_covariance(scales[i], q)

        # Rotate to camera frame
        cov_cam = R_cam @ cov_3d @ R_cam.T

        # Project to 2D: Sigma_2D = J @ Sigma_cam @ J^T
        cov_2d = J @ cov_cam @ J.T
        covs_2d.append(cov_2d)

        # Eigendecomposition for analysis
        eigvals, eigvecs = np.linalg.eigh(cov_2d)

        print(f"\n  Gaussian {i}:")
        print(f"    3D pos: ({positions[i, 0]:.2f}, {positions[i, 1]:.2f}, "
              f"{positions[i, 2]:.2f})")
        print(f"    Depth: {depth:.2f}")
        print(f"    2D center: ({px:.1f}, {py:.1f})")
        print(f"    2D cov eigenvalues: ({eigvals[0]:.4f}, {eigvals[1]:.4f})")
        print(f"    2D extent (3-sigma): ({3*np.sqrt(eigvals[0]):.1f}, "
              f"{3*np.sqrt(eigvals[1]):.1f}) pixels")

    # Evaluate a Gaussian at pixels
    print(f"\n  Evaluation of Gaussian 0 at sample pixels:")
    if means_2d[0] is not None and covs_2d[0] is not None:
        mu = means_2d[0]
        cov = covs_2d[0]
        cov_inv = np.linalg.inv(cov)

        test_offsets = [(0, 0), (5, 0), (0, 5), (10, 10), (20, 20)]
        for dx, dy in test_offsets:
            pixel = mu + np.array([dx, dy], dtype=np.float64)
            diff = pixel - mu
            exponent = -0.5 * diff @ cov_inv @ diff
            value = np.exp(exponent)
            print(f"    Offset ({dx:+3d}, {dy:+3d}): value={value:.6f}")

    return means_2d, covs_2d


# =============================================================================
# Exercise 3: Alpha Compositing (Splatting)
# =============================================================================

def exercise_3_alpha_compositing():
    """
    Implement front-to-back alpha compositing for Gaussian splatting.

    C(pixel) = sum_i c_i * alpha_i * T_i
    T_i = prod_{j<i} (1 - alpha_j)

    Returns:
        rendered_image (h, w, 3)
    """
    np.random.seed(42)
    h, w = 50, 60

    print("Alpha Compositing (Splatting)")
    print(f"  Image: {w}x{h}")
    print("=" * 60)

    # Define 2D Gaussians (already projected)
    gaussians_2d = [
        {'mean': np.array([30, 25]), 'cov': np.array([[40, 5], [5, 30]]),
         'color': np.array([0.8, 0.2, 0.1]), 'opacity': 0.9, 'depth': 3.0},
        {'mean': np.array([35, 28]), 'cov': np.array([[25, -3], [-3, 35]]),
         'color': np.array([0.1, 0.7, 0.2]), 'opacity': 0.8, 'depth': 4.0},
        {'mean': np.array([20, 35]), 'cov': np.array([[50, 0], [0, 20]]),
         'color': np.array([0.2, 0.3, 0.9]), 'opacity': 0.7, 'depth': 5.0},
        {'mean': np.array([45, 20]), 'cov': np.array([[15, 0], [0, 15]]),
         'color': np.array([0.9, 0.9, 0.1]), 'opacity': 0.95, 'depth': 2.5},
        {'mean': np.array([25, 15]), 'cov': np.array([[30, 10], [10, 25]]),
         'color': np.array([0.6, 0.1, 0.8]), 'opacity': 0.6, 'depth': 3.5},
    ]

    n_gaussians = len(gaussians_2d)

    # Sort by depth (front to back)
    sorted_gaussians = sorted(gaussians_2d, key=lambda g: g['depth'])

    print(f"\n  Gaussians (sorted by depth):")
    for i, g in enumerate(sorted_gaussians):
        print(f"    [{i}] depth={g['depth']:.1f}, "
              f"color=({g['color'][0]:.1f}, {g['color'][1]:.1f}, {g['color'][2]:.1f}), "
              f"opacity={g['opacity']:.2f}")

    # Render
    image = np.zeros((h, w, 3), dtype=np.float64)
    accumulated_alpha = np.zeros((h, w), dtype=np.float64)

    pixel_coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1)
    # pixel_coords shape: (h, w, 2) with (x, y)

    for i, g in enumerate(sorted_gaussians):
        mu = g['mean']
        cov = g['cov']
        color = g['color']
        base_opacity = g['opacity']

        cov_inv = np.linalg.inv(cov)

        # Evaluate Gaussian at all pixels
        diff = pixel_coords.astype(np.float64) - mu  # (h, w, 2)
        # Mahalanobis distance: diff @ cov_inv @ diff^T per pixel
        exponent = np.zeros((h, w), dtype=np.float64)
        for row in range(h):
            for col in range(w):
                d = diff[row, col]
                exponent[row, col] = -0.5 * d @ cov_inv @ d

        gaussian_val = np.exp(exponent)
        alpha = base_opacity * gaussian_val

        # Alpha compositing
        weight = alpha * (1.0 - accumulated_alpha)
        for c in range(3):
            image[:, :, c] += weight * color[c]
        accumulated_alpha += weight

    # Background (white)
    for c in range(3):
        image[:, :, c] += (1.0 - accumulated_alpha) * 1.0

    image = np.clip(image, 0, 1)

    # Statistics
    print(f"\n  Rendering Statistics:")
    print(f"    Color range: [{image.min():.4f}, {image.max():.4f}]")
    print(f"    Mean color: ({image[:,:,0].mean():.3f}, "
          f"{image[:,:,1].mean():.3f}, {image[:,:,2].mean():.3f})")
    print(f"    Mean opacity: {accumulated_alpha.mean():.4f}")
    print(f"    Fully opaque pixels: {(accumulated_alpha > 0.99).sum()}")
    print(f"    Background pixels: {(accumulated_alpha < 0.01).sum()}")

    # Per-Gaussian contribution
    print(f"\n  Per-Gaussian Contribution:")
    for i, g in enumerate(sorted_gaussians):
        mu = g['mean']
        cov = g['cov']
        eigvals = np.linalg.eigvalsh(cov)
        footprint = np.pi * 3 * np.sqrt(eigvals[0]) * 3 * np.sqrt(eigvals[1])
        pct = min(footprint / (h * w) * 100, 100)
        print(f"    Gaussian {i}: footprint ~{footprint:.0f} px ({pct:.1f}%)")

    return image


# =============================================================================
# Exercise 4: Adaptive Density Control
# =============================================================================

def exercise_4_density_control():
    """
    Simulate adaptive density control for Gaussian splatting.

    Operations:
    1. Clone: small Gaussian with large gradient -> duplicate nearby
    2. Split: large Gaussian with large gradient -> split into two
    3. Prune: low opacity or too large -> remove

    Returns:
        (initial_count, final_count, gaussians)
    """
    np.random.seed(42)

    print("Adaptive Density Control")
    print("=" * 60)

    # Initialize Gaussians
    n_initial = 30
    positions = np.random.uniform(-3, 3, (n_initial, 3))
    scales = np.exp(np.random.uniform(-2, 0, (n_initial, 3)))
    opacities = np.random.uniform(0.1, 1.0, n_initial)
    colors = np.random.uniform(0.1, 0.9, (n_initial, 3))
    # Simulate gradients (position gradient magnitude)
    grad_magnitudes = np.random.exponential(0.5, n_initial)

    print(f"\n  Initial state: {n_initial} Gaussians")
    print(f"    Scale range: [{scales.min():.4f}, {scales.max():.4f}]")
    print(f"    Opacity range: [{opacities.min():.3f}, {opacities.max():.3f}]")
    print(f"    Gradient range: [{grad_magnitudes.min():.4f}, "
          f"{grad_magnitudes.max():.4f}]")

    # Thresholds
    grad_threshold = 0.5
    scale_threshold = 0.5  # Max scale for cloning
    opacity_threshold = 0.05
    max_scale = 2.0

    # Lists for new state
    new_positions = list(positions)
    new_scales = list(scales)
    new_opacities = list(opacities)
    new_colors = list(colors)

    cloned = 0
    split_count = 0
    pruned = 0

    indices_to_remove = set()

    for i in range(n_initial):
        mean_scale = scales[i].mean()

        if grad_magnitudes[i] > grad_threshold:
            if mean_scale < scale_threshold:
                # CLONE: small Gaussian with large gradient
                offset = np.random.randn(3) * scales[i] * 0.5
                new_positions.append(positions[i] + offset)
                new_scales.append(scales[i].copy())
                new_opacities.append(opacities[i])
                new_colors.append(colors[i].copy())
                cloned += 1
            else:
                # SPLIT: large Gaussian with large gradient -> two smaller
                offset = np.random.randn(3) * scales[i] * 0.3
                half_scale = scales[i] * 0.7

                indices_to_remove.add(i)
                # Add two new smaller Gaussians
                for sign in [1, -1]:
                    new_positions.append(positions[i] + sign * offset)
                    new_scales.append(half_scale.copy())
                    new_opacities.append(opacities[i])
                    new_colors.append(colors[i].copy())
                split_count += 1

        # PRUNE: low opacity
        if opacities[i] < opacity_threshold:
            indices_to_remove.add(i)
            pruned += 1

        # PRUNE: too large
        if mean_scale > max_scale:
            indices_to_remove.add(i)
            pruned += 1

    # Remove pruned/split originals
    final_positions = []
    final_scales = []
    final_opacities = []
    final_colors = []

    for i in range(len(new_positions)):
        if i < n_initial and i in indices_to_remove:
            continue
        final_positions.append(new_positions[i])
        final_scales.append(new_scales[i])
        final_opacities.append(new_opacities[i])
        final_colors.append(new_colors[i])

    n_final = len(final_positions)

    print(f"\n  Density Control Operations:")
    print(f"    Cloned: {cloned}")
    print(f"    Split: {split_count}")
    print(f"    Pruned: {pruned}")
    print(f"\n  Final state: {n_final} Gaussians")
    print(f"    Change: {n_initial} -> {n_final} "
          f"({n_final - n_initial:+d})")

    final_scales_arr = np.array(final_scales)
    final_opacities_arr = np.array(final_opacities)

    print(f"    Scale range: [{final_scales_arr.min():.4f}, "
          f"{final_scales_arr.max():.4f}]")
    print(f"    Opacity range: [{final_opacities_arr.min():.3f}, "
          f"{final_opacities_arr.max():.3f}]")

    # Simulate multiple iterations
    print(f"\n  Density Control Over Training:")
    counts = [n_initial]
    current = n_initial
    for iteration in range(10):
        # Simulate growth and pruning
        new = int(current * np.random.uniform(0.05, 0.15))
        removed = int(current * np.random.uniform(0.02, 0.08))
        current = current + new - removed
        counts.append(current)

    print(f"    {'Iter':>5} | {'Count':>8}")
    print(f"    {'-'*18}")
    for i, c in enumerate(counts):
        bar = "#" * (c // 5)
        print(f"    {i:>5} | {c:>8} {bar}")

    return n_initial, n_final, {'positions': final_positions, 'scales': final_scales}


# =============================================================================
# Exercise 5: Quality Metrics (PSNR, SSIM)
# =============================================================================

def exercise_5_quality_metrics():
    """
    Compute image quality metrics for Gaussian splatting evaluation.

    Metrics:
    1. PSNR (Peak Signal-to-Noise Ratio)
    2. SSIM (Structural Similarity Index)
    3. L1 loss

    Returns:
        dict of metric_name -> value
    """
    np.random.seed(42)
    h, w = 40, 50

    print("Quality Metrics for 3DGS Evaluation")
    print(f"  Image: {w}x{h}")
    print("=" * 60)

    # Generate reference image (smooth gradient with shapes)
    ref_image = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            ref_image[i, j, 0] = 0.3 + 0.4 * (j / w)
            ref_image[i, j, 1] = 0.2 + 0.3 * (i / h)
            ref_image[i, j, 2] = 0.5 - 0.2 * ((i + j) / (h + w))

    # Add a bright region
    yy, xx = np.ogrid[:h, :w]
    circle = ((xx - 25)**2 + (yy - 20)**2) <= 8**2
    ref_image[circle] = [0.9, 0.8, 0.2]

    def compute_psnr(pred, ref):
        """Peak Signal-to-Noise Ratio in dB."""
        mse = np.mean((pred - ref) ** 2)
        if mse < 1e-10:
            return float('inf')
        return -10 * np.log10(mse)

    def compute_ssim(pred, ref, window_size=7):
        """Structural Similarity Index (simplified)."""
        C1 = (0.01) ** 2  # Stabilization constant
        C2 = (0.03) ** 2

        ssim_map = np.zeros((h, w), dtype=np.float64)
        half = window_size // 2
        count = 0
        total_ssim = 0.0

        for i in range(half, h - half):
            for j in range(half, w - half):
                p_win = pred[i-half:i+half+1, j-half:j+half+1].mean(axis=2)
                r_win = ref[i-half:i+half+1, j-half:j+half+1].mean(axis=2)

                mu_p = p_win.mean()
                mu_r = r_win.mean()
                sigma_p = p_win.std()
                sigma_r = r_win.std()
                sigma_pr = np.mean((p_win - mu_p) * (r_win - mu_r))

                num = (2 * mu_p * mu_r + C1) * (2 * sigma_pr + C2)
                den = (mu_p**2 + mu_r**2 + C1) * (sigma_p**2 + sigma_r**2 + C2)
                ssim_val = num / den

                ssim_map[i, j] = ssim_val
                total_ssim += ssim_val
                count += 1

        return total_ssim / count if count > 0 else 0.0

    # Test with different noise levels (simulating different training stages)
    noise_levels = [0.3, 0.1, 0.05, 0.02, 0.01, 0.005]
    results = {}

    print(f"\n  {'Noise':>8} | {'PSNR':>8} | {'SSIM':>8} | {'L1':>8} | {'MSE':>8}")
    print(f"  {'-'*50}")

    for noise in noise_levels:
        pred = ref_image + np.random.randn(h, w, 3) * noise
        pred = np.clip(pred, 0, 1)

        psnr = compute_psnr(pred, ref_image)
        ssim = compute_ssim(pred, ref_image)
        l1 = np.mean(np.abs(pred - ref_image))
        mse = np.mean((pred - ref_image) ** 2)

        results[noise] = {
            'PSNR': psnr, 'SSIM': ssim, 'L1': l1, 'MSE': mse
        }
        print(f"  {noise:>8.3f} | {psnr:>8.2f} | {ssim:>8.4f} | "
              f"{l1:>8.5f} | {mse:>8.6f}")

    # Compare with typical NeRF vs 3DGS results
    print(f"\n  Typical Benchmark Comparison:")
    print(f"    NeRF:   PSNR ~31-33 dB, SSIM ~0.95")
    print(f"    3DGS:   PSNR ~33-35 dB, SSIM ~0.96")
    print(f"    Speed:  NeRF ~1 FPS, 3DGS ~100+ FPS")

    return results


# =============================================================================
# Exercise 6: NeRF vs 3DGS Comparison
# =============================================================================

def exercise_6_nerf_vs_3dgs():
    """
    Compare NeRF (implicit) and 3DGS (explicit) representations.

    Analyzes:
    1. Rendering cost per pixel
    2. Memory usage scaling
    3. Training convergence simulation
    4. Editability comparison

    Returns:
        comparison dict
    """
    np.random.seed(42)

    print("NeRF vs 3D Gaussian Splatting Comparison")
    print("=" * 60)

    # 1. Rendering cost analysis
    print(f"\n  [1] Rendering Cost Per Pixel:")

    resolutions = [(640, 480), (1280, 720), (1920, 1080)]
    nerf_samples = 192  # 64 coarse + 128 fine
    nerf_mlp_flops = 256 * 256 * 8  # 8 layers, 256 hidden

    for res_w, res_h in resolutions:
        n_pixels = res_w * res_h
        nerf_flops = n_pixels * nerf_samples * nerf_mlp_flops
        nerf_gflops = nerf_flops / 1e9

        # 3DGS: average ~10 Gaussians per pixel (after culling)
        avg_gaussians_per_pixel = 10
        ops_per_gaussian = 50  # 2D eval + alpha blend
        gs_flops = n_pixels * avg_gaussians_per_pixel * ops_per_gaussian
        gs_gflops = gs_flops / 1e9

        speedup = nerf_gflops / gs_gflops if gs_gflops > 0 else 0
        print(f"    {res_w}x{res_h}: NeRF={nerf_gflops:.1f} GFLOP, "
              f"3DGS={gs_gflops:.2f} GFLOP, speedup={speedup:.0f}x")

    # 2. Memory usage
    print(f"\n  [2] Memory Usage:")
    print(f"    {'Component':>20} | {'NeRF':>12} | {'3DGS (1M pts)':>14}")
    print(f"    {'-'*52}")

    nerf_params = 256 * 256 * 8 + 256 * 63 + 256 * 27  # Approx
    nerf_mem = nerf_params * 4 / (1024 * 1024)

    n_points = 1000000
    gs_params = n_points * 59  # pos + scale + quat + SH + opacity
    gs_mem = gs_params * 4 / (1024 * 1024)

    print(f"    {'Model weights':>20} | {nerf_mem:>10.1f} MB | {gs_mem:>12.1f} MB")
    print(f"    {'Parameters':>20} | {nerf_params:>10,} | {gs_params:>12,}")

    # 3. Training convergence simulation
    print(f"\n  [3] Training Convergence (simulated):")
    iterations = np.arange(0, 30001, 5000)

    print(f"    {'Iteration':>10} | {'NeRF PSNR':>10} | {'3DGS PSNR':>10}")
    print(f"    {'-'*35}")

    for it in iterations:
        # NeRF: slower convergence
        nerf_psnr = 20 + 13 * (1 - np.exp(-it / 50000))
        # 3DGS: faster convergence
        gs_psnr = 22 + 12 * (1 - np.exp(-it / 10000))

        print(f"    {it:>10} | {nerf_psnr:>10.2f} | {gs_psnr:>10.2f}")

    # 4. Editability
    print(f"\n  [4] Editability Comparison:")
    operations = [
        ("Delete object", "Retrain or inpaint", "Remove Gaussians in region"),
        ("Move object", "Not directly supported", "Translate Gaussian positions"),
        ("Change color", "Requires fine-tuning", "Modify SH coefficients"),
        ("Add object", "Retrain from scratch", "Add new Gaussians"),
        ("Change lighting", "Need relighting model", "Adjust SH coefficients"),
    ]

    print(f"    {'Operation':>18} | {'NeRF':>25} | {'3DGS':>28}")
    print(f"    {'-'*75}")
    for op, nerf_approach, gs_approach in operations:
        print(f"    {op:>18} | {nerf_approach:>25} | {gs_approach:>28}")

    comparison = {
        'rendering_speedup': '50-200x',
        'training_speedup': '5-20x',
        'memory_ratio': gs_mem / nerf_mem if nerf_mem > 0 else 0,
        'quality': 'comparable (PSNR within 1-2 dB)',
        'editability': '3DGS much easier',
    }

    return comparison


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: 3D Gaussian Representation")
    exercise_1_gaussian_representation()

    print("\n>>> Exercise 2: 2D Gaussian Projection")
    exercise_2_gaussian_projection()

    print("\n>>> Exercise 3: Alpha Compositing (Splatting)")
    exercise_3_alpha_compositing()

    print("\n>>> Exercise 4: Adaptive Density Control")
    exercise_4_density_control()

    print("\n>>> Exercise 5: Quality Metrics (PSNR, SSIM)")
    exercise_5_quality_metrics()

    print("\n>>> Exercise 6: NeRF vs 3DGS Comparison")
    exercise_6_nerf_vs_3dgs()

    print("\nAll exercises completed successfully.")
