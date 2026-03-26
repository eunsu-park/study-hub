"""
Exercise Solutions for Lesson 28: Neural Radiance Fields (NeRF)
Computer Vision - Positional Encoding, Volume Rendering, Ray Marching, NeRF MLP

Topics covered:
- Positional encoding with sinusoidal functions
- Ray generation from camera parameters
- Volumetric rendering (alpha compositing along rays)
- NeRF MLP forward pass simulation
- Hierarchical sampling (coarse-to-fine)
- Depth and normal extraction from density fields
"""

import numpy as np


# =============================================================================
# Helper: Camera and ray utilities
# =============================================================================

def make_camera(f=200.0, cx=64, cy=48, h=96, w=128):
    """Create camera intrinsics matrix."""
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1],
    ], dtype=np.float64)
    return K, h, w


def rotation_y(angle_deg):
    """Create rotation matrix about Y axis."""
    a = np.radians(angle_deg)
    return np.array([
        [np.cos(a), 0, np.sin(a)],
        [0, 1, 0],
        [-np.sin(a), 0, np.cos(a)],
    ], dtype=np.float64)


# =============================================================================
# Exercise 1: Positional Encoding
# =============================================================================

def exercise_1_positional_encoding():
    """
    Implement sinusoidal positional encoding for NeRF.

    gamma(p) = [p, sin(2^0 * pi * p), cos(2^0 * pi * p), ...,
                   sin(2^(L-1) * pi * p), cos(2^(L-1) * pi * p)]

    Steps:
    1. Encode 3D positions with L=10
    2. Encode viewing directions with L=4
    3. Analyze frequency coverage
    4. Compare with and without positional encoding

    Returns:
        (pos_encoded, dir_encoded)
    """
    np.random.seed(42)

    print("Positional Encoding for NeRF")
    print("=" * 60)

    def positional_encoding(x, L=10):
        """
        Map input coordinates to higher-dimensional space.
        Input: (N, D) -> Output: (N, D * (1 + 2L))
        """
        encodings = [x]
        for i in range(L):
            freq = 2.0 ** i * np.pi
            encodings.append(np.sin(freq * x))
            encodings.append(np.cos(freq * x))
        return np.concatenate(encodings, axis=-1)

    # Sample 3D positions
    n_points = 20
    positions = np.random.uniform(-1, 1, (n_points, 3))

    # Sample viewing directions (unit vectors)
    theta = np.random.uniform(0, np.pi, n_points)
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    directions = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])

    # Encode positions (L=10)
    L_pos = 10
    pos_encoded = positional_encoding(positions, L=L_pos)
    pos_dim = 3 * (1 + 2 * L_pos)

    # Encode directions (L=4)
    L_dir = 4
    dir_encoded = positional_encoding(directions, L=L_dir)
    dir_dim = 3 * (1 + 2 * L_dir)

    print(f"\n  Position Encoding:")
    print(f"    Input dim: 3 -> Output dim: {pos_dim}")
    print(f"    L = {L_pos} (frequencies: 2^0 to 2^{L_pos-1})")
    print(f"    Encoded shape: {pos_encoded.shape}")
    print(f"    Value range: [{pos_encoded.min():.3f}, {pos_encoded.max():.3f}]")

    print(f"\n  Direction Encoding:")
    print(f"    Input dim: 3 -> Output dim: {dir_dim}")
    print(f"    L = {L_dir} (frequencies: 2^0 to 2^{L_dir-1})")
    print(f"    Encoded shape: {dir_encoded.shape}")

    # Frequency analysis
    print(f"\n  Frequency Coverage:")
    for i in range(L_pos):
        freq = 2.0 ** i * np.pi
        wavelength = 2.0 / (2.0 ** i)
        print(f"    L={i}: freq={freq:.2f}, wavelength={wavelength:.4f}")

    # Distinctiveness: how well does PE separate nearby points?
    print(f"\n  Distinctiveness Analysis:")
    p1 = np.array([[0.1, 0.2, 0.3]])
    deltas = [0.001, 0.01, 0.1, 0.5]

    for delta in deltas:
        p2 = p1 + delta

        # Without PE
        raw_dist = np.linalg.norm(p1 - p2)

        # With PE
        pe1 = positional_encoding(p1, L=L_pos)
        pe2 = positional_encoding(p2, L=L_pos)
        pe_dist = np.linalg.norm(pe1 - pe2)

        ratio = pe_dist / (raw_dist + 1e-10)
        print(f"    delta={delta:.3f}: raw_dist={raw_dist:.4f}, "
              f"PE_dist={pe_dist:.4f}, amplification={ratio:.1f}x")

    return pos_encoded, dir_encoded


# =============================================================================
# Exercise 2: Ray Generation
# =============================================================================

def exercise_2_ray_generation():
    """
    Generate camera rays for NeRF rendering.

    Steps:
    1. Define camera intrinsics and extrinsics (pose)
    2. Generate rays for each pixel (origin + direction)
    3. Verify ray properties (unit direction, correct origin)
    4. Sample points along rays

    Returns:
        (rays_o, rays_d, sample_points)
    """
    np.random.seed(42)
    K, h, w = make_camera()

    # Camera pose: looking at origin from distance 4
    R = rotation_y(30)
    t = np.array([0, 0, 4], dtype=np.float64)
    camera_pos = -R.T @ t  # Camera position in world

    print("Ray Generation for NeRF")
    print(f"  Image: {w}x{h}")
    print(f"  Focal length: {K[0, 0]:.0f}")
    print(f"  Camera position: ({camera_pos[0]:.2f}, "
          f"{camera_pos[1]:.2f}, {camera_pos[2]:.2f})")
    print("=" * 60)

    # Generate rays for every pixel
    rays_o = np.zeros((h, w, 3), dtype=np.float64)
    rays_d = np.zeros((h, w, 3), dtype=np.float64)

    f_x, f_y = K[0, 0], K[1, 1]
    c_x, c_y = K[0, 2], K[1, 2]

    for i in range(h):
        for j in range(w):
            # Pixel to camera coordinates
            x_cam = (j - c_x) / f_x
            y_cam = (i - c_y) / f_y
            z_cam = 1.0

            # Direction in camera frame
            d_cam = np.array([x_cam, y_cam, z_cam])

            # Transform to world frame
            d_world = R.T @ d_cam
            d_world /= np.linalg.norm(d_world)

            rays_o[i, j] = camera_pos
            rays_d[i, j] = d_world

    # Verify properties
    print(f"\n  Ray Properties:")
    print(f"    Total rays: {h * w}")
    print(f"    Origins (all same): {rays_o[0, 0]}")

    # Direction statistics
    all_dirs = rays_d.reshape(-1, 3)
    norms = np.linalg.norm(all_dirs, axis=1)
    print(f"    Direction norms: mean={norms.mean():.6f}, "
          f"std={norms.std():.2e}")

    # Center ray should point toward origin
    center_dir = rays_d[h // 2, w // 2]
    to_origin = -camera_pos / np.linalg.norm(camera_pos)
    alignment = np.dot(center_dir, to_origin)
    print(f"    Center ray alignment with origin: {alignment:.4f}")

    # Field of view
    corner_dirs = [rays_d[0, 0], rays_d[0, -1], rays_d[-1, 0], rays_d[-1, -1]]
    fov_h = np.degrees(np.arccos(np.clip(
        np.dot(rays_d[h//2, 0], rays_d[h//2, -1]), -1, 1)))
    fov_v = np.degrees(np.arccos(np.clip(
        np.dot(rays_d[0, w//2], rays_d[-1, w//2]), -1, 1)))
    print(f"    FOV horizontal: {fov_h:.1f} degrees")
    print(f"    FOV vertical: {fov_v:.1f} degrees")

    # Sample points along center ray
    n_samples = 64
    t_near, t_far = 2.0, 6.0
    t_vals = np.linspace(t_near, t_far, n_samples)

    center_o = rays_o[h // 2, w // 2]
    center_d = rays_d[h // 2, w // 2]
    sample_points = center_o[None, :] + t_vals[:, None] * center_d[None, :]

    print(f"\n  Sample Points (center ray):")
    print(f"    Range: t=[{t_near}, {t_far}], N={n_samples}")
    print(f"    First point: ({sample_points[0, 0]:.2f}, "
          f"{sample_points[0, 1]:.2f}, {sample_points[0, 2]:.2f})")
    print(f"    Last point:  ({sample_points[-1, 0]:.2f}, "
          f"{sample_points[-1, 1]:.2f}, {sample_points[-1, 2]:.2f})")
    print(f"    Spacing: {(t_far - t_near) / (n_samples - 1):.4f}")

    return rays_o, rays_d, sample_points


# =============================================================================
# Exercise 3: Volumetric Rendering
# =============================================================================

def exercise_3_volume_rendering():
    """
    Implement volumetric rendering along rays.

    C(r) = sum_i T_i * alpha_i * c_i
    T_i = prod_{j<i} (1 - alpha_j)
    alpha_i = 1 - exp(-sigma_i * delta_i)

    Steps:
    1. Define a simple implicit scene (sphere)
    2. Query density and color along rays
    3. Compute transmittance and alpha compositing
    4. Render a pixel color and estimate depth

    Returns:
        (rendered_image, depth_map)
    """
    np.random.seed(42)
    K, h, w = make_camera(f=150.0, cx=40, cy=30, h=60, w=80)

    # Camera at (0, 0, 4) looking at origin
    camera_pos = np.array([0.0, 0.0, 4.0])

    print("Volumetric Rendering")
    print(f"  Image: {w}x{h}")
    print("=" * 60)

    def scene_density_color(point):
        """
        Simple implicit scene: two spheres.
        Returns (density, color_rgb).
        """
        # Sphere 1: center (0, 0, 0), radius 1, red
        dist1 = np.linalg.norm(point - np.array([0, 0, 0]))
        sigma1 = max(0, 50.0 * (1.0 - dist1 / 1.0)) if dist1 < 1.0 else 0.0
        color1 = np.array([0.8, 0.2, 0.2])

        # Sphere 2: center (1.5, 0.5, -0.5), radius 0.6, blue
        dist2 = np.linalg.norm(point - np.array([1.5, 0.5, -0.5]))
        sigma2 = max(0, 40.0 * (1.0 - dist2 / 0.6)) if dist2 < 0.6 else 0.0
        color2 = np.array([0.2, 0.3, 0.9])

        # Combine
        sigma = sigma1 + sigma2
        if sigma > 0:
            color = (sigma1 * color1 + sigma2 * color2) / sigma
        else:
            color = np.array([0.0, 0.0, 0.0])

        return sigma, color

    def render_ray(origin, direction, t_near=1.0, t_far=6.0, n_samples=96):
        """Render a single ray using quadrature."""
        # Stratified sampling
        t_vals = np.linspace(t_near, t_far, n_samples)
        noise = np.random.uniform(0, (t_far - t_near) / n_samples, n_samples)
        t_vals = t_vals + noise

        deltas = np.diff(t_vals, append=t_vals[-1] + 0.1)

        # Query scene at each sample
        sigmas = np.zeros(n_samples)
        colors = np.zeros((n_samples, 3))

        for k in range(n_samples):
            point = origin + t_vals[k] * direction
            sigmas[k], colors[k] = scene_density_color(point)

        # Alpha compositing
        alpha = 1.0 - np.exp(-sigmas * deltas)
        transmittance = np.ones(n_samples)
        for k in range(1, n_samples):
            transmittance[k] = transmittance[k-1] * (1.0 - alpha[k-1])

        weights = transmittance * alpha

        # Rendered color
        rgb = np.sum(weights[:, None] * colors, axis=0)

        # Depth estimate
        depth = np.sum(weights * t_vals)

        # Accumulated opacity
        opacity = np.sum(weights)

        return rgb, depth, opacity, weights

    # Render image
    rendered = np.zeros((h, w, 3), dtype=np.float64)
    depth_map = np.zeros((h, w), dtype=np.float64)
    opacity_map = np.zeros((h, w), dtype=np.float64)

    f_x, f_y = K[0, 0], K[1, 1]
    c_x, c_y = K[0, 2], K[1, 2]

    # Render a subset for speed (every 2nd pixel)
    step = 2
    for i in range(0, h, step):
        for j in range(0, w, step):
            x_cam = (j - c_x) / f_x
            y_cam = (i - c_y) / f_y
            direction = np.array([x_cam, y_cam, -1.0])
            direction /= np.linalg.norm(direction)

            rgb, depth, opacity, _ = render_ray(camera_pos, direction)
            rendered[i, j] = rgb
            depth_map[i, j] = depth
            opacity_map[i, j] = opacity

            # Fill neighboring pixels
            for di in range(step):
                for dj in range(step):
                    if i+di < h and j+dj < w:
                        rendered[i+di, j+dj] = rgb
                        depth_map[i+di, j+dj] = depth

    # Statistics
    valid = opacity_map > 0.1
    print(f"\n  Rendering Results:")
    print(f"    Pixels rendered: {h * w}")
    print(f"    Pixels with content: {valid.sum()}")
    print(f"    Color range: [{rendered.min():.3f}, {rendered.max():.3f}]")
    if valid.any():
        print(f"    Depth range (valid): [{depth_map[valid].min():.2f}, "
              f"{depth_map[valid].max():.2f}]")
    print(f"    Mean opacity: {opacity_map.mean():.4f}")

    # Show center ray details
    center_dir = np.array([0, 0, -1.0])
    rgb_c, depth_c, opacity_c, weights_c = render_ray(
        camera_pos, center_dir, n_samples=96)
    print(f"\n  Center Ray Detail:")
    print(f"    Color: ({rgb_c[0]:.3f}, {rgb_c[1]:.3f}, {rgb_c[2]:.3f})")
    print(f"    Depth: {depth_c:.3f}")
    print(f"    Opacity: {opacity_c:.4f}")
    print(f"    Peak weight at sample: {np.argmax(weights_c)}")

    return rendered, depth_map


# =============================================================================
# Exercise 4: NeRF MLP Simulation
# =============================================================================

def exercise_4_nerf_mlp():
    """
    Simulate the NeRF MLP architecture using numpy.

    Architecture:
    - 8 FC layers (256 units each) for position
    - Skip connection at layer 4
    - Density output (view-independent)
    - Color output (view-dependent, conditioned on direction)

    Returns:
        (outputs, network_stats)
    """
    np.random.seed(42)

    pos_dim = 63   # 3 * (1 + 2*10)
    dir_dim = 27   # 3 * (1 + 2*4)
    hidden = 64    # Reduced from 256 for speed
    n_layers = 8

    print("NeRF MLP Simulation")
    print(f"  Position input dim: {pos_dim}")
    print(f"  Direction input dim: {dir_dim}")
    print(f"  Hidden dim: {hidden}")
    print(f"  Layers: {n_layers}")
    print("=" * 60)

    # Initialize weights (Xavier initialization)
    def xavier_init(fan_in, fan_out):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_in, fan_out))

    # Position layers
    pos_weights = []
    for i in range(n_layers):
        if i == 0:
            w = xavier_init(pos_dim, hidden)
        elif i == 4:
            # Skip connection: input is hidden + pos_dim
            w = xavier_init(hidden + pos_dim, hidden)
        else:
            w = xavier_init(hidden, hidden)
        b = np.zeros(hidden)
        pos_weights.append((w, b))

    # Sigma layer
    sigma_w = xavier_init(hidden, 1)
    sigma_b = np.zeros(1)

    # Color layers
    feature_w = xavier_init(hidden, hidden)
    feature_b = np.zeros(hidden)

    dir_w = xavier_init(hidden + dir_dim, hidden // 2)
    dir_b = np.zeros(hidden // 2)

    rgb_w = xavier_init(hidden // 2, 3)
    rgb_b = np.zeros(3)

    def relu(x):
        return np.maximum(x, 0)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def forward(pos_enc, dir_enc):
        """Forward pass through NeRF MLP."""
        x = pos_enc

        for i, (w, b) in enumerate(pos_weights):
            if i == 4:
                x = np.concatenate([x, pos_enc])
            x = relu(x @ w + b)

        # Density (view-independent)
        sigma = relu(x @ sigma_w + sigma_b)[0]

        # Color (view-dependent)
        feat = relu(x @ feature_w + feature_b)
        combined = np.concatenate([feat, dir_enc])
        color_hidden = relu(combined @ dir_w + dir_b)
        rgb = sigmoid(color_hidden @ rgb_w + rgb_b)

        return rgb, sigma

    # Test with multiple points and directions
    n_test = 50
    test_positions = np.random.uniform(-1, 1, (n_test, 3))
    test_directions = np.random.randn(n_test, 3)
    test_directions /= np.linalg.norm(test_directions, axis=1, keepdims=True)

    # Positional encoding
    def pe(x, L):
        enc = [x]
        for i in range(L):
            freq = 2.0 ** i * np.pi
            enc.append(np.sin(freq * x))
            enc.append(np.cos(freq * x))
        return np.concatenate(enc)

    rgb_outputs = []
    sigma_outputs = []

    for i in range(n_test):
        pos_enc = pe(test_positions[i], L=10)
        dir_enc = pe(test_directions[i], L=4)
        rgb, sigma = forward(pos_enc, dir_enc)
        rgb_outputs.append(rgb)
        sigma_outputs.append(sigma)

    rgb_arr = np.array(rgb_outputs)
    sigma_arr = np.array(sigma_outputs)

    print(f"\n  Network Output Statistics ({n_test} points):")
    print(f"    RGB range: [{rgb_arr.min():.4f}, {rgb_arr.max():.4f}]")
    print(f"    RGB mean: ({rgb_arr[:, 0].mean():.4f}, "
          f"{rgb_arr[:, 1].mean():.4f}, {rgb_arr[:, 2].mean():.4f})")
    print(f"    Sigma range: [{sigma_arr.min():.4f}, {sigma_arr.max():.4f}]")
    print(f"    Sigma mean: {sigma_arr.mean():.4f}")

    # View-dependence test: same point, different directions
    print(f"\n  View-Dependence Test:")
    test_pos = np.array([0.5, 0.3, -0.2])
    pos_enc = pe(test_pos, L=10)

    view_angles = [0, 45, 90, 135, 180]
    for angle in view_angles:
        d = np.array([np.cos(np.radians(angle)), 0, np.sin(np.radians(angle))])
        dir_enc = pe(d, L=4)
        rgb, sigma = forward(pos_enc, dir_enc)
        print(f"    Angle={angle:3d}: RGB=({rgb[0]:.3f}, {rgb[1]:.3f}, "
              f"{rgb[2]:.3f}), sigma={sigma:.3f}")

    # Parameter count
    total_params = sum(w.size + b.size for w, b in pos_weights)
    total_params += sigma_w.size + sigma_b.size
    total_params += feature_w.size + feature_b.size
    total_params += dir_w.size + dir_b.size
    total_params += rgb_w.size + rgb_b.size
    print(f"\n  Network Size:")
    print(f"    Total parameters: {total_params:,}")
    memory_mb = total_params * 8 / (1024 * 1024)
    print(f"    Memory (float64): {memory_mb:.2f} MB")

    return {'rgb': rgb_arr, 'sigma': sigma_arr, 'n_params': total_params}


# =============================================================================
# Exercise 5: Hierarchical Sampling
# =============================================================================

def exercise_5_hierarchical_sampling():
    """
    Implement NeRF's two-stage hierarchical sampling.

    Stage 1: Coarse network with uniform samples
    Stage 2: Fine network with importance samples based on coarse weights

    Returns:
        (coarse_result, fine_result)
    """
    np.random.seed(42)
    t_near, t_far = 2.0, 6.0

    print("Hierarchical Sampling")
    print(f"  Range: [{t_near}, {t_far}]")
    print("=" * 60)

    # Define a 1D density profile (simulates what NeRF sees along a ray)
    def density_profile(t):
        """Simulate density along a ray with two surfaces."""
        # Surface 1 at t=3.0
        sigma1 = 20.0 * np.exp(-((t - 3.0) / 0.1) ** 2)
        # Surface 2 at t=4.5
        sigma2 = 15.0 * np.exp(-((t - 4.5) / 0.15) ** 2)
        return sigma1 + sigma2

    def color_profile(t):
        """Color along the ray."""
        r = 0.5 + 0.3 * np.sin(t * 2)
        g = 0.3 + 0.2 * np.cos(t * 1.5)
        b = 0.4 + 0.1 * np.sin(t * 3)
        return np.array([r, g, b])

    def render_samples(t_vals):
        """Render using given sample positions."""
        n = len(t_vals)
        deltas = np.diff(t_vals, append=t_vals[-1] + 0.1)

        sigmas = np.array([density_profile(t) for t in t_vals])
        colors = np.array([color_profile(t) for t in t_vals])

        alpha = 1.0 - np.exp(-sigmas * deltas)
        transmittance = np.ones(n)
        for k in range(1, n):
            transmittance[k] = transmittance[k-1] * (1.0 - alpha[k-1])

        weights = transmittance * alpha
        rgb = np.sum(weights[:, None] * colors, axis=0)
        depth = np.sum(weights * t_vals)
        opacity = np.sum(weights)

        return rgb, depth, opacity, weights

    # Stage 1: Coarse sampling (uniform)
    n_coarse = 64
    t_coarse = np.linspace(t_near, t_far, n_coarse)
    # Add stratified noise
    noise = np.random.uniform(0, (t_far - t_near) / n_coarse, n_coarse)
    t_coarse = t_coarse + noise

    rgb_coarse, depth_coarse, opacity_coarse, weights_coarse = render_samples(t_coarse)

    print(f"\n  Coarse Stage ({n_coarse} samples):")
    print(f"    Color: ({rgb_coarse[0]:.4f}, {rgb_coarse[1]:.4f}, {rgb_coarse[2]:.4f})")
    print(f"    Depth: {depth_coarse:.4f}")
    print(f"    Opacity: {opacity_coarse:.4f}")

    # Stage 2: Importance sampling from coarse weights
    n_fine = 64
    weights_norm = weights_coarse / (weights_coarse.sum() + 1e-10)

    # CDF-based inverse sampling
    cdf = np.cumsum(weights_norm)
    cdf = np.concatenate([[0], cdf])
    t_bins = np.concatenate([[t_near], t_coarse, [t_far]])

    # Sample from CDF
    u = np.random.uniform(0, 1, n_fine)
    u.sort()
    t_importance = np.zeros(n_fine)

    for i in range(n_fine):
        idx = np.searchsorted(cdf, u[i], side='right') - 1
        idx = max(0, min(idx, len(t_bins) - 2))
        # Linear interpolation within bin
        frac = (u[i] - cdf[idx]) / max(cdf[idx + 1] - cdf[idx], 1e-10)
        t_importance[i] = t_bins[idx] + frac * (t_bins[idx + 1] - t_bins[idx])

    # Combine coarse and fine samples
    t_combined = np.sort(np.concatenate([t_coarse, t_importance]))

    rgb_fine, depth_fine, opacity_fine, weights_fine = render_samples(t_combined)

    print(f"\n  Fine Stage ({n_fine} importance + {n_coarse} coarse = "
          f"{len(t_combined)} total):")
    print(f"    Color: ({rgb_fine[0]:.4f}, {rgb_fine[1]:.4f}, {rgb_fine[2]:.4f})")
    print(f"    Depth: {depth_fine:.4f}")
    print(f"    Opacity: {opacity_fine:.4f}")

    # Analysis: where do importance samples concentrate?
    print(f"\n  Importance Sample Distribution:")
    n_bins = 8
    bin_edges = np.linspace(t_near, t_far, n_bins + 1)
    for b in range(n_bins):
        count_coarse = np.sum((t_coarse >= bin_edges[b]) & (t_coarse < bin_edges[b+1]))
        count_imp = np.sum((t_importance >= bin_edges[b]) & (t_importance < bin_edges[b+1]))
        bar_c = "#" * count_coarse
        bar_i = "#" * count_imp
        print(f"    [{bin_edges[b]:.1f}-{bin_edges[b+1]:.1f}]: "
              f"coarse={count_coarse:2d} {bar_c}")
        print(f"    {'':>16}  import={count_imp:2d} {bar_i}")

    # Quality comparison
    color_diff = np.linalg.norm(rgb_fine - rgb_coarse)
    depth_diff = abs(depth_fine - depth_coarse)
    print(f"\n  Quality Improvement:")
    print(f"    Color difference: {color_diff:.6f}")
    print(f"    Depth difference: {depth_diff:.6f}")
    print(f"    Samples increased: {n_coarse} -> {len(t_combined)}")

    return {'coarse': (rgb_coarse, depth_coarse), 'fine': (rgb_fine, depth_fine)}


# =============================================================================
# Exercise 6: Depth and Normal Extraction
# =============================================================================

def exercise_6_depth_normals():
    """
    Extract depth maps and surface normals from a density field.

    Steps:
    1. Define implicit scene with known geometry
    2. Render depth map from multiple views
    3. Compute surface normals from depth gradients
    4. Compare with analytical normals

    Returns:
        (depth_map, normal_map)
    """
    np.random.seed(42)
    h, w = 40, 50

    print("Depth and Normal Extraction from NeRF")
    print(f"  Image: {w}x{h}")
    print("=" * 60)

    # Implicit sphere: center (0,0,0), radius 1
    sphere_center = np.array([0.0, 0.0, 0.0])
    sphere_radius = 1.0

    def sphere_density(point, center, radius, sharpness=50.0):
        """Density field for a sphere."""
        dist = np.linalg.norm(point - center)
        # Sharp transition at surface
        return max(0, sharpness * (1.0 - dist / radius)) if dist < radius * 1.5 else 0.0

    # Camera
    K, _, _ = make_camera(f=100.0, cx=25, cy=20, h=h, w=w)
    camera_pos = np.array([0.0, 0.0, 3.0])

    f_x, f_y = K[0, 0], K[1, 1]
    c_x, c_y = K[0, 2], K[1, 2]

    # Render depth map
    depth_map = np.zeros((h, w), dtype=np.float64)
    n_samples = 64

    for i in range(h):
        for j in range(w):
            direction = np.array([
                (j - c_x) / f_x,
                (i - c_y) / f_y,
                -1.0
            ])
            direction /= np.linalg.norm(direction)

            t_vals = np.linspace(1.0, 5.0, n_samples)
            sigmas = np.zeros(n_samples)
            for k in range(n_samples):
                point = camera_pos + t_vals[k] * direction
                sigmas[k] = sphere_density(point, sphere_center, sphere_radius)

            deltas = np.diff(t_vals, append=t_vals[-1] + 0.1)
            alpha = 1.0 - np.exp(-sigmas * deltas)
            trans = np.ones(n_samples)
            for k in range(1, n_samples):
                trans[k] = trans[k-1] * (1.0 - alpha[k-1])
            weights = trans * alpha

            depth_map[i, j] = np.sum(weights * t_vals)

    # Compute normals from depth gradients
    normal_map = np.zeros((h, w, 3), dtype=np.float64)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Depth gradients
            dz_dx = (depth_map[i, j+1] - depth_map[i, j-1]) / 2.0
            dz_dy = (depth_map[i+1, j] - depth_map[i-1, j]) / 2.0

            # Normal from depth gradient
            normal = np.array([-dz_dx, -dz_dy, 1.0])
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-6:
                normal /= norm_len
            normal_map[i, j] = normal

    # Compare with analytical normals
    print(f"\n  Depth Map Statistics:")
    valid_depth = depth_map > 0.1
    if valid_depth.any():
        print(f"    Range: [{depth_map[valid_depth].min():.3f}, "
              f"{depth_map[valid_depth].max():.3f}]")
        print(f"    Mean: {depth_map[valid_depth].mean():.3f}")
        print(f"    Valid pixels: {valid_depth.sum()}")

    # Analytical normal at sphere surface: (point - center) / radius
    print(f"\n  Normal Comparison (center pixels):")
    for di, dj in [(-5, 0), (0, 0), (5, 0), (0, -5), (0, 5)]:
        pi, pj = h // 2 + di, w // 2 + dj
        if 0 <= pi < h and 0 <= pj < w:
            depth_val = depth_map[pi, pj]
            est_normal = normal_map[pi, pj]

            # Compute 3D point on sphere surface
            direction = np.array([
                (pj - c_x) / f_x,
                (pi - c_y) / f_y,
                -1.0
            ])
            direction /= np.linalg.norm(direction)
            surface_point = camera_pos + depth_val * direction
            analytical_normal = surface_point - sphere_center
            an_len = np.linalg.norm(analytical_normal)
            if an_len > 1e-6:
                analytical_normal /= an_len

            dot = np.dot(est_normal, analytical_normal)
            angle = np.degrees(np.arccos(np.clip(abs(dot), 0, 1)))
            print(f"    ({pi:2d},{pj:2d}): depth={depth_val:.3f}, "
                  f"normal_error={angle:.1f} deg")

    return depth_map, normal_map


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Positional Encoding")
    exercise_1_positional_encoding()

    print("\n>>> Exercise 2: Ray Generation")
    exercise_2_ray_generation()

    print("\n>>> Exercise 3: Volumetric Rendering")
    exercise_3_volume_rendering()

    print("\n>>> Exercise 4: NeRF MLP Simulation")
    exercise_4_nerf_mlp()

    print("\n>>> Exercise 5: Hierarchical Sampling")
    exercise_5_hierarchical_sampling()

    print("\n>>> Exercise 6: Depth and Normal Extraction")
    exercise_6_depth_normals()

    print("\nAll exercises completed successfully.")
