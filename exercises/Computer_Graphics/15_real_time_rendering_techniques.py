"""
Exercises for Lesson 15: Real-Time Rendering Techniques
Topic: Computer_Graphics
Solutions to practice problems from the lesson.
"""

import numpy as np

# We use Agg backend so the script runs headless (no display required).
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize(v):
    """Normalize a vector, returning zero-vector if magnitude is negligible."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


# ---------------------------------------------------------------------------
# Exercise 1 -- G-buffer Visualization
# ---------------------------------------------------------------------------

def exercise_1():
    """
    Extend the G-buffer simulation to render 5 spheres of different materials.
    Visualize each G-buffer channel as a separate image (albedo, normals,
    depth, roughness, metallic).
    """
    width, height = 160, 120

    # G-buffer arrays: albedo (RGB), normal (XYZ), depth, metallic, roughness
    albedo_buf = np.zeros((height, width, 3))
    normal_buf = np.zeros((height, width, 3))
    depth_buf = np.full((height, width), np.inf)
    metallic_buf = np.zeros((height, width))
    roughness_buf = np.zeros((height, width))

    # Sphere definitions: (cx_frac, cy_frac, radius_frac, albedo, metallic, roughness)
    spheres = [
        # Red rubber ball -- low metallic, medium roughness
        (0.20, 0.45, 0.14, np.array([0.9, 0.15, 0.10]), 0.0, 0.7),
        # Gold sphere -- high metallic, low roughness
        (0.40, 0.50, 0.12, np.array([1.0, 0.77, 0.34]), 1.0, 0.2),
        # Blue plastic -- low metallic, medium-low roughness
        (0.60, 0.45, 0.13, np.array([0.15, 0.30, 0.90]), 0.0, 0.4),
        # Chrome sphere -- full metallic, mirror-like
        (0.80, 0.50, 0.11, np.array([0.95, 0.95, 0.97]), 1.0, 0.05),
        # Green matte -- zero metallic, very rough
        (0.50, 0.70, 0.10, np.array([0.20, 0.80, 0.20]), 0.0, 0.95),
    ]

    # Rasterize each sphere into the G-buffer with a depth test.
    for cx_frac, cy_frac, r_frac, albedo, metallic, roughness in spheres:
        cx = int(cx_frac * width)
        cy = int(cy_frac * height)
        r = int(r_frac * min(width, height))
        base_depth = 3.0 + cx_frac  # stagger depths slightly

        for y in range(max(0, cy - r), min(height, cy + r + 1)):
            for x in range(max(0, cx - r), min(width, cx + r + 1)):
                dx = (x - cx) / r
                dy = (y - cy) / r
                dist2 = dx * dx + dy * dy
                if dist2 > 1.0:
                    continue
                dz = np.sqrt(1.0 - dist2)
                pixel_depth = base_depth - dz * 0.5

                # Depth test: only write the closest surface
                if pixel_depth < depth_buf[y, x]:
                    depth_buf[y, x] = pixel_depth
                    albedo_buf[y, x] = albedo
                    normal_buf[y, x] = normalize(np.array([dx, -dy, dz]))
                    metallic_buf[y, x] = metallic
                    roughness_buf[y, x] = roughness

    # Compute memory estimate (mirrors real GPU G-buffer sizing)
    bytes_per_pixel = (3 * 4 + 3 * 4 + 4 + 4 + 4)  # albedo + normal + depth + met + rough
    memory_mb = width * height * bytes_per_pixel / (1024 * 1024)

    print("Exercise 1: G-buffer Visualization (5 spheres)")
    print(f"  Resolution: {width}x{height}")
    print(f"  G-buffer memory estimate: {memory_mb:.3f} MB")
    print(f"  At 1920x1080 this would be: "
          f"{1920 * 1080 * bytes_per_pixel / (1024 * 1024):.1f} MB")

    # Visualize each channel
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(albedo_buf)
    axes[0, 0].set_title('G-Buffer: Albedo (RGB)')

    # Remap normals from [-1,1] to [0,1] for display
    norm_vis = (normal_buf + 1.0) / 2.0
    norm_vis[depth_buf == np.inf] = 0  # black background
    axes[0, 1].imshow(norm_vis)
    axes[0, 1].set_title('G-Buffer: Normal (remapped)')

    depth_display = depth_buf.copy()
    depth_display[depth_display == np.inf] = np.nan
    axes[0, 2].imshow(depth_display, cmap='gray_r')
    axes[0, 2].set_title('G-Buffer: Depth (closer=brighter)')

    axes[1, 0].imshow(roughness_buf, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('G-Buffer: Roughness')

    axes[1, 1].imshow(metallic_buf, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('G-Buffer: Metallic')

    # Simple deferred lighting result
    light_pos = np.array([0.0, 5.0, 5.0])
    light_color = np.array([1.0, 0.98, 0.92])
    camera_pos = np.array([0.0, 0.0, 10.0])
    lit = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            if depth_buf[y, x] == np.inf:
                continue
            # Reconstruct approximate world position
            world_pos = np.array([
                (x / width - 0.5) * 2.0 * depth_buf[y, x],
                (0.5 - y / height) * 2.0 * depth_buf[y, x],
                -depth_buf[y, x]
            ])
            N = normal_buf[y, x]
            L = normalize(light_pos - world_pos)
            V = normalize(camera_pos - world_pos)
            H = normalize(L + V)

            NdotL = max(0.0, np.dot(N, L))
            NdotH = max(0.0, np.dot(N, H))
            spec_pow = max(1.0, (1.0 - roughness_buf[y, x]) * 256)

            diffuse = (1.0 - metallic_buf[y, x]) * albedo_buf[y, x] * NdotL
            spec = (metallic_buf[y, x] * albedo_buf[y, x] +
                    (1.0 - metallic_buf[y, x]) * 0.04) * (NdotH ** spec_pow)
            ambient = 0.03 * albedo_buf[y, x]
            lit[y, x] = ambient + (diffuse + spec) * light_color

    # Reinhard tone map + gamma
    lit = lit / (1.0 + lit)
    lit = np.power(np.clip(lit, 0, 1), 1.0 / 2.2)

    axes[1, 2].imshow(lit)
    axes[1, 2].set_title('Deferred Lighting Result')

    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('ex15_gbuffer_visualization.png', dpi=100)
    plt.close()
    print("  Saved: ex15_gbuffer_visualization.png")

    # Report per-sphere material summary
    for i, (cx_f, cy_f, r_f, alb, met, rough) in enumerate(spheres):
        print(f"  Sphere {i}: albedo=({alb[0]:.2f},{alb[1]:.2f},{alb[2]:.2f}), "
              f"metallic={met:.2f}, roughness={rough:.2f}")


# ---------------------------------------------------------------------------
# Exercise 2 -- Shadow Mapping (2D top-down)
# ---------------------------------------------------------------------------

def exercise_2():
    """
    Implement a basic 2D shadow map for a top-down scene with a directional
    light.  Build the shadow map from the light's perspective, then shade the
    scene using shadow map lookups.
    """
    # 2D scene: rectangles as occluders on a 1D scanline viewed from above.
    # Light shines from the top (y = +inf direction, i.e., column-wise).
    # Shadow map stores the first hit depth per column.

    scene_w, scene_h = 200, 200
    shadow_map_res = 200  # 1 texel per column

    # Define rectangular occluders: (x_min, x_max, y_top, y_bottom)
    # y increases downward; light comes from y=0 (top)
    occluders = [
        (30, 70, 40, 55),     # wide short block near top
        (100, 130, 70, 90),   # medium block
        (50, 65, 120, 140),   # block lower down
        (140, 180, 30, 50),   # right-side block
    ]

    # Receiver: the floor at y = 180 (bottom of scene)
    floor_y = 180

    # --- Pass 1: Build shadow map from light's perspective ---
    # Light is directional from top.  For each column x, find the smallest y
    # (first surface hit by light traveling downward).
    shadow_map = np.full(shadow_map_res, np.inf)

    for x_min, x_max, y_top, y_bottom in occluders:
        for x in range(x_min, min(x_max, shadow_map_res)):
            if y_top < shadow_map[x]:
                shadow_map[x] = y_top

    print("Exercise 2: Shadow Mapping (2D top-down directional light)")
    print(f"  Scene size: {scene_w}x{scene_h}")
    print(f"  Shadow map resolution: {shadow_map_res}")
    print(f"  Occluders: {len(occluders)}")

    # --- Pass 2: Shade the scene ---
    image = np.ones((scene_h, scene_w, 3)) * 0.85  # light gray background

    # Draw occluders (dark gray)
    for x_min, x_max, y_top, y_bottom in occluders:
        image[y_top:y_bottom, x_min:x_max] = np.array([0.4, 0.4, 0.5])

    # Draw floor and apply shadow test
    bias = 2.0  # shadow bias to avoid self-shadowing
    lit_count = 0
    shadowed_count = 0

    for x in range(scene_w):
        # Floor pixel at (x, floor_y)
        # Shadow test: is the floor pixel deeper than the shadow map value?
        if x < shadow_map_res and shadow_map[x] < np.inf:
            if floor_y > shadow_map[x] + bias:
                # In shadow
                image[floor_y - 2:floor_y + 3, x] = np.array([0.2, 0.2, 0.25])
                shadowed_count += 1
            else:
                image[floor_y - 2:floor_y + 3, x] = np.array([0.95, 0.90, 0.70])
                lit_count += 1
        else:
            # No occluder: fully lit
            image[floor_y - 2:floor_y + 3, x] = np.array([0.95, 0.90, 0.70])
            lit_count += 1

    print(f"  Floor pixels: {lit_count} lit, {shadowed_count} shadowed")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Shadow map 1D visualization
    sm_display = shadow_map.copy()
    sm_display[sm_display == np.inf] = scene_h
    axes[0].bar(range(shadow_map_res), sm_display, width=1.0, color='steelblue')
    axes[0].set_title('Shadow Map (depth per column, lower = closer to light)')
    axes[0].set_xlabel('Column (x)')
    axes[0].set_ylabel('Depth (y from top)')
    axes[0].set_ylim(0, scene_h)
    axes[0].invert_yaxis()

    # Scene with shadows
    axes[1].imshow(image)
    axes[1].set_title('Scene with Shadow Mapping')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')

    plt.tight_layout()
    plt.savefig('ex15_shadow_mapping.png', dpi=100)
    plt.close()
    print("  Saved: ex15_shadow_mapping.png")


# ---------------------------------------------------------------------------
# Exercise 3 -- PCF Implementation
# ---------------------------------------------------------------------------

def exercise_3():
    """
    Add Percentage-Closer Filtering (PCF) to shadow mapping.
    Compare 1-sample (hard), 3x3, and 5x5 PCF kernels.
    Measure visual quality improvement and performance cost.
    """
    import time

    # Build a 1D scene with 2D shadow map (light from top, scene is XY plane)
    sm_res = 256
    scene_h = 200
    floor_y = 180.0
    bias = 1.5

    # Occluder: a single block casting a shadow with clear edges
    occ_x0, occ_x1 = 80, 170
    occ_y = 50.0  # top of occluder

    # Build shadow map (1D: depth per column)
    shadow_map = np.full(sm_res, np.inf)
    for x in range(occ_x0, occ_x1):
        if x < sm_res:
            shadow_map[x] = occ_y

    def shadow_test_pcf(x, depth, kernel_size):
        """
        PCF shadow test: sample kernel_size x 1 neighbors and average.
        Returns shadow factor in [0, 1] where 1 = fully lit, 0 = fully shadowed.
        """
        half_k = kernel_size // 2
        total = 0
        count = 0
        for dx in range(-half_k, half_k + 1):
            sx = x + dx
            if 0 <= sx < sm_res:
                # Each sample is an independent depth comparison, then averaged.
                # This is the correct PCF approach: compare-then-average.
                if shadow_map[sx] == np.inf or depth <= shadow_map[sx] + bias:
                    total += 1.0
                count += 1
        return total / count if count > 0 else 1.0

    kernels = {'1-sample (hard)': 1, '3x3 PCF': 3, '5x5 PCF': 5}
    results = {}

    print("Exercise 3: PCF Implementation")
    print(f"  Occluder at columns [{occ_x0}, {occ_x1}), depth={occ_y}")
    print(f"  Floor at depth={floor_y}, bias={bias}")
    print()

    for label, ksize in kernels.items():
        t0 = time.perf_counter()
        shadow_row = np.zeros(sm_res)
        iterations = 500  # repeat for timing
        for _ in range(iterations):
            for x in range(sm_res):
                shadow_row[x] = shadow_test_pcf(x, floor_y, ksize)
        elapsed = (time.perf_counter() - t0) / iterations * 1000  # ms per row
        results[label] = (shadow_row, elapsed)

        # Count transition pixels (neither 0 nor 1)
        soft_pixels = np.sum((shadow_row > 0.01) & (shadow_row < 0.99))
        print(f"  {label:18s}: transition pixels = {soft_pixels:3d}, "
              f"time = {elapsed:.3f} ms/row")

    # Visualize comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    colors = ['red', 'blue', 'green']
    for i, (label, (row, elapsed)) in enumerate(results.items()):
        axes[0].plot(row, label=f'{label} ({elapsed:.2f} ms)', color=colors[i])
    axes[0].set_title('Shadow Factor per Column (1 = lit, 0 = shadow)')
    axes[0].set_xlabel('Column (x)')
    axes[0].set_ylabel('Shadow factor')
    axes[0].legend()
    axes[0].axvspan(occ_x0, occ_x1, alpha=0.1, color='gray', label='Occluder')

    # Create 2D visualization of each method
    strip_h = 30
    combined = np.ones((strip_h * 3, sm_res, 3)) * 0.8
    for i, (label, (row, _)) in enumerate(results.items()):
        for x in range(sm_res):
            val = row[x]
            combined[i * strip_h:(i + 1) * strip_h, x] = val * np.array([0.95, 0.9, 0.7])
    axes[1].imshow(combined)
    axes[1].set_title('Visual: Hard (top) | 3x3 PCF (mid) | 5x5 PCF (bottom)')
    axes[1].set_xlabel('Column (x)')
    axes[1].set_yticks([strip_h // 2, strip_h + strip_h // 2, 2 * strip_h + strip_h // 2])
    axes[1].set_yticklabels(['1-sample', '3x3 PCF', '5x5 PCF'])

    plt.tight_layout()
    plt.savefig('ex15_pcf_comparison.png', dpi=100)
    plt.close()
    print("  Saved: ex15_pcf_comparison.png")


# ---------------------------------------------------------------------------
# Exercise 4 -- SSAO Quality
# ---------------------------------------------------------------------------

def exercise_4():
    """
    Implement SSAO for the full G-buffer.  Compare results with 8, 16, 32,
    and 64 samples.  Add a bilateral blur post-process to smooth the AO
    without blurring across edges.
    """
    width, height = 80, 60

    # Build a simple G-buffer scene: floor + 2 spheres near a wall
    depth_buf = np.full((height, width), np.inf)
    normal_buf = np.zeros((height, width, 3))

    # Floor: y > height/2, flat normal pointing toward camera (+Z)
    for y in range(height // 2, height):
        for x in range(width):
            depth_buf[y, x] = 5.0 + (y - height // 2) * 0.08
            normal_buf[y, x] = np.array([0.0, 0.0, 1.0])

    # Back wall: leftmost 10 columns
    for y in range(10, height):
        for x in range(0, 12):
            d = 5.5
            if d < depth_buf[y, x]:
                depth_buf[y, x] = d
                normal_buf[y, x] = np.array([1.0, 0.0, 0.0])  # facing right

    # Sphere in the corner (creates occlusion)
    scx, scy, sr = 18, height // 2 + 5, 8
    for y in range(max(0, scy - sr), min(height, scy + sr)):
        for x in range(max(0, scx - sr), min(width, scx + sr)):
            dx, dy = (x - scx) / sr, (y - scy) / sr
            if dx * dx + dy * dy <= 1.0:
                dz = np.sqrt(1.0 - dx * dx - dy * dy)
                d = 4.0 - dz * 0.5
                if d < depth_buf[y, x]:
                    depth_buf[y, x] = d
                    normal_buf[y, x] = normalize(np.array([dx, -dy, dz]))

    # Second sphere
    scx2, scy2, sr2 = 50, height // 2 + 8, 6
    for y in range(max(0, scy2 - sr2), min(height, scy2 + sr2)):
        for x in range(max(0, scx2 - sr2), min(width, scx2 + sr2)):
            dx, dy = (x - scx2) / sr2, (y - scy2) / sr2
            if dx * dx + dy * dy <= 1.0:
                dz = np.sqrt(1.0 - dx * dx - dy * dy)
                d = 4.2 - dz * 0.4
                if d < depth_buf[y, x]:
                    depth_buf[y, x] = d
                    normal_buf[y, x] = normalize(np.array([dx, -dy, dz]))

    def compute_ssao(depth_buf, normal_buf, num_samples, radius=0.5):
        """Compute SSAO for the entire buffer with a given sample count."""
        h, w = depth_buf.shape
        ao = np.ones((h, w))
        rng = np.random.RandomState(42)

        for y in range(h):
            for x in range(w):
                if depth_buf[y, x] >= 1e8:
                    continue
                center_depth = depth_buf[y, x]
                N = normal_buf[y, x]
                occlusion = 0.0

                for _ in range(num_samples):
                    # Random direction in hemisphere above surface
                    rand_dir = rng.randn(3)
                    rand_dir = normalize(rand_dir)
                    if np.dot(rand_dir, N) < 0:
                        rand_dir = -rand_dir

                    offset = rand_dir * radius * rng.random()

                    sx = int(x + offset[0] * w * 0.05)
                    sy = int(y - offset[1] * h * 0.05)
                    sample_depth = center_depth - offset[2]

                    if not (0 <= sx < w and 0 <= sy < h):
                        continue

                    buf_depth = depth_buf[sy, sx]
                    if buf_depth < sample_depth and (sample_depth - buf_depth) < radius:
                        occlusion += 1.0

                ao[y, x] = 1.0 - (occlusion / num_samples)
        return ao

    def bilateral_blur(ao, depth_buf, kernel_size=3, sigma_space=1.0, sigma_depth=0.5):
        """
        Bilateral blur: smooths AO but preserves edges where depth changes sharply.
        This prevents blurring shadow across depth discontinuities.
        """
        h, w = ao.shape
        result = ao.copy()
        half_k = kernel_size // 2

        for y in range(h):
            for x in range(w):
                if depth_buf[y, x] >= 1e8:
                    continue
                center_d = depth_buf[y, x]
                total_weight = 0.0
                total_val = 0.0

                for dy in range(-half_k, half_k + 1):
                    for dx in range(-half_k, half_k + 1):
                        ny, nx = y + dy, x + dx
                        if not (0 <= ny < h and 0 <= nx < w):
                            continue
                        if depth_buf[ny, nx] >= 1e8:
                            continue

                        # Spatial weight (Gaussian on pixel distance)
                        spatial = np.exp(-(dx * dx + dy * dy) /
                                         (2.0 * sigma_space * sigma_space))
                        # Depth weight (Gaussian on depth difference)
                        depth_diff = abs(depth_buf[ny, nx] - center_d)
                        depth_w = np.exp(-(depth_diff * depth_diff) /
                                          (2.0 * sigma_depth * sigma_depth))

                        w_val = spatial * depth_w
                        total_weight += w_val
                        total_val += ao[ny, nx] * w_val

                if total_weight > 0:
                    result[y, x] = total_val / total_weight
        return result

    sample_counts = [8, 16, 32, 64]
    ao_results = {}

    print("Exercise 4: SSAO Quality Comparison")
    print(f"  Scene: {width}x{height}, floor + wall + 2 spheres")

    for n_samples in sample_counts:
        ao = compute_ssao(depth_buf, normal_buf, n_samples, radius=0.5)
        ao_blurred = bilateral_blur(ao, depth_buf, kernel_size=3)
        ao_results[n_samples] = (ao, ao_blurred)

        # Statistics
        valid_mask = depth_buf < 1e8
        mean_ao = ao[valid_mask].mean()
        std_ao = ao[valid_mask].std()
        mean_blurred = ao_blurred[valid_mask].mean()
        print(f"  {n_samples:3d} samples: mean AO = {mean_ao:.3f} "
              f"(std={std_ao:.3f}), after blur = {mean_blurred:.3f}")

    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, n_samples in enumerate(sample_counts):
        ao_raw, ao_blurred = ao_results[n_samples]
        axes[0, i].imshow(ao_raw, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'SSAO {n_samples} samples (raw)')
        axes[0, i].axis('off')

        axes[1, i].imshow(ao_blurred, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'SSAO {n_samples} samples (bilateral blur)')
        axes[1, i].axis('off')

    plt.suptitle('SSAO: More samples reduce noise; bilateral blur preserves edges',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig('ex15_ssao_comparison.png', dpi=100)
    plt.close()
    print("  Saved: ex15_ssao_comparison.png")


# ---------------------------------------------------------------------------
# Exercise 5 -- Bloom Parameter Exploration
# ---------------------------------------------------------------------------

def exercise_5():
    """
    Implement the multi-level bloom pipeline.  Render a scene with one very
    bright light and several dim lights.  Experiment with threshold (0.5, 1.0,
    2.0), bloom strength, and number of blur levels.
    """
    width, height = 160, 120

    # Create a synthetic HDR image with one bright emitter and dim background
    hdr = np.zeros((height, width, 3))

    # Background: dim warm fill
    hdr[:, :] = np.array([0.15, 0.12, 0.10])

    # Bright emitter (sun-like spot)
    for y in range(height):
        for x in range(width):
            dx = (x - 40) / 15.0
            dy = (y - 30) / 15.0
            dist2 = dx * dx + dy * dy
            if dist2 < 1.0:
                brightness = 8.0 * (1.0 - dist2)  # HDR value >> 1
                hdr[y, x] += np.array([brightness, brightness * 0.9, brightness * 0.6])

    # Medium-bright light
    for y in range(height):
        for x in range(width):
            dx = (x - 120) / 10.0
            dy = (y - 60) / 10.0
            dist2 = dx * dx + dy * dy
            if dist2 < 1.0:
                brightness = 2.5 * (1.0 - dist2)
                hdr[y, x] += np.array([brightness * 0.4, brightness * 0.6, brightness])

    # Dim light
    for y in range(height):
        for x in range(width):
            dx = (x - 80) / 8.0
            dy = (y - 90) / 8.0
            dist2 = dx * dx + dy * dy
            if dist2 < 1.0:
                brightness = 0.6 * (1.0 - dist2)
                hdr[y, x] += np.array([brightness * 0.3, brightness, brightness * 0.3])

    def extract_bright(image, threshold):
        brightness = np.max(image, axis=2)
        mask = (brightness > threshold).astype(float)[:, :, np.newaxis]
        return image * mask

    def box_downsample(image, factor=2):
        h, w = image.shape[:2]
        nh, nw = h // factor, w // factor
        return (image[:nh * factor, :nw * factor]
                .reshape(nh, factor, nw, factor, -1)
                .mean(axis=(1, 3)))

    def box_upsample(image, target_h, target_w):
        sh, sw = image.shape[:2]
        result = np.zeros((target_h, target_w, image.shape[2]))
        for y in range(target_h):
            for x in range(target_w):
                sy = min(y * sh // target_h, sh - 1)
                sx = min(x * sw // target_w, sw - 1)
                result[y, x] = image[sy, sx]
        return result

    def apply_bloom(hdr_img, threshold, strength, levels):
        bright = extract_bright(hdr_img, threshold)

        # Build mip chain
        mips = [bright]
        current = bright
        for _ in range(levels):
            if min(current.shape[:2]) < 4:
                break
            current = box_downsample(current)
            mips.append(current)

        # Accumulate upsampled mips
        bloom = np.zeros_like(hdr_img)
        for mip in mips:
            bloom += box_upsample(mip, hdr_img.shape[0], hdr_img.shape[1])
        bloom /= len(mips)

        return hdr_img + strength * bloom

    def reinhard_tonemap(img):
        mapped = img / (1.0 + img)
        return np.power(np.clip(mapped, 0, 1), 1.0 / 2.2)

    # Test different parameter combinations
    thresholds = [0.5, 1.0, 2.0]
    strengths = [0.3, 0.7, 1.5]
    level_counts = [2, 3, 4]

    print("Exercise 5: Bloom Parameter Exploration")
    print(f"  HDR range: [{hdr.min():.2f}, {hdr.max():.2f}]")
    print()

    # Threshold comparison (fixed strength=0.7, levels=3)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # Top row: threshold comparison
    axes[0, 0].imshow(reinhard_tonemap(hdr))
    axes[0, 0].set_title('No Bloom (tone mapped)')
    axes[0, 0].axis('off')

    for i, thresh in enumerate(thresholds):
        bloomed = apply_bloom(hdr, thresh, strength=0.7, levels=3)
        bright_pixels = np.sum(np.max(hdr, axis=2) > thresh)
        axes[0, i + 1].imshow(reinhard_tonemap(bloomed))
        axes[0, i + 1].set_title(f'Threshold={thresh}\n({bright_pixels} bright px)')
        axes[0, i + 1].axis('off')
        print(f"  Threshold={thresh}: {bright_pixels} pixels above threshold")

    # Bottom row: strength comparison (fixed threshold=1.0, levels=3)
    axes[1, 0].imshow(reinhard_tonemap(hdr))
    axes[1, 0].set_title('No Bloom')
    axes[1, 0].axis('off')

    for i, strength in enumerate(strengths):
        bloomed = apply_bloom(hdr, threshold=1.0, strength=strength, levels=3)
        axes[1, i + 1].imshow(reinhard_tonemap(bloomed))
        axes[1, i + 1].set_title(f'Strength={strength}')
        axes[1, i + 1].axis('off')

    print()
    print("  Lower threshold -> more glow (catches dimmer areas)")
    print("  Higher strength -> more intense bloom effect")
    print("  More levels -> wider bloom (spreads further from source)")

    plt.suptitle('Bloom: threshold controls which pixels glow, '
                 'strength controls intensity', fontsize=12)
    plt.tight_layout()
    plt.savefig('ex15_bloom_exploration.png', dpi=100)
    plt.close()
    print("  Saved: ex15_bloom_exploration.png")


# ---------------------------------------------------------------------------
# Exercise 6 -- Tone Mapping Comparison
# ---------------------------------------------------------------------------

def exercise_6():
    """
    Implement Reinhard, ACES, and Hable (Uncharted 2) tone mapping.
    Plot the transfer curves and apply each to the same HDR image.
    """
    # --- Tone mapping operators ---

    def tonemap_reinhard(x):
        """
        Reinhard: L / (1 + L).
        Simple, maps [0, inf) -> [0, 1). Does not have a nice shoulder.
        """
        return x / (1.0 + x)

    def tonemap_aces(x):
        """
        ACES filmic curve (fitted by Krzysztof Narkowicz).
        Industry standard for games and film.
        f(x) = (x*(a*x + b)) / (x*(c*x + d) + e)
        """
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)

    def tonemap_hable(x):
        """
        Hable / Uncharted 2 filmic curve.
        f(x) = ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F)) - E/F
        Parameters from John Hable's presentation.
        """
        A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
        def curve(t):
            return ((t * (A * t + C * B) + D * E) /
                    (t * (A * t + B) + D * F)) - E / F

        # Hable needs a white point normalization
        white = 11.2
        return curve(x) / curve(white)

    # --- Plot transfer curves ---
    x_range = np.linspace(0, 10, 500)
    reinhard_y = tonemap_reinhard(x_range)
    aces_y = tonemap_aces(x_range)
    hable_y = tonemap_hable(x_range)

    print("Exercise 6: Tone Mapping Comparison")
    print()
    print("  Transfer curve characteristics:")
    print(f"  {'Operator':12s} | f(0.5) | f(1.0) | f(2.0) | f(5.0) | f(10.0)")
    print(f"  {'-'*12:s}-+-{'-'*6:s}-+-{'-'*6:s}-+-{'-'*6:s}-+-{'-'*6:s}-+-{'-'*6:s}")
    for name, func in [('Reinhard', tonemap_reinhard),
                        ('ACES', tonemap_aces),
                        ('Hable', tonemap_hable)]:
        vals = [func(np.array([v]))[0] if isinstance(func(np.array([v])), np.ndarray)
                else func(v) for v in [0.5, 1.0, 2.0, 5.0, 10.0]]
        print(f"  {name:12s} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | "
              f"{vals[3]:.4f} | {vals[4]:.4f}")

    # --- Create HDR test image ---
    h, w = 120, 200
    hdr = np.zeros((h, w, 3))

    # Gradient from 0 to 10 across the width (same in all rows)
    for x in range(w):
        val = 10.0 * x / w
        hdr[:, x] = val

    # Add bright spots
    for cy, cx, brightness, color in [
        (30, 40, 8.0, np.array([1.0, 0.9, 0.6])),
        (60, 120, 5.0, np.array([0.5, 0.7, 1.0])),
        (90, 80, 3.0, np.array([0.3, 1.0, 0.4])),
    ]:
        for y in range(h):
            for x in range(w):
                dist2 = ((x - cx) / 15.0) ** 2 + ((y - cy) / 15.0) ** 2
                if dist2 < 1.0:
                    hdr[y, x] += brightness * (1.0 - dist2) * color

    # Apply each tone mapper + gamma
    def apply_and_gamma(hdr_img, tonemap_fn):
        mapped = tonemap_fn(hdr_img)
        return np.power(np.clip(mapped, 0, 1), 1.0 / 2.2)

    result_reinhard = apply_and_gamma(hdr, tonemap_reinhard)
    result_aces = apply_and_gamma(hdr, tonemap_aces)
    result_hable = apply_and_gamma(hdr, tonemap_hable)

    # --- Visualize ---
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])

    # Transfer curves
    ax_curve = fig.add_subplot(gs[0, :])
    ax_curve.plot(x_range, reinhard_y, 'r-', linewidth=2, label='Reinhard')
    ax_curve.plot(x_range, aces_y, 'b-', linewidth=2, label='ACES')
    ax_curve.plot(x_range, hable_y, 'g-', linewidth=2, label='Hable (Uncharted 2)')
    ax_curve.plot(x_range, np.ones_like(x_range), 'k--', alpha=0.3, label='y=1 (display max)')
    ax_curve.set_xlabel('HDR Input Luminance')
    ax_curve.set_ylabel('Display Output')
    ax_curve.set_title('Tone Mapping Transfer Curves')
    ax_curve.legend(fontsize=11)
    ax_curve.set_ylim(0, 1.1)
    ax_curve.grid(True, alpha=0.3)

    # Applied images
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(result_reinhard)
    ax1.set_title('Reinhard\n(simple, no shoulder)')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(result_aces)
    ax2.set_title('ACES\n(industry standard, nice contrast)')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[1, 2])
    ax3.imshow(result_hable)
    ax3.set_title('Hable / Uncharted 2\n(filmic, strong shoulder)')
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig('ex15_tonemapping_comparison.png', dpi=100)
    plt.close()

    print()
    print("  Reinhard: simplest, maps [0,inf)->[0,1), bright areas look washed out")
    print("  ACES:     industry standard, good contrast, nice saturation roll-off")
    print("  Hable:    filmic look, strong shoulder compresses highlights naturally")
    print("  Saved: ex15_tonemapping_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 15: Real-Time Rendering Techniques -- Exercises")
    print("=" * 70)
    print()

    exercise_1()
    print()
    exercise_2()
    print()
    exercise_3()
    print()
    exercise_4()
    print()
    exercise_5()
    print()
    exercise_6()

    print()
    print("=" * 70)
    print("All exercises completed.")
    print("=" * 70)
