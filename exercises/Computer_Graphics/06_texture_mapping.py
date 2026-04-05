"""
Exercises for Lesson 06: Texture Mapping
Topic: Computer_Graphics
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ===================================================================
# Helper functions (from the lesson)
# ===================================================================

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def wrap_uv(u, v, mode='repeat'):
    if mode == 'repeat':
        u = u % 1.0
        v = v % 1.0
    elif mode == 'clamp':
        u = np.clip(u, 0.0, 1.0 - 1e-6)
        v = np.clip(v, 0.0, 1.0 - 1e-6)
    elif mode == 'mirrored_repeat':
        u = u % 2.0
        v = v % 2.0
        if u > 1.0:
            u = 2.0 - u
        if v > 1.0:
            v = 2.0 - v
    return u, v


def sample_nearest(texture, u, v, mode='repeat'):
    h, w = texture.shape[:2]
    u, v = wrap_uv(u, v, mode)
    x = int(u * w) % w
    y = int(v * h) % h
    return texture[y, x].astype(float) / 255.0 if texture.dtype == np.uint8 else texture[y, x]


def sample_bilinear(texture, u, v, mode='repeat'):
    h, w = texture.shape[:2]
    u, v = wrap_uv(u, v, mode)
    x = u * w - 0.5
    y = v * h - 0.5
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    fx = x - x0
    fy = y - y0
    x0w, x1w = x0 % w, (x0 + 1) % w
    y0w, y1w = y0 % h, (y0 + 1) % h
    if texture.dtype == np.uint8:
        c00 = texture[y0w, x0w].astype(float) / 255.0
        c10 = texture[y0w, x1w].astype(float) / 255.0
        c01 = texture[y1w, x0w].astype(float) / 255.0
        c11 = texture[y1w, x1w].astype(float) / 255.0
    else:
        c00 = texture[y0w, x0w]
        c10 = texture[y0w, x1w]
        c01 = texture[y1w, x0w]
        c11 = texture[y1w, x1w]
    top = c00 * (1 - fx) + c10 * fx
    bottom = c01 * (1 - fx) + c11 * fx
    return top * (1 - fy) + bottom * fy


def create_checkerboard(size=256, squares=8):
    texture = np.zeros((size, size, 3), dtype=np.uint8)
    block = size // squares
    for y in range(size):
        for x in range(size):
            if ((x // block) + (y // block)) % 2 == 0:
                texture[y, x] = [200, 200, 200]
            else:
                texture[y, x] = [50, 50, 50]
    return texture


def generate_mipmaps(texture):
    mipmaps = [texture.astype(float)]
    h, w = texture.shape[:2]
    while h > 1 or w > 1:
        new_h = max(1, h // 2)
        new_w = max(1, w // 2)
        prev = mipmaps[-1]
        new_level = np.zeros((new_h, new_w, prev.shape[2]), dtype=float)
        for y in range(new_h):
            for x in range(new_w):
                y0, y1 = y * 2, min(y * 2 + 1, h - 1)
                x0, x1 = x * 2, min(x * 2 + 1, w - 1)
                new_level[y, x] = (prev[y0, x0] + prev[y0, x1] +
                                   prev[y1, x0] + prev[y1, x1]) / 4.0
        mipmaps.append(new_level)
        h, w = new_h, new_w
    return mipmaps


def exercise_1():
    """
    Bilinear vs Nearest: Create a 4x4 checkerboard texture and magnify it 16x
    using both nearest-neighbor and bilinear sampling. Compare results.
    """
    # Create a tiny 4x4 checkerboard
    tex = create_checkerboard(size=4, squares=2)
    magnification = 16
    out_size = 4 * magnification  # 64x64 output

    img_nearest = np.zeros((out_size, out_size, 3))
    img_bilinear = np.zeros((out_size, out_size, 3))

    print("Bilinear vs Nearest-Neighbor Sampling")
    print(f"Texture: 4x4 checkerboard, magnified {magnification}x to {out_size}x{out_size}")
    print()

    for y in range(out_size):
        for x in range(out_size):
            u = (x + 0.5) / out_size
            v = (y + 0.5) / out_size
            img_nearest[y, x] = sample_nearest(tex, u, v)
            img_bilinear[y, x] = sample_bilinear(tex, u, v)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(tex, interpolation='nearest')
    axes[0].set_title('Original 4x4 Texture')
    axes[1].imshow(img_nearest, interpolation='nearest')
    axes[1].set_title(f'Nearest-Neighbor ({magnification}x)')
    axes[2].imshow(img_bilinear, interpolation='nearest')
    axes[2].set_title(f'Bilinear ({magnification}x)')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('ex06_bilinear_vs_nearest.png', dpi=150, bbox_inches='tight')
    print("Saved ex06_bilinear_vs_nearest.png")
    print()
    print("Comparison:")
    print("  Nearest-neighbor: Preserves sharp edges, blocky appearance")
    print("    - Each original texel maps to a 16x16 block of identical pixels")
    print("    - Good for pixel art where sharpness is desired")
    print()
    print("  Bilinear: Smooth blending between texels")
    print("    - Transitions between texels are gradual (blurred edges)")
    print("    - Better for photographic textures but loses crispness")


def exercise_2():
    """
    Mipmap Visualization: Generate a mipmap chain, color-code each level,
    and render a textured floor to show which level is used at each distance.
    """
    tex = create_checkerboard(256, 8)
    mipmaps = generate_mipmaps(tex)

    print("Mipmap Chain Visualization")
    print(f"Base texture: {tex.shape[1]}x{tex.shape[0]}")
    print(f"Mipmap levels: {len(mipmaps)}")
    for i, m in enumerate(mipmaps):
        print(f"  Level {i}: {int(m.shape[1])}x{int(m.shape[0])}")
    print()

    # Create color-coded mipmaps (tint each level a different color)
    level_colors = [
        [1.0, 0.2, 0.2],   # Level 0: Red
        [0.2, 1.0, 0.2],   # Level 1: Green
        [0.2, 0.2, 1.0],   # Level 2: Blue
        [1.0, 1.0, 0.2],   # Level 3: Yellow
        [1.0, 0.2, 1.0],   # Level 4: Magenta
        [0.2, 1.0, 1.0],   # Level 5: Cyan
        [1.0, 0.5, 0.0],   # Level 6: Orange
        [0.5, 0.0, 1.0],   # Level 7: Purple
        [0.0, 0.5, 0.0],   # Level 8: Dark green
    ]

    # Render a floor with mipmap level visualization
    floor_w, floor_h = 400, 200
    floor_image = np.zeros((floor_h, floor_w, 3))

    for y in range(floor_h):
        for x in range(floor_w):
            screen_x = (x / floor_w - 0.5) * 2
            screen_y = (y / floor_h)
            if screen_y < 0.01:
                continue

            u = screen_x / screen_y * 2.0
            v_coord = 1.0 / screen_y

            # Estimate LOD
            lod = np.log2(max(abs(1.0 / screen_y) * 256 / floor_h, 1.0))
            lod = np.clip(lod, 0, len(mipmaps) - 1)
            level = int(np.round(lod))
            level = min(level, len(level_colors) - 1)

            # Color-code by mipmap level
            floor_image[y, x] = level_colors[level]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Show mipmap levels side by side
    combined = np.zeros((256, 400, 3))
    x_offset = 0
    for i, m in enumerate(mipmaps[:6]):
        h_m, w_m = int(m.shape[0]), int(m.shape[1])
        if x_offset + w_m > 400:
            break
        tinted = (m / 255.0 if m.max() > 1 else m) * 0.5
        color = np.array(level_colors[min(i, len(level_colors) - 1)])
        tinted = tinted + 0.5 * color
        combined[0:h_m, x_offset:x_offset + w_m] = np.clip(tinted, 0, 1)
        x_offset += w_m + 5

    axes[0].imshow(combined)
    axes[0].set_title('Mipmap Levels (color-coded)')
    axes[0].axis('off')

    axes[1].imshow(floor_image)
    axes[1].set_title('Floor: Mipmap Level Selection by Distance')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Distance (near at bottom)')
    plt.tight_layout()
    plt.savefig('ex06_mipmap_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved ex06_mipmap_visualization.png")
    print()
    print("Interpretation:")
    print("  Near (bottom): Level 0 (red) - full resolution")
    print("  Far (top): Higher levels (blue, yellow, ...) - progressively blurred")
    print("  The GPU selects the level that best matches the screen-space texel density")


def exercise_3():
    """
    Normal Map Creation: Given a heightmap, compute the normal map by taking
    the finite difference gradient in u and v directions.
    """
    # Create a synthetic heightmap (a bump/dome)
    size = 64
    heightmap = np.zeros((size, size), dtype=float)
    for y in range(size):
        for x in range(size):
            dx = (x - size / 2) / (size / 4)
            dy = (y - size / 2) / (size / 4)
            r2 = dx * dx + dy * dy
            heightmap[y, x] = max(0, 1.0 - r2) * 0.5  # Dome shape

    # Add some noise/bumps
    np.random.seed(42)
    for _ in range(5):
        cx, cy = np.random.randint(10, size - 10, 2)
        for y in range(size):
            for x in range(size):
                dx = (x - cx) / 5.0
                dy = (y - cy) / 5.0
                heightmap[y, x] += max(0, 0.3 - (dx * dx + dy * dy)) * 0.2

    print("Normal Map Creation from Heightmap")
    print(f"Heightmap size: {size}x{size}")
    print(f"Height range: [{heightmap.min():.3f}, {heightmap.max():.3f}]")
    print()

    # Compute normal map using finite differences
    normal_map = np.zeros((size, size, 3), dtype=float)
    strength = 2.0  # Normal map strength multiplier

    for y in range(size):
        for x in range(size):
            # Finite differences with wrapping
            h_left = heightmap[y, (x - 1) % size]
            h_right = heightmap[y, (x + 1) % size]
            h_up = heightmap[(y - 1) % size, x]
            h_down = heightmap[(y + 1) % size, x]

            # Tangent-space normal from height gradient
            # dh/du = (h_right - h_left) / 2
            # dh/dv = (h_down - h_up) / 2
            # normal = normalize(-dh/du, -dh/dv, 1)
            dhdx = (h_right - h_left) / 2.0 * strength
            dhdy = (h_down - h_up) / 2.0 * strength

            n = normalize(np.array([-dhdx, -dhdy, 1.0]))
            normal_map[y, x] = n

    # Encode to RGB [0, 1]: normal = color * 2 - 1
    normal_map_rgb = normal_map * 0.5 + 0.5

    print(f"Normal map computed using finite differences")
    print(f"  Strength multiplier: {strength}")
    print(f"  Flat area normal: (0, 0, 1) -> RGB ({0.5:.1f}, {0.5:.1f}, {1.0:.1f})")
    print()

    # Simple shading test: render a flat surface with the normal map
    render_size = 128
    shaded = np.zeros((render_size, render_size, 3))
    light_dir = normalize(np.array([1.0, 1.0, 1.0]))

    for y in range(render_size):
        for x in range(render_size):
            u = x / render_size
            v = y / render_size
            # Sample normal map
            nx_idx = int(u * size) % size
            ny_idx = int(v * size) % size
            n = normal_map[ny_idx, nx_idx]
            diff = max(np.dot(n, light_dir), 0.0)
            shaded[y, x] = [0.7, 0.5, 0.3] * (0.15 + 0.85 * diff)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(heightmap, cmap='gray', origin='lower')
    axes[0].set_title('Heightmap')
    axes[1].imshow(normal_map_rgb, origin='lower')
    axes[1].set_title('Normal Map (RGB encoded)')
    axes[2].imshow(shaded, origin='lower')
    axes[2].set_title('Shaded with Normal Map')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('ex06_normal_map.png', dpi=150, bbox_inches='tight')
    print("Saved ex06_normal_map.png")


def exercise_4():
    """
    UV Wrapping Modes: Render a quad with UV coordinates from (-1,-1) to (2,2)
    using all three wrapping modes.
    """
    tex = create_checkerboard(64, 4)
    out_size = 200

    modes = ['repeat', 'clamp', 'mirrored_repeat']
    images = {}

    print("UV Wrapping Modes Comparison")
    print(f"UV range: (-1, -1) to (2, 2)")
    print()

    for mode in modes:
        img = np.zeros((out_size, out_size, 3))
        for y in range(out_size):
            for x in range(out_size):
                # Map pixel to UV range [-1, 2]
                u = -1.0 + 3.0 * x / out_size
                v = -1.0 + 3.0 * y / out_size
                img[y, x] = sample_bilinear(tex, u, v, mode=mode)
        images[mode] = img

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, mode in zip(axes, modes):
        ax.imshow(images[mode])
        ax.set_title(f'{mode.replace("_", " ").title()}')
        ax.set_xlabel('U: -1 to 2')
        ax.set_ylabel('V: -1 to 2')
        # Draw unit square boundary
        px0 = int(1.0 / 3.0 * out_size)
        px1 = int(2.0 / 3.0 * out_size)
        rect = plt.Rectangle((px0, px0), px1 - px0, px1 - px0,
                              fill=False, edgecolor='red', linewidth=2, linestyle='--')
        ax.add_patch(rect)
    plt.suptitle('UV Wrapping Modes (red box = [0,1] range)', fontsize=14)
    plt.tight_layout()
    plt.savefig('ex06_wrapping_modes.png', dpi=150, bbox_inches='tight')
    print("Saved ex06_wrapping_modes.png")
    print()
    print("When each mode is appropriate:")
    print("  Repeat: Tiling surfaces (brick walls, floor tiles, fabric)")
    print("  Clamp: Edge should extend (skyboxes, UI textures)")
    print("  Mirrored Repeat: Seamless patterns without visible seams")


def exercise_5():
    """
    Perspective Correction: Render a textured quad with and without
    perspective-correct UV interpolation.
    """
    print("Perspective Correct UV Interpolation")
    print("=" * 50)
    print()
    print("This exercise demonstrates the same concept as Exercise 4 in Lesson 04.")
    print("When a textured surface is viewed in perspective, linear interpolation")
    print("of UV coordinates in screen space produces incorrect results.")
    print()
    print("Key insight:")
    print("  Screen-space: equal pixel steps != equal world-space steps")
    print("  Near parts of the surface are magnified, far parts are compressed")
    print()
    print("  Without correction: straight lines on the texture appear curved")
    print("  With correction: texture mapping is geometrically accurate")
    print()
    print("  The fix: interpolate attr/w and 1/w separately, then divide:")
    print("    attr_correct = (alpha*a0/w0 + beta*a1/w1 + gamma*a2/w2) /")
    print("                   (alpha/w0 + beta/w1 + gamma/w2)")
    print()
    print("  Modern GPUs do this automatically for all 'in' variables.")
    print("  See Lesson 04 Exercise 4 for the full rendering implementation.")


def exercise_6():
    """
    Environment Reflection: Implement a simple environment-mapped sphere
    using an equirectangular projection.
    """
    w, h = 300, 200

    # Create a simple synthetic environment (gradient sky + ground)
    env_w, env_h = 256, 128
    env_map = np.zeros((env_h, env_w, 3), dtype=float)
    for ey in range(env_h):
        for ex in range(env_w):
            # Sky gradient (top) to ground (bottom)
            t = ey / env_h
            if t < 0.45:
                # Sky
                env_map[ey, ex] = [0.3 + 0.4 * (1 - t), 0.5 + 0.3 * (1 - t), 0.8 + 0.2 * (1 - t)]
            elif t < 0.55:
                # Horizon
                env_map[ey, ex] = [0.8, 0.7, 0.6]
            else:
                # Ground
                env_map[ey, ex] = [0.2, 0.3 + 0.1 * np.sin(ex * 0.1), 0.15]

    def direction_to_equirect_uv(d):
        dx, dy, dz = normalize(d)
        u = 0.5 + np.arctan2(dz, dx) / (2 * np.pi)
        v = 0.5 - np.arcsin(np.clip(dy, -1, 1)) / np.pi
        return u, v

    def sample_env(direction, blur_level=0):
        u, v = direction_to_equirect_uv(direction)
        # Simulate blur by using box filter on the env map
        if blur_level > 0:
            # Average over a region
            samples = []
            for _ in range(max(1, int(blur_level * 8))):
                offset_u = np.random.normal(0, blur_level * 0.05)
                offset_v = np.random.normal(0, blur_level * 0.05)
                su, sv = wrap_uv(u + offset_u, v + offset_v, 'repeat')
                sx = int(su * env_w) % env_w
                sy = int(sv * env_h) % env_h
                samples.append(env_map[sy, sx])
            return np.mean(samples, axis=0)
        else:
            x = int(u * env_w) % env_w
            y = int(v * env_h) % env_h
            return env_map[y, x]

    print("Environment-Mapped Sphere with Varying Roughness")
    print("=" * 50)

    np.random.seed(42)
    roughness_values = [0.0, 0.3, 0.6, 0.9]
    images = []

    for roughness in roughness_values:
        image = np.zeros((h, w, 3))
        sphere_center = np.array([0.0, 0.0, -3.0])
        sphere_radius = 1.0
        view_pos = np.array([0.0, 0.0, 0.0])
        aspect = w / h

        for y in range(h):
            for x in range(w):
                u_coord = (2.0 * x / w - 1.0) * aspect
                v_coord = 1.0 - 2.0 * y / h
                ray_dir = normalize(np.array([u_coord, v_coord, -1.0]))

                oc = view_pos - sphere_center
                a = np.dot(ray_dir, ray_dir)
                b = 2.0 * np.dot(oc, ray_dir)
                c_val = np.dot(oc, oc) - sphere_radius ** 2
                disc = b * b - 4 * a * c_val

                if disc < 0:
                    # Sample background
                    eu, ev = direction_to_equirect_uv(ray_dir)
                    ex_i = int(eu * env_w) % env_w
                    ey_i = int(ev * env_h) % env_h
                    image[y, x] = env_map[ey_i, ex_i] * 0.5
                    continue

                t = (-b - np.sqrt(disc)) / (2 * a)
                if t < 0:
                    image[y, x] = [0.05, 0.05, 0.08]
                    continue

                hit = view_pos + t * ray_dir
                normal = normalize(hit - sphere_center)
                V = normalize(view_pos - hit)

                # Reflection vector
                R = 2.0 * np.dot(normal, V) * normal - V

                # Sample environment with blur based on roughness
                refl_color = sample_env(R, blur_level=roughness * 5)

                # Add Fresnel effect
                n_dot_v = max(np.dot(normal, V), 0.0)
                fresnel = 0.04 + 0.96 * (1.0 - n_dot_v) ** 5

                color = refl_color * fresnel
                # Tone map
                color = color / (color + 1.0)
                color = np.power(np.clip(color, 0, 1), 1.0 / 2.2)
                image[y, x] = color

        images.append(image)
        print(f"  Roughness {roughness:.1f}: rendered")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, img, r in zip(axes, images, roughness_values):
        ax.imshow(img)
        ax.set_title(f'Roughness = {r:.1f}')
        ax.axis('off')
    plt.suptitle('Environment-Mapped Sphere: Roughness Variation', fontsize=14)
    plt.tight_layout()
    plt.savefig('ex06_environment_reflection.png', dpi=150, bbox_inches='tight')
    print("\nSaved ex06_environment_reflection.png")
    print()
    print("Observation:")
    print("  Roughness 0.0: Sharp mirror reflection of the environment")
    print("  Roughness 0.3: Slightly blurred reflection (glossy)")
    print("  Roughness 0.6: Very blurred reflection (semi-matte)")
    print("  Roughness 0.9: Almost no visible reflection (matte)")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Bilinear vs Nearest ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Mipmap Visualization ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Normal Map Creation ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: UV Wrapping Modes ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Perspective Correction ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Environment Reflection ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
