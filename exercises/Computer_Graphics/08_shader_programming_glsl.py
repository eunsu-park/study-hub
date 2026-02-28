"""
Lesson 08 - Shader Programming (GLSL)
Topic: Computer Graphics

Exercises covering GLSL shader concepts adapted to Python: vector swizzling,
PBR (Cook-Torrance) shading, procedural textures with anti-aliasing,
toon/cel shading, Gaussian blur (separable), and heat distortion effects.
Since GLSL runs on the GPU, these exercises implement the equivalent per-pixel
computations in Python using numpy and matplotlib.
"""

import numpy as np

matplotlib_available = True
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib_available = False


# ═══════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════

def normalize(v):
    """Normalize a vector or array of vectors."""
    if v.ndim == 1:
        n = np.linalg.norm(v)
        return v / n if n > 1e-10 else v
    else:
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return v / norms


def dot_clamped(a, b):
    """Compute dot product clamped to [0, 1]."""
    return np.clip(np.sum(a * b, axis=-1), 0.0, 1.0)


def ray_sphere_intersect(origin, direction, center, radius):
    """
    Ray-sphere intersection.
    Returns (hit, t) where hit is bool and t is the distance along the ray.
    """
    oc = origin - center
    a = np.dot(direction, direction)
    b = 2.0 * np.dot(oc, direction)
    c = np.dot(oc, oc) - radius * radius
    disc = b * b - 4 * a * c
    if disc < 0:
        return False, 0.0
    t = (-b - np.sqrt(disc)) / (2 * a)
    if t < 0:
        t = (-b + np.sqrt(disc)) / (2 * a)
    return t > 0, t


def render_sphere(width, height, shade_func, **kwargs):
    """
    Render a sphere centered at origin using a custom shading function.
    The shade_func receives (N, V, L, uv) and returns RGB color.
    """
    image = np.zeros((height, width, 3), dtype=np.float32)
    aspect = width / height
    center = np.array([0.0, 0.0, 0.0])
    radius = 0.8
    eye = np.array([0.0, 0.0, 3.0])
    light_pos = kwargs.get("light_pos", np.array([2.0, 2.0, 3.0]))

    for y in range(height):
        for x in range(width):
            # NDC coordinates
            u = (2.0 * x / width - 1.0) * aspect
            v = 1.0 - 2.0 * y / height
            direction = normalize(np.array([u, v, -1.0]))

            hit, t = ray_sphere_intersect(eye, direction, center, radius)
            if hit:
                point = eye + t * direction
                normal = normalize(point - center)
                view_dir = normalize(eye - point)
                light_dir = normalize(light_pos - point)

                # Compute spherical UV
                theta = np.arctan2(normal[0], normal[2])
                phi = np.arcsin(np.clip(normal[1], -1, 1))
                uv = np.array([(theta / np.pi + 1.0) * 0.5, phi / np.pi + 0.5])

                color = shade_func(normal, view_dir, light_dir, uv, **kwargs)
                image[y, x] = np.clip(color, 0.0, 1.0)

    return image


# ═══════════════════════════════════════════════════════════════
# Exercise functions
# ═══════════════════════════════════════════════════════════════

def exercise_1():
    """
    Exercise 1: Swizzle Practice

    Implement GLSL-style swizzling in Python and verify the results.
    Given vec4 v = vec4(1.0, 2.0, 3.0, 4.0), compute various swizzle operations.
    Then render a visualization showing how different swizzle combinations
    produce different colors when applied to a gradient.
    """
    # GLSL vec4 components:
    # Index:  0    1    2    3
    # xyzw:   x    y    z    w
    # rgba:   r    g    b    a
    # stpq:   s    t    p    q

    v = np.array([1.0, 2.0, 3.0, 4.0])
    print(f"  v = vec4({v[0]}, {v[1]}, {v[2]}, {v[3]})")
    print()

    # Swizzle component mapping
    comp = {'x': 0, 'y': 1, 'z': 2, 'w': 3,
            'r': 0, 'g': 1, 'b': 2, 'a': 3,
            's': 0, 't': 1, 'p': 2, 'q': 3}

    def swizzle(vec, pattern):
        """Perform GLSL-style swizzle on a vector."""
        return np.array([vec[comp[c]] for c in pattern])

    # Test cases from the exercise
    test_cases = [
        ("v.wzyx",  "wzyx"),
        ("v.xxyy",  "xxyy"),
        ("v.rgb",   "rgb"),
        ("v.stp",   "stp"),
        ("v.xy",    "xy"),
        ("v.rrr",   "rrr"),
        ("v.bgr",   "bgr"),
        ("v.yyxx",  "yyxx"),
    ]

    print("  Swizzle Results:")
    print("  ─────────────────────────────────────────")
    for name, pattern in test_cases:
        result = swizzle(v, pattern)
        result_str = ", ".join(f"{x:.1f}" for x in result)
        vec_type = f"vec{len(result)}"
        print(f"    {name:10s} = {vec_type}({result_str})")
    print()

    # Verify specific answers
    print("  Verification:")
    assert np.allclose(swizzle(v, "wzyx"), [4, 3, 2, 1])
    print("    v.wzyx = vec4(4.0, 3.0, 2.0, 1.0) -- reversed order")
    assert np.allclose(swizzle(v, "xxyy"), [1, 1, 2, 2])
    print("    v.xxyy = vec4(1.0, 1.0, 2.0, 2.0) -- duplicated pairs")
    assert np.allclose(swizzle(v, "rgb"), [1, 2, 3])
    print("    v.rgb  = vec3(1.0, 2.0, 3.0) -- same as xyz")
    assert np.allclose(swizzle(v, "stp"), [1, 2, 3])
    print("    v.stp  = vec3(1.0, 2.0, 3.0) -- same as xyz (texture naming)")
    print()

    # Swizzle write demonstration
    print("  Swizzle Write (left-hand side):")
    v_copy = v.copy()
    # v.xy = vec2(5.0, 6.0)
    v_copy[comp['x']] = 5.0
    v_copy[comp['y']] = 6.0
    print(f"    v.xy = vec2(5.0, 6.0) -> v = vec4({v_copy[0]}, {v_copy[1]}, {v_copy[2]}, {v_copy[3]})")
    print()

    # Common shader patterns using swizzle
    print("  Common Shader Patterns:")
    position = np.array([1.5, 2.5, -3.0, 1.0])
    color = np.array([0.8, 0.2, 0.1, 1.0])

    # Extract xyz from vec4 (e.g., gl_Position.xyz)
    pos3d = swizzle(position, "xyz")
    print(f"    gl_Position.xyz = vec3({pos3d[0]}, {pos3d[1]}, {pos3d[2]})")

    # Luminance calculation using dot product with rgb swizzle
    lum_weights = np.array([0.299, 0.587, 0.114])
    rgb = swizzle(color, "rgb")
    luminance = np.dot(rgb, lum_weights)
    print(f"    luminance = dot(color.rgb, vec3(0.299, 0.587, 0.114)) = {luminance:.4f}")

    # Visualization: render a gradient and show different swizzle orderings
    if matplotlib_available:
        size = 128
        # Create a gradient image where R increases left->right, G increases top->bottom
        img = np.zeros((size, size, 4), dtype=np.float32)
        for y in range(size):
            for x in range(size):
                img[y, x] = [x / size, y / size, 0.5, 1.0]

        swizzle_patterns = [
            ("rgba (original)", "rgba"),
            ("bgra (swap R/B)", "bgra"),
            ("rrra (red only)", "rrra"),
            ("ggga (green only)", "ggga"),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i, (label, pattern) in enumerate(swizzle_patterns):
            swizzled = np.zeros((size, size, 4), dtype=np.float32)
            for y in range(size):
                for x in range(size):
                    s = swizzle(img[y, x], pattern)
                    swizzled[y, x] = s
            axes[i].imshow(swizzled[:, :, :3])
            axes[i].set_title(label, fontsize=10)
            axes[i].axis('off')

        plt.suptitle("Swizzle Visualization: RGBA Gradient", fontsize=13)
        plt.tight_layout()
        plt.savefig("08_ex1_swizzle_practice.png", dpi=100)
        plt.close()
        print("\n  Saved: 08_ex1_swizzle_practice.png")


def exercise_2():
    """
    Exercise 2: Phong to PBR Upgrade

    Implement both Phong (Blinn-Phong) and PBR (Cook-Torrance) shading
    in Python, simulating the GLSL upgrade path. Compare the visual results
    of the two models with varying roughness and metallic parameters.
    """
    # ── Phong/Blinn-Phong shading (from lesson 05/08) ──
    def shade_phong(N, V, L, uv, **kwargs):
        """Blinn-Phong shading model."""
        albedo = kwargs.get("albedo", np.array([0.8, 0.3, 0.2]))
        shininess = kwargs.get("shininess", 64.0)
        ambient_strength = 0.1
        specular_strength = 0.5

        H = normalize(L + V)
        NdotL = max(np.dot(N, L), 0.0)
        NdotH = max(np.dot(N, H), 0.0)

        ambient = ambient_strength * albedo
        diffuse = NdotL * albedo
        specular = specular_strength * (NdotH ** shininess) * np.ones(3)

        return ambient + diffuse + specular

    # ── PBR Cook-Torrance shading ──
    def distribution_ggx(NdotH, roughness):
        """GGX/Trowbridge-Reitz normal distribution function."""
        a = roughness * roughness
        a2 = a * a
        denom = NdotH * NdotH * (a2 - 1.0) + 1.0
        return a2 / (np.pi * denom * denom + 1e-10)

    def fresnel_schlick(cos_theta, F0):
        """Schlick approximation of Fresnel reflectance."""
        return F0 + (1.0 - F0) * (1.0 - cos_theta) ** 5

    def geometry_schlick_ggx(NdotV, roughness):
        """Schlick-GGX geometry function for a single direction."""
        r = roughness + 1.0
        k = (r * r) / 8.0
        return NdotV / (NdotV * (1.0 - k) + k + 1e-10)

    def geometry_smith(NdotV, NdotL, roughness):
        """Smith geometry function (combines shadowing and masking)."""
        ggx1 = geometry_schlick_ggx(NdotV, roughness)
        ggx2 = geometry_schlick_ggx(NdotL, roughness)
        return ggx1 * ggx2

    def shade_pbr(N, V, L, uv, **kwargs):
        """PBR Cook-Torrance BRDF shading model."""
        albedo = kwargs.get("albedo", np.array([0.8, 0.3, 0.2]))
        roughness = kwargs.get("roughness", 0.5)
        metallic = kwargs.get("metallic", 0.0)

        H = normalize(L + V)
        NdotL = max(np.dot(N, L), 0.0)
        NdotV = max(np.dot(N, V), 0.0)
        NdotH = max(np.dot(N, H), 0.0)
        HdotV = max(np.dot(H, V), 0.0)

        # Dielectric F0 = 0.04, metallic F0 = albedo
        F0 = 0.04 * (1.0 - metallic) + albedo * metallic

        # Cook-Torrance BRDF terms
        D = distribution_ggx(NdotH, roughness)
        F = fresnel_schlick(HdotV, F0)
        G = geometry_smith(NdotV, NdotL, roughness)

        # Specular BRDF
        numerator = D * F * G
        denominator = 4.0 * NdotV * NdotL + 1e-4
        specular = numerator / denominator

        # Energy conservation: diffuse fraction = 1 - specular
        kS = F
        kD = (1.0 - kS) * (1.0 - metallic)

        # Combine
        diffuse = kD * albedo / np.pi
        Lo = (diffuse + specular) * NdotL

        # Ambient
        ambient = 0.03 * albedo
        color = ambient + Lo

        # Reinhard tone mapping
        color = color / (color + 1.0)
        # Gamma correction
        color = np.power(np.clip(color, 0, 1), 1.0 / 2.2)

        return color

    # Render comparisons
    size = 128
    print("  Rendering Phong vs PBR comparison spheres...")
    print()

    # Compare Phong with different shininess vs PBR with different roughness
    comparisons = [
        ("Phong (shininess=16)",  shade_phong, {"shininess": 16.0}),
        ("Phong (shininess=128)", shade_phong, {"shininess": 128.0}),
        ("PBR (rough=0.8, met=0.0)", shade_pbr, {"roughness": 0.8, "metallic": 0.0}),
        ("PBR (rough=0.3, met=0.0)", shade_pbr, {"roughness": 0.3, "metallic": 0.0}),
        ("PBR (rough=0.3, met=1.0)", shade_pbr, {"roughness": 0.3, "metallic": 1.0}),
        ("PBR (rough=0.1, met=1.0)", shade_pbr, {"roughness": 0.1, "metallic": 1.0}),
    ]

    if matplotlib_available:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for idx, (label, shade_fn, params) in enumerate(comparisons):
        albedo = np.array([0.8, 0.3, 0.2])
        img = render_sphere(size, size, shade_fn, albedo=albedo, **params)
        print(f"  [{idx + 1}] {label}")

        if matplotlib_available:
            ax = axes[idx // 3, idx % 3]
            ax.imshow(img)
            ax.set_title(label, fontsize=9)
            ax.axis('off')

    if matplotlib_available:
        plt.suptitle("Phong vs PBR Shading Comparison", fontsize=13)
        plt.tight_layout()
        plt.savefig("08_ex2_phong_to_pbr.png", dpi=100)
        plt.close()
        print("\n  Saved: 08_ex2_phong_to_pbr.png")

    print()
    print("  Key differences between Phong and PBR:")
    print("  - Phong: shininess parameter is arbitrary (no physical meaning)")
    print("  - PBR: roughness (0-1) and metallic (0-1) have physical interpretation")
    print("  - PBR: energy-conserving (diffuse + specular <= incoming light)")
    print("  - PBR: Fresnel effect at grazing angles (edges brighter)")
    print("  - PBR: metals have colored specular, dielectrics have white specular")


def exercise_3():
    """
    Exercise 3: Procedural Texture

    Generate a procedural checkerboard pattern using only screen-space coordinates
    and mathematical functions (step/mod for hard edges, smoothstep for anti-aliased
    edges). This mirrors what a GLSL fragment shader would compute using
    gl_FragCoord and built-in functions.
    """
    width, height = 256, 256

    # GLSL built-in function equivalents
    def step(edge, x):
        """GLSL step(): returns 0.0 if x < edge, 1.0 otherwise."""
        return np.where(x < edge, 0.0, 1.0)

    def smoothstep(edge0, edge1, x):
        """GLSL smoothstep(): smooth Hermite interpolation."""
        t = np.clip((x - edge0) / (edge1 - edge0 + 1e-10), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def glsl_mod(x, y):
        """GLSL mod(): always positive modulo."""
        return x - y * np.floor(x / y)

    # Generate screen-space coordinates (gl_FragCoord equivalent)
    x_coords = np.arange(width, dtype=np.float32) + 0.5
    y_coords = np.arange(height, dtype=np.float32) + 0.5
    frag_x, frag_y = np.meshgrid(x_coords, y_coords)

    checker_size = 32.0  # pixels per checker square

    # ── Method 1: Hard-edged checkerboard using step() and mod() ──
    print("  Method 1: Hard-edged checkerboard (step + mod)")
    cx = np.floor(frag_x / checker_size)
    cy = np.floor(frag_y / checker_size)
    checker_hard = glsl_mod(cx + cy, 2.0)
    print(f"    Formula: mod(floor(x/{checker_size:.0f}) + floor(y/{checker_size:.0f}), 2.0)")

    # ── Method 2: Anti-aliased checkerboard using smoothstep() ──
    print("\n  Method 2: Anti-aliased checkerboard (smoothstep)")
    # Compute distance to nearest checker edge using fract
    frac_x = glsl_mod(frag_x / checker_size, 1.0)
    frac_y = glsl_mod(frag_y / checker_size, 1.0)

    # Anti-aliasing width in UV space (proportional to 1 pixel)
    aa_width = 1.0 / checker_size  # ~1 pixel in UV space

    # Create smooth transitions at checker edges
    sx = smoothstep(0.5 - aa_width, 0.5 + aa_width, frac_x)
    sx = sx * 2.0 - 1.0  # remap to [-1, 1]
    sy = smoothstep(0.5 - aa_width, 0.5 + aa_width, frac_y)
    sy = sy * 2.0 - 1.0

    checker_smooth = (sx * sy + 1.0) * 0.5

    # ── Method 3: Checkerboard with colored squares ──
    print("\n  Method 3: Colored checkerboard")
    color_a = np.array([0.9, 0.85, 0.7])   # warm white
    color_b = np.array([0.15, 0.15, 0.2])   # dark blue-gray
    checker_colored = np.zeros((height, width, 3), dtype=np.float32)
    for c in range(3):
        # mix(colorA, colorB, checker) in GLSL
        checker_colored[:, :, c] = color_a[c] * (1.0 - checker_hard) + color_b[c] * checker_hard

    # ── Method 4: Multi-frequency procedural pattern ──
    print("\n  Method 4: Multi-frequency procedural pattern (3 octaves)")
    pattern = np.zeros((height, width), dtype=np.float32)
    for octave, (freq, amplitude) in enumerate([(32.0, 0.5), (16.0, 0.3), (8.0, 0.2)]):
        cx_o = np.floor(frag_x / freq)
        cy_o = np.floor(frag_y / freq)
        pattern += amplitude * glsl_mod(cx_o + cy_o, 2.0)
        print(f"    Octave {octave}: freq={freq:.0f}px, amplitude={amplitude:.1f}")

    if matplotlib_available:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(checker_hard, cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title("Hard-edged (step + mod)")

        axes[0, 1].imshow(checker_smooth, cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title("Anti-aliased (smoothstep)")

        axes[1, 0].imshow(checker_colored)
        axes[1, 0].set_title("Colored Checkerboard")

        axes[1, 1].imshow(pattern, cmap='inferno', vmin=0, vmax=1)
        axes[1, 1].set_title("Multi-frequency Pattern")

        for ax in axes.flat:
            ax.set_xlabel("gl_FragCoord.x")
            ax.set_ylabel("gl_FragCoord.y")

        plt.suptitle("Procedural Textures (GLSL Fragment Shader Concepts)", fontsize=13)
        plt.tight_layout()
        plt.savefig("08_ex3_procedural_texture.png", dpi=100)
        plt.close()
        print("\n  Saved: 08_ex3_procedural_texture.png")

    # Show zoom comparison for anti-aliasing
    print("\n  Anti-aliasing comparison (zoomed to edge region):")
    edge_y = int(checker_size)  # row at checker boundary
    row_hard = checker_hard[edge_y, :]
    row_smooth = checker_smooth[edge_y, :]

    transition_region = slice(int(checker_size * 0.5 - 3), int(checker_size * 0.5 + 4))
    print(f"    Hard edge values at y={edge_y}:   {row_hard[transition_region]}")
    print(f"    Smooth edge values at y={edge_y}: "
          f"{np.array2string(row_smooth[transition_region], precision=2)}")
    print("    Smooth version has gradual transition (anti-aliased).")


def exercise_4():
    """
    Exercise 4: Toon Shading (Cel Shading)

    Implement a cel/toon shader that quantizes diffuse lighting into discrete
    bands and adds a black outline by detecting silhouette edges where
    dot(N, V) is close to 0.
    """
    size = 200

    def shade_toon(N, V, L, uv, **kwargs):
        """
        Toon/cel shading: quantize diffuse into discrete bands
        and add a silhouette outline.
        """
        albedo = kwargs.get("albedo", np.array([0.2, 0.6, 0.9]))
        num_bands = kwargs.get("num_bands", 4)
        outline_threshold = kwargs.get("outline_threshold", 0.25)
        outline_softness = kwargs.get("outline_softness", 0.05)

        NdotL = max(np.dot(N, L), 0.0)
        NdotV = max(np.dot(N, V), 0.0)

        # Quantize diffuse lighting into discrete bands
        # In GLSL: float band = floor(NdotL * numBands) / numBands;
        band = np.floor(NdotL * num_bands) / num_bands

        # Apply quantized lighting to albedo
        color = albedo * (0.2 + 0.8 * band)  # ambient floor + quantized diffuse

        # Silhouette outline detection
        # When N is nearly perpendicular to V (silhouette edge), darken to black
        # In GLSL: float outline = smoothstep(threshold - softness, threshold + softness, NdotV);
        outline = np.clip(
            (NdotV - (outline_threshold - outline_softness))
            / (2 * outline_softness + 1e-10),
            0.0, 1.0
        )
        # Hermite smoothstep
        outline = outline * outline * (3.0 - 2.0 * outline)

        # Mix color with black based on outline factor
        color = color * outline

        return color

    print("  Rendering toon-shaded spheres with different band counts...")
    print()

    band_configs = [
        {"label": "2 bands", "num_bands": 2, "outline_threshold": 0.3},
        {"label": "3 bands", "num_bands": 3, "outline_threshold": 0.25},
        {"label": "4 bands", "num_bands": 4, "outline_threshold": 0.25},
        {"label": "8 bands (smooth-ish)", "num_bands": 8, "outline_threshold": 0.2},
    ]

    if matplotlib_available:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for idx, config in enumerate(band_configs):
        img = render_sphere(size, size, shade_toon,
                            albedo=np.array([0.2, 0.6, 0.9]),
                            num_bands=config["num_bands"],
                            outline_threshold=config["outline_threshold"],
                            outline_softness=0.05)

        # Count distinct brightness levels in the rendered sphere
        mask = np.any(img > 0.01, axis=2)
        if np.any(mask):
            brightness = np.mean(img[mask], axis=1)
            quantized = np.round(brightness * config["num_bands"]) / config["num_bands"]
            unique_levels = len(np.unique(quantized))
        else:
            unique_levels = 0

        print(f"  [{idx + 1}] {config['label']:25s} | outline={config['outline_threshold']:.2f} | "
              f"~{unique_levels} distinct brightness levels")

        if matplotlib_available:
            axes[idx].imshow(img)
            axes[idx].set_title(config["label"], fontsize=10)
            axes[idx].axis('off')

    if matplotlib_available:
        plt.suptitle("Toon/Cel Shading with Silhouette Outlines", fontsize=13)
        plt.tight_layout()
        plt.savefig("08_ex4_toon_shading.png", dpi=100)
        plt.close()
        print("\n  Saved: 08_ex4_toon_shading.png")

    print()
    print("  GLSL implementation notes:")
    print("  - Quantization: band = floor(NdotL * N_BANDS) / N_BANDS;")
    print("  - Outline: smoothstep(threshold - soft, threshold + soft, dot(N, V));")
    print("  - The outline is a screen-space approximation of silhouette detection")
    print("  - More accurate outlines use edge detection on depth/normal buffers")
    print("    in a post-processing pass (Sobel filter on G-buffer normals)")


def exercise_5():
    """
    Exercise 5: Gaussian Blur (Separable Two-Pass)

    Implement the separable Gaussian blur from the lesson. A 2D Gaussian blur
    can be decomposed into two 1D passes (horizontal then vertical), reducing
    complexity from O(r^2) to O(2r) samples per pixel. This mirrors the
    WebGL multi-pass rendering approach using framebuffer objects.
    """
    # Create a test image to blur (a scene with sharp features)
    size = 256
    image = np.zeros((size, size, 3), dtype=np.float32)

    # Draw some geometric shapes
    # Red circle
    cy, cx = np.ogrid[:size, :size]
    circle1 = ((cx - 80) ** 2 + (cy - 80) ** 2) < 30 ** 2
    image[circle1] = [0.9, 0.2, 0.1]

    # Green rectangle
    image[120:180, 100:200] = [0.1, 0.8, 0.2]

    # Blue triangle
    for y in range(50, 120):
        half_width = int((120 - y) * 0.8)
        center_x = 200
        x_start = max(0, center_x - half_width)
        x_end = min(size, center_x + half_width)
        image[y, x_start:x_end] = [0.1, 0.3, 0.9]

    # White dots (noise/stars)
    np.random.seed(42)
    for _ in range(30):
        rx, ry = np.random.randint(0, size, 2)
        image[ry, rx] = [1.0, 1.0, 1.0]

    # Background gradient
    for y in range(size):
        for x in range(size):
            if np.all(image[y, x] == 0):
                image[y, x] = [0.05 + 0.1 * y / size,
                               0.05 + 0.05 * x / size,
                               0.1 + 0.1 * y / size]

    def compute_gaussian_weights(radius, sigma=None):
        """
        Compute normalized Gaussian kernel weights.
        In the GLSL lesson, these are precomputed as float weights[5].
        """
        if sigma is None:
            sigma = radius / 2.5
        weights = np.array([np.exp(-0.5 * (i / sigma) ** 2) for i in range(radius + 1)])
        # Normalize so weights sum to 1 (center + 2 * sides)
        total = weights[0] + 2.0 * np.sum(weights[1:])
        weights /= total
        return weights

    def blur_pass(img, direction, weights):
        """
        Single-direction blur pass (simulates the GLSL blur fragment shader).
        direction: (1, 0) for horizontal, (0, 1) for vertical.
        """
        h, w = img.shape[:2]
        result = np.zeros_like(img)
        radius = len(weights) - 1

        dx, dy = direction

        for y in range(h):
            for x in range(w):
                # Center sample
                color = img[y, x] * weights[0]

                # Symmetric samples (same as GLSL for loop)
                for i in range(1, radius + 1):
                    # Sample in positive direction
                    sx1 = min(max(x + i * dx, 0), w - 1)
                    sy1 = min(max(y + i * dy, 0), h - 1)
                    color += img[sy1, sx1] * weights[i]

                    # Sample in negative direction
                    sx2 = min(max(x - i * dx, 0), w - 1)
                    sy2 = min(max(y - i * dy, 0), h - 1)
                    color += img[sy2, sx2] * weights[i]

                result[y, x] = color

        return result

    # Apply separable Gaussian blur at different radii
    radii = [2, 5, 10]
    results = [image.copy()]  # original
    labels = ["Original"]

    print("  Gaussian Blur - Separable Two-Pass Implementation")
    print("  ═════════════════════════════════════════════════")
    print()
    print("  Gaussian kernel weights (from GLSL lesson):")
    for r in radii:
        weights = compute_gaussian_weights(r)
        w_str = ", ".join(f"{w:.6f}" for w in weights)
        print(f"    radius={r:2d}: [{w_str}]")
    print()

    print("  Complexity comparison:")
    for r in radii:
        naive_samples = (2 * r + 1) ** 2
        separable_samples = 2 * (2 * r + 1)
        savings = 1.0 - separable_samples / naive_samples
        print(f"    radius={r:2d}: naive={naive_samples:5d} samples, "
              f"separable={separable_samples:3d} samples ({savings * 100:.0f}% fewer)")
    print()

    print("  Rendering blur passes...")
    for r in radii:
        weights = compute_gaussian_weights(r)

        # Pass 1: Horizontal blur (input -> intermediate)
        # In WebGL: render to FBO with direction=(1,0)
        horizontal = blur_pass(image, (1, 0), weights)

        # Pass 2: Vertical blur (intermediate -> output)
        # In WebGL: render to screen with direction=(0,1)
        final = blur_pass(horizontal, (0, 1), weights)

        results.append(final)
        labels.append(f"Radius = {r}")
        print(f"    Blur radius={r}: done (kernel size={2 * r + 1})")

    if matplotlib_available:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i, (img_data, label) in enumerate(zip(results, labels)):
            axes[i].imshow(np.clip(img_data, 0, 1))
            axes[i].set_title(label)
            axes[i].axis('off')

        plt.suptitle("Separable Gaussian Blur (Two-Pass, like WebGL FBO Pipeline)", fontsize=12)
        plt.tight_layout()
        plt.savefig("08_ex5_gaussian_blur.png", dpi=100)
        plt.close()
        print("\n  Saved: 08_ex5_gaussian_blur.png")

    print()
    print("  WebGL multi-pass pipeline:")
    print("    1. Render scene to FBO (color texture)")
    print("    2. Bind color texture, render fullscreen quad with horizontal blur shader")
    print("    3. Bind horizontal result, render fullscreen quad with vertical blur shader")
    print("    4. Output: blurred scene on screen")


def exercise_6():
    """
    Exercise 6: Heat Distortion (Post-Processing UV Offset)

    Create a post-processing shader effect that distorts the scene by
    offsetting UV coordinates based on a scrolling noise pattern.
    This simulates the "heat haze" or "underwater" effect from the exercise.
    In GLSL, this would use sin() and a noise texture to perturb the
    texture sampling coordinates.
    """
    width, height = 256, 256

    # Create a base scene image (a landscape-like gradient with some shapes)
    scene = np.zeros((height, width, 3), dtype=np.float32)

    # Sky gradient
    for y in range(height // 2):
        t = y / (height / 2)
        scene[y, :] = [0.4 - 0.3 * t, 0.6 - 0.2 * t, 0.9 - 0.1 * t]

    # Ground
    for y in range(height // 2, height):
        t = (y - height // 2) / (height / 2)
        scene[y, :] = [0.3 + 0.2 * t, 0.5 - 0.2 * t, 0.15 + 0.1 * t]

    # Sun
    cy_arr, cx_arr = np.ogrid[:height, :width]
    sun = ((cx_arr - 200) ** 2 + (cy_arr - 40) ** 2) < 25 ** 2
    scene[sun] = [1.0, 0.9, 0.5]

    # Buildings (rectangles)
    scene[80:128, 30:55] = [0.3, 0.3, 0.35]
    scene[60:128, 60:80] = [0.25, 0.25, 0.3]
    scene[90:128, 90:120] = [0.35, 0.3, 0.3]
    scene[70:128, 150:175] = [0.3, 0.28, 0.32]

    # Horizon line
    scene[127:130, :] = [0.4, 0.35, 0.25]

    def generate_noise_texture(w, h, scale=1.0, seed=0):
        """
        Generate a simple 2D noise texture (value noise).
        In a real WebGL app, this would be a preloaded texture.
        """
        rng = np.random.RandomState(seed)
        # Low-resolution noise, upsampled with bilinear interpolation
        small_w, small_h = max(w // 8, 1), max(h // 8, 1)
        small_noise = rng.rand(small_h, small_w).astype(np.float32)

        # Bilinear upsample
        noise = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                u = x / w * (small_w - 1)
                v = y / h * (small_h - 1)
                x0, y0 = int(u), int(v)
                x1 = min(x0 + 1, small_w - 1)
                y1 = min(y0 + 1, small_h - 1)
                fx, fy = u - x0, v - y0
                noise[y, x] = (
                    small_noise[y0, x0] * (1 - fx) * (1 - fy) +
                    small_noise[y0, x1] * fx * (1 - fy) +
                    small_noise[y1, x0] * (1 - fx) * fy +
                    small_noise[y1, x1] * fx * fy
                )
        return noise * scale

    noise_tex = generate_noise_texture(width, height, scale=1.0, seed=42)

    def apply_heat_distortion(scene_img, noise, time, strength=0.02, speed=1.0,
                              frequency=10.0):
        """
        Post-processing heat distortion shader.

        GLSL equivalent:
            vec2 uv = vTexCoord;
            float noise_val = texture(uNoiseTexture, uv * frequency + vec2(0, uTime * speed)).r;
            vec2 offset = vec2(sin(uv.y * freq + uTime) * strength,
                               cos(uv.x * freq + uTime * 0.7) * strength * 0.5);
            offset += (noise_val - 0.5) * strength;
            vec4 color = texture(uSceneTexture, uv + offset);
        """
        h, w = scene_img.shape[:2]
        result = np.zeros_like(scene_img)

        for y in range(h):
            for x in range(w):
                # Normalized coordinates (0 to 1)
                u = x / w
                v = y / h

                # Scrolling noise lookup (simulates uTime scroll)
                noise_u = int((u * frequency * w) % w)
                noise_v = int(((v * frequency + time * speed) * h) % h)
                noise_val = noise[noise_v % h, noise_u % w]

                # Compute UV offset using sin/cos (heat shimmer)
                offset_x = np.sin(v * frequency * 2 * np.pi + time) * strength
                offset_y = np.cos(u * frequency * 2 * np.pi + time * 0.7) * strength * 0.5

                # Add noise-based perturbation
                offset_x += (noise_val - 0.5) * strength * 2
                offset_y += (noise_val - 0.5) * strength

                # Apply distortion
                src_x = int(np.clip((u + offset_x) * w, 0, w - 1))
                src_y = int(np.clip((v + offset_y) * h, 0, h - 1))

                result[y, x] = scene_img[src_y, src_x]

        return result

    # Render multiple frames at different time values
    print("  Heat Distortion Post-Processing Effect")
    print("  ═══════════════════════════════════════")
    print()
    print("  GLSL shader concept:")
    print("    1. Render scene to FBO (color texture)")
    print("    2. In post-process pass, sample scene texture with offset UVs")
    print("    3. Offsets computed from sin()/cos() + noise texture + uTime")
    print()

    distortion_configs = [
        {"label": "No distortion",       "strength": 0.0,  "time": 0.0},
        {"label": "Subtle heat haze",    "strength": 0.01, "time": 1.0},
        {"label": "Medium distortion",   "strength": 0.025, "time": 2.0},
        {"label": "Strong underwater",   "strength": 0.05, "time": 3.0},
    ]

    if matplotlib_available:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for idx, config in enumerate(distortion_configs):
        if config["strength"] == 0:
            result = scene.copy()
        else:
            result = apply_heat_distortion(
                scene, noise_tex,
                time=config["time"],
                strength=config["strength"],
                speed=1.0,
                frequency=8.0
            )

        # Compute distortion magnitude statistics
        if config["strength"] > 0:
            diff = np.abs(result.astype(float) - scene.astype(float))
            mean_diff = np.mean(diff)
            max_diff = np.max(diff)
            print(f"  [{idx + 1}] {config['label']:25s} | strength={config['strength']:.3f} | "
                  f"mean_diff={mean_diff:.4f}, max_diff={max_diff:.4f}")
        else:
            print(f"  [{idx + 1}] {config['label']:25s} | (reference)")

        if matplotlib_available:
            ax = axes[idx // 2, idx % 2]
            ax.imshow(np.clip(result, 0, 1))
            ax.set_title(config["label"], fontsize=11)
            ax.axis('off')

    if matplotlib_available:
        plt.suptitle("Heat Distortion Post-Processing (UV Offset Shader)", fontsize=13)
        plt.tight_layout()
        plt.savefig("08_ex6_heat_distortion.png", dpi=100)
        plt.close()
        print("\n  Saved: 08_ex6_heat_distortion.png")

    print()
    print("  Implementation details:")
    print("    - The distortion uses two components:")
    print("      (a) Deterministic: sin/cos waves scrolling with time")
    print("      (b) Stochastic: noise texture lookup for organic variation")
    print("    - Strength controls maximum UV offset (in normalized coordinates)")
    print("    - In WebGL, the noise texture would be a tileable Perlin noise")
    print("    - For best results, use separable blur on the distorted result")
    print("      to simulate refraction scattering (heat mirages are slightly blurry)")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Swizzle Practice", exercise_1),
        ("Exercise 2: Phong to PBR Upgrade", exercise_2),
        ("Exercise 3: Procedural Texture", exercise_3),
        ("Exercise 4: Toon Shading", exercise_4),
        ("Exercise 5: Gaussian Blur", exercise_5),
        ("Exercise 6: Heat Distortion", exercise_6),
    ]

    for title, func in exercises:
        print(f"\n{'=' * 60}")
        print(f" {title}")
        print(f"{'=' * 60}")
        func()

    print(f"\n{'=' * 60}")
    print(" All exercises completed!")
    print(f"{'=' * 60}")
