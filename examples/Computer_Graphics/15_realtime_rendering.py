"""
Real-Time Rendering Techniques
================================

Implements core real-time rendering techniques in a software simulation:
  1. Deferred rendering G-buffer generation
  2. Shadow mapping with depth-buffer shadow test
  3. Screen-Space Ambient Occlusion (SSAO)
  4. HDR tone mapping (Reinhard and ACES)
  5. Bloom post-processing via Gaussian blur on bright pixels

These techniques are the backbone of every modern game engine.  Each one
solves a specific problem: deferred rendering decouples geometry from
lighting, shadow maps add directional shadows, SSAO approximates contact
shadows, and HDR/bloom handle realistic light intensity.

Dependencies: numpy, matplotlib, scipy (optional, falls back to manual blur)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# ---------------------------------------------------------------------------
# 1. G-Buffer -- the foundation of deferred rendering
# ---------------------------------------------------------------------------


class GBuffer:
    """A Geometry Buffer storing per-pixel surface attributes.

    Why a G-buffer?  In forward rendering, each object evaluates ALL
    lights in its fragment shader.  With 100 lights and 1000 objects,
    that's 100k light evaluations -- many wasted on occluded fragments.

    Deferred rendering splits the work:
      Pass 1 (geometry): write surface attributes to the G-buffer
      Pass 2 (lighting): read the G-buffer and evaluate only visible pixels

    Cost becomes: geometry_pass(objects) + lighting_pass(lights * pixels).
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # Each G-buffer channel stores one surface attribute
        self.albedo = np.zeros((height, width, 3), dtype=float)
        self.normal = np.zeros((height, width, 3), dtype=float)
        self.depth = np.full((height, width), np.inf, dtype=float)
        self.position = np.zeros((height, width, 3), dtype=float)

    def clear(self):
        self.albedo[:] = 0
        self.normal[:] = 0
        self.depth[:] = np.inf
        self.position[:] = 0


def generate_synthetic_gbuffer(width: int = 256,
                               height: int = 256) -> GBuffer:
    """Generate a synthetic G-buffer with procedural geometry.

    We simulate a scene with a ground plane, two spheres, and a box
    by analytically computing normals, depth, and albedo for each pixel.
    This avoids the need for a full rasterizer while demonstrating
    the G-buffer concept.
    """
    gb = GBuffer(width, height)

    # Camera setup
    aspect = width / height
    fov_half = np.tan(np.radians(30))

    for y in range(height):
        for x in range(width):
            # Ray direction from camera
            u = (2 * (x + 0.5) / width - 1) * fov_half * aspect
            v = (1 - 2 * (y + 0.5) / height) * fov_half
            ray_dir = np.array([u, v, -1.0])
            ray_dir /= np.linalg.norm(ray_dir)
            ray_org = np.array([0.0, 2.0, 5.0])

            best_t = np.inf
            best_normal = np.zeros(3)
            best_albedo = np.zeros(3)
            best_pos = np.zeros(3)

            # Ground plane at y=0
            if abs(ray_dir[1]) > 1e-8:
                t_plane = (0 - ray_org[1]) / ray_dir[1]
                if 0.01 < t_plane < best_t:
                    p = ray_org + t_plane * ray_dir
                    if abs(p[0]) < 8 and abs(p[2]) < 15:
                        best_t = t_plane
                        best_normal = np.array([0, 1, 0])
                        checker = (int(np.floor(p[0])) + int(np.floor(p[2]))) % 2
                        best_albedo = (np.array([0.6, 0.6, 0.55]) if checker
                                       else np.array([0.3, 0.3, 0.28]))
                        best_pos = p

            # Sphere 1 (left)
            c1 = np.array([-1.5, 1.0, -2.0])
            oc = ray_org - c1
            a = np.dot(ray_dir, ray_dir)
            b = 2 * np.dot(oc, ray_dir)
            c = np.dot(oc, oc) - 1.0
            disc = b * b - 4 * a * c
            if disc >= 0:
                t_sphere = (-b - np.sqrt(disc)) / (2 * a)
                if 0.01 < t_sphere < best_t:
                    p = ray_org + t_sphere * ray_dir
                    best_t = t_sphere
                    best_normal = (p - c1) / np.linalg.norm(p - c1)
                    best_albedo = np.array([0.8, 0.2, 0.2])
                    best_pos = p

            # Sphere 2 (right)
            c2 = np.array([1.8, 0.8, -3.0])
            oc = ray_org - c2
            b = 2 * np.dot(oc, ray_dir)
            c = np.dot(oc, oc) - 0.8 * 0.8
            disc = b * b - 4 * a * c
            if disc >= 0:
                t_sphere = (-b - np.sqrt(disc)) / (2 * a)
                if 0.01 < t_sphere < best_t:
                    p = ray_org + t_sphere * ray_dir
                    best_t = t_sphere
                    best_normal = (p - c2) / np.linalg.norm(p - c2)
                    best_albedo = np.array([0.2, 0.5, 0.8])
                    best_pos = p

            if best_t < np.inf:
                gb.depth[y, x] = best_t
                gb.normal[y, x] = best_normal
                gb.albedo[y, x] = best_albedo
                gb.position[y, x] = best_pos

    return gb


# ---------------------------------------------------------------------------
# 2. Deferred Lighting Pass
# ---------------------------------------------------------------------------


def deferred_lighting(gb: GBuffer,
                      light_positions: list,
                      light_colors: list) -> np.ndarray:
    """Evaluate lighting for all pixels using the G-buffer.

    Why is this efficient?  We only shade pixels that have actual geometry
    (depth != inf).  Each light reads from the same G-buffer -- no need
    to re-render geometry.  Adding more lights is cheap: just one more
    screen-space pass per light.
    """
    h, w = gb.depth.shape
    output = np.zeros((h, w, 3), dtype=float)
    mask = gb.depth < np.inf

    for light_pos, light_col in zip(light_positions, light_colors):
        light_pos = np.asarray(light_pos, dtype=float)
        light_col = np.asarray(light_col, dtype=float)

        for y in range(h):
            for x in range(w):
                if not mask[y, x]:
                    continue
                # Diffuse lighting
                L = light_pos - gb.position[y, x]
                dist = np.linalg.norm(L)
                L /= dist
                NdotL = max(np.dot(gb.normal[y, x], L), 0)
                attenuation = 1.0 / (1.0 + 0.05 * dist * dist)
                output[y, x] += gb.albedo[y, x] * light_col * NdotL * attenuation

    # Add ambient
    output[mask] += gb.albedo[mask] * 0.08
    return np.clip(output, 0, None)  # Allow HDR values > 1


# ---------------------------------------------------------------------------
# 3. Shadow Mapping
# ---------------------------------------------------------------------------


def generate_shadow_map(gb: GBuffer,
                        light_pos: np.ndarray,
                        map_size: int = 128) -> np.ndarray:
    """Generate a depth map from the light's perspective.

    Shadow mapping works in two passes:
      Pass 1: Render the scene from the light's POV, store depth
      Pass 2: For each camera pixel, project into light space and compare
              its depth with the shadow map -- if farther, it's in shadow

    Why a separate depth buffer?  The light "sees" only the closest
    surfaces.  Anything behind those surfaces is in shadow.
    """
    shadow_map = np.full((map_size, map_size), np.inf, dtype=float)

    # Simple orthographic projection from light
    mask = gb.depth < np.inf
    positions = gb.position[mask]

    if len(positions) == 0:
        return shadow_map

    # Project world positions into light space (simplified)
    light_dir = np.array([0, -1, -0.5])
    light_dir /= np.linalg.norm(light_dir)

    right = np.cross(light_dir, np.array([0, 0, 1]))
    right /= np.linalg.norm(right)
    up = np.cross(right, light_dir)

    for pos in positions:
        rel = pos - light_pos
        lx = np.dot(rel, right)
        ly = np.dot(rel, up)
        lz = np.dot(rel, light_dir)

        # Map to shadow map coordinates
        sx = int((lx / 10 + 0.5) * map_size)
        sy = int((ly / 10 + 0.5) * map_size)

        if 0 <= sx < map_size and 0 <= sy < map_size:
            shadow_map[sy, sx] = min(shadow_map[sy, sx], lz)

    return shadow_map


def apply_shadow(gb: GBuffer, lit_image: np.ndarray,
                 light_pos: np.ndarray,
                 shadow_map: np.ndarray,
                 map_size: int = 128,
                 bias: float = 0.05) -> np.ndarray:
    """Apply shadow mapping to the lit image.

    The bias parameter prevents "shadow acne" -- self-shadowing artifacts
    caused by floating-point precision.  Too little bias = acne,
    too much bias = shadows detach from objects ("Peter Panning").
    """
    h, w = gb.depth.shape
    result = lit_image.copy()

    light_dir = np.array([0, -1, -0.5])
    light_dir /= np.linalg.norm(light_dir)
    right = np.cross(light_dir, np.array([0, 0, 1]))
    right /= np.linalg.norm(right)
    up = np.cross(right, light_dir)

    for y in range(h):
        for x in range(w):
            if gb.depth[y, x] >= np.inf:
                continue
            pos = gb.position[y, x]
            rel = pos - light_pos
            lx = np.dot(rel, right)
            ly = np.dot(rel, up)
            lz = np.dot(rel, light_dir)

            sx = int((lx / 10 + 0.5) * map_size)
            sy = int((ly / 10 + 0.5) * map_size)

            if 0 <= sx < map_size and 0 <= sy < map_size:
                if lz > shadow_map[sy, sx] + bias:
                    result[y, x] *= 0.3  # In shadow: reduce to ambient

    return result


# ---------------------------------------------------------------------------
# 4. Screen-Space Ambient Occlusion (SSAO)
# ---------------------------------------------------------------------------


def compute_ssao(gb: GBuffer, num_samples: int = 16,
                 radius: float = 0.5) -> np.ndarray:
    """Compute SSAO -- approximate ambient occlusion from the depth buffer.

    SSAO works by sampling random points around each pixel's 3D position.
    If many samples are occluded (their depth is closer to the camera
    than the depth buffer says), the pixel is in a "crevice" and should
    be darker.

    Why "screen-space"?  We only use information already in the G-buffer
    (depth + normals).  No scene geometry is needed.  This makes it O(1)
    in scene complexity -- it costs the same whether the scene has 100
    or 10 million triangles.
    """
    h, w = gb.depth.shape
    ao = np.ones((h, w), dtype=float)

    np.random.seed(42)
    # Pre-generate sample kernel (hemisphere oriented along +Z)
    kernel = np.random.randn(num_samples, 3)
    kernel[:, 2] = np.abs(kernel[:, 2])  # Hemisphere
    for i in range(num_samples):
        kernel[i] = kernel[i] / np.linalg.norm(kernel[i])
        # Scale samples closer to center (more samples near surface)
        scale = (i + 1) / num_samples
        kernel[i] *= radius * (0.1 + 0.9 * scale * scale)

    step = 2  # Process every other pixel for speed
    for y in range(0, h, step):
        for x in range(0, w, step):
            if gb.depth[y, x] >= np.inf:
                continue

            normal = gb.normal[y, x]
            pos = gb.position[y, x]

            # Build TBN matrix to orient samples along surface normal
            tangent = np.array([1, 0, 0])
            if abs(np.dot(tangent, normal)) > 0.9:
                tangent = np.array([0, 1, 0])
            tangent = tangent - np.dot(tangent, normal) * normal
            tangent /= np.linalg.norm(tangent)
            bitangent = np.cross(normal, tangent)
            tbn = np.column_stack([tangent, bitangent, normal])

            occlusion = 0
            for s in range(num_samples):
                sample_pos = pos + tbn @ kernel[s]
                # Project sample to screen space (simplified)
                dy = int((sample_pos[1] - pos[1]) * 30)
                dx = int((sample_pos[0] - pos[0]) * 30)
                sy, sx = y + dy, x + dx

                if 0 <= sy < h and 0 <= sx < w:
                    if gb.depth[sy, sx] < gb.depth[y, x] - 0.02:
                        occlusion += 1

            ao_val = 1.0 - occlusion / num_samples
            # Fill the step x step block
            for fy in range(step):
                for fx in range(step):
                    if y + fy < h and x + fx < w:
                        ao[y + fy, x + fx] = ao_val

    return ao


# ---------------------------------------------------------------------------
# 5. HDR Tone Mapping
# ---------------------------------------------------------------------------


def tone_map_reinhard(hdr: np.ndarray) -> np.ndarray:
    """Reinhard tone mapping: maps HDR [0, inf) to LDR [0, 1].

    The formula x / (1 + x) is a simple sigmoid that compresses bright
    values while preserving dark detail.  It's the "hello world" of
    tone mapping operators.
    """
    return hdr / (1.0 + hdr)


def tone_map_aces(hdr: np.ndarray) -> np.ndarray:
    """ACES filmic tone mapping -- the industry standard.

    Used in Unreal Engine, Unity, and most AAA games.  It produces
    a more pleasing S-curve with richer contrast than Reinhard:
    darker shadows, brighter highlights, and a slight warm shift.
    """
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((hdr * (a * hdr + b)) / (hdr * (c * hdr + d) + e), 0, 1)


# ---------------------------------------------------------------------------
# 6. Bloom
# ---------------------------------------------------------------------------


def gaussian_blur(image: np.ndarray, sigma: float = 3.0,
                  kernel_size: int = 15) -> np.ndarray:
    """Apply Gaussian blur (used for bloom spread)."""
    k = kernel_size // 2
    ax = np.arange(-k, k + 1)
    kernel_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()

    # Separable blur: horizontal then vertical
    result = image.copy()
    h, w = image.shape[:2]

    # Horizontal pass
    padded = np.pad(result, ((0, 0), (k, k), (0, 0)), mode='edge')
    for i, weight in enumerate(kernel_1d):
        result += weight * padded[:, i:i + w, :]
    result /= 2  # Normalize

    # Vertical pass
    padded = np.pad(result, ((k, k), (0, 0), (0, 0)), mode='edge')
    temp = np.zeros_like(result)
    for i, weight in enumerate(kernel_1d):
        temp += weight * padded[i:i + h, :, :]
    return temp


def apply_bloom(hdr_image: np.ndarray,
                threshold: float = 1.0,
                intensity: float = 0.4) -> np.ndarray:
    """Extract bright pixels and add blurred glow.

    Bloom simulates the scattering of light in camera lenses and the
    human eye.  Bright light sources "bleed" into surrounding pixels.
    The effect is simple: threshold -> blur -> add back.
    """
    # Extract pixels brighter than threshold
    luminance = 0.2126 * hdr_image[:, :, 0] + 0.7152 * hdr_image[:, :, 1] + 0.0722 * hdr_image[:, :, 2]
    bright_mask = (luminance > threshold)[:, :, np.newaxis]
    bright = hdr_image * bright_mask

    # Blur the bright pixels
    blurred = gaussian_blur(bright, sigma=4.0, kernel_size=21)

    # Add bloom back to the image
    return hdr_image + blurred * intensity


# ---------------------------------------------------------------------------
# 7. Demonstrations
# ---------------------------------------------------------------------------


def demo_deferred_rendering():
    """Show G-buffer channels and deferred lighting."""
    print("  Generating synthetic G-buffer (this may take a moment)...")
    gb = generate_synthetic_gbuffer(200, 200)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Deferred Rendering: G-Buffer Channels",
                 fontsize=14, fontweight='bold')

    # G-buffer channels
    axes[0, 0].imshow(np.flipud(gb.albedo), interpolation='nearest')
    axes[0, 0].set_title("Albedo (base color)")
    axes[0, 0].axis('off')

    normal_vis = np.flipud(gb.normal * 0.5 + 0.5)
    axes[0, 1].imshow(normal_vis, interpolation='nearest')
    axes[0, 1].set_title("World Normals (remapped)")
    axes[0, 1].axis('off')

    depth_vis = np.flipud(gb.depth.copy())
    depth_vis[depth_vis == np.inf] = np.nan
    axes[0, 2].imshow(depth_vis, cmap='viridis', interpolation='nearest')
    axes[0, 2].set_title("Depth")
    axes[0, 2].axis('off')

    # Deferred lighting with multiple lights
    print("  Computing deferred lighting...")
    lights_pos = [[3, 5, 2], [-3, 4, 0], [0, 3, 5]]
    lights_col = [[1.0, 0.9, 0.8], [0.3, 0.5, 1.0], [0.8, 1.0, 0.6]]
    lit = deferred_lighting(gb, lights_pos, lights_col)

    axes[1, 0].imshow(np.flipud(np.clip(lit, 0, 1)), interpolation='nearest')
    axes[1, 0].set_title("3-Light Deferred (LDR clamp)")
    axes[1, 0].axis('off')

    # Single light for comparison
    lit_single = deferred_lighting(gb, [lights_pos[0]], [lights_col[0]])
    axes[1, 1].imshow(np.flipud(np.clip(lit_single, 0, 1)), interpolation='nearest')
    axes[1, 1].set_title("1-Light Deferred")
    axes[1, 1].axis('off')

    # SSAO
    print("  Computing SSAO...")
    ao = compute_ssao(gb, num_samples=12, radius=0.4)
    axes[1, 2].imshow(np.flipud(ao), cmap='gray', vmin=0, vmax=1,
                       interpolation='nearest')
    axes[1, 2].set_title("SSAO (ambient occlusion)")
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_15_deferred.png", dpi=100)
    plt.show()


def demo_tone_mapping_bloom():
    """Compare tone mapping operators and bloom effect."""
    print("  Generating HDR scene...")
    gb = generate_synthetic_gbuffer(200, 200)
    lights_pos = [[2, 6, 3], [-2, 5, 1]]
    lights_col = [[2.0, 1.8, 1.5], [0.5, 0.8, 2.0]]  # HDR light intensities
    hdr = deferred_lighting(gb, lights_pos, lights_col)
    hdr_flipped = np.flipud(hdr)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle("HDR Tone Mapping and Bloom", fontsize=14, fontweight='bold')

    # Raw HDR (clamped for display)
    axes[0, 0].imshow(np.clip(hdr_flipped, 0, 1), interpolation='nearest')
    axes[0, 0].set_title("HDR Clamped (overexposed)")
    axes[0, 0].axis('off')

    # Reinhard
    reinhard = tone_map_reinhard(hdr_flipped)
    axes[0, 1].imshow(reinhard, interpolation='nearest')
    axes[0, 1].set_title("Reinhard Tone Map")
    axes[0, 1].axis('off')

    # ACES
    aces = tone_map_aces(hdr_flipped)
    axes[1, 0].imshow(aces, interpolation='nearest')
    axes[1, 0].set_title("ACES Filmic Tone Map")
    axes[1, 0].axis('off')

    # ACES + Bloom
    bloomed = apply_bloom(hdr_flipped, threshold=0.8, intensity=0.5)
    aces_bloom = tone_map_aces(bloomed)
    axes[1, 1].imshow(aces_bloom, interpolation='nearest')
    axes[1, 1].set_title("ACES + Bloom")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_15_tonemapping.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Real-Time Rendering Techniques")
    print("=" * 60)

    print("\n[1/2] Deferred rendering + G-buffer + SSAO...")
    demo_deferred_rendering()

    print("\n[2/2] HDR tone mapping + bloom...")
    demo_tone_mapping_bloom()

    print("\nDone!")


if __name__ == "__main__":
    main()
