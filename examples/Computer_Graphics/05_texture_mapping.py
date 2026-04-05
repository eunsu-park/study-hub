"""
Texture Mapping
================

Demonstrates the core concepts of texture mapping in computer graphics:

1. Procedural texture generation (checkerboard, Perlin-like noise)
2. UV coordinate calculation for a sphere (spherical mapping)
3. Nearest-neighbor vs bilinear sampling comparison
4. Mipmap generation (successive half-resolution levels)
5. Normal mapping effect

Textures are what make 3D objects look realistic -- without them,
every surface would be a flat color.  The mapping from 3D surface
points to 2D texture coordinates (UVs) is the fundamental bridge
between geometry and appearance.

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Procedural Texture Generation
# ---------------------------------------------------------------------------


def generate_checkerboard(size: int = 256, tiles: int = 8) -> np.ndarray:
    """Generate a checkerboard texture.

    Why checkerboard?  It's the classic texture for debugging UV mapping
    because any distortion, stretching, or seam is immediately visible.
    It also clearly shows the difference between sampling modes.

    Returns shape (size, size, 3) with values in [0, 1].
    """
    # Why integer division for pattern?  Each pixel's tile index is
    # determined by which tile "cell" it falls into.  XOR creates the
    # alternating pattern (same parity = white, different = black).
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)

    tile_size = size // tiles
    pattern = ((X // tile_size) + (Y // tile_size)) % 2

    texture = np.zeros((size, size, 3))
    texture[pattern == 0] = [0.9, 0.9, 0.9]  # Light squares
    texture[pattern == 1] = [0.2, 0.2, 0.8]  # Dark blue squares

    return texture


def generate_noise_texture(size: int = 256, scale: float = 20.0,
                            octaves: int = 4) -> np.ndarray:
    """Generate a value-noise texture with multiple octaves (fBm-like).

    Why not Perlin noise exactly?  Perlin noise uses gradient interpolation,
    which is complex to implement from scratch.  Value noise (random values
    at grid points, interpolated) produces similar visual results and is
    much simpler.  We add multiple octaves (like fractal Brownian motion)
    for natural-looking detail at multiple scales.

    Returns shape (size, size, 3) with values in [0, 1].
    """
    result = np.zeros((size, size))

    for octave in range(octaves):
        # Each octave doubles the frequency and halves the amplitude
        # Why this pattern?  It mimics natural textures (wood, marble, clouds)
        # which have structure at multiple scales.
        freq = scale * (2 ** octave)
        amp = 0.5 ** octave

        # Generate random grid values
        grid_w = int(freq) + 2
        grid_h = int(freq) + 2
        grid = np.random.RandomState(42 + octave).rand(grid_h, grid_w)

        # Map pixel coordinates to grid coordinates
        x = np.linspace(0, grid_w - 1.001, size)
        y = np.linspace(0, grid_h - 1.001, size)
        X, Y = np.meshgrid(x, y)

        # Bilinear interpolation of grid values
        # Why bilinear?  Nearest-neighbor would create blocky artifacts.
        # Bilinear produces smooth transitions between grid cells.
        x0 = X.astype(int)
        y0 = Y.astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        # Fractional parts for interpolation weights
        fx = X - x0
        fy = Y - y0

        # Smoothstep for smoother transitions (avoids visible grid lines)
        fx = fx * fx * (3 - 2 * fx)
        fy = fy * fy * (3 - 2 * fy)

        # Clamp indices
        x0 = np.clip(x0, 0, grid_w - 1)
        x1 = np.clip(x1, 0, grid_w - 1)
        y0 = np.clip(y0, 0, grid_h - 1)
        y1 = np.clip(y1, 0, grid_h - 1)

        # Bilinear interpolation
        v00 = grid[y0, x0]
        v10 = grid[y0, x1]
        v01 = grid[y1, x0]
        v11 = grid[y1, x1]

        value = (v00 * (1-fx) * (1-fy) + v10 * fx * (1-fy) +
                 v01 * (1-fx) * fy + v11 * fx * fy)

        result += amp * value

    # Normalize to [0, 1]
    result = (result - result.min()) / (result.max() - result.min() + 1e-10)

    # Convert to RGB with a warm color palette (earth tones)
    texture = np.zeros((size, size, 3))
    texture[:, :, 0] = 0.3 + 0.5 * result  # Red channel
    texture[:, :, 1] = 0.2 + 0.3 * result  # Green channel
    texture[:, :, 2] = 0.1 + 0.1 * result  # Blue channel

    return np.clip(texture, 0, 1)


def demo_procedural_textures():
    """Display generated procedural textures."""
    checker = generate_checkerboard(256, 8)
    noise = generate_noise_texture(256, 6.0, 4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Procedural Textures", fontsize=14, fontweight='bold')

    ax1.imshow(checker)
    ax1.set_title("Checkerboard (8x8)")
    ax1.axis('off')

    ax2.imshow(noise)
    ax2.set_title("Value Noise (4 octaves)")
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_05_procedural_textures.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 2. UV Coordinates for a Sphere (Spherical Mapping)
# ---------------------------------------------------------------------------

def spherical_uv(normals: np.ndarray) -> tuple:
    """Compute UV coordinates from sphere surface normals.

    Maps 3D sphere surface points to 2D texture coordinates using
    spherical (equirectangular) mapping:
      u = 0.5 + atan2(z, x) / (2*pi)
      v = 0.5 - asin(y) / pi

    Why spherical mapping?  It's the most intuitive mapping for spheres
    and corresponds to latitude/longitude.  The inevitable singularity
    at the poles (where u wraps around) is a fundamental limitation of
    mapping a sphere to a rectangle -- this is why real-world maps of
    Earth also distort at the poles.

    Parameters
    ----------
    normals : (H, W, 3) array of unit normals (= positions on unit sphere)

    Returns
    -------
    u, v : (H, W) arrays of texture coordinates in [0, 1]
    """
    x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]

    # atan2(z, x) gives the azimuthal angle [-pi, pi]
    # Adding 0.5 shifts from [-0.5, 0.5] to [0, 1]
    u = 0.5 + np.arctan2(z, x) / (2 * np.pi)

    # asin(y) gives the polar angle [-pi/2, pi/2]
    v = 0.5 - np.arcsin(np.clip(y, -1, 1)) / np.pi

    return u, v


def generate_sphere_data(size: int = 256) -> tuple:
    """Generate sphere normals and mask (hemisphere visible from front).

    Returns (normals, mask) where normals is (H, W, 3).
    """
    y_coords = np.linspace(1, -1, size)
    x_coords = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x_coords, y_coords)

    R_sq = X**2 + Y**2
    mask = R_sq <= 1.0

    Z = np.zeros_like(X)
    Z[mask] = np.sqrt(1.0 - R_sq[mask])

    normals = np.stack([X, Y, Z], axis=-1)
    return normals, mask


def demo_uv_mapping():
    """Visualize UV coordinates on a sphere and the mapped texture."""
    normals, mask = generate_sphere_data(300)
    u, v = spherical_uv(normals)

    # Apply checkerboard texture
    texture = generate_checkerboard(512, 8)
    tex_h, tex_w = texture.shape[:2]

    # Sample texture using UV coordinates
    # Map UV [0,1] to pixel indices [0, tex_size-1]
    tx = (u * (tex_w - 1)).astype(int)
    ty = (v * (tex_h - 1)).astype(int)
    tx = np.clip(tx, 0, tex_w - 1)
    ty = np.clip(ty, 0, tex_h - 1)

    image = np.zeros((300, 300, 3))
    image[mask] = texture[ty[mask], tx[mask]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Spherical UV Mapping", fontsize=14, fontweight='bold')

    # UV visualization (u as red, v as green)
    uv_vis = np.zeros((300, 300, 3))
    uv_vis[mask, 0] = u[mask]
    uv_vis[mask, 1] = v[mask]
    axes[0].imshow(uv_vis)
    axes[0].set_title("UV Coordinates (R=u, G=v)")
    axes[0].axis('off')

    # The texture itself
    axes[1].imshow(texture)
    axes[1].set_title("Texture (Checkerboard)")
    axes[1].axis('off')

    # Textured sphere
    axes[2].imshow(image)
    axes[2].set_title("Textured Sphere")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_05_uv_mapping.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 3. Texture Sampling: Nearest vs Bilinear
# ---------------------------------------------------------------------------

def sample_nearest(texture: np.ndarray, u: np.ndarray,
                   v: np.ndarray) -> np.ndarray:
    """Sample texture using nearest-neighbor filtering.

    Simply rounds UV to the nearest texel.

    Why use this ever?  It's fast, preserves sharp pixel edges (important
    for pixel art), and doesn't blur the texture.  But it creates
    aliasing artifacts (shimmering, blocky edges) when the texture is
    minified or magnified.
    """
    h, w = texture.shape[:2]
    tx = np.clip(np.round(u * (w - 1)).astype(int), 0, w - 1)
    ty = np.clip(np.round(v * (h - 1)).astype(int), 0, h - 1)
    return texture[ty, tx]


def sample_bilinear(texture: np.ndarray, u: np.ndarray,
                    v: np.ndarray) -> np.ndarray:
    """Sample texture using bilinear filtering.

    Interpolates between the 4 nearest texels, weighted by proximity.

    Why bilinear?  It produces smooth results when the texture is
    stretched (magnified).  The interpolation prevents the blocky
    staircase artifacts of nearest-neighbor.

    However, bilinear still aliases when minifying (showing the texture
    at a smaller size than its resolution).  That's where mipmaps help.
    """
    h, w = texture.shape[:2]

    # Continuous texel coordinates
    x = u * (w - 1)
    y = v * (h - 1)

    # Four surrounding texels
    x0 = np.clip(np.floor(x).astype(int), 0, w - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.clip(np.floor(y).astype(int), 0, h - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    # Fractional offsets
    fx = (x - x0)[..., np.newaxis]
    fy = (y - y0)[..., np.newaxis]

    # Bilinear interpolation: weighted average of 4 texels
    result = (texture[y0, x0] * (1-fx) * (1-fy) +
              texture[y1, x0] * (1-fx) * fy +
              texture[y0, x1] * fx * (1-fy) +
              texture[y1, x1] * fx * fy)

    return result


def demo_sampling_comparison():
    """Compare nearest-neighbor and bilinear sampling on a zoomed texture."""
    # Small texture (8x8) to make the difference dramatic
    texture = generate_checkerboard(8, 2)

    normals, mask = generate_sphere_data(300)
    u, v = spherical_uv(normals)

    nearest_img = np.zeros((300, 300, 3))
    bilinear_img = np.zeros((300, 300, 3))

    nearest_result = sample_nearest(texture, u, v)
    bilinear_result = sample_bilinear(texture, u, v)

    nearest_img[mask] = nearest_result[mask]
    bilinear_img[mask] = bilinear_result[mask]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Texture Sampling: Nearest vs Bilinear (8x8 texture on sphere)",
                 fontsize=13, fontweight='bold')

    # Show the tiny source texture magnified
    axes[0].imshow(texture, interpolation='nearest')
    axes[0].set_title("Source Texture (8x8)")
    axes[0].axis('off')

    axes[1].imshow(nearest_img)
    axes[1].set_title("Nearest-Neighbor (blocky)")
    axes[1].axis('off')

    axes[2].imshow(bilinear_img)
    axes[2].set_title("Bilinear (smooth)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_05_sampling_comparison.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 4. Mipmap Generation
# ---------------------------------------------------------------------------

def generate_mipmaps(texture: np.ndarray, levels: int = 0) -> list:
    """Generate a mipmap chain by successive half-resolution averaging.

    Mipmaps (Williams, 1983) are pre-filtered versions of a texture at
    progressively lower resolutions: level 0 is the original, level 1
    is half-size, level 2 is quarter-size, etc.

    Why mipmaps?
    1. **Anti-aliasing**: When a texture is viewed from far away (minified),
       many texels map to one pixel.  Without mipmaps, sampling one texel
       misses the others, causing flickering/shimmering.  Mipmaps pre-
       average, so each sample represents the correct area.
    2. **Performance**: Smaller mip levels are faster to sample (better
       cache locality, fewer memory fetches).

    Memory cost is only 1/3 extra (sum of 1/4 + 1/16 + 1/64 + ... = 1/3).

    Parameters
    ----------
    texture : Original texture, shape (H, W, 3)
    levels  : Number of mip levels (0 = auto, generate until 1x1)
    """
    h, w = texture.shape[:2]
    if levels == 0:
        levels = int(np.log2(min(h, w)))

    mipmaps = [texture.copy()]

    current = texture.copy()
    for level in range(1, levels + 1):
        new_h = max(1, current.shape[0] // 2)
        new_w = max(1, current.shape[1] // 2)

        # Box filter: average 2x2 blocks
        # Why box filter instead of something fancier?  It's simple, fast,
        # and produces acceptable quality.  Higher-quality filters (Lanczos,
        # Gaussian) are better but the visual difference at mip boundaries
        # is minimal when trilinear filtering blends between levels.
        downsampled = np.zeros((new_h, new_w, 3))
        for dy in range(2):
            for dx in range(2):
                downsampled += current[dy:dy+new_h*2:2, dx:dx+new_w*2:2, :3]
        downsampled /= 4.0

        mipmaps.append(downsampled)
        current = downsampled

    return mipmaps


def demo_mipmaps():
    """Visualize a mipmap chain and show the memory layout."""
    texture = generate_checkerboard(256, 16)
    mipmaps = generate_mipmaps(texture, levels=8)

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    fig.suptitle("Mipmap Chain (each level = half resolution of previous)",
                 fontsize=14, fontweight='bold')

    # First row: individual levels
    for i in range(min(5, len(mipmaps))):
        ax = axes[0, i]
        ax.imshow(mipmaps[i], interpolation='nearest')
        h, w = mipmaps[i].shape[:2]
        ax.set_title(f"Level {i} ({w}x{h})", fontsize=9)
        ax.axis('off')

    # Second row: continued + info
    for i in range(5, min(10, len(mipmaps))):
        ax = axes[1, i - 5]
        ax.imshow(mipmaps[i], interpolation='nearest')
        h, w = mipmaps[i].shape[:2]
        ax.set_title(f"Level {i} ({w}x{h})", fontsize=9)
        ax.axis('off')

    # Fill remaining axes
    for i in range(len(mipmaps) - 5, 5):
        if i >= 0:
            axes[1, i].axis('off')

    # Memory usage info
    total_pixels = sum(m.shape[0] * m.shape[1] for m in mipmaps)
    base_pixels = mipmaps[0].shape[0] * mipmaps[0].shape[1]
    axes[1, 4].text(0.1, 0.5,
                    f"Mipmap levels: {len(mipmaps)}\n"
                    f"Base: {base_pixels:,} texels\n"
                    f"Total: {total_pixels:,} texels\n"
                    f"Overhead: {(total_pixels/base_pixels - 1)*100:.1f}%",
                    transform=axes[1, 4].transAxes, fontsize=10,
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axes[1, 4].axis('off')

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_05_mipmaps.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 5. Normal Mapping
# ---------------------------------------------------------------------------

def generate_normal_map(size: int = 256) -> np.ndarray:
    """Generate a procedural normal map (brick-like pattern).

    Normal maps store perturbed surface normals in tangent space as RGB:
      R = x component [-1, 1] mapped to [0, 1]
      G = y component [-1, 1] mapped to [0, 1]
      B = z component [0, 1]  (always positive -- pointing outward)

    Why normal maps?  They add surface detail (bumps, grooves, brick
    patterns) without adding any geometry.  A flat polygon with a normal
    map can look like a rough brick wall because the shading uses the
    perturbed normals instead of the geometric normal.

    This is one of the most impactful optimizations in real-time graphics:
    millions of polygons of detail replaced by a single texture lookup.
    """
    normal_map = np.zeros((size, size, 3))
    normal_map[:, :, 2] = 1.0  # Default: flat surface pointing up

    # Create a brick-like height pattern
    height = np.zeros((size, size))

    brick_h, brick_w = 32, 64
    mortar = 3  # Width of mortar lines

    for row in range(size // brick_h):
        y_start = row * brick_h
        y_end = min(y_start + brick_h, size)

        # Offset alternating rows (running bond pattern)
        offset = (brick_w // 2) * (row % 2)

        # Horizontal mortar line
        if y_start + mortar < size:
            height[y_start:y_start + mortar, :] = -0.5

        for col in range(-1, size // brick_w + 1):
            x_start = col * brick_w + offset
            if 0 <= x_start < size:
                x_end = min(x_start + mortar, size)
                height[y_start:y_end, max(0, x_start):x_end] = -0.5

    # Convert height map to normal map using finite differences
    # Why finite differences?  The gradient of the height map gives the
    # surface slope, which defines the tangent-space normal perturbation.
    dx = np.zeros_like(height)
    dy = np.zeros_like(height)
    dx[:, 1:-1] = (height[:, 2:] - height[:, :-2]) / 2.0
    dy[1:-1, :] = (height[2:, :] - height[:-2, :]) / 2.0

    strength = 3.0  # Normal map strength (higher = more pronounced bumps)
    normal_map[:, :, 0] = -dx * strength
    normal_map[:, :, 1] = -dy * strength
    normal_map[:, :, 2] = 1.0

    # Normalize each normal vector
    norms = np.linalg.norm(normal_map, axis=-1, keepdims=True)
    normal_map = normal_map / (norms + 1e-10)

    return normal_map, height


def demo_normal_mapping():
    """Demonstrate the visual effect of normal mapping on a flat surface.

    Compares shading with geometric normals (flat) vs. perturbed normals
    (bumpy) on the same geometry.
    """
    size = 300
    normal_map, height_map = generate_normal_map(size)

    # Light direction
    light_dir = np.array([0.5, 0.5, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)

    # Flat geometric normal (pointing straight up/out of screen)
    flat_normal = np.zeros((size, size, 3))
    flat_normal[:, :, 2] = 1.0

    # Compute diffuse shading for both
    # N.L for each pixel
    flat_ndotl = np.sum(flat_normal * light_dir, axis=-1)
    flat_ndotl = np.maximum(flat_ndotl, 0)

    bumpy_ndotl = np.sum(normal_map * light_dir, axis=-1)
    bumpy_ndotl = np.maximum(bumpy_ndotl, 0)

    # Apply to a base color (brick red)
    base_color = np.array([0.7, 0.3, 0.2])
    ambient = 0.15

    flat_image = ambient + (1 - ambient) * flat_ndotl[..., np.newaxis] * base_color
    bumpy_image = ambient + (1 - ambient) * bumpy_ndotl[..., np.newaxis] * base_color

    # Normal map visualization (remap from [-1,1] to [0,1] for display)
    normal_vis = (normal_map + 1) * 0.5

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Normal Mapping: Surface Detail Without Extra Geometry",
                 fontsize=14, fontweight='bold')

    axes[0, 0].imshow(height_map, cmap='gray')
    axes[0, 0].set_title("Height Map (source)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(normal_vis)
    axes[0, 1].set_title("Normal Map (RGB = XYZ)")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(np.clip(flat_image, 0, 1))
    axes[1, 0].set_title("Flat Shading (no normal map)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(np.clip(bumpy_image, 0, 1))
    axes[1, 1].set_title("Normal-Mapped Shading (same geometry!)")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_05_normal_mapping.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 6. Full textured sphere with noise texture
# ---------------------------------------------------------------------------

def demo_textured_sphere():
    """Render a sphere with a noise texture and basic Lambertian shading."""
    size = 350
    normals, mask = generate_sphere_data(size)
    u, v = spherical_uv(normals)

    texture = generate_noise_texture(512, 8.0, 5)

    # Sample texture with bilinear filtering
    sampled = sample_bilinear(texture, u, v)

    # Apply Lambertian shading
    light_dir = np.array([1.0, 1.0, 1.5])
    light_dir = light_dir / np.linalg.norm(light_dir)

    ndotl = np.sum(normals * light_dir, axis=-1, keepdims=True)
    ndotl = np.maximum(ndotl, 0)

    ambient = 0.1
    shaded = sampled * (ambient + (1 - ambient) * ndotl)

    image = np.zeros((size, size, 3))
    image[mask] = np.clip(shaded[mask], 0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Textured Sphere with Noise Texture", fontsize=14,
                 fontweight='bold')

    ax1.imshow(texture)
    ax1.set_title("Noise Texture")
    ax1.axis('off')

    ax2.imshow(image)
    ax2.set_title("Sphere with Texture + Lambert Shading")
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_05_textured_sphere.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Texture Mapping")
    print("=" * 60)

    print("\n[1/5] Procedural textures...")
    demo_procedural_textures()

    print("\n[2/5] UV mapping on sphere...")
    demo_uv_mapping()

    print("\n[3/5] Nearest vs bilinear sampling...")
    demo_sampling_comparison()

    print("\n[4/5] Mipmap generation...")
    demo_mipmaps()

    print("\n[5/5] Normal mapping...")
    demo_normal_mapping()

    print("\n[Bonus] Textured sphere with noise texture...")
    demo_textured_sphere()

    print("\nDone!")


if __name__ == "__main__":
    main()
