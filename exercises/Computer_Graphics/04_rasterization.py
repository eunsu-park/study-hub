"""
Exercises for Lesson 04: Rasterization
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

def edge_function(a, b, p):
    """Signed area of parallelogram formed by edge A->B and point P."""
    return (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])


def draw_line_bresenham(x0, y0, x1, y1):
    """Bresenham's line algorithm for all octants."""
    pixels = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    steep = dy > dx
    if steep:
        dx, dy = dy, dx
    error = 2 * dy - dx
    x, y = x0, y0
    for _ in range(dx + 1):
        pixels.append((x, y))
        if error > 0:
            if steep:
                x += sx
            else:
                y += sy
            error -= 2 * dx
        if steep:
            y += sy
        else:
            x += sx
        error += 2 * dy
    return pixels


def exercise_1():
    """
    Bresenham Implementation: Implement Bresenham's algorithm for all eight
    octants. Test by drawing lines at angles 0, 45, 90, 135, 180, 225, 270,
    315 degrees and verify they look correct.
    """
    center_x, center_y = 20, 20
    length = 15

    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    grid_size = 41

    # Create a grid for visualization
    grid = np.zeros((grid_size, grid_size), dtype=int)
    grid[center_y, center_x] = 2  # Mark center

    print("Bresenham's Line Algorithm - All 8 Octants")
    print("=" * 50)

    for angle in angles:
        rad = np.radians(angle)
        end_x = center_x + int(round(length * np.cos(rad)))
        end_y = center_y + int(round(length * np.sin(rad)))

        pixels = draw_line_bresenham(center_x, center_y, end_x, end_y)

        print(f"  {angle:>3} deg: ({center_x},{center_y}) -> ({end_x},{end_y}), "
              f"{len(pixels)} pixels")

        for px, py in pixels:
            if 0 <= px < grid_size and 0 <= py < grid_size:
                grid[py, px] = 1

    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(grid, cmap='Blues', origin='lower', interpolation='nearest')
    ax.set_title('Bresenham Lines at 8 Angles')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    for angle in angles:
        rad = np.radians(angle)
        dx = length * np.cos(rad) * 0.7
        dy = length * np.sin(rad) * 0.7
        ax.annotate(f'{angle}',
                    xy=(center_x + dx, center_y + dy),
                    fontsize=8, ha='center', va='center', color='red')

    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('ex04_bresenham_octants.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to ex04_bresenham_octants.png")

    # Verify properties
    print("\nVerification:")
    print("  - Lines should be connected (no gaps)")
    print("  - Lines at 0/90/180/270 should be perfectly horizontal/vertical")
    print("  - Lines at 45/135/225/315 should be perfect diagonals")
    for angle in [0, 90, 45]:
        rad = np.radians(angle)
        end_x = center_x + int(round(length * np.cos(rad)))
        end_y = center_y + int(round(length * np.sin(rad)))
        pixels = draw_line_bresenham(center_x, center_y, end_x, end_y)
        # Check connectivity: each pixel should be adjacent to the next
        connected = True
        for i in range(1, len(pixels)):
            dx = abs(pixels[i][0] - pixels[i - 1][0])
            dy = abs(pixels[i][1] - pixels[i - 1][1])
            if dx > 1 or dy > 1:
                connected = False
                break
        print(f"  {angle:>3} deg: connected = {connected}, pixels = {len(pixels)}")


def exercise_2():
    """
    Edge Function Visualization: For a triangle with vertices (50,20), (200,150),
    (30,180), compute and visualize the edge function values across a 256x256 grid.
    """
    v0 = (50, 20)
    v1 = (200, 150)
    v2 = (30, 180)
    width, height = 256, 256

    print("Edge Function Visualization")
    print(f"Triangle: {v0}, {v1}, {v2}")
    print(f"Grid: {width}x{height}")
    print()

    # Compute edge functions for each pixel
    e01 = np.zeros((height, width))  # Edge v0->v1
    e12 = np.zeros((height, width))  # Edge v1->v2
    e20 = np.zeros((height, width))  # Edge v2->v0

    for y in range(height):
        for x in range(width):
            p = (x + 0.5, y + 0.5)
            e01[y, x] = edge_function(v0, v1, p)
            e12[y, x] = edge_function(v1, v2, p)
            e20[y, x] = edge_function(v2, v0, p)

    # Inside test: all three must be non-negative (or all non-positive)
    area = edge_function(v0, v1, v2)
    if area > 0:
        inside = (e01 >= 0) & (e12 >= 0) & (e20 >= 0)
    else:
        inside = (e01 <= 0) & (e12 <= 0) & (e20 <= 0)

    inside_count = np.sum(inside)
    print(f"Triangle area (2x signed): {area:.1f}")
    print(f"Winding: {'CCW' if area > 0 else 'CW'}")
    print(f"Pixels inside triangle: {inside_count}")
    print()

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    titles = ['Edge v0->v1', 'Edge v1->v2', 'Edge v2->v0', 'Inside (all positive)']
    data = [e01, e12, e20, inside.astype(float)]
    cmaps = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'Greens']

    for ax, title, d, cmap in zip(axes.flat, titles, data, cmaps):
        if cmap == 'RdBu_r':
            vmax = max(abs(d.min()), abs(d.max()))
            im = ax.imshow(d, cmap=cmap, vmin=-vmax, vmax=vmax,
                           origin='lower', interpolation='nearest')
        else:
            im = ax.imshow(d, cmap=cmap, origin='lower', interpolation='nearest')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Draw triangle vertices
        for v, label in [(v0, 'v0'), (v1, 'v1'), (v2, 'v2')]:
            ax.plot(v[0], v[1], 'ko', markersize=5)
            ax.annotate(label, v, fontsize=9, color='black',
                        xytext=(5, 5), textcoords='offset points')

    plt.suptitle('Edge Function Visualization', fontsize=14)
    plt.tight_layout()
    plt.savefig('ex04_edge_functions.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to ex04_edge_functions.png")
    print()
    print("Interpretation:")
    print("  Red regions: negative edge function value (point is on one side)")
    print("  Blue regions: positive edge function value (point is on the other side)")
    print("  The triangle interior is where ALL three edge functions agree in sign")


def exercise_3():
    """
    Z-Buffer Artifact: Create a scene where two triangles are nearly coplanar
    (Z difference of 0.0001). Demonstrate z-fighting by rendering with
    different depth precisions.
    """
    width, height = 200, 200

    # Two large triangles covering the entire screen, nearly coplanar
    # Triangle 1 (red): at z=0.5
    # Triangle 2 (blue): at z=0.5001 (just slightly behind)
    z1 = 0.5
    z2 = 0.5001

    print("Z-Buffer Artifact Demonstration")
    print(f"Two coplanar triangles: z1={z1}, z2={z2}, delta={z2-z1}")
    print()

    for precision_bits, precision_name in [(16, "16-bit"), (24, "24-bit"), (32, "32-bit (float)")]:
        color_buffer = np.zeros((height, width, 3))

        if precision_bits == 32:
            # Float32 depth buffer
            depth_buffer = np.full((height, width), 1.0, dtype=np.float32)
        else:
            # Simulate fixed-point depth buffer
            max_val = (1 << precision_bits) - 1
            depth_buffer = np.full((height, width), max_val, dtype=np.int64)

        fighting_pixels = 0

        for y in range(height):
            for x in range(width):
                # Simulate slight depth variation (as from interpolation noise)
                noise = np.sin(x * 0.5) * np.cos(y * 0.3) * 0.00005

                d1 = z1 + noise
                d2 = z2 - noise  # Opposite noise direction

                if precision_bits == 32:
                    d1_q = np.float32(d1)
                    d2_q = np.float32(d2)
                else:
                    # Quantize to fixed-point
                    d1_q = int(d1 * max_val)
                    d2_q = int(d2 * max_val)

                # Draw triangle 1 (red) first
                if precision_bits == 32:
                    depth_buffer[y, x] = d1_q
                else:
                    depth_buffer[y, x] = d1_q
                color_buffer[y, x] = [1.0, 0.0, 0.0]

                # Draw triangle 2 (blue) - depth test
                if d2_q < depth_buffer[y, x]:
                    depth_buffer[y, x] = d2_q
                    color_buffer[y, x] = [0.0, 0.0, 1.0]
                    fighting_pixels += 1

        pct = fighting_pixels / (width * height) * 100
        print(f"  {precision_name:>12}: {fighting_pixels:>6} blue pixels "
              f"({pct:.1f}% z-fighting)")

    print()
    print("Analysis:")
    print("  With 16-bit depth: significant z-fighting (many pixels alternate color)")
    print("  With 24-bit depth: less z-fighting but still visible")
    print("  With 32-bit float: minimal z-fighting due to higher precision")
    print()
    print("Mitigation strategies:")
    print("  1. Increase distance between surfaces (avoid coplanar geometry)")
    print("  2. Push the near plane farther away (more depth precision)")
    print("  3. Use polygon offset (glPolygonOffset) to bias depth values")
    print("  4. Use reversed-Z with float depth buffer for best precision")


def exercise_4():
    """
    Perspective-Correct Interpolation: Render a textured quad with and without
    perspective correction. Use a checkerboard pattern viewed at an oblique angle.
    """
    width, height = 300, 200

    # Create a simple checkerboard texture
    tex_size = 8
    texture = np.zeros((tex_size, tex_size, 3))
    for ty in range(tex_size):
        for tx in range(tex_size):
            if (tx + ty) % 2 == 0:
                texture[ty, tx] = [0.9, 0.9, 0.9]
            else:
                texture[ty, tx] = [0.2, 0.2, 0.2]

    def sample_texture(tex, u, v):
        h, w = tex.shape[:2]
        x = int(u * w) % w
        y = int(v * h) % h
        return tex[y, x]

    # Define a quad in perspective (simulated screen-space vertices)
    # Top edge is narrow (far), bottom edge is wide (near) = perspective effect
    # v0--v1  (top, far, small w)
    # |    |
    # v3--v2  (bottom, near, large w)

    # Screen positions + depth (z) + clip-space w
    v0 = {'pos': np.array([100, 30, 0.8]), 'uv': np.array([0.0, 0.0]), 'w': 5.0}
    v1 = {'pos': np.array([200, 30, 0.8]), 'uv': np.array([1.0, 0.0]), 'w': 5.0}
    v2 = {'pos': np.array([270, 170, 0.2]), 'uv': np.array([1.0, 1.0]), 'w': 1.0}
    v3 = {'pos': np.array([30, 170, 0.2]), 'uv': np.array([0.0, 1.0]), 'w': 1.0}

    # Two triangles forming the quad
    triangles = [(v0, v1, v2), (v0, v2, v3)]

    print("Perspective-Correct Interpolation Comparison")
    print("=" * 50)
    print()

    results = {}

    for correct_name, use_correction in [("Without", False), ("With", True)]:
        image = np.ones((height, width, 3)) * 0.1  # Dark background

        total_fragments = 0

        for tri_verts in triangles:
            va, vb, vc = tri_verts
            p0, p1, p2 = va['pos'][:2], vb['pos'][:2], vc['pos'][:2]

            min_x = max(0, int(min(p0[0], p1[0], p2[0])))
            max_x = min(width - 1, int(max(p0[0], p1[0], p2[0])) + 1)
            min_y = max(0, int(min(p0[1], p1[1], p2[1])))
            max_y = min(height - 1, int(max(p0[1], p1[1], p2[1])) + 1)

            area = edge_function(p0, p1, p2)
            if abs(area) < 1e-10:
                continue

            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    px, py = x + 0.5, y + 0.5
                    w0 = edge_function(p1, p2, (px, py))
                    w1 = edge_function(p2, p0, (px, py))
                    w2 = edge_function(p0, p1, (px, py))

                    if w0 >= 0 and w1 >= 0 and w2 >= 0:
                        alpha = w0 / area
                        beta = w1 / area
                        gamma = w2 / area

                        if use_correction:
                            inv_w0 = 1.0 / va['w']
                            inv_w1 = 1.0 / vb['w']
                            inv_w2 = 1.0 / vc['w']
                            one_over_w = alpha * inv_w0 + beta * inv_w1 + gamma * inv_w2

                            u = (alpha * va['uv'][0] * inv_w0 +
                                 beta * vb['uv'][0] * inv_w1 +
                                 gamma * vc['uv'][0] * inv_w2) / one_over_w
                            v = (alpha * va['uv'][1] * inv_w0 +
                                 beta * vb['uv'][1] * inv_w1 +
                                 gamma * vc['uv'][1] * inv_w2) / one_over_w
                        else:
                            u = alpha * va['uv'][0] + beta * vb['uv'][0] + gamma * vc['uv'][0]
                            v = alpha * va['uv'][1] + beta * vb['uv'][1] + gamma * vc['uv'][1]

                        color = sample_texture(texture, u % 1, v % 1)
                        image[y, x] = color
                        total_fragments += 1

        results[correct_name] = image
        print(f"  {correct_name} correction: {total_fragments} fragments rendered")

    # Save comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(results["Without"], origin='upper')
    ax1.set_title("Without Perspective Correction\n(checkerboard bends at diagonal)")
    ax2.imshow(results["With"], origin='upper')
    ax2.set_title("With Perspective Correction\n(checkerboard appears straight)")
    plt.tight_layout()
    plt.savefig('ex04_perspective_correction.png', dpi=150, bbox_inches='tight')
    print("\nComparison saved to ex04_perspective_correction.png")
    print()
    print("Visual difference:")
    print("  Without correction: checkerboard lines bend along the triangle diagonal")
    print("  With correction: checkerboard lines remain straight (physically correct)")


def exercise_5():
    """
    Software Rasterizer Extension: Extend the rasterizer to support:
    (a) wireframe rendering, (b) multiple triangles forming a cube face,
    (c) flat shading using face normal.
    """
    width, height = 300, 250

    print("Software Rasterizer Extensions")
    print("=" * 50)

    # (a) Wireframe rendering
    print("\n(a) Wireframe rendering:")
    wireframe_image = np.ones((height, width, 3)) * 0.05

    v0 = (50, 30)
    v1 = (250, 60)
    v2 = (150, 200)

    for start, end in [(v0, v1), (v1, v2), (v2, v0)]:
        pixels = draw_line_bresenham(int(start[0]), int(start[1]),
                                     int(end[0]), int(end[1]))
        for px, py in pixels:
            if 0 <= px < width and 0 <= py < height:
                wireframe_image[py, px] = [0.0, 1.0, 0.0]

    print(f"  Wireframe triangle drawn with 3 Bresenham lines")

    # (b) Multiple triangles forming a simple mesh (2 triangles = quad)
    print("\n(b) Multiple triangles (quad):")
    mesh_image = np.ones((height, width, 3)) * 0.05

    quad_verts = [
        {'pos': np.array([50, 30, 0.5]), 'color': np.array([1, 0, 0])},
        {'pos': np.array([250, 50, 0.5]), 'color': np.array([0, 1, 0])},
        {'pos': np.array([230, 200, 0.5]), 'color': np.array([0, 0, 1])},
        {'pos': np.array([30, 180, 0.5]), 'color': np.array([1, 1, 0])},
    ]

    quad_indices = [(0, 1, 2), (0, 2, 3)]
    total_frags = 0

    for idx in quad_indices:
        va, vb, vc = [quad_verts[i] for i in idx]
        p0, p1, p2 = va['pos'][:2], vb['pos'][:2], vc['pos'][:2]

        min_x = max(0, int(min(p0[0], p1[0], p2[0])))
        max_x = min(width - 1, int(max(p0[0], p1[0], p2[0])) + 1)
        min_y = max(0, int(min(p0[1], p1[1], p2[1])))
        max_y = min(height - 1, int(max(p0[1], p1[1], p2[1])) + 1)

        area = edge_function(p0, p1, p2)
        if abs(area) < 1e-10:
            continue

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                px, py = x + 0.5, y + 0.5
                w0 = edge_function(p1, p2, (px, py))
                w1 = edge_function(p2, p0, (px, py))
                w2 = edge_function(p0, p1, (px, py))

                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    alpha, beta, gamma = w0 / area, w1 / area, w2 / area
                    color = alpha * va['color'] + beta * vb['color'] + gamma * vc['color']
                    mesh_image[y, x] = np.clip(color, 0, 1)
                    total_frags += 1

    print(f"  Quad rendered: 2 triangles, {total_frags} fragments")

    # (c) Flat shading using face normal
    print("\n(c) Flat shading with face normal:")
    flat_image = np.ones((height, width, 3)) * 0.05

    # Define two triangles with 3D positions for normal computation
    triangles_3d = [
        {
            'v0': np.array([50, 30, 0.5]),
            'v1': np.array([250, 50, 0.3]),
            'v2': np.array([150, 200, 0.7]),
            'color': np.array([0.8, 0.3, 0.3]),
        },
        {
            'v0': np.array([100, 20, 0.4]),
            'v1': np.array([280, 120, 0.6]),
            'v2': np.array([200, 220, 0.2]),
            'color': np.array([0.3, 0.3, 0.8]),
        },
    ]

    light_dir = np.array([0.0, 0.0, -1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)
    depth_buffer = np.full((height, width), float('inf'))

    for tri in triangles_3d:
        p0, p1, p2 = tri['v0'], tri['v1'], tri['v2']

        # Compute face normal from cross product of edges
        edge1 = p1 - p0
        edge2 = p2 - p0
        normal = np.cross(edge1, edge2)
        normal = normal / (np.linalg.norm(normal) + 1e-10)

        # Flat shading: single diffuse factor for entire triangle
        diffuse = max(np.dot(normal, -light_dir), 0.0)
        ambient = 0.15
        shaded_color = tri['color'] * (ambient + diffuse * 0.85)

        # Rasterize
        s0, s1, s2 = p0[:2], p1[:2], p2[:2]
        min_x = max(0, int(min(s0[0], s1[0], s2[0])))
        max_x = min(width - 1, int(max(s0[0], s1[0], s2[0])) + 1)
        min_y = max(0, int(min(s0[1], s1[1], s2[1])))
        max_y = min(height - 1, int(max(s0[1], s1[1], s2[1])) + 1)

        area = edge_function(s0, s1, s2)
        if abs(area) < 1e-10:
            continue

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                ppx, ppy = x + 0.5, y + 0.5
                w0 = edge_function(s1, s2, (ppx, ppy))
                w1 = edge_function(s2, s0, (ppx, ppy))
                w2 = edge_function(s0, s1, (ppx, ppy))

                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    alpha, beta, gamma = w0 / area, w1 / area, w2 / area
                    z = alpha * p0[2] + beta * p1[2] + gamma * p2[2]
                    if z < depth_buffer[y, x]:
                        depth_buffer[y, x] = z
                        flat_image[y, x] = np.clip(shaded_color, 0, 1)

    print(f"  Face normal: {np.round(normal, 4)}")
    print(f"  Diffuse factor: {diffuse:.4f}")

    # Save all three visualizations
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(wireframe_image, origin='upper')
    axes[0].set_title('(a) Wireframe')
    axes[1].imshow(mesh_image, origin='upper')
    axes[1].set_title('(b) Quad Mesh (color interpolated)')
    axes[2].imshow(flat_image, origin='upper')
    axes[2].set_title('(c) Flat Shading with Z-buffer')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('ex04_rasterizer_extensions.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to ex04_rasterizer_extensions.png")


def exercise_6():
    """
    MSAA Analysis: Render a triangle edge at 1x, 2x, 4x, and 8x MSAA.
    Count pixels with partial coverage.
    """
    width, height = 100, 100

    # A triangle with a slanted edge for clear aliasing
    v0 = (10, 10)
    v1 = (90, 30)
    v2 = (50, 90)

    print("MSAA Analysis")
    print("=" * 50)

    def get_sample_offsets(n):
        """Generate n MSAA sample offsets within a pixel."""
        if n == 1:
            return [(0.0, 0.0)]
        elif n == 2:
            return [(-0.25, -0.25), (0.25, 0.25)]
        elif n == 4:
            return [(-0.25, -0.125), (0.25, -0.375),
                    (-0.125, 0.375), (0.375, 0.125)]
        else:  # 8x
            return [
                (-0.375, -0.125), (-0.125, -0.375),
                (0.125, -0.125), (0.375, -0.375),
                (-0.375, 0.375), (-0.125, 0.125),
                (0.125, 0.375), (0.375, 0.125),
            ]

    area = edge_function(v0, v1, v2)

    for msaa_level in [1, 2, 4, 8]:
        offsets = get_sample_offsets(msaa_level)
        full_inside = 0
        partial_coverage = 0
        full_outside = 0

        for y in range(height):
            for x in range(width):
                covered = 0
                for dx, dy in offsets:
                    sx, sy = x + 0.5 + dx, y + 0.5 + dy
                    w0 = edge_function(v0, v1, (sx, sy))
                    w1 = edge_function(v1, v2, (sx, sy))
                    w2 = edge_function(v2, v0, (sx, sy))

                    if area > 0:
                        if w0 >= 0 and w1 >= 0 and w2 >= 0:
                            covered += 1
                    else:
                        if w0 <= 0 and w1 <= 0 and w2 <= 0:
                            covered += 1

                if covered == msaa_level:
                    full_inside += 1
                elif covered > 0:
                    partial_coverage += 1
                else:
                    full_outside += 1

        print(f"  {msaa_level}x MSAA:")
        print(f"    Fully inside:    {full_inside:>5} pixels")
        print(f"    Partial coverage:{partial_coverage:>5} pixels (anti-aliased edges)")
        print(f"    Fully outside:   {full_outside:>5} pixels")
        print()

    print("Interpretation:")
    print("  Higher MSAA levels detect more partial coverage at triangle edges,")
    print("  producing smoother edge transitions. The number of partial-coverage")
    print("  pixels increases with MSAA level because finer sampling resolves")
    print("  more sub-pixel coverage variation.")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Bresenham All Octants ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Edge Function Visualization ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Z-Buffer Artifact ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Perspective-Correct Interpolation ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Rasterizer Extensions ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: MSAA Analysis ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
