"""
Software Rasterizer from Scratch
=================================

Implements a minimal but complete rasterization pipeline:
  1. Bresenham line drawing (integer arithmetic only)
  2. Triangle rasterization via edge functions (barycentric coords)
  3. Z-buffer depth testing
  4. Vertex attribute interpolation (color, UV)
  5. Rendering a simple 3D scene

Rasterization is how GPUs turn triangles into pixels.  Understanding it
at the software level demystifies what happens between your vertex shader
output and the pixels on screen.

Why edge functions instead of scanline?  Edge functions (equivalent to
barycentric coordinates) are more GPU-friendly (embarrassingly parallel)
and naturally handle attribute interpolation.  Scanline was historically
used for CPU rendering but is largely obsolete.

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional

# ---------------------------------------------------------------------------
# 1. Framebuffer
# ---------------------------------------------------------------------------


class Framebuffer:
    """A simple software framebuffer with color buffer and Z-buffer.

    Why separate color and depth?  The depth buffer is tested *before*
    writing color, allowing us to skip occluded pixels early.  This is
    the same architecture real GPUs use.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # Color buffer: RGBA float [0, 1] -- float for easy blending later
        self.color = np.zeros((height, width, 3), dtype=float)
        # Depth buffer: initialized to +inf (furthest possible)
        # Why +inf?  Any real depth will be "closer" and pass the test.
        self.depth = np.full((height, width), np.inf, dtype=float)

    def clear(self, color: Tuple[float, float, float] = (0.1, 0.1, 0.15)):
        """Clear both buffers.  Background color defaults to dark blue-gray."""
        self.color[:] = color
        self.depth[:] = np.inf

    def set_pixel(self, x: int, y: int, z: float,
                  color: np.ndarray) -> bool:
        """Write a pixel if it passes the depth test.

        Returns True if the pixel was written (passed depth test).

        Why return a bool?  Useful for debugging -- tells you how many
        pixels were actually drawn vs rejected by depth testing.
        """
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False

        # Depth test: only write if this fragment is closer
        if z < self.depth[y, x]:
            self.depth[y, x] = z
            self.color[y, x] = np.clip(color, 0, 1)
            return True
        return False

    def to_image(self) -> np.ndarray:
        """Return the color buffer as a displayable image.

        Why flip vertically?  Our coordinate system has Y=0 at bottom,
        but image arrays have row 0 at top.  Flipping fixes this mismatch.
        """
        return np.flipud(self.color)


# ---------------------------------------------------------------------------
# 2. Bresenham Line Drawing
# ---------------------------------------------------------------------------

def bresenham_line(fb: Framebuffer, x0: int, y0: int, x1: int, y1: int,
                   color: np.ndarray = np.array([1, 1, 1])):
    """Draw a line using Bresenham's algorithm.

    Why Bresenham?  It uses only integer arithmetic (additions and
    comparisons) -- no floating point, no division.  This made it
    essential for early hardware and it remains the standard reference
    for line rasterization.

    The key insight: at each step, we choose between two candidate
    pixels based on which is closer to the true line.  The "error"
    accumulator tracks the fractional deviation using only integers.
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1  # Step direction in x
    sy = 1 if y0 < y1 else -1  # Step direction in y

    # Why this error initialization?  It starts at dx - dy, which
    # biases the first step direction based on the line's slope.
    err = dx - dy

    while True:
        fb.set_pixel(x0, y0, 0.0, color)

        if x0 == x1 and y0 == y1:
            break

        # Double the error to avoid floating-point half-step comparisons
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def demo_bresenham():
    """Demonstrate Bresenham line drawing with various slopes."""
    fb = Framebuffer(200, 200)
    fb.clear()

    center_x, center_y = 100, 100

    # Draw lines radiating from center at different angles
    # Why radiating pattern?  It tests all octants of the algorithm,
    # which is important because Bresenham handles each octant differently.
    colors = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([1, 1, 0]),
        np.array([1, 0, 1]),
        np.array([0, 1, 1]),
        np.array([1, 0.5, 0]),
        np.array([0.5, 1, 0]),
    ]

    for i, angle_deg in enumerate(range(0, 360, 45)):
        angle = np.radians(angle_deg)
        ex = int(center_x + 80 * np.cos(angle))
        ey = int(center_y + 80 * np.sin(angle))
        bresenham_line(fb, center_x, center_y, ex, ey, colors[i % len(colors)])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(fb.to_image(), interpolation='nearest')
    ax.set_title("Bresenham Line Drawing (8 directions)", fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_03_bresenham.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 3. Triangle Rasterization (Edge Functions / Barycentric Coordinates)
# ---------------------------------------------------------------------------

def edge_function(v0: np.ndarray, v1: np.ndarray, p: np.ndarray) -> float:
    """Compute the edge function for the edge v0->v1 evaluated at point p.

    The edge function is the 2D cross product: (v1-v0) x (p-v0).
    It returns:
      > 0 if p is to the LEFT of the edge (inside for CCW triangles)
      = 0 if p is exactly ON the edge
      < 0 if p is to the RIGHT

    Why this function?  It simultaneously:
    1. Tests if a point is inside the triangle (all three edges positive)
    2. Computes barycentric coordinates (proportional to triangle areas)
    3. Is trivially parallelizable (each pixel is independent)
    """
    return (v1[0] - v0[0]) * (p[1] - v0[1]) - (v1[1] - v0[1]) * (p[0] - v0[0])


def rasterize_triangle(fb: Framebuffer,
                        v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                        c0: np.ndarray, c1: np.ndarray, c2: np.ndarray,
                        z0: float, z1: float, z2: float):
    """Rasterize a single triangle with per-vertex colors and depth.

    Parameters
    ----------
    fb      : Target framebuffer
    v0..v2  : 2D screen-space vertex positions (x, y)
    c0..c2  : Per-vertex colors (RGB, [0..1])
    z0..z2  : Per-vertex depth values

    Why bounding box + edge test?  We could test every pixel in the
    framebuffer, but that's O(width * height).  The bounding box clips
    to only the relevant rectangular region -- a simple but effective
    optimization that real GPUs also use (in a more sophisticated form
    via tile-based rasterization).
    """
    # Compute bounding box of the triangle
    min_x = max(0, int(np.floor(min(v0[0], v1[0], v2[0]))))
    max_x = min(fb.width - 1, int(np.ceil(max(v0[0], v1[0], v2[0]))))
    min_y = max(0, int(np.floor(min(v0[1], v1[1], v2[1]))))
    max_y = min(fb.height - 1, int(np.ceil(max(v0[1], v1[1], v2[1]))))

    # Total area of the triangle (for normalizing barycentric coords)
    area = edge_function(v0, v1, v2)
    if abs(area) < 1e-10:
        return  # Degenerate triangle -- zero area, nothing to draw

    # Why 1/area?  We'll divide by area for every pixel, so precomputing
    # the reciprocal avoids repeated division (a classic optimization).
    inv_area = 1.0 / area

    # Iterate over every pixel in the bounding box
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            p = np.array([x + 0.5, y + 0.5])  # Sample at pixel center

            # Compute barycentric coordinates via edge functions
            # w0 corresponds to v0's weight (area of sub-triangle opposite v0)
            w0 = edge_function(v1, v2, p) * inv_area
            w1 = edge_function(v2, v0, p) * inv_area
            w2 = edge_function(v0, v1, p) * inv_area

            # Point is inside triangle if all weights are non-negative
            # Why >= 0?  We include edges to avoid gaps between adjacent triangles.
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                # Interpolate depth
                z = w0 * z0 + w1 * z1 + w2 * z2

                # Interpolate color
                color = w0 * c0 + w1 * c1 + w2 * c2

                fb.set_pixel(x, y, z, color)


def demo_triangle_rasterization():
    """Rasterize triangles with per-vertex colors, showing interpolation."""
    fb = Framebuffer(300, 300)
    fb.clear()

    # Triangle 1: RGB corners
    # Why RGB at vertices?  It clearly shows barycentric interpolation --
    # the smooth color gradient proves the weights are computed correctly.
    v0, v1, v2 = np.array([50, 50]), np.array([250, 80]), np.array([150, 280])
    c0, c1, c2 = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    rasterize_triangle(fb, v0, v1, v2, c0, c1, c2, 0.5, 0.5, 0.5)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(fb.to_image(), interpolation='nearest')
    ax.set_title("Triangle Rasterization with Barycentric Color Interpolation",
                 fontsize=11)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_03_triangle_raster.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 4. Z-Buffer Depth Testing Demo
# ---------------------------------------------------------------------------

def demo_depth_testing():
    """Show how the Z-buffer resolves occlusion between overlapping triangles.

    Without depth testing, the last-drawn triangle always appears on top,
    regardless of its actual 3D position.  The Z-buffer fixes this by
    tracking the closest depth at each pixel.
    """
    fb = Framebuffer(300, 300)
    fb.clear()

    # Two overlapping triangles at different depths
    # Triangle A (red/yellow) is closer (smaller z = closer to camera)
    # Triangle B (blue/cyan) is further away

    # Triangle B (further, drawn first)
    rasterize_triangle(
        fb,
        np.array([30, 50]), np.array([270, 100]), np.array([150, 270]),
        np.array([0, 0, 1]), np.array([0, 1, 1]), np.array([0.2, 0.2, 1]),
        z0=0.7, z1=0.7, z2=0.7
    )

    # Triangle A (closer, drawn second -- but depth test prevents it from
    # being hidden under B where A is actually further)
    # Here, A is genuinely closer everywhere, so it overwrites B
    rasterize_triangle(
        fb,
        np.array([80, 30]), np.array([280, 180]), np.array([50, 250]),
        np.array([1, 0, 0]), np.array([1, 1, 0]), np.array([1, 0.5, 0]),
        z0=0.3, z1=0.3, z2=0.3
    )

    # Triangle C (partially behind A, partially in front of B)
    # This has a depth gradient so some pixels pass and some fail
    rasterize_triangle(
        fb,
        np.array([120, 80]), np.array([290, 250]), np.array([20, 200]),
        np.array([0, 1, 0]), np.array([0.5, 1, 0]), np.array([0, 1, 0.5]),
        z0=0.1, z1=0.9, z2=0.5  # Depth varies across the triangle
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(fb.to_image(), interpolation='nearest')
    ax1.set_title("Color Buffer (3 overlapping triangles)", fontsize=11)
    ax1.axis('off')

    # Show depth buffer as grayscale
    depth_vis = np.flipud(fb.depth.copy())
    depth_vis[depth_vis == np.inf] = 1.0  # Background = max depth
    ax2.imshow(depth_vis, cmap='gray_r', interpolation='nearest')
    ax2.set_title("Depth Buffer (darker = closer)", fontsize=11)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_03_depth_test.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 5. Full 3D Rasterization Pipeline
# ---------------------------------------------------------------------------

@dataclass
class Vertex:
    """A vertex with position, color, and (optional) UV coordinates.

    Why a dataclass?  Vertices have multiple attributes that travel
    together through the pipeline.  A struct/dataclass is the natural
    representation -- it's essentially what a vertex buffer stores.
    """
    position: np.ndarray  # (x, y, z) in object space
    color: np.ndarray     # (r, g, b)
    uv: Optional[np.ndarray] = None  # (u, v) texture coordinates


def look_at(eye, target, up):
    """Minimal LookAt matrix (repeated from 02 for self-containment)."""
    eye, target, up = [np.asarray(v, dtype=float) for v in (eye, target, up)]
    f = eye - target
    f /= np.linalg.norm(f)
    r = np.cross(up, f)
    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    return np.array([
        [r[0], r[1], r[2], -np.dot(r, eye)],
        [u[0], u[1], u[2], -np.dot(u, eye)],
        [f[0], f[1], f[2], -np.dot(f, eye)],
        [0, 0, 0, 1],
    ], dtype=float)


def perspective(fov_deg, aspect, near, far):
    """Minimal perspective matrix (repeated from 02 for self-containment)."""
    t = np.tan(np.radians(fov_deg) / 2)
    return np.array([
        [1/(aspect*t), 0, 0, 0],
        [0, 1/t, 0, 0],
        [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
        [0, 0, -1, 0],
    ], dtype=float)


def project_vertex(v: np.ndarray, mvp: np.ndarray,
                   width: int, height: int) -> Tuple[np.ndarray, float]:
    """Transform a 3D point through the MVP pipeline to screen coordinates.

    Returns (screen_xy, ndc_z) where screen_xy is in pixel coordinates
    and ndc_z is the depth for Z-buffer testing.

    Why return NDC z separately?  Screen coordinates are 2D (pixel x, y)
    but we still need depth for the Z-buffer.  NDC z is in [-1, 1].
    """
    # Apply MVP
    clip = mvp @ np.array([*v, 1.0])

    # Perspective divide
    if abs(clip[3]) < 1e-10:
        return np.array([0, 0]), 0  # Degenerate -- point at infinity

    ndc = clip[:3] / clip[3]

    # Viewport transform: NDC [-1, 1] -> screen [0, width/height]
    # Why (ndc + 1) / 2?  NDC range is [-1, 1], we map to [0, 1] first,
    # then scale to pixel dimensions.
    sx = (ndc[0] + 1) * 0.5 * width
    sy = (ndc[1] + 1) * 0.5 * height

    return np.array([sx, sy]), ndc[2]


def render_scene(fb: Framebuffer, triangles: list,
                 mvp: np.ndarray):
    """Render a list of triangles through the full pipeline.

    Each triangle is a tuple of three Vertex objects.

    Why process one triangle at a time?  In a real GPU, triangles are
    processed in parallel, but the *logical* pipeline is per-triangle:
    vertex transform -> clipping -> rasterization -> fragment processing.
    """
    w, h = fb.width, fb.height

    for tri in triangles:
        v0, v1, v2 = tri

        # Vertex shader stage: transform to screen space
        s0, z0 = project_vertex(v0.position, mvp, w, h)
        s1, z1 = project_vertex(v1.position, mvp, w, h)
        s2, z2 = project_vertex(v2.position, mvp, w, h)

        # Simple back-face culling using screen-space winding order
        # Why cull?  Triangles facing away from the camera are invisible.
        # Discarding them early saves ~50% of rasterization work.
        edge = (s1[0] - s0[0]) * (s2[1] - s0[1]) - (s1[1] - s0[1]) * (s2[0] - s0[0])
        if edge < 0:
            continue  # Back-facing -- skip

        # Rasterize with per-vertex colors
        rasterize_triangle(
            fb, s0, s1, s2,
            v0.color, v1.color, v2.color,
            z0, z1, z2
        )


def make_colored_cube_triangles() -> list:
    """Create a cube as a list of triangle tuples with per-vertex colors.

    Why triangles instead of quads?  GPUs only rasterize triangles.
    Every quad must be split into two triangles before rasterization.
    """
    # Cube vertices
    positions = np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
    ], dtype=float)

    # Face colors (each face gets a distinct color for visual clarity)
    face_colors = [
        np.array([1.0, 0.3, 0.3]),  # back   - red
        np.array([0.3, 1.0, 0.3]),  # front  - green
        np.array([0.3, 0.3, 1.0]),  # bottom - blue
        np.array([1.0, 1.0, 0.3]),  # top    - yellow
        np.array([1.0, 0.3, 1.0]),  # left   - magenta
        np.array([0.3, 1.0, 1.0]),  # right  - cyan
    ]

    # Each face = 2 triangles
    face_indices = [
        [(0, 1, 2), (0, 2, 3)],  # back
        [(4, 6, 5), (4, 7, 6)],  # front
        [(0, 5, 1), (0, 4, 5)],  # bottom
        [(2, 7, 3), (2, 6, 7)],  # top
        [(0, 3, 7), (0, 7, 4)],  # left
        [(1, 5, 6), (1, 6, 2)],  # right
    ]

    triangles = []
    for face_idx, tris in enumerate(face_indices):
        color = face_colors[face_idx]
        for (i0, i1, i2) in tris:
            # Slight color variation per vertex for visible interpolation
            c0 = np.clip(color * 0.8, 0, 1)
            c1 = np.clip(color * 1.0, 0, 1)
            c2 = np.clip(color * 0.6, 0, 1)
            triangles.append((
                Vertex(positions[i0], c0),
                Vertex(positions[i1], c1),
                Vertex(positions[i2], c2),
            ))

    return triangles


def demo_full_pipeline():
    """Render a 3D cube using the complete software rasterization pipeline.

    This is the culmination: vertex transform -> back-face cull ->
    triangle rasterize -> depth test -> framebuffer output.
    """
    fb = Framebuffer(400, 400)
    fb.clear(color=(0.05, 0.05, 0.1))

    triangles = make_colored_cube_triangles()

    # Set up camera
    V = look_at(eye=[3, 2.5, 4], target=[0, 0, 0], up=[0, 1, 0])
    P = perspective(fov_deg=60, aspect=1.0, near=0.1, far=100)

    # Model transform: slight rotation to see 3 faces
    cos30, sin30 = np.cos(np.radians(25)), np.sin(np.radians(25))
    cos15, sin15 = np.cos(np.radians(15)), np.sin(np.radians(15))
    Ry = np.array([[cos30, 0, sin30, 0], [0, 1, 0, 0],
                    [-sin30, 0, cos30, 0], [0, 0, 0, 1]])
    Rx = np.array([[1, 0, 0, 0], [0, cos15, -sin15, 0],
                    [0, sin15, cos15, 0], [0, 0, 0, 1]])
    M = Ry @ Rx

    mvp = P @ V @ M
    render_scene(fb, triangles, mvp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(fb.to_image(), interpolation='nearest')
    ax1.set_title("Software Rasterizer: 3D Cube", fontsize=12)
    ax1.axis('off')

    # Depth buffer visualization
    depth_vis = np.flipud(fb.depth.copy())
    depth_vis[depth_vis == np.inf] = np.nan
    ax2.imshow(depth_vis, cmap='viridis', interpolation='nearest')
    ax2.set_title("Depth Buffer", fontsize=12)
    ax2.axis('off')

    plt.suptitle("Complete Software Rasterization Pipeline", fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_03_full_rasterizer.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 6. Multi-object scene with depth sorting
# ---------------------------------------------------------------------------

def demo_multi_object_scene():
    """Render multiple objects in a scene, demonstrating proper depth sorting.

    Why is this test important?  Without the Z-buffer, object draw order
    determines visibility.  With it, objects can intersect and overlap
    freely and the correct pixel always wins.
    """
    fb = Framebuffer(500, 400)
    fb.clear(color=(0.08, 0.08, 0.12))

    V = look_at(eye=[5, 4, 7], target=[0, 0, 0], up=[0, 1, 0])
    P = perspective(fov_deg=50, aspect=500 / 400, near=0.1, far=100)

    cube_tris = make_colored_cube_triangles()

    # Render 3 cubes at different positions (some overlapping in screen space)
    model_transforms = [
        np.eye(4),  # Center
        np.array([[1, 0, 0, 2.5], [0, 1, 0, 0], [0, 0, 1, -1],
                  [0, 0, 0, 1]], dtype=float),  # Right-back
        np.array([[0.7, 0, 0, -1.5], [0, 0.7, 0, 1], [0, 0, 0.7, 0.5],
                  [0, 0, 0, 1]], dtype=float),  # Left-up (smaller)
    ]

    for M in model_transforms:
        mvp = P @ V @ M
        render_scene(fb, cube_tris, mvp)

    # Also add a ground plane (two triangles)
    ground_y = -1.2
    gc = np.array([0.3, 0.3, 0.25])
    ground_tris = [
        (Vertex(np.array([-5, ground_y, -5]), gc * 0.7),
         Vertex(np.array([5, ground_y, -5]), gc * 1.0),
         Vertex(np.array([5, ground_y, 5]), gc * 0.8)),
        (Vertex(np.array([-5, ground_y, -5]), gc * 0.7),
         Vertex(np.array([5, ground_y, 5]), gc * 0.8),
         Vertex(np.array([-5, ground_y, 5]), gc * 0.9)),
    ]
    mvp_ground = P @ V @ np.eye(4)
    render_scene(fb, ground_tris, mvp_ground)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(fb.to_image(), interpolation='nearest')
    ax.set_title("Multi-Object Scene with Z-Buffer", fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_03_multi_object.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Software Rasterizer")
    print("=" * 60)

    print("\n[1/4] Bresenham line drawing...")
    demo_bresenham()

    print("\n[2/4] Triangle rasterization with barycentric interpolation...")
    demo_triangle_rasterization()

    print("\n[3/4] Z-buffer depth testing...")
    demo_depth_testing()

    print("\n[4/4] Full 3D rasterization pipeline...")
    demo_full_pipeline()

    print("\n[5/5] Multi-object scene...")
    demo_multi_object_scene()

    print("\nDone!")


if __name__ == "__main__":
    main()
