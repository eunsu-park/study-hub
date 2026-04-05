"""
Graphics Pipeline Simulation
==============================

Implements a simplified but complete real-time graphics pipeline in software:
  1. Application stage (scene description, vertex buffers)
  2. Vertex processing (model-view-projection transforms)
  3. Primitive assembly (grouping vertices into triangles)
  4. Clipping (discarding geometry outside the view frustum)
  5. Rasterization (converting triangles to fragments)
  6. Fragment processing (per-pixel shading, depth testing)
  7. Framebuffer output (final image)

This example traces a single frame through every stage so you can see
what the GPU does between "draw call" and "pixels on screen."  Each
stage is isolated into its own function, mirroring the real pipeline.

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

# ---------------------------------------------------------------------------
# 1. Data structures -- what flows through the pipeline
# ---------------------------------------------------------------------------


@dataclass
class Vertex:
    """A vertex carrying position and color through the pipeline.

    Why bundle position and color together?  In real GPUs, a vertex
    buffer stores all per-vertex attributes (position, normal, UV,
    color) as an interleaved or structured array.  Each pipeline stage
    reads and transforms these attributes.
    """
    position: np.ndarray   # (x, y, z) -- changes space at each stage
    color: np.ndarray      # (r, g, b)


@dataclass
class Fragment:
    """A fragment is a candidate pixel produced by rasterization.

    Why "fragment" and not "pixel"?  A fragment carries interpolated
    attributes and depth.  It only becomes a pixel if it passes the
    depth test.  Multiple fragments can compete for the same pixel.
    """
    x: int
    y: int
    depth: float
    color: np.ndarray


@dataclass
class Triangle:
    """Three vertices forming a renderable primitive."""
    v0: Vertex
    v1: Vertex
    v2: Vertex


# ---------------------------------------------------------------------------
# 2. Application stage -- scene description
# ---------------------------------------------------------------------------


def application_stage() -> Tuple[List[Vertex], List[Tuple[int, int, int]]]:
    """Define the scene as vertex and index buffers.

    Why separate vertex and index buffers?  Vertices are shared between
    triangles (a cube corner belongs to 3 faces).  The index buffer
    avoids duplicating vertex data, saving memory and bandwidth.

    Returns (vertices, index_buffer) where each index triple defines
    one triangle.
    """
    # A simple pyramid (4 triangles + base quad = 6 triangles)
    vertices = [
        Vertex(np.array([0.0,  1.0,  0.0]), np.array([1.0, 0.9, 0.2])),   # 0: apex
        Vertex(np.array([-1.0, -0.5,  1.0]), np.array([0.8, 0.2, 0.2])),  # 1: front-left
        Vertex(np.array([1.0, -0.5,  1.0]), np.array([0.2, 0.8, 0.2])),   # 2: front-right
        Vertex(np.array([1.0, -0.5, -1.0]), np.array([0.2, 0.2, 0.8])),   # 3: back-right
        Vertex(np.array([-1.0, -0.5, -1.0]), np.array([0.8, 0.2, 0.8])),  # 4: back-left
    ]

    # Index buffer: each tuple is (i0, i1, i2) forming one triangle
    indices = [
        (0, 1, 2),  # front face
        (0, 2, 3),  # right face
        (0, 3, 4),  # back face
        (0, 4, 1),  # left face
        (1, 3, 2),  # base triangle 1
        (1, 4, 3),  # base triangle 2
    ]

    return vertices, indices


# ---------------------------------------------------------------------------
# 3. Vertex processing -- transforms and projection
# ---------------------------------------------------------------------------


def build_model_matrix(angle_y_deg: float) -> np.ndarray:
    """Build a model matrix that rotates around the Y axis."""
    c = np.cos(np.radians(angle_y_deg))
    s = np.sin(np.radians(angle_y_deg))
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1],
    ], dtype=float)


def build_view_matrix(eye, target, up) -> np.ndarray:
    """LookAt view matrix -- transforms world space to camera space."""
    eye, target, up = [np.asarray(v, float) for v in (eye, target, up)]
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


def build_projection_matrix(fov_deg, aspect, near, far) -> np.ndarray:
    """Perspective projection matrix."""
    t = np.tan(np.radians(fov_deg) / 2)
    return np.array([
        [1 / (aspect * t), 0, 0, 0],
        [0, 1 / t, 0, 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],
    ], dtype=float)


def vertex_processing(vertices: List[Vertex], mvp: np.ndarray) -> List[np.ndarray]:
    """Transform each vertex through the model-view-projection matrix.

    Returns clip-space coordinates (homogeneous 4D).

    Why return homogeneous coordinates?  The perspective divide (w division)
    must happen AFTER clipping.  Clipping in clip space is simpler because
    the view frustum becomes a regular cube: -w <= x,y,z <= w.
    """
    clip_coords = []
    for v in vertices:
        pos_h = np.array([*v.position, 1.0])
        clip = mvp @ pos_h
        clip_coords.append(clip)
    return clip_coords


# ---------------------------------------------------------------------------
# 4. Primitive assembly
# ---------------------------------------------------------------------------


def primitive_assembly(vertices: List[Vertex],
                       clip_coords: List[np.ndarray],
                       indices: List[Tuple[int, int, int]]) -> List[dict]:
    """Group vertices into triangle primitives.

    Each assembled triangle carries its three vertices in clip space
    along with their original colors.  This is the input to clipping.
    """
    triangles = []
    for i0, i1, i2 in indices:
        tri = {
            'clips': [clip_coords[i0], clip_coords[i1], clip_coords[i2]],
            'colors': [vertices[i0].color, vertices[i1].color, vertices[i2].color],
        }
        triangles.append(tri)
    return triangles


# ---------------------------------------------------------------------------
# 5. Clipping (simplified -- discard triangles fully outside frustum)
# ---------------------------------------------------------------------------


def clip_triangle(tri: dict) -> bool:
    """Simple accept/reject clipping.

    A production pipeline clips triangles that partially overlap the
    frustum boundary, generating new vertices.  Here we use the simpler
    approach: accept if any vertex is inside the frustum, reject if all
    are outside on the same side.

    Why clip at all?  Without clipping, vertices behind the camera
    produce inverted or infinitely large screen coordinates after the
    perspective divide.
    """
    for axis in range(3):
        if all(c[axis] > c[3] for c in tri['clips']):
            return False  # All vertices beyond +w on this axis
        if all(c[axis] < -c[3] for c in tri['clips']):
            return False  # All vertices beyond -w on this axis
    # Reject if all vertices are behind the near plane
    if all(c[2] < -c[3] for c in tri['clips']):
        return False
    return True


# ---------------------------------------------------------------------------
# 6. Perspective divide + viewport transform
# ---------------------------------------------------------------------------


def ndc_and_viewport(clip: np.ndarray,
                     width: int, height: int) -> Tuple[np.ndarray, float]:
    """Convert clip-space coordinates to screen-space (pixel) coordinates.

    Step 1: Perspective divide -- divide by w to get NDC [-1, 1].
    Step 2: Viewport transform -- map NDC to pixel coordinates.

    Returns (screen_xy, ndc_z).
    """
    if abs(clip[3]) < 1e-10:
        return np.array([0.0, 0.0]), 0.0
    ndc = clip[:3] / clip[3]
    sx = (ndc[0] + 1) * 0.5 * width
    sy = (ndc[1] + 1) * 0.5 * height
    return np.array([sx, sy]), ndc[2]


# ---------------------------------------------------------------------------
# 7. Rasterization -- edge-function triangle fill
# ---------------------------------------------------------------------------


def edge_function(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    """2D cross product for inside-triangle testing."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def rasterize(screen_verts: List[np.ndarray],
              depths: List[float],
              colors: List[np.ndarray],
              width: int, height: int) -> List[Fragment]:
    """Convert a screen-space triangle into fragments.

    Uses the edge-function method with barycentric interpolation for
    color and depth.  Each fragment carries interpolated attributes
    ready for the depth test.
    """
    v0, v1, v2 = screen_verts
    area = edge_function(v0, v1, v2)
    if abs(area) < 1e-6:
        return []

    inv_area = 1.0 / area
    min_x = max(0, int(np.floor(min(v0[0], v1[0], v2[0]))))
    max_x = min(width - 1, int(np.ceil(max(v0[0], v1[0], v2[0]))))
    min_y = max(0, int(np.floor(min(v0[1], v1[1], v2[1]))))
    max_y = min(height - 1, int(np.ceil(max(v0[1], v1[1], v2[1]))))

    fragments = []
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            p = np.array([x + 0.5, y + 0.5])
            w0 = edge_function(v1, v2, p) * inv_area
            w1 = edge_function(v2, v0, p) * inv_area
            w2 = edge_function(v0, v1, p) * inv_area

            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                depth = w0 * depths[0] + w1 * depths[1] + w2 * depths[2]
                color = w0 * colors[0] + w1 * colors[1] + w2 * colors[2]
                fragments.append(Fragment(x, y, depth, np.clip(color, 0, 1)))

    return fragments


# ---------------------------------------------------------------------------
# 8. Fragment processing -- depth test and framebuffer write
# ---------------------------------------------------------------------------


def fragment_processing(fragments: List[Fragment],
                        color_buf: np.ndarray,
                        depth_buf: np.ndarray) -> int:
    """Process fragments: depth test then write to framebuffer.

    Returns the number of fragments that passed the depth test.

    Why count passes?  It measures overdraw.  If you draw 100k fragments
    but only 30k pass, you have 3.3x overdraw -- a common performance
    metric in real engines.
    """
    passed = 0
    for frag in fragments:
        if frag.depth < depth_buf[frag.y, frag.x]:
            depth_buf[frag.y, frag.x] = frag.depth
            color_buf[frag.y, frag.x] = frag.color
            passed += 1
    return passed


# ---------------------------------------------------------------------------
# 9. Full pipeline orchestration
# ---------------------------------------------------------------------------


def run_pipeline(width: int = 400, height: int = 400,
                 rotation_deg: float = 35.0) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Execute the entire graphics pipeline for one frame.

    Returns (color_image, depth_image, stats) where stats captures
    per-stage diagnostic information.
    """
    stats = {}

    # --- Application stage ---
    vertices, indices = application_stage()
    stats['vertices'] = len(vertices)
    stats['triangles_submitted'] = len(indices)

    # --- Build transform matrices ---
    M = build_model_matrix(rotation_deg)
    V = build_view_matrix(eye=[3, 2, 4], target=[0, 0.2, 0], up=[0, 1, 0])
    P = build_projection_matrix(fov_deg=60, aspect=width / height, near=0.1, far=50)
    mvp = P @ V @ M

    # --- Vertex processing ---
    clip_coords = vertex_processing(vertices, mvp)

    # --- Primitive assembly ---
    assembled = primitive_assembly(vertices, clip_coords, indices)

    # --- Clipping ---
    visible = [tri for tri in assembled if clip_triangle(tri)]
    stats['triangles_after_clip'] = len(visible)

    # --- Framebuffer ---
    color_buf = np.full((height, width, 3), 0.08, dtype=float)
    depth_buf = np.full((height, width), np.inf, dtype=float)

    total_fragments = 0
    total_passed = 0

    for tri in visible:
        # Perspective divide + viewport transform
        screen_verts = []
        depths = []
        for clip in tri['clips']:
            sv, d = ndc_and_viewport(clip, width, height)
            screen_verts.append(sv)
            depths.append(d)

        # Back-face culling (screen-space winding)
        e = ((screen_verts[1][0] - screen_verts[0][0]) *
             (screen_verts[2][1] - screen_verts[0][1]) -
             (screen_verts[1][1] - screen_verts[0][1]) *
             (screen_verts[2][0] - screen_verts[0][0]))
        if e < 0:
            continue

        # Rasterization
        frags = rasterize(screen_verts, depths, tri['colors'], width, height)
        total_fragments += len(frags)

        # Fragment processing
        total_passed += fragment_processing(frags, color_buf, depth_buf)

    stats['fragments_generated'] = total_fragments
    stats['fragments_passed'] = total_passed
    stats['overdraw'] = (total_fragments / total_passed
                         if total_passed > 0 else 0)

    # Flip Y for display
    color_image = np.flipud(color_buf)
    depth_display = np.flipud(depth_buf.copy())
    depth_display[depth_display == np.inf] = np.nan

    return color_image, depth_display, stats


# ---------------------------------------------------------------------------
# 10. Visualization
# ---------------------------------------------------------------------------


def demo_pipeline():
    """Render a pyramid and show per-stage diagnostics."""
    color_img, depth_img, stats = run_pipeline(400, 400, rotation_deg=35)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(color_img, interpolation='nearest')
    axes[0].set_title("Color Buffer (Final Output)", fontsize=12)
    axes[0].axis('off')

    im = axes[1].imshow(depth_img, cmap='viridis', interpolation='nearest')
    axes[1].set_title("Depth Buffer", fontsize=12)
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Print pipeline statistics
    info = (f"Vertices: {stats['vertices']}  |  "
            f"Triangles: {stats['triangles_submitted']} submitted, "
            f"{stats['triangles_after_clip']} after clip\n"
            f"Fragments: {stats['fragments_generated']} generated, "
            f"{stats['fragments_passed']} passed depth test  |  "
            f"Overdraw: {stats['overdraw']:.2f}x")

    fig.suptitle("Graphics Pipeline Simulation", fontsize=14, fontweight='bold')
    fig.text(0.5, 0.02, info, ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_13_pipeline.png", dpi=100)
    plt.show()


def demo_pipeline_stages():
    """Visualize intermediate pipeline stages for a single triangle."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Pipeline Stage Visualization (Single Triangle)",
                 fontsize=14, fontweight='bold')

    # Stage descriptions and corresponding rotations to show progression
    stages = [
        ("1. Object Space\n(raw vertex data)", 0),
        ("2. World Space\n(model transform)", 15),
        ("3. View Space\n(camera transform)", 25),
        ("4. Clip Space\n(projection)", 35),
        ("5. Screen Space\n(viewport)", 45),
        ("6. Final Raster\n(fragments + depth)", 55),
    ]

    for ax, (label, rot) in zip(axes.flat, stages):
        color_img, _, _ = run_pipeline(200, 200, rotation_deg=rot)
        ax.imshow(color_img, interpolation='nearest')
        ax.set_title(label, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_13_stages.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Graphics Pipeline Simulation")
    print("=" * 60)

    print("\n[1/2] Full pipeline render with diagnostics...")
    demo_pipeline()

    print("\n[2/2] Pipeline stage visualization...")
    demo_pipeline_stages()

    print("\nDone!")


if __name__ == "__main__":
    main()
