"""
3D Transformations and the MVP Pipeline
========================================

Implements the full Model-View-Projection pipeline used in every 3D
graphics application.  Each stage has a clear geometric purpose:

  Model matrix  : object space  -> world space
  View matrix   : world space   -> camera (eye) space
  Projection    : eye space     -> clip space (then NDC after /w)

Understanding this pipeline is essential -- every vertex you see on
screen has been multiplied by M * V * P (read right-to-left).

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------------------------------------------------------------------------
# 1. Basic 4x4 Transform Matrices
# ---------------------------------------------------------------------------
# Why 4x4?  Same reason as 3x3 in 2D -- homogeneous coordinates let us
# express translation as a matrix multiply.  4D homogeneous = (x, y, z, 1).


def translate(tx: float, ty: float, tz: float) -> np.ndarray:
    """4x4 translation matrix."""
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0,  1],
    ], dtype=float)


def scale(sx: float, sy: float, sz: float) -> np.ndarray:
    """4x4 scaling matrix."""
    return np.array([
        [sx,  0,  0, 0],
        [ 0, sy,  0, 0],
        [ 0,  0, sz, 0],
        [ 0,  0,  0, 1],
    ], dtype=float)


def rotate_x(angle_deg: float) -> np.ndarray:
    """4x4 rotation about the X axis.

    Why separate rotate_x/y/z?  Euler rotations compose as
    Rz * Ry * Rx (or other orders) and having individual axis
    rotations is the building block.
    """
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([
        [1,  0,  0, 0],
        [0,  c, -s, 0],
        [0,  s,  c, 0],
        [0,  0,  0, 1],
    ], dtype=float)


def rotate_y(angle_deg: float) -> np.ndarray:
    """4x4 rotation about the Y axis."""
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ], dtype=float)


def rotate_z(angle_deg: float) -> np.ndarray:
    """4x4 rotation about the Z axis."""
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ], dtype=float)


def rotate_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues' rotation formula as a 4x4 matrix.

    Why Rodrigues?  Euler angles suffer from gimbal lock.  Axis-angle
    representation avoids this by specifying an arbitrary rotation axis.
    This is also the foundation for quaternion rotations.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)  # Normalize -- essential for correctness
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    x, y, z = axis

    # Rodrigues' formula expanded into matrix form
    R = np.array([
        [c + x*x*(1-c),     x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s,   c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s,   z*y*(1-c) + x*s, c + z*z*(1-c)],
    ])

    M = np.eye(4)
    M[:3, :3] = R
    return M


# ---------------------------------------------------------------------------
# 2. View Matrix (Camera)
# ---------------------------------------------------------------------------

def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Construct a LookAt view matrix.

    Why LookAt?  Specifying a camera by position + target point + up
    vector is far more intuitive than manually composing rotations.
    This function builds an orthonormal basis for camera space.

    The view matrix transforms world coordinates into the camera's
    local coordinate system where the camera sits at the origin
    looking down -Z (OpenGL convention).

    Parameters
    ----------
    eye    : Camera position in world space
    target : Point the camera looks at
    up     : World up direction (typically [0, 1, 0])

    Returns
    -------
    4x4 view matrix
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    # Forward vector: from target to eye (camera looks down -Z)
    # Why target-to-eye instead of eye-to-target?  Convention: the camera
    # looks along -Z in its local space, so the Z basis vector points
    # *away* from the scene.
    f = eye - target
    f = f / np.linalg.norm(f)

    # Right vector: perpendicular to both up and forward
    r = np.cross(up, f)
    r = r / np.linalg.norm(r)

    # Recompute true up: perpendicular to right and forward
    # Why recompute?  The input `up` may not be exactly perpendicular
    # to the forward vector -- this ensures an orthonormal basis.
    u = np.cross(f, r)

    # The view matrix = rotation * translation
    # Rotation aligns world axes to camera axes
    # Translation moves the world so the camera is at origin
    V = np.array([
        [r[0], r[1], r[2], -np.dot(r, eye)],
        [u[0], u[1], u[2], -np.dot(u, eye)],
        [f[0], f[1], f[2], -np.dot(f, eye)],
        [   0,    0,    0,               1],
    ], dtype=float)

    return V


# ---------------------------------------------------------------------------
# 3. Projection Matrices
# ---------------------------------------------------------------------------

def perspective(fov_deg: float, aspect: float,
                near: float, far: float) -> np.ndarray:
    """Perspective projection matrix (OpenGL convention, right-handed).

    Why perspective?  It mimics human vision -- distant objects appear
    smaller.  The fov controls the "zoom" and aspect prevents stretching.

    The near/far planes define the depth range.  Objects outside are
    clipped.  A common pitfall: making far/near ratio too large causes
    Z-fighting (depth buffer loses precision).

    Parameters
    ----------
    fov_deg : Vertical field of view in degrees
    aspect  : Width / height ratio
    near    : Distance to near clipping plane (positive)
    far     : Distance to far clipping plane (positive)
    """
    t = np.tan(np.radians(fov_deg) / 2)
    return np.array([
        [1 / (aspect * t),      0,                          0,  0],
        [               0, 1 / t,                           0,  0],
        [               0,      0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [               0,      0,                          -1,  0],
    ], dtype=float)


def orthographic(left: float, right: float,
                 bottom: float, top: float,
                 near: float, far: float) -> np.ndarray:
    """Orthographic projection matrix.

    Why orthographic?  No perspective foreshortening -- parallel lines
    stay parallel.  Used in CAD, architectural drawing, 2D games,
    and shadow mapping.  Also useful for understanding projection
    before tackling the more complex perspective case.
    """
    return np.array([
        [2 / (right - left), 0, 0, -(right + left) / (right - left)],
        [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
        [0, 0, -2 / (far - near),  -(far + near) / (far - near)],
        [0, 0, 0, 1],
    ], dtype=float)


# ---------------------------------------------------------------------------
# 4. Geometry: Unit Cube
# ---------------------------------------------------------------------------

def make_cube_vertices() -> np.ndarray:
    """Return 8 vertices of a unit cube centered at origin, in homogeneous coords.

    Returns shape (4, 8): each column is [x, y, z, 1].
    """
    # Why enumerate all 8 corners explicitly?  Clarity over cleverness.
    pts = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1],
    ], dtype=float).T  # shape (3, 8)

    ones = np.ones((1, 8))
    return np.vstack([pts, ones])  # shape (4, 8)


# Edges of a cube: pairs of vertex indices
CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # back face
    (4, 5), (5, 6), (6, 7), (7, 4),  # front face
    (0, 4), (1, 5), (2, 6), (3, 7),  # connecting edges
]

# Faces of a cube: quads as vertex index tuples (for filled rendering)
CUBE_FACES = [
    (0, 1, 2, 3),  # back
    (4, 5, 6, 7),  # front
    (0, 1, 5, 4),  # bottom
    (2, 3, 7, 6),  # top
    (0, 3, 7, 4),  # left
    (1, 2, 6, 5),  # right
]


def project_to_screen(clip_coords: np.ndarray) -> np.ndarray:
    """Perspective divide: clip space -> NDC.

    Why separate this step?  The GPU does this automatically after the
    vertex shader, but in our software pipeline we must do it explicitly.
    Division by w is what creates the perspective foreshortening effect.
    """
    # Divide x, y, z by w
    w = clip_coords[3]
    ndc = clip_coords[:3] / w
    return ndc


# ---------------------------------------------------------------------------
# 5. MVP Pipeline Visualization
# ---------------------------------------------------------------------------

def demo_mvp_pipeline():
    """Walk through the full Model-View-Projection pipeline step by step.

    Shows a wireframe cube at each transformation stage:
    1. Object space (the raw cube)
    2. World space (after model transform)
    3. Eye space (after view transform)
    4. Clip/NDC space (after projection + perspective divide)
    """
    cube = make_cube_vertices()

    # Model: rotate the cube and move it slightly
    M_model = translate(2, 0, 0) @ rotate_y(30) @ rotate_x(20)

    # View: camera at (5, 4, 8) looking at origin
    M_view = look_at(
        eye=np.array([5, 4, 8]),
        target=np.array([0, 0, 0]),
        up=np.array([0, 1, 0])
    )

    # Projection: perspective
    M_proj = perspective(fov_deg=60, aspect=1.0, near=0.1, far=100)

    # Compute vertices at each stage
    world_pts = M_model @ cube
    eye_pts = M_view @ world_pts
    clip_pts = M_proj @ eye_pts

    # Perspective divide to get NDC
    ndc_pts = np.zeros((3, 8))
    for i in range(8):
        ndc_pts[:, i] = project_to_screen(clip_pts[:, i])

    # --- Visualization ---
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Model-View-Projection Pipeline", fontsize=16, fontweight='bold')

    stages = [
        ("1. Object Space", cube[:3]),
        ("2. World Space (Model applied)", world_pts[:3]),
        ("3. Eye Space (View applied)", eye_pts[:3]),
    ]

    # 3D plots for first three stages
    for idx, (title, pts) in enumerate(stages):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        ax.set_title(title, fontsize=10)

        # Draw wireframe edges
        for (i, j) in CUBE_EDGES:
            ax.plot3D(
                [pts[0, i], pts[0, j]],
                [pts[1, i], pts[1, j]],
                [pts[2, i], pts[2, j]],
                'b-', linewidth=1.5
            )

        # Draw vertices
        ax.scatter(pts[0], pts[1], pts[2], c='red', s=30, zorder=5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set reasonable limits
        max_range = np.abs(pts).max() * 1.2
        if max_range < 0.1:
            max_range = 2
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)

    # 2D plot for NDC (the final projected result)
    ax_ndc = fig.add_subplot(2, 3, 4)
    ax_ndc.set_title("4. NDC (after projection + perspective divide)", fontsize=10)

    for (i, j) in CUBE_EDGES:
        ax_ndc.plot(
            [ndc_pts[0, i], ndc_pts[0, j]],
            [ndc_pts[1, i], ndc_pts[1, j]],
            'b-', linewidth=1.5
        )
    ax_ndc.scatter(ndc_pts[0], ndc_pts[1], c='red', s=30, zorder=5)
    ax_ndc.set_xlim(-1.5, 1.5)
    ax_ndc.set_ylim(-1.5, 1.5)
    ax_ndc.set_aspect('equal')
    ax_ndc.grid(True, alpha=0.3)
    ax_ndc.axhline(y=0, color='k', linewidth=0.5)
    ax_ndc.axvline(x=0, color='k', linewidth=0.5)

    # Show the matrices
    ax_info = fig.add_subplot(2, 3, 5)
    ax_info.axis('off')
    info_text = (
        "Matrices used:\n\n"
        f"Model =\n  Translate(2,0,0) @ RotY(30) @ RotX(20)\n\n"
        f"View =\n  LookAt(eye=[5,4,8], target=[0,0,0])\n\n"
        f"Projection =\n  Perspective(fov=60, aspect=1, near=0.1, far=100)\n\n"
        "Pipeline:\n"
        "  v_clip = P * V * M * v_object\n"
        "  v_ndc  = v_clip.xyz / v_clip.w"
    )
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_02_mvp_pipeline.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 6. Perspective vs Orthographic comparison
# ---------------------------------------------------------------------------

def demo_projection_comparison():
    """Compare perspective and orthographic projections side by side.

    Why compare?  Understanding the visual difference is crucial:
    - Perspective: realistic depth cues (further = smaller)
    - Orthographic: preserves parallel lines and relative sizes

    We place two cubes at different depths to make the difference obvious.
    """
    cube = make_cube_vertices()

    # Two cubes at different depths
    M1 = translate(-1.5, 0, -3) @ rotate_y(25) @ rotate_x(15)
    M2 = translate(1.5, 0, -7) @ rotate_y(-15) @ rotate_x(10)

    V = look_at(eye=np.array([0, 2, 5]),
                target=np.array([0, 0, -3]),
                up=np.array([0, 1, 0]))

    P_persp = perspective(fov_deg=60, aspect=1.0, near=0.1, far=100)
    P_ortho = orthographic(-5, 5, -5, 5, 0.1, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Perspective vs Orthographic Projection", fontsize=14, fontweight='bold')

    for ax, P, title in [(ax1, P_persp, "Perspective"),
                          (ax2, P_ortho, "Orthographic")]:
        ax.set_title(title, fontsize=12)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        for M, color in [(M1, 'blue'), (M2, 'green')]:
            world = M @ cube
            eye = V @ world
            clip = P @ eye

            ndc = np.zeros((3, 8))
            for i in range(8):
                ndc[:, i] = project_to_screen(clip[:, i])

            for (i, j) in CUBE_EDGES:
                ax.plot([ndc[0, i], ndc[0, j]],
                        [ndc[1, i], ndc[1, j]],
                        color=color, linewidth=1.5)
            ax.scatter(ndc[0], ndc[1], c=color, s=20, zorder=5)

    ax1.annotate("Near cube appears larger", xy=(0, -1.2), fontsize=9,
                 ha='center', style='italic')
    ax2.annotate("Both cubes appear same size", xy=(0, -1.2), fontsize=9,
                 ha='center', style='italic')

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_02_projection_comparison.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 7. Interactive wireframe cube with model rotation
# ---------------------------------------------------------------------------

def demo_wireframe_rotation():
    """Animate a wireframe cube rotating, showing the full MVP pipeline.

    Why animate?  Static wireframes can be ambiguous (is the cube convex
    or concave?).  Rotation resolves the ambiguity via motion parallax.
    """
    import matplotlib.animation as animation

    cube = make_cube_vertices()

    V = look_at(
        eye=np.array([4, 3, 5]),
        target=np.array([0, 0, 0]),
        up=np.array([0, 1, 0])
    )
    P = perspective(fov_deg=60, aspect=1.0, near=0.1, far=100)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title("Wireframe Cube - Rotating via MVP Pipeline", fontsize=12)
    ax.grid(True, alpha=0.2)

    lines = []
    for _ in CUBE_EDGES:
        line, = ax.plot([], [], 'b-', linewidth=1.5)
        lines.append(line)

    angle_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         fontsize=9, verticalalignment='top')

    def update(frame):
        angle = frame * 2  # degrees per frame
        M = rotate_y(angle) @ rotate_x(angle * 0.3)
        mvp = P @ V @ M

        clip = mvp @ cube
        ndc = np.zeros((3, 8))
        for i in range(8):
            ndc[:, i] = project_to_screen(clip[:, i])

        for line, (i, j) in zip(lines, CUBE_EDGES):
            line.set_data([ndc[0, i], ndc[0, j]],
                          [ndc[1, i], ndc[1, j]])

        angle_text.set_text(f"Y rotation: {angle % 360:.0f} deg")
        return lines + [angle_text]

    anim = animation.FuncAnimation(fig, update, frames=360,
                                   interval=33, blit=True)
    plt.tight_layout()
    plt.show()
    return anim


# ---------------------------------------------------------------------------
# 8. Multiple cubes in a scene (scene composition)
# ---------------------------------------------------------------------------

def demo_scene_composition():
    """Render multiple objects with different model matrices in one scene.

    Why demonstrate this?  In a real application, each object has its
    own model matrix.  The view and projection are shared.  This is the
    fundamental pattern for any 3D scene.
    """
    cube = make_cube_vertices()

    V = look_at(eye=np.array([8, 6, 10]),
                target=np.array([0, 0, 0]),
                up=np.array([0, 1, 0]))
    P = perspective(fov_deg=50, aspect=1.0, near=0.1, far=100)

    # Define several objects with different transforms
    objects = [
        ("Origin cube", np.eye(4), 'blue'),
        ("Translated", translate(3, 0, 0), 'red'),
        ("Scaled", translate(-3, 0, 0) @ scale(0.5, 2, 0.5), 'green'),
        ("Rotated", translate(0, 0, -4) @ rotate_y(45), 'purple'),
        ("Composite", translate(0, 3, 0) @ rotate_z(30) @ scale(0.7, 0.7, 0.7), 'orange'),
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_title("Scene with Multiple Objects (shared V and P)", fontsize=12)

    for name, M, color in objects:
        mvp = P @ V @ M
        clip = mvp @ cube
        ndc = np.zeros((3, 8))
        for i in range(8):
            ndc[:, i] = project_to_screen(clip[:, i])

        for (i, j) in CUBE_EDGES:
            ax.plot([ndc[0, i], ndc[0, j]],
                    [ndc[1, i], ndc[1, j]],
                    color=color, linewidth=1.2)
        ax.scatter(ndc[0], ndc[1], c=color, s=15, zorder=5, label=name)

    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_02_scene_composition.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("3D Transformations and MVP Pipeline")
    print("=" * 60)

    print("\n[1/4] Full MVP pipeline walkthrough...")
    demo_mvp_pipeline()

    print("\n[2/4] Perspective vs Orthographic...")
    demo_projection_comparison()

    print("\n[3/4] Scene with multiple objects...")
    demo_scene_composition()

    print("\n[4/4] Animated wireframe rotation (close window to exit)...")
    anim = demo_wireframe_rotation()

    print("\nDone!")


if __name__ == "__main__":
    main()
