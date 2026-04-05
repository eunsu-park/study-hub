"""
2D Transformations in Computer Graphics
========================================

Demonstrates fundamental 2D transformation matrices using homogeneous
coordinates. Every 2D affine transformation (translation, rotation,
scaling, shear) is expressed as a 3x3 matrix so that composition is
just matrix multiplication -- this unification is *the* reason we use
homogeneous coordinates in graphics.

Key insight: transformation order matters because matrix multiplication
is NOT commutative.  Rotate-then-translate != Translate-then-rotate.

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation

# ---------------------------------------------------------------------------
# 1. Transformation matrix constructors
# ---------------------------------------------------------------------------
# Why homogeneous coordinates?  In Cartesian 2D, translation cannot be
# expressed as a matrix multiply -- it's an addition.  By lifting points
# to (x, y, 1), translation becomes a 3x3 multiply, letting us compose
# *all* affine transforms with a single matrix product.


def translation_matrix(tx: float, ty: float) -> np.ndarray:
    """Return a 3x3 homogeneous translation matrix."""
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0,  1],
    ], dtype=float)


def rotation_matrix(angle_deg: float) -> np.ndarray:
    """Return a 3x3 homogeneous rotation matrix (CCW, about origin).

    Why radians internally?  NumPy trig functions expect radians, but
    degrees are more intuitive for the API -- convert at the boundary.
    """
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ], dtype=float)


def scaling_matrix(sx: float, sy: float) -> np.ndarray:
    """Return a 3x3 homogeneous scaling matrix (about origin).

    Why separate sx, sy?  Non-uniform scaling is essential for squash &
    stretch effects in animation and for fitting content to viewports.
    """
    return np.array([
        [sx,  0, 0],
        [ 0, sy, 0],
        [ 0,  0, 1],
    ], dtype=float)


def shear_matrix(shx: float, shy: float) -> np.ndarray:
    """Return a 3x3 homogeneous shear matrix.

    Shear slides coordinates proportionally to the other axis.
    shx: how much x shifts per unit y
    shy: how much y shifts per unit x

    Why include shear?  Any 2D affine transform decomposes into
    rotation, scale, and shear -- so shear completes the toolkit.
    """
    return np.array([
        [  1, shx, 0],
        [shy,   1, 0],
        [  0,   0, 1],
    ], dtype=float)


def rotation_about_point(angle_deg: float, cx: float, cy: float) -> np.ndarray:
    """Rotate about an arbitrary point (cx, cy).

    Why three multiplications?  Rotation matrices only rotate about the
    origin.  To rotate about (cx, cy), we:
      1. Translate (cx, cy) to origin
      2. Rotate
      3. Translate back
    This is the standard "move-transform-move back" pattern used
    throughout graphics for transforms about arbitrary pivot points.
    """
    T_to_origin = translation_matrix(-cx, -cy)
    R = rotation_matrix(angle_deg)
    T_back = translation_matrix(cx, cy)
    # Read right-to-left: first T_to_origin, then R, then T_back
    return T_back @ R @ T_to_origin


# ---------------------------------------------------------------------------
# 2. Geometry helpers
# ---------------------------------------------------------------------------

def make_house() -> np.ndarray:
    """Return vertices of a simple house shape in homogeneous coords.

    Why a house?  It's asymmetric, so rotations/reflections are visually
    obvious -- a circle or square would hide some transforms.

    Returns shape (3, N) where each column is [x, y, 1].
    """
    # House body + roof, as a closed polygon
    pts = np.array([
        # Body (rectangle)
        [0, 0],
        [4, 0],
        [4, 3],
        # Roof (triangle)
        [4.5, 3],
        [2, 5],
        [-0.5, 3],
        # Back to body
        [0, 3],
        [0, 0],  # close
    ], dtype=float).T  # shape (2, N)

    # Lift to homogeneous: append row of ones
    ones = np.ones((1, pts.shape[1]))
    return np.vstack([pts, ones])


def make_arrow() -> np.ndarray:
    """Return vertices of an arrow pointing right, in homogeneous coords.

    Useful for showing direction changes under transformation.
    """
    pts = np.array([
        [0, 0.3],
        [2, 0.3],
        [2, 0.7],
        [3, 0],
        [2, -0.7],
        [2, -0.3],
        [0, -0.3],
        [0, 0.3],
    ], dtype=float).T
    ones = np.ones((1, pts.shape[1]))
    return np.vstack([pts, ones])


def transform_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 3x3 homogeneous transform M to points (3, N).

    Why not just M @ pts?  This function is trivial, but naming it
    clarifies intent and centralizes any future normalization logic
    (e.g., for projective transforms where w != 1).
    """
    return M @ pts


# ---------------------------------------------------------------------------
# 3. Visualization helpers
# ---------------------------------------------------------------------------

def plot_shape(ax, pts: np.ndarray, color='blue', label=None,
               alpha=0.3, linewidth=2, linestyle='-'):
    """Plot a 2D shape from homogeneous coordinates.

    Why extract x, y from rows 0, 1?  Row 2 is always 1 for affine
    transforms -- we skip it for plotting.
    """
    x, y = pts[0], pts[1]
    ax.fill(x, y, alpha=alpha, color=color, label=label)
    ax.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle)


def setup_axes(ax, title='', xlim=(-8, 12), ylim=(-8, 12)):
    """Configure axes with grid and equal aspect ratio.

    Why equal aspect?  Without it, circles look like ellipses and
    rotations appear to change shape -- misleading for geometry demos.
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    if title:
        ax.set_title(title, fontsize=11)


# ---------------------------------------------------------------------------
# 4. Demo: Individual transforms
# ---------------------------------------------------------------------------

def demo_individual_transforms():
    """Show each basic 2D transform applied to a house shape.

    Why side-by-side?  Comparing original vs. transformed makes it
    immediately clear what each matrix does geometrically.
    """
    house = make_house()

    transforms = [
        ("Translation (3, 2)", translation_matrix(3, 2)),
        ("Rotation 45deg", rotation_matrix(45)),
        ("Scale (1.5, 0.7)", scaling_matrix(1.5, 0.7)),
        ("Shear (0.5, 0)", shear_matrix(0.5, 0)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Basic 2D Transformations (Homogeneous Coordinates)",
                 fontsize=14, fontweight='bold')

    for ax, (title, M) in zip(axes.flat, transforms):
        setup_axes(ax, title)
        plot_shape(ax, house, color='gray', label='Original', alpha=0.2)
        transformed = transform_points(M, house)
        plot_shape(ax, transformed, color='blue', label='Transformed')

        # Annotate the matrix
        mat_str = np.array2string(M, precision=2, suppress_small=True)
        ax.text(0.02, 0.98, f"M =\n{mat_str}",
                transform=ax.transAxes, fontsize=7,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_01_individual_transforms.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 5. Demo: Composition order matters
# ---------------------------------------------------------------------------

def demo_composition_order():
    """Demonstrate that T * R != R * T.

    This is the single most important lesson about transform composition:
    matrix multiplication is not commutative.  In practice, getting the
    order wrong is the #1 source of "my object is in the wrong place" bugs.
    """
    house = make_house()

    T = translation_matrix(5, 0)
    R = rotation_matrix(45)

    # Order 1: Rotate first, then translate
    # The object rotates about the origin, THEN moves right
    M1 = T @ R  # Read right-to-left: R first, then T

    # Order 2: Translate first, then rotate
    # The object moves right, THEN rotates about the origin
    # (which swings it in a wide arc)
    M2 = R @ T  # Read right-to-left: T first, then R

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Composition Order Matters: T*R vs R*T",
                 fontsize=14, fontweight='bold')

    for ax, M, title in [(ax1, M1, "T @ R  (Rotate, then Translate)"),
                          (ax2, M2, "R @ T  (Translate, then Rotate)")]:
        setup_axes(ax, title, xlim=(-8, 12), ylim=(-8, 12))
        plot_shape(ax, house, color='gray', label='Original', alpha=0.15)

        # Show intermediate step
        if title.startswith("T"):
            intermediate = transform_points(R, house)
            plot_shape(ax, intermediate, color='green',
                       label='After rotation', alpha=0.2, linestyle='--')
        else:
            intermediate = transform_points(T, house)
            plot_shape(ax, intermediate, color='green',
                       label='After translation', alpha=0.2, linestyle='--')

        final = transform_points(M, house)
        plot_shape(ax, final, color='red', label='Final result')
        ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_01_composition_order.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 6. Demo: Rotation about an arbitrary point
# ---------------------------------------------------------------------------

def demo_rotation_about_point():
    """Rotate a house about its own center, not the origin.

    Why is this important?  In real applications, objects rarely sit at
    the origin.  The three-step pattern (translate to origin, transform,
    translate back) is used constantly in game engines and DCC tools.
    """
    house = make_house()
    # Compute centroid of the house vertices (excluding the closing point)
    cx = np.mean(house[0, :-1])
    cy = np.mean(house[1, :-1])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Rotation About an Arbitrary Point",
                 fontsize=14, fontweight='bold')

    angles = [0, 30, 90]
    for ax, angle in zip(axes, angles):
        M = rotation_about_point(angle, cx, cy)
        setup_axes(ax, f"Rotate {angle} deg about ({cx:.1f}, {cy:.1f})",
                   xlim=(-4, 10), ylim=(-4, 10))
        plot_shape(ax, house, color='gray', alpha=0.15, label='Original')
        transformed = transform_points(M, house)
        plot_shape(ax, transformed, color='purple', label=f'{angle} deg')
        ax.plot(cx, cy, 'ro', markersize=8, label='Pivot')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_01_rotation_about_point.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 7. Demo: Animated transformation sequence
# ---------------------------------------------------------------------------

def demo_animated_sequence():
    """Animate a shape through a sequence of transforms.

    Why animate?  Static plots show the result but not the *process*.
    Seeing smooth interpolation between transforms builds intuition
    about what each matrix "does" to space.

    We interpolate between identity and each target transform using
    linear interpolation of the matrix entries.  (Note: this is only
    correct for small rotations and scales; for production code, you'd
    decompose into TRS and interpolate each channel separately.)
    """
    arrow = make_arrow()

    # Define a sequence of cumulative transforms
    transforms = [
        ("Scale 2x", scaling_matrix(2, 2)),
        ("Rotate 45 deg", rotation_matrix(45)),
        ("Translate (3, 2)", translation_matrix(3, 2)),
        ("Shear (0.4, 0)", shear_matrix(0.4, 0)),
    ]

    # Build cumulative matrices: each step compounds on the previous
    cumulative = [np.eye(3)]
    for name, M in transforms:
        cumulative.append(M @ cumulative[-1])

    # Animation parameters
    frames_per_step = 30
    total_frames = frames_per_step * len(transforms) + 10  # +10 pause at end

    fig, ax = plt.subplots(figsize=(10, 8))
    setup_axes(ax, "Animated Transform Sequence", xlim=(-6, 14), ylim=(-8, 12))

    # Pre-draw all ghost shapes for the trajectory
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightsalmon']
    for i in range(len(cumulative)):
        pts = transform_points(cumulative[i], arrow)
        ax.fill(pts[0], pts[1], alpha=0.1, color='gray')
        ax.plot(pts[0], pts[1], color='gray', linewidth=0.5, alpha=0.3)

    fill_patch = ax.fill(arrow[0], arrow[1], alpha=0.5, color='blue')[0]
    line, = ax.plot(arrow[0], arrow[1], color='darkblue', linewidth=2)
    step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def update(frame):
        """Compute interpolated transform for the current frame.

        Why linear matrix interpolation?  For this demo, transforms are
        small enough that lerping matrix entries gives visually smooth
        results.  In production, you'd use slerp for rotation and
        separate lerp for translation/scale.
        """
        step = min(frame // frames_per_step, len(transforms) - 1)
        local_t = (frame % frames_per_step) / frames_per_step

        if frame >= frames_per_step * len(transforms):
            # Pause at the end
            M = cumulative[-1]
            step_text.set_text("Done! All transforms applied.")
        else:
            # Interpolate between cumulative[step] and cumulative[step+1]
            M = (1 - local_t) * cumulative[step] + local_t * cumulative[step + 1]
            name = transforms[step][0]
            step_text.set_text(f"Step {step + 1}: {name}  (t={local_t:.2f})")

        pts = transform_points(M, arrow)
        fill_patch.set_xy(np.column_stack([pts[0], pts[1]]))
        line.set_data(pts[0], pts[1])
        return fill_patch, line, step_text

    anim = animation.FuncAnimation(fig, update, frames=total_frames,
                                   interval=33, blit=False)
    plt.tight_layout()
    plt.show()
    return anim  # Keep reference to prevent garbage collection


# ---------------------------------------------------------------------------
# 8. Demo: Transform decomposition visualization
# ---------------------------------------------------------------------------

def demo_decomposition():
    """Show how a single composite matrix decomposes into TRS components.

    Why decompose?  Game engines store transforms as separate Translation,
    Rotation, Scale (TRS) because:
    - Interpolation is more meaningful per component
    - Artists can edit each channel independently
    - Avoids accumulated floating-point drift

    We compose T * R * S, then show each component's contribution.
    """
    T = translation_matrix(4, 2)
    R = rotation_matrix(30)
    S = scaling_matrix(1.5, 0.8)

    house = make_house()

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Transform Decomposition: T * R * S",
                 fontsize=14, fontweight='bold')

    steps = [
        ("1. Original", np.eye(3)),
        ("2. After Scale", S),
        ("3. After Rotation", R @ S),
        ("4. After Translation (Final)", T @ R @ S),
    ]

    for ax, (title, M) in zip(axes, steps):
        setup_axes(ax, title, xlim=(-4, 12), ylim=(-4, 10))
        plot_shape(ax, house, color='gray', alpha=0.1)
        pts = transform_points(M, house)
        plot_shape(ax, pts, color='teal')

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_01_decomposition.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("2D Transformations in Computer Graphics")
    print("=" * 60)

    print("\n[1/5] Individual transforms...")
    demo_individual_transforms()

    print("\n[2/5] Composition order matters...")
    demo_composition_order()

    print("\n[3/5] Rotation about arbitrary point...")
    demo_rotation_about_point()

    print("\n[4/5] Transform decomposition...")
    demo_decomposition()

    print("\n[5/5] Animated transform sequence (close window to exit)...")
    anim = demo_animated_sequence()

    print("\nDone!")


if __name__ == "__main__":
    main()
