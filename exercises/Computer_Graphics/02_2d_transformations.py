"""
Exercises for Lesson 02: 2D Transformations
Topic: Computer_Graphics
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


# ===================================================================
# Helper functions (from the lesson)
# ===================================================================

def translate(tx, ty):
    """Create a 2D translation matrix in homogeneous coordinates."""
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=float)


def rotate(theta_deg):
    """Create a 2D rotation matrix (CCW about origin) in homogeneous coordinates."""
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def scale(sx, sy=None):
    """Create a 2D scaling matrix in homogeneous coordinates."""
    if sy is None:
        sy = sx
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=float)


def shear_x(k):
    """Shear along x-axis: x' = x + k*y."""
    return np.array([[1, k, 0], [0, 1, 0], [0, 0, 1]], dtype=float)


def shear_y(k):
    """Shear along y-axis: y' = y + k*x."""
    return np.array([[1, 0, 0], [k, 1, 0], [0, 0, 1]], dtype=float)


def transform_points(matrix, points):
    """Apply a 3x3 transformation matrix to an array of 2D points."""
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    homogeneous = np.hstack([points, np.ones((n, 1))])
    transformed = (matrix @ homogeneous.T).T
    w = transformed[:, 2:3]
    return transformed[:, :2] / w


def compose(*transforms):
    """Compose transformations (applied right to left)."""
    result = np.eye(3)
    for t in transforms:
        result = result @ t
    return result


def plot_shape(ax, points, color='blue', alpha=0.3, label=None):
    """Plot a 2D polygon."""
    polygon = Polygon(points, closed=True)
    p = PatchCollection([polygon], alpha=alpha, facecolors=[color],
                        edgecolors=[color], linewidths=2)
    ax.add_collection(p)
    if label:
        centroid = points.mean(axis=0)
        ax.annotate(label, centroid, fontsize=9, ha='center',
                    fontweight='bold', color=color)


def exercise_1():
    """
    Matrix Construction: Write the single 3x3 matrix that:
    (a) scales by factor 2,
    (b) rotates 30 degrees counterclockwise,
    (c) translates by (3, -1),
    applied in that order. Verify by applying it to the point (1, 0).
    """
    # Order: scale first, then rotate, then translate
    # In matrix form (right to left): M = T * R * S
    S = scale(2, 2)
    R = rotate(30)
    T = translate(3, -1)

    M = compose(T, R, S)

    print("Individual matrices:")
    print(f"Scale(2):\n{S}\n")
    print(f"Rotate(30):\n{np.round(R, 4)}\n")
    print(f"Translate(3, -1):\n{T}\n")
    print(f"Composed M = T @ R @ S:\n{np.round(M, 4)}\n")

    # Verify with point (1, 0)
    point = np.array([[1, 0]])
    result = transform_points(M, point)[0]

    # Step-by-step verification
    after_scale = transform_points(S, point)[0]
    after_rotate = transform_points(R, np.array([after_scale]))[0]
    after_translate = transform_points(T, np.array([after_rotate]))[0]

    print("Verification with point (1, 0):")
    print(f"  After scale(2):         ({after_scale[0]:.4f}, {after_scale[1]:.4f})")
    print(f"  After rotate(30):       ({after_rotate[0]:.4f}, {after_rotate[1]:.4f})")
    print(f"  After translate(3, -1): ({after_translate[0]:.4f}, {after_translate[1]:.4f})")
    print(f"  Using composed matrix:  ({result[0]:.4f}, {result[1]:.4f})")
    print(f"  Results match: {np.allclose(result, after_translate)}")


def exercise_2():
    """
    Order Investigation: For a unit square, compute the result of:
    (a) rotate 45 degrees then scale by 2, and
    (b) scale by 2 then rotate 45 degrees.
    Plot both results and explain the visual difference.
    """
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    # (a) Rotate first, then scale: M = S * R
    M_a = compose(scale(2), rotate(45))
    result_a = transform_points(M_a, square)

    # (b) Scale first, then rotate: M = R * S
    M_b = compose(rotate(45), scale(2))
    result_b = transform_points(M_b, square)

    print("Original square corners:", square.tolist())
    print()
    print("(a) Rotate 45 then Scale 2 (M = S * R):")
    print(f"    Result corners: {np.round(result_a, 4).tolist()}")
    print()
    print("(b) Scale 2 then Rotate 45 (M = R * S):")
    print(f"    Result corners: {np.round(result_b, 4).tolist()}")
    print()
    print(f"Same result? {np.allclose(result_a, result_b)}")
    print()

    # In this specific case, rotation and uniform scaling commute!
    print("Explanation:")
    print("  For UNIFORM scaling (same factor on both axes), rotation and")
    print("  scaling commute: S * R == R * S. Both produce the same result.")
    print("  This is because uniform scaling preserves all angles.")
    print()

    # Now demonstrate with NON-UNIFORM scaling
    S_nonuniform = scale(2, 0.5)
    M_c = compose(S_nonuniform, rotate(45))
    M_d = compose(rotate(45), S_nonuniform)
    result_c = transform_points(M_c, square)
    result_d = transform_points(M_d, square)

    print("With NON-UNIFORM scaling (2, 0.5):")
    print(f"  (a) Rotate then Scale: {np.round(result_c, 4).tolist()}")
    print(f"  (b) Scale then Rotate: {np.round(result_d, 4).tolist()}")
    print(f"  Same result? {np.allclose(result_c, result_d)}")
    print()
    print("  With non-uniform scaling, order DOES matter:")
    print("  - Rotate-then-Scale: rotates the square, then stretches it along axes")
    print("    (produces a skewed diamond shape)")
    print("  - Scale-then-Rotate: stretches first (rectangle), then rotates as a unit")
    print("    (produces a rotated rectangle)")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax in axes:
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

    axes[0].set_title("Rotate 45 then Scale(2, 0.5)")
    plot_shape(axes[0], square, 'blue', 0.2, 'Original')
    plot_shape(axes[0], result_c, 'red', 0.4, 'Result')

    axes[1].set_title("Scale(2, 0.5) then Rotate 45")
    plot_shape(axes[1], square, 'blue', 0.2, 'Original')
    plot_shape(axes[1], result_d, 'green', 0.4, 'Result')

    plt.tight_layout()
    plt.savefig('ex02_order_investigation.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to ex02_order_investigation.png")


def exercise_3():
    """
    Arbitrary Rotation: Derive the composite matrix for rotating 60 degrees
    about the point (3, 4). Verify that the point (3, 4) maps to itself.
    """
    cx, cy = 3.0, 4.0
    theta = 60.0

    # Composite: translate center to origin, rotate, translate back
    T_to_origin = translate(-cx, -cy)
    R = rotate(theta)
    T_back = translate(cx, cy)

    M = compose(T_back, R, T_to_origin)

    print(f"Rotation of {theta} degrees about point ({cx}, {cy})")
    print()
    print("Composite matrix M = T(cx,cy) @ R(60) @ T(-cx,-cy):")
    print(np.round(M, 4))
    print()

    # Verify: (3, 4) should map to itself
    center = np.array([[cx, cy]])
    result = transform_points(M, center)[0]
    print(f"Point ({cx}, {cy}) maps to ({result[0]:.6f}, {result[1]:.6f})")
    print(f"Maps to itself: {np.allclose(result, [cx, cy])}")
    print()

    # Test with another point
    test_point = np.array([[4, 4]])
    test_result = transform_points(M, test_point)[0]
    distance_before = np.linalg.norm(test_point[0] - center[0])
    distance_after = np.linalg.norm(test_result - center[0])
    print(f"Test point (4, 4) maps to ({test_result[0]:.4f}, {test_result[1]:.4f})")
    print(f"Distance from center before: {distance_before:.4f}")
    print(f"Distance from center after:  {distance_after:.4f}")
    print(f"Distance preserved: {np.isclose(distance_before, distance_after)}")


def exercise_4():
    """
    Inverse Transform: An object has been transformed by
    M = T(2,3) * R(45) * S(2,1). Write the inverse matrix M^{-1} that would
    undo this transformation. Verify that M^{-1} * M = I.
    """
    T_mat = translate(2, 3)
    R_mat = rotate(45)
    S_mat = scale(2, 1)

    M = compose(T_mat, R_mat, S_mat)

    # Analytical inverse: reverse order, inverse of each
    # (A*B*C)^{-1} = C^{-1} * B^{-1} * A^{-1}
    S_inv = scale(1 / 2, 1 / 1)       # Inverse of scale
    R_inv = rotate(-45)                 # Inverse of rotation
    T_inv = translate(-2, -3)           # Inverse of translation

    M_inv_analytical = compose(S_inv, R_inv, T_inv)

    # Numerical inverse for comparison
    M_inv_numerical = np.linalg.inv(M)

    print("Forward transform M = T(2,3) @ R(45) @ S(2,1):")
    print(np.round(M, 4))
    print()

    print("Analytical inverse M^{-1} = S^{-1} @ R^{-1} @ T^{-1}:")
    print(np.round(M_inv_analytical, 4))
    print()

    print("Numerical inverse (np.linalg.inv):")
    print(np.round(M_inv_numerical, 4))
    print()

    print(f"Analytical matches numerical: {np.allclose(M_inv_analytical, M_inv_numerical)}")
    print()

    # Verify M^{-1} * M = I
    product = M_inv_analytical @ M
    print("M^{-1} @ M:")
    print(np.round(product, 6))
    print(f"Is identity: {np.allclose(product, np.eye(3))}")
    print()

    # Verify with a point
    original_point = np.array([[5, 7]])
    transformed = transform_points(M, original_point)
    recovered = transform_points(M_inv_analytical, transformed)
    print(f"Original point: {original_point[0]}")
    print(f"After M:        {np.round(transformed[0], 4)}")
    print(f"After M^{{-1}}:   {np.round(recovered[0], 4)}")
    print(f"Recovered:      {np.allclose(original_point, recovered)}")


def exercise_5():
    """
    Shear Decomposition: Show that a rotation by angle theta can be decomposed
    into three shears. Implement this and verify that the result matches the
    standard rotation matrix.
    """
    theta_deg = 30.0
    theta = np.radians(theta_deg)

    # A 2D rotation can be decomposed as three shears:
    # R(theta) = Shx(-tan(theta/2)) * Shy(sin(theta)) * Shx(-tan(theta/2))
    #
    # Proof: The product of these three shear matrices equals the rotation matrix.
    # This decomposition was used in early graphics hardware because shears
    # are simpler to implement than rotations (only additions, no multiplications
    # of coordinates by transcendental values at runtime once the matrix is set).

    half_tan = -np.tan(theta / 2)
    sin_val = np.sin(theta)

    Sh1 = shear_x(half_tan)
    Sh2 = shear_y(sin_val)
    Sh3 = shear_x(half_tan)

    # Compose: apply Sh3 first, then Sh2, then Sh1
    M_shear = compose(Sh1, Sh2, Sh3)

    # Standard rotation matrix
    R = rotate(theta_deg)

    print(f"Rotation decomposition into 3 shears (theta = {theta_deg} degrees)")
    print()
    print(f"Step 1: Shear X by -tan(theta/2) = {half_tan:.6f}")
    print(f"  {np.round(Sh1, 6).tolist()}")
    print()
    print(f"Step 2: Shear Y by sin(theta) = {sin_val:.6f}")
    print(f"  {np.round(Sh2, 6).tolist()}")
    print()
    print(f"Step 3: Shear X by -tan(theta/2) = {half_tan:.6f}")
    print(f"  {np.round(Sh3, 6).tolist()}")
    print()

    print("Composed shear matrix:")
    print(np.round(M_shear, 6))
    print()
    print("Standard rotation matrix:")
    print(np.round(R, 6))
    print()
    print(f"Matrices match: {np.allclose(M_shear, R)}")
    print()

    # Verify with a test point
    test = np.array([[3.0, 2.0]])
    result_shear = transform_points(M_shear, test)[0]
    result_rotate = transform_points(R, test)[0]
    print(f"Test point (3, 2):")
    print(f"  Via shear decomposition: ({result_shear[0]:.6f}, {result_shear[1]:.6f})")
    print(f"  Via rotation matrix:     ({result_rotate[0]:.6f}, {result_rotate[1]:.6f})")
    print(f"  Match: {np.allclose(result_shear, result_rotate)}")

    # Verify for multiple angles
    print("\nVerification across multiple angles:")
    for deg in [0, 15, 45, 60, 90, 120, 179]:
        rad = np.radians(deg)
        if abs(np.cos(rad / 2)) < 1e-10:
            print(f"  {deg:>4} deg: skipped (tan(theta/2) undefined)")
            continue
        ht = -np.tan(rad / 2)
        sv = np.sin(rad)
        M = shear_x(ht) @ shear_y(sv) @ shear_x(ht)
        R_ref = rotate(deg)
        match = np.allclose(M, R_ref)
        print(f"  {deg:>4} deg: match = {match}")


def exercise_6():
    """
    Projective Transform: A projective transformation maps the unit square
    {(0,0), (1,0), (1,1), (0,1)} to the quadrilateral {(0,0), (2,0), (1.5,1), (0.5,1)}.
    Set up the system of equations to find the 3x3 projective matrix.
    """
    # Source points (unit square)
    src = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ], dtype=float)

    # Destination points (quadrilateral)
    dst = np.array([
        [0, 0],
        [2, 0],
        [1.5, 1],
        [0.5, 1]
    ], dtype=float)

    # The projective matrix is:
    # | a  b  c |   | x |   | x' * w |
    # | d  e  f | * | y | = | y' * w |
    # | g  h  1 |   | 1 |   |   w    |
    #
    # After perspective division: (x'/w, y'/w) = destination point
    #
    # For each point (x, y) -> (x', y'):
    #   a*x + b*y + c = x' * (g*x + h*y + 1)
    #   d*x + e*y + f = y' * (g*x + h*y + 1)
    #
    # Rearranging:
    #   a*x + b*y + c - g*x*x' - h*y*x' = x'
    #   d*x + e*y + f - g*x*y' - h*y*y' = y'
    #
    # 8 equations for 8 unknowns (a, b, c, d, e, f, g, h)

    print("Setting up the system of equations for the projective transform")
    print()
    print("Source (unit square) -> Destination (quadrilateral):")
    for i in range(4):
        print(f"  ({src[i,0]:.0f}, {src[i,1]:.0f}) -> ({dst[i,0]:.1f}, {dst[i,1]:.1f})")
    print()

    # Build the 8x8 system: A @ [a,b,c,d,e,f,g,h]^T = b_vec
    A = np.zeros((8, 8))
    b_vec = np.zeros(8)

    for i in range(4):
        x, y = src[i]
        xp, yp = dst[i]

        # Equation 1: a*x + b*y + c - g*x*x' - h*y*x' = x'
        A[2 * i] = [x, y, 1, 0, 0, 0, -x * xp, -y * xp]
        b_vec[2 * i] = xp

        # Equation 2: d*x + e*y + f - g*x*y' - h*y*y' = y'
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -x * yp, -y * yp]
        b_vec[2 * i + 1] = yp

    print("System of equations (A @ x = b):")
    print("A =")
    for row in A:
        print(f"  [{', '.join(f'{v:7.1f}' for v in row)}]")
    print(f"b = {b_vec}")
    print()

    # Solve the system
    params = np.linalg.solve(A, b_vec)
    a, b, c, d, e, f, g, h = params

    # Construct the 3x3 projective matrix
    M_proj = np.array([
        [a, b, c],
        [d, e, f],
        [g, h, 1]
    ])

    print("Solution:")
    print(f"  a={a:.6f}, b={b:.6f}, c={c:.6f}")
    print(f"  d={d:.6f}, e={e:.6f}, f={f:.6f}")
    print(f"  g={g:.6f}, h={h:.6f}")
    print()
    print("Projective matrix:")
    print(np.round(M_proj, 6))
    print()

    # Verify: transform source points and check they match destination
    print("Verification:")
    for i in range(4):
        p = np.array([src[i, 0], src[i, 1], 1.0])
        result = M_proj @ p
        result_2d = result[:2] / result[2]  # Perspective division
        expected = dst[i]
        match = np.allclose(result_2d, expected)
        print(f"  ({src[i,0]:.0f}, {src[i,1]:.0f}) -> "
              f"({result_2d[0]:.4f}, {result_2d[1]:.4f}) "
              f"expected ({expected[0]:.1f}, {expected[1]:.1f}) "
              f"{'OK' if match else 'FAIL'}")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Matrix Construction ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Order Investigation ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Arbitrary Rotation ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Inverse Transform ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Shear Decomposition ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Projective Transform ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
