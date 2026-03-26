"""
Vector Operations with NumPy

Demonstrates fundamental vector operations:
- Vector addition and scalar multiplication
- Dot product and cross product
- L1, L2, and Lp norms
- Vector projection
- Angle between vectors
- Visualization of vector operations

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def vector_arithmetic():
    """Demonstrate vector addition, scalar multiplication, and linear combinations."""
    print("=" * 60)
    print("VECTOR ARITHMETIC")
    print("=" * 60)

    a = np.array([3, 1])
    b = np.array([1, 4])

    # Addition
    c = a + b
    print(f"\na = {a}")
    print(f"b = {b}")
    print(f"a + b = {c}")

    # Scalar multiplication
    print(f"\n2 * a = {2 * a}")
    print(f"-0.5 * b = {-0.5 * b}")

    # Linear combination
    alpha, beta = 2, -1
    lc = alpha * a + beta * b
    print(f"\n{alpha}*a + {beta}*b = {lc}")

    # Higher dimensions
    u = np.array([1, 2, 3, 4, 5])
    v = np.array([5, 4, 3, 2, 1])
    print(f"\nIn R^5:")
    print(f"u = {u}")
    print(f"v = {v}")
    print(f"u + v = {u + v}")
    print(f"u - v = {u - v}")

    return a, b


def dot_and_cross_product():
    """Demonstrate dot product and cross product."""
    print("\n" + "=" * 60)
    print("DOT PRODUCT AND CROSS PRODUCT")
    print("=" * 60)

    # Dot product in R^2
    a = np.array([3, 1])
    b = np.array([1, 4])
    dot = np.dot(a, b)
    print(f"\na = {a}, b = {b}")
    print(f"a . b = {dot}")
    print(f"Manual: {a[0]*b[0] + a[1]*b[1]}")

    # Dot product equals a^T b
    print(f"a^T @ b = {a @ b}")

    # Angle between vectors
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    print(f"\nAngle between a and b:")
    print(f"  cos(theta) = {cos_theta:.4f}")
    print(f"  theta = {np.degrees(theta):.2f} degrees")

    # Orthogonal vectors have dot product = 0
    p = np.array([1, 0])
    q = np.array([0, 1])
    print(f"\np = {p}, q = {q}")
    print(f"p . q = {np.dot(p, q)} (orthogonal)")

    # Cross product in R^3
    print("\n--- Cross Product (R^3 only) ---")
    u = np.array([1, 0, 0])
    v = np.array([0, 1, 0])
    w = np.cross(u, v)
    print(f"\nu = {u}")
    print(f"v = {v}")
    print(f"u x v = {w}")
    print(f"u . (u x v) = {np.dot(u, w)} (perpendicular to u)")
    print(f"v . (u x v) = {np.dot(v, w)} (perpendicular to v)")

    # Cross product magnitude = area of parallelogram
    a3 = np.array([3, 0, 0])
    b3 = np.array([0, 4, 0])
    cross = np.cross(a3, b3)
    area = np.linalg.norm(cross)
    print(f"\na = {a3}, b = {b3}")
    print(f"a x b = {cross}")
    print(f"|a x b| = {area} (area of parallelogram)")

    # Anti-commutativity
    print(f"\na x b = {np.cross(a3, b3)}")
    print(f"b x a = {np.cross(b3, a3)} (anti-commutative)")


def vector_norms():
    """Demonstrate L1, L2, Lp, and infinity norms."""
    print("\n" + "=" * 60)
    print("VECTOR NORMS")
    print("=" * 60)

    v = np.array([3, -4, 5, -2])
    print(f"\nv = {v}")

    # L1 norm (Manhattan distance)
    l1 = np.linalg.norm(v, ord=1)
    print(f"\nL1 norm (sum of absolutes): {l1}")
    print(f"  Manual: {np.sum(np.abs(v))}")

    # L2 norm (Euclidean distance)
    l2 = np.linalg.norm(v, ord=2)
    print(f"\nL2 norm (Euclidean): {l2:.4f}")
    print(f"  Manual: {np.sqrt(np.sum(v**2)):.4f}")

    # L-infinity norm (max absolute value)
    linf = np.linalg.norm(v, ord=np.inf)
    print(f"\nL-inf norm (max absolute): {linf}")
    print(f"  Manual: {np.max(np.abs(v))}")

    # General Lp norm
    for p in [1, 2, 3, 5, 10]:
        lp = np.linalg.norm(v, ord=p)
        print(f"  L{p} norm: {lp:.4f}")

    # Unit vector (normalization)
    v2 = np.array([3, 4])
    unit = v2 / np.linalg.norm(v2)
    print(f"\nv = {v2}")
    print(f"Unit vector: {unit}")
    print(f"||unit|| = {np.linalg.norm(unit):.6f} (should be 1)")

    # Norm properties
    a = np.array([1, 2, 3])
    b = np.array([4, -1, 2])
    print(f"\n--- Norm Properties ---")
    print(f"||a|| >= 0: {np.linalg.norm(a) >= 0}")
    print(f"||2*a|| = 2*||a||: {np.isclose(np.linalg.norm(2*a), 2*np.linalg.norm(a))}")
    print(f"Triangle inequality: ||a+b|| <= ||a|| + ||b||:")
    print(f"  {np.linalg.norm(a+b):.4f} <= {np.linalg.norm(a) + np.linalg.norm(b):.4f}: "
          f"{np.linalg.norm(a+b) <= np.linalg.norm(a) + np.linalg.norm(b) + 1e-10}")


def vector_projection():
    """Demonstrate vector projection and decomposition."""
    print("\n" + "=" * 60)
    print("VECTOR PROJECTION")
    print("=" * 60)

    a = np.array([3, 4])
    b = np.array([5, 0])

    # Scalar projection of a onto b
    scalar_proj = np.dot(a, b) / np.linalg.norm(b)
    print(f"\na = {a}, b = {b}")
    print(f"Scalar projection of a onto b: {scalar_proj}")

    # Vector projection of a onto b
    proj = (np.dot(a, b) / np.dot(b, b)) * b
    print(f"Vector projection: {proj}")

    # Orthogonal component (rejection)
    rej = a - proj
    print(f"Rejection (orthogonal component): {rej}")

    # Verify decomposition
    print(f"\nVerification:")
    print(f"  proj + rej = {proj + rej} (should equal a)")
    print(f"  proj . rej = {np.dot(proj, rej):.10f} (should be 0)")

    # Projection in R^3
    print("\n--- Projection in R^3 ---")
    u = np.array([1, 2, 3])
    v = np.array([1, 1, 0])
    proj_3d = (np.dot(u, v) / np.dot(v, v)) * v
    rej_3d = u - proj_3d
    print(f"u = {u}, v = {v}")
    print(f"proj_v(u) = {proj_3d}")
    print(f"u - proj_v(u) = {rej_3d}")
    print(f"Orthogonal check: {np.isclose(np.dot(proj_3d, rej_3d), 0)}")

    return a, b, proj, rej


def visualize_vector_operations(a, b, proj, rej):
    """Create visualization of vector operations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Vector addition
    ax = axes[0]
    ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1,
              color='blue', label='a', width=0.02)
    ax.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1,
              color='red', label='b', width=0.02)
    ax.quiver(0, 0, a[0]+b[0], a[1]+b[1], angles='xy', scale_units='xy', scale=1,
              color='green', label='a + b', width=0.02)
    # Parallelogram
    ax.quiver(a[0], a[1], b[0], b[1], angles='xy', scale_units='xy', scale=1,
              color='red', alpha=0.3, width=0.01)
    ax.quiver(b[0], b[1], a[0], a[1], angles='xy', scale_units='xy', scale=1,
              color='blue', alpha=0.3, width=0.01)
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Vector Addition\n(Parallelogram Rule)')

    # Plot 2: Vector projection
    ax = axes[1]
    ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1,
              color='blue', label='a', width=0.02)
    ax.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1,
              color='red', label='b', width=0.02)
    ax.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1,
              color='green', label='proj_b(a)', width=0.02)
    ax.quiver(proj[0], proj[1], rej[0], rej[1], angles='xy', scale_units='xy', scale=1,
              color='orange', label='rejection', width=0.02)
    # Dashed line from a to projection
    ax.plot([a[0], proj[0]], [a[1], proj[1]], 'k--', alpha=0.3)
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Vector Projection\nproj_b(a) = (a.b / b.b) * b')

    # Plot 3: Unit circle norms
    ax = axes[2]
    theta = np.linspace(0, 2 * np.pi, 200)
    # L2 unit circle
    ax.plot(np.cos(theta), np.sin(theta), 'b-', label='L2', linewidth=2)
    # L1 unit diamond
    t = np.linspace(0, 1, 50)
    l1_x = np.concatenate([t, -t, -t, t])
    l1_y = np.concatenate([1-t, t-1+1, -(1-t), -(t-1+1)])
    l1_x = np.concatenate([t, -t[::-1], -t, t[::-1]])
    l1_y = np.concatenate([1-t, 1-t[::-1], -(1-t), -(1-t[::-1])])
    ax.plot(l1_x, l1_y, 'r-', label='L1', linewidth=2)
    # L-inf unit square
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    ax.plot(sq[:, 0], sq[:, 1], 'g-', label='L-inf', linewidth=2)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Unit Balls\n{x : ||x||_p <= 1}')

    plt.tight_layout()
    plt.savefig('vector_operations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: vector_operations.png")


def distance_metrics():
    """Demonstrate various distance metrics between vectors."""
    print("\n" + "=" * 60)
    print("DISTANCE METRICS")
    print("=" * 60)

    a = np.array([1, 2, 3])
    b = np.array([4, 0, 5])
    print(f"\na = {a}, b = {b}")

    # Euclidean distance (L2)
    d_eucl = np.linalg.norm(a - b)
    print(f"\nEuclidean distance: {d_eucl:.4f}")

    # Manhattan distance (L1)
    d_man = np.linalg.norm(a - b, ord=1)
    print(f"Manhattan distance: {d_man}")

    # Chebyshev distance (L-inf)
    d_cheb = np.linalg.norm(a - b, ord=np.inf)
    print(f"Chebyshev distance: {d_cheb}")

    # Cosine distance
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_dist = 1 - cos_sim
    print(f"\nCosine similarity: {cos_sim:.4f}")
    print(f"Cosine distance: {cos_dist:.4f}")

    # Cosine similarity is scale-invariant
    a_scaled = 100 * a
    cos_sim_scaled = np.dot(a_scaled, b) / (np.linalg.norm(a_scaled) * np.linalg.norm(b))
    print(f"Cosine similarity (100*a vs b): {cos_sim_scaled:.4f} (same as above)")


if __name__ == "__main__":
    a, b = vector_arithmetic()
    dot_and_cross_product()
    vector_norms()
    a2, b2, proj, rej = vector_projection()
    visualize_vector_operations(a2, b2, proj, rej)
    distance_metrics()
    print("\nAll examples completed!")
