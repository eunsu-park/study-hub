"""
Inverse Kinematics: Analytical and Numerical Methods
=====================================================
Given a desired end-effector pose, find joint angles that achieve it.

Inverse kinematics (IK) is the "reverse" of FK: instead of computing where
the end-effector goes for given joints, we find which joints produce a
desired pose. IK is harder than FK because:
  - There may be multiple solutions (e.g., elbow-up vs elbow-down)
  - Some poses are unreachable (outside the workspace)
  - Near singularities, the solution is numerically ill-conditioned

We implement two approaches:
  1. Analytical (geometric): Exact, fast, but only works for simple geometries
  2. Numerical (Jacobian-based): General-purpose, iterative, works for any chain
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Analytical IK for 2-link planar arm
# ---------------------------------------------------------------------------
def ik_2link_analytical(x: float, y: float, L1: float, L2: float,
                         elbow_up: bool = True) -> Optional[Tuple[float, float]]:
    """Analytical IK for a 2-link planar manipulator using the geometric method.

    The key insight is the law of cosines: for a triangle formed by the base,
    the elbow, and the target point, we know all three side lengths
    (L1, L2, and the distance to target). This gives us theta2 directly.

    Args:
        x, y: Desired end-effector position
        L1, L2: Link lengths
        elbow_up: Choose between two possible configurations

    Returns:
        (theta1, theta2) or None if unreachable

    Why two solutions? The elbow can bend "up" or "down" relative to the
    line from base to target. Both configurations reach the same point
    but with different joint angle combinations.
    """
    r_sq = x**2 + y**2
    r = np.sqrt(r_sq)

    # Check reachability: target must be within [|L1-L2|, L1+L2]
    if r > L1 + L2 or r < abs(L1 - L2):
        return None

    # Law of cosines to find theta2
    cos_theta2 = (r_sq - L1**2 - L2**2) / (2 * L1 * L2)
    # Clamp to [-1, 1] to handle numerical errors at workspace boundary
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)

    if elbow_up:
        theta2 = np.arctan2(np.sqrt(1 - cos_theta2**2), cos_theta2)
    else:
        theta2 = np.arctan2(-np.sqrt(1 - cos_theta2**2), cos_theta2)

    # Use atan2 for theta1 to handle all quadrants correctly
    # beta is the angle of the target point, phi is the angle offset due to L2
    beta = np.arctan2(y, x)
    phi = np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
    theta1 = beta - phi

    return theta1, theta2


# ---------------------------------------------------------------------------
# Forward kinematics helper for 2-link planar
# ---------------------------------------------------------------------------
def fk_2link(theta1: float, theta2: float,
             L1: float, L2: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute joint positions for visualization.

    Returns the elbow and end-effector positions.
    """
    elbow = np.array([L1 * np.cos(theta1), L1 * np.sin(theta1)])
    ee = elbow + np.array([L2 * np.cos(theta1 + theta2),
                           L2 * np.sin(theta1 + theta2)])
    return elbow, ee


# ---------------------------------------------------------------------------
# Numerical IK using Jacobian pseudo-inverse
# ---------------------------------------------------------------------------
def jacobian_2link(theta1: float, theta2: float,
                    L1: float, L2: float) -> np.ndarray:
    """Compute the 2x2 Jacobian for the 2-link planar arm.

    The Jacobian J maps joint velocities to end-effector velocities:
        [dx/dt, dy/dt]^T = J @ [dtheta1/dt, dtheta2/dt]^T

    Each column represents how much the end-effector moves when one joint
    changes by a small amount, with the other held fixed.
    """
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s12, c12 = np.sin(theta1 + theta2), np.cos(theta1 + theta2)

    return np.array([
        [-L1 * s1 - L2 * s12, -L2 * s12],
        [ L1 * c1 + L2 * c12,  L2 * c12]
    ])


def ik_jacobian_pseudoinverse(target: np.ndarray, q0: np.ndarray,
                                L1: float, L2: float,
                                max_iter: int = 100,
                                tol: float = 1e-4) -> Tuple[np.ndarray, list]:
    """Numerical IK using the Jacobian pseudo-inverse method.

    Algorithm:
        1. Compute current position via FK
        2. Compute error = target - current
        3. Compute Jacobian at current configuration
        4. Compute dq = J^+ @ error  (J^+ = pseudo-inverse)
        5. Update q = q + alpha * dq
        6. Repeat until convergence

    The pseudo-inverse J^+ = J^T (J J^T)^{-1} gives the minimum-norm
    joint velocity that achieves the desired Cartesian velocity.

    Drawback: Near singularities, J becomes rank-deficient and J^+ explodes.
    """
    q = q0.copy()
    errors = []

    for iteration in range(max_iter):
        _, ee = fk_2link(q[0], q[1], L1, L2)
        error = target - ee
        err_norm = np.linalg.norm(error)
        errors.append(err_norm)

        if err_norm < tol:
            break

        J = jacobian_2link(q[0], q[1], L1, L2)
        # Moore-Penrose pseudo-inverse: works even for non-square Jacobians
        J_pinv = np.linalg.pinv(J)
        dq = J_pinv @ error
        q += 0.5 * dq  # Step size < 1 for stability

    return q, errors


def ik_damped_least_squares(target: np.ndarray, q0: np.ndarray,
                              L1: float, L2: float,
                              damping: float = 0.1,
                              max_iter: int = 100,
                              tol: float = 1e-4) -> Tuple[np.ndarray, list]:
    """Numerical IK using the Damped Least-Squares (DLS) method.

    Also known as the Levenberg-Marquardt method for IK. It adds a
    regularization term to handle singularities gracefully:
        dq = J^T (J J^T + lambda^2 I)^{-1} error

    Why damping helps: Near singularities, J J^T has near-zero eigenvalues.
    Adding lambda^2 I ensures the matrix is always invertible. The trade-off
    is that we sacrifice accuracy for numerical stability — the solution
    won't perfectly reach the target near singularities, but it won't explode.

    The damping factor lambda controls this trade-off:
      - lambda = 0: equivalent to pseudo-inverse (may blow up)
      - lambda large: very stable but slow convergence
    """
    q = q0.copy()
    errors = []

    for iteration in range(max_iter):
        _, ee = fk_2link(q[0], q[1], L1, L2)
        error = target - ee
        err_norm = np.linalg.norm(error)
        errors.append(err_norm)

        if err_norm < tol:
            break

        J = jacobian_2link(q[0], q[1], L1, L2)
        # DLS formula: J^T (J J^T + lambda^2 I)^{-1}
        JJT = J @ J.T
        dq = J.T @ np.linalg.solve(JJT + damping**2 * np.eye(2), error)
        q += dq

    return q, errors


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def draw_arm(ax, theta1, theta2, L1, L2, color='b', label='', alpha=1.0):
    """Draw a 2-link planar arm."""
    elbow, ee = fk_2link(theta1, theta2, L1, L2)
    xs = [0, elbow[0], ee[0]]
    ys = [0, elbow[1], ee[1]]
    ax.plot(xs, ys, '-o', color=color, linewidth=3, markersize=8,
            alpha=alpha, label=label)
    ax.plot(ee[0], ee[1], '*', color=color, markersize=15, alpha=alpha)


def demo_inverse_kinematics():
    """Demonstrate analytical and numerical IK methods."""
    print("=" * 60)
    print("Inverse Kinematics Demo")
    print("=" * 60)

    L1, L2 = 1.0, 0.8

    # --- Analytical IK: elbow-up vs elbow-down ---
    target = np.array([1.2, 0.8])
    sol_up = ik_2link_analytical(target[0], target[1], L1, L2, elbow_up=True)
    sol_down = ik_2link_analytical(target[0], target[1], L1, L2, elbow_up=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    # Draw workspace boundary
    theta_ws = np.linspace(0, 2 * np.pi, 200)
    ax.plot((L1 + L2) * np.cos(theta_ws), (L1 + L2) * np.sin(theta_ws),
            '--', color='gray', alpha=0.5, label='Workspace boundary')
    ax.plot(abs(L1 - L2) * np.cos(theta_ws), abs(L1 - L2) * np.sin(theta_ws),
            '--', color='gray', alpha=0.3)

    if sol_up:
        t1, t2 = sol_up
        draw_arm(ax, t1, t2, L1, L2, color='#1f77b4',
                 label=f'Elbow-up: ({np.degrees(t1):.1f}, {np.degrees(t2):.1f}) deg')
        print(f"[Analytical] Elbow-up:   theta1={np.degrees(t1):.2f}, theta2={np.degrees(t2):.2f}")
    if sol_down:
        t1, t2 = sol_down
        draw_arm(ax, t1, t2, L1, L2, color='#ff7f0e',
                 label=f'Elbow-down: ({np.degrees(t1):.1f}, {np.degrees(t2):.1f}) deg')
        print(f"[Analytical] Elbow-down: theta1={np.degrees(t1):.2f}, theta2={np.degrees(t2):.2f}")

    ax.plot(target[0], target[1], 'r^', markersize=15, label='Target')
    ax.set_xlim([-2.2, 2.2])
    ax.set_ylim([-2.2, 2.2])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Analytical IK: Two Solutions")

    # --- Numerical IK comparison ---
    target2 = np.array([0.9, 1.0])
    q0 = np.array([0.1, 0.1])  # Initial guess far from solution

    q_pinv, errors_pinv = ik_jacobian_pseudoinverse(target2, q0, L1, L2)
    q_dls, errors_dls = ik_damped_least_squares(target2, q0, L1, L2)

    print(f"\n[Pseudo-inverse] Final q = ({np.degrees(q_pinv[0]):.2f}, {np.degrees(q_pinv[1]):.2f}) deg")
    print(f"[DLS]            Final q = ({np.degrees(q_dls[0]):.2f}, {np.degrees(q_dls[1]):.2f}) deg")

    ax2 = axes[1]
    draw_arm(ax2, q_pinv[0], q_pinv[1], L1, L2, color='#2ca02c', label='Pseudo-inverse')
    draw_arm(ax2, q_dls[0], q_dls[1], L1, L2, color='#9467bd', label='DLS', alpha=0.7)
    ax2.plot(target2[0], target2[1], 'r^', markersize=15, label='Target')
    ax2.set_xlim([-2.2, 2.2])
    ax2.set_ylim([-2.2, 2.2])
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_title("Numerical IK Solutions")

    # --- Convergence comparison ---
    ax3 = axes[2]
    ax3.semilogy(errors_pinv, '-o', markersize=3, label='Pseudo-inverse')
    ax3.semilogy(errors_dls, '-s', markersize=3, label='DLS (λ=0.1)')
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Position Error (m)")
    ax3.set_title("Convergence Comparison")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("03_inverse_kinematics.png", dpi=120)
    plt.show()

    # --- Workspace analysis: reachable vs unreachable targets ---
    print("\n--- Workspace Reachability ---")
    test_targets = [
        np.array([0.5, 0.5]),    # Reachable
        np.array([1.5, 0.5]),    # Reachable
        np.array([2.0, 0.0]),    # Boundary
        np.array([2.5, 0.0]),    # Unreachable
    ]
    for t in test_targets:
        sol = ik_2link_analytical(t[0], t[1], L1, L2)
        status = "REACHABLE" if sol else "UNREACHABLE"
        dist = np.linalg.norm(t)
        print(f"  Target ({t[0]:.1f}, {t[1]:.1f}), dist={dist:.2f}: {status}")


if __name__ == "__main__":
    demo_inverse_kinematics()
