"""
Jacobian Computation and Singularity Analysis
==============================================
The Jacobian maps joint velocities to end-effector velocities.

The Jacobian is arguably the most important matrix in robot kinematics.
It connects the joint space to the task space (Cartesian space):

    ẋ = J(q) @ q̇       (velocity kinematics)
    F = J(q)^T @ τ       (force/torque duality)

Key properties:
  - When J loses rank (singularity), the robot cannot move in certain directions
  - The manipulability ellipsoid shows how "well-conditioned" the motion is
  - Condition number of J indicates how close we are to a singularity
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import Tuple


# ---------------------------------------------------------------------------
# Robot model: 2-link planar arm
# ---------------------------------------------------------------------------
class TwoLinkArm:
    """A 2-link planar manipulator for studying Jacobian properties.

    Why start with a 2-link arm? It has a 2x2 Jacobian, making it easy to
    visualize as an ellipse in 2D. The concepts generalize to higher DOF
    but are much harder to visualize.
    """

    def __init__(self, L1: float = 1.0, L2: float = 0.8):
        self.L1 = L1
        self.L2 = L2

    def fk(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward kinematics: joint angles -> positions.

        Returns positions of elbow and end-effector.
        """
        t1, t2 = q
        elbow = np.array([self.L1 * np.cos(t1),
                          self.L1 * np.sin(t1)])
        ee = elbow + np.array([self.L2 * np.cos(t1 + t2),
                               self.L2 * np.sin(t1 + t2)])
        return elbow, ee

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute the 2x2 geometric Jacobian.

        For a planar arm, the Jacobian relates [dθ1, dθ2] to [dx, dy]:
            J = [∂x/∂θ1  ∂x/∂θ2]
                [∂y/∂θ1  ∂y/∂θ2]

        Column 1: how the EE moves when θ1 changes (with θ2 fixed)
        Column 2: how the EE moves when θ2 changes (with θ1 fixed)
        """
        t1, t2 = q
        s1, c1 = np.sin(t1), np.cos(t1)
        s12, c12 = np.sin(t1 + t2), np.cos(t1 + t2)

        return np.array([
            [-self.L1 * s1 - self.L2 * s12, -self.L2 * s12],
            [ self.L1 * c1 + self.L2 * c12,  self.L2 * c12]
        ])

    def numerical_jacobian(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Compute Jacobian numerically via finite differences.

        Useful for verification and for robots where the analytical Jacobian
        is too complex to derive by hand. The central difference formula
        provides O(eps^2) accuracy.
        """
        n = len(q)
        _, ee0 = self.fk(q)
        m = len(ee0)
        J = np.zeros((m, n))

        for i in range(n):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[i] += eps
            q_minus[i] -= eps
            _, ee_plus = self.fk(q_plus)
            _, ee_minus = self.fk(q_minus)
            J[:, i] = (ee_plus - ee_minus) / (2 * eps)

        return J


# ---------------------------------------------------------------------------
# Singularity analysis
# ---------------------------------------------------------------------------
def analyze_singularity(J: np.ndarray) -> dict:
    """Analyze the Jacobian for singularity indicators.

    A singularity occurs when:
      - det(J) = 0 (for square Jacobians)
      - One or more singular values are zero
      - The condition number is infinite

    Near singularities:
      - The robot loses a degree of freedom
      - IK becomes ill-conditioned (tiny Cartesian errors -> huge joint movements)
      - Control torques may spike to dangerous levels
    """
    U, sigma, Vt = np.linalg.svd(J)

    det = np.linalg.det(J) if J.shape[0] == J.shape[1] else None
    cond = sigma[0] / sigma[-1] if sigma[-1] > 1e-10 else float('inf')
    # Yoshikawa's manipulability measure: sqrt(det(J J^T))
    # For square J, this equals |det(J)|
    manipulability = np.sqrt(np.linalg.det(J @ J.T))

    return {
        "singular_values": sigma,
        "determinant": det,
        "condition_number": cond,
        "manipulability": manipulability,
        "U": U,  # Left singular vectors (Cartesian directions)
        "Vt": Vt  # Right singular vectors (joint space directions)
    }


def manipulability_ellipse(J: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """Compute the manipulability ellipsoid from the Jacobian.

    The manipulability ellipsoid shows how easily the end-effector can move
    in different Cartesian directions. It is defined by the set of all
    end-effector velocities achievable with unit joint velocity:
        {ẋ : ẋ = J q̇, ||q̇|| <= 1}

    The ellipse axes are the singular values of J, and the directions
    are the left singular vectors of J.

    A thin/elongated ellipse = the robot can move easily along one direction
    but not another (near singularity).
    A circular ellipse = equal mobility in all directions (isotropic config).
    """
    U, sigma, _ = np.linalg.svd(J)

    # The angle of the ellipse's major axis
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))

    return U, sigma[0], sigma[1], angle


# ---------------------------------------------------------------------------
# Force/torque duality
# ---------------------------------------------------------------------------
def force_torque_duality(J: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Compute end-effector force from joint torques using duality.

    The static force relationship is:  τ = J^T @ F
    Inverting:  F = (J^T)^{-1} @ τ  (for square, non-singular J)

    This duality is fundamental:
      - The same Jacobian that maps velocities also maps forces (transposed)
      - At singularities, the robot can exert large forces in certain directions
        (the same directions where it cannot move!)
    """
    if J.shape[0] == J.shape[1]:
        F = np.linalg.solve(J.T, tau)
    else:
        F = np.linalg.pinv(J.T) @ tau
    return F


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def demo_jacobian():
    """Demonstrate Jacobian computation, singularity analysis, and manipulability."""
    print("=" * 60)
    print("Jacobian and Singularity Analysis Demo")
    print("=" * 60)

    arm = TwoLinkArm(L1=1.0, L2=0.8)

    # --- Verify analytical vs numerical Jacobian ---
    q_test = np.array([np.pi / 4, np.pi / 6])
    J_analytical = arm.jacobian(q_test)
    J_numerical = arm.numerical_jacobian(q_test)
    print(f"\n[Verification] Analytical Jacobian:\n{J_analytical}")
    print(f"[Verification] Numerical Jacobian:\n{J_numerical}")
    print(f"[Verification] Max difference: {np.max(np.abs(J_analytical - J_numerical)):.2e}")

    # --- Singularity analysis at different configurations ---
    configs = {
        "Extended (q2=0)": np.array([np.pi / 4, 0.0]),
        "Near singular (q2~pi)": np.array([np.pi / 4, np.pi - 0.01]),
        "Folded (q2=pi/2)": np.array([np.pi / 4, np.pi / 2]),
        "General": np.array([np.pi / 6, np.pi / 3]),
    }

    print("\n--- Singularity Analysis ---")
    for name, q in configs.items():
        J = arm.jacobian(q)
        info = analyze_singularity(J)
        print(f"\n  {name}: q = {np.degrees(q)} deg")
        print(f"    det(J) = {info['determinant']:.6f}")
        print(f"    cond(J) = {info['condition_number']:.2f}")
        print(f"    manipulability = {info['manipulability']:.6f}")
        print(f"    singular values = {info['singular_values']}")

    # --- Plot manipulability ellipses across configurations ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax, (name, q) in zip(axes.flat, configs.items()):
        elbow, ee = arm.fk(q)

        # Draw arm
        xs = [0, elbow[0], ee[0]]
        ys = [0, elbow[1], ee[1]]
        ax.plot(xs, ys, 'b-o', linewidth=3, markersize=8)
        ax.plot(ee[0], ee[1], 'b*', markersize=15)

        # Compute and draw manipulability ellipse centered at EE
        J = arm.jacobian(q)
        U, w, h, angle = manipulability_ellipse(J)

        # Scale the ellipse for visibility
        scale = 0.5
        ellipse = Ellipse(xy=ee, width=2 * w * scale, height=2 * h * scale,
                          angle=angle, fill=False, edgecolor='red',
                          linewidth=2, linestyle='--')
        ax.add_patch(ellipse)

        # Draw principal axes of the ellipse
        for i in range(2):
            sigma_val = [w, h][i]
            direction = U[:, i] * sigma_val * scale
            ax.annotate('', xy=ee + direction, xytext=ee,
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        info = analyze_singularity(J)
        ax.set_title(f"{name}\ndet={info['determinant']:.4f}, "
                     f"cond={info['condition_number']:.1f}, "
                     f"manip={info['manipulability']:.4f}", fontsize=9)
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle("Manipulability Ellipsoids at Different Configurations", fontsize=14)
    plt.tight_layout()
    plt.savefig("04_jacobian_ellipsoids.png", dpi=120)
    plt.show()

    # --- Manipulability map across workspace ---
    fig, ax = plt.subplots(figsize=(8, 8))
    n_pts = 80
    t1_range = np.linspace(-np.pi, np.pi, n_pts)
    t2_range = np.linspace(-np.pi, np.pi, n_pts)

    x_all, y_all, m_all = [], [], []
    for t1 in t1_range:
        for t2 in t2_range:
            q = np.array([t1, t2])
            _, ee = arm.fk(q)
            J = arm.jacobian(q)
            info = analyze_singularity(J)
            x_all.append(ee[0])
            y_all.append(ee[1])
            m_all.append(info['manipulability'])

    scatter = ax.scatter(x_all, y_all, c=m_all, s=2, cmap='hot', alpha=0.6)
    plt.colorbar(scatter, label='Manipulability')
    ax.set_aspect('equal')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Manipulability Map (Cartesian Space)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("04_jacobian_manipulability_map.png", dpi=120)
    plt.show()

    # --- Force/torque duality ---
    print("\n--- Force/Torque Duality ---")
    q = np.array([np.pi / 4, np.pi / 3])
    J = arm.jacobian(q)
    tau = np.array([1.0, 0.5])  # Joint torques
    F = force_torque_duality(J, tau)
    print(f"  Joint torques: tau = {tau}")
    print(f"  End-effector force: F = {F}")
    print(f"  Verification: J^T @ F = {J.T @ F}  (should equal tau)")


if __name__ == "__main__":
    demo_jacobian()
