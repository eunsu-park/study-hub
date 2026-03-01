"""
Robot Dynamics: Euler-Lagrange Formulation for a 2-Link Planar Arm
==================================================================
Compute equations of motion and simulate forward/inverse dynamics.

Robot dynamics describes the relationship between forces/torques and motion:
  - Forward dynamics: given torques τ, compute accelerations q̈
      M(q) q̈ + C(q, q̇) q̇ + g(q) = τ
  - Inverse dynamics: given desired trajectory (q, q̇, q̈), compute required τ
      τ = M(q) q̈ + C(q, q̇) q̇ + g(q)

The Euler-Lagrange approach derives these equations from kinetic and potential
energy — it is systematic and avoids the complexity of free-body diagrams.
For a 2-link arm, we can derive closed-form expressions.

Key matrices:
  M(q): inertia matrix — resistance to angular acceleration (always positive definite)
  C(q,q̇): Coriolis/centrifugal — velocity-dependent forces (gyroscopic effects)
  g(q): gravity vector — configuration-dependent gravitational loading
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class TwoLinkDynamics:
    """Euler-Lagrange dynamics for a 2-link planar manipulator.

    The robot has two revolute joints in the vertical plane (gravity acts
    along -Y). Each link is modeled as a uniform rod with mass concentrated
    at its center.

    Why Euler-Lagrange over Newton-Euler?
      - Gives a compact matrix formulation M q̈ + C q̇ + g = τ
      - No need to compute internal constraint forces
      - Directly reveals dynamic coupling between joints
      - Better suited for control design (e.g., computed torque)
    """

    def __init__(self, m1: float = 1.0, m2: float = 0.8,
                 L1: float = 1.0, L2: float = 0.8,
                 lc1: float = 0.5, lc2: float = 0.4,
                 I1: float = 0.083, I2: float = 0.043,
                 g: float = 9.81):
        """Initialize link parameters.

        Args:
            m1, m2: Link masses (kg)
            L1, L2: Link lengths (m)
            lc1, lc2: Distance from joint to link center of mass (m)
            I1, I2: Link moments of inertia about their centers (kg*m^2)
            g: Gravitational acceleration (m/s^2)

        Default inertias assume uniform rods: I = (1/12) m L^2
        """
        self.m1, self.m2 = m1, m2
        self.L1, self.L2 = L1, L2
        self.lc1, self.lc2 = lc1, lc2
        self.I1, self.I2 = I1, I2
        self.g = g

    def inertia_matrix(self, q: np.ndarray) -> np.ndarray:
        """Compute the 2x2 inertia (mass) matrix M(q).

        M(q) captures how much torque is needed to produce unit angular
        acceleration at each joint. Off-diagonal terms represent dynamic
        coupling: accelerating joint 1 requires torque at joint 2 and vice versa.

        M is always symmetric and positive definite — this ensures the
        kinetic energy T = 0.5 q̇^T M q̇ is always non-negative.
        """
        c2 = np.cos(q[1])
        # Pre-compute common terms to reduce redundancy
        h = self.m2 * self.L1 * self.lc2 * c2

        M11 = (self.m1 * self.lc1**2 + self.I1
               + self.m2 * (self.L1**2 + self.lc2**2 + 2 * self.L1 * self.lc2 * c2)
               + self.I2)
        M12 = self.m2 * self.lc2**2 + self.I2 + h
        M22 = self.m2 * self.lc2**2 + self.I2

        return np.array([[M11, M12],
                         [M12, M22]])

    def coriolis_matrix(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Compute the 2x2 Coriolis/centrifugal matrix C(q, q̇).

        Coriolis forces arise from the coupling between rotating frames.
        Centrifugal forces push outward when a link rotates.

        We use the Christoffel symbols approach:
            C_{ij} = sum_k c_{ijk} q̇_k
            c_{ijk} = 0.5 (∂M_{ij}/∂q_k + ∂M_{ik}/∂q_j - ∂M_{jk}/∂q_i)

        This formulation guarantees that (Ṁ - 2C) is skew-symmetric,
        which is essential for proving stability of many control laws.
        """
        s2 = np.sin(q[1])
        h = self.m2 * self.L1 * self.lc2 * s2

        return np.array([[-h * qd[1],    -h * (qd[0] + qd[1])],
                         [ h * qd[0],     0]])

    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """Compute the 2x1 gravity vector g(q).

        Gravity loads depend on the configuration: when links are horizontal,
        gravity torque is maximum; when vertical, it is zero.
        """
        g1 = ((self.m1 * self.lc1 + self.m2 * self.L1) * self.g * np.cos(q[0])
              + self.m2 * self.lc2 * self.g * np.cos(q[0] + q[1]))
        g2 = self.m2 * self.lc2 * self.g * np.cos(q[0] + q[1])

        return np.array([g1, g2])

    def forward_dynamics(self, q: np.ndarray, qd: np.ndarray,
                          tau: np.ndarray) -> np.ndarray:
        """Compute joint accelerations from applied torques.

        q̈ = M(q)^{-1} [τ - C(q,q̇) q̇ - g(q)]

        This is what a physics simulator computes at each time step:
        given the current state and applied torques, what is the acceleration?
        """
        M = self.inertia_matrix(q)
        C = self.coriolis_matrix(q, qd)
        g = self.gravity_vector(q)

        # Solve M @ qdd = tau - C @ qd - g  (more numerically stable than inv(M))
        qdd = np.linalg.solve(M, tau - C @ qd - g)
        return qdd

    def inverse_dynamics(self, q: np.ndarray, qd: np.ndarray,
                          qdd: np.ndarray) -> np.ndarray:
        """Compute required torques for a given trajectory.

        τ = M(q) q̈ + C(q,q̇) q̇ + g(q)

        Used in computed-torque control: if we can perfectly cancel the
        nonlinear dynamics (C, g) and decouple the inertia (M), the system
        becomes a set of independent double integrators — easy to control.
        """
        M = self.inertia_matrix(q)
        C = self.coriolis_matrix(q, qd)
        g = self.gravity_vector(q)

        return M @ qdd + C @ qd + g

    def simulate(self, q0: np.ndarray, qd0: np.ndarray,
                  torque_func, t_span: tuple, dt: float = 0.01) -> dict:
        """Simulate robot motion using forward dynamics.

        We integrate the ODE:  d/dt [q, q̇] = [q̇, M^{-1}(τ - Cq̇ - g)]

        Using scipy's solve_ivp with RK45 (adaptive Runge-Kutta) for accuracy.
        """
        def dynamics_ode(t, state):
            q = state[:2]
            qd = state[2:]
            tau = torque_func(t, q, qd)
            qdd = self.forward_dynamics(q, qd, tau)
            return np.concatenate([qd, qdd])

        t_eval = np.arange(t_span[0], t_span[1], dt)
        state0 = np.concatenate([q0, qd0])

        sol = solve_ivp(dynamics_ode, t_span, state0, t_eval=t_eval,
                        method='RK45', max_step=dt)

        return {
            "t": sol.t,
            "q": sol.y[:2].T,    # Joint angles
            "qd": sol.y[2:].T,   # Joint velocities
        }


# ---------------------------------------------------------------------------
# Trajectory generation for inverse dynamics demo
# ---------------------------------------------------------------------------
def cubic_trajectory(t: float, t_f: float, q0: float, qf: float):
    """Cubic polynomial trajectory: smooth start and end (zero velocity at endpoints).

    q(t) = a0 + a1*t + a2*t^2 + a3*t^3
    Boundary conditions: q(0)=q0, q(tf)=qf, q̇(0)=0, q̇(tf)=0
    """
    s = t / t_f  # Normalized time [0, 1]
    s = np.clip(s, 0, 1)
    # Cubic: 3s^2 - 2s^3  (Hermite basis with zero endpoint velocities)
    pos = q0 + (qf - q0) * (3 * s**2 - 2 * s**3)
    vel = (qf - q0) * (6 * s - 6 * s**2) / t_f
    acc = (qf - q0) * (6 - 12 * s) / t_f**2
    return pos, vel, acc


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo_dynamics():
    """Demonstrate forward and inverse dynamics of a 2-link arm."""
    print("=" * 60)
    print("Robot Dynamics (Euler-Lagrange) Demo")
    print("=" * 60)

    robot = TwoLinkDynamics()

    # --- Forward dynamics: free fall from horizontal ---
    print("\n--- Forward Dynamics: Free Fall (zero torque) ---")
    q0 = np.array([np.pi / 4, np.pi / 6])  # Initial angles
    qd0 = np.array([0.0, 0.0])              # Start from rest

    def zero_torque(t, q, qd):
        return np.array([0.0, 0.0])

    result = robot.simulate(q0, qd0, zero_torque, t_span=(0, 3.0))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot joint angles
    axes[0, 0].plot(result["t"], np.degrees(result["q"][:, 0]), label="θ₁")
    axes[0, 0].plot(result["t"], np.degrees(result["q"][:, 1]), label="θ₂")
    axes[0, 0].set_ylabel("Angle (deg)")
    axes[0, 0].set_title("Free Fall: Joint Angles")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot joint velocities
    axes[0, 1].plot(result["t"], result["qd"][:, 0], label="θ̇₁")
    axes[0, 1].plot(result["t"], result["qd"][:, 1], label="θ̇₂")
    axes[0, 1].set_ylabel("Angular velocity (rad/s)")
    axes[0, 1].set_title("Free Fall: Joint Velocities")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # --- Inverse dynamics: follow a smooth trajectory ---
    print("\n--- Inverse Dynamics: Computed Torques for Trajectory ---")
    t_f = 2.0
    q_start = np.array([0.0, 0.0])
    q_end = np.array([np.pi / 3, np.pi / 4])

    t_traj = np.linspace(0, t_f, 200)
    q_traj = np.zeros((len(t_traj), 2))
    qd_traj = np.zeros((len(t_traj), 2))
    qdd_traj = np.zeros((len(t_traj), 2))
    tau_traj = np.zeros((len(t_traj), 2))

    for i, t in enumerate(t_traj):
        for j in range(2):
            q_traj[i, j], qd_traj[i, j], qdd_traj[i, j] = cubic_trajectory(
                t, t_f, q_start[j], q_end[j])
        tau_traj[i] = robot.inverse_dynamics(q_traj[i], qd_traj[i], qdd_traj[i])

    # Plot desired trajectory
    axes[1, 0].plot(t_traj, np.degrees(q_traj[:, 0]), label="θ₁ desired")
    axes[1, 0].plot(t_traj, np.degrees(q_traj[:, 1]), label="θ₂ desired")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Angle (deg)")
    axes[1, 0].set_title("Inverse Dynamics: Desired Trajectory")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot required torques
    axes[1, 1].plot(t_traj, tau_traj[:, 0], label="τ₁")
    axes[1, 1].plot(t_traj, tau_traj[:, 1], label="τ₂")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Torque (N·m)")
    axes[1, 1].set_title("Inverse Dynamics: Required Torques")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("2-Link Planar Arm Dynamics", fontsize=14)
    plt.tight_layout()
    plt.savefig("05_dynamics.png", dpi=120)
    plt.show()

    # --- Verify: simulate with computed torques, should track trajectory ---
    print("\n--- Verification: Simulate with Computed Torques ---")

    def computed_torque(t, q, qd):
        """Look up the pre-computed inverse dynamics torque at time t."""
        idx = int(t / t_f * (len(t_traj) - 1))
        idx = np.clip(idx, 0, len(t_traj) - 1)
        return tau_traj[idx]

    result_check = robot.simulate(q_start, np.array([0.0, 0.0]),
                                   computed_torque, t_span=(0, t_f))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_traj, np.degrees(q_traj[:, 0]), 'b--', label="θ₁ desired", linewidth=2)
    ax.plot(t_traj, np.degrees(q_traj[:, 1]), 'r--', label="θ₂ desired", linewidth=2)
    ax.plot(result_check["t"], np.degrees(result_check["q"][:, 0]),
            'b-', label="θ₁ actual", alpha=0.7)
    ax.plot(result_check["t"], np.degrees(result_check["q"][:, 1]),
            'r-', label="θ₂ actual", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Verification: Desired vs Simulated Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("05_dynamics_verification.png", dpi=120)
    plt.show()

    max_err = np.max(np.abs(result_check["q"][-1] - q_end))
    print(f"  Max final angle error: {np.degrees(max_err):.4f} deg")


if __name__ == "__main__":
    demo_dynamics()
