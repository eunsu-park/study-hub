"""
PID Control and Computed Torque Control for Robotic Arms
========================================================
Simulate closed-loop control of a 2-link planar manipulator.

Control is the bridge between planning and execution. A motion planner
generates a desired trajectory, but the robot must track it despite:
  - Gravity loading (configuration-dependent)
  - Dynamic coupling between joints
  - Disturbances and model uncertainty

We implement three controllers of increasing sophistication:
  1. PID: Simple, model-free, but struggles with nonlinear dynamics
  2. PD + Gravity Compensation: Cancels the dominant nonlinearity
  3. Computed Torque: Fully linearizes and decouples the dynamics

Each controller is tested on the same step response and trajectory
tracking tasks for fair comparison.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Robot dynamics (reused from 05_dynamics.py)
# ---------------------------------------------------------------------------
class TwoLinkDynamics:
    """Simplified 2-link planar arm dynamics for control simulation.

    We reimplement the essential dynamics here to keep the file self-contained.
    See 05_dynamics.py for detailed derivations and explanations.
    """

    def __init__(self, m1=1.0, m2=0.8, L1=1.0, L2=0.8,
                 lc1=0.5, lc2=0.4, I1=0.083, I2=0.043, g=9.81):
        self.m1, self.m2 = m1, m2
        self.L1, self.L2 = L1, L2
        self.lc1, self.lc2 = lc1, lc2
        self.I1, self.I2 = I1, I2
        self.g = g

    def M(self, q):
        """Inertia matrix."""
        c2 = np.cos(q[1])
        h = self.m2 * self.L1 * self.lc2 * c2
        M11 = self.m1 * self.lc1**2 + self.I1 + self.m2 * (self.L1**2 + self.lc2**2) + 2*h + self.I2
        M12 = self.m2 * self.lc2**2 + self.I2 + h
        M22 = self.m2 * self.lc2**2 + self.I2
        return np.array([[M11, M12], [M12, M22]])

    def C(self, q, qd):
        """Coriolis matrix."""
        s2 = np.sin(q[1])
        h = self.m2 * self.L1 * self.lc2 * s2
        return np.array([[-h*qd[1], -h*(qd[0]+qd[1])],
                         [h*qd[0], 0]])

    def G(self, q):
        """Gravity vector."""
        g1 = ((self.m1*self.lc1 + self.m2*self.L1) * self.g * np.cos(q[0])
              + self.m2*self.lc2*self.g*np.cos(q[0]+q[1]))
        g2 = self.m2*self.lc2*self.g*np.cos(q[0]+q[1])
        return np.array([g1, g2])

    def forward_dynamics(self, q, qd, tau):
        """q̈ = M^{-1}(τ - Cq̇ - g)"""
        return np.linalg.solve(self.M(q), tau - self.C(q, qd) @ qd - self.G(q))


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------
class PIDController:
    """Independent-joint PID controller.

    Each joint has its own PID loop, treating the arm as if the joints
    were decoupled. This ignores dynamic coupling (the inertia matrix
    is not diagonal) and gravity.

    Why PID despite its limitations?
      - Simple to tune (3 gains per joint)
      - No model needed (model-free)
      - Works reasonably well for slow motions or small workspaces
      - Ubiquitous in industry as a baseline controller
    """

    def __init__(self, Kp: np.ndarray, Ki: np.ndarray, Kd: np.ndarray):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = np.zeros(2)

    def compute(self, q_des, qd_des, q, qd, dt=0.01):
        """Compute PID torque command.

        τ = Kp * e + Ki * ∫e dt + Kd * ė
        where e = q_des - q, ė = qd_des - qd
        """
        error = q_des - q
        error_dot = qd_des - qd
        self.integral += error * dt

        # Anti-windup: clamp integral to prevent unbounded growth
        self.integral = np.clip(self.integral, -10.0, 10.0)

        return self.Kp * error + self.Ki * self.integral + self.Kd * error_dot


class PDGravityController:
    """PD control with gravity compensation.

    τ = Kp * e + Kd * ė + g(q)

    By adding the gravity term g(q), we cancel the largest nonlinearity.
    The remaining dynamics (inertia coupling, Coriolis) are small for
    slow motions, so PD handles them adequately.

    This is often the "sweet spot" in practice: much better than pure PID
    for robotic arms, but requires knowing the gravity model.
    """

    def __init__(self, Kp: np.ndarray, Kd: np.ndarray, robot: TwoLinkDynamics):
        self.Kp = Kp
        self.Kd = Kd
        self.robot = robot

    def compute(self, q_des, qd_des, q, qd, dt=0.01):
        error = q_des - q
        error_dot = qd_des - qd
        return self.Kp * error + self.Kd * error_dot + self.robot.G(q)


class ComputedTorqueController:
    """Computed torque (inverse dynamics) control.

    τ = M(q) * (q̈_des + Kp * e + Kd * ė) + C(q,q̇) * q̇ + g(q)

    This fully linearizes and decouples the system. After canceling M, C, g,
    the closed-loop dynamics become:
        ë + Kd ė + Kp e = 0  (linear, decoupled!)

    With Kp and Kd chosen for critical damping, each joint behaves as an
    independent second-order system. This is the gold standard for model-based
    robot control.

    Drawback: Requires an accurate dynamic model. Model errors lead to
    imperfect cancellation and residual coupling.
    """

    def __init__(self, Kp: np.ndarray, Kd: np.ndarray, robot: TwoLinkDynamics):
        self.Kp = Kp
        self.Kd = Kd
        self.robot = robot

    def compute(self, q_des, qd_des, qdd_des, q, qd, dt=0.01):
        error = q_des - q
        error_dot = qd_des - qd

        # Desired acceleration with PD correction
        v = qdd_des + self.Kd * error_dot + self.Kp * error

        # Inverse dynamics: cancel all nonlinearities
        tau = (self.robot.M(q) @ v
               + self.robot.C(q, qd) @ qd
               + self.robot.G(q))
        return tau


# ---------------------------------------------------------------------------
# Trajectory for testing
# ---------------------------------------------------------------------------
def desired_trajectory(t, t_f=3.0):
    """Minimum-jerk trajectory from 0 to target angles."""
    q_start = np.array([0.0, 0.0])
    q_end = np.array([np.pi / 3, np.pi / 4])

    s = np.clip(t / t_f, 0, 1)
    poly = 10 * s**3 - 15 * s**4 + 6 * s**5
    dpoly = (30 * s**2 - 60 * s**3 + 30 * s**4) / t_f
    ddpoly = (60 * s - 180 * s**2 + 120 * s**3) / t_f**2

    q_des = q_start + (q_end - q_start) * poly
    qd_des = (q_end - q_start) * dpoly
    qdd_des = (q_end - q_start) * ddpoly

    return q_des, qd_des, qdd_des


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate_control(robot, controller, controller_type, t_span=(0, 4.0), dt=0.005):
    """Simulate the closed-loop system."""

    def dynamics(t, state):
        q = state[:2]
        qd = state[2:]
        q_des, qd_des, qdd_des = desired_trajectory(t)

        if controller_type == "computed_torque":
            tau = controller.compute(q_des, qd_des, qdd_des, q, qd, dt)
        else:
            tau = controller.compute(q_des, qd_des, q, qd, dt)

        # Clamp torques to realistic limits
        tau = np.clip(tau, -50.0, 50.0)
        qdd = robot.forward_dynamics(q, qd, tau)
        return np.concatenate([qd, qdd])

    t_eval = np.arange(t_span[0], t_span[1], dt)
    state0 = np.zeros(4)

    sol = solve_ivp(dynamics, t_span, state0, t_eval=t_eval,
                    method='RK45', max_step=dt)

    # Compute desired trajectory for comparison
    q_des_arr = np.array([desired_trajectory(ti)[0] for ti in sol.t])

    return {
        "t": sol.t,
        "q": sol.y[:2].T,
        "qd": sol.y[2:].T,
        "q_des": q_des_arr,
        "error": q_des_arr - sol.y[:2].T
    }


def demo_pid_control():
    """Demonstrate and compare different control strategies."""
    print("=" * 60)
    print("PID and Computed Torque Control Demo")
    print("=" * 60)

    robot = TwoLinkDynamics()

    # Controller gains — tuned for fair comparison
    # PID gains: moderate, typical starting values
    pid = PIDController(
        Kp=np.array([80.0, 60.0]),
        Ki=np.array([5.0, 3.0]),
        Kd=np.array([20.0, 15.0])
    )

    # PD + gravity: higher PD gains since we compensate gravity
    pd_grav = PDGravityController(
        Kp=np.array([100.0, 80.0]),
        Kd=np.array([25.0, 20.0]),
        robot=robot
    )

    # Computed torque: can use lower gains since dynamics are canceled
    # Kp, Kd chosen for critical damping: Kd = 2*sqrt(Kp)
    ct = ComputedTorqueController(
        Kp=np.array([100.0, 100.0]),
        Kd=np.array([20.0, 20.0]),
        robot=robot
    )

    # Simulate all three controllers
    print("\nSimulating PID control...")
    res_pid = simulate_control(robot, pid, "pid")
    print("Simulating PD + Gravity Compensation...")
    res_pdg = simulate_control(robot, pd_grav, "pd_gravity")
    print("Simulating Computed Torque control...")
    res_ct = simulate_control(robot, ct, "computed_torque")

    # --- Plot results ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    controllers = [
        ("PID", res_pid, '#1f77b4'),
        ("PD+Gravity", res_pdg, '#ff7f0e'),
        ("Computed Torque", res_ct, '#2ca02c')
    ]

    for name, res, color in controllers:
        # Joint 1 tracking
        axes[0, 0].plot(res["t"], np.degrees(res["q"][:, 0]),
                         color=color, linewidth=1.5, label=f'{name}')
        # Joint 2 tracking
        axes[0, 1].plot(res["t"], np.degrees(res["q"][:, 1]),
                         color=color, linewidth=1.5, label=f'{name}')
        # Joint 1 error
        axes[1, 0].plot(res["t"], np.degrees(res["error"][:, 0]),
                         color=color, linewidth=1.5, label=f'{name}')
        # Joint 2 error
        axes[1, 1].plot(res["t"], np.degrees(res["error"][:, 1]),
                         color=color, linewidth=1.5, label=f'{name}')

    # Plot desired trajectory
    axes[0, 0].plot(res_pid["t"], np.degrees(res_pid["q_des"][:, 0]),
                     'k--', linewidth=2, label='Desired')
    axes[0, 1].plot(res_pid["t"], np.degrees(res_pid["q_des"][:, 1]),
                     'k--', linewidth=2, label='Desired')

    titles = [
        ("Joint 1: Tracking", "Joint 2: Tracking"),
        ("Joint 1: Error", "Joint 2: Error"),
    ]

    for i in range(2):
        for j in range(2):
            axes[i, j].set_title(titles[i][j])
            axes[i, j].legend(fontsize=8)
            axes[i, j].grid(True, alpha=0.3)
            if i == 0:
                axes[i, j].set_ylabel("Angle (deg)")
            else:
                axes[i, j].set_ylabel("Error (deg)")

    # RMS error comparison
    rms_errors = {}
    for name, res, color in controllers:
        rms = np.sqrt(np.mean(res["error"]**2, axis=0))
        rms_errors[name] = rms
        print(f"  {name:20s}: RMS error = [{np.degrees(rms[0]):.3f}, {np.degrees(rms[1]):.3f}] deg")

    # Bar chart of RMS errors
    names = list(rms_errors.keys())
    rms_j1 = [np.degrees(rms_errors[n][0]) for n in names]
    rms_j2 = [np.degrees(rms_errors[n][1]) for n in names]
    x = np.arange(len(names))
    width = 0.35

    axes[2, 0].bar(x - width/2, rms_j1, width, label='Joint 1', color='steelblue')
    axes[2, 0].bar(x + width/2, rms_j2, width, label='Joint 2', color='coral')
    axes[2, 0].set_xticks(x)
    axes[2, 0].set_xticklabels(names, fontsize=9)
    axes[2, 0].set_ylabel("RMS Error (deg)")
    axes[2, 0].set_title("RMS Tracking Error Comparison")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3, axis='y')

    # Max error comparison
    max_j1 = [np.degrees(np.max(np.abs(rms_errors[n][0]))) for n in names]
    max_j2 = [np.degrees(np.max(np.abs(rms_errors[n][1]))) for n in names]
    axes[2, 1].bar(x - width/2, max_j1, width, label='Joint 1', color='steelblue')
    axes[2, 1].bar(x + width/2, max_j2, width, label='Joint 2', color='coral')
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(names, fontsize=9)
    axes[2, 1].set_ylabel("Max Error (deg)")
    axes[2, 1].set_title("Max Tracking Error Comparison")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle("Controller Comparison: PID vs PD+Gravity vs Computed Torque", fontsize=14)
    plt.tight_layout()
    plt.savefig("08_pid_control.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    demo_pid_control()
