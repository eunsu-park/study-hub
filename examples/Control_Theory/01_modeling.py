"""
Control Theory — Lesson 2: Mathematical Modeling of Physical Systems

Demonstrates modeling of:
1. Mass-spring-damper system
2. DC motor (electrical + mechanical coupling)
3. Linearization of nonlinear pendulum
"""
import numpy as np
from typing import NamedTuple


# ── 1. Mass-Spring-Damper ────────────────────────────────────────────────

class MassSpringDamper(NamedTuple):
    m: float  # mass [kg]
    b: float  # damping coefficient [N·s/m]
    k: float  # spring constant [N/m]

    @property
    def omega_n(self) -> float:
        """Natural frequency [rad/s]."""
        return np.sqrt(self.k / self.m)

    @property
    def zeta(self) -> float:
        """Damping ratio (dimensionless)."""
        return self.b / (2 * np.sqrt(self.m * self.k))

    def state_space(self):
        """Return (A, B, C, D) matrices.  State: [x, x_dot]."""
        A = np.array([[0, 1],
                      [-self.k / self.m, -self.b / self.m]])
        B = np.array([[0], [1 / self.m]])
        C = np.array([[1, 0]])
        D = np.array([[0]])
        return A, B, C, D


def simulate_msd(sys: MassSpringDamper, F_func, x0, t):
    """Euler integration of mass-spring-damper."""
    A, B, _, _ = sys.state_space()
    dt = t[1] - t[0]
    x = np.zeros((len(t), 2))
    x[0] = x0
    for i in range(len(t) - 1):
        u = np.array([F_func(t[i])])
        x[i + 1] = x[i] + dt * (A @ x[i] + (B @ u).flatten())
    return x


# ── 2. DC Motor Model ───────────────────────────────────────────────────

class DCMotor(NamedTuple):
    Ra: float   # armature resistance [Ω]
    La: float   # armature inductance [H]
    Kt: float   # torque constant [N·m/A]
    Kb: float   # back-EMF constant [V·s/rad]
    J: float    # rotor inertia [kg·m²]
    B: float    # viscous friction [N·m·s/rad]

    def state_space(self):
        """State: [θ, θ_dot, i_a].  Input: v_a.  Output: θ."""
        A = np.array([
            [0, 1, 0],
            [0, -self.B / self.J, self.Kt / self.J],
            [0, -self.Kb / self.La, -self.Ra / self.La]
        ])
        B = np.array([[0], [0], [1 / self.La]])
        C = np.array([[1, 0, 0]])
        D = np.array([[0]])
        return A, B, C, D

    def transfer_function_approx(self):
        """TF θ(s)/V_a(s) when La ≈ 0 (first-order armature)."""
        num = self.Kt
        den_coeffs = [self.J * self.Ra,
                      self.B * self.Ra + self.Kt * self.Kb,
                      0]  # s * (Js*Ra + B*Ra + Kt*Kb)
        return num, den_coeffs


# ── 3. Linearization: Nonlinear Pendulum ────────────────────────────────

def pendulum_linearize(m: float, l: float, g: float, theta_eq: float):
    """
    Linearize  ml²θ̈ + mgl·sin(θ) = τ  around θ = theta_eq.

    Returns (A, B) for the perturbation system δẋ = Aδx + Bδu.
    State: [δθ, δθ_dot].  Input: δτ.
    """
    # d/dθ [sin(θ)] at θ_eq = cos(θ_eq)
    cos_eq = np.cos(theta_eq)
    A = np.array([[0, 1],
                  [-g * cos_eq / l, 0]])
    B = np.array([[0], [1 / (m * l**2)]])
    return A, B


# ── Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Mass-spring-damper
    msd = MassSpringDamper(m=1.0, b=0.5, k=4.0)
    print("=== Mass-Spring-Damper ===")
    print(f"  ωn = {msd.omega_n:.3f} rad/s")
    print(f"  ζ  = {msd.zeta:.3f}")
    A, B, C, D = msd.state_space()
    print(f"  A = {A.tolist()}")
    print(f"  B = {B.flatten().tolist()}")
    eigvals = np.linalg.eigvals(A)
    print(f"  Poles: {eigvals}")

    # DC motor
    motor = DCMotor(Ra=1.0, La=0.01, Kt=0.05, Kb=0.05, J=0.001, B=0.0001)
    print("\n=== DC Motor ===")
    A, B, C, D = motor.state_space()
    print(f"  A =\n{A}")
    poles = np.linalg.eigvals(A)
    print(f"  Poles: {np.sort(poles.real)}")

    # Pendulum linearization
    print("\n=== Pendulum Linearization ===")
    m, l, g = 1.0, 1.0, 9.81
    # Hanging equilibrium (θ = 0)
    A_hang, _ = pendulum_linearize(m, l, g, theta_eq=0)
    eig_hang = np.linalg.eigvals(A_hang)
    print(f"  θ_eq = 0 (hanging):  eigenvalues = {eig_hang}")
    print(f"    → {'Stable' if all(e.real <= 0 for e in eig_hang) else 'Unstable'}")

    # Inverted equilibrium (θ = π)
    A_inv, _ = pendulum_linearize(m, l, g, theta_eq=np.pi)
    eig_inv = np.linalg.eigvals(A_inv)
    print(f"  θ_eq = π (inverted): eigenvalues = {eig_inv}")
    print(f"    → {'Stable' if all(e.real <= 0 for e in eig_inv) else 'Unstable'}")

    # Step response of MSD
    t = np.linspace(0, 10, 1000)
    step_force = lambda t: 1.0  # unit step
    x = simulate_msd(msd, step_force, x0=[0, 0], t=t)
    final_value = x[-1, 0]
    expected = 1.0 / msd.k  # F/k for unit step
    print(f"\n=== MSD Step Response ===")
    print(f"  Final value: {final_value:.4f}  (expected: {expected:.4f})")
