"""
Control Theory — Lesson 14: Optimal Control (LQR and Kalman Filter)

Demonstrates:
1. LQR via algebraic Riccati equation (ARE)
2. Effect of Q/R weighting on response
3. Kalman filter for state estimation with noise
4. LQG (LQR + Kalman filter) combined controller
"""
import numpy as np
from numpy.linalg import inv, eigvals, solve


# ── 1. Algebraic Riccati Equation Solver ─────────────────────────────────

def solve_care(A: np.ndarray, B: np.ndarray, Q: np.ndarray,
               R: np.ndarray, max_iter: int = 1000,
               tol: float = 1e-10) -> np.ndarray:
    """
    Solve the continuous-time algebraic Riccati equation (CARE):
        A'P + PA - PBR^{-1}B'P + Q = 0

    Uses iterative (Schur/doubling-like) method.
    """
    n = A.shape[0]
    P = np.eye(n) * 10  # initial guess
    R_inv = inv(R)
    BR = B @ R_inv @ B.T

    for _ in range(max_iter):
        P_new = P + tol * (A.T @ P + P @ A - P @ BR @ P + Q)
        # More stable: solve Lyapunov-like iteration
        # (A - BR @ P)' P_new + P_new (A - BR @ P) = -(Q + P @ BR @ P)
        A_cl = A - BR @ P
        rhs = -(Q + P @ BR @ P)
        # Solve Lyapunov: A_cl' P_new + P_new A_cl = rhs
        P_new = solve_lyapunov(A_cl.T, rhs)
        if np.max(np.abs(P_new - P)) < tol:
            return P_new
        P = P_new

    return P


def solve_lyapunov(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Solve A'X + XA = Q using vectorization.
    """
    n = A.shape[0]
    # vec(A'X + XA) = (I⊗A' + A'⊗I) vec(X) = vec(Q)
    M = np.kron(np.eye(n), A.T) + np.kron(A.T, np.eye(n))
    x = solve(M, Q.flatten())
    return x.reshape(n, n)


def lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray,
         R: np.ndarray) -> tuple:
    """
    Compute LQR gain.

    Returns: (K, P, closed_loop_poles)
        K: optimal gain matrix (u = -Kx)
        P: solution of ARE
        poles: eigenvalues of A-BK
    """
    P = solve_care(A, B, Q, R)
    K = inv(R) @ B.T @ P
    poles = eigvals(A - B @ K)
    return K, P, poles


# ── 2. Kalman Filter ─────────────────────────────────────────────────────

def kalman_gain(A: np.ndarray, C: np.ndarray, W: np.ndarray,
                V: np.ndarray) -> tuple:
    """
    Compute steady-state Kalman filter gain.

    Plant: ẋ = Ax + Bu + Gw,  y = Cx + v
    (Assuming G = I for simplicity)

    Returns: (L, Pf, observer_poles)
    """
    # Filter ARE: A Pf + Pf A' - Pf C' V^{-1} C Pf + W = 0
    # This is dual to LQR: solve with A→A', B→C', Q→W, R→V
    Pf = solve_care(A.T, C.T, W, V)
    L = Pf @ C.T @ inv(V)
    poles = eigvals(A - L @ C)
    return L, Pf, poles


# ── 3. LQG Simulation ───────────────────────────────────────────────────

def simulate_lqg(A, B, C, K, L, x0, t, dt,
                 process_noise_std=0.0, meas_noise_std=0.0,
                 r_func=None):
    """Simulate LQG: LQR + Kalman filter."""
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float).reshape(-1, 1)
    C = np.array(C, dtype=float).reshape(1, -1)
    K = np.array(K, dtype=float).reshape(1, -1)
    L = np.array(L, dtype=float).reshape(-1, 1)
    n = A.shape[0]

    x = np.zeros((len(t), n))
    x_hat = np.zeros((len(t), n))
    y_meas = np.zeros(len(t))
    u = np.zeros(len(t))
    x[0] = x0
    x_hat[0] = np.zeros(n)

    for i in range(len(t) - 1):
        xi = x[i].reshape(-1, 1)
        xi_hat = x_hat[i].reshape(-1, 1)

        # Measurement with noise
        y_true = (C @ xi).item()
        noise_v = np.random.randn() * meas_noise_std
        y_meas[i] = y_true + noise_v

        # Control: u = -K x_hat (+ reference if given)
        r = r_func(t[i]) if r_func else 0
        # Compute Nr for tracking
        if r != 0:
            try:
                A_cl = A - B @ K
                Nr = 1.0 / (C @ inv(-A_cl) @ B).item()
            except Exception:
                Nr = 0
            ui = (-K @ xi_hat + Nr * r).item()
        else:
            ui = (-K @ xi_hat).item()
        u[i] = ui

        # Process noise
        w = np.random.randn(n) * process_noise_std

        # Plant dynamics
        x_next = xi + dt * (A @ xi + B * ui) + (dt * w).reshape(-1, 1)
        x[i + 1] = x_next.flatten()

        # Kalman filter update
        innovation = y_meas[i] - (C @ xi_hat).item()
        x_hat_next = xi_hat + dt * (A @ xi_hat + B * ui + L * innovation)
        x_hat[i + 1] = x_hat_next.flatten()

    y_meas[-1] = (C @ x[-1].reshape(-1, 1)).item()

    return {"x": x, "x_hat": x_hat, "y": y_meas, "u": u}


# ── Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Double integrator
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])

    # LQR design with different weightings
    print("=== LQR: Double Integrator ===")
    print(f"{'Q/R ratio':>12s} | {'K':>20s} | {'CL Poles':>30s} | {'ζ':>6s}")
    print("-" * 75)

    for rho in [10, 1, 0.1, 0.01]:
        Q = np.diag([1, 0])
        R = np.array([[rho]])
        K_lqr, P, poles = lqr(A, B, Q, R)
        wn = abs(poles[0])
        zeta = -poles[0].real / wn if wn > 0 else 0
        print(f"  Q/R = {1/rho:>5.0f}  | K = [{K_lqr[0,0]:6.3f}, {K_lqr[0,1]:6.3f}] | "
              f"poles = {poles} | ζ = {zeta:.3f}")

    # Standard LQR design
    Q = np.diag([1, 0])
    R = np.array([[1.0]])
    K_lqr, P_lqr, poles_ctrl = lqr(A, B, Q, R)
    print(f"\n  Selected: Q=diag(1,0), R=1")
    print(f"  K = {K_lqr.flatten()}")
    print(f"  CL poles = {poles_ctrl}")

    # Kalman filter
    print("\n=== Kalman Filter ===")
    W = np.diag([0.1, 0.1])  # process noise covariance
    V = np.array([[1.0]])     # measurement noise covariance
    L_kf, Pf, poles_obs = kalman_gain(A, C, W, V)
    print(f"  W = diag(0.1, 0.1), V = 1")
    print(f"  L = {L_kf.flatten()}")
    print(f"  Observer poles = {poles_obs}")

    # LQG simulation
    print("\n=== LQG Simulation (with noise) ===")
    np.random.seed(42)
    dt = 0.01
    t = np.arange(0, 10, dt)

    result = simulate_lqg(
        A, B, C, K_lqr, L_kf,
        x0=[1, 0],  # initial displacement
        t=t, dt=dt,
        process_noise_std=0.01,
        meas_noise_std=0.1,
    )

    # Performance
    x_final = result["x"][-100:, 0]  # last 1 second
    mean_error = np.mean(np.abs(x_final))
    max_u = np.max(np.abs(result["u"]))
    est_error = np.mean(np.abs(result["x"][-100:] - result["x_hat"][-100:]))

    print(f"  Mean |x1| (last 1s): {mean_error:.4f}")
    print(f"  Max |u|: {max_u:.3f}")
    print(f"  Mean estimation error (last 1s): {est_error:.4f}")

    # LQR guaranteed margins
    print("\n=== LQR Robustness (SISO) ===")
    print("  Guaranteed margins:")
    print("    Gain margin: [1/2, ∞) = [-6 dB, ∞)")
    print("    Phase margin: ≥ 60°")
    print("  Note: LQG does NOT have these guarantees!")

    # Verify return difference inequality |1 + KΦ(jω)B| ≥ 1
    omega = np.logspace(-2, 2, 1000)
    min_rd = float('inf')
    for w in omega:
        s = 1j * w
        Phi = inv(s * np.eye(2) - A)
        L_loop = (K_lqr @ Phi @ B).item()
        rd = abs(1 + L_loop)
        min_rd = min(min_rd, rd)
    print(f"  min|1 + K(jωI-A)⁻¹B| = {min_rd:.4f} (≥ 1 guaranteed)")
