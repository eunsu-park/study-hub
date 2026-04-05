"""
Control Theory — Lesson 13: State Feedback and Observer Design

Demonstrates:
1. Pole placement via Ackermann's formula
2. Luenberger observer design
3. Observer-based feedback (separation principle)
4. Reference tracking with feedforward gain
"""
import numpy as np
from numpy.linalg import inv, eigvals, matrix_rank


# ── 1. Pole Placement (Ackermann's Formula) ──────────────────────────────

def ackermann_gain(A: np.ndarray, B: np.ndarray,
                   desired_poles: list[complex]) -> np.ndarray:
    """
    Compute state feedback gain K using Ackermann's formula (SISO).

    u = -Kx places eigenvalues of (A-BK) at desired_poles.
    K = [0 0 ... 1] * C^{-1} * Δ_d(A)
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float).reshape(-1, 1)
    n = A.shape[0]

    # Controllability matrix
    C_mat = B.copy()
    Ak_B = B.copy()
    for _ in range(1, n):
        Ak_B = A @ Ak_B
        C_mat = np.hstack([C_mat, Ak_B])

    if matrix_rank(C_mat) < n:
        raise ValueError("System is not controllable — pole placement impossible")

    # Desired characteristic polynomial: Δ_d(s) = Π(s - p_i)
    # Coefficients [1, α_{n-1}, ..., α_0]
    char_poly = np.real(np.poly(desired_poles))

    # Evaluate Δ_d(A) = A^n + α_{n-1}A^{n-1} + ... + α_0·I
    Delta_A = np.zeros_like(A)
    A_power = np.eye(n)
    for i in range(n + 1):
        Delta_A += char_poly[n - i] * A_power
        if i < n:
            A_power = A_power @ A

    # K = e_n^T * C^{-1} * Δ_d(A)
    e_n = np.zeros(n)
    e_n[-1] = 1.0
    K = e_n @ inv(C_mat) @ Delta_A

    return K.reshape(1, -1)


# ── 2. Observer Gain (Dual of Pole Placement) ────────────────────────────

def observer_gain(A: np.ndarray, C: np.ndarray,
                  desired_poles: list[complex]) -> np.ndarray:
    """
    Compute observer gain L using duality with Ackermann.

    Observer: ẋ̂ = Aẋ̂ + Bu + L(y - Cẋ̂)
    Eigenvalues of (A - LC) placed at desired_poles.
    """
    A = np.array(A, dtype=float)
    C = np.array(C, dtype=float).reshape(1, -1)

    # By duality: L = (K_dual)^T where K_dual places poles of (A^T - C^T K_dual^T)
    K_dual = ackermann_gain(A.T, C.T, desired_poles)
    L = K_dual.T

    return L


# ── 3. Feedforward Gain for Reference Tracking ──────────────────────────

def feedforward_gain(A: np.ndarray, B: np.ndarray, C: np.ndarray,
                     K: np.ndarray) -> float:
    """
    Compute N_r such that y_ss = r for step reference.
    N_r = 1 / [C(-A+BK)^{-1}B] = 1/G_cl(0)
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float).reshape(-1, 1)
    C = np.array(C, dtype=float).reshape(1, -1)
    K = np.array(K, dtype=float).reshape(1, -1)

    A_cl = A - B @ K
    G_cl_0 = C @ inv(-A_cl) @ B
    return 1.0 / G_cl_0[0, 0]


# ── 4. Simulation ────────────────────────────────────────────────────────

def simulate_observer_feedback(A, B, C, K, L, Nr, x0, x0_hat, r_func,
                               t, dt):
    """Simulate observer-based state feedback."""
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float).reshape(-1, 1)
    C = np.array(C, dtype=float).reshape(1, -1)
    K = np.array(K, dtype=float).reshape(1, -1)
    L = np.array(L, dtype=float).reshape(-1, 1)
    n = A.shape[0]

    x = np.zeros((len(t), n))
    x_hat = np.zeros((len(t), n))
    y = np.zeros(len(t))
    u = np.zeros(len(t))
    e_obs = np.zeros((len(t), n))  # observer error

    x[0] = x0
    x_hat[0] = x0_hat

    for i in range(len(t) - 1):
        r = r_func(t[i])
        xi = x[i].reshape(-1, 1)
        xi_hat = x_hat[i].reshape(-1, 1)

        # Output
        yi = (C @ xi).item()
        yi_hat = (C @ xi_hat).item()
        y[i] = yi

        # Control law
        ui = (-K @ xi_hat + Nr * r).item()
        u[i] = ui

        # Plant dynamics
        x_next = xi + dt * (A @ xi + B * ui)
        x[i + 1] = x_next.flatten()

        # Observer dynamics
        innovation = yi - yi_hat
        x_hat_next = xi_hat + dt * (A @ xi_hat + B * ui + L * innovation)
        x_hat[i + 1] = x_hat_next.flatten()

        e_obs[i] = (xi - xi_hat).flatten()

    y[-1] = (C @ x[-1].reshape(-1, 1)).item()
    e_obs[-1] = x[-1] - x_hat[-1]

    return {"x": x, "x_hat": x_hat, "y": y, "u": u, "e_obs": e_obs}


# ── Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Double integrator: ẍ = u
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])

    # Controller: place poles at s = -3 ± j4
    print("=== State Feedback: Double Integrator ===")
    desired_ctrl = [-3 + 4j, -3 - 4j]
    K = ackermann_gain(A, B, desired_ctrl)
    print(f"  Desired poles: {desired_ctrl}")
    print(f"  K = {K.flatten()}")
    A_cl = A - B @ K
    print(f"  CL poles: {eigvals(A_cl)}")
    wn = abs(desired_ctrl[0])
    zeta = -desired_ctrl[0].real / wn
    print(f"  ωn = {wn:.2f}, ζ = {zeta:.3f}")

    # Feedforward gain
    Nr = feedforward_gain(A, B, C, K)
    print(f"  Nr = {Nr:.4f}")

    # Observer: place poles 4x faster
    print("\n=== Observer Design ===")
    desired_obs = [-12 + 16j, -12 - 16j]
    L = observer_gain(A, C, desired_obs)
    print(f"  Desired observer poles: {desired_obs}")
    print(f"  L = {L.flatten()}")
    A_obs = A - L @ C
    print(f"  Observer poles: {eigvals(A_obs)}")

    # Separation principle: combined system eigenvalues
    print("\n=== Separation Principle ===")
    n = A.shape[0]
    A_combined = np.block([
        [A - B @ K, B @ K],
        [np.zeros((n, n)), A - L @ C]
    ])
    combined_eigs = eigvals(A_combined)
    print(f"  Combined system eigenvalues:")
    for e in sorted(combined_eigs, key=lambda x: x.real):
        print(f"    {e:.3f}")
    print(f"  = controller poles ∪ observer poles ✓")

    # Simulate
    print("\n=== Simulation ===")
    dt = 0.001
    t = np.arange(0, 5, dt)
    r_func = lambda t: 1.0  # unit step

    # Initial state mismatch (observer starts at wrong estimate)
    result = simulate_observer_feedback(
        A, B, C, K, L, Nr,
        x0=[0, 0],        # actual initial state
        x0_hat=[0.5, -1], # wrong initial estimate
        r_func=r_func, t=t, dt=dt
    )

    y = result["y"]
    e_obs = result["e_obs"]
    overshoot = (np.max(y) - 1.0) * 100
    ss_error = abs(1.0 - y[-1])
    obs_converge_time = 0
    for i in range(len(t)):
        if np.max(np.abs(e_obs[i:])) < 0.01:
            obs_converge_time = t[i]
            break

    print(f"  y_final = {y[-1]:.4f}")
    print(f"  Overshoot = {overshoot:.1f}%")
    print(f"  SS error = {ss_error:.6f}")
    print(f"  Observer converged in {obs_converge_time:.3f} s")
    print(f"  Max |u| = {np.max(np.abs(result['u'])):.2f}")
