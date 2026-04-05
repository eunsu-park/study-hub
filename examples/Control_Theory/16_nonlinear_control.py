"""
Control Theory — Lesson 16: Nonlinear and Advanced Control

Demonstrates:
1. Phase-plane analysis (Van der Pol oscillator)
2. Describing function for relay nonlinearity
3. Lyapunov stability analysis (energy-based)
4. Sliding mode control
5. Linearization and Jacobian computation
"""
import numpy as np


# ── 1. Phase-Plane Simulation ─────────────────────────────────────────

def van_der_pol(mu: float, x0: np.ndarray, dt: float = 0.01,
                t_end: float = 30.0) -> dict:
    """
    Simulate the Van der Pol oscillator:
      x'' - μ(1 - x²)x' + x = 0

    State: x1 = x, x2 = x'
      x1' = x2
      x2' = μ(1 - x1²)x2 - x1

    Args:
        mu: nonlinearity parameter (μ > 0 for limit cycle)
        x0: initial state [x1_0, x2_0]
        dt: time step
        t_end: simulation duration

    Returns:
        dict with 't', 'x1', 'x2' arrays
    """
    steps = int(t_end / dt)
    t = np.linspace(0, t_end, steps)
    x1 = np.zeros(steps)
    x2 = np.zeros(steps)
    x1[0], x2[0] = x0

    for i in range(steps - 1):
        # RK4 integration
        def f(state):
            return np.array([
                state[1],
                mu * (1 - state[0]**2) * state[1] - state[0]
            ])

        state = np.array([x1[i], x2[i]])
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        new_state = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        x1[i + 1] = new_state[0]
        x2[i + 1] = new_state[1]

    return {"t": t, "x1": x1, "x2": x2}


# ── 2. Describing Function: Relay ─────────────────────────────────────

def describing_function_relay(M: float, amplitude: float) -> complex:
    """
    Describing function N(A) for an ideal relay with output ±M.

    N(A) = 4M / (πA)   (real, no phase shift for ideal relay)

    Args:
        M: relay output magnitude
        amplitude: input sinusoid amplitude A
    """
    if amplitude <= 0:
        return float("inf")
    return 4 * M / (np.pi * amplitude)


def describing_function_relay_deadzone(M: float, delta: float,
                                       amplitude: float) -> complex:
    """
    Describing function for relay with dead zone ±δ.

    N(A) = (4M / πA) * sqrt(1 - (δ/A)²)   for A > δ
         = 0                                 for A ≤ δ

    Args:
        M: relay output magnitude
        delta: dead zone half-width
        amplitude: input sinusoid amplitude A
    """
    if amplitude <= delta:
        return 0.0
    return (4 * M / (np.pi * amplitude)) * np.sqrt(1 - (delta / amplitude)**2)


def predict_limit_cycle(M: float, plant_num: list[float],
                        plant_den: list[float],
                        omega_range: np.ndarray) -> dict:
    """
    Predict limit cycle using describing function method.

    The limit cycle occurs where:  -1/N(A) = G(jω)
    For an ideal relay: -1/N(A) = -πA/(4M)  (real negative axis)

    Find intersection of G(jω) with the negative real axis.
    """
    s = 1j * omega_range
    G = np.polyval(plant_num, s) / np.polyval(plant_den, s)

    # Find where imaginary part crosses zero (negative real axis crossing)
    crossings = []
    for i in range(len(omega_range) - 1):
        if G[i].imag * G[i + 1].imag < 0 and G[i].real < 0:
            # Interpolate
            frac = abs(G[i].imag) / (abs(G[i].imag) + abs(G[i + 1].imag))
            w_cross = omega_range[i] + frac * (omega_range[i + 1] - omega_range[i])
            g_cross = G[i].real + frac * (G[i + 1].real - G[i].real)
            # -1/N(A) = g_cross  =>  A = -4M / (π * g_cross)
            A_lc = -4 * M / (np.pi * g_cross)
            if A_lc > 0:
                crossings.append({"omega": w_cross, "amplitude": A_lc})

    return {"limit_cycles": crossings}


# ── 3. Lyapunov Stability ────────────────────────────────────────────

def lyapunov_quadratic(A: np.ndarray) -> dict:
    """
    Check Lyapunov stability using a quadratic Lyapunov function
    V(x) = x^T P x for the linear system x' = Ax.

    Solves the Lyapunov equation: A^T P + P A = -Q
    with Q = I (identity matrix).

    If P is positive definite, the system is stable.
    Uses a direct solver for small systems.
    """
    n = A.shape[0]
    Q = np.eye(n)

    # Solve A^T P + P A = -Q using vectorization (Kronecker product)
    I_n = np.eye(n)
    M = np.kron(I_n, A.T) + np.kron(A.T, I_n)
    p_vec = np.linalg.solve(M, -Q.flatten())
    P = p_vec.reshape(n, n)

    # Check positive definiteness via eigenvalues
    eig_P = np.linalg.eigvalsh(P)
    pd = bool(np.all(eig_P > 0))

    # System eigenvalues for verification
    eig_A = np.linalg.eigvals(A)

    return {
        "P": P,
        "P_eigenvalues": eig_P.tolist(),
        "P_positive_definite": pd,
        "system_stable": pd,
        "system_eigenvalues": np.sort_complex(eig_A).tolist(),
    }


# ── 4. Sliding Mode Control ──────────────────────────────────────────

def sliding_mode_sim(plant_a: float, plant_b: float,
                     reference: float, eta: float = 1.0,
                     lam: float = 2.0,
                     dt: float = 0.001, t_end: float = 5.0) -> dict:
    """
    Sliding mode controller for a second-order plant:
      x'' = a*x' + b*u + d(t)

    Sliding surface: s = e' + λe, where e = x - x_ref
    Control law: u = (1/b)[-a*x' + x_ref'' - λ*e' - η*sign(s)]

    Args:
        plant_a: plant parameter (damping)
        plant_b: plant parameter (input gain)
        reference: constant reference position
        eta: switching gain (robustness vs chattering)
        lam: sliding surface slope
        dt: time step
        t_end: simulation duration
    """
    steps = int(t_end / dt)
    t = np.linspace(0, t_end, steps)
    x = np.zeros(steps)      # position
    xd = np.zeros(steps)     # velocity
    u = np.zeros(steps)      # control input
    s_surf = np.zeros(steps) # sliding surface

    # Disturbance
    def disturbance(ti):
        return 0.5 * np.sin(3 * ti)

    for i in range(steps - 1):
        e = x[i] - reference
        ed = xd[i]  # reference velocity = 0

        s_surf[i] = ed + lam * e

        # Control law (reference is constant, so ref'' = 0)
        u_eq = (1 / plant_b) * (-plant_a * xd[i] - lam * ed)
        u_sw = -(eta / plant_b) * np.sign(s_surf[i])
        u[i] = u_eq + u_sw

        # Plant dynamics with disturbance
        d = disturbance(t[i])
        xdd = plant_a * xd[i] + plant_b * u[i] + d

        # Euler integration
        xd[i + 1] = xd[i] + dt * xdd
        x[i + 1] = x[i] + dt * xd[i]

    s_surf[-1] = xd[-1] + lam * (x[-1] - reference)
    u[-1] = u[-2]

    return {"t": t, "x": x, "xd": xd, "u": u, "s": s_surf}


# ── 5. Linearization via Jacobian ─────────────────────────────────────

def numerical_jacobian(f, x_eq: np.ndarray,
                       eps: float = 1e-7) -> np.ndarray:
    """
    Compute the Jacobian matrix ∂f/∂x at equilibrium point x_eq
    using central finite differences.

    Args:
        f: function f(x) -> np.ndarray (state derivative)
        x_eq: equilibrium state
        eps: perturbation size
    """
    n = len(x_eq)
    J = np.zeros((n, n))
    for j in range(n):
        x_plus = x_eq.copy()
        x_minus = x_eq.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        J[:, j] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return J


def linearize_pendulum(g: float = 9.81, L: float = 1.0,
                       b: float = 0.1) -> dict:
    """
    Linearize the nonlinear pendulum at equilibria.

    θ'' + (b/mL²)θ' + (g/L)sin(θ) = 0

    State: x = [θ, θ']
    Equilibria: θ = 0 (downward) and θ = π (upward)
    """
    def pendulum_dynamics(x):
        theta, theta_dot = x
        return np.array([
            theta_dot,
            -(b) * theta_dot - (g / L) * np.sin(theta)
        ])

    results = {}
    for name, theta_eq in [("downward (θ=0)", 0.0), ("upward (θ=π)", np.pi)]:
        x_eq = np.array([theta_eq, 0.0])
        A = numerical_jacobian(pendulum_dynamics, x_eq)
        eigenvalues = np.linalg.eigvals(A)
        stable = bool(np.all(eigenvalues.real < 0))
        results[name] = {
            "x_eq": x_eq.tolist(),
            "A": A.tolist(),
            "eigenvalues": eigenvalues.tolist(),
            "stable": stable,
        }

    return results


# ── Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Van der Pol oscillator
    print("=== Van der Pol Oscillator (μ=1.0) ===")
    vdp = van_der_pol(mu=1.0, x0=np.array([0.5, 0.0]), t_end=40.0)
    # Report limit cycle amplitude (approximate)
    peak_x1 = np.max(np.abs(vdp["x1"][-2000:]))
    print(f"  Limit cycle amplitude: ≈ {peak_x1:.3f}")
    print(f"  (Analytical: ≈ 2.0 for μ=1)")

    # 2. Describing function — relay
    print("\n=== Describing Function: Relay ===")
    M = 1.0
    for A in [0.5, 1.0, 2.0, 5.0]:
        N = describing_function_relay(M, A)
        print(f"  A = {A:.1f}: N(A) = {N:.4f}, -1/N = {-1/N:.4f}")

    # Predict limit cycle for relay + integrator chain
    print("\n  Limit cycle prediction: Relay + G(s) = 1/[s(s+1)]")
    plant_num = [1.0]
    plant_den = [1, 1, 0]  # s^2 + s
    omega_scan = np.linspace(0.01, 20, 50000)
    lc = predict_limit_cycle(M, plant_num, plant_den, omega_scan)
    for cycle in lc["limit_cycles"]:
        print(f"    ω = {cycle['omega']:.3f} rad/s, "
              f"A = {cycle['amplitude']:.3f}")

    # 3. Lyapunov stability
    print("\n=== Lyapunov Stability ===")
    # Stable system
    A_stable = np.array([[-1, 0.5], [-0.5, -2]])
    lyap = lyapunov_quadratic(A_stable)
    print(f"  A = {A_stable.tolist()}")
    print(f"  P eigenvalues: {[f'{e:.4f}' for e in lyap['P_eigenvalues']]}")
    print(f"  P positive definite: {lyap['P_positive_definite']}")
    print(f"  System eigenvalues: {lyap['system_eigenvalues']}")
    print(f"  Stable: {lyap['system_stable']}")

    # 4. Sliding mode control
    print("\n=== Sliding Mode Control ===")
    smc = sliding_mode_sim(plant_a=-0.5, plant_b=1.0,
                           reference=1.0, eta=2.0, lam=3.0)
    print(f"  Reference: 1.0")
    print(f"  Final position: {smc['x'][-1]:.4f}")
    print(f"  Final velocity: {smc['xd'][-1]:.6f}")
    print(f"  Final sliding surface: {smc['s'][-1]:.6f}")
    print(f"  Tracking error: {abs(smc['x'][-1] - 1.0):.6f}")

    # 5. Pendulum linearization
    print("\n=== Pendulum Linearization ===")
    pend = linearize_pendulum()
    for name, data in pend.items():
        print(f"\n  Equilibrium: {name}")
        print(f"    A = {[[f'{v:.4f}' for v in row] for row in data['A']]}")
        eigs = data["eigenvalues"]
        print(f"    Eigenvalues: {[f'{e:.4f}' for e in eigs]}")
        print(f"    Stable: {data['stable']}")
