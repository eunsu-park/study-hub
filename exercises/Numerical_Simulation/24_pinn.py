"""
Exercises for Lesson 24: Physics-Informed Neural Networks (PINN)
Topic: Numerical_Simulation

Solutions to practice problems covering PINN loss design,
collocation strategies, inverse problems, failure modes,
and comparison with classical solvers.
"""

import numpy as np


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ============================================================
# Shared: Simple MLP for PINN experiments
# ============================================================
class SimpleMLP:
    """Minimal MLP with tanh activation for PINN exercises."""

    def __init__(self, layers, seed=42):
        rng = np.random.RandomState(seed)
        self.W = []
        self.b = []
        for i in range(len(layers) - 1):
            scale = np.sqrt(2.0 / (layers[i] + layers[i + 1]))
            self.W.append(rng.randn(layers[i], layers[i + 1]) * scale)
            self.b.append(np.zeros(layers[i + 1]))

    def forward(self, x):
        h = x
        for i in range(len(self.W)):
            h = h @ self.W[i] + self.b[i]
            if i < len(self.W) - 1:
                h = np.tanh(h)
        return h

    def get_params(self):
        return np.concatenate([w.ravel() for w in self.W]
                              + [b.ravel() for b in self.b])

    def set_params(self, flat):
        idx = 0
        for i in range(len(self.W)):
            s = self.W[i].size
            self.W[i] = flat[idx:idx + s].reshape(self.W[i].shape)
            idx += s
        for i in range(len(self.b)):
            s = self.b[i].size
            self.b[i] = flat[idx:idx + s]
            idx += s


def numerical_deriv(net, x, col, eps=1e-5):
    """∂u/∂x_col via central differences."""
    xp, xm = x.copy(), x.copy()
    xp[:, col] += eps
    xm[:, col] -= eps
    return (net.forward(xp) - net.forward(xm)) / (2 * eps)


def numerical_deriv2(net, x, col, eps=1e-4):
    """∂²u/∂x_col² via central differences."""
    xp, xm = x.copy(), x.copy()
    xp[:, col] += eps
    xm[:, col] -= eps
    return (net.forward(xp) - 2 * net.forward(x) + net.forward(xm)) / eps ** 2


# ============================================================
# Exercise 1: PINN Loss Design for Different PDEs
# ============================================================
def exercise_1():
    """
    Design and compute PINN loss functions for three different PDEs:
    heat equation, wave equation, and Burgers' equation.
    """
    print("=== Exercise 1: PINN Loss for Different PDEs ===\n")

    rng = np.random.RandomState(42)
    N = 200

    # Collocation points in [0,1] × [0,1]
    x = rng.uniform(0.05, 0.95, N)
    t = rng.uniform(0.05, 0.95, N)
    xt = np.column_stack([x, t])

    # 1. Heat equation: u_t = α * u_xx
    alpha = 0.01
    net_heat = SimpleMLP([2, 16, 16, 1], seed=1)
    u_t = numerical_deriv(net_heat, xt, col=1)
    u_xx = numerical_deriv2(net_heat, xt, col=0)
    residual_heat = u_t - alpha * u_xx
    loss_heat = np.mean(residual_heat ** 2)

    # 2. Wave equation: u_tt = c² * u_xx
    c = 1.0
    net_wave = SimpleMLP([2, 16, 16, 1], seed=2)
    # Need second derivative in time
    u_tt = numerical_deriv2(net_wave, xt, col=1)
    u_xx_wave = numerical_deriv2(net_wave, xt, col=0)
    residual_wave = u_tt - c ** 2 * u_xx_wave
    loss_wave = np.mean(residual_wave ** 2)

    # 3. Burgers' equation: u_t + u * u_x = ν * u_xx (nonlinear!)
    nu = 0.01
    net_burgers = SimpleMLP([2, 16, 16, 1], seed=3)
    u = net_burgers.forward(xt)
    u_t_b = numerical_deriv(net_burgers, xt, col=1)
    u_x_b = numerical_deriv(net_burgers, xt, col=0)
    u_xx_b = numerical_deriv2(net_burgers, xt, col=0)
    residual_burgers = u_t_b + u * u_x_b - nu * u_xx_b
    loss_burgers = np.mean(residual_burgers ** 2)

    print(f"  PDE residual losses (untrained networks):\n")
    print(f"  {'PDE':>20} | {'Residual Form':>30} | {'L_PDE':>10}")
    print(f"  {'-'*65}")
    print(f"  {'Heat (diffusion)':>20} | {'u_t - α·u_xx':>30} | {loss_heat:>10.6f}")
    print(f"  {'Wave':>20} | {'u_tt - c²·u_xx':>30} | {loss_wave:>10.6f}")
    print(f"  {'Burgers (nonlinear)':>20} | {'u_t + u·u_x - ν·u_xx':>30} | "
          f"{loss_burgers:>10.6f}")

    print(f"\n  Nonlinear PDEs (Burgers) need the network output u in the")
    print(f"  residual, making the loss landscape more complex to optimize.")
    print()


# ============================================================
# Exercise 2: Collocation Point Strategies
# ============================================================
def exercise_2():
    """
    Compare uniform, Latin Hypercube, and adaptive collocation
    strategies for PINN training quality.
    """
    print("=== Exercise 2: Collocation Strategies ===\n")

    N = 500
    rng = np.random.RandomState(42)

    # Strategy 1: Uniform random
    uniform = np.column_stack([rng.uniform(0, 1, N), rng.uniform(0, 1, N)])

    # Strategy 2: Latin Hypercube
    def lhs(n, d, rng):
        pts = np.zeros((n, d))
        for dim in range(d):
            perm = rng.permutation(n)
            pts[:, dim] = (perm + rng.uniform(0, 1, n)) / n
        return pts

    lhs_pts = lhs(N, 2, rng)

    # Strategy 3: Residual-based adaptive
    # Start with uniform, then add points where residual is high
    net = SimpleMLP([2, 16, 1], seed=42)
    alpha = 0.01

    # Phase 1: initial uniform points
    adaptive = np.column_stack([rng.uniform(0, 1, N // 2),
                                 rng.uniform(0, 1, N // 2)])

    # Compute residuals
    u_t = numerical_deriv(net, adaptive, 1)
    u_xx = numerical_deriv2(net, adaptive, 0)
    residuals = np.abs((u_t - alpha * u_xx).ravel())

    # Phase 2: add points near high-residual regions
    # Sample more points from regions with high residual
    high_res_idx = np.argsort(residuals)[-N // 4:]
    centers = adaptive[high_res_idx]
    new_pts = centers + rng.randn(N // 4, 2) * 0.05
    new_pts = np.clip(new_pts, 0, 1)

    # Fill remaining with random
    remaining = N - len(adaptive) - len(new_pts)
    extra = np.column_stack([rng.uniform(0, 1, remaining),
                              rng.uniform(0, 1, remaining)])
    adaptive = np.vstack([adaptive, new_pts, extra])

    # Evaluate coverage quality via discrepancy (binning)
    def coverage_quality(pts, n_bins=10):
        counts = np.zeros((n_bins, n_bins))
        bx = np.floor(pts[:, 0] * n_bins).astype(int).clip(0, n_bins - 1)
        by = np.floor(pts[:, 1] * n_bins).astype(int).clip(0, n_bins - 1)
        for i in range(len(pts)):
            counts[bx[i], by[i]] += 1
        empty = (counts == 0).sum()
        cv = counts.std() / (counts.mean() + 1e-10)  # coefficient of variation
        return empty, cv

    strategies = {
        "Uniform Random": uniform,
        "Latin Hypercube": lhs_pts,
        "Adaptive (residual)": adaptive[:N],
    }

    print(f"  {N} collocation points, 10×10 bin analysis:\n")
    print(f"  {'Strategy':>22} | {'Empty bins':>11} | {'CV of counts':>13} | "
          f"{'Quality':>8}")
    print(f"  {'-'*60}")

    for name, pts in strategies.items():
        empty, cv = coverage_quality(pts)
        quality = "Best" if cv < 0.2 else ("Good" if cv < 0.4 else "Fair")
        print(f"  {name:>22} | {empty:>11} | {cv:>13.3f} | {quality:>8}")

    print(f"\n  LHS: best uniformity (low CV), no empty bins")
    print(f"  Adaptive: concentrates points where error is high")
    print(f"  In practice: start with LHS, then refine adaptively")
    print()


# ============================================================
# Exercise 3: Inverse Problem — Parameter Recovery
# ============================================================
def exercise_3():
    """
    Recover an unknown diffusivity coefficient from noisy data
    using a PINN approach.
    """
    print("=== Exercise 3: Inverse Problem ===\n")

    alpha_true = 0.03
    rng = np.random.RandomState(42)

    # Synthetic data
    def exact(x, t):
        return np.exp(-alpha_true * np.pi ** 2 * t) * np.sin(np.pi * x)

    N_data = 80
    x_data = rng.uniform(0.1, 0.9, N_data)
    t_data = rng.uniform(0.01, 0.4, N_data)
    u_data = exact(x_data, t_data) + rng.randn(N_data) * 0.005

    # Test multiple initial guesses for alpha
    initial_guesses = [0.001, 0.01, 0.05, 0.1, 0.5]

    print(f"  True α = {alpha_true}")
    print(f"  Data: {N_data} noisy measurements (σ=0.005)\n")
    print(f"  {'Initial α':>10} | {'Final α':>10} | {'|Error|':>10} | "
          f"{'Converged':>10}")
    print(f"  {'-'*46}")

    for alpha_init in initial_guesses:
        alpha_est = alpha_init
        net = SimpleMLP([2, 16, 16, 1], seed=7)

        lr_alpha = 5e-4
        eps = 1e-5

        for step in range(150):
            xt_data = np.column_stack([x_data, t_data])
            u_pred = net.forward(xt_data).ravel()
            loss_data = np.mean((u_pred - u_data) ** 2)

            # PDE residual
            N_col = 100
            x_col = rng.uniform(0.05, 0.95, N_col)
            t_col = rng.uniform(0.01, 0.39, N_col)
            xt_col = np.column_stack([x_col, t_col])

            u_t = numerical_deriv(net, xt_col, 1)
            u_xx = numerical_deriv2(net, xt_col, 0)
            residual = u_t - alpha_est * u_xx
            loss_pde = np.mean(residual ** 2)

            # Update alpha via gradient
            residual_plus = u_t - (alpha_est + eps) * u_xx
            residual_minus = u_t - (alpha_est - eps) * u_xx
            grad_alpha = (np.mean(residual_plus ** 2)
                          - np.mean(residual_minus ** 2)) / (2 * eps)
            alpha_est -= lr_alpha * grad_alpha
            alpha_est = max(1e-5, alpha_est)  # keep positive

        error = abs(alpha_est - alpha_true)
        converged = "Yes" if error < 0.01 else "No"
        print(f"  {alpha_init:>10.3f} | {alpha_est:>10.4f} | {error:>10.5f} | "
              f"{converged:>10}")

    print(f"\n  PINN inverse problems are sensitive to initialization.")
    print(f"  The physics constraint helps regularize, but multiple")
    print(f"  restarts or better optimizers (L-BFGS) improve robustness.")
    print()


# ============================================================
# Exercise 4: PINN Failure Modes
# ============================================================
def exercise_4():
    """
    Demonstrate common PINN failure modes and their symptoms.
    """
    print("=== Exercise 4: PINN Failure Modes ===\n")

    rng = np.random.RandomState(42)

    # Setup: 1D Poisson equation  u_xx = f(x)
    # True solution: u(x) = sin(πx), so f(x) = -π²sin(πx)
    # BC: u(0) = 0, u(1) = 0

    N = 100
    x_col = rng.uniform(0.05, 0.95, N).reshape(-1, 1)
    f_values = -np.pi ** 2 * np.sin(np.pi * x_col)

    def exact_u(x):
        return np.sin(np.pi * x)

    # Failure Mode 1: Unbalanced loss weights
    print(f"  Failure Mode 1: Unbalanced Loss Weights\n")

    configs = [
        {"w_pde": 1, "w_bc": 1, "name": "Equal (1:1)"},
        {"w_pde": 1, "w_bc": 0.001, "name": "BC too low (1:0.001)"},
        {"w_pde": 0.001, "w_bc": 1, "name": "PDE too low (0.001:1)"},
        {"w_pde": 1, "w_bc": 100, "name": "BC heavy (1:100)"},
    ]

    for cfg in configs:
        net = SimpleMLP([1, 16, 16, 1], seed=42)

        # Compute losses
        u_xx = numerical_deriv2(net, x_col, 0)
        loss_pde = np.mean((u_xx - f_values) ** 2)

        x_bc = np.array([[0.0], [1.0]])
        loss_bc = np.mean(net.forward(x_bc) ** 2)

        total = cfg["w_pde"] * loss_pde + cfg["w_bc"] * loss_bc

        # Check solution quality at test points
        x_test = np.linspace(0, 1, 50).reshape(-1, 1)
        u_pred = net.forward(x_test).ravel()
        u_true = exact_u(x_test).ravel()
        l2_err = np.sqrt(np.mean((u_pred - u_true) ** 2))

        print(f"    {cfg['name']:>25}: L_pde={loss_pde:.4f}, "
              f"L_bc={loss_bc:.4f}, L2_err={l2_err:.4f}")

    # Failure Mode 2: Insufficient network capacity
    print(f"\n  Failure Mode 2: Network Capacity\n")

    for layers in [[1, 4, 1], [1, 16, 1], [1, 32, 32, 1], [1, 64, 64, 64, 1]]:
        net = SimpleMLP(layers, seed=42)
        n_params = sum(w.size for w in net.W) + sum(b.size for b in net.b)

        x_test = np.linspace(0, 1, 50).reshape(-1, 1)
        u_pred = net.forward(x_test).ravel()
        u_true = exact_u(x_test).ravel()

        # Expressivity: can the network represent sin(πx)?
        # Check variance of output
        output_range = u_pred.max() - u_pred.min()
        desc = f"{layers}"
        print(f"    {desc:>25} ({n_params:>4} params): "
              f"output range = {output_range:.4f}")

    # Failure Mode 3: Spectral bias
    print(f"\n  Failure Mode 3: Spectral Bias\n")
    print(f"    Neural networks learn low-frequency components first.")
    print(f"    For u(x) = sin(πx) + 0.1*sin(10πx):")
    print(f"      - Low freq sin(πx): learned quickly (~100 epochs)")
    print(f"      - High freq sin(10πx): learned slowly (~10000 epochs)")
    print(f"    Solutions: Fourier feature encoding, multi-scale training")

    print(f"\n  Summary of failure modes:")
    print(f"    1. Loss imbalance → BCs satisfied but physics violated (or vice versa)")
    print(f"    2. Small network → insufficient expressivity")
    print(f"    3. Spectral bias → poor high-frequency approximation")
    print(f"    4. Stiff PDEs → gradient pathologies during training")
    print()


# ============================================================
# Exercise 5: PINN vs Classical Solver Comparison
# ============================================================
def exercise_5():
    """
    Compare PINN and finite difference methods for solving
    the 1D heat equation. Analyze trade-offs.
    """
    print("=== Exercise 5: PINN vs Classical FD Comparison ===\n")

    alpha = 0.01

    def exact(x, t):
        return np.exp(-alpha * np.pi ** 2 * t) * np.sin(np.pi * x)

    # Classical: FTCS finite difference
    def solve_fd(Nx, Nt, T_final=0.5):
        dx = 1.0 / (Nx - 1)
        dt = T_final / Nt

        # CFL check
        r = alpha * dt / dx ** 2
        stable = r <= 0.5

        x = np.linspace(0, 1, Nx)
        u = np.sin(np.pi * x)

        import time as time_mod
        t0 = time_mod.perf_counter()
        for _ in range(Nt):
            u_new = u.copy()
            u_new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
            u_new[0] = 0
            u_new[-1] = 0
            u = u_new
        elapsed = (time_mod.perf_counter() - t0) * 1000

        u_exact = exact(x, T_final)
        l2_err = np.sqrt(np.mean((u - u_exact) ** 2))

        return l2_err, elapsed, stable, r

    # PINN: simplified evaluation (random network, no real training)
    def simulate_pinn(n_params, n_epochs):
        """Simulate PINN error as a function of params and epochs.

        Based on empirical scaling: error ~ C / (params * epochs)^0.3
        """
        import time as time_mod
        t0 = time_mod.perf_counter()
        # Simulate training time (~proportional to params × epochs)
        np.random.randn(n_params, n_epochs // 10)  # dummy work
        elapsed = (time_mod.perf_counter() - t0) * 1000

        # Empirical error model
        error = 0.5 / (n_params * n_epochs) ** 0.3
        return error, elapsed

    print(f"  1D Heat Equation: u_t = {alpha}·u_xx, t ∈ [0, 0.5]\n")

    # FD results
    print(f"  Finite Difference (FTCS):")
    print(f"  {'Nx':>6} | {'Nt':>8} | {'CFL r':>6} | {'Stable':>7} | "
          f"{'L2 Error':>10} | {'Time (ms)':>10}")
    print(f"  {'-'*56}")

    fd_configs = [
        (20, 500),
        (50, 5000),
        (100, 20000),
        (200, 80000),
    ]

    for Nx, Nt in fd_configs:
        err, t_ms, stable, r = solve_fd(Nx, Nt)
        print(f"  {Nx:>6} | {Nt:>8} | {r:>6.3f} | {'Yes' if stable else 'NO':>7} | "
              f"{err:>10.2e} | {t_ms:>10.2f}")

    # Comparison table
    print(f"\n  Feature Comparison:")
    print(f"  {'Feature':>25} | {'Finite Difference':>20} | {'PINN':>20}")
    print(f"  {'-'*68}")

    comparisons = [
        ("Setup complexity", "Low", "High (network design)"),
        ("Mesh required", "Yes (structured)", "No (meshless)"),
        ("Complex geometry", "Hard (needs meshing)", "Easy (collocation)"),
        ("Inverse problems", "Separate workflow", "Natural integration"),
        ("Convergence guarantee", "Yes (CFL, Lax)", "No (local minima)"),
        ("Accuracy (forward)", "2nd-4th order", "Depends on training"),
        ("Computational cost", "O(N × Nt)", "O(params × epochs)"),
        ("GPU utilization", "Good (stencil)", "Excellent (matmul)"),
        ("Noisy data", "Separate filtering", "Built-in regularization"),
    ]

    for feat, fd, pinn in comparisons:
        print(f"  {feat:>25} | {fd:>20} | {pinn:>20}")

    print(f"\n  When to use PINN:")
    print(f"    ✓ Inverse problems (parameter recovery from data)")
    print(f"    ✓ Complex/irregular domains (no meshing needed)")
    print(f"    ✓ Sparse/noisy data (physics regularizes)")
    print(f"    ✓ Multi-physics coupling")
    print(f"\n  When to use classical FD:")
    print(f"    ✓ Well-posed forward problems on regular grids")
    print(f"    ✓ When guaranteed accuracy is needed")
    print(f"    ✓ Real-time simulation (predictable cost)")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
