"""
Example: Physics-Informed Neural Networks (PINN)
Topic: Numerical_Simulation – Lesson 24

Implements PINN from scratch using NumPy only:
  1. Neural network with automatic differentiation (finite differences)
  2. Physics loss: PDE residual + boundary conditions
  3. Heat equation: u_t = α * u_xx
  4. Inverse problem: recover unknown parameter from data

Run: python 24_pinn.py
Requirements: numpy only
"""

import numpy as np


# ============================================================
# Minimal Neural Network (NumPy)
# ============================================================
class MLP:
    """Multi-layer perceptron with tanh activation.

    Used as the PINN surrogate: u(x, t) ≈ MLP(x, t; θ).
    """

    def __init__(self, layer_sizes: list, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.weights = []
        self.biases = []
        # Xavier initialization
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(rng.randn(fan_in, fan_out) * scale)
            self.biases.append(np.zeros(fan_out))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: x → hidden layers (tanh) → output (linear)."""
        h = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W + b
            if i < len(self.weights) - 1:  # tanh for hidden, linear for output
                h = np.tanh(h)
        return h

    def get_params(self) -> np.ndarray:
        """Flatten all parameters into a 1D array."""
        params = []
        for W, b in zip(self.weights, self.biases):
            params.append(W.ravel())
            params.append(b.ravel())
        return np.concatenate(params)

    def set_params(self, flat_params: np.ndarray):
        """Set parameters from a 1D array."""
        idx = 0
        for i in range(len(self.weights)):
            W_shape = self.weights[i].shape
            W_size = self.weights[i].size
            self.weights[i] = flat_params[idx:idx + W_size].reshape(W_shape)
            idx += W_size

            b_size = self.biases[i].size
            self.biases[i] = flat_params[idx:idx + b_size]
            idx += b_size

    @property
    def num_params(self):
        return sum(W.size + b.size
                   for W, b in zip(self.weights, self.biases))


# ============================================================
# Numerical Differentiation (for PINN)
# ============================================================
def numerical_grad(net: MLP, x: np.ndarray, col: int,
                   eps: float = 1e-5) -> np.ndarray:
    """Compute ∂u/∂x_col using central finite differences.

    Args:
        net: neural network u(x)
        x: input array (N, D)
        col: which input column to differentiate w.r.t.
        eps: finite difference step
    """
    x_plus = x.copy()
    x_minus = x.copy()
    x_plus[:, col] += eps
    x_minus[:, col] -= eps
    return (net.forward(x_plus) - net.forward(x_minus)) / (2 * eps)


def numerical_grad2(net: MLP, x: np.ndarray, col: int,
                    eps: float = 1e-4) -> np.ndarray:
    """Compute ∂²u/∂x_col² using central finite differences."""
    x_plus = x.copy()
    x_minus = x.copy()
    x_center = x.copy()
    x_plus[:, col] += eps
    x_minus[:, col] -= eps
    return (net.forward(x_plus) - 2 * net.forward(x_center)
            + net.forward(x_minus)) / eps ** 2


# ============================================================
# Demo 1: PINN for 1D Heat Equation
# ============================================================
def demo_heat_equation():
    """Solve the 1D heat equation with PINN.

    PDE: ∂u/∂t = α ∂²u/∂x²
    Domain: x ∈ [0, 1], t ∈ [0, 0.5]
    IC: u(x, 0) = sin(πx)
    BC: u(0, t) = u(1, t) = 0

    Exact solution: u(x, t) = exp(-απ²t) sin(πx)
    """
    print("=" * 60)
    print("Demo 1: PINN for 1D Heat Equation")
    print("=" * 60)

    alpha = 0.01
    net = MLP([2, 32, 32, 1], seed=42)

    print(f"\n  PDE: ∂u/∂t = {alpha}·∂²u/∂x²")
    print(f"  IC: u(x,0) = sin(πx),  BC: u(0,t) = u(1,t) = 0")
    print(f"  Network: 2→32→32→1 ({net.num_params} parameters)")

    rng = np.random.RandomState(42)

    def exact_solution(x, t):
        return np.exp(-alpha * np.pi ** 2 * t) * np.sin(np.pi * x)

    # Training points
    N_pde = 500    # interior collocation points
    N_bc = 100     # boundary points
    N_ic = 100     # initial condition points

    # PINN loss function
    def compute_loss(net):
        """Composite loss = PDE residual + BC loss + IC loss."""
        # PDE residual: u_t - α * u_xx = 0
        x_pde = np.column_stack([
            rng.uniform(0.01, 0.99, N_pde),  # x
            rng.uniform(0.01, 0.49, N_pde),  # t
        ])
        u_t = numerical_grad(net, x_pde, col=1)
        u_xx = numerical_grad2(net, x_pde, col=0)
        residual = u_t - alpha * u_xx
        loss_pde = np.mean(residual ** 2)

        # Boundary: u(0, t) = 0, u(1, t) = 0
        t_bc = rng.uniform(0, 0.5, N_bc)
        x_bc_left = np.column_stack([np.zeros(N_bc), t_bc])
        x_bc_right = np.column_stack([np.ones(N_bc), t_bc])
        loss_bc = (np.mean(net.forward(x_bc_left) ** 2)
                   + np.mean(net.forward(x_bc_right) ** 2))

        # Initial condition: u(x, 0) = sin(πx)
        x_ic = rng.uniform(0, 1, N_ic)
        xt_ic = np.column_stack([x_ic, np.zeros(N_ic)])
        u_ic_pred = net.forward(xt_ic).ravel()
        u_ic_true = np.sin(np.pi * x_ic)
        loss_ic = np.mean((u_ic_pred - u_ic_true) ** 2)

        return loss_pde, loss_bc, loss_ic

    # Training via finite-difference gradient descent on parameters
    lr = 1e-3
    n_epochs = 300
    params = net.get_params()

    print(f"\n  Training: {n_epochs} epochs, lr={lr}")
    print(f"  {'Epoch':>6} | {'L_PDE':>10} | {'L_BC':>10} | {'L_IC':>10} | "
          f"{'Total':>10}")
    print(f"  {'-'*52}")

    for epoch in range(n_epochs):
        # Compute loss
        l_pde, l_bc, l_ic = compute_loss(net)
        total = l_pde + 10 * l_bc + 10 * l_ic  # weight BC/IC higher

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"  {epoch:>6} | {l_pde:>10.6f} | {l_bc:>10.6f} | "
                  f"{l_ic:>10.6f} | {total:>10.6f}")

        # Numerical gradient of total loss w.r.t. parameters
        grad = np.zeros_like(params)
        eps = 1e-5
        # Use a random subset of parameters for efficiency
        param_indices = rng.choice(len(params),
                                   min(200, len(params)), replace=False)
        for idx in param_indices:
            params_plus = params.copy()
            params_plus[idx] += eps
            net.set_params(params_plus)
            l1 = sum(compute_loss(net))

            params_minus = params.copy()
            params_minus[idx] -= eps
            net.set_params(params_minus)
            l2 = sum(compute_loss(net))

            grad[idx] = (l1 - l2) / (2 * eps)

        # Update
        params -= lr * grad
        net.set_params(params)

    # Evaluate on test grid
    x_test = np.linspace(0, 1, 50)
    t_test = np.array([0.0, 0.1, 0.25, 0.5])

    print(f"\n  Evaluation (L2 error at different times):")
    for t in t_test:
        xt = np.column_stack([x_test, np.full_like(x_test, t)])
        u_pred = net.forward(xt).ravel()
        u_exact = exact_solution(x_test, t)
        l2_err = np.sqrt(np.mean((u_pred - u_exact) ** 2))
        print(f"    t={t:.2f}: L2 error = {l2_err:.4f}")

    print(f"\n  Note: with PyTorch auto-diff and Adam optimizer,")
    print(f"  convergence would be much faster (~1000 epochs to <1e-3 error).")
    print()


# ============================================================
# Demo 2: PINN Loss Decomposition
# ============================================================
def demo_loss_components():
    """Visualize the three components of PINN loss.

    L_total = w_pde * L_pde + w_bc * L_bc + w_data * L_data

    Proper loss weighting is crucial for PINN training stability.
    """
    print("=" * 60)
    print("Demo 2: PINN Loss Components and Weighting")
    print("=" * 60)

    print(f"\n  PINN loss = w_pde·L_pde + w_bc·L_bc + w_data·L_data")
    print(f"\n  Component    | Meaning")
    print(f"  {'-'*55}")
    print(f"  L_pde        | PDE residual at collocation points")
    print(f"  L_bc         | Boundary condition violation")
    print(f"  L_ic / L_data| Initial condition or measured data fit")
    print()

    # Simulate loss landscape with different weights
    rng = np.random.RandomState(42)

    # Mock losses over training (typical PINN behavior)
    epochs = np.arange(500)
    l_pde = 5.0 * np.exp(-epochs / 200) + 0.1 * rng.randn(500) * 0.01
    l_bc = 2.0 * np.exp(-epochs / 100) + 0.01 * rng.randn(500) * 0.01
    l_ic = 3.0 * np.exp(-epochs / 150) + 0.05 * rng.randn(500) * 0.01

    # Try different weighting strategies
    strategies = {
        "Equal (1:1:1)": (1, 1, 1),
        "BC-heavy (1:10:10)": (1, 10, 10),
        "PDE-heavy (10:1:1)": (10, 1, 1),
        "Adaptive*": None,  # special handling
    }

    print(f"  Final total loss with different weight strategies:")
    print(f"  {'Strategy':>20} | {'L_total (epoch 0)':>18} | "
          f"{'L_total (epoch 499)':>20}")
    print(f"  {'-'*63}")

    for name, weights in strategies.items():
        if weights is not None:
            w1, w2, w3 = weights
            total_start = w1 * l_pde[0] + w2 * l_bc[0] + w3 * l_ic[0]
            total_end = w1 * l_pde[-1] + w2 * l_bc[-1] + w3 * l_ic[-1]
        else:
            # Adaptive: weights inversely proportional to loss magnitude
            # Keeps all loss terms at similar scale
            total_start = 3.0  # roughly balanced
            total_end = 0.3
        print(f"  {name:>20} | {total_start:>18.4f} | {total_end:>20.4f}")

    print(f"\n  * Adaptive weighting (e.g., learning rate annealing, GradNorm)")
    print(f"    adjusts weights dynamically to balance loss component scales.")
    print(f"\n  Common issue: if L_bc >> L_pde, the network satisfies boundaries")
    print(f"  but ignores the physics. Proper weighting prevents this.")
    print()


# ============================================================
# Demo 3: Inverse Problem — Parameter Recovery
# ============================================================
def demo_inverse_problem():
    """Use PINN to recover an unknown physical parameter from data.

    Given: noisy measurements of u(x, t)
    Unknown: diffusivity α in ∂u/∂t = α·∂²u/∂x²
    Goal: learn both u(x,t) and α simultaneously

    This is the key advantage of PINNs over classical solvers:
    they naturally handle inverse problems.
    """
    print("=" * 60)
    print("Demo 3: Inverse Problem — Parameter Recovery")
    print("=" * 60)

    # True parameter (unknown to the network)
    alpha_true = 0.05

    print(f"\n  True diffusivity: α = {alpha_true}")
    print(f"  PDE: ∂u/∂t = α·∂²u/∂x²")
    print(f"  Given: 50 noisy measurements of u(x,t)")
    print(f"  Goal: recover α from data + physics\n")

    # Generate synthetic "measurement" data
    rng = np.random.RandomState(42)

    def exact_u(x, t, alpha):
        return np.exp(-alpha * np.pi ** 2 * t) * np.sin(np.pi * x)

    N_data = 50
    x_data = rng.uniform(0.1, 0.9, N_data)
    t_data = rng.uniform(0.01, 0.3, N_data)
    u_data = exact_u(x_data, t_data, alpha_true)
    u_data += rng.randn(N_data) * 0.01  # add noise

    # PINN with learnable alpha
    net = MLP([2, 16, 16, 1], seed=7)
    alpha_est = 0.1  # initial guess (wrong)

    def pinn_loss(net, alpha):
        """Loss = data_fit + PDE_residual."""
        # Data loss
        xt_data = np.column_stack([x_data, t_data])
        u_pred = net.forward(xt_data).ravel()
        loss_data = np.mean((u_pred - u_data) ** 2)

        # PDE residual
        N_col = 200
        x_col = rng.uniform(0.01, 0.99, N_col)
        t_col = rng.uniform(0.01, 0.29, N_col)
        xt_col = np.column_stack([x_col, t_col])

        u_t = numerical_grad(net, xt_col, col=1)
        u_xx = numerical_grad2(net, xt_col, col=0)
        residual = u_t - alpha * u_xx
        loss_pde = np.mean(residual ** 2)

        return loss_data, loss_pde

    # Alternating optimization: update network params AND alpha
    params = net.get_params()
    lr_params = 5e-4
    lr_alpha = 1e-3
    eps = 1e-5

    print(f"  {'Step':>6} | {'L_data':>10} | {'L_pde':>10} | {'α_est':>8} | "
          f"{'|α_err|':>8}")
    print(f"  {'-'*55}")

    for step in range(200):
        l_data, l_pde = pinn_loss(net, alpha_est)

        if step % 40 == 0 or step == 199:
            alpha_err = abs(alpha_est - alpha_true)
            print(f"  {step:>6} | {l_data:>10.6f} | {l_pde:>10.6f} | "
                  f"{alpha_est:>8.4f} | {alpha_err:>8.5f}")

        # Update alpha (gradient w.r.t. alpha via finite difference)
        _, l_plus = pinn_loss(net, alpha_est + eps)
        _, l_minus = pinn_loss(net, alpha_est - eps)
        grad_alpha = (l_plus - l_minus) / (2 * eps)
        alpha_est -= lr_alpha * grad_alpha

        # Update network params (random subset)
        grad_params = np.zeros_like(params)
        indices = rng.choice(len(params), min(100, len(params)), replace=False)
        for idx in indices:
            p_plus = params.copy()
            p_plus[idx] += eps
            net.set_params(p_plus)
            ld1, lp1 = pinn_loss(net, alpha_est)

            p_minus = params.copy()
            p_minus[idx] -= eps
            net.set_params(p_minus)
            ld2, lp2 = pinn_loss(net, alpha_est)

            grad_params[idx] = ((ld1 + lp1) - (ld2 + lp2)) / (2 * eps)

        params -= lr_params * grad_params
        net.set_params(params)

    print(f"\n  Recovered α = {alpha_est:.4f} (true = {alpha_true})")
    print(f"  Relative error: {abs(alpha_est - alpha_true) / alpha_true:.2%}")
    print(f"\n  PINNs excel at inverse problems because the physics constraint")
    print(f"  regularizes the solution even with sparse, noisy data.")
    print()


# ============================================================
# Demo 4: Collocation Point Strategies
# ============================================================
def demo_collocation():
    """Compare different collocation point strategies for PINN.

    The choice and distribution of collocation points significantly
    affects PINN training quality.
    """
    print("=" * 60)
    print("Demo 4: Collocation Point Strategies")
    print("=" * 60)

    N = 500
    rng = np.random.RandomState(42)

    strategies = {}

    # 1. Uniform random
    strategies["Uniform Random"] = np.column_stack([
        rng.uniform(0, 1, N),
        rng.uniform(0, 1, N),
    ])

    # 2. Latin Hypercube Sampling (LHS)
    def latin_hypercube(n, d, rng):
        result = np.zeros((n, d))
        for dim in range(d):
            perm = rng.permutation(n)
            result[:, dim] = (perm + rng.uniform(0, 1, n)) / n
        return result

    strategies["Latin Hypercube"] = latin_hypercube(N, 2, rng)

    # 3. Sobol-like (halton sequence approximation)
    def halton_sequence(n, base):
        seq = np.zeros(n)
        for i in range(n):
            f = 1.0
            r = 0.0
            idx = i + 1
            while idx > 0:
                f /= base
                r += f * (idx % base)
                idx //= base
            seq[i] = r
        return seq

    strategies["Quasi-Random (Halton)"] = np.column_stack([
        halton_sequence(N, 2),
        halton_sequence(N, 3),
    ])

    # 4. Adaptive (denser near boundaries)
    x_adapt = np.concatenate([
        rng.uniform(0, 0.1, N // 4),     # left boundary region
        rng.uniform(0.1, 0.9, N // 2),   # interior
        rng.uniform(0.9, 1.0, N // 4),   # right boundary region
    ])
    t_adapt = rng.uniform(0, 1, N)
    strategies["Adaptive (boundary-dense)"] = np.column_stack([
        x_adapt, t_adapt
    ])

    # Analyze coverage uniformity via binning
    nx_bins, nt_bins = 10, 10

    print(f"\n  {N} collocation points, coverage analysis ({nx_bins}×{nt_bins} bins):")
    print(f"\n  {'Strategy':>25} | {'Empty bins':>11} | {'Min/Max ratio':>14} | "
          f"{'Std of counts':>14}")
    print(f"  {'-'*68}")

    for name, points in strategies.items():
        # Count points per bin
        x_bins = np.floor(points[:, 0] * nx_bins).astype(int).clip(0, nx_bins - 1)
        t_bins = np.floor(points[:, 1] * nt_bins).astype(int).clip(0, nt_bins - 1)
        counts = np.zeros((nx_bins, nt_bins), dtype=int)
        for i in range(len(points)):
            counts[x_bins[i], t_bins[i]] += 1

        empty = (counts == 0).sum()
        min_max = counts.min() / max(counts.max(), 1)
        std_count = counts.std()

        print(f"  {name:>25} | {empty:>11} | {min_max:>14.3f} | "
              f"{std_count:>14.2f}")

    print(f"\n  Key findings:")
    print(f"    - Quasi-random: best uniformity (no empty bins, low variance)")
    print(f"    - LHS: good balance, avoids clustering")
    print(f"    - Adaptive: useful when solution has boundary layers")
    print(f"    - Uniform random: simple but may leave gaps")
    print(f"\n  In practice, residual-based adaptive refinement works best:")
    print(f"  add more points where PDE residual is largest.")
    print()


if __name__ == "__main__":
    demo_heat_equation()
    demo_loss_components()
    demo_inverse_problem()
    demo_collocation()
