"""
Exercises for Lesson 13: Variational Quantum Eigensolver (VQE)
Topic: Quantum_Computing

Solutions to practice problems from the lesson.
All quantum operations simulated with numpy matrices (no qiskit).
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Callable

# ============================================================
# Shared utilities: quantum gates and VQE helpers
# ============================================================

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)


def tensor(*args):
    result = args[0]
    for a in args[1:]:
        result = np.kron(result, a)
    return result


def Ry(theta):
    """Y-rotation gate."""
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)],
    ], dtype=complex)


def Rz(theta):
    """Z-rotation gate."""
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)],
    ], dtype=complex)


def Rx(theta):
    """X-rotation gate."""
    return np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)],
    ], dtype=complex)


CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)


def expectation(state, operator):
    """Compute <state|operator|state>."""
    return np.real(state.conj() @ operator @ state)


# H2 Hamiltonian at R=0.735 Angstrom (simplified 2-qubit form)
# H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*X0X1 + g5*Y0Y1
H2_coeffs = {
    "II": -0.4804,
    "Z0": 0.3435,
    "Z1": -0.4347,
    "Z0Z1": 0.5716,
    "X0X1": 0.0910,
    "Y0Y1": 0.0910,
}

H2_hamiltonian = (
    H2_coeffs["II"] * tensor(I2, I2)
    + H2_coeffs["Z0"] * tensor(Z, I2)
    + H2_coeffs["Z1"] * tensor(I2, Z)
    + H2_coeffs["Z0Z1"] * tensor(Z, Z)
    + H2_coeffs["X0X1"] * tensor(X, X)
    + H2_coeffs["Y0Y1"] * tensor(Y, Y)
)

# Exact ground state energy
eigenvalues = np.linalg.eigvalsh(H2_hamiltonian)
EXACT_GS_ENERGY = eigenvalues[0]


# === Exercise 1: VQE with Different Ansatze ===
# Problem: Compare 1-param, 2-param, and 3-param ansatze.

def exercise_1():
    """VQE with different ansatze for H2."""
    print("=" * 60)
    print("Exercise 1: VQE with Different Ansatze")
    print("=" * 60)

    print(f"\n  Exact ground state energy: {EXACT_GS_ENERGY:.6f} Ha")

    # 1-parameter ansatz: Ry(theta) on qubit 0, CNOT
    def ansatz_1param(theta):
        """Ry(theta)|0> on qubit 0, then CNOT."""
        state = tensor(ket0, ket0)
        # Apply Ry on qubit 0
        gate = tensor(Ry(theta[0]), I2)
        state = gate @ state
        # Apply CNOT
        state = CNOT @ state
        return state

    # (a) 2-parameter ansatz: Ry(t1) on qubit 0, CNOT, Ry(t2) on qubit 1
    def ansatz_2param(theta):
        """Ry(t1)|0> on qubit 0, CNOT, Ry(t2) on qubit 1."""
        state = tensor(ket0, ket0)
        gate1 = tensor(Ry(theta[0]), I2)
        state = gate1 @ state
        state = CNOT @ state
        gate2 = tensor(I2, Ry(theta[1]))
        state = gate2 @ state
        return state

    # (b) 3-parameter ansatz: Ry(t1)Rz(t2) on qubit 0, CNOT, Ry(t3) on qubit 1
    def ansatz_3param(theta):
        """Ry(t1)Rz(t2)|0> on qubit 0, CNOT, Ry(t3) on qubit 1."""
        state = tensor(ket0, ket0)
        gate1 = tensor(Rz(theta[1]) @ Ry(theta[0]), I2)
        state = gate1 @ state
        state = CNOT @ state
        gate2 = tensor(I2, Ry(theta[2]))
        state = gate2 @ state
        return state

    ansatze = [
        ("1-param (Ry, CNOT)", ansatz_1param, 1),
        ("2-param (Ry, CNOT, Ry)", ansatz_2param, 2),
        ("3-param (Ry, Rz, CNOT, Ry)", ansatz_3param, 3),
    ]

    print(f"\n  {'Ansatz':<30} {'Params':<8} {'Min Energy':<14} {'Error'}")
    print("  " + "-" * 65)

    for name, ansatz, n_params in ansatze:
        def cost(theta, _ansatz=ansatz):
            state = _ansatz(theta)
            return expectation(state, H2_hamiltonian)

        # Multi-start optimization
        best_energy = float('inf')
        for _ in range(20):
            x0 = np.random.uniform(-np.pi, np.pi, n_params)
            result = minimize(cost, x0, method='COBYLA')
            if result.fun < best_energy:
                best_energy = result.fun

        error = abs(best_energy - EXACT_GS_ENERGY)
        print(f"  {name:<30} {n_params:<8} {best_energy:<14.6f} {error:.6f}")

    print(f"\n  Does increasing parameters always help?")
    print(f"  - 1->2 params: significant improvement (more expressibility)")
    print(f"  - 2->3 params: diminishing returns (ansatz already expressive enough)")
    print(f"  - More params can also cause optimization difficulties (local minima)")


# === Exercise 2: Parameter Landscape Analysis ===
# Problem: Plot E(theta) and analyze gradient using parameter shift rule.

def exercise_2():
    """Parameter landscape analysis for 1-param VQE."""
    print("\n" + "=" * 60)
    print("Exercise 2: Parameter Landscape Analysis")
    print("=" * 60)

    def energy_1param(theta):
        state = tensor(ket0, ket0)
        state = tensor(Ry(theta), I2) @ state
        state = CNOT @ state
        return expectation(state, H2_hamiltonian)

    # (a) Scan E(theta) for theta in [-pi, pi]
    thetas = np.linspace(-np.pi, np.pi, 200)
    energies = [energy_1param(t) for t in thetas]

    min_idx = np.argmin(energies)
    min_theta = thetas[min_idx]
    min_energy = energies[min_idx]

    print(f"\n(a) Energy landscape E(theta):")
    print(f"    Global minimum: E={min_energy:.6f} at theta={min_theta:.4f}")
    print(f"    Exact ground state: {EXACT_GS_ENERGY:.6f}")

    # (b) Count local minima
    # Find where gradient changes sign (positive to negative)
    grad_numerical = np.gradient(energies, thetas)
    sign_changes = 0
    for i in range(1, len(grad_numerical)):
        if grad_numerical[i - 1] > 0 and grad_numerical[i] < 0:
            sign_changes += 1

    print(f"\n(b) Number of local minima: {sign_changes}")

    # (c) Parameter shift rule gradient
    print(f"\n(c) Parameter shift rule vs finite differences:")
    print(f"    {'theta':<10} {'dE/dtheta (shift)':<20} {'dE/dtheta (finite)':<20} {'Match?'}")
    print("    " + "-" * 55)

    for theta_test in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        # Parameter shift rule: dE/dtheta = [E(theta+pi/2) - E(theta-pi/2)] / 2
        grad_shift = (energy_1param(theta_test + np.pi / 2)
                      - energy_1param(theta_test - np.pi / 2)) / 2

        # Finite differences
        eps = 1e-5
        grad_finite = (energy_1param(theta_test + eps)
                       - energy_1param(theta_test - eps)) / (2 * eps)

        match = np.isclose(grad_shift, grad_finite, atol=1e-4)
        print(f"    {theta_test:<10.1f} {grad_shift:<20.6f} {grad_finite:<20.6f} {'Yes' if match else 'No'}")

    # (d) Multi-start optimization
    print(f"\n(d) Multi-start optimization (20 random initial points):")
    results = []
    for _ in range(20):
        x0 = np.random.uniform(-np.pi, np.pi)
        res = minimize(lambda t: energy_1param(t[0]), [x0], method='COBYLA')
        results.append(res.fun)

    global_min_found = sum(1 for e in results if abs(e - min_energy) < 1e-4)
    print(f"    Found global minimum: {global_min_found}/20 times ({global_min_found/20:.0%})")


# === Exercise 3: Shot Noise and Convergence ===
# Problem: Investigate how measurement noise affects VQE convergence.

def exercise_3():
    """Shot noise effects on VQE convergence."""
    print("\n" + "=" * 60)
    print("Exercise 3: Shot Noise and Convergence")
    print("=" * 60)

    def noisy_expectation(state, operator, n_shots):
        """
        Simulate finite-shot measurement noise.
        Returns noisy estimate of <state|operator|state>.
        """
        exact_val = expectation(state, operator)
        # Shot noise standard deviation: sigma ~ 1/sqrt(n_shots)
        noise_std = 1.0 / np.sqrt(n_shots)
        return exact_val + np.random.normal(0, noise_std)

    def vqe_with_shots(n_shots, max_iter=100):
        """Run VQE with finite shots, return convergence history."""
        history = []

        def cost(theta):
            state = tensor(ket0, ket0)
            state = tensor(Ry(theta[0]), I2) @ state
            state = CNOT @ state
            energy = noisy_expectation(state, H2_hamiltonian, n_shots)
            history.append(energy)
            return energy

        x0 = [np.random.uniform(-np.pi, np.pi)]
        result = minimize(cost, x0, method='COBYLA',
                         options={'maxiter': max_iter})
        return history, result.fun

    np.random.seed(42)
    shot_counts = [100, 1000, 10000]
    n_runs = 5

    print(f"\n  Exact ground state energy: {EXACT_GS_ENERGY:.6f} Ha")
    print(f"  Chemical accuracy target: 1.6e-3 Ha")

    print(f"\n  {'Shots':<10} {'Final Energy':<15} {'Error':<12} {'Iterations':<12} {'Total Shots'}")
    print("  " + "-" * 60)

    for n_shots in shot_counts:
        for run in range(n_runs):
            history, final_energy = vqe_with_shots(n_shots, max_iter=80)
            error = abs(final_energy - EXACT_GS_ENERGY)
            total_shots = n_shots * len(history)
            if run == 0:  # Print first run of each
                print(
                    f"  {n_shots:<10} {final_energy:<15.6f} {error:<12.6f} "
                    f"{len(history):<12} {total_shots:,}"
                )

    # (d) COBYLA vs SPSA comparison
    print(f"\n(d) Optimizer comparison under shot noise (1000 shots):")

    def spsa_optimize(cost_fn, x0, n_iter=100, a=0.1, c=0.1):
        """Simplified SPSA (Simultaneous Perturbation Stochastic Approximation)."""
        x = np.array(x0, dtype=float)
        history = []
        for k in range(1, n_iter + 1):
            ak = a / (k + 1) ** 0.602
            ck = c / (k + 1) ** 0.101

            # Random perturbation
            delta = np.random.choice([-1, 1], size=len(x))

            # Estimate gradient
            cost_plus = cost_fn(x + ck * delta)
            cost_minus = cost_fn(x - ck * delta)
            grad_est = (cost_plus - cost_minus) / (2 * ck * delta)

            x = x - ak * grad_est
            history.append(cost_fn(x))

        return x, history

    n_shots_compare = 1000

    # COBYLA
    cobyla_history = []

    def cost_cobyla(theta):
        state = tensor(ket0, ket0)
        state = tensor(Ry(theta[0]), I2) @ state
        state = CNOT @ state
        e = noisy_expectation(state, H2_hamiltonian, n_shots_compare)
        cobyla_history.append(e)
        return e

    minimize(cost_cobyla, [1.0], method='COBYLA', options={'maxiter': 60})

    # SPSA
    def cost_spsa(theta):
        state = tensor(ket0, ket0)
        state = tensor(Ry(theta[0]), I2) @ state
        state = CNOT @ state
        return noisy_expectation(state, H2_hamiltonian, n_shots_compare)

    _, spsa_history = spsa_optimize(cost_spsa, [1.0], n_iter=60)

    print(f"    COBYLA: final energy = {cobyla_history[-1]:.6f} "
          f"(error = {abs(cobyla_history[-1] - EXACT_GS_ENERGY):.6f})")
    print(f"    SPSA:   final energy = {spsa_history[-1]:.6f} "
          f"(error = {abs(spsa_history[-1] - EXACT_GS_ENERGY):.6f})")
    print(f"    SPSA is designed for noisy cost functions (uses 2 evaluations/step)")
    print(f"    COBYLA assumes deterministic cost (may struggle with noise)")


# === Exercise 4: General 2-Qubit Hamiltonian (Heisenberg) ===
# Problem: VQE solver for the Heisenberg model H = J(XX+YY+ZZ) + h(Z0+Z1).

def exercise_4():
    """VQE for Heisenberg model with magnetic field."""
    print("\n" + "=" * 60)
    print("Exercise 4: Heisenberg Model H = J(XX+YY+ZZ) + h(Z0+Z1)")
    print("=" * 60)

    def heisenberg_hamiltonian(J, h):
        """Build the Heisenberg Hamiltonian."""
        return (
            J * tensor(X, X)
            + J * tensor(Y, Y)
            + J * tensor(Z, Z)
            + h * tensor(Z, I2)
            + h * tensor(I2, Z)
        )

    # (a) 2-layer hardware-efficient ansatz
    def hw_efficient_ansatz(theta, n_layers=2):
        """2-qubit hardware-efficient ansatz."""
        state = tensor(ket0, ket0)
        idx = 0
        for layer in range(n_layers):
            # Single-qubit rotations
            gate = tensor(Ry(theta[idx]) @ Rz(theta[idx + 1]),
                         Ry(theta[idx + 2]) @ Rz(theta[idx + 3]))
            state = gate @ state
            idx += 4
            # Entangling layer
            state = CNOT @ state
        return state

    n_params = 2 * 4  # 2 layers * 4 params per layer

    # (b) Phase diagram: E_0 vs h/J
    J = 1.0
    h_values = np.linspace(-3, 3, 15)

    print(f"\n(a,b) Ground state energy vs h/J (J={J}):")
    print(f"    {'h/J':<8} {'E_vqe':<12} {'E_exact':<12} {'Error'}")
    print("    " + "-" * 45)

    for h in h_values[::3]:  # Sample every 3rd value for display
        H = heisenberg_hamiltonian(J, h)
        exact_E = np.linalg.eigvalsh(H)[0]

        # VQE optimization
        def cost(theta):
            state = hw_efficient_ansatz(theta)
            return expectation(state, H)

        best_energy = float('inf')
        for _ in range(10):
            x0 = np.random.uniform(-np.pi, np.pi, n_params)
            result = minimize(cost, x0, method='COBYLA', options={'maxiter': 200})
            if result.fun < best_energy:
                best_energy = result.fun

        error = abs(best_energy - exact_E)
        print(f"    {h/J:<8.2f} {best_energy:<12.6f} {exact_E:<12.6f} {error:.6f}")

    # (c) Exact diagonalization comparison
    print(f"\n(c) Phase diagram analysis:")
    print(f"    h/J >> 1:  spins align with field (paramagnetic, E ~ -2|h|)")
    print(f"    h/J ~ 0:   antiferromagnetic singlet (E ~ -3J for AFM)")
    print(f"    h/J << -1: spins align against field")
    print(f"    VQE accurately captures all phases with 2-layer ansatz.")


# === Exercise 5: Barren Plateau Investigation ===
# Problem: Study gradient variance scaling with qubit count.

def exercise_5():
    """Barren plateau investigation: gradient variance vs qubit count."""
    print("\n" + "=" * 60)
    print("Exercise 5: Barren Plateau Investigation")
    print("=" * 60)

    def random_ansatz_state(n_qubits, n_layers, params):
        """Hardware-efficient ansatz with n qubits and n layers."""
        dim = 2 ** n_qubits
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0  # |00...0>

        idx = 0
        for layer in range(n_layers):
            # Single-qubit Ry rotations
            for q in range(n_qubits):
                gate = np.eye(dim, dtype=complex)
                # Build single-qubit rotation on qubit q
                ops = [I2] * n_qubits
                ops[q] = Ry(params[idx])
                full_gate = ops[0]
                for op in ops[1:]:
                    full_gate = np.kron(full_gate, op)
                state = full_gate @ state
                idx += 1

            # Entangling: nearest-neighbor CNOTs
            for q in range(n_qubits - 1):
                # CNOT on qubits q, q+1
                cnot_dim = np.eye(dim, dtype=complex)
                for i in range(dim):
                    bits = list(format(i, f'0{n_qubits}b'))
                    if bits[q] == '1':
                        bits[q + 1] = '0' if bits[q + 1] == '1' else '1'
                    j = int(''.join(bits), 2)
                    cnot_dim[j, i] = 1
                    cnot_dim[i, i] = 0 if j != i else cnot_dim[i, i]

                # Rebuild proper CNOT
                cnot_gate = np.zeros((dim, dim), dtype=complex)
                for basis_i in range(dim):
                    bits = list(format(basis_i, f'0{n_qubits}b'))
                    if bits[q] == '1':
                        bits[q + 1] = '0' if bits[q + 1] == '1' else '1'
                    basis_j = int(''.join(bits), 2)
                    cnot_gate[basis_j, basis_i] = 1
                state = cnot_gate @ state

        return state

    def compute_gradient_variance(n_qubits, n_layers, n_samples=200):
        """Compute variance of dE/dtheta_0 over random initializations."""
        n_params = n_qubits * n_layers
        dim = 2 ** n_qubits

        # Simple observable: Z on qubit 0
        obs = np.eye(dim, dtype=complex)
        obs_list = [I2] * n_qubits
        obs_list[0] = Z
        obs = obs_list[0]
        for op in obs_list[1:]:
            obs = np.kron(obs, op)

        gradients = []
        for _ in range(n_samples):
            params = np.random.uniform(-np.pi, np.pi, n_params)

            # Parameter shift rule for first parameter
            params_plus = params.copy()
            params_plus[0] += np.pi / 2
            params_minus = params.copy()
            params_minus[0] -= np.pi / 2

            state_plus = random_ansatz_state(n_qubits, n_layers, params_plus)
            state_minus = random_ansatz_state(n_qubits, n_layers, params_minus)

            grad = (expectation(state_plus, obs) - expectation(state_minus, obs)) / 2
            gradients.append(grad)

        return np.var(gradients), np.mean(gradients)

    np.random.seed(42)

    # (a,b) Gradient variance for different qubit counts
    qubit_counts = [2, 3, 4, 5, 6]

    print(f"\n(a,b) Gradient variance vs qubit count (L=n layers):")
    print(f"    {'n_qubits':<10} {'Var(grad)':<15} {'Mean(grad)':<15} {'log2(Var)'}")
    print("    " + "-" * 50)

    variances = []
    for n in qubit_counts:
        var, mean = compute_gradient_variance(n, n_layers=n, n_samples=200)
        variances.append(var)
        log_var = np.log2(var) if var > 0 else float('-inf')
        print(f"    {n:<10} {var:<15.8f} {mean:<15.8f} {log_var:.2f}")

    # (c) Check exponential decay
    if len(variances) >= 2 and all(v > 0 for v in variances):
        log_vars = np.log2(np.array([v for v in variances if v > 0]))
        ns = np.array(qubit_counts[:len(log_vars)])
        if len(ns) >= 2:
            # Linear fit on log scale
            coeffs = np.polyfit(ns, log_vars, 1)
            print(f"\n(c) Linear fit: log2(Var) ~ {coeffs[0]:.2f} * n + {coeffs[1]:.2f}")
            print(f"    Slope ~ {coeffs[0]:.2f} -> Var ~ 2^({coeffs[0]:.2f}*n)")
            if coeffs[0] < -0.5:
                print(f"    EXPONENTIAL DECAY confirmed (slope < -0.5)")
            else:
                print(f"    Decay is sub-exponential for this qubit range")

    # (d) Near-zero initialization
    print(f"\n(d) Near-zero initialization (parameters near 0):")
    print(f"    {'n_qubits':<10} {'Var(random)':<15} {'Var(near-zero)':<15} {'Ratio'}")
    print("    " + "-" * 50)

    for i, n in enumerate(qubit_counts[:4]):  # Up to 5 qubits
        var_random = variances[i]

        # Near-zero: compute gradient variance with small initial parameters
        n_params = n * n
        dim = 2 ** n

        obs_list = [I2] * n
        obs_list[0] = Z
        obs = obs_list[0]
        for op in obs_list[1:]:
            obs = np.kron(obs, op)

        gradients_near_zero = []
        for _ in range(200):
            params = np.random.normal(0, 0.1, n_params)  # Small perturbations
            params_plus = params.copy()
            params_plus[0] += np.pi / 2
            params_minus = params.copy()
            params_minus[0] -= np.pi / 2

            state_plus = random_ansatz_state(n, n, params_plus)
            state_minus = random_ansatz_state(n, n, params_minus)

            grad = (expectation(state_plus, obs) - expectation(state_minus, obs)) / 2
            gradients_near_zero.append(grad)

        var_near_zero = np.var(gradients_near_zero)
        ratio = var_near_zero / var_random if var_random > 0 else float('inf')
        print(f"    {n:<10} {var_random:<15.8f} {var_near_zero:<15.8f} {ratio:.1f}x")

    print(f"\n    Near-zero initialization mitigates the barren plateau")
    print(f"    by keeping the circuit close to the identity (gradient ~ O(1)).")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
