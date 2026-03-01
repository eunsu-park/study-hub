"""
11_vqe.py — Variational Quantum Eigensolver (VQE)

Demonstrates:
  - The VQE algorithm for finding ground state energies
  - Parameterized quantum circuit (ansatz) construction
  - Energy expectation value computation ⟨ψ(θ)|H|ψ(θ)⟩
  - Classical optimization loop using scipy.optimize
  - Application: H₂ molecule ground state energy analog
  - Comparison of VQE result with exact diagonalization

Uses NumPy + scipy.optimize.minimize (the only non-stdlib dependency beyond numpy).
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Callable

# ---------------------------------------------------------------------------
# Pauli matrices
# ---------------------------------------------------------------------------
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

KET_0 = np.array([1, 0], dtype=complex)


def tensor(*matrices: np.ndarray) -> np.ndarray:
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


# ---------------------------------------------------------------------------
# Rotation gates
# ---------------------------------------------------------------------------

def Rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def Rz(theta: float) -> np.ndarray:
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)


CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)


# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------

def build_h2_hamiltonian() -> np.ndarray:
    """Build a simplified H₂ molecule Hamiltonian in the STO-3G basis.

    H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*X0X1 + g5*Y0Y1

    Why: The real H₂ Hamiltonian in the minimal basis (STO-3G) after
    Jordan-Wigner transformation reduces to a 2-qubit operator with these
    Pauli terms.  The coefficients depend on the bond distance — we use
    values near the equilibrium geometry (~0.735 Å).

    These specific coefficients come from the standard quantum chemistry
    benchmark and represent the electronic structure of the hydrogen molecule.
    """
    # Coefficients at bond distance ≈ 0.735 Å
    g0 = -0.4804
    g1 = +0.3435
    g2 = -0.4347
    g3 = +0.5716
    g4 = +0.0910
    g5 = +0.0910

    H = (g0 * tensor(I, I) +
         g1 * tensor(Z, I) +
         g2 * tensor(I, Z) +
         g3 * tensor(Z, Z) +
         g4 * tensor(X, X) +
         g5 * tensor(Y, Y))

    return H


def build_simple_hamiltonian() -> np.ndarray:
    """Build a simple 2-qubit Hamiltonian for demonstration.

    H = -Z⊗Z - 0.5*(X⊗I + I⊗X)

    Why: This is the transverse-field Ising model on 2 sites.  It's a
    standard test case: the ZZ term favors aligned spins (ferromagnetic),
    while the X terms create quantum fluctuations.  The ground state is
    a non-trivial entangled state.
    """
    return -tensor(Z, Z) - 0.5 * (tensor(X, I) + tensor(I, X))


# ---------------------------------------------------------------------------
# Ansatz circuits
# ---------------------------------------------------------------------------

def hardware_efficient_ansatz(params: np.ndarray, n_qubits: int,
                               n_layers: int) -> np.ndarray:
    """Construct a hardware-efficient ansatz state.

    Structure per layer:
        Ry(θ) Rz(θ) on each qubit → CNOT chain

    Why: Hardware-efficient ansatze use gates native to the quantum processor,
    minimizing compilation overhead.  Ry+Rz gives full single-qubit coverage,
    and CNOT chains create entanglement.  The depth is controlled by n_layers,
    trading expressivity for circuit depth (and thus noise sensitivity).
    """
    dim = 2 ** n_qubits
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # Start from |00...0⟩

    param_idx = 0

    for layer in range(n_layers):
        # Single-qubit rotations
        for q in range(n_qubits):
            # Ry rotation
            gate = Ry(params[param_idx])
            ops = [I] * n_qubits
            ops[q] = gate
            full_gate = ops[0]
            for op in ops[1:]:
                full_gate = np.kron(full_gate, op)
            state = full_gate @ state
            param_idx += 1

            # Rz rotation
            gate = Rz(params[param_idx])
            ops = [I] * n_qubits
            ops[q] = gate
            full_gate = ops[0]
            for op in ops[1:]:
                full_gate = np.kron(full_gate, op)
            state = full_gate @ state
            param_idx += 1

        # Entangling CNOT layer
        # Why: Linear CNOT chain (0→1, 1→2, ...) is hardware-friendly on
        # linear qubit topologies (common in superconducting processors).
        for q in range(n_qubits - 1):
            # Build controlled-X
            proj_0 = np.array([[1, 0], [0, 0]], dtype=complex)
            proj_1 = np.array([[0, 0], [0, 1]], dtype=complex)
            ops_0 = [I] * n_qubits
            ops_0[q] = proj_0
            term_0 = ops_0[0]
            for op in ops_0[1:]:
                term_0 = np.kron(term_0, op)
            ops_1 = [I] * n_qubits
            ops_1[q] = proj_1
            ops_1[q + 1] = X
            term_1 = ops_1[0]
            for op in ops_1[1:]:
                term_1 = np.kron(term_1, op)
            cnot_full = term_0 + term_1
            state = cnot_full @ state

    return state


def count_params(n_qubits: int, n_layers: int) -> int:
    """Count number of variational parameters."""
    return n_layers * n_qubits * 2  # 2 params per qubit per layer (Ry, Rz)


# ---------------------------------------------------------------------------
# VQE core
# ---------------------------------------------------------------------------

def compute_energy(params: np.ndarray, hamiltonian: np.ndarray,
                   n_qubits: int, n_layers: int) -> float:
    """Compute the energy expectation value ⟨ψ(θ)|H|ψ(θ)⟩.

    Why: This is the cost function that VQE minimizes.  By the variational
    principle, ⟨ψ|H|ψ⟩ ≥ E₀ for any state |ψ⟩, where E₀ is the true
    ground state energy.  Minimizing over parameters θ gives the best
    possible approximation within the ansatz.
    """
    state = hardware_efficient_ansatz(params, n_qubits, n_layers)
    energy = np.real(state.conj() @ hamiltonian @ state)
    return energy


def run_vqe(hamiltonian: np.ndarray, n_qubits: int, n_layers: int,
            n_restarts: int = 5, verbose: bool = True) -> Dict:
    """Run VQE with multiple random restarts.

    Why: The VQE landscape can have local minima, so we run multiple
    optimizations from different initial points and keep the best result.
    This is a practical necessity — VQE has no guarantee of finding the
    global minimum, unlike classical eigensolvers.
    """
    n_params = count_params(n_qubits, n_layers)
    best_result = None
    best_energy = np.inf
    history = []

    for restart in range(n_restarts):
        # Random initial parameters
        x0 = np.random.uniform(-np.pi, np.pi, n_params)

        # Track optimization
        energies = []

        def callback(xk):
            e = compute_energy(xk, hamiltonian, n_qubits, n_layers)
            energies.append(e)

        # Why: We use L-BFGS-B, a quasi-Newton method well-suited for
        # smooth optimization with many parameters.  On real hardware,
        # gradient-free methods (COBYLA, Nelder-Mead) are often preferred
        # because gradients are noisy.
        result = minimize(
            compute_energy,
            x0,
            args=(hamiltonian, n_qubits, n_layers),
            method='L-BFGS-B',
            callback=callback,
            options={'maxiter': 200, 'ftol': 1e-12}
        )

        if result.fun < best_energy:
            best_energy = result.fun
            best_result = result
            history = energies

        if verbose:
            print(f"    Restart {restart + 1}: E = {result.fun:.8f}, "
                  f"iterations = {result.nit}, converged = {result.success}")

    return {
        'energy': best_energy,
        'params': best_result.x,
        'result': best_result,
        'history': history,
        'n_params': n_params,
    }


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_simple_hamiltonian():
    """VQE on a simple transverse-field Ising model."""
    print("=" * 60)
    print("DEMO 1: VQE on Transverse-Field Ising Model (2 qubits)")
    print("=" * 60)

    H = build_simple_hamiltonian()
    n_qubits = 2

    # Exact solution for comparison
    eigenvalues = np.linalg.eigvalsh(H)
    E_exact = eigenvalues[0]
    print(f"\n  Hamiltonian: H = -Z⊗Z - 0.5(X⊗I + I⊗X)")
    print(f"  Exact eigenvalues: {np.sort(eigenvalues)}")
    print(f"  Exact ground state energy: {E_exact:.8f}")

    for n_layers in [1, 2, 3]:
        n_params = count_params(n_qubits, n_layers)
        print(f"\n  --- {n_layers} layer(s), {n_params} parameters ---")
        result = run_vqe(H, n_qubits, n_layers, n_restarts=5)
        error = abs(result['energy'] - E_exact)
        print(f"    Best VQE energy: {result['energy']:.8f}")
        print(f"    Error: {error:.2e}")
        print(f"    Chemical accuracy (<1.6 mHa)? {'Yes' if error < 0.0016 else 'No'}")


def demo_h2_molecule():
    """VQE for the H₂ molecule analog."""
    print("\n" + "=" * 60)
    print("DEMO 2: VQE for H₂ Molecule (Simplified)")
    print("=" * 60)

    H = build_h2_hamiltonian()
    n_qubits = 2

    eigenvalues = np.linalg.eigvalsh(H)
    E_exact = eigenvalues[0]

    print(f"\n  H₂ Hamiltonian (STO-3G basis, bond distance ≈ 0.735 Å)")
    print(f"  Exact ground state energy: {E_exact:.8f} Hartree")
    print(f"  All eigenvalues: {np.sort(eigenvalues)}")

    # Why: For H₂, even 1 layer of the hardware-efficient ansatz is often
    # sufficient because the ground state has limited entanglement.
    for n_layers in [1, 2]:
        n_params = count_params(n_qubits, n_layers)
        print(f"\n  --- {n_layers} layer(s), {n_params} parameters ---")
        result = run_vqe(H, n_qubits, n_layers, n_restarts=8, verbose=True)
        error = abs(result['energy'] - E_exact)
        print(f"    Best VQE energy: {result['energy']:.8f} Hartree")
        print(f"    Error: {error:.2e} Hartree")
        print(f"    Chemical accuracy (<1.6 mHa)? {'Yes' if error < 0.0016 else 'No'}")


def demo_optimization_landscape():
    """Visualize the energy landscape for a 1-parameter ansatz."""
    print("\n" + "=" * 60)
    print("DEMO 3: VQE Energy Landscape (1-Parameter Slice)")
    print("=" * 60)

    H = build_simple_hamiltonian()
    n_qubits = 2
    n_layers = 1

    # Fix all params except one and scan
    np.random.seed(42)
    base_params = np.random.uniform(-np.pi, np.pi, count_params(n_qubits, n_layers))

    # Optimize first
    result = minimize(
        compute_energy,
        base_params,
        args=(H, n_qubits, n_layers),
        method='L-BFGS-B',
    )
    opt_params = result.x

    # Scan parameter 0
    print(f"\n  Scanning parameter θ₀ while others fixed at optimum:")
    thetas = np.linspace(-np.pi, np.pi, 31)
    energies = []
    for theta in thetas:
        params = opt_params.copy()
        params[0] = theta
        e = compute_energy(params, H, n_qubits, n_layers)
        energies.append(e)

    E_exact = np.linalg.eigvalsh(H)[0]

    print(f"\n  {'θ₀':>8} {'Energy':>12}")
    print(f"  {'─' * 22}")
    for theta, e in zip(thetas, energies):
        bar_len = int((e - min(energies)) / (max(energies) - min(energies)) * 30)
        bar = '#' * bar_len
        print(f"  {theta:>8.3f} {e:>12.6f}  {bar}")

    # Why: The VQE landscape for parametrized circuits can have multiple local
    # minima and saddle points.  This is the "barren plateau" problem — for
    # deep circuits with many qubits, gradients can vanish exponentially.
    print(f"\n  Optimal energy at θ₀ ≈ {opt_params[0]:.4f}: E ≈ {result.fun:.6f}")
    print(f"  Exact ground state: {E_exact:.6f}")


def demo_convergence():
    """Show VQE convergence over optimization iterations."""
    print("\n" + "=" * 60)
    print("DEMO 4: VQE Convergence")
    print("=" * 60)

    H = build_h2_hamiltonian()
    n_qubits = 2
    n_layers = 2
    E_exact = np.linalg.eigvalsh(H)[0]

    np.random.seed(123)
    n_params = count_params(n_qubits, n_layers)
    x0 = np.random.uniform(-np.pi, np.pi, n_params)

    # Track every function evaluation
    eval_energies = []

    def objective(params):
        e = compute_energy(params, H, n_qubits, n_layers)
        eval_energies.append(e)
        return e

    result = minimize(objective, x0, method='L-BFGS-B',
                      options={'maxiter': 100, 'ftol': 1e-14})

    print(f"\n  Optimization convergence ({len(eval_energies)} function evaluations):")
    print(f"  {'Eval #':<10} {'Energy':>12} {'Error':>12}")
    print(f"  {'─' * 36}")

    # Show selected iterations
    indices = [0, 1, 2, 5, 10, 20, 50, len(eval_energies) - 1]
    indices = [i for i in indices if i < len(eval_energies)]

    for i in indices:
        e = eval_energies[i]
        error = abs(e - E_exact)
        print(f"  {i:<10} {e:>12.8f} {error:>12.2e}")

    # Why: VQE converges quickly for small systems but may struggle for
    # larger molecules due to barren plateaus and local minima.  In practice,
    # problem-specific ansatze (like UCCSD) converge better than hardware-efficient ones.
    print(f"\n  Final energy: {result.fun:.8f} (exact: {E_exact:.8f})")
    print(f"  Final error: {abs(result.fun - E_exact):.2e}")


def demo_ansatz_expressibility():
    """Compare different ansatz depths."""
    print("\n" + "=" * 60)
    print("DEMO 5: Ansatz Expressibility vs Depth")
    print("=" * 60)

    H = build_h2_hamiltonian()
    n_qubits = 2
    E_exact = np.linalg.eigvalsh(H)[0]

    print(f"\n  H₂ Hamiltonian, exact E₀ = {E_exact:.8f}")
    print(f"\n  {'Layers':<10} {'Params':<10} {'Best Energy':>14} {'Error':>12} {'Evaluations':>14}")
    print(f"  {'─' * 62}")

    for n_layers in [1, 2, 3, 4]:
        n_params = count_params(n_qubits, n_layers)

        best_e = np.inf
        best_nfev = 0
        for _ in range(10):
            x0 = np.random.uniform(-np.pi, np.pi, n_params)
            result = minimize(compute_energy, x0,
                              args=(H, n_qubits, n_layers),
                              method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-14})
            if result.fun < best_e:
                best_e = result.fun
                best_nfev = result.nfev

        error = abs(best_e - E_exact)
        print(f"  {n_layers:<10} {n_params:<10} {best_e:>14.8f} {error:>12.2e} {best_nfev:>14}")

    # Why: Deeper ansatze can express more states but have more parameters,
    # making optimization harder.  There's a sweet spot where the ansatz is
    # expressive enough to capture the ground state without being so deep
    # that optimization becomes intractable.
    print(f"\n  More layers → more expressive, but harder to optimize.")
    print(f"  For H₂ (simple system), 1-2 layers usually suffice.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Quantum Computing — 11: Variational Quantum Eigensolver║")
    print("╚══════════════════════════════════════════════════════════╝")

    np.random.seed(2026)

    demo_simple_hamiltonian()
    demo_h2_molecule()
    demo_optimization_landscape()
    demo_convergence()
    demo_ansatz_expressibility()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
