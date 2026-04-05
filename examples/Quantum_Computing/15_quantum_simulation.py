"""
15_quantum_simulation.py — Quantum Simulation: Trotter-Suzuki Decomposition

Demonstrates:
  - Transverse-field Ising model Hamiltonian construction
  - First-order Trotter-Suzuki decomposition
  - Second-order Trotter-Suzuki decomposition
  - Comparison of Trotter approximation against exact matrix exponential
  - Error scaling with number of Trotter steps
  - Time evolution of observables (magnetization)

All computations use pure NumPy.
"""

import numpy as np
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Pauli matrices and helpers
# ---------------------------------------------------------------------------
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_list(ops: List[np.ndarray]) -> np.ndarray:
    """Tensor product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def matrix_exp(A: np.ndarray) -> np.ndarray:
    """Compute matrix exponential via eigendecomposition.

    Why: For Hermitian H, e^{-iHt} is computed exactly via diagonalization:
    H = UDU†  →  e^{-iHt} = U·e^{-iDt}·U†.  This serves as our "exact"
    reference for evaluating Trotter approximation accuracy.
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    return eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.conj().T


# ---------------------------------------------------------------------------
# Hamiltonian Construction
# ---------------------------------------------------------------------------

def transverse_ising_hamiltonian(n: int, J: float = 1.0,
                                  h: float = 1.0) -> Tuple[np.ndarray,
                                                             np.ndarray,
                                                             np.ndarray]:
    """Build the transverse-field Ising model Hamiltonian.

    H = -J Σ_i Z_i Z_{i+1}  -  h Σ_i X_i

    Returns (H_total, H_ZZ, H_X) where H_ZZ is the Ising coupling term
    and H_X is the transverse field term.

    Why: The transverse-field Ising model is a paradigmatic quantum many-body
    system with a quantum phase transition at J/h = 1.  It is one of the
    simplest Hamiltonians that is hard to simulate classically but natural
    for quantum simulation.  The two non-commuting terms [H_ZZ, H_X] ≠ 0
    make Trotterization necessary.
    """
    dim = 2 ** n

    # ZZ coupling: -J Σ_i Z_i Z_{i+1}
    H_ZZ = np.zeros((dim, dim), dtype=complex)
    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i + 1] = Z
        H_ZZ += -J * kron_list(ops)

    # Transverse field: -h Σ_i X_i
    H_X = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H_X += -h * kron_list(ops)

    return H_ZZ + H_X, H_ZZ, H_X


# ---------------------------------------------------------------------------
# Trotter-Suzuki Decomposition
# ---------------------------------------------------------------------------

def trotter_first_order(H_terms: List[np.ndarray], t: float,
                         n_steps: int) -> np.ndarray:
    """First-order Trotter-Suzuki decomposition.

    e^{-i(A+B)t} ≈ (e^{-iAΔt} · e^{-iBΔt})^n

    where Δt = t/n.

    Why: When [A, B] ≠ 0, we cannot simply compute e^{-i(A+B)t} as a product
    e^{-iAt}·e^{-iBt}.  Trotter's formula splits the evolution into small
    steps where the error per step is O(Δt²), giving total error O(t²/n).
    This is the foundation of quantum simulation algorithms.
    """
    dim = H_terms[0].shape[0]
    dt = t / n_steps

    # Why: Compute the single-step propagator once, then exponentiate.
    # Each term e^{-iH_k·Δt} is a unitary that can be implemented with
    # quantum gates when H_k has a simple structure.
    step_unitary = np.eye(dim, dtype=complex)
    for H_k in H_terms:
        step_unitary = matrix_exp(-1j * H_k * dt) @ step_unitary

    # Apply n_steps times
    result = np.eye(dim, dtype=complex)
    for _ in range(n_steps):
        result = step_unitary @ result

    return result


def trotter_second_order(H_terms: List[np.ndarray], t: float,
                          n_steps: int) -> np.ndarray:
    """Second-order Trotter-Suzuki decomposition (symmetric Trotter).

    e^{-i(A+B)t} ≈ (e^{-iAΔt/2} · e^{-iBΔt} · e^{-iAΔt/2})^n

    Why: The symmetric (Strang) splitting cancels the leading error term,
    reducing the per-step error to O(Δt³) and total error to O(t³/n²).
    This quadratic improvement in step count is significant for practical
    quantum simulation, as each Trotter step costs quantum gates.
    """
    dim = H_terms[0].shape[0]
    dt = t / n_steps

    # Build the symmetric step: e^{-iH_1·Δt/2} · e^{-iH_2·Δt} · ... · e^{-iH_1·Δt/2}
    half_first = matrix_exp(-1j * H_terms[0] * dt / 2)
    full_terms = [matrix_exp(-1j * H_k * dt) for H_k in H_terms[1:]]

    step_unitary = half_first.copy()
    for U_k in full_terms:
        step_unitary = U_k @ step_unitary
    step_unitary = half_first @ step_unitary

    # Apply n_steps times
    result = np.eye(dim, dtype=complex)
    for _ in range(n_steps):
        result = step_unitary @ result

    return result


def operator_error(U_approx: np.ndarray, U_exact: np.ndarray) -> float:
    """Compute the operator norm error ||U_approx - U_exact||.

    Why: The spectral norm (largest singular value of the difference)
    gives the worst-case error over all input states.  This is the
    standard metric for evaluating Trotter accuracy.
    """
    diff = U_approx - U_exact
    return np.linalg.norm(diff, ord=2)


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_ising_hamiltonian():
    """Build and display the transverse-field Ising Hamiltonian."""
    print("=" * 60)
    print("DEMO 1: Transverse-Field Ising Model")
    print("=" * 60)

    for n in [2, 3, 4]:
        H_total, H_ZZ, H_X = transverse_ising_hamiltonian(n, J=1.0, h=1.0)
        dim = 2 ** n

        # Why: The ground state energy reveals the phase structure.
        # At J/h = 1, the system is at the quantum critical point.
        eigvals = np.sort(np.linalg.eigvalsh(H_total))
        gap = eigvals[1] - eigvals[0]

        print(f"\n  {n} qubits (dim = {dim}):")
        print(f"    H = -J·Σ Z_i Z_{{i+1}} - h·Σ X_i  (J=1, h=1)")
        print(f"    Ground state energy: {eigvals[0]:.6f}")
        print(f"    Spectral gap: {gap:.6f}")
        print(f"    Commutator [H_ZZ, H_X] = 0? "
              f"{np.allclose(H_ZZ @ H_X, H_X @ H_ZZ)}")

    # Why: The non-commutativity of H_ZZ and H_X is precisely why we need
    # Trotter decomposition — we cannot factorize e^{-i(H_ZZ+H_X)t} directly.


def demo_trotter_comparison():
    """Compare first and second order Trotter against exact evolution."""
    print("\n" + "=" * 60)
    print("DEMO 2: Trotter vs Exact Evolution")
    print("=" * 60)

    n = 3
    H_total, H_ZZ, H_X = transverse_ising_hamiltonian(n, J=1.0, h=1.0)
    t = 1.0

    # Exact evolution
    U_exact = matrix_exp(-1j * H_total * t)

    print(f"\n  {n}-qubit Ising model, t = {t}")
    print(f"\n  {'Steps':>8} {'1st Order Error':>18} {'2nd Order Error':>18} {'Ratio':>10}")
    print(f"  {'─' * 56}")

    prev_err1 = None
    prev_err2 = None
    for n_steps in [1, 2, 4, 8, 16, 32, 64]:
        U_t1 = trotter_first_order([H_ZZ, H_X], t, n_steps)
        U_t2 = trotter_second_order([H_ZZ, H_X], t, n_steps)

        err1 = operator_error(U_t1, U_exact)
        err2 = operator_error(U_t2, U_exact)

        # Why: For first-order Trotter, doubling steps should halve the error
        # (O(1/n) scaling).  For second-order, error should decrease by 4x
        # (O(1/n²) scaling).
        ratio1 = f"{prev_err1/err1:.2f}" if prev_err1 and err1 > 1e-15 else "—"
        ratio2 = f"{prev_err2/err2:.2f}" if prev_err2 and err2 > 1e-15 else "—"

        print(f"  {n_steps:8d} {err1:18.2e} {err2:18.2e} "
              f"{ratio1:>5}/{ratio2:>5}")

        prev_err1 = err1
        prev_err2 = err2

    print(f"\n  Expected ratios: ~2.0 (1st order) / ~4.0 (2nd order)")


def demo_error_scaling():
    """Show Trotter error scaling with number of steps."""
    print("\n" + "=" * 60)
    print("DEMO 3: Error Scaling Analysis")
    print("=" * 60)

    # Why: Verifying the theoretical error scaling O(t²/n) for first-order
    # and O(t³/n²) for second-order Trotter is essential for choosing the
    # right number of steps in practical simulations.

    n = 3
    H_total, H_ZZ, H_X = transverse_ising_hamiltonian(n, J=1.0, h=0.5)

    print(f"\n  Error scaling for different evolution times:")
    print(f"\n  {'t':>6} {'Steps':>8} {'1st Err':>12} {'2nd Err':>12} "
          f"{'1st·n':>12} {'2nd·n²':>12}")
    print(f"  {'─' * 66}")

    for t in [0.5, 1.0, 2.0]:
        U_exact = matrix_exp(-1j * H_total * t)
        for n_steps in [8, 16, 32]:
            U_t1 = trotter_first_order([H_ZZ, H_X], t, n_steps)
            U_t2 = trotter_second_order([H_ZZ, H_X], t, n_steps)

            err1 = operator_error(U_t1, U_exact)
            err2 = operator_error(U_t2, U_exact)

            # Why: err1 * n should be roughly constant (confirming O(1/n)),
            # and err2 * n² should be roughly constant (confirming O(1/n²)).
            print(f"  {t:6.1f} {n_steps:8d} {err1:12.2e} {err2:12.2e} "
                  f"{err1 * n_steps:12.4f} {err2 * n_steps**2:12.4f}")
        print()


def demo_time_evolution():
    """Simulate time evolution and track observables."""
    print("\n" + "=" * 60)
    print("DEMO 4: Time Evolution of Magnetization")
    print("=" * 60)

    n = 4
    H_total, H_ZZ, H_X = transverse_ising_hamiltonian(n, J=1.0, h=0.5)
    dim = 2 ** n

    # Why: Starting from the fully polarized state |↑↑↑↑⟩ = |0000⟩ (all spin up),
    # we track how the transverse field causes the magnetization to oscillate.
    # This is a direct quantum simulation observable.
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |0000⟩ = all spin up in Z basis

    # Magnetization operator: (1/n) Σ_i Z_i
    M_z = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        ops = [I] * n
        ops[i] = Z
        M_z += kron_list(ops)
    M_z /= n

    # Transverse magnetization: (1/n) Σ_i X_i
    M_x = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        M_x += kron_list(ops)
    M_x /= n

    t_max = 5.0
    n_points = 20
    n_trotter = 50

    print(f"\n  {n}-qubit Ising model, J=1.0, h=0.5")
    print(f"  Initial state: |{'0' * n}⟩ (fully polarized)")
    print(f"  Trotter steps per time point: {n_trotter}")
    print(f"\n  {'t':>6} {'⟨M_z⟩':>10} {'⟨M_x⟩':>10} {'|ψ(t)|²':>10}")
    print(f"  {'─' * 40}")

    for i in range(n_points + 1):
        t = t_max * i / n_points

        if t == 0:
            psi_t = state.copy()
        else:
            U = trotter_second_order([H_ZZ, H_X], t, n_trotter)
            psi_t = U @ state

        mz = np.real(psi_t.conj() @ M_z @ psi_t)
        mx = np.real(psi_t.conj() @ M_x @ psi_t)
        norm = np.real(psi_t.conj() @ psi_t)

        print(f"  {t:6.2f} {mz:10.4f} {mx:10.4f} {norm:10.6f}")


def demo_phase_diagram():
    """Explore the quantum phase transition in the Ising model."""
    print("\n" + "=" * 60)
    print("DEMO 5: Quantum Phase Transition (Ising Model)")
    print("=" * 60)

    # Why: The transverse-field Ising model has a quantum phase transition
    # at J/h = 1.  For J >> h, the ground state is ferromagnetic (|000...0⟩
    # or |111...1⟩).  For h >> J, it is paramagnetic (product of |+⟩ states).
    # The ground state magnetization is the order parameter.

    n = 6
    dim = 2 ** n

    # Magnetization operators
    M_z_sq = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        for j in range(n):
            ops = [I] * n
            ops[i] = Z
            if i != j:
                ops[j] = Z
            M_z_sq += kron_list(ops)
    M_z_sq /= n ** 2

    print(f"\n  {n}-qubit Ising chain, varying h/J:")
    print(f"\n  {'h/J':>8} {'E_0':>10} {'Gap':>10} {'⟨M_z²⟩':>10}")
    print(f"  {'─' * 42}")

    for ratio in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 3.0]:
        H_total, _, _ = transverse_ising_hamiltonian(n, J=1.0, h=ratio)
        eigvals, eigvecs = np.linalg.eigh(H_total)
        gs = eigvecs[:, 0]
        e0 = eigvals[0]
        gap = eigvals[1] - eigvals[0]
        mz_sq = np.real(gs.conj() @ M_z_sq @ gs)

        print(f"  {ratio:8.2f} {e0:10.4f} {gap:10.4f} {mz_sq:10.4f}")

    # Why: ⟨M_z²⟩ ≈ 1 in the ferromagnetic phase (h/J < 1) and drops toward
    # 1/n in the paramagnetic phase (h/J > 1).  The gap closes at the
    # critical point h/J = 1 (for infinite systems; finite-size effects
    # smooth it for small n).


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Quantum Computing — 15: Quantum Simulation            ║")
    print("╚══════════════════════════════════════════════════════════╝")

    np.random.seed(2026)

    demo_ising_hamiltonian()
    demo_trotter_comparison()
    demo_error_scaling()
    demo_time_evolution()
    demo_phase_diagram()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
