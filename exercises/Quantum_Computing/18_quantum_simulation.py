"""
Exercises for Lesson 18: Quantum Simulation
Topic: Quantum_Computing

Solutions to practice problems covering Trotter-Suzuki decomposition,
time evolution observables, qDRIFT, QPE, and ADAPT-VQE.
"""

import numpy as np
from scipy.linalg import expm

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_list(ops):
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def build_ising(n, J=1.0, h=0.5):
    """Build transverse-field Ising Hamiltonian."""
    N = 2 ** n
    H = np.zeros((N, N), dtype=complex)
    terms = []
    for i in range(n - 1):
        ops = [I] * n; ops[i] = Z; ops[i+1] = Z
        t = -J * kron_list(ops)
        H += t; terms.append((-J, kron_list(ops) / (-J)))
    for i in range(n):
        ops = [I] * n; ops[i] = X
        t = -h * kron_list(ops)
        H += t; terms.append((-h, kron_list(ops) / (-h)))
    return H, terms


# === Exercise 1: Trotter Error Analysis ===
def exercise_1():
    """Trotter error scaling for 4-qubit Ising model."""
    print("=" * 60)
    print("Exercise 1: Trotter Error Analysis")
    print("=" * 60)

    n_qubits = 4
    H, terms = build_ising(n_qubits)
    t = 1.0

    print(f"\n{'Steps':>8} {'1st order err':>16} {'2nd order err':>16}")
    print("-" * 44)

    for r in [1, 2, 5, 10, 20, 50]:
        dt = t / r
        U_exact = expm(-1j * H * t)

        # First order
        N = H.shape[0]
        U1 = np.eye(N, dtype=complex)
        for _ in range(r):
            for c, op in terms:
                U1 = expm(-1j * c * op * dt) @ U1
        err1 = np.linalg.norm(U1 - U_exact, ord=2)

        # Second order
        U2 = np.eye(N, dtype=complex)
        for _ in range(r):
            for c, op in terms:
                U2 = expm(-1j * c * op * dt / 2) @ U2
            for c, op in reversed(terms):
                U2 = expm(-1j * c * op * dt / 2) @ U2
        err2 = np.linalg.norm(U2 - U_exact, ord=2)

        print(f"{r:8d} {err1:16.2e} {err2:16.2e}")


# === Exercise 2: Observable Dynamics ===
def exercise_2():
    """Time evolution of magnetization and entanglement entropy."""
    print("\n" + "=" * 60)
    print("Exercise 2: Observable Dynamics")
    print("=" * 60)

    n_qubits = 5
    H, _ = build_ising(n_qubits, J=1.0, h=0.5)
    N = 2 ** n_qubits

    # Initial state: |11111>
    psi0 = np.zeros(N, dtype=complex)
    psi0[N - 1] = 1.0

    print(f"\n{'Time':>6} {'<Mz>':>12} {'S(2|3)':>12}")
    print("-" * 34)

    for t in np.linspace(0, 10, 21):
        psi = expm(-1j * H * t) @ psi0

        # Magnetization
        mz = 0
        for i in range(n_qubits):
            ops = [I] * n_qubits; ops[i] = Z
            mz += np.real(psi.conj() @ kron_list(ops) @ psi)
        mz /= n_qubits

        # Entanglement entropy (bipartition: first 2 | last 3)
        psi_mat = psi.reshape(4, 8)  # 2^2 x 2^3
        rho_A = psi_mat @ psi_mat.conj().T
        evals = np.real(np.linalg.eigvalsh(rho_A))
        evals = evals[evals > 1e-12]
        entropy = -np.sum(evals * np.log2(evals))

        print(f"{t:6.2f} {mz:12.4f} {entropy:12.4f}")


# === Exercise 3: qDRIFT vs Trotter ===
def exercise_3():
    """Compare qDRIFT and second-order Trotter."""
    print("\n" + "=" * 60)
    print("Exercise 3: qDRIFT vs Trotter (Heisenberg model)")
    print("=" * 60)

    n = 4
    N = 2 ** n
    H = np.zeros((N, N), dtype=complex)
    terms = []
    for i in range(n - 1):
        for P in [X, Y, Z]:
            ops = [I] * n; ops[i] = P; ops[i+1] = P
            t_op = kron_list(ops)
            H += t_op; terms.append((1.0, t_op))

    t_total = 1.0
    U_exact = expm(-1j * H * t_total)

    gate_budget = 100
    dt_trotter = t_total / (gate_budget // len(terms))
    n_steps = gate_budget // len(terms)

    U_t = np.eye(N, dtype=complex)
    for _ in range(n_steps):
        for c, op in terms:
            U_t = expm(-1j * c * op * dt_trotter / 2) @ U_t
        for c, op in reversed(terms):
            U_t = expm(-1j * c * op * dt_trotter / 2) @ U_t
    err_trotter = np.linalg.norm(U_t - U_exact, ord=2)

    print(f"\nGate budget: {gate_budget}")
    print(f"Trotter-2 error: {err_trotter:.2e}")

    # qDRIFT
    rng = np.random.default_rng(42)
    coeffs = np.array([abs(c) for c, _ in terms])
    lam = np.sum(coeffs)
    probs = coeffs / lam
    tau = lam * t_total / gate_budget

    errors_qd = []
    for seed in range(50):
        rng2 = np.random.default_rng(seed)
        U_qd = np.eye(N, dtype=complex)
        for _ in range(gate_budget):
            k = rng2.choice(len(terms), p=probs)
            c, op = terms[k]
            U_qd = expm(-1j * np.sign(c) * op * tau) @ U_qd
        errors_qd.append(np.linalg.norm(U_qd - U_exact, ord=2))

    print(f"qDRIFT error (mean): {np.mean(errors_qd):.2e} +/- {np.std(errors_qd):.2e}")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
