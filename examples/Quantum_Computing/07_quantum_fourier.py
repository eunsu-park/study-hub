"""
07_quantum_fourier.py — Quantum Fourier Transform and Phase Estimation

Demonstrates:
  - QFT matrix construction from scratch
  - Comparison between QFT and classical DFT (numpy.fft)
  - QFT decomposition into elementary gates (H + controlled rotations)
  - Inverse QFT
  - Quantum Phase Estimation (QPE) algorithm
  - Period finding using QFT (foundation for Shor's algorithm)

All computations use pure NumPy.
"""

import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# Gate definitions
# ---------------------------------------------------------------------------
I = np.eye(2, dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)


def Rk(k: int) -> np.ndarray:
    """Controlled rotation gate R_k = diag(1, e^{2πi/2^k}).

    Why: The QFT circuit uses controlled-R_k gates with decreasing rotation
    angles (2π/2¹, 2π/2², ...).  These fine-grained phase rotations encode
    frequency information — each qubit accumulates phase from all others,
    mirroring the discrete Fourier transform structure.
    """
    angle = 2 * np.pi / (2 ** k)
    return np.array([[1, 0], [0, np.exp(1j * angle)]], dtype=complex)


# ---------------------------------------------------------------------------
# QFT Matrix Construction
# ---------------------------------------------------------------------------

def qft_matrix(n: int) -> np.ndarray:
    """Construct the n-qubit QFT matrix directly.

    QFT_N[j,k] = (1/√N) · ω^{jk}  where ω = e^{2πi/N}, N = 2^n

    Why: The QFT is the quantum analog of the classical Discrete Fourier
    Transform (DFT).  It transforms computational basis amplitudes into
    frequency-domain amplitudes.  The key difference from classical FFT:
    the QFT acts on 2^n amplitudes using only O(n²) gates, while classical
    FFT needs O(N log N) = O(n·2^n) operations.
    """
    N = 2 ** n
    omega = np.exp(2j * np.pi / N)

    # Why: We construct the unitary matrix element by element.
    # F[j,k] = ω^{jk}/√N — each element is a root of unity.
    F = np.zeros((N, N), dtype=complex)
    for j in range(N):
        for k in range(N):
            F[j, k] = omega ** (j * k)
    return F / np.sqrt(N)


def inverse_qft_matrix(n: int) -> np.ndarray:
    """Construct the inverse QFT matrix.

    Why: Since QFT is unitary, QFT⁻¹ = QFT†.  The inverse QFT transforms
    from frequency domain back to computational basis — essential for reading
    out results in algorithms like phase estimation and Shor's.
    """
    return qft_matrix(n).conj().T


# ---------------------------------------------------------------------------
# QFT via Circuit Decomposition
# ---------------------------------------------------------------------------

def qft_circuit(state: np.ndarray, n: int) -> np.ndarray:
    """Apply QFT using the standard circuit decomposition.

    The circuit applies:
      1. For each qubit j (from MSB to LSB):
         a. Hadamard on qubit j
         b. Controlled-R_k from qubit j+1, j+2, ... with k=2,3,...
      2. SWAP qubits to reverse bit order

    Why: This decomposition uses O(n²) gates total — far fewer than the
    O(2^n) parameters in the full unitary matrix.  The circuit exploits
    the recursive structure of the DFT: each qubit independently encodes
    one bit of the frequency, conditioned on the lower bits.
    """
    N = 2 ** n
    result = state.copy()

    # Why: We build full-system operators for each gate.  In a real simulator,
    # we'd apply gates more efficiently, but this makes the circuit structure clear.
    for j in range(n):
        # Step 1: Hadamard on qubit j
        ops = [I] * n
        ops[j] = H
        H_full = ops[0]
        for op in ops[1:]:
            H_full = np.kron(H_full, op)
        result = H_full @ result

        # Step 2: Controlled rotations from qubits j+1, j+2, ...
        for k_offset in range(1, n - j):
            control = j + k_offset
            k = k_offset + 1  # R_2, R_3, ...

            # Build controlled-R_k with control=`control`, target=`j`
            proj_0 = np.array([[1, 0], [0, 0]], dtype=complex)
            proj_1 = np.array([[0, 0], [0, 1]], dtype=complex)
            gate = Rk(k)

            # Why: We use the projector formulation: C-U = |0⟩⟨0|⊗I + |1⟩⟨1|⊗U
            # with appropriate tensor product positioning for control and target qubits.
            ops_0 = [I] * n
            ops_0[control] = proj_0
            term_0 = ops_0[0]
            for op in ops_0[1:]:
                term_0 = np.kron(term_0, op)

            ops_1 = [I] * n
            ops_1[control] = proj_1
            ops_1[j] = gate
            term_1 = ops_1[0]
            for op in ops_1[1:]:
                term_1 = np.kron(term_1, op)

            CR = term_0 + term_1
            result = CR @ result

    # Why: The QFT circuit produces outputs in reversed bit order compared
    # to the standard DFT convention.  We swap qubits to match.
    result = _swap_bits(result, n)
    return result


def _swap_bits(state: np.ndarray, n: int) -> np.ndarray:
    """Reverse the qubit ordering of a state vector.

    Why: The standard QFT circuit produces results with qubit ordering reversed
    relative to the DFT convention.  This swap corrects for that — it's a
    purely classical reindexing of the amplitudes.
    """
    N = len(state)
    new_state = np.zeros(N, dtype=complex)
    for i in range(N):
        # Reverse the n-bit representation of i
        reversed_i = int(format(i, f'0{n}b')[::-1], 2)
        new_state[reversed_i] = state[i]
    return new_state


# ---------------------------------------------------------------------------
# Phase Estimation
# ---------------------------------------------------------------------------

def phase_estimation(unitary: np.ndarray, eigenstate: np.ndarray,
                     n_precision: int) -> Tuple[float, np.ndarray]:
    """Quantum Phase Estimation (QPE) algorithm.

    Given a unitary U and its eigenstate |u⟩ with U|u⟩ = e^{2πiφ}|u⟩,
    QPE estimates the phase φ using n_precision qubits.

    Why: QPE is one of the most important subroutines in quantum computing.
    It powers Shor's algorithm (period finding), quantum chemistry (energy
    estimation), and many other applications.  The key idea: use controlled-U^{2^k}
    to create a state whose QFT encodes φ in binary.

    Circuit:
      |0⟩ ─ H ─ C-U^{2^{n-1}} ─ ─ ─ ─ ─ ─ ─ ┐
      |0⟩ ─ H ─ ─ ─ ─ ─ ─ C-U^{2^{n-2}} ─ ─ ┤
       :                                        │ QFT⁻¹
      |0⟩ ─ H ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ C-U^{2^0} ┤
      |u⟩ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
    """
    n_eigen = int(np.log2(len(eigenstate)))
    n_total = n_precision + n_eigen
    dim_total = 2 ** n_total
    dim_precision = 2 ** n_precision
    dim_eigen = 2 ** n_eigen

    # Initialize: |0...0⟩ ⊗ |u⟩
    state = np.zeros(dim_total, dtype=complex)
    for i in range(dim_eigen):
        state[i] = eigenstate[i]  # |00...0⟩ ⊗ |u⟩

    # Step 1: Hadamard on all precision qubits
    for q in range(n_precision):
        ops = [I] * n_total
        ops[q] = H
        H_full = ops[0]
        for op in ops[1:]:
            H_full = np.kron(H_full, op)
        state = H_full @ state

    # Step 2: Controlled-U^{2^k} for each precision qubit k
    # Why: The k-th precision qubit applies U^{2^k} — this accumulates phase
    # 2^k·φ, effectively reading out bit k of the binary expansion of φ.
    for k in range(n_precision):
        control_qubit = n_precision - 1 - k  # MSB first
        power = 2 ** k

        # Compute U^{2^k}
        U_power = np.linalg.matrix_power(unitary, power)

        # Build controlled-U^{2^k}
        proj_0 = np.array([[1, 0], [0, 0]], dtype=complex)
        proj_1 = np.array([[0, 0], [0, 1]], dtype=complex)

        ops_0 = [I] * n_precision + [np.eye(dim_eigen, dtype=complex)]
        ops_0[control_qubit] = proj_0
        term_0 = ops_0[0]
        for op in ops_0[1:]:
            term_0 = np.kron(term_0, op)

        ops_1 = [I] * n_precision + [U_power]
        ops_1[control_qubit] = proj_1
        term_1 = ops_1[0]
        for op in ops_1[1:]:
            term_1 = np.kron(term_1, op)

        CU = term_0 + term_1
        state = CU @ state

    # Step 3: Inverse QFT on precision qubits
    # Why: After the controlled-U operations, the precision register encodes
    # the phase in the Fourier basis.  The inverse QFT converts it to the
    # computational basis so we can read it out.
    iqft = inverse_qft_matrix(n_precision)
    iqft_full = np.kron(iqft, np.eye(dim_eigen, dtype=complex))
    state = iqft_full @ state

    # Extract probabilities of precision register
    probs_precision = np.zeros(dim_precision)
    for p_idx in range(dim_precision):
        for e_idx in range(dim_eigen):
            idx = p_idx * dim_eigen + e_idx
            probs_precision[p_idx] += np.abs(state[idx]) ** 2

    # Most likely outcome
    best = np.argmax(probs_precision)
    estimated_phase = best / dim_precision

    return estimated_phase, probs_precision


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_qft_matrix():
    """Construct and display QFT matrices."""
    print("=" * 60)
    print("DEMO 1: QFT Matrix Construction")
    print("=" * 60)

    for n in [1, 2, 3]:
        F = qft_matrix(n)
        N = 2 ** n
        print(f"\n  QFT_{N} ({n} qubits):")
        for row in F:
            entries = [f"{v.real:+6.3f}{v.imag:+6.3f}j" if abs(v.imag) > 1e-10
                       else f"{v.real:+6.3f}       " for v in row]
            print(f"    [{', '.join(entries)}]")

        # Verify unitarity
        product = F.conj().T @ F
        print(f"    Unitary? {np.allclose(product, np.eye(N))}")


def demo_qft_vs_fft():
    """Compare QFT with classical FFT."""
    print("\n" + "=" * 60)
    print("DEMO 2: QFT vs Classical DFT (numpy.fft)")
    print("=" * 60)

    n = 3
    N = 2 ** n

    # Create a test state
    np.random.seed(42)
    state = np.random.randn(N) + 1j * np.random.randn(N)
    state = state / np.linalg.norm(state)

    # QFT
    F = qft_matrix(n)
    qft_result = F @ state

    # Classical DFT (normalized to match QFT convention)
    # Why: numpy.fft uses the convention F[k] = Σ x[n]·e^{-2πink/N}, without
    # the 1/√N normalization.  QFT uses F[k] = (1/√N) Σ x[n]·e^{+2πink/N}.
    # We adjust for both the sign and normalization to compare.
    dft_result = np.fft.fft(state) / np.sqrt(N)

    print(f"\n  Input state (normalized random, {N} elements):")
    for i, amp in enumerate(state):
        print(f"    |{format(i, f'0{n}b')}⟩: {amp.real:+.4f}{amp.imag:+.4f}j")

    print(f"\n  {'Basis':<8} {'QFT':>22} {'DFT (adjusted)':>22} {'Match?':>8}")
    print(f"  {'─' * 62}")
    for i in range(N):
        q = qft_result[i]
        d = dft_result[i]
        match = np.abs(q - d) < 1e-10
        print(f"  |{format(i, f'0{n}b')}⟩   {q.real:+.4f}{q.imag:+.4f}j"
              f"   {d.real:+.4f}{d.imag:+.4f}j   {'Yes' if match else 'No':>8}")

    # Why: QFT and DFT should agree up to sign convention.  The sign difference
    # (e^{+2πi} vs e^{-2πi}) means QFT = DFT†, so QFT|x⟩ corresponds to the
    # inverse DFT of the amplitudes.
    print(f"\n  Note: QFT uses ω = e^{{+2πi/N}}, numpy.fft uses e^{{-2πi/N}}")
    print(f"  QFT matrix = conjugate of DFT matrix (up to normalization)")


def demo_qft_circuit():
    """Verify circuit decomposition matches matrix QFT."""
    print("\n" + "=" * 60)
    print("DEMO 3: QFT Circuit Decomposition")
    print("=" * 60)

    for n in [2, 3, 4]:
        N = 2 ** n
        # Test with several states
        all_match = True
        for test_idx in range(min(N, 4)):
            state = np.zeros(N, dtype=complex)
            state[test_idx] = 1.0

            matrix_result = qft_matrix(n) @ state
            circuit_result = qft_circuit(state, n)

            if not np.allclose(matrix_result, circuit_result):
                all_match = False

        n_gates = n * (n + 1) // 2  # H gates + controlled rotations
        print(f"  {n} qubits: circuit ≡ matrix? {all_match}  "
              f"(≈{n_gates} gates, depth ≈ {n})")

    # Why: The circuit decomposition uses n(n+1)/2 gates — O(n²).
    # This is exponentially fewer than the 2^n × 2^n matrix elements,
    # making QFT practical on quantum hardware.
    print(f"\n  Gate count: O(n²) — exponentially better than O(N log N) classical FFT")
    print(f"  when considering that QFT processes N = 2^n amplitudes")


def demo_qft_on_basis_states():
    """Show QFT action on computational basis states."""
    print("\n" + "=" * 60)
    print("DEMO 4: QFT on Computational Basis States")
    print("=" * 60)

    n = 3
    N = 2 ** n
    F = qft_matrix(n)

    print(f"\n  QFT|k⟩ creates a state with frequency k:")
    for k in range(N):
        state = np.zeros(N, dtype=complex)
        state[k] = 1.0
        result = F @ state

        # The phases
        phases = np.angle(result) / np.pi  # in units of π
        print(f"\n  QFT|{format(k, f'0{n}b')}⟩ (k={k}):")
        print(f"    Amplitudes: all {1/np.sqrt(N):.4f}")
        print(f"    Phases/π:   [{', '.join(f'{p:+.2f}' for p in phases)}]")

    # Why: QFT|k⟩ = (1/√N) Σ_j e^{2πijk/N} |j⟩ — a uniform superposition
    # where each basis state carries a phase proportional to j·k.
    # Higher k → faster phase rotation → higher "frequency".


def demo_phase_estimation():
    """Run Quantum Phase Estimation."""
    print("\n" + "=" * 60)
    print("DEMO 5: Quantum Phase Estimation (QPE)")
    print("=" * 60)

    # Why: We test QPE with a known unitary whose eigenphase we can verify.
    # The simplest case: a phase gate with known angle.

    # Test 1: Phase gate with φ = 1/4 (exact binary fraction → perfect estimation)
    print(f"\n  Test 1: U = Rz with phase φ = 1/4")
    phi_true = 0.25
    U = np.array([[1, 0], [0, np.exp(2j * np.pi * phi_true)]], dtype=complex)
    eigenstate = np.array([0, 1], dtype=complex)  # |1⟩ is eigenstate

    for n_prec in [3, 4, 5]:
        phi_est, probs = phase_estimation(U, eigenstate, n_prec)
        print(f"    {n_prec} precision qubits: φ_estimated = {phi_est:.6f}"
              f"  (true: {phi_true})  error: {abs(phi_est - phi_true):.6f}")

    # Why: φ = 1/4 = 0.01 in binary, so 3+ precision qubits give exact results.
    # Non-binary fractions will have estimation error that decreases with more qubits.

    # Test 2: Phase φ = 1/3 (not exact binary fraction → approximation)
    print(f"\n  Test 2: U with phase φ = 1/3 (non-exact binary fraction)")
    phi_true = 1 / 3
    U = np.array([[1, 0], [0, np.exp(2j * np.pi * phi_true)]], dtype=complex)

    for n_prec in [3, 4, 5, 6, 8]:
        phi_est, probs = phase_estimation(U, eigenstate, n_prec)
        print(f"    {n_prec} precision qubits: φ_estimated = {phi_est:.6f}"
              f"  (true: {phi_true:.6f})  error: {abs(phi_est - phi_true):.6f}")

    # Test 3: 2×2 unitary with known eigenvalues
    print(f"\n  Test 3: Hadamard gate H (eigenvalues ±1 → phases 0 and 1/2)")
    # H has eigenvalues +1 (phase 0) and -1 (phase 1/2)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    for i, (eigval, eigvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        phase_true = np.angle(eigval) / (2 * np.pi) % 1
        phi_est, _ = phase_estimation(H, eigvec, 4)
        print(f"    Eigenvalue {eigval:+.4f}: true phase = {phase_true:.4f},"
              f" estimated = {phi_est:.4f}")


def demo_period_finding_qft():
    """Use QFT to find period of a simple periodic state."""
    print("\n" + "=" * 60)
    print("DEMO 6: Period Finding with QFT")
    print("=" * 60)

    n = 4
    N = 2 ** n

    # Why: If a state has period r, the QFT will peak at multiples of N/r.
    # This is exactly how Shor's algorithm extracts the period of modular
    # exponentiation — QFT converts spatial periodicity to frequency peaks.
    for period in [2, 4, 8]:
        # Create a periodic state: nonzero at positions 0, r, 2r, ...
        state = np.zeros(N, dtype=complex)
        for k in range(0, N, period):
            state[k] = 1.0
        state = state / np.linalg.norm(state)

        # Apply QFT
        F = qft_matrix(n)
        freq = F @ state
        probs = np.abs(freq) ** 2

        print(f"\n  Period r = {period}:")
        print(f"    Input:  nonzero at positions {list(range(0, N, period))}")
        peaks = [i for i in range(N) if probs[i] > 0.01]
        print(f"    QFT peaks at: {peaks}")
        print(f"    Peak spacing: {N // period} (= N/r = {N}/{period})")
        print(f"    → Estimated period: N/{N // period} = {period}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Quantum Computing — 07: Quantum Fourier Transform     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    np.random.seed(2026)

    demo_qft_matrix()
    demo_qft_vs_fft()
    demo_qft_circuit()
    demo_qft_on_basis_states()
    demo_phase_estimation()
    demo_period_finding_qft()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
