"""
21_topological_qc.py — Topological Quantum Computing

Demonstrates:
  - Anyon braiding: Fibonacci anyons and braid group representations
  - Abelian vs non-Abelian exchange statistics
  - Surface code: stabilizer construction and syndrome extraction
  - Logical qubit encoding in the surface code
  - Error detection and correction on the surface code
  - Topological protection: error threshold analysis

All computations use pure NumPy.
"""

import numpy as np
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Pauli matrices
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


# ---------------------------------------------------------------------------
# Fibonacci Anyons and Braiding
# ---------------------------------------------------------------------------

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


def fibonacci_f_matrix() -> np.ndarray:
    """The F-matrix (associativity matrix) for Fibonacci anyons.

    Why: Fibonacci anyons have two fusion channels: τ × τ = 1 + τ.
    The F-matrix encodes how different fusion orderings relate to each
    other.  It is the fundamental building block for braid matrices.
    """
    return np.array([
        [1.0 / PHI, np.sqrt(1.0 / PHI)],
        [np.sqrt(1.0 / PHI), -1.0 / PHI],
    ], dtype=complex)


def fibonacci_r_matrix() -> Tuple[complex, complex]:
    """The R-matrix eigenvalues for Fibonacci anyons.

    Why: When two anyons are exchanged (braided), the state acquires a
    phase determined by the R-matrix.  For Fibonacci anyons:
    R_1 = e^{-4πi/5} (trivial channel), R_τ = e^{3πi/5} (τ channel).
    """
    R_1 = np.exp(-4j * np.pi / 5)
    R_tau = np.exp(3j * np.pi / 5)
    return R_1, R_tau


def fibonacci_braid_matrix(sigma_idx: int) -> np.ndarray:
    """Compute the braid matrix for Fibonacci anyons with 3 anyons.

    Why: With 3 Fibonacci anyons, the fusion space is 2-dimensional
    (the two fusion trees for τ⊗τ⊗τ → τ).  Braiding the i-th and
    (i+1)-th anyons acts as a 2×2 unitary on this space.  These
    braiding matrices are dense (non-diagonal for non-Abelian anyons)
    and can approximate any single-qubit gate to arbitrary precision.
    """
    F = fibonacci_f_matrix()
    R_1, R_tau = fibonacci_r_matrix()
    R_diag = np.diag([R_1, R_tau])

    if sigma_idx == 1:
        # Braid anyons 1 and 2: σ₁ = R (diagonal in standard basis)
        return R_diag
    elif sigma_idx == 2:
        # Braid anyons 2 and 3: σ₂ = F⁻¹ R F
        F_inv = np.linalg.inv(F)
        return F_inv @ R_diag @ F
    else:
        raise ValueError(f"Invalid braid index: {sigma_idx}")


def compile_braid_sequence(target_gate: np.ndarray,
                           max_length: int = 8) -> Tuple[List[int], float]:
    """Find a braid sequence that approximates a target single-qubit gate.

    Why: Fibonacci anyons are computationally universal — any unitary can
    be approximated by a sequence of braids.  The Solovay-Kitaev theorem
    guarantees that O(log^c(1/ε)) braids suffice for precision ε.
    This brute-force search demonstrates the principle for short sequences.
    """
    sigma1 = fibonacci_braid_matrix(1)
    sigma2 = fibonacci_braid_matrix(2)
    sigma1_inv = sigma1.conj().T
    sigma2_inv = sigma2.conj().T

    generators = {
        1: sigma1, -1: sigma1_inv,
        2: sigma2, -2: sigma2_inv,
    }

    best_fidelity = 0.0
    best_sequence = []

    # Brute-force search over short braid words
    def search(seq: List[int], U_current: np.ndarray, depth: int):
        nonlocal best_fidelity, best_sequence

        # Gate fidelity: |Tr(U† V)|² / d²
        fid = float(np.abs(np.trace(target_gate.conj().T @ U_current)) ** 2) / 4.0
        if fid > best_fidelity:
            best_fidelity = fid
            best_sequence = seq.copy()

        if depth >= max_length:
            return

        for gen_idx in [1, -1, 2, -2]:
            # Avoid immediate cancellation
            if seq and gen_idx == -seq[-1]:
                continue
            new_U = generators[gen_idx] @ U_current
            seq.append(gen_idx)
            search(seq, new_U, depth + 1)
            seq.pop()

    search([], np.eye(2, dtype=complex), 0)
    return best_sequence, best_fidelity


# ---------------------------------------------------------------------------
# Surface Code
# ---------------------------------------------------------------------------

def build_surface_code(d: int) -> Tuple[List[np.ndarray], List[np.ndarray],
                                         int]:
    """Build stabilizer generators for a d×d surface code.

    Why: The surface code is the most practical topological error-correcting
    code.  It encodes 1 logical qubit in d² physical qubits with distance d,
    meaning it can correct ⌊(d-1)/2⌋ errors.  Stabilizers are products of
    X or Z on qubits around each face (X-stabilizer) or vertex (Z-stabilizer).

    Returns (x_stabilizers, z_stabilizers, n_data_qubits).
    We work with simplified stabilizer checks as binary vectors.
    """
    n_qubits = d * d

    # Qubit layout: d×d grid, qubit at (row, col) has index row*d + col
    x_stabs = []  # Plaquette (face) stabilizers
    z_stabs = []  # Vertex stabilizers

    # X-stabilizers: one per face of the grid
    for row in range(d - 1):
        for col in range(d - 1):
            support = np.zeros(n_qubits, dtype=int)
            support[row * d + col] = 1
            support[row * d + (col + 1)] = 1
            support[(row + 1) * d + col] = 1
            support[(row + 1) * d + (col + 1)] = 1
            x_stabs.append(support)

    # Z-stabilizers: one per vertex (interior)
    # Simplified: each Z stabilizer acts on qubits sharing an edge
    for row in range(1, d):
        for col in range(1, d):
            support = np.zeros(n_qubits, dtype=int)
            support[row * d + col] = 1
            support[(row - 1) * d + col] = 1
            support[row * d + (col - 1)] = 1
            if row + 1 < d:
                support[(row + 1) * d + col] = 1
            if col + 1 < d:
                support[row * d + (col + 1)] = 1
            z_stabs.append(support)

    return x_stabs, z_stabs, n_qubits


def surface_code_syndrome(error_pattern: np.ndarray,
                          stabilizers: List[np.ndarray]) -> np.ndarray:
    """Compute the syndrome for a given error pattern.

    Why: The syndrome is the pattern of stabilizer measurement outcomes
    that reveals where errors occurred without revealing the encoded
    logical information.  This is the key to topological protection:
    errors are detected by their boundary (syndrome), not their bulk.
    """
    syndrome = np.zeros(len(stabilizers), dtype=int)
    for i, stab in enumerate(stabilizers):
        # Syndrome bit = parity of overlap between error and stabilizer
        syndrome[i] = np.sum(error_pattern * stab) % 2
    return syndrome


def minimum_weight_decoder(syndrome: np.ndarray,
                           stabilizers: List[np.ndarray],
                           n_qubits: int) -> np.ndarray:
    """Simple minimum-weight decoder for the surface code.

    Why: The decoder maps a syndrome to a correction.  The minimum-weight
    decoder finds the lowest-weight error consistent with the syndrome.
    Real decoders (MWPM, union-find) are more sophisticated, but this
    greedy approach demonstrates the concept.
    """
    correction = np.zeros(n_qubits, dtype=int)
    remaining_syndrome = syndrome.copy()

    # Greedy: for each non-zero syndrome bit, flip the qubit with
    # maximum overlap with unsatisfied stabilizers
    while np.any(remaining_syndrome):
        best_qubit = -1
        best_score = -1
        for q in range(n_qubits):
            score = 0
            for i, stab in enumerate(stabilizers):
                if remaining_syndrome[i] and stab[q]:
                    score += 1
            if score > best_score:
                best_score = score
                best_qubit = q

        if best_qubit < 0 or best_score == 0:
            break

        correction[best_qubit] ^= 1
        for i, stab in enumerate(stabilizers):
            if stab[best_qubit]:
                remaining_syndrome[i] ^= 1

    return correction


# ---------------------------------------------------------------------------
# Error threshold analysis
# ---------------------------------------------------------------------------

def surface_code_error_rate(d: int, physical_error: float,
                            n_trials: int = 500) -> float:
    """Estimate logical error rate for a surface code via Monte Carlo.

    Why: The surface code has a threshold error rate p_th ≈ 1%.  Below
    this threshold, increasing the code distance d exponentially suppresses
    the logical error rate: p_L ~ (p/p_th)^{d/2}.  Above threshold,
    larger codes perform worse.
    """
    x_stabs, z_stabs, n_qubits = build_surface_code(d)

    logical_errors = 0
    for _ in range(n_trials):
        # Random X errors with probability physical_error
        error = (np.random.random(n_qubits) < physical_error).astype(int)

        # Compute syndrome using Z stabilizers (detects X errors)
        syndrome = surface_code_syndrome(error, z_stabs)

        # Decode
        correction = minimum_weight_decoder(syndrome, z_stabs, n_qubits)

        # Residual error after correction
        residual = (error + correction) % 2

        # Check if residual is a logical error (spans the code)
        # Simplified: logical error if residual has odd weight on first row
        if np.sum(residual[:d]) % 2 == 1:
            logical_errors += 1

    return logical_errors / n_trials


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_fibonacci_braiding():
    """Show Fibonacci anyon braiding matrices."""
    print("=" * 60)
    print("DEMO 1: Fibonacci Anyon Braiding")
    print("=" * 60)

    F = fibonacci_f_matrix()
    R_1, R_tau = fibonacci_r_matrix()

    print(f"\n  Golden ratio φ = {PHI:.6f}")
    print(f"\n  F-matrix (fusion basis change):")
    for i in range(2):
        print(f"    [{F[i, 0].real:+.6f}, {F[i, 1].real:+.6f}]")

    print(f"\n  R-matrix eigenvalues:")
    print(f"    R_1 = e^{{-4πi/5}} = {R_1.real:.4f} {R_1.imag:+.4f}i")
    print(f"    R_τ = e^{{3πi/5}}  = {R_tau.real:.4f} {R_tau.imag:+.4f}i")

    sigma1 = fibonacci_braid_matrix(1)
    sigma2 = fibonacci_braid_matrix(2)

    print(f"\n  Braid matrix σ₁ (exchange anyons 1,2):")
    for i in range(2):
        print(f"    [{sigma1[i, 0].real:+.4f}{sigma1[i, 0].imag:+.4f}i, "
              f"{sigma1[i, 1].real:+.4f}{sigma1[i, 1].imag:+.4f}i]")

    print(f"\n  Braid matrix σ₂ (exchange anyons 2,3):")
    for i in range(2):
        print(f"    [{sigma2[i, 0].real:+.4f}{sigma2[i, 0].imag:+.4f}i, "
              f"{sigma2[i, 1].real:+.4f}{sigma2[i, 1].imag:+.4f}i]")

    # Verify unitarity
    print(f"\n  σ₁ unitary: {np.allclose(sigma1 @ sigma1.conj().T, I)}")
    print(f"  σ₂ unitary: {np.allclose(sigma2 @ sigma2.conj().T, I)}")

    # Verify braid relation: σ₁σ₂σ₁ = σ₂σ₁σ₂
    lhs = sigma1 @ sigma2 @ sigma1
    rhs = sigma2 @ sigma1 @ sigma2
    print(f"  Braid relation σ₁σ₂σ₁ = σ₂σ₁σ₂: {np.allclose(lhs, rhs)}")


def demo_braid_compilation():
    """Compile target gates from braid sequences."""
    print("\n" + "=" * 60)
    print("DEMO 2: Braid Compilation (Gate Approximation)")
    print("=" * 60)

    # Why: Any single-qubit gate can be approximated by braiding Fibonacci
    # anyons.  Longer braid sequences yield higher precision.

    targets = {
        "Pauli-X (π rotation)": np.array([[0, 1], [1, 0]], dtype=complex),
        "Hadamard-like": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
        "T-gate-like": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
    }

    for name, target in targets.items():
        seq, fid = compile_braid_sequence(target, max_length=6)
        braid_str = "".join([f"σ{abs(s)}{'⁻¹' if s < 0 else ''}" for s in seq])
        print(f"\n  Target: {name}")
        print(f"    Best sequence (len {len(seq)}): {braid_str}")
        print(f"    Gate fidelity: {fid:.6f}")


def demo_surface_code():
    """Build and test a surface code."""
    print("\n" + "=" * 60)
    print("DEMO 3: Surface Code Construction")
    print("=" * 60)

    for d in [3, 5]:
        x_stabs, z_stabs, n_qubits = build_surface_code(d)
        print(f"\n  Distance-{d} surface code:")
        print(f"    Data qubits: {n_qubits}")
        print(f"    X-stabilizers: {len(x_stabs)}")
        print(f"    Z-stabilizers: {len(z_stabs)}")
        print(f"    Correctable errors: {(d - 1) // 2}")

        # Test syndrome for a single error
        for err_pos in [0, d // 2 * d + d // 2, n_qubits - 1]:
            error = np.zeros(n_qubits, dtype=int)
            error[err_pos] = 1
            syndrome = surface_code_syndrome(error, z_stabs)
            n_triggered = np.sum(syndrome)
            print(f"    Single X error at qubit {err_pos}: "
                  f"{n_triggered} stabilizers triggered")


def demo_error_correction():
    """Demonstrate error detection and correction on surface code."""
    print("\n" + "=" * 60)
    print("DEMO 4: Surface Code Error Correction")
    print("=" * 60)

    d = 5
    x_stabs, z_stabs, n_qubits = build_surface_code(d)

    # Why: The surface code corrects up to ⌊(d-1)/2⌋ = 2 errors for d=5.
    # We test with 1 and 2 errors (should correct) and 3 errors (may fail).

    print(f"\n  Distance-{d} code (corrects up to {(d-1)//2} errors):")

    for n_errors in [1, 2, 3]:
        n_success = 0
        n_trials = 200

        for _ in range(n_trials):
            # Random error positions
            positions = np.random.choice(n_qubits, size=n_errors, replace=False)
            error = np.zeros(n_qubits, dtype=int)
            error[positions] = 1

            syndrome = surface_code_syndrome(error, z_stabs)
            correction = minimum_weight_decoder(syndrome, z_stabs, n_qubits)
            residual = (error + correction) % 2

            # Success if residual has no logical effect
            if np.sum(residual[:d]) % 2 == 0:
                n_success += 1

        rate = n_success / n_trials
        print(f"    {n_errors} error(s): correction success = {rate:.2%} "
              f"({n_success}/{n_trials})")


def demo_error_threshold():
    """Analyze the surface code error threshold."""
    print("\n" + "=" * 60)
    print("DEMO 5: Error Threshold Analysis")
    print("=" * 60)

    # Why: Below the threshold p_th, increasing code distance exponentially
    # suppresses the logical error rate.  Above threshold, larger codes
    # perform worse.  This is the fundamental reason topological codes work.

    print(f"\n  {'p_phys':>8}", end="")
    for d in [3, 5, 7]:
        print(f"  {'d=' + str(d):>10}", end="")
    print()
    print(f"  {'─' * 42}")

    for p in [0.001, 0.003, 0.005, 0.008, 0.01, 0.02, 0.03, 0.05, 0.1]:
        print(f"  {p:8.3f}", end="")
        for d in [3, 5, 7]:
            p_L = surface_code_error_rate(d, p, n_trials=300)
            print(f"  {p_L:10.4f}", end="")
        print()

    print(f"\n  Threshold: where larger d stops helping (~1-3%)")
    print(f"  Below threshold: p_L ~ (p/p_th)^{{d/2}} → exponential suppression")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("+" + "=" * 58 + "+")
    print("|   Quantum Computing - 21: Topological Quantum Computing    |")
    print("+" + "=" * 58 + "+")

    np.random.seed(2026)

    demo_fibonacci_braiding()
    demo_braid_compilation()
    demo_surface_code()
    demo_error_correction()
    demo_error_threshold()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
