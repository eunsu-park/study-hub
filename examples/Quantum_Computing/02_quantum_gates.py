"""
02_quantum_gates.py — Standard Quantum Gate Matrices and Gate Operations

Demonstrates:
  - Defining all standard single-qubit gates as 2×2 unitary matrices
  - Rotation gates Rx(θ), Ry(θ), Rz(θ) parametrized by angle
  - Applying gates to qubit states and verifying known identities
  - Verifying unitarity (U†U = I) for every gate
  - Constructing controlled gates (CNOT, CZ, controlled-U)
  - Universal gate decomposition: any single-qubit U = Rz Ry Rz (up to global phase)

All computations use pure NumPy.
"""

import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# Computational basis
# ---------------------------------------------------------------------------
KET_0 = np.array([1, 0], dtype=complex)
KET_1 = np.array([0, 1], dtype=complex)
KET_PLUS = (KET_0 + KET_1) / np.sqrt(2)
KET_MINUS = (KET_0 - KET_1) / np.sqrt(2)

# ---------------------------------------------------------------------------
# Single-qubit gate matrices
# ---------------------------------------------------------------------------

# Why: The identity gate does nothing, but it's essential as a placeholder
# when constructing multi-qubit operators via tensor products (Kronecker).
I = np.eye(2, dtype=complex)

# Why: Pauli matrices form a basis for all 2×2 Hermitian matrices and generate
# rotations on the Bloch sphere.  X = bit-flip, Z = phase-flip, Y = iXZ.
X = np.array([[0, 1], [1, 0]], dtype=complex)       # Pauli-X (NOT)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)    # Pauli-Y
Z = np.array([[1, 0], [0, -1]], dtype=complex)       # Pauli-Z

# Why: The Hadamard gate maps between computational (Z) and Hadamard (X) bases.
# It is the workhorse of quantum algorithms — creating superposition from |0⟩.
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Why: S and T gates add phase to |1⟩.  T = π/8 gate.  Together with H, the
# set {H, T} is universal for single-qubit operations (Solovay-Kitaev theorem).
S = np.array([[1, 0], [0, 1j]], dtype=complex)       # Phase gate (√Z)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)  # π/8 gate


def Rx(theta: float) -> np.ndarray:
    """Rotation around X-axis by angle θ.

    Rx(θ) = exp(-iθX/2) = cos(θ/2)I - i·sin(θ/2)X

    Why: Rotation gates parametrize continuous families of unitaries.
    Rx(π) = -iX, so Pauli gates are special cases of rotation gates.
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def Ry(theta: float) -> np.ndarray:
    """Rotation around Y-axis by angle θ.

    Ry(θ) = exp(-iθY/2) = cos(θ/2)I - i·sin(θ/2)Y

    Why: Ry is the only standard rotation that maps real states to real states
    (no imaginary components), making it ideal for variational ansatze.
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def Rz(theta: float) -> np.ndarray:
    """Rotation around Z-axis by angle θ.

    Rz(θ) = exp(-iθZ/2) = diag(e^{-iθ/2}, e^{iθ/2})

    Why: Rz only changes the relative phase between |0⟩ and |1⟩.  This makes
    it diagonal in the computational basis — very efficient on hardware.
    """
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)


# ---------------------------------------------------------------------------
# Two-qubit gate matrices
# ---------------------------------------------------------------------------

# Why: CNOT (controlled-NOT) is the standard entangling gate.  Combined with
# single-qubit rotations, it forms a universal gate set for quantum computing.
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)

# Why: CZ is symmetric — control and target are interchangeable.  This symmetry
# is exploited in many architectures (e.g., Google's Sycamore uses CZ natively).
CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1],
], dtype=complex)

SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=complex)


def controlled_gate(U: np.ndarray) -> np.ndarray:
    """Construct a controlled-U gate from any single-qubit unitary U.

    The 4×4 matrix acts as Identity on the target when control=|0⟩,
    and applies U on the target when control=|1⟩.

    Why: This is the general recipe: C-U = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ U.
    It directly encodes the conditional quantum logic that makes entanglement
    and phase kickback possible.
    """
    proj_0 = np.array([[1, 0], [0, 0]], dtype=complex)  # |0⟩⟨0|
    proj_1 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1⟩⟨1|
    return np.kron(proj_0, I) + np.kron(proj_1, U)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def is_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if a matrix is unitary: U†U = I."""
    product = U.conj().T @ U
    return np.allclose(product, np.eye(len(U)), atol=tol)


def apply_gate(state: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Apply a gate (matrix) to a state vector via matrix-vector multiplication."""
    return gate @ state


def format_state(state: np.ndarray) -> str:
    """Pretty-print a state vector."""
    n_qubits = int(np.log2(len(state)))
    terms = []
    for idx in range(len(state)):
        amp = state[idx]
        if np.abs(amp) < 1e-10:
            continue
        label = format(idx, f'0{n_qubits}b')
        if np.abs(amp.imag) < 1e-10:
            amp_str = f"{amp.real:+.4f}"
        elif np.abs(amp.real) < 1e-10:
            amp_str = f"{amp.imag:+.4f}i"
        else:
            amp_str = f"({amp.real:+.4f}{amp.imag:+.4f}j)"
        terms.append(f"{amp_str}|{label}⟩")
    return " ".join(terms) if terms else "0"


def format_matrix(M: np.ndarray, label: str = "") -> str:
    """Pretty-print a matrix."""
    lines = []
    if label:
        lines.append(f"  {label}:")
    for row in M:
        entries = []
        for val in row:
            if np.abs(val.imag) < 1e-10:
                entries.append(f"{val.real:7.4f}")
            elif np.abs(val.real) < 1e-10:
                entries.append(f"{val.imag:+6.4f}i")
            else:
                entries.append(f"{val.real:+.2f}{val.imag:+.2f}j")
        lines.append("    [" + ", ".join(entries) + "]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_gate_definitions():
    """Show all standard gate matrices."""
    print("=" * 60)
    print("DEMO 1: Standard Quantum Gate Matrices")
    print("=" * 60)

    gates = [
        ("Identity (I)", I), ("Pauli-X", X), ("Pauli-Y", Y), ("Pauli-Z", Z),
        ("Hadamard (H)", H), ("Phase (S)", S), ("T gate", T),
    ]
    for name, gate in gates:
        print(format_matrix(gate, name))
        print()

    print("  Rotation gates (θ = π/4):")
    theta = np.pi / 4
    for name, gate in [("Rx(π/4)", Rx(theta)), ("Ry(π/4)", Ry(theta)), ("Rz(π/4)", Rz(theta))]:
        print(format_matrix(gate, name))
        print()


def demo_gate_application():
    """Apply gates and verify well-known identities."""
    print("=" * 60)
    print("DEMO 2: Gate Application & Identities")
    print("=" * 60)

    # X|0⟩ = |1⟩
    result = apply_gate(KET_0, X)
    print(f"\n  X|0⟩ = {format_state(result)}  (should be |1⟩)")

    # H|0⟩ = |+⟩
    result = apply_gate(KET_0, H)
    print(f"  H|0⟩ = {format_state(result)}  (should be |+⟩)")

    # HH = I (Hadamard is its own inverse)
    result = apply_gate(KET_PLUS, H)
    print(f"  H|+⟩ = {format_state(result)}  (should be |0⟩)")

    # Z|+⟩ = |−⟩
    result = apply_gate(KET_PLUS, Z)
    print(f"  Z|+⟩ = {format_state(result)}  (should be |−⟩)")

    # Why: The identity HZH = X shows that conjugation by H swaps the X and Z bases.
    # This is the Hadamard "change of basis" — a cornerstone of many algorithms.
    HZH = H @ Z @ H
    print(f"\n  HZH ≈ X? {np.allclose(HZH, X)}")
    print(f"  HXH ≈ Z? {np.allclose(H @ X @ H, Z)}")

    # Why: XX = YY = ZZ = I — Pauli gates are involutions (their own inverse).
    # This self-inverse property is unique to Pauli gates among standard gates.
    print(f"  X² = I? {np.allclose(X @ X, I)}")
    print(f"  Y² = I? {np.allclose(Y @ Y, I)}")
    print(f"  Z² = I? {np.allclose(Z @ Z, I)}")

    # S² = Z, T² = S
    print(f"\n  S² = Z? {np.allclose(S @ S, Z)}")
    print(f"  T² = S? {np.allclose(T @ T, S)}")

    # Rotation identities
    # Why: Rx(π) should equal -iX (up to global phase), connecting rotation gates
    # back to the Pauli matrices — every Pauli is a π rotation about its axis.
    print(f"\n  Rx(π) ≈ -iX? {np.allclose(Rx(np.pi), -1j * X)}")
    print(f"  Ry(π) ≈ -iY? {np.allclose(Ry(np.pi), -1j * Y)}")
    print(f"  Rz(π) ≈ -iZ? {np.allclose(Rz(np.pi), -1j * Z)}")


def demo_unitarity():
    """Verify all gates are unitary."""
    print("\n" + "=" * 60)
    print("DEMO 3: Unitarity Verification (U†U = I)")
    print("=" * 60)

    # Why: Unitarity guarantees reversibility and norm preservation — the two
    # fundamental constraints on quantum evolution.  If U†U ≠ I, the gate
    # would create or destroy probability, violating quantum mechanics.
    gates_to_check = [
        ("I", I), ("X", X), ("Y", Y), ("Z", Z),
        ("H", H), ("S", S), ("T", T),
        ("Rx(1.23)", Rx(1.23)), ("Ry(2.71)", Ry(2.71)), ("Rz(0.42)", Rz(0.42)),
        ("CNOT", CNOT), ("CZ", CZ), ("SWAP", SWAP),
    ]

    print(f"\n  {'Gate':<15} {'Unitary?':>10} {'max|U†U - I|':>15}")
    print(f"  {'─' * 42}")
    for name, gate in gates_to_check:
        product = gate.conj().T @ gate
        error = np.max(np.abs(product - np.eye(len(gate))))
        unitary = is_unitary(gate)
        print(f"  {name:<15} {'Yes' if unitary else 'NO':>10} {error:>15.2e}")


def demo_controlled_gates():
    """Demonstrate controlled gate construction."""
    print("\n" + "=" * 60)
    print("DEMO 4: Controlled Gate Construction")
    print("=" * 60)

    # Why: We verify that our general controlled_gate() function reproduces
    # the known CNOT matrix when given X as the target gate.
    CX_constructed = controlled_gate(X)
    print(f"\n  controlled_gate(X) equals CNOT matrix? {np.allclose(CX_constructed, CNOT)}")

    CZ_constructed = controlled_gate(Z)
    print(f"  controlled_gate(Z) equals CZ matrix?   {np.allclose(CZ_constructed, CZ)}")

    # Controlled-S gate
    CS = controlled_gate(S)
    print(f"\n  Controlled-S gate:")
    print(format_matrix(CS, "C-S"))
    print(f"  Is unitary? {is_unitary(CS)}")

    # Why: Controlled-H is less common but illustrates that ANY unitary can be
    # controlled.  C-H creates interesting entangled states from product states.
    CH = controlled_gate(H)
    state_00 = np.kron(KET_0, KET_0)
    state_10 = np.kron(KET_1, KET_0)

    result_00 = CH @ state_00
    result_10 = CH @ state_10
    print(f"\n  C-H|00⟩ = {format_state(result_00)}  (control=0, no action)")
    print(f"  C-H|10⟩ = {format_state(result_10)}  (control=1, apply H to target)")


def demo_two_qubit_operations():
    """Show how to apply gates in multi-qubit systems using Kronecker products."""
    print("\n" + "=" * 60)
    print("DEMO 5: Multi-Qubit Gate Application via Kronecker Products")
    print("=" * 60)

    # Why: In a multi-qubit system, applying gate U to qubit k while leaving
    # others unchanged requires: I ⊗ ... ⊗ U ⊗ ... ⊗ I (tensor product).
    # np.kron implements this tensor product for matrices.

    # H on qubit 0 only (2-qubit system): H ⊗ I
    H_on_q0 = np.kron(H, I)
    state_00 = np.kron(KET_0, KET_0)
    result = H_on_q0 @ state_00
    print(f"\n  (H⊗I)|00⟩ = {format_state(result)}")
    print(f"  Expected: (1/√2)(|00⟩ + |10⟩)")

    # H on qubit 1 only: I ⊗ H
    H_on_q1 = np.kron(I, H)
    result = H_on_q1 @ state_00
    print(f"\n  (I⊗H)|00⟩ = {format_state(result)}")
    print(f"  Expected: (1/√2)(|00⟩ + |01⟩)")

    # Create Bell state: CNOT · (H⊗I)|00⟩
    bell = CNOT @ (H_on_q0 @ state_00)
    print(f"\n  CNOT·(H⊗I)|00⟩ = {format_state(bell)}")
    print(f"  Expected: (1/√2)(|00⟩ + |11⟩)  — Bell state |Φ+⟩")


def demo_universal_decomposition():
    """Decompose an arbitrary unitary into Rz·Ry·Rz (ZYZ decomposition)."""
    print("\n" + "=" * 60)
    print("DEMO 6: Universal Gate Decomposition (ZYZ)")
    print("=" * 60)

    # Why: The Euler ZYZ decomposition proves that {Ry, Rz} are universal for
    # single-qubit gates.  Any U ∈ SU(2) can be written as:
    #     U = e^{iδ} · Rz(α) · Ry(β) · Rz(γ)
    # This is the quantum analog of Euler angles for 3D rotations.

    def zyz_decompose(U: np.ndarray) -> Tuple[float, float, float, float]:
        """Decompose U into e^{iδ} Rz(α) Ry(β) Rz(γ).

        Returns (delta, alpha, beta, gamma).
        """
        # Extract parameters from the unitary matrix elements
        # U = e^{iδ} [[e^{-i(α+γ)/2} cos(β/2), -e^{-i(α-γ)/2} sin(β/2)],
        #              [e^{i(α-γ)/2}  sin(β/2),  e^{i(α+γ)/2}  cos(β/2)]]

        # Why: We compute β from the absolute values (which cancel the phase δ),
        # then extract α and γ from the argument of specific matrix elements.
        det_U = np.linalg.det(U)
        delta = np.angle(det_U) / 2  # global phase from det = e^{2iδ}

        # Remove global phase
        V = U * np.exp(-1j * delta)

        beta = 2 * np.arccos(np.clip(np.abs(V[0, 0]), 0, 1))

        if np.abs(np.sin(beta / 2)) < 1e-10:
            # β ≈ 0 → U ≈ e^{iδ} Rz(α+γ); split arbitrarily
            alpha = np.angle(V[0, 0]) + np.angle(V[1, 1])
            gamma = 0.0
        elif np.abs(np.cos(beta / 2)) < 1e-10:
            # β ≈ π → cos term vanishes
            alpha = np.angle(V[1, 0]) - np.angle(V[0, 1])
            gamma = 0.0
        else:
            sum_ag = -np.angle(V[0, 0]) * 2  # -(α+γ)/2 * 2
            diff_ag = np.angle(V[1, 0]) * 2  # (α-γ)/2 * 2
            # Wait — let's be more careful:
            # V[0,0] = e^{-i(α+γ)/2} cos(β/2)
            # V[1,0] = e^{+i(α-γ)/2} sin(β/2)
            phase_00 = np.angle(V[0, 0] / np.cos(beta / 2))  # = -(α+γ)/2
            phase_10 = np.angle(V[1, 0] / np.sin(beta / 2))  # = +(α-γ)/2
            alpha = -phase_00 + phase_10
            gamma = -phase_00 - phase_10

        return (delta, alpha, beta, gamma)

    def reconstruct_zyz(delta, alpha, beta, gamma):
        """Reconstruct U from ZYZ parameters."""
        return np.exp(1j * delta) * Rz(alpha) @ Ry(beta) @ Rz(gamma)

    # Test with several gates
    test_gates = [
        ("Hadamard (H)", H),
        ("Pauli-X", X),
        ("T gate", T),
        ("Rx(1.5)", Rx(1.5)),
        ("Arbitrary", Ry(0.7) @ Rz(1.2) @ Rx(0.3)),
    ]

    print(f"\n  {'Gate':<20} {'δ':>7} {'α':>7} {'β':>7} {'γ':>7} {'Reconstructed ≈ Original?':>28}")
    print(f"  {'─' * 78}")

    for name, gate in test_gates:
        delta, alpha, beta, gamma = zyz_decompose(gate)
        recon = reconstruct_zyz(delta, alpha, beta, gamma)
        match = np.allclose(recon, gate)
        print(f"  {name:<20} {delta:>7.3f} {alpha:>7.3f} {beta:>7.3f} {gamma:>7.3f} "
              f"{'Yes' if match else 'NO':>28}")

    print("\n  Every single-qubit gate decomposes into Rz·Ry·Rz (plus global phase).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Quantum Computing — 02: Quantum Gates                 ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_gate_definitions()
    demo_gate_application()
    demo_unitarity()
    demo_controlled_gates()
    demo_two_qubit_operations()
    demo_universal_decomposition()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
