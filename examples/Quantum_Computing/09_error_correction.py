"""
09_error_correction.py — Quantum Error Correction

Demonstrates:
  - 3-qubit bit-flip code: encode, error, syndrome, correct
  - 3-qubit phase-flip code: similar but for phase errors
  - Shor's 9-qubit code: handles arbitrary single-qubit errors
  - Syndrome measurement without disturbing the encoded state
  - Error detection vs error correction

All computations use pure NumPy.
"""

import numpy as np
from typing import Tuple, List, Dict

# ---------------------------------------------------------------------------
# Basic gates and states
# ---------------------------------------------------------------------------
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

KET_0 = np.array([1, 0], dtype=complex)
KET_1 = np.array([0, 1], dtype=complex)

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)


def tensor(*matrices: np.ndarray) -> np.ndarray:
    """Compute tensor product of multiple matrices."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray,
                             target: int, n_qubits: int) -> np.ndarray:
    """Apply a single-qubit gate to the target qubit in an n-qubit state."""
    ops = [I] * n_qubits
    ops[target] = gate
    full_op = ops[0]
    for op in ops[1:]:
        full_op = np.kron(full_op, op)
    return full_op @ state


def apply_cnot(state: np.ndarray, control: int, target: int,
               n_qubits: int) -> np.ndarray:
    """Apply CNOT between control and target qubits."""
    proj_0 = np.array([[1, 0], [0, 0]], dtype=complex)
    proj_1 = np.array([[0, 0], [0, 1]], dtype=complex)

    ops_0 = [I] * n_qubits
    ops_0[control] = proj_0
    term_0 = ops_0[0]
    for op in ops_0[1:]:
        term_0 = np.kron(term_0, op)

    ops_1 = [I] * n_qubits
    ops_1[control] = proj_1
    ops_1[target] = X
    term_1 = ops_1[0]
    for op in ops_1[1:]:
        term_1 = np.kron(term_1, op)

    return (term_0 + term_1) @ state


def format_state(state: np.ndarray, n_qubits: int) -> str:
    terms = []
    for idx in range(len(state)):
        amp = state[idx]
        if np.abs(amp) < 1e-10:
            continue
        label = format(idx, f'0{n_qubits}b')
        if np.abs(amp.imag) < 1e-10:
            amp_str = f"{amp.real:+.4f}"
        else:
            amp_str = f"({amp.real:+.4f}{amp.imag:+.4f}j)"
        terms.append(f"{amp_str}|{label}⟩")
    return " ".join(terms) if terms else "0"


# ---------------------------------------------------------------------------
# 3-Qubit Bit-Flip Code
# ---------------------------------------------------------------------------

def bitflip_encode(alpha: complex, beta: complex) -> np.ndarray:
    """Encode a logical qubit α|0⟩ + β|1⟩ using the 3-qubit bit-flip code.

    |0_L⟩ = |000⟩
    |1_L⟩ = |111⟩
    α|0⟩ + β|1⟩ → α|000⟩ + β|111⟩

    Why: The bit-flip code is the simplest quantum error correction code.
    It protects against X (bit-flip) errors on any single qubit by encoding
    each logical qubit into 3 physical qubits.  This is the quantum analog
    of classical repetition codes — but the no-cloning theorem means we
    can't simply copy the qubit; we must use entanglement.
    """
    n_qubits = 3
    # Start with α|0⟩ + β|1⟩ on qubit 0, |00⟩ on qubits 1,2
    state = np.kron(np.array([alpha, beta], dtype=complex), np.array([1, 0, 0, 0], dtype=complex))

    # CNOT from qubit 0 to qubit 1
    state = apply_cnot(state, 0, 1, n_qubits)
    # CNOT from qubit 0 to qubit 2
    state = apply_cnot(state, 0, 2, n_qubits)

    return state


def bitflip_syndrome(state: np.ndarray) -> Tuple[int, int]:
    """Measure the bit-flip syndrome without collapsing the logical state.

    Syndrome bits:
        s1 = parity(qubit 0, qubit 1)  — measures Z⊗Z⊗I
        s2 = parity(qubit 1, qubit 2)  — measures I⊗Z⊗Z

    Why: Syndrome measurement is the key insight of quantum error correction.
    We measure PARITY between qubits, not individual qubits.  This reveals
    which qubit (if any) was flipped, without revealing the encoded information
    (α and β remain undisturbed).
    """
    n_qubits = 3

    # Why: The syndrome operators Z⊗Z⊗I and I⊗Z⊗Z have eigenvalues ±1.
    # +1 means the two qubits agree (same parity), -1 means they disagree.
    ZZI = tensor(Z, Z, I)
    IZZ = tensor(I, Z, Z)

    s1_exp = np.real(state.conj() @ ZZI @ state)
    s2_exp = np.real(state.conj() @ IZZ @ state)

    # Convert expectation to syndrome bit: +1 → 0 (no error), -1 → 1 (error)
    s1 = 0 if s1_exp > 0 else 1
    s2 = 0 if s2_exp > 0 else 1

    return (s1, s2)


def bitflip_correct(state: np.ndarray, syndrome: Tuple[int, int]) -> np.ndarray:
    """Apply correction based on syndrome.

    Syndrome → Error location:
        (0,0) → no error
        (1,0) → qubit 0 flipped
        (1,1) → qubit 1 flipped
        (0,1) → qubit 2 flipped

    Why: The syndrome uniquely identifies which qubit was flipped (or none).
    Correction is simply applying X to the identified qubit — undoing the
    bit flip.  This works because X² = I (flipping twice returns to original).
    """
    s1, s2 = syndrome
    n_qubits = 3

    if (s1, s2) == (0, 0):
        return state  # No error
    elif (s1, s2) == (1, 0):
        return apply_single_qubit_gate(state, X, 0, n_qubits)
    elif (s1, s2) == (1, 1):
        return apply_single_qubit_gate(state, X, 1, n_qubits)
    elif (s1, s2) == (0, 1):
        return apply_single_qubit_gate(state, X, 2, n_qubits)
    return state


# ---------------------------------------------------------------------------
# 3-Qubit Phase-Flip Code
# ---------------------------------------------------------------------------

def phaseflip_encode(alpha: complex, beta: complex) -> np.ndarray:
    """Encode using the 3-qubit phase-flip code.

    |0_L⟩ = |+++⟩
    |1_L⟩ = |−−−⟩

    Why: The phase-flip code protects against Z errors.  It works by encoding
    in the Hadamard basis — since HZH = X, a phase flip in the computational
    basis becomes a bit flip in the Hadamard basis, which we can correct.
    This illustrates the deep symmetry between bit flips and phase flips.
    """
    n_qubits = 3
    # First do bit-flip encoding
    state = bitflip_encode(alpha, beta)
    # Then apply H to all qubits to switch to Hadamard basis
    for q in range(n_qubits):
        state = apply_single_qubit_gate(state, H, q, n_qubits)
    return state


def phaseflip_syndrome(state: np.ndarray) -> Tuple[int, int]:
    """Measure phase-flip syndrome using X⊗X stabilizers.

    Why: In the Hadamard basis, Z errors become X errors.  The stabilizers
    for the phase-flip code are X⊗X⊗I and I⊗X⊗X (instead of Z⊗Z for bit-flip).
    """
    XXI = tensor(X, X, I)
    IXX = tensor(I, X, X)

    s1_exp = np.real(state.conj() @ XXI @ state)
    s2_exp = np.real(state.conj() @ IXX @ state)

    s1 = 0 if s1_exp > 0 else 1
    s2 = 0 if s2_exp > 0 else 1

    return (s1, s2)


def phaseflip_correct(state: np.ndarray, syndrome: Tuple[int, int]) -> np.ndarray:
    """Correct phase-flip error based on syndrome."""
    s1, s2 = syndrome
    n_qubits = 3

    if (s1, s2) == (0, 0):
        return state
    elif (s1, s2) == (1, 0):
        return apply_single_qubit_gate(state, Z, 0, n_qubits)
    elif (s1, s2) == (1, 1):
        return apply_single_qubit_gate(state, Z, 1, n_qubits)
    elif (s1, s2) == (0, 1):
        return apply_single_qubit_gate(state, Z, 2, n_qubits)
    return state


# ---------------------------------------------------------------------------
# Shor's 9-Qubit Code
# ---------------------------------------------------------------------------

def shor_encode(alpha: complex, beta: complex) -> np.ndarray:
    """Encode using Shor's 9-qubit code.

    |0_L⟩ = (|000⟩+|111⟩)(|000⟩+|111⟩)(|000⟩+|111⟩) / 2√2
    |1_L⟩ = (|000⟩-|111⟩)(|000⟩-|111⟩)(|000⟩-|111⟩) / 2√2

    Why: Shor's code is a concatenation of the phase-flip code (outer) and
    the bit-flip code (inner).  This two-level encoding corrects BOTH bit-flip
    AND phase-flip errors — and since any single-qubit error can be decomposed
    into I, X, Y, Z, this means it corrects ANY single-qubit error.
    This was the first quantum error correcting code ever discovered (1995).
    """
    n_qubits = 9

    # Build the logical basis states directly
    # Why: We construct |0_L⟩ and |1_L⟩ as 512-element vectors.
    # Each is a tensor product of three (|000⟩±|111⟩)/√2 blocks.
    plus_block = (np.kron(np.kron(KET_0, KET_0), KET_0) +
                  np.kron(np.kron(KET_1, KET_1), KET_1)) / np.sqrt(2)
    minus_block = (np.kron(np.kron(KET_0, KET_0), KET_0) -
                   np.kron(np.kron(KET_1, KET_1), KET_1)) / np.sqrt(2)

    ket_0L = np.kron(np.kron(plus_block, plus_block), plus_block)
    ket_1L = np.kron(np.kron(minus_block, minus_block), minus_block)

    # Encode: α|0_L⟩ + β|1_L⟩
    state = alpha * ket_0L + beta * ket_1L
    return state


def shor_syndrome(state: np.ndarray) -> Dict[str, int]:
    """Compute syndrome for Shor's 9-qubit code.

    Why: The syndrome has two parts:
    1. Bit-flip detection within each block (qubits 0-2, 3-5, 6-8)
       using Z⊗Z on adjacent pairs within blocks
    2. Phase-flip detection between blocks
       using X^⊗3 ⊗ X^⊗3 on adjacent blocks
    """
    n = 9
    syndrome = {}

    # Bit-flip syndromes within each block of 3
    for block in range(3):
        base = block * 3
        # Z_base Z_{base+1}
        ops1 = [I] * n
        ops1[base] = Z
        ops1[base + 1] = Z
        stab1 = ops1[0]
        for op in ops1[1:]:
            stab1 = np.kron(stab1, op)

        # Z_{base+1} Z_{base+2}
        ops2 = [I] * n
        ops2[base + 1] = Z
        ops2[base + 2] = Z
        stab2 = ops2[0]
        for op in ops2[1:]:
            stab2 = np.kron(stab2, op)

        s1 = 0 if np.real(state.conj() @ stab1 @ state) > 0 else 1
        s2 = 0 if np.real(state.conj() @ stab2 @ state) > 0 else 1

        syndrome[f'block{block}_s1'] = s1
        syndrome[f'block{block}_s2'] = s2

    # Phase-flip syndromes between blocks
    # X^⊗3 on block 0 ⊗ X^⊗3 on block 1
    ops_p1 = [X, X, X, X, X, X, I, I, I]
    stab_p1 = ops_p1[0]
    for op in ops_p1[1:]:
        stab_p1 = np.kron(stab_p1, op)

    # X^⊗3 on block 1 ⊗ X^⊗3 on block 2
    ops_p2 = [I, I, I, X, X, X, X, X, X]
    stab_p2 = ops_p2[0]
    for op in ops_p2[1:]:
        stab_p2 = np.kron(stab_p2, op)

    syndrome['phase_s1'] = 0 if np.real(state.conj() @ stab_p1 @ state) > 0 else 1
    syndrome['phase_s2'] = 0 if np.real(state.conj() @ stab_p2 @ state) > 0 else 1

    return syndrome


def shor_correct(state: np.ndarray, syndrome: Dict[str, int]) -> np.ndarray:
    """Correct errors based on Shor code syndrome."""
    n = 9

    # First correct bit flips within each block
    for block in range(3):
        base = block * 3
        s1 = syndrome[f'block{block}_s1']
        s2 = syndrome[f'block{block}_s2']

        if (s1, s2) == (1, 0):
            state = apply_single_qubit_gate(state, X, base, n)
        elif (s1, s2) == (1, 1):
            state = apply_single_qubit_gate(state, X, base + 1, n)
        elif (s1, s2) == (0, 1):
            state = apply_single_qubit_gate(state, X, base + 2, n)

    # Then correct phase flips between blocks
    ps1 = syndrome['phase_s1']
    ps2 = syndrome['phase_s2']

    # Why: Phase correction applies Z to ALL qubits in the affected block.
    # This is because the phase-flip code uses blocks of 3, and Z on any qubit
    # in the block has the same effect on the logical state.
    if (ps1, ps2) == (1, 0):
        for q in range(0, 3):
            state = apply_single_qubit_gate(state, Z, q, n)
    elif (ps1, ps2) == (1, 1):
        for q in range(3, 6):
            state = apply_single_qubit_gate(state, Z, q, n)
    elif (ps1, ps2) == (0, 1):
        for q in range(6, 9):
            state = apply_single_qubit_gate(state, Z, q, n)

    return state


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_bitflip_code():
    """Demonstrate the 3-qubit bit-flip code."""
    print("=" * 60)
    print("DEMO 1: 3-Qubit Bit-Flip Code")
    print("=" * 60)

    alpha, beta = 1 / np.sqrt(3), np.sqrt(2 / 3)
    print(f"\n  Logical qubit: {alpha:.4f}|0⟩ + {beta:.4f}|1⟩")

    encoded = bitflip_encode(alpha, beta)
    print(f"  Encoded:       {format_state(encoded, 3)}")

    # No error
    syn = bitflip_syndrome(encoded)
    print(f"\n  No error — syndrome: {syn} → no correction needed")

    # Introduce errors on each qubit
    for target in range(3):
        errored = apply_single_qubit_gate(encoded.copy(), X, target, 3)
        syn = bitflip_syndrome(errored)
        corrected = bitflip_correct(errored, syn)
        match = np.allclose(corrected, encoded)

        print(f"\n  X error on qubit {target}:")
        print(f"    Errored:   {format_state(errored, 3)}")
        print(f"    Syndrome:  {syn}")
        print(f"    Corrected: {format_state(corrected, 3)}")
        print(f"    Matches original? {match}")


def demo_phaseflip_code():
    """Demonstrate the 3-qubit phase-flip code."""
    print("\n" + "=" * 60)
    print("DEMO 2: 3-Qubit Phase-Flip Code")
    print("=" * 60)

    alpha, beta = 1 / np.sqrt(2), 1 / np.sqrt(2)
    print(f"\n  Logical qubit: {alpha:.4f}|0⟩ + {beta:.4f}|1⟩")

    encoded = phaseflip_encode(alpha, beta)
    print(f"  Encoded (in Hadamard basis, 8 amplitudes):")
    # Only show nonzero amplitudes
    nonzero_count = np.sum(np.abs(encoded) > 1e-10)
    print(f"    ({nonzero_count} nonzero amplitudes)")

    # No error
    syn = phaseflip_syndrome(encoded)
    print(f"\n  No error — syndrome: {syn}")

    # Introduce Z errors
    for target in range(3):
        errored = apply_single_qubit_gate(encoded.copy(), Z, target, 3)
        syn = phaseflip_syndrome(errored)
        corrected = phaseflip_correct(errored, syn)
        match = np.allclose(corrected, encoded)

        print(f"\n  Z error on qubit {target}:")
        print(f"    Syndrome:  {syn}")
        print(f"    Corrected matches original? {match}")


def demo_shor_code():
    """Demonstrate Shor's 9-qubit code."""
    print("\n" + "=" * 60)
    print("DEMO 3: Shor's 9-Qubit Code (Corrects Any Single-Qubit Error)")
    print("=" * 60)

    alpha = np.cos(np.pi / 5)
    beta = np.sin(np.pi / 5) * np.exp(1j * np.pi / 7)
    print(f"\n  Logical qubit: α|0⟩ + β|1⟩")
    print(f"    α = {alpha:.4f}, β = {beta:.4f}")

    encoded = shor_encode(alpha, beta)
    norm = np.linalg.norm(encoded)
    print(f"  Encoded state norm: {norm:.6f}")

    # Test all three error types on several qubits
    print(f"\n  Testing error correction (each error on qubit 4):")
    error_types = [("X (bit-flip)", X), ("Z (phase-flip)", Z), ("Y (both)", Y)]

    for error_name, error_gate in error_types:
        errored = apply_single_qubit_gate(encoded.copy(), error_gate, 4, 9)
        syn = shor_syndrome(errored)
        corrected = shor_correct(errored, syn)
        match = np.allclose(corrected, encoded)

        # Why: Y = iXZ, so a Y error is both a bit-flip AND a phase-flip.
        # Shor's code can correct it because it handles X and Z independently.
        syn_str = ', '.join(f'{k}={v}' for k, v in syn.items() if v != 0)
        if not syn_str:
            syn_str = "all zeros"
        print(f"    {error_name}: syndrome=[{syn_str}], corrected={match}")

    # Test error on each qubit
    print(f"\n  X error on each of the 9 qubits:")
    all_ok = True
    for q in range(9):
        errored = apply_single_qubit_gate(encoded.copy(), X, q, 9)
        syn = shor_syndrome(errored)
        corrected = shor_correct(errored, syn)
        match = np.allclose(corrected, encoded)
        if not match:
            all_ok = False
            print(f"    Qubit {q}: FAILED")
    print(f"    All 9 qubits: {'PASS' if all_ok else 'SOME FAILED'}")

    # Z errors
    print(f"\n  Z error on each of the 9 qubits:")
    all_ok = True
    for q in range(9):
        errored = apply_single_qubit_gate(encoded.copy(), Z, q, 9)
        syn = shor_syndrome(errored)
        corrected = shor_correct(errored, syn)
        match = np.allclose(corrected, encoded)
        if not match:
            all_ok = False
            print(f"    Qubit {q}: FAILED")
    print(f"    All 9 qubits: {'PASS' if all_ok else 'SOME FAILED'}")


def demo_error_detection_vs_correction():
    """Show the difference between detection and correction."""
    print("\n" + "=" * 60)
    print("DEMO 4: Error Detection vs Error Correction")
    print("=" * 60)

    alpha, beta = 1 / np.sqrt(2), 1 / np.sqrt(2)
    encoded = bitflip_encode(alpha, beta)

    # Single error: detected AND corrected
    single_err = apply_single_qubit_gate(encoded.copy(), X, 0, 3)
    syn = bitflip_syndrome(single_err)
    print(f"\n  Single bit-flip (qubit 0):")
    print(f"    Syndrome: {syn} → error detected and localized")
    corrected = bitflip_correct(single_err, syn)
    print(f"    Correction successful? {np.allclose(corrected, encoded)}")

    # Two errors: detected but MIS-corrected
    double_err = apply_single_qubit_gate(encoded.copy(), X, 0, 3)
    double_err = apply_single_qubit_gate(double_err, X, 1, 3)
    syn = bitflip_syndrome(double_err)
    print(f"\n  Double bit-flip (qubits 0 and 1):")
    print(f"    Syndrome: {syn} → error detected, but wrong qubit identified!")
    corrected = bitflip_correct(double_err, syn)
    print(f"    Correction successful? {np.allclose(corrected, encoded)}")

    # Why: The 3-qubit code can detect up to 2 errors but can only CORRECT
    # 1 error.  With 2 errors, the syndrome points to the wrong qubit, and
    # "correction" makes things worse.  This is a fundamental tradeoff:
    # more redundancy (more physical qubits) allows correcting more errors.
    print(f"\n  Key lesson:")
    print(f"  - 3-qubit code: detects ≤2 errors, corrects ≤1 error")
    print(f"  - Shor 9-qubit code: corrects any single-qubit error (X, Y, or Z)")
    print(f"  - Surface codes (modern): correct more errors with better overhead")


def demo_no_cloning():
    """Illustrate why error correction is subtle in quantum computing."""
    print("\n" + "=" * 60)
    print("DEMO 5: Why Quantum Error Correction Is Hard")
    print("=" * 60)

    print(f"""
  Classical error correction is simple:
    - Copy the bit 3 times: 0 → 000, 1 → 111
    - If one bit flips, majority vote corrects it

  Quantum error correction faces three obstacles:

  1. NO-CLONING THEOREM: Cannot copy an unknown quantum state |ψ⟩.
     We can't do |ψ⟩ → |ψ⟩|ψ⟩|ψ⟩.
     Solution: Encode into entangled states instead of copying.

  2. MEASUREMENT DESTROYS INFORMATION: Reading a qubit collapses it.
     Solution: Syndrome measurement — measure PARITY, not individual qubits.
     This reveals error information without learning the encoded data.

  3. CONTINUOUS ERRORS: A qubit can rotate by any angle, not just flip.
     Solution: Any small rotation error is projected onto {I, X, Y, Z}
     by the syndrome measurement.  Correct the discrete error, and the
     continuous error is automatically fixed!""")

    # Why: Point 3 is perhaps the most surprising.  Even though errors are
    # continuous (any unitary can afflict a qubit), the syndrome measurement
    # discretizes them — after measurement, the state has either no error or
    # a definite Pauli error, which we can correct.
    print(f"\n  Demonstration: Continuous rotation error → discretized by syndrome")

    alpha, beta = 1 / np.sqrt(2), 1 / np.sqrt(2)
    encoded = bitflip_encode(alpha, beta)

    # Apply a small rotation (continuous error) to qubit 0
    theta = 0.3  # Small angle
    Rx = np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)]
    ], dtype=complex)

    errored = apply_single_qubit_gate(encoded.copy(), Rx, 0, 3)

    # Syndrome measurement projects onto error/no-error subspace
    syn = bitflip_syndrome(errored)
    print(f"    Rotation Rx({theta:.2f}) on qubit 0: syndrome = {syn}")
    print(f"    The continuous error is projected onto a discrete outcome")
    print(f"    (either 'no error happened' or 'full X error on qubit 0')")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Quantum Computing — 09: Quantum Error Correction      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    np.random.seed(2026)

    demo_bitflip_code()
    demo_phaseflip_code()
    demo_shor_code()
    demo_error_detection_vs_correction()
    demo_no_cloning()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
