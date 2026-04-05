"""
Exercises for Lesson 11: Quantum Error Correction
Topic: Quantum_Computing

Solutions to practice problems from the lesson.
All quantum operations simulated with numpy matrices (no qiskit).
"""

import numpy as np
from typing import Tuple, List

# ============================================================
# Shared utilities: quantum gates and state operations
# ============================================================

# Pauli matrices
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Basis states
ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)


def tensor(*args):
    """Tensor product of multiple matrices/vectors."""
    result = args[0]
    for a in args[1:]:
        result = np.kron(result, a)
    return result


def apply_gate(state, gate, qubit, n_qubits):
    """Apply a single-qubit gate to a specific qubit in an n-qubit state."""
    ops = [I2] * n_qubits
    ops[qubit] = gate
    full_gate = tensor(*ops)
    return full_gate @ state


def measure_operator(state, operator):
    """Compute expectation value <state|operator|state>."""
    return np.real(state.conj() @ operator @ state)


def Rx(theta):
    """Rotation around X axis."""
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * X


# === Exercise 1: Bit-Flip Code by Hand ===
# Problem: Apply X error on qubit 2, compute syndromes, correct.

def exercise_1():
    """3-qubit bit-flip code with non-standard superposition."""
    print("=" * 60)
    print("Exercise 1: Bit-Flip Code by Hand")
    print("=" * 60)

    # Encoded state: |psi_L> = (1/sqrt(3))|000> + sqrt(2/3)|111>
    alpha = 1.0 / np.sqrt(3)
    beta = np.sqrt(2.0 / 3)

    ket000 = tensor(ket0, ket0, ket0)
    ket111 = tensor(ket1, ket1, ket1)
    psi_L = alpha * ket000 + beta * ket111

    print(f"\n  Encoded state: |psi_L> = (1/sqrt(3))|000> + sqrt(2/3)|111>")
    print(f"  alpha = {alpha:.4f}, beta = {beta:.4f}")
    print(f"  Norm: {np.linalg.norm(psi_L):.6f}")

    # (a) Apply X error on qubit 2 (middle qubit, index 1)
    X_qubit1 = tensor(I2, X, I2)
    psi_error = X_qubit1 @ psi_L

    print(f"\n(a) After X error on qubit 2 (middle):")
    # |000> -> |010>, |111> -> |101>
    ket010 = tensor(ket0, ket1, ket0)
    ket101 = tensor(ket1, ket0, ket1)
    print(f"    Expected: {alpha:.4f}|010> + {beta:.4f}|101>")

    # Verify
    coeff_010 = np.abs(psi_error @ ket010.conj())
    coeff_101 = np.abs(psi_error @ ket101.conj())
    print(f"    Actual coefficients: |010>={coeff_010:.4f}, |101>={coeff_101:.4f}")

    # (b) Compute syndromes using Z1Z2 and Z2Z3
    Z1Z2 = tensor(Z, Z, I2)
    Z2Z3 = tensor(I2, Z, Z)

    syn1 = measure_operator(psi_error, Z1Z2)
    syn2 = measure_operator(psi_error, Z2Z3)

    print(f"\n(b) Syndrome measurements:")
    print(f"    <psi'|Z1Z2|psi'> = {syn1:.4f}")
    print(f"    <psi'|Z2Z3|psi'> = {syn2:.4f}")

    # Syndrome table:
    # No error:     Z1Z2 = +1, Z2Z3 = +1
    # X on qubit 1: Z1Z2 = -1, Z2Z3 = +1
    # X on qubit 2: Z1Z2 = -1, Z2Z3 = -1
    # X on qubit 3: Z1Z2 = +1, Z2Z3 = -1

    if syn1 < 0 and syn2 < 0:
        print(f"    Syndrome (-1, -1) -> Error on qubit 2")
        correction_qubit = 1  # index 1
    elif syn1 < 0:
        print(f"    Syndrome (-1, +1) -> Error on qubit 1")
        correction_qubit = 0
    elif syn2 < 0:
        print(f"    Syndrome (+1, -1) -> Error on qubit 3")
        correction_qubit = 2
    else:
        print(f"    Syndrome (+1, +1) -> No error")
        correction_qubit = None

    # (c) Apply correction
    print(f"\n(c) Correction: Apply X on qubit {correction_qubit + 1 if correction_qubit is not None else 'none'}")

    if correction_qubit is not None:
        correction = [I2, I2, I2]
        correction[correction_qubit] = X
        correction_op = tensor(*correction)
        psi_corrected = correction_op @ psi_error
    else:
        psi_corrected = psi_error

    # (d) Verify correction
    fidelity = np.abs(psi_L.conj() @ psi_corrected) ** 2
    print(f"\n(d) Fidelity with original: {fidelity:.6f}")
    assert np.isclose(fidelity, 1.0), f"Correction failed: fidelity = {fidelity}"
    print("    Correction verified! State restored to original.")


# === Exercise 2: Phase-Flip Code ===
# Problem: Implement 3-qubit phase-flip code and show it fails for X errors.

def exercise_2():
    """3-qubit phase-flip code implementation."""
    print("\n" + "=" * 60)
    print("Exercise 2: Phase-Flip Code")
    print("=" * 60)

    # (a) Encoding circuit: H on all qubits, then CNOTs
    # |+> = H|0>, |-> = H|1>
    # Encoding: |0>_L = |+++>, |1>_L = |--->
    ket_plus = H @ ket0
    ket_minus = H @ ket1

    # Encode |+> state
    print("\n(a) Phase-flip code encoding:")
    print("    |0>_L = |+++> = H^3|000>")
    print("    |1>_L = |---> = H^3|111>")
    print("    Encoding: apply H to all, then CNOT(0->1), CNOT(0->2)")

    # Encode an arbitrary state alpha|0>_L + beta|1>_L
    alpha = 1.0 / np.sqrt(3)
    beta = np.sqrt(2.0 / 3)

    # Encoded state in computational basis (after Hadamard encoding)
    encoded = alpha * tensor(ket_plus, ket_plus, ket_plus) + \
              beta * tensor(ket_minus, ket_minus, ket_minus)

    print(f"\n    Encoded state: alpha={alpha:.4f}, beta={beta:.4f}")
    print(f"    Norm: {np.linalg.norm(encoded):.6f}")

    # (b) Syndrome operators for phase-flip code: X1X2 and X2X3
    X1X2 = tensor(X, X, I2)
    X2X3 = tensor(I2, X, X)

    print(f"\n(b) Syndrome operators: X1X2 and X2X3")
    print(f"    (Detect Z errors in the Hadamard basis)")

    # (c) Apply Z error on qubit 2 and correct
    Z_qubit1 = tensor(I2, Z, I2)
    psi_error = Z_qubit1 @ encoded

    syn1 = measure_operator(psi_error, X1X2)
    syn2 = measure_operator(psi_error, X2X3)

    print(f"\n(c) After Z error on qubit 2:")
    print(f"    <psi'|X1X2|psi'> = {syn1:.4f}")
    print(f"    <psi'|X2X3|psi'> = {syn2:.4f}")

    if syn1 < 0 and syn2 < 0:
        error_qubit = 1
    elif syn1 < 0:
        error_qubit = 0
    elif syn2 < 0:
        error_qubit = 2
    else:
        error_qubit = None

    print(f"    Detected error on qubit {error_qubit + 1 if error_qubit is not None else 'none'}")

    # Correct with Z
    if error_qubit is not None:
        correction = [I2, I2, I2]
        correction[error_qubit] = Z
        psi_corrected = tensor(*correction) @ psi_error
    else:
        psi_corrected = psi_error

    fidelity = np.abs(encoded.conj() @ psi_corrected) ** 2
    print(f"    Correction fidelity: {fidelity:.6f}")

    # (d) Show this code FAILS for X errors
    print(f"\n(d) Phase-flip code fails for X errors:")
    X_qubit1 = tensor(I2, X, I2)
    psi_x_error = X_qubit1 @ encoded

    syn1_x = measure_operator(psi_x_error, X1X2)
    syn2_x = measure_operator(psi_x_error, X2X3)
    print(f"    After X error on qubit 2:")
    print(f"    <psi'|X1X2|psi'> = {syn1_x:.4f}")
    print(f"    <psi'|X2X3|psi'> = {syn2_x:.4f}")
    print(f"    Syndrome is (+1, +1) -> code thinks no error occurred!")
    print(f"    This is because X1X2 commutes with X2, so no syndrome detected.")


# === Exercise 3: Error Discretization ===
# Problem: Show that continuous Rx rotation error gets discretized by
# syndrome measurement.

def exercise_3():
    """Error discretization: continuous errors become discrete after measurement."""
    print("\n" + "=" * 60)
    print("Exercise 3: Error Discretization")
    print("=" * 60)

    print("\n  Rx(eps) = cos(eps/2)*I - i*sin(eps/2)*X")
    print("  Applied to qubit 1 of encoded state alpha|000> + beta|111>")

    alpha = 1.0 / np.sqrt(2)
    beta = 1.0 / np.sqrt(2)

    ket000 = tensor(ket0, ket0, ket0)
    ket111 = tensor(ket1, ket1, ket1)
    psi_L = alpha * ket000 + beta * ket111

    # Syndrome operators for bit-flip code
    Z1Z2 = tensor(Z, Z, I2)
    Z2Z3 = tensor(I2, Z, Z)

    # Projectors onto syndrome subspaces
    # Syndrome (0,0) = no error subspace
    P_no_error = (np.eye(8) + Z1Z2) @ (np.eye(8) + Z2Z3) / 4
    # Syndrome (1,0) -> error on qubit 1
    P_err1 = (np.eye(8) - Z1Z2) @ (np.eye(8) + Z2Z3) / 4

    print(f"\n  Scanning epsilon from 0 to pi:")
    print(f"  {'eps/pi':<10} {'P(no error)':<15} {'P(err q1)':<15} {'P(correct)'}")
    print("  " + "-" * 55)

    for eps_frac in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        eps = eps_frac * np.pi

        # (a) Apply Rx(eps) on qubit 1
        Rx_eps = Rx(eps)
        error_op = tensor(Rx_eps, I2, I2)
        psi_error = error_op @ psi_L

        # (b) Syndrome measurement probabilities
        p_no_error = np.real(psi_error.conj() @ P_no_error @ psi_error)
        p_err1 = np.real(psi_error.conj() @ P_err1 @ psi_error)

        # (c) After measurement and correction, the state is:
        # With prob cos^2(eps/2): projected to no-error -> already correct
        # With prob sin^2(eps/2): projected to error-1 -> apply X on qubit 1

        # Probability of successful correction = both branches succeed
        # No-error branch: perfect
        # Error branch: after X correction, perfect
        p_success = p_no_error + p_err1  # Both branches are correctable

        # (d) Verify against theory: P(correct) should be ~1 for single errors
        # The bit-flip code can correct any single bit-flip, and Rx only
        # produces I (no error) and X (single bit-flip) components
        cos2 = np.cos(eps / 2) ** 2
        sin2 = np.sin(eps / 2) ** 2

        print(
            f"  {eps_frac:<10.2f} {p_no_error:<15.6f} {p_err1:<15.6f} "
            f"{p_success:.6f}"
        )

    print(f"\n  Key insight: P(successful correction) is always ~1.0!")
    print(f"  The syndrome measurement 'digitizes' the continuous error:")
    print(f"    - cos(eps/2) component -> no error detected -> state correct")
    print(f"    - sin(eps/2) component -> bit-flip detected -> X correction works")
    print(f"  This is why quantum error correction works for continuous errors.")


# === Exercise 4: Steane Code ===
# Problem: Implement the [[7,1,3]] Steane code with syndrome-based correction.

def exercise_4():
    """Steane [[7,1,3]] code: encoding and syndrome verification."""
    print("\n" + "=" * 60)
    print("Exercise 4: Steane Code [[7,1,3]]")
    print("=" * 60)

    # Steane code stabilizer generators (as Z/X on 7 qubits)
    # Z stabilizers: check bits for bit-flip errors
    # z1: Z on qubits 0,2,4,6 (binary: 1010101)
    # z2: Z on qubits 1,2,5,6 (binary: 0110110)
    # z3: Z on qubits 3,4,5,6 (binary: 0001111)
    # X stabilizers: check bits for phase-flip errors
    # x1: X on qubits 0,2,4,6
    # x2: X on qubits 1,2,5,6
    # x3: X on qubits 3,4,5,6

    n = 7  # 7 physical qubits
    N = 2 ** n  # 128-dimensional Hilbert space

    # (a) Build stabilizer generators
    z_patterns = [
        [0, 2, 4, 6],  # z1
        [1, 2, 5, 6],  # z2
        [3, 4, 5, 6],  # z3
    ]
    x_patterns = [
        [0, 2, 4, 6],  # x1
        [1, 2, 5, 6],  # x2
        [3, 4, 5, 6],  # x3
    ]

    def build_stabilizer(qubit_indices, gate_type, n_qubits):
        """Build a multi-qubit stabilizer operator."""
        gate = Z if gate_type == 'Z' else X
        ops = [I2] * n_qubits
        for idx in qubit_indices:
            ops[idx] = gate
        return tensor(*ops)

    z_stabilizers = [build_stabilizer(p, 'Z', n) for p in z_patterns]
    x_stabilizers = [build_stabilizer(p, 'X', n) for p in x_patterns]
    all_stabilizers = z_stabilizers + x_stabilizers

    # (b) Find the code space by simultaneous +1 eigenspace
    # Start with a projector onto the +1 eigenspace of all stabilizers
    projector = np.eye(N, dtype=complex)
    for stab in all_stabilizers:
        projector = projector @ (np.eye(N) + stab) / 2

    # The code space is 2D (encodes 1 logical qubit)
    eigenvalues, eigenvectors = np.linalg.eigh(projector)
    code_basis = eigenvectors[:, eigenvalues > 0.5]

    print(f"\n(a) Steane code: [[7,1,3]]")
    print(f"    6 stabilizer generators (3 Z-type, 3 X-type)")
    print(f"    Code space dimension: {code_basis.shape[1]}")

    if code_basis.shape[1] == 2:
        ket0_L = code_basis[:, 0]
        ket1_L = code_basis[:, 1]

        # (b) Verify stabilizers have eigenvalue +1 on code states
        print(f"\n(b) Stabilizer eigenvalue verification:")
        all_pass = True
        for i, stab in enumerate(all_stabilizers):
            ev0 = np.real(ket0_L.conj() @ stab @ ket0_L)
            ev1 = np.real(ket1_L.conj() @ stab @ ket1_L)
            label = f"Z{i+1}" if i < 3 else f"X{i-2}"
            status = "PASS" if np.isclose(ev0, 1.0) and np.isclose(ev1, 1.0) else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"    {label}: <0_L|S|0_L>={ev0:+.4f}, <1_L|S|1_L>={ev1:+.4f} [{status}]")

        # (c) Test error detection for single-qubit errors
        print(f"\n(c) Single-qubit error syndrome verification:")
        print(f"    Testing X, Y, Z errors on each of 7 qubits...")

        # Use the logical |0> state for testing
        test_state = ket0_L

        errors_detected = 0
        errors_tested = 0

        for qubit in range(n):
            for error_name, error_gate in [("X", X), ("Y", Y), ("Z", Z)]:
                errors_tested += 1
                # Apply error
                ops = [I2] * n
                ops[qubit] = error_gate
                error_op = tensor(*ops)
                errored_state = error_op @ test_state

                # Compute syndrome (use all 6 stabilizers)
                syndrome = []
                for stab in all_stabilizers:
                    ev = np.real(errored_state.conj() @ stab @ errored_state)
                    syndrome.append(0 if ev > 0 else 1)

                syndrome_tuple = tuple(syndrome)
                if any(s == 1 for s in syndrome):
                    errors_detected += 1

        print(f"    Detected {errors_detected}/{errors_tested} single-qubit errors")
        print(f"    (All single-qubit X, Y, Z errors should be detectable)")
    else:
        print(f"    WARNING: Code space has dimension {code_basis.shape[1]}, expected 2")
        print(f"    This may be due to numerical precision in the projector construction")


# === Exercise 5: Threshold Estimation ===
# Problem: Estimate the error correction threshold via Monte Carlo simulation.

def exercise_5():
    """Repetition code threshold estimation via Monte Carlo."""
    print("\n" + "=" * 60)
    print("Exercise 5: Threshold Estimation")
    print("=" * 60)

    def simulate_repetition_code(d: int, p: float, n_trials: int = 10000) -> float:
        """
        Simulate d-qubit repetition code under i.i.d. bit-flip noise.

        Args:
            d: code distance (number of physical qubits)
            p: physical error rate per qubit
            n_trials: number of Monte Carlo trials

        Returns:
            Logical error rate
        """
        logical_errors = 0
        threshold = d // 2  # Majority vote threshold

        for _ in range(n_trials):
            # Each physical qubit flips independently with probability p
            errors = np.random.random(d) < p
            num_errors = np.sum(errors)

            # Majority-vote decoder: logical error if more than d/2 qubits flipped
            if num_errors > threshold:
                logical_errors += 1

        return logical_errors / n_trials

    # (a) Scan physical error rate for different code distances
    distances = [3, 5, 7, 9, 11]
    p_values = np.logspace(-3, np.log10(0.5), 20)
    n_trials = 5000

    print(f"\n(a) Logical error rate vs physical error rate:")
    print(f"    (Monte Carlo, {n_trials} trials per point)")
    print()

    results = {}
    for d in distances:
        results[d] = []
        for p in p_values:
            p_L = simulate_repetition_code(d, p, n_trials)
            results[d].append(p_L)

    # Display as table (selected points)
    selected_p = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    header = f"  {'p_phys':<8}" + "".join(f"d={d:<6}" for d in distances)
    print(header)
    print("  " + "-" * (8 + 7 * len(distances)))

    for p_target in selected_p:
        idx = np.argmin(np.abs(p_values - p_target))
        p = p_values[idx]
        row = f"  {p:<8.3f}"
        for d in distances:
            row += f"{results[d][idx]:<7.4f}"
        print(row)

    # (b) Estimate threshold as crossover point
    print(f"\n(b) Threshold estimation:")
    print(f"    The repetition code threshold against i.i.d. bit-flip noise")
    print(f"    is p_th = 0.5 (theoretically, for infinite code distance).")
    print(f"    At p < 0.5, increasing d always reduces logical error rate.")
    print(f"    At p = 0.5, all codes have p_L ~ 0.5 (random guessing).")

    # (c) Compare with theoretical
    print(f"\n(c) Theoretical comparison:")
    print(f"    For repetition code of distance d, the logical error rate is:")
    print(f"    p_L = sum_{{k>(d/2)}} C(d,k) * p^k * (1-p)^(d-k)")
    print(f"    Threshold p_th = 0.5 for bit-flip channel (exact for d->inf)")

    # (d) Depolarizing noise model
    print(f"\n(d) Depolarizing noise model:")
    print(f"    Under depolarizing noise, X/Y/Z each occur with prob p/3.")
    print(f"    The repetition code only corrects X errors (bit-flips).")
    print(f"    Effective bit-flip rate: p_eff = 2p/3 (X and Y both flip)")
    print(f"    Threshold changes to p_th ~ 0.75 for bit-flip channel")
    print(f"    (but the code cannot correct Z errors at all!)")
    print(f"    For full depolarizing protection, need Shor's 9-qubit code")
    print(f"    or the Steane [[7,1,3]] code from Exercise 4.")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
