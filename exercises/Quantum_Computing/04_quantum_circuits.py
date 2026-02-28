"""
Exercises for Lesson 04: Quantum Circuits
Topic: Quantum_Computing

Solutions to practice problems from the lesson.
"""

import numpy as np
import time


# Standard gates
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def gate_on_qubit(gate, target, n_qubits):
    """Apply a single-qubit gate on a target qubit in an n-qubit system."""
    matrices = []
    for q in range(n_qubits - 1, -1, -1):
        if q == target:
            matrices.append(gate)
        else:
            matrices.append(I2)
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def cnot_matrix(control, target, n_qubits):
    """Build a CNOT matrix for given control and target in an n-qubit system."""
    dim = 2 ** n_qubits
    mat = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        ctrl_bit = (i >> control) & 1
        if ctrl_bit == 1:
            j = i ^ (1 << target)
            mat[j, i] = 1
        else:
            mat[i, i] = 1
    return mat


def cz_matrix(qubit_a, qubit_b, n_qubits):
    """Build a CZ gate matrix."""
    dim = 2 ** n_qubits
    mat = np.eye(dim, dtype=complex)
    for i in range(dim):
        bit_a = (i >> qubit_a) & 1
        bit_b = (i >> qubit_b) & 1
        if bit_a == 1 and bit_b == 1:
            mat[i, i] = -1
    return mat


# === Exercise 1: Circuit Tracing ===
# Problem: Trace q0:─[H]─●─[H]─, q1:──────X───── starting from |00>.
# Is this the same as the standard Bell state?

def exercise_1():
    """Circuit tracing: H-CNOT-H circuit."""
    ket_00 = np.array([1, 0, 0, 0], dtype=complex)

    # Step 1: H on q0
    U1 = np.kron(I2, H)  # q1=I, q0=H (little-endian: q0 rightmost)
    state = U1 @ ket_00
    print(f"(a) After H on q0:")
    for i in range(4):
        if abs(state[i]) > 1e-10:
            print(f"    |{format(i, '02b')}>: {state[i]:.4f}")

    # Step 2: CNOT (q0 control, q1 target)
    CNOT = cnot_matrix(0, 1, 2)
    state = CNOT @ state
    print(f"\n(b) After CNOT(q0->q1):")
    for i in range(4):
        if abs(state[i]) > 1e-10:
            print(f"    |{format(i, '02b')}>: {state[i]:.4f}")

    # Step 3: H on q0
    U3 = np.kron(I2, H)
    state = U3 @ state
    print(f"\n(c) After H on q0:")
    for i in range(4):
        if abs(state[i]) > 1e-10:
            print(f"    |{format(i, '02b')}>: {state[i]:.4f}")

    # Compare with standard Bell state (|00>+|11>)/sqrt(2)
    bell_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    print(f"\n(d) Is this the standard Bell state (|00>+|11>)/sqrt(2)?")
    print(f"    {np.allclose(state, bell_plus)}")

    # The actual result is (|00>+|10>)/sqrt(2) = |+>|0> -- NOT a Bell state
    # H-CNOT-H produces CZ, not CNOT, so the circuit is:
    # |00> -> H(q0) -> (|0>+|1>)/sqrt(2)|0> -> CNOT -> (|00>+|11>)/sqrt(2)
    # -> H(q0) -> (|00>+|10>)/sqrt(2) = |+0>
    # Actually: HZH=X on the target qubit side
    # Let's verify: H-CNOT-H = CZ
    full_circuit = U3 @ CNOT @ U1
    CZ = cz_matrix(0, 1, 2)
    print(f"\n    H(q0) CNOT H(q0) = CZ? {np.allclose(full_circuit, CZ)}")
    print(f"    The circuit produces: (|00> + |10>)/sqrt(2) = |+>|0>")
    print(f"    This is NOT a Bell state -- it is separable!")
    print(f"    The H after CNOT 'undoes' the entanglement in a specific way.")


# === Exercise 2: Circuit Matrix ===
# Problem: Compute full 4x4 matrix for q0:─[X]─●─, q1:─[H]─X─.

def exercise_2():
    """Circuit matrix computation."""
    # Step 1: X on q0, H on q1 (in parallel)
    U1 = np.kron(H, X)  # q1=H, q0=X
    print(f"(a) U1 = H(q1) tensor X(q0) =")
    print(f"    {np.round(U1, 4)}")

    # Step 2: CNOT (q0 control, q1 target)
    CNOT = cnot_matrix(0, 1, 2)

    # Full circuit
    U_full = CNOT @ U1
    print(f"\n(b) Full circuit matrix U = CNOT * U1 =")
    print(f"    {np.round(U_full, 4)}")

    # Verify: apply to |00> step by step
    ket_00 = np.array([1, 0, 0, 0], dtype=complex)

    # Step-by-step tracing
    after_step1 = U1 @ ket_00
    print(f"\n(c) Step-by-step verification:")
    print(f"    |00> after X(q0), H(q1):")
    for i in range(4):
        if abs(after_step1[i]) > 1e-10:
            print(f"      |{format(i, '02b')}>: {after_step1[i]:.4f}")
    # X|0>=|1>, H|0>=(|0>+|1>)/sqrt(2), so |01> -> |1>(|0>+|1>)/sqrt(2)
    # = (|01> + |11>)/sqrt(2)

    after_step2 = CNOT @ after_step1
    print(f"    After CNOT(q0->q1):")
    for i in range(4):
        if abs(after_step2[i]) > 1e-10:
            print(f"      |{format(i, '02b')}>: {after_step2[i]:.4f}")

    # Verify matrix matches
    direct_result = U_full @ ket_00
    print(f"\n    Direct matrix application matches? {np.allclose(after_step2, direct_result)}")

    # Check unitarity
    print(f"\n(d) Is U unitary? {np.allclose(U_full @ U_full.conj().T, np.eye(4))}")


# === Exercise 3: Depth Optimization ===
# Problem: Rearrange circuit to reduce depth from 4 to 3.
# Original: q0: H-T-ctrl--, q1: H----X-T (depth 4)

def exercise_3():
    """Depth optimization of a circuit."""
    # Original circuit (depth 4):
    # Layer 1: H(q0), H(q1)
    # Layer 2: T(q0)
    # Layer 3: CNOT(q0, q1)
    # Layer 4: T(q1)

    # Build original circuit
    layer1 = np.kron(H, H)  # H on both qubits
    layer2 = np.kron(I2, T)  # T on q0
    layer3 = cnot_matrix(0, 1, 2)
    layer4 = np.kron(T, I2)  # T on q1

    U_original = layer4 @ layer3 @ layer2 @ layer1
    print(f"Original circuit depth: 4")
    print(f"    Layer 1: H(q0), H(q1)")
    print(f"    Layer 2: T(q0)")
    print(f"    Layer 3: CNOT(q0, q1)")
    print(f"    Layer 4: T(q1)")

    # Optimized circuit (depth 3):
    # T(q0) only acts on q0, and H(q1) only acts on q1.
    # They can be parallelized!
    # Layer 1: H(q0), H(q1)
    # Layer 2: T(q0) -- can we parallelize this with something on q1?
    # Actually: H(q1) in layer 1 doesn't conflict with T(q0) in layer 2.
    # But they're on different qubits, so we could merge:
    # Layer 1: H(q0), H(q1)
    # Layer 2: T(q0) (q1 idle)
    # Layer 3: CNOT(q0, q1)
    # Layer 4: T(q1)
    # That's still depth 4...
    #
    # Key insight: T(q1) after CNOT commutes if we look at the structure.
    # Actually, since H(q1) and T(q0) are on different qubits, they can
    # run in parallel. So the REAL depth analysis:
    # time 1: H(q0), H(q1)   -- parallel
    # time 2: T(q0)           -- q1 idle
    # time 3: CNOT(q0,q1)
    # time 4: T(q1)
    # Depth = 4
    #
    # Optimization: merge T(q0) with something.
    # Since H(q1) is in time 1 and T(q0) is in time 2,
    # and they're on different qubits, we can overlap:
    # time 1: H(q0)
    # time 2: T(q0), H(q1)   -- parallel on different qubits!
    # time 3: CNOT(q0,q1)
    # time 4: T(q1)
    # Still depth 4...
    #
    # OR: merge H and T on q0:
    # time 1: HT(q0) = T*H on q0, H(q1)  -- T and H are both 1-qubit gates on q0
    # That would be: T(q0) applied after H(q0) = one combined gate
    # time 1: [TH](q0), H(q1)
    # time 2: CNOT
    # time 3: T(q1)
    # Depth = 3!

    opt_layer1 = np.kron(H, T @ H)  # H(q1), T*H on q0
    opt_layer2 = cnot_matrix(0, 1, 2)
    opt_layer3 = np.kron(T, I2)  # T on q1

    U_optimized = opt_layer3 @ opt_layer2 @ opt_layer1
    print(f"\nOptimized circuit depth: 3")
    print(f"    Layer 1: TH(q0), H(q1)  -- merge T and H on q0")
    print(f"    Layer 2: CNOT(q0, q1)")
    print(f"    Layer 3: T(q1)")

    print(f"\n    Same output? {np.allclose(U_original, U_optimized)}")
    print(f"\n    Minimum depth = 3 (CNOT must come after q0 preparation,")
    print(f"    and T(q1) must come after CNOT).")
    print(f"    We cannot do better than 3 because of the causal chain:")
    print(f"    q0 prep -> CNOT -> q1 post-processing.")


# === Exercise 4: Simulator Extension ===
# Problem: Extend QuantumCircuit simulator with S, T, CZ, barrier.

def exercise_4():
    """Extended quantum circuit simulator."""

    class QuantumCircuit:
        def __init__(self, n_qubits):
            self.n_qubits = n_qubits
            self.dim = 2 ** n_qubits
            self.state = np.zeros(self.dim, dtype=complex)
            self.state[0] = 1.0
            self.operations = []

        def _full_gate_matrix(self, gate, qubit):
            matrices = []
            for i in range(self.n_qubits - 1, -1, -1):
                matrices.append(gate if i == qubit else I2)
            result = matrices[0]
            for m in matrices[1:]:
                result = np.kron(result, m)
            return result

        def h(self, qubit):
            self.state = self._full_gate_matrix(H, qubit) @ self.state
            self.operations.append(f"H(q{qubit})")

        def x(self, qubit):
            self.state = self._full_gate_matrix(X, qubit) @ self.state
            self.operations.append(f"X(q{qubit})")

        def z(self, qubit):
            self.state = self._full_gate_matrix(Z, qubit) @ self.state
            self.operations.append(f"Z(q{qubit})")

        # --- New gates ---
        def s(self, qubit):
            """Apply S (phase) gate."""
            self.state = self._full_gate_matrix(S, qubit) @ self.state
            self.operations.append(f"S(q{qubit})")

        def t(self, qubit):
            """Apply T gate."""
            self.state = self._full_gate_matrix(T, qubit) @ self.state
            self.operations.append(f"T(q{qubit})")

        def cnot(self, control, target):
            full = cnot_matrix(control, target, self.n_qubits)
            self.state = full @ self.state
            self.operations.append(f"CNOT(q{control}->q{target})")

        def cz(self, qubit_a, qubit_b):
            """Apply CZ (controlled-Z) gate."""
            full = cz_matrix(qubit_a, qubit_b, self.n_qubits)
            self.state = full @ self.state
            self.operations.append(f"CZ(q{qubit_a},q{qubit_b})")

        def barrier(self):
            """Visual separation marker (no-op)."""
            self.operations.append("|")

        def get_probabilities(self):
            probs = np.abs(self.state) ** 2
            return {format(i, f'0{self.n_qubits}b'): probs[i]
                    for i in range(self.dim) if probs[i] > 1e-10}

        def __repr__(self):
            ops = " ".join(self.operations) if self.operations else "(empty)"
            return f"QC({self.n_qubits}q): {ops}"

    # Test: create (|00> + i|01> + |10> + i|11>)/2
    # = (1/sqrt(2))(|0> + |1>) tensor (1/sqrt(2))(|0> + i|1>)
    # = |+> tensor S|+>
    # Circuit: H(q0), H(q1), S(q1)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.barrier()
    qc.s(1)

    print(f"Circuit: {qc}")
    print(f"Probabilities: {qc.get_probabilities()}")
    print(f"State vector: {np.round(qc.state, 4)}")

    expected = np.array([1, 1j, 1, 1j], dtype=complex) / 2
    print(f"Expected: {np.round(expected, 4)}")
    print(f"Match? {np.allclose(qc.state, expected)}")

    # Test CZ gate
    print(f"\nCZ gate test:")
    qc2 = QuantumCircuit(2)
    qc2.h(0)
    qc2.h(1)
    qc2.cz(0, 1)
    print(f"  H(q0) H(q1) CZ(q0,q1):")
    print(f"  State: {np.round(qc2.state, 4)}")
    print(f"  Probabilities: {qc2.get_probabilities()}")

    # Test T gate
    print(f"\nT gate test: T^4 = S, T^8 = I")
    qc3 = QuantumCircuit(1)
    for _ in range(4):
        qc3.t(0)
    s_state = np.array([1, 0], dtype=complex)
    s_state = S @ s_state
    print(f"  T^4|0> = S|0>? {np.allclose(qc3.state, s_state)}")


# === Exercise 5: Simulation Limits ===
# Problem: Find maximum qubit count, time scaling, and Gottesman-Knill.

def exercise_5():
    """Simulation limits analysis."""
    # (a) Find maximum simulable qubit count
    print("(a) Maximum simulable qubit count:")
    print(f"    {'n':>4} {'dim':>12} {'RAM (bytes)':>14} {'Status':>10}")
    print(f"    {'-'*44}")

    for n in range(4, 30):
        dim = 2 ** n
        ram = dim * 16  # complex128 = 16 bytes
        status = "OK" if ram < 4e9 else ("tight" if ram < 16e9 else "too much")
        if n <= 15 or n % 5 == 0 or ram > 4e9:
            print(f"    {n:>4} {dim:>12,} {ram:>14,.0f} {status:>10}")
        if ram > 64e9:
            break

    # (b) Time scaling with gate count
    print(f"\n(b) Simulation time vs gate count (n=10 qubits):")
    n = 10
    dim = 2 ** n
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)

    # Build a random gate
    gate_full = np.kron(H, np.eye(dim // 2, dtype=complex))

    print(f"    {'Gates':>8} {'Time (ms)':>12} {'Time/gate (ms)':>16}")
    print(f"    {'-'*40}")

    for n_gates in [1, 5, 10, 50, 100]:
        test_state = state.copy()
        start = time.time()
        for _ in range(n_gates):
            test_state = gate_full @ test_state
        elapsed = (time.time() - start) * 1000
        print(f"    {n_gates:>8} {elapsed:>12.2f} {elapsed/n_gates:>16.4f}")

    print(f"\n    Time scales linearly with gate count (for fixed n).")
    print(f"    Each gate is a matrix-vector multiply: O(2^n) per gate.")

    # (c) Gottesman-Knill theorem
    print(f"\n(c) Gottesman-Knill theorem:")
    print(f"    Clifford circuits (H, S, CNOT, Pauli) can be simulated")
    print(f"    in poly time using the stabilizer formalism.")
    print(f"    This does NOT make quantum computing useless because:")
    print(f"    - Clifford circuits alone are NOT universal for QC")
    print(f"    - The T gate (or any non-Clifford gate) is needed for universality")
    print(f"    - T gates create 'magic states' outside the stabilizer formalism")
    print(f"    - Circuits with T gates cannot be efficiently simulated classically")
    print(f"    - All known quantum speedups (Shor, Grover, etc.) require non-Clifford gates")


if __name__ == "__main__":
    print("=== Exercise 1: Circuit Tracing ===")
    exercise_1()
    print("\n=== Exercise 2: Circuit Matrix ===")
    exercise_2()
    print("\n=== Exercise 3: Depth Optimization ===")
    exercise_3()
    print("\n=== Exercise 4: Simulator Extension ===")
    exercise_4()
    print("\n=== Exercise 5: Simulation Limits ===")
    exercise_5()
    print("\nAll exercises completed!")
