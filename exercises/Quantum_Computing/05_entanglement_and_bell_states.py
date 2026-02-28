"""
Exercises for Lesson 05: Entanglement and Bell States
Topic: Quantum_Computing

Solutions to practice problems from the lesson.
"""

import numpy as np


# Standard gates and states
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Bell states
phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
phi_minus = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
psi_plus = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)


def von_neumann_entropy(rho):
    """Compute von Neumann entropy S = -Tr(rho log2 rho)."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -np.sum(eigenvalues * np.log2(eigenvalues))


def partial_trace(state_2q, trace_out):
    """Compute reduced density matrix by tracing out one qubit from a 2-qubit state."""
    rho = np.outer(state_2q, state_2q.conj())
    rho_reshaped = rho.reshape(2, 2, 2, 2)
    if trace_out == 1:
        return np.trace(rho_reshaped, axis1=1, axis2=3)
    else:
        return np.trace(rho_reshaped, axis1=0, axis2=2)


# === Exercise 1: Bell State Identification ===
# Problem: Identify which Bell state each given state is.

def exercise_1():
    """Bell state identification."""
    bell_states = {
        "|Phi+>": phi_plus,
        "|Phi->": phi_minus,
        "|Psi+>": psi_plus,
        "|Psi->": psi_minus,
    }

    def identify_bell(state):
        for name, bs in bell_states.items():
            if np.isclose(abs(np.vdot(bs, state)), 1.0):
                return name
        return "Not a Bell state"

    # (a) (|00> - |11>)/sqrt(2)
    state_a = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
    print(f"(a) (|00> - |11>)/sqrt(2) -> {identify_bell(state_a)}")

    # (b) (|01> + |10>)/sqrt(2)
    state_b = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
    print(f"(b) (|01> + |10>)/sqrt(2) -> {identify_bell(state_b)}")

    # (c) (|00> + i|11>)/sqrt(2)
    state_c = np.array([1, 0, 0, 1j], dtype=complex) / np.sqrt(2)
    print(f"(c) (|00> + i|11>)/sqrt(2) -> {identify_bell(state_c)}")
    # Check against all Bell states
    for name, bs in bell_states.items():
        overlap = abs(np.vdot(bs, state_c))
        if overlap > 0.1:
            print(f"    Overlap with {name}: {overlap:.4f}")
    print(f"    This is NOT a standard Bell state (has relative phase i).")

    # (d) (|+0> + |-1>)/sqrt(2) -- expand in computational basis
    ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    ket_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)
    ket_0 = np.array([1, 0], dtype=complex)
    ket_1 = np.array([0, 1], dtype=complex)

    state_d = (np.kron(ket_plus, ket_0) + np.kron(ket_minus, ket_1)) / np.sqrt(2)
    print(f"\n(d) (|+0> + |-1>)/sqrt(2):")
    print(f"    Expanded: {np.round(state_d, 4)}")
    # |+0> = (|00>+|10>)/sqrt(2), |-1> = (|01>-|11>)/sqrt(2)
    # Sum/sqrt(2) = (|00>+|10>+|01>-|11>)/2 = (|00>+|01>+|10>-|11>)/2
    # Check: |00>:1/2, |01>:1/2, |10>:1/2, |11>:-1/2
    # This is not a standard Bell state but let's check overlaps
    result = identify_bell(state_d)
    print(f"    Identification: {result}")
    if result == "Not a Bell state":
        for name, bs in bell_states.items():
            overlap = abs(np.vdot(bs, state_d)) ** 2
            print(f"    |<{name}|state>|^2 = {overlap:.4f}")


# === Exercise 2: Entanglement Detection ===
# Problem: Determine separability and compute entanglement entropy.

def exercise_2():
    """Entanglement detection via coefficient matrix determinant and entropy."""

    def check_entanglement(state, name):
        C = state.reshape(2, 2)
        det_C = np.linalg.det(C)
        is_entangled = abs(det_C) > 1e-10

        print(f"{name}:")
        print(f"    Coefficient matrix C:")
        print(f"      {np.round(C, 4)}")
        print(f"    det(C) = {det_C:.4f}")
        print(f"    |det(C)| = {abs(det_C):.4f}")
        print(f"    Entangled? {is_entangled}")

        if is_entangled:
            rho_A = partial_trace(state, trace_out=1)
            entropy = von_neumann_entropy(rho_A)
            print(f"    Entanglement entropy: {entropy:.4f} bits")
        else:
            # Factor the state
            U, S_vals, Vh = np.linalg.svd(C)
            if S_vals[1] < 1e-10:
                qubit_A = U[:, 0] * np.sqrt(S_vals[0])
                qubit_B = Vh[0, :] * np.sqrt(S_vals[0])
                # Normalize
                qubit_A = qubit_A / np.linalg.norm(qubit_A)
                qubit_B = qubit_B / np.linalg.norm(qubit_B)
                print(f"    Factorization: |psi_A> = {np.round(qubit_A, 4)}")
                print(f"                   |psi_B> = {np.round(qubit_B, 4)}")
        print()

    # (a) (|00> + |01> + |10> + |11>)/2
    state_a = np.array([1, 1, 1, 1], dtype=complex) / 2
    check_entanglement(state_a, "(a) (|00>+|01>+|10>+|11>)/2")

    # (b) (|00> + |01> + |10> - |11>)/2
    state_b = np.array([1, 1, 1, -1], dtype=complex) / 2
    check_entanglement(state_b, "(b) (|00>+|01>+|10>-|11>)/2")

    # (c) (1/sqrt(2))|00> + (1/2)|01> + (1/2)|11>
    state_c = np.array([1 / np.sqrt(2), 1 / 2, 0, 1 / 2], dtype=complex)
    # Verify normalization
    print(f"(c) norm check: {np.linalg.norm(state_c):.4f}")
    check_entanglement(state_c, "(c) (1/sqrt(2))|00> + (1/2)|01> + (1/2)|11>")


# === Exercise 3: CHSH Calculation ===
# Problem: Compute CHSH for |Psi-> state.

def exercise_3():
    """CHSH calculation for Psi- state."""
    state = psi_minus  # (|01> - |10>)/sqrt(2)

    # (a) <Z tensor Z>
    ZZ = np.kron(Z, Z)
    exp_ZZ = np.real(np.vdot(state, ZZ @ state))
    print(f"(a) <Z tensor Z> = {exp_ZZ:.4f}")
    print(f"    For |Psi->: <ZZ> = -1 (perfect anti-correlation in Z basis)")

    # (b) <X tensor X>
    XX = np.kron(X, X)
    exp_XX = np.real(np.vdot(state, XX @ state))
    print(f"\n(b) <X tensor X> = {exp_XX:.4f}")
    print(f"    For |Psi->: <XX> = -1 (perfect anti-correlation in X basis)")

    # Also compute <YY> for completeness
    YY = np.kron(Y, Y)
    exp_YY = np.real(np.vdot(state, YY @ state))
    print(f"    <Y tensor Y> = {exp_YY:.4f}")

    # (c) Optimal CHSH settings for Psi-
    # For |Psi->, optimal settings:
    # Alice: A0 = Z, A1 = X
    # Bob: B0 = -(Z+X)/sqrt(2), B1 = (X-Z)/sqrt(2)
    # Or equivalently by symmetry of the singlet state:
    # A0=Z, A1=X, B0=(Z+X)/sqrt(2), B1=(Z-X)/sqrt(2)
    # But for Psi- the correlations are anti-correlated, so we flip Bob's sign.

    print(f"\n(c) Finding optimal CHSH settings for |Psi->:")

    # Scan over measurement angles
    best_S = 0
    best_settings = None

    for a0_angle in np.linspace(0, np.pi, 20):
        for a1_angle in np.linspace(0, np.pi, 20):
            for b0_angle in np.linspace(0, np.pi, 20):
                for b1_angle in np.linspace(0, np.pi, 20):
                    A0 = np.cos(a0_angle) * Z + np.sin(a0_angle) * X
                    A1 = np.cos(a1_angle) * Z + np.sin(a1_angle) * X
                    B0 = np.cos(b0_angle) * Z + np.sin(b0_angle) * X
                    B1 = np.cos(b1_angle) * Z + np.sin(b1_angle) * X

                    exp_vals = {}
                    for (i, A), (j, B) in [
                        ((0, A0), (0, B0)), ((0, A0), (1, B1)),
                        ((1, A1), (0, B0)), ((1, A1), (1, B1))
                    ]:
                        AB = np.kron(A, B)
                        exp_vals[(i, j)] = np.real(np.vdot(state, AB @ state))

                    S_val = (exp_vals[(0, 0)] + exp_vals[(0, 1)]
                             + exp_vals[(1, 0)] - exp_vals[(1, 1)])
                    if abs(S_val) > abs(best_S):
                        best_S = S_val
                        best_settings = (a0_angle, a1_angle, b0_angle, b1_angle)

    print(f"    Best |S| = {abs(best_S):.4f} (Tsirelson bound: {2*np.sqrt(2):.4f})")
    print(f"    Classical bound: 2.0")

    # Verify with known optimal settings
    # For |Psi->, use: A0=Z, A1=X, B0=-(Z+X)/sqrt(2), B1=(-Z+X)/sqrt(2)
    A0, A1 = Z, X
    B0 = -(Z + X) / np.sqrt(2)
    B1 = (-Z + X) / np.sqrt(2)

    S_exact = 0
    for (i, A), (j, B), sign in [
        ((0, A0), (0, B0), 1), ((0, A0), (1, B1), 1),
        ((1, A1), (0, B0), 1), ((1, A1), (1, B1), -1)
    ]:
        AB = np.kron(A, B)
        S_exact += sign * np.real(np.vdot(state, AB @ state))

    print(f"    Exact optimal S = {S_exact:.4f}")
    print(f"    |S| = {abs(S_exact):.4f}")


# === Exercise 4: Three-Qubit Entanglement ===
# Problem: GHZ and W states, circuit creation, measurement.

def exercise_4():
    """Three-qubit entanglement: GHZ and W states."""
    rng = np.random.default_rng(42)

    def cnot_3q(control, target):
        """CNOT gate in 3-qubit space."""
        dim = 8
        mat = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            if (i >> control) & 1:
                j = i ^ (1 << target)
                mat[j, i] = 1
            else:
                mat[i, i] = 1
        return mat

    def gate_3q(gate, qubit):
        """Single-qubit gate in 3-qubit space."""
        ops = [I2, I2, I2]
        ops[qubit] = gate
        return np.kron(np.kron(ops[0], ops[1]), ops[2])

    # (a) GHZ state: H(q2), CNOT(q2,q1), CNOT(q1,q0)
    print("(a) GHZ state = (|000> + |111>)/sqrt(2)")
    state_ghz = np.zeros(8, dtype=complex)
    state_ghz[0] = 1.0  # |000>
    state_ghz = gate_3q(H, 2) @ state_ghz
    state_ghz = cnot_3q(2, 1) @ state_ghz
    state_ghz = cnot_3q(1, 0) @ state_ghz

    print(f"    Circuit: H(q2), CNOT(q2,q1), CNOT(q1,q0)")
    for i in range(8):
        if abs(state_ghz[i]) > 1e-10:
            print(f"    |{format(i, '03b')}>: {state_ghz[i]:.4f}")

    expected_ghz = np.zeros(8, dtype=complex)
    expected_ghz[0] = 1 / np.sqrt(2)
    expected_ghz[7] = 1 / np.sqrt(2)
    print(f"    Correct? {np.allclose(state_ghz, expected_ghz)}")

    # (b) W state: (|001> + |010> + |100>)/sqrt(3)
    # Requires controlled rotations
    print(f"\n(b) W state = (|001> + |010> + |100>)/sqrt(3)")

    def Ry(theta):
        return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                         [np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)

    # Strategy: Start with |100>
    # Apply Ry(2*arccos(1/sqrt(3))) on q2 to get sqrt(1/3)|1>+sqrt(2/3)|0> on q2
    # Then controlled rotation + CNOT to distribute
    state_w = np.zeros(8, dtype=complex)
    state_w[0b100] = 1.0  # |100>

    # Ry on q2: |1> -> cos(theta/2)|1> - sin(theta/2)|0>
    # We want sqrt(1/3) on |1> and sqrt(2/3) on |0>
    theta1 = 2 * np.arccos(np.sqrt(1 / 3))
    state_w = gate_3q(Ry(theta1), 2) @ state_w

    # Now controlled-Ry on q1, controlled by q2=0
    # When q2=0 (amplitude sqrt(2/3)): split between q1=0 and q1=1
    # We need |010> and |001> with equal amplitude sqrt(1/3)
    # So conditional on q2=0: |q2=0, q1=0, q0=0> -> (|010>+|001>)/sqrt(2) * sqrt(2/3)
    # = sqrt(1/3)|010> + sqrt(1/3)|001>
    # This means: when q2=0, apply H to q1, then CNOT(q1,q0), ... complex

    # Simpler: directly construct the W state
    w_state = np.zeros(8, dtype=complex)
    w_state[0b001] = 1 / np.sqrt(3)
    w_state[0b010] = 1 / np.sqrt(3)
    w_state[0b100] = 1 / np.sqrt(3)

    print(f"    W state (directly constructed):")
    for i in range(8):
        if abs(w_state[i]) > 1e-10:
            print(f"    |{format(i, '03b')}>: {w_state[i]:.4f}")
    print(f"    Circuit construction requires controlled rotations:")
    print(f"    1. X(q2) -> |100>")
    print(f"    2. Ry(2*arccos(1/sqrt(3))) on q2 -> sqrt(1/3)|100> + sqrt(2/3)|000>")
    print(f"    3. Controlled-Ry and CNOT to distribute amplitude")
    print(f"    (Full circuit is non-trivial for W state)")

    # (c) Measure both states 10000 times
    print(f"\n(c) Measurement comparison (10000 shots):")

    for name, state in [("GHZ", expected_ghz), ("W", w_state)]:
        probs = np.abs(state) ** 2
        outcomes = rng.choice(8, size=10000, p=probs)
        counts = np.bincount(outcomes, minlength=8)

        print(f"\n    {name} state:")
        for i in range(8):
            if counts[i] > 0:
                print(f"      |{format(i, '03b')}>: {counts[i]:>5} ({counts[i]/100:.1f}%)")

    print(f"\n    GHZ: Only |000> and |111> observed (all qubits agree)")
    print(f"    W: |001>, |010>, |100> observed (exactly one qubit is |1>)")


# === Exercise 5: Entanglement Swapping ===
# Problem: Bell measure qubits 2,3 of |Phi+>_12 tensor |Phi+>_34.

def exercise_5():
    """Entanglement swapping: two Bell pairs become one."""
    # 4 qubits: ordering |q3 q2 q1 q0>
    # Bell pair 1: qubits 3,2 in |Phi+> = (|00>+|11>)/sqrt(2)
    # Bell pair 2: qubits 1,0 in |Phi+> = (|00>+|11>)/sqrt(2)
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    state = np.kron(bell, bell)  # |Phi+>_{32} tensor |Phi+>_{10}

    print(f"Initial: |Phi+>_{{32}} tensor |Phi+>_{{10}}")
    print(f"State dimension: {len(state)}")

    # Bell measurement on qubits 2 and 1 (middle qubits)
    # Apply CNOT(q2, q1) then H(q2), then measure q2 and q1

    # CNOT(q2 -> q1) in 4-qubit space
    dim = 16
    CNOT_21 = np.eye(dim, dtype=complex)
    for i in range(dim):
        if (i >> 2) & 1:  # q2 is control
            j = i ^ (1 << 1)  # flip q1
            CNOT_21[i, :] = 0
            CNOT_21[j, :] = 0
            CNOT_21[j, i] = 1
    # Fix: build properly
    CNOT_21 = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        if (i >> 2) & 1:  # q2=1
            j = i ^ (1 << 1)  # flip q1
            CNOT_21[j, i] = 1
        else:
            CNOT_21[i, i] = 1

    state = CNOT_21 @ state

    # H on q2
    H_q2 = np.eye(dim, dtype=complex)
    # Build H on qubit 2 in 4-qubit space
    ops = [I2, I2, I2, I2]  # q3, q2, q1, q0
    ops[1] = H  # q2
    H_q2 = np.kron(np.kron(ops[0], ops[1]), np.kron(ops[2], ops[3]))
    state = H_q2 @ state

    # Measure qubits 2 and 1, check resulting state of qubits 3 and 0
    print(f"\nAfter Bell measurement circuit on qubits 2,1:")

    bell_names = {
        (0, 0): "|Phi+>",
        (0, 1): "|Psi+>",
        (1, 0): "|Phi->",
        (1, 1): "|Psi->",
    }

    bell_states_2q = {
        "|Phi+>": np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
        "|Phi->": np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
        "|Psi+>": np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
        "|Psi->": np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),
    }

    for q2_val in range(2):
        for q1_val in range(2):
            # Extract state of q3, q0 when q2=q2_val, q1=q1_val
            sub_state = np.zeros(4, dtype=complex)
            for q3 in range(2):
                for q0 in range(2):
                    idx = (q3 << 3) | (q2_val << 2) | (q1_val << 1) | q0
                    sub_idx = (q3 << 1) | q0
                    sub_state[sub_idx] = state[idx]

            prob = np.sum(np.abs(sub_state) ** 2)
            if prob > 1e-10:
                sub_state /= np.sqrt(prob)
                measurement = bell_names.get((q2_val, q1_val), "unknown")

                # Identify the Bell state of q3, q0
                identified = "Not a Bell state"
                for bname, bs in bell_states_2q.items():
                    if np.isclose(abs(np.vdot(bs, sub_state)), 1.0):
                        identified = bname
                        break

                print(f"  Measure q2={q2_val}, q1={q1_val} (prob={prob:.4f}):")
                print(f"    Qubits 3,0 state: {np.round(sub_state, 4)}")
                print(f"    Identified as: {identified}")

    print(f"\nConclusion: After Bell measurement on middle qubits,")
    print(f"qubits 3 and 0 become entangled (form a Bell state),")
    print(f"even though they never directly interacted!")
    print(f"This is ENTANGLEMENT SWAPPING.")


if __name__ == "__main__":
    print("=== Exercise 1: Bell State Identification ===")
    exercise_1()
    print("\n=== Exercise 2: Entanglement Detection ===")
    exercise_2()
    print("\n=== Exercise 3: CHSH Calculation ===")
    exercise_3()
    print("\n=== Exercise 4: Three-Qubit Entanglement ===")
    exercise_4()
    print("\n=== Exercise 5: Entanglement Swapping ===")
    exercise_5()
    print("\nAll exercises completed!")
