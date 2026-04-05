"""
Exercises for Lesson 03: Quantum Gates
Topic: Quantum_Computing

Solutions to practice problems from the lesson.
"""

import numpy as np

# Standard gates
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
S_dag = S.conj().T
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def Rx(theta):
    return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                      [-1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)


def Ry(theta):
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                      [np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)


def Rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0],
                      [0, np.exp(1j * theta / 2)]], dtype=complex)


def same_up_to_phase(A, B):
    """Check if A = e^{i*alpha} * B for some alpha."""
    ratio = A @ np.linalg.inv(B)
    phase = ratio[0, 0]
    return np.allclose(ratio, phase * np.eye(A.shape[0]))


# === Exercise 1: Gate Verification ===
# Problem: Verify HZH=X, HXH=Z, SXS_dag=Y (up to phase)

def exercise_1():
    """Gate conjugation identities."""
    # (a) HZH = X
    result_a = H @ Z @ H
    print(f"(a) HZH =\n    {np.round(result_a, 4)}")
    print(f"    Equal to X? {np.allclose(result_a, X)}")

    # (b) HXH = Z
    result_b = H @ X @ H
    print(f"\n(b) HXH =\n    {np.round(result_b, 4)}")
    print(f"    Equal to Z? {np.allclose(result_b, Z)}")

    # (c) SXS_dag = Y (up to phase)
    result_c = S @ X @ S_dag
    print(f"\n(c) SXS^dagger =\n    {np.round(result_c, 4)}")
    print(f"    Equal to Y? {np.allclose(result_c, Y)}")
    # Actually SXS^dag = iXZ = Y (no phase needed here)
    if not np.allclose(result_c, Y):
        print(f"    Equal to Y up to phase? {same_up_to_phase(result_c, Y)}")


# === Exercise 2: Bloch Sphere Rotations ===
# Problem: Apply rotation gates and check non-commutativity.

def exercise_2():
    """Rotation gates and non-commutativity."""
    ket_0 = np.array([1, 0], dtype=complex)
    ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)

    # (a) Ry(pi/4)|0>
    state_a = Ry(np.pi / 4) @ ket_0
    print(f"(a) Ry(pi/4)|0> = {np.round(state_a, 4)}")
    print(f"    = cos(pi/8)|0> + sin(pi/8)|1>")
    print(f"    = {np.cos(np.pi/8):.4f}|0> + {np.sin(np.pi/8):.4f}|1>")
    print(f"    Bloch sphere: theta = pi/4 (22.5 deg from north pole), phi = 0")

    # (b) Rz(pi/2)|+>
    state_b = Rz(np.pi / 2) @ ket_plus
    print(f"\n(b) Rz(pi/2)|+> = {np.round(state_b, 4)}")
    print(f"    = e^{{-i*pi/4}}(|0> + e^{{i*pi/2}}|1>)/sqrt(2)")
    print(f"    This is |i> (up to global phase) -- rotated to Y-axis on Bloch sphere")

    # (c) Rx(theta)*Ry(theta) vs Ry(theta)*Rx(theta) for theta=pi/4
    theta = np.pi / 4
    rx_ry = Rx(theta) @ Ry(theta)
    ry_rx = Ry(theta) @ Rx(theta)
    print(f"\n(c) For theta = pi/4:")
    print(f"    Rx*Ry =\n    {np.round(rx_ry, 4)}")
    print(f"    Ry*Rx =\n    {np.round(ry_rx, 4)}")
    print(f"    Are they equal? {np.allclose(rx_ry, ry_rx)}")
    diff_norm = np.linalg.norm(rx_ry - ry_rx)
    print(f"    ||Rx*Ry - Ry*Rx|| = {diff_norm:.4f}")
    print(f"    Quantum gates generally do NOT commute!")


# === Exercise 3: Controlled Gate Construction ===
# Problem: Build Controlled-H as a 4x4 matrix.

def exercise_3():
    """Controlled-Hadamard gate."""
    # CH = |0><0| x I + |1><1| x H
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)  # |0><0|
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1><1|
    CH = np.kron(P0, I2) + np.kron(P1, H)

    print(f"Controlled-H gate (4x4):")
    print(f"{np.round(CH, 4)}")
    print(f"Is unitary? {np.allclose(CH @ CH.conj().T, np.eye(4))}")

    # (a) CH|00> = |00>
    ket_00 = np.array([1, 0, 0, 0], dtype=complex)
    result_a = CH @ ket_00
    print(f"\n(a) CH|00> = {np.round(result_a, 4)}")
    print(f"    = |00>? {np.allclose(result_a, ket_00)}")

    # (b) CH|10> = |1>x H|0> = |1,+>
    ket_10 = np.array([0, 0, 1, 0], dtype=complex)
    result_b = CH @ ket_10
    ket_1_plus = np.kron(np.array([0, 1], dtype=complex),
                         np.array([1, 1], dtype=complex) / np.sqrt(2))
    print(f"\n(b) CH|10> = {np.round(result_b, 4)}")
    print(f"    = |1,+>? {np.allclose(result_b, ket_1_plus)}")


# === Exercise 4: Gate Count ===
# Problem: Count gates from {H, T, CNOT} for various gates.

def exercise_4():
    """Gate count for standard gates from {H, T, CNOT}."""
    print("(a) X gate = HZH = H * T^4 * H")
    print("    Z = T^4, so X = H * T * T * T * T * H")
    print("    Gate count: 2 H + 4 T = 6 gates")
    # Verify
    X_from_HT = H @ np.linalg.matrix_power(T, 4) @ H
    print(f"    Verification: {np.allclose(X_from_HT, X)}")

    print("\n(b) S gate = T^2")
    print("    Gate count: 2 T")
    S_from_T = T @ T
    print(f"    Verification: {np.allclose(S_from_T, S)}")

    print("\n(c) SWAP gate = 3 CNOTs")
    print("    SWAP = CNOT_01 * CNOT_10 * CNOT_01")
    print("    Gate count: 3 CNOT")

    print("\n(d) Toffoli gate requires:")
    print("    6 CNOTs + several single-qubit gates (H, T, T_dag)")
    print("    Standard decomposition: 6 CNOT + 2 H + 7 T/T_dag gates = 15 gates")


# === Exercise 5: Custom Gate ===
# Problem: Design gate mapping |0> -> (1/sqrt(3))|0> + sqrt(2/3)|1>.

def exercise_5():
    """Custom gate design and ZYZ decomposition."""
    # Target: U|0> = (1/sqrt(3))|0> + sqrt(2/3)|1>
    # This means U's first column is [(1/sqrt(3)), sqrt(2/3)]
    # For unitary, columns must be orthonormal
    # Second column (U|1>) must be orthogonal: [-sqrt(2/3), 1/sqrt(3)]

    a = 1 / np.sqrt(3)
    b = np.sqrt(2 / 3)
    U = np.array([[a, -b],
                   [b, a]], dtype=complex)

    print("Custom gate U:")
    print(f"  U = [{a:.4f}, {-b:.4f}]")
    print(f"      [{b:.4f}, {a:.4f}]")
    print(f"  Is unitary? {np.allclose(U @ U.conj().T, np.eye(2))}")

    # Verify mapping
    ket_0 = np.array([1, 0], dtype=complex)
    ket_1 = np.array([0, 1], dtype=complex)
    result_0 = U @ ket_0
    result_1 = U @ ket_1
    print(f"\n  U|0> = {np.round(result_0, 4)}")
    print(f"  Expected: [{a:.4f}, {b:.4f}]")
    print(f"  Match? {np.allclose(result_0, np.array([a, b]))}")
    print(f"\n  U|1> = {np.round(result_1, 4)}")
    print(f"  Orthogonal to U|0>? {np.isclose(np.vdot(result_0, result_1), 0)}")

    # ZYZ decomposition: U = e^{i*alpha} Rz(beta) Ry(gamma) Rz(delta)
    # This is a Ry rotation: U = Ry(gamma) where cos(gamma/2) = 1/sqrt(3)
    gamma = 2 * np.arccos(a)
    print(f"\n  ZYZ Decomposition:")
    print(f"    This gate is a pure Ry rotation: U = Ry({gamma:.4f})")
    print(f"    gamma = 2*arccos(1/sqrt(3)) = {gamma:.4f} rad = {gamma*180/np.pi:.2f} deg")
    print(f"    alpha = 0, beta = 0, delta = 0")

    # Verify
    U_reconstructed = Ry(gamma)
    print(f"    Ry({gamma:.4f}) = U? {np.allclose(U, U_reconstructed)}")


if __name__ == "__main__":
    print("=== Exercise 1: Gate Verification ===")
    exercise_1()
    print("\n=== Exercise 2: Bloch Sphere Rotations ===")
    exercise_2()
    print("\n=== Exercise 3: Controlled Gate Construction ===")
    exercise_3()
    print("\n=== Exercise 4: Gate Count ===")
    exercise_4()
    print("\n=== Exercise 5: Custom Gate ===")
    exercise_5()
    print("\nAll exercises completed!")
