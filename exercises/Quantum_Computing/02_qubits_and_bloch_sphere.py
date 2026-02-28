"""
Exercises for Lesson 02: Qubits and the Bloch Sphere
Topic: Quantum_Computing

Solutions to practice problems from the lesson.
"""

import numpy as np


def qubit_to_bloch(state):
    """Convert a qubit state vector to Bloch sphere coordinates."""
    alpha, beta = state[0], state[1]
    if abs(alpha) > 1e-10:
        phase = np.exp(-1j * np.angle(alpha))
        alpha = alpha * phase
        beta = beta * phase
    alpha_real = np.clip(np.real(alpha), -1, 1)
    theta = 2 * np.arccos(alpha_real)
    phi = np.angle(beta) if abs(beta) > 1e-10 else 0.0
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return theta, phi, x, y, z


# === Exercise 1: Bloch Sphere Mapping ===
# Problem: Compute (theta, phi) and (x, y, z) for three given states.

def exercise_1():
    """Bloch sphere coordinates for specific states."""
    states = {
        "(a) cos(pi/8)|0> + sin(pi/8)|1>": np.array(
            [np.cos(np.pi / 8), np.sin(np.pi / 8)], dtype=complex),
        "(b) (1/sqrt(2))|0> + (e^{i*pi/3}/sqrt(2))|1>": np.array(
            [1 / np.sqrt(2), np.exp(1j * np.pi / 3) / np.sqrt(2)], dtype=complex),
        "(c) (1/2)|0> + (sqrt(3)/2)|1>": np.array(
            [1 / 2, np.sqrt(3) / 2], dtype=complex),
    }

    for name, state in states.items():
        theta, phi, x, y, z = qubit_to_bloch(state)
        print(f"{name}:")
        print(f"  theta = {theta:.4f} rad = {theta / np.pi:.4f}*pi")
        print(f"  phi   = {phi:.4f} rad = {phi / np.pi:.4f}*pi")
        print(f"  (x, y, z) = ({x:.4f}, {y:.4f}, {z:.4f})")
        print(f"  P(0) = {abs(state[0])**2:.4f}, P(1) = {abs(state[1])**2:.4f}")
        print()


# === Exercise 2: Phase Distinction ===
# Problem: Compare |psi1> = (|0>+|1>)/sqrt(2) and |psi2> = (|0>+e^{i*pi/4}|1>)/sqrt(2)

def exercise_2():
    """Distinguishing states by measuring in different bases."""
    psi1 = np.array([1, 1], dtype=complex) / np.sqrt(2)
    psi2 = np.array([1, np.exp(1j * np.pi / 4)], dtype=complex) / np.sqrt(2)

    # Z-basis
    print("(a) Z-basis measurement:")
    for name, psi in [("psi1", psi1), ("psi2", psi2)]:
        p0 = abs(psi[0]) ** 2
        p1 = abs(psi[1]) ** 2
        print(f"  {name}: P(0) = {p0:.4f}, P(1) = {p1:.4f}")
    print("  Distinguishable in Z-basis? No (both 50/50)")

    # X-basis
    ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    ket_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)
    print("\n(b) X-basis measurement:")
    for name, psi in [("psi1", psi1), ("psi2", psi2)]:
        p_plus = abs(np.vdot(ket_plus, psi)) ** 2
        p_minus = abs(np.vdot(ket_minus, psi)) ** 2
        print(f"  {name}: P(+) = {p_plus:.4f}, P(-) = {p_minus:.4f}")
    print("  Distinguishable in X-basis? Yes (different probabilities)")

    # (c) Maximally distinguishing basis
    print("\n(c) Finding maximally distinguishing basis:")
    print("  On the Bloch sphere:")
    for name, psi in [("psi1", psi1), ("psi2", psi2)]:
        _, phi_b, x, y, z = qubit_to_bloch(psi)
        print(f"  {name}: Bloch = ({x:.4f}, {y:.4f}, {z:.4f}), phi = {phi_b/np.pi:.4f}*pi")
    print("  The states differ by a relative phase of pi/4 (on the equator).")
    print("  Measure along the axis bisecting them for maximal distinction.")
    # Midpoint angle on equator
    phi_mid = np.pi / 8
    b0 = np.array([1, np.exp(1j * phi_mid)], dtype=complex) / np.sqrt(2)
    b1 = np.array([1, -np.exp(1j * phi_mid)], dtype=complex) / np.sqrt(2)
    print(f"  Optimal basis angle: phi = {phi_mid/np.pi:.4f}*pi")
    for name, psi in [("psi1", psi1), ("psi2", psi2)]:
        p0 = abs(np.vdot(b0, psi)) ** 2
        p1 = abs(np.vdot(b1, psi)) ** 2
        print(f"  {name}: P(b0) = {p0:.4f}, P(b1) = {p1:.4f}")


# === Exercise 3: Tensor Products ===
# Problem: Compute tensor products and check separability.

def exercise_3():
    """Tensor products and separability."""
    ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    ket_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)

    # (a) |+> tensor |->
    result_a = np.kron(ket_plus, ket_minus)
    print(f"(a) |+> x |-> = {result_a}")

    # (b) |-> tensor |+>
    result_b = np.kron(ket_minus, ket_plus)
    print(f"\n(b) |-> x |+> = {result_b}")
    print(f"    Same as (a)? {np.allclose(result_a, result_b)}")
    print(f"    Tensor product is NOT commutative!")

    # (c) (1/2)(|00>+|01>+|10>+|11>) = |+> x |+>
    state_c = np.array([1, 1, 1, 1], dtype=complex) / 2
    expected_c = np.kron(ket_plus, ket_plus)
    print(f"\n(c) (|00>+|01>+|10>+|11>)/2 = |+> x |+>")
    print(f"    Verification: {np.allclose(state_c, expected_c)}")

    # (d) (|01>+|10>)/sqrt(2) is NOT separable
    state_d = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
    C = state_d.reshape(2, 2)
    det = np.linalg.det(C)
    print(f"\n(d) State (|01>+|10>)/sqrt(2):")
    print(f"    Coefficient matrix:\n    {C}")
    print(f"    det(C) = {det:.4f}")
    print(f"    |det| = {abs(det):.4f} != 0, so the state is ENTANGLED")
    print(f"    This state cannot be written as a tensor product of single-qubit states.")


# === Exercise 4: State Space Counting ===
# Problem: Parameters for 3-qubit systems.

def exercise_4():
    """Counting parameters in multi-qubit state spaces."""
    n = 3
    dim = 2 ** n

    # (a) Real parameters for general 3-qubit state
    real_params_general = 2 * dim - 2
    print(f"(a) General {n}-qubit state:")
    print(f"    Dimension: {dim}")
    print(f"    Complex amplitudes: {dim}")
    print(f"    Real parameters: 2*{dim} - 2 = {real_params_general}")
    print(f"    (subtract 1 for normalization, 1 for global phase)")

    # (b) Real parameters for separable 3-qubit state
    real_params_separable = n * 2  # 2 real params per qubit (theta, phi on Bloch sphere)
    print(f"\n(b) Separable {n}-qubit state:")
    print(f"    Each qubit: 2 real parameters (theta, phi on Bloch sphere)")
    print(f"    Total: {n} * 2 = {real_params_separable} real parameters")

    # (c) Fraction
    fraction = real_params_separable / real_params_general
    print(f"\n(c) Fraction: {real_params_separable}/{real_params_general} = {fraction:.4f}")
    print(f"    Separable states use only {fraction*100:.1f}% of the parameter space.")
    print(f"    The vast majority of states are entangled!")
    print(f"    As n grows, this fraction -> 0 exponentially: 2n / (2^(n+1) - 2)")


# === Exercise 5: Bloch Sphere Trajectories ===
# Problem: Random qubit states, measurement probabilities in X, Y, Z bases.

def exercise_5():
    """Random Bloch sphere states and measurement in three bases."""
    rng = np.random.default_rng(42)

    # (a) Generate 100 random states uniformly on the Bloch sphere
    cos_theta = rng.uniform(-1, 1, 100)
    theta_vals = np.arccos(cos_theta)
    phi_vals = rng.uniform(0, 2 * np.pi, 100)

    states = []
    for theta, phi in zip(theta_vals, phi_vals):
        state = np.array([np.cos(theta / 2),
                          np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)
        states.append(state)

    # (b) Measurement probabilities in all three bases
    ket_0 = np.array([1, 0], dtype=complex)
    ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    ket_i = np.array([1, 1j], dtype=complex) / np.sqrt(2)

    sums = []
    print("(a-b) First 10 random states and their P(0) in each basis:")
    print(f"    {'theta/pi':>10} {'phi/pi':>10} {'P_Z(0)':>8} {'P_X(+)':>8} {'P_Y(i)':>8} {'Sum':>8}")
    print(f"    {'-'*58}")

    for idx, state in enumerate(states):
        p_z = abs(np.vdot(ket_0, state)) ** 2
        p_x = abs(np.vdot(ket_plus, state)) ** 2
        p_y = abs(np.vdot(ket_i, state)) ** 2
        total = p_z + p_x + p_y
        sums.append(total)
        if idx < 10:
            theta, phi, _, _, _ = qubit_to_bloch(state)
            print(f"    {theta/np.pi:>10.4f} {phi/np.pi:>10.4f} "
                  f"{p_z:>8.4f} {p_x:>8.4f} {p_y:>8.4f} {total:>8.4f}")

    # (c) Range of P(0)_Z + P(0)_X + P(0)_Y
    sums = np.array(sums)
    print(f"\n(c) Sum P_Z(0) + P_X(+) + P_Y(i) statistics over 100 random states:")
    print(f"    Min:  {sums.min():.4f}")
    print(f"    Max:  {sums.max():.4f}")
    print(f"    Mean: {sums.mean():.4f}")
    print(f"\n    Theoretical range: [1, 2]")
    print(f"    The sum equals (3 + x + y + z) / 2 where (x,y,z) is the Bloch vector.")
    print(f"    Since x^2+y^2+z^2 = 1 (pure states), we get 1 <= sum <= 2.")
    print(f"    Min = 1 (when x=y=z=-1/sqrt(3)), Max = 2 (when x=y=z=1/sqrt(3)).")


if __name__ == "__main__":
    print("=== Exercise 1: Bloch Sphere Mapping ===")
    exercise_1()
    print("\n=== Exercise 2: Phase Distinction ===")
    exercise_2()
    print("\n=== Exercise 3: Tensor Products ===")
    exercise_3()
    print("\n=== Exercise 4: State Space Counting ===")
    exercise_4()
    print("\n=== Exercise 5: Bloch Sphere Trajectories ===")
    exercise_5()
    print("\nAll exercises completed!")
