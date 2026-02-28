"""
Exercises for Lesson 06: Quantum Measurement
Topic: Quantum_Computing

Solutions to practice problems from the lesson.
"""

import numpy as np


# Standard definitions
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)
ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
ket_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)


def Rx(theta):
    return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                     [-1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)


def Ry(theta):
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                     [np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)


def Rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)


# === Exercise 1: Projective Measurement ===
# Problem: |psi> = (1+i)/2 |0> + 1/2 |1>

def exercise_1():
    """Projective measurement of a complex-amplitude state."""
    psi = np.array([(1 + 1j) / 2, 1 / 2], dtype=complex)
    norm_sq = np.vdot(psi, psi)
    print(f"State: |psi> = ((1+i)/2)|0> + (1/2)|1>")
    print(f"Normalization check: <psi|psi> = {norm_sq:.4f}")
    print(f"Normalized? {np.isclose(norm_sq, 1.0)}")

    # (a) Z-basis probabilities
    p0 = abs(psi[0]) ** 2
    p1 = abs(psi[1]) ** 2
    print(f"\n(a) Z-basis (computational basis):")
    print(f"    P(|0>) = |(1+i)/2|^2 = {p0:.4f}")
    print(f"    P(|1>) = |1/2|^2 = {p1:.4f}")

    # (b) X-basis probabilities
    p_plus = abs(np.vdot(ket_plus, psi)) ** 2
    p_minus = abs(np.vdot(ket_minus, psi)) ** 2
    print(f"\n(b) X-basis:")
    print(f"    P(|+>) = {p_plus:.4f}")
    print(f"    P(|->) = {p_minus:.4f}")

    # (c) Expectation values
    exp_X = np.real(np.vdot(psi, X @ psi))
    exp_Y = np.real(np.vdot(psi, Y @ psi))
    exp_Z = np.real(np.vdot(psi, Z @ psi))
    print(f"\n(c) Expectation values:")
    print(f"    <X> = {exp_X:.4f}")
    print(f"    <Y> = {exp_Y:.4f}")
    print(f"    <Z> = {exp_Z:.4f}")

    # (d) Bloch sphere coordinates
    x, y, z = exp_X, exp_Y, exp_Z
    print(f"\n(d) Bloch sphere coordinates:")
    print(f"    (x, y, z) = ({x:.4f}, {y:.4f}, {z:.4f})")
    print(f"    |r| = {np.sqrt(x**2 + y**2 + z**2):.4f} (=1 for pure state)")
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    print(f"    theta = {theta:.4f} rad = {theta * 180 / np.pi:.2f} deg")
    print(f"    phi = {phi:.4f} rad = {phi * 180 / np.pi:.2f} deg")


# === Exercise 2: Partial Measurement ===
# Problem: |psi> = (1/2)(|000> + |011> + |100> + |111>)

def exercise_2():
    """Partial measurement of a 3-qubit state."""
    psi = np.array([1, 0, 0, 1, 1, 0, 0, 1], dtype=complex) / 2
    # Verify: |000>=1/2, |011>=1/2, |100>=1/2, |111>=1/2
    print(f"State: (1/2)(|000> + |011> + |100> + |111>)")
    print(f"Normalization: {np.linalg.norm(psi):.4f}")

    def partial_measure(state, qubit, outcome, n_qubits):
        dim = 2 ** n_qubits
        new_state = np.zeros(dim, dtype=complex)
        for i in range(dim):
            if (i >> qubit) & 1 == outcome:
                new_state[i] = state[i]
        prob = np.sum(np.abs(new_state) ** 2)
        if prob > 1e-15:
            new_state /= np.sqrt(prob)
        return prob, new_state

    # (a) Measure qubit 2 (MSB), get |0>
    prob_a, state_a = partial_measure(psi, 2, 0, 3)
    print(f"\n(a) Measure qubit 2, get |0>:")
    print(f"    Probability: {prob_a:.4f}")
    print(f"    Post-measurement state:")
    for i in range(8):
        if abs(state_a[i]) > 1e-10:
            print(f"      |{format(i, '03b')}>: {state_a[i]:.4f}")

    # (b) Then measure qubit 1 of post-measurement state, get |1>
    prob_b, state_b = partial_measure(state_a, 1, 1, 3)
    print(f"\n(b) Then measure qubit 1, get |1>:")
    print(f"    Probability: {prob_b:.4f}")
    print(f"    Post-measurement state:")
    for i in range(8):
        if abs(state_b[i]) > 1e-10:
            print(f"      |{format(i, '03b')}>: {state_b[i]:.4f}")
    print(f"    Remaining state for qubit 0: |{1 if abs(state_b[3]) > 0.9 else 0}>")

    # (c) Reduced density matrix of qubit 0
    rho_full = np.outer(psi, psi.conj())
    rho_reshaped = rho_full.reshape(2, 2, 2, 2, 2, 2)
    # Trace out qubits 2 and 1 (keeping qubit 0)
    rho_0 = np.trace(rho_reshaped, axis1=0, axis2=3)  # trace qubit 2
    rho_0 = np.trace(rho_0, axis1=0, axis2=2)  # trace qubit 1
    print(f"\n(c) Reduced density matrix of qubit 0:")
    print(f"    rho_0 = ")
    print(f"      [{rho_0[0, 0]:.4f}  {rho_0[0, 1]:.4f}]")
    print(f"      [{rho_0[1, 0]:.4f}  {rho_0[1, 1]:.4f}]")

    eigenvalues = np.linalg.eigvalsh(rho_0)
    print(f"    Eigenvalues: {np.round(eigenvalues, 4)}")
    is_pure = np.isclose(np.trace(rho_0 @ rho_0), 1.0)
    print(f"    Tr(rho^2) = {np.real(np.trace(rho_0 @ rho_0)):.4f}")
    print(f"    Pure state? {is_pure}")
    if not is_pure:
        entropy = -np.sum(eigenvalues[eigenvalues > 1e-10]
                          * np.log2(eigenvalues[eigenvalues > 1e-10]))
        print(f"    Entropy: {entropy:.4f} bits (mixed state)")


# === Exercise 3: POVM Design ===
# Problem: Design POVM to distinguish |0>, |+>, |i>.

def exercise_3():
    """Minimum-error POVM for three non-orthogonal states."""
    states = {
        "|0>": ket_0,
        "|+>": ket_plus,
        "|i>": np.array([1, 1j], dtype=complex) / np.sqrt(2),
    }

    # For minimum-error discrimination with equal priors (1/3 each),
    # one approach: use the "pretty good measurement" (PGM).
    # POVM elements: E_k = rho^{-1/2} * (1/3) * |psi_k><psi_k| * rho^{-1/2}
    # where rho = (1/3) * sum_k |psi_k><psi_k|

    state_list = list(states.values())
    n_states = len(state_list)
    prior = 1 / n_states

    # Average state
    rho = np.zeros((2, 2), dtype=complex)
    for psi in state_list:
        rho += prior * np.outer(psi, psi.conj())

    # rho^{-1/2}
    eigvals, eigvecs = np.linalg.eigh(rho)
    rho_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.conj().T

    # POVM elements
    povm_elements = []
    for psi in state_list:
        E_k = rho_inv_sqrt @ (prior * np.outer(psi, psi.conj())) @ rho_inv_sqrt
        povm_elements.append(E_k)

    # Verify completeness
    total = sum(povm_elements)
    print(f"Pretty Good Measurement (PGM) POVM:")
    print(f"Completeness check (sum = I)? {np.allclose(total, np.eye(2))}")

    # Verify positive semi-definiteness
    for i, (name, E) in enumerate(zip(states.keys(), povm_elements)):
        eigvals_E = np.linalg.eigvalsh(E)
        print(f"\n  E_{name}:")
        print(f"    {np.round(E, 4)}")
        print(f"    Eigenvalues: {np.round(eigvals_E, 4)}")
        print(f"    PSD? {all(e >= -1e-10 for e in eigvals_E)}")

    # Test: probability of correct identification
    print(f"\nDiscrimination probabilities:")
    total_correct = 0
    for i, (name_in, psi) in enumerate(states.items()):
        print(f"  Input {name_in}:")
        for j, (name_out, E) in enumerate(zip(states.keys(), povm_elements)):
            prob = np.real(np.vdot(psi, E @ psi))
            marker = " <-- correct" if i == j else ""
            print(f"    P(detect {name_out}) = {prob:.4f}{marker}")
            if i == j:
                total_correct += prob

    avg_success = total_correct / n_states
    print(f"\n  Average success probability: {avg_success:.4f}")
    print(f"  Random guessing: {1/n_states:.4f}")
    print(f"  Improvement over random: {avg_success / (1/n_states):.2f}x")


# === Exercise 4: Quantum Zeno Variations ===
# Problem: Zeno effect with Rx, X-basis measurement, and anti-Zeno.

def exercise_4():
    """Quantum Zeno effect variations."""
    rng = np.random.default_rng(42)
    total_theta = np.pi / 2

    # (a) Rx rotation instead of Ry
    print("(a) Zeno effect with Rx(theta) instead of Ry(theta):")
    print(f"    {'N':>6} {'P_survive (Ry)':>16} {'P_survive (Rx)':>16}")
    print(f"    {'-'*42}")

    for N in [1, 5, 10, 50, 100]:
        step = total_theta / N

        # Ry version
        p_ry = np.cos(step / 2) ** (2 * N)

        # Rx version: Rx(theta)|0> has |<0|Rx(theta)|0>|^2 = cos^2(theta/2)
        # Same survival probability structure
        p_rx = np.cos(step / 2) ** (2 * N)

        print(f"    {N:>6} {p_ry:>16.6f} {p_rx:>16.6f}")

    print(f"    Zeno effect occurs for BOTH Rx and Ry.")
    print(f"    Both rotations move |0> away from itself at the same rate,")
    print(f"    so repeated Z-basis measurement freezes the state identically.")

    # (b) X-basis measurement after each Ry step
    print(f"\n(b) X-basis measurement after each Ry(theta/N) step:")
    print(f"    Starting from |0>, measuring in X-basis {{|+>, |->}}:")

    for N in [1, 5, 10, 50, 100]:
        step = total_theta / N
        n_trials = 10000
        survived = 0

        for _ in range(n_trials):
            state = ket_0.copy()
            alive = True
            # Track which X-basis outcome we got first
            first_outcome = None

            for k in range(N):
                state = Ry(step) @ state
                # Measure in X-basis
                p_plus = abs(np.vdot(ket_plus, state)) ** 2
                if rng.random() < p_plus:
                    state = ket_plus.copy()
                    if first_outcome is None:
                        first_outcome = "+"
                else:
                    state = ket_minus.copy()
                    if first_outcome is None:
                        first_outcome = "-"

            # Check if final state is still |0> (in Z basis)
            p0_final = abs(state[0]) ** 2
            survived += 1 if p0_final > 0.9 else 0

        print(f"    N={N:>3}: P(end in |0>) = {survived/n_trials:.4f}")

    print(f"    With X-basis measurement, the state is projected to |+> or |->.")
    print(f"    These are NOT eigenstates of Ry, so the Zeno effect does NOT")
    print(f"    freeze the state in |0>. The measurement basis matters!")

    # (c) Anti-Zeno effect (conceptual)
    print(f"\n(c) Anti-Zeno effect:")
    print(f"    The anti-Zeno effect occurs when frequent measurement ACCELERATES")
    print(f"    transitions rather than inhibiting them. This happens when:")
    print(f"    1. The system-environment coupling has a specific spectral structure")
    print(f"    2. The measurement rate matches a resonance frequency")
    print(f"    3. The energy spectrum of the reservoir is non-flat (structured)")
    print(f"    In the standard Zeno setting with unitary rotation, anti-Zeno")
    print(f"    does not occur. It requires open quantum systems with specific")
    print(f"    spectral densities (e.g., Ohmic or sub-Ohmic environments).")


# === Exercise 5: Measurement-Based Computation ===
# Problem: Cluster state + adaptive measurement implements gates.

def exercise_5():
    """Measurement-based computation on a 2-qubit cluster state."""
    # (a) Prepare cluster state: CZ(|+> tensor |+>)
    plus_plus = np.kron(ket_plus, ket_plus)

    # CZ gate in 2-qubit space
    CZ = np.diag([1, 1, 1, -1]).astype(complex)
    cluster = CZ @ plus_plus

    print(f"(a) Cluster state |CS> = CZ(|+> tensor |+>):")
    for i in range(4):
        if abs(cluster[i]) > 1e-10:
            print(f"    |{format(i, '02b')}>: {cluster[i]:.4f}")

    # (b) Measure qubit 0 in Rz(alpha) rotated basis
    # Basis: {Rz(alpha)|+>, Rz(alpha)|->}
    print(f"\n(b-c) Measurement-based gate implementation:")

    for alpha in [0, np.pi / 4, np.pi / 2, np.pi]:
        # Measurement basis vectors for qubit 0
        basis_0 = Rz(alpha) @ ket_plus
        basis_1 = Rz(alpha) @ ket_minus

        # Probability and post-measurement states for qubit 1
        for outcome, basis_vec in enumerate([basis_0, basis_1]):
            # Project qubit 0 onto basis_vec
            # <basis_vec|_0 |cluster>_{01} gives qubit 1 state
            qubit1_state = np.zeros(2, dtype=complex)
            for q0 in range(2):
                for q1 in range(2):
                    idx = q0 * 2 + q1
                    qubit1_state[q1] += basis_vec.conj()[q0] * cluster[idx]

            prob = np.sum(np.abs(qubit1_state) ** 2)
            if prob > 1e-10:
                qubit1_state /= np.sqrt(prob)

            # Compare with Rz(alpha)|+> or X*Rz(alpha)|+>
            target_0 = Rz(alpha) @ ket_plus
            target_1 = X @ Rz(alpha) @ ket_plus

            match_0 = np.isclose(abs(np.vdot(target_0, qubit1_state)), 1.0)
            match_1 = np.isclose(abs(np.vdot(target_1, qubit1_state)), 1.0)

            if prob > 0.01:
                label = f"alpha={alpha/np.pi:.2f}pi"
                print(f"    {label}, outcome={outcome}: "
                      f"prob={prob:.4f}, "
                      f"qubit1={np.round(qubit1_state, 3)}, "
                      f"~Rz(a)|+>? {match_0}, ~X*Rz(a)|+>? {match_1}")

    print(f"\n(d) Interpretation:")
    print(f"    Measuring qubit 0 in the Rz(alpha)-rotated basis effectively")
    print(f"    applies Rz(alpha) to qubit 1 (up to a Pauli X correction")
    print(f"    depending on the measurement outcome).")
    print(f"    This demonstrates that MEASUREMENT + ENTANGLEMENT can")
    print(f"    implement quantum gates -- the basis of measurement-based QC!")


if __name__ == "__main__":
    print("=== Exercise 1: Projective Measurement ===")
    exercise_1()
    print("\n=== Exercise 2: Partial Measurement ===")
    exercise_2()
    print("\n=== Exercise 3: POVM Design ===")
    exercise_3()
    print("\n=== Exercise 4: Quantum Zeno Variations ===")
    exercise_4()
    print("\n=== Exercise 5: Measurement-Based Computation ===")
    exercise_5()
    print("\nAll exercises completed!")
