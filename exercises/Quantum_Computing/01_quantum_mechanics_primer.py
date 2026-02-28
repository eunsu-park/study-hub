"""
Exercises for Lesson 01: Quantum Mechanics Primer
Topic: Quantum_Computing

Solutions to practice problems from the lesson.
"""

import numpy as np


# === Exercise 1: Normalization ===
# Problem: A qubit is in the state |psi> = (3/5)|0> + (4/5)*e^{i*pi/3}|1>.
# a) Verify normalized. b) Measurement probabilities. c) Phase effect on Z-basis.
# d) Phase effect on different basis.

def exercise_1():
    """Normalization, measurement probabilities, and phase effects."""
    alpha = 3 / 5
    beta = (4 / 5) * np.exp(1j * np.pi / 3)
    psi = np.array([alpha, beta], dtype=complex)

    # (a) Verify normalization
    norm_sq = np.abs(alpha) ** 2 + np.abs(beta) ** 2
    print(f"(a) |alpha|^2 + |beta|^2 = {np.abs(alpha)**2:.4f} + {np.abs(beta)**2:.4f} = {norm_sq:.4f}")
    print(f"    Normalized? {np.isclose(norm_sq, 1.0)}")

    # (b) Measurement probabilities
    p0 = np.abs(alpha) ** 2
    p1 = np.abs(beta) ** 2
    print(f"\n(b) P(|0>) = |3/5|^2 = {p0:.4f}")
    print(f"    P(|1>) = |4/5 * e^(i*pi/3)|^2 = {p1:.4f}")

    # (c) Does phase factor e^{i*pi/3} affect computational basis probabilities?
    beta_no_phase = 4 / 5
    p1_no_phase = np.abs(beta_no_phase) ** 2
    print(f"\n(c) Without phase: P(|1>) = |4/5|^2 = {p1_no_phase:.4f}")
    print(f"    With phase:    P(|1>) = |4/5 * e^(i*pi/3)|^2 = {p1:.4f}")
    print(f"    Same? {np.isclose(p1, p1_no_phase)}")
    print(f"    The phase e^(i*pi/3) does NOT affect computational basis probabilities.")

    # (d) Would phase matter in different basis?
    ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    ket_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)

    psi_with_phase = np.array([alpha, beta], dtype=complex)
    psi_no_phase = np.array([alpha, 4 / 5], dtype=complex)

    p_plus_with = np.abs(np.vdot(ket_plus, psi_with_phase)) ** 2
    p_plus_without = np.abs(np.vdot(ket_plus, psi_no_phase)) ** 2
    print(f"\n(d) In X-basis: P(|+>) with phase = {p_plus_with:.4f}")
    print(f"                P(|+>) without phase = {p_plus_without:.4f}")
    print(f"    Different? {not np.isclose(p_plus_with, p_plus_without)}")
    print(f"    Yes! Relative phase IS observable in a different measurement basis.")


# === Exercise 2: Interference Calculation ===
# Problem: alpha1 = 1/2, alpha2 = (1/2)*e^{i*phi}. Compute |alpha_total|^2 vs phi.

def exercise_2():
    """Interference of two complex amplitudes as a function of phase."""
    # (a) |alpha_total|^2 = |1/2 + 1/2 * e^{i*phi}|^2
    # = 1/4 + 1/4 + 2*(1/2)*(1/2)*cos(phi) = 1/2 + 1/2*cos(phi) = cos^2(phi/2)
    print("(a) |alpha_total|^2 = |1/2 + 1/2 * e^(i*phi)|^2")
    print("    = 1/4 + 1/4 + 2*(1/4)*cos(phi) = 1/2 * (1 + cos(phi)) = cos^2(phi/2)")

    # (b) Maximum at phi = 0
    phi_max = 0
    p_max = np.cos(phi_max / 2) ** 2
    print(f"\n(b) Maximum at phi = {phi_max}: P = {p_max:.4f}")

    # (c) Minimum at phi = pi
    phi_min = np.pi
    p_min = np.cos(phi_min / 2) ** 2
    print(f"\n(c) Minimum at phi = pi: P = {p_min:.4f}")

    # (d) Compute P(phi) for phi in [0, 2*pi]
    print("\n(d) P(phi) for selected values:")
    phis = np.linspace(0, 2 * np.pi, 9)
    print(f"    {'phi/pi':>8}  {'P(phi)':>8}  {'cos^2(phi/2)':>14}")
    print(f"    {'-'*34}")
    for phi in phis:
        alpha1 = 0.5
        alpha2 = 0.5 * np.exp(1j * phi)
        p_numerical = np.abs(alpha1 + alpha2) ** 2
        p_formula = np.cos(phi / 2) ** 2
        print(f"    {phi/np.pi:>8.3f}  {p_numerical:>8.4f}  {p_formula:>14.4f}")


# === Exercise 3: Dirac Notation Practice ===
# Problem: |psi> = (1/2)|0> + (sqrt(3)/2)|1>

def exercise_3():
    """Dirac notation: column vector, bra, inner product, projector."""
    alpha = 1 / 2
    beta = np.sqrt(3) / 2

    # (a) Column vector
    ket_psi = np.array([[alpha], [beta]], dtype=complex)
    print(f"(a) |psi> as column vector:\n    {ket_psi.flatten()}")

    # (b) Bra (conjugate transpose)
    bra_psi = ket_psi.conj().T
    print(f"\n(b) <psi| as row vector:\n    {bra_psi.flatten()}")

    # (c) <psi|psi>
    inner = (bra_psi @ ket_psi)[0, 0]
    print(f"\n(c) <psi|psi> = {inner:.4f} (should be 1.0)")

    # (d) Projector |psi><psi|
    projector = ket_psi @ bra_psi
    print(f"\n(d) |psi><psi| =")
    print(f"    [{projector[0, 0]:.4f}  {projector[0, 1]:.4f}]")
    print(f"    [{projector[1, 0]:.4f}  {projector[1, 1]:.4f}]")

    # (e) Verify (|psi><psi|)^2 = |psi><psi|
    proj_sq = projector @ projector
    is_projector = np.allclose(proj_sq, projector)
    print(f"\n(e) (|psi><psi|)^2 = |psi><psi|? {is_projector}")
    print(f"    Difference norm: {np.linalg.norm(proj_sq - projector):.2e}")


# === Exercise 4: Hilbert Space Exploration ===
# Problem: 4-qubit system dimensions, random state, measurement histogram.

def exercise_4():
    """Hilbert space dimensions and random state measurement."""
    n = 4
    dim = 2 ** n

    # (a) Complex amplitudes needed
    print(f"(a) A general {n}-qubit state needs {dim} complex amplitudes.")

    # (b) Real parameters (accounting for normalization and global phase)
    real_params = 2 * dim - 2
    print(f"\n(b) Real parameters = 2*{dim} - 2 = {real_params}")
    print(f"    (2 reals per complex number, minus 1 for normalization, minus 1 for global phase)")

    # (c) Random 4-qubit state, measure 10000 times, compare with amplitudes
    rng = np.random.default_rng(42)
    raw = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    state = raw / np.linalg.norm(raw)

    theoretical_probs = np.abs(state) ** 2
    num_shots = 10000
    outcomes = rng.choice(dim, size=num_shots, p=theoretical_probs)
    counts = np.bincount(outcomes, minlength=dim)
    empirical_probs = counts / num_shots

    print(f"\n(c) Random 4-qubit state measurement ({num_shots} shots):")
    print(f"    {'Basis':>6} {'|amp|^2':>10} {'Empirical':>10} {'Counts':>8}")
    print(f"    {'-'*38}")
    for i in range(dim):
        label = format(i, f'0{n}b')
        print(f"    |{label}> {theoretical_probs[i]:>10.4f} {empirical_probs[i]:>10.4f} {counts[i]:>8d}")

    chi_sq = np.sum((empirical_probs - theoretical_probs) ** 2 / (theoretical_probs + 1e-15))
    print(f"\n    Chi-squared statistic: {chi_sq:.6f} (small = good match)")


# === Exercise 5: Conceptual Understanding ===
# Problem: Conceptual questions about superposition, interference, and quantum randomness.

def exercise_5():
    """Conceptual answers demonstrated with code."""
    # (a) Why "simultaneously 0 and 1" is misleading
    print("(a) 'A qubit is simultaneously 0 and 1' is misleading because:")
    print("    The qubit is in ONE definite quantum state |psi> = alpha|0> + beta|1>.")
    print("    This state is DIFFERENT from either |0> or |1>.")
    print("    It has AMPLITUDES for each outcome, not that it IS both outcomes.")
    print("    More accurate: 'The qubit has probability amplitudes assigned to |0> and |1>.'")

    # (b) Can real-valued amplitudes produce destructive interference?
    print("\n(b) Can real amplitudes produce destructive interference?")
    a1 = 0.5
    a2 = -0.5
    print(f"    a1 = {a1}, a2 = {a2}")
    print(f"    |a1 + a2|^2 = |{a1} + ({a2})|^2 = {abs(a1 + a2)**2:.4f}")
    print(f"    YES! Real numbers CAN cancel (positive + negative = 0).")
    print(f"    Complex numbers are NOT strictly required for destructive interference.")
    print(f"    However, complex amplitudes are needed for the FULL range of quantum")
    print(f"    behavior, including all possible interference patterns and phase encoding.")

    # (c) Quantum vs classical randomness
    print("\n(c) Two differences between quantum and classical randomness:")
    print("    1. INTERFERENCE: Quantum probabilities can be LESS than the sum of")
    print("       individual probabilities (destructive interference). Classical")
    print("       probabilities always add.")

    alpha_a = 1 / np.sqrt(2)
    alpha_b = -1 / np.sqrt(2)
    p_quantum = abs(alpha_a + alpha_b) ** 2
    p_classical = abs(alpha_a) ** 2 + abs(alpha_b) ** 2
    print(f"       Example: Two paths with amplitudes {alpha_a:.4f} and {alpha_b:.4f}")
    print(f"       Quantum: |a+b|^2 = {p_quantum:.4f}, Classical: |a|^2+|b|^2 = {p_classical:.4f}")

    print("\n    2. NO-CLONING: Quantum states cannot be copied, unlike classical coins.")
    print("       A coin's state can be inspected and duplicated; a qubit's unknown")
    print("       state cannot. This is a fundamental physical law, not a technological")
    print("       limitation.")


if __name__ == "__main__":
    print("=== Exercise 1: Normalization ===")
    exercise_1()
    print("\n=== Exercise 2: Interference Calculation ===")
    exercise_2()
    print("\n=== Exercise 3: Dirac Notation Practice ===")
    exercise_3()
    print("\n=== Exercise 4: Hilbert Space Exploration ===")
    exercise_4()
    print("\n=== Exercise 5: Conceptual Understanding ===")
    exercise_5()
    print("\nAll exercises completed!")
