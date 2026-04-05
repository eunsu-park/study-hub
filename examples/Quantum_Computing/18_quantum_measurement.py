"""
18_quantum_measurement.py — Quantum Measurement Theory

Demonstrates:
  - Projective (von Neumann) measurement in computational basis
  - Measurement in arbitrary bases (Hadamard, arbitrary angle)
  - POVM (generalized) measurements
  - Weak measurement simulation
  - Quantum Zeno effect: frequent measurement freezes evolution
  - Measurement-induced state collapse and post-measurement statistics

All computations use pure NumPy.
"""

import numpy as np
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Pauli matrices and standard states
# ---------------------------------------------------------------------------

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (X + Z) / np.sqrt(2)  # Hadamard

KET_0 = np.array([1, 0], dtype=complex)
KET_1 = np.array([0, 1], dtype=complex)
KET_PLUS = (KET_0 + KET_1) / np.sqrt(2)
KET_MINUS = (KET_0 - KET_1) / np.sqrt(2)


def kron_list(ops: List[np.ndarray]) -> np.ndarray:
    """Tensor product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


# ---------------------------------------------------------------------------
# Projective measurement
# ---------------------------------------------------------------------------

def projective_measurement(state: np.ndarray,
                           projectors: List[np.ndarray]
                           ) -> Tuple[List[float], List[np.ndarray]]:
    """Perform projective measurement defined by a set of projectors.

    Why: In the von Neumann measurement postulate, measuring observable
    O = Σ_k λ_k P_k collapses the state |ψ⟩ to P_k|ψ⟩/||P_k|ψ⟩||
    with probability p_k = ⟨ψ|P_k|ψ⟩.  The projectors satisfy
    Σ_k P_k = I and P_k² = P_k.
    """
    probabilities = []
    post_states = []

    for P in projectors:
        prob = float(np.real(state.conj() @ P @ state))
        probabilities.append(prob)

        if prob > 1e-12:
            collapsed = P @ state
            collapsed /= np.linalg.norm(collapsed)
            post_states.append(collapsed)
        else:
            post_states.append(np.zeros_like(state))

    return probabilities, post_states


def measure_in_basis(state: np.ndarray,
                     basis_vectors: List[np.ndarray]
                     ) -> Tuple[List[float], List[np.ndarray]]:
    """Measure in an arbitrary orthonormal basis.

    Why: The computational basis {|0⟩, |1⟩} is not the only measurement
    basis.  Measuring in the X-basis {|+⟩, |−⟩} or any other basis
    yields different statistics.  The choice of measurement basis is
    a key resource in quantum protocols (e.g., BB84 QKD).
    """
    projectors = [np.outer(v, v.conj()) for v in basis_vectors]
    return projective_measurement(state, projectors)


# ---------------------------------------------------------------------------
# POVM measurement
# ---------------------------------------------------------------------------

def povm_measurement(state: np.ndarray,
                     effects: List[np.ndarray]
                     ) -> List[float]:
    """Compute outcome probabilities for a POVM measurement.

    Why: POVMs (Positive Operator-Valued Measures) generalize projective
    measurements.  POVM elements {E_k} satisfy E_k ≥ 0 and Σ_k E_k = I,
    but E_k need not be projectors (E_k² ≠ E_k in general).  POVMs can
    distinguish non-orthogonal states better than any projective measurement.
    """
    rho = np.outer(state, state.conj())
    probs = []
    for E in effects:
        p = float(np.real(np.trace(E @ rho)))
        probs.append(p)
    return probs


def optimal_unambiguous_discrimination(
        psi0: np.ndarray, psi1: np.ndarray
) -> Tuple[List[np.ndarray], float]:
    """Construct the optimal POVM for unambiguous state discrimination.

    Why: Given two non-orthogonal states, no measurement can distinguish
    them perfectly.  Unambiguous discrimination allows an inconclusive
    outcome but never misidentifies a state.  The failure probability
    is |⟨ψ₀|ψ₁⟩|, which is the optimal bound (Ivanovic-Dieks-Peres).
    """
    overlap = float(np.abs(psi0.conj() @ psi1))

    # Orthogonal complement of |ψ₁⟩
    psi1_perp = np.array([-psi1[1].conj(), psi1[0].conj()])
    psi1_perp /= np.linalg.norm(psi1_perp)

    # Orthogonal complement of |ψ₀⟩
    psi0_perp = np.array([-psi0[1].conj(), psi0[0].conj()])
    psi0_perp /= np.linalg.norm(psi0_perp)

    # POVM elements
    # E_0 detects |ψ₀⟩: proportional to |ψ₁_perp⟩⟨ψ₁_perp|
    c = 1.0 / (1.0 + overlap)
    E0 = c * np.outer(psi1_perp, psi1_perp.conj())
    E1 = c * np.outer(psi0_perp, psi0_perp.conj())
    E_fail = I - E0 - E1

    # Ensure E_fail is positive semidefinite (clamp small negatives)
    eigvals, eigvecs = np.linalg.eigh(E_fail)
    eigvals = np.maximum(eigvals, 0)
    E_fail = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T

    return [E0, E1, E_fail], overlap


# ---------------------------------------------------------------------------
# Quantum Zeno effect
# ---------------------------------------------------------------------------

def zeno_effect(n_measurements: int, total_time: float,
                omega: float) -> float:
    """Simulate the quantum Zeno effect for a two-level system.

    Why: If we measure a decaying quantum system frequently enough,
    the measurement "freezes" the evolution.  The survival probability
    after time T with n equally spaced measurements is:

        P_survive ≈ cos²(ωT/(2n))^n → 1 as n → ∞

    This is because each measurement projects back to the initial state,
    and for small intervals the transition probability is O((Δt)²),
    so n measurements give P_decay ~ n·(Δt)² = T²ω²/n → 0.
    """
    dt = total_time / n_measurements

    # Hamiltonian: H = (ω/2) X (Rabi oscillation between |0⟩ and |1⟩)
    # U(dt) = exp(-i H dt) = cos(ωdt/2)I - i sin(ωdt/2)X
    c = np.cos(omega * dt / 2)
    s = np.sin(omega * dt / 2)
    U = c * I - 1j * s * X

    state = KET_0.copy()
    survival = 1.0

    for _ in range(n_measurements):
        state = U @ state

        # Measure: project onto |0⟩
        prob_0 = float(np.abs(state[0]) ** 2)
        survival *= prob_0

        # Post-measurement state (collapse to |0⟩ if outcome is 0)
        if prob_0 > 1e-15:
            state = KET_0.copy()
        else:
            break

    return survival


# ---------------------------------------------------------------------------
# Weak measurement
# ---------------------------------------------------------------------------

def weak_measurement_simulation(state: np.ndarray, observable: np.ndarray,
                                coupling: float,
                                n_trials: int) -> Tuple[np.ndarray, float]:
    """Simulate weak measurement using a Gaussian pointer.

    Why: Weak measurements extract partial information about a quantum
    system without fully collapsing it.  The pointer state shifts by
    the weak value, but the disturbance to the system is small.
    For a pre-selected state |ψ⟩ and observable A, the pointer
    shifts proportionally to ⟨ψ|A|ψ⟩ (the expectation value) in the
    weak coupling limit.
    """
    # Eigendecompose observable
    eigvals, eigvecs = np.linalg.eigh(observable)

    # Probability of each eigenvalue
    probs = np.array([float(np.abs(eigvecs[:, k].conj() @ state) ** 2)
                      for k in range(len(eigvals))])

    # Simulate pointer readings: eigenvalue + Gaussian noise ~ 1/coupling
    pointer_width = 1.0 / coupling
    readings = np.zeros(n_trials)
    for i in range(n_trials):
        outcome = np.random.choice(len(eigvals), p=probs)
        readings[i] = eigvals[outcome] + np.random.normal(0, pointer_width)

    mean_reading = np.mean(readings)
    return readings, mean_reading


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_projective():
    """Projective measurement in computational and Hadamard bases."""
    print("=" * 60)
    print("DEMO 1: Projective Measurement")
    print("=" * 60)

    # Computational basis projectors
    P0 = np.outer(KET_0, KET_0.conj())
    P1 = np.outer(KET_1, KET_1.conj())

    states = [
        ("|0⟩", KET_0),
        ("|1⟩", KET_1),
        ("|+⟩", KET_PLUS),
        ("0.6|0⟩+0.8|1⟩", np.array([0.6, 0.8], dtype=complex)),
    ]

    print(f"\n  Measurement in Z-basis (computational):")
    print(f"  {'State':<20} {'P(0)':>8} {'P(1)':>8}")
    print(f"  {'─' * 38}")
    for label, psi in states:
        probs, _ = projective_measurement(psi, [P0, P1])
        print(f"  {label:<20} {probs[0]:8.4f} {probs[1]:8.4f}")

    print(f"\n  Measurement in X-basis (Hadamard):")
    print(f"  {'State':<20} {'P(+)':>8} {'P(−)':>8}")
    print(f"  {'─' * 38}")
    for label, psi in states:
        probs, _ = measure_in_basis(psi, [KET_PLUS, KET_MINUS])
        print(f"  {label:<20} {probs[0]:8.4f} {probs[1]:8.4f}")


def demo_measurement_statistics():
    """Verify Born rule with simulated measurement trials."""
    print("\n" + "=" * 60)
    print("DEMO 2: Born Rule Verification (Simulated Trials)")
    print("=" * 60)

    n_trials = 50000
    state = np.array([1, 1j], dtype=complex) / np.sqrt(2)  # (|0⟩ + i|1⟩)/√2
    p_theory = [0.5, 0.5]

    # Why: The Born rule states P(outcome k) = |⟨k|ψ⟩|².  By repeating
    # the measurement many times and counting outcomes, the empirical
    # frequencies should converge to the theoretical probabilities.
    outcomes = np.random.choice([0, 1], size=n_trials,
                                p=[float(np.abs(state[0]) ** 2),
                                   float(np.abs(state[1]) ** 2)])
    freq_0 = np.sum(outcomes == 0) / n_trials
    freq_1 = np.sum(outcomes == 1) / n_trials

    print(f"\n  State: (|0⟩ + i|1⟩)/√2")
    print(f"  Trials: {n_trials}")
    print(f"\n  {'Outcome':>10} {'Theory':>10} {'Measured':>10} {'Error':>10}")
    print(f"  {'─' * 44}")
    print(f"  {'|0⟩':>10} {p_theory[0]:10.4f} {freq_0:10.4f} "
          f"{abs(freq_0 - p_theory[0]):10.4f}")
    print(f"  {'|1⟩':>10} {p_theory[1]:10.4f} {freq_1:10.4f} "
          f"{abs(freq_1 - p_theory[1]):10.4f}")

    # Multi-qubit measurement
    print(f"\n  Two-qubit Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2:")
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    probs_4 = np.abs(bell) ** 2

    outcomes_2q = np.random.choice([0, 1, 2, 3], size=n_trials, p=probs_4)
    print(f"  {'Outcome':>10} {'Theory':>10} {'Measured':>10}")
    print(f"  {'─' * 34}")
    for k, label in enumerate(['|00⟩', '|01⟩', '|10⟩', '|11⟩']):
        freq = np.sum(outcomes_2q == k) / n_trials
        print(f"  {label:>10} {probs_4[k]:10.4f} {freq:10.4f}")


def demo_povm():
    """POVM for unambiguous state discrimination."""
    print("\n" + "=" * 60)
    print("DEMO 3: POVM — Unambiguous State Discrimination")
    print("=" * 60)

    # Why: Two non-orthogonal states cannot be perfectly distinguished.
    # A POVM with three outcomes (detect ψ₀, detect ψ₁, inconclusive)
    # can identify each state without error, at the cost of sometimes
    # returning "inconclusive."

    theta = np.pi / 6  # 30 degrees between states
    psi0 = np.array([np.cos(0), np.sin(0)], dtype=complex)
    psi1 = np.array([np.cos(theta), np.sin(theta)], dtype=complex)

    effects, overlap = optimal_unambiguous_discrimination(psi0, psi1)
    E0, E1, E_fail = effects

    print(f"\n  |ψ₀⟩ = |0⟩")
    print(f"  |ψ₁⟩ = cos({theta:.4f})|0⟩ + sin({theta:.4f})|1⟩")
    print(f"  |⟨ψ₀|ψ₁⟩| = {overlap:.4f}")

    # Verify POVM completeness
    total = E0 + E1 + E_fail
    print(f"\n  Completeness check E₀+E₁+E_fail = I: "
          f"{np.allclose(total, I)}")

    # Test on each state
    for label, psi in [("ψ₀", psi0), ("ψ₁", psi1)]:
        probs = povm_measurement(psi, effects)
        print(f"\n  Input: |{label}⟩")
        print(f"    P(detect ψ₀) = {probs[0]:.4f}")
        print(f"    P(detect ψ₁) = {probs[1]:.4f}")
        print(f"    P(inconclusive) = {probs[2]:.4f}")


def demo_zeno():
    """Demonstrate the quantum Zeno effect."""
    print("\n" + "=" * 60)
    print("DEMO 4: Quantum Zeno Effect")
    print("=" * 60)

    omega = 1.0
    T = np.pi  # Half period: without measurement, full transition |0⟩→|1⟩

    # Why: Without measurement, after time T = π/ω the system oscillates
    # fully from |0⟩ to |1⟩.  With frequent measurements, the system is
    # "frozen" in |0⟩.  This is the Zeno paradox: "a watched pot never boils."

    print(f"\n  Rabi oscillation: ω = {omega}, T = π (full transition time)")
    print(f"  Without measurement: P(survive in |0⟩) = cos²(ωT/2) = 0.0000")
    print(f"\n  {'Measurements':>14} {'P(survive)':>12} {'Zeno frozen?':>14}")
    print(f"  {'─' * 44}")

    for n in [1, 2, 5, 10, 20, 50, 100, 500, 1000]:
        p_surv = zeno_effect(n, T, omega)
        frozen = "Yes" if p_surv > 0.9 else ("Partial" if p_surv > 0.3 else "No")
        print(f"  {n:14d} {p_surv:12.6f} {frozen:>14}")


def demo_weak_measurement():
    """Simulate weak measurement of spin."""
    print("\n" + "=" * 60)
    print("DEMO 5: Weak Measurement")
    print("=" * 60)

    # Why: In weak measurement, the coupling between system and pointer
    # is small, so the system is barely disturbed.  The pointer reading
    # is noisy, but averaging over many trials recovers ⟨A⟩.
    # Surprisingly, the "weak value" ⟨A⟩_w = ⟨f|A|i⟩/⟨f|i⟩ can lie
    # outside the eigenvalue range when pre- and post-selected states
    # are nearly orthogonal.

    state = KET_PLUS  # Pre-selected in |+⟩
    n_trials = 10000

    print(f"\n  Pre-selected state: |+⟩")
    print(f"  Observable: Z (eigenvalues ±1)")
    print(f"  ⟨+|Z|+⟩ = {float(np.real(KET_PLUS.conj() @ Z @ KET_PLUS)):.4f}")

    print(f"\n  {'Coupling g':>12} {'Mean reading':>14} {'Std dev':>10} "
          f"{'Close to ⟨Z⟩?':>16}")
    print(f"  {'─' * 56}")

    for g in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        readings, mean_r = weak_measurement_simulation(state, Z, g, n_trials)
        std_r = np.std(readings)
        close = "Yes" if abs(mean_r) < 0.15 else "No"
        print(f"  {g:12.3f} {mean_r:14.4f} {std_r:10.4f} {close:>16}")

    # Why: For small g (weak coupling), readings are very noisy but the
    # mean converges to ⟨Z⟩ = 0.  For large g (strong coupling),
    # individual readings cluster around eigenvalues ±1.


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("+" + "=" * 58 + "+")
    print("|   Quantum Computing - 18: Quantum Measurement              |")
    print("+" + "=" * 58 + "+")

    np.random.seed(2026)

    demo_projective()
    demo_measurement_statistics()
    demo_povm()
    demo_zeno()
    demo_weak_measurement()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
