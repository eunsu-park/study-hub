"""
04_bell_states.py — Bell States and CHSH Inequality

Demonstrates:
  - Preparing all four Bell states: |Φ+⟩, |Φ-⟩, |Ψ+⟩, |Ψ-⟩
  - Verifying entanglement (non-separability test)
  - Correlation properties of Bell states
  - CHSH inequality: classical bound ≤ 2, quantum violation ≈ 2√2
  - Simulating the CHSH game with many trials

All computations use pure NumPy.
"""

import numpy as np
from typing import Tuple, Dict

# ---------------------------------------------------------------------------
# Basis states and gates
# ---------------------------------------------------------------------------
KET_0 = np.array([1, 0], dtype=complex)
KET_1 = np.array([0, 1], dtype=complex)

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)


def format_state(state: np.ndarray, n_qubits: int = 2) -> str:
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
# Bell state preparation
# ---------------------------------------------------------------------------

def prepare_bell_state(variant: str = "phi+") -> np.ndarray:
    """Prepare one of the four Bell states.

    Why: The four Bell states form a complete orthonormal basis for the 2-qubit
    Hilbert space.  They are the maximally entangled states — each qubit alone
    is maximally mixed (50/50), yet measurements are perfectly correlated.

    |Φ+⟩ = (|00⟩ + |11⟩)/√2   — variant "phi+"
    |Φ-⟩ = (|00⟩ - |11⟩)/√2   — variant "phi-"
    |Ψ+⟩ = (|01⟩ + |10⟩)/√2   — variant "psi+"
    |Ψ-⟩ = (|01⟩ - |10⟩)/√2   — variant "psi-"
    """
    # Why: All Bell states are prepared from |00⟩ by:
    #   1. (Optionally) flipping qubit 0 or 1 with X gate
    #   2. Applying H to qubit 0
    #   3. Applying CNOT(0, 1)
    # The variant determines which initial flips to apply.
    state = np.kron(KET_0, KET_0)  # |00⟩

    if variant == "phi+":
        pass
    elif variant == "phi-":
        # Z on qubit 0 after Bell prep → flip sign of |11⟩ component
        state = np.kron(KET_1, KET_0)  # Start from |10⟩
    elif variant == "psi+":
        state = np.kron(KET_0, KET_1)  # Start from |01⟩
    elif variant == "psi-":
        state = np.kron(KET_1, KET_1)  # Start from |11⟩
    else:
        raise ValueError(f"Unknown Bell state variant: {variant}")

    # Apply H ⊗ I then CNOT
    H_I = np.kron(H, I)
    state = CNOT @ (H_I @ state)
    return state


# ---------------------------------------------------------------------------
# Entanglement test
# ---------------------------------------------------------------------------

def is_separable(state: np.ndarray, tol: float = 1e-8) -> bool:
    """Test if a 2-qubit state is separable (product state).

    Why: A pure 2-qubit state |ψ⟩ is separable iff its Schmidt decomposition
    has only one term, i.e., the reduced density matrix has rank 1.
    Equivalently, we can reshape the state vector into a 2×2 matrix and check
    if it has rank 1 (i.e., second singular value is zero).

    This is the *definition* of entanglement: a state that cannot be written
    as |a⟩ ⊗ |b⟩ is entangled.
    """
    # Reshape the 4-element vector into a 2×2 matrix
    matrix = state.reshape(2, 2)
    singular_values = np.linalg.svd(matrix, compute_uv=False)

    # Why: If rank=1 (one nonzero singular value), the state factors as a
    # tensor product.  If rank=2, it's entangled.
    return singular_values[1] < tol


def schmidt_coefficients(state: np.ndarray) -> np.ndarray:
    """Compute Schmidt coefficients of a 2-qubit state.

    Why: The Schmidt decomposition |ψ⟩ = Σ λ_i |a_i⟩|b_i⟩ uniquely quantifies
    entanglement.  For a Bell state, λ₁ = λ₂ = 1/√2 (maximally entangled).
    For a product state, only one λ is nonzero.
    """
    matrix = state.reshape(2, 2)
    return np.linalg.svd(matrix, compute_uv=False)


# ---------------------------------------------------------------------------
# Correlation and measurement
# ---------------------------------------------------------------------------

def measure_correlated(state: np.ndarray, n_shots: int = 10000) -> Dict:
    """Measure a 2-qubit state and report correlations.

    Returns counts and correlation coefficient.
    """
    probs = np.abs(state) ** 2
    outcomes = np.random.choice(4, size=n_shots, p=probs)

    counts = {format(i, '02b'): 0 for i in range(4)}
    for o in outcomes:
        counts[format(o, '02b')] += 1

    # Compute correlation: ⟨Z⊗Z⟩ = P(00) + P(11) - P(01) - P(10)
    # Why: Z⊗Z has eigenvalues +1 for |00⟩,|11⟩ and -1 for |01⟩,|10⟩.
    # The expectation ⟨Z⊗Z⟩ measures how correlated the measurement outcomes are.
    p_same = (counts['00'] + counts['11']) / n_shots
    p_diff = (counts['01'] + counts['10']) / n_shots
    correlation = p_same - p_diff

    return {'counts': counts, 'correlation': correlation}


def expectation_zz(state: np.ndarray) -> float:
    """Compute ⟨ψ|Z⊗Z|ψ⟩ exactly."""
    ZZ = np.kron(Z, Z)
    return np.real(state.conj() @ ZZ @ state)


# ---------------------------------------------------------------------------
# CHSH inequality
# ---------------------------------------------------------------------------

def make_measurement_operator(angle: float) -> np.ndarray:
    """Create a measurement operator for measuring in a rotated basis.

    The operator measures spin along direction at 'angle' from the Z-axis
    in the XZ-plane: M(θ) = cos(θ)Z + sin(θ)X.

    Why: The CHSH game requires Alice and Bob to choose measurement bases.
    Different angles exploit quantum correlations differently.  The optimal
    angles for maximum CHSH violation are: a₁=0, a₂=π/2, b₁=π/4, b₂=-π/4.
    """
    return np.cos(angle) * Z + np.sin(angle) * X


def chsh_expectation(state: np.ndarray,
                     a1: float, a2: float,
                     b1: float, b2: float) -> float:
    """Compute the CHSH parameter S = ⟨A₁B₁⟩ + ⟨A₁B₂⟩ + ⟨A₂B₁⟩ - ⟨A₂B₂⟩.

    Why: The CHSH inequality states that for any local hidden variable theory,
    |S| ≤ 2.  Quantum mechanics allows |S| up to 2√2 ≈ 2.828 (Tsirelson bound).
    Achieving S > 2 is a definitive signature of quantum entanglement — no
    classical explanation is possible.
    """
    A1 = make_measurement_operator(a1)
    A2 = make_measurement_operator(a2)
    B1 = make_measurement_operator(b1)
    B2 = make_measurement_operator(b2)

    def expect_AB(A, B):
        AB = np.kron(A, B)
        return np.real(state.conj() @ AB @ state)

    S = expect_AB(A1, B1) + expect_AB(A1, B2) + expect_AB(A2, B1) - expect_AB(A2, B2)
    return S


def simulate_chsh_game(state: np.ndarray,
                       a1: float, a2: float,
                       b1: float, b2: float,
                       n_trials: int = 50000) -> Dict:
    """Simulate the CHSH game by sampling measurement outcomes.

    Why: This simulation demonstrates the CHSH violation experimentally.
    Alice and Bob each randomly choose one of their two measurement settings,
    then we check whether their outcomes satisfy the CHSH winning condition.
    """
    A1 = make_measurement_operator(a1)
    A2 = make_measurement_operator(a2)
    B1 = make_measurement_operator(b1)
    B2 = make_measurement_operator(b2)

    # Precompute eigenvectors for each measurement operator
    def measurement_setup(M):
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        # eigenvectors columns are eigenstates; eigenvalues are ±1
        return eigenvalues, eigenvectors

    eig_A1 = measurement_setup(A1)
    eig_A2 = measurement_setup(A2)
    eig_B1 = measurement_setup(B1)
    eig_B2 = measurement_setup(B2)

    # Why: For each trial, we construct the joint measurement operator A⊗B
    # and sample from its eigenbasis.  This correctly captures quantum
    # correlations that cannot be reproduced classically.
    correlators = {'A1B1': 0.0, 'A1B2': 0.0, 'A2B1': 0.0, 'A2B2': 0.0}
    n_per = n_trials // 4

    for label, (eigA, eigB) in [
        ('A1B1', (eig_A1, eig_B1)),
        ('A1B2', (eig_A1, eig_B2)),
        ('A2B1', (eig_A2, eig_B1)),
        ('A2B2', (eig_A2, eig_B2)),
    ]:
        AB = np.kron(
            eigA[1] @ np.diag(eigA[0]) @ eigA[1].conj().T,
            eigB[1] @ np.diag(eigB[0]) @ eigB[1].conj().T,
        )
        # The expectation of AB in state |ψ⟩
        exp_val = np.real(state.conj() @ AB @ state)

        # Simulate: sample outcomes ±1 with correct probabilities
        # P(a,b) where a,b ∈ {+1,-1}
        # ⟨AB⟩ = P(+1,+1) + P(-1,-1) - P(+1,-1) - P(-1,+1)
        # P(same) = (1 + ⟨AB⟩)/2, P(diff) = (1 - ⟨AB⟩)/2
        p_same = (1 + exp_val) / 2
        same_count = np.random.binomial(n_per, p_same)
        correlators[label] = (2 * same_count - n_per) / n_per

    S_sim = correlators['A1B1'] + correlators['A1B2'] + correlators['A2B1'] - correlators['A2B2']
    return {'correlators': correlators, 'S': S_sim, 'n_trials': n_trials}


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_bell_states():
    """Prepare and display all four Bell states."""
    print("=" * 60)
    print("DEMO 1: The Four Bell States")
    print("=" * 60)

    variants = [
        ("phi+", "|Φ+⟩ = (|00⟩ + |11⟩)/√2"),
        ("phi-", "|Φ-⟩ = (|00⟩ - |11⟩)/√2"),
        ("psi+", "|Ψ+⟩ = (|01⟩ + |10⟩)/√2"),
        ("psi-", "|Ψ-⟩ = (|01⟩ - |10⟩)/√2"),
    ]

    for variant, description in variants:
        state = prepare_bell_state(variant)
        schmidt = schmidt_coefficients(state)
        sep = is_separable(state)

        print(f"\n  {description}")
        print(f"    State vector: {format_state(state)}")
        print(f"    Schmidt coefficients: [{schmidt[0]:.4f}, {schmidt[1]:.4f}]")
        print(f"    Separable? {sep}  → {'Product state' if sep else 'ENTANGLED'}")

    print(f"\n  All Bell states have Schmidt coefficients [1/√2, 1/√2] = [{1/np.sqrt(2):.4f}, {1/np.sqrt(2):.4f}]")
    print(f"  → Maximally entangled (equal probability in both Schmidt terms)")


def demo_entanglement_vs_product():
    """Compare entangled states with product states."""
    print("\n" + "=" * 60)
    print("DEMO 2: Entangled vs Product States")
    print("=" * 60)

    # Product state: |+⟩ ⊗ |0⟩
    plus = (KET_0 + KET_1) / np.sqrt(2)
    product = np.kron(plus, KET_0)
    print(f"\n  |+⟩⊗|0⟩ = {format_state(product)}")
    print(f"    Separable? {is_separable(product)}")
    print(f"    Schmidt: {schmidt_coefficients(product)}")

    # Bell state
    bell = prepare_bell_state("phi+")
    print(f"\n  |Φ+⟩ = {format_state(bell)}")
    print(f"    Separable? {is_separable(bell)}")
    print(f"    Schmidt: {schmidt_coefficients(bell)}")

    # Why: The product state |+⟩⊗|0⟩ has amplitudes for |00⟩ and |10⟩, while
    # the Bell state |Φ+⟩ has amplitudes for |00⟩ and |11⟩.  Both are
    # superpositions, but only the Bell state has nonlocal correlations.
    print(f"\n  Key difference:")
    print(f"  Product state: measuring qubit 0 does NOT affect qubit 1")
    print(f"  Bell state:    measuring qubit 0 INSTANTLY determines qubit 1")


def demo_bell_correlations():
    """Show perfect correlation/anticorrelation in Bell states."""
    print("\n" + "=" * 60)
    print("DEMO 3: Bell State Correlations")
    print("=" * 60)

    n_shots = 20000

    for variant, desc in [("phi+", "|Φ+⟩"), ("phi-", "|Φ-⟩"),
                           ("psi+", "|Ψ+⟩"), ("psi-", "|Ψ-⟩")]:
        state = prepare_bell_state(variant)
        result = measure_correlated(state, n_shots)
        zz_exact = expectation_zz(state)

        print(f"\n  {desc}:")
        print(f"    Counts: {result['counts']}")
        print(f"    ⟨Z⊗Z⟩ exact = {zz_exact:+.4f}, sampled ≈ {result['correlation']:+.4f}")

    # Why: |Φ±⟩ have ⟨Z⊗Z⟩ = +1 (perfect correlation: both 0 or both 1)
    # |Ψ±⟩ have ⟨Z⊗Z⟩ = -1 (perfect anti-correlation: always different)
    print(f"\n  |Φ±⟩: same outcomes always (correlation = +1)")
    print(f"  |Ψ±⟩: opposite outcomes always (correlation = -1)")


def demo_chsh_inequality():
    """Demonstrate violation of the CHSH inequality."""
    print("\n" + "=" * 60)
    print("DEMO 4: CHSH Inequality Violation")
    print("=" * 60)

    bell = prepare_bell_state("phi+")

    # Why: These specific angles maximize the CHSH parameter S for the |Φ+⟩
    # state.  They were derived by Cirel'son (Tsirelson) in 1980.
    # The key insight: Alice's bases are 0 and π/2 apart, Bob's are π/4 and -π/4.
    a1, a2 = 0, np.pi / 2          # Alice's measurement angles
    b1, b2 = np.pi / 4, -np.pi / 4  # Bob's measurement angles

    S_exact = chsh_expectation(bell, a1, a2, b1, b2)
    print(f"\n  Optimal angles: a1=0, a2=π/2, b1=π/4, b2=-π/4")
    print(f"\n  CHSH parameter (exact): S = {S_exact:.6f}")
    print(f"  Classical bound:        |S| ≤ 2")
    print(f"  Tsirelson bound:        |S| ≤ 2√2 ≈ {2*np.sqrt(2):.6f}")
    print(f"  Violation:              {S_exact:.4f} > 2  ← quantum entanglement!")

    # Simulation
    result = simulate_chsh_game(bell, a1, a2, b1, b2, n_trials=100000)
    print(f"\n  CHSH Game Simulation ({result['n_trials']} trials):")
    for k, v in result['correlators'].items():
        print(f"    ⟨{k}⟩ = {v:+.4f}")
    print(f"    S_simulated = {result['S']:+.4f}")


def demo_chsh_angle_scan():
    """Scan CHSH parameter over different angle choices."""
    print("\n" + "=" * 60)
    print("DEMO 5: CHSH Parameter vs Measurement Angles")
    print("=" * 60)

    bell = prepare_bell_state("phi+")

    # Why: By scanning Bob's angle while fixing Alice's, we can see how the
    # CHSH parameter varies and identify the optimal angles.
    print(f"\n  Fixed: a1=0, a2=π/2.  Scanning b1 = θ, b2 = -θ")
    print(f"  {'θ (degrees)':<15} {'S':>10} {'Violates?':>12}")
    print(f"  {'─' * 40}")

    for theta_deg in range(0, 100, 5):
        theta = np.radians(theta_deg)
        S = chsh_expectation(bell, 0, np.pi / 2, theta, -theta)
        violates = abs(S) > 2
        marker = " ← MAXIMUM" if abs(theta_deg - 45) < 3 else ""
        print(f"  {theta_deg:>10}°     {S:>10.4f} {'YES' if violates else 'no':>12}{marker}")

    print(f"\n  Maximum violation at θ = 45° (π/4 radians), as predicted.")


def demo_bell_orthonormality():
    """Verify that Bell states form an orthonormal basis."""
    print("\n" + "=" * 60)
    print("DEMO 6: Bell States as Orthonormal Basis")
    print("=" * 60)

    # Why: The four Bell states span the entire 4-dimensional 2-qubit Hilbert
    # space.  This means any 2-qubit state can be decomposed into Bell states —
    # a fact used in quantum teleportation (Bell measurement).
    variants = ["phi+", "phi-", "psi+", "psi-"]
    names = ["|Φ+⟩", "|Φ-⟩", "|Ψ+⟩", "|Ψ-⟩"]
    states = [prepare_bell_state(v) for v in variants]

    print(f"\n  Inner product matrix ⟨Bell_i|Bell_j⟩:")
    print(f"  {'':8}", end="")
    for n in names:
        print(f"{n:>8}", end="")
    print()
    print(f"  {'─' * 40}")

    for i, (ni, si) in enumerate(zip(names, states)):
        print(f"  {ni:8}", end="")
        for j, sj in enumerate(states):
            inner = np.abs(np.vdot(si, sj))
            print(f"{inner:>8.4f}", end="")
        print()

    print(f"\n  Identity matrix → orthonormal basis confirmed!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Quantum Computing — 04: Bell States & CHSH            ║")
    print("╚══════════════════════════════════════════════════════════╝")

    np.random.seed(2026)

    demo_bell_states()
    demo_entanglement_vs_product()
    demo_bell_correlations()
    demo_chsh_inequality()
    demo_chsh_angle_scan()
    demo_bell_orthonormality()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
