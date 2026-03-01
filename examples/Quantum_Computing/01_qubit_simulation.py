"""
01_qubit_simulation.py — Qubit State Vectors, Bloch Sphere, and Measurement

Demonstrates the fundamental building blocks of quantum computing:
  - Representing qubit states as 2D complex vectors (α|0⟩ + β|1⟩)
  - Converting states to Bloch sphere coordinates (θ, φ)
  - Simulating measurement via the Born rule
  - Verifying that measurement statistics match theoretical probabilities

All computations use pure NumPy — no quantum computing libraries required.
"""

import numpy as np
from typing import Tuple, Dict

# ---------------------------------------------------------------------------
# Computational basis states
# ---------------------------------------------------------------------------

# Why: |0⟩ and |1⟩ form the computational basis — every qubit state is a
# superposition α|0⟩ + β|1⟩ where |α|² + |β|² = 1 (normalization).
KET_0 = np.array([1, 0], dtype=complex)
KET_1 = np.array([0, 1], dtype=complex)

# Why: |+⟩ and |−⟩ are the eigenstates of the Pauli-X (NOT) gate and form the
# Hadamard basis. They appear constantly in interference-based algorithms.
KET_PLUS = (KET_0 + KET_1) / np.sqrt(2)
KET_MINUS = (KET_0 - KET_1) / np.sqrt(2)


# ---------------------------------------------------------------------------
# Qubit construction helpers
# ---------------------------------------------------------------------------

def make_qubit(alpha: complex, beta: complex) -> np.ndarray:
    """Create a qubit state α|0⟩ + β|1⟩, normalizing automatically.

    Why: In theory every qubit must satisfy |α|² + |β|² = 1.  Rather than
    requiring callers to pre-normalize, we do it here so that downstream
    functions (measurement, Bloch conversion) always receive valid states.
    """
    state = np.array([alpha, beta], dtype=complex)
    norm = np.linalg.norm(state)
    if norm < 1e-15:
        raise ValueError("Cannot normalize the zero vector — not a valid qubit state.")
    return state / norm


def make_qubit_from_angles(theta: float, phi: float) -> np.ndarray:
    """Construct a qubit from Bloch sphere angles θ ∈ [0, π] and φ ∈ [0, 2π).

    The parametrization is:
        |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩

    Why: This is the *standard* Bloch parametrization.  The factor of θ/2
    (not θ) ensures that antipodal points on the sphere correspond to
    orthogonal quantum states — a subtle but critical mapping between the
    2-sphere and the projective Hilbert space of a qubit.
    """
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    return np.array([alpha, beta], dtype=complex)


# ---------------------------------------------------------------------------
# Bloch sphere coordinates
# ---------------------------------------------------------------------------

def state_to_bloch(state: np.ndarray) -> Tuple[float, float, float]:
    """Convert a qubit state vector to Bloch sphere Cartesian coordinates (x, y, z).

    Why: The Bloch sphere gives a geometric picture of a single qubit.
    The Cartesian coordinates are the expectation values of the Pauli matrices:
        x = ⟨ψ|X|ψ⟩,  y = ⟨ψ|Y|ψ⟩,  z = ⟨ψ|Z|ψ⟩
    This makes rotations around axes correspond to Pauli rotations — an
    invaluable tool for visualizing gate operations.
    """
    # Why: We extract θ and φ from the amplitudes, then convert to Cartesian.
    # Removing global phase first: multiply so that α is real and non-negative.
    alpha, beta = state[0], state[1]
    if np.abs(alpha) > 1e-15:
        phase = np.exp(-1j * np.angle(alpha))
        alpha = (alpha * phase).real
        beta = beta * phase
    else:
        # α ≈ 0 → state is near |1⟩; Bloch south pole
        return (0.0, 0.0, -1.0)

    theta = 2 * np.arccos(np.clip(alpha.real, -1, 1))
    phi = np.angle(beta) if np.abs(beta) > 1e-15 else 0.0

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return (float(x), float(y), float(z))


def state_to_bloch_angles(state: np.ndarray) -> Tuple[float, float]:
    """Return (θ, φ) Bloch sphere angles for a qubit state."""
    alpha, beta = state[0], state[1]
    if np.abs(alpha) > 1e-15:
        phase = np.exp(-1j * np.angle(alpha))
        alpha = (alpha * phase).real
        beta = beta * phase
    else:
        return (np.pi, 0.0)

    theta = 2 * np.arccos(np.clip(alpha.real, -1, 1))
    phi = np.angle(beta) if np.abs(beta) > 1e-15 else 0.0
    return (float(theta), float(phi))


# ---------------------------------------------------------------------------
# Measurement simulation
# ---------------------------------------------------------------------------

def measure(state: np.ndarray, n_shots: int = 10000) -> Dict[str, object]:
    """Simulate projective measurement in the computational basis.

    Returns a dict with:
        'outcomes'   — array of sampled basis-state indices
        'counts'     — dict mapping basis label to count
        'probs_theory' — theoretical probabilities from Born rule
        'probs_expt'   — experimental frequencies from sampling

    Why: The Born rule states P(k) = |⟨k|ψ⟩|² = |amplitude_k|².
    We use numpy's weighted random choice to mimic this probabilistic collapse.
    With enough shots the experimental frequencies converge to the Born-rule
    probabilities — a direct demonstration of quantum statistical behavior.
    """
    probs = np.abs(state) ** 2

    # Why: np.random.choice with p=probs implements the Born rule sampling.
    # Each call is an independent "measurement" of identically prepared qubits.
    outcomes = np.random.choice(len(state), size=n_shots, p=probs)

    n_qubits = int(np.log2(len(state)))
    counts: Dict[str, int] = {}
    for idx in range(len(state)):
        label = format(idx, f'0{n_qubits}b')
        counts[label] = int(np.sum(outcomes == idx))

    probs_expt = {label: c / n_shots for label, c in counts.items()}
    probs_theory = {}
    for idx in range(len(state)):
        label = format(idx, f'0{n_qubits}b')
        probs_theory[label] = float(probs[idx])

    return {
        'outcomes': outcomes,
        'counts': counts,
        'probs_theory': probs_theory,
        'probs_expt': probs_expt,
    }


def measure_single_qubit(state: np.ndarray, n_shots: int = 10000) -> Dict[str, object]:
    """Convenience wrapper for single-qubit measurement with pretty labels."""
    result = measure(state, n_shots)
    # Relabel '0' → '|0⟩', '1' → '|1⟩' for readability
    relabel = {'0': '|0⟩', '1': '|1⟩'}
    result['counts'] = {relabel.get(k, k): v for k, v in result['counts'].items()}
    result['probs_theory'] = {relabel.get(k, k): v for k, v in result['probs_theory'].items()}
    result['probs_expt'] = {relabel.get(k, k): v for k, v in result['probs_expt'].items()}
    return result


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def format_state(state: np.ndarray, n_qubits: int = 1) -> str:
    """Pretty-print a state vector in Dirac notation."""
    terms = []
    for idx in range(len(state)):
        amp = state[idx]
        if np.abs(amp) < 1e-10:
            continue
        label = format(idx, f'0{n_qubits}b')
        # Format amplitude
        if np.abs(amp.imag) < 1e-10:
            amp_str = f"{amp.real:+.4f}"
        else:
            amp_str = f"({amp.real:+.4f}{amp.imag:+.4f}j)"
        terms.append(f"{amp_str}|{label}⟩")
    return " ".join(terms) if terms else "0"


def print_measurement_results(result: Dict, label: str = "") -> None:
    """Print measurement results in a readable table."""
    if label:
        print(f"\n  Measurement of {label}:")
    print(f"  {'Outcome':<10} {'Theory':>10} {'Experiment':>10} {'Counts':>8}")
    print(f"  {'─' * 42}")
    for key in sorted(result['probs_theory'].keys()):
        th = result['probs_theory'][key]
        ex = result['probs_expt'][key]
        ct = result['counts'][key]
        print(f"  {key:<10} {th:>10.4f} {ex:>10.4f} {ct:>8d}")


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_basis_states():
    """Show properties of the standard basis states."""
    print("=" * 60)
    print("DEMO 1: Computational and Hadamard Basis States")
    print("=" * 60)

    states = {
        '|0⟩': KET_0,
        '|1⟩': KET_1,
        '|+⟩': KET_PLUS,
        '|−⟩': KET_MINUS,
    }

    for name, state in states.items():
        x, y, z = state_to_bloch(state)
        theta, phi = state_to_bloch_angles(state)
        print(f"\n  {name} = {format_state(state)}")
        print(f"    Bloch: (x={x:.3f}, y={y:.3f}, z={z:.3f})")
        print(f"    Angles: θ={theta:.4f} rad ({np.degrees(theta):.1f}°), "
              f"φ={phi:.4f} rad ({np.degrees(phi):.1f}°)")

    # Why: Orthogonal states map to antipodal points on the Bloch sphere.
    # |0⟩ → north pole (z=+1), |1⟩ → south pole (z=−1)
    # |+⟩ → +x axis, |−⟩ → −x axis
    print("\n  Key insight: orthogonal states sit at opposite poles of the Bloch sphere.")


def demo_arbitrary_states():
    """Create and inspect arbitrary qubit states."""
    print("\n" + "=" * 60)
    print("DEMO 2: Arbitrary Qubit States")
    print("=" * 60)

    # Why: An arbitrary qubit with unequal amplitudes demonstrates partial
    # superposition — neither fully |0⟩ nor fully |1⟩.
    psi = make_qubit(np.sqrt(0.3), np.sqrt(0.7))
    x, y, z = state_to_bloch(psi)
    print(f"\n  |ψ⟩ = √0.3|0⟩ + √0.7|1⟩")
    print(f"    State vector: {format_state(psi)}")
    print(f"    Bloch: (x={x:.3f}, y={y:.3f}, z={z:.3f})")
    print(f"    P(|0⟩) = |α|² = {np.abs(psi[0])**2:.4f}")
    print(f"    P(|1⟩) = |β|² = {np.abs(psi[1])**2:.4f}")

    # State with a relative phase
    # Why: The relative phase e^{iφ} between amplitudes has no effect on
    # measurement probabilities in the Z-basis, but determines the azimuthal
    # angle φ on the Bloch sphere — it becomes visible in X or Y measurements.
    psi_phase = make_qubit(1 / np.sqrt(2), np.exp(1j * np.pi / 4) / np.sqrt(2))
    x, y, z = state_to_bloch(psi_phase)
    print(f"\n  |ψ⟩ = (1/√2)|0⟩ + (e^{{iπ/4}}/√2)|1⟩")
    print(f"    Bloch: (x={x:.3f}, y={y:.3f}, z={z:.3f})")
    print(f"    Note: z≈0 (equator) because P(|0⟩)=P(|1⟩)=0.5")
    print(f"    The phase shifts the point around the equator.")


def demo_roundtrip_bloch():
    """Verify Bloch ↔ state vector round-trip conversion."""
    print("\n" + "=" * 60)
    print("DEMO 3: Bloch Sphere Round-Trip Verification")
    print("=" * 60)

    # Why: Testing the round-trip (angles → state → angles) catches
    # implementation bugs in the trigonometric conversions, especially
    # at the poles where φ is degenerate.
    test_angles = [
        (0.0, 0.0, "|0⟩ (north pole)"),
        (np.pi, 0.0, "|1⟩ (south pole)"),
        (np.pi / 2, 0.0, "|+⟩ (equator, x+)"),
        (np.pi / 2, np.pi, "|−⟩ (equator, x−)"),
        (np.pi / 2, np.pi / 2, "|+i⟩ (equator, y+)"),
        (np.pi / 3, np.pi / 4, "arbitrary"),
        (2 * np.pi / 3, 5 * np.pi / 3, "arbitrary 2"),
    ]

    print(f"\n  {'Description':<25} {'θ_in':>7} {'φ_in':>7}  →  {'θ_out':>7} {'φ_out':>7} {'Match?':>7}")
    print(f"  {'─' * 72}")

    for theta_in, phi_in, desc in test_angles:
        state = make_qubit_from_angles(theta_in, phi_in)
        theta_out, phi_out = state_to_bloch_angles(state)

        # Normalize phi to [0, 2π) for comparison
        phi_in_norm = phi_in % (2 * np.pi)
        phi_out_norm = phi_out % (2 * np.pi)

        # At the poles θ=0 or π, φ is undefined — any value is correct
        if theta_in < 1e-10 or np.abs(theta_in - np.pi) < 1e-10:
            match = np.abs(theta_out - theta_in) < 1e-6
        else:
            match = (np.abs(theta_out - theta_in) < 1e-6 and
                     np.abs(phi_out_norm - phi_in_norm) < 1e-6)

        print(f"  {desc:<25} {theta_in:>7.4f} {phi_in:>7.4f}  →  "
              f"{theta_out:>7.4f} {phi_out:>7.4f} {'  ✓' if match else '  ✗':>7}")


def demo_measurement():
    """Simulate measurements and compare with Born rule predictions."""
    print("\n" + "=" * 60)
    print("DEMO 4: Measurement Simulation (Born Rule)")
    print("=" * 60)

    n_shots = 50000
    print(f"  Using {n_shots} shots per measurement\n")

    # |0⟩ — deterministic outcome
    result = measure_single_qubit(KET_0, n_shots)
    print_measurement_results(result, "|0⟩ (deterministic)")

    # |+⟩ — equal superposition
    result = measure_single_qubit(KET_PLUS, n_shots)
    print_measurement_results(result, "|+⟩ (equal superposition)")

    # Biased superposition: √0.2|0⟩ + √0.8|1⟩
    psi = make_qubit(np.sqrt(0.2), np.sqrt(0.8))
    result = measure_single_qubit(psi, n_shots)
    print_measurement_results(result, "√0.2|0⟩ + √0.8|1⟩ (biased)")

    # Why: With many shots, the empirical frequency should closely match the
    # theoretical probability — the deviation shrinks as O(1/√n_shots).
    # This is the quantum-mechanical analog of the classical law of large numbers.
    print(f"\n  With {n_shots} shots, typical deviation ≈ {1/np.sqrt(n_shots):.4f}")


def demo_normalization_constraint():
    """Show that probabilities always sum to 1."""
    print("\n" + "=" * 60)
    print("DEMO 5: Normalization Constraint")
    print("=" * 60)

    np.random.seed(42)
    print("\n  Generating 5 random qubit states and checking normalization:")
    print(f"  {'State':<40} {'|α|² + |β|²':>12}")
    print(f"  {'─' * 54}")

    for i in range(5):
        # Why: Random complex amplitudes won't be normalized — make_qubit fixes this.
        # This demonstrates that the normalization constraint is fundamental, not optional.
        alpha = complex(np.random.randn(), np.random.randn())
        beta = complex(np.random.randn(), np.random.randn())
        state = make_qubit(alpha, beta)
        norm_sq = np.abs(state[0]) ** 2 + np.abs(state[1]) ** 2
        print(f"  {format_state(state):<40} {norm_sq:>12.10f}")

    print("\n  All states have |α|² + |β|² = 1 (up to floating-point precision).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Quantum Computing — 01: Qubit State Simulation        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    np.random.seed(2026)

    demo_basis_states()
    demo_arbitrary_states()
    demo_roundtrip_bloch()
    demo_measurement()
    demo_normalization_constraint()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
