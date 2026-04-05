"""
14_quantum_noise.py — Noise and Quantum Channels

Demonstrates:
  - Kraus operator formalism for quantum channels
  - Depolarizing channel on single-qubit density matrices
  - Amplitude damping channel (T1 relaxation)
  - Phase damping channel (T2 dephasing)
  - Process tomography simulation (χ-matrix reconstruction)
  - Channel composition and fidelity decay

All computations use pure NumPy.
"""

import numpy as np
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Pauli matrices and helpers
# ---------------------------------------------------------------------------
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS = [I, X, Y, Z]
PAULI_LABELS = ['I', 'X', 'Y', 'Z']

KET_0 = np.array([1, 0], dtype=complex)
KET_1 = np.array([0, 1], dtype=complex)
KET_PLUS = (KET_0 + KET_1) / np.sqrt(2)


def density_matrix(state: np.ndarray) -> np.ndarray:
    """Convert a pure state vector to a density matrix."""
    return np.outer(state, state.conj())


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute fidelity between two density matrices.

    Why: For a pure target state σ = |ψ⟩⟨ψ|, fidelity simplifies to
    F = ⟨ψ|ρ|ψ⟩.  For general mixed states we use the full formula,
    but the pure-state shortcut is numerically simpler and common in
    noise analysis.
    """
    # Use the general formula: F = (Tr√(√ρ σ √ρ))²
    sqrt_rho = _matrix_sqrt(rho)
    product = sqrt_rho @ sigma @ sqrt_rho
    sqrt_product = _matrix_sqrt(product)
    return float(np.real(np.trace(sqrt_product)) ** 2)


def _matrix_sqrt(A: np.ndarray) -> np.ndarray:
    """Compute matrix square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0)  # Clamp numerical negatives
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T


def apply_channel(rho: np.ndarray, kraus_ops: List[np.ndarray]) -> np.ndarray:
    """Apply a quantum channel defined by Kraus operators.

    Why: Any completely positive trace-preserving (CPTP) map can be written
    as ε(ρ) = Σ_k E_k ρ E_k†.  The Kraus operators {E_k} satisfy
    Σ_k E_k† E_k = I, ensuring trace preservation.
    """
    result = np.zeros_like(rho)
    for E in kraus_ops:
        result += E @ rho @ E.conj().T
    return result


def verify_cptp(kraus_ops: List[np.ndarray]) -> bool:
    """Verify that Kraus operators satisfy the CPTP condition Σ E_k† E_k = I."""
    d = kraus_ops[0].shape[0]
    total = np.zeros((d, d), dtype=complex)
    for E in kraus_ops:
        total += E.conj().T @ E
    return np.allclose(total, np.eye(d))


# ---------------------------------------------------------------------------
# Quantum Channels
# ---------------------------------------------------------------------------

def depolarizing_channel(p: float) -> List[np.ndarray]:
    """Kraus operators for the depolarizing channel.

    ε(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)

    Why: The depolarizing channel is the most symmetric noise model — it
    replaces the state with the maximally mixed state I/2 with probability
    4p/3.  It serves as the standard benchmark for error correction because
    it treats all error types equally.
    """
    E0 = np.sqrt(1 - p) * I
    E1 = np.sqrt(p / 3) * X
    E2 = np.sqrt(p / 3) * Y
    E3 = np.sqrt(p / 3) * Z
    return [E0, E1, E2, E3]


def amplitude_damping_channel(gamma: float) -> List[np.ndarray]:
    """Kraus operators for the amplitude damping channel.

    Why: Amplitude damping models energy relaxation (T1 decay) — the qubit
    decays from |1⟩ to |0⟩ with probability γ.  This is the dominant
    decoherence mechanism in superconducting qubits and trapped ions.
    The channel is NOT unital: it maps I/2 → biased state (not I/2).
    """
    E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    E1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [E0, E1]


def phase_damping_channel(lam: float) -> List[np.ndarray]:
    """Kraus operators for the phase damping channel.

    Why: Phase damping (T2 dephasing) destroys off-diagonal coherence
    without changing populations.  It models random phase kicks from the
    environment.  Combined with amplitude damping, it gives the full
    T1/T2 relaxation picture.
    """
    E0 = np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex)
    E1 = np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex)
    return [E0, E1]


def bit_flip_channel(p: float) -> List[np.ndarray]:
    """Kraus operators for the bit-flip channel.

    Why: The simplest non-trivial noise model — flips |0⟩↔|1⟩ with
    probability p.  Easy to correct with the 3-qubit repetition code.
    """
    E0 = np.sqrt(1 - p) * I
    E1 = np.sqrt(p) * X
    return [E0, E1]


def phase_flip_channel(p: float) -> List[np.ndarray]:
    """Kraus operators for the phase-flip channel.

    Why: Applies Z with probability p, flipping the relative phase.
    Correctable by the 3-qubit phase-flip code (repetition code in the
    Hadamard basis).
    """
    E0 = np.sqrt(1 - p) * I
    E1 = np.sqrt(p) * Z
    return [E0, E1]


# ---------------------------------------------------------------------------
# Process Tomography
# ---------------------------------------------------------------------------

def process_tomography(channel_fn, d: int = 2) -> np.ndarray:
    """Reconstruct the χ-matrix (process matrix) of a single-qubit channel.

    Why: The χ-matrix fully characterizes a quantum channel in the Pauli
    basis: ε(ρ) = Σ_{mn} χ_{mn} P_m ρ P_n†.  Process tomography determines
    χ experimentally by preparing a complete set of input states and
    measuring the output.  Here we simulate the procedure using the
    Choi-Jamiolkowski isomorphism.
    """
    # Why: The Choi matrix Λ is obtained by applying the channel to one half
    # of a maximally entangled state.  From Λ we can extract χ.
    # Maximally entangled state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    bell = np.zeros(d * d, dtype=complex)
    for i in range(d):
        e_i = np.zeros(d, dtype=complex)
        e_i[i] = 1.0
        bell += np.kron(e_i, e_i)
    bell /= np.sqrt(d)

    rho_bell = np.outer(bell, bell.conj())

    # Apply channel to the second subsystem
    choi = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            # Extract block [i,j] of the input density matrix
            e_i = np.zeros(d, dtype=complex)
            e_j = np.zeros(d, dtype=complex)
            e_i[i] = 1.0
            e_j[j] = 1.0
            block_in = np.outer(e_i, e_j)
            block_out = channel_fn(block_in)
            for k in range(d):
                for l in range(d):
                    choi[i * d + k, j * d + l] = block_out[k, l]

    # Convert Choi matrix to χ-matrix in Pauli basis
    chi = np.zeros((4, 4), dtype=complex)
    for m in range(4):
        for n in range(4):
            chi[m, n] = 0
            for i in range(d):
                for j in range(d):
                    for k in range(d):
                        for l in range(d):
                            chi[m, n] += (PAULIS[m][i, k] *
                                          choi[i * d + k, j * d + l] *
                                          PAULIS[n][l, j].conj())
            chi[m, n] /= d

    return chi


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_kraus_operators():
    """Display and verify Kraus operators for common channels."""
    print("=" * 60)
    print("DEMO 1: Kraus Operator Formalism")
    print("=" * 60)

    channels = [
        ("Depolarizing (p=0.1)", depolarizing_channel(0.1)),
        ("Amplitude Damping (γ=0.3)", amplitude_damping_channel(0.3)),
        ("Phase Damping (λ=0.2)", phase_damping_channel(0.2)),
        ("Bit Flip (p=0.05)", bit_flip_channel(0.05)),
        ("Phase Flip (p=0.1)", phase_flip_channel(0.1)),
    ]

    for name, kraus_ops in channels:
        cptp = verify_cptp(kraus_ops)
        print(f"\n  {name}:")
        print(f"    Number of Kraus operators: {len(kraus_ops)}")
        print(f"    CPTP condition Σ E_k† E_k = I: {cptp}")
        for i, E in enumerate(kraus_ops):
            print(f"    E_{i}:")
            for row in E:
                entries = [f"{v.real:+.4f}{v.imag:+.4f}j" if abs(v.imag) > 1e-10
                           else f"{v.real:+.4f}       " for v in row]
                print(f"      [{', '.join(entries)}]")


def demo_depolarizing():
    """Apply depolarizing channel to various states."""
    print("\n" + "=" * 60)
    print("DEMO 2: Depolarizing Channel")
    print("=" * 60)

    # Why: The depolarizing channel contracts the Bloch sphere uniformly.
    # A state at distance r from the center moves to (1-4p/3)·r.
    states = [
        ("|0⟩", KET_0),
        ("|1⟩", KET_1),
        ("|+⟩", KET_PLUS),
    ]

    print(f"\n  {'State':<8} {'p':>6} {'Tr(ρ²)':>10} {'⟨Z⟩':>8} {'⟨X⟩':>8} {'Fidelity':>10}")
    print(f"  {'─' * 55}")

    for label, psi in states:
        rho_ideal = density_matrix(psi)
        for p in [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]:
            kraus = depolarizing_channel(p)
            rho_out = apply_channel(rho_ideal, kraus)
            purity = np.real(np.trace(rho_out @ rho_out))
            exp_z = np.real(np.trace(Z @ rho_out))
            exp_x = np.real(np.trace(X @ rho_out))
            fid = np.real(psi.conj() @ rho_out @ psi)
            print(f"  {label:<8} {p:6.2f} {purity:10.4f} {exp_z:8.4f} "
                  f"{exp_x:8.4f} {fid:10.4f}")
        print()


def demo_amplitude_damping():
    """Show amplitude damping (T1 relaxation) dynamics."""
    print("\n" + "=" * 60)
    print("DEMO 3: Amplitude Damping Channel (T1 Decay)")
    print("=" * 60)

    # Why: Amplitude damping drives |1⟩ → |0⟩ over time.  The probability
    # of being in |1⟩ decays as (1-γ)^n after n applications.  This models
    # the spontaneous emission in two-level systems.

    print(f"\n  Starting from |1⟩:")
    print(f"  {'γ':>8} {'P(|0⟩)':>10} {'P(|1⟩)':>10} {'Purity':>10} {'Coherence |ρ₀₁|':>18}")
    print(f"  {'─' * 60}")

    # Start from |1⟩
    rho = density_matrix(KET_1)
    for gamma in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
        kraus = amplitude_damping_channel(gamma)
        rho_out = apply_channel(density_matrix(KET_1), kraus)
        p0 = np.real(rho_out[0, 0])
        p1 = np.real(rho_out[1, 1])
        purity = np.real(np.trace(rho_out @ rho_out))
        coherence = np.abs(rho_out[0, 1])
        print(f"  {gamma:8.2f} {p0:10.4f} {p1:10.4f} {purity:10.4f} {coherence:18.4f}")

    # Why: At γ=1 the qubit has fully decayed to |0⟩.  The purity dips
    # (mixed state) then returns to 1 as it approaches pure |0⟩.

    print(f"\n  Starting from |+⟩:")
    print(f"  {'γ':>8} {'P(|0⟩)':>10} {'P(|1⟩)':>10} {'Coherence |ρ₀₁|':>18}")
    print(f"  {'─' * 42}")

    for gamma in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
        kraus = amplitude_damping_channel(gamma)
        rho_out = apply_channel(density_matrix(KET_PLUS), kraus)
        p0 = np.real(rho_out[0, 0])
        p1 = np.real(rho_out[1, 1])
        coherence = np.abs(rho_out[0, 1])
        print(f"  {gamma:8.2f} {p0:10.4f} {p1:10.4f} {coherence:18.4f}")


def demo_phase_damping():
    """Show phase damping (T2 dephasing) dynamics."""
    print("\n" + "=" * 60)
    print("DEMO 4: Phase Damping Channel (T2 Dephasing)")
    print("=" * 60)

    # Why: Phase damping destroys off-diagonal elements (coherence) without
    # changing diagonal elements (populations).  It models random phase
    # fluctuations from the environment.

    print(f"\n  Starting from |+⟩ = (|0⟩ + |1⟩)/√2:")
    print(f"  {'λ':>8} {'P(|0⟩)':>10} {'P(|1⟩)':>10} {'|ρ₀₁|':>10} {'⟨X⟩':>8} {'Purity':>10}")
    print(f"  {'─' * 60}")

    for lam in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        kraus = phase_damping_channel(lam)
        rho_out = apply_channel(density_matrix(KET_PLUS), kraus)
        p0 = np.real(rho_out[0, 0])
        p1 = np.real(rho_out[1, 1])
        coherence = np.abs(rho_out[0, 1])
        exp_x = np.real(np.trace(X @ rho_out))
        purity = np.real(np.trace(rho_out @ rho_out))
        print(f"  {lam:8.2f} {p0:10.4f} {p1:10.4f} {coherence:10.4f} "
              f"{exp_x:8.4f} {purity:10.4f}")

    # Why: At λ=1, all coherence is lost → ρ = I/2 (maximally mixed).
    # The populations remain unchanged at 0.5 each.


def demo_process_tomography():
    """Reconstruct the χ-matrix for quantum channels."""
    print("\n" + "=" * 60)
    print("DEMO 5: Process Tomography (χ-Matrix)")
    print("=" * 60)

    # Why: The χ-matrix in the Pauli basis reveals which error types dominate.
    # For a depolarizing channel, χ is diagonal with equal X, Y, Z weights.
    # For amplitude damping, the structure is more complex.

    channels = [
        ("Identity", lambda rho: rho),
        ("Depolarizing (p=0.1)", lambda rho: apply_channel(
            rho, depolarizing_channel(0.1))),
        ("Amplitude Damping (γ=0.3)", lambda rho: apply_channel(
            rho, amplitude_damping_channel(0.3))),
        ("Phase Damping (λ=0.2)", lambda rho: apply_channel(
            rho, phase_damping_channel(0.2))),
    ]

    for name, channel_fn in channels:
        chi = process_tomography(channel_fn)
        print(f"\n  {name}:")
        print(f"    χ-matrix (Pauli basis: I, X, Y, Z):")
        header = "        " + "".join(f"{'  ' + PAULI_LABELS[j]:>10}" for j in range(4))
        print(header)
        for i in range(4):
            entries = []
            for j in range(4):
                val = chi[i, j]
                if abs(val.imag) < 1e-6:
                    entries.append(f"{val.real:10.4f}")
                else:
                    entries.append(f"{val.real:+.3f}{val.imag:+.3f}j")
            print(f"    {PAULI_LABELS[i]:>4} {''.join(entries)}")


def demo_fidelity_decay():
    """Show how fidelity decays with repeated noise applications."""
    print("\n" + "=" * 60)
    print("DEMO 6: Fidelity Decay Under Repeated Noise")
    print("=" * 60)

    # Why: In a real quantum computation, noise accumulates with each gate.
    # Understanding how fidelity scales with circuit depth is crucial for
    # determining whether error correction or mitigation is needed.

    n_steps = 10
    initial = density_matrix(KET_PLUS)

    channels = [
        ("Depolarizing (p=0.05)", depolarizing_channel(0.05)),
        ("Amplitude Damping (γ=0.05)", amplitude_damping_channel(0.05)),
        ("Phase Damping (λ=0.05)", phase_damping_channel(0.05)),
    ]

    print(f"\n  Starting from |+⟩, applying channel n times:")
    print(f"\n  {'n':>4}", end="")
    for name, _ in channels:
        short = name.split("(")[0].strip()
        print(f"  {short:>16}", end="")
    print()
    print(f"  {'─' * (4 + 18 * len(channels))}")

    for n in range(n_steps + 1):
        print(f"  {n:4d}", end="")
        for _, kraus_ops in channels:
            rho = initial.copy()
            for _ in range(n):
                rho = apply_channel(rho, kraus_ops)
            fid = np.real(KET_PLUS.conj() @ rho @ KET_PLUS)
            print(f"  {fid:16.6f}", end="")
        print()

    # Why: Depolarizing fidelity decays as (1-4p/3)^n, phase damping as
    # (1-λ)^{n/2} for coherence, while amplitude damping has a more complex
    # decay that depends on the initial state.


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Quantum Computing — 14: Noise and Quantum Channels    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    np.random.seed(2026)

    demo_kraus_operators()
    demo_depolarizing()
    demo_amplitude_damping()
    demo_phase_damping()
    demo_process_tomography()
    demo_fidelity_decay()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
