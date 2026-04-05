"""
10_teleportation.py — Quantum Teleportation and Superdense Coding

Demonstrates:
  - Quantum teleportation protocol: transfer a qubit using entanglement + 2 classical bits
  - Bell measurement implementation
  - Verification that teleported state matches the original
  - Superdense coding: send 2 classical bits using 1 qubit + shared entanglement
  - The deep connection between teleportation and superdense coding (they are duals)

All computations use pure NumPy.
"""

import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# Gates and basis states
# ---------------------------------------------------------------------------
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

KET_0 = np.array([1, 0], dtype=complex)
KET_1 = np.array([0, 1], dtype=complex)

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)


def tensor(*matrices: np.ndarray) -> np.ndarray:
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def apply_single(state: np.ndarray, gate: np.ndarray,
                  target: int, n_qubits: int) -> np.ndarray:
    ops = [I] * n_qubits
    ops[target] = gate
    full = ops[0]
    for op in ops[1:]:
        full = np.kron(full, op)
    return full @ state


def apply_cnot(state: np.ndarray, control: int, target: int,
               n_qubits: int) -> np.ndarray:
    proj_0 = np.array([[1, 0], [0, 0]], dtype=complex)
    proj_1 = np.array([[0, 0], [0, 1]], dtype=complex)
    ops_0 = [I] * n_qubits
    ops_0[control] = proj_0
    term_0 = ops_0[0]
    for op in ops_0[1:]:
        term_0 = np.kron(term_0, op)
    ops_1 = [I] * n_qubits
    ops_1[control] = proj_1
    ops_1[target] = X
    term_1 = ops_1[0]
    for op in ops_1[1:]:
        term_1 = np.kron(term_1, op)
    return (term_0 + term_1) @ state


def format_state(state: np.ndarray, n_qubits: int) -> str:
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


def make_bell_pair() -> np.ndarray:
    """Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2."""
    state = tensor(KET_0, KET_0)
    state = tensor(H, I) @ state
    state = CNOT @ state
    return state


# ---------------------------------------------------------------------------
# Quantum Teleportation
# ---------------------------------------------------------------------------

def teleport(psi: np.ndarray, verbose: bool = True) -> np.ndarray:
    """Teleport qubit state |ψ⟩ from Alice to Bob.

    Protocol:
    1. Alice and Bob share a Bell pair (qubits 1, 2)
    2. Alice has |ψ⟩ on qubit 0
    3. Alice performs Bell measurement on qubits 0, 1
    4. Alice sends 2 classical bits to Bob
    5. Bob applies corrections based on those bits

    Why: Teleportation demonstrates that quantum information can be transmitted
    using shared entanglement + classical communication.  The qubit is destroyed
    at Alice's end and recreated at Bob's end — no cloning occurs.
    This is NOT faster-than-light communication because Alice must send
    2 classical bits to Bob for him to reconstruct the state.
    """
    n_qubits = 3

    if verbose:
        print(f"    Input |ψ⟩ = {format_state(psi, 1)}")

    # Step 1: Full initial state: |ψ⟩_A ⊗ |Φ+⟩_{A'B}
    # Why: We build the 3-qubit state as |ψ⟩ ⊗ (|00⟩+|11⟩)/√2.
    # Qubit 0 = Alice's message, Qubit 1 = Alice's Bell pair half,
    # Qubit 2 = Bob's Bell pair half.
    bell = make_bell_pair()
    state = tensor(psi, bell)

    if verbose:
        print(f"    Initial 3-qubit state: {format_state(state, 3)}")

    # Step 2: Alice's Bell measurement (CNOT then H on qubit 0)
    # Why: The Bell measurement projects qubits 0,1 into one of the four
    # Bell states.  We implement it as CNOT(0→1) followed by H(0).
    state = apply_cnot(state, 0, 1, n_qubits)
    state = apply_single(state, H, 0, n_qubits)

    if verbose:
        print(f"    After Bell measurement circuit: {format_state(state, 3)}")

    # Step 3: Measure qubits 0 and 1
    # Why: Instead of probabilistic measurement, we'll show all four possible
    # outcomes and demonstrate that correction always works.
    # For the simulation, we pick a random outcome.
    probs = np.abs(state) ** 2

    # Probabilities for each measurement outcome (m0, m1)
    p = {}
    for m0 in range(2):
        for m1 in range(2):
            p_val = sum(probs[m0 * 4 + m1 * 2 + b] for b in range(2))
            p[(m0, m1)] = p_val

    # Sample measurement outcome
    outcomes = list(p.keys())
    probs_list = [p[o] for o in outcomes]
    idx = np.random.choice(len(outcomes), p=probs_list)
    m0, m1 = outcomes[idx]

    if verbose:
        print(f"    Alice's measurement: ({m0}, {m1})")
        print(f"    Outcome probabilities: {dict((k, f'{v:.4f}') for k, v in p.items())}")

    # Collapse the state
    # Why: After measuring qubits 0,1 as |m0 m1⟩, Bob's qubit (qubit 2)
    # is in a state that differs from |ψ⟩ by a known Pauli correction.
    bob_state = np.zeros(2, dtype=complex)
    for b in range(2):
        idx_3q = m0 * 4 + m1 * 2 + b
        bob_state[b] = state[idx_3q]
    bob_state = bob_state / np.linalg.norm(bob_state)

    if verbose:
        print(f"    Bob's state before correction: {format_state(bob_state, 1)}")

    # Step 4: Bob's corrections
    # Why: The correction is determined by Alice's measurement outcome:
    #   (0,0) → I (no correction)
    #   (0,1) → X
    #   (1,0) → Z
    #   (1,1) → ZX (= iY up to global phase)
    if m1 == 1:
        bob_state = X @ bob_state
    if m0 == 1:
        bob_state = Z @ bob_state

    if verbose:
        correction = {(0, 0): "I", (0, 1): "X", (1, 0): "Z", (1, 1): "ZX"}
        print(f"    Correction applied: {correction[(m0, m1)]}")
        print(f"    Bob's state after correction: {format_state(bob_state, 1)}")

    return bob_state


def teleport_all_outcomes(psi: np.ndarray) -> None:
    """Show all four possible teleportation outcomes and corrections."""
    n_qubits = 3
    bell = make_bell_pair()
    state = tensor(psi, bell)

    # Bell measurement circuit
    state = apply_cnot(state, 0, 1, n_qubits)
    state = apply_single(state, H, 0, n_qubits)

    corrections = {(0, 0): I, (0, 1): X, (1, 0): Z, (1, 1): Z @ X}
    names = {(0, 0): "I", (0, 1): "X", (1, 0): "Z", (1, 1): "ZX"}

    print(f"\n    All four possible outcomes:")
    print(f"    {'Outcome':<12} {'Bob before':>20} {'Correction':<12} {'Bob after':>20} {'Matches |ψ⟩?':>15}")
    print(f"    {'─' * 80}")

    for m0 in range(2):
        for m1 in range(2):
            bob = np.zeros(2, dtype=complex)
            for b in range(2):
                idx = m0 * 4 + m1 * 2 + b
                bob[b] = state[idx]
            bob = bob / np.linalg.norm(bob)

            bob_before = format_state(bob, 1)
            corrected = corrections[(m0, m1)] @ bob
            bob_after = format_state(corrected, 1)

            # Check match (up to global phase)
            overlap = np.abs(np.vdot(psi, corrected))
            match = overlap > 0.999

            print(f"    ({m0},{m1})       {bob_before:>20} {names[(m0,m1)]:<12} {bob_after:>20} {str(match):>15}")


# ---------------------------------------------------------------------------
# Superdense Coding
# ---------------------------------------------------------------------------

def superdense_coding(bits: Tuple[int, int], verbose: bool = True) -> Tuple[int, int]:
    """Superdense coding: send 2 classical bits using 1 qubit + entanglement.

    Protocol:
    1. Alice and Bob share a Bell pair
    2. Alice encodes 2 classical bits by applying a gate to her half
    3. Alice sends her qubit to Bob
    4. Bob performs Bell measurement to recover both bits

    Why: Superdense coding is the dual of teleportation.
    Teleportation: 1 qubit + 2 classical bits → transmit 1 qubit
    Superdense:    1 qubit + entanglement    → transmit 2 classical bits
    Both use exactly 1 ebit (entangled bit pair) as a resource.
    """
    if verbose:
        print(f"    Bits to send: ({bits[0]}, {bits[1]})")

    # Step 1: Shared Bell pair |Φ+⟩
    bell = make_bell_pair()
    if verbose:
        print(f"    Shared Bell pair: {format_state(bell, 2)}")

    # Step 2: Alice's encoding
    # Why: Alice applies one of {I, X, Z, ZX} to her qubit, transforming the
    # Bell pair into one of the four orthogonal Bell states.  Since Bell states
    # are orthogonal, Bob can perfectly distinguish them → 2 bits of information.
    encoding = {
        (0, 0): I,     # |Φ+⟩ → |Φ+⟩
        (0, 1): X,     # |Φ+⟩ → |Ψ+⟩
        (1, 0): Z,     # |Φ+⟩ → |Φ-⟩
        (1, 1): Z @ X, # |Φ+⟩ → |Ψ-⟩  (iY up to phase)
    }
    encoding_names = {(0, 0): "I", (0, 1): "X", (1, 0): "Z", (1, 1): "ZX"}

    gate = encoding[bits]
    encoded = tensor(gate, I) @ bell

    if verbose:
        print(f"    Alice applies {encoding_names[bits]} to her qubit")
        print(f"    Encoded state: {format_state(encoded, 2)}")

    # Step 3: Bob's Bell measurement
    # Why: Bob applies CNOT then H (inverse of Bell state preparation) to
    # map each Bell state to a unique computational basis state:
    #   |Φ+⟩ → |00⟩, |Ψ+⟩ → |01⟩, |Φ-⟩ → |10⟩, |Ψ-⟩ → |11⟩
    decoded = CNOT @ encoded
    decoded = tensor(H, I) @ decoded

    if verbose:
        print(f"    After Bob's decoding: {format_state(decoded, 2)}")

    # Measure
    probs = np.abs(decoded) ** 2
    outcome = np.argmax(probs)
    b0 = outcome >> 1
    b1 = outcome & 1

    if verbose:
        print(f"    Bob measures: ({b0}, {b1})")
        print(f"    Match? {(b0, b1) == bits}")

    return (b0, b1)


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_teleportation_basic():
    """Basic teleportation of several states."""
    print("=" * 60)
    print("DEMO 1: Quantum Teleportation — Basic Examples")
    print("=" * 60)

    test_states = [
        ("|0⟩", KET_0),
        ("|1⟩", KET_1),
        ("|+⟩", (KET_0 + KET_1) / np.sqrt(2)),
        ("|−⟩", (KET_0 - KET_1) / np.sqrt(2)),
        ("arbitrary", np.array([np.cos(0.3), np.exp(1j * 0.7) * np.sin(0.3)], dtype=complex)),
    ]

    for name, psi in test_states:
        print(f"\n  --- Teleporting {name} ---")
        result = teleport(psi, verbose=True)
        overlap = np.abs(np.vdot(psi, result))
        print(f"    Fidelity: |⟨ψ|result⟩| = {overlap:.6f}")


def demo_teleportation_all_outcomes():
    """Show all four measurement outcomes for teleportation."""
    print("\n" + "=" * 60)
    print("DEMO 2: All Four Teleportation Outcomes")
    print("=" * 60)

    psi = np.array([np.cos(np.pi / 5), np.exp(1j * np.pi / 3) * np.sin(np.pi / 5)], dtype=complex)
    print(f"\n  State to teleport: {format_state(psi, 1)}")

    teleport_all_outcomes(psi)

    # Why: Regardless of which outcome Alice gets (each with probability 1/4),
    # Bob can always recover |ψ⟩ with the appropriate correction.  This is
    # a deterministic protocol — it succeeds with 100% probability.
    print(f"\n  Every outcome leads to perfect recovery after correction!")


def demo_superdense_coding():
    """Demonstrate superdense coding for all 4 possible bit pairs."""
    print("\n" + "=" * 60)
    print("DEMO 3: Superdense Coding")
    print("=" * 60)

    print(f"\n  Sending 2 classical bits using 1 qubit + entanglement:\n")

    all_correct = True
    for b0 in range(2):
        for b1 in range(2):
            print(f"  --- Sending ({b0}, {b1}) ---")
            received = superdense_coding((b0, b1), verbose=True)
            if received != (b0, b1):
                all_correct = False
            print()

    print(f"  All transmissions correct? {all_correct}")


def demo_teleportation_no_cloning():
    """Verify that teleportation respects the no-cloning theorem."""
    print("\n" + "=" * 60)
    print("DEMO 4: No-Cloning — Teleportation Destroys the Original")
    print("=" * 60)

    # Why: In teleportation, Alice's measurement destroys her copy of |ψ⟩.
    # After measurement, qubits 0 and 1 are in a definite state |m0 m1⟩ —
    # all information about |ψ⟩ is gone from Alice's side.
    # This is not a bug, it's a feature: no-cloning is preserved!

    psi = (KET_0 + np.exp(1j * np.pi / 4) * KET_1) / np.sqrt(2)
    print(f"\n  Original |ψ⟩ = {format_state(psi, 1)}")

    n_qubits = 3
    bell = make_bell_pair()
    state = tensor(psi, bell)

    # Bell measurement
    state = apply_cnot(state, 0, 1, n_qubits)
    state = apply_single(state, H, 0, n_qubits)

    # Measure qubits 0, 1 (collapse to random outcome)
    probs = np.abs(state) ** 2
    full_probs = np.zeros(4)
    for m0 in range(2):
        for m1 in range(2):
            for b in range(2):
                full_probs[m0 * 2 + m1] += probs[m0 * 4 + m1 * 2 + b]

    m_idx = np.random.choice(4, p=full_probs)
    m0, m1 = m_idx // 2, m_idx % 2

    # After measurement, Alice's qubits are in state |m0 m1⟩
    print(f"  After measurement: Alice's qubits = |{m0}{m1}⟩ (definite, no info about |ψ⟩)")
    print(f"  Bob's qubit: still needs correction, but CONTAINS the information")
    print(f"\n  → The quantum state was MOVED from Alice to Bob, not copied.")
    print(f"  → This is consistent with the no-cloning theorem.")


def demo_resource_comparison():
    """Compare teleportation and superdense coding resource usage."""
    print("\n" + "=" * 60)
    print("DEMO 5: Duality — Teleportation vs Superdense Coding")
    print("=" * 60)

    # Why: Teleportation and superdense coding are dual protocols — they trade
    # quantum and classical resources in opposite ways, with entanglement as
    # the bridge.  This duality reveals a deep connection in quantum information.
    print(f"""
  Resource comparison:

  ┌─────────────────────┬────────────────────┬───────────────────────┐
  │                     │  Teleportation     │  Superdense Coding    │
  ├─────────────────────┼────────────────────┼───────────────────────┤
  │ Purpose             │ Send 1 qubit       │ Send 2 classical bits │
  │ Entanglement used   │ 1 Bell pair        │ 1 Bell pair           │
  │ Qubits sent         │ 0                  │ 1                     │
  │ Classical bits sent │ 2                  │ 0                     │
  │ Who acts first      │ Alice (measures)   │ Alice (encodes)       │
  │ Who acts last       │ Bob (corrects)     │ Bob (measures)        │
  └─────────────────────┴────────────────────┴───────────────────────┘

  Teleportation:  2 cbits + 1 ebit  →  1 qubit transmitted
  Superdense:     1 qubit + 1 ebit  →  2 cbits transmitted

  Key insight: Entanglement is a RESOURCE that can convert between
  quantum and classical communication.  Neither protocol violates
  special relativity — both require some physical transmission.
    """)


def demo_teleportation_statistics():
    """Run many teleportation attempts and verify high fidelity."""
    print("=" * 60)
    print("DEMO 6: Teleportation Fidelity Statistics")
    print("=" * 60)

    n_trials = 100
    fidelities = []

    # Random states to teleport
    for _ in range(n_trials):
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        psi = np.array([np.cos(theta / 2),
                        np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)

        result = teleport(psi, verbose=False)
        fidelity = np.abs(np.vdot(psi, result)) ** 2
        fidelities.append(fidelity)

    fidelities = np.array(fidelities)
    print(f"\n  Teleported {n_trials} random states:")
    print(f"  Mean fidelity:    {fidelities.mean():.6f}")
    print(f"  Min fidelity:     {fidelities.min():.6f}")
    print(f"  Std deviation:    {fidelities.std():.6f}")
    print(f"  All perfect (>0.999)? {np.all(fidelities > 0.999)}")

    # Why: In noiseless simulation, teleportation always achieves perfect
    # fidelity (1.0).  On real hardware, noise and gate errors reduce fidelity —
    # which is why quantum error correction (lesson 09) is essential.


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Quantum Computing — 10: Teleportation & Superdense    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    np.random.seed(2026)

    demo_teleportation_basic()
    demo_teleportation_all_outcomes()
    demo_superdense_coding()
    demo_teleportation_no_cloning()
    demo_resource_comparison()
    demo_teleportation_statistics()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
