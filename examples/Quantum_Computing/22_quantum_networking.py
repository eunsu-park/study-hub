"""
22_quantum_networking.py — Quantum Networking and QKD

Demonstrates:
  - BB84 quantum key distribution protocol simulation
  - E91 (Ekert) entanglement-based QKD protocol
  - Eavesdropper detection via error rate analysis
  - Entanglement swapping for quantum repeaters
  - Entanglement distillation (purification)
  - Quantum repeater chain performance analysis

All computations use pure NumPy.
"""

import numpy as np
from typing import Tuple, List

# ---------------------------------------------------------------------------
# Quantum states and operations
# ---------------------------------------------------------------------------

KET_0 = np.array([1, 0], dtype=complex)
KET_1 = np.array([0, 1], dtype=complex)
KET_PLUS = (KET_0 + KET_1) / np.sqrt(2)
KET_MINUS = (KET_0 - KET_1) / np.sqrt(2)

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H_GATE = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)


def bell_state_phi_plus() -> np.ndarray:
    """Create |Φ+⟩ = (|00⟩ + |11⟩)/√2."""
    return np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)


def bell_state_psi_plus() -> np.ndarray:
    """Create |Ψ+⟩ = (|01⟩ + |10⟩)/√2."""
    return np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)


def measure_qubit(state: np.ndarray, basis: str) -> Tuple[int, np.ndarray]:
    """Measure a single-qubit state in the specified basis.

    Why: In QKD, Alice prepares qubits in either the Z-basis {|0⟩,|1⟩}
    or X-basis {|+⟩,|−⟩}, and Bob measures in a randomly chosen basis.
    Matching bases yield correlated bits; mismatched bases give random results.
    """
    if basis == 'Z':
        prob_0 = float(np.abs(state[0]) ** 2)
        outcome = 0 if np.random.random() < prob_0 else 1
        post_state = KET_0 if outcome == 0 else KET_1
    elif basis == 'X':
        # Transform to X basis then measure
        state_x = H_GATE @ state
        prob_0 = float(np.abs(state_x[0]) ** 2)
        outcome = 0 if np.random.random() < prob_0 else 1
        post_state = KET_PLUS if outcome == 0 else KET_MINUS
    else:
        raise ValueError(f"Unknown basis: {basis}")

    return outcome, post_state


# ---------------------------------------------------------------------------
# BB84 Protocol
# ---------------------------------------------------------------------------

def bb84_protocol(n_bits: int, eve_present: bool = False,
                  eve_strategy: str = "intercept_resend"
                  ) -> dict:
    """Simulate the BB84 QKD protocol.

    Why: BB84 (Bennett & Brassard, 1984) is the first and most widely
    deployed QKD protocol.  Alice sends random bits encoded in random
    bases.  Bob measures in random bases.  They publicly compare bases
    and keep bits where bases match.  An eavesdropper (Eve) introduces
    detectable errors because measuring a quantum state disturbs it.
    """
    # Alice: choose random bits and bases
    alice_bits = np.random.randint(0, 2, n_bits)
    alice_bases = np.random.randint(0, 2, n_bits)  # 0=Z, 1=X

    # Alice prepares states
    alice_states = []
    for bit, basis in zip(alice_bits, alice_bases):
        if basis == 0:  # Z basis
            state = KET_0 if bit == 0 else KET_1
        else:  # X basis
            state = KET_PLUS if bit == 0 else KET_MINUS
        alice_states.append(state.copy())

    # Eve's interception (optional)
    eve_bits = np.zeros(n_bits, dtype=int)
    if eve_present:
        for i in range(n_bits):
            eve_basis = np.random.randint(0, 2)
            basis_str = 'Z' if eve_basis == 0 else 'X'
            eve_bits[i], post_state = measure_qubit(alice_states[i], basis_str)
            # Eve resends the post-measurement state
            alice_states[i] = post_state

    # Bob: choose random bases and measure
    bob_bases = np.random.randint(0, 2, n_bits)
    bob_bits = np.zeros(n_bits, dtype=int)
    for i in range(n_bits):
        basis_str = 'Z' if bob_bases[i] == 0 else 'X'
        bob_bits[i], _ = measure_qubit(alice_states[i], basis_str)

    # Sifting: keep bits where Alice and Bob used the same basis
    matching_bases = alice_bases == bob_bases
    sifted_alice = alice_bits[matching_bases]
    sifted_bob = bob_bits[matching_bases]

    # Error rate estimation (use a subset)
    n_sifted = len(sifted_alice)
    n_check = min(n_sifted // 4, n_sifted)
    if n_check > 0:
        check_alice = sifted_alice[:n_check]
        check_bob = sifted_bob[:n_check]
        error_rate = float(np.sum(check_alice != check_bob)) / n_check
    else:
        error_rate = 0.0

    # Final key (remaining sifted bits)
    key_alice = sifted_alice[n_check:]
    key_bob = sifted_bob[n_check:]
    key_errors = np.sum(key_alice != key_bob) if len(key_alice) > 0 else 0

    return {
        "n_sent": n_bits,
        "n_sifted": n_sifted,
        "n_check": n_check,
        "error_rate": error_rate,
        "key_length": len(key_alice),
        "key_errors": int(key_errors),
        "eve_present": eve_present,
    }


# ---------------------------------------------------------------------------
# E91 Protocol
# ---------------------------------------------------------------------------

def e91_protocol(n_pairs: int, eve_present: bool = False) -> dict:
    """Simulate the E91 (Ekert) QKD protocol.

    Why: E91 uses entangled Bell pairs shared between Alice and Bob.
    Security is guaranteed by the violation of Bell's inequality (CHSH).
    If Eve intercepts, entanglement is broken and the CHSH value drops
    below 2√2, revealing her presence.
    """
    # Measurement angles (Alice: 0, π/8, π/4; Bob: π/8, π/4, 3π/8)
    alice_angles = [0, np.pi / 8, np.pi / 4]
    bob_angles = [np.pi / 8, np.pi / 4, 3 * np.pi / 8]

    alice_results = np.zeros(n_pairs, dtype=int)
    bob_results = np.zeros(n_pairs, dtype=int)
    alice_choices = np.random.randint(0, 3, n_pairs)
    bob_choices = np.random.randint(0, 3, n_pairs)

    for i in range(n_pairs):
        theta_a = alice_angles[alice_choices[i]]
        theta_b = bob_angles[bob_choices[i]]

        if not eve_present:
            # Perfect Bell state: correlation = -cos(θ_a - θ_b)
            angle_diff = theta_a - theta_b
            p_same = (1 - np.cos(2 * angle_diff)) / 2  # P(different outcomes)
            if np.random.random() < 0.5:
                alice_results[i] = 0
                bob_results[i] = 0 if np.random.random() > p_same else 1
            else:
                alice_results[i] = 1
                bob_results[i] = 1 if np.random.random() > p_same else 0
        else:
            # Eve measures, breaking entanglement → classical correlation
            alice_results[i] = np.random.randint(0, 2)
            bob_results[i] = np.random.randint(0, 2)

    # CHSH test (using specific angle pairs)
    # S = E(a1,b1) - E(a1,b3) + E(a3,b1) + E(a3,b3)
    def correlator(a_choice, b_choice):
        mask = (alice_choices == a_choice) & (bob_choices == b_choice)
        if np.sum(mask) == 0:
            return 0.0
        a_vals = 2 * alice_results[mask] - 1  # Map to ±1
        b_vals = 2 * bob_results[mask] - 1
        return float(np.mean(a_vals * b_vals))

    S = (correlator(0, 0) - correlator(0, 2) +
         correlator(2, 0) + correlator(2, 2))

    # Sifted key: Alice angle 1 and Bob angle 1 (same angle π/8)
    key_mask = (alice_choices == 1) & (bob_choices == 0)
    key_alice = alice_results[key_mask]
    key_bob = bob_results[key_mask]

    error_rate = 0.0
    if len(key_alice) > 0:
        error_rate = float(np.sum(key_alice != key_bob)) / len(key_alice)

    return {
        "n_pairs": n_pairs,
        "chsh_value": S,
        "chsh_violation": abs(S) > 2.0,
        "key_length": len(key_alice),
        "error_rate": error_rate,
        "eve_present": eve_present,
    }


# ---------------------------------------------------------------------------
# Entanglement Swapping
# ---------------------------------------------------------------------------

def entanglement_swapping(fidelity_ab: float,
                          fidelity_bc: float) -> float:
    """Compute the fidelity of entanglement after swapping.

    Why: Entanglement swapping extends entanglement over longer distances.
    If A-B share a Bell pair and B-C share a Bell pair, a Bell measurement
    at B creates entanglement between A and C (who never interacted).
    The output fidelity is F_AC = F_AB · F_BC + (1-F_AB)(1-F_BC)/3
    for Werner states, showing that fidelity degrades with each swap.
    """
    # Werner state model: ρ = F|Φ+⟩⟨Φ+| + (1-F)I/4
    f_out = (fidelity_ab * fidelity_bc +
             (1 - fidelity_ab) * (1 - fidelity_bc) / 3)
    return f_out


def quantum_repeater_chain(n_segments: int, link_fidelity: float,
                           distillation: bool = False,
                           n_distill_rounds: int = 1) -> float:
    """Compute end-to-end fidelity of a quantum repeater chain.

    Why: Direct transmission of photons over optical fiber is limited
    to ~100 km due to exponential loss.  Quantum repeaters use
    entanglement swapping at intermediate nodes to extend the range.
    Without distillation, fidelity degrades exponentially with distance.
    With distillation, high fidelity can be maintained at the cost of
    lower rate.
    """
    current_fidelity = link_fidelity

    if distillation:
        # Entanglement distillation (BBPSSW protocol, simplified)
        # Each round: F_out = F² / (F² + (1-F)²) (for 2-to-1 distillation)
        for _ in range(n_distill_rounds):
            f = current_fidelity
            current_fidelity = f ** 2 / (f ** 2 + (1 - f) ** 2)

    # Swap through the chain
    # n_segments links → log2(n_segments) levels of swapping
    n_levels = int(np.ceil(np.log2(n_segments)))
    fid = current_fidelity

    for _ in range(n_levels):
        fid = entanglement_swapping(fid, current_fidelity)

    return fid


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_bb84():
    """Simulate BB84 protocol with and without eavesdropper."""
    print("=" * 60)
    print("DEMO 1: BB84 Quantum Key Distribution")
    print("=" * 60)

    print(f"\n  {'Scenario':<25} {'Sent':>6} {'Sifted':>8} {'Key len':>8} "
          f"{'QBER':>8} {'Secure?':>8}")
    print(f"  {'─' * 68}")

    for eve in [False, True]:
        for n in [100, 500, 2000]:
            result = bb84_protocol(n, eve_present=eve)
            secure = "Yes" if result['error_rate'] < 0.11 else "No"
            label = f"{'Eve' if eve else 'No Eve'}, n={n}"
            print(f"  {label:<25} {result['n_sent']:6d} {result['n_sifted']:8d} "
                  f"{result['key_length']:8d} {result['error_rate']:8.3f} "
                  f"{secure:>8}")

    # Why: Without Eve, QBER ≈ 0.  With intercept-resend Eve, QBER ≈ 25%
    # because Eve's random basis choice matches Alice's only 50% of the time,
    # and wrong-basis measurement introduces 50% error → net 25%.
    print(f"\n  Security threshold: QBER < 11% (BB84 with one-way key distillation)")
    print(f"  Intercept-resend attack: expected QBER ≈ 25%")


def demo_e91():
    """Simulate E91 protocol with CHSH test."""
    print("\n" + "=" * 60)
    print("DEMO 2: E91 Entanglement-Based QKD")
    print("=" * 60)

    print(f"\n  {'Scenario':<20} {'Pairs':>7} {'CHSH S':>8} {'Violation':>10} "
          f"{'QBER':>8} {'Key len':>8}")
    print(f"  {'─' * 66}")

    for eve in [False, True]:
        for n in [500, 2000, 5000]:
            result = e91_protocol(n, eve_present=eve)
            label = f"{'Eve' if eve else 'No Eve'}, n={n}"
            print(f"  {label:<20} {result['n_pairs']:7d} "
                  f"{result['chsh_value']:8.3f} "
                  f"{'Yes' if result['chsh_violation'] else 'No':>10} "
                  f"{result['error_rate']:8.3f} {result['key_length']:8d}")

    # Why: The CHSH value S = 2√2 ≈ 2.83 for perfect Bell states (quantum).
    # Classical bound is |S| ≤ 2.  Eve's interception breaks entanglement,
    # causing S ≈ 0 (no correlation), which is detected.
    print(f"\n  Quantum bound: |S| = 2√2 ≈ {2 * np.sqrt(2):.4f}")
    print(f"  Classical bound: |S| ≤ 2.0")


def demo_entanglement_swapping():
    """Show entanglement swapping fidelity."""
    print("\n" + "=" * 60)
    print("DEMO 3: Entanglement Swapping")
    print("=" * 60)

    print(f"\n  {'F_AB':>8} {'F_BC':>8} {'F_AC (swapped)':>16}")
    print(f"  {'─' * 36}")

    for f_ab in [0.99, 0.95, 0.90, 0.85, 0.80]:
        for f_bc in [0.99, 0.95, 0.90]:
            f_ac = entanglement_swapping(f_ab, f_bc)
            print(f"  {f_ab:8.3f} {f_bc:8.3f} {f_ac:16.4f}")
        print()


def demo_repeater_chain():
    """Analyze quantum repeater chain performance."""
    print("\n" + "=" * 60)
    print("DEMO 4: Quantum Repeater Chain")
    print("=" * 60)

    # Why: Without repeaters, entanglement fidelity decays exponentially
    # with fiber length.  Repeaters with distillation can maintain high
    # fidelity at the cost of lower key generation rate.

    print(f"\n  Link fidelity = 0.95")
    print(f"  {'Segments':>10} {'No distill':>12} {'1 round':>12} {'2 rounds':>12}")
    print(f"  {'─' * 50}")

    for n_seg in [2, 4, 8, 16, 32]:
        f_no = quantum_repeater_chain(n_seg, 0.95, distillation=False)
        f_1 = quantum_repeater_chain(n_seg, 0.95, distillation=True, n_distill_rounds=1)
        f_2 = quantum_repeater_chain(n_seg, 0.95, distillation=True, n_distill_rounds=2)
        print(f"  {n_seg:10d} {f_no:12.4f} {f_1:12.4f} {f_2:12.4f}")

    print(f"\n  Link fidelity comparison (8 segments):")
    print(f"  {'Link F':>10} {'No distill':>12} {'With distill':>14}")
    print(f"  {'─' * 40}")
    for f_link in [0.99, 0.97, 0.95, 0.92, 0.90, 0.85, 0.80]:
        f_no = quantum_repeater_chain(8, f_link, distillation=False)
        f_yes = quantum_repeater_chain(8, f_link, distillation=True, n_distill_rounds=2)
        print(f"  {f_link:10.3f} {f_no:12.4f} {f_yes:14.4f}")


def demo_photon_loss():
    """Analyze photon loss and repeater spacing."""
    print("\n" + "=" * 60)
    print("DEMO 5: Photon Loss and Repeater Spacing")
    print("=" * 60)

    # Why: Optical fiber has ~0.2 dB/km loss at 1550 nm.  The transmission
    # probability decays as η = 10^{-αL/10}, making direct QKD impractical
    # beyond ~100 km.  Repeaters break the distance into shorter segments
    # where η is reasonable.

    loss_db_per_km = 0.2  # Standard telecom fiber

    print(f"\n  Fiber loss: {loss_db_per_km} dB/km")
    print(f"\n  {'Distance':>10} {'Direct η':>12} {'Repeaters':>10} "
          f"{'Segment η':>12} {'Key rate':>12}")
    print(f"  {'─' * 60}")

    for distance_km in [10, 50, 100, 200, 500, 1000, 5000]:
        direct_eta = 10 ** (-loss_db_per_km * distance_km / 10)

        # Optimal repeater spacing: ~50 km segments
        segment_km = 50
        n_repeaters = max(0, int(distance_km / segment_km) - 1)
        n_segments = n_repeaters + 1
        segment_eta = 10 ** (-loss_db_per_km * (distance_km / n_segments) / 10)

        # Key rate ~ η for direct, ~ η^{1/n} for repeaters (simplified)
        key_rate = segment_eta if n_repeaters > 0 else direct_eta

        print(f"  {distance_km:10d} {direct_eta:12.2e} {n_repeaters:10d} "
              f"{segment_eta:12.4f} {key_rate:12.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("+" + "=" * 58 + "+")
    print("|   Quantum Computing - 22: Quantum Networking               |")
    print("+" + "=" * 58 + "+")

    np.random.seed(2026)

    demo_bb84()
    demo_e91()
    demo_entanglement_swapping()
    demo_repeater_chain()
    demo_photon_loss()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
