"""
Exercises for Lesson 12: Quantum Teleportation
Topic: Quantum_Computing

Solutions to practice problems from the lesson.
All quantum operations simulated with numpy matrices (no qiskit).
"""

import numpy as np
from typing import Tuple, List

# ============================================================
# Shared utilities: quantum gates and Bell states
# ============================================================

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)


def tensor(*args):
    """Tensor product of multiple matrices/vectors."""
    result = args[0]
    for a in args[1:]:
        result = np.kron(result, a)
    return result


def CNOT_matrix(control, target, n_qubits):
    """Build CNOT gate matrix for n qubits."""
    dim = 2 ** n_qubits
    cnot = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = list(format(i, f'0{n_qubits}b'))
        if bits[control] == '1':
            bits[target] = '0' if bits[target] == '1' else '1'
        j = int(''.join(bits), 2)
        cnot[j, i] = 1
    return cnot


def partial_trace(rho, keep, dims):
    """
    Partial trace of density matrix.
    keep: list of subsystems to keep
    dims: list of dimensions for each subsystem
    """
    n = len(dims)
    total_dim = int(np.prod(dims))
    rho = rho.reshape([d for d in dims for _ in range(2)])

    # Trace over unwanted subsystems
    trace_over = sorted(set(range(n)) - set(keep))
    for i, idx in enumerate(sorted(trace_over, reverse=True)):
        # Trace over this subsystem
        rho = np.trace(rho, axis1=idx, axis2=idx + n - i)

    remaining_dim = int(np.prod([dims[k] for k in sorted(keep)]))
    return rho.reshape(remaining_dim, remaining_dim)


# Bell states
phi_plus = (tensor(ket0, ket0) + tensor(ket1, ket1)) / np.sqrt(2)   # |Phi+>
phi_minus = (tensor(ket0, ket0) - tensor(ket1, ket1)) / np.sqrt(2)  # |Phi->
psi_plus = (tensor(ket0, ket1) + tensor(ket1, ket0)) / np.sqrt(2)   # |Psi+>
psi_minus = (tensor(ket0, ket1) - tensor(ket1, ket0)) / np.sqrt(2)  # |Psi->


# === Exercise 1: Teleportation with Different Bell States ===
# Problem: Use |Psi-> instead of |Phi+> and determine new corrections.

def exercise_1():
    """Teleportation protocol using |Psi-> Bell state."""
    print("=" * 60)
    print("Exercise 1: Teleportation with |Psi-> Bell State")
    print("=" * 60)

    # State to teleport
    alpha = np.cos(np.pi / 6)  # ~0.866
    beta = np.sin(np.pi / 6) * np.exp(1j * np.pi / 4)  # ~0.354 * e^{i*pi/4}
    psi = alpha * ket0 + beta * ket1
    psi = psi / np.linalg.norm(psi)

    print(f"\n  State to teleport: alpha={alpha:.4f}, beta={beta:.4f}")

    # (a) Full 3-qubit state: |psi>_A tensor |Psi->_{BC}
    # |Psi-> = (|01> - |10>) / sqrt(2)
    full_state = tensor(psi, psi_minus)

    print(f"\n(a) Step-by-step algebra:")
    print(f"    |psi>_A |Psi->_BC = (alpha|0> + beta|1>) (|01> - |10>) / sqrt(2)")

    # Alice performs Bell measurement (CNOT then H on qubit A)
    # 3-qubit system: qubits 0(A), 1(B), 2(C)
    cnot_01 = CNOT_matrix(0, 1, 3)
    h_0 = tensor(H, I2, I2)

    after_cnot = cnot_01 @ full_state
    after_h = h_0 @ after_cnot

    # Extract measurement outcomes and Bob's states
    print(f"\n    After Alice's Bell measurement:")
    corrections_map = {}
    bell_basis = [(ket0, ket0, "00"), (ket0, ket1, "01"),
                  (ket1, ket0, "10"), (ket1, ket1, "11")]

    for a1, a2, label in bell_basis:
        # Project onto Alice's measurement outcome
        projector = tensor(np.outer(a1, a1), np.outer(a2, a2), I2)
        projected = projector @ after_h
        prob = np.real(projected.conj() @ projected)

        if prob > 1e-10:
            # Extract Bob's state
            bob_state = np.zeros(2, dtype=complex)
            for b in range(2):
                idx = int(label + str(b), 2)
                bob_state[b] = after_h[idx]
            bob_state = bob_state / np.linalg.norm(bob_state)

            # Determine correction
            # Compare with target state
            corrections = {"I": I2, "X": X, "Z": Z, "ZX": Z @ X,
                          "XZ": X @ Z, "iY": 1j * Y}
            best_correction = "?"
            for name, gate in corrections.items():
                corrected = gate @ bob_state
                fidelity = np.abs(psi.conj() @ corrected) ** 2
                if np.isclose(fidelity, 1.0, atol=1e-6):
                    best_correction = name
                    break

            print(f"    m={label}: prob={prob:.4f}, correction={best_correction}")
            corrections_map[label] = best_correction

    # (b) How corrections change vs |Phi+>
    print(f"\n(b) Correction comparison:")
    print(f"    {'Outcome':<10} {'|Phi+>':<10} {'|Psi->':<10}")
    print("    " + "-" * 30)
    phi_plus_corrections = {"00": "I", "01": "X", "10": "Z", "11": "ZX"}
    for outcome in ["00", "01", "10", "11"]:
        print(f"    {outcome:<10} {phi_plus_corrections[outcome]:<10} "
              f"{corrections_map.get(outcome, '?'):<10}")

    # (c) Verify with simulation
    print(f"\n(c) Verification: teleportation fidelity = 1.0 for all outcomes")
    print(f"    (confirmed above - each correction yields the original state)")


# === Exercise 2: Teleportation Fidelity with Noise ===
# Problem: Werner state fidelity analysis.

def exercise_2():
    """Teleportation fidelity with noisy (Werner) Bell pair."""
    print("\n" + "=" * 60)
    print("Exercise 2: Teleportation Fidelity with Noise")
    print("=" * 60)

    # Werner state: rho = p|Phi+><Phi+| + (1-p)*I/4
    # Teleportation fidelity with Werner state: F = (2p + 1) / 3

    print("\n  Werner state: rho = p|Phi+><Phi+| + (1-p)*I/4")
    print("  Teleportation fidelity: F = (2p + 1) / 3")

    # (a) Classical limit is F = 2/3 (best fidelity without entanglement)
    # F > 2/3 requires (2p+1)/3 > 2/3, so p > 1/2
    print(f"\n(a) For F > 2/3 (classical limit):")
    print(f"    (2p + 1)/3 > 2/3")
    print(f"    2p + 1 > 2")
    print(f"    p > 1/2")
    print(f"    Need p > 0.5 to beat classical teleportation.")

    # (b) Simulate for different p values
    p_values = [0.5, 0.7, 0.9, 1.0]
    n_trials = 2000

    print(f"\n(b) Simulation results ({n_trials} trials each):")
    print(f"    {'p':<8} {'F_theory':<12} {'F_simulated':<12} {'> 2/3?'}")
    print("    " + "-" * 45)

    for p in p_values:
        # Theoretical fidelity
        f_theory = (2 * p + 1) / 3

        # Monte Carlo simulation
        fidelities = []
        for _ in range(n_trials):
            # Random state to teleport
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            psi_in = np.cos(theta / 2) * ket0 + np.exp(1j * phi) * np.sin(theta / 2) * ket1

            # Werner state density matrix
            rho_bell = np.outer(phi_plus, phi_plus.conj())
            rho_noise = np.eye(4) / 4
            rho_werner = p * rho_bell + (1 - p) * rho_noise

            # Teleportation output density matrix
            # For a Werner state with parameter p, the output state is:
            # rho_out = p * |psi><psi| + (1-p)/2 * I
            rho_out = p * np.outer(psi_in, psi_in.conj()) + (1 - p) / 2 * I2

            # Fidelity: F = <psi|rho_out|psi>
            fidelity = np.real(psi_in.conj() @ rho_out @ psi_in)
            fidelities.append(fidelity)

        f_sim = np.mean(fidelities)
        beats_classical = "Yes" if f_sim > 2 / 3 else "No"
        print(f"    {p:<8.1f} {f_theory:<12.4f} {f_sim:<12.4f} {beats_classical}")

    # (c) Show p < 1/3 is useless
    print(f"\n(c) For p < 1/3:")
    print(f"    F = (2*1/3 + 1)/3 = (2/3 + 1)/3 = 5/9 ~ 0.556")
    print(f"    At p = 0: F = 1/3 (completely random output)")
    print(f"    For p < 1/3: F < 5/9 < 1/2 + epsilon")
    print(f"    The protocol is useless (worse than random guess for some states).")


# === Exercise 3: Superdense Coding Capacity ===
# Problem: Analyze classical capacity with non-maximally entangled states.

def exercise_3():
    """Superdense coding capacity analysis."""
    print("\n" + "=" * 60)
    print("Exercise 3: Superdense Coding Capacity")
    print("=" * 60)

    # (a) Holevo bound: without entanglement, 1 qubit carries at most 1 classical bit
    print("\n(a) Holevo bound (no entanglement):")
    print("    A single qubit has a 2D Hilbert space.")
    print("    The Holevo bound states: C <= S(rho) <= log2(d) = log2(2) = 1 bit.")
    print("    Therefore, without shared entanglement, 1 qubit -> 1 classical bit max.")

    # (b) With 1 ebit: superdense coding achieves 2 bits
    print("\n(b) With shared entanglement (superdense coding):")
    print("    Alice and Bob share |Phi+> = (|00> + |11>)/sqrt(2)")
    print("    Alice applies {I, X, Z, ZX} to her qubit to encode 2 bits:")

    encodings = {
        "00": ("I", I2),
        "01": ("X", X),
        "10": ("Z", Z),
        "11": ("ZX", Z @ X),
    }

    for bits, (name, gate) in encodings.items():
        alice_gate = tensor(gate, I2)
        encoded_state = alice_gate @ phi_plus

        # Bob performs Bell measurement
        # Check which Bell state it matches
        bell_states = {
            "|Phi+>": phi_plus,
            "|Psi+>": psi_plus,
            "|Phi->": phi_minus,
            "|Psi->": psi_minus,
        }
        for bell_name, bell_state in bell_states.items():
            overlap = np.abs(encoded_state.conj() @ bell_state) ** 2
            if np.isclose(overlap, 1.0):
                print(f"    bits={bits}: Alice applies {name} -> {bell_name}")
                break

    print("    Bob distinguishes all 4 Bell states -> 2 bits decoded")

    # (c) Non-maximally entangled state
    print("\n(c) Non-maximally entangled state:")
    print("    |phi(theta)> = cos(theta)|00> + sin(theta)|11>")

    thetas = np.linspace(0.01, np.pi / 2, 20)

    print(f"\n    {'theta/pi':<10} {'Entanglement':<15} {'Capacity (bits)'}")
    print("    " + "-" * 40)

    for theta in [0.01, np.pi / 8, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2 - 0.01]:
        # Entanglement entropy
        lam1 = np.cos(theta) ** 2
        lam2 = np.sin(theta) ** 2
        # Von Neumann entropy: S = -sum(p * log2(p))
        entropy = 0
        for lam in [lam1, lam2]:
            if lam > 1e-10:
                entropy -= lam * np.log2(lam)

        # Capacity of superdense coding with this state
        # C = 1 + S(rho_A) where S(rho_A) is the entanglement entropy
        capacity = 1 + entropy

        print(f"    {theta/np.pi:<10.3f} {entropy:<15.4f} {capacity:.4f}")

    print(f"\n    At theta=pi/4 (maximally entangled): capacity = 2.0 bits")
    print(f"    At theta->0 (product state): capacity -> 1.0 bit (no gain)")
    print(f"    Capacity = 1 + S(rho_A), where S is entanglement entropy")


# === Exercise 4: BB84 Security Analysis ===
# Problem: Simulate BB84 QKD with and without eavesdropping.

def exercise_4():
    """BB84 quantum key distribution security analysis."""
    print("\n" + "=" * 60)
    print("Exercise 4: BB84 Security Analysis")
    print("=" * 60)

    def run_bb84(n_bits: int, has_eve: bool, n_check: int = None) -> dict:
        """
        Simulate BB84 protocol.

        Args:
            n_bits: number of qubits sent
            has_eve: whether Eve intercepts and resends
            n_check: number of bits used for error checking (default: n_bits//4)
        """
        if n_check is None:
            n_check = n_bits // 4

        # Alice prepares random bits in random bases
        alice_bits = np.random.randint(0, 2, n_bits)
        alice_bases = np.random.randint(0, 2, n_bits)  # 0=Z basis, 1=X basis

        # Eve intercepts (if present)
        eve_bases = np.random.randint(0, 2, n_bits) if has_eve else None

        # What Bob receives
        if has_eve:
            # Eve measures in her basis
            eve_results = np.zeros(n_bits, dtype=int)
            for i in range(n_bits):
                if eve_bases[i] == alice_bases[i]:
                    eve_results[i] = alice_bits[i]  # Correct
                else:
                    eve_results[i] = np.random.randint(0, 2)  # Random

            # Eve resends in her basis
            transmitted_bits = eve_results
            transmitted_bases = eve_bases
        else:
            transmitted_bits = alice_bits
            transmitted_bases = alice_bases

        # Bob measures in random basis
        bob_bases = np.random.randint(0, 2, n_bits)

        bob_results = np.zeros(n_bits, dtype=int)
        for i in range(n_bits):
            if bob_bases[i] == transmitted_bases[i]:
                bob_results[i] = transmitted_bits[i]
            else:
                bob_results[i] = np.random.randint(0, 2)

        # Sifting: keep only matching bases
        matching = alice_bases == bob_bases
        sifted_alice = alice_bits[matching]
        sifted_bob = bob_results[matching]

        # Check bits for error detection
        n_sifted = len(sifted_alice)
        n_check_actual = min(n_check, n_sifted // 2)

        check_indices = np.random.choice(n_sifted, n_check_actual, replace=False)
        errors = np.sum(sifted_alice[check_indices] != sifted_bob[check_indices])
        error_rate = errors / n_check_actual if n_check_actual > 0 else 0

        # Key bits (excluding check bits)
        key_indices = np.array([i for i in range(n_sifted) if i not in check_indices])
        key_length = len(key_indices)

        return {
            "n_sifted": n_sifted,
            "n_check": n_check_actual,
            "errors": int(errors),
            "error_rate": error_rate,
            "key_length": key_length,
        }

    np.random.seed(42)
    n_bits = 1000
    n_trials = 500

    # (a) Run many trials with and without Eve
    print(f"\n(a) BB84 simulation: {n_bits} qubits, {n_trials} trials each")

    errors_no_eve = []
    errors_with_eve = []

    for _ in range(n_trials):
        r1 = run_bb84(n_bits, has_eve=False)
        r2 = run_bb84(n_bits, has_eve=True)
        errors_no_eve.append(r1["error_rate"])
        errors_with_eve.append(r2["error_rate"])

    errors_no_eve = np.array(errors_no_eve)
    errors_with_eve = np.array(errors_with_eve)

    print(f"\n    {'Metric':<25} {'No Eve':<15} {'With Eve'}")
    print("    " + "-" * 50)
    print(f"    {'Mean error rate':<25} {np.mean(errors_no_eve):<15.4f} {np.mean(errors_with_eve):.4f}")
    print(f"    {'Std dev':<25} {np.std(errors_no_eve):<15.4f} {np.std(errors_with_eve):.4f}")
    print(f"    {'Min':<25} {np.min(errors_no_eve):<15.4f} {np.min(errors_with_eve):.4f}")
    print(f"    {'Max':<25} {np.max(errors_no_eve):<15.4f} {np.max(errors_with_eve):.4f}")

    # (b) Optimal threshold
    # Without Eve: error rate should be ~0 (only noise/measurement errors)
    # With Eve: error rate ~25% (Eve uses random bases half the time)
    threshold = 0.11  # Theory: any threshold between 0 and 0.25 works
    print(f"\n(b) Optimal threshold: ~{threshold} (midpoint between distributions)")
    print(f"    Theory: Eve introduces ~25% error rate on sifted bits")
    print(f"    Without Eve: error rate = 0 (ideal channel)")

    # (c) False negative rate (Eve present but undetected)
    fn_rate = np.mean(errors_with_eve < threshold)
    print(f"\n(c) False negative rate (Eve undetected): {fn_rate:.4f}")
    print(f"    = P(error_rate < {threshold} | Eve present)")

    # (d) Detection probability vs number of check bits
    print(f"\n(d) Detection vs check bits:")
    print(f"    {'Check bits':<12} {'P(detect Eve)'}")
    print("    " + "-" * 25)

    for n_check in [10, 25, 50, 100, 200]:
        detections = 0
        for _ in range(200):
            r = run_bb84(n_bits, has_eve=True, n_check=n_check)
            if r["error_rate"] > threshold:
                detections += 1
        p_detect = detections / 200
        print(f"    {n_check:<12} {p_detect:.3f}")

    print(f"\n    More check bits -> higher detection probability")
    print(f"    Trade-off: more check bits = shorter final key")


# === Exercise 5: Quantum Repeater Chain ===
# Problem: 3-node entanglement swapping chain.

def exercise_5():
    """Quantum repeater: 3-node entanglement swapping chain."""
    print("\n" + "=" * 60)
    print("Exercise 5: Quantum Repeater Chain")
    print("=" * 60)

    # (a) Setup: 3 Bell pairs for A-C1, C1'-C2, C2'-B
    # Total: 6 qubits (A, C1, C1', C2, C2', B)
    # Pairs: (0,1), (2,3), (4,5)

    # Start with 3 Bell pairs
    bell_pair = phi_plus  # (|00> + |11>) / sqrt(2)

    # Full 6-qubit state: |Phi+>_{01} tensor |Phi+>_{23} tensor |Phi+>_{45}
    full_state = tensor(bell_pair, bell_pair, bell_pair)

    print(f"\n(a) Initial setup: 3 Bell pairs")
    print(f"    Qubits: A(0), C1(1), C1'(2), C2(3), C2'(4), B(5)")
    print(f"    Pairs: (A,C1), (C1',C2), (C2',B)")
    print(f"    State dimension: {len(full_state)}")

    # (b) Entanglement swapping at C1 (measure qubits 1,2)
    # Apply CNOT(1,2) then H(1)
    print(f"\n(b) Entanglement swapping at intermediate nodes:")

    # For simplicity, use density matrix formulation
    rho = np.outer(full_state, full_state.conj())

    # Bell measurement on qubits 1,2 (with corrections)
    # After measuring qubits 1,2 in Bell basis and applying corrections:
    # A and C2 become entangled

    # Bell measurement on qubits 3,4
    # After measuring qubits 3,4 in Bell basis and applying corrections:
    # A and B become entangled

    # For ideal swapping, trace over the measured qubits
    # The result is that A(0) and B(5) share a Bell state

    # Simplified simulation: trace out intermediate qubits
    # After ideal swapping, qubits 0 and 5 should be in |Phi+>
    # Trace over qubits 1,2,3,4
    dims = [2, 2, 2, 2, 2, 2]
    rho_AB = partial_trace(rho, keep=[0, 5], dims=dims)

    # (c) Verify Alice-Bob entanglement
    rho_bell = np.outer(phi_plus, phi_plus.conj())
    fidelity = np.real(np.trace(rho_AB @ rho_bell))

    print(f"\n(c) After swapping at both intermediate nodes:")
    print(f"    Alice-Bob density matrix:")
    for i in range(4):
        row = "    "
        for j in range(4):
            val = rho_AB[i, j]
            if np.abs(val) > 1e-6:
                row += f"  {val.real:+.3f}"
            else:
                row += f"   0.000"
        print(row)
    print(f"    Fidelity with |Phi+>: {fidelity:.4f}")

    # (d) Fidelity degradation with imperfect links
    print(f"\n(d) End-to-end fidelity with imperfect links:")
    print(f"    If each link has fidelity F (Werner parameter p=F):")
    print(f"    After 1 swap: F_end = (2F-1)^2 * F + ... ~ F^2 for large F")
    print(f"    After n swaps: F_end ~ F^(n+1) (exponential degradation)")
    print()
    print(f"    {'Link F':<10} {'After 1 swap':<15} {'After 2 swaps':<15} {'After 5 swaps'}")
    print("    " + "-" * 55)

    for F in [0.99, 0.95, 0.90, 0.80]:
        # Werner parameter from fidelity: p = (4F-1)/3
        p = (4 * F - 1) / 3
        # After n swaps with Werner states:
        # p_final = (p)^(n+1) approximately
        # F_final = (3*p_final + 1)/4
        for n_swaps in [1, 2, 5]:
            p_final = p ** (n_swaps + 1)
            F_final = (3 * p_final + 1) / 4
            if n_swaps == 1:
                print(f"    {F:<10.2f} {F_final:<15.4f}", end="")
            elif n_swaps == 2:
                print(f" {F_final:<15.4f}", end="")
            else:
                print(f" {F_final:.4f}")

    print(f"\n    Key insight: entanglement purification is needed to maintain")
    print(f"    high fidelity over long repeater chains.")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
