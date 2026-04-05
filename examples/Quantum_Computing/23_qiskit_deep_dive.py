"""
23_qiskit_deep_dive.py — Qiskit Concepts: Transpilation, Noise, and Mitigation

Demonstrates (without requiring Qiskit installation):
  - Circuit representation and gate decomposition into basis gates
  - Transpilation: routing and mapping to hardware topology
  - Noise model simulation (depolarizing + readout errors)
  - Measurement error mitigation via calibration matrix
  - Zero-noise extrapolation (ZNE) for error mitigation
  - Parameterized circuit evaluation for variational algorithms

All computations use pure NumPy, mirroring Qiskit's internal logic.
"""

import numpy as np
from typing import List, Tuple, Dict

# ---------------------------------------------------------------------------
# Gate definitions (hardware basis gates for IBM devices)
# ---------------------------------------------------------------------------

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                  [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    """Rz(θ) rotation gate."""
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)


def sx() -> np.ndarray:
    """√X gate (IBM basis gate)."""
    return np.array([[1 + 1j, 1 - 1j],
                     [1 - 1j, 1 + 1j]], dtype=complex) / 2


def kron_list(ops: List[np.ndarray]) -> np.ndarray:
    """Tensor product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


# ---------------------------------------------------------------------------
# Circuit Representation
# ---------------------------------------------------------------------------

class SimpleCircuit:
    """Minimal circuit representation mirroring Qiskit's QuantumCircuit.

    Why: Understanding how circuits are represented internally — as a
    list of (gate_name, qubits, params) tuples — is essential for
    understanding transpilation and optimization.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.gates: List[Tuple[str, List[int], List[float]]] = []

    def h(self, qubit: int):
        self.gates.append(("H", [qubit], []))

    def cx(self, control: int, target: int):
        self.gates.append(("CX", [control, target], []))

    def rz_gate(self, theta: float, qubit: int):
        self.gates.append(("RZ", [qubit], [theta]))

    def sx_gate(self, qubit: int):
        self.gates.append(("SX", [qubit], []))

    def x(self, qubit: int):
        self.gates.append(("X", [qubit], []))

    def depth(self) -> int:
        """Compute circuit depth (longest path through the circuit)."""
        qubit_depth = [0] * self.n_qubits
        for gate_name, qubits, _ in self.gates:
            max_d = max(qubit_depth[q] for q in qubits)
            for q in qubits:
                qubit_depth[q] = max_d + 1
        return max(qubit_depth) if qubit_depth else 0

    def gate_count(self) -> Dict[str, int]:
        """Count gates by type."""
        counts: Dict[str, int] = {}
        for gate_name, _, _ in self.gates:
            counts[gate_name] = counts.get(gate_name, 0) + 1
        return counts

    def to_unitary(self) -> np.ndarray:
        """Compute the full unitary matrix of the circuit."""
        dim = 2 ** self.n_qubits
        U = np.eye(dim, dtype=complex)

        gate_matrices = {
            "H": H, "X": X, "SX": sx(), "CX": CNOT,
        }

        for gate_name, qubits, params in self.gates:
            if gate_name == "RZ":
                mat = rz(params[0])
            else:
                mat = gate_matrices[gate_name]

            if len(qubits) == 1:
                ops = [I] * self.n_qubits
                ops[qubits[0]] = mat
                full_gate = kron_list(ops)
            elif len(qubits) == 2:
                full_gate = _two_qubit_gate(mat, qubits[0], qubits[1],
                                             self.n_qubits)
            else:
                raise ValueError(f"Unsupported gate arity: {len(qubits)}")

            U = full_gate @ U

        return U


def _two_qubit_gate(gate_2q: np.ndarray, q0: int, q1: int,
                    n_qubits: int) -> np.ndarray:
    """Embed a 2-qubit gate into the full Hilbert space."""
    dim = 2 ** n_qubits
    result = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        bits_i = [(i >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits)]
        for j in range(dim):
            bits_j = [(j >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits)]

            # Check that non-target qubits match
            other_match = all(bits_i[k] == bits_j[k]
                              for k in range(n_qubits) if k not in [q0, q1])
            if not other_match:
                continue

            # Extract 2-qubit indices
            i_local = bits_i[q0] * 2 + bits_i[q1]
            j_local = bits_j[q0] * 2 + bits_j[q1]
            result[i, j] = gate_2q[i_local, j_local]

    return result


# ---------------------------------------------------------------------------
# Transpilation: Gate Decomposition
# ---------------------------------------------------------------------------

def decompose_h_to_basis(circ: SimpleCircuit, qubit: int) -> SimpleCircuit:
    """Decompose H gate into IBM basis gates {Rz, SX, CX}.

    Why: Hardware quantum computers support only a limited set of native
    gates.  IBM devices use {Rz(θ), √X, CX}.  The transpiler must
    decompose arbitrary gates: H = Rz(π)·√X·Rz(π) (up to global phase).
    """
    circ.rz_gate(np.pi, qubit)
    circ.sx_gate(qubit)
    circ.rz_gate(np.pi, qubit)
    return circ


def transpile_circuit(source: SimpleCircuit,
                      coupling_map: List[Tuple[int, int]]) -> SimpleCircuit:
    """Transpile a circuit to a given hardware coupling map.

    Why: Real quantum hardware has limited connectivity — not all qubit
    pairs can interact directly.  The transpiler inserts SWAP gates
    (3 CX gates each) to route interactions through connected qubits.
    This increases circuit depth and gate count, which is why hardware
    topology matters.
    """
    result = SimpleCircuit(source.n_qubits)

    # Build adjacency from coupling map
    connected = set()
    for q0, q1 in coupling_map:
        connected.add((q0, q1))
        connected.add((q1, q0))

    for gate_name, qubits, params in source.gates:
        if len(qubits) == 1:
            # Single-qubit gates: decompose H to basis, pass through others
            if gate_name == "H":
                decompose_h_to_basis(result, qubits[0])
            elif gate_name == "RZ":
                result.rz_gate(params[0], qubits[0])
            elif gate_name == "SX":
                result.sx_gate(qubits[0])
            elif gate_name == "X":
                result.sx_gate(qubits[0])
                result.sx_gate(qubits[0])
            else:
                result.gates.append((gate_name, qubits, params))
        elif len(qubits) == 2 and gate_name == "CX":
            q0, q1 = qubits
            if (q0, q1) in connected:
                result.cx(q0, q1)
            elif (q1, q0) in connected:
                # Reverse direction using H-CX-H decomposition
                decompose_h_to_basis(result, q0)
                decompose_h_to_basis(result, q1)
                result.cx(q1, q0)
                decompose_h_to_basis(result, q0)
                decompose_h_to_basis(result, q1)
            else:
                # Need SWAP routing (simplified: insert SWAPs)
                result.cx(q0, q1)  # Placeholder — real transpiler uses BFS
        else:
            result.gates.append((gate_name, qubits, params))

    return result


# ---------------------------------------------------------------------------
# Noise Model
# ---------------------------------------------------------------------------

def noisy_simulation(U_ideal: np.ndarray, n_qubits: int,
                     gate_error: float, readout_error: float,
                     n_shots: int) -> np.ndarray:
    """Simulate a noisy quantum circuit execution.

    Why: Real quantum hardware introduces errors at two stages:
    (1) gate errors (depolarizing noise after each gate), and
    (2) readout errors (bit flips during measurement).  Understanding
    these noise sources is essential for interpreting NISQ results.
    """
    dim = 2 ** n_qubits

    # Start in |0...0⟩
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0

    # Apply ideal unitary (noise is modeled as overall depolarization)
    state = U_ideal @ state

    # Gate noise: approximate as overall depolarizing channel
    rho = np.outer(state, state.conj())
    total_noise = gate_error * dim  # Simplified: proportional to circuit size
    total_noise = min(total_noise, 1.0)
    rho = (1 - total_noise) * rho + total_noise * np.eye(dim, dtype=complex) / dim

    # Sample measurements
    ideal_probs = np.real(np.diag(rho))
    ideal_probs = np.maximum(ideal_probs, 0)
    ideal_probs /= np.sum(ideal_probs)

    # Apply readout errors
    noisy_probs = ideal_probs.copy()
    for i in range(dim):
        bits = [(i >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits)]
        # Each bit has probability readout_error of flipping
        for bit_idx in range(n_qubits):
            if np.random.random() < readout_error:
                bits[bit_idx] ^= 1
        j = sum(b << (n_qubits - 1 - k) for k, b in enumerate(bits))
        # Simplified: mix probabilities
        noisy_probs[j] = ((1 - readout_error) ** n_qubits * ideal_probs[i] +
                           readout_error * ideal_probs[j])

    noisy_probs = np.maximum(noisy_probs, 0)
    noisy_probs /= np.sum(noisy_probs)

    # Sample shots
    counts = np.zeros(dim, dtype=int)
    outcomes = np.random.choice(dim, size=n_shots, p=noisy_probs)
    for o in outcomes:
        counts[o] += 1

    return counts


# ---------------------------------------------------------------------------
# Error Mitigation
# ---------------------------------------------------------------------------

def measurement_calibration(n_qubits: int, readout_error: float,
                            n_shots: int = 10000) -> np.ndarray:
    """Build a measurement calibration matrix.

    Why: Readout errors can be characterized by preparing each
    computational basis state and measuring.  The resulting calibration
    matrix M[i,j] = P(measure i | prepared j) can be inverted to
    correct measurement results: p_corrected = M⁻¹ · p_measured.
    """
    dim = 2 ** n_qubits
    cal_matrix = np.zeros((dim, dim))

    for prepared in range(dim):
        # Prepare state |j⟩ and simulate measurement
        state = np.zeros(dim, dtype=complex)
        state[prepared] = 1.0
        counts = noisy_simulation(np.eye(dim, dtype=complex), n_qubits,
                                  gate_error=0.0,
                                  readout_error=readout_error,
                                  n_shots=n_shots)
        cal_matrix[:, prepared] = counts / n_shots

    return cal_matrix


def mitigate_readout(counts: np.ndarray,
                     cal_matrix: np.ndarray) -> np.ndarray:
    """Apply measurement error mitigation using the calibration matrix.

    Why: Given noisy counts p_noisy = M · p_ideal, we recover
    p_ideal = M⁻¹ · p_noisy.  Negative values are clipped to zero
    and the result is renormalized.
    """
    n_shots = np.sum(counts)
    probs_noisy = counts / n_shots

    # Pseudo-inverse for stability
    cal_inv = np.linalg.pinv(cal_matrix)
    probs_mitigated = cal_inv @ probs_noisy

    # Clip negatives and renormalize
    probs_mitigated = np.maximum(probs_mitigated, 0)
    if np.sum(probs_mitigated) > 0:
        probs_mitigated /= np.sum(probs_mitigated)

    return probs_mitigated * n_shots


def zero_noise_extrapolation(circuit_unitary: np.ndarray, n_qubits: int,
                              base_error: float,
                              n_shots: int = 5000) -> Tuple[float, float]:
    """Zero-noise extrapolation (ZNE) for expectation value estimation.

    Why: ZNE runs the circuit at multiple noise levels (1x, 2x, 3x
    the base noise), measures the observable at each level, and
    extrapolates to the zero-noise limit.  This is the simplest
    error mitigation technique and requires no knowledge of the noise
    model — only the ability to amplify noise (e.g., by stretching pulses
    or inserting identity gates as CNOT pairs).
    """
    dim = 2 ** n_qubits

    # Observable: Z on first qubit
    Z_obs = np.zeros((dim, dim), dtype=complex)
    ops = [I] * n_qubits
    ops[0] = Z
    Z_obs = kron_list(ops)

    # Ideal expectation value
    state_ideal = np.zeros(dim, dtype=complex)
    state_ideal[0] = 1.0
    state_ideal = circuit_unitary @ state_ideal
    exp_ideal = float(np.real(state_ideal.conj() @ Z_obs @ state_ideal))

    # Measure at noise scale factors 1, 2, 3
    scale_factors = [1, 2, 3]
    exp_values = []

    for scale in scale_factors:
        scaled_error = base_error * scale
        counts = noisy_simulation(circuit_unitary, n_qubits,
                                  gate_error=scaled_error,
                                  readout_error=0.0, n_shots=n_shots)
        probs = counts / n_shots

        # Compute ⟨Z⟩ from measurement counts
        exp_z = 0.0
        for i in range(dim):
            bit_0 = (i >> (n_qubits - 1)) & 1
            sign = 1 - 2 * bit_0  # +1 for |0⟩, -1 for |1⟩
            exp_z += sign * probs[i]
        exp_values.append(exp_z)

    # Richardson extrapolation (linear fit to zero noise)
    # Using two-point: E(0) ≈ 2·E(1) - E(2)
    exp_mitigated = 2 * exp_values[0] - exp_values[1]

    return exp_ideal, exp_mitigated


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_gate_decomposition():
    """Show gate decomposition into hardware basis gates."""
    print("=" * 60)
    print("DEMO 1: Gate Decomposition (Transpilation)")
    print("=" * 60)

    # Original circuit: Bell state preparation
    circ = SimpleCircuit(2)
    circ.h(0)
    circ.cx(0, 1)

    print(f"\n  Original circuit:")
    print(f"    Gates: {circ.gate_count()}")
    print(f"    Depth: {circ.depth()}")

    # Transpile to IBM basis {Rz, SX, CX}
    coupling_map = [(0, 1), (1, 2)]
    transpiled = transpile_circuit(circ, coupling_map)

    print(f"\n  Transpiled to {{Rz, SX, CX}} basis:")
    print(f"    Gates: {transpiled.gate_count()}")
    print(f"    Depth: {transpiled.depth()}")

    # Verify equivalence (up to global phase)
    U_orig = circ.to_unitary()
    U_trans = transpiled.to_unitary()

    # Check if unitaries are equivalent up to global phase
    product = U_orig.conj().T @ U_trans
    phase = product[0, 0]
    equiv = np.allclose(product / phase, np.eye(product.shape[0]), atol=1e-6)
    print(f"    Equivalent to original: {equiv}")


def demo_hardware_topology():
    """Show how topology affects transpilation overhead."""
    print("\n" + "=" * 60)
    print("DEMO 2: Hardware Topology Impact")
    print("=" * 60)

    # Why: A fully-connected topology needs no SWAP gates.  A linear chain
    # may need many SWAPs for non-adjacent interactions.  IBM heavy-hex
    # topology is a compromise between connectivity and fabrication yield.

    topologies = {
        "Linear (0-1-2-3)": [(0, 1), (1, 2), (2, 3)],
        "Ring (0-1-2-3-0)": [(0, 1), (1, 2), (2, 3), (3, 0)],
        "Star (0 center)": [(0, 1), (0, 2), (0, 3)],
        "Full (all-to-all)": [(i, j) for i in range(4) for j in range(i+1, 4)],
    }

    # Test circuit: 4-qubit GHZ state
    circ = SimpleCircuit(4)
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(0, 2)
    circ.cx(0, 3)

    print(f"\n  Original: H(0) CX(0,1) CX(0,2) CX(0,3)")
    print(f"  Original gates: {circ.gate_count()}, depth: {circ.depth()}")

    print(f"\n  {'Topology':<25} {'Gates':>20} {'Depth':>8}")
    print(f"  {'─' * 56}")

    for name, cmap in topologies.items():
        trans = transpile_circuit(circ, cmap)
        print(f"  {name:<25} {str(trans.gate_count()):>20} {trans.depth():8d}")


def demo_noise_simulation():
    """Simulate noisy circuit execution."""
    print("\n" + "=" * 60)
    print("DEMO 3: Noisy Circuit Simulation")
    print("=" * 60)

    n_qubits = 2
    n_shots = 10000

    # Bell state circuit
    circ = SimpleCircuit(n_qubits)
    circ.h(0)
    circ.cx(0, 1)
    U = circ.to_unitary()

    # Ideal result: should be |00⟩ and |11⟩ each with 50%
    print(f"\n  Bell state circuit: H(0) CX(0,1)")
    print(f"  Expected: |00⟩ = 50%, |11⟩ = 50%")

    print(f"\n  {'Gate err':>10} {'Readout err':>12} "
          f"{'P(00)':>8} {'P(01)':>8} {'P(10)':>8} {'P(11)':>8}")
    print(f"  {'─' * 58}")

    for g_err in [0.0, 0.01, 0.05, 0.1]:
        for r_err in [0.0, 0.02, 0.05]:
            counts = noisy_simulation(U, n_qubits, g_err, r_err, n_shots)
            probs = counts / n_shots
            print(f"  {g_err:10.3f} {r_err:12.3f} "
                  f"{probs[0]:8.3f} {probs[1]:8.3f} "
                  f"{probs[2]:8.3f} {probs[3]:8.3f}")


def demo_measurement_mitigation():
    """Demonstrate measurement error mitigation."""
    print("\n" + "=" * 60)
    print("DEMO 4: Measurement Error Mitigation")
    print("=" * 60)

    n_qubits = 2
    readout_error = 0.05
    n_shots = 10000

    # Build calibration matrix
    cal_matrix = measurement_calibration(n_qubits, readout_error, n_shots=20000)

    print(f"\n  Calibration matrix (readout error = {readout_error}):")
    labels = [f"|{i:0{n_qubits}b}⟩" for i in range(2 ** n_qubits)]
    header_label = "Meas \\ Prep"
    print(f"  {header_label:>12}", end="")
    for l in labels:
        print(f" {l:>8}", end="")
    print()
    for i in range(2 ** n_qubits):
        print(f"  {labels[i]:>12}", end="")
        for j in range(2 ** n_qubits):
            print(f" {cal_matrix[i, j]:8.3f}", end="")
        print()

    # Apply mitigation to a Bell state measurement
    circ = SimpleCircuit(n_qubits)
    circ.h(0)
    circ.cx(0, 1)
    U = circ.to_unitary()

    noisy_counts = noisy_simulation(U, n_qubits, gate_error=0.0,
                                    readout_error=readout_error,
                                    n_shots=n_shots)
    mitigated_counts = mitigate_readout(noisy_counts, cal_matrix)

    print(f"\n  Bell state measurement ({n_shots} shots):")
    print(f"  {'State':>8} {'Ideal':>8} {'Noisy':>8} {'Mitigated':>10}")
    print(f"  {'─' * 38}")
    ideal = np.array([0.5, 0, 0, 0.5])
    for i in range(4):
        print(f"  {labels[i]:>8} {ideal[i]:8.3f} "
              f"{noisy_counts[i] / n_shots:8.3f} "
              f"{mitigated_counts[i] / n_shots:10.3f}")


def demo_zne():
    """Demonstrate zero-noise extrapolation."""
    print("\n" + "=" * 60)
    print("DEMO 5: Zero-Noise Extrapolation (ZNE)")
    print("=" * 60)

    # Why: ZNE is model-free — it works with any noise type.
    # The key idea: if E(λε) is the expectation value at noise level λε,
    # then E(0) can be estimated by extrapolation from E(ε), E(2ε), E(3ε).

    n_qubits = 2
    circ = SimpleCircuit(n_qubits)
    circ.h(0)
    circ.cx(0, 1)
    U = circ.to_unitary()

    print(f"\n  Circuit: Bell state, Observable: Z₀")
    print(f"\n  {'Base error':>12} {'Ideal ⟨Z⟩':>12} {'Mitigated ⟨Z⟩':>14} "
          f"{'Improvement':>12}")
    print(f"  {'─' * 54}")

    for base_err in [0.01, 0.02, 0.05, 0.1, 0.15]:
        ideal_exp, mitigated_exp = zero_noise_extrapolation(
            U, n_qubits, base_err, n_shots=10000)
        # Noisy (unmitigated) for comparison
        dim = 2 ** n_qubits
        counts = noisy_simulation(U, n_qubits, gate_error=base_err,
                                  readout_error=0.0, n_shots=10000)
        probs = counts / np.sum(counts)
        noisy_exp = sum((1 - 2 * ((i >> (n_qubits - 1)) & 1)) * probs[i]
                        for i in range(dim))

        err_noisy = abs(noisy_exp - ideal_exp)
        err_mitigated = abs(mitigated_exp - ideal_exp)
        improvement = err_noisy / err_mitigated if err_mitigated > 1e-6 else float('inf')

        print(f"  {base_err:12.3f} {ideal_exp:12.4f} {mitigated_exp:14.4f} "
              f"{improvement:12.1f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("+" + "=" * 58 + "+")
    print("|   Quantum Computing - 23: Qiskit Deep Dive                 |")
    print("+" + "=" * 58 + "+")

    np.random.seed(2026)

    demo_gate_decomposition()
    demo_hardware_topology()
    demo_noise_simulation()
    demo_measurement_mitigation()
    demo_zne()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
