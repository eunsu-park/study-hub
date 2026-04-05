"""
03_quantum_circuits.py — Quantum Circuit Simulator

Demonstrates:
  - A simple circuit builder class that accumulates gates and applies them
  - Support for single-qubit and multi-qubit (controlled) gates
  - Multi-qubit state simulation via tensor products and full-system operators
  - Measurement of multi-qubit states (Born rule)
  - Circuit depth and gate count analysis
  - Building entangled states (Bell, GHZ) with the circuit API

All computations use pure NumPy.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional

# ---------------------------------------------------------------------------
# Standard gate matrices
# ---------------------------------------------------------------------------
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

# Why: CNOT is defined as a 4×4 matrix for direct 2-qubit application.
# For the circuit simulator, we store it and remap qubit indices on the fly.
CNOT_MATRIX = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)

SWAP_MATRIX = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=complex)


def Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def Rz(theta: float) -> np.ndarray:
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)


# ---------------------------------------------------------------------------
# Quantum Circuit class
# ---------------------------------------------------------------------------

class QuantumCircuit:
    """A minimal quantum circuit simulator.

    Why: Rather than tracking a unitary matrix for the whole circuit (which is
    2^n × 2^n), we store a list of gate instructions and apply them one by one
    to the state vector.  This 'state-vector simulation' approach scales as
    O(2^n) per gate application, matching how real simulators work.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        # Why: Initialize to |00...0⟩ — the standard starting state in almost
        # all quantum algorithms.  Algorithms begin by applying Hadamard to
        # create superposition from this deterministic start.
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0
        # Gate log: list of (gate_name, qubit_indices, matrix)
        self.gates: List[Tuple[str, Tuple[int, ...], np.ndarray]] = []

    def _validate_qubits(self, qubits: Tuple[int, ...]):
        for q in qubits:
            if q < 0 or q >= self.n_qubits:
                raise ValueError(f"Qubit index {q} out of range [0, {self.n_qubits})")

    def _build_full_operator(self, gate: np.ndarray, target: int) -> np.ndarray:
        """Build the 2^n × 2^n operator for a single-qubit gate on target qubit.

        Why: To apply a single-qubit gate U to qubit k in an n-qubit system,
        we compute the tensor product: I ⊗ ... ⊗ U ⊗ ... ⊗ I, where U sits
        at position k (counting from the most significant qubit = 0).
        np.kron implements the tensor product for matrices.
        """
        ops = [I] * self.n_qubits
        ops[target] = gate
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    def _build_controlled_operator(self, gate: np.ndarray,
                                    control: int, target: int) -> np.ndarray:
        """Build operator for a controlled gate: C-U with arbitrary control and target.

        Why: For non-adjacent or reordered control/target qubits, we cannot
        simply use the standard 4×4 CNOT matrix.  Instead, we construct the
        operator using projectors:
            C-U = |0⟩⟨0|_c ⊗ I_t  +  |1⟩⟨1|_c ⊗ U_t
        tensored with identity on all other qubits.
        """
        proj_0 = np.array([[1, 0], [0, 0]], dtype=complex)
        proj_1 = np.array([[0, 0], [0, 1]], dtype=complex)

        # Build the "control=0 → do nothing" part
        ops_0 = [I] * self.n_qubits
        ops_0[control] = proj_0
        term_0 = ops_0[0]
        for op in ops_0[1:]:
            term_0 = np.kron(term_0, op)

        # Build the "control=1 → apply U" part
        ops_1 = [I] * self.n_qubits
        ops_1[control] = proj_1
        ops_1[target] = gate
        term_1 = ops_1[0]
        for op in ops_1[1:]:
            term_1 = np.kron(term_1, op)

        return term_0 + term_1

    # -- Single-qubit gate methods --

    def h(self, qubit: int) -> 'QuantumCircuit':
        """Apply Hadamard gate."""
        self._validate_qubits((qubit,))
        self.gates.append(("H", (qubit,), H))
        return self

    def x(self, qubit: int) -> 'QuantumCircuit':
        self._validate_qubits((qubit,))
        self.gates.append(("X", (qubit,), X))
        return self

    def y(self, qubit: int) -> 'QuantumCircuit':
        self._validate_qubits((qubit,))
        self.gates.append(("Y", (qubit,), Y))
        return self

    def z(self, qubit: int) -> 'QuantumCircuit':
        self._validate_qubits((qubit,))
        self.gates.append(("Z", (qubit,), Z))
        return self

    def s(self, qubit: int) -> 'QuantumCircuit':
        self._validate_qubits((qubit,))
        self.gates.append(("S", (qubit,), S))
        return self

    def t(self, qubit: int) -> 'QuantumCircuit':
        self._validate_qubits((qubit,))
        self.gates.append(("T", (qubit,), T))
        return self

    def ry(self, qubit: int, theta: float) -> 'QuantumCircuit':
        self._validate_qubits((qubit,))
        self.gates.append((f"Ry({theta:.3f})", (qubit,), Ry(theta)))
        return self

    def rz(self, qubit: int, theta: float) -> 'QuantumCircuit':
        self._validate_qubits((qubit,))
        self.gates.append((f"Rz({theta:.3f})", (qubit,), Rz(theta)))
        return self

    # -- Two-qubit gate methods --

    def cx(self, control: int, target: int) -> 'QuantumCircuit':
        """Apply CNOT (controlled-X) gate."""
        self._validate_qubits((control, target))
        if control == target:
            raise ValueError("Control and target must be different qubits.")
        self.gates.append(("CX", (control, target), X))
        return self

    def cz(self, control: int, target: int) -> 'QuantumCircuit':
        """Apply controlled-Z gate."""
        self._validate_qubits((control, target))
        self.gates.append(("CZ", (control, target), Z))
        return self

    def swap(self, q1: int, q2: int) -> 'QuantumCircuit':
        """SWAP via three CNOTs."""
        self.cx(q1, q2).cx(q2, q1).cx(q1, q2)
        return self

    # -- Execution --

    def run(self) -> np.ndarray:
        """Execute all gates sequentially on the state vector.

        Why: This is the core simulation loop.  Each gate is expanded into a
        full 2^n × 2^n matrix and multiplied into the state vector.  This is
        the simplest (though not most memory-efficient) approach.
        """
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0

        for name, qubits, gate_matrix in self.gates:
            if len(qubits) == 1:
                full_op = self._build_full_operator(gate_matrix, qubits[0])
            elif len(qubits) == 2:
                full_op = self._build_controlled_operator(gate_matrix, qubits[0], qubits[1])
            else:
                raise ValueError(f"Unsupported gate arity: {len(qubits)}")
            self.state = full_op @ self.state

        return self.state

    # -- Measurement --

    def measure(self, n_shots: int = 10000) -> Dict[str, int]:
        """Measure all qubits in the computational basis.

        Why: We sample from the probability distribution |amplitude_k|²
        using numpy's random.choice, then format results as bitstrings.
        """
        probs = np.abs(self.state) ** 2
        outcomes = np.random.choice(self.dim, size=n_shots, p=probs)
        counts: Dict[str, int] = {}
        for idx in range(self.dim):
            label = format(idx, f'0{self.n_qubits}b')
            count = int(np.sum(outcomes == idx))
            if count > 0:
                counts[label] = count
        return dict(sorted(counts.items()))

    def probabilities(self) -> Dict[str, float]:
        """Return exact measurement probabilities."""
        probs = np.abs(self.state) ** 2
        result = {}
        for idx in range(self.dim):
            label = format(idx, f'0{self.n_qubits}b')
            if probs[idx] > 1e-15:
                result[label] = float(probs[idx])
        return dict(sorted(result.items()))

    # -- Analysis --

    def depth(self) -> int:
        """Calculate circuit depth (maximum number of gates on any single qubit).

        Why: Circuit depth determines execution time on hardware (gates on
        different qubits can execute in parallel).  Lower depth = faster,
        less decoherence.
        """
        qubit_depth = [0] * self.n_qubits
        for name, qubits, _ in self.gates:
            # All qubits involved get the same layer
            max_d = max(qubit_depth[q] for q in qubits)
            for q in qubits:
                qubit_depth[q] = max_d + 1
        return max(qubit_depth) if qubit_depth else 0

    def gate_count(self) -> Dict[str, int]:
        """Count each type of gate."""
        counts: Dict[str, int] = {}
        for name, _, _ in self.gates:
            base_name = name.split("(")[0]
            counts[base_name] = counts.get(base_name, 0) + 1
        return counts

    def __repr__(self) -> str:
        lines = [f"QuantumCircuit({self.n_qubits} qubits, {len(self.gates)} gates, depth={self.depth()})"]
        for name, qubits, _ in self.gates:
            q_str = ",".join(str(q) for q in qubits)
            lines.append(f"  {name} @ q[{q_str}]")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

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
    return " ".join(terms)


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_basic_circuit():
    """Build and run a simple circuit."""
    print("=" * 60)
    print("DEMO 1: Basic Circuit — Hadamard on Single Qubit")
    print("=" * 60)

    qc = QuantumCircuit(1)
    qc.h(0)
    state = qc.run()
    print(f"\n  Circuit: H on qubit 0")
    print(f"  State: {format_state(state, 1)}")
    print(f"  Probabilities: {qc.probabilities()}")
    print(f"  Depth: {qc.depth()}")


def demo_bell_state():
    """Create a Bell state with H + CNOT."""
    print("\n" + "=" * 60)
    print("DEMO 2: Bell State |Φ+⟩ = (|00⟩ + |11⟩)/√2")
    print("=" * 60)

    # Why: The Bell state is the simplest example of entanglement.
    # H creates superposition on qubit 0, then CNOT copies it to qubit 1,
    # producing a state that cannot be written as a tensor product.
    qc = QuantumCircuit(2)
    qc.h(0).cx(0, 1)
    state = qc.run()

    print(f"\n  Circuit:")
    print(f"  {qc}")
    print(f"\n  State: {format_state(state, 2)}")
    print(f"  Probabilities: {qc.probabilities()}")

    counts = qc.measure(10000)
    print(f"  Measurement (10000 shots): {counts}")
    print(f"  → Only |00⟩ and |11⟩ outcomes — perfect correlation!")


def demo_ghz_state():
    """Create a GHZ state on 3 qubits."""
    print("\n" + "=" * 60)
    print("DEMO 3: GHZ State (|000⟩ + |111⟩)/√2")
    print("=" * 60)

    # Why: GHZ extends Bell entanglement to 3+ qubits. It's the maximally
    # entangled state — measuring ANY qubit collapses ALL others. This makes
    # GHZ states useful for multiparty quantum protocols and error detection.
    qc = QuantumCircuit(3)
    qc.h(0).cx(0, 1).cx(0, 2)
    state = qc.run()

    print(f"\n  Circuit:")
    print(f"  {qc}")
    print(f"\n  State: {format_state(state, 3)}")
    print(f"  Probabilities: {qc.probabilities()}")

    counts = qc.measure(10000)
    print(f"  Measurement: {counts}")
    print(f"  → Only |000⟩ and |111⟩ — 3-qubit entanglement!")


def demo_non_adjacent_cnot():
    """Demonstrate CNOT between non-adjacent qubits."""
    print("\n" + "=" * 60)
    print("DEMO 4: Non-adjacent CNOT (qubit 0 controls qubit 2)")
    print("=" * 60)

    # Why: On real hardware, CNOT is typically only between adjacent qubits.
    # Our simulator handles arbitrary qubit pairs via projector construction,
    # demonstrating the abstract circuit model before hardware constraints.
    qc = QuantumCircuit(3)
    qc.x(0)         # Set qubit 0 to |1⟩
    qc.cx(0, 2)     # CNOT: qubit 0 → qubit 2 (non-adjacent)
    state = qc.run()

    print(f"\n  Circuit: X(0), CNOT(0→2)")
    print(f"  State: {format_state(state, 3)}")
    print(f"  Expected: |101⟩ (qubit 0=1, qubit 1=0, qubit 2=flipped to 1)")
    print(f"  Probabilities: {qc.probabilities()}")


def demo_multi_gate_circuit():
    """A more complex circuit with various gates."""
    print("\n" + "=" * 60)
    print("DEMO 5: Complex Circuit with Analysis")
    print("=" * 60)

    qc = QuantumCircuit(3)
    # Layer 1: superposition
    qc.h(0).h(1).h(2)
    # Layer 2: entangle
    qc.cx(0, 1).cx(1, 2)
    # Layer 3: phases
    qc.t(0).s(1).z(2)

    state = qc.run()

    print(f"\n  Circuit:")
    print(f"  {qc}")
    print(f"\n  Gate count: {qc.gate_count()}")
    print(f"  Circuit depth: {qc.depth()}")
    print(f"\n  State vector ({2**3} amplitudes):")
    for idx in range(2**3):
        amp = state[idx]
        label = format(idx, '03b')
        prob = np.abs(amp)**2
        if prob > 1e-15:
            print(f"    |{label}⟩: amplitude = {amp:+.4f}, P = {prob:.4f}")


def demo_superposition_interference():
    """Show constructive and destructive interference."""
    print("\n" + "=" * 60)
    print("DEMO 6: Quantum Interference")
    print("=" * 60)

    # Why: Two Hadamards cancel (HH = I), demonstrating destructive interference
    # of the |1⟩ amplitudes.  This is the fundamental mechanism behind
    # algorithms like Deutsch-Jozsa and Grover's search.
    qc1 = QuantumCircuit(1)
    qc1.h(0).h(0)
    state1 = qc1.run()
    print(f"\n  H·H|0⟩ = {format_state(state1, 1)}")
    print(f"  → Back to |0⟩ (destructive interference of |1⟩ amplitudes)")

    # X between two Hadamards: HXH|0⟩ = Z|0⟩ = |0⟩ (but with phase on |1⟩)
    qc2 = QuantumCircuit(1)
    qc2.h(0).x(0).h(0)
    state2 = qc2.run()
    print(f"\n  H·X·H|0⟩ = {format_state(state2, 1)}")
    print(f"  → HXH = Z, and Z|0⟩ = |0⟩")

    # Z between two Hadamards: HZH|0⟩ = X|0⟩ = |1⟩
    qc3 = QuantumCircuit(1)
    qc3.h(0).z(0).h(0)
    state3 = qc3.run()
    print(f"\n  H·Z·H|0⟩ = {format_state(state3, 1)}")
    print(f"  → HZH = X, and X|0⟩ = |1⟩")


def demo_measurement_statistics():
    """Run measurement on a 3-qubit state and show statistics."""
    print("\n" + "=" * 60)
    print("DEMO 7: Measurement Statistics")
    print("=" * 60)

    qc = QuantumCircuit(3)
    qc.h(0).cx(0, 1).cx(0, 2)  # GHZ
    qc.run()

    n_shots = 20000
    counts = qc.measure(n_shots)
    probs = qc.probabilities()

    print(f"\n  GHZ state measurement ({n_shots} shots):")
    print(f"  {'Outcome':<10} {'Theory':>10} {'Experiment':>10} {'Counts':>8}")
    print(f"  {'─' * 42}")

    # Why: Only |000⟩ and |111⟩ should appear with equal probability.
    # Any other outcome indicates a bug in the simulator.
    for label in [format(i, '03b') for i in range(8)]:
        theory = probs.get(label, 0.0)
        count = counts.get(label, 0)
        expt = count / n_shots
        if theory > 0 or count > 0:
            print(f"  |{label}⟩     {theory:>10.4f} {expt:>10.4f} {count:>8d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Quantum Computing — 03: Quantum Circuits              ║")
    print("╚══════════════════════════════════════════════════════════╝")

    np.random.seed(2026)

    demo_basic_circuit()
    demo_bell_state()
    demo_ghz_state()
    demo_non_adjacent_cnot()
    demo_multi_gate_circuit()
    demo_superposition_interference()
    demo_measurement_statistics()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
