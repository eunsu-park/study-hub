# Lesson 4: Quantum Circuits

[<- Previous: Quantum Gates](03_Quantum_Gates.md) | [Next: Entanglement and Bell States ->](05_Entanglement_and_Bell_States.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Read and interpret quantum circuit diagrams, identifying wires (qubits), gates, and measurement operations
2. Explain the circuit model of quantum computation and how it differs from classical circuits
3. Define circuit depth and width and explain their significance for algorithm complexity
4. Translate a quantum circuit diagram into a matrix expression and compute the output state
5. Handle the classical-quantum boundary where measurement converts quantum states to classical bits
6. Build a simple quantum circuit simulator in Python using matrix multiplication
7. Analyze the computational cost of classically simulating quantum circuits

---

The quantum circuit model is the standard framework for describing quantum computations. Just as classical algorithms can be expressed as sequences of logic gates wired together, quantum algorithms are expressed as sequences of quantum gates applied to qubits. This model provides a clean abstraction layer between the mathematical description of quantum operations (unitary matrices, as studied in [Lesson 3](03_Quantum_Gates.md)) and their physical implementation on hardware.

Understanding how to read, write, and simulate quantum circuits is essential for studying every quantum algorithm from here forward. In this lesson, we develop both the conceptual framework and a practical Python simulator that we will use in subsequent lessons on the Deutsch-Jozsa algorithm ([Lesson 7](07_Deutsch_Jozsa_Algorithm.md)) and Grover's search ([Lesson 8](08_Grovers_Search.md)).

> **Analogy:** A quantum circuit is like a musical score -- each line (wire) represents a qubit, and the gates are notes that must be played in sequence from left to right. Just as a musician reads a score to know which notes to play and when, a quantum computer reads a circuit to know which operations to perform and in what order. And just as the beauty of music comes from the interplay of multiple voices, the power of quantum circuits comes from the interaction between multiple qubits.

## Table of Contents

1. [The Circuit Model](#1-the-circuit-model)
2. [Circuit Notation](#2-circuit-notation)
3. [Circuit Depth and Width](#3-circuit-depth-and-width)
4. [Matrix Representation of Circuits](#4-matrix-representation-of-circuits)
5. [The Classical-Quantum Boundary](#5-the-classical-quantum-boundary)
6. [Building a Circuit Simulator](#6-building-a-circuit-simulator)
7. [Simulation Complexity](#7-simulation-complexity)
8. [Exercises](#8-exercises)

---

## 1. The Circuit Model

### 1.1 Components

A quantum circuit has three types of components:

1. **Wires (horizontal lines)**: Each wire represents one qubit. Time flows from left to right along the wire. The qubit persists throughout the circuit; it is not consumed like signals in some classical gate models.

2. **Gates (boxes on wires)**: Each gate is a unitary operation applied to one or more qubits. Single-qubit gates sit on one wire; multi-qubit gates span multiple wires.

3. **Measurements (meter symbols)**: Measurements convert quantum information to classical information. They are typically placed at the end of the circuit.

### 1.2 Execution Model

A quantum circuit executes as follows:

1. **Initialization**: All qubits start in a known state, typically $|0\rangle$.
2. **Gate application**: Gates are applied from left to right. Gates that act on different qubits and appear in the same time step can be executed in parallel.
3. **Measurement**: At the end (or at intermediate points), qubits are measured to produce classical output bits.

### 1.3 Circuit vs Algorithm

A quantum circuit describes a *fixed* computation for a fixed input size. A quantum *algorithm* is a family of circuits -- one for each input size -- along with a classical description of how to construct the circuit for any given input.

For example, Grover's search on $N = 2^n$ elements is described by a family of circuits, one for each $n$. The circuit for $n = 10$ has 10 qubits and a specific sequence of gates; the circuit for $n = 20$ has 20 qubits and a different (larger) sequence.

---

## 2. Circuit Notation

### 2.1 Text-Based Circuit Diagrams

Since we are working in a text environment, we represent circuits using ASCII notation. Here is an example of a Bell state preparation circuit:

```
q0: ─[H]─●─
          │
q1: ──────X─
```

- `q0`, `q1`: qubit labels
- `[H]`: Hadamard gate on q0
- `●`: control qubit of CNOT
- `X`: target qubit of CNOT (NOT gate applied conditionally)
- `│`: vertical line connecting control to target

### 2.2 Common Gate Symbols

| Symbol | Gate | Description |
|--------|------|-------------|
| `[H]` | Hadamard | Creates/destroys superposition |
| `[X]` | Pauli-X | Bit flip |
| `[Z]` | Pauli-Z | Phase flip |
| `[S]` | S gate | $\pi/2$ phase |
| `[T]` | T gate | $\pi/4$ phase |
| `●─X` | CNOT | Controlled-NOT |
| `●─●` | CZ | Controlled-Z |
| `[M]` | Measurement | Collapses qubit to classical bit |
| `[Rz(θ)]` | Rotation | Parameterized gate |

### 2.3 Ordering Convention

**Important**: In our notation and code, qubit 0 is the *least significant bit*. For a 2-qubit system:

$$|q_1 q_0\rangle \quad \text{where } q_1 \text{ is the most significant bit}$$

The state vector is ordered as $|00\rangle, |01\rangle, |10\rangle, |11\rangle$ corresponding to indices 0, 1, 2, 3.

```python
import numpy as np

# Circuit notation: translating diagrams to operations

# Bell state preparation circuit:
# q0: -[H]-*-
#           |
# q1: ------X-

# Step 1: Initialize |00>
state = np.array([1, 0, 0, 0], dtype=complex)

# Step 2: Apply H to q0 (H tensor I on the full state space)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
I = np.eye(2, dtype=complex)

# Why kron(H, I) and not kron(I, H)?
# Because q0 is the FIRST qubit in the tensor product.
# Our convention: |q1 q0> means q0 is rightmost (least significant).
# But in the Kronecker product, the FIRST matrix acts on the FIRST qubit.
# So H on q0 = kron(I_q1, H_q0) if we label q1 as "first/left" and q0 as "second/right".
# HOWEVER, we follow the common CS convention where q0 is the LEAST significant bit,
# and the Kronecker product has q0 as the rightmost factor.
# So H on q0 = kron(I, H) in |q1, q0> ordering.

# Actually, let's be precise:
# State vector order: |00>, |01>, |10>, |11> where rightmost bit = q0
# To apply H to q0: kron(I, H)  (I acts on q1, H acts on q0)
# To apply H to q1: kron(H, I)  (H acts on q1, I acts on q0)

H_on_q0 = np.kron(I, H)  # H on qubit 0
state = H_on_q0 @ state
print("After H on q0:")
print(f"  State: {state}")
print(f"  = ({state[0]:.4f})|00> + ({state[1]:.4f})|01> + "
      f"({state[2]:.4f})|10> + ({state[3]:.4f})|11>")

# Step 3: Apply CNOT (control=q0, target=q1)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

state = CNOT @ state
print("\nAfter CNOT (control=q0, target=q1):")
print(f"  State: {state}")
print(f"  = ({state[0]:.4f})|00> + ({state[1]:.4f})|01> + "
      f"({state[2]:.4f})|10> + ({state[3]:.4f})|11>")
print(f"\nThis is the Bell state |Phi+> = (|00> + |11>)/sqrt(2)")
```

---

## 3. Circuit Depth and Width

### 3.1 Definitions

- **Width**: The number of qubits in the circuit. Determines the dimension of the state space ($2^{\text{width}}$).

- **Depth**: The number of time steps (layers) in the circuit, where gates in the same layer act on disjoint qubits and can execute in parallel. This is the critical metric for algorithm speed.

### 3.2 Example: Depth Counting

Consider the circuit:
```
q0: ─[H]─●─────[T]─
          │
q1: ─[H]─X──●──────
             │
q2: ─────────X──[H]─
```

- **Width**: 3 qubits
- **Layer 1**: H on q0, H on q1 (parallel -- different qubits)
- **Layer 2**: CNOT(q0, q1)
- **Layer 3**: CNOT(q1, q2)
- **Layer 4**: T on q0, H on q2 (parallel)
- **Depth**: 4

### 3.3 Why Depth Matters

Quantum states are fragile -- they lose coherence over time (decoherence). A circuit with depth $d$ must complete all $d$ layers before the qubits decohere. This makes circuit depth the most important complexity metric for near-term quantum computing.

**Gate count vs Depth**: A circuit might have many gates but shallow depth if most gates can be parallelized. Conversely, a circuit with few gates but sequential dependencies has high depth.

```python
import numpy as np

# Analyzing circuit structure

class CircuitAnalyzer:
    """
    Simple circuit structure analyzer.

    Why track layers instead of just gates? On real quantum hardware,
    the total execution time is determined by the DEPTH (number of
    sequential layers), not the total gate count. Gates on different
    qubits in the same layer run in parallel.
    """
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.gates = []  # List of (gate_name, qubit_indices)

    def add_gate(self, name, qubits):
        """Add a gate acting on specified qubits."""
        self.gates.append((name, tuple(qubits)))

    def compute_depth(self):
        """
        Compute circuit depth using a greedy layering algorithm.

        Why greedy? We assign each gate to the earliest possible layer
        (the first layer where none of its qubits are already occupied).
        This gives the minimum depth, assuming all gates take 1 time unit.
        """
        # Track the latest layer each qubit is involved in
        qubit_latest_layer = [-1] * self.n_qubits
        layers = []

        for name, qubits in self.gates:
            # This gate must go after all previous gates on any of its qubits
            min_layer = max(qubit_latest_layer[q] for q in qubits) + 1

            # Ensure layers list is long enough
            while len(layers) <= min_layer:
                layers.append([])

            layers[min_layer].append((name, qubits))

            # Update qubit usage
            for q in qubits:
                qubit_latest_layer[q] = min_layer

        return len(layers), layers

# Example: Bell state circuit
print("=== Bell State Circuit ===\n")
bell = CircuitAnalyzer(2)
bell.add_gate("H", [0])
bell.add_gate("CNOT", [0, 1])

depth, layers = bell.compute_depth()
print(f"Width: {bell.n_qubits} qubits")
print(f"Depth: {depth} layers")
print(f"Gate count: {len(bell.gates)}")
for i, layer in enumerate(layers):
    print(f"  Layer {i}: {layer}")

# Example: GHZ state circuit (3-qubit entangled state)
print("\n=== GHZ State Circuit ===\n")
ghz = CircuitAnalyzer(3)
ghz.add_gate("H", [0])
ghz.add_gate("CNOT", [0, 1])
ghz.add_gate("CNOT", [1, 2])

depth, layers = ghz.compute_depth()
print(f"Width: {ghz.n_qubits} qubits")
print(f"Depth: {depth} layers")
print(f"Gate count: {len(ghz.gates)}")
for i, layer in enumerate(layers):
    print(f"  Layer {i}: {layer}")

# Example: Circuit with parallelism
print("\n=== Circuit with Parallelism ===\n")
par = CircuitAnalyzer(4)
par.add_gate("H", [0])
par.add_gate("H", [1])
par.add_gate("H", [2])
par.add_gate("H", [3])
par.add_gate("CNOT", [0, 1])
par.add_gate("CNOT", [2, 3])
par.add_gate("CNOT", [1, 2])

depth, layers = par.compute_depth()
print(f"Width: {par.n_qubits} qubits")
print(f"Depth: {depth} layers")
print(f"Gate count: {len(par.gates)}")
for i, layer in enumerate(layers):
    print(f"  Layer {i}: {layer}")
print("\nNote: 4 H gates execute in parallel (depth 1, not 4)")
print("and 2 CNOTs on disjoint qubits also parallelize!")
```

---

## 4. Matrix Representation of Circuits

### 4.1 Sequential Gates = Matrix Multiplication

If a circuit applies gate $U_1$, then $U_2$, then $U_3$, the overall unitary is:

$$U_{\text{circuit}} = U_3 \cdot U_2 \cdot U_1$$

Note the reversed order: because matrix multiplication is applied right-to-left (the rightmost matrix acts first on the state vector).

The final state is:

$$|\psi_{\text{out}}\rangle = U_{\text{circuit}} |\psi_{\text{in}}\rangle = U_3 U_2 U_1 |\psi_{\text{in}}\rangle$$

### 4.2 Gates on Subsets of Qubits

When a gate $G$ acts on only some qubits in an $n$-qubit system, we must "pad" it with identity operations on the other qubits using tensor products:

- Gate on qubit 0 only (in a 3-qubit system): $I \otimes I \otimes G$
- Gate on qubit 1 only: $I \otimes G \otimes I$
- Two-qubit gate on qubits 0,1: $I \otimes G_{01}$

The general rule for qubit ordering $|q_{n-1} \cdots q_1 q_0\rangle$: the tensor product has the gate acting on the rightmost qubit ($q_0$) as the rightmost factor.

### 4.3 Example: Full Matrix for a 3-Qubit Circuit

Consider the circuit:
```
q0: ─[H]─●─
          │
q1: ──────X─
q2: ──────── (idle)
```

Step 1: $U_1 = I_{q2} \otimes I_{q1} \otimes H_{q0}$

Step 2: $U_2 = I_{q2} \otimes \text{CNOT}_{q0,q1}$

Overall: $U_{\text{circuit}} = U_2 \cdot U_1$

```python
import numpy as np

# Matrix representation of multi-qubit circuits

def gate_on_qubit(gate, target_qubit, n_qubits):
    """
    Create the full-system matrix for a single-qubit gate acting
    on a specific qubit in an n-qubit system.

    Why tensor products? Each qubit has its own 2D space. The full
    system lives in the tensor product of all these spaces. A gate
    on one qubit acts as identity on all others.
    """
    # Build up the full matrix using Kronecker products
    # Qubit ordering: |q_{n-1} ... q_1 q_0>
    # The tensor product factors are ordered from q_{n-1} (left) to q_0 (right)
    matrices = []
    for q in range(n_qubits - 1, -1, -1):  # From q_{n-1} down to q_0
        if q == target_qubit:
            matrices.append(gate)
        else:
            matrices.append(np.eye(2, dtype=complex))

    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)

    return result

# Standard gates
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)

# CNOT for 3 qubits (control=q0, target=q1, q2 idle)
# This is I_q2 tensor CNOT_{q1,q0}
CNOT_q0q1 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)
CNOT_3q = np.kron(I2, CNOT_q0q1)  # I on q2, CNOT on q1,q0

# Build the circuit: H on q0, then CNOT(q0, q1)
print("=== 3-Qubit Circuit: H(q0) then CNOT(q0,q1) ===\n")

# Step 1: H on q0
U1 = gate_on_qubit(H, 0, 3)
print(f"U1 (H on q0) shape: {U1.shape}")

# Step 2: CNOT on q0, q1
U2 = CNOT_3q
print(f"U2 (CNOT q0,q1) shape: {U2.shape}")

# Full circuit
U_circuit = U2 @ U1
print(f"U_circuit shape: {U_circuit.shape}")

# Apply to |000>
state_in = np.zeros(8, dtype=complex)
state_in[0] = 1  # |000>
state_out = U_circuit @ state_in

print(f"\nInput:  |000>")
print(f"Output: ")
for i in range(8):
    if abs(state_out[i]) > 1e-10:
        label = format(i, '03b')
        print(f"  |{label}>: {state_out[i]:.4f} (P = {abs(state_out[i])**2:.4f})")

print("\nResult: (|000> + |110>)/sqrt(2)")
print("q2 remains |0>, while q0 and q1 form a Bell pair.")
```

---

## 5. The Classical-Quantum Boundary

### 5.1 Measurement in Circuits

Measurement converts a qubit from quantum to classical. In a circuit, measurement is typically represented by a meter symbol and produces a classical bit.

**Key rules**:
1. Measurement is irreversible (collapses the superposition)
2. After measurement, the qubit is in a definite state ($|0\rangle$ or $|1\rangle$)
3. Further quantum gates on a measured qubit are still possible but act on the collapsed state
4. Measurement outcomes are probabilistic

### 5.2 Deferred Measurement Principle

An important theorem: **measurements can always be deferred to the end of the circuit** without changing the output probability distribution. This means we can treat a quantum circuit as:

1. A unitary transformation $U$ on all qubits
2. Followed by measurement of all qubits

Even if the original circuit has intermediate measurements, we can simulate it by:
- Replacing each intermediate measurement with a CNOT to a fresh "ancilla" qubit
- Measuring all ancillas at the end

This is both a theoretical simplification and a practical tool for circuit analysis.

### 5.3 Classical Control

Some circuits use measurement outcomes to control subsequent gates (classical feedback). For example:

```
q0: ─[H]─[M]═══╗
                ║
q1: ────────[X if M=1]─
```

Here, the X gate on q1 is applied only if q0 was measured as 1. This is called **classically controlled** operation and is essential for protocols like quantum teleportation.

```python
import numpy as np

# Simulating measurement in quantum circuits

def simulate_measurement(state, qubit, n_qubits):
    """
    Simulate measuring one qubit in the computational basis.

    Why is this non-trivial? Measuring one qubit in a multi-qubit system
    requires:
    1. Computing probabilities of 0 and 1 for that qubit
    2. Randomly choosing an outcome
    3. Collapsing the full state to the post-measurement state
    4. Re-normalizing

    Returns: (outcome, post_measurement_state)
    """
    dim = 2**n_qubits

    # Compute probability of measuring qubit as |0>
    # Sum |amplitude|^2 over all basis states where the target qubit is 0
    prob_0 = 0.0
    for i in range(dim):
        # Check if bit 'qubit' of index i is 0
        if (i >> qubit) & 1 == 0:
            prob_0 += abs(state[i])**2

    prob_1 = 1.0 - prob_0

    # Random measurement outcome
    outcome = 0 if np.random.random() < prob_0 else 1

    # Collapse: zero out amplitudes inconsistent with outcome
    new_state = np.zeros(dim, dtype=complex)
    for i in range(dim):
        bit_value = (i >> qubit) & 1
        if bit_value == outcome:
            new_state[i] = state[i]

    # Re-normalize
    norm = np.linalg.norm(new_state)
    if norm > 1e-15:
        new_state = new_state / norm

    return outcome, new_state

# Example: Measure one qubit of a Bell state
np.random.seed(42)

print("=== Measuring a Bell State ===\n")

# Bell state: (|00> + |11>)/sqrt(2)
bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
print(f"Initial state: (|00> + |11>)/sqrt(2)")

# Measure qubit 0
print("\nMeasuring qubit 0 ten times:")
for trial in range(10):
    outcome, post_state = simulate_measurement(bell.copy(), 0, 2)
    # Format post-measurement state
    components = []
    for i in range(4):
        if abs(post_state[i]) > 1e-10:
            components.append(f"|{format(i, '02b')}>")
    print(f"  Trial {trial}: q0={outcome}, collapsed to {', '.join(components)}")

# Show correlation: if we measure q0=0, then q1 must be 0
# if we measure q0=1, then q1 must be 1
print("\n--- Demonstrating Entanglement Correlation ---")
print("In a Bell state, measuring one qubit determines the other:\n")

outcomes = {0: 0, 1: 0}
for _ in range(10000):
    outcome0, post = simulate_measurement(bell.copy(), 0, 2)
    outcome1, _ = simulate_measurement(post, 1, 2)
    assert outcome0 == outcome1, "Bell state qubits must agree!"
    outcomes[outcome0] += 1

print(f"  10000 trials: q0=q1=0 occurred {outcomes[0]} times, "
      f"q0=q1=1 occurred {outcomes[1]} times")
print(f"  Outcomes ALWAYS agree (entanglement!)")
```

---

## 6. Building a Circuit Simulator

Let us build a simple but complete quantum circuit simulator that supports single-qubit gates, CNOT, and measurement.

```python
import numpy as np

class QuantumCircuit:
    """
    A simple quantum circuit simulator using state vector simulation.

    Why state vector simulation? It tracks the full quantum state
    (all 2^n amplitudes) and applies gates via matrix multiplication.
    This is the most straightforward simulation method and gives
    exact results, but scales exponentially with qubit count.

    For small circuits (< ~25 qubits), this is perfectly practical.
    """

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        # Initialize to |00...0>
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0
        self.operations = []  # Log of operations for display

    def _full_gate_matrix(self, gate_matrix, target_qubits):
        """
        Build the full 2^n x 2^n matrix for a gate acting on specific qubits.

        Why build the full matrix? For small systems, explicit matrix construction
        is clear and correct. Production simulators use more efficient approaches
        (e.g., applying gates directly to the state vector without building the
        full matrix), but this is ideal for learning.
        """
        if len(target_qubits) == 1:
            # Single-qubit gate: tensor product with identities
            q = target_qubits[0]
            matrices = []
            for i in range(self.n_qubits - 1, -1, -1):
                if i == q:
                    matrices.append(gate_matrix)
                else:
                    matrices.append(np.eye(2, dtype=complex))
            result = matrices[0]
            for m in matrices[1:]:
                result = np.kron(result, m)
            return result
        else:
            # Multi-qubit gate: more complex embedding
            # For now, handle specific cases
            return gate_matrix  # Assumes gate_matrix is already full-size

    def h(self, qubit):
        """Apply Hadamard gate to a qubit."""
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        full = self._full_gate_matrix(H, [qubit])
        self.state = full @ self.state
        self.operations.append(f"H(q{qubit})")

    def x(self, qubit):
        """Apply Pauli-X gate to a qubit."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        full = self._full_gate_matrix(X, [qubit])
        self.state = full @ self.state
        self.operations.append(f"X(q{qubit})")

    def z(self, qubit):
        """Apply Pauli-Z gate to a qubit."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        full = self._full_gate_matrix(Z, [qubit])
        self.state = full @ self.state
        self.operations.append(f"Z(q{qubit})")

    def rz(self, qubit, theta):
        """Apply Rz(theta) gate to a qubit."""
        Rz = np.array([
            [np.exp(-1j*theta/2), 0],
            [0, np.exp(1j*theta/2)]
        ], dtype=complex)
        full = self._full_gate_matrix(Rz, [qubit])
        self.state = full @ self.state
        self.operations.append(f"Rz({theta:.3f})(q{qubit})")

    def cnot(self, control, target):
        """
        Apply CNOT gate with specified control and target qubits.

        Why build CNOT this way? The CNOT matrix depends on which qubits
        are control and target. We construct it by iterating over all basis
        states: for each state, if the control bit is 1, flip the target bit.
        """
        full = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(self.dim):
            control_bit = (i >> control) & 1
            if control_bit == 1:
                # Flip the target bit
                j = i ^ (1 << target)
                full[j, i] = 1
            else:
                full[i, i] = 1
        self.state = full @ self.state
        self.operations.append(f"CNOT(q{control}->q{target})")

    def measure_all(self, shots=1024):
        """
        Simulate measuring all qubits, repeated 'shots' times.

        Why multiple shots? A single measurement gives one random outcome.
        To estimate the probability distribution, we need many repetitions.
        """
        probabilities = np.abs(self.state)**2
        outcomes = np.random.choice(self.dim, size=shots, p=probabilities)
        counts = {}
        for outcome in outcomes:
            label = format(outcome, f'0{self.n_qubits}b')
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))

    def get_statevector(self):
        """Return the current state vector."""
        return self.state.copy()

    def get_probabilities(self):
        """Return measurement probabilities for each basis state."""
        probs = np.abs(self.state)**2
        result = {}
        for i in range(self.dim):
            if probs[i] > 1e-10:
                label = format(i, f'0{self.n_qubits}b')
                result[label] = probs[i]
        return result

    def __repr__(self):
        ops = " -> ".join(self.operations) if self.operations else "(empty)"
        return f"QuantumCircuit({self.n_qubits}q): {ops}"


# === Demonstration ===

# Bell state preparation
print("=== Bell State Circuit ===\n")
qc = QuantumCircuit(2)
qc.h(0)
qc.cnot(0, 1)
print(f"Circuit: {qc}")
print(f"State vector: {qc.get_statevector()}")
print(f"Probabilities: {qc.get_probabilities()}")
np.random.seed(42)
print(f"Measurement (1000 shots): {qc.measure_all(1000)}")

# GHZ state: (|000> + |111>)/sqrt(2)
print("\n=== GHZ State Circuit ===\n")
qc = QuantumCircuit(3)
qc.h(0)
qc.cnot(0, 1)
qc.cnot(1, 2)
print(f"Circuit: {qc}")
print(f"Probabilities: {qc.get_probabilities()}")
print(f"Measurement (1000 shots): {qc.measure_all(1000)}")

# Superposition of all states
print("\n=== Uniform Superposition (3 qubits) ===\n")
qc = QuantumCircuit(3)
for i in range(3):
    qc.h(i)
print(f"Circuit: {qc}")
probs = qc.get_probabilities()
print(f"Probabilities (should all be 1/8 = 0.125):")
for label, prob in probs.items():
    print(f"  |{label}>: {prob:.4f}")
```

Here is a more complex example using the simulator:

```python
import numpy as np

# Using the QuantumCircuit class defined above
# (In practice, you would import or define it in the same file)

# Demonstrate: creating a specific superposition
# Target: |psi> = (|00> + |01> + |10>)/sqrt(3)
# This requires more careful gate selection

print("=== Custom State Preparation ===\n")
print("Goal: prepare (|00> + |01> + |10>)/sqrt(3)")
print("Strategy: Use Ry rotation to get amplitude 1/sqrt(3) on |0> and")
print("sqrt(2/3) on |1>, then conditionally create superposition.\n")

# This demonstrates that not every state has a simple circuit!
# For now, let's verify the simulator on known circuits.

# Verify: X gate followed by CNOT creates |11> from |00>
qc = QuantumCircuit(2)
qc.x(0)           # |00> -> |01>
qc.cnot(0, 1)     # |01> -> |11>
print(f"X(q0) then CNOT(q0->q1):")
print(f"  Expected: |11>")
print(f"  Got: {qc.get_probabilities()}")

# Verify: H on both qubits creates uniform superposition
qc = QuantumCircuit(2)
qc.h(0)
qc.h(1)
print(f"\nH(q0) then H(q1):")
print(f"  Expected: uniform over |00>, |01>, |10>, |11>")
print(f"  Got: {qc.get_probabilities()}")
```

---

## 7. Simulation Complexity

### 7.1 Classical Simulation Cost

Simulating a quantum circuit on a classical computer requires tracking $2^n$ complex amplitudes for $n$ qubits. Each gate application involves matrix-vector multiplication:

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Store state | -- | $O(2^n)$ |
| Apply 1-qubit gate | $O(2^n)$ | $O(2^n)$ |
| Apply 2-qubit gate | $O(4^n)$ naive, $O(2^n)$ smart | $O(2^n)$ |
| Measurement | $O(2^n)$ | $O(2^n)$ |

### 7.2 The Exponential Wall

The exponential space requirement is the fundamental barrier:

| Qubits | State vector size | RAM needed |
|:---:|:---:|:---:|
| 20 | ~$10^6$ | 16 MB |
| 30 | ~$10^9$ | 16 GB |
| 40 | ~$10^{12}$ | 16 TB |
| 50 | ~$10^{15}$ | 16 PB |

Beyond ~45-50 qubits, full state vector simulation becomes impractical on any classical computer. This is the regime where quantum computers potentially offer advantage.

### 7.3 Efficient Simulation of Special Cases

Not all quantum circuits are hard to simulate classically:

- **Clifford circuits** (only H, S, CNOT, Pauli gates): Can be simulated in polynomial time using the Gottesman-Knill theorem. This is why the T gate is essential for universality.
- **Low-entanglement circuits**: Tensor network methods can efficiently simulate circuits that generate limited entanglement.
- **Shallow circuits**: Constant-depth circuits on 2D lattices can sometimes be simulated efficiently.

```python
import numpy as np
import time

# Benchmarking simulation cost

def benchmark_simulation(n_qubits, n_gates=10):
    """
    Measure the time to simulate a circuit of given size.

    Why benchmark? Understanding the exponential cost of simulation
    is crucial for appreciating why quantum computers could be useful.
    A circuit that takes seconds to simulate at 20 qubits would take
    longer than the age of the universe at 300 qubits.
    """
    dim = 2**n_qubits
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0

    # Create a random single-qubit gate
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    start = time.time()
    for _ in range(n_gates):
        # Apply H to qubit 0 (build full matrix)
        # In a smarter simulator, we would avoid building the full matrix
        matrices = [H] + [np.eye(2, dtype=complex)] * (n_qubits - 1)
        full_gate = matrices[0]
        for m in matrices[1:]:
            full_gate = np.kron(full_gate, m)
        state = full_gate @ state
    elapsed = time.time() - start

    return elapsed, dim

print("=== Simulation Cost Scaling ===\n")
print(f"{'Qubits':>8} {'Dim':>12} {'Time (s)':>12} {'Time/gate (ms)':>16}")
print("-" * 52)

for n in range(4, 15):
    try:
        elapsed, dim = benchmark_simulation(n, n_gates=5)
        print(f"{n:>8} {dim:>12,} {elapsed:>12.4f} {elapsed/5*1000:>16.2f}")
    except MemoryError:
        print(f"{n:>8} {'(memory limit)':>12}")
        break

print("\nObserve: time roughly quadruples with each additional qubit (exponential growth).")
print("This is why quantum computers are needed for large quantum circuits!")
```

---

## 8. Exercises

### Exercise 1: Circuit Tracing

Trace through the following circuit step by step, writing the state after each gate:

```
q0: ─[H]─●─[H]─
          │
q1: ──────X─────
```

Starting from $|00\rangle$, what is the final state? Is this the same as the Bell state from Section 2? Why or why not?

### Exercise 2: Circuit Matrix

Compute the full $4 \times 4$ unitary matrix for this circuit:

```
q0: ─[X]─●─
          │
q1: ─[H]─X─
```

Verify your answer by applying it to $|00\rangle$ and checking the result against step-by-step tracing.

### Exercise 3: Depth Optimization

The following circuit has depth 4:

```
q0: ─[H]─[T]─●─────
              │
q1: ─[H]─────X─[T]─
```

Can you rearrange the gates to achieve the same output with depth 3? What is the minimum possible depth? (Hint: think about which gates can be parallelized.)

### Exercise 4: Simulator Extension

Extend the `QuantumCircuit` simulator to support:
a) The S gate and T gate
b) The CZ gate (controlled-Z)
c) A `barrier()` method that does nothing but marks a visual separation in the circuit log

Test your extensions by constructing a circuit that creates the state $\frac{1}{2}(|00\rangle + i|01\rangle + |10\rangle + i|11\rangle)$.

### Exercise 5: Simulation Limits

a) What is the largest circuit (number of qubits) you can simulate on your machine using the `QuantumCircuit` simulator? Empirically determine this by trying increasing qubit counts.
b) How does the simulation time scale with the number of gates for a fixed number of qubits? Run experiments and plot the results.
c) Explain why the Gottesman-Knill theorem (efficient simulation of Clifford circuits) does not mean quantum computing is useless. What is the missing ingredient?

---

[<- Previous: Quantum Gates](03_Quantum_Gates.md) | [Next: Entanglement and Bell States ->](05_Entanglement_and_Bell_States.md)
