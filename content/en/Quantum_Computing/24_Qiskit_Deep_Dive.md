# Lesson 24: Qiskit Deep Dive

[← Previous: Quantum Networking](23_Quantum_Networking.md) | [Next: Capstone Quantum Application →](25_Capstone_Quantum_Application.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Build and manipulate quantum circuits using the Qiskit SDK (QuantumCircuit, gates, measurements)
2. Explain the transpilation pipeline: optimization levels, basis gate decomposition, routing
3. Construct realistic noise models using Qiskit Aer for simulation
4. Execute circuits on the Aer simulator with configurable noise profiles
5. Apply error mitigation techniques (measurement calibration, ZNE, twirled readout)
6. Design and analyze parameterized circuits for variational algorithms in Qiskit
7. Write complete quantum programs using Qiskit patterns: map, transpile, execute, post-process

---

Qiskit (Quantum Information Science Kit) is the most widely used open-source quantum computing SDK, developed by IBM. It provides tools for building quantum circuits, simulating them classically, running them on real quantum hardware via IBM Quantum, and analyzing the results. This lesson takes a deep dive into advanced Qiskit usage: the transpilation pipeline that converts abstract circuits into hardware-executable instructions, the Aer simulator with realistic noise models, and the error mitigation techniques that extract useful results from noisy quantum devices.

Rather than reproducing the Qiskit documentation, this lesson focuses on the **conceptual understanding** behind each tool and provides Python-based demonstrations using NumPy simulations that mirror Qiskit's functionality. This approach ensures the material remains useful even without a Qiskit installation, while teaching the principles that apply to any quantum computing framework.

> **Analogy:** Qiskit is to quantum computing what GCC is to classical computing — a toolchain that takes a high-level description (quantum circuit) and compiles it into something a specific machine can execute (transpiled circuit with hardware-native gates). Just as understanding compiler optimization is essential for writing efficient classical code, understanding transpilation is essential for writing efficient quantum programs.

## Table of Contents

1. [Qiskit Architecture Overview](#1-qiskit-architecture-overview)
2. [Circuit Construction](#2-circuit-construction)
3. [Transpilation Pipeline](#3-transpilation-pipeline)
4. [Noise Models](#4-noise-models)
5. [Aer Simulator](#5-aer-simulator)
6. [Real Device Execution](#6-real-device-execution)
7. [Error Mitigation](#7-error-mitigation)
8. [Parameterized Circuits and VQE](#8-parameterized-circuits-and-vqe)
9. [Python Implementation](#9-python-implementation)
10. [Exercises](#10-exercises)

---

## 1. Qiskit Architecture Overview

### 1.1 Qiskit Components (2025)

| Component | Purpose |
|-----------|---------|
| **qiskit** (Terra) | Core: circuits, transpilation, visualization |
| **qiskit-aer** | High-performance simulators with noise models |
| **qiskit-ibm-runtime** | Interface to IBM Quantum hardware |
| **qiskit-algorithms** | Quantum algorithms (VQE, QAOA, Grover, etc.) |
| **qiskit-nature** | Chemistry and physics applications |
| **qiskit-machine-learning** | Quantum ML models |

### 1.2 The Qiskit Pattern

The recommended workflow for any quantum computation:

```
1. Map:        Problem → Quantum circuit
2. Transpile:  Abstract circuit → Hardware-compatible circuit
3. Execute:    Run on simulator/hardware, collect results
4. Post-process: Analyze results, error mitigate, extract answers
```

### 1.3 Circuit Representation

Qiskit represents circuits as directed acyclic graphs (DAGs):
- **Nodes**: Gates, measurements, barriers
- **Edges**: Qubit/classical bit wires
- **Topological ordering**: Gates that can execute in parallel are at the same DAG level

This DAG representation enables efficient optimization during transpilation.

---

## 2. Circuit Construction

### 2.1 Basic Gates

| Gate | Qiskit name | Matrix | Qubits |
|------|-------------|--------|--------|
| Pauli X | `x(q)` | $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ | 1 |
| Hadamard | `h(q)` | $\frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$ | 1 |
| CNOT | `cx(c,t)` | Controlled-X | 2 |
| Phase | `p(theta,q)` | $\begin{pmatrix}1&0\\0&e^{i\theta}\end{pmatrix}$ | 1 |
| Rotation | `ry(theta,q)` | $R_y(\theta)$ | 1 |
| Toffoli | `ccx(c1,c2,t)` | Controlled-controlled-X | 3 |

### 2.2 Circuit Depth and Width

- **Width**: Number of qubits
- **Depth**: Longest path through the circuit (number of sequential gate layers)
- **Size**: Total number of gates
- **CNOT count**: Number of two-qubit gates (dominant error source)

Optimization goal: minimize CNOT count and depth while preserving the circuit's unitary.

### 2.3 Parameterized Circuits

Parameterized circuits use symbolic parameters that are bound later:

```python
# Qiskit style (conceptual)
from qiskit.circuit import Parameter
theta = Parameter('theta')
qc.ry(theta, 0)
# Bind later: qc.assign_parameters({theta: 0.5})
```

This is essential for variational algorithms where parameters are optimized classically.

---

## 3. Transpilation Pipeline

### 3.1 Why Transpile?

The abstract circuit uses ideal gates on any qubit pair. Real hardware has constraints:
- **Basis gates**: Only certain gates are natively supported (e.g., $\{CX, I, R_z, \sqrt{X}, X\}$ for IBM devices)
- **Connectivity**: Not all qubit pairs can interact directly (e.g., heavy-hex topology)
- **Gate errors**: Different qubits and connections have different error rates

### 3.2 Transpilation Stages

| Stage | Purpose | Key operations |
|-------|---------|---------------|
| **Unrolling** | Decompose custom gates into basis gates | $\text{Toffoli} \to 6 \text{ CNOT} + \text{singles}$ |
| **Routing** | Map logical qubits to physical qubits | Insert SWAP gates for non-adjacent interactions |
| **Optimization** | Reduce gate count and depth | Gate cancellation, commutation, resynthesis |
| **Scheduling** | Determine execution timing | ALAP/ASAP scheduling, dynamical decoupling |

### 3.3 Optimization Levels

| Level | Passes | CNOT reduction | Time |
|-------|--------|----------------|------|
| 0 | Unroll + route only | None | Fast |
| 1 | Light optimization | $\sim 10\%$ | Moderate |
| 2 | Medium optimization | $\sim 20\%$ | Moderate |
| 3 | Heavy optimization | $\sim 30\%$ | Slow |

### 3.4 Qubit Routing

For a circuit requiring a CNOT between non-adjacent qubits, SWAP gates must be inserted:

```
Logical circuit:    CX(q0, q3)
Hardware topology:  q0 - q1 - q2 - q3

Routed circuit:     SWAP(q2, q3) → CX(q0, q2) → SWAP(q2, q3)
```

Each SWAP costs 3 CNOTs, so routing can significantly increase the circuit cost.

**Routing algorithms**:
- **Trivial**: No routing (fails if connectivity insufficient)
- **Stochastic**: Random SWAP insertion with scoring
- **SABRE**: Heuristic forward-backward search (default in Qiskit)
- **Optimal**: Exact solution via SAT solver (exponential time)

### 3.5 Gate Decomposition Examples

| Original gate | Decomposition | CNOT count |
|--------------|---------------|-----------|
| SWAP | 3 CNOTs | 3 |
| Toffoli (CCX) | 6 CNOTs + singles | 6 |
| Controlled-Ry | 2 CNOTs + singles | 2 |
| $R_{zz}(\theta)$ | 2 CNOTs + $R_z$ | 2 |
| Arbitrary 2-qubit | Up to 3 CNOTs | $\leq 3$ |

---

## 4. Noise Models

### 4.1 Device Noise Model

A realistic noise model includes:

| Noise source | Model | Typical magnitude |
|-------------|-------|------------------|
| Single-qubit gate error | Depolarizing | $10^{-4}$ to $10^{-3}$ |
| Two-qubit gate error | Depolarizing | $10^{-3}$ to $10^{-2}$ |
| Readout error | Bit-flip | $10^{-3}$ to $10^{-1}$ |
| $T_1$ relaxation | Amplitude damping | $T_1 \sim 100$ $\mu$s |
| $T_2$ dephasing | Phase damping | $T_2 \sim 50$-$200$ $\mu$s |
| Crosstalk | Correlated error | Device-specific |

### 4.2 Building Noise Models in Aer

Qiskit Aer allows constructing noise models from:
1. **Backend properties**: Import the calibration data from a real IBM device
2. **Custom models**: Specify error rates for each gate and qubit
3. **Parameterized models**: Scale error rates for what-if analysis

### 4.3 Thermal Relaxation

For a gate of duration $t_g$:
- $T_1$ decay probability: $p_1 = 1 - e^{-t_g/T_1}$
- $T_2$ decay probability: $p_2 = 1 - e^{-t_g/T_2}$

The combined thermal relaxation channel applies amplitude damping ($T_1$) and phase damping ($T_2$) after each gate.

---

## 5. Aer Simulator

### 5.1 Simulation Methods

| Method | Description | Max qubits | Speed |
|--------|-------------|-----------|-------|
| `statevector` | Full state vector | $\sim 30$ | Exact |
| `density_matrix` | Full density matrix | $\sim 15$ | Noise-aware |
| `stabilizer` | Clifford circuits only | $> 1000$ | Very fast |
| `matrix_product_state` | MPS approximation | $\sim 50$ | Approximate |
| `automatic` | Chooses best method | Varies | Adaptive |

### 5.2 Shot-Based Simulation

Real hardware produces measurement outcomes (bit strings), not state vectors. Aer simulates this:

1. Compute the final state (state vector or density matrix)
2. Sample bit strings according to the probability distribution
3. Return counts (histogram of outcomes)

With noise, the probabilities are modified by the noise model before sampling.

### 5.3 GPU Acceleration

Aer supports GPU simulation via cuStateVec:
- Up to $\sim 33$ qubits on a single GPU (32 GB)
- Multi-GPU for larger systems
- $100\times$ to $1000\times$ speedup over CPU for 25+ qubits

---

## 6. Real Device Execution

### 6.1 IBM Quantum Access

IBM provides free and paid access to quantum devices through the IBM Quantum Platform:

- **Open plan**: Free access to 127-qubit devices, limited monthly execution time
- **Premium plan**: Dedicated access, faster queues, larger systems

### 6.2 Execution Pipeline

```
1. Circuit → Transpile (target device) → ISA circuit
2. ISA circuit → Submit to IBM Runtime
3. Runtime → Execute on device → Raw results
4. Raw results → Error mitigation → Final results
```

### 6.3 Dynamic Circuits

Modern IBM devices support **dynamic circuits**: mid-circuit measurements and classical control flow:

```python
# Conditional reset
qc.measure(0, 0)
with qc.if_test((0, 1)):
    qc.x(0)  # Reset to |0> if measured |1>
```

This enables:
- Quantum error correction (measure syndrome, apply corrections)
- Repeat-until-success protocols
- Adaptive measurement schemes

### 6.4 Device Selection

Choose the best device based on:
- **Gate error rates**: Lower is better (check calibration data)
- **Queue time**: Choose less busy devices for faster turnaround
- **Topology**: Match circuit connectivity to device layout
- **Qubit quality**: Some qubits on a device are better than others

---

## 7. Error Mitigation

### 7.1 Measurement Error Mitigation

Readout errors are the largest noise source for many circuits. Mitigation:

1. **Calibration**: Prepare all $2^n$ basis states and measure, building the response matrix $M$ where $M_{ij} = P(\text{measure } i | \text{prepared } j)$
2. **Correction**: Given measured counts vector $\vec{c}_{\text{noisy}}$, the corrected counts are $\vec{c}_{\text{ideal}} = M^{-1} \vec{c}_{\text{noisy}}$

For $n > 10$ qubits, full calibration is impractical ($2^n$ states). Use:
- **Tensored mitigation**: Assume readout errors are independent per qubit
- **Local mitigation**: Characterize $2 \times 2$ matrices per qubit ($2n$ circuits)

### 7.2 Zero-Noise Extrapolation (ZNE)

1. Run the circuit at the physical noise level $\epsilon$
2. Amplify noise by stretching gate durations: noise levels $\epsilon, 2\epsilon, 3\epsilon$
3. Fit the results and extrapolate to $\epsilon = 0$

Noise amplification methods:
- **Pulse stretching**: Increase gate duration (scales thermal noise)
- **Unitary folding**: Replace $U$ with $U U^\dagger U$ (doubles the effective noise)
- **Digital insertion**: Insert pairs of CNOT gates ($\text{CNOT}^2 = I$ ideally)

### 7.3 Probabilistic Error Cancellation (PEC)

1. Characterize the noise channel $\mathcal{N}$ for each gate
2. Decompose the ideal gate as a quasi-probability distribution over noisy operations:
   $\mathcal{G}_{\text{ideal}} = \sum_i \eta_i \mathcal{O}_i$ where $\eta_i$ can be negative
3. Sample operations according to $|\eta_i|/\gamma$ where $\gamma = \sum |eta_i|$
4. Weight each sample by $\gamma \cdot \text{sign}(\eta_i)$

**Overhead**: Sampling cost scales as $\gamma^{2G}$ where $G$ is the number of gates.

### 7.4 Twirled Readout Error Extinction (TREX)

A more efficient alternative to full readout calibration:
1. Before measurement, randomly apply $X$ gates to each qubit
2. Flip the corresponding bits in the classical output
3. The net effect is that readout noise becomes a symmetric bit-flip channel
4. Correct by dividing by the bit-flip probability (estimated from the twirling data)

---

## 8. Parameterized Circuits and VQE

### 8.1 Estimator Primitive

The Qiskit `Estimator` primitive computes expectation values:

$$\langle O \rangle = \text{Tr}[O \cdot \rho(\theta)]$$

where $O$ is an observable (Pauli string) and $\rho(\theta)$ is the parameterized circuit output.

### 8.2 VQE with Qiskit

The VQE workflow in Qiskit:

1. Define the Hamiltonian as a `SparsePauliOp`
2. Choose an ansatz (e.g., `EfficientSU2`, `UCC`)
3. Choose an optimizer (e.g., `COBYLA`, `SPSA`)
4. Run VQE: the optimizer calls the Estimator with different parameters until convergence

### 8.3 SPSA Optimizer

For noisy hardware, gradient-based optimizers struggle because gradient estimates are noisy. **SPSA** (Simultaneous Perturbation Stochastic Approximation) is preferred:

- Estimates the gradient using only 2 function evaluations (regardless of parameter count)
- Robust to noise in the objective function
- Hyperparameters: learning rate $a_k = a/(A+k+1)^\alpha$, perturbation $c_k = c/(k+1)^\gamma$

---

## 9. Python Implementation

### 9.1 Circuit Simulation Framework

```python
import numpy as np
from scipy.linalg import expm

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def kron_list(ops):
    """Tensor product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


class QuantumCircuit:
    """Simple quantum circuit simulator (mirrors Qiskit interface).

    This implementation stores gates as a list of operations and
    applies them sequentially to compute the final state vector.
    It demonstrates the core concepts behind Qiskit's QuantumCircuit.
    """

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.N = 2 ** n_qubits
        self.gates = []
        self.parameters = {}

    def h(self, qubit):
        """Hadamard gate."""
        self.gates.append(('H', qubit))

    def x(self, qubit):
        """Pauli X gate."""
        self.gates.append(('X', qubit))

    def z(self, qubit):
        """Pauli Z gate."""
        self.gates.append(('Z', qubit))

    def ry(self, angle, qubit):
        """Ry rotation gate."""
        self.gates.append(('RY', qubit, angle))

    def rz(self, angle, qubit):
        """Rz rotation gate."""
        self.gates.append(('RZ', qubit, angle))

    def cx(self, control, target):
        """CNOT (controlled-X) gate."""
        self.gates.append(('CX', control, target))

    def measure_all(self):
        """Mark all qubits for measurement."""
        self.gates.append(('MEASURE',))

    def get_unitary(self, params=None):
        """Compute the unitary matrix of the circuit.

        This is the 'simulation' step — converting the gate list
        into a single unitary matrix.
        """
        U = np.eye(self.N, dtype=complex)

        for gate in self.gates:
            if gate[0] == 'MEASURE':
                continue
            elif gate[0] == 'H':
                q = gate[1]
                ops = [I] * self.n_qubits
                ops[q] = H_gate
                G = kron_list(ops)
            elif gate[0] == 'X':
                q = gate[1]
                ops = [I] * self.n_qubits
                ops[q] = X
                G = kron_list(ops)
            elif gate[0] == 'Z':
                q = gate[1]
                ops = [I] * self.n_qubits
                ops[q] = Z
                G = kron_list(ops)
            elif gate[0] == 'RY':
                q, angle = gate[1], gate[2]
                if params and isinstance(angle, str):
                    angle = params[angle]
                ry = np.array([[np.cos(angle/2), -np.sin(angle/2)],
                              [np.sin(angle/2), np.cos(angle/2)]], dtype=complex)
                ops = [I] * self.n_qubits
                ops[q] = ry
                G = kron_list(ops)
            elif gate[0] == 'RZ':
                q, angle = gate[1], gate[2]
                if params and isinstance(angle, str):
                    angle = params[angle]
                rz = np.array([[np.exp(-1j*angle/2), 0],
                              [0, np.exp(1j*angle/2)]], dtype=complex)
                ops = [I] * self.n_qubits
                ops[q] = rz
                G = kron_list(ops)
            elif gate[0] == 'CX':
                ctrl, tgt = gate[1], gate[2]
                G = np.eye(self.N, dtype=complex)
                for s in range(self.N):
                    if (s >> (self.n_qubits - 1 - ctrl)) & 1:
                        new_s = s ^ (1 << (self.n_qubits - 1 - tgt))
                        G[s, s] = 0
                        G[new_s, s] = 1
                        G[s, new_s] = 0 if new_s != s else G[s, new_s]
                # Rebuild properly
                G = np.eye(self.N, dtype=complex)
                for s in range(self.N):
                    ctrl_bit = (s >> (self.n_qubits - 1 - ctrl)) & 1
                    if ctrl_bit:
                        new_s = s ^ (1 << (self.n_qubits - 1 - tgt))
                        G[s, :] = 0
                        G[new_s, :] = 0
                for s in range(self.N):
                    ctrl_bit = (s >> (self.n_qubits - 1 - ctrl)) & 1
                    if ctrl_bit:
                        new_s = s ^ (1 << (self.n_qubits - 1 - tgt))
                        G[new_s, s] = 1
                    else:
                        G[s, s] = 1
            else:
                continue

            U = G @ U

        return U

    def simulate(self, shots=1024, noise_model=None, params=None):
        """Simulate the circuit and return measurement counts.

        Args:
            shots: Number of measurement samples
            noise_model: Optional noise model dict
            params: Parameter bindings

        Returns:
            counts: Dict mapping bit strings to counts
        """
        U = self.get_unitary(params)
        state = np.zeros(self.N, dtype=complex)
        state[0] = 1.0
        state = U @ state

        # Apply noise if specified
        if noise_model:
            rho = np.outer(state, state.conj())
            rho = noise_model.apply(rho, self.n_qubits)
            probs = np.real(np.diag(rho))
            probs = np.maximum(probs, 0)
            probs /= np.sum(probs)
        else:
            probs = np.abs(state) ** 2

        # Sample
        rng = np.random.default_rng()
        outcomes = rng.choice(self.N, size=shots, p=probs)

        counts = {}
        for outcome in outcomes:
            bitstring = format(outcome, f'0{self.n_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts

    @property
    def depth(self):
        """Estimate circuit depth."""
        return len([g for g in self.gates if g[0] != 'MEASURE'])

    @property
    def num_cnots(self):
        """Count CNOT gates."""
        return len([g for g in self.gates if g[0] == 'CX'])

    def __str__(self):
        lines = [f"QuantumCircuit({self.n_qubits} qubits, {self.depth} gates, {self.num_cnots} CNOTs)"]
        for g in self.gates:
            lines.append(f"  {g}")
        return '\n'.join(lines)


# Demonstrate circuit construction
print("=" * 65)
print("Quantum Circuit Construction and Simulation")
print("=" * 65)

# Bell state circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

print(f"\nBell state circuit:")
print(qc)

counts = qc.simulate(shots=10000)
print(f"\nMeasurement results ({sum(counts.values())} shots):")
for bitstring, count in sorted(counts.items()):
    print(f"  |{bitstring}⟩: {count} ({count/10000:.3f})")

# GHZ state
print(f"\n--- 4-qubit GHZ state ---")
qc_ghz = QuantumCircuit(4)
qc_ghz.h(0)
for i in range(3):
    qc_ghz.cx(i, i + 1)
qc_ghz.measure_all()

counts = qc_ghz.simulate(shots=10000)
print(f"Results:")
for bs, c in sorted(counts.items(), key=lambda x: -x[1])[:5]:
    print(f"  |{bs}⟩: {c} ({c/10000:.3f})")
```

### 9.2 Transpilation Simulation

```python
import numpy as np

def decompose_toffoli():
    """Decompose a Toffoli gate into CNOT + single-qubit gates.

    The standard decomposition uses 6 CNOTs and several T/T-dagger/H gates.
    This is what a transpiler does for non-native multi-qubit gates.
    """
    print("=" * 65)
    print("Gate Decomposition: Toffoli → CNOT + Singles")
    print("=" * 65)

    # Build Toffoli directly
    N = 8  # 3 qubits
    toffoli = np.eye(N, dtype=complex)
    toffoli[6, 6] = 0
    toffoli[7, 7] = 0
    toffoli[6, 7] = 1
    toffoli[7, 6] = 1

    # Decomposed version using our QuantumCircuit
    qc = QuantumCircuit(3)
    # Standard decomposition (simplified)
    qc.h(2)
    qc.cx(1, 2)
    qc.rz(-np.pi/4, 2)
    qc.cx(0, 2)
    qc.rz(np.pi/4, 2)
    qc.cx(1, 2)
    qc.rz(-np.pi/4, 2)
    qc.cx(0, 2)
    qc.rz(np.pi/4, 1)
    qc.rz(np.pi/4, 2)
    qc.h(2)
    qc.cx(0, 1)
    qc.rz(np.pi/4, 0)
    qc.rz(-np.pi/4, 1)
    qc.cx(0, 1)

    U_decomposed = qc.get_unitary()

    # Compare (up to global phase)
    phase = np.exp(1j * np.angle(U_decomposed[0, 0] / toffoli[0, 0]))
    error = np.linalg.norm(U_decomposed - phase * toffoli)

    print(f"\n  Original: Toffoli (1 gate)")
    print(f"  Decomposed: {qc.num_cnots} CNOTs + {qc.depth - qc.num_cnots} single-qubit gates")
    print(f"  Decomposition error: {error:.2e}")


def simulate_routing(n_qubits, connectivity, target_cx_pairs):
    """Simulate qubit routing for a given connectivity graph.

    When a CNOT is needed between non-adjacent qubits,
    SWAP gates must be inserted. Each SWAP = 3 CNOTs.
    """
    print(f"\n--- Routing simulation ---")
    print(f"  Qubits: {n_qubits}")
    print(f"  Connectivity: {connectivity}")
    print(f"  Required CX pairs: {target_cx_pairs}")

    # Check which CX pairs need routing
    total_swaps = 0
    for ctrl, tgt in target_cx_pairs:
        if (ctrl, tgt) in connectivity or (tgt, ctrl) in connectivity:
            print(f"  CX({ctrl},{tgt}): direct (0 SWAPs)")
        else:
            # Simple BFS to find shortest path
            from collections import deque
            adj = {i: set() for i in range(n_qubits)}
            for a, b in connectivity:
                adj[a].add(b)
                adj[b].add(a)

            visited = {ctrl}
            queue = deque([(ctrl, 0)])
            dist = -1
            while queue:
                node, d = queue.popleft()
                if node == tgt:
                    dist = d
                    break
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, d + 1))

            n_swaps = max(0, dist - 1)
            total_swaps += n_swaps
            print(f"  CX({ctrl},{tgt}): needs {n_swaps} SWAPs ({n_swaps * 3} CNOTs)")

    print(f"  Total SWAP overhead: {total_swaps} SWAPs = {total_swaps * 3} extra CNOTs")


decompose_toffoli()

# Routing example: linear chain topology
print("\n" + "=" * 65)
print("Qubit Routing Simulation")
print("=" * 65)

connectivity = [(0,1), (1,2), (2,3), (3,4)]  # Linear chain
target_pairs = [(0,1), (0,4), (1,3), (0,2)]
simulate_routing(5, connectivity, target_pairs)

# Heavy-hex topology (subset)
connectivity_hex = [(0,1), (1,2), (1,3), (3,4), (3,5), (5,6)]
target_pairs_hex = [(0,6), (2,4), (0,5)]
simulate_routing(7, connectivity_hex, target_pairs_hex)
```

### 9.3 Noise Model Simulation

```python
import numpy as np

class NoiseModel:
    """Simple noise model for quantum circuit simulation.

    Models depolarizing errors on gates and readout errors on measurements.
    This mirrors the core functionality of Qiskit Aer's noise models.
    """

    def __init__(self, single_qubit_error=0.001, two_qubit_error=0.01,
                 readout_error=0.02):
        self.sq_error = single_qubit_error
        self.tq_error = two_qubit_error
        self.readout_error = readout_error

    def apply(self, rho, n_qubits):
        """Apply readout noise to the density matrix.

        For simplicity, we model readout error as a classical bit-flip
        on the diagonal of the density matrix (in the computational basis).
        """
        N = 2 ** n_qubits
        probs = np.real(np.diag(rho))

        # Readout error: each bit flips with probability readout_error
        new_probs = np.zeros(N)
        for s in range(N):
            prob = probs[s]
            # For each qubit, probability of correct readout
            for q in range(n_qubits):
                pass  # Applied during sampling instead
            new_probs[s] = prob

        rho_out = np.diag(new_probs.astype(complex))
        return rho_out

    def noisy_sample(self, probs, n_qubits, shots, rng=None):
        """Sample with readout errors.

        After ideal sampling, each bit is flipped independently
        with probability readout_error.
        """
        if rng is None:
            rng = np.random.default_rng()

        N = len(probs)
        outcomes = rng.choice(N, size=shots, p=probs)

        # Apply readout errors
        for i in range(shots):
            for q in range(n_qubits):
                if rng.random() < self.readout_error:
                    outcomes[i] ^= (1 << (n_qubits - 1 - q))

        return outcomes


def simulate_with_noise(qc, noise_model, shots=10000):
    """Simulate a circuit with a noise model.

    Applies the circuit unitary, computes ideal probabilities,
    then applies readout errors during sampling.
    """
    U = qc.get_unitary()
    state = np.zeros(qc.N, dtype=complex)
    state[0] = 1.0
    state = U @ state
    probs = np.abs(state) ** 2

    # Depolarizing noise on state (approximate)
    n_gates = qc.depth
    n_cnots = qc.num_cnots
    circuit_fidelity = ((1 - noise_model.sq_error) ** (n_gates - n_cnots) *
                        (1 - noise_model.tq_error) ** n_cnots)

    # Mix with uniform distribution
    uniform = np.ones(qc.N) / qc.N
    noisy_probs = circuit_fidelity * probs + (1 - circuit_fidelity) * uniform
    noisy_probs /= np.sum(noisy_probs)

    # Sample with readout errors
    rng = np.random.default_rng(42)
    outcomes = noise_model.noisy_sample(noisy_probs, qc.n_qubits, shots, rng)

    counts = {}
    for outcome in outcomes:
        bs = format(outcome, f'0{qc.n_qubits}b')
        counts[bs] = counts.get(bs, 0) + 1

    return counts


# Demonstrate noise effects
print("=" * 65)
print("Noise Model Simulation")
print("=" * 65)

# Bell state circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

print("\nBell state with varying noise levels:")
for sq_err, tq_err, ro_err in [(0, 0, 0), (0.001, 0.01, 0.02), (0.01, 0.05, 0.05)]:
    noise = NoiseModel(sq_err, tq_err, ro_err)
    counts = simulate_with_noise(qc, noise, shots=10000)

    print(f"\n  SQ_err={sq_err}, TQ_err={tq_err}, RO_err={ro_err}:")
    for bs in sorted(counts.keys()):
        print(f"    |{bs}⟩: {counts.get(bs, 0):>5} ({counts.get(bs, 0)/10000:.3f})")
```

### 9.4 Measurement Error Mitigation

```python
import numpy as np

def calibrate_readout(n_qubits, noise_model, shots=10000):
    """Calibrate readout errors by preparing and measuring all basis states.

    For n qubits, prepare |0...0>, |0...01>, ..., |1...1> and measure.
    The resulting matrix M has M[i][j] = P(measure i | prepared j).

    For large n, use tensored mitigation: calibrate each qubit independently.

    Args:
        n_qubits: Number of qubits
        noise_model: Noise model with readout errors
        shots: Calibration shots per basis state

    Returns:
        M: Response matrix (2^n x 2^n)
    """
    N = 2 ** n_qubits
    M = np.zeros((N, N))
    rng = np.random.default_rng(42)

    for prepared_state in range(N):
        # Ideal: delta distribution at prepared_state
        probs = np.zeros(N)
        probs[prepared_state] = 1.0

        # Apply readout noise via sampling
        outcomes = noise_model.noisy_sample(probs, n_qubits, shots, rng)

        for outcome in outcomes:
            M[outcome, prepared_state] += 1
        M[:, prepared_state] /= shots

    return M


def mitigate_readout(counts, M, n_qubits, shots):
    """Apply readout error mitigation using the inverse response matrix.

    Given noisy counts c_noisy and response matrix M,
    the corrected counts are c_ideal = M^{-1} c_noisy.

    In practice, we use least-squares with non-negativity constraints.
    """
    N = 2 ** n_qubits

    # Convert counts to probability vector
    noisy_vec = np.zeros(N)
    for bs, count in counts.items():
        idx = int(bs, 2)
        noisy_vec[idx] = count / shots

    # Invert (using pseudo-inverse for numerical stability)
    M_inv = np.linalg.pinv(M)
    corrected_vec = M_inv @ noisy_vec

    # Project to valid probability distribution (non-negative, sum to 1)
    corrected_vec = np.maximum(corrected_vec, 0)
    if np.sum(corrected_vec) > 0:
        corrected_vec /= np.sum(corrected_vec)

    # Convert back to counts
    corrected_counts = {}
    for i in range(N):
        if corrected_vec[i] > 1e-6:
            bs = format(i, f'0{n_qubits}b')
            corrected_counts[bs] = int(round(corrected_vec[i] * shots))

    return corrected_counts


# Demonstrate readout error mitigation
print("=" * 65)
print("Readout Error Mitigation")
print("=" * 65)

noise = NoiseModel(single_qubit_error=0.001, two_qubit_error=0.01,
                    readout_error=0.05)

# Calibrate
n_qubits = 2
M = calibrate_readout(n_qubits, noise, shots=50000)

print(f"\nResponse matrix (2-qubit calibration):")
labels = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
print(f"  Measured \\ Prepared: ", end="")
for l in labels:
    print(f"  |{l}⟩", end="")
print()
for i, li in enumerate(labels):
    print(f"  |{li}⟩:             ", end="")
    for j in range(2**n_qubits):
        print(f"  {M[i,j]:.3f}", end="")
    print()

# Run a noisy circuit and mitigate
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

shots = 10000
noisy_counts = simulate_with_noise(qc, noise, shots=shots)
mitigated_counts = mitigate_readout(noisy_counts, M, n_qubits, shots)

print(f"\nBell state results:")
print(f"  {'State':>8} {'Ideal':>10} {'Noisy':>10} {'Mitigated':>10}")
print(f"  {'-' * 42}")
for bs in ['00', '01', '10', '11']:
    ideal = 5000 if bs in ['00', '11'] else 0
    noisy = noisy_counts.get(bs, 0)
    mitigated = mitigated_counts.get(bs, 0)
    print(f"  |{bs}⟩ {ideal:10d} {noisy:10d} {mitigated:10d}")
```

### 9.5 Zero-Noise Extrapolation

```python
import numpy as np
from scipy.optimize import curve_fit

def zero_noise_extrapolation(circuit_func, noise_levels, true_value=None):
    """Apply zero-noise extrapolation to improve expectation value estimates.

    Run the circuit at multiple noise levels and extrapolate to zero noise.
    This works because the expectation value is a smooth function of
    the noise parameter, and we can fit and extrapolate.

    Args:
        circuit_func: Function(noise_level) -> expectation value
        noise_levels: List of noise amplification factors [1, 2, 3, ...]
        true_value: True (noiseless) value for comparison

    Returns:
        extrapolated: Zero-noise extrapolated value
    """
    # Compute expectation values at each noise level
    values = [circuit_func(nl) for nl in noise_levels]

    # Linear extrapolation
    if len(noise_levels) >= 2:
        coeffs_lin = np.polyfit(noise_levels, values, 1)
        linear_extrap = np.polyval(coeffs_lin, 0)
    else:
        linear_extrap = values[0]

    # Quadratic extrapolation
    if len(noise_levels) >= 3:
        coeffs_quad = np.polyfit(noise_levels, values, 2)
        quadratic_extrap = np.polyval(coeffs_quad, 0)
    else:
        quadratic_extrap = linear_extrap

    # Exponential extrapolation
    try:
        def exp_model(x, a, b, c):
            return a * np.exp(b * np.array(x)) + c
        popt, _ = curve_fit(exp_model, noise_levels, values,
                           p0=[values[0]-values[-1], -0.1, values[-1]], maxfev=5000)
        exp_extrap = exp_model(0, *popt)
    except (RuntimeError, ValueError):
        exp_extrap = linear_extrap

    return {
        'noisy_values': dict(zip(noise_levels, values)),
        'linear': linear_extrap,
        'quadratic': quadratic_extrap,
        'exponential': exp_extrap,
        'true': true_value,
    }


# Demonstrate ZNE
print("=" * 65)
print("Zero-Noise Extrapolation")
print("=" * 65)

# Simulated noisy expectation value
true_value = 0.85  # True <Z> for some circuit
base_noise = 0.02

def noisy_expectation(noise_factor):
    """Simulate a noisy expectation value with noise amplification."""
    effective_noise = base_noise * noise_factor
    # Model: <Z>_noisy = <Z>_ideal * (1 - 2*p)^n_gates where p is noise
    n_gates = 20
    fidelity = (1 - effective_noise) ** n_gates
    rng = np.random.default_rng(int(noise_factor * 1000))
    shot_noise = rng.normal(0, 0.01)
    return true_value * fidelity + shot_noise

noise_levels = [1, 1.5, 2, 2.5, 3]
result = zero_noise_extrapolation(noisy_expectation, noise_levels, true_value)

print(f"\nTrue value: {true_value:.4f}")
print(f"\nNoisy measurements:")
for nl, val in result['noisy_values'].items():
    print(f"  Noise x{nl:.1f}: {val:.4f}")

print(f"\nExtrapolated values:")
print(f"  Linear:      {result['linear']:.4f} (error: {abs(result['linear'] - true_value):.4f})")
print(f"  Quadratic:   {result['quadratic']:.4f} (error: {abs(result['quadratic'] - true_value):.4f})")
print(f"  Exponential: {result['exponential']:.4f} (error: {abs(result['exponential'] - true_value):.4f})")
print(f"  No mitigation: {result['noisy_values'][1]:.4f} (error: {abs(result['noisy_values'][1] - true_value):.4f})")
```

---

## 10. Exercises

### Exercise 1: Circuit Optimization

(a) Build a 4-qubit circuit that prepares the W state $|W\rangle = (|1000\rangle + |0100\rangle + |0010\rangle + |0001\rangle)/2$.
(b) Count the CNOT gates in your implementation. Can you reduce it?
(c) Decompose each gate into the IBM basis set $\{CX, I, R_z, \sqrt{X}, X\}$.
(d) Simulate the circuit with and without optimization and compare fidelity under noise.

### Exercise 2: Routing Challenge

For a 5-qubit circuit requiring CX gates between all pairs $(0,4)$, $(1,3)$, $(2,4)$, $(0,3)$:
(a) Draw the linear chain topology and identify which CX pairs need SWAPs.
(b) Find the minimum number of SWAP gates needed.
(c) Compare the CNOT overhead for linear chain vs. T-shape vs. ring topologies.
(d) Implement a simple SABRE-like routing heuristic.

### Exercise 3: Noise Characterization

(a) Simulate a noisy Bell state circuit with $T_1 = 100\mu s$, $T_2 = 50\mu s$, gate time $= 100ns$.
(b) Compute the expected fidelity analytically and compare with simulation.
(c) How does the fidelity change if you add dynamical decoupling (DD) sequences?
(d) At what error rate does the Bell state become useless (fidelity $< 0.5$)?

### Exercise 4: Readout Mitigation Scaling

(a) Implement tensored readout mitigation for 4 qubits (calibrate each qubit independently).
(b) Compare with full calibration (16 basis states). When does tensored mitigation fail?
(c) Add correlated readout errors (crosstalk) and show that tensored mitigation is insufficient.
(d) How many calibration circuits are needed for $n = 10, 20, 50$ qubits with each method?

### Exercise 5: VQE Under Noise

Run VQE for the 2-qubit Heisenberg model $H = X_0X_1 + Y_0Y_1 + Z_0Z_1$:
(a) Find the ground state energy without noise (exact: $-3$).
(b) Add depolarizing noise and observe how the VQE energy changes.
(c) Apply ZNE to the VQE energy. How much does it improve?
(d) Compare COBYLA and SPSA optimizers under noise. Which converges more reliably?

---

[← Previous: Quantum Networking](23_Quantum_Networking.md) | [Next: Capstone Quantum Application →](25_Capstone_Quantum_Application.md)
