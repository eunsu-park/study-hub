# Lesson 22: Topological Quantum Computing

[← Previous: Quantum Chemistry](21_Quantum_Chemistry.md) | [Next: Quantum Networking →](23_Quantum_Networking.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the concept of topological protection and why it leads to inherently fault-tolerant qubits
2. Describe anyons, their braiding statistics, and how braiding implements quantum gates
3. Distinguish between Abelian and non-Abelian anyons and their computational power
4. Analyze the surface code as a practical topological error-correcting code
5. Explain the role of Majorana fermions in topological quantum computing
6. Compare topological approaches with conventional gate-based quantum computing
7. Implement surface code simulation and anyon braiding in Python

---

Topological quantum computing is a radically different approach to building a quantum computer. Instead of encoding information in fragile quantum states that must be carefully isolated from noise, topological quantum computing encodes information in global, topological properties of a system — properties that are inherently resistant to local perturbations. A topological qubit is like a knot in a rope: you can shake the rope, stretch it, or deform it locally, but the knot remains unchanged. Only a global topological change (cutting the rope) can destroy the information.

The mathematical framework rests on **anyons** — exotic quasi-particles that exist only in two-dimensional systems. Unlike bosons (which are unchanged when two particles are exchanged) or fermions (which pick up a minus sign), anyons can acquire arbitrary phases or even undergo non-trivial unitary transformations when braided around each other. These braiding operations are inherently topological: they depend only on the topology of the particle worldlines, not on the precise paths taken.

> **Analogy:** Imagine information stored as knots in strings. Local noise is like jiggling the string — it cannot change the knot type. Only cutting and reconnecting the string (a topological operation) can alter the encoded information. Computation is performed by braiding the strings around each other, and the result depends only on the braid pattern, not on the exact path of each string.

## Table of Contents

1. [Motivation: Topological Protection](#1-motivation-topological-protection)
2. [Anyons and Exchange Statistics](#2-anyons-and-exchange-statistics)
3. [Braiding and Quantum Gates](#3-braiding-and-quantum-gates)
4. [Non-Abelian Anyons](#4-non-abelian-anyons)
5. [Majorana Fermions](#5-majorana-fermions)
6. [Surface Codes](#6-surface-codes)
7. [Topological Codes in Practice](#7-topological-codes-in-practice)
8. [Comparison with Conventional Approaches](#8-comparison-with-conventional-approaches)
9. [Python Implementation](#9-python-implementation)
10. [Exercises](#10-exercises)

---

## 1. Motivation: Topological Protection

### 1.1 The Decoherence Problem

Conventional qubits are fragile: local perturbations (electromagnetic fluctuations, thermal phonons, cosmic rays) can flip, dephase, or otherwise corrupt the quantum state. The error rate per gate is typically $10^{-3}$ to $10^{-4}$, and coherence times are microseconds to milliseconds.

### 1.2 Topological Solution

Topological quantum computing encodes information in properties that are invariant under continuous local deformations:

- **Local perturbations** cannot change topological properties (like changing a coffee mug into a donut without cutting)
- Information is stored **non-locally**, spread across the entire system
- Errors must be large-scale (spanning the entire system) to corrupt the information

The error rate for topological qubits scales as $e^{-cL}$ where $L$ is the system size, providing exponential protection.

### 1.3 Mathematical Basis

Topological quantum computing uses concepts from:

| Mathematical concept | Role in TQC |
|---------------------|-------------|
| Topology | Classification of knots and braids |
| Category theory | Algebraic structure of anyons |
| Representation theory | Braid group representations |
| Algebraic topology | Topological invariants, homology |
| Conformal field theory | Connection to 2D physics |

---

## 2. Anyons and Exchange Statistics

### 2.1 Particle Statistics in 2D

In three dimensions, exchanging two identical particles twice returns the system to its original state. This allows only two possibilities:
- **Bosons**: Exchange gives factor $+1$
- **Fermions**: Exchange gives factor $-1$

In two dimensions, the topology of particle exchange paths is richer (the fundamental group of the configuration space is the braid group, not the symmetric group). This allows:
- **Anyons**: Exchange gives factor $e^{i\theta}$ for any $\theta$ (Abelian anyons)
- **Non-Abelian anyons**: Exchange applies a unitary matrix (not just a phase)

### 2.2 Abelian Anyons

For Abelian anyons, exchanging particles $i$ and $j$ gives:

$$|\psi\rangle \to e^{i\theta_{ij}}|\psi\rangle$$

The phase $\theta_{ij}$ depends on the anyon types but not on the state. This is useful for error correction (e.g., the toric code) but not sufficient for universal quantum computation.

### 2.3 The Braid Group

The braid group $B_n$ on $n$ strands is generated by elementary braids $\sigma_1, \sigma_2, \ldots, \sigma_{n-1}$ satisfying:

- **Yang-Baxter equation**: $\sigma_i \sigma_{i+1} \sigma_i = \sigma_{i+1} \sigma_i \sigma_{i+1}$
- **Far commutativity**: $\sigma_i \sigma_j = \sigma_j \sigma_i$ for $|i-j| \geq 2$

Unlike the symmetric group, $\sigma_i^2 \neq 1$ — braiding twice is not the same as not braiding at all.

### 2.4 Braid Diagrams

The elementary braid generator $\sigma_i$ exchanges strands $i$ and $i+1$, with strand $i$ passing *over* strand $i+1$. Time flows downward:

```
  σ₁ (strand 1 over strand 2):      σ₁⁻¹ (strand 2 over strand 1):

  1    2    3                         1    2    3
  |    |    |                         |    |    |
  |    |    |                         |    |    |
   \  /     |                          \  /     |
    \/      |                           \/      |
    /\      |      strand 1 passes      /\      |
   /  \     |      OVER strand 2       /  \     |
  |    |    |                         |    |    |
  2    1    3                         2    1    3

  Identity (σ₁ σ₁⁻¹):               Yang-Baxter (σ₁ σ₂ σ₁ = σ₂ σ₁ σ₂):

  1    2    3                         Strands can slide past each other
  |    |    |                         as long as the crossing pattern
   \  /     |                         (over/under) is preserved. This
    \/      |                         topological invariance is what
    /\      |                         makes braiding robust against
   /  \     |                         local perturbations.
    \/      |
    /\      |
   /  \     |
  |    |    |
  1    2    3  (back to start)
```

### 2.5 Topological Charge

Anyons carry a **topological charge** (or anyon type). When two anyons are brought together, their charges can **fuse** according to fusion rules:

$$a \times b = \sum_c N_{ab}^c \, c$$

where $N_{ab}^c$ is the number of ways $a$ and $b$ can fuse to give $c$. If $N_{ab}^c > 1$ for some $c$, the anyons are non-Abelian.

---

## 3. Braiding and Quantum Gates

### 3.1 How Braiding Computes

The central idea of topological quantum computing:

1. **Initialize**: Create pairs of anyons from the vacuum (each pair has trivial total charge)
2. **Compute**: Move anyons around each other (braiding). Each braid pattern implements a specific unitary operation
3. **Readout**: Fuse anyon pairs and measure the total charge (which encodes the computation result)

### 3.2 Braid Group Representations

A quantum system with $n$ anyons of type $a$ has a degenerate ground state space. The braid group acts on this space through a unitary representation:

$$\sigma_i \to R_i \in U(\text{dim of ground space})$$

The matrices $R_i$ must satisfy the braid group relations.

### 3.3 Example: Fibonacci Anyons

Fibonacci anyons have two topological charges: $1$ (vacuum) and $\tau$ (non-trivial). Fusion rules:

$$\tau \times \tau = 1 + \tau$$

The fusion space of $n$ Fibonacci anyons has dimension $F_{n-1}$ (Fibonacci number). Three anyons have a 2-dimensional fusion space, encoding one qubit.

The braid matrices for Fibonacci anyons are:

$$\sigma_1 = \begin{pmatrix} e^{-4\pi i/5} & 0 \\ 0 & e^{3\pi i/5} \end{pmatrix}, \quad \sigma_2 = \begin{pmatrix} \phi^{-1}e^{-4\pi i/5} & \phi^{-1/2}e^{3\pi i/5} \\ \phi^{-1/2}e^{3\pi i/5} & -\phi^{-1} \end{pmatrix}$$

where $\phi = (1 + \sqrt{5})/2$ is the golden ratio.

### 3.4 Measurement via Anyon Fusion

In topological quantum computing, measurement is performed by **fusing** anyons — bringing two anyons together and observing the resulting topological charge. For Fibonacci anyons with fusion rule $\tau \times \tau = 1 + \tau$, fusing two $\tau$ anyons can yield either the vacuum $1$ or another $\tau$ anyon. The outcome is probabilistic and determined by the quantum state of the system, which has been shaped by prior braiding operations.

The fusion outcome is detected by an **interferometric measurement**: a probe anyon is sent around the fused pair, and the phase acquired reveals the total charge. If the pair fuses to $1$ (vacuum), the probe acquires no phase; if the pair fuses to $\tau$, the probe acquires a non-trivial phase. This measurement is inherently topological — it depends only on the total charge enclosed, not on microscopic details — providing the same robustness as the braiding operations themselves.

For a qubit encoded in three Fibonacci anyons ($\tau_1, \tau_2, \tau_3$), measurement corresponds to fusing $\tau_1$ and $\tau_2$ and observing whether the result is $1$ (logical $|0\rangle$) or $\tau$ (logical $|1\rangle$).

### 3.5 Universality

**Fibonacci anyons are universal for quantum computation**: any single-qubit unitary can be approximated to arbitrary precision by a finite sequence of braids. This was proven by Freedman, Larsen, and Wang (2002).

The approximation error decreases exponentially with the braid length: $\epsilon \sim e^{-cL}$ where $L$ is the number of elementary braids.

---

## 4. Non-Abelian Anyons

### 4.1 Ising Anyons

Ising anyons are a simpler (but less powerful) non-Abelian system. Three anyon types: $1$ (vacuum), $\sigma$ (Ising anyon), $\psi$ (fermion).

Fusion rules:
$$\sigma \times \sigma = 1 + \psi, \quad \sigma \times \psi = \sigma, \quad \psi \times \psi = 1$$

**Limitation**: Ising anyons can implement Clifford gates by braiding but NOT the T gate. They are therefore not universal by braiding alone and require "magic state distillation" for universality.

### 4.2 $SU(2)_k$ Anyons

The $SU(2)_k$ Chern-Simons theory produces anyons labeled by spin $j = 0, 1/2, 1, \ldots, k/2$:

| $k$ | Anyon model | Universal? | Physical system |
|-----|-------------|-----------|----------------|
| 1 | Semion | No (Abelian) | $\nu = 1/2$ Laughlin |
| 2 | Ising | No (Clifford only) | $\nu = 5/2$ FQHE, Majorana |
| 3 | Fibonacci | Yes | $\nu = 12/5$ FQHE (theoretical) |
| 4 | $SO(3)_2$ | No | Certain spin liquids |

### 4.3 Physical Realizations

| System | Anyon type | Status |
|--------|-----------|--------|
| Fractional quantum Hall ($\nu = 5/2$) | Ising (Majorana) | Experimental evidence |
| Topological superconductors | Majorana zero modes | Active research |
| Kitaev spin liquid | Non-Abelian anyons | Theoretical |
| Photonic systems | Simulated anyons | Proof of concept |

---

## 5. Majorana Fermions

### 5.1 What Are Majorana Fermions?

A Majorana fermion is its own antiparticle: $\gamma = \gamma^\dagger$. In condensed matter, **Majorana zero modes** (MZMs) appear at the ends of topological superconducting wires. They obey:

$$\gamma_i = \gamma_i^\dagger, \quad \{\gamma_i, \gamma_j\} = 2\delta_{ij}$$

### 5.2 Majorana Qubits

Two Majorana zero modes $\gamma_1, \gamma_2$ form one fermionic mode:

$$c = \frac{\gamma_1 + i\gamma_2}{2}, \quad c^\dagger = \frac{\gamma_1 - i\gamma_2}{2}$$

The occupation of this mode ($n = c^\dagger c = 0$ or $1$) is the qubit. Since $\gamma_1$ and $\gamma_2$ are spatially separated, local perturbations cannot change the occupation.

### 5.3 Braiding Majorana Zero Modes

Exchanging two MZMs implements:

$$\gamma_i \to \gamma_j, \quad \gamma_j \to -\gamma_i$$

This corresponds to the unitary: $U_{ij} = \frac{1}{\sqrt{2}}(1 + \gamma_i\gamma_j)$

For four MZMs ($\gamma_1, \gamma_2, \gamma_3, \gamma_4$) encoding one qubit:
- Exchanging $\gamma_1 \leftrightarrow \gamma_2$: implements $e^{i\pi/4}$ phase gate
- Exchanging $\gamma_2 \leftrightarrow \gamma_3$: implements a rotation

### 5.4 Microsoft's Approach

Microsoft has invested heavily in topological quantum computing using Majorana zero modes:

1. **Platform**: Semiconductor-superconductor nanowires (InAs/Al, InSb/Al)
2. **Goal**: Create and manipulate Majorana zero modes at wire endpoints
3. **Status**: Claimed evidence of topological superconductivity (2023), working toward topological qubit demonstration
4. **Challenge**: Distinguishing true MZMs from trivial zero-energy states

---

## 6. Surface Codes

### 6.1 Definition

The surface code is a topological error-correcting code defined on a 2D lattice of qubits. It is the most practical topological code for near-term implementation.

**Structure**:
- **Data qubits**: On the edges of a square lattice
- **X stabilizers**: Products of $X$ operators around each face (plaquette)
- **Z stabilizers**: Products of $Z$ operators around each vertex (star)

For an $L \times L$ lattice:
- Data qubits: $2L^2 - 2L + 1$
- Logical qubits: 1
- Code distance: $L$ (can correct $\lfloor(L-1)/2\rfloor$ errors)

### 6.2 Stabilizers

$$A_s = \prod_{i \in \text{star}(s)} X_i, \quad B_p = \prod_{i \in \text{plaquette}(p)} Z_i$$

All stabilizers commute: $[A_s, B_p] = 0$ for all $s, p$.

The code space is the $+1$ eigenspace of all stabilizers: $A_s|\psi\rangle = |\psi\rangle$, $B_p|\psi\rangle = |\psi\rangle$.

### 6.3 Error Detection

An $X$ error on qubit $i$ anticommutes with neighboring $Z$ stabilizers, flipping their eigenvalue to $-1$. By measuring all stabilizers, we obtain a **syndrome** that reveals the error locations (up to equivalence).

**Syndrome pattern**: Errors create pairs of "defects" (stabilizers with eigenvalue $-1$). The decoder must match defects into pairs and determine the correction.

### 6.4 Logical Operators

Logical operators are strings of Pauli operators that commute with all stabilizers but are not themselves stabilizers:

- **Logical $\bar{X}$**: A chain of $X$ operators crossing the lattice horizontally
- **Logical $\bar{Z}$**: A chain of $Z$ operators crossing the lattice vertically

Since these operators span the entire lattice ($L$ qubits), they cannot be caused by fewer than $L$ single-qubit errors.

### 6.5 Error Threshold

The surface code has a high error threshold:
- **Phenomenological threshold**: $\sim 3\%$ (with perfect syndrome measurements)
- **Circuit-level threshold**: $\sim 1\%$ (with realistic noisy syndrome extraction)
- **Current hardware**: approaching $10^{-3}$ error rates, within striking distance

### 6.6 Decoding

The **minimum-weight perfect matching** (MWPM) decoder is the standard:
1. Model the syndrome as a graph (defects are nodes, distances are edge weights)
2. Find the minimum-weight perfect matching
3. Apply corrections along the matching paths

More advanced decoders (union-find, neural network, tensor network) can approach the theoretical threshold more closely.

---

## 7. Topological Codes in Practice

### 7.1 Surface Code Operations

**State preparation**: Initialize all data qubits in $|0\rangle$, then measure stabilizers.

**Syndrome extraction**: Each stabilizer measurement requires an ancilla qubit and a sequence of CNOT gates. One round of syndrome extraction takes $O(1)$ depth.

**Logical gates**:
- $\bar{X}$, $\bar{Z}$: Transversal (apply Pauli string across lattice)
- $\bar{H}$: Requires lattice surgery or code deformation
- $\bar{S}$: Requires magic state injection
- $\bar{T}$: Requires magic state distillation (most expensive)

### 7.2 Lattice Surgery

Instead of braiding, the surface code implements logical gates through **lattice surgery**: merging and splitting code patches by measuring joint stabilizers along their boundaries.

- **Merge**: Join two code patches by measuring XX or ZZ stabilizers along the boundary → implements logical CNOT or CZ
- **Split**: Separate a merged patch by measuring single-qubit operators along the cut

### 7.3 Magic State Distillation

The T gate ($T = \text{diag}(1, e^{i\pi/4})$) cannot be implemented transversally in the surface code. Instead:

1. Prepare noisy T states: $|T\rangle = T|+\rangle = (|0\rangle + e^{i\pi/4}|1\rangle)/\sqrt{2}$
2. Use a distillation protocol that consumes many noisy T states to produce one high-fidelity T state
3. Inject the distilled T state into the computation

**Cost**: Each distilled T state requires $\sim 15$ noisy T states. Multiple rounds of distillation reduce the error exponentially but consume significant resources.

### 7.4 Resource Overhead

| Component | Overhead |
|-----------|----------|
| Physical qubits per logical qubit | $2L^2$ where $L \sim 10$-$30$ |
| Syndrome extraction cycles | $L$ per error correction round |
| T gate (distilled) | $\sim 15^d$ T states for $d$ distillation rounds |
| Total for useful computation | $10^6$-$10^8$ physical qubits |

---

## 8. Comparison with Conventional Approaches

### 8.1 Topological vs. Gate-Based

| Aspect | Topological QC | Gate-based QC |
|--------|---------------|---------------|
| Error protection | Built-in (topological) | Active (error correction) |
| Gate implementation | Braiding / lattice surgery | Microwave/laser pulses |
| Universality | Depends on anyon type | Universal by construction |
| Current status | Pre-experimental | 1000+ qubit processors |
| Scalability | Potentially excellent | Challenging (error rates) |
| Hardware | Exotic materials | Superconducting, ions, etc. |

### 8.2 Hybrid Approaches

In practice, the surface code (a topological code) is implemented on conventional hardware (superconducting qubits). This is a hybrid approach that combines:

- **Topological error correction** (surface code structure)
- **Conventional qubit hardware** (superconducting transmons)
- **Classical decoding** (MWPM, union-find)

---

## 9. Python Implementation

### 9.1 Fibonacci Anyon Braiding

```python
import numpy as np

def fibonacci_braid_matrices():
    """Compute the braid matrices for Fibonacci anyons.

    Fibonacci anyons have fusion rule tau x tau = 1 + tau.
    Three tau anyons span a 2D space (one qubit).
    The braid matrices are dense and generate a universal gate set.

    Returns:
        sigma1: Braid matrix for exchanging anyons 1,2
        sigma2: Braid matrix for exchanging anyons 2,3
    """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    # Braid phases
    theta_1 = np.exp(-4j * np.pi / 5)  # e^{-4*pi*i/5}
    theta_tau = np.exp(3j * np.pi / 5)  # e^{3*pi*i/5}

    # sigma1: diagonal (anyons 1,2 exchange)
    sigma1 = np.array([
        [theta_1, 0],
        [0, theta_tau]
    ], dtype=complex)

    # sigma2: non-diagonal (anyons 2,3 exchange)
    sigma2 = np.array([
        [theta_1 / phi, theta_tau / np.sqrt(phi)],
        [theta_tau / np.sqrt(phi), -theta_1 / phi]
    ], dtype=complex)

    return sigma1, sigma2


def braid_sequence(sigma1, sigma2, sequence):
    """Apply a sequence of braids.

    Args:
        sigma1, sigma2: Braid matrices
        sequence: String like '1122' meaning sigma1 sigma1 sigma2 sigma2
                  Use '-1' for inverse braids

    Returns:
        Total unitary matrix
    """
    U = np.eye(2, dtype=complex)
    sigma1_inv = np.linalg.inv(sigma1)
    sigma2_inv = np.linalg.inv(sigma2)

    ops = {'1': sigma1, '2': sigma2, '3': sigma1_inv, '4': sigma2_inv}

    for char in sequence:
        U = ops[char] @ U

    return U


def approximate_gate_by_braiding(target_gate, sigma1, sigma2, max_length=8):
    """Find a braid sequence that approximates a target single-qubit gate.

    Uses brute-force search over all braid sequences up to a given length.
    The Solovay-Kitaev theorem guarantees that O(log^c(1/epsilon))
    braids suffice for epsilon accuracy.

    Args:
        target_gate: 2x2 unitary to approximate
        sigma1, sigma2: Braid matrices
        max_length: Maximum braid sequence length

    Returns:
        best_sequence: Best braid sequence found
        best_error: Approximation error
        best_gate: The approximating unitary
    """
    sigma1_inv = np.linalg.inv(sigma1)
    sigma2_inv = np.linalg.inv(sigma2)

    generators = {
        '1': sigma1, '2': sigma2,
        '3': sigma1_inv, '4': sigma2_inv
    }

    best_error = float('inf')
    best_sequence = ''
    best_gate = np.eye(2, dtype=complex)

    # Normalize target (remove global phase)
    target_norm = target_gate / np.exp(1j * np.angle(np.linalg.det(target_gate)) / 2)

    from itertools import product as cart_product
    for length in range(1, max_length + 1):
        for seq_tuple in cart_product('1234', repeat=length):
            seq = ''.join(seq_tuple)
            U = np.eye(2, dtype=complex)
            for char in seq:
                U = generators[char] @ U

            # Normalize (remove global phase)
            U_norm = U / np.exp(1j * np.angle(np.linalg.det(U)) / 2)

            # Frobenius norm error
            error = np.linalg.norm(U_norm - target_norm)
            if error < best_error:
                best_error = error
                best_sequence = seq
                best_gate = U

    return best_sequence, best_error, best_gate


# Demonstrate Fibonacci anyon braiding
print("=" * 65)
print("Fibonacci Anyon Braiding")
print("=" * 65)

sigma1, sigma2 = fibonacci_braid_matrices()

print("\nBraid matrices:")
print(f"sigma1 =\n{sigma1}")
print(f"\nsigma2 =\n{sigma2}")

# Verify braid relation: sigma1 sigma2 sigma1 = sigma2 sigma1 sigma2
lhs = sigma1 @ sigma2 @ sigma1
rhs = sigma2 @ sigma1 @ sigma2
print(f"\nYang-Baxter relation satisfied: {np.allclose(lhs, rhs)}")

# Test some braid sequences
print(f"\n{'Sequence':>12} {'Trace':>20} {'Det':>20}")
print("-" * 55)

sequences = ['12', '1122', '121', '212', '112211', '12121', '1221']
for seq in sequences:
    U = braid_sequence(sigma1, sigma2, seq)
    print(f"{seq:>12} {np.trace(U):>20.4f} {np.linalg.det(U):>20.4f}")
```

### 9.2 Gate Approximation

```python
import numpy as np

def demonstrate_gate_approximation():
    """Show that Fibonacci anyons can approximate standard quantum gates."""
    print("=" * 65)
    print("Gate Approximation by Braiding")
    print("=" * 65)

    sigma1, sigma2 = fibonacci_braid_matrices()

    # Target gates
    H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    T_gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    X_gate = np.array([[0, 1], [1, 0]], dtype=complex)
    Z_gate = np.array([[1, 0], [0, -1]], dtype=complex)

    targets = {
        'Hadamard': H_gate,
        'T gate': T_gate,
        'X gate': X_gate,
        'Z gate': Z_gate,
    }

    for name, target in targets.items():
        seq, error, approx = approximate_gate_by_braiding(target, sigma1, sigma2,
                                                            max_length=7)
        print(f"\n{name}:")
        print(f"  Best sequence: {seq} (length {len(seq)})")
        print(f"  Error: {error:.6f}")
        print(f"  Process fidelity: {1 - error**2/4:.6f}")

demonstrate_gate_approximation()
```

### 9.3 Surface Code Simulation

```python
import numpy as np

class SurfaceCode:
    """Simulate a distance-L surface code.

    The surface code is defined on an L x L lattice with:
    - Data qubits on edges
    - X stabilizers on faces (plaquettes)
    - Z stabilizers on vertices (stars)

    For simplicity, we simulate the classical syndrome extraction
    and decoding, tracking Pauli errors on data qubits.
    """

    def __init__(self, L):
        """Initialize an L x L surface code.

        Args:
            L: Code distance (odd integer)
        """
        self.L = L
        self.n_data = 2 * L * L - 2 * L + 1  # Approximate
        # Use a simpler model: L x L grid with horizontal and vertical edges
        self.n_horizontal = L * (L - 1)  # Horizontal edges
        self.n_vertical = (L - 1) * L    # Vertical edges
        self.n_qubits = self.n_horizontal + self.n_vertical

        # Initialize error state (0 = no error, 1 = X error, 2 = Z error, 3 = Y error)
        self.x_errors = np.zeros(self.n_qubits, dtype=int)
        self.z_errors = np.zeros(self.n_qubits, dtype=int)

    def apply_noise(self, p, rng=None):
        """Apply independent depolarizing noise to each data qubit.

        Each qubit gets X error with prob p/3, Y with p/3, Z with p/3.
        """
        if rng is None:
            rng = np.random.default_rng()

        for q in range(self.n_qubits):
            r = rng.random()
            if r < p / 3:
                self.x_errors[q] ^= 1  # X error
            elif r < 2 * p / 3:
                self.x_errors[q] ^= 1  # Y = iXZ
                self.z_errors[q] ^= 1
            elif r < p:
                self.z_errors[q] ^= 1  # Z error

    def measure_z_syndrome(self):
        """Measure Z stabilizers (detect X errors).

        Z stabilizers are products of Z on edges around each vertex.
        An X error anticommutes with adjacent Z stabilizers.
        """
        L = self.L
        syndrome = []

        for row in range(L - 1):
            for col in range(L - 1):
                # Each interior vertex connects 4 edges
                parity = 0
                # Top horizontal edge
                h_idx = row * (L - 1) + col
                parity ^= self.x_errors[h_idx]
                # Bottom horizontal edge
                if row + 1 < L:
                    h_idx2 = (row + 1) * (L - 1) + col
                    if h_idx2 < self.n_horizontal:
                        parity ^= self.x_errors[h_idx2]
                # Left vertical edge
                v_idx = self.n_horizontal + col * (L - 1) + row
                if v_idx < self.n_qubits:
                    parity ^= self.x_errors[v_idx]
                # Right vertical edge
                v_idx2 = self.n_horizontal + (col + 1) * (L - 1) + row
                if v_idx2 < self.n_qubits:
                    parity ^= self.x_errors[v_idx2]

                syndrome.append(parity)

        return np.array(syndrome)

    def count_errors(self):
        """Count the number of X and Z errors."""
        return np.sum(self.x_errors), np.sum(self.z_errors)

    def has_logical_error(self):
        """Check if errors form a logical operator (chain spanning the lattice)."""
        L = self.L
        # Check if X errors form a horizontal chain
        # (simplified: check if any row of horizontal edges is fully errored)
        for row in range(L):
            h_start = row * (L - 1)
            h_end = h_start + (L - 1)
            if h_end <= self.n_horizontal:
                if np.sum(self.x_errors[h_start:h_end]) % 2 == 1:
                    # Check if it connects boundaries (simplified)
                    pass
        # Simplified check: count total X error parity
        return np.sum(self.x_errors) % 2 == 1 and np.sum(self.x_errors) > self.L // 2


def surface_code_threshold_simulation(L_values, p_values, n_trials=1000):
    """Estimate the logical error rate for different code distances and error rates.

    The threshold is the error rate p* where curves for different L cross:
    - For p < p*, larger L gives lower logical error rate
    - For p > p*, larger L gives higher logical error rate
    """
    results = {}

    for L in L_values:
        results[L] = []
        for p in p_values:
            n_logical_errors = 0
            rng = np.random.default_rng(42)

            for _ in range(n_trials):
                code = SurfaceCode(L)
                code.apply_noise(p, rng)
                if code.has_logical_error():
                    n_logical_errors += 1

            logical_error_rate = n_logical_errors / n_trials
            results[L].append(logical_error_rate)

    return results


# Demonstrate surface code
print("=" * 65)
print("Surface Code Simulation")
print("=" * 65)

# Basic properties
for L in [3, 5, 7]:
    code = SurfaceCode(L)
    print(f"\nDistance L = {L}:")
    print(f"  Data qubits: {code.n_qubits}")
    print(f"  Correctable errors: {(L-1)//2}")

# Threshold simulation
print("\nLogical error rate vs physical error rate:")
L_values = [3, 5, 7]
p_values = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]

results = surface_code_threshold_simulation(L_values, p_values, n_trials=500)

print(f"\n{'p':>8}", end="")
for L in L_values:
    print(f"  L={L:>2}", end="")
print()
print("-" * 35)

for i, p in enumerate(p_values):
    print(f"{p:8.3f}", end="")
    for L in L_values:
        print(f"  {results[L][i]:5.3f}", end="")
    print()
```

### 9.4 Toric Code

```python
import numpy as np

class ToricCode:
    """Simulate the toric code on an L x L torus.

    The toric code is a topological code with periodic boundary conditions.
    It encodes 2 logical qubits in L^2 + L^2 = 2L^2 physical qubits.

    Stabilizers:
    - Star (vertex) operators: product of X on edges adjacent to each vertex
    - Plaquette operators: product of Z on edges around each face
    """

    def __init__(self, L):
        self.L = L
        self.n_qubits = 2 * L * L  # L^2 horizontal + L^2 vertical edges

        # Error tracking
        self.x_errors = np.zeros(self.n_qubits, dtype=int)
        self.z_errors = np.zeros(self.n_qubits, dtype=int)

    def h_edge(self, row, col):
        """Index of horizontal edge at (row, col)."""
        return (row % self.L) * self.L + (col % self.L)

    def v_edge(self, row, col):
        """Index of vertical edge at (row, col)."""
        return self.L * self.L + (row % self.L) * self.L + (col % self.L)

    def star_operator_edges(self, row, col):
        """Edges adjacent to vertex (row, col)."""
        return [
            self.h_edge(row, col),          # right
            self.h_edge(row, col - 1),      # left
            self.v_edge(row, col),          # down
            self.v_edge(row - 1, col),      # up
        ]

    def plaquette_operator_edges(self, row, col):
        """Edges around face (row, col)."""
        return [
            self.h_edge(row, col),          # top
            self.h_edge(row + 1, col),      # bottom
            self.v_edge(row, col),          # left
            self.v_edge(row, col + 1),      # right
        ]

    def apply_noise(self, p, rng=None):
        """Apply iid depolarizing noise."""
        if rng is None:
            rng = np.random.default_rng()

        for q in range(self.n_qubits):
            r = rng.random()
            if r < p / 3:
                self.x_errors[q] ^= 1
            elif r < 2 * p / 3:
                self.x_errors[q] ^= 1
                self.z_errors[q] ^= 1
            elif r < p:
                self.z_errors[q] ^= 1

    def measure_star_syndrome(self):
        """Measure all star (vertex) operators to detect X errors."""
        L = self.L
        syndrome = np.zeros((L, L), dtype=int)
        for row in range(L):
            for col in range(L):
                parity = 0
                for edge in self.star_operator_edges(row, col):
                    parity ^= self.x_errors[edge]
                syndrome[row, col] = parity
        return syndrome

    def measure_plaquette_syndrome(self):
        """Measure all plaquette (face) operators to detect Z errors."""
        L = self.L
        syndrome = np.zeros((L, L), dtype=int)
        for row in range(L):
            for col in range(L):
                parity = 0
                for edge in self.plaquette_operator_edges(row, col):
                    parity ^= self.z_errors[edge]
                syndrome[row, col] = parity
        return syndrome


# Demonstrate toric code
print("=" * 65)
print("Toric Code Simulation")
print("=" * 65)

L = 5
code = ToricCode(L)
rng = np.random.default_rng(42)

print(f"\nToric code: L = {L}")
print(f"Physical qubits: {code.n_qubits}")
print(f"Logical qubits: 2")
print(f"Code distance: {L}")

# Apply noise and measure syndrome
code.apply_noise(0.05, rng)
star_syn = code.measure_star_syndrome()
plaq_syn = code.measure_plaquette_syndrome()

n_x_err, n_z_err = np.sum(code.x_errors), np.sum(code.z_errors)
n_star_defects = np.sum(star_syn)
n_plaq_defects = np.sum(plaq_syn)

print(f"\nAfter noise (p=0.05):")
print(f"  X errors: {n_x_err}")
print(f"  Z errors: {n_z_err}")
print(f"  Star defects: {n_star_defects} (always even)")
print(f"  Plaquette defects: {n_plaq_defects} (always even)")

print(f"\nStar syndrome (X error detection):")
for row in range(L):
    print(f"  ", end="")
    for col in range(L):
        print(f"{'*' if star_syn[row, col] else '.'}", end=" ")
    print()
```

### 9.5 Majorana Chain Simulation

```python
import numpy as np
from scipy.linalg import expm

def kitaev_chain(L, t_hop=1.0, delta=1.0, mu=0.0):
    """Build the Kitaev chain Hamiltonian.

    H = -mu * sum_i n_i - t * sum_i (c^dag_i c_{i+1} + h.c.)
        + delta * sum_i (c_i c_{i+1} + h.c.)

    In the topological phase (|mu| < 2t), Majorana zero modes
    appear at the chain ends. These are the building blocks of
    topological qubits.

    Args:
        L: Chain length (number of sites)
        t_hop: Hopping amplitude
        delta: Superconducting pairing
        mu: Chemical potential

    Returns:
        H_bdg: BdG Hamiltonian (2L x 2L)
        energies: Energy eigenvalues
    """
    # Bogoliubov-de Gennes (BdG) Hamiltonian
    # In Nambu basis (c_1, c_2, ..., c_L, c^dag_1, ..., c^dag_L)
    H_bdg = np.zeros((2 * L, 2 * L), dtype=complex)

    # On-site (chemical potential)
    for i in range(L):
        H_bdg[i, i] = -mu / 2
        H_bdg[L + i, L + i] = mu / 2

    # Hopping
    for i in range(L - 1):
        H_bdg[i, i + 1] = -t_hop
        H_bdg[i + 1, i] = -t_hop
        H_bdg[L + i, L + i + 1] = t_hop
        H_bdg[L + i + 1, L + i] = t_hop

    # Pairing
    for i in range(L - 1):
        H_bdg[i, L + i + 1] = delta
        H_bdg[i + 1, L + i] = -delta
        H_bdg[L + i + 1, i] = delta.conjugate()
        H_bdg[L + i, i + 1] = -delta.conjugate()

    energies = np.linalg.eigvalsh(H_bdg)
    return H_bdg, np.sort(energies)


def find_majorana_modes(H_bdg, L, threshold=0.1):
    """Identify Majorana zero modes from the BdG spectrum.

    In the topological phase, two eigenvalues are exponentially
    close to zero. The corresponding eigenvectors are localized
    at the chain ends - these are the Majorana zero modes.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H_bdg)

    # Find near-zero eigenvalues
    zero_mode_indices = np.where(np.abs(eigenvalues) < threshold)[0]

    if len(zero_mode_indices) >= 2:
        # The two zero modes
        mode1 = eigenvectors[:, zero_mode_indices[0]]
        mode2 = eigenvectors[:, zero_mode_indices[1]]

        # Localization: compute weight on each site
        loc1 = np.abs(mode1[:L]) ** 2 + np.abs(mode1[L:]) ** 2
        loc2 = np.abs(mode2[:L]) ** 2 + np.abs(mode2[L:]) ** 2

        return eigenvalues[zero_mode_indices], loc1, loc2

    return None, None, None


# Demonstrate Kitaev chain
print("=" * 65)
print("Kitaev Chain: Majorana Zero Modes")
print("=" * 65)

L = 20

# Phase diagram: vary mu
print(f"\nChain length L = {L}")
print(f"\n{'mu':>6} {'Lowest |E|':>14} {'Phase':>20}")
print("-" * 44)

for mu in np.linspace(-3, 3, 13):
    _, energies = kitaev_chain(L, t_hop=1.0, delta=1.0, mu=mu)
    min_E = np.min(np.abs(energies))
    phase = "Topological" if abs(mu) < 2.0 else "Trivial"
    marker = " <-- MZM!" if min_E < 0.01 else ""
    print(f"{mu:6.2f} {min_E:14.6f} {phase:>20}{marker}")

# Localization of zero modes
print(f"\nMajorana zero mode localization (mu=0, topological phase):")
H_bdg, _ = kitaev_chain(L, t_hop=1.0, delta=1.0, mu=0.0)
energies_zm, loc1, loc2 = find_majorana_modes(H_bdg, L)

if loc1 is not None:
    print(f"  Zero mode energies: {energies_zm}")
    print(f"\n  {'Site':>6} {'Mode 1':>10} {'Mode 2':>10}")
    print(f"  {'-' * 28}")
    for i in range(L):
        if loc1[i] > 0.01 or loc2[i] > 0.01:
            print(f"  {i:6d} {loc1[i]:10.4f} {loc2[i]:10.4f}")
    print(f"\n  Mode 1 concentrated at left end, Mode 2 at right end")
    print(f"  → Non-local encoding protects the qubit from local noise")
```

---

## 10. Exercises

### Exercise 1: Anyon Fusion Rules

For Fibonacci anyons ($\tau \times \tau = 1 + \tau$):
(a) Compute the dimension of the fusion space for $n = 3, 4, 5, 6, 7$ anyons of type $\tau$. Verify the Fibonacci number pattern.
(b) How many qubits can be encoded in $n$ Fibonacci anyons? What is the encoding efficiency as $n \to \infty$?
(c) For Ising anyons ($\sigma \times \sigma = 1 + \psi$), compute the fusion space dimension for $n = 4, 6, 8$ $\sigma$ anyons.

### Exercise 2: Braid Compilation

(a) Find the shortest braid sequence (using Fibonacci anyons) that approximates the Hadamard gate to error $< 0.1$.
(b) Find a sequence for the T gate with error $< 0.1$.
(c) Combine these to approximate the circuit $HTH$. What is the total braid length and error?
(d) How does the approximation error decrease as you allow longer braid sequences? Plot error vs. sequence length.

### Exercise 3: Surface Code Decoding

Implement a simple decoder for the surface code:
(a) For a distance-3 code, enumerate all single-qubit errors and their syndromes.
(b) Implement a lookup table decoder that corrects all weight-1 errors.
(c) What fraction of weight-2 errors can the decoder correct? Which weight-2 errors cause logical errors?
(d) Implement the minimum-weight perfect matching decoder for distance 5.

### Exercise 4: Topological Phase Transition

For the Kitaev chain:
(a) Plot the energy gap (minimum excitation energy) as a function of $\mu$ for $L = 10, 20, 50$.
(b) Locate the phase transition at $|\mu| = 2t$. How does the gap close as $L \to \infty$?
(c) In the topological phase, verify that the MZM energy splitting decreases exponentially with $L$.
(d) Add disorder: random on-site potentials $\mu_i = \mu + W\xi_i$ where $\xi_i \in [-1, 1]$. At what disorder strength $W$ do the MZMs disappear?

### Exercise 5: Lattice Surgery

Simulate the merge operation for two distance-3 surface codes:
(a) Model two code patches side by side, each encoding one logical qubit.
(b) Implement the merge operation by measuring XX stabilizers along the boundary.
(c) Verify that the merged code encodes one logical qubit with distance 3.
(d) What is the probability of a logical error during the merge, as a function of the physical error rate?

---

[← Previous: Quantum Chemistry](21_Quantum_Chemistry.md) | [Next: Quantum Networking →](23_Quantum_Networking.md)
