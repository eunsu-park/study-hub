# Lesson 18: Quantum Simulation

[← Previous: Quantum Computing Landscape and Future](16_Landscape_and_Future.md) | [Next: Quantum Walks →](19_Quantum_Walks.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the concept of Hamiltonian simulation and why quantum computers are natural simulators
2. Derive and implement Trotter-Suzuki decomposition for time evolution
3. Analyze the error bounds of product formulas at different orders
4. Describe advances in variational quantum eigensolver (VQE) beyond the basics
5. Apply quantum phase estimation (QPE) for energy eigenvalue extraction
6. Compare different simulation approaches: product formulas, LCU, qubitization
7. Implement Hamiltonian simulation algorithms in Python/NumPy

---

Quantum simulation is arguably the most promising near-term application of quantum computing. Richard Feynman's original 1982 insight was precisely this: simulating quantum systems with classical computers is exponentially hard, but a quantum computer could do it naturally. A system of $n$ interacting quantum particles has a state space of dimension $2^n$, making classical simulation intractable for even modest sizes (e.g., $n \geq 40$). A quantum computer with $n$ qubits, however, can represent this state directly.

The central problem of quantum simulation is: given a Hamiltonian $H$ describing a quantum system, compute the time evolution operator $U(t) = e^{-iHt}$ and apply it to an initial state $|\psi_0\rangle$. This lesson covers the key algorithms for achieving this, from Trotter-Suzuki product formulas to modern techniques like linear combination of unitaries (LCU) and quantum signal processing.

> **Analogy:** Simulating quantum systems classically is like trying to track every possible move in a chess game by writing them all down — the combinatorial explosion is overwhelming. A quantum simulator, by contrast, plays the game directly, using its own quantum nature to mirror the system being studied.

## Table of Contents

1. [Why Quantum Simulation?](#1-why-quantum-simulation)
2. [Hamiltonian Simulation Problem](#2-hamiltonian-simulation-problem)
3. [Trotter-Suzuki Decomposition](#3-trotter-suzuki-decomposition)
4. [Higher-Order Product Formulas](#4-higher-order-product-formulas)
5. [Quantum Phase Estimation for Simulation](#5-quantum-phase-estimation-for-simulation)
6. [VQE Advances](#6-vqe-advances)
7. [Beyond Product Formulas](#7-beyond-product-formulas)
8. [Applications](#8-applications)
9. [Python Implementation](#9-python-implementation)
10. [Exercises](#10-exercises)

---

## 1. Why Quantum Simulation?

### 1.1 Feynman's Vision

In 1982, Richard Feynman observed that simulating a quantum system of $n$ particles on a classical computer requires storing and manipulating $2^n$ complex amplitudes. For $n = 50$ particles, this requires $2^{50} \approx 10^{15}$ complex numbers — roughly 16 petabytes of memory. A quantum computer with 50 qubits naturally represents this state in its quantum register.

### 1.2 Classical Intractability

Consider a system of $n$ spin-1/2 particles with pairwise interactions:

$$H = \sum_{i<j} J_{ij} \vec{\sigma}_i \cdot \vec{\sigma}_j + \sum_i h_i \sigma_i^z$$

The Hilbert space dimension is $2^n$. Classical exact diagonalization requires $O(2^{3n})$ time and $O(2^{2n})$ memory. Even approximate classical methods (DMRG, tensor networks) fail for certain systems — particularly those with strong entanglement, frustrated interactions, or high dimensionality.

### 1.3 Types of Quantum Simulation

| Type | Description | Example |
|------|-------------|---------|
| **Digital** | Gate-based quantum circuit approximating $e^{-iHt}$ | Trotter-Suzuki simulation |
| **Analog** | Directly engineering a Hamiltonian that mimics the target | Cold atom lattices |
| **Variational** | Parameterized circuit optimized to approximate ground state | VQE, QAOA |
| **Hybrid** | Combining quantum and classical resources | Quantum-classical feedback loops |

This lesson focuses primarily on digital quantum simulation.

---

## 2. Hamiltonian Simulation Problem

### 2.1 Problem Statement

**Given**: A Hamiltonian $H$ acting on $n$ qubits and a time $t$

**Goal**: Implement the unitary $U(t) = e^{-iHt}$ on a quantum computer

**Challenge**: $H$ is typically a sum of non-commuting terms: $H = \sum_{k=1}^{L} H_k$, and $e^{-i(A+B)t} \neq e^{-iAt}e^{-iBt}$ when $[A, B] \neq 0$.

### 2.2 Hamiltonian Decomposition

Most physical Hamiltonians decompose naturally into a sum of local terms:

$$H = \sum_{k=1}^{L} \alpha_k P_k$$

where each $P_k$ is a tensor product of Pauli operators (Pauli string) and $\alpha_k$ are real coefficients. For example, the Heisenberg model on a 1D chain:

$$H = J\sum_{i=1}^{n-1} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}) + h\sum_{i=1}^{n} Z_i$$

This has $L = 4(n-1) + n$ terms. Each individual term $e^{-i\alpha_k P_k t}$ can be implemented efficiently with quantum gates (at most $O(n)$ gates for an $n$-qubit Pauli string).

### 2.3 Simulation Complexity

The key question is: how many quantum gates are needed to simulate $e^{-iHt}$ to accuracy $\epsilon$?

| Method | Gate complexity |
|--------|----------------|
| First-order Trotter | $O(L^2 t^2 / \epsilon)$ |
| Second-order Trotter | $O(L^{5/2} t^{3/2} / \epsilon^{1/2})$ |
| $2k$-th order Trotter | $O(L^2 (Lt)^{1+1/2k} / \epsilon^{1/2k})$ |
| Linear Combination of Unitaries | $O(L\alpha t \cdot \text{polylog}(1/\epsilon))$ |
| Quantum Signal Processing | $O(\alpha t + \log(1/\epsilon))$ |

where $\alpha = \sum_k |\alpha_k|$ is the 1-norm of the Hamiltonian coefficients.

---

## 3. Trotter-Suzuki Decomposition

### 3.1 First-Order Trotter Formula

The Lie-Trotter product formula approximates the exponential of a sum:

$$e^{-i(H_1 + H_2 + \cdots + H_L)t} \approx \left(\prod_{k=1}^{L} e^{-iH_k t/r}\right)^r$$

where $r$ is the number of Trotter steps. The error per step is:

$$\left\|e^{-i(A+B)\delta t} - e^{-iA\delta t}e^{-iB\delta t}\right\| \leq \frac{(\delta t)^2}{2}\|[A, B]\|$$

where $\delta t = t/r$. The total error scales as:

$$\epsilon_{\text{total}} = O\left(\frac{L^2 \Lambda^2 t^2}{r}\right)$$

where $\Lambda = \max_k \|H_k\|$. To achieve error $\leq \epsilon$, we need $r = O(L^2 \Lambda^2 t^2 / \epsilon)$ Trotter steps.

### 3.2 Why Trotterization Works

The key insight is that while each Trotter step introduces an error of $O(\delta t^2)$, the errors do not simply accumulate linearly. For many physical systems, error cancellation occurs due to the structure of the Hamiltonian, leading to much better performance than the worst-case bounds suggest.

**Geometric picture**: In the space of unitaries, the Trotter product traces a path that zigzags around the true evolution path. Each zig-zag deviates by $O(\delta t^2)$, but over $r$ steps the total deviation is controlled by the commutator structure.

### 3.3 Circuit Construction

To implement $e^{-i\alpha P t}$ where $P = \sigma_{i_1} \otimes \sigma_{i_2} \otimes \cdots \otimes \sigma_{i_m}$ is a Pauli string:

1. **Basis rotation**: Apply single-qubit gates to rotate each qubit into the Z basis
   - $X_i$: apply $H_i$ (Hadamard)
   - $Y_i$: apply $R_x(\pi/2)_i$
   - $Z_i$: no gate needed
   - $I_i$: skip this qubit

2. **Parity computation**: Apply CNOT ladder to compute parity into an ancilla qubit

3. **Phase rotation**: Apply $R_z(2\alpha t)$ on the target qubit

4. **Uncompute**: Reverse the CNOT ladder and basis rotations

This requires $O(m)$ CNOT gates and $O(m)$ single-qubit gates per Pauli string.

### 3.4 Circuit for $e^{-i\alpha Z_i Z_j t}$

The following circuit implements the two-qubit ZZ interaction term $e^{-i\alpha Z_i Z_j t}$:

```
  q_i ──●────────────────●──
        |                |
  q_j ──⊕── Rz(2αt) ────⊕──

  Step-by-step:
  1. CNOT(q_i → q_j): computes parity of q_i, q_j into q_j
  2. Rz(2αt) on q_j:   applies phase based on parity
  3. CNOT(q_i → q_j): uncomputes the parity (restores q_j)

  Net effect: |ab⟩ → e^{-iαt(-1)^(a⊕b)} |ab⟩
            = e^{-iαt·Z_i·Z_j} |ab⟩
```

This pattern generalizes to longer Pauli strings $Z_i Z_j Z_k \cdots$: extend the CNOT ladder to compute the multi-qubit parity, apply $R_z$, then reverse the ladder.

### 3.5 Example: Ising Model

For the transverse-field Ising model:

$$H = -J\sum_{i=1}^{n-1} Z_i Z_{i+1} - h\sum_{i=1}^{n} X_i$$

Each Trotter step implements:

$$\prod_{i=1}^{n-1} e^{iJ\delta t Z_i Z_{i+1}} \cdot \prod_{i=1}^{n} e^{ih\delta t X_i}$$

The $ZZ$ terms require CNOT + $R_z$ + CNOT patterns, and the $X$ terms are simple $R_x$ rotations.

---

## 4. Higher-Order Product Formulas

### 4.1 Second-Order (Symmetrized) Trotter

The Suzuki-Trotter second-order formula symmetrizes the product:

$$S_2(\delta t) = \prod_{k=1}^{L} e^{-iH_k \delta t/2} \cdot \prod_{k=L}^{1} e^{-iH_k \delta t/2}$$

The error per step improves to:

$$\left\|e^{-iH\delta t} - S_2(\delta t)\right\| = O(\delta t^3)$$

This means the total error for $r$ steps is $O(t^3/r^2)$, requiring $r = O(t^{3/2}/\epsilon^{1/2})$ steps.

### 4.2 Higher-Order Suzuki Formulas

Suzuki's recursive construction builds $(2k)$-th order formulas from $(2k-2)$-th order:

$$S_{2k}(\delta t) = S_{2k-2}(p_k \delta t)^2 \cdot S_{2k-2}((1 - 4p_k)\delta t) \cdot S_{2k-2}(p_k \delta t)^2$$

where $p_k = (4 - 4^{1/(2k-1)})^{-1}$.

| Order | Error per step | Total Trotter steps for error $\epsilon$ |
|-------|---------------|------------------------------------------|
| 1st | $O(\delta t^2)$ | $O(t^2/\epsilon)$ |
| 2nd | $O(\delta t^3)$ | $O(t^{3/2}/\epsilon^{1/2})$ |
| 4th | $O(\delta t^5)$ | $O(t^{5/4}/\epsilon^{1/4})$ |
| $2k$-th | $O(\delta t^{2k+1})$ | $O(t^{1+1/2k}/\epsilon^{1/2k})$ |

### 4.3 Tradeoff: Order vs. Circuit Depth

Higher-order formulas reduce the number of Trotter steps but increase the number of exponentials per step:

- 1st order: $L$ exponentials per step
- 2nd order: $2L$ exponentials per step
- 4th order: $10L$ exponentials per step (5 copies of $S_2$)
- $2k$-th order: $5^{k-1} \cdot 2L$ exponentials per step

The total gate count is minimized at an optimal order that depends on $L$, $t$, and $\epsilon$.

### 4.4 Randomized Product Formulas

A recent development is **randomized Trotter** (qDRIFT):

$$U_{\text{qDRIFT}} = \prod_{j=1}^{N} e^{-i\lambda\tau H_{k_j}}$$

where each $k_j$ is sampled randomly with probability $p_k = |\alpha_k|/\lambda$ and $\lambda = \sum_k |\alpha_k|$. The error scales as $O(\lambda^2 t^2/N)$, independent of $L$ — a significant advantage for Hamiltonians with many terms but small norm.

---

## 5. Quantum Phase Estimation for Simulation

### 5.1 QPE Review

Quantum phase estimation (see Lesson 09) extracts eigenvalues of a unitary $U$. Given $U|\psi\rangle = e^{2\pi i\phi}|\psi\rangle$, QPE outputs $\phi$ to $m$ bits of precision using:

- $m$ ancilla qubits
- Controlled-$U^{2^j}$ operations for $j = 0, 1, \ldots, m-1$
- Inverse QFT on the ancilla register

### 5.2 Energy Estimation via QPE

For Hamiltonian simulation, set $U = e^{-iHt}$. If $|\psi\rangle$ is an eigenstate of $H$ with energy $E$:

$$e^{-iHt}|\psi\rangle = e^{-iEt}|\psi\rangle$$

QPE extracts $\phi = Et/(2\pi)$, giving $E = 2\pi\phi/t$.

**Precision**: With $m$ ancilla qubits, the energy precision is $\Delta E = 2\pi/(2^m t)$.

**Total cost**: To estimate $E$ to precision $\Delta E$, we need $t = O(1/\Delta E)$ and $m = O(\log(1/\Delta E))$ qubits. The dominant cost is implementing controlled-$e^{-iHt}$ and controlled-$e^{-iH \cdot 2^j t}$.

### 5.3 Initial State Preparation

QPE requires the input state to have significant overlap with the target eigenstate. For ground state energy estimation, we need $|\langle E_0|\psi_0\rangle|^2 = \Omega(1)$. Common strategies:

- **Hartree-Fock state**: Good overlap for weakly correlated systems
- **Adiabatic state preparation**: Start from a simple Hamiltonian and slowly evolve to the target
- **VQE-prepared state**: Use VQE to get a good approximation, then refine with QPE

### 5.4 QPE vs. VQE

| Aspect | QPE | VQE |
|--------|-----|-----|
| Circuit depth | Deep (requires controlled-$U^{2^m}$) | Shallow (variational ansatz) |
| Precision | Systematic ($\Delta E \sim 2^{-m}$) | Limited by optimization |
| Noise tolerance | Requires error correction | NISQ-compatible |
| Initial state | Needs good overlap | Optimizes from scratch |
| Classical cost | Minimal | Heavy optimization loop |

---

## 6. VQE Advances

### 6.1 Beyond Basic VQE (Lesson 13 Review)

Lesson 13 covered the basics of VQE: variational principle, simple ansatze, parameter optimization. Here we explore recent advances that address VQE's limitations.

### 6.2 Adaptive VQE (ADAPT-VQE)

Instead of fixing the ansatz structure before optimization, ADAPT-VQE grows the circuit iteratively:

1. Start with a reference state $|\psi_0\rangle$ (e.g., Hartree-Fock)
2. Compute the gradient $\partial E/\partial \theta_k$ for each operator $A_k$ in an operator pool
3. Add the operator with the largest gradient to the ansatz
4. Re-optimize all parameters
5. Repeat until the gradient norm falls below a threshold

**Advantages**: More compact circuits, avoids barren plateaus, converges faster
**Operator pool**: Typically single and double excitation operators from the unitary coupled cluster (UCC) framework

### 6.3 Error Mitigation Techniques

Since VQE runs on noisy hardware, error mitigation is essential:

**Zero-noise extrapolation (ZNE)**:
1. Run the circuit at noise levels $\lambda_1 < \lambda_2 < \lambda_3$ (e.g., by stretching pulse durations)
2. Fit the results to a model (linear, exponential, polynomial)
3. Extrapolate to zero noise

**Probabilistic error cancellation (PEC)**:
1. Characterize the noise channel $\mathcal{N}$
2. Decompose $\mathcal{N}^{-1}$ as a linear combination of implementable operations
3. Sample operations according to the decomposition
4. Average results with appropriate signs and weights

**Symmetry verification**:
1. The target state should satisfy certain symmetries (e.g., particle number, spin)
2. Measure symmetry operators and post-select on correct symmetry sector
3. This eliminates errors that violate symmetries

### 6.4 Measurement Optimization

VQE requires estimating $\langle H \rangle = \sum_k \alpha_k \langle P_k \rangle$ for many Pauli strings. Measurement optimization reduces the number of distinct measurement circuits:

- **Qubit-wise commutativity**: Group Pauli strings that can be measured simultaneously (e.g., $ZZI$ and $ZIZ$ commute qubit-wise)
- **General commutativity**: Group Pauli strings into commuting cliques using graph coloring
- **Classical shadows**: Use randomized measurements to estimate all expectation values from a fixed number of measurement circuits

---

## 7. Beyond Product Formulas

### 7.1 Linear Combination of Unitaries (LCU)

Express $e^{-iHt}$ as a linear combination:

$$e^{-iHt} \approx \sum_{j} c_j U_j$$

where each $U_j$ is a unitary that can be implemented efficiently. The LCU technique uses:

1. A **prepare** oracle to create $\sum_j \sqrt{|c_j|/\lambda} |j\rangle$
2. A **select** oracle to apply $U_j$ conditioned on $|j\rangle$
3. **Oblivious amplitude amplification** to boost the success probability

Gate complexity: $O(\lambda t \cdot \text{polylog}(1/\epsilon))$ where $\lambda = \sum_j |c_j|$.

### 7.2 Quantum Signal Processing (QSP)

QSP achieves optimal Hamiltonian simulation by implementing polynomial transformations of a block-encoded Hamiltonian. Given a block encoding of $H/\alpha$ (where $\alpha \geq \|H\|$), QSP can implement any bounded polynomial $p(H/\alpha)$ using $O(\deg(p))$ queries.

For time evolution, $p(x) \approx e^{-i\alpha t x}$ can be approximated by a polynomial of degree $O(\alpha t + \log(1/\epsilon))$, giving the optimal complexity.

### 7.3 Comparison Summary

| Method | Queries to $H$ | Dependence on $\epsilon$ | Optimal? |
|--------|----------------|--------------------------|----------|
| 1st-order Trotter | $O(\alpha^2 t^2/\epsilon)$ | $1/\epsilon$ | No |
| 2nd-order Trotter | $O(\alpha^{5/2} t^{3/2}/\epsilon^{1/2})$ | $1/\epsilon^{1/2}$ | No |
| LCU (Taylor) | $O(\alpha t \cdot \text{polylog}(1/\epsilon))$ | polylog | Nearly |
| QSP/QSVT | $O(\alpha t + \log(1/\epsilon))$ | $\log(1/\epsilon)$ | Yes |

---

## 8. Applications

### 8.1 Condensed Matter Physics

- **Hubbard model**: Simulating strongly correlated electron systems relevant to high-temperature superconductivity
- **Spin chains**: Studying quantum phase transitions, many-body localization
- **Topological phases**: Detecting topological order and anyonic excitations

### 8.2 Quantum Chemistry

Overlap with Lesson 21 (Quantum Chemistry):

- **Molecular ground states**: Computing ground state energies beyond classical tractability
- **Reaction dynamics**: Simulating chemical reactions in real time
- **Excited states**: Calculating absorption spectra and photochemical pathways

### 8.3 High-Energy Physics

- **Lattice gauge theories**: Simulating quantum chromodynamics on a lattice
- **Real-time dynamics**: Studying thermalization, quark-gluon plasma
- **Particle scattering**: Computing scattering amplitudes

### 8.4 Materials Science

- **Battery materials**: Simulating electrochemical processes at the quantum level
- **Catalysts**: Understanding catalytic mechanisms in enzymes and industrial catalysts
- **Superconductors**: Predicting properties of new superconducting materials

---

## 9. Python Implementation

### 9.1 Trotter-Suzuki Simulation

```python
import numpy as np
from scipy.linalg import expm

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def kron_list(ops):
    """Compute tensor product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def build_ising_hamiltonian(n_qubits, J=1.0, h=0.5):
    """Build the transverse-field Ising model Hamiltonian.

    H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i

    This is one of the most studied models in quantum physics.
    The competition between the ZZ interaction (favoring aligned spins)
    and the transverse field X (favoring superposition) produces
    a quantum phase transition at J = h.

    Args:
        n_qubits: Number of spins in the chain
        J: Nearest-neighbor interaction strength
        h: Transverse field strength

    Returns:
        H: Full Hamiltonian matrix (2^n x 2^n)
        terms: List of (coefficient, matrix) pairs for Trotterization
    """
    N = 2 ** n_qubits
    H = np.zeros((N, N), dtype=complex)
    terms = []

    # ZZ interaction terms
    for i in range(n_qubits - 1):
        ops = [I] * n_qubits
        ops[i] = Z
        ops[i + 1] = Z
        term = -J * kron_list(ops)
        H += term
        terms.append((-J, kron_list(ops) / (-J)))  # (coefficient, Pauli operator)

    # Transverse field terms
    for i in range(n_qubits):
        ops = [I] * n_qubits
        ops[i] = X
        term = -h * kron_list(ops)
        H += term
        terms.append((-h, kron_list(ops) / (-h)))

    return H, terms


def trotter_step_first_order(terms, dt):
    """Compute one first-order Trotter step.

    First-order Trotter: prod_k exp(-i * c_k * H_k * dt)

    Each term contributes a unitary that can be computed independently.
    The approximation error per step is O(dt^2 * ||[H_j, H_k]||).

    Args:
        terms: List of (coefficient, operator) pairs
        dt: Time step size

    Returns:
        Unitary matrix for one Trotter step
    """
    N = terms[0][1].shape[0]
    U = np.eye(N, dtype=complex)
    for coeff, op in terms:
        U = expm(-1j * coeff * op * dt) @ U
    return U


def trotter_step_second_order(terms, dt):
    """Compute one second-order (symmetrized) Trotter step.

    S2(dt) = prod_k exp(-i*c_k*H_k*dt/2) * prod_k_reverse exp(-i*c_k*H_k*dt/2)

    The symmetrization cancels the first-order error term, improving
    the error to O(dt^3). This is the workhorse of practical Trotter simulation.
    """
    N = terms[0][1].shape[0]
    U = np.eye(N, dtype=complex)

    # Forward sweep (half step)
    for coeff, op in terms:
        U = expm(-1j * coeff * op * dt / 2) @ U

    # Backward sweep (half step)
    for coeff, op in reversed(terms):
        U = expm(-1j * coeff * op * dt / 2) @ U

    return U


def trotter_simulation(H, terms, t_total, n_steps, order=1):
    """Simulate time evolution using Trotter-Suzuki decomposition.

    Compares the Trotter approximation with the exact evolution
    to measure the approximation quality.

    Args:
        H: Full Hamiltonian matrix
        terms: List of (coefficient, operator) pairs
        t_total: Total simulation time
        n_steps: Number of Trotter steps
        order: Trotter order (1 or 2)

    Returns:
        U_trotter: Approximate unitary
        U_exact: Exact unitary
        error: Operator norm of the difference
    """
    dt = t_total / n_steps
    N = H.shape[0]

    # Build Trotter unitary
    if order == 1:
        U_step = trotter_step_first_order(terms, dt)
    else:
        U_step = trotter_step_second_order(terms, dt)

    U_trotter = np.linalg.matrix_power(U_step, n_steps)

    # Exact evolution
    U_exact = expm(-1j * H * t_total)

    # Error (operator norm)
    error = np.linalg.norm(U_trotter - U_exact, ord=2)

    return U_trotter, U_exact, error


# Demonstration: Trotter simulation of the Ising model
print("=" * 65)
print("Trotter-Suzuki Simulation of Transverse-Field Ising Model")
print("=" * 65)

n_qubits = 3
J, h = 1.0, 0.5
H, terms = build_ising_hamiltonian(n_qubits, J, h)

print(f"\nSystem: {n_qubits}-qubit Ising chain, J={J}, h={h}")
print(f"Hamiltonian dimension: {2**n_qubits} x {2**n_qubits}")
print(f"Number of terms: {len(terms)}")
print(f"Eigenvalues: {np.sort(np.real(np.linalg.eigvalsh(H)))[:4]} ...")

# Compare orders and step counts
t_total = 2.0
print(f"\nSimulation time: t = {t_total}")
print(f"\n{'Steps':>8} {'Order':>6} {'Error':>14} {'Error/step':>14}")
print("-" * 46)

for order in [1, 2]:
    for n_steps in [5, 10, 20, 50, 100]:
        _, _, error = trotter_simulation(H, terms, t_total, n_steps, order)
        print(f"{n_steps:8d} {order:6d} {error:14.2e} {error/n_steps:14.2e}")
    print()
```

### 9.2 Time Evolution and Observables

```python
import numpy as np
from scipy.linalg import expm

def evolve_state(H, terms, psi0, t_total, n_steps, order=2):
    """Evolve a quantum state under Hamiltonian H using Trotter decomposition.

    Args:
        H: Full Hamiltonian (for reference)
        terms: Trotter terms
        psi0: Initial state vector
        t_total: Total evolution time
        n_steps: Number of Trotter steps
        order: Trotter order (1 or 2)

    Returns:
        psi_trotter: State after Trotter evolution
        psi_exact: State after exact evolution
    """
    dt = t_total / n_steps
    psi = psi0.copy()

    for _ in range(n_steps):
        if order == 1:
            for coeff, op in terms:
                psi = expm(-1j * coeff * op * dt) @ psi
        else:
            for coeff, op in terms:
                psi = expm(-1j * coeff * op * dt / 2) @ psi
            for coeff, op in reversed(terms):
                psi = expm(-1j * coeff * op * dt / 2) @ psi

    psi_exact = expm(-1j * H * t_total) @ psi0
    return psi, psi_exact


def measure_magnetization(psi, n_qubits):
    """Measure the average Z magnetization <M_z> = (1/n) sum_i <Z_i>.

    Magnetization tells us about the ordering of spins.
    In the Ising model, high |<M_z>| indicates ferromagnetic order,
    while <M_z> ~ 0 indicates paramagnetic (disordered) phase.
    """
    N = 2 ** n_qubits
    mz = 0.0
    for i in range(n_qubits):
        ops = [I] * n_qubits
        ops[i] = Z
        Z_i = kron_list(ops)
        mz += np.real(psi.conj() @ Z_i @ psi)
    return mz / n_qubits


def measure_entanglement_entropy(psi, n_qubits, partition):
    """Compute entanglement entropy for a bipartition.

    The entanglement entropy S = -Tr(rho_A log rho_A) quantifies
    the quantum correlations between subsystem A and its complement.
    High entropy indicates strong entanglement.
    """
    N = 2 ** n_qubits
    n_A = len(partition)
    n_B = n_qubits - n_A
    complement = [q for q in range(n_qubits) if q not in partition]

    # Reshape state into bipartite form
    psi_matrix = psi.reshape([2] * n_qubits)

    # Reorder axes: partition qubits first, complement second
    order = list(partition) + complement
    psi_reordered = np.transpose(psi_matrix, order)
    psi_bipartite = psi_reordered.reshape(2 ** n_A, 2 ** n_B)

    # Reduced density matrix
    rho_A = psi_bipartite @ psi_bipartite.conj().T

    # Von Neumann entropy
    eigenvalues = np.real(np.linalg.eigvalsh(rho_A))
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))

    return entropy


# Demonstrate time evolution
print("=" * 65)
print("Time Evolution of Ising Model")
print("=" * 65)

n_qubits = 4
J, h = 1.0, 0.5
H, terms = build_ising_hamiltonian(n_qubits, J, h)

# Initial state: all spins up |0000⟩
N = 2 ** n_qubits
psi0 = np.zeros(N, dtype=complex)
psi0[0] = 1.0

print(f"\nSystem: {n_qubits}-qubit Ising chain, J={J}, h={h}")
print(f"Initial state: |{'0' * n_qubits}⟩ (all spins up)")
print(f"\n{'Time':>6} {'<Mz> Trotter':>14} {'<Mz> Exact':>14} {'S(A|B)':>10}")
print("-" * 48)

n_steps = 50
for t in np.linspace(0, 4, 17):
    if t == 0:
        psi_t = psi0.copy()
        psi_exact = psi0.copy()
    else:
        psi_t, psi_exact = evolve_state(H, terms, psi0, t, n_steps, order=2)

    mz_trotter = measure_magnetization(psi_t, n_qubits)
    mz_exact = measure_magnetization(psi_exact, n_qubits)
    entropy = measure_entanglement_entropy(psi_exact, n_qubits, [0, 1])

    print(f"{t:6.2f} {mz_trotter:14.6f} {mz_exact:14.6f} {entropy:10.4f}")
```

### 9.3 qDRIFT Randomized Simulation

```python
import numpy as np
from scipy.linalg import expm

def qdrift_simulation(terms, t_total, n_samples, seed=42):
    """Simulate time evolution using the qDRIFT protocol.

    qDRIFT randomly samples individual Hamiltonian terms, with
    probability proportional to their coefficient magnitude.
    The key advantage: gate count is independent of the number
    of terms L, depending only on the total norm lambda.

    Args:
        terms: List of (coefficient, operator) pairs
        t_total: Total simulation time
        n_samples: Number of random samples (gates)
        seed: Random seed for reproducibility

    Returns:
        U_qdrift: Approximate unitary from qDRIFT
    """
    rng = np.random.default_rng(seed)

    # Compute probabilities
    coeffs = np.array([abs(c) for c, _ in terms])
    lam = np.sum(coeffs)
    probs = coeffs / lam

    tau = lam * t_total / n_samples
    N = terms[0][1].shape[0]
    U = np.eye(N, dtype=complex)

    for _ in range(n_samples):
        # Sample a term
        k = rng.choice(len(terms), p=probs)
        coeff, op = terms[k]
        sign = np.sign(coeff)

        # Apply exp(-i * sign * lambda * tau * H_k)
        U = expm(-1j * sign * op * tau) @ U

    return U


# Compare qDRIFT with deterministic Trotter
print("=" * 65)
print("qDRIFT vs Deterministic Trotter")
print("=" * 65)

n_qubits = 3
H, terms = build_ising_hamiltonian(n_qubits, J=1.0, h=0.5)
t_total = 1.0
U_exact = expm(-1j * H * t_total)

print(f"\nSystem: {n_qubits}-qubit Ising chain, t={t_total}")
print(f"Number of Hamiltonian terms: {len(terms)}")
print(f"\n{'Method':>20} {'Samples/Steps':>15} {'Error':>12}")
print("-" * 50)

# Trotter results
for n_steps in [10, 20, 50, 100]:
    _, _, error = trotter_simulation(H, terms, t_total, n_steps, order=2)
    print(f"{'Trotter-2':>20} {n_steps:>15d} {error:>12.2e}")

print()

# qDRIFT results (average over several random seeds)
for n_samples in [50, 100, 200, 500]:
    errors = []
    for seed in range(10):
        U_qd = qdrift_simulation(terms, t_total, n_samples, seed=seed)
        errors.append(np.linalg.norm(U_qd - U_exact, ord=2))
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    print(f"{'qDRIFT (mean)':>20} {n_samples:>15d} {mean_err:>12.2e} +/- {std_err:.2e}")
```

### 9.4 Quantum Phase Estimation for Energy

```python
import numpy as np
from scipy.linalg import expm

def qpe_energy_estimation(H, psi0, t, n_ancilla):
    """Simulate quantum phase estimation for energy eigenvalue extraction.

    QPE maps energy eigenvalues to phases of the time evolution operator.
    Given |psi> = sum_k c_k |E_k>, QPE produces a distribution over
    binary strings encoding the energies E_k with probabilities |c_k|^2.

    Args:
        H: Hamiltonian matrix
        psi0: Initial state (should have overlap with target eigenstate)
        t: Time parameter (larger t gives better resolution)
        n_ancilla: Number of ancilla qubits (precision bits)

    Returns:
        energies: Estimated energies
        probabilities: Corresponding probabilities
        exact_energies: Exact eigenvalues for comparison
    """
    N = H.shape[0]
    n_system = int(np.log2(N))

    # Exact eigendecomposition (for comparison)
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Overlap of psi0 with each eigenstate
    overlaps = np.abs(eigenvectors.T @ psi0) ** 2

    # QPE simulation: for each eigenvalue, the phase is phi = E*t/(2*pi)
    n_phases = 2 ** n_ancilla
    phase_distribution = np.zeros(n_phases)

    for k in range(len(eigenvalues)):
        if overlaps[k] < 1e-10:
            continue

        E = eigenvalues[k]
        phi = (E * t) / (2 * np.pi)

        # QPE maps phi to an integer j in [0, 2^m - 1]
        # The probability of measuring j given true phase phi is:
        # P(j) = |1/2^m * sum_l exp(2*pi*i*(phi - j/2^m)*l)|^2
        for j in range(n_phases):
            phase_diff = phi - j / n_phases
            # Geometric sum formula
            if abs(phase_diff * n_phases) < 1e-10:
                prob = 1.0
            else:
                prob = abs(np.sin(np.pi * phase_diff * n_phases) /
                          (n_phases * np.sin(np.pi * phase_diff))) ** 2
            phase_distribution[j] += overlaps[k] * prob

    # Convert phases back to energies
    estimated_phases = np.arange(n_phases) / n_phases
    estimated_energies = 2 * np.pi * estimated_phases / t

    # Shift to handle negative energies (phases wrap around)
    for i in range(n_phases):
        if estimated_energies[i] > np.pi / t:
            estimated_energies[i] -= 2 * np.pi / t

    return estimated_energies, phase_distribution, eigenvalues


# Demonstrate QPE for energy estimation
print("=" * 65)
print("Quantum Phase Estimation for Energy Eigenvalues")
print("=" * 65)

n_qubits = 3
H, _ = build_ising_hamiltonian(n_qubits, J=1.0, h=0.5)

# Exact eigenvalues
exact_E = np.sort(np.linalg.eigvalsh(H))
print(f"\nExact eigenvalues: {exact_E}")

# Initial state: superposition to overlap with multiple eigenstates
N = 2 ** n_qubits
psi0 = np.ones(N, dtype=complex) / np.sqrt(N)

# QPE with increasing precision
for n_ancilla in [4, 6, 8]:
    t = 2 * np.pi / (max(abs(exact_E)) + 1)  # Scale t appropriately
    energies, probs, _ = qpe_energy_estimation(H, psi0, t, n_ancilla)

    # Find peaks in the distribution
    peak_indices = np.where(probs > 0.01)[0]
    peak_energies = energies[peak_indices]
    peak_probs = probs[peak_indices]

    # Sort by probability
    sort_idx = np.argsort(-peak_probs)
    peak_energies = peak_energies[sort_idx]
    peak_probs = peak_probs[sort_idx]

    print(f"\nQPE with {n_ancilla} ancilla qubits:")
    print(f"  {'Estimated E':>14} {'Probability':>14} {'Nearest exact':>14} {'Error':>10}")
    for e, p in zip(peak_energies[:6], peak_probs[:6]):
        nearest = exact_E[np.argmin(np.abs(exact_E - e))]
        print(f"  {e:14.4f} {p:14.4f} {nearest:14.4f} {abs(e - nearest):10.4f}")
```

### 9.5 ADAPT-VQE Demonstration

```python
import numpy as np
from scipy.optimize import minimize

def adapt_vqe(H, n_qubits, operator_pool, max_iterations=10, threshold=1e-4):
    """Implement the ADAPT-VQE algorithm.

    ADAPT-VQE grows the ansatz adaptively by selecting operators
    from a pool based on their energy gradient. This avoids the
    fixed-structure ansatz problem and typically produces shorter circuits.

    Args:
        H: Hamiltonian matrix
        n_qubits: Number of qubits
        operator_pool: List of anti-Hermitian operators (generators)
        max_iterations: Maximum number of operators to add
        threshold: Gradient convergence threshold

    Returns:
        energy_history: List of energies at each iteration
        selected_operators: Indices of selected operators
        optimal_params: Final optimized parameters
    """
    N = 2 ** n_qubits

    # Start with the Hartree-Fock state (|0...0⟩)
    psi_ref = np.zeros(N, dtype=complex)
    psi_ref[0] = 1.0

    selected_ops = []
    all_params = []
    energy_history = []

    for iteration in range(max_iterations):
        # Current state with current parameters
        def build_state(params):
            psi = psi_ref.copy()
            for idx, (op_idx, _) in enumerate(selected_ops):
                A = operator_pool[op_idx]
                psi = expm(params[idx] * A) @ psi
            return psi

        # Compute gradient for each operator in the pool
        current_psi = build_state(all_params)
        gradients = []
        for pool_idx, A in enumerate(operator_pool):
            # Gradient = d<E>/d(theta) at theta=0 for new operator
            # = 2 * Re(<psi| H * A |psi>)  (since A is anti-Hermitian)
            grad = 2 * np.real(current_psi.conj() @ H @ A @ current_psi)
            gradients.append(abs(grad))

        max_grad_idx = np.argmax(gradients)
        max_grad = gradients[max_grad_idx]

        print(f"  Iteration {iteration + 1}: max gradient = {max_grad:.6f}", end="")

        if max_grad < threshold:
            print(" (converged)")
            break

        # Add the operator with largest gradient
        selected_ops.append((max_grad_idx, max_grad))
        all_params.append(0.0)

        # Re-optimize all parameters
        def cost(params):
            psi = build_state(params)
            return np.real(psi.conj() @ H @ psi)

        result = minimize(cost, all_params, method='COBYLA',
                         options={'maxiter': 200, 'rhobeg': 0.1})
        all_params = list(result.x)
        energy = result.fun
        energy_history.append(energy)

        print(f", energy = {energy:.6f}")

    return energy_history, selected_ops, all_params


# Build operator pool for ADAPT-VQE
def build_operator_pool(n_qubits):
    """Build a pool of single-excitation operators.

    These operators generate rotations between computational basis states.
    In chemistry, they correspond to single and double excitations.
    """
    N = 2 ** n_qubits
    pool = []

    # Single-qubit rotations (iX, iY generators)
    for q in range(n_qubits):
        for pauli in [X, Y]:
            ops_list = [I] * n_qubits
            ops_list[q] = pauli
            A = 1j * kron_list(ops_list)
            # Make anti-Hermitian: A -> (A - A^dagger) / 2
            A = (A - A.conj().T) / 2
            if np.linalg.norm(A) > 1e-10:
                pool.append(A)

    # Two-qubit excitation operators
    for q1 in range(n_qubits):
        for q2 in range(q1 + 1, n_qubits):
            for p1, p2 in [(X, Y), (Y, X)]:
                ops_list = [I] * n_qubits
                ops_list[q1] = p1
                ops_list[q2] = p2
                A = 1j * kron_list(ops_list)
                A = (A - A.conj().T) / 2
                if np.linalg.norm(A) > 1e-10:
                    pool.append(A)

    return pool


# Demonstrate ADAPT-VQE
print("=" * 65)
print("ADAPT-VQE for Ising Model Ground State")
print("=" * 65)

n_qubits = 3
H, _ = build_ising_hamiltonian(n_qubits, J=1.0, h=1.0)
exact_ground = np.min(np.linalg.eigvalsh(H))
print(f"\nExact ground state energy: {exact_ground:.6f}")
print(f"Operator pool construction...")

pool = build_operator_pool(n_qubits)
print(f"Pool size: {len(pool)} operators\n")

energy_history, selected_ops, params = adapt_vqe(
    H, n_qubits, pool, max_iterations=8, threshold=1e-3
)

print(f"\nFinal energy: {energy_history[-1]:.6f}")
print(f"Exact energy: {exact_ground:.6f}")
print(f"Error: {abs(energy_history[-1] - exact_ground):.2e}")
print(f"Operators used: {len(selected_ops)} (out of {len(pool)} available)")
```

---

## 10. Exercises

### Exercise 1: Trotter Error Analysis

For the 4-qubit transverse-field Ising model with $J = 1$, $h = 0.5$:
(a) Compute the Trotter error $\|U_{\text{Trotter}} - U_{\text{exact}}\|$ for $t = 1$ with $r = 1, 2, 5, 10, 20, 50$ steps.
(b) Plot the error vs. $r$ on a log-log scale. Verify the $O(1/r)$ scaling for first-order and $O(1/r^2)$ for second-order Trotter.
(c) Compute the total number of gates (counting each $e^{-iH_k dt}$ as one gate) for each case. At what step count does second-order Trotter become more efficient than first-order for a target error of $10^{-3}$?

### Exercise 2: Observable Dynamics

Simulate the 5-qubit Ising model starting from $|11111\rangle$ (all spins down):
(a) Track the magnetization $\langle M_z \rangle$ over time $t \in [0, 10]$.
(b) Track the entanglement entropy of the left 2 qubits with the right 3 qubits.
(c) At what time does the entanglement entropy first reach its maximum? How does this relate to the Lieb-Robinson velocity?

### Exercise 3: qDRIFT vs. Trotter

Compare qDRIFT and second-order Trotter for the Heisenberg model $H = \sum_{i} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})$ with $n = 4$ qubits:
(a) For a fixed total gate budget of 100, which method achieves lower error?
(b) Run qDRIFT 50 times with different random seeds. What is the variance of the error?
(c) How does the relative performance change as $n$ increases (try $n = 3, 4, 5, 6$)?

### Exercise 4: QPE Energy Resolution

For the 3-qubit Ising model:
(a) Choose $t$ so that all eigenvalues map to distinct phases (no aliasing).
(b) Determine the minimum number of ancilla qubits needed to resolve the two lowest energy levels.
(c) How does the QPE energy estimate improve as you increase ancilla qubits from 4 to 10?
(d) If the initial state is the Hartree-Fock state $|000\rangle$, what is its overlap with the ground state? How does this overlap affect QPE success probability?

### Exercise 5: ADAPT-VQE Convergence

Using the 4-qubit Ising model at the critical point ($J = h = 1$):
(a) Run ADAPT-VQE and record the energy after each operator addition.
(b) Compare the convergence rate with a fixed hardware-efficient ansatz of the same depth.
(c) Which operators from the pool are selected first? Do they have a physical interpretation?
(d) How does the convergence change when you start from $|0000\rangle$ vs. a random product state?

---

[← Previous: Quantum Computing Landscape and Future](16_Landscape_and_Future.md) | [Next: Quantum Walks →](19_Quantum_Walks.md)
