# Lesson 14: QAOA and Combinatorial Optimization

[← Previous: Variational Quantum Eigensolver](13_VQE.md) | [Next: Quantum Machine Learning →](15_Quantum_Machine_Learning.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Formulate combinatorial optimization problems as Ising Hamiltonians
2. Explain the MaxCut problem and its graph-theoretic formulation
3. Construct the QAOA circuit with alternating cost and mixer unitaries
4. Describe the role of the cost Hamiltonian and mixing Hamiltonian
5. Analyze the depth-performance tradeoff in $p$-layer QAOA
6. Connect QAOA to adiabatic quantum computing
7. Implement QAOA for MaxCut and compare with brute-force solutions

---

Combinatorial optimization is one of the most important classes of problems in computer science and operations research. From scheduling airline routes to designing circuits, from portfolio optimization to drug discovery, countless real-world problems reduce to finding the best configuration among exponentially many possibilities. Most of these problems are NP-hard, meaning no known classical algorithm solves them efficiently in the worst case.

The Quantum Approximate Optimization Algorithm (QAOA), introduced by Farhi, Goldstone, and Gutmann in 2014, provides a systematic framework for attacking combinatorial optimization on quantum computers. Like VQE (Lesson 13), QAOA is a hybrid quantum-classical algorithm designed for NISQ devices. It encodes the optimization problem into a quantum Hamiltonian and uses a structured parameterized circuit to search for approximate solutions. As the circuit depth increases, QAOA interpolates between a random guess and the optimal solution, with theoretical guarantees of convergence.

> **Analogy:** QAOA is like sculpting. Each layer alternately chisels (cost unitary) and smooths (mixer unitary) the quantum state, gradually revealing the optimal solution hidden in the stone. The cost unitary carves toward solutions that score well, while the mixer unitary prevents the state from getting trapped in poor configurations. With enough layers, the sculpture converges to the optimal form.

## Table of Contents

1. [Combinatorial Optimization Problems](#1-combinatorial-optimization-problems)
2. [MaxCut Problem](#2-maxcut-problem)
3. [The QAOA Framework](#3-the-qaoa-framework)
4. [QAOA Circuit Construction](#4-qaoa-circuit-construction)
5. [Cost and Mixer Hamiltonians](#5-cost-and-mixer-hamiltonians)
6. [Parameter Optimization](#6-parameter-optimization)
7. [Connection to Adiabatic Quantum Computing](#7-connection-to-adiabatic-quantum-computing)
8. [Performance Analysis](#8-performance-analysis)
9. [Python Implementation](#9-python-implementation)
10. [Exercises](#10-exercises)

---

## 1. Combinatorial Optimization Problems

### 1.1 General Framework

A combinatorial optimization problem asks: given a finite set of candidate solutions $\{z\}$ and an objective function $C(z)$, find the solution $z^*$ that maximizes (or minimizes) $C$:

$$z^* = \arg\max_{z \in \{0,1\}^n} C(z)$$

where $z = (z_1, z_2, \ldots, z_n)$ is a binary string of length $n$.

### 1.2 NP-Hard Problems

Many combinatorial optimization problems are NP-hard:

| Problem | Description | Applications |
|---------|-------------|-------------|
| MaxCut | Partition graph vertices to maximize cut edges | Circuit layout, social network analysis |
| Traveling Salesman | Shortest route visiting all cities | Logistics, routing |
| Graph Coloring | Color vertices with minimum colors, no adjacent same | Scheduling, register allocation |
| Max-SAT | Maximize satisfied clauses in Boolean formula | Verification, AI planning |
| Portfolio Optimization | Maximize return subject to risk constraints | Finance |

### 1.3 Ising Formulation

Many combinatorial problems can be encoded as finding the ground state of an **Ising Hamiltonian**:

$$H_C = \sum_{(i,j) \in E} J_{ij} Z_i Z_j + \sum_i h_i Z_i$$

where $Z_i \in \{+1, -1\}$ are Ising spin variables. The mapping to binary variables is $z_i = (1 - Z_i)/2$, so $Z_i = +1 \leftrightarrow z_i = 0$ and $Z_i = -1 \leftrightarrow z_i = 1$.

This Ising formulation is the bridge between classical optimization and quantum algorithms: the Ising Hamiltonian directly becomes the cost Hamiltonian in QAOA.

---

## 2. MaxCut Problem

### 2.1 Definition

Given an undirected graph $G = (V, E)$ with $n$ vertices and $m$ edges:

- **Partition** the vertices into two disjoint sets $S$ and $\bar{S}$
- **Cut**: an edge $(i, j)$ is "cut" if $i \in S$ and $j \in \bar{S}$ (or vice versa)
- **Objective**: maximize the number of cut edges

Formally, with $z_i \in \{0, 1\}$ indicating which set vertex $i$ belongs to:

$$C(z) = \sum_{(i,j) \in E} z_i (1 - z_j) + (1 - z_i) z_j = \sum_{(i,j) \in E} z_i \oplus z_j$$

### 2.2 Ising Formulation

Using $Z_i = 1 - 2z_i$:

$$C = \sum_{(i,j) \in E} \frac{1 - Z_i Z_j}{2} = \frac{|E|}{2} - \frac{1}{2}\sum_{(i,j) \in E} Z_i Z_j$$

Maximizing $C$ is equivalent to minimizing $H_C = \sum_{(i,j) \in E} Z_i Z_j$.

As a quantum operator (replacing classical spins with Pauli-Z):

$$\hat{H}_C = \sum_{(i,j) \in E} \hat{Z}_i \hat{Z}_j$$

The ground state of $\hat{H}_C$ encodes the MaxCut solution.

### 2.3 Example: Triangle Graph

For a triangle (3 vertices, 3 edges: (0,1), (1,2), (0,2)):

| Assignment $z$ | Cut value $C(z)$ |
|-----------------|-------------------|
| 000 | 0 |
| 001 | 2 |
| 010 | 2 |
| 011 | 2 |
| 100 | 2 |
| 101 | 2 |
| 110 | 2 |
| 111 | 0 |

Maximum cut = 2, achieved by any assignment with exactly 1 or 2 vertices in $S$. For a triangle, we cannot cut all 3 edges (this would require a bipartite graph).

### 2.4 Complexity

MaxCut is NP-hard for general graphs. The best classical approximation algorithm (Goemans-Williamson, 1995) achieves a ratio of $\alpha_{GW} \approx 0.878$ (i.e., the solution is at least 87.8% of optimal). Whether QAOA can beat this ratio is an open question.

---

## 3. The QAOA Framework

### 3.1 Overview

QAOA is parameterized by an integer $p \geq 1$ (the number of layers, or "depth"). A $p$-layer QAOA circuit applies alternating cost and mixer unitaries:

$$|\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle = U_M(\beta_p) U_C(\gamma_p) \cdots U_M(\beta_1) U_C(\gamma_1) |+\rangle^{\otimes n}$$

where:
- $\boldsymbol{\gamma} = (\gamma_1, \ldots, \gamma_p)$ and $\boldsymbol{\beta} = (\beta_1, \ldots, \beta_p)$ are $2p$ real parameters
- $U_C(\gamma) = e^{-i\gamma \hat{H}_C}$ is the **cost unitary**
- $U_M(\beta) = e^{-i\beta \hat{H}_M}$ is the **mixer unitary**
- $|+\rangle^{\otimes n}$ is the uniform superposition (initial state)

### 3.2 The Optimization Loop

1. Choose $p$ (circuit depth)
2. Initialize parameters $\boldsymbol{\gamma}_0, \boldsymbol{\beta}_0$
3. **Quantum**: prepare $|\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle$, measure $\langle \hat{H}_C \rangle$
4. **Classical**: update parameters to minimize $\langle \hat{H}_C \rangle$
5. Iterate until convergence
6. Final measurement: sample from $|\boldsymbol{\gamma}^*, \boldsymbol{\beta}^*\rangle$ to obtain candidate solutions

### 3.3 What Makes QAOA Different from VQE?

| Feature | VQE | QAOA |
|---------|-----|------|
| Ansatz | Problem-agnostic or chemistry-inspired | Problem-structured (cost + mixer) |
| Parameters | Arbitrary rotation angles | Alternating $\gamma_i, \beta_i$ |
| Structure | Generic layers | Alternating cost/mixer unitaries |
| Convergence | Variational (upper bound) | $p \to \infty$ gives exact solution |
| Target | Ground state energy | Optimization solution (bitstring) |

---

## 4. QAOA Circuit Construction

### 4.1 Initial State

Start in the uniform superposition, which is the ground state of the mixer Hamiltonian $\hat{H}_M = \sum_i X_i$:

$$|+\rangle^{\otimes n} = H^{\otimes n}|0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{z \in \{0,1\}^n} |z\rangle$$

### 4.2 Cost Unitary

For MaxCut, $\hat{H}_C = \sum_{(i,j) \in E} Z_i Z_j$, so:

$$U_C(\gamma) = e^{-i\gamma \hat{H}_C} = \prod_{(i,j) \in E} e^{-i\gamma Z_i Z_j}$$

Each factor $e^{-i\gamma Z_i Z_j}$ can be implemented as:

```
q_i ─── ●──── Rz(2γ) ──── ●────
        │                   │
q_j ─── ⊕─────────────── ⊕────
```

That is: CNOT(i→j), then $R_z(2\gamma)$ on qubit $j$, then CNOT(i→j) again. This applies a phase $e^{-i\gamma}$ when $z_i = z_j$ and $e^{+i\gamma}$ when $z_i \neq z_j$.

Alternatively, the $ZZ$ interaction can be decomposed as:

$$e^{-i\gamma Z_i Z_j} = \text{CNOT}_{ij} \cdot (I \otimes R_z(2\gamma)) \cdot \text{CNOT}_{ij}$$

### 4.3 Mixer Unitary

The standard mixer is the transverse field:

$$\hat{H}_M = \sum_{i=1}^{n} X_i$$

$$U_M(\beta) = e^{-i\beta \hat{H}_M} = \prod_{i=1}^{n} e^{-i\beta X_i} = \prod_{i=1}^{n} R_x(2\beta)$$

Each factor is simply an $R_x(2\beta)$ rotation on a single qubit, which requires no entangling gates.

### 4.4 Complete Circuit (p=1)

```
|0⟩ ── H ── ●── Rz ──●── ●── Rz ──●── Rx(2β) ── M
             │        │   │        │
|0⟩ ── H ── ⊕────────⊕── │── Rz ──│── Rx(2β) ── M
                           │        │
|0⟩ ── H ──────────────── ⊕────────⊕── Rx(2β) ── M

     H⊗n        U_C(γ)                U_M(β)
```

### 4.5 Gate Count

For a graph with $n$ vertices and $m$ edges:
- **Initial layer**: $n$ Hadamard gates
- **Per QAOA layer**: $2m$ CNOT + $m$ $R_z$ + $n$ $R_x$ gates
- **Total for $p$ layers**: $n + p(2m + m + n) = n + p(3m + n)$ gates

---

## 5. Cost and Mixer Hamiltonians

### 5.1 Cost Hamiltonian Properties

The cost Hamiltonian $\hat{H}_C$ is diagonal in the computational basis:

$$\hat{H}_C|z\rangle = C(z)|z\rangle$$

where $C(z)$ is the classical objective function value. This means $U_C(\gamma)$ applies a phase to each computational basis state proportional to its objective value:

$$U_C(\gamma)|z\rangle = e^{-i\gamma C(z)}|z\rangle$$

States with lower $C(z)$ (better solutions for minimization) get different phases than states with higher $C(z)$. The QFT-like interference in subsequent operations amplifies good solutions.

### 5.2 Mixer Hamiltonian Properties

The mixer Hamiltonian $\hat{H}_M = \sum_i X_i$ generates transitions between computational basis states. Each $X_i$ flips qubit $i$:

$$e^{-i\beta X_i}|z_1 \cdots z_i \cdots z_n\rangle = \cos\beta |z_1 \cdots z_i \cdots z_n\rangle - i\sin\beta |z_1 \cdots \bar{z}_i \cdots z_n\rangle$$

The mixer "explores" neighboring solutions by flipping individual bits with amplitude $\sin\beta$.

### 5.3 Interplay: Exploration vs Exploitation

- **Cost unitary** ($U_C$): Exploitation — amplifies states with good objective values
- **Mixer unitary** ($U_M$): Exploration — allows transitions to neighboring states

By alternating the two, QAOA balances exploration and exploitation, similar to simulated annealing or evolutionary algorithms.

### 5.4 Custom Mixers

For constrained optimization problems, the standard $X$ mixer may produce infeasible solutions. Custom mixers can enforce constraints:

- **Grover mixer**: preserves the uniform superposition over feasible solutions
- **XY mixer**: preserves the number of 1-bits (Hamming weight constraint)
- **Parity mixer**: preserves parity

---

## 6. Parameter Optimization

### 6.1 The Optimization Landscape

The QAOA energy $E(\boldsymbol{\gamma}, \boldsymbol{\beta}) = \langle\boldsymbol{\gamma}, \boldsymbol{\beta}|\hat{H}_C|\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle$ is a smooth function of $2p$ real parameters. The optimization landscape can have:

- Multiple local minima
- Saddle points
- Symmetries (periodicity in parameters)

### 6.2 Parameter Periodicity

Due to the structure of QAOA, the parameters have periodicities:

$$\gamma_i \in [0, 2\pi), \quad \beta_i \in [0, \pi)$$

For MaxCut on unweighted graphs, there are additional symmetries that further reduce the search space.

### 6.3 Optimization Strategies

**Grid search** (for small $p$): Evaluate $E$ on a grid of $(\gamma, \beta)$ values. Feasible only for $p = 1$ or $p = 2$.

**Gradient-based**: Use the parameter shift rule (as in VQE) to compute gradients and apply methods like Adam or L-BFGS-B.

**Interpolation** (INTERP): Optimize $p$-layer QAOA, then use the optimal parameters as a starting point for $(p+1)$-layer QAOA. Specifically, interpolate the $p$ parameters to $p+1$ parameters.

**Fourier heuristic**: Parameterize $\gamma_i$ and $\beta_i$ as Fourier series in $i$, reducing the number of free parameters.

### 6.4 Parameter Transfer

A practical strategy: optimize QAOA for a small instance of the same graph family, then use those parameters (appropriately scaled) as the starting point for larger instances. This "parameter transfer" often works surprisingly well.

---

## 7. Connection to Adiabatic Quantum Computing

### 7.1 Adiabatic Quantum Computing (AQC)

AQC encodes the optimization problem in a Hamiltonian $\hat{H}_C$ and slowly evolves from the ground state of a simple initial Hamiltonian $\hat{H}_M$ to the ground state of $\hat{H}_C$:

$$\hat{H}(t) = \left(1 - \frac{t}{T}\right)\hat{H}_M + \frac{t}{T}\hat{H}_C, \quad t \in [0, T]$$

The **adiabatic theorem** guarantees that if $T$ is large enough (inversely proportional to the square of the minimum energy gap), the system remains in the ground state throughout the evolution.

### 7.2 QAOA as Trotterized Adiabatic Evolution

QAOA can be viewed as a Trotterized approximation to adiabatic evolution. With $p$ layers, the evolution from $\hat{H}_M$ to $\hat{H}_C$ is discretized into $p$ steps:

$$U_{\text{QAOA}} = \prod_{k=1}^{p} e^{-i\beta_k \hat{H}_M} e^{-i\gamma_k \hat{H}_C}$$

In the adiabatic limit with $p \to \infty$, $\beta_k \to 0$ and $\gamma_k \to 0$ (with appropriate scaling), this converges to exact adiabatic evolution.

### 7.3 QAOA vs AQC: Key Differences

| Aspect | QAOA | AQC |
|--------|------|-----|
| Parameters | Free ($2p$ optimized) | Fixed by schedule $s(t)$ |
| Hardware | Gate-based | Annealing-based |
| Depth | $p$ layers (finite) | Continuous (long time $T$) |
| Optimization | Classical outer loop | None (physics does the work) |
| Flexibility | Can outperform adiabatic for small $p$ | Guaranteed for large $T$ |

### 7.4 Convergence Guarantee

**Theorem** (Farhi et al.): For any combinatorial optimization problem, QAOA with $p \to \infty$ and optimized parameters converges to the exact ground state of $\hat{H}_C$.

This means QAOA is, in principle, a universal optimization algorithm. The practical question is: how large must $p$ be to find a good solution?

---

## 8. Performance Analysis

### 8.1 QAOA $p=1$ for MaxCut

For $p = 1$ on 3-regular graphs (every vertex has degree 3), Farhi et al. proved:

$$\frac{\langle C \rangle_{\text{QAOA}}}{C_{\max}} \geq 0.6924$$

This means QAOA at $p = 1$ guarantees at least 69.24% of the optimal cut value. For comparison, a random assignment gives 50%.

### 8.2 Scaling with $p$

As $p$ increases:
- The approximation ratio improves
- The number of parameters ($2p$) grows, making optimization harder
- The circuit depth increases, making it more susceptible to noise

Empirically, for many graph instances, $p = 3\text{-}5$ gives solutions close to optimal.

### 8.3 Comparison with Classical Algorithms

| Algorithm | Approximation ratio | Type |
|-----------|-------------------|------|
| Random assignment | 0.500 | Trivial |
| QAOA $p=1$ (3-regular) | 0.6924 | Quantum |
| QAOA $p=2$ (3-regular) | ~0.756 | Quantum |
| Greedy algorithm | ~0.5 (worst case) | Classical |
| Goemans-Williamson SDP | 0.878 | Classical |
| QAOA $p \to \infty$ | 1.000 | Quantum |

Whether finite-$p$ QAOA can beat Goemans-Williamson remains an open problem.

---

## 9. Python Implementation

### 9.1 MaxCut QAOA Core

```python
import numpy as np
from scipy.optimize import minimize
from itertools import product

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def build_cost_hamiltonian(n_qubits, edges):
    """Build the MaxCut cost Hamiltonian as a matrix.

    Why matrix form? For small instances (n ≤ 12), the full matrix allows
    exact simulation and verification. For larger instances, one would use
    state-vector simulation or actual quantum hardware.
    """
    N = 2**n_qubits
    H_C = np.zeros((N, N), dtype=complex)

    for (i, j) in edges:
        # Z_i Z_j term
        op = [I] * n_qubits
        op[i] = Z
        op[j] = Z
        term = op[0]
        for k in range(1, n_qubits):
            term = np.kron(term, op[k])
        H_C += term

    return H_C

def build_mixer_hamiltonian(n_qubits):
    """Build the mixer Hamiltonian H_M = Σ X_i."""
    N = 2**n_qubits
    H_M = np.zeros((N, N), dtype=complex)

    for i in range(n_qubits):
        op = [I] * n_qubits
        op[i] = X
        term = op[0]
        for k in range(1, n_qubits):
            term = np.kron(term, op[k])
        H_M += term

    return H_M

def maxcut_value(bitstring, edges):
    """Compute the MaxCut value for a given bitstring.

    A cut edge is one where the two endpoints have different bit values.
    """
    return sum(1 for (i, j) in edges if bitstring[i] != bitstring[j])

def brute_force_maxcut(n_qubits, edges):
    """Find the optimal MaxCut by exhaustive enumeration.

    Why brute force? For small instances, this gives the exact answer to
    compare against QAOA. For n > 20, brute force becomes infeasible,
    which is precisely where quantum algorithms might help.
    """
    best_cut = 0
    best_assignment = None

    for bits in product([0, 1], repeat=n_qubits):
        cut = maxcut_value(bits, edges)
        if cut > best_cut:
            best_cut = cut
            best_assignment = bits

    return best_cut, best_assignment

def qaoa_circuit(params, n_qubits, H_C, H_M, p):
    """Simulate the QAOA circuit and return the final state.

    The circuit alternates p layers of:
    1. Cost unitary: e^{-iγ H_C} (phases based on objective value)
    2. Mixer unitary: e^{-iβ H_M} (transitions between solutions)
    """
    gammas = params[:p]
    betas = params[p:]
    N = 2**n_qubits

    # Initial state: uniform superposition |+⟩^n
    state = np.ones(N, dtype=complex) / np.sqrt(N)

    for layer in range(p):
        # Cost unitary: e^{-iγ H_C}
        # Since H_C is diagonal in computational basis, this is efficient
        # But for full matrix simulation, we use matrix exponential
        U_C = np.diag(np.exp(-1j * gammas[layer] * np.diag(H_C).real))
        state = U_C @ state

        # Mixer unitary: e^{-iβ H_M}
        # H_M = Σ X_i, and X_i commute when acting on different qubits
        # So e^{-iβ Σ X_i} = ⊗ e^{-iβ X_i} = ⊗ Rx(2β)
        Rx = np.array([[np.cos(betas[layer]), -1j*np.sin(betas[layer])],
                       [-1j*np.sin(betas[layer]), np.cos(betas[layer])]])
        U_M = Rx
        for _ in range(n_qubits - 1):
            U_M = np.kron(U_M, Rx)
        state = U_M @ state

    return state

def qaoa_expectation(params, n_qubits, edges, H_C, H_M, p):
    """Compute the QAOA expected cost value.

    This is the objective function for the classical optimizer:
    minimize ⟨γ,β|H_C|γ,β⟩ (for MaxCut, we want the minimum of ZZ terms,
    which corresponds to the maximum cut).
    """
    state = qaoa_circuit(params, n_qubits, H_C, H_M, p)
    energy = np.real(state.conj() @ H_C @ state)
    return energy

def run_qaoa_maxcut(n_qubits, edges, p=1, n_restarts=10):
    """Run QAOA for MaxCut with multiple random restarts.

    Why multiple restarts? The QAOA landscape can have multiple local minima.
    Running from different starting points increases the chance of finding
    the global optimum.
    """
    H_C = build_cost_hamiltonian(n_qubits, edges)
    H_M = build_mixer_hamiltonian(n_qubits)

    # Brute force for comparison
    optimal_cut, optimal_assignment = brute_force_maxcut(n_qubits, edges)

    print(f"Graph: {n_qubits} vertices, {len(edges)} edges")
    print(f"Optimal MaxCut: {optimal_cut} (assignment: {optimal_assignment})")
    print(f"QAOA depth: p = {p}")

    best_energy = float('inf')
    best_params = None

    for restart in range(n_restarts):
        # Random initial parameters
        gamma0 = np.random.uniform(0, 2*np.pi, p)
        beta0 = np.random.uniform(0, np.pi, p)
        params0 = np.concatenate([gamma0, beta0])

        result = minimize(qaoa_expectation, params0,
                         args=(n_qubits, edges, H_C, H_M, p),
                         method='COBYLA',
                         options={'maxiter': 500})

        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x

    # Analyze the best solution
    state = qaoa_circuit(best_params, n_qubits, H_C, H_M, p)
    probs = np.abs(state)**2

    # Convert energy to cut value
    # H_C = Σ Z_i Z_j, and cut value C = (|E| - ⟨H_C⟩) / 2
    expected_cut = (len(edges) - best_energy) / 2

    print(f"\nQAOA results:")
    print(f"  Expected cut value: {expected_cut:.4f}")
    print(f"  Approximation ratio: {expected_cut/optimal_cut:.4f}")
    print(f"  Optimal γ: {best_params[:p].round(4)}")
    print(f"  Optimal β: {best_params[p:].round(4)}")

    # Top solutions by probability
    print(f"\n  Top 5 measurement outcomes:")
    top_indices = np.argsort(probs)[::-1][:5]
    for idx in top_indices:
        bits = tuple(int(b) for b in f"{idx:0{n_qubits}b}")
        cut = maxcut_value(bits, edges)
        print(f"    |{''.join(map(str, bits))}⟩: prob={probs[idx]:.4f}, cut={cut}")

    return expected_cut, optimal_cut, best_params

# === Example 1: Triangle graph ===
print("=" * 55)
print("QAOA for MaxCut: Triangle Graph")
print("=" * 55)
run_qaoa_maxcut(3, [(0,1), (1,2), (0,2)], p=1)

# === Example 2: 4-vertex graph ===
print("\n" + "=" * 55)
print("QAOA for MaxCut: 4-Vertex Graph")
print("=" * 55)
edges_4 = [(0,1), (1,2), (2,3), (0,3), (0,2)]
run_qaoa_maxcut(4, edges_4, p=2)
```

### 9.2 Parameter Landscape Visualization

```python
import numpy as np

def qaoa_landscape(n_qubits, edges, gamma_range, beta_range, resolution=50):
    """Compute the QAOA energy landscape for p=1.

    Why visualize? The landscape reveals the optimization difficulty and
    the structure of the parameter space. Smooth landscapes with few
    local minima are easy to optimize; rugged landscapes with many
    local minima require more sophisticated strategies.
    """
    H_C = build_cost_hamiltonian(n_qubits, edges)
    H_M = build_mixer_hamiltonian(n_qubits)

    gammas = np.linspace(gamma_range[0], gamma_range[1], resolution)
    betas = np.linspace(beta_range[0], beta_range[1], resolution)

    landscape = np.zeros((resolution, resolution))

    for i, gamma in enumerate(gammas):
        for j, beta in enumerate(betas):
            landscape[i, j] = qaoa_expectation(
                [gamma, beta], n_qubits, edges, H_C, H_M, p=1)

    # Convert to cut values
    cut_landscape = (len(edges) - landscape) / 2

    # Find the optimal point
    max_idx = np.unravel_index(np.argmax(cut_landscape), cut_landscape.shape)
    opt_gamma = gammas[max_idx[0]]
    opt_beta = betas[max_idx[1]]
    opt_cut = cut_landscape[max_idx]

    print(f"Landscape for {n_qubits}-vertex graph, {len(edges)} edges")
    print(f"Optimal parameters: γ={opt_gamma:.4f}, β={opt_beta:.4f}")
    print(f"Maximum expected cut: {opt_cut:.4f}")
    print(f"Brute force optimal: {brute_force_maxcut(n_qubits, edges)[0]}")

    # Print a text-based visualization
    print(f"\nLandscape (cut value, rows=γ, cols=β):")
    print(f"  β: {beta_range[0]:.2f} {'→':>20} {beta_range[1]:.2f}")
    step = max(1, resolution // 10)
    for i in range(0, resolution, step):
        row = ""
        for j in range(0, resolution, step):
            val = cut_landscape[i, j]
            if val > 0.9 * opt_cut:
                row += "█"
            elif val > 0.7 * opt_cut:
                row += "▓"
            elif val > 0.5 * opt_cut:
                row += "▒"
            elif val > 0.3 * opt_cut:
                row += "░"
            else:
                row += " "
        print(f"  γ={gammas[i]:5.2f} |{row}|")

    return cut_landscape, gammas, betas

# Visualize for a 4-vertex graph
print("=" * 55)
print("QAOA Parameter Landscape (p=1)")
print("=" * 55)
edges_4 = [(0,1), (1,2), (2,3), (0,3)]
qaoa_landscape(4, edges_4, (0, 2*np.pi), (0, np.pi))
```

### 9.3 Depth Scaling Experiment

```python
import numpy as np
from scipy.optimize import minimize

def qaoa_depth_experiment(n_qubits, edges, max_p=5, n_restarts=5):
    """Study how QAOA performance scales with circuit depth p.

    Key insight: increasing p should improve the approximation ratio,
    but at the cost of more parameters to optimize and deeper circuits
    (more susceptible to noise on real hardware).
    """
    H_C = build_cost_hamiltonian(n_qubits, edges)
    H_M = build_mixer_hamiltonian(n_qubits)
    optimal_cut, _ = brute_force_maxcut(n_qubits, edges)

    print(f"Graph: {n_qubits} vertices, {len(edges)} edges, MaxCut = {optimal_cut}")
    print(f"\n{'p':>4} {'Expected Cut':>14} {'Approx Ratio':>14} {'Best Sampled':>14}")
    print("-" * 50)

    for p in range(1, max_p + 1):
        best_energy = float('inf')
        best_params = None

        for _ in range(n_restarts):
            gamma0 = np.random.uniform(0, np.pi, p)
            beta0 = np.random.uniform(0, np.pi/2, p)
            params0 = np.concatenate([gamma0, beta0])

            result = minimize(qaoa_expectation, params0,
                            args=(n_qubits, edges, H_C, H_M, p),
                            method='COBYLA',
                            options={'maxiter': 1000})

            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x

        expected_cut = (len(edges) - best_energy) / 2
        ratio = expected_cut / optimal_cut

        # Sample from the optimal state
        state = qaoa_circuit(best_params, n_qubits, H_C, H_M, p)
        probs = np.abs(state)**2
        best_sampled = max(maxcut_value(
            tuple(int(b) for b in f"{idx:0{n_qubits}b}"), edges)
            for idx in np.argsort(probs)[::-1][:3])

        print(f"{p:4d} {expected_cut:14.4f} {ratio:14.4f} {best_sampled:14d}")

    return

# Experiment on a 5-vertex graph
print("=" * 55)
print("QAOA Depth Scaling Experiment")
print("=" * 55)
edges_5 = [(0,1), (1,2), (2,3), (3,4), (4,0), (0,2), (1,3)]
qaoa_depth_experiment(5, edges_5, max_p=5)
```

### 9.4 Comparison: QAOA vs Random vs Greedy

```python
import numpy as np

def compare_algorithms(n_qubits, edges, p_qaoa=2, n_trials=1000):
    """Compare QAOA against classical heuristics for MaxCut.

    This puts QAOA in context: how does it compare to simple classical
    baselines? Understanding this is crucial for assessing whether quantum
    algorithms provide a real advantage.
    """
    optimal_cut, _ = brute_force_maxcut(n_qubits, edges)

    # 1. Random assignment (baseline)
    random_cuts = []
    for _ in range(n_trials):
        bits = tuple(np.random.randint(0, 2, n_qubits))
        random_cuts.append(maxcut_value(bits, edges))
    avg_random = np.mean(random_cuts)

    # 2. Greedy algorithm
    def greedy_maxcut():
        assignment = [0] * n_qubits
        for v in range(n_qubits):
            # Try v=0 and v=1, pick the one that maximizes cut so far
            cut_0, cut_1 = 0, 0
            for (i, j) in edges:
                if i == v or j == v:
                    other = j if i == v else i
                    if other < v:  # Only count edges to already-assigned vertices
                        cut_0 += (0 != assignment[other])
                        cut_1 += (1 != assignment[other])
            assignment[v] = 1 if cut_1 > cut_0 else 0
        return maxcut_value(tuple(assignment), edges)

    greedy_cut = greedy_maxcut()

    # 3. QAOA
    H_C = build_cost_hamiltonian(n_qubits, edges)
    H_M = build_mixer_hamiltonian(n_qubits)

    best_energy = float('inf')
    for _ in range(10):
        params0 = np.random.uniform(0, np.pi, 2 * p_qaoa)
        result = minimize(qaoa_expectation, params0,
                        args=(n_qubits, edges, H_C, H_M, p_qaoa),
                        method='COBYLA', options={'maxiter': 500})
        if result.fun < best_energy:
            best_energy = result.fun
    qaoa_cut = (len(edges) - best_energy) / 2

    print(f"\nMaxCut Comparison ({n_qubits} vertices, {len(edges)} edges)")
    print(f"{'Algorithm':>20} {'Cut Value':>12} {'Ratio':>8}")
    print("-" * 44)
    print(f"{'Random (avg)':>20} {avg_random:12.2f} {avg_random/optimal_cut:8.4f}")
    print(f"{'Greedy':>20} {greedy_cut:12d} {greedy_cut/optimal_cut:8.4f}")
    print(f"{'QAOA (p={p_qaoa})':>20} {qaoa_cut:12.4f} {qaoa_cut/optimal_cut:8.4f}")
    print(f"{'Optimal':>20} {optimal_cut:12d} {1.0:8.4f}")

# Compare on different graphs
print("=" * 55)
print("Algorithm Comparison for MaxCut")
print("=" * 55)

# Pentagon with diagonals
edges = [(0,1), (1,2), (2,3), (3,4), (4,0), (0,2), (1,3)]
compare_algorithms(5, edges, p_qaoa=2)

# 6-vertex random graph
edges_6 = [(0,1), (0,2), (0,5), (1,2), (1,3), (2,4), (3,4), (3,5), (4,5)]
compare_algorithms(6, edges_6, p_qaoa=2)
```

---

## 10. Exercises

### Exercise 1: QAOA for a Square Graph

Apply QAOA to the 4-vertex square graph with edges $\{(0,1), (1,2), (2,3), (3,0)\}$:
(a) What is the optimal MaxCut? List all optimal assignments.
(b) Run QAOA with $p = 1$. What approximation ratio do you achieve?
(c) Plot the energy landscape $E(\gamma, \beta)$ and identify all local minima.
(d) Run QAOA with $p = 2, 3$. How does the approximation ratio improve?

### Exercise 2: Weighted MaxCut

Extend the QAOA implementation to handle weighted graphs, where each edge $(i,j)$ has a weight $w_{ij}$:

$$C(z) = \sum_{(i,j) \in E} w_{ij} (z_i \oplus z_j)$$

(a) Modify the cost Hamiltonian to include weights.
(b) Test on a 4-vertex graph with edges $\{(0,1,3), (1,2,1), (2,3,2), (0,3,4)\}$ (format: (i, j, weight)).
(c) Compare QAOA results with brute force.

### Exercise 3: QAOA for Max Independent Set

An independent set $S$ of a graph $G = (V, E)$ is a subset of vertices with no edges between them. The Max Independent Set problem asks for the largest such set.

(a) Formulate Max Independent Set as an Ising Hamiltonian with penalty terms for adjacent selected vertices.
(b) Implement QAOA for this problem.
(c) Test on the Petersen graph (10 vertices). Compare with the known maximum independent set size of 4.

### Exercise 4: Noise Effects on QAOA

Simulate the effect of depolarizing noise on QAOA:
(a) After each gate in the QAOA circuit, apply depolarizing noise with probability $p_{\text{noise}}$.
(b) Run MaxCut QAOA on the triangle graph with $p_{\text{noise}} = 0, 0.001, 0.01, 0.05$.
(c) Plot the approximation ratio vs noise level.
(d) For each noise level, what is the optimal QAOA depth $p$? (Deeper circuits accumulate more noise.)

### Exercise 5: Parameter Concentration

(a) Generate 20 random 3-regular graphs with 8 vertices each.
(b) Run $p=1$ QAOA on each graph and record the optimal $(\gamma^*, \beta^*)$.
(c) How concentrated are the optimal parameters across different graphs? (Plot the distribution.)
(d) Use the average optimal parameters from one set of graphs on a new set. How much performance is lost compared to instance-specific optimization?

---

[← Previous: Variational Quantum Eigensolver](13_VQE.md) | [Next: Quantum Machine Learning →](15_Quantum_Machine_Learning.md)
