# Lesson 25: Capstone Quantum Application

[← Previous: Qiskit Deep Dive](24_Qiskit_Deep_Dive.md) | [Back to Overview](00_Overview.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Design an end-to-end quantum computing project from problem formulation to result analysis
2. Map a real-world optimization problem to a quantum algorithm (QAOA or VQE)
3. Construct, transpile, and simulate a complete quantum circuit with noise
4. Apply error mitigation techniques to improve results from noisy simulations
5. Benchmark quantum solutions against classical baselines
6. Critically evaluate when quantum approaches provide genuine advantage
7. Document and present quantum computing results with appropriate caveats

---

This capstone lesson brings together everything from the preceding 24 lessons into a single, complete project. You will design, implement, simulate, and analyze a quantum algorithm for a practical problem — the **Molecular Ground State Energy** problem using VQE and the **Max-Cut** problem using QAOA. Both are leading candidates for near-term quantum advantage.

The goal is not merely to run the algorithm, but to engage with the full engineering workflow: choosing the right problem encoding, designing an efficient ansatz, handling noise, applying error mitigation, and honestly evaluating the results against classical alternatives. This mirrors the workflow of a quantum computing researcher or applications engineer.

> **No analogy needed here.** This is the real thing — applying quantum algorithms to solve actual problems. The capstone tests your understanding of quantum mechanics (Lessons 1-6), algorithms (Lessons 7-10, 18-19), error handling (Lessons 11, 20), applications (Lessons 13-15, 21-23), and tools (Lesson 24).

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project A: Molecular Ground State via VQE](#2-project-a-molecular-ground-state-via-vqe)
3. [Project B: Max-Cut via QAOA](#3-project-b-max-cut-via-qaoa)
4. [Common Pipeline: Noise and Mitigation](#4-common-pipeline-noise-and-mitigation)
5. [Classical Baselines](#5-classical-baselines)
6. [Analysis and Evaluation](#6-analysis-and-evaluation)
7. [Python Implementation: VQE Capstone](#7-python-implementation-vqe-capstone)
8. [Python Implementation: QAOA Capstone](#8-python-implementation-qaoa-capstone)
9. [Results Discussion](#9-results-discussion)
10. [Exercises](#10-exercises)

---

## 1. Project Overview

### 1.1 The Two Capstone Projects

| Aspect | Project A: VQE | Project B: QAOA |
|--------|---------------|-----------------|
| Problem | H$_2$ ground state energy | Max-Cut on random graphs |
| Algorithm | Variational Quantum Eigensolver | Quantum Approximate Optimization |
| Qubits | 4 (H$_2$ STO-3G) | 4-8 (graph vertices) |
| Ansatz | UCCSD-inspired | $p$-layer QAOA |
| Classical optimizer | COBYLA / SPSA | COBYLA |
| Noise model | Depolarizing + readout | Depolarizing + readout |
| Error mitigation | ZNE + measurement calibration | Measurement calibration |
| Classical baseline | Exact diagonalization, HF | Brute-force, SDP relaxation |

### 1.2 End-to-End Workflow

```
1. Problem Definition
   └→ What are we computing? What is the correct answer?

2. Quantum Encoding
   └→ Map problem to Hamiltonian / circuit

3. Ansatz Design
   └→ Choose parameterized circuit (expressibility vs. trainability)

4. Simulation (Ideal)
   └→ Verify algorithm works without noise

5. Noise Simulation
   └→ Add realistic noise model, observe degradation

6. Error Mitigation
   └→ Apply ZNE / readout correction, recover accuracy

7. Classical Comparison
   └→ Compare with exact and approximate classical methods

8. Analysis
   └→ Evaluate accuracy, resource cost, scaling
```

---

## 2. Project A: Molecular Ground State via VQE

### 2.1 Problem Statement

**Goal**: Compute the ground state energy of the H$_2$ molecule at bond length $R = 0.74$ angstroms using the STO-3G basis set.

**Expected answer**: $E_0 \approx -1.137$ Ha (exact within the basis set)
**Chemical accuracy threshold**: Error $< 1.6$ mHa

### 2.2 Hamiltonian Construction

Following Lesson 21 (Quantum Chemistry):

1. Compute molecular integrals $h_{pq}$ and $h_{pqrs}$ for H$_2$/STO-3G
2. Apply Jordan-Wigner transform to get qubit Hamiltonian
3. Result: $\sim 15$ Pauli strings on 4 qubits

### 2.3 Ansatz Choice

We use a UCCSD-inspired ansatz:

```
|ψ(θ)⟩ = U(θ)|HF⟩ = exp(θ(a†₂a†₃a₁a₀ - a†₀a†₁a₃a₂))|1100⟩
```

After Jordan-Wigner mapping, this becomes a parameterized circuit with $\sim 10$-$20$ CNOT gates.

### 2.4 Optimization

**Optimizer**: COBYLA (derivative-free, robust to noise)
**Initial parameters**: $\theta = 0$ (Hartree-Fock starting point)
**Convergence criterion**: Energy change $< 10^{-5}$ Ha between iterations

---

## 3. Project B: Max-Cut via QAOA

### 3.1 Problem Statement

**Goal**: Find an approximate maximum cut of a random graph $G = (V, E)$ with $|V| = 6$ vertices.

**Max-Cut**: Partition vertices into two sets $S, \bar{S}$ to maximize the number of edges between them:

$$C(z) = \sum_{(i,j) \in E} \frac{1 - z_i z_j}{2}$$

where $z_i \in \{+1, -1\}$ indicates which set vertex $i$ belongs to.

### 3.2 QAOA Circuit

Following Lesson 14:

$$|\gamma, \beta\rangle = \prod_{l=1}^{p} e^{-i\beta_l H_M} e^{-i\gamma_l H_C} |+\rangle^{\otimes n}$$

where $H_C = \sum_{(i,j) \in E} \frac{1-Z_iZ_j}{2}$ and $H_M = \sum_i X_i$.

### 3.3 QAOA Depth

| Depth $p$ | Parameters | Approximation ratio (typical) |
|-----------|-----------|------------------------------|
| 1 | 2 | $\geq 0.6924$ (guaranteed for 3-regular) |
| 2 | 4 | $\sim 0.75$-$0.85$ |
| 3 | 6 | $\sim 0.80$-$0.90$ |
| $p \to \infty$ | $2p$ | $\to 1.0$ (exact) |

---

## 4. Common Pipeline: Noise and Mitigation

### 4.1 Noise Model

We simulate a realistic NISQ device:

| Parameter | Value |
|-----------|-------|
| Single-qubit gate error | $5 \times 10^{-4}$ |
| Two-qubit gate error | $5 \times 10^{-3}$ |
| Readout error | $2 \times 10^{-2}$ |
| $T_1$ | $100$ $\mu$s |
| $T_2$ | $80$ $\mu$s |
| Gate time (1Q) | $50$ ns |
| Gate time (2Q) | $300$ ns |

### 4.2 Error Mitigation Strategy

1. **Readout calibration**: Characterize measurement errors using basis state preparation
2. **Zero-noise extrapolation**: Run at noise factors $1\times, 1.5\times, 2\times$ and extrapolate
3. **Post-selection**: Discard results violating known symmetries (e.g., particle number)

---

## 5. Classical Baselines

### 5.1 For VQE (Chemistry)

| Method | Result | Time |
|--------|--------|------|
| Hartree-Fock | $E_{\text{HF}} = -1.117$ Ha | $O(N^4)$ |
| Full CI (exact) | $E_0 = -1.137$ Ha | $O(2^N)$ |
| CCSD(T) | $E_0 \approx -1.137$ Ha | $O(N^7)$ |

### 5.2 For QAOA (Max-Cut)

| Method | Guarantee | Time |
|--------|-----------|------|
| Brute force | Optimal | $O(2^n)$ |
| Goemans-Williamson SDP | $\geq 0.878 \times \text{OPT}$ | $O(n^3)$ |
| Greedy | $\geq 0.5 \times \text{OPT}$ | $O(n)$ |

---

## 6. Analysis and Evaluation

### 6.1 Metrics

| Metric | VQE | QAOA |
|--------|-----|------|
| Primary | Energy error (mHa) | Approximation ratio |
| Secondary | Number of circuit evaluations | Success probability |
| Resource | Total CNOT count | Circuit depth |
| Noise impact | Energy bias | Ratio degradation |

### 6.2 Key Questions

1. Does the quantum algorithm find the correct answer (within error bars)?
2. How much does noise degrade the result?
3. How much does error mitigation recover?
4. At what problem size would the quantum approach outperform the classical baseline?
5. What are the dominant error sources and how could they be reduced?

---

## 7. Python Implementation: VQE Capstone

```python
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

# === Pauli matrices ===
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def kron_list(ops):
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

# === Step 1: Build H2 Hamiltonian ===
def build_h2_qubit_hamiltonian():
    """Build the H2/STO-3G qubit Hamiltonian at R=0.74 A.

    Uses pre-computed Pauli coefficients from Jordan-Wigner transform.
    These coefficients are standard for the H2 benchmark.
    """
    n_qubits = 4
    N = 2 ** n_qubits

    # H2 STO-3G Hamiltonian in JW representation (standard coefficients)
    # H = g0*I + g1*Z0 + g2*Z1 + g3*Z2 + g4*Z3
    #   + g5*Z0Z1 + g6*Z0Z2 + g7*Z0Z3 + g8*Z1Z2 + g9*Z1Z3 + g10*Z2Z3
    #   + g11*(X0X1Y2Y3 - X0Y1Y2X3 + Y0X1X2Y3 - Y0Y1X2X3)

    # Coefficients at R = 0.74 A
    g = {
        'I': -0.8105,
        'Z0': 0.1721,
        'Z1': 0.1721,
        'Z2': -0.2232,
        'Z3': -0.2232,
        'Z0Z1': 0.1686,
        'Z0Z2': 0.1205,
        'Z0Z3': 0.1659,
        'Z1Z2': 0.1659,
        'Z1Z3': 0.1205,
        'Z2Z3': 0.1743,
        'XXYY': -0.0453,
    }

    H = g['I'] * kron_list([I2]*4)
    H += g['Z0'] * kron_list([Z, I2, I2, I2])
    H += g['Z1'] * kron_list([I2, Z, I2, I2])
    H += g['Z2'] * kron_list([I2, I2, Z, I2])
    H += g['Z3'] * kron_list([I2, I2, I2, Z])
    H += g['Z0Z1'] * kron_list([Z, Z, I2, I2])
    H += g['Z0Z2'] * kron_list([Z, I2, Z, I2])
    H += g['Z0Z3'] * kron_list([Z, I2, I2, Z])
    H += g['Z1Z2'] * kron_list([I2, Z, Z, I2])
    H += g['Z1Z3'] * kron_list([I2, Z, I2, Z])
    H += g['Z2Z3'] * kron_list([I2, I2, Z, Z])

    # Double excitation term
    c = g['XXYY']
    H += c * (kron_list([X,X,Y,Y]) - kron_list([X,Y,Y,X])
            + kron_list([Y,X,X,Y]) - kron_list([Y,Y,X,X]))

    return H, n_qubits


def vqe_ansatz(params, n_qubits):
    """Build the UCCSD-inspired VQE ansatz for H2.

    Circuit: |HF> -> Ry layer -> CNOT entangling -> Ry layer -> double excitation

    The key parameter is the double excitation angle theta,
    which controls the correlation between bonding and antibonding orbitals.
    """
    N = 2 ** n_qubits

    # Start from Hartree-Fock state |1100>
    state = np.zeros(N, dtype=complex)
    state[0b1100] = 1.0  # Orbitals 0,1 occupied

    # Layer 1: Single-qubit rotations
    for q in range(n_qubits):
        angle = params[q]
        ry = np.array([[np.cos(angle/2), -np.sin(angle/2)],
                       [np.sin(angle/2), np.cos(angle/2)]], dtype=complex)
        ops = [I2] * n_qubits
        ops[q] = ry
        state = kron_list(ops) @ state

    # CNOT entangling (ring)
    for q in range(n_qubits - 1):
        cnot = np.eye(N, dtype=complex)
        for s in range(N):
            if (s >> (n_qubits - 1 - q)) & 1:
                new_s = s ^ (1 << (n_qubits - 1 - (q + 1)))
                cnot[s, s] = 0
                cnot[new_s, s] = 1
        state = cnot @ state

    # Layer 2: Single-qubit rotations
    for q in range(n_qubits):
        angle = params[n_qubits + q]
        ry = np.array([[np.cos(angle/2), -np.sin(angle/2)],
                       [np.sin(angle/2), np.cos(angle/2)]], dtype=complex)
        ops = [I2] * n_qubits
        ops[q] = ry
        state = kron_list(ops) @ state

    # Double excitation: exp(i*theta*(|1100><0011| - h.c.))
    theta = params[2 * n_qubits]
    idx_hf = 0b1100  # |1100> = 12
    idx_ex = 0b0011  # |0011> = 3
    a, b = state[idx_hf], state[idx_ex]
    state[idx_hf] = np.cos(theta) * a - np.sin(theta) * b
    state[idx_ex] = np.sin(theta) * a + np.cos(theta) * b

    return state


def add_depolarizing_noise(state, n_qubits, error_rate):
    """Add depolarizing noise to a state vector (convert to density matrix)."""
    rho = np.outer(state, state.conj())
    N = 2 ** n_qubits
    noise_strength = 1 - (1 - error_rate) ** 20  # Approximate: 20 gates
    rho_noisy = (1 - noise_strength) * rho + noise_strength * np.eye(N) / N
    return rho_noisy


def vqe_energy(params, H, n_qubits, noise_level=0):
    """Compute the VQE energy <psi(params)|H|psi(params)>.

    Args:
        params: Ansatz parameters
        H: Qubit Hamiltonian
        n_qubits: Number of qubits
        noise_level: Depolarizing noise rate (0 = noiseless)

    Returns:
        Energy expectation value
    """
    state = vqe_ansatz(params, n_qubits)

    if noise_level > 0:
        rho = add_depolarizing_noise(state, n_qubits, noise_level)
        energy = np.real(np.trace(H @ rho))
    else:
        energy = np.real(state.conj() @ H @ state)

    return energy


# === Run VQE Capstone ===
print("=" * 70)
print("CAPSTONE PROJECT A: VQE for H2 Ground State Energy")
print("=" * 70)

H, n_qubits = build_h2_qubit_hamiltonian()
exact_energy = np.min(np.linalg.eigvalsh(H))
hf_energy = vqe_energy(np.zeros(2 * n_qubits + 1), H, n_qubits)

print(f"\nSystem: H2 / STO-3G / R = 0.74 A")
print(f"Qubits: {n_qubits}")
print(f"Exact ground state energy: {exact_energy:.6f} Ha")
print(f"Hartree-Fock energy: {hf_energy:.6f} Ha")
print(f"Correlation energy: {(exact_energy - hf_energy)*1000:.2f} mHa")

# Step 1: Ideal VQE
print(f"\n--- Step 1: Ideal (noiseless) VQE ---")
n_params = 2 * n_qubits + 1
best_energy_ideal = float('inf')
best_params_ideal = None

for trial in range(5):
    params0 = np.random.uniform(-0.1, 0.1, n_params)
    result = minimize(lambda p: vqe_energy(p, H, n_qubits, 0),
                     params0, method='COBYLA', options={'maxiter': 500})
    if result.fun < best_energy_ideal:
        best_energy_ideal = result.fun
        best_params_ideal = result.x

error_ideal = abs(best_energy_ideal - exact_energy) * 1000
print(f"VQE energy: {best_energy_ideal:.6f} Ha")
print(f"Error: {error_ideal:.2f} mHa")
print(f"Chemical accuracy (<1.6 mHa): {'YES' if error_ideal < 1.6 else 'NO'}")

# Step 2: Noisy VQE
print(f"\n--- Step 2: Noisy VQE (depolarizing p=0.005) ---")
noise_level = 0.005
best_energy_noisy = float('inf')

for trial in range(5):
    params0 = np.random.uniform(-0.1, 0.1, n_params)
    result = minimize(lambda p: vqe_energy(p, H, n_qubits, noise_level),
                     params0, method='COBYLA', options={'maxiter': 500})
    if result.fun < best_energy_noisy:
        best_energy_noisy = result.fun

error_noisy = abs(best_energy_noisy - exact_energy) * 1000
print(f"VQE energy: {best_energy_noisy:.6f} Ha")
print(f"Error: {error_noisy:.2f} mHa")
print(f"Noise degradation: {error_noisy - error_ideal:.2f} mHa")

# Step 3: ZNE
print(f"\n--- Step 3: Zero-Noise Extrapolation ---")
noise_levels = [1.0, 1.5, 2.0, 2.5]
energies_at_noise = []

for factor in noise_levels:
    e = vqe_energy(best_params_ideal, H, n_qubits, noise_level * factor)
    energies_at_noise.append(e)
    print(f"  Noise x{factor:.1f}: E = {e:.6f} Ha")

# Linear extrapolation
coeffs = np.polyfit(noise_levels, energies_at_noise, 1)
e_zne_linear = np.polyval(coeffs, 0)
error_zne = abs(e_zne_linear - exact_energy) * 1000

print(f"\nZNE (linear) energy: {e_zne_linear:.6f} Ha")
print(f"ZNE error: {error_zne:.2f} mHa")
print(f"Improvement over noisy: {error_noisy - error_zne:.2f} mHa")

# Summary
print(f"\n--- Summary ---")
print(f"{'Method':>25} {'Energy (Ha)':>14} {'Error (mHa)':>14}")
print(f"{'-' * 55}")
print(f"{'Exact (Full CI)':>25} {exact_energy:14.6f} {'0.00':>14}")
print(f"{'Hartree-Fock':>25} {hf_energy:14.6f} {abs(hf_energy-exact_energy)*1000:14.2f}")
print(f"{'VQE (ideal)':>25} {best_energy_ideal:14.6f} {error_ideal:14.2f}")
print(f"{'VQE (noisy)':>25} {best_energy_noisy:14.6f} {error_noisy:14.2f}")
print(f"{'VQE + ZNE':>25} {e_zne_linear:14.6f} {error_zne:14.2f}")
```

---

## 8. Python Implementation: QAOA Capstone

```python
import numpy as np
from scipy.optimize import minimize

def build_maxcut_hamiltonian(adjacency):
    """Build the Max-Cut cost Hamiltonian.

    C = sum_{(i,j) in E} (1 - Z_i Z_j) / 2

    Each edge contributes 1 to the cost if the adjacent vertices
    are in different partitions (Z_i != Z_j).
    """
    n = adjacency.shape[0]
    N = 2 ** n
    H_C = np.zeros((N, N), dtype=complex)

    n_edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                n_edges += 1
                ops = [I2] * n
                ops[i] = Z
                ops[j] = Z
                ZiZj = kron_list(ops)
                H_C += (np.eye(N) - ZiZj) / 2

    return H_C, n_edges


def build_mixer_hamiltonian(n):
    """Build the QAOA mixer Hamiltonian: H_M = sum_i X_i."""
    N = 2 ** n
    H_M = np.zeros((N, N), dtype=complex)
    for i in range(n):
        ops = [I2] * n
        ops[i] = X
        H_M += kron_list(ops)
    return H_M


def qaoa_state(gamma, beta, H_C, H_M, n):
    """Prepare the QAOA state for given parameters.

    |gamma, beta> = prod_l exp(-i*beta_l*H_M) exp(-i*gamma_l*H_C) |+>^n
    """
    N = 2 ** n
    state = np.ones(N, dtype=complex) / np.sqrt(N)  # |+>^n

    p = len(gamma)
    for l in range(p):
        # Cost layer
        state = expm(-1j * gamma[l] * H_C) @ state
        # Mixer layer
        state = expm(-1j * beta[l] * H_M) @ state

    return state


def qaoa_expectation(params, H_C, H_M, n, p, noise_level=0):
    """Compute <C> for QAOA with given parameters."""
    gamma = params[:p]
    beta = params[p:]
    state = qaoa_state(gamma, beta, H_C, H_M, n)

    if noise_level > 0:
        N = 2 ** n
        rho = np.outer(state, state.conj())
        noise = 1 - (1 - noise_level) ** (2 * p * n)
        rho = (1 - noise) * rho + noise * np.eye(N) / N
        return -np.real(np.trace(H_C @ rho))  # Negative for minimization
    else:
        return -np.real(state.conj() @ H_C @ state)


def brute_force_maxcut(adjacency):
    """Find the maximum cut by brute force."""
    n = adjacency.shape[0]
    best_cut = 0
    best_partition = 0

    for z in range(2 ** n):
        cut = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j]:
                    zi = 1 - 2 * ((z >> (n - 1 - i)) & 1)
                    zj = 1 - 2 * ((z >> (n - 1 - j)) & 1)
                    cut += (1 - zi * zj) / 2

        if cut > best_cut:
            best_cut = cut
            best_partition = z

    return int(best_cut), best_partition


# === Run QAOA Capstone ===
print("\n" + "=" * 70)
print("CAPSTONE PROJECT B: QAOA for Max-Cut")
print("=" * 70)

# Generate a random graph
np.random.seed(42)
n = 6
edge_prob = 0.5
adjacency = np.zeros((n, n), dtype=int)
for i in range(n):
    for j in range(i + 1, n):
        if np.random.random() < edge_prob:
            adjacency[i, j] = 1
            adjacency[j, i] = 1

H_C, n_edges = build_maxcut_hamiltonian(adjacency)
H_M = build_mixer_hamiltonian(n)

# Brute force optimal
optimal_cut, optimal_partition = brute_force_maxcut(adjacency)

print(f"\nGraph: {n} vertices, {n_edges} edges")
print(f"Adjacency matrix:")
for row in adjacency:
    print(f"  {row}")
print(f"\nOptimal cut: {optimal_cut}")
print(f"Optimal partition: {format(optimal_partition, f'0{n}b')}")

# Run QAOA at different depths
print(f"\n--- QAOA Results (Ideal) ---")
print(f"{'Depth p':>10} {'Best <C>':>12} {'Ratio':>10} {'Params':>30}")
print(f"{'-' * 65}")

for p in [1, 2, 3, 4]:
    best_val = 0
    best_params = None

    for trial in range(10):
        params0 = np.random.uniform(0, np.pi, 2 * p)
        result = minimize(lambda par: qaoa_expectation(par, H_C, H_M, n, p),
                         params0, method='COBYLA', options={'maxiter': 300})
        val = -result.fun
        if val > best_val:
            best_val = val
            best_params = result.x

    ratio = best_val / optimal_cut
    params_str = ', '.join(f'{x:.2f}' for x in best_params[:4])
    print(f"{p:10d} {best_val:12.4f} {ratio:10.4f} [{params_str}...]")

# Noisy QAOA
print(f"\n--- QAOA Results (Noisy, p=2) ---")
p = 2
noise_levels_test = [0, 0.001, 0.005, 0.01, 0.02]

print(f"{'Noise':>10} {'<C>':>12} {'Ratio':>10} {'Degradation':>14}")
print(f"{'-' * 50}")

for nl in noise_levels_test:
    best_val = 0
    for trial in range(5):
        params0 = np.random.uniform(0, np.pi, 2 * p)
        result = minimize(lambda par: qaoa_expectation(par, H_C, H_M, n, p, nl),
                         params0, method='COBYLA', options={'maxiter': 200})
        val = -result.fun
        if val > best_val:
            best_val = val

    ratio = best_val / optimal_cut
    degrad = 1.0 - ratio
    print(f"{nl:10.4f} {best_val:12.4f} {ratio:10.4f} {degrad:14.4f}")

# Sampling analysis
print(f"\n--- Sampling Analysis (p=2, ideal) ---")
p = 2
best_params_qaoa = None
best_val_qaoa = 0

for trial in range(20):
    params0 = np.random.uniform(0, np.pi, 2 * p)
    result = minimize(lambda par: qaoa_expectation(par, H_C, H_M, n, p),
                     params0, method='COBYLA', options={'maxiter': 300})
    if -result.fun > best_val_qaoa:
        best_val_qaoa = -result.fun
        best_params_qaoa = result.x

# Get the state and compute probabilities
gamma = best_params_qaoa[:p]
beta = best_params_qaoa[p:]
state = qaoa_state(gamma, beta, H_C, H_M, n)
probs = np.abs(state) ** 2

# Compute cut value for each bitstring
cuts = []
for z in range(2 ** n):
    cut = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                zi = 1 - 2 * ((z >> (n - 1 - i)) & 1)
                zj = 1 - 2 * ((z >> (n - 1 - j)) & 1)
                cut += (1 - zi * zj) / 2
    cuts.append(int(cut))

# Sort by probability
sorted_indices = np.argsort(-probs)
print(f"\nTop 10 measurement outcomes:")
print(f"{'Bitstring':>12} {'Probability':>14} {'Cut value':>12} {'Optimal?':>10}")
print(f"{'-' * 52}")
for idx in sorted_indices[:10]:
    bs = format(idx, f'0{n}b')
    is_opt = '***' if cuts[idx] == optimal_cut else ''
    print(f"{bs:>12} {probs[idx]:14.4f} {cuts[idx]:12d} {is_opt:>10}")

# Probability of finding optimal solution
p_optimal = sum(probs[z] for z in range(2**n) if cuts[z] == optimal_cut)
print(f"\nProbability of optimal solution: {p_optimal:.4f}")
print(f"Expected shots to find optimal: {1/p_optimal:.0f}" if p_optimal > 0 else "")
```

---

## 9. Results Discussion

### 9.1 VQE Analysis

Key findings from the VQE capstone:

1. **Ideal VQE** achieves chemical accuracy ($< 1.6$ mHa) for H$_2$ with a simple ansatz
2. **Noise** shifts the energy upward (toward the maximally mixed state energy)
3. **ZNE** partially recovers the ideal energy, reducing the error by $\sim 50$-$70\%$
4. The **dominant error source** is the two-qubit gate error (depolarizing)

### 9.2 QAOA Analysis

Key findings from the QAOA capstone:

1. **Higher depth** ($p$) improves the approximation ratio, approaching the optimal cut
2. **Noise degrades** the approximation ratio, particularly at higher depths (more gates = more errors)
3. For small graphs, **sampling the QAOA state** a few hundred times is sufficient to find the optimal cut with high probability
4. The **optimal parameters** show structure: $\gamma$ values increase and $\beta$ values decrease with layer index

### 9.3 Quantum vs. Classical Verdict

For the problem sizes in this capstone:

- **VQE**: Classical exact diagonalization is trivially fast for 4 qubits. Quantum advantage requires $\gtrsim 50$ qubits (beyond classical Full CI)
- **QAOA**: Classical brute force is instant for 6 vertices. Quantum advantage for Max-Cut is expected for $\gtrsim 100$ vertices (beyond GW-SDP)

**Honest assessment**: For these small problem sizes, quantum computers offer no advantage. The value of these exercises is in validating the algorithms and building intuition for when quantum resources become sufficient.

---

## 10. Exercises

### Exercise 1: Extended VQE Project

Extend the VQE capstone to LiH (12 qubits in STO-3G):
(a) Construct the LiH Hamiltonian using Jordan-Wigner transform.
(b) Use qubit tapering to reduce the qubit count.
(c) Design an ADAPT-VQE ansatz and compare convergence with hardware-efficient ansatz.
(d) Compute the potential energy curve from $R = 0.5$ to $R = 4.0$ A.
(e) At what bond length does classical HF fail most dramatically?

### Exercise 2: QAOA Scaling

Study QAOA scaling for Max-Cut:
(a) Generate random 3-regular graphs with $n = 4, 6, 8, 10, 12$ vertices.
(b) Run QAOA at $p = 1, 2, 3$ for each graph size.
(c) Plot the approximation ratio vs. graph size for each depth.
(d) At what size does the classical optimizer (COBYLA) start struggling with the parameter landscape?
(e) Implement the INTERP strategy for transferring optimal parameters from smaller to larger instances.

### Exercise 3: Complete Noise Pipeline

Build a comprehensive noise simulation pipeline:
(a) Model thermal relaxation ($T_1, T_2$) during gate execution.
(b) Add correlated errors (crosstalk between adjacent qubits).
(c) Compare the impact of different noise sources on VQE and QAOA.
(d) Determine the noise threshold below which VQE achieves chemical accuracy for H$_2$.

### Exercise 4: Error Mitigation Comparison

Compare error mitigation techniques on both capstone projects:
(a) Implement readout calibration, ZNE, and symmetry verification.
(b) Apply each technique individually and in combination.
(c) Which technique provides the most improvement? Which is most efficient (fewest extra circuits)?
(d) At what noise level does each technique fail to provide meaningful improvement?

### Exercise 5: Your Own Capstone

Design and implement your own quantum computing capstone project. Choose one:
(a) **Quantum chemistry**: Ground state of H$_2$O using an active space VQE.
(b) **Optimization**: Portfolio optimization using QAOA.
(c) **Quantum ML**: Variational quantum classifier for a simple dataset.
(d) **Simulation**: Time evolution of a Heisenberg spin chain.

For your chosen project:
- Define the problem and expected answer
- Design the quantum algorithm and ansatz
- Implement and test without noise
- Add realistic noise and apply error mitigation
- Compare with classical baselines
- Write a 1-page report summarizing findings

---

[← Previous: Qiskit Deep Dive](24_Qiskit_Deep_Dive.md) | [Back to Overview](00_Overview.md)
