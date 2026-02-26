# Lesson 13: Variational Quantum Eigensolver (VQE)

[← Previous: Quantum Teleportation and Communication](12_Quantum_Teleportation.md) | [Next: QAOA and Combinatorial Optimization →](14_QAOA.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the molecular electronic structure problem and why it is classically intractable
2. State the variational principle and how it guarantees an upper bound on the ground state energy
3. Describe the hybrid quantum-classical VQE loop
4. Compare hardware-efficient and chemically-inspired ansatze (UCCSD)
5. Explain Hamiltonian encoding methods: Jordan-Wigner and Bravyi-Kitaev transformations
6. Walk through a VQE calculation for the H₂ molecule
7. Discuss practical challenges: barren plateaus, noise, and optimizer convergence

---

Quantum chemistry is widely considered one of the most promising near-term applications of quantum computers. The electronic structure problem — finding the ground state energy of a molecule — is exponentially hard for classical computers because the number of quantum states grows exponentially with the number of electrons. Even with decades of algorithmic development, classical methods like Full Configuration Interaction (FCI) are limited to ~20 electrons, and approximate methods (DFT, CCSD(T)) sacrifice accuracy for tractability.

The Variational Quantum Eigensolver (VQE), proposed by Peruzzo et al. in 2014, was designed specifically for noisy intermediate-scale quantum (NISQ) devices. Unlike Shor's algorithm, which requires deep fault-tolerant circuits, VQE uses shallow quantum circuits combined with classical optimization. This hybrid approach trades circuit depth for many circuit repetitions, making it more resilient to noise — the dominant challenge on current hardware.

> **Analogy:** VQE is like tuning a radio. The quantum circuit generates candidate solutions (static), and the classical optimizer adjusts the dial (parameters) until the signal (energy) is at its clearest (minimum). You do not need to understand the internal workings of the radio (full quantum state) — you just need to evaluate the output quality (energy expectation value) and adjust.

## Table of Contents

1. [The Electronic Structure Problem](#1-the-electronic-structure-problem)
2. [The Variational Principle](#2-the-variational-principle)
3. [The VQE Algorithm](#3-the-vqe-algorithm)
4. [Ansatz Design](#4-ansatz-design)
5. [Hamiltonian Encoding](#5-hamiltonian-encoding)
6. [VQE for H₂: A Complete Walkthrough](#6-vqe-for-h-a-complete-walkthrough)
7. [Classical Optimizer Strategies](#7-classical-optimizer-strategies)
8. [Challenges and Limitations](#8-challenges-and-limitations)
9. [Python Implementation](#9-python-implementation)
10. [Exercises](#10-exercises)

---

## 1. The Electronic Structure Problem

### 1.1 The Problem

Given a molecule with $M$ nuclei at fixed positions (Born-Oppenheimer approximation) and $N$ electrons, find the ground state energy $E_0$ of the electronic Hamiltonian:

$$\hat{H} = -\sum_{i=1}^{N} \frac{\nabla_i^2}{2} - \sum_{i=1}^{N}\sum_{A=1}^{M} \frac{Z_A}{|\mathbf{r}_i - \mathbf{R}_A|} + \sum_{i<j}^{N} \frac{1}{|\mathbf{r}_i - \mathbf{r}_j|}$$

The three terms represent:
1. **Kinetic energy** of electrons
2. **Electron-nucleus attraction** (Coulomb)
3. **Electron-electron repulsion** (Coulomb)

### 1.2 Why It Is Hard

In second quantization, the Hamiltonian is written in terms of creation ($a_p^\dagger$) and annihilation ($a_q$) operators acting on $K$ molecular orbitals:

$$\hat{H} = \sum_{pq} h_{pq} \, a_p^\dagger a_q + \frac{1}{2}\sum_{pqrs} h_{pqrs} \, a_p^\dagger a_q^\dagger a_r a_s$$

The Hilbert space dimension is $\binom{2K}{N}$, which grows exponentially. For a modest molecule like caffeine ($C_8H_{10}N_4O_2$) with ~100 electrons and ~200 orbitals, the Hilbert space has $\sim 10^{75}$ dimensions — far beyond any classical computer.

### 1.3 Classical Approaches and Their Limits

| Method | Accuracy | Scaling | Max system size |
|--------|----------|---------|-----------------|
| Hartree-Fock (HF) | ~99% correlation energy missed | $O(K^4)$ | Thousands of atoms |
| DFT | ~1 kcal/mol (depends on functional) | $O(K^3)$ | Thousands of atoms |
| MP2 | Good for weak correlation | $O(K^5)$ | Hundreds of atoms |
| CCSD(T) | "Gold standard" ~1 kcal/mol | $O(K^7)$ | ~30 atoms |
| FCI (exact) | Exact (within basis) | $O(K!)$ | ~20 electrons |

The quantum advantage hypothesis: a quantum computer with $K$ qubits can represent the full $\binom{2K}{N}$ Hilbert space directly, potentially solving the electronic structure problem in polynomial time.

---

## 2. The Variational Principle

### 2.1 Statement

For any normalized trial state $|\psi(\boldsymbol{\theta})\rangle$:

$$E(\boldsymbol{\theta}) = \langle\psi(\boldsymbol{\theta})|\hat{H}|\psi(\boldsymbol{\theta})\rangle \geq E_0$$

where $E_0$ is the true ground state energy. Equality holds if and only if $|\psi(\boldsymbol{\theta})\rangle$ is the ground state.

### 2.2 Proof

Expand the trial state in the eigenbasis of $\hat{H}$: $|\psi\rangle = \sum_i c_i |E_i\rangle$ where $\hat{H}|E_i\rangle = E_i|E_i\rangle$ and $E_0 \leq E_1 \leq \cdots$.

$$\langle\psi|\hat{H}|\psi\rangle = \sum_i |c_i|^2 E_i \geq E_0 \sum_i |c_i|^2 = E_0$$

### 2.3 Significance for VQE

The variational principle guarantees that:
1. Any trial state gives an **upper bound** on the ground state energy
2. **Minimizing** the energy over the parameter space $\boldsymbol{\theta}$ gives the best approximation within the ansatz family
3. We never need to compute the full eigendecomposition — just the expectation value of $\hat{H}$

This is the foundation of VQE: use a quantum computer to prepare $|\psi(\boldsymbol{\theta})\rangle$ and estimate $\langle\hat{H}\rangle$, then use a classical optimizer to find the parameters $\boldsymbol{\theta}^*$ that minimize the energy.

---

## 3. The VQE Algorithm

### 3.1 Algorithm Overview

```
┌──────────────────────────────────────────────────┐
│              Classical Computer                   │
│                                                   │
│  1. Initialize parameters θ                       │
│  2. Receive E(θ) from quantum computer            │
│  3. Update θ using classical optimizer             │
│  4. Check convergence → if not, go to step 5      │
│  5. Send updated θ to quantum computer             │
│                                                   │
└───────────────┬──────────────────┬───────────────┘
                │                  ↑
                ↓                  │
┌───────────────┴──────────────────┴───────────────┐
│              Quantum Computer                     │
│                                                   │
│  1. Prepare |ψ(θ)⟩ using parameterized circuit    │
│  2. Measure ⟨ψ(θ)|H|ψ(θ)⟩ term by term          │
│  3. Return E(θ) to classical computer              │
│                                                   │
└──────────────────────────────────────────────────┘
```

### 3.2 Detailed Steps

**Step 1: Hamiltonian decomposition**

Write $\hat{H}$ as a sum of Pauli strings (after qubit mapping):

$$\hat{H} = \sum_i c_i P_i, \quad P_i \in \{I, X, Y, Z\}^{\otimes n}$$

For example, $\hat{H} = 0.5 \, Z_0 Z_1 - 0.3 \, X_0 + 0.2 \, I$.

**Step 2: Initialize parameters**

Choose initial parameters $\boldsymbol{\theta}_0$ (often randomly or from Hartree-Fock).

**Step 3: Prepare trial state**

Apply the parameterized quantum circuit (ansatz) $U(\boldsymbol{\theta})$ to an initial state $|0\rangle^{\otimes n}$:

$$|\psi(\boldsymbol{\theta})\rangle = U(\boldsymbol{\theta})|0\rangle^{\otimes n}$$

**Step 4: Estimate energy**

For each Pauli string $P_i$, measure $\langle P_i \rangle$ by repeated preparation-and-measurement. Sum up:

$$E(\boldsymbol{\theta}) = \sum_i c_i \langle P_i \rangle$$

Each $\langle P_i \rangle$ requires many "shots" (repetitions) for statistical accuracy.

**Step 5: Classical optimization**

Feed $E(\boldsymbol{\theta})$ to a classical optimizer (COBYLA, L-BFGS-B, SPSA, etc.) to get updated parameters.

**Step 6: Iterate** until convergence.

### 3.3 Number of Measurements

If $\hat{H}$ has $M$ Pauli terms and we want energy precision $\epsilon$, each term requires $O(c_i^2 / \epsilon^2)$ measurements. The total measurement count scales as:

$$N_{\text{shots}} \sim O\left(\frac{(\sum_i |c_i|)^2}{\epsilon^2}\right)$$

For typical molecules, $M$ ranges from hundreds to millions of Pauli terms, making measurement overhead a significant bottleneck.

---

## 4. Ansatz Design

### 4.1 What Is an Ansatz?

An **ansatz** (German for "approach" or "assumption") is the parameterized quantum circuit $U(\boldsymbol{\theta})$ that defines the family of trial states. The choice of ansatz critically determines:

- **Expressibility**: Can the ansatz represent the true ground state?
- **Trainability**: Can the optimizer find good parameters?
- **Hardware efficiency**: How deep is the circuit?

### 4.2 Hardware-Efficient Ansatz (HEA)

The HEA uses whatever gates the hardware natively supports, arranged in layers:

```
Layer 1:        Layer 2:        ...
|0⟩ ─ Ry(θ₁) ─ CNOT ─ Ry(θ₅) ─ CNOT ─ ...
|0⟩ ─ Ry(θ₂) ─ ──●── Ry(θ₆) ─ ──●── ...
|0⟩ ─ Ry(θ₃) ─ CNOT ─ Ry(θ₇) ─ CNOT ─ ...
|0⟩ ─ Ry(θ₄) ─ ──●── Ry(θ₈) ─ ──●── ...
```

**Pros**: Short circuit depth, hardware-native gates
**Cons**: No chemical intuition, prone to barren plateaus (Section 8), may need many parameters

### 4.3 Unitary Coupled Cluster (UCCSD)

The UCCSD ansatz is inspired by the coupled-cluster method from quantum chemistry:

$$U_{\text{UCCSD}}(\boldsymbol{\theta}) = e^{T(\boldsymbol{\theta}) - T^\dagger(\boldsymbol{\theta})}$$

where $T = T_1 + T_2$ includes single and double excitations:

$$T_1 = \sum_{i,a} \theta_i^a \, a_a^\dagger a_i, \quad T_2 = \sum_{ij,ab} \theta_{ij}^{ab} \, a_a^\dagger a_b^\dagger a_j a_i$$

Here $i, j$ index occupied orbitals and $a, b$ index virtual (unoccupied) orbitals.

**Pros**: Chemically motivated, systematically improvable
**Cons**: Deep circuits (many Trotter steps), many parameters for large systems

### 4.4 Other Ansatze

| Ansatz | Description | Depth | Parameters |
|--------|-------------|-------|-----------|
| HEA | Hardware layers | $O(L)$ layers | $O(nL)$ |
| UCCSD | Chemical excitations | $O(n^4)$ | $O(n^2 K^2)$ |
| ADAPT-VQE | Iteratively grown | Variable | Adaptive |
| Symmetry-preserving | Respects symmetries | $O(n^2)$ | Reduced |
| Qubit-ADAPT | Qubit-level ADAPT | Variable | Adaptive |

---

## 5. Hamiltonian Encoding

### 5.1 The Encoding Problem

The electronic Hamiltonian is written in terms of fermionic operators ($a_p^\dagger$, $a_q$), but quantum computers operate on qubits (spin-1/2 systems). We need a mapping from fermions to qubits.

### 5.2 Jordan-Wigner Transformation

The Jordan-Wigner (JW) mapping assigns each molecular orbital to one qubit:

$$a_p^\dagger \to \frac{1}{2}(X_p - iY_p) \otimes Z_{p-1} \otimes Z_{p-2} \otimes \cdots \otimes Z_0$$

The string of $Z$ operators enforces the fermionic anti-commutation relations $\{a_p, a_q^\dagger\} = \delta_{pq}$.

**Example**: For 4 orbitals:
- $a_0^\dagger = \frac{1}{2}(X_0 - iY_0)$
- $a_1^\dagger = \frac{1}{2}(X_1 - iY_1) \otimes Z_0$
- $a_2^\dagger = \frac{1}{2}(X_2 - iY_2) \otimes Z_1 \otimes Z_0$

**Pros**: Simple, intuitive (qubit $p$ = orbital $p$)
**Cons**: $Z$-strings grow with system size, making some operators highly non-local

### 5.3 Bravyi-Kitaev Transformation

The BK mapping uses a binary tree structure to achieve $O(\log n)$ non-locality instead of $O(n)$:

$$a_p^\dagger \to O(\log n) \text{-weight Pauli operators}$$

This reduces the number of terms in the qubit Hamiltonian and can lead to more efficient circuits.

### 5.4 Example: H₂ Hamiltonian

Using the minimal STO-3G basis and JW transformation, the H₂ Hamiltonian (4 qubits, reduced to 2 by symmetry) becomes:

$$\hat{H}_{H_2} = g_0 I + g_1 Z_0 + g_2 Z_1 + g_3 Z_0 Z_1 + g_4 X_0 X_1 + g_5 Y_0 Y_1$$

where the $g_i$ coefficients depend on the bond length $R$.

At $R = 0.735$ \AA (equilibrium):

$$\hat{H} \approx -0.4804 I + 0.3435 Z_0 - 0.4347 Z_1 + 0.5716 Z_0 Z_1 + 0.0910 X_0 X_1 + 0.0910 Y_0 Y_1$$

---

## 6. VQE for H₂: A Complete Walkthrough

### 6.1 Setup

- **Molecule**: H₂ at bond length $R$
- **Basis**: STO-3G (minimal basis, 2 spatial orbitals, 4 spin-orbitals)
- **Symmetry reduction**: 4 qubits → 2 qubits (using $\mathbb{Z}_2$ symmetries)
- **Ansatz**: Single-parameter $R_y(\theta)$ circuit

### 6.2 Ansatz Circuit

For the 2-qubit reduced H₂ problem, a simple ansatz suffices:

```
|0⟩ ─── X ─── Ry(θ) ─── CNOT ─── M
|0⟩ ─────────────────── ──●──── M
```

The $X$ gate initializes qubit 0 to $|1\rangle$ (representing the Hartree-Fock state $|01\rangle$). The $R_y(\theta)$ gate and CNOT create entanglement, parameterized by $\theta$.

At $\theta = 0$: the state is $|10\rangle$ (Hartree-Fock)
At $\theta = \pi$: the state is $|01\rangle$
Intermediate $\theta$: superposition of configurations

### 6.3 Energy as a Function of θ

The energy $E(\theta)$ is computed by measuring each Pauli term:

$$E(\theta) = g_0 + g_1 \langle Z_0\rangle + g_2 \langle Z_1\rangle + g_3 \langle Z_0 Z_1\rangle + g_4 \langle X_0 X_1\rangle + g_5 \langle Y_0 Y_1\rangle$$

For the simple ansatz, this is a function of the single parameter $\theta$, which we can plot and minimize.

### 6.4 Bond Dissociation Curve

By running VQE at different bond lengths $R$, we obtain the potential energy surface $E(R)$. The minimum gives the equilibrium bond length and dissociation energy.

- **Hartree-Fock** gives a good result near equilibrium but fails at dissociation
- **VQE** can capture the strong correlation at large $R$ (where two electrons become localized on separate atoms)
- **FCI** (exact) is the benchmark

---

## 7. Classical Optimizer Strategies

### 7.1 Gradient-Free Methods

| Optimizer | Description | Pros | Cons |
|-----------|-------------|------|------|
| COBYLA | Constrained optimization by linear approximation | Robust, no gradients | Slow convergence |
| Nelder-Mead | Simplex method | Simple, no gradients | Poor in high dimensions |
| Powell | Direction-set method | Good for smooth landscapes | Sensitive to noise |

### 7.2 Gradient-Based Methods

| Optimizer | Description | Pros | Cons |
|-----------|-------------|------|------|
| L-BFGS-B | Quasi-Newton | Fast convergence | Needs accurate gradients |
| Adam | Adaptive learning rate | Good for noisy gradients | Hyperparameter tuning |
| SPSA | Simultaneous perturbation | Only 2 evaluations per step | Noisy convergence |

### 7.3 Parameter Shift Rule

Quantum gradients can be computed exactly using the **parameter shift rule**. For a gate $R_y(\theta) = e^{-i\theta Y/2}$:

$$\frac{\partial E}{\partial \theta} = \frac{E(\theta + \pi/2) - E(\theta - \pi/2)}{2}$$

This requires only two additional circuit evaluations per parameter, making gradient-based optimization feasible on quantum hardware.

---

## 8. Challenges and Limitations

### 8.1 Barren Plateaus

For random parameterized circuits, the gradient variance decreases **exponentially** with the number of qubits:

$$\text{Var}\left[\frac{\partial E}{\partial \theta_i}\right] \leq O(2^{-n})$$

This means that for large systems, the energy landscape becomes exponentially flat (a "barren plateau"), and gradient-based optimization becomes ineffective — the gradients are too small to provide useful direction.

**Mitigation strategies**:
- Use problem-specific ansatze (UCCSD, symmetry-preserving)
- Layer-by-layer training
- Initialization near classically computed states (e.g., Hartree-Fock)

### 8.2 Noise

On NISQ devices, gate errors ($\sim 10^{-3}$ to $10^{-2}$) and decoherence corrupt the quantum state. For a circuit with $d$ gates, the expected fidelity decreases as:

$$F \approx (1 - p)^d \approx e^{-pd}$$

This limits the useful circuit depth to $d \lesssim 1/p \sim 100\text{-}1000$ gates.

**Mitigation strategies**:
- Error mitigation (zero-noise extrapolation, probabilistic error cancellation)
- Shallow ansatze (hardware-efficient)
- Noise-aware optimization

### 8.3 Measurement Overhead

Estimating $\langle H \rangle$ to precision $\epsilon$ requires $O(M / \epsilon^2)$ measurements, where $M$ is the number of Pauli terms. For large molecules, this can require billions of circuit executions.

**Mitigation strategies**:
- Grouping commuting Pauli terms (measure simultaneously)
- Classical shadows
- Importance sampling

### 8.4 Quantum Advantage Question

It remains an open question whether VQE on NISQ devices can outperform the best classical methods for practically relevant molecules. Current demonstrations are limited to small molecules (H₂, LiH, BeH₂) that are trivially solvable classically.

---

## 9. Python Implementation

### 9.1 Simple VQE for a 2-Qubit Hamiltonian

```python
import numpy as np
from scipy.optimize import minimize

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def tensor(A, B):
    """Tensor (Kronecker) product of two matrices."""
    return np.kron(A, B)

def build_h2_hamiltonian(R=0.735):
    """Build the 2-qubit H₂ Hamiltonian for a given bond length.

    Why these specific coefficients? They come from computing the molecular
    integrals in the STO-3G basis and applying the Jordan-Wigner transformation.
    The coefficients vary with bond length R, defining the potential energy surface.
    """
    # Coefficients for H₂ in STO-3G basis (2-qubit reduced Hamiltonian)
    # These are approximate values at equilibrium bond length R=0.735 Å
    # In a real application, these would be computed from molecular integrals
    if abs(R - 0.735) < 0.01:
        g = [-0.4804, 0.3435, -0.4347, 0.5716, 0.0910, 0.0910]
    else:
        # Simplified parametric model for demonstration
        # Real computation would use PySCF or similar
        g = [-0.4804 + 0.1*(R-0.735),
             0.3435 * np.exp(-0.5*(R-0.735)**2),
             -0.4347 * np.exp(-0.3*(R-0.735)**2),
             0.5716 / (1 + 0.5*abs(R-0.735)),
             0.0910 * np.exp(-0.2*(R-0.735)**2),
             0.0910 * np.exp(-0.2*(R-0.735)**2)]

    H = (g[0] * tensor(I, I) +
         g[1] * tensor(Z, I) +
         g[2] * tensor(I, Z) +
         g[3] * tensor(Z, Z) +
         g[4] * tensor(X, X) +
         g[5] * tensor(Y, Y))

    return H, g

def ansatz_state(theta):
    """Prepare the VQE trial state for the 2-qubit H₂ problem.

    Why this specific circuit? Starting from the Hartree-Fock state |10⟩,
    the Ry rotation and CNOT create a parameterized superposition of
    |10⟩ and |01⟩ — the two dominant configurations for H₂. The parameter
    θ controls the mixing, and the optimizer finds the optimal value.
    """
    # Start with |00⟩
    state = np.array([1, 0, 0, 0], dtype=complex)

    # Apply X to qubit 0: |00⟩ → |10⟩ (Hartree-Fock state)
    X_gate = tensor(X, I)
    state = X_gate @ state

    # Apply Ry(θ) to qubit 0
    Ry = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                   [np.sin(theta/2),  np.cos(theta/2)]], dtype=complex)
    Ry_full = tensor(Ry, I)
    state = Ry_full @ state

    # Apply CNOT (qubit 0 controls qubit 1)
    CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    state = CNOT @ state

    return state

def vqe_energy(theta, H):
    """Compute the energy expectation value ⟨ψ(θ)|H|ψ(θ)⟩.

    On a real quantum computer, this would be estimated from many measurement
    shots. Here we compute it exactly from the state vector.
    """
    state = ansatz_state(theta[0])
    return np.real(state.conj() @ H @ state)

def run_vqe(R=0.735, verbose=True):
    """Run the complete VQE algorithm for H₂.

    Returns the optimized energy and parameters.
    """
    H, g = build_h2_hamiltonian(R)

    # Exact ground state energy (for comparison)
    eigenvalues = np.linalg.eigvalsh(H)
    exact_energy = eigenvalues[0]

    # Hartree-Fock energy (θ=0)
    hf_energy = vqe_energy([0], H)

    if verbose:
        print(f"Bond length R = {R:.3f} Å")
        print(f"Exact ground state energy: {exact_energy:.6f} Ha")
        print(f"Hartree-Fock energy (θ=0): {hf_energy:.6f} Ha")

    # Run optimization
    result = minimize(vqe_energy, x0=[0.1], args=(H,), method='COBYLA',
                     options={'maxiter': 200, 'rhobeg': 0.5})

    vqe_e = result.fun
    opt_theta = result.x[0]

    if verbose:
        print(f"VQE optimized energy: {vqe_e:.6f} Ha")
        print(f"Optimal θ: {opt_theta:.4f} rad")
        print(f"Error: {abs(vqe_e - exact_energy):.2e} Ha")
        print(f"Chemical accuracy (1.6 mHa): "
              f"{'ACHIEVED' if abs(vqe_e - exact_energy) < 0.0016 else 'NOT achieved'}")

    return vqe_e, exact_energy, opt_theta

# Run VQE at equilibrium
print("=" * 55)
print("VQE for H₂ Molecule")
print("=" * 55)
run_vqe(0.735)
```

### 9.2 Energy Landscape Visualization

```python
import numpy as np

def visualize_energy_landscape(R=0.735):
    """Plot the energy as a function of the ansatz parameter θ.

    Why visualize? The energy landscape reveals the optimization difficulty.
    A smooth, single-minimum landscape is easy to optimize; multiple local
    minima or flat regions indicate potential convergence problems.
    """
    H, g = build_h2_hamiltonian(R)

    thetas = np.linspace(-np.pi, np.pi, 200)
    energies = [vqe_energy([t], H) for t in thetas]

    # Exact energies
    eigenvalues = sorted(np.linalg.eigvalsh(H))

    print(f"Energy landscape for H₂ at R = {R:.3f} Å")
    print(f"{'θ':>8} {'E(θ)':>12} {'Distance from E_0':>18}")
    print("-" * 42)

    # Sample a few points
    for t in np.linspace(-np.pi, np.pi, 13):
        e = vqe_energy([t], H)
        dist = e - eigenvalues[0]
        bar = "#" * int(min(dist * 200, 40))
        print(f"{t:8.3f} {e:12.6f} {dist:18.6f} {bar}")

    # Find the minimum
    min_idx = np.argmin(energies)
    print(f"\nMinimum at θ = {thetas[min_idx]:.4f}, E = {energies[min_idx]:.6f}")
    print(f"Exact E₀ = {eigenvalues[0]:.6f}")
    print(f"Exact E₁ = {eigenvalues[1]:.6f}")

visualize_energy_landscape()
```

### 9.3 Bond Dissociation Curve

```python
import numpy as np
from scipy.optimize import minimize

def bond_dissociation_curve():
    """Compute the H₂ potential energy surface using VQE.

    Why scan bond lengths? The PES reveals the equilibrium geometry
    (minimum energy) and dissociation behavior. VQE should capture
    the correct dissociation limit where Hartree-Fock fails.
    """
    print("=" * 55)
    print("H₂ Bond Dissociation Curve: VQE vs HF vs Exact")
    print("=" * 55)

    bond_lengths = np.linspace(0.3, 3.0, 28)
    vqe_energies = []
    hf_energies = []
    exact_energies = []

    for R in bond_lengths:
        H, g = build_h2_hamiltonian(R)
        eigenvalues = sorted(np.linalg.eigvalsh(H))

        # Exact
        exact_energies.append(eigenvalues[0])

        # Hartree-Fock (θ = 0)
        hf_e = vqe_energy([0], H)
        hf_energies.append(hf_e)

        # VQE
        result = minimize(vqe_energy, x0=[0.1], args=(H,), method='COBYLA')
        vqe_energies.append(result.fun)

    # Print comparison table
    print(f"\n{'R (Å)':>8} {'Exact':>10} {'VQE':>10} {'HF':>10} {'VQE err':>10}")
    print("-" * 52)
    for i, R in enumerate(bond_lengths):
        if i % 3 == 0:  # Print every 3rd point
            err = abs(vqe_energies[i] - exact_energies[i])
            print(f"{R:8.3f} {exact_energies[i]:10.6f} {vqe_energies[i]:10.6f} "
                  f"{hf_energies[i]:10.6f} {err:10.2e}")

    # Find equilibrium
    min_idx = np.argmin(exact_energies)
    print(f"\nEquilibrium bond length: R = {bond_lengths[min_idx]:.3f} Å")
    print(f"Equilibrium energy (exact): {exact_energies[min_idx]:.6f} Ha")

bond_dissociation_curve()
```

### 9.4 Measurement Simulation with Shot Noise

```python
import numpy as np

def measure_pauli_expectation(state, pauli_ops, n_shots=1000):
    """Simulate measuring a Pauli operator with finite shots.

    Why simulate shots? On a real quantum computer, we estimate ⟨P⟩ from
    a finite number of measurements. Each measurement gives +1 or -1
    with probabilities determined by the quantum state. The statistical
    error scales as 1/√(n_shots).
    """
    # Build the full Pauli operator
    op = pauli_ops[0]
    for p in pauli_ops[1:]:
        op = np.kron(op, p)

    # Compute exact expectation value for comparison
    exact = np.real(state.conj() @ op @ state)

    # Simulate measurement: diagonalize the Pauli operator
    eigenvalues, eigenvectors = np.linalg.eigh(op)
    # Probability of each eigenvalue
    probs = np.abs(eigenvectors.conj().T @ state)**2

    # Sample from the distribution
    outcomes = np.random.choice(eigenvalues, size=n_shots, p=probs)
    estimated = np.mean(outcomes)
    std_error = np.std(outcomes) / np.sqrt(n_shots)

    return estimated, std_error, exact

def vqe_with_shot_noise(theta, H_terms, n_shots=1000):
    """Run VQE energy estimation with simulated shot noise.

    This demonstrates the realistic scenario where each Pauli term
    must be estimated from a finite number of measurements.
    """
    state = ansatz_state(theta)

    # H₂ Hamiltonian terms: (coefficient, [pauli_ops])
    total_energy = 0
    total_variance = 0

    for coeff, ops in H_terms:
        if all(np.allclose(op, I) for op in ops):
            # Identity term: exact, no measurement needed
            total_energy += coeff
            continue

        est, err, exact = measure_pauli_expectation(state, ops, n_shots)
        total_energy += coeff * est
        total_variance += (coeff * err)**2

    return total_energy, np.sqrt(total_variance)

# Demonstrate shot noise effect
print("=" * 55)
print("VQE with Finite Measurement Shots")
print("=" * 55)

H, g = build_h2_hamiltonian(0.735)
H_terms = [
    (g[0], [I, I]),
    (g[1], [Z, I]),
    (g[2], [I, Z]),
    (g[3], [Z, Z]),
    (g[4], [X, X]),
    (g[5], [Y, Y]),
]

theta_opt = 0.2267  # Approximately optimal

exact_energy = vqe_energy([theta_opt], H)
print(f"Exact energy at θ={theta_opt:.4f}: {exact_energy:.6f} Ha\n")

for n_shots in [10, 100, 1000, 10000, 100000]:
    energies = [vqe_with_shot_noise(theta_opt, H_terms, n_shots)[0]
                for _ in range(20)]
    mean_e = np.mean(energies)
    std_e = np.std(energies)
    print(f"  {n_shots:>7} shots: E = {mean_e:.6f} ± {std_e:.6f} Ha "
          f"(error: {abs(mean_e - exact_energy):.6f})")
```

---

## 10. Exercises

### Exercise 1: VQE with Different Ansatze

Implement two additional ansatze for the 2-qubit H₂ problem:
(a) **Two-parameter ansatz**: $R_y(\theta_1)$ on qubit 0, CNOT, $R_y(\theta_2)$ on qubit 1.
(b) **Three-parameter ansatz**: $R_y(\theta_1) R_z(\theta_2)$ on qubit 0, CNOT, $R_y(\theta_3)$ on qubit 1.
Compare the minimum achievable energy for each ansatz. Does increasing parameters always help?

### Exercise 2: Parameter Landscape Analysis

For the 2-qubit H₂ VQE at $R = 0.735$ \AA:
(a) Plot $E(\theta)$ for $\theta \in [-\pi, \pi]$ using 200 points.
(b) How many local minima are there?
(c) Compute $dE/d\theta$ using the parameter shift rule. Verify it matches finite differences.
(d) Run the optimizer from 20 random initial points. How often does it find the global minimum?

### Exercise 3: Shot Noise and Convergence

Investigate how measurement noise affects VQE convergence:
(a) Run VQE optimization with 100, 1000, and 10000 shots per energy evaluation.
(b) Plot the convergence curve (energy vs iteration) for each case.
(c) How many total shots are needed to achieve chemical accuracy ($1.6 \times 10^{-3}$ Ha)?
(d) Compare COBYLA (gradient-free) vs SPSA (stochastic gradient) under shot noise.

### Exercise 4: General 2-Qubit Hamiltonian

Consider the general 2-qubit Hamiltonian $H = J_x X_0 X_1 + J_y Y_0 Y_1 + J_z Z_0 Z_1 + h (Z_0 + Z_1)$ (the Heisenberg model in a field).
(a) Write a VQE solver for this Hamiltonian with a 2-layer hardware-efficient ansatz.
(b) Compute the phase diagram: ground state energy as a function of $J_x = J_y = J_z = J$ and $h/J$.
(c) Compare VQE results with exact diagonalization.

### Exercise 5: Barren Plateau Investigation

(a) Implement a random hardware-efficient ansatz with $n$ qubits and $L$ layers.
(b) For $n = 2, 4, 6, 8$ qubits, compute the variance of $\partial E / \partial \theta_1$ over 1000 random parameter initializations.
(c) Plot the variance vs $n$ on a log scale. Does it decrease exponentially?
(d) Repeat with a problem-specific initialization (e.g., near Hartree-Fock). Is the barren plateau mitigated?

---

[← Previous: Quantum Teleportation and Communication](12_Quantum_Teleportation.md) | [Next: QAOA and Combinatorial Optimization →](14_QAOA.md)
