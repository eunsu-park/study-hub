# Lesson 21: Quantum Chemistry

[← Previous: Noise and Quantum Channels](20_Noise_and_Quantum_Channels.md) | [Next: Topological Quantum Computing →](22_Topological_Quantum_Computing.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Express molecular Hamiltonians in second quantization form using creation and annihilation operators
2. Explain basis sets and their role in discretizing the electronic structure problem
3. Apply the Jordan-Wigner and Bravyi-Kitaev transforms to map fermionic operators to qubit operators
4. Design VQE ansatze for molecular simulation (UCCSD, hardware-efficient)
5. Analyze where quantum advantage is expected for chemistry problems
6. Estimate resource requirements for classically intractable chemical simulations
7. Implement molecular Hamiltonian construction and qubit mapping in Python

---

Quantum chemistry is widely regarded as the most promising near-term application of quantum computers. The reason is fundamental: molecules are quantum systems, and simulating quantum systems is exponentially hard for classical computers. The electronic structure problem — finding the ground state energy of a molecule given its nuclear configuration — is central to chemistry, materials science, and drug design. Yet exact classical solutions scale exponentially with the number of electrons.

A quantum computer can represent the electronic wavefunction directly in its quantum register. The key challenge is mapping the fermionic Hamiltonian (electrons are fermions, obeying the Pauli exclusion principle) to a qubit Hamiltonian that a quantum computer can process. This lesson covers the complete pipeline: from the molecular Hamiltonian through second quantization, basis set discretization, fermion-to-qubit mapping, and quantum algorithms for finding ground state energies.

> **Analogy:** Classical computers trying to simulate molecules are like someone trying to describe a symphony by listing every possible combination of notes that could be playing at each moment. A quantum computer represents the symphony directly — each qubit is like an instrument, and the quantum state encodes all the complex correlations between them.

## Table of Contents

1. [The Electronic Structure Problem](#1-the-electronic-structure-problem)
2. [Second Quantization](#2-second-quantization)
3. [Basis Sets](#3-basis-sets)
4. [Fermion-to-Qubit Mappings](#4-fermion-to-qubit-mappings)
5. [Molecular Hamiltonians as Qubit Operators](#5-molecular-hamiltonians-as-qubit-operators)
6. [Quantum Algorithms for Chemistry](#6-quantum-algorithms-for-chemistry)
7. [Quantum Advantage in Chemistry](#7-quantum-advantage-in-chemistry)
8. [Resource Estimates](#8-resource-estimates)
9. [Python Implementation](#9-python-implementation)
10. [Exercises](#10-exercises)

---

## 1. The Electronic Structure Problem

### 1.1 The Molecular Hamiltonian

For a molecule with $N_e$ electrons and $N_n$ nuclei, the full Hamiltonian (in atomic units, $\hbar = m_e = e = 1$) is:

$$H = -\sum_{i=1}^{N_e} \frac{\nabla_i^2}{2} - \sum_{A=1}^{N_n} \frac{\nabla_A^2}{2M_A} + \sum_{i<j}^{N_e} \frac{1}{|r_i - r_j|} - \sum_{i=1}^{N_e} \sum_{A=1}^{N_n} \frac{Z_A}{|r_i - R_A|} + \sum_{A<B}^{N_n} \frac{Z_A Z_B}{|R_A - R_B|}$$

### 1.2 Born-Oppenheimer Approximation

Since nuclei are much heavier than electrons ($M_A \gg m_e$), we fix nuclear positions and solve for the electronic energy. The **electronic Hamiltonian** is:

$$H_{\text{elec}} = -\sum_{i=1}^{N_e} \frac{\nabla_i^2}{2} + \sum_{i<j}^{N_e} \frac{1}{|r_i - r_j|} - \sum_{i=1}^{N_e} \sum_{A=1}^{N_n} \frac{Z_A}{|r_i - R_A|}$$

The nuclear repulsion $\sum_{A<B} Z_A Z_B / |R_A - R_B|$ is a constant for fixed nuclear geometry.

### 1.3 Why It Is Hard

The electronic wavefunction $\Psi(r_1, r_2, \ldots, r_{N_e})$ is a function of $3N_e$ continuous variables. Discretizing each coordinate into $M$ grid points requires storing $M^{3N_e}$ amplitudes — exponential in $N_e$.

Moreover, electrons are **fermions**: $\Psi$ must be antisymmetric under exchange of any two electrons. This constraint (the Pauli exclusion principle) creates complex correlations (electron correlation) that are the central challenge of computational chemistry.

### 1.4 Classical Methods Hierarchy

| Method | Scaling | Accuracy | Captures correlation? |
|--------|---------|----------|----------------------|
| Hartree-Fock (HF) | $O(N^4)$ | $\sim 10$ kcal/mol | Mean-field only |
| DFT | $O(N^3)$ | $\sim 3$ kcal/mol | Approximate |
| MP2 | $O(N^5)$ | $\sim 2$ kcal/mol | Perturbative |
| CCSD | $O(N^6)$ | $\sim 1$ kcal/mol | Partial |
| CCSD(T) | $O(N^7)$ | Chemical accuracy | Partial |
| Full CI | $O(\binom{N}{N_e}^2)$ | Exact (in basis) | Full |

Chemical accuracy: $1$ kcal/mol $\approx 1.6$ mHa $\approx 43$ meV. Full CI (full configuration interaction) is exact within a given basis set but scales exponentially.

---

## 2. Second Quantization

### 2.1 From First to Second Quantization

First quantization represents electrons as wavefunctions. Second quantization represents them using **creation** ($a_p^\dagger$) and **annihilation** ($a_p$) operators acting on **occupation number states**.

A basis of $M$ spin-orbitals $\{\phi_1, \phi_2, \ldots, \phi_M\}$ defines the Fock space. Each spin-orbital is either occupied (1) or unoccupied (0):

$$|n_1, n_2, \ldots, n_M\rangle, \quad n_p \in \{0, 1\}$$

### 2.2 Creation and Annihilation Operators

$$a_p^\dagger |n_1, \ldots, 0_p, \ldots, n_M\rangle = (-1)^{\sum_{q<p} n_q} |n_1, \ldots, 1_p, \ldots, n_M\rangle$$

$$a_p |n_1, \ldots, 1_p, \ldots, n_M\rangle = (-1)^{\sum_{q<p} n_q} |n_1, \ldots, 0_p, \ldots, n_M\rangle$$

The sign factor $(-1)^{\sum_{q<p} n_q}$ enforces fermionic antisymmetry.

### 2.3 Anticommutation Relations

$$\{a_p, a_q^\dagger\} = \delta_{pq}, \quad \{a_p, a_q\} = 0, \quad \{a_p^\dagger, a_q^\dagger\} = 0$$

These relations encode the Pauli exclusion principle: no two electrons can occupy the same spin-orbital.

### 2.4 Electronic Hamiltonian in Second Quantization

$$H = \sum_{p,q} h_{pq} a_p^\dagger a_q + \frac{1}{2}\sum_{p,q,r,s} h_{pqrs} a_p^\dagger a_q^\dagger a_s a_r + E_{\text{nuc}}$$

where:
- $h_{pq} = \int \phi_p^*(r)\left(-\frac{\nabla^2}{2} - \sum_A \frac{Z_A}{|r - R_A|}\right)\phi_q(r)\,dr$ (one-electron integrals)
- $h_{pqrs} = \int\int \frac{\phi_p^*(r_1)\phi_q^*(r_2)\phi_r(r_1)\phi_s(r_2)}{|r_1 - r_2|}\,dr_1 dr_2$ (two-electron integrals)
- $E_{\text{nuc}} = \sum_{A<B} Z_A Z_B / |R_A - R_B|$ (nuclear repulsion constant)

---

## 3. Basis Sets

### 3.1 What Is a Basis Set?

A basis set is a finite set of mathematical functions used to expand the molecular orbitals:

$$\psi_i(r) = \sum_{\mu=1}^{K} C_{\mu i} \chi_\mu(r)$$

where $\{\chi_\mu\}$ are the basis functions and $C_{\mu i}$ are expansion coefficients determined by solving the Hartree-Fock equations.

### 3.2 Common Basis Sets

| Basis set | Functions per atom (C) | Spin-orbitals | Accuracy |
|-----------|----------------------|---------------|----------|
| STO-3G (minimal) | 5 | 10 per C | Qualitative |
| 6-31G | 9 | 18 per C | Semi-quantitative |
| 6-31G* | 15 | 30 per C | Moderate |
| cc-pVDZ | 14 | 28 per C | Quantitative |
| cc-pVTZ | 30 | 60 per C | High accuracy |
| cc-pVQZ | 55 | 110 per C | Near CBS limit |

**CBS limit**: Complete basis set limit — the answer as the basis becomes infinite.

### 3.3 Basis Set for Quantum Computing

The number of qubits needed equals the number of spin-orbitals $M$:

| Molecule | Basis set | Spin-orbitals (= qubits) | Electrons |
|----------|-----------|-------------------------|-----------|
| H$_2$ | STO-3G | 4 | 2 |
| LiH | STO-3G | 12 | 4 |
| H$_2$O | STO-3G | 14 | 10 |
| N$_2$ | STO-3G | 20 | 14 |
| FeMoCo | cc-pVDZ | ~200 | ~113 |

### 3.4 Active Space Approximation

For large molecules, we cannot include all orbitals on a quantum computer. The **active space** approach:

1. Run classical HF to get molecular orbitals
2. Identify **active orbitals** (those near the Fermi level with strong correlation)
3. Freeze core electrons and virtual orbitals
4. Map only the active space to qubits

Example: FeMoCo (the active site of nitrogenase) has hundreds of electrons, but only ~50-60 active orbitals need quantum treatment.

---

## 4. Fermion-to-Qubit Mappings

### 4.1 The Mapping Problem

Fermions obey anticommutation relations; qubits obey standard commutation relations (Pauli algebra). We need a mapping that encodes fermionic statistics into qubit operators.

### 4.2 Jordan-Wigner Transform

The Jordan-Wigner (JW) transform maps each fermionic mode to one qubit:

$$a_p^\dagger \to \frac{1}{2}(X_p - iY_p) \otimes Z_{p-1} \otimes Z_{p-2} \otimes \cdots \otimes Z_0$$

$$a_p \to \frac{1}{2}(X_p + iY_p) \otimes Z_{p-1} \otimes Z_{p-2} \otimes \cdots \otimes Z_0$$

The **Z string** ($Z_{p-1} \otimes \cdots \otimes Z_0$) tracks the fermionic parity and enforces antisymmetry.

**Properties**:
- Direct mapping: qubit $p$ stores the occupation of spin-orbital $p$
- Number operator: $a_p^\dagger a_p = (I - Z_p)/2$
- Non-local: Z strings can span all qubits, making some operations expensive

### 4.3 Parity Transform

The parity transform stores the cumulative parity instead of occupation:

$$\text{qubit } p \text{ stores } \bigoplus_{q \leq p} n_q$$

This makes number operators local ($a_p^\dagger a_p$ involves only qubits $p$ and $p+1$) but creation/annihilation operators still have Z strings.

### 4.4 Bravyi-Kitaev Transform

The Bravyi-Kitaev (BK) transform balances locality using a binary tree encoding:

- Even-indexed qubits store occupation numbers
- Odd-indexed qubits store partial parities
- Z strings have length $O(\log M)$ instead of $O(M)$

**Comparison**:

| Mapping | Z-string length | Number operator weight | Total Pauli weight |
|---------|----------------|----------------------|-------------------|
| Jordan-Wigner | $O(M)$ | $O(1)$ | $O(M^4)$ |
| Parity | $O(M)$ | $O(1)$ | $O(M^4)$ |
| Bravyi-Kitaev | $O(\log M)$ | $O(\log M)$ | $O(M^4 \log M)$ |

For small molecules (few qubits), JW is simplest. For larger systems, BK reduces circuit depth.

---

## 5. Molecular Hamiltonians as Qubit Operators

### 5.1 Pauli String Representation

After fermion-to-qubit mapping, the Hamiltonian becomes a weighted sum of Pauli strings:

$$H = \sum_k \alpha_k P_k$$

where each $P_k$ is a tensor product of Pauli operators ($I, X, Y, Z$) and $\alpha_k$ are real coefficients.

### 5.2 Example: H$_2$ in STO-3G

For H$_2$ with 4 spin-orbitals (2 spatial orbitals $\times$ 2 spins), the Jordan-Wigner Hamiltonian has about 15 Pauli terms:

$$H = g_0 I + g_1 Z_0 + g_2 Z_1 + g_3 Z_2 + g_4 Z_3 + g_5 Z_0 Z_1 + g_6 Z_0 Z_2 + \cdots$$

$$+ g_{12}(X_0 Y_1 Y_2 X_3 - X_0 X_1 Y_2 Y_3 + Y_0 X_1 X_2 Y_3 - Y_0 Y_1 X_2 X_3)$$

The last group of terms represents the **double excitation** that is the core of electron correlation.

### 5.3 Number of Pauli Terms

The number of Pauli strings grows as $O(M^4)$ for a molecular Hamiltonian with $M$ spin-orbitals:

| Molecule | Basis | Qubits | Pauli terms (JW) |
|----------|-------|--------|-------------------|
| H$_2$ | STO-3G | 4 | 15 |
| LiH | STO-3G | 12 | 631 |
| H$_2$O | STO-3G | 14 | 1086 |
| N$_2$ | STO-3G | 20 | 2951 |

### 5.4 Symmetry Reduction

Symmetries can reduce the number of qubits:

- **Number conservation**: Fix electron number → eliminate 1 qubit
- **Spin conservation**: Fix total spin → eliminate 1 qubit
- **Point group symmetry**: Exploit molecular symmetry → eliminate additional qubits
- **Tapering**: Remove qubits that are constants of motion

For H$_2$ in STO-3G: 4 qubits → 2 qubits after tapering.

---

## 6. Quantum Algorithms for Chemistry

### 6.1 Variational Quantum Eigensolver (VQE)

VQE (covered in Lesson 13) is the primary NISQ algorithm for chemistry:

1. Prepare a parameterized ansatz state $|\psi(\theta)\rangle$
2. Measure $\langle\psi(\theta)|H|\psi(\theta)\rangle$ by decomposing $H$ into Pauli terms
3. Optimize $\theta$ classically to minimize energy

**Ansatze for chemistry**:

| Ansatz | Description | Circuit depth | Chemical accuracy? |
|--------|-------------|---------------|-------------------|
| UCCSD | Unitary coupled cluster singles+doubles | $O(M^4)$ | Usually yes |
| k-UpCCGSD | $k$ layers of pair doubles | $O(kM^2)$ | Often yes |
| Hardware-efficient | Alternating Ry + CNOT layers | $O(LM)$ | Sometimes |
| ADAPT-VQE | Iteratively grown | Problem-dependent | Yes |

### 6.2 Quantum Phase Estimation (QPE)

For fault-tolerant quantum computers, QPE (Lesson 18) gives exponentially precise energies:

1. Prepare an initial state $|\psi_0\rangle$ with overlap $|\langle E_0|\psi_0\rangle|^2 > 0$
2. Implement controlled-$e^{-iHt}$ using Trotter or LCU
3. QPE extracts the eigenvalue to $m$ bits of precision

**Advantage over VQE**: Systematic precision, no optimization loop
**Disadvantage**: Requires deep circuits (fault tolerance)

### 6.3 Quantum Subspace Expansion (QSE)

A hybrid approach:
1. Use VQE to find approximate ground state $|\psi_0\rangle$
2. Construct a subspace $\{|\psi_0\rangle, H|\psi_0\rangle, H^2|\psi_0\rangle, \ldots\}$
3. Diagonalize $H$ in this subspace classically

This can improve energy estimates and find excited states.

---

## 7. Quantum Advantage in Chemistry

### 7.1 Where Classical Methods Fail

**Strongly correlated systems**: Multiple electronic configurations contribute significantly to the ground state. Examples:
- Transition metal complexes (catalytic centers)
- Bond-breaking processes (stretched bonds)
- Excited states (multiconfigurational character)
- Superconducting materials (high-$T_c$ cuprates)

### 7.2 Target Molecules

| Molecule | Why quantum? | Estimated qubits | Impact |
|----------|-------------|------------------|--------|
| FeMoCo | Nitrogen fixation catalyst | 50-100 (active space) | Agriculture, energy |
| Cytochrome P450 | Drug metabolism | 40-80 | Pharmaceutical |
| Li-S battery | Polysulfide intermediates | 30-60 | Energy storage |
| High-$T_c$ cuprates | Superconductivity mechanism | 50-100 | Materials |

### 7.3 Timeline for Quantum Advantage

| Milestone | Qubits (logical) | Error rate | Estimated year |
|-----------|------------------|------------|---------------|
| Match CCSD(T) for small molecules | 20-50 | $10^{-3}$ (NISQ) | Current |
| Exceed CCSD(T) for medium molecules | 50-100 | $10^{-6}$ (FT) | ~2028-2030 |
| FeMoCo ground state | 100-200 | $10^{-8}$ (FT) | ~2030-2035 |
| Full protein-scale simulation | 1000+ | $10^{-10}$ (FT) | ~2035+ |

### 7.4 Caveats

- The classical frontier is also advancing (DMRG, FCIQMC, neural network wavefunctions)
- Active space selection is still an art, and a bad active space negates quantum advantage
- Practical quantum advantage requires not just more qubits, but also low error rates

---

## 8. Resource Estimates

### 8.1 Gate Counts

For a single Trotter step of a molecular Hamiltonian with $M$ spin-orbitals:
- JW mapping: $O(M^5)$ gates per Trotter step
- BK mapping: $O(M^4 \log M)$ gates per Trotter step

For QPE with precision $\epsilon$: total gates $\sim O(M^4 t / \epsilon)$ where $t$ depends on spectral range.

### 8.2 Measurement Costs

VQE requires measuring $O(M^4)$ Pauli terms. With measurement grouping:
- Qubit-wise commuting groups: $O(M^3)$ groups
- General commuting groups: $O(M^2)$ groups
- Classical shadows: $O(M^2 / \epsilon^2)$ total shots (independent of grouping)

### 8.3 Comparison: VQE vs. QPE Resources

| Resource | VQE | QPE |
|----------|-----|-----|
| Qubits | $M$ | $M + m$ ancilla |
| Circuit depth | $O(M^2)$ per iteration | $O(M^5 / \epsilon)$ |
| Repetitions | $O(M^4 / \epsilon^2)$ measurements | $O(1)$ (coherent) |
| Classical cost | Optimization loop | Minimal |
| Error tolerance | NISQ compatible | Requires FT |

---

## 9. Python Implementation

### 9.1 Molecular Integrals and Hamiltonian Construction

```python
import numpy as np

# Pauli matrices
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def build_h2_hamiltonian(bond_length=0.74):
    """Build the H2 Hamiltonian in the STO-3G basis.

    The H2 molecule is the simplest non-trivial chemical system:
    2 electrons in 4 spin-orbitals (2 spatial x 2 spin).
    Exact diagonalization is feasible, making it the standard
    test case for quantum chemistry algorithms.

    The integrals are pre-computed for common bond lengths.
    In practice, these would come from a classical quantum chemistry
    package like PySCF.

    Args:
        bond_length: H-H distance in Angstroms

    Returns:
        h1: One-electron integral matrix (4x4)
        h2: Two-electron integral tensor (4x4x4x4)
        E_nuc: Nuclear repulsion energy
        n_qubits: Number of qubits (= number of spin-orbitals)
    """
    # Pre-computed integrals for H2 at various bond lengths (STO-3G)
    # These approximate the actual molecular integrals
    R = bond_length

    # Nuclear repulsion
    E_nuc = 1.0 / R  # in atomic units (1/R_AB for H-H)

    # One-electron integrals (spatial orbitals, 2x2)
    # h_pq = <p|T + V|q> (kinetic + nuclear attraction)
    # Approximate parameterization based on STO-3G H2
    h_core = np.array([
        [-1.2563 + 0.2 * (R - 0.74), -0.4719 + 0.1 * (R - 0.74)],
        [-0.4719 + 0.1 * (R - 0.74), -0.4719 + 0.3 * (R - 0.74)]
    ])

    # Two-electron integrals (spatial, chemist notation: (pq|rs))
    # Approximate for H2 STO-3G
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 0.6746
    eri[0, 0, 1, 1] = 0.6636
    eri[1, 1, 0, 0] = 0.6636
    eri[1, 1, 1, 1] = 0.6974
    eri[0, 1, 0, 1] = 0.1813
    eri[0, 1, 1, 0] = 0.1813
    eri[1, 0, 0, 1] = 0.1813
    eri[1, 0, 1, 0] = 0.1813

    # Expand to spin-orbital basis (4 spin-orbitals)
    # Spin-orbitals: 0↑, 0↓, 1↑, 1↓
    n_spatial = 2
    n_spin = 4

    h1 = np.zeros((n_spin, n_spin), dtype=complex)
    h2 = np.zeros((n_spin, n_spin, n_spin, n_spin), dtype=complex)

    for p in range(n_spatial):
        for q in range(n_spatial):
            # Same spin
            h1[2 * p, 2 * q] = h_core[p, q]       # alpha-alpha
            h1[2 * p + 1, 2 * q + 1] = h_core[p, q]  # beta-beta

    for p in range(n_spatial):
        for q in range(n_spatial):
            for r in range(n_spatial):
                for s in range(n_spatial):
                    # Physicists' notation: <pq||rs> = <pq|rs> - <pq|sr>
                    for sp in range(2):  # spin of p
                        for sq in range(2):  # spin of q
                            pi, qi = 2 * p + sp, 2 * q + sq
                            ri, si = 2 * r + sp, 2 * s + sq
                            h2[pi, qi, ri, si] += eri[p, r, q, s]
                            # Exchange (same spin only)
                            if sp == sq:
                                ri2, si2 = 2 * r + sq, 2 * s + sp
                                # Already included via antisymmetry

    return h1, h2, E_nuc, n_spin


# Demonstrate H2 Hamiltonian
print("=" * 65)
print("H2 Molecular Hamiltonian (STO-3G)")
print("=" * 65)

for R in [0.5, 0.74, 1.0, 1.5, 2.0]:
    h1, h2, E_nuc, n_q = build_h2_hamiltonian(R)
    print(f"\nBond length R = {R:.2f} A:")
    print(f"  Nuclear repulsion: {E_nuc:.4f} Ha")
    print(f"  Spin-orbitals (qubits): {n_q}")
    print(f"  One-electron integrals (non-zero): {np.count_nonzero(np.abs(h1) > 1e-10)}")
    print(f"  Two-electron integrals (non-zero): {np.count_nonzero(np.abs(h2) > 1e-10)}")
```

### 9.2 Jordan-Wigner Transform

```python
import numpy as np
from functools import reduce

def kron_list(ops):
    """Tensor product of a list of operators."""
    return reduce(np.kron, ops)


def jordan_wigner_creation(p, n_qubits):
    """Compute the Jordan-Wigner representation of a^dagger_p.

    JW transform: a^dagger_p = (X_p - iY_p)/2 * Z_{p-1} * ... * Z_0

    The Z-string ensures fermionic antisymmetry: when creating a
    particle at orbital p, the sign depends on the parity of all
    occupied orbitals below p.

    Args:
        p: Orbital index (0-indexed)
        n_qubits: Total number of qubits

    Returns:
        Matrix representation (2^n x 2^n)
    """
    # (X - iY) / 2 = |0><1| = [[0, 1], [0, 0]]
    raising = (X - 1j * Y) / 2

    ops = []
    for q in range(n_qubits):
        if q < p:
            ops.append(Z)    # Z-string for antisymmetry
        elif q == p:
            ops.append(raising)
        else:
            ops.append(I2)

    return kron_list(ops)


def jordan_wigner_annihilation(p, n_qubits):
    """JW representation of a_p (annihilation operator)."""
    return jordan_wigner_creation(p, n_qubits).conj().T


def jordan_wigner_number(p, n_qubits):
    """JW representation of n_p = a^dagger_p a_p = (I - Z_p) / 2."""
    ops = [I2] * n_qubits
    ops[p] = (I2 - Z) / 2
    return kron_list(ops)


def build_qubit_hamiltonian_jw(h1, h2, E_nuc, n_qubits):
    """Build the qubit Hamiltonian using Jordan-Wigner transform.

    H = sum_{pq} h_{pq} a^dag_p a_q + (1/2) sum_{pqrs} h_{pqrs} a^dag_p a^dag_q a_s a_r + E_nuc

    Each fermionic term is converted to a sum of Pauli strings.

    Args:
        h1: One-electron integrals
        h2: Two-electron integrals
        E_nuc: Nuclear repulsion energy
        n_qubits: Number of qubits

    Returns:
        H_qubit: Full Hamiltonian matrix (2^n x 2^n)
    """
    N = 2 ** n_qubits
    H = E_nuc * np.eye(N, dtype=complex)

    # One-electron terms
    for p in range(n_qubits):
        for q in range(n_qubits):
            if abs(h1[p, q]) > 1e-12:
                a_p_dag = jordan_wigner_creation(p, n_qubits)
                a_q = jordan_wigner_annihilation(q, n_qubits)
                H += h1[p, q] * (a_p_dag @ a_q)

    # Two-electron terms
    for p in range(n_qubits):
        for q in range(n_qubits):
            for r in range(n_qubits):
                for s in range(n_qubits):
                    if abs(h2[p, q, r, s]) > 1e-12:
                        a_p_dag = jordan_wigner_creation(p, n_qubits)
                        a_q_dag = jordan_wigner_creation(q, n_qubits)
                        a_s = jordan_wigner_annihilation(s, n_qubits)
                        a_r = jordan_wigner_annihilation(r, n_qubits)
                        H += 0.5 * h2[p, q, r, s] * (a_p_dag @ a_q_dag @ a_s @ a_r)

    # Ensure Hermiticity
    H = (H + H.conj().T) / 2

    return H


# Demonstrate Jordan-Wigner mapping
print("=" * 65)
print("Jordan-Wigner Transform for H2")
print("=" * 65)

h1, h2, E_nuc, n_qubits = build_h2_hamiltonian(0.74)
H_qubit = build_qubit_hamiltonian_jw(h1, h2, E_nuc, n_qubits)

# Exact diagonalization
eigenvalues = np.sort(np.real(np.linalg.eigvalsh(H_qubit)))

# The physical states have the correct electron number (N_e = 2)
# Check which eigenstates have N_e = 2
N_op = sum(jordan_wigner_number(p, n_qubits) for p in range(n_qubits))
eigvals_full, eigvecs_full = np.linalg.eigh(H_qubit)

print(f"\nH2 at R = 0.74 A (equilibrium)")
print(f"Number of qubits: {n_qubits}")
print(f"Hamiltonian dimension: {2**n_qubits} x {2**n_qubits}")
print(f"\nAll eigenvalues:")
for i, E in enumerate(eigenvalues):
    N_exp = np.real(eigvecs_full[:, i].conj() @ N_op @ eigvecs_full[:, i])
    print(f"  E_{i} = {E:.6f} Ha, <N> = {N_exp:.1f}")

# Filter to N=2 sector
print(f"\nN=2 sector eigenvalues (physical states):")
for i, E in enumerate(eigvals_full):
    N_exp = np.real(eigvecs_full[:, i].conj() @ N_op @ eigvecs_full[:, i])
    if abs(N_exp - 2) < 0.1:
        print(f"  E = {E:.6f} Ha")
```

### 9.3 Potential Energy Curve

```python
import numpy as np

def compute_h2_energy_curve(bond_lengths):
    """Compute the H2 potential energy curve.

    At each bond length, build the Hamiltonian, diagonalize,
    and extract the ground state energy (N_e=2 sector).

    This demonstrates the dissociation problem: at large R,
    Hartree-Fock breaks down (restricted HF gives wrong dissociation limit)
    while exact diagonalization (full CI) remains correct.
    """
    energies_exact = []
    energies_hf = []

    for R in bond_lengths:
        h1, h2, E_nuc, n_q = build_h2_hamiltonian(R)
        H = build_qubit_hamiltonian_jw(h1, h2, E_nuc, n_q)

        # Exact diagonalization
        N_op = sum(jordan_wigner_number(p, n_q) for p in range(n_q))
        eigvals, eigvecs = np.linalg.eigh(H)

        # Find lowest N=2 eigenvalue
        e_min = float('inf')
        for i in range(len(eigvals)):
            N_exp = np.real(eigvecs[:, i].conj() @ N_op @ eigvecs[:, i])
            if abs(N_exp - 2) < 0.1 and eigvals[i] < e_min:
                e_min = eigvals[i]
        energies_exact.append(e_min)

        # Hartree-Fock energy (just the HF determinant |1100>)
        hf_state = np.zeros(2 ** n_q, dtype=complex)
        hf_state[0b1100] = 1.0  # orbitals 0 and 1 occupied (MSB order)
        # Actually, |1100> = first two spin-orbitals occupied
        hf_state = np.zeros(2 ** n_q, dtype=complex)
        hf_state[3] = 1.0  # |0011> in little-endian = orbitals 0,1 occupied
        # Fix: map to correct occupation
        # In JW, qubit p stores occupation of orbital p
        # |n_0 n_1 n_2 n_3> with n_0=1, n_1=1, n_2=0, n_3=0
        # Binary: 1100 = 12 (big-endian) or 0011 = 3 (little-endian)
        hf_state = np.zeros(2 ** n_q, dtype=complex)
        hf_state[12] = 1.0  # |1100> in big-endian notation

        e_hf = np.real(hf_state.conj() @ H @ hf_state)
        energies_hf.append(e_hf)

    return energies_exact, energies_hf


print("=" * 65)
print("H2 Potential Energy Curve")
print("=" * 65)

bond_lengths = np.arange(0.3, 3.01, 0.1)
energies_exact, energies_hf = compute_h2_energy_curve(bond_lengths)

print(f"\n{'R (A)':>8} {'E_exact (Ha)':>14} {'E_HF (Ha)':>14} {'Correlation (mHa)':>18}")
print("-" * 58)
for R, e_ex, e_hf in zip(bond_lengths, energies_exact, energies_hf):
    corr = (e_ex - e_hf) * 1000
    print(f"{R:8.2f} {e_ex:14.6f} {e_hf:14.6f} {corr:18.2f}")

# Find equilibrium bond length
min_idx = np.argmin(energies_exact)
print(f"\nEquilibrium bond length: {bond_lengths[min_idx]:.2f} A")
print(f"Ground state energy: {energies_exact[min_idx]:.6f} Ha")
```

### 9.4 VQE for H2

```python
import numpy as np
from scipy.optimize import minimize

def vqe_h2(bond_length=0.74, n_layers=2, method='COBYLA'):
    """Run VQE for H2 using a simple parameterized circuit.

    The ansatz is inspired by UCCSD: it includes single and double
    excitation operators that generate the relevant electron correlations.

    For H2 with 4 qubits, the key correlation is the double excitation
    |1100> <-> |0011>, which requires a specific 2-qubit gate pattern.
    """
    h1, h2, E_nuc, n_q = build_h2_hamiltonian(bond_length)
    H = build_qubit_hamiltonian_jw(h1, h2, E_nuc, n_q)
    N = 2 ** n_q

    def ansatz_state(params):
        """Prepare the VQE ansatz state.

        Start from HF state |1100> and apply parameterized excitations.
        """
        # Start from HF: |1100> (orbitals 0,1 occupied)
        state = np.zeros(N, dtype=complex)
        state[12] = 1.0  # |1100> = 12 in decimal (big-endian)

        # Layer of single-qubit rotations
        for layer in range(n_layers):
            base = layer * (n_q + 1)
            for q in range(n_q):
                angle = params[base + q]
                # Ry rotation on qubit q
                cos_half = np.cos(angle / 2)
                sin_half = np.sin(angle / 2)

                new_state = np.zeros_like(state)
                for s in range(N):
                    bit = (s >> (n_q - 1 - q)) & 1
                    s_flipped = s ^ (1 << (n_q - 1 - q))

                    if bit == 0:
                        new_state[s] += cos_half * state[s]
                        new_state[s_flipped] += sin_half * state[s]
                    else:
                        new_state[s] += cos_half * state[s]
                        new_state[s_flipped] -= sin_half * state[s]

                state = new_state

            # CNOT entangling layer
            for q in range(n_q - 1):
                new_state = np.zeros_like(state)
                for s in range(N):
                    ctrl = (s >> (n_q - 1 - q)) & 1
                    if ctrl:
                        s_new = s ^ (1 << (n_q - 1 - (q + 1)))
                        new_state[s_new] += state[s]
                    else:
                        new_state[s] += state[s]
                state = new_state

            # Double excitation parameter (key for H2 correlation)
            theta_de = params[base + n_q]
            # Apply exp(i*theta*(|1100><0011| - |0011><1100|))
            cos_t = np.cos(theta_de)
            sin_t = np.sin(theta_de)
            idx_1100 = 12  # |1100>
            idx_0011 = 3   # |0011>
            a, b = state[idx_1100], state[idx_0011]
            state[idx_1100] = cos_t * a + sin_t * b
            state[idx_0011] = -sin_t * a + cos_t * b

        return state

    def cost(params):
        state = ansatz_state(params)
        return np.real(state.conj() @ H @ state)

    n_params = n_layers * (n_q + 1)
    best_energy = float('inf')
    best_params = None

    # Multiple random starts
    for trial in range(5):
        params0 = np.random.uniform(-0.5, 0.5, n_params)
        params0[0:n_q] = 0.0  # Start near HF

        result = minimize(cost, params0, method=method,
                         options={'maxiter': 500})
        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x

    return best_energy, best_params


# Run VQE for H2 at various bond lengths
print("=" * 65)
print("VQE for H2 Molecule")
print("=" * 65)

N_op_global = None
for R in [0.5, 0.74, 1.0, 1.5, 2.0]:
    h1, h2, E_nuc, n_q = build_h2_hamiltonian(R)
    H = build_qubit_hamiltonian_jw(h1, h2, E_nuc, n_q)

    # Exact energy for comparison
    N_op = sum(jordan_wigner_number(p, n_q) for p in range(n_q))
    eigvals, eigvecs = np.linalg.eigh(H)
    e_exact = float('inf')
    for i in range(len(eigvals)):
        N_exp = np.real(eigvecs[:, i].conj() @ N_op @ eigvecs[:, i])
        if abs(N_exp - 2) < 0.1 and eigvals[i] < e_exact:
            e_exact = eigvals[i]

    e_vqe, _ = vqe_h2(R, n_layers=2)
    error_mha = abs(e_vqe - e_exact) * 1000

    print(f"\nR = {R:.2f} A:")
    print(f"  VQE energy:   {e_vqe:.6f} Ha")
    print(f"  Exact energy: {e_exact:.6f} Ha")
    print(f"  Error: {error_mha:.2f} mHa {'(chemical accuracy)' if error_mha < 1.6 else ''}")
```

### 9.5 Bravyi-Kitaev Transform

```python
import numpy as np

def bravyi_kitaev_transform(n_qubits):
    """Compute the Bravyi-Kitaev transformation matrix.

    The BK transform uses a binary tree to balance locality:
    - Even qubits store occupation numbers
    - Odd qubits store partial sums (parities)

    This reduces the Z-string length from O(n) to O(log n).

    Args:
        n_qubits: Number of qubits

    Returns:
        beta: BK transformation matrix (maps occupation to BK basis)
    """
    # Build the BK transformation matrix recursively
    beta = np.eye(n_qubits, dtype=int)

    # The BK matrix has a specific recursive structure
    for j in range(n_qubits):
        # Qubit j stores the parity of a specific set of orbitals
        # determined by the binary representation of j
        update_set = _bk_update_set(j, n_qubits)
        for k in update_set:
            if k != j:
                beta[j, k] = 1

    return beta % 2


def _bk_update_set(j, n):
    """Compute the update set for qubit j in BK transform."""
    update = {j}
    # Find qubits that qubit j contributes to
    bit = 0
    while (1 << bit) <= j:
        if j & (1 << bit):
            parent = j | (1 << (bit + 1))
            if parent < n:
                update.add(parent)
        bit += 1
    return update


def compare_jw_bk(n_qubits):
    """Compare Jordan-Wigner and Bravyi-Kitaev representations.

    Shows how the Z-string length differs between the two mappings.
    """
    print(f"\nComparison for {n_qubits} qubits:")
    print(f"  {'Orbital':>8} {'JW Z-string':>15} {'BK Z-string':>15}")
    print(f"  {'-' * 40}")

    for p in range(n_qubits):
        # JW: Z-string of length p
        jw_z_length = p

        # BK: Z-string of length O(log n)
        # The exact set depends on the binary tree structure
        bk_z_length = bin(p + 1).count('1')  # Approximate

        print(f"  {p:8d} {jw_z_length:15d} {bk_z_length:15d}")


print("=" * 65)
print("Jordan-Wigner vs Bravyi-Kitaev Transform")
print("=" * 65)

for n in [4, 8, 16, 32]:
    compare_jw_bk(n)
    beta = bravyi_kitaev_transform(min(n, 8))
    if n <= 8:
        print(f"  BK matrix ({n}x{n}):")
        for row in beta:
            print(f"    {row}")
```

---

## 10. Exercises

### Exercise 1: Molecular Integrals

For the H2 molecule:
(a) Verify that the one-electron integral matrix is Hermitian.
(b) Verify the symmetry of two-electron integrals: $h_{pqrs} = h_{qpsr} = h_{rspq}^*$.
(c) How many unique two-electron integrals exist for $M$ spin-orbitals? Compare with the naive $M^4$.

### Exercise 2: Jordan-Wigner Verification

(a) Verify that the JW-transformed creation and annihilation operators satisfy the fermionic anticommutation relations $\{a_p, a_q^\dagger\} = \delta_{pq}$.
(b) Compute $a_0^\dagger a_1$ explicitly for 4 qubits. How many Pauli strings does it decompose into?
(c) Compare the number of non-identity Pauli operators in the JW representation of $a_p^\dagger a_q$ for $|p - q| = 1$ vs. $|p - q| = n/2$.

### Exercise 3: H2 Dissociation

Compute the H2 potential energy curve from $R = 0.3$ to $R = 3.0$ A:
(a) Plot the exact (full CI) curve and the Hartree-Fock curve.
(b) At what bond length does the correlation energy become significant ($> 10$ mHa)?
(c) The exact dissociation limit should be $2 \times E(\text{H atom})$. Verify this.
(d) How does the VQE error change along the dissociation curve?

### Exercise 4: Qubit Reduction via Symmetry

For H2 in STO-3G:
(a) Identify the conserved quantum numbers: total electron number $N$ and total spin $S_z$.
(b) Construct the $N = 2, S_z = 0$ sector of the Hamiltonian. How many qubits are needed?
(c) Verify that the ground state energy in this reduced space matches the full space result.
(d) Implement qubit tapering to reduce from 4 to 2 qubits.

### Exercise 5: VQE Ansatz Comparison

Compare different VQE ansatze for H2 at $R = 1.5$ A (stretched bond):
(a) Hardware-efficient ansatz: Ry + CNOT layers. How many layers are needed for chemical accuracy?
(b) UCCSD-inspired ansatz: Include the double excitation operator. Does 1 layer suffice?
(c) ADAPT-VQE: Which operators are selected from the pool?
(d) Plot the energy convergence (iterations vs. energy error) for each ansatz.

---

[← Previous: Noise and Quantum Channels](20_Noise_and_Quantum_Channels.md) | [Next: Topological Quantum Computing →](22_Topological_Quantum_Computing.md)
