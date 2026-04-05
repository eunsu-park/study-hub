# Lesson 20: Noise and Quantum Channels

[← Previous: Quantum Walks](19_Quantum_Walks.md) | [Next: Quantum Chemistry →](21_Quantum_Chemistry.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe common noise models: depolarizing, amplitude damping, phase damping, and bit-flip channels
2. Derive and apply the Kraus operator representation of quantum channels
3. Explain the Choi-Jamiolkowski isomorphism and its use in channel characterization
4. Perform quantum process tomography to reconstruct an unknown quantum channel
5. Characterize noise through randomized benchmarking and gate set tomography
6. Analyze the effect of noise on quantum algorithms and error thresholds
7. Implement noise models, Kraus operators, and process tomography in Python

---

Every real quantum computer is noisy. Qubits interact with their environment, gates are imperfect, and measurements introduce errors. Understanding noise is not merely an academic exercise — it determines whether a quantum algorithm will produce meaningful results on real hardware. The mathematical framework of **quantum channels** provides a rigorous and complete description of noise processes.

A quantum channel is any physical process that transforms quantum states. It generalizes unitary evolution (ideal gates) to include irreversible processes like decoherence, energy relaxation, and measurement. The Kraus operator formalism gives a compact representation: any quantum channel can be written as $\rho \mapsto \sum_k E_k \rho E_k^\dagger$ where the Kraus operators $\{E_k\}$ satisfy $\sum_k E_k^\dagger E_k = I$.

> **Analogy:** A quantum channel is like a noisy telephone line for quantum information. A perfect line transmits the signal unchanged (unitary channel). A noisy line might randomly scramble the signal (depolarizing), gradually reduce its amplitude (amplitude damping), or randomly flip bits (bit-flip). Process tomography is like testing the line by sending known signals and measuring what comes out.

## Table of Contents

1. [Quantum Noise Models](#1-quantum-noise-models)
2. [Kraus Operator Representation](#2-kraus-operator-representation)
3. [Common Quantum Channels](#3-common-quantum-channels)
4. [Choi-Jamiolkowski Isomorphism](#4-choi-jamiolkowski-isomorphism)
5. [Quantum Process Tomography](#5-quantum-process-tomography)
6. [Noise Characterization Methods](#6-noise-characterization-methods)
7. [Noise in Quantum Algorithms](#7-noise-in-quantum-algorithms)
8. [Error Thresholds and Fault Tolerance](#8-error-thresholds-and-fault-tolerance)
9. [Python Implementation](#9-python-implementation)
10. [Exercises](#10-exercises)

---

## 1. Quantum Noise Models

### 1.1 Open Quantum Systems

A closed quantum system evolves unitarily: $\rho(t) = U(t)\rho(0)U(t)^\dagger$. But real qubits are **open systems** that interact with their environment:

$$|\Psi_{SE}\rangle = |\psi_S\rangle \otimes |0_E\rangle \xrightarrow{U_{SE}} |\Phi_{SE}\rangle$$

The system state is obtained by tracing out the environment:

$$\rho_S(t) = \text{Tr}_E[U_{SE}(|\psi_S\rangle\langle\psi_S| \otimes |0_E\rangle\langle 0_E|)U_{SE}^\dagger]$$

This trace operation is what introduces non-unitary (noisy) dynamics.

### 1.2 Types of Noise

| Noise type | Physical origin | Effect on qubit | Timescale |
|------------|----------------|-----------------|-----------|
| **Relaxation** ($T_1$) | Energy exchange with environment | $|1\rangle \to |0\rangle$ decay | $T_1 \sim 50-500 \mu s$ |
| **Dephasing** ($T_2$) | Random phase fluctuations | Loss of coherence | $T_2 \leq 2T_1$ |
| **Gate error** | Imperfect control pulses | Wrong rotation angle/axis | Per gate: $10^{-4}$ to $10^{-2}$ |
| **Measurement error** | Readout crosstalk, thermal excitation | Wrong bit value | $10^{-3}$ to $10^{-1}$ |
| **Crosstalk** | Unwanted qubit-qubit coupling | Correlated errors | Depends on connectivity |
| **Leakage** | Population leaving computational subspace | State leaves $\{|0\rangle, |1\rangle\}$ | Varies |

### 1.3 Coherent vs. Incoherent Errors

**Coherent errors**: Systematic over/under-rotation of gates. These are unitary and can be corrected by recalibrating the gate. Example: applying $R_z(\theta + \epsilon)$ instead of $R_z(\theta)$.

**Incoherent errors**: Random, irreversible processes described by quantum channels. These are harder to correct and are the primary concern for error correction.

---

## 2. Kraus Operator Representation

### 2.1 Definition

A quantum channel $\mathcal{E}$ is a completely positive, trace-preserving (CPTP) map on density matrices:

$$\mathcal{E}(\rho) = \sum_{k=0}^{r-1} E_k \rho E_k^\dagger$$

where the **Kraus operators** $\{E_k\}$ satisfy the completeness relation:

$$\sum_{k=0}^{r-1} E_k^\dagger E_k = I$$

The number $r$ of Kraus operators is called the **Kraus rank**. For a single-qubit channel, $r \leq 4$.

### 2.2 Physical Interpretation

Each Kraus operator $E_k$ represents a possible "outcome" of the noise process:
- $E_k \rho E_k^\dagger$ is the (unnormalized) state conditioned on outcome $k$
- $p_k = \text{Tr}[E_k \rho E_k^\dagger]$ is the probability of outcome $k$
- The channel averages over all outcomes: $\mathcal{E}(\rho) = \sum_k p_k \frac{E_k \rho E_k^\dagger}{p_k}$

### 2.3 Non-Uniqueness

The Kraus representation is not unique. If $\{E_k\}$ is a Kraus representation, then so is $\{F_j\}$ where $F_j = \sum_k U_{jk} E_k$ for any unitary matrix $U$. This freedom is analogous to the freedom of choosing a basis for the environment.

### 2.4 Stinespring Dilation

Every quantum channel can be realized as a unitary on a larger system:

$$\mathcal{E}(\rho) = \text{Tr}_E[V(\rho \otimes |0\rangle\langle 0|_E)V^\dagger]$$

where $V$ is an isometry (partial unitary) from system to system+environment. The Kraus operators are related to $V$ by $E_k = \langle k_E|V|0_E\rangle$.

---

## 3. Common Quantum Channels

### 3.1 Bit-Flip Channel

With probability $p$, the qubit is flipped ($|0\rangle \leftrightarrow |1\rangle$):

$$E_0 = \sqrt{1-p}\,I, \quad E_1 = \sqrt{p}\,X$$

$$\mathcal{E}(\rho) = (1-p)\rho + p\,X\rho X$$

Effect on Bloch sphere: shrinks the $z$ component by $(1-2p)$ and the $y$ component by $(1-2p)$, preserving $x$.

### 3.2 Phase-Flip Channel

With probability $p$, the relative phase is flipped:

$$E_0 = \sqrt{1-p}\,I, \quad E_1 = \sqrt{p}\,Z$$

$$\mathcal{E}(\rho) = (1-p)\rho + p\,Z\rho Z$$

Effect: shrinks the $x$ and $y$ components of the Bloch vector by $(1-2p)$, preserving $z$.

### 3.3 Depolarizing Channel

The qubit is replaced by the maximally mixed state with probability $p$:

$$\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

Equivalently: $\mathcal{E}(\rho) = (1 - 4p/3)\rho + (4p/3)(I/2)$

Kraus operators:

$$E_0 = \sqrt{1-p}\,I, \quad E_1 = \sqrt{p/3}\,X, \quad E_2 = \sqrt{p/3}\,Y, \quad E_3 = \sqrt{p/3}\,Z$$

Effect: uniformly shrinks the Bloch vector by $(1 - 4p/3)$. The Bloch sphere becomes a smaller sphere.

### 3.4 Amplitude Damping Channel

Models energy relaxation ($T_1$ decay), where $|1\rangle$ decays to $|0\rangle$ with probability $\gamma$:

$$E_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad E_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

Effect on Bloch sphere:
- $x$ component: $r_x \to \sqrt{1-\gamma}\,r_x$
- $y$ component: $r_y \to \sqrt{1-\gamma}\,r_y$
- $z$ component: $r_z \to (1-\gamma)r_z + \gamma$

The Bloch sphere shrinks and shifts upward toward $|0\rangle$. At $\gamma = 1$, all states map to $|0\rangle$.

### 3.5 Phase Damping (Dephasing) Channel

Models pure dephasing ($T_2$ process without energy loss):

$$E_0 = \sqrt{1-\lambda}\,I, \quad E_1 = \sqrt{\lambda}\,|0\rangle\langle 0|, \quad E_2 = \sqrt{\lambda}\,|1\rangle\langle 1|$$

Or equivalently with two Kraus operators:

$$E_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\lambda} \end{pmatrix}, \quad E_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\lambda} \end{pmatrix}$$

Effect: shrinks the $x$ and $y$ Bloch components by $\sqrt{1-\lambda}$, preserving $z$.

### 3.6 Generalized Amplitude Damping

Models relaxation to a thermal state (not just $|0\rangle$) at temperature $T$:

$$E_0 = \sqrt{p}\begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad E_1 = \sqrt{p}\begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

$$E_2 = \sqrt{1-p}\begin{pmatrix} \sqrt{1-\gamma} & 0 \\ 0 & 1 \end{pmatrix}, \quad E_3 = \sqrt{1-p}\begin{pmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{pmatrix}$$

where $p = 1/(1 + e^{-\hbar\omega/k_BT})$ is the thermal population of $|0\rangle$.

### 3.7 Channel Comparison

| Channel | Kraus rank | Bloch sphere geometry | Shrink factors | Physical model |
|---------|-----------|----------------------|----------------|---------------|
| Bit-flip | 2 | Sphere $\to$ ellipsoid (x preserved) | $r_x \to r_x$, $r_y \to (1{-}2p)r_y$, $r_z \to (1{-}2p)r_z$ | Random X error |
| Phase-flip | 2 | Sphere $\to$ ellipsoid (z preserved) | $r_x \to (1{-}2p)r_x$, $r_y \to (1{-}2p)r_y$, $r_z \to r_z$ | Random Z error |
| Depolarizing | 4 | Sphere $\to$ smaller sphere | All $r_i \to (1{-}4p/3)r_i$ | Random Pauli error |
| Amplitude damping | 2 | Sphere $\to$ egg shape shifted toward $\|0\rangle$ | $r_{x,y} \to \sqrt{1{-}\gamma}\,r_{x,y}$, $r_z \to (1{-}\gamma)r_z + \gamma$ | $T_1$ relaxation |
| Phase damping | 2 | Sphere $\to$ ellipsoid (z preserved) | $r_{x,y} \to \sqrt{1{-}\lambda}\,r_{x,y}$, $r_z \to r_z$ | $T_2$ dephasing |

---

## 4. Choi-Jamiolkowski Isomorphism

### 4.1 The Choi Matrix

The Choi matrix of a channel $\mathcal{E}$ on $n$ qubits is defined as:

$$\Lambda_{\mathcal{E}} = (\mathcal{E} \otimes \mathcal{I})(|\Omega\rangle\langle\Omega|)$$

where $|\Omega\rangle = \frac{1}{\sqrt{d}}\sum_{i=0}^{d-1}|i\rangle|i\rangle$ is the maximally entangled state and $d = 2^n$.

For a single qubit ($d = 2$):

$$|\Omega\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

### 4.2 Properties

The Choi matrix encodes everything about the channel:

- **Complete positivity**: $\Lambda_{\mathcal{E}} \geq 0$ (positive semidefinite)
- **Trace preservation**: $\text{Tr}_1[\Lambda_{\mathcal{E}}] = I/d$
- **Kraus operators**: The eigenvectors of $\Lambda_{\mathcal{E}}$ give (rescaled) Kraus operators
- **Channel rank**: equals the rank of $\Lambda_{\mathcal{E}}$

### 4.3 Recovering the Channel

Given the Choi matrix $\Lambda$, the channel action on any state $\rho$ is:

$$\mathcal{E}(\rho) = d \cdot \text{Tr}_2[(\rho^T \otimes I)\Lambda]$$

### 4.4 Fidelity Measures

**Average gate fidelity**: How close the noisy gate is to the ideal:

$$\bar{F}(\mathcal{E}, \mathcal{U}) = \int d\psi \langle\psi|\mathcal{U}^\dagger(\mathcal{E}(|\psi\rangle\langle\psi|))|\psi\rangle$$

This can be computed from the Choi matrix:

$$\bar{F} = \frac{d \cdot F_e + 1}{d + 1}$$

where $F_e = \text{Tr}[\Lambda_{\mathcal{E}} \Lambda_{\mathcal{U}}] / d$ is the entanglement fidelity.

---

## 5. Quantum Process Tomography

### 5.1 Standard Process Tomography

To completely characterize an unknown single-qubit channel $\mathcal{E}$:

1. **Prepare** a set of input states that form a basis for the density matrix space. For a single qubit, 4 states suffice:
   $\{|0\rangle, |1\rangle, |+\rangle, |+i\rangle\}$

2. **Apply** the channel $\mathcal{E}$ to each input state

3. **Measure** each output state using quantum state tomography (measure in $X$, $Y$, $Z$ bases)

4. **Reconstruct** the channel from the input-output data

### 5.2 Chi Matrix Representation

Express the channel in the Pauli basis:

$$\mathcal{E}(\rho) = \sum_{m,n} \chi_{mn} P_m \rho P_n^\dagger$$

where $\{P_m\} = \{I, X, Y, Z\}$ and $\chi$ is a $4 \times 4$ positive semidefinite matrix (the **chi matrix** or **process matrix**).

The chi matrix is related to the Kraus operators by:

$$E_k = \sum_m e_{km} P_m \implies \chi_{mn} = \sum_k e_{km}^* e_{kn}$$

### 5.3 Maximum Likelihood Estimation

Raw tomography data may yield an unphysical chi matrix (not positive semidefinite). Maximum likelihood estimation enforces physicality:

$$\hat{\chi} = \arg\max_{\chi \geq 0, \text{TP}} \prod_{i} P(\text{data}_i | \chi)$$

### 5.4 Gate Set Tomography (GST)

Standard process tomography assumes perfect state preparation and measurement. **Gate Set Tomography** relaxes this assumption by simultaneously characterizing:

- The gate operations
- The state preparations
- The measurements

This self-consistent approach gives more accurate results but requires more data and computation.

---

## 6. Noise Characterization Methods

### 6.1 Randomized Benchmarking (RB)

RB measures the average error rate of gates without full tomography:

1. **Generate** $m$ random sequences of Clifford gates of increasing length $L$
2. **Append** the inverse Clifford to each sequence (making the ideal net operation = identity)
3. **Measure** the survival probability $p(L)$ (probability of returning to the initial state)
4. **Fit** $p(L) = A \cdot r^L + B$ where $r$ is the depolarizing parameter
5. **Extract** the average error per Clifford: $\epsilon = (1-r)(1 - 1/d)$ where $d = 2^n$

**Advantages**:
- Robust against state preparation and measurement (SPAM) errors
- Gives a single, meaningful error number
- Efficient: polynomial in the number of qubits

### 6.2 Interleaved Randomized Benchmarking

To measure the error of a specific gate $G$:

1. Run standard RB to get $r_{\text{ref}}$
2. Run RB with gate $G$ interleaved between random Cliffords to get $r_G$
3. The gate error is: $\epsilon_G = (1 - r_G/r_{\text{ref}})(1 - 1/d)$

### 6.3 Cycle Benchmarking

For multi-qubit systems, cycle benchmarking characterizes noise on specific circuit layers (cycles of parallel gates), capturing crosstalk effects that single-gate benchmarking misses.

### 6.4 Spectral Analysis of Noise

Noise can have temporal correlations (non-Markovian noise). Spectral methods characterize the noise power spectrum:

$$S(\omega) = \int_{-\infty}^{\infty} \langle \beta(t)\beta(0)\rangle e^{-i\omega t} dt$$

where $\beta(t)$ is the noise process. Dynamical decoupling sequences (like CPMG) can probe $S(\omega)$ at different frequencies.

---

### 6.5 Pauli Twirling

**Pauli twirling** is a technique that converts an arbitrary noise channel into a Pauli channel (a probabilistic mixture of Pauli errors), simplifying noise analysis and error correction.

The idea is to sandwich the noisy operation $\mathcal{E}$ between random Pauli gates. Before the gate, apply a uniformly random Pauli $P_i$; after the gate, apply $P_i^\dagger$. Averaging over all $4^n$ Pauli operators:

$$\mathcal{E}_{\text{twirled}}(\rho) = \frac{1}{4^n}\sum_{i} P_i \mathcal{E}(P_i \rho P_i^\dagger) P_i^\dagger$$

The twirled channel $\mathcal{E}_{\text{twirled}}$ is always a **Pauli channel**: $\mathcal{E}_{\text{twirled}}(\rho) = \sum_j p_j P_j \rho P_j$ where $p_j \geq 0$ and $\sum_j p_j = 1$. Crucially, the average fidelity is preserved — twirling does not make the noise worse on average, it only simplifies its structure.

Pauli twirling is widely used in quantum error correction analysis (where Pauli channels are much easier to decode) and in error mitigation techniques like probabilistic error cancellation. In practice, the random Pauli gates can be compiled into the existing circuit with negligible overhead.

---

## 7. Noise in Quantum Algorithms

### 7.1 Effect on Circuit Fidelity

For a circuit with $G$ gates, each with error rate $\epsilon$, the overall circuit fidelity is approximately:

$$F_{\text{circuit}} \approx (1 - \epsilon)^G \approx e^{-\epsilon G}$$

For useful computation, we need $F_{\text{circuit}} \gtrsim 0.5$, giving the constraint:

$$G \lesssim \frac{1}{\epsilon}$$

With current error rates ($\epsilon \sim 10^{-3}$), this limits circuits to $\sim 1000$ gates.

### 7.2 Effect on Specific Algorithms

| Algorithm | Gate count | Required $\epsilon$ | NISQ feasible? |
|-----------|-----------|-------------------|---------------|
| 5-qubit VQE | $\sim 50$ | $\sim 10^{-2}$ | Yes |
| QAOA $p=3$ | $\sim 200$ | $\sim 5 \times 10^{-3}$ | Maybe |
| Grover ($N=10^6$) | $\sim 10^4$ | $\sim 10^{-4}$ | No |
| Shor (2048-bit RSA) | $\sim 10^{12}$ | $\sim 10^{-12}$ | No (needs FT) |

### 7.3 Error Mitigation vs. Error Correction

**Error mitigation** (for NISQ):
- Post-processing techniques to reduce the effect of noise
- No additional qubits needed
- Imperfect correction, introduces sampling overhead
- Examples: ZNE, PEC, symmetry verification, Clifford data regression

**Error correction** (for fault-tolerant QC):
- Encodes logical qubits in many physical qubits
- Actively detects and corrects errors
- Overhead: $\sim 1000$-$10000$ physical qubits per logical qubit
- Below the threshold error rate, arbitrarily long computations are possible

---

## 8. Error Thresholds and Fault Tolerance

### 8.1 Threshold Theorem

If the physical error rate $p$ is below a threshold $p_{\text{th}}$, then arbitrarily long quantum computations can be performed with error rate:

$$p_{\text{logical}} \leq c \left(\frac{p}{p_{\text{th}}}\right)^{2^L}$$

where $L$ is the number of levels of concatenated error correction and $c$ is a constant.

### 8.2 Threshold Values

| Error correction code | Threshold |
|----------------------|-----------|
| Steane code (concatenated) | $\sim 10^{-5}$ |
| Surface code | $\sim 1\%$ |
| Color code | $\sim 0.1\%$ |
| Topological codes | $\sim 1\%$ |

The surface code threshold of $\sim 1\%$ is the most relevant for near-term hardware, as current gate error rates are approaching $10^{-3}$.

### 8.3 Resource Estimates

For factoring a 2048-bit RSA key using the surface code:
- Logical qubits needed: $\sim 4000$
- Physical qubits per logical qubit: $\sim 1000-10000$ (depends on error rate)
- Total physical qubits: $\sim 4 \times 10^6$ to $4 \times 10^7$
- Runtime: hours to days (at GHz clock rates)

---

## 9. Python Implementation

### 9.1 Quantum Channels with Kraus Operators

```python
import numpy as np

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def apply_channel(rho, kraus_ops):
    """Apply a quantum channel defined by Kraus operators to a density matrix.

    The channel is: E(rho) = sum_k E_k rho E_k^dagger
    The completeness condition sum_k E_k^dagger E_k = I ensures
    that the output is a valid density matrix (trace 1, positive semidefinite).

    Args:
        rho: Input density matrix
        kraus_ops: List of Kraus operator matrices

    Returns:
        Output density matrix
    """
    rho_out = np.zeros_like(rho)
    for E in kraus_ops:
        rho_out += E @ rho @ E.conj().T
    return rho_out


def verify_completeness(kraus_ops):
    """Verify the completeness relation sum_k E_k^dagger E_k = I."""
    d = kraus_ops[0].shape[0]
    total = np.zeros((d, d), dtype=complex)
    for E in kraus_ops:
        total += E.conj().T @ E
    return np.allclose(total, np.eye(d))


def bit_flip_channel(p):
    """Bit-flip channel: flips qubit with probability p."""
    E0 = np.sqrt(1 - p) * I
    E1 = np.sqrt(p) * X
    return [E0, E1]


def phase_flip_channel(p):
    """Phase-flip channel: applies Z with probability p."""
    E0 = np.sqrt(1 - p) * I
    E1 = np.sqrt(p) * Z
    return [E0, E1]


def depolarizing_channel(p):
    """Depolarizing channel: applies random Pauli with probability p."""
    E0 = np.sqrt(1 - p) * I
    E1 = np.sqrt(p / 3) * X
    E2 = np.sqrt(p / 3) * Y
    E3 = np.sqrt(p / 3) * Z
    return [E0, E1, E2, E3]


def amplitude_damping_channel(gamma):
    """Amplitude damping: |1> decays to |0> with probability gamma."""
    E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    E1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [E0, E1]


def phase_damping_channel(lam):
    """Phase damping: dephasing with probability lambda."""
    E0 = np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex)
    E1 = np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex)
    return [E0, E1]


def density_to_bloch(rho):
    """Convert density matrix to Bloch vector (rx, ry, rz)."""
    rx = 2 * np.real(rho[0, 1])
    ry = 2 * np.imag(rho[1, 0])
    rz = np.real(rho[0, 0] - rho[1, 1])
    return np.array([rx, ry, rz])


def bloch_to_density(r):
    """Convert Bloch vector to density matrix."""
    return (I + r[0] * X + r[1] * Y + r[2] * Z) / 2


# Demonstrate channels
print("=" * 65)
print("Quantum Channel Effects on Bloch Vector")
print("=" * 65)

# Test state: |+> = (|0> + |1>) / sqrt(2)
rho_plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
# Test state: |+i> = (|0> + i|1>) / sqrt(2)
rho_plus_i = np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=complex)
# Test state: |0>
rho_zero = np.array([[1, 0], [0, 0]], dtype=complex)

test_states = {
    '|+>': rho_plus,
    '|+i>': rho_plus_i,
    '|0>': rho_zero,
}

channels = {
    'Bit-flip (p=0.1)': bit_flip_channel(0.1),
    'Phase-flip (p=0.1)': phase_flip_channel(0.1),
    'Depolarizing (p=0.1)': depolarizing_channel(0.1),
    'Amp. damping (g=0.3)': amplitude_damping_channel(0.3),
    'Phase damping (l=0.3)': phase_damping_channel(0.3),
}

for state_name, rho in test_states.items():
    bloch_in = density_to_bloch(rho)
    print(f"\nInput: {state_name}, Bloch = ({bloch_in[0]:.2f}, {bloch_in[1]:.2f}, {bloch_in[2]:.2f})")
    print(f"  {'Channel':>25} {'Bloch out':>30} {'|r| shrink':>12}")
    print(f"  {'-' * 70}")

    for ch_name, kraus in channels.items():
        rho_out = apply_channel(rho, kraus)
        bloch_out = density_to_bloch(rho_out)
        shrink = np.linalg.norm(bloch_out) / max(np.linalg.norm(bloch_in), 1e-10)
        print(f"  {ch_name:>25} ({bloch_out[0]:7.4f}, {bloch_out[1]:7.4f}, {bloch_out[2]:7.4f})"
              f" {shrink:12.4f}")
```

### 9.2 Choi Matrix Construction

```python
import numpy as np

def compute_choi_matrix(kraus_ops):
    """Compute the Choi-Jamiolkowski matrix of a quantum channel.

    The Choi matrix Lambda = (E tensor I)(|Omega><Omega|)
    where |Omega> = (1/sqrt(d)) * sum_i |i>|i> is the maximally entangled state.

    The Choi matrix completely characterizes the channel:
    - Positive semidefinite <=> completely positive
    - Tr_1[Lambda] = I/d <=> trace-preserving

    Args:
        kraus_ops: List of Kraus operator matrices

    Returns:
        choi: Choi matrix (d^2 x d^2)
    """
    d = kraus_ops[0].shape[0]

    # Maximally entangled state |Omega>
    omega = np.zeros(d * d, dtype=complex)
    for i in range(d):
        omega[i * d + i] = 1.0 / np.sqrt(d)

    rho_omega = np.outer(omega, omega.conj())

    # Apply channel to first subsystem
    choi = np.zeros((d * d, d * d), dtype=complex)
    for E in kraus_ops:
        # E tensor I
        E_tensor_I = np.kron(E, np.eye(d, dtype=complex))
        choi += E_tensor_I @ rho_omega @ E_tensor_I.conj().T

    return choi


def choi_to_kraus(choi, d=2):
    """Extract Kraus operators from Choi matrix via eigendecomposition.

    The Choi matrix Lambda = sum_k lambda_k |v_k><v_k|
    gives Kraus operators E_k = sqrt(d * lambda_k) * reshape(v_k).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(choi)

    kraus_ops = []
    for k in range(len(eigenvalues)):
        if eigenvalues[k] > 1e-10:
            vec = eigenvectors[:, k]
            E = np.sqrt(d * eigenvalues[k]) * vec.reshape(d, d)
            kraus_ops.append(E)

    return kraus_ops


# Demonstrate Choi matrix
print("=" * 65)
print("Choi-Jamiolkowski Isomorphism")
print("=" * 65)

for name, kraus in [
    ("Identity", [I]),
    ("Depolarizing (p=0.2)", depolarizing_channel(0.2)),
    ("Amplitude damping (g=0.3)", amplitude_damping_channel(0.3)),
]:
    choi = compute_choi_matrix(kraus)
    eigenvals = np.sort(np.real(np.linalg.eigvalsh(choi)))[::-1]

    print(f"\n{name}:")
    print(f"  Choi matrix shape: {choi.shape}")
    print(f"  Positive semidefinite: {np.all(eigenvals >= -1e-10)}")
    print(f"  Eigenvalues: {eigenvals}")

    # Verify trace-preservation
    d = 2
    partial_trace = np.zeros((d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                partial_trace[i, j] += choi[i * d + k, j * d + k]
    print(f"  Tr_1[Lambda] ≈ I/d: {np.allclose(partial_trace, np.eye(d) / d)}")

    # Recover Kraus operators
    recovered_kraus = choi_to_kraus(choi)
    print(f"  Kraus rank: {len(recovered_kraus)}")

    # Verify recovered channel equals original
    rho_test = np.array([[0.7, 0.3j], [-0.3j, 0.3]], dtype=complex)
    rho_orig = apply_channel(rho_test, kraus)
    rho_recovered = apply_channel(rho_test, recovered_kraus)
    print(f"  Channel recovery error: {np.linalg.norm(rho_orig - rho_recovered):.2e}")
```

### 9.3 Quantum Process Tomography

```python
import numpy as np

def process_tomography_single_qubit(channel_func):
    """Perform quantum process tomography on a single-qubit channel.

    Procedure:
    1. Prepare 4 input states {|0>, |1>, |+>, |+i>}
    2. Apply the channel to each
    3. Perform state tomography on each output (measure X, Y, Z)
    4. Reconstruct the chi matrix in the Pauli basis

    Args:
        channel_func: Function that takes density matrix, returns output density matrix

    Returns:
        chi: Process matrix in the Pauli basis (4x4)
    """
    # Pauli basis (normalized)
    paulis = [I, X, Y, Z]
    pauli_labels = ['I', 'X', 'Y', 'Z']

    # Input states for tomography
    ket0 = np.array([1, 0], dtype=complex)
    ket1 = np.array([0, 1], dtype=complex)
    ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    ket_plus_i = np.array([1, 1j], dtype=complex) / np.sqrt(2)

    input_states = [
        np.outer(ket0, ket0.conj()),
        np.outer(ket1, ket1.conj()),
        np.outer(ket_plus, ket_plus.conj()),
        np.outer(ket_plus_i, ket_plus_i.conj()),
    ]

    # Apply channel to each input
    output_states = [channel_func(rho) for rho in input_states]

    # Reconstruct chi matrix
    # The channel is E(rho) = sum_{mn} chi_{mn} P_m rho P_n^dagger
    # We need to solve for chi using the input-output pairs

    # Build the linear system: vec(E(rho_j)) = Lambda * vec(chi)
    # where Lambda encodes the Pauli products P_m rho_j P_n^dagger

    n_paulis = 4
    # Build system matrix
    A_matrix = np.zeros((4 * 4, n_paulis * n_paulis), dtype=complex)

    for j in range(4):  # for each input state
        rho_j = input_states[j]
        rho_j_out = output_states[j]

        for m in range(n_paulis):
            for n in range(n_paulis):
                contrib = paulis[m] @ rho_j @ paulis[n].conj().T
                for a in range(2):
                    for b in range(2):
                        row = j * 4 + a * 2 + b
                        col = m * n_paulis + n
                        A_matrix[row, col] = contrib[a, b]

    # Target vector
    b_vector = np.zeros(4 * 4, dtype=complex)
    for j in range(4):
        rho_out = output_states[j]
        for a in range(2):
            for b in range(2):
                b_vector[j * 4 + a * 2 + b] = rho_out[a, b]

    # Solve (least squares)
    chi_flat, _, _, _ = np.linalg.lstsq(A_matrix, b_vector, rcond=None)
    chi = chi_flat.reshape(n_paulis, n_paulis)

    return chi, pauli_labels


# Demonstrate process tomography
print("=" * 65)
print("Quantum Process Tomography")
print("=" * 65)

test_channels = {
    'Identity': lambda rho: rho,
    'X gate': lambda rho: X @ rho @ X.conj().T,
    'Depolarizing (p=0.1)': lambda rho: apply_channel(rho, depolarizing_channel(0.1)),
    'Amplitude damping (g=0.3)': lambda rho: apply_channel(rho, amplitude_damping_channel(0.3)),
}

for name, channel_func in test_channels.items():
    chi, labels = process_tomography_single_qubit(channel_func)

    print(f"\n{name}:")
    print(f"  Chi matrix (Pauli basis):")
    print(f"  {'':>5}", end="")
    for l in labels:
        print(f" {l:>10}", end="")
    print()

    for i, li in enumerate(labels):
        print(f"  {li:>5}", end="")
        for j in range(4):
            val = chi[i, j]
            if abs(val.imag) < 1e-6:
                print(f" {val.real:10.4f}", end="")
            else:
                print(f" {val:10.4f}", end="")
        print()

    # Check: for identity channel, chi should have chi[0,0] = 1, rest 0
    # For X gate, chi[1,1] = 1, rest 0
    print(f"  Trace(chi) = {np.trace(chi).real:.4f} (should be 1)")
```

### 9.4 Randomized Benchmarking Simulation

```python
import numpy as np
from scipy.optimize import curve_fit

# Single-qubit Clifford group (24 elements)
def generate_single_qubit_cliffords():
    """Generate the 24 elements of the single-qubit Clifford group.

    The Clifford group consists of all unitaries that map Pauli operators
    to Pauli operators under conjugation. For a single qubit, there are
    24 such unitaries (the symmetry group of the octahedron).
    """
    H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S_gate = np.array([[1, 0], [0, 1j]], dtype=complex)
    Sdg = S_gate.conj().T

    # Generate by composing H and S (they generate the group)
    cliffords = set()
    queue = [np.eye(2, dtype=complex)]
    seen_keys = set()

    def matrix_key(M):
        # Normalize global phase
        for i in range(4):
            if abs(M.flat[i]) > 0.1:
                M = M / (M.flat[i] / abs(M.flat[i]))
                break
        return tuple(np.round(M.flatten(), 6))

    while queue:
        current = queue.pop(0)
        key = matrix_key(current)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        if len(seen_keys) > 24:
            break

        queue.append(H_gate @ current)
        queue.append(S_gate @ current)

    # Convert to list
    result = []
    for key in seen_keys:
        arr = np.array(key, dtype=complex).reshape(2, 2)
        result.append(arr)

    return result[:24]


def noisy_gate(U, error_rate):
    """Apply a gate followed by depolarizing noise.

    This models the typical error process on real hardware:
    the intended unitary is applied, but followed by random Pauli errors.
    """
    d = U.shape[0]
    kraus = depolarizing_channel(error_rate)

    def apply(rho):
        rho = U @ rho @ U.conj().T
        return apply_channel(rho, kraus)

    return apply


def randomized_benchmarking(error_rate, sequence_lengths, n_sequences=50, seed=42):
    """Simulate a randomized benchmarking experiment.

    RB measures the average error rate of gates by:
    1. Applying random Clifford sequences of increasing length
    2. Appending the inverse to make the ideal net operation = identity
    3. Measuring how survival probability decays with sequence length

    The decay rate directly gives the average gate error.

    Args:
        error_rate: Depolarizing error per gate
        sequence_lengths: List of sequence lengths to test
        n_sequences: Number of random sequences per length
        seed: Random seed

    Returns:
        sequence_lengths: Input lengths
        survival_probs: Average survival probability at each length
    """
    rng = np.random.default_rng(seed)
    cliffords = generate_single_qubit_cliffords()
    n_cliffords = len(cliffords)

    survival_probs = []

    for L in sequence_lengths:
        survivals = []

        for _ in range(n_sequences):
            # Random Clifford sequence
            indices = rng.integers(0, n_cliffords, size=L)
            gates = [cliffords[i] for i in indices]

            # Compute inverse
            total_unitary = np.eye(2, dtype=complex)
            for g in gates:
                total_unitary = g @ total_unitary
            inverse_gate = total_unitary.conj().T

            # Apply noisy sequence to |0>
            rho = np.array([[1, 0], [0, 0]], dtype=complex)

            for g in gates:
                noisy = noisy_gate(g, error_rate)
                rho = noisy(rho)

            # Apply noisy inverse
            noisy_inv = noisy_gate(inverse_gate, error_rate)
            rho = noisy_inv(rho)

            # Survival probability: probability of being in |0>
            survival = np.real(rho[0, 0])
            survivals.append(survival)

        survival_probs.append(np.mean(survivals))

    return sequence_lengths, survival_probs


# Run randomized benchmarking
print("=" * 65)
print("Randomized Benchmarking Simulation")
print("=" * 65)

error_rate = 0.01  # 1% depolarizing error per gate
lengths = [1, 2, 5, 10, 20, 50, 100, 200]

print(f"\nError rate per gate: {error_rate}")
print(f"Sequence lengths: {lengths}\n")

lengths, probs = randomized_benchmarking(error_rate, lengths, n_sequences=100)

print(f"{'Length':>8} {'Survival':>12}")
print("-" * 22)
for l, p in zip(lengths, probs):
    print(f"{l:8d} {p:12.4f}")

# Fit exponential decay: p(L) = A * r^L + B
def rb_model(L, A, r, B):
    return A * r ** np.array(L) + B

try:
    popt, pcov = curve_fit(rb_model, lengths, probs, p0=[0.5, 0.99, 0.5], maxfev=5000)
    A, r, B = popt

    # Error per Clifford gate
    d = 2  # single qubit
    epsilon_rb = (1 - r) * (1 - 1 / d)

    print(f"\nFit: p(L) = {A:.4f} * {r:.6f}^L + {B:.4f}")
    print(f"Depolarizing parameter r = {r:.6f}")
    print(f"Error per Clifford: {epsilon_rb:.6f}")
    print(f"True error rate: {error_rate}")
    print(f"Ratio (Cliffords are ~1.875 gates on average): {epsilon_rb / error_rate:.2f}")
except RuntimeError:
    print("\nCurve fitting did not converge — try more sequences or different initial parameters.")
```

### 9.5 Noise Impact on Quantum Algorithms

```python
import numpy as np
from scipy.linalg import expm

def noisy_circuit_simulation(n_qubits, n_gates, error_rate, n_trials=1000):
    """Simulate the effect of depolarizing noise on circuit fidelity.

    Applies random Pauli errors after each gate and measures
    the probability that the final state matches the ideal output.

    This demonstrates why deep circuits fail on noisy hardware.
    """
    rng = np.random.default_rng(42)
    N = 2 ** n_qubits

    # Ideal initial state
    psi_ideal = np.zeros(N, dtype=complex)
    psi_ideal[0] = 1.0

    # Generate a random circuit (sequence of random unitaries)
    gates = []
    for _ in range(n_gates):
        # Random single-qubit gate on a random qubit
        qubit = rng.integers(0, n_qubits)
        angle = rng.uniform(0, 2 * np.pi)
        axis = rng.choice(['x', 'y', 'z'])

        pauli = {'x': X, 'y': Y, 'z': Z}[axis]
        U_local = expm(-1j * angle / 2 * pauli)

        # Full unitary
        ops = [np.eye(2, dtype=complex)] * n_qubits
        ops[qubit] = U_local
        U_full = ops[0]
        for op in ops[1:]:
            U_full = np.kron(U_full, op)

        gates.append((U_full, qubit))

    # Ideal evolution
    psi = psi_ideal.copy()
    for U, _ in gates:
        psi = U @ psi
    rho_ideal = np.outer(psi, psi.conj())

    # Noisy evolution (density matrix)
    rho = np.outer(psi_ideal, psi_ideal.conj())
    for U, qubit in gates:
        rho = U @ rho @ U.conj().T

        # Apply depolarizing noise on the target qubit
        if error_rate > 0:
            # Single-qubit depolarizing on specified qubit
            for pauli_op in [X, Y, Z]:
                ops = [np.eye(2, dtype=complex)] * n_qubits
                ops[qubit] = pauli_op
                P_full = ops[0]
                for op in ops[1:]:
                    P_full = np.kron(P_full, op)
                rho = (1 - error_rate) * rho + (error_rate / 3) * P_full @ rho @ P_full

    fidelity = np.real(np.trace(rho_ideal @ rho))
    return fidelity


print("=" * 65)
print("Noise Impact on Circuit Fidelity")
print("=" * 65)

print(f"\n{'n_qubits':>10} {'n_gates':>10} {'error_rate':>12} {'fidelity':>12} {'theory':>12}")
print("-" * 60)

for n_qubits in [2, 3, 4]:
    for n_gates in [10, 50, 100]:
        for error_rate in [0.001, 0.01]:
            fidelity = noisy_circuit_simulation(n_qubits, n_gates, error_rate)
            theory = (1 - error_rate) ** n_gates
            print(f"{n_qubits:10d} {n_gates:10d} {error_rate:12.4f} {fidelity:12.4f} {theory:12.4f}")
```

---

## 10. Exercises

### Exercise 1: Channel Composition

For two channels applied sequentially, $\mathcal{E}_2 \circ \mathcal{E}_1$:
(a) Compute the Kraus operators of the composition of two depolarizing channels with parameters $p_1 = 0.1$ and $p_2 = 0.2$.
(b) Is the result equivalent to a single depolarizing channel? If so, what is its parameter?
(c) Repeat for the composition of an amplitude damping channel ($\gamma = 0.3$) followed by a phase damping channel ($\lambda = 0.2$).
(d) Is channel composition commutative in general? Verify numerically.

### Exercise 2: Choi Matrix Analysis

For the depolarizing channel with parameter $p \in [0, 1]$:
(a) Compute the Choi matrix as a function of $p$.
(b) Plot the eigenvalues of the Choi matrix as a function of $p$.
(c) At what value of $p$ does the channel become entanglement-breaking (all eigenvalues of the partial transpose are non-negative)?
(d) Compute the diamond norm distance between the depolarizing channel and the identity.

### Exercise 3: Process Tomography with Noise

Simulate process tomography with measurement noise:
(a) Add Gaussian noise with standard deviation $\sigma$ to each tomographic measurement.
(b) How does the reconstructed chi matrix change as $\sigma$ increases?
(c) Implement maximum likelihood estimation to ensure the chi matrix is physical.
(d) How many measurement repetitions are needed to achieve a process fidelity error of $< 0.01$?

### Exercise 4: Randomized Benchmarking Extensions

Extend the RB simulation:
(a) Implement interleaved RB to measure the error rate of the Hadamard gate specifically.
(b) Add non-Markovian noise (correlated errors between gates) and observe how it affects the RB decay curve.
(c) Compare the RB error rate with the process fidelity from tomography. When do they disagree?

### Exercise 5: Noise-Aware Algorithm Design

For the 3-qubit VQE on the Ising model:
(a) Simulate VQE with depolarizing noise at rates $p = 0, 0.001, 0.01, 0.05$.
(b) Implement zero-noise extrapolation: run at $p, 2p, 3p$ and extrapolate to $p = 0$.
(c) How much does ZNE improve the energy estimate?
(d) At what noise rate does VQE become useless (error larger than the chemical accuracy of 1.6 mHa)?

---

[← Previous: Quantum Walks](19_Quantum_Walks.md) | [Next: Quantum Chemistry →](21_Quantum_Chemistry.md)
