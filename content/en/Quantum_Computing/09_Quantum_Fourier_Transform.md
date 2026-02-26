# Lesson 9: Quantum Fourier Transform

[← Previous: Grover's Search Algorithm](08_Grovers_Search.md) | [Next: Shor's Factoring Algorithm →](10_Shors_Algorithm.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the connection between the classical Discrete Fourier Transform and the Quantum Fourier Transform
2. Write the QFT transformation in Dirac notation and matrix form
3. Construct the QFT circuit using Hadamard gates and controlled rotation gates
4. Analyze the gate complexity of QFT: $O(n^2)$ quantum vs $O(n \cdot 2^n)$ classical
5. Describe the Quantum Phase Estimation (QPE) algorithm and its role as a subroutine
6. Implement the inverse QFT and explain when it is needed
7. Simulate QFT and phase estimation using NumPy

---

The Quantum Fourier Transform is arguably the most important subroutine in quantum computing. It lies at the heart of Shor's factoring algorithm, quantum phase estimation, and numerous other quantum algorithms that achieve exponential speedups. Understanding QFT is the gateway to understanding why quantum computers threaten modern cryptography and how they can solve problems in chemistry, optimization, and beyond.

What makes QFT remarkable is not that it computes a different transform than the classical DFT — it computes exactly the same mathematical transformation. The magic is in *how* it computes it: a quantum circuit with only $O(n^2)$ gates can transform $2^n$ amplitudes, whereas the best classical algorithm (FFT) requires $O(n \cdot 2^n)$ operations. This exponential reduction in gate count is the engine that powers quantum speedups in period-finding problems.

> **Analogy:** QFT is like a prism for quantum states — it decomposes a state into its frequency components, revealing hidden periodic structure. Just as white light passing through a prism separates into a rainbow of frequencies, the QFT separates a quantum state into a superposition of phase-shifted basis states, making hidden periodicities visible to measurement.

## Table of Contents

1. [Classical DFT Review](#1-classical-dft-review)
2. [From Classical DFT to Quantum FT](#2-from-classical-dft-to-quantum-ft)
3. [QFT Definition and Matrix Structure](#3-qft-definition-and-matrix-structure)
4. [The QFT Circuit](#4-the-qft-circuit)
5. [Gate Complexity Analysis](#5-gate-complexity-analysis)
6. [Inverse QFT](#6-inverse-qft)
7. [Quantum Phase Estimation](#7-quantum-phase-estimation)
8. [Python Implementation](#8-python-implementation)
9. [Exercises](#9-exercises)

---

## 1. Classical DFT Review

Before diving into the quantum version, let us recall the classical Discrete Fourier Transform (see also Signal_Processing L05 for a thorough treatment).

### 1.1 The DFT Definition

Given a sequence of $N$ complex numbers $(x_0, x_1, \ldots, x_{N-1})$, the DFT produces another sequence $(y_0, y_1, \ldots, y_{N-1})$ defined by:

$$y_k = \frac{1}{\sqrt{N}} \sum_{j=0}^{N-1} x_j \, \omega^{jk}$$

where $\omega = e^{2\pi i / N}$ is the primitive $N$-th root of unity.

The factor $\frac{1}{\sqrt{N}}$ is the *unitary* normalization convention (some references use $\frac{1}{N}$ or no factor). We use the unitary convention because quantum transformations must be unitary.

### 1.2 The DFT Matrix

The DFT can be written as a matrix-vector multiplication $\mathbf{y} = F_N \mathbf{x}$, where:

$$F_N = \frac{1}{\sqrt{N}} \begin{pmatrix} 1 & 1 & 1 & \cdots & 1 \\ 1 & \omega & \omega^2 & \cdots & \omega^{N-1} \\ 1 & \omega^2 & \omega^4 & \cdots & \omega^{2(N-1)} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & \omega^{N-1} & \omega^{2(N-1)} & \cdots & \omega^{(N-1)^2} \end{pmatrix}$$

The entry in row $k$, column $j$ is:

$$(F_N)_{kj} = \frac{1}{\sqrt{N}} \omega^{jk} = \frac{1}{\sqrt{N}} e^{2\pi i \, jk / N}$$

### 1.3 Key Properties

- **Unitarity**: $F_N^{\dagger} F_N = I$, so $F_N^{-1} = F_N^{\dagger}$
- **Periodicity detection**: If $x_j$ has period $r$ (i.e., $x_j = x_{j+r}$), then $y_k$ is concentrated at multiples of $N/r$
- **Complexity**: Direct computation is $O(N^2)$; the Fast Fourier Transform (FFT) achieves $O(N \log N)$

The periodicity detection property is the reason QFT is so powerful in quantum algorithms — it reveals hidden periods that would take exponential classical time to find.

### 1.4 Python: Classical DFT

```python
import numpy as np

def classical_dft(x):
    """Compute the DFT using the unitary normalization convention.

    Why unitary normalization? Quantum mechanics requires all transformations
    to preserve the norm of state vectors. The 1/sqrt(N) factor ensures
    that the DFT matrix is unitary (F†F = I), making it a valid quantum gate.
    """
    N = len(x)
    # Build the DFT matrix element by element
    # (F_N)_{kj} = (1/sqrt(N)) * exp(2πi * j * k / N)
    j = np.arange(N)
    k = np.arange(N).reshape(-1, 1)
    omega_matrix = np.exp(2j * np.pi * j * k / N)
    return omega_matrix @ x / np.sqrt(N)

# Example: DFT of a periodic signal
N = 8
# Signal with period 2: [1, 0, 1, 0, 1, 0, 1, 0]
x = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=complex)
y = classical_dft(x)

print("Input signal:", x.real)
print("DFT magnitudes:", np.abs(y).round(4))
# The peak at index 4 = N/2 reveals period r=2 (since N/r = 8/2 = 4)
```

---

## 2. From Classical DFT to Quantum FT

### 2.1 The Conceptual Bridge

The key insight connecting the classical DFT to the quantum version is simple: **the QFT performs exactly the same mathematical transformation, but on the amplitudes of a quantum state**.

In the classical case, we transform a vector of $N$ numbers. In the quantum case, we transform the $N = 2^n$ amplitudes of an $n$-qubit state. If we have a state:

$$|\psi\rangle = \sum_{j=0}^{N-1} x_j |j\rangle$$

then the QFT produces:

$$\text{QFT}|\psi\rangle = \sum_{k=0}^{N-1} y_k |k\rangle$$

where $y_k = \frac{1}{\sqrt{N}} \sum_{j=0}^{N-1} x_j \, e^{2\pi i \, jk/N}$, exactly the DFT formula.

### 2.2 Why the Quantum Version is Special

The classical DFT on $N = 2^n$ numbers requires $O(N \log N) = O(n \cdot 2^n)$ operations with FFT. The quantum version requires only $O(n^2)$ gates. This is an *exponential* reduction.

However, there is a crucial subtlety: **we cannot directly read out all $N$ amplitudes**. Measurement collapses the state to a single basis state $|k\rangle$ with probability $|y_k|^2$. So the QFT is not a general-purpose replacement for the classical FFT. Its power lies in algorithms where we only need to *sample* from the Fourier-transformed distribution, not read every coefficient.

This sampling is sufficient for period finding (Shor's algorithm), phase estimation, and the hidden subgroup problem — all cases where the Fourier spectrum has a sharp structure.

---

## 3. QFT Definition and Matrix Structure

### 3.1 Formal Definition

The Quantum Fourier Transform on $N = 2^n$ basis states is defined by its action on computational basis states:

$$\text{QFT}|j\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{2\pi i \, jk / N} |k\rangle$$

This is a unitary transformation $U_{\text{QFT}}$ with matrix elements:

$$(U_{\text{QFT}})_{kj} = \frac{1}{\sqrt{N}} e^{2\pi i \, jk / N}$$

### 3.2 Product Representation

The QFT has a beautiful product representation that directly yields the circuit implementation. To derive it, write $j$ and $k$ in binary:

$$j = j_1 j_2 \cdots j_n = j_1 \cdot 2^{n-1} + j_2 \cdot 2^{n-2} + \cdots + j_n \cdot 2^0$$

Using the binary fraction notation $0.j_l j_{l+1} \cdots j_m = j_l/2 + j_{l+1}/4 + \cdots + j_m/2^{m-l+1}$, we can write:

$$\text{QFT}|j_1 j_2 \cdots j_n\rangle = \frac{1}{\sqrt{2^n}} \bigotimes_{l=1}^{n} \left( |0\rangle + e^{2\pi i \, 0.j_{n-l+1} \cdots j_n} |1\rangle \right)$$

This factored form shows that each qubit of the output depends on only some of the input bits, and the output is a *tensor product* — no entanglement! This product structure is what makes the QFT circuit efficient.

### 3.3 Explicit Example: 2-Qubit QFT

For $n = 2$, $N = 4$, and $\omega = e^{2\pi i/4} = i$:

$$U_{\text{QFT}}^{(2)} = \frac{1}{2} \begin{pmatrix} 1 & 1 & 1 & 1 \\ 1 & i & -1 & -i \\ 1 & -1 & 1 & -1 \\ 1 & -i & -1 & i \end{pmatrix}$$

Let us verify on the state $|2\rangle = |10\rangle$:

$$\text{QFT}|2\rangle = \frac{1}{2}(|0\rangle + e^{2\pi i \cdot 2/4}|1\rangle + e^{2\pi i \cdot 4/4}|2\rangle + e^{2\pi i \cdot 6/4}|3\rangle) = \frac{1}{2}(|0\rangle - |1\rangle + |2\rangle - |3\rangle)$$

### 3.4 Explicit Example: 3-Qubit QFT

For $n = 3$, $N = 8$, and $\omega = e^{2\pi i/8} = e^{i\pi/4}$:

$$U_{\text{QFT}}^{(3)} = \frac{1}{\sqrt{8}} \begin{pmatrix} 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & \omega & \omega^2 & \omega^3 & \omega^4 & \omega^5 & \omega^6 & \omega^7 \\ 1 & \omega^2 & \omega^4 & \omega^6 & 1 & \omega^2 & \omega^4 & \omega^6 \\ \vdots & & & & & & & \vdots \\ 1 & \omega^7 & \omega^6 & \omega^5 & \omega^4 & \omega^3 & \omega^2 & \omega \end{pmatrix}$$

The matrix has a highly structured pattern: row $k$, column $j$ contains $\omega^{jk}$. This structure is what allows the factorization into $O(n^2)$ gates.

### 3.5 Python: QFT Matrix Construction

```python
import numpy as np

def qft_matrix(n):
    """Construct the QFT matrix for n qubits.

    Why build the full matrix? For small n, the full matrix lets us verify
    our circuit implementation against the mathematical definition. For
    large n, only the circuit approach is feasible (the matrix is 2^n × 2^n).
    """
    N = 2**n
    omega = np.exp(2j * np.pi / N)

    # Build using outer product of indices: (F_N)_{kj} = omega^{jk} / sqrt(N)
    indices = np.arange(N)
    # indices[:, None] * indices[None, :] creates the jk product matrix
    return np.array([[omega**(j*k) for j in range(N)] for k in range(N)]) / np.sqrt(N)

# Verify unitarity: F†F should equal I
n = 3
F = qft_matrix(n)
print("Unitarity check (should be ~identity):")
print(np.round(F.conj().T @ F, 10).real)

# Apply QFT to |5⟩ = |101⟩ for 3 qubits
state = np.zeros(8, dtype=complex)
state[5] = 1.0  # |101⟩
result = F @ state
print("\nQFT|5⟩ amplitudes:")
for k in range(8):
    print(f"  |{k:03b}⟩: {result[k]:.4f}  (magnitude: {abs(result[k]):.4f})")
```

---

## 4. The QFT Circuit

### 4.1 Building Blocks

The QFT circuit uses two types of gates:

**Hadamard gate** $H$:

$$H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

which maps $|0\rangle \to \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ and $|1\rangle \to \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$.

**Controlled rotation gate** $CR_k$ (controlled-$R_k$):

$$R_k = \begin{pmatrix} 1 & 0 \\ 0 & e^{2\pi i / 2^k} \end{pmatrix}$$

The controlled version $CR_k$ applies $R_k$ to the target qubit only when the control qubit is $|1\rangle$. Note that $R_1 = Z$ (the Pauli-Z gate) and $R_2 = S$ (the phase gate).

### 4.2 Circuit Construction

The QFT circuit for $n$ qubits proceeds as follows:

**Step 1**: Apply $H$ to qubit 1, then $CR_2$ (controlled by qubit 2, target qubit 1), then $CR_3$ (controlled by qubit 3, target qubit 1), ..., then $CR_n$ (controlled by qubit $n$, target qubit 1).

After this step, qubit 1 is in the state:

$$\frac{1}{\sqrt{2}} \left( |0\rangle + e^{2\pi i \, 0.j_1 j_2 \cdots j_n} |1\rangle \right)$$

**Step 2**: Apply $H$ to qubit 2, then $CR_2$ (controlled by qubit 3), ..., $CR_{n-1}$ (controlled by qubit $n$).

**Steps 3 through n**: Continue the pattern, with each step applying $H$ and progressively fewer controlled rotations.

**Final step**: Apply SWAP gates to reverse the qubit order (the product representation has the qubits in reverse order compared to the standard convention).

### 4.3 Circuit Diagram (3-qubit QFT)

```
q₁ ─── H ─── CR₂ ─── CR₃ ─── × ───────────────────
              │        │       │
q₂ ─────────●──── H ─ CR₂ ── │ ── × ───────────────
                       │       │    │
q₃ ──────────────●────●────── × ── × ── H ──────────
```

where `●` denotes control qubits and `×` denotes SWAP operations.

### 4.4 Why This Circuit Works

Let us trace through the circuit for $|j\rangle = |j_1 j_2 j_3\rangle$ step by step.

**After H on qubit 1**: Qubit 1 becomes $\frac{1}{\sqrt{2}}(|0\rangle + e^{2\pi i \cdot 0.j_1}|1\rangle)$.
- If $j_1 = 0$: phase is $e^0 = 1$, so we get $(|0\rangle + |1\rangle)/\sqrt{2}$
- If $j_1 = 1$: phase is $e^{i\pi} = -1$, so we get $(|0\rangle - |1\rangle)/\sqrt{2}$

**After $CR_2$ (controlled by qubit 2)**: If $j_2 = 1$, qubit 1 picks up an additional phase $e^{2\pi i / 4}$, giving $\frac{1}{\sqrt{2}}(|0\rangle + e^{2\pi i \cdot 0.j_1 j_2}|1\rangle)$.

**After $CR_3$ (controlled by qubit 3)**: If $j_3 = 1$, another phase is added, giving $\frac{1}{\sqrt{2}}(|0\rangle + e^{2\pi i \cdot 0.j_1 j_2 j_3}|1\rangle)$.

This matches the first factor in the product representation. The same logic applies to qubits 2 and 3 in subsequent steps. The SWAP at the end reverses the qubit ordering to match the standard convention.

### 4.5 Python: QFT Circuit Simulation

```python
import numpy as np

def controlled_rotation(n_qubits, control, target, k):
    """Build a controlled-R_k gate for an n-qubit system.

    Why build the full matrix? For small systems (n ≤ 6), the full matrix
    approach is conceptually clear and easy to verify. For larger systems,
    one would use sparse representations or gate-by-gate state updates.
    """
    N = 2**n_qubits
    gate = np.eye(N, dtype=complex)
    phase = np.exp(2j * np.pi / 2**k)

    # Apply phase only when both control and target are |1⟩
    for state in range(N):
        # Check if control bit is 1 AND target bit is 1
        if (state >> (n_qubits - 1 - control)) & 1 and \
           (state >> (n_qubits - 1 - target)) & 1:
            gate[state, state] = phase
    return gate

def hadamard_on_qubit(n_qubits, qubit):
    """Build Hadamard gate acting on specified qubit in n-qubit system."""
    # Construct using tensor products: I ⊗ ... ⊗ H ⊗ ... ⊗ I
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    I = np.eye(2)

    gate = np.array([[1]])
    for q in range(n_qubits):
        gate = np.kron(gate, H if q == qubit else I)
    return gate

def swap_gate(n_qubits, q1, q2):
    """Build SWAP gate between two qubits."""
    N = 2**n_qubits
    gate = np.zeros((N, N), dtype=complex)
    for state in range(N):
        bit1 = (state >> (n_qubits - 1 - q1)) & 1
        bit2 = (state >> (n_qubits - 1 - q2)) & 1
        if bit1 != bit2:
            # Swap the two bits
            new_state = state ^ (1 << (n_qubits - 1 - q1)) ^ (1 << (n_qubits - 1 - q2))
            gate[new_state, state] = 1
        else:
            gate[state, state] = 1
    return gate

def qft_circuit(n):
    """Build the QFT as a product of gates (circuit simulation).

    This constructs the QFT matrix by multiplying individual gates in
    the order they appear in the circuit. The result should match
    qft_matrix(n) from the direct definition.
    """
    N = 2**n
    circuit = np.eye(N, dtype=complex)

    # Main QFT gates
    for target in range(n):
        # Hadamard on target qubit
        circuit = hadamard_on_qubit(n, target) @ circuit

        # Controlled rotations: CR_2, CR_3, ..., CR_{n-target}
        for control in range(target + 1, n):
            k = control - target + 1  # R_k rotation angle
            circuit = controlled_rotation(n, control, target, k) @ circuit

    # SWAP gates to reverse qubit order
    for i in range(n // 2):
        circuit = swap_gate(n, i, n - 1 - i) @ circuit

    return circuit

# Verify: circuit QFT should match the matrix definition
n = 3
F_matrix = qft_matrix(n)  # From the earlier function
F_circuit = qft_circuit(n)

print("Max difference between matrix and circuit QFT:",
      np.max(np.abs(F_matrix - F_circuit)))

# Test on a specific state: |5⟩ = |101⟩
state = np.zeros(8, dtype=complex)
state[5] = 1.0
result = F_circuit @ state
print("\nQFT|5⟩ via circuit:")
for k in range(8):
    mag = abs(result[k])
    phase = np.angle(result[k]) / np.pi
    if mag > 1e-10:
        print(f"  |{k:03b}⟩: magnitude={mag:.4f}, phase={phase:.4f}π")
```

---

## 5. Gate Complexity Analysis

### 5.1 Counting Gates

The QFT circuit for $n$ qubits contains:

- **Hadamard gates**: $n$ total (one per qubit)
- **Controlled rotation gates**: $(n-1) + (n-2) + \cdots + 1 + 0 = \frac{n(n-1)}{2}$ total
- **SWAP gates**: $\lfloor n/2 \rfloor$ total (each decomposable into 3 CNOTs)

**Total gate count**: $n + \frac{n(n-1)}{2} + \lfloor n/2 \rfloor = O(n^2)$

### 5.2 Comparison with Classical

| Method | Input size | Operations |
|--------|-----------|-----------|
| Classical DFT (direct) | $N = 2^n$ numbers | $O(N^2) = O(4^n)$ |
| Classical FFT | $N = 2^n$ numbers | $O(N \log N) = O(n \cdot 2^n)$ |
| Quantum QFT | $n$ qubits ($2^n$ amplitudes) | $O(n^2)$ gates |

The QFT achieves an *exponential* speedup over the classical FFT. However, this comparison is subtle:

- The classical FFT transforms $N$ explicitly given numbers
- The QFT transforms $N$ amplitudes that are *implicitly* encoded in $n$ qubits
- We cannot efficiently load arbitrary classical data into a quantum state (this is the "state preparation" bottleneck)
- We cannot read out all $N$ amplitudes (measurement gives one sample)

The QFT's power is realized when it is used as a *subroutine* within larger quantum algorithms where the input state is prepared by quantum operations (not classical data loading).

### 5.3 Approximate QFT

For practical implementations, distant controlled rotations $CR_k$ with $k \gg 1$ apply tiny phases $e^{2\pi i / 2^k} \approx 1$. We can *truncate* the circuit by omitting all $CR_k$ with $k > m$ for some cutoff $m$, yielding an **approximate QFT** with:

- Gate count: $O(nm)$ instead of $O(n^2)$
- Error: $O(n \cdot 2^{-m})$ in operator norm

Setting $m = O(\log n)$ gives error $O(n^{-c})$ with only $O(n \log n)$ gates — matching the FFT scaling!

---

## 6. Inverse QFT

### 6.1 Definition

Since the QFT is unitary, its inverse is simply its conjugate transpose:

$$\text{QFT}^{-1} = \text{QFT}^{\dagger}$$

Applied to a state:

$$\text{QFT}^{-1}|k\rangle = \frac{1}{\sqrt{N}} \sum_{j=0}^{N-1} e^{-2\pi i \, jk / N} |j\rangle$$

The only difference from the forward QFT is the sign of the exponent: $e^{-2\pi i jk/N}$ instead of $e^{+2\pi i jk/N}$.

### 6.2 Inverse QFT Circuit

To obtain the inverse QFT circuit, we:

1. Reverse the order of all gates in the QFT circuit
2. Replace each $R_k$ with $R_k^{\dagger}$ (i.e., negate the phase: $e^{2\pi i/2^k} \to e^{-2\pi i/2^k}$)

This is a general property of quantum circuits: to invert a unitary built from gates $U_1 U_2 \cdots U_m$, apply $U_m^{\dagger} \cdots U_2^{\dagger} U_1^{\dagger}$.

### 6.3 When Do We Need the Inverse QFT?

The inverse QFT appears in:

- **Quantum Phase Estimation**: The inverse QFT extracts the phase from the register after controlled-$U$ operations (see Section 7)
- **Shor's algorithm**: Uses inverse QFT to read out the period (see Lesson 10)
- **Quantum simulation**: Converting from the momentum basis back to position basis

```python
import numpy as np

def inverse_qft_matrix(n):
    """Construct the inverse QFT matrix.

    Why conjugate transpose? The QFT is unitary, so its inverse equals
    its conjugate transpose. This is the quantum analog of the inverse DFT.
    """
    return qft_matrix(n).conj().T

# Verify: QFT followed by inverse QFT should be identity
n = 3
F = qft_matrix(n)
F_inv = inverse_qft_matrix(n)
print("QFT · QFT⁻¹ ≈ I:", np.allclose(F @ F_inv, np.eye(2**n)))

# Round-trip test: start with |3⟩, apply QFT, then inverse QFT
state = np.zeros(8, dtype=complex)
state[3] = 1.0
recovered = F_inv @ (F @ state)
print("Round-trip recovery of |3⟩:", np.argmax(np.abs(recovered)))
```

---

## 7. Quantum Phase Estimation

### 7.1 The Problem

Quantum Phase Estimation (QPE) is one of the most important applications of the QFT. Given:

- A unitary operator $U$
- An eigenstate $|\psi\rangle$ such that $U|\psi\rangle = e^{2\pi i \theta}|\psi\rangle$

QPE estimates the phase $\theta \in [0, 1)$ to $n$ bits of precision using $n$ ancilla qubits.

### 7.2 The QPE Circuit

The QPE algorithm uses two registers:

- **Counting register**: $n$ qubits initialized to $|0\rangle^{\otimes n}$
- **Eigenstate register**: initialized to $|\psi\rangle$

**Step 1**: Apply $H^{\otimes n}$ to the counting register, creating an equal superposition.

**Step 2**: Apply controlled-$U^{2^j}$ operations. The $j$-th counting qubit controls $U^{2^j}$ acting on $|\psi\rangle$.

Since $U|\psi\rangle = e^{2\pi i \theta}|\psi\rangle$, we have $U^{2^j}|\psi\rangle = e^{2\pi i \cdot 2^j \theta}|\psi\rangle$. After all controlled-$U^{2^j}$ operations, the state becomes:

$$\frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n - 1} e^{2\pi i k \theta} |k\rangle \otimes |\psi\rangle$$

**Step 3**: Apply the **inverse QFT** to the counting register.

If $\theta = m/2^n$ for some integer $m$ (the phase is exactly representable in $n$ bits), then the inverse QFT maps the counting register to exactly $|m\rangle$. Measuring the counting register yields $m$, from which we compute $\theta = m/2^n$.

If $\theta$ is not exactly representable, the measurement yields the closest $n$-bit approximation with high probability. Adding more ancilla qubits increases precision.

### 7.3 Circuit Diagram

```
|0⟩ ─── H ─── ────────── ────── ────── ┌──────────┐
|0⟩ ─── H ─── ────────── ────── U²    │          │
|0⟩ ─── H ─── ────────── U⁴   ────── │ QFT⁻¹   │ ──── Measure
|0⟩ ─── H ─── U⁸        ────── ────── │          │
                                        └──────────┘
|ψ⟩ ─── ───── target ── target  target ───────────────
```

### 7.4 QPE Accuracy

For an $n$-qubit counting register:

- If $\theta$ is exactly representable in $n$ bits: QPE succeeds with probability 1
- Otherwise: the probability of getting the best $n$-bit approximation $\tilde{\theta}$ satisfying $|\theta - \tilde{\theta}| \leq 2^{-n}$ is at least $4/\pi^2 \approx 0.405$

Adding $O(\log(1/\epsilon))$ extra qubits boosts the success probability to $1 - \epsilon$.

### 7.5 Python: Phase Estimation Simulation

```python
import numpy as np

def phase_estimation(unitary, eigenstate, n_counting):
    """Simulate Quantum Phase Estimation.

    Why simulate the full algorithm? This demonstrates the mathematical
    structure: controlled-U operations create phase-encoded superpositions,
    and the inverse QFT extracts the phase as a binary fraction.

    Args:
        unitary: d×d unitary matrix
        eigenstate: d-dimensional eigenvector of unitary
        n_counting: number of counting qubits (precision bits)

    Returns:
        probabilities: probability distribution over counting register states
        estimated_phase: most likely phase estimate
    """
    d = len(eigenstate)
    N = 2**n_counting

    # Step 1: Create equal superposition in counting register
    # After H⊗n, counting register is (1/√N) Σ|k⟩
    # After controlled-U operations, state is:
    # (1/√N) Σ_k e^{2πi·k·θ} |k⟩ ⊗ |ψ⟩

    # Compute the eigenvalue phase
    # U|ψ⟩ = e^{2πiθ}|ψ⟩, so θ = angle(⟨ψ|U|ψ⟩) / (2π)
    eigenvalue = np.dot(eigenstate.conj(), unitary @ eigenstate)
    theta = np.angle(eigenvalue) / (2 * np.pi)
    if theta < 0:
        theta += 1  # Ensure θ ∈ [0, 1)

    # Step 2: The counting register state before inverse QFT
    counting_state = np.array([np.exp(2j * np.pi * k * theta) for k in range(N)]) / np.sqrt(N)

    # Step 3: Apply inverse QFT to counting register
    F_inv = np.conj(qft_matrix(n_counting)).T
    result = F_inv @ counting_state

    # Measurement probabilities
    probs = np.abs(result)**2

    best_k = np.argmax(probs)
    estimated_phase = best_k / N

    return probs, estimated_phase, theta

# Example: Estimate the phase of a rotation gate R_z(2π·0.375)
# The eigenvalue should be e^{2πi·0.375}, so θ = 0.375 = 3/8
theta_true = 0.375
Rz = np.array([[1, 0], [0, np.exp(2j * np.pi * theta_true)]])
eigenstate = np.array([0, 1], dtype=complex)  # |1⟩ is an eigenstate of Rz

n_counting = 4
probs, estimated, true_theta = phase_estimation(Rz, eigenstate, n_counting)

print(f"True phase: θ = {true_theta:.6f}")
print(f"Estimated phase: θ = {estimated:.6f}")
print(f"Counting register probabilities:")
for k in range(2**n_counting):
    if probs[k] > 0.01:
        print(f"  |{k:0{n_counting}b}⟩ = |{k}⟩: {probs[k]:.4f}  → θ = {k/2**n_counting:.4f}")
```

---

## 8. Python Implementation

### 8.1 Complete QFT with Visualization

```python
import numpy as np

def qft_matrix(n):
    """Build the full QFT matrix for n qubits."""
    N = 2**n
    omega = np.exp(2j * np.pi / N)
    return np.array([[omega**(j*k) for j in range(N)] for k in range(N)]) / np.sqrt(N)

def demonstrate_periodicity_detection():
    """Show how QFT detects hidden periodicity.

    This is the core reason QFT matters: given a quantum state with
    amplitudes that repeat with some period r, the QFT concentrates
    the probability at multiples of N/r. This is the foundation of
    Shor's algorithm.
    """
    n = 4  # 4 qubits → 16-dimensional space
    N = 2**n

    print("=" * 60)
    print("QFT Periodicity Detection Demo")
    print("=" * 60)

    for period in [2, 4, 8]:
        # Create a state with the given period: nonzero at 0, r, 2r, ...
        state = np.zeros(N, dtype=complex)
        num_peaks = N // period
        for i in range(num_peaks):
            state[i * period] = 1.0
        state /= np.linalg.norm(state)  # Normalize

        # Apply QFT
        F = qft_matrix(n)
        result = F @ state
        probs = np.abs(result)**2

        print(f"\nPeriod r = {period}:")
        print(f"  Input state nonzero at: {[i*period for i in range(num_peaks)]}")
        print(f"  QFT peaks at (prob > 0.01):")
        for k in range(N):
            if probs[k] > 0.01:
                print(f"    k = {k} (N/r = {N}/{period} = {N//period}, "
                      f"k is {'a' if k % (N//period) == 0 else 'NOT a'} multiple)")

def demonstrate_phase_estimation_accuracy():
    """Show how QPE accuracy improves with more counting qubits.

    Key insight: adding one counting qubit doubles the precision of
    our phase estimate. This is why QPE can achieve exponential precision
    with only linear resources.
    """
    theta_true = 1/3  # θ = 1/3, not exactly representable in binary
    U = np.array([[1, 0], [0, np.exp(2j * np.pi * theta_true)]])
    eigenstate = np.array([0, 1], dtype=complex)

    print("\n" + "=" * 60)
    print(f"QPE Accuracy vs Counting Qubits (θ = 1/3 ≈ 0.3333...)")
    print("=" * 60)

    for n_count in range(3, 9):
        N = 2**n_count
        # Simulate QPE
        counting = np.array([np.exp(2j * np.pi * k * theta_true)
                            for k in range(N)]) / np.sqrt(N)
        F_inv = np.conj(qft_matrix(n_count)).T
        result = F_inv @ counting
        probs = np.abs(result)**2

        best_k = np.argmax(probs)
        estimated = best_k / N
        error = abs(estimated - theta_true)

        print(f"  n = {n_count}: best estimate = {estimated:.6f}, "
              f"error = {error:.6f}, prob = {probs[best_k]:.4f}")

# Run demonstrations
demonstrate_periodicity_detection()
demonstrate_phase_estimation_accuracy()
```

### 8.2 QFT Circuit Gate-by-Gate

```python
import numpy as np

def qft_step_by_step(state_vector, n):
    """Apply QFT gate by gate, printing intermediate states.

    Why trace through step by step? Understanding the QFT circuit at the
    gate level builds intuition for how the product representation arises.
    Each gate adds one more bit to the binary fraction in the phase.
    """
    state = state_vector.copy()
    N = 2**n

    def apply_single_qubit_gate(state, gate, target, n):
        """Apply a single-qubit gate to the target qubit."""
        I = np.eye(2)
        full_gate = np.array([[1]])
        for q in range(n):
            full_gate = np.kron(full_gate, gate if q == target else I)
        return full_gate @ state

    def apply_controlled_phase(state, control, target, phase, n):
        """Apply controlled phase gate."""
        N = 2**n
        new_state = state.copy()
        for s in range(N):
            c_bit = (s >> (n - 1 - control)) & 1
            t_bit = (s >> (n - 1 - target)) & 1
            if c_bit == 1 and t_bit == 1:
                new_state[s] *= phase
        return new_state

    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    print(f"Initial state: {np.array2string(state, precision=3)}")

    for target in range(n):
        # Hadamard
        state = apply_single_qubit_gate(state, H, target, n)
        print(f"After H on q{target}: ", end="")
        # Show nonzero amplitudes
        nonzero = [(k, state[k]) for k in range(N) if abs(state[k]) > 1e-10]
        print(", ".join(f"|{k:0{n}b}⟩:{v:.3f}" for k, v in nonzero[:4]),
              "..." if len(nonzero) > 4 else "")

        # Controlled rotations
        for control in range(target + 1, n):
            k = control - target + 1
            phase = np.exp(2j * np.pi / 2**k)
            state = apply_controlled_phase(state, control, target, phase, n)
            print(f"After CR_{k}(q{control}→q{target}): applied phase 2π/2^{k}")

    # Swap qubits
    for i in range(n // 2):
        j = n - 1 - i
        # Swap qubits i and j
        new_state = np.zeros_like(state)
        for s in range(N):
            bi = (s >> (n-1-i)) & 1
            bj = (s >> (n-1-j)) & 1
            if bi != bj:
                ns = s ^ (1 << (n-1-i)) ^ (1 << (n-1-j))
            else:
                ns = s
            new_state[ns] = state[s]
        state = new_state
        print(f"After SWAP(q{i},q{j})")

    return state

# Trace QFT|6⟩ = QFT|110⟩ for 3 qubits
n = 3
state = np.zeros(2**n, dtype=complex)
state[6] = 1.0  # |110⟩
print("=" * 60)
print("Step-by-step QFT on |110⟩ (3 qubits)")
print("=" * 60)
result = qft_step_by_step(state, n)
```

---

## 9. Exercises

### Exercise 1: QFT by Hand (2-qubit)

Compute $\text{QFT}|3\rangle$ for 2-qubit QFT by:
(a) Directly applying the QFT matrix $F_4$ to the vector $(0, 0, 0, 1)^T$
(b) Using the product representation: write $|3\rangle = |11\rangle$ and expand $\frac{1}{2}(|0\rangle + e^{2\pi i \cdot 0.j_2}|1\rangle) \otimes (|0\rangle + e^{2\pi i \cdot 0.j_1 j_2}|1\rangle)$
(c) Verify that both methods agree.

### Exercise 2: Period Finding with QFT

Suppose a quantum oracle prepares the state $|\psi\rangle = \frac{1}{2}(|0\rangle + |3\rangle + |6\rangle + |9\rangle)$ in a 4-qubit register (period $r = 3$, $N = 16$).

(a) Compute $\text{QFT}|\psi\rangle$ explicitly (or use the Python simulation).
(b) What are the measurement probabilities? Where are the peaks?
(c) How would you extract the period $r = 3$ from the measurement outcomes?
(d) Why is this more subtle than the $r = 2, 4, 8$ cases? (Hint: $N/r$ is not an integer.)

### Exercise 3: QPE Precision

Consider QPE applied to the gate $R_z(\phi)$ with $\phi = 2\pi \cdot 0.1101_2 = 2\pi \cdot (13/16)$ and eigenstate $|1\rangle$.

(a) What is the minimum number of counting qubits needed to estimate $\theta = 13/16$ exactly?
(b) Simulate QPE with 4, 5, and 6 counting qubits. What happens to the probability distribution?
(c) Now change $\theta = 0.3$ (not exactly representable in binary). Simulate with $n = 4, 6, 8, 10$ counting qubits and plot the error vs $n$.

### Exercise 4: Approximate QFT

Implement the approximate QFT by modifying the circuit to omit all $CR_k$ gates with $k > m$ for a cutoff parameter $m$.

(a) For $n = 6$ qubits, compute the operator norm error $\|U_{\text{QFT}} - U_{\text{approx}}\|$ for $m = 2, 3, 4, 5$.
(b) How does the gate count scale with $m$?
(c) At what $m$ does the error become negligible (say, $< 10^{-4}$)?

### Exercise 5: Inverse QFT Application

Given the state $|\phi\rangle = \frac{1}{\sqrt{8}} \sum_{k=0}^{7} e^{2\pi i \cdot 5k/8} |k\rangle$ (a Fourier mode with frequency 5):

(a) Apply the inverse QFT. What state do you get?
(b) Verify using both the matrix method and the circuit method.
(c) Generalize: what is $\text{QFT}^{-1} \left(\frac{1}{\sqrt{N}} \sum_k e^{2\pi i mk/N} |k\rangle\right)$ for arbitrary integer $m$?

---

[← Previous: Grover's Search Algorithm](08_Grovers_Search.md) | [Next: Shor's Factoring Algorithm →](10_Shors_Algorithm.md)
