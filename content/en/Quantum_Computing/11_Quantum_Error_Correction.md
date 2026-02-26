# Lesson 11: Quantum Error Correction

[← Previous: Shor's Factoring Algorithm](10_Shors_Algorithm.md) | [Next: Quantum Teleportation and Communication →](12_Quantum_Teleportation.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why quantum error correction is fundamentally different from classical error correction
2. Describe the no-cloning theorem and its implications for error correction
3. Construct the 3-qubit bit-flip code and phase-flip code
4. Explain Shor's 9-qubit code that corrects arbitrary single-qubit errors
5. Use the stabilizer formalism to describe quantum error-correcting codes
6. Outline CSS (Calderbank-Shor-Steane) codes and surface codes
7. State the fault-tolerant threshold theorem and its significance

---

Every computation makes errors. Classical computers handle this with redundancy — store three copies of each bit and take a majority vote. Quantum computers face a far more daunting challenge: the no-cloning theorem forbids copying quantum states, continuous errors (rotations by tiny angles) seem to require infinite precision to correct, and measurement destroys the very information we want to protect. For decades, many physicists believed quantum error correction was impossible.

The discovery that quantum error correction *is* possible — by Shor (1995) and Steane (1996) — was one of the most surprising results in quantum information theory. It showed that despite all these obstacles, clever encoding can protect quantum information from noise without ever learning what that information is. This breakthrough underpins the entire enterprise of fault-tolerant quantum computing: without it, no quantum algorithm could run long enough to be useful.

> **Analogy:** Quantum error correction is like encoding a message in a choir. If one singer goes off-key, the others can detect and correct the error without ever learning the original note. The choir members never know the melody being sung — they only know the *relationships* between their notes (the harmonies). By checking these relationships (syndromes), they can identify and fix any single singer's mistake while the song continues.

## Table of Contents

1. [Why QEC Is Different](#1-why-qec-is-different)
2. [The No-Cloning Theorem](#2-the-no-cloning-theorem)
3. [The 3-Qubit Bit-Flip Code](#3-the-3-qubit-bit-flip-code)
4. [The 3-Qubit Phase-Flip Code](#4-the-3-qubit-phase-flip-code)
5. [Shor's 9-Qubit Code](#5-shors-9-qubit-code)
6. [Stabilizer Formalism](#6-stabilizer-formalism)
7. [CSS Codes](#7-css-codes)
8. [Surface Codes](#8-surface-codes)
9. [Fault-Tolerant Threshold Theorem](#9-fault-tolerant-threshold-theorem)
10. [Python Implementation](#10-python-implementation)
11. [Exercises](#11-exercises)

---

## 1. Why QEC Is Different

### 1.1 Classical Error Correction Review

Classical error correction is straightforward: to protect a bit $b \in \{0, 1\}$, encode it as three copies:

$$0 \to 000, \quad 1 \to 111$$

If one bit flips (e.g., $000 \to 010$), majority voting recovers the original. This **repetition code** can correct any single bit-flip.

### 1.2 Three Quantum Obstacles

Quantum error correction faces three challenges that have no classical analog:

**Obstacle 1: No-Cloning Theorem**

We cannot copy an unknown quantum state $|\psi\rangle$. So the naive approach of encoding $|\psi\rangle$ as $|\psi\rangle|\psi\rangle|\psi\rangle$ is impossible. We need a fundamentally different notion of redundancy.

**Obstacle 2: Continuous Errors**

Classical bits can only flip ($0 \leftrightarrow 1$). But a qubit can undergo any unitary rotation — a continuous family of errors. A small rotation $R_x(\epsilon)$ moves $|0\rangle$ slightly toward $|1\rangle$:

$$R_x(\epsilon)|0\rangle = \cos(\epsilon/2)|0\rangle - i\sin(\epsilon/2)|1\rangle$$

How can we correct a continuous range of errors with a discrete code?

**Obstacle 3: Measurement Destroys Information**

In classical error correction, we can freely inspect the encoded bits. Measuring a quantum state collapses it, potentially destroying the information we want to protect.

### 1.3 The Solutions

Remarkably, all three obstacles have elegant solutions:

1. **No cloning**: Instead of copying $|\psi\rangle$, we *entangle* it across multiple qubits. The information is encoded in correlations, not in individual qubits.

2. **Continuous errors**: Any single-qubit error can be written as a linear combination of the identity $I$ and three Pauli matrices $X$, $Y$, $Z$. Correcting these four discrete errors automatically corrects all continuous errors (by linearity of quantum mechanics).

3. **Measurement**: We perform *syndrome measurements* that detect errors without revealing the encoded information. These measurements project the error onto one of the discrete Pauli errors, which we then correct.

---

## 2. The No-Cloning Theorem

### 2.1 Statement

**No-Cloning Theorem**: There is no unitary operation $U$ that can clone an arbitrary quantum state:

$$U|\psi\rangle|0\rangle = |\psi\rangle|\psi\rangle \quad \text{for all } |\psi\rangle$$

### 2.2 Proof Sketch

Suppose such a $U$ exists. Then for two states $|\psi\rangle$ and $|\phi\rangle$:

$$U|\psi\rangle|0\rangle = |\psi\rangle|\psi\rangle$$
$$U|\phi\rangle|0\rangle = |\phi\rangle|\phi\rangle$$

Taking the inner product of both sides:

$$\langle\psi|\phi\rangle \cdot \langle 0|0\rangle = \langle\psi|\phi\rangle^2$$
$$\langle\psi|\phi\rangle = \langle\psi|\phi\rangle^2$$

This is only satisfied when $\langle\psi|\phi\rangle = 0$ or $\langle\psi|\phi\rangle = 1$. So $U$ can only clone states that are identical or orthogonal — not arbitrary states. Contradiction.

### 2.3 Implications for Error Correction

Since we cannot clone, we cannot use the classical repetition strategy. Instead, quantum codes encode a single logical qubit into the *entangled state* of multiple physical qubits. The encoding maps:

$$\alpha|0\rangle + \beta|1\rangle \to \alpha|0_L\rangle + \beta|1_L\rangle$$

where $|0_L\rangle$ and $|1_L\rangle$ are multi-qubit *codewords* and the coefficients $\alpha, \beta$ are preserved without being copied.

---

## 3. The 3-Qubit Bit-Flip Code

### 3.1 Encoding

The bit-flip code encodes one logical qubit into three physical qubits:

$$|0\rangle \to |0_L\rangle = |000\rangle$$
$$|1\rangle \to |1_L\rangle = |111\rangle$$

A general state $\alpha|0\rangle + \beta|1\rangle$ is encoded as:

$$|\psi_L\rangle = \alpha|000\rangle + \beta|111\rangle$$

Note: this is NOT three copies of $\alpha|0\rangle + \beta|1\rangle$ (which would be $(\alpha|0\rangle + \beta|1\rangle)^{\otimes 3}$). It is an *entangled* state — the information about $\alpha$ and $\beta$ is stored in the correlations between the three qubits.

### 3.2 Error Model

A bit-flip error on qubit $i$ applies the Pauli $X$ operator to that qubit:

$$X_1: |000\rangle \to |100\rangle, \quad |111\rangle \to |011\rangle$$
$$X_2: |000\rangle \to |010\rangle, \quad |111\rangle \to |101\rangle$$
$$X_3: |000\rangle \to |001\rangle, \quad |111\rangle \to |110\rangle$$

### 3.3 Syndrome Measurement

We detect errors without measuring the encoded information by measuring **parity checks**:

- $Z_1 Z_2$: measures whether qubits 1 and 2 have the same value (+1) or different values (-1)
- $Z_2 Z_3$: measures whether qubits 2 and 3 have the same value (+1) or different values (-1)

The **syndrome** is the pair of measurement outcomes $(s_1, s_2)$:

| Error | State (from $|000\rangle$) | $Z_1 Z_2$ | $Z_2 Z_3$ | Syndrome |
|-------|--------------------------|-----------|-----------|----------|
| None | $|000\rangle$ | +1 | +1 | (0, 0) |
| $X_1$ | $|100\rangle$ | -1 | +1 | (1, 0) |
| $X_2$ | $|010\rangle$ | -1 | -1 | (1, 1) |
| $X_3$ | $|001\rangle$ | +1 | -1 | (0, 1) |

Each error produces a unique syndrome, allowing unambiguous identification and correction. Crucially, these measurements do not reveal whether the state was $|000\rangle$ or $|111\rangle$ (or a superposition) — they only reveal *parity information*.

### 3.4 Why Syndrome Measurement Preserves Quantum Information

The key insight is that $Z_1 Z_2$ has eigenvalue $+1$ for both $|000\rangle$ and $|111\rangle$:

$$Z_1 Z_2 |000\rangle = (+1)(+1)|000\rangle = +|000\rangle$$
$$Z_1 Z_2 |111\rangle = (-1)(-1)|111\rangle = +|111\rangle$$

So the superposition $\alpha|000\rangle + \beta|111\rangle$ is an eigenstate of $Z_1 Z_2$ with eigenvalue $+1$. Measuring $Z_1 Z_2$ does not disturb this superposition — it only tells us whether an error has broken the parity relationship.

### 3.5 Limitation

The bit-flip code only corrects $X$ (bit-flip) errors. It cannot correct $Z$ (phase-flip) errors:

$$Z_1(\alpha|000\rangle + \beta|111\rangle) = \alpha|000\rangle - \beta|111\rangle$$

This phase error changes the relative phase between $|0_L\rangle$ and $|1_L\rangle$ and is undetectable by the $Z_1 Z_2$ and $Z_2 Z_3$ syndrome measurements.

---

## 4. The 3-Qubit Phase-Flip Code

### 4.1 The Phase-Flip Error

A phase-flip error applies the Pauli $Z$ operator: $Z|0\rangle = |0\rangle$, $Z|1\rangle = -|1\rangle$.

To correct phase flips, we encode in the Hadamard (X) basis:

$$|0\rangle \to |0_L\rangle = |{+}{+}{+}\rangle$$
$$|1\rangle \to |1_L\rangle = |{-}{-}{-}\rangle$$

where $|+\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$ and $|-\rangle = (|0\rangle - |1\rangle)/\sqrt{2}$.

### 4.2 How It Works

A phase flip on qubit $i$ maps $|+\rangle \to |-\rangle$ and vice versa. In the $\{|+\rangle, |-\rangle\}$ basis, a $Z$ error acts like a bit flip! So we can use the same syndrome measurement strategy, but in the X basis.

The syndrome measurements are $X_1 X_2$ and $X_2 X_3$, which check parity in the X basis.

### 4.3 Limitation

The phase-flip code corrects $Z$ errors but not $X$ errors — the complementary problem to the bit-flip code.

---

## 5. Shor's 9-Qubit Code

### 5.1 The Key Idea

Shor's brilliant insight was to combine both codes: first encode against phase flips (outer code), then encode each qubit of that code against bit flips (inner code). The result is a 9-qubit code that corrects any single-qubit error.

### 5.2 Encoding

**Step 1 (phase-flip encoding)**: Encode the logical qubit:

$$|0\rangle \to |{+}{+}{+}\rangle, \quad |1\rangle \to |{-}{-}{-}\rangle$$

**Step 2 (bit-flip encoding on each qubit)**: Encode each $|+\rangle$ and $|-\rangle$ using the 3-qubit repetition code in the Hadamard basis:

$$|+\rangle \to \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle) \equiv |+_L\rangle$$
$$|-\rangle \to \frac{1}{\sqrt{2}}(|000\rangle - |111\rangle) \equiv |-_L\rangle$$

**Complete encoding**:

$$|0\rangle \to |0_L\rangle = \frac{1}{2\sqrt{2}} (|000\rangle + |111\rangle)(|000\rangle + |111\rangle)(|000\rangle + |111\rangle)$$

$$|1\rangle \to |1_L\rangle = \frac{1}{2\sqrt{2}} (|000\rangle - |111\rangle)(|000\rangle - |111\rangle)(|000\rangle - |111\rangle)$$

### 5.3 Error Correction

**Bit-flip errors** ($X_i$ on any qubit $i$): Detected by parity checks within each block of 3 qubits, using $Z_i Z_j$ measurements. Same as the 3-qubit bit-flip code, applied independently to each block.

**Phase-flip errors** ($Z_i$ on any qubit $i$): A $Z$ error on any qubit within a block flips the sign of $|111\rangle$ in that block, effectively converting $|+_L\rangle \leftrightarrow |-_L\rangle$. This is detected by measuring $X_1 X_2 X_3 X_4 X_5 X_6$ and $X_4 X_5 X_6 X_7 X_8 X_9$ (comparing blocks pairwise).

**$Y$ errors**: Since $Y = iXZ$, a $Y$ error produces both a bit flip and a phase flip, both of which are detected and corrected by the syndrome measurements above.

### 5.4 Why Continuous Errors Are Handled

An arbitrary single-qubit error $E$ can be decomposed:

$$E = \alpha_I I + \alpha_X X + \alpha_Y Y + \alpha_Z Z$$

After syndrome measurement, the error is *projected* onto one of $\{I, X, Y, Z\}$. The syndrome tells us which Pauli error occurred, and we apply the corresponding correction. The linearity of quantum mechanics ensures this works even for continuous errors — the measurement "digitizes" the error.

### 5.5 Parameters

Shor's 9-qubit code: $[[9, 1, 3]]$

- **9 physical qubits** encode **1 logical qubit**
- **Distance 3**: can correct any single-qubit error (any error acting on $\leq 1$ qubit)
- **8 syndrome measurements** (2 per block of 3, plus 2 inter-block) uniquely identify the error

---

## 6. Stabilizer Formalism

### 6.1 Motivation

As quantum codes grow larger, describing them by their codewords becomes unwieldy. The **stabilizer formalism** provides a compact and powerful framework for describing quantum error-correcting codes.

### 6.2 The Pauli Group

The **$n$-qubit Pauli group** $\mathcal{P}_n$ consists of all $n$-fold tensor products of Pauli matrices $\{I, X, Y, Z\}$, multiplied by phases $\{\pm 1, \pm i\}$:

$$\mathcal{P}_n = \{\pm 1, \pm i\} \times \{I, X, Y, Z\}^{\otimes n}$$

For example, $X_1 Z_3 = X \otimes I \otimes Z$ is an element of $\mathcal{P}_3$.

### 6.3 Stabilizer Codes

A **stabilizer code** $C$ is defined by an abelian subgroup $\mathcal{S} \subset \mathcal{P}_n$ (the **stabilizer group**) such that:

$$C = \{|\psi\rangle : S|\psi\rangle = |\psi\rangle \text{ for all } S \in \mathcal{S}\}$$

The code space is the simultaneous $+1$ eigenspace of all stabilizer operators.

**Key property**: The stabilizer group is generated by $n - k$ independent generators, where $k$ is the number of logical qubits encoded. Measuring these generators yields the syndrome.

### 6.4 Bit-Flip Code in Stabilizer Language

For the 3-qubit bit-flip code:

- **Stabilizer generators**: $S_1 = Z_1 Z_2$, $S_2 = Z_2 Z_3$
- **Code space**: states with $Z_1 Z_2 = +1$ and $Z_2 Z_3 = +1$, i.e., $|000\rangle$ and $|111\rangle$
- **Logical operators**: $\bar{X} = X_1 X_2 X_3$ (logical bit flip), $\bar{Z} = Z_1$ (logical phase flip)

The code is $[[3, 1, 1]]$: 3 physical qubits, 1 logical qubit, distance 1 (cannot correct phase-flip errors).

### 6.5 Shor's Code in Stabilizer Language

The 9-qubit Shor code has 8 stabilizer generators:

| Generator | Qubits | Detects |
|-----------|--------|---------|
| $Z_1 Z_2$ | Block 1 | Bit flip in block 1 |
| $Z_2 Z_3$ | Block 1 | Bit flip in block 1 |
| $Z_4 Z_5$ | Block 2 | Bit flip in block 2 |
| $Z_5 Z_6$ | Block 2 | Bit flip in block 2 |
| $Z_7 Z_8$ | Block 3 | Bit flip in block 3 |
| $Z_8 Z_9$ | Block 3 | Bit flip in block 3 |
| $X_1 X_2 X_3 X_4 X_5 X_6$ | Blocks 1-2 | Phase flip between blocks |
| $X_4 X_5 X_6 X_7 X_8 X_9$ | Blocks 2-3 | Phase flip between blocks |

Parameters: $[[9, 1, 3]]$ — 9 physical, 1 logical, distance 3.

### 6.6 Error Detection with Stabilizers

An error $E$ is detected by stabilizer $S$ if $\{E, S\} = 0$ (they anti-commute):

$$SE|\psi\rangle = -ES|\psi\rangle = -E|\psi\rangle$$

So measuring $S$ yields $-1$ when error $E$ has occurred, flagging the error in the syndrome.

---

## 7. CSS Codes

### 7.1 Construction

**Calderbank-Shor-Steane (CSS) codes** are a family of quantum codes constructed from two classical linear codes $C_1$ and $C_2$ where $C_2 \subset C_1$:

- $C_1$: an $[n, k_1, d_1]$ classical code (corrects bit flips)
- $C_2 \subset C_1$: an $[n, k_2, d_2]$ classical code (corrects phase flips via dual code)

The resulting quantum code is $[[n, k_1 - k_2, \min(d_1, d_2^\perp)]]$.

### 7.2 The Steane Code

The most famous CSS code is the **Steane $[[7, 1, 3]]$ code**, constructed from the classical Hamming $[7, 4, 3]$ code. It encodes 1 logical qubit in 7 physical qubits with distance 3 (corrects any single-qubit error).

**Stabilizer generators**:

| Generator | Type |
|-----------|------|
| $X_1 X_3 X_5 X_7$ | X-type |
| $X_2 X_3 X_6 X_7$ | X-type |
| $X_4 X_5 X_6 X_7$ | X-type |
| $Z_1 Z_3 Z_5 Z_7$ | Z-type |
| $Z_2 Z_3 Z_6 Z_7$ | Z-type |
| $Z_4 Z_5 Z_6 Z_7$ | Z-type |

**Advantages over Shor's code**: The Steane code uses only 7 qubits (vs 9 for Shor) and has a symmetric structure that simplifies transversal gate implementation.

### 7.3 CSS Code Properties

- X-type stabilizers detect Z errors (and vice versa)
- Syndrome decoding reduces to classical decoding of $C_1$ and $C_2^\perp$
- Transversal CNOT is automatically fault-tolerant

---

## 8. Surface Codes

### 8.1 Why Surface Codes Matter

The **surface code** is widely considered the most promising error-correcting code for near-term quantum computers because:

1. **Local operations only**: All stabilizer measurements involve only nearest-neighbor qubits on a 2D lattice
2. **High threshold**: The error threshold is approximately 1% per gate — the highest of any known code
3. **Simple syndrome extraction**: Only weight-4 measurements (4 qubits per check)

### 8.2 Structure

A surface code on an $L \times L$ lattice uses:

- $n = 2L^2 - 2L + 1$ physical qubits (data qubits on edges)
- $L^2 - L$ X-type stabilizers (plaquette operators)
- $L^2 - L$ Z-type stabilizers (vertex operators)
- Encodes $k = 1$ logical qubit
- Code distance $d = L$

```
      Z──Z──Z
      │  │  │
   ───●──●──●───
      │  │  │
      X──X──X
      │  │  │
   ───●──●──●───
      │  │  │
      Z──Z──Z

  ● = data qubits
  Z = Z-stabilizer (vertex)
  X = X-stabilizer (plaquette)
```

### 8.3 Error Correction Procedure

1. Measure all stabilizers repeatedly (to handle measurement errors)
2. Identify defects: stabilizer outcomes that are $-1$
3. Use a **decoder** (e.g., minimum-weight perfect matching) to pair up defects
4. Apply corrections based on the pairing

### 8.4 Overhead

To correct errors at rate $p$ with the surface code:

- Required code distance: $d \sim O(\log(1/\epsilon))$ for target logical error rate $\epsilon$
- Physical qubits per logical qubit: $\sim 2d^2$
- For $p = 10^{-3}$ and $\epsilon = 10^{-10}$: $d \approx 17$, requiring $\sim 578$ physical qubits per logical qubit

---

## 9. Fault-Tolerant Threshold Theorem

### 9.1 The Problem

Even with error correction, the correction circuits themselves can introduce errors! A faulty CNOT gate used in syndrome measurement can spread errors from one qubit to another. This creates a seemingly circular problem: we need error correction to run quantum circuits, but error correction circuits themselves have errors.

### 9.2 The Threshold Theorem

**Theorem** (Aharonov & Ben-Or, 1997; Knill, Laflamme & Zurek, 1998): If the error rate per gate is below a threshold $p_{\text{th}}$, then arbitrary long quantum computations can be performed with arbitrarily small logical error rate.

More precisely, a quantum circuit of size $L$ can be simulated with logical error rate $\epsilon$ using:

$$O\left(L \cdot \text{polylog}(L/\epsilon)\right)$$

physical gates, provided $p < p_{\text{th}}$.

### 9.3 Threshold Values

| Code Family | Threshold $p_{\text{th}}$ | Notes |
|-------------|-------------------------|-------|
| Concatenated codes | $\sim 10^{-5}$ to $10^{-4}$ | Original proofs |
| Surface codes | $\sim 10^{-2}$ (1%) | Most practical |
| Color codes | $\sim 3 \times 10^{-3}$ | Transversal gates |

Current state-of-the-art qubit error rates: $\sim 10^{-3}$ to $10^{-2}$, which is right at the surface code threshold. Achieving consistently sub-threshold error rates is one of the main engineering challenges in quantum computing.

### 9.4 Implications

The threshold theorem is an *existence proof* for fault-tolerant quantum computing. It says that in principle, quantum computers can be made reliable. The practical question is whether the overhead (thousands of physical qubits per logical qubit) can be reduced enough to be technologically feasible. This remains the central challenge of the field.

---

## 10. Python Implementation

### 10.1 3-Qubit Bit-Flip Code

```python
import numpy as np

def encode_bit_flip(alpha, beta):
    """Encode a logical qubit |ψ⟩ = α|0⟩ + β|1⟩ into the 3-qubit bit-flip code.

    Why this encoding? The bit-flip code maps |0⟩→|000⟩ and |1⟩→|111⟩.
    By linearity, α|0⟩+β|1⟩ → α|000⟩+β|111⟩. This is an entangled state,
    NOT three copies of |ψ⟩ — consistent with the no-cloning theorem.
    """
    # 3-qubit state: 8-dimensional vector
    # Basis order: |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩
    state = np.zeros(8, dtype=complex)
    state[0b000] = alpha  # α|000⟩
    state[0b111] = beta   # β|111⟩
    return state

def apply_error(state, error_type, qubit):
    """Apply a Pauli error to a specific qubit in a 3-qubit state.

    Why use Pauli errors? Any single-qubit error can be decomposed as
    a linear combination of I, X, Y, Z. By correcting these four discrete
    errors, we automatically correct all continuous errors.
    """
    n = 3
    N = 8

    # Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    paulis = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

    # Build the n-qubit error operator
    ops = [I, I, I]
    ops[qubit] = paulis[error_type]

    error_op = np.kron(np.kron(ops[0], ops[1]), ops[2])
    return error_op @ state

def measure_syndrome_bit_flip(state):
    """Measure the bit-flip syndrome without collapsing the logical information.

    Why these specific measurements? Z_1⊗Z_2 checks if qubits 1 and 2
    have the same parity. Z_2⊗Z_3 checks qubits 2 and 3. Together, they
    uniquely identify which qubit (if any) was flipped, without revealing
    whether the logical state is |0_L⟩ or |1_L⟩.
    """
    n = 3
    I = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Syndrome operator Z1Z2 ⊗ I3
    Z1Z2 = np.kron(np.kron(Z, Z), I)
    # Syndrome operator I1 ⊗ Z2Z3
    Z2Z3 = np.kron(np.kron(I, Z), Z)

    # Expectation values give syndrome bits
    s1 = np.real(state.conj() @ Z1Z2 @ state)
    s2 = np.real(state.conj() @ Z2Z3 @ state)

    # Convert to syndrome: +1 → 0 (no error), -1 → 1 (error detected)
    syndrome = (int(round(-s1 + 1) / 2), int(round(-s2 + 1) / 2))

    return syndrome

def correct_bit_flip(state, syndrome):
    """Apply correction based on the syndrome.

    The syndrome uniquely identifies the error:
    (0,0) → no error, (1,0) → flip qubit 1, (1,1) → flip qubit 2, (0,1) → flip qubit 3
    """
    correction_map = {
        (0, 0): None,
        (1, 0): 0,  # Error on qubit 0
        (1, 1): 1,  # Error on qubit 1
        (0, 1): 2,  # Error on qubit 2
    }

    qubit = correction_map[syndrome]
    if qubit is not None:
        state = apply_error(state, 'X', qubit)  # Apply X to correct the flip
    return state

def decode_bit_flip(state):
    """Decode the 3-qubit state back to a single logical qubit.

    Returns the amplitudes (α, β) of the logical state α|0⟩ + β|1⟩.
    """
    alpha = state[0b000]
    beta = state[0b111]
    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    return alpha/norm, beta/norm

# Demonstration: encode, introduce error, detect, correct, decode
print("=" * 60)
print("3-Qubit Bit-Flip Code Demonstration")
print("=" * 60)

# Encode |ψ⟩ = (|0⟩ + i|1⟩)/√2
alpha, beta = 1/np.sqrt(2), 1j/np.sqrt(2)
print(f"\nOriginal state: α = {alpha:.4f}, β = {beta:.4f}")

state = encode_bit_flip(alpha, beta)
print(f"Encoded state (3 qubits): {state.round(4)}")

# Introduce a bit-flip error on qubit 1 (middle qubit)
error_qubit = 1
state_with_error = apply_error(state, 'X', error_qubit)
print(f"\nAfter X error on qubit {error_qubit}: {state_with_error.round(4)}")

# Measure syndrome
syndrome = measure_syndrome_bit_flip(state_with_error)
print(f"Syndrome: {syndrome}")
syndrome_meaning = {(0,0): "no error", (1,0): "qubit 0", (1,1): "qubit 1", (0,1): "qubit 2"}
print(f"Diagnosis: {syndrome_meaning[syndrome]}")

# Correct the error
corrected = correct_bit_flip(state_with_error, syndrome)
print(f"\nCorrected state: {corrected.round(4)}")

# Decode
alpha_out, beta_out = decode_bit_flip(corrected)
print(f"Decoded: α = {alpha_out:.4f}, β = {beta_out:.4f}")
print(f"Fidelity: {abs(alpha * alpha_out.conj() + beta * beta_out.conj())**2:.6f}")
```

### 10.2 Shor's 9-Qubit Code

```python
import numpy as np

def encode_shor_9qubit(alpha, beta):
    """Encode a logical qubit using Shor's 9-qubit code.

    The encoding is:
    |0_L⟩ = (|000⟩+|111⟩)(|000⟩+|111⟩)(|000⟩+|111⟩) / 2√2
    |1_L⟩ = (|000⟩-|111⟩)(|000⟩-|111⟩)(|000⟩-|111⟩) / 2√2

    Why two layers? The outer code (3-qubit phase-flip) handles Z errors,
    and the inner code (3-qubit bit-flip per block) handles X errors.
    Together, they correct any single-qubit Pauli error.
    """
    N = 2**9  # 512-dimensional Hilbert space

    # Build |+_L⟩ = (|000⟩ + |111⟩)/√2 for each block
    plus_block = np.zeros(8, dtype=complex)
    plus_block[0b000] = 1/np.sqrt(2)
    plus_block[0b111] = 1/np.sqrt(2)

    # Build |-_L⟩ = (|000⟩ - |111⟩)/√2 for each block
    minus_block = np.zeros(8, dtype=complex)
    minus_block[0b000] = 1/np.sqrt(2)
    minus_block[0b111] = -1/np.sqrt(2)

    # |0_L⟩ = |+_L⟩ ⊗ |+_L⟩ ⊗ |+_L⟩
    zero_L = np.kron(np.kron(plus_block, plus_block), plus_block)

    # |1_L⟩ = |-_L⟩ ⊗ |-_L⟩ ⊗ |-_L⟩
    one_L = np.kron(np.kron(minus_block, minus_block), minus_block)

    return alpha * zero_L + beta * one_L

def apply_9qubit_error(state, error_type, qubit):
    """Apply a Pauli error to a specific qubit in the 9-qubit code."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    paulis = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

    ops = [I] * 9
    ops[qubit] = paulis[error_type]

    # Build 9-qubit operator via tensor product
    error_op = ops[0]
    for i in range(1, 9):
        error_op = np.kron(error_op, ops[i])

    return error_op @ state

def measure_shor_syndrome(state):
    """Measure all 8 syndrome operators for Shor's 9-qubit code.

    Stabilizers:
    - Z1Z2, Z2Z3 (block 1 bit-flip)
    - Z4Z5, Z5Z6 (block 2 bit-flip)
    - Z7Z8, Z8Z9 (block 3 bit-flip)
    - X1X2X3X4X5X6 (blocks 1-2 phase comparison)
    - X4X5X6X7X8X9 (blocks 2-3 phase comparison)
    """
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    def build_9qubit_op(op_list):
        """Build 9-qubit operator from list of single-qubit ops."""
        result = op_list[0]
        for i in range(1, 9):
            result = np.kron(result, op_list[i])
        return result

    def measure_stabilizer(state, ops):
        """Measure expectation value of a stabilizer."""
        op = build_9qubit_op(ops)
        return int(round(np.real(state.conj() @ op @ state)))

    # Bit-flip syndromes (Z-type)
    bit_syndromes = []
    for block_start in [0, 3, 6]:
        for pair in [(0, 1), (1, 2)]:
            ops = [I] * 9
            ops[block_start + pair[0]] = Z
            ops[block_start + pair[1]] = Z
            val = measure_stabilizer(state, ops)
            bit_syndromes.append(val)

    # Phase-flip syndromes (X-type)
    phase_syndromes = []
    for block_pair in [(0, 3), (3, 6)]:
        ops = [I] * 9
        for i in range(block_pair[0], block_pair[0] + 3):
            ops[i] = X
        for i in range(block_pair[1], block_pair[1] + 3):
            ops[i] = X
        val = measure_stabilizer(state, ops)
        phase_syndromes.append(val)

    return bit_syndromes, phase_syndromes

def correct_shor_code(state, bit_syndromes, phase_syndromes):
    """Correct errors based on Shor code syndromes.

    Why separate bit-flip and phase-flip correction? Shor's code has a
    hierarchical structure: bit flips are corrected within each block of 3,
    then phase flips are corrected between blocks.
    """
    # Bit-flip correction (within each block)
    for block in range(3):
        s1 = bit_syndromes[2 * block]
        s2 = bit_syndromes[2 * block + 1]

        if s1 == -1 and s2 == 1:
            qubit = block * 3 + 0
            state = apply_9qubit_error(state, 'X', qubit)
        elif s1 == -1 and s2 == -1:
            qubit = block * 3 + 1
            state = apply_9qubit_error(state, 'X', qubit)
        elif s1 == 1 and s2 == -1:
            qubit = block * 3 + 2
            state = apply_9qubit_error(state, 'X', qubit)

    # Phase-flip correction (between blocks)
    p1, p2 = phase_syndromes
    if p1 == -1 and p2 == 1:
        # Phase error in block 1
        state = apply_9qubit_error(state, 'Z', 0)
    elif p1 == -1 and p2 == -1:
        # Phase error in block 2
        state = apply_9qubit_error(state, 'Z', 3)
    elif p1 == 1 and p2 == -1:
        # Phase error in block 3
        state = apply_9qubit_error(state, 'Z', 6)

    return state

# Test Shor's 9-qubit code with all error types
print("=" * 60)
print("Shor's 9-Qubit Code: Correcting X, Y, Z Errors")
print("=" * 60)

alpha, beta = np.sqrt(0.3), np.sqrt(0.7) * np.exp(1j * np.pi / 4)
print(f"\nLogical state: α={alpha:.4f}, β={beta:.4f}")

for error_type in ['X', 'Y', 'Z']:
    for error_qubit in [0, 4, 8]:  # Test one qubit in each block
        state = encode_shor_9qubit(alpha, beta)
        state_err = apply_9qubit_error(state, error_type, error_qubit)

        bit_syn, phase_syn = measure_shor_syndrome(state_err)
        corrected = correct_shor_code(state_err, bit_syn, phase_syn)

        # Check fidelity
        fidelity = abs(np.dot(state.conj(), corrected))**2
        print(f"  {error_type} on q{error_qubit}: "
              f"bit_syn={['+' if s==1 else '-' for s in bit_syn]}, "
              f"phase_syn={['+' if s==1 else '-' for s in phase_syn]}, "
              f"fidelity={fidelity:.6f}")
```

### 10.3 Error Rate vs Logical Error Rate

```python
import numpy as np

def simulate_error_correction_threshold(physical_error_rate, code_distance,
                                         n_trials=10000):
    """Simulate error correction to demonstrate the threshold effect.

    For a distance-d code, logical errors occur when ⌈d/2⌉ or more physical
    qubits have errors. Below threshold, increasing d exponentially suppresses
    the logical error rate. Above threshold, increasing d makes things worse.
    """
    n_qubits = code_distance  # Simplified: n = d for repetition code
    t = (code_distance - 1) // 2  # Number of correctable errors

    logical_errors = 0
    for _ in range(n_trials):
        # Each qubit has independent error with probability p
        errors = np.random.random(n_qubits) < physical_error_rate
        n_errors = np.sum(errors)

        # Code fails if more than t errors occur
        if n_errors > t:
            logical_errors += 1

    return logical_errors / n_trials

print("=" * 60)
print("Error Correction Threshold Demonstration")
print("=" * 60)
print("\nLogical error rate vs physical error rate for various code distances:")
print(f"{'p_phys':>8} {'d=3':>10} {'d=5':>10} {'d=7':>10} {'d=9':>10} {'d=11':>10}")

for p in [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]:
    rates = []
    for d in [3, 5, 7, 9, 11]:
        rate = simulate_error_correction_threshold(p, d, n_trials=50000)
        rates.append(rate)
    print(f"{p:8.3f} {rates[0]:10.6f} {rates[1]:10.6f} {rates[2]:10.6f} "
          f"{rates[3]:10.6f} {rates[4]:10.6f}")

print("\nNote: Below the threshold (~0.11 for repetition code),")
print("increasing d dramatically reduces logical error rate.")
print("Above threshold, increasing d makes things WORSE.")
```

---

## 11. Exercises

### Exercise 1: Bit-Flip Code by Hand

Consider the encoded state $|\psi_L\rangle = \frac{1}{\sqrt{3}}|000\rangle + \sqrt{\frac{2}{3}}|111\rangle$ (a non-standard superposition).

(a) Apply a bit-flip error ($X$) on qubit 2 (the middle qubit). What is the resulting state?
(b) Compute the syndrome by evaluating $\langle \psi'|Z_1 Z_2|\psi'\rangle$ and $\langle \psi'|Z_2 Z_3|\psi'\rangle$.
(c) Based on the syndrome, which correction should be applied?
(d) Verify that the corrected state equals the original encoded state.

### Exercise 2: Phase-Flip Code

Implement the 3-qubit phase-flip code:
(a) Write the encoding circuit in terms of Hadamard and CNOT gates.
(b) Define the syndrome operators ($X_1 X_2$ and $X_2 X_3$).
(c) Simulate encoding $|+\rangle$ state, applying a $Z$ error on qubit 2, and correcting it.
(d) Show that this code fails for $X$ errors.

### Exercise 3: Error Discretization

Consider a continuous rotation error $R_x(\epsilon) = \cos(\epsilon/2)I - i\sin(\epsilon/2)X$ on qubit 1 of the 3-qubit bit-flip code.
(a) Apply $R_x(\epsilon)$ to the encoded state $\alpha|000\rangle + \beta|111\rangle$.
(b) Compute the syndrome measurement probabilities.
(c) Show that after syndrome measurement, the state is projected into either the error-free subspace or the single-bit-flip subspace — the continuous error has been "digitized."
(d) Compute the probability of successful correction as a function of $\epsilon$.

### Exercise 4: Steane Code Implementation

Implement the Steane $[[7, 1, 3]]$ code:
(a) Look up the Steane code stabilizer generators and implement the encoding.
(b) Verify that all 6 stabilizer generators have eigenvalue $+1$ on the code space.
(c) Simulate single-qubit $X$, $Y$, and $Z$ errors on each of the 7 qubits and verify that the syndrome uniquely identifies each error.

### Exercise 5: Threshold Estimation

Using Monte Carlo simulation:
(a) For the repetition code with $d = 3, 5, 7, 9, 11$, plot the logical error rate as a function of physical error rate $p \in [0.001, 0.5]$.
(b) Estimate the threshold $p_{\text{th}}$ as the crossover point where all curves intersect.
(c) Compare with the theoretical threshold $p_{\text{th}} = 0.5$ for the infinite repetition code against i.i.d. bit-flip noise.
(d) How does this change for a depolarizing noise model (where $X$, $Y$, and $Z$ errors each occur with probability $p/3$)?

---

[← Previous: Shor's Factoring Algorithm](10_Shors_Algorithm.md) | [Next: Quantum Teleportation and Communication →](12_Quantum_Teleportation.md)
