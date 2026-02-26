# Lesson 6: Quantum Measurement

[<- Previous: Entanglement and Bell States](05_Entanglement_and_Bell_States.md) | [Next: Deutsch-Jozsa Algorithm ->](07_Deutsch_Jozsa_Algorithm.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Formalize projective (von Neumann) measurement using projection operators and explain the measurement postulate
2. Perform measurements in arbitrary bases and compute post-measurement states
3. Analyze partial measurement of multi-qubit systems and derive the resulting state of unmeasured qubits
4. Describe the general measurement formalism using $\{M_m\}$ operators
5. Explain POVM (positive operator-valued measure) and when it is useful
6. Describe the quantum Zeno effect and its physical implications
7. Simulate measurements and partial traces in Python

---

Measurement is the most subtle and philosophically challenging aspect of quantum mechanics. It is the only non-unitary, non-reversible operation in quantum computing -- the point where the quantum world meets the classical world. While gates transform quantum states smoothly and reversibly, measurement abruptly and irreversibly projects a superposition onto a definite outcome. Understanding measurement deeply is critical for quantum algorithm design, because the entire purpose of a quantum algorithm is to arrange the computation so that the *right* answer has high probability when we finally measure.

In this lesson, we formalize the measurement postulate introduced informally in [Lesson 1](01_Quantum_Mechanics_Primer.md) and extend it to multi-qubit systems. We will learn how measuring one qubit of an entangled pair affects the other, how to measure in bases other than the computational basis, and how more general measurements (POVMs) expand the toolkit available to quantum information theorists.

> **Analogy:** Measurement in quantum computing is like opening a gift box -- before opening, the gift could be many things (superposition). After opening, it is one definite item, and you can never un-open it. Moreover, the act of opening changes the gift: the quantum state of the system is permanently altered by the measurement, leaving it in the state corresponding to the observed outcome.

## Table of Contents

1. [Projective Measurement](#1-projective-measurement)
2. [Measurement in Different Bases](#2-measurement-in-different-bases)
3. [Partial Measurement](#3-partial-measurement)
4. [General Measurement Operators](#4-general-measurement-operators)
5. [POVM: Positive Operator-Valued Measure](#5-povm-positive-operator-valued-measure)
6. [The Quantum Zeno Effect](#6-the-quantum-zeno-effect)
7. [Post-Measurement States and Applications](#7-post-measurement-states-and-applications)
8. [Exercises](#8-exercises)

---

## 1. Projective Measurement

### 1.1 The Measurement Postulate

Projective measurement (also called von Neumann measurement) is the standard measurement model in quantum computing. Given a state $|\psi\rangle$ and an orthonormal basis $\{|m\rangle\}$:

1. **Probability of outcome $m$**: $P(m) = |\langle m|\psi\rangle|^2 = \langle\psi|P_m|\psi\rangle$
2. **Post-measurement state**: $|\psi_m'\rangle = \frac{P_m|\psi\rangle}{\sqrt{P(m)}}$

where $P_m = |m\rangle\langle m|$ is the **projection operator** onto state $|m\rangle$.

### 1.2 Projection Operators

A projector $P_m = |m\rangle\langle m|$ has two essential properties:

1. **Idempotent**: $P_m^2 = P_m$ (projecting twice gives the same result as projecting once)
2. **Hermitian**: $P_m^\dagger = P_m$ (it is its own adjoint)

The projectors for a complete measurement form a **resolution of the identity**:

$$\sum_m P_m = \sum_m |m\rangle\langle m| = I$$

This ensures probabilities sum to 1.

### 1.3 Measurement as an Observable

A projective measurement corresponds to measuring an **observable** -- a Hermitian operator $O$ with eigenvalues $\lambda_m$ and eigenvectors $|m\rangle$:

$$O = \sum_m \lambda_m |m\rangle\langle m| = \sum_m \lambda_m P_m$$

- The possible measurement outcomes are the eigenvalues $\lambda_m$
- The probability of outcome $\lambda_m$ is $P(m) = \langle\psi|P_m|\psi\rangle$
- The **expectation value** is $\langle O \rangle = \langle\psi|O|\psi\rangle = \sum_m \lambda_m P(m)$

For example, measuring the Pauli-$Z$ observable ($Z = |0\rangle\langle 0| - |1\rangle\langle 1|$) gives outcome $+1$ for $|0\rangle$ and $-1$ for $|1\rangle$.

```python
import numpy as np

# Projective measurement formalism

def projective_measurement(state, basis_states, labels=None):
    """
    Perform projective measurement in an arbitrary orthonormal basis.

    Why formalize this? In quantum computing, we often need to measure
    in bases other than computational (e.g., Bell basis, Hadamard basis).
    This function computes probabilities and post-measurement states
    for any complete orthonormal basis.
    """
    n = len(basis_states)
    if labels is None:
        labels = [str(i) for i in range(n)]

    results = []
    for i, (basis_vec, label) in enumerate(zip(basis_states, labels)):
        # Probability: P(m) = |<m|psi>|^2
        amplitude = np.vdot(basis_vec, state)
        prob = abs(amplitude)**2

        # Post-measurement state: P_m|psi> / sqrt(P(m))
        if prob > 1e-15:
            projected = np.vdot(basis_vec, state) * basis_vec
            post_state = projected / np.linalg.norm(projected)
        else:
            post_state = None

        results.append({
            'label': label,
            'probability': prob,
            'amplitude': amplitude,
            'post_state': post_state,
        })

    return results

# Example: Measure |+> = (|0> + |1>)/sqrt(2) in computational basis
print("=== Projective Measurement of |+> ===\n")

ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
basis = [np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)]

results = projective_measurement(ket_plus, basis, ["|0>", "|1>"])
for r in results:
    print(f"Outcome {r['label']}:")
    print(f"  Amplitude: {r['amplitude']:.4f}")
    print(f"  Probability: {r['probability']:.4f}")
    print(f"  Post-measurement state: {r['post_state']}")
    print()

# Verify: projectors sum to identity
P0 = np.outer(basis[0], basis[0].conj())
P1 = np.outer(basis[1], basis[1].conj())
print(f"P0 + P1 = I? {np.allclose(P0 + P1, np.eye(2))}")

# Verify: projectors are idempotent
print(f"P0^2 = P0? {np.allclose(P0 @ P0, P0)}")
print(f"P1^2 = P1? {np.allclose(P1 @ P1, P1)}")

# Expectation value of Z observable
Z = np.array([[1, 0], [0, -1]], dtype=complex)
exp_Z = np.real(np.vdot(ket_plus, Z @ ket_plus))
print(f"\nExpectation value <+|Z|+> = {exp_Z:.4f}")
print("(Makes sense: |+> gives +1 and -1 with equal probability, so <Z> = 0)")
```

---

## 2. Measurement in Different Bases

### 2.1 Changing the Measurement Basis

To measure in basis $\{|b_0\rangle, |b_1\rangle\}$ instead of $\{|0\rangle, |1\rangle\}$, we have two equivalent approaches:

**Approach 1** (Direct): Compute probabilities using the new basis:

$$P(b_i) = |\langle b_i|\psi\rangle|^2$$

**Approach 2** (Circuit): Apply a unitary $U$ that maps the desired basis to the computational basis, then measure in the computational basis:

$$U|b_i\rangle = |i\rangle \quad \Rightarrow \quad \text{Measure } U|\psi\rangle \text{ in computational basis}$$

Approach 2 is how measurements are actually performed on quantum hardware -- most quantum computers can only measure in the computational basis, so we rotate the state first.

### 2.2 Common Measurement Bases

| Basis | States | Rotation from $Z$-basis |
|-------|--------|------------------------|
| $Z$ (computational) | $\{\|0\rangle, \|1\rangle\}$ | None (identity) |
| $X$ (Hadamard) | $\{\|+\rangle, \|-\rangle\}$ | Apply $H$ |
| $Y$ (circular) | $\{\|i\rangle, \|-i\rangle\}$ | Apply $S^\dagger H$ |
| Bell | $\{\|\Phi^\pm\rangle, \|\Psi^\pm\rangle\}$ | Apply CNOT then $H$ |

### 2.3 Physical Significance

Different measurement bases extract different information from the same state. This is a key difference from classical physics, where measurement simply reads a pre-existing value. In quantum mechanics, the *choice* of what to measure actively shapes the outcomes.

This is directly related to the uncertainty principle: measuring precisely in the $Z$-basis gives maximal uncertainty about $X$-basis properties, and vice versa.

```python
import numpy as np

# Measurement in different bases

def measure_in_basis(state, basis_name="Z"):
    """
    Compute measurement probabilities in a named basis.

    Why support multiple bases? On real quantum hardware, you typically
    can only measure in the Z-basis. To measure in another basis,
    you first apply a rotation gate, then measure in Z.
    This function simulates both approaches and verifies they agree.
    """
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S_dag = np.array([[1, 0], [0, -1j]], dtype=complex)

    if basis_name == "Z":
        # Computational basis: no rotation needed
        rotation = np.eye(2, dtype=complex)
        basis_labels = ["|0>", "|1>"]
        basis_states = [np.array([1, 0], dtype=complex),
                       np.array([0, 1], dtype=complex)]
    elif basis_name == "X":
        # Hadamard basis: rotate by H
        rotation = H
        basis_labels = ["|+>", "|->"]
        basis_states = [np.array([1, 1], dtype=complex) / np.sqrt(2),
                       np.array([1, -1], dtype=complex) / np.sqrt(2)]
    elif basis_name == "Y":
        # Circular basis: rotate by S^dag @ H
        rotation = S_dag @ H
        basis_labels = ["|i>", "|-i>"]
        basis_states = [np.array([1, 1j], dtype=complex) / np.sqrt(2),
                       np.array([1, -1j], dtype=complex) / np.sqrt(2)]
    else:
        raise ValueError(f"Unknown basis: {basis_name}")

    # Method 1: Direct computation using basis states
    probs_direct = [abs(np.vdot(b, state))**2 for b in basis_states]

    # Method 2: Rotate then measure in Z
    rotated = rotation @ state
    probs_rotated = [abs(rotated[0])**2, abs(rotated[1])**2]

    return basis_labels, probs_direct, probs_rotated

# Test state: |psi> = cos(pi/8)|0> + sin(pi/8)|1>
theta = np.pi / 4  # Bloch sphere theta
psi = np.array([np.cos(theta/2), np.sin(theta/2)], dtype=complex)

print("=== Measuring the Same State in Three Bases ===\n")
print(f"State: cos(pi/8)|0> + sin(pi/8)|1>")
print(f"  (Bloch sphere: theta=pi/4, phi=0 -- tilted 22.5 deg from north pole)\n")

for basis in ["Z", "X", "Y"]:
    labels, probs_d, probs_r = measure_in_basis(psi, basis)
    print(f"{basis}-basis measurement:")
    for i in range(2):
        print(f"  P({labels[i]}) = {probs_d[i]:.4f} "
              f"(via rotation: {probs_r[i]:.4f})")
    print()

# Demonstrate: a state defined in X-basis
print("=== State |-> Measured in All Bases ===\n")
ket_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)

for basis in ["Z", "X", "Y"]:
    labels, probs_d, _ = measure_in_basis(ket_minus, basis)
    print(f"{basis}-basis: P({labels[0]}) = {probs_d[0]:.4f}, "
          f"P({labels[1]}) = {probs_d[1]:.4f}")

print("\n|-> is definite in X but random in Z and Y -- complementarity!")
```

---

## 3. Partial Measurement

### 3.1 Measuring a Subset of Qubits

In a multi-qubit system, we often measure only some qubits while leaving others untouched. This is **partial measurement**, and it is fundamentally different from measuring all qubits.

For a two-qubit state $|\psi\rangle = \alpha_{00}|00\rangle + \alpha_{01}|01\rangle + \alpha_{10}|10\rangle + \alpha_{11}|11\rangle$:

**Measuring qubit 0 and getting outcome $|0\rangle$**:

$$P(q_0 = 0) = |\alpha_{00}|^2 + |\alpha_{10}|^2$$

Post-measurement state (after renormalization):

$$|\psi'\rangle = \frac{\alpha_{00}|00\rangle + \alpha_{10}|10\rangle}{\sqrt{|\alpha_{00}|^2 + |\alpha_{10}|^2}}$$

The key insight: measuring qubit 0 as $|0\rangle$ **collapses** only the measured qubit but also *updates* our knowledge about qubit 1.

### 3.2 Effect on Entangled States

For a Bell state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$:

- Measuring qubit 0 and getting $|0\rangle$: Post-state is $|00\rangle$ (qubit 1 is forced to $|0\rangle$)
- Measuring qubit 0 and getting $|1\rangle$: Post-state is $|11\rangle$ (qubit 1 is forced to $|1\rangle$)

The measurement of one qubit **instantaneously determines** the state of the other, regardless of distance. This is the "spooky action at a distance" Einstein objected to (see [Lesson 5](05_Entanglement_and_Bell_States.md)).

### 3.3 The Partial Trace

The **partial trace** is the mathematical operation that describes the state of a subsystem when we ignore (do not measure) the other subsystem. For a density matrix $\rho_{AB}$:

$$\rho_A = \text{Tr}_B(\rho_{AB}) = \sum_i (I_A \otimes \langle i|_B) \rho_{AB} (I_A \otimes |i\rangle_B)$$

If the full system is in a pure entangled state, the reduced density matrix $\rho_A$ is a **mixed state** -- it has lost the purity that comes with complete quantum information.

```python
import numpy as np

# Partial measurement of multi-qubit systems

def partial_measure(state, qubit, outcome, n_qubits):
    """
    Compute the post-measurement state after measuring one qubit.

    Why is partial measurement important? In quantum algorithms, we often
    measure ancilla qubits while keeping the data qubits intact. The
    measurement outcome on the ancilla tells us something about the data
    register, and the post-measurement state of the data register depends
    on what we observed.

    Parameters:
        state: full state vector (2^n_qubits complex amplitudes)
        qubit: which qubit to measure (0-indexed)
        outcome: desired measurement outcome (0 or 1)
        n_qubits: number of qubits

    Returns:
        probability of this outcome, post-measurement state
    """
    dim = 2**n_qubits
    new_state = np.zeros(dim, dtype=complex)

    for i in range(dim):
        bit = (i >> qubit) & 1
        if bit == outcome:
            new_state[i] = state[i]

    prob = np.linalg.norm(new_state)**2

    if prob > 1e-15:
        new_state = new_state / np.linalg.norm(new_state)

    return prob, new_state

def partial_trace(state_vector, trace_out_qubit, n_qubits):
    """
    Compute the reduced density matrix by tracing out one qubit.

    Why partial trace? When we have an entangled system and want to
    describe just one subsystem, the partial trace gives the correct
    reduced description. For entangled states, this will be a mixed
    state (not a pure state), reflecting our uncertainty about the
    subsystem.
    """
    rho = np.outer(state_vector, state_vector.conj())
    dim = 2**n_qubits
    dim_remaining = dim // 2

    # Reshape to separate out the traced qubit
    # For n qubits, we need to trace out one index
    shape = [2] * (2 * n_qubits)
    rho_tensor = rho.reshape(shape)

    # Trace over the specified qubit
    # The qubit indices in the tensor are: qubit for bra and ket
    # ket indices: 0, 1, ..., n-1 (MSB to LSB)
    # bra indices: n, n+1, ..., 2n-1
    ket_idx = n_qubits - 1 - trace_out_qubit  # Position in tensor
    bra_idx = 2 * n_qubits - 1 - trace_out_qubit

    rho_reduced = np.trace(rho_tensor, axis1=ket_idx, axis2=bra_idx)
    return rho_reduced.reshape(dim_remaining, dim_remaining)

# === Partial measurement of Bell state ===
print("=== Partial Measurement of Bell State |Phi+> ===\n")

bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

for outcome in [0, 1]:
    prob, post = partial_measure(bell, qubit=0, outcome=outcome, n_qubits=2)
    print(f"Measure q0, get |{outcome}>:")
    print(f"  Probability: {prob:.4f}")
    print(f"  Post-measurement state: {np.round(post, 4)}")
    # Show what happened to qubit 1
    nonzero = [(format(i, '02b'), post[i]) for i in range(4) if abs(post[i]) > 1e-10]
    for label, amp in nonzero:
        print(f"    |{label}>: amplitude = {amp:.4f}")
    print()

# === Partial trace: reduced density matrix ===
print("=== Partial Trace ===\n")

# Bell state: trace out qubit 1
print("Bell state |Phi+> = (|00> + |11>)/sqrt(2):")
rho_q0 = partial_trace(bell, trace_out_qubit=1, n_qubits=2)
print(f"  rho_q0 (after tracing out q1):\n    {rho_q0}")
eigenvalues = np.linalg.eigvalsh(rho_q0)
print(f"  Eigenvalues: {eigenvalues}")
print(f"  This is a MAXIMALLY MIXED state (I/2)!")
print(f"  Entropy: {-sum(e*np.log2(e) for e in eigenvalues if e > 1e-10):.4f} bits\n")

# Product state: trace out qubit 1
product = np.kron(
    np.array([1, 1], dtype=complex) / np.sqrt(2),  # |+>
    np.array([1, 0], dtype=complex)                  # |0>
)
print("Product state |+>|0>:")
rho_q0 = partial_trace(product, trace_out_qubit=1, n_qubits=2)
print(f"  rho_q0 (after tracing out q1):\n    {rho_q0}")
eigenvalues = np.linalg.eigvalsh(rho_q0)
print(f"  Eigenvalues: {eigenvalues}")
print(f"  This is a PURE state |+><+|!")
print(f"  Entropy: {-sum(e*np.log2(e) for e in eigenvalues if e > 1e-10):.4f} bits")
```

---

## 4. General Measurement Operators

### 4.1 Beyond Projective Measurement

The projective measurement formalism requires measurement operators to be orthogonal projectors. A more general framework uses **measurement operators** $\{M_m\}$ where:

1. **Probability**: $P(m) = \langle\psi|M_m^\dagger M_m|\psi\rangle$
2. **Post-measurement state**: $|\psi_m'\rangle = \frac{M_m|\psi\rangle}{\sqrt{P(m)}}$
3. **Completeness**: $\sum_m M_m^\dagger M_m = I$ (ensures probabilities sum to 1)

Unlike projective measurement:
- The $M_m$ need not be Hermitian
- The $M_m$ need not be orthogonal to each other
- The number of outcomes can exceed the dimension of the Hilbert space

### 4.2 Relationship to Projective Measurement

Projective measurement is the special case where $M_m = P_m = |m\rangle\langle m|$. Then:

$$M_m^\dagger M_m = P_m^\dagger P_m = P_m^2 = P_m$$

and the completeness condition becomes $\sum_m P_m = I$ (resolution of the identity).

### 4.3 Neumark's Theorem

Any general measurement on a system can be realized as a projective measurement on a larger system (system + ancilla). This is important practically: it means we can implement any measurement we want using the tools we already have (unitary gates + projective measurement on ancillas).

```python
import numpy as np

# General measurement operators

def general_measurement(state, measurement_ops, labels=None):
    """
    Perform a generalized measurement using operators {M_m}.

    Why generalize beyond projective measurement? Sometimes we want
    measurements that have more outcomes than dimensions, or that
    partially distinguish between non-orthogonal states. The general
    formalism handles all these cases.

    Completeness check: sum(M_m^dag @ M_m) = I
    """
    n_outcomes = len(measurement_ops)
    dim = len(state)
    if labels is None:
        labels = [f"m={i}" for i in range(n_outcomes)]

    # Verify completeness
    completeness = sum(M.conj().T @ M for M in measurement_ops)
    assert np.allclose(completeness, np.eye(dim)), "Measurement operators not complete!"

    results = []
    for M, label in zip(measurement_ops, labels):
        # P(m) = <psi|M_m^dag M_m|psi>
        prob = np.real(np.vdot(state, M.conj().T @ M @ state))

        # Post-measurement state: M_m|psi> / sqrt(P(m))
        if prob > 1e-15:
            post = M @ state
            post = post / np.linalg.norm(post)
        else:
            post = None

        results.append({'label': label, 'probability': prob, 'post_state': post})

    return results

# Example: Projective measurement (special case)
print("=== Projective Measurement as Special Case ===\n")

state = np.array([1, 1], dtype=complex) / np.sqrt(2)  # |+>

M0 = np.array([[1, 0], [0, 0]], dtype=complex)  # |0><0|
M1 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1><1|

results = general_measurement(state, [M0, M1], ["|0>", "|1>"])
for r in results:
    print(f"Outcome {r['label']}: P = {r['probability']:.4f}, "
          f"post-state = {r['post_state']}")

# Example: Non-projective measurement (three outcomes for a qubit!)
print("\n=== Non-Projective Measurement (3 outcomes, 2D system) ===\n")

# SIC-POVM: Symmetric Informationally Complete POVM
# Three measurement directions separated by 120 degrees in the Bloch sphere
# M_m = (2/3) |psi_m><psi_m|  (scaled projectors)
psi0 = np.array([1, 0], dtype=complex)
psi1 = np.array([1/2, np.sqrt(3)/2], dtype=complex)
psi2 = np.array([1/2, -np.sqrt(3)/2], dtype=complex)

# Measurement operators: M_m = sqrt(2/3) |psi_m><psi_m|
# Why sqrt? Because M_m^dag M_m = (2/3)|psi_m><psi_m|, and sum = I
factor = np.sqrt(2/3)
M_ops = [factor * np.outer(psi, psi.conj()) for psi in [psi0, psi1, psi2]]

# Verify completeness
total = sum(M.conj().T @ M for M in M_ops)
print(f"Completeness check (should be I):\n{np.round(total, 4)}")

# Apply to |+> state
state = np.array([1, 1], dtype=complex) / np.sqrt(2)
results = general_measurement(state, M_ops, ["up", "120-deg", "240-deg"])
print(f"\nMeasuring |+> with trine POVM:")
for r in results:
    print(f"  Outcome '{r['label']}': P = {r['probability']:.4f}")
print(f"\nNote: 3 outcomes for a 2-dimensional system!")
print("This is impossible with projective measurement.")
```

---

## 5. POVM: Positive Operator-Valued Measure

### 5.1 Definition

A **POVM** is defined by a set of positive semi-definite operators $\{E_m\}$ satisfying:

$$E_m \geq 0 \quad \text{(positive semi-definite)}, \qquad \sum_m E_m = I$$

The probability of outcome $m$ is:

$$P(m) = \langle\psi|E_m|\psi\rangle = \text{Tr}(E_m \rho)$$

where $E_m = M_m^\dagger M_m$ are called **POVM elements** or **effects**.

### 5.2 POVM vs Projective Measurement

| Feature | Projective | POVM |
|---------|-----------|------|
| Operators | $P_m = \|m\rangle\langle m\|$ | $E_m \geq 0$, $\sum E_m = I$ |
| Number of outcomes | $\leq$ dimension | Any number |
| Post-measurement state | Well-defined | May not be uniquely defined |
| Repeatability | Yes ($P_m^2 = P_m$) | Not necessarily |
| Physical realization | Direct | Requires ancilla + projective |

### 5.3 When Are POVMs Useful?

POVMs are useful when:

1. **State discrimination**: Distinguishing between non-orthogonal states (e.g., $|0\rangle$ vs $|+\rangle$) is sometimes better with POVMs than projective measurements.

2. **Quantum cryptography**: BB84 protocol analysis uses POVMs to characterize the optimal eavesdropping strategy.

3. **Quantum tomography**: Reconstructing an unknown quantum state requires informationally complete measurements, which are naturally described as POVMs.

### 5.4 Example: Unambiguous State Discrimination

Given a qubit that is either $|0\rangle$ or $|+\rangle$ with equal prior probability, we want to identify which state it is. Projective measurement cannot perfectly distinguish non-orthogonal states. But a POVM can give *unambiguous* answers with an "inconclusive" outcome:

- Outcome "is $|0\rangle$": only triggered by $|0\rangle$, never by $|+\rangle$
- Outcome "is $|+\rangle$": only triggered by $|+\rangle$, never by $|0\rangle$
- Outcome "inconclusive": can be triggered by either

```python
import numpy as np

# Unambiguous state discrimination with POVM

# States to distinguish
ket_0 = np.array([1, 0], dtype=complex)
ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)

# Overlap between the states
overlap = abs(np.vdot(ket_0, ket_plus))**2
print("=== Unambiguous State Discrimination ===\n")
print(f"Distinguishing |0> from |+>")
print(f"Overlap: |<0|+>|^2 = {overlap:.4f}")
print(f"These states are NOT orthogonal, so projective measurement")
print(f"cannot perfectly distinguish them.\n")

# POVM for unambiguous discrimination
# E_0: detects |0> (never fires for |+>)
# E_+: detects |+> (never fires for |0>)
# E_?: inconclusive

# |+> is orthogonal to |-> = (|0>-|1>)/sqrt(2)
# So a detector based on |-> never fires for |+>
ket_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)

# |0> is orthogonal to |1>
# So a detector based on |1> never fires for |0>
ket_1 = np.array([0, 1], dtype=complex)

# POVM elements (scaled to ensure completeness)
# Why these scaling factors? We need E_0 + E_+ + E_? = I
# and each E must be positive semi-definite.
# The optimal scaling maximizes the probability of a conclusive result.

# E_detect_0 = c * |1><1| (fires for |0> because |0> has zero |1> component... wait)
# Actually: E_detect_plus should never fire for |0>, so E_detect_plus ~ |1><1|
# E_detect_0 should never fire for |+>, so E_detect_0 ~ |-><-|

# Scale factor for optimality
p_fail = abs(np.vdot(ket_0, ket_plus))  # = 1/sqrt(2)

E_detect_0 = (1 - p_fail) * np.outer(ket_minus, ket_minus.conj())  # Detects |0>
E_detect_plus = (1 - p_fail) * np.outer(ket_1, ket_1.conj())       # Detects |+>
E_inconclusive = np.eye(2) - E_detect_0 - E_detect_plus

print("POVM elements:")
print(f"E_detect_0 =\n{np.round(E_detect_0, 4)}")
print(f"E_detect_+ =\n{np.round(E_detect_plus, 4)}")
print(f"E_inconclusive =\n{np.round(E_inconclusive, 4)}")

# Verify completeness
print(f"\nCompleteness (sum = I)? {np.allclose(E_detect_0 + E_detect_plus + E_inconclusive, np.eye(2))}")

# Verify positive semi-definiteness
for name, E in [("E_detect_0", E_detect_0), ("E_detect_+", E_detect_plus),
                ("E_?", E_inconclusive)]:
    eigvals = np.linalg.eigvalsh(E)
    print(f"{name} eigenvalues: {np.round(eigvals, 6)} (all >= 0? {all(e >= -1e-10 for e in eigvals)})")

# Test: probabilities for each input state
print("\n--- Results ---\n")
for name, state in [("|0>", ket_0), ("|+>", ket_plus)]:
    p_0 = np.real(np.vdot(state, E_detect_0 @ state))
    p_plus = np.real(np.vdot(state, E_detect_plus @ state))
    p_inc = np.real(np.vdot(state, E_inconclusive @ state))

    print(f"Input: {name}")
    print(f"  P(detect |0>) = {p_0:.4f}")
    print(f"  P(detect |+>) = {p_plus:.4f}")
    print(f"  P(inconclusive) = {p_inc:.4f}")
    print()

print("Key: When the POVM gives a conclusive answer, it is ALWAYS correct!")
print("The cost: sometimes it says 'inconclusive'.")
```

---

## 6. The Quantum Zeno Effect

### 6.1 Description

The **quantum Zeno effect** (sometimes called the "watched pot" effect) states that frequently measuring a quantum system inhibits its evolution. A system that would naturally evolve away from its initial state can be "frozen" by repeated measurements.

### 6.2 Mathematical Explanation

Consider a qubit initially in $|0\rangle$ undergoing rotation $R_y(\theta)$ (which gradually rotates toward $|1\rangle$). After one full rotation of angle $\theta$:

$$P(0) = \cos^2(\theta/2)$$

Now suppose we divide the rotation into $N$ steps of $\theta/N$ each, measuring after each step. The probability of remaining in $|0\rangle$ through all $N$ measurements is:

$$P_{\text{survive}} = \left[\cos^2\left(\frac{\theta}{2N}\right)\right]^N$$

As $N \to \infty$:

$$\lim_{N \to \infty} \left[\cos^2\left(\frac{\theta}{2N}\right)\right]^N = \lim_{N \to \infty} \left[1 - \frac{\theta^2}{4N^2}\right]^N = 1$$

The system is frozen in $|0\rangle$ by continuous measurement.

### 6.3 Intuition

Each measurement "resets" the system to $|0\rangle$ (assuming that is what we observe). With each reset, only a tiny amount of evolution has occurred, so the probability of finding $|0\rangle$ is very high. In the limit of continuous measurement, the system never gets a chance to evolve.

```python
import numpy as np

# Quantum Zeno effect simulation

def Ry(theta):
    """Y-axis rotation gate."""
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)

# Total rotation: pi/2 (would take |0> to (|0>+|1>)/sqrt(2) without measurement)
total_theta = np.pi / 2
ket_0 = np.array([1, 0], dtype=complex)

print("=== Quantum Zeno Effect ===\n")
print(f"Total rotation: Ry(pi/2)")
print(f"Without measurement: P(0) = cos^2(pi/4) = {np.cos(np.pi/4)**2:.4f}\n")

print(f"{'N measurements':>15} {'Theta/step':>15} {'P(survive in |0>)':>20} "
      f"{'MC estimate':>15}")
print("-" * 70)

np.random.seed(42)
for N in [1, 2, 5, 10, 20, 50, 100, 1000]:
    step_theta = total_theta / N

    # Exact probability of surviving all N measurements in |0>
    p_exact = np.cos(step_theta / 2)**(2 * N)

    # Monte Carlo simulation
    n_trials = 10000
    survived = 0
    for _ in range(n_trials):
        state = ket_0.copy()
        alive = True
        for _ in range(N):
            # Apply small rotation
            state = Ry(step_theta) @ state
            # Measure: P(0) = |<0|state>|^2
            p0 = abs(state[0])**2
            if np.random.random() < p0:
                # Collapse to |0>
                state = ket_0.copy()
            else:
                alive = False
                break
        if alive:
            survived += 1

    p_mc = survived / n_trials
    print(f"{N:>15} {step_theta:>15.6f} {p_exact:>20.6f} {p_mc:>15.4f}")

print(f"\nAs N increases, P(survive) -> 1.0")
print(f"The system is 'frozen' by frequent measurement!")
print(f"This is the quantum Zeno effect.")
```

---

## 7. Post-Measurement States and Applications

### 7.1 Measurement-Based State Preparation

Measurement can be used as a tool for *creating* desired quantum states, not just for reading information:

1. **Heralded state preparation**: Prepare a superposition, measure an ancilla, and keep the result only if the ancilla gives the desired outcome. This "post-selects" a particular state.

2. **Measurement-based quantum computation (MBQC)**: An entire paradigm of quantum computing where the computation proceeds entirely through adaptive measurements on a large entangled state (cluster state). No unitary gates are needed after the initial entanglement.

### 7.2 Mid-Circuit Measurement

Modern quantum computers support **mid-circuit measurement**: measuring some qubits in the middle of a circuit and using the outcomes to control subsequent operations. This is used in:

- **Quantum error correction**: Syndrome measurements detect errors without collapsing the logical qubit
- **Quantum teleportation**: Alice measures and classically communicates to Bob
- **Adaptive algorithms**: Measurement outcomes guide the choice of subsequent gates

### 7.3 Measurement in Quantum Algorithms

In the algorithms we will study next:

- **Deutsch-Jozsa** ([Lesson 7](07_Deutsch_Jozsa_Algorithm.md)): The final measurement distinguishes constant from balanced functions. The algorithm is designed so that the correct answer has probability 1 (deterministic).

- **Grover's search** ([Lesson 8](08_Grovers_Search.md)): The final measurement finds the marked item. The algorithm amplifies the correct answer's probability to near 1 through repeated oracle/diffusion iterations.

```python
import numpy as np

# Heralded state preparation example

def heralded_preparation(target_prob, n_attempts=10000):
    """
    Prepare a specific state by post-selecting on measurement outcomes.

    Why post-selection? Sometimes we need a state that is hard to
    prepare directly with unitary gates. By preparing a larger system,
    measuring part of it, and keeping only certain outcomes, we can
    "filter" for the desired state. The cost is that some attempts fail.
    """
    # Goal: prepare |psi> = sqrt(1/3)|0> + sqrt(2/3)|1>
    # Strategy: prepare |psi>|0> + ... in a 2-qubit system,
    # measure ancilla, keep result only if ancilla gives |0>

    # Use a 2-qubit circuit:
    # 1. Start with |00>
    # 2. Apply Ry(2*arccos(sqrt(1/3))) to qubit 0 -> sqrt(1/3)|0> + sqrt(2/3)|1>
    # 3. This is actually direct preparation! For heralding, let's make it more
    #    interesting: prepare a Bell state and measure one qubit.

    # Heralded Bell pair:
    # Prepare (|00> + |11>)/sqrt(2), measure qubit 1
    # If qubit 1 = 0, qubit 0 is in |0>
    # If qubit 1 = 1, qubit 0 is in |1>
    # This is trivial -- let's do something more interesting

    # Prepare sqrt(1/3)|00> + sqrt(1/3)|01> + sqrt(1/3)|10>
    # Measure qubit 1: if we get |0>, qubit 0 is in (|0> + |1>)/sqrt(2) (unnormalized)
    # Wait, this is getting complicated. Let me just demonstrate the concept.

    # Simple example: We want |+>, but can only prepare |0> and measure
    # Strategy: Apply H, which directly gives |+>. But for heralding demo:
    # Prepare |phi> = (sqrt(3)|00> + |11>)/2
    # Measure qubit 0. If get |0> (prob 3/4), qubit 1 is |0>.
    # If get |1> (prob 1/4), qubit 1 is |1>.
    # This is just heralded state preparation of |0> or |1>.

    print("=== Heralded State Preparation ===\n")
    print("Strategy: Prepare a 2-qubit state, measure ancilla,")
    print("keep data qubit only when ancilla gives desired outcome.\n")

    # Prepare |psi> = sqrt(1/3)|00> + sqrt(2/3)|01>  on (data, ancilla)
    # If ancilla is |0>: data is in |0> (probability 1/3)
    # If ancilla is |1>: data is in |0> (probability 2/3)

    # Better example: Entangle then measure
    # |psi_2q> = 1/sqrt(3) |0>|0> + sqrt(2/3) |1>|1>
    # Measure ancilla (qubit 1):
    #   If ancilla=0 (prob 1/3): data qubit is |0>  -> heralded |0>
    #   If ancilla=1 (prob 2/3): data qubit is |1>  -> heralded |1>

    # Most useful example: preparing a superposition state through heralding
    # Start: |00>
    # Apply H to both qubits: |++> = (|00> + |01> + |10> + |11>)/2
    # Measure qubit 1:
    #   If qubit 1 = 0: qubit 0 is |+> = (|0>+|1>)/sqrt(2)
    #   If qubit 1 = 1: qubit 0 is |+> = (|0>+|1>)/sqrt(2)
    # (Not interesting because qubits are separable)

    # Truly useful: preparing an unusual state
    # |psi> = (|00> + |01> + |10>)/sqrt(3)
    # Measure qubit 1:
    #   qubit1=0 (prob 2/3): qubit0 = (|0>+|1>)/sqrt(2) = |+>
    #   qubit1=1 (prob 1/3): qubit0 = |0>

    psi = np.array([1, 1, 1, 0], dtype=complex) / np.sqrt(3)

    success_count = 0
    data_states = []

    for _ in range(n_attempts):
        # Measure qubit 0 (ancilla)
        p0 = abs(psi[0])**2 + abs(psi[2])**2  # |x0> components

        if np.random.random() < p0:
            # Got qubit0=0, post-select
            success_count += 1
            # Data qubit (qubit 1) state
            post = np.array([psi[0], psi[2]], dtype=complex)
            post = post / np.linalg.norm(post)
            data_states.append(post)

    print(f"Initial 2-qubit state: (|00> + |01> + |10>)/sqrt(3)")
    print(f"Post-select on ancilla (q0) = |0>")
    print(f"  Success rate: {success_count/n_attempts:.4f} (expected: 2/3 = {2/3:.4f})")

    if data_states:
        # All heralded data qubit states should be the same (up to measurement randomness)
        avg_state = np.mean(data_states, axis=0)
        print(f"  Heralded data state: {data_states[0]}")
        print(f"  Expected: |+> = {np.array([1,1])/np.sqrt(2)}")

heralded_preparation(1/3)
```

---

## 8. Exercises

### Exercise 1: Projective Measurement

A qubit is in state $|\psi\rangle = \frac{1+i}{2}|0\rangle + \frac{1}{2}|1\rangle$. (Verify this is normalized first.)

a) What are the probabilities of measuring $|0\rangle$ and $|1\rangle$ in the computational basis?
b) What are the probabilities in the $X$-basis ($\{|+\rangle, |-\rangle\}$)?
c) Compute the expectation values $\langle X \rangle$, $\langle Y \rangle$, and $\langle Z \rangle$.
d) What are the Bloch sphere coordinates of this state?

### Exercise 2: Partial Measurement

Consider the 3-qubit state $|\psi\rangle = \frac{1}{2}(|000\rangle + |011\rangle + |100\rangle + |111\rangle)$.

a) If we measure qubit 2 and get $|0\rangle$, what is the probability and post-measurement state?
b) If we then measure qubit 1 of the post-measurement state and get $|1\rangle$, what state remains?
c) Compute the reduced density matrix of qubit 0 by tracing out qubits 1 and 2. Is qubit 0 in a pure or mixed state?

### Exercise 3: POVM Design

Design a POVM with three elements that can distinguish between the states $|0\rangle$, $|+\rangle$, and $|i\rangle$ with some probability (unambiguous discrimination is impossible for three non-orthogonal states in 2D, but you can design a "minimum-error" POVM). Write Python code to implement and test your POVM.

### Exercise 4: Quantum Zeno Variations

a) Modify the Zeno effect simulation to use $R_x(\theta)$ rotation instead of $R_y(\theta)$. Does the effect still occur? Why or why not?
b) What happens if instead of measuring in the $Z$-basis, we measure in the $X$-basis after each step? Simulate and explain.
c) **Anti-Zeno effect**: Under certain conditions, frequent measurement can *accelerate* decay. Research and explain when this occurs (conceptual answer is fine).

### Exercise 5: Measurement-Based Computation

Implement a simple measurement-based computation:
a) Prepare a 2-qubit cluster state: $|CS\rangle = \text{CZ}(|+\rangle \otimes |+\rangle)$
b) Measure qubit 0 in the basis $\{R_z(\alpha)|+\rangle, R_z(\alpha)|-\rangle\}$ for angle $\alpha$ of your choice
c) Show that depending on the measurement outcome and $\alpha$, qubit 1 ends up in a state equivalent to $R_z(\alpha)|+\rangle$ or its $X$-corrected version
d) This demonstrates that measurement + entanglement can implement a quantum gate!

---

[<- Previous: Entanglement and Bell States](05_Entanglement_and_Bell_States.md) | [Next: Deutsch-Jozsa Algorithm ->](07_Deutsch_Jozsa_Algorithm.md)
