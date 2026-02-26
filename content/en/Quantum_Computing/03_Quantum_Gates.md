# Lesson 3: Quantum Gates

[<- Previous: Qubits and the Bloch Sphere](02_Qubits_and_Bloch_Sphere.md) | [Next: Quantum Circuits ->](04_Quantum_Circuits.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe single-qubit gates (X, Y, Z, H, S, T, $R_z(\theta)$) as unitary matrices and visualize their action on the Bloch sphere
2. Apply quantum gates to state vectors using matrix-vector multiplication
3. Explain controlled gates (CNOT, CZ, controlled-U) and their role in creating entanglement
4. Construct multi-qubit gates like SWAP from elementary gates
5. Define what makes a set of gates "universal" and why $\{H, T, \text{CNOT}\}$ suffices
6. Decompose arbitrary single-qubit gates into sequences of elementary gates
7. Implement all major gates as numpy matrices and apply them to quantum state vectors

---

Quantum gates are the basic operations of quantum computing, analogous to classical logic gates (AND, OR, NOT) but with crucial differences. While classical gates are irreversible (knowing the output of an AND gate does not uniquely determine the inputs), quantum gates are always *reversible* -- every gate is a unitary transformation with a well-defined inverse. This reversibility is a fundamental constraint imposed by quantum mechanics and shapes the entire paradigm of quantum circuit design.

In this lesson, we build up the complete toolkit of standard quantum gates, from single-qubit rotations to two-qubit entangling operations. By the end, you will understand why a small set of gates is sufficient to approximate any quantum computation -- the quantum analog of the fact that NAND gates alone can build any classical circuit.

> **Analogy:** Quantum gates are like polarizing filters for light -- each one transforms the quantum state in a specific way, and composing them creates complex transformations. Just as a sequence of polarizing filters can rotate light's polarization to any desired angle, a sequence of quantum gates can transform any quantum state into any other.

## Table of Contents

1. [Unitarity: The Fundamental Constraint](#1-unitarity-the-fundamental-constraint)
2. [Pauli Gates: X, Y, Z](#2-pauli-gates-x-y-z)
3. [The Hadamard Gate](#3-the-hadamard-gate)
4. [Phase Gates: S and T](#4-phase-gates-s-and-t)
5. [Rotation Gates](#5-rotation-gates)
6. [Two-Qubit Gates](#6-two-qubit-gates)
7. [Universal Gate Sets](#7-universal-gate-sets)
8. [Gate Decomposition](#8-gate-decomposition)
9. [Exercises](#9-exercises)

---

## 1. Unitarity: The Fundamental Constraint

### 1.1 Why Unitary?

Every quantum gate is represented by a **unitary matrix** $U$ satisfying:

$$U^\dagger U = UU^\dagger = I$$

where $U^\dagger = (U^*)^T$ is the conjugate transpose (adjoint) of $U$.

This constraint comes from two physical requirements:
1. **Normalization preservation**: If $\langle\psi|\psi\rangle = 1$ before the gate, then $\langle\psi|U^\dagger U|\psi\rangle = \langle\psi|\psi\rangle = 1$ after.
2. **Reversibility**: The inverse of $U$ is $U^\dagger$, which is also unitary. Every quantum operation can be undone.

### 1.2 Properties of Unitary Matrices

For an $n \times n$ unitary matrix:

| Property | Meaning |
|----------|---------|
| $U^\dagger U = I$ | Columns form an orthonormal set |
| $\|U\|_\text{op} = 1$ | Does not change the length of any vector |
| $\|\det(U)\| = 1$ | Determinant has unit magnitude |
| Eigenvalues $= e^{i\lambda}$ | All eigenvalues lie on the unit circle |

### 1.3 How Many Parameters?

An $n \times n$ unitary matrix has $n^2$ real parameters. For quantum computing:
- Single-qubit gate ($n=2$): 4 parameters (but one is a global phase, leaving 3)
- Two-qubit gate ($n=4$): 16 parameters
- $k$-qubit gate ($n=2^k$): $4^k$ parameters

```python
import numpy as np

def is_unitary(M, tol=1e-10):
    """
    Check if a matrix is unitary: U^dagger @ U = I.

    Why check this? In quantum computing, every gate MUST be unitary.
    This is the fundamental constraint from quantum mechanics. A non-unitary
    operation would either create or destroy probability, which is unphysical.
    """
    n = M.shape[0]
    product = M.conj().T @ M
    return np.allclose(product, np.eye(n), atol=tol)

def apply_gate(gate, state):
    """
    Apply a quantum gate (matrix) to a state vector.

    Why matrix-vector multiplication? Quantum mechanics says time evolution
    is linear, so the map from input state to output state is a linear
    transformation -- exactly what matrices do.
    """
    new_state = gate @ state
    return new_state

# Example: verify that a random unitary is indeed unitary
# Why generate random unitaries this way? The QR decomposition of a random
# complex matrix gives a uniformly distributed (Haar random) unitary matrix.
rng = np.random.default_rng(42)
random_matrix = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
Q, R = np.linalg.qr(random_matrix)

print("=== Unitarity Check ===\n")
print(f"Random Q from QR decomposition:\n{Q}")
print(f"Is unitary? {is_unitary(Q)}")
print(f"Q^dagger @ Q =\n{Q.conj().T @ Q}")
print(f"\nDeterminant: {np.linalg.det(Q):.4f}")
print(f"|det| = {abs(np.linalg.det(Q)):.6f}")

# Non-unitary matrix for comparison
non_unitary = np.array([[1, 1], [0, 1]], dtype=complex)
print(f"\nNon-unitary matrix [[1,1],[0,1]]:")
print(f"Is unitary? {is_unitary(non_unitary)}")
```

---

## 2. Pauli Gates: X, Y, Z

The three Pauli matrices are the fundamental single-qubit gates. They correspond to 180-degree rotations around the $x$, $y$, and $z$ axes of the Bloch sphere.

### 2.1 Pauli-X (NOT Gate)

$$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

**Action**: Flips $|0\rangle \leftrightarrow |1\rangle$, just like the classical NOT gate.

$$X|0\rangle = |1\rangle, \quad X|1\rangle = |0\rangle$$

**Bloch sphere**: 180-degree rotation around the $x$-axis.

### 2.2 Pauli-Y

$$Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$$

**Action**: Flips with a phase.

$$Y|0\rangle = i|1\rangle, \quad Y|1\rangle = -i|0\rangle$$

**Bloch sphere**: 180-degree rotation around the $y$-axis.

### 2.3 Pauli-Z (Phase Flip)

$$Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

**Action**: Leaves $|0\rangle$ unchanged, flips the sign of $|1\rangle$.

$$Z|0\rangle = |0\rangle, \quad Z|1\rangle = -|1\rangle$$

**Bloch sphere**: 180-degree rotation around the $z$-axis.

### 2.4 Pauli Algebra

The Pauli matrices satisfy important algebraic relations:

$$X^2 = Y^2 = Z^2 = I$$

$$XY = iZ, \quad YZ = iX, \quad ZX = iY$$

$$\{X, Y\} = XY + YX = 0 \quad \text{(anticommutation)}$$

These relations are used constantly in quantum computing proofs and simplifications.

```python
import numpy as np

# Pauli gates

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)
ket_plus = (ket_0 + ket_1) / np.sqrt(2)

print("=== Pauli Gate Actions ===\n")

for name, gate in [("X", X), ("Y", Y), ("Z", Z)]:
    print(f"{name} gate:")
    print(f"  {name}|0> = {gate @ ket_0}")
    print(f"  {name}|1> = {gate @ ket_1}")
    print(f"  {name}|+> = {gate @ ket_plus}")
    print(f"  Is unitary? {is_unitary(gate)}")
    print()

# Verify Pauli algebra
print("=== Pauli Algebra ===\n")
print(f"X^2 = I? {np.allclose(X @ X, I)}")
print(f"Y^2 = I? {np.allclose(Y @ Y, I)}")
print(f"Z^2 = I? {np.allclose(Z @ Z, I)}")
print(f"XY = iZ? {np.allclose(X @ Y, 1j * Z)}")
print(f"YZ = iX? {np.allclose(Y @ Z, 1j * X)}")
print(f"ZX = iY? {np.allclose(Z @ X, 1j * Y)}")

# Anticommutation: {X,Y} = XY + YX = 0
print(f"\n{{X,Y}} = 0? {np.allclose(X @ Y + Y @ X, np.zeros((2,2)))}")
print(f"{{Y,Z}} = 0? {np.allclose(Y @ Z + Z @ Y, np.zeros((2,2)))}")
print(f"{{Z,X}} = 0? {np.allclose(Z @ X + X @ Z, np.zeros((2,2)))}")

# Eigenvalues (should all be +1 and -1 for Paulis)
print("\n=== Eigenvalues ===\n")
for name, gate in [("X", X), ("Y", Y), ("Z", Z)]:
    eigenvalues = np.linalg.eigvals(gate)
    print(f"Eigenvalues of {name}: {eigenvalues}")
```

---

## 3. The Hadamard Gate

The Hadamard gate $H$ is arguably the most important single-qubit gate in quantum computing. It creates superpositions from basis states and is used in nearly every quantum algorithm.

### 3.1 Matrix Form

$$H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

### 3.2 Action on Basis States

$$H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}} = |+\rangle$$

$$H|1\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}} = |-\rangle$$

In reverse:

$$H|+\rangle = |0\rangle, \quad H|-\rangle = |1\rangle$$

The Hadamard gate converts between the $Z$-basis and the $X$-basis.

### 3.3 Key Properties

- **Self-inverse**: $H^2 = I$ (applying it twice returns to the original state)
- **Relation to Paulis**: $H = \frac{X + Z}{\sqrt{2}}$, and $HXH = Z$, $HZH = X$
- **Bloch sphere**: 180-degree rotation around the axis halfway between $x$ and $z$ (the diagonal axis in the $xz$-plane)

### 3.4 Hadamard on n Qubits

Applying $H$ to each of $n$ qubits initialized to $|0\rangle$ creates a uniform superposition over all $2^n$ basis states:

$$H^{\otimes n}|0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{x=0}^{2^n-1} |x\rangle$$

This is the starting point for many quantum algorithms including Deutsch-Jozsa ([Lesson 7](07_Deutsch_Jozsa_Algorithm.md)) and Grover's search ([Lesson 8](08_Grovers_Search.md)).

More generally, the Hadamard acting on a single basis state $|x\rangle$ where $x \in \{0, 1\}$ gives:

$$H|x\rangle = \frac{1}{\sqrt{2}}\sum_{y=0}^{1}(-1)^{xy}|y\rangle$$

This generalizes to $n$ qubits:

$$H^{\otimes n}|x\rangle = \frac{1}{\sqrt{2^n}}\sum_{y=0}^{2^n-1}(-1)^{x \cdot y}|y\rangle$$

where $x \cdot y$ denotes the bitwise inner product modulo 2.

```python
import numpy as np

# The Hadamard gate

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

print("=== Hadamard Gate ===\n")
print(f"H|0> = {H @ ket_0}  (= |+>)")
print(f"H|1> = {H @ ket_1}  (= |->)")
print(f"H|+> = {H @ (ket_0 + ket_1)/np.sqrt(2)}  (= |0>)")
print(f"\nH^2 = I? {np.allclose(H @ H, np.eye(2))}")

# Hadamard on n qubits: creates uniform superposition
def hadamard_n(n):
    """
    Create the n-qubit Hadamard gate H^{tensor n}.

    Why tensor product? When gates act independently on separate qubits,
    the combined operation is the tensor product of individual gates.
    H^n creates a uniform superposition from |00...0>, which is the
    starting step of most quantum algorithms.
    """
    H_n = np.array([[1]], dtype=complex)
    for _ in range(n):
        H_n = np.kron(H_n, H)
    return H_n

# Demonstrate uniform superposition for n = 1, 2, 3
for n in range(1, 4):
    H_n = hadamard_n(n)
    initial = np.zeros(2**n, dtype=complex)
    initial[0] = 1  # |00...0>
    result = H_n @ initial

    print(f"\nH^{n} |{'0'*n}> = equal superposition of {2**n} states")
    print(f"  Amplitudes: {result}")
    print(f"  Each amplitude = 1/sqrt({2**n}) = {1/np.sqrt(2**n):.4f}")
    print(f"  Each probability = 1/{2**n} = {1/2**n:.4f}")

# Verify the (-1)^{x.y} formula for H|x>
print("\n=== Hadamard Action Formula: H|x> = (1/sqrt(2)) sum_y (-1)^{x*y} |y> ===\n")

for x in [0, 1]:
    ket_x = np.array([1-x, x], dtype=complex)
    result = H @ ket_x
    print(f"H|{x}> = {result}")
    for y in [0, 1]:
        expected_amp = (-1)**(x*y) / np.sqrt(2)
        print(f"  Coefficient of |{y}>: computed={result[y]:.4f}, "
              f"formula=(-1)^({x}*{y})/sqrt(2)={expected_amp:.4f}")
```

---

## 4. Phase Gates: S and T

Phase gates modify the relative phase between $|0\rangle$ and $|1\rangle$ without changing measurement probabilities in the computational basis.

### 4.1 General Phase Gate

$$R_\phi = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\phi} \end{pmatrix}$$

This leaves $|0\rangle$ unchanged and multiplies $|1\rangle$ by $e^{i\phi}$.

### 4.2 The S Gate (Phase Gate, $\sqrt{Z}$)

$$S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix} = R_{\pi/2}$$

Properties:
- $S^2 = Z$ (hence "$\sqrt{Z}$")
- $S|+\rangle = |i\rangle$: rotates from the $x$-axis to the $y$-axis on the Bloch equator
- **Bloch sphere**: 90-degree rotation around the $z$-axis

### 4.3 The T Gate ($\pi/8$ Gate, $\sqrt{S}$)

$$T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix} = R_{\pi/4}$$

Properties:
- $T^2 = S$, $T^4 = Z$ (hence "$\sqrt[4]{Z}$")
- The T gate is called the "$\pi/8$ gate" for historical reasons (it equals $e^{i\pi/8} R_z(\pi/4)$, which has $e^{i\pi/8}$ as a global phase factor)
- **Critical role**: The T gate is what makes quantum computing universal when combined with H and CNOT (Section 7)
- **Bloch sphere**: 45-degree rotation around the $z$-axis

### 4.4 The Gate Hierarchy

$$T \xrightarrow{T^2} S \xrightarrow{S^2} Z \xrightarrow{Z^2} I$$

Each gate is the "square root" of the next.

```python
import numpy as np

# Phase gates

S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

print("=== Phase Gate Hierarchy ===\n")
print(f"T^2 = S? {np.allclose(T @ T, S)}")
print(f"S^2 = Z? {np.allclose(S @ S, Z)}")
print(f"Z^2 = I? {np.allclose(Z @ Z, I)}")
print(f"T^4 = Z? {np.allclose(T @ T @ T @ T, Z)}")
print(f"T^8 = I? {np.allclose(np.linalg.matrix_power(T, 8), I)}")

# Action on |+> state
ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)

print("\n=== Phase Gates Acting on |+> ===\n")
print(f"|+> = {ket_plus}")

for name, gate in [("T", T), ("S", S), ("Z", Z)]:
    result = gate @ ket_plus
    # Compute the phase angle of the |1> component relative to |0>
    rel_phase = np.angle(result[1]) - np.angle(result[0])
    print(f"{name}|+> = {result}")
    print(f"  Relative phase = {rel_phase:.4f} rad = {rel_phase/np.pi:.4f}*pi")
    print(f"  P(0) = {abs(result[0])**2:.4f}, P(1) = {abs(result[1])**2:.4f}")
    print()

print("Key insight: Phase gates don't change P(0) or P(1) in the Z-basis,")
print("but they rotate the state around the equator of the Bloch sphere.")
print("The phase change becomes visible when measuring in the X or Y basis.")
```

---

## 5. Rotation Gates

### 5.1 General Rotation

Any single-qubit unitary can be decomposed as a rotation around an axis on the Bloch sphere. The rotation gates around the three principal axes are:

$$R_x(\theta) = e^{-i\theta X/2} = \cos\frac{\theta}{2}\,I - i\sin\frac{\theta}{2}\,X = \begin{pmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

$$R_y(\theta) = e^{-i\theta Y/2} = \cos\frac{\theta}{2}\,I - i\sin\frac{\theta}{2}\,Y = \begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

$$R_z(\theta) = e^{-i\theta Z/2} = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

### 5.2 Connecting Rotations to Named Gates

The named gates are special cases of rotations:

| Gate | Rotation | Angle |
|------|----------|-------|
| $X$ | $R_x(\pi)$ | $180°$ around $x$ (up to global phase) |
| $Y$ | $R_y(\pi)$ | $180°$ around $y$ (up to global phase) |
| $Z$ | $R_z(\pi)$ | $180°$ around $z$ (up to global phase) |
| $H$ | $R_y(\pi/2) \cdot R_z(\pi)$ | Combined rotation |
| $S$ | $R_z(\pi/2)$ | $90°$ around $z$ (up to global phase) |
| $T$ | $R_z(\pi/4)$ | $45°$ around $z$ (up to global phase) |

Note: "Up to global phase" means the rotation matrix may differ from the named gate by a factor of $e^{i\alpha}$, which is physically irrelevant.

### 5.3 Euler Decomposition

Any single-qubit gate $U$ can be written as:

$$U = e^{i\alpha} R_z(\beta) R_y(\gamma) R_z(\delta)$$

for some real angles $\alpha, \beta, \gamma, \delta$. This is the quantum analog of the Euler angle decomposition from classical mechanics. It means any single-qubit operation can be built from just $R_z$ and $R_y$ rotations.

```python
import numpy as np

# Rotation gates

def Rx(theta):
    """Rotation around x-axis by angle theta."""
    return np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)

def Ry(theta):
    """Rotation around y-axis by angle theta."""
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)

def Rz(theta):
    """
    Rotation around z-axis by angle theta.

    Why does Rz have this form? The Z-axis on the Bloch sphere corresponds
    to the computational basis. Rz simply applies a relative phase between
    |0> and |1>, which is a rotation around the Z-axis.
    """
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ], dtype=complex)

# Verify named gates are special cases of rotations
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)

print("=== Rotation Gate Verification ===\n")
print("Checking named gates as special rotations (up to global phase):\n")

# Why check "up to global phase"? Two unitary matrices that differ by a
# global phase e^{i*alpha} represent the same physical operation.
def same_up_to_phase(A, B):
    """Check if A = e^{i*alpha} * B for some alpha."""
    # If A = e^{i*alpha} * B, then A @ B.conj().T should be proportional to I
    ratio = A @ np.linalg.inv(B)
    phase = ratio[0, 0]
    return np.allclose(ratio, phase * np.eye(2))

print(f"X = Rx(pi) up to phase? {same_up_to_phase(X, Rx(np.pi))}")
print(f"Z = Rz(pi) up to phase? {same_up_to_phase(Z, Rz(np.pi))}")
print(f"S = Rz(pi/2) up to phase? {same_up_to_phase(S, Rz(np.pi/2))}")
print(f"T = Rz(pi/4) up to phase? {same_up_to_phase(T, Rz(np.pi/4))}")

# Demonstrate rotation with varying angles
print("\n=== Rz(theta) acting on |+> for various theta ===\n")
ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)

for angle_frac, label in [(0, "0"), (1/4, "pi/4"), (1/2, "pi/2"),
                           (1, "pi"), (3/2, "3pi/2")]:
    theta = angle_frac * np.pi
    result = Rz(theta) @ ket_plus
    rel_phase = np.angle(result[1]) - np.angle(result[0])
    print(f"Rz({label})|+> = [{result[0]:.4f}, {result[1]:.4f}], "
          f"relative phase = {rel_phase/np.pi:.3f}*pi")
```

---

## 6. Two-Qubit Gates

Single-qubit gates alone cannot create entanglement or perform universal quantum computation. We need gates that act on two (or more) qubits simultaneously.

### 6.1 CNOT (Controlled-NOT)

The CNOT gate is the most important two-qubit gate. It has a **control** qubit and a **target** qubit:

$$\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

**Action**: If the control is $|0\rangle$, do nothing. If the control is $|1\rangle$, apply $X$ to the target.

$$\text{CNOT}|00\rangle = |00\rangle, \quad \text{CNOT}|01\rangle = |01\rangle$$
$$\text{CNOT}|10\rangle = |11\rangle, \quad \text{CNOT}|11\rangle = |10\rangle$$

In short: $\text{CNOT}|a, b\rangle = |a, a \oplus b\rangle$ where $\oplus$ is XOR.

### 6.2 CZ (Controlled-Z)

$$\text{CZ} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}$$

**Action**: Apply $Z$ to the target if the control is $|1\rangle$. Equivalently, apply a phase of $-1$ to the $|11\rangle$ component only.

**Symmetric**: Unlike CNOT, CZ is symmetric -- swapping control and target gives the same gate. This is because $Z$ is diagonal.

### 6.3 SWAP Gate

$$\text{SWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

**Action**: Swaps the states of two qubits: $\text{SWAP}|a, b\rangle = |b, a\rangle$.

**Decomposition**: SWAP can be built from three CNOTs:

$$\text{SWAP} = \text{CNOT}_{12} \cdot \text{CNOT}_{21} \cdot \text{CNOT}_{12}$$

### 6.4 Controlled-U

Any single-qubit gate $U$ can be turned into a controlled gate:

$$C\text{-}U = \begin{pmatrix} I & 0 \\ 0 & U \end{pmatrix} = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes U$$

If the control qubit is $|0\rangle$, the target is unchanged. If the control is $|1\rangle$, $U$ is applied to the target.

```python
import numpy as np

# Two-qubit gates

# CNOT gate (control=qubit 0, target=qubit 1)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

# CZ gate
CZ = np.diag([1, 1, 1, -1]).astype(complex)

# SWAP gate
SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

# Basis states for 2 qubits
basis_2q = {
    "|00>": np.array([1, 0, 0, 0], dtype=complex),
    "|01>": np.array([0, 1, 0, 0], dtype=complex),
    "|10>": np.array([0, 0, 1, 0], dtype=complex),
    "|11>": np.array([0, 0, 0, 1], dtype=complex),
}

print("=== CNOT Action ===\n")
for label, state in basis_2q.items():
    result = CNOT @ state
    # Find the basis state label for the result
    for rlabel, rstate in basis_2q.items():
        if np.allclose(result, rstate):
            print(f"CNOT{label} = {rlabel}")
            break

print("\n=== CZ Action ===\n")
for label, state in basis_2q.items():
    result = CZ @ state
    for rlabel, rstate in basis_2q.items():
        if np.allclose(result, rstate):
            print(f"CZ{label} = {rlabel}")
            break
        if np.allclose(result, -rstate):
            print(f"CZ{label} = -{rlabel}")
            break

# Verify SWAP = CNOT_12 * CNOT_21 * CNOT_12
# Why does this work? Each CNOT transfers information one direction.
# Three alternating CNOTs effectively swap the two qubits, similar
# to the classical trick: a ^= b; b ^= a; a ^= b;
print("\n=== SWAP Decomposition ===\n")

# CNOT with control=0, target=1 (standard)
CNOT_01 = CNOT.copy()

# CNOT with control=1, target=0 (reversed)
# This is SWAP * CNOT * SWAP, but we can write it directly
CNOT_10 = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0]
], dtype=complex)

swap_from_cnots = CNOT_01 @ CNOT_10 @ CNOT_01
print(f"SWAP = CNOT_01 * CNOT_10 * CNOT_01? {np.allclose(swap_from_cnots, SWAP)}")

# Creating entanglement with CNOT!
print("\n=== Creating Entanglement ===\n")
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Apply H to qubit 0, then CNOT
# H tensor I
H_I = np.kron(H, np.eye(2, dtype=complex))

# Start from |00>
state = np.array([1, 0, 0, 0], dtype=complex)
state = H_I @ state       # Apply H to qubit 0 -> (|00> + |10>)/sqrt(2)
state = CNOT @ state      # CNOT -> (|00> + |11>)/sqrt(2) = Bell state!

print(f"H(x)I |00>  then CNOT:")
for label, bstate in basis_2q.items():
    amp = np.vdot(bstate, state)
    if abs(amp) > 1e-10:
        print(f"  {label}: amplitude = {amp:.4f}, P = {abs(amp)**2:.4f}")

print("\nThis is the Bell state |Phi+> = (|00> + |11>)/sqrt(2)!")
print("We will study Bell states in Lesson 5.")
```

---

## 7. Universal Gate Sets

### 7.1 What Is Universality?

A set of quantum gates is **universal** if any unitary operation on any number of qubits can be approximated to arbitrary precision using a finite sequence of gates from the set.

This is analogous to the classical result that NAND gates alone are universal for classical computation.

### 7.2 The Standard Universal Set: {H, T, CNOT}

The set $\{H, T, \text{CNOT}\}$ is universal for quantum computing. Here is why each gate is needed:

| Gate | Role | What it provides |
|------|------|-----------------|
| $H$ | Creates superpositions | Access to the $X$-basis; combined with $T$, enables arbitrary $z$-rotations |
| $T$ | Fine-grained phase | Combined with $H$, approximates arbitrary single-qubit gates |
| CNOT | Two-qubit entanglement | Cannot create entanglement with single-qubit gates alone |

### 7.3 Why T and Not Just S?

The $S$ gate gives $\pi/2$ rotation around $z$. Combined with $H$, we can only reach a finite set of states (the Clifford group). Adding $T$ ($\pi/4$ rotation) breaks out of this limitation and allows approximation of *any* single-qubit rotation.

The **Solovay-Kitaev theorem** guarantees that any single-qubit gate can be approximated to precision $\epsilon$ using $O(\log^c(1/\epsilon))$ gates from $\{H, T\}$ (where $c \approx 3.97$). This is remarkably efficient.

### 7.4 Other Universal Sets

Several other gate sets are also universal:

- $\{H, T, \text{CNOT}\}$ -- Standard set
- $\{H, \text{Toffoli}\}$ -- Hadamard + 3-qubit gate
- $\{R_y(\theta), R_z(\phi), \text{CNOT}\}$ for irrational $\theta/\pi$ -- Continuous rotation set
- $\{\text{CNOT}, \text{any non-Clifford single-qubit gate}\}$

```python
import numpy as np

# Demonstrating universality: approximating arbitrary gates with H, T

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
T_dag = T.conj().T  # T-dagger (inverse of T)

# Key identity: HT^k H generates rotations that, when composed,
# can approximate any single-qubit gate

print("=== Building Gates from H and T ===\n")

# Some useful compositions
S = T @ T  # S = T^2
Z = S @ S  # Z = S^2 = T^4

print(f"T^2 = S? {np.allclose(T @ T, np.array([[1,0],[0,1j]]))}")
print(f"T^4 = Z? {np.allclose(np.linalg.matrix_power(T, 4), np.array([[1,0],[0,-1]]))}")

# HTH gives a rotation in a different direction
HTH = H @ T @ H
print(f"\nHTH (a rotation mixing X and Z):\n{HTH}")
print(f"Is unitary? {np.allclose(HTH @ HTH.conj().T, np.eye(2))}")

# Approximating Rx(pi/8) using H and T sequences
# The Solovay-Kitaev theorem guarantees this is possible
# Here we show that composing HT sequences creates diverse rotations

gates_from_HT = {
    "H": H,
    "T": T,
    "HT": H @ T,
    "TH": T @ H,
    "HTH": H @ T @ H,
    "THT": T @ H @ T,
    "HTHT": H @ T @ H @ T,
    "THTH": T @ H @ T @ H,
}

print("\n=== Diverse Gates from H, T Compositions ===\n")
print(f"{'Sequence':<10} {'Eigenvalues':>30} {'Det':>15}")
print("-" * 60)

for name, gate in gates_from_HT.items():
    eigs = np.linalg.eigvals(gate)
    det = np.linalg.det(gate)
    print(f"{name:<10} {str(np.round(eigs, 4)):>30} {det:>15.4f}")

print("\nThe diverse eigenvalues show that H and T compositions")
print("reach many different rotations -- the basis of universality.")
```

---

## 8. Gate Decomposition

### 8.1 ZYZ Decomposition

Any single-qubit unitary $U$ can be decomposed as:

$$U = e^{i\alpha} R_z(\beta) R_y(\gamma) R_z(\delta)$$

This uses only 4 real parameters, matching the 4 degrees of freedom of a 2x2 unitary matrix (3 for the $SU(2)$ part + 1 global phase).

### 8.2 ABC Decomposition for Controlled Gates

Any controlled-$U$ gate can be decomposed into single-qubit gates and at most 2 CNOTs using the identity:

$$C\text{-}U = (I \otimes A) \cdot \text{CNOT} \cdot (I \otimes B) \cdot \text{CNOT} \cdot (I \otimes C)$$

where $ABC = I$ and $e^{i\alpha} AXBXC = U$ for appropriate matrices $A, B, C$ and phase $\alpha$.

### 8.3 Practical Example

Let us decompose the controlled-$S$ gate using the fact that $S = T^2$ and $T$ is a $\pi/4$ rotation around $z$.

```python
import numpy as np

# Gate decomposition examples

def Rz(theta):
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ], dtype=complex)

def Ry(theta):
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)

def zyz_decompose(U):
    """
    Decompose a single-qubit unitary into Rz-Ry-Rz form.

    Why ZYZ? Any rotation in 3D can be decomposed into three rotations
    about two fixed axes (Euler angles). In quantum computing, we use
    the Z-Y-Z convention because Rz gates are often 'free' (implemented
    by adjusting control software timing) on many hardware platforms.
    """
    # Extract global phase
    det = np.linalg.det(U)
    alpha = np.angle(det) / 2
    # Remove global phase to get SU(2) matrix
    V = U * np.exp(-1j * alpha)

    # For SU(2): V = [[a, -b*], [b, a*]] with |a|^2 + |b|^2 = 1
    a = V[0, 0]
    b = V[1, 0]

    # gamma = 2*arccos(|a|)
    gamma = 2 * np.arccos(np.clip(abs(a), -1, 1))

    if np.isclose(gamma, 0):
        beta = np.angle(a) * 2
        delta = 0
    elif np.isclose(gamma, np.pi):
        beta = np.angle(b) * 2
        delta = 0
    else:
        beta = np.angle(a) - np.angle(b)  # Phase of |0>-like component
        delta = np.angle(a) + np.angle(b)  # Phase of |1>-like component

    return alpha, beta, gamma, delta

# Test with Hadamard gate
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
alpha, beta, gamma, delta = zyz_decompose(H)

print("=== ZYZ Decomposition of Hadamard ===\n")
print(f"H = exp(i*{alpha/np.pi:.4f}*pi) * Rz({beta/np.pi:.4f}*pi) "
      f"* Ry({gamma/np.pi:.4f}*pi) * Rz({delta/np.pi:.4f}*pi)")

# Verify
H_reconstructed = np.exp(1j * alpha) * Rz(beta) @ Ry(gamma) @ Rz(delta)
print(f"\nOriginal H:\n{H}")
print(f"\nReconstructed:\n{np.round(H_reconstructed, 4)}")
print(f"Match? {np.allclose(H, H_reconstructed)}")

# Decompose a random single-qubit gate
print("\n=== ZYZ Decomposition of Random Gate ===\n")
rng = np.random.default_rng(123)
random_matrix = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
Q, R = np.linalg.qr(random_matrix)
U_random = Q

alpha, beta, gamma, delta = zyz_decompose(U_random)
U_reconstructed = np.exp(1j * alpha) * Rz(beta) @ Ry(gamma) @ Rz(delta)

print(f"Parameters: alpha={alpha:.4f}, beta={beta:.4f}, "
      f"gamma={gamma:.4f}, delta={delta:.4f}")
print(f"Match? {np.allclose(U_random, U_reconstructed)}")
```

---

## 9. Exercises

### Exercise 1: Gate Verification

Verify by explicit matrix multiplication that:
a) $HZH = X$ (the Hadamard conjugates $Z$ to $X$)
b) $HXH = Z$
c) $SXS^\dagger = Y$ (up to phase)

Write Python code to check each identity.

### Exercise 2: Bloch Sphere Rotations

a) Starting from $|0\rangle$, apply $R_y(\pi/4)$. What is the resulting state? Where is it on the Bloch sphere?
b) Starting from $|+\rangle$, apply $R_z(\pi/2)$. What is the resulting state?
c) Verify that $R_x(\theta)R_y(\theta)$ is NOT the same as $R_y(\theta)R_x(\theta)$ for $\theta = \pi/4$. Quantum gates generally do not commute!

### Exercise 3: Controlled Gate Construction

Construct the Controlled-$H$ gate as a 4x4 matrix. Verify that:
a) $CH|00\rangle = |00\rangle$
b) $CH|10\rangle = |1\rangle \otimes H|0\rangle = |1,+\rangle$

### Exercise 4: Gate Count

How many gates from $\{H, T, \text{CNOT}\}$ are needed to exactly implement each of the following?
a) $X$ gate
b) $S$ gate
c) SWAP gate
d) Toffoli gate (Hint: the Toffoli requires 6 CNOTs and several single-qubit gates)

### Exercise 5: Custom Gate

Design a single-qubit gate that maps $|0\rangle \to \frac{1}{\sqrt{3}}|0\rangle + \sqrt{\frac{2}{3}}|1\rangle$ and find its ZYZ decomposition. Verify that the full $2 \times 2$ unitary matrix is consistent with this mapping. (Remember: you also need to specify what happens to $|1\rangle$ for the gate to be fully defined.)

---

[<- Previous: Qubits and the Bloch Sphere](02_Qubits_and_Bloch_Sphere.md) | [Next: Quantum Circuits ->](04_Quantum_Circuits.md)
