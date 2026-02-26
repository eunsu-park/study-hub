# Lesson 2: Qubits and the Bloch Sphere

[<- Previous: Quantum Mechanics Primer](01_Quantum_Mechanics_Primer.md) | [Next: Quantum Gates ->](03_Quantum_Gates.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Represent a qubit state as $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ and explain the normalization constraint
2. Distinguish between global phase and relative phase, and explain why only relative phase is observable
3. Map any single-qubit state to a point on the Bloch sphere using the $(\theta, \phi)$ parametrization
4. Work with multiple measurement bases including $\{|+\rangle, |-\rangle\}$ and $\{|i\rangle, |-i\rangle\}$
5. Construct multi-qubit states using tensor products
6. Calculate the dimension of an $n$-qubit state space
7. Implement qubit states and Bloch sphere coordinates in Python

---

The qubit is the fundamental unit of quantum information, analogous to the classical bit. But whereas a classical bit can be either 0 or 1, a qubit can exist in any superposition of $|0\rangle$ and $|1\rangle$, parameterized by two complex numbers. Understanding how to represent, visualize, and manipulate qubit states is the essential starting point for all of quantum computing.

In this lesson, we develop a complete mathematical framework for single-qubit and multi-qubit states. The Bloch sphere provides a beautiful geometric visualization of single-qubit states, making abstract quantum states tangible and building intuition for quantum gates (which we will study in the next lesson).

> **Analogy:** The Bloch sphere is like a globe for quantum states -- latitude determines how much $|0\rangle$ vs $|1\rangle$ (the "north pole" is $|0\rangle$, the "south pole" is $|1\rangle$), and longitude determines the phase relationship between them. Just as any point on Earth can be specified by latitude and longitude, any single-qubit state can be specified by two angles on the Bloch sphere.

## Table of Contents

1. [The Qubit State](#1-the-qubit-state)
2. [Global Phase vs Relative Phase](#2-global-phase-vs-relative-phase)
3. [The Bloch Sphere](#3-the-bloch-sphere)
4. [Important Bases](#4-important-bases)
5. [Multi-Qubit Systems](#5-multi-qubit-systems)
6. [N-Qubit State Space](#6-n-qubit-state-space)
7. [Exercises](#7-exercises)

---

## 1. The Qubit State

### 1.1 Mathematical Definition

A **qubit** (quantum bit) is a two-level quantum system. Its state is a vector in a 2-dimensional complex Hilbert space $\mathbb{C}^2$:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle = \begin{pmatrix} \alpha \\ \beta \end{pmatrix}$$

where $\alpha, \beta \in \mathbb{C}$ are complex **amplitudes** subject to the normalization constraint:

$$|\alpha|^2 + |\beta|^2 = 1$$

This normalization ensures the total probability of all measurement outcomes sums to 1 (Born rule from [Lesson 1](01_Quantum_Mechanics_Primer.md)).

### 1.2 What Does "Two-Level" Mean?

Physically, a qubit can be realized by any quantum system with two distinguishable states:

| Physical System | $\|0\rangle$ | $\|1\rangle$ |
|----------------|-------------|-------------|
| Photon polarization | Horizontal | Vertical |
| Electron spin | Spin up | Spin down |
| Superconducting circuit | Ground state | Excited state |
| Trapped ion | Ground energy level | Excited energy level |
| Quantum dot | No electron | One electron |

The mathematics is identical regardless of the physical implementation. This abstraction is one of the great strengths of quantum computing theory.

### 1.3 Degrees of Freedom

The state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ has two complex numbers = 4 real parameters. But:
- Normalization removes 1 degree of freedom: $|\alpha|^2 + |\beta|^2 = 1$
- Global phase removes 1 more (as we will see in the next section)

This leaves **2 real parameters** to specify a qubit state -- exactly the two angles $(\theta, \phi)$ on the Bloch sphere.

```python
import numpy as np

# Creating qubit states

# Basis states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

def make_qubit(alpha, beta):
    """
    Create a qubit state |psi> = alpha|0> + beta|1>.

    Why check normalization? A non-normalized state would give probabilities
    that don't sum to 1, which is physically meaningless. This is the
    fundamental constraint on all quantum states.
    """
    state = np.array([alpha, beta], dtype=complex)
    norm = np.linalg.norm(state)
    if not np.isclose(norm, 1.0):
        print(f"  Warning: State not normalized (norm = {norm:.6f}). Normalizing...")
        state = state / norm
    return state

# Example states
print("=== Various Qubit States ===\n")

states = {
    "|0>": make_qubit(1, 0),
    "|1>": make_qubit(0, 1),
    "|+>": make_qubit(1/np.sqrt(2), 1/np.sqrt(2)),
    "|->": make_qubit(1/np.sqrt(2), -1/np.sqrt(2)),
    "|i>": make_qubit(1/np.sqrt(2), 1j/np.sqrt(2)),
    "custom": make_qubit(1/np.sqrt(3), np.sqrt(2/3) * np.exp(1j * np.pi/4)),
}

for name, state in states.items():
    p0 = abs(state[0])**2
    p1 = abs(state[1])**2
    print(f"{name:>10}: [{state[0]:.4f}, {state[1]:.4f}]  "
          f"P(0)={p0:.4f}, P(1)={p1:.4f}")

# A deliberately non-normalized state (will be corrected)
print("\n--- Non-normalized state ---")
bad_state = make_qubit(1, 1)  # Sum of squares = 2, not 1
print(f"Normalized: [{bad_state[0]:.4f}, {bad_state[1]:.4f}]")
```

---

## 2. Global Phase vs Relative Phase

Understanding the distinction between global and relative phase is crucial for quantum computing. It explains why a qubit has only 2 real degrees of freedom, not 3.

### 2.1 Global Phase Is Unobservable

Two states that differ by an overall (global) phase factor $e^{i\gamma}$ are **physically identical**:

$$|\psi\rangle \equiv e^{i\gamma}|\psi\rangle \quad \text{for all } \gamma \in \mathbb{R}$$

Why? Because all observable quantities depend on $|\langle\phi|\psi\rangle|^2$, and:

$$|\langle\phi|e^{i\gamma}\psi\rangle|^2 = |e^{i\gamma}|^2 \cdot |\langle\phi|\psi\rangle|^2 = |\langle\phi|\psi\rangle|^2$$

Since $|e^{i\gamma}| = 1$ always.

**Practical consequence**: We can always choose $\alpha$ to be real and non-negative without loss of generality. If $\alpha = |\alpha|e^{i\gamma}$, we multiply the entire state by $e^{-i\gamma}$:

$$|\psi\rangle = |\alpha|e^{i\gamma}|0\rangle + \beta|1\rangle \equiv |\alpha||0\rangle + \beta e^{-i\gamma}|1\rangle$$

### 2.2 Relative Phase Is Observable

The phase *between* components of a superposition is physically meaningful. Consider:

$$|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle), \quad |-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

These states have the same measurement probabilities in the $\{|0\rangle, |1\rangle\}$ basis (50/50 each). But they are physically different states! The relative phase (0 vs $\pi$) between the $|0\rangle$ and $|1\rangle$ components is different.

How can we tell them apart? By measuring in a *different* basis. In the $\{|+\rangle, |-\rangle\}$ basis:
- $|+\rangle$ gives $|+\rangle$ with probability 1
- $|-\rangle$ gives $|-\rangle$ with probability 1

The relative phase has real physical consequences -- it just requires the right measurement to reveal them.

```python
import numpy as np

# Demonstrating global vs relative phase

print("=== Global Phase: Unobservable ===\n")

# State 1: |psi> = (1/sqrt(2))|0> + (1/sqrt(2))|1>
psi1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)

# State 2: e^(i*pi/4) * |psi>  (global phase of pi/4)
gamma = np.pi / 4
psi2 = np.exp(1j * gamma) * psi1

print(f"State 1: {psi1}")
print(f"State 2: {psi2}  (State 1 * e^(i*pi/4))")
print(f"\nMeasurement probabilities in {{|0>, |1>}} basis:")
print(f"  State 1: P(0) = {abs(psi1[0])**2:.4f}, P(1) = {abs(psi1[1])**2:.4f}")
print(f"  State 2: P(0) = {abs(psi2[0])**2:.4f}, P(1) = {abs(psi2[1])**2:.4f}")
print("  => Identical! Global phase has no effect on measurement.")

# Also check in the {|+>, |->} basis
ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
ket_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)

# Why use inner product for basis change? The probability of outcome |phi>
# given state |psi> is always |<phi|psi>|^2, regardless of basis
p_plus_1 = abs(np.vdot(ket_plus, psi1))**2
p_plus_2 = abs(np.vdot(ket_plus, psi2))**2
print(f"\nIn {{|+>, |->}} basis:")
print(f"  State 1: P(+) = {p_plus_1:.4f}")
print(f"  State 2: P(+) = {p_plus_2:.4f}")
print("  => Still identical in any basis!")

print("\n\n=== Relative Phase: Observable ===\n")

# |+> = (|0> + |1>)/sqrt(2)         relative phase = 0
# |-> = (|0> - |1>)/sqrt(2)         relative phase = pi
# |i> = (|0> + i|1>)/sqrt(2)        relative phase = pi/2

states = {
    "|+> (phase=0)": np.array([1, 1], dtype=complex) / np.sqrt(2),
    "|-> (phase=pi)": np.array([1, -1], dtype=complex) / np.sqrt(2),
    "|i> (phase=pi/2)": np.array([1, 1j], dtype=complex) / np.sqrt(2),
}

print("Measurement in {|0>, |1>} basis (relative phase invisible):")
for name, state in states.items():
    print(f"  {name}: P(0) = {abs(state[0])**2:.4f}, P(1) = {abs(state[1])**2:.4f}")

print("\nMeasurement in {|+>, |->} basis (relative phase revealed!):")
for name, state in states.items():
    p_p = abs(np.vdot(ket_plus, state))**2
    p_m = abs(np.vdot(ket_minus, state))**2
    print(f"  {name}: P(+) = {p_p:.4f}, P(-) = {p_m:.4f}")
```

---

## 3. The Bloch Sphere

### 3.1 Parametrization

Since a single-qubit pure state has 2 real degrees of freedom, we can map it to a point on the surface of a unit sphere -- the **Bloch sphere**.

Using the convention that $\alpha$ is real and non-negative, any qubit state can be written as:

$$|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle$$

where:
- $\theta \in [0, \pi]$ is the **polar angle** (colatitude from the north pole)
- $\phi \in [0, 2\pi)$ is the **azimuthal angle** (longitude)

### 3.2 Why $\theta/2$ Instead of $\theta$?

The factor of $1/2$ is not arbitrary. It arises because:

1. **Orthogonal states map to antipodal points**: $|0\rangle$ and $|1\rangle$ are orthogonal in Hilbert space but correspond to the north and south poles ($\theta = 0$ and $\theta = \pi$), which are diametrically opposite. Without the $1/2$, we could not achieve this mapping.

2. **Full coverage**: $\theta$ ranges from $0$ to $\pi$, so $\theta/2$ ranges from $0$ to $\pi/2$, which means $\cos(\theta/2)$ ranges from 1 to 0 and $\sin(\theta/2)$ ranges from 0 to 1. This covers all possible amplitude distributions.

### 3.3 Special Points on the Bloch Sphere

| State | $\theta$ | $\phi$ | Bloch Coordinates $(x, y, z)$ |
|-------|:---:|:---:|:---:|
| $\|0\rangle$ | $0$ | any | $(0, 0, 1)$ — North pole |
| $\|1\rangle$ | $\pi$ | any | $(0, 0, -1)$ — South pole |
| $\|+\rangle = \frac{\|0\rangle + \|1\rangle}{\sqrt{2}}$ | $\pi/2$ | $0$ | $(1, 0, 0)$ — Positive $x$ |
| $\|-\rangle = \frac{\|0\rangle - \|1\rangle}{\sqrt{2}}$ | $\pi/2$ | $\pi$ | $(-1, 0, 0)$ — Negative $x$ |
| $\|i\rangle = \frac{\|0\rangle + i\|1\rangle}{\sqrt{2}}$ | $\pi/2$ | $\pi/2$ | $(0, 1, 0)$ — Positive $y$ |
| $\|-i\rangle = \frac{\|0\rangle - i\|1\rangle}{\sqrt{2}}$ | $\pi/2$ | $3\pi/2$ | $(0, -1, 0)$ — Negative $y$ |

### 3.4 Cartesian Coordinates

The Bloch sphere coordinates $(x, y, z)$ are related to the angles by:

$$x = \sin\theta\cos\phi, \quad y = \sin\theta\sin\phi, \quad z = \cos\theta$$

These coordinates also correspond to the expectation values of the **Pauli matrices** (which we will meet in [Lesson 3](03_Quantum_Gates.md)):

$$x = \langle\psi|X|\psi\rangle, \quad y = \langle\psi|Y|\psi\rangle, \quad z = \langle\psi|Z|\psi\rangle$$

### 3.5 Geometric Interpretation of Measurement

Measurement in the computational basis ($\{|0\rangle, |1\rangle\}$) corresponds to projecting the Bloch vector onto the $z$-axis:

- $P(0) = \cos^2(\theta/2) = (1 + z)/2$
- $P(1) = \sin^2(\theta/2) = (1 - z)/2$

States near the north pole ($z \approx 1$) are "mostly $|0\rangle$," and states near the south pole ($z \approx -1$) are "mostly $|1\rangle$." States on the equator ($z = 0$) are perfectly balanced superpositions.

```python
import numpy as np

def qubit_to_bloch(state):
    """
    Convert a qubit state vector to Bloch sphere coordinates.

    Why extract theta and phi this way? We use the standard parametrization
    |psi> = cos(theta/2)|0> + e^(i*phi)*sin(theta/2)|1>, first removing
    the global phase to make alpha real and non-negative.

    Returns (theta, phi, x, y, z).
    """
    alpha, beta = state[0], state[1]

    # Remove global phase: make alpha real and non-negative
    if abs(alpha) > 1e-10:
        phase = np.exp(-1j * np.angle(alpha))
        alpha = alpha * phase
        beta = beta * phase

    # Extract theta from |alpha| = cos(theta/2)
    alpha_real = np.real(alpha)  # Now alpha is real
    alpha_real = np.clip(alpha_real, -1, 1)  # Numerical safety
    theta = 2 * np.arccos(alpha_real)

    # Extract phi from beta = e^(i*phi) * sin(theta/2)
    if abs(beta) > 1e-10:
        phi = np.angle(beta)
    else:
        phi = 0.0

    # Cartesian coordinates on the Bloch sphere
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return theta, phi, x, y, z

def bloch_to_qubit(theta, phi):
    """
    Convert Bloch sphere angles to a qubit state vector.

    Why this is useful: it lets us think geometrically about quantum states.
    Any point on the Bloch sphere corresponds to a unique qubit state
    (up to global phase, which is unobservable).
    """
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    return np.array([alpha, beta], dtype=complex)

# Map special states to the Bloch sphere
print("=== Special States on the Bloch Sphere ===\n")
print(f"{'State':<20} {'theta/pi':>10} {'phi/pi':>10} {'x':>8} {'y':>8} {'z':>8}")
print("-" * 70)

special_states = {
    "|0>": np.array([1, 0], dtype=complex),
    "|1>": np.array([0, 1], dtype=complex),
    "|+>": np.array([1, 1], dtype=complex) / np.sqrt(2),
    "|->": np.array([1, -1], dtype=complex) / np.sqrt(2),
    "|i>": np.array([1, 1j], dtype=complex) / np.sqrt(2),
    "|-i>": np.array([1, -1j], dtype=complex) / np.sqrt(2),
}

for name, state in special_states.items():
    theta, phi, x, y, z = qubit_to_bloch(state)
    print(f"{name:<20} {theta/np.pi:>10.4f} {phi/np.pi:>10.4f} "
          f"{x:>8.4f} {y:>8.4f} {z:>8.4f}")

# Round trip: angles -> state -> angles
print("\n=== Round-Trip Verification ===\n")
test_angles = [(np.pi/3, np.pi/4), (np.pi/2, np.pi), (2*np.pi/3, 3*np.pi/2)]

for theta_in, phi_in in test_angles:
    state = bloch_to_qubit(theta_in, phi_in)
    theta_out, phi_out, x, y, z = qubit_to_bloch(state)
    print(f"Input: theta={theta_in/np.pi:.3f}*pi, phi={phi_in/np.pi:.3f}*pi")
    print(f"  State: [{state[0]:.4f}, {state[1]:.4f}]")
    print(f"  Recovered: theta={theta_out/np.pi:.3f}*pi, phi={phi_out/np.pi:.3f}*pi")
    print(f"  Bloch: ({x:.4f}, {y:.4f}, {z:.4f})")
    print(f"  P(0) = {abs(state[0])**2:.4f}, P(1) = {abs(state[1])**2:.4f}")
    print()
```

---

## 4. Important Bases

The computational basis $\{|0\rangle, |1\rangle\}$ is the most commonly used, but other bases are essential for quantum computing.

### 4.1 The X-Basis (Hadamard Basis)

$$|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}, \quad |-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$$

This basis lies on the equator of the Bloch sphere along the $x$-axis. It is called the "Hadamard basis" because the Hadamard gate (see [Lesson 3](03_Quantum_Gates.md)) converts between the $Z$-basis and the $X$-basis.

**Inverse relations** (equally useful):

$$|0\rangle = \frac{|+\rangle + |-\rangle}{\sqrt{2}}, \quad |1\rangle = \frac{|+\rangle - |-\rangle}{\sqrt{2}}$$

### 4.2 The Y-Basis (Circular Basis)

$$|i\rangle = \frac{|0\rangle + i|1\rangle}{\sqrt{2}}, \quad |-i\rangle = \frac{|0\rangle - i|1\rangle}{\sqrt{2}}$$

This basis lies on the equator along the $y$-axis. It is sometimes called the "circular" basis because in the photon polarization realization, these correspond to right- and left-circular polarization.

### 4.3 Measurement in Different Bases

To measure in basis $\{|b_0\rangle, |b_1\rangle\}$, compute:

$$P(b_0) = |\langle b_0|\psi\rangle|^2, \quad P(b_1) = |\langle b_1|\psi\rangle|^2$$

The choice of measurement basis is a key degree of freedom in quantum computing. Different bases reveal different information about the state.

```python
import numpy as np

# Measurement in different bases

def measure_in_basis(state, basis_states, basis_names):
    """
    Compute measurement probabilities in an arbitrary orthonormal basis.

    Why allow arbitrary bases? In quantum computing, the choice of
    measurement basis is as important as the state preparation. Some
    algorithms require measuring in the X-basis or Y-basis, not just
    the computational (Z) basis.
    """
    print(f"  State: [{state[0]:.4f}, {state[1]:.4f}]")
    for bstate, bname in zip(basis_states, basis_names):
        # Inner product <basis|state>
        amplitude = np.vdot(bstate, state)
        prob = abs(amplitude)**2
        print(f"  P({bname}) = |<{bname}|psi>|^2 = |{amplitude:.4f}|^2 = {prob:.4f}")
    print()

# Define three measurement bases
bases = {
    "Z-basis {|0>, |1>}": (
        [np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)],
        ["|0>", "|1>"]
    ),
    "X-basis {|+>, |->}": (
        [np.array([1, 1], dtype=complex)/np.sqrt(2),
         np.array([1, -1], dtype=complex)/np.sqrt(2)],
        ["|+>", "|->"]
    ),
    "Y-basis {|i>, |-i>}": (
        [np.array([1, 1j], dtype=complex)/np.sqrt(2),
         np.array([1, -1j], dtype=complex)/np.sqrt(2)],
        ["|i>", "|-i>"]
    ),
}

# Test with |+> state
test_state = np.array([1, 1], dtype=complex) / np.sqrt(2)  # |+>
print("=== Measuring |+> in Different Bases ===\n")
for basis_name, (bstates, bnames) in bases.items():
    print(f"{basis_name}:")
    measure_in_basis(test_state, bstates, bnames)

# Test with |i> state
test_state = np.array([1, 1j], dtype=complex) / np.sqrt(2)  # |i>
print("=== Measuring |i> in Different Bases ===\n")
for basis_name, (bstates, bnames) in bases.items():
    print(f"{basis_name}:")
    measure_in_basis(test_state, bstates, bnames)

# Key insight: each basis reveals different information!
print("Key Insight:")
print("  |+> is definite in X-basis but random in Y and Z bases")
print("  |i> is definite in Y-basis but random in X and Z bases")
print("  This is the quantum uncertainty principle in action!")
```

---

## 5. Multi-Qubit Systems

### 5.1 Tensor Products

When we have two qubits, the combined state space is the **tensor product** of the individual spaces:

$$\mathcal{H}_{AB} = \mathcal{H}_A \otimes \mathcal{H}_B$$

If qubit A is in state $|\psi_A\rangle$ and qubit B is in state $|\psi_B\rangle$ (and they are not entangled), the combined state is:

$$|\psi_{AB}\rangle = |\psi_A\rangle \otimes |\psi_B\rangle$$

For the computational basis:

$$|0\rangle \otimes |0\rangle = |00\rangle, \quad |0\rangle \otimes |1\rangle = |01\rangle, \quad |1\rangle \otimes |0\rangle = |10\rangle, \quad |1\rangle \otimes |1\rangle = |11\rangle$$

### 5.2 Tensor Product as Kronecker Product

In matrix representation, the tensor product is computed as the **Kronecker product**:

$$|0\rangle \otimes |1\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \otimes \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 1 \cdot 0 \\ 1 \cdot 1 \\ 0 \cdot 0 \\ 0 \cdot 1 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \end{pmatrix} = |01\rangle$$

The rule: each element of the first vector multiplies the entire second vector.

### 5.3 Two-Qubit Computational Basis

$$|00\rangle = \begin{pmatrix} 1\\0\\0\\0 \end{pmatrix}, \quad |01\rangle = \begin{pmatrix} 0\\1\\0\\0 \end{pmatrix}, \quad |10\rangle = \begin{pmatrix} 0\\0\\1\\0 \end{pmatrix}, \quad |11\rangle = \begin{pmatrix} 0\\0\\0\\1 \end{pmatrix}$$

The binary string determines the position of the 1: $|b_1 b_0\rangle$ has the 1 in position $2b_1 + b_0$ (interpreting the binary string as an integer).

### 5.4 General Two-Qubit State

A general two-qubit state is:

$$|\psi\rangle = \alpha_{00}|00\rangle + \alpha_{01}|01\rangle + \alpha_{10}|10\rangle + \alpha_{11}|11\rangle$$

with normalization $|\alpha_{00}|^2 + |\alpha_{01}|^2 + |\alpha_{10}|^2 + |\alpha_{11}|^2 = 1$.

**Important**: Not all two-qubit states can be written as a tensor product of single-qubit states. States that cannot are called **entangled** -- a topic we will explore in depth in [Lesson 5](05_Entanglement_and_Bell_States.md).

```python
import numpy as np

# Multi-qubit states via tensor products

def tensor_product(state_a, state_b):
    """
    Compute the tensor product of two quantum states using Kronecker product.

    Why Kronecker product? It implements the tensor product of vector spaces
    in matrix representation. For n-qubit systems, we chain multiple
    Kronecker products together.
    """
    return np.kron(state_a, state_b)

# Single-qubit basis states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

# Two-qubit computational basis
print("=== Two-Qubit Computational Basis ===\n")
basis_2q = {}
for b1 in [0, 1]:
    for b0 in [0, 1]:
        label = f"|{b1}{b0}>"
        state = tensor_product(
            ket_0 if b1 == 0 else ket_1,
            ket_0 if b0 == 0 else ket_1
        )
        basis_2q[label] = state
        print(f"{label} = {state}")

# Product state: |+> tensor |0>
print("\n=== Product State: |+> (x) |0> ===\n")
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
product = tensor_product(ket_plus, ket_0)
print(f"|+> (x) |0> = {product}")
print(f"= ({product[0]:.4f})|00> + ({product[1]:.4f})|01> + "
      f"({product[2]:.4f})|10> + ({product[3]:.4f})|11>")

# Verify normalization
print(f"Norm: {np.linalg.norm(product):.4f}")

# Measurement probabilities
print(f"\nMeasurement probabilities:")
for i, label in enumerate(["00", "01", "10", "11"]):
    print(f"  P({label}) = {abs(product[i])**2:.4f}")

# Three-qubit state
print("\n=== Three-Qubit State: |+> (x) |0> (x) |1> ===\n")
three_qubit = tensor_product(tensor_product(ket_plus, ket_0), ket_1)
print(f"State vector (8 components): {three_qubit}")
print(f"Non-zero components:")
for i in range(8):
    if abs(three_qubit[i]) > 1e-10:
        label = format(i, '03b')
        print(f"  |{label}> : amplitude = {three_qubit[i]:.4f}, "
              f"P = {abs(three_qubit[i])**2:.4f}")
```

---

## 6. N-Qubit State Space

### 6.1 Exponential Growth

For $n$ qubits, the state space is $\mathbb{C}^{2^n}$. A general state requires $2^n$ complex amplitudes:

$$|\psi\rangle = \sum_{x=0}^{2^n - 1} \alpha_x |x\rangle$$

where $|x\rangle$ is the computational basis state corresponding to the binary representation of integer $x$.

The normalization constraint is:

$$\sum_{x=0}^{2^n - 1} |\alpha_x|^2 = 1$$

### 6.2 The Power and the Challenge

This exponential scaling is a double-edged sword:

**Power**: An $n$-qubit quantum computer can, in principle, manipulate $2^n$ amplitudes simultaneously. A 50-qubit system involves $\sim 10^{15}$ amplitudes -- more than any classical computer can track efficiently.

**Challenge**: We cannot simply read out all $2^n$ amplitudes. Measurement collapses the state to a single basis state, yielding only $n$ classical bits of information. The art of quantum algorithm design is arranging interference so that the *useful* answer has high probability when we measure.

### 6.3 Separable vs Entangled States

An $n$-qubit state is **separable** (product state) if it can be written as:

$$|\psi\rangle = |\psi_1\rangle \otimes |\psi_2\rangle \otimes \cdots \otimes |\psi_n\rangle$$

A separable state of $n$ qubits is described by only $2n$ complex numbers (2 per qubit), not $2^n$. The remaining states -- the vast majority of the Hilbert space -- are **entangled**.

Entanglement is the resource that makes quantum computing more powerful than classical computing. We will study it in detail in [Lesson 5](05_Entanglement_and_Bell_States.md).

```python
import numpy as np

# Exploring the exponential growth of quantum state spaces

print("=== Quantum State Space Scaling ===\n")
print(f"{'Qubits':>8} {'Dimensions':>15} {'Classical bits':>16} {'Memory (complex128)':>22}")
print("-" * 65)

for n in range(1, 21):
    dim = 2**n
    # Each complex128 is 16 bytes (8 bytes real + 8 bytes imaginary)
    memory_bytes = dim * 16
    if memory_bytes < 1024:
        mem_str = f"{memory_bytes} B"
    elif memory_bytes < 1024**2:
        mem_str = f"{memory_bytes/1024:.1f} KB"
    elif memory_bytes < 1024**3:
        mem_str = f"{memory_bytes/1024**2:.1f} MB"
    elif memory_bytes < 1024**4:
        mem_str = f"{memory_bytes/1024**3:.1f} GB"
    else:
        mem_str = f"{memory_bytes/1024**4:.1f} TB"
    print(f"{n:>8} {dim:>15,} {n:>16} {mem_str:>22}")

# Practical demonstration: create and manipulate small multi-qubit states
print("\n=== Separable vs Entangled States (2 qubits) ===\n")

# Separable state: |+>|0> = (|00> + |10>)/sqrt(2)
separable = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)

# Entangled state (Bell state): (|00> + |11>)/sqrt(2)
# Cannot be written as |a> (x) |b> for any single-qubit states |a>, |b>
entangled = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

def check_separability(state_2q):
    """
    Check if a 2-qubit state is separable by attempting to factorize it.

    Why this approach? A 2-qubit state |psi> = a|00> + b|01> + c|10> + d|11>
    is separable if and only if ad = bc (the matrix [[a,b],[c,d]] has rank 1).
    This is a necessary and sufficient condition.
    """
    # Reshape into 2x2 matrix
    M = state_2q.reshape(2, 2)
    # Check rank via determinant (rank 1 means det = 0)
    det = np.linalg.det(M)
    is_separable = np.isclose(abs(det), 0)
    return is_separable, abs(det)

for name, state in [("Separable |+>|0>", separable), ("Entangled (Bell)", entangled)]:
    is_sep, det = check_separability(state)
    print(f"{name}:")
    print(f"  State: {state}")
    print(f"  |det| = {det:.6f}")
    print(f"  Separable: {is_sep}")
    print()
```

---

## 7. Exercises

### Exercise 1: Bloch Sphere Mapping

For each of the following states, compute the Bloch sphere coordinates $(\theta, \phi)$ and the Cartesian coordinates $(x, y, z)$:

a) $|\psi\rangle = \cos(\pi/8)|0\rangle + \sin(\pi/8)|1\rangle$
b) $|\psi\rangle = \frac{1}{\sqrt{2}}|0\rangle + \frac{e^{i\pi/3}}{\sqrt{2}}|1\rangle$
c) $|\psi\rangle = \frac{1}{2}|0\rangle + \frac{\sqrt{3}}{2}|1\rangle$

Verify your answers using the Python code provided in this lesson.

### Exercise 2: Phase Distinction

Consider the states $|\psi_1\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ and $|\psi_2\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$.

a) Compute the measurement probabilities in the $Z$-basis ($\{|0\rangle, |1\rangle\}$) for both states. Are they distinguishable in this basis?
b) Compute the measurement probabilities in the $X$-basis ($\{|+\rangle, |-\rangle\}$) for both states. Are they distinguishable?
c) Find a measurement basis in which the two states give maximally different outcomes. (Hint: think about the Bloch sphere.)

### Exercise 3: Tensor Products

a) Compute $|+\rangle \otimes |-\rangle$ as a 4-component vector.
b) Compute $|-\rangle \otimes |+\rangle$ as a 4-component vector. Is it the same as (a)?
c) Write the state $\frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$ as a tensor product of two single-qubit states.
d) Prove that the state $\frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$ cannot be written as a tensor product of two single-qubit states. (Hint: use the determinant criterion from the code above.)

### Exercise 4: State Space Counting

a) How many real parameters are needed to specify a general 3-qubit state? (Account for normalization and global phase.)
b) How many real parameters are needed for a separable 3-qubit state?
c) What fraction of the parameter space is "used" by separable states? What does this tell you about entanglement?

### Exercise 5: Bloch Sphere Trajectories

Write Python code that:
a) Generates 100 random qubit states uniformly distributed on the Bloch sphere (Hint: $\theta$ should not be uniformly distributed -- use $\cos\theta$ uniform in $[-1, 1]$ and $\phi$ uniform in $[0, 2\pi)$.)
b) For each state, computes the measurement probabilities in all three bases ($X$, $Y$, $Z$).
c) Verifies that for every state, the sum $P(0)_{Z} + P(0)_{X} + P(0)_{Y}$ is not constant but lies in a specific range. What is that range?

---

[<- Previous: Quantum Mechanics Primer](01_Quantum_Mechanics_Primer.md) | [Next: Quantum Gates ->](03_Quantum_Gates.md)
