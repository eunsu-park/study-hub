# Lesson 1: Quantum Mechanics Primer

| [Next: Qubits and the Bloch Sphere ->](02_Qubits_and_Bloch_Sphere.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain wave-particle duality and its implications through the double-slit experiment
2. Describe the superposition principle correctly using amplitude-based language
3. Apply the Born rule to calculate measurement probabilities from quantum states
4. Read and write quantum states using Dirac (bra-ket) notation
5. Work with complex amplitudes and compute probabilities from them
6. Understand the role of Hilbert spaces as the mathematical arena for quantum computing
7. Distinguish between classical probabilistic mixtures and genuine quantum superpositions

---

Quantum computing harnesses the laws of quantum mechanics -- the physics governing atoms, electrons, and photons -- to process information in fundamentally new ways. Before we can understand quantum algorithms or build quantum circuits, we must first internalize the core principles that make quantum behavior so different from our everyday classical experience. These principles are not merely mathematical abstractions; they are experimentally verified facts about how the universe works at the smallest scales.

This lesson lays the conceptual and mathematical foundation for everything that follows. We will encounter ideas that challenge classical intuition: particles that behave like waves, states that exist as combinations of possibilities, and measurements that irreversibly collapse those possibilities into definite outcomes. Getting comfortable with these ideas -- and especially with the mathematical framework that describes them -- is essential before we can use them for computation.

> **Analogy:** A coin spinning in the air is like a qubit in superposition -- it is not heads AND tails, but rather it has potential for either outcome. Only when it lands (measurement) does it become definite. The key quantum twist is that the "spinning" is described by complex numbers called amplitudes, and these amplitudes can interfere with each other, sometimes canceling out outcomes that would otherwise be possible.

## Table of Contents

1. [Wave-Particle Duality](#1-wave-particle-duality)
2. [The Superposition Principle](#2-the-superposition-principle)
3. [The Measurement Postulate and the Born Rule](#3-the-measurement-postulate-and-the-born-rule)
4. [Dirac Notation](#4-dirac-notation)
5. [Complex Amplitudes and Probability](#5-complex-amplitudes-and-probability)
6. [Hilbert Spaces for Quantum Computing](#6-hilbert-spaces-for-quantum-computing)
7. [Classical vs Quantum Probability](#7-classical-vs-quantum-probability)
8. [Exercises](#8-exercises)

---

## 1. Wave-Particle Duality

One of the most startling discoveries of 20th-century physics is that objects we think of as "particles" -- electrons, photons, even entire molecules -- can exhibit wave-like behavior.

### 1.1 The Double-Slit Experiment

The double-slit experiment is the quintessential demonstration of quantum mechanics. Here is the setup:

1. **Source**: A device emits particles (say, electrons) one at a time toward a barrier.
2. **Barrier**: The barrier has two narrow slits, A and B.
3. **Screen**: Behind the barrier, a detection screen records where each particle lands.

**Classical expectation**: If particles were tiny billiard balls, we would expect two clusters of hits on the screen, one behind each slit. The total pattern would be the sum of individual slit patterns.

**Quantum reality**: Instead, we observe an *interference pattern* -- alternating bright and dark bands across the screen. This pattern is characteristic of *waves* passing through two slits and interfering constructively (bright) and destructively (dark).

The astonishing part: this pattern appears even when particles are sent *one at a time*. Each individual particle lands at a specific point on the screen (particle-like), but over many trials, the statistical distribution forms an interference pattern (wave-like).

### 1.2 The Collapse Upon Observation

If we place detectors at the slits to determine which slit each particle passes through, the interference pattern *disappears*. We get the classical two-cluster pattern instead. The act of acquiring "which path" information destroys the quantum interference.

This is not about the physical disturbance of measurement (a common misconception). It is a fundamental feature of quantum mechanics: information about the path and interference are complementary -- you cannot have both.

### 1.3 Mathematical Description

We describe the particle's state as a *wave function* $\psi(x)$. When both slits are open:

$$\psi(x) = \psi_A(x) + \psi_B(x)$$

The probability of detecting the particle at position $x$ is:

$$P(x) = |\psi(x)|^2 = |\psi_A(x) + \psi_B(x)|^2$$

Expanding this:

$$P(x) = |\psi_A(x)|^2 + |\psi_B(x)|^2 + 2\,\text{Re}[\psi_A^*(x)\,\psi_B(x)]$$

The third term is the *interference term*. It can be positive (constructive) or negative (destructive), producing the characteristic pattern. This term has no classical analog.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulating double-slit interference pattern
# We model each slit as a point source of waves

x = np.linspace(-5, 5, 1000)   # Screen positions
d = 1.0    # Slit separation
lam = 0.5  # Wavelength (de Broglie wavelength of the particle)
L = 10.0   # Distance from slits to screen

# Wave amplitudes from each slit (simplified far-field approximation)
# Phase difference depends on path length difference
# Why: The path difference creates a position-dependent phase shift,
# which is the physical origin of the interference pattern
theta = np.arctan(x / L)
delta_phi = (2 * np.pi / lam) * d * np.sin(theta)

# Amplitudes from slit A and B (equal magnitude, phase-shifted)
psi_A = np.ones_like(x) * np.exp(1j * delta_phi / 2)
psi_B = np.ones_like(x) * np.exp(-1j * delta_phi / 2)

# Quantum case: add amplitudes THEN square
psi_total = psi_A + psi_B
P_quantum = np.abs(psi_total)**2

# Classical case: add probabilities (no interference)
P_classical = np.abs(psi_A)**2 + np.abs(psi_B)**2

print("Quantum interference pattern (first 5 positions):")
print(f"  Positions: {x[:5]}")
print(f"  Probabilities: {P_quantum[:5]}")
print(f"\nMax quantum probability: {P_quantum.max():.4f}")
print(f"Min quantum probability: {P_quantum.min():.4f}")
print(f"Classical probability (constant): {P_classical[0]:.4f}")
print(f"\nKey insight: Quantum probabilities vary from {P_quantum.min():.4f} to "
      f"{P_quantum.max():.4f}")
print(f"while classical probability is uniformly {P_classical[0]:.4f}")
print("The interference term creates the oscillation!")
```

### 1.4 Key Takeaway

Wave-particle duality tells us that quantum objects do not fit neatly into either the "wave" or "particle" category. They are described by amplitudes that propagate like waves and are detected like particles. For quantum computing, the critical insight is: **quantum information is carried by amplitudes, and these amplitudes can interfere**.

---

## 2. The Superposition Principle

Superposition is the most fundamental principle of quantum mechanics and the cornerstone of quantum computing's power.

### 2.1 What Superposition Really Means

A common but misleading description says that a quantum system "is in multiple states at once." This is not quite right. A more accurate description:

> A quantum system in superposition has *amplitudes* assigned to each possible outcome. These amplitudes determine the probabilities of outcomes, but the system is not simultaneously "in" all those outcomes. It is in a *single* quantum state that encodes potentialities.

Consider a qubit that can be measured as $|0\rangle$ or $|1\rangle$. A general state is:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

where $\alpha$ and $\beta$ are complex numbers. This expression means:
- The qubit is in *one* definite quantum state: $|\psi\rangle$
- If measured, the probability of outcome $|0\rangle$ is $|\alpha|^2$ and of $|1\rangle$ is $|\beta|^2$
- Before measurement, the qubit is not "both 0 and 1" -- it is in a state that is fundamentally different from either

### 2.2 Superposition vs Classical Probability

A classical coin can be described probabilistically: "50% chance heads, 50% chance tails." How is this different from a qubit with $|\alpha|^2 = |\beta|^2 = 1/2$?

The crucial difference is **interference**. Classical probabilities always add:

$$P_{\text{classical}}(\text{outcome}) = P_A(\text{outcome}) + P_B(\text{outcome}) \geq 0$$

Quantum amplitudes can cancel:

$$\alpha_{\text{total}} = \alpha_A + \alpha_B \quad \Rightarrow \quad P = |\alpha_A + \alpha_B|^2$$

This means quantum probabilities can be *less* than what you would get from adding individual probabilities -- destructive interference can make certain outcomes impossible even when each individual path would allow them.

### 2.3 The Linearity of Quantum Mechanics

If $|\psi_1\rangle$ and $|\psi_2\rangle$ are valid quantum states, then any linear combination:

$$|\psi\rangle = c_1|\psi_1\rangle + c_2|\psi_2\rangle$$

is also a valid quantum state (after normalization). This is the superposition principle in its mathematical form. It arises because the Schrodinger equation governing quantum evolution is *linear*.

```python
import numpy as np

# Demonstrating superposition and interference
# Two quantum states represented as column vectors

# |0> and |1> basis states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

# A superposition state: |+> = (|0> + |1>) / sqrt(2)
# Why divide by sqrt(2)? Normalization ensures probabilities sum to 1
ket_plus = (ket_0 + ket_1) / np.sqrt(2)

# Another superposition: |-> = (|0> - |1>) / sqrt(2)
# The MINUS sign is the key difference -- it changes the relative phase
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

print("State |+>:", ket_plus)
print("State |->:", ket_minus)

# Both have the same measurement probabilities in the {|0>, |1>} basis!
print(f"\n|+> probabilities: P(0) = {abs(ket_plus[0])**2:.2f}, "
      f"P(1) = {abs(ket_plus[1])**2:.2f}")
print(f"|-> probabilities: P(0) = {abs(ket_minus[0])**2:.2f}, "
      f"P(1) = {abs(ket_minus[1])**2:.2f}")

# But they are DIFFERENT states! Adding them:
# |+> + |-> = sqrt(2)|0> -- interference eliminated |1>!
sum_state = ket_plus + ket_minus
sum_normalized = sum_state / np.linalg.norm(sum_state)
print(f"\nAfter interference (|+> + |->): {sum_state}")
print(f"Normalized: {sum_normalized}")
print(f"Probabilities: P(0) = {abs(sum_normalized[0])**2:.2f}, "
      f"P(1) = {abs(sum_normalized[1])**2:.2f}")
print("\nThe |1> component completely canceled out -- destructive interference!")
```

---

## 3. The Measurement Postulate and the Born Rule

### 3.1 The Measurement Problem

In quantum mechanics, measurement is not a passive act of reading a pre-existing value. It is an active process that *changes* the state of the system.

**Before measurement**: The system is in state $|\psi\rangle = \sum_i c_i |i\rangle$, a superposition of possible outcomes $|i\rangle$.

**During measurement**: The system "collapses" to one of the possible outcomes $|i\rangle$.

**After measurement**: The system is definitively in state $|i\rangle$. The superposition is destroyed.

### 3.2 The Born Rule

The Born rule, proposed by Max Born in 1926, tells us the probability of each outcome:

$$P(i) = |c_i|^2 = |\langle i|\psi\rangle|^2$$

where:
- $c_i$ is the amplitude of outcome $|i\rangle$ in the state $|\psi\rangle$
- $|c_i|^2$ means the modulus squared of the complex number $c_i$
- $\langle i|\psi\rangle$ is the inner product (we will define this precisely in the next section)

**Normalization**: Since probabilities must sum to 1:

$$\sum_i |c_i|^2 = 1$$

### 3.3 Example: Measuring a Qubit

Consider the state $|\psi\rangle = \frac{1}{\sqrt{3}}|0\rangle + \sqrt{\frac{2}{3}}|1\rangle$.

- $P(0) = \left|\frac{1}{\sqrt{3}}\right|^2 = \frac{1}{3} \approx 33.3\%$
- $P(1) = \left|\sqrt{\frac{2}{3}}\right|^2 = \frac{2}{3} \approx 66.7\%$
- Check: $\frac{1}{3} + \frac{2}{3} = 1$ (normalized)

After measurement, if we get outcome $|0\rangle$, the state is now definitively $|0\rangle$. The amplitude $\sqrt{2/3}$ for $|1\rangle$ is gone -- irreversibly.

### 3.4 Repeated Measurement

If we immediately measure the same qubit again (in the same basis), we will always get the same result. The state has collapsed, and there is no superposition left to produce a different outcome. This is called the *idempotency* of measurement.

```python
import numpy as np

# Simulating quantum measurement with the Born rule

def measure_qubit(state, num_shots=10000):
    """
    Simulate measuring a qubit state many times.

    Why simulate many shots? A single measurement gives one outcome.
    To verify the Born rule probabilities, we need statistics from
    many identical preparations and measurements.
    """
    # Extract probabilities using the Born rule: P(i) = |c_i|^2
    probabilities = np.abs(state)**2

    # Simulate measurements by sampling from the probability distribution
    # Why use np.random.choice? It samples from discrete distribution,
    # which is exactly what quantum measurement does
    outcomes = np.random.choice(len(state), size=num_shots, p=probabilities)

    # Count occurrences of each outcome
    counts = np.bincount(outcomes, minlength=len(state))
    frequencies = counts / num_shots

    return counts, frequencies

# State: |psi> = (1/sqrt(3))|0> + sqrt(2/3)|1>
psi = np.array([1/np.sqrt(3), np.sqrt(2/3)], dtype=complex)

# Verify normalization
norm = np.sum(np.abs(psi)**2)
print(f"State: ({psi[0]:.4f})|0> + ({psi[1]:.4f})|1>")
print(f"Normalization check: |alpha|^2 + |beta|^2 = {norm:.6f}")

# Theoretical probabilities
print(f"\nTheoretical: P(0) = {abs(psi[0])**2:.4f}, P(1) = {abs(psi[1])**2:.4f}")

# Simulated measurements
np.random.seed(42)  # For reproducibility
counts, freqs = measure_qubit(psi, num_shots=100000)
print(f"Simulated (100000 shots): P(0) = {freqs[0]:.4f}, P(1) = {freqs[1]:.4f}")
print(f"Counts: |0> = {counts[0]}, |1> = {counts[1]}")

# Demonstrating collapse: after measuring |0>, re-measuring always gives |0>
print("\n--- Post-measurement collapse ---")
collapsed_state = np.array([1, 0], dtype=complex)  # Collapsed to |0>
_, collapsed_freqs = measure_qubit(collapsed_state, num_shots=1000)
print(f"After collapse to |0>, re-measurement: P(0) = {collapsed_freqs[0]:.4f}, "
      f"P(1) = {collapsed_freqs[1]:.4f}")
```

---

## 4. Dirac Notation

Paul Dirac introduced an elegant notation for quantum mechanics that is universally used in quantum computing. It may look unfamiliar at first, but it is extremely practical once you internalize it.

### 4.1 Kets, Bras, and Brackets

| Notation | Name | Mathematical Object | Column/Row |
|----------|------|---------------------|------------|
| $\|v\rangle$ | "ket v" | Column vector | $\begin{pmatrix} v_1 \\ v_2 \\ \vdots \end{pmatrix}$ |
| $\langle v\|$ | "bra v" | Row vector (conjugate transpose) | $(v_1^*, v_2^*, \ldots)$ |
| $\langle u\|v\rangle$ | "bracket" or "inner product" | Complex number | $\sum_i u_i^* v_i$ |
| $\|u\rangle\langle v\|$ | "outer product" | Matrix | $u_i v_j^*$ |

The names come from "bra-ket" splitting the word "bracket."

### 4.2 The Computational Basis

For a single qubit, the two computational basis states are:

$$|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad |1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

These are orthonormal:
- $\langle 0|0\rangle = 1$, $\langle 1|1\rangle = 1$ (normalized)
- $\langle 0|1\rangle = 0$, $\langle 1|0\rangle = 0$ (orthogonal)

A general qubit state is:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle = \begin{pmatrix} \alpha \\ \beta \end{pmatrix}$$

### 4.3 Inner Products and Amplitudes

The amplitude of outcome $|i\rangle$ in state $|\psi\rangle$ is the inner product $\langle i|\psi\rangle$:

$$\langle 0|\psi\rangle = \alpha, \quad \langle 1|\psi\rangle = \beta$$

The Born rule then gives:

$$P(0) = |\langle 0|\psi\rangle|^2 = |\alpha|^2, \quad P(1) = |\langle 1|\psi\rangle|^2 = |\beta|^2$$

### 4.4 Outer Products and Projectors

The outer product $|i\rangle\langle i|$ creates a *projection operator*:

$$|0\rangle\langle 0| = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \begin{pmatrix} 1 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$$

Applying this to a state extracts the component along $|0\rangle$:

$$(|0\rangle\langle 0|)|\psi\rangle = |0\rangle(\langle 0|\psi\rangle) = \alpha|0\rangle$$

The **completeness relation** (resolution of the identity) states:

$$|0\rangle\langle 0| + |1\rangle\langle 1| = I$$

This is a fundamental identity used constantly in quantum computing proofs.

```python
import numpy as np

# Dirac notation in practice using numpy

# Basis kets as column vectors
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

# Bras are conjugate transposes of kets
# Why conjugate transpose (not just transpose)? Because quantum amplitudes
# are complex, and the inner product must be positive-definite
bra_0 = ket_0.conj().T
bra_1 = ket_1.conj().T

# Inner products (brackets)
print("Inner products:")
print(f"  <0|0> = {(bra_0 @ ket_0)[0,0]}")
print(f"  <1|1> = {(bra_1 @ ket_1)[0,0]}")
print(f"  <0|1> = {(bra_0 @ ket_1)[0,0]}")
print(f"  <1|0> = {(bra_1 @ ket_0)[0,0]}")

# A general state
alpha = 1/np.sqrt(3)
beta = np.sqrt(2/3) * np.exp(1j * np.pi/4)  # Complex amplitude with phase!
psi = alpha * ket_0 + beta * ket_1
print(f"\n|psi> = ({alpha:.4f})|0> + ({beta:.4f})|1>")

# Extract amplitudes via inner products
amp_0 = (bra_0 @ psi)[0,0]
amp_1 = (bra_1 @ psi)[0,0]
print(f"<0|psi> = {amp_0:.4f}")
print(f"<1|psi> = {amp_1:.4f}")

# Born rule probabilities
print(f"\nP(0) = |<0|psi>|^2 = {abs(amp_0)**2:.4f}")
print(f"P(1) = |<1|psi>|^2 = {abs(amp_1)**2:.4f}")
print(f"Sum = {abs(amp_0)**2 + abs(amp_1)**2:.4f}")

# Outer products (projectors)
proj_0 = ket_0 @ bra_0  # |0><0|
proj_1 = ket_1 @ bra_1  # |1><1|
print(f"\n|0><0| = \n{proj_0}")
print(f"\n|1><1| = \n{proj_1}")

# Completeness relation: |0><0| + |1><1| = I
identity = proj_0 + proj_1
print(f"\n|0><0| + |1><1| = \n{identity}")
print(f"Is identity? {np.allclose(identity, np.eye(2))}")
```

---

## 5. Complex Amplitudes and Probability

### 5.1 Why Complex Numbers?

Quantum amplitudes are complex numbers $\alpha = a + bi$ where $i = \sqrt{-1}$. This is not a mathematical convenience -- it is a physical necessity. Complex amplitudes are required to:

1. **Describe interference**: Real numbers can only add constructively. Complex numbers can cancel each other out.
2. **Encode phase information**: The relative phase between amplitudes affects interference but not individual probabilities.
3. **Represent unitary evolution**: Quantum gates are unitary matrices, which naturally live in the complex number field.

### 5.2 Modulus and Phase

Any complex number can be written in polar form:

$$\alpha = |\alpha| \cdot e^{i\phi} = |\alpha|(\cos\phi + i\sin\phi)$$

where:
- $|\alpha| = \sqrt{a^2 + b^2}$ is the **modulus** (magnitude)
- $\phi = \text{atan2}(b, a)$ is the **phase** (angle)

The probability is $|\alpha|^2$, which depends only on the modulus, not the phase. So why does phase matter? Because when amplitudes *combine* (through gates or interference), their phases determine whether they add constructively or destructively.

### 5.3 Interference of Complex Amplitudes

Consider two amplitudes combining:

$$\alpha_{\text{total}} = \alpha_1 + \alpha_2 = |\alpha_1|e^{i\phi_1} + |\alpha_2|e^{i\phi_2}$$

The resulting probability:

$$P = |\alpha_{\text{total}}|^2 = |\alpha_1|^2 + |\alpha_2|^2 + 2|\alpha_1||\alpha_2|\cos(\phi_1 - \phi_2)$$

The interference term $2|\alpha_1||\alpha_2|\cos(\phi_1 - \phi_2)$ depends on the *phase difference* $\phi_1 - \phi_2$:
- **Constructive** ($\phi_1 - \phi_2 = 0$): Amplitudes reinforce, $P$ is maximized
- **Destructive** ($\phi_1 - \phi_2 = \pi$): Amplitudes cancel, $P$ is minimized
- **Partial**: Other phase differences give intermediate results

```python
import numpy as np

# Demonstrating how phase affects interference

def interference_probability(mag1, phase1, mag2, phase2):
    """
    Compute the probability when two complex amplitudes combine.

    Why separate magnitude and phase? This makes the physics transparent:
    magnitudes set the 'strength' of each path, while phases determine
    whether the paths reinforce or cancel each other.
    """
    alpha1 = mag1 * np.exp(1j * phase1)
    alpha2 = mag2 * np.exp(1j * phase2)
    total = alpha1 + alpha2
    return np.abs(total)**2

# Two equal-magnitude amplitudes with varying phase difference
mag = 1 / np.sqrt(2)  # Each amplitude contributes 50% individually
phases = np.linspace(0, 2*np.pi, 8)

print("Phase difference vs Combined probability")
print("=" * 50)
print(f"Individual probability of each path: {mag**2:.4f}")
print(f"Classical sum (no interference): {2*mag**2:.4f}")
print()

for phi in phases:
    P = interference_probability(mag, 0, mag, phi)
    interference_type = ""
    if np.isclose(phi % (2*np.pi), 0):
        interference_type = "(constructive)"
    elif np.isclose(phi % (2*np.pi), np.pi):
        interference_type = "(destructive)"
    print(f"  Phase diff = {phi/np.pi:.2f}*pi  =>  P = {P:.4f}  {interference_type}")

print(f"\nKey insight: Same individual probabilities, but combined probability")
print(f"ranges from 0.00 to 2.00 depending on PHASE!")
print(f"This is the power quantum computing exploits.")
```

### 5.4 Global Phase vs Relative Phase

An important distinction:

- **Global phase**: Multiplying the entire state by $e^{i\theta}$ has NO observable effect. The states $|\psi\rangle$ and $e^{i\theta}|\psi\rangle$ are physically identical because $|e^{i\theta}\alpha|^2 = |\alpha|^2$.

- **Relative phase**: The phase *between* components of a superposition IS physically meaningful. The states $|0\rangle + |1\rangle$ and $|0\rangle + e^{i\phi}|1\rangle$ are physically different for $\phi \neq 0$.

We will explore this distinction in depth in [Lesson 2: Qubits and the Bloch Sphere](02_Qubits_and_Bloch_Sphere.md).

---

## 6. Hilbert Spaces for Quantum Computing

### 6.1 What Is a Hilbert Space?

A **Hilbert space** is a complete vector space equipped with an inner product. For quantum computing, we work with *finite-dimensional* complex Hilbert spaces, which are much simpler than the infinite-dimensional spaces of general quantum mechanics.

For a single qubit, the Hilbert space is $\mathbb{C}^2$ -- the space of all 2-component complex vectors. For $n$ qubits, it is $\mathbb{C}^{2^n}$.

### 6.2 Properties We Need

The key properties of our Hilbert spaces:

| Property | Definition | QC Relevance |
|----------|-----------|--------------|
| **Vector space** | Can add states and multiply by scalars | Superposition principle |
| **Complex field** | Scalars are complex numbers | Phase and interference |
| **Inner product** | $\langle u\|v\rangle \in \mathbb{C}$ | Probabilities via Born rule |
| **Completeness** | Every Cauchy sequence converges | Automatic for finite dimensions |
| **Finite dimension** | $\dim = 2^n$ for $n$ qubits | Tractable for small systems |

### 6.3 Orthonormal Bases

An orthonormal basis $\{|e_i\rangle\}$ satisfies:

$$\langle e_i|e_j\rangle = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

For $n$ qubits, the computational basis consists of $2^n$ states. For 2 qubits:

$$\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$$

These are the states we measure in, and any quantum state can be written as a superposition over them.

### 6.4 Why Exponential Growth Matters

The dimension of the Hilbert space grows *exponentially* with the number of qubits. This exponential growth is the source of quantum computing's potential power:

| Qubits ($n$) | Dimension ($2^n$) | Complex numbers needed |
|:---:|:---:|:---:|
| 1 | 2 | 2 |
| 10 | 1,024 | 1,024 |
| 20 | ~1 million | ~1 million |
| 50 | ~$10^{15}$ | ~$10^{15}$ |
| 300 | $2^{300} > 10^{90}$ | More than atoms in the universe |

A 300-qubit quantum system cannot be fully simulated by any classical computer, even in principle. This is the fundamental reason quantum computing can potentially solve problems beyond classical reach.

```python
import numpy as np

# Exploring Hilbert space dimensions

def describe_hilbert_space(n_qubits):
    """
    Describe the Hilbert space for n qubits.

    Why is this important? The exponential growth of state space is THE
    fundamental resource that quantum algorithms exploit. Understanding
    this scaling is essential for appreciating quantum advantage.
    """
    dim = 2**n_qubits

    # A general state needs 'dim' complex amplitudes (2*dim real numbers)
    # minus 1 for normalization, minus 1 for global phase
    real_params = 2 * dim - 2

    return dim, real_params

print("Hilbert Space Dimensions for n Qubits")
print("=" * 60)
print(f"{'Qubits':>8} {'Dimension':>15} {'Real Parameters':>18}")
print("-" * 60)

for n in [1, 2, 3, 5, 10, 20, 30]:
    dim, params = describe_hilbert_space(n)
    print(f"{n:>8} {dim:>15,} {params:>18,}")

# Demonstrate: creating a random state in n-qubit Hilbert space
n = 3
dim = 2**n
print(f"\n--- Random {n}-qubit state (dimension {dim}) ---")

# Why generate random states this way? A truly random quantum state (Haar random)
# is created by generating random complex numbers and normalizing.
rng = np.random.default_rng(42)
raw = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
state = raw / np.linalg.norm(raw)  # Normalize

print(f"State vector ({dim} complex amplitudes):")
for i in range(dim):
    # Convert index to binary string for qubit label
    label = format(i, f'0{n}b')
    prob = abs(state[i])**2
    print(f"  |{label}> : amplitude = {state[i]:.4f}, P = {prob:.4f}")

print(f"\nTotal probability: {sum(abs(state[i])**2 for i in range(dim)):.6f}")
```

---

## 7. Classical vs Quantum Probability

It is worth making the distinction between classical and quantum probability crystal clear, as confusion here is the source of many misconceptions about quantum computing.

### 7.1 Classical Probability: Ignorance

When we say "the coin has a 50% chance of being heads," we mean we *do not know* which side it will land on. In principle, if we knew the exact initial conditions (force, angle, air resistance), we could predict the outcome with certainty. Classical probability reflects our *ignorance*.

Classical states can be mixed (probabilistic) using a **probability distribution**:

$$\rho_{\text{classical}} = \{(p_1, \text{state}_1), (p_2, \text{state}_2), \ldots\}$$

### 7.2 Quantum Probability: Fundamental

Quantum probability is not about ignorance -- it is about the *fundamental nature of reality*. Even with complete knowledge of the quantum state $|\psi\rangle$ (which is the maximum possible knowledge), measurement outcomes are inherently probabilistic. This is not a limitation of our technology or knowledge; it is a feature of the universe.

### 7.3 The No-Cloning Theorem (Preview)

One consequence of quantum probability being fundamental: you cannot copy an unknown quantum state. If you could, you could make many copies and measure them all to deduce the state perfectly, violating the measurement postulate. This **no-cloning theorem** has profound implications for quantum computing and quantum cryptography (discussed further in later lessons).

### 7.4 Comparison Summary

| Aspect | Classical Probability | Quantum Probability |
|--------|----------------------|---------------------|
| **Nature** | Subjective (ignorance) | Objective (fundamental) |
| **Values** | Real, $0 \leq p \leq 1$ | Complex amplitudes |
| **Combination** | Probabilities add | Amplitudes add, then square |
| **Interference** | Never | Yes (constructive and destructive) |
| **Copying** | Possible | Forbidden (no-cloning) |
| **States** | Can always be fully known | Complete knowledge still gives random outcomes |

```python
import numpy as np

# Classical vs Quantum: a concrete comparison

np.random.seed(42)

print("=== Classical Coin vs Quantum Qubit ===\n")

# Scenario: Two paths (A and B) lead to a detector.
# Each path has a 50% probability of triggering the detector.

# Classical case: Probabilities add
p_A = 0.5
p_B = 0.5
p_classical = p_A + p_B  # Capped at 1 in practice, but the principle is additive
print(f"Classical: P(A) + P(B) = {p_A} + {p_B} = {min(p_classical, 1.0)}")
print(f"  (Probabilities always add -- no cancellation possible)")

# Quantum case: Amplitudes add, THEN we square
# Case 1: Same phase (constructive interference)
alpha_A = 1/np.sqrt(2)
alpha_B = 1/np.sqrt(2)
p_constructive = abs(alpha_A + alpha_B)**2
print(f"\nQuantum (constructive): |alpha_A + alpha_B|^2 = |{alpha_A:.4f} + {alpha_B:.4f}|^2 "
      f"= {p_constructive:.4f}")

# Case 2: Opposite phase (destructive interference)
alpha_A = 1/np.sqrt(2)
alpha_B = -1/np.sqrt(2)  # Note the minus sign!
p_destructive = abs(alpha_A + alpha_B)**2
print(f"Quantum (destructive): |alpha_A + alpha_B|^2 = |{alpha_A:.4f} + ({alpha_B:.4f})|^2 "
      f"= {p_destructive:.4f}")

print(f"\nSame individual probabilities ({abs(alpha_A)**2:.2f} each), but:")
print(f"  Classical result: always {min(p_A + p_B, 1.0):.2f}")
print(f"  Quantum (constructive): {p_constructive:.2f}  <- probability INCREASED")
print(f"  Quantum (destructive):  {p_destructive:.2f}  <- probability VANISHED!")
print(f"\nThis is the essence of quantum computing:")
print(f"  Arrange amplitudes so wrong answers cancel (destructive)")
print(f"  and right answers reinforce (constructive).")
```

---

## 8. Exercises

### Exercise 1: Normalization

A qubit is in the state $|\psi\rangle = \frac{3}{5}|0\rangle + \frac{4}{5}e^{i\pi/3}|1\rangle$.

a) Verify that this state is normalized.
b) What are the probabilities of measuring $|0\rangle$ and $|1\rangle$?
c) Does the phase factor $e^{i\pi/3}$ affect the measurement probabilities in the computational basis?
d) Would this phase factor matter if we measured in a different basis? (Conceptual answer is fine.)

### Exercise 2: Interference Calculation

Two amplitudes $\alpha_1 = \frac{1}{2}$ and $\alpha_2 = \frac{1}{2}e^{i\phi}$ combine to give $\alpha_{\text{total}} = \alpha_1 + \alpha_2$.

a) Compute $|\alpha_{\text{total}}|^2$ as a function of $\phi$.
b) For what value of $\phi$ is the probability maximized? What is the maximum?
c) For what value of $\phi$ is the probability minimized? What is the minimum?
d) Write a Python script to plot $P(\phi)$ for $\phi \in [0, 2\pi]$.

### Exercise 3: Dirac Notation Practice

Given the state $|\psi\rangle = \frac{1}{2}|0\rangle + \frac{\sqrt{3}}{2}|1\rangle$:

a) Write $|\psi\rangle$ as a column vector.
b) Write $\langle\psi|$ as a row vector.
c) Compute $\langle\psi|\psi\rangle$.
d) Compute the projector $|\psi\rangle\langle\psi|$ as a 2x2 matrix.
e) Verify that $(|\psi\rangle\langle\psi|)^2 = |\psi\rangle\langle\psi|$ (this is the defining property of a projector).

### Exercise 4: Hilbert Space Exploration

a) For a 4-qubit system, how many complex amplitudes are needed to describe a general state?
b) How many *real* parameters are needed (accounting for normalization and global phase)?
c) Write Python code to create a random 4-qubit state, measure it 10,000 times, and plot a histogram of the outcomes. How does the distribution compare to what you would expect from the amplitudes?

### Exercise 5: Conceptual Understanding

a) Explain in your own words why "a qubit is simultaneously 0 and 1" is a misleading description of superposition. What would be a more accurate description?
b) Why can interference only occur with *complex* amplitudes? Could real-valued amplitudes produce destructive interference? (Hint: consider whether real numbers can cancel.)
c) A friend claims that quantum randomness is "just like flipping a coin." Provide at least two specific ways in which quantum randomness is fundamentally different from classical coin-flip randomness.

---

| [Next: Qubits and the Bloch Sphere ->](02_Qubits_and_Bloch_Sphere.md)
