# Lesson 5: Entanglement and Bell States

[<- Previous: Quantum Circuits](04_Quantum_Circuits.md) | [Next: Quantum Measurement ->](06_Quantum_Measurement.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between separable (product) and entangled quantum states using mathematical criteria
2. Construct and identify all four Bell states and explain their physical significance
3. Build Bell states using Hadamard and CNOT gates in a quantum circuit
4. Explain the EPR paradox and its resolution through quantum mechanics
5. State Bell's theorem and explain how the CHSH inequality rules out local hidden variable theories
6. Describe the monogamy of entanglement and its implications for quantum information
7. Simulate entangled states and Bell inequality tests in Python

---

Entanglement is perhaps the most profoundly non-classical feature of quantum mechanics. When two or more qubits are entangled, their quantum states become correlated in ways that have no classical explanation -- measuring one qubit instantaneously determines the state of the other, regardless of the physical distance between them. Einstein famously called this "spooky action at a distance," and it troubled him deeply.

Far from being a mere curiosity, entanglement is the engine that powers quantum computing's advantages. Without entanglement, a quantum computer would be no more powerful than a classical probabilistic machine. Every quantum algorithm that achieves exponential speedup does so by creating and manipulating entangled states. In this lesson, we study entanglement through its most fundamental examples -- the Bell states -- and examine the theoretical framework that proves entanglement is genuinely non-classical.

> **Analogy:** Entangled qubits are like a pair of magic dice -- no matter how far apart, when one shows 6, the other always shows 1. The outcomes are correlated in ways that classical physics cannot explain. No matter what "hidden instructions" you try to write on the dice before separating them, the quantum correlations are provably stronger than any classical correlation could ever be.

## Table of Contents

1. [Separable vs Entangled States](#1-separable-vs-entangled-states)
2. [The Four Bell States](#2-the-four-bell-states)
3. [Creating Bell States](#3-creating-bell-states)
4. [The EPR Paradox](#4-the-epr-paradox)
5. [Bell's Theorem and the CHSH Inequality](#5-bells-theorem-and-the-chsh-inequality)
6. [Monogamy of Entanglement](#6-monogamy-of-entanglement)
7. [Entanglement as a Resource](#7-entanglement-as-a-resource)
8. [Exercises](#8-exercises)

---

## 1. Separable vs Entangled States

### 1.1 Definition

A two-qubit pure state $|\psi\rangle_{AB}$ is **separable** (also called a **product state**) if it can be written as a tensor product of single-qubit states:

$$|\psi\rangle_{AB} = |\alpha\rangle_A \otimes |\beta\rangle_B$$

If no such factorization exists, the state is **entangled**.

### 1.2 Example: Separable State

$$|+\rangle \otimes |0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \otimes |0\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)$$

This is separable because we can identify the factors: qubit A is in $|+\rangle$ and qubit B is in $|0\rangle$. Measuring qubit A has no effect on qubit B.

### 1.3 Example: Entangled State

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

Can we write this as $|\alpha\rangle \otimes |\beta\rangle$? Let us try:

$$(\alpha_0|0\rangle + \alpha_1|1\rangle) \otimes (\beta_0|0\rangle + \beta_1|1\rangle) = \alpha_0\beta_0|00\rangle + \alpha_0\beta_1|01\rangle + \alpha_1\beta_0|10\rangle + \alpha_1\beta_1|11\rangle$$

For this to equal $\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$, we need:
- $\alpha_0\beta_0 = \frac{1}{\sqrt{2}}$ (coefficient of $|00\rangle$)
- $\alpha_0\beta_1 = 0$ (coefficient of $|01\rangle$)
- $\alpha_1\beta_0 = 0$ (coefficient of $|10\rangle$)
- $\alpha_1\beta_1 = \frac{1}{\sqrt{2}}$ (coefficient of $|11\rangle$)

From the second equation, either $\alpha_0 = 0$ or $\beta_1 = 0$. But if $\alpha_0 = 0$, the first equation fails. If $\beta_1 = 0$, the fourth equation fails. There is no solution. The state is entangled.

### 1.4 The Schmidt Decomposition Test

For a two-qubit state written as $|\psi\rangle = \sum_{ij} c_{ij}|ij\rangle$, we can arrange the coefficients in a matrix $C$:

$$C = \begin{pmatrix} c_{00} & c_{01} \\ c_{10} & c_{11} \end{pmatrix}$$

The state is separable if and only if $\text{rank}(C) = 1$, which is equivalent to $\det(C) = 0$.

For entangled states, the **Schmidt rank** is 2, and the **Schmidt coefficients** quantify the degree of entanglement.

```python
import numpy as np

# Separable vs Entangled states

def check_entanglement(state_2q, label=""):
    """
    Determine if a 2-qubit state is separable or entangled.

    Why use the determinant? A 2-qubit state |psi> = sum c_ij |ij>
    is separable iff the coefficient matrix C has rank 1, which means
    det(C) = 0. Non-zero determinant implies entanglement.

    For a more quantitative measure, we use the Schmidt decomposition.
    """
    # Reshape state vector into 2x2 coefficient matrix
    C = state_2q.reshape(2, 2)

    # Determinant test
    det = np.linalg.det(C)
    is_entangled = not np.isclose(abs(det), 0)

    # Schmidt decomposition via SVD
    # The singular values are the Schmidt coefficients
    U, s, Vh = np.linalg.svd(C)
    schmidt_coeffs = s[s > 1e-10]  # Non-zero Schmidt coefficients

    # Entanglement entropy: S = -sum(lambda^2 * log(lambda^2))
    probs = schmidt_coeffs**2
    entropy = -np.sum(probs * np.log2(probs + 1e-15))

    print(f"State: {label}")
    print(f"  Coefficient matrix:\n    {C[0]}\n    {C[1]}")
    print(f"  |det(C)| = {abs(det):.6f}")
    print(f"  Schmidt coefficients: {schmidt_coeffs}")
    print(f"  Entanglement entropy: {entropy:.4f} bits")
    print(f"  {'ENTANGLED' if is_entangled else 'SEPARABLE'}")
    print()

# Test various states
print("=== Entanglement Analysis ===\n")

# Separable: |+>|0> = (|00> + |10>)/sqrt(2)
check_entanglement(
    np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2),
    "|+>|0>"
)

# Entangled: Bell state |Phi+> = (|00> + |11>)/sqrt(2)
check_entanglement(
    np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
    "|Phi+> = (|00>+|11>)/sqrt(2)"
)

# Partially entangled: sqrt(3/4)|00> + sqrt(1/4)|11>
check_entanglement(
    np.array([np.sqrt(3/4), 0, 0, np.sqrt(1/4)], dtype=complex),
    "sqrt(3/4)|00> + sqrt(1/4)|11>"
)

# Product state: |+>|->
check_entanglement(
    np.kron(
        np.array([1, 1], dtype=complex) / np.sqrt(2),
        np.array([1, -1], dtype=complex) / np.sqrt(2)
    ),
    "|+>|-> (product)"
)
```

---

## 2. The Four Bell States

### 2.1 Definition

The four **Bell states** (also called EPR pairs or Bell basis) are the maximally entangled two-qubit states:

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) \qquad |\Phi^-\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$$

$$|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle) \qquad |\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$$

### 2.2 Properties

All four Bell states share these properties:

1. **Maximally entangled**: The entanglement entropy is 1 bit (the maximum for a 2-qubit system).
2. **Orthonormal**: $\langle \Phi^+|\Phi^-\rangle = 0$, etc. They form a complete basis for the 4-dimensional two-qubit space.
3. **Local measurements are maximally random**: Measuring either qubit alone gives 50/50 outcomes, with no information about the state.
4. **Perfect correlations**: Measuring both qubits in the same basis always gives correlated results.

### 2.3 Correlation Patterns

| Bell State | Z-basis correlation | X-basis correlation |
|-----------|-------|-------|
| $\|\Phi^+\rangle$ | Same: both 0 or both 1 | Same: both + or both - |
| $\|\Phi^-\rangle$ | Same | Opposite |
| $\|\Psi^+\rangle$ | Opposite: one 0 one 1 | Same |
| $\|\Psi^-\rangle$ | Opposite | Opposite |

### 2.4 The Bell Basis as an Alternative

Just as $\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$ is the computational basis for 2 qubits, $\{|\Phi^+\rangle, |\Phi^-\rangle, |\Psi^+\rangle, |\Psi^-\rangle\}$ is an equally valid basis. Any two-qubit state can be written as a superposition of Bell states. "Bell measurement" projects onto this basis and is a key operation in quantum teleportation and dense coding.

```python
import numpy as np

# The four Bell states and their properties

# Define Bell states
bell_states = {
    "|Phi+>": np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
    "|Phi->": np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
    "|Psi+>": np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
    "|Psi->": np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),
}

# Verify orthonormality
print("=== Bell State Orthonormality ===\n")
names = list(bell_states.keys())
states = list(bell_states.values())
for i in range(4):
    for j in range(4):
        overlap = np.vdot(states[i], states[j])
        if abs(overlap) > 1e-10:
            print(f"  <{names[i]}|{names[j]}> = {overlap:.4f}")

# Correlation analysis: measure both qubits in Z-basis
print("\n=== Correlation Patterns (Z-basis, 10000 trials) ===\n")
np.random.seed(42)

for name, state in bell_states.items():
    probs = np.abs(state)**2  # Probabilities for |00>, |01>, |10>, |11>
    outcomes = np.random.choice(4, size=10000, p=probs)

    # Count correlations
    same = sum(1 for o in outcomes if o in [0, 3])    # |00> or |11> -> same
    diff = sum(1 for o in outcomes if o in [1, 2])     # |01> or |10> -> different
    print(f"{name}: Same={same/100:.1f}%, Different={diff/100:.1f}%  "
          f"({'correlated' if same > diff else 'anti-correlated'})")

# Correlation in X-basis
print("\n=== Correlation Patterns (X-basis, 10000 trials) ===\n")

# X-basis measurement: apply H to each qubit before Z-basis measurement
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
H_H = np.kron(H, H)  # H on both qubits

for name, state in bell_states.items():
    # Transform to X-basis
    state_x = H_H @ state
    probs = np.abs(state_x)**2
    outcomes = np.random.choice(4, size=10000, p=probs)

    same = sum(1 for o in outcomes if o in [0, 3])
    diff = sum(1 for o in outcomes if o in [1, 2])
    print(f"{name}: Same={same/100:.1f}%, Different={diff/100:.1f}%  "
          f"({'correlated' if same > diff else 'anti-correlated'})")
```

---

## 3. Creating Bell States

### 3.1 The Bell Circuit

All four Bell states can be created from computational basis states using just a Hadamard gate and a CNOT:

```
q0: ─[H]─●─
          │
q1: ──────X─
```

The mapping:

| Input | Output |
|-------|--------|
| $\|00\rangle$ | $\|\Phi^+\rangle = \frac{1}{\sqrt{2}}(\|00\rangle + \|11\rangle)$ |
| $\|01\rangle$ | $\|\Psi^+\rangle = \frac{1}{\sqrt{2}}(\|01\rangle + \|10\rangle)$ |
| $\|10\rangle$ | $\|\Phi^-\rangle = \frac{1}{\sqrt{2}}(\|00\rangle - \|11\rangle)$ |
| $\|11\rangle$ | $\|\Psi^-\rangle = \frac{1}{\sqrt{2}}(\|01\rangle - \|10\rangle)$ |

### 3.2 Step-by-Step Derivation

Starting from $|00\rangle$:

**Step 1**: Apply H to qubit 0:

$$H \otimes I |00\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \otimes |0\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)$$

**Step 2**: Apply CNOT (control=q0, target=q1):

$$\text{CNOT}\left[\frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)\right] = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = |\Phi^+\rangle$$

The CNOT copies the control qubit's value to the target (when target starts as $|0\rangle$), creating the correlation that is the hallmark of entanglement.

### 3.3 The Inverse: Bell Measurement

To measure in the Bell basis, reverse the circuit:

```
q0: ─●─[H]─[M]─
     │
q1: ─X─────[M]─
```

Apply CNOT then H, then measure in the computational basis. The measurement outcome tells you which Bell state the input was in.

```python
import numpy as np

# Creating Bell states step by step

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
I2 = np.eye(2, dtype=complex)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

# H on qubit 0, Identity on qubit 1
# Convention: |q1 q0>, so H on q0 = kron(I, H)
H_I = np.kron(I2, H)

# Bell circuit unitary
Bell_circuit = CNOT @ H_I

print("=== Bell State Creation ===\n")

# Apply to all four computational basis inputs
inputs = {
    "|00>": np.array([1, 0, 0, 0], dtype=complex),
    "|01>": np.array([0, 1, 0, 0], dtype=complex),
    "|10>": np.array([0, 0, 1, 0], dtype=complex),
    "|11>": np.array([0, 0, 0, 1], dtype=complex),
}

bell_names = {
    (1, 0, 0, 1): "|Phi+>",
    (0, 1, 1, 0): "|Psi+>",
    (1, 0, 0, -1): "|Phi->",
    (0, 1, -1, 0): "|Psi->",
}

for in_name, in_state in inputs.items():
    out_state = Bell_circuit @ in_state

    # Intermediate step: after H
    after_H = H_I @ in_state

    print(f"Input: {in_name}")
    print(f"  After H on q0: {np.round(after_H, 4)}")
    print(f"  After CNOT:    {np.round(out_state, 4)}")

    # Identify which Bell state
    rounded = tuple(np.round(out_state * np.sqrt(2)).real.astype(int))
    bell_name = bell_names.get(rounded, "unknown")
    print(f"  = {bell_name}")
    print()

# Verify the Bell circuit is unitary
print(f"Bell circuit is unitary? {np.allclose(Bell_circuit @ Bell_circuit.conj().T, np.eye(4))}")

# Inverse Bell circuit (for Bell measurement)
Bell_inverse = H_I @ CNOT  # Reverse order: CNOT first, then H

print(f"\n=== Bell Measurement (Inverse Circuit) ===\n")
bell_states_map = {
    "|Phi+>": np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
    "|Phi->": np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
    "|Psi+>": np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
    "|Psi->": np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),
}

for bell_name, bell_state in bell_states_map.items():
    measured = Bell_inverse @ bell_state
    # Find which computational basis state we get
    for i in range(4):
        if abs(measured[i]) > 0.9:
            label = format(i, '02b')
            print(f"{bell_name} -> |{label}> (measurement outcome: {label})")
```

---

## 4. The EPR Paradox

### 4.1 Einstein, Podolsky, and Rosen (1935)

In their famous 1935 paper, Einstein, Podolsky, and Rosen (EPR) argued that quantum mechanics must be *incomplete*. Their reasoning:

1. **Premise**: If we can predict the value of a physical quantity with certainty without disturbing the system, then there exists an "element of physical reality" corresponding to that quantity.

2. **Setup**: Prepare two particles in the entangled state $|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$ and send them far apart.

3. **Observation**: If we measure particle A in the $Z$-basis and get $|0\rangle$, we know with certainty that particle B is in $|1\rangle$ -- without touching particle B. By the premise, the $Z$-value of B is an element of reality.

4. **Similarly**: If we instead measure A in the $X$-basis and get $|+\rangle$, we know B is in $|-\rangle$. So the $X$-value of B is also an element of reality.

5. **Paradox**: Both $Z$ and $X$ values are "elements of reality" for particle B, but quantum mechanics says a qubit cannot simultaneously have definite values for both $Z$ and $X$ (they are complementary observables). Therefore, quantum mechanics must be incomplete.

### 4.2 The Hidden Variable Hypothesis

EPR's resolution was that there must be **hidden variables** -- additional information not captured by the quantum state -- that predetermine all measurement outcomes. The particles carry this hidden information from the moment they are created, like sealed envelopes containing pre-written instructions.

### 4.3 The Resolution

For 30 years, EPR's argument seemed philosophical rather than scientific -- there was no way to test whether hidden variables existed. Then in 1964, John Bell proved a remarkable theorem that changed everything (Section 5).

---

## 5. Bell's Theorem and the CHSH Inequality

### 5.1 Bell's Theorem (Informal Statement)

**No local hidden variable theory can reproduce all the predictions of quantum mechanics.**

This means: if we assume that (1) measurement outcomes are predetermined by hidden variables and (2) the measurement on one particle cannot instantaneously influence the other (locality), then there are specific *inequalities* that the measurement correlations must satisfy. Quantum mechanics *violates* these inequalities, and experiments confirm the quantum predictions.

### 5.2 The CHSH Inequality

The Clauser-Horne-Shimony-Holt (CHSH) inequality is the most practical form of Bell's theorem. Consider two parties, Alice and Bob, who each choose between two measurement settings:

- Alice measures $A_0$ or $A_1$ (two different observables)
- Bob measures $B_0$ or $B_1$ (two different observables)

Each measurement gives outcome $+1$ or $-1$.

Define the **CHSH quantity**:

$$S = \langle A_0 B_0 \rangle + \langle A_0 B_1 \rangle + \langle A_1 B_0 \rangle - \langle A_1 B_1 \rangle$$

where $\langle A_i B_j \rangle$ is the average product of Alice's and Bob's outcomes.

### 5.3 The Classical Bound

For any local hidden variable theory:

$$|S| \leq 2$$

This is the **CHSH inequality**. It holds for *any* pre-determined strategy the particles could carry.

### 5.4 The Quantum Violation

Quantum mechanics predicts that for the Bell state $|\Phi^+\rangle$ with optimal measurement choices:

$$S_{\text{quantum}} = 2\sqrt{2} \approx 2.828$$

This exceeds the classical bound of 2, proving that no local hidden variable theory can explain quantum correlations. The optimal measurement settings are:

- Alice: $A_0 = Z$, $A_1 = X$
- Bob: $B_0 = \frac{Z + X}{\sqrt{2}}$, $B_1 = \frac{Z - X}{\sqrt{2}}$

(Bob measures along axes rotated 45 degrees from Alice's.)

### 5.5 Tsirelson's Bound

The maximum value of $|S|$ achievable by quantum mechanics is exactly $2\sqrt{2}$ (Tsirelson's bound). Interestingly, the mathematical maximum (without any physical constraints) is 4, but neither classical nor quantum physics can reach it.

$$2 < 2\sqrt{2} < 4$$

$$\text{Classical} < \text{Quantum} < \text{Algebraic maximum}$$

```python
import numpy as np

# CHSH inequality simulation

def chsh_experiment(state_2q, alice_obs, bob_obs, n_trials=100000):
    """
    Simulate a CHSH experiment.

    Why simulate this? The CHSH inequality is the most important test
    distinguishing quantum mechanics from classical hidden variable theories.
    Simulating it lets us verify the quantum violation numerically.

    alice_obs, bob_obs: lists of two 2x2 observable matrices each
    Each observable has eigenvalues +1 and -1.
    """
    results = {}

    for i, A in enumerate(alice_obs):
        for j, B in enumerate(bob_obs):
            # Joint observable A tensor B
            AB = np.kron(A, B)

            # Expectation value: <psi|A(x)B|psi>
            # Why use this formula? For a pure state, the expectation value
            # of an observable is <psi|O|psi>. For the product of two
            # local observables, O = A tensor B.
            expectation = np.real(np.vdot(state_2q, AB @ state_2q))
            results[(i, j)] = expectation

    # CHSH quantity: S = <A0 B0> + <A0 B1> + <A1 B0> - <A1 B1>
    S = results[(0,0)] + results[(0,1)] + results[(1,0)] - results[(1,1)]
    return S, results

# Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

# Bell state |Phi+>
phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

# === Classical bound: Alice and Bob measure in the same bases ===
print("=== CHSH with Aligned Measurements ===\n")

alice_obs = [Z, X]
bob_obs = [Z, X]

S, results = chsh_experiment(phi_plus, alice_obs, bob_obs)
print("Alice: {Z, X}, Bob: {Z, X}")
for (i,j), val in results.items():
    print(f"  <A{i} B{j}> = {val:.4f}")
print(f"  S = {S:.4f}")
print(f"  |S| = {abs(S):.4f} (classical bound: 2)")

# === Quantum optimal: Bob's axes rotated 45 degrees ===
print("\n=== CHSH with Optimal Measurements (Tsirelson bound) ===\n")

# Bob's optimal observables: (Z+X)/sqrt(2) and (Z-X)/sqrt(2)
B0 = (Z + X) / np.sqrt(2)
B1 = (Z - X) / np.sqrt(2)

alice_obs = [Z, X]
bob_obs = [B0, B1]

S, results = chsh_experiment(phi_plus, alice_obs, bob_obs)
print("Alice: {Z, X}, Bob: {(Z+X)/sqrt(2), (Z-X)/sqrt(2)}")
for (i,j), val in results.items():
    print(f"  <A{i} B{j}> = {val:.4f}")
print(f"  S = {S:.4f}")
print(f"  |S| = {abs(S):.4f}")
print(f"  Classical bound: 2.0000")
print(f"  Tsirelson bound: {2*np.sqrt(2):.4f}")
print(f"\n  VIOLATION: |S| = {abs(S):.4f} > 2.0000!")
print(f"  No local hidden variable theory can produce this result.")

# === Monte Carlo simulation of the CHSH game ===
print("\n\n=== Monte Carlo CHSH Game ===\n")

def chsh_game_quantum(state_2q, alice_obs, bob_obs, n_rounds=10000):
    """
    Simulate actual measurement outcomes (not just expectation values).

    Why simulate actual measurements? The expectation value calculation
    above gives the exact quantum prediction. This simulation shows that
    individual measurement outcomes are random, but their CORRELATIONS
    violate the classical bound.
    """
    np.random.seed(42)
    total_S_contribution = 0

    for _ in range(n_rounds):
        # Randomly choose measurement settings
        i = np.random.randint(2)
        j = np.random.randint(2)

        A = alice_obs[i]
        B = bob_obs[j]

        # Joint measurement
        AB = np.kron(A, B)
        eig_vals, eig_vecs = np.linalg.eigh(AB)

        # Probability of each joint outcome
        probs = np.abs(eig_vecs.conj().T @ state_2q)**2
        outcome_idx = np.random.choice(4, p=probs)
        outcome_value = eig_vals[outcome_idx]

        # CHSH sign: +1 for (0,0), (0,1), (1,0), and -1 for (1,1)
        sign = -1 if (i == 1 and j == 1) else 1
        total_S_contribution += sign * outcome_value

    # Each setting pair is chosen ~n_rounds/4 times
    # S = sum of sign * outcome / (n_rounds/4)
    S_estimated = total_S_contribution / (n_rounds / 4)
    return S_estimated

S_mc = chsh_game_quantum(phi_plus, [Z, X], [B0, B1], n_rounds=40000)
print(f"Monte Carlo S estimate (40000 rounds): {S_mc:.4f}")
print(f"Exact quantum value: {2*np.sqrt(2):.4f}")
```

---

## 6. Monogamy of Entanglement

### 6.1 The Principle

**Monogamy of entanglement**: If qubit A is maximally entangled with qubit B, then A cannot be entangled with any other qubit C *at all*.

More precisely, for qubits A, B, C, the entanglement satisfies:

$$E(A:B) + E(A:C) \leq E(A:BC)$$

where $E$ is a measure of entanglement. For maximal $E(A:B)$, there is nothing left for $E(A:C)$.

### 6.2 Implications

Monogamy has profound consequences:

1. **Quantum cryptography**: If Alice and Bob share a maximally entangled pair, no eavesdropper Eve can be entangled with either of them. This is the basis for quantum key distribution security.

2. **No-cloning**: Monogamy is closely related to the no-cloning theorem. If we could clone qubit A's entanglement with B, we would violate monogamy.

3. **Entanglement distribution**: In a multi-qubit system, entanglement is a finite resource that must be "shared" carefully.

### 6.3 Contrast with Classical Correlations

Classical correlations are NOT monogamous. If Alice's coin is correlated with Bob's (e.g., both always land the same way), Alice's coin can also be perfectly correlated with Charlie's coin. Monogamy is a uniquely quantum phenomenon.

```python
import numpy as np

# Demonstrating monogamy: GHZ state vs W state

# GHZ state: (|000> + |111>)/sqrt(2)
# Qubit 0 is maximally correlated with qubits 1,2 jointly,
# but NOT maximally entangled with qubit 1 alone
ghz = np.zeros(8, dtype=complex)
ghz[0] = 1/np.sqrt(2)  # |000>
ghz[7] = 1/np.sqrt(2)  # |111>

# W state: (|001> + |010> + |100>)/sqrt(3)
# Entanglement is more "spread out" among all pairs
w = np.zeros(8, dtype=complex)
w[1] = 1/np.sqrt(3)  # |001>
w[2] = 1/np.sqrt(3)  # |010>
w[4] = 1/np.sqrt(3)  # |100>

def reduced_density_matrix(state_3q, trace_out_qubit):
    """
    Compute the reduced density matrix by tracing out one qubit from a 3-qubit state.

    Why reduced density matrices? When we have a multi-qubit system and want to
    describe only a subsystem, we need the reduced density matrix. For entangled
    states, the reduced density matrix is MIXED (not pure), and the degree of
    mixing quantifies the entanglement.
    """
    rho = np.outer(state_3q, state_3q.conj())  # Full 8x8 density matrix
    rho_reshaped = rho.reshape(2, 2, 2, 2, 2, 2)

    # Trace out the specified qubit
    # Qubit ordering: q2, q1, q0 (q0 is least significant)
    if trace_out_qubit == 0:
        # Trace out q0: sum over q0 indices
        reduced = np.trace(rho_reshaped, axis1=2, axis2=5)  # 4x4
    elif trace_out_qubit == 1:
        reduced = np.trace(rho_reshaped, axis1=1, axis2=4)  # 4x4
    else:  # trace_out_qubit == 2
        reduced = np.trace(rho_reshaped, axis1=0, axis2=3)  # 4x4

    return reduced.reshape(4, 4)

def von_neumann_entropy(rho):
    """Compute von Neumann entropy S = -Tr(rho * log2(rho))."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove zeros
    return -np.sum(eigenvalues * np.log2(eigenvalues))

print("=== Monogamy of Entanglement ===\n")
print("Comparing GHZ and W states:\n")

for name, state in [("GHZ", ghz), ("W", w)]:
    print(f"--- {name} State ---")
    for q in range(3):
        rho_pair = reduced_density_matrix(state, q)
        S = von_neumann_entropy(rho_pair)
        other_qubits = [i for i in range(3) if i != q]
        print(f"  Trace out q{q}: S(q{other_qubits[0]},q{other_qubits[1]}) = {S:.4f} bits")
    print()

print("GHZ: Qubits are maximally correlated collectively but each pair")
print("     has limited pairwise entanglement (monogamy constraint).")
print("W:   Entanglement is more evenly distributed among pairs.")
```

---

## 7. Entanglement as a Resource

### 7.1 Why Entanglement Matters for Quantum Computing

Entanglement is not just a curious physical phenomenon -- it is a *computational resource*:

1. **Exponential state space**: An $n$-qubit entangled state lives in a $2^n$-dimensional space that cannot be efficiently described by separate qubit descriptions.

2. **Quantum speedup**: Every known exponential quantum speedup requires entanglement. Without entanglement, quantum circuits can be efficiently simulated classically (Gottesman-Knill theorem for Clifford circuits, or tensor product structure for separable states).

3. **Quantum communication**: Entanglement enables teleportation (transferring a quantum state using entanglement + classical communication) and superdense coding (sending 2 classical bits using 1 qubit + entanglement).

### 7.2 Entanglement in Algorithms

In the quantum algorithms we will study:

- **Deutsch-Jozsa** ([Lesson 7](07_Deutsch_Jozsa_Algorithm.md)): The oracle creates entanglement between the query register and the ancilla qubit, enabling the interference that distinguishes constant from balanced functions.

- **Grover's search** ([Lesson 8](08_Grovers_Search.md)): The oracle and diffusion operator create and manipulate multi-qubit entanglement to amplify the correct answer's amplitude.

- **Shor's algorithm** (later lessons): Entanglement between the input and output registers of modular exponentiation enables the quantum Fourier transform to extract the period.

### 7.3 How Much Entanglement?

The **entanglement entropy** of a bipartite state $|\psi\rangle_{AB}$ is the von Neumann entropy of either reduced density matrix:

$$S(\rho_A) = -\text{Tr}(\rho_A \log_2 \rho_A)$$

For a maximally entangled state of two qubits, $S = 1$ bit. For a maximally entangled state of two $d$-dimensional systems, $S = \log_2 d$ bits.

---

## 8. Exercises

### Exercise 1: Bell State Identification

For each of the following states, determine which Bell state it is (or if it is not a Bell state):

a) $\frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$
b) $\frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$
c) $\frac{1}{\sqrt{2}}(|00\rangle + i|11\rangle)$
d) $\frac{1}{\sqrt{2}}(|+0\rangle + |-1\rangle)$ (Expand in the computational basis first.)

### Exercise 2: Entanglement Detection

Determine whether each state is separable or entangled. For entangled states, compute the entanglement entropy.

a) $\frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$
b) $\frac{1}{2}(|00\rangle + |01\rangle + |10\rangle - |11\rangle)$
c) $\frac{1}{\sqrt{2}}|00\rangle + \frac{1}{2}|01\rangle + \frac{1}{2}|11\rangle$

### Exercise 3: CHSH Calculation

For the Bell state $|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$:

a) Compute $\langle Z \otimes Z \rangle$ (expectation value of $Z$ on both qubits).
b) Compute $\langle X \otimes X \rangle$.
c) Find measurement settings for Alice and Bob that maximize the CHSH quantity $S$ for this state.

### Exercise 4: Three-Qubit Entanglement

a) Create the GHZ state $\frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$ using a quantum circuit (specify the gates).
b) Create the W state $\frac{1}{\sqrt{3}}(|001\rangle + |010\rangle + |100\rangle)$. This is harder -- it requires controlled rotations.
c) Measure both states in the computational basis 10,000 times and compare the outcome distributions.

### Exercise 5: Entanglement Swapping

Consider the state $|\Phi^+\rangle_{12} \otimes |\Phi^+\rangle_{34}$ (two independent Bell pairs). If we perform a Bell measurement on qubits 2 and 3, what happens to qubits 1 and 4? Show that qubits 1 and 4 become entangled, even though they never directly interacted. This is **entanglement swapping**. (Hint: rewrite the full 4-qubit state in the Bell basis for qubits 2,3.)

---

[<- Previous: Quantum Circuits](04_Quantum_Circuits.md) | [Next: Quantum Measurement ->](06_Quantum_Measurement.md)
