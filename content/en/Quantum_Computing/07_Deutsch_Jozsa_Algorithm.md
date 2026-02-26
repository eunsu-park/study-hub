# Lesson 7: Deutsch-Jozsa Algorithm

[<- Previous: Quantum Measurement](06_Quantum_Measurement.md) | [Next: Grover's Search Algorithm ->](08_Grovers_Search.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define the Deutsch-Jozsa promise problem and explain the distinction between constant and balanced functions
2. Explain the quantum oracle model and why it is important for analyzing quantum algorithms
3. Derive and implement Deutsch's algorithm (the 1-qubit version) step by step
4. Generalize to the full Deutsch-Jozsa algorithm for $n$-qubit inputs
5. Walk through a complete example of the algorithm with a 2-qubit oracle
6. Explain why the Deutsch-Jozsa algorithm demonstrates quantum advantage and its limitations
7. Construct oracle circuits for specific functions and simulate the full algorithm in Python

---

The Deutsch-Jozsa algorithm, proposed by David Deutsch and Richard Jozsa in 1992, was the first algorithm to demonstrate an exponential separation between quantum and classical computation. While the problem it solves is admittedly artificial, the algorithm introduces several ideas that are central to quantum computing: the oracle model, phase kickback, and the power of quantum parallelism combined with interference.

The problem is simple: given a function $f: \{0,1\}^n \to \{0,1\}$ that is either *constant* (same output for all inputs) or *balanced* (outputs 0 for exactly half the inputs and 1 for the other half), determine which type it is. Classically, you need to evaluate $f$ on at least $2^{n-1} + 1$ inputs in the worst case. Quantum mechanically, you need exactly *one* evaluation.

> **Analogy:** Imagine determining if a coin is fair or double-headed -- classically you might need many flips, but quantum mechanics lets you check in a single toss. The coin is like the function $f$: it either always gives the same answer (constant/double-headed) or gives balanced answers (fair). The quantum "toss" queries all possible inputs simultaneously through superposition, and interference makes the answer evident in a single measurement.

## Table of Contents

1. [The Promise Problem](#1-the-promise-problem)
2. [The Oracle Model](#2-the-oracle-model)
3. [Deutsch's Algorithm (1 Qubit)](#3-deutschs-algorithm-1-qubit)
4. [The Deutsch-Jozsa Algorithm (n Qubits)](#4-the-deutsch-jozsa-algorithm-n-qubits)
5. [Step-by-Step Example](#5-step-by-step-example)
6. [Why It Matters](#6-why-it-matters)
7. [Exercises](#7-exercises)

---

## 1. The Promise Problem

### 1.1 Problem Statement

**Input**: A function $f: \{0,1\}^n \to \{0,1\}$ with the promise that $f$ is either:
- **Constant**: $f(x) = 0$ for all $x$, or $f(x) = 1$ for all $x$
- **Balanced**: $f(x) = 0$ for exactly $2^{n-1}$ inputs and $f(x) = 1$ for the other $2^{n-1}$ inputs

**Output**: Determine whether $f$ is constant or balanced.

### 1.2 Classical Complexity

Classically, we must evaluate $f$ on different inputs until we can determine the answer:
- **Best case**: We get lucky and see two different outputs after 2 queries. Answer: balanced.
- **Worst case**: We evaluate $2^{n-1} + 1$ inputs. If all give the same output, we know by the pigeonhole principle that $f$ cannot be balanced, so it must be constant.
- **Average case**: With randomization, $O(1)$ queries suffice with high probability (but not certainty).

The key point: for *deterministic* classical computation, the worst case requires $2^{n-1} + 1$ queries -- exponential in $n$.

### 1.3 Quantum Complexity

The Deutsch-Jozsa algorithm determines the answer with certainty using exactly **one** query to $f$. This is an exponential speedup over deterministic classical computation.

Note: The speedup is only over *deterministic* classical algorithms. A randomized classical algorithm can solve this with $O(1)$ queries with high probability. So the Deutsch-Jozsa algorithm demonstrates an exponential gap for exact (zero-error) algorithms, not for bounded-error algorithms. This is still important as a proof of concept.

```python
import numpy as np

# Classical Deutsch-Jozsa: demonstrating the classical query complexity

def classical_deutsch_jozsa(f, n):
    """
    Classical algorithm for Deutsch-Jozsa problem.

    Why do we need 2^{n-1} + 1 queries in the worst case?
    Because the first 2^{n-1} evaluations could all give the same value,
    which is consistent with BOTH constant and balanced. Only the
    (2^{n-1} + 1)-th query can distinguish: if it also gives the same
    value, f must be constant (a balanced function can't have more than
    2^{n-1} identical outputs by definition).
    """
    first_value = f(0)
    queries = 1

    for x in range(1, 2**n):
        if f(x) != first_value:
            return "balanced", queries + 1
        queries += 1
        # Early termination: if we've checked half + 1 and all same, must be constant
        if queries > 2**(n-1):
            return "constant", queries

    return "constant", queries

# Example functions for n=3
n = 3

# Constant function: f(x) = 0 for all x
f_constant = lambda x: 0

# Balanced function: f(x) = parity of x (balanced for power-of-2 domain)
f_balanced = lambda x: bin(x).count('1') % 2

# Worst-case constant: all outputs are 0
result, queries = classical_deutsch_jozsa(f_constant, n)
print(f"=== Classical Deutsch-Jozsa (n={n}, 2^n={2**n} inputs) ===\n")
print(f"Constant f(x)=0: result={result}, queries={queries}")
print(f"  (Worst case: had to query {queries} times out of {2**n})")

result, queries = classical_deutsch_jozsa(f_balanced, n)
print(f"Balanced f(x)=parity(x): result={result}, queries={queries}")
print(f"  (Got lucky: found disagreement after {queries} queries)")

# Show why worst case is 2^{n-1}+1
print(f"\n  Theoretical worst case: 2^{n-1}+1 = {2**(n-1)+1} queries")
print(f"  Total possible inputs: 2^{n} = {2**n}")
```

---

## 2. The Oracle Model

### 2.1 What Is an Oracle?

In quantum computing, we access the function $f$ through a **quantum oracle** (also called a "black box"). The oracle is a unitary operator $U_f$ that encodes $f$ in a reversible way.

The standard oracle takes an $n$-qubit input register $|x\rangle$ and a 1-qubit output register $|y\rangle$:

$$U_f|x\rangle|y\rangle = |x\rangle|y \oplus f(x)\rangle$$

where $\oplus$ is addition modulo 2 (XOR).

### 2.2 Why XOR?

The XOR construction ensures the oracle is *reversible* (and therefore unitary):
- $U_f U_f |x\rangle|y\rangle = |x\rangle|y \oplus f(x) \oplus f(x)\rangle = |x\rangle|y\rangle$
- So $U_f^{-1} = U_f$ (the oracle is its own inverse)

### 2.3 Phase Oracle (Phase Kickback)

A key technique in quantum algorithms is using the oracle to impart a *phase* rather than flipping a bit. Set the output qubit to $|-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$:

$$U_f|x\rangle|-\rangle = |x\rangle|f(x) \oplus -\rangle$$

But $|b \oplus -\rangle = (-1)^b|-\rangle$ for $b \in \{0, 1\}$ (verify this!), so:

$$U_f|x\rangle|-\rangle = (-1)^{f(x)}|x\rangle|-\rangle$$

The oracle has "kicked back" a phase of $(-1)^{f(x)}$ onto the input register, while the output qubit remains in $|-\rangle$ unchanged. This **phase kickback** trick converts the oracle from a bit-flip operation into a phase operation, which is essential for interference.

```python
import numpy as np

# Oracle construction

def build_oracle_matrix(f, n):
    """
    Build the unitary matrix for oracle U_f on n+1 qubits.
    U_f|x>|y> = |x>|y XOR f(x)>

    Why build the full matrix? For small n, this is the clearest way
    to understand what the oracle does. Each column of the matrix shows
    what happens to each basis state.
    """
    dim = 2**(n+1)  # n input qubits + 1 output qubit
    U = np.zeros((dim, dim), dtype=complex)

    for x in range(2**n):
        for y in range(2):
            # Input basis state: |x>|y>
            # Index in state vector: x * 2 + y
            in_idx = x * 2 + y

            # Output: |x>|y XOR f(x)>
            out_y = y ^ f(x)
            out_idx = x * 2 + out_y

            U[out_idx, in_idx] = 1

    return U

def build_phase_oracle(f, n):
    """
    Build the phase oracle: U_f|x> = (-1)^f(x) |x>
    This is the oracle after phase kickback (output qubit traced out).

    Why a separate function? In the Deutsch-Jozsa algorithm, the output
    qubit stays in |-> throughout and can be ignored. The effective
    operation on the input register is just a diagonal matrix of phases.
    """
    dim = 2**n
    U = np.zeros((dim, dim), dtype=complex)

    for x in range(dim):
        U[x, x] = (-1)**f(x)

    return U

# Example: n=2, balanced function f(x) = x0 (least significant bit)
n = 2
f_balanced = lambda x: x & 1  # f(00)=0, f(01)=1, f(10)=0, f(11)=1

print("=== Oracle for f(x) = x0 (LSB), n=2 ===\n")

# Full oracle
U_f = build_oracle_matrix(f_balanced, n)
print(f"Full oracle U_f ({2**(n+1)}x{2**(n+1)} matrix):")
print(np.real(U_f).astype(int))

# Phase oracle
U_phase = build_phase_oracle(f_balanced, n)
print(f"\nPhase oracle ({2**n}x{2**n} diagonal matrix):")
print(f"Diagonal: {np.diag(U_phase).real}")
print("(-1)^f(x) for x = 00, 01, 10, 11:", [(-1)**f_balanced(x) for x in range(4)])

# Verify: phase kickback
print("\n=== Phase Kickback Demonstration ===\n")
ket_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)

for x in range(2**n):
    # Prepare |x>|->
    ket_x = np.zeros(2**n, dtype=complex)
    ket_x[x] = 1
    input_state = np.kron(ket_x, ket_minus)

    # Apply oracle
    output_state = U_f @ input_state

    # The expected result: (-1)^f(x) |x>|->
    expected = (-1)**f_balanced(x) * input_state

    match = np.allclose(output_state, expected)
    print(f"|{format(x, f'0{n}b')}>|->  ->  (-1)^{f_balanced(x)} |{format(x, f'0{n}b')}>|->  "
          f"{'OK' if match else 'FAIL'}")
```

---

## 3. Deutsch's Algorithm (1 Qubit)

### 3.1 The Simplest Case

Deutsch's algorithm (1985) is the $n = 1$ version. The function $f: \{0, 1\} \to \{0, 1\}$ has four possibilities:

| Function | $f(0)$ | $f(1)$ | Type |
|----------|--------|--------|------|
| $f_1$ | 0 | 0 | Constant |
| $f_2$ | 1 | 1 | Constant |
| $f_3$ | 0 | 1 | Balanced |
| $f_4$ | 1 | 0 | Balanced |

Classically, we need to evaluate $f$ on both inputs to determine the type. Deutsch's algorithm does it with one query.

### 3.2 The Circuit

```
q0: ─[H]─────[U_f]─[H]─[M]─
              │
q1: ─[X]─[H]─[U_f]─────────
```

1. Start: $|01\rangle$ (q0 = $|0\rangle$, q1 = $|1\rangle$ after X gate)
2. Apply H to both qubits
3. Apply oracle $U_f$
4. Apply H to q0
5. Measure q0

### 3.3 Step-by-Step Derivation

**Step 0**: Initialize $|0\rangle|1\rangle$

**Step 1**: Apply H to both qubits:

$$|0\rangle|1\rangle \xrightarrow{H \otimes H} |+\rangle|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \otimes \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

**Step 2**: Apply $U_f$ (using phase kickback):

$$\frac{1}{\sqrt{2}}\left[(-1)^{f(0)}|0\rangle + (-1)^{f(1)}|1\rangle\right] \otimes |-\rangle$$

**Step 3**: Focus on q0 (ignoring q1 which stays in $|-\rangle$):

$$\frac{1}{\sqrt{2}}\left[(-1)^{f(0)}|0\rangle + (-1)^{f(1)}|1\rangle\right]$$

Factor out $(-1)^{f(0)}$:

$$(-1)^{f(0)} \cdot \frac{1}{\sqrt{2}}\left[|0\rangle + (-1)^{f(0) \oplus f(1)}|1\rangle\right]$$

**Step 4**: Apply H to q0:

$$(-1)^{f(0)} \cdot H\left[\frac{1}{\sqrt{2}}\left(|0\rangle + (-1)^{f(0) \oplus f(1)}|1\rangle\right)\right]$$

Using $H|+\rangle = |0\rangle$ and $H|-\rangle = |1\rangle$:

- If $f(0) \oplus f(1) = 0$ (constant): $H|+\rangle = |0\rangle$
- If $f(0) \oplus f(1) = 1$ (balanced): $H|-\rangle = |1\rangle$

**Step 5**: Measure q0:
- **Result 0**: $f$ is constant
- **Result 1**: $f$ is balanced

One query. 100% correct. Always.

```python
import numpy as np

# Deutsch's algorithm implementation

def deutsch_algorithm(f):
    """
    Deutsch's algorithm for n=1 Deutsch-Jozsa problem.

    Why does this work? The key insight is phase kickback + interference.
    1. Hadamard creates superposition: query |0> and |1> simultaneously
    2. Phase kickback: oracle imparts (-1)^f(x) phases
    3. Second Hadamard: interference between the two amplitudes
       - If f(0)=f(1) (constant): amplitudes add constructively -> |0>
       - If f(0)!=f(1) (balanced): amplitudes cancel -> |1>
    """
    # Step 1: Initial state |01>
    # After H on both: |+>|->
    # After oracle with phase kickback: state of qubit 0 is
    # ((-1)^f(0)|0> + (-1)^f(1)|1>) / sqrt(2)
    # After H on qubit 0: measure

    # Direct computation
    state_q0 = np.array([(-1)**f(0), (-1)**f(1)], dtype=complex) / np.sqrt(2)

    # Apply Hadamard
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    final = H @ state_q0

    # Measure: probability of |0> and |1>
    p0 = abs(final[0])**2
    p1 = abs(final[1])**2

    measurement = 0 if p0 > 0.5 else 1
    result = "constant" if measurement == 0 else "balanced"

    return result, p0, p1

# Test all four possible functions
print("=== Deutsch's Algorithm ===\n")
functions = {
    "f1: f(0)=0, f(1)=0": lambda x: 0,
    "f2: f(0)=1, f(1)=1": lambda x: 1,
    "f3: f(0)=0, f(1)=1": lambda x: x,
    "f4: f(0)=1, f(1)=0": lambda x: 1 - x,
}

for name, f in functions.items():
    result, p0, p1 = deutsch_algorithm(f)
    print(f"{name}")
    print(f"  P(0) = {p0:.4f}, P(1) = {p1:.4f}")
    print(f"  Result: {result}")
    expected = "constant" if f(0) == f(1) else "balanced"
    print(f"  Correct? {result == expected}")
    print()

# Full circuit simulation (including ancilla qubit)
print("=== Full Circuit Simulation ===\n")

def deutsch_full_simulation(f):
    """Simulate the complete 2-qubit circuit."""
    # Gates
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    # Oracle matrix for n=1
    U_f = np.zeros((4, 4), dtype=complex)
    for x in range(2):
        for y in range(2):
            in_idx = x * 2 + y
            out_idx = x * 2 + (y ^ f(x))
            U_f[out_idx, in_idx] = 1

    # Initial state: |0>|0>
    state = np.array([1, 0, 0, 0], dtype=complex)

    # Apply X to qubit 1 (ancilla): |0>|0> -> |0>|1>
    state = np.kron(I2, X) @ state

    # Apply H to both
    state = np.kron(H, H) @ state

    # Apply oracle
    state = U_f @ state

    # Apply H to qubit 0 (input)
    state = np.kron(H, I2) @ state

    # Measure qubit 0
    # P(q0=0) = |<00|state>|^2 + |<01|state>|^2
    p_q0_0 = abs(state[0])**2 + abs(state[1])**2
    p_q0_1 = abs(state[2])**2 + abs(state[3])**2

    return "constant" if p_q0_0 > 0.5 else "balanced"

# Verify full simulation matches simplified version
for name, f in functions.items():
    result = deutsch_full_simulation(f)
    expected = "constant" if f(0) == f(1) else "balanced"
    print(f"{name}: {result} ({'CORRECT' if result == expected else 'WRONG'})")
```

---

## 4. The Deutsch-Jozsa Algorithm (n Qubits)

### 4.1 The Circuit

```
         ┌───┐     ┌─────┐ ┌───┐ ┌───┐
q0:   |0>┤ H ├─────┤     ├─┤ H ├─┤ M ├
         ├───┤     │     │ ├───┤ ├───┤
q1:   |0>┤ H ├─────┤     ├─┤ H ├─┤ M ├
         ├───┤     │ U_f │ ├───┤ ├───┤
...       ...      │     │  ...   ...
         ├───┤     │     │ ├───┤ ├───┤
q_{n-1}|0>┤ H ├───┤     ├─┤ H ├─┤ M ├
         ├───┤     │     │ └───┘ └───┘
ancilla|1>┤ H ├───┤     ├──────────────
         └───┘     └─────┘
```

### 4.2 Mathematical Derivation

**Step 1**: Start with $|0\rangle^{\otimes n}|1\rangle$

**Step 2**: Apply $H^{\otimes (n+1)}$:

$$\frac{1}{\sqrt{2^n}} \sum_{x=0}^{2^n-1} |x\rangle \otimes |-\rangle$$

**Step 3**: Apply $U_f$ (with phase kickback):

$$\frac{1}{\sqrt{2^n}} \sum_{x=0}^{2^n-1} (-1)^{f(x)} |x\rangle \otimes |-\rangle$$

**Step 4**: Apply $H^{\otimes n}$ to the input register. Using the identity:

$$H^{\otimes n}|x\rangle = \frac{1}{\sqrt{2^n}} \sum_{z=0}^{2^n-1} (-1)^{x \cdot z}|z\rangle$$

where $x \cdot z = \bigoplus_i x_i z_i$ is the bitwise inner product modulo 2:

$$\frac{1}{2^n} \sum_{z=0}^{2^n-1} \left[\sum_{x=0}^{2^n-1} (-1)^{f(x) + x \cdot z}\right] |z\rangle \otimes |-\rangle$$

**Step 5**: Measure the input register. The probability of outcome $|0\rangle^{\otimes n}$ is:

$$P(0^n) = \left|\frac{1}{2^n} \sum_{x=0}^{2^n-1} (-1)^{f(x)}\right|^2$$

**Key insight**:
- If $f$ is **constant**: All $(-1)^{f(x)}$ terms have the same sign. The sum is $\pm 2^n$. So $P(0^n) = 1$.
- If $f$ is **balanced**: Half the terms are $+1$ and half are $-1$. The sum is 0. So $P(0^n) = 0$.

**Result**: Measure $|0\rangle^{\otimes n}$ with probability 1 if constant, probability 0 if balanced. One query. Deterministic.

### 4.3 The Power of Interference

The algorithm works because of *constructive* and *destructive* interference:
- For a constant function, all terms interfere constructively at $|0^n\rangle$
- For a balanced function, the terms perfectly cancel at $|0^n\rangle$

This is the fundamental pattern of quantum algorithms: arrange the computation so that wrong answers interfere destructively and the right answer interferes constructively.

```python
import numpy as np

def deutsch_jozsa(f, n):
    """
    Full Deutsch-Jozsa algorithm simulation.

    Why this structure? The algorithm has three phases:
    1. SUPERPOSITION: H gates put the input into equal superposition
       of all 2^n inputs -- "querying all inputs simultaneously"
    2. PHASE MARKING: Oracle marks each input x with (-1)^f(x) phase
    3. INTERFERENCE: H gates cause constructive/destructive interference
       that concentrates amplitude on |0...0> for constant functions

    The measurement then deterministically distinguishes the two cases.
    """
    dim = 2**n

    # Step 1: Create uniform superposition
    # |psi_0> = H^n |0...0> = (1/sqrt(2^n)) sum_x |x>
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)

    # Step 2: Apply phase oracle
    # |psi_1> = (1/sqrt(2^n)) sum_x (-1)^f(x) |x>
    for x in range(dim):
        state[x] *= (-1)**f(x)

    # Step 3: Apply H^n
    # Build H^n matrix
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    H_n = np.array([[1]], dtype=complex)
    for _ in range(n):
        H_n = np.kron(H_n, H)

    state = H_n @ state

    # Step 4: Measure
    # Probability of |0...0>
    p_zero = abs(state[0])**2

    # All probabilities
    probs = np.abs(state)**2

    return {
        'p_all_zeros': p_zero,
        'result': 'constant' if p_zero > 0.5 else 'balanced',
        'state': state,
        'probabilities': probs,
    }

# Test with various functions
print("=== Deutsch-Jozsa Algorithm ===\n")

for n in [2, 3, 4]:
    print(f"--- n = {n} ({2**n} inputs) ---\n")

    # Constant function: f(x) = 0
    f_const = lambda x: 0
    result = deutsch_jozsa(f_const, n)
    print(f"  f(x) = 0 (constant):")
    print(f"    P(0...0) = {result['p_all_zeros']:.6f}")
    print(f"    Result: {result['result']}")

    # Constant function: f(x) = 1
    f_const1 = lambda x: 1
    result = deutsch_jozsa(f_const1, n)
    print(f"  f(x) = 1 (constant):")
    print(f"    P(0...0) = {result['p_all_zeros']:.6f}")
    print(f"    Result: {result['result']}")

    # Balanced function: f(x) = MSB of x
    f_balanced = lambda x, n=n: (x >> (n-1)) & 1
    result = deutsch_jozsa(f_balanced, n)
    print(f"  f(x) = MSB(x) (balanced):")
    print(f"    P(0...0) = {result['p_all_zeros']:.6f}")
    print(f"    Result: {result['result']}")

    # Balanced function: f(x) = parity of x
    f_parity = lambda x: bin(x).count('1') % 2
    result = deutsch_jozsa(f_parity, n)
    print(f"  f(x) = parity(x) (balanced):")
    print(f"    P(0...0) = {result['p_all_zeros']:.6f}")
    print(f"    Result: {result['result']}")
    print()
```

---

## 5. Step-by-Step Example

Let us trace through the algorithm with $n = 2$ and the balanced function $f(x) = x_0$ (least significant bit).

### 5.1 The Function

| $x$ (binary) | $x$ (decimal) | $f(x) = x_0$ |
|:---:|:---:|:---:|
| 00 | 0 | 0 |
| 01 | 1 | 1 |
| 10 | 2 | 0 |
| 11 | 3 | 1 |

This is balanced: $f$ outputs 0 for $\{00, 10\}$ and 1 for $\{01, 11\}$.

### 5.2 Detailed State Evolution

**Step 1**: Superposition:

$$|\psi_0\rangle = H^{\otimes 2}|00\rangle = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$$

**Step 2**: Oracle applies phases:

$$|\psi_1\rangle = \frac{1}{2}\left[(-1)^0|00\rangle + (-1)^1|01\rangle + (-1)^0|10\rangle + (-1)^1|11\rangle\right]$$

$$= \frac{1}{2}(|00\rangle - |01\rangle + |10\rangle - |11\rangle)$$

**Step 3**: Apply $H^{\otimes 2}$:

Recall $H^{\otimes 2}$ maps:
- $|00\rangle \to \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$
- $|01\rangle \to \frac{1}{2}(|00\rangle - |01\rangle + |10\rangle - |11\rangle)$
- $|10\rangle \to \frac{1}{2}(|00\rangle + |01\rangle - |10\rangle - |11\rangle)$
- $|11\rangle \to \frac{1}{2}(|00\rangle - |01\rangle - |10\rangle + |11\rangle)$

Computing $H^{\otimes 2}|\psi_1\rangle$:

$$= \frac{1}{2}\left[\frac{1}{2}(+,+,+,+) - \frac{1}{2}(+,-,+,-) + \frac{1}{2}(+,+,-,-) - \frac{1}{2}(+,-,-,+)\right]$$

where $(a,b,c,d)$ means $a|00\rangle + b|01\rangle + c|10\rangle + d|11\rangle$.

Coefficient of $|00\rangle$: $\frac{1}{4}(1 - 1 + 1 - 1) = 0$
Coefficient of $|01\rangle$: $\frac{1}{4}(1 + 1 + 1 + 1) = 1$
Coefficient of $|10\rangle$: $\frac{1}{4}(1 - 1 - 1 + 1) = 0$
Coefficient of $|11\rangle$: $\frac{1}{4}(1 + 1 - 1 - 1) = 0$

**Result**: $|\psi_2\rangle = |01\rangle$

$P(00) = 0$ -- not all zeros, so the function is balanced. Correct!

```python
import numpy as np

# Detailed step-by-step trace for n=2

n = 2
f = lambda x: x & 1  # f(x) = x0 (LSB)

print("=== Step-by-Step Trace: n=2, f(x) = x0 ===\n")

# Standard gates
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
H2 = np.kron(H, H)

# Step 1: Superposition
state = np.array([1, 0, 0, 0], dtype=complex)  # |00>
state = H2 @ state
print("Step 1: H tensor H on |00>")
print(f"  |psi_0> = {state}")
print(f"  = (1/2)(|00> + |01> + |10> + |11>)\n")

# Step 2: Oracle
print("Step 2: Apply phase oracle f(x) = x0")
print("  Phases: ", end="")
for x in range(4):
    phase = (-1)**f(x)
    state[x] *= phase
    print(f"(-1)^{f(x)}={phase:+d} for |{format(x, '02b')}>", end="  ")
print()
print(f"  |psi_1> = {state}")
print(f"  = (1/2)(|00> - |01> + |10> - |11>)\n")

# Step 3: H tensor H
state = H2 @ state
print("Step 3: Apply H tensor H")
print(f"  |psi_2> = {np.round(state, 6)}")
print()

# Show result
for i in range(4):
    label = format(i, '02b')
    if abs(state[i]) > 1e-10:
        print(f"  |{label}>: amplitude = {state[i]:.4f}, P = {abs(state[i])**2:.4f}")
    else:
        print(f"  |{label}>: amplitude = {state[i]:.4f}, P = {abs(state[i])**2:.4f} (zero!)")

p_00 = abs(state[0])**2
print(f"\nP(00) = {p_00:.6f}")
print(f"Result: {'constant' if p_00 > 0.5 else 'balanced'}")
print(f"Expected: balanced")
print(f"\nThe amplitude at |00> is ZERO because destructive interference")
print(f"perfectly canceled it. This is the quantum advantage!")
```

---

## 6. Why It Matters

### 6.1 Historical Significance

The Deutsch-Jozsa algorithm (and its predecessor, Deutsch's algorithm) was the first to demonstrate that quantum computers can solve *some* problems with exponentially fewer queries than classical computers. It proved that quantum computation is not merely a theoretical curiosity but a genuinely different computational model.

### 6.2 Conceptual Contributions

The algorithm introduced several ideas used in virtually all subsequent quantum algorithms:

| Concept | Description | Used in |
|---------|-------------|---------|
| **Oracle model** | Black-box access to function | Grover, Shor, many others |
| **Phase kickback** | Convert bit oracle to phase oracle | Grover, QPE, Shor |
| **Quantum parallelism** | Superposition queries all inputs | All quantum algorithms |
| **Interference** | Constructive for right answer | Grover, Shor, VQE |
| **Hadamard sandwich** | $H^n \to \text{Oracle} \to H^n$ | Bernstein-Vazirani, Simon's |

### 6.3 Limitations

- The problem is artificial (the "promise" that $f$ is constant or balanced is strong)
- Randomized classical algorithms solve it efficiently (BPP perspective)
- No practical application (no one needs to distinguish constant from balanced functions)
- The advantage is exact (zero error) vs bounded error, not practical speedup

### 6.4 What Comes Next

The ideas from Deutsch-Jozsa directly lead to:
- **Bernstein-Vazirani algorithm**: Finds a hidden string $s$ where $f(x) = s \cdot x$ in one query
- **Simon's algorithm**: Finds period of a function with exponential speedup
- **Grover's search** ([Lesson 8](08_Grovers_Search.md)): Quadratic speedup for unstructured search
- **Shor's algorithm**: Exponential speedup for factoring (builds on Simon's ideas)

```python
import numpy as np

# Comparing classical and quantum query complexity

print("=== Query Complexity Comparison ===\n")
print(f"{'n':>5} {'2^n':>10} {'Classical (det)':>18} {'Classical (rand)':>18} {'Quantum':>10}")
print("-" * 65)

for n in range(1, 16):
    dim = 2**n
    classical_det = 2**(n-1) + 1
    classical_rand = 2  # O(1) queries with high probability
    quantum = 1

    print(f"{n:>5} {dim:>10,} {classical_det:>18,} {classical_rand:>18} {quantum:>10}")

print(f"\nThe quantum advantage is EXPONENTIAL over deterministic classical,")
print(f"but only CONSTANT over randomized classical.")
print(f"\nThis illustrates an important nuance: quantum speedups depend on")
print(f"what classical model you compare against.")

# Bonus: Bernstein-Vazirani (closely related algorithm)
print("\n\n=== Bonus: Bernstein-Vazirani Algorithm ===\n")
print("Problem: Find hidden string s where f(x) = s . x (mod 2)")
print("Classical: n queries. Quantum: 1 query.\n")

def bernstein_vazirani(f, n):
    """
    Bernstein-Vazirani algorithm.

    Why is this related to Deutsch-Jozsa? It uses the exact same circuit!
    The only difference is the function class: f(x) = s.x (bitwise inner
    product with a hidden string s). The same H-oracle-H structure
    directly reveals s in one shot.
    """
    dim = 2**n
    # Same circuit as Deutsch-Jozsa!
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)
    for x in range(dim):
        state[x] *= (-1)**f(x)
    H_n = np.array([[1]], dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    for _ in range(n):
        H_n = np.kron(H_n, H)
    state = H_n @ state
    # The state is now |s> with probability 1
    s = np.argmax(np.abs(state)**2)
    return format(s, f'0{n}b')

# Test
n = 5
secret = 0b10110  # Hidden string "10110"

def f_bv(x, s=secret):
    """f(x) = s . x (bitwise inner product mod 2)."""
    return bin(x & s).count('1') % 2

result = bernstein_vazirani(f_bv, n)
print(f"Hidden string s = {format(secret, f'0{n}b')}")
print(f"Algorithm found: {result}")
print(f"Correct? {result == format(secret, f'0{n}b')}")
print(f"Queries used: 1 (classical would need {n})")
```

---

## 7. Exercises

### Exercise 1: Oracle Construction

Construct the phase oracle matrix for the following functions on $n = 2$ qubits:

a) $f(x) = 1$ for all $x$ (constant)
b) $f(x) = x_1$ (most significant bit)
c) $f(x) = x_0 \oplus x_1$ (XOR of bits)

For each, run the Deutsch-Jozsa algorithm and verify the output.

### Exercise 2: Scaling Analysis

a) Implement the Deutsch-Jozsa algorithm for $n = 1$ through $n = 15$.
b) For each $n$, verify that $P(0^n) = 1$ for constant functions and $P(0^n) = 0$ for balanced functions.
c) Measure the classical simulation time for each $n$. At what $n$ does the simulation become slow? (This illustrates why we need real quantum hardware!)

### Exercise 3: Non-Promise Functions

What happens if we run the Deutsch-Jozsa algorithm on a function that is neither constant nor balanced? For example, $f(x) = 1$ if $x < 2^{n-1}/3$ and $f(x) = 0$ otherwise.

a) Compute $P(0^n)$ analytically.
b) Verify with simulation.
c) Can we still extract useful information from the output?

### Exercise 4: Implementing the Full Circuit

Using the `QuantumCircuit` simulator from [Lesson 4](04_Quantum_Circuits.md), implement the full Deutsch-Jozsa circuit (including the ancilla qubit) for $n = 3$. Run it on at least two constant and two balanced functions, and verify the results match the theory.

### Exercise 5: Bernstein-Vazirani Extension

a) Implement the Bernstein-Vazirani algorithm for $n = 8$ and verify it correctly finds hidden strings.
b) Can you modify the algorithm to handle the case where $f(x) = s \cdot x \oplus b$ for an unknown bit $b$? How does this affect the output?
c) Compare the quantum query count (1) with the classical count ($n$) for various $n$. Why does this gap matter less than the Deutsch-Jozsa gap in practical terms?

---

[<- Previous: Quantum Measurement](06_Quantum_Measurement.md) | [Next: Grover's Search Algorithm ->](08_Grovers_Search.md)
