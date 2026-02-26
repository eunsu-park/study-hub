# Lesson 8: Grover's Search Algorithm

[<- Previous: Deutsch-Jozsa Algorithm](07_Deutsch_Jozsa_Algorithm.md) | [Next: Quantum Fourier Transform ->](09_Quantum_Fourier_Transform.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define the unstructured search problem and explain the classical lower bound of $O(N)$ queries
2. State Grover's quantum bound of $O(\sqrt{N})$ queries and explain why this quadratic speedup is provably optimal
3. Construct the Grover oracle (phase marking) for specific target elements
4. Derive and implement the diffusion operator (inversion about the mean)
5. Explain the geometric interpretation of Grover's algorithm as rotation in a 2D subspace
6. Calculate the optimal number of iterations $\lfloor \pi/4 \cdot \sqrt{N} \rfloor$ and handle the multi-solution case
7. Implement and visualize the complete Grover algorithm in Python, including amplitude evolution

---

Grover's search algorithm, published by Lov Grover in 1996, provides a quadratic speedup for searching an unsorted database. While this may sound modest compared to the exponential speedups of Shor's algorithm, Grover's result is remarkable for its *generality*: it applies to any problem that can be formulated as searching for an item that satisfies some condition. Moreover, this quadratic speedup has been proven optimal -- no quantum algorithm can do better for unstructured search.

The algorithm has far-reaching applications beyond simple database search. Any NP problem (satisfiability, graph coloring, optimization) can be cast as a search problem, and Grover's algorithm provides a square-root speedup for all of them. For a problem with $N$ candidates, classical brute force needs $O(N)$ checks, while Grover's algorithm needs only $O(\sqrt{N})$.

> **Analogy:** Grover's algorithm is like a crowd doing the wave -- each iteration amplifies the amplitude of the target, like each wave cycle lifts the target person slightly higher until they stand out from the crowd. The oracle "marks" the target person, and the diffusion operator propagates that marking through the crowd, gradually making the target more and more conspicuous.

## Table of Contents

1. [The Unstructured Search Problem](#1-the-unstructured-search-problem)
2. [The Grover Oracle](#2-the-grover-oracle)
3. [The Diffusion Operator](#3-the-diffusion-operator)
4. [The Algorithm](#4-the-algorithm)
5. [Geometric Interpretation](#5-geometric-interpretation)
6. [Optimal Number of Iterations](#6-optimal-number-of-iterations)
7. [Multiple Solutions](#7-multiple-solutions)
8. [Visualization of Amplitude Evolution](#8-visualization-of-amplitude-evolution)
9. [Exercises](#9-exercises)

---

## 1. The Unstructured Search Problem

### 1.1 Problem Statement

**Given**: A function $f: \{0, 1, \ldots, N-1\} \to \{0, 1\}$ where $N = 2^n$, with the promise that there exists a unique "target" element $w$ such that $f(w) = 1$ and $f(x) = 0$ for all $x \neq w$.

**Goal**: Find $w$ using as few evaluations of $f$ as possible.

Think of this as searching a phone book with $N$ entries for a specific name, where the entries are in random order. You cannot use any structure (like alphabetical ordering) -- you must check entries one by one.

### 1.2 Classical Bound

Classically, the best we can do is check entries one at a time:
- **Expected queries**: $N/2$ (on average, we check half the entries before finding the target)
- **Worst case**: $N$ queries (the target is the last entry we check)
- **Lower bound**: $\Omega(N)$ queries are necessary (provable by an adversary argument)

### 1.3 Quantum Bound

Grover's algorithm finds the target with high probability using:

$$O(\sqrt{N})$$

queries to $f$. Specifically, approximately $\frac{\pi}{4}\sqrt{N}$ queries suffice.

This has been proven optimal: the **BBBV theorem** (Bennett, Bernstein, Brassard, Vazirani) shows that any quantum algorithm for unstructured search requires $\Omega(\sqrt{N})$ queries.

### 1.4 Speedup Summary

| Approach | Queries | For $N = 10^6$ | For $N = 10^{12}$ |
|----------|---------|----------------|-------------------|
| Classical | $O(N)$ | $\sim 500{,}000$ | $\sim 500{,}000{,}000{,}000$ |
| Grover | $O(\sqrt{N})$ | $\sim 785$ | $\sim 785{,}000$ |
| Speedup | $\sqrt{N}$ | $\sim 637\times$ | $\sim 637{,}000\times$ |

```python
import numpy as np

# Classical vs quantum search comparison

print("=== Unstructured Search: Classical vs Quantum ===\n")
print(f"{'N':>15} {'Classical':>15} {'Grover':>15} {'Speedup':>12}")
print("-" * 60)

for n in [5, 10, 15, 20, 25, 30, 40, 50]:
    N = 2**n
    classical = N // 2
    grover = int(np.pi / 4 * np.sqrt(N))
    speedup = classical / grover if grover > 0 else float('inf')

    def format_num(x):
        if x < 1e6:
            return f"{x:,}"
        elif x < 1e9:
            return f"{x/1e6:.1f}M"
        elif x < 1e12:
            return f"{x/1e9:.1f}B"
        else:
            return f"{x:.2e}"

    print(f"2^{n:>2} = {format_num(N):>10}  {format_num(classical):>14} "
          f"{format_num(grover):>14} {speedup:>11.0f}x")

print(f"\nThe quadratic speedup means Grover converts 'infeasible' into 'feasible'")
print(f"for many practical problem sizes.")
```

---

## 2. The Grover Oracle

### 2.1 Phase Oracle

The Grover oracle marks the target element by flipping its phase:

$$O_f|x\rangle = (-1)^{f(x)}|x\rangle = \begin{cases} -|x\rangle & \text{if } x = w \text{ (target)} \\ |x\rangle & \text{otherwise} \end{cases}$$

In matrix form, this is:

$$O_f = I - 2|w\rangle\langle w|$$

The oracle is a reflection: it reflects the state vector about the hyperplane orthogonal to $|w\rangle$.

### 2.2 Oracle Construction

In practice, the oracle is implemented as a quantum circuit that evaluates $f(x)$ and applies a phase kickback (exactly as in the Deutsch-Jozsa algorithm, [Lesson 7](07_Deutsch_Jozsa_Algorithm.md)). The key requirement is that we can evaluate $f$ quantumly (in superposition), even though we do not know which $x$ satisfies $f(x) = 1$.

For example, if we are searching for a solution to a SAT formula, the oracle circuit evaluates the formula on the input $|x\rangle$ and flips the phase if the formula is satisfied.

### 2.3 Oracle Complexity

The oracle counts as a *single query* to $f$, regardless of the internal circuit complexity. This is the standard convention in query complexity. Of course, in practice, the oracle circuit has its own gate cost, which contributes to the total running time.

```python
import numpy as np

def grover_oracle(n, targets):
    """
    Build the Grover oracle that marks target states with a -1 phase.

    O_f = I - 2 * sum_w |w><w|

    Why this form? The oracle is a REFLECTION about the subspace
    orthogonal to the target states. Geometrically, it flips the
    component of the state that lies along the target direction.

    Parameters:
        n: number of qubits
        targets: list of target indices (states to mark)
    """
    dim = 2**n
    O = np.eye(dim, dtype=complex)
    for w in targets:
        O[w, w] = -1  # Flip phase of target state
    return O

# Example: oracle for searching among 8 elements (n=3), target = |101> = 5
n = 3
target = 5

print(f"=== Grover Oracle for n={n}, target=|{format(target, f'0{n}b')}> ===\n")

oracle = grover_oracle(n, [target])

# Apply to uniform superposition
dim = 2**n
uniform = np.ones(dim, dtype=complex) / np.sqrt(dim)

print("Before oracle (uniform superposition):")
for i in range(dim):
    print(f"  |{format(i, f'0{n}b')}>: amplitude = {uniform[i]:.4f}")

after_oracle = oracle @ uniform

print("\nAfter oracle (target flipped):")
for i in range(dim):
    marker = " <-- FLIPPED" if i == target else ""
    print(f"  |{format(i, f'0{n}b')}>: amplitude = {after_oracle[i]:.4f}{marker}")

print(f"\nOnly the target state |{format(target, f'0{n}b')}> had its sign flipped.")
print("All probabilities remain the same! The oracle works through PHASE,")
print("which will be exploited by the diffusion operator.")
```

---

## 3. The Diffusion Operator

### 3.1 Definition

The **diffusion operator** (also called the Grover diffusion or "inversion about the mean") is:

$$D = 2|s\rangle\langle s| - I$$

where $|s\rangle = H^{\otimes n}|0\rangle^{\otimes n} = \frac{1}{\sqrt{N}} \sum_{x=0}^{N-1} |x\rangle$ is the uniform superposition state.

### 3.2 What Does It Do?

The diffusion operator reflects the state vector about the uniform superposition $|s\rangle$. Equivalently, it performs an "inversion about the mean": for each amplitude $\alpha_x$, it maps:

$$\alpha_x \to 2\bar{\alpha} - \alpha_x$$

where $\bar{\alpha} = \frac{1}{N}\sum_x \alpha_x$ is the mean amplitude.

### 3.3 Why Inversion About the Mean?

After the oracle flips the target's amplitude from $+a$ to $-a$, the mean amplitude *decreases* slightly. The diffusion operator then:
1. Computes the new (slightly reduced) mean
2. Reflects each amplitude about this mean

The target (which was flipped to $-a$) gets reflected to a value *above* the mean, while all other amplitudes (at $+a$) get reflected to values *below* the mean. Net effect: the target's amplitude increases, and all others decrease.

### 3.4 Circuit Implementation

The diffusion operator can be implemented as:

$$D = H^{\otimes n} (2|0\rangle\langle 0| - I) H^{\otimes n}$$

The middle operator $2|0\rangle\langle 0| - I$ flips the phase of every state except $|0^n\rangle$. It can be built using a multi-controlled Z gate.

```python
import numpy as np

def diffusion_operator(n):
    """
    Build the Grover diffusion operator D = 2|s><s| - I.

    Why this specific form? The diffusion operator is a REFLECTION about
    the uniform superposition |s>. Combined with the oracle (reflection
    about the target's orthogonal complement), it creates a ROTATION
    toward the target state. Each iteration rotates by a fixed angle,
    making the target's amplitude grow.

    Equivalently, D = H^n (2|0><0| - I) H^n, which shows how to
    implement it as a circuit: H gates, phase flip on |0...0>, H gates.
    """
    dim = 2**n
    # |s> = uniform superposition
    s = np.ones(dim, dtype=complex) / np.sqrt(dim)
    # D = 2|s><s| - I
    D = 2 * np.outer(s, s.conj()) - np.eye(dim, dtype=complex)
    return D

# Demonstrate inversion about the mean
n = 3
dim = 2**n
target = 5

print(f"=== Diffusion Operator: Inversion About the Mean ===\n")
print(f"n={n}, target=|{format(target, f'0{n}b')}>\n")

# Start with uniform superposition, apply oracle
oracle = grover_oracle(n, [target])
D = diffusion_operator(n)

state = np.ones(dim, dtype=complex) / np.sqrt(dim)
print("Step 0 (uniform):")
mean_amp = np.mean(state.real)
print(f"  Mean amplitude: {mean_amp:.4f}")
print(f"  Target amplitude: {state[target]:.4f}")

# Apply oracle
state = oracle @ state
print("\nAfter oracle:")
mean_amp = np.mean(state.real)
print(f"  Mean amplitude: {mean_amp:.4f}")
print(f"  Target amplitude: {state[target]:.4f}")
print(f"  Other amplitudes: {state[0]:.4f}")

# Apply diffusion
state = D @ state
print("\nAfter diffusion (one Grover iteration):")
mean_amp = np.mean(state.real)
print(f"  Mean amplitude: {mean_amp:.4f}")
print(f"  Target amplitude: {state[target]:.4f}")
print(f"  Other amplitudes: {state[0]:.4f}")
print(f"  P(target) = {abs(state[target])**2:.4f}")
print(f"  P(other) = {abs(state[0])**2:.4f} each")

# Show the inversion about the mean explicitly
print("\n--- Inversion About the Mean (Detailed) ---\n")
# After oracle, amplitudes are:
amps = oracle @ (np.ones(dim, dtype=complex) / np.sqrt(dim))
mean = np.mean(amps).real
print(f"After oracle amplitudes: target={amps[target].real:.4f}, "
      f"others={amps[0].real:.4f}")
print(f"Mean = {mean:.4f}")
print(f"Inversion: target -> 2*{mean:.4f} - ({amps[target].real:.4f}) = "
      f"{2*mean - amps[target].real:.4f}")
print(f"Inversion: other  -> 2*{mean:.4f} - ({amps[0].real:.4f}) = "
      f"{2*mean - amps[0].real:.4f}")
```

---

## 4. The Algorithm

### 4.1 Grover's Algorithm

```
1. Initialize: |psi> = H^n |0...0> = (1/sqrt(N)) sum_x |x>
2. Repeat k = floor(pi/4 * sqrt(N)) times:
   a. Apply oracle:    O_f |psi>
   b. Apply diffusion: D |psi>
3. Measure all qubits
```

### 4.2 The Grover Iteration

Each **Grover iteration** (also called a "Grover step") consists of the oracle followed by the diffusion:

$$G = D \cdot O_f$$

The full algorithm applies $G$ exactly $k$ times to the initial uniform superposition:

$$|\psi_k\rangle = G^k |s\rangle$$

### 4.3 Complete Implementation

```python
import numpy as np

def grovers_algorithm(n, targets, verbose=True):
    """
    Complete Grover's search algorithm.

    Why floor(pi/4 * sqrt(N/M)) iterations? This is the number of
    Grover rotations needed to bring the state closest to the target
    subspace. More iterations would OVERSHOOT, reducing the success
    probability. This is a crucial difference from classical search:
    in Grover's, you can search TOO MUCH!

    Parameters:
        n: number of qubits
        targets: list of target indices
        verbose: print step-by-step info

    Returns:
        final state, measurement probabilities, success probability
    """
    dim = 2**n
    M = len(targets)  # Number of solutions

    # Optimal number of iterations
    k_optimal = int(np.round(np.pi / 4 * np.sqrt(dim / M)))

    # Build operators
    oracle = grover_oracle(n, targets)
    D = diffusion_operator(n)
    G = D @ oracle  # Grover iteration

    # Initialize uniform superposition
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)

    if verbose:
        print(f"Grover's Algorithm: n={n}, N={dim}, M={M} solutions")
        print(f"Optimal iterations: k = floor(pi/4 * sqrt({dim}/{M})) = {k_optimal}")
        print(f"Target states: {[format(t, f'0{n}b') for t in targets]}")
        print()

    # Record amplitude evolution for visualization
    target_amps = [abs(state[targets[0]])]
    other_amps = [abs(state[0]) if 0 not in targets else abs(state[1])]

    # Apply Grover iterations
    for i in range(k_optimal):
        state = G @ state
        target_amps.append(abs(state[targets[0]]))
        non_target = 0 if 0 not in targets else 1
        other_amps.append(abs(state[non_target]))

        if verbose and (i < 5 or i == k_optimal - 1):
            p_success = sum(abs(state[t])**2 for t in targets)
            print(f"  Iteration {i+1}: P(target) = {p_success:.6f}")

    # Final probabilities
    probs = np.abs(state)**2
    p_success = sum(probs[t] for t in targets)

    if verbose:
        print(f"\nFinal success probability: {p_success:.6f}")
        print(f"Expected measurement outcome: one of the targets with P = {p_success:.6f}")

    return state, probs, p_success, target_amps, other_amps

# Run Grover's algorithm for various sizes
print("=== Grover's Algorithm Demo ===\n")
for n in [3, 4, 5]:
    target = 2**n - 1  # Search for the last element
    state, probs, p_success, _, _ = grovers_algorithm(n, [target])
    print()
```

Here is a compact version demonstrating the core loop:

```python
import numpy as np

# Core Grover loop (minimal version for clarity)

def grover_search(n, target):
    """
    Minimal Grover's algorithm.

    Why is this so simple? The entire algorithm is just:
    1. Start in uniform superposition
    2. Repeat: oracle + diffusion
    3. Measure

    The mathematical beauty is that these two simple reflections
    compose into a rotation that steadily increases the target's amplitude.
    """
    N = 2**n
    k = int(np.pi / 4 * np.sqrt(N))

    # State vector (start uniform)
    state = np.ones(N, dtype=complex) / np.sqrt(N)

    # Grover iterations
    for _ in range(k):
        # Oracle: flip target phase
        state[target] *= -1

        # Diffusion: inversion about the mean
        mean = np.mean(state)
        state = 2 * mean - state

    return np.argmax(np.abs(state)**2), np.abs(state)**2

# Test
print("=== Minimal Grover Implementation ===\n")
for n in [4, 6, 8, 10]:
    target = 42 % (2**n)  # Some target
    found, probs = grover_search(n, target)
    print(f"n={n:>2}, N={2**n:>5}, target={target:>4} -> "
          f"found={found:>4}, P(correct)={probs[target]:.4f}, "
          f"correct={found == target}")
```

---

## 5. Geometric Interpretation

### 5.1 The Two-Dimensional Subspace

The key insight is that Grover's algorithm operates entirely within a 2-dimensional subspace spanned by:

$$|w\rangle = \text{target state}$$
$$|w^\perp\rangle = \frac{1}{\sqrt{N-1}} \sum_{x \neq w} |x\rangle = \text{uniform superposition of non-targets}$$

The initial state $|s\rangle$ can be written as:

$$|s\rangle = \sin\theta \cdot |w\rangle + \cos\theta \cdot |w^\perp\rangle$$

where $\sin\theta = \frac{1}{\sqrt{N}}$ (the amplitude of the target in the uniform superposition) and thus $\theta = \arcsin(1/\sqrt{N}) \approx 1/\sqrt{N}$ for large $N$.

### 5.2 Grover Iteration as Rotation

Each Grover iteration $G = D \cdot O_f$ rotates the state vector by angle $2\theta$ toward $|w\rangle$ in this 2D plane:

1. **Oracle** ($O_f$): Reflects about $|w^\perp\rangle$ (flips the component along $|w\rangle$)
2. **Diffusion** ($D$): Reflects about $|s\rangle$

The composition of two reflections is a rotation by twice the angle between their reflection axes. The angle between $|s\rangle$ and $|w^\perp\rangle$ is $\theta$, so the rotation angle is $2\theta$.

### 5.3 After $k$ Iterations

After $k$ iterations, the state is:

$$|s_k\rangle = \sin((2k+1)\theta) |w\rangle + \cos((2k+1)\theta) |w^\perp\rangle$$

The success probability is:

$$P(\text{find target}) = \sin^2((2k+1)\theta)$$

Maximum when $(2k+1)\theta \approx \pi/2$, giving $k \approx \frac{\pi}{4\theta} - \frac{1}{2} \approx \frac{\pi}{4}\sqrt{N}$.

```python
import numpy as np

# Geometric interpretation of Grover's algorithm

def grover_geometric(N, k_iterations):
    """
    Track the geometric evolution of Grover's algorithm in the 2D subspace.

    Why 2D? The oracle and diffusion operator only mix the target component
    with the non-target component. They never create amplitude in any other
    direction. This means the ENTIRE algorithm lives in a 2D plane,
    making it easy to analyze geometrically as a rotation.
    """
    # Initial angle
    theta = np.arcsin(1 / np.sqrt(N))

    angles = [(2*0 + 1) * theta]  # Initial angle from |w_perp> axis

    for k in range(1, k_iterations + 1):
        angle = (2*k + 1) * theta
        angles.append(angle)

    # Success probabilities
    success_probs = [np.sin(a)**2 for a in angles]
    # Amplitudes in the {|w>, |w_perp>} basis
    target_amps = [np.sin(a) for a in angles]
    other_amps = [np.cos(a) for a in angles]

    return angles, success_probs, target_amps, other_amps

# Demonstrate the rotation for N=64 (n=6)
N = 64
n = 6
theta = np.arcsin(1 / np.sqrt(N))
k_opt = int(np.round(np.pi / (4 * theta) - 0.5))

print(f"=== Geometric Interpretation: N={N}, n={n} ===\n")
print(f"Initial angle: theta = arcsin(1/sqrt({N})) = {theta:.4f} rad = {theta*180/np.pi:.2f} deg")
print(f"Rotation per iteration: 2*theta = {2*theta:.4f} rad = {2*theta*180/np.pi:.2f} deg")
print(f"Optimal iterations: k = pi/(4*theta) - 1/2 = {np.pi/(4*theta) - 0.5:.2f} -> {k_opt}\n")

# Track evolution
angles, probs, t_amps, o_amps = grover_geometric(N, k_opt + 3)

print(f"{'Iteration':>10} {'Angle (deg)':>15} {'P(target)':>12} "
      f"{'amp(target)':>14} {'amp(other)':>14}")
print("-" * 70)

for k in range(len(angles)):
    marker = " <-- optimal" if k == k_opt else ""
    if k > k_opt:
        marker = " (overshooting!)" if probs[k] < probs[k-1] else ""
    print(f"{k:>10} {angles[k]*180/np.pi:>15.2f} {probs[k]:>12.6f} "
          f"{t_amps[k]:>14.6f} {o_amps[k]:>14.6f}{marker}")

print(f"\nKey observations:")
print(f"  1. Each iteration rotates by {2*theta*180/np.pi:.2f} degrees")
print(f"  2. Maximum P(target) at iteration {k_opt} (angle ~90 deg)")
print(f"  3. OVERSHOOTING past the optimal point DECREASES success probability!")
print(f"  4. This is why Grover's algorithm requires knowing when to stop.")
```

---

## 6. Optimal Number of Iterations

### 6.1 Derivation

We want to maximize $P = \sin^2((2k+1)\theta)$, which is maximized when:

$$(2k+1)\theta = \frac{\pi}{2}$$

$$k = \frac{\pi}{4\theta} - \frac{1}{2} \approx \frac{\pi}{4}\sqrt{N}$$

(since $\theta \approx 1/\sqrt{N}$ for large $N$).

We round $k$ to the nearest integer. The resulting success probability is:

$$P_{\max} = \sin^2\left(\frac{\pi}{2} - O(1/\sqrt{N})\right) = 1 - O(1/N)$$

For large $N$, this is very close to 1.

### 6.2 The Danger of Over-Iterating

Unlike classical search, where more iterations always help, Grover's algorithm can *overshoot*. After the optimal $k$ iterations, additional iterations rotate the state *away* from $|w\rangle$, decreasing the success probability. This is a fundamental feature of quantum algorithms.

In the extreme, after $\frac{\pi}{2\theta}$ iterations (twice the optimal), the state has rotated back to nearly $|w^\perp\rangle$, and the success probability is close to 0 -- worse than random guessing!

### 6.3 When You Don't Know $N$

In some applications, we do not know the number of solutions $M$. There are techniques to handle this:

1. **Exponential search**: Try $k = 1, 2, 4, 8, \ldots$ iterations. The overhead is only a constant factor.
2. **Quantum counting**: Use the quantum Fourier transform (next lesson) to estimate $M$ before running Grover.

```python
import numpy as np

# Demonstrating optimal iterations and overshooting

print("=== Optimal Iterations and Overshooting ===\n")

for n in [3, 5, 8, 10]:
    N = 2**n
    theta = np.arcsin(1 / np.sqrt(N))
    k_exact = np.pi / (4 * theta) - 0.5
    k_opt = int(np.round(k_exact))

    # Success probability at optimal k
    p_opt = np.sin((2*k_opt + 1) * theta)**2

    # Success probability at 2*k_opt (twice optimal)
    k_double = 2 * k_opt
    p_double = np.sin((2*k_double + 1) * theta)**2

    # Success probability at half optimal
    k_half = max(1, k_opt // 2)
    p_half = np.sin((2*k_half + 1) * theta)**2

    print(f"N = 2^{n} = {N:>6}")
    print(f"  k_optimal = {k_opt} (exact: {k_exact:.2f})")
    print(f"  P(k={k_half:>3}) = {p_half:.6f}  (under-iterating)")
    print(f"  P(k={k_opt:>3}) = {p_opt:.6f}  (optimal)")
    print(f"  P(k={k_double:>3}) = {p_double:.6f}  (over-iterating!)")
    print()

# Show the full probability oscillation
print("=== Full Probability Oscillation (N=256, n=8) ===\n")
N = 256
n = 8
theta = np.arcsin(1 / np.sqrt(N))
k_opt = int(np.round(np.pi / (4 * theta) - 0.5))

print(f"{'Iteration':>10} {'P(target)':>12} {'Status':>20}")
print("-" * 45)

for k in range(0, 3*k_opt + 1):
    p = np.sin((2*k + 1) * theta)**2
    if k == k_opt:
        status = "OPTIMAL"
    elif k == 0:
        status = "initial (1/N)"
    elif abs(p - 1/N) < 0.01:
        status = "back to ~1/N"
    elif p < 0.01:
        status = "near ZERO!"
    else:
        status = ""
    print(f"{k:>10} {p:>12.6f} {status:>20}")
```

---

## 7. Multiple Solutions

### 7.1 Generalization

If there are $M$ target elements ($M$ solutions), the algorithm generalizes naturally:

$$\sin\theta_M = \sqrt{\frac{M}{N}}$$

The optimal number of iterations becomes:

$$k_{\text{opt}} = \left\lfloor \frac{\pi}{4} \sqrt{\frac{N}{M}} \right\rfloor$$

The success probability remains close to 1.

### 7.2 Special Cases

| $M$ | $k_{\text{opt}}$ | Success Probability | Notes |
|-----|:---:|:---:|-------|
| 1 | $\sim \frac{\pi}{4}\sqrt{N}$ | $\sim 1$ | Standard case |
| $N/4$ | $\sim 1$ | $\sim 1$ | Few iterations needed |
| $N/2$ | $\sim 1$ | Exactly 1 | One iteration suffices |
| $> N/2$ | 0 | $> 1/2$ | Random guess already works |

### 7.3 Unknown Number of Solutions

When $M$ is unknown, we cannot directly compute $k_{\text{opt}}$. The exponential search strategy works:

1. Pick a random $k \in \{1, 2, \ldots, \lambda\}$ where $\lambda$ starts at 1
2. Run Grover for $k$ iterations and measure
3. If the measurement gives a solution, stop
4. Otherwise, set $\lambda \to \min(2\lambda, \sqrt{N})$ and repeat

This finds a solution in $O(\sqrt{N/M})$ total queries with high probability.

```python
import numpy as np

# Grover's algorithm with multiple solutions

def grovers_multi_solution(n, targets):
    """
    Grover's algorithm with multiple target states.

    Why does more solutions mean fewer iterations? Each solution
    contributes to the 'target subspace'. More solutions means
    the initial state has a larger overlap with this subspace
    (larger theta), so fewer rotations are needed to reach it.
    """
    N = 2**n
    M = len(targets)
    theta = np.arcsin(np.sqrt(M / N))
    k_opt = max(0, int(np.round(np.pi / (4 * theta) - 0.5)))

    # Build operators
    oracle = grover_oracle(n, targets)
    D = diffusion_operator(n)
    G = D @ oracle

    # Initialize
    state = np.ones(N, dtype=complex) / np.sqrt(N)

    # Iterate
    for _ in range(k_opt):
        state = G @ state

    # Success probability
    p_success = sum(abs(state[t])**2 for t in targets)

    return k_opt, p_success

# Test with varying number of solutions
print("=== Multiple Solutions ===\n")
n = 8
N = 2**n
print(f"n={n}, N={N}\n")

print(f"{'M solutions':>12} {'k_opt':>8} {'P(success)':>12} {'Queries saved vs N/2':>22}")
print("-" * 58)

for M in [1, 2, 4, 8, 16, 32, 64, 128]:
    k, p = grovers_multi_solution(n, list(range(M)))
    classical = N // (2 * M)  # Expected classical queries to find 1 of M targets
    print(f"{M:>12} {k:>8} {p:>12.6f} {f'{classical}/{k} = {classical/max(k,1):.1f}x':>22}")

print(f"\nMore solutions -> fewer iterations (larger theta -> fewer rotations needed)")
```

---

## 8. Visualization of Amplitude Evolution

```python
import numpy as np

# Comprehensive visualization of Grover's algorithm amplitude evolution

def grover_visualization(n, target, extra_iterations=3):
    """
    Visualize how amplitudes evolve through Grover iterations.

    Why visualize amplitudes? The core mechanism of Grover's algorithm
    is the gradual amplification of the target's amplitude through
    repeated oracle + diffusion operations. Seeing the amplitudes
    change iteration by iteration builds intuition for:
    1. Why the algorithm works (constructive interference at target)
    2. Why there's an optimal number of iterations (overshooting)
    3. The sinusoidal oscillation pattern
    """
    N = 2**n
    theta = np.arcsin(1 / np.sqrt(N))
    k_opt = int(np.round(np.pi / (4 * theta) - 0.5))
    total_iterations = k_opt + extra_iterations

    # Track evolution
    oracle = grover_oracle(n, [target])
    D = diffusion_operator(n)
    G = D @ oracle

    state = np.ones(N, dtype=complex) / np.sqrt(N)
    history = [state.copy()]

    for k in range(total_iterations):
        state = G @ state
        history.append(state.copy())

    # Print amplitude bars (text-based visualization)
    print(f"=== Grover's Algorithm Amplitude Evolution ===")
    print(f"    n={n}, N={N}, target=|{format(target, f'0{n}b')}>, "
          f"k_opt={k_opt}")
    print()

    bar_width = 50

    for k in range(len(history)):
        state = history[k]
        p_target = abs(state[target])**2
        p_other = abs(state[0 if 0 != target else 1])**2

        # Create bar for target
        target_bar_len = int(p_target * bar_width)
        other_bar_len = int(p_other * bar_width * N)  # Scale up to see non-targets

        marker = " <-- OPTIMAL" if k == k_opt + 1 else ""
        if k == 0:
            label = "Initial"
        else:
            label = f"Iter {k:>2}"

        target_bar = "#" * target_bar_len + "." * (bar_width - target_bar_len)
        print(f"  {label}: P(target)={p_target:.4f} [{target_bar}]{marker}")

    print()

    # Summary statistics
    print("=== Summary ===\n")
    print(f"{'Iteration':>10} {'P(target)':>12} {'P(each other)':>15} {'Amp(target)':>14}")
    print("-" * 55)
    for k, state in enumerate(history):
        p_t = abs(state[target])**2
        non_target = 0 if 0 != target else 1
        p_o = abs(state[non_target])**2
        a_t = state[target].real
        print(f"{k:>10} {p_t:>12.6f} {p_o:>15.8f} {a_t:>14.6f}")

    return history

# Run visualization for different sizes
history = grover_visualization(n=4, target=7, extra_iterations=4)

print("\n\n")
history = grover_visualization(n=6, target=42, extra_iterations=5)

# Verify the sinusoidal pattern
print("\n\n=== Verifying Sinusoidal Pattern ===\n")
n = 8
N = 2**n
target = 100
theta = np.arcsin(1 / np.sqrt(N))

oracle = grover_oracle(n, [target])
D = diffusion_operator(n)
G = D @ oracle
state = np.ones(N, dtype=complex) / np.sqrt(N)

print(f"n={n}, N={N}, theta={theta:.6f}")
print(f"\n{'Iter':>5} {'Simulated P':>14} {'Theoretical P':>16} {'Match':>8}")
print("-" * 48)

for k in range(int(2 * np.pi / (2 * theta))):
    p_sim = abs(state[target])**2
    p_theory = np.sin((2*k + 1) * theta)**2
    match = np.isclose(p_sim, p_theory, atol=1e-6)
    if k <= 15 or k % 5 == 0:
        print(f"{k:>5} {p_sim:>14.8f} {p_theory:>16.8f} {'YES' if match else 'NO':>8}")
    state = G @ state

print("\nThe simulation perfectly matches the theoretical sin^2 formula!")
```

---

## 9. Exercises

### Exercise 1: Basic Grover

Implement Grover's algorithm for $n = 4$ qubits ($N = 16$ elements).

a) Search for target $w = 7$. How many iterations are needed? What is the final success probability?
b) Run the algorithm for 0, 1, 2, ..., 10 iterations and plot the success probability vs iteration count.
c) Verify that the success probability oscillates sinusoidally.

### Exercise 2: Oracle Construction

For a 3-qubit system ($N = 8$), construct Grover oracles for:

a) A single target $w = 5$ (binary: 101)
b) Two targets $w \in \{3, 5\}$
c) A target defined by a condition: $f(x) = 1$ if $x$ has exactly two 1-bits

For each, implement the oracle as a circuit (sequence of gates) rather than as a direct diagonal matrix. (Hint: use multi-controlled Z gates.)

### Exercise 3: Amplitude Amplification

Grover's algorithm is a special case of a more general technique called **amplitude amplification**. In the general version, the initial state need not be uniform:

a) Start with the state $\frac{1}{\sqrt{3}}|0\rangle + \sqrt{\frac{2}{3}}|1\rangle$ (for a 1-qubit system) and apply the diffusion operator about *this* state. What happens?
b) Generalize: for a 2-qubit system, prepare a non-uniform initial state and use it as the "starting superposition" for Grover's algorithm. Does the algorithm still work? How does the initial amplitude of the target affect the number of iterations needed?

### Exercise 4: Application to SAT

A boolean satisfiability (SAT) problem: find $x = (x_2, x_1, x_0) \in \{0,1\}^3$ such that:

$$(x_0 \lor x_1) \land (\lnot x_1 \lor x_2) \land (x_0 \lor \lnot x_2)$$

a) Enumerate all 8 possible inputs and determine which satisfy the formula.
b) Construct a Grover oracle that marks the satisfying assignments.
c) Run Grover's algorithm and verify it finds a satisfying assignment.
d) If there are $M$ satisfying assignments, what is the optimal number of Grover iterations?

### Exercise 5: Grover's Algorithm with Noise

In real quantum computers, gates are imperfect. Simulate the effect of noise:

a) Add random phase errors to each qubit after every Grover iteration: $|x\rangle \to e^{i\epsilon_x}|x\rangle$ where $\epsilon_x$ is drawn from $\mathcal{N}(0, \sigma^2)$.
b) Run the noisy algorithm for $\sigma = 0, 0.01, 0.05, 0.1, 0.5$ and plot how the success probability degrades.
c) At what noise level does the quantum advantage over classical search disappear?

---

[<- Previous: Deutsch-Jozsa Algorithm](07_Deutsch_Jozsa_Algorithm.md) | [Next: Quantum Fourier Transform ->](09_Quantum_Fourier_Transform.md)
