# Lesson 19: Quantum Walks

[← Previous: Quantum Simulation](18_Quantum_Simulation.md) | [Next: Noise and Quantum Channels →](20_Noise_and_Quantum_Channels.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define discrete-time and continuous-time quantum walks and contrast them with classical random walks
2. Construct coined quantum walks on lines and graphs using shift and coin operators
3. Explain the quadratic speedup of quantum walks for spatial search problems
4. Apply quantum walk frameworks to graph isomorphism and element distinctness
5. Analyze the spreading behavior of quantum walks and their ballistic transport
6. Implement discrete and continuous quantum walks in Python

---

Quantum walks are the quantum mechanical generalization of classical random walks. Where a classical random walker moves left or right with equal probability, a quantum walker exists in a superposition of moving in all directions simultaneously. This leads to fundamentally different behavior: while a classical random walk on a line spreads as $\sigma \sim \sqrt{t}$ (diffusive), a quantum walk spreads as $\sigma \sim t$ (ballistic) — a quadratic speedup in spreading.

This speedup is not merely a curiosity. Quantum walks underpin several important quantum algorithms, including Grover's search (which can be recast as a quantum walk), algorithms for element distinctness, and approaches to graph isomorphism. They also provide an alternative model of universal quantum computation.

> **Analogy:** Imagine a classical random walker at a fork in the road choosing left or right by flipping a coin. A quantum walker, instead, travels down both paths simultaneously as a wave. When the paths reconverge, the waves interfere — constructively at some locations (high probability) and destructively at others (low probability). This interference pattern is the source of quantum speedup.

## Table of Contents

1. [Classical Random Walks Review](#1-classical-random-walks-review)
2. [Discrete-Time Quantum Walks](#2-discrete-time-quantum-walks)
3. [Continuous-Time Quantum Walks](#3-continuous-time-quantum-walks)
4. [Coined Quantum Walks](#4-coined-quantum-walks)
5. [Quantum Walk Search](#5-quantum-walk-search)
6. [Applications to Graph Problems](#6-applications-to-graph-problems)
7. [Quantum Speedups from Walks](#7-quantum-speedups-from-walks)
8. [Universality of Quantum Walks](#8-universality-of-quantum-walks)
9. [Python Implementation](#9-python-implementation)
10. [Exercises](#10-exercises)

---

## 1. Classical Random Walks Review

### 1.1 Random Walk on a Line

A classical random walk on the integers: at each step, move left or right with probability $1/2$ each.

**Position after $t$ steps**: $X_t = \sum_{i=1}^{t} s_i$ where $s_i \in \{-1, +1\}$ uniformly.

**Properties**:
- Mean: $\langle X_t \rangle = 0$
- Variance: $\text{Var}(X_t) = t$
- Standard deviation: $\sigma = \sqrt{t}$ (diffusive spreading)
- Distribution: Approximately Gaussian for large $t$ (CLT)

### 1.2 Random Walk on Graphs

A random walk on a graph $G = (V, E)$ with adjacency matrix $A$:

$$P_{ij} = \frac{A_{ij}}{d_j}$$

where $d_j$ is the degree of vertex $j$. The walker moves to a random neighbor at each step.

**Mixing time**: The number of steps until the distribution is close to the stationary distribution $\pi_i = d_i / (2|E|)$. For an $n$-vertex graph, mixing time is typically $O(n \log n)$ to $O(n^3)$.

**Hitting time**: The expected number of steps to reach a target vertex from a source. For a line of length $n$: hitting time is $O(n^2)$.

### 1.3 Applications of Classical Random Walks

| Application | Complexity |
|-------------|-----------|
| Graph connectivity | $O(n^3)$ |
| 2-SAT | $O(n^2)$ randomized |
| Undirected s-t connectivity | $O(n^2)$ |
| Estimating volume of convex body | Polynomial (via Markov chains) |
| PageRank | Mixing time of the web graph |

---

## 2. Discrete-Time Quantum Walks

### 2.1 Definition

A discrete-time quantum walk (DTQW) on a line requires two registers:

- **Position register**: $|x\rangle$ where $x \in \mathbb{Z}$ (or a finite subset)
- **Coin register**: $|c\rangle$ where $c \in \{0, 1\}$ (for a walk on a line)

The Hilbert space is $\mathcal{H} = \mathcal{H}_{\text{coin}} \otimes \mathcal{H}_{\text{position}}$.

### 2.2 Walk Operators

Each step consists of two operations:

**Coin operator** $C$: Acts on the coin register, creating superposition:

$$C = H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

(The Hadamard coin is the most common choice, but other coins give different behavior.)

**Shift operator** $S$: Moves the walker based on the coin state:

$$S|0\rangle|x\rangle = |0\rangle|x-1\rangle, \quad S|1\rangle|x\rangle = |1\rangle|x+1\rangle$$

In matrix form: $S = |0\rangle\langle 0| \otimes \sum_x |x-1\rangle\langle x| + |1\rangle\langle 1| \otimes \sum_x |x+1\rangle\langle x|$

**One step**: $U = S \cdot (C \otimes I_{\text{position}})$

### 2.3 Ballistic Spreading

Starting from $|0\rangle|0\rangle$ (coin in $|0\rangle$, position at origin), after $t$ steps:

$$|\psi(t)\rangle = U^t |0\rangle|0\rangle$$

The probability distribution $P(x, t) = \sum_{c} |\langle c, x|\psi(t)\rangle|^2$ shows:

- **Not Gaussian**: Unlike classical walk, the distribution is bimodal with peaks near $x = \pm t/\sqrt{2}$
- **Ballistic spreading**: $\sigma \sim t$ (linear in time, not $\sqrt{t}$)
- **Asymmetry**: The Hadamard coin produces a left-biased distribution for initial coin state $|0\rangle$

### 2.4 Coin Dependence

The choice of coin operator dramatically affects the walk:

| Coin | Distribution shape | Bias |
|------|-------------------|------|
| Hadamard $H$ | Asymmetric, biased left for $\|0\rangle$ | Left |
| $Y$ gate | Asymmetric, biased right | Right |
| Balanced: $\frac{1}{\sqrt{2}}\begin{pmatrix}1 & i\\i & 1\end{pmatrix}$ | Symmetric | None |
| Grover coin (higher dim) | Uniform spreading | None |

---

## 3. Continuous-Time Quantum Walks

### 3.1 Definition

A continuous-time quantum walk (CTQW) is defined by the Schrödinger equation on a graph:

$$i\frac{d}{dt}|\psi(t)\rangle = H_{\text{walk}}|\psi(t)\rangle$$

where $H_{\text{walk}}$ is derived from the graph structure. Common choices:

- **Adjacency matrix**: $H_{\text{walk}} = \gamma A$ where $A$ is the adjacency matrix and $\gamma$ is the hopping rate
- **Laplacian**: $H_{\text{walk}} = \gamma L$ where $L = D - A$ (degree matrix minus adjacency)

The time evolution is:

$$|\psi(t)\rangle = e^{-iH_{\text{walk}}t}|\psi(0)\rangle$$

### 3.2 No Coin Needed

Unlike discrete-time quantum walks, CTQW does not require a coin register. The graph structure itself determines the dynamics. This is often more natural for physical systems.

### 3.3 CTQW on a Line

For an $n$-site line, $A$ is the tridiagonal matrix:

$$A_{ij} = \begin{cases} 1 & \text{if } |i-j| = 1 \\ 0 & \text{otherwise} \end{cases}$$

Starting from position $|0\rangle$:

$$\langle x|\psi(t)\rangle = i^{-x} J_x(2\gamma t)$$

where $J_x$ is the Bessel function of the first kind. This gives ballistic spreading with $\sigma \sim \gamma t$.

### 3.4 CTQW on Specific Graphs

| Graph | Spreading behavior | Notable property |
|-------|-------------------|-----------------|
| Line | Ballistic ($\sigma \sim t$) | Bessel function profile |
| Cycle ($n$ vertices) | Periodic revival | Period $\sim n$ |
| Complete graph ($K_n$) | Instantaneous mixing | Perfect state transfer possible |
| Hypercube ($\{0,1\}^n$) | Fast mixing | Exponential speedup for hitting |
| Star graph | Periodic | Perfect state transfer |

---

## 4. Coined Quantum Walks

### 4.1 Generalization to Graphs

For a quantum walk on a $d$-regular graph $G$ (every vertex has degree $d$), the coined walk uses:

- **Coin space**: $\mathbb{C}^d$ at each vertex (one basis state per edge)
- **Position space**: $\mathbb{C}^{|V|}$
- **Total Hilbert space**: $\mathbb{C}^{d|V|}$

### 4.2 The Grover Coin

For $d$-regular graphs, the Grover diffusion operator is a natural coin:

$$G_d = \frac{2}{d}\mathbf{1}\mathbf{1}^T - I_d$$

where $\mathbf{1} = (1, 1, \ldots, 1)^T / \sqrt{d}$. This coin treats all directions equally and is the quantum analog of the uniform random walk.

### 4.3 Shift Operator on Graphs

The shift operator on a graph moves the walker along the edge indicated by the coin state:

$$S|c, v\rangle = |c', u\rangle$$

where $u$ is the neighbor of $v$ along edge $c$, and $c'$ is the coin state corresponding to the reverse edge (from $u$ back to $v$).

### 4.4 Szegedy Quantum Walk

An alternative formulation by Szegedy directly quantizes any classical Markov chain $P$:

1. Define $|p_x\rangle = \sum_y \sqrt{P_{xy}} |y\rangle$ for each vertex $x$
2. Construct $|\psi_x\rangle = |x\rangle|p_x\rangle$ in a doubled Hilbert space
3. Define reflections $R_A$ and $R_B$ around the spans of $\{|\psi_x\rangle\}$
4. The walk operator is $W = R_B R_A$

**Key property**: The eigenvalue gap of $W$ is related to the spectral gap $\delta$ of $P$ by:

$$\Delta_W = \Theta(\sqrt{\delta})$$

This quadratic improvement in the spectral gap leads to quadratic speedups for Markov chain problems.

---

## 5. Quantum Walk Search

### 5.1 Grover's Algorithm as a Quantum Walk

Grover's search can be recast as a quantum walk on the complete graph $K_N$:

- Walk operator: $W = S \cdot (C_{\text{marked}} \otimes I)$ where $C_{\text{marked}}$ flips the phase at marked vertices
- The walk amplifies the amplitude at marked vertices in $O(\sqrt{N})$ steps

### 5.2 Spatial Search

**Problem**: Given a graph $G$ with $N$ vertices and $M$ marked vertices, find a marked vertex.

**Classical random walk**: Hitting time $T_{\text{classical}}$

**Quantum walk search**: Hitting time $T_{\text{quantum}} = O(\sqrt{T_{\text{classical}} / M})$

| Graph | Classical hitting | Quantum hitting |
|-------|------------------|-----------------|
| Complete $K_N$ | $O(N)$ | $O(\sqrt{N})$ |
| 2D grid $\sqrt{N} \times \sqrt{N}$ | $O(N \log N)$ | $O(\sqrt{N} \log N)$ |
| Hypercube $\{0,1\}^n$ | $O(2^n)$ | $O(\sqrt{2^n})$ |

### 5.3 The AKR Algorithm

Ambainis, Kempe, and Rivest showed that quantum walks on 2D grids can find a marked vertex in $O(\sqrt{N}\log N)$ steps, nearly matching Grover's $O(\sqrt{N})$ lower bound.

**Algorithm**:
1. Start in the uniform superposition over all vertices
2. Apply $O(\sqrt{N}\log N)$ steps of the quantum walk with a modified coin at the marked vertex
3. Measure to find the marked vertex

The modification: at unmarked vertices, use the Grover coin; at the marked vertex, use $-I$ (phase flip).

### 5.4 Limitations

- Not all graphs give optimal speedup. The spatial structure matters.
- For 1D chains, quantum walks give no speedup over classical for search (both $O(N)$).
- The overhead of implementing the walk operator on a quantum computer can be significant.

---

## 6. Applications to Graph Problems

### 6.1 Element Distinctness

**Problem**: Given $N$ elements, determine if any two are equal.

**Classical**: $O(N)$ (with hashing) or $O(N \log N)$ (comparison-based)
**Quantum**: $O(N^{2/3})$ using Ambainis' quantum walk algorithm

**Algorithm sketch**:
1. Maintain a quantum walk on the Johnson graph $J(N, r)$ where vertices are subsets of size $r$
2. Each vertex stores the values of its $r$ elements in a quantum database
3. A "marked" vertex is one whose subset contains a collision
4. Quantum walk search finds a marked vertex in $O(N^{2/3})$ queries

### 6.2 Triangle Finding

**Problem**: Given a graph on $n$ vertices, determine if it contains a triangle.

**Classical**: $O(n^2)$ using adjacency matrix, $O(n^\omega) \approx O(n^{2.37})$ using matrix multiplication
**Quantum**: $O(n^{5/4})$ using quantum walk (Le Gall, 2014)

### 6.3 Graph Isomorphism

**Problem**: Determine if two graphs $G_1$ and $G_2$ are isomorphic.

Quantum walks provide a heuristic approach: the probability distributions generated by quantum walks on $G_1$ and $G_2$ differ when the graphs are non-isomorphic. Specifically:

- CTQW: Compare $|\langle j|e^{-iAt}|k\rangle|^2$ for various pairs $(j,k)$ and times $t$
- Coined walks: Compare the mixing behavior

**Limitations**: This is not a complete invariant — non-isomorphic graphs can produce identical quantum walk distributions (e.g., strongly regular graphs with the same parameters).

### 6.4 Summary of Quantum Walk Speedups

| Problem | Classical | Quantum (walk-based) | Speedup |
|---------|-----------|---------------------|---------|
| Search (unstructured) | $O(N)$ | $O(\sqrt{N})$ | Quadratic |
| Element distinctness | $O(N)$ | $O(N^{2/3})$ | Polynomial |
| Triangle finding | $O(n^2)$ | $O(n^{5/4})$ | Polynomial |
| Matrix product verification | $O(n^2)$ | $O(n^{5/3})$ | Polynomial |
| Group commutativity testing | $O(n)$ | $O(n^{2/3})$ | Polynomial |

---

## 7. Quantum Speedups from Walks

### 7.1 Why Quantum Walks Are Fast

The speedup of quantum walks comes from two quantum phenomena:

**Interference**: The quantum walker explores multiple paths simultaneously. Paths that lead to the target interfere constructively, while paths that lead away interfere destructively. This focuses the probability amplitude on the target.

**Ballistic transport**: While a classical random walk spreads diffusively ($\sigma \sim \sqrt{t}$), a quantum walk spreads ballistically ($\sigma \sim t$). This means the quantum walker covers a distance $d$ in time $O(d)$ rather than $O(d^2)$.

### 7.2 Quantum Walk as a Computational Model

Quantum walks are **universal for quantum computation**: any quantum circuit can be simulated by a quantum walk on an appropriately constructed graph. This was shown by Childs (2009) for continuous-time walks and by Lovett et al. (2010) for discrete-time walks.

### 7.3 Connection to Other Quantum Algorithms

| Algorithm | Quantum walk formulation |
|-----------|------------------------|
| Grover search | Walk on complete graph |
| Phase estimation | Walk on the line (in phase space) |
| QAOA | Alternating walk steps on problem/mixer graphs |
| Adiabatic computation | Continuous walk with time-dependent Hamiltonian |

---

## 8. Universality of Quantum Walks

### 8.1 Continuous-Time Walks

Childs proved that CTQW on a sparse graph can simulate any quantum computation. The key idea is to encode the quantum circuit as a graph where:

- Each computational basis state corresponds to a vertex
- Gates are encoded as edges with appropriate weights
- Time evolution implements the quantum circuit

### 8.2 Discrete-Time Walks

For discrete-time coined walks, universality requires:
- A sufficiently complex graph structure
- The ability to choose different coins at different vertices
- Appropriate initial state preparation

### 8.3 Implications

The universality of quantum walks means they provide:
- An alternative model of quantum computation (equivalent to the circuit model)
- A natural framework for designing quantum algorithms on graphs
- A bridge between physics (quantum dynamics on graphs) and computer science (algorithmic complexity)

---

## 9. Python Implementation

### 9.1 Discrete-Time Quantum Walk on a Line

```python
import numpy as np

def discrete_quantum_walk_line(n_steps, n_positions=None, coin='hadamard',
                                initial_coin=None):
    """Simulate a discrete-time quantum walk on an integer line.

    The walker has a coin (internal) degree of freedom and a position
    (external) degree of freedom. At each step:
    1. Apply the coin operator to the coin register
    2. Shift position conditioned on coin state

    This produces ballistic spreading (sigma ~ t) instead of the
    diffusive spreading (sigma ~ sqrt(t)) of classical random walks.

    Args:
        n_steps: Number of walk steps
        n_positions: Number of position sites (default: 2*n_steps + 1)
        coin: Coin type ('hadamard', 'balanced', 'grover')
        initial_coin: Initial coin state (default: |0>)

    Returns:
        positions: Array of position indices
        prob_dist: Probability distribution over positions
        state: Full quantum state (coin x position)
    """
    if n_positions is None:
        n_positions = 2 * n_steps + 1

    center = n_positions // 2

    # Coin operators
    coins = {
        'hadamard': np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
        'balanced': np.array([[1, 1j], [1j, 1]], dtype=complex) / np.sqrt(2),
        'Y': np.array([[1, -1j], [1j, 1]], dtype=complex) / np.sqrt(2),
    }
    C = coins.get(coin, coins['hadamard'])

    # State: |coin> x |position>  (2 * n_positions dimensional)
    state = np.zeros(2 * n_positions, dtype=complex)

    # Initial state: coin in |0> (or custom), position at center
    if initial_coin is None:
        initial_coin = np.array([1, 0], dtype=complex)

    state[0 * n_positions + center] = initial_coin[0]  # |0>|center>
    state[1 * n_positions + center] = initial_coin[1]  # |1>|center>

    # Shift operator
    def shift(state):
        new_state = np.zeros_like(state)
        # Coin=0 -> move left
        for x in range(1, n_positions):
            new_state[0 * n_positions + x - 1] = state[0 * n_positions + x]
        # Coin=1 -> move right
        for x in range(n_positions - 1):
            new_state[1 * n_positions + x + 1] = state[1 * n_positions + x]
        return new_state

    # Coin operator (applied to coin register at each position)
    def apply_coin(state, C):
        new_state = np.zeros_like(state)
        for x in range(n_positions):
            c0 = state[0 * n_positions + x]
            c1 = state[1 * n_positions + x]
            new_state[0 * n_positions + x] = C[0, 0] * c0 + C[0, 1] * c1
            new_state[1 * n_positions + x] = C[1, 0] * c0 + C[1, 1] * c1
        return new_state

    # Walk
    for step in range(n_steps):
        state = apply_coin(state, C)
        state = shift(state)

    # Compute probability distribution (trace out coin)
    prob_dist = np.zeros(n_positions)
    for x in range(n_positions):
        prob_dist[x] = (abs(state[0 * n_positions + x]) ** 2 +
                        abs(state[1 * n_positions + x]) ** 2)

    positions = np.arange(n_positions) - center

    return positions, prob_dist, state


def classical_random_walk_distribution(n_steps, n_positions=None):
    """Compute the exact probability distribution of a classical random walk.

    Uses the binomial distribution: after t steps, position x has
    probability C(t, (t+x)/2) / 2^t (when t+x is even).
    """
    if n_positions is None:
        n_positions = 2 * n_steps + 1

    center = n_positions // 2
    positions = np.arange(n_positions) - center
    prob_dist = np.zeros(n_positions)

    for idx, x in enumerate(positions):
        if (n_steps + x) % 2 != 0:
            continue
        k = (n_steps + x) // 2
        if 0 <= k <= n_steps:
            from math import comb
            prob_dist[idx] = comb(n_steps, k) / 2 ** n_steps

    return positions, prob_dist


# Demonstrate quantum vs classical walk
print("=" * 65)
print("Discrete-Time Quantum Walk vs Classical Random Walk")
print("=" * 65)

n_steps = 50
positions_q, prob_q, _ = discrete_quantum_walk_line(n_steps, coin='hadamard')
positions_c, prob_c = classical_random_walk_distribution(n_steps)

# Statistics
mean_q = np.sum(positions_q * prob_q)
std_q = np.sqrt(np.sum(positions_q ** 2 * prob_q) - mean_q ** 2)
mean_c = np.sum(positions_c * prob_c)
std_c = np.sqrt(np.sum(positions_c ** 2 * prob_c) - mean_c ** 2)

print(f"\nAfter {n_steps} steps:")
print(f"  Classical: mean = {mean_c:.2f}, std = {std_c:.2f}")
print(f"  Quantum:   mean = {mean_q:.2f}, std = {std_q:.2f}")
print(f"  Speedup in spreading: {std_q / std_c:.2f}x")

# Compare spreading rates for different step counts
print(f"\n{'Steps':>8} {'Classical std':>15} {'Quantum std':>15} {'Ratio':>10}")
print("-" * 52)
for t in [10, 20, 50, 100, 200]:
    _, pq, _ = discrete_quantum_walk_line(t)
    _, pc = classical_random_walk_distribution(t)
    pos = np.arange(len(pq)) - len(pq) // 2

    sq = np.sqrt(np.sum(pos ** 2 * pq))
    sc = np.sqrt(np.sum(pos ** 2 * pc))
    print(f"{t:8d} {sc:15.2f} {sq:15.2f} {sq / sc:10.2f}")
```

### 9.2 Coin Dependence

```python
import numpy as np

def compare_coins(n_steps=50):
    """Compare quantum walks with different coin operators.

    The coin operator determines the walk's symmetry and spreading pattern.
    The Hadamard coin produces an asymmetric distribution,
    while the balanced coin produces a symmetric one.
    """
    print("=" * 65)
    print("Effect of Coin Choice on Quantum Walk")
    print("=" * 65)

    coins = {
        'Hadamard |0>': ('hadamard', np.array([1, 0], dtype=complex)),
        'Hadamard |1>': ('hadamard', np.array([0, 1], dtype=complex)),
        'Hadamard |+>': ('hadamard', np.array([1, 1], dtype=complex) / np.sqrt(2)),
        'Balanced |0>': ('balanced', np.array([1, 0], dtype=complex)),
    }

    print(f"\nStep count: {n_steps}")
    print(f"\n{'Coin + Initial':>20} {'Mean':>10} {'Std':>10} {'Max P':>10} {'Peak pos':>10}")
    print("-" * 64)

    for name, (coin, init) in coins.items():
        pos, prob, _ = discrete_quantum_walk_line(n_steps, coin=coin,
                                                    initial_coin=init)
        mean = np.sum(pos * prob)
        std = np.sqrt(np.sum(pos ** 2 * prob) - mean ** 2)
        max_p = np.max(prob)
        peak_pos = pos[np.argmax(prob)]

        print(f"{name:>20} {mean:10.2f} {std:10.2f} {max_p:10.4f} {peak_pos:10d}")

compare_coins()
```

### 9.3 Continuous-Time Quantum Walk

```python
import numpy as np
from scipy.linalg import expm

def continuous_quantum_walk(adjacency, gamma, t, initial_vertex):
    """Simulate a continuous-time quantum walk on a graph.

    The CTQW evolves under H = gamma * A, where A is the adjacency matrix.
    No coin register is needed - the graph structure alone determines
    the dynamics. This is more natural for physical implementations.

    Args:
        adjacency: Adjacency matrix of the graph
        gamma: Hopping rate
        t: Evolution time
        initial_vertex: Starting vertex index

    Returns:
        prob_dist: Probability distribution over vertices
        state: Quantum state vector
    """
    N = adjacency.shape[0]
    H = gamma * adjacency

    # Initial state
    psi0 = np.zeros(N, dtype=complex)
    psi0[initial_vertex] = 1.0

    # Time evolution
    psi_t = expm(-1j * H * t) @ psi0
    prob_dist = np.abs(psi_t) ** 2

    return prob_dist, psi_t


def build_graph(graph_type, n):
    """Build adjacency matrix for common graph types."""
    if graph_type == 'line':
        A = np.zeros((n, n))
        for i in range(n - 1):
            A[i, i + 1] = 1
            A[i + 1, i] = 1
        return A

    elif graph_type == 'cycle':
        A = np.zeros((n, n))
        for i in range(n):
            A[i, (i + 1) % n] = 1
            A[(i + 1) % n, i] = 1
        return A

    elif graph_type == 'complete':
        A = np.ones((n, n)) - np.eye(n)
        return A

    elif graph_type == 'star':
        A = np.zeros((n, n))
        for i in range(1, n):
            A[0, i] = 1
            A[i, 0] = 1
        return A

    elif graph_type == 'hypercube':
        # n-dimensional hypercube has 2^n vertices
        N = 2 ** n
        A = np.zeros((N, N))
        for i in range(N):
            for bit in range(n):
                j = i ^ (1 << bit)
                A[i, j] = 1
        return A

    else:
        raise ValueError(f"Unknown graph type: {graph_type}")


# Demonstrate CTQW on different graphs
print("=" * 65)
print("Continuous-Time Quantum Walk on Different Graphs")
print("=" * 65)

gamma = 1.0
graphs = [
    ('line', 15, 0),
    ('cycle', 12, 0),
    ('complete', 8, 0),
    ('star', 8, 0),
]

for graph_type, n, start in graphs:
    A = build_graph(graph_type, n)
    print(f"\n--- {graph_type.capitalize()} graph (n={n}), start at vertex {start} ---")
    print(f"{'Time':>6}", end="")
    for v in range(min(n, 8)):
        print(f"  P(v={v:d})", end="")
    print()

    for t in [0.0, 0.5, 1.0, 2.0, 5.0]:
        prob, _ = continuous_quantum_walk(A, gamma, t, start)
        print(f"{t:6.1f}", end="")
        for v in range(min(n, 8)):
            print(f"  {prob[v]:7.4f}", end="")
        print()


# CTQW on hypercube
print(f"\n--- Hypercube (dimension 3, 8 vertices), start at vertex 0 ---")
A = build_graph('hypercube', 3)
print(f"{'Time':>6}", end="")
for v in range(8):
    print(f"  P(v={v:d})", end="")
print()

for t in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
    prob, _ = continuous_quantum_walk(A, gamma, t, 0)
    print(f"{t:6.1f}", end="")
    for v in range(8):
        print(f"  {prob[v]:7.4f}", end="")
    print()
```

### 9.4 Quantum Walk Search on a Graph

```python
import numpy as np
from scipy.linalg import expm

def quantum_walk_search(adjacency, marked_vertices, gamma, t_max, n_time_steps=200):
    """Simulate quantum walk search on a graph.

    Modify the CTQW Hamiltonian to include an oracle term that
    'attracts' the walker to marked vertices:
    H = gamma * A - sum_{m in marked} |m><m|

    The walker's amplitude concentrates on marked vertices over time.

    Args:
        adjacency: Graph adjacency matrix
        marked_vertices: List of target vertex indices
        gamma: Hopping rate (must be tuned for optimal search)
        t_max: Maximum evolution time
        n_time_steps: Number of time points to sample

    Returns:
        times: Array of time points
        success_probs: Probability of measuring a marked vertex at each time
    """
    N = adjacency.shape[0]

    # Hamiltonian: gamma * A - oracle
    H = gamma * adjacency.astype(complex)
    for m in marked_vertices:
        H[m, m] -= 1.0

    # Initial state: uniform superposition
    psi0 = np.ones(N, dtype=complex) / np.sqrt(N)

    times = np.linspace(0, t_max, n_time_steps)
    success_probs = np.zeros(n_time_steps)

    for i, t in enumerate(times):
        psi_t = expm(-1j * H * t) @ psi0
        prob = np.abs(psi_t) ** 2
        success_probs[i] = sum(prob[m] for m in marked_vertices)

    return times, success_probs


# Demonstrate search on complete graph
print("=" * 65)
print("Quantum Walk Search")
print("=" * 65)

for graph_type, n, optimal_gamma in [('complete', 64, 1/64), ('complete', 16, 1/16)]:
    if graph_type == 'complete':
        A = np.ones((n, n)) - np.eye(n)

    marked = [0]  # Mark vertex 0
    times, probs = quantum_walk_search(A, marked, optimal_gamma, t_max=10 * np.sqrt(n))

    peak_time = times[np.argmax(probs)]
    peak_prob = np.max(probs)

    print(f"\n{graph_type.capitalize()} graph K_{n}, marked vertex: {marked}")
    print(f"  Optimal gamma: {optimal_gamma:.4f} (= 1/N)")
    print(f"  Peak success probability: {peak_prob:.4f} at t = {peak_time:.2f}")
    print(f"  Expected O(sqrt(N)) time: {np.sqrt(n):.2f}")
    print(f"  Actual peak time / sqrt(N): {peak_time / np.sqrt(n):.2f}")
```

### 9.5 Graph Isomorphism via Quantum Walk

```python
import numpy as np
from scipy.linalg import expm

def quantum_walk_fingerprint(adjacency, gamma=1.0, n_times=20, t_max=10.0):
    """Compute a quantum walk fingerprint for a graph.

    The fingerprint consists of the probability distributions at
    multiple time points, starting from each vertex. Non-isomorphic
    graphs (usually) produce different fingerprints.

    Note: This is a heuristic - some non-isomorphic graphs produce
    identical fingerprints (e.g., certain strongly regular graphs).

    Args:
        adjacency: Adjacency matrix
        gamma: Hopping rate
        n_times: Number of time points
        t_max: Maximum time

    Returns:
        fingerprint: Sorted tuple of probability values (invariant under relabeling)
    """
    N = adjacency.shape[0]
    H = gamma * adjacency.astype(complex)
    times = np.linspace(0.1, t_max, n_times)

    # Collect return probabilities from each vertex
    all_probs = []
    for start in range(N):
        psi0 = np.zeros(N, dtype=complex)
        psi0[start] = 1.0

        for t in times:
            psi_t = expm(-1j * H * t) @ psi0
            prob = np.abs(psi_t) ** 2
            # Return probability (probability of being at start vertex)
            all_probs.append(round(prob[start], 8))

    # Sort to make invariant under vertex relabeling
    return tuple(sorted(all_probs))


# Test graph isomorphism detection
print("=" * 65)
print("Graph Isomorphism Detection via Quantum Walks")
print("=" * 65)

# Create pairs of graphs
# Pair 1: Isomorphic graphs (same graph, permuted vertices)
A1 = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 0, 1],
    [0, 1, 0, 0, 1],
    [0, 0, 1, 1, 0],
], dtype=float)

# Permute vertices: 0->2, 1->0, 2->1, 3->4, 4->3
P = np.array([
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
], dtype=float)
A2 = P @ A1 @ P.T  # Same graph, different labeling

# Pair 2: Non-isomorphic graphs (same degree sequence)
A3 = np.array([
    [0, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 1],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 0],
], dtype=float)

A4 = np.array([
    [0, 1, 0, 1, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 0],
], dtype=float)

fp1 = quantum_walk_fingerprint(A1)
fp2 = quantum_walk_fingerprint(A2)
fp3 = quantum_walk_fingerprint(A3)
fp4 = quantum_walk_fingerprint(A4)

print("\nPair 1: Isomorphic graphs (permuted vertices)")
print(f"  Fingerprints match: {fp1 == fp2}")
print(f"  Max difference: {max(abs(a - b) for a, b in zip(fp1, fp2)):.2e}")

print("\nPair 2: Non-isomorphic graphs (same degree sequence)")
print(f"  Fingerprints match: {fp3 == fp4}")
if fp3 != fp4:
    diffs = [abs(a - b) for a, b in zip(sorted(list(fp3) + [0] * (len(fp4) - len(fp3))),
                                          sorted(list(fp4) + [0] * (len(fp3) - len(fp4))))]
    print(f"  Max difference: {max(diffs):.4f}")
else:
    print("  (Quantum walk cannot distinguish these graphs)")
```

---

## 10. Exercises

### Exercise 1: Quantum Walk Spreading

For the discrete-time quantum walk on a line with the Hadamard coin:
(a) Simulate walks of $t = 10, 20, 50, 100, 200$ steps and compute the standard deviation $\sigma(t)$.
(b) Fit $\sigma(t) = a \cdot t^b$ and verify $b \approx 1$ (ballistic).
(c) Repeat with the balanced coin. How does the spreading rate change?
(d) What happens if you measure the coin register after each step? (Hint: it should become a classical random walk.)

### Exercise 2: CTQW Perfect State Transfer

For a CTQW on a path of $n$ vertices:
(a) Starting at vertex 0, find the time $t^*$ at which the probability of being at vertex $n-1$ is maximized.
(b) For $n = 2, 3, 4, 5, 6$, does perfect state transfer occur (probability 1 at the opposite end)?
(c) Try a weighted path where edge weights are $w_i = \sqrt{i(n-i)}$. Does this improve transfer fidelity?

### Exercise 3: Szegedy Walk Spectrum

Implement a Szegedy quantum walk for a classical random walk on the cycle $C_n$:
(a) Compute the spectrum of the walk operator $W = R_B R_A$ for $n = 8, 12, 16$.
(b) Compare the spectral gap of $W$ with the spectral gap $\delta$ of the classical transition matrix.
(c) Verify the quadratic relationship: $\Delta_W = \Theta(\sqrt{\delta})$.

### Exercise 4: Quantum Walk Search Optimization

For quantum walk search on the 2D grid ($\sqrt{N} \times \sqrt{N}$):
(a) Implement the CTQW search Hamiltonian $H = \gamma L - |m\rangle\langle m|$ where $L$ is the graph Laplacian.
(b) Find the optimal $\gamma$ by scanning over values and measuring peak success probability.
(c) Plot success probability vs. time for grid sizes $N = 16, 36, 64$.
(d) Verify the $O(\sqrt{N} \log N)$ scaling of the optimal time.

### Exercise 5: Quantum Walk Graph Classification

Use quantum walk fingerprints to classify small graphs:
(a) Generate all non-isomorphic connected graphs on 5 vertices (there are 21).
(b) Compute the CTQW fingerprint for each graph.
(c) How many pairs of non-isomorphic graphs can the fingerprint distinguish?
(d) For any pairs that cannot be distinguished, investigate what structural property they share.

---

[← Previous: Quantum Simulation](18_Quantum_Simulation.md) | [Next: Noise and Quantum Channels →](20_Noise_and_Quantum_Channels.md)
