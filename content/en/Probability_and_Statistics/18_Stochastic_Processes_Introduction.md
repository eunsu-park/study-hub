# Stochastic Processes: Introduction

**Previous**: [Regression and ANOVA](./17_Regression_and_ANOVA.md) | **Next**: (End of current series)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define stochastic processes and classify them by time and state space
2. Formulate Markov chains and compute transition probabilities
3. Apply the Chapman-Kolmogorov equations for multi-step transitions
4. Classify states as transient, recurrent, or absorbing
5. Compute stationary distributions for ergodic Markov chains
6. Describe the Poisson process and its fundamental properties
7. Analyze random walks (simple and with drift)
8. Solve the gambler's ruin problem
9. Distinguish strict-sense and wide-sense stationarity
10. Compute and interpret the autocorrelation function

---

A stochastic process is a collection of random variables indexed by time (or space). It provides the mathematical framework for modeling systems that evolve randomly, with applications in physics, finance, biology, computer science, and engineering.

---

## 1. Definition and Classification

### 1.1 Formal Definition

A **stochastic process** is a collection of random variables $\{X(t) : t \in T\}$ defined on a common probability space, where $T$ is the **index set** (typically representing time).

### 1.2 Classification

| | Discrete State | Continuous State |
|---|---|---|
| **Discrete Time** | Markov chain, random walk | AR process, Gaussian sequence |
| **Continuous Time** | Birth-death process | Brownian motion, Poisson process |

- **Discrete time**: $T = \{0, 1, 2, \ldots\}$, written $X_0, X_1, X_2, \ldots$
- **Continuous time**: $T = [0, \infty)$, written $X(t)$
- **State space** $S$: the set of possible values of $X(t)$

### 1.3 Key Properties

- **Sample path (realization)**: A specific trajectory $\{X(t, \omega) : t \in T\}$ for a fixed outcome $\omega$.
- **Finite-dimensional distributions**: The joint distribution of $(X(t_1), X(t_2), \ldots, X(t_n))$ for any finite collection of times characterizes the process (Kolmogorov consistency theorem).

---

## 2. Markov Chains

### 2.1 The Markov Property

A discrete-time stochastic process $\{X_n\}$ is a **Markov chain** if:

$$P(X_{n+1} = j \mid X_n = i, X_{n-1} = i_{n-1}, \ldots, X_0 = i_0) = P(X_{n+1} = j \mid X_n = i)$$

The future depends on the present only, not on the past. This is the **memoryless** property.

### 2.2 Transition Matrix

For a **time-homogeneous** Markov chain with state space $S = \{1, 2, \ldots, m\}$:

$$p_{ij} = P(X_{n+1} = j \mid X_n = i)$$

The **transition matrix** $P$ has entries $p_{ij}$:

$$P = \begin{pmatrix} p_{11} & p_{12} & \cdots & p_{1m} \\ p_{21} & p_{22} & \cdots & p_{2m} \\ \vdots & \vdots & \ddots & \vdots \\ p_{m1} & p_{m2} & \cdots & p_{mm} \end{pmatrix}$$

Properties: $p_{ij} \geq 0$ and $\sum_j p_{ij} = 1$ for each row $i$ (each row is a probability distribution).

### 2.3 Chapman-Kolmogorov Equations

The $n$-step transition probabilities satisfy:

$$p_{ij}^{(n)} = P(X_n = j \mid X_0 = i) = (P^n)_{ij}$$

$$p_{ij}^{(n+m)} = \sum_k p_{ik}^{(n)} \cdot p_{kj}^{(m)}$$

In matrix form: $P^{(n+m)} = P^{(n)} \cdot P^{(m)}$.

```python
def matrix_power(P, n):
    """Compute P^n for a square matrix P (list of lists)."""
    size = len(P)
    # Start with identity matrix
    result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]

    base = [row[:] for row in P]  # copy
    while n > 0:
        if n % 2 == 1:
            result = mat_mult(result, base)
        base = mat_mult(base, base)
        n //= 2
    return result

def mat_mult(A, B):
    """Multiply two square matrices."""
    size = len(A)
    C = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            for k in range(size):
                C[i][j] += A[i][k] * B[k][j]
    return C

# Example: Weather Markov chain
# States: 0=Sunny, 1=Cloudy, 2=Rainy
P = [[0.7, 0.2, 0.1],
     [0.3, 0.4, 0.3],
     [0.2, 0.3, 0.5]]

print("Transition matrix P:")
for row in P:
    print([f"{x:.1f}" for x in row])

# 2-step transitions
P2 = matrix_power(P, 2)
print("\nP^2 (2-step transitions):")
for row in P2:
    print([f"{x:.4f}" for x in row])

# 10-step transitions
P10 = matrix_power(P, 10)
print("\nP^10 (10-step transitions):")
for row in P10:
    print([f"{x:.4f}" for x in row])
```

---

## 3. Classification of States

### 3.1 Accessibility and Communication

- State $j$ is **accessible** from state $i$ if $p_{ij}^{(n)} > 0$ for some $n \geq 0$.
- States $i$ and $j$ **communicate** if each is accessible from the other. Write $i \leftrightarrow j$.
- Communication is an equivalence relation, partitioning the state space into **communicating classes**.
- A Markov chain is **irreducible** if there is only one communicating class (all states communicate).

### 3.2 Transient and Recurrent States

Let $f_{ii}$ be the probability of ever returning to state $i$ starting from $i$:

- **Recurrent**: $f_{ii} = 1$ (the chain returns to $i$ with probability 1).
- **Transient**: $f_{ii} < 1$ (there is a positive probability of never returning).

**Criterion**: State $i$ is recurrent if and only if $\sum_{n=0}^{\infty} p_{ii}^{(n)} = \infty$.

### 3.3 Absorbing States

A state $i$ is **absorbing** if $p_{ii} = 1$ (once entered, the chain stays there forever). Absorbing states are trivially recurrent.

### 3.4 Periodicity

The **period** of state $i$ is $d_i = \gcd\{n \geq 1 : p_{ii}^{(n)} > 0\}$.
- $d_i = 1$: state $i$ is **aperiodic**.
- $d_i > 1$: state $i$ is **periodic** with period $d_i$.

An irreducible Markov chain where all states are aperiodic and recurrent is called **ergodic**.

---

## 4. Stationary Distribution

### 4.1 Definition

A probability vector $\pi = (\pi_1, \pi_2, \ldots, \pi_m)$ is a **stationary distribution** if:

$$\pi P = \pi, \quad \sum_i \pi_i = 1, \quad \pi_i \geq 0$$

If the chain starts with distribution $\pi$, it remains in distribution $\pi$ at all future times.

### 4.2 Existence and Uniqueness

- An irreducible, positive recurrent Markov chain has a **unique** stationary distribution.
- For finite-state irreducible chains, the stationary distribution always exists and is unique.
- For an ergodic chain: $\lim_{n \to \infty} P^n_{ij} = \pi_j$ regardless of the initial state $i$.

### 4.3 Computing the Stationary Distribution

Solve the system $\pi P = \pi$ with the constraint $\sum \pi_i = 1$.

```python
def stationary_distribution(P, iterations=1000):
    """Find stationary distribution by repeated matrix multiplication.

    For an ergodic chain, any row of P^n converges to pi.
    """
    size = len(P)
    Pn = matrix_power(P, iterations)
    # Each row should converge to the same distribution
    pi = Pn[0]
    return pi

def stationary_by_solving(P):
    """Find stationary distribution by solving pi*P = pi.

    For a 3x3 matrix using simple Gaussian-like approach.
    Solves (P^T - I)pi = 0 with sum(pi) = 1.
    """
    size = len(P)
    # Set up equations: pi_j = sum_i pi_i * P_ij
    # Rearrange: sum_i pi_i * (P_ij - delta_ij) = 0
    # Replace last equation with sum(pi) = 1

    # Build augmented matrix [A | b]
    A = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            A[j][i] = P[i][j]  # Transpose
    for i in range(size):
        A[i][i] -= 1.0

    # Replace last row with sum constraint
    A[-1] = [1.0] * size
    b = [0.0] * size
    b[-1] = 1.0

    # Gaussian elimination (partial pivoting)
    for col in range(size):
        # Find pivot
        max_row = col
        for row in range(col + 1, size):
            if abs(A[row][col]) > abs(A[max_row][col]):
                max_row = row
        A[col], A[max_row] = A[max_row], A[col]
        b[col], b[max_row] = b[max_row], b[col]

        for row in range(col + 1, size):
            if A[col][col] == 0:
                continue
            factor = A[row][col] / A[col][col]
            for j in range(size):
                A[row][j] -= factor * A[col][j]
            b[row] -= factor * b[col]

    # Back substitution
    pi = [0.0] * size
    for i in range(size - 1, -1, -1):
        pi[i] = b[i]
        for j in range(i + 1, size):
            pi[i] -= A[i][j] * pi[j]
        if A[i][i] != 0:
            pi[i] /= A[i][i]
    return pi

# Weather chain stationary distribution
pi_iter = stationary_distribution(P)
pi_exact = stationary_by_solving(P)
print("Stationary distribution (iterative):", [f"{x:.4f}" for x in pi_iter])
print("Stationary distribution (solving):  ", [f"{x:.4f}" for x in pi_exact])
print("Interpretation: Long-run fraction of Sunny/Cloudy/Rainy days")
```

---

## 5. Poisson Process

### 5.1 Definition

A counting process $\{N(t), t \geq 0\}$ is a **Poisson process** with rate $\lambda > 0$ if:

1. $N(0) = 0$
2. Independent increments: counts on disjoint intervals are independent.
3. $N(t+s) - N(t) \sim \text{Poisson}(\lambda s)$ for all $t, s \geq 0$.

### 5.2 Properties

- **Inter-arrival times**: $T_1, T_2, \ldots$ are i.i.d. $\text{Exponential}(\lambda)$.
- **Arrival times**: $S_n = T_1 + \cdots + T_n \sim \text{Gamma}(n, \lambda)$.
- **Memoryless property**: $P(T > t + s \mid T > t) = P(T > s)$.
- **Merging**: If $N_1(t) \sim \text{Poisson}(\lambda_1 t)$ and $N_2(t) \sim \text{Poisson}(\lambda_2 t)$ are independent, then $N_1(t) + N_2(t) \sim \text{Poisson}((\lambda_1 + \lambda_2)t)$.
- **Splitting**: Each event independently assigned to type 1 (probability $p$) or type 2 yields two independent Poisson processes with rates $\lambda p$ and $\lambda(1-p)$.

### 5.3 Conditional Distribution of Arrivals

Given $N(t) = n$, the $n$ arrival times are distributed as the order statistics of $n$ i.i.d. $\text{Uniform}(0, t)$ random variables.

```python
import random
import math

def simulate_poisson_process(lam, T, seed=42):
    """Simulate a Poisson process on [0, T] with rate lambda."""
    random.seed(seed)
    arrivals = []
    t = 0
    while True:
        # Inter-arrival time ~ Exponential(lambda)
        inter_arrival = -math.log(1 - random.random()) / lam
        t += inter_arrival
        if t > T:
            break
        arrivals.append(t)
    return arrivals

# Simulate customer arrivals (rate = 3 per hour) over 8 hours
arrivals = simulate_poisson_process(lam=3, T=8)
print(f"Poisson process (lambda=3, T=8):")
print(f"  Total arrivals: {len(arrivals)} (expected: {3*8})")

# Count arrivals per hour
for hour in range(8):
    count = sum(1 for a in arrivals if hour <= a < hour + 1)
    bar = "#" * count
    print(f"  Hour {hour}-{hour+1}: {count:>2} arrivals  {bar}")

# Inter-arrival times
if len(arrivals) > 1:
    inter_arrivals = [arrivals[0]] + [arrivals[i] - arrivals[i-1] for i in range(1, len(arrivals))]
    mean_ia = sum(inter_arrivals) / len(inter_arrivals)
    print(f"  Mean inter-arrival time: {mean_ia:.3f} (expected: {1/3:.3f})")
```

---

## 6. Random Walk

### 6.1 Simple Random Walk

At each step, the walker moves $+1$ with probability $p$ or $-1$ with probability $q = 1 - p$:

$$X_n = X_0 + \sum_{i=1}^{n} Z_i, \quad Z_i = \begin{cases} +1 & \text{prob } p \\ -1 & \text{prob } q \end{cases}$$

**Properties**:
- $E[X_n] = X_0 + n(p - q) = X_0 + n(2p - 1)$
- $\text{Var}(X_n) = 4npq$
- **Symmetric** ($p = 1/2$): Recurrent in 1D and 2D, transient in 3D+.

### 6.2 Random Walk with Drift

When $p \neq 1/2$, the walk has a drift:

$$E[X_n] = X_0 + n\mu, \quad \mu = p - q$$

For $p > 1/2$, the walk drifts upward; for $p < 1/2$, downward.

```python
import random

def simulate_random_walks(n_steps, n_walks, p=0.5, seed=42):
    """Simulate multiple random walks."""
    random.seed(seed)
    walks = []
    for _ in range(n_walks):
        position = 0
        path = [position]
        for _ in range(n_steps):
            step = 1 if random.random() < p else -1
            position += step
            path.append(position)
        walks.append(path)
    return walks

# Symmetric random walk
walks = simulate_random_walks(n_steps=100, n_walks=5, p=0.5)
print("Symmetric Random Walk (p=0.5), 5 walks of 100 steps:")
print(f"{'Walk':<6} {'Final':>6} {'Max':>6} {'Min':>6}")
for i, w in enumerate(walks):
    print(f"{i+1:<6} {w[-1]:>6} {max(w):>6} {min(w):>6}")

# Random walk with drift
walks_drift = simulate_random_walks(n_steps=100, n_walks=5, p=0.6)
print(f"\nRandom Walk with Drift (p=0.6):")
print(f"Expected final position: {100 * (0.6 - 0.4):.0f}")
print(f"{'Walk':<6} {'Final':>6}")
for i, w in enumerate(walks_drift):
    print(f"{i+1:<6} {w[-1]:>6}")
```

---

## 7. Gambler's Ruin Problem

### 7.1 Setup

A gambler starts with $i$ dollars and plays a fair (or biased) game. Each round: win \$1 with probability $p$, lose \$1 with probability $q = 1 - p$. The game ends when the gambler reaches $N$ dollars (wins) or $0$ (ruin).

### 7.2 Ruin Probability

Let $r_i = P(\text{ruin} \mid X_0 = i)$.

**Fair game** ($p = q = 1/2$):

$$r_i = 1 - \frac{i}{N}$$

**Biased game** ($p \neq q$):

$$r_i = \frac{(q/p)^i - (q/p)^N}{1 - (q/p)^N}$$

### 7.3 Expected Duration

For the fair game starting at $i$:

$$E[\text{duration}] = i(N - i)$$

```python
import random

def gamblers_ruin_analytical(i, N, p):
    """Compute ruin probability analytically."""
    q = 1 - p
    if abs(p - 0.5) < 1e-10:  # fair game
        return 1 - i / N
    else:
        ratio = q / p
        return (ratio**i - ratio**N) / (1 - ratio**N)

def gamblers_ruin_simulation(i, N, p, n_simulations=10000, seed=42):
    """Simulate the gambler's ruin problem."""
    random.seed(seed)
    ruins = 0
    total_steps = 0
    for _ in range(n_simulations):
        position = i
        steps = 0
        while 0 < position < N:
            position += 1 if random.random() < p else -1
            steps += 1
        if position == 0:
            ruins += 1
        total_steps += steps

    return {
        "ruin_prob": ruins / n_simulations,
        "avg_duration": total_steps / n_simulations
    }

# Example: start with $20, target $100
i, N = 20, 100
for p in [0.5, 0.49, 0.48, 0.45]:
    analytical = gamblers_ruin_analytical(i, N, p)
    sim = gamblers_ruin_simulation(i, N, p, n_simulations=5000)
    print(f"p={p:.2f}: P(ruin) analytical={analytical:.4f}, simulated={sim['ruin_prob']:.4f}, "
          f"avg duration={sim['avg_duration']:.0f}")
```

---

## 8. Stationarity

### 8.1 Strict-Sense Stationarity (SSS)

A process $\{X(t)\}$ is **strictly stationary** if for all $n$, all times $t_1, \ldots, t_n$, and all shifts $\tau$:

$$(X(t_1), \ldots, X(t_n)) \overset{d}{=} (X(t_1 + \tau), \ldots, X(t_n + \tau))$$

All finite-dimensional distributions are invariant to time shifts. This is a very strong condition.

### 8.2 Wide-Sense (Weak) Stationarity (WSS)

A process is **wide-sense stationary** if:

1. $E[X(t)] = \mu$ (constant mean, independent of $t$)
2. $\text{Cov}(X(t), X(t+\tau)) = C(\tau)$ (autocovariance depends only on the lag $\tau$, not on $t$)
3. $E[|X(t)|^2] < \infty$

SSS implies WSS (when second moments exist), but WSS does not imply SSS in general. Exception: for Gaussian processes, WSS and SSS are equivalent.

---

## 9. Autocorrelation Function

### 9.1 Definition

For a WSS process with mean $\mu$:

**Autocovariance function**:
$$C(\tau) = \text{Cov}(X(t), X(t+\tau)) = E[(X(t) - \mu)(X(t+\tau) - \mu)]$$

**Autocorrelation function (ACF)**:
$$R(\tau) = \frac{C(\tau)}{C(0)} = \frac{\text{Cov}(X(t), X(t+\tau))}{\text{Var}(X(t))}$$

### 9.2 Properties

- $C(0) = \text{Var}(X(t)) \geq 0$
- $C(\tau) = C(-\tau)$ (symmetric)
- $|C(\tau)| \leq C(0)$ (bounded)
- $C(\tau)$ is positive semi-definite

### 9.3 Sample Autocorrelation

For a time series $x_1, \ldots, x_n$:

$$\hat{R}(k) = \frac{\sum_{t=1}^{n-k}(x_t - \bar{x})(x_{t+k} - \bar{x})}{\sum_{t=1}^{n}(x_t - \bar{x})^2}$$

```python
def sample_acf(data, max_lag=None):
    """Compute sample autocorrelation function."""
    n = len(data)
    if max_lag is None:
        max_lag = min(n // 4, 20)
    mean = sum(data) / n
    var = sum((x - mean)**2 for x in data)

    acf_values = []
    for k in range(max_lag + 1):
        cov_k = sum((data[t] - mean) * (data[t + k] - mean) for t in range(n - k))
        acf_values.append(cov_k / var if var > 0 else 0)
    return acf_values

# Example: ACF of a simulated AR(1) process
# X_t = 0.8 * X_{t-1} + epsilon_t
random.seed(42)
n = 200
phi = 0.8
x = [0.0]
for t in range(1, n):
    x.append(phi * x[-1] + random.gauss(0, 1))

acf = sample_acf(x, max_lag=10)
print("Sample ACF of AR(1) process (phi=0.8):")
print(f"Theoretical ACF: R(k) = 0.8^k")
print(f"{'Lag':>4} {'Sample':>8} {'Theory':>8}")
for k, r in enumerate(acf):
    print(f"{k:>4} {r:>8.4f} {phi**k:>8.4f}")
```

---

## 10. Markov Chain Simulation: Complete Example

```python
import random

def simulate_markov_chain(P, initial_state, n_steps, seed=42):
    """Simulate a Markov chain trajectory.

    Args:
        P: transition matrix (list of lists)
        initial_state: starting state index
        n_steps: number of transitions
        seed: random seed
    """
    random.seed(seed)
    states = [initial_state]
    current = initial_state

    for _ in range(n_steps):
        r = random.random()
        cumsum = 0
        for j, prob in enumerate(P[current]):
            cumsum += prob
            if r < cumsum:
                current = j
                break
        states.append(current)
    return states

# Two-state Markov chain (healthy/sick)
P_health = [[0.95, 0.05],  # Healthy -> Healthy, Healthy -> Sick
            [0.30, 0.70]]  # Sick -> Healthy, Sick -> Sick

labels = ["Healthy", "Sick"]
chain = simulate_markov_chain(P_health, initial_state=0, n_steps=365)

# Count state frequencies
counts = [0, 0]
for s in chain:
    counts[s] += 1

print("=== Health Markov Chain (365 days) ===")
print(f"Days Healthy: {counts[0]}, Days Sick: {counts[1]}")
print(f"Empirical: Healthy={counts[0]/len(chain):.3f}, Sick={counts[1]/len(chain):.3f}")

# Analytical stationary distribution
pi = stationary_by_solving(P_health)
print(f"Stationary:  Healthy={pi[0]:.3f}, Sick={pi[1]:.3f}")

# Absorbing Markov chain example
print("\n=== Absorbing Markov Chain ===")
# States: 0=Start, 1=Middle, 2=Win(absorbing), 3=Lose(absorbing)
P_abs = [[0.0, 0.6, 0.3, 0.1],
         [0.2, 0.0, 0.5, 0.3],
         [0.0, 0.0, 1.0, 0.0],  # Absorbing (Win)
         [0.0, 0.0, 0.0, 1.0]]  # Absorbing (Lose)

n_sim = 10000
wins = 0
total_steps_to_absorb = 0
random.seed(42)
for _ in range(n_sim):
    state = 0
    steps = 0
    while state not in [2, 3]:
        r = random.random()
        cumsum = 0
        for j, prob in enumerate(P_abs[state]):
            cumsum += prob
            if r < cumsum:
                state = j
                break
        steps += 1
    if state == 2:
        wins += 1
    total_steps_to_absorb += steps

print(f"P(Win) from state 0: {wins/n_sim:.4f}")
print(f"P(Lose) from state 0: {(n_sim-wins)/n_sim:.4f}")
print(f"Avg steps to absorption: {total_steps_to_absorb/n_sim:.2f}")
```

---

## 11. Key Takeaways

| Concept | Key Point |
|---|---|
| Stochastic process | Collection of RVs $\{X(t), t \in T\}$; classified by time and state space |
| Markov property | Future depends only on present, not past |
| Transition matrix | $P^n$ gives $n$-step transition probabilities |
| State classification | Recurrent ($f_{ii} = 1$), transient ($f_{ii} < 1$), absorbing ($p_{ii} = 1$) |
| Stationary distribution | $\pi P = \pi$; long-run proportion of time in each state |
| Poisson process | Memoryless counting process; exponential inter-arrivals |
| Random walk | Recurrent in 1D/2D, transient in 3D+; $E[X_n] = n(2p-1)$ |
| Gambler's ruin | $P(\text{ruin}) = 1 - i/N$ for fair game; always ruins in fair infinite game |
| Wide-sense stationarity | Constant mean, autocovariance depends only on lag |
| ACF | $R(\tau)$: measures linear dependence at lag $\tau$ |

**Looking ahead**: These foundational concepts lead to continuous-time processes (Brownian motion), martingales, Markov Chain Monte Carlo (MCMC), hidden Markov models, and stochastic calculus -- the mathematical backbone of finance, signal processing, and modern machine learning.

---

**Previous**: [Regression and ANOVA](./17_Regression_and_ANOVA.md) | **Next**: (End of current series)
