# Convergence Concepts

**Previous**: [Multivariate Normal Distribution](./09_Multivariate_Normal_Distribution.md) | **Next**: [Law of Large Numbers and CLT](./11_Law_of_Large_Numbers_and_CLT.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define and distinguish four modes of convergence: in distribution, in probability, almost surely, and in $L^p$
2. State the hierarchy of implications among these modes
3. Construct counterexamples showing that certain implications do not reverse
4. Apply Slutsky's theorem to simplify limit arguments
5. Use the continuous mapping theorem to pass limits through continuous functions
6. State the Delta method and apply it to approximate distributions of transformed estimators
7. Simulate sequences of random variables to illustrate each convergence mode

---

As sample sizes grow, sequences of random variables often settle down to predictable limits. But "settling down" can mean different things depending on the sense of convergence. This lesson formalises four notions of convergence, maps their logical relationships, and introduces the key theorems that make convergence arguments practical.

---

## 1. Convergence in Distribution (Weak Convergence)

### 1.1 Definition

A sequence $\{X_n\}$ **converges in distribution** to $X$, written $X_n \xrightarrow{d} X$, if:

$$\lim_{n \to \infty} F_{X_n}(x) = F_X(x) \quad \text{at every continuity point } x \text{ of } F_X$$

Equivalently, $E[g(X_n)] \to E[g(X)]$ for every bounded continuous function $g$.

### 1.2 Intuition

Convergence in distribution says that the CDFs approach each other pointwise (at continuity points). It is a statement about the **shapes** of distributions, not about the random variables being close on the same probability space.

### 1.3 Key Properties

- It is the **weakest** of the four modes: it makes no requirement that $X_n$ and $X$ be defined on the same probability space.
- The limit $X$ may be a constant $c$; then $X_n \xrightarrow{d} c$ means $F_{X_n}(x) \to \mathbf{1}(x \ge c)$.
- **Levy's Continuity Theorem**: $X_n \xrightarrow{d} X$ if and only if the characteristic functions converge pointwise: $\varphi_{X_n}(t) \to \varphi_X(t)$ for all $t$.

### 1.4 Example

Let $X_n \sim N(0, 1/n)$. Then $X_n \xrightarrow{d} 0$ because the distribution concentrates at zero as $n$ grows.

---

## 2. Convergence in Probability

### 2.1 Definition

$\{X_n\}$ **converges in probability** to $X$, written $X_n \xrightarrow{P} X$, if for every $\varepsilon > 0$:

$$\lim_{n \to \infty} P(|X_n - X| > \varepsilon) = 0$$

### 2.2 Intuition

For large $n$, the probability that $X_n$ is far from $X$ becomes negligible. Unlike convergence in distribution, this requires $X_n$ and $X$ to be on the **same** probability space and involves the **difference** $X_n - X$.

### 2.3 Example

Let $X_n \sim \text{Uniform}(0, 1/n)$. Then for any $\varepsilon > 0$ and $n > 1/\varepsilon$:

$$P(|X_n| > \varepsilon) = P(X_n > \varepsilon) = 0$$

So $X_n \xrightarrow{P} 0$.

### 2.4 Relationship to Convergence in Distribution

$$X_n \xrightarrow{P} X \implies X_n \xrightarrow{d} X$$

**Special case**: $X_n \xrightarrow{d} c$ (a constant) implies $X_n \xrightarrow{P} c$. This reverse direction holds only when the limit is a constant.

---

## 3. Almost Sure Convergence

### 3.1 Definition

$\{X_n\}$ **converges almost surely** (a.s.) to $X$, written $X_n \xrightarrow{a.s.} X$, if:

$$P\!\left(\lim_{n \to \infty} X_n = X\right) = 1$$

Equivalently: $P\!\left(\omega : X_n(\omega) \to X(\omega)\right) = 1$.

### 3.2 Intuition

For almost every outcome $\omega$, the sequence of numbers $X_1(\omega), X_2(\omega), \ldots$ converges to $X(\omega)$ in the ordinary calculus sense. The set of "bad" outcomes where convergence fails has probability zero.

### 3.3 Comparison with Convergence in Probability

Almost sure convergence is **stronger** than convergence in probability:

$$X_n \xrightarrow{a.s.} X \implies X_n \xrightarrow{P} X$$

The converse is false. The distinction: convergence in probability allows occasional "lapses" (where $X_n$ is far from $X$) as long as their probability shrinks. Almost sure convergence demands that the individual sample paths eventually stay close.

### 3.4 Borel-Cantelli Criterion

A practical tool: if $\sum_{n=1}^{\infty} P(|X_n - X| > \varepsilon) < \infty$ for every $\varepsilon > 0$, then $X_n \xrightarrow{a.s.} X$.

---

## 4. $L^p$ Convergence

### 4.1 Definition

$\{X_n\}$ **converges in $L^p$** to $X$ (for $p \ge 1$), written $X_n \xrightarrow{L^p} X$, if:

$$\lim_{n \to \infty} E\!\left[|X_n - X|^p\right] = 0$$

The most common case is $p = 2$ (**mean-square convergence**):

$$E\!\left[(X_n - X)^2\right] \to 0$$

### 4.2 Intuition

$L^p$ convergence controls the average magnitude of the deviation. It is sensitive to tail behaviour: if $X_n$ has a heavy tail that produces occasional large deviations, $L^p$ convergence may fail even when convergence in probability holds.

### 4.3 Relationship to Other Modes

$$X_n \xrightarrow{L^p} X \implies X_n \xrightarrow{P} X$$

This follows from Markov's inequality: $P(|X_n - X| > \varepsilon) \le E[|X_n - X|^p] / \varepsilon^p$.

Also, if $p > q \ge 1$, then $L^p$ convergence implies $L^q$ convergence (by Jensen's inequality).

---

## 5. The Hierarchy of Convergence Modes

### 5.1 Summary of Implications

```
        a.s.
         |
         v
  Lp --> Prob --> Dist
```

In detail:

| From | To | Holds? |
|------|----|--------|
| Almost sure | In probability | Yes |
| $L^p$ | In probability | Yes |
| In probability | In distribution | Yes |
| In distribution | In probability | Only if limit is a constant |
| In probability | Almost sure | No (in general) |
| In probability | $L^p$ | No (in general) |
| Almost sure | $L^p$ | No (in general) |
| $L^p$ | Almost sure | No (in general) |

### 5.2 Additional Conditions for Reverse Implications

- **Prob to a.s.**: If the convergence in probability is fast enough (e.g., summable tail probabilities via Borel-Cantelli), then a.s. convergence follows.
- **Prob to $L^p$**: If the sequence $\{|X_n - X|^p\}$ is **uniformly integrable**, then convergence in probability implies $L^p$ convergence.

---

## 6. Counterexamples

### 6.1 Convergence in Probability but Not Almost Surely

**The Typewriter Sequence**: Define random variables on $\Omega = [0, 1]$ with uniform measure. For each $n$, partition $[0, 1]$ into $n$ equal subintervals and cycle through them:

- $X_1 = \mathbf{1}_{[0, 1]}$
- $X_2 = \mathbf{1}_{[0, 1/2)}$, $X_3 = \mathbf{1}_{[1/2, 1]}$
- $X_4 = \mathbf{1}_{[0, 1/3)}$, $X_5 = \mathbf{1}_{[1/3, 2/3)}$, $X_6 = \mathbf{1}_{[2/3, 1]}$
- ...

For any $\varepsilon > 0$, $P(X_n > \varepsilon) \to 0$ (since the interval width shrinks), so $X_n \xrightarrow{P} 0$.

But for any $\omega \in [0, 1]$, infinitely many $X_n(\omega) = 1$, so $X_n(\omega) \not\to 0$. Hence $X_n \not\xrightarrow{a.s.} 0$.

### 6.2 Convergence in Probability but Not in $L^1$

Let $X_n = n$ with probability $1/n$ and $X_n = 0$ with probability $1 - 1/n$.

- $P(|X_n| > \varepsilon) = 1/n \to 0$, so $X_n \xrightarrow{P} 0$.
- $E[|X_n|] = n \cdot (1/n) = 1 \not\to 0$, so $X_n \not\xrightarrow{L^1} 0$.

The rare but large values prevent $L^1$ convergence.

### 6.3 Convergence in Distribution but Not in Probability

Let $X \sim N(0,1)$ and define $X_n = -X$ for all $n$. Then $X_n \sim N(0,1)$ for every $n$, so $X_n \xrightarrow{d} X$. But $P(|X_n - X| > \varepsilon) = P(|2X| > \varepsilon) > 0$ for all $n$, so $X_n \not\xrightarrow{P} X$.

---

## 7. Slutsky's Theorem

### 7.1 Statement

If $X_n \xrightarrow{d} X$ and $Y_n \xrightarrow{P} c$ (a constant), then:

1. $X_n + Y_n \xrightarrow{d} X + c$
2. $X_n Y_n \xrightarrow{d} cX$
3. $X_n / Y_n \xrightarrow{d} X / c$ (provided $c \ne 0$)

### 7.2 Importance

Slutsky's theorem is the workhorse for asymptotic arguments. It allows us to combine a term converging in distribution with a term converging in probability to a constant. This is used repeatedly in deriving the asymptotic distribution of test statistics.

### 7.3 Example Application

Suppose $\bar{X}_n \xrightarrow{d} N(\mu, \sigma^2/n)$ (via CLT) and $S_n^2 \xrightarrow{P} \sigma^2$ (by WLLN). Then:

$$T_n = \frac{\bar{X}_n - \mu}{S_n / \sqrt{n}} \xrightarrow{d} N(0, 1)$$

because $S_n/\sigma \xrightarrow{P} 1$ and we apply Slutsky's theorem.

---

## 8. Continuous Mapping Theorem

### 8.1 Statement

If $X_n \xrightarrow{d} X$ and $g$ is a continuous function, then $g(X_n) \xrightarrow{d} g(X)$.

The same holds with convergence in probability or almost sure convergence:

- $X_n \xrightarrow{P} X \implies g(X_n) \xrightarrow{P} g(X)$
- $X_n \xrightarrow{a.s.} X \implies g(X_n) \xrightarrow{a.s.} g(X)$

### 8.2 Example

If $X_n \xrightarrow{d} N(0, 1)$, then $X_n^2 \xrightarrow{d} \chi^2(1)$ by the continuous mapping theorem with $g(x) = x^2$.

---

## 9. Delta Method

### 9.1 Statement

Suppose $\sqrt{n}(X_n - \theta) \xrightarrow{d} N(0, \sigma^2)$ and $g$ is differentiable at $\theta$ with $g'(\theta) \ne 0$. Then:

$$\sqrt{n}\!\left(g(X_n) - g(\theta)\right) \xrightarrow{d} N\!\left(0,\; [g'(\theta)]^2 \sigma^2\right)$$

### 9.2 Intuition

The Delta method uses a first-order Taylor expansion: $g(X_n) \approx g(\theta) + g'(\theta)(X_n - \theta)$. The linear approximation inherits the asymptotic normality of $X_n$, with the variance scaled by $[g'(\theta)]^2$.

### 9.3 Example: Variance-Stabilising Transformation

If $X_n \sim \text{Poisson}(\lambda)/n$ (sample mean of Poissons), then $\sqrt{n}(X_n - \lambda) \xrightarrow{d} N(0, \lambda)$.

Applying $g(x) = \sqrt{x}$ with $g'(\lambda) = 1/(2\sqrt{\lambda})$:

$$\sqrt{n}\!\left(\sqrt{X_n} - \sqrt{\lambda}\right) \xrightarrow{d} N\!\left(0,\; \frac{1}{4}\right)$$

The variance no longer depends on $\lambda$, which is why the square-root transformation is called "variance-stabilising" for Poisson data.

### 9.4 Multivariate Delta Method

If $\sqrt{n}(\mathbf{X}_n - \boldsymbol{\theta}) \xrightarrow{d} N(\mathbf{0}, \boldsymbol{\Sigma})$ and $g: \mathbb{R}^p \to \mathbb{R}$ is differentiable at $\boldsymbol{\theta}$, then:

$$\sqrt{n}\!\left(g(\mathbf{X}_n) - g(\boldsymbol{\theta})\right) \xrightarrow{d} N\!\left(0,\; \nabla g(\boldsymbol{\theta})^T \boldsymbol{\Sigma}\, \nabla g(\boldsymbol{\theta})\right)$$

---

## 10. Python Simulation Examples

### 10.1 Convergence in Probability: Sample Mean

```python
import random

random.seed(42)

def sample_mean(dist_sampler, n):
    """Compute sample mean of n draws."""
    return sum(dist_sampler() for _ in range(n)) / n

# X_i ~ Uniform(0, 1), E[X] = 0.5
mu = 0.5
eps = 0.05

print("Convergence in probability: P(|X_bar_n - 0.5| > 0.05)")
print("-" * 50)

for n in [10, 50, 100, 500, 1000, 5000]:
    n_sim = 20_000
    violations = 0
    for _ in range(n_sim):
        x_bar = sample_mean(random.random, n)
        if abs(x_bar - mu) > eps:
            violations += 1
    prob = violations / n_sim
    print(f"  n = {n:5d}:  P(|X_bar - 0.5| > 0.05) = {prob:.4f}")
```

### 10.2 Almost Sure Convergence: Running Maximum of Averages

```python
import random

random.seed(7)
n_paths = 5
N = 2000

print("\nAlmost sure convergence: sample paths of cumulative mean")
print("(Each path should converge to 0.5)")
print("-" * 60)

for path in range(n_paths):
    running_sum = 0.0
    deviations_at_checkpoints = []
    for i in range(1, N + 1):
        running_sum += random.random()
        if i in [10, 100, 500, 1000, 2000]:
            mean_i = running_sum / i
            deviations_at_checkpoints.append((i, mean_i))

    results = ", ".join(f"n={i}: {m:.4f}" for i, m in deviations_at_checkpoints)
    print(f"  Path {path + 1}: {results}")
```

### 10.3 Convergence in Distribution: CLT Preview

```python
import random
import math

random.seed(123)
n_sim = 50_000

# Standardised sum of Exp(1) variables (mean=1, var=1)
# Should converge to N(0,1)

print("\nConvergence in distribution (CLT):")
print("Fraction of standardised means in [-1.96, 1.96] (should -> 0.95)")
print("-" * 60)

for n in [5, 20, 50, 200]:
    count_in = 0
    for _ in range(n_sim):
        vals = [random.expovariate(1.0) for _ in range(n)]
        x_bar = sum(vals) / n
        z = (x_bar - 1.0) / (1.0 / math.sqrt(n))
        if -1.96 <= z <= 1.96:
            count_in += 1
    fraction = count_in / n_sim
    print(f"  n = {n:4d}:  P(-1.96 < Z < 1.96) = {fraction:.4f}")
```

### 10.4 Counterexample: Prob Convergence Without L1 Convergence

```python
import random

random.seed(999)
n_sim = 100_000

print("\nCounterexample: Convergence in prob but not L1")
print("X_n = n with prob 1/n, else 0")
print("-" * 50)

for n in [10, 100, 1000, 10000]:
    deviations = 0
    total_abs = 0.0
    for _ in range(n_sim):
        if random.random() < 1.0 / n:
            x = n
        else:
            x = 0
        if abs(x) > 0.5:
            deviations += 1
        total_abs += abs(x)

    p_dev = deviations / n_sim
    e_abs = total_abs / n_sim
    print(f"  n = {n:5d}:  P(|Xn|>0.5) = {p_dev:.4f},  E[|Xn|] = {e_abs:.4f}")
```

### 10.5 Slutsky's Theorem Illustration

```python
import random
import math

random.seed(456)
n_sim = 50_000

print("\nSlutsky's theorem: X_n + Y_n where X_n ->d N(0,1), Y_n ->P 3")
print("-" * 60)

for n in [10, 50, 200, 1000]:
    sums = []
    for _ in range(n_sim):
        # X_n: standardised mean of n Uniform(0,1) -> N(0,1) by CLT
        vals = [random.random() for _ in range(n)]
        x_bar = sum(vals) / n
        x_n = (x_bar - 0.5) / (math.sqrt(1.0 / (12 * n)))

        # Y_n: converges to 3 (e.g., sample mean of Uniform(2,4))
        y_vals = [random.uniform(2, 4) for _ in range(n)]
        y_n = sum(y_vals) / n

        sums.append(x_n + y_n)

    mean_sum = sum(sums) / n_sim
    var_sum = sum((s - mean_sum) ** 2 for s in sums) / (n_sim - 1)
    print(f"  n = {n:4d}:  mean(Xn+Yn) = {mean_sum:.4f} (->3.0),  "
          f"var(Xn+Yn) = {var_sum:.4f} (->1.0)")
```

### 10.6 Delta Method: Square Root of Poisson Mean

```python
import random
import math

random.seed(789)
n_sim = 50_000
lam = 4.0  # Poisson parameter

print(f"\nDelta method: sqrt(X_bar) for Poisson({lam}) data")
print(f"Asymptotic variance of sqrt(n)*(sqrt(X_bar)-sqrt(lam)) should be 1/4")
print("-" * 60)

for n in [30, 100, 500, 2000]:
    scaled_diffs = []
    for _ in range(n_sim):
        # Generate n Poisson(lam) using standard library
        # Poisson via inverse transform
        total = 0
        for _ in range(n):
            L = math.exp(-lam)
            k = 0
            p = 1.0
            while True:
                p *= random.random()
                if p < L:
                    break
                k += 1
            total += k
        x_bar = total / n
        if x_bar > 0:
            scaled_diffs.append(math.sqrt(n) * (math.sqrt(x_bar) - math.sqrt(lam)))

    mean_d = sum(scaled_diffs) / len(scaled_diffs)
    var_d = sum((d - mean_d) ** 2 for d in scaled_diffs) / (len(scaled_diffs) - 1)
    print(f"  n = {n:4d}:  mean = {mean_d:.4f} (->0),  var = {var_d:.4f} (->0.25)")
```

---

## Key Takeaways

1. **Four modes of convergence** exist, from weakest to strongest (roughly): distribution, probability, $L^p$, almost sure.
2. **Almost sure** and **$L^p$** convergence both imply convergence in probability, which in turn implies convergence in distribution. No other general implications hold without extra conditions.
3. **Counterexamples** demonstrate that these modes are genuinely distinct: convergence in probability does not guarantee almost sure convergence or $L^p$ convergence.
4. **Slutsky's theorem** lets us combine a distributional limit with a probability limit to a constant, which is essential for deriving test statistics.
5. The **continuous mapping theorem** preserves all modes of convergence through continuous functions.
6. The **Delta method** extends asymptotic normality through differentiable transformations, providing approximate distributions for functions of estimators.

---

*Next lesson: [Law of Large Numbers and CLT](./11_Law_of_Large_Numbers_and_CLT.md)*
