# Law of Large Numbers and Central Limit Theorem

**Previous**: [Convergence Concepts](./10_Convergence_Concepts.md) | **Next**: [Point Estimation](./12_Point_Estimation.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. State and prove the Weak Law of Large Numbers using Chebyshev's inequality
2. State the Strong Law of Large Numbers and explain how it differs from the WLLN
3. Apply LLN to Monte Carlo estimation problems
4. State the Central Limit Theorem with its conditions and interpret its meaning
5. Describe the Berry-Esseen bound and its implications for convergence rate
6. Apply the continuity correction when approximating discrete distributions with the normal
7. Preview how the CLT underpins confidence intervals and polling
8. Simulate LLN and CLT in Python to build visual intuition

---

The Law of Large Numbers (LLN) and the Central Limit Theorem (CLT) are the two most important theorems in probability. Together they explain why averages behave predictably and why the normal distribution appears everywhere. The LLN tells us **where** the sample mean converges; the CLT tells us **how** it fluctuates around that limit.

---

## 1. Weak Law of Large Numbers (WLLN)

### 1.1 Statement

Let $X_1, X_2, \ldots$ be i.i.d. random variables with $E[X_i] = \mu$ and $\text{Var}(X_i) = \sigma^2 < \infty$. Define the sample mean:

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$$

Then for every $\varepsilon > 0$:

$$\lim_{n \to \infty} P\!\left(|\bar{X}_n - \mu| > \varepsilon\right) = 0$$

That is, $\bar{X}_n \xrightarrow{P} \mu$.

### 1.2 Proof via Chebyshev's Inequality

Chebyshev's inequality states: $P(|Y - E[Y]| \ge \varepsilon) \le \text{Var}(Y)/\varepsilon^2$.

Apply it to $Y = \bar{X}_n$:

$$E[\bar{X}_n] = \mu, \qquad \text{Var}(\bar{X}_n) = \frac{\sigma^2}{n}$$

Therefore:

$$P\!\left(|\bar{X}_n - \mu| \ge \varepsilon\right) \le \frac{\sigma^2}{n\varepsilon^2} \to 0 \text{ as } n \to \infty$$

This proof is elementary but requires the finite variance assumption. More general versions of the WLLN require only that $E[|X_i|] < \infty$.

### 1.3 Interpretation

The WLLN justifies the use of sample averages as estimates of population means. It guarantees that with enough data, the sample mean is likely to be close to the true mean.

---

## 2. Strong Law of Large Numbers (SLLN)

### 2.1 Statement

Under the same conditions as the WLLN (or more generally, only requiring $E[|X_i|] < \infty$ for the Kolmogorov version):

$$P\!\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1$$

That is, $\bar{X}_n \xrightarrow{a.s.} \mu$.

### 2.2 WLLN vs. SLLN

| Property | WLLN | SLLN |
|----------|------|------|
| Convergence mode | In probability | Almost sure |
| Conditions | Finite variance (simple proof) | Finite mean (Kolmogorov) |
| Strength | Weaker | Stronger |
| Meaning | $P(\lvert\bar{X}_n - \mu\rvert > \varepsilon) \to 0$ | Sample paths converge |

The SLLN says that a **single** infinitely long sequence of observations will produce a sample mean that converges to $\mu$. The WLLN only says that the probability of a large deviation vanishes.

### 2.3 Intuition for the Proof

The SLLN proof (e.g., via the Kolmogorov three-series theorem or truncation arguments) is substantially more involved than the WLLN. The key idea is that the fourth moment bound $E[(\bar{X}_n - \mu)^4] = O(1/n^2)$ makes the tail probabilities summable, allowing application of the Borel-Cantelli lemma.

---

## 3. Practical Implications: Monte Carlo Estimation

### 3.1 The Monte Carlo Method

To estimate $\theta = E[g(X)]$ where direct computation is difficult:

1. Draw $X_1, X_2, \ldots, X_n$ i.i.d. from the distribution of $X$.
2. Compute $\hat{\theta}_n = \frac{1}{n}\sum_{i=1}^n g(X_i)$.
3. By the SLLN, $\hat{\theta}_n \to \theta$ almost surely.

### 3.2 Example: Estimating $\pi$

Consider the unit square $[0,1]^2$ and the quarter-circle $x^2 + y^2 \le 1$. The area of the quarter-circle is $\pi/4$. If $(U_1, U_2) \sim \text{Uniform}([0,1]^2)$:

$$\pi = 4 \cdot E\!\left[\mathbf{1}(U_1^2 + U_2^2 \le 1)\right]$$

### 3.3 Monte Carlo Error

By the CLT (covered next), the Monte Carlo error is approximately:

$$\hat{\theta}_n - \theta \approx N\!\left(0,\, \frac{\text{Var}(g(X))}{n}\right)$$

The error decreases as $O(1/\sqrt{n})$, meaning we need 100 times more samples to gain one extra decimal digit of accuracy.

---

## 4. Central Limit Theorem (CLT)

### 4.1 Statement

Let $X_1, X_2, \ldots$ be i.i.d. with $E[X_i] = \mu$ and $0 < \text{Var}(X_i) = \sigma^2 < \infty$. Then:

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} = \frac{\sqrt{n}(\bar{X}_n - \mu)}{\sigma} \xrightarrow{d} N(0, 1)$$

Equivalently, $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} N(0, \sigma^2)$.

### 4.2 Conditions

The classical (Lindeberg-Levy) CLT requires:

1. **Independence**: The $X_i$ are independent.
2. **Identical distribution**: All $X_i$ have the same distribution.
3. **Finite variance**: $0 < \sigma^2 < \infty$.

The distribution of each $X_i$ can be **anything** (discrete, continuous, skewed, multimodal) as long as these conditions hold.

### 4.3 Why It Matters

The CLT explains:

- Why the normal distribution appears so frequently in nature (heights, measurement errors, etc.)
- Why sample means are approximately normal even when the underlying data is not
- Why many statistical procedures based on normality work well even for non-normal data

### 4.4 Proof Sketch (via MGF)

The MGF of $Z_n = \sqrt{n}(\bar{X}_n - \mu)/\sigma$ is:

$$M_{Z_n}(t) = \left[M_X\!\left(\frac{t}{\sigma\sqrt{n}}\right) e^{-\mu t/(\sigma\sqrt{n})}\right]^n$$

Expanding the MGF of $X$ around zero and taking $n \to \infty$, one shows $M_{Z_n}(t) \to e^{t^2/2}$, which is the MGF of $N(0,1)$.

---

## 5. Berry-Esseen Bound

### 5.1 Statement

Under the CLT assumptions, if additionally $E[|X_i - \mu|^3] = \rho < \infty$, then:

$$\sup_x \left|P\!\left(\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \le x\right) - \Phi(x)\right| \le \frac{C\rho}{\sigma^3 \sqrt{n}}$$

where $C$ is a universal constant (known to be at most $0.4748$).

### 5.2 Interpretation

- The CLT tells us convergence happens; the Berry-Esseen bound tells us **how fast**: the error in the normal approximation is $O(1/\sqrt{n})$.
- Distributions with large third moments (heavy skew) converge more slowly.
- The bound is uniform over all $x$, so it controls the worst-case CDF discrepancy.

---

## 6. Normal Approximation and Continuity Correction

### 6.1 Approximating Discrete Distributions

When using the CLT to approximate a discrete distribution (e.g., Binomial) with the normal:

$$P(X = k) \approx P(k - 0.5 < Y < k + 0.5)$$

where $Y \sim N(\mu, \sigma^2)$.

### 6.2 Example: Binomial Approximation

Let $X \sim \text{Binomial}(100, 0.3)$. Then $\mu = 30$, $\sigma^2 = 21$, $\sigma \approx 4.583$.

To find $P(X \le 25)$:

- **Without correction**: $P\!\left(Z \le \frac{25 - 30}{4.583}\right) = P(Z \le -1.091) \approx 0.1377$
- **With correction**: $P\!\left(Z \le \frac{25.5 - 30}{4.583}\right) = P(Z \le -0.982) \approx 0.1631$

The continuity correction adds 0.5 to account for the discrete-to-continuous gap. For moderate $n$, this correction significantly improves accuracy.

---

## 7. Applications

### 7.1 Confidence Intervals (Preview)

By the CLT, for large $n$:

$$P\!\left(-z_{\alpha/2} \le \frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \le z_{\alpha/2}\right) \approx 1 - \alpha$$

Rearranging:

$$P\!\left(\bar{X}_n - z_{\alpha/2}\frac{\sigma}{\sqrt{n}} \le \mu \le \bar{X}_n + z_{\alpha/2}\frac{\sigma}{\sqrt{n}}\right) \approx 1 - \alpha$$

For a 95% confidence interval, $z_{0.025} = 1.96$, giving:

$$\bar{X}_n \pm 1.96 \frac{\sigma}{\sqrt{n}}$$

### 7.2 Polling and Margin of Error

In election polling, each respondent is a Bernoulli trial with $p$ = true proportion. For $n$ respondents:

$$\hat{p} = \bar{X}_n, \qquad \text{SE}(\hat{p}) = \sqrt{\frac{p(1-p)}{n}} \approx \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

The "margin of error" reported by polls is typically $\pm 1.96 \cdot \text{SE}(\hat{p})$. For $n = 1000$ and $\hat{p} = 0.5$, this is approximately $\pm 3.1\%$.

### 7.3 Sample Size Determination

To achieve a margin of error $\varepsilon$ with confidence $1 - \alpha$:

$$n \ge \left(\frac{z_{\alpha/2}\, \sigma}{\varepsilon}\right)^2$$

For proportions with worst-case $p = 0.5$ and 95% confidence: $n \ge (1.96)^2/(4\varepsilon^2) = 0.9604/\varepsilon^2$.

---

## 8. Multivariate CLT (Brief)

### 8.1 Statement

Let $\mathbf{X}_1, \mathbf{X}_2, \ldots$ be i.i.d. random vectors in $\mathbb{R}^p$ with $E[\mathbf{X}_i] = \boldsymbol{\mu}$ and $\text{Cov}(\mathbf{X}_i) = \boldsymbol{\Sigma}$. Then:

$$\sqrt{n}(\bar{\mathbf{X}}_n - \boldsymbol{\mu}) \xrightarrow{d} N_p(\mathbf{0}, \boldsymbol{\Sigma})$$

### 8.2 Consequence

Each component of $\bar{\mathbf{X}}_n$ is asymptotically normal, and the joint distribution is asymptotically multivariate normal. This justifies multivariate confidence regions (ellipsoids) and simultaneous inference.

---

## 9. Python Simulation Examples

### 9.1 Law of Large Numbers: Convergence of Sample Mean

```python
import random

random.seed(42)

# Simulate rolling a fair die: E[X] = 3.5
mu = 3.5
n = 10_000
running_sum = 0

checkpoints = [10, 50, 100, 500, 1000, 5000, 10000]
print("LLN: Sample mean of fair die rolls converging to 3.5")
print("-" * 50)

for i in range(1, n + 1):
    running_sum += random.randint(1, 6)
    if i in checkpoints:
        mean_i = running_sum / i
        print(f"  n = {i:5d}:  X_bar = {mean_i:.4f}  "
              f"(error = {abs(mean_i - mu):.4f})")
```

### 9.2 Monte Carlo Estimation of Pi

```python
import random

random.seed(314)

print("\nMonte Carlo estimation of pi")
print("-" * 50)

for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
    inside = 0
    for _ in range(n):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            inside += 1
    pi_est = 4.0 * inside / n
    error = abs(pi_est - 3.141592653589793)
    print(f"  n = {n:>9,}:  pi ~ {pi_est:.6f}  (error = {error:.6f})")
```

### 9.3 CLT: Averaging Dice Rolls

```python
import random
import math

random.seed(2024)
n_sim = 50_000

# For a fair die: mu=3.5, sigma^2=35/12, sigma~1.7078
mu = 3.5
sigma = math.sqrt(35.0 / 12.0)

print("\nCLT: Distribution of standardised mean of n dice rolls")
print("Fraction in [-1.96, 1.96] should approach 0.95")
print("-" * 55)

for n in [2, 5, 10, 30, 100]:
    count_in = 0
    for _ in range(n_sim):
        rolls = [random.randint(1, 6) for _ in range(n)]
        x_bar = sum(rolls) / n
        z = (x_bar - mu) / (sigma / math.sqrt(n))
        if -1.96 <= z <= 1.96:
            count_in += 1
    frac = count_in / n_sim
    print(f"  n = {n:3d}:  P(-1.96 < Z < 1.96) = {frac:.4f}")
```

### 9.4 CLT Histogram via Text

```python
import random
import math

random.seed(100)
n = 30          # samples per mean
n_sim = 20_000  # number of means to compute

# Exponential(1): mu=1, sigma=1, heavily right-skewed
mu, sigma = 1.0, 1.0

z_values = []
for _ in range(n_sim):
    vals = [random.expovariate(1.0) for _ in range(n)]
    x_bar = sum(vals) / n
    z = (x_bar - mu) / (sigma / math.sqrt(n))
    z_values.append(z)

# Build histogram from -4 to 4 with 20 bins
n_bins = 20
lo, hi = -4.0, 4.0
bin_width = (hi - lo) / n_bins
bins = [0] * n_bins

for z in z_values:
    idx = int((z - lo) / bin_width)
    if 0 <= idx < n_bins:
        bins[idx] += 1

print(f"\nCLT Histogram: standardised mean of {n} Exp(1) samples")
print(f"(n_sim = {n_sim})")
print("-" * 55)
max_count = max(bins)
for i in range(n_bins):
    left = lo + i * bin_width
    bar_len = int(bins[i] * 50 / max_count) if max_count > 0 else 0
    bar = '#' * bar_len
    print(f"  [{left:5.1f},{left+bin_width:5.1f}): {bar}")
```

### 9.5 Continuity Correction Comparison

```python
import random
import math

random.seed(55)

# Binomial(100, 0.3): exact P(X <= 25)
# We estimate by simulation, then compare normal approx with/without correction
n_binom = 100
p = 0.3
mu = n_binom * p          # 30
sigma = math.sqrt(n_binom * p * (1 - p))  # sqrt(21) ~ 4.583

n_sim = 500_000
count_le_25 = 0
for _ in range(n_sim):
    x = sum(1 for _ in range(n_binom) if random.random() < p)
    if x <= 25:
        count_le_25 += 1

sim_prob = count_le_25 / n_sim

# Normal CDF approximation using error function
def normal_cdf(x, mu=0.0, sigma=1.0):
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

approx_no_corr = normal_cdf(25, mu, sigma)
approx_with_corr = normal_cdf(25.5, mu, sigma)

print(f"\nContinuity correction: P(Binomial(100, 0.3) <= 25)")
print(f"  Simulation:          {sim_prob:.4f}")
print(f"  Normal (no corr):    {approx_no_corr:.4f}")
print(f"  Normal (with corr):  {approx_with_corr:.4f}")
```

### 9.6 Sample Size for Polling

```python
import math

print("\nSample sizes needed for margin of error (95% CI, worst case p=0.5)")
print("-" * 50)

z = 1.96
for margin in [0.05, 0.03, 0.02, 0.01, 0.005]:
    n_needed = math.ceil((z ** 2) * 0.25 / (margin ** 2))
    print(f"  Margin = {margin:.3f} ({margin*100:.1f}%):  n >= {n_needed:,}")
```

---

## Key Takeaways

1. The **Weak Law of Large Numbers** guarantees that $\bar{X}_n \xrightarrow{P} \mu$; the proof via Chebyshev is direct and requires only finite variance.
2. The **Strong Law of Large Numbers** strengthens this to almost sure convergence, meaning individual sample paths converge.
3. **Monte Carlo methods** are a direct application of the LLN: estimate expectations by averaging random samples, with error $O(1/\sqrt{n})$.
4. The **Central Limit Theorem** states that standardised sample means are approximately $N(0,1)$ regardless of the underlying distribution, provided the variance is finite.
5. The **Berry-Esseen bound** quantifies the convergence rate of the CLT as $O(1/\sqrt{n})$.
6. For discrete distributions, the **continuity correction** improves the normal approximation.
7. The CLT underpins **confidence intervals**, **hypothesis tests**, and **polling margins of error**, making it the single most important theorem in applied statistics.

---

*Next lesson: [Point Estimation](./12_Point_Estimation.md)*
