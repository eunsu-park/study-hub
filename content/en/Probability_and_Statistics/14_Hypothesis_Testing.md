# Hypothesis Testing

**Previous**: [Interval Estimation](./13_Interval_Estimation.md) | **Next**: [Bayesian Inference](./15_Bayesian_Inference.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Formulate null and alternative hypotheses for common testing scenarios
2. Distinguish between Type I error, Type II error, and statistical power
3. Construct test statistics and determine rejection regions
4. State and apply the Neyman-Pearson lemma for most powerful tests
5. Compute and interpret p-values correctly
6. Perform z-tests, t-tests, and chi-squared tests
7. Apply the likelihood ratio test and Wilks' theorem
8. Address the multiple testing problem using Bonferroni and FDR corrections
9. Differentiate between statistical significance and practical significance

---

Hypothesis testing provides a formal framework for making decisions about population parameters based on sample data. It is the backbone of scientific inference, clinical trials, A/B testing, and quality control.

---

## 1. Null and Alternative Hypotheses

### 1.1 Framework

- **Null hypothesis** $H_0$: A statement of "no effect" or "no difference." It represents the status quo.
- **Alternative hypothesis** $H_1$ (or $H_a$): The claim we seek evidence for.

**Examples**:
- Testing a drug: $H_0: \mu_{\text{drug}} = \mu_{\text{placebo}}$ vs. $H_1: \mu_{\text{drug}} \neq \mu_{\text{placebo}}$
- Quality control: $H_0: p \leq 0.02$ vs. $H_1: p > 0.02$

### 1.2 Types of Tests

| Test Type | $H_0$ | $H_1$ |
|---|---|---|
| Two-sided | $\theta = \theta_0$ | $\theta \neq \theta_0$ |
| Right-tailed | $\theta \leq \theta_0$ | $\theta > \theta_0$ |
| Left-tailed | $\theta \geq \theta_0$ | $\theta < \theta_0$ |

---

## 2. Errors and Power

### 2.1 Type I and Type II Errors

|  | $H_0$ true | $H_0$ false |
|---|---|---|
| Reject $H_0$ | **Type I error** ($\alpha$) | Correct (Power) |
| Fail to reject $H_0$ | Correct | **Type II error** ($\beta$) |

- **Significance level** $\alpha = P(\text{reject } H_0 \mid H_0 \text{ true})$: Typically set at 0.05, 0.01, or 0.10.
- **Type II error** $\beta = P(\text{fail to reject } H_0 \mid H_1 \text{ true})$
- **Power** $= 1 - \beta = P(\text{reject } H_0 \mid H_1 \text{ true})$

### 2.2 Power Analysis

Power depends on:
1. **Effect size**: Larger true effects are easier to detect.
2. **Sample size** $n$: More data yields more power.
3. **Significance level** $\alpha$: Larger $\alpha$ gives more power (but more Type I errors).
4. **Variability** $\sigma$: Less noise yields more power.

```python
import math

def power_one_sample_z(mu_0, mu_1, sigma, n, alpha=0.05):
    """Compute power for a one-sample z-test (two-sided).

    H0: mu = mu_0 vs H1: mu = mu_1
    """
    z_alpha = 1.96 if alpha == 0.05 else 1.645  # two-sided for 0.05
    se = sigma / math.sqrt(n)
    # Non-centrality: how far mu_1 is from mu_0 in SE units
    delta = abs(mu_1 - mu_0) / se
    # Power ≈ P(Z > z_alpha - delta) + P(Z < -z_alpha - delta)
    # Using the approximation Phi(x) ≈ via the logistic function
    def phi(x):
        """Approximate standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    power = 1 - phi(z_alpha - delta) + phi(-z_alpha - delta)
    return power

# Example: detect a shift from mu_0=100 to mu_1=105, sigma=15
for n in [10, 25, 50, 100, 200]:
    pwr = power_one_sample_z(100, 105, sigma=15, n=n)
    print(f"n={n:>3d}: power = {pwr:.4f}")
```

---

## 3. Test Statistic and Rejection Region

### 3.1 General Procedure

1. State $H_0$ and $H_1$.
2. Choose significance level $\alpha$.
3. Compute a **test statistic** $T(X)$ from the sample.
4. Determine the **rejection region** $\mathcal{R}$: the set of values of $T$ that lead to rejecting $H_0$.
5. Reject $H_0$ if $T(X) \in \mathcal{R}$; otherwise fail to reject.

### 3.2 Example: One-Sample z-Test

$H_0: \mu = \mu_0$ vs. $H_1: \mu \neq \mu_0$ (two-sided), $\sigma$ known.

$$T = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}}$$

Rejection region: $|T| > z_{\alpha/2}$.

---

## 4. Neyman-Pearson Lemma

### 4.1 Statement

For testing simple hypotheses $H_0: \theta = \theta_0$ vs. $H_1: \theta = \theta_1$, the **most powerful** test of size $\alpha$ has rejection region:

$$\mathcal{R} = \left\{ x : \frac{L(\theta_1 \mid x)}{L(\theta_0 \mid x)} > k \right\}$$

where $k$ is chosen so that $P(X \in \mathcal{R} \mid H_0) = \alpha$.

### 4.2 Interpretation

The likelihood ratio test is **optimal** in the sense that no other test of the same size has higher power against $\theta_1$. This foundational result justifies the widespread use of likelihood-based test statistics.

### 4.3 Example

Testing $H_0: \mu = 0$ vs. $H_1: \mu = 1$ for $X_1, \ldots, X_n \sim N(\mu, 1)$:

$$\frac{L(1 \mid x)}{L(0 \mid x)} = \exp\left(n\bar{x} - \frac{n}{2}\right) > k$$

This reduces to rejecting when $\bar{x} > c$ for some constant $c$, which is exactly the one-sided z-test.

---

## 5. The p-Value

### 5.1 Definition

The **p-value** is the probability, under $H_0$, of observing a test statistic at least as extreme as the one actually observed:

$$p = P(T \geq T_{\text{obs}} \mid H_0) \quad \text{(one-sided)}$$

$$p = P(|T| \geq |T_{\text{obs}}| \mid H_0) \quad \text{(two-sided)}$$

### 5.2 Interpretation

- A small p-value indicates that the observed data are unlikely under $H_0$.
- Reject $H_0$ if $p \leq \alpha$.
- The p-value is **not** the probability that $H_0$ is true.
- The p-value is **not** the probability of making an error.

### 5.3 Common Misinterpretations to Avoid

1. "$p = 0.03$ means there is a 3% chance $H_0$ is true." -- **Wrong.**
2. "$1 - p$ is the probability that the result will replicate." -- **Wrong.**
3. "A smaller p-value means a larger effect." -- **Wrong.** p depends on both effect size and sample size.

---

## 6. Common Parametric Tests

### 6.1 One-Sample t-Test

$H_0: \mu = \mu_0$, $\sigma$ unknown:

$$T = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \sim t_{n-1} \quad \text{under } H_0$$

### 6.2 Two-Sample t-Test

**Independent samples** (equal variance assumed):

$$T = \frac{\bar{X}_1 - \bar{X}_2}{S_p \sqrt{1/n_1 + 1/n_2}}, \quad S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}$$

**Welch's t-test** (unequal variances): uses separate variances with Satterthwaite df.

### 6.3 Paired t-Test

For paired observations $(X_i, Y_i)$, compute differences $D_i = X_i - Y_i$ and apply a one-sample t-test on $D$:

$$T = \frac{\bar{D}}{S_D / \sqrt{n}} \sim t_{n-1}$$

```python
import math
import statistics

def one_sample_t_test(data, mu_0):
    """One-sample t-test (two-sided)."""
    n = len(data)
    x_bar = statistics.mean(data)
    s = statistics.stdev(data)
    t_stat = (x_bar - mu_0) / (s / math.sqrt(n))
    df = n - 1
    # Approximate two-sided p-value using normal (good for df > 30)
    def phi(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    p_value = 2 * (1 - phi(abs(t_stat)))
    return {"t_statistic": t_stat, "df": df, "p_value_approx": p_value}

def paired_t_test(x, y):
    """Paired t-test (two-sided)."""
    diffs = [xi - yi for xi, yi in zip(x, y)]
    return one_sample_t_test(diffs, mu_0=0)

# Example: test if mean differs from 50
data = [52, 48, 55, 51, 49, 53, 47, 56, 50, 54, 52, 48, 51, 53, 50]
result = one_sample_t_test(data, mu_0=50)
print(f"t = {result['t_statistic']:.4f}, df = {result['df']}, p ≈ {result['p_value_approx']:.4f}")

# Example: paired test (before vs after treatment)
before = [120, 135, 128, 140, 132, 125, 138, 130, 142, 136]
after  = [115, 130, 125, 136, 128, 120, 133, 126, 137, 131]
result_p = paired_t_test(before, after)
print(f"Paired t = {result_p['t_statistic']:.4f}, p ≈ {result_p['p_value_approx']:.4f}")
```

---

## 7. Chi-Squared Tests

### 7.1 Goodness of Fit

Test whether observed frequencies match expected frequencies under a theoretical distribution.

$$\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i} \sim \chi^2_{k-1-m}$$

where $k$ is the number of categories and $m$ is the number of estimated parameters.

### 7.2 Test of Independence

For an $r \times c$ contingency table, test whether two categorical variables are independent:

$$\chi^2 = \sum_{i=1}^{r}\sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad E_{ij} = \frac{R_i \cdot C_j}{N}$$

Degrees of freedom: $(r-1)(c-1)$.

```python
def chi_squared_gof(observed, expected):
    """Chi-squared goodness-of-fit test."""
    assert len(observed) == len(expected)
    chi2 = sum((o - e)**2 / e for o, e in zip(observed, expected))
    df = len(observed) - 1
    return {"chi2": chi2, "df": df}

def chi_squared_independence(table):
    """Chi-squared test of independence for a 2D contingency table.

    table: list of lists (rows x cols)
    """
    rows = len(table)
    cols = len(table[0])
    N = sum(sum(row) for row in table)
    row_totals = [sum(row) for row in table]
    col_totals = [sum(table[r][c] for r in range(rows)) for c in range(cols)]

    chi2 = 0
    for i in range(rows):
        for j in range(cols):
            expected = row_totals[i] * col_totals[j] / N
            chi2 += (table[i][j] - expected) ** 2 / expected
    df = (rows - 1) * (cols - 1)
    return {"chi2": chi2, "df": df}

# Goodness of fit: test if a die is fair
observed = [18, 15, 22, 20, 12, 13]  # 100 rolls
expected = [100/6] * 6
print("GoF:", chi_squared_gof(observed, expected))

# Independence: gender vs preference
table = [[30, 10, 20],   # Male:   A, B, C
         [25, 20, 15]]   # Female: A, B, C
print("Independence:", chi_squared_independence(table))
```

---

## 8. Likelihood Ratio Test and Wilks' Theorem

### 8.1 Generalized Likelihood Ratio

For testing $H_0: \theta \in \Theta_0$ vs. $H_1: \theta \in \Theta \setminus \Theta_0$:

$$\Lambda = \frac{\sup_{\theta \in \Theta_0} L(\theta \mid x)}{\sup_{\theta \in \Theta} L(\theta \mid x)}$$

Reject $H_0$ when $\Lambda$ is small (i.e., the restricted MLE fits much worse than the unrestricted MLE).

### 8.2 Wilks' Theorem

Under regularity conditions, as $n \to \infty$:

$$-2 \ln \Lambda \xrightarrow{d} \chi^2_r$$

where $r = \dim(\Theta) - \dim(\Theta_0)$ is the difference in the number of free parameters.

This is extremely useful because it provides an asymptotic null distribution without needing to derive the exact distribution of $\Lambda$.

---

## 9. Multiple Testing Problem

### 9.1 The Problem

When performing $m$ simultaneous tests at level $\alpha$, the probability of at least one false rejection (family-wise error rate, FWER) is:

$$\text{FWER} = 1 - (1 - \alpha)^m$$

For $m = 20$ tests at $\alpha = 0.05$: FWER $\approx 0.64$. Nearly two-thirds of the time, we get at least one false positive.

### 9.2 Bonferroni Correction

Reject the $i$-th hypothesis if $p_i \leq \alpha / m$. This controls FWER at level $\alpha$ but is conservative.

### 9.3 False Discovery Rate (FDR)

The **Benjamini-Hochberg (BH) procedure** controls the expected proportion of false discoveries among all rejections:

1. Order p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$.
2. Find the largest $k$ such that $p_{(k)} \leq \frac{k}{m} \alpha$.
3. Reject all hypotheses with $p_{(i)} \leq p_{(k)}$.

FDR control is less conservative than FWER and is preferred in high-dimensional settings (e.g., genomics).

```python
def bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction."""
    m = len(p_values)
    threshold = alpha / m
    results = [(i, p, p <= threshold) for i, p in enumerate(p_values)]
    return threshold, results

def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR procedure."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    # Find the BH threshold
    bh_threshold = 0
    for rank, (idx, p) in enumerate(indexed, 1):
        if p <= rank / m * alpha:
            bh_threshold = p
    # Reject all with p <= bh_threshold
    rejected = [idx for idx, p in enumerate(p_values) if p <= bh_threshold]
    return bh_threshold, rejected

# Example: 10 tests, some with small p-values
p_values = [0.001, 0.008, 0.039, 0.041, 0.052, 0.10, 0.21, 0.45, 0.67, 0.91]
bonf_thresh, bonf_results = bonferroni_correction(p_values)
bh_thresh, bh_rejected = benjamini_hochberg(p_values)

print(f"Bonferroni threshold: {bonf_thresh:.4f}")
print(f"Bonferroni rejections: {[i for i, p, rej in bonf_results if rej]}")
print(f"BH threshold: {bh_thresh:.4f}")
print(f"BH rejections: {bh_rejected}")
```

---

## 10. Effect Size and Practical Significance

### 10.1 The Distinction

**Statistical significance** ($p \leq \alpha$) means the observed effect is unlikely under $H_0$. It does **not** mean the effect is large or important.

**Practical significance** asks: is the effect large enough to matter in the real world?

### 10.2 Common Effect Size Measures

| Measure | Formula | Interpretation |
|---|---|---|
| Cohen's $d$ | $d = (\bar{X}_1 - \bar{X}_2)/S_p$ | Small: 0.2, Medium: 0.5, Large: 0.8 |
| Correlation $r$ | Pearson or point-biserial | Small: 0.1, Medium: 0.3, Large: 0.5 |
| Odds ratio | $\text{OR} = \frac{p_1/(1-p_1)}{p_2/(1-p_2)}$ | 1 = no effect |
| $\eta^2$ (ANOVA) | $\text{SS}_B / \text{SS}_T$ | Proportion of variance explained |

### 10.3 Why Both Matter

With a very large sample, even a trivial effect can be statistically significant. Conversely, a meaningful effect can be non-significant if the sample is too small. Always report both the p-value and the effect size.

```python
import math
import statistics

def cohens_d(group1, group2):
    """Compute Cohen's d for two independent groups."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
    var1 = statistics.variance(group1)
    var2 = statistics.variance(group2)
    # Pooled standard deviation
    sp = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (mean1 - mean2) / sp

# Large sample, tiny effect
import random
random.seed(42)
group_a = [random.gauss(100.0, 15) for _ in range(10000)]
group_b = [random.gauss(100.5, 15) for _ in range(10000)]
d = cohens_d(group_a, group_b)
print(f"Cohen's d = {d:.4f} (very small effect)")
print("Even with n=10000, a tiny d can yield a significant p-value.")
```

---

## 11. Key Takeaways

| Concept | Key Point |
|---|---|
| $H_0$ vs $H_1$ | $H_0$ is the status quo; burden of proof is on $H_1$ |
| Type I / Type II | Trade-off: reducing $\alpha$ increases $\beta$ |
| Power | Increases with $n$, effect size, and $\alpha$; decreases with $\sigma$ |
| Neyman-Pearson | LRT is the most powerful test for simple hypotheses |
| p-value | $P(\text{data this extreme or more} \mid H_0)$; not $P(H_0)$ |
| Multiple testing | Bonferroni controls FWER (conservative); BH controls FDR |
| Effect size | Statistical significance $\neq$ practical significance |
| Wilks' theorem | $-2\ln\Lambda \to \chi^2_r$ asymptotically |

**Reporting best practice**: State the test used, the test statistic, degrees of freedom, p-value, confidence interval, and effect size.

---

**Previous**: [Interval Estimation](./13_Interval_Estimation.md) | **Next**: [Bayesian Inference](./15_Bayesian_Inference.md)
