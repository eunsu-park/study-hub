# A/B Testing for ML Models

[← Previous: 22. Production ML Serving](22_Production_ML_Serving.md) | [Next: Overview →](00_Overview.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why offline evaluation metrics alone are insufficient and why online A/B testing is necessary for ML models
2. Design an A/B test for an ML model by defining hypotheses, choosing metrics, calculating sample size, and selecting the randomization unit
3. Apply frequentist hypothesis testing (t-test, chi-square) and compute confidence intervals for model comparison
4. Use sequential testing methods (SPRT) to make valid early-stopping decisions without inflating false positive rates
5. Implement multi-armed bandit algorithms (epsilon-greedy, UCB, Thompson Sampling) as adaptive alternatives to fixed A/B splits
6. Define guardrail metrics to ensure model changes do not harm business-critical KPIs

---

You built a better model. Cross-validation says it's 2% more accurate. SHAP values look reasonable. The latency is within budget. Should you ship it? Not yet. Offline metrics measure *predictive accuracy*, but business value comes from *user behavior*. A recommendation model with higher accuracy might actually reduce revenue if its "accurate" suggestions are too niche. A fraud model with better precision might block too many legitimate transactions. A/B testing bridges the gap between "better by the metrics" and "better for the business." This lesson teaches the statistical methodology of online experimentation — how to design, run, and analyze A/B tests specifically for ML model comparisons.

---

> **Analogy**: Offline evaluation is like tasting a recipe in your kitchen — you know it's good. A/B testing is like serving it in a restaurant — you learn whether customers actually order it, finish it, and come back. Sometimes a recipe that tastes perfect to the chef doesn't sell, and sometimes a "worse" recipe outsells the "better" one because diners have different preferences than professional tasters.

---

## 1. Why A/B Test ML Models?

### 1.1 The Offline-Online Gap

```python
"""
Offline Metrics (what you measure in development):
  - Accuracy, F1, AUC-ROC, NDCG, BLEU, ...
  - Measured on historical, static test sets
  - Assumes past data represents future data

Online Metrics (what matters in production):
  - Click-through rate, conversion rate, revenue per user
  - User engagement, session duration, retention
  - Customer satisfaction, support ticket volume

Why they diverge:
  ┌──────────────────────────────────────────────────────────────────┐
  │ Model A: AUC 0.92 → recommends popular items → CTR +3%          │
  │ Model B: AUC 0.95 → recommends niche items  → CTR -1%           │
  │                                                                  │
  │ Model B is "better" offline, but WORSE for the business.         │
  │ This happens because offline metrics don't capture:              │
  │   1. User preference diversity                                   │
  │   2. Feedback loops (popular items get more clicks → more data)  │
  │   3. Position bias (users click top results regardless)          │
  │   4. Novelty effects (new model = new experience → short-term   │
  │      engagement, may not last)                                   │
  └──────────────────────────────────────────────────────────────────┘
"""
```

### 1.2 When to A/B Test

```python
"""
✓ Always A/B test when:
  - Deploying a new ML model version to production
  - Changing the model architecture or feature set
  - Modifying business logic downstream of the model
  - The change could affect user experience or revenue

✗ Skip A/B testing when:
  - Bug fix (model was clearly broken, fix is verified offline)
  - Regulatory/compliance requirement (must deploy regardless)
  - No traffic (new product with zero users)
  - Cost of testing exceeds potential benefit

⚠ Consider alternatives when:
  - Very slow feedback loop (months to measure outcome)
    → Use leading indicators or proxy metrics
  - Ethical concerns (can't withhold treatment from control)
    → Use quasi-experimental methods (diff-in-diff, regression discontinuity)
"""
```

---

## 2. Experimental Design

### 2.1 The Anatomy of an A/B Test

```python
"""
Components of an A/B Test:

1. Hypothesis:
   H₀ (null):        New model performs same as current model
   H₁ (alternative): New model performs BETTER than current model

2. Primary Metric (OEC — Overall Evaluation Criterion):
   One metric that captures the business goal.
   Example: "Revenue per user per session"

3. Randomization Unit:
   What gets randomly assigned to treatment/control.
   Options: user_id, session_id, request_id, cookie_id

4. Treatment/Control Split:
   Typically 50/50, but can be asymmetric (e.g., 90/10 for risky changes)

5. Sample Size:
   How many observations needed for statistical significance.
   Determined by: effect size, significance level, statistical power.

6. Duration:
   How long to run the test.
   Must cover at least: 1 full business cycle (usually 1-2 weeks)
"""
```

### 2.2 Choosing the Randomization Unit

```python
"""
Unit         Pros                            Cons
──────────────────────────────────────────────────────────────────
user_id      Consistent experience per       Need user authentication
             user; captures long-term        Slower to reach sample size
             effects

session_id   Faster data collection;         Same user may see both
             no auth needed                  treatments (inconsistent UX)

request_id   Fastest data collection;        No consistency within user
             simplest to implement           journey; high variance

cookie_id    Compromise between user         Cookie deletion/multiple
             and session                     devices cause spillover

Recommendation for ML models:
  - Personalization/recommendation → user_id (consistency matters)
  - Search ranking → session_id or query_id (faster iteration)
  - Fraud detection → request_id (each transaction is independent)
"""
```

### 2.3 Sample Size Calculation (Power Analysis)

```python
import numpy as np
from scipy import stats

def required_sample_size(baseline_rate, minimum_detectable_effect,
                         alpha=0.05, power=0.80):
    """
    Calculate the minimum sample size per group for a two-proportion z-test.

    Why: Running a test too short → underpowered → you may miss a real effect.
    Running a test too long → wasted time and resources.
    Power analysis tells you exactly how many observations you need.

    Parameters:
        baseline_rate: Current conversion rate (e.g., 0.05 for 5%)
        minimum_detectable_effect: Smallest effect worth detecting
            (relative, e.g., 0.10 for a 10% relative lift)
        alpha: Significance level (false positive rate)
        power: Statistical power (1 - false negative rate)

    Returns:
        Required sample size per group
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_detectable_effect)

    # Pooled standard error under H₀ and H₁
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # Two-tailed
    z_beta = stats.norm.ppf(power)

    p_pool = (p1 + p2) / 2
    se_null = np.sqrt(2 * p_pool * (1 - p_pool))
    se_alt = np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))

    n = ((z_alpha * se_null + z_beta * se_alt) / (p2 - p1)) ** 2
    return int(np.ceil(n))

# Example: 5% baseline CTR, want to detect 10% relative lift (5% → 5.5%)
n = required_sample_size(
    baseline_rate=0.05,
    minimum_detectable_effect=0.10,
    alpha=0.05,
    power=0.80
)
print(f"Required sample size per group: {n:,}")
print(f"Total samples needed: {2 * n:,}")

# How long will it take?
daily_traffic = 10000
days_needed = np.ceil(2 * n / daily_traffic)
print(f"At {daily_traffic:,} users/day: {days_needed:.0f} days")
```

### 2.4 Duration Considerations

```python
"""
Minimum test duration checklist:

□ Covers at least 1 full weekly cycle (weekday + weekend patterns)
□ Reaches required sample size
□ Avoids holidays and special events (Black Friday, etc.)
□ Accounts for novelty/primacy effects (users react differently to new things)
□ Allows time for long-term effects to manifest (1-2 week minimum)

Common mistake: "We got significant results after 2 hours!"
Problem: Early significance is often driven by:
  1. Time-of-day bias (morning users differ from evening users)
  2. Day-of-week effects (Monday traffic ≠ Saturday traffic)
  3. Multiple peeking (checking p-values repeatedly inflates Type I error)
"""
```

---

## 3. Statistical Foundations

### 3.1 Hypothesis Testing for Model Comparison

```python
def ab_test_proportions(n_control, conversions_control,
                        n_treatment, conversions_treatment,
                        alpha=0.05):
    """
    Two-proportion z-test for comparing conversion rates.

    Why z-test: For large sample sizes (n > 30), the Central Limit Theorem
    guarantees the sample proportion is approximately normal, making the
    z-test both simple and reliable.
    """
    p_control = conversions_control / n_control
    p_treatment = conversions_treatment / n_treatment
    p_pool = (conversions_control + conversions_treatment) / (n_control + n_treatment)

    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment))
    z_stat = (p_treatment - p_control) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed

    # Confidence interval for the difference
    se_diff = np.sqrt(p_control * (1 - p_control) / n_control +
                      p_treatment * (1 - p_treatment) / n_treatment)
    ci_lower = (p_treatment - p_control) - stats.norm.ppf(1 - alpha/2) * se_diff
    ci_upper = (p_treatment - p_control) + stats.norm.ppf(1 - alpha/2) * se_diff

    relative_lift = (p_treatment - p_control) / p_control * 100

    print(f"Control:   {p_control:.4f} ({conversions_control}/{n_control})")
    print(f"Treatment: {p_treatment:.4f} ({conversions_treatment}/{n_treatment})")
    print(f"Lift:      {relative_lift:+.2f}%")
    print(f"95% CI:    [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"p-value:   {p_value:.4f}")
    print(f"Result:    {'Significant' if p_value < alpha else 'Not significant'}")

    return {
        "p_control": p_control, "p_treatment": p_treatment,
        "lift": relative_lift, "p_value": p_value,
        "ci": (ci_lower, ci_upper),
        "significant": p_value < alpha
    }

# Example: New recommendation model vs. current
result = ab_test_proportions(
    n_control=50000, conversions_control=2500,       # 5.0% CTR
    n_treatment=50000, conversions_treatment=2650,    # 5.3% CTR
)
```

### 3.2 Continuous Metrics (Revenue, Engagement Time)

```python
def ab_test_means(control_values, treatment_values, alpha=0.05):
    """
    Welch's t-test for comparing means of continuous metrics.

    Why Welch's: Unlike Student's t-test, it doesn't assume equal variances.
    Revenue and engagement time often have very different variances between
    control and treatment (e.g., a new model might increase variance by
    creating more extreme outcomes).
    """
    n1, n2 = len(control_values), len(treatment_values)
    mean1, mean2 = np.mean(control_values), np.mean(treatment_values)
    std1, std2 = np.std(control_values, ddof=1), np.std(treatment_values, ddof=1)

    t_stat, p_value = stats.ttest_ind(control_values, treatment_values,
                                       equal_var=False)

    se_diff = np.sqrt(std1**2 / n1 + std2**2 / n2)
    ci_lower = (mean2 - mean1) - stats.t.ppf(1 - alpha/2, df=min(n1, n2) - 1) * se_diff
    ci_upper = (mean2 - mean1) + stats.t.ppf(1 - alpha/2, df=min(n1, n2) - 1) * se_diff

    print(f"Control mean:   {mean1:.4f} (std={std1:.4f})")
    print(f"Treatment mean: {mean2:.4f} (std={std2:.4f})")
    print(f"Difference:     {mean2 - mean1:+.4f}")
    print(f"95% CI:         [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"p-value:        {p_value:.4f}")

    return {"p_value": p_value, "ci": (ci_lower, ci_upper)}

# Example: Revenue per user (highly skewed)
np.random.seed(42)
control_revenue = np.random.exponential(scale=10, size=10000)
treatment_revenue = np.random.exponential(scale=10.5, size=10000)  # 5% lift
ab_test_means(control_revenue, treatment_revenue)
```

### 3.3 Multiple Comparison Correction

```python
"""
The Problem:
  Testing 1 metric at α=0.05 → 5% false positive rate
  Testing 10 metrics at α=0.05 → 1 - (1-0.05)^10 = 40% chance of ANY false positive!

Correction Methods:

  1. Bonferroni: α_adjusted = α / n_tests
     Simple but conservative. Use when tests are independent.
     10 metrics → α = 0.05/10 = 0.005

  2. Benjamini-Hochberg (FDR): Controls the false discovery rate
     Less conservative than Bonferroni. Use when you can tolerate some false positives.
     Sort p-values, compare each p(i) to (i/m) × α

  3. Primary + Secondary metric framework:
     Designate ONE primary metric (apply α=0.05)
     All others are "guardrails" (directional, no strict threshold)
     Most practical approach for ML A/B tests.
"""
```

```python
from scipy.stats import false_discovery_control

def apply_corrections(p_values, alpha=0.05):
    """Compare Bonferroni and BH corrections on multiple p-values."""
    n = len(p_values)
    metrics = [f"metric_{i+1}" for i in range(n)]

    # Bonferroni
    bonferroni_threshold = alpha / n

    # Benjamini-Hochberg
    sorted_indices = np.argsort(p_values)
    bh_thresholds = [(i + 1) / n * alpha for i in range(n)]
    bh_significant = [False] * n
    for rank, idx in enumerate(sorted_indices):
        if p_values[idx] <= bh_thresholds[rank]:
            bh_significant[idx] = True

    print(f"{'Metric':<12s} {'p-value':>8s} {'Raw':>5s} {'Bonf':>5s} {'BH':>5s}")
    print("-" * 42)
    for i in range(n):
        raw_sig = "✓" if p_values[i] < alpha else ""
        bonf_sig = "✓" if p_values[i] < bonferroni_threshold else ""
        bh_sig = "✓" if bh_significant[i] else ""
        print(f"{metrics[i]:<12s} {p_values[i]:>8.4f} {raw_sig:>5s} {bonf_sig:>5s} {bh_sig:>5s}")

# Example: 6 metrics tested, some are false positives
p_values = np.array([0.001, 0.013, 0.029, 0.041, 0.15, 0.88])
apply_corrections(p_values)
```

---

## 4. Sequential Testing

### 4.1 The Peeking Problem

```python
"""
The peeking problem:

  Day 1: p = 0.03 → "Significant! Ship it!"  ← WRONG!
  Day 2: p = 0.12 → "Hmm, not significant anymore..."
  Day 3: p = 0.04 → "It's back! Ship it!"     ← STILL WRONG!

Why it's wrong:
  Each peek is an independent test. If you check daily for 14 days,
  the probability of seeing at least one false positive is:
    1 - (1 - 0.05)^14 ≈ 51%

  The p-value is only valid for a SINGLE, pre-committed analysis.
  Repeated checking inflates the false positive rate dramatically.

Solutions:
  1. Pre-commit to a fixed sample size and analyze ONCE
  2. Use sequential testing methods (designed for continuous monitoring)
  3. Use Bayesian methods (naturally handle continuous updating)
"""
```

### 4.2 Sequential Probability Ratio Test (SPRT)

```python
def sequential_ab_test(control_outcomes, treatment_outcomes,
                       p0=0.05, p1=0.055, alpha=0.05, beta=0.20):
    """
    Sequential Probability Ratio Test for A/B testing.

    Why SPRT: It allows you to monitor results continuously and stop
    as soon as there's enough evidence — without inflating false positive rates.
    On average, SPRT needs 50-70% fewer samples than a fixed-sample test.

    Parameters:
        control_outcomes: Array of 0/1 outcomes for control
        treatment_outcomes: Array of 0/1 outcomes for treatment
        p0: Expected rate under H₀ (no difference)
        p1: Expected rate under H₁ (treatment is better)
        alpha: Type I error rate
        beta: Type II error rate
    """
    # Wald boundaries
    A = np.log((1 - beta) / alpha)     # Upper boundary (reject H₀)
    B = np.log(beta / (1 - alpha))     # Lower boundary (accept H₀)

    log_likelihood_ratio = 0
    decisions = []

    n = min(len(control_outcomes), len(treatment_outcomes))

    for i in range(n):
        x_c = control_outcomes[i]
        x_t = treatment_outcomes[i]

        # Log-likelihood ratio update
        if x_t == 1:
            log_likelihood_ratio += np.log(p1 / p0)
        else:
            log_likelihood_ratio += np.log((1 - p1) / (1 - p0))

        if log_likelihood_ratio >= A:
            decisions.append((i + 1, "REJECT H₀", "Treatment is better"))
            break
        elif log_likelihood_ratio <= B:
            decisions.append((i + 1, "ACCEPT H₀", "No significant difference"))
            break

    if not decisions:
        decisions.append((n, "INCONCLUSIVE", "Need more data"))

    sample_used, decision, interpretation = decisions[0]
    print(f"Decision at sample {sample_used}: {decision}")
    print(f"Interpretation: {interpretation}")
    print(f"Final log-LR: {log_likelihood_ratio:.4f}")
    print(f"Boundaries: reject H₀ > {A:.4f}, accept H₀ < {B:.4f}")
    return decisions

# Simulate: treatment has a real 10% relative improvement
np.random.seed(42)
control = np.random.binomial(1, 0.05, size=10000)
treatment = np.random.binomial(1, 0.055, size=10000)
sequential_ab_test(control, treatment)
```

---

## 5. Multi-Armed Bandits

### 5.1 A/B Testing vs. Bandits

```python
"""
A/B Testing:                      Multi-Armed Bandits:
┌─────────────────────────┐      ┌─────────────────────────┐
│ Fixed 50/50 split       │      │ Adaptive allocation     │
│ Explore for N samples   │      │ Explore + exploit        │
│ Then exploit winner     │      │ simultaneously           │
│                         │      │                         │
│ Regret: high during     │      │ Regret: lower because   │
│ exploration phase       │      │ traffic shifts to winner │
│                         │      │ automatically            │
│ Statistical rigor: ✓✓✓  │      │ Statistical rigor: ✓✓   │
│ Speed to deploy: slow   │      │ Speed to deploy: fast   │
└─────────────────────────┘      └─────────────────────────┘

When to use A/B testing:
  ✓ Need rigorous statistical conclusions
  ✓ Measuring long-term effects
  ✓ Management requires p-value evidence
  ✓ Few variants (2-3)

When to use bandits:
  ✓ Many variants to test (5+)
  ✓ High cost of serving worse variant
  ✓ Need to minimize regret during experiment
  ✓ Short-lived content (news, promotions)
"""
```

### 5.2 Epsilon-Greedy

```python
class EpsilonGreedy:
    """
    The simplest bandit algorithm.

    Why it works: With probability (1-ε), choose the best arm (exploit).
    With probability ε, choose a random arm (explore).
    Simple but effective — often a good baseline.
    """
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)       # How many times each arm was pulled
        self.rewards = np.zeros(n_arms)      # Total reward per arm
        self.estimates = np.zeros(n_arms)    # Estimated mean reward per arm

    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)  # Explore
        return np.argmax(self.estimates)            # Exploit

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.rewards[arm] += reward
        # Why: Incremental mean update — avoids storing all past rewards.
        # new_mean = old_mean + (reward - old_mean) / count
        self.estimates[arm] += (reward - self.estimates[arm]) / self.counts[arm]

# Simulate: 3 model variants with different true CTRs
np.random.seed(42)
true_rates = [0.05, 0.052, 0.058]  # Model C is best
bandit = EpsilonGreedy(n_arms=3, epsilon=0.1)

total_reward = 0
for t in range(10000):
    arm = bandit.select_arm()
    reward = np.random.binomial(1, true_rates[arm])
    bandit.update(arm, reward)
    total_reward += reward

print("Epsilon-Greedy Results:")
for i in range(3):
    print(f"  Model {chr(65+i)}: pulled {int(bandit.counts[i]):>5d} times, "
          f"est. rate={bandit.estimates[i]:.4f} (true={true_rates[i]:.3f})")
print(f"Total reward: {total_reward}")
```

### 5.3 Upper Confidence Bound (UCB)

```python
class UCB1:
    """
    UCB1 — balances exploration and exploitation using confidence intervals.

    Why UCB: Instead of exploring randomly (epsilon-greedy), UCB explores
    arms with high UNCERTAINTY. An arm that hasn't been tried much gets a
    wide confidence interval, making it more likely to be selected.
    This is provably optimal (up to constants) in the regret sense.
    """
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.estimates = np.zeros(n_arms)
        self.total_pulls = 0

    def select_arm(self):
        # Pull each arm once first
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i

        # UCB formula: estimate + sqrt(2 * ln(t) / count)
        # Why: The second term is the "exploration bonus" — it grows when
        # an arm hasn't been pulled recently, encouraging exploration.
        ucb_values = self.estimates + np.sqrt(
            2 * np.log(self.total_pulls) / self.counts
        )
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.total_pulls += 1
        self.estimates[arm] += (reward - self.estimates[arm]) / self.counts[arm]

# Compare UCB vs Epsilon-Greedy
ucb = UCB1(n_arms=3)
total_reward_ucb = 0
for t in range(10000):
    arm = ucb.select_arm()
    reward = np.random.binomial(1, true_rates[arm])
    ucb.update(arm, reward)
    total_reward_ucb += reward

print("\nUCB1 Results:")
for i in range(3):
    print(f"  Model {chr(65+i)}: pulled {int(ucb.counts[i]):>5d} times, "
          f"est. rate={ucb.estimates[i]:.4f} (true={true_rates[i]:.3f})")
print(f"Total reward: {total_reward_ucb} (vs ε-greedy: {total_reward})")
```

### 5.4 Thompson Sampling

```python
class ThompsonSampling:
    """
    Thompson Sampling — Bayesian approach to multi-armed bandits.

    Why Thompson Sampling: It maintains a probability distribution over
    each arm's true reward rate. Arms with high uncertainty get explored
    naturally because their samples occasionally produce high values.
    As evidence accumulates, the distribution narrows and the algorithm
    exploits the best arm.

    Often achieves the best empirical performance among bandit algorithms.
    """
    def __init__(self, n_arms):
        self.n_arms = n_arms
        # Beta(1, 1) = Uniform prior (no prior knowledge)
        self.alpha = np.ones(n_arms)  # Successes + 1
        self.beta = np.ones(n_arms)   # Failures + 1

    def select_arm(self):
        # Why: Sample from each arm's posterior Beta distribution.
        # The arm with the highest sample gets pulled.
        # High-uncertainty arms produce high samples more often → explored more.
        samples = [np.random.beta(self.alpha[i], self.beta[i])
                   for i in range(self.n_arms)]
        return np.argmax(samples)

    def update(self, arm, reward):
        # Why: Beta-Bernoulli conjugacy gives exact posterior update.
        # Success → alpha += 1, Failure → beta += 1.
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

# Run Thompson Sampling
np.random.seed(42)
ts = ThompsonSampling(n_arms=3)
total_reward_ts = 0
for t in range(10000):
    arm = ts.select_arm()
    reward = np.random.binomial(1, true_rates[arm])
    ts.update(arm, reward)
    total_reward_ts += reward

print("\nThompson Sampling Results:")
for i in range(3):
    pulls = int(ts.alpha[i] + ts.beta[i] - 2)
    est = (ts.alpha[i] - 1) / pulls if pulls > 0 else 0
    print(f"  Model {chr(65+i)}: pulled {pulls:>5d} times, "
          f"est. rate={est:.4f} (true={true_rates[i]:.3f})")
print(f"Total reward: {total_reward_ts} (vs UCB: {total_reward_ucb}, "
      f"ε-greedy: {total_reward})")
```

---

## 6. Interleaving Experiments

### 6.1 Why Interleaving?

```python
"""
For ranking/recommendation models, A/B testing requires MANY users because
each user sees either model A or model B results. Individual preference
is noisy, so you need large samples to detect small improvements.

Interleaving shows BOTH models' results to the SAME user in a merged list.
The user's clicks directly reveal which model's results they prefer.

Sensitivity: Interleaving needs ~100x fewer users than A/B testing
to detect the same effect (Chapelle et al., 2012).

Use cases: Search ranking, recommendation lists, ad ordering.

┌─────────────────────────────────────────────────────┐
│  A/B Test:                                          │
│    User 1 sees:  [A₁, A₂, A₃, A₄, A₅]             │
│    User 2 sees:  [B₁, B₂, B₃, B₄, B₅]             │
│    Compare: CTR(group A) vs CTR(group B)            │
│                                                     │
│  Interleaving:                                      │
│    User 1 sees:  [A₁, B₁, A₂, B₂, A₃]  (merged)   │
│    Count clicks on A-originated vs B-originated items│
│    Much more sensitive — same user, direct comparison│
└─────────────────────────────────────────────────────┘
"""
```

### 6.2 Team Draft Interleaving

```python
def team_draft_interleaving(ranking_a, ranking_b, k=10):
    """
    Team Draft Interleaving algorithm.

    Why Team Draft: Like picking teams in sports — models take turns choosing
    their top-ranked item. This ensures balanced representation from both
    models while preserving relative ordering.

    Args:
        ranking_a: Ordered list of items from model A
        ranking_b: Ordered list of items from model B
        k: Number of items in the interleaved list
    """
    interleaved = []
    team_a = set()  # Items attributed to model A
    team_b = set()  # Items attributed to model B
    idx_a, idx_b = 0, 0

    for position in range(k):
        # Alternate which model picks first (with tie-breaking by team size)
        a_picks = len(team_a) <= len(team_b)

        if a_picks:
            # Model A picks its highest-ranked item not yet in the list
            while idx_a < len(ranking_a) and ranking_a[idx_a] in interleaved:
                idx_a += 1
            if idx_a < len(ranking_a):
                item = ranking_a[idx_a]
                interleaved.append(item)
                team_a.add(item)
                idx_a += 1
        else:
            while idx_b < len(ranking_b) and ranking_b[idx_b] in interleaved:
                idx_b += 1
            if idx_b < len(ranking_b):
                item = ranking_b[idx_b]
                interleaved.append(item)
                team_b.add(item)
                idx_b += 1

    return interleaved, team_a, team_b

# Example
ranking_a = ['item_1', 'item_3', 'item_5', 'item_7', 'item_9']
ranking_b = ['item_2', 'item_1', 'item_4', 'item_3', 'item_6']

interleaved, team_a, team_b = team_draft_interleaving(ranking_a, ranking_b, k=6)
print(f"Model A ranking: {ranking_a}")
print(f"Model B ranking: {ranking_b}")
print(f"Interleaved:     {interleaved}")
print(f"Team A items:    {team_a}")
print(f"Team B items:    {team_b}")
```

---

## 7. Guardrail Metrics

### 7.1 What Are Guardrail Metrics?

```python
"""
Primary metric: The ONE metric you're optimizing for.
  Example: Revenue per user

Guardrail metrics: Metrics that must NOT degrade.
  They protect against unintended side effects.

┌────────────────────────┬────────────────────────────────────┐
│ Category               │ Example Guardrails                  │
├────────────────────────┼────────────────────────────────────┤
│ User experience        │ Page load time, error rate,         │
│                        │ crash rate, bounce rate              │
├────────────────────────┼────────────────────────────────────┤
│ Business health        │ Revenue, transactions per user,     │
│                        │ customer support contacts            │
├────────────────────────┼────────────────────────────────────┤
│ Engineering health     │ Server latency (p99), CPU usage,    │
│                        │ memory consumption                   │
├────────────────────────┼────────────────────────────────────┤
│ Fairness               │ Outcome equality across protected   │
│                        │ groups (gender, age, region)         │
└────────────────────────┴────────────────────────────────────┘

Decision rule:
  IF primary metric improves AND all guardrails pass → SHIP
  IF primary metric improves BUT guardrail fails → INVESTIGATE
  IF primary metric doesn't improve → DON'T SHIP
"""
```

### 7.2 Implementing Guardrails

```python
def evaluate_experiment(primary_result, guardrail_results,
                        primary_alpha=0.05, guardrail_threshold=0.01):
    """
    Evaluate A/B test with primary metric and guardrails.

    Why separate thresholds: The primary metric uses standard α=0.05.
    Guardrails use a stricter threshold for degradation (e.g., 0.01)
    because we want to be VERY sure there's no harm before ignoring
    a guardrail regression.
    """
    # Primary metric: is it significantly better?
    primary_pass = (primary_result['p_value'] < primary_alpha and
                    primary_result['lift'] > 0)

    # Guardrails: is any significantly WORSE?
    guardrail_failures = []
    for name, result in guardrail_results.items():
        if result['p_value'] < guardrail_threshold and result['lift'] < 0:
            guardrail_failures.append((name, result['lift']))

    print("=" * 50)
    print("EXPERIMENT DECISION")
    print("=" * 50)
    print(f"\nPrimary metric: {'PASS ✓' if primary_pass else 'FAIL ✗'}")
    print(f"  Lift: {primary_result['lift']:+.2f}%, p={primary_result['p_value']:.4f}")

    print(f"\nGuardrails: {len(guardrail_failures)} failures")
    for name, lift in guardrail_failures:
        print(f"  ✗ {name}: {lift:+.2f}%")

    if primary_pass and not guardrail_failures:
        print("\n→ DECISION: SHIP IT")
    elif primary_pass and guardrail_failures:
        print("\n→ DECISION: INVESTIGATE guardrail failures before shipping")
    else:
        print("\n→ DECISION: DO NOT SHIP")

# Example
primary = {"lift": 3.5, "p_value": 0.008}
guardrails = {
    "latency_p99":    {"lift": -0.5, "p_value": 0.45},    # Not significant
    "error_rate":     {"lift": 0.1,  "p_value": 0.82},    # Not significant
    "revenue":        {"lift": 1.2,  "p_value": 0.12},    # Positive but not sig
}
evaluate_experiment(primary, guardrails)
```

---

## 8. Practical A/B Testing Workflow

### 8.1 End-to-End Checklist

```python
"""
Phase 1: DESIGN
  □ Define primary metric (one metric, clearly measurable)
  □ Define guardrail metrics (2-5 metrics that must not degrade)
  □ State hypothesis (H₀ and H₁)
  □ Calculate required sample size (power analysis)
  □ Choose randomization unit (user_id, session_id, etc.)
  □ Determine test duration (≥ 1 weekly cycle)
  □ Document in experiment brief (share with stakeholders)

Phase 2: IMPLEMENT
  □ Implement feature flag / traffic routing
  □ Verify randomization is uniform (chi-square test on assignment)
  □ Log all necessary metrics (before running the test)
  □ Set up monitoring dashboard

Phase 3: RUN
  □ Start with small percentage (5-10%) → ramp to full
  □ Monitor guardrails daily (automated alerts)
  □ Do NOT peek at primary metric p-values (or use sequential testing)
  □ Run for pre-committed duration

Phase 4: ANALYZE
  □ Compute primary metric: effect size, CI, p-value
  □ Check guardrails: any significant degradation?
  □ Segment analysis: does the effect hold across segments?
  □ Check for novelty effects: first-week vs second-week trends
  □ Apply multiple comparison correction if testing multiple metrics

Phase 5: DECIDE
  □ Primary positive + guardrails pass → Ship
  □ Primary positive + guardrail fails → Investigate
  □ Primary neutral/negative → Don't ship
  □ Document decision and learnings
"""
```

### 8.2 Common Pitfalls

```python
"""
Pitfall                          Fix
──────────────────────────────────────────────────────────────────────
1. Peeking at results early      Use sequential testing or pre-commit
                                 to a fixed analysis date

2. Stopping at first             Run for the full pre-committed
   significance                  duration

3. Wrong randomization unit      User-level for personalization,
                                 session-level for search

4. Network effects ignored       Use cluster-based randomization
                                 (geo, time buckets)

5. Survivorship bias             Include ALL users in analysis, not
                                 just those who completed actions

6. Seasonal confounds            Run across full business cycles;
                                 use year-over-year comparisons

7. Underpowered test             Always do power analysis first;
                                 better to run longer than to guess

8. Too many metrics              One primary metric. Everything
                                 else is a guardrail or secondary.
"""
```

---

## Exercises

### Exercise 1: Power Analysis and Test Design

```python
"""
A recommendation model has a 3.2% click-through rate (baseline).
You want to detect a 5% relative improvement (3.2% → 3.36%).

1. Calculate the required sample size per group (α=0.05, power=0.80)
2. If your site gets 5,000 users/day, how many days do you need?
3. Recalculate for: (a) 10% relative improvement, (b) power=0.90
4. Plot: required sample size vs. minimum detectable effect (MDE)
   for MDE from 1% to 20% relative improvement
5. Discussion: What if you can only run the test for 2 weeks?
   What is the smallest effect you can detect?
"""
```

### Exercise 2: A/B Test Analysis

```python
"""
You ran an A/B test for 14 days. Results:

  Control:   n=25,000, conversions=1,200, avg_revenue=$12.50 (std=$45.00)
  Treatment: n=25,000, conversions=1,350, avg_revenue=$13.10 (std=$48.00)

1. Perform a two-proportion z-test on conversion rate
2. Perform a Welch's t-test on revenue per user
3. Calculate 95% confidence intervals for both
4. The treatment also increased p99 latency by 15ms (from 85ms to 100ms).
   Is this a guardrail violation? How would you decide?
5. Should you ship the new model? Justify your decision.
"""
```

### Exercise 3: Multi-Armed Bandit Comparison

```python
"""
You have 5 model variants with true CTRs: [0.04, 0.042, 0.045, 0.048, 0.05].
(You don't know these — the bandit must discover them.)

1. Implement and run for 50,000 rounds:
   a. Epsilon-greedy (ε=0.1)
   b. UCB1
   c. Thompson Sampling
   d. Random (baseline — choose uniformly at random)
2. For each algorithm, track:
   - Cumulative reward over time
   - Fraction of pulls allocated to each arm
   - Cumulative regret (vs. always choosing the best arm)
3. Plot cumulative regret curves for all 4 algorithms
4. Which algorithm converges fastest to the best arm?
"""
```

### Exercise 4: Sequential Testing

```python
"""
1. Implement an SPRT-based sequential A/B test
2. Simulate 1,000 experiments where:
   a. H₀ is true (both arms have 5% CTR)
   b. H₁ is true (control=5%, treatment=5.5%)
3. For each experiment, record:
   - Sample size at stopping
   - Decision (reject H₀ or accept H₀)
4. Calculate:
   - False positive rate (should be ≤ α=0.05)
   - True positive rate / power (should be ≥ 1-β=0.80)
   - Average sample size (compare to fixed-sample test)
5. Bonus: Compare SPRT to a naive approach that checks p-value daily
"""
```

---

## 9. Summary

### Key Takeaways

| Concept | Description |
|---------|-------------|
| **Offline-online gap** | Offline metrics (AUC, F1) don't always predict business outcomes — A/B testing bridges the gap |
| **Power analysis** | Calculate sample size BEFORE running the test; underpowered tests waste time |
| **Multiple comparisons** | Testing many metrics inflates false positives — use Bonferroni or BH correction |
| **Sequential testing** | SPRT allows continuous monitoring without inflating false positive rates |
| **Multi-armed bandits** | Adaptive alternatives to fixed A/B splits; minimize regret during experimentation |
| **Thompson Sampling** | Bayesian bandit — often best empirical performance, naturally balances explore/exploit |
| **Interleaving** | 100x more sensitive than A/B for ranking/recommendation — shows both models to same user |
| **Guardrail metrics** | Protect business-critical KPIs; ship only when primary improves AND guardrails pass |

### Connections to Other Lessons

- **L04 (Model Evaluation)**: Offline metrics are necessary but not sufficient — A/B testing is the online complement
- **L16 (Model Explainability)**: Explaining *why* a model is better helps interpret A/B test results
- **L22 (Production ML Serving)**: A/B testing validates that production-optimized models still deliver business value
- **Data_Science L08 (Hypothesis Testing)**: Foundational statistics used in A/B test analysis
- **MLOps L08-09 (Model Serving)**: Infrastructure for routing traffic between model versions
- **MLOps L13 (CI/CD for ML)**: Automated deployment pipelines that integrate A/B testing gates
