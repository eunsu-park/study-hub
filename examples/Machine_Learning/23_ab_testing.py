"""
A/B Testing for ML Models
==========================

Demonstrates statistical methodology for online experimentation:
1. Power analysis and sample size calculation
2. Frequentist hypothesis testing (z-test, Welch's t-test)
3. Multiple comparison correction (Bonferroni, Benjamini-Hochberg)
4. Sequential testing (SPRT)
5. Multi-armed bandits (Epsilon-Greedy, UCB1, Thompson Sampling)
6. Interleaving experiments (Team Draft)
7. Guardrail metric evaluation

Requirements:
    pip install numpy scipy matplotlib
"""

import numpy as np
from scipy import stats


# ============================================================
# 1. Power Analysis — Sample Size Calculation
# ============================================================

print("=" * 60)
print("1. Power Analysis — Sample Size Calculation")
print("=" * 60)


def required_sample_size(baseline_rate, mde_relative,
                         alpha=0.05, power=0.80):
    """
    Calculate minimum sample size per group for a two-proportion z-test.

    Args:
        baseline_rate: Current conversion rate (e.g., 0.05 for 5%)
        mde_relative: Minimum detectable effect as relative change
                      (e.g., 0.10 for 10% relative lift)
        alpha: Significance level (Type I error rate)
        power: Statistical power (1 - Type II error rate)
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde_relative)

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    p_pool = (p1 + p2) / 2
    se_null = np.sqrt(2 * p_pool * (1 - p_pool))
    se_alt = np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))

    n = ((z_alpha * se_null + z_beta * se_alt) / (p2 - p1)) ** 2
    return int(np.ceil(n))


# Demonstrate for different scenarios
print("\nBaseline CTR = 5%, α=0.05, power=0.80:")
print(f"{'MDE (relative)':>15s} {'p2':>8s} {'n per group':>12s} {'Total':>10s}")
print("-" * 50)

for mde in [0.05, 0.10, 0.15, 0.20, 0.30]:
    n = required_sample_size(0.05, mde)
    p2 = 0.05 * (1 + mde)
    print(f"{mde:>14.0%} {p2:>8.3f} {n:>12,} {2*n:>10,}")

# Duration estimation
daily_traffic = 10000
n_needed = required_sample_size(0.05, 0.10)
days = int(np.ceil(2 * n_needed / daily_traffic))
print(f"\nWith {daily_traffic:,} users/day and 10% MDE: {days} days needed")


# ============================================================
# 2. Hypothesis Testing — Proportion Test
# ============================================================

print("\n" + "=" * 60)
print("2. Hypothesis Testing — Two-Proportion Z-Test")
print("=" * 60)


def ab_test_proportions(n_c, conv_c, n_t, conv_t, alpha=0.05):
    """Two-proportion z-test for conversion rate comparison."""
    p_c = conv_c / n_c
    p_t = conv_t / n_t
    p_pool = (conv_c + conv_t) / (n_c + n_t)

    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_c + 1/n_t))
    z = (p_t - p_c) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    se_diff = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = ((p_t - p_c) - z_crit * se_diff,
          (p_t - p_c) + z_crit * se_diff)

    lift = (p_t - p_c) / p_c * 100
    significant = p_value < alpha

    print(f"  Control:   {p_c:.4f} ({conv_c:,}/{n_c:,})")
    print(f"  Treatment: {p_t:.4f} ({conv_t:,}/{n_t:,})")
    print(f"  Lift:      {lift:+.2f}%")
    print(f"  95% CI:    [{ci[0]:.5f}, {ci[1]:.5f}]")
    print(f"  p-value:   {p_value:.4f}")
    print(f"  Result:    {'✓ Significant' if significant else '✗ Not significant'}")

    return {"p_c": p_c, "p_t": p_t, "lift": lift, "p_value": p_value,
            "ci": ci, "significant": significant}


# Scenario: New recommendation model
print("\nScenario: New recommendation model A/B test")
result = ab_test_proportions(
    n_c=50000, conv_c=2500,   # Control: 5.00% CTR
    n_t=50000, conv_t=2650,   # Treatment: 5.30% CTR
)


# ============================================================
# 3. Hypothesis Testing — Continuous Metric (Revenue)
# ============================================================

print("\n" + "=" * 60)
print("3. Welch's t-Test — Revenue per User")
print("=" * 60)

np.random.seed(42)
control_revenue = np.random.exponential(scale=10, size=10000)
treatment_revenue = np.random.exponential(scale=10.5, size=10000)

t_stat, p_value = stats.ttest_ind(
    control_revenue, treatment_revenue, equal_var=False
)

mean_c = np.mean(control_revenue)
mean_t = np.mean(treatment_revenue)
lift = (mean_t - mean_c) / mean_c * 100

print(f"  Control mean:   ${mean_c:.2f} (std=${np.std(control_revenue):.2f})")
print(f"  Treatment mean: ${mean_t:.2f} (std=${np.std(treatment_revenue):.2f})")
print(f"  Lift:           {lift:+.2f}%")
print(f"  p-value:        {p_value:.4f}")
print(f"  Result:         {'✓ Significant' if p_value < 0.05 else '✗ Not significant'}")


# ============================================================
# 4. Multiple Comparison Correction
# ============================================================

print("\n" + "=" * 60)
print("4. Multiple Comparison Correction")
print("=" * 60)

p_values = np.array([0.001, 0.013, 0.029, 0.041, 0.15, 0.88])
n_tests = len(p_values)
alpha = 0.05

# Bonferroni
bonf_threshold = alpha / n_tests

# Benjamini-Hochberg
sorted_idx = np.argsort(p_values)
bh_significant = np.zeros(n_tests, dtype=bool)
for rank, idx in enumerate(sorted_idx):
    bh_threshold = (rank + 1) / n_tests * alpha
    if p_values[idx] <= bh_threshold:
        bh_significant[idx] = True

print(f"\n{'Metric':<10s} {'p-value':>8s} {'Raw(α=.05)':>11s} "
      f"{'Bonferroni':>11s} {'BH (FDR)':>11s}")
print("-" * 55)
for i in range(n_tests):
    raw = "SIG" if p_values[i] < alpha else ""
    bonf = "SIG" if p_values[i] < bonf_threshold else ""
    bh = "SIG" if bh_significant[i] else ""
    print(f"metric_{i+1:<3d} {p_values[i]:>8.4f} {raw:>11s} {bonf:>11s} {bh:>11s}")

print(f"\nBonferroni threshold: {bonf_threshold:.4f}")
print(f"Raw significant: {sum(p_values < alpha)}, "
      f"Bonferroni: {sum(p_values < bonf_threshold)}, "
      f"BH: {sum(bh_significant)}")


# ============================================================
# 5. Sequential Testing (SPRT)
# ============================================================

print("\n" + "=" * 60)
print("5. Sequential Probability Ratio Test (SPRT)")
print("=" * 60)


def run_sprt(control, treatment, p0=0.05, p1=0.055, alpha=0.05, beta=0.20):
    """Run SPRT and return (sample_at_stop, decision)."""
    A = np.log((1 - beta) / alpha)
    B = np.log(beta / (1 - alpha))
    llr = 0
    n = min(len(control), len(treatment))

    for i in range(n):
        x = treatment[i]
        if x == 1:
            llr += np.log(p1 / p0)
        else:
            llr += np.log((1 - p1) / (1 - p0))

        if llr >= A:
            return i + 1, "reject_H0"
        elif llr <= B:
            return i + 1, "accept_H0"

    return n, "inconclusive"


# Single demonstration
np.random.seed(42)
ctrl = np.random.binomial(1, 0.05, size=20000)
trt = np.random.binomial(1, 0.055, size=20000)
n_stop, decision = run_sprt(ctrl, trt)
fixed_n = required_sample_size(0.05, 0.10)

print(f"\nSingle run:")
print(f"  SPRT stopped at sample:  {n_stop:,}")
print(f"  Decision:                {decision}")
print(f"  Fixed-sample equivalent: {fixed_n:,}")
print(f"  Savings:                 {(1 - n_stop / fixed_n) * 100:.1f}% fewer samples")

# Monte Carlo validation: check error rates
print("\nMonte Carlo validation (1000 simulations)...")

n_sims = 1000
# Under H₀ (no difference)
h0_decisions = []
for _ in range(n_sims):
    c = np.random.binomial(1, 0.05, size=20000)
    t = np.random.binomial(1, 0.05, size=20000)  # Same rate
    _, dec = run_sprt(c, t)
    h0_decisions.append(dec)

fp_rate = sum(1 for d in h0_decisions if d == "reject_H0") / n_sims

# Under H₁ (treatment is better)
h1_decisions = []
h1_samples = []
for _ in range(n_sims):
    c = np.random.binomial(1, 0.05, size=20000)
    t = np.random.binomial(1, 0.055, size=20000)
    n_s, dec = run_sprt(c, t)
    h1_decisions.append(dec)
    h1_samples.append(n_s)

power = sum(1 for d in h1_decisions if d == "reject_H0") / n_sims
avg_samples = np.mean(h1_samples)

print(f"  False positive rate: {fp_rate:.3f} (target ≤ 0.05)")
print(f"  Power:               {power:.3f} (target ≥ 0.80)")
print(f"  Avg samples (H₁):   {avg_samples:,.0f} (vs fixed: {fixed_n:,})")


# ============================================================
# 6. Multi-Armed Bandits
# ============================================================

print("\n" + "=" * 60)
print("6. Multi-Armed Bandits Comparison")
print("=" * 60)

true_rates = [0.04, 0.042, 0.045, 0.048, 0.05]
n_arms = len(true_rates)
n_rounds = 50000
best_rate = max(true_rates)


class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.estimates = np.zeros(n_arms)

    def select(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.estimates)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.estimates[arm] += (reward - self.estimates[arm]) / self.counts[arm]


class UCB1:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.estimates = np.zeros(n_arms)
        self.total = 0

    def select(self):
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i
        ucb = self.estimates + np.sqrt(2 * np.log(self.total) / self.counts)
        return np.argmax(ucb)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.total += 1
        self.estimates[arm] += (reward - self.estimates[arm]) / self.counts[arm]


class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select(self):
        samples = [np.random.beta(self.alpha[i], self.beta[i])
                   for i in range(self.n_arms)]
        return np.argmax(samples)

    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    @property
    def counts(self):
        return self.alpha + self.beta - 2


algorithms = {
    "ε-Greedy(0.1)":    EpsilonGreedy(n_arms, epsilon=0.1),
    "UCB1":             UCB1(n_arms),
    "Thompson":         ThompsonSampling(n_arms),
}

np.random.seed(42)

print(f"\nTrue rates: {true_rates}")
print(f"Best arm: {np.argmax(true_rates)} (rate={best_rate})")
print(f"Rounds: {n_rounds:,}\n")

for alg_name, alg in algorithms.items():
    total_reward = 0
    cumulative_regret = 0

    for t in range(n_rounds):
        arm = alg.select()
        reward = np.random.binomial(1, true_rates[arm])
        alg.update(arm, reward)
        total_reward += reward
        cumulative_regret += best_rate - true_rates[arm]

    print(f"{alg_name}:")
    print(f"  Total reward:     {total_reward:,}")
    print(f"  Cumulative regret: {cumulative_regret:.1f}")
    counts = alg.counts
    best_arm_pct = counts[np.argmax(true_rates)] / sum(counts) * 100
    print(f"  Best arm pulls:   {best_arm_pct:.1f}%")
    print(f"  Arm distribution: {[int(c) for c in counts]}")
    print()


# ============================================================
# 7. Team Draft Interleaving
# ============================================================

print("=" * 60)
print("7. Team Draft Interleaving")
print("=" * 60)


def team_draft_interleave(ranking_a, ranking_b, k=10):
    """Team Draft Interleaving for ranking model comparison."""
    interleaved = []
    team_a, team_b = set(), set()
    idx_a, idx_b = 0, 0

    for pos in range(k):
        a_picks = len(team_a) <= len(team_b)

        if a_picks:
            while idx_a < len(ranking_a) and ranking_a[idx_a] in interleaved:
                idx_a += 1
            if idx_a < len(ranking_a):
                interleaved.append(ranking_a[idx_a])
                team_a.add(ranking_a[idx_a])
                idx_a += 1
        else:
            while idx_b < len(ranking_b) and ranking_b[idx_b] in interleaved:
                idx_b += 1
            if idx_b < len(ranking_b):
                interleaved.append(ranking_b[idx_b])
                team_b.add(ranking_b[idx_b])
                idx_b += 1

    return interleaved, team_a, team_b


ranking_a = ['item_1', 'item_3', 'item_5', 'item_7', 'item_9', 'item_11']
ranking_b = ['item_2', 'item_1', 'item_4', 'item_3', 'item_6', 'item_8']

interleaved, team_a, team_b = team_draft_interleave(ranking_a, ranking_b, k=6)
print(f"\nModel A ranking: {ranking_a[:6]}")
print(f"Model B ranking: {ranking_b[:6]}")
print(f"Interleaved:     {interleaved}")
print(f"Team A items:    {sorted(team_a)}")
print(f"Team B items:    {sorted(team_b)}")

# Simulate user clicks
clicked = {'item_1', 'item_4', 'item_5'}
a_clicks = len(clicked & team_a)
b_clicks = len(clicked & team_b)
print(f"\nUser clicked: {sorted(clicked)}")
print(f"Clicks on A items: {a_clicks}")
print(f"Clicks on B items: {b_clicks}")
print(f"Winner: Model {'A' if a_clicks > b_clicks else 'B' if b_clicks > a_clicks else 'Tie'}")


# ============================================================
# 8. Guardrail Metric Evaluation
# ============================================================

print("\n" + "=" * 60)
print("8. Guardrail Metric Evaluation")
print("=" * 60)


def evaluate_experiment(primary, guardrails, alpha=0.05, guard_alpha=0.01):
    """Evaluate A/B test with primary metric and guardrails."""
    primary_pass = primary['p_value'] < alpha and primary['lift'] > 0

    failures = []
    for name, g in guardrails.items():
        if g['p_value'] < guard_alpha and g['lift'] < 0:
            failures.append((name, g['lift'], g['p_value']))

    print(f"\n  Primary metric: {'PASS ✓' if primary_pass else 'FAIL ✗'}")
    print(f"    Lift: {primary['lift']:+.2f}%, p={primary['p_value']:.4f}")

    print(f"\n  Guardrails ({len(guardrails)}):")
    for name, g in guardrails.items():
        status = "✗ FAIL" if any(f[0] == name for f in failures) else "✓ pass"
        print(f"    {status} {name}: lift={g['lift']:+.1f}%, p={g['p_value']:.3f}")

    if primary_pass and not failures:
        decision = "SHIP IT ✓"
    elif primary_pass and failures:
        decision = "INVESTIGATE guardrail failures"
    else:
        decision = "DO NOT SHIP ✗"

    print(f"\n  → Decision: {decision}")
    return decision


# Scenario 1: Clean win
print("\nScenario 1: Clean win")
evaluate_experiment(
    primary={"lift": 3.5, "p_value": 0.008},
    guardrails={
        "latency_p99":  {"lift": -0.5, "p_value": 0.45},
        "error_rate":   {"lift": 0.1,  "p_value": 0.82},
        "revenue":      {"lift": 1.2,  "p_value": 0.12},
    }
)

# Scenario 2: Primary wins but guardrail fails
print("\nScenario 2: Guardrail violation")
evaluate_experiment(
    primary={"lift": 5.0, "p_value": 0.001},
    guardrails={
        "latency_p99":  {"lift": -15.0, "p_value": 0.003},
        "error_rate":   {"lift": 0.2,   "p_value": 0.78},
        "revenue":      {"lift": 2.5,   "p_value": 0.06},
    }
)

# Scenario 3: Not significant
print("\nScenario 3: Not significant")
evaluate_experiment(
    primary={"lift": 0.8, "p_value": 0.35},
    guardrails={
        "latency_p99": {"lift": -0.2, "p_value": 0.90},
        "error_rate":  {"lift": -0.1, "p_value": 0.85},
    }
)


print("\n✓ All A/B testing demos completed.")
