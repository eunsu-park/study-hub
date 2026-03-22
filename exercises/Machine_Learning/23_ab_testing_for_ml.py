"""
Exercises: A/B Testing for ML Models
======================================

Practice problems for designing, running, and analyzing A/B tests
for ML model comparisons.

Requirements:
    pip install numpy scipy
"""

import numpy as np
from scipy import stats


# ============================================================
# Exercise 1: Power Analysis and Test Design
# ============================================================
"""
A recommendation model has a 3.2% click-through rate (baseline).
You want to detect a 5% relative improvement (3.2% → 3.36%).

1. Implement a sample size calculator for two-proportion z-test
2. Calculate required sample size per group (α=0.05, power=0.80)
3. If your site gets 5,000 users/day, how many days do you need?
4. Recalculate for: (a) 10% relative improvement, (b) power=0.90
5. Print a table: MDE from 1% to 20% vs required sample size
6. What is the smallest effect detectable in a 2-week test window?
"""

# Your code here:


# ============================================================
# Exercise 2: A/B Test Analysis
# ============================================================
"""
You ran an A/B test for 14 days. Results:

  Control:   n=25,000, conversions=1,200, avg_revenue=$12.50 (std=$45.00)
  Treatment: n=25,000, conversions=1,350, avg_revenue=$13.10 (std=$48.00)

1. Perform a two-proportion z-test on conversion rate
   - Calculate: z-statistic, p-value, 95% confidence interval
2. Perform Welch's t-test on revenue per user
   - Simulate revenue data using np.random.exponential matching
     the given mean and std, then run the test
3. Calculate relative lift and confidence intervals for both metrics
4. The treatment increased p99 latency by 15ms (85ms → 100ms).
   Define this as a guardrail and evaluate it.
5. Should you ship the new model? Write your decision with justification.
"""

# Your code here:


# ============================================================
# Exercise 3: Multi-Armed Bandit Comparison
# ============================================================
"""
You have 5 model variants with true CTRs: [0.04, 0.042, 0.045, 0.048, 0.05].
(You don't know these — the bandit must discover them.)

1. Implement three bandit algorithms:
   a. Epsilon-greedy (ε=0.1)
   b. UCB1
   c. Thompson Sampling
2. Run each for 50,000 rounds (use the same random seed for fairness)
3. Track for each algorithm:
   - Total cumulative reward
   - Cumulative regret (vs always choosing the best arm)
   - Number of pulls per arm
4. Print a comparison table of all three algorithms
5. Which algorithm achieves the lowest regret? The highest reward?
"""

# Your code here:


# ============================================================
# Exercise 4: Sequential Testing (SPRT)
# ============================================================
"""
1. Implement an SPRT-based sequential A/B test:
   - H₀: p = 0.05, H₁: p = 0.055
   - α = 0.05, β = 0.20
2. Run a single test with simulated data (treatment rate = 0.055)
   and print the sample size at stopping and the decision
3. Monte Carlo validation (500 simulations):
   a. Under H₀ (both arms at 5%): measure false positive rate
   b. Under H₁ (control=5%, treatment=5.5%): measure power
   c. Record average sample size at stopping under each scenario
4. Compare SPRT sample efficiency vs fixed-sample test
5. Print: FP rate, power, avg samples (SPRT vs fixed)
"""

# Your code here:


# ============================================================
# Exercise 5: Guardrail Evaluation Framework
# ============================================================
"""
Build a complete experiment evaluation framework:

1. Define a function evaluate_experiment(primary, guardrails) that:
   - Checks if primary metric is significantly positive
   - Checks if any guardrail is significantly negative
   - Returns one of: "SHIP", "INVESTIGATE", "DO NOT SHIP"

2. Test with three scenarios:
   a. Primary +5% (p=0.001), all guardrails pass → should SHIP
   b. Primary +3% (p=0.02), latency guardrail fails (-20%, p=0.003) → INVESTIGATE
   c. Primary +0.5% (p=0.35), no guardrail issues → DO NOT SHIP

3. Add a summary report that includes:
   - Primary metric: lift, CI, p-value
   - Each guardrail: status (pass/fail), lift, p-value
   - Final decision with reasoning
"""

# Your code here:
