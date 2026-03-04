"""
Probability and Statistics — Bayesian Inference
Exercises covering prior selection, Beta-Binomial posterior computation,
Normal-Normal update, and MAP estimation.
"""
import math
import random
from typing import List, Tuple


# === Exercise 1: Prior Selection and Sensitivity ===
def exercise_1() -> None:
    """Explore how different prior choices affect the posterior for a
    binomial proportion, demonstrating prior sensitivity."""
    print("=== Exercise 1: Prior Selection and Sensitivity ===")

    # Data: 7 successes out of 10 trials
    k = 7
    n = 10

    priors = [
        ("Uniform (Beta(1,1))", 1.0, 1.0),
        ("Jeffreys (Beta(0.5,0.5))", 0.5, 0.5),
        ("Weakly informative (Beta(2,2))", 2.0, 2.0),
        ("Strong prior at 0.5 (Beta(10,10))", 10.0, 10.0),
        ("Strong prior at 0.3 (Beta(6,14))", 6.0, 14.0),
    ]

    mle = k / n
    print(f"\n  Data: {k} successes in {n} trials (MLE = {mle:.4f})")
    print(f"\n  {'Prior':>35} {'Posterior':>15} {'Post Mean':>10} "
          f"{'Post Mode':>10} {'Post SD':>8}")
    print(f"  {'-' * 35} {'-' * 15} {'-' * 10} {'-' * 10} {'-' * 8}")

    for name, a, b in priors:
        a_post = a + k
        b_post = b + (n - k)
        post_mean = a_post / (a_post + b_post)
        if a_post > 1 and b_post > 1:
            post_mode = (a_post - 1) / (a_post + b_post - 2)
        else:
            post_mode = float('nan')
        post_var = (a_post * b_post) / \
                   ((a_post + b_post) ** 2 * (a_post + b_post + 1))
        post_sd = math.sqrt(post_var)

        print(f"  {name:>35} Beta({a_post:.1f},{b_post:.1f})"
              f"{'':<3} {post_mean:>10.4f} {post_mode:>10.4f} {post_sd:>8.4f}")

    print(f"\n  Observations:")
    print(f"  - Uniform and Jeffreys priors give posteriors close to MLE")
    print(f"  - Strong priors shrink the posterior mean toward the prior mean")
    print(f"  - With only n={n} observations, the prior has substantial influence\n")


# === Exercise 2: Beta-Binomial Posterior with Sequential Updates ===
def exercise_2() -> None:
    """Demonstrate sequential Bayesian updating of a Beta-Binomial model
    as coin flip data arrives one observation at a time."""
    print("=== Exercise 2: Beta-Binomial Posterior (Sequential Updates) ===")

    a, b = 2.0, 2.0  # Prior: Beta(2,2)
    data = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1,
            1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]

    print(f"\n  Prior: Beta({a:.0f}, {b:.0f})")
    print(f"  Data: {len(data)} coin flips ({sum(data)} heads, "
          f"{len(data) - sum(data)} tails)")
    print(f"\n  {'Flip':>6} {'Result':>8} {'a':>6} {'b':>6} "
          f"{'E[p]':>8} {'95% CI width':>13}")
    print(f"  {'----':>6} {'------':>8} {'--':>6} {'--':>6} "
          f"{'----':>8} {'------------':>13}")

    # Print prior
    ci_width = _beta_ci_width(a, b)
    print(f"  {'Prior':>6} {'':>8} {a:>6.0f} {b:>6.0f} "
          f"{a / (a + b):>8.4f} {ci_width:>13.4f}")

    for i, x in enumerate(data, 1):
        if x == 1:
            a += 1
        else:
            b += 1
        if i in [1, 2, 5, 10, 15, 20, 25, 30]:
            ci_width = _beta_ci_width(a, b)
            print(f"  {i:>6} {'H' if x else 'T':>8} {a:>6.0f} {b:>6.0f} "
                  f"{a / (a + b):>8.4f} {ci_width:>13.4f}")

    mle = sum(data) / len(data)
    post_mean = a / (a + b)
    print(f"\n  Final posterior: Beta({a:.0f}, {b:.0f})")
    print(f"  Posterior mean = {post_mean:.4f}")
    print(f"  MLE = {mle:.4f}")
    print(f"  Posterior concentrates as data accumulates (CI shrinks).\n")


def _beta_ci_width(a: float, b: float) -> float:
    """Approximate 95% CI width for Beta(a, b) using normal approx."""
    mean = a / (a + b)
    var = (a * b) / ((a + b) ** 2 * (a + b + 1))
    sd = math.sqrt(var)
    return 2 * 1.96 * sd


# === Exercise 3: Normal-Normal Conjugate Update ===
def exercise_3() -> None:
    """Perform a Normal-Normal Bayesian update with known variance,
    comparing posterior mean to MLE and showing shrinkage."""
    print("=== Exercise 3: Normal-Normal Conjugate Update ===")

    # Setup: estimate mu with known sigma^2
    sigma_sq = 9.0  # known variance
    prior_mu = 0.0
    prior_var = 100.0  # vague prior

    random.seed(42)
    true_mu = 5.0
    n = 15
    data = [random.gauss(true_mu, math.sqrt(sigma_sq)) for _ in range(n)]
    x_bar = sum(data) / n

    # Posterior computation
    prior_prec = 1 / prior_var
    data_prec = n / sigma_sq
    post_prec = prior_prec + data_prec
    post_var = 1 / post_prec
    post_mu = (prior_prec * prior_mu + data_prec * x_bar) / post_prec
    post_sd = math.sqrt(post_var)

    # Credible interval
    ci_lo = post_mu - 1.96 * post_sd
    ci_hi = post_mu + 1.96 * post_sd

    print(f"\n  Known sigma^2 = {sigma_sq}, true mu = {true_mu}")
    print(f"  Prior: N({prior_mu}, {prior_var})")
    print(f"  Data: n={n}, x_bar={x_bar:.4f}")

    print(f"\n  Bayesian update:")
    print(f"    Prior precision  = {prior_prec:.6f}")
    print(f"    Data precision   = {data_prec:.6f}")
    print(f"    Post precision   = {post_prec:.6f}")
    print(f"    Post variance    = {post_var:.6f}")
    print(f"\n  Posterior: N({post_mu:.4f}, {post_var:.6f})")
    print(f"  Posterior mean = {post_mu:.4f}")
    print(f"  MLE (x_bar)   = {x_bar:.4f}")
    print(f"  95% Credible interval: ({ci_lo:.4f}, {ci_hi:.4f})")

    # Show effect of sample size on posterior
    print(f"\n  Posterior mean vs sample size:")
    print(f"  {'n':>5} {'x_bar':>10} {'Post mean':>10} {'Data weight':>12}")
    print(f"  {'---':>5} {'-----':>10} {'---------':>10} {'-----------':>12}")
    for sample_n in [1, 5, 10, 50, 200]:
        sample = [random.gauss(true_mu, math.sqrt(sigma_sq))
                  for _ in range(sample_n)]
        s_bar = sum(sample) / sample_n
        dp = sample_n / sigma_sq
        pp = prior_prec + dp
        pm = (prior_prec * prior_mu + dp * s_bar) / pp
        w_data = dp / pp
        print(f"  {sample_n:>5} {s_bar:>10.4f} {pm:>10.4f} {w_data:>12.4f}")
    print()


# === Exercise 4: MAP Estimation ===
def exercise_4() -> None:
    """Compare MLE, MAP, and posterior mean estimates for the Beta-Binomial
    model, showing how MAP balances data and prior information."""
    print("=== Exercise 4: MAP Estimation ===")

    print(f"\n  Definitions:")
    print(f"    MLE:  argmax f(x|theta)        -- data only")
    print(f"    MAP:  argmax f(x|theta)*pi(theta) -- data + prior")
    print(f"    Post: E[theta|x]               -- full posterior")

    # Case 1: Small sample
    print(f"\n  Case 1: Small sample (k=3 heads, n=3 flips)")
    k, n = 3, 3
    mle = k / n
    print(f"    MLE = {mle:.4f}")

    priors = [
        ("Beta(1,1)", 1.0, 1.0),
        ("Beta(2,2)", 2.0, 2.0),
        ("Beta(5,5)", 5.0, 5.0),
        ("Beta(10,10)", 10.0, 10.0),
    ]

    print(f"\n    {'Prior':>14} {'MAP':>8} {'Post Mean':>10}")
    print(f"    {'-' * 14} {'-' * 8} {'-' * 10}")
    for name, a, b in priors:
        a_post = a + k
        b_post = b + (n - k)
        post_mean = a_post / (a_post + b_post)
        if a_post > 1 and b_post > 1:
            map_est = (a_post - 1) / (a_post + b_post - 2)
        else:
            map_est = mle
        print(f"    {name:>14} {map_est:>8.4f} {post_mean:>10.4f}")

    # Case 2: Large sample
    print(f"\n  Case 2: Large sample (k=60 heads, n=100 flips)")
    k, n = 60, 100
    mle = k / n
    print(f"    MLE = {mle:.4f}")

    print(f"\n    {'Prior':>14} {'MAP':>8} {'Post Mean':>10}")
    print(f"    {'-' * 14} {'-' * 8} {'-' * 10}")
    for name, a, b in priors:
        a_post = a + k
        b_post = b + (n - k)
        post_mean = a_post / (a_post + b_post)
        if a_post > 1 and b_post > 1:
            map_est = (a_post - 1) / (a_post + b_post - 2)
        else:
            map_est = mle
        print(f"    {name:>14} {map_est:>8.4f} {post_mean:>10.4f}")

    print(f"\n  Key observations:")
    print(f"  - With small n, MLE can be extreme (1.0 from 3/3)")
    print(f"  - MAP and posterior mean shrink toward prior")
    print(f"  - With large n, all three estimates converge")
    print(f"  - MAP = MLE when prior is uniform (Beta(1,1))")
    print(f"  - Stronger priors produce more regularization\n")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
