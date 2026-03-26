"""
Bayesian Inference

Demonstrates:
1. Beta-Binomial Conjugate Model
2. Normal-Normal Conjugate Model
3. MAP Estimation
4. Posterior Predictive Distribution

Theory:
- Posterior ∝ Likelihood × Prior: π(θ|x) ∝ f(x|θ)π(θ)
- Conjugate priors: posterior in same family as prior
- MAP: θ̂_MAP = argmax π(θ|x)
- Predictive: p(x_new|x) = ∫ f(x_new|θ)π(θ|x)dθ

Adapted from Probability and Statistics Lesson 15.
"""

import math
import random


# ─────────────────────────────────────────────────
# 1. BETA-BINOMIAL CONJUGATE MODEL
# ─────────────────────────────────────────────────

def beta_posterior(prior_a: float, prior_b: float,
                   successes: int, failures: int) -> tuple[float, float]:
    """Beta-Binomial conjugate update: Beta(a+s, b+f)."""
    return prior_a + successes, prior_b + failures


def beta_mean(a: float, b: float) -> float:
    return a / (a + b)


def beta_mode(a: float, b: float) -> float:
    """MAP estimate (mode of Beta)."""
    if a > 1 and b > 1:
        return (a - 1) / (a + b - 2)
    return float('nan')


def beta_credible_interval(a: float, b: float,
                            n_samples: int = 50000,
                            level: float = 0.95) -> tuple:
    """Equal-tailed credible interval via simulation."""
    # Generate Beta samples using Gamma ratio trick
    samples = []
    for _ in range(n_samples):
        x = sum(-math.log(random.random()) for _ in range(int(a)))
        y = sum(-math.log(random.random()) for _ in range(int(b)))
        if x + y > 0:
            samples.append(x / (x + y))

    if not samples:
        return (0, 1)
    samples.sort()
    alpha = 1 - level
    lo = samples[int(len(samples) * alpha / 2)]
    hi = samples[int(len(samples) * (1 - alpha / 2))]
    return lo, hi


def demo_beta_binomial():
    print("=" * 60)
    print("  Beta-Binomial Conjugate Model")
    print("=" * 60)

    # Coin flipping with sequential updating
    a, b = 2, 2  # weakly informative prior
    data = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1,
            1, 0, 1, 1, 1]  # 15 heads, 5 tails

    print(f"\n  Prior: Beta({a}, {b}), E[p] = {beta_mean(a, b):.3f}")
    print(f"\n  Sequential Bayesian updating:")
    print(f"  {'Flip':>6} {'Data':>6} {'a':>6} {'b':>6} {'E[p]':>8} {'MAP':>8}")
    print(f"  {'─' * 42}")

    for i, x in enumerate(data, 1):
        if x == 1:
            a += 1
        else:
            b += 1
        if i in [1, 2, 5, 10, 15, 20]:
            map_est = beta_mode(a, b) if a > 1 and b > 1 else float('nan')
            print(f"  {i:>6} {'H' if x else 'T':>6} {a:>6.0f} {b:>6.0f} "
                  f"{beta_mean(a, b):>8.3f} {map_est:>8.3f}")

    print(f"\n  Final: Beta({a:.0f}, {b:.0f})")
    print(f"  Posterior mean: {beta_mean(a, b):.4f}")
    print(f"  MLE: {15/20:.4f}")

    # Prior sensitivity
    print(f"\n  Prior sensitivity (same data: 15H, 5T):")
    priors = [(1, 1, "Uniform"), (2, 2, "Weakly informative"),
              (10, 10, "Strong prior at 0.5"), (1, 5, "Prior favors tails")]
    for pa, pb, desc in priors:
        post_a, post_b = beta_posterior(pa, pb, 15, 5)
        print(f"  Beta({pa},{pb}) {desc:<25} → E[p]={beta_mean(post_a, post_b):.4f}")


# ─────────────────────────────────────────────────
# 2. NORMAL-NORMAL CONJUGATE
# ─────────────────────────────────────────────────

def normal_normal_update(prior_mu: float, prior_tau: float,
                          data: list[float],
                          sigma2: float) -> tuple[float, float]:
    """Normal-Normal conjugate (known σ²).
    Prior: μ ~ N(μ₀, 1/τ₀)
    τ₀ = precision of prior = 1/σ²₀.
    Posterior: μ|x ~ N(μ_post, 1/τ_post)
    """
    n = len(data)
    data_precision = n / sigma2
    tau_post = prior_tau + data_precision
    mu_post = (prior_tau * prior_mu + data_precision * sum(data) / n * n / sigma2 * sigma2 / n) / tau_post
    # Simplify:
    mu_post = (prior_tau * prior_mu + sum(data) / sigma2) / tau_post
    return mu_post, tau_post


def demo_normal_normal():
    print("\n" + "=" * 60)
    print("  Normal-Normal Conjugate Model")
    print("=" * 60)

    # Known σ² = 4, estimate μ
    sigma2 = 4.0
    prior_mu = 0.0
    prior_sigma2 = 100.0  # vague prior
    prior_tau = 1 / prior_sigma2

    random.seed(42)
    true_mu = 5.0
    data = [random.gauss(true_mu, math.sqrt(sigma2)) for _ in range(20)]
    x_bar = sum(data) / len(data)

    mu_post, tau_post = normal_normal_update(prior_mu, prior_tau, data, sigma2)
    sigma2_post = 1 / tau_post

    print(f"\n  Known σ² = {sigma2}, true μ = {true_mu}")
    print(f"  Prior: N({prior_mu}, {prior_sigma2})")
    print(f"  Data: n={len(data)}, x̄={x_bar:.3f}")
    print(f"  Posterior: N({mu_post:.3f}, {sigma2_post:.4f})")
    print(f"  Posterior SD: {math.sqrt(sigma2_post):.4f}")
    print(f"\n  MLE (x̄) = {x_bar:.3f}")
    print(f"  Posterior mean = {mu_post:.3f}")
    print(f"  (Shrunk toward prior — less so with vague prior)")

    # Effect of sample size
    print(f"\n  Posterior mean vs n (prior at 0, vague):")
    for n in [1, 5, 10, 50, 200]:
        sample = [random.gauss(true_mu, math.sqrt(sigma2)) for _ in range(n)]
        mp, tp = normal_normal_update(prior_mu, prior_tau, sample, sigma2)
        weight_data = (n / sigma2) / tp
        print(f"  n={n:>3}: μ_post={mp:.3f}, data weight={weight_data:.3f}")


# ─────────────────────────────────────────────────
# 3. MAP ESTIMATION
# ─────────────────────────────────────────────────

def demo_map():
    print("\n" + "=" * 60)
    print("  MAP vs MLE vs Posterior Mean")
    print("=" * 60)

    print("""
  MLE:  θ̂_MLE = argmax f(x|θ)        — data only
  MAP:  θ̂_MAP = argmax f(x|θ)π(θ)    — data + prior
  Post: E[θ|x] = ∫ θ π(θ|x) dθ       — full posterior
""")

    # Beta-Binomial example
    print(f"  Coin flipping (3 heads, 0 tails):")
    mle = 3 / 3
    for a, b, name in [(1, 1, "Uniform"), (2, 2, "Beta(2,2)"), (5, 5, "Beta(5,5)")]:
        pa, pb = a + 3, b + 0
        post_mean = beta_mean(pa, pb)
        map_est = beta_mode(pa, pb) if pa > 1 and pb > 1 else mle
        print(f"  Prior {name:<12}: MLE={mle:.3f}, MAP={map_est:.3f}, "
              f"PostMean={post_mean:.3f}")

    print(f"\n  With more data (30 heads, 0 tails):")
    for a, b, name in [(1, 1, "Uniform"), (2, 2, "Beta(2,2)"), (5, 5, "Beta(5,5)")]:
        pa, pb = a + 30, b + 0
        post_mean = beta_mean(pa, pb)
        map_est = beta_mode(pa, pb)
        print(f"  Prior {name:<12}: MAP={map_est:.3f}, PostMean={post_mean:.3f}")
    print(f"  → Prior influence diminishes with more data")


# ─────────────────────────────────────────────────
# 4. POSTERIOR PREDICTIVE
# ─────────────────────────────────────────────────

def demo_predictive():
    print("\n" + "=" * 60)
    print("  Posterior Predictive Distribution")
    print("=" * 60)

    # Beta-Binomial: predict next coin flip
    # After 15H, 5T with Beta(2,2) prior → Beta(17, 7)
    a, b = 17, 7
    p_head = a / (a + b)  # = E[p|data] = posterior mean
    print(f"\n  After 15H/5T, Beta(17,7) posterior:")
    print(f"  P(next flip = H | data) = E[p | data] = {p_head:.4f}")

    # Predict k heads in next 10 flips (Beta-Binomial predictive)
    print(f"\n  P(k heads in next 10 | data):")
    n_future = 10
    for k in range(n_future + 1):
        # Beta-Binomial PMF
        log_p = (math.lgamma(n_future + 1) - math.lgamma(k + 1) - math.lgamma(n_future - k + 1)
                 + math.lgamma(k + a) + math.lgamma(n_future - k + b)
                 - math.lgamma(n_future + a + b)
                 + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))
        p = math.exp(log_p)
        bar = "█" * int(p * 60)
        print(f"  k={k:>2}: {p:.4f} {bar}")

    print(f"\n  Predictive accounts for parameter uncertainty")
    print(f"  (wider than plug-in Binomial with p̂ = {p_head:.3f})")


if __name__ == "__main__":
    demo_beta_binomial()
    demo_normal_normal()
    demo_map()
    demo_predictive()
