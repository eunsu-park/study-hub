"""
Point Estimation

Demonstrates:
1. Method of Moments (MoM)
2. Maximum Likelihood Estimation (MLE)
3. Fisher Information and Cramér-Rao Bound
4. Bias, Consistency, Efficiency

Theory:
- MoM: equate sample moments to population moments
- MLE: θ̂ = argmax L(θ|x) = argmax Σ log f(xi|θ)
- Fisher info: I(θ) = -E[∂²logf/∂θ²]
- CRLB: Var(θ̂) ≥ 1/(n·I(θ))

Adapted from Probability and Statistics Lesson 12.
"""

import math
import random


# ─────────────────────────────────────────────────
# 1. METHOD OF MOMENTS
# ─────────────────────────────────────────────────

def mom_normal(data: list[float]) -> tuple[float, float]:
    """MoM for Normal: μ̂ = x̄, σ̂² = m₂."""
    n = len(data)
    m1 = sum(data) / n
    m2 = sum((x - m1)**2 for x in data) / n
    return m1, m2


def mom_gamma(data: list[float]) -> tuple[float, float]:
    """MoM for Gamma(α, β): α = m₁²/m₂c, β = m₁/m₂c."""
    n = len(data)
    m1 = sum(data) / n
    m2c = sum((x - m1)**2 for x in data) / n
    alpha = m1**2 / m2c
    beta = m1 / m2c
    return alpha, beta


def demo_mom():
    print("=" * 60)
    print("  Method of Moments Estimation")
    print("=" * 60)

    random.seed(42)

    # Normal
    true_mu, true_sigma = 5.0, 2.0
    data = [random.gauss(true_mu, true_sigma) for _ in range(100)]
    mu_hat, var_hat = mom_normal(data)
    print(f"\n  Normal(μ={true_mu}, σ={true_sigma}):")
    print(f"  MoM: μ̂={mu_hat:.4f}, σ̂²={var_hat:.4f} (true: {true_sigma**2})")

    # Gamma
    true_alpha, true_beta = 3.0, 2.0
    data = [sum(random.expovariate(true_beta) for _ in range(int(true_alpha)))
            for _ in range(200)]
    a_hat, b_hat = mom_gamma(data)
    print(f"\n  Gamma(α={true_alpha}, β={true_beta}):")
    print(f"  MoM: α̂={a_hat:.4f}, β̂={b_hat:.4f}")


# ─────────────────────────────────────────────────
# 2. MAXIMUM LIKELIHOOD ESTIMATION
# ─────────────────────────────────────────────────

def mle_normal(data: list[float]) -> tuple[float, float]:
    """MLE for Normal: μ̂ = x̄, σ̂² = (1/n)Σ(xi - x̄)²."""
    n = len(data)
    mu_hat = sum(data) / n
    sigma2_hat = sum((x - mu_hat)**2 for x in data) / n
    return mu_hat, sigma2_hat


def mle_poisson(data: list[int]) -> float:
    """MLE for Poisson: λ̂ = x̄."""
    return sum(data) / len(data)


def mle_exponential(data: list[float]) -> float:
    """MLE for Exponential: λ̂ = 1/x̄."""
    return len(data) / sum(data)


def log_likelihood_normal(data: list[float], mu: float,
                           sigma2: float) -> float:
    """Log-likelihood for Normal."""
    n = len(data)
    return -n/2 * math.log(2 * math.pi * sigma2) - sum((x - mu)**2 for x in data) / (2 * sigma2)


def demo_mle():
    print("\n" + "=" * 60)
    print("  Maximum Likelihood Estimation")
    print("=" * 60)

    random.seed(42)

    # Normal MLE
    true_mu, true_sigma = 10.0, 3.0
    data = [random.gauss(true_mu, true_sigma) for _ in range(200)]
    mu_hat, s2_hat = mle_normal(data)
    ll = log_likelihood_normal(data, mu_hat, s2_hat)
    print(f"\n  Normal MLE (n=200):")
    print(f"  μ̂ = {mu_hat:.4f} (true: {true_mu})")
    print(f"  σ̂² = {s2_hat:.4f} (true: {true_sigma**2})")
    print(f"  Log-likelihood = {ll:.2f}")

    # Poisson MLE
    true_lam = 4.5
    pois_data = []
    for _ in range(200):
        count = 0
        L = math.exp(-true_lam)
        p = 1.0
        while p > L:
            count += 1
            p *= random.random()
        pois_data.append(count - 1)
    lam_hat = mle_poisson(pois_data)
    print(f"\n  Poisson MLE (n=200):")
    print(f"  λ̂ = {lam_hat:.4f} (true: {true_lam})")

    # Exponential MLE
    true_lam_exp = 2.0
    exp_data = [random.expovariate(true_lam_exp) for _ in range(200)]
    lam_hat_exp = mle_exponential(exp_data)
    print(f"\n  Exponential MLE (n=200):")
    print(f"  λ̂ = {lam_hat_exp:.4f} (true: {true_lam_exp})")

    # MLE surface for Normal (μ only, σ² fixed)
    print(f"\n  Log-likelihood surface (varying μ, σ²=σ̂²):")
    for mu_try in [mu_hat - 1, mu_hat - 0.5, mu_hat, mu_hat + 0.5, mu_hat + 1]:
        ll_try = log_likelihood_normal(data, mu_try, s2_hat)
        marker = " ← MLE" if abs(mu_try - mu_hat) < 0.01 else ""
        print(f"  μ={mu_try:>7.3f}: ℓ={ll_try:>10.2f}{marker}")


# ─────────────────────────────────────────────────
# 3. FISHER INFORMATION AND CRAMÉR-RAO BOUND
# ─────────────────────────────────────────────────

def fisher_info_normal_mu(sigma2: float) -> float:
    """I(μ) = 1/σ² for Normal with known σ²."""
    return 1 / sigma2


def fisher_info_poisson(lam: float) -> float:
    """I(λ) = 1/λ for Poisson."""
    return 1 / lam


def fisher_info_exponential(lam: float) -> float:
    """I(λ) = 1/λ² for Exponential."""
    return 1 / (lam ** 2)


def demo_fisher_crlb():
    print("\n" + "=" * 60)
    print("  Fisher Information and Cramér-Rao Bound")
    print("=" * 60)

    random.seed(42)

    # Normal μ estimation
    sigma2 = 9.0
    n_samples = [20, 50, 100, 500]
    true_mu = 10.0

    print(f"\n  Normal(μ, σ²={sigma2}), estimating μ:")
    print(f"  CRLB: Var(μ̂) ≥ σ²/n")
    print(f"\n  {'n':>6} {'CRLB':>10} {'Sim Var(μ̂)':>12} {'Efficient?':>12}")
    print(f"  {'─' * 42}")

    for n in n_samples:
        crlb = sigma2 / n
        # Simulate variance of μ̂
        estimates = []
        for _ in range(2000):
            data = [random.gauss(true_mu, math.sqrt(sigma2)) for _ in range(n)]
            estimates.append(sum(data) / n)
        sim_var = sum((e - true_mu)**2 for e in estimates) / len(estimates)
        efficient = "Yes" if abs(sim_var - crlb) / crlb < 0.15 else "No"
        print(f"  {n:>6} {crlb:>10.4f} {sim_var:>12.4f} {efficient:>12}")

    print(f"\n  X̄ achieves the CRLB → efficient estimator for Normal μ")


# ─────────────────────────────────────────────────
# 4. BIAS AND CONSISTENCY
# ─────────────────────────────────────────────────

def demo_bias_consistency():
    print("\n" + "=" * 60)
    print("  Bias, Consistency, and Efficiency")
    print("=" * 60)

    random.seed(42)
    true_sigma2 = 4.0
    true_mu = 0

    # Compare biased vs unbiased variance estimators
    print(f"\n  Variance estimators for N(0, {true_sigma2}):")
    print(f"  S²_MLE = (1/n)Σ(xi-x̄)² (biased)")
    print(f"  S²_unb = (1/(n-1))Σ(xi-x̄)² (unbiased)")

    print(f"\n  {'n':>6} {'E[S²_MLE]':>10} {'E[S²_unb]':>10} {'Bias_MLE':>10}")
    print(f"  {'─' * 38}")

    for n in [5, 10, 30, 100, 500]:
        mle_ests, unb_ests = [], []
        for _ in range(5000):
            data = [random.gauss(true_mu, math.sqrt(true_sigma2)) for _ in range(n)]
            m = sum(data) / n
            ss = sum((x - m)**2 for x in data)
            mle_ests.append(ss / n)
            unb_ests.append(ss / (n - 1))
        e_mle = sum(mle_ests) / len(mle_ests)
        e_unb = sum(unb_ests) / len(unb_ests)
        bias = e_mle - true_sigma2
        print(f"  {n:>6} {e_mle:>10.4f} {e_unb:>10.4f} {bias:>10.4f}")

    print(f"\n  MLE bias = -σ²/n → 0 as n → ∞ (consistent)")
    print(f"  Despite bias, MLE has lower MSE for small n!")


if __name__ == "__main__":
    demo_mom()
    demo_mle()
    demo_fisher_crlb()
    demo_bias_consistency()
