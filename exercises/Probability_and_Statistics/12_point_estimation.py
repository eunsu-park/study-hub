"""
Point Estimation — Exercises

Topics covered:
- Maximum Likelihood Estimation (MLE)
- Method of Moments (MoM)
- Bias of estimators
- Fisher information and Cramer-Rao Lower Bound (CRLB)
"""

import math
import random
from typing import List, Tuple


# === Exercise 1: MLE for Exponential Distribution ===
def exercise_1() -> None:
    """MLE for Exponential: compute lambda_hat = n / sum(xi) from sample data."""
    print("=" * 70)
    print("Exercise 1: MLE for Exponential Distribution")
    print("=" * 70)
    print()
    print("Problem:")
    print("  Given i.i.d. X1,...,Xn ~ Exponential(lambda), the likelihood is:")
    print("  L(lambda) = lambda^n * exp(-lambda * sum(xi))")
    print("  The MLE is lambda_hat = n / sum(xi) = 1 / X̄n.")
    print("  Compute lambda_hat from a sample and verify with simulation.")
    print()

    random.seed(42)
    true_lambda: float = 2.5
    n: int = 50

    # Generate sample from Exponential(lambda) using inverse transform
    sample: List[float] = [-math.log(random.random()) / true_lambda for _ in range(n)]

    sample_sum: float = sum(sample)
    sample_mean: float = sample_sum / n
    lambda_hat: float = n / sample_sum  # equivalently 1 / sample_mean

    print("Solution:")
    print(f"  True lambda = {true_lambda}")
    print(f"  Sample size n = {n}")
    print(f"  Sample mean X̄ = {sample_mean:.6f}")
    print(f"  Sum of samples = {sample_sum:.6f}")
    print()
    print(f"  MLE: lambda_hat = n / sum(xi) = {n} / {sample_sum:.6f} = {lambda_hat:.6f}")
    print(f"  Alternatively: lambda_hat = 1 / X̄ = 1 / {sample_mean:.6f} = {1.0 / sample_mean:.6f}")
    print()

    # Verify with repeated simulation
    num_sims: int = 10000
    mle_values: List[float] = []
    for _ in range(num_sims):
        sim_sample: List[float] = [-math.log(random.random()) / true_lambda for _ in range(n)]
        sim_mean: float = sum(sim_sample) / n
        mle_values.append(1.0 / sim_mean)

    mle_mean: float = sum(mle_values) / num_sims
    mle_var: float = sum((v - mle_mean) ** 2 for v in mle_values) / (num_sims - 1)

    print(f"  Simulation ({num_sims} repetitions):")
    print(f"    E[lambda_hat] = {mle_mean:.6f} (true: {true_lambda})")
    print(f"    Note: E[1/X̄] != 1/E[X̄] — MLE is biased for finite n")
    print(f"    Bias = {mle_mean - true_lambda:.6f}")
    print(f"    Var(lambda_hat) = {mle_var:.6f}")
    print(f"    CRLB = lambda^2/n = {true_lambda ** 2 / n:.6f}")
    print()


# === Exercise 2: Method of Moments for Gamma(alpha, beta) ===
def exercise_2() -> None:
    """Method of Moments for Gamma(alpha, beta): estimate from sample mean and variance."""
    print("=" * 70)
    print("Exercise 2: Method of Moments for Gamma Distribution")
    print("=" * 70)
    print()
    print("Problem:")
    print("  X ~ Gamma(alpha, beta) with E[X] = alpha*beta, Var(X) = alpha*beta^2.")
    print("  Method of Moments: set sample moments = population moments.")
    print("  X̄ = alpha*beta  =>  beta_hat = S^2 / X̄")
    print("  S^2 = alpha*beta^2  =>  alpha_hat = X̄^2 / S^2")
    print("  Estimate from generated sample data.")
    print()

    random.seed(42)
    true_alpha: float = 3.0
    true_beta: float = 2.0
    n: int = 200

    # Generate Gamma(alpha, beta) using sum of Exponentials
    # Gamma(alpha, beta) = sum of alpha Exponential(1/beta) when alpha is integer
    # For non-integer alpha, use acceptance-rejection or other method
    # Here alpha=3 is integer, so Gamma(3, 2) = sum of 3 Exp(1/2) = sum of 3 Exp(rate=0.5)
    sample: List[float] = []
    for _ in range(n):
        # Sum of alpha Exponential(beta) random variables
        x: float = sum(-true_beta * math.log(random.random()) for _ in range(int(true_alpha)))
        sample.append(x)

    x_bar: float = sum(sample) / n
    s_sq: float = sum((x - x_bar) ** 2 for x in sample) / (n - 1)

    alpha_hat: float = x_bar ** 2 / s_sq
    beta_hat: float = s_sq / x_bar

    print("Solution:")
    print(f"  True parameters: alpha = {true_alpha}, beta = {true_beta}")
    print(f"  E[X] = alpha*beta = {true_alpha * true_beta}")
    print(f"  Var(X) = alpha*beta^2 = {true_alpha * true_beta ** 2}")
    print(f"  Sample size n = {n}")
    print()
    print(f"  Sample statistics:")
    print(f"    X̄ = {x_bar:.4f} (theoretical: {true_alpha * true_beta:.1f})")
    print(f"    S^2 = {s_sq:.4f} (theoretical: {true_alpha * true_beta ** 2:.1f})")
    print()
    print(f"  Method of Moments estimates:")
    print(f"    alpha_hat = X̄^2 / S^2 = {x_bar:.4f}^2 / {s_sq:.4f} = {alpha_hat:.4f}")
    print(f"    beta_hat  = S^2 / X̄   = {s_sq:.4f} / {x_bar:.4f} = {beta_hat:.4f}")
    print()
    print(f"  Comparison:")
    print(f"    alpha: true = {true_alpha:.1f}, estimate = {alpha_hat:.4f}, error = {abs(alpha_hat - true_alpha):.4f}")
    print(f"    beta:  true = {true_beta:.1f}, estimate = {beta_hat:.4f}, error = {abs(beta_hat - true_beta):.4f}")
    print()


# === Exercise 3: Bias of MLE for Normal Variance ===
def exercise_3() -> None:
    """Verify MLE for Normal is biased for sigma^2 (compare 1/n vs 1/(n-1) estimators)."""
    print("=" * 70)
    print("Exercise 3: Bias of MLE for Normal Variance")
    print("=" * 70)
    print()
    print("Problem:")
    print("  For X1,...,Xn ~ N(mu, sigma^2), the MLE for sigma^2 is:")
    print("    sigma_hat^2_MLE = (1/n) * sum((xi - x_bar)^2)")
    print("  The unbiased estimator is:")
    print("    S^2 = (1/(n-1)) * sum((xi - x_bar)^2)")
    print("  E[sigma_hat^2_MLE] = ((n-1)/n) * sigma^2 (biased)")
    print("  E[S^2] = sigma^2 (unbiased)")
    print("  Verify by simulation for small n.")
    print()

    random.seed(42)
    true_mu: float = 5.0
    true_sigma_sq: float = 4.0
    true_sigma: float = math.sqrt(true_sigma_sq)
    sample_sizes: List[int] = [5, 10, 30, 100]
    num_sims: int = 50000

    def box_muller_pair() -> Tuple[float, float]:
        u1: float = random.random()
        u2: float = random.random()
        while u1 == 0.0:
            u1 = random.random()
        z1: float = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        z2: float = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
        return z1, z2

    def generate_normal(mu: float, sigma: float, n: int) -> List[float]:
        result: List[float] = []
        for _ in range((n + 1) // 2):
            z1, z2 = box_muller_pair()
            result.append(mu + sigma * z1)
            result.append(mu + sigma * z2)
        return result[:n]

    print("Solution:")
    print(f"  True mu = {true_mu}, sigma^2 = {true_sigma_sq}")
    print(f"  Number of simulations = {num_sims}")
    print()
    print(f"  {'n':>5}  {'E[MLE]':>10}  {'Bias(MLE)':>10}  {'(n-1)/n*s2':>12}  {'E[S^2]':>10}  {'Bias(S^2)':>10}")
    print(f"  {'-' * 5}  {'-' * 10}  {'-' * 10}  {'-' * 12}  {'-' * 10}  {'-' * 10}")

    for n in sample_sizes:
        mle_values: List[float] = []
        s2_values: List[float] = []

        for _ in range(num_sims):
            data: List[float] = generate_normal(true_mu, true_sigma, n)
            x_bar: float = sum(data) / n
            ss: float = sum((x - x_bar) ** 2 for x in data)
            mle_values.append(ss / n)
            s2_values.append(ss / (n - 1))

        e_mle: float = sum(mle_values) / num_sims
        e_s2: float = sum(s2_values) / num_sims
        theoretical_bias: float = (n - 1) / n * true_sigma_sq

        print(f"  {n:>5}  {e_mle:>10.4f}  {e_mle - true_sigma_sq:>+10.4f}  {theoretical_bias:>12.4f}  {e_s2:>10.4f}  {e_s2 - true_sigma_sq:>+10.4f}")

    print()
    print("  The MLE divides by n and is biased downward: E[MLE] = (n-1)/n * sigma^2.")
    print("  S^2 divides by (n-1) and is unbiased: E[S^2] = sigma^2.")
    print("  Bias of MLE decreases as n grows.")
    print()


# === Exercise 4: Fisher Information and CRLB for Bernoulli ===
def exercise_4() -> None:
    """Compute Fisher information for Bernoulli(p), verify CRLB vs simulated Var(p_hat)."""
    print("=" * 70)
    print("Exercise 4: Fisher Information and CRLB for Bernoulli(p)")
    print("=" * 70)
    print()
    print("Problem:")
    print("  X ~ Bernoulli(p). The log-likelihood for one observation:")
    print("    l(p) = x*log(p) + (1-x)*log(1-p)")
    print("  Score: l'(p) = x/p - (1-x)/(1-p)")
    print("  Fisher information: I(p) = E[l'(p)^2] = 1/(p*(1-p))")
    print("  For n i.i.d. observations: In(p) = n/(p*(1-p))")
    print("  CRLB: Var(p_hat) >= 1/In(p) = p*(1-p)/n")
    print("  The MLE p_hat = X̄ achieves the CRLB (efficient estimator).")
    print("  Verify by simulation.")
    print()

    random.seed(42)
    true_p: float = 0.3
    sample_sizes: List[int] = [10, 30, 100, 500, 1000]
    num_sims: int = 50000

    print("Solution:")
    print(f"  True p = {true_p}")
    print(f"  Fisher information I(p) = 1/(p*(1-p)) = {1.0 / (true_p * (1 - true_p)):.4f}")
    print(f"  Number of simulations = {num_sims}")
    print()
    print(f"  {'n':>6}  {'CRLB':>10}  {'Sim Var(p̂)':>12}  {'Ratio':>8}  {'Efficient?':>11}")
    print(f"  {'-' * 6}  {'-' * 10}  {'-' * 12}  {'-' * 8}  {'-' * 11}")

    for n in sample_sizes:
        crlb: float = true_p * (1.0 - true_p) / n

        p_hat_values: List[float] = []
        for _ in range(num_sims):
            successes: int = sum(1 for _ in range(n) if random.random() < true_p)
            p_hat_values.append(successes / n)

        p_hat_mean: float = sum(p_hat_values) / num_sims
        p_hat_var: float = sum((v - p_hat_mean) ** 2 for v in p_hat_values) / (num_sims - 1)
        ratio: float = p_hat_var / crlb
        efficient: str = "Yes" if abs(ratio - 1.0) < 0.05 else "Close"

        print(f"  {n:>6}  {crlb:>10.6f}  {p_hat_var:>12.6f}  {ratio:>8.4f}  {efficient:>11}")

    print()
    print("  The MLE p_hat = X̄ achieves the CRLB (ratio ~ 1.0).")
    print("  It is the most efficient unbiased estimator for p.")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
