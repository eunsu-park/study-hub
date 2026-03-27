"""
GARCH Volatility Modeling
=========================
Demonstrates:
- ARCH/GARCH model intuition and mechanics
- Volatility clustering simulation
- GARCH(1,1) parameter estimation (simplified)
- Conditional variance forecasting
- Comparison with rolling-window volatility

Theory:
- Financial returns exhibit volatility clustering: large changes
  tend to be followed by large changes (of either sign).
- ARCH(q): conditional variance depends on past squared returns.
    sigma_t^2 = omega + sum(alpha_i * r_{t-i}^2)
- GARCH(p,q): adds lagged conditional variance terms.
    sigma_t^2 = omega + sum(alpha_i * r_{t-i}^2) + sum(beta_j * sigma_{t-j}^2)
- GARCH(1,1) is the most widely used:
    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
  where alpha + beta < 1 for stationarity.
- Unconditional variance: sigma^2 = omega / (1 - alpha - beta)

Note: This is a pure NumPy implementation for educational purposes.
For production use, see the arch library (pip install arch).

Adapted from Data_Science Lesson 21.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


# ── GARCH(1,1) Simulation ────────────────────────────────────────

@dataclass
class GARCHParams:
    """GARCH(1,1) parameters."""
    omega: float     # Constant term
    alpha: float     # ARCH coefficient (shock impact)
    beta: float      # GARCH coefficient (persistence)

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def unconditional_var(self) -> float:
        if self.persistence >= 1:
            return float('inf')
        return self.omega / (1 - self.persistence)

    @property
    def half_life(self) -> float:
        """Half-life of volatility shock in periods."""
        if self.persistence <= 0 or self.persistence >= 1:
            return float('inf')
        return np.log(0.5) / np.log(self.persistence)


def simulate_garch(params: GARCHParams, n: int = 1000,
                   seed: int = 42) -> dict:
    """Simulate a GARCH(1,1) process."""
    rng = np.random.default_rng(seed)

    returns = np.zeros(n)
    sigma2 = np.zeros(n)  # Conditional variance
    sigma2[0] = params.unconditional_var

    for t in range(1, n):
        sigma2[t] = (params.omega
                     + params.alpha * returns[t-1]**2
                     + params.beta * sigma2[t-1])
        returns[t] = np.sqrt(sigma2[t]) * rng.standard_normal()

    return {
        "returns": returns,
        "sigma2": sigma2,
        "volatility": np.sqrt(sigma2),
    }


# ── Simple GARCH(1,1) Estimation ─────────────────────────────────

def estimate_garch_mle(returns: np.ndarray,
                       max_iter: int = 200) -> GARCHParams:
    """Estimate GARCH(1,1) via grid search on log-likelihood.

    Note: This is a simplified approach for educational purposes.
    Production code should use scipy.optimize or the arch library.
    """
    r2 = returns**2
    n = len(returns)
    sample_var = np.var(returns)

    best_ll = -np.inf
    best_params = GARCHParams(0.01, 0.05, 0.9)

    # Grid search over (alpha, beta) space
    for alpha in np.arange(0.01, 0.30, 0.02):
        for beta in np.arange(0.50, 0.99, 0.02):
            if alpha + beta >= 0.999:
                continue
            omega = sample_var * (1 - alpha - beta)
            if omega <= 0:
                continue

            # Compute conditional variances
            sigma2 = np.zeros(n)
            sigma2[0] = sample_var
            for t in range(1, n):
                sigma2[t] = omega + alpha * r2[t-1] + beta * sigma2[t-1]
                if sigma2[t] <= 0:
                    break

            if np.any(sigma2 <= 0):
                continue

            # Log-likelihood (Gaussian)
            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + r2 / sigma2)

            if ll > best_ll:
                best_ll = ll
                best_params = GARCHParams(omega, alpha, beta)

    return best_params


def garch_forecast(params: GARCHParams, last_return: float,
                   last_sigma2: float, horizon: int = 20) -> np.ndarray:
    """Forecast conditional variance h steps ahead."""
    forecasts = np.zeros(horizon)
    sigma2 = params.omega + params.alpha * last_return**2 + params.beta * last_sigma2

    for h in range(horizon):
        forecasts[h] = sigma2
        # Multi-step: E[sigma2_{t+h}] converges to unconditional variance
        sigma2 = params.omega + (params.alpha + params.beta) * sigma2

    return forecasts


# ── Rolling Volatility (Baseline) ─────────────────────────────────

def rolling_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Simple rolling-window standard deviation."""
    n = len(returns)
    vol = np.full(n, np.nan)
    for i in range(window, n):
        vol[i] = np.std(returns[i-window:i])
    return vol


# ── Demos ─────────────────────────────────────────────────────────

def demo_simulation():
    print("=" * 60)
    print("GARCH(1,1) SIMULATION")
    print("=" * 60)

    params = GARCHParams(omega=0.00001, alpha=0.08, beta=0.90)
    print(f"\n  Parameters:")
    print(f"    omega = {params.omega}")
    print(f"    alpha = {params.alpha}")
    print(f"    beta  = {params.beta}")
    print(f"    alpha + beta = {params.persistence:.2f}")
    print(f"    Unconditional vol = {np.sqrt(params.unconditional_var)*100:.2f}%")
    print(f"    Half-life = {params.half_life:.1f} periods")

    result = simulate_garch(params, n=1000)
    returns = result["returns"]
    vol = result["volatility"]

    print(f"\n  Simulated data:")
    print(f"    Mean return: {returns.mean():.6f}")
    print(f"    Std return:  {returns.std():.6f}")
    print(f"    Min return:  {returns.min():.6f}")
    print(f"    Max return:  {returns.max():.6f}")
    print(f"    Mean vol:    {vol.mean():.6f}")

    # Check for volatility clustering
    abs_returns = np.abs(returns)
    acf_1 = np.corrcoef(abs_returns[1:], abs_returns[:-1])[0, 1]
    acf_5 = np.corrcoef(abs_returns[5:], abs_returns[:-5])[0, 1]
    print(f"\n  Volatility clustering evidence:")
    print(f"    ACF(|r|, lag=1) = {acf_1:.4f}")
    print(f"    ACF(|r|, lag=5) = {acf_5:.4f}")
    print(f"    (Positive autocorrelation in |returns| confirms clustering)")


def demo_estimation():
    print("\n" + "=" * 60)
    print("GARCH(1,1) ESTIMATION")
    print("=" * 60)

    # Generate data with known parameters
    true_params = GARCHParams(omega=0.00002, alpha=0.10, beta=0.85)
    result = simulate_garch(true_params, n=2000, seed=123)
    returns = result["returns"]

    print(f"\n  True parameters:")
    print(f"    omega={true_params.omega}, alpha={true_params.alpha}, "
          f"beta={true_params.beta}")

    # Estimate
    est_params = estimate_garch_mle(returns)
    print(f"\n  Estimated parameters:")
    print(f"    omega={est_params.omega:.6f}, alpha={est_params.alpha:.2f}, "
          f"beta={est_params.beta:.2f}")
    print(f"    Persistence: {est_params.persistence:.4f} "
          f"(true: {true_params.persistence:.4f})")
    print(f"    Uncond. vol: {np.sqrt(est_params.unconditional_var)*100:.2f}% "
          f"(true: {np.sqrt(true_params.unconditional_var)*100:.2f}%)")


def demo_forecasting():
    print("\n" + "=" * 60)
    print("VOLATILITY FORECASTING")
    print("=" * 60)

    params = GARCHParams(omega=0.00001, alpha=0.08, beta=0.90)
    result = simulate_garch(params, n=500, seed=42)
    returns = result["returns"]
    sigma2 = result["sigma2"]

    # Forecast from last observed values
    horizon = 20
    forecasts = garch_forecast(
        params,
        last_return=returns[-1],
        last_sigma2=sigma2[-1],
        horizon=horizon,
    )

    print(f"\n  Last observed return: {returns[-1]:.6f}")
    print(f"  Last conditional vol: {np.sqrt(sigma2[-1])*100:.4f}%")
    print(f"  Unconditional vol:    {np.sqrt(params.unconditional_var)*100:.4f}%")

    print(f"\n  Volatility forecasts (annualized, h=1..{horizon}):")
    for h in [1, 5, 10, 20]:
        if h <= horizon:
            vol_pct = np.sqrt(forecasts[h-1]) * 100
            uncond = np.sqrt(params.unconditional_var) * 100
            print(f"    h={h:>2}: {vol_pct:.4f}% "
                  f"(converging to {uncond:.4f}%)")


def demo_comparison():
    print("\n" + "=" * 60)
    print("GARCH vs ROLLING VOLATILITY")
    print("=" * 60)

    params = GARCHParams(omega=0.00001, alpha=0.10, beta=0.88)
    result = simulate_garch(params, n=500, seed=42)
    returns = result["returns"]
    garch_vol = result["volatility"]
    roll_vol = rolling_volatility(returns, window=20)

    # Compare at points where rolling is available
    valid = ~np.isnan(roll_vol)
    garch_subset = garch_vol[valid]
    roll_subset = roll_vol[valid]

    correlation = np.corrcoef(garch_subset, roll_subset)[0, 1]
    mae = np.mean(np.abs(garch_subset - roll_subset))

    print(f"\n  Correlation: {correlation:.4f}")
    print(f"  Mean absolute difference: {mae:.6f}")

    print(f"""
  GARCH Advantages:
    - Captures volatility clustering dynamically
    - Provides multi-step forecasts
    - Reacts quickly to shocks (alpha controls speed)
    - Smooth volatility estimates (less noisy than rolling)

  Rolling Window Advantages:
    - Simple to implement and explain
    - No parametric assumptions
    - Robust to model misspecification

  When to use GARCH:
    - Financial risk management (VaR, ES)
    - Option pricing (implied vs realized vol)
    - Portfolio optimization with time-varying risk""")


def demo_arch_effects_test():
    print("\n" + "=" * 60)
    print("TESTING FOR ARCH EFFECTS")
    print("=" * 60)

    # Simulate GARCH data and iid data
    params = GARCHParams(omega=0.00001, alpha=0.10, beta=0.85)
    garch_result = simulate_garch(params, n=500, seed=42)
    rng = np.random.default_rng(42)
    iid_returns = rng.normal(0, 0.01, 500)

    for name, returns in [("GARCH process", garch_result["returns"]),
                          ("IID Normal", iid_returns)]:
        r2 = returns**2
        n = len(r2)

        # Ljung-Box style test on squared returns
        acfs = []
        for lag in range(1, 11):
            acf = np.corrcoef(r2[lag:], r2[:-lag])[0, 1]
            acfs.append(acf)

        # Engle's ARCH test statistic (simplified): n * R^2 from
        # regressing r_t^2 on r_{t-1}^2, ..., r_{t-q}^2
        q = 5
        Y = r2[q:]
        X = np.column_stack([r2[q-i-1:n-i-1] for i in range(q)])
        X = np.column_stack([np.ones(len(Y)), X])

        # OLS: beta = (X'X)^-1 X'Y
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        Y_hat = X @ beta
        ss_res = np.sum((Y - Y_hat)**2)
        ss_tot = np.sum((Y - Y.mean())**2)
        r_squared = 1 - ss_res / ss_tot
        test_stat = len(Y) * r_squared

        print(f"\n  {name}:")
        print(f"    ACF of r^2 (lags 1-5): "
              f"{[f'{a:.3f}' for a in acfs[:5]]}")
        print(f"    ARCH LM test stat (q={q}): {test_stat:.2f}")
        print(f"    R^2 of squared returns regression: {r_squared:.4f}")
        sig = "ARCH effects detected" if test_stat > 11.07 else "No ARCH effects"
        print(f"    Conclusion: {sig} (chi2_5 critical: 11.07)")


if __name__ == "__main__":
    demo_simulation()
    demo_estimation()
    demo_forecasting()
    demo_comparison()
    demo_arch_effects_test()
