"""
Model Comparison Examples
- WAIC, LOO-CV, Bayes factors, posterior predictive checks
"""
import numpy as np
from scipy import stats


def savage_dickey_bf(posterior_samples, null_value, prior_density_at_null):
    """Bayes factor via Savage-Dickey density ratio."""
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(posterior_samples)
    post_density = kde(null_value)[0]
    bf_01 = post_density / prior_density_at_null
    return bf_01, 1/bf_01


def posterior_predictive_check(y_obs, y_rep, stat_fn=np.mean):
    """Compute posterior predictive p-value."""
    obs_stat = stat_fn(y_obs)
    rep_stats = np.array([stat_fn(yr) for yr in y_rep])
    p_value = np.mean(rep_stats >= obs_stat)
    return p_value


def model_comparison_demo():
    """Compare polynomial models using information criteria proxy."""
    np.random.seed(42)
    n = 50
    x = np.random.uniform(-3, 3, n)
    y = 1.0 + 0.5*x + 0.3*x**2 + np.random.normal(0, 0.5, n)

    for degree in [1, 2, 3, 5]:
        X = np.column_stack([x**d for d in range(degree+1)])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred = X @ beta
        resid = y - y_pred
        rss = np.sum(resid**2)
        k = degree + 2  # coefficients + sigma
        aic = n * np.log(rss/n) + 2*k
        bic = n * np.log(rss/n) + k*np.log(n)
        print(f"Degree {degree}: AIC={aic:.1f}, BIC={bic:.1f}, RSS={rss:.2f}")


if __name__ == "__main__":
    print("=== Model Comparison (AIC/BIC proxy) ===")
    model_comparison_demo()

    print("\n=== Savage-Dickey Bayes Factor ===")
    # Simulate posterior samples for a coefficient
    post = np.random.normal(0.3, 0.1, 10000)
    prior_at_0 = stats.norm.pdf(0, 0, 2)
    bf01, bf10 = savage_dickey_bf(post, 0, prior_at_0)
    print(f"BF_01 (favor null): {bf01:.4f}")
    print(f"BF_10 (against null): {bf10:.4f}")

    print("\n=== Posterior Predictive Check ===")
    y_obs = np.random.normal(5, 2, 50)
    y_rep = [np.random.normal(5.1, 2.1, 50) for _ in range(1000)]
    for stat_name, stat_fn in [("mean", np.mean), ("std", np.std), ("min", np.min)]:
        pval = posterior_predictive_check(y_obs, y_rep, stat_fn)
        print(f"  {stat_name}: p-value = {pval:.3f}")
