"""
Bayesian Regression Examples
- Linear, logistic, Poisson, robust regression
"""
import numpy as np
from scipy import stats


def bayesian_linear_regression_analytic():
    """Analytic Bayesian linear regression with conjugate prior."""
    np.random.seed(42)
    n = 50
    X = np.column_stack([np.ones(n), np.random.randn(n)])
    true_beta = np.array([2.0, -1.5])
    sigma = 0.8
    y = X @ true_beta + np.random.normal(0, sigma, n)

    # Conjugate posterior: beta | y, sigma ~ Normal(m_n, S_n)
    S_0_inv = np.eye(2) * 0.01  # weak prior precision
    m_0 = np.zeros(2)
    S_n_inv = S_0_inv + X.T @ X / sigma**2
    S_n = np.linalg.inv(S_n_inv)
    m_n = S_n @ (S_0_inv @ m_0 + X.T @ y / sigma**2)

    print("Analytic Bayesian Linear Regression:")
    print(f"  Posterior mean: {m_n.round(3)}")
    print(f"  True beta:     {true_beta}")
    print(f"  Posterior std:  {np.sqrt(np.diag(S_n)).round(3)}")


def robust_vs_normal():
    """Compare normal vs Student-t regression on data with outliers."""
    np.random.seed(42)
    n = 50
    x = np.random.uniform(0, 10, n)
    y = 2.0 + 1.5 * x + np.random.normal(0, 1.0, n)
    # Add outliers
    y[np.random.choice(n, 3, replace=False)] += np.random.normal(0, 15, 3)

    # OLS (sensitive to outliers)
    X = np.column_stack([np.ones(n), x])
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]

    # Huber robust regression
    from scipy.optimize import minimize
    def huber_loss(beta, X, y, delta=1.35):
        r = y - X @ beta
        return np.sum(np.where(np.abs(r) <= delta, 0.5*r**2, delta*(np.abs(r) - 0.5*delta)))
    result = minimize(huber_loss, [0, 0], args=(X, y))

    print(f"\nRobust vs Normal Regression (true: intercept=2.0, slope=1.5):")
    print(f"  OLS:   intercept={beta_ols[0]:.3f}, slope={beta_ols[1]:.3f}")
    print(f"  Huber: intercept={result.x[0]:.3f}, slope={result.x[1]:.3f}")


if __name__ == "__main__":
    bayesian_linear_regression_analytic()
    robust_vs_normal()
