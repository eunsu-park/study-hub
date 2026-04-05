"""
Causal Inference Examples
- Backdoor adjustment, instrumental variables, propensity scores
"""
import numpy as np
from sklearn.linear_model import LinearRegression


def backdoor_adjustment():
    """Estimate causal effect via backdoor adjustment."""
    np.random.seed(42)
    n = 1000
    C = np.random.normal(0, 1, n)        # confounder
    X = 0.8 * C + np.random.normal(0, 0.5, n)  # treatment
    Y = 1.5 * X - 0.5 * C + np.random.normal(0, 0.5, n)  # outcome

    # Naive (biased)
    naive = LinearRegression().fit(X.reshape(-1,1), Y).coef_[0]
    # Adjusted (control for C)
    adjusted = LinearRegression().fit(np.column_stack([X, C]), Y).coef_[0]

    print("Backdoor Adjustment:")
    print(f"  Naive effect:    {naive:.3f} (biased by confounder)")
    print(f"  Adjusted effect: {adjusted:.3f} (true: 1.5)")


def instrumental_variables():
    """Two-stage least squares for IV estimation."""
    np.random.seed(42)
    n = 1000
    Z = np.random.normal(0, 1, n)        # instrument
    C = np.random.normal(0, 1, n)        # unobserved confounder
    X = 0.5 * Z + 0.8 * C + np.random.normal(0, 0.5, n)
    Y = 1.5 * X - 0.5 * C + np.random.normal(0, 0.5, n)

    # Stage 1: X ~ Z
    reg1 = LinearRegression().fit(Z.reshape(-1,1), X)
    X_hat = reg1.predict(Z.reshape(-1,1))
    # Stage 2: Y ~ X_hat
    iv_effect = LinearRegression().fit(X_hat.reshape(-1,1), Y).coef_[0]

    print(f"\nInstrumental Variables (2SLS):")
    print(f"  IV effect: {iv_effect:.3f} (true: 1.5)")


def propensity_score_demo():
    """Simple propensity score estimation."""
    np.random.seed(42)
    n = 500
    x = np.random.randn(n, 3)
    p_treat = 1 / (1 + np.exp(-(0.5*x[:,0] - 0.3*x[:,1])))
    treatment = np.random.binomial(1, p_treat)
    y = 2.0 * treatment + 0.5*x[:,0] + np.random.normal(0, 1, n)

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression().fit(x, treatment)
    ps = lr.predict_proba(x)[:,1]

    # IPW estimate
    w1 = treatment / ps
    w0 = (1-treatment) / (1-ps)
    ate_ipw = (w1 * y).mean() / w1.mean() - (w0 * y).mean() / w0.mean()
    print(f"\nPropensity Score IPW:")
    print(f"  ATE estimate: {ate_ipw:.3f} (true: 2.0)")


if __name__ == "__main__":
    backdoor_adjustment()
    instrumental_variables()
    propensity_score_demo()
