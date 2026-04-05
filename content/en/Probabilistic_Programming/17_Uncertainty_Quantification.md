# 17. Uncertainty Quantification

[Previous: Model Comparison](./16_Model_Comparison.md) | [Next: Capstone Applied Bayesian](./18_Capstone_Applied_Bayesian.md)

---

> **Framework Note**: This lesson uses NumPy, scikit-learn, and PyMC for UQ methods.
>
> Installation: `pip install numpy scipy matplotlib scikit-learn pymc arviz mapie`

## Learning Objectives

- Understand calibration and how to measure it
- Implement conformal prediction for distribution-free coverage guarantees
- Apply decision theory under uncertainty
- Build calibrated probabilistic forecasts
- Use UQ for active learning and risk management

---

## 1. What is Calibration?

A model is well-calibrated if its predicted probabilities match observed frequencies. For a model predicting 80% probability of rain, it should actually rain ~80% of the time.

### 1.1 Measuring Calibration

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
X_train, X_test = X[:3000], X[3000:]
y_train, y_test = y[:3000], y[3000:]

# Train two models
lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

# Calibration curves
fig, ax = plt.subplots(figsize=(8, 8))
for name, model in [("Logistic Regression", lr), ("Random Forest", rf)]:
    prob_pred = model.predict_proba(X_test)[:, 1]
    fraction_positive, mean_predicted = calibration_curve(y_test, prob_pred, n_bins=10)
    ax.plot(mean_predicted, fraction_positive, 's-', label=name)

ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Fraction of positives')
ax.set_title('Calibration Curve (Reliability Diagram)')
ax.legend()
plt.tight_layout()
plt.savefig("calibration_curve.png", dpi=100)
plt.show()
```

### 1.2 Expected Calibration Error (ECE)

```python
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute ECE: weighted average of calibration gaps."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_accuracy = y_true[mask].mean()
        bin_confidence = y_prob[mask].mean()
        bin_weight = mask.sum() / len(y_true)
        ece += bin_weight * abs(bin_accuracy - bin_confidence)
    return ece

for name, model in [("LR", lr), ("RF", rf)]:
    probs = model.predict_proba(X_test)[:, 1]
    ece = expected_calibration_error(y_test, probs)
    print(f"{name} ECE: {ece:.4f}")
```

---

## 2. Calibration Methods

### 2.1 Platt Scaling

```python
from sklearn.calibration import CalibratedClassifierCV

# Platt scaling (logistic calibration)
rf_calibrated = CalibratedClassifierCV(rf, method='sigmoid', cv=5)
rf_calibrated.fit(X_train, y_train)

probs_uncalibrated = rf.predict_proba(X_test)[:, 1]
probs_calibrated = rf_calibrated.predict_proba(X_test)[:, 1]

ece_before = expected_calibration_error(y_test, probs_uncalibrated)
ece_after = expected_calibration_error(y_test, probs_calibrated)
print(f"RF ECE before calibration: {ece_before:.4f}")
print(f"RF ECE after Platt scaling: {ece_after:.4f}")
```

### 2.2 Temperature Scaling (for Neural Networks)

```python
import torch
import torch.nn as nn

class TemperatureScaling(nn.Module):
    """Temperature scaling for neural network calibration."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, labels, lr=0.01, n_steps=100):
        """Optimize temperature on validation set."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=n_steps)
        nll = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"Optimal temperature: {self.temperature.item():.3f}")
```

---

## 3. Conformal Prediction

Conformal prediction provides **distribution-free** prediction intervals with finite-sample coverage guarantees.

### 3.1 Split Conformal Prediction

```python
def split_conformal_regression(X_train, y_train, X_cal, y_cal, X_test, model, alpha=0.1):
    """
    Split conformal prediction for regression.
    Guarantees P(Y_new ∈ C(X_new)) ≥ 1 - alpha.
    """
    # Fit model on training data
    model.fit(X_train, y_train)

    # Compute residuals on calibration set
    y_cal_pred = model.predict(X_cal)
    residuals = np.abs(y_cal - y_cal_pred)

    # Quantile of residuals
    n_cal = len(y_cal)
    q = np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
    q_residual = np.quantile(residuals, q)

    # Prediction intervals for test set
    y_test_pred = model.predict(X_test)
    lower = y_test_pred - q_residual
    upper = y_test_pred + q_residual

    print(f"Conformal interval width: ±{q_residual:.3f}")
    return y_test_pred, lower, upper


# Example
from sklearn.ensemble import GradientBoostingRegressor

np.random.seed(42)
n = 1000
X_all = np.random.uniform(-3, 3, n).reshape(-1, 1)
y_all = np.sin(X_all.flatten()) + np.random.normal(0, 0.3, n)

X_tr, y_tr = X_all[:500], y_all[:500]
X_cal, y_cal = X_all[500:750], y_all[500:750]
X_te, y_te = X_all[750:], y_all[750:]

gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
pred, lower, upper = split_conformal_regression(X_tr, y_tr, X_cal, y_cal, X_te, gb, alpha=0.1)

coverage = np.mean((y_te >= lower) & (y_te <= upper))
print(f"Empirical coverage: {coverage:.3f} (target: 0.90)")
```

### 3.2 Conformalized Quantile Regression

```python
def cqr_prediction(X_train, y_train, X_cal, y_cal, X_test, alpha=0.1):
    """
    Conformalized Quantile Regression (CQR).
    Adaptive intervals that widen/narrow based on local uncertainty.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    # Train quantile regressors
    qr_lo = GradientBoostingRegressor(loss='quantile', alpha=alpha/2, random_state=42)
    qr_hi = GradientBoostingRegressor(loss='quantile', alpha=1-alpha/2, random_state=42)

    qr_lo.fit(X_train, y_train)
    qr_hi.fit(X_train, y_train)

    # Calibration scores
    cal_lo = qr_lo.predict(X_cal)
    cal_hi = qr_hi.predict(X_cal)
    scores = np.maximum(cal_lo - y_cal, y_cal - cal_hi)

    # Quantile of scores
    n_cal = len(y_cal)
    q = np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
    Q = np.quantile(scores, min(q, 1.0))

    # Adaptive prediction intervals
    test_lo = qr_lo.predict(X_test) - Q
    test_hi = qr_hi.predict(X_test) + Q

    return test_lo, test_hi

lo, hi = cqr_prediction(X_tr, y_tr, X_cal, y_cal, X_te)
coverage_cqr = np.mean((y_te >= lo) & (y_te <= hi))
avg_width = np.mean(hi - lo)
print(f"CQR coverage: {coverage_cqr:.3f}, avg width: {avg_width:.3f}")
```

---

## 4. Decision-Making Under Uncertainty

### 4.1 Expected Loss Framework

```python
def optimal_decision(posterior_samples, loss_functions, actions):
    """Select the action minimizing expected loss under posterior uncertainty."""
    expected_losses = {}
    for action_name, action_value in actions.items():
        losses = []
        for theta in posterior_samples:
            loss = loss_functions[action_name](theta, action_value)
            losses.append(loss)
        expected_losses[action_name] = np.mean(losses)

    best_action = min(expected_losses, key=expected_losses.get)
    print("Expected losses:")
    for name, el in expected_losses.items():
        marker = " ← BEST" if name == best_action else ""
        print(f"  {name}: {el:.4f}{marker}")
    return best_action


# Example: inventory decision
# theta = true demand rate (uncertain)
posterior_demand = np.random.gamma(50, 2, size=10000)  # uncertain demand

actions = {"stock_80": 80, "stock_100": 100, "stock_120": 120, "stock_150": 150}
loss_fns = {
    name: lambda theta, a, act=val: (
        2 * max(0, theta - act) +  # stockout cost: $2 per unit
        0.5 * max(0, act - theta)   # overstock cost: $0.5 per unit
    ) for name, val in actions.items()
}

best = optimal_decision(posterior_demand, loss_fns, actions)
```

### 4.2 Value of Information

```python
def value_of_perfect_information(posterior_samples, actions, loss_fn):
    """Compute Expected Value of Perfect Information (EVPI)."""
    # Current best: minimize expected loss under uncertainty
    current_losses = {}
    for name, val in actions.items():
        current_losses[name] = np.mean([loss_fn(theta, val) for theta in posterior_samples])
    current_best = min(current_losses.values())

    # With perfect info: for each theta, pick the best action
    perfect_info_losses = []
    for theta in posterior_samples:
        losses = [loss_fn(theta, val) for val in actions.values()]
        perfect_info_losses.append(min(losses))
    expected_perfect = np.mean(perfect_info_losses)

    evpi = current_best - expected_perfect
    print(f"Current best expected loss: {current_best:.2f}")
    print(f"Expected loss with perfect info: {expected_perfect:.2f}")
    print(f"EVPI (max you should pay for info): {evpi:.2f}")
    return evpi
```

---

## 5. Prediction Interval Evaluation

```python
def evaluate_prediction_intervals(y_true, lower, upper, alpha=0.1):
    """Comprehensive evaluation of prediction intervals."""
    n = len(y_true)
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    avg_width = np.mean(upper - lower)
    median_width = np.median(upper - lower)

    # Winkler score: penalizes both width and miscoverage
    winkler = upper - lower
    below = y_true < lower
    above = y_true > upper
    winkler[below] += (2 / alpha) * (lower[below] - y_true[below])
    winkler[above] += (2 / alpha) * (y_true[above] - upper[above])
    avg_winkler = winkler.mean()

    # Conditional coverage by quantile
    print(f"Overall coverage: {coverage:.3f} (target: {1-alpha:.2f})")
    print(f"Average width: {avg_width:.3f}")
    print(f"Winkler score: {avg_winkler:.3f}")

    # Check conditional coverage
    n_bins = 5
    for i in range(n_bins):
        lo_q = np.percentile(y_true, i * 100 / n_bins)
        hi_q = np.percentile(y_true, (i+1) * 100 / n_bins)
        mask = (y_true >= lo_q) & (y_true < hi_q)
        if mask.sum() > 0:
            bin_cov = np.mean((y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask]))
            print(f"  Bin {i}: coverage={bin_cov:.3f}, n={mask.sum()}")

evaluate_prediction_intervals(y_te, lower, upper)
```

---

## 6. Active Learning with Uncertainty

```python
def uncertainty_sampling(model, X_pool, X_train, y_train, n_query=10):
    """Select most uncertain points for labeling."""
    model.fit(X_train, y_train)

    # For regression: use prediction variance
    # For BNN/GP: use epistemic uncertainty
    preds = np.array([model.predict(X_pool) for _ in range(50)])
    uncertainty = preds.std(axis=0)

    # Select top-k most uncertain
    query_idx = np.argsort(uncertainty)[-n_query:]
    return query_idx

# For Bayesian models, epistemic uncertainty is the natural acquisition function
# High epistemic uncertainty = model is uncertain = worth querying
```

---

## 7. Bayesian Credible vs Frequentist Confidence

```python
def compare_intervals():
    """Side-by-side comparison of credible and confidence intervals."""
    np.random.seed(42)
    data = np.random.normal(5.0, 2.0, 20)

    # Frequentist CI
    from scipy.stats import t
    n = len(data)
    mean, se = data.mean(), data.std(ddof=1) / np.sqrt(n)
    t_crit = t.ppf(0.975, n - 1)
    ci = (mean - t_crit * se, mean + t_crit * se)

    # Bayesian HDI (with flat prior)
    # Posterior: mu ~ t(n-1, mean, se)
    posterior_samples = t.rvs(n - 1, loc=mean, scale=se, size=100000)
    hdi = np.percentile(posterior_samples, [2.5, 97.5])

    print(f"Frequentist 95% CI:  [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"Bayesian 95% HDI:    [{hdi[0]:.3f}, {hdi[1]:.3f}]")
    print(f"\nFrequentist: '95% of such intervals contain the true mean'")
    print(f"Bayesian:    'There is 95% probability the mean is in this interval'")

compare_intervals()
```

---

## Summary

| Method | Coverage Guarantee | Adaptivity | Assumptions |
|--------|-------------------|------------|-------------|
| Bayesian credible | Conditional on model | High | Model specification |
| Conformal (split) | Marginal (finite-sample) | Low (fixed width) | Exchangeability |
| CQR | Marginal (finite-sample) | High (adaptive width) | Exchangeability |
| Bootstrap | Asymptotic | Medium | Large sample |
| Calibrated NN | Empirical | High | Calibration data |

---

## References

1. Gneiting, T. & Raftery, A. E. (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation." *JASA*.
2. Romano, Y., Patterson, E., & Candes, E. (2019). "Conformalized Quantile Regression." *NeurIPS*.
3. Angelopoulos, A. & Bates, S. (2023). "Conformal Prediction: A Gentle Introduction." *Foundations and Trends in ML*.
4. Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." *ICML*.

---

[Previous: Model Comparison](./16_Model_Comparison.md) | [Next: Capstone Applied Bayesian →](./18_Capstone_Applied_Bayesian.md)
