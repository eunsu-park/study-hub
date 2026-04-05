# 11. Bayesian Optimization

[Previous: Bayesian Time Series](./10_Bayesian_Time_Series.md) | [Next: Pyro and NumPyro](./12_Pyro_and_NumPyro.md)

---

> **Framework Note**: This lesson uses NumPy for BO fundamentals and scikit-optimize for production use.
>
> Installation: `pip install numpy scipy matplotlib scikit-optimize`

## Learning Objectives

- Understand the Bayesian optimization loop: surrogate model + acquisition function
- Implement Expected Improvement (EI) and Upper Confidence Bound (UCB) from scratch
- Apply BO to hyperparameter tuning of ML models
- Understand multi-fidelity and constrained optimization
- Use scikit-optimize for practical optimization tasks

---

## 1. The Bayesian Optimization Framework

BO is designed for optimizing expensive black-box functions. It uses a probabilistic surrogate model (typically a GP) to decide where to evaluate next.

### 1.1 The BO Loop

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize as scipy_minimize
from scipy.spatial.distance import cdist

# 1. Evaluate initial points
# 2. Fit surrogate model (GP)
# 3. Optimize acquisition function → find next point
# 4. Evaluate objective at next point
# 5. Update surrogate and repeat

def objective_function(x):
    """Expensive black-box function to minimize."""
    return -(np.sin(3*x) * np.sin(x) + 0.5 * np.sin(7*x)) + 0.1 * x**2
```

### 1.2 GP Surrogate

```python
def rbf_kernel(X1, X2, l=1.0, sigma_f=1.0):
    dists = cdist(X1.reshape(-1, 1), X2.reshape(-1, 1), 'sqeuclidean')
    return sigma_f**2 * np.exp(-0.5 * dists / l**2)

class GPSurrogate:
    def __init__(self, l=1.0, sigma_f=1.0, sigma_n=1e-6):
        self.l, self.sigma_f, self.sigma_n = l, sigma_f, sigma_n

    def fit(self, X, y):
        self.X, self.y = X, y
        K = rbf_kernel(X, X, self.l, self.sigma_f) + self.sigma_n * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)

    def predict(self, X_star):
        K_star = rbf_kernel(X_star, self.X, self.l, self.sigma_f)
        K_ss = rbf_kernel(X_star, X_star, self.l, self.sigma_f)
        mu = K_star @ self.K_inv @ self.y
        cov = K_ss - K_star @ self.K_inv @ K_star.T
        return mu, np.sqrt(np.diag(cov).clip(0))
```

---

## 2. Acquisition Functions

### 2.1 Expected Improvement (EI)

```python
def expected_improvement(X, gp, y_best, xi=0.01):
    """Expected Improvement acquisition function."""
    mu, sigma = gp.predict(X)
    sigma = np.maximum(sigma, 1e-8)
    Z = (y_best - mu - xi) / sigma
    ei = (y_best - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei

def upper_confidence_bound(X, gp, beta=2.0):
    """UCB acquisition (for minimization: LCB)."""
    mu, sigma = gp.predict(X)
    return -(mu - beta * sigma)  # negative because we minimize

def probability_of_improvement(X, gp, y_best, xi=0.01):
    """Probability of Improvement acquisition function."""
    mu, sigma = gp.predict(X)
    sigma = np.maximum(sigma, 1e-8)
    Z = (y_best - mu - xi) / sigma
    return norm.cdf(Z)
```

### 2.2 Full BO Loop

```python
def bayesian_optimization(objective, bounds, n_init=5, n_iter=20, acq='ei'):
    """Complete Bayesian optimization loop."""
    # Initial random evaluations
    X = np.random.uniform(bounds[0], bounds[1], n_init)
    y = np.array([objective(x) for x in X])

    gp = GPSurrogate(l=0.5, sigma_f=2.0, sigma_n=0.01)
    history = list(zip(X, y))

    for i in range(n_iter):
        gp.fit(X, y)
        y_best = y.min()

        # Optimize acquisition function
        x_candidates = np.linspace(bounds[0], bounds[1], 1000)
        if acq == 'ei':
            acq_values = expected_improvement(x_candidates, gp, y_best)
        elif acq == 'ucb':
            acq_values = upper_confidence_bound(x_candidates, gp)
        else:
            acq_values = probability_of_improvement(x_candidates, gp, y_best)

        x_next = x_candidates[np.argmax(acq_values)]
        y_next = objective(x_next)

        X = np.append(X, x_next)
        y = np.append(y, y_next)
        history.append((x_next, y_next))

        if (i + 1) % 5 == 0:
            print(f"Iter {i+1}: x={x_next:.3f}, y={y_next:.3f}, best={y.min():.3f}")

    best_idx = np.argmin(y)
    return X[best_idx], y[best_idx], history

x_opt, y_opt, history = bayesian_optimization(objective_function, bounds=(-3, 3))
print(f"\nOptimum: x={x_opt:.4f}, f(x)={y_opt:.4f}")
```

---

## 3. Visualization of BO Process

```python
def plot_bo_step(gp, X_train, y_train, objective, bounds, acq_fn, step):
    """Visualize one step of Bayesian optimization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    x_plot = np.linspace(bounds[0], bounds[1], 500)
    y_true = np.array([objective(x) for x in x_plot])
    mu, sigma = gp.predict(x_plot)

    # Surrogate model
    ax1.plot(x_plot, y_true, 'k--', alpha=0.3, label='True function')
    ax1.plot(x_plot, mu, 'b-', linewidth=2, label='GP mean')
    ax1.fill_between(x_plot, mu - 2*sigma, mu + 2*sigma, alpha=0.2, label='±2σ')
    ax1.scatter(X_train, y_train, c='red', s=80, zorder=5, label='Observations')
    ax1.legend()
    ax1.set_title(f"BO Step {step}: Surrogate Model")
    ax1.set_ylabel("f(x)")

    # Acquisition function
    y_best = y_train.min()
    ei = expected_improvement(x_plot, gp, y_best)
    ax2.plot(x_plot, ei, 'g-', linewidth=2)
    ax2.fill_between(x_plot, 0, ei, alpha=0.2, color='green')
    x_next = x_plot[np.argmax(ei)]
    ax2.axvline(x_next, color='r', linestyle='--', label=f'Next: x={x_next:.2f}')
    ax2.legend()
    ax2.set_title("Expected Improvement")
    ax2.set_xlabel("x")
    ax2.set_ylabel("EI(x)")

    plt.tight_layout()
    plt.savefig(f"bo_step_{step}.png", dpi=100)
    plt.show()
```

---

## 4. Hyperparameter Tuning with BO

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

# Create dataset
X_data, y_data = make_classification(n_samples=500, n_features=20,
                                      n_informative=10, random_state=42)

def rf_objective(params):
    """Objective: negative cross-validation accuracy."""
    n_estimators = int(params[0])
    max_depth = int(params[1])
    min_samples_split = int(params[2])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
    )
    scores = cross_val_score(clf, X_data, y_data, cv=5, scoring='accuracy')
    return -scores.mean()  # minimize negative accuracy

# Using scikit-optimize
from skopt import gp_minimize
from skopt.space import Integer

result = gp_minimize(
    rf_objective,
    dimensions=[
        Integer(10, 300, name='n_estimators'),
        Integer(2, 30, name='max_depth'),
        Integer(2, 20, name='min_samples_split'),
    ],
    n_calls=30,
    n_initial_points=10,
    random_state=42,
    acq_func='EI',
)

print(f"Best accuracy: {-result.fun:.4f}")
print(f"Best params: n_estimators={result.x[0]}, max_depth={result.x[1]}, "
      f"min_samples_split={result.x[2]}")
```

---

## 5. Multi-Fidelity Optimization

```python
# Concept: use cheaper approximations to guide expensive evaluations
# e.g., train with fewer epochs, smaller dataset, or lower resolution

def multi_fidelity_objective(x, fidelity=1.0):
    """Objective with controllable fidelity (0 to 1)."""
    true_val = objective_function(x)
    noise = np.random.normal(0, 0.5 * (1 - fidelity))  # less noise at higher fidelity
    bias = 0.3 * (1 - fidelity)  # slight bias at low fidelity
    return true_val + noise + bias

# Strategy: explore with low fidelity, exploit with high fidelity
# This is the basis of methods like BOHB, Hyperband, and Fabolas
```

---

## 6. Constrained Bayesian Optimization

```python
def constrained_bo(objective, constraint, bounds, n_init=5, n_iter=20):
    """BO with a probabilistic constraint: P(c(x) <= 0) >= 0.95."""
    X = np.random.uniform(bounds[0], bounds[1], n_init)
    y = np.array([objective(x) for x in X])
    c = np.array([constraint(x) for x in X])

    gp_obj = GPSurrogate(l=0.5, sigma_f=2.0, sigma_n=0.01)
    gp_con = GPSurrogate(l=0.5, sigma_f=1.0, sigma_n=0.01)

    for i in range(n_iter):
        gp_obj.fit(X, y)
        gp_con.fit(X, c)

        x_cand = np.linspace(bounds[0], bounds[1], 1000)
        ei = expected_improvement(x_cand, gp_obj, y[c <= 0].min() if any(c <= 0) else y.min())

        # Probability of feasibility
        mu_c, sigma_c = gp_con.predict(x_cand)
        p_feasible = norm.cdf(-mu_c / np.maximum(sigma_c, 1e-8))

        # Constrained EI = EI * P(feasible)
        constrained_ei = ei * p_feasible

        x_next = x_cand[np.argmax(constrained_ei)]
        X = np.append(X, x_next)
        y = np.append(y, objective(x_next))
        c = np.append(c, constraint(x_next))

    feasible = c <= 0
    if any(feasible):
        best_idx = np.argmin(y[feasible])
        return X[feasible][best_idx], y[feasible][best_idx]
    return None, None
```

---

## 7. Batch Bayesian Optimization

```python
def batch_bo(objective, bounds, batch_size=4, n_init=5, n_batches=10):
    """Parallel BO: select multiple points per iteration."""
    X = np.random.uniform(bounds[0], bounds[1], n_init)
    y = np.array([objective(x) for x in X])

    gp = GPSurrogate(l=0.5, sigma_f=2.0, sigma_n=0.01)

    for batch in range(n_batches):
        gp.fit(X, y)
        batch_points = []

        for j in range(batch_size):
            x_cand = np.linspace(bounds[0], bounds[1], 1000)
            ei = expected_improvement(x_cand, gp, y.min())
            x_next = x_cand[np.argmax(ei)]
            batch_points.append(x_next)

            # "Hallucinate" observation at GP mean for diversity
            mu_hall, _ = gp.predict(np.array([x_next]))
            X = np.append(X, x_next)
            y = np.append(y, mu_hall[0])
            gp.fit(X, y)

        # Actually evaluate the batch
        X = X[:-(batch_size)]  # remove hallucinated
        y = y[:-(batch_size)]
        for x in batch_points:
            X = np.append(X, x)
            y = np.append(y, objective(x))

    return X[np.argmin(y)], y.min()
```

---

## 8. Thompson Sampling for BO

```python
def thompson_sampling_bo(objective, bounds, n_init=5, n_iter=20):
    """BO using Thompson Sampling acquisition."""
    X = np.random.uniform(bounds[0], bounds[1], n_init)
    y = np.array([objective(x) for x in X])
    gp = GPSurrogate(l=0.5, sigma_f=2.0, sigma_n=0.01)

    for i in range(n_iter):
        gp.fit(X, y)
        x_cand = np.linspace(bounds[0], bounds[1], 500)
        mu, sigma = gp.predict(x_cand)

        # Draw one sample from the GP posterior
        K_star = rbf_kernel(x_cand, x_cand, gp.l, gp.sigma_f)
        K_star += 1e-6 * np.eye(len(x_cand))
        L = np.linalg.cholesky(K_star)
        # Approximate: just use mu + sigma * z
        f_sample = mu + sigma * np.random.randn(len(x_cand))

        x_next = x_cand[np.argmin(f_sample)]
        X = np.append(X, x_next)
        y = np.append(y, objective(x_next))

    return X[np.argmin(y)], y.min()
```

---

## Summary

| Acquisition Function | Behavior | Best For |
|---------------------|----------|----------|
| Expected Improvement (EI) | Balances exploration/exploitation | General purpose |
| Upper Confidence Bound (UCB) | Explicit exploration parameter β | Theoretical guarantees |
| Probability of Improvement (PI) | Greedy; can get stuck | Final refinement |
| Thompson Sampling | Sample-based; naturally parallel | Batch optimization |

| Method | Scaling | Use Case |
|--------|---------|----------|
| Standard BO | O(n³) per step | < 100 evaluations |
| Sparse GP BO | O(nm²) | 100-1000 evaluations |
| Multi-fidelity | Cheap + expensive | Training ML models |
| Batch BO | Parallel evaluations | Cloud/cluster computing |

---

## References

1. Shahriari, B., et al. (2016). "Taking the Human Out of the Loop: A Review of Bayesian Optimization." *Proc. IEEE*.
2. Snoek, J., Larochelle, H., & Adams, R. (2012). "Practical Bayesian Optimization of ML Algorithms." *NeurIPS*.
3. Frazier, P. I. (2018). "A Tutorial on Bayesian Optimization." arXiv:1807.02811.
4. Garnett, R. (2023). *Bayesian Optimization*. Cambridge University Press.

---

[Previous: Bayesian Time Series](./10_Bayesian_Time_Series.md) | [Next: Pyro and NumPyro →](./12_Pyro_and_NumPyro.md)
