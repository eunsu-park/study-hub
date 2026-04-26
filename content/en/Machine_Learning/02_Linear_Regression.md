# Linear Regression

**Previous**: [ML Overview](./01_ML_Overview.md) | **Next**: [Logistic Regression](./03_Logistic_Regression.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the mathematical formulation of simple and multiple linear regression
2. Implement linear regression using both the analytical (OLS) solution and gradient descent
3. Compare batch, stochastic, and mini-batch gradient descent approaches
4. Describe how Ridge (L2), Lasso (L1), and Elastic Net regularization prevent overfitting
5. Apply polynomial regression to model nonlinear relationships
6. Calculate and interpret regression evaluation metrics (MSE, RMSE, MAE, R-squared, MAPE)
7. Distinguish when to use Ridge versus Lasso based on feature characteristics

---

Linear regression is the most basic regression algorithm that predicts continuous values, and it remains one of the most widely used models in practice. By modeling the linear relationship between input and output variables, it provides an interpretable baseline that every practitioner should master before tackling more complex algorithms.

---

## Theory & Principles

The scikit-learn API hides four very different mathematical objects behind one verb (`.fit()`): the closed-form least-squares solution, gradient-based iterative solvers, and the L2 / L1 / Elastic Net regularizers. To choose between them you need to know what each one is actually solving and where it breaks.

### A. Ordinary Least Squares: a Closed-Form Solution

Stack `N` observations into the design matrix `X ∈ ℝ^{N×p}` (with a leading column of ones for the intercept) and target vector `y ∈ ℝ^N`. The OLS objective is

```
L(β) = ‖y - Xβ‖² = (y - Xβ)ᵀ(y - Xβ)
```

Take the gradient with respect to `β`, set it to zero:

```
∇_β L = -2 Xᵀ(y - Xβ) = 0
   ⟹  XᵀX β = Xᵀy
   ⟹  β̂ = (XᵀX)⁻¹ Xᵀy            ← normal equations
```

The Hessian `2 XᵀX` is positive semi-definite, so the critical point is a global minimum. This is one of the few ML algorithms with an *exact* solution — no iterations, no learning rate, no convergence to worry about.

The catch is the inverse `(XᵀX)⁻¹`. It exists only when `XᵀX` has full rank — i.e., when the columns of `X` are linearly independent. Two ways this fails: (1) you have more features than samples (`p > N`), and (2) two features are perfectly correlated. In both cases the system has infinitely many least-squares solutions and OLS becomes ill-defined. Numerically, even *near*-singularity blows up the variance of `β̂`.

Computational cost is `O(p² N + p³)` — fine for thousands of features, infeasible for millions. Below that ceiling, OLS is the optimum starting point: deterministic, reproducible, calibrated.

### B. Gradient Descent: Trading Exactness for Scalability

When `p` or `N` is too large for the normal equations, you minimize the same loss iteratively. The update rule is

```
β_{t+1} = β_t - η · ∇_β L(β_t) = β_t + (2η/N) · Xᵀ(y - X β_t)
```

The three flavours differ only in *which* gradient you use at each step:

- **Batch GD**: gradient over the full `N` examples. Smooth descent, expensive per step.
- **Stochastic GD (SGD)**: gradient on one random example. Cheap per step, noisy trajectory — the noise can actually *help* escape sharp curvature regions.
- **Mini-batch GD**: gradient over `B` examples (typical `B = 32-512`). Hits a sweet spot: enough averaging to be stable, small enough to run in vectorized hardware.

For the convex linear regression loss, all three converge to the *same* OLS optimum given a small enough learning rate. The difference is wall-clock cost, not the answer. Step-size choice matters: too large diverges, too small wastes compute. Practical defaults (Adam, learning-rate schedules) automate this for you.

### C. Regularization: Constraining the Solution

When `XᵀX` is near-singular or `p` is comparable to `N`, OLS has high variance: tiny changes to `y` produce large swings in `β̂`. Regularization adds a penalty that pulls `β` toward zero, trading a bit of bias for a lot of variance reduction.

#### C.1 Ridge (L2): the closed form survives

```
β̂_ridge = argmin_β  ‖y - Xβ‖² + λ ‖β‖²₂
        = (XᵀX + λI)⁻¹ Xᵀy
```

Adding `λI` makes the matrix invertible no matter what — this single fact is half the reason Ridge exists. As `λ → 0` you recover OLS; as `λ → ∞` all coefficients shrink to zero. Coefficients shrink *proportionally* but never reach zero, so Ridge keeps every feature in the model.

#### C.2 Lasso (L1): why it produces sparsity

```
β̂_lasso = argmin_β  ‖y - Xβ‖² + λ ‖β‖₁
```

The L1 penalty is non-differentiable at `β_j = 0`. Use the subdifferential:

```
∂|β_j| = { sign(β_j)         if β_j ≠ 0
         { [-1, +1]           if β_j = 0
```

The optimality condition for coordinate `j` reads `(Xᵀ(y - Xβ))_j ∈ λ · ∂|β_j|`. When the unpenalized residual correlation `|(Xᵀ(y - Xβ))_j|` is below `λ`, the only way to satisfy the inclusion is to set `β_j = 0` exactly. Geometrically, the L1 ball has corners on the axes — and constrained least-squares solutions tend to land on those corners. The L2 ball is round and has no corners, so Ridge cannot produce exact zeros.

This is *the* reason to choose Lasso: automatic feature selection. The price is no closed form (you need coordinate descent or proximal gradient methods) and instability when features are highly correlated — Lasso arbitrarily picks one of a correlated group and zeros the rest.

#### C.3 Elastic Net: convex combination

```
β̂_en = argmin_β  ‖y - Xβ‖² + λ [α ‖β‖₁ + (1-α) ‖β‖²₂]
```

Mixes L1 and L2 with mixing weight `α ∈ [0, 1]`. Inherits Lasso's sparsity and Ridge's stability under correlated features (it tends to keep correlated features as a group rather than picking one). When `α = 1` it reduces to Lasso; `α = 0` to Ridge.

### D. Choosing the Right Tool

| Situation | Best choice | Why |
|-----------|-------------|-----|
| Small `p`, well-conditioned `XᵀX` | OLS | Exact, free |
| Many correlated features, want all kept | Ridge | Stable shrinkage, closed form |
| `p > N` or want feature selection | Lasso | Sparsity from L1 corners |
| Correlated features + want sparsity | Elastic Net | Group selection |
| `N > 10⁶` | SGD / mini-batch | Normal equations infeasible |

The choice is not stylistic — it follows from the conditioning of `XᵀX` and what you want from the coefficient vector.

### From Theory to the Code Below

- Section 1's `LinearRegression().fit(X, y)` solves the normal equations from (A) directly.
- Section 2's `SGDRegressor` loop is the gradient-descent recursion from (B).
- Section 3 exposes `Ridge`, `Lasso`, and `ElasticNet` — the three regularizers from (C). The `alpha` parameter in scikit-learn is the `λ` in our formulas; the `l1_ratio` parameter is the `α` of Elastic Net.
- The polynomial features in Section 4 do not change any of this — they only expand `X` before fitting, so all the same mathematics applies in the higher-dimensional design matrix.

---

## 1. Simple Linear Regression

### 1.1 Concept

Predict dependent variable (y) using one independent variable (X).

```
y = β₀ + β₁x + ε

- β₀: intercept
- β₁: slope
- ε: error term
```

### 1.2 Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

# Train model
model = LinearRegression()
model.fit(X, y)

# Check coefficients
print(f"Intercept (β₀): {model.intercept_[0]:.4f}")
print(f"Slope (β₁): {model.coef_[0][0]:.4f}")

# Predict
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)
print(f"\nPredictions: X=0 → y={y_pred[0][0]:.2f}, X=2 → y={y_pred[1][0]:.2f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7, label='Data')
plt.plot(X_new, y_pred, 'r-', linewidth=2, label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()
```

### 1.3 Ordinary Least Squares (OLS)

```python
# OLS: Minimize residual sum of squares (RSS)
# RSS = Σ(yᵢ - ŷᵢ)²

# Analytical solution
X_b = np.c_[np.ones((100, 1)), X]  # Add bias
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print(f"Analytical solution:")
print(f"θ₀ = {theta_best[0][0]:.4f}")
print(f"θ₁ = {theta_best[1][0]:.4f}")
```

---

## 2. Multiple Linear Regression

### 2.1 Concept

Predict dependent variable using multiple independent variables.

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

### 2.2 Implementation

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
print(f"Features: {diabetes.feature_names}")
print(f"Data shape: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)

print(f"\nMSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# Check coefficients
coefficients = pd.DataFrame({
    'feature': diabetes.feature_names,
    'coefficient': model.coef_
}).sort_values('coefficient', key=abs, ascending=False)
print(f"\nRegression coefficients:")
print(coefficients)
```

---

## 3. Gradient Descent

### 3.1 Batch Gradient Descent

```python
# Cost function: J(θ) = (1/2m) Σ(h(xᵢ) - yᵢ)²
# Update: θ = θ - α * ∇J(θ)

def batch_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m = len(y)
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias
    theta = np.random.randn(2, 1)  # Random initialization

    cost_history = []

    for iteration in range(n_iterations):
        gradients = (1/m) * X_b.T @ (X_b @ theta - y)
        theta = theta - learning_rate * gradients

        cost = (1/(2*m)) * np.sum((X_b @ theta - y)**2)
        cost_history.append(cost)

    return theta, cost_history

# Execute
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

theta, cost_history = batch_gradient_descent(X, y, learning_rate=0.1, n_iterations=1000)

print(f"θ₀ = {theta[0][0]:.4f}")
print(f"θ₁ = {theta[1][0]:.4f}")

# Visualize cost function convergence
plt.figure(figsize=(10, 4))
plt.plot(cost_history[:100])
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Gradient Descent Convergence')
plt.show()
```

### 3.2 Stochastic Gradient Descent (SGD)

```python
from sklearn.linear_model import SGDRegressor

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), test_size=0.2)

# Scaling (required for SGD)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SGD regression
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None,
                       eta0=0.01, random_state=42)
sgd_reg.fit(X_train_scaled, y_train)

print(f"SGD intercept: {sgd_reg.intercept_[0]:.4f}")
print(f"SGD coefficient: {sgd_reg.coef_[0]:.4f}")
```

### 3.3 Mini-batch Gradient Descent

```python
def mini_batch_gradient_descent(X, y, batch_size=20, learning_rate=0.01, n_epochs=50):
    m = len(y)
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(2, 1)

    for epoch in range(n_epochs):
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for i in range(0, m, batch_size):
            xi = X_b_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            gradients = (1/len(yi)) * xi.T @ (xi @ theta - yi)
            theta = theta - learning_rate * gradients

    return theta

theta = mini_batch_gradient_descent(X, y)
print(f"Mini-batch GD result: θ₀={theta[0][0]:.4f}, θ₁={theta[1][0]:.4f}")
```

---

## 4. Regularization

Penalize model complexity to prevent overfitting.

### 4.1 Ridge Regression (L2 Regularization)

```python
from sklearn.linear_model import Ridge

# Cost function: J(θ) = MSE + α * Σθᵢ²

# Experiment with different alpha values
alphas = [0, 0.1, 1, 10, 100]

plt.figure(figsize=(12, 4))
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    y_pred = ridge.predict(X_test_scaled)
    print(f"Alpha={alpha}: R²={r2_score(y_test, y_pred):.4f}, Coef sum={sum(abs(ridge.coef_)):.4f}")
```

### 4.2 Lasso Regression (L1 Regularization)

```python
from sklearn.linear_model import Lasso

# Cost function: J(θ) = MSE + α * Σ|θᵢ|
# Feature: Sets some coefficients to zero (feature selection)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

# Check non-zero coefficients
non_zero = np.sum(lasso.coef_ != 0)
print(f"Number of non-zero coefficients: {non_zero}/{len(lasso.coef_)}")

y_pred = lasso.predict(X_test_scaled)
print(f"Lasso R²: {r2_score(y_test, y_pred):.4f}")
```

### 4.3 Elastic Net

```python
from sklearn.linear_model import ElasticNet

# Combines L1 and L2
# Cost function: J(θ) = MSE + r*α*Σ|θᵢ| + (1-r)*α*Σθᵢ²/2

elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio = r
elastic.fit(X_train_scaled, y_train)

y_pred = elastic.predict(X_test_scaled)
print(f"Elastic Net R²: {r2_score(y_test, y_pred):.4f}")
```

### 4.4 Regularization Comparison

```python
from sklearn.datasets import make_regression

# Generate data (features > samples)
X, y = make_regression(n_samples=50, n_features=100, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Compare models
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

print("Regularization method comparison:")
for name, model in models.items():
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    non_zero = np.sum(model.coef_ != 0) if hasattr(model, 'coef_') else len(model.coef_)
    print(f"{name:12}: Train R²={train_score:.3f}, Test R²={test_score:.3f}, Non-zero coefs={non_zero}")
```

---

## 5. Polynomial Regression

Model nonlinear relationships using linear regression.

```python
from sklearn.preprocessing import PolynomialFeatures

# Generate nonlinear data
np.random.seed(42)
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)

# Generate polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print(f"Original features: {X.shape}")
print(f"Polynomial features: {X_poly.shape}")
print(f"Feature names: {poly.get_feature_names_out()}")

# Apply linear regression
model = LinearRegression()
model.fit(X_poly, y)

print(f"\nCoefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Visualization
X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7)
plt.plot(X_plot, y_plot, 'r-', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression (degree=2)')
plt.show()
```

---

## 6. Regression Evaluation Metrics

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# Predictions
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

# MAE (Mean Absolute Error)
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.4f}")

# MSE (Mean Squared Error)
mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.4f}")

# RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

# R² (Coefficient of Determination)
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.4f}")

# MAPE (Mean Absolute Percentage Error)
mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape:.4f}")
```

---

## Practice Problems

### Problem 1: Simple Linear Regression
Train a linear regression model with the following data and predict the value when X=7.

```python
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([2, 4, 5, 4, 5, 7])

# Solution
model = LinearRegression()
model.fit(X, y)
prediction = model.predict([[7]])
print(f"Prediction when X=7: {prediction[0]:.2f}")
print(f"R²: {model.score(X, y):.4f}")
```

### Problem 2: Ridge vs Lasso
Compare the performance of Ridge and Lasso on diabetes data.

```python
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# Solution
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

for Model, name in [(Ridge, 'Ridge'), (Lasso, 'Lasso')]:
    model = Model(alpha=1)
    model.fit(X_train_s, y_train)
    print(f"{name} R²: {model.score(X_test_s, y_test):.4f}")
```

---

## Summary

| Method | Features | When to Use |
|--------|----------|-------------|
| Linear Regression | Basic, interpretable | Baseline model |
| Ridge (L2) | Shrinks coefficients, prevents overfitting | Multicollinearity |
| Lasso (L1) | Feature selection, sparse model | Many features |
| Elastic Net | L1+L2 combination | Correlated features |
| Polynomial Regression | Nonlinear relationships | Curved patterns |
