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
