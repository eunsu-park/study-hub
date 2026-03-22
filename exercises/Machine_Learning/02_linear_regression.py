"""
Linear Regression - Exercise Solutions
=======================================
Lesson 02: Linear Regression

Exercises cover:
  1. Simple linear regression and prediction
  2. Ridge vs Lasso regularization comparison on diabetes data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


# ============================================================
# Exercise 1: Simple Linear Regression
# Train a linear regression model and predict the value when X=7.
# ============================================================
def exercise_1_simple_linear_regression():
    """Fit a simple linear regression and predict at X=7.

    This demonstrates the most fundamental regression task: fitting a line
    y = beta_0 + beta_1 * x to a small dataset and using it for prediction.
    R-squared measures how well the line explains the variance in y.
    """
    print("=" * 60)
    print("Exercise 1: Simple Linear Regression")
    print("=" * 60)

    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([2, 4, 5, 4, 5, 7])

    model = LinearRegression()
    model.fit(X, y)

    # Model parameters
    print(f"Intercept (beta_0): {model.intercept_:.4f}")
    print(f"Slope (beta_1):     {model.coef_[0]:.4f}")

    # Prediction at X=7
    prediction = model.predict([[7]])
    print(f"\nPrediction when X=7: {prediction[0]:.2f}")

    # R-squared: 1.0 means perfect fit, 0.0 means model is no better
    # than predicting the mean of y for every input
    print(f"R-squared: {model.score(X, y):.4f}")

    # Visualize the fit
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color="blue", s=80, label="Data points")
    X_line = np.linspace(0, 8, 100).reshape(-1, 1)
    plt.plot(X_line, model.predict(X_line), "r-", linewidth=2, label="Fit line")
    plt.scatter([[7]], prediction, color="green", s=120, zorder=5,
                marker="*", label=f"Prediction at X=7: {prediction[0]:.2f}")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Simple Linear Regression")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("02_ex1_simple_lr.png", dpi=100)
    plt.close()
    print("Plot saved: 02_ex1_simple_lr.png")


# ============================================================
# Exercise 2: Ridge vs Lasso
# Compare Ridge and Lasso on the diabetes dataset.
# ============================================================
def exercise_2_ridge_vs_lasso():
    """Compare Ridge (L2) and Lasso (L1) on the diabetes dataset.

    Key differences:
    - Ridge shrinks all coefficients toward zero but never exactly to zero.
      Use when all features contribute to the prediction.
    - Lasso can shrink some coefficients to exactly zero, performing
      automatic feature selection. Use when you suspect many irrelevant features.
    - Both help prevent overfitting by penalizing large coefficients.
    """
    print("\n" + "=" * 60)
    print("Exercise 2: Ridge vs Lasso Comparison")
    print("=" * 60)

    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, test_size=0.2, random_state=42
    )

    # StandardScaler is important for regularized models because L1/L2
    # penalties treat all features equally -- without scaling, features
    # with larger magnitudes would be penalized more heavily
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Compare across multiple alpha values to see regularization effect
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

    print(f"\n{'Model':<12} {'Alpha':<8} {'R² (test)':<12} {'Non-zero coefs':<16} {'RMSE':<10}")
    print("-" * 60)

    for alpha in alphas:
        # Ridge (L2): penalizes sum of squared coefficients
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_s, y_train)
        r2_ridge = ridge.score(X_test_s, y_test)
        nonzero_ridge = np.sum(np.abs(ridge.coef_) > 1e-6)
        rmse_ridge = np.sqrt(mean_squared_error(y_test, ridge.predict(X_test_s)))

        # Lasso (L1): penalizes sum of absolute coefficients
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_train_s, y_train)
        r2_lasso = lasso.score(X_test_s, y_test)
        nonzero_lasso = np.sum(np.abs(lasso.coef_) > 1e-6)
        rmse_lasso = np.sqrt(mean_squared_error(y_test, lasso.predict(X_test_s)))

        print(f"{'Ridge':<12} {alpha:<8.2f} {r2_ridge:<12.4f} {nonzero_ridge:<16} {rmse_ridge:<10.2f}")
        print(f"{'Lasso':<12} {alpha:<8.2f} {r2_lasso:<12.4f} {nonzero_lasso:<16} {rmse_lasso:<10.2f}")

    # Best single comparison at alpha=1
    print("\n--- Detailed comparison at alpha=1.0 ---")
    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=1.0, max_iter=10000)
    ridge.fit(X_train_s, y_train)
    lasso.fit(X_train_s, y_train)

    print(f"\nRidge R²: {ridge.score(X_test_s, y_test):.4f}")
    print(f"Lasso R²: {lasso.score(X_test_s, y_test):.4f}")

    # Coefficient comparison shows Lasso's feature selection behavior
    print(f"\nFeature coefficients:")
    print(f"{'Feature':<12} {'Ridge':>10} {'Lasso':>10}")
    print("-" * 34)
    for name, rc, lc in zip(diabetes.feature_names, ridge.coef_, lasso.coef_):
        marker = " *" if abs(lc) < 1e-6 else ""
        print(f"{name:<12} {rc:>10.3f} {lc:>10.3f}{marker}")
    print("\n* indicates Lasso zeroed this coefficient (feature selection)")


if __name__ == "__main__":
    exercise_1_simple_linear_regression()
    exercise_2_ridge_vs_lasso()
