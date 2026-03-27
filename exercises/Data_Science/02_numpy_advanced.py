"""
Exercises for Lesson 02: NumPy Advanced
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np


# === Exercise 1: Linear Regression ===
# Problem: Use least squares to find linear regression coefficients for the data.
def exercise_1():
    """Solution using np.linalg.lstsq for ordinary least squares regression."""
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2.2, 2.8, 3.6, 4.5, 5.1])

    # Construct the design matrix A = [x, 1] (Vandermonde-like)
    # Each row is [x_i, 1], so we're fitting y = m*x + c
    A = np.vstack([x, np.ones(len(x))]).T
    print("Design matrix A:")
    print(A)

    # np.linalg.lstsq minimizes ||Ax - y||^2
    # Returns: (solution, residuals, rank, singular values)
    result = np.linalg.lstsq(A, y, rcond=None)
    m, c = result[0]
    residuals = result[1]

    print(f"\nSlope (m): {m:.4f}")
    print(f"Intercept (c): {c:.4f}")
    print(f"Equation: y = {m:.4f}x + {c:.4f}")

    # Compute R-squared to measure goodness of fit
    y_pred = m * x + c
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    print(f"R-squared: {r_squared:.4f}")
    print(f"Predicted values: {y_pred}")


# === Exercise 2: Eigenvalue Decomposition of Covariance Matrix ===
# Problem: Calculate the covariance matrix of 3-variable data and perform
#          eigenvalue decomposition.
def exercise_2():
    """Solution demonstrating PCA-like analysis via eigendecomposition."""
    # Generate correlated data: columns 0 and 1 are strongly correlated
    np.random.seed(42)
    data = np.random.randn(100, 3)
    data[:, 1] = data[:, 0] * 2 + np.random.randn(100) * 0.1  # strong correlation

    # Covariance matrix captures pairwise linear relationships
    # np.cov expects variables as rows, so we transpose
    cov_matrix = np.cov(data.T)
    print("Covariance matrix:")
    print(np.round(cov_matrix, 4))

    # Eigenvalue decomposition: Sigma = V * diag(lambda) * V^T
    # Eigenvalues indicate variance explained along each eigenvector direction
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort by descending eigenvalue (largest variance first)
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    print(f"\nEigenvalues (sorted): {eigenvalues}")
    print(f"\nEigenvectors (columns):\n{np.round(eigenvectors, 4)}")

    # Proportion of variance explained by each component
    # This is the core idea behind PCA: how much variance each direction captures
    total_var = np.sum(eigenvalues)
    explained_ratio = eigenvalues / total_var
    print(f"\nVariance explained ratio: {np.round(explained_ratio, 4)}")
    print(f"Cumulative: {np.round(np.cumsum(explained_ratio), 4)}")

    # The first eigenvalue should dominate because X0 and X1 are highly correlated
    # Their combined variance collapses mostly into one principal component
    print(f"\nFirst PC explains {explained_ratio[0]*100:.1f}% of total variance")


# === Exercise 3: Monte Carlo Simulation ===
# Problem: Estimate the area of a circle (pi) using random numbers.
def exercise_3():
    """Solution using Monte Carlo integration to estimate pi.

    Geometric reasoning:
    - A unit circle (r=1) inscribed in a 2x2 square has area pi.
    - The square has area 4.
    - Ratio of random points inside the circle to total points approximates pi/4.
    - Therefore: pi ~ 4 * (points inside circle) / (total points)
    """
    n = 1_000_000
    rng = np.random.default_rng(42)

    # Generate uniform random points in [-1, 1] x [-1, 1]
    x = rng.uniform(-1, 1, n)
    y = rng.uniform(-1, 1, n)

    # A point (x, y) is inside the unit circle if x^2 + y^2 <= 1
    inside = (x**2 + y**2) <= 1
    pi_estimate = 4 * inside.sum() / n

    print(f"Number of samples: {n:,}")
    print(f"Points inside circle: {inside.sum():,}")
    print(f"pi estimate: {pi_estimate:.6f}")
    print(f"Actual pi:   {np.pi:.6f}")
    print(f"Error:       {abs(pi_estimate - np.pi):.6f}")

    # Demonstrate convergence: the estimate improves with more samples
    # Standard error of Monte Carlo estimate ~ 1/sqrt(n)
    print("\nConvergence with increasing sample size:")
    for n_sub in [100, 1000, 10000, 100000, 1000000]:
        x_sub = rng.uniform(-1, 1, n_sub)
        y_sub = rng.uniform(-1, 1, n_sub)
        inside_sub = (x_sub**2 + y_sub**2) <= 1
        est = 4 * inside_sub.sum() / n_sub
        print(f"  n={n_sub:>9,}: pi ~ {est:.6f}  (error={abs(est - np.pi):.6f})")


if __name__ == "__main__":
    print("=== Exercise 1: Linear Regression ===")
    exercise_1()
    print("\n=== Exercise 2: Eigenvalue Decomposition ===")
    exercise_2()
    print("\n=== Exercise 3: Monte Carlo Simulation ===")
    exercise_3()
    print("\nAll exercises completed!")
