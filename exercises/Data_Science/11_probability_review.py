"""
Exercises for Lesson 11: Probability Review
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np
from scipy import stats


# === Exercise 1: Probability Calculation ===
# Problem: The lifespan of parts follows N(1000, 100^2).
#   (a) P(X >= 900)
#   (b) Minimum time for the top 5% lifespan
def exercise_1():
    """Solution using the normal distribution CDF and PPF.

    The normal CDF gives P(X <= x). To find P(X >= x), we use the
    complement: P(X >= x) = 1 - P(X <= x).
    The PPF (percent point function) is the inverse CDF: given a probability,
    it returns the corresponding quantile.
    """
    mu, sigma = 1000, 100
    dist = stats.norm(loc=mu, scale=sigma)

    # (a) P(X >= 900) = 1 - P(X < 900) = 1 - CDF(900)
    p_at_least_900 = 1 - dist.cdf(900)
    # Equivalently: dist.sf(900) where sf = survival function = 1 - CDF
    p_at_least_900_alt = dist.sf(900)

    print("Part (a): P(X >= 900)")
    print(f"  Z-score: (900 - {mu}) / {sigma} = {(900 - mu) / sigma:.2f}")
    print(f"  P(X >= 900) = {p_at_least_900:.6f}")
    print(f"  Verification via sf: {p_at_least_900_alt:.6f}")

    # (b) Top 5% means P(X >= x) = 0.05, so P(X <= x) = 0.95
    top_5_threshold = dist.ppf(0.95)
    print(f"\nPart (b): Minimum time for top 5%")
    print(f"  95th percentile: {top_5_threshold:.2f} hours")
    print(f"  Z-score: {(top_5_threshold - mu) / sigma:.4f}")
    print(f"  Verification: P(X >= {top_5_threshold:.2f}) = {dist.sf(top_5_threshold):.4f}")


# === Exercise 2: Central Limit Theorem ===
# Problem: Draw n=50 from Poisson(lambda=4). Find P(3.5 <= X_bar <= 4.5) via CLT.
def exercise_2():
    """Solution applying the Central Limit Theorem to a Poisson distribution.

    For Poisson(lambda): mean = lambda, variance = lambda.
    By CLT, the sample mean X_bar ~ N(lambda, lambda/n) for large n.
    The standard error of the mean is sqrt(lambda/n).
    """
    lam = 4      # Poisson parameter
    n = 50       # sample size

    # CLT parameters for the sampling distribution of the mean
    mu_xbar = lam                    # E[X_bar] = lambda
    se_xbar = np.sqrt(lam / n)      # SE = sqrt(Var(X)/n) = sqrt(lambda/n)

    print(f"Poisson(lambda={lam}), sample size n={n}")
    print(f"  Sampling distribution: X_bar ~ N({mu_xbar}, {se_xbar:.4f}^2)")
    print(f"  Standard error: {se_xbar:.4f}")

    # P(3.5 <= X_bar <= 4.5) using the normal approximation
    z_lower = (3.5 - mu_xbar) / se_xbar
    z_upper = (4.5 - mu_xbar) / se_xbar
    prob = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)

    print(f"\n  Z_lower = (3.5 - {mu_xbar}) / {se_xbar:.4f} = {z_lower:.4f}")
    print(f"  Z_upper = (4.5 - {mu_xbar}) / {se_xbar:.4f} = {z_upper:.4f}")
    print(f"  P(3.5 <= X_bar <= 4.5) = {prob:.6f}")

    # Monte Carlo verification
    rng = np.random.default_rng(42)
    n_sim = 100_000
    sample_means = np.array([rng.poisson(lam, n).mean() for _ in range(n_sim)])
    mc_prob = np.mean((sample_means >= 3.5) & (sample_means <= 4.5))
    print(f"\n  Monte Carlo verification ({n_sim:,} simulations): {mc_prob:.6f}")


# === Exercise 3: Joint Distribution ===
# Problem: Given the joint PMF of X and Y, find marginal distributions,
#          test independence, and compute Cov(X, Y).
def exercise_3():
    """Solution for joint, marginal distributions and covariance.

    Joint PMF:
        P(X=0, Y=0) = 0.1,  P(X=0, Y=1) = 0.2
        P(X=1, Y=0) = 0.3,  P(X=1, Y=1) = 0.4
    """
    # Joint probability table
    joint = np.array([[0.1, 0.2],   # X=0: Y=0, Y=1
                      [0.3, 0.4]])  # X=1: Y=0, Y=1
    x_vals = np.array([0, 1])
    y_vals = np.array([0, 1])

    print("Joint PMF:")
    print(f"  P(X=0, Y=0)={joint[0,0]}, P(X=0, Y=1)={joint[0,1]}")
    print(f"  P(X=1, Y=0)={joint[1,0]}, P(X=1, Y=1)={joint[1,1]}")
    print(f"  Total: {joint.sum()}")

    # (a) Marginal distributions
    # P(X=x) = sum over y of P(X=x, Y=y)
    p_x = joint.sum(axis=1)  # sum along columns (over Y)
    p_y = joint.sum(axis=0)  # sum along rows (over X)

    print(f"\n(a) Marginal distributions:")
    print(f"  P(X=0) = {p_x[0]}, P(X=1) = {p_x[1]}")
    print(f"  P(Y=0) = {p_y[0]}, P(Y=1) = {p_y[1]}")

    # (b) Independence test: X and Y are independent iff P(X,Y) = P(X)*P(Y) for all x,y
    independent = True
    print(f"\n(b) Independence check: P(X,Y) == P(X)*P(Y)?")
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            product = p_x[i] * p_y[j]
            match = np.isclose(joint[i, j], product)
            independent = independent and match
            print(f"  P(X={x},Y={y})={joint[i,j]:.2f} vs "
                  f"P(X={x})*P(Y={y})={product:.2f} -> {'Match' if match else 'Mismatch'}")
    print(f"  Conclusion: X and Y are {'independent' if independent else 'NOT independent'}")

    # (c) Cov(X, Y) = E[XY] - E[X]*E[Y]
    E_X = np.sum(x_vals * p_x)
    E_Y = np.sum(y_vals * p_y)

    # E[XY] = sum over all (x, y) of x * y * P(X=x, Y=y)
    E_XY = 0
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            E_XY += x * y * joint[i, j]

    cov_xy = E_XY - E_X * E_Y

    print(f"\n(c) Covariance:")
    print(f"  E[X] = {E_X:.4f}")
    print(f"  E[Y] = {E_Y:.4f}")
    print(f"  E[XY] = {E_XY:.4f}")
    print(f"  Cov(X, Y) = E[XY] - E[X]*E[Y] = {E_XY:.4f} - {E_X:.4f}*{E_Y:.4f} = {cov_xy:.4f}")

    # Correlation coefficient
    var_x = np.sum((x_vals - E_X)**2 * p_x)
    var_y = np.sum((y_vals - E_Y)**2 * p_y)
    corr = cov_xy / np.sqrt(var_x * var_y)
    print(f"  Corr(X, Y) = {corr:.4f}")


if __name__ == "__main__":
    print("=== Exercise 1: Probability Calculation ===")
    exercise_1()
    print("\n=== Exercise 2: Central Limit Theorem ===")
    exercise_2()
    print("\n=== Exercise 3: Joint Distribution ===")
    exercise_3()
    print("\nAll exercises completed!")
