"""
Exercises for Lesson 20: Monte Carlo Simulation
Topic: Numerical_Simulation

Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats


# === Exercise 1: Volume of a d-dimensional Unit Sphere ===
# Problem: Estimate the volume of a d-dimensional unit sphere using Monte Carlo.
# For d=3, the exact value is 4*pi/3 ~ 4.19.

def exercise_1():
    """Estimate volume of d-dimensional unit sphere via Monte Carlo."""

    def sphere_volume_mc(d, n_samples):
        """
        Estimate volume of d-dimensional unit sphere.

        Sample uniformly from [-1, 1]^d hypercube.
        Count fraction of points inside the unit sphere.
        Volume = fraction * (2^d).
        """
        points = np.random.uniform(-1, 1, (n_samples, d))
        inside = np.sum(points**2, axis=1) <= 1
        cube_volume = 2**d
        return cube_volume * np.mean(inside)

    def exact_sphere_volume(d):
        """Exact volume of d-dimensional unit sphere: pi^(d/2) / Gamma(d/2 + 1)."""
        from math import gamma
        return np.pi**(d / 2) / gamma(d / 2 + 1)

    np.random.seed(42)
    n_samples = 100000

    print("Volume of d-dimensional Unit Sphere")
    print("=" * 60)
    print(f"{'d':<5}{'MC Estimate':<18}{'Exact':<18}{'Rel. Error':<15}")
    print("-" * 60)

    dims = [2, 3, 4, 5, 8, 10]
    mc_volumes = []
    exact_volumes = []

    for d in dims:
        v_mc = sphere_volume_mc(d, n_samples)
        v_exact = exact_sphere_volume(d)
        rel_err = abs(v_mc - v_exact) / v_exact
        mc_volumes.append(v_mc)
        exact_volumes.append(v_exact)
        print(f"{d:<5}{v_mc:<18.6f}{v_exact:<18.6f}{rel_err:<15.4%}")

    print(f"\nNote: As d increases, the sphere volume shrinks (most of the")
    print(f"hypercube volume is in the corners), making the estimate harder.")

    # Convergence analysis for d=3
    print("\nConvergence analysis for d=3:")
    N_values = [100, 1000, 10000, 100000, 1000000]
    errors = []
    for N in N_values:
        est = sphere_volume_mc(3, N)
        err = abs(est - exact_sphere_volume(3))
        errors.append(err)
        print(f"  N = {N:>8d}: estimate = {est:.6f}, error = {err:.6f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(N_values, errors, 'bo-', label='Actual error')
    ax.loglog(N_values, 5 / np.sqrt(N_values), 'r--', label='O(1/sqrt(N))')
    ax.set_xlabel('Number of samples N')
    ax.set_ylabel('|Estimate - Exact|')
    ax.set_title('MC Sphere Volume Convergence (d=3)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exercise_20_1_sphere_volume.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to exercise_20_1_sphere_volume.png")


# === Exercise 2: Asian Option Pricing ===
# Problem: Simulate the price of a path-dependent option (Asian call option).
# The payoff depends on the arithmetic average of the price path.

def exercise_2():
    """Asian call option pricing via Monte Carlo simulation."""

    np.random.seed(42)

    # Parameters
    S0 = 100       # Initial stock price
    K = 100        # Strike price
    T = 1.0        # Time to maturity (1 year)
    r = 0.05       # Risk-free rate
    sigma = 0.2    # Volatility
    n_steps = 252  # Trading days
    n_paths = 100000

    dt = T / n_steps

    # Simulate paths using geometric Brownian motion
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0

    for i in range(n_steps):
        Z = np.random.normal(0, 1, n_paths)
        S[:, i+1] = S[:, i] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    # Asian option: payoff based on arithmetic average price
    S_avg = np.mean(S[:, 1:], axis=1)
    payoffs_asian = np.maximum(S_avg - K, 0)
    price_asian = np.exp(-r * T) * np.mean(payoffs_asian)
    std_err_asian = np.exp(-r * T) * np.std(payoffs_asian) / np.sqrt(n_paths)

    # European option for comparison
    payoffs_european = np.maximum(S[:, -1] - K, 0)
    price_european = np.exp(-r * T) * np.mean(payoffs_european)
    std_err_european = np.exp(-r * T) * np.std(payoffs_european) / np.sqrt(n_paths)

    # Black-Scholes formula for European call (analytical benchmark)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bs_price = S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

    print("Asian Call Option Pricing")
    print("=" * 50)
    print(f"Parameters: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"Paths: {n_paths}, Steps: {n_steps}")
    print()
    print(f"Asian call price:    {price_asian:.4f} +/- {std_err_asian:.4f}")
    print(f"European call price: {price_european:.4f} +/- {std_err_european:.4f}")
    print(f"Black-Scholes (European): {bs_price:.4f}")
    print()
    print(f"Asian < European: {price_asian < price_european} (averaging reduces volatility)")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sample paths
    for i in range(20):
        axes[0].plot(np.arange(n_steps + 1) / n_steps, S[i], alpha=0.5, linewidth=0.5)
    axes[0].axhline(y=K, color='r', linestyle='--', label=f'Strike K={K}')
    axes[0].set_xlabel('Time (years)')
    axes[0].set_ylabel('Stock Price')
    axes[0].set_title('Sample GBM Paths')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Payoff distributions
    axes[1].hist(payoffs_asian[payoffs_asian > 0], bins=50, density=True,
                 alpha=0.7, label='Asian')
    axes[1].hist(payoffs_european[payoffs_european > 0], bins=50, density=True,
                 alpha=0.5, label='European')
    axes[1].set_xlabel('Payoff')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Payoff Distribution (non-zero)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise_20_2_asian_option.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to exercise_20_2_asian_option.png")


if __name__ == "__main__":
    print("=== Exercise 1: Volume of a d-dimensional Sphere ===")
    exercise_1()

    print("\n=== Exercise 2: Asian Option Pricing ===")
    exercise_2()

    print("\nAll exercises completed!")
