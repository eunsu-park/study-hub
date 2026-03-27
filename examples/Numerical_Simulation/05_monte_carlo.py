"""
Monte Carlo Simulation
Monte Carlo Methods

Numerical computation and simulation using probabilistic methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple


# =============================================================================
# 1. Estimating pi (area of a circle)
# =============================================================================
def estimate_pi(n_samples: int) -> Tuple[float, float]:
    """
    Estimate pi using the unit circle

    Scatter random points in the square [-1, 1] x [-1, 1]
    and compute the ratio that falls inside the unit circle

    Circle area / Square area = pi / 4
    -> pi = 4 * (points inside circle / total points)

    Error: O(1/sqrt(n))
    """
    # Generate points from uniform distribution
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)

    # Count points inside the circle
    inside = np.sum(x**2 + y**2 <= 1)

    pi_estimate = 4 * inside / n_samples
    std_error = 4 * np.sqrt(inside * (n_samples - inside) / n_samples**3)

    return pi_estimate, std_error


def estimate_pi_convergence(max_samples: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
    """Convergence process of pi estimation"""
    x = np.random.uniform(-1, 1, max_samples)
    y = np.random.uniform(-1, 1, max_samples)
    inside = (x**2 + y**2 <= 1).cumsum()
    n = np.arange(1, max_samples + 1)
    return n, 4 * inside / n


# =============================================================================
# 2. Monte Carlo Integration
# =============================================================================
def monte_carlo_integrate(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n_samples: int
) -> Tuple[float, float]:
    """
    1D Monte Carlo integration

    integral[a,b] f(x)dx ~ (b-a) * (1/n) * sum(f(x_i))

    Args:
        f: Integrand function
        a, b: Integration interval
        n_samples: Number of samples

    Returns:
        (estimated integral, standard error)
    """
    x = np.random.uniform(a, b, n_samples)
    fx = f(x)

    integral = (b - a) * np.mean(fx)
    variance = np.var(fx)
    std_error = (b - a) * np.sqrt(variance / n_samples)

    return integral, std_error


def monte_carlo_integrate_nd(
    f: Callable[[np.ndarray], float],
    bounds: list,
    n_samples: int
) -> Tuple[float, float]:
    """
    Multidimensional Monte Carlo integration

    Args:
        f: Multivariate function f(x) where x is an array
        bounds: [(a1,b1), (a2,b2), ...] bounds for each dimension
        n_samples: Number of samples
    """
    dim = len(bounds)
    volume = np.prod([b - a for a, b in bounds])

    # Uniform sampling in each dimension
    samples = np.array([
        np.random.uniform(a, b, n_samples)
        for a, b in bounds
    ]).T  # shape: (n_samples, dim)

    values = np.array([f(x) for x in samples])

    integral = volume * np.mean(values)
    std_error = volume * np.std(values) / np.sqrt(n_samples)

    return integral, std_error


# =============================================================================
# 3. Importance Sampling
# =============================================================================
def importance_sampling(
    f: Callable[[np.ndarray], np.ndarray],
    g: Callable[[np.ndarray], np.ndarray],
    sample_g: Callable[[int], np.ndarray],
    n_samples: int
) -> Tuple[float, float]:
    """
    Importance Sampling

    Goal: Estimate integral f(x)dx more efficiently

    Using g(x) as the importance distribution:
    integral f(x)dx = integral (f(x)/g(x))g(x)dx = E_g[f(x)/g(x)]

    Variance reduction: choose g(x) similar to f(x)

    Args:
        f: Integrand * original distribution
        g: Importance distribution PDF
        sample_g: Function to generate samples from g
        n_samples: Number of samples
    """
    x = sample_g(n_samples)
    weights = f(x) / g(x)

    integral = np.mean(weights)
    std_error = np.std(weights) / np.sqrt(n_samples)

    return integral, std_error


# =============================================================================
# 4. Random Walk Simulation
# =============================================================================
def random_walk_1d(n_steps: int, n_walks: int = 1) -> np.ndarray:
    """
    1D Random Walk

    Move +1 or -1 at each step
    """
    steps = np.random.choice([-1, 1], size=(n_walks, n_steps))
    positions = np.cumsum(steps, axis=1)
    return positions


def random_walk_2d(n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """2D Random Walk"""
    angles = np.random.uniform(0, 2*np.pi, n_steps)
    dx = np.cos(angles)
    dy = np.sin(angles)
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    return x, y


# =============================================================================
# 5. Option Pricing (Black-Scholes Monte Carlo)
# =============================================================================
def black_scholes_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_simulations: int,
    option_type: str = 'call'
) -> Tuple[float, float]:
    """
    Monte Carlo estimation of European option price

    Geometric Brownian Motion:
    S_T = S_0 * exp((r - sigma^2/2)T + sigma*sqrt(T) * Z)

    Args:
        S0: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free interest rate
        sigma: Volatility
        n_simulations: Number of simulations
        option_type: 'call' or 'put'
    """
    # Stock price simulation
    Z = np.random.standard_normal(n_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Payoff computation
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:  # put
        payoffs = np.maximum(K - ST, 0)

    # Discount to present value
    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)

    return price, std_error


def black_scholes_analytical(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call'
) -> float:
    """Black-Scholes analytical solution (for comparison)"""
    from scipy.stats import norm

    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    return price


# =============================================================================
# 6. Buffon's Needle Problem
# =============================================================================
def buffon_needle(
    needle_length: float,
    line_spacing: float,
    n_drops: int
) -> Tuple[float, float]:
    """
    Estimate pi using Buffon's needle problem

    Probability of crossing a line when dropping a needle between parallel lines
    = 2L / (pi * D)

    -> pi = 2L * n_drops / (D * crossings)

    Args:
        needle_length: Needle length (L)
        line_spacing: Line spacing (D), L <= D
        n_drops: Number of needle drops
    """
    if needle_length > line_spacing:
        raise ValueError("Needle length must be less than line spacing")

    # Position of needle center (0 ~ D/2)
    y_center = np.random.uniform(0, line_spacing/2, n_drops)

    # Needle angle (0 ~ pi)
    theta = np.random.uniform(0, np.pi, n_drops)

    # Check if needle tip crosses a line
    # Change in y coordinate of needle tip: (L/2) * sin(theta)
    crosses = y_center <= (needle_length/2) * np.sin(theta)
    n_crossings = np.sum(crosses)

    if n_crossings == 0:
        return float('inf'), float('inf')

    pi_estimate = 2 * needle_length * n_drops / (line_spacing * n_crossings)

    return pi_estimate, 1 / np.sqrt(n_crossings)  # Approximate error


# =============================================================================
# Visualization
# =============================================================================
def plot_monte_carlo_examples():
    """Monte Carlo example visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Pi estimation
    ax = axes[0, 0]
    n = 10000
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    inside = x**2 + y**2 <= 1

    ax.scatter(x[inside], y[inside], c='blue', s=1, alpha=0.5)
    ax.scatter(x[~inside], y[~inside], c='red', s=1, alpha=0.5)
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    pi_est = 4 * np.sum(inside) / n
    ax.set_title(f'Pi estimate: {pi_est:.4f} (n={n})')

    # 2. Pi estimation convergence
    ax = axes[0, 1]
    n, estimates = estimate_pi_convergence(50000)
    ax.semilogx(n, estimates, 'b-', alpha=0.7)
    ax.axhline(y=np.pi, color='r', linestyle='--', label=f'pi = {np.pi:.6f}')
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Pi estimate')
    ax.set_title('Pi Estimation Convergence')
    ax.legend()
    ax.grid(True)

    # 3. Random walk
    ax = axes[1, 0]
    for _ in range(5):
        x, y = random_walk_2d(1000)
        ax.plot(np.concatenate([[0], x]), np.concatenate([[0], y]), alpha=0.7)
    ax.plot(0, 0, 'ko', markersize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('2D Random Walk (5 paths)')
    ax.axis('equal')
    ax.grid(True)

    # 4. Option price convergence
    ax = axes[1, 1]
    S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    exact = black_scholes_analytical(S0, K, T, r, sigma)

    ns = np.logspace(2, 5, 20).astype(int)
    estimates = []
    errors = []
    for n in ns:
        est, err = black_scholes_mc(S0, K, T, r, sigma, n)
        estimates.append(est)
        errors.append(err)

    ax.errorbar(ns, estimates, yerr=errors, fmt='o-', capsize=3)
    ax.axhline(y=exact, color='r', linestyle='--', label=f'Analytical: {exact:.4f}')
    ax.set_xscale('log')
    ax.set_xlabel('Number of simulations')
    ax.set_ylabel('Option price')
    ax.set_title('Call Option Price Convergence')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Numerical_Simulation/examples/monte_carlo.png', dpi=150)
    plt.close()
    print("Graph saved: monte_carlo.png")


# =============================================================================
# Test
# =============================================================================
def main():
    print("=" * 60)
    print("Monte Carlo Simulation")
    print("=" * 60)

    np.random.seed(42)

    # 1. Pi estimation
    print("\n[1] Pi Estimation (area of circle)")
    print("-" * 40)
    for n in [1000, 10000, 100000, 1000000]:
        pi_est, std_err = estimate_pi(n)
        print(f"n={n:>8}: pi ~ {pi_est:.6f} +/- {std_err:.6f}, "
              f"error: {abs(pi_est - np.pi):.6f}")

    # 2. Monte Carlo integration
    print("\n[2] Monte Carlo Integration")
    print("-" * 40)

    # integral[0,1] x^2 dx = 1/3
    f1 = lambda x: x**2
    integral, error = monte_carlo_integrate(f1, 0, 1, 100000)
    print(f"integral x^2 dx [0,1] = {integral:.6f} +/- {error:.6f} (exact: 0.333333)")

    # integral[0,pi] sin(x) dx = 2
    f2 = lambda x: np.sin(x)
    integral, error = monte_carlo_integrate(f2, 0, np.pi, 100000)
    print(f"integral sin(x)dx [0,pi] = {integral:.6f} +/- {error:.6f} (exact: 2.0)")

    # 3. Multidimensional integration
    print("\n[3] Multidimensional Integration")
    print("-" * 40)

    # Volume of unit sphere: 4*pi/3 ~ 4.189
    def sphere_indicator(x):
        return 1 if np.sum(x**2) <= 1 else 0

    volume, error = monte_carlo_integrate_nd(sphere_indicator, [(-1,1)]*3, 100000)
    print(f"Unit sphere volume = {volume:.4f} +/- {error:.4f} (exact: {4*np.pi/3:.4f})")

    # 4. Option pricing
    print("\n[4] European Call Option Price")
    print("-" * 40)

    S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    exact = black_scholes_analytical(S0, K, T, r, sigma)
    mc_price, mc_error = black_scholes_mc(S0, K, T, r, sigma, 100000)

    print(f"Black-Scholes analytical: {exact:.4f}")
    print(f"Monte Carlo (n=100000): {mc_price:.4f} +/- {mc_error:.4f}")
    print(f"Error: {abs(mc_price - exact):.4f}")

    # 5. Buffon's needle
    print("\n[5] Buffon's Needle Problem")
    print("-" * 40)

    pi_est, _ = buffon_needle(1, 2, 100000)
    print(f"Pi estimate (L=1, D=2, n=100000): {pi_est:.6f}")

    # 6. Random walk statistics
    print("\n[6] 1D Random Walk Statistics")
    print("-" * 40)

    n_steps = 1000
    n_walks = 10000
    positions = random_walk_1d(n_steps, n_walks)
    final_positions = positions[:, -1]

    print(f"After {n_steps} steps ({n_walks} walks):")
    print(f"  Mean position: {np.mean(final_positions):.2f} (theory: 0)")
    print(f"  Std deviation: {np.std(final_positions):.2f} (theory: {np.sqrt(n_steps):.2f})")

    # Visualization
    try:
        plot_monte_carlo_examples()
    except Exception as e:
        print(f"Graph generation failed: {e}")

    print("\n" + "=" * 60)
    print("Monte Carlo Methods Summary")
    print("=" * 60)
    print("""
    Advantages:
    - Convergence rate maintained in high dimensions (avoids curse of dimensionality)
    - Easy to apply to complex domains/conditions
    - Simple implementation

    Disadvantages:
    - Slow convergence: O(1/sqrt(n))
    - Stochastic -> uncertainty in results
    - Requires many samples

    Variance reduction techniques:
    - Importance Sampling
    - Control Variates
    - Antithetic Variates
    - Stratified Sampling
    - Quasi-random (Low-discrepancy sequences)

    Applications:
    - Finance: Derivative pricing, risk analysis
    - Physics: Statistical mechanics, particle simulations
    - Computer Graphics: Path tracing rendering
    - Optimization: Simulated annealing, MCMC
    """)


if __name__ == "__main__":
    main()
