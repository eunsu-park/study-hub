# Monte Carlo Simulation

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the principles of the Monte Carlo method and identify problem classes where it is advantageous over deterministic methods.
2. Implement Monte Carlo integration and analyze how the law of large numbers governs convergence and error scaling.
3. Apply Monte Carlo simulation to estimate probabilities and expectations in physical and financial models.
4. Implement Markov Chain Monte Carlo (MCMC) sampling and explain the Metropolis-Hastings algorithm.
5. Compare variance reduction techniques (importance sampling, control variates) and evaluate their effectiveness.
6. Design and run a complete Monte Carlo simulation for a real-world problem and interpret the statistical results.

---

## Overview

The Monte Carlo method is a stochastic algorithm that uses random numbers to obtain numerical results. It is applied in various fields including complex integration, optimization, and physical system simulation.

**Why This Lesson Matters:** All the methods covered so far are deterministic -- given the same inputs, they produce the same output. Monte Carlo methods embrace randomness instead, using random sampling to estimate quantities that would be prohibitively expensive to compute deterministically. Their superpower: the convergence rate $O(1/\sqrt{N})$ is independent of dimension, making Monte Carlo the method of choice for high-dimensional integrals (10+ dimensions) where grid-based methods are hopeless due to the curse of dimensionality.

---

## 1. Introduction to Monte Carlo Methods

### 1.1 History and Concepts

```python
"""
History of Monte Carlo methods:
- Developed in the 1940s by Stanislaw Ulam and John von Neumann during the Manhattan Project
- Named after the Monte Carlo Casino in Monaco
- Core idea: Solve deterministic problems through random sampling

Application areas:
- Numerical integration (high-dimensional integrals)
- Statistical physics (Ising model, molecular dynamics)
- Financial engineering (option pricing, risk analysis)
- Computer graphics (ray tracing)
- Machine learning (MCMC, Bayesian inference)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
```

### 1.2 Basic Principles

```python
def monte_carlo_principle():
    """
    Basic principle of Monte Carlo integration

    ∫f(x)dx ≈ (b-a)/N * Σf(xᵢ)

    where xᵢ are uniformly sampled from [a, b]
    """
    # Example: ∫₀¹ x² dx = 1/3

    def f(x):
        return x**2

    N_values = [100, 1000, 10000, 100000]

    print("∫₀¹ x² dx = 1/3 ≈ 0.3333...")
    print()

    for N in N_values:
        samples = np.random.uniform(0, 1, N)
        estimate = np.mean(f(samples))  # (b-a) = 1
        error = abs(estimate - 1/3)
        print(f"N = {N:6d}: estimate = {estimate:.6f}, error = {error:.6f}")

monte_carlo_principle()
```

---

## 2. Random Number Generation

### 2.1 Pseudo-random Numbers

```python
def random_number_basics():
    """NumPy random number generator basics"""

    # Set seed (for reproducibility)
    rng = np.random.default_rng(seed=42)

    # Uniform distribution [0, 1)
    uniform = rng.random(5)
    print(f"Uniform distribution: {uniform}")

    # Integer random numbers
    integers = rng.integers(1, 7, size=10)  # dice
    print(f"Dice rolls (10x): {integers}")

    # Normal distribution
    normal = rng.normal(loc=0, scale=1, size=5)
    print(f"Standard normal: {normal}")

    # Other distributions
    exponential = rng.exponential(scale=1.0, size=5)
    poisson = rng.poisson(lam=5, size=5)

    print(f"Exponential distribution: {exponential}")
    print(f"Poisson: {poisson}")

random_number_basics()
```

### 2.2 Sampling from Distributions

```python
def distribution_sampling():
    """Sampling from various probability distributions"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    n_samples = 10000

    # 1. Uniform distribution
    samples = np.random.uniform(-1, 1, n_samples)
    axes[0, 0].hist(samples, bins=50, density=True, alpha=0.7)
    axes[0, 0].set_title('Uniform Distribution U(-1, 1)')

    # 2. Normal distribution
    samples = np.random.normal(0, 1, n_samples)
    axes[0, 1].hist(samples, bins=50, density=True, alpha=0.7)
    x = np.linspace(-4, 4, 100)
    axes[0, 1].plot(x, stats.norm.pdf(x), 'r-', linewidth=2)
    axes[0, 1].set_title('Normal Distribution N(0, 1)')

    # 3. Exponential distribution
    samples = np.random.exponential(1, n_samples)
    axes[0, 2].hist(samples, bins=50, density=True, alpha=0.7)
    x = np.linspace(0, 6, 100)
    axes[0, 2].plot(x, stats.expon.pdf(x), 'r-', linewidth=2)
    axes[0, 2].set_title('Exponential Distribution Exp(1)')

    # 4. Gamma distribution
    samples = np.random.gamma(2, 1, n_samples)
    axes[1, 0].hist(samples, bins=50, density=True, alpha=0.7)
    x = np.linspace(0, 10, 100)
    axes[1, 0].plot(x, stats.gamma.pdf(x, 2), 'r-', linewidth=2)
    axes[1, 0].set_title('Gamma Distribution Γ(2, 1)')

    # 5. Beta distribution
    samples = np.random.beta(2, 5, n_samples)
    axes[1, 1].hist(samples, bins=50, density=True, alpha=0.7)
    x = np.linspace(0, 1, 100)
    axes[1, 1].plot(x, stats.beta.pdf(x, 2, 5), 'r-', linewidth=2)
    axes[1, 1].set_title('Beta Distribution Beta(2, 5)')

    # 6. Chi-square distribution
    samples = np.random.chisquare(3, n_samples)
    axes[1, 2].hist(samples, bins=50, density=True, alpha=0.7)
    x = np.linspace(0, 15, 100)
    axes[1, 2].plot(x, stats.chi2.pdf(x, 3), 'r-', linewidth=2)
    axes[1, 2].set_title('Chi-square χ²(3)')

    plt.tight_layout()
    plt.show()

distribution_sampling()
```

### 2.3 Inverse Transform Sampling

```python
def inverse_transform_sampling():
    """
    Inverse transform method for sampling from arbitrary distributions

    If U ~ Uniform(0,1),
    then X = F⁻¹(U) follows distribution with CDF F
    """

    # Example: Exponential distribution
    # CDF: F(x) = 1 - e^(-λx)
    # Inverse: F⁻¹(u) = -ln(1-u)/λ

    def sample_exponential(lam, n):
        u = np.random.uniform(0, 1, n)
        return -np.log(1 - u) / lam

    lam = 2.0
    samples = sample_exponential(lam, 10000)

    plt.figure(figsize=(10, 5))
    plt.hist(samples, bins=50, density=True, alpha=0.7, label='Inverse transform samples')

    x = np.linspace(0, 4, 100)
    plt.plot(x, lam * np.exp(-lam * x), 'r-', linewidth=2,
             label=f'Theoretical: Exp({lam})')

    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Inverse Transform Sampling: Exponential Distribution')
    plt.legend()
    plt.grid(True)
    plt.show()

inverse_transform_sampling()
```

---

## 3. Monte Carlo Integration

### 3.1 Estimating π

```python
def estimate_pi():
    """
    Estimating π using a circle

    Ratio of points inside unit circle within unit square:
    π/4 = (circle area) / (square area)
    """

    def estimate_pi_once(n_points):
        # Sample from [-1, 1] x [-1, 1] square
        x = np.random.uniform(-1, 1, n_points)
        y = np.random.uniform(-1, 1, n_points)

        # Ratio of points inside circle
        inside = x**2 + y**2 <= 1
        return 4 * np.sum(inside) / n_points

    # Convergence analysis
    N_values = np.logspace(1, 6, 20).astype(int)
    estimates = [estimate_pi_once(n) for n in N_values]
    errors = [abs(e - np.pi) for e in estimates]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Visualization (N=1000)
    n = 1000
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    inside = x**2 + y**2 <= 1

    axes[0].scatter(x[inside], y[inside], c='blue', s=1, alpha=0.5)
    axes[0].scatter(x[~inside], y[~inside], c='red', s=1, alpha=0.5)
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    axes[0].add_patch(circle)
    axes[0].set_xlim(-1.1, 1.1)
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].set_aspect('equal')
    axes[0].set_title(f'π Estimation (N={n}): {4*np.sum(inside)/n:.4f}')

    # Convergence
    axes[1].loglog(N_values, errors, 'bo-', label='Actual error')
    axes[1].loglog(N_values, 1/np.sqrt(N_values), 'r--', label='O(1/√N)')
    axes[1].set_xlabel('Number of samples N')
    axes[1].set_ylabel('|estimate - π|')
    axes[1].set_title('Convergence Rate')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    print(f"π = {np.pi:.10f}")
    print(f"Estimate (N=10⁶): {estimates[-1]:.10f}")

estimate_pi()
```

### 3.2 Multidimensional Integration

```python
def multidimensional_integration():
    """
    Monte Carlo's strength in high-dimensional integration

    Curse of dimensionality: Grid-based methods become exponentially slower
    Monte Carlo: O(1/√N) convergence rate is dimension-independent
    """

    def integrand(x):
        """n-dimensional Gaussian integral: ∫...∫ exp(-||x||²) dx"""
        return np.exp(-np.sum(x**2, axis=1))

    def mc_integrate(dim, n_samples, limits=(-3, 3)):
        """Integrate over d-dimensional hypercube"""
        # Uniform sampling
        samples = np.random.uniform(limits[0], limits[1], (n_samples, dim))
        values = integrand(samples)

        # Integral estimate
        volume = (limits[1] - limits[0]) ** dim
        estimate = volume * np.mean(values)
        std_error = volume * np.std(values) / np.sqrt(n_samples)

        return estimate, std_error

    # Theoretical value: π^(d/2)
    print("Multidimensional Gaussian integral:")
    print(f"{'Dim':<8}{'Estimate':<15}{'Theoretical':<15}{'Rel. Error':<12}")
    print("-" * 50)

    for dim in [1, 2, 3, 5, 10]:
        estimate, std_err = mc_integrate(dim, 100000)
        theoretical = np.pi ** (dim/2)
        rel_error = abs(estimate - theoretical) / theoretical

        print(f"{dim:<8}{estimate:<15.6f}{theoretical:<15.6f}{rel_error:<12.4%}")

multidimensional_integration()
```

### 3.3 Importance Sampling

```python
def importance_sampling():
    """
    Importance Sampling

    ∫f(x)dx = ∫[f(x)/g(x)]g(x)dx ≈ (1/N)Σ f(xᵢ)/g(xᵢ)

    where xᵢ ~ g(x)

    Variance decreases when g(x) is similar to f(x)
    """

    # Example: ∫₀^∞ x² * e^(-x) dx = 2 (gamma function)

    def f(x):
        return x**2 * np.exp(-x)

    n_samples = 10000

    # Method 1: Uniform sampling (truncated integral)
    # Approximate as ∫₀^10 x² * e^(-x) dx
    x_uniform = np.random.uniform(0, 10, n_samples)
    estimate_uniform = 10 * np.mean(f(x_uniform))

    # Method 2: Importance sampling (proposal distribution: exponential)
    # g(x) = e^(-x), f(x)/g(x) = x²
    x_exp = np.random.exponential(1, n_samples)
    estimate_is = np.mean(x_exp**2)  # f(x)/g(x) = x²

    print("∫₀^∞ x² * e^(-x) dx = 2")
    print(f"Uniform sampling (0~10): {estimate_uniform:.6f}")
    print(f"Importance sampling (Exp): {estimate_is:.6f}")

    # Variance comparison
    var_uniform = 10**2 * np.var(f(x_uniform)) / n_samples
    var_is = np.var(x_exp**2) / n_samples

    print(f"\nVariance comparison:")
    print(f"Uniform sampling variance: {var_uniform:.6f}")
    print(f"Importance sampling variance: {var_is:.6f}")
    print(f"Variance reduction factor: {var_uniform/var_is:.1f}x")

importance_sampling()
```

---

## 4. Stochastic Simulation

### 4.1 Random Walk

```python
def random_walk():
    """1D and 2D random walks"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1D random walk
    n_steps = 1000
    n_walks = 5

    for _ in range(n_walks):
        steps = np.random.choice([-1, 1], n_steps)
        position = np.cumsum(steps)
        axes[0].plot(position, alpha=0.7)

    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Position')
    axes[0].set_title('1D Random Walk')
    axes[0].grid(True)

    # 2D random walk
    n_steps = 5000
    directions = np.random.randint(0, 4, n_steps)
    dx = np.where(directions == 0, 1, np.where(directions == 1, -1, 0))
    dy = np.where(directions == 2, 1, np.where(directions == 3, -1, 0))

    x = np.cumsum(dx)
    y = np.cumsum(dy)

    axes[1].plot(x, y, alpha=0.7, linewidth=0.5)
    axes[1].scatter([0], [0], color='green', s=100, zorder=5, label='Start')
    axes[1].scatter([x[-1]], [y[-1]], color='red', s=100, zorder=5, label='End')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('2D Random Walk')
    axes[1].legend()
    axes[1].set_aspect('equal')

    # Mean squared displacement (diffusion)
    n_simulations = 1000
    n_steps = 500
    final_distances = []

    for _ in range(n_simulations):
        steps = np.random.choice([-1, 1], n_steps)
        final_pos = np.sum(steps)
        final_distances.append(final_pos**2)

    print(f"1D Random Walk ({n_steps} steps):")
    print(f"  Mean squared displacement: {np.mean(final_distances):.2f}")
    print(f"  Theoretical value (N): {n_steps}")

    # MSD vs time
    msd = []
    for t in range(1, n_steps + 1):
        positions = [np.sum(np.random.choice([-1, 1], t)) for _ in range(500)]
        msd.append(np.mean(np.array(positions)**2))

    axes[2].plot(range(1, n_steps + 1), msd, 'b-', alpha=0.7, label='Simulation')
    axes[2].plot(range(1, n_steps + 1), range(1, n_steps + 1), 'r--', label='⟨x²⟩ = t')
    axes[2].set_xlabel('Time t')
    axes[2].set_ylabel('⟨x²⟩')
    axes[2].set_title('Mean Squared Displacement')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

random_walk()
```

### 4.2 Brownian Motion

```python
def brownian_motion():
    """Geometric Brownian Motion"""

    # dS = μSdt + σSdW
    # S(t) = S(0) * exp((μ - σ²/2)t + σW(t))

    S0 = 100      # Initial price
    mu = 0.1      # Expected return (annual)
    sigma = 0.2   # Volatility (annual)
    T = 1.0       # 1 year
    n_steps = 252 # Number of trading days
    n_paths = 100

    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Path simulation
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    for i in range(n_steps):
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        paths[:, i+1] = paths[:, i] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW)

    for path in paths[:20]:
        axes[0].plot(t, path, alpha=0.5, linewidth=0.8)

    axes[0].set_xlabel('Time (years)')
    axes[0].set_ylabel('Price')
    axes[0].set_title('Geometric Brownian Motion Paths')
    axes[0].grid(True)

    # Final price distribution
    final_prices = paths[:, -1]

    axes[1].hist(final_prices, bins=30, density=True, alpha=0.7, label='Simulation')

    # Theoretical distribution: lognormal
    log_mean = np.log(S0) + (mu - 0.5*sigma**2)*T
    log_std = sigma * np.sqrt(T)

    x = np.linspace(50, 200, 100)
    pdf = stats.lognorm.pdf(x, log_std, scale=np.exp(log_mean))
    axes[1].plot(x, pdf, 'r-', linewidth=2, label='Theoretical (lognormal)')

    axes[1].set_xlabel('Final price')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Final Price Distribution')
    axes[1].legend()
    axes[1].grid(True)

    print(f"Geometric Brownian Motion simulation:")
    print(f"  Initial price: {S0}")
    print(f"  Mean final price: {np.mean(final_prices):.2f}")
    print(f"  Theoretical expectation: {S0 * np.exp(mu * T):.2f}")

    plt.tight_layout()
    plt.show()

brownian_motion()
```

---

## 5. Physical Systems

### 5.1 Ising Model

```python
def ising_model():
    """
    2D Ising Model: Metropolis Algorithm

    H = -J Σ sᵢsⱼ

    Spin sᵢ = ±1, J > 0 (ferromagnetic)
    """

    def calculate_energy(lattice, J=1):
        """Calculate total energy"""
        energy = 0
        N = lattice.shape[0]
        for i in range(N):
            for j in range(N):
                S = lattice[i, j]
                neighbors = (lattice[(i+1)%N, j] + lattice[(i-1)%N, j] +
                            lattice[i, (j+1)%N] + lattice[i, (j-1)%N])
                energy -= J * S * neighbors
        return energy / 2  # Correct for double counting

    def metropolis_step(lattice, beta, J=1):
        """One Metropolis algorithm step"""
        N = lattice.shape[0]
        i, j = np.random.randint(0, N, 2)

        S = lattice[i, j]
        neighbors = (lattice[(i+1)%N, j] + lattice[(i-1)%N, j] +
                    lattice[i, (j+1)%N] + lattice[i, (j-1)%N])

        dE = 2 * J * S * neighbors

        if dE <= 0 or np.random.random() < np.exp(-beta * dE):
            lattice[i, j] = -S

        return lattice

    def simulate_ising(N, T, n_steps, n_equilibrate):
        """Simulate Ising model"""
        beta = 1 / T
        lattice = np.random.choice([-1, 1], (N, N))

        magnetizations = []

        for step in range(n_steps + n_equilibrate):
            for _ in range(N * N):  # N² attempts = 1 MC step
                lattice = metropolis_step(lattice, beta)

            if step >= n_equilibrate:
                M = np.abs(np.mean(lattice))
                magnetizations.append(M)

        return lattice, np.mean(magnetizations)

    # Phase transition with temperature
    N = 20
    temperatures = np.linspace(1.0, 4.0, 20)
    magnetizations = []

    print("Simulating 2D Ising model...")
    for T in temperatures:
        _, M = simulate_ising(N, T, n_steps=100, n_equilibrate=50)
        magnetizations.append(M)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Magnetization vs temperature
    Tc = 2.269  # Critical temperature (2D Ising)
    axes[0].plot(temperatures, magnetizations, 'bo-')
    axes[0].axvline(x=Tc, color='r', linestyle='--', label=f'Tc = {Tc:.3f}')
    axes[0].set_xlabel('Temperature T')
    axes[0].set_ylabel('Magnetization |M|')
    axes[0].set_title('Magnetization vs Temperature')
    axes[0].legend()
    axes[0].grid(True)

    # Low temperature state (ordered)
    lattice_low, _ = simulate_ising(30, T=1.5, n_steps=200, n_equilibrate=100)
    axes[1].imshow(lattice_low, cmap='coolwarm', interpolation='nearest')
    axes[1].set_title(f'T = 1.5 (Low temperature, ordered)')
    axes[1].axis('off')

    # High temperature state (disordered)
    lattice_high, _ = simulate_ising(30, T=4.0, n_steps=200, n_equilibrate=100)
    axes[2].imshow(lattice_high, cmap='coolwarm', interpolation='nearest')
    axes[2].set_title(f'T = 4.0 (High temperature, disordered)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

ising_model()
```

### 5.2 Molecular Dynamics (Simple Example)

```python
def lennard_jones_mc():
    """
    Monte Carlo simulation of Lennard-Jones gas

    V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
    """

    def lj_potential(r, epsilon=1, sigma=1):
        """Lennard-Jones potential"""
        return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

    def total_energy(positions, L, epsilon=1, sigma=1):
        """Total energy (periodic boundary conditions)"""
        N = len(positions)
        energy = 0
        for i in range(N):
            for j in range(i+1, N):
                dr = positions[j] - positions[i]
                # Minimum image convention
                dr = dr - L * np.round(dr / L)
                r = np.linalg.norm(dr)
                if r < 3 * sigma:  # Cutoff
                    energy += lj_potential(r, epsilon, sigma)
        return energy

    def mc_step(positions, L, T, delta=0.1):
        """MC move attempt"""
        N = len(positions)
        beta = 1 / T

        old_E = total_energy(positions, L)

        # Select random particle and move
        i = np.random.randint(N)
        old_pos = positions[i].copy()
        positions[i] += np.random.uniform(-delta, delta, 2)

        # Periodic boundary conditions
        positions[i] = positions[i] % L

        new_E = total_energy(positions, L)
        dE = new_E - old_E

        if dE > 0 and np.random.random() > np.exp(-beta * dE):
            positions[i] = old_pos  # Reject
            return False
        return True

    # Simulation
    N = 20
    L = 5.0  # Box size
    T = 1.0

    # Initial configuration (random)
    positions = np.random.uniform(0, L, (N, 2))

    # Equilibration
    for _ in range(5000):
        mc_step(positions, L, T)

    # Sampling
    n_samples = 100
    snapshots = []
    energies = []

    for i in range(n_samples):
        for _ in range(100):  # Sample every 100 steps
            mc_step(positions, L, T)
        snapshots.append(positions.copy())
        energies.append(total_energy(positions, L))

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(positions[:, 0], positions[:, 1], s=100)
    axes[0].set_xlim(0, L)
    axes[0].set_ylim(0, L)
    axes[0].set_aspect('equal')
    axes[0].set_title(f'LJ Gas (N={N}, T={T})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    axes[1].plot(energies)
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Energy')
    axes[1].set_title('Energy Time Series')
    axes[1].grid(True)

    print(f"Average energy: {np.mean(energies):.4f}")

    plt.tight_layout()
    plt.show()

lennard_jones_mc()
```

---

## 6. Financial and Engineering Applications

### 6.1 Option Pricing

```python
def option_pricing():
    """
    Black-Scholes Monte Carlo simulation

    European call option: C = E[max(S(T) - K, 0)] * e^(-rT)
    """

    def black_scholes_call(S0, K, T, r, sigma):
        """Black-Scholes formula (analytical)"""
        d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S0 * stats.norm.cdf(d1) - K * np.exp(-r*T) * stats.norm.cdf(d2)

    def monte_carlo_call(S0, K, T, r, sigma, n_paths=100000):
        """Monte Carlo simulation"""
        # Simulate final price
        Z = np.random.normal(0, 1, n_paths)
        ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)

        # Payoff
        payoffs = np.maximum(ST - K, 0)

        # Discounted expectation
        price = np.exp(-r*T) * np.mean(payoffs)
        std_error = np.exp(-r*T) * np.std(payoffs) / np.sqrt(n_paths)

        return price, std_error

    # Parameters
    S0 = 100      # Current stock price
    K = 100       # Strike price
    T = 1.0       # Time to maturity (years)
    r = 0.05      # Risk-free rate
    sigma = 0.2   # Volatility

    # Price comparison
    bs_price = black_scholes_call(S0, K, T, r, sigma)
    mc_price, mc_error = monte_carlo_call(S0, K, T, r, sigma)

    print("European call option price:")
    print(f"  Black-Scholes formula: {bs_price:.4f}")
    print(f"  Monte Carlo: {mc_price:.4f} ± {mc_error:.4f}")

    # Convergence analysis
    n_values = [1000, 5000, 10000, 50000, 100000, 500000]
    mc_prices = []
    mc_errors = []

    for n in n_values:
        price, error = monte_carlo_call(S0, K, T, r, sigma, n)
        mc_prices.append(price)
        mc_errors.append(error)

    plt.figure(figsize=(10, 5))
    plt.errorbar(n_values, mc_prices, yerr=mc_errors, fmt='bo-', capsize=3)
    plt.axhline(y=bs_price, color='r', linestyle='--', label='Black-Scholes')
    plt.xscale('log')
    plt.xlabel('Number of simulations')
    plt.ylabel('Option price')
    plt.title('Monte Carlo Option Price Convergence')
    plt.legend()
    plt.grid(True)
    plt.show()

option_pricing()
```

### 6.2 Reliability Analysis

```python
def reliability_analysis():
    """
    System reliability Monte Carlo analysis

    Failure probability of series/parallel systems
    """

    def component_lifetime(mean_life, n_simulations):
        """Exponential distribution lifetime"""
        return np.random.exponential(mean_life, n_simulations)

    def serial_system(mean_lives, n_simulations):
        """
        Series system: system fails if any component fails
        System lifetime = min(component lifetimes)
        """
        lifetimes = np.array([component_lifetime(m, n_simulations)
                             for m in mean_lives])
        return np.min(lifetimes, axis=0)

    def parallel_system(mean_lives, n_simulations):
        """
        Parallel system: system fails when all components fail
        System lifetime = max(component lifetimes)
        """
        lifetimes = np.array([component_lifetime(m, n_simulations)
                             for m in mean_lives])
        return np.max(lifetimes, axis=0)

    # Simulation
    n_sim = 100000
    mean_lives = [100, 150, 200]  # Mean lifetime of each component

    serial_life = serial_system(mean_lives, n_sim)
    parallel_life = parallel_system(mean_lives, n_sim)

    # Result analysis
    t = 100  # Target time

    serial_reliability = np.mean(serial_life > t)
    parallel_reliability = np.mean(parallel_life > t)

    print(f"Reliability at t = {t}:")
    print(f"  Series system: {serial_reliability:.4f}")
    print(f"  Parallel system: {parallel_reliability:.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Lifetime distribution
    axes[0].hist(serial_life, bins=50, density=True, alpha=0.7, label='Series')
    axes[0].hist(parallel_life, bins=50, density=True, alpha=0.7, label='Parallel')
    axes[0].axvline(x=t, color='r', linestyle='--', label=f't={t}')
    axes[0].set_xlabel('Lifetime')
    axes[0].set_ylabel('Density')
    axes[0].set_title('System Lifetime Distribution')
    axes[0].legend()
    axes[0].grid(True)

    # Reliability function
    t_range = np.linspace(0, 500, 100)
    serial_R = [np.mean(serial_life > t) for t in t_range]
    parallel_R = [np.mean(parallel_life > t) for t in t_range]

    axes[1].plot(t_range, serial_R, 'b-', label='Series')
    axes[1].plot(t_range, parallel_R, 'r-', label='Parallel')
    axes[1].set_xlabel('Time t')
    axes[1].set_ylabel('R(t)')
    axes[1].set_title('Reliability Function')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

reliability_analysis()
```

---

## 7. Variance Reduction Techniques

### 7.1 Antithetic Variates

```python
def antithetic_variates():
    """
    Antithetic variates: Use U and 1-U together to reduce variance

    If U ~ Uniform(0,1), then 1-U has the same distribution
    Variance reduces when f(U) and f(1-U) are negatively correlated
    """

    # Example: E[e^U] where U ~ Uniform(0,1)
    # Exact value: e - 1 ≈ 1.71828

    n = 10000
    true_value = np.e - 1

    # Standard Monte Carlo
    U = np.random.uniform(0, 1, n)
    standard_estimate = np.mean(np.exp(U))
    standard_var = np.var(np.exp(U)) / n

    # Antithetic variates
    U = np.random.uniform(0, 1, n // 2)
    f_U = np.exp(U)
    f_1mU = np.exp(1 - U)
    av_estimate = np.mean((f_U + f_1mU) / 2)
    av_var = np.var((f_U + f_1mU) / 2) / (n // 2)

    print("E[e^U] estimation (true value = 1.71828):")
    print(f"\nStandard MC:")
    print(f"  Estimate: {standard_estimate:.6f}")
    print(f"  Variance: {standard_var:.2e}")

    print(f"\nAntithetic variates:")
    print(f"  Estimate: {av_estimate:.6f}")
    print(f"  Variance: {av_var:.2e}")

    print(f"\nVariance reduction factor: {standard_var / av_var:.2f}x")

antithetic_variates()
```

### 7.2 Stratified Sampling

```python
def stratified_sampling():
    """
    Stratified sampling: Divide domain and sample uniformly from each stratum

    Variance reduction: Only within-stratum variance remains
    """

    # Example: ∫₀¹ x² dx = 1/3

    n_total = 10000

    # Standard MC
    X = np.random.uniform(0, 1, n_total)
    standard_estimate = np.mean(X**2)
    standard_var = np.var(X**2) / n_total

    # Stratified sampling (10 strata)
    n_strata = 10
    n_per_stratum = n_total // n_strata

    stratified_estimates = []
    for i in range(n_strata):
        low = i / n_strata
        high = (i + 1) / n_strata
        X_stratum = np.random.uniform(low, high, n_per_stratum)
        stratum_mean = np.mean(X_stratum**2)
        stratified_estimates.append(stratum_mean)

    stratified_estimate = np.mean(stratified_estimates)

    # Stratified sampling variance (within-stratum variance only)
    within_vars = []
    for i in range(n_strata):
        low = i / n_strata
        high = (i + 1) / n_strata
        X_stratum = np.random.uniform(low, high, 1000)
        within_vars.append(np.var(X_stratum**2))

    stratified_var = np.mean(within_vars) / n_total

    print("∫₀¹ x² dx = 1/3 ≈ 0.3333")
    print(f"\nStandard MC:")
    print(f"  Estimate: {standard_estimate:.6f}")
    print(f"  Variance: {standard_var:.2e}")

    print(f"\nStratified sampling (10 strata):")
    print(f"  Estimate: {stratified_estimate:.6f}")
    print(f"  Variance: {stratified_var:.2e}")

    print(f"\nVariance reduction factor: {standard_var / stratified_var:.2f}x")

stratified_sampling()
```

### 7.3 Control Variates

```python
def control_variates():
    """
    Control variates: Use variable with known expectation to reduce variance

    θ̂_cv = θ̂ - c(Ŷ - E[Y])

    where Y is a control variate with known expectation E[Y]
    c is the coefficient that minimizes variance
    """

    # Example: E[e^U] using U as control variate
    # E[U] = 0.5 (known value)

    n = 10000
    true_value = np.e - 1

    U = np.random.uniform(0, 1, n)
    f = np.exp(U)

    # Standard MC
    standard_estimate = np.mean(f)
    standard_var = np.var(f) / n

    # Control variates
    Y = U
    EY = 0.5  # E[U]

    # Estimate optimal c
    cov_fY = np.cov(f, Y)[0, 1]
    var_Y = np.var(Y)
    c_opt = cov_fY / var_Y

    # Control variate estimator
    cv_estimate = np.mean(f - c_opt * (Y - EY))
    cv_var = np.var(f - c_opt * (Y - EY)) / n

    print("E[e^U] estimation (true value = 1.71828):")
    print(f"\nStandard MC:")
    print(f"  Estimate: {standard_estimate:.6f}")
    print(f"  Variance: {standard_var:.2e}")

    print(f"\nControl variates (c = {c_opt:.4f}):")
    print(f"  Estimate: {cv_estimate:.6f}")
    print(f"  Variance: {cv_var:.2e}")

    print(f"\nVariance reduction factor: {standard_var / cv_var:.2f}x")

    # Predict variance reduction from correlation
    corr = np.corrcoef(f, Y)[0, 1]
    theoretical_reduction = 1 / (1 - corr**2)
    print(f"Theoretical variance reduction (ρ² = {corr**2:.4f}): {theoretical_reduction:.2f}x")

control_variates()
```

---

## Exercises

### Exercise 1: Volume of a Sphere
Estimate the volume of a d-dimensional unit sphere using Monte Carlo. (For d=3, 4π/3 ≈ 4.19)

```python
def exercise_1():
    def sphere_volume_mc(d, n_samples):
        points = np.random.uniform(-1, 1, (n_samples, d))
        inside = np.sum(points**2, axis=1) <= 1
        cube_volume = 2**d
        return cube_volume * np.mean(inside)

    for d in [2, 3, 4, 5]:
        volume = sphere_volume_mc(d, 100000)
        theoretical = np.pi**(d/2) / np.math.gamma(d/2 + 1)
        print(f"d={d}: MC={volume:.4f}, theoretical={theoretical:.4f}")

exercise_1()
```

### Exercise 2: Asian Option
Simulate the price of a path-dependent option (Asian call option).

```python
def exercise_2():
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    n_steps, n_paths = 252, 100000

    dt = T / n_steps
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0

    for i in range(n_steps):
        Z = np.random.normal(0, 1, n_paths)
        S[:, i+1] = S[:, i] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)

    # Arithmetic average
    S_avg = np.mean(S[:, 1:], axis=1)
    payoffs = np.maximum(S_avg - K, 0)
    price = np.exp(-r*T) * np.mean(payoffs)

    print(f"Asian call option price: {price:.4f}")

exercise_2()
```

---

## Summary

| Technique | Description | Use Case |
|-----------|-------------|----------|
| Basic MC | Uniform sampling for integration | General integration |
| Importance sampling | Using proposal distribution | Rare events, variance reduction |
| Antithetic variates | Using U and 1-U | Monotonic functions |
| Stratified sampling | Domain partitioning | Uniform coverage |
| Control variates | Leveraging correlated variables | When expectation is known |

| Application | Examples |
|-------------|----------|
| Physics | Ising model, molecular dynamics |
| Finance | Option pricing, VaR |
| Engineering | Reliability analysis, uncertainty quantification |
