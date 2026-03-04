"""
Stochastic Processes Introduction

Demonstrates:
1. Markov Chain Simulation
2. Stationary Distribution
3. Poisson Process
4. Random Walk and Gambler's Ruin

Theory:
- Markov chain: P(Xn+1|Xn,...,X0) = P(Xn+1|Xn)
- Stationary distribution: πP = π
- Poisson process: inter-arrivals ~ Exp(λ), N(t) ~ Poisson(λt)
- Random walk: Sn = X1 + X2 + ... + Xn

Adapted from Probability and Statistics Lesson 18.
"""

import math
import random


# ─────────────────────────────────────────────────
# 1. MARKOV CHAIN SIMULATION
# ─────────────────────────────────────────────────

def markov_step(state: int, transition: list[list[float]]) -> int:
    """Take one step in a Markov chain."""
    probs = transition[state]
    r = random.random()
    cumsum = 0
    for next_state, p in enumerate(probs):
        cumsum += p
        if r < cumsum:
            return next_state
    return len(probs) - 1


def simulate_markov(transition: list[list[float]], start: int,
                     steps: int) -> list[int]:
    """Simulate a Markov chain path."""
    path = [start]
    for _ in range(steps):
        path.append(markov_step(path[-1], transition))
    return path


def demo_markov():
    print("=" * 60)
    print("  Markov Chain Simulation")
    print("=" * 60)

    # Weather model: 0=Sunny, 1=Cloudy, 2=Rainy
    states = ["Sunny", "Cloudy", "Rainy"]
    P = [
        [0.7, 0.2, 0.1],  # Sunny →
        [0.3, 0.4, 0.3],  # Cloudy →
        [0.2, 0.3, 0.5],  # Rainy →
    ]

    print(f"\n  Weather Markov Chain:")
    print(f"  Transition matrix P:")
    for i, row in enumerate(P):
        print(f"    {states[i]:>6} → {[f'{p:.1f}' for p in row]}")

    random.seed(42)
    path = simulate_markov(P, start=0, steps=30)
    print(f"\n  Sample path (30 days starting Sunny):")
    line = "  "
    for s in path[:31]:
        line += states[s][0]
    print(line)

    # Empirical distribution from long run
    long_path = simulate_markov(P, start=0, steps=100000)
    from collections import Counter
    counts = Counter(long_path)
    n = len(long_path)
    print(f"\n  Empirical distribution (100K steps):")
    for i, s in enumerate(states):
        print(f"    {s}: {counts[i]/n:.4f}")


# ─────────────────────────────────────────────────
# 2. STATIONARY DISTRIBUTION
# ─────────────────────────────────────────────────

def mat_power(P: list[list[float]], n: int) -> list[list[float]]:
    """Compute P^n by repeated multiplication."""
    k = len(P)
    result = [[1 if i == j else 0 for j in range(k)] for i in range(k)]
    base = [row[:] for row in P]

    while n > 0:
        if n % 2 == 1:
            result = mat_mul(result, base)
        base = mat_mul(base, base)
        n //= 2
    return result


def mat_mul(A, B):
    k = len(A)
    C = [[0]*k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            for l in range(k):
                C[i][j] += A[i][l] * B[l][j]
    return C


def find_stationary(P: list[list[float]], tol: float = 1e-10) -> list[float]:
    """Find stationary distribution by power iteration."""
    k = len(P)
    pi = [1/k] * k

    for _ in range(1000):
        new_pi = [0] * k
        for j in range(k):
            for i in range(k):
                new_pi[j] += pi[i] * P[i][j]

        # Check convergence
        if all(abs(new_pi[i] - pi[i]) < tol for i in range(k)):
            return new_pi
        pi = new_pi

    return pi


def demo_stationary():
    print("\n" + "=" * 60)
    print("  Stationary Distribution")
    print("=" * 60)

    states = ["Sunny", "Cloudy", "Rainy"]
    P = [[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]]

    pi = find_stationary(P)
    print(f"\n  Stationary π (πP = π):")
    for i, s in enumerate(states):
        print(f"    π({s}) = {pi[i]:.6f}")

    # Verify: πP = π
    pi_P = [sum(pi[i] * P[i][j] for i in range(3)) for j in range(3)]
    print(f"\n  Verification πP:")
    for i, s in enumerate(states):
        print(f"    (πP)({s}) = {pi_P[i]:.6f} ≈ π({s}) = {pi[i]:.6f}")

    # Convergence of P^n
    print(f"\n  P^n convergence (row 0):")
    for n in [1, 2, 5, 10, 50]:
        Pn = mat_power(P, n)
        print(f"  n={n:>2}: [{', '.join(f'{x:.4f}' for x in Pn[0])}]")
    print(f"  π:   [{', '.join(f'{x:.4f}' for x in pi)}]")


# ─────────────────────────────────────────────────
# 3. POISSON PROCESS
# ─────────────────────────────────────────────────

def simulate_poisson_process(lam: float,
                              T: float) -> list[float]:
    """Simulate arrival times of Poisson(λ) process on [0, T]."""
    arrivals = []
    t = 0
    while True:
        inter = random.expovariate(lam)
        t += inter
        if t > T:
            break
        arrivals.append(t)
    return arrivals


def demo_poisson_process():
    print("\n" + "=" * 60)
    print("  Poisson Process")
    print("=" * 60)

    random.seed(42)
    lam = 3.0
    T = 10.0

    print(f"\n  Poisson process with λ={lam}, T={T}")

    # Simulate multiple paths
    for path_id in range(3):
        arrivals = simulate_poisson_process(lam, T)
        print(f"\n  Path {path_id+1}: {len(arrivals)} arrivals")
        # Timeline
        line = [" "] * 50
        for a in arrivals:
            pos = int(a / T * 49)
            line[pos] = "│"
        print(f"  0{''.join(line)}{T:.0f}")

    # Count distribution verification
    n_sims = 10000
    counts = [len(simulate_poisson_process(lam, T)) for _ in range(n_sims)]
    mean_count = sum(counts) / n_sims
    var_count = sum((c - mean_count)**2 for c in counts) / n_sims
    print(f"\n  N(T={T:.0f}) distribution ({n_sims} simulations):")
    print(f"  E[N] = {mean_count:.2f} (theory: λT = {lam*T:.0f})")
    print(f"  Var[N] = {var_count:.2f} (theory: λT = {lam*T:.0f})")

    # Inter-arrival times
    arrivals = simulate_poisson_process(lam, 100)
    inter_arrivals = [arrivals[i] - arrivals[i-1] for i in range(1, len(arrivals))]
    mean_ia = sum(inter_arrivals) / len(inter_arrivals)
    print(f"\n  Inter-arrival times:")
    print(f"  Mean = {mean_ia:.4f} (theory: 1/λ = {1/lam:.4f})")


# ─────────────────────────────────────────────────
# 4. RANDOM WALK AND GAMBLER'S RUIN
# ─────────────────────────────────────────────────

def random_walk(n: int, p: float = 0.5) -> list[int]:
    """Simple random walk: +1 with prob p, -1 with prob 1-p."""
    path = [0]
    for _ in range(n):
        step = 1 if random.random() < p else -1
        path.append(path[-1] + step)
    return path


def gamblers_ruin(initial: int, target: int,
                   p: float = 0.5) -> tuple[bool, int]:
    """Simulate gambler's ruin: start at initial, target=win, 0=lose."""
    current = initial
    steps = 0
    while 0 < current < target:
        current += 1 if random.random() < p else -1
        steps += 1
    return current == target, steps


def demo_random_walk():
    print("\n" + "=" * 60)
    print("  Random Walk and Gambler's Ruin")
    print("=" * 60)

    random.seed(42)

    # Simple symmetric random walk
    print(f"\n  Symmetric random walk (p=0.5, n=50):")
    for i in range(3):
        path = random_walk(50)
        # ASCII visualization
        min_v = min(path)
        max_v = max(path)
        rng = max(max_v - min_v, 1)
        line = ""
        for v in path:
            pos = int((v - min_v) / rng * 20)
            line += " " * pos + "·" + " " * (20 - pos) + "|"
        # Just show endpoints
        print(f"  Path {i+1}: start=0, end={path[-1]:>3}, "
              f"min={min_v:>3}, max={max_v:>3}")

    # Gambler's ruin
    print(f"\n  Gambler's Ruin (initial=$20, target=$100):")
    for p in [0.5, 0.49, 0.45]:
        n_sims = 5000
        wins = 0
        total_steps = 0
        for _ in range(n_sims):
            won, steps = gamblers_ruin(20, 100, p)
            if won:
                wins += 1
            total_steps += steps
        win_rate = wins / n_sims

        # Theoretical ruin probability
        if p == 0.5:
            p_win_theory = 20 / 100
        else:
            q = 1 - p
            r = q / p
            p_win_theory = (1 - r**20) / (1 - r**100)

        print(f"  p={p}: P(win)={win_rate:.3f} (theory: {p_win_theory:.3f}), "
              f"avg steps={total_steps/n_sims:.0f}")


if __name__ == "__main__":
    demo_markov()
    demo_stationary()
    demo_poisson_process()
    demo_random_walk()
