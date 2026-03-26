"""
Probability and Statistics — Stochastic Processes
Exercises covering transition matrix computation, stationary distribution,
absorption probability, and Poisson process simulation.
"""
import math
import random
from typing import Dict, List, Tuple


# === Exercise 1: Transition Matrix Computation ===
def exercise_1() -> None:
    """Compute transition probabilities from observed sequence data,
    simulate the Markov chain, and verify empirical transitions."""
    print("=== Exercise 1: Transition Matrix Computation ===")

    # Observed weather sequence: 0=Sunny, 1=Cloudy, 2=Rainy
    states = ["Sunny", "Cloudy", "Rainy"]
    sequence = [0, 0, 1, 2, 2, 1, 0, 0, 0, 1, 1, 2, 2, 2, 1, 0, 0, 1, 2, 1,
                0, 0, 0, 0, 1, 2, 2, 1, 1, 0, 0, 1, 0, 0, 0, 1, 2, 1, 0, 0]

    # Count transitions
    n_states = 3
    counts: List[List[int]] = [[0] * n_states for _ in range(n_states)]
    for i in range(len(sequence) - 1):
        counts[sequence[i]][sequence[i + 1]] += 1

    # Compute transition probabilities
    P: List[List[float]] = [[0.0] * n_states for _ in range(n_states)]
    for i in range(n_states):
        row_total = sum(counts[i])
        if row_total > 0:
            for j in range(n_states):
                P[i][j] = counts[i][j] / row_total

    print(f"\n  Observed sequence length: {len(sequence)}")
    print(f"\n  Transition counts:")
    print(f"  {'':>8}", end="")
    for s in states:
        print(f" {s:>8}", end="")
    print()
    for i in range(n_states):
        print(f"  {states[i]:>8}", end="")
        for j in range(n_states):
            print(f" {counts[i][j]:>8}", end="")
        print()

    print(f"\n  Estimated transition matrix P:")
    print(f"  {'':>8}", end="")
    for s in states:
        print(f" {s:>8}", end="")
    print()
    for i in range(n_states):
        print(f"  {states[i]:>8}", end="")
        for j in range(n_states):
            print(f" {P[i][j]:>8.4f}", end="")
        print()

    # Verify rows sum to 1
    for i in range(n_states):
        row_sum = sum(P[i])
        assert abs(row_sum - 1.0) < 1e-10, f"Row {i} sums to {row_sum}"
    print(f"\n  All rows sum to 1.0.")

    # Simulate and compare
    random.seed(42)
    n_sim = 50000
    sim_counts: List[List[int]] = [[0] * n_states for _ in range(n_states)]
    state = 0
    for _ in range(n_sim):
        r = random.random()
        cumsum = 0.0
        next_state = 0
        for j in range(n_states):
            cumsum += P[state][j]
            if r < cumsum:
                next_state = j
                break
        sim_counts[state][next_state] += 1
        state = next_state

    print(f"\n  Simulation verification ({n_sim:,} steps):")
    for i in range(n_states):
        row_total = sum(sim_counts[i])
        if row_total > 0:
            emp = [sim_counts[i][j] / row_total for j in range(n_states)]
            print(f"    {states[i]:>8}: [{', '.join(f'{p:.4f}' for p in emp)}] "
                  f"(theory: [{', '.join(f'{P[i][j]:.4f}' for j in range(n_states))}])")
    print()


# === Exercise 2: Stationary Distribution ===
def exercise_2() -> None:
    """Find the stationary distribution of a Markov chain using power
    iteration and verify by checking pi*P = pi."""
    print("=== Exercise 2: Stationary Distribution ===")

    states = ["Sunny", "Cloudy", "Rainy"]
    P = [
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5],
    ]
    n_states = len(P)

    print(f"\n  Transition matrix P:")
    for i in range(n_states):
        print(f"    {states[i]:>8} -> [{', '.join(f'{p:.1f}' for p in P[i])}]")

    # Method 1: Power iteration
    pi = [1 / n_states] * n_states
    for iteration in range(1000):
        new_pi = [0.0] * n_states
        for j in range(n_states):
            for i in range(n_states):
                new_pi[j] += pi[i] * P[i][j]
        if all(abs(new_pi[i] - pi[i]) < 1e-12 for i in range(n_states)):
            print(f"\n  Power iteration converged in {iteration + 1} steps")
            pi = new_pi
            break
        pi = new_pi

    print(f"\n  Stationary distribution pi:")
    for i in range(n_states):
        print(f"    pi({states[i]}) = {pi[i]:.6f}")
    print(f"    Sum = {sum(pi):.6f}")

    # Verify pi*P = pi
    print(f"\n  Verification (pi*P should equal pi):")
    pi_P = [sum(pi[i] * P[i][j] for i in range(n_states))
            for j in range(n_states)]
    max_diff = 0.0
    for i in range(n_states):
        diff = abs(pi_P[i] - pi[i])
        max_diff = max(max_diff, diff)
        print(f"    (pi*P)[{states[i]}] = {pi_P[i]:.6f} vs pi[{states[i]}] = {pi[i]:.6f}")
    print(f"    Max |difference| = {max_diff:.2e}")

    # Convergence of P^n
    def mat_mul_3x3(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        C = [[0.0] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                for k_idx in range(3):
                    C[i][j] += A[i][k_idx] * B[k_idx][j]
        return C

    print(f"\n  Convergence of P^n (row 0):")
    Pn = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for step in range(1, 51):
        Pn = mat_mul_3x3(Pn, P)
        if step in [1, 2, 5, 10, 20, 50]:
            print(f"    P^{step:>2}: [{', '.join(f'{x:.6f}' for x in Pn[0])}]")
    print(f"    pi:   [{', '.join(f'{x:.6f}' for x in pi)}]\n")


# === Exercise 3: Absorption Probability ===
def exercise_3() -> None:
    """Compute absorption probabilities for a Markov chain with absorbing
    states (Gambler's Ruin formulation)."""
    print("=== Exercise 3: Absorption Probability (Gambler's Ruin) ===")

    # Gambler's ruin: states 0, 1, ..., N where 0 and N are absorbing
    N = 10
    p = 0.45  # P(win)
    q = 1 - p  # P(lose)

    print(f"\n  Gambler's Ruin: N={N}, p={p}, q={q}")
    print(f"  States 0 (ruin) and {N} (target) are absorbing")
    print(f"\n  P(reach {N} | start at i) = ?")

    # Analytical solution:
    # P(reach N | start=i) = (1 - r^i) / (1 - r^N) where r = q/p, p != q
    r = q / p

    print(f"\n  Analytical formula: P(win | i) = (1 - r^i) / (1 - r^N)")
    print(f"  where r = q/p = {r:.6f}")

    print(f"\n  {'State i':>8} {'P(win)':>10} {'P(ruin)':>10}")
    print(f"  {'-------':>8} {'------':>10} {'-------':>10}")
    analytic: List[float] = []
    for i in range(N + 1):
        if i == 0:
            p_win = 0.0
        elif i == N:
            p_win = 1.0
        elif abs(p - q) < 1e-10:
            p_win = i / N
        else:
            p_win = (1 - r ** i) / (1 - r ** N)
        analytic.append(p_win)
        print(f"  {i:>8} {p_win:>10.6f} {1 - p_win:>10.6f}")

    # Simulation verification
    random.seed(42)
    n_sims = 10000

    print(f"\n  Simulation verification ({n_sims:,} trials per starting state):")
    print(f"  {'State i':>8} {'P(win) theory':>14} {'P(win) sim':>12} {'|diff|':>8}")
    print(f"  {'-------':>8} {'-------------':>14} {'----------':>12} {'------':>8}")

    for start in [1, 3, 5, 7, 9]:
        wins = 0
        for _ in range(n_sims):
            state = start
            while 0 < state < N:
                if random.random() < p:
                    state += 1
                else:
                    state -= 1
            if state == N:
                wins += 1
        p_sim = wins / n_sims
        diff = abs(p_sim - analytic[start])
        print(f"  {start:>8} {analytic[start]:>14.6f} {p_sim:>12.4f} {diff:>8.4f}")

    # Expected duration
    print(f"\n  Expected game duration E[T | start=i]:")
    print(f"  {'State i':>8} {'E[T]':>10}")
    print(f"  {'-------':>8} {'----':>10}")
    for i in [1, 3, 5, 7, 9]:
        if abs(p - q) < 1e-10:
            e_t = i * (N - i)
        else:
            e_t = i / (q - p) - N / (q - p) * (1 - r ** i) / (1 - r ** N)
        print(f"  {i:>8} {e_t:>10.1f}")
    print()


# === Exercise 4: Poisson Process Simulation ===
def exercise_4() -> None:
    """Simulate a Poisson process and verify that N(t) ~ Poisson(lambda*t)
    and inter-arrival times are exponentially distributed."""
    print("=== Exercise 4: Poisson Process Simulation ===")

    random.seed(42)
    lam = 3.0
    T = 10.0

    print(f"\n  Poisson process: lambda={lam}, T={T}")
    print(f"  Expected N(T) = lambda*T = {lam * T:.0f}")

    # Simulate via exponential inter-arrivals
    def simulate_poisson(lam_rate: float, t_end: float) -> List[float]:
        arrivals = []
        t = 0.0
        while True:
            inter = random.expovariate(lam_rate)
            t += inter
            if t > t_end:
                break
            arrivals.append(t)
        return arrivals

    # Show 3 sample paths
    print(f"\n  Sample paths:")
    for path_id in range(3):
        arrivals = simulate_poisson(lam, T)
        print(f"    Path {path_id + 1}: N({T:.0f}) = {len(arrivals):>2} events")
        # Timeline
        line = [' '] * 50
        for a in arrivals:
            pos = int(a / T * 49)
            pos = max(0, min(49, pos))
            line[pos] = '|'
        print(f"    0{''.join(line)}{T:.0f}")

    # Distribution verification
    n_sims = 20000
    counts = [len(simulate_poisson(lam, T)) for _ in range(n_sims)]

    mean_count = sum(counts) / n_sims
    var_count = sum((c - mean_count) ** 2 for c in counts) / n_sims

    print(f"\n  Distribution of N(T) from {n_sims:,} simulations:")
    print(f"    E[N(T)]   = {mean_count:.4f}  (theory: {lam * T:.0f})")
    print(f"    Var[N(T)] = {var_count:.4f}  (theory: {lam * T:.0f})")
    print(f"    Var/Mean  = {var_count / mean_count:.4f}  (theory: 1.0)")

    # PMF comparison
    mu = lam * T
    print(f"\n  P(N(T) = k) comparison:")
    print(f"  {'k':>4} {'Sim':>8} {'Poisson':>8} {'|diff|':>8}")
    print(f"  {'--':>4} {'---':>8} {'-------':>8} {'------':>8}")
    for k_val in range(int(mu - 3 * math.sqrt(mu)),
                       int(mu + 3 * math.sqrt(mu)) + 1):
        if k_val < 0:
            continue
        sim_prob = sum(1 for c in counts if c == k_val) / n_sims
        th_prob = math.exp(-mu + k_val * math.log(mu) - math.lgamma(k_val + 1))
        diff = abs(sim_prob - th_prob)
        print(f"  {k_val:>4} {sim_prob:>8.4f} {th_prob:>8.4f} {diff:>8.4f}")

    # Inter-arrival time verification
    long_arrivals = simulate_poisson(lam, 100.0)
    inter_arrivals = [long_arrivals[i] - long_arrivals[i - 1]
                      for i in range(1, len(long_arrivals))]
    if inter_arrivals:
        ia_mean = sum(inter_arrivals) / len(inter_arrivals)
        ia_var = sum((x - ia_mean) ** 2 for x in inter_arrivals) / len(inter_arrivals)
        print(f"\n  Inter-arrival time verification:")
        print(f"    E[inter-arrival] = {ia_mean:.5f}  (theory: 1/lambda = {1 / lam:.5f})")
        print(f"    Var[inter-arrival] = {ia_var:.5f}  (theory: 1/lambda^2 = {1 / lam ** 2:.5f})")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
