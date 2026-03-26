"""
Joint Distributions

Demonstrates:
1. Joint PMF Table
2. Marginal and Conditional Distributions
3. Covariance and Correlation
4. Iterated Expectation

Theory:
- Joint PMF: p(x,y) = P(X=x, Y=y), ΣΣ p(x,y) = 1
- Marginal: p_X(x) = Σ_y p(x,y)
- Conditional: p(y|x) = p(x,y) / p_X(x)
- Cov(X,Y) = E[XY] - E[X]E[Y]
- ρ(X,Y) = Cov(X,Y) / (σ_X · σ_Y)

Adapted from Probability and Statistics Lesson 05.
"""

import math
import random


# ─────────────────────────────────────────────────
# 1. JOINT PMF TABLE
# ─────────────────────────────────────────────────

class JointPMF:
    """Joint probability mass function as a 2D table."""

    def __init__(self, x_vals: list, y_vals: list,
                 probs: dict[tuple, float]):
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.probs = probs

    def p(self, x, y) -> float:
        return self.probs.get((x, y), 0.0)

    def marginal_x(self) -> dict:
        return {x: sum(self.p(x, y) for y in self.y_vals)
                for x in self.x_vals}

    def marginal_y(self) -> dict:
        return {y: sum(self.p(x, y) for x in self.x_vals)
                for y in self.y_vals}

    def conditional_y_given_x(self, x) -> dict:
        px = sum(self.p(x, y) for y in self.y_vals)
        if px == 0:
            return {}
        return {y: self.p(x, y) / px for y in self.y_vals}

    def expected_xy(self) -> float:
        return sum(x * y * self.p(x, y)
                   for x in self.x_vals for y in self.y_vals)

    def display(self) -> str:
        lines = []
        header = f"  {'':>6}" + "".join(f"  Y={y:>3}" for y in self.y_vals) + "  p_X(x)"
        lines.append(header)
        lines.append("  " + "─" * (len(header) - 2))

        mx = self.marginal_x()
        for x in self.x_vals:
            row = f"  X={x:>3}"
            for y in self.y_vals:
                row += f"  {self.p(x, y):>5.3f}"
            row += f"  {mx[x]:>5.3f}"
            lines.append(row)

        my = self.marginal_y()
        footer = f"  {'p_Y(y)':>6}"
        for y in self.y_vals:
            footer += f"  {my[y]:>5.3f}"
        footer += f"  {sum(my.values()):>5.3f}"
        lines.append("  " + "─" * (len(header) - 2))
        lines.append(footer)
        return "\n".join(lines)


def demo_joint_pmf():
    print("=" * 60)
    print("  Joint PMF Table")
    print("=" * 60)

    # Two dice: X = min, Y = max
    probs = {}
    for d1 in range(1, 7):
        for d2 in range(1, 7):
            x, y = min(d1, d2), max(d1, d2)
            probs[(x, y)] = probs.get((x, y), 0) + 1/36

    x_vals = list(range(1, 7))
    y_vals = list(range(1, 7))
    joint = JointPMF(x_vals, y_vals, probs)

    print(f"\n  X = min(D1,D2), Y = max(D1,D2):")
    print(joint.display())


# ─────────────────────────────────────────────────
# 2. MARGINAL AND CONDITIONAL DISTRIBUTIONS
# ─────────────────────────────────────────────────

def demo_marginal_conditional():
    print("\n" + "=" * 60)
    print("  Marginal and Conditional Distributions")
    print("=" * 60)

    # Insurance example
    # X = number of claims (0,1,2), Y = claim amount (0, 100, 500)
    probs = {
        (0, 0): 0.60,
        (1, 100): 0.15, (1, 500): 0.05,
        (2, 100): 0.10, (2, 500): 0.10,
    }
    # Fill missing with 0
    x_vals = [0, 1, 2]
    y_vals = [0, 100, 500]
    for x in x_vals:
        for y in y_vals:
            probs.setdefault((x, y), 0.0)

    joint = JointPMF(x_vals, y_vals, probs)
    print(f"\n  Insurance: X=claims, Y=amount")
    print(joint.display())

    # Marginals
    mx = joint.marginal_x()
    my = joint.marginal_y()
    print(f"\n  Marginal of X: {mx}")
    print(f"  Marginal of Y: {my}")

    # Conditional
    for x in x_vals:
        cond = joint.conditional_y_given_x(x)
        if cond:
            formatted = {k: f"{v:.3f}" for k, v in cond.items() if v > 0}
            print(f"  P(Y|X={x}): {formatted}")

    # E[Y|X=2]
    cond_2 = joint.conditional_y_given_x(2)
    e_y_given_x2 = sum(y * p for y, p in cond_2.items())
    print(f"\n  E[Y|X=2] = {e_y_given_x2:.1f}")


# ─────────────────────────────────────────────────
# 3. COVARIANCE AND CORRELATION
# ─────────────────────────────────────────────────

def covariance_from_joint(joint: JointPMF) -> float:
    """Cov(X,Y) = E[XY] - E[X]E[Y]."""
    mx = joint.marginal_x()
    my = joint.marginal_y()
    e_x = sum(x * p for x, p in mx.items())
    e_y = sum(y * p for y, p in my.items())
    e_xy = joint.expected_xy()
    return e_xy - e_x * e_y


def correlation_from_joint(joint: JointPMF) -> float:
    """ρ(X,Y) = Cov(X,Y) / (σ_X · σ_Y)."""
    cov = covariance_from_joint(joint)
    mx = joint.marginal_x()
    my = joint.marginal_y()

    e_x = sum(x * p for x, p in mx.items())
    e_x2 = sum(x**2 * p for x, p in mx.items())
    var_x = e_x2 - e_x**2

    e_y = sum(y * p for y, p in my.items())
    e_y2 = sum(y**2 * p for y, p in my.items())
    var_y = e_y2 - e_y**2

    if var_x <= 0 or var_y <= 0:
        return 0.0
    return cov / (math.sqrt(var_x) * math.sqrt(var_y))


def demo_covariance():
    print("\n" + "=" * 60)
    print("  Covariance and Correlation")
    print("=" * 60)

    # Correlated: X, Y = exam scores
    probs = {
        (60, 50): 0.05, (60, 70): 0.05, (60, 90): 0.00,
        (70, 50): 0.05, (70, 70): 0.15, (70, 90): 0.05,
        (80, 50): 0.00, (80, 70): 0.15, (80, 90): 0.15,
        (90, 50): 0.00, (90, 70): 0.05, (90, 90): 0.30,
    }
    x_vals = [60, 70, 80, 90]
    y_vals = [50, 70, 90]
    joint = JointPMF(x_vals, y_vals, probs)

    cov = covariance_from_joint(joint)
    rho = correlation_from_joint(joint)
    print(f"\n  Exam scores: X=midterm, Y=final")
    print(f"  Cov(X,Y) = {cov:.2f}")
    print(f"  ρ(X,Y) = {rho:.4f}")

    # Simulation comparison
    print(f"\n  Monte Carlo correlation check:")
    random.seed(42)
    n = 50000
    # Generate correlated normals
    rho_target = 0.7
    x_data = [random.gauss(0, 1) for _ in range(n)]
    z_data = [random.gauss(0, 1) for _ in range(n)]
    y_data = [rho_target * x + math.sqrt(1 - rho_target**2) * z
              for x, z in zip(x_data, z_data)]

    mean_x = sum(x_data) / n
    mean_y = sum(y_data) / n
    cov_sim = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_data, y_data)) / n
    var_x_sim = sum((x - mean_x)**2 for x in x_data) / n
    var_y_sim = sum((y - mean_y)**2 for y in y_data) / n
    rho_sim = cov_sim / (math.sqrt(var_x_sim) * math.sqrt(var_y_sim))
    print(f"  Target ρ = {rho_target}, Simulated ρ = {rho_sim:.4f}")


# ─────────────────────────────────────────────────
# 4. ITERATED EXPECTATION
# ─────────────────────────────────────────────────

def demo_iterated_expectation():
    print("\n" + "=" * 60)
    print("  Iterated Expectation: E[Y] = E[E[Y|X]]")
    print("=" * 60)

    # Factory example: X = machine (1,2,3), Y = defects per batch
    # P(X=1) = 0.5, P(X=2) = 0.3, P(X=3) = 0.2
    # E[Y|X=1] = 2, E[Y|X=2] = 5, E[Y|X=3] = 10
    p_x = {1: 0.5, 2: 0.3, 3: 0.2}
    e_y_given_x = {1: 2, 2: 5, 3: 10}

    e_y = sum(p_x[x] * e_y_given_x[x] for x in p_x)

    print(f"\n  Factory defects:")
    for x in p_x:
        print(f"  Machine {x}: P(X={x})={p_x[x]:.1f}, E[Y|X={x}]={e_y_given_x[x]}")

    print(f"\n  E[Y] = E[E[Y|X]]")
    terms = " + ".join(f"{p_x[x]:.1f}×{e_y_given_x[x]}" for x in p_x)
    print(f"       = {terms}")
    print(f"       = {e_y:.1f}")

    # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
    var_y_given_x = {1: 1, 2: 4, 3: 9}
    e_var = sum(p_x[x] * var_y_given_x[x] for x in p_x)
    var_e = sum(p_x[x] * (e_y_given_x[x] - e_y)**2 for x in p_x)
    total_var = e_var + var_e

    print(f"\n  Law of Total Variance:")
    print(f"  Var(Y) = E[Var(Y|X)] + Var(E[Y|X])")
    print(f"         = {e_var:.2f} + {var_e:.2f} = {total_var:.2f}")


if __name__ == "__main__":
    demo_joint_pmf()
    demo_marginal_conditional()
    demo_covariance()
    demo_iterated_expectation()
