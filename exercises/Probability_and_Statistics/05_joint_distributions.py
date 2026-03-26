"""
Probability and Statistics — Joint Distributions
Exercises covering joint PMF, marginal distributions, covariance,
correlation, the law of total expectation, and independence testing.
"""
import math
from typing import Dict, List, Tuple


# === Exercise 1: Joint PMF for (min, max) of Two Dice ===
def exercise_1() -> None:
    """Build the joint PMF table for (U, V) = (min(D1,D2), max(D1,D2))
    where D1, D2 are independent fair dice.  Compute marginal distributions."""
    print("=== Exercise 1: Joint PMF of (min, max) for Two Dice ===")

    total = 36
    joint: Dict[Tuple[int, int], float] = {}

    for d1 in range(1, 7):
        for d2 in range(1, 7):
            u = min(d1, d2)
            v = max(d1, d2)
            key = (u, v)
            joint[key] = joint.get(key, 0) + 1 / total

    # Display joint PMF table
    print("  Joint PMF P(U=u, V=v):")
    header = "U\\V"
    print(f"  {header:>5}", end="")
    for v in range(1, 7):
        print(f"  V={v:>3}", end="")
    print("  | Marginal P(U=u)")
    print("  " + "-" * 62)

    marginal_u: Dict[int, float] = {}
    marginal_v: Dict[int, float] = {}

    for u in range(1, 7):
        print(f"  U={u:>2}", end="")
        row_sum = 0.0
        for v in range(1, 7):
            p = joint.get((u, v), 0)
            row_sum += p
            marginal_v[v] = marginal_v.get(v, 0) + p
            if p > 0:
                print(f" {p:>6.4f}", end="")
            else:
                print(f" {'  ---':>6}", end="")
        marginal_u[u] = row_sum
        print(f"  | {row_sum:.4f}")

    print("  " + "-" * 62)
    print(f"  {'Marg':>5}", end="")
    for v in range(1, 7):
        print(f" {marginal_v.get(v, 0):>6.4f}", end="")
    print()

    # Verify marginals sum to 1
    assert abs(sum(marginal_u.values()) - 1.0) < 1e-12
    assert abs(sum(marginal_v.values()) - 1.0) < 1e-12

    # E[U] and E[V]
    e_u = sum(u * p for u, p in marginal_u.items())
    e_v = sum(v * p for v, p in marginal_v.items())
    print(f"\n  E[U] = E[min(D1,D2)] = {e_u:.4f}")
    print(f"  E[V] = E[max(D1,D2)] = {e_v:.4f}")
    print(f"  E[U] + E[V] = {e_u + e_v:.4f}  (should equal E[D1]+E[D2] = 7.0)\n")


# === Exercise 2: Covariance and Correlation from Joint PMF ===
def exercise_2() -> None:
    """Given a joint PMF table for (X, Y), compute Cov(X, Y) and the
    Pearson correlation coefficient rho."""
    print("=== Exercise 2: Cov(X,Y) and Correlation from Joint PMF ===")

    # Joint PMF for (X, Y) where X in {0,1,2}, Y in {0,1,2}
    joint: Dict[Tuple[int, int], float] = {
        (0, 0): 0.10, (0, 1): 0.05, (0, 2): 0.02,
        (1, 0): 0.05, (1, 1): 0.20, (1, 2): 0.05,
        (2, 0): 0.02, (2, 1): 0.08, (2, 2): 0.43,
    }

    # Verify
    total = sum(joint.values())
    assert abs(total - 1.0) < 1e-12, f"Joint PMF sums to {total}"

    # Marginals
    x_vals = sorted(set(k[0] for k in joint))
    y_vals = sorted(set(k[1] for k in joint))

    p_x: Dict[int, float] = {}
    p_y: Dict[int, float] = {}
    for x in x_vals:
        p_x[x] = sum(joint.get((x, y), 0) for y in y_vals)
    for y in y_vals:
        p_y[y] = sum(joint.get((x, y), 0) for x in x_vals)

    # E[X], E[Y], E[XY]
    e_x = sum(x * p for x, p in p_x.items())
    e_y = sum(y * p for y, p in p_y.items())
    e_xy = sum(x * y * joint[(x, y)] for x, y in joint)

    # Var(X), Var(Y)
    e_x2 = sum(x**2 * p for x, p in p_x.items())
    e_y2 = sum(y**2 * p for y, p in p_y.items())
    var_x = e_x2 - e_x**2
    var_y = e_y2 - e_y**2

    # Cov and correlation
    cov_xy = e_xy - e_x * e_y
    rho = cov_xy / math.sqrt(var_x * var_y)

    print("  Joint PMF table:")
    xy_header = "X\\Y"
    print(f"  {xy_header:>5}", end="")
    for y in y_vals:
        print(f"  Y={y:>4}", end="")
    print(f"  | P(X=x)")
    for x in x_vals:
        print(f"  X={x:>2}", end="")
        for y in y_vals:
            print(f" {joint[(x, y)]:>6.2f}", end="")
        print(f"  | {p_x[x]:.2f}")

    print(f"\n  E[X]    = {e_x:.4f}")
    print(f"  E[Y]    = {e_y:.4f}")
    print(f"  E[XY]   = {e_xy:.4f}")
    print(f"  Var(X)  = {var_x:.4f}")
    print(f"  Var(Y)  = {var_y:.4f}")
    print(f"  Cov(X,Y) = E[XY] - E[X]E[Y] = {e_xy:.4f} - {e_x:.4f}*{e_y:.4f} "
          f"= {cov_xy:.4f}")
    print(f"  rho(X,Y) = Cov(X,Y) / (Std(X)*Std(Y)) = {rho:.4f}")
    print(f"  X and Y are {'positively' if rho > 0 else 'negatively'} correlated.\n")


# === Exercise 3: Verify E[X] = E[E[X|Y]] (Law of Total Expectation) ===
def exercise_3() -> None:
    """For a specific joint distribution of (X, Y), verify the tower
    property: E[X] = E_Y[E[X|Y]]."""
    print("=== Exercise 3: Verify E[X] = E[E[X|Y]] ===")

    # Joint PMF
    joint: Dict[Tuple[int, int], float] = {
        (1, 1): 0.10, (1, 2): 0.05, (1, 3): 0.05,
        (2, 1): 0.10, (2, 2): 0.15, (2, 3): 0.05,
        (3, 1): 0.05, (3, 2): 0.10, (3, 3): 0.10,
        (4, 1): 0.02, (4, 2): 0.05, (4, 3): 0.18,
    }

    x_vals = sorted(set(k[0] for k in joint))
    y_vals = sorted(set(k[1] for k in joint))

    # Marginal of Y
    p_y: Dict[int, float] = {}
    for y in y_vals:
        p_y[y] = sum(joint.get((x, y), 0) for x in x_vals)

    # Direct computation of E[X]
    p_x: Dict[int, float] = {}
    for x in x_vals:
        p_x[x] = sum(joint.get((x, y), 0) for y in y_vals)

    e_x_direct = sum(x * p for x, p in p_x.items())

    # Compute E[X|Y=y] for each y, then E_Y[E[X|Y]]
    print(f"  Computing E[X|Y=y] for each y:")
    e_x_given_y: Dict[int, float] = {}
    for y in y_vals:
        # Conditional PMF: P(X=x|Y=y) = P(X=x,Y=y) / P(Y=y)
        conditional_mean = sum(
            x * joint.get((x, y), 0) / p_y[y] for x in x_vals
        )
        e_x_given_y[y] = conditional_mean
        print(f"    E[X|Y={y}] = {conditional_mean:.4f}  (P(Y={y}) = {p_y[y]:.2f})")

    # Tower property: E[X] = sum_y E[X|Y=y] * P(Y=y)
    e_x_tower = sum(e_x_given_y[y] * p_y[y] for y in y_vals)

    print(f"\n  E[X] (direct)          = {e_x_direct:.6f}")
    print(f"  E_Y[E[X|Y]] (tower)   = {e_x_tower:.6f}")
    print(f"  Difference             = {abs(e_x_direct - e_x_tower):.2e}")
    assert abs(e_x_direct - e_x_tower) < 1e-12, "Tower property violated!"
    print("  Law of total expectation verified.\n")


# === Exercise 4: Independence Check from Joint PMF ===
def exercise_4() -> None:
    """Given a joint PMF for (X, Y), check whether X and Y are independent
    by verifying P(X=x, Y=y) = P(X=x)*P(Y=y) for all (x, y)."""
    print("=== Exercise 4: Independence Check from Joint PMF ===")

    # Case 1: Independent joint PMF (constructed as product of marginals)
    print("  Case 1: Constructed independent PMF")
    mx = {1: 0.3, 2: 0.5, 3: 0.2}
    my = {10: 0.4, 20: 0.6}
    joint_indep: Dict[Tuple[int, int], float] = {
        (x, y): px * py for x, px in mx.items() for y, py in my.items()
    }

    _check_independence(joint_indep, "Case 1")

    # Case 2: Dependent joint PMF
    print("  Case 2: Dependent joint PMF")
    joint_dep: Dict[Tuple[int, int], float] = {
        (0, 0): 0.30, (0, 1): 0.10,
        (1, 0): 0.10, (1, 1): 0.50,
    }

    _check_independence(joint_dep, "Case 2")


def _check_independence(
    joint: Dict[Tuple[int, int], float], label: str
) -> None:
    """Helper to check independence for a given joint PMF."""
    x_vals = sorted(set(k[0] for k in joint))
    y_vals = sorted(set(k[1] for k in joint))

    p_x: Dict[int, float] = {}
    p_y: Dict[int, float] = {}
    for x in x_vals:
        p_x[x] = sum(joint.get((x, y), 0) for y in y_vals)
    for y in y_vals:
        p_y[y] = sum(joint.get((x, y), 0) for x in x_vals)

    independent = True
    max_diff = 0.0
    for x in x_vals:
        for y in y_vals:
            p_joint = joint.get((x, y), 0)
            p_product = p_x[x] * p_y[y]
            diff = abs(p_joint - p_product)
            max_diff = max(max_diff, diff)
            if diff > 1e-12:
                independent = False

    print(f"    Max |P(X=x,Y=y) - P(X=x)P(Y=y)| = {max_diff:.6f}")
    print(f"    X and Y are {'INDEPENDENT' if independent else 'DEPENDENT'}.\n")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
