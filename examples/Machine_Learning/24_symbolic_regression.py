"""
Symbolic Regression with Genetic Programming (from scratch)

Demonstrates:
  1. Expression tree representation and evaluation
  2. Genetic programming: selection, crossover, mutation
  3. Pareto front (accuracy vs complexity)
  4. Discovering equations from synthetic data

No external SR libraries required — pure NumPy implementation.

Expected output:
  - Discovers y = x0^2 + sin(x1) (or close approximation)
  - Pareto front visualization saved to symbolic_regression_results.png
  - Training converges within 100 generations (RMSE < 0.5)

Usage:
  python 24_symbolic_regression.py
"""

import numpy as np
import warnings
import random
import copy
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=RuntimeWarning)  # protected div


# ============================================================
# 1. Expression Tree
# ============================================================

class Node:
    """A node in an expression tree."""

    def __init__(self, op=None, value=None, left=None, right=None):
        self.op = op
        self.value = value
        self.left = left
        self.right = right

    def evaluate(self, variables):
        """Recursively evaluate the tree given variable bindings."""
        if self.value is not None:
            if isinstance(self.value, str):
                return variables[self.value]
            return np.full_like(list(variables.values())[0],
                                self.value, dtype=float)

        left_val = self.left.evaluate(variables)

        # Unary operators
        if self.op == 'sin':
            return np.sin(left_val)
        if self.op == 'cos':
            return np.cos(left_val)

        # Binary operators
        right_val = self.right.evaluate(variables)
        if self.op == '+':
            return left_val + right_val
        if self.op == '-':
            return left_val - right_val
        if self.op == '*':
            return left_val * right_val
        if self.op == '/':
            return np.where(np.abs(right_val) > 1e-10,
                            left_val / right_val, 0.0)

    def __str__(self):
        if self.value is not None:
            if isinstance(self.value, float):
                return f"{self.value:.2f}"
            return str(self.value)
        if self.op in ('sin', 'cos'):
            return f"{self.op}({self.left})"
        return f"({self.left} {self.op} {self.right})"

    @property
    def size(self):
        """Total number of nodes (complexity measure)."""
        if self.value is not None:
            return 1
        s = 1 + self.left.size
        if self.right:
            s += self.right.size
        return s

    def all_nodes(self, parent=None, attr=None):
        """Collect all (node, parent, attr) tuples."""
        result = [(self, parent, attr)]
        if self.left:
            result.extend(self.left.all_nodes(self, 'left'))
        if self.right:
            result.extend(self.right.all_nodes(self, 'right'))
        return result


# ============================================================
# 2. Genetic Operators
# ============================================================

BINARY_OPS = ['+', '-', '*', '/']
UNARY_OPS = ['sin', 'cos']


def random_tree(variables, max_depth=4, depth=0):
    """Generate a random expression tree."""
    if depth >= max_depth or (depth > 0 and random.random() < 0.3):
        if random.random() < 0.6:
            return Node(value=random.choice(variables))
        else:
            return Node(value=round(random.uniform(-3, 3), 1))

    if random.random() < 0.25:
        op = random.choice(UNARY_OPS)
        return Node(op=op, left=random_tree(variables, max_depth, depth + 1))
    else:
        op = random.choice(BINARY_OPS)
        return Node(
            op=op,
            left=random_tree(variables, max_depth, depth + 1),
            right=random_tree(variables, max_depth, depth + 1),
        )


def crossover(p1, p2):
    """Subtree crossover: swap random subtrees between two parents."""
    child = copy.deepcopy(p1)
    donor = copy.deepcopy(p2)

    nodes_c = child.all_nodes()
    nodes_d = donor.all_nodes()

    if len(nodes_c) < 2:
        return child

    _, c_parent, c_attr = random.choice(nodes_c[1:])
    d_node, _, _ = random.choice(nodes_d)

    if c_parent and c_attr:
        setattr(c_parent, c_attr, d_node)

    return child


def mutate(tree, variables, rate=0.15):
    """Point mutation on random nodes."""
    tree = copy.deepcopy(tree)

    def _mut(node):
        if random.random() < rate:
            if node.value is not None:
                if random.random() < 0.5 and variables:
                    node.value = random.choice(variables)
                else:
                    node.value = round(random.uniform(-3, 3), 1)
            elif node.op in UNARY_OPS:
                node.op = random.choice(UNARY_OPS)
            elif node.op in BINARY_OPS:
                node.op = random.choice(BINARY_OPS)
        if node.left:
            _mut(node.left)
        if node.right:
            _mut(node.right)

    _mut(tree)
    return tree


# ============================================================
# 3. Fitness & Selection
# ============================================================

def rmse(tree, X, y):
    """RMSE fitness (lower = better). Returns inf on failure."""
    try:
        variables = {f'x{i}': X[:, i] for i in range(X.shape[1])}
        y_pred = tree.evaluate(variables)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.full_like(y, y_pred)
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return float('inf')
        return float(np.sqrt(np.mean((y - y_pred) ** 2)))
    except Exception:
        return float('inf')


def tournament_select(scored, k=5):
    """Select the best individual from k random candidates."""
    candidates = random.sample(scored, min(k, len(scored)))
    return min(candidates, key=lambda x: x[1])[0]


# ============================================================
# 4. Main GP Loop
# ============================================================

def run_gp(X, y, pop_size=300, generations=100, max_depth=5,
           max_size=25, verbose=True):
    """Run genetic programming for symbolic regression."""
    variables = [f'x{i}' for i in range(X.shape[1])]
    population = [random_tree(variables, max_depth) for _ in range(pop_size)]

    best = None
    best_score = float('inf')
    history = []

    for gen in range(generations):
        scored = [(t, rmse(t, X, y)) for t in population]
        scored.sort(key=lambda x: x[1])

        gen_best_score = scored[0][1]
        if gen_best_score < best_score:
            best_score = gen_best_score
            best = copy.deepcopy(scored[0][0])

        history.append(best_score)

        if verbose and gen % 20 == 0:
            expr_str = str(scored[0][0])
            if len(expr_str) > 60:
                expr_str = expr_str[:57] + "..."
            print(f"  Gen {gen:3d} | RMSE {gen_best_score:.6f} | "
                  f"size {scored[0][0].size:2d} | {expr_str}")

        # Build next generation
        new_pop = [copy.deepcopy(scored[0][0])]  # elitism

        while len(new_pop) < pop_size:
            p1 = tournament_select(scored)
            p2 = tournament_select(scored)
            child = crossover(p1, p2)
            child = mutate(child, variables)
            if child.size <= max_size:
                new_pop.append(child)

        population = new_pop

    return best, best_score, history, scored


def extract_pareto(scored):
    """Extract Pareto front from (tree, rmse) pairs."""
    valid = [(t, s, t.size) for t, s in scored if s < float('inf')]
    valid.sort(key=lambda x: x[2])  # sort by complexity

    front = []
    best_rmse = float('inf')
    for t, s, c in valid:
        if s < best_rmse:
            front.append((t, s, c))
            best_rmse = s

    return front


# ============================================================
# 5. Demonstration
# ============================================================

def main():
    print("=" * 60)
    print("Symbolic Regression via Genetic Programming")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    # --- Task 1: Discover y = x0^2 + sin(x1) ---
    print("\n--- Task 1: Discover y = x0^2 + sin(x1) ---\n")

    n = 300
    X = np.random.uniform(-3, 3, (n, 2))
    y = X[:, 0]**2 + np.sin(X[:, 1])

    best, score, history, scored = run_gp(
        X, y, pop_size=400, generations=100, max_depth=5
    )

    print(f"\n  Best expression: {best}")
    print(f"  RMSE: {score:.6f}")
    print(f"  Complexity: {best.size} nodes")

    # --- Task 2: Discover y = x0 * x1 (F = m * a) ---
    print("\n--- Task 2: Discover y = x0 * x1 (F = m*a) ---\n")

    mass = np.random.uniform(1, 50, n)
    accel = np.random.uniform(0.1, 10, n)
    force = mass * accel + np.random.normal(0, 0.3, n)
    X2 = np.column_stack([mass, accel])

    best2, score2, history2, scored2 = run_gp(
        X2, force, pop_size=300, generations=80, max_depth=4
    )

    print(f"\n  Best expression: {best2}")
    print(f"  RMSE: {score2:.6f}")

    # --- Pareto Front ---
    front = extract_pareto(scored)
    print(f"\n--- Pareto Front (Task 1): {len(front)} expressions ---")
    for t, s, c in front[:8]:
        expr = str(t)
        if len(expr) > 50:
            expr = expr[:47] + "..."
        print(f"  complexity={c:2d}, RMSE={s:.4f}: {expr}")

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Convergence (Task 1)
    axes[0].plot(history, 'b-', linewidth=1.5)
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Best RMSE')
    axes[0].set_title('Task 1: Convergence (x0² + sin(x1))')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # Pareto front
    complexities = [c for _, _, c in front]
    rmses = [s for _, s, _ in front]
    axes[1].scatter(complexities, rmses, c='red', s=60, zorder=5)
    axes[1].plot(complexities, rmses, 'r--', alpha=0.5)
    axes[1].set_xlabel('Complexity (nodes)')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Task 1: Pareto Front')
    axes[1].grid(True, alpha=0.3)

    # Convergence (Task 2)
    axes[2].plot(history2, 'g-', linewidth=1.5)
    axes[2].set_xlabel('Generation')
    axes[2].set_ylabel('Best RMSE')
    axes[2].set_title('Task 2: Convergence (x0 * x1)')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('symbolic_regression_results.png', dpi=150)
    plt.close()
    print("\nResult image saved: symbolic_regression_results.png")


if __name__ == "__main__":
    main()
