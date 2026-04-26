# Symbolic Regression

[← Previous: 23. A/B Testing for ML](23_AB_Testing_for_ML.md) | [Next: Overview →](00_Overview.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what symbolic regression is and how it differs from parametric regression
2. Describe expression trees as the representation for candidate equations
3. Outline how genetic programming searches the space of mathematical expressions
4. Evaluate candidate equations using a Pareto front of accuracy vs. complexity
5. Use PySR and gplearn to discover interpretable equations from data
6. Apply symbolic regression to physics-inspired and engineering problems
7. Compare symbolic regression with black-box ML models for interpretability and generalization

---

Traditional regression fits parameters to a fixed equation form: linear regression assumes `y = wx + b`, polynomial regression assumes `y = Σ wᵢxⁱ`. You choose the structure; the algorithm fills in the numbers. Symbolic regression flips this: it searches over both the structure and the parameters simultaneously, discovering equations like `y = x₁² + sin(x₂)` directly from data. The result is a compact, human-readable formula rather than a black-box model with thousands of parameters.

---

## Theory & Principles

Symbolic regression is fundamentally different from every other algorithm in this curriculum: it searches over the *space of equations* rather than the parameters of a fixed equation. The two dominant approaches — genetic programming and sparse linear regression over a function library (SINDy) — solve this search in completely different ways, with completely different assumptions and trade-offs.

### A. The Search Space: Expression Trees

An equation can be represented as a tree:

```
y = x₁² + sin(x₂)

       +
      / \
     ²   sin
     |    |
    x₁   x₂
```

Internal nodes are *operators* (`+`, `-`, `*`, `/`, `sin`, `exp`, `log`, ...); leaves are *variables* (`x₁`, `x₂`, ...) or *constants* (`1.0`, `π`, ...). The search space is all valid expression trees up to some maximum depth — exponentially large but discrete.

This is a fundamentally different problem from parameter optimization. Linear regression has a smooth, convex loss surface in `ℝ^p`; symbolic regression has a combinatorial search space with no useful gradient. Different algorithmic toolkits are required.

### B. Genetic Programming: Evolution as Search

**Genetic programming (GP)** treats expression trees as the "individuals" in a Darwinian evolutionary process:

```
1. Initialize a population of random expression trees.
2. Evaluate fitness of each tree: f(tree) = -error_on_data + λ · complexity_penalty
3. Select trees probabilistically by fitness (tournament or roulette).
4. Apply genetic operators to selected trees:
     - Crossover: swap subtrees between two parents
     - Mutation: replace a random subtree with a new random subtree
     - Reproduction: copy unchanged
5. Replace the old population with the offspring.
6. Repeat from step 2 until convergence or budget exhausted.
```

The fitness function is the heart of the algorithm. The two terms encode **Pareto trade-off** between accuracy and parsimony:

```
fitness = α · (-MSE)  -  β · complexity(tree)
```

Without the complexity penalty, GP overfits — finding huge trees that memorize the training data. With the right `β`, GP finds the *Pareto frontier*: the set of trees that are best at each level of complexity. The user picks the trade-off afterwards.

### B.1 GP Strengths and Weaknesses

Strengths:
- No assumption about equation form (linear, polynomial, trigonometric, mixed — all in the same search).
- Outputs are interpretable closed-form expressions.
- Can re-discover known physical laws from data (Kepler's laws, F = ma).

Weaknesses:
- Computationally expensive (population × generations × evaluations).
- Stochastic — different runs can find different equations.
- Hyperparameter-heavy (population size, mutation rate, max depth, fitness weights).
- Can fall into local optima of the search space.

Modern implementations (PySR, gplearn) include sophisticated tricks — multi-population islands, adaptive mutation rates, simplification rules — to mitigate these.

### C. SINDy: Sparse Regression Over a Library

**SINDy** (Sparse Identification of Nonlinear Dynamics, Brunton et al., 2016) takes a completely different approach. Instead of searching expression trees, it builds a *library* `Θ(x)` of candidate functions:

```
Θ(x) = [1, x₁, x₂, x₁², x₁·x₂, x₂², sin(x₁), cos(x₁), exp(x₂), ...]
```

Then it solves a linear regression with **sparsity constraint**:

```
y = Θ(x) · ξ           where ‖ξ‖_0 should be small
```

Most coefficients in `ξ` are forced to zero, so the discovered equation uses only a few library terms. The sparsity is what gives interpretability — a 100-term polynomial regression is not interpretable; one with 3 nonzero terms is.

The optimization typically uses **STLSQ** (Sequentially Thresholded Least Squares):

```
ξ = least-squares solve of y = Θ ξ
loop:
    set entries of ξ with |ξ_i| < threshold to zero
    re-solve least squares on the remaining entries
until convergence
```

This is a discrete approximation of the L0-penalized regression, faster than exact L0 and more sparse than L1 (Lasso) typically gives.

### C.1 SINDy Strengths and Weaknesses

Strengths:
- Fast: a single linear regression instead of an evolutionary search.
- Deterministic.
- Excellent for dynamical systems (`dx/dt = f(x)`) where the library can include time derivatives, polynomial nonlinearities, and trigonometric forcing.
- The sparsity threshold is the only critical hyperparameter.

Weaknesses:
- The library must be specified in advance; if the right term is not in the library, SINDy cannot find it.
- Library size grows combinatorially in input dimension and term complexity.
- Works best when the true equation is *exactly* sparse in the chosen library; struggles with novel non-linearities.

SINDy and GP are complementary: SINDy when you have domain knowledge about likely terms, GP when you do not.

### D. The Pareto Frontier: Accuracy vs Complexity

Both methods produce families of candidate equations along a Pareto frontier:

```
        higher complexity
            |
   error    |  ●         ← complex, accurate
            |    ●
            |      ●
            |        ●   ← simple, less accurate
            |          ●
            +----------- complexity
```

The "best" equation depends on what you value. For physics discovery, you typically want the simplest equation that adequately fits — Occam's razor. For prediction, you might pick the most accurate within a complexity budget. PySR returns the entire frontier; you choose.

### E. Where Symbolic Regression Wins

Symbolic regression is not a replacement for tabular ML. Gradient-boosted trees still dominate for opaque-but-accurate predictions. Symbolic regression wins when:

- **Discoverable structure exists**: the underlying process really is governed by a compact equation (physics, chemistry, biology, engineering).
- **Interpretability is a hard constraint**: regulatory, scientific publication, embedded-system deployment.
- **Extrapolation matters**: a closed-form equation can extrapolate beyond the training distribution; tree models cannot.
- **Compact deployment matters**: an evolved equation is bytes; a tree ensemble is megabytes.

When *none* of these hold, gradient boosting will out-predict symbolic regression at a fraction of the search cost. Symbolic regression is a specialty tool, not a general-purpose default.

### F. Validation: Symbolic Regression Has the Same Statistical Pitfalls

Despite its different search algorithm, symbolic regression is still ML and is subject to the same evaluation discipline:

- **Train/test split**: required, just as for any model.
- **Cross-validation**: harder because each fit is expensive, but possible (small `K`, parallelize folds).
- **Beware of multiple-comparison effects**: searching over many equations and reporting only the best is exactly the same statistical sin as peeking in A/B testing. The reported test error is biased downward.
- **Held-out validation set for the Pareto frontier**: pick the spot on the frontier using a validation set, then report the chosen equation's error on a separate test set.

The discipline matters even more here because GP can produce equations that look elegant and overfit simultaneously — the symbolic form is reassuring but the test error is what matters.

### From Theory to the Code Below

- Section 2's expression-tree visualization and operator/terminal sets are the search-space definition from (A).
- Section 3's `gplearn.SymbolicRegressor` or `pysr.PySRRegressor` runs the genetic programming loop from (B); the `parsimony_coefficient` parameter is the `β` in the fitness function.
- Section 4's `pysindy.SINDy` builds the function library and runs STLSQ from (C); `threshold` is the sparsity parameter.
- Section 5's Pareto-frontier plot and equation selection is the trade-off explorer from (D).
- Section 6's "when to use symbolic regression" guidance maps to the niche analysis in (E).

---

## 1. Core Concepts

### 1.1 What is Symbolic Regression?

```python
"""
Standard Regression vs Symbolic Regression

Standard (Parametric) Regression:
  - You choose the model family: y = w0 + w1*x + w2*x^2
  - Algorithm finds optimal parameters: w0=1.2, w1=-0.5, w2=3.1
  - Fixed structure, optimized coefficients

Symbolic Regression:
  - Algorithm searches over BOTH structure AND parameters
  - Input: data (X, y)
  - Output: y = x1^2 + sin(x2)  (discovered automatically)
  - Variable structure, variable coefficients

Key Advantage:
  - Produces interpretable, closed-form expressions
  - Can generalize beyond training distribution
  - Discovered equations may reveal underlying physics
"""
```

### 1.2 Expression Trees

Every mathematical expression has a natural tree representation:

```python
"""
Expression: y = x1^2 + sin(x2)

        [+]
       /   \
     [^]   [sin]
    /   \     |
  [x1]  [2] [x2]

Nodes:
- Internal nodes: operators (+, -, *, /, ^, sin, cos, exp, log, ...)
- Leaf nodes: variables (x1, x2, ...) or constants (2, 3.14, ...)

The search space is the set of ALL valid expression trees
up to some maximum depth.
"""

# Expression tree node
class Node:
    def __init__(self, op=None, value=None, left=None, right=None):
        self.op = op          # '+', '-', '*', '/', 'sin', 'cos', ...
        self.value = value    # For leaf nodes: variable name or constant
        self.left = left
        self.right = right

    def evaluate(self, variables):
        """Recursively evaluate the expression tree."""
        if self.value is not None:
            if isinstance(self.value, str):
                return variables[self.value]
            return self.value

        left_val = self.left.evaluate(variables)

        if self.op in ('sin', 'cos', 'exp', 'log', 'sqrt', 'abs'):
            import numpy as np
            return getattr(np, self.op)(left_val)

        right_val = self.right.evaluate(variables)
        if self.op == '+': return left_val + right_val
        if self.op == '-': return left_val - right_val
        if self.op == '*': return left_val * right_val
        if self.op == '/':
            return np.where(np.abs(right_val) > 1e-10,
                            left_val / right_val, 0.0)
        if self.op == '^': return np.power(left_val, right_val)

    def __str__(self):
        if self.value is not None:
            return str(self.value)
        if self.op in ('sin', 'cos', 'exp', 'log', 'sqrt', 'abs'):
            return f"{self.op}({self.left})"
        return f"({self.left} {self.op} {self.right})"

    @property
    def complexity(self):
        """Count total number of nodes."""
        if self.value is not None:
            return 1
        c = 1 + self.left.complexity
        if self.right:
            c += self.right.complexity
        return c
```

---

## 2. Genetic Programming

### 2.1 Algorithm Overview

```python
"""
Genetic Programming for Symbolic Regression

1. INITIALIZE: Generate random population of expression trees
2. EVALUATE: Compute fitness = f(accuracy, complexity) for each tree
3. SELECT: Choose parents via tournament selection
4. CROSSOVER: Swap subtrees between two parents
5. MUTATE: Randomly modify nodes in offspring
6. REPLACE: New generation replaces old
7. REPEAT: Until convergence or max generations

Key genetic operators:

Crossover (subtree swap):
  Parent A:  [+]           Parent B:  [*]
            / \                      / \
          [x1] [sin]              [x2] [3]
                 |
               [x2]

  Child:    [+]           (x2 subtree from B replaces sin(x2) in A)
           / \
         [x1] [x2]

Mutation types:
  - Point mutation: change operator (+ → *)
  - Subtree mutation: replace subtree with new random tree
  - Constant mutation: perturb a numeric constant
  - Hoist mutation: replace tree with one of its subtrees (simplification)
"""
```

### 2.2 Minimal GP Implementation

```python
import numpy as np
import random

BINARY_OPS = ['+', '-', '*', '/']
UNARY_OPS = ['sin', 'cos']
ALL_OPS = BINARY_OPS + UNARY_OPS

def random_tree(variables, max_depth=4, depth=0):
    """Generate a random expression tree."""
    # Base case: leaf node
    if depth >= max_depth or (depth > 0 and random.random() < 0.3):
        if random.random() < 0.6:
            return Node(value=random.choice(variables))
        else:
            return Node(value=round(random.uniform(-5, 5), 2))

    op = random.choice(ALL_OPS)
    left = random_tree(variables, max_depth, depth + 1)

    if op in UNARY_OPS:
        return Node(op=op, left=left)
    else:
        right = random_tree(variables, max_depth, depth + 1)
        return Node(op=op, left=left, right=right)


def crossover(parent1, parent2):
    """Swap random subtrees between two parents."""
    import copy
    child = copy.deepcopy(parent1)

    # Collect all nodes with their parent references
    def get_nodes(node, parent=None, attr=None):
        result = [(node, parent, attr)]
        if node.left:
            result.extend(get_nodes(node.left, node, 'left'))
        if node.right:
            result.extend(get_nodes(node.right, node, 'right'))
        return result

    nodes1 = get_nodes(child)
    nodes2 = get_nodes(copy.deepcopy(parent2))

    # Pick random crossover points
    _, p1_parent, p1_attr = random.choice(nodes1[1:]) if len(nodes1) > 1 else nodes1[0]
    donor, _, _ = random.choice(nodes2)

    if p1_parent and p1_attr:
        setattr(p1_parent, p1_attr, donor)

    return child


def mutate(tree, variables, mutation_rate=0.1):
    """Apply point mutation to random nodes."""
    import copy
    tree = copy.deepcopy(tree)

    def _mutate(node):
        if random.random() < mutation_rate:
            if node.value is not None:
                # Mutate leaf
                if random.random() < 0.5:
                    node.value = random.choice(variables)
                else:
                    node.value = round(random.uniform(-5, 5), 2)
            elif node.op:
                # Mutate operator
                if node.op in UNARY_OPS:
                    node.op = random.choice(UNARY_OPS)
                else:
                    node.op = random.choice(BINARY_OPS)
        if node.left:
            _mutate(node.left)
        if node.right:
            _mutate(node.right)

    _mutate(tree)
    return tree


def fitness(tree, X, y):
    """RMSE as fitness (lower is better)."""
    try:
        variables = {f'x{i}': X[:, i] for i in range(X.shape[1])}
        y_pred = tree.evaluate(variables)
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return float('inf')
        return np.sqrt(np.mean((y - y_pred) ** 2))
    except Exception:
        return float('inf')


def symbolic_regression(X, y, pop_size=200, generations=50, max_depth=4):
    """Run genetic programming for symbolic regression."""
    variables = [f'x{i}' for i in range(X.shape[1])]

    # Initialize population
    population = [random_tree(variables, max_depth) for _ in range(pop_size)]

    best_overall = None
    best_fitness = float('inf')

    for gen in range(generations):
        # Evaluate fitness
        scores = [(tree, fitness(tree, X, y)) for tree in population]
        scores.sort(key=lambda x: x[1])

        # Track best
        if scores[0][1] < best_fitness:
            best_fitness = scores[0][1]
            best_overall = scores[0][0]

        if gen % 10 == 0:
            print(f"Gen {gen:3d}: best RMSE = {scores[0][1]:.6f}, "
                  f"expr = {scores[0][0]}")

        # Selection + reproduction
        new_pop = [scores[0][0]]  # Elitism: keep best

        while len(new_pop) < pop_size:
            # Tournament selection
            tournament = random.sample(scores, k=5)
            p1 = min(tournament, key=lambda x: x[1])[0]
            tournament = random.sample(scores, k=5)
            p2 = min(tournament, key=lambda x: x[1])[0]

            child = crossover(p1, p2)
            child = mutate(child, variables)

            # Depth limit
            if child.complexity <= 2 ** (max_depth + 1):
                new_pop.append(child)

        population = new_pop

    return best_overall, best_fitness
```

---

## 3. Pareto Front: Accuracy vs. Complexity

### 3.1 Multi-Objective Optimization

```python
"""
Why not just minimize error?
  → Without complexity penalty, GP produces bloated expressions
  → y = (x + 0.001) * (1/0.001) - x + sin(0) + ... (overfitting noise)

Pareto Front:
  - Plot: x-axis = complexity (number of nodes), y-axis = error (RMSE)
  - Pareto-optimal: no other expression is BOTH simpler AND more accurate

  Error
  │
  │ ●                          ← complex but accurate
  │   ●
  │     ●  ● ← Pareto front
  │        ●
  │           ●
  │              ●             ← simple but inaccurate
  └──────────────────── Complexity

The "knee" of the Pareto front often gives the best trade-off:
  - Enough accuracy to be useful
  - Simple enough to be interpretable
"""

def pareto_front(population, X, y):
    """Extract Pareto-optimal expressions (accuracy vs complexity)."""
    results = []
    for tree in population:
        rmse = fitness(tree, X, y)
        if rmse < float('inf'):
            results.append((tree, rmse, tree.complexity))

    # Sort by complexity
    results.sort(key=lambda x: x[2])

    # Extract Pareto front
    front = []
    best_rmse = float('inf')
    for tree, rmse, comp in results:
        if rmse < best_rmse:
            front.append((tree, rmse, comp))
            best_rmse = rmse

    return front
```

### 3.2 Complexity Measures

| Measure | Description | Example |
|---------|-------------|---------|
| Node count | Total nodes in expression tree | `x + sin(y)` → 4 |
| Tree depth | Maximum depth | `x + sin(y)` → 2 |
| Description length | Bits to encode expression | MDL-based |
| Number of operations | Count of operator nodes | `x + sin(y)` → 2 |

---

## 4. Tools: PySR and gplearn

### 4.1 PySR

```python
"""
PySR (Python Symbolic Regression):
  - Built on Julia's SymbolicRegression.jl (high performance)
  - Automatically manages the Pareto front
  - Supports custom operators, constraints, dimensional analysis
  - pip install pysr
"""
from pysr import PySRRegressor
import numpy as np

# Generate data: y = x0^2 + sin(x1)
np.random.seed(42)
X = np.random.randn(200, 2)
y = X[:, 0]**2 + np.sin(X[:, 1])

# Configure and run
model = PySRRegressor(
    niterations=40,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "exp", "square"],
    populations=8,
    population_size=50,
    maxsize=20,           # Max expression complexity
    parsimony=0.0032,     # Complexity penalty
    random_state=42,
)

model.fit(X, y)

# Results: Pareto front of equations
print(model)
# Complexity | Loss       | Equation
# 1          | 1.234      | x0
# 3          | 0.567      | x0^2
# 5          | 0.089      | x0^2 + sin(x1)   ← discovered!

# Best equation
print(f"Best: {model.sympy()}")

# Predict with discovered equation
y_pred = model.predict(X)
```

### 4.2 gplearn

```python
"""
gplearn:
  - Pure Python, sklearn-compatible API
  - Simpler than PySR but more limited
  - Good for quick experiments and sklearn pipelines
  - pip install gplearn
"""
from gplearn.genetic import SymbolicRegressor

est = SymbolicRegressor(
    population_size=1000,
    generations=20,
    tournament_size=20,
    stopping_criteria=0.01,
    function_set=['add', 'sub', 'mul', 'div', 'sin', 'cos'],
    metric='mse',
    parsimony_coefficient=0.001,
    random_state=42,
    verbose=1,
)

est.fit(X, y)

print(f"Program: {est._program}")
print(f"Fitness: {est._program.fitness_}")

# sklearn pipeline integration
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('sr', SymbolicRegressor(generations=20, random_state=42)),
])
pipe.fit(X, y)
```

---

## 5. Applications

### 5.1 Physics Equation Discovery

```python
"""
Rediscovering Physical Laws from Data

Example: Newton's Law of Gravitation
  - Given: mass1, mass2, distance, measured force
  - Discover: F = G * m1 * m2 / r^2

Example: Kepler's Third Law
  - Given: orbital period, semi-major axis
  - Discover: T^2 ∝ a^3

Example: Ohm's Law
  - Given: voltage, current, resistance measurements
  - Discover: V = I * R

This approach has been used in real research:
  - AI Feynman (Udrescu & Tegmark, 2020): Rediscovered 100 physics equations
  - SINDy (Brunton et al., 2016): Discovered governing differential equations
  - PDE-Net: Learned partial differential equations from simulation data
"""

# Toy example: discover F = m * a
np.random.seed(42)
n = 500
mass = np.random.uniform(1, 100, n)
acceleration = np.random.uniform(0.1, 10, n)
force = mass * acceleration + np.random.normal(0, 0.5, n)  # Noise

X_physics = np.column_stack([mass, acceleration])

# With PySR
# model = PySRRegressor(
#     niterations=40,
#     binary_operators=["+", "-", "*", "/"],
#     maxsize=10,
# )
# model.fit(X_physics, force)
# Expected output: x0 * x1  (i.e., mass * acceleration)
```

### 5.2 Feature Engineering with Symbolic Regression

```python
"""
Use symbolic regression to discover new features for downstream ML:

1. Run SR on (X, y) → get top-k Pareto-optimal expressions
2. Evaluate each expression on X → new feature columns
3. Append to original X → enhanced feature matrix
4. Train standard ML model on enhanced features

This combines interpretability of SR with power of gradient boosting.
"""
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# Original features
X_original = np.random.randn(500, 3)
y = X_original[:, 0]**2 + np.sin(X_original[:, 1]) * X_original[:, 2]

# Suppose SR discovered these expressions:
sr_feature_1 = X_original[:, 0]**2
sr_feature_2 = np.sin(X_original[:, 1])

X_enhanced = np.column_stack([X_original, sr_feature_1, sr_feature_2])

# Compare
gb_original = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_enhanced = GradientBoostingRegressor(n_estimators=100, random_state=42)

score_orig = cross_val_score(gb_original, X_original, y, cv=5,
                             scoring='neg_mean_squared_error')
score_enh = cross_val_score(gb_enhanced, X_enhanced, y, cv=5,
                            scoring='neg_mean_squared_error')

print(f"Original features MSE:  {-score_orig.mean():.4f}")
print(f"Enhanced features MSE:  {-score_enh.mean():.4f}")
```

---

## 6. Comparison with Black-Box Models

### 6.1 When to Use Symbolic Regression

| Criterion | Symbolic Regression | Black-Box ML |
|-----------|-------------------|--------------|
| Interpretability | High (closed-form equation) | Low (SHAP/LIME for post-hoc) |
| Extrapolation | Often good (if true law found) | Poor (interpolation only) |
| High-dimensional data | Weak (>10 features is hard) | Strong |
| Large datasets | Slow (GP search is expensive) | Fast (gradient-based) |
| Noise tolerance | Moderate | High |
| Domain knowledge | Can encode operator constraints | Feature engineering |
| Output | Mathematical formula | Prediction function |

### 6.2 Limitations

```python
"""
Limitations of Symbolic Regression:

1. Scalability: GP search is O(pop_size * generations * data_size)
   - Practical limit: ~10 input features, ~10k samples
   - For larger problems, use SR for feature discovery, then ML

2. Search space explosion:
   - With 4 binary ops, 2 unary ops, 5 variables, depth 5:
   - Possible trees > 10^10
   - No guarantee of finding the global optimum

3. Overfitting:
   - Complex expressions can memorize noise
   - Pareto front / parsimony pressure is essential

4. Numeric instability:
   - Division by near-zero, exp overflow
   - Protected operators needed: div(a,b) = a/b if |b|>ε else 0

5. Constant optimization:
   - GP is bad at tuning numeric constants
   - Modern tools (PySR) use gradient descent for constants
"""
```

---

## 7. Modern Advances

### 7.1 Neural-Guided Symbolic Regression

```python
"""
Hybrid approaches combining neural networks with symbolic search:

1. AI Feynman (2020):
   - Neural network identifies symmetries and separability
   - Reduces search space before symbolic regression
   - Rediscovered 100 physics equations from the Feynman Lectures

2. Deep Symbolic Regression (Petersen et al., 2021):
   - RNN generates expression trees token-by-token
   - Trained with reinforcement learning (reward = fitness)
   - Faster than GP for some problem classes

3. Symbolic GPT / E2E Transformers (Kamienny et al., 2022):
   - Transformer trained on (data, equation) pairs
   - Given new data, predicts equation in one forward pass
   - Orders of magnitude faster than iterative search

4. SymbolicRegression.jl (Cranmer, 2023):
   - Backend for PySR
   - Multi-population evolutionary search
   - Gradient-optimized constants
   - State-of-the-art on SRBench benchmark
"""
```

### 7.2 SINDy: Sparse Identification of Nonlinear Dynamics

```python
"""
SINDy (Brunton et al., 2016):
  - Discovers governing differential equations: dx/dt = f(x)
  - Builds a library of candidate terms: [1, x, x^2, sin(x), ...]
  - Sparse regression (LASSO) selects active terms
  - Not GP-based: uses sparsity instead of tree search

Applications:
  - Fluid dynamics: Navier-Stokes approximation
  - Biological systems: Lotka-Volterra from population data
  - Control systems: data-driven model discovery
"""
import numpy as np

def sindy(X, X_dot, candidate_library, threshold=0.1):
    """
    Sparse Identification of Nonlinear Dynamics.

    Args:
        X: state measurements (n_samples, n_features)
        X_dot: time derivatives (n_samples, n_features)
        candidate_library: function that builds library from X
        threshold: sparsity threshold for sequential thresholding

    Returns:
        coefficients: sparse coefficient matrix
    """
    # Build library: [1, x1, x2, x1^2, x1*x2, x2^2, sin(x1), ...]
    Theta = candidate_library(X)

    # Sparse regression via sequential thresholded least-squares
    n_targets = X_dot.shape[1]
    Xi = np.linalg.lstsq(Theta, X_dot, rcond=None)[0]

    for _ in range(10):  # Iterate until convergence
        for j in range(n_targets):
            small = np.abs(Xi[:, j]) < threshold
            Xi[small, j] = 0
            big = ~small
            if np.any(big):
                Xi[big, j] = np.linalg.lstsq(
                    Theta[:, big], X_dot[:, j], rcond=None
                )[0]

    return Xi
```

---

## 8. Summary

| Concept | Description |
|---------|-------------|
| Symbolic regression | Search for mathematical expressions that fit data |
| Expression tree | Tree representation of equations (operators + operands) |
| Genetic programming | Evolutionary search: selection, crossover, mutation |
| Pareto front | Trade-off curve between accuracy and complexity |
| PySR | High-performance symbolic regression (Julia backend) |
| gplearn | sklearn-compatible GP for symbolic regression |
| SINDy | Sparse regression for discovering differential equations |
| Neural-guided SR | Hybrid neural + symbolic approaches (AI Feynman, DSR) |

### Symbolic Regression vs. Related Techniques

```
Regression Family
    │
    ├── Parametric: Fixed structure, optimize coefficients
    │       ├── Linear / Polynomial / Logistic
    │       └── Neural Networks (fixed architecture)
    │
    ├── Nonparametric: No fixed structure, data-driven
    │       ├── k-NN Regression
    │       ├── Kernel Methods
    │       └── Gaussian Processes
    │
    └── Symbolic: Search over BOTH structure AND coefficients
            ├── Genetic Programming (GP)
            ├── SINDy (sparse regression on function library)
            └── Neural-Guided SR (transformer/RL-based)
```

---

## References

- Cranmer, M. (2023). "Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl." *arXiv:2305.01582*
- Udrescu, S. M. & Tegmark, M. (2020). "AI Feynman: A Physics-Inspired Method for Symbolic Regression." *Science Advances*
- Brunton, S. L. et al. (2016). "Discovering Governing Equations from Data by Sparse Identification of Nonlinear Dynamical Systems." *PNAS*
- Petersen, B. K. et al. (2021). "Deep Symbolic Regression." *ICLR 2021*
- Kamienny, P. et al. (2022). "End-to-End Symbolic Regression with Transformers." *NeurIPS 2022*
- [PySR Documentation](https://astroautomata.com/PySR/)
- [gplearn Documentation](https://gplearn.readthedocs.io/)
- [SRBench Benchmark](https://cavalab.org/srbench/)
