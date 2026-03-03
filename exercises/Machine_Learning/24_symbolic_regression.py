"""
Exercises: Symbolic Regression
==============================

Practice problems for understanding and implementing symbolic regression
using genetic programming.

Requirements:
    pip install numpy matplotlib
"""

import numpy as np


# ============================================================
# Exercise 1: Expression Tree Basics
# ============================================================
"""
Build and evaluate expression trees manually.

1. Implement a Node class with evaluate() and __str__() methods
2. Construct the tree for: y = (x0 + x1) * sin(x0)
3. Evaluate it on x0=1.0, x1=2.0 (expected: 3.0 * sin(1.0) ≈ 2.524)
4. Implement a complexity() method that counts total nodes
5. Implement a depth() method that returns maximum tree depth
6. Print the expression, complexity, and depth
"""

# Your code here:


# ============================================================
# Exercise 2: Random Tree Generation
# ============================================================
"""
Implement random expression tree generation.

1. Write random_tree(variables, max_depth) that generates
   random expression trees using:
   - Binary ops: +, -, *, /
   - Unary ops: sin, cos
   - Variables: ['x0', 'x1']
   - Random constants in [-5, 5]
2. Generate 100 random trees with max_depth=4
3. Print statistics: mean/min/max complexity, mean/min/max depth
4. How many trees evaluate without error on random data?
5. Plot a histogram of tree complexities
"""

# Your code here:


# ============================================================
# Exercise 3: Genetic Operators
# ============================================================
"""
Implement and test crossover and mutation.

1. Create two parent trees manually:
   - Parent A: x0 + x1
   - Parent B: sin(x0) * x1
2. Implement subtree crossover: swap a random subtree
3. Run crossover 20 times, print each child expression
4. Implement point mutation (change random operator or leaf)
5. Mutate Parent A 20 times, print results
6. Verify that parents are NOT modified (use deepcopy)
"""

# Your code here:


# ============================================================
# Exercise 4: Fitness Evaluation
# ============================================================
"""
Compare fitness of hand-crafted expressions on a target function.

Target: y = x0^2 - 3*x1

1. Generate 200 data points: x0, x1 ~ Uniform(-5, 5)
2. Build expression trees for these candidates:
   a) y = x0
   b) y = x0 * x0
   c) y = x0 * x0 - x1
   d) y = x0 * x0 - 3 * x1  (true equation)
3. Compute RMSE for each
4. Plot predicted vs actual for each candidate (2x2 subplot)
5. Which candidate has the best accuracy-complexity trade-off?
"""

# Your code here:


# ============================================================
# Exercise 5: Mini Symbolic Regression
# ============================================================
"""
Implement a complete (minimal) GP-based symbolic regression.

Target: y = x0 * sin(x1)  (200 data points, x ~ Uniform(-3, 3))

1. Initialize population of 200 random trees
2. Implement tournament selection (k=5)
3. Run for 60 generations with crossover + mutation
4. Track best RMSE per generation
5. Plot the convergence curve
6. Print the best expression found and its RMSE
7. Compare: how close is the discovered expression to x0 * sin(x1)?
"""

# Your code here:


# ============================================================
# Exercise 6: Pareto Front Analysis
# ============================================================
"""
Analyze the accuracy-complexity trade-off.

Using the final population from Exercise 5:

1. Compute (RMSE, complexity) for every individual
2. Extract the Pareto front (non-dominated solutions)
3. Plot all individuals as gray dots, Pareto front as red
4. Print the Pareto-optimal expressions in order of complexity
5. Identify the "knee" of the Pareto front (largest RMSE drop
   per unit complexity increase)
6. Is the knee expression close to the true function?
"""

# Your code here:
