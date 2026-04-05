"""
Exercises for Lesson 06: Advanced SHAP
Topic: Interpretable_AI

Solutions to practice problems from the lesson.
"""

import numpy as np
from itertools import combinations


# === Exercise 1: Compute DeepSHAP for a 2-Layer Network by Hand ===
# Problem: Given a 2-layer neural network with known weights, compute
# DeepSHAP attributions by propagating Shapley values layer by layer.

def exercise_1():
    """Compute DeepSHAP attributions for a 2-layer network by hand."""
    print("=" * 60)
    print("Exercise 1: DeepSHAP for a 2-Layer Network")
    print("=" * 60)

    # Network: 2 inputs (x1, x2) -> 2 hidden (ReLU) -> 1 output
    # f(x) = w3*relu(w1*x1 + w2*x2 + b1) + w4*relu(w5*x1 + w6*x2 + b2) + b3
    w1, w2, b1 = 1.0, 2.0, 0.0    # hidden neuron 1
    w5, w6, b2 = -1.0, 1.0, 0.5   # hidden neuron 2
    w3, w4, b3 = 1.5, 0.5, 0.0    # output neuron

    x = np.array([3.0, 1.0])
    baseline = np.array([0.0, 0.0])

    def relu(z):
        return max(0.0, z)

    def forward(inp):
        h1 = relu(w1 * inp[0] + w2 * inp[1] + b1)
        h2 = relu(w5 * inp[0] + w6 * inp[1] + b2)
        return w3 * h1 + w4 * h2 + b3

    f_x = forward(x)
    f_base = forward(baseline)

    print(f"\n  Network: f(x1,x2) = 1.5*relu(x1 + 2*x2) + 0.5*relu(-x1 + x2 + 0.5)")
    print(f"  Input:    x = {x}")
    print(f"  Baseline: b = {baseline}")
    print(f"  f(x) = {f_x:.4f}")
    print(f"  f(baseline) = {f_base:.4f}")
    print(f"  Difference: {f_x - f_base:.4f}")

    # DeepSHAP: DeepLIFT multipliers propagated using Shapley-like rules
    # Step 1: Compute intermediate values
    z1_x = w1 * x[0] + w2 * x[1] + b1      # hidden 1 pre-relu at x
    z1_b = w1 * baseline[0] + w2 * baseline[1] + b1
    h1_x = relu(z1_x)
    h1_b = relu(z1_b)

    z2_x = w5 * x[0] + w6 * x[1] + b2
    z2_b = w5 * baseline[0] + w6 * baseline[1] + b2
    h2_x = relu(z2_x)
    h2_b = relu(z2_b)

    print(f"\n  Hidden neuron 1: z1(x)={z1_x}, h1(x)={h1_x}, z1(b)={z1_b}, h1(b)={h1_b}")
    print(f"  Hidden neuron 2: z2(x)={z2_x}, h2(x)={h2_x}, z2(b)={z2_b}, h2(b)={h2_b}")

    # Step 2: DeepLIFT multipliers for ReLU
    # m = (relu(z_x) - relu(z_b)) / (z_x - z_b)  (rescale rule)
    m1 = (h1_x - h1_b) / (z1_x - z1_b) if abs(z1_x - z1_b) > 1e-10 else 0
    m2 = (h2_x - h2_b) / (z2_x - z2_b) if abs(z2_x - z2_b) > 1e-10 else 0

    print(f"\n  DeepLIFT multipliers (rescale rule):")
    print(f"    m1 = (h1(x) - h1(b)) / (z1(x) - z1(b)) = {m1:.4f}")
    print(f"    m2 = (h2(x) - h2(b)) / (z2(x) - z2(b)) = {m2:.4f}")

    # Step 3: Propagate attributions back to inputs
    # Attribution to x1 through h1: w3 * m1 * w1 * (x1 - b1)
    # Attribution to x2 through h1: w3 * m1 * w2 * (x2 - b2)
    attr_x1_via_h1 = w3 * m1 * w1 * (x[0] - baseline[0])
    attr_x2_via_h1 = w3 * m1 * w2 * (x[1] - baseline[1])
    attr_x1_via_h2 = w4 * m2 * w5 * (x[0] - baseline[0])
    attr_x2_via_h2 = w4 * m2 * w6 * (x[1] - baseline[1])

    attr_x1 = attr_x1_via_h1 + attr_x1_via_h2
    attr_x2 = attr_x2_via_h1 + attr_x2_via_h2

    print(f"\n  Attributions through hidden neuron 1:")
    print(f"    x1: w3*m1*w1*(x1-b1) = {attr_x1_via_h1:.4f}")
    print(f"    x2: w3*m1*w2*(x2-b2) = {attr_x2_via_h1:.4f}")
    print(f"  Attributions through hidden neuron 2:")
    print(f"    x1: w4*m2*w5*(x1-b1) = {attr_x1_via_h2:.4f}")
    print(f"    x2: w4*m2*w6*(x2-b2) = {attr_x2_via_h2:.4f}")
    print(f"\n  Total DeepSHAP attributions:")
    print(f"    x1: {attr_x1:.4f}")
    print(f"    x2: {attr_x2:.4f}")
    print(f"    Sum: {attr_x1 + attr_x2:.4f}")
    print(f"    Target (f(x) - f(b)): {f_x - f_base:.4f}")
    print(f"    Completeness: {abs(attr_x1 + attr_x2 - (f_x - f_base)) < 1e-6}")


# === Exercise 2: SHAP Interaction Values ===
# Problem: Compute SHAP interaction values for a function with known
# feature interactions. Verify that main effects + interactions = SHAP.

def exercise_2():
    """Calculate SHAP interaction values for a model with interactions."""
    print("\n" + "=" * 60)
    print("Exercise 2: SHAP Interaction Values")
    print("=" * 60)

    # f(x1, x2, x3) = 2*x1 + 3*x2 + x3 + 4*x1*x2  (interaction between x1, x2)
    def f(x):
        return 2 * x[0] + 3 * x[1] + x[2] + 4 * x[0] * x[1]

    x = np.array([1.0, 2.0, 0.5])
    baseline = np.array([0.0, 0.0, 0.0])
    n_features = 3

    print(f"\n  f(x1,x2,x3) = 2*x1 + 3*x2 + x3 + 4*x1*x2")
    print(f"  x = {x}, baseline = {baseline}")
    print(f"  f(x) = {f(x):.4f}, f(baseline) = {f(baseline):.4f}")

    # Compute exact Shapley values by enumerating all coalitions
    def shapley_values(f, x, baseline, n):
        from math import factorial
        phi = np.zeros(n)
        for i in range(n):
            for size in range(n):
                # All subsets of N\{i} of given size
                others = [j for j in range(n) if j != i]
                for subset in combinations(others, size):
                    subset_set = set(subset)
                    # Marginal contribution of i to subset
                    z_with = baseline.copy()
                    z_without = baseline.copy()
                    for j in subset_set:
                        z_with[j] = x[j]
                        z_without[j] = x[j]
                    z_with[i] = x[i]
                    marginal = f(z_with) - f(z_without)
                    # Weight
                    weight = (factorial(size) * factorial(n - size - 1)
                              / factorial(n))
                    phi[i] += weight * marginal
        return phi

    phi = shapley_values(f, x, baseline, n_features)

    print(f"\n  Shapley values:")
    for i in range(n_features):
        print(f"    phi(x{i+1}) = {phi[i]:.4f}")
    print(f"    Sum = {sum(phi):.4f} (should = {f(x) - f(baseline):.4f})")

    # SHAP interaction values: phi_ij
    # phi_ij = sum over S (weight * delta_ij(S))
    # delta_ij(S) = f(S+{i,j}) - f(S+{i}) - f(S+{j}) + f(S)
    def shap_interactions(f, x, baseline, n):
        from math import factorial
        phi_int = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                others = [k for k in range(n) if k != i and k != j]
                for size in range(len(others) + 1):
                    for subset in combinations(others, size):
                        subset_set = set(subset)
                        # Build coalition vectors
                        z_base = baseline.copy()
                        for k in subset_set:
                            z_base[k] = x[k]

                        z_ij = z_base.copy()
                        z_ij[i] = x[i]
                        z_ij[j] = x[j]

                        z_i = z_base.copy()
                        z_i[i] = x[i]

                        z_j = z_base.copy()
                        z_j[j] = x[j]

                        delta = f(z_ij) - f(z_i) - f(z_j) + f(z_base)
                        weight = (factorial(size) * factorial(n - size - 2)
                                  / (2 * factorial(n - 1)))
                        phi_int[i, j] += weight * delta
        # Main (diagonal) effects
        for i in range(n):
            phi_int[i, i] = phi[i] - sum(phi_int[i, j] for j in range(n) if j != i)
        return phi_int

    phi_int = shap_interactions(f, x, baseline, n_features)

    print(f"\n  SHAP Interaction matrix:")
    header = "         " + "  ".join(f"x{j+1:>6}" for j in range(n_features))
    print(f"  {header}")
    for i in range(n_features):
        row = "  ".join(f"{phi_int[i, j]:8.4f}" for j in range(n_features))
        print(f"    x{i+1}:  {row}")

    print(f"\n  Interpretation:")
    print(f"    - Diagonal entries: main effects (direct feature contributions)")
    print(f"    - Off-diagonal entries: interaction effects")
    print(f"    - phi_int[0,1] = phi_int[1,0] = {phi_int[0,1]:.4f} "
          f"(x1*x2 interaction)")
    print(f"    - phi_int[0,2] and phi_int[1,2] ~ 0 (no x1*x3 or x2*x3 terms)")
    print(f"    - Row sums equal Shapley values:")
    for i in range(n_features):
        row_sum = sum(phi_int[i])
        print(f"      sum(row {i}) = {row_sum:.4f}, phi(x{i+1}) = {phi[i]:.4f}")


# === Exercise 3: Interventional vs Observational Conditionals ===
# Problem: Compare SHAP under interventional (feature independence)
# vs observational (feature correlation) conditional expectations.

def exercise_3():
    """Compare interventional vs observational SHAP conditionals."""
    print("\n" + "=" * 60)
    print("Exercise 3: Interventional vs Observational Conditionals")
    print("=" * 60)

    np.random.seed(42)

    # Model: f(x1, x2) = x1 + x2  (simple additive)
    # But x1 and x2 are correlated in the data!

    n_samples = 1000
    # Generate correlated features: x2 = 0.8*x1 + noise
    x1 = np.random.randn(n_samples)
    x2 = 0.8 * x1 + 0.2 * np.random.randn(n_samples)
    data = np.column_stack([x1, x2])
    correlation = np.corrcoef(x1, x2)[0, 1]

    def f(x):
        return x[0] + x[1]

    # Test point
    x_test = np.array([2.0, 1.6])
    f_test = f(x_test)

    print(f"\n  Model: f(x1, x2) = x1 + x2")
    print(f"  Data correlation: r(x1, x2) = {correlation:.3f}")
    print(f"  Test point: x = {x_test}, f(x) = {f_test:.2f}")

    # Interventional SHAP: assume features are independent
    # E[f | do(x1=v)] = v + E[x2] (x2 sampled from marginal)
    mean_x1 = np.mean(x1)
    mean_x2 = np.mean(x2)
    baseline_pred = f(np.array([mean_x1, mean_x2]))

    # Interventional: marginal expectations
    # phi_1 = E[f(x_test_1, X2)] - E[f(X1, X2)]
    #       = x_test_1 + E[X2] - (E[X1] + E[X2])
    #       = x_test_1 - E[X1]
    phi1_interv = x_test[0] - mean_x1
    phi2_interv = x_test[1] - mean_x2

    print(f"\n  Interventional SHAP (assumes independence):")
    print(f"    E[X1] = {mean_x1:.3f}, E[X2] = {mean_x2:.3f}")
    print(f"    phi_1 = x1 - E[X1] = {phi1_interv:.3f}")
    print(f"    phi_2 = x2 - E[X2] = {phi2_interv:.3f}")
    print(f"    Sum = {phi1_interv + phi2_interv:.3f}")

    # Observational SHAP: condition on actual data distribution
    # E[f | x1=v] = v + E[x2 | x1=v] (x2 sampled conditionally)
    # Since x2 ~ 0.8*x1 + noise:
    # E[x2 | x1=v] ~ 0.8*v
    cond_mean_x2_given_x1 = 0.8 * x_test[0]
    cond_mean_x1_given_x2 = x_test[1] / 0.8  # approximate inverse

    # Observational Shapley values (computed analytically for this case)
    # phi_1 = 0.5 * [f(x1, x2) - f(E[x1|x2], x2)] +
    #         0.5 * [f(x1, E[x2|x1]) - f(E[x1], E[x2])]
    f_x1_condx2 = f(np.array([cond_mean_x1_given_x2, x_test[1]]))
    f_condx2_x1 = f(np.array([x_test[0], cond_mean_x2_given_x1]))

    phi1_obs = 0.5 * (f_test - f(np.array([cond_mean_x1_given_x2, x_test[1]]))) + \
               0.5 * (f(np.array([x_test[0], cond_mean_x2_given_x1])) - baseline_pred)
    phi2_obs = (f_test - baseline_pred) - phi1_obs

    print(f"\n  Observational SHAP (respects correlation):")
    print(f"    E[X2 | X1={x_test[0]}] ~ {cond_mean_x2_given_x1:.3f}")
    print(f"    phi_1 = {phi1_obs:.3f}")
    print(f"    phi_2 = {phi2_obs:.3f}")
    print(f"    Sum = {phi1_obs + phi2_obs:.3f}")

    print(f"\n  Comparison:")
    print(f"  {'Method':<25} {'phi_1':<10} {'phi_2':<10} {'Sum':<10}")
    print("  " + "-" * 55)
    print(f"  {'Interventional':<25} {phi1_interv:<10.3f} "
          f"{phi2_interv:<10.3f} {phi1_interv + phi2_interv:<10.3f}")
    print(f"  {'Observational':<25} {phi1_obs:<10.3f} "
          f"{phi2_obs:<10.3f} {phi1_obs + phi2_obs:<10.3f}")

    print(f"\n  Key insight: When features are correlated, interventional SHAP")
    print(f"  (KernelSHAP default) treats them as independent, potentially")
    print(f"  creating unrealistic feature combinations. Observational SHAP")
    print(f"  respects the data distribution but can 'leak' information between")
    print(f"  correlated features, complicating causal interpretation.")


# === Exercise 4: TreeSHAP Bias in Correlated Features ===
# Problem: Demonstrate how TreeSHAP (interventional) can produce biased
# attributions when features are highly correlated, and a redundant
# feature absorbs credit.

def exercise_4():
    """Analyze TreeSHAP bias with correlated features."""
    print("\n" + "=" * 60)
    print("Exercise 4: TreeSHAP Bias in Correlated Features")
    print("=" * 60)

    np.random.seed(42)

    # Simulate a scenario: true relationship is y = f(x1) = 3*x1
    # But we also have x2 = x1 + small_noise (redundant correlated feature)
    # A tree might split on either x1 or x2 arbitrarily

    n_samples = 200
    x1 = np.random.randn(n_samples)
    x2 = x1 + np.random.randn(n_samples) * 0.05  # near-perfect copy
    y = 3 * x1  # true model only depends on x1

    # Simulate a decision tree that splits on x1 and x2 roughly equally
    # (because they are nearly identical, the tree can use either)

    # Simplified tree prediction: uses both features
    def tree_predict(x1_val, x2_val):
        """Simulated tree that arbitrarily uses both correlated features."""
        if x1_val > 0:
            if x2_val > 0.5:
                return 2.5
            else:
                return 1.5
        else:
            if x2_val < -0.5:
                return -2.5
            else:
                return -1.0

    # Compute interventional TreeSHAP-like attributions
    # (approximate by sampling marginals independently)
    def approx_shap(x1_val, x2_val, data_x1, data_x2, n_bg=100):
        """Approximate SHAP values by marginal sampling."""
        n = min(n_bg, len(data_x1))
        f_full = tree_predict(x1_val, x2_val)
        f_baseline = np.mean([tree_predict(data_x1[i], data_x2[i])
                              for i in range(n)])

        # E[f(x1, X2)] - marginalizing over x2
        f_x1_marginal = np.mean([tree_predict(x1_val, data_x2[i])
                                 for i in range(n)])
        # E[f(X1, x2)] - marginalizing over x1
        f_x2_marginal = np.mean([tree_predict(data_x1[i], x2_val)
                                 for i in range(n)])

        phi_1 = 0.5 * (f_full - f_x2_marginal) + 0.5 * (f_x1_marginal - f_baseline)
        phi_2 = 0.5 * (f_full - f_x1_marginal) + 0.5 * (f_x2_marginal - f_baseline)
        return phi_1, phi_2

    # Test on several points
    test_points = [
        (2.0, 2.05),
        (1.0, 0.98),
        (-1.5, -1.52),
        (0.3, 0.28),
    ]

    print(f"\n  True model: y = 3*x1 (x2 is irrelevant but correlated)")
    print(f"  Correlation: r(x1, x2) = {np.corrcoef(x1, x2)[0,1]:.4f}")
    print(f"\n  Interventional SHAP attributions (tree model):")
    print(f"  {'x1':<8} {'x2':<8} {'f(x)':<8} {'phi_1':<10} {'phi_2':<10} "
          f"{'phi2/phi1':<12} {'Issue?':<10}")
    print("  " + "-" * 66)

    for x1_val, x2_val in test_points:
        f_val = tree_predict(x1_val, x2_val)
        phi_1, phi_2 = approx_shap(x1_val, x2_val, x1, x2)
        ratio = abs(phi_2 / phi_1) if abs(phi_1) > 1e-6 else float("inf")
        issue = "YES" if ratio > 0.3 else "no"
        print(f"  {x1_val:<8.2f} {x2_val:<8.2f} {f_val:<8.2f} "
              f"{phi_1:<10.4f} {phi_2:<10.4f} {ratio:<12.4f} {issue:<10}")

    print(f"\n  Problem: Even though x2 is irrelevant to the true relationship,")
    print(f"  interventional TreeSHAP assigns it non-trivial attribution because:")
    print(f"  1. The tree split on x2 (it's interchangeable with x1)")
    print(f"  2. Interventional conditioning breaks the x1-x2 correlation,")
    print(f"     creating impossible combinations (high x1, low x2)")
    print(f"\n  This is the 'feature attribution bias' in TreeSHAP identified")
    print(f"  by Sundararajan & Najmi (2020). Mitigations include:")
    print(f"  - Using observational conditional (TreeSHAP 'path-dependent')")
    print(f"  - Feature selection to remove redundant features before modeling")
    print(f"  - Grouping correlated features and computing group SHAP values")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
