"""
Exercises for Lesson 08: Counterfactual Explanations
Topic: Interpretable_AI

Solutions to practice problems from the lesson.
"""

import numpy as np


# === Exercise 1: Generate Counterfactuals via Optimization ===
# Problem: Given a binary classifier and a rejected input, find the
# minimal perturbation that flips the decision using gradient-based
# optimization.

def exercise_1():
    """Generate counterfactual explanations via gradient-based optimization."""
    print("=" * 60)
    print("Exercise 1: Counterfactual Generation via Optimization")
    print("=" * 60)

    np.random.seed(42)

    # Binary classifier: loan approval model
    # f(x) = sigmoid(w @ x + b), threshold at 0.5
    feature_names = ["income", "credit_score", "debt_ratio", "employment_years"]
    n_features = len(feature_names)

    w = np.array([0.8, 0.6, -0.9, 0.4])
    b = -2.0

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def predict(x):
        return sigmoid(w @ x + b)

    def predict_class(x):
        return 1 if predict(x) > 0.5 else 0

    # Rejected applicant
    x_original = np.array([0.5, 0.4, 0.7, 0.3])  # normalized features
    original_prob = predict(x_original)
    original_class = predict_class(x_original)

    print(f"\n  Feature names: {feature_names}")
    print(f"  Model weights: {w}")
    print(f"  Model bias: {b}")
    print(f"\n  Original input: {x_original}")
    print(f"  P(approved) = {original_prob:.4f}")
    print(f"  Decision: {'Approved' if original_class == 1 else 'Rejected'}")

    # Optimization: minimize ||x' - x||^2 + lambda * max(0, 0.5 - f(x'))
    # Using projected gradient descent
    x_cf = x_original.copy()
    lr = 0.05
    lambda_flip = 10.0  # weight for flipping constraint
    n_steps = 500

    losses = []
    for step in range(n_steps):
        prob = predict(x_cf)
        # Loss: distance + hinge loss for class flip
        distance = np.sum((x_cf - x_original) ** 2)
        hinge = max(0, 0.5 - prob)
        loss = distance + lambda_flip * hinge

        # Gradients
        sigmoid_grad = prob * (1 - prob)
        grad_distance = 2 * (x_cf - x_original)
        if prob < 0.5:
            grad_hinge = -lambda_flip * sigmoid_grad * w
        else:
            grad_hinge = np.zeros(n_features)

        grad = grad_distance + grad_hinge
        x_cf -= lr * grad

        # Clip to valid range [0, 1]
        x_cf = np.clip(x_cf, 0, 1)

        if step % 100 == 0:
            losses.append(loss)

    cf_prob = predict(x_cf)
    cf_class = predict_class(x_cf)

    print(f"\n  Optimization ({n_steps} steps, lr={lr}, lambda={lambda_flip}):")
    print(f"  Loss progression: {[f'{l:.4f}' for l in losses]}")
    print(f"\n  Counterfactual: {x_cf}")
    print(f"  P(approved) = {cf_prob:.4f}")
    print(f"  Decision: {'Approved' if cf_class == 1 else 'Rejected'}")

    print(f"\n  Changes needed:")
    print(f"  {'Feature':<20} {'Original':<12} {'Counterfactual':<16} {'Change':<12}")
    print("  " + "-" * 60)
    for i in range(n_features):
        change = x_cf[i] - x_original[i]
        direction = "increase" if change > 0.01 else ("decrease" if change < -0.01 else "~same")
        print(f"  {feature_names[i]:<20} {x_original[i]:<12.4f} "
              f"{x_cf[i]:<16.4f} {change:+.4f} ({direction})")

    l1_dist = np.sum(np.abs(x_cf - x_original))
    l2_dist = np.sqrt(np.sum((x_cf - x_original) ** 2))
    print(f"\n  L1 distance: {l1_dist:.4f}")
    print(f"  L2 distance: {l2_dist:.4f}")


# === Exercise 2: Counterfactual Quality Metrics ===
# Problem: Evaluate counterfactual explanations using multiple quality
# metrics: validity, proximity, sparsity, plausibility, and stability.

def exercise_2():
    """Compute counterfactual quality metrics."""
    print("\n" + "=" * 60)
    print("Exercise 2: Counterfactual Quality Metrics")
    print("=" * 60)

    feature_names = ["income", "credit_score", "debt_ratio", "employment_years"]
    n_features = len(feature_names)

    w = np.array([0.8, 0.6, -0.9, 0.4])
    b = -2.0

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def predict(x):
        return sigmoid(w @ x + b)

    # Original rejected applicant
    x_original = np.array([0.5, 0.4, 0.7, 0.3])

    # Three candidate counterfactuals
    counterfactuals = {
        "CF1 (minimal change)": np.array([0.65, 0.55, 0.55, 0.35]),
        "CF2 (single feature)": np.array([0.5, 0.4, 0.1, 0.3]),
        "CF3 (large change)":   np.array([0.9, 0.9, 0.1, 0.9]),
    }

    # Training data distribution (for plausibility check)
    np.random.seed(42)
    n_train = 500
    train_data = np.random.beta(2, 2, size=(n_train, n_features))
    train_mean = np.mean(train_data, axis=0)
    train_cov = np.cov(train_data.T)
    train_cov_inv = np.linalg.inv(train_cov + 1e-6 * np.eye(n_features))

    def mahalanobis_distance(x, mean, cov_inv):
        diff = x - mean
        return np.sqrt(diff @ cov_inv @ diff)

    print(f"\n  Original: {x_original}")
    print(f"  P(approved) = {predict(x_original):.4f}")

    print(f"\n  {'Metric':<22}", end="")
    for name in counterfactuals:
        print(f"  {name:<22}", end="")
    print()
    print("  " + "-" * (22 + 24 * len(counterfactuals)))

    # Metric 1: Validity (does it flip the class?)
    print(f"  {'Validity':<22}", end="")
    for name, cf in counterfactuals.items():
        prob = predict(cf)
        valid = prob > 0.5
        print(f"  {str(valid) + f' (p={prob:.3f})':<22}", end="")
    print()

    # Metric 2: Proximity (L1 distance)
    print(f"  {'L1 Proximity':<22}", end="")
    for name, cf in counterfactuals.items():
        l1 = np.sum(np.abs(cf - x_original))
        print(f"  {l1:<22.4f}", end="")
    print()

    # Metric 3: L2 distance
    print(f"  {'L2 Proximity':<22}", end="")
    for name, cf in counterfactuals.items():
        l2 = np.sqrt(np.sum((cf - x_original) ** 2))
        print(f"  {l2:<22.4f}", end="")
    print()

    # Metric 4: Sparsity (number of features changed > threshold)
    threshold = 0.02
    print(f"  {'Sparsity (# changed)':<22}", end="")
    for name, cf in counterfactuals.items():
        n_changed = np.sum(np.abs(cf - x_original) > threshold)
        print(f"  {n_changed:<22}", end="")
    print()

    # Metric 5: Plausibility (Mahalanobis distance from training data)
    print(f"  {'Plausibility (Mahal.)':<22}", end="")
    for name, cf in counterfactuals.items():
        md = mahalanobis_distance(cf, train_mean, train_cov_inv)
        print(f"  {md:<22.4f}", end="")
    print()

    # Metric 6: Stability (sensitivity to small input changes)
    perturbation = np.random.randn(n_features) * 0.01
    x_perturbed = x_original + perturbation
    print(f"  {'Stability':<22}", end="")
    for name, cf in counterfactuals.items():
        # Recompute CF for perturbed input (approximate: same direction)
        cf_perturbed = x_perturbed + (cf - x_original)
        stability = 1.0 / (1.0 + np.sum((cf_perturbed - cf) ** 2))
        print(f"  {stability:<22.4f}", end="")
    print()

    print(f"\n  Analysis:")
    print(f"  - CF1: Balanced changes across features, moderate proximity, plausible.")
    print(f"  - CF2: Sparse (1 feature changed), but large single-feature change")
    print(f"    may be less plausible in practice.")
    print(f"  - CF3: Achieves flip easily but requires extreme changes,")
    print(f"    poor proximity, and may suggest unrealistic actions.")


# === Exercise 3: Implementing Actionability Constraints ===
# Problem: Generate counterfactuals with actionability constraints:
# some features are immutable, some can only increase, etc.

def exercise_3():
    """Implement actionability constraints in counterfactual generation."""
    print("\n" + "=" * 60)
    print("Exercise 3: Actionability Constraints")
    print("=" * 60)

    feature_names = ["age", "income", "credit_score", "debt_ratio",
                     "employment_years", "education_level"]
    n_features = len(feature_names)

    # Constraints for each feature
    constraints = {
        "age":              {"type": "immutable",      "reason": "Cannot change age"},
        "income":           {"type": "increase_only",  "reason": "Income can grow, not shrink arbitrarily"},
        "credit_score":     {"type": "increase_only",  "reason": "Can improve credit, not worsen it"},
        "debt_ratio":       {"type": "any",            "reason": "Can increase or decrease debt"},
        "employment_years": {"type": "increase_only",  "reason": "Experience only grows"},
        "education_level":  {"type": "increase_only",  "reason": "Can gain more education"},
    }

    w = np.array([0.1, 0.7, 0.5, -0.8, 0.3, 0.4])
    b = -2.5

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def predict(x):
        return sigmoid(w @ x + b)

    # Rejected applicant
    x_original = np.array([0.35, 0.4, 0.3, 0.6, 0.2, 0.3])

    print(f"\n  Original applicant:")
    for i, fname in enumerate(feature_names):
        ctype = constraints[fname]["type"]
        print(f"    {fname:<20} = {x_original[i]:.2f}  "
              f"[constraint: {ctype}]")
    print(f"  P(approved) = {predict(x_original):.4f}")

    # Optimization with constraints
    x_cf_unconstrained = x_original.copy()
    x_cf_constrained = x_original.copy()
    lr = 0.05
    lambda_flip = 15.0
    n_steps = 800

    for step in range(n_steps):
        # Unconstrained CF
        prob_u = predict(x_cf_unconstrained)
        grad_d_u = 2 * (x_cf_unconstrained - x_original)
        if prob_u < 0.5:
            grad_h_u = -lambda_flip * prob_u * (1 - prob_u) * w
        else:
            grad_h_u = np.zeros(n_features)
        x_cf_unconstrained -= lr * (grad_d_u + grad_h_u)
        x_cf_unconstrained = np.clip(x_cf_unconstrained, 0, 1)

        # Constrained CF
        prob_c = predict(x_cf_constrained)
        grad_d_c = 2 * (x_cf_constrained - x_original)
        if prob_c < 0.5:
            grad_h_c = -lambda_flip * prob_c * (1 - prob_c) * w
        else:
            grad_h_c = np.zeros(n_features)
        x_cf_constrained -= lr * (grad_d_c + grad_h_c)
        x_cf_constrained = np.clip(x_cf_constrained, 0, 1)

        # Apply constraints
        for i, fname in enumerate(feature_names):
            ctype = constraints[fname]["type"]
            if ctype == "immutable":
                x_cf_constrained[i] = x_original[i]
            elif ctype == "increase_only":
                x_cf_constrained[i] = max(x_cf_constrained[i], x_original[i])

    print(f"\n  {'Feature':<20} {'Original':<10} {'Unconstrained':<15} "
          f"{'Constrained':<15} {'Constraint':<15}")
    print("  " + "-" * 75)
    for i, fname in enumerate(feature_names):
        ctype = constraints[fname]["type"]
        print(f"  {fname:<20} {x_original[i]:<10.4f} "
              f"{x_cf_unconstrained[i]:<15.4f} "
              f"{x_cf_constrained[i]:<15.4f} {ctype:<15}")

    prob_u = predict(x_cf_unconstrained)
    prob_c = predict(x_cf_constrained)
    print(f"\n  Unconstrained CF: P(approved) = {prob_u:.4f} "
          f"({'Approved' if prob_u > 0.5 else 'Rejected'})")
    print(f"  Constrained CF:   P(approved) = {prob_c:.4f} "
          f"({'Approved' if prob_c > 0.5 else 'Rejected'})")

    # Distance comparison
    l2_u = np.sqrt(np.sum((x_cf_unconstrained - x_original) ** 2))
    l2_c = np.sqrt(np.sum((x_cf_constrained - x_original) ** 2))
    print(f"\n  L2 distance (unconstrained): {l2_u:.4f}")
    print(f"  L2 distance (constrained):   {l2_c:.4f}")

    print(f"\n  Key insight: Constrained counterfactuals are more actionable")
    print(f"  but may require larger overall changes (or may not achieve")
    print(f"  the flip at all if constraints are too restrictive).")
    print(f"  The constrained version only suggests actions the user can")
    print(f"  actually take: 'Increase your income by X and reduce debt by Y'")
    print(f"  rather than 'Be 10 years younger'.")


# === Exercise 4: Diverse vs Single Counterfactuals ===
# Problem: Generate multiple diverse counterfactual explanations and
# compare them against a single closest counterfactual.

def exercise_4():
    """Compare diverse counterfactuals vs single closest counterfactual."""
    print("\n" + "=" * 60)
    print("Exercise 4: Diverse vs Single Counterfactuals")
    print("=" * 60)

    np.random.seed(42)

    feature_names = ["income", "credit_score", "debt_ratio", "employment_years"]
    n_features = len(feature_names)

    w = np.array([0.8, 0.6, -0.9, 0.4])
    b = -2.0

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def predict(x):
        return sigmoid(w @ x + b)

    x_original = np.array([0.5, 0.4, 0.7, 0.3])

    # Generate diverse counterfactuals using DiverseCF-like approach
    # Strategy: optimize multiple CFs with a diversity penalty
    n_diverse = 4

    def generate_diverse_cfs(x_orig, n_cfs, lambda_flip=15.0,
                              lambda_div=2.0, lr=0.05, n_steps=600):
        """Generate diverse counterfactuals with pairwise distance penalty."""
        cfs = []
        for k in range(n_cfs):
            # Initialize with different random perturbations
            x_cf = x_orig + np.random.randn(n_features) * 0.1
            x_cf = np.clip(x_cf, 0, 1)

            for step in range(n_steps):
                prob = predict(x_cf)
                # Distance loss
                grad_dist = 2 * (x_cf - x_orig)
                # Flip loss
                if prob < 0.5:
                    grad_flip = -lambda_flip * prob * (1 - prob) * w
                else:
                    grad_flip = np.zeros(n_features)
                # Diversity loss: repel from existing CFs
                grad_div = np.zeros(n_features)
                for prev_cf in cfs:
                    diff = x_cf - prev_cf
                    dist = np.sum(diff ** 2) + 1e-6
                    # Gradient of -1/dist (maximize distance from previous)
                    grad_div += lambda_div * 2 * diff / (dist ** 2)

                x_cf -= lr * (grad_dist + grad_flip - grad_div)
                x_cf = np.clip(x_cf, 0, 1)

            cfs.append(x_cf.copy())
        return cfs

    diverse_cfs = generate_diverse_cfs(x_original, n_diverse)

    print(f"\n  Original: {x_original}")
    print(f"  P(approved) = {predict(x_original):.4f}")

    print(f"\n  Diverse Counterfactual Explanations:")
    print(f"  {'CF':<6} ", end="")
    for fname in feature_names:
        print(f"{fname:<16} ", end="")
    print(f"{'P(approved)':<14} {'L2 dist':<10} {'Valid?':<8}")
    print("  " + "-" * 90)

    for k, cf in enumerate(diverse_cfs):
        prob = predict(cf)
        l2 = np.sqrt(np.sum((cf - x_original) ** 2))
        valid = prob > 0.5
        print(f"  CF{k+1:<3} ", end="")
        for i in range(n_features):
            change = cf[i] - x_original[i]
            indicator = "^" if change > 0.02 else ("v" if change < -0.02 else " ")
            print(f"{cf[i]:.3f}{indicator:<12} ", end="")
        print(f"{prob:<14.4f} {l2:<10.4f} {str(valid):<8}")

    # Feature change summary across diverse CFs
    print(f"\n  Feature change patterns across diverse CFs:")
    for i, fname in enumerate(feature_names):
        changes = [cf[i] - x_original[i] for cf in diverse_cfs]
        n_increase = sum(1 for c in changes if c > 0.02)
        n_decrease = sum(1 for c in changes if c < -0.02)
        n_same = n_diverse - n_increase - n_decrease
        print(f"    {fname:<20} Increased in {n_increase}/{n_diverse} CFs, "
              f"Decreased in {n_decrease}/{n_diverse}, Same in {n_same}/{n_diverse}")

    # Pairwise diversity among CFs
    print(f"\n  Pairwise L2 distances between diverse CFs:")
    print(f"  {'':8}", end="")
    for k in range(n_diverse):
        print(f"{'CF' + str(k+1):<10}", end="")
    print()
    for i in range(n_diverse):
        print(f"  {'CF' + str(i+1):<8}", end="")
        for j in range(n_diverse):
            if i == j:
                print(f"{'---':<10}", end="")
            else:
                dist = np.sqrt(np.sum((diverse_cfs[i] - diverse_cfs[j]) ** 2))
                print(f"{dist:<10.4f}", end="")
        print()

    print(f"\n  Benefits of diverse counterfactuals:")
    print(f"  1. Multiple actionable paths: User can choose the most feasible")
    print(f"     option (e.g., increase income OR improve credit score)")
    print(f"  2. Reveal model sensitivity: Consistent feature changes across")
    print(f"     CFs indicate robust model behavior; inconsistency suggests")
    print(f"     the decision boundary is complex in that region")
    print(f"  3. Fairness audit: If all CFs require changing a protected")
    print(f"     attribute, it signals potential bias")
    print(f"  4. User autonomy: People differ in which changes are feasible,")
    print(f"     so providing options respects individual circumstances")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
