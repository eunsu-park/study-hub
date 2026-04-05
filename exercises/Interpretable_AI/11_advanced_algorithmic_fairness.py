"""
Exercises for Lesson 11: Advanced Algorithmic Fairness
Topic: Interpretable_AI

Solutions to practice problems from the lesson.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# === Exercise 1: Computing Individual Fairness Violations ===
# Problem: Given a distance metric on individuals, check whether similar
# individuals receive similar predictions (Lipschitz condition).

def exercise_1():
    """Compute individual fairness violations using the Lipschitz condition."""
    np.random.seed(42)

    # Generate individuals with features: [credit_score, income, debt_ratio]
    n = 200
    X = np.random.randn(n, 3)
    X[:, 0] = X[:, 0] * 100 + 600  # Credit score ~ N(600, 100)
    X[:, 1] = np.abs(X[:, 1]) * 30000 + 20000  # Income
    X[:, 2] = np.abs(X[:, 2]) * 0.3  # Debt ratio

    # Train a model
    y = (0.01 * X[:, 0] + 0.00002 * X[:, 1] - 3 * X[:, 2] > 5).astype(int)
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]

    # Normalize features for distance computation
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

    # Check individual fairness: d_X(x_i, x_j) small => |f(x_i) - f(x_j)| small
    # Lipschitz condition: |f(x_i) - f(x_j)| <= L * d_X(x_i, x_j)
    lipschitz_threshold = 2.0  # Maximum allowed Lipschitz constant
    distance_threshold = 0.5   # Only check "similar" pairs

    violations = 0
    total_pairs_checked = 0
    max_ratio = 0.0
    violation_examples = []

    for i in range(min(n, 100)):
        for j in range(i + 1, min(n, 100)):
            d_input = np.linalg.norm(X_norm[i] - X_norm[j])
            if d_input < distance_threshold and d_input > 1e-8:
                d_output = abs(probs[i] - probs[j])
                ratio = d_output / d_input
                total_pairs_checked += 1

                if ratio > lipschitz_threshold:
                    violations += 1
                    if len(violation_examples) < 3:
                        violation_examples.append((i, j, d_input, d_output, ratio))

                max_ratio = max(max_ratio, ratio)

    print(f"  Individual Fairness Analysis:")
    print(f"    Pairs checked (distance < {distance_threshold}): {total_pairs_checked}")
    print(f"    Violations (Lipschitz > {lipschitz_threshold}): {violations}")
    print(f"    Violation rate: {violations / max(total_pairs_checked, 1):.2%}")
    print(f"    Max Lipschitz ratio: {max_ratio:.4f}")

    if violation_examples:
        print(f"\n  Example violations:")
        for i, j, d_in, d_out, ratio in violation_examples:
            print(f"    Pair ({i}, {j}): input_dist={d_in:.4f}, "
                  f"output_diff={d_out:.4f}, ratio={ratio:.4f}")

    print(f"\n  Individual fairness requires that similar individuals")
    print(f"  (in the task-relevant metric) receive similar outcomes.")


# === Exercise 2: Counterfactual Fairness with a Simple SCM ===
# Problem: Check whether a model's prediction would change if we
# counterfactually changed a protected attribute in a simple SCM.

def exercise_2():
    """Check counterfactual fairness using a simple structural causal model."""
    np.random.seed(42)
    n = 1000

    # SCM: Gender -> Education, Gender -> Income, Education -> Income
    # Gender -> Hired (direct), Education -> Hired, Income -> Hired
    gender = np.random.binomial(1, 0.5, n)  # 0 or 1
    education = 12 + 2 * gender + np.random.normal(0, 2, n)  # Gender affects education
    income = 30000 + 5000 * education + 8000 * gender + np.random.normal(0, 5000, n)
    hired = (0.0001 * income + 0.1 * education + 0.5 * gender > 3).astype(int)

    X = np.column_stack([gender, education, income])
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X, hired)

    # Counterfactual fairness check:
    # For each individual, compute prediction with actual gender
    # and counterfactual prediction with flipped gender (propagating through SCM)
    cf_violations = 0
    total = 0
    prob_diffs = []

    for i in range(n):
        actual_gender = gender[i]
        cf_gender = 1 - actual_gender

        # Counterfactual: recompute downstream variables
        # Education_cf = Education - effect_of_gender + effect_of_cf_gender
        education_cf = education[i] - 2 * actual_gender + 2 * cf_gender
        # Income_cf
        income_cf = income[i] - 8000 * actual_gender + 8000 * cf_gender \
                    - 5000 * 2 * actual_gender + 5000 * 2 * cf_gender

        x_actual = np.array([[actual_gender, education[i], income[i]]])
        x_cf = np.array([[cf_gender, education_cf, income_cf]])

        prob_actual = model.predict_proba(x_actual)[0, 1]
        prob_cf = model.predict_proba(x_cf)[0, 1]
        diff = abs(prob_actual - prob_cf)
        prob_diffs.append(diff)

        pred_actual = int(prob_actual >= 0.5)
        pred_cf = int(prob_cf >= 0.5)
        if pred_actual != pred_cf:
            cf_violations += 1
        total += 1

    print(f"  Counterfactual Fairness Analysis:")
    print(f"    Total individuals: {total}")
    print(f"    Counterfactual prediction flips: {cf_violations} ({cf_violations/total:.2%})")
    print(f"    Mean |P(Y|actual) - P(Y|counterfactual)|: {np.mean(prob_diffs):.4f}")
    print(f"    Max probability difference: {np.max(prob_diffs):.4f}")

    # Break down by original gender
    for g in [0, 1]:
        mask = gender == g
        mean_diff = np.mean([prob_diffs[i] for i in range(n) if mask[i]])
        print(f"    Gender={g} mean prob difference: {mean_diff:.4f}")

    print(f"\n  The model is NOT counterfactually fair because predictions")
    print(f"  change when gender is counterfactually altered through the SCM.")
    print(f"  A counterfactually fair model should be invariant to such changes.")


# === Exercise 3: Proving the Impossibility Theorem (Binary Case) ===
# Problem: Show that for a binary classifier with imperfect predictions,
# it is impossible to simultaneously satisfy demographic parity, equalized
# odds, and predictive parity across groups with different base rates.

def exercise_3():
    """Demonstrate the impossibility theorem for fairness in a binary case."""
    np.random.seed(42)

    # Two groups with different base rates
    n_a, n_b = 500, 500
    base_rate_a = 0.6  # Group A: 60% positive
    base_rate_b = 0.3  # Group B: 30% positive

    y_a = np.random.binomial(1, base_rate_a, n_a)
    y_b = np.random.binomial(1, base_rate_b, n_b)

    # Classifier with fixed accuracy (TPR=0.8, FPR=0.15 for both groups)
    tpr, fpr = 0.8, 0.15

    def simulate_predictions(y_true, tpr_val, fpr_val):
        preds = np.zeros_like(y_true)
        for i, yt in enumerate(y_true):
            if yt == 1:
                preds[i] = np.random.binomial(1, tpr_val)
            else:
                preds[i] = np.random.binomial(1, fpr_val)
        return preds

    pred_a = simulate_predictions(y_a, tpr, fpr)
    pred_b = simulate_predictions(y_b, tpr, fpr)

    def compute_metrics(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {
            "positive_rate": (tp + fp) / len(y_true),
            "tpr": tp / max(tp + fn, 1),
            "fpr": fp / max(fp + tn, 1),
            "ppv": tp / max(tp + fp, 1),
            "npv": tn / max(tn + fn, 1),
        }

    metrics_a = compute_metrics(y_a, pred_a)
    metrics_b = compute_metrics(y_b, pred_b)

    print("  Group Statistics:")
    print(f"    Group A base rate: {np.mean(y_a):.3f}, "
          f"Group B base rate: {np.mean(y_b):.3f}")

    print(f"\n  With equal TPR and FPR across groups:")
    print(f"  {'Metric':<22} {'Group A':>10} {'Group B':>10} {'Gap':>10}")
    print(f"  {'-' * 52}")

    criteria = {
        "Demographic Parity": ("positive_rate", "="),
        "Equal Opportunity": ("tpr", "="),
        "Equalized Odds (FPR)": ("fpr", "="),
        "Predictive Parity": ("ppv", "="),
    }

    for name, (key, _) in criteria.items():
        val_a = metrics_a[key]
        val_b = metrics_b[key]
        gap = abs(val_a - val_b)
        satisfied = "OK" if gap < 0.05 else "FAIL"
        print(f"  {name:<22} {val_a:>10.4f} {val_b:>10.4f} {gap:>8.4f} [{satisfied}]")

    print(f"\n  Mathematical proof sketch (binary case):")
    print(f"    Let br_A, br_B be base rates with br_A != br_B.")
    print(f"    Demographic parity: TPR_g * br_g + FPR_g * (1-br_g) = c for both g.")
    print(f"    Equalized odds: TPR_A = TPR_B and FPR_A = FPR_B.")
    print(f"    Predictive parity: PPV_A = PPV_B.")
    print(f"    PPV_g = TPR_g * br_g / (TPR_g * br_g + FPR_g * (1-br_g)).")
    print(f"    If TPR_A = TPR_B and FPR_A = FPR_B but br_A != br_B,")
    print(f"    then PPV_A != PPV_B (violates predictive parity).")
    print(f"    Simultaneously satisfying all three is impossible")
    print(f"    unless the classifier is perfect or base rates are equal.")


# === Exercise 4: Auditing a Model with Fairlearn-style MetricFrame ===
# Problem: Implement a simplified MetricFrame that computes fairness metrics
# across subgroups and identifies disparities.

def exercise_4():
    """Audit a model using a simplified MetricFrame approach."""
    np.random.seed(42)

    # Generate dataset with protected attributes
    n = 1000
    age_group = np.random.choice(["young", "middle", "senior"], n, p=[0.3, 0.5, 0.2])
    gender = np.random.choice(["M", "F"], n)

    # Features (unrelated to protected attributes for simplicity)
    X = np.random.randn(n, 4)
    # But introduce bias: model slightly favors "middle" age and "M" gender
    bias = np.zeros(n)
    bias[age_group == "middle"] += 0.3
    bias[gender == "M"] += 0.2
    y = (X[:, 0] + X[:, 1] + bias + np.random.normal(0, 0.5, n) > 0.5).astype(int)

    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X, y)
    y_pred = model.predict(X)

    # Simplified MetricFrame implementation
    class SimpleMetricFrame:
        def __init__(self, metrics, y_true, y_pred, sensitive_features):
            self.metrics = metrics
            self.y_true = y_true
            self.y_pred = y_pred
            self.sensitive_features = sensitive_features
            self.results = {}
            self._compute()

        def _compute(self):
            groups = {}
            for i, key in enumerate(self.sensitive_features):
                if key not in groups:
                    groups[key] = []
                groups[key].append(i)

            for group_name, indices in sorted(groups.items()):
                yt = self.y_true[indices]
                yp = self.y_pred[indices]
                self.results[group_name] = {}
                for metric_name, metric_fn in self.metrics.items():
                    self.results[group_name][metric_name] = metric_fn(yt, yp)

        def overall(self):
            result = {}
            for metric_name, metric_fn in self.metrics.items():
                result[metric_name] = metric_fn(self.y_true, self.y_pred)
            return result

        def difference(self):
            diffs = {}
            for metric_name in self.metrics:
                values = [self.results[g][metric_name] for g in self.results]
                diffs[metric_name] = max(values) - min(values)
            return diffs

        def ratio(self):
            ratios = {}
            for metric_name in self.metrics:
                values = [self.results[g][metric_name] for g in self.results]
                min_val = min(values)
                max_val = max(values)
                ratios[metric_name] = min_val / max_val if max_val > 0 else 0.0
            return ratios

    # Define metrics
    def selection_rate(y_true, y_pred):
        return np.mean(y_pred)

    def true_positive_rate(y_true, y_pred):
        mask = y_true == 1
        if mask.sum() == 0:
            return 0.0
        return np.mean(y_pred[mask])

    def false_positive_rate(y_true, y_pred):
        mask = y_true == 0
        if mask.sum() == 0:
            return 0.0
        return np.mean(y_pred[mask])

    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    metrics = {
        "selection_rate": selection_rate,
        "tpr": true_positive_rate,
        "fpr": false_positive_rate,
        "accuracy": accuracy,
    }

    # Audit by age group
    print("  === Audit by Age Group ===")
    mf_age = SimpleMetricFrame(metrics, y, y_pred, age_group)

    print(f"  {'Group':<10}", end="")
    for m in metrics:
        print(f"  {m:>16}", end="")
    print()

    for group, group_metrics in sorted(mf_age.results.items()):
        print(f"  {group:<10}", end="")
        for m in metrics:
            print(f"  {group_metrics[m]:>16.4f}", end="")
        print()

    diffs = mf_age.difference()
    print(f"  {'Max Diff':<10}", end="")
    for m in metrics:
        print(f"  {diffs[m]:>16.4f}", end="")
    print()

    ratios = mf_age.ratio()
    print(f"  {'Min Ratio':<10}", end="")
    for m in metrics:
        print(f"  {ratios[m]:>16.4f}", end="")
    print()

    # Audit by gender
    print(f"\n  === Audit by Gender ===")
    mf_gender = SimpleMetricFrame(metrics, y, y_pred, gender)

    print(f"  {'Group':<10}", end="")
    for m in metrics:
        print(f"  {m:>16}", end="")
    print()

    for group, group_metrics in sorted(mf_gender.results.items()):
        print(f"  {group:<10}", end="")
        for m in metrics:
            print(f"  {group_metrics[m]:>16.4f}", end="")
        print()

    diffs_g = mf_gender.difference()
    print(f"\n  Disparity summary:")
    for m in metrics:
        flag = " [ALERT]" if diffs_g[m] > 0.1 else ""
        print(f"    Gender {m} gap: {diffs_g[m]:.4f}{flag}")

    print(f"\n  Four-fifths rule check (ratio >= 0.8):")
    ratios_g = mf_gender.ratio()
    for m in metrics:
        status = "PASS" if ratios_g[m] >= 0.8 else "FAIL"
        print(f"    Gender {m} ratio: {ratios_g[m]:.4f} [{status}]")


if __name__ == "__main__":
    print("=== Exercise 1: Individual Fairness Violations ===")
    exercise_1()
    print("\n=== Exercise 2: Counterfactual Fairness with SCM ===")
    exercise_2()
    print("\n=== Exercise 3: Impossibility Theorem (Binary Case) ===")
    exercise_3()
    print("\n=== Exercise 4: Model Audit with MetricFrame ===")
    exercise_4()
    print("\nAll exercises completed!")
