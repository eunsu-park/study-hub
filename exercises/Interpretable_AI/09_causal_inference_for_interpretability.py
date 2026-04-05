"""
Exercises for Lesson 09: Causal Inference for Interpretability
Topic: Interpretable_AI

Solutions to practice problems from the lesson.
"""

import numpy as np
from collections import defaultdict


# === Exercise 1: Build a Structural Causal Model (SCM) / DAG ===
# Problem: Given a loan approval scenario with features (Income, Education,
# Age, Credit_Score, Loan_Amount), construct the DAG adjacency list and
# identify causal paths to the outcome.

def exercise_1():
    """Build an SCM/DAG for a loan approval scenario and analyze causal paths."""
    np.random.seed(42)

    # Define the DAG as adjacency list
    # Nodes: Age, Education, Income, Credit_Score, Loan_Amount, Approved
    dag = {
        "Age": ["Income", "Credit_Score"],
        "Education": ["Income"],
        "Income": ["Credit_Score", "Loan_Amount", "Approved"],
        "Credit_Score": ["Approved"],
        "Loan_Amount": ["Approved"],
        "Approved": [],
    }

    print("  Loan Approval SCM (DAG adjacency list):")
    for node, children in dag.items():
        arrow = " -> " + ", ".join(children) if children else " (outcome)"
        print(f"    {node}{arrow}")

    # Find all causal paths from each feature to Approved
    def find_all_paths(graph, start, end, path=None):
        if path is None:
            path = []
        path = path + [start]
        if start == end:
            return [path]
        paths = []
        for neighbor in graph.get(start, []):
            if neighbor not in path:
                new_paths = find_all_paths(graph, neighbor, end, path)
                paths.extend(new_paths)
        return paths

    print("\n  Causal paths to Approved:")
    features = ["Age", "Education", "Income", "Credit_Score", "Loan_Amount"]
    for feat in features:
        paths = find_all_paths(dag, feat, "Approved")
        for p in paths:
            print(f"    {' -> '.join(p)}")

    # Identify parents of Approved (direct causes)
    parents_of_approved = [node for node, children in dag.items()
                           if "Approved" in children]
    print(f"\n  Direct causes of Approved: {parents_of_approved}")

    # Identify confounders: nodes that are parents of both a feature and the outcome
    # Income is a confounder between Credit_Score and Approved
    print("  Confounder example: Income affects both Credit_Score and Approved")
    print("  To estimate causal effect of Credit_Score on Approved,")
    print("  we must adjust for Income (a backdoor path confounder).")


# === Exercise 2: Backdoor Adjustment Formula ===
# Problem: Apply the backdoor adjustment to estimate the causal effect of
# Treatment (T) on Outcome (Y) given confounder (Z) from observational data.

def exercise_2():
    """Compute causal effect using the backdoor adjustment formula."""
    np.random.seed(42)
    n = 5000

    # Generate data from a known SCM:
    #   Z -> T, Z -> Y, T -> Y
    # True causal effect of T on Y is beta_t = 2.0
    Z = np.random.binomial(1, 0.6, n)          # Confounder
    T = np.random.binomial(1, 0.3 + 0.4 * Z, n)  # Treatment influenced by Z
    Y = 2.0 * T + 3.0 * Z + np.random.normal(0, 0.5, n)  # Outcome

    # Naive (unadjusted) estimate: E[Y|T=1] - E[Y|T=0]
    naive_effect = Y[T == 1].mean() - Y[T == 0].mean()
    print(f"  Naive (unadjusted) estimate: {naive_effect:.4f}")
    print("  (Biased because Z confounds T and Y)")

    # Backdoor adjustment: P(Y|do(T=t)) = sum_z P(Y|T=t, Z=z) * P(Z=z)
    causal_effect = 0.0
    for z_val in [0, 1]:
        # P(Z = z)
        p_z = np.mean(Z == z_val)
        # E[Y | T=1, Z=z] - E[Y | T=0, Z=z]
        mask_t1_z = (T == 1) & (Z == z_val)
        mask_t0_z = (T == 0) & (Z == z_val)
        if mask_t1_z.sum() > 0 and mask_t0_z.sum() > 0:
            e_y_t1_z = Y[mask_t1_z].mean()
            e_y_t0_z = Y[mask_t0_z].mean()
            causal_effect += (e_y_t1_z - e_y_t0_z) * p_z

    print(f"  Backdoor-adjusted estimate: {causal_effect:.4f}")
    print(f"  True causal effect:         2.0000")
    print(f"  Adjustment removes confounding bias from Z.")


# === Exercise 3: Causal vs Observational Feature Importance ===
# Problem: Compare standard correlation-based importance with causal importance
# derived from interventional reasoning.

def exercise_3():
    """Compute causal vs observational feature importance and compare."""
    np.random.seed(42)
    n = 3000

    # SCM: X1 -> Y, X2 -> Y, X1 -> X2 (X1 is a common cause)
    # True effects: X1 -> Y is 1.0, X2 -> Y is 2.0, X1 -> X2 is 3.0
    X1 = np.random.normal(0, 1, n)
    X2 = 3.0 * X1 + np.random.normal(0, 0.5, n)  # X2 is caused by X1
    Y = 1.0 * X1 + 2.0 * X2 + np.random.normal(0, 0.5, n)

    X = np.column_stack([X1, X2])

    # Observational importance: correlation with Y
    corr_x1_y = np.corrcoef(X1, Y)[0, 1]
    corr_x2_y = np.corrcoef(X2, Y)[0, 1]
    print("  Observational Feature Importance (|correlation with Y|):")
    print(f"    X1: {abs(corr_x1_y):.4f}")
    print(f"    X2: {abs(corr_x2_y):.4f}")

    # Causal importance: interventional effect via do-calculus
    # do(X1 = x1+1) vs do(X1 = x1): total effect = direct + indirect through X2
    # Direct effect of X1 on Y = 1.0
    # Indirect effect through X2 = 3.0 * 2.0 = 6.0
    # Total causal effect of X1 = 1.0 + 6.0 = 7.0
    # Direct causal effect of X2 on Y = 2.0

    # Estimate by intervention simulation
    def estimate_causal_effect(feature_idx, delta=1.0):
        X_base = X.copy()
        Y_base = 1.0 * X_base[:, 0] + 2.0 * X_base[:, 1]

        X_intervened = X.copy()
        X_intervened[:, feature_idx] += delta
        # Propagate causal effects
        if feature_idx == 0:
            X_intervened[:, 1] = 3.0 * X_intervened[:, 0] + np.random.normal(0, 0.5, n)
        Y_intervened = 1.0 * X_intervened[:, 0] + 2.0 * X_intervened[:, 1]

        return np.mean(Y_intervened - Y_base) / delta

    causal_x1 = estimate_causal_effect(0)
    causal_x2 = estimate_causal_effect(1)

    print("\n  Causal Feature Importance (interventional effect on Y):")
    print(f"    X1 (total causal effect): {causal_x1:.4f}  (true: 7.0)")
    print(f"    X2 (direct causal effect): {causal_x2:.4f}  (true: 2.0)")
    print("\n  Insight: X1 has higher total causal importance (1 + 3*2 = 7)")
    print("  despite X2 having higher observational correlation,")
    print("  because X1 also affects Y indirectly through X2.")


# === Exercise 4: Detecting Simpson's Paradox ===
# Problem: Given data on treatment success rates across hospitals,
# detect and explain Simpson's paradox.

def exercise_4():
    """Detect Simpson's paradox in treatment effectiveness data."""
    np.random.seed(42)

    # Simulate hospital treatment data where Simpson's paradox occurs
    # Hospital A: treats mostly severe cases
    # Hospital B: treats mostly mild cases
    data = {
        "Hospital_A": {
            "severe": {"treated": 100, "recovered_treated": 70,
                       "untreated": 10, "recovered_untreated": 5},
            "mild":   {"treated": 10,  "recovered_treated": 9,
                       "untreated": 100, "recovered_untreated": 85},
        },
        "Hospital_B": {
            "severe": {"treated": 10,  "recovered_treated": 6,
                       "untreated": 100, "recovered_untreated": 40},
            "mild":   {"treated": 100, "recovered_treated": 95,
                       "untreated": 10, "recovered_untreated": 8},
        },
    }

    print("  === Aggregate Statistics (ignoring severity) ===")
    for hospital, groups in data.items():
        total_treated = sum(g["treated"] for g in groups.values())
        total_recovered_t = sum(g["recovered_treated"] for g in groups.values())
        total_untreated = sum(g["untreated"] for g in groups.values())
        total_recovered_u = sum(g["recovered_untreated"] for g in groups.values())

        rate_treated = total_recovered_t / total_treated
        rate_untreated = total_recovered_u / total_untreated
        print(f"    {hospital}: Treatment recovery={rate_treated:.2%}, "
              f"No-treatment recovery={rate_untreated:.2%}")

    print("\n  === Stratified Statistics (by severity) ===")
    for hospital, groups in data.items():
        print(f"    {hospital}:")
        for severity, counts in groups.items():
            rate_t = counts["recovered_treated"] / counts["treated"]
            rate_u = counts["recovered_untreated"] / counts["untreated"]
            print(f"      {severity}: Treatment={rate_t:.2%}, "
                  f"No-treatment={rate_u:.2%}")

    # Detection algorithm
    print("\n  === Simpson's Paradox Detection ===")
    for hospital, groups in data.items():
        aggregate_treated = sum(g["treated"] for g in groups.values())
        aggregate_recovered = sum(g["recovered_treated"] for g in groups.values())
        aggregate_rate = aggregate_recovered / aggregate_treated

        aggregate_untreated = sum(g["untreated"] for g in groups.values())
        aggregate_unrec = sum(g["recovered_untreated"] for g in groups.values())
        aggregate_unt_rate = aggregate_unrec / aggregate_untreated

        aggregate_better = aggregate_rate > aggregate_unt_rate

        all_strata_agree = True
        for severity, counts in groups.items():
            stratum_better = (counts["recovered_treated"] / counts["treated"] >
                              counts["recovered_untreated"] / counts["untreated"])
            if stratum_better != aggregate_better:
                all_strata_agree = False
                break

        paradox = not all_strata_agree
        status = "PARADOX DETECTED" if paradox else "No paradox"
        print(f"    {hospital}: {status}")

    print("\n  Causal insight: severity is a confounder. The correct causal")
    print("  effect must be estimated by stratifying on severity (backdoor")
    print("  adjustment), not by looking at aggregate rates.")


if __name__ == "__main__":
    print("=== Exercise 1: Build SCM/DAG for Loan Approval ===")
    exercise_1()
    print("\n=== Exercise 2: Backdoor Adjustment Formula ===")
    exercise_2()
    print("\n=== Exercise 3: Causal vs Observational Feature Importance ===")
    exercise_3()
    print("\n=== Exercise 4: Detecting Simpson's Paradox ===")
    exercise_4()
    print("\nAll exercises completed!")
