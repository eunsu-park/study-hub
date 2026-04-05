"""
Exercises for Lesson 12: Fairness Mitigation
Topic: Interpretable_AI

Solutions to practice problems from the lesson.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# === Exercise 1: Sample Reweighing for Fairness ===
# Problem: Implement the reweighing pre-processing technique that assigns
# sample weights to balance selection rates across protected groups.

def exercise_1():
    """Implement sample reweighing to achieve demographic parity."""
    np.random.seed(42)
    n = 1000

    # Generate biased dataset
    group = np.random.binomial(1, 0.5, n)  # Protected attribute (0 or 1)
    X = np.random.randn(n, 4)
    # Bias: group 1 more likely to get positive outcome
    y = (X[:, 0] + X[:, 1] + 0.8 * group + np.random.normal(0, 0.5, n) > 0.5).astype(int)

    # Compute reweighing weights
    # W(g, y) = P(Y=y) * P(G=g) / P(Y=y, G=g)
    def compute_reweighing_weights(group_arr, y_arr):
        weights = np.ones(len(y_arr))
        n_total = len(y_arr)
        for g in [0, 1]:
            for label in [0, 1]:
                p_y = np.mean(y_arr == label)
                p_g = np.mean(group_arr == g)
                p_yg = np.mean((y_arr == label) & (group_arr == g))
                if p_yg > 0:
                    w = (p_y * p_g) / p_yg
                else:
                    w = 1.0
                mask = (group_arr == g) & (y_arr == label)
                weights[mask] = w
        return weights

    weights = compute_reweighing_weights(group, y)

    print("  Reweighing weights per (group, label) combination:")
    for g in [0, 1]:
        for label in [0, 1]:
            mask = (group == g) & (y == label)
            w = weights[mask][0] if mask.sum() > 0 else 0
            count = mask.sum()
            print(f"    Group={g}, Y={label}: weight={w:.4f}, count={count}")

    # Train models with and without reweighing
    X_train, X_test = X[:700], X[700:]
    y_train, y_test = y[:700], y[700:]
    g_train, g_test = group[:700], group[700:]
    w_train = compute_reweighing_weights(g_train, y_train)

    # Unweighted model
    model_unw = LogisticRegression(random_state=42, max_iter=500)
    model_unw.fit(X_train, y_train)
    pred_unw = model_unw.predict(X_test)

    # Weighted model
    model_w = LogisticRegression(random_state=42, max_iter=500)
    model_w.fit(X_train, y_train, sample_weight=w_train[:700])
    pred_w = model_w.predict(X_test)

    print(f"\n  Without reweighing:")
    print(f"    Accuracy: {accuracy_score(y_test, pred_unw):.4f}")
    for g in [0, 1]:
        mask = g_test == g
        sr = np.mean(pred_unw[mask])
        print(f"    Group {g} selection rate: {sr:.4f}")
    dp_gap_unw = abs(np.mean(pred_unw[g_test == 0]) - np.mean(pred_unw[g_test == 1]))
    print(f"    Demographic parity gap: {dp_gap_unw:.4f}")

    print(f"\n  With reweighing:")
    print(f"    Accuracy: {accuracy_score(y_test, pred_w):.4f}")
    for g in [0, 1]:
        mask = g_test == g
        sr = np.mean(pred_w[mask])
        print(f"    Group {g} selection rate: {sr:.4f}")
    dp_gap_w = abs(np.mean(pred_w[g_test == 0]) - np.mean(pred_w[g_test == 1]))
    print(f"    Demographic parity gap: {dp_gap_w:.4f}")
    print(f"\n  Reweighing reduced the DP gap by {dp_gap_unw - dp_gap_w:.4f}.")


# === Exercise 2: Training with a Fairness Penalty (In-processing) ===
# Problem: Add a demographic parity regularization term to the training
# objective and optimize via gradient descent.

def exercise_2():
    """Train a model with a fairness penalty term in the loss function."""
    np.random.seed(42)
    n = 800

    # Generate biased data
    group = np.random.binomial(1, 0.5, n)
    X = np.random.randn(n, 3)
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.6 * group + np.random.normal(0, 0.3, n) > 0.3).astype(float)

    X_train, X_test = X[:600], X[600:]
    y_train, y_test = y[:600], y[600:]
    g_train, g_test = group[:600], group[600:]

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

    def train_fair_model(X_tr, y_tr, g_tr, lam=0.0, lr=0.01, epochs=200):
        """Logistic regression with DP fairness penalty."""
        n_feat = X_tr.shape[1]
        w = np.zeros(n_feat)
        b = 0.0

        for epoch in range(epochs):
            z = X_tr @ w + b
            p = sigmoid(z)

            # Binary cross-entropy gradient
            grad_w = X_tr.T @ (p - y_tr) / len(y_tr)
            grad_b = np.mean(p - y_tr)

            # Fairness penalty: (mean_pred_g0 - mean_pred_g1)^2
            mask_g0 = g_tr == 0
            mask_g1 = g_tr == 1
            mean_p0 = p[mask_g0].mean()
            mean_p1 = p[mask_g1].mean()
            dp_diff = mean_p0 - mean_p1

            # Gradient of fairness penalty w.r.t. w
            dp_grad_w = np.zeros(n_feat)
            p_deriv = p * (1 - p)  # sigmoid derivative
            dp_grad_w += (X_tr[mask_g0].T @ p_deriv[mask_g0]) / mask_g0.sum()
            dp_grad_w -= (X_tr[mask_g1].T @ p_deriv[mask_g1]) / mask_g1.sum()
            dp_grad_w *= 2 * dp_diff

            dp_grad_b = 2 * dp_diff * (
                p_deriv[mask_g0].mean() - p_deriv[mask_g1].mean()
            )

            w -= lr * (grad_w + lam * dp_grad_w)
            b -= lr * (grad_b + lam * dp_grad_b)

        return w, b

    print("  Training with different fairness penalty strengths (lambda):")
    print(f"  {'Lambda':>8} {'Accuracy':>10} {'SR Group 0':>12} {'SR Group 1':>12} {'DP Gap':>8}")

    for lam in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
        w, b = train_fair_model(X_train, y_train, g_train, lam=lam)
        preds = (sigmoid(X_test @ w + b) >= 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        sr0 = np.mean(preds[g_test == 0])
        sr1 = np.mean(preds[g_test == 1])
        dp_gap = abs(sr0 - sr1)
        print(f"  {lam:>8.1f} {acc:>10.4f} {sr0:>12.4f} {sr1:>12.4f} {dp_gap:>8.4f}")

    print(f"\n  Higher lambda reduces DP gap but may decrease accuracy.")
    print(f"  The practitioner must choose the fairness-accuracy trade-off.")


# === Exercise 3: Threshold Optimization (Post-processing) ===
# Problem: Find group-specific classification thresholds that equalize
# a fairness criterion while maximizing overall accuracy.

def exercise_3():
    """Apply threshold optimization to equalize selection rates across groups."""
    np.random.seed(42)
    n = 1000

    # Generate data with group-dependent score distributions
    group = np.random.binomial(1, 0.5, n)
    X = np.random.randn(n, 4)
    y = (X[:, 0] + X[:, 1] - 0.5 * X[:, 2] + 0.7 * group
         + np.random.normal(0, 0.5, n) > 0.3).astype(int)

    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X, y)
    scores = model.predict_proba(X)[:, 1]

    # Default threshold: 0.5 for both groups
    pred_default = (scores >= 0.5).astype(int)
    sr0_default = np.mean(pred_default[group == 0])
    sr1_default = np.mean(pred_default[group == 1])

    print(f"  Default threshold (0.5 for both groups):")
    print(f"    Accuracy: {accuracy_score(y, pred_default):.4f}")
    print(f"    Group 0 selection rate: {sr0_default:.4f}")
    print(f"    Group 1 selection rate: {sr1_default:.4f}")
    print(f"    DP gap: {abs(sr0_default - sr1_default):.4f}")

    # Grid search for group-specific thresholds to equalize selection rates
    target_rate = np.mean(pred_default)  # Target overall selection rate
    best_thresholds = {0: 0.5, 1: 0.5}
    best_dp_gap = float("inf")
    best_acc = 0.0

    thresholds_to_try = np.arange(0.1, 0.9, 0.01)

    for t0 in thresholds_to_try:
        for t1 in thresholds_to_try:
            pred = np.zeros(n, dtype=int)
            pred[group == 0] = (scores[group == 0] >= t0).astype(int)
            pred[group == 1] = (scores[group == 1] >= t1).astype(int)

            sr0 = np.mean(pred[group == 0])
            sr1 = np.mean(pred[group == 1])
            dp_gap = abs(sr0 - sr1)
            acc = accuracy_score(y, pred)

            if dp_gap < best_dp_gap or (dp_gap == best_dp_gap and acc > best_acc):
                best_dp_gap = dp_gap
                best_acc = acc
                best_thresholds = {0: t0, 1: t1}

    # Apply best thresholds
    pred_opt = np.zeros(n, dtype=int)
    pred_opt[group == 0] = (scores[group == 0] >= best_thresholds[0]).astype(int)
    pred_opt[group == 1] = (scores[group == 1] >= best_thresholds[1]).astype(int)
    sr0_opt = np.mean(pred_opt[group == 0])
    sr1_opt = np.mean(pred_opt[group == 1])

    print(f"\n  Optimized group-specific thresholds:")
    print(f"    Group 0 threshold: {best_thresholds[0]:.2f}")
    print(f"    Group 1 threshold: {best_thresholds[1]:.2f}")
    print(f"    Accuracy: {accuracy_score(y, pred_opt):.4f}")
    print(f"    Group 0 selection rate: {sr0_opt:.4f}")
    print(f"    Group 1 selection rate: {sr1_opt:.4f}")
    print(f"    DP gap: {abs(sr0_opt - sr1_opt):.4f}")

    print(f"\n  Threshold optimization reduced DP gap from "
          f"{abs(sr0_default - sr1_default):.4f} to {abs(sr0_opt - sr1_opt):.4f}")
    print(f"  without retraining the model (post-processing).")


# === Exercise 4: Fairness-Accuracy Pareto Frontier ===
# Problem: Vary the fairness constraint strength and plot the Pareto frontier
# of achievable (accuracy, fairness) trade-offs.

def exercise_4():
    """Plot a fairness-accuracy Pareto frontier using text-based visualization."""
    np.random.seed(42)
    n = 800

    group = np.random.binomial(1, 0.5, n)
    X = np.random.randn(n, 4)
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.6 * group
         + np.random.normal(0, 0.4, n) > 0.3).astype(int)

    X_train, X_test = X[:600], X[600:]
    y_train, y_test = y[:600], y[600:]
    g_train, g_test = group[:600], group[600:]

    # Generate Pareto frontier by varying threshold per group
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train, y_train)
    scores_test = model.predict_proba(X_test)[:, 1]

    frontier_points = []
    thresholds = np.arange(0.1, 0.9, 0.05)

    for t0 in thresholds:
        for t1 in thresholds:
            pred = np.zeros(len(X_test), dtype=int)
            pred[g_test == 0] = (scores_test[g_test == 0] >= t0).astype(int)
            pred[g_test == 1] = (scores_test[g_test == 1] >= t1).astype(int)

            acc = accuracy_score(y_test, pred)
            sr0 = np.mean(pred[g_test == 0])
            sr1 = np.mean(pred[g_test == 1])
            dp_gap = abs(sr0 - sr1)

            frontier_points.append((acc, dp_gap, t0, t1))

    # Extract Pareto-optimal points (higher acc, lower dp_gap is better)
    frontier_points.sort(key=lambda x: (-x[0], x[1]))
    pareto = []
    best_dp = float("inf")
    for acc, dp, t0, t1 in frontier_points:
        if dp < best_dp:
            pareto.append((acc, dp, t0, t1))
            best_dp = dp

    pareto.sort(key=lambda x: x[1])  # Sort by DP gap

    print("  Fairness-Accuracy Pareto Frontier:")
    print(f"  {'DP Gap':>8} {'Accuracy':>10} {'Thresh G0':>10} {'Thresh G1':>10}")
    print(f"  {'-' * 40}")
    for acc, dp, t0, t1 in pareto[:12]:
        print(f"  {dp:>8.4f} {acc:>10.4f} {t0:>10.2f} {t1:>10.2f}")

    # Text-based scatter plot
    print(f"\n  Pareto Frontier (text plot):")
    print(f"  Accuracy")
    plot_height = 15
    plot_width = 50

    acc_vals = [p[0] for p in pareto]
    dp_vals = [p[1] for p in pareto]
    acc_min, acc_max = min(acc_vals) - 0.02, max(acc_vals) + 0.02
    dp_min, dp_max = 0.0, max(dp_vals) + 0.02

    grid = [[" " for _ in range(plot_width)] for _ in range(plot_height)]

    for acc, dp, _, _ in pareto:
        col = int((dp - dp_min) / (dp_max - dp_min) * (plot_width - 1))
        row = int((1.0 - (acc - acc_min) / (acc_max - acc_min)) * (plot_height - 1))
        col = max(0, min(col, plot_width - 1))
        row = max(0, min(row, plot_height - 1))
        grid[row][col] = "*"

    for r in range(plot_height):
        acc_label = acc_max - r * (acc_max - acc_min) / (plot_height - 1)
        print(f"  {acc_label:.3f} |{''.join(grid[r])}|")
    print(f"        +{'-' * plot_width}+")
    print(f"         {dp_min:.2f}" + " " * (plot_width - 10) + f"{dp_max:.2f}")
    print(f"                      DP Gap -->")

    print(f"\n  The Pareto frontier shows that reducing unfairness (DP gap)")
    print(f"  generally comes at the cost of some accuracy.")
    print(f"  Points on the frontier represent optimal trade-offs;")
    print(f"  the practitioner selects based on domain requirements.")


if __name__ == "__main__":
    print("=== Exercise 1: Sample Reweighing ===")
    exercise_1()
    print("\n=== Exercise 2: Training with Fairness Penalty ===")
    exercise_2()
    print("\n=== Exercise 3: Threshold Optimization ===")
    exercise_3()
    print("\n=== Exercise 4: Fairness-Accuracy Pareto Frontier ===")
    exercise_4()
    print("\nAll exercises completed!")
