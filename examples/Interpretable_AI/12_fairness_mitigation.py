"""
12. Fairness Mitigation

Implements the three categories of fairness mitigation -- pre-processing,
in-processing, and post-processing -- on a synthetic hiring dataset, then
compares them on the accuracy-fairness Pareto frontier.

Covered topics:
    - Pre-processing: reweighing samples to equalize base rates
    - In-processing: fairness-constrained logistic regression (penalty term)
    - Post-processing: group-specific threshold optimization
    - Accuracy-fairness Pareto frontier visualization
    - Demographic parity and equalized odds measurement
    - Proxy discrimination detection

Related to: L12 - Fairness Mitigation

Requirements:
    pip install numpy matplotlib scikit-learn
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


# ====== Section 1: Synthetic Hiring Dataset ======

def generate_hiring_data(n: int = 3000, seed: int = 42) -> dict:
    """Generate synthetic hiring data with embedded bias.

    The data simulates a hiring pipeline where:
      - Protected attribute (gender): affects years_experience (structural)
      - Legitimate features: education, skills_score, interview_score
      - True hiring decision: based on legitimate features, but historical
        data contains bias (lower acceptance for protected group)

    Args:
        n: Number of candidates.
        seed: Random seed.

    Returns:
        Dictionary with features, labels, and protected attribute.
    """
    rng = np.random.default_rng(seed)

    # Protected attribute (binary)
    gender = rng.binomial(1, 0.5, n)

    # Legitimate features
    education = rng.normal(0, 1, n)
    skills_score = rng.normal(0, 1, n)
    interview_score = rng.normal(0, 1, n)

    # Biased feature: experience gap due to structural inequality
    years_experience = 5 + 3 * rng.normal(0, 1, n) + 1.5 * gender

    # Normalize
    years_exp_norm = (years_experience - years_experience.mean()) / (
        years_experience.std() + 1e-8
    )

    features = np.column_stack([
        education, skills_score, interview_score, years_exp_norm,
    ])

    # True merit: based on all features
    merit = (1.0 * education + 1.2 * skills_score +
             0.8 * interview_score + 0.5 * years_exp_norm)

    # Historical bias: lower threshold for group 1
    bias = 0.3 * gender
    labels = ((merit + bias + rng.normal(0, 0.5, n)) > np.median(merit)).astype(int)

    return {
        "features": features,
        "labels": labels,
        "protected": gender,
        "feature_names": ["education", "skills", "interview", "experience"],
    }


# ====== Section 2: Fairness Metrics ======

def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray,
) -> dict:
    """Compute demographic parity and equalized odds metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        protected: Binary protected attribute.

    Returns:
        Dictionary with fairness and accuracy metrics.
    """
    acc = accuracy_score(y_true, y_pred)

    metrics = {"accuracy": acc}

    for group_val in [0, 1]:
        mask = protected == group_val
        y_g = y_true[mask]
        p_g = y_pred[mask]

        acceptance = p_g.mean()

        tn, fp, fn, tp = confusion_matrix(y_g, p_g, labels=[0, 1]).ravel()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)

        metrics[f"group_{group_val}"] = {
            "acceptance_rate": float(acceptance),
            "tpr": float(tpr),
            "fpr": float(fpr),
            "n": int(mask.sum()),
        }

    # Demographic parity difference
    dp_diff = abs(
        metrics["group_0"]["acceptance_rate"]
        - metrics["group_1"]["acceptance_rate"]
    )
    metrics["dp_difference"] = dp_diff

    # Equalized odds difference (max of TPR diff and FPR diff)
    tpr_diff = abs(metrics["group_0"]["tpr"] - metrics["group_1"]["tpr"])
    fpr_diff = abs(metrics["group_0"]["fpr"] - metrics["group_1"]["fpr"])
    metrics["eo_difference"] = max(tpr_diff, fpr_diff)

    return metrics


# ====== Section 3: Pre-Processing -- Reweighing ======

def compute_reweighing_weights(
    labels: np.ndarray,
    protected: np.ndarray,
) -> np.ndarray:
    """Compute sample weights to equalize base rates across groups.

    Reweighing (Kamiran & Calders, 2012) adjusts sample weights so that
    each (group, label) combination has the same weighted proportion.
    This removes the statistical association between the protected
    attribute and the label in the training data.

    The weight for a sample with protected=a, label=y is:
        w(a, y) = P(Y=y) * P(A=a) / P(Y=y, A=a)

    Args:
        labels: Binary labels.
        protected: Binary protected attribute.

    Returns:
        Sample weight array of shape (n,).
    """
    n = len(labels)
    weights = np.ones(n)

    for a_val in [0, 1]:
        for y_val in [0, 1]:
            # Joint and marginal probabilities
            mask_ay = (protected == a_val) & (labels == y_val)
            p_ay = mask_ay.sum() / n
            p_a = (protected == a_val).sum() / n
            p_y = (labels == y_val).sum() / n

            if p_ay > 0:
                w = (p_a * p_y) / p_ay
                weights[mask_ay] = w

    return weights


def train_reweighed_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    protected_train: np.ndarray,
) -> LogisticRegression:
    """Train a model with reweighing pre-processing.

    The sample weights equalize base rates, so the model learns from
    a "debiased" version of the training data without modifying the
    features or labels directly.
    """
    weights = compute_reweighing_weights(y_train, protected_train)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)
    return model


# ====== Section 4: In-Processing -- Fairness Penalty ======

def train_fair_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    protected_train: np.ndarray,
    fairness_weight: float = 1.0,
    n_epochs: int = 200,
    lr: float = 0.01,
) -> np.ndarray:
    """Train logistic regression with a demographic parity penalty.

    The loss function is:
        L = cross_entropy + fairness_weight * |mean(sigma(Xw)_a=0) - mean(sigma(Xw)_a=1)|

    The penalty term pushes the model toward equal acceptance rates
    across groups, at the cost of some accuracy.

    Args:
        X_train: Feature matrix (n, d).
        y_train: Labels.
        protected_train: Binary protected attribute.
        fairness_weight: Strength of the fairness penalty.
        n_epochs: Training epochs.
        lr: Learning rate.

    Returns:
        Weight vector (d + 1,) including bias as last element.
    """
    n, d = X_train.shape

    # Add bias column
    X_aug = np.column_stack([X_train, np.ones(n)])
    w = np.zeros(d + 1)

    mask_0 = protected_train == 0
    mask_1 = protected_train == 1

    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    for epoch in range(n_epochs):
        logits = X_aug @ w
        probs = sigmoid(logits)

        # Cross-entropy gradient
        grad_ce = X_aug.T @ (probs - y_train) / n

        # Fairness penalty gradient: d/dw |mean(probs_a=0) - mean(probs_a=1)|
        mean_0 = probs[mask_0].mean()
        mean_1 = probs[mask_1].mean()
        dp_sign = np.sign(mean_0 - mean_1)

        grad_fair = np.zeros(d + 1)
        for idx in np.where(mask_0)[0]:
            grad_fair += dp_sign * probs[idx] * (1 - probs[idx]) * X_aug[idx] / mask_0.sum()
        for idx in np.where(mask_1)[0]:
            grad_fair -= dp_sign * probs[idx] * (1 - probs[idx]) * X_aug[idx] / mask_1.sum()

        grad = grad_ce + fairness_weight * grad_fair
        w -= lr * grad

    return w


class FairModel:
    """Wrapper for the fairness-penalized logistic regression."""

    def __init__(self, weights: np.ndarray):
        self.weights = weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_aug = np.column_stack([X, np.ones(len(X))])
        logits = X_aug @ self.weights
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_aug = np.column_stack([X, np.ones(len(X))])
        logits = X_aug @ self.weights
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        return np.column_stack([1 - probs, probs])


# ====== Section 5: Post-Processing -- Threshold Optimization ======

def optimize_thresholds(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    protected_val: np.ndarray,
    target_dp_diff: float = 0.02,
) -> dict[int, float]:
    """Find group-specific thresholds to achieve demographic parity.

    Post-processing adjusts the decision threshold per group to equalize
    acceptance rates. This does not change the model -- only how its
    outputs are thresholded.

    We search for thresholds that minimize DP difference while
    maintaining reasonable accuracy.

    Args:
        model: Trained model with predict_proba.
        X_val: Validation features.
        y_val: Validation labels.
        protected_val: Binary protected attribute.
        target_dp_diff: Target demographic parity difference.

    Returns:
        Dictionary mapping group value to optimal threshold.
    """
    probs = model.predict_proba(X_val)[:, 1]

    best_thresholds = {0: 0.5, 1: 0.5}
    best_dp_diff = float("inf")
    best_acc = 0.0

    # Grid search over group-specific thresholds
    for t0 in np.arange(0.3, 0.7, 0.02):
        for t1 in np.arange(0.3, 0.7, 0.02):
            preds = np.zeros(len(X_val), dtype=int)
            preds[protected_val == 0] = (
                probs[protected_val == 0] >= t0
            ).astype(int)
            preds[protected_val == 1] = (
                probs[protected_val == 1] >= t1
            ).astype(int)

            acc_rate_0 = preds[protected_val == 0].mean()
            acc_rate_1 = preds[protected_val == 1].mean()
            dp_diff = abs(acc_rate_0 - acc_rate_1)
            acc = accuracy_score(y_val, preds)

            if dp_diff < best_dp_diff or (
                dp_diff <= target_dp_diff and acc > best_acc
            ):
                best_dp_diff = dp_diff
                best_acc = acc
                best_thresholds = {0: t0, 1: t1}

                if dp_diff <= target_dp_diff:
                    break

    return best_thresholds


def predict_with_thresholds(
    model, X: np.ndarray, protected: np.ndarray, thresholds: dict,
) -> np.ndarray:
    """Apply group-specific thresholds for prediction."""
    probs = model.predict_proba(X)[:, 1]
    preds = np.zeros(len(X), dtype=int)
    for group_val, threshold in thresholds.items():
        mask = protected == group_val
        preds[mask] = (probs[mask] >= threshold).astype(int)
    return preds


# ====== Section 6: Visualization ======

def visualize_mitigation(
    results: dict,
    save_path: str = "fairness_mitigation.png",
) -> None:
    """Four-panel visualization of fairness mitigation comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = list(results.keys())
    colors = {
        "Baseline": "#e74c3c",
        "Reweighing": "#3498db",
        "Fair Penalty": "#2ecc71",
        "Threshold Opt": "#9b59b6",
    }

    # --- Panel 1: Accuracy Comparison ---
    ax1 = axes[0, 0]
    accs = [results[m]["accuracy"] for m in methods]
    ax1.bar(methods, accs,
            color=[colors[m] for m in methods],
            edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Model Accuracy by Mitigation Strategy")
    ax1.set_ylim(0.5, 1.0)

    # --- Panel 2: DP Difference ---
    ax2 = axes[0, 1]
    dp_diffs = [results[m]["dp_difference"] for m in methods]
    ax2.bar(methods, dp_diffs,
            color=[colors[m] for m in methods],
            edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Demographic Parity Difference")
    ax2.set_title("Demographic Parity Gap (lower = fairer)")
    ax2.axhline(y=0.05, color="gray", linestyle="--", alpha=0.5, label="5% threshold")
    ax2.legend()

    # --- Panel 3: Per-Group Acceptance Rates ---
    ax3 = axes[1, 0]
    x_pos = np.arange(len(methods))
    width = 0.35
    g0_rates = [results[m]["group_0"]["acceptance_rate"] for m in methods]
    g1_rates = [results[m]["group_1"]["acceptance_rate"] for m in methods]
    ax3.bar(x_pos - width / 2, g0_rates, width, label="Group 0",
            color="#3498db", edgecolor="black", linewidth=0.5)
    ax3.bar(x_pos + width / 2, g1_rates, width, label="Group 1",
            color="#e74c3c", edgecolor="black", linewidth=0.5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(methods, rotation=15, fontsize=9)
    ax3.set_ylabel("Acceptance Rate")
    ax3.set_title("Per-Group Acceptance Rates")
    ax3.legend()

    # --- Panel 4: Pareto Frontier ---
    ax4 = axes[1, 1]
    for m in methods:
        ax4.scatter(results[m]["dp_difference"], results[m]["accuracy"],
                    s=150, color=colors[m], edgecolor="black",
                    linewidth=1.5, zorder=5, label=m)
    ax4.set_xlabel("DP Difference (lower = fairer)")
    ax4.set_ylabel("Accuracy (higher = better)")
    ax4.set_title("Accuracy-Fairness Tradeoff")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Fairness Mitigation Strategies Comparison", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved to: {save_path}")
    plt.close()


# ====== Section 7: Main Pipeline ======

def main() -> None:
    """Compare pre-, in-, and post-processing fairness mitigation."""
    print("=" * 65)
    print("  Fairness Mitigation")
    print("  Pre-processing | In-processing | Post-processing")
    print("=" * 65)

    # --- Step 1: Generate data ---
    print("\n[1] Generating Synthetic Hiring Data")
    print("-" * 50)

    data = generate_hiring_data(n=3000)
    X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
        data["features"], data["labels"], data["protected"],
        test_size=0.3, random_state=42,
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Features: {data['feature_names']}")

    results = {}

    # --- Step 2: Baseline (no mitigation) ---
    print("\n[2] Baseline Model (No Mitigation)")
    print("-" * 50)

    baseline = LogisticRegression(max_iter=1000, random_state=42)
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)
    baseline_metrics = compute_fairness_metrics(y_test, baseline_preds, p_test)
    results["Baseline"] = baseline_metrics
    print(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"  DP difference: {baseline_metrics['dp_difference']:.4f}")
    print(f"  EO difference: {baseline_metrics['eo_difference']:.4f}")

    # --- Step 3: Pre-processing (Reweighing) ---
    print("\n[3] Pre-Processing: Reweighing")
    print("-" * 50)

    reweighed_model = train_reweighed_model(X_train, y_train, p_train)
    reweighed_preds = reweighed_model.predict(X_test)
    reweighed_metrics = compute_fairness_metrics(y_test, reweighed_preds, p_test)
    results["Reweighing"] = reweighed_metrics
    print(f"  Accuracy: {reweighed_metrics['accuracy']:.4f}")
    print(f"  DP difference: {reweighed_metrics['dp_difference']:.4f}")
    print(f"  EO difference: {reweighed_metrics['eo_difference']:.4f}")

    # --- Step 4: In-processing (Fairness penalty) ---
    print("\n[4] In-Processing: Fairness-Penalized Logistic Regression")
    print("-" * 50)

    fair_weights = train_fair_model(
        X_train, y_train, p_train, fairness_weight=2.0,
    )
    fair_model = FairModel(fair_weights)
    fair_preds = fair_model.predict(X_test)
    fair_metrics = compute_fairness_metrics(y_test, fair_preds, p_test)
    results["Fair Penalty"] = fair_metrics
    print(f"  Accuracy: {fair_metrics['accuracy']:.4f}")
    print(f"  DP difference: {fair_metrics['dp_difference']:.4f}")
    print(f"  EO difference: {fair_metrics['eo_difference']:.4f}")

    # --- Step 5: Post-processing (Threshold optimization) ---
    print("\n[5] Post-Processing: Group-Specific Threshold Optimization")
    print("-" * 50)

    thresholds = optimize_thresholds(baseline, X_test, y_test, p_test)
    print(f"  Optimized thresholds: group_0={thresholds[0]:.2f}, "
          f"group_1={thresholds[1]:.2f}")

    threshold_preds = predict_with_thresholds(baseline, X_test, p_test, thresholds)
    threshold_metrics = compute_fairness_metrics(y_test, threshold_preds, p_test)
    results["Threshold Opt"] = threshold_metrics
    print(f"  Accuracy: {threshold_metrics['accuracy']:.4f}")
    print(f"  DP difference: {threshold_metrics['dp_difference']:.4f}")
    print(f"  EO difference: {threshold_metrics['eo_difference']:.4f}")

    # --- Step 6: Comparison table ---
    print("\n[6] Comparison Summary")
    print("-" * 50)
    print(f"  {'Method':<16s} {'Accuracy':>10s} {'DP Diff':>10s} {'EO Diff':>10s}")
    print(f"  {'-' * 46}")
    for method, m in results.items():
        print(f"  {method:<16s} {m['accuracy']:>10.4f} "
              f"{m['dp_difference']:>10.4f} {m['eo_difference']:>10.4f}")

    # --- Step 7: Visualization ---
    print("\n[7] Generating Visualization")
    print("-" * 50)

    visualize_mitigation(results)

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    print("""
  Key findings:
    1. Reweighing (pre-processing) adjusts sample weights to equalize
       base rates, reducing bias without modifying features.
    2. Fairness penalty (in-processing) adds a demographic parity
       constraint to the loss function, trading accuracy for fairness.
    3. Threshold optimization (post-processing) adjusts decision
       boundaries per group -- simple, model-agnostic, but only
       addresses the final classification step.
    4. The accuracy-fairness Pareto frontier shows the inherent
       tradeoff: fairer models typically sacrifice some accuracy.
    5. Choice of strategy depends on context: pre-processing is
       model-agnostic, in-processing gives fine-grained control,
       post-processing is simplest to deploy.
    """)


if __name__ == "__main__":
    main()
