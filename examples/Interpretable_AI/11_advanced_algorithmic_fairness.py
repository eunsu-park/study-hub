"""
11. Advanced Algorithmic Fairness

Explores fairness beyond simple group metrics by implementing individual
fairness, counterfactual fairness, the impossibility theorem, and
intersectional analysis on a synthetic credit-scoring dataset.

Covered topics:
    - Individual fairness via Lipschitz constraint checking
    - Counterfactual fairness using a structural causal model
    - Impossibility theorem: calibration vs. error-rate parity
    - Intersectional fairness across multiple protected attributes
    - Fairness metric dashboard with visualizations

Related to: L11 - Advanced Algorithmic Fairness

Requirements:
    pip install numpy matplotlib scikit-learn
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# ====== Section 1: Synthetic Credit Dataset ======

@dataclass
class CreditData:
    """Container for synthetic credit-scoring data."""
    features: np.ndarray       # (n, d) legitimate features
    protected_a: np.ndarray    # binary protected attribute (e.g., race)
    protected_b: np.ndarray    # binary protected attribute (e.g., gender)
    labels: np.ndarray         # ground-truth creditworthiness
    feature_names: list[str]


def generate_credit_data(n: int = 3000, seed: int = 42) -> CreditData:
    """Generate synthetic credit-scoring data with two protected attributes.

    The data generation process embeds realistic structural biases:
      - Protected attribute A affects income (historical inequality)
      - Protected attribute B affects credit history length
      - The true label depends on income and credit history, NOT
        on the protected attributes directly. However, a model
        trained on this data may learn proxies for A and B.

    Args:
        n: Number of samples.
        seed: Random seed.

    Returns:
        CreditData with features, protected attributes, and labels.
    """
    rng = np.random.default_rng(seed)

    # Protected attributes (binary)
    A = rng.binomial(1, 0.5, n)  # e.g., race
    B = rng.binomial(1, 0.5, n)  # e.g., gender

    # Legitimate features (affected by protected attributes)
    income = 50000 + 20000 * rng.normal(0, 1, n) + 10000 * A
    credit_history = 5 + 3 * rng.normal(0, 1, n) + 2 * B
    debt_ratio = 0.3 + 0.15 * rng.normal(0, 1, n)
    savings = 10000 + 8000 * rng.normal(0, 1, n)

    # Normalize features
    features = np.column_stack([income, credit_history, debt_ratio, savings])
    for j in range(features.shape[1]):
        features[:, j] = (features[:, j] - features[:, j].mean()) / (
            features[:, j].std() + 1e-8
        )

    # True label: depends on legitimate features, not protected attributes
    score = 1.5 * features[:, 0] + 1.0 * features[:, 1] - 0.8 * features[:, 2] + 0.5 * features[:, 3]
    score += rng.normal(0, 0.5, n)
    labels = (score > np.median(score)).astype(int)

    return CreditData(
        features=features,
        protected_a=A,
        protected_b=B,
        labels=labels,
        feature_names=["income", "credit_history", "debt_ratio", "savings"],
    )


# ====== Section 2: Individual Fairness ======

def check_individual_fairness(
    model: LogisticRegression,
    X: np.ndarray,
    n_pairs: int = 5000,
    lipschitz_threshold: float = 2.0,
    seed: int = 42,
) -> dict:
    """Check individual fairness via Lipschitz constraint.

    Individual fairness (Dwork et al., 2012) requires that similar
    individuals receive similar predictions:
        d_Y(f(x1), f(x2)) <= L * d_X(x1, x2)

    where L is the Lipschitz constant. We estimate L by sampling pairs
    and computing the maximum ratio of output distance to input distance.

    A high Lipschitz constant means the model's predictions change
    sharply between similar inputs -- a fairness concern.

    Args:
        model: Trained classifier.
        X: Feature matrix (n, d).
        n_pairs: Number of random pairs to sample.
        lipschitz_threshold: Maximum acceptable Lipschitz constant.
        seed: Random seed.

    Returns:
        Dictionary with Lipschitz statistics and violation count.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    probs = model.predict_proba(X)[:, 1]

    ratios = []
    violations = 0

    for _ in range(n_pairs):
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue

        input_dist = np.linalg.norm(X[i] - X[j])
        if input_dist < 1e-8:
            continue

        output_dist = abs(probs[i] - probs[j])
        ratio = output_dist / input_dist
        ratios.append(ratio)

        if ratio > lipschitz_threshold:
            violations += 1

    ratios = np.array(ratios)
    return {
        "estimated_lipschitz": float(np.max(ratios)) if len(ratios) > 0 else 0.0,
        "mean_ratio": float(np.mean(ratios)) if len(ratios) > 0 else 0.0,
        "median_ratio": float(np.median(ratios)) if len(ratios) > 0 else 0.0,
        "violations": violations,
        "violation_rate": violations / max(len(ratios), 1),
        "threshold": lipschitz_threshold,
        "pairs_checked": len(ratios),
    }


# ====== Section 3: Counterfactual Fairness ======

def check_counterfactual_fairness(
    model: LogisticRegression,
    data: CreditData,
    n_samples: int = 500,
    seed: int = 42,
) -> dict:
    """Check counterfactual fairness by flipping protected attributes.

    Counterfactual fairness (Kusner et al., 2017): a prediction is
    counterfactually fair if it would remain the same had the individual
    belonged to a different protected group, with everything else
    adjusted according to the causal model.

    We approximate this by:
    1. For each sample, flip protected attribute A.
    2. Adjust causally-downstream features (income) accordingly.
    3. Check if the prediction changes.

    A counterfactually fair model should produce the same prediction
    regardless of the protected attribute, after causal adjustment.

    Args:
        model: Trained classifier.
        data: CreditData with protected attributes.
        n_samples: Number of samples to check.
        seed: Random seed.

    Returns:
        Dictionary with counterfactual fairness statistics.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(data.features), n_samples, replace=False)

    prediction_changes = 0
    prob_diffs = []

    for idx in indices:
        # Original prediction
        x_original = data.features[idx:idx + 1]
        prob_original = model.predict_proba(x_original)[0, 1]

        # Counterfactual: flip protected attribute A
        # Adjust income (feature 0) to remove the effect of A
        x_cf = x_original.copy()
        a_original = data.protected_a[idx]
        a_counterfactual = 1 - a_original

        # Causal adjustment: income shifts by the effect of A
        # In our SCM, A adds ~10000 to income (before normalization)
        # The normalized effect is approximately 10000/20000 = 0.5
        causal_shift = 0.5 * (a_counterfactual - a_original)
        x_cf[0, 0] += causal_shift

        prob_cf = model.predict_proba(x_cf)[0, 1]
        diff = abs(prob_original - prob_cf)
        prob_diffs.append(diff)

        if (prob_original >= 0.5) != (prob_cf >= 0.5):
            prediction_changes += 1

    prob_diffs = np.array(prob_diffs)
    return {
        "prediction_flip_rate": prediction_changes / n_samples,
        "mean_prob_diff": float(prob_diffs.mean()),
        "max_prob_diff": float(prob_diffs.max()),
        "std_prob_diff": float(prob_diffs.std()),
        "samples_checked": n_samples,
    }


# ====== Section 4: Impossibility Theorem ======

def demonstrate_impossibility(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    protected: np.ndarray,
) -> dict:
    """Demonstrate the impossibility of simultaneous fairness metrics.

    Choraś et al. (2020) / Kleinberg et al. (2016): it is mathematically
    impossible to simultaneously achieve:
      1. Calibration: P(Y=1 | S=s, A=a) = s for all groups
      2. FPR parity: equal false positive rates across groups
      3. FNR parity: equal false negative rates across groups

    unless the base rates P(Y=1 | A=a) are equal across groups or the
    model is perfect.

    We compute all three metrics and show the tension.

    Args:
        model: Trained classifier.
        X: Feature matrix.
        y: Ground-truth labels.
        protected: Binary protected attribute.

    Returns:
        Dictionary with per-group metrics showing the impossibility.
    """
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    results = {}

    for group_val in [0, 1]:
        mask = protected == group_val
        y_g = y[mask]
        pred_g = preds[mask]
        prob_g = probs[mask]

        tn, fp, fn, tp = confusion_matrix(y_g, pred_g, labels=[0, 1]).ravel()

        # Base rate
        base_rate = y_g.mean()

        # FPR and FNR
        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(fn + tp, 1)

        # Calibration: among those predicted positive, fraction truly positive
        pred_pos_mask = pred_g == 1
        if pred_pos_mask.sum() > 0:
            ppv = y_g[pred_pos_mask].mean()
        else:
            ppv = 0.0

        results[f"group_{group_val}"] = {
            "base_rate": float(base_rate),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "ppv": float(ppv),
            "n": int(mask.sum()),
        }

    return results


# ====== Section 5: Intersectional Fairness ======

def intersectional_analysis(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
) -> dict:
    """Analyze fairness across intersections of two protected attributes.

    Intersectional fairness (Crenshaw, 1989; Buolamwini & Gebru, 2018)
    recognizes that discrimination can affect subgroups defined by the
    intersection of multiple attributes in ways not captured by analyzing
    each attribute independently.

    For each of the 4 subgroups (A=0,B=0), (A=0,B=1), (A=1,B=0), (A=1,B=1),
    we compute acceptance rate, accuracy, and calibration.

    Args:
        model: Trained classifier.
        X: Feature matrix.
        y: Ground-truth labels.
        A: First binary protected attribute.
        B: Second binary protected attribute.

    Returns:
        Dictionary with per-subgroup metrics.
    """
    preds = model.predict(X)
    results = {}

    for a_val in [0, 1]:
        for b_val in [0, 1]:
            mask = (A == a_val) & (B == b_val)
            group_name = f"A={a_val},B={b_val}"

            y_g = y[mask]
            pred_g = preds[mask]

            acceptance_rate = pred_g.mean()
            accuracy = (pred_g == y_g).mean()
            base_rate = y_g.mean()

            # TPR (recall)
            pos_mask = y_g == 1
            tpr = pred_g[pos_mask].mean() if pos_mask.sum() > 0 else 0.0

            results[group_name] = {
                "n": int(mask.sum()),
                "base_rate": float(base_rate),
                "acceptance_rate": float(acceptance_rate),
                "accuracy": float(accuracy),
                "tpr": float(tpr),
            }

    return results


# ====== Section 6: Visualization ======

def visualize_fairness(
    individual: dict,
    counterfactual: dict,
    impossibility: dict,
    intersectional: dict,
    save_path: str = "advanced_fairness.png",
) -> None:
    """Four-panel visualization of advanced fairness analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Individual Fairness Lipschitz Distribution ---
    ax1 = axes[0, 0]
    stats = [
        ("Mean", individual["mean_ratio"]),
        ("Median", individual["median_ratio"]),
        ("Max (Lipschitz)", individual["estimated_lipschitz"]),
        ("Threshold", individual["threshold"]),
    ]
    names, vals = zip(*stats)
    colors = ["#3498db", "#3498db", "#e74c3c", "#95a5a6"]
    ax1.barh(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("Value")
    ax1.set_title(f"Individual Fairness (Lipschitz)\n"
                   f"Violations: {individual['violations']}/{individual['pairs_checked']}")

    # --- Panel 2: Counterfactual Fairness ---
    ax2 = axes[0, 1]
    cf_stats = [
        ("Flip Rate", counterfactual["prediction_flip_rate"]),
        ("Mean P(diff)", counterfactual["mean_prob_diff"]),
        ("Max P(diff)", counterfactual["max_prob_diff"]),
    ]
    names, vals = zip(*cf_stats)
    ax2.bar(names, vals, color=["#e74c3c", "#f39c12", "#e74c3c"],
            edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Value")
    ax2.set_title("Counterfactual Fairness\n(flip protected attr A)")

    # --- Panel 3: Impossibility Theorem ---
    ax3 = axes[1, 0]
    groups = list(impossibility.keys())
    x_pos = np.arange(3)
    width = 0.35
    metrics = ["fpr", "fnr", "ppv"]
    metric_labels = ["FPR", "FNR", "PPV"]
    for i, group in enumerate(groups):
        vals = [impossibility[group][m] for m in metrics]
        offset = (i - 0.5) * width
        ax3.bar(x_pos + offset, vals, width, label=group,
                edgecolor="black", linewidth=0.5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(metric_labels)
    ax3.set_ylabel("Rate")
    ax3.set_title("Impossibility Theorem\n(FPR, FNR, PPV by group)")
    ax3.legend()

    # --- Panel 4: Intersectional Analysis ---
    ax4 = axes[1, 1]
    subgroups = list(intersectional.keys())
    acc_rates = [intersectional[g]["acceptance_rate"] for g in subgroups]
    tpr_rates = [intersectional[g]["tpr"] for g in subgroups]
    x_pos = np.arange(len(subgroups))
    width = 0.35
    ax4.bar(x_pos - width / 2, acc_rates, width, label="Acceptance Rate",
            color="#3498db", edgecolor="black", linewidth=0.5)
    ax4.bar(x_pos + width / 2, tpr_rates, width, label="TPR",
            color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(subgroups, rotation=15, fontsize=9)
    ax4.set_ylabel("Rate")
    ax4.set_title("Intersectional Fairness (4 subgroups)")
    ax4.legend()

    plt.suptitle("Advanced Algorithmic Fairness Analysis", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved to: {save_path}")
    plt.close()


# ====== Section 7: Main Pipeline ======

def main() -> None:
    """Run advanced algorithmic fairness experiments."""
    print("=" * 65)
    print("  Advanced Algorithmic Fairness")
    print("  Individual | Counterfactual | Impossibility | Intersectional")
    print("=" * 65)

    # --- Step 1: Generate data ---
    print("\n[1] Generating Synthetic Credit-Scoring Data")
    print("-" * 50)

    data = generate_credit_data(n=3000)
    print(f"  Samples: {len(data.labels)}")
    print(f"  Features: {data.feature_names}")
    print(f"  Protected A distribution: {data.protected_a.mean():.2f}")
    print(f"  Protected B distribution: {data.protected_b.mean():.2f}")
    print(f"  Label balance: {data.labels.mean():.2f}")

    # --- Step 2: Train model ---
    print("\n[2] Training Logistic Regression")
    print("-" * 50)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(data.features, data.labels)
    acc = (model.predict(data.features) == data.labels).mean()
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Coefficients: {dict(zip(data.feature_names, model.coef_[0].round(4)))}")

    # --- Step 3: Individual fairness ---
    print("\n[3] Individual Fairness (Lipschitz Constraint)")
    print("-" * 50)

    individual = check_individual_fairness(model, data.features)
    print(f"  Estimated Lipschitz constant: {individual['estimated_lipschitz']:.4f}")
    print(f"  Mean ratio: {individual['mean_ratio']:.4f}")
    print(f"  Violations (L > {individual['threshold']}): "
          f"{individual['violations']}/{individual['pairs_checked']} "
          f"({individual['violation_rate']:.2%})")

    # --- Step 4: Counterfactual fairness ---
    print("\n[4] Counterfactual Fairness")
    print("-" * 50)

    counterfactual = check_counterfactual_fairness(model, data)
    print(f"  Prediction flip rate: {counterfactual['prediction_flip_rate']:.4f}")
    print(f"  Mean probability difference: {counterfactual['mean_prob_diff']:.4f}")
    print(f"  Max probability difference: {counterfactual['max_prob_diff']:.4f}")

    # --- Step 5: Impossibility theorem ---
    print("\n[5] Impossibility Theorem Demonstration")
    print("-" * 50)

    impossibility = demonstrate_impossibility(
        model, data.features, data.labels, data.protected_a,
    )
    print(f"  {'Group':<10s} {'Base Rate':>10s} {'FPR':>8s} {'FNR':>8s} {'PPV':>8s}")
    print(f"  {'-' * 38}")
    for group, metrics in impossibility.items():
        print(f"  {group:<10s} {metrics['base_rate']:>10.4f} "
              f"{metrics['fpr']:>8.4f} {metrics['fnr']:>8.4f} "
              f"{metrics['ppv']:>8.4f}")
    print()
    print("  Note: if base rates differ, it is mathematically impossible")
    print("  to equalize FPR, FNR, and PPV simultaneously.")

    # --- Step 6: Intersectional analysis ---
    print("\n[6] Intersectional Fairness Analysis")
    print("-" * 50)

    intersectional = intersectional_analysis(
        model, data.features, data.labels,
        data.protected_a, data.protected_b,
    )
    print(f"  {'Subgroup':<12s} {'N':>6s} {'Base':>6s} {'Accept':>8s} "
          f"{'Acc':>6s} {'TPR':>6s}")
    print(f"  {'-' * 46}")
    for group, m in intersectional.items():
        print(f"  {group:<12s} {m['n']:>6d} {m['base_rate']:>6.3f} "
              f"{m['acceptance_rate']:>8.3f} {m['accuracy']:>6.3f} "
              f"{m['tpr']:>6.3f}")

    # --- Step 7: Visualization ---
    print("\n[7] Generating Visualization")
    print("-" * 50)

    visualize_fairness(individual, counterfactual, impossibility, intersectional)

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    print("""
  Key findings:
    1. Individual fairness: the Lipschitz constant quantifies how
       sharply predictions change between similar individuals.
       Violations indicate potential discrimination.
    2. Counterfactual fairness: flipping protected attribute A
       (with causal adjustment) shows whether the model's predictions
       depend on group membership.
    3. Impossibility theorem: when base rates differ between groups,
       we cannot simultaneously achieve calibration, FPR parity, and
       FNR parity. This is a mathematical limitation, not a modeling
       failure.
    4. Intersectional analysis: subgroups defined by multiple attributes
       (A x B) can experience disparities invisible when analyzing
       each attribute separately.
    """)


if __name__ == "__main__":
    main()
