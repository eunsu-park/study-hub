"""
07. Fairness Toolkit for Machine Learning

End-to-end fairness audit and mitigation pipeline for a binary classifier.
Generates a synthetic credit scoring dataset with embedded demographic bias,
computes group and individual fairness metrics, applies two mitigation
strategies (pre-processing reweighing and post-processing threshold
optimization), and visualizes the fairness-accuracy Pareto frontier.

Covered topics:
    - Synthetic biased credit scoring dataset generation
    - Group fairness metrics: demographic parity, equalized odds, calibration
    - Individual fairness via k-neighbor consistency
    - Pre-processing mitigation: sample reweighing
    - Post-processing mitigation: group-specific threshold optimization
    - Fairness-accuracy Pareto frontier visualization

Related to: L11-L12 - Fairness in Machine Learning

Requirements:
    pip install numpy scikit-learn matplotlib pandas
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


# ====== Section 1: Synthetic Biased Credit Scoring Dataset ======

def generate_biased_credit_data(
    n_samples: int = 3000,
    bias_strength: float = 0.8,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic credit scoring dataset with embedded demographic bias.

    The bias is injected through two mechanisms:
      1. **Label bias**: The protected group has a lower base approval rate,
         even when creditworthiness features are identical.
      2. **Feature bias**: Income is systematically lower for the protected
         group, reflecting historical inequality.

    This mirrors real-world scenarios where both the data *and* the labels
    carry societal biases that a naive model would reproduce.

    Args:
        n_samples: Total dataset size.
        bias_strength: Controls how strong the demographic bias is.
                       0.0 = no bias, 1.0 = maximum bias.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with features, protected attribute, and outcome.
    """
    np.random.seed(seed)

    # Protected attribute: 0 = majority group, 1 = minority group
    # Unequal representation (60/40) reflects real imbalances
    protected = np.random.binomial(1, 0.4, n_samples)

    # Creditworthiness features (should be the only things that matter)
    credit_score = np.random.normal(650, 80, n_samples).clip(300, 850)
    annual_income = np.random.exponential(55, n_samples).clip(15, 300)
    debt_to_income = np.random.beta(2, 5, n_samples)
    employment_years = np.random.exponential(6, n_samples).clip(0, 40)
    num_accounts = np.random.poisson(3, n_samples).clip(0, 15)

    # Inject feature bias: minority group has systematically lower income
    # This creates an indirect pathway for discrimination even if the
    # model does not see the protected attribute directly
    income_penalty = bias_strength * 12 * protected
    annual_income = (annual_income - income_penalty).clip(15, 300)

    # Ground truth creditworthiness (bias-free)
    logit_fair = (
        0.015 * (credit_score - 600)
        + 0.025 * annual_income
        - 2.5 * debt_to_income
        + 0.08 * employment_years
        + 0.1 * num_accounts
        - 2.0
    )

    # Inject label bias: protected group faces a harsher threshold
    # This simulates historically biased lending decisions
    label_bias = bias_strength * 1.5 * protected
    logit_biased = logit_fair - label_bias

    prob = 1.0 / (1.0 + np.exp(-logit_biased))
    approved = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "credit_score": credit_score,
        "annual_income": annual_income,
        "debt_to_income": debt_to_income,
        "employment_years": employment_years,
        "num_accounts": num_accounts,
        "protected": protected,
        "approved": approved,
    })

    return df


FEATURE_COLS = [
    "credit_score", "annual_income", "debt_to_income",
    "employment_years", "num_accounts",
]


# ====== Section 2: Group Fairness Metrics ======

def demographic_parity_difference(y_pred: np.ndarray, protected: np.ndarray) -> float:
    """Compute the demographic parity difference (DPD).

    DPD = P(Y_hat=1 | A=0) - P(Y_hat=1 | A=1)

    A value of 0 means both groups have equal positive prediction rates.
    Positive values indicate the majority group is favoured.  The "80%
    rule" (or 4/5ths rule) from US employment law corresponds to a
    ratio threshold, but DPD is more commonly used in ML fairness.
    """
    rate_majority = y_pred[protected == 0].mean()
    rate_minority = y_pred[protected == 1].mean()
    return rate_majority - rate_minority


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray,
) -> dict:
    """Compute equalized odds differences for both outcomes.

    Equalized odds requires equal TPR and FPR across groups:
      - TPR gap = P(Y_hat=1|Y=1, A=0) - P(Y_hat=1|Y=1, A=1)
      - FPR gap = P(Y_hat=1|Y=0, A=0) - P(Y_hat=1|Y=0, A=1)

    A model satisfying equalized odds is equally accurate for both
    groups in terms of both false positives and false negatives.
    """
    results = {}
    for label, name in [(1, "tpr_gap"), (0, "fpr_gap")]:
        mask = y_true == label
        rate_maj = y_pred[(protected == 0) & mask].mean() if ((protected == 0) & mask).sum() > 0 else 0
        rate_min = y_pred[(protected == 1) & mask].mean() if ((protected == 1) & mask).sum() > 0 else 0
        results[name] = rate_maj - rate_min

    # Maximum absolute gap (used as a single summary statistic)
    results["max_gap"] = max(abs(results["tpr_gap"]), abs(results["fpr_gap"]))
    return results


def calibration_difference(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    protected: np.ndarray,
    n_bins: int = 5,
) -> float:
    """Compute calibration difference between groups.

    For each probability bin, we check whether P(Y=1 | score in bin)
    is similar across groups.  Good calibration means the model's
    confidence is equally reliable for both groups.

    Returns the mean absolute calibration gap across bins.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    gaps = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        for group in [0, 1]:
            mask = (protected == group) & (y_prob >= lo) & (y_prob < hi)
            if mask.sum() >= 5:
                # Enough samples for a meaningful estimate
                pass
            else:
                continue

        # Compare actual positive rate in this bin across groups
        mask_maj = (protected == 0) & (y_prob >= lo) & (y_prob < hi)
        mask_min = (protected == 1) & (y_prob >= lo) & (y_prob < hi)

        if mask_maj.sum() >= 5 and mask_min.sum() >= 5:
            rate_maj = y_true[mask_maj].mean()
            rate_min = y_true[mask_min].mean()
            gaps.append(abs(rate_maj - rate_min))

    return np.mean(gaps) if gaps else 0.0


# ====== Section 3: Individual Fairness ======

def individual_fairness_consistency(
    X: np.ndarray,
    y_pred: np.ndarray,
    k: int = 5,
) -> float:
    """Compute individual fairness via k-neighbor consistency.

    Individual fairness (Dwork et al. 2012) requires that similar
    individuals receive similar predictions.  We operationalise this
    as: for each individual, what fraction of their k nearest neighbors
    received the same prediction?

    A consistency of 1.0 means perfectly consistent predictions among
    neighbors; lower values indicate that similar people are treated
    differently.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    consistencies = []
    for i in range(len(X)):
        # Exclude the point itself (first neighbor)
        neighbor_preds = y_pred[indices[i, 1:]]
        # Fraction of neighbors with the same prediction
        consistency = (neighbor_preds == y_pred[i]).mean()
        consistencies.append(consistency)

    return np.mean(consistencies)


# ====== Section 4: Pre-Processing Mitigation -- Reweighing ======

def compute_reweighing_weights(
    y_true: np.ndarray,
    protected: np.ndarray,
) -> np.ndarray:
    """Compute sample weights that equalize outcome rates across groups.

    Reweighing (Kamiran & Calders, 2012) assigns higher weights to
    under-represented (group, label) combinations and lower weights to
    over-represented ones.  The idea is to undo the historical bias
    in the training labels by making the weighted distribution fair.

    Weight formula for sample with (A=a, Y=y):
        w(a, y) = P(Y=y) * P(A=a) / P(Y=y, A=a)
    """
    n = len(y_true)
    weights = np.ones(n)

    for a in [0, 1]:
        for y in [0, 1]:
            mask = (protected == a) & (y_true == y)
            n_ay = mask.sum()
            if n_ay == 0:
                continue

            # Marginal probabilities
            p_y = (y_true == y).sum() / n
            p_a = (protected == a).sum() / n
            # Joint probability
            p_ay = n_ay / n

            # Weight = expected / observed
            w = (p_y * p_a) / p_ay
            weights[mask] = w

    return weights


# ====== Section 5: Post-Processing Mitigation -- Threshold Optimization ======

def optimize_group_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    protected: np.ndarray,
    target_metric: str = "demographic_parity",
    n_thresholds: int = 50,
) -> dict:
    """Find group-specific thresholds that minimize unfairness.

    Instead of using the same 0.5 threshold for everyone, we search
    for per-group thresholds that best balance accuracy and fairness.
    This is a post-processing technique -- we do not retrain the model,
    only adjust the decision boundary for each group.

    Args:
        target_metric: Which fairness metric to optimize.
                       "demographic_parity" or "equalized_odds".
        n_thresholds: Resolution of the threshold search grid.

    Returns:
        Dict with optimal thresholds and resulting metrics.
    """
    thresholds = np.linspace(0.1, 0.9, n_thresholds)

    best_result = None
    best_unfairness = float("inf")

    for t_maj in thresholds:
        for t_min in thresholds:
            # Apply group-specific thresholds
            y_pred = np.zeros_like(y_true)
            y_pred[(protected == 0) & (y_prob >= t_maj)] = 1
            y_pred[(protected == 1) & (y_prob >= t_min)] = 1

            acc = accuracy_score(y_true, y_pred)

            if target_metric == "demographic_parity":
                unfairness = abs(demographic_parity_difference(y_pred, protected))
            elif target_metric == "equalized_odds":
                eo = equalized_odds_difference(y_true, y_pred, protected)
                unfairness = eo["max_gap"]
            else:
                raise ValueError(f"Unknown metric: {target_metric}")

            # We want to minimize unfairness while keeping accuracy reasonable
            # Use a combined score: low unfairness + high accuracy
            score = unfairness - 0.5 * acc  # lower is better

            if score < best_unfairness:
                best_unfairness = score
                best_result = {
                    "threshold_majority": t_maj,
                    "threshold_minority": t_min,
                    "accuracy": acc,
                    "unfairness": unfairness,
                    "y_pred": y_pred.copy(),
                }

    return best_result


# ====== Section 6: Fairness-Accuracy Pareto Frontier ======

def compute_pareto_frontier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    protected: np.ndarray,
    n_thresholds: int = 30,
) -> tuple[list[float], list[float], list[tuple]]:
    """Sweep group-specific thresholds and collect (accuracy, fairness) pairs.

    Returns all non-dominated points forming the Pareto frontier --
    configurations where you cannot improve fairness without sacrificing
    accuracy, and vice versa.
    """
    thresholds = np.linspace(0.2, 0.8, n_thresholds)
    points = []

    for t_maj in thresholds:
        for t_min in thresholds:
            y_pred = np.zeros_like(y_true)
            y_pred[(protected == 0) & (y_prob >= t_maj)] = 1
            y_pred[(protected == 1) & (y_prob >= t_min)] = 1

            acc = accuracy_score(y_true, y_pred)
            dpd = abs(demographic_parity_difference(y_pred, protected))

            points.append((acc, dpd, (t_maj, t_min)))

    # Extract Pareto-optimal points (maximize accuracy, minimize unfairness)
    # A point is Pareto-optimal if no other point is both more accurate and
    # more fair simultaneously
    points.sort(key=lambda p: (-p[0], p[1]))

    pareto_acc = []
    pareto_dpd = []
    pareto_thresholds = []
    min_dpd_so_far = float("inf")

    for acc, dpd, thresh in points:
        if dpd <= min_dpd_so_far:
            pareto_acc.append(acc)
            pareto_dpd.append(dpd)
            pareto_thresholds.append(thresh)
            min_dpd_so_far = dpd

    all_acc = [p[0] for p in points]
    all_dpd = [p[1] for p in points]

    return all_acc, all_dpd, pareto_acc, pareto_dpd, pareto_thresholds


def visualize_fairness_audit(
    metrics_baseline: dict,
    metrics_reweighed: dict,
    metrics_threshold: dict,
    all_acc: list,
    all_dpd: list,
    pareto_acc: list,
    pareto_dpd: list,
    save_path: str = "fairness_audit.png",
) -> None:
    """Create a two-panel figure: metric comparison + Pareto frontier."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- Panel 1: Bar chart comparing metrics across methods ---
    ax1 = axes[0]
    metric_names = ["DP Diff", "EO Max Gap", "Calibration Gap", "Accuracy"]
    baseline_vals = [
        metrics_baseline["dp_diff"],
        metrics_baseline["eo_max_gap"],
        metrics_baseline["calib_diff"],
        metrics_baseline["accuracy"],
    ]
    reweighed_vals = [
        metrics_reweighed["dp_diff"],
        metrics_reweighed["eo_max_gap"],
        metrics_reweighed["calib_diff"],
        metrics_reweighed["accuracy"],
    ]
    threshold_vals = [
        metrics_threshold["dp_diff"],
        metrics_threshold["eo_max_gap"],
        metrics_threshold["calib_diff"],
        metrics_threshold["accuracy"],
    ]

    x = np.arange(len(metric_names))
    w = 0.25

    ax1.bar(x - w, baseline_vals, w, label="Baseline", color="#e74c3c", alpha=0.85)
    ax1.bar(x, reweighed_vals, w, label="Reweighed", color="#3498db", alpha=0.85)
    ax1.bar(x + w, threshold_vals, w, label="Threshold Opt", color="#2ecc71", alpha=0.85)

    ax1.set_xlabel("Metric", fontsize=11)
    ax1.set_ylabel("Value", fontsize=11)
    ax1.set_title("Fairness Metrics Comparison", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, fontsize=10)
    ax1.legend()
    ax1.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    # --- Panel 2: Pareto frontier ---
    ax2 = axes[1]
    ax2.scatter(all_acc, all_dpd, s=8, alpha=0.15, color="gray",
                label="All threshold combos")
    ax2.plot(pareto_acc, pareto_dpd, "ro-", markersize=5, linewidth=1.5,
             label="Pareto frontier", zorder=5)

    # Mark the three methods
    ax2.scatter([metrics_baseline["accuracy"]], [metrics_baseline["dp_diff"]],
                s=120, marker="X", color="#e74c3c", edgecolor="black",
                linewidth=1, zorder=10, label="Baseline")
    ax2.scatter([metrics_reweighed["accuracy"]], [metrics_reweighed["dp_diff"]],
                s=120, marker="D", color="#3498db", edgecolor="black",
                linewidth=1, zorder=10, label="Reweighed")
    ax2.scatter([metrics_threshold["accuracy"]], [metrics_threshold["dp_diff"]],
                s=120, marker="s", color="#2ecc71", edgecolor="black",
                linewidth=1, zorder=10, label="Threshold Opt")

    ax2.set_xlabel("Accuracy", fontsize=11)
    ax2.set_ylabel("|Demographic Parity Difference|", fontsize=11)
    ax2.set_title("Fairness-Accuracy Pareto Frontier", fontsize=13)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved to: {save_path}")
    plt.close()


# ====== Section 7: Full Audit Helper ======

def full_fairness_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    protected: np.ndarray,
    X_scaled: np.ndarray,
    label: str = "Model",
) -> dict:
    """Run a comprehensive fairness audit and print results."""
    dp = demographic_parity_difference(y_pred, protected)
    eo = equalized_odds_difference(y_true, y_pred, protected)
    calib = calibration_difference(y_true, y_prob, protected)
    acc = accuracy_score(y_true, y_pred)
    ind_fair = individual_fairness_consistency(X_scaled, y_pred, k=5)

    # Approval rates per group
    rate_maj = y_pred[protected == 0].mean()
    rate_min = y_pred[protected == 1].mean()

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    print(f"\n  --- {label} ---")
    print(f"  Accuracy:        {acc:.4f}   |   AUC: {auc:.4f}")
    print(f"  Approval rate (majority):  {rate_maj:.4f}")
    print(f"  Approval rate (minority):  {rate_min:.4f}")
    print(f"  Demographic parity diff:   {dp:+.4f}")
    print(f"  Equalized odds TPR gap:    {eo['tpr_gap']:+.4f}")
    print(f"  Equalized odds FPR gap:    {eo['fpr_gap']:+.4f}")
    print(f"  Equalized odds max gap:    {eo['max_gap']:.4f}")
    print(f"  Calibration difference:    {calib:.4f}")
    print(f"  Individual fairness (k=5): {ind_fair:.4f}")

    return {
        "accuracy": acc,
        "auc": auc,
        "dp_diff": abs(dp),
        "eo_max_gap": eo["max_gap"],
        "calib_diff": calib,
        "individual_fairness": ind_fair,
    }


# ====== Section 8: Main Pipeline ======

def main() -> None:
    """Run the full fairness audit and mitigation pipeline."""
    print("=" * 65)
    print("  Fairness Toolkit for Machine Learning")
    print("  Metrics | Reweighing | Threshold Optimization | Pareto")
    print("=" * 65)

    # --- Step 1: Generate biased dataset ---
    print("\n[1] Generating Synthetic Biased Credit Scoring Dataset")
    print("-" * 50)

    df = generate_biased_credit_data(n_samples=3000, bias_strength=0.8)

    print(f"  Total samples: {len(df)}")
    print(f"  Majority group (A=0): {(df['protected'] == 0).sum()}")
    print(f"  Minority group (A=1): {(df['protected'] == 1).sum()}")
    print(f"  Overall approval rate: {df['approved'].mean():.3f}")
    print(f"  Majority approval rate: {df[df['protected'] == 0]['approved'].mean():.3f}")
    print(f"  Minority approval rate: {df[df['protected'] == 1]['approved'].mean():.3f}")
    print(f"  Disparity (intentional bias): "
          f"{df[df['protected'] == 0]['approved'].mean() - df[df['protected'] == 1]['approved'].mean():.3f}")

    # --- Step 2: Train-test split ---
    print("\n[2] Preparing Data and Training Baseline Classifier")
    print("-" * 50)

    X = df[FEATURE_COLS].values
    y = df["approved"].values
    protected = df["protected"].values

    X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
        X, y, protected, test_size=0.25, random_state=42, stratify=y,
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Train baseline (no fairness intervention)
    clf_baseline = LogisticRegression(max_iter=1000, random_state=42)
    clf_baseline.fit(X_train_sc, y_train)

    y_pred_base = clf_baseline.predict(X_test_sc)
    y_prob_base = clf_baseline.predict_proba(X_test_sc)[:, 1]

    print(f"  Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # --- Step 3: Baseline fairness audit ---
    print("\n[3] Baseline Fairness Audit")
    print("-" * 50)

    metrics_baseline = full_fairness_audit(
        y_test, y_pred_base, y_prob_base, prot_test, X_test_sc,
        label="Baseline (no intervention)",
    )

    # --- Step 4: Reweighing mitigation ---
    print("\n[4] Pre-Processing Mitigation: Reweighing")
    print("-" * 50)

    weights = compute_reweighing_weights(y_train, prot_train)
    print(f"  Weight statistics:")
    for a in [0, 1]:
        for yval in [0, 1]:
            mask = (prot_train == a) & (y_train == yval)
            if mask.sum() > 0:
                w = weights[mask][0]
                print(f"    A={a}, Y={yval}: weight={w:.4f}  (n={mask.sum()})")

    # Retrain with sample weights
    clf_reweighed = LogisticRegression(max_iter=1000, random_state=42)
    clf_reweighed.fit(X_train_sc, y_train, sample_weight=weights)

    y_pred_rw = clf_reweighed.predict(X_test_sc)
    y_prob_rw = clf_reweighed.predict_proba(X_test_sc)[:, 1]

    metrics_reweighed = full_fairness_audit(
        y_test, y_pred_rw, y_prob_rw, prot_test, X_test_sc,
        label="Reweighed (pre-processing)",
    )

    # --- Step 5: Threshold optimization ---
    print("\n[5] Post-Processing Mitigation: Threshold Optimization")
    print("-" * 50)

    opt_result = optimize_group_thresholds(
        y_test, y_prob_base, prot_test,
        target_metric="demographic_parity",
    )
    print(f"  Optimal thresholds:")
    print(f"    Majority group: {opt_result['threshold_majority']:.3f}")
    print(f"    Minority group: {opt_result['threshold_minority']:.3f}")

    y_pred_thresh = opt_result["y_pred"]

    metrics_threshold = full_fairness_audit(
        y_test, y_pred_thresh, y_prob_base, prot_test, X_test_sc,
        label="Threshold-optimized (post-processing)",
    )

    # --- Step 6: Pareto frontier ---
    print("\n[6] Fairness-Accuracy Pareto Frontier")
    print("-" * 50)

    all_acc, all_dpd, pareto_acc, pareto_dpd, pareto_thresh = compute_pareto_frontier(
        y_test, y_prob_base, prot_test,
    )
    print(f"  Evaluated {len(all_acc)} threshold combinations")
    print(f"  Pareto-optimal points: {len(pareto_acc)}")
    print(f"  Accuracy range on frontier: [{min(pareto_acc):.3f}, {max(pareto_acc):.3f}]")
    print(f"  DPD range on frontier: [{min(pareto_dpd):.3f}, {max(pareto_dpd):.3f}]")

    # --- Step 7: Visualization ---
    print("\n[7] Generating Fairness Audit Visualization")
    print("-" * 50)

    visualize_fairness_audit(
        metrics_baseline, metrics_reweighed, metrics_threshold,
        all_acc, all_dpd, pareto_acc, pareto_dpd,
    )

    # --- Step 8: Comparative summary ---
    print("\n[8] Comparative Summary")
    print("-" * 50)

    header = f"  {'Method':30s} {'Accuracy':>10s} {'|DP Diff|':>10s} {'EO Gap':>10s} {'Ind.Fair':>10s}"
    print(header)
    print("  " + "-" * 72)
    for name, m in [
        ("Baseline", metrics_baseline),
        ("Reweighed", metrics_reweighed),
        ("Threshold-optimized", metrics_threshold),
    ]:
        print(f"  {name:30s} {m['accuracy']:10.4f} {m['dp_diff']:10.4f} "
              f"{m['eo_max_gap']:10.4f} {m['individual_fairness']:10.4f}")

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    print("""
  Key takeaways:
    1. Demographic parity (equal approval rates) and equalized odds
       (equal TPR/FPR) are distinct fairness criteria that often
       conflict with each other.
    2. Reweighing adjusts the training data so the model does not
       learn from historical bias in the labels.  It is effective
       but may reduce accuracy.
    3. Threshold optimization is a post-processing technique that
       preserves the model's learned representations but adjusts
       the decision boundary per group.
    4. The Pareto frontier reveals the fundamental trade-off:
       perfect fairness and maximum accuracy are generally not
       achievable simultaneously.
    5. Individual fairness (similar people get similar predictions)
       is complementary to group fairness and can reveal issues
       that group metrics miss.
    6. In practice, the choice of fairness criterion depends on
       the application's legal and ethical context.
    """)


if __name__ == "__main__":
    main()
