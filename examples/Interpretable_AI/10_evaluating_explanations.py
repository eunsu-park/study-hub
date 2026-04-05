"""
10. Evaluating Explanations

Implements quantitative metrics for measuring explanation quality on a
synthetic tabular classification task. Compares multiple attribution
methods using faithfulness, stability, and sparsity metrics.

Covered topics:
    - Comprehensiveness (feature removal) and sufficiency (feature retention)
    - Monotonicity deletion curves for attribution ranking validation
    - Explanation stability via Lipschitz continuity estimation
    - Sensitivity analysis (max-sensitivity) for robustness
    - ROAR-style benchmark (Remove-and-Retrain approximation)
    - Benchmarking multiple methods on the same model/dataset

Related to: L10 - Evaluating Explanations

Requirements:
    pip install numpy matplotlib scikit-learn
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


# ====== Section 1: Synthetic Dataset ======

def generate_data(
    n: int = 2000,
    n_features: int = 10,
    n_informative: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a classification dataset with known informative features.

    The first n_informative features are truly predictive (non-zero
    ground-truth importance). The remaining features are noise. This
    lets us later check whether explanation methods correctly identify
    the informative subset.

    Args:
        n: Number of samples.
        n_features: Total number of features.
        n_informative: Number of truly predictive features.
        seed: Random seed.

    Returns:
        X: Feature matrix (n, n_features).
        y: Binary labels (n,).
        true_weights: Ground-truth feature importance vector (n_features,).
    """
    rng = np.random.default_rng(seed)

    X = rng.normal(0, 1, (n, n_features))

    # True decision function: only first n_informative features matter
    true_weights = np.zeros(n_features)
    true_weights[:n_informative] = rng.uniform(0.5, 2.0, n_informative)
    true_weights[:n_informative] *= rng.choice([-1, 1], n_informative)

    logits = X @ true_weights + rng.normal(0, 0.3, n)
    y = (logits > 0).astype(int)

    return X, y, true_weights


# ====== Section 2: Simple Attribution Methods ======

def gradient_importance(model, X: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """Finite-difference gradient attribution for a sklearn model.

    Approximates d(prediction)/d(feature_i) using central differences.
    This is the tabular equivalent of vanilla saliency for neural nets.

    Args:
        model: Trained classifier with predict_proba.
        X: Input samples (n, d).
        epsilon: Perturbation size.

    Returns:
        Attribution matrix (n, d).
    """
    n, d = X.shape
    base_probs = model.predict_proba(X)[:, 1]
    attributions = np.zeros((n, d))

    for j in range(d):
        X_plus = X.copy()
        X_minus = X.copy()
        X_plus[:, j] += epsilon
        X_minus[:, j] -= epsilon

        prob_plus = model.predict_proba(X_plus)[:, 1]
        prob_minus = model.predict_proba(X_minus)[:, 1]

        attributions[:, j] = (prob_plus - prob_minus) / (2 * epsilon)

    return attributions


def permutation_importance_local(
    model, X: np.ndarray, n_repeats: int = 20, seed: int = 42,
) -> np.ndarray:
    """Local permutation importance: per-sample feature importance.

    For each sample, we measure how much the prediction changes when
    each feature is replaced with random values from the marginal
    distribution.

    Args:
        model: Trained classifier with predict_proba.
        X: Input samples (n, d).
        n_repeats: Permutation repeats per feature.
        seed: Random seed.

    Returns:
        Attribution matrix (n, d).
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    base_probs = model.predict_proba(X)[:, 1]
    attributions = np.zeros((n, d))

    for j in range(d):
        diffs = np.zeros(n)
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_probs = model.predict_proba(X_perm)[:, 1]
            diffs += np.abs(base_probs - perm_probs)
        attributions[:, j] = diffs / n_repeats

    return attributions


def random_attribution(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """Random baseline: assigns uniformly random importance.

    Serves as a sanity check -- any real method should outperform random
    attribution on faithfulness metrics.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, X.shape)


# ====== Section 3: Faithfulness Metrics ======

def comprehensiveness(
    model, X: np.ndarray, attributions: np.ndarray, top_k: int = 3,
) -> float:
    """Comprehensiveness: prediction drop when top-k features are removed.

    If the top-k attributed features are truly important, removing them
    (setting to zero) should significantly change the prediction. Higher
    comprehensiveness = the explanation correctly identifies important
    features.

    Args:
        model: Trained classifier.
        X: Input samples (n, d).
        attributions: Attribution matrix (n, d).
        top_k: Number of top features to remove.

    Returns:
        Mean absolute prediction change.
    """
    base_probs = model.predict_proba(X)[:, 1]
    changes = []

    for i in range(len(X)):
        # Find top-k features for this sample
        top_indices = np.argsort(np.abs(attributions[i]))[::-1][:top_k]

        X_masked = X[i].copy()
        X_masked[top_indices] = 0.0  # remove by zeroing

        new_prob = model.predict_proba(X_masked.reshape(1, -1))[0, 1]
        changes.append(abs(base_probs[i] - new_prob))

    return float(np.mean(changes))


def sufficiency(
    model, X: np.ndarray, attributions: np.ndarray, top_k: int = 3,
) -> float:
    """Sufficiency: prediction change when ONLY top-k features are kept.

    If the explanation is sufficient, keeping only the top-k features
    and zeroing everything else should preserve the prediction. Lower
    sufficiency = the top-k features capture the model's reasoning.

    Args:
        model: Trained classifier.
        X: Input samples (n, d).
        attributions: Attribution matrix (n, d).
        top_k: Number of features to keep.

    Returns:
        Mean absolute prediction change (lower = better sufficiency).
    """
    base_probs = model.predict_proba(X)[:, 1]
    changes = []

    for i in range(len(X)):
        top_indices = set(np.argsort(np.abs(attributions[i]))[::-1][:top_k])
        X_masked = np.zeros_like(X[i])
        for j in top_indices:
            X_masked[j] = X[i, j]

        new_prob = model.predict_proba(X_masked.reshape(1, -1))[0, 1]
        changes.append(abs(base_probs[i] - new_prob))

    return float(np.mean(changes))


def monotonicity_deletion_curve(
    model, X: np.ndarray, attributions: np.ndarray,
) -> list[float]:
    """Deletion curve: progressively remove features by importance rank.

    A faithful attribution should produce a monotonically decreasing
    prediction confidence as features are removed in order of attributed
    importance.

    Returns:
        List of mean prediction probabilities after removing 0, 1, ..., d features.
    """
    n, d = X.shape
    base_probs = model.predict_proba(X)[:, 1]
    curve = [float(np.mean(base_probs))]

    for step in range(d):
        probs = []
        for i in range(n):
            ranked = np.argsort(np.abs(attributions[i]))[::-1]
            X_masked = X[i].copy()
            X_masked[ranked[: step + 1]] = 0.0
            prob = model.predict_proba(X_masked.reshape(1, -1))[0, 1]
            probs.append(prob)
        curve.append(float(np.mean(probs)))

    return curve


# ====== Section 4: Stability Metrics ======

def max_sensitivity(
    model, X: np.ndarray, attr_fn, n_perturbations: int = 20,
    epsilon: float = 0.1, seed: int = 42,
) -> float:
    """Max-sensitivity: largest attribution change under small input perturbation.

    A stable explanation should not change drastically when the input is
    perturbed by a small amount.  High sensitivity indicates the
    explanation is fragile.

    Args:
        model: Trained classifier.
        X: Input samples (n, d).
        attr_fn: Function that returns attribution matrix for X.
        n_perturbations: Number of random perturbations.
        epsilon: L-infinity perturbation bound.
        seed: Random seed.

    Returns:
        Mean max-sensitivity across samples.
    """
    rng = np.random.default_rng(seed)
    base_attr = attr_fn(model, X)
    n = min(len(X), 100)  # limit for efficiency

    max_diffs = []
    for i in range(n):
        worst = 0.0
        for _ in range(n_perturbations):
            noise = rng.uniform(-epsilon, epsilon, X.shape[1])
            X_pert = X[i:i + 1].copy()
            X_pert[0] += noise
            pert_attr = attr_fn(model, X_pert)
            diff = np.linalg.norm(base_attr[i] - pert_attr[0])
            worst = max(worst, diff)
        max_diffs.append(worst)

    return float(np.mean(max_diffs))


# ====== Section 5: Visualization ======

def visualize_evaluation(
    results: dict,
    deletion_curves: dict,
    save_path: str = "evaluating_explanations.png",
) -> None:
    """Four-panel visualization of explanation evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = list(results.keys())
    colors = {"Gradient": "#2ecc71", "Permutation": "#3498db", "Random": "#e74c3c"}

    # --- Panel 1: Comprehensiveness ---
    ax1 = axes[0, 0]
    comp_vals = [results[m]["comprehensiveness"] for m in methods]
    bars = ax1.bar(methods, comp_vals,
                   color=[colors[m] for m in methods],
                   edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Mean Prediction Change")
    ax1.set_title("Comprehensiveness (higher = better)\nDrop when top-3 removed")

    # --- Panel 2: Sufficiency ---
    ax2 = axes[0, 1]
    suf_vals = [results[m]["sufficiency"] for m in methods]
    ax2.bar(methods, suf_vals,
            color=[colors[m] for m in methods],
            edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Mean Prediction Change")
    ax2.set_title("Sufficiency (lower = better)\nDrop when only top-3 kept")

    # --- Panel 3: Deletion Curves ---
    ax3 = axes[1, 0]
    for method in methods:
        ax3.plot(deletion_curves[method], label=method,
                 color=colors[method], linewidth=2)
    ax3.set_xlabel("Features Removed (by importance rank)")
    ax3.set_ylabel("Mean Prediction Probability")
    ax3.set_title("Monotonicity Deletion Curves")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Sensitivity ---
    ax4 = axes[1, 1]
    sens_vals = [results[m]["sensitivity"] for m in methods]
    ax4.bar(methods, sens_vals,
            color=[colors[m] for m in methods],
            edgecolor="black", linewidth=0.5)
    ax4.set_ylabel("Max Sensitivity")
    ax4.set_title("Explanation Stability (lower = more stable)")

    plt.suptitle("Evaluating Explanation Quality", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved to: {save_path}")
    plt.close()


# ====== Section 6: Main Pipeline ======

def main() -> None:
    """Evaluate multiple attribution methods on faithfulness and stability."""
    print("=" * 65)
    print("  Evaluating Explanations")
    print("  Faithfulness | Stability | Deletion Curves")
    print("=" * 65)

    # --- Step 1: Generate data ---
    print("\n[1] Generating Synthetic Dataset")
    print("-" * 50)

    X, y, true_weights = generate_data(n=2000, n_features=10, n_informative=4)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )
    print(f"  Features: {X.shape[1]} (4 informative, 6 noise)")
    print(f"  Train/Test: {len(X_train)}/{len(X_test)}")
    print(f"  True weights: {true_weights}")

    # --- Step 2: Train model ---
    print("\n[2] Training Gradient Boosting Classifier")
    print("-" * 50)

    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=42,
    )
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")

    # --- Step 3: Compute attributions ---
    print("\n[3] Computing Attributions")
    print("-" * 50)

    # Use a subset for efficiency
    X_eval = X_test[:200]

    attr_gradient = gradient_importance(model, X_eval)
    print("  Gradient attribution computed.")

    attr_permutation = permutation_importance_local(model, X_eval)
    print("  Permutation attribution computed.")

    attr_random = random_attribution(X_eval)
    print("  Random baseline attribution generated.")

    # --- Step 4: Faithfulness metrics ---
    print("\n[4] Faithfulness Metrics")
    print("-" * 50)

    methods_attrs = {
        "Gradient": attr_gradient,
        "Permutation": attr_permutation,
        "Random": attr_random,
    }

    results = {}
    for name, attr in methods_attrs.items():
        comp = comprehensiveness(model, X_eval, attr, top_k=3)
        suf = sufficiency(model, X_eval, attr, top_k=3)
        results[name] = {"comprehensiveness": comp, "sufficiency": suf}
        print(f"  {name:12s}: comprehensiveness={comp:.4f}, sufficiency={suf:.4f}")

    # --- Step 5: Deletion curves ---
    print("\n[5] Monotonicity Deletion Curves")
    print("-" * 50)

    # Use smaller subset for deletion curves
    X_curve = X_eval[:50]
    deletion_curves = {}
    for name, attr in methods_attrs.items():
        curve = monotonicity_deletion_curve(model, X_curve, attr[:50])
        deletion_curves[name] = curve
        auc = np.trapz(curve, dx=1.0)
        print(f"  {name:12s}: AUC={auc:.4f} (lower = more faithful)")

    # --- Step 6: Stability ---
    print("\n[6] Explanation Stability (Max-Sensitivity)")
    print("-" * 50)

    attr_fns = {
        "Gradient": gradient_importance,
        "Permutation": lambda m, x: permutation_importance_local(m, x, n_repeats=5),
        "Random": lambda m, x: random_attribution(x),
    }

    for name, attr_fn in attr_fns.items():
        sens = max_sensitivity(model, X_eval[:50], attr_fn,
                               n_perturbations=10, epsilon=0.1)
        results[name]["sensitivity"] = sens
        print(f"  {name:12s}: max_sensitivity={sens:.4f}")

    # --- Step 7: Summary table ---
    print("\n[7] Summary Table")
    print("-" * 50)
    print(f"  {'Method':<12s} {'Compreh.':>10s} {'Suffic.':>10s} {'Sensitiv.':>10s}")
    print(f"  {'-' * 42}")
    for name in results:
        r = results[name]
        print(f"  {name:<12s} {r['comprehensiveness']:>10.4f} "
              f"{r['sufficiency']:>10.4f} {r['sensitivity']:>10.4f}")

    # --- Step 8: Visualization ---
    print("\n[8] Generating Visualization")
    print("-" * 50)

    visualize_evaluation(results, deletion_curves)

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    print("""
  Key findings:
    1. Comprehensiveness: higher means the method correctly identified
       the features the model relies on. Gradient and Permutation
       should both outperform Random.
    2. Sufficiency: lower means the top-k features are enough to
       preserve the prediction. A good explanation is both comprehensive
       AND sufficient.
    3. Deletion curves: a faithful method produces a steep drop when
       features are removed in importance order.
    4. Max-sensitivity: lower means the explanation is stable under
       small input perturbations. Random has low sensitivity (it ignores
       the input) but also low faithfulness.
    5. No single metric is sufficient -- combining faithfulness and
       stability gives a complete picture of explanation quality.
    """)


if __name__ == "__main__":
    main()
