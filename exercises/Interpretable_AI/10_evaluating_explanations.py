"""
Exercises for Lesson 10: Evaluating Explanations
Topic: Interpretable_AI

Solutions to practice problems from the lesson.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# === Exercise 1: Comprehensiveness and Sufficiency Scores ===
# Problem: Given a model and feature attributions, compute comprehensiveness
# (drop in prediction when top features removed) and sufficiency (prediction
# retained when only top features kept).

def exercise_1():
    """Compute comprehensiveness and sufficiency scores for feature attributions."""
    np.random.seed(42)

    # Train a simple model
    n = 500
    X = np.random.randn(n, 8)
    # True: features 0, 1, 2 matter; rest are noise
    y = (2.0 * X[:, 0] + 1.5 * X[:, 1] - 1.0 * X[:, 2] > 0).astype(int)

    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X, y)

    # Feature attributions (absolute coefficient magnitude as proxy)
    attributions = np.abs(model.coef_[0])
    ranked_features = np.argsort(-attributions)

    print("  Feature attributions (|coefficient|):")
    for i, idx in enumerate(ranked_features):
        print(f"    Rank {i+1}: Feature {idx} = {attributions[idx]:.4f}")

    # Compute comprehensiveness: for top-k features, replace them with zeros
    # and measure prediction probability drop
    test_X = X[:100]
    original_probs = model.predict_proba(test_X)
    original_preds = np.max(original_probs, axis=1)

    print("\n  Comprehensiveness (higher = more faithful explanation):")
    for k in [1, 2, 3, 5]:
        top_k = ranked_features[:k]
        X_masked = test_X.copy()
        X_masked[:, top_k] = 0.0  # Remove top-k features
        masked_probs = model.predict_proba(X_masked)
        masked_preds = np.max(masked_probs, axis=1)
        comprehensiveness = np.mean(original_preds - masked_preds)
        print(f"    Top-{k} removed: comprehensiveness = {comprehensiveness:.4f}")

    # Compute sufficiency: keep only top-k features, zero out the rest
    print("\n  Sufficiency (lower = more sufficient explanation):")
    for k in [1, 2, 3, 5]:
        top_k = ranked_features[:k]
        X_sufficient = np.zeros_like(test_X)
        X_sufficient[:, top_k] = test_X[:, top_k]
        sufficient_probs = model.predict_proba(X_sufficient)
        sufficient_preds = np.max(sufficient_probs, axis=1)
        sufficiency = np.mean(original_preds - sufficient_preds)
        print(f"    Top-{k} kept:    sufficiency = {sufficiency:.4f}")

    print("\n  Good explanations have high comprehensiveness and low sufficiency.")


# === Exercise 2: Explanation Stability Under Perturbation ===
# Problem: Measure how stable explanations are when input features are
# slightly perturbed. Stable explanations should not change dramatically.

def exercise_2():
    """Measure explanation stability under small input perturbations."""
    np.random.seed(42)

    # Train model
    n = 500
    X = np.random.randn(n, 6)
    y = (X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.5 * np.random.randn(n) > 0).astype(int)

    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X, y)

    # Generate explanations for a set of test points using coefficient-based attribution
    test_points = X[:20]
    coefficients = model.coef_[0]

    def get_explanation(x):
        """Feature attribution as coefficient * feature value."""
        return coefficients * x

    # Measure stability: perturb inputs and check explanation similarity
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    print("  Explanation Stability (Spearman rank correlation, higher = more stable):")
    print(f"  {'Noise Level':>12} {'Mean Rank Corr':>15} {'Mean L2 Shift':>14}")

    for noise in noise_levels:
        rank_correlations = []
        l2_shifts = []

        for x in test_points:
            original_expl = get_explanation(x)
            original_rank = np.argsort(-np.abs(original_expl))

            # Perturbed explanation
            x_perturbed = x + np.random.normal(0, noise, size=x.shape)
            perturbed_expl = get_explanation(x_perturbed)
            perturbed_rank = np.argsort(-np.abs(perturbed_expl))

            # Rank correlation (simplified: fraction of top-3 that match)
            top3_orig = set(original_rank[:3])
            top3_pert = set(perturbed_rank[:3])
            rank_corr = len(top3_orig & top3_pert) / 3.0
            rank_correlations.append(rank_corr)

            # L2 distance between normalized explanations
            norm_orig = original_expl / (np.linalg.norm(original_expl) + 1e-10)
            norm_pert = perturbed_expl / (np.linalg.norm(perturbed_expl) + 1e-10)
            l2_shifts.append(np.linalg.norm(norm_orig - norm_pert))

        print(f"  {noise:>12.3f} {np.mean(rank_correlations):>15.4f} "
              f"{np.mean(l2_shifts):>14.4f}")

    print("\n  Stable explanations maintain consistent feature rankings")
    print("  even under small input perturbations.")


# === Exercise 3: Simplified ROAR Evaluation ===
# Problem: Implement RemOve And Retrain (ROAR) to evaluate explanation quality.
# Remove top-k% features according to attributions, retrain, measure accuracy drop.

def exercise_3():
    """Implement simplified ROAR evaluation for explanation methods."""
    np.random.seed(42)

    # Generate dataset
    n = 800
    X = np.random.randn(n, 10)
    # Only features 0-3 are truly important
    y = (1.5 * X[:, 0] + X[:, 1] - 0.8 * X[:, 2] + 0.5 * X[:, 3] > 0).astype(int)

    X_train, X_test = X[:600], X[600:]
    y_train, y_test = y[:600], y[600:]

    # Train original model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    original_acc = accuracy_score(y_test, model.predict(X_test))

    # Two attribution methods to compare
    def attribution_correct(m, X_data):
        """Uses model's feature importances (informed)."""
        return m.feature_importances_

    def attribution_random(m, X_data):
        """Random attribution baseline."""
        rng = np.random.RandomState(0)
        return rng.rand(X_data.shape[1])

    methods = {
        "Model Feature Importance": attribution_correct,
        "Random Attribution": attribution_random,
    }

    percentages = [0, 10, 20, 30, 50, 70, 90]

    print(f"  Original model accuracy: {original_acc:.4f}")
    print(f"\n  ROAR Evaluation (accuracy after removing top-k% features and retraining):")
    print(f"  {'% Removed':>10}", end="")
    for method_name in methods:
        print(f"  {method_name:>28}", end="")
    print()

    for pct in percentages:
        print(f"  {pct:>9}%", end="")
        for method_name, attr_fn in methods.items():
            attributions = attr_fn(model, X_train)
            n_remove = int(len(attributions) * pct / 100)
            top_features = np.argsort(-attributions)[:n_remove]

            # Remove features by replacing with column mean
            X_train_masked = X_train.copy()
            X_test_masked = X_test.copy()
            for f in top_features:
                col_mean = X_train[:, f].mean()
                X_train_masked[:, f] = col_mean
                X_test_masked[:, f] = col_mean

            # Retrain and evaluate
            retrained = RandomForestClassifier(n_estimators=50, random_state=42)
            retrained.fit(X_train_masked, y_train)
            acc = accuracy_score(y_test, retrained.predict(X_test_masked))
            print(f"  {acc:>28.4f}", end="")
        print()

    print("\n  A good attribution method causes faster accuracy drop in ROAR,")
    print("  because it correctly identifies the most important features.")


# === Exercise 4: Comparing Faithfulness of Two Explanation Methods ===
# Problem: Compare a faithful (coefficient-based) and unfaithful (random)
# explanation method using multiple faithfulness metrics.

def exercise_4():
    """Compare faithfulness of two explanation methods using multiple metrics."""
    np.random.seed(42)

    # Train model
    n = 600
    X = np.random.randn(n, 8)
    y = (2 * X[:, 0] - X[:, 1] + 1.5 * X[:, 2] + 0.3 * np.random.randn(n) > 0).astype(int)

    X_train, X_test = X[:500], X[500:]
    y_train, y_test = y[:500], y[500:]

    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train, y_train)

    # Method A: Coefficient-based (faithful)
    def method_a(x):
        return model.coef_[0] * x

    # Method B: Random attribution (unfaithful baseline)
    rng = np.random.RandomState(99)
    def method_b(x):
        return rng.randn(len(x))

    print("  Faithfulness Comparison: Coefficient-based vs Random Attribution")
    print("  " + "=" * 60)

    # Metric 1: Feature deletion fidelity
    def deletion_fidelity(x, explanation, model, k=3):
        original_prob = model.predict_proba(x.reshape(1, -1))[0].max()
        top_k = np.argsort(-np.abs(explanation))[:k]
        x_del = x.copy()
        x_del[top_k] = 0.0
        deleted_prob = model.predict_proba(x_del.reshape(1, -1))[0].max()
        return original_prob - deleted_prob

    fid_a, fid_b = [], []
    for x in X_test:
        fid_a.append(deletion_fidelity(x, method_a(x), model))
        fid_b.append(deletion_fidelity(x, method_b(x), model))

    print(f"\n  Metric 1: Deletion Fidelity (higher = more faithful)")
    print(f"    Method A (Coefficients): {np.mean(fid_a):.4f}")
    print(f"    Method B (Random):       {np.mean(fid_b):.4f}")

    # Metric 2: Monotonicity - removing features in attribution order
    # should cause monotonic decrease in prediction confidence
    def monotonicity_score(x, explanation, model):
        ranked = np.argsort(-np.abs(explanation))
        probs = [model.predict_proba(x.reshape(1, -1))[0].max()]
        x_mod = x.copy()
        violations = 0
        for i, feat in enumerate(ranked):
            x_mod[feat] = 0.0
            prob = model.predict_proba(x_mod.reshape(1, -1))[0].max()
            if prob > probs[-1] + 1e-8:
                violations += 1
            probs.append(prob)
        return 1.0 - violations / len(ranked)

    mono_a, mono_b = [], []
    for x in X_test:
        mono_a.append(monotonicity_score(x, method_a(x), model))
        mono_b.append(monotonicity_score(x, method_b(x), model))

    print(f"\n  Metric 2: Monotonicity (higher = more faithful)")
    print(f"    Method A (Coefficients): {np.mean(mono_a):.4f}")
    print(f"    Method B (Random):       {np.mean(mono_b):.4f}")

    # Metric 3: Infidelity - expected squared difference between
    # dot(explanation, perturbation) and model output change
    def infidelity_score(x, explanation, model, n_perturb=50):
        errors = []
        for _ in range(n_perturb):
            perturb = np.random.normal(0, 0.1, x.shape)
            predicted_change = np.dot(explanation, perturb)
            x_pert = x + perturb
            actual_change = (model.predict_proba(x_pert.reshape(1, -1))[0].max() -
                             model.predict_proba(x.reshape(1, -1))[0].max())
            errors.append((predicted_change - actual_change) ** 2)
        return np.mean(errors)

    inf_a, inf_b = [], []
    for x in X_test[:50]:  # Subset for speed
        inf_a.append(infidelity_score(x, method_a(x), model))
        inf_b.append(infidelity_score(x, method_b(x), model))

    print(f"\n  Metric 3: Infidelity (lower = more faithful)")
    print(f"    Method A (Coefficients): {np.mean(inf_a):.4f}")
    print(f"    Method B (Random):       {np.mean(inf_b):.4f}")

    # Summary
    print(f"\n  Summary:")
    a_wins = (np.mean(fid_a) > np.mean(fid_b)) + \
             (np.mean(mono_a) > np.mean(mono_b)) + \
             (np.mean(inf_a) < np.mean(inf_b))
    print(f"    Method A wins on {a_wins}/3 metrics.")
    print(f"    Coefficient-based attribution is more faithful because it")
    print(f"    directly reflects the model's decision-making process.")


if __name__ == "__main__":
    print("=== Exercise 1: Comprehensiveness and Sufficiency Scores ===")
    exercise_1()
    print("\n=== Exercise 2: Explanation Stability Under Perturbation ===")
    exercise_2()
    print("\n=== Exercise 3: Simplified ROAR Evaluation ===")
    exercise_3()
    print("\n=== Exercise 4: Comparing Faithfulness of Two Methods ===")
    exercise_4()
    print("\nAll exercises completed!")
