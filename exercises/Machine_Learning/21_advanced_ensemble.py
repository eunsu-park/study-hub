"""
Advanced Ensemble - Exercise Solutions
========================================
Lesson 21: Advanced Ensemble

Exercises cover:
  1. Build and compare ensemble strategies (averaging, blending, stacking)
  2. Diversity analysis: correlation among base models
  3. Stacking vs single best model across dataset sizes
  4. Competition-style ensemble with greedy selection
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_classification, load_digits
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# ============================================================
# Exercise 1: Build and Compare Ensemble Strategies
# Averaging, blending, and stacking on breast cancer.
# ============================================================
def exercise_1_compare_ensembles():
    """Compare 3 ensemble strategies: averaging, blending, stacking.

    - Simple averaging: average predicted probabilities, no training needed
    - Blending: hold out 20% for training a meta-learner on OOF predictions
    - Stacking: use cross-validated OOF predictions (more data-efficient)

    Stacking is more robust than blending because it uses all training data
    for both base models and meta-learner via cross-validation.
    """
    print("=" * 60)
    print("Exercise 1: Compare Ensemble Strategies")
    print("=" * 60)

    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define 5 diverse base models
    base_models = {
        "LR": Pipeline([("s", StandardScaler()),
                        ("m", LogisticRegression(max_iter=5000, random_state=42))]),
        "RF": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "GB": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM": Pipeline([("s", StandardScaler()),
                         ("m", SVC(probability=True, random_state=42))]),
        "KNN": Pipeline([("s", StandardScaler()),
                         ("m", KNeighborsClassifier(n_neighbors=5))]),
    }

    # Individual model performance
    print(f"{'Model':<10} {'CV Accuracy':>12}")
    print("-" * 24)
    individual_scores = {}
    for name, model in base_models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        individual_scores[name] = scores.mean()
        print(f"{name:<10} {scores.mean():>12.4f}")

    # Strategy 1: Simple probability averaging
    probas = []
    for name, model in base_models.items():
        model.fit(X_train, y_train)
        probas.append(model.predict_proba(X_test)[:, 1])
    avg_proba = np.mean(probas, axis=0)
    y_pred_avg = (avg_proba > 0.5).astype(int)
    acc_avg = accuracy_score(y_test, y_pred_avg)

    # Strategy 2: Blending (20% holdout for meta-learner)
    X_base, X_blend, y_base, y_blend = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    blend_features_train = []
    blend_features_test = []
    for name, model in base_models.items():
        model.fit(X_base, y_base)
        blend_features_train.append(model.predict_proba(X_blend)[:, 1])
        blend_features_test.append(model.predict_proba(X_test)[:, 1])

    meta_X_train = np.column_stack(blend_features_train)
    meta_X_test = np.column_stack(blend_features_test)
    meta_lr = LogisticRegression(random_state=42)
    meta_lr.fit(meta_X_train, y_blend)
    y_pred_blend = meta_lr.predict(meta_X_test)
    acc_blend = accuracy_score(y_test, y_pred_blend)

    # Strategy 3: Stacking with StackingClassifier
    estimators = [(name, model) for name, model in base_models.items()]
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )
    stacking.fit(X_train, y_train)
    y_pred_stack = stacking.predict(X_test)
    acc_stack = accuracy_score(y_test, y_pred_stack)

    # Also test with passthrough=True
    stacking_pt = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=42),
        cv=5,
        passthrough=True
    )
    stacking_pt.fit(X_train, y_train)
    acc_stack_pt = accuracy_score(y_test, stacking_pt.predict(X_test))

    print(f"\n{'Strategy':<25} {'Test Accuracy':>14}")
    print("-" * 40)
    print(f"{'Best individual':<25} {max(individual_scores.values()):>14.4f}")
    print(f"{'Simple averaging':<25} {acc_avg:>14.4f}")
    print(f"{'Blending (holdout)':<25} {acc_blend:>14.4f}")
    print(f"{'Stacking (CV=5)':<25} {acc_stack:>14.4f}")
    print(f"{'Stacking (passthrough)':<25} {acc_stack_pt:>14.4f}")


# ============================================================
# Exercise 2: Diversity Analysis
# Measure correlation among base model predictions.
# ============================================================
def exercise_2_diversity_analysis():
    """Analyze prediction diversity among base models.

    Diversity is the key to successful ensembles: if all models make the
    same errors, combining them won't help. We measure diversity by
    computing the correlation matrix of out-of-fold (OOF) predictions.

    Low correlation = high diversity = more room for ensemble improvement.
    We compare stacking with the 3 most diverse vs 3 least diverse models.
    """
    print("\n" + "=" * 60)
    print("Exercise 2: Diversity Analysis")
    print("=" * 60)

    X, y = make_classification(
        n_samples=3000, n_features=30, n_informative=15,
        n_redundant=5, random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "LR": Pipeline([("s", StandardScaler()),
                        ("m", LogisticRegression(max_iter=5000, random_state=42))]),
        "RF": RandomForestClassifier(n_estimators=100, random_state=42),
        "GB": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM": Pipeline([("s", StandardScaler()),
                         ("m", SVC(probability=True, random_state=42))]),
        "KNN": Pipeline([("s", StandardScaler()),
                         ("m", KNeighborsClassifier(n_neighbors=10))]),
        "RF_deep": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
    }

    # Generate OOF predictions for each model
    oof_preds = {}
    for name, model in models.items():
        oof = cross_val_predict(model, X, y, cv=cv, method="predict")
        oof_preds[name] = oof

    # Correlation matrix
    names = list(oof_preds.keys())
    n_models = len(names)
    corr_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            corr_matrix[i, j] = np.corrcoef(
                oof_preds[names[i]], oof_preds[names[j]]
            )[0, 1]

    print("Prediction correlation matrix:")
    print(f"{'':>10}", end="")
    for name in names:
        print(f"{name:>10}", end="")
    print()
    for i, name in enumerate(names):
        print(f"{name:>10}", end="")
        for j in range(n_models):
            print(f"{corr_matrix[i, j]:>10.3f}", end="")
        print()

    # Find 3 most diverse and 3 least diverse pairs
    # Average off-diagonal correlation per model
    avg_corr = {}
    for i, name in enumerate(names):
        other_corrs = [corr_matrix[i, j] for j in range(n_models) if i != j]
        avg_corr[name] = np.mean(other_corrs)

    sorted_by_diversity = sorted(avg_corr.items(), key=lambda x: x[1])
    most_diverse = [name for name, _ in sorted_by_diversity[:3]]
    least_diverse = [name for name, _ in sorted_by_diversity[-3:]]

    print(f"\nMost diverse models (lowest avg correlation): {most_diverse}")
    print(f"Least diverse models (highest avg correlation): {least_diverse}")

    # Compare stacking performance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    for label, selected in [("Most Diverse", most_diverse),
                             ("Least Diverse", least_diverse)]:
        estimators = [(name, models[name]) for name in selected]
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        )
        stack.fit(X_train, y_train)
        acc = stack.score(X_test, y_test)
        print(f"\n{label} Stack ({', '.join(selected)}): accuracy = {acc:.4f}")


# ============================================================
# Exercise 3: Stacking vs Single Best Model
# Compare at different dataset sizes.
# ============================================================
def exercise_3_stacking_vs_single():
    """Compare stacking vs best single model across dataset sizes.

    Hypothesis: stacking benefits more from larger datasets because:
    1. More data for reliable OOF predictions
    2. More data for the meta-learner to learn combination weights
    3. Less risk of overfitting the meta-features

    On very small datasets, stacking can overfit because the meta-learner
    trains on noisy OOF predictions from limited data.
    """
    print("\n" + "=" * 60)
    print("Exercise 3: Stacking vs Single Best at Different Sizes")
    print("=" * 60)

    sizes = [200, 500, 1000, 3000, 5000]
    results = {"size": [], "best_single": [], "stacking": []}

    base_estimators = [
        ("lr", Pipeline([("s", StandardScaler()),
                         ("m", LogisticRegression(max_iter=5000, random_state=42))])),
        ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ]

    for size in sizes:
        X, y = make_classification(
            n_samples=size, n_features=20, n_informative=10,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Best single model
        best_single = 0
        for name, model in base_estimators:
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            best_single = max(best_single, acc)

        # Stacking
        stack = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        )
        stack.fit(X_train, y_train)
        stack_acc = stack.score(X_test, y_test)

        results["size"].append(size)
        results["best_single"].append(best_single)
        results["stacking"].append(stack_acc)

    print(f"{'Size':<8} {'Best Single':>12} {'Stacking':>10} {'Gap':>8}")
    print("-" * 40)
    for i, size in enumerate(sizes):
        gap = results["stacking"][i] - results["best_single"][i]
        print(f"{size:<8} {results['best_single'][i]:>12.4f} "
              f"{results['stacking'][i]:>10.4f} {gap:>+8.4f}")

    # Plot
    plt.figure(figsize=(9, 5))
    plt.plot(sizes, results["best_single"], "o-", label="Best Single Model")
    plt.plot(sizes, results["stacking"], "s-", label="Stacking")
    plt.xlabel("Dataset Size")
    plt.ylabel("Test Accuracy")
    plt.title("Stacking vs Best Single Model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("21_ex3_stacking_size.png", dpi=100)
    plt.close()
    print("\nPlot saved: 21_ex3_stacking_size.png")


# ============================================================
# Exercise 4: Competition-Style Ensemble
# Stacking + greedy ensemble selection on digits.
# ============================================================
def exercise_4_competition_ensemble():
    """Competition-style ensemble with greedy model selection.

    Greedy ensemble selection (Caruana et al., 2004):
    1. Start with the best single model
    2. At each step, add the model that improves the ensemble the most
    3. Stop when no model improves performance

    This automatically selects a subset of models and handles redundancy:
    adding a highly correlated model won't improve the ensemble.
    """
    print("\n" + "=" * 60)
    print("Exercise 4: Competition-Style Ensemble")
    print("=" * 60)

    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5+ base models
    models = {
        "LR": Pipeline([("s", StandardScaler()),
                        ("m", LogisticRegression(max_iter=5000, random_state=42))]),
        "RF_100": RandomForestClassifier(n_estimators=100, random_state=42),
        "RF_200": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
        "GB_100": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "GB_200": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                              random_state=42),
        "KNN": Pipeline([("s", StandardScaler()),
                         ("m", KNeighborsClassifier(n_neighbors=3))]),
    }

    # Get OOF predictions (probabilities) for each model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probas = {}  # model_name -> (n_samples, n_classes) probabilities
    test_probas = {}

    for name, model in models.items():
        # OOF predictions
        oof = cross_val_predict(model, X_train, y_train, cv=cv,
                                method="predict_proba")
        oof_probas[name] = oof

        # Test predictions
        model.fit(X_train, y_train)
        test_probas[name] = model.predict_proba(X_test)

    # Individual model accuracy
    print(f"{'Model':<12} {'OOF Accuracy':>14}")
    print("-" * 28)
    for name in models:
        oof_pred = np.argmax(oof_probas[name], axis=1)
        acc = accuracy_score(y_train, oof_pred)
        print(f"{name:<12} {acc:>14.4f}")

    # Greedy ensemble selection
    model_names = list(models.keys())
    selected = []
    best_acc = 0

    while True:
        best_candidate = None
        best_candidate_acc = best_acc

        for name in model_names:
            if name in selected:
                continue

            # Trial: add this model to current ensemble
            candidate = selected + [name]
            # Average OOF probabilities
            avg_proba = np.mean([oof_probas[n] for n in candidate], axis=0)
            pred = np.argmax(avg_proba, axis=1)
            acc = accuracy_score(y_train, pred)

            if acc > best_candidate_acc:
                best_candidate_acc = acc
                best_candidate = name

        if best_candidate is None:
            break

        selected.append(best_candidate)
        best_acc = best_candidate_acc
        print(f"\nAdded {best_candidate}: OOF accuracy = {best_acc:.4f}")

    print(f"\nGreedy selection: {selected}")

    # Final ensemble on test set
    avg_test_proba = np.mean([test_probas[n] for n in selected], axis=0)
    y_pred_greedy = np.argmax(avg_test_proba, axis=1)
    acc_greedy = accuracy_score(y_test, y_pred_greedy)

    # Full ensemble (all models)
    avg_test_all = np.mean(list(test_probas.values()), axis=0)
    y_pred_all = np.argmax(avg_test_all, axis=1)
    acc_all = accuracy_score(y_test, y_pred_all)

    print(f"\n{'Approach':<25} {'Test Accuracy':>14}")
    print("-" * 40)
    print(f"{'Greedy selection':<25} {acc_greedy:>14.4f}")
    print(f"{'All models averaged':<25} {acc_all:>14.4f}")

    # Best single model on test
    best_name = max(models.keys(),
                    key=lambda n: accuracy_score(y_test,
                                                  np.argmax(test_probas[n], axis=1)))
    best_single_acc = accuracy_score(y_test, np.argmax(test_probas[best_name], axis=1))
    print(f"{'Best single (' + best_name + ')':<25} {best_single_acc:>14.4f}")


if __name__ == "__main__":
    exercise_1_compare_ensembles()
    exercise_2_diversity_analysis()
    exercise_3_stacking_vs_single()
    exercise_4_competition_ensemble()
