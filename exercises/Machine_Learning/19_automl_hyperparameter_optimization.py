"""
AutoML and Hyperparameter Optimization - Exercise Solutions
============================================================
Lesson 19: AutoML and Hyperparameter Optimization

Exercises cover:
  1. Manual Bayesian-style HPO for classification
  2. Systematic model comparison with time budget
  3. Multi-objective optimization (accuracy vs speed vs complexity)

Note: Uses only sklearn (no Optuna/FLAML dependency) to remain self-contained.
The concepts and patterns demonstrated here translate directly to Optuna/FLAML usage.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score


# ============================================================
# Exercise 1: HPO for Classification
# Search over multiple algorithms with conditional hyperparameters.
# ============================================================
def exercise_1_hpo_classification():
    """Hyperparameter optimization across multiple algorithms.

    This mimics what Optuna does: for each trial, we:
    1. Sample an algorithm
    2. Sample algorithm-specific hyperparameters
    3. Evaluate with cross-validation
    4. Track the best configuration

    In production, use Optuna's suggest_* API for smarter sampling
    (TPE sampler uses past results to guide future trials).
    """
    print("=" * 60)
    print("Exercise 1: HPO for Classification")
    print("=" * 60)

    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    np.random.seed(42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    n_trials = 50
    results = []

    for trial in range(n_trials):
        # Sample algorithm
        algo = np.random.choice(["RF", "GB", "SVM"])

        if algo == "RF":
            params = {
                "n_estimators": np.random.choice([50, 100, 200]),
                "max_depth": np.random.choice([None, 5, 10, 20]),
                "min_samples_leaf": np.random.choice([1, 2, 5, 10]),
            }
            pipe = Pipeline([
                ("clf", RandomForestClassifier(random_state=42, n_jobs=-1, **params))
            ])
        elif algo == "GB":
            params = {
                "n_estimators": np.random.choice([50, 100, 200]),
                "learning_rate": np.random.choice([0.01, 0.05, 0.1, 0.2]),
                "max_depth": np.random.choice([3, 5, 7]),
            }
            pipe = Pipeline([
                ("clf", GradientBoostingClassifier(random_state=42, **params))
            ])
        else:  # SVM
            params = {
                "C": 10 ** np.random.uniform(-2, 2),
                "gamma": 10 ** np.random.uniform(-4, 0),
            }
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", random_state=42, **params))
            ])

        # Evaluate
        start = time.time()
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")
        elapsed = time.time() - start

        results.append({
            "trial": trial,
            "algo": algo,
            "params": params,
            "f1_mean": scores.mean(),
            "f1_std": scores.std(),
            "time": elapsed,
        })

    # Find best
    best = max(results, key=lambda r: r["f1_mean"])
    print(f"Total trials: {n_trials}")
    print(f"\nBest configuration:")
    print(f"  Algorithm: {best['algo']}")
    print(f"  Params: {best['params']}")
    print(f"  F1 score: {best['f1_mean']:.4f} +/- {best['f1_std']:.4f}")
    print(f"  Time: {best['time']:.2f}s")

    # Compare with default RF
    default_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    default_scores = cross_val_score(default_rf, X, y, cv=cv, scoring="f1")
    print(f"\nDefault RF F1: {default_scores.mean():.4f} +/- {default_scores.std():.4f}")
    print(f"HPO improvement: {best['f1_mean'] - default_scores.mean():+.4f}")

    # Optimization history
    running_best = [results[0]["f1_mean"]]
    for r in results[1:]:
        running_best.append(max(running_best[-1], r["f1_mean"]))

    plt.figure(figsize=(9, 5))
    plt.plot([r["f1_mean"] for r in results], "o", alpha=0.4, markersize=4,
             label="Trial F1")
    plt.plot(running_best, "r-", linewidth=2, label="Best so far")
    plt.xlabel("Trial")
    plt.ylabel("F1 Score")
    plt.title("HPO Optimization History")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("19_ex1_hpo_history.png", dpi=100)
    plt.close()
    print("Plot saved: 19_ex1_hpo_history.png")


# ============================================================
# Exercise 2: Model Comparison with Time Budget
# Compare algorithms under the same time budget.
# ============================================================
def exercise_2_model_comparison():
    """Compare multiple algorithms under a fixed time budget.

    AutoML systems like FLAML automatically allocate time budget across
    algorithms. Here we simulate this: each algorithm gets the same total
    time, and we see how many configurations it can evaluate.

    This reveals the speed-quality trade-off: fast models (LR) can explore
    more configurations, while slow models (GB) may find better optima
    per configuration but explore fewer.
    """
    print("\n" + "=" * 60)
    print("Exercise 2: Model Comparison (Time Budget)")
    print("=" * 60)

    X, y = make_regression(
        n_samples=2000, n_features=20, n_informative=10,
        noise=10, random_state=42
    )

    time_budget = 10  # seconds per algorithm

    algorithms = {
        "Ridge": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10 ** np.random.uniform(-3, 3)))
        ]),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=np.random.choice([50, 100, 200]),
            max_depth=np.random.choice([None, 5, 10, 20]),
            random_state=42, n_jobs=-1
        ),
        "GradientBoosting": lambda: GradientBoostingRegressor(
            n_estimators=np.random.choice([50, 100, 200]),
            learning_rate=np.random.choice([0.01, 0.05, 0.1]),
            max_depth=np.random.choice([3, 5]),
            random_state=42
        ),
    }

    print(f"Time budget per algorithm: {time_budget}s")
    print(f"\n{'Algorithm':<20} {'Trials':>8} {'Best R²':>10} {'Avg Time/Trial':>15}")
    print("-" * 55)

    np.random.seed(42)
    for name, model_fn in algorithms.items():
        start = time.time()
        best_score = -np.inf
        n_trials = 0

        while time.time() - start < time_budget:
            model = model_fn()
            scores = cross_val_score(model, X, y, cv=3, scoring="r2")
            score = scores.mean()
            if score > best_score:
                best_score = score
            n_trials += 1

        elapsed = time.time() - start
        avg_time = elapsed / max(n_trials, 1)
        print(f"{name:<20} {n_trials:>8} {best_score:>10.4f} {avg_time:>14.2f}s")


# ============================================================
# Exercise 3: Multi-Objective Optimization
# Optimize accuracy, training time, and model complexity.
# ============================================================
def exercise_3_multi_objective():
    """Multi-objective optimization: accuracy vs speed vs complexity.

    In production, you rarely optimize only accuracy. Common trade-offs:
    - Accuracy vs latency (real-time serving)
    - Accuracy vs model size (edge deployment)
    - Accuracy vs training cost (budget constraints)

    We compute the Pareto front: configurations where no other configuration
    is better in ALL objectives simultaneously.
    """
    print("\n" + "=" * 60)
    print("Exercise 3: Multi-Objective Optimization")
    print("=" * 60)

    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    np.random.seed(42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    configs = []

    # Generate diverse configurations
    for n_est in [10, 25, 50, 100, 200, 300]:
        for max_depth in [2, 3, 5, 10, None]:
            for lr in [0.01, 0.05, 0.1, 0.3]:
                model = GradientBoostingClassifier(
                    n_estimators=n_est, learning_rate=lr,
                    max_depth=max_depth if max_depth else 10,
                    random_state=42
                )

                start = time.time()
                scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
                elapsed = time.time() - start

                # Model complexity proxy: n_estimators * max_depth
                depth = max_depth if max_depth else 10
                complexity = n_est * depth

                configs.append({
                    "n_estimators": n_est,
                    "max_depth": max_depth,
                    "learning_rate": lr,
                    "accuracy": scores.mean(),
                    "time": elapsed,
                    "complexity": complexity,
                })

    # Find Pareto front (maximize accuracy, minimize time and complexity)
    def is_dominated(a, b):
        """Returns True if configuration b dominates a."""
        return (b["accuracy"] >= a["accuracy"] and
                b["time"] <= a["time"] and
                b["complexity"] <= a["complexity"] and
                (b["accuracy"] > a["accuracy"] or
                 b["time"] < a["time"] or
                 b["complexity"] < a["complexity"]))

    pareto = []
    for c in configs:
        if not any(is_dominated(c, other) for other in configs if other is not c):
            pareto.append(c)

    print(f"Total configurations evaluated: {len(configs)}")
    print(f"Pareto-optimal configurations: {len(pareto)}")

    print(f"\nPareto front:")
    print(f"{'Accuracy':>10} {'Time (s)':>10} {'Complexity':>12} "
          f"{'n_est':>6} {'depth':>6} {'lr':>6}")
    print("-" * 58)
    for c in sorted(pareto, key=lambda x: -x["accuracy"])[:10]:
        print(f"{c['accuracy']:>10.4f} {c['time']:>10.2f} {c['complexity']:>12d} "
              f"{c['n_estimators']:>6d} {str(c['max_depth']):>6} "
              f"{c['learning_rate']:>6.2f}")

    # Recommend a balanced choice
    # Normalize objectives to [0,1] and find the config closest to ideal
    all_acc = [c["accuracy"] for c in pareto]
    all_time = [c["time"] for c in pareto]
    all_comp = [c["complexity"] for c in pareto]

    if len(pareto) > 1:
        best_balanced = min(pareto, key=lambda c: (
            -(c["accuracy"] - min(all_acc)) / (max(all_acc) - min(all_acc) + 1e-8) +
            (c["time"] - min(all_time)) / (max(all_time) - min(all_time) + 1e-8) +
            (c["complexity"] - min(all_comp)) / (max(all_comp) - min(all_comp) + 1e-8)
        ))

        print(f"\nRecommended balanced config:")
        print(f"  n_estimators={best_balanced['n_estimators']}, "
              f"max_depth={best_balanced['max_depth']}, "
              f"lr={best_balanced['learning_rate']}")
        print(f"  Accuracy={best_balanced['accuracy']:.4f}, "
              f"Time={best_balanced['time']:.2f}s, "
              f"Complexity={best_balanced['complexity']}")

    # Plot accuracy vs time (2D projection of Pareto front)
    plt.figure(figsize=(9, 6))
    all_acc_full = [c["accuracy"] for c in configs]
    all_time_full = [c["time"] for c in configs]
    plt.scatter(all_time_full, all_acc_full, alpha=0.3, s=20, label="All configs")

    pareto_time = [c["time"] for c in pareto]
    pareto_acc = [c["accuracy"] for c in pareto]
    plt.scatter(pareto_time, pareto_acc, color="red", s=60, zorder=5,
                label="Pareto front", edgecolors="black")

    plt.xlabel("Training Time (s)")
    plt.ylabel("CV Accuracy")
    plt.title("Multi-Objective: Accuracy vs Training Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("19_ex3_pareto.png", dpi=100)
    plt.close()
    print("\nPlot saved: 19_ex3_pareto.png")


if __name__ == "__main__":
    exercise_1_hpo_classification()
    exercise_2_model_comparison()
    exercise_3_multi_objective()
