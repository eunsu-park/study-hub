"""
MLOps — Deepchecks Integration for ML Validation
==================================================
Demonstrates:
- Train/test validation suite (distribution comparison)
- Data integrity checks (nulls, duplicates, outliers)
- Model performance checks (per-class, confusion matrix)
- Custom check creation (domain-specific validation)
- Report generation and CI/CD integration

Prerequisites: pip install deepchecks scikit-learn pandas

Run: python deepchecks_suite.py <example>
Available: integrity, train_test, model, custom, all
"""

import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Suppress noisy warnings for cleaner demo output
warnings.filterwarnings("ignore", category=FutureWarning)


# ── 1. Data Integrity Suite ─────────────────────────────────────

# Why check data integrity before training?
#   Deepchecks' integrity checks catch issues that schema validation
#   alone might miss: mixed data types within a column, string mismatch
#   patterns, feature-feature correlations, and more.

def demo_data_integrity():
    """Demonstrate Deepchecks data integrity checks."""
    print("=" * 60)
    print("1. DATA INTEGRITY SUITE")
    print("=" * 60)

    try:
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.suites import data_integrity
    except ImportError:
        print("\n[SKIP] deepchecks not installed. Install with:")
        print("  pip install deepchecks")
        _show_integrity_concept()
        return

    # Create sample dataset with intentional issues
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "age": np.random.normal(40, 15, n),
        "income": np.random.exponential(50000, n),
        "score": np.random.uniform(0, 100, n),
        "category": np.random.choice(["A", "B", "C", None], n, p=[0.4, 0.3, 0.2, 0.1]),
        "target": np.random.binomial(1, 0.3, n),
    })

    # Inject some data quality issues for Deepchecks to catch
    df.loc[0:5, "age"] = np.nan           # Missing values
    df.loc[10:12, "income"] = -1000        # Negative income (invalid)
    df = pd.concat([df, df.iloc[:3]])      # Duplicate rows

    # Wrap in Deepchecks Dataset object
    # Why Dataset wrapper? It provides metadata (feature types, label)
    # that Deepchecks uses to select appropriate checks
    ds = Dataset(df, label="target", cat_features=["category"])

    # Run the built-in data integrity suite
    # Why a suite instead of individual checks?
    #   Suites run a curated set of checks at once, ensuring you don't
    #   forget any common issue. You can always add/remove checks.
    suite = data_integrity()
    result = suite.run(ds)

    # Show results summary
    print("\nData Integrity Results:")
    print(f"  Total checks: {len(result.results)}")
    passed = sum(1 for r in result.results if r.passed_conditions())
    failed = len(result.results) - passed
    print(f"  Passed: {passed}")
    print(f"  Failed/Warning: {failed}")

    # Print failing checks
    for check_result in result.results:
        if not check_result.passed_conditions():
            print(f"\n  ✗ {check_result.header}:")
            for cond in check_result.conditions_results:
                if not cond.is_pass():
                    print(f"    {cond.details}")


def _show_integrity_concept():
    """Show the concept without deepchecks installed."""
    print("\nConcept: Data Integrity Checks")
    print("  Deepchecks runs these checks automatically:")
    print("  - Mixed Data Types: columns with inconsistent types")
    print("  - Special Characters: unexpected characters in strings")
    print("  - Null Ratio: percentage of missing values per column")
    print("  - Duplicate Rows: exact duplicate samples")
    print("  - Outlier Detection: statistical outliers per feature")
    print("  - Feature-Feature Correlation: redundant features")
    print("  - String Mismatch: similar but not identical strings")
    print("  - Conflicting Labels: same features → different labels")


# ── 2. Train/Test Validation Suite ──────────────────────────────

# Why validate train vs test distributions?
#   If train and test distributions differ significantly, test metrics
#   are unreliable — they measure a different population than what the
#   model learned. This is a form of data leakage (temporal or sampling).

def demo_train_test_validation():
    """Demonstrate train/test distribution comparison."""
    print("\n" + "=" * 60)
    print("2. TRAIN/TEST VALIDATION SUITE")
    print("=" * 60)

    try:
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.suites import train_test_validation
    except ImportError:
        print("\n[SKIP] deepchecks not installed.")
        _show_train_test_concept()
        return

    # Create dataset with intentional train/test skew
    np.random.seed(42)
    n_train, n_test = 800, 200

    # Training data: younger population
    train_df = pd.DataFrame({
        "age": np.random.normal(30, 8, n_train),
        "income": np.random.normal(50000, 15000, n_train),
        "experience": np.random.uniform(0, 15, n_train),
        "target": np.random.binomial(1, 0.4, n_train),
    })

    # Test data: older population (distribution shift!)
    # Why intentional skew? To demonstrate Deepchecks' drift detection
    test_df = pd.DataFrame({
        "age": np.random.normal(50, 10, n_test),          # Shifted!
        "income": np.random.normal(70000, 20000, n_test),  # Shifted!
        "experience": np.random.uniform(5, 30, n_test),
        "target": np.random.binomial(1, 0.6, n_test),
    })

    train_ds = Dataset(train_df, label="target")
    test_ds = Dataset(test_df, label="target")

    # Run train/test validation suite
    suite = train_test_validation()
    result = suite.run(train_ds, test_ds)

    print("\nTrain/Test Validation Results:")
    print(f"  Total checks: {len(result.results)}")
    for check_result in result.results:
        status = "✓" if check_result.passed_conditions() else "✗"
        print(f"  {status} {check_result.header}")
        if not check_result.passed_conditions():
            for cond in check_result.conditions_results:
                if not cond.is_pass():
                    print(f"      {cond.details}")


def _show_train_test_concept():
    """Show the concept without deepchecks installed."""
    print("\nConcept: Train/Test Validation")
    print("  Deepchecks compares train and test distributions:")
    print("  - Feature Drift: KS test, PSI for each feature")
    print("  - Label Drift: target distribution comparison")
    print("  - Index Leakage: shared indices (data leakage)")
    print("  - Train/Test Size Ratio: adequate test set size")
    print("  - Feature-Label Correlation: consistency check")
    print("  - New Category: categories in test not in train")


# ── 3. Model Performance Suite ───────────────────────────────────

# Why use Deepchecks for model evaluation instead of just sklearn metrics?
#   Deepchecks provides structured analysis: per-class breakdown,
#   confusion matrix, calibration curves, segment performance, and
#   more — all in a single report with pass/fail conditions.

def demo_model_performance():
    """Demonstrate model performance validation with Deepchecks."""
    print("\n" + "=" * 60)
    print("3. MODEL PERFORMANCE SUITE")
    print("=" * 60)

    try:
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.suites import model_evaluation
    except ImportError:
        print("\n[SKIP] deepchecks not installed.")
        _show_model_concept()
        return

    # Train a model
    np.random.seed(42)
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=7,
        n_classes=2, random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(10)]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Wrap in Deepchecks Dataset
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df["target"] = y_train
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["target"] = y_test

    train_ds = Dataset(train_df, label="target")
    test_ds = Dataset(test_df, label="target")

    # Run model evaluation suite
    # Why model evaluation as a suite?
    #   Catches overfitting, class imbalance performance gaps,
    #   calibration issues, and feature importance instability
    suite = model_evaluation()
    result = suite.run(train_ds, test_ds, model)

    print("\nModel Evaluation Results:")
    for check_result in result.results:
        status = "✓" if check_result.passed_conditions() else "✗"
        print(f"  {status} {check_result.header}")


def _show_model_concept():
    """Show the concept without deepchecks installed."""
    print("\nConcept: Model Performance Checks")
    print("  Deepchecks evaluates the trained model:")
    print("  - Performance Report: accuracy, F1, precision, recall")
    print("  - Confusion Matrix: per-class error analysis")
    print("  - ROC / AUC: threshold-independent quality")
    print("  - Calibration: predicted probabilities vs actual rates")
    print("  - Overfitting: train vs test performance gap")
    print("  - Weak Segments: cohorts with poor performance")
    print("  - Feature Importance: which features drive predictions")


# ── 4. Custom Checks ────────────────────────────────────────────

# Why create custom checks?
#   Built-in checks cover common issues, but domain-specific problems
#   need domain-specific tests. Custom checks integrate with Deepchecks'
#   reporting and condition system.

def demo_custom_checks():
    """Demonstrate creating custom Deepchecks checks."""
    print("\n" + "=" * 60)
    print("4. CUSTOM CHECKS")
    print("=" * 60)

    try:
        from deepchecks.tabular import Dataset, Suite
        from deepchecks.tabular.base import SingleDatasetCheck
        from deepchecks.core import CheckResult, ConditionResult, ConditionCategory
    except ImportError:
        print("\n[SKIP] deepchecks not installed.")
        _show_custom_concept()
        return

    # Custom Check 1: Target Leakage Detection
    # Why a custom check for leakage?
    #   Deepchecks has built-in leakage checks, but you might need
    #   domain-specific ones (e.g., features derived from the target)
    class HighCorrelationWithTarget(SingleDatasetCheck):
        """Detect features with suspiciously high correlation to the target.

        A feature with > 0.95 correlation to the target likely encodes
        the answer (target leakage), giving artificially high accuracy
        that won't hold in production.
        """

        def __init__(self, threshold=0.95, **kwargs):
            super().__init__(**kwargs)
            self.threshold = threshold

        def run_logic(self, context, dataset_kind=None):
            df = context.dataset.data
            label = context.dataset.label_name
            leaky = {}

            for col in df.select_dtypes(include=[np.number]).columns:
                if col == label:
                    continue
                corr = abs(df[col].corr(df[label]))
                if corr > self.threshold:
                    leaky[col] = round(corr, 4)

            return CheckResult(
                value=leaky,
                display=[
                    f"Found {len(leaky)} potentially leaky features"
                    if leaky else "No leaky features detected"
                ],
            )

    # Custom Check 2: Class Balance Check
    class MinimumClassRepresentation(SingleDatasetCheck):
        """Ensure each class has at least N samples.

        Why minimum samples per class?
          With too few samples, per-class metrics are unreliable.
          Stratified splitting helps, but the root cause is data collection.
        """

        def __init__(self, min_samples=50, **kwargs):
            super().__init__(**kwargs)
            self.min_samples = min_samples

        def run_logic(self, context, dataset_kind=None):
            label = context.dataset.label_name
            counts = context.dataset.data[label].value_counts()
            below_min = {str(k): int(v) for k, v in counts.items()
                         if v < self.min_samples}

            return CheckResult(
                value={"below_minimum": below_min, "counts": counts.to_dict()},
                display=[
                    f"Classes below {self.min_samples} samples: {below_min}"
                    if below_min else "All classes have sufficient samples"
                ],
            )

    # Create and run a custom suite
    np.random.seed(42)
    df = pd.DataFrame({
        "feature_a": np.random.normal(0, 1, 300),
        "feature_b": np.random.normal(5, 2, 300),
        "leaky_feature": np.random.normal(0, 1, 300),  # Will correlate with target
        "target": np.random.binomial(1, 0.3, 300),
    })
    # Make leaky_feature highly correlated with target
    df["leaky_feature"] = df["target"] * 0.98 + np.random.normal(0, 0.02, 300)

    ds = Dataset(df, label="target")

    custom_suite = Suite(
        "Custom ML Validation",
        HighCorrelationWithTarget(threshold=0.9),
        MinimumClassRepresentation(min_samples=50),
    )

    result = custom_suite.run(ds)
    print("\nCustom Suite Results:")
    for check_result in result.results:
        print(f"  {check_result.header}: {check_result.value}")


def _show_custom_concept():
    """Show the concept without deepchecks installed."""
    print("\nConcept: Custom Deepchecks")
    print("  You can create domain-specific checks by subclassing:")
    print("  - SingleDatasetCheck: validates one dataset")
    print("  - TrainTestCheck: compares train vs test")
    print("  - ModelOnlyCheck: validates model properties")
    print()
    print("  Each custom check:")
    print("  1. Inherits from a base check class")
    print("  2. Implements run_logic() → returns CheckResult")
    print("  3. Can add conditions (pass/fail thresholds)")
    print("  4. Integrates with Deepchecks Suite and HTML reports")
    print()
    print("  Example custom checks:")
    print("  - Target leakage detection (high correlation)")
    print("  - Business rule validation (domain constraints)")
    print("  - Temporal consistency (time-series ordering)")
    print("  - Label noise estimation (confident learning)")


# ── Main ─────────────────────────────────────────────────────────

DEMOS = {
    "integrity": demo_data_integrity,
    "train_test": demo_train_test_validation,
    "model": demo_model_performance,
    "custom": demo_custom_checks,
}


def main():
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"

    if choice == "all":
        for fn in DEMOS.values():
            fn()
    elif choice in DEMOS:
        DEMOS[choice]()
    else:
        print(f"Unknown: {choice}")
        print(f"Available: {', '.join(DEMOS.keys())}, all")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Deepchecks demo completed.")


if __name__ == "__main__":
    main()
