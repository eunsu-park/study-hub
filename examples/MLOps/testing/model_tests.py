"""
MLOps — Comprehensive ML Model Testing
========================================
Demonstrates:
- Data validation with schema checks (pytest-style)
- Model quality assertions (accuracy, F1 thresholds)
- Behavioral tests (invariance, directional)
- Fairness metric checks (disparate impact, equalized odds)
- Infrastructure tests (latency, throughput)
- Integration with CI/CD pipelines

Run: python model_tests.py <example>
Available: data, quality, behavioral, fairness, infra, all
"""

import sys
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# ── 1. Data Validation Tests ────────────────────────────────────

# Why validate data before training?
#   Garbage in → garbage out. Data bugs are the #1 cause of ML failures.
#   Schema validation catches upstream changes before they corrupt models.

@dataclass
class ColumnRule:
    """Defines expected properties for one column."""
    name: str
    dtype: str
    nullable: bool = False
    min_value: float = None
    max_value: float = None
    allowed_values: list = field(default_factory=list)


class DataValidator:
    """Validates a DataFrame against a set of column rules.

    Why return all errors instead of failing on the first one?
      - Developers can fix multiple issues in a single iteration
      - Gives a complete picture of data health
    """

    def __init__(self, rules: list[ColumnRule], min_rows: int = 1):
        self.rules = {r.name: r for r in rules}
        self.min_rows = min_rows

    def validate(self, df: pd.DataFrame) -> list[str]:
        errors = []

        # Row count check
        if len(df) < self.min_rows:
            errors.append(f"Row count {len(df)} below minimum {self.min_rows}")

        # Missing columns — model expects these features to exist
        expected = set(self.rules.keys())
        actual = set(df.columns)
        if missing := expected - actual:
            errors.append(f"Missing columns: {missing}")
        if extra := actual - expected:
            errors.append(f"Unexpected columns (schema change?): {extra}")

        # Per-column checks
        for name, rule in self.rules.items():
            if name not in df.columns:
                continue
            col = df[name]

            # Null check — unexpected nulls propagate NaN through the model
            if not rule.nullable and col.isnull().any():
                errors.append(f"'{name}': {col.isnull().sum()} unexpected nulls")

            # Range check — catches sensor errors, unit changes, corruption
            if rule.min_value is not None and (col < rule.min_value).any():
                errors.append(f"'{name}': values below minimum {rule.min_value}")
            if rule.max_value is not None and (col > rule.max_value).any():
                errors.append(f"'{name}': values above maximum {rule.max_value}")

            # Categorical domain — new categories require retraining
            if rule.allowed_values:
                invalid = set(col.dropna().unique()) - set(rule.allowed_values)
                if invalid:
                    errors.append(f"'{name}': unknown categories {invalid}")

        return errors


def demo_data_validation():
    """Demonstrate data validation with intentional errors."""
    print("=" * 60)
    print("1. DATA VALIDATION TESTS")
    print("=" * 60)

    rules = [
        ColumnRule("age", "float64", min_value=0, max_value=120),
        ColumnRule("income", "float64", min_value=0),
        ColumnRule("gender", "object", allowed_values=["M", "F", "Other"]),
        ColumnRule("target", "int64", nullable=False),
    ]
    validator = DataValidator(rules, min_rows=10)

    # Good data
    good_df = pd.DataFrame({
        "age": [25.0, 35.0, 45.0, 55.0, 30.0, 40.0, 50.0, 60.0, 28.0, 38.0],
        "income": [50000.0, 70000.0, 90000.0, 60000.0, 45000.0,
                   80000.0, 95000.0, 55000.0, 48000.0, 72000.0],
        "gender": ["M", "F", "M", "F", "Other", "M", "F", "M", "F", "Other"],
        "target": [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    })
    errors = validator.validate(good_df)
    print(f"\nGood data errors: {len(errors)}")
    assert len(errors) == 0, "Good data should pass"
    print("  ✓ All checks passed")

    # Bad data — intentional errors to demonstrate detection
    bad_df = pd.DataFrame({
        "age": [25.0, -5.0, 200.0, None, 30.0],   # Negative age, > 120, null
        "income": [50000.0, -1000.0, 70000.0, 80000.0, 90000.0],  # Negative income
        "gender": ["M", "F", "Unknown", "X", "M"],  # Unknown categories
        "target": [0, 1, None, 0, 1],                # Null in non-nullable
    })
    errors = validator.validate(bad_df)
    print(f"\nBad data errors: {len(errors)}")
    for err in errors:
        print(f"  ✗ {err}")
    assert len(errors) > 0, "Bad data should fail validation"


# ── 2. Model Quality Tests ──────────────────────────────────────

# Why test model quality with thresholds instead of "best effort"?
#   Production models must meet SLAs. A model that scores 60% accuracy
#   should never be deployed to a system expecting 85%+.

class ModelQualityTester:
    """Tests a trained model against minimum quality thresholds."""

    def __init__(self, model, X_test, y_test, thresholds: dict = None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.thresholds = thresholds or {
            "accuracy": 0.80,
            "f1": 0.75,
            "precision": 0.70,
            "recall": 0.65,
        }

    def run(self) -> dict:
        y_pred = self.model.predict(self.X_test)
        results = {}

        metric_fns = {
            "accuracy": lambda: accuracy_score(self.y_test, y_pred),
            "f1": lambda: f1_score(self.y_test, y_pred, average="weighted"),
            "precision": lambda: precision_score(self.y_test, y_pred, average="weighted"),
            "recall": lambda: recall_score(self.y_test, y_pred, average="weighted"),
        }

        for name, fn in metric_fns.items():
            value = fn()
            threshold = self.thresholds.get(name, 0)
            results[name] = {
                "value": round(value, 4),
                "threshold": threshold,
                "passed": value >= threshold,
            }

        return results

    def regression_test(self, baseline: dict, margin: float = 0.02) -> dict:
        """Check that the model doesn't regress from a known baseline.

        Why allow a margin?
          Small fluctuations are normal due to randomness in evaluation.
          A 2% margin prevents flaky tests while catching real degradation.
        """
        current = self.run()
        regressions = {}
        for metric, baseline_val in baseline.items():
            if metric in current:
                cur_val = current[metric]["value"]
                regressions[metric] = {
                    "baseline": baseline_val,
                    "current": cur_val,
                    "delta": round(cur_val - baseline_val, 4),
                    "regressed": cur_val < (baseline_val - margin),
                }
        return regressions


def demo_model_quality():
    """Demonstrate model quality testing."""
    print("\n" + "=" * 60)
    print("2. MODEL QUALITY TESTS")
    print("=" * 60)

    # Train a simple model
    X, y = make_classification(n_samples=1000, n_features=20,
                                n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Quality tests
    tester = ModelQualityTester(model, X_test, y_test)
    results = tester.run()

    print("\nModel Quality Results:")
    all_passed = True
    for metric, data in results.items():
        status = "✓ PASS" if data["passed"] else "✗ FAIL"
        print(f"  {status} {metric}: {data['value']:.4f} (threshold: {data['threshold']})")
        if not data["passed"]:
            all_passed = False

    # Regression test against a fake baseline
    baseline = {"accuracy": 0.90, "f1": 0.88}
    regressions = tester.regression_test(baseline)
    print("\nRegression Test vs Baseline:")
    for metric, data in regressions.items():
        status = "✗ REGRESSED" if data["regressed"] else "✓ OK"
        print(f"  {status} {metric}: {data['current']:.4f} vs baseline {data['baseline']:.4f} "
              f"(delta: {data['delta']:+.4f})")


# ── 3. Behavioral Tests ─────────────────────────────────────────

# Why behavioral tests?
#   A model with 95% accuracy might still fail on basic cases.
#   Behavioral tests check SPECIFIC capabilities, not just aggregate metrics.

class BehavioralTester:
    """Behavioral test suite for ML classifiers.

    Three test types (Ribeiro et al., 2020):
      - MFT (Minimum Functionality Test): obvious cases must pass
      - INV (Invariance): label-preserving changes must not flip prediction
      - DIR (Directional): specific changes should move prediction direction
    """

    def __init__(self, model):
        self.model = model

    def minimum_functionality(self, test_cases: list[tuple]) -> dict:
        """Test that model handles clear-cut cases correctly.

        test_cases: [(input_features, expected_label), ...]
        """
        passed = 0
        failures = []
        for features, expected in test_cases:
            pred = self.model.predict([features])[0]
            if pred == expected:
                passed += 1
            else:
                failures.append({"expected": expected, "got": pred})

        return {
            "type": "MFT",
            "total": len(test_cases),
            "passed": passed,
            "failure_rate": round(1 - passed / max(len(test_cases), 1), 4),
            "failures": failures[:5],
        }

    def invariance(self, base_inputs: list, perturb_fn: callable) -> dict:
        """Test that small perturbations don't change predictions.

        Why invariance matters:
          If adding noise to a feature changes the prediction,
          the model is fragile and will fail on slightly different data.
        """
        violations = 0
        for x in base_inputs:
            x_perturbed = perturb_fn(x)
            if self.model.predict([x])[0] != self.model.predict([x_perturbed])[0]:
                violations += 1

        return {
            "type": "INV",
            "total": len(base_inputs),
            "violations": violations,
            "violation_rate": round(violations / max(len(base_inputs), 1), 4),
        }

    def directional(self, base_inputs, perturbed_inputs,
                     expected_directions) -> dict:
        """Test that specific changes move prediction probability in expected direction.

        expected_directions: list of 'increase' or 'decrease'
        """
        if not hasattr(self.model, "predict_proba"):
            return {"type": "DIR", "error": "Model lacks predict_proba"}

        violations = 0
        for base, pert, direction in zip(base_inputs, perturbed_inputs,
                                          expected_directions):
            base_prob = self.model.predict_proba([base])[0][1]
            pert_prob = self.model.predict_proba([pert])[0][1]

            if direction == "increase" and pert_prob <= base_prob:
                violations += 1
            elif direction == "decrease" and pert_prob >= base_prob:
                violations += 1

        return {
            "type": "DIR",
            "total": len(base_inputs),
            "violations": violations,
            "violation_rate": round(violations / max(len(base_inputs), 1), 4),
        }


def demo_behavioral():
    """Demonstrate behavioral testing."""
    print("\n" + "=" * 60)
    print("3. BEHAVIORAL TESTS")
    print("=" * 60)

    X, y = make_classification(n_samples=500, n_features=5,
                                n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    tester = BehavioralTester(model)

    # MFT: create obvious positive/negative cases
    extreme_pos = [3.0, 3.0, 3.0, 3.0, 3.0]
    extreme_neg = [-3.0, -3.0, -3.0, -3.0, -3.0]
    mft_cases = [(extreme_pos, 1), (extreme_neg, 0)]
    mft_result = tester.minimum_functionality(mft_cases)
    print(f"\nMFT: {mft_result['passed']}/{mft_result['total']} passed "
          f"(failure rate: {mft_result['failure_rate']:.2%})")

    # INV: small noise should not change prediction
    base_samples = X_test[:20].tolist()
    def add_tiny_noise(x):
        return [xi + np.random.normal(0, 0.01) for xi in x]

    inv_result = tester.invariance(base_samples, add_tiny_noise)
    print(f"INV: {inv_result['violations']}/{inv_result['total']} violations "
          f"(rate: {inv_result['violation_rate']:.2%})")


# ── 4. Fairness Tests ────────────────────────────────────────────

# Why fairness testing is non-optional:
#   Models trained on historical data encode historical biases.
#   Without explicit testing, you deploy discrimination at scale.

def fairness_audit(y_true, y_pred, sensitive_attr, threshold=0.8) -> dict:
    """Compute fairness metrics across groups of a sensitive attribute.

    Uses the four-fifths rule (EEOC guideline):
      If the positive prediction rate for any group is less than
      80% of the highest group's rate → evidence of adverse impact.

    Disparate Impact Ratio = min_rate / max_rate
    If DIR < 0.8 → potential discrimination
    """
    groups = np.unique(sensitive_attr)
    group_metrics = {}

    for group in groups:
        mask = sensitive_attr == group
        y_t = y_true[mask]
        y_p = y_pred[mask]

        positive_rate = y_p.mean()
        tpr = y_p[y_t == 1].mean() if (y_t == 1).any() else 0.0
        fpr = y_p[y_t == 0].mean() if (y_t == 0).any() else 0.0

        group_metrics[str(group)] = {
            "n": int(mask.sum()),
            "positive_rate": round(float(positive_rate), 4),
            "tpr": round(float(tpr), 4),
            "fpr": round(float(fpr), 4),
        }

    rates = [m["positive_rate"] for m in group_metrics.values()]
    dir_value = min(rates) / max(rates) if max(rates) > 0 else 0.0

    tprs = [m["tpr"] for m in group_metrics.values()]
    fprs = [m["fpr"] for m in group_metrics.values()]

    return {
        "groups": group_metrics,
        "disparate_impact_ratio": round(dir_value, 4),
        "four_fifths_passed": dir_value >= threshold,
        "tpr_gap": round(max(tprs) - min(tprs), 4),
        "fpr_gap": round(max(fprs) - min(fprs), 4),
        "equalized_odds_passed": (
            (max(tprs) - min(tprs)) < 0.1 and (max(fprs) - min(fprs)) < 0.1
        ),
    }


def demo_fairness():
    """Demonstrate fairness testing."""
    print("\n" + "=" * 60)
    print("4. FAIRNESS TESTS")
    print("=" * 60)

    np.random.seed(42)
    n = 500
    # Simulate a biased dataset where group "B" has less positive outcomes
    sensitive = np.random.choice(["A", "B"], size=n, p=[0.6, 0.4])
    y_true = np.where(sensitive == "A",
                      np.random.binomial(1, 0.7, n),
                      np.random.binomial(1, 0.4, n))
    y_pred = np.where(sensitive == "A",
                      np.random.binomial(1, 0.65, n),
                      np.random.binomial(1, 0.35, n))

    result = fairness_audit(y_true, y_pred, sensitive)

    print("\nGroup Metrics:")
    for group, metrics in result["groups"].items():
        print(f"  Group {group}: n={metrics['n']}, "
              f"positive_rate={metrics['positive_rate']:.4f}, "
              f"TPR={metrics['tpr']:.4f}, FPR={metrics['fpr']:.4f}")

    print(f"\nDisparate Impact Ratio: {result['disparate_impact_ratio']:.4f}")
    print(f"Four-Fifths Rule: {'✓ PASS' if result['four_fifths_passed'] else '✗ FAIL'}")
    print(f"TPR Gap: {result['tpr_gap']:.4f}")
    print(f"FPR Gap: {result['fpr_gap']:.4f}")
    print(f"Equalized Odds: {'✓ PASS' if result['equalized_odds_passed'] else '✗ FAIL'}")


# ── 5. Infrastructure Tests ──────────────────────────────────────

# Why test infrastructure?
#   A 99% accurate model with 10-second latency is useless for real-time.
#   Infrastructure tests ensure non-functional requirements are met.

class InfrastructureTester:
    """Test model serving performance characteristics."""

    def __init__(self, model, sample_input):
        self.model = model
        self.sample = sample_input

    def latency_test(self, n_iter=200, max_p50_ms=50, max_p99_ms=200) -> dict:
        """Measure prediction latency distribution.

        Why p50 AND p99?
          p50 shows typical experience; p99 shows worst-case.
          Tail latency (p99) often matters more for user-facing systems.
        """
        # Warm up — first calls are slower due to JIT, caching
        for _ in range(20):
            self.model.predict(self.sample)

        latencies = []
        for _ in range(n_iter):
            start = time.perf_counter()
            self.model.predict(self.sample)
            latencies.append((time.perf_counter() - start) * 1000)

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p99 = latencies[int(len(latencies) * 0.99)]

        return {
            "p50_ms": round(p50, 3),
            "p99_ms": round(p99, 3),
            "mean_ms": round(np.mean(latencies), 3),
            "p50_passed": p50 <= max_p50_ms,
            "p99_passed": p99 <= max_p99_ms,
        }

    def throughput_test(self, duration_sec=3, min_rps=100) -> dict:
        """Measure prediction throughput (requests per second).

        Why throughput matters:
          If you expect 1000 requests/second in production,
          the model must handle at least that on a single instance
          (or you need horizontal scaling).
        """
        count = 0
        start = time.perf_counter()
        while time.perf_counter() - start < duration_sec:
            self.model.predict(self.sample)
            count += 1
        elapsed = time.perf_counter() - start
        rps = count / elapsed

        return {
            "rps": round(rps, 1),
            "total_predictions": count,
            "duration_sec": round(elapsed, 2),
            "min_rps": min_rps,
            "passed": rps >= min_rps,
        }


def demo_infra():
    """Demonstrate infrastructure testing."""
    print("\n" + "=" * 60)
    print("5. INFRASTRUCTURE TESTS")
    print("=" * 60)

    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)

    tester = InfrastructureTester(model, X[:1])

    lat = tester.latency_test()
    print(f"\nLatency: p50={lat['p50_ms']:.3f}ms, p99={lat['p99_ms']:.3f}ms")
    print(f"  p50: {'✓ PASS' if lat['p50_passed'] else '✗ FAIL'} (max 50ms)")
    print(f"  p99: {'✓ PASS' if lat['p99_passed'] else '✗ FAIL'} (max 200ms)")

    tp = tester.throughput_test()
    print(f"\nThroughput: {tp['rps']:.1f} req/sec ({tp['total_predictions']} in {tp['duration_sec']}s)")
    print(f"  {'✓ PASS' if tp['passed'] else '✗ FAIL'} (min {tp['min_rps']} req/sec)")


# ── Main ─────────────────────────────────────────────────────────

DEMOS = {
    "data": demo_data_validation,
    "quality": demo_model_quality,
    "behavioral": demo_behavioral,
    "fairness": demo_fairness,
    "infra": demo_infra,
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
    print("All selected tests completed.")


if __name__ == "__main__":
    main()
