[← Previous: 15. LLMOps](15_LLMOps.md) | [Next: 17. Cloud MLOps Platforms →](17_Cloud_MLOps_Platforms.md)

# Model Testing and Validation

## Learning Objectives

1. Understand why ML model testing fundamentally differs from traditional software testing
2. Implement the four categories of ML tests: data validation, model quality, behavioral, and infrastructure
3. Use testing frameworks (pytest, Great Expectations, Deepchecks) for automated ML validation
4. Design pre-training data quality gates that prevent garbage-in-garbage-out failures
5. Build post-training validation pipelines with performance benchmarks and fairness metrics
6. Apply shadow deployment and canary release strategies to safely roll out models
7. Set up continuous validation loops that detect production degradation and trigger retraining

---

## Overview

In traditional software engineering, a function either returns the correct result or it doesn't — testing is deterministic. Machine learning breaks this assumption. A model can pass every unit test and still fail catastrophically in production because the data shifted, a subgroup was underrepresented, or a subtle feature interaction changed. Model testing and validation is the discipline of building systematic safeguards across the entire ML lifecycle — from the moment raw data enters your pipeline to the day your model serves its millionth prediction.

This lesson covers the full spectrum of ML testing: what to test, when to test it, and how to automate it. We will build up from data validation through model quality checks, behavioral testing, fairness audits, shadow deployments, and continuous production monitoring. By the end, you will be able to design a testing strategy that catches failures before they reach users.

> **Analogy**: Testing an ML model is like quality control in a factory. You inspect raw materials before they enter the production line (data validation), test the assembly process itself (training validation), inspect finished products against specifications (model evaluation), and monitor field performance through customer reports (production monitoring). Skipping any stage means defective products reach customers — and in ML, "defective" can mean biased, inaccurate, or dangerously wrong predictions.

---

## 1. Why ML Testing Differs from Software Testing

### 1.1 The Fundamental Challenge

```python
"""
Software Testing vs ML Testing:

  Traditional Software:
    Input → Deterministic Function → Expected Output
    assert add(2, 3) == 5  # Always true or always false

  Machine Learning:
    Input → Learned Function (statistical) → Probabilistic Output
    assert model.predict(x) == y  # Might be true 95% of the time

  Key Differences:
  ┌────────────────────┬──────────────────────┬──────────────────────────┐
  │ Dimension          │ Software Testing     │ ML Testing               │
  ├────────────────────┼──────────────────────┼──────────────────────────┤
  │ Correctness        │ Binary (pass/fail)   │ Statistical (thresholds) │
  │ Inputs             │ Known, finite        │ High-dimensional, noisy  │
  │ Behavior           │ Deterministic        │ Stochastic               │
  │ Failure modes      │ Crashes, wrong output│ Subtle degradation       │
  │ Test oracle        │ Expected output      │ Often unavailable        │
  │ Environment dep.   │ Minimal              │ Data distribution        │
  │ Regression         │ Code changes         │ Code, data, OR world     │
  └────────────────────┴──────────────────────┴──────────────────────────┘

  The 3 axes of ML change (any can cause failure):
    1. Code changes    → traditional CI catches these
    2. Data changes    → need data validation
    3. World changes   → need production monitoring
"""
```

### 1.2 The ML Test Pyramid

```python
"""
ML Test Pyramid (inspired by Google's ML Test Score):

  The traditional test pyramid (unit → integration → e2e) needs expansion
  for ML systems. Here is the ML-specific test pyramid:

                    ┌─────────────┐
                    │  Production  │   Continuous monitoring,
                    │  Monitoring  │   shadow testing, A/B tests
                  ┌─┴─────────────┴─┐
                  │   Behavioral     │   Invariance, directional,
                  │   Tests          │   minimum functionality
                ┌─┴─────────────────┴─┐
                │   Model Quality      │   Accuracy, F1, AUC thresholds,
                │   Tests              │   regression vs baseline
              ┌─┴─────────────────────┴─┐
              │   Training Pipeline      │   Reproducibility, convergence,
              │   Tests                  │   resource usage
            ┌─┴─────────────────────────┴─┐
            │   Data Validation            │   Schema, distributions,
            │   Tests                      │   missing values, anomalies
          ┌─┴─────────────────────────────┴─┐
          │   Infrastructure / Unit Tests     │   Feature engineering,
          │                                   │   preprocessing, I/O
          └───────────────────────────────────┘

  Each layer catches different failure modes.
  Lower layers run faster and should catch most issues.
"""
```

### 1.3 The Cost of Skipping Tests

```python
"""
Why invest in ML testing?

  Without testing:
    Data bug → trains on corrupted data → deploys bad model → users affected
    Time to detect: hours to weeks
    Cost: retraining + reputation + rollback

  With testing:
    Data bug → data validation catches it → pipeline halts → alert sent
    Time to detect: minutes
    Cost: pipeline pause + fix

  Real-world failures from insufficient testing:
    - Zillow's iBuying algorithm: $500M loss from model errors
    - Amazon's recruiting tool: biased against women (no fairness tests)
    - Healthcare algorithms: systematically under-served Black patients

  ML Test Score (Google, 2017):
    - 28-point rubric for ML system maturity
    - Categories: data tests, model tests, ML infra tests, monitoring
    - Score < 10: "risky"; 10-20: "needs improvement"; > 20: "mature"
"""
```

---

## 2. Data Validation Tests

Data is the foundation. If your data is wrong, no amount of model tuning will save you.

### 2.1 Schema Validation

```python
"""
Schema validation ensures data structure hasn't changed unexpectedly.

Why schema validation matters:
  - Upstream systems change column names/types without notice
  - New categories appear (e.g., a new country code)
  - Column order changes break positional indexing
  - Null handling changes silently
"""
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class ColumnSchema:
    """Define expected properties for a single column."""
    name: str
    dtype: str                          # 'int64', 'float64', 'object', etc.
    nullable: bool = False
    min_value: float = None             # For numeric columns
    max_value: float = None
    allowed_values: list = field(default_factory=list)  # For categorical


@dataclass
class DatasetSchema:
    """Full dataset schema with validation logic."""
    columns: list                       # List of ColumnSchema
    min_rows: int = 1
    max_rows: int = None

    def validate(self, df: pd.DataFrame) -> list:
        """Validate a DataFrame against this schema.

        Returns a list of error messages (empty = valid).
        Why return errors instead of raising?
          - Collect ALL issues at once for easier debugging
          - Caller decides severity (warn vs block)
        """
        errors = []

        # Check row count
        if len(df) < self.min_rows:
            errors.append(f"Too few rows: {len(df)} < {self.min_rows}")
        if self.max_rows and len(df) > self.max_rows:
            errors.append(f"Too many rows: {len(df)} > {self.max_rows}")

        # Check each column
        expected_cols = {c.name for c in self.columns}
        actual_cols = set(df.columns)

        # Missing columns are critical — model expects these features
        missing = expected_cols - actual_cols
        if missing:
            errors.append(f"Missing columns: {missing}")

        # Extra columns might indicate upstream schema change
        extra = actual_cols - expected_cols
        if extra:
            errors.append(f"Unexpected columns: {extra}")

        # Per-column validation
        for col_schema in self.columns:
            if col_schema.name not in df.columns:
                continue

            series = df[col_schema.name]

            # Type check
            if str(series.dtype) != col_schema.dtype:
                errors.append(
                    f"Column '{col_schema.name}': expected {col_schema.dtype}, "
                    f"got {series.dtype}"
                )

            # Null check — unexpected nulls can crash or bias the model
            if not col_schema.nullable and series.isnull().any():
                null_count = series.isnull().sum()
                errors.append(
                    f"Column '{col_schema.name}': {null_count} unexpected nulls"
                )

            # Range check — catches sensor errors, data corruption
            if col_schema.min_value is not None:
                below = (series < col_schema.min_value).sum()
                if below > 0:
                    errors.append(
                        f"Column '{col_schema.name}': {below} values below "
                        f"minimum {col_schema.min_value}"
                    )

            # Categorical domain check — new categories need retraining
            if col_schema.allowed_values:
                invalid = set(series.dropna().unique()) - set(col_schema.allowed_values)
                if invalid:
                    errors.append(
                        f"Column '{col_schema.name}': unexpected values {invalid}"
                    )

        return errors
```

### 2.2 Distribution Validation

```python
"""
Distribution validation detects data drift before training.

Why check distributions?
  A model trained on data from distribution P(X) will perform poorly
  if production data follows a different distribution Q(X).

  Statistically, we test the null hypothesis:
    H₀: P_train(X) = P_current(X)

  Common tests:
    - KS test (continuous): Kolmogorov-Smirnov statistic
    - Chi-squared test (categorical): χ² statistic
    - PSI (Population Stability Index): banking industry standard
      PSI = Σ (p_i - q_i) × ln(p_i / q_i)
      PSI < 0.1: no shift; 0.1-0.2: moderate; > 0.2: significant
"""
import numpy as np
from scipy import stats


def compute_psi(expected: np.ndarray, actual: np.ndarray,
                n_bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions.

    Why PSI over KS test?
      - PSI gives a single interpretable number with industry thresholds
      - KS test p-value depends on sample size (large N → always rejects)
      - PSI is symmetric and additive across features
    """
    # Bin the expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # Count proportions in each bin
    expected_counts = np.histogram(expected, breakpoints)[0]
    actual_counts = np.histogram(actual, breakpoints)[0]

    # Convert to proportions, avoid zero (add small epsilon)
    # Why epsilon? log(0) is undefined; this is standard practice
    eps = 1e-6
    expected_pct = expected_counts / len(expected) + eps
    actual_pct = actual_counts / len(actual) + eps

    # PSI formula: Σ (actual% - expected%) * ln(actual% / expected%)
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi


def validate_distributions(reference_df, current_df, numeric_cols,
                           psi_threshold=0.2, ks_alpha=0.05):
    """Validate that current data distributions match the reference.

    Why two tests?
      - PSI gives business-interpretable thresholds
      - KS test provides statistical rigor
      - Agreement between both increases confidence
    """
    results = {}
    for col in numeric_cols:
        ref = reference_df[col].dropna().values
        cur = current_df[col].dropna().values

        psi = compute_psi(ref, cur)
        ks_stat, ks_pvalue = stats.ks_2samp(ref, cur)

        results[col] = {
            "psi": round(psi, 4),
            "psi_status": (
                "OK" if psi < 0.1
                else "WARNING" if psi < psi_threshold
                else "ALERT"
            ),
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(ks_pvalue, 4),
            "ks_status": "OK" if ks_pvalue > ks_alpha else "DRIFT_DETECTED",
        }

    return results
```

### 2.3 Data Quality with Great Expectations

```python
"""
Great Expectations: declarative data validation framework.

Why Great Expectations over custom validation?
  - Declarative syntax: describe WHAT, not HOW
  - Built-in expectation library (150+ expectations)
  - Auto-documentation (data docs)
  - Integration with Airflow, Spark, pandas
  - Validation results are stored and auditable
"""
# pip install great-expectations

import great_expectations as gx


def build_data_validation_suite():
    """Build a Great Expectations validation suite for ML training data.

    This demonstrates the key concept of 'expectations' — declarative
    assertions about your data that are version-controlled and reusable.
    """
    context = gx.get_context()

    # Create a data source connected to our training data
    # Why use GX context? It manages configuration, stores, and docs
    datasource = context.data_sources.add_pandas("training_data")

    # Define expectations — each one is a testable assertion
    suite = context.add_expectation_suite("ml_training_validation")

    # Column presence — model requires these features
    suite.add_expectation(
        gx.expectations.ExpectColumnToExist(column="age")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnToExist(column="income")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnToExist(column="target")
    )

    # No nulls in critical columns — nulls cause NaN propagation
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="target")
    )

    # Value ranges — catches data corruption and unit errors
    # Why hardcode ranges? These come from domain knowledge
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="age", min_value=0, max_value=120
        )
    )

    # Uniqueness — duplicate IDs cause data leakage
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(column="user_id")
    )

    # Target distribution — imbalanced data needs special handling
    suite.add_expectation(
        gx.expectations.ExpectColumnProportionOfUniqueValuesToBeBetween(
            column="target", min_value=0.01, max_value=0.99
        )
    )

    return suite
```

---

## 3. Model Quality Tests

Once data is validated, we test the model itself.

### 3.1 Performance Threshold Tests

```python
"""
Performance threshold tests ensure the model meets minimum quality bars.

Why thresholds instead of "best effort"?
  - A deployed model must meet SLAs (Service Level Agreements)
  - Regression from baseline is unacceptable
  - Different metrics for different stakeholders:
    * Business: revenue impact, precision
    * Engineering: latency, throughput
    * Ethics: fairness, equal error rates
"""
import pytest
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, mean_absolute_error
)


class ModelQualityTests:
    """Test suite for model quality validation.

    Why a class instead of standalone functions?
      - Shared setup (model loading, data preparation)
      - Configurable thresholds per deployment environment
      - Easy to extend with new metrics
    """

    def __init__(self, model, X_test, y_test, thresholds=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        # Default thresholds — override for specific use cases
        self.thresholds = thresholds or {
            "accuracy": 0.85,
            "f1": 0.80,
            "precision": 0.75,
            "recall": 0.70,
            "auc": 0.85,
        }

    def run_all(self) -> dict:
        """Run all quality tests and return results.

        Returns dict with pass/fail for each metric.
        Why return results instead of asserting?
          - Some metrics may be "soft" (warning vs blocking)
          - Logging all results is useful for trend analysis
        """
        y_pred = self.model.predict(self.X_test)
        y_prob = (
            self.model.predict_proba(self.X_test)[:, 1]
            if hasattr(self.model, "predict_proba")
            else None
        )

        results = {}

        # Accuracy — overall correctness
        acc = accuracy_score(self.y_test, y_pred)
        results["accuracy"] = {
            "value": round(acc, 4),
            "threshold": self.thresholds["accuracy"],
            "passed": acc >= self.thresholds["accuracy"],
        }

        # F1 — harmonic mean of precision and recall
        # Why F1? It balances both types of errors
        f1 = f1_score(self.y_test, y_pred, average="weighted")
        results["f1"] = {
            "value": round(f1, 4),
            "threshold": self.thresholds["f1"],
            "passed": f1 >= self.thresholds["f1"],
        }

        # AUC — threshold-independent ranking quality
        if y_prob is not None:
            auc = roc_auc_score(self.y_test, y_prob)
            results["auc"] = {
                "value": round(auc, 4),
                "threshold": self.thresholds["auc"],
                "passed": auc >= self.thresholds["auc"],
            }

        return results

    def regression_test(self, baseline_metrics: dict) -> dict:
        """Test that the new model doesn't regress from baseline.

        Why regression testing?
          A model might meet absolute thresholds but still be worse
          than the currently deployed model. This catches that.

          Allowed regression margin (e.g., 2%) prevents flaky tests
          from random variation in evaluation data.
        """
        current = self.run_all()
        regression_margin = 0.02  # Allow 2% regression

        regressions = {}
        for metric, baseline_val in baseline_metrics.items():
            if metric in current:
                current_val = current[metric]["value"]
                regressed = current_val < (baseline_val - regression_margin)
                regressions[metric] = {
                    "baseline": baseline_val,
                    "current": current_val,
                    "regressed": regressed,
                    "delta": round(current_val - baseline_val, 4),
                }

        return regressions
```

### 3.2 Slice-Based Testing

```python
"""
Slice-based testing evaluates model performance on data subgroups.

Why slice testing?
  Overall metrics can hide poor performance on critical subgroups.

  Example:
    Overall accuracy: 92% ← looks great
    Accuracy on age > 65: 71% ← unacceptable for healthcare
    Accuracy on rare disease: 45% ← dangerous

  Mathematically, the overall metric is a weighted average:
    metric_overall = Σᵢ wᵢ × metric_sliceᵢ

  A large majority group can "drown out" poor minority performance.
"""


def slice_evaluation(model, X_test, y_test, slice_column, slice_values=None):
    """Evaluate model on each slice of a categorical column.

    Why explicit slicing?
      - Catches bias before deployment
      - Required for fairness compliance (ECOA, GDPR, etc.)
      - Identifies where to focus data collection
    """
    import pandas as pd

    if slice_values is None:
        slice_values = X_test[slice_column].unique()

    results = {}
    for value in slice_values:
        mask = X_test[slice_column] == value
        X_slice = X_test[mask]
        y_slice = y_test[mask]

        if len(y_slice) < 10:  # Too few samples for reliable metrics
            results[value] = {"status": "INSUFFICIENT_DATA", "n": len(y_slice)}
            continue

        y_pred = model.predict(X_slice)
        results[value] = {
            "n": len(y_slice),
            "accuracy": round(accuracy_score(y_slice, y_pred), 4),
            "f1": round(f1_score(y_slice, y_pred, average="weighted"), 4),
            "precision": round(precision_score(y_slice, y_pred, average="weighted"), 4),
            "recall": round(recall_score(y_slice, y_pred, average="weighted"), 4),
        }

    # Check for disparate performance
    # Why? Regulatory requirements + ethical responsibility
    accuracies = [r["accuracy"] for r in results.values()
                  if isinstance(r.get("accuracy"), float)]
    if accuracies:
        gap = max(accuracies) - min(accuracies)
        results["_summary"] = {
            "max_accuracy_gap": round(gap, 4),
            "flag": "DISPARITY" if gap > 0.15 else "OK",
        }

    return results
```

---

## 4. Behavioral Testing

### 4.1 The CheckList Framework

```python
"""
Behavioral Testing for NLP (Ribeiro et al., 2020 — "Beyond Accuracy"):

  Traditional evaluation tells us overall performance.
  Behavioral testing tells us what CAPABILITIES the model has.

  Three test types:
  ┌─────────────────────────────────────────────────────────────────┐
  │ Type          │ Tests             │ Example                     │
  ├───────────────┼───────────────────┼─────────────────────────────┤
  │ Minimum       │ Basic capability  │ "This is great" → positive  │
  │ Functionality │ (sanity checks)   │ "This is terrible" → neg    │
  │ (MFT)         │                   │                             │
  ├───────────────┼───────────────────┼─────────────────────────────┤
  │ Invariance    │ Label-preserving  │ "Great movie" → positive    │
  │ (INV)         │ perturbations     │ "Great film" → positive     │
  │               │ should NOT change │ (synonym shouldn't flip)    │
  │               │ prediction        │                             │
  ├───────────────┼───────────────────┼─────────────────────────────┤
  │ Directional   │ Perturbations     │ "Good food" → 0.8 positive  │
  │ Expectation   │ that SHOULD       │ "Good food but rude staff"  │
  │ (DIR)         │ change prediction │ → less positive (< 0.8)     │
  └───────────────┴───────────────────┴─────────────────────────────┘

  Why behavioral tests?
    - A model with 95% accuracy might still fail on basic cases
    - Reveals specific capabilities, not just aggregate performance
    - Guides targeted improvement: if INV tests fail → need augmentation
"""


class BehavioralTestSuite:
    """Behavioral test suite for classification models.

    Applicable to any classifier, not just NLP.
    Each test returns (passed, details) for reporting.
    """

    def __init__(self, model, predict_fn=None):
        """
        Args:
            model: trained model with .predict() method
            predict_fn: optional custom prediction function
                        (e.g., for pipeline with preprocessing)
        """
        self.model = model
        self.predict = predict_fn or model.predict

    def minimum_functionality_test(self, test_cases: list) -> dict:
        """Test that model handles obvious cases correctly.

        test_cases: list of (input, expected_label) tuples

        Why MFT?
          If the model can't classify "This is amazing" as positive,
          something is fundamentally wrong — no amount of accuracy
          on the test set compensates for this failure.
        """
        passed = 0
        failed_cases = []

        for x, expected in test_cases:
            pred = self.predict([x])[0]
            if pred == expected:
                passed += 1
            else:
                failed_cases.append({
                    "input": x,
                    "expected": expected,
                    "predicted": pred,
                })

        return {
            "test_type": "MFT",
            "total": len(test_cases),
            "passed": passed,
            "failure_rate": round(1 - passed / len(test_cases), 4),
            "failed_cases": failed_cases,
        }

    def invariance_test(self, base_inputs: list,
                        perturbation_fn: callable) -> dict:
        """Test that label-preserving perturbations don't change prediction.

        Why invariance tests?
          A robust model should give the same prediction for:
            - "The movie was great" vs "The film was great" (synonym)
            - "John is a good doctor" vs "Jane is a good doctor" (name)
            - Different typos, formatting, etc.
        """
        violations = 0
        violation_cases = []

        for x in base_inputs:
            x_perturbed = perturbation_fn(x)
            pred_original = self.predict([x])[0]
            pred_perturbed = self.predict([x_perturbed])[0]

            if pred_original != pred_perturbed:
                violations += 1
                violation_cases.append({
                    "original": x,
                    "perturbed": x_perturbed,
                    "pred_original": pred_original,
                    "pred_perturbed": pred_perturbed,
                })

        return {
            "test_type": "INV",
            "total": len(base_inputs),
            "violations": violations,
            "violation_rate": round(violations / len(base_inputs), 4),
            "violation_cases": violation_cases[:5],  # Limit output
        }

    def directional_expectation_test(self, test_cases: list,
                                     predict_proba_fn=None) -> dict:
        """Test that specific perturbations change prediction in expected direction.

        test_cases: list of (base_input, perturbed_input, expected_direction)
          expected_direction: 'increase' or 'decrease' for class probability

        Why directional tests?
          Adding "but the service was terrible" to a restaurant review
          SHOULD decrease the positive sentiment score. If it doesn't,
          the model isn't actually understanding sentiment.
        """
        proba_fn = predict_proba_fn or self.model.predict_proba
        violations = 0

        for base, perturbed, direction in test_cases:
            base_prob = proba_fn([base])[0][1]       # P(positive class)
            pert_prob = proba_fn([perturbed])[0][1]

            if direction == "decrease" and pert_prob >= base_prob:
                violations += 1
            elif direction == "increase" and pert_prob <= base_prob:
                violations += 1

        return {
            "test_type": "DIR",
            "total": len(test_cases),
            "violations": violations,
            "violation_rate": round(violations / len(test_cases), 4),
        }
```

---

## 5. Infrastructure Tests

### 5.1 Serving Performance Tests

```python
"""
Infrastructure tests verify non-functional requirements.

Why test infrastructure?
  A model with 99% accuracy but 10-second latency is useless for
  real-time applications. Infrastructure tests catch:
    - Latency regressions (model got too complex)
    - Memory leaks (model grows in memory over time)
    - Throughput drops (can't handle production load)
    - Model size bloat (too large for edge deployment)
"""
import time
import tracemalloc
import statistics


class InfrastructureTests:
    """Test model serving infrastructure requirements."""

    def __init__(self, model, sample_input):
        self.model = model
        self.sample_input = sample_input

    def latency_test(self, max_p50_ms=50, max_p99_ms=200,
                     n_iterations=100) -> dict:
        """Test prediction latency against SLA thresholds.

        Why p50 AND p99?
          - p50 (median): typical user experience
          - p99: worst-case experience (tail latency)
          - A model might have good median but terrible tail latency
            due to garbage collection, dynamic batching, etc.
        """
        latencies = []

        # Warm-up — first calls are slower due to JIT, caching, etc.
        for _ in range(10):
            self.model.predict(self.sample_input)

        # Measure
        for _ in range(n_iterations):
            start = time.perf_counter()
            self.model.predict(self.sample_input)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p99 = latencies[int(len(latencies) * 0.99)]

        return {
            "p50_ms": round(p50, 2),
            "p99_ms": round(p99, 2),
            "mean_ms": round(statistics.mean(latencies), 2),
            "max_ms": round(max(latencies), 2),
            "p50_passed": p50 <= max_p50_ms,
            "p99_passed": p99 <= max_p99_ms,
        }

    def memory_test(self, max_memory_mb=500) -> dict:
        """Test model memory footprint.

        Why measure memory?
          - Container memory limits in K8s (OOMKilled)
          - Edge devices have limited RAM
          - Memory leaks cause gradual degradation
        """
        tracemalloc.start()

        # Load model and make predictions
        for _ in range(100):
            self.model.predict(self.sample_input)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024

        return {
            "peak_memory_mb": round(peak_mb, 2),
            "max_allowed_mb": max_memory_mb,
            "passed": peak_mb <= max_memory_mb,
        }

    def throughput_test(self, min_rps=100, duration_seconds=5) -> dict:
        """Test prediction throughput (requests per second).

        Why throughput tests?
          - Production systems have peak load requirements
          - Ensures model can handle expected traffic
          - Identifies if batch prediction is needed vs real-time
        """
        count = 0
        start = time.perf_counter()

        while time.perf_counter() - start < duration_seconds:
            self.model.predict(self.sample_input)
            count += 1

        elapsed = time.perf_counter() - start
        rps = count / elapsed

        return {
            "requests_per_second": round(rps, 1),
            "min_required_rps": min_rps,
            "total_predictions": count,
            "duration_seconds": round(elapsed, 2),
            "passed": rps >= min_rps,
        }
```

---

## 6. Fairness Testing

### 6.1 Fairness Metrics

```python
"""
Fairness testing ensures the model doesn't discriminate against protected groups.

Why fairness testing is non-optional:
  - Legal: ECOA, GDPR, EU AI Act require non-discrimination
  - Ethical: Models amplify historical biases in training data
  - Business: Biased models damage trust and brand

Key fairness metrics:

  1. Demographic Parity (Statistical Parity):
     P(ŷ = 1 | A = a) = P(ŷ = 1 | A = b)
     "Positive prediction rate should be equal across groups"

  2. Equalized Odds:
     P(ŷ = 1 | Y = y, A = a) = P(ŷ = 1 | Y = y, A = b)  ∀ y
     "True positive and false positive rates should be equal"

  3. Predictive Parity:
     P(Y = 1 | ŷ = 1, A = a) = P(Y = 1 | ŷ = 1, A = b)
     "Precision should be equal across groups"

  Important: These metrics can be mutually exclusive!
  (Chouldechova, 2017: impossibility theorem)
  Choose based on your application's harm model.
"""


def fairness_audit(y_true, y_pred, sensitive_attr,
                   threshold=0.8) -> dict:
    """Compute fairness metrics across groups of a sensitive attribute.

    Args:
        y_true: ground truth labels
        y_pred: model predictions
        sensitive_attr: array of group membership (e.g., gender, race)
        threshold: four-fifths rule threshold (EEOC guideline)

    Why the four-fifths rule?
      The EEOC (US Equal Employment Opportunity Commission) states:
      A selection rate for any group that is less than 4/5 (80%) of
      the rate for the group with the highest rate is evidence of
      adverse impact.

      Disparate Impact Ratio = min(rate_a / rate_b, rate_b / rate_a)
      If DIR < 0.8 → potential discrimination
    """
    groups = np.unique(sensitive_attr)
    group_metrics = {}

    for group in groups:
        mask = sensitive_attr == group
        y_t = y_true[mask]
        y_p = y_pred[mask]

        n = len(y_t)
        positive_rate = y_p.mean()                       # P(ŷ=1 | A=group)
        tpr = y_p[y_t == 1].mean() if (y_t == 1).any() else 0  # True positive rate
        fpr = y_p[y_t == 0].mean() if (y_t == 0).any() else 0  # False positive rate

        group_metrics[group] = {
            "n": n,
            "positive_rate": round(positive_rate, 4),
            "true_positive_rate": round(tpr, 4),
            "false_positive_rate": round(fpr, 4),
        }

    # Compute disparate impact ratio (demographic parity)
    rates = [m["positive_rate"] for m in group_metrics.values()]
    dir_value = min(rates) / max(rates) if max(rates) > 0 else 0

    # Compute equalized odds gap
    tprs = [m["true_positive_rate"] for m in group_metrics.values()]
    fprs = [m["false_positive_rate"] for m in group_metrics.values()]

    return {
        "group_metrics": group_metrics,
        "disparate_impact_ratio": round(dir_value, 4),
        "demographic_parity_passed": dir_value >= threshold,
        "tpr_gap": round(max(tprs) - min(tprs), 4),
        "fpr_gap": round(max(fprs) - min(fprs), 4),
        "equalized_odds_passed": (max(tprs) - min(tprs)) < 0.1
                                  and (max(fprs) - min(fprs)) < 0.1,
    }
```

---

## 7. Pre-Training and Post-Training Validation

### 7.1 Pre-Training Data Quality Gates

```python
"""
Pre-training validation: quality gates that data must pass before training begins.

Why gate BEFORE training?
  Training is expensive (time, compute, carbon).
  Catching data issues before training saves:
    - Hours of GPU time
    - Debugging effort
    - Delayed deployments

  Gate Architecture:
  ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐
  │ Raw Data │──▶│ Schema    │──▶│ Distrib. │──▶│ Leakage  │──▶ Training
  │          │   │ Validation│   │ Check    │   │ Check    │
  └──────────┘   └───────────┘   └──────────┘   └──────────┘
                      ↓               ↓              ↓
                   BLOCK if        WARN if         BLOCK if
                   schema fails    drift detected  leakage found
"""


class PreTrainingGate:
    """Data quality gate that must pass before training starts.

    Why a gate pattern?
      - Clear pass/fail semantics (pipeline stops on failure)
      - Audit trail (all checks logged)
      - Configurable severity (BLOCK vs WARN)
    """

    def __init__(self, schema, reference_stats=None):
        self.schema = schema
        self.reference_stats = reference_stats
        self.checks = []

    def check_schema(self, df):
        """Validate data schema — BLOCKING."""
        errors = self.schema.validate(df)
        self.checks.append({
            "name": "schema_validation",
            "severity": "BLOCK",
            "passed": len(errors) == 0,
            "errors": errors,
        })
        return len(errors) == 0

    def check_target_leakage(self, df, target_col, feature_cols,
                             correlation_threshold=0.95):
        """Detect potential target leakage — BLOCKING.

        Why check for leakage?
          Target leakage gives artificially high training metrics
          but fails in production where the leaky feature isn't available.

          Common sources:
            - Future information (e.g., outcome timestamp < prediction time)
            - Derived features that encode the target
            - Proxy variables with near-perfect correlation
        """
        leaky_features = []
        for col in feature_cols:
            if df[col].dtype in ['int64', 'float64']:
                corr = abs(df[col].corr(df[target_col]))
                if corr > correlation_threshold:
                    leaky_features.append((col, round(corr, 4)))

        passed = len(leaky_features) == 0
        self.checks.append({
            "name": "target_leakage",
            "severity": "BLOCK",
            "passed": passed,
            "leaky_features": leaky_features,
        })
        return passed

    def check_class_balance(self, df, target_col,
                            min_minority_ratio=0.05):
        """Check for severe class imbalance — WARNING.

        Why warn instead of block?
          Imbalanced data isn't necessarily wrong — fraud detection
          is naturally imbalanced. But the team should be aware and
          use appropriate techniques (oversampling, class weights, etc.).
        """
        value_counts = df[target_col].value_counts(normalize=True)
        minority_ratio = value_counts.min()

        passed = minority_ratio >= min_minority_ratio
        self.checks.append({
            "name": "class_balance",
            "severity": "WARN",
            "passed": passed,
            "minority_ratio": round(minority_ratio, 4),
            "class_distribution": value_counts.to_dict(),
        })
        return passed

    def run_all(self, df, target_col, feature_cols) -> dict:
        """Run all pre-training checks and return gate decision."""
        self.checks = []

        self.check_schema(df)
        self.check_target_leakage(df, target_col, feature_cols)
        self.check_class_balance(df, target_col)

        # Gate decision: any BLOCK failure → reject
        blocking_failures = [
            c for c in self.checks
            if c["severity"] == "BLOCK" and not c["passed"]
        ]
        warnings = [
            c for c in self.checks
            if c["severity"] == "WARN" and not c["passed"]
        ]

        return {
            "gate_passed": len(blocking_failures) == 0,
            "blocking_failures": len(blocking_failures),
            "warnings": len(warnings),
            "checks": self.checks,
        }
```

### 7.2 Post-Training Validation

```python
"""
Post-training validation: comprehensive checks after training completes.

  This is the last gate before a model can be registered or deployed.

  Post-Training Validation Checklist:
  ┌─────────────────────────────────────────────────────────────────┐
  │ Check                          │ Blocking?  │ Threshold         │
  ├────────────────────────────────┼────────────┼───────────────────┤
  │ Performance vs baseline        │ YES        │ Within 2% margin  │
  │ Performance on slices          │ YES        │ Min 70% per slice │
  │ Fairness metrics               │ YES        │ DIR ≥ 0.8         │
  │ Behavioral tests               │ YES        │ < 5% failure rate │
  │ Latency SLA                    │ YES        │ p99 < 200ms       │
  │ Model size                     │ NO (warn)  │ < 500MB           │
  │ Feature importance stability   │ NO (warn)  │ Top-5 unchanged   │
  └────────────────────────────────┴────────────┴───────────────────┘
"""


def post_training_validation(model, X_test, y_test, baseline_metrics,
                             sensitive_attr=None):
    """Comprehensive post-training validation pipeline.

    Why run all checks together?
      - Single pass over test data (efficiency)
      - Unified report for reviewers
      - Clear go/no-go decision
    """
    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # 1. Performance threshold check
    quality = ModelQualityTests(model, X_test, y_test)
    report["quality"] = quality.run_all()

    # 2. Regression check vs baseline
    report["regression"] = quality.regression_test(baseline_metrics)

    # 3. Fairness audit (if sensitive attribute provided)
    if sensitive_attr is not None:
        y_pred = model.predict(X_test)
        report["fairness"] = fairness_audit(
            y_test.values, y_pred, sensitive_attr.values
        )

    # 4. Infrastructure checks
    sample = X_test.iloc[:1]
    infra = InfrastructureTests(model, sample)
    report["latency"] = infra.latency_test()
    report["memory"] = infra.memory_test()

    # 5. Overall gate decision
    blockers = []
    if not all(m["passed"] for m in report["quality"].values()):
        blockers.append("quality_threshold_failed")
    if any(r["regressed"] for r in report.get("regression", {}).values()):
        blockers.append("regression_detected")
    if report.get("fairness") and not report["fairness"]["demographic_parity_passed"]:
        blockers.append("fairness_violation")
    if not report["latency"]["p99_passed"]:
        blockers.append("latency_sla_violation")

    report["gate_decision"] = "APPROVED" if not blockers else "REJECTED"
    report["blockers"] = blockers

    return report
```

---

## 8. Shadow Deployment and Canary Releases

### 8.1 Shadow Deployment

```python
"""
Shadow deployment: new model receives production traffic but its
predictions are NOT served to users. Only logged for comparison.

Why shadow deployment?
  - Zero risk to users (old model still serves)
  - Tests on REAL production traffic (not just test sets)
  - Catches issues that test data can't: distribution mismatch,
    latency under load, edge cases in real inputs

  Architecture:
  ┌──────────┐    ┌───────────────────────────────────────┐
  │  Client   │───▶│           Load Balancer                │
  └──────────┘    └───────┬───────────────┬─────────────────┘
                          │               │
                    ┌─────▼─────┐   ┌─────▼─────┐
                    │ Production │   │  Shadow    │
                    │ Model v1   │   │  Model v2  │
                    │ (serves)   │   │ (logs only)│
                    └─────┬─────┘   └─────┬──────┘
                          │               │
                    User gets v1     v2 predictions
                    prediction       logged for
                                     comparison
"""


class ShadowDeployment:
    """Shadow deployment pattern for safe model comparison.

    Why implement in application code?
      In practice, this is often handled by service mesh (Istio)
      or ML serving platforms. But understanding the pattern
      helps you configure those tools correctly.
    """

    def __init__(self, production_model, shadow_model, logger=None):
        self.production = production_model
        self.shadow = shadow_model
        self.logger = logger or self._default_logger
        self.comparison_log = []

    def predict(self, input_data):
        """Route prediction through both models, serve only production.

        Why time both?
          Shadow model latency is important — if it's 10x slower,
          it can't replace the production model even if more accurate.
        """
        # Production prediction (this is what the user sees)
        start = time.perf_counter()
        prod_result = self.production.predict(input_data)
        prod_latency = (time.perf_counter() - start) * 1000

        # Shadow prediction (logged, not served)
        # Why try/except? Shadow failures should NEVER affect production
        try:
            start = time.perf_counter()
            shadow_result = self.shadow.predict(input_data)
            shadow_latency = (time.perf_counter() - start) * 1000
        except Exception as e:
            shadow_result = None
            shadow_latency = None
            self.logger("SHADOW_ERROR", str(e))

        # Log comparison
        self.comparison_log.append({
            "production": prod_result.tolist() if hasattr(prod_result, 'tolist')
                          else prod_result,
            "shadow": shadow_result.tolist() if hasattr(shadow_result, 'tolist')
                      else shadow_result,
            "agree": np.array_equal(prod_result, shadow_result),
            "prod_latency_ms": round(prod_latency, 2),
            "shadow_latency_ms": round(shadow_latency, 2) if shadow_latency else None,
        })

        return prod_result  # Always serve production

    def agreement_rate(self):
        """Calculate how often shadow agrees with production.

        Why track agreement?
          - High agreement (>95%): shadow is safe to promote
          - Low agreement: investigate where they differ (may be better OR worse)
          - Track over time to detect instability
        """
        if not self.comparison_log:
            return 0.0
        agreements = sum(1 for log in self.comparison_log if log["agree"])
        return round(agreements / len(self.comparison_log), 4)

    @staticmethod
    def _default_logger(level, message):
        print(f"[{level}] {message}")
```

### 8.2 Canary Release

```python
"""
Canary release: gradually shift traffic from old model to new model.

Why canary over big-bang deployment?
  - Limits blast radius (only X% of users affected by issues)
  - Statistical comparison on real traffic
  - Automated rollback based on metrics

  Traffic progression:
    Stage 1: 5% → new model   (detect catastrophic failures)
    Stage 2: 25% → new model  (statistical significance)
    Stage 3: 50% → new model  (confirm at scale)
    Stage 4: 100% → new model (full rollout)

  Rollback trigger: if error rate > threshold at any stage
"""


class CanaryRelease:
    """Canary release controller for ML models."""

    def __init__(self, current_model, canary_model, stages=None):
        self.current = current_model
        self.canary = canary_model
        self.stages = stages or [0.05, 0.25, 0.50, 1.0]
        self.current_stage = 0
        self.metrics = {"current": [], "canary": []}

    def route_request(self, input_data) -> tuple:
        """Route request to current or canary model based on traffic split.

        Returns (prediction, model_version) for logging.
        """
        import random

        canary_pct = self.stages[self.current_stage]
        use_canary = random.random() < canary_pct

        if use_canary:
            pred = self.canary.predict(input_data)
            return pred, "canary"
        else:
            pred = self.current.predict(input_data)
            return pred, "current"

    def record_outcome(self, model_version, prediction, actual):
        """Record prediction outcome for comparison."""
        correct = int(prediction == actual)
        self.metrics[model_version].append(correct)

    def should_promote(self, min_samples=100, min_improvement=0.0) -> dict:
        """Decide whether to advance to next canary stage.

        Why min_samples?
          Need enough observations for statistical significance.
          With fewer samples, random variation dominates.

          For a 95% confidence interval on accuracy:
          margin of error ≈ 1.96 × √(p(1-p)/n)
          At n=100, p=0.9: margin ≈ ±5.9%
          At n=1000, p=0.9: margin ≈ ±1.9%
        """
        canary_data = self.metrics.get("canary", [])
        current_data = self.metrics.get("current", [])

        if len(canary_data) < min_samples or len(current_data) < min_samples:
            return {"decision": "WAIT", "reason": "insufficient_samples"}

        canary_acc = np.mean(canary_data)
        current_acc = np.mean(current_data)
        improvement = canary_acc - current_acc

        if improvement < -0.05:  # Canary is significantly worse
            return {
                "decision": "ROLLBACK",
                "canary_acc": round(canary_acc, 4),
                "current_acc": round(current_acc, 4),
                "improvement": round(improvement, 4),
            }
        elif improvement >= min_improvement:
            return {
                "decision": "PROMOTE",
                "canary_acc": round(canary_acc, 4),
                "current_acc": round(current_acc, 4),
                "improvement": round(improvement, 4),
                "next_stage": min(self.current_stage + 1, len(self.stages) - 1),
            }
        else:
            return {
                "decision": "HOLD",
                "reason": "improvement_below_threshold",
            }
```

---

## 9. Continuous Validation in Production

### 9.1 Production Monitoring Loop

```python
"""
Continuous validation: automated monitoring → alerting → retraining loop.

Why continuous validation?
  Models degrade over time (data drift, concept drift, world changes).
  Continuous validation detects this BEFORE users complain.

  The monitoring loop:
  ┌──────────────────────────────────────────────────────────────┐
  │                    Continuous Validation Loop                 │
  │                                                              │
  │   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌──────────┐  │
  │   │ Collect  │──▶│ Compute │──▶│ Compare │──▶│ Decision │  │
  │   │ Preds +  │   │ Metrics │   │ Against │   │          │  │
  │   │ Labels   │   │         │   │ Baseline│   │          │  │
  │   └─────────┘   └─────────┘   └─────────┘   └──────────┘  │
  │        ↑                                          │         │
  │        │           ┌──────────┐              ┌────▼────┐    │
  │        └───────────│ Retrain  │◀─────────────│  Alert  │    │
  │                    └──────────┘              └─────────┘    │
  └──────────────────────────────────────────────────────────────┘
"""


class ContinuousValidator:
    """Production model validator with alerting and automated retraining triggers.

    Why not just check accuracy?
      Accuracy requires ground truth labels, which may be delayed
      (e.g., loan default takes months to observe). We need proxy
      metrics that are available immediately:
        - Prediction distribution shift
        - Feature drift
        - Confidence score distribution
        - Serving errors and latency
    """

    def __init__(self, baseline_metrics, alert_thresholds=None):
        self.baseline = baseline_metrics
        self.thresholds = alert_thresholds or {
            "accuracy_drop": 0.05,       # Alert if accuracy drops > 5%
            "psi_threshold": 0.2,        # Alert if PSI > 0.2
            "latency_p99_ms": 200,       # Alert if p99 > 200ms
            "error_rate": 0.01,          # Alert if error rate > 1%
            "confidence_drop": 0.1,      # Alert if mean confidence drops > 10%
        }
        self.alert_history = []

    def check_prediction_distribution(self, predictions, reference_predictions):
        """Check if prediction distribution has shifted.

        Why check prediction distribution?
          Even without ground truth, a large shift in predictions
          suggests something has changed (data, feature engineering, etc.).
          This is a fast proxy signal.
        """
        psi = compute_psi(reference_predictions, predictions)
        alert = psi > self.thresholds["psi_threshold"]

        if alert:
            self.alert_history.append({
                "type": "prediction_distribution_shift",
                "psi": round(psi, 4),
                "threshold": self.thresholds["psi_threshold"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "action": "INVESTIGATE → consider retraining",
            })

        return {"psi": round(psi, 4), "alert": alert}

    def check_confidence_scores(self, confidence_scores):
        """Monitor model confidence distribution.

        Why monitor confidence?
          A model encountering out-of-distribution data often shows:
            - Lower average confidence
            - More predictions near the decision boundary (0.5)
            - Bimodal distribution instead of peaked at 0/1

          This is detectable IMMEDIATELY, no labels needed.
        """
        mean_conf = np.mean(confidence_scores)
        baseline_conf = self.baseline.get("mean_confidence", 0.85)
        drop = baseline_conf - mean_conf

        alert = drop > self.thresholds["confidence_drop"]
        if alert:
            self.alert_history.append({
                "type": "confidence_drop",
                "current_confidence": round(mean_conf, 4),
                "baseline_confidence": round(baseline_conf, 4),
                "drop": round(drop, 4),
                "action": "CHECK for data drift or feature pipeline issues",
            })

        return {
            "mean_confidence": round(mean_conf, 4),
            "baseline_confidence": round(baseline_conf, 4),
            "drop": round(drop, 4),
            "alert": alert,
        }

    def should_retrain(self) -> dict:
        """Decide whether to trigger automated retraining.

        Retraining decision matrix:
          ┌────────────────────────────┬─────────────────────────────┐
          │ Signal                     │ Action                      │
          ├────────────────────────────┼─────────────────────────────┤
          │ Accuracy drop > 5%         │ RETRAIN (urgent)            │
          │ PSI > 0.2                  │ RETRAIN (data shifted)      │
          │ Confidence drop > 10%      │ INVESTIGATE then retrain    │
          │ PSI > 0.1 AND conf drop    │ RETRAIN (compounding signal)│
          │ Latency increase only      │ OPTIMIZE (not retrain)      │
          └────────────────────────────┴─────────────────────────────┘
        """
        recent_alerts = [a for a in self.alert_history[-10:]]

        alert_types = [a["type"] for a in recent_alerts]
        n_alerts = len(recent_alerts)

        if n_alerts == 0:
            return {"decision": "NO_ACTION", "reason": "no_recent_alerts"}

        # Multiple alert types → compound signal is stronger than individual alerts
        if len(set(alert_types)) >= 2:
            return {
                "decision": "RETRAIN",
                "reason": "compound_signals",
                "alert_types": list(set(alert_types)),
                "n_alerts": n_alerts,
            }

        # Persistent single-type alerts suggest systemic issue, not noise
        if n_alerts >= 3:
            return {
                "decision": "RETRAIN",
                "reason": "persistent_alerts",
                "alert_type": alert_types[0],
                "n_alerts": n_alerts,
            }

        return {
            "decision": "MONITOR",
            "reason": "isolated_alert",
            "n_alerts": n_alerts,
        }
```

---

## 10. Test Automation in CI/CD Pipelines

### 10.1 ML Testing in GitHub Actions

```yaml
# .github/workflows/model_tests.yaml
# Why a dedicated ML test workflow?
#   - Separate from code tests (different triggers, longer runtime)
#   - Can run on GPU runners for model inference tests
#   - Data validation runs on every PR; full model tests on main

name: ML Model Testing

on:
  pull_request:
    paths:
      - 'src/models/**'
      - 'src/features/**'
      - 'tests/ml/**'
  push:
    branches: [main]

jobs:
  data-validation:
    # Runs fast — catches data issues early
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements-test.txt

      - name: Run data validation tests
        run: pytest tests/ml/test_data_validation.py -v

      - name: Run schema tests
        run: pytest tests/ml/test_schema.py -v

  model-quality:
    # Runs after data validation passes
    needs: data-validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Download test model and data
        run: dvc pull models/latest data/test

      - name: Run model quality tests
        run: pytest tests/ml/test_model_quality.py -v --tb=long

      - name: Run behavioral tests
        run: pytest tests/ml/test_behavioral.py -v

      - name: Run fairness tests
        run: pytest tests/ml/test_fairness.py -v

      # if: always() ensures report is uploaded even on test failure — needed for debugging
      - name: Upload test report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: ml-test-report
          path: reports/ml_test_results.json

  integration:
    # Tests the full serving pipeline
    needs: model-quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build serving container
        run: docker build -t model-server:test .

      - name: Run serving container
        run: docker run -d -p 8080:8080 model-server:test

      - name: Run latency and throughput tests
        run: pytest tests/ml/test_infrastructure.py -v

      - name: Run integration tests
        run: pytest tests/ml/test_integration.py -v
```

### 10.2 Pytest Fixtures for ML Testing

```python
"""
Pytest fixtures for ML testing — reusable test setup.

Why fixtures?
  - Model loading is expensive; fixtures cache it across tests
  - Test data preparation is shared across test classes
  - Configuration is centralized and overridable
"""
import pytest
import joblib
import pandas as pd


@pytest.fixture(scope="session")
def trained_model():
    """Load the trained model once per test session.

    Why session scope?
      Model loading can take seconds (large models).
      Loading once and reusing across all tests saves time.
      Safe because tests should not modify the model.
    """
    return joblib.load("models/latest/model.pkl")


@pytest.fixture(scope="session")
def test_data():
    """Load test dataset once per session."""
    df = pd.read_parquet("data/test/test_data.parquet")
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


@pytest.fixture
def baseline_metrics():
    """Load baseline metrics for regression testing.

    Why from a file?
      Baseline metrics are set when a model is deployed.
      They represent the "minimum acceptable" performance.
      Stored in version control alongside the model.
    """
    import json
    with open("models/latest/baseline_metrics.json") as f:
        return json.load(f)


# ── Test classes using fixtures ──────────────────────────────────

class TestModelQuality:
    """Model quality tests with automatic threshold checking."""

    def test_accuracy_above_threshold(self, trained_model, test_data):
        X, y = test_data
        y_pred = trained_model.predict(X)
        acc = accuracy_score(y, y_pred)
        assert acc >= 0.85, f"Accuracy {acc:.4f} below threshold 0.85"

    def test_no_regression_from_baseline(self, trained_model, test_data,
                                         baseline_metrics):
        X, y = test_data
        y_pred = trained_model.predict(X)
        current_f1 = f1_score(y, y_pred, average="weighted")
        baseline_f1 = baseline_metrics["f1"]
        margin = 0.02  # Allow 2% regression

        assert current_f1 >= baseline_f1 - margin, (
            f"F1 regressed: {current_f1:.4f} vs baseline {baseline_f1:.4f} "
            f"(margin: {margin})"
        )

    def test_prediction_distribution(self, trained_model, test_data):
        """Sanity check: predictions aren't all the same class.
        A constant predictor can still pass accuracy thresholds on imbalanced data.
        """
        X, _ = test_data
        y_pred = trained_model.predict(X)
        unique_preds = len(set(y_pred))
        assert unique_preds > 1, "Model predicts only one class"
```

---

## 11. Model Card Generation

### 11.1 Automated Model Cards

```python
"""
Model Card (Mitchell et al., 2019): standardized documentation for ML models.

Why model cards?
  - Transparency: users know what the model can and cannot do
  - Accountability: documented training data, metrics, limitations
  - Regulatory: EU AI Act requires model documentation
  - Reproducibility: all training details recorded

  Model Card Sections:
    1. Model Details (architecture, version, owner)
    2. Intended Use (primary use cases, out-of-scope uses)
    3. Training Data (sources, size, preprocessing)
    4. Evaluation Data (test set description)
    5. Metrics (performance on slices)
    6. Ethical Considerations (fairness, bias analysis)
    7. Caveats and Recommendations
"""
from datetime import datetime


def generate_model_card(model_name, model_version, training_config,
                        evaluation_results, fairness_results=None) -> str:
    """Generate a Markdown model card from evaluation results.

    Why auto-generate?
      - Manual model cards get outdated quickly
      - Automated generation ensures every deployed model is documented
      - Integrates into CI/CD: generate card after validation passes
    """
    card = []
    card.append(f"# Model Card: {model_name}")
    card.append(f"\n**Version**: {model_version}")
    card.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    card.append(f"**Status**: {'APPROVED' if evaluation_results.get('gate_decision') == 'APPROVED' else 'PENDING'}")

    # Model Details
    card.append("\n## Model Details")
    card.append(f"- **Architecture**: {training_config.get('model_type', 'N/A')}")
    card.append(f"- **Framework**: {training_config.get('framework', 'scikit-learn')}")
    card.append(f"- **Training Duration**: {training_config.get('training_duration', 'N/A')}")
    card.append(f"- **Dataset Size**: {training_config.get('dataset_size', 'N/A')} samples")

    # Performance Metrics
    card.append("\n## Performance Metrics")
    card.append("\n| Metric | Value | Threshold | Status |")
    card.append("|--------|-------|-----------|--------|")

    quality = evaluation_results.get("quality", {})
    for metric_name, metric_data in quality.items():
        status = "PASS" if metric_data["passed"] else "FAIL"
        card.append(
            f"| {metric_name} | {metric_data['value']:.4f} | "
            f"{metric_data['threshold']:.4f} | {status} |"
        )

    # Fairness Analysis
    if fairness_results:
        card.append("\n## Fairness Analysis")
        card.append(f"- **Disparate Impact Ratio**: {fairness_results['disparate_impact_ratio']:.4f}")
        card.append(f"- **Demographic Parity**: {'PASS' if fairness_results['demographic_parity_passed'] else 'FAIL'}")
        card.append(f"- **Equalized Odds**: {'PASS' if fairness_results['equalized_odds_passed'] else 'FAIL'}")

    # Limitations
    card.append("\n## Limitations and Caveats")
    card.append("- Model performance may degrade on data outside the training distribution")
    card.append("- Regular monitoring and retraining is required")
    card.append(f"- Fairness evaluated on available protected attributes only")

    return "\n".join(card)
```

---

## 12. Putting It All Together

### 12.1 End-to-End Testing Pipeline

```python
"""
Complete ML testing pipeline integrating all test types.

  Pipeline Flow:
  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
  │ Pre-    │──▶│ Train   │──▶│ Post-   │──▶│ Shadow  │──▶│ Prod    │
  │ Training│   │ Model   │   │Training │   │ Deploy  │   │ Monitor │
  │ Gate    │   │         │   │Validate │   │ Test    │   │         │
  └────┬────┘   └─────────┘   └────┬────┘   └────┬────┘   └────┬────┘
       │                            │              │              │
   Data valid?              Quality OK?     Agreement    Drift detected?
   Leakage free?            Fair?           high?        Retrain?
   Schema OK?               Fast enough?
"""


def run_complete_ml_testing_pipeline(
    raw_data,
    trained_model,
    baseline_metrics,
    schema,
    sensitive_column=None,
    reference_predictions=None,
):
    """Execute the full ML testing pipeline.

    This function orchestrates all testing layers from the ML test pyramid.
    Each layer's failure blocks downstream tests (fail-fast).
    """
    pipeline_results = {"stages": []}

    # Stage 1: Pre-training data validation
    gate = PreTrainingGate(schema)
    target_col = "target"
    feature_cols = [c for c in raw_data.columns if c != target_col]
    gate_result = gate.run_all(raw_data, target_col, feature_cols)
    pipeline_results["stages"].append({
        "name": "pre_training_gate",
        "result": gate_result,
    })

    # Fail-fast: bad data makes all downstream tests meaningless
    if not gate_result["gate_passed"]:
        pipeline_results["overall"] = "BLOCKED_AT_DATA_VALIDATION"
        return pipeline_results

    # Stage 2: Model quality tests
    X_test = raw_data[feature_cols]
    y_test = raw_data[target_col]
    quality = ModelQualityTests(trained_model, X_test, y_test)
    quality_result = quality.run_all()
    pipeline_results["stages"].append({
        "name": "model_quality",
        "result": quality_result,
    })

    # Stage 3: Fairness audit
    if sensitive_column and sensitive_column in raw_data.columns:
        y_pred = trained_model.predict(X_test)
        fairness_result = fairness_audit(
            y_test.values, y_pred,
            raw_data[sensitive_column].values
        )
        pipeline_results["stages"].append({
            "name": "fairness_audit",
            "result": fairness_result,
        })

    # Stage 4: Infrastructure tests
    sample = X_test.iloc[:1]
    infra = InfrastructureTests(trained_model, sample)
    pipeline_results["stages"].append({
        "name": "latency",
        "result": infra.latency_test(),
    })

    # Final decision
    all_passed = all(
        stage["result"].get("gate_passed", True) and
        all(
            v.get("passed", True) if isinstance(v, dict) else True
            for v in (stage["result"].values()
                      if isinstance(stage["result"], dict) else [])
        )
        for stage in pipeline_results["stages"]
    )

    pipeline_results["overall"] = "APPROVED" if all_passed else "NEEDS_REVIEW"
    return pipeline_results
```

---

## Summary

Model testing and validation is the discipline that separates hobby ML projects from production-grade ML systems. The key takeaways:

| Layer | What It Tests | When It Runs | Key Tools |
|-------|---------------|--------------|-----------|
| Data validation | Schema, distributions, leakage | Before training | Great Expectations, Pandera |
| Model quality | Accuracy, F1, AUC thresholds | After training | pytest, custom metrics |
| Behavioral tests | Invariance, directional, MFT | After training | CheckList, custom suites |
| Fairness | Demographic parity, equalized odds | After training | Fairlearn, Aequitas |
| Infrastructure | Latency, memory, throughput | Before deployment | Locust, custom benchmarks |
| Shadow testing | Real traffic comparison | During deployment | Service mesh, custom routing |
| Continuous monitoring | Drift, confidence, errors | In production | Evidently, Prometheus |

The ML testing philosophy is: **test early, test often, test on real data, and test for the unexpected.**

---

## Exercises

### Exercise 1: Data Validation Suite
Build a data validation suite for a dataset of your choice. Include:
- Schema validation with at least 5 column rules
- Distribution validation using PSI
- Missing value analysis with configurable thresholds
- Test the suite on intentionally corrupted data to verify it catches issues

### Exercise 2: Behavioral Test Suite
For a text classification model (sentiment, spam, etc.):
- Write 10 minimum functionality test cases
- Implement 5 invariance tests with synonym substitution
- Implement 3 directional expectation tests
- Report the failure rate for each test type

### Exercise 3: Fairness Audit Report
Using a binary classification model trained on a dataset with protected attributes:
- Compute demographic parity, equalized odds, and predictive parity
- Apply the four-fifths rule and document violations
- Propose mitigation strategies for any detected bias
- Generate a model card documenting your findings

### Exercise 4: CI/CD Integration
Create a GitHub Actions workflow that:
- Runs data validation on every pull request
- Runs model quality tests only when model code changes
- Blocks merging if any BLOCKING test fails
- Generates and uploads a test report as an artifact

### Exercise 5: Shadow Deployment Simulator
Implement a shadow deployment simulator that:
- Routes requests to production and shadow models
- Tracks agreement rate, latency comparison, and accuracy difference
- Implements the canary promotion logic
- Generates a report recommending promote, hold, or rollback

---

[← Previous: 15. LLMOps](15_LLMOps.md) | [Next: 17. Cloud MLOps Platforms →](17_Cloud_MLOps_Platforms.md)
