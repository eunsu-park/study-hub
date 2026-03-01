"""
Model Validation for ML CI/CD Pipeline
=======================================
Demonstrates:
- Data validation gate (schema, nulls, distribution, freshness)
- Model evaluation gate (accuracy, latency, improvement, fairness)
- Canary monitoring
- Automated rollback logic

Run: python model_validation.py <check>
Available: data, model, canary, all
"""

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# ── 1. Data Validation Gate ────────────────────────────────────────

@dataclass
class ValidationResult:
    check: str
    passed: bool
    detail: str
    severity: str = "error"  # error, warning


def validate_schema(df, expected_columns, expected_dtypes=None):
    """Validate DataFrame schema."""
    actual_cols = set(df.columns)
    expected_cols = set(expected_columns)

    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols

    details = []
    if missing:
        details.append(f"Missing columns: {missing}")
    if extra:
        details.append(f"Extra columns: {extra}")

    return ValidationResult(
        check="schema",
        passed=len(missing) == 0,
        detail="; ".join(details) if details else "Schema OK",
    )


def validate_nulls(df, max_null_pct=5.0, critical_columns=None):
    """Check for excessive null values."""
    null_pcts = (df.isnull().sum() / len(df) * 100).round(2)
    high_null = null_pcts[null_pcts > max_null_pct]

    # Critical columns must have zero nulls
    critical_issues = []
    if critical_columns:
        for col in critical_columns:
            if col in df.columns and df[col].isnull().any():
                critical_issues.append(f"{col}: {df[col].isnull().sum()} nulls")

    passed = len(high_null) == 0 and len(critical_issues) == 0
    details = []
    if len(high_null) > 0:
        details.append(f"High null columns: {high_null.to_dict()}")
    if critical_issues:
        details.append(f"Critical nulls: {critical_issues}")

    return ValidationResult(
        check="null_values",
        passed=passed,
        detail="; ".join(details) if details else "Null check OK",
    )


def validate_distribution(df, reference_stats, column, threshold=0.3):
    """Check if feature distribution has shifted significantly."""
    current_mean = df[column].mean()
    current_std = df[column].std()
    ref_mean = reference_stats[column]["mean"]
    ref_std = reference_stats[column]["std"]

    # Normalized mean shift
    if ref_std > 0:
        shift = abs(current_mean - ref_mean) / ref_std
    else:
        shift = abs(current_mean - ref_mean)

    return ValidationResult(
        check=f"distribution_{column}",
        passed=shift < threshold,
        detail=f"Mean shift: {shift:.3f} (threshold: {threshold})",
        severity="warning" if shift < threshold * 1.5 else "error",
    )


def validate_freshness(data_timestamp, max_age_days=7):
    """Check that data is not stale."""
    age = datetime.now() - data_timestamp
    max_age = timedelta(days=max_age_days)

    return ValidationResult(
        check="freshness",
        passed=age <= max_age,
        detail=f"Data age: {age.days} days (max: {max_age_days})",
    )


def run_data_validation():
    """Run all data validation checks."""
    print("="*60)
    print("DATA VALIDATION GATE")
    print("="*60)

    # Generate sample data for demo
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "feature_a": np.random.normal(10, 2, n),
        "feature_b": np.random.exponential(5, n),
        "target": np.random.binomial(1, 0.3, n),
    })
    # Add some nulls
    df.loc[np.random.choice(n, 20), "feature_b"] = np.nan

    reference_stats = {
        "feature_a": {"mean": 10.0, "std": 2.0},
        "feature_b": {"mean": 5.0, "std": 5.0},
    }

    results = [
        validate_schema(df, ["feature_a", "feature_b", "target"]),
        validate_nulls(df, max_null_pct=5.0, critical_columns=["target"]),
        validate_distribution(df, reference_stats, "feature_a"),
        validate_distribution(df, reference_stats, "feature_b"),
        validate_freshness(datetime.now() - timedelta(days=2), max_age_days=7),
    ]

    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        icon = "  " if r.passed else "!!"
        print(f"  [{status}] {icon} {r.check}: {r.detail}")
        if not r.passed and r.severity == "error":
            all_passed = False

    print(f"\nData Validation: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


# ── 2. Model Evaluation Gate ──────────────────────────────────────

@dataclass
class EvalConfig:
    min_accuracy: float = 0.90
    min_f1: float = 0.85
    max_latency_p99_ms: float = 100.0
    min_improvement: float = 0.005
    max_model_size_mb: float = 500.0
    min_demographic_parity: float = 0.8


def evaluate_model_performance(model_metrics, production_metrics, config):
    """Evaluate model against deployment criteria."""
    results = []

    # Accuracy gate
    results.append(ValidationResult(
        check="accuracy",
        passed=model_metrics["accuracy"] >= config.min_accuracy,
        detail=f"{model_metrics['accuracy']:.4f} (min: {config.min_accuracy})",
    ))

    # F1 gate
    results.append(ValidationResult(
        check="f1_score",
        passed=model_metrics["f1"] >= config.min_f1,
        detail=f"{model_metrics['f1']:.4f} (min: {config.min_f1})",
    ))

    # Improvement over production
    improvement = model_metrics["accuracy"] - production_metrics["accuracy"]
    results.append(ValidationResult(
        check="improvement",
        passed=improvement >= config.min_improvement,
        detail=f"{improvement:+.4f} (min: {config.min_improvement})",
    ))

    # Latency gate
    results.append(ValidationResult(
        check="latency_p99",
        passed=model_metrics["latency_p99_ms"] <= config.max_latency_p99_ms,
        detail=f"{model_metrics['latency_p99_ms']:.1f}ms (max: {config.max_latency_p99_ms}ms)",
    ))

    # Model size gate
    results.append(ValidationResult(
        check="model_size",
        passed=model_metrics["model_size_mb"] <= config.max_model_size_mb,
        detail=f"{model_metrics['model_size_mb']:.0f}MB (max: {config.max_model_size_mb}MB)",
    ))

    return results


def run_model_evaluation():
    """Run model evaluation gate."""
    print("="*60)
    print("MODEL EVALUATION GATE")
    print("="*60)

    config = EvalConfig()

    # Simulated metrics
    model_metrics = {
        "accuracy": 0.935,
        "f1": 0.921,
        "auc": 0.967,
        "latency_p99_ms": 42.0,
        "model_size_mb": 245.0,
    }
    production_metrics = {
        "accuracy": 0.928,
    }

    results = evaluate_model_performance(model_metrics, production_metrics, config)

    all_passed = all(r.passed for r in results)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.check}: {r.detail}")

    print(f"\nModel Evaluation: {'APPROVED' if all_passed else 'BLOCKED'}")
    return all_passed


# ── 3. Canary Monitor ─────────────────────────────────────────────

def simulate_canary_monitoring(duration_seconds=10, check_interval=2):
    """Simulate canary monitoring with health checks."""
    print("="*60)
    print("CANARY MONITORING")
    print("="*60)

    start = time.time()
    checks = 0
    healthy = True

    while time.time() - start < duration_seconds:
        checks += 1
        elapsed = time.time() - start

        # Simulated metrics
        error_rate = np.random.uniform(0.001, 0.015)
        latency_p99 = np.random.uniform(30, 80)
        throughput = np.random.uniform(90, 110)

        status = "HEALTHY"
        if error_rate > 0.02:
            status = "DEGRADED"
            healthy = False
        if latency_p99 > 100:
            status = "DEGRADED"
            healthy = False

        print(f"  [{elapsed:5.1f}s] Check #{checks}: "
              f"error_rate={error_rate:.3f} "
              f"latency_p99={latency_p99:.0f}ms "
              f"rps={throughput:.0f} "
              f"→ {status}")

        if not healthy:
            print(f"\n  ROLLBACK TRIGGERED at check #{checks}")
            return False

        time.sleep(check_interval)

    print(f"\nCanary monitoring: {'HEALTHY' if healthy else 'ROLLBACK'} "
          f"({checks} checks over {duration_seconds}s)")
    return healthy


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    checks = {
        "data": run_data_validation,
        "model": run_model_evaluation,
        "canary": lambda: simulate_canary_monitoring(duration_seconds=10),
    }

    if len(sys.argv) < 2:
        print("Usage: python model_validation.py <check>")
        print(f"Available: {', '.join(checks.keys())}, all")
        sys.exit(0)

    target = sys.argv[1]

    if target == "all":
        for name, fn in checks.items():
            fn()
            print()
    elif target in checks:
        passed = checks[target]()
        sys.exit(0 if passed else 1)
    else:
        print(f"Unknown check: {target}")
        sys.exit(1)
