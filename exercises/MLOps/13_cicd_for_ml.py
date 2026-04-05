"""
Exercise Solutions: CI/CD for ML
===========================================
Lesson 13 from MLOps topic.

Exercises
---------
1. Identify ML CI/CD Differences — Analyze key differences between
   traditional software CI/CD and ML CI/CD pipelines.
2. Write Data Validation Script — Implement a data validation gate
   that checks schema, statistics, and distribution properties.
3. Implement Evaluation Gate — Build a model evaluation gate that
   enforces quality thresholds before deployment.
4. Compare Deployment Strategies — Simulate and compare canary,
   shadow, and blue-green deployment strategies.
5. Design Full ML CI/CD Pipeline — Assemble a complete CI/CD pipeline
   with all stages from code commit to production deployment.
"""

import math
import random
import json
import time
from datetime import datetime, timedelta


# ============================================================
# Exercise 1: Identify ML CI/CD Differences
# ============================================================

def exercise_1_ml_cicd_differences():
    """Analyze differences between traditional CI/CD and ML CI/CD.

    Traditional CI/CD tests code. ML CI/CD must additionally test data,
    models, and the interaction between all three.
    """

    differences = {
        "Artifact Types": {
            "Traditional": "Application binary / container image",
            "ML": "Model artifact + serving config + feature schema + data reference",
            "Implication": "ML artifacts are larger and more complex to version and store",
        },
        "Testing Scope": {
            "Traditional": "Unit tests, integration tests, E2E tests",
            "ML": "All above + data validation, model evaluation, bias testing, "
                  "performance benchmarks, drift checks",
            "Implication": "ML test suites take much longer and need specialized infrastructure",
        },
        "Reproducibility": {
            "Traditional": "Same code + deps = same binary (deterministic)",
            "ML": "Same code + deps + data + seed + hardware may still differ "
                  "(floating point, GPU non-determinism)",
            "Implication": "ML needs stricter environment pinning and tolerance-based assertions",
        },
        "Rollback Complexity": {
            "Traditional": "Deploy previous container version",
            "ML": "Model rollback + feature store compatibility + data pipeline alignment",
            "Implication": "ML rollbacks may require coordinated changes across multiple systems",
        },
        "Validation Criteria": {
            "Traditional": "All tests pass (boolean)",
            "ML": "Metrics above thresholds, no regression on slices, fairness constraints",
            "Implication": "ML gates are statistical, not deterministic — need confidence intervals",
        },
        "Pipeline Triggers": {
            "Traditional": "Code push, PR merge, schedule",
            "ML": "All above + data drift, performance degradation, new labeled data available",
            "Implication": "ML pipelines need monitoring-driven triggers, not just code events",
        },
        "Environment Requirements": {
            "Traditional": "Standard compute (CPU)",
            "ML": "GPU/TPU for training, specialized serving infrastructure",
            "Implication": "ML CI/CD needs access to expensive compute resources",
        },
    }

    print("ML CI/CD vs Traditional CI/CD")
    print("=" * 60)

    for dimension, details in differences.items():
        print(f"\n  {dimension}:")
        print(f"    Traditional: {details['Traditional']}")
        print(f"    ML:          {details['ML']}")
        print(f"    Implication: {details['Implication']}")

    # Summary table
    print("\n\nSummary: Additional ML CI/CD Stages")
    print("-" * 60)
    ml_specific_stages = [
        ("Data Validation Gate", "Verify schema, stats, distribution before training"),
        ("Training Pipeline", "Reproducible training with experiment tracking"),
        ("Model Evaluation Gate", "Metric thresholds, slice analysis, bias checks"),
        ("Shadow Deployment", "Run new model alongside current without serving"),
        ("Canary Release", "Gradual traffic shift with rollback capability"),
        ("Monitoring Integration", "Automated drift detection and retraining triggers"),
    ]
    for stage, description in ml_specific_stages:
        print(f"  {stage:<25s} {description}")

    return differences


# ============================================================
# Exercise 2: Write Data Validation Script
# ============================================================

def exercise_2_data_validation():
    """Implement a data validation gate for CI/CD.

    Checks:
    1. Schema validation (columns, types)
    2. Statistical validation (mean, std within expected ranges)
    3. Distribution validation (no extreme drift from reference)
    4. Completeness validation (null rates, row counts)
    """

    class DataValidationGate:
        """CI/CD gate that validates training data before model training."""

        def __init__(self, reference_stats):
            self.reference_stats = reference_stats
            self.checks = []

        def validate_schema(self, data, expected_columns):
            """Check that all expected columns exist with correct types."""
            actual_columns = set(data[0].keys()) if data else set()
            missing = set(expected_columns.keys()) - actual_columns
            extra = actual_columns - set(expected_columns.keys())

            type_errors = []
            for col, expected_type in expected_columns.items():
                if col not in actual_columns:
                    continue
                sample = next((row[col] for row in data if row[col] is not None), None)
                if sample is not None and not isinstance(sample, expected_type):
                    type_errors.append(f"{col}: expected {expected_type.__name__}, "
                                       f"got {type(sample).__name__}")

            passed = len(missing) == 0 and len(type_errors) == 0
            result = {
                "check": "schema_validation",
                "passed": passed,
                "missing_columns": list(missing),
                "extra_columns": list(extra),
                "type_errors": type_errors,
            }
            self.checks.append(result)
            return result

        def validate_statistics(self, data, column, tolerance=0.3):
            """Check that column statistics are within tolerance of reference."""
            values = [row[column] for row in data if row.get(column) is not None
                      and isinstance(row[column], (int, float))]
            if not values:
                result = {"check": f"stats_{column}", "passed": False,
                          "reason": "No valid values"}
                self.checks.append(result)
                return result

            actual_mean = sum(values) / len(values)
            actual_std = math.sqrt(sum((v - actual_mean) ** 2 for v in values) / len(values))

            ref = self.reference_stats.get(column, {})
            ref_mean = ref.get("mean", actual_mean)
            ref_std = ref.get("std", actual_std)

            mean_drift = abs(actual_mean - ref_mean) / (ref_std + 1e-8)
            std_ratio = actual_std / (ref_std + 1e-8)

            passed = mean_drift < tolerance and 0.5 < std_ratio < 2.0
            result = {
                "check": f"stats_{column}",
                "passed": passed,
                "actual_mean": round(actual_mean, 4),
                "reference_mean": round(ref_mean, 4),
                "mean_drift_sigmas": round(mean_drift, 4),
                "std_ratio": round(std_ratio, 4),
                "tolerance": tolerance,
            }
            self.checks.append(result)
            return result

        def validate_completeness(self, data, min_rows=100, max_null_rate=0.05):
            """Check data completeness."""
            n_rows = len(data)
            null_rates = {}
            for col in data[0].keys() if data else []:
                null_count = sum(1 for row in data if row.get(col) is None)
                null_rates[col] = null_count / n_rows if n_rows > 0 else 0

            high_null_cols = {col: rate for col, rate in null_rates.items()
                             if rate > max_null_rate}

            passed = n_rows >= min_rows and len(high_null_cols) == 0
            result = {
                "check": "completeness",
                "passed": passed,
                "n_rows": n_rows,
                "min_rows_required": min_rows,
                "null_rates": {k: round(v, 4) for k, v in null_rates.items()},
                "high_null_columns": high_null_cols,
            }
            self.checks.append(result)
            return result

        def validate_label_distribution(self, data, label_col, min_class_ratio=0.05):
            """Check that label distribution is not too imbalanced."""
            labels = [row[label_col] for row in data if row.get(label_col) is not None]
            counts = {}
            for l in labels:
                counts[l] = counts.get(l, 0) + 1

            total = len(labels)
            ratios = {k: v / total for k, v in counts.items()}
            min_ratio = min(ratios.values()) if ratios else 0

            passed = min_ratio >= min_class_ratio
            result = {
                "check": "label_distribution",
                "passed": passed,
                "label_counts": counts,
                "label_ratios": {k: round(v, 4) for k, v in ratios.items()},
                "min_ratio": round(min_ratio, 4),
                "min_ratio_required": min_class_ratio,
            }
            self.checks.append(result)
            return result

        def get_summary(self):
            all_passed = all(c["passed"] for c in self.checks)
            return {
                "overall": "PASSED" if all_passed else "FAILED",
                "total_checks": len(self.checks),
                "passed": sum(1 for c in self.checks if c["passed"]),
                "failed": sum(1 for c in self.checks if not c["passed"]),
            }

    # --- Generate test data ---
    random.seed(42)
    data = []
    for i in range(500):
        row = {
            "age": random.randint(18, 80),
            "income": round(30000 + random.gauss(25000, 15000), 2),
            "tenure_months": random.randint(1, 60),
            "monthly_charges": round(40 + random.gauss(30, 15), 2),
            "churned": random.choice([0, 0, 0, 1]),
        }
        # Inject some nulls
        if random.random() < 0.02:
            row["income"] = None
        data.append(row)

    reference_stats = {
        "age": {"mean": 49, "std": 18},
        "income": {"mean": 55000, "std": 15000},
        "monthly_charges": {"mean": 70, "std": 15},
    }

    # --- Run validation ---
    print("Data Validation Gate")
    print("=" * 60)

    gate = DataValidationGate(reference_stats)

    expected_schema = {
        "age": int, "income": float, "tenure_months": int,
        "monthly_charges": float, "churned": int,
    }

    r1 = gate.validate_schema(data, expected_schema)
    print(f"\n  Schema: {'PASS' if r1['passed'] else 'FAIL'}")

    for col in ["age", "income", "monthly_charges"]:
        r = gate.validate_statistics(data, col)
        print(f"  Stats ({col}): {'PASS' if r['passed'] else 'FAIL'} "
              f"(drift={r['mean_drift_sigmas']:.3f}σ)")

    r_comp = gate.validate_completeness(data)
    print(f"  Completeness: {'PASS' if r_comp['passed'] else 'FAIL'} "
          f"(rows={r_comp['n_rows']})")

    r_label = gate.validate_label_distribution(data, "churned")
    print(f"  Label dist: {'PASS' if r_label['passed'] else 'FAIL'} "
          f"(min_ratio={r_label['min_ratio']:.3f})")

    summary = gate.get_summary()
    print(f"\n  Overall: {summary['overall']} "
          f"({summary['passed']}/{summary['total_checks']} checks passed)")

    return gate


# ============================================================
# Exercise 3: Implement Evaluation Gate
# ============================================================

def exercise_3_evaluation_gate():
    """Build a model evaluation gate for CI/CD.

    The gate checks:
    1. Absolute metric thresholds (accuracy > 0.80, etc.)
    2. Relative comparison vs current production model
    3. Per-slice performance (no demographic bias)
    4. Latency requirements
    """

    class EvaluationGate:
        def __init__(self, thresholds, production_metrics=None):
            self.thresholds = thresholds
            self.production_metrics = production_metrics
            self.results = []

        def check_absolute(self, metrics):
            """Check metrics against absolute thresholds."""
            failures = []
            for metric, threshold in self.thresholds.items():
                actual = metrics.get(metric)
                if actual is not None and actual < threshold:
                    failures.append({
                        "metric": metric,
                        "actual": actual,
                        "threshold": threshold,
                    })
            passed = len(failures) == 0
            self.results.append({
                "check": "absolute_thresholds",
                "passed": passed,
                "failures": failures,
            })
            return passed

        def check_relative(self, metrics, max_regression=0.02):
            """Check that new model is not worse than production."""
            if not self.production_metrics:
                self.results.append({
                    "check": "relative_comparison",
                    "passed": True,
                    "note": "No production model to compare against",
                })
                return True

            regressions = []
            for metric in self.production_metrics:
                prod_val = self.production_metrics[metric]
                new_val = metrics.get(metric, 0)
                if new_val < prod_val - max_regression:
                    regressions.append({
                        "metric": metric,
                        "production": prod_val,
                        "new": new_val,
                        "regression": round(prod_val - new_val, 4),
                    })

            passed = len(regressions) == 0
            self.results.append({
                "check": "relative_comparison",
                "passed": passed,
                "regressions": regressions,
            })
            return passed

        def check_slices(self, slice_metrics, min_metric="f1", min_value=0.70):
            """Check per-slice performance for fairness."""
            failing_slices = []
            for slice_name, metrics in slice_metrics.items():
                val = metrics.get(min_metric, 0)
                if val < min_value:
                    failing_slices.append({
                        "slice": slice_name,
                        "metric": min_metric,
                        "value": val,
                        "threshold": min_value,
                    })

            passed = len(failing_slices) == 0
            self.results.append({
                "check": "slice_analysis",
                "passed": passed,
                "failing_slices": failing_slices,
            })
            return passed

    # --- Run evaluation gate ---
    print("Model Evaluation Gate")
    print("=" * 60)

    production_metrics = {"accuracy": 0.83, "precision": 0.75, "recall": 0.80, "f1": 0.77}
    thresholds = {"accuracy": 0.80, "precision": 0.70, "recall": 0.75, "f1": 0.72}

    new_model_metrics = {"accuracy": 0.86, "precision": 0.78, "recall": 0.82, "f1": 0.80}
    slice_metrics = {
        "age_18_30": {"f1": 0.78, "accuracy": 0.84},
        "age_30_50": {"f1": 0.82, "accuracy": 0.88},
        "age_50_70": {"f1": 0.75, "accuracy": 0.82},
        "age_70+": {"f1": 0.68, "accuracy": 0.76},  # Below threshold
    }

    gate = EvaluationGate(thresholds, production_metrics)

    abs_passed = gate.check_absolute(new_model_metrics)
    rel_passed = gate.check_relative(new_model_metrics)
    slice_passed = gate.check_slices(slice_metrics)

    print(f"\n  Absolute thresholds: {'PASS' if abs_passed else 'FAIL'}")
    print(f"  Relative comparison: {'PASS' if rel_passed else 'FAIL'}")
    print(f"  Slice analysis:      {'PASS' if slice_passed else 'FAIL'}")

    if not slice_passed:
        for r in gate.results:
            if r["check"] == "slice_analysis":
                for fs in r["failing_slices"]:
                    print(f"    FAIL: {fs['slice']} {fs['metric']}={fs['value']} "
                          f"(threshold={fs['threshold']})")

    overall = abs_passed and rel_passed  # Note: slice failure is a warning
    print(f"\n  Overall gate: {'PASS' if overall else 'FAIL'}")
    if not slice_passed:
        print(f"  WARNING: Slice analysis failed — investigate before production")

    return gate


# ============================================================
# Exercise 4: Compare Deployment Strategies
# ============================================================

def exercise_4_deployment_strategies():
    """Simulate and compare canary, shadow, and blue-green deployments."""

    random.seed(42)

    def simulate_model(model_name, error_rate, latency_base, n_requests):
        results = []
        for _ in range(n_requests):
            latency = latency_base + random.gauss(0, latency_base * 0.15)
            error = random.random() < error_rate
            results.append({"latency_ms": latency, "error": error})
        return results

    print("Deployment Strategy Comparison")
    print("=" * 60)

    # --- Canary ---
    print("\n  1. Canary Deployment")
    print(f"  {'-'*40}")
    stages = [(5, 100), (10, 200), (25, 500), (50, 500), (100, 500)]
    for pct, requests in stages:
        canary_results = simulate_model("v2", 0.01, 10, int(requests * pct / 100))
        prod_results = simulate_model("v1", 0.02, 12, int(requests * (100 - pct) / 100))
        c_err = sum(1 for r in canary_results if r["error"]) / max(1, len(canary_results))
        p_err = sum(1 for r in prod_results if r["error"]) / max(1, len(prod_results))
        print(f"    {pct:3d}% traffic: canary_err={c_err:.1%} prod_err={p_err:.1%}")
    print(f"    Pros: Gradual rollout, easy rollback, real traffic validation")
    print(f"    Cons: Slow rollout, affects real users during testing")

    # --- Shadow ---
    print("\n  2. Shadow Deployment")
    print(f"  {'-'*40}")
    shadow_results = simulate_model("v2_shadow", 0.01, 10, 1000)
    prod_results = simulate_model("v1_prod", 0.02, 12, 1000)

    shadow_latencies = [r["latency_ms"] for r in shadow_results]
    prod_latencies = [r["latency_ms"] for r in prod_results]
    s_err = sum(1 for r in shadow_results if r["error"]) / len(shadow_results)
    p_err = sum(1 for r in prod_results if r["error"]) / len(prod_results)

    print(f"    Shadow: err={s_err:.1%}, avg_latency={sum(shadow_latencies)/len(shadow_latencies):.1f}ms")
    print(f"    Prod:   err={p_err:.1%}, avg_latency={sum(prod_latencies)/len(prod_latencies):.1f}ms")
    print(f"    Pros: Zero risk to users, full production traffic testing")
    print(f"    Cons: Double compute cost, cannot test user-facing impact")

    # --- Blue-Green ---
    print("\n  3. Blue-Green Deployment")
    print(f"  {'-'*40}")
    blue_results = simulate_model("v1_blue", 0.02, 12, 500)
    green_results = simulate_model("v2_green", 0.01, 10, 500)
    b_err = sum(1 for r in blue_results if r["error"]) / len(blue_results)
    g_err = sum(1 for r in green_results if r["error"]) / len(green_results)
    print(f"    Blue (current):  err={b_err:.1%}")
    print(f"    Green (new):     err={g_err:.1%}")
    print(f"    Switch: Instant traffic cutover from Blue to Green")
    print(f"    Pros: Instant rollback, simple mental model")
    print(f"    Cons: Requires 2x infrastructure, all-or-nothing switch")

    # Comparison table
    print("\n  Strategy Comparison:")
    print(f"  {'Strategy':<15s} {'Risk':>6s} {'Cost':>8s} {'Rollback':>10s} {'Speed':>8s}")
    print(f"  {'-'*50}")
    print(f"  {'Canary':<15s} {'Low':>6s} {'Low':>8s} {'Gradual':>10s} {'Slow':>8s}")
    print(f"  {'Shadow':<15s} {'None':>6s} {'High':>8s} {'N/A':>10s} {'Fast':>8s}")
    print(f"  {'Blue-Green':<15s} {'Med':>6s} {'High':>8s} {'Instant':>10s} {'Fast':>8s}")

    return {"canary": canary_results, "shadow": shadow_results, "blue_green": green_results}


# ============================================================
# Exercise 5: Design Full ML CI/CD Pipeline
# ============================================================

def exercise_5_full_pipeline():
    """Assemble a complete ML CI/CD pipeline."""

    class MLCICDPipeline:
        def __init__(self, name):
            self.name = name
            self.stages = []
            self.execution_log = []

        def add_stage(self, name, check_fn, blocking=True):
            self.stages.append({
                "name": name,
                "check": check_fn,
                "blocking": blocking,
            })

        def execute(self, context):
            print(f"\n  Pipeline: {self.name}")
            print(f"  Stages: {len(self.stages)}")
            print(f"  {'-'*50}")

            for i, stage in enumerate(self.stages, 1):
                result = stage["check"](context)
                status = "PASS" if result["passed"] else "FAIL"
                block = " (BLOCKING)" if stage["blocking"] and not result["passed"] else ""
                print(f"  [{i:2d}] {stage['name']:<35s} [{status}]{block}")

                if result.get("details"):
                    print(f"       {result['details']}")

                self.execution_log.append({
                    "stage": stage["name"],
                    "passed": result["passed"],
                    "blocking": stage["blocking"],
                })

                if stage["blocking"] and not result["passed"]:
                    print(f"\n  PIPELINE HALTED at stage: {stage['name']}")
                    return False

            print(f"\n  PIPELINE COMPLETED SUCCESSFULLY")
            return True

    # --- Build the pipeline ---
    pipeline = MLCICDPipeline("churn-model-cicd")

    random.seed(42)

    # Stage checks
    pipeline.add_stage("Code linting & formatting", lambda ctx: {
        "passed": True, "details": "flake8: 0 errors, black: formatted"
    })
    pipeline.add_stage("Unit tests", lambda ctx: {
        "passed": True, "details": "42 tests passed in 3.2s"
    })
    pipeline.add_stage("Data validation", lambda ctx: {
        "passed": True, "details": "Schema OK, stats within bounds, 500 rows"
    })
    pipeline.add_stage("Training pipeline", lambda ctx: {
        "passed": True, "details": "Model trained in 45s, loss=0.312"
    })
    pipeline.add_stage("Evaluation gate (absolute)", lambda ctx: {
        "passed": True, "details": "acc=0.86, f1=0.80 (thresholds: 0.80, 0.72)"
    })
    pipeline.add_stage("Evaluation gate (relative)", lambda ctx: {
        "passed": True, "details": "No regression vs production (acc +0.03, f1 +0.03)"
    })
    pipeline.add_stage("Slice analysis", lambda ctx: {
        "passed": False, "details": "age_70+ f1=0.68 < 0.70 threshold"
    }, blocking=False)  # Warning only
    pipeline.add_stage("Security scan", lambda ctx: {
        "passed": True, "details": "No vulnerabilities found"
    })
    pipeline.add_stage("Container build", lambda ctx: {
        "passed": True, "details": "Image built: churn-model:v2.1 (245MB)"
    })
    pipeline.add_stage("Integration tests", lambda ctx: {
        "passed": True, "details": "API tests: 15/15 passed, latency p99: 45ms"
    })
    pipeline.add_stage("Shadow deployment", lambda ctx: {
        "passed": True, "details": "24h shadow test: err=0.8% (prod=1.2%)"
    })
    pipeline.add_stage("Canary rollout (5%)", lambda ctx: {
        "passed": True, "details": "1h canary: err=0.9%, latency=11ms"
    })
    pipeline.add_stage("Canary rollout (25%)", lambda ctx: {
        "passed": True, "details": "2h canary: err=0.8%, latency=10ms"
    })
    pipeline.add_stage("Full production deployment", lambda ctx: {
        "passed": True, "details": "Deployed to 100% traffic"
    })
    pipeline.add_stage("Post-deploy monitoring check", lambda ctx: {
        "passed": True, "details": "30min post-deploy: no anomalies"
    })

    print("Full ML CI/CD Pipeline")
    print("=" * 60)

    pipeline.execute({})

    # Summary
    total = len(pipeline.execution_log)
    passed = sum(1 for r in pipeline.execution_log if r["passed"])
    print(f"\n  Summary: {passed}/{total} stages passed")

    return pipeline


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Exercise 1: ML CI/CD Differences")
    print("=" * 60)
    exercise_1_ml_cicd_differences()

    print("\n\n")
    print("Exercise 2: Data Validation Gate")
    print("=" * 60)
    exercise_2_data_validation()

    print("\n\n")
    print("Exercise 3: Evaluation Gate")
    print("=" * 60)
    exercise_3_evaluation_gate()

    print("\n\n")
    print("Exercise 4: Deployment Strategies")
    print("=" * 60)
    exercise_4_deployment_strategies()

    print("\n\n")
    print("Exercise 5: Full ML CI/CD Pipeline")
    print("=" * 60)
    exercise_5_full_pipeline()
