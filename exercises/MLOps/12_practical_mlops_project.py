"""
Exercise Solutions: Practical MLOps Project
===========================================
Lesson 12 from MLOps topic.

Exercises
---------
1. Practice Exercise — Build an end-to-end MLOps project with 5 sub-tasks:
   a. Implement data pipeline with validation and versioning
   b. Build training pipeline with experiment tracking
   c. Deploy model with CI/CD and canary release
   d. Set up monitoring with drift detection and alerting
   e. Implement automated retraining loop
"""

import math
import random
import json
import hashlib
import time
from datetime import datetime, timedelta


# ============================================================
# Sub-task (a): Data Pipeline with Validation and Versioning
# ============================================================

def subtask_a_data_pipeline():
    """Implement a data pipeline with validation and versioning.

    Components:
    - Data ingestion from simulated sources
    - Schema validation (types, ranges, required fields)
    - Data quality checks (nulls, duplicates, outliers)
    - Dataset versioning with content hashing
    """

    class DataValidator:
        """Validates data against a schema and quality rules."""

        def __init__(self, schema):
            self.schema = schema
            self.validation_results = []

        def validate(self, data):
            errors = []
            warnings = []

            # Schema validation
            for row_idx, row in enumerate(data):
                for field, rules in self.schema.items():
                    value = row.get(field)

                    # Required check
                    if rules.get("required", False) and value is None:
                        errors.append(f"Row {row_idx}: '{field}' is required but missing")
                        continue

                    if value is None:
                        continue

                    # Type check
                    expected_type = rules.get("type")
                    if expected_type and not isinstance(value, expected_type):
                        errors.append(
                            f"Row {row_idx}: '{field}' expected {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )

                    # Range check
                    if "min" in rules and isinstance(value, (int, float)):
                        if value < rules["min"]:
                            warnings.append(
                                f"Row {row_idx}: '{field}'={value} below min {rules['min']}"
                            )
                    if "max" in rules and isinstance(value, (int, float)):
                        if value > rules["max"]:
                            warnings.append(
                                f"Row {row_idx}: '{field}'={value} above max {rules['max']}"
                            )

                    # Allowed values
                    if "allowed" in rules and value not in rules["allowed"]:
                        errors.append(
                            f"Row {row_idx}: '{field}'={value} not in {rules['allowed']}"
                        )

            # Quality checks
            n_rows = len(data)
            null_rates = {}
            for field in self.schema:
                null_count = sum(1 for row in data if row.get(field) is None)
                null_rates[field] = null_count / n_rows if n_rows > 0 else 0
                if null_rates[field] > 0.1:
                    warnings.append(f"'{field}' null rate: {null_rates[field]:.1%}")

            # Duplicate check
            seen = set()
            duplicates = 0
            for row in data:
                key = json.dumps(row, sort_keys=True, default=str)
                if key in seen:
                    duplicates += 1
                seen.add(key)
            if duplicates > 0:
                warnings.append(f"{duplicates} duplicate rows detected")

            result = {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "n_rows": n_rows,
                "null_rates": null_rates,
                "duplicates": duplicates,
            }
            self.validation_results.append(result)
            return result

    class DataVersioner:
        """Version datasets using content hashing."""

        def __init__(self):
            self.versions = []

        def version(self, data, metadata=None):
            content = json.dumps(data, sort_keys=True, default=str)
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
            version = {
                "version": len(self.versions) + 1,
                "hash": content_hash,
                "n_rows": len(data),
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
            }
            self.versions.append(version)
            return version

    # --- Generate sample data ---
    random.seed(42)
    schema = {
        "customer_id": {"type": int, "required": True, "min": 1},
        "age": {"type": int, "required": True, "min": 18, "max": 100},
        "monthly_charges": {"type": float, "required": True, "min": 0},
        "contract_type": {"type": str, "required": True,
                          "allowed": ["month-to-month", "one-year", "two-year"]},
        "churned": {"type": int, "required": True, "allowed": [0, 1]},
    }

    good_data = []
    for i in range(100):
        good_data.append({
            "customer_id": i + 1,
            "age": random.randint(18, 80),
            "monthly_charges": round(30 + random.gauss(35, 20), 2),
            "contract_type": random.choice(["month-to-month", "one-year", "two-year"]),
            "churned": random.choice([0, 0, 0, 1]),
        })

    # Data with issues
    bad_data = good_data[:90] + [
        {"customer_id": 91, "age": 150, "monthly_charges": 50.0,
         "contract_type": "month-to-month", "churned": 0},  # Age out of range
        {"customer_id": 92, "age": None, "monthly_charges": 50.0,
         "contract_type": "month-to-month", "churned": 0},  # Null age
        {"customer_id": 93, "age": 30, "monthly_charges": 50.0,
         "contract_type": "invalid_type", "churned": 0},  # Invalid contract
        {"customer_id": 93, "age": 30, "monthly_charges": 50.0,
         "contract_type": "invalid_type", "churned": 0},  # Duplicate
    ] + [good_data[0]]  # Another duplicate

    print("(a) Data Pipeline")
    print("=" * 60)

    validator = DataValidator(schema)
    versioner = DataVersioner()

    # Validate good data
    print("\n  Validating clean data:")
    result = validator.validate(good_data)
    print(f"    Valid: {result['valid']}")
    print(f"    Rows: {result['n_rows']}")
    print(f"    Errors: {len(result['errors'])}")
    print(f"    Warnings: {len(result['warnings'])}")

    v1 = versioner.version(good_data, {"source": "batch_20250301", "quality": "passed"})
    print(f"    Version: v{v1['version']} (hash: {v1['hash']})")

    # Validate problematic data
    print("\n  Validating data with issues:")
    result = validator.validate(bad_data)
    print(f"    Valid: {result['valid']}")
    print(f"    Errors ({len(result['errors'])}):")
    for e in result['errors'][:5]:
        print(f"      - {e}")
    print(f"    Warnings ({len(result['warnings'])}):")
    for w in result['warnings'][:5]:
        print(f"      - {w}")

    return validator, versioner


# ============================================================
# Sub-task (b): Training Pipeline with Experiment Tracking
# ============================================================

def subtask_b_training_pipeline():
    """Build a training pipeline with experiment tracking."""

    class ExperimentTracker:
        def __init__(self):
            self.experiments = {}
            self.current_run = None

        def start_run(self, experiment, name):
            self.current_run = {
                "experiment": experiment,
                "name": name,
                "params": {},
                "metrics": {},
                "artifacts": [],
                "start_time": datetime.now(),
            }
            return self.current_run

        def log_params(self, params):
            self.current_run["params"].update(params)

        def log_metrics(self, metrics):
            self.current_run["metrics"].update(metrics)

        def log_artifact(self, name):
            self.current_run["artifacts"].append(name)

        def end_run(self):
            self.current_run["end_time"] = datetime.now()
            exp = self.current_run["experiment"]
            if exp not in self.experiments:
                self.experiments[exp] = []
            self.experiments[exp].append(self.current_run)
            run = self.current_run
            self.current_run = None
            return run

    random.seed(42)
    tracker = ExperimentTracker()

    # Generate data
    data = []
    for _ in range(500):
        x = [random.gauss(0, 1) for _ in range(6)]
        y = 1 if sum(w * xi for w, xi in zip([0.5, -0.3, 0.8, 0.1, -0.6, 0.4], x)) > 0 else 0
        data.append((x, y))

    train_data = data[:400]
    test_data = data[400:]

    print("\n(b) Training Pipeline")
    print("=" * 60)

    configs = [
        {"name": "logreg_baseline", "lr": 0.01, "reg": 0.001, "epochs": 100},
        {"name": "logreg_tuned", "lr": 0.05, "reg": 0.01, "epochs": 200},
        {"name": "logreg_heavy_reg", "lr": 0.01, "reg": 0.1, "epochs": 150},
    ]

    for config in configs:
        run = tracker.start_run("churn_prediction", config["name"])
        tracker.log_params(config)

        # Train
        n_f = 6
        w = [0.0] * n_f
        b = 0.0
        for _ in range(config["epochs"]):
            for x, y in train_data:
                z = max(-500, min(500, sum(wi * xi for wi, xi in zip(w, x)) + b))
                p = 1 / (1 + math.exp(-z))
                e = p - y
                for j in range(n_f):
                    w[j] -= config["lr"] * (e * x[j] + config["reg"] * w[j])
                b -= config["lr"] * e

        # Evaluate
        tp = fp = tn = fn = 0
        for x, y in test_data:
            z = max(-500, min(500, sum(wi * xi for wi, xi in zip(w, x)) + b))
            pred = 1 if (1 / (1 + math.exp(-z))) >= 0.5 else 0
            if pred == 1 and y == 1: tp += 1
            elif pred == 1 and y == 0: fp += 1
            elif pred == 0 and y == 0: tn += 1
            else: fn += 1

        acc = (tp + tn) / (tp + fp + tn + fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        tracker.log_metrics({"accuracy": round(acc, 4), "precision": round(prec, 4),
                             "recall": round(rec, 4), "f1": round(f1, 4)})
        tracker.log_artifact("model_weights.json")
        tracker.end_run()

        print(f"\n  Run: {config['name']}")
        print(f"    Params: lr={config['lr']}, reg={config['reg']}, epochs={config['epochs']}")
        print(f"    Metrics: acc={acc:.4f}, f1={f1:.4f}")

    # Select best
    all_runs = tracker.experiments["churn_prediction"]
    best = max(all_runs, key=lambda r: r["metrics"]["f1"])
    print(f"\n  Best model: {best['name']} (F1={best['metrics']['f1']:.4f})")

    return tracker


# ============================================================
# Sub-task (c): Deployment with CI/CD and Canary Release
# ============================================================

def subtask_c_deployment():
    """Deploy model with CI/CD pipeline and canary release."""

    class CITests:
        @staticmethod
        def run(model_info):
            results = {
                "unit_tests": True,
                "integration_tests": True,
                "accuracy_gate": model_info["metrics"]["accuracy"] >= 0.70,
                "latency_gate": True,
                "security_scan": True,
            }
            return results, all(results.values())

    class CanaryDeployment:
        def __init__(self, current_model, new_model):
            self.current = current_model
            self.new = new_model
            self.traffic_pct = 0
            self.metrics = {"current": [], "new": []}

        def set_traffic(self, pct):
            self.traffic_pct = pct

        def simulate_requests(self, n_requests):
            random.seed(42)
            for _ in range(n_requests):
                if random.random() < self.traffic_pct / 100:
                    latency = 10 + random.gauss(0, 2)
                    error = random.random() < 0.01
                    self.metrics["new"].append({"latency": latency, "error": error})
                else:
                    latency = 12 + random.gauss(0, 3)
                    error = random.random() < 0.02
                    self.metrics["current"].append({"latency": latency, "error": error})

        def evaluate(self):
            def calc_stats(data):
                if not data:
                    return {"avg_latency": 0, "error_rate": 0, "count": 0}
                return {
                    "avg_latency": sum(d["latency"] for d in data) / len(data),
                    "error_rate": sum(1 for d in data if d["error"]) / len(data),
                    "count": len(data),
                }
            return {
                "current": calc_stats(self.metrics["current"]),
                "new": calc_stats(self.metrics["new"]),
            }

    print("\n(c) Deployment with CI/CD")
    print("=" * 60)

    model_info = {
        "name": "churn_predictor_v2",
        "version": 2,
        "metrics": {"accuracy": 0.85, "f1": 0.82},
    }

    # CI/CD
    print("\n  CI/CD Pipeline:")
    results, passed = CITests.run(model_info)
    for test, result in results.items():
        print(f"    [{'PASS' if result else 'FAIL'}] {test}")
    print(f"    Pipeline: {'PASSED' if passed else 'FAILED'}")

    # Canary deployment
    print("\n  Canary Deployment:")
    canary = CanaryDeployment("v1", "v2")

    for pct in [5, 10, 25, 50, 100]:
        canary.set_traffic(pct)
        canary.simulate_requests(200)
        eval_result = canary.evaluate()

        new_stats = eval_result["new"]
        cur_stats = eval_result["current"]
        print(f"    Traffic {pct:>3d}%: "
              f"new(latency={new_stats['avg_latency']:.1f}ms, err={new_stats['error_rate']:.1%}) "
              f"current(latency={cur_stats['avg_latency']:.1f}ms, err={cur_stats['error_rate']:.1%})")

    print(f"    -> Canary successful, promoted v2 to 100% traffic")

    return canary


# ============================================================
# Sub-task (d): Monitoring with Drift Detection
# ============================================================

def subtask_d_monitoring():
    """Set up monitoring with drift detection and alerting."""

    print("\n(d) Monitoring Setup")
    print("=" * 60)

    random.seed(42)
    base_time = datetime(2025, 3, 1)

    alerts = []
    print("\n  24-Hour Monitoring:")
    print(f"  {'Hour':>4s} {'Accuracy':>9s} {'Drift':>7s} {'Latency':>8s} {'Status':>8s}")
    print(f"  {'-'*40}")

    for hour in range(24):
        drift = 0.05 + hour * 0.008 + random.gauss(0, 0.02)
        accuracy = 0.85 - hour * 0.003 + random.gauss(0, 0.01)
        latency = 15 + hour * 0.5 + random.gauss(0, 3)

        status = "OK"
        if drift > 0.15:
            status = "WARN"
            alerts.append(f"Hour {hour}: drift={drift:.3f}")
        if accuracy < 0.80:
            status = "ALERT"
            alerts.append(f"Hour {hour}: accuracy={accuracy:.3f}")

        print(f"  {hour:>4d} {accuracy:>9.4f} {drift:>7.3f} {latency:>7.1f}ms {status:>8s}")

    print(f"\n  Alerts fired: {len(alerts)}")
    for a in alerts[:5]:
        print(f"    - {a}")

    return alerts


# ============================================================
# Sub-task (e): Automated Retraining Loop
# ============================================================

def subtask_e_retraining_loop():
    """Implement automated retraining triggered by monitoring."""

    print("\n(e) Automated Retraining Loop")
    print("=" * 60)

    random.seed(42)

    class RetrainingOrchestrator:
        def __init__(self):
            self.retraining_log = []
            self.current_model_accuracy = 0.85

        def check_trigger(self, metrics):
            triggers = []
            if metrics.get("drift_score", 0) > 0.15:
                triggers.append("data_drift")
            if metrics.get("accuracy", 1) < 0.80:
                triggers.append("accuracy_drop")
            if metrics.get("days_since_training", 0) > 7:
                triggers.append("staleness")
            return triggers

        def retrain(self, trigger_reason):
            # Simulate retraining
            new_accuracy = self.current_model_accuracy + random.gauss(0.02, 0.01)
            new_accuracy = min(0.95, max(0.70, new_accuracy))

            passed = new_accuracy > self.current_model_accuracy * 0.98

            log_entry = {
                "trigger": trigger_reason,
                "old_accuracy": round(self.current_model_accuracy, 4),
                "new_accuracy": round(new_accuracy, 4),
                "deployed": passed,
                "timestamp": datetime.now().isoformat(),
            }
            self.retraining_log.append(log_entry)

            if passed:
                self.current_model_accuracy = new_accuracy
            return log_entry

    orchestrator = RetrainingOrchestrator()

    print("\n  Simulating 30-day retraining loop:")
    print(f"  {'Day':>4s} {'Triggers':>20s} {'Old Acc':>8s} {'New Acc':>8s} {'Deployed':>9s}")
    print(f"  {'-'*55}")

    for day in range(30):
        drift = 0.08 + day * 0.005 + random.gauss(0, 0.02)
        accuracy = orchestrator.current_model_accuracy - day * 0.002 + random.gauss(0, 0.01)

        metrics = {
            "drift_score": max(0, drift),
            "accuracy": max(0, accuracy),
            "days_since_training": day % 10,
        }

        triggers = orchestrator.check_trigger(metrics)
        if triggers:
            result = orchestrator.retrain(triggers)
            print(f"  {day:>4d} {','.join(triggers):>20s} "
                  f"{result['old_accuracy']:>8.4f} {result['new_accuracy']:>8.4f} "
                  f"{'YES' if result['deployed'] else 'NO':>9s}")

    print(f"\n  Total retrainings: {len(orchestrator.retraining_log)}")
    print(f"  Successful deployments: "
          f"{sum(1 for r in orchestrator.retraining_log if r['deployed'])}")
    print(f"  Final accuracy: {orchestrator.current_model_accuracy:.4f}")

    return orchestrator


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Practical MLOps Project — End-to-End Exercise")
    print("=" * 60)

    subtask_a_data_pipeline()
    subtask_b_training_pipeline()
    subtask_c_deployment()
    subtask_d_monitoring()
    subtask_e_retraining_loop()
