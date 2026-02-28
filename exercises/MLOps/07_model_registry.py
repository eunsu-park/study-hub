"""
Exercise Solutions: Model Registry
===========================================
Lesson 07 from MLOps topic.

Exercises
---------
1. Practice Exercise — Implement a comprehensive model registry system with
   7 sub-tasks:
   a. Train 3 model versions with different algorithms
   b. Register all versions in the registry
   c. Implement approval workflow with role-based access
   d. Add CI/CD integration with automated testing
   e. Set up monitoring for the production model
   f. Compare versions across multiple metrics
   g. Promote the best model to production with full audit trail
"""

import math
import random
import json
import hashlib
from datetime import datetime, timedelta


# ============================================================
# Model Registry System (comprehensive implementation)
# ============================================================

class ModelArtifact:
    """Represents a trained model with its metadata."""

    def __init__(self, name, algorithm, weights, metrics, hyperparams, training_data_hash):
        self.name = name
        self.algorithm = algorithm
        self.weights = weights
        self.metrics = metrics
        self.hyperparams = hyperparams
        self.training_data_hash = training_data_hash
        self.created_at = datetime.now()
        # Content-addressable hash for deduplication
        content = json.dumps({
            "weights": str(weights)[:100],
            "algorithm": algorithm,
            "hyperparams": hyperparams,
        }, sort_keys=True)
        self.artifact_hash = hashlib.sha256(content.encode()).hexdigest()[:16]


class ModelVersion:
    """A versioned entry in the model registry."""

    def __init__(self, version, artifact, description=""):
        self.version = version
        self.artifact = artifact
        self.description = description
        self.stage = "None"
        self.tags = {}
        self.annotations = []
        self.created_at = datetime.now()
        self.transition_history = []
        self.test_results = {}

    def add_annotation(self, author, message):
        self.annotations.append({
            "author": author,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        })

    def transition_to(self, new_stage, author, comment=""):
        old_stage = self.stage
        self.stage = new_stage
        self.transition_history.append({
            "from": old_stage,
            "to": new_stage,
            "author": author,
            "comment": comment,
            "timestamp": datetime.now().isoformat(),
        })


class ModelRegistry:
    """Full-featured model registry with approval workflows and CI/CD."""

    def __init__(self):
        self.models = {}  # name -> {versions: [], metadata: {}}
        self.roles = {}   # user -> role
        self.approval_required_transitions = {
            ("Staging", "Production"): ["ml_engineer", "ml_lead"],
            ("Production", "Archived"): ["ml_lead"],
        }

    def add_user(self, username, role):
        """Register a user with a role."""
        self.roles[username] = role

    def create_model(self, name, description="", owner=""):
        """Create a new registered model."""
        self.models[name] = {
            "versions": [],
            "description": description,
            "owner": owner,
            "created_at": datetime.now().isoformat(),
        }

    def register_version(self, model_name, artifact, description=""):
        """Register a new version of a model."""
        model = self.models[model_name]
        version_num = len(model["versions"]) + 1
        version = ModelVersion(version_num, artifact, description)
        model["versions"].append(version)
        return version

    def request_transition(self, model_name, version_num, target_stage, requester, comment=""):
        """Request a stage transition (may require approval)."""
        model = self.models[model_name]
        version = model["versions"][version_num - 1]
        current_stage = version.stage

        transition = (current_stage, target_stage)
        required_roles = self.approval_required_transitions.get(transition)

        if required_roles:
            requester_role = self.roles.get(requester, "unknown")
            if requester_role not in required_roles:
                return {
                    "status": "denied",
                    "reason": f"Role '{requester_role}' cannot approve "
                              f"{current_stage} -> {target_stage}. "
                              f"Required: {required_roles}",
                }

        # Archive current production model if promoting to production
        if target_stage == "Production":
            for v in model["versions"]:
                if v.stage == "Production":
                    v.transition_to("Archived", requester,
                                    f"Replaced by version {version_num}")

        version.transition_to(target_stage, requester, comment)
        return {"status": "approved", "version": version_num, "stage": target_stage}

    def run_ci_tests(self, model_name, version_num, test_suite):
        """Run CI/CD test suite against a model version."""
        version = self.models[model_name]["versions"][version_num - 1]
        results = {}
        all_passed = True

        for test_name, test_fn in test_suite.items():
            try:
                passed = test_fn(version.artifact)
                results[test_name] = {"passed": passed, "error": None}
                if not passed:
                    all_passed = False
            except Exception as e:
                results[test_name] = {"passed": False, "error": str(e)}
                all_passed = False

        version.test_results = results
        version.tags["ci_status"] = "passed" if all_passed else "failed"
        return results, all_passed

    def compare_versions(self, model_name, version_nums, metric_names=None):
        """Compare multiple versions across metrics."""
        model = self.models[model_name]
        versions = [model["versions"][v - 1] for v in version_nums]

        if not metric_names:
            metric_names = list(versions[0].artifact.metrics.keys())

        comparison = {"metrics": {}, "versions": version_nums}
        for metric in metric_names:
            values = []
            for v in versions:
                val = v.artifact.metrics.get(metric, None)
                values.append(val)
            comparison["metrics"][metric] = values

        return comparison

    def get_production_model(self, model_name):
        """Get the current production model version."""
        for v in self.models[model_name]["versions"]:
            if v.stage == "Production":
                return v
        return None


# ============================================================
# Exercise 1: Full Registry Practice
# ============================================================

def exercise_1_model_registry_practice():
    """Implement a comprehensive model registry workflow with 7 sub-tasks."""

    registry = ModelRegistry()

    # Set up users and roles
    registry.add_user("alice", "data_scientist")
    registry.add_user("bob", "ml_engineer")
    registry.add_user("carol", "ml_lead")

    registry.create_model(
        "customer-churn-predictor",
        description="Predicts customer churn probability for subscription service",
        owner="alice",
    )

    print("Model Registry Practice Exercise")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Sub-task (a): Train 3 model versions with different algorithms
    # ------------------------------------------------------------------
    print("\n(a) Training 3 Model Versions")
    print("-" * 40)

    random.seed(42)
    n_samples = 400
    data = []
    for _ in range(n_samples):
        features = [random.gauss(0, 1) for _ in range(8)]
        # Churn influenced by usage patterns
        logit = -1.0 + 0.5 * features[0] - 0.3 * features[1] + 0.8 * features[2]
        prob = 1 / (1 + math.exp(-logit))
        label = 1 if random.random() < prob else 0
        data.append({"features": features, "label": label})

    split = int(n_samples * 0.8)
    train_data, test_data = data[:split], data[split:]

    def train_logistic(train_data, lr=0.01, reg=0.001, epochs=100):
        n_f = len(train_data[0]["features"])
        w = [random.gauss(0, 0.01) for _ in range(n_f)]
        b = 0.0
        for _ in range(epochs):
            for s in train_data:
                z = max(-500, min(500, sum(wi * xi for wi, xi in zip(w, s["features"])) + b))
                p = 1 / (1 + math.exp(-z))
                e = p - s["label"]
                for j in range(n_f):
                    w[j] -= lr * (e * s["features"][j] + reg * w[j])
                b -= lr * e
        return w, b

    def evaluate(weights, bias, test_data, threshold=0.5):
        tp = fp = tn = fn = 0
        for s in test_data:
            z = max(-500, min(500, sum(w * x for w, x in zip(weights, s["features"])) + bias))
            pred = 1 if (1 / (1 + math.exp(-z))) >= threshold else 0
            y = s["label"]
            if pred == 1 and y == 1: tp += 1
            elif pred == 1 and y == 0: fp += 1
            elif pred == 0 and y == 0: tn += 1
            else: fn += 1
        acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return {"accuracy": round(acc, 4), "precision": round(prec, 4),
                "recall": round(rec, 4), "f1": round(f1, 4),
                "latency_ms": round(random.uniform(5, 20), 1)}

    # Version 1: Simple Logistic Regression
    w1, b1 = train_logistic(train_data, lr=0.01, reg=0.001, epochs=50)
    m1 = evaluate(w1, b1, test_data)
    artifact1 = ModelArtifact("churn-v1", "logistic_regression",
                              {"w": w1, "b": b1}, m1,
                              {"lr": 0.01, "reg": 0.001, "epochs": 50},
                              "data_hash_abc123")
    print(f"  V1 (Logistic Regression): F1={m1['f1']:.4f}")

    # Version 2: Regularized Logistic (stronger regularization)
    w2, b2 = train_logistic(train_data, lr=0.01, reg=0.01, epochs=100)
    m2 = evaluate(w2, b2, test_data)
    artifact2 = ModelArtifact("churn-v2", "logistic_regression_l2",
                              {"w": w2, "b": b2}, m2,
                              {"lr": 0.01, "reg": 0.01, "epochs": 100},
                              "data_hash_abc123")
    print(f"  V2 (L2 Regularized):      F1={m2['f1']:.4f}")

    # Version 3: Ensemble (average of two models)
    w3 = [(a + b) / 2 for a, b in zip(w1, w2)]
    b3 = (b1 + b2) / 2
    m3 = evaluate(w3, b3, test_data)
    artifact3 = ModelArtifact("churn-v3", "ensemble_average",
                              {"w": w3, "b": b3}, m3,
                              {"method": "average", "components": ["v1", "v2"]},
                              "data_hash_abc123")
    print(f"  V3 (Ensemble Average):    F1={m3['f1']:.4f}")

    # ------------------------------------------------------------------
    # Sub-task (b): Register all versions
    # ------------------------------------------------------------------
    print("\n(b) Registering Versions")
    print("-" * 40)

    v1 = registry.register_version("customer-churn-predictor", artifact1,
                                   "Baseline logistic regression")
    v2 = registry.register_version("customer-churn-predictor", artifact2,
                                   "L2-regularized logistic regression")
    v3 = registry.register_version("customer-churn-predictor", artifact3,
                                   "Ensemble of v1 and v2")

    for v in [v1, v2, v3]:
        print(f"  Registered version {v.version}: {v.description}")

    # ------------------------------------------------------------------
    # Sub-task (c): Approval workflow with role-based access
    # ------------------------------------------------------------------
    print("\n(c) Approval Workflow")
    print("-" * 40)

    # Data scientist moves to Staging (no approval needed)
    result = registry.request_transition(
        "customer-churn-predictor", 3, "Staging",
        "alice", "Best F1 score, moving to staging for testing"
    )
    print(f"  alice (data_scientist) -> Staging: {result['status']}")

    # Data scientist tries to promote to Production (should be denied)
    result = registry.request_transition(
        "customer-churn-predictor", 3, "Production",
        "alice", "Ready for production"
    )
    print(f"  alice (data_scientist) -> Production: {result['status']}")
    if result["status"] == "denied":
        print(f"    Reason: {result['reason']}")

    # ML Lead approves Production
    result = registry.request_transition(
        "customer-churn-predictor", 3, "Production",
        "carol", "Approved after review of staging metrics"
    )
    print(f"  carol (ml_lead) -> Production: {result['status']}")

    # ------------------------------------------------------------------
    # Sub-task (d): CI/CD integration
    # ------------------------------------------------------------------
    print("\n(d) CI/CD Test Suite")
    print("-" * 40)

    test_suite = {
        "accuracy_threshold": lambda a: a.metrics["accuracy"] >= 0.70,
        "latency_requirement": lambda a: a.metrics["latency_ms"] <= 50,
        "no_nan_weights": lambda a: all(
            not math.isnan(w) for w in a.weights.get("w", [])
        ),
        "artifact_hash_exists": lambda a: len(a.artifact_hash) > 0,
        "training_data_tracked": lambda a: len(a.training_data_hash) > 0,
    }

    for v_num in [1, 2, 3]:
        results, passed = registry.run_ci_tests(
            "customer-churn-predictor", v_num, test_suite
        )
        status = "PASSED" if passed else "FAILED"
        print(f"  Version {v_num}: {status}")
        for test_name, result in results.items():
            mark = "PASS" if result["passed"] else "FAIL"
            print(f"    [{mark}] {test_name}")

    # ------------------------------------------------------------------
    # Sub-task (e): Monitoring for production model
    # ------------------------------------------------------------------
    print("\n(e) Production Model Monitoring")
    print("-" * 40)

    class ModelMonitor:
        def __init__(self, model_version):
            self.model_version = model_version
            self.metrics_history = []
            self.alerts = []

        def record_metrics(self, timestamp, metrics):
            self.metrics_history.append({"timestamp": timestamp, **metrics})
            # Check alert conditions
            if metrics.get("accuracy", 1.0) < 0.65:
                self.alerts.append({
                    "type": "performance_degradation",
                    "timestamp": timestamp,
                    "message": f"Accuracy dropped to {metrics['accuracy']:.4f}",
                })
            if metrics.get("latency_p99_ms", 0) > 100:
                self.alerts.append({
                    "type": "latency_spike",
                    "timestamp": timestamp,
                    "message": f"P99 latency: {metrics['latency_p99_ms']}ms",
                })

    prod_version = registry.get_production_model("customer-churn-predictor")
    monitor = ModelMonitor(prod_version)

    base_time = datetime(2025, 3, 1)
    for hour in range(24):
        ts = (base_time + timedelta(hours=hour)).isoformat()
        # Simulate gradually degrading accuracy
        degradation = hour * 0.005
        monitor.record_metrics(ts, {
            "accuracy": max(0.5, prod_version.artifact.metrics["accuracy"] - degradation
                            + random.gauss(0, 0.02)),
            "latency_p50_ms": 10 + random.gauss(0, 2),
            "latency_p99_ms": 30 + random.gauss(0, 10) + (5 if hour > 18 else 0),
            "requests_per_minute": 100 + random.randint(-20, 50),
        })

    print(f"  Monitored {len(monitor.metrics_history)} hourly checkpoints")
    print(f"  Alerts triggered: {len(monitor.alerts)}")
    for alert in monitor.alerts:
        print(f"    [{alert['type']}] {alert['message']}")

    # ------------------------------------------------------------------
    # Sub-task (f): Compare versions
    # ------------------------------------------------------------------
    print("\n(f) Version Comparison")
    print("-" * 40)

    comparison = registry.compare_versions(
        "customer-churn-predictor", [1, 2, 3],
        ["accuracy", "precision", "recall", "f1", "latency_ms"],
    )

    print(f"  {'Metric':<15s} {'V1':>10s} {'V2':>10s} {'V3':>10s} {'Best':>6s}")
    print(f"  {'-'*50}")
    for metric, values in comparison["metrics"].items():
        best_idx = values.index(max(v for v in values if v is not None))
        if metric == "latency_ms":
            best_idx = values.index(min(v for v in values if v is not None))
        vals_str = [f"{v:>10.4f}" if v is not None else f"{'N/A':>10s}" for v in values]
        print(f"  {metric:<15s} {''.join(vals_str)} {'V' + str(best_idx + 1):>6s}")

    # ------------------------------------------------------------------
    # Sub-task (g): Promote best model with audit trail
    # ------------------------------------------------------------------
    print("\n(g) Audit Trail")
    print("-" * 40)

    prod = registry.get_production_model("customer-churn-predictor")
    if prod:
        print(f"  Current Production: Version {prod.version}")
        print(f"  Algorithm: {prod.artifact.algorithm}")
        print(f"  F1 Score: {prod.artifact.metrics['f1']:.4f}")
        print(f"\n  Full Transition History:")
        for t in prod.transition_history:
            print(f"    {t['from']} -> {t['to']}")
            print(f"      Author: {t['author']}")
            print(f"      Comment: {t['comment']}")
            print(f"      Time: {t['timestamp']}")

    print(f"\n  All Versions State:")
    for v in registry.models["customer-churn-predictor"]["versions"]:
        ci = v.tags.get("ci_status", "not_run")
        print(f"    V{v.version}: stage={v.stage:<12s} ci={ci:<8s} "
              f"f1={v.artifact.metrics['f1']:.4f} ({v.artifact.algorithm})")

    return registry


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Exercise 1: Model Registry Practice")
    print("=" * 60)
    exercise_1_model_registry_practice()
