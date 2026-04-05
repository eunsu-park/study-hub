"""
16. Edge AI Deployment and Monitoring

Demonstrates deployment workflows and runtime monitoring techniques
for edge AI models in production environments.

Covers:
- Model packaging and versioning
- Health checks and readiness probes
- Inference metrics collection (latency, throughput, errors)
- Model drift detection
- A/B testing framework for model updates
- Over-the-air (OTA) model update simulation
- Alerting and anomaly detection

Requirements:
    pip install torch numpy
"""

import torch
import torch.nn as nn
import time
import json
import hashlib
import os
import tempfile
import statistics
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import random

print("=" * 60)
print("Edge AI — Deployment and Monitoring")
print("=" * 60)


# ============================================
# 1. Model Package and Versioning
# ============================================
print("\n[1] Model Packaging and Versioning")
print("-" * 40)


@dataclass
class ModelPackage:
    """Encapsulates a model with metadata for deployment."""
    name: str
    version: str
    model_path: str
    checksum: str
    input_shape: list
    num_classes: int
    framework: str = "pytorch"
    quantized: bool = False
    created_at: str = ""
    size_bytes: int = 0

    def to_manifest(self) -> dict:
        return asdict(self)


class SimpleModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        return self.fc(self.features(x))


def package_model(model, name, version, input_shape, num_classes):
    """Save and package a model with metadata."""
    model_dir = os.path.join(tempfile.gettempdir(), "edge_models")
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, f"{name}_v{version}.pt")
    traced = torch.jit.trace(model.eval(), torch.randn(*input_shape))
    traced.save(model_path)

    # Compute checksum
    with open(model_path, "rb") as f:
        checksum = hashlib.sha256(f.read()).hexdigest()[:16]

    pkg = ModelPackage(
        name=name,
        version=version,
        model_path=model_path,
        checksum=checksum,
        input_shape=list(input_shape),
        num_classes=num_classes,
        created_at=datetime.now().isoformat(),
        size_bytes=os.path.getsize(model_path),
    )
    # Save manifest
    manifest_path = model_path.replace(".pt", "_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(pkg.to_manifest(), f, indent=2)

    return pkg


model_v1 = SimpleModel(num_classes=10)
pkg = package_model(model_v1, "edge_classifier", "1.0.0", (1, 3, 64, 64), 10)

print(f"Model: {pkg.name} v{pkg.version}")
print(f"Size: {pkg.size_bytes / 1024:.1f} KB")
print(f"Checksum: {pkg.checksum}")
print(f"Path: {pkg.model_path}")


# ============================================
# 2. Health Check and Readiness Probe
# ============================================
print("\n[2] Health Checks and Readiness Probes")
print("-" * 40)


class ModelHealthChecker:
    """Monitors model health and readiness on an edge device."""

    def __init__(self, model, input_shape, max_latency_ms=100):
        self.model = model
        self.input_shape = input_shape
        self.max_latency_ms = max_latency_ms
        self.is_loaded = False
        self.last_inference_time = None

    def load_model(self):
        """Simulate model loading and warmup."""
        self.model.eval()
        # Warmup inference
        dummy = torch.randn(*self.input_shape)
        with torch.no_grad():
            self.model(dummy)
        self.is_loaded = True

    def liveness_check(self) -> dict:
        """Check if the model process is alive."""
        return {
            "status": "alive",
            "model_loaded": self.is_loaded,
            "timestamp": datetime.now().isoformat(),
        }

    def readiness_check(self) -> dict:
        """Check if the model is ready to serve inference."""
        if not self.is_loaded:
            return {"status": "not_ready", "reason": "model not loaded"}

        # Test inference
        dummy = torch.randn(*self.input_shape)
        start = time.perf_counter()
        with torch.no_grad():
            self.model(dummy)
        latency_ms = (time.perf_counter() - start) * 1000

        ready = latency_ms < self.max_latency_ms
        self.last_inference_time = datetime.now()

        return {
            "status": "ready" if ready else "degraded",
            "latency_ms": round(latency_ms, 2),
            "max_latency_ms": self.max_latency_ms,
            "timestamp": datetime.now().isoformat(),
        }


health = ModelHealthChecker(model_v1, (1, 3, 64, 64), max_latency_ms=50)

liveness = health.liveness_check()
print(f"Liveness: {liveness['status']}, loaded={liveness['model_loaded']}")

health.load_model()
readiness = health.readiness_check()
print(f"Readiness: {readiness['status']}, latency={readiness['latency_ms']} ms")


# ============================================
# 3. Inference Metrics Collector
# ============================================
print("\n[3] Inference Metrics Collection")
print("-" * 40)


class MetricsCollector:
    """Collect and aggregate inference metrics."""

    def __init__(self, window_size=100):
        self.latencies = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        self.total_inferences = 0
        self.total_errors = 0
        self.start_time = time.time()

    def record_inference(self, latency_ms: float, prediction: int,
                         error: bool = False):
        self.latencies.append(latency_ms)
        self.predictions.append(prediction)
        self.errors.append(error)
        self.total_inferences += 1
        if error:
            self.total_errors += 1

    def get_summary(self) -> dict:
        if not self.latencies:
            return {"status": "no data"}

        uptime_s = time.time() - self.start_time
        lat_list = list(self.latencies)

        return {
            "total_inferences": self.total_inferences,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / max(self.total_inferences, 1),
            "throughput_qps": self.total_inferences / max(uptime_s, 0.01),
            "latency_mean_ms": statistics.mean(lat_list),
            "latency_p50_ms": statistics.median(lat_list),
            "latency_p95_ms": sorted(lat_list)[int(0.95 * len(lat_list))],
            "latency_p99_ms": sorted(lat_list)[int(0.99 * len(lat_list))],
            "uptime_seconds": round(uptime_s, 1),
        }

    def get_class_distribution(self) -> dict:
        if not self.predictions:
            return {}
        counts = {}
        for p in self.predictions:
            counts[p] = counts.get(p, 0) + 1
        total = len(self.predictions)
        return {k: round(v / total, 3) for k, v in sorted(counts.items())}


# Simulate inference workload
metrics = MetricsCollector(window_size=200)
model_v1.eval()

for i in range(200):
    x = torch.randn(1, 3, 64, 64)
    start = time.perf_counter()
    try:
        with torch.no_grad():
            out = model_v1(x)
            pred = out.argmax(1).item()
        latency = (time.perf_counter() - start) * 1000
        metrics.record_inference(latency, pred, error=False)
    except Exception:
        metrics.record_inference(0, -1, error=True)

summary = metrics.get_summary()
dist = metrics.get_class_distribution()

print(f"Total inferences: {summary['total_inferences']}")
print(f"Error rate: {summary['error_rate']:.3f}")
print(f"Latency (mean): {summary['latency_mean_ms']:.2f} ms")
print(f"Latency (P95):  {summary['latency_p95_ms']:.2f} ms")
print(f"Latency (P99):  {summary['latency_p99_ms']:.2f} ms")
print(f"Class distribution: {dist}")


# ============================================
# 4. Model Drift Detection
# ============================================
print("\n[4] Model Drift Detection")
print("-" * 40)
print("Detect when input distribution shifts from training data.\n")


class DriftDetector:
    """Simple statistical drift detector for edge deployment."""

    def __init__(self, reference_mean: float, reference_std: float,
                 window_size: int = 50, threshold_sigma: float = 3.0):
        self.ref_mean = reference_mean
        self.ref_std = reference_std
        self.window = deque(maxlen=window_size)
        self.threshold = threshold_sigma
        self.drift_count = 0

    def update(self, feature_value: float) -> bool:
        """Add observation and check for drift."""
        self.window.append(feature_value)
        if len(self.window) < 10:
            return False

        window_mean = statistics.mean(self.window)
        z_score = abs(window_mean - self.ref_mean) / (self.ref_std + 1e-8)
        is_drift = z_score > self.threshold
        if is_drift:
            self.drift_count += 1
        return is_drift

    def status(self) -> dict:
        if len(self.window) < 10:
            return {"status": "insufficient_data"}
        window_mean = statistics.mean(self.window)
        z = abs(window_mean - self.ref_mean) / (self.ref_std + 1e-8)
        return {
            "ref_mean": round(self.ref_mean, 4),
            "window_mean": round(window_mean, 4),
            "z_score": round(z, 4),
            "drift_detected": z > self.threshold,
            "drift_events": self.drift_count,
        }


# Simulate: reference data has mean~0, then input shifts to mean~0.5
detector = DriftDetector(reference_mean=0.0, reference_std=1.0,
                         window_size=30, threshold_sigma=2.5)

# Normal data
for _ in range(40):
    detector.update(random.gauss(0.0, 1.0))
print(f"After normal inputs: {detector.status()}")

# Drifted data
for _ in range(40):
    detector.update(random.gauss(2.0, 1.0))
print(f"After drifted inputs: {detector.status()}")


# ============================================
# 5. A/B Testing Framework
# ============================================
print("\n[5] A/B Testing for Model Updates")
print("-" * 40)


class ABTestRunner:
    """Run A/B tests between two model versions on edge."""

    def __init__(self, model_a, model_b, traffic_split=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.results_a = {"latencies": [], "predictions": []}
        self.results_b = {"latencies": [], "predictions": []}

    def predict(self, x: torch.Tensor) -> tuple:
        """Route to model A or B based on traffic split."""
        use_b = random.random() < self.traffic_split

        model = self.model_b if use_b else self.model_a
        results = self.results_b if use_b else self.results_a
        variant = "B" if use_b else "A"

        start = time.perf_counter()
        with torch.no_grad():
            out = model(x)
        latency = (time.perf_counter() - start) * 1000

        pred = out.argmax(1).item()
        results["latencies"].append(latency)
        results["predictions"].append(pred)

        return pred, variant

    def summary(self) -> dict:
        def stats(r):
            if not r["latencies"]:
                return {}
            return {
                "count": len(r["latencies"]),
                "mean_latency_ms": round(statistics.mean(r["latencies"]), 2),
                "unique_classes": len(set(r["predictions"])),
            }
        return {"model_a": stats(self.results_a), "model_b": stats(self.results_b)}


# Create two model versions
model_a = SimpleModel(num_classes=10)
model_a.eval()
model_b = SimpleModel(num_classes=10)  # Different random init = "updated" model
model_b.eval()

ab_test = ABTestRunner(model_a, model_b, traffic_split=0.3)

for _ in range(100):
    x = torch.randn(1, 3, 64, 64)
    ab_test.predict(x)

ab_summary = ab_test.summary()
print(f"Model A: {ab_summary['model_a']}")
print(f"Model B: {ab_summary['model_b']}")
print(f"Traffic split: {(1 - ab_test.traffic_split)*100:.0f}% A / "
      f"{ab_test.traffic_split*100:.0f}% B")


# ============================================
# 6. OTA Model Update Simulation
# ============================================
print("\n[6] Over-the-Air (OTA) Model Update")
print("-" * 40)


class OTAUpdateManager:
    """Manage model updates on an edge device."""

    def __init__(self, current_package: ModelPackage):
        self.current = current_package
        self.update_history = []
        self.rollback_path = None

    def check_update(self, server_manifest: dict) -> bool:
        """Check if a newer version is available."""
        return server_manifest["version"] > self.current.version

    def apply_update(self, new_package: ModelPackage) -> dict:
        """Apply a model update with rollback capability."""
        # Backup current model
        self.rollback_path = self.current.model_path

        # Validate checksum
        if not os.path.exists(new_package.model_path):
            return {"status": "error", "reason": "model file not found"}

        with open(new_package.model_path, "rb") as f:
            actual_checksum = hashlib.sha256(f.read()).hexdigest()[:16]

        if actual_checksum != new_package.checksum:
            return {"status": "error", "reason": "checksum mismatch"}

        # Validate model loads and runs
        try:
            loaded = torch.jit.load(new_package.model_path)
            dummy = torch.randn(*new_package.input_shape)
            with torch.no_grad():
                loaded(dummy)
        except Exception as e:
            return {"status": "error", "reason": f"validation failed: {e}"}

        # Apply update
        old_version = self.current.version
        self.current = new_package
        self.update_history.append({
            "from": old_version,
            "to": new_package.version,
            "timestamp": datetime.now().isoformat(),
        })

        return {"status": "success", "version": new_package.version}

    def rollback(self) -> dict:
        """Rollback to previous model version."""
        if not self.rollback_path:
            return {"status": "error", "reason": "no rollback available"}
        return {"status": "rolled_back", "path": self.rollback_path}


# Simulate OTA update
model_v2 = SimpleModel(num_classes=10)
pkg_v2 = package_model(model_v2, "edge_classifier", "2.0.0", (1, 3, 64, 64), 10)

ota = OTAUpdateManager(pkg)
server_manifest = pkg_v2.to_manifest()

has_update = ota.check_update(server_manifest)
print(f"Current version: {pkg.version}")
print(f"Update available: {has_update} (server has v{server_manifest['version']})")

result = ota.apply_update(pkg_v2)
print(f"Update result: {result['status']}")
print(f"Now running: v{ota.current.version}")
print(f"Update history: {len(ota.update_history)} updates applied")


# ============================================
# 7. Anomaly Alerting
# ============================================
print("\n[7] Anomaly Detection and Alerting")
print("-" * 40)


class AlertManager:
    """Monitor metrics and raise alerts when thresholds are breached."""

    def __init__(self):
        self.rules = []
        self.alerts = []

    def add_rule(self, name: str, metric_fn, threshold: float,
                 comparison: str = "gt", cooldown_s: float = 60.0):
        self.rules.append({
            "name": name,
            "metric_fn": metric_fn,
            "threshold": threshold,
            "comparison": comparison,
            "cooldown_s": cooldown_s,
            "last_alert": 0,
        })

    def check(self) -> List[dict]:
        """Evaluate all rules and return triggered alerts."""
        new_alerts = []
        now = time.time()

        for rule in self.rules:
            value = rule["metric_fn"]()
            triggered = False

            if rule["comparison"] == "gt" and value > rule["threshold"]:
                triggered = True
            elif rule["comparison"] == "lt" and value < rule["threshold"]:
                triggered = True

            if triggered and (now - rule["last_alert"]) > rule["cooldown_s"]:
                alert = {
                    "rule": rule["name"],
                    "value": round(value, 4),
                    "threshold": rule["threshold"],
                    "timestamp": datetime.now().isoformat(),
                }
                new_alerts.append(alert)
                self.alerts.append(alert)
                rule["last_alert"] = now

        return new_alerts


# Set up alerting
alert_mgr = AlertManager()

# Add rules based on metrics collector
alert_mgr.add_rule(
    "high_latency",
    lambda: summary["latency_p95_ms"],
    threshold=50.0,
    comparison="gt",
    cooldown_s=0,
)
alert_mgr.add_rule(
    "high_error_rate",
    lambda: summary["error_rate"],
    threshold=0.05,
    comparison="gt",
    cooldown_s=0,
)
alert_mgr.add_rule(
    "low_throughput",
    lambda: summary["throughput_qps"],
    threshold=1.0,
    comparison="lt",
    cooldown_s=0,
)

alerts = alert_mgr.check()
print(f"Active rules: {len(alert_mgr.rules)}")
print(f"Triggered alerts: {len(alerts)}")
for alert in alerts:
    print(f"  [{alert['rule']}] value={alert['value']} "
          f"(threshold={alert['threshold']})")


# ============================================
# 8. Deployment Checklist
# ============================================
print("\n[8] Edge Deployment Checklist")
print("-" * 40)

checklist = [
    ("Model packaged with version and checksum", True),
    ("Health check endpoints implemented", True),
    ("Metrics collection (latency, errors, throughput)", True),
    ("Drift detection configured", True),
    ("A/B testing framework ready", True),
    ("OTA update mechanism with rollback", True),
    ("Alerting rules defined", True),
    ("Logging and diagnostics enabled", True),
]

for item, done in checklist:
    status = "[x]" if done else "[ ]"
    print(f"  {status} {item}")

print(f"\nAll {sum(d for _, d in checklist)}/{len(checklist)} items completed.")

# Cleanup
model_dir = os.path.join(tempfile.gettempdir(), "edge_models")
if os.path.exists(model_dir):
    for f in os.listdir(model_dir):
        os.remove(os.path.join(model_dir, f))
    os.rmdir(model_dir)


# ============================================
# 9. Summary
# ============================================
print("\n[9] Summary")
print("-" * 40)
print("Key takeaways:")
print("- Package models with version, checksum, and manifest for traceability")
print("- Implement liveness and readiness probes for reliable serving")
print("- Collect P50/P95/P99 latency, not just mean, for SLA monitoring")
print("- Drift detection catches distribution shift before accuracy degrades")
print("- A/B testing enables safe rollout of model updates on edge fleets")
print("- OTA updates with validation and rollback prevent bricked devices")
