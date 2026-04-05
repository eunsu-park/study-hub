# 16. 배포와 모니터링

**이전**: [NLP를 위한 Edge AI](./15_Edge_AI_for_NLP.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 롤백 안전성과 무결성 검증을 갖춘 OTA(over-the-air) 모델 업데이트를 구현할 수 있다
2. 재현 가능한 엣지 배포를 위한 모델 버전 관리 체계를 설계할 수 있다
3. 프로덕션에서 모델 성능을 비교하기 위해 엣지 디바이스에서 A/B testing을 설정할 수 있다
4. 배포된 모델의 데이터 드리프트, 성능 저하, 이상 징후를 모니터링할 수 있다
5. 중앙 집중식 오케스트레이션으로 엣지 AI 디바이스 플릿을 관리할 수 있다
6. 훈련부터 프로덕션 모니터링까지 엔드투엔드 엣지 MLOps 파이프라인을 구축할 수 있다

---

하나의 디바이스에 모델을 배포하는 것은 엔지니어링입니다. 수천 대의 디바이스에 모델을 배포하고 원활하게 유지하는 것은 운영입니다. 엣지 MLOps는 클라우드 MLOps보다 어렵습니다. 모든 디바이스에 SSH를 할 수 없고, 업데이트는 불안정한 네트워크에서도 작동해야 하며, 잘못된 모델 배포는 물리적으로 접근할 수 없는 위치의 디바이스를 무력화시킬 수 있기 때문입니다. 이 레슨에서는 엣지 AI를 대규모로 안정적으로 만드는 운영 실무를 다룹니다: 안전한 모델 업데이트, 버전 관리, 통제된 롤아웃, 지속적 모니터링, 플릿 오케스트레이션.

---

## 1. OTA 모델 업데이트

### 1.1 OTA 업데이트 아키텍처

```
+-----------------------------------------------------------------+
|              OTA Model Update Pipeline                            |
+-----------------------------------------------------------------+
|                                                                   |
|   Cloud (Update Server)                                          |
|   +---------------------------------------------------------+   |
|   | 1. New model trained and validated                        |   |
|   | 2. Package: model + metadata + checksum                   |   |
|   | 3. Upload to CDN / artifact registry                      |   |
|   | 4. Push update manifest to devices                        |   |
|   +---------------------------------------------------------+   |
|        |                                                         |
|        v (HTTPS + signed manifest)                               |
|                                                                   |
|   Edge Device                                                    |
|   +---------------------------------------------------------+   |
|   | 5. Download model to staging partition                    |   |
|   | 6. Verify checksum + signature                            |   |
|   | 7. Run validation inference (sanity check)                |   |
|   | 8. Atomic swap: staging -> active                          |   |
|   | 9. Report status to cloud                                 |   |
|   +---------------------------------------------------------+   |
|                                                                   |
|   Rollback safety:                                               |
|   - Keep previous model on device (at least N-1)                 |
|   - If new model fails validation -> auto-rollback               |
|   - If new model degrades metrics -> flag for review             |
|                                                                   |
+-----------------------------------------------------------------+
```

### 1.2 OTA 업데이트 매니저

```python
#!/usr/bin/env python3
"""OTA model update manager for edge devices."""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import urllib.request


@dataclass
class ModelManifest:
    """Metadata for a model update package."""
    model_id: str
    version: str
    url: str
    sha256: str
    size_bytes: int
    min_runtime_version: str
    release_notes: str
    rollout_percentage: float  # 0.0 - 1.0
    created_at: str


class OTAUpdateManager:
    """Manages safe OTA model updates on edge devices.

    Update flow:
    1. Check for updates (poll or push notification)
    2. Download to staging area
    3. Verify integrity (SHA-256)
    4. Run validation inference
    5. Atomic swap to active slot
    6. Report success/failure to cloud

    Safety features:
    - Two-slot system (active + staging) prevents bricking
    - SHA-256 verification prevents corrupted models
    - Validation inference catches incompatible models
    - Automatic rollback on failure
    """

    def __init__(self, models_dir: str, device_id: str,
                 update_server: str = "https://models.example.com"):
        self.models_dir = Path(models_dir)
        self.device_id = device_id
        self.update_server = update_server

        # Create directory structure
        self.active_dir = self.models_dir / "active"
        self.staging_dir = self.models_dir / "staging"
        self.rollback_dir = self.models_dir / "rollback"

        for d in [self.active_dir, self.staging_dir, self.rollback_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.models_dir / "manifest.json"

    def check_for_updates(self) -> Optional[ModelManifest]:
        """Check if a new model version is available."""
        try:
            url = f"{self.update_server}/api/updates?device={self.device_id}"
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read())

            if not data.get("update_available"):
                return None

            manifest = ModelManifest(**data["manifest"])

            # Check rollout eligibility
            device_hash = int(hashlib.md5(
                self.device_id.encode()
            ).hexdigest()[:8], 16) / 0xFFFFFFFF

            if device_hash > manifest.rollout_percentage:
                print(f"Device not in rollout group "
                      f"({manifest.rollout_percentage * 100:.0f}%)")
                return None

            return manifest

        except Exception as e:
            print(f"Update check failed: {e}")
            return None

    def download_model(self, manifest: ModelManifest) -> bool:
        """Download model to staging directory."""
        staging_path = self.staging_dir / f"{manifest.model_id}.tflite"

        try:
            print(f"Downloading {manifest.model_id} v{manifest.version}...")
            urllib.request.urlretrieve(manifest.url, staging_path)

            # Verify checksum
            sha256 = self._compute_sha256(staging_path)
            if sha256 != manifest.sha256:
                print(f"Checksum mismatch: expected {manifest.sha256[:16]}..., "
                      f"got {sha256[:16]}...")
                staging_path.unlink()
                return False

            # Save manifest
            manifest_path = self.staging_dir / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(asdict(manifest), f, indent=2)

            print(f"Download verified: {staging_path}")
            return True

        except Exception as e:
            print(f"Download failed: {e}")
            if staging_path.exists():
                staging_path.unlink()
            return False

    def validate_model(self, manifest: ModelManifest,
                       validation_fn=None) -> bool:
        """Run validation inference on the staged model.

        The validation function should:
        1. Load the model and check it runs without errors
        2. Run inference on known test inputs
        3. Check that output shape and range are reasonable
        4. Optionally compare accuracy with the active model
        """
        staging_path = self.staging_dir / f"{manifest.model_id}.tflite"

        if validation_fn is None:
            # Default: check model loads and runs
            try:
                from tflite_runtime.interpreter import Interpreter
                import numpy as np

                interp = Interpreter(str(staging_path))
                interp.allocate_tensors()

                inp = interp.get_input_details()
                out = interp.get_output_details()

                dummy = np.random.randn(
                    *inp[0]["shape"]
                ).astype(inp[0]["dtype"])
                interp.set_tensor(inp[0]["index"], dummy)
                interp.invoke()

                output = interp.get_tensor(out[0]["index"])
                if np.isnan(output).any() or np.isinf(output).any():
                    print("Validation failed: NaN/Inf in output")
                    return False

                print("Validation passed: model loads and produces valid output")
                return True

            except Exception as e:
                print(f"Validation failed: {e}")
                return False
        else:
            return validation_fn(str(staging_path))

    def activate_model(self, manifest: ModelManifest) -> bool:
        """Atomic swap: move staging model to active slot."""
        try:
            model_name = f"{manifest.model_id}.tflite"
            staging_path = self.staging_dir / model_name
            active_path = self.active_dir / model_name
            rollback_path = self.rollback_dir / model_name

            # Backup current active model for rollback
            if active_path.exists():
                shutil.copy2(active_path, rollback_path)

            # Atomic move (on same filesystem, this is rename = atomic)
            shutil.move(str(staging_path), str(active_path))

            # Update manifest
            with open(self.manifest_path, "w") as f:
                json.dump(asdict(manifest), f, indent=2)

            print(f"Activated: {manifest.model_id} v{manifest.version}")
            return True

        except Exception as e:
            print(f"Activation failed: {e}")
            return False

    def rollback(self, manifest: ModelManifest) -> bool:
        """Rollback to previous model version."""
        model_name = f"{manifest.model_id}.tflite"
        rollback_path = self.rollback_dir / model_name
        active_path = self.active_dir / model_name

        if not rollback_path.exists():
            print("No rollback model available")
            return False

        shutil.copy2(rollback_path, active_path)
        print(f"Rolled back {manifest.model_id} to previous version")
        return True

    def perform_update(self, validation_fn=None) -> bool:
        """Full update cycle: check -> download -> validate -> activate."""
        manifest = self.check_for_updates()
        if manifest is None:
            print("No updates available")
            return False

        if not self.download_model(manifest):
            return False

        if not self.validate_model(manifest, validation_fn):
            print("Validation failed, aborting update")
            return False

        if not self.activate_model(manifest):
            return False

        self._report_status(manifest, success=True)
        return True

    def _compute_sha256(self, filepath: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _report_status(self, manifest: ModelManifest, success: bool):
        """Report update status back to the server."""
        payload = {
            "device_id": self.device_id,
            "model_id": manifest.model_id,
            "version": manifest.version,
            "success": success,
            "timestamp": time.time(),
        }
        # In production: POST to update_server/api/status
        print(f"Reported status: {'success' if success else 'failure'}")
```

---

## 2. 모델 버전 관리

### 2.1 버전 관리 체계

```
+-----------------------------------------------------------------+
|              Model Versioning Best Practices                     |
+-----------------------------------------------------------------+
|                                                                   |
|   Version format: <model_name>-v<major>.<minor>.<patch>-<quant> |
|                                                                   |
|   Examples:                                                      |
|   - detector-v2.1.0-fp32.tflite   (base model)                  |
|   - detector-v2.1.0-int8.tflite   (quantized variant)           |
|   - detector-v2.1.0-int8-edgetpu.tflite  (compiled for Coral)   |
|                                                                   |
|   Major: Architecture change (incompatible I/O)                  |
|   Minor: Retrained with new data (same I/O, better accuracy)    |
|   Patch: Quantization/optimization change (same weights)         |
|                                                                   |
|   Metadata to track per version:                                 |
|   - Training data hash                                           |
|   - Training hyperparameters                                     |
|   - Accuracy metrics (on standard benchmark)                     |
|   - Input/output specifications                                  |
|   - Target hardware + framework versions                         |
|   - File hash (SHA-256)                                          |
|                                                                   |
+-----------------------------------------------------------------+
```

### 2.2 모델 레지스트리

```python
#!/usr/bin/env python3
"""Model registry for edge AI version management."""

import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional


@dataclass
class ModelVersion:
    """Complete metadata for a model version."""
    model_name: str
    version: str                      # Semantic version: major.minor.patch
    framework: str                    # "tflite", "onnx", "executorch"
    quantization: str                 # "fp32", "fp16", "int8", "q4_k_m"
    input_spec: dict                  # {"shape": [1,224,224,3], "dtype": "float32"}
    output_spec: dict                 # {"shape": [1,10], "dtype": "float32"}
    metrics: dict                     # {"accuracy": 0.95, "mAP": 0.42, ...}
    file_hash: str                    # SHA-256 of model file
    file_size_bytes: int
    training_config: dict = field(default_factory=dict)
    target_hardware: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    notes: str = ""


class ModelRegistry:
    """Local model registry for tracking deployed versions."""

    def __init__(self, registry_dir: str):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "registry.json"

        self.models = self._load_registry()

    def _load_registry(self) -> dict:
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        with open(self.registry_file, "w") as f:
            json.dump(self.models, f, indent=2)

    def register(self, model_path: str, version: ModelVersion):
        """Register a new model version."""
        key = f"{version.model_name}/{version.version}/{version.quantization}"

        # Compute file hash if not provided
        if not version.file_hash:
            version.file_hash = self._hash_file(model_path)
            version.file_size_bytes = os.path.getsize(model_path)

        self.models[key] = asdict(version)
        self._save_registry()

        print(f"Registered: {key}")
        print(f"  Metrics: {version.metrics}")
        print(f"  Size: {version.file_size_bytes / 1024:.1f} KB")

    def get_version(self, model_name: str, version: str,
                    quantization: str = "int8") -> Optional[dict]:
        """Look up a specific model version."""
        key = f"{model_name}/{version}/{quantization}"
        return self.models.get(key)

    def get_latest(self, model_name: str,
                   quantization: str = "int8") -> Optional[dict]:
        """Get the latest version of a model."""
        matching = [
            (k, v) for k, v in self.models.items()
            if k.startswith(f"{model_name}/") and k.endswith(f"/{quantization}")
        ]

        if not matching:
            return None

        # Sort by version (semantic versioning)
        matching.sort(key=lambda x: [
            int(p) for p in x[1]["version"].split(".")
        ])

        return matching[-1][1]

    def compare_versions(self, model_name: str,
                         v1: str, v2: str) -> dict:
        """Compare metrics between two versions."""
        ver1 = self.get_version(model_name, v1)
        ver2 = self.get_version(model_name, v2)

        if not ver1 or not ver2:
            return {"error": "Version not found"}

        comparison = {"v1": v1, "v2": v2, "metrics_diff": {}}
        for key in ver1.get("metrics", {}):
            if key in ver2.get("metrics", {}):
                diff = ver2["metrics"][key] - ver1["metrics"][key]
                comparison["metrics_diff"][key] = {
                    "v1": ver1["metrics"][key],
                    "v2": ver2["metrics"][key],
                    "diff": round(diff, 4),
                    "improved": diff > 0,
                }

        return comparison

    def list_versions(self, model_name: str) -> list:
        """List all versions of a model."""
        return [
            v for k, v in self.models.items()
            if k.startswith(f"{model_name}/")
        ]

    @staticmethod
    def _hash_file(filepath: str) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
```

---

## 3. 엣지에서의 A/B Testing

### 3.1 엣지 A/B Testing 프레임워크

```python
#!/usr/bin/env python3
"""A/B testing framework for edge AI models."""

import hashlib
import json
import time
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from pathlib import Path


@dataclass
class ABExperiment:
    """Defines an A/B test experiment."""
    experiment_id: str
    model_a_path: str          # Control model
    model_b_path: str          # Treatment model
    traffic_split: float       # Fraction of traffic to model B (0.0-1.0)
    metrics_to_track: list     # ["latency_ms", "accuracy", "confidence"]
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "running"    # "running", "completed", "stopped"


class EdgeABTester:
    """A/B testing for edge AI models.

    Uses deterministic assignment: each input gets assigned to the
    same model variant consistently (based on a hash), enabling
    fair comparison even with non-uniform input distributions.
    """

    def __init__(self, experiment: ABExperiment, model_loader):
        self.experiment = experiment
        self.model_a = model_loader(experiment.model_a_path)
        self.model_b = model_loader(experiment.model_b_path)

        self.metrics_a = {m: [] for m in experiment.metrics_to_track}
        self.metrics_b = {m: [] for m in experiment.metrics_to_track}

        self.total_a = 0
        self.total_b = 0

    def assign_variant(self, request_id: str) -> str:
        """Deterministically assign a request to A or B.

        Deterministic assignment ensures the same input always goes
        to the same variant, preventing confounding from repeated
        inputs being split across variants.
        """
        hash_val = int(
            hashlib.sha256(request_id.encode()).hexdigest()[:8], 16
        ) / 0xFFFFFFFF

        return "B" if hash_val < self.experiment.traffic_split else "A"

    def run_inference(self, input_data, request_id: str) -> dict:
        """Route inference to the assigned variant and record metrics."""
        variant = self.assign_variant(request_id)

        model = self.model_a if variant == "A" else self.model_b
        metrics_store = self.metrics_a if variant == "A" else self.metrics_b

        start = time.perf_counter()
        result = model.predict(input_data)
        latency_ms = (time.perf_counter() - start) * 1000

        # Record metrics
        if "latency_ms" in metrics_store:
            metrics_store["latency_ms"].append(latency_ms)

        if variant == "A":
            self.total_a += 1
        else:
            self.total_b += 1

        return {
            "result": result,
            "variant": variant,
            "latency_ms": latency_ms,
        }

    def get_results(self) -> dict:
        """Compute A/B test results with statistical summary."""
        import numpy as np

        results = {"experiment_id": self.experiment.experiment_id}

        for metric in self.experiment.metrics_to_track:
            a_vals = np.array(self.metrics_a.get(metric, []))
            b_vals = np.array(self.metrics_b.get(metric, []))

            if len(a_vals) == 0 or len(b_vals) == 0:
                continue

            results[metric] = {
                "A": {
                    "count": len(a_vals),
                    "mean": round(float(np.mean(a_vals)), 4),
                    "std": round(float(np.std(a_vals)), 4),
                    "p50": round(float(np.percentile(a_vals, 50)), 4),
                    "p95": round(float(np.percentile(a_vals, 95)), 4),
                },
                "B": {
                    "count": len(b_vals),
                    "mean": round(float(np.mean(b_vals)), 4),
                    "std": round(float(np.std(b_vals)), 4),
                    "p50": round(float(np.percentile(b_vals, 50)), 4),
                    "p95": round(float(np.percentile(b_vals, 95)), 4),
                },
                "diff_pct": round(
                    (np.mean(b_vals) - np.mean(a_vals)) / np.mean(a_vals) * 100,
                    2
                ),
            }

        results["total_samples"] = {
            "A": self.total_a,
            "B": self.total_b,
        }

        return results

    def should_stop_early(self, min_samples: int = 100,
                          significance_threshold: float = 0.05) -> bool:
        """Check if the experiment has enough data to conclude.

        Uses a simple effect-size check (not a full statistical test).
        For production, use proper sequential testing (e.g., CUPED or
        Bayesian approaches) to avoid peeking problems.
        """
        if self.total_a < min_samples or self.total_b < min_samples:
            return False

        results = self.get_results()
        for metric in self.experiment.metrics_to_track:
            if metric in results:
                diff_pct = abs(results[metric].get("diff_pct", 0))
                if diff_pct > 5:  # >5% difference
                    return True

        return False
```

---

## 4. 엣지 모니터링

### 4.1 모니터링 아키텍처

```
+-----------------------------------------------------------------+
|            Edge AI Monitoring System                              |
+-----------------------------------------------------------------+
|                                                                   |
|   Edge Device                        Cloud Dashboard             |
|   +---------------------------+     +---------------------+     |
|   | Model Inference           |     | Aggregated Metrics  |     |
|   |   |                      |     |                     |     |
|   |   v                      |     | - Fleet health      |     |
|   | Metric Collector         |---->| - Drift alerts      |     |
|   |   - Latency              |     | - Performance trends|     |
|   |   - Throughput           |     | - Error rates       |     |
|   |   - Confidence dist.    |     | - Model comparison  |     |
|   |   - Input statistics    |     |                     |     |
|   |   - Error counts        |     +---------------------+     |
|   |   |                      |                                  |
|   |   v                      |                                  |
|   | Drift Detector           |                                  |
|   |   - Feature drift        |                                  |
|   |   - Prediction drift     |                                  |
|   |   - Confidence drift     |                                  |
|   |   |                      |                                  |
|   |   v                      |                                  |
|   | Local Alert Engine       |                                  |
|   |   - Threshold alerts     |                                  |
|   |   - Anomaly detection    |                                  |
|   +---------------------------+                                  |
|                                                                   |
+-----------------------------------------------------------------+
```

### 4.2 드리프트 탐지

```python
#!/usr/bin/env python3
"""Data and model drift detection for edge AI."""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class DriftAlert:
    """Alert when drift is detected."""
    drift_type: str        # "feature", "prediction", "confidence"
    metric_name: str
    baseline_value: float
    current_value: float
    severity: str          # "warning", "critical"
    timestamp: float


class DriftDetector:
    """Detect distribution shifts in input features and model outputs.

    Why drift matters on edge:
    - Environment changes (lighting, weather, new object types)
    - Sensor degradation (camera blur, microphone noise)
    - Population shift (new user demographics)
    - Adversarial inputs (attempted attacks)

    A drifted model makes confident but wrong predictions.
    """

    def __init__(self, window_size: int = 1000,
                 baseline_size: int = 5000,
                 warning_threshold: float = 0.05,
                 critical_threshold: float = 0.1):
        self.window_size = window_size
        self.baseline_size = baseline_size
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        # Sliding windows
        self.feature_window = deque(maxlen=window_size)
        self.prediction_window = deque(maxlen=window_size)
        self.confidence_window = deque(maxlen=window_size)

        # Baseline statistics (computed from initial data)
        self.baseline_feature_mean = None
        self.baseline_feature_std = None
        self.baseline_prediction_dist = None
        self.baseline_confidence_mean = None
        self.baseline_confidence_std = None

        self.alerts = []
        self.calibrated = False
        self.samples_seen = 0

    def calibrate(self, features: np.ndarray,
                  predictions: np.ndarray,
                  confidences: np.ndarray):
        """Set baseline statistics from initial deployment data."""
        self.baseline_feature_mean = features.mean(axis=0)
        self.baseline_feature_std = features.std(axis=0) + 1e-8
        self.baseline_prediction_dist = np.bincount(
            predictions, minlength=predictions.max() + 1
        ) / len(predictions)
        self.baseline_confidence_mean = confidences.mean()
        self.baseline_confidence_std = confidences.std() + 1e-8
        self.calibrated = True
        print(f"Drift detector calibrated with {len(features)} samples")

    def observe(self, features: np.ndarray,
                prediction: int,
                confidence: float) -> Optional[DriftAlert]:
        """Record one observation and check for drift."""
        self.feature_window.append(features)
        self.prediction_window.append(prediction)
        self.confidence_window.append(confidence)
        self.samples_seen += 1

        if not self.calibrated:
            return None

        if len(self.feature_window) < self.window_size:
            return None

        # Check for drift every 100 samples
        if self.samples_seen % 100 != 0:
            return None

        alerts = []

        # Feature drift (PSI or mean shift)
        feature_alert = self._check_feature_drift()
        if feature_alert:
            alerts.append(feature_alert)

        # Prediction distribution drift
        pred_alert = self._check_prediction_drift()
        if pred_alert:
            alerts.append(pred_alert)

        # Confidence drift
        conf_alert = self._check_confidence_drift()
        if conf_alert:
            alerts.append(conf_alert)

        self.alerts.extend(alerts)
        return alerts[0] if alerts else None

    def _check_feature_drift(self) -> Optional[DriftAlert]:
        """Check for feature distribution drift using normalized mean shift."""
        current_features = np.array(list(self.feature_window))
        current_mean = current_features.mean(axis=0)

        # Normalized mean shift
        shift = np.abs(
            (current_mean - self.baseline_feature_mean) /
            self.baseline_feature_std
        ).mean()

        if shift > self.critical_threshold * 10:
            return DriftAlert(
                "feature", "mean_shift",
                0.0, float(shift), "critical", time.time()
            )
        elif shift > self.warning_threshold * 10:
            return DriftAlert(
                "feature", "mean_shift",
                0.0, float(shift), "warning", time.time()
            )
        return None

    def _check_prediction_drift(self) -> Optional[DriftAlert]:
        """Check if prediction distribution has shifted (PSI-like)."""
        predictions = np.array(list(self.prediction_window))
        current_dist = np.bincount(
            predictions,
            minlength=len(self.baseline_prediction_dist)
        ) / len(predictions)

        # Jensen-Shannon divergence
        m = 0.5 * (current_dist + self.baseline_prediction_dist + 1e-10)
        kl_p = np.sum(
            (self.baseline_prediction_dist + 1e-10) *
            np.log((self.baseline_prediction_dist + 1e-10) / m)
        )
        kl_q = np.sum(
            (current_dist + 1e-10) *
            np.log((current_dist + 1e-10) / m)
        )
        jsd = 0.5 * (kl_p + kl_q)

        if jsd > self.critical_threshold:
            return DriftAlert(
                "prediction", "distribution_shift",
                0.0, float(jsd), "critical", time.time()
            )
        elif jsd > self.warning_threshold:
            return DriftAlert(
                "prediction", "distribution_shift",
                0.0, float(jsd), "warning", time.time()
            )
        return None

    def _check_confidence_drift(self) -> Optional[DriftAlert]:
        """Check if model confidence has dropped (degradation signal)."""
        current_conf = np.mean(list(self.confidence_window))

        shift = abs(current_conf - self.baseline_confidence_mean)
        normalized = shift / self.baseline_confidence_std

        if normalized > 3.0:
            return DriftAlert(
                "confidence", "confidence_shift",
                float(self.baseline_confidence_mean),
                float(current_conf),
                "critical", time.time()
            )
        elif normalized > 2.0:
            return DriftAlert(
                "confidence", "confidence_shift",
                float(self.baseline_confidence_mean),
                float(current_conf),
                "warning", time.time()
            )
        return None
```

### 4.3 성능 추적기

```python
#!/usr/bin/env python3
"""Edge AI performance tracking and reporting."""

import time
import json
import numpy as np
from collections import defaultdict, deque
from pathlib import Path


class PerformanceTracker:
    """Track and report edge AI inference performance metrics."""

    def __init__(self, device_id: str, report_interval_s: float = 300,
                 log_dir: str = "/var/log/edge_ai"):
        self.device_id = device_id
        self.report_interval = report_interval_s
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Metric accumulators
        self.latencies = deque(maxlen=10000)
        self.throughput_counter = 0
        self.error_counter = 0
        self.model_metrics = defaultdict(list)

        self.start_time = time.time()
        self.last_report_time = time.time()

    def record_inference(self, latency_ms: float,
                         model_name: str = "default",
                         success: bool = True,
                         extra_metrics: dict = None):
        """Record a single inference event."""
        self.latencies.append(latency_ms)
        self.throughput_counter += 1

        if not success:
            self.error_counter += 1

        if extra_metrics:
            for key, value in extra_metrics.items():
                self.model_metrics[key].append(value)

        # Auto-report at intervals
        if time.time() - self.last_report_time > self.report_interval:
            self.generate_report()

    def generate_report(self) -> dict:
        """Generate a performance report."""
        now = time.time()
        elapsed = now - self.last_report_time

        latency_arr = np.array(list(self.latencies)) if self.latencies else np.array([0])

        report = {
            "device_id": self.device_id,
            "timestamp": now,
            "period_s": round(elapsed, 1),
            "inference": {
                "total_count": self.throughput_counter,
                "error_count": self.error_counter,
                "error_rate": round(
                    self.error_counter / max(1, self.throughput_counter), 4
                ),
                "throughput_fps": round(
                    self.throughput_counter / max(1, elapsed), 2
                ),
            },
            "latency_ms": {
                "mean": round(float(np.mean(latency_arr)), 2),
                "p50": round(float(np.percentile(latency_arr, 50)), 2),
                "p95": round(float(np.percentile(latency_arr, 95)), 2),
                "p99": round(float(np.percentile(latency_arr, 99)), 2),
                "max": round(float(np.max(latency_arr)), 2),
            },
            "custom_metrics": {
                key: {
                    "mean": round(float(np.mean(vals[-1000:])), 4),
                    "count": len(vals),
                }
                for key, vals in self.model_metrics.items()
            },
        }

        # Save to disk
        log_file = self.log_dir / f"report_{int(now)}.json"
        with open(log_file, "w") as f:
            json.dump(report, f, indent=2)

        # Reset counters
        self.throughput_counter = 0
        self.error_counter = 0
        self.last_report_time = now

        print(f"Report saved: {log_file}")
        return report
```

---

## 5. 플릿 관리

### 5.1 플릿 오케스트레이터

```python
#!/usr/bin/env python3
"""Fleet management for edge AI devices."""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from enum import Enum


class DeviceStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    UPDATING = "updating"
    ERROR = "error"
    DEGRADED = "degraded"


@dataclass
class DeviceInfo:
    device_id: str
    status: str = "online"
    model_version: str = ""
    last_heartbeat: float = 0
    metrics: dict = field(default_factory=dict)
    hardware: str = ""
    location: str = ""
    tags: List[str] = field(default_factory=list)


class FleetManager:
    """Manage a fleet of edge AI devices.

    Fleet management responsibilities:
    - Track device health and connectivity
    - Orchestrate model rollouts (canary, staged, full)
    - Aggregate metrics across devices
    - Detect and respond to fleet-wide issues
    """

    def __init__(self):
        self.devices: Dict[str, DeviceInfo] = {}
        self.rollout_history = []

    def register_device(self, device_info: DeviceInfo):
        """Register a new device in the fleet."""
        self.devices[device_info.device_id] = device_info
        print(f"Registered device: {device_info.device_id} "
              f"({device_info.hardware})")

    def heartbeat(self, device_id: str, metrics: dict):
        """Process a device heartbeat with metrics."""
        if device_id not in self.devices:
            return

        device = self.devices[device_id]
        device.last_heartbeat = time.time()
        device.metrics = metrics
        device.status = DeviceStatus.ONLINE.value

    def get_fleet_health(self) -> dict:
        """Get overall fleet health summary."""
        now = time.time()
        timeout = 300  # 5 minutes

        status_counts = {"online": 0, "offline": 0, "error": 0,
                         "updating": 0, "degraded": 0}
        version_counts = {}
        latencies = []

        for device in self.devices.values():
            # Check if device is offline
            if now - device.last_heartbeat > timeout:
                device.status = DeviceStatus.OFFLINE.value

            status_counts[device.status] = status_counts.get(
                device.status, 0
            ) + 1

            version_counts[device.model_version] = version_counts.get(
                device.model_version, 0
            ) + 1

            if "latency_ms" in device.metrics:
                latencies.append(device.metrics["latency_ms"])

        return {
            "total_devices": len(self.devices),
            "status": status_counts,
            "model_versions": version_counts,
            "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 2),
        }

    def plan_rollout(self, model_version: str,
                     strategy: str = "canary",
                     target_tags: List[str] = None) -> dict:
        """Plan a model rollout across the fleet.

        Strategies:
        - canary: 5% -> monitor -> 25% -> monitor -> 100%
        - staged: 10% -> 25% -> 50% -> 100% (time-based)
        - immediate: 100% at once (use for hotfixes only)
        """
        eligible = self._get_eligible_devices(target_tags)

        if strategy == "canary":
            stages = [
                {"percentage": 5, "duration_hours": 2, "auto_proceed": False},
                {"percentage": 25, "duration_hours": 4, "auto_proceed": True},
                {"percentage": 100, "duration_hours": 0, "auto_proceed": True},
            ]
        elif strategy == "staged":
            stages = [
                {"percentage": 10, "duration_hours": 1, "auto_proceed": True},
                {"percentage": 25, "duration_hours": 2, "auto_proceed": True},
                {"percentage": 50, "duration_hours": 4, "auto_proceed": True},
                {"percentage": 100, "duration_hours": 0, "auto_proceed": True},
            ]
        elif strategy == "immediate":
            stages = [
                {"percentage": 100, "duration_hours": 0, "auto_proceed": True},
            ]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        rollout_plan = {
            "model_version": model_version,
            "strategy": strategy,
            "eligible_devices": len(eligible),
            "stages": stages,
            "rollback_criteria": {
                "error_rate_threshold": 0.05,
                "latency_increase_pct": 20,
                "confidence_drop_pct": 10,
            },
        }

        self.rollout_history.append(rollout_plan)
        return rollout_plan

    def _get_eligible_devices(self, tags: List[str] = None) -> list:
        """Get devices eligible for update."""
        eligible = []
        for device in self.devices.values():
            if device.status == DeviceStatus.OFFLINE.value:
                continue
            if tags and not any(t in device.tags for t in tags):
                continue
            eligible.append(device)
        return eligible


# Demonstration
if __name__ == "__main__":
    fleet = FleetManager()

    # Register devices
    for i in range(20):
        fleet.register_device(DeviceInfo(
            device_id=f"cam-{i:03d}",
            model_version="detector-v2.0.0-int8",
            last_heartbeat=time.time(),
            hardware="jetson-orin-nano",
            location=f"building-{i // 5}",
            tags=["camera", f"floor-{i % 3}"],
        ))

    # Check health
    health = fleet.get_fleet_health()
    print(f"Fleet health: {json.dumps(health, indent=2)}")

    # Plan rollout
    plan = fleet.plan_rollout("detector-v2.1.0-int8", strategy="canary")
    print(f"\nRollout plan: {json.dumps(plan, indent=2)}")
```

---

## 6. 엣지 MLOps 파이프라인

### 6.1 엔드투엔드 파이프라인

```
+-----------------------------------------------------------------+
|              Edge MLOps Pipeline                                  |
+-----------------------------------------------------------------+
|                                                                   |
|   1. Train          2. Optimize       3. Validate                |
|   +--------+       +----------+      +-----------+              |
|   | Cloud  |------>| Quantize |----->| Edge      |              |
|   | Train  |       | Prune    |      | Benchmark |              |
|   +--------+       | Convert  |      | Accuracy  |              |
|                     +----------+      +-----------+              |
|                                            |                     |
|   6. Monitor        5. Rollout        4. Register                |
|   +--------+       +----------+      +-----------+              |
|   | Drift  |<------| Canary   |<-----| Version   |              |
|   | Perf   |       | Staged   |      | Metadata  |              |
|   | Alerts |       | Full     |      | Artifact  |              |
|   +--------+       +----------+      +-----------+              |
|       |                                                          |
|       v                                                          |
|   7. Retrain (if drift detected) -> back to Step 1              |
|                                                                   |
+-----------------------------------------------------------------+
```

### 6.2 파이프라인 오케스트레이터

```python
#!/usr/bin/env python3
"""Edge MLOps pipeline orchestrator."""

import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class PipelineConfig:
    """Configuration for the edge MLOps pipeline."""
    model_name: str
    train_fn: Optional[Callable] = None
    optimize_fn: Optional[Callable] = None
    validate_fn: Optional[Callable] = None
    accuracy_threshold: float = 0.90
    latency_threshold_ms: float = 50.0
    rollout_strategy: str = "canary"


class EdgeMLOpsPipeline:
    """Orchestrate the full edge MLOps lifecycle."""

    def __init__(self, config: PipelineConfig,
                 model_registry=None,
                 fleet_manager=None,
                 drift_detector=None):
        self.config = config
        self.registry = model_registry
        self.fleet = fleet_manager
        self.drift = drift_detector

    def run_pipeline(self, training_data=None,
                     version: str = "1.0.0") -> dict:
        """Execute the full pipeline."""
        results = {"version": version, "stages": {}}

        # Stage 1: Train
        print("\n[1/6] Training model...")
        if self.config.train_fn and training_data:
            model = self.config.train_fn(training_data)
            results["stages"]["train"] = "completed"
        else:
            results["stages"]["train"] = "skipped (no train_fn)"
            model = None

        # Stage 2: Optimize
        print("[2/6] Optimizing model...")
        if self.config.optimize_fn and model:
            optimized_path = self.config.optimize_fn(model)
            results["stages"]["optimize"] = "completed"
        else:
            results["stages"]["optimize"] = "skipped"
            optimized_path = None

        # Stage 3: Validate
        print("[3/6] Validating model...")
        if self.config.validate_fn and optimized_path:
            metrics = self.config.validate_fn(optimized_path)
            passed = (
                metrics.get("accuracy", 0) >= self.config.accuracy_threshold
                and metrics.get("latency_ms", float("inf")) <= self.config.latency_threshold_ms
            )
            results["stages"]["validate"] = {
                "metrics": metrics,
                "passed": passed,
            }
            if not passed:
                print("Validation FAILED -- aborting pipeline")
                results["status"] = "failed_validation"
                return results
        else:
            results["stages"]["validate"] = "skipped"

        # Stage 4: Register
        print("[4/6] Registering model version...")
        if self.registry:
            self.registry.register(
                optimized_path or "model.tflite",
                ModelVersion(
                    model_name=self.config.model_name,
                    version=version,
                    framework="tflite",
                    quantization="int8",
                    input_spec={},
                    output_spec={},
                    metrics=metrics if self.config.validate_fn else {},
                    file_hash="",
                    file_size_bytes=0,
                )
            )
            results["stages"]["register"] = "completed"

        # Stage 5: Rollout
        print("[5/6] Planning rollout...")
        if self.fleet:
            plan = self.fleet.plan_rollout(
                f"{self.config.model_name}-v{version}-int8",
                strategy=self.config.rollout_strategy
            )
            results["stages"]["rollout"] = plan

        # Stage 6: Monitor
        print("[6/6] Monitoring enabled")
        results["stages"]["monitor"] = "active"
        results["status"] = "deployed"

        print(f"\nPipeline complete: {self.config.model_name} v{version}")
        return results


if __name__ == "__main__":
    config = PipelineConfig(
        model_name="person_detector",
        accuracy_threshold=0.90,
        latency_threshold_ms=30,
        rollout_strategy="canary",
    )

    pipeline = EdgeMLOpsPipeline(config)
    result = pipeline.run_pipeline(version="2.1.0")
    print(f"\nPipeline result: {result['status']}")
```

---

## 연습 문제

### 연습 1: OTA 업데이트 시스템
1. 모델 파일을 제공하는 로컬 HTTP 서버와 함께 `OTAUpdateManager`를 구현하십시오
2. 전체 주기를 테스트하십시오: 다운로드 -> 체크섬 검증 -> 검증 -> 활성화
3. 검증 실패를 시뮬레이션하고 롤백이 올바르게 작동하는지 확인하십시오

### 연습 2: 드리프트 탐지
1. 1000개의 정상 샘플로 `DriftDetector`를 보정하십시오
2. 입력 분포를 점진적으로 변화시키십시오 (예: 이미지 밝기 증가)
3. 드리프트 탐지기가 경고와 위험 알림을 언제 발생시키는지 관찰하십시오

### 연습 3: 플릿 롤아웃
1. `FleetManager`로 50대의 시뮬레이션 디바이스로 구성된 플릿을 생성하십시오
2. 카나리 롤아웃을 계획하십시오: 5% -> 25% -> 100%
3. 카나리 그룹의 한 디바이스가 높은 오류율을 보고하는 것을 시뮬레이션하고 해당 단계에 대한 자동 롤백을 구현하십시오

---

**이전**: [NLP를 위한 Edge AI](./15_Edge_AI_for_NLP.md)
