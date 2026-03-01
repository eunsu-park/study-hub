"""
Production ML — Model Serving Patterns
=======================================

Demonstrates key concepts for preparing ML models for production:
1. Model optimization (compression, quantization comparison)
2. ONNX export and inference speed comparison
3. Training-serving skew prevention with sklearn Pipeline
4. Batch vs single-sample inference benchmarking
5. Data drift detection (KS test, PSI)
6. Latency-accuracy Pareto frontier visualization

Requirements:
    pip install scikit-learn numpy scipy matplotlib joblib
    pip install skl2onnx onnxruntime  # Optional, for ONNX demo
"""

import numpy as np
import time
import os
import json
import joblib
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from scipy import stats


# ============================================================
# 1. Dataset Preparation
# ============================================================

print("=" * 60)
print("1. Dataset Preparation")
print("=" * 60)

X, y = make_classification(
    n_samples=10000, n_features=20, n_informative=15,
    n_redundant=3, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# ============================================================
# 2. Model Optimization Comparison
# ============================================================

print("\n" + "=" * 60)
print("2. Model Optimization — Size and Speed Comparison")
print("=" * 60)

# Train a large model (baseline)
large_model = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, random_state=42
)
large_model.fit(X_train, y_train)

# Optimized variants
small_model = GradientBoostingClassifier(
    n_estimators=50, max_depth=3, random_state=42
)
small_model.fit(X_train, y_train)

# Knowledge distillation: student learns from teacher's soft labels
soft_labels = large_model.predict_proba(X_train)[:, 1]
student = LogisticRegression(max_iter=1000, random_state=42)
student.fit(X_train, (soft_labels > 0.5).astype(int))

models = {
    "GBM-200 (baseline)": large_model,
    "GBM-50 (fewer trees)": small_model,
    "LR (distilled)": student,
}

print(f"\n{'Model':<25s} {'Accuracy':>8s} {'Size(KB)':>10s} {'p50(ms)':>8s} {'p99(ms)':>8s}")
print("-" * 65)

single_sample = X_test[:1]

for name, model in models.items():
    acc = accuracy_score(y_test, model.predict(X_test))

    # Model size
    path = f"/tmp/model_{name.split()[0]}.joblib"
    joblib.dump(model, path, compress=3)
    size_kb = os.path.getsize(path) / 1024

    # Latency benchmark (single sample)
    model.predict(single_sample)  # warm-up
    latencies = []
    for _ in range(200):
        start = time.perf_counter()
        model.predict(single_sample)
        latencies.append((time.perf_counter() - start) * 1000)

    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)

    print(f"{name:<25s} {acc:>8.4f} {size_kb:>10.1f} {p50:>8.3f} {p99:>8.3f}")


# ============================================================
# 3. Training-Serving Skew Prevention
# ============================================================

print("\n" + "=" * 60)
print("3. Training-Serving Skew Prevention with Pipeline")
print("=" * 60)

# Introduce missing values and noise to simulate real data
X_noisy = X.copy()
mask = np.random.RandomState(42).random(X_noisy.shape) < 0.05
X_noisy[mask] = np.nan

X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
    X_noisy, y, test_size=0.2, random_state=42
)

# CORRECT: Pipeline bundles preprocessing + model
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X_train_n, y_train_n)
pipeline_acc = accuracy_score(y_test_n, pipeline.predict(X_test_n))

# WRONG: Separate preprocessing (simulating skew)
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train_n))
skew_model = RandomForestClassifier(n_estimators=100, random_state=42)
skew_model.fit(X_train_proc, y_train_n)

# Simulate serving-time skew: use mean imputation instead of median
imputer_wrong = SimpleImputer(strategy='mean')  # Different strategy!
scaler_wrong = StandardScaler()
X_test_skewed = scaler_wrong.fit_transform(  # fit_transform on test data!
    imputer_wrong.fit_transform(X_test_n)
)
skew_acc = accuracy_score(y_test_n, skew_model.predict(X_test_skewed))

print(f"Pipeline (no skew):    {pipeline_acc:.4f}")
print(f"Separate (with skew):  {skew_acc:.4f}")
print(f"Accuracy drop from skew: {(pipeline_acc - skew_acc) * 100:.2f}%")

# Save and reload pipeline
joblib.dump(pipeline, '/tmp/production_pipeline.joblib', compress=3)
loaded = joblib.load('/tmp/production_pipeline.joblib')
reload_acc = accuracy_score(y_test_n, loaded.predict(X_test_n))
print(f"Reloaded pipeline:     {reload_acc:.4f} (matches: {reload_acc == pipeline_acc})")


# ============================================================
# 4. Batch vs Single-Sample Inference
# ============================================================

print("\n" + "=" * 60)
print("4. Batch vs Single-Sample Inference Throughput")
print("=" * 60)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

large_data = np.random.randn(10000, 20)

# Single-sample loop
start = time.time()
for i in range(len(large_data)):
    model.predict(large_data[i:i+1])
single_time = time.time() - start

# Batch prediction
batch_sizes = [10, 100, 1000, 10000]
print(f"\n{'Method':<25s} {'Time(s)':>8s} {'Throughput':>15s}")
print("-" * 50)
print(f"{'Single-sample loop':<25s} {single_time:>8.3f} "
      f"{len(large_data)/single_time:>12.0f} samples/s")

for bs in batch_sizes:
    start = time.time()
    for i in range(0, len(large_data), bs):
        model.predict(large_data[i:i+bs])
    batch_time = time.time() - start
    print(f"{'Batch (size=' + str(bs) + ')':<25s} {batch_time:>8.3f} "
          f"{len(large_data)/batch_time:>12.0f} samples/s")


# ============================================================
# 5. Data Drift Detection
# ============================================================

print("\n" + "=" * 60)
print("5. Data Drift Detection (KS Test)")
print("=" * 60)

# Reference distribution (training data)
reference = np.random.randn(2000, 5)

# Production data with drift in some features
production = np.random.randn(2000, 5)
production[:, 0] += 0.3   # Slight mean shift
production[:, 2] += 1.0   # Large mean shift
production[:, 4] *= 2.0   # Variance change

feature_names = [f"feature_{i}" for i in range(5)]

print(f"\n{'Feature':<12s} {'KS Stat':>8s} {'p-value':>10s} {'Drift?':>8s}")
print("-" * 42)

for i, name in enumerate(feature_names):
    stat, p_val = stats.ks_2samp(reference[:, i], production[:, i])
    drifted = "YES" if p_val < 0.01 else "no"
    print(f"{name:<12s} {stat:>8.4f} {p_val:>10.6f} {drifted:>8s}")


# PSI (Population Stability Index)
def compute_psi(reference, production, n_bins=10):
    """Compute PSI between two distributions."""
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
    prod_counts = np.histogram(production, bins=breakpoints)[0] / len(production)

    # Avoid division by zero
    ref_counts = np.clip(ref_counts, 1e-6, None)
    prod_counts = np.clip(prod_counts, 1e-6, None)

    psi = np.sum((prod_counts - ref_counts) * np.log(prod_counts / ref_counts))
    return psi


print(f"\n{'Feature':<12s} {'PSI':>8s} {'Severity':>12s}")
print("-" * 35)
for i, name in enumerate(feature_names):
    psi = compute_psi(reference[:, i], production[:, i])
    if psi < 0.1:
        severity = "negligible"
    elif psi < 0.25:
        severity = "moderate"
    else:
        severity = "MAJOR"
    print(f"{name:<12s} {psi:>8.4f} {severity:>12s}")


# ============================================================
# 6. Latency-Accuracy Pareto Frontier
# ============================================================

print("\n" + "=" * 60)
print("6. Latency-Accuracy Trade-off")
print("=" * 60)

candidates = {
    "LogisticReg":  LogisticRegression(max_iter=1000, random_state=42),
    "DT(depth=3)":  DecisionTreeClassifier(max_depth=3, random_state=42),
    "DT(depth=10)": DecisionTreeClassifier(max_depth=10, random_state=42),
    "RF(10)":       RandomForestClassifier(n_estimators=10, random_state=42),
    "RF(50)":       RandomForestClassifier(n_estimators=50, random_state=42),
    "RF(200)":      RandomForestClassifier(n_estimators=200, random_state=42),
    "GBM(50)":      GradientBoostingClassifier(n_estimators=50, random_state=42),
    "GBM(200)":     GradientBoostingClassifier(n_estimators=200, random_state=42),
    "KNN(5)":       KNeighborsClassifier(n_neighbors=5),
}

results = []
for name, m in candidates.items():
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))

    m.predict(single_sample)  # warm-up
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        m.predict(single_sample)
        latencies.append((time.perf_counter() - start) * 1000)
    p50 = np.percentile(latencies, 50)
    results.append((name, acc, p50))

# Identify Pareto-optimal models
results.sort(key=lambda x: x[2])  # Sort by latency
pareto = []
best_acc = 0
for name, acc, lat in results:
    if acc > best_acc:
        pareto.append(name)
        best_acc = acc

print(f"\n{'Model':<15s} {'Accuracy':>8s} {'p50(ms)':>8s} {'Pareto?':>8s}")
print("-" * 42)
for name, acc, lat in results:
    is_pareto = "★" if name in pareto else ""
    print(f"{name:<15s} {acc:>8.4f} {lat:>8.3f} {is_pareto:>8s}")


# ============================================================
# 7. Model Artifact Metadata
# ============================================================

print("\n" + "=" * 60)
print("7. Model Artifact with Metadata")
print("=" * 60)

metadata = {
    "model_version": "1.0.0",
    "model_class": "RandomForestClassifier",
    "training_date": datetime.utcnow().isoformat() + "Z",
    "n_training_samples": len(X_train),
    "n_features": X_train.shape[1],
    "feature_names": [f"feature_{i}" for i in range(X_train.shape[1])],
    "target_classes": [0, 1],
    "performance_metrics": {
        "accuracy": round(accuracy_score(y_test, model.predict(X_test)), 4),
    },
    "serving_info": {
        "serialization_format": "joblib",
        "compression_level": 3,
        "expected_latency_p50_ms": round(float(np.median(latencies)), 3),
    }
}

print(json.dumps(metadata, indent=2))

print("\n✓ All production ML serving demos completed.")
