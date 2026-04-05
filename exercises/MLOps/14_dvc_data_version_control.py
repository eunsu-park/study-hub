"""
Exercise Solutions: DVC (Data Version Control)
===========================================
Lesson 14 from MLOps topic.

Exercises
---------
1. Initialize DVC Project — Set up a DVC project with remote storage
   configuration and .dvc/.dvcignore files.
2. Build Three-Stage DVC Pipeline — Create a DVC pipeline with
   preprocess, train, and evaluate stages with dependency tracking.
3. Hyperparameter Search with DVC Experiments — Run multiple
   experiments with different hyperparameters using DVC exp.
4. CML Pull Request Report — Generate a CML-style report comparing
   experiment results for pull request review.
5. MLflow vs DVC Comparison — Compare both tools across key dimensions
   with practical recommendations.
"""

import math
import random
import json
import hashlib
from datetime import datetime


# ============================================================
# Simulated DVC System (pure Python)
# ============================================================

class SimulatedDVC:
    """Simulates core DVC operations without requiring actual DVC."""

    def __init__(self, repo_root="/project"):
        self.repo_root = repo_root
        self.tracked_files = {}  # path -> {hash, size, remote}
        self.pipelines = {}      # stage_name -> {cmd, deps, outs, params}
        self.experiments = []
        self.remotes = {}
        self.params = {}
        self.metrics = {}

    def init(self):
        """Initialize DVC in the current repository."""
        return {
            "initialized": True,
            "files_created": [".dvc/", ".dvc/config", ".dvcignore"],
        }

    def remote_add(self, name, url):
        self.remotes[name] = url

    def add(self, filepath, size_bytes=0):
        """Track a file with DVC."""
        content_hash = hashlib.md5(filepath.encode()).hexdigest()
        self.tracked_files[filepath] = {
            "md5": content_hash,
            "size": size_bytes,
            "dvc_file": f"{filepath}.dvc",
        }
        return self.tracked_files[filepath]

    def add_stage(self, name, cmd, deps=None, outs=None, params=None, metrics=None):
        """Add a pipeline stage to dvc.yaml."""
        self.pipelines[name] = {
            "cmd": cmd,
            "deps": deps or [],
            "outs": outs or [],
            "params": params or [],
            "metrics": metrics or [],
        }

    def repro(self, stage=None):
        """Reproduce the pipeline (simulate execution)."""
        if stage:
            stages = [stage]
        else:
            stages = list(self.pipelines.keys())

        results = []
        for s in stages:
            pipe = self.pipelines.get(s)
            if not pipe:
                continue
            # Check if any dependency changed
            results.append({
                "stage": s,
                "cmd": pipe["cmd"],
                "deps_checked": len(pipe["deps"]),
                "outs_produced": len(pipe["outs"]),
                "status": "reproduced",
            })
        return results

    def exp_run(self, params_override=None):
        """Run an experiment with optional parameter overrides."""
        experiment = {
            "id": f"exp-{len(self.experiments):04d}",
            "params": {**self.params, **(params_override or {})},
            "metrics": {},
            "timestamp": datetime.now().isoformat(),
        }
        self.experiments.append(experiment)
        return experiment

    def exp_show(self):
        """Show all experiments."""
        return self.experiments


# ============================================================
# Exercise 1: Initialize DVC Project
# ============================================================

def exercise_1_init_project():
    """Set up a DVC project with remote storage and proper configuration."""

    dvc = SimulatedDVC("/ml-project")

    print("Initialize DVC Project")
    print("=" * 60)

    # Step 1: Initialize
    init_result = dvc.init()
    print(f"\n  1. dvc init")
    print(f"     Created: {init_result['files_created']}")

    # Step 2: Configure remote storage
    dvc.remote_add("myremote", "s3://ml-data-bucket/dvc-store")
    print(f"\n  2. dvc remote add myremote s3://ml-data-bucket/dvc-store")

    # Step 3: Track data files
    files_to_track = [
        ("data/raw/train.csv", 150_000_000),
        ("data/raw/test.csv", 30_000_000),
        ("models/model.pkl", 95_000_000),
    ]

    print(f"\n  3. Track data files:")
    for filepath, size in files_to_track:
        result = dvc.add(filepath, size)
        print(f"     dvc add {filepath}")
        print(f"       -> {result['dvc_file']} (md5: {result['md5'][:8]}..., "
              f"size: {size/1e6:.0f}MB)")

    # Step 4: Show .dvcignore
    dvcignore = """# DVC ignore patterns
# IDE files
.idea/
.vscode/
*.swp

# Python cache
__pycache__/
*.pyc
.pytest_cache/

# Large files already tracked by DVC
*.csv.bak
*.pkl.bak

# OS files
.DS_Store
Thumbs.db
"""

    print(f"\n  4. .dvcignore:")
    print(f"     {dvcignore}")

    # Step 5: Show project structure
    print(f"  5. Project Structure:")
    structure = """
     ml-project/
     ├── .dvc/
     │   ├── config          # Remote storage config
     │   └── .gitignore
     ├── .dvcignore           # Patterns to ignore
     ├── data/
     │   └── raw/
     │       ├── train.csv    # (DVC-tracked, not in git)
     │       ├── train.csv.dvc # (In git, points to DVC cache)
     │       └── test.csv.dvc
     ├── models/
     │   ├── model.pkl        # (DVC-tracked)
     │   └── model.pkl.dvc
     ├── src/
     │   ├── preprocess.py
     │   ├── train.py
     │   └── evaluate.py
     ├── params.yaml           # Hyperparameters
     ├── dvc.yaml              # Pipeline definition
     ├── dvc.lock              # Reproducibility lock file
     └── .gitignore            # Excludes DVC-tracked files"""
    print(structure)

    # Key commands
    print(f"\n  6. Key DVC Commands:")
    commands = [
        ("dvc push", "Upload tracked data to remote storage"),
        ("dvc pull", "Download tracked data from remote storage"),
        ("dvc status", "Check which files changed since last run"),
        ("dvc diff", "Show changes between commits/branches"),
        ("dvc repro", "Reproduce the full pipeline"),
    ]
    for cmd, desc in commands:
        print(f"     {cmd:<20s} — {desc}")

    return dvc


# ============================================================
# Exercise 2: Build Three-Stage DVC Pipeline
# ============================================================

def exercise_2_three_stage_pipeline():
    """Create a DVC pipeline with preprocess, train, evaluate stages."""

    dvc = SimulatedDVC("/ml-project")

    # --- Define pipeline stages ---
    dvc.add_stage(
        name="preprocess",
        cmd="python src/preprocess.py",
        deps=["src/preprocess.py", "data/raw/train.csv"],
        outs=["data/processed/train_features.parquet", "data/processed/test_features.parquet"],
        params=["preprocess.test_ratio", "preprocess.random_seed"],
    )

    dvc.add_stage(
        name="train",
        cmd="python src/train.py",
        deps=["src/train.py", "data/processed/train_features.parquet"],
        outs=["models/model.pkl"],
        params=["train.learning_rate", "train.n_estimators", "train.max_depth"],
    )

    dvc.add_stage(
        name="evaluate",
        cmd="python src/evaluate.py",
        deps=["src/evaluate.py", "models/model.pkl", "data/processed/test_features.parquet"],
        metrics=["metrics/eval.json"],
        params=["evaluate.threshold"],
    )

    # --- Show dvc.yaml ---
    dvc_yaml = """stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - src/preprocess.py
      - data/raw/train.csv
    params:
      - preprocess.test_ratio
      - preprocess.random_seed
    outs:
      - data/processed/train_features.parquet
      - data/processed/test_features.parquet

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/train_features.parquet
    params:
      - train.learning_rate
      - train.n_estimators
      - train.max_depth
    outs:
      - models/model.pkl

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/model.pkl
      - data/processed/test_features.parquet
    params:
      - evaluate.threshold
    metrics:
      - metrics/eval.json:
          cache: false
"""

    params_yaml = """preprocess:
  test_ratio: 0.2
  random_seed: 42

train:
  learning_rate: 0.01
  n_estimators: 100
  max_depth: 10

evaluate:
  threshold: 0.5
"""

    print("Three-Stage DVC Pipeline")
    print("=" * 60)

    print(f"\n  dvc.yaml:")
    print(dvc_yaml)

    print(f"  params.yaml:")
    print(params_yaml)

    # --- Simulate pipeline execution ---
    print("  Pipeline Execution (dvc repro):")
    print(f"  {'-'*50}")

    # Simulate execution with actual computation
    random.seed(42)
    n_samples = 500
    data = []
    for _ in range(n_samples):
        x = [random.gauss(0, 1) for _ in range(5)]
        y = 1 if sum(w * xi for w, xi in zip([0.5, -0.3, 0.8, 0.1, -0.6], x)) > 0 else 0
        data.append((x, y))

    train_data = data[:400]
    test_data = data[400:]

    print(f"\n  Stage 1: preprocess")
    print(f"    Input: {n_samples} samples")
    print(f"    Output: {len(train_data)} train, {len(test_data)} test")

    # Train
    w = [0.0] * 5
    b = 0.0
    for _ in range(100):
        for x, y in train_data:
            z = max(-500, min(500, sum(wi * xi for wi, xi in zip(w, x)) + b))
            p = 1 / (1 + math.exp(-z))
            e = p - y
            for j in range(5):
                w[j] -= 0.01 * (e * x[j] + 0.001 * w[j])
            b -= 0.01 * e

    print(f"\n  Stage 2: train")
    print(f"    Algorithm: Logistic Regression")
    print(f"    Training samples: {len(train_data)}")

    # Evaluate
    tp = fp = tn = fn = 0
    for x, y in test_data:
        z = max(-500, min(500, sum(wi * xi for wi, xi in zip(w, x)) + b))
        pred = 1 if (1 / (1 + math.exp(-z))) >= 0.5 else 0
        if pred == 1 and y == 1: tp += 1
        elif pred == 1 and y == 0: fp += 1
        elif pred == 0 and y == 0: tn += 1
        else: fn += 1

    metrics = {
        "accuracy": round((tp + tn) / (tp + fp + tn + fn), 4),
        "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0,
    }
    metrics["f1"] = round(2 * metrics["precision"] * metrics["recall"] /
                          (metrics["precision"] + metrics["recall"]), 4) if (
        metrics["precision"] + metrics["recall"]) > 0 else 0

    print(f"\n  Stage 3: evaluate")
    print(f"    metrics/eval.json: {json.dumps(metrics)}")

    # DAG visualization
    print(f"\n  Pipeline DAG:")
    print(f"    data/raw/train.csv ─┐")
    print(f"    src/preprocess.py ──┤")
    print(f"    params.yaml ────────┼─> [preprocess] ─> data/processed/")
    print(f"                        │                       │")
    print(f"    src/train.py ───────┤                       │")
    print(f"    params.yaml ────────┼──────────────────────>│")
    print(f"                        └─> [train] ─> models/model.pkl")
    print(f"                                           │")
    print(f"    src/evaluate.py ───────────────────────>│")
    print(f"                        └─> [evaluate] ─> metrics/eval.json")

    return dvc


# ============================================================
# Exercise 3: Hyperparameter Search with DVC Experiments
# ============================================================

def exercise_3_dvc_experiments():
    """Run multiple experiments with different hyperparameters."""

    dvc = SimulatedDVC("/ml-project")
    random.seed(42)

    # Generate fixed dataset
    data = []
    for _ in range(500):
        x = [random.gauss(0, 1) for _ in range(5)]
        y = 1 if sum(w * xi for w, xi in zip([0.5, -0.3, 0.8, 0.1, -0.6], x)) > 0 else 0
        data.append((x, y))
    train_data, test_data = data[:400], data[400:]

    # Define experiment grid
    experiments = [
        {"learning_rate": 0.001, "epochs": 200, "reg": 0.001},
        {"learning_rate": 0.01, "epochs": 100, "reg": 0.001},
        {"learning_rate": 0.01, "epochs": 100, "reg": 0.01},
        {"learning_rate": 0.05, "epochs": 50, "reg": 0.001},
        {"learning_rate": 0.01, "epochs": 200, "reg": 0.001},
        {"learning_rate": 0.01, "epochs": 100, "reg": 0.1},
    ]

    print("DVC Experiments")
    print("=" * 60)
    print(f"\n  Running {len(experiments)} experiments:")
    print(f"  {'Exp':>5s} {'LR':>8s} {'Epochs':>7s} {'Reg':>8s} "
          f"{'Accuracy':>9s} {'F1':>7s}")
    print(f"  {'-'*50}")

    results = []
    for i, params in enumerate(experiments):
        # Train with these params
        w = [0.0] * 5
        b = 0.0
        for _ in range(params["epochs"]):
            for x, y in train_data:
                z = max(-500, min(500, sum(wi * xi for wi, xi in zip(w, x)) + b))
                p = 1 / (1 + math.exp(-z))
                e = p - y
                for j in range(5):
                    w[j] -= params["learning_rate"] * (e * x[j] + params["reg"] * w[j])
                b -= params["learning_rate"] * e

        # Evaluate
        tp = fp = tn = fn = 0
        for x, y in test_data:
            z = max(-500, min(500, sum(wi * xi for wi, xi in zip(w, x)) + b))
            pred = 1 if (1 / (1 + math.exp(-z))) >= 0.5 else 0
            if pred == 1 and y == 1: tp += 1
            elif pred == 1 and y == 0: fp += 1
            elif pred == 0 and y == 0: tn += 1
            else: fn += 1

        acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        exp = dvc.exp_run(params)
        exp["metrics"] = {"accuracy": round(acc, 4), "f1": round(f1, 4)}
        results.append(exp)

        print(f"  {exp['id']:>5s} {params['learning_rate']:>8.4f} {params['epochs']:>7d} "
              f"{params['reg']:>8.4f} {acc:>9.4f} {f1:>7.4f}")

    # Best experiment
    best = max(results, key=lambda r: r["metrics"]["f1"])
    print(f"\n  Best: {best['id']} (F1={best['metrics']['f1']:.4f})")
    print(f"  Params: {json.dumps(best['params'])}")
    print(f"\n  To apply: dvc exp apply {best['id']}")

    return results


# ============================================================
# Exercise 4: CML Pull Request Report
# ============================================================

def exercise_4_cml_report():
    """Generate a CML-style pull request report."""

    print("CML Pull Request Report")
    print("=" * 60)

    # Simulated metrics comparison
    main_metrics = {"accuracy": 0.830, "precision": 0.750, "recall": 0.800, "f1": 0.774}
    pr_metrics = {"accuracy": 0.862, "precision": 0.780, "recall": 0.825, "f1": 0.802}

    report = f"""
## Model Evaluation Report

### Metrics Comparison (main vs this PR)

| Metric    | main   | this PR | Delta  | Status |
|-----------|--------|---------|--------|--------|"""

    for metric in ["accuracy", "precision", "recall", "f1"]:
        main_val = main_metrics[metric]
        pr_val = pr_metrics[metric]
        delta = pr_val - main_val
        status = "improved" if delta > 0 else "regressed" if delta < 0 else "same"
        sign = "+" if delta > 0 else ""
        report += f"\n| {metric:<9s} | {main_val:.3f}  | {pr_val:.3f}   | {sign}{delta:.3f}  | {status} |"

    report += f"""

### Parameters Changed

| Parameter      | main  | this PR |
|----------------|-------|---------|
| learning_rate  | 0.010 | 0.010   |
| n_estimators   | 100   | 150     |
| max_depth      | 10    | 12      |
| regularization | 0.001 | 0.005   |

### Training Summary
- Training time: 45s (main: 38s)
- Dataset: 5,000 samples (train: 4,000 / test: 1,000)
- Data version: `d3a5f2c` (unchanged)

### Slice Analysis

| Slice     | F1 (main) | F1 (PR) | Status  |
|-----------|-----------|---------|---------|
| age_18_30 | 0.75      | 0.79    | +0.04   |
| age_30_50 | 0.80      | 0.83    | +0.03   |
| age_50_70 | 0.77      | 0.78    | +0.01   |
| age_70+   | 0.65      | 0.72    | +0.07   |

### Recommendation
All metrics improved. Slice analysis shows consistent improvement
across all age groups, with the largest improvement in the previously
underperforming 70+ group. **Recommend merge.**
"""

    print(report)

    # CML commands that would generate this
    print("\n  CML Commands (in CI/CD):")
    print("  " + "-" * 50)
    cml_commands = [
        'dvc repro                              # Run pipeline',
        'dvc metrics diff --md >> report.md     # Metrics diff table',
        'dvc params diff --md >> report.md      # Params diff table',
        'cml comment create report.md           # Post to PR',
    ]
    for cmd in cml_commands:
        print(f"    {cmd}")

    return report


# ============================================================
# Exercise 5: MLflow vs DVC Comparison
# ============================================================

def exercise_5_mlflow_vs_dvc():
    """Compare MLflow and DVC across key dimensions."""

    print("MLflow vs DVC Comparison")
    print("=" * 60)

    comparison = {
        "Primary Purpose": {
            "MLflow": "Experiment tracking + model registry + serving",
            "DVC": "Data/model versioning + pipeline reproducibility",
        },
        "Data Versioning": {
            "MLflow": "Artifacts (manual, per-run)",
            "DVC": "Git-like data versioning (automatic, content-addressed)",
        },
        "Experiment Tracking": {
            "MLflow": "Built-in UI, parameters, metrics, artifacts",
            "DVC": "Git branches/tags + dvc metrics + Studio (optional)",
        },
        "Pipeline Definition": {
            "MLflow": "MLflow Projects (MLproject file)",
            "DVC": "dvc.yaml with DAG, dependency tracking, caching",
        },
        "Reproducibility": {
            "MLflow": "Via Projects + conda/docker environments",
            "DVC": "Full DAG reproduction with dvc repro, lock files",
        },
        "Model Registry": {
            "MLflow": "Built-in (staging/production/archived stages)",
            "DVC": "Via Git tags + GTO (optional), or integrate with MLflow",
        },
        "Model Serving": {
            "MLflow": "Built-in (REST API, batch, Spark UDF)",
            "DVC": "Not included (use external tools)",
        },
        "Storage Backend": {
            "MLflow": "Local, S3, Azure Blob, GCS (for artifacts)",
            "DVC": "Local, S3, GCS, Azure, SSH, HTTP (for data+models)",
        },
        "Team Collaboration": {
            "MLflow": "MLflow Tracking Server (centralized)",
            "DVC": "Git workflows (PR-based) + DVC Studio (optional)",
        },
        "CI/CD Integration": {
            "MLflow": "API-based integration",
            "DVC": "Native (CML for GitHub/GitLab, dvc repro in CI)",
        },
        "Learning Curve": {
            "MLflow": "Low (Python API, familiar concepts)",
            "DVC": "Medium (Git concepts, pipeline YAML)",
        },
        "Best For": {
            "MLflow": "Teams focused on experiment comparison and model serving",
            "DVC": "Teams needing data versioning, reproducibility, and Git-based workflows",
        },
    }

    for dimension, tools in comparison.items():
        print(f"\n  {dimension}:")
        print(f"    MLflow: {tools['MLflow']}")
        print(f"    DVC:    {tools['DVC']}")

    print("\n\n  Recommendation Matrix:")
    print(f"  {'-'*60}")
    print(f"  {'Scenario':<45s} {'Tool':>12s}")
    print(f"  {'-'*60}")
    scenarios = [
        ("Need experiment tracking UI", "MLflow"),
        ("Need data versioning (large datasets)", "DVC"),
        ("Need model serving endpoint", "MLflow"),
        ("Need reproducible pipelines", "DVC"),
        ("Need model registry with stages", "MLflow"),
        ("Need Git-based collaboration", "DVC"),
        ("Need CI/CD for ML", "DVC + CML"),
        ("Need end-to-end platform", "Both together"),
    ]
    for scenario, tool in scenarios:
        print(f"  {scenario:<45s} {tool:>12s}")

    print(f"\n  Practical combination: DVC for data/pipeline + MLflow for tracking/registry")

    return comparison


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Exercise 1: Initialize DVC Project")
    print("=" * 60)
    exercise_1_init_project()

    print("\n\n")
    print("Exercise 2: Three-Stage Pipeline")
    print("=" * 60)
    exercise_2_three_stage_pipeline()

    print("\n\n")
    print("Exercise 3: DVC Experiments")
    print("=" * 60)
    exercise_3_dvc_experiments()

    print("\n\n")
    print("Exercise 4: CML PR Report")
    print("=" * 60)
    exercise_4_cml_report()

    print("\n\n")
    print("Exercise 5: MLflow vs DVC")
    print("=" * 60)
    exercise_5_mlflow_vs_dvc()
