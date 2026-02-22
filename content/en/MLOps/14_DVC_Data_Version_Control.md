# DVC — Data Version Control

## Overview

DVC (Data Version Control) extends Git to handle large files, datasets, and ML pipelines. While Git tracks code, DVC tracks data and model artifacts using lightweight metafiles. This lesson covers DVC fundamentals, remote storage, pipelines, experiment tracking, CML for CI/CD integration, and best practices for data versioning in ML projects.

---

## 1. Why DVC?

### 1.1 The Data Versioning Problem

```python
"""
Problem: Git doesn't handle large files well.

  Git Repository:
    src/train.py        ← 5 KB    ✓ Git tracks this fine
    configs/params.yaml ← 1 KB    ✓ Git tracks this fine
    data/training.csv   ← 5 GB    ✗ Git is not designed for this
    models/model.pkl    ← 2 GB    ✗ Git is not designed for this

  Common bad solutions:
    1. Don't version data → "Which dataset trained this model?"
    2. Git LFS → Expensive for large datasets, locks vendor
    3. Manual naming → data_v1.csv, data_v2_final.csv, data_v2_final_REAL.csv

DVC Solution:
  Git tracks: .dvc metafiles (small, text)
  DVC tracks: actual data files (in remote storage)

  Repository:
    src/train.py          ← Git
    configs/params.yaml   ← Git
    data/training.csv.dvc ← Git (metafile, ~100 bytes)
    models/model.pkl.dvc  ← Git (metafile, ~100 bytes)

  Remote Storage (S3/GCS/Azure):
    data/training.csv     ← DVC (actual 5GB file)
    models/model.pkl      ← DVC (actual 2GB file)
"""
```

### 1.2 DVC Architecture

```python
"""
DVC Architecture:

  ┌──────────────────────┐     ┌──────────────────────┐
  │   Git Repository     │     │   DVC Remote Storage  │
  │                      │     │   (S3, GCS, Azure)    │
  │  src/train.py        │     │                       │
  │  params.yaml         │     │  /ab/cd1234...  (data)│
  │  data.csv.dvc ───────│────▶│  /ef/gh5678...  (model)│
  │  model.pkl.dvc ──────│────▶│                       │
  │  dvc.yaml            │     └──────────────────────┘
  │  dvc.lock            │
  └──────────────────────┘

  .dvc metafile (data.csv.dvc):
    outs:
    - md5: abcd1234efgh5678...
      size: 5368709120
      path: data.csv

  Key Concepts:
    - .dvc files: pointers to data (tracked by Git)
    - dvc.yaml: pipeline definition
    - dvc.lock: exact versions of pipeline outputs
    - DVC cache: local cache of data files (~/.dvc/cache)
    - DVC remote: shared storage (S3, GCS, SSH, etc.)
"""
```

---

## 2. Getting Started

### 2.1 Basic Commands

```bash
# Initialize DVC in a Git repo
git init
dvc init

# Track a data file
dvc add data/training.csv
# Creates: data/training.csv.dvc (metafile)
# Adds:   data/training.csv to .gitignore

# Commit the metafile
git add data/training.csv.dvc data/.gitignore
git commit -m "Add training data v1"

# Configure remote storage
dvc remote add -d myremote s3://my-bucket/dvc-store
git add .dvc/config
git commit -m "Configure DVC remote"

# Push data to remote
dvc push

# Pull data from remote (on another machine)
git clone <repo>
dvc pull
```

### 2.2 Data Versioning Workflow

```bash
# Update data (new version)
# ... (update data/training.csv with new data)
dvc add data/training.csv
git add data/training.csv.dvc
git commit -m "Update training data v2"
dvc push

# Switch between data versions
git checkout v1.0           # Checkout code + .dvc files
dvc checkout                # Download matching data version

# Compare data versions
dvc diff HEAD~1             # Show data changes vs previous commit

# List tracked files
dvc list . --dvc-only       # Show all DVC-tracked files
```

---

## 3. DVC Pipelines

### 3.1 Pipeline Definition

```yaml
# dvc.yaml — ML pipeline definition
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - src/prepare.py
      - data/raw/
    params:
      - prepare.split_ratio
      - prepare.seed
    outs:
      - data/prepared/train.csv
      - data/prepared/test.csv

  featurize:
    cmd: python src/featurize.py
    deps:
      - src/featurize.py
      - data/prepared/
    params:
      - featurize.max_features
      - featurize.ngram_range
    outs:
      - data/features/

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/features/
    params:
      - train.n_estimators
      - train.learning_rate
      - train.max_depth
    outs:
      - models/model.pkl
    metrics:
      - metrics/train_metrics.json:
          cache: false
    plots:
      - metrics/roc_curve.csv:
          x: fpr
          y: tpr

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/model.pkl
      - data/prepared/test.csv
    metrics:
      - metrics/eval_metrics.json:
          cache: false
    plots:
      - metrics/confusion_matrix.csv:
          x: predicted
          y: actual
          template: confusion
```

```yaml
# params.yaml — hyperparameters
prepare:
  split_ratio: 0.2
  seed: 42

featurize:
  max_features: 5000
  ngram_range: [1, 2]

train:
  n_estimators: 200
  learning_rate: 0.1
  max_depth: 6
```

### 3.2 Running Pipelines

```bash
# Run the full pipeline
dvc repro

# Run from a specific stage
dvc repro train

# Force re-run (even if nothing changed)
dvc repro --force

# Show pipeline graph
dvc dag

# Output:
#   +----------+
#   | prepare  |
#   +----------+
#        |
#   +-----------+
#   | featurize |
#   +-----------+
#        |
#   +-------+
#   | train  |
#   +-------+
#        |
#   +----------+
#   | evaluate |
#   +----------+

# View metrics
dvc metrics show
# Output:
#   metrics/eval_metrics.json:
#     accuracy: 0.935
#     f1: 0.921
#     auc: 0.967

# Compare metrics across branches/commits
dvc metrics diff HEAD~1
# Output:
#   Path                       Metric    Old     New     Change
#   metrics/eval_metrics.json  accuracy  0.928   0.935   0.007
#   metrics/eval_metrics.json  f1        0.912   0.921   0.009

# View plots
dvc plots show
```

---

## 4. Experiment Tracking

### 4.1 DVC Experiments

```bash
# Run an experiment with modified parameters
dvc exp run --set-param train.n_estimators=500 --set-param train.learning_rate=0.05

# Run multiple experiments in parallel
dvc exp run --queue --set-param train.n_estimators=100
dvc exp run --queue --set-param train.n_estimators=200
dvc exp run --queue --set-param train.n_estimators=500
dvc exp run --run-all --parallel 3

# List experiments
dvc exp show
# Output:
#  ┌──────────────────────────────────────────────────────────────┐
#  │ Experiment              │ accuracy │ n_estimators │ lr     │
#  ├──────────────────────────────────────────────────────────────┤
#  │ workspace               │ 0.935    │ 200          │ 0.1    │
#  │ ├── exp-abc123          │ 0.941    │ 500          │ 0.05   │
#  │ ├── exp-def456          │ 0.928    │ 100          │ 0.1    │
#  │ └── exp-ghi789          │ 0.937    │ 200          │ 0.05   │
#  └──────────────────────────────────────────────────────────────┘

# Apply the best experiment to workspace
dvc exp apply exp-abc123

# Push experiment to a Git branch
dvc exp push origin exp-abc123

# Compare experiments
dvc exp diff exp-abc123 exp-def456

# Remove experiments
dvc exp remove exp-def456
```

### 4.2 Experiment in Python

```python
"""Pipeline stage that logs metrics for DVC experiment tracking."""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def train():
    # Load params
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    # Load data
    X_train = pd.read_csv("data/features/X_train.csv")
    y_train = pd.read_csv("data/features/y_train.csv").values.ravel()
    X_test = pd.read_csv("data/features/X_test.csv")
    y_test = pd.read_csv("data/features/y_test.csv").values.ravel()

    # Train
    model = GradientBoostingClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    # Save metrics (DVC tracks this file)
    Path("metrics").mkdir(exist_ok=True)
    with open("metrics/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model
    import pickle
    Path("models").mkdir(exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    train()
```

---

## 5. CML — Continuous Machine Learning

### 5.1 CML with GitHub Actions

```yaml
# .github/workflows/cml.yaml
name: CML Report

on:
  push:
    branches: [main]
  pull_request:

jobs:
  train-and-report:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - uses: iterative/setup-cml@v2
      - uses: iterative/setup-dvc@v1

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Pull data
        run: dvc pull
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Train and evaluate
        run: dvc repro

      - name: Create CML report
        env:
          REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          # Metrics comparison
          echo "## Model Metrics" >> report.md
          echo "" >> report.md
          dvc metrics diff --md >> report.md
          echo "" >> report.md

          # Plots
          echo "## Plots" >> report.md
          dvc plots diff --open >> report.md

          # Publish report as PR comment
          cml comment create report.md
```

### 5.2 CML Runner (Cloud Training)

```yaml
# .github/workflows/cml_cloud.yaml
name: CML Cloud Training

on:
  push:
    branches: [main]

jobs:
  launch-runner:
    runs-on: ubuntu-latest
    steps:
      - uses: iterative/setup-cml@v2
      - name: Launch cloud runner
        env:
          REPO_TOKEN: ${{ secrets.PERSONAL_ACCESS_TOKEN }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          cml runner launch \
            --cloud aws \
            --cloud-region us-east-1 \
            --cloud-type g4dn.xlarge \
            --labels cml-gpu

  train:
    needs: launch-runner
    runs-on: [self-hosted, cml-gpu]
    steps:
      - uses: actions/checkout@v4
      - name: Train on GPU
        run: |
          pip install -r requirements.txt
          dvc pull
          dvc repro
          dvc push
```

---

## 6. Remote Storage Configuration

### 6.1 Storage Backends

```bash
# Amazon S3
dvc remote add -d s3remote s3://my-bucket/dvc-store
dvc remote modify s3remote region us-east-1

# Google Cloud Storage
dvc remote add -d gcsremote gs://my-bucket/dvc-store

# Azure Blob Storage
dvc remote add -d azremote azure://my-container/dvc-store
dvc remote modify azremote account_name myaccount

# SSH / SFTP
dvc remote add -d sshremote ssh://user@host/path/to/dvc-store

# Local / NFS
dvc remote add -d localremote /mnt/shared/dvc-store

# HTTP (read-only, for sharing)
dvc remote add -d httpremote https://my-server.com/dvc-store
```

### 6.2 Access Control

```bash
# Use environment variables for credentials (CI/CD)
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

# Or configure in DVC
dvc remote modify --local s3remote access_key_id mykey
dvc remote modify --local s3remote secret_access_key mysecret
# --local stores in .dvc/config.local (gitignored)

# Push/pull specific files
dvc push data/training.csv.dvc   # Push specific file
dvc pull models/                  # Pull specific directory
```

---

## 7. Practice Problems

### Exercise 1: DVC Pipeline

```python
"""
Set up a DVC-tracked ML project:
1. Initialize Git + DVC
2. Add training data with dvc add
3. Create a 3-stage pipeline (prepare → train → evaluate) in dvc.yaml
4. Define params in params.yaml
5. Run dvc repro and verify metrics output
6. Change a hyperparameter and run again
7. Compare metrics: dvc metrics diff
8. Configure an S3 remote and push
"""
```

### Exercise 2: Experiment Tracking

```python
"""
Use DVC experiments for hyperparameter search:
1. Create a training pipeline with configurable hyperparameters
2. Queue 10 experiments with different parameter combinations
3. Run all experiments in parallel (dvc exp run --run-all --parallel 4)
4. View results with dvc exp show
5. Apply the best experiment
6. Set up a CML workflow that posts a report on PR
"""
```

---

## 8. Summary

### Key Takeaways

| Concept | Description |
|---------|-------------|
| **DVC** | Git for data — tracks large files via lightweight metafiles |
| **.dvc files** | Pointers to data in remote storage (tracked by Git) |
| **dvc.yaml** | Pipeline definition (stages, deps, params, outs, metrics) |
| **dvc repro** | Run pipeline, skip unchanged stages |
| **dvc exp** | Experiment tracking with parameter sweeps |
| **CML** | CI/CD for ML — auto-reports on PRs, cloud runners |
| **Remote storage** | S3, GCS, Azure, SSH — shared data storage |

### Best Practices

1. **Track everything** — data, models, configs, and pipeline definitions
2. **Use pipelines** — `dvc.yaml` makes experiments reproducible
3. **Never commit data to Git** — use `.dvc` metafiles + remote storage
4. **Automate with CML** — PR comments with metrics comparison
5. **Parameterize** — all hyperparameters in `params.yaml`, not in code
6. **Tag releases** — `git tag v1.0` + `dvc push` for reproducible releases

### Next Steps

- **L15**: LLMOps — operational patterns for LLM applications
- Return to **L03** (MLflow Basics) to compare MLflow vs DVC for experiment tracking
