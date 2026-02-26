# DVC — Data Version Control

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why Git alone is insufficient for ML project versioning and describe how DVC's metafile approach bridges the gap for large data and model files
2. Initialize DVC in a project, track datasets and model artifacts, and connect to remote storage backends (S3, GCS, Azure Blob)
3. Define DVC pipelines with `dvc.yaml` and `params.yaml` to create reproducible, DAG-based ML workflows with dependency tracking
4. Use DVC experiments to compare runs across different hyperparameters and data versions, and visualize results with `dvc plots`
5. Integrate DVC with GitHub Actions and CML (Continuous Machine Learning) to automate training and report model metrics in pull requests

---

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
  # Each stage declares deps, params, outs — DVC hashes these to skip unchanged stages
  prepare:
    cmd: python src/prepare.py
    deps:
      # Both code and data as deps — changing either triggers re-run
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
      # cache: false keeps metrics in git — enables dvc metrics diff across commits
      - metrics/train_metrics.json:
          cache: false
    plots:
      # x/y mapping lets dvc plots render interactive charts without extra code
      - metrics/roc_curve.csv:
          x: fpr
          y: tpr

  evaluate:
    cmd: python src/evaluate.py
    deps:
      # Depends on model + test data — changing training code triggers full downstream
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
    # Load params from params.yaml — single source of truth for hyperparameters
    # DVC tracks params.yaml changes to decide which stages need re-running
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

    # JSON format enables dvc metrics diff — DVC parses and compares values across commits
    Path("metrics").mkdir(exist_ok=True)
    with open("metrics/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model — DVC caches this artifact and links it to the commit hash
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
      # CML and DVC installed as separate steps — version-pinned actions ensure reproducibility
      - uses: iterative/setup-cml@v2
      - uses: iterative/setup-dvc@v1

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Pull data
        run: dvc pull
        env:
          # Credentials as secrets, not in config — never expose keys in repo
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Train and evaluate
        # dvc repro skips unchanged stages — CI runs only what the PR actually changed
        run: dvc repro

      - name: Create CML report
        env:
          REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          # --md outputs a markdown table — renders natively in PR comments
          echo "## Model Metrics" >> report.md
          echo "" >> report.md
          dvc metrics diff --md >> report.md
          echo "" >> report.md

          # Plots
          echo "## Plots" >> report.md
          dvc plots diff --open >> report.md

          # PR comment makes metrics visible in code review — no need to check logs
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
          # PERSONAL_ACCESS_TOKEN (not GITHUB_TOKEN) — needs repo scope to register self-hosted runners
          REPO_TOKEN: ${{ secrets.PERSONAL_ACCESS_TOKEN }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          # CML provisions a GPU instance on-demand — no persistent infrastructure cost
          cml runner launch \
            --cloud aws \
            --cloud-region us-east-1 \
            --cloud-type g4dn.xlarge \
            --labels cml-gpu

  train:
    # needs: ensures the GPU runner is ready before training starts
    needs: launch-runner
    # Label matching routes this job to the CML-provisioned GPU instance
    runs-on: [self-hosted, cml-gpu]
    steps:
      - uses: actions/checkout@v4
      - name: Train on GPU
        run: |
          pip install -r requirements.txt
          dvc pull
          dvc repro
          # Push updated artifacts back — other branches/PRs can pull the latest model
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
# Environment variables preferred in CI/CD — no credential files to manage
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

# For local development — --local flag is critical for security
dvc remote modify --local s3remote access_key_id mykey
dvc remote modify --local s3remote secret_access_key mysecret
# --local stores in .dvc/config.local (gitignored) — never leaks to repo

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

---

## Exercises

### Exercise 1: Initialize a DVC Project

Set up a minimal DVC-tracked ML project from scratch:

1. Create a new directory, initialize Git, then run `dvc init`
2. Create a small CSV file (`data/sample.csv`) with at least 100 rows of synthetic data and track it with `dvc add`
3. Inspect the generated `.dvc` metafile — what fields does it contain, and what is the purpose of the `md5` hash?
4. Commit the metafile to Git. Then modify the CSV (add 10 more rows), re-track with `dvc add`, and commit the updated metafile
5. Run `dvc diff HEAD~1` and interpret the output. What changed between commits?
6. Configure a local directory (e.g., `/tmp/dvc-remote`) as a DVC remote and push your data: `dvc push`

### Exercise 2: Build a Three-Stage DVC Pipeline

Using the `dvc.yaml` structure from Section 3.1 as a reference, create a three-stage pipeline for a text classification task:

1. **`prepare` stage**: Read a raw CSV file (`data/raw.csv`) and split it 80/20 into train and test sets. Parameterize the split ratio and random seed in `params.yaml`
2. **`featurize` stage**: Convert text in the train/test sets to TF-IDF features. Parameterize `max_features` and `ngram_range`
3. **`train` stage**: Train a logistic regression model on the features. Parameterize `C` (regularization) and `max_iter`. Save metrics to `metrics/eval.json` with at least `accuracy` and `f1`

Then:
- Run `dvc repro` and confirm the full pipeline executes
- Change `max_features` in `params.yaml` and run `dvc repro` again — confirm only the `featurize` and `train` stages re-run (not `prepare`)
- Run `dvc metrics show` and `dvc dag` to inspect results

### Exercise 3: Hyperparameter Search with DVC Experiments

Using the pipeline from Exercise 2, conduct a hyperparameter search:

1. Queue five experiments varying the `C` parameter: `[0.01, 0.1, 1.0, 10.0, 100.0]`
2. Run all experiments in parallel with `dvc exp run --run-all --parallel 3`
3. Display results with `dvc exp show` — which value of `C` gives the best F1 score?
4. Apply the best experiment to the workspace with `dvc exp apply`
5. Commit the winning parameters to Git with a descriptive message (e.g., `"feat: tuned C=10 gives F1=0.91"`)
6. Compare the best and worst experiments with `dvc exp diff`

### Exercise 4: CML Pull Request Report

Set up a GitHub Actions workflow using CML that automatically posts a model metrics report on every pull request:

1. Create `.github/workflows/cml_report.yaml` that triggers on `pull_request`
2. The workflow should: install DVC and CML, pull data from a configured remote, run `dvc repro`, and generate a report
3. The report (`report.md`) must include: a metrics comparison table (`dvc metrics diff --md`), and a note about which stage(s) were re-executed
4. Use `cml comment create report.md` to post the report as a PR comment
5. Describe what secrets you would need to configure in GitHub and why (e.g., `AWS_ACCESS_KEY_ID` for S3 remote access)

### Exercise 5: MLflow vs DVC Comparison

Write a structured comparison of MLflow and DVC for experiment tracking in a team ML project:

1. List three capabilities that DVC provides that MLflow does not (or does poorly)
2. List three capabilities that MLflow provides that DVC does not (or does poorly)
3. Describe a project scenario where you would choose DVC over MLflow, and explain why
4. Describe a project scenario where you would choose MLflow over DVC, and explain why
5. Propose an architecture where both tools are used together — what role does each play, and how do they complement each other without duplication?
