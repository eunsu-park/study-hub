# DVC — 데이터 버전 관리

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. ML 프로젝트 버전 관리에 Git만으로는 부족한 이유를 설명하고 DVC의 메타파일(metafile) 방식이 대용량 데이터 및 모델 파일에 대한 간극을 어떻게 해소하는지 서술할 수 있다
2. 프로젝트에서 DVC를 초기화하고, 데이터셋과 모델 아티팩트를 추적하며, 원격 스토리지 백엔드(S3, GCS, Azure Blob)에 연결할 수 있다
3. `dvc.yaml`과 `params.yaml`로 DVC 파이프라인을 정의하여 의존성 추적(dependency tracking)이 있는 재현 가능한 DAG 기반 ML 워크플로우를 생성할 수 있다
4. DVC 실험(experiments)을 사용하여 다양한 하이퍼파라미터와 데이터 버전에 걸친 실행을 비교하고 `dvc plots`로 결과를 시각화할 수 있다
5. DVC를 GitHub Actions 및 CML(Continuous Machine Learning)과 통합하여 학습을 자동화하고 풀 리퀘스트(pull request)에서 모델 메트릭을 보고할 수 있다

---

## 개요

DVC(Data Version Control)는 대용량 파일, 데이터셋, ML 파이프라인을 처리할 수 있도록 Git을 확장한 도구입니다. Git이 코드를 추적하는 동안 DVC는 경량 메타파일(Metafile)을 이용해 데이터와 모델 아티팩트(Artifact)를 추적합니다. 이 레슨에서는 DVC의 기초, 원격 스토리지(Remote Storage), 파이프라인, 실험 추적, CI/CD 연동을 위한 CML, 그리고 ML 프로젝트에서의 데이터 버전 관리 모범 사례를 다룹니다.

---

## 1. DVC가 필요한 이유

### 1.1 데이터 버전 관리 문제

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

### 1.2 DVC 아키텍처(Architecture)

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

## 2. 시작하기

### 2.1 기본 명령어

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

### 2.2 데이터 버전 관리 워크플로우

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

## 3. DVC 파이프라인(Pipelines)

### 3.1 파이프라인 정의

```yaml
# dvc.yaml — ML pipeline definition
stages:
  # 각 스테이지가 deps, params, outs를 선언 — DVC가 해시하여 변경되지 않은 스테이지 건너뜀
  prepare:
    cmd: python src/prepare.py
    deps:
      # 코드와 데이터 모두 의존성(deps)으로 선언 — 어느 쪽이든 변경되면 재실행
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
      # cache: false는 메트릭(metrics)을 Git에 유지 — dvc metrics diff로 커밋 간 비교 가능
      - metrics/train_metrics.json:
          cache: false
    plots:
      # x/y 매핑으로 dvc plots가 별도 코드 없이 인터랙티브 차트 생성
      - metrics/roc_curve.csv:
          x: fpr
          y: tpr

  evaluate:
    cmd: python src/evaluate.py
    deps:
      # 모델 + 테스트 데이터에 의존 — 훈련 코드 변경 시 전체 다운스트림 재실행
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

### 3.2 파이프라인 실행

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

## 4. 실험 추적(Experiment Tracking)

### 4.1 DVC 실험

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

### 4.2 Python에서의 실험

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
    # params.yaml에서 로드 — 하이퍼파라미터의 단일 진실 공급원(single source of truth)
    # DVC가 params.yaml 변경을 추적하여 어떤 스테이지를 재실행할지 결정
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

    # JSON 형식으로 저장 — dvc metrics diff가 커밋 간 값을 파싱하여 비교
    Path("metrics").mkdir(exist_ok=True)
    with open("metrics/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 모델 저장 — DVC가 아티팩트(artifact)를 캐시하고 커밋 해시에 연결
    import pickle
    Path("models").mkdir(exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    train()
```

---

## 5. CML — 지속적 머신러닝(Continuous Machine Learning)

### 5.1 GitHub Actions와 CML

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
      # CML과 DVC를 별도 설치 — 버전 고정 액션(actions)으로 재현성 보장
      - uses: iterative/setup-cml@v2
      - uses: iterative/setup-dvc@v1

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Pull data
        run: dvc pull
        env:
          # 자격 증명(credentials)은 시크릿(secrets)으로 — 설정 파일이 아닌 환경 변수로 주입
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Train and evaluate
        # dvc repro는 변경되지 않은 스테이지 건너뜀 — PR이 실제 변경한 부분만 실행
        run: dvc repro

      - name: Create CML report
        env:
          REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          # --md로 마크다운 테이블 출력 — PR 코멘트에서 자동 렌더링
          echo "## Model Metrics" >> report.md
          echo "" >> report.md
          dvc metrics diff --md >> report.md
          echo "" >> report.md

          # Plots
          echo "## Plots" >> report.md
          dvc plots diff --open >> report.md

          # PR 코멘트로 메트릭 공개 — 로그를 확인할 필요 없이 코드 리뷰에서 바로 확인
          cml comment create report.md
```

### 5.2 CML 러너(Runner) — 클라우드 학습

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
          # PERSONAL_ACCESS_TOKEN 사용 (GITHUB_TOKEN이 아님) — 셀프 호스트 러너 등록에 repo 스코프 필요
          REPO_TOKEN: ${{ secrets.PERSONAL_ACCESS_TOKEN }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          # CML이 GPU 인스턴스를 온디맨드(on-demand) 프로비저닝 — 상시 인프라 비용 없음
          cml runner launch \
            --cloud aws \
            --cloud-region us-east-1 \
            --cloud-type g4dn.xlarge \
            --labels cml-gpu

  train:
    # needs: GPU 러너가 준비된 후에 훈련 시작 보장
    needs: launch-runner
    # 라벨 매칭(label matching)으로 CML이 프로비저닝한 GPU 인스턴스에 라우팅
    runs-on: [self-hosted, cml-gpu]
    steps:
      - uses: actions/checkout@v4
      - name: Train on GPU
        run: |
          pip install -r requirements.txt
          dvc pull
          dvc repro
          # 업데이트된 아티팩트를 원격에 푸시 — 다른 브랜치/PR에서 최신 모델 풀(pull) 가능
          dvc push
```

---

## 6. 원격 스토리지(Remote Storage) 설정

### 6.1 스토리지 백엔드(Storage Backends)

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

### 6.2 접근 제어(Access Control)

```bash
# CI/CD에서는 환경 변수 선호 — 자격 증명 파일 관리 불필요
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

# 로컬 개발용 — --local 플래그가 보안의 핵심
dvc remote modify --local s3remote access_key_id mykey
dvc remote modify --local s3remote secret_access_key mysecret
# --local은 .dvc/config.local에 저장 (gitignored) — 저장소에 유출되지 않음

# Push/pull specific files
dvc push data/training.csv.dvc   # Push specific file
dvc pull models/                  # Pull specific directory
```

---

## 7. 연습 문제

### 연습 1: DVC 파이프라인

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

### 연습 2: 실험 추적

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

## 8. 요약

### 핵심 정리

| 개념 | 설명 |
|------|------|
| **DVC** | 데이터를 위한 Git — 경량 메타파일을 통해 대용량 파일 추적 |
| **.dvc 파일** | 원격 스토리지의 데이터를 가리키는 포인터 (Git으로 추적) |
| **dvc.yaml** | 파이프라인 정의 (스테이지(Stage), 의존성, 파라미터, 출력, 메트릭) |
| **dvc repro** | 파이프라인 실행, 변경되지 않은 스테이지는 건너뜀 |
| **dvc exp** | 파라미터 탐색(Parameter Sweep)을 포함한 실험 추적 |
| **CML** | ML을 위한 CI/CD — PR에 자동 리포트 게시, 클라우드 러너 지원 |
| **원격 스토리지** | S3, GCS, Azure, SSH — 공유 데이터 스토리지 |

### 모범 사례

1. **모든 것을 추적** — 데이터, 모델, 설정, 파이프라인 정의 모두 버전 관리한다
2. **파이프라인 사용** — `dvc.yaml`로 실험을 재현 가능하게 만든다
3. **데이터를 Git에 커밋하지 않기** — `.dvc` 메타파일과 원격 스토리지를 사용한다
4. **CML로 자동화** — 메트릭 비교를 PR 코멘트로 자동 게시한다
5. **파라미터화** — 모든 하이퍼파라미터는 코드가 아닌 `params.yaml`에 정의한다
6. **릴리스 태그 지정** — 재현 가능한 릴리스를 위해 `git tag v1.0` + `dvc push`를 사용한다

### 다음 단계

- **L15**: LLMOps — LLM 애플리케이션의 운영 패턴
- **L03** (MLflow 기초)으로 돌아가 실험 추적 관점에서 MLflow와 DVC를 비교해보세요

---

## 연습 문제

### 연습 1: DVC 프로젝트 초기화

처음부터 최소 DVC 추적 ML 프로젝트를 설정하세요:

1. 새 디렉토리를 만들고 Git을 초기화한 다음 `dvc init`을 실행하세요
2. 최소 100행의 합성 데이터로 작은 CSV 파일(`data/sample.csv`)을 만들고 `dvc add`로 추적하세요
3. 생성된 `.dvc` 메타파일을 검사하세요 — 어떤 필드가 있으며, `md5` 해시의 목적은 무엇인가요?
4. 메타파일을 Git에 커밋하세요. 그런 다음 CSV를 수정하고(10행 추가), `dvc add`로 재추적하고, 업데이트된 메타파일을 커밋하세요
5. `dvc diff HEAD~1`을 실행하고 출력을 해석하세요. 커밋 간에 무엇이 변경되었나요?
6. 로컬 디렉토리(예: `/tmp/dvc-remote`)를 DVC 원격으로 설정하고 데이터를 푸시하세요: `dvc push`

### 연습 2: 3단계 DVC 파이프라인 구축

3.1절의 `dvc.yaml` 구조를 참고하여 텍스트 분류 작업을 위한 3단계 파이프라인을 만드세요:

1. **`prepare` 단계**: 원시 CSV 파일(`data/raw.csv`)을 읽고 80/20으로 훈련/테스트 세트로 분할하세요. 분할 비율과 랜덤 시드(random seed)를 `params.yaml`에서 파라미터화하세요
2. **`featurize` 단계**: 훈련/테스트 세트의 텍스트를 TF-IDF 피처로 변환하세요. `max_features`와 `ngram_range`를 파라미터화하세요
3. **`train` 단계**: 피처로 로지스틱 회귀(logistic regression) 모델을 훈련하세요. `C`(정규화, regularization)와 `max_iter`를 파라미터화하세요. 최소 `accuracy`와 `f1`을 포함하여 `metrics/eval.json`에 메트릭을 저장하세요

그런 다음:
- `dvc repro`를 실행하고 전체 파이프라인이 실행되는지 확인하세요
- `params.yaml`에서 `max_features`를 변경하고 `dvc repro`를 다시 실행하세요 — `featurize`와 `train` 단계만 재실행되는지(`prepare`는 재실행되지 않는지) 확인하세요
- `dvc metrics show`와 `dvc dag`를 실행하여 결과를 검사하세요

### 연습 3: DVC 실험으로 하이퍼파라미터 탐색

연습 2의 파이프라인을 사용하여 하이퍼파라미터 탐색을 수행하세요:

1. `C` 파라미터를 `[0.01, 0.1, 1.0, 10.0, 100.0]`로 변경하는 실험 5개를 큐에 추가하세요
2. `dvc exp run --run-all --parallel 3`으로 모든 실험을 병렬 실행하세요
3. `dvc exp show`로 결과를 표시하세요 — 어떤 `C` 값이 가장 높은 F1 점수를 내나요?
4. `dvc exp apply`로 최고의 실험을 워크스페이스에 적용하세요
5. 승리한 파라미터를 Git에 설명이 있는 메시지와 함께 커밋하세요(예: `"feat: tuned C=10 gives F1=0.91"`)
6. `dvc exp diff`로 최고와 최저 실험을 비교하세요

### 연습 4: CML 풀 리퀘스트 보고서

모든 풀 리퀘스트(pull request)에 자동으로 모델 메트릭 보고서를 게시하는 GitHub Actions 워크플로우를 CML로 설정하세요:

1. `pull_request`에서 트리거되는 `.github/workflows/cml_report.yaml`을 만드세요
2. 워크플로우는 다음을 수행해야 합니다: DVC와 CML 설치, 설정된 원격에서 데이터 풀(pull), `dvc repro` 실행, 보고서 생성
3. 보고서(`report.md`)에는 메트릭 비교 테이블(`dvc metrics diff --md`)과 재실행된 단계에 대한 메모가 포함되어야 합니다
4. `cml comment create report.md`를 사용하여 보고서를 PR 코멘트로 게시하세요
5. GitHub에서 설정해야 할 시크릿(secrets)과 그 이유를 설명하세요(예: S3 원격 접근을 위한 `AWS_ACCESS_KEY_ID`)

### 연습 5: MLflow vs DVC 비교

팀 ML 프로젝트에서 실험 추적을 위한 MLflow와 DVC의 구조적 비교를 작성하세요:

1. DVC가 제공하지만 MLflow가 제공하지 않는(또는 잘 못하는) 기능 3가지를 나열하세요
2. MLflow가 제공하지만 DVC가 제공하지 않는(또는 잘 못하는) 기능 3가지를 나열하세요
3. MLflow보다 DVC를 선택할 프로젝트 시나리오를 설명하고 그 이유를 설명하세요
4. DVC보다 MLflow를 선택할 프로젝트 시나리오를 설명하고 그 이유를 설명하세요
5. 두 도구를 함께 사용하는 아키텍처를 제안하세요 — 각 도구가 어떤 역할을 하며, 중복 없이 어떻게 서로를 보완하나요?
