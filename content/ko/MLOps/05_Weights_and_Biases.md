# Weights & Biases (W&B)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Weights & Biases의 핵심 기능인 Experiments, Sweeps, Artifacts, Tables, Reports를 설명하고 각각을 언제 사용해야 하는지 서술할 수 있다
2. `wandb.init`, `wandb.log`, `wandb.finish`를 사용하여 학습 루프(training loop)에 메트릭(metric), 하이퍼파라미터(hyperparameter), 모델 체크포인트(checkpoint) 추적을 추가할 수 있다
3. 그리드(grid), 랜덤(random), 베이지안(Bayesian) 탐색 전략을 사용한 자동 하이퍼파라미터 최적화를 위해 W&B Sweeps를 구성하고 실행할 수 있다
4. W&B Artifacts를 사용하여 데이터셋과 모델을 버전 관리하고 재현 가능한 실험 계보(lineage) 추적을 구현할 수 있다
5. 실험 결과와 시각화를 동료와 공유하는 W&B Reports를 생성할 수 있다

---

## 1. W&B 개요

Weights & Biases는 ML 실험 추적, 하이퍼파라미터 튜닝, 모델 관리를 위한 플랫폼입니다.

### 1.1 핵심 기능

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Weights & Biases 기능                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   │ Experiments │    │   Sweeps    │    │  Artifacts  │            │
│   │             │    │             │    │             │            │
│   │ - 실험 추적  │    │ - 하이퍼     │    │ - 데이터셋   │            │
│   │ - 메트릭    │    │   파라미터   │    │ - 모델      │            │
│   │ - 시각화    │    │   튜닝      │    │ - 버전 관리  │            │
│   └─────────────┘    └─────────────┘    └─────────────┘            │
│                                                                     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   │   Tables    │    │   Reports   │    │   Models    │            │
│   │             │    │             │    │             │            │
│   │ - 데이터    │    │ - 문서화    │    │ - 모델      │            │
│   │   시각화    │    │ - 공유      │    │   레지스트리 │            │
│   └─────────────┘    └─────────────┘    └─────────────┘            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 설치 및 설정

```bash
# 설치
pip install wandb

# 로그인
wandb login
# API 키 입력 (https://wandb.ai/authorize)

# 환경 변수로 설정
export WANDB_API_KEY=your-api-key
```

```python
# Python에서 로그인
import wandb
wandb.login(key="your-api-key")
```

---

## 2. 기본 실험 추적

### 2.1 첫 번째 실험

```python
"""
W&B 기본 사용법
"""

import wandb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 데이터 준비
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# W&B 초기화 — 이후 모든 wandb.log() 호출이 이 run에 연결됨;
# config는 스냅샷으로 저장되므로 나중에 파라미터를 변경해도 로깅된 기록에 영향 없음
wandb.init(
    project="iris-classification",    # 프로젝트 이름
    name="random-forest-baseline",    # 실행 이름
    config={                          # 하이퍼파라미터
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    },
    tags=["baseline", "random-forest"],
    notes="Initial baseline experiment"
)

# wandb.config을 통해 접근 — 하이퍼파라미터의 단일 진실 공급원(single source of truth)을 보장하여
# Sweeps가 코드 변경 없이 값을 오버라이드 가능
config = wandb.config

# 모델 학습
model = RandomForestClassifier(
    n_estimators=config.n_estimators,
    max_depth=config.max_depth,
    random_state=config.random_state
)
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 메트릭 로깅
wandb.log({
    "accuracy": accuracy,
    "test_size": len(X_test),
    "train_size": len(X_train)
})

# 실행 종료
wandb.finish()
```

### 2.2 학습 과정 로깅

```python
"""
학습 과정 실시간 로깅
"""

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 초기화
wandb.init(project="pytorch-training")

# 모델 정의
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=wandb.config.get("lr", 0.001))

# wandb.watch가 autograd에 훅을 걸어 그래디언트와 파라미터 노름을 자동 기록 —
# 수동 계측 없이 기울기 소실/폭발(vanishing/exploding gradient) 진단 가능
wandb.watch(model, criterion, log="all", log_freq=100)

# 학습 루프
for epoch in range(wandb.config.get("epochs", 10)):
    model.train()
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # 배치별 로깅 (선택)
        if batch_idx % 100 == 0:
            wandb.log({
                "batch_loss": loss.item(),
                "epoch": epoch,
                "batch": batch_idx
            })

    # 에폭별 로깅
    avg_loss = train_loss / len(train_loader)
    val_accuracy = evaluate(model, val_loader)

    wandb.log({
        "epoch": epoch,
        "train_loss": avg_loss,
        "val_accuracy": val_accuracy
    })

    # 개선 시에만 저장 — 중복 체크포인트 저장 방지;
    # wandb.save가 클라우드로 업로드하므로 머신이 선점(preempt)되어도 체크포인트 보존
    if val_accuracy > best_accuracy:
        torch.save(model.state_dict(), "best_model.pth")
        wandb.save("best_model.pth")
        best_accuracy = val_accuracy

wandb.finish()
```

### 2.3 다양한 데이터 로깅

```python
"""
다양한 데이터 타입 로깅
"""

import wandb
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

wandb.init(project="data-logging-demo")

# 1. 이미지 로깅
images = wandb.Image(
    np.random.rand(100, 100, 3),
    caption="Random Image"
)
wandb.log({"random_image": images})

# PIL 이미지
pil_image = Image.open("sample.png")
wandb.log({"pil_image": wandb.Image(pil_image)})

# 여러 이미지
wandb.log({
    "examples": [wandb.Image(img, caption=f"Sample {i}")
                 for i, img in enumerate(image_batch[:10])]
})

# 2. 플롯 로깅
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
ax.set_title("Training Curve")
wandb.log({"plot": wandb.Image(fig)})
plt.close()

# 또는 plotly 사용
import plotly.express as px
fig = px.scatter(x=[1, 2, 3], y=[1, 4, 9])
wandb.log({"plotly_chart": fig})

# 3. 히스토그램
wandb.log({"predictions": wandb.Histogram(predictions)})

# 4. 테이블
columns = ["id", "image", "prediction", "label"]
data = [
    [i, wandb.Image(img), pred, label]
    for i, (img, pred, label) in enumerate(zip(images, preds, labels))
]
table = wandb.Table(columns=columns, data=data)
wandb.log({"predictions_table": table})

# 5. Confusion Matrix
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        y_true=y_true,
        preds=y_pred,
        class_names=class_names
    )
})

# 6. ROC Curve
wandb.log({
    "roc_curve": wandb.plot.roc_curve(
        y_true, y_scores, labels=class_names
    )
})

# 7. PR Curve
wandb.log({
    "pr_curve": wandb.plot.pr_curve(
        y_true, y_scores, labels=class_names
    )
})

wandb.finish()
```

---

## 3. Sweeps (하이퍼파라미터 튜닝)

### 3.1 Sweep 설정

```python
"""
W&B Sweeps 설정
"""

import wandb

# Sweep 설정
sweep_config = {
    "name": "hyperparam-sweep",
    # 베이지안(Bayesian) 방법이 메트릭을 파라미터의 함수로 모델링 — 유망한 영역을
    # 더 밀도 있게 탐색하여 random/grid보다 빠르게 수렴
    "method": "bayes",
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-1
        },
        "batch_size": {
            "values": [16, 32, 64, 128]
        },
        "epochs": {
            "value": 50  # 고정값
        },
        "optimizer": {
            "values": ["adam", "sgd", "rmsprop"]
        },
        "hidden_dim": {
            "distribution": "int_uniform",
            "min": 32,
            "max": 256
        },
        "dropout": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.5
        }
    },
    # Hyperband 조기 종료(early termination)로 성능이 낮은 실행을 조기에 중단 —
    # 현재 최고 성능을 이길 가능성이 낮은 실행의 컴퓨팅 비용을 절약
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 5,
        "eta": 3
    }
}

# Sweep 생성
sweep_id = wandb.sweep(sweep_config, project="my-project")
print(f"Sweep ID: {sweep_id}")
```

### 3.2 Sweep Agent 실행

```python
"""
Sweep 학습 함수
"""

import wandb
import torch

def train_sweep():
    """Sweep에서 실행될 학습 함수"""
    # W&B 초기화 (sweep이 config를 제공)
    wandb.init()
    config = wandb.config

    # 모델 생성
    model = create_model(
        hidden_dim=config.hidden_dim,
        dropout=config.dropout
    )

    # 옵티마이저 설정
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    # 데이터로더
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # 학습
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_accuracy = evaluate(model, val_loader)

        wandb.log({
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
            "epoch": epoch
        })

    wandb.finish()

# Sweep 실행
wandb.agent(
    sweep_id,
    function=train_sweep,
    count=50  # 최대 실행 횟수
)
```

### 3.3 CLI에서 Sweep 실행

```bash
# sweep.yaml 파일 생성
# sweep 시작
wandb sweep sweep.yaml

# Agent 실행 (여러 머신에서 병렬 가능)
wandb agent username/project/sweep_id
```

```yaml
# sweep.yaml
name: hyperparameter-sweep
method: bayes
metric:
  name: val_accuracy
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 0.00001
    max: 0.1
  batch_size:
    values: [16, 32, 64]
  hidden_dim:
    distribution: int_uniform
    min: 64
    max: 512
```

---

## 4. Artifacts

### 4.1 데이터셋 버전 관리

```python
"""
W&B Artifacts로 데이터셋 관리
"""

import wandb

# 아티팩트 생성 및 업로드
wandb.init(project="dataset-versioning")

# 데이터셋 아티팩트 생성
dataset_artifact = wandb.Artifact(
    name="mnist-dataset",
    type="dataset",
    description="MNIST dataset for classification",
    metadata={
        "size": 70000,
        "classes": 10,
        "source": "torchvision"
    }
)

# 파일/디렉토리 추가
dataset_artifact.add_file("data/train.csv")
dataset_artifact.add_dir("data/images/")

# add_reference는 데이터가 아닌 포인터를 저장 — 대규모 데이터셋 복제를 피하면서도
# 계보 그래프(lineage graph)에서 추적 가능
dataset_artifact.add_reference("s3://bucket/large_data/")

# 업로드
wandb.log_artifact(dataset_artifact)
wandb.finish()
```

### 4.2 모델 아티팩트

```python
"""
모델 아티팩트 관리
"""

import wandb
import torch

wandb.init(project="model-artifacts")

# 학습 후...

# 모델 아티팩트 생성
model_artifact = wandb.Artifact(
    name="churn-model",
    type="model",
    description="Customer churn prediction model",
    metadata={
        "accuracy": 0.95,
        "framework": "pytorch",
        "architecture": "MLP"
    }
)

# 모델 파일 저장 및 추가
torch.save(model.state_dict(), "model.pth")
model_artifact.add_file("model.pth")

# 설정 파일을 모델과 함께 번들링 — 추론이 동일한 하이퍼파라미터를 사용하도록 보장
model_artifact.add_file("config.yaml")

# 업로드
wandb.log_artifact(model_artifact)

# 별칭(alias)은 가변 포인터(git tag와 유사) — "production"이 항상 현재 서빙 버전을 가리켜
# 다운스트림 소비자가 버전에 무관하게 사용 가능
wandb.run.link_artifact(model_artifact, "model-registry/churn-model", aliases=["latest", "production"])

wandb.finish()
```

### 4.3 아티팩트 사용

```python
"""
아티팩트 다운로드 및 사용
"""

import wandb

wandb.init(project="using-artifacts")

# 아티팩트 다운로드
artifact = wandb.use_artifact("mnist-dataset:latest")  # 또는 :v0, :v1 등
artifact_dir = artifact.download()

print(f"Downloaded to: {artifact_dir}")

# 아티팩트 파일 직접 접근
with artifact.file("train.csv") as f:
    df = pd.read_csv(f)

# 의존성 기록 (이 run이 이 artifact를 사용함)
# use_artifact()가 자동으로 처리

wandb.finish()
```

### 4.4 아티팩트 리니지

```python
"""
아티팩트 리니지 추적
"""

import wandb

# 데이터 → 학습 → 모델 리니지
wandb.init(project="lineage-demo")

# 1. 입력 아티팩트 (데이터셋)
dataset = wandb.use_artifact("processed-data:latest")

# 2. 학습 수행
# ...

# 3. 출력 아티팩트 (모델)
model_artifact = wandb.Artifact("trained-model", type="model")
model_artifact.add_file("model.pth")
wandb.log_artifact(model_artifact)

# W&B UI에서 전체 리니지 그래프 확인 가능
# 데이터셋 → (학습 run) → 모델

wandb.finish()
```

---

## 5. MLflow와 비교

### 5.1 기능 비교

```python
"""
MLflow vs W&B 비교
"""

comparison = {
    "실험 추적": {
        "MLflow": "오픈소스, self-hosted",
        "W&B": "SaaS 기반, 무료 티어 제공"
    },
    "시각화": {
        "MLflow": "기본 시각화",
        "W&B": "풍부한 시각화, 실시간 업데이트"
    },
    "협업": {
        "MLflow": "제한적",
        "W&B": "팀 기능, 리포트 공유"
    },
    "하이퍼파라미터 튜닝": {
        "MLflow": "외부 도구 필요 (Optuna 등)",
        "W&B": "Sweeps 내장"
    },
    "모델 레지스트리": {
        "MLflow": "완전한 기능",
        "W&B": "Model Registry (최근 추가)"
    },
    "배포": {
        "MLflow": "MLflow Serving",
        "W&B": "직접 지원 없음 (다른 도구 연동)"
    },
    "비용": {
        "MLflow": "무료 (인프라 비용만)",
        "W&B": "무료 티어 + 유료 플랜"
    }
}
```

### 5.2 함께 사용하기

```python
"""
MLflow와 W&B 동시 사용
"""

import mlflow
import wandb

# 이중 추적: MLflow는 모델 레지스트리/서빙, W&B는 시각화/협업 —
# 각 플랫폼의 강점이 상호 보완
wandb.init(project="dual-tracking")
mlflow.set_experiment("dual-tracking")

with mlflow.start_run():
    # 공통 설정
    params = {"lr": 0.001, "epochs": 100}

    # 양쪽에 파라미터 로깅
    mlflow.log_params(params)
    wandb.config.update(params)

    # 학습 루프
    for epoch in range(params["epochs"]):
        loss = train_one_epoch()
        accuracy = evaluate()

        # 양쪽에 메트릭 로깅
        mlflow.log_metrics({"loss": loss, "accuracy": accuracy}, step=epoch)
        wandb.log({"loss": loss, "accuracy": accuracy, "epoch": epoch})

    # 모델 저장
    mlflow.sklearn.log_model(model, "model")
    wandb.save("model.pkl")

wandb.finish()
```

---

## 6. 고급 기능

### 6.1 팀 협업

```python
"""
팀 프로젝트 설정
"""

import wandb

# 팀 프로젝트에 로깅
wandb.init(
    entity="team-name",           # 팀 이름
    project="shared-project",     # 프로젝트 이름
    group="experiment-group",     # 실험 그룹 (관련 실험 묶기)
    job_type="training"           # 작업 유형
)
```

### 6.2 리포트 생성

```python
"""
W&B Reports API
"""

import wandb

# 리포트는 주로 UI에서 생성하지만 API로도 가능
api = wandb.Api()

# 프로젝트의 모든 실행 조회
runs = api.runs("username/project")

for run in runs:
    print(f"Run: {run.name}")
    print(f"  Config: {run.config}")
    print(f"  Summary: {run.summary}")
    print(f"  History: {run.history().shape}")
```

### 6.3 알림 설정

```python
"""
W&B Alerts
"""

import wandb

wandb.init(project="alerting-demo")

# 학습 중 알림 트리거
for epoch in range(100):
    accuracy = train_and_evaluate()

    if accuracy > 0.95:
        wandb.alert(
            title="High Accuracy Achieved!",
            text=f"Model achieved {accuracy:.2%} accuracy at epoch {epoch}",
            level=wandb.AlertLevel.INFO
        )

    if accuracy < 0.5:
        wandb.alert(
            title="Training Issue",
            text=f"Accuracy dropped to {accuracy:.2%}",
            level=wandb.AlertLevel.WARN
        )

    wandb.log({"accuracy": accuracy, "epoch": epoch})

wandb.finish()
```

---

## 연습 문제

### 문제 1: 기본 실험 추적
MNIST 데이터셋으로 CNN 모델을 학습하고 W&B로 실험을 추적하세요.

### 문제 2: Sweeps 실행
3개 이상의 하이퍼파라미터에 대해 Bayesian 최적화 sweep을 실행하세요.

### 문제 3: Artifacts
데이터셋과 모델을 아티팩트로 저장하고 리니지를 확인하세요.

---

## 요약

| 기능 | W&B | MLflow |
|------|-----|--------|
| 실험 추적 | wandb.log() | mlflow.log_metrics() |
| 하이퍼파라미터 튜닝 | Sweeps | 외부 도구 |
| 데이터/모델 버전 | Artifacts | Model Registry |
| 시각화 | 풍부한 대시보드 | 기본 UI |
| 협업 | 팀, 리포트 | 제한적 |
| 호스팅 | SaaS / Self-hosted | Self-hosted |

---

## 참고 자료

- [W&B Documentation](https://docs.wandb.ai/)
- [W&B Sweeps](https://docs.wandb.ai/guides/sweeps)
- [W&B Artifacts](https://docs.wandb.ai/guides/artifacts)
- [W&B Reports](https://docs.wandb.ai/guides/reports)
