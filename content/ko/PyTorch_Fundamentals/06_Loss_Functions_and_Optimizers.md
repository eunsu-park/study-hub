# 손실 함수와 옵티마이저 (Loss Functions and Optimizers)

**이전**: [nn.Module](./05_nn_Module.md) | **다음**: [데이터셋과 데이터로더](./07_Dataset_and_DataLoader.md)

---

## 학습 목표 (Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 분류, 회귀, 랭킹 작업에 적합한 손실 함수를 선택할 수 있습니다
2. `nn.CrossEntropyLoss`, `nn.MSELoss`, `nn.BCEWithLogitsLoss` 등을 구현하고 사용할 수 있습니다
3. 각 손실 함수의 수학적 공식을 이해할 수 있습니다
4. 옵티마이저(`SGD`, `Adam`, `AdamW`)를 적절한 하이퍼파라미터로 구성할 수 있습니다
5. 학습률 스케줄러를 적용하여 학습 수렴을 개선할 수 있습니다
6. 파라미터별 최적화(다른 레이어에 다른 학습률)를 사용할 수 있습니다
7. 손실, 그래디언트, 파라미터 업데이트 간의 관계를 설명할 수 있습니다

---

손실 함수는 모델 예측이 얼마나 잘못되었는지를 측정합니다. 옵티마이저는 손실의 그래디언트를 사용하여 모델 파라미터를 업데이트합니다. 이 둘이 학습 과정의 핵심을 형성합니다.

---

## 1. 분류용 손실 함수

### 1.1 CrossEntropyLoss

다중 클래스 분류의 주력 손실 함수입니다. `LogSoftmax`와 `NLLLoss`를 결합합니다:

```python
import torch
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()

# 원시 로짓 (소프트맥스 출력이 아님)
logits = torch.tensor([[2.0, 1.0, 0.1],    # 샘플 1: 클래스 0 예측
                        [0.1, 2.0, 0.3]])   # 샘플 2: 클래스 1 예측
targets = torch.tensor([0, 1])               # 실제 클래스

loss = loss_fn(logits, targets)
print(loss)  # tensor(0.4170)
```

**핵심 사항**:
- 입력: shape `[배치, 클래스 수]`의 원시 로짓 -- 먼저 softmax를 적용하지 마세요
- 타겟: shape `[배치]`의 클래스 인덱스 (정수, 원-핫이 아님)
- 클래스 가중치 지원: `nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 1.0]))`
- 레이블 스무딩 지원: `nn.CrossEntropyLoss(label_smoothing=0.1)`

### 1.2 BCEWithLogitsLoss

이진 또는 다중 레이블 분류에 사용합니다:

```python
loss_fn = nn.BCEWithLogitsLoss()

logits = torch.tensor([0.5, -1.0, 2.0])
targets = torch.tensor([1.0, 0.0, 1.0])

loss = loss_fn(logits, targets)
```

---

## 2. 회귀용 손실 함수

### 2.1 MSELoss (L2 손실)

```python
loss_fn = nn.MSELoss()

predictions = torch.tensor([2.5, 0.0, 2.0, 8.0])
targets = torch.tensor([3.0, -0.5, 2.0, 7.0])

loss = loss_fn(predictions, targets)
# 수식: mean((2.5-3)^2 + (0-(-0.5))^2 + (2-2)^2 + (8-7)^2) / 4
```

### 2.2 L1Loss (MAE)

```python
loss_fn = nn.L1Loss()
loss = loss_fn(predictions, targets)
```

### 2.3 손실 함수 선택 가이드

| 작업 | 손실 함수 | 입력 | 타겟 |
|------|----------|------|------|
| 다중 클래스 분류 | `CrossEntropyLoss` | 로짓 `[B, C]` | 클래스 인덱스 `[B]` |
| 이진 분류 | `BCEWithLogitsLoss` | 로짓 `[B]` | Float 0/1 `[B]` |
| 다중 레이블 분류 | `BCEWithLogitsLoss` | 로짓 `[B, C]` | Float 0/1 `[B, C]` |
| 회귀 | `MSELoss` | 예측 `[B]` | 타겟 `[B]` |

---

## 3. 옵티마이저

### 3.1 SGD

```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### 3.2 Adam / AdamW

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# AdamW: 가중치 감쇠를 올바르게 적용 (Adam보다 권장)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

### 3.3 옵티마이저 선택 가이드

| 옵티마이저 | 적합한 경우 | 기본 LR |
|-----------|-----------|---------|
| `SGD + 모멘텀` | CNN 학습, LR을 신중하게 조정할 때 | 0.01 - 0.1 |
| `Adam` | 빠른 프로토타이핑, 대부분의 작업 | 1e-3 |
| `AdamW` | Transformer, 가중치 감쇠 사용 시 | 1e-3 ~ 5e-5 |

---

## 4. 학습 스텝

```python
optimizer.zero_grad()       # 1. 이전 스텝의 그래디언트 초기화
output = model(x)           # 2. 순전파
loss = loss_fn(output, y)   # 3. 손실 계산
loss.backward()             # 4. 역전파 (그래디언트 계산)
optimizer.step()            # 5. 파라미터 업데이트
```

### 4.1 파라미터별 옵션

```python
optimizer = optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-4},  # 느린 LR
    {'params': model.head.parameters(), 'lr': 1e-3},      # 빠른 LR
], weight_decay=0.01)
```

### 4.2 그래디언트 누적

더 큰 배치 크기를 시뮬레이션합니다:

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (x, y) in enumerate(dataloader):
    output = model(x)
    loss = loss_fn(output, y) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 5. 학습률 스케줄러

### 5.1 StepLR

```python
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
# LR: 에포크 0-9는 0.01, 10-19는 0.001, ...
```

### 5.2 CosineAnnealingLR

```python
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
```

### 5.3 OneCycleLR

```python
from torch.optim.lr_scheduler import OneCycleLR
scheduler = OneCycleLR(optimizer, max_lr=0.01,
                        total_steps=len(dataloader) * num_epochs)
# 배치마다 scheduler.step() 호출 (에포크마다가 아님)
```

### 5.4 ReduceLROnPlateau

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
# scheduler.step(val_loss)  # 모니터링 메트릭을 전달
```

---

## 요약

| 개념 | 핵심 내용 |
|------|----------|
| CrossEntropyLoss | 다중 클래스 분류; 입력은 원시 로짓, softmax 아님 |
| BCEWithLogitsLoss | 이진/다중 레이블; 입력은 원시 로짓 |
| MSELoss | 회귀; L2 손실 |
| Adam/AdamW | 기본 옵티마이저 선택; 가중치 감쇠에는 AdamW |
| zero_grad -> forward -> loss -> backward -> step | 신성한 학습 시퀀스 |
| LR 스케줄러 | 학습 중 LR을 조정하여 수렴 개선 |
| 파라미터 그룹 | backbone과 head에 다른 LR/weight_decay |

---

**다음**: [데이터셋과 데이터로더](./07_Dataset_and_DataLoader.md) -- 효율적인 데이터 파이프라인 구축.
