# 학습 루프 (Training Loop)

**이전**: [데이터셋과 데이터로더](./07_Dataset_and_DataLoader.md) | **다음**: [모델 저장과 로드](./09_Model_Saving_and_Loading.md)

---

## 학습 목표 (Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 순전파, 손실, 역전파, 옵티마이저 스텝이 포함된 완전한 학습 루프를 작성할 수 있습니다
2. 적절한 `eval()` 모드와 `no_grad()` 컨텍스트로 검증 루프를 구현할 수 있습니다
3. 에포크 전반에 걸쳐 학습 메트릭(손실, 정확도)을 추적하고 기록할 수 있습니다
4. 과적합을 방지하기 위한 조기 종료(early stopping)를 구현할 수 있습니다
5. 프로그레스 바와 로깅으로 학습을 모니터링할 수 있습니다
6. 가독성과 재사용성을 위해 학습 코드를 구조화할 수 있습니다
7. 일반적인 학습 문제(NaN 손실, 과적합, 과소적합)를 처리할 수 있습니다

---

학습 루프는 모든 것이 합쳐지는 곳입니다. 이 레슨에서는 깔끔하고 정확하며 프로덕션 품질의 학습 코드를 작성하는 방법을 가르칩니다.

---

## 1. 최소 학습 루프

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

X = torch.randn(200, 10)
y = torch.randint(0, 2, (200,))
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

for epoch in range(10):
    for batch_X, batch_y in loader:
        optimizer.zero_grad()           # 1. 그래디언트 초기화
        output = model(batch_X)         # 2. 순전파
        loss = loss_fn(output, batch_y) # 3. 손실 계산
        loss.backward()                 # 4. 역전파
        optimizer.step()                # 5. 파라미터 업데이트
    print(f"에포크 {epoch+1}, 손실: {loss.item():.4f}")
```

---

## 2. 학습 + 검증 루프

```python
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_X)
        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_X.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == batch_y).sum().item()
        total += batch_X.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        output = model(batch_X)
        loss = loss_fn(output, batch_y)

        total_loss += loss.item() * batch_X.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == batch_y).sum().item()
        total += batch_X.size(0)

    return total_loss / total, correct / total
```

---

## 3. 조기 종료 (Early Stopping)

```python
class EarlyStopping:
    """검증 손실이 개선되지 않으면 학습을 중단합니다."""

    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
```

---

## 4. 학습 곡선 기록

```python
import matplotlib.pyplot as plt

history = {'train_loss': [], 'val_loss': [],
           'train_acc': [], 'val_acc': []}

for epoch in range(num_epochs):
    t_loss, t_acc = train_one_epoch(model, train_loader,
                                     optimizer, loss_fn, device)
    v_loss, v_acc = validate(model, val_loader, loss_fn, device)

    history['train_loss'].append(t_loss)
    history['val_loss'].append(v_loss)
    history['train_acc'].append(t_acc)
    history['val_acc'].append(v_acc)

# 학습 곡선 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history['train_loss'], label='학습')
ax1.plot(history['val_loss'], label='검증')
ax1.set_xlabel('에포크')
ax1.set_ylabel('손실')
ax1.legend()
ax1.set_title('손실 곡선')

ax2.plot(history['train_acc'], label='학습')
ax2.plot(history['val_acc'], label='검증')
ax2.set_xlabel('에포크')
ax2.set_ylabel('정확도')
ax2.legend()
ax2.set_title('정확도 곡선')

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=100)
plt.close()
```

---

## 5. 일반적인 학습 문제

### 5.1 NaN 손실

```python
loss = loss_fn(output, batch_y)
if torch.isnan(loss):
    print("NaN 손실 감지!")
    print(f"  출력 범위: [{output.min():.4f}, {output.max():.4f}]")
    break

# 일반적 원인과 수정:
# 1. 학습률이 너무 높음 -> LR 감소
# 2. 수치 오버플로 -> 그래디언트 클리핑 사용
# 3. 0의 로그 -> 엡실론 추가 (log(x + 1e-8))
```

### 5.2 그래디언트 클리핑

```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()  # 클리핑은 step 전에
```

### 5.3 과적합 증상과 해결

```
증상: 학습 손실은 감소, 검증 손실은 증가
해결:
- 드롭아웃 추가: nn.Dropout(p=0.5)
- 가중치 감쇠 추가: optimizer = Adam(params, weight_decay=1e-4)
- 모델 크기 축소
- 데이터 증강
- 조기 종료
```

### 5.4 과소적합 증상과 해결

```
증상: 학습 손실이 감소하지 않음
해결:
- 모델 용량 증가: 더 많은 레이어/뉴런
- 학습률 증가
- 더 많은 에포크 학습
- 과도한 정규화 제거
```

---

## 요약

| 개념 | 핵심 내용 |
|------|----------|
| 학습 스텝 | zero_grad -> forward -> loss -> backward -> step |
| 검증 | `model.eval()` + `torch.no_grad()`, optimizer step 없음 |
| 메트릭 | loss * batch_size를 누적, 전체 샘플로 나누기 |
| 조기 종료 | val loss가 `patience` 에포크 동안 개선되지 않으면 중단 |
| 그래디언트 클리핑 | `optimizer.step()` 전에 `clip_grad_norm_` |
| 학습 곡선 | train vs val 손실/정확도를 시각화하여 문제 진단 |

---

**다음**: [모델 저장과 로드](./09_Model_Saving_and_Loading.md) -- 체크포인팅, state dict, 모델 내보내기.
