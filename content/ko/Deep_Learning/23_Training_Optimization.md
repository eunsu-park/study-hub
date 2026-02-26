[이전: Vision Transformer (ViT)](./22_Impl_ViT.md) | [다음: 손실 함수](./24_Loss_Functions.md)

---

# 23. 학습 최적화

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 하이퍼파라미터 튜닝 전략 적용 (그리드, 랜덤, Optuna)
2. Warmup과 코사인 감쇠를 포함한 학습률 스케줄링 심화 구현
3. PyTorch AMP를 활용한 Mixed Precision Training 사용
4. 대규모 유효 배치를 위한 Gradient Accumulation 구현
5. `torch.compile()`로 학습 가속 (PyTorch 2.0+)
6. DDP와 FSDP로 멀티 GPU 학습 스케일링

---

## 1. 하이퍼파라미터 튜닝

### 주요 하이퍼파라미터

| 파라미터 | 영향 | 일반적 범위 |
|----------|------|------------|
| Learning Rate | 수렴 속도/안정성 | 1e-5 ~ 1e-2 |
| Batch Size | 메모리/일반화 | 16 ~ 512 |
| Weight Decay | 과적합 방지 | 1e-5 ~ 1e-2 |
| Dropout | 과적합 방지 | 0.1 ~ 0.5 |
| Epochs | 학습량 | 데이터 의존적 |

### 탐색 전략

```python
# Grid Search
learning_rates = [1e-4, 1e-3, 1e-2]
batch_sizes = [32, 64, 128]

for lr in learning_rates:
    for bs in batch_sizes:
        train_and_evaluate(lr, bs)

# Random Search (더 효율적)
import random
for _ in range(20):
    lr = 10 ** random.uniform(-5, -2)  # 로그 스케일
    bs = random.choice([16, 32, 64, 128])
    train_and_evaluate(lr, bs)
```

### Optuna 사용

```python
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)

    model = create_model(dropout)
    accuracy = train_and_evaluate(model, lr, batch_size)
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best params: {study.best_params}")
print(f"Best accuracy: {study.best_value}")
```

---

## 2. 학습률 스케줄링 심화

### Warmup

```python
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.base_lr * min(1.0, self.step_num / self.warmup_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

### Warmup + Cosine Decay

```python
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### OneCycleLR

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,      # 10% warmup
    anneal_strategy='cos'
)

# 매 배치마다 호출
for batch in train_loader:
    train_step(batch)
    scheduler.step()
```

---

## 3. Mixed Precision Training

### 개념

```
FP32 (32비트) → FP16 (16비트)
- 메모리 절약 (약 50%)
- 속도 향상 (약 2-3배)
- 정확도 유지
```

### PyTorch AMP

```python
# PyTorch 2.x 최신 API (권장)
scaler = torch.amp.GradScaler('cuda')

for data, target in train_loader:
    optimizer.zero_grad()

    # 자동 Mixed Precision
    with torch.amp.autocast('cuda'):
        output = model(data)
        loss = criterion(output, target)

    # 스케일링된 역전파
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

> **참고**: `torch.cuda.amp.autocast`와 `torch.cuda.amp.GradScaler`는 PyTorch 2.x부터 deprecated되었습니다. 대신 `torch.amp.autocast('cuda')`와 `torch.amp.GradScaler('cuda')`를 사용하세요.

### 전체 학습 루프

```python
def train_with_amp(model, train_loader, optimizer, epochs):
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                output = model(data)
                loss = F.cross_entropy(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
```

---

## 4. Gradient Accumulation

### 개념

```
작은 배치를 여러 번 → 큰 배치 효과
GPU 메모리 부족 시 유용
```

### 구현

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target)
    loss = loss / accumulation_steps  # 스케일링
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### AMP와 함께 사용

```python
accumulation_steps = 4
scaler = torch.amp.GradScaler('cuda')
optimizer.zero_grad()

for i, (data, target) in enumerate(train_loader):
    with torch.amp.autocast('cuda'):
        output = model(data)
        loss = criterion(output, target) / accumulation_steps

    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

---

## 5. Gradient Clipping

### 기울기 폭발 방지

```python
# Norm 클리핑 (권장)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Value 클리핑
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### 학습 루프에서

```python
for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    # 클리핑 후 업데이트
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

---

## 6. 조기 종료 심화

### Patience와 Delta

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = None
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best:
                    model.load_state_dict(self.best_weights)
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()
```

---

## 7. 학습 모니터링

### TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

for epoch in range(epochs):
    train_loss = train(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

writer.close()
```

### Weights & Biases

```python
import wandb

wandb.init(project="my-project", config={
    "learning_rate": lr,
    "batch_size": batch_size,
    "epochs": epochs
})

for epoch in range(epochs):
    train_loss = train(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

wandb.finish()
```

---

## 8. 재현성 (Reproducibility)

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

## 9. torch.compile() (PyTorch 2.0+)

### 개념

`torch.compile()`은 PyTorch 2.0의 핵심 기능입니다. 연산 그래프를 자동으로 추적하고 융합된 하드웨어 특화 커널을 생성하여 모델 코드 변경 없이 최적화합니다.

```
model → torch.compile(model) → 최적화된 모델
                                 - 연산자 융합(Operator Fusion)
                                 - 메모리 계획
                                 - 하드웨어 특화 커널 (GPU에서 Triton)
```

### 기본 사용법

```python
import torch

model = MyModel().cuda()

# 한 줄로 컴파일 — 코드 변경 불필요
compiled_model = torch.compile(model)

# 기존 모델과 동일하게 사용
output = compiled_model(input_data)
```

첫 번째 순전파에서 컴파일이 발생합니다 (수 초 소요). 이후 호출에서는 최적화된 커널을 사용합니다.

### 컴파일 모드

```python
# default: 균형 잡힌 최적화 (시작점으로 적합)
compiled = torch.compile(model)

# max-autotune: 느린 컴파일, 가장 빠른 추론
# 적합: 프로덕션 추론, 반복적 학습 실행
compiled = torch.compile(model, mode="max-autotune")

# reduce-overhead: CPU 오버헤드 최소화
# 적합: 커널 실행 오버헤드가 지배적인 작은 모델
compiled = torch.compile(model, mode="reduce-overhead")
```

| 모드 | 컴파일 시간 | 런타임 속도 | 적합한 용도 |
|------|-----------|-----------|-----------|
| `default` | 보통 | 좋음 | 범용 |
| `max-autotune` | 느림 | 최고 | 프로덕션, 대규모 모델 |
| `reduce-overhead` | 보통 | 소규모 모델에 좋음 | 작은 배치, 저지연 |

### torch.compile()로 학습

```python
model = MyModel().cuda()
compiled_model = torch.compile(model)
optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler('cuda')

for epoch in range(epochs):
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            output = compiled_model(data)  # 최적화된 순전파
            loss = F.cross_entropy(output, target)

        scaler.scale(loss).backward()  # 최적화된 역전파
        scaler.step(optimizer)
        scaler.update()
```

### 동적 형상(Dynamic Shapes)

기본적으로 `torch.compile()`은 입력 형상이 변경되면 재컴파일합니다. 가변 길이 입력(예: NLP)에는 `dynamic=True`를 사용하세요:

```python
# 형상 변경 시 재컴파일 방지
compiled_model = torch.compile(model, dynamic=True)
```

### 제한 사항과 팁

- **첫 호출이 느림**: 첫 순전파에서 컴파일이 발생합니다. 워밍업 시간을 고려하세요.
- **모든 연산이 지원되지 않음**: 대부분의 표준 PyTorch 연산은 동작합니다. 커스텀 CUDA 확장이나 특수한 연산은 eager 모드로 폴백될 수 있습니다.
- **디버깅**: `torch._dynamo.config.verbose = True`로 컴파일 대상을 확인할 수 있습니다.
- **그래프 브레이크**: 데이터 의존적 제어 흐름(예: `if x.sum() > 0`)은 그래프 브레이크를 유발하여 최적화 효과를 줄입니다.

```python
# 컴파일 대상 확인
torch._dynamo.config.verbose = True
compiled_model = torch.compile(model)
output = compiled_model(sample_input)  # 컴파일 정보 출력
```

---

## 10. 분산 학습 (DDP & FSDP)

### DataParallel vs. DistributedDataParallel

```
DataParallel (DP) — 간단하지만 느림 (GIL 병목, 단일 프로세스)
DistributedDataParallel (DDP) — GPU당 하나의 프로세스, 효율적 그래디언트 동기화
```

> 항상 DP보다 DDP를 사용하세요. `nn.DataParallel`은 멀티 GPU 학습에서 사실상 deprecated되었습니다.

### DDP 설정

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # 각 GPU가 데이터의 다른 부분 집합을 받음
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # 에포크마다 다르게 셔플
        for data, target in loader:
            data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()  # 그래디언트가 자동으로 동기화됨
            optimizer.step()

    cleanup()
```

### DDP 실행

```bash
# torchrun으로 실행 (권장, torch.distributed.launch 대체)
torchrun --nproc_per_node=4 train.py
```

```python
# train.py에서
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
```

### FSDP (Fully Sharded Data Parallel)

FSDP는 모델 파라미터, 그래디언트, 옵티마이저 상태를 GPU 간에 샤딩합니다 — 단일 GPU에 맞지 않는 모델의 학습을 가능하게 합니다.

```
DDP:  각 GPU가 전체 모델 복사본 보유 + 그래디언트 동기화
FSDP: 각 GPU가 파라미터의 샤드만 보유 + 필요 시 수집
      → GPU당 훨씬 적은 메모리
```

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

def train_fsdp(rank, world_size):
    setup(rank, world_size)

    model = LargeModel().to(rank)

    # FSDP로 래핑 — 파라미터가 GPU 간에 샤딩됨
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # 최대 메모리 절약
        device_id=rank,
    )

    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for data, target in loader:
            data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = fsdp_model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

    cleanup()
```

### FSDP 샤딩 전략

| 전략 | 메모리 절약 | 통신 비용 | 사용 사례 |
|------|-----------|----------|---------|
| `FULL_SHARD` | 최대 | 높음 | 매우 큰 모델 |
| `SHARD_GRAD_OP` | 중간 | 중간 | 중대형 모델 |
| `NO_SHARD` | 없음 (= DDP) | 최소 | 기준선 비교 |

### FSDP + torch.compile()

FSDP와 `torch.compile()`을 결합하면 메모리 효율성과 연산 속도 향상을 모두 얻을 수 있습니다:

```python
model = LargeModel().to(rank)

# 먼저 컴파일한 후 FSDP로 래핑
compiled_model = torch.compile(model)
fsdp_model = FSDP(compiled_model, device_id=rank)
```

### 언제 무엇을 사용할지

| 시나리오 | 권장 |
|---------|------|
| 단일 GPU | `torch.compile(model)` |
| 멀티 GPU, 모델이 GPU당 적합 | DDP + `torch.compile()` |
| 멀티 GPU, 모델이 너무 큰 경우 | FSDP |
| 멀티 GPU, 큰 모델 + 속도 | FSDP + `torch.compile()` |

---

## 정리

### 체크리스트

- [ ] 학습률 적절히 설정 (1e-4 시작 권장)
- [ ] Warmup 사용 (Transformer 필수)
- [ ] Mixed Precision 적용 (GPU 효율)
- [ ] Gradient Clipping (RNN/Transformer)
- [ ] 조기 종료 설정
- [ ] 재현성 시드 설정
- [ ] 로깅/모니터링 설정
- [ ] `torch.compile()` 사용하여 속도 향상 (PyTorch 2.0+)
- [ ] 멀티 GPU 학습 시 DDP/FSDP 고려

### 권장 설정

```python
# PyTorch 2.x 최적화 설정
model = MyModel().cuda()
compiled_model = torch.compile(model)  # 자동 최적화
optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = OneCycleLR(optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(loader))
scaler = torch.amp.GradScaler('cuda')  # AMP
early_stopping = EarlyStopping(patience=10)
```

---

## 연습 문제

### 연습 1: torch.compile() 속도 향상 측정

CIFAR-10에서 간단한 CNN을 학습하며 `torch.compile()` 유무에 따른 에포크당 학습 시간을 비교하는 스크립트를 작성하세요.

<details>
<summary>정답 보기</summary>

```python
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8 * 8, 256), nn.ReLU(), nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def train_one_epoch(model, loader, optimizer):
    model.train()
    for data, target in loader:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(data), target)
        loss.backward()
        optimizer.step()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

for use_compile in [False, True]:
    model = SimpleCNN().cuda()
    if use_compile:
        model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 워밍업
    train_one_epoch(model, loader, optimizer)

    # 측정
    start = time.time()
    train_one_epoch(model, loader, optimizer)
    elapsed = time.time() - start
    label = "compiled" if use_compile else "eager"
    print(f"{label}: {elapsed:.2f}s per epoch")
```

</details>

### 연습 2: AMP + Gradient Accumulation

PyTorch 2.x 최신 API(`torch.amp`)를 사용하여 AMP와 Gradient Accumulation(4단계)을 결합하는 학습 루프를 구현하세요.

<details>
<summary>정답 보기</summary>

```python
import torch
import torch.nn.functional as F

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler('cuda')
accumulation_steps = 4

model.train()
optimizer.zero_grad()

for i, (data, target) in enumerate(train_loader):
    data, target = data.cuda(), target.cuda()

    with torch.amp.autocast('cuda'):
        output = model(data)
        loss = F.cross_entropy(output, target) / accumulation_steps

    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

</details>

### 연습 3: DDP vs. 단일 GPU 비교

`nn.DataParallel`과 `DistributedDataParallel`의 주요 차이점을 설명하세요. 왜 DDP가 선호되나요?

<details>
<summary>정답 보기</summary>

| 측면 | DataParallel (DP) | DistributedDataParallel (DDP) |
|------|-------------------|-------------------------------|
| 프로세스 | 단일 프로세스, 멀티 스레드 | GPU당 하나의 프로세스 |
| GIL | Python GIL 영향 | GIL 병목 없음 |
| 그래디언트 동기화 | GPU 0에 수집 후 브로드캐스트 | All-reduce (균형) |
| 메모리 | GPU 0이 더 많은 메모리 사용 | GPU 간 균등한 메모리 |
| 확장성 | 2-4 GPU 이상 성능 저하 | 수백 GPU까지 확장 |
| 멀티 노드 | 미지원 | 지원 |

DDP가 선호되는 이유:
1. 각 GPU가 별도의 프로세스에서 실행되어 Python GIL 병목 회피
2. All-reduce로 그래디언트를 동기화하여 통신이 균등하게 분산
3. GPU 간 메모리가 균형 ("GPU 0 병목" 없음)
4. GPU 수에 비례하여 선형적으로 확장
5. 여러 머신에 걸친 멀티 노드 학습 지원

</details>

---

## 다음 단계

[모델 저장 및 배포](./41_Model_Saving_Deployment.md)에서 모델 저장과 배포를 학습합니다.
