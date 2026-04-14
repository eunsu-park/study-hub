# GPU 학습 (GPU Training)

**이전**: [모델 저장과 로드](./09_Model_Saving_and_Loading.md) | **다음**: [PyTorch 디버깅](./11_Debugging_PyTorch.md)

---

## 학습 목표 (Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `.to(device)`로 모델과 데이터를 GPU로 이동하고 장치 불가지론 코드를 작성할 수 있습니다
2. GPU 메모리 사용량을 모니터링하고 OOM(메모리 부족) 에러를 진단할 수 있습니다
3. `DataParallel`과 `DistributedDataParallel`로 멀티 GPU 학습을 수행할 수 있습니다
4. 자동 혼합 정밀도(AMP)를 적용하여 학습 속도를 높이고 메모리를 줄일 수 있습니다
5. `torch.cuda` 유틸리티를 프로파일링과 동기화에 사용할 수 있습니다
6. CUDA 스트림과 비동기 실행을 이해할 수 있습니다

---

GPU 가속은 현대 딥러닝을 가능하게 하는 핵심입니다. 단일 GPU로 CPU 대비 10-100배 속도 향상을 제공할 수 있습니다.

---

## 1. 장치 관리

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 장치: {device}")

model = model.to(device)

for batch_X, batch_y in dataloader:
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)
    output = model(batch_X)
```

---

## 2. GPU 메모리 관리

```python
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"할당: {allocated:.1f} MB, 예약: {reserved:.1f} MB")

# 메모리 절약 방법:
# 1. 추론 시 torch.no_grad() 사용
# 2. loss.item()으로 스칼라 추출 (그래프 유지 방지)
# 3. 그래디언트 체크포인팅
# 4. 배치 크기 축소
# 5. 혼합 정밀도 사용
```

### OOM 디버깅

```python
# 나쁜 예: 계산 그래프 누적
total_loss = 0
for batch in loader:
    loss = loss_fn(model(batch), target)
    total_loss += loss  # 전체 그래프를 메모리에 유지!

# 좋은 예: 스칼라 값 추출
total_loss = 0
for batch in loader:
    loss = loss_fn(model(batch), target)
    total_loss += loss.item()  # 스칼라, 그래프 없음
```

---

## 3. 멀티 GPU 학습

### 3.1 DataParallel (간단하지만 제한적)

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)

# 저장 시 원본 모델 접근
original = model.module if isinstance(model, nn.DataParallel) else model
torch.save(original.state_dict(), 'model.pt')
```

### 3.2 DistributedDataParallel (권장)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def train_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    for epoch in range(10):
        sampler.set_epoch(epoch)  # 셔플링에 중요
        for batch_X, batch_y in loader:
            # ... 학습 스텝 ...
            pass

    dist.destroy_process_group()

# torchrun으로 실행:
# torchrun --nproc_per_node=4 train.py
```

---

## 4. 자동 혼합 정밀도 (AMP)

`float16`으로 대부분의 연산을 수행하고, 수치적으로 민감한 연산에는 `float32`를 사용하여 ~2배 속도 향상과 ~50% 메모리 절감을 달성합니다.

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

for batch_X, batch_y in train_loader:
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)

    optimizer.zero_grad()

    # 혼합 정밀도로 순전파
    with autocast(device_type='cuda'):
        output = model(batch_X)
        loss = loss_fn(output, batch_y)

    # 그래디언트 스케일링으로 역전파
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 5. GPU 타이밍

```python
# 잘못된 방법: CUDA 연산은 비동기!
import time
start = time.time()
output = model(x.cuda())
elapsed = time.time() - start  # 실행 시간이 아닌 실행 시작 시간

# 올바른 방법: 동기화 후 측정
torch.cuda.synchronize()
start = time.time()
output = model(x.cuda())
torch.cuda.synchronize()
elapsed = time.time() - start
```

---

## 6. 재현성

```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 요약

| 개념 | 핵심 내용 |
|------|----------|
| 장치 불가지론 코드 | `device = torch.device('cuda' if ... else 'cpu')` |
| 메모리 모니터링 | `torch.cuda.memory_allocated()`, `memory_summary()` |
| OOM 해결 | 배치 크기 축소, `loss.item()` 사용, 그래디언트 체크포인팅 |
| DataParallel | 간단한 멀티 GPU; 배치를 GPU 간 분할 |
| DDP | 권장 멀티 GPU; GPU당 하나의 프로세스 |
| AMP | autocast + GradScaler로 ~2배 빠른 학습 |
| 타이밍 | 정확한 GPU 타이밍에는 `torch.cuda.synchronize()` 필요 |

---

**다음**: [PyTorch 디버깅](./11_Debugging_PyTorch.md) -- 일반적인 PyTorch 에러 찾기와 수정.
