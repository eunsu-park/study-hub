# 데이터셋과 데이터로더 (Dataset and DataLoader)

**이전**: [손실 함수와 옵티마이저](./06_Loss_Functions_and_Optimizers.md) | **다음**: [학습 루프](./08_Training_Loop.md)

---

## 학습 목표 (Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `torch.utils.data.Dataset`을 상속하여 커스텀 데이터셋을 구현할 수 있습니다
2. 배치 크기, 셔플링, 워커 프로세스로 `DataLoader`를 구성할 수 있습니다
3. `torchvision.transforms`와 v2 API를 사용하여 데이터 변환을 적용할 수 있습니다
4. 이미지, 텍스트, 표형 데이터를 위한 데이터 파이프라인을 구축할 수 있습니다
5. `torchvision.datasets`의 내장 데이터셋을 사용할 수 있습니다
6. 커스텀 `collate_fn`으로 가변 길이 데이터를 처리할 수 있습니다
7. `Subset`, `ConcatDataset`, `random_split`으로 데이터셋을 조작할 수 있습니다
8. `num_workers`와 `pin_memory`로 데이터 로딩 성능을 최적화할 수 있습니다

---

효율적인 데이터 로딩은 학습 성능에 매우 중요합니다. PyTorch의 `Dataset`과 `DataLoader`는 데이터 접근을 학습 로직에서 분리하는 깔끔한 추상화를 제공합니다.

---

## 1. Dataset 클래스

### 1.1 맵 스타일 데이터셋

```python
import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = SimpleDataset([[1, 2], [3, 4], [5, 6]], [0, 1, 0])
print(len(dataset))       # 3
print(dataset[0])         # (tensor([1., 2.]), tensor(0))
```

### 1.2 이미지 데이터셋

```python
import os
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            class_idx = len(self.samples)
            for img_name in os.listdir(class_dir):
                self.samples.append((os.path.join(class_dir, img_name), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
```

---

## 2. DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,          # 배치당 샘플 수
    shuffle=True,           # 학습에는 True, 검증에는 False
    num_workers=4,          # 병렬 데이터 로딩 프로세스
    pin_memory=True,        # 더 빠른 CPU->GPU 전송
    drop_last=True,         # 불완전한 마지막 배치 버림
)

for batch_X, batch_y in loader:
    # ... 학습 스텝 ...
    pass
```

---

## 3. 변환 (Transforms)

```python
from torchvision import transforms

# 학습용 변환 (데이터 증강 포함)
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 검증용 변환 (증강 없음)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

---

## 4. 내장 데이터셋

```python
from torchvision import datasets

# MNIST
train_data = datasets.MNIST('./data', train=True, download=True,
                              transform=transforms.ToTensor())

# CIFAR-10
train_data = datasets.CIFAR10('./data', train=True, download=True,
                                transform=train_transform)

# ImageFolder (커스텀 이미지 디렉토리)
train_data = datasets.ImageFolder('./data/train', transform=train_transform)
print(train_data.classes)       # ['cat', 'dog']
```

---

## 5. 커스텀 collate_fn

샘플의 크기가 다를 때 커스텀 collate 함수가 필요합니다:

```python
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """가변 길이 시퀀스를 배치 내 최대 길이로 패딩합니다."""
    sequences, labels = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return padded, labels
```

---

## 6. 데이터셋 조작

### 6.1 학습/검증 분할

```python
from torch.utils.data import random_split

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_set, val_set = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
```

### 6.2 Subset

```python
from torch.utils.data import Subset

# 처음 1000개 샘플만 사용 (디버깅용)
subset = Subset(full_dataset, indices=range(1000))
```

---

## 7. 성능 최적화

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,              # CPU 코어 수 정도로 설정
    pin_memory=True,            # CUDA 사용 시 활성화
    persistent_workers=True,    # 에포크 간 워커 유지
    prefetch_factor=2,          # 워커당 2 배치 프리페치
)

# pinned memory와 함께 사용:
for x, y in loader:
    x = x.to(device, non_blocking=True)  # 비동기 전송
    y = y.to(device, non_blocking=True)
```

---

## 요약

| 개념 | 핵심 내용 |
|------|----------|
| Dataset | `__len__`과 `__getitem__` 구현; 한 번에 하나의 샘플 |
| DataLoader | 배치화, 셔플링, 병렬 로딩, 장치 전송 |
| Transforms | `Compose`로 체인; 학습(증강)과 평가에 다르게 |
| 내장 데이터셋 | MNIST, CIFAR, ImageFolder로 빠른 실험 |
| collate_fn | 패딩이나 잘라내기로 가변 길이 데이터 처리 |
| random_split | 재현 가능한 generator로 학습/검증 분할 |
| num_workers | 병렬 데이터 로딩; 0부터 시작, 필요에 따라 증가 |
| pin_memory | 더 빠른 GPU 전송; `non_blocking=True`와 함께 사용 |

---

**다음**: [학습 루프](./08_Training_Loop.md) -- 검증과 로깅이 포함된 완전한 학습 루프 작성.
