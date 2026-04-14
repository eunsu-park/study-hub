# Dataset and DataLoader

**Previous**: [Loss Functions and Optimizers](./06_Loss_Functions_and_Optimizers.md) | **Next**: [Training Loop](./08_Training_Loop.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement custom datasets by subclassing `torch.utils.data.Dataset`
2. Configure `DataLoader` with batch size, shuffling, and worker processes
3. Apply data transformations using `torchvision.transforms` and the v2 API
4. Build data pipelines for image, text, and tabular data
5. Use built-in datasets from `torchvision.datasets`
6. Handle variable-length data with custom `collate_fn`
7. Use `Subset`, `ConcatDataset`, and `random_split` for dataset manipulation
8. Optimize data loading performance with `num_workers` and `pin_memory`

---

Efficient data loading is critical for training performance. PyTorch's `Dataset` and `DataLoader` provide a clean abstraction that separates data access from training logic.

---

## 1. The Dataset Class

### 1.1 Map-Style Dataset

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

# Usage
X = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [0, 1, 0, 1]
dataset = SimpleDataset(X, y)

print(len(dataset))       # 4
print(dataset[0])         # (tensor([1., 2.]), tensor(0))
```

### 1.2 Image Dataset from Directory

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
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
```

### 1.3 CSV/Tabular Dataset

```python
import pandas as pd

class CSVDataset(Dataset):
    def __init__(self, csv_path, target_col):
        df = pd.read_csv(csv_path)
        self.features = torch.tensor(
            df.drop(columns=[target_col]).values, dtype=torch.float32
        )
        self.targets = torch.tensor(df[target_col].values, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
```

---

## 2. DataLoader

### 2.1 Basic Usage

```python
from torch.utils.data import DataLoader

dataset = SimpleDataset([[1, 2], [3, 4], [5, 6], [7, 8]], [0, 1, 0, 1])

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,     # randomize order each epoch
)

for batch_X, batch_y in loader:
    print(f"X: {batch_X.shape}, y: {batch_y.shape}")
# X: torch.Size([2, 2]), y: torch.Size([2])
# X: torch.Size([2, 2]), y: torch.Size([2])
```

### 2.2 Key Parameters

```python
loader = DataLoader(
    dataset,
    batch_size=32,          # samples per batch
    shuffle=True,           # True for training, False for validation
    num_workers=4,          # parallel data loading processes
    pin_memory=True,        # faster CPU->GPU transfer
    drop_last=True,         # drop incomplete final batch
    persistent_workers=True, # keep workers alive between epochs
)
```

### 2.3 Iterating

```python
# Standard iteration
for batch in loader:
    features, labels = batch
    # ... training step ...

# With enumerate (for step counting)
for step, (features, labels) in enumerate(loader):
    print(f"Step {step}: batch size = {features.shape[0]}")

# Single batch (for testing)
batch = next(iter(loader))
features, labels = batch
```

---

## 3. Transforms

### 3.1 torchvision.transforms (v1)

```python
from torchvision import transforms

# Compose multiple transforms
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),             # PIL/numpy -> tensor, scales to [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],    # ImageNet statistics
        std=[0.229, 0.224, 0.225]
    ),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

### 3.2 torchvision.transforms.v2 (Recommended)

```python
from torchvision.transforms import v2

train_transform = v2.Compose([
    v2.Resize(256),
    v2.RandomCrop(224),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),  # replaces ToTensor
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])
```

### 3.3 Custom Transforms

```python
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

transform = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(std=0.05),
])
```

---

## 4. Built-in Datasets

### 4.1 torchvision Datasets

```python
from torchvision import datasets

# MNIST
train_data = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# CIFAR-10
train_data = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_transform
)

# ImageFolder (for custom image directories)
# Expects: root/class1/img1.jpg, root/class2/img2.jpg, ...
train_data = datasets.ImageFolder(
    root='./data/train',
    transform=train_transform
)
print(train_data.classes)      # ['cat', 'dog']
print(train_data.class_to_idx) # {'cat': 0, 'dog': 1}
```

### 4.2 Quick Setup for Experiments

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Complete MNIST pipeline in 10 lines
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = datasets.MNIST('./data', train=True, download=True,
                            transform=transform)
test_set = datasets.MNIST('./data', train=False, download=True,
                           transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

# Check shapes
images, labels = next(iter(train_loader))
print(images.shape)  # [64, 1, 28, 28]
print(labels.shape)  # [64]
```

---

## 5. Custom collate_fn

When samples have different sizes, you need a custom collate function:

### 5.1 Variable-Length Sequences

```python
def collate_fn(batch):
    """Pad variable-length sequences to the longest in the batch."""
    sequences, labels = zip(*batch)
    lengths = [len(s) for s in sequences]
    max_len = max(lengths)

    # Pad sequences
    padded = torch.zeros(len(sequences), max_len)
    for i, (seq, length) in enumerate(zip(sequences, lengths)):
        padded[i, :length] = seq

    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    return padded, labels, lengths

class TextDataset(Dataset):
    def __init__(self):
        self.data = [
            (torch.tensor([1, 2, 3]), 0),
            (torch.tensor([4, 5, 6, 7, 8]), 1),
            (torch.tensor([9, 10]), 0),
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

loader = DataLoader(TextDataset(), batch_size=2, collate_fn=collate_fn)
```

### 5.2 Using pad_sequence

```python
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    sequences, labels = zip(*batch)
    # pad_sequence pads to longest, batch_first=True gives [B, T]
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return padded, labels
```

---

## 6. Dataset Manipulation

### 6.1 Train/Validation Split

```python
from torch.utils.data import random_split

full_dataset = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.ToTensor())

# Split 80/20
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_set, val_set = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # reproducible
)

print(len(train_set), len(val_set))  # 48000, 12000
```

### 6.2 Subset

```python
from torch.utils.data import Subset

# Use only the first 1000 samples (for debugging)
subset = Subset(full_dataset, indices=range(1000))

# Use specific indices
indices = [i for i, (_, label) in enumerate(full_dataset) if label in [0, 1]]
binary_dataset = Subset(full_dataset, indices)
```

### 6.3 ConcatDataset

```python
from torch.utils.data import ConcatDataset

dataset_a = SimpleDataset([[1, 2]], [0])
dataset_b = SimpleDataset([[3, 4], [5, 6]], [1, 0])

combined = ConcatDataset([dataset_a, dataset_b])
print(len(combined))  # 3
```

---

## 7. Performance Optimization

### 7.1 num_workers

```python
# Rule of thumb: num_workers = number of CPU cores
# Start with 0 (main process), increase if data loading is the bottleneck
loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

### 7.2 pin_memory

```python
# Faster CPU->GPU transfer when using CUDA
loader = DataLoader(dataset, batch_size=32, pin_memory=True)

# In training loop with pinned memory:
for x, y in loader:
    x = x.to(device, non_blocking=True)  # async transfer
    y = y.to(device, non_blocking=True)
```

### 7.3 Prefetching with persistent_workers

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    persistent_workers=True,   # don't restart workers each epoch
    prefetch_factor=2,         # prefetch 2 batches per worker
)
```

### 7.4 Measuring Data Loading Time

```python
import time

start = time.time()
for batch in loader:
    pass  # just iterate, no training
elapsed = time.time() - start
print(f"Data loading: {elapsed:.2f}s for {len(loader)} batches")
```

---

## 8. Sampler

### 8.1 Weighted Random Sampling

For imbalanced datasets:

```python
from torch.utils.data import WeightedRandomSampler

# Suppose classes are imbalanced: 90% class 0, 10% class 1
targets = [0]*900 + [1]*100
class_counts = [900, 100]
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = class_weights[targets]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

loader = DataLoader(dataset, batch_size=32, sampler=sampler)
# Note: don't use shuffle=True with a sampler
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Dataset | Implement `__len__` and `__getitem__`; one sample at a time |
| DataLoader | Batching, shuffling, parallel loading, device transfer |
| Transforms | Chain with `Compose`; different for train (augmentation) and eval |
| Built-in datasets | MNIST, CIFAR, ImageFolder for quick experiments |
| collate_fn | Handle variable-length data by padding or truncating |
| random_split | Train/val split with reproducible generator |
| num_workers | Parallel data loading; start at 0, increase as needed |
| pin_memory | Faster GPU transfer; use with `non_blocking=True` |

---

**Next**: [Training Loop](./08_Training_Loop.md) -- Writing a complete training loop with validation and logging.
