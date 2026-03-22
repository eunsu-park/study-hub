[Previous: Vision Transformer (ViT)](./22_Impl_ViT.md) | [Next: Loss Functions](./24_Loss_Functions.md)

---

# 23. Training Optimization

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply hyperparameter tuning strategies (grid, random, Optuna)
2. Implement advanced learning rate scheduling with warmup and cosine decay
3. Use Mixed Precision Training with PyTorch AMP
4. Implement gradient accumulation for large effective batch sizes
5. Accelerate training with `torch.compile()` (PyTorch 2.0+)
6. Scale training across multiple GPUs with DDP and FSDP

---

## 1. Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Impact | Typical Range |
|-----------|--------|--------------|
| Learning Rate | Convergence speed/stability | 1e-5 ~ 1e-2 |
| Batch Size | Memory/generalization | 16 ~ 512 |
| Weight Decay | Overfitting prevention | 1e-5 ~ 1e-2 |
| Dropout | Overfitting prevention | 0.1 ~ 0.5 |
| Epochs | Training amount | Data-dependent |

### Search Strategies

```python
# Grid Search
learning_rates = [1e-4, 1e-3, 1e-2]
batch_sizes = [32, 64, 128]

for lr in learning_rates:
    for bs in batch_sizes:
        train_and_evaluate(lr, bs)

# Random Search (more efficient)
import random
for _ in range(20):
    lr = 10 ** random.uniform(-5, -2)  # log scale
    bs = random.choice([16, 32, 64, 128])
    train_and_evaluate(lr, bs)
```

### Using Optuna

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

## 2. Advanced Learning Rate Scheduling

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

# Called every batch
for batch in train_loader:
    train_step(batch)
    scheduler.step()
```

---

## 3. Mixed Precision Training

### Concept

```
FP32 (32-bit) → FP16 (16-bit)
- Memory savings (approximately 50%)
- Speed improvement (approximately 2-3x)
- Accuracy preservation
```

### PyTorch AMP

```python
# PyTorch 2.x modern API (recommended)
scaler = torch.amp.GradScaler('cuda')

for data, target in train_loader:
    optimizer.zero_grad()

    # Automatic Mixed Precision
    with torch.amp.autocast('cuda'):
        output = model(data)
        loss = criterion(output, target)

    # Scaled backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

> **Note**: `torch.cuda.amp.autocast` and `torch.cuda.amp.GradScaler` are deprecated since PyTorch 2.x. Use `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')` instead.

### Complete Training Loop

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

### Concept

```
Multiple small batches → Large batch effect
Useful when GPU memory is limited
```

### Implementation

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target)
    loss = loss / accumulation_steps  # scaling
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Combined with AMP

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

### Preventing Gradient Explosion

```python
# Norm clipping (recommended)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Value clipping
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### In Training Loop

```python
for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    # Update after clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

---

## 6. Advanced Early Stopping

### Patience and Delta

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

## 7. Training Monitoring

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

## 8. Reproducibility

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

### Concept

`torch.compile()` is the flagship feature of PyTorch 2.0. It automatically optimizes your model by tracing the computation graph and generating fused, hardware-specific kernels — without changing your model code.

```
model → torch.compile(model) → Optimized model
                                 - Operator fusion
                                 - Memory planning
                                 - Hardware-specific kernels (Triton on GPU)
```

### Basic Usage

```python
import torch

model = MyModel().cuda()

# One-line compilation — no code changes needed
compiled_model = torch.compile(model)

# Use exactly like the original model
output = compiled_model(input_data)
```

The first forward pass triggers compilation (takes a few seconds). Subsequent calls use the optimized kernels.

### Compilation Modes

```python
# Default: balanced optimization (good starting point)
compiled = torch.compile(model)

# max-autotune: slower compilation, fastest inference
# Best for: production inference, repeated training runs
compiled = torch.compile(model, mode="max-autotune")

# reduce-overhead: minimizes CPU overhead
# Best for: small models where kernel launch overhead dominates
compiled = torch.compile(model, mode="reduce-overhead")
```

| Mode | Compile Time | Runtime Speed | Best For |
|------|-------------|---------------|----------|
| `default` | Moderate | Good | General use |
| `max-autotune` | Slow | Best | Production, large models |
| `reduce-overhead` | Moderate | Good for small models | Small batch, low latency |

### Training with torch.compile()

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
            output = compiled_model(data)  # Optimized forward pass
            loss = F.cross_entropy(output, target)

        scaler.scale(loss).backward()  # Optimized backward pass
        scaler.step(optimizer)
        scaler.update()
```

### Dynamic Shapes

By default, `torch.compile()` recompiles when input shapes change. For variable-length inputs (e.g., NLP), use `dynamic=True`:

```python
# Avoid recompilation on shape changes
compiled_model = torch.compile(model, dynamic=True)
```

### Limitations and Tips

- **First call is slow**: Compilation happens on the first forward pass. Budget extra time for warmup.
- **Not all ops are supported**: Most standard PyTorch ops work. Custom CUDA extensions or obscure ops may fall back to eager mode.
- **Debugging**: Use `torch._dynamo.config.verbose = True` to see what gets compiled.
- **Graph breaks**: Data-dependent control flow (e.g., `if x.sum() > 0`) causes graph breaks, reducing optimization opportunities.

```python
# Check what gets compiled
torch._dynamo.config.verbose = True
compiled_model = torch.compile(model)
output = compiled_model(sample_input)  # Prints compilation info
```

---

## 10. Distributed Training (DDP & FSDP)

### Data Parallel vs. Distributed Data Parallel

```
DataParallel (DP) — simple but slow (GIL bottleneck, single process)
DistributedDataParallel (DDP) — one process per GPU, efficient gradient sync
```

> Always prefer DDP over DP. `nn.DataParallel` is essentially deprecated for multi-GPU training.

### DDP Setup

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

    # Each GPU gets a different subset of data
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch
        for data, target in loader:
            data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()  # Gradients automatically synchronized
            optimizer.step()

    cleanup()
```

### Launching DDP

```bash
# Launch with torchrun (recommended, replaces torch.distributed.launch)
torchrun --nproc_per_node=4 train.py
```

```python
# In train.py
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
```

### FSDP (Fully Sharded Data Parallel)

FSDP shards model parameters, gradients, and optimizer states across GPUs — enabling training of models that don't fit on a single GPU.

```
DDP:  Each GPU holds full model copy + syncs gradients
FSDP: Each GPU holds a shard of parameters + gathers on demand
      → Much lower memory per GPU
```

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

def train_fsdp(rank, world_size):
    setup(rank, world_size)

    model = LargeModel().to(rank)

    # Wrap with FSDP — parameters are sharded across GPUs
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # Maximum memory savings
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

### FSDP Sharding Strategies

| Strategy | Memory Savings | Communication Cost | Use Case |
|----------|---------------|-------------------|----------|
| `FULL_SHARD` | Maximum | Higher | Very large models |
| `SHARD_GRAD_OP` | Moderate | Moderate | Medium-large models |
| `NO_SHARD` | None (= DDP) | Lowest | Baseline comparison |

### FSDP + torch.compile()

Combining FSDP with `torch.compile()` gives both memory efficiency and computational speedup:

```python
model = LargeModel().to(rank)

# Compile first, then wrap with FSDP
compiled_model = torch.compile(model)
fsdp_model = FSDP(compiled_model, device_id=rank)
```

### When to Use What

| Scenario | Recommendation |
|----------|---------------|
| Single GPU | `torch.compile(model)` |
| Multi-GPU, model fits per GPU | DDP + `torch.compile()` |
| Multi-GPU, model too large | FSDP |
| Multi-GPU, large model + speed | FSDP + `torch.compile()` |

---

## Summary

### Checklist

- [ ] Set learning rate appropriately (recommend starting with 1e-4)
- [ ] Use warmup (essential for Transformers)
- [ ] Apply Mixed Precision (GPU efficiency)
- [ ] Gradient Clipping (RNN/Transformer)
- [ ] Configure early stopping
- [ ] Set reproducibility seed
- [ ] Set up logging/monitoring
- [ ] Use `torch.compile()` for speedup (PyTorch 2.0+)
- [ ] Consider DDP/FSDP for multi-GPU training

### Recommended Configuration

```python
# PyTorch 2.x optimization configuration
model = MyModel().cuda()
compiled_model = torch.compile(model)  # Automatic optimization
optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = OneCycleLR(optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(loader))
scaler = torch.amp.GradScaler('cuda')  # AMP
early_stopping = EarlyStopping(patience=10)
```

---

## Exercises

### Exercise 1: torch.compile() Speedup Measurement

Write a script that trains a simple CNN on CIFAR-10 and compares training time per epoch with and without `torch.compile()`.

<details>
<summary>Show Answer</summary>

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

    # Warmup
    train_one_epoch(model, loader, optimizer)

    # Measure
    start = time.time()
    train_one_epoch(model, loader, optimizer)
    elapsed = time.time() - start
    label = "compiled" if use_compile else "eager"
    print(f"{label}: {elapsed:.2f}s per epoch")
```

</details>

### Exercise 2: AMP + Gradient Accumulation

Implement a training loop that combines AMP with gradient accumulation (4 steps) using the modern PyTorch 2.x API (`torch.amp`).

<details>
<summary>Show Answer</summary>

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

### Exercise 3: DDP vs. Single GPU Comparison

Explain the key differences between `nn.DataParallel` and `DistributedDataParallel`. Why is DDP preferred?

<details>
<summary>Show Answer</summary>

| Aspect | DataParallel (DP) | DistributedDataParallel (DDP) |
|--------|-------------------|-------------------------------|
| Processes | Single process, multi-thread | One process per GPU |
| GIL | Affected by Python GIL | No GIL bottleneck |
| Gradient sync | Gather to GPU 0, then broadcast | All-reduce (balanced) |
| Memory | GPU 0 uses more memory | Equal memory across GPUs |
| Scalability | Poor beyond 2-4 GPUs | Scales to hundreds of GPUs |
| Multi-node | Not supported | Supported |

DDP is preferred because:
1. Each GPU runs in a separate process, avoiding the Python GIL bottleneck
2. Gradients are synchronized using all-reduce, distributing communication evenly
3. Memory is balanced across GPUs (no "GPU 0 bottleneck")
4. Scales linearly with the number of GPUs
5. Supports multi-node training across machines

</details>

---

## Next Steps

Learn about model saving and deployment in [41_Model_Saving_Deployment.md](./41_Model_Saving_Deployment.md).
