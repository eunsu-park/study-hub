# GPU Training

**Previous**: [Model Saving and Loading](./09_Model_Saving_and_Loading.md) | **Next**: [Debugging PyTorch](./11_Debugging_PyTorch.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Move models and data to GPU using `.to(device)` and write device-agnostic code
2. Monitor GPU memory usage and diagnose out-of-memory errors
3. Use `DataParallel` and `DistributedDataParallel` for multi-GPU training
4. Apply Automatic Mixed Precision (AMP) to speed up training and reduce memory
5. Use `torch.cuda` utilities for profiling and synchronization
6. Understand CUDA streams and asynchronous execution

---

GPU acceleration is what makes modern deep learning feasible. A single GPU can provide 10-100x speedup over CPU for tensor operations. This lesson covers practical GPU usage in PyTorch.

---

## 1. Device Management

### 1.1 Device-Agnostic Code

```python
import torch

# The standard pattern
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model to device
model = model.to(device)

# Move data to device (in training loop)
for batch_X, batch_y in dataloader:
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)
    output = model(batch_X)
```

### 1.2 Multiple GPUs

```python
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Use a specific GPU
    device = torch.device('cuda:1')  # second GPU
    model.to(device)

    # Move between GPUs
    x_gpu0 = torch.randn(3, device='cuda:0')
    x_gpu1 = x_gpu0.to('cuda:1')
```

### 1.3 Creating Tensors on GPU

```python
# Create directly on GPU (avoids CPU->GPU transfer)
x = torch.randn(1000, 1000, device='cuda')
y = torch.zeros(100, device='cuda')
z = torch.ones_like(x)  # same device as x (cuda)

# Random generator on GPU
g = torch.Generator(device='cuda')
g.manual_seed(42)
x = torch.randn(3, 4, generator=g, device='cuda')
```

---

## 2. GPU Memory Management

### 2.1 Monitoring Memory

```python
if torch.cuda.is_available():
    # Current memory usage
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"Allocated: {allocated:.1f} MB")
    print(f"Reserved:  {reserved:.1f} MB")

    # Peak memory
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak:      {peak:.1f} MB")

    # Reset peak stats
    torch.cuda.reset_peak_memory_stats()

    # Detailed memory summary
    print(torch.cuda.memory_summary())
```

### 2.2 Reducing Memory Usage

```python
# 1. Use torch.no_grad() during inference
with torch.no_grad():
    output = model(x)  # no gradient storage

# 2. Delete intermediate tensors
del intermediate_tensor
torch.cuda.empty_cache()  # returns unused memory to CUDA

# 3. Use gradient checkpointing
from torch.utils.checkpoint import checkpoint

class MemoryEfficientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 10)

    def forward(self, x):
        # Recomputes layer1 and layer2 during backward (saves memory)
        x = checkpoint(self._block, x, use_reentrant=False)
        return self.layer3(x)

    def _block(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return x

# 4. Reduce batch size if OOM
# 5. Use mixed precision (see Section 4)
```

### 2.3 Out-of-Memory (OOM) Debugging

```python
# Common OOM causes and fixes:
# 1. Batch size too large -> reduce batch_size
# 2. Accumulating loss tensors -> use loss.item() instead of loss
# 3. Storing all predictions -> detach and move to CPU

# BAD: accumulates computation graph
total_loss = 0
for batch in loader:
    loss = loss_fn(model(batch), target)
    total_loss += loss  # keeps entire graph in memory!

# GOOD: extract scalar value
total_loss = 0
for batch in loader:
    loss = loss_fn(model(batch), target)
    total_loss += loss.item()  # scalar, no graph
```

---

## 3. Multi-GPU Training

### 3.1 DataParallel (Simple but Limited)

```python
model = MyModel()

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model.to(device)

# Training is the same -- DataParallel handles splitting/gathering
for batch_X, batch_y in loader:
    output = model(batch_X.to(device))
    loss = loss_fn(output, batch_y.to(device))
    loss.backward()
    optimizer.step()

# Access the original model (wrapped inside .module)
original_model = model.module if isinstance(model, nn.DataParallel) else model
torch.save(original_model.state_dict(), 'model.pt')
```

### 3.2 DistributedDataParallel (Recommended)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    setup(rank, world_size)

    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    sampler = DistributedSampler(dataset, num_replicas=world_size,
                                  rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        sampler.set_epoch(epoch)  # important for shuffling
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(rank)
            batch_y = batch_y.to(rank)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()

    cleanup()

# Launch with torchrun:
# torchrun --nproc_per_node=4 train.py
```

---

## 4. Automatic Mixed Precision (AMP)

Mixed precision uses `float16` for most operations and `float32` for numerically sensitive ones, giving ~2x speedup with ~50% memory reduction.

### 4.1 Basic AMP

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

for batch_X, batch_y in train_loader:
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)

    optimizer.zero_grad()

    # Forward pass in mixed precision
    with autocast(device_type='cuda'):
        output = model(batch_X)
        loss = loss_fn(output, batch_y)

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 4.2 AMP with Gradient Clipping

```python
scaler = GradScaler()

for batch_X, batch_y in train_loader:
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)

    optimizer.zero_grad()

    with autocast(device_type='cuda'):
        output = model(batch_X)
        loss = loss_fn(output, batch_y)

    scaler.scale(loss).backward()

    # Unscale gradients before clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
```

### 4.3 When to Use AMP

| Scenario | Use AMP? |
|----------|----------|
| Training large models (Transformers, ResNet-50+) | Yes |
| Inference with large models | Yes |
| Small models (few layers) | Maybe (overhead may negate gains) |
| Models with complex custom ops | Test carefully |
| CPU training | No (AMP is for CUDA) |

---

## 5. CUDA Synchronization and Timing

### 5.1 Correct GPU Timing

```python
# WRONG: CUDA operations are asynchronous!
import time
start = time.time()
output = model(x.cuda())
elapsed = time.time() - start  # measures launch time, not execution time

# CORRECT: synchronize before timing
torch.cuda.synchronize()
start = time.time()
output = model(x.cuda())
torch.cuda.synchronize()  # wait for GPU to finish
elapsed = time.time() - start

# Or use CUDA events (more precise)
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
output = model(x.cuda())
end_event.record()
torch.cuda.synchronize()

print(f"Time: {start_event.elapsed_time(end_event):.2f} ms")
```

### 5.2 CPU vs GPU Speed Comparison

```python
import torch
import time

def benchmark(device, size=4096, n_iters=100):
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)

    # Warmup
    for _ in range(10):
        C = A @ B

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_iters):
        C = A @ B
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start

    return elapsed / n_iters

cpu_time = benchmark(torch.device('cpu'))
print(f"CPU: {cpu_time*1000:.2f} ms")

if torch.cuda.is_available():
    gpu_time = benchmark(torch.device('cuda'))
    print(f"GPU: {gpu_time*1000:.2f} ms")
    print(f"Speedup: {cpu_time/gpu_time:.1f}x")
```

---

## 6. Practical GPU Tips

### 6.1 Environment Variables

```bash
# Select specific GPU(s)
CUDA_VISIBLE_DEVICES=0,1 python train.py

# Deterministic mode (slower but reproducible)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py
```

### 6.2 Reproducibility on GPU

```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 6.3 cudnn.benchmark

```python
# Enable cuDNN auto-tuner for fixed input sizes (faster after warmup)
torch.backends.cudnn.benchmark = True

# Disable for variable input sizes (e.g., NLP with different lengths)
torch.backends.cudnn.benchmark = False
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Device-agnostic code | Use `device = torch.device('cuda' if ... else 'cpu')` |
| Memory monitoring | `torch.cuda.memory_allocated()`, `memory_summary()` |
| OOM fixes | Reduce batch size, use `loss.item()`, gradient checkpointing |
| DataParallel | Simple multi-GPU; splits batch across GPUs |
| DDP | Recommended multi-GPU; one process per GPU |
| AMP | ~2x faster training with autocast + GradScaler |
| Timing | Must `torch.cuda.synchronize()` for accurate GPU timing |
| Reproducibility | Set manual seeds + deterministic mode |

---

**Next**: [Debugging PyTorch](./11_Debugging_PyTorch.md) -- Finding and fixing common PyTorch errors.
