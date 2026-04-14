# Debugging PyTorch

**Previous**: [GPU Training](./10_GPU_Training.md) | **Next**: [Custom Layers and Functions](./12_Custom_Layers_and_Functions.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Diagnose and fix shape mismatch errors systematically
2. Inspect gradients to identify vanishing, exploding, or None gradient problems
3. Use forward and backward hooks for intermediate value inspection
4. Apply `torch.autograd.detect_anomaly()` to find NaN-producing operations
5. Debug device mismatch and dtype errors
6. Use breakpoints and print-based debugging effectively in PyTorch
7. Profile model performance to find bottlenecks

---

Debugging is where you spend most of your time when building neural networks. This lesson covers the most common PyTorch errors and systematic approaches to diagnose them.

---

## 1. Shape Mismatch Errors

### 1.1 Reading Shape Error Messages

```python
import torch
import torch.nn as nn

# Common error: matrix multiplication with incompatible shapes
a = torch.randn(3, 4)
b = torch.randn(5, 2)
# a @ b  -> RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x4 and 5x2)

# The fix: check shapes before operations
print(f"a: {a.shape}, b: {b.shape}")
# a is [3, 4], b is [5, 2] -- inner dimensions (4 != 5) don't match
```

### 1.2 Systematic Shape Debugging

```python
class DebugModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        print(f"Input:       {x.shape}")       # [B, 3, 28, 28]
        x = self.pool(torch.relu(self.conv1(x)))
        print(f"After conv1: {x.shape}")       # [B, 16, 14, 14]
        x = self.pool(torch.relu(self.conv2(x)))
        print(f"After conv2: {x.shape}")       # [B, 32, 7, 7]
        x = x.flatten(1)
        print(f"After flat:  {x.shape}")       # [B, 32*7*7] = [B, 1568]
        x = self.fc(x)
        print(f"Output:      {x.shape}")       # [B, 10]
        return x

# Test with a single batch
model = DebugModel()
x = torch.randn(1, 3, 28, 28)
output = model(x)
```

### 1.3 Common Shape Fixes

```python
# 1. Missing batch dimension
x = torch.randn(784)           # [784]
x = x.unsqueeze(0)             # [1, 784]

# 2. Wrong flatten
x = torch.randn(8, 32, 7, 7)
x = x.view(8, -1)              # [8, 1568]  NOT x.view(-1) which gives [12544]

# 3. CrossEntropyLoss shape mismatch
output = torch.randn(32, 10)   # logits: [batch, classes]
target = torch.randint(0, 10, (32,))  # labels: [batch] (NOT [batch, classes])
loss = nn.CrossEntropyLoss()(output, target)

# 4. BCEWithLogitsLoss shape mismatch
output = torch.randn(32, 1)     # [batch, 1]
target = torch.randn(32, 1)     # must match! [batch, 1]
# or
output = torch.randn(32)        # [batch]
target = torch.randn(32)        # [batch]
```

---

## 2. Gradient Debugging

### 2.1 Checking for None Gradients

```python
model = nn.Linear(10, 5)
x = torch.randn(3, 10)
y = model(x).sum()
y.backward()

for name, param in model.named_parameters():
    if param.grad is None:
        print(f"WARNING: {name} has None gradient!")
    else:
        print(f"{name}: grad norm = {param.grad.norm():.6f}")
```

### 2.2 Common Causes of None Gradients

```python
# 1. Tensor created without requires_grad
x = torch.tensor(2.0)  # requires_grad=False
y = x ** 2
y.backward()
print(x.grad)  # None

# 2. Detached intermediate
features = encoder(x).detach()  # breaks gradient flow
output = decoder(features)
output.sum().backward()
# encoder parameters have None gradients

# 3. In-place operations that modify the graph
x = torch.randn(3, requires_grad=True)
y = x * 2
y_copy = y  # alias
y_copy.data.fill_(0)  # modifies y in-place, corrupts graph
```

### 2.3 Gradient Monitoring

```python
def check_gradients(model, threshold=1e-7):
    """Check for vanishing or exploding gradients."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_max = param.grad.abs().max().item()

            if grad_norm < threshold:
                print(f"VANISHING: {name} grad_norm={grad_norm:.2e}")
            elif grad_norm > 1000:
                print(f"EXPLODING: {name} grad_norm={grad_norm:.2e}")
            elif torch.isnan(param.grad).any():
                print(f"NaN GRAD:  {name}")
            else:
                print(f"OK:        {name} grad_norm={grad_norm:.4f}")

# Call after backward
loss.backward()
check_gradients(model)
```

---

## 3. Hooks

### 3.1 Forward Hooks

```python
# Inspect intermediate activations without modifying the model

activations = {}

def save_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# Register hooks
model[0].register_forward_hook(save_activation('linear1'))
model[1].register_forward_hook(save_activation('relu'))
model[2].register_forward_hook(save_activation('linear2'))

# Forward pass
output = model(torch.randn(1, 784))

# Inspect activations
for name, act in activations.items():
    print(f"{name}: shape={act.shape}, "
          f"mean={act.mean():.4f}, std={act.std():.4f}")
```

### 3.2 Backward Hooks

```python
# Inspect gradients flowing through the network

gradient_info = {}

def save_gradient(name):
    def hook(module, grad_input, grad_output):
        gradient_info[name] = {
            'grad_output': [g.norm().item() if g is not None else None
                           for g in grad_output],
        }
    return hook

model[0].register_full_backward_hook(save_gradient('linear1'))
model[2].register_full_backward_hook(save_gradient('linear2'))

output = model(torch.randn(1, 784))
output.sum().backward()

for name, info in gradient_info.items():
    print(f"{name}: grad_output norms = {info['grad_output']}")
```

### 3.3 Removing Hooks

```python
# Hooks return a handle for removal
handle = model[0].register_forward_hook(save_activation('linear1'))
# ... use the hook ...
handle.remove()  # clean up
```

---

## 4. Anomaly Detection

### 4.1 torch.autograd.detect_anomaly

```python
# Detects the operation that produced NaN or Inf during backward pass
with torch.autograd.detect_anomaly():
    x = torch.randn(3, requires_grad=True)
    y = x / 0  # will cause an issue
    y.sum().backward()

# Output shows the EXACT forward operation that caused the NaN
# WARNING: significantly slower, only use for debugging
```

### 4.2 Setting Anomaly Detection Globally

```python
# Enable for the entire script (for debugging only)
torch.autograd.set_detect_anomaly(True)

# Disable when done
torch.autograd.set_detect_anomaly(False)
```

---

## 5. Device and dtype Errors

### 5.1 Device Mismatch

```python
# Error: expected all tensors to be on the same device
cpu_t = torch.tensor([1.0])
# gpu_t = torch.tensor([2.0], device='cuda')
# cpu_t + gpu_t  # ERROR

# Systematic fix: always use a device variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Check model device
def get_model_device(model):
    return next(model.parameters()).device

# Check all tensors in a dict
def check_devices(tensors_dict):
    for name, t in tensors_dict.items():
        if isinstance(t, torch.Tensor):
            print(f"{name}: device={t.device}")
```

### 5.2 dtype Mismatch

```python
# Common: mixing float and double
a = torch.tensor([1.0])           # float32
b = torch.tensor([2.0]).double()  # float64
# nn.Linear expects float32 by default

# Fix: ensure consistent dtypes
model = model.float()  # force float32
x = x.float()          # convert input to float32

# Common issue: integer targets for MSELoss
target = torch.tensor([1, 2, 3])        # int64
# nn.MSELoss()(pred, target)            # ERROR
target = torch.tensor([1, 2, 3]).float() # float32 -- OK
```

---

## 6. Debugging Strategies

### 6.1 The Debugging Checklist

```
1. Shape: Print .shape after every operation
2. Device: Print .device for all tensors
3. dtype: Print .dtype for all tensors
4. Values: Print .min(), .max(), .mean() to check for NaN/Inf
5. Gradients: Check .grad is not None after backward
6. Mode: Verify model.training is correct (train vs eval)
```

### 6.2 Using breakpoint()

```python
class MyModel(nn.Module):
    def forward(self, x):
        x = self.layer1(x)
        if torch.isnan(x).any():
            breakpoint()  # drops into Python debugger
        x = self.layer2(x)
        return x

# In the debugger:
# (Pdb) x.shape
# (Pdb) x.min(), x.max()
# (Pdb) self.layer1.weight.grad
```

### 6.3 Minimal Reproducible Example

```python
# When you're stuck, reduce to the smallest possible example

# Instead of debugging a 50-layer model on full dataset:
model = nn.Linear(3, 2)  # simplest model
x = torch.randn(1, 3)    # single sample
y = torch.tensor([0])     # single label

output = model(x)
loss = nn.CrossEntropyLoss()(output, y)
loss.backward()

# Does it work? If yes, gradually add complexity.
# If no, you've found the bug.
```

---

## 7. Performance Profiling

### 7.1 torch.profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(5):
        output = model(torch.randn(32, 784).to(device))
        output.sum().backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 7.2 Simple Timing

```python
import time

class Timer:
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.time()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.time() - self.start
        print(f"{self.name}: {self.elapsed*1000:.2f} ms")

# Usage
with Timer("Forward"):
    output = model(x)
with Timer("Backward"):
    output.sum().backward()
```

---

## Summary

| Issue | Diagnostic Tool |
|-------|----------------|
| Shape mismatch | Print `.shape` after each operation |
| None gradients | Check `requires_grad`, `detach()` usage |
| Vanishing/exploding gradients | Monitor gradient norms per layer |
| NaN in backward | `torch.autograd.detect_anomaly()` |
| Device mismatch | Print `.device` for all tensors |
| dtype mismatch | Print `.dtype`; use `.float()` consistently |
| Intermediate values | Forward hooks |
| Gradient flow | Backward hooks |
| Performance | `torch.profiler` or custom Timer |

---

**Next**: [Custom Layers and Functions](./12_Custom_Layers_and_Functions.md) -- Implementing custom autograd functions and layers.
