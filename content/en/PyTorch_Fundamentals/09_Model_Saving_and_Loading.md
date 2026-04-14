# Model Saving and Loading

**Previous**: [Training Loop](./08_Training_Loop.md) | **Next**: [GPU Training](./10_GPU_Training.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Save and load model weights using `state_dict`
2. Save complete training checkpoints (model, optimizer, epoch, loss)
3. Resume training from a checkpoint
4. Export models to ONNX format for cross-framework deployment
5. Handle device mapping when loading models on different hardware
6. Understand the differences between `torch.save`, `state_dict`, and `torch.jit.save`
7. Implement best-model saving during training

---

Saving and loading models is essential for deployment, sharing, and resuming interrupted training. PyTorch offers multiple serialization formats for different use cases.

---

## 1. Saving and Loading state_dict (Recommended)

### 1.1 Save Weights Only

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 10)
)

# Save
torch.save(model.state_dict(), 'model_weights.pt')

# Load
model2 = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 10)
)
model2.load_state_dict(torch.load('model_weights.pt'))
model2.eval()  # set to eval mode for inference
```

### 1.2 Why state_dict Over torch.save(model)?

```python
# NOT recommended: saves the entire model (including class definition)
torch.save(model, 'entire_model.pt')
model = torch.load('entire_model.pt')
# Problem: requires the exact same class definition to be importable

# Recommended: save only the state dict
torch.save(model.state_dict(), 'model_weights.pt')
# Can load into any model with matching architecture
```

### 1.3 Inspecting state_dict

```python
sd = model.state_dict()
for key, tensor in sd.items():
    print(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")
# 0.weight: shape=torch.Size([256, 784]), dtype=torch.float32
# 0.bias: shape=torch.Size([256]), dtype=torch.float32
# 2.weight: shape=torch.Size([10, 256]), dtype=torch.float32
# 2.bias: shape=torch.Size([10]), dtype=torch.float32
```

---

## 2. Training Checkpoints

### 2.1 Save a Complete Checkpoint

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'best_val_loss': best_val_loss,
}
torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
```

### 2.2 Resume Training from Checkpoint

```python
# Recreate model and optimizer (same architecture and config)
model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Load checkpoint
checkpoint = torch.load('checkpoint_epoch_25.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
best_val_loss = checkpoint['best_val_loss']

# Resume training
for epoch in range(start_epoch, 100):
    train_loss, _ = train_one_epoch(model, train_loader,
                                     optimizer, loss_fn, device)
    val_loss, _ = validate(model, val_loader, loss_fn, device)
    print(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")
```

### 2.3 Automatic Best-Model Saving

```python
class ModelCheckpoint:
    """Save the best model based on validation metric."""

    def __init__(self, path='best_model.pt', mode='min'):
        self.path = path
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')

    def __call__(self, metric, model):
        improved = (metric < self.best if self.mode == 'min'
                    else metric > self.best)
        if improved:
            self.best = metric
            torch.save(model.state_dict(), self.path)
            return True
        return False

# Usage
checkpoint = ModelCheckpoint('best_model.pt', mode='min')

for epoch in range(100):
    # ... training ...
    val_loss, val_acc = validate(model, val_loader, loss_fn, device)

    if checkpoint(val_loss, model):
        print(f"Epoch {epoch}: New best model (val_loss={val_loss:.4f})")
```

---

## 3. Device Mapping

### 3.1 Loading on a Different Device

```python
# Model was saved on GPU, loading on CPU
model.load_state_dict(
    torch.load('model_weights.pt', map_location='cpu')
)

# Model was saved on CPU, loading on GPU
model.load_state_dict(
    torch.load('model_weights.pt', map_location='cuda:0')
)

# Map from one GPU to another
model.load_state_dict(
    torch.load('model_weights.pt', map_location={'cuda:1': 'cuda:0'})
)

# Device-agnostic loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(
    torch.load('model_weights.pt', map_location=device)
)
model.to(device)
```

### 3.2 weights_only Parameter

```python
# PyTorch 2.x: use weights_only=True for security (prevents pickle attacks)
state_dict = torch.load('model_weights.pt', weights_only=True)
model.load_state_dict(state_dict)

# If loading a full checkpoint (dict with non-tensor values):
checkpoint = torch.load('checkpoint.pt', weights_only=False)
```

---

## 4. Partial Loading (Transfer Learning)

### 4.1 Load Matching Keys Only

```python
# Old model: 10 classes
old_state = torch.load('old_model.pt')

# New model: 20 classes (different final layer)
new_model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 20)  # different output size
)

# Load only matching keys
new_state = new_model.state_dict()
pretrained = {k: v for k, v in old_state.items()
              if k in new_state and v.shape == new_state[k].shape}
new_state.update(pretrained)
new_model.load_state_dict(new_state)

print(f"Loaded {len(pretrained)}/{len(new_state)} parameters")
```

### 4.2 strict=False

```python
# Allow missing/unexpected keys
model.load_state_dict(old_state, strict=False)
# Prints warnings about missing and unexpected keys
```

---

## 5. ONNX Export

### 5.1 Export to ONNX

```python
model.eval()

# Dummy input with correct shape
dummy_input = torch.randn(1, 784)

torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},    # variable batch size
        'output': {0: 'batch_size'},
    },
    opset_version=17,
)

print("Exported to model.onnx")
```

### 5.2 Verify ONNX Model

```python
import onnx

model_onnx = onnx.load('model.onnx')
onnx.checker.check_model(model_onnx)
print("ONNX model is valid")
```

### 5.3 Run ONNX with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('model.onnx')
input_data = np.random.randn(5, 784).astype(np.float32)
outputs = session.run(None, {'input': input_data})
print(f"Output shape: {outputs[0].shape}")  # (5, 10)
```

---

## 6. SafeTensors Format

```python
# safetensors: a safer, faster alternative to pickle-based torch.save
# pip install safetensors

from safetensors.torch import save_file, load_file

# Save
save_file(model.state_dict(), 'model.safetensors')

# Load
state_dict = load_file('model.safetensors')
model.load_state_dict(state_dict)
```

**Advantages of safetensors**:
- No pickle (no arbitrary code execution risk)
- Faster loading (memory-mapped)
- Widely used in HuggingFace ecosystem

---

## 7. File Format Comparison

| Format | Use Case | Security | Speed |
|--------|----------|----------|-------|
| `state_dict` (.pt) | Standard PyTorch | Pickle (risky with untrusted files) | Fast |
| Full model (.pt) | Quick prototyping | Pickle | Fast |
| ONNX (.onnx) | Cross-framework deployment | Safe | Varies |
| TorchScript (.pt) | Production deployment | Safe | Fast |
| SafeTensors (.safetensors) | HuggingFace ecosystem | Safe | Very fast |

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| state_dict | Preferred way to save/load; architecture-independent |
| Checkpoint | Save model + optimizer + epoch for training resumption |
| map_location | Handle CPU/GPU device differences when loading |
| weights_only=True | Security: prevent pickle-based attacks |
| strict=False | Allow partial loading for transfer learning |
| ONNX export | Cross-framework deployment; use dynamic_axes for variable batch |
| SafeTensors | Safer, faster alternative to pickle-based serialization |

---

**Next**: [GPU Training](./10_GPU_Training.md) -- Moving training to GPU with device management and mixed precision.
